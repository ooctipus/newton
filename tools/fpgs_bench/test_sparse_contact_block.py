# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guarded sticking-block checks; the old eight-sweep trajectory is not an oracle."""

import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from tools.fpgs_bench.test_sparse_factor import fixture, original_eight, physical_rows, unpack

CAPTURES = Path("/tmp/fpgs-g1-sparse-operator-paired16k-20260913-02")
REPLAY = Path("/tmp/fpgs-g1-tree-factor-OzkV0Het/replay_operator_assessment.py")
REPLAY_SHA = "7b82e5bbce00088b131153c05f5437cf0ddadac3709b0259eea4ca70ceb73e90"
PAYLOAD_SHA = {
    (0, 1600): "4841c835c11341b8681ca3d7e24527c9a584bc39afcb75c3758cc391fbad41bc",
    (0, 1601): "cea512719353aa1a1f1f1ee2bd3ee6bf409e7cea60de4507c05cf2b04834b1db",
    (1, 1600): "323fc0f71d749706b63de8758a09bf5e0250112435129235432f0df011dcd740",
    (1, 1601): "be3cbe45678f5a6850d65ba9fe5776b7102a8708b57f7d00bd2ad625200836fd",
}
BLOCK_ENV = {"FEATHER_PGS_SPARSE_CONTACT_BLOCK": "1", "FEATHER_PGS_SPARSE_PACKETS": "0"}


def saved_records():
    """Reuse the existing audited current/held capture owner, with exact pins."""
    if not REPLAY.is_file() or not CAPTURES.is_dir():
        raise unittest.SkipTest("Pinned saved G1 operator artifacts are not installed")
    if hashlib.sha256(REPLAY.read_bytes()).hexdigest() != REPLAY_SHA:
        raise AssertionError("Changed existing replay helper")
    spec = importlib.util.spec_from_file_location("contact_block_saved_replay", REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = []
    for gpu in (0, 1):
        for record, data in module.load_records(CAPTURES, gpu):
            if record["sha256"] != PAYLOAD_SHA[gpu, record["step"]]:
                raise AssertionError("Unexpected saved operator payload")
            result.append((gpu, record, data))
    return module, result


def reference(
    J,
    Y,
    diagonal,
    rhs,
    types,
    parents,
    mu,
    vhat,
    *,
    block=True,
    iterations=8,
    omega=1.0,
    friction_start=0,
    incoming=None,
    templates=None,
):
    """Independent FP64 local solve, preserving denominator-only CFM and scalar fallback."""
    v = np.asarray(vhat, float).copy()
    lam = np.zeros(len(J)) if incoming is None else np.asarray(incoming, float).copy()
    templates = np.zeros(len(J), int) if templates is None else np.asarray(templates)
    gram = J @ Y.T
    stats = {"stick": 0, "open": 0, "slip": 0, "unsafe": 0}
    for iteration in range(iterations):
        row = 0
        while row < len(J):
            valid = (
                block
                and types[row] == 0
                and row + 2 < len(J)
                and iteration >= friction_start
                and np.isfinite(omega)
                and 0 < omega <= 1
                and np.all(types[row + 1 : row + 3] == 2)
                and np.all(parents[row + 1 : row + 3] == row)
                and np.all(templates[row : row + 3] == templates[row])
                and np.isfinite(mu[row + 1])
                and mu[row + 1] >= 0
                and mu[row + 1] == mu[row + 2]
            )
            if valid:
                take = slice(row, row + 3)
                residual = J[take] @ v + rhs[take]
                if not np.any(lam[take]) and residual[0] >= 0:
                    stats["open"] += 1
                    row += 3
                    continue
                local = 0.5 * (gram[take, take] + gram[take, take].T)
                np.fill_diagonal(local, diagonal[take])
                # Principal minors give the LDL pivots independently of native elimination.
                pivots = np.array([local[0, 0], 0.0, 0.0])
                if pivots[0] > 0:
                    pivots[1] = np.linalg.det(local[:2, :2]) / pivots[0]
                    if pivots[1] > 0:
                        pivots[2] = np.linalg.det(local) / (pivots[0] * pivots[1])
                safe = np.isfinite(local).all() and np.all(pivots > 1e-7 * max(diagonal[take]))
                if safe:
                    trial = lam[take] + omega * np.linalg.solve(local, -residual)
                    if (
                        np.isfinite(trial).all()
                        and trial[0] >= 0
                        and np.linalg.norm(trial[1:]) <= mu[row + 1] * trial[0]
                    ):
                        v += Y[take].T @ (trial - lam[take])
                        lam[take] = trial
                        stats["stick"] += 1
                        row += 3
                        continue
                    stats["slip"] += 1
                else:
                    stats["unsafe"] += 1
            if types[row] == 2 and iteration < friction_start:
                lam[row] = 0.0  # Original incoming-lambda/du=0 lifecycle, not a warm kinetic state.
            elif diagonal[row] > 0:
                old = lam[row]
                next_value = old - omega * (J[row] @ v + rhs[row]) / diagonal[row]
                if types[row] in (0, 3):
                    next_value = max(next_value, 0.0)
                elif types[row] == 2:
                    parent = parents[row]
                    radius = max(mu[row] * lam[parent], 0.0)
                    sibling = parent + (2 if row == parent + 1 else 1)
                    if radius <= 0:
                        next_value = 0.0
                    elif np.hypot(next_value, lam[sibling]) > radius:
                        scale = radius / np.hypot(next_value, lam[sibling])
                        next_value *= scale
                        other = lam[sibling] * scale
                        v += Y[sibling] * (other - lam[sibling])
                        lam[sibling] = other
                lam[row] = next_value
                v += Y[row] * (next_value - old)
            row += 1
    return v, lam, stats


def residual_metrics(J, diagonal, rhs, types, parents, mu, vhat, v, lam):
    """Natural projected residual and physical cone feasibility, without CFM*lambda."""
    residual = J @ v + rhs
    correction = np.zeros(len(J))
    cone = 0.0
    for row, kind in enumerate(types):
        if kind in (0, 3):
            correction[row] = lam[row] - max(0.0, lam[row] - residual[row] / diagonal[row])
            cone = max(cone, -lam[row])
        elif row == parents[row] + 1:
            pair = slice(row, row + 2)
            radius = max(0.0, mu[row] * lam[parents[row]])
            trial = lam[pair] - residual[pair] / max(diagonal[pair])
            projected = trial * min(1.0, radius / max(np.linalg.norm(trial), 1e-300))
            correction[pair] = lam[pair] - projected
            cone = max(cone, np.linalg.norm(lam[pair]) - radius)
    scale = 1 + np.max(np.abs(J @ vhat + rhs) / np.sqrt(diagonal), initial=0)
    return float(np.max(np.sqrt(diagonal) * np.abs(correction), initial=0) / scale), float(cone)


def simple_case(kind="stick"):
    z = np.array([[1.0, 0.0, 0.0], [0.3, 1.0, 0.0], [-0.2, 0.15, 0.8]])
    rhs = np.array([-1.0, -0.1, 0.1])
    mu = np.array([0.0, 0.8, 0.8])
    if kind == "open":
        rhs[0] = 1.0
    elif kind == "slip":
        rhs[1:] = [3.0, -4.0]
    elif kind == "singular":
        z[2] = z[1]
    diagonal = np.sum(z * z, axis=1) + (0.2 if kind == "cfm" else 0.0)
    return z, diagonal, rhs, np.array([0, 2, 2]), np.array([-1, 0, 0]), mu


class TestSparseContactBlockCPU(unittest.TestCase):
    def test_local_stick_fallback_and_denominator_only_cfm(self):
        """Check local cone cases and regularization against the independent block law."""
        for kind in ("stick", "open", "slip", "singular", "cfm"):
            z, diagonal, rhs, types, parents, mu = simple_case(kind)
            v, lam, stats = reference(z, z, diagonal, rhs, types, parents, mu, np.zeros(3), iterations=1)
            self.assertTrue(np.isfinite(v).all())
            self.assertLess(residual_metrics(z, diagonal, rhs, types, parents, mu, np.zeros(3), v, lam)[1], 1e-14)
            if kind in ("stick", "cfm"):
                self.assertEqual(stats["stick"], 1)
                np.testing.assert_allclose(
                    lam, np.linalg.solve(z @ z.T + np.diag(diagonal - np.sum(z * z, axis=1)), -rhs)
                )
            elif kind == "open":
                self.assertEqual(stats["open"], 1)
            else:
                old_v, old_lam, _ = reference(
                    z, z, diagonal, rhs, types, parents, mu, np.zeros(3), block=False, iterations=1
                )
                np.testing.assert_array_equal(v, old_v)
                np.testing.assert_array_equal(lam, old_lam)
                self.assertEqual(stats["unsafe" if kind == "singular" else "slip"], 1)
        z, diagonal, rhs, types, parents, mu = simple_case("cfm")
        v, lam, _ = reference(z, z, diagonal, rhs, types, parents, mu, np.zeros(3), iterations=64)
        np.testing.assert_allclose(z @ v + rhs, 0, atol=1e-13)
        self.assertGreater(np.linalg.norm(0.2 * lam), 0.1)

    def test_saved_sixteen_epoch_residual_screen(self):
        """Bound finite-eight residual changes on the pinned current/held sample."""
        _, records = saved_records()
        totals = {"stick": 0, "open": 0, "slip": 0, "unsafe": 0}
        scores = []
        for gpu, record, data in records:
            for index, world in enumerate(data["worlds"]):
                J = data[f"J_world_{world}"].astype(float)
                L = data["L_by_size"][index].astype(float)
                Y = np.linalg.solve(L.T, np.linalg.solve(L, J.T)).T
                diagonal, rhs, types, parents, mu = (
                    data[f"{key}_{world}"] for key in ("diag", "rhs", "row_type", "row_parent", "row_mu")
                )
                vhat = data["v_hat"].reshape(4, 43)[index].astype(float)
                original_v, original_lam = original_eight(J, Y, diagonal, rhs, types, parents, mu, vhat)
                scalar_v, scalar_lam, _ = reference(J, Y, diagonal, rhs, types, parents, mu, vhat, block=False)
                np.testing.assert_allclose(scalar_v, original_v, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(scalar_lam, original_lam, rtol=1e-12, atol=1e-12)
                v, lam, stats = reference(J, Y, diagonal, rhs, types, parents, mu, vhat)
                score, cone = residual_metrics(J, diagonal, rhs, types, parents, mu, vhat, v, lam)
                old_score, _ = residual_metrics(J, diagonal, rhs, types, parents, mu, vhat, original_v, original_lam)
                self.assertTrue(np.isfinite(v).all() and np.isfinite(lam).all())
                self.assertLess(cone, 1e-10)
                np.testing.assert_allclose(v, vhat + Y.T @ lam, rtol=1e-11, atol=1e-11)
                # A selected-snapshot screen, not a population hit rate or promotion gate.
                # RTX 7410 refresh is ~6.7% worse in this metric; normal complementarity improves.
                self.assertLessEqual(score, 1.08 * old_score + 1e-8)
                scores.append((old_score, score))
                for key in totals:
                    totals[key] += stats[key]
                print(
                    f"block_saved gpu={gpu} step={record['step']} world={world} natural={old_score:.9g}->{score:.9g} {stats}"
                )
        self.assertEqual(len(scores), 16)
        self.assertEqual(totals, {"stick": 44, "open": 673, "slip": 75, "unsafe": 0})
        self.assertLessEqual(max(v[1] for v in scores), max(v[0] for v in scores))


def full_rows(owner, count):
    packed = owner.data.Z.numpy()[0, :count]
    templates = owner.data.support.numpy()[0, :count]
    full = np.zeros((count, 43))
    for row, template in enumerate(templates):
        length = owner.host["support_count"][template]
        full[row, owner.host["support_nodes"][template, :length]] = packed[row, :length]
    return full, templates


def check_native(test, f, *, iterations=8, omega=1.0, friction_start=0):
    s, owner = f["solver"], f["owner"]
    test.assertTrue(owner.block_contacts)
    test.assertEqual(owner.kernels.solve.key, "sparse_contact_block43_s18_c100")
    count = int(s.constraint_count.numpy()[0])
    z, templates = full_rows(owner, count)
    diagonal, rhs, types, parents, mu, incoming = (
        getattr(s, name).numpy()[0, :count] for name in ("diag", "rhs", "row_type", "row_parent", "row_mu", "impulses")
    )
    seed = rhs.astype(float) + owner.data.incident.numpy()[0, :count]
    du, expected_lam, _ = reference(
        z,
        z,
        diagonal,
        seed,
        types,
        parents,
        mu,
        np.zeros(43),
        iterations=iterations,
        omega=omega,
        friction_start=friction_start,
        incoming=incoming,
        templates=templates,
    )
    vhat, W = s.v_hat.numpy().copy(), unpack(owner)
    owner.solve(s.rhs, iterations, omega, friction_start)
    owner.check()
    actual, lam = s.v_out.numpy(), s.impulses.numpy()[0, :count]
    np.testing.assert_allclose(lam, expected_lam, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(actual, vhat + (W.T @ du)[::-1], rtol=3e-4, atol=3e-5)
    test.assertTrue(np.isfinite(actual).all() and np.isfinite(lam).all())
    if not np.any(incoming):
        np.testing.assert_allclose(actual, vhat + (W.T @ (z.T @ lam))[::-1], rtol=3e-5, atol=3e-6)
    return actual.copy(), lam.copy()


@unittest.skipUnless(wp.is_cuda_available(), "Native contact-block controls require CUDA")
class TestSparseContactBlockCUDA(unittest.TestCase):
    def test_native_stick_open_slip_and_guarded_fallback(self):
        """Match native impulse transactions to independent stick and fallback cases."""
        with patch.dict(os.environ, BLOCK_ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        template = int(np.flatnonzero(owner.host["support_count"] >= 3)[0])
        owner.data.support.fill_(template)
        owner.data.incident.zero_()
        s.v_hat.zero_()
        for kind in ("stick", "open", "slip", "singular", "cfm", "incoming", "delayed", "overrelaxed"):
            with self.subTest(kind=kind):
                z, diagonal, rhs, types, parents, mu = simple_case(kind)
                packed = np.zeros((1, 100, 18), np.float32)
                packed[0, :3, :3] = z
                owner.data.Z.assign(packed)
                s.constraint_count.assign(np.array([3], np.int32))
                for name, values in (
                    ("diag", diagonal),
                    ("rhs", rhs),
                    ("row_type", types),
                    ("row_parent", parents),
                    ("row_mu", mu),
                ):
                    array = getattr(s, name)
                    data = np.zeros(array.shape, values.dtype)
                    data[0, :3] = values
                    array.assign(data)
                s.impulses.zero_()
                if kind in ("incoming", "delayed"):
                    impulses = s.impulses.numpy()
                    impulses[0, :3] = [0.3, 0.01, -0.01]
                    s.impulses.assign(impulses)
                check_native(
                    self,
                    f,
                    iterations=1,
                    friction_start=int(kind == "delayed"),
                    omega=1.2 if kind == "overrelaxed" else 1.0,
                )

    def test_native_current_held_graph_and_empty(self):
        """Check current rows, held factors, replay cache lifetime and empty output."""
        with patch.dict(os.environ, BLOCK_ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        W = unpack(owner)
        action = (W.T @ W)[::-1, ::-1]
        defect = np.linalg.norm(f["H"] @ action - np.eye(43), np.inf) / (
            1 + np.linalg.norm(f["H"], np.inf) * np.linalg.norm(action, np.inf)
        )
        self.assertLess(defect, 2e-6)
        saved = owner.data.W.numpy().copy()
        for held in (False, True):
            if held:
                s.mass_update_mask.zero_()
                s.body_I_c.fill_(wp.spatial_matrix(np.nan))
                owner.refresh(s)
                np.testing.assert_array_equal(owner.data.W.numpy(), saved)
                points = f["contacts"].rigid_contact_point0.numpy()
                points[:3, 0] += 0.007
                f["contacts"].rigid_contact_point0.assign(points)
            s.v_hat.assign(np.random.default_rng(12 + held).normal(0, 0.02, 43).astype(np.float32))
            owner.build_rows(f["state"], s, f["contacts"], 0.0025)
            s.check_constraint_capacity()
            count = int(s.constraint_count.numpy()[0])
            z, _ = full_rows(owner, count)
            J = physical_rows(f)
            np.testing.assert_allclose(z, (W @ J[:, ::-1].T).T, rtol=3e-5, atol=3e-6)
            s._stage4_compute_rhs_world(0.0025)
            owner.restitution(0.0025)
            s.impulses.zero_()
            expected, expected_lam = check_native(self, f)
            with wp.ScopedCapture(device="cuda:0") as capture:
                s.impulses.zero_()
                owner.solve(s.rhs, 8, 1.0, 0)
            for _ in range(2):
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(s.v_out.numpy(), expected)
                np.testing.assert_array_equal(s.impulses.numpy()[0, :count], expected_lam)
        s.constraint_count.zero_()
        s.v_out.fill_(np.nan)
        owner.solve(s.rhs, 8, 1.0, 0)
        np.testing.assert_array_equal(s.v_out.numpy(), s.v_hat.numpy())

    def test_native_saved_sixteen_current_held_epochs(self):
        """Check native block numerics on all sixteen pinned world/epoch payloads."""
        replay, records = saved_records()
        for gpu, record, data in records:
            for index, world in enumerate(data["worlds"]):
                with self.subTest(gpu=gpu, step=record["step"], world=int(world)), patch.dict(os.environ, BLOCK_ENV):
                    f = replay.bind_world(record, data, index, "cuda:0")
                    s, owner = f["solver"], f["owner"]
                    lower = np.linalg.cholesky(f["H"][::-1, ::-1])
                    W = np.linalg.solve(lower, np.eye(43))
                    owner.data.W.assign(W[owner.host["row"], owner.host["col"]][None].astype(np.float32))
                    owner.data.valid.fill_(1)
                    owner.build_rows(f["state"], s, f["contacts"], 0.0025)
                    s.check_constraint_capacity()
                    s._stage4_compute_rhs_world(0.0025)
                    owner.restitution(0.0025)
                    s.impulses.zero_()
                    count = int(s.constraint_count.numpy()[0])
                    J = physical_rows(f)
                    z, _ = full_rows(owner, count)
                    # Warm geometry has known FP32 coefficient cancellation.
                    # Report the unchanged diagnostic, then assess the solve and
                    # physical response instead of stopping at an intermediate.
                    coefficient = replay.component_diagnostic(z, (unpack(owner) @ J[:, ::-1].T).T)
                    actual, lam = check_native(self, f)
                    diagonal, rhs, types, parents, mu = (
                        getattr(s, name).numpy()[0, :count]
                        for name in ("diag", "rhs", "row_type", "row_parent", "row_mu")
                    )
                    score, cone = residual_metrics(J, diagonal, rhs, types, parents, mu, s.v_hat.numpy(), actual, lam)
                    delta = actual.astype(float) - s.v_hat.numpy().astype(float)
                    force = J.T @ lam.astype(float)
                    momentum = float(
                        np.linalg.norm(f["H"] @ delta - force, np.inf)
                        / (
                            1
                            + np.linalg.norm(f["H"], np.inf) * np.linalg.norm(delta, np.inf)
                            + np.linalg.norm(force, np.inf)
                        )
                    )
                    print(
                        "contact_block_saved_native "
                        + json.dumps(
                            {
                                "gpu_fixture": gpu,
                                "step": record["step"],
                                "world": int(world),
                                "coefficient": coefficient,
                                "natural_residual": score,
                                "cone_error": cone,
                                "momentum_defect": momentum,
                            }
                        ),
                        flush=True,
                    )
                    self.assertTrue(np.isfinite(score))
                    self.assertLess(cone, 3e-5)
                    self.assertLess(momentum, 2e-6)


if __name__ == "__main__":
    unittest.main()

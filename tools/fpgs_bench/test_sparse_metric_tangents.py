# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check fixed-normal metric-disk tangents without assuming old GS convergence."""

import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from tools.fpgs_bench.test_sparse_contact_block import (
    full_rows,
    residual_metrics,
    saved_records,
    simple_case,
)
from tools.fpgs_bench.test_sparse_contact_block import (
    reference as scalar_reference,
)
from tools.fpgs_bench.test_sparse_factor import fixture, physical_rows, unpack

METRIC_ENV = {
    "FEATHER_PGS_SPARSE_METRIC_TANGENTS": "1",
    "FEATHER_PGS_SPARSE_PACKETS": "0",
    "FEATHER_PGS_SPARSE_CONTACT_BLOCK": "0",
}


def metric_disk(matrix, linear, radius):
    """Solve the FP64 disk QP by eigenvalue bracketing, independently of native Newton."""
    matrix, linear = np.asarray(matrix, float), np.asarray(linear, float)
    values, vectors = np.linalg.eigh(matrix)
    if not np.isfinite(values).all() or values[0] <= 0 or radius < 0:
        raise ValueError("Require a finite positive-definite disk QP")
    if radius == 0:
        return np.zeros(2), 0.0
    transformed = vectors.T @ linear
    result = -vectors @ (transformed / values)
    if np.linalg.norm(result) <= radius:
        return result, 0.0
    lower, upper = 0.0, np.linalg.norm(linear) / radius
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if np.linalg.norm(transformed / (values + middle)) > radius:
            lower = middle
        else:
            upper = middle
    return -vectors @ (transformed / (values + upper)), upper


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
    iterations=8,
    omega=1.0,
    friction_start=0,
    incoming=None,
    templates=None,
    metric=True,
):
    """Apply scalar normals and independent metric tangents with original scalar fallback."""
    J, Y = np.asarray(J, float), np.asarray(Y, float)
    diagonal, rhs = np.asarray(diagonal, float), np.asarray(rhs, float)
    v = np.asarray(vhat, float).copy()
    lam = np.zeros(len(J)) if incoming is None else np.asarray(incoming, float).copy()
    templates = np.zeros(len(J), int) if templates is None else np.asarray(templates)
    gram = J @ Y.T
    stats = {"open": 0, "stick": 0, "slip": 0, "unsafe": 0}
    for iteration in range(iterations):
        row = 0
        changed = False
        while row < len(J):
            valid = (
                metric
                and types[row] == 0
                and row + 2 < len(J)
                and iteration >= friction_start
                and omega == 1.0
                and np.all(types[row + 1 : row + 3] == 2)
                and np.all(parents[row + 1 : row + 3] == row)
                and np.all(templates[row : row + 3] == templates[row])
                and np.isfinite(mu[row + 1])
                and mu[row + 1] >= 0
                and mu[row + 1] == mu[row + 2]
            )
            if valid:
                take, pair = slice(row, row + 3), slice(row + 1, row + 3)
                old = lam[take].copy()
                normal_residual = J[row] @ v + rhs[row]
                normal_safe = (
                    np.isfinite(normal_residual)
                    and np.isfinite(diagonal[row])
                    and diagonal[row] > 0
                    and np.isfinite(old).all()
                )
                if normal_safe:
                    normal = max(old[0] - normal_residual / diagonal[row], 0.0)
                    radius = mu[row + 1] * normal
                    trial = np.array([normal, 0.0, 0.0])
                    safe = np.isfinite(trial).all() and np.isfinite(radius)
                    kind = "open"
                    if safe and radius > 0:
                        local = 0.5 * (gram[pair, pair] + gram[pair, pair].T)
                        np.fill_diagonal(local, diagonal[pair])
                        residual = J[pair] @ v + rhs[pair] + gram[pair, row] * (normal - old[0])
                        linear = residual - local @ old[1:]
                        safe = np.isfinite(local).all() and np.isfinite(linear).all()
                        if safe:
                            safe = np.linalg.eigvalsh(local)[0] > 1e-7 * max(diagonal[pair])
                        if safe:
                            trial[1:], alpha = metric_disk(local, linear, radius)
                            kind = "slip" if alpha > 0 else "stick"
                    if safe and np.isfinite(J[take]).all():
                        delta = trial - old
                        v += Y[take].T @ delta
                        lam[take] = trial
                        changed |= bool(np.any(delta))
                        stats[kind] += 1
                        row += 3
                        continue
                stats["unsafe"] += 1
            if types[row] == 2 and iteration < friction_start:
                # Match the existing incoming-lambda/du=0 lifecycle: this is
                # not an initially applied warm impulse that must be removed.
                lam[row] = 0.0
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
                        changed |= other != lam[sibling]
                        lam[sibling] = other
                lam[row] = next_value
                v += Y[row] * (next_value - old)
                changed |= next_value != old
            row += 1
        if iteration >= friction_start and not changed:
            break
    return v, lam, stats


def physical_metrics(J, diagonal, rhs, types, parents, mu, vhat, v, lam):
    """Report physical normal, complementarity and dissipation defects without CFM*lambda."""
    natural, cone = residual_metrics(J, diagonal, rhs, types, parents, mu, vhat, v, lam)
    residual = J @ v + rhs
    normal = np.flatnonzero(np.isin(types, (0, 3)))
    normal_error = float(np.max(np.maximum(-residual[normal], 0), initial=0))
    complementarity = float(np.max(np.abs(lam[normal] * residual[normal]), initial=0))
    mdp = 0.0
    for row in np.flatnonzero(types == 2):
        if row == parents[row] + 1:
            pair = slice(row, row + 2)
            radius = max(0.0, mu[row] * lam[parents[row]])
            mdp = max(mdp, abs(lam[pair] @ residual[pair] + radius * np.linalg.norm(residual[pair])))
    return {"natural": natural, "cone": cone, "normal": normal_error, "complementarity": complementarity, "mdp": mdp}


def check_native(test, f, *, iterations=8, omega=1.0, friction_start=0, metric=True):
    """Compare the native law on current rounded rows and verify its impulse publication."""
    s, owner = f["solver"], f["owner"]
    test.assertTrue(owner.metric_tangents)
    expected_key = (
        "sparse_metric_expiry43_s18_c100"
        if getattr(owner, "zero_expiry", False)
        else "sparse_metric_tangent43_s18_c100"
    )
    test.assertEqual(owner.kernels.solve.key, expected_key)
    count = int(s.constraint_count.numpy()[0])
    z, templates = full_rows(owner, count)
    diagonal, rhs, types, parents, mu, incoming = (
        getattr(s, name).numpy()[0, :count] for name in ("diag", "rhs", "row_type", "row_parent", "row_mu", "impulses")
    )
    seed = rhs.astype(float) + owner.data.incident.numpy()[0, :count]
    du, expected_lam, stats = reference(
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
        metric=metric,
    )
    vhat, W = s.v_hat.numpy().copy(), unpack(owner)
    owner.solve(s.rhs, iterations, omega, friction_start)
    owner.check()
    actual, lam = s.v_out.numpy(), s.impulses.numpy()[0, :count]
    np.testing.assert_allclose(lam, expected_lam, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(actual, vhat + (W.T @ du)[::-1], rtol=3e-4, atol=3e-5)
    test.assertTrue(np.isfinite(actual).all() and np.isfinite(lam).all())
    if friction_start == 0 or not np.any(incoming):
        np.testing.assert_allclose(actual, vhat + (W.T @ (z.T @ (lam - incoming)))[::-1], rtol=3e-5, atol=3e-6)
    return actual.copy(), lam.copy(), stats


def check_reference_law(test):
    """Distinguish the metric disk law, scalar normal and proximal-only CFM on CPU."""
    matrix = np.array([[4.0, 0.8], [0.8, 1.0]])
    linear, radius = np.array([3.0, -4.0]), 0.7
    trial, alpha = metric_disk(matrix, linear, radius)
    np.testing.assert_allclose((matrix + alpha * np.eye(2)) @ trial + linear, 0, atol=2e-14)
    test.assertAlmostEqual(np.linalg.norm(trial), radius, places=14)
    unconstrained = -np.linalg.solve(matrix, linear)
    radial = unconstrained * radius / np.linalg.norm(unconstrained)

    def objective(value):
        return 0.5 * value @ matrix @ value + linear @ value

    test.assertLess(objective(trial), objective(radial) - 1e-3)
    np.testing.assert_array_equal(metric_disk(matrix, linear, 0.0)[0], np.zeros(2))
    for kind in ("stick", "open", "slip", "singular", "cfm"):
        with test.subTest(reference=kind):
            z, diagonal, rhs, types, parents, mu = simple_case(kind)
            v, lam, stats = reference(z, z, diagonal, rhs, types, parents, mu, np.zeros(3), iterations=1)
            test.assertTrue(np.isfinite(v).all())
            test.assertLess(residual_metrics(z, diagonal, rhs, types, parents, mu, np.zeros(3), v, lam)[1], 1e-14)
            np.testing.assert_allclose(v, z.T @ lam, atol=1e-14)
            if kind == "singular":
                old_v, old_lam, _ = scalar_reference(
                    z, z, diagonal, rhs, types, parents, mu, np.zeros(3), block=False, iterations=1
                )
                np.testing.assert_array_equal(v, old_v)
                np.testing.assert_array_equal(lam, old_lam)
                test.assertEqual(stats["unsafe"], 1)
            elif kind == "open":
                test.assertEqual(stats["open"], 1)
                np.testing.assert_array_equal(lam, np.zeros(3))
            else:
                test.assertAlmostEqual(lam[0], max(-rhs[0] / diagonal[0], 0))
                local = z[1:] @ z[1:].T
                np.fill_diagonal(local, diagonal[1:])
                linear = rhs[1:] + (z[1:] @ z[0]) * lam[0]
                expected, alpha = metric_disk(local, linear, mu[1] * lam[0])
                np.testing.assert_allclose(lam[1:], expected, atol=1e-14)
                physical = z[1:] @ v + rhs[1:]
                proximal = (diagonal[1:] - np.sum(z[1:] ** 2, axis=1)) * lam[1:]
                np.testing.assert_allclose(physical + proximal + alpha * lam[1:], 0, atol=2e-14)
                test.assertEqual(stats["slip" if kind == "slip" else "stick"], 1)
    z, diagonal, rhs, types, parents, mu = simple_case("cfm")
    v, lam, _ = reference(z, z, diagonal, rhs, types, parents, mu, np.zeros(3), iterations=128)
    np.testing.assert_allclose(z @ v + rhs, 0, atol=1e-12)
    test.assertGreater(np.linalg.norm(0.2 * lam), 0.1)
    for friction_start, omega in ((1, 1.0), (0, 1.2)):
        incoming = np.array([0.3, 0.01, -0.01])
        args = (z, z, diagonal, rhs, types, parents, mu, np.zeros(3))
        controls = {"iterations": 1, "incoming": incoming, "friction_start": friction_start, "omega": omega}
        v, lam, _ = reference(*args, **controls)
        old_v, old_lam, _ = scalar_reference(*args, block=False, **controls)
        np.testing.assert_array_equal(v, old_v)
        np.testing.assert_array_equal(lam, old_lam)


class TestSparseMetricTangentsCPU(unittest.TestCase):
    def test_owner_admission(self):
        """Require explicit metric ownership and reject conflicting modes."""
        with patch.dict(os.environ, METRIC_ENV):
            owner = fixture("cpu")["owner"]
        self.assertTrue(owner.metric_tangents)
        self.assertFalse(owner.block_contacts)
        self.assertEqual(owner.kernels.solve.key, "sparse_metric_tangent43_s18_c100")
        with patch.dict(os.environ, {**METRIC_ENV, "FEATHER_PGS_SPARSE_METRIC_TANGENTS": "0"}):
            ordinary = fixture("cpu")["owner"]
        self.assertFalse(ordinary.metric_tangents)
        for flag in ("FEATHER_PGS_SPARSE_PACKETS", "FEATHER_PGS_SPARSE_CONTACT_BLOCK"):
            with self.subTest(flag=flag), patch.dict(os.environ, {**METRIC_ENV, flag: "1"}):
                with self.assertRaises(ValueError):
                    fixture("cpu")
        with patch.dict(os.environ, METRIC_ENV):
            other_capacity = fixture("cpu", capacity=101)["owner"]
        self.assertFalse(other_capacity.metric_tangents)
        self.assertEqual(other_capacity.kernels.solve.key, "sparse_factor_gs43_s18_c101")
        check_reference_law(self)


@unittest.skipUnless(wp.is_cuda_available(), "Native metric-tangent controls require CUDA")
class TestSparseMetricTangentsCUDA(unittest.TestCase):
    def test_native_stick_open_slip_and_guarded_fallback(self):
        """Check fused scalar-normal/QP transactions, zero disks and pre-mutation fallback."""
        with patch.dict(os.environ, METRIC_ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        templates = np.flatnonzero(owner.host["support_count"] >= 3)
        template = int(templates[0])
        owner.data.incident.zero_()
        s.v_hat.zero_()
        kinds = (
            "stick",
            "open",
            "slip",
            "singular",
            "cfm",
            "incoming",
            "delayed",
            "overrelaxed",
            "zero_mu",
            "closing_old_tangents",
            "mismatched_support",
            "mismatched_mu",
            "root_overflow",
            "limit",
        )
        for kind in kinds:
            with self.subTest(kind=kind):
                z, diagonal, rhs, types, parents, mu = simple_case(kind)
                incoming = np.zeros(3)
                if kind in ("incoming", "delayed", "zero_mu", "closing_old_tangents"):
                    incoming[:] = [0.3, 0.01, -0.01]
                if kind == "zero_mu":
                    mu[1:] = 0
                elif kind == "closing_old_tangents":
                    rhs[0] = 1
                elif kind == "mismatched_mu":
                    mu[2] *= 0.5
                elif kind == "root_overflow":
                    # Keep the radius normal-range (not flushed to zero), but
                    # overflow beta_norm/radius so the native root must reject.
                    mu[1:] = 1e-37
                    rhs[1:] = [1e3, -1e3]
                elif kind == "limit":
                    types[:] = 3
                packed = np.zeros((1, 100, 18), np.float32)
                packed[0, :3, :3] = z
                owner.data.Z.assign(packed)
                owner.data.support.fill_(template)
                if kind == "mismatched_support":
                    support = owner.data.support.numpy()
                    support[0, 2] = int(templates[-1])
                    self.assertNotEqual(support[0, 2], template)
                    owner.data.support.assign(support)
                s.constraint_count.assign(np.array([3], np.int32))
                for name, values in (
                    ("diag", diagonal),
                    ("rhs", rhs),
                    ("row_type", types),
                    ("row_parent", parents),
                    ("row_mu", mu),
                    ("impulses", incoming),
                ):
                    array = getattr(s, name)
                    data = np.zeros(array.shape, dtype=array.numpy().dtype)
                    data[0, :3] = values
                    array.assign(data)
                _, _, stats = check_native(
                    self,
                    f,
                    iterations=1,
                    friction_start=int(kind == "delayed"),
                    omega=1.2 if kind == "overrelaxed" else 1.0,
                    metric=kind != "root_overflow",
                )
                if kind in ("stick", "cfm", "slip"):
                    self.assertEqual(stats["slip" if kind == "slip" else "stick"], 1)
                elif kind in ("open", "zero_mu", "closing_old_tangents"):
                    self.assertEqual(stats["open"], 1)
                elif kind == "singular":
                    self.assertEqual(stats["unsafe"], 1)
        s.impulses.zero_()
        s.v_out.fill_(np.nan)
        check_native(self, f, iterations=0)

    def test_native_current_held_graph_and_empty(self):
        """Check held-factor rows and reinitialize lazy cross terms on graph replay and empty transitions."""
        with patch.dict(os.environ, METRIC_ENV):
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
        graph = None
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
            self.assertTrue(np.isfinite(z).all() and np.isfinite(J).all())
            s._stage4_compute_rhs_world(0.0025)
            owner.restitution(0.0025)
            s.impulses.zero_()
            expected, expected_lam, _ = check_native(self, f)
            if graph is None:
                with wp.ScopedCapture(device="cuda:0") as capture:
                    s.impulses.zero_()
                    owner.solve(s.rhs, 8, 1.0, 0)
                graph = capture.graph
            # Reuse the SAME captured solve with changed current rows/incident
            # and held W; per-launch shared cross coefficients cannot survive.
            for _ in range(2):
                wp.capture_launch(graph)
                np.testing.assert_array_equal(s.v_out.numpy(), expected)
                np.testing.assert_array_equal(s.impulses.numpy()[0, :count], expected_lam)
            s.constraint_count.zero_()
            s.v_out.fill_(np.nan)
            wp.capture_launch(graph)
            np.testing.assert_array_equal(s.v_out.numpy(), s.v_hat.numpy())
            s.constraint_count.assign(np.array([count], np.int32))
            wp.capture_launch(graph)
            np.testing.assert_array_equal(s.v_out.numpy(), expected)
            np.testing.assert_array_equal(s.impulses.numpy()[0, :count], expected_lam)

    def test_native_saved_sixteen_current_held_epochs(self):
        """Check the metric law and physical response on sixteen pinned current/held payloads."""
        replay, records = saved_records()
        tested = 0
        for gpu, record, data in records:
            for index, world in enumerate(data["worlds"]):
                with self.subTest(gpu=gpu, step=record["step"], world=int(world)), patch.dict(os.environ, METRIC_ENV):
                    f = replay.bind_world(record, data, index, "cuda:0")
                    s, owner = f["solver"], f["owner"]
                    # The audited replay reconstructs held H from valid L;
                    # saved H_by_size is unused zero storage, not the operator.
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
                    z, templates = full_rows(owner, count)
                    coefficient = replay.component_diagnostic(z, (unpack(owner) @ J[:, ::-1].T).T)
                    diagonal, rhs, types, parents, mu = (
                        getattr(s, name).numpy()[0, :count]
                        for name in ("diag", "rhs", "row_type", "row_parent", "row_mu")
                    )
                    # The current atomic row order and FP32 Z/incident are the
                    # same-input oracle. Original-capture order is diagnostic.
                    seed = rhs.astype(float) + owner.data.incident.numpy()[0, :count]
                    scalar_du, scalar_lam, _ = scalar_reference(
                        z, z, diagonal, seed, types, parents, mu, np.zeros(43), block=False, templates=templates
                    )
                    scalar_v = s.v_hat.numpy() + (unpack(owner).T @ scalar_du)[::-1]
                    scalar = physical_metrics(
                        J, diagonal, rhs, types, parents, mu, s.v_hat.numpy(), scalar_v, scalar_lam
                    )
                    permutation = replay.row_permutation(f, count)
                    rhs_diagnostic = replay.component_diagnostic(rhs, data[f"rhs_{world}"][permutation])
                    actual, lam, stats = check_native(self, f)
                    metrics = physical_metrics(J, diagonal, rhs, types, parents, mu, s.v_hat.numpy(), actual, lam)
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
                        "metric_tangent_saved_native "
                        + json.dumps(
                            {
                                "gpu_fixture": gpu,
                                "step": record["step"],
                                "world": int(world),
                                "coefficient": coefficient,
                                "current_to_captured_row": permutation.tolist(),
                                "rhs_vs_captured": rhs_diagnostic,
                                "scalar_current": scalar,
                                "metric_current": metrics,
                                "momentum_defect": momentum,
                                "oracle_transactions": stats,
                            }
                        ),
                        flush=True,
                    )
                    self.assertTrue(np.isfinite(list(metrics.values())).all())
                    self.assertLess(metrics["cone"], 3e-5)
                    self.assertLess(momentum, 2e-6)
                    tested += 1
        self.assertEqual(tested, 16)


if __name__ == "__main__":
    unittest.main()

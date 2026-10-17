# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guarded sticking-block checks; the old eight-sweep trajectory is not an oracle."""

import hashlib
import importlib.util
import unittest
from pathlib import Path

import numpy as np

CAPTURES = Path("/tmp/fpgs-g1-sparse-operator-paired16k-20260913-02")
REPLAY = Path("/tmp/fpgs-g1-tree-factor-OzkV0Het/replay_operator_assessment.py")
REPLAY_SHA = "7b82e5bbce00088b131153c05f5437cf0ddadac3709b0259eea4ca70ceb73e90"
PAYLOAD_SHA = {
    (0, 1600): "4841c835c11341b8681ca3d7e24527c9a584bc39afcb75c3758cc391fbad41bc",
    (0, 1601): "cea512719353aa1a1f1f1ee2bd3ee6bf409e7cea60de4507c05cf2b04834b1db",
    (1, 1600): "323fc0f71d749706b63de8758a09bf5e0250112435129235432f0df011dcd740",
    (1, 1601): "be3cbe45678f5a6850d65ba9fe5776b7102a8708b57f7d00bd2ad625200836fd",
}


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


def original_eight(J, Y, diag, rhs, types, parent, mu, vhat):
    """Independent physical-velocity recurrence, original current friction law."""
    v = np.array(vhat, float)
    lam = np.zeros(len(J), float)
    for _ in range(8):
        for row in range(len(J)):
            old = lam[row]
            new = old - (J[row] @ v + rhs[row]) / diag[row]
            if types[row] in (0, 3):
                new = max(new, 0.0)
            elif types[row] == 2:
                p = parent[row]
                radius = max(mu[row] * lam[p], 0.0)
                if radius <= 0:
                    new = 0.0
                else:
                    sib = p + 2 if row == p + 1 else p + 1
                    mag = np.hypot(new, lam[sib])
                    if mag > radius:
                        scale = radius / mag
                        new *= scale
                        other = lam[sib] * scale
                        v += Y[sib] * (other - lam[sib])
                        lam[sib] = other
            lam[row] = new
            v += Y[row] * (new - old)
    return v, lam


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


if __name__ == "__main__":
    unittest.main()

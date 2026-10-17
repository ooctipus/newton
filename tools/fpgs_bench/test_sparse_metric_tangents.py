# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check fixed-normal metric-disk tangents without assuming old GS convergence."""

import unittest

import numpy as np

from tools.fpgs_bench.test_sparse_contact_block import (
    reference as scalar_reference,
)
from tools.fpgs_bench.test_sparse_contact_block import (
    residual_metrics,
    simple_case,
)


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
    def test_reference_law(self):
        """Distinguish the metric disk law, scalar normal and proximal-only CFM on CPU."""
        check_reference_law(self)


if __name__ == "__main__":
    unittest.main()

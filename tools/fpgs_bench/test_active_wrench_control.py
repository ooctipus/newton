# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for the bounded physical working-set hypothesis."""

import unittest

import numpy as np

from tools.fpgs_bench import active_wrench_control as control
from tools.fpgs_bench.test_sparse_contact_block import simple_case


class TestActiveWrenchControl(unittest.TestCase):
    def test_rank_basic_solution_keeps_dependent_variables(self):
        """Solve only independent columns and reject inconsistent equations."""
        matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
        value, info = control.rank_basic_solve(matrix, np.array([3.0, 6.0]))
        self.assertEqual(info["rank"], 1)
        self.assertTrue(info["consistent"])
        self.assertEqual(np.count_nonzero(value), 1)
        np.testing.assert_allclose(matrix @ value, [3.0, 6.0], atol=1e-12)
        _, info = control.rank_basic_solve(matrix, np.array([3.0, 7.0]))
        self.assertFalse(info["consistent"])

    def test_polar_disk_equations_and_derivative(self):
        """Keep sliding forces on their disk and differentiate physical curvature."""
        gram = np.array([[2.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.7]])
        blocks = [(0, "slide", 0.6)]
        x = np.array([0.8, 0.7])
        impulse, derivative = control.decode(x, blocks, 3)
        self.assertAlmostEqual(np.linalg.norm(impulse[1:]), 0.6 * impulse[0])
        offset = np.array([-0.4, 0.9, 0.2])
        residual = offset + gram @ impulse
        value, jacobian = control.equations(x, blocks, residual, gram @ derivative)
        numerical = np.empty_like(jacobian)
        for col in range(len(x)):
            step = np.zeros(len(x))
            step[col] = 1e-6
            plus, dp = control.decode(x + step, blocks, 3)
            minus, dm = control.decode(x - step, blocks, 3)
            fp, _ = control.equations(x + step, blocks, offset + gram @ plus, gram @ dp)
            fm, _ = control.equations(x - step, blocks, offset + gram @ minus, gram @ dm)
            numerical[:, col] = (fp - fm) / 2e-6
        self.assertEqual(value.shape, (2,))
        np.testing.assert_allclose(jacobian, numerical, rtol=2e-8, atol=2e-9)

    def test_original_disk_cfm_and_fallback_budget(self):
        """Preserve impulse feasibility, momentum and at most eight committed slots."""
        for kind in ("open", "stick", "slip", "singular", "cfm"):
            with self.subTest(kind=kind):
                z, diagonal, rhs, types, parents, mu = simple_case(kind)
                result = control.solve(z, np.eye(3), diagonal, rhs, types, parents, mu, np.zeros(3))
                self.assertLessEqual(result.work["committed"] + result.work["fallback_sweeps"], 8)
                np.testing.assert_allclose(result.velocity, z.T @ result.impulses, rtol=2e-12, atol=2e-12)
                self.assertGreaterEqual(result.impulses[0], -1e-12)
                self.assertLessEqual(np.linalg.norm(result.impulses[1:]), mu[1] * result.impulses[0] + 1e-12)
        mu[:] = 0.0
        frictionless = control.solve(z, np.eye(3), diagonal, rhs, types, parents, mu, np.zeros(3))
        np.testing.assert_array_equal(frictionless.impulses[1:], 0.0)

    def test_counted_reference_is_unchanged(self):
        """Line counters must leave the actual original recurrence byte-identical."""
        z, diagonal, rhs, types, parents, mu = simple_case("slip")
        args = (z, z, diagonal, rhs, types, parents, mu, np.zeros(3))
        plain = control.reference(*args, iterations=3)
        measured, work = control.counted_reference(*args, iterations=3)
        np.testing.assert_array_equal(plain[0], measured[0])
        np.testing.assert_array_equal(plain[1], measured[1])
        self.assertEqual(plain[2], measured[2])
        self.assertEqual(work["sweeps"], 3)
        self.assertEqual(work["row_transactions"], 3)
        self.assertEqual(work["residual_row_dots"], 9)
        self.assertEqual(work["sliding_roots"], 3)
        self.assertEqual(work["root_probes"], 240)


if __name__ == "__main__":
    unittest.main()

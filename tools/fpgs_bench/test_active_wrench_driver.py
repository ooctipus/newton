# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small physical controls distinguishing driver pivots from all-closing Newton."""

import unittest

import numpy as np

from tools.fpgs_bench import active_wrench_driver as driver


class TestActiveWrenchDriver(unittest.TestCase):
    def check_normals(self, J, rhs, expected, *, release=False):
        """Check physical complementarity without any original-sweep fallback."""
        J, rhs = np.asarray(J, float), np.asarray(rhs, float)
        n, dofs = J.shape
        result = driver.solve(
            J, np.eye(dofs), np.sum(J * J, axis=1), rhs, np.zeros(n, int), np.full(n, -1), np.zeros(n), np.zeros(dofs)
        )
        self.assertEqual(result.work["fallback_sweeps"], 0, result.work)
        self.assertLessEqual(result.work["committed"], 8)
        np.testing.assert_allclose(result.impulses, expected, atol=1e-10)
        residual = J @ result.velocity + rhs
        self.assertGreaterEqual(float(np.min(residual)), -1e-10)
        np.testing.assert_allclose(result.impulses * residual, 0, atol=1e-10)
        np.testing.assert_allclose(result.velocity, J.T @ result.impulses, atol=1e-10)
        if release:
            self.assertGreater(result.work["membership_pivots"], 0)
            self.assertTrue(any(abs(t.get("incoming_response_rate", 1)) < 1e-12 for t in result.work["trace"]))

    def test_incompatible_closing_equalities(self):
        """Dependent closing normals need only the stronger contact to carry force."""
        self.check_normals([[1], [1]], [-1, -2], [0, 2])

    def test_negative_all_closing_impulse(self):
        """A closing cold normal may legitimately finish inactive and separating."""
        self.check_normals([[1, 0], [1, np.sqrt(3)]], [-1, -0.5], [1, 0])

    def test_unprocessed_violation_and_zero_wrench_release(self):
        """Do not protect unprocessed rows or discard a real null-wrench release."""
        self.check_normals(np.eye(2), [-2, -1], [2, 1])
        self.check_normals([[1, 0], [0.8, 0.6], [0.8, -0.6]], [-1, -0.9, -0.9], [0, 0.703125, 0.703125], release=True)

    def test_small_cold_residual_still_checks_weighted_defect(self):
        """A small raw residual is not proof that an unpublished correction is zero."""
        result = driver.solve(
            np.array([[1e-10]]),
            np.eye(1),
            np.array([1e-20]),
            np.array([-5e-11]),
            np.array([3]),
            np.array([-1]),
            np.zeros(1),
            np.zeros(1),
        )
        self.assertEqual(result.work["reason"], "unresolved_small_cold_residual")
        self.assertGreater(result.work["fallback_sweeps"], 0)
        self.assertEqual(result.work["materialized_rows"], 1)


if __name__ == "__main__":
    unittest.main()

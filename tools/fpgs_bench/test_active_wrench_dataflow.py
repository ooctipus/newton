# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for retiring the all-row diagonal producer contract."""

import unittest

import numpy as np

from tools.fpgs_bench import active_wrench_driver as driver


class TestActiveWrenchDataflow(unittest.TestCase):
    def test_unused_open_diagonal_is_not_a_solver_input(self):
        """Finish without a diagonal that the lazy producer never publishes."""
        result = driver.solve(
            np.eye(2),
            np.eye(2),
            np.array([1.0, np.nan]),
            np.array([-1.0, 1.0]),
            np.full(2, 3),
            np.full(2, -1),
            np.zeros(2),
            np.zeros(2),
        )
        np.testing.assert_array_equal(result.impulses, [1.0, 0.0])
        np.testing.assert_array_equal(result.velocity, [1.0, 0.0])
        self.assertEqual(result.work["materialized_rows"], 1)
        self.assertEqual(result.work["committed"], 1, result.work)
        self.assertEqual(result.work["fallback_sweeps"], 0)
        self.assertEqual(result.work["reason"], "physical_diagnostic_stop")

    def test_over_budget_closing_set_uses_original_cold_state(self):
        """Reject an estimated workload above eight before any driver correction."""
        result = driver.solve(
            np.eye(9),
            np.eye(9),
            np.ones(9),
            -np.ones(9),
            np.full(9, 3),
            np.full(9, -1),
            np.zeros(9),
            np.zeros(9),
        )
        np.testing.assert_array_equal(result.impulses, np.ones(9))
        self.assertEqual(result.work["committed"], 0, result.work)
        self.assertGreater(result.work["fallback_sweeps"], 0)
        self.assertLessEqual(result.work["fallback_sweeps"], 8)


if __name__ == "__main__":
    unittest.main()

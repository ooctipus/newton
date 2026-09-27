# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify full relaxation recovery with the unchanged per-pass work allowance."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import coulomb_block_jacobi_recover as recover


class TestJacobiRecover(unittest.TestCase):
    def test_full_step_returns_after_a_midpoint(self):
        """Use a full next proposal after one merit-increasing pass is halved."""
        energies = iter((10.0, 20.0, 12.0, 11.0))

        def score(*_args):
            return {
                "natural": next(energies) * 1e-5,
                "normal": 0.0,
                "complementarity": 0.0,
                "mdp": 0.0,
                "cone": 0.0,
                "stop": False,
            }

        with patch.object(recover.frozen.block, "physical_from_residual", side_effect=score):
            result = recover.solve(
                np.ones((1, 1)),
                np.eye(1),
                np.ones(1),
                -np.ones(1),
                np.array([3]),
                np.array([-1]),
                np.zeros(1),
                np.zeros(1),
                iterations=2,
            )
        np.testing.assert_array_equal(result.impulses, [1.0])
        np.testing.assert_array_equal(result.velocity, [1.0])
        self.assertEqual(result.work["sweeps"], 2)
        self.assertEqual(result.work["operator_products"], 4)
        self.assertEqual(result.work["midpoint_row_projections"], 1)
        self.assertEqual(result.work["midpoint_steps"], [1])
        self.assertEqual([c["relaxation"] for c in result.work["curve"]], [0.5, 1.0])
        self.assertNotIn("latch_step", result.work)


if __name__ == "__main__":
    unittest.main()

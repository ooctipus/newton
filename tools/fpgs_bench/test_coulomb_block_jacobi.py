# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Same-state proposals, one-way relaxation and local fallback controls."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import coulomb_block_jacobi as jacobi
from tools.fpgs_bench.test_coulomb_block_dataflow import TestCoulombBlockDataflow


class TestCoulombBlockJacobi(unittest.TestCase):
    def test_all_proposals_read_the_same_residual(self):
        """Keep the second proposal independent of the first committed row."""
        z = np.linalg.cholesky(np.array([[1.0, 0.2], [0.2, 1.0]]))
        result = jacobi.solve(
            z,
            np.eye(2),
            np.ones(2),
            -np.ones(2),
            np.zeros(2, int),
            -np.ones(2, int),
            np.zeros(2),
            np.zeros(2),
            iterations=1,
        )
        np.testing.assert_allclose(result.impulses, [1.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(result.velocity, z.T @ result.impulses, atol=1e-14)
        self.assertEqual(result.work["operator_products"], 8)
        self.assertIsNone(result.work["latch_step"])

    def test_latch_is_once_only_and_midpoint_reuses_operator(self):
        """Charge two attempts while using one permanent half-relaxation latch."""
        energies = iter((10.0, 20.0, 12.0, 13.0))

        def score(*_args):
            return {
                "natural": next(energies) * 1e-5,
                "normal": 0.0,
                "complementarity": 0.0,
                "mdp": 0.0,
                "cone": 0.0,
                "stop": False,
            }

        with patch.object(jacobi.block, "physical_from_residual", side_effect=score):
            result = jacobi.solve(
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
        np.testing.assert_array_equal(result.impulses, [0.75])
        np.testing.assert_array_equal(result.velocity, [0.75])
        self.assertEqual(result.work["sweeps"], 2)
        self.assertEqual(result.work["latch_step"], 1)
        self.assertEqual(result.work["operator_products"], 4)
        self.assertEqual(result.work["midpoint_row_projections"], 1)
        self.assertEqual(result.work["relaxation"], 0.5)

    def test_unsafe_local_metric_preserves_incoming_delta(self):
        """Reproduce independent original metric blocks from the current state."""
        z, diagonal, rhs, types, parents, mu = TestCoulombBlockDataflow().problem()
        incoming = np.tile([0.3, -0.06, 0.08], 2)
        current = np.array([0.1, -0.2, 0.05, -0.05, 0.1, -0.1])
        args = (z, np.eye(6), diagonal, rhs, types, parents, mu, current)
        expected = jacobi.block.solve(*args, iterations=1, incoming=incoming, coupled=False)
        with patch.object(jacobi.block, "local_block", return_value=(None, {"probes": 0, "kind": "unsafe"})):
            result = jacobi.solve(*args, iterations=1, incoming=incoming)
        np.testing.assert_allclose(result.impulses, expected.impulses, atol=1e-12)
        np.testing.assert_allclose(result.velocity, expected.velocity, atol=1e-12)
        np.testing.assert_allclose(result.velocity, current + z.T @ (result.impulses - incoming), atol=1e-12)
        self.assertEqual(result.work["metric_fallbacks"], 2)


if __name__ == "__main__":
    unittest.main()

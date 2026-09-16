# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent simultaneous proposals, fixed latch and physical delta checks."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import spectral_jacobi_control as candidate


class TestSpectralJacobiControl(unittest.TestCase):
    def test_dense_independent_proposals_and_incoming_delta(self):
        """Correct each tangent for its own normal, never another contact."""
        rng = np.random.default_rng(12)
        z = rng.normal(size=(6, 6)) + 3 * np.eye(6)
        gram = z @ z.T
        diagonal = gram.diagonal() + np.array([0.1, 0.2, 0.5, 0.3, 0.4, 0.6])
        incoming = np.array([0.5, 0.03, -0.02, 0.4, -0.03, 0.04])
        current = np.array([0.1, -0.05, 0.03, -0.04, 0.02, 0.06])
        rhs = np.array([-8.0, 2.0, -1.0, -6.0, -2.0, 3.0])
        types, parents = np.tile([0, 2, 2], 2), np.array([-1, 0, 0, -1, 3, 3])
        mu = np.array([0.0, 0.5, 0.5, 0.0, 0.7, 0.7])
        residual = z @ current + rhs
        expected = incoming.copy()
        for normal in (0, 3):
            take = slice(normal + 1, normal + 3)
            expected[normal] = max(0.0, incoming[normal] - residual[normal] / diagonal[normal])
            cross = gram[take, normal] * (expected[normal] - incoming[normal])
            denominator = np.linalg.eigvalsh(gram[take, take])[-1] + max(diagonal[take] - gram.diagonal()[take])
            tangent = incoming[take] - (residual[take] + cross) / denominator
            radius = mu[normal + 1] * expected[normal]
            expected[take] = tangent * min(1.0, radius / np.linalg.norm(tangent))
        # Isolate proposal math from the independently tested latch policy.
        with patch.object(candidate, "merit", return_value=1.0):
            result = candidate.solve(
                z,
                np.eye(6),
                diagonal,
                rhs,
                types,
                parents,
                mu,
                current,
                incoming=incoming,
                iterations=1,
                early_stop=False,
            )
        np.testing.assert_allclose(result.impulses, expected, atol=1e-13)
        np.testing.assert_allclose(result.velocity, current + z.T @ (expected - incoming), atol=1e-13)
        self.assertEqual(result.work["operator_products"], 72)
        self.assertEqual(result.work["setup_cross_products"], 36)
        self.assertEqual(result.work["root_probes"], 0)
        self.assertEqual(result.work["majorizer_products"], 0)

    def test_duplicate_normals_use_frozen_state(self):
        z = np.linalg.cholesky(np.array([[1.0, 0.2], [0.2, 1.0]]))
        result = candidate.solve(
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
        self.assertIsNone(result.work["latch_step"])

    def test_permanent_latch_has_no_second_operator(self):
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

        with patch.object(candidate, "physical_score", side_effect=score):
            result = candidate.solve(
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
        self.assertEqual(result.work["latch_step"], 1)
        self.assertEqual(result.work["relaxation"], 0.5)
        self.assertEqual(result.work["operator_products"], 4)
        self.assertEqual(result.work["midpoint_row_projections"], 1)

    def test_sliding_fixed_point_keeps_denominator_only_cfm(self):
        z = np.array([[2.0, 0.0, 0.0], [0.3, 1.4, 0.0], [-0.2, 0.1, 1.1]])
        incoming = np.array([0.8, -0.4, 0.0])
        result = candidate.solve(
            z,
            np.eye(3),
            np.sum(z * z, axis=1) + np.array([0.2, 0.3, 0.7]),
            np.array([0.0, 0.5, 0.0]),
            np.array([0, 2, 2]),
            np.array([-1, 0, 0]),
            np.array([0.0, 0.5, 0.5]),
            np.zeros(3),
            incoming=incoming,
        )
        np.testing.assert_allclose(result.impulses, incoming, atol=1e-14)
        np.testing.assert_allclose(result.velocity, 0.0, atol=1e-14)
        self.assertTrue(result.work["physical_stop"])
        self.assertEqual(result.work["sweeps"], 1)


if __name__ == "__main__":
    unittest.main()

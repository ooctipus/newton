# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent bounds and workflow controls for the CPU lookahead study."""

import unittest

import numpy as np

from tools.fpgs_bench import spectral_jacobi_control as frozen
from tools.fpgs_bench import spectral_residual_control as candidate


class SpectralResidualTests(unittest.TestCase):
    def test_bound_mixed_diagonals_cross_and_radius(self):
        rng = np.random.default_rng(71)
        for _ in range(512):
            Z = rng.normal(size=(3, 5)) * np.exp(rng.uniform(-2, 2, (3, 1)))
            diagonal = np.sum(Z * Z, axis=1) + np.exp(rng.uniform(-8, 3, 3))
            types, parents, mu = np.array([0, 2, 2]), np.array([-1, 0, 0]), np.full(3, rng.uniform(0, 2))
            impulse = rng.normal(size=3)
            impulse[0] = abs(impulse[0])
            impulse[1:] *= min(1, 0.99 * mu[1] * impulse[0] / np.linalg.norm(impulse[1:]))
            residual = rng.normal(size=3) * 4
            prepared = candidate.prepare(Z, np.eye(5), diagonal, types, parents, mu, impulse)
            proposal, q = candidate.propose(prepared, impulse, residual, 1.0)
            score = frozen.physical_score(residual, impulse, diagonal, types, parents, mu, 1.0)
            self.assertLessEqual(score["natural"], q + 1e-11 * max(1, q))
            self.assertGreaterEqual(prepared["cache"][0][4], max(diagonal[1:]))
            self.assertLessEqual(np.linalg.norm(proposal[1:]), mu[1] * proposal[0] + 1e-12)

    def test_same_proposal_as_independent_normal_first_map(self):
        Z = np.array([[2.0, 1.0], [0.8, 1.3], [1.2, -0.7]])
        diagonal = np.sum(Z * Z, axis=1) + np.array([0.2, 0.1, 0.7])
        types, parents, mu = np.array([0, 2, 2]), np.array([-1, 0, 0]), np.full(3, 0.7)
        rhs = np.array([-1.2, 0.9, -0.4])
        prepared = candidate.prepare(Z, np.eye(2), diagonal, types, parents, mu, np.zeros(3))
        proposal, _q = candidate.propose(prepared, np.zeros(3), rhs, 1.0)
        gram = Z @ Z.T
        expected = np.zeros(3)
        expected[0] = max(0, -rhs[0] / diagonal[0])
        spectral = np.linalg.eigvalsh(gram[1:, 1:])[-1] + 0.7
        expected[1:] = -(rhs[1:] + gram[1:, 0] * expected[0]) / spectral
        expected[1:] *= min(1, mu[1] * expected[0] / np.linalg.norm(expected[1:]))
        np.testing.assert_allclose(proposal, expected, rtol=2e-15, atol=2e-15)

    def test_small_q_does_not_skip_full_complementarity(self):
        result = candidate.solve(
            np.ones((1, 1)),
            np.eye(1),
            np.array([1e12]),
            np.array([1e-3]),
            np.array([0]),
            np.array([-1]),
            np.array([0.0]),
            np.zeros(1),
            iterations=2,
            incoming=np.ones(1),
        )
        self.assertEqual(result.work["sweeps"], 2)
        self.assertEqual(result.work["full_checks"], 2)
        self.assertEqual(result.work["failed_q_gates"], 2)
        self.assertIsNone(result.work["first_stop"])
        self.assertGreater(result.work["final_physical"]["complementarity"], 3e-5)

    def test_lookahead_and_midpoint_accounting_and_momentum(self):
        Z = np.array([[1.0], [1.0], [1.0]])
        result = candidate.solve(
            Z,
            np.eye(1),
            np.ones(3),
            -np.ones(3),
            np.zeros(3, int),
            np.full(3, -1),
            np.zeros(3),
            np.zeros(1),
            iterations=4,
            incoming=np.ones(3),
            early_stop=False,
        )
        work = result.work
        self.assertEqual(work["initial_proposals"], 1)
        self.assertEqual(work["lookahead_proposals"], 4)
        self.assertEqual(work["final_lookahead_proposals"], 1)
        self.assertEqual(work["midpoint_replacement_proposals"], 1)
        self.assertEqual(work["proposal_row_projections"], 18)
        self.assertEqual(work["operator_products"], 24)
        self.assertEqual(work["relaxation"], 0.5)
        self.assertEqual(work["latch_step"], 1)
        np.testing.assert_allclose(result.velocity, Z.T @ (result.impulses - 1), atol=1e-15)
        self.assertLess(work["residual_carry_error"], 1e-14)

    def test_signed_cfm_direct_bound_admission_and_zero_allowance(self):
        args = (
            np.ones((1, 1)),
            np.eye(1),
            np.array([0.8]),
            -np.ones(1),
            np.zeros(1, int),
            np.array([-1]),
            np.zeros(1),
            np.zeros(1),
        )
        result = candidate.solve(*args, iterations=2)
        self.assertEqual(result.work["fallbacks"], 0)
        result = candidate.solve(*args, iterations=2, incoming=-np.ones(1))
        expected = frozen.solve(*args, iterations=2, incoming=-np.ones(1))
        self.assertEqual(result.work["fallbacks"], 1)
        self.assertEqual(result.work["fallback_reason"], "infeasible_incoming")
        np.testing.assert_array_equal(result.impulses, expected.impulses)
        result = candidate.solve(*args[:2], np.ones(1), *args[3:], iterations=0)
        self.assertEqual(result.work["full_checks"], 1)
        self.assertEqual(result.work["proposal_row_projections"], 0)
        self.assertEqual(result.work["operator_products"], 0)


if __name__ == "__main__":
    unittest.main()

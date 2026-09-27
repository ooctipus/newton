# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent continuation and iteration-budget checks for the hybrid owner."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench import coulomb_block_hybrid as hybrid
from tools.fpgs_bench.active_wrench_control import Result


class TestCoulombHybridDataflow(unittest.TestCase):
    def test_parallel_continuation_matches_original_nonzero_recurrence(self):
        """Continue the original projected momentum recurrence without reseeding."""
        z = np.array([[1.0, 0.4, 0.0], [0.2, 1.0, 0.3], [0.3, -0.1, 0.8]])
        gram = z @ z.T
        diagonal = np.diag(gram) + np.array([3.0, 4.0, 5.0])
        rhs = np.array([-1.0, -0.7, -0.8])
        incoming = np.array([0.4, 0.2, 0.1])
        types, parents, mu = np.array([0, 3, 3]), np.full(3, -1), np.zeros(3)
        for value in (z, diagonal, rhs, incoming, types, parents, mu):
            value.setflags(write=False)

        step = np.minimum(np.diag(gram) / np.sum(np.abs(gram), axis=1), 1.0) / diagonal
        expected, extrapolated, momentum = incoming.copy(), incoming.copy(), 1.0
        consumed = 0
        for _ in range(18):
            proposed = np.maximum(extrapolated - step * (rhs + gram @ extrapolated), 0.0)
            change = proposed - expected
            if (extrapolated - proposed) @ change > 0:
                momentum = 1.0
            next_momentum = 0.5 * (1 + np.sqrt(1 + 4 * momentum**2))
            extrapolated = proposed + (momentum - 1) / next_momentum * change
            expected, momentum = proposed, next_momentum
            consumed += 1
            if np.all(np.abs(change) <= 1e-4 * (np.abs(proposed) + 1e-4)):
                break

        result = hybrid.parallel_continue(z, diagonal, rhs, types, parents, mu, incoming, iterations=18)
        np.testing.assert_allclose(result.impulses, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(result.velocity, z.T @ (expected - incoming), rtol=2e-12, atol=2e-12)
        self.assertEqual(result.work["sweeps"], consumed)

    def test_handoff_preserves_reference_and_actual_remaining_budget(self):
        """Hand off current impulses with exactly the unspent original allowance."""
        jacobian = np.array([[1.0, 0.25], [-0.4, 0.8]])
        factor = np.array([[2.0, 0.0], [0.3, 1.5]])
        z = np.linalg.solve(factor, jacobian.T).T
        response = np.linalg.solve(factor.T, z.T).T
        diagonal = np.sum(z**2, axis=1) + 0.1
        rhs, predictor = np.array([-0.6, -0.2]), np.array([-0.3, 0.1])
        types, parents, mu = np.array([0, 3]), np.full(2, -1), np.zeros(2)
        incoming, current, final = np.array([0.2, 0.4]), np.array([0.7, 1.1]), np.array([0.9, 1.2])
        current_velocity = predictor + response.T @ (current - incoming)
        template = block.solve(
            jacobian, factor, diagonal, rhs, types, parents, mu, predictor, iterations=0, incoming=incoming
        )
        work = dict(template.work, sweeps=3, first_stop=None)
        phase = Result(current_velocity, current, work)
        continuation = Result(
            z.T @ (final - current), final, {"sweeps": 21, "majorizer_products": 8, "restart_count": 0}
        )
        with patch.object(block, "solve", return_value=phase) as first:
            with patch.object(hybrid, "parallel_continue", return_value=continuation) as second:
                result = hybrid.solve(
                    jacobian, factor, diagonal, rhs, types, parents, mu, predictor, iterations=24, incoming=incoming
                )
        self.assertEqual(first.call_args.kwargs["iterations"], 6)
        self.assertEqual(second.call_args.kwargs["iterations"], 21)
        np.testing.assert_array_equal(second.call_args.args[6], current)
        expected_zero_rhs = jacobian @ current_velocity + rhs - z @ (z.T @ current)
        np.testing.assert_allclose(second.call_args.args[2], expected_zero_rhs, atol=2e-15)
        np.testing.assert_allclose(result.velocity, predictor + response.T @ (final - incoming), atol=2e-15)
        self.assertEqual(result.work["coupled_sweeps"] + result.work["parallel_sweeps"], 24)

    def test_zero_allowance_does_not_build_majorizer_or_move_state(self):
        """Return the incoming state unchanged when no iterations remain."""
        incoming = np.array([0.3, 0.4])
        result = hybrid.parallel_continue(
            np.eye(2), np.ones(2), -np.ones(2), np.array([0, 3]), np.full(2, -1), np.zeros(2), incoming, iterations=0
        )
        np.testing.assert_array_equal(result.impulses, incoming)
        np.testing.assert_array_equal(result.velocity, np.zeros(2))
        self.assertEqual(result.work["sweeps"], 0)
        self.assertEqual(result.work["majorizer_products"], 0)


if __name__ == "__main__":
    unittest.main()

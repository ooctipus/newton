# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression-first controls for coupled analytic actions and feasible chords."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import krylov_chord_control as control
from tools.fpgs_bench.coulomb_block_hybrid import parallel_continue
from tools.fpgs_bench.test_normal_block_control import problem


class TestKrylovChordControl(unittest.TestCase):
    def test_analytic_action_open_stick_slide_and_moving_radius(self):
        """The complete derivative includes the projected normal-radius term."""
        args = problem(
            [[1, 0, 0], [0.2, 1, 0], [-0.1, 0.3, 1]], [0, 0, 0], types=[0, 2, 2], parents=[-1, 0, 0], mu=[0.5] * 3
        )
        context = control._prepare(*args)
        z, eta = context["Z"], context["eta"]
        x, v = np.array([0.8, 0.1, -0.2]), np.array([0.4, -0.1, 0.3])
        response = z @ (z.T @ v)
        for q in ([-1, 0.3, 0.4], [1, 0.1, 0.1], [1, 2, 0.4]):
            with self.subTest(q=q):
                residual = (x - q) / eta
                _, _, mode, _ = control._map(x, residual, context)
                actual = control._jvp(v, response, mode, context)
                epsilon = 1e-6
                plus = control._map(x + epsilon * v, residual + epsilon * response, context)[0]
                minus = control._map(x - epsilon * v, residual - epsilon * response, context)[0]
                np.testing.assert_allclose(actual, (plus - minus) / (2 * epsilon), rtol=1e-7, atol=1e-8)
        _, _, mode, _ = control._map(x, (x - [1, 2, 0.4]) / eta, context)
        action = control._jvp(np.array([1.0, 0, 0]), np.zeros(3), mode, context)
        self.assertGreater(np.linalg.norm(action[1:]), 0.1)

    def test_chord_is_feasible_and_not_projection_of_line(self):
        """Keep both documented counterexamples to accidental old-line reuse."""
        context = control._prepare(*problem(np.eye(3), [0, 0, 0], types=[0, 2, 2], parents=[-1, 0, 0], mu=[1.0] * 3))
        for initial, step, alpha in (([1, 0, 0], [0, 2, 0], 0.25), ([1, 1, 0], [-2, 0, 1], 0.5)):
            x, delta = np.asarray(initial, float), np.asarray(step, float)
            endpoint = control._retract(x + delta, context)
            chord = x + alpha * (endpoint - x)
            old = control._retract(x + alpha * delta, context)
            self.assertGreater(np.linalg.norm(chord - old), 0.1)
            self.assertLessEqual(np.linalg.norm(chord[1:]), chord[0] + 1e-15)

    def test_coupled_normals_and_literal_singular_continuation(self):
        """Coupled normals solve physically; inconsistent rank loss is guarded."""
        args = problem(np.linalg.cholesky([[1, -0.4], [-0.4, 1]]), [-1, -0.5])
        result = control.solve(*args)
        np.testing.assert_allclose(args[0] @ result.velocity + args[3], 0, atol=3e-5)
        self.assertTrue(result.work["physical_stop"])
        self.assertGreater(result.work["krylov_steps"], 0)
        bad = problem([[1], [1]], [-1, -2])
        result = control.solve(*bad, iterations=4)
        self.assertTrue(result.work["fallback"])
        seed = np.asarray(result.work["fallback_entry_impulse"])
        original = parallel_continue(
            bad[0], bad[2], bad[3], *bad[4:7], seed, iterations=4 - result.work["candidate_sweeps"]
        )
        np.testing.assert_array_equal(result.impulses, original.impulses)

    def test_inactive_rhs_shift_and_full_forcing_norm(self):
        """Nonzero inactive deltas remain in the complete linear equation."""
        args = problem(np.linalg.cholesky([[1, 0.3], [0.3, 1]]), [-1, 2])
        context = control._prepare(*args)
        x = np.array([0.2, 0.4])
        r = args[3] + args[0] @ (args[0].T @ x)
        f, pg, mode, active = control._map(x, r, context)
        self.assertEqual(active.tolist(), [0])
        delta, detail = control._direction(x, r, f, pg, mode, active, np.linalg.norm(f), context)
        np.testing.assert_allclose(delta[1], -x[1], atol=1e-15)
        matrix = np.column_stack(
            [control._jvp(np.eye(2)[i], args[0] @ (args[0].T @ np.eye(2)[i]), mode, context) for i in range(2)]
        )
        self.assertLessEqual(np.linalg.norm(matrix @ delta + f), detail["forcing_target"] + 1e-12)
        self.assertAlmostEqual(detail["forcing_target"], 0.5 * np.linalg.norm(f))
        self.assertGreater(context["work"]["inactive_coupling_products"], 0)

    def test_current_rows_held_factor_and_incoming_applied_state(self):
        """Publication decodes committed deltas, including unsupported warm input."""
        z = np.array([[1, 0], [0.3, 1]])
        held = np.array([[1.4, 0], [0.2, 0.9]])
        args = list(problem(z, [-1, -0.5]))
        args[0], args[1] = z @ held.T, held
        incoming = np.array([0.2, 0.1])
        args[7] = np.linalg.solve(held.T, z.T @ incoming)
        result = control.solve(*args, incoming=incoming, iterations=4)
        original = parallel_continue(z, args[2], args[3], *args[4:7], incoming, iterations=4)
        self.assertEqual(result.work["fallback_reason"], "nonzero_incoming")
        np.testing.assert_allclose(result.impulses, original.impulses, atol=1e-14)
        np.testing.assert_allclose(held.T @ (result.velocity - args[7]), z.T @ (result.impulses - incoming), atol=1e-13)

    def test_late_guard_does_not_reset_budget_or_state(self):
        """Injected slow feasible corrections leave exactly one original pass."""
        args = problem([[1]], [-1])
        calls = 0

        def slow(x, residual, f, pg, mode, active, initial_norm, context):
            """Exercise bookkeeping with a controlled descent, not a new policy."""
            nonlocal calls
            calls += 1
            if calls == 24:
                raise control.KrylovFailure("test_late_guard")
            return 0.25 * (pg - x), {"krylov_steps": 0, "forcing_target": 0.0}

        with patch.object(control, "_direction", side_effect=slow):
            result = control.solve(*args, early_stop=False)
        self.assertEqual(result.work["candidate_sweeps"], 23)
        self.assertEqual(result.work["fallback_allowance"], 1)
        seed = np.asarray(result.work["fallback_entry_impulse"])
        original = parallel_continue(args[0], args[2], args[3], *args[4:7], seed, iterations=1)
        np.testing.assert_array_equal(result.impulses, original.impulses)
        self.assertLessEqual(result.work["sweeps"], 24)

    def test_zero_budget_and_zero_residual_are_not_false_passes(self):
        """Always check physical state, even without a Newton direction."""
        result = control.solve(*problem([[1]], [-1]), iterations=0)
        self.assertFalse(result.work["physical_stop"])
        self.assertEqual(result.work["sweeps"], 0)
        self.assertEqual(result.work["fresh_checks"], 1)
        result = control.solve(*problem([[1]], [1]))
        self.assertTrue(result.work["physical_stop"])
        self.assertEqual(result.work["krylov_steps"], 0)

    def test_false_carried_stop_requires_fresh_physical_check(self):
        """A false carried pass cannot consume a free correction or escape final checks."""
        original = control._physical

        def optimistic(x, residual, context):
            """Inject a carried-state false positive without changing the fresh gate."""
            score = original(x, residual, context)
            if context["work"]["fresh_checks"] == 0:
                score["stop"] = True
            return score

        with patch.object(control, "_physical", side_effect=optimistic):
            zero = control.solve(*problem([[1]], [-1]), iterations=0)
            one = control.solve(*problem([[1]], [-1]), iterations=1)
        self.assertFalse(zero.work["physical_stop"])
        self.assertEqual(zero.work["sweeps"], 0)
        self.assertEqual(one.work["false_carried_stops"], 1)
        self.assertEqual(one.work["sweeps"], 1)
        self.assertTrue(one.work["physical_stop"])

    def test_complete_sticking_contact_matches_independent_linear_solution(self):
        """A full contact, not only normal rows, exercises the right block solve."""
        z = np.array([[1, 0, 0], [0.2, 1, 0], [-0.1, 0.3, 1]])
        args = problem(z, [-1, 0.7, -0.3], types=[0, 2, 2], parents=[-1, 0, 0], mu=[10.0] * 3)
        result = control.solve(*args)
        expected = np.linalg.solve(z @ z.T, -args[3])
        np.testing.assert_allclose(result.impulses, expected, atol=3e-5)
        np.testing.assert_allclose(result.velocity, z.T @ result.impulses, atol=1e-12)
        self.assertTrue(result.work["physical_stop"])


if __name__ == "__main__":
    unittest.main()

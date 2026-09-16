# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for bounded physical normal elimination and literal fallback."""

import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import normal_block_control as control
from tools.fpgs_bench.coulomb_block_hybrid import parallel_continue


def problem(z, bias, *, types=None, parents=None, mu=None, cfm=0.02):
    """Make an independently specified physical kinetic problem."""
    z = np.asarray(z, float)
    count, dofs = z.shape
    return (
        z,
        np.eye(dofs),
        np.sum(z * z, axis=1) + cfm,
        np.asarray(bias, float),
        np.full(count, 3) if types is None else np.asarray(types),
        np.full(count, -1) if parents is None else np.asarray(parents),
        np.zeros(count) if mu is None else np.asarray(mu, float),
        np.zeros(dofs),
    )


class TestNormalBlockControl(unittest.TestCase):
    def test_dropped_positive_normal_updates_every_residual(self):
        """A dropped working-set ID must remain in the net all-row update."""
        z = np.array([[1, 0], [0.8, 0.6], [0.2, 0.9]])
        impulse = np.array([0.1, 0.5, 0.0])
        residual = z @ (z.T @ impulse) + [-1, -0.1, 0.2]
        context = {
            "Z": z,
            "work": Counter(),
            "columns": {},
            "factor_ids": None,
            "factor": None,
            "normal_rows": np.array([0, 1]),
            "diagonal": np.sum(z * z, axis=1) + 0.02,
            "cold_scale": 2.0,
        }
        proposed, carried, detail = control._normal_step(impulse, residual, context)
        np.testing.assert_allclose(proposed, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(carried, residual + z @ (z.T @ (proposed - impulse)), atol=1e-12)
        self.assertIn(1, detail["dropped_ids"])
        self.assertIn(1, detail["changed_normal_ids"])
        self.assertEqual(context["work"]["normal_publish_products"], 6)

    def test_open_normal_activation_and_denominator_only_cfm(self):
        """An initially open normal must enter; CFM cannot change the LCP."""
        z = np.linalg.cholesky([[1, -0.8], [-0.8, 1]])
        args = problem(z, [-1, 0.5], cfm=9.0)
        result = control.solve(*args)
        np.testing.assert_allclose(result.impulses, [5 / 3, 5 / 6], atol=1e-12)
        np.testing.assert_allclose(z @ result.velocity + args[3], 0, atol=1e-12)
        self.assertFalse(result.work["fallback"])
        self.assertEqual(result.work["column_ids"], [0, 1])
        self.assertGreaterEqual(result.work["pivot_enters"], 1)
        self.assertTrue(result.work["physical_stop"])

    def test_singular_and_inconsistent_normals_use_original(self):
        """Never regularize duplicate or infeasible normal blocks silently."""
        for z, bias in (([[1], [1]], [-1, -1]), ([[1], [1]], [-1, -2]), ([[1], [-1]], [-1, -1])):
            with self.subTest(z=z, bias=bias):
                args = problem(z, bias)
                result = control.solve(*args, iterations=5)
                expected = parallel_continue(*args[:1], args[2], args[3], *args[4:7], np.zeros(2), iterations=5)
                self.assertTrue(result.work["fallback"])
                self.assertEqual(result.work["fallback_reason"], "rank")
                self.assertEqual(result.work["candidate_sweeps"], 0)
                np.testing.assert_array_equal(result.impulses, expected.impulses)
                np.testing.assert_allclose(result.velocity, expected.velocity, atol=1e-14)
                self.assertGreater(result.work["fallback_majorizer_products"], 0)

    def test_disk_current_rows_held_factor_and_momentum(self):
        """Current rows with held L preserve each cone and committed momentum."""
        z = np.array([[1, 0, 0], [0.2, 1, 0], [-0.1, 0.3, 1]])
        args = list(problem(z, [-1, 0.7, -0.3], types=[0, 2, 2], parents=[-1, 0, 0], mu=[0.5] * 3))
        held = np.array([[2, 0, 0], [0.2, 1.4, 0], [-0.1, 0.1, 0.9]])
        args[0], args[1] = z @ held.T, held
        result = control.solve(*args)
        self.assertGreaterEqual(result.impulses[0], 0)
        self.assertLessEqual(np.linalg.norm(result.impulses[1:]), 0.5 * result.impulses[0] + 1e-12)
        np.testing.assert_allclose(held.T @ result.velocity, z.T @ result.impulses, atol=1e-12)
        self.assertLess(result.work["residual_carry_error"], 1e-12)
        self.assertGreater(result.work["full_scores"], 0)

    def test_nonzero_incoming_continues_without_reapplying(self):
        """Unsupported warm input falls back from incoming-applied velocity."""
        args = list(problem([[1, 0], [0.3, 1]], [-1, -0.5]))
        incoming = np.array([0.2, 0.1])
        args[7] = args[0].T @ incoming
        result = control.solve(*args, incoming=incoming, iterations=4)
        expected = parallel_continue(args[0], args[2], args[3], *args[4:7], incoming, iterations=4)
        self.assertEqual(result.work["fallback_reason"], "nonzero_incoming")
        np.testing.assert_array_equal(result.impulses, expected.impulses)
        np.testing.assert_allclose(result.velocity, args[7] + expected.velocity, atol=1e-14)
        self.assertEqual(result.work["incoming_rhs_products"], 2 * args[0].size)

    def test_late_guard_keeps_current_state_and_remaining_one(self):
        """A failed twenty-fourth trial cannot reset either state or budget."""
        args = problem(
            [[1, 0, 0], [0.2, 1, 0], [0, 0.3, 1]], [-1, 0.7, -0.3], types=[0, 2, 2], parents=[-1, 0, 0], mu=[0.5] * 3
        )
        original = control._normal_step
        calls = 0

        def late(*values):
            """Inject a diagnosed guard at the current trial boundary only."""
            nonlocal calls
            calls += 1
            if calls == 24:
                original(*values)
                raise control.NormalFailure("pivot_guard")
            return original(*values)

        with patch.object(control, "_normal_step", side_effect=late):
            result = control.solve(*args, early_stop=False)
        self.assertEqual(result.work["candidate_sweeps"], 23)
        self.assertEqual(result.work["fallback_allowance"], 1)
        seed = np.asarray(result.work["fallback_entry_impulse"])
        expected = parallel_continue(args[0], args[2], args[3], *args[4:7], seed, iterations=1)
        np.testing.assert_array_equal(result.impulses, expected.impulses)
        np.testing.assert_allclose(result.velocity, args[0].T @ seed + expected.velocity, atol=1e-14)
        self.assertLessEqual(result.work["sweeps"], 24)

    def test_transition_guard_and_zero_budget(self):
        """Exactly 2m transitions are allowed; a zero allowance does no solve."""
        work = {"pivot_transitions": 0, "pivot_enters": 0, "pivot_drops": 0}
        for _ in range(4):
            control._transition(work, 2, 0, "enter")
        with self.assertRaisesRegex(control.NormalFailure, "pivot_guard"):
            control._transition(work, 2, 0, "drop")
        result = control.solve(*problem([[1]], [-1]), iterations=0)
        np.testing.assert_array_equal(result.impulses, [0])
        self.assertEqual(result.work["sweeps"], 0)
        self.assertEqual(result.work["column_builds"], 0)


if __name__ == "__main__":
    unittest.main()

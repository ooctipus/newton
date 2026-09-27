# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rollback, endpoint selection and allowance controls for global safeguarding."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import coulomb_block_guarded as guarded
from tools.fpgs_bench.active_wrench_control import Result


class TestCoulombBlockGuarded(unittest.TestCase):
    def problem(self):
        """A scalar proximal iteration advances lambda by one half each visit."""
        return (
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.array([2.0]),
            np.array([-1.0]),
            np.array([0]),
            np.array([-1]),
            np.zeros(1),
            np.zeros(1),
        )

    def test_rejected_pass_restores_both_states_and_remains_charged(self):
        """Controlled diagnostic scores isolate strict acceptance and rollback."""
        for energies, expected, attempts in (((10, 5, 6), 0.5, 2), ((10, 10), 0.0, 1)):
            with self.subTest(energies=energies):
                sequence = iter(energies)

                def score(*_args, scores=sequence):
                    return {
                        "natural": 0.0,
                        "normal": next(scores) * 3e-5,
                        "complementarity": 0.0,
                        "mdp": 0.0,
                        "cone": 0.0,
                        "stop": False,
                    }

                with patch.dict(guarded.prefix_solve.__globals__, physical_from_residual=score):
                    result = guarded.prefix_solve(*self.problem(), iterations=6, early_stop=True)
                np.testing.assert_array_equal(result.impulses, [expected])
                np.testing.assert_array_equal(result.velocity, [expected])
                self.assertEqual(result.work["sweeps"], attempts)
                self.assertEqual(result.work["accepted_coupled_sweeps"], attempts - 1)
                self.assertEqual(result.work["snapshot_values"], 2 * attempts)
                self.assertEqual(result.work["rollback_values"], 2)
                self.assertTrue(result.work["guard_rejected"])

    def test_worse_tail_endpoint_restores_handoff_without_refunding_work(self):
        """One final comparison selects a matched lambda/velocity state."""
        args = self.problem()
        phase = guarded.prefix_solve(*args, iterations=1, early_stop=True)
        phase.work.update(sweeps=2, guard_rejected=True, rollback_values=2)
        # This physical endpoint is worse than the handoff, but still consumes
        # all twenty-two remaining original iterations before selection.
        tail = Result(np.array([2.5]), np.array([3.0]), {"sweeps": 22, "majorizer_products": 1})
        with patch.object(guarded, "prefix_solve", return_value=phase):
            with patch.object(guarded, "parallel_continue", return_value=tail) as continuation:
                result = guarded.solve(*args, iterations=24)
        self.assertEqual(continuation.call_args.kwargs["iterations"], 22)
        np.testing.assert_array_equal(continuation.call_args.args[6], [0.5])
        np.testing.assert_array_equal(result.impulses, [0.5])
        np.testing.assert_array_equal(result.velocity, [0.5])
        self.assertTrue(result.work["selected_handoff"])
        self.assertEqual(result.work["coupled_sweeps"] + result.work["parallel_sweeps"], 24)
        self.assertEqual(result.work["post_tail_scan_row_dots"], 1)
        self.assertEqual(result.work["unused_allowance"], 0)

    def test_zero_allowance_does_not_attempt_or_continue(self):
        """An empty allowance preserves current incoming state with no majorizer."""
        args = self.problem()
        with patch.object(guarded, "parallel_continue") as continuation:
            result = guarded.solve(*args, iterations=0, incoming=np.array([0.25]))
        continuation.assert_not_called()
        np.testing.assert_array_equal(result.impulses, [0.25])
        np.testing.assert_array_equal(result.velocity, [0.0])
        self.assertEqual(result.work["coupled_sweeps"], 0)
        self.assertEqual(result.work["majorizer_products"], 0)
        self.assertEqual(result.work["snapshot_values"], 0)


if __name__ == "__main__":
    unittest.main()

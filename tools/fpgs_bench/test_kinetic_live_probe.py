# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host controls for the live eager lifecycle observer, not physics acceptance."""

import unittest

import numpy as np

from tools.fpgs_bench.kinetic_live_probe import check_held, check_publication, transition


class TestKineticLiveProbe(unittest.TestCase):
    def test_reuse_rejects_changed_held_values(self):
        """Reject operator mutation on an unrequested reuse call."""
        old = {"T": np.ones((2, 180)), "generation": np.array([1, 1])}
        check_held(old, old, False)
        changed = {**old, "T": old["T"] * 2}
        with self.assertRaises(ValueError):
            check_held(old, changed, False)
        check_held(old, changed, True)

    def test_publication_uses_physical_quaternion_equivalence(self):
        """Accept quaternion sign while rejecting wrong physical body velocity."""
        q = np.array([[1, 2, 3, 0, 0, 0, 1]], dtype=float)
        same = q.copy()
        same[:, 3:] *= -1
        velocity = np.zeros((1, 6))
        check_publication(q, velocity, same, velocity)
        with self.assertRaises(ValueError):
            check_publication(q, velocity + 1, q, velocity)

    def test_fixed_transition_schedule(self):
        """Keep perturbations explicit and outside ordinary performance mode."""
        self.assertEqual(
            [transition(i) for i in range(7)], [None, "force_target", None, "subset_reset", None, "odd_refresh", None]
        )


if __name__ == "__main__":
    unittest.main()

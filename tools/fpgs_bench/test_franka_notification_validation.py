# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete host topology checks without per-world Python execution."""

import sys
import unittest

import numpy as np

from newton._src.solvers.feather_pgs.franka_row_packets import validate_prefix_topology


def make_topology(count, seed=7):
    """Mix free/primary articulations and independently shuffle every list."""
    rng = np.random.default_rng(seed)
    primary = np.zeros(2 * count, dtype=bool)
    primary[rng.permutation(2 * count)[:count]] = True
    groups = rng.permutation(np.flatnonzero(primary))
    worlds = np.empty(2 * count, dtype=np.int32)
    worlds[primary] = rng.permutation(count)
    worlds[~primary] = rng.permutation(count)
    starts = np.r_[0, np.cumsum(np.where(primary, 9, 6))]
    counts = np.where(primary, rng.integers(0, 3, size=2 * count), 0)
    mimic_start = np.r_[0, np.cumsum(counts)]
    entries = rng.permutation(mimic_start[-1])
    dof0, dof1, mimic_world = (np.empty(len(entries), dtype=np.int32) for _ in range(3))
    for art in groups:
        indices = entries[mimic_start[art] : mimic_start[art + 1]]
        dof0[indices] = starts[art] + 7
        dof1[indices] = starts[art] + 8
        mimic_world[indices] = worlds[art]
    return [groups, worlds, starts, mimic_start, entries, dof0, dof1, mimic_world]


def scalar_reference(groups, worlds, starts, mimic_start, entries, dof0, dof1, mimic_world):
    """Original all-entry semantics, independent of the vectorized implementation."""
    if len(groups) != len(set(worlds[groups])) or len(groups) != len(set(worlds)):
        return False
    for art in groups:
        if starts[art + 1] - starts[art] != 9 or mimic_start[art + 1] - mimic_start[art] > 2:
            return False
        for entry in entries[mimic_start[art] : mimic_start[art + 1]]:
            if mimic_world is not None and mimic_world[entry] != worlds[art]:
                return False
            if not starts[art] <= dof0[entry] < starts[art + 1]:
                return False
            if not starts[art] <= dof1[entry] < starts[art + 1]:
                return False
    return True


class TestFrankaNotificationValidation(unittest.TestCase):
    def test_interpreted_work_is_independent_of_world_count(self):
        """Detect the old O(worlds) Python loop without timing noisy CPU clocks."""
        counts = []
        for size in (32, 4096):
            topology = make_topology(size)
            events = 0

            def trace(frame, event, arg):
                nonlocal events
                if frame.f_code is validate_prefix_topology.__code__:
                    events += event == "line"
                    return trace
                return None

            previous = sys.gettrace()
            try:
                sys.settrace(trace)
                result = validate_prefix_topology(*topology)
            finally:
                sys.settrace(previous)
            self.assertTrue(result)
            counts.append(events)
        self.assertLessEqual(counts[1], counts[0] + 100)

    def test_randomized_complete_scalar_equivalence(self):
        """Check reordered valid plans and late invalid entries against the old law."""
        for seed in range(16):
            original = make_topology(31, seed)
            mutations = [None, "dof0", "dof1", "world", "duplicate", "width"]
            for mutation in mutations:
                arrays = [array.copy() for array in original]
                groups, worlds, starts, mimic_start, entries, dof0, dof1, mimic_world = arrays
                occupied = groups[np.diff(mimic_start)[groups] > 0]
                art = occupied[-1]
                entry = entries[mimic_start[art + 1] - 1]
                if mutation == "dof0":
                    dof0[entry] = starts[art + 1]
                elif mutation == "dof1":
                    dof1[entry] = starts[art] - 1
                elif mutation == "world":
                    mimic_world[entry] = (worlds[art] + 1) % 31
                elif mutation == "duplicate":
                    worlds[groups[-1]] = worlds[groups[0]]
                elif mutation == "width":
                    starts[groups[-1] + 1] -= 1
                for world_check in (True, False):
                    arrays[-1] = mimic_world if world_check else None
                    with self.subTest(seed=seed, mutation=mutation, world_check=world_check):
                        expected = scalar_reference(*arrays)
                        self.assertEqual(validate_prefix_topology(*arrays), expected)

    def test_empty_and_zero_one_two_mimics(self):
        """Preserve empty admission and enforce the two-entry bound."""
        self.assertTrue(validate_prefix_topology(*make_topology(0)))
        for count in range(4):
            args = [
                np.array([0]),
                np.array([0]),
                np.array([0, 9]),
                np.array([0, count]),
                np.arange(count),
                np.full(count, 7),
                np.full(count, 8),
                np.zeros(count, dtype=int),
            ]
            self.assertEqual(validate_prefix_topology(*args), count <= 2)

    def test_malformed_indices_reject_before_gather(self):
        """Reject malformed indices without wrapping or raising on array gathers."""
        for name in ("group", "start", "end", "entry", "dof_length", "world_length"):
            args = [
                np.array([0]),
                np.array([0]),
                np.array([0, 9]),
                np.array([0, 1]),
                np.array([0]),
                np.array([7]),
                np.array([8]),
                np.array([0]),
            ]
            if name == "group":
                args[0][0] = -1
            elif name == "start":
                args[3][0] = -1
            elif name == "end":
                args[3][-1] = 2
            elif name == "entry":
                args[4][0] = -1
            elif name == "dof_length":
                args[5] = np.array([], dtype=int)
            elif name == "world_length":
                args[7] = np.array([], dtype=int)
            with self.subTest(name=name):
                self.assertFalse(validate_prefix_topology(*args))


if __name__ == "__main__":
    unittest.main()

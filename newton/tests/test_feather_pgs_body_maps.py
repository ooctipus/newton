# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check tree ownership and response supports built during solver setup."""

import unittest

import numpy as np

from newton._src.solvers.feather_pgs.body_mapping import build_body_maps
from newton._src.solvers.feather_pgs.tree_plan import TreePlan


def maps(bodies, child, ancestor, qd, starts, ends, response_start, response_count):
    child, qd, starts, ends, response_start, response_count = (
        np.asarray(values, dtype=np.int32) for values in (child, qd, starts, ends, response_start, response_count)
    )
    forest = TreePlan.build(np.asarray(ancestor, dtype=np.int32))
    return build_body_maps(bodies, child, forest, qd, starts, ends, response_start, response_count)


class TestBodyMaps(unittest.TestCase):
    def test_loop_tail_fixed_root_and_unowned_body(self):
        actual = maps(
            6, [0, 1, 2, 1, 3, 4], [-1, 0, 1, 2, -1, 4], [0, 0, 1, 3, 3, 38, 39], [0, 4, 6], [3, 6], [0, 3], [3, 36]
        )
        expected = (
            [0, 1, 2, 4, 5, -1],
            [0, 0, 0, 1, 1, -1],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 7, 0xFFFFFFFF, 0xFFFFFFFF, 0],
            [-1, 0, -1, -1, -1, -1],
        )
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(observed, reference)

    def test_response_window_and_high_singleton(self):
        actual = maps(3, [0, 1, 2], [-1, 0, 0], [0, 6, 7, 46], [0, 3], [3], [6], [40])
        np.testing.assert_array_equal(actual[2], [0, 1, 1])
        np.testing.assert_array_equal(actual[3], [0, 1, 0xFFFFFFFE])
        np.testing.assert_array_equal(actual[4], [-1, 6, -1])
        actual = maps(2, [0, 1], [-1, -1], [0, 39, 40], [0, 2], [2], [0], [40])
        self.assertEqual(actual[2][1], 1)
        self.assertEqual(actual[3][1], 0)
        self.assertEqual(actual[4][1], 39)

    def test_unsorted_acyclic_ancestry(self):
        actual = maps(3, [0, 1, 2], [2, -1, 1], [0, 1, 2, 3], [0, 3], [3], [0], [3])
        np.testing.assert_array_equal(actual[3], [7, 2, 6])
        np.testing.assert_array_equal(actual[4], [-1, 1, -1])

    def test_cyclic_ancestry_is_no_forest(self):
        """A joint ancestry with a cycle anywhere, reachable from an owned body or not, is no joint forest."""
        for ancestor in ([0, -1], [1, 0]):
            with self.subTest(ancestor=ancestor), self.assertRaisesRegex(ValueError, "cycle|other nodes"):
                TreePlan.build(np.asarray(ancestor, dtype=np.int32))

    def test_last_tree_owner_and_empty_articulation(self):
        actual = maps(1, [0, 0], [-1, 0], [0, 1, 2], [0, 0, 2], [0, 2], [0, 0], [0, 2])
        self.assertEqual(actual[0][0], 1)
        self.assertEqual(actual[1][0], 1)
        self.assertEqual(actual[3][0], 3)
        actual = maps(2, [], [], [0], [0, 0], [0], [0], [0])
        np.testing.assert_array_equal(actual[4], [-1, -1])


if __name__ == "__main__":
    unittest.main()

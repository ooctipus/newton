# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of FeatherPGS's host index toolkit against element-by-element references."""

import unittest

import numpy as np

from newton._src.solvers.feather_pgs import host_index


class TestFeatherPGSHostIndex(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_offsets_bound_each_segment(self):
        counts = self.rng.integers(0, 5, 40)
        bounds = host_index.offsets(counts)
        self.assertEqual(bounds.dtype, np.int64)
        self.assertEqual(bounds.tolist(), [0, *np.cumsum(counts).tolist()])
        self.assertEqual(host_index.offsets(np.array([3, 4], dtype=np.int32), dtype=np.int32).dtype, np.int32)
        self.assertEqual(host_index.offsets([]).tolist(), [0])

    def test_expand_lists_each_element_with_its_rank(self):
        counts = self.rng.integers(0, 5, 40)
        segment, rank = host_index.expand(counts)
        expected = [(s, k) for s, count in enumerate(counts) for k in range(count)]
        self.assertEqual(list(zip(segment.tolist(), rank.tolist(), strict=True)), expected)

    def test_segmented_rank_restarts_at_each_segment(self):
        segment = np.sort(self.rng.integers(0, 8, 60))
        weight = self.rng.integers(0, 4, 60)
        for weights in (None, weight, weight > 1):
            values = np.ones(60, dtype=np.int64) if weights is None else np.asarray(weights, dtype=np.int64)
            expected = [int(values[:i][segment[:i] == segment[i]].sum()) for i in range(60)]
            self.assertEqual(host_index.segmented_rank(segment, weights).tolist(), expected)
        self.assertEqual(host_index.segmented_rank(np.zeros(0, dtype=np.int32)).size, 0)

    def test_group_by_keeps_each_keys_elements_in_order(self):
        keys = self.rng.integers(0, 6, 50)
        bounds, order = host_index.group_by(keys, 8)
        self.assertEqual(bounds.size, 9)
        for key in range(8):
            self.assertEqual(order[bounds[key] : bounds[key + 1]].tolist(), np.flatnonzero(keys == key).tolist())

    def test_ordered_unique_keeps_first_occurrences(self):
        values = self.rng.integers(0, 9, 50)
        expected = list(dict.fromkeys(values.tolist()))
        self.assertEqual(host_index.ordered_unique(values).tolist(), expected)

    def test_lookup_finds_the_holding_segment(self):
        bounds = np.array([2, 2, 5, 9, 9, 12])
        items = np.arange(-1, 14)
        expected = [next((s for s in range(5) if bounds[s] <= i < bounds[s + 1]), -1) for i in items]
        self.assertEqual(host_index.lookup(bounds, items).tolist(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete raw-ID bucketing without changing contact capacity."""

import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.raw_world_contacts import RawWorldContactBuckets


class TestRawWorldContacts(unittest.TestCase):
    def setUp(self):
        """Use CPU by default and only the explicitly selected CUDA device."""
        self.device = os.environ.get("FPGS_TEST_DEVICE", "cpu")

    def case(self, pairs, *, worlds=3, capacity=None, count=None):
        """Build responsive, prescribed and global endpoints in three worlds."""
        capacity = max(1, len(pairs)) if capacity is None else capacity

        def array(values):
            return wp.array(values, dtype=int, device=self.device)

        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        shapes = np.full((capacity, 2), -999, dtype=np.int32)
        shapes[: len(pairs)] = pairs
        # Shapes 0..5 belong to alternating world primary/prescribed bodies;
        # shape 6 is global. No response mask may remove prescribed pairs.
        inputs = [
            array([len(pairs) if count is None else count]),
            array(shapes[:, 0]),
            array(shapes[:, 1]),
            array([0, 1, 2, 3, 4, 5, -1]),
            array([0, 1, 2, 3, 4, 5]),
            array([0, 0, 1, 1, 2, 2]),
        ]
        owner = RawWorldContactBuckets(worlds, capacity, device=self.device)
        return owner, inputs

    def groups(self, owner):
        """Read only the materialized prefixes, ignoring deliberately stale tails."""
        offsets, ids = owner.data.offsets.numpy(), owner.data.ids.numpy()
        return [sorted(ids[offsets[w] : offsets[w + 1]].tolist()) for w in range(owner.world_count)]

    def test_complete_ids_and_prescribed_endpoints(self):
        """Retain every raw ID, including contacts with no responsive endpoint."""
        owner, inputs = self.case([(4, 6), (0, 1), (3, 6), (5, 6), (0, 6), (2, 3)])
        owner.build(*inputs)
        self.assertEqual(owner.data.invalid.numpy()[0], 0)
        self.assertEqual(self.groups(owner), [[1, 4], [2, 5], [0, 3]])
        self.assertEqual(owner.data.ids.shape[0], 6)
        self.assertEqual(owner.storage_bytes, (6 + 2 * 4 + 1) * 4)

    def test_unbounded_world_segment(self):
        """Bucket more than the solver's dense row capacity without truncation."""
        owner, inputs = self.case([(0, 6)] * 1025)
        owner.build(*inputs)
        self.assertEqual(owner.data.invalid.numpy()[0], 0)
        self.assertEqual(self.groups(owner), [list(range(1025)), [], []])

    def test_rebuild_empty_and_poisoned_tail(self):
        """Replace previous prefixes and ignore invalid descriptors beyond raw count."""
        owner, inputs = self.case([(0, 6), (2, 6)], capacity=8)
        owner.build(*inputs)
        inputs[0].assign([0])
        owner.build(*inputs)
        np.testing.assert_array_equal(owner.data.offsets.numpy(), [0, 0, 0, 0])
        self.assertEqual(owner.data.invalid.numpy()[0], 0)
        inputs[0].assign([1])
        owner.build(*inputs)
        self.assertEqual(self.groups(owner), [[0], [], []])
        self.assertEqual(owner.data.invalid.numpy()[0], 0)

    def test_cross_world_unowned_and_invalid_shape(self):
        """Reject the entire optimized frame for unsafe raw ownership."""
        for pair in ((0, 2), (6, 6), (-1, -1), (-2, 0), (7, 0)):
            with self.subTest(pair=pair):
                owner, inputs = self.case([pair])
                owner.build(*inputs)
                self.assertNotEqual(owner.data.invalid.numpy()[0], 0)

    def test_invalid_count_and_map_ranges(self):
        """Flag overflow or malformed maps without indexing outside allocated storage."""
        for count in (-1, 2):
            owner, inputs = self.case([(0, 6)], count=count)
            owner.build(*inputs)
            self.assertNotEqual(owner.data.invalid.numpy()[0], 0)
        for index, values in ((3, [6, 1, 2, 3, 4, 5, -1]), (4, [6, 1, 2, 3, 4, 5]), (5, [3, 0, 1, 1, 2, 2])):
            owner, inputs = self.case([(0, 6)])
            inputs[index].assign(values)
            owner.build(*inputs)
            self.assertNotEqual(owner.data.invalid.numpy()[0], 0)

    def test_raw_identity_preserves_anchor_neighborhood(self):
        """Keep original neighbors addressable instead of treating CSR order as raw order."""
        pairs = [(0, 6), (2, 6), (4, 6)] * 17
        owner, inputs = self.case(pairs)
        owner.build(*inputs)
        for world, ids in enumerate(self.groups(owner)):
            self.assertEqual(ids, list(range(world, len(pairs), 3)))
            for raw_id in ids:
                self.assertEqual(tuple(pairs[raw_id]), (2 * world, 6))

    def test_layout_mismatch_rejected_before_launch(self):
        """Refuse descriptor lengths that cannot represent the exact allocated prefix."""
        owner, inputs = self.case([(0, 6)])
        inputs[2] = wp.array([6, 6], dtype=int, device=self.device)
        with self.assertRaises(ValueError):
            owner.build(*inputs)
        for worlds, capacity in ((0, 1), (1, -1)):
            with self.assertRaises(ValueError):
                RawWorldContactBuckets(worlds, capacity, device=self.device)

    @unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "cpu").startswith("cuda"), "CUDA graph control")
    def test_graph_reentry_and_prefix_changes(self):
        """Replay alternate graphs against one scratch owner without stale count admission."""
        owner, inputs = self.case([(0, 6), (2, 6)], capacity=8)
        owner.build(*inputs)
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as first:
            owner.build(*inputs)
        with wp.ScopedCapture(device=self.device) as second:
            owner.build(*inputs)
        for count, graph in ((2, first.graph), (0, second.graph), (1, first.graph), (2, second.graph)):
            inputs[0].assign([count])
            wp.capture_launch(graph)
            self.assertEqual(owner.data.invalid.numpy()[0], 0)
            self.assertEqual(int(owner.data.offsets.numpy()[-1]), count)


if __name__ == "__main__":
    unittest.main()

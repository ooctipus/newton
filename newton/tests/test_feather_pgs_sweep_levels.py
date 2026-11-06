# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The level builder (:mod:`.sweep_levels`): its work lists.

The Jacobi tiers' worlds share nothing, so the builder buckets them by their rows into one
:class:`~newton._src.core.execution.Queue`, a bucket a tier: each tier's bucket holds exactly the worlds of its rows.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.core.execution import Queue
from newton._src.solvers.feather_pgs import sweep_levels


def _devices():
    return ("cpu", *(("cuda:0",) if wp.is_cuda_available() else ()))


class TestWorldLists(unittest.TestCase):
    def test_each_tier_buckets_exactly_its_worlds(self):
        """A world of dense rows only and at most the bound lands in the bucket of its 32-row tier, once; others in none."""
        rng = np.random.default_rng(7)
        worlds, bound, width = 500, 100, 32
        rows = rng.integers(0, 140, worlds).astype(np.int32)
        free = (rng.random(worlds) < 0.2).astype(np.int32) * rng.integers(1, 9, worlds).astype(np.int32)
        tiers = -(-bound // width)
        for device in _devices():
            with self.subTest(device=device):
                lists = Queue(tiers * worlds, device, buckets=tiers)
                wp.launch(
                    sweep_levels.list_worlds,
                    dim=worlds,
                    inputs=[
                        wp.array(rows, dtype=int, device=device),
                        wp.array(free, dtype=int, device=device),
                        bound,
                        width,
                        lists.region,
                    ],
                    outputs=[lists.items, lists.count],
                    device=device,
                )
                for tier in range(tiers):
                    count, items = (array.numpy() for array in lists.bucket(tier))
                    lo, hi = tier * width, min((tier + 1) * width, bound)
                    want = np.flatnonzero((rows > lo) & (rows <= hi) & (free == 0))
                    self.assertEqual(int(count[0]), want.size)
                    np.testing.assert_array_equal(np.sort(items[: count[0]]), want)

    def test_a_bucket_is_a_list_of_its_own(self):
        """A bucket's views are its counter and its region of the slots."""
        lists = Queue(12, "cpu", buckets=3)
        self.assertEqual(lists.region, 4)
        count, items = lists.bucket(2)
        self.assertEqual((count.shape, items.shape), ((1,), (4,)))
        self.assertEqual(items.ptr, lists.items.ptr + 8 * 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

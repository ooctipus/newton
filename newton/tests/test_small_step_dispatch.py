# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check geometry/limit admission without conflating it with solved impulses."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.small_step_dispatch import classify


class TestSmallStepDispatchCPU(unittest.TestCase):
    def test_limits_and_contact_boundaries(self):
        """Retain every fallback row across empty and 32-to-33 transitions."""
        counts = np.array([0, 7, 32, 33, 9, 32], dtype=np.int32)
        types = np.full((6, 100), 3, dtype=np.int32)
        for world, contacts in enumerate((0, 0, 8, 8, 9, 8)):
            types[world, :contacts] = 0
        selected = wp.full(6, -1, dtype=int, device="cpu")
        fallback = wp.full(6, -1, dtype=int, device="cpu")
        count_array = wp.array(counts, dtype=int, device="cpu")
        type_array = wp.array(types, dtype=int, device="cpu")
        for expected in ([1, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 1]):
            wp.launch(classify, dim=6, inputs=[count_array, type_array, selected, fallback], device="cpu")
            np.testing.assert_array_equal(selected.numpy(), expected)
            np.testing.assert_array_equal(fallback.numpy(), counts * (1 - np.asarray(expected)))
            counts[[2, 3]] = counts[[3, 2]]
            count_array.assign(counts)


if __name__ == "__main__":
    unittest.main()

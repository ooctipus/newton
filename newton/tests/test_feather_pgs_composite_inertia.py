# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.spatial_laws import INERTIA_CHANNELS, inertia_record
from newton._src.solvers.feather_pgs.tree_passes import composite_inertia_lanes


class TestFeatherPGSCompositeInertia(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "warp-parallel composite inertia requires CUDA")
    def test_warp_reduction_sums_each_subtree_in_serial_order(self):
        """A warp per articulation, lanes over the 13 record channels: each composite is its body's record plus
        its children's composites, added leaf to root as the serial pass adds them (bit for bit)."""
        device = wp.get_cuda_device()
        starts = np.array([0, 4, 7], dtype=np.int32)
        ancestor = np.array([-1, 0, 0, 2, -1, 4, 5], dtype=np.int32)
        records = np.random.default_rng(42).normal(size=(7, INERTIA_CHANNELS)).astype(np.float32)
        expected = records.copy()
        for art in range(2):
            for joint in range(starts[art + 1] - 1, starts[art] - 1, -1):
                if ancestor[joint] >= starts[art]:
                    expected[ancestor[joint]] += expected[joint]

        composite = wp.zeros(7, dtype=inertia_record, device=device)
        warps_per_block = 4
        wp.launch_tiled(
            composite_inertia_lanes(warps_per_block),
            dim=[1],
            inputs=[
                2,
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array(starts, dtype=wp.int32, device=device),
                wp.array(starts[1:], dtype=wp.int32, device=device),
                wp.array(ancestor, dtype=wp.int32, device=device),
                wp.array(np.arange(7, dtype=np.int32), dtype=wp.int32, device=device),
                wp.array(records, dtype=inertia_record, device=device),
                composite,
            ],
            block_dim=32 * warps_per_block,
            device=device,
        )
        np.testing.assert_array_equal(composite.numpy(), expected)


if __name__ == "__main__":
    unittest.main()

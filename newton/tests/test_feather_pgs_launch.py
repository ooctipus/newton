# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The launch primitive: grids from items, the thread mapping and the SM count."""

import types
import unittest
from unittest import mock

from newton._src.solvers.feather_pgs import launch as launch_module
from newton._src.solvers.feather_pgs.launch import Mapping, grid


class TestLaunchGrid(unittest.TestCase):
    def test_direct_launch_covers_every_item(self):
        self.assertEqual(grid(0, Mapping()), 0)
        self.assertEqual(grid(1, Mapping(lanes=32, items_per_block=2)), 1)
        self.assertEqual(grid(129, Mapping(lanes=16, items_per_block=4)), 33)
        self.assertEqual(Mapping(lanes=16, items_per_block=4).block_dim, 64)

    def test_persistent_launch_caps_at_a_resident_wave(self):
        device = types.SimpleNamespace(is_cuda=True, sm_count=188)
        with mock.patch.object(launch_module.wp, "get_device", return_value=device):
            mapping = Mapping(lanes=32, persistent=True, blocks_per_sm=8)
            self.assertEqual(grid(16384, mapping), 8 * 188)
            self.assertEqual(grid(100, mapping), 100)
            self.assertEqual(grid(1, mapping), 1)

    def test_persistent_launch_on_the_cpu_counts_one_sm(self):
        device = types.SimpleNamespace(is_cuda=False)
        with mock.patch.object(launch_module.wp, "get_device", return_value=device):
            self.assertEqual(grid(100, Mapping(persistent=True, blocks_per_sm=4)), 4)

    def test_mapping_rejects_empty_shapes(self):
        with self.assertRaises(ValueError):
            Mapping(lanes=0)
        with self.assertRaises(ValueError):
            Mapping(items_per_block=0)


if __name__ == "__main__":
    unittest.main()

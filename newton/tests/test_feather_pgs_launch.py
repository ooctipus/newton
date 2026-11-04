# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The execution layer (:mod:`newton._src.core.execution`): one shape rule against today's launch rules.

The table is ``SP/core/FL/HANDOFF_ZEX.md`` §3.2's, on the suite's two devices: the rule reproduces the walk's group,
the held lanes, the support walkers' grid, the launch bounds and H8's and ZY's carveouts, and differs only where ZEX
names it (the fill at 512-4,096 worlds, the writer's lanes at 16,384 worlds, the predictor's bound on sm_103).
"""

import types
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.core import execution, memory
from newton._src.core.execution import Queue, Use, Work, shape
from newton._src.core.memory import CPU, Device, carveout
from newton._src.solvers.feather_pgs import tree_kinetic_state, walk_rows
from newton._src.solvers.feather_pgs.factor import walk as factor_walk

PRO = Device.of(120, 188, 101_376)
GB300 = Device.of(103, 152, 232_448)
WORLDS = (64, 128, 512, 1024, 4096, 16384, 65536)


def _warp_device(device: Device):
    """A stand-in Warp device of ``device`` for today's rules, which read ``is_cuda`` and ``sm_count``."""
    return types.SimpleNamespace(is_cuda=device.cuda, sm_count=device.sms)


class TestDevice(unittest.TestCase):
    def test_suite_devices(self):
        self.assertEqual((PRO.threads_per_sm, PRO.blocks_per_sm, PRO.shared_per_block), (1536, 24, 100_352))
        self.assertEqual((GB300.threads_per_sm, GB300.blocks_per_sm, GB300.shared_per_block), (2048, 32, 231_424))
        # A multiprocessor's shared memory is the opt-in maximum a block and the driver's reservation.
        self.assertEqual(PRO.shared_per_sm, 101_376 + memory.RESERVED_SHARED_BYTES)
        self.assertEqual(GB300.shared_per_sm, 232_448 + memory.RESERVED_SHARED_BYTES)

    def test_unknown_architecture_is_refused(self):
        with self.assertRaises(ValueError):
            Device.of(1, 10, 49_152)

    def test_cpu_reads_as_the_cpu(self):
        self.assertIs(memory.device("cpu"), CPU)
        self.assertFalse(CPU.cuda)

    def test_budgets_by_architecture(self):
        self.assertEqual(memory.arch_budget(0), memory.CPU_STAGING_BYTES)
        self.assertEqual(memory.arch_budget("offline"), memory.STATIC_SHARED_BYTES)
        self.assertEqual(execution.device_lanes(0), 1)
        self.assertEqual(execution.device_lanes("sm_120", 16), 16)
        # Offline (no device of the architecture): the smallest suite GPU's multiprocessor.
        if not any(found.arch == 12345 for found in wp.get_cuda_devices()):
            self.assertEqual(memory.resident_blocks(1664, 12345), memory.SM_SHARED_BYTES // 2688)


class TestShapeAgainstTodaysRules(unittest.TestCase):
    """ZEX §3.2's table, each row on both devices."""

    def test_walk_group_is_a_feed_at_its_line(self):
        for device in (PRO, GB300):
            for trees in (1024, 4096, 16384, 65536):
                for bodies, joins in ((30, 2), (13, 1), (106, 1), (2, 1), (8, 4)):
                    with self.subTest(arch=device.arch, trees=trees, bodies=bodies, joins=joins):
                        old = factor_walk.walk_group(bodies, joins, trees, device.sms)
                        least = 1 << max(joins - 1, 0).bit_length()
                        work = Work(trees, Use.FEED, width=bodies, least=least, registers=128)
                        self.assertEqual(shape(work, device).lanes, old)

    def test_held_lanes_are_a_scan(self):
        for device in (PRO, GB300):
            for bodies in (1, 3, 8, 13, 20, 30):
                with self.subTest(arch=device.arch, bodies=bodies):
                    old = tree_kinetic_state.held_lanes(types.SimpleNamespace(count=bodies), 16384)
                    self.assertEqual(shape(Work(16384, Use.SCAN, width=bodies, least=8), device).lanes, old)

    def test_support_walkers_are_two_waves_of_their_line(self):
        for device in (PRO, GB300):
            with self.subTest(arch=device.arch), mock.patch.object(wp, "get_device", return_value=_warp_device(device)):
                work = Work(10**7, lanes=1, block=256, registers=128, persistent=True, waves=2)
                got = shape(work, device)
                self.assertEqual(got.grid * got.block, walk_rows.support_threads(10**7, None))
                self.assertEqual(got.bounds, (walk_rows.SUPPORT_BLOCK, walk_rows.SUPPORT_BLOCKS))
                self.assertEqual(got.registers, walk_rows.SUPPORT_REGISTERS)

    def test_bounds_follow_the_register_line(self):
        cases = (
            ("writer", 128, 80, (128, walk_rows.WRITER_BLOCKS[0])),
            ("writer with blocks", 128, 96, (128, walk_rows.WRITER_BLOCKS[1])),
            ("support", 256, 128, (256, 2)),
            ("colored", 256, 255, (256, 1)),
            ("predictor 32", 32, 56, (32, min(24, 1152 // 32))),
            ("predictor 64", 64, 56, (64, min(24, 1152 // 64))),
        )
        for device in (PRO, GB300):
            for name, block, registers, old in cases:
                with self.subTest(arch=device.arch, kernel=name):
                    work = Work(10**6, lanes=min(block, 32), block=block, registers=registers)
                    bounds = shape(work, device).bounds
                    if device is GB300 and name == "predictor 32":
                        # ZP7's case: sm_103 keeps 32 blocks a multiprocessor, today's bound copies sm_120's 24.
                        self.assertEqual(bounds, (32, 32))
                    else:
                        self.assertEqual(bounds, old)

    def test_carveout_steps(self):
        # H8: a predictor block over 1,664 B moves the PRO from 64 KiB to 100 KiB; ZY: 1 KiB at 256 threads keeps 16.
        for block, shared, step in (
            (32, 1664, 64),
            (32, 2200, 100),
            (64, 1664, 64),
            (256, 1024, 16),
            (256, 28672, 100),
        ):
            with self.subTest(block=block, shared=shared):
                self.assertEqual(carveout(PRO, block, shared), step)
        self.assertEqual(carveout(GB300, 32, 1664), 100)  # 32 blocks of 2,688 B: 84 KiB
        self.assertEqual(carveout(GB300, 256, 28672), 228)
        got = shape(Work(4096, lanes=32, block=32, shared=1664), PRO)
        self.assertEqual((got.carveout_kb, got.l1_kb), (64, 64))

    def test_fill_differs_only_where_named(self):
        # The rule fills the device's resident threads (1,536 or 2,048 a multiprocessor), fill_lanes 1,024.
        for device in (PRO, GB300):
            with mock.patch.object(wp, "get_device", return_value=_warp_device(device)):
                for worlds in WORLDS:
                    with self.subTest(arch=device.arch, worlds=worlds):
                        old = execution.fill_lanes(worlds, None)
                        new = shape(Work(worlds, Use.SPLIT, width=1 << 20, least=32, most=1024), device).lanes
                        self.assertEqual(new, 2 * old if worlds in (512, 1024, 4096) else old)

    def test_writer_lanes_differ_only_where_named(self):
        named = {(120, 20): 16, (120, 40): 16, (103, 12): 8, (103, 20): 8, (103, 40): 8}
        for device in (PRO, GB300):
            with mock.patch.object(wp, "get_device", return_value=_warp_device(device)):
                for worlds in (1024, 4096, 16384):
                    for rows in (2, 6, 12, 20, 40):
                        with self.subTest(arch=device.arch, worlds=worlds, rows=rows):
                            old = walk_rows.writer_lanes(rows, 256, None)
                            new = shape(Work(worlds, Use.SPLIT, width=rows, block=128, registers=80), device).lanes
                            expected = named.get((device.arch, rows), old) if worlds == 16384 else old
                            self.assertEqual(new, expected)


class TestShape(unittest.TestCase):
    def test_cpu_runs_an_item_a_thread(self):
        got = shape(Work(100, Use.SPLIT, width=64, least=32, block=128, registers=80), CPU)
        self.assertEqual((got.lanes, got.block, got.grid, got.bounds), (1, 1, 100, None))

    def test_grid_covers_the_items(self):
        got = shape(Work(129, lanes=16, block=64), PRO)
        self.assertEqual((got.groups, got.grid), (4, 33))
        self.assertEqual(shape(Work(0, lanes=32), PRO).grid, 0)

    def test_persistent_grid_caps_at_waves(self):
        self.assertEqual(shape(Work(10**6, lanes=32, block=256, persistent=True), PRO).grid, 6 * 188)
        self.assertEqual(shape(Work(10**6, lanes=32, block=256, persistent=True, grid_per_sm=8), PRO).grid, 8 * 188)
        self.assertEqual(shape(Work(100, lanes=32, block=256, persistent=True), PRO).grid, 13)

    def test_work_rejects_ragged_lanes(self):
        for bad in ({"least": 3}, {"most": 48}, {"lanes": 0}, {"least": 64, "most": 32}, {"registers": 300}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                Work(1, **bad)

    def test_queue_work_is_sized_at_its_capacity(self):
        queue = Queue(1000, "cpu", kinds=2, buckets=16)
        self.assertEqual(queue.count.shape, (32,))
        self.assertEqual(shape(Work(queue, lanes=1, block=256, persistent=True), PRO).grid, 4)


@wp.kernel
def _mark(out: wp.array[int]):
    out[wp.tid()] = 1


class TestLaunch(unittest.TestCase):
    def test_plain_launch_runs_the_grid(self):
        out = wp.zeros(8, dtype=int, device="cpu")
        execution.launch(_mark, shape(Work(8), CPU), outputs=(out,), device="cpu", tiled=False)
        np.testing.assert_array_equal(out.numpy(), np.ones(8, dtype=int))


if __name__ == "__main__":
    unittest.main()

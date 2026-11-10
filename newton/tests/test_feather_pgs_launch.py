# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The execution layer (:mod:`newton._src.core.execution`): one shape rule against today's launch rules.

The table is ``SP/core/FL/HANDOFF_ZEX.md`` §3.2's, on the suite's two devices: the rule reproduces the walk's group,
the held lanes, the support walkers' grid, the launch bounds and H8's and ZY's carveouts, and differs only where ZEX
names it (the fill at 512-4,096 worlds, the writer's lanes at 16,384 worlds, the predictor's bound on sm_103).
"""

import dataclasses
import types
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.core import execution, memory
from newton._src.core.execution import Queue, Use, Work, shape
from newton._src.core.memory import CPU, Device, Place, carveout
from newton._src.solvers.feather_pgs import sweep_routes, tree_kinetic_state, walk_rows
from newton._src.solvers.feather_pgs.factor import walk as factor_walk
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout

PRO = Device.of(120, 188, 101_376)
GB300 = Device.of(103, 152, 232_448)
WORLDS = (64, 128, 512, 1024, 4096, 16384, 65536)


def _current_device(device: Device):
    """Warp's current device as a stand-in of ``device`` for today's rules, which read ``is_cuda``, ``sm_count`` and
    ``arch``. A device asked for by name stays Warp's own."""
    stand_in = types.SimpleNamespace(is_cuda=device.cuda, sm_count=device.sms, arch=device.arch)
    get_device = wp.get_device

    def current(ident=None):
        return stand_in if ident is None else get_device(ident)

    return mock.patch.object(wp, "get_device", side_effect=current)


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
        # A factory builds for the device of its architecture: its table offline, sm_120's when it names none.
        self.assertIs(memory.built_for(0), CPU)
        if not any(found.arch == 103 for found in wp.get_cuda_devices()):
            self.assertEqual(memory.built_for("sm_103").shared_per_sm, 228 * 1024)
        if not wp.get_cuda_devices():
            self.assertEqual(memory.built_for(None).arch, memory.REFERENCE_ARCH)


class TestShapeAgainstTodaysRules(unittest.TestCase):
    """ZEX §3.2's table, each row on both devices."""

    def test_walk_group_is_a_feed_at_its_line(self):
        # The walk's group as it was written before it declared its work (ZR's rule): the largest power of two keeping
        # the trees in one wave of 512 threads an SM, within the tree and a warp, never fewer lanes than its joins.
        def walk_group(bodies, joins, trees, sms):
            fit, lanes = max(sms * 512 // max(trees, 1), 1), 1
            while lanes * 2 <= fit and lanes < min(bodies, 32):
                lanes *= 2
            while lanes < joins:
                lanes *= 2
            return lanes

        for device in (PRO, GB300):
            for trees in (1024, 4096, 16384, 65536):
                for bodies, joins in ((30, 2), (13, 1), (106, 1), (2, 1), (8, 4)):
                    with self.subTest(arch=device.arch, trees=trees, bodies=bodies, joins=joins):
                        got = shape(factor_walk.walk_work(trees, bodies, joins), device)
                        self.assertEqual((got.lanes, got.block), (walk_group(bodies, joins, trees, device.sms), 64))

    def test_held_lanes_are_a_scan(self):
        def tree(bodies, dofs):
            return types.SimpleNamespace(count=bodies, dof_start=np.array([0, dofs]))

        for device in (PRO, GB300):
            for bodies, groups, lanes in ((1, 16384, 8), (3, 16384, 8), (13, 16384, 16), (30, 16384, 32), (40, 3, 64)):
                with self.subTest(arch=device.arch, bodies=bodies):
                    self.assertEqual(tree_kinetic_state.held_lanes(tree(bodies, bodies), groups, device), lanes)
            # A one-body rest tree (no coordinates) takes one lane; a two-body one scans its weld at eight.
            self.assertEqual(tree_kinetic_state.held_lanes(tree(1, 0), 16384, device), 1)
            self.assertEqual(tree_kinetic_state.held_lanes(tree(2, 0), 16384, device), 8)
            # Whole 32-thread blocks: three one-body trees take a warp each.
            self.assertEqual(tree_kinetic_state.held_lanes(tree(1, 1), 3, device), 32)
            self.assertEqual(tree_kinetic_state.held_lanes(tree(1, 0), 3, device), 32)
        self.assertEqual(tree_kinetic_state.held_lanes(tree(1, 1), 3, CPU), 32)

    def test_support_walkers_are_two_waves_of_their_line(self):
        for device in (PRO, GB300):
            with self.subTest(arch=device.arch), _current_device(device):
                # The line its four-row walker holds without a spill (164, in whole granules of 8).
                work = Work(10**7, lanes=1, block=128, registers=168, persistent=True, waves=2)
                got = shape(work, device)
                self.assertEqual(got.grid * got.block, walk_rows.support_threads(10**7, None))
                self.assertEqual(got.bounds, walk_rows.WRITERS["support"].bounds(device.arch))
                self.assertEqual(got.registers, walk_rows.WRITERS["support"].registers(device.arch))

    def test_bounds_follow_the_register_line(self):
        cases = (
            # The one walker's line: the general writer's six blocks of 128, the star's three of 256.
            ("writer", 128, walk_rows.WALKER_LINE, (128, 6)),
            ("star", 256, walk_rows.WALKER_LINE, (256, 3)),
            ("writer with blocks", 128, 96, (128, 5)),
            ("support", 256, 128, (256, 2)),
            ("colored", 256, 255, (256, 1)),
            ("predictor 32", 32, 56, (32, min(24, 1152 // 32))),
            ("predictor 64", 64, 56, (64, min(24, 1152 // 64))),
            # The sweeps' lines: whole warps a register sub-partition (6 at 80 registers, 8 at 64).
            ("tier of a warp", 32, 80, (32, 24)),
            ("five warps at the line", 160, 80, (160, 4)),
            ("sparse", 64, 64, (64, 16)),
        )
        for device in (PRO, GB300):
            for name, block, registers, old in cases:
                with self.subTest(arch=device.arch, kernel=name):
                    work = Work(10**6, lanes=min(block, 32), block=block, registers=registers)
                    bounds = shape(work, device).bounds
                    if device is GB300 and name == "predictor 32":
                        # ZP7's case: sm_103 keeps 32 blocks a multiprocessor, the predictor built for it (B3).
                        self.assertEqual(bounds, (32, 32))
                    else:
                        self.assertEqual(bounds, old)
        # The writers' bounds, built for each device: the measured lines, which bind on both.
        for device in (PRO, GB300):
            bounds = {name: writer.bounds(device.arch) for name, writer in walk_rows.WRITERS.items()}
            self.assertEqual(
                bounds,
                {
                    "packed": (128, 6),
                    "packed_blocks": (128, 5),
                    "slots": (256, 3),
                    "support_units": (256, 4),
                    "support": (128, 3),
                    "free_body": (256, 2),
                },
            )
            self.assertEqual(walk_rows.WRITERS["support"].registers(device.arch), 170)

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

    def test_stage_pays_in_warps(self):
        """franka-cabinet's general sweep streams its records (6 and 14 warps staged, 32 streamed), g1's stages them at
        16 warps, and a grid of a block an SM and the CPU stage everything: the warps rule (memory.stage)."""
        cabinet = sweep_routes.GeneralSweep(
            256, 32, 13, "120", max_row_dofs=13, free_rows=False, lanes_per_world=16, block_warps=2, torsion=True
        )
        g1 = sweep_routes.GeneralSweep(
            192, 32, 49, "120", max_row_dofs=29, free_rows=False, block_warps=2, support_classes=True, active_rows=True
        )

        def placed(config, device, worlds=16384):
            return config.places.placed(shape(config.work(worlds, config.staging()), device).staging)

        for device in (PRO, GB300):
            self.assertIs(placed(cabinet, device).metadata, Place.STREAMED)
        for config, device, worlds in ((g1, PRO, 16384), (cabinet, PRO, 128), (cabinet, CPU, 16384)):
            self.assertEqual(placed(config, device, worlds), sweep_routes.STAGED_ROWS)

    def test_a_bucket_beside_the_tiers_streams_its_record(self):
        """allegro's record stages over every world on the GB300 and streams beside the tiers, sure of no live world."""
        allegro = sweep_routes.GeneralSweep(
            258,
            30,
            22,
            "103",
            pitch=288,
            max_row_dofs=22,
            free_rows=False,
            block_warps=2,
            active_rows=True,
            torsion=True,
        )

        def placed(device, live):
            work = dataclasses.replace(allegro.work(16384, allegro.staging()), live=live)
            return allegro.places.placed(shape(work, device).staging).metadata

        expected = {GB300: (Place.SHARED, Place.STREAMED), PRO: (Place.STREAMED,) * 2, CPU: (Place.SHARED,) * 2}
        for device, places in expected.items():
            self.assertEqual((placed(device, None), placed(device, 0)), places)

    def test_the_held_tree_stages_at_its_own_level(self):
        """kuka's held forest keeps its state staged on the PRO at 14 of 24 warps, which the general level declines."""
        roots = {"roots": (0, 30, 31), "root_coordinates": (0, 6, 6), "root_offsets": (0, 23, 29)}
        layout = TreeLayout(32, 35, 35, 1, 0, 0, tuple(range(35)), lanes=32, root_families=(35, 0, 0), **roots)
        self.assertTrue(tree_kinetic_state.held_stage(layout, 16384, PRO).staged)
        with mock.patch.object(tree_kinetic_state, "HELD_HIDING", memory.HIDING_WARPS):
            self.assertFalse(tree_kinetic_state.held_stage(layout, 16384, PRO).staged)
        self.assertTrue(tree_kinetic_state.held_stage(layout, 16384, GB300).staged)

    def test_fill_differs_only_where_named(self):
        # The rule fills the device's resident threads (1,536 or 2,048 a multiprocessor), fill_lanes 1,024.
        for device in (PRO, GB300):
            with _current_device(device):
                for worlds in WORLDS:
                    with self.subTest(arch=device.arch, worlds=worlds):
                        old = execution.fill_lanes(worlds, None)
                        new = shape(Work(worlds, Use.SPLIT, width=1 << 20, least=32, most=1024), device).lanes
                        self.assertEqual(new, 2 * old if worlds in (512, 1024, 4096) else old)

    def test_writer_lanes_differ_only_where_named(self):
        named = {(120, 20): 16, (120, 40): 16, (103, 12): 8, (103, 20): 8, (103, 40): 8}
        for device in (PRO, GB300):
            with _current_device(device):
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

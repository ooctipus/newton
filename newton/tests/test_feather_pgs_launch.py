# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The execution layer (:mod:`newton._src.core.execution`): one shape rule against today's launch rules.

The table is ``SP/core/FL/HANDOFF_ZEX.md`` §3.2's, on the suite's two devices: the rule reproduces the walk's group,
the held lanes, the support walkers' grid, the launch bounds and H8's and ZY's carveouts, and differs only where ZEX
names it (the fill at 512-4,096 worlds, the writer's lanes at 16,384 worlds, the predictor's bound on sm_103).
"""

import ctypes
import dataclasses
import types
import unittest
from typing import Any
from unittest import mock

import numpy as np
import warp as wp

from newton._src.core import execution, memory
from newton._src.core.execution import Queue, Use, Work, shape
from newton._src.core.memory import CPU, Buffer, Device, Place, Staging, carveout, carveout_share
from newton._src.solvers.feather_pgs import kernel_template, sweep_routes, tree_kinetic_state, walk_rows
from newton._src.solvers.feather_pgs.device_preludes import SHARED_STORAGE, DevicePrelude, dual_native, lane_group
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
        # A stage takes the opt-in maximum a block of its architecture, offline as on the device.
        self.assertEqual(memory.stage_budget(120), PRO.shared_per_block)
        self.assertEqual(memory.stage_budget(103), GB300.shared_per_block)


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
            for bodies, groups, lanes in ((3, 16384, 8), (13, 16384, 16), (30, 16384, 32), (40, 3, 64)):
                with self.subTest(arch=device.arch, bodies=bodies):
                    self.assertEqual(tree_kinetic_state.held_lanes(tree(bodies, bodies), groups, device), lanes)
            # A one-body tree fills one wave of its finish's residency, up to its coordinates: a free body at eight
            # lanes where its trees are few, at four for one a world of 16,384, at one for NIST's 81,920 (10 a world
            # of 8,192); a rest tree (no coordinates) and a one-coordinate body at one. A two-body rest tree scans its
            # weld at eight.
            for groups, lanes in ((1344, 8), (16384, 4), (81920, 1)):
                with self.subTest(arch=device.arch, groups=groups):
                    self.assertEqual(tree_kinetic_state.held_lanes(tree(1, 6), groups, device), lanes)
            self.assertEqual(tree_kinetic_state.held_lanes(tree(1, 0), 16384, device), 1)
            self.assertEqual(tree_kinetic_state.held_lanes(tree(1, 1), 16384, device), 1)
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
        # The smallest step holding the resident blocks: on the PRO 24 blocks of 1,664 B keep 64 KiB and of 2,200 B
        # take 100 KiB (H8, the thread and block limits' 24); the general sweep's 8 register-bound blocks of 4,776 B
        # keep 64 KiB, where the driver's own choice (24 of them) takes 100 KiB and leaves L1 28 KiB.
        for resident, shared, step in ((24, 1664, 64), (24, 2200, 100), (8, 4776, 64), (16, 1024, 32), (3, 28672, 100)):
            with self.subTest(resident=resident, shared=shared):
                self.assertEqual(carveout(PRO, resident, shared), step)
        self.assertEqual(carveout(GB300, 32, 1664), 100)  # 32 blocks of 2,688 B: 84 KiB
        self.assertEqual(carveout(GB300, 8, 4776), 64)
        self.assertEqual(carveout(GB300, 8, 28672), 228)
        self.assertEqual(carveout(CPU, 8, 4776), 0)
        got = shape(Work(4096, lanes=32, block=32, shared=1664), PRO)
        self.assertEqual((got.blocks_per_sm, got.carveout_kb, got.l1_kb), (24, 64, 64))

    def test_a_share_lands_on_its_step(self):
        # The driver reads a preferred carveout as a share of the unified memory and takes the next step up: each
        # step's share lands on it (the PRO's 64 KiB step is 50%; the share of its largest step, 64%, is 82 KiB of the
        # unified 128 and lands on 100 KiB, the driver's default, which leaves the general sweep 28 KiB of L1).
        for device in (PRO, GB300):
            for step in device.carveouts_kb:
                with self.subTest(arch=device.arch, step=step):
                    asked = carveout_share(device, step) * device.unified_kb / 100
                    self.assertEqual(next(s for s in device.carveouts_kb if s >= asked), step)
        self.assertEqual(carveout_share(PRO, 64), 50)
        self.assertEqual(carveout_share(CPU, 64), 0)

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
        """The held tree's state block holds its outputs on its inputs' words (19 words a body, 6 a coordinate, 6 a
        root), so kuka's held forest keeps 22 of the PRO's 24 blocks in flight a multiprocessor; a 128-body chain's
        block leaves 14 warps, which the held tree's level stages and the general level declines."""
        roots = {"roots": (0, 30, 31), "root_coordinates": (0, 6, 6), "root_offsets": (0, 23, 29)}
        layout = TreeLayout(32, 35, 35, 1, 0, 0, tuple(range(35)), lanes=32, root_families=(35, 0, 0), **roots)
        self.assertEqual(layout.total, 19 * 32 + 6 * 35 + 6 * 3)
        self.assertTrue(tree_kinetic_state.held_stage(layout, 16384, PRO).staged)
        self.assertEqual(shape(Work(16384, lanes=32, block=32, shared=4 * layout.total), PRO).blocks_per_sm, 22)
        chain = TreeLayout(128, 128, 128, 1, 127, 0, tuple(range(128)), lanes=64)
        self.assertEqual(shape(Work(16384, lanes=64, block=64, shared=4 * chain.total), PRO).blocks_per_sm, 7)
        self.assertTrue(tree_kinetic_state.held_stage(chain, 16384, PRO).staged)
        with mock.patch.object(tree_kinetic_state, "HELD_HIDING", memory.HIDING_WARPS):
            self.assertFalse(tree_kinetic_state.held_stage(chain, 16384, PRO).staged)
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


@wp.kernel
def _copy_any(source: wp.array[Any], target: wp.array[Any]):
    # A plain grid is whole blocks (grid * block threads): the kernel bounds its items, as the solver's do.
    i = wp.tid()
    if i < target.shape[0]:
        target[i] = source[i]


_STAGE_WORDS = 16 * 1024
"""A 64 KiB stage: past the static limit, so it takes dynamic shared memory on CUDA."""


def _stage_kernel(device):
    """A block group's kernel whose stage of :data:`_STAGE_WORDS` floats each lane fills and the block reads back
    reversed, its sum a block in ``out``; and its staging."""
    arch = wp.get_device(device).arch if wp.get_device(device).is_cuda else 0
    staging = Staging((Buffer("s_words", "float", _STAGE_WORDS), Buffer("s_sum", "float", 1)), threads=128)
    body = f"""{staging.declarations()}
    for (int i = LANE; i < {_STAGE_WORDS}; i += STRIDE) s_words[i] = (float)(i % 7 + block);
    if (LANE == 0) s_sum[0] = 0.0f;
    SYNC();
    float sum = 0.0f;
    for (int i = LANE; i < {_STAGE_WORDS}; i += STRIDE) sum += s_words[{_STAGE_WORDS} - 1 - i];
    SUM(sum);
    if (LANE % WIDTH == 0) ADD(s_sum[0], sum);
    SYNC();
    if (LANE == 0) out.data[block] = s_sum[0];
"""
    preludes = (lane_group(128), SHARED_STORAGE, DevicePrelude("add", *memory.words("ADD")))
    kernel = kernel_template.KernelTemplate(
        f"stage_kernel_{arch}",
        (("out", wp.array[float]),),
        dual_native(body, preludes=preludes),
        kernel_template.items("block"),
        composed=True,
        stage=staging.dynamic_bytes(arch),
    ).kernel()
    return kernel, staging


class TestLaunch(unittest.TestCase):
    def test_plain_launch_runs_the_grid(self):
        out = wp.zeros(8, dtype=int, device="cpu")
        execution.launch(_mark, shape(Work(8), CPU), outputs=(out,), device="cpu", tiled=False)
        np.testing.assert_array_equal(out.numpy(), np.ones(8, dtype=int))

    def test_a_stage_past_the_static_limit_is_dynamic(self):
        small = Staging((Buffer("a", "float", 1024),), threads=32)
        big = Staging((Buffer("a", "float", _STAGE_WORDS), Buffer("b", "int", 16)), threads=32)
        self.assertFalse(small.dynamic)
        self.assertTrue(big.dynamic)
        self.assertEqual(big.dynamic_bytes("sm_120"), 4 * _STAGE_WORDS + 64)
        self.assertEqual((big.dynamic_bytes(0), small.dynamic_bytes(120)), (0, 0))
        self.assertIn("STAGE + 65536", big.declarations())

    def test_a_dynamic_stage_launches(self):
        # A block's 64 KiB stage: dynamic shared memory on CUDA (the launch opts in past 48 KiB), the stack on the CPU.
        expected = [sum(float(i % 7 + block) for i in range(_STAGE_WORDS)) for block in range(3)]
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=str(device)):
                kernel, staging = _stage_kernel(device)
                cuda = wp.get_device(device).is_cuda
                self.assertEqual(staging.dynamic_bytes(wp.get_device(device).arch if cuda else 0) > 0, cuda)
                out = wp.zeros(3, dtype=float, device=device)
                wp.launch_tiled(kernel, dim=[3], inputs=[out], block_dim=128 if cuda else 1, device=device)
                np.testing.assert_allclose(out.numpy(), expected)

    def test_a_kernel_prefers_the_step_of_its_resident_blocks(self):
        # The driver's occupancy of the loaded code sets the step (memory.fit_carveout); the function keeps the
        # preference, once, and the launch runs under it.
        expected = [sum(float(i % 7 + block) for i in range(_STAGE_WORDS)) for block in range(3)]
        for device in wp.get_cuda_devices():
            with self.subTest(device=str(device)):
                kernel, staging = _stage_kernel(device)
                spec = memory.device(device)
                step = memory.fit_carveout(kernel, 128, device)
                self.assertIn(step, spec.carveouts_kb)
                self.assertGreaterEqual(step * 1024, memory.granules(staging.dynamic_bytes(spec.arch)))
                self.assertEqual(memory.fit_carveout(kernel, 128, device), step)
                hooks = kernel.module.load(device, 128).get_kernel_hooks(kernel)
                share = ctypes.c_int(-2)
                memory._call("cuFuncGetAttribute", ctypes.byref(share), ctypes.c_int(9), ctypes.c_void_p(hooks.forward))
                self.assertEqual(share.value, memory.carveout_share(spec, step))
                out = wp.zeros(3, dtype=float, device=device)
                wp.launch_tiled(kernel, dim=[3], inputs=[out], block_dim=128, device=device)
                np.testing.assert_allclose(out.numpy(), expected)
        self.assertEqual(memory.fit_carveout(_mark, 32, "cpu"), 0)

    def test_a_generic_kernel_fits_the_overload_it_launches(self):
        # A generic kernel has no code of its own: the layer's launch fits the overload its arguments select. The
        # plain grid runs whole blocks (on CUDA one block of 256 threads for the 8 items): the source and target are
        # the heads of buffers as long as the grid, so a thread past the items would read and write inside them, and
        # the words past the target keep their marks.
        for device in ("cpu", *wp.get_cuda_devices()):
            with self.subTest(device=str(device)):
                grid = shape(Work(8), memory.device(device))
                threads = grid.grid * grid.block
                words = np.arange(2 * threads, dtype=np.float32).reshape(threads, 2)
                source_words = wp.array(words, dtype=wp.vec2, device=device)
                target_words = wp.full(threads, wp.vec2(-1.0, -1.0), dtype=wp.vec2, device=device)
                execution.launch(
                    _copy_any,
                    grid,
                    inputs=(source_words[:8],),
                    outputs=(target_words[:8],),
                    device=device,
                    tiled=False,
                )
                np.testing.assert_array_equal(target_words.numpy()[:8], words[:8])
                np.testing.assert_array_equal(target_words.numpy()[8:], -1.0)
                np.testing.assert_array_equal(source_words.numpy(), words)


if __name__ == "__main__":
    unittest.main()

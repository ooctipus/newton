# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A compliant group of regions visited as one block (``region_sweep.form_blocks``), and the blocks' arena.

A plate rests on a grid of small immovable spheres, a block sits on the plate, every shape authors NIST's part
material. With more spheres under the plate than its six coordinates, the plate's regions against them close at the
predictor in excess of its coordinates and the listing makes them one block; with fewer, they stay regions. Either
way the plate comes to rest on its springs.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

PART = (1.6e5, 800.0)
"""NIST's authored part material (ke [N/m], kd [N s/m])."""


def resting_plate(side: int, steps: int = 400):
    """The plate on a ``side`` x ``side`` grid of spheres, stepped on the CPU: the plate's heights and speeds a step,
    and the solver."""
    device = wp.get_device("cpu")
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.75, ke=PART[0], kd=PART[1])
    radius, pitch = 0.004, 0.02
    for i in range(side):
        for j in range(side):
            offset = (np.array([i, j]) - 0.5 * (side - 1)) * pitch
            builder.add_shape_sphere(
                -1,
                xform=wp.transform(wp.vec3(offset[0], offset[1], radius), wp.quat_identity()),
                radius=radius,
                cfg=cfg,
            )
    half = 0.5 * pitch * side
    plate = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * radius + 0.004), wp.quat_identity()))
    builder.add_shape_box(
        plate, hx=half, hy=half, hz=0.004, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, **_material())
    )
    top = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * radius + 0.0125), wp.quat_identity()))
    builder.add_shape_box(
        top, hx=0.01, hy=0.01, hz=0.004, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, **_material())
    )
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(model, pgs_iterations=16, dense_max_constraints=16, mf_max_constraints=128)
    state, next_state, control = model.state(), model.state(), model.control()
    heights, speeds = [], []
    for _ in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / 800.0)
        state, next_state = next_state, state
        heights.append(float(state.body_q.numpy()[plate, 2]))
        speeds.append(float(np.abs(state.body_qd.numpy()[plate]).max()))
    return np.array(heights), np.array(speeds), solver


def _material():
    return {"mu": 0.75, "ke": PART[0], "kd": PART[1]}


class TestRegionBlock(unittest.TestCase):
    def test_an_over_determined_group_is_one_block_and_rests(self):
        """Nine spheres under the plate (more closing regions than its six coordinates): one block, a quiet rest."""
        heights, speeds, solver = resting_plate(3)
        regions = solver.sweep.regions
        self.assertIsNotNone(regions)
        pairs = regions.blk_pair.numpy()
        self.assertTrue(((pairs[:, 0] >= 0) & (pairs[:, 1] < 0)).any(), "the plate's group against the spheres")
        self.assertLess(speeds[-50:].max(), 1.0e-3)
        self.assertLess(np.ptp(heights[-50:]), 1.0e-5)

    def test_a_group_within_its_coordinates_keeps_its_regions(self):
        """Four spheres under the plate: its regions stay regions; the same rest."""
        heights, speeds, solver = resting_plate(2)
        pairs = solver.sweep.regions.blk_pair.numpy()
        self.assertFalse((pairs[:, 0] >= 0).any())
        self.assertLess(speeds[-50:].max(), 1.0e-3)
        self.assertLess(np.ptp(heights[-50:]), 1.0e-5)


def placed_blocks(device, sizes, arena_words):
    """``form_blocks`` and ``list_blocks`` over worlds of one free body each against immovable partners, world ``w``
    holding ``sizes[w]`` regions of one closing candidate each, in an arena of ``arena_words`` words: each world's
    first word and words (``-1``: no block listed), the listing's words and the record."""
    from newton._src.core.ordered_reserve import OrderedTiles  # noqa: PLC0415
    from newton._src.solvers.feather_pgs import region_sweep as rs  # noqa: PLC0415

    device = wp.get_device(device)
    worlds = len(sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    regions = int(starts[-1])
    lanes = int(rs._WORLD_LANES) if device.is_cuda else 1
    tile_threads = rs._WORLD_BLOCK if device.is_cuda else 1
    tiles = -(-worlds * lanes // tile_threads)
    world_regions = np.stack([starts[:-1], starts[1:]], axis=1).astype(np.int32)
    reg_body = np.stack([np.repeat(np.arange(worlds), sizes), np.full(regions, -1)], axis=1).astype(np.int32)
    reg_cand = np.stack([np.arange(regions), np.arange(regions) + 1], axis=1).astype(np.int32)
    cand = np.zeros((regions, 4), dtype=np.float32)
    cand[:, 2] = -1.0

    def arr(values, dtype):
        return wp.array(values, dtype=dtype, device=device)

    relist = arr([1, 0], wp.int32)
    reg_kind = arr(np.full(regions, int(rs.KIND_SPRINGS)), wp.int32)
    world_regions, reg_body, reg_cand = arr(world_regions, wp.vec2i), arr(reg_body, wp.vec2i), arr(reg_cand, wp.vec2i)
    cand = arr(cand, wp.vec4)
    world_free_start = arr(np.arange(worlds + 1), wp.int32)
    world_free = arr(np.arange(worlds), wp.int32)
    body_level = wp.zeros(worlds, dtype=wp.int32, device=device)
    blk_count = wp.zeros((worlds, 256), dtype=wp.int32, device=device)
    blk_words = wp.zeros(1, dtype=wp.int32, device=device)
    blk_place = wp.zeros(worlds, dtype=wp.vec4i, device=device)
    record = wp.zeros(2, dtype=wp.int32, device=device)
    site = OrderedTiles(tiles, device).site
    wp.launch(
        rs.form_blocks,
        dim=tiles * tile_threads,
        inputs=[
            relist,
            world_regions,
            world_free_start,
            world_free,
            reg_body,
            rs.BLOCK_CLOSING,
            reg_cand,
            cand,
            reg_kind,
            wp.zeros(regions, dtype=wp.float32, device=device),
            tile_threads,
            tiles,
            site,
        ],
        outputs=[body_level, blk_count, blk_words, blk_place, record],
        device=device,
        block_dim=rs._WORLD_BLOCK,
    )
    blocks = worlds * int(rs.PAIR_BLOCKS)
    blk_blob = wp.zeros(blocks, dtype=wp.vec4i, device=device)
    blk_pair = wp.full(blocks, wp.vec2i(-1, -1), dtype=wp.vec2i, device=device)
    wp.launch(
        rs.list_blocks,
        dim=worlds * lanes,
        inputs=[
            relist,
            world_regions,
            world_free_start,
            reg_body,
            reg_cand,
            body_level,
            blk_place,
            blk_words,
            arr([arena_words], wp.int32),
        ],
        outputs=[
            record,
            blk_blob,
            wp.zeros(blocks, dtype=wp.int32, device=device),
            wp.zeros(regions, dtype=wp.vec3i, device=device),
            reg_kind,
            blk_pair,
            wp.zeros(regions, dtype=wp.int32, device=device),
        ],
        device=device,
        block_dim=rs._WORLD_BLOCK,
    )
    blob, pair = blk_blob.numpy()[:: int(rs.PAIR_BLOCKS)], blk_pair.numpy()[:: int(rs.PAIR_BLOCKS)]
    first = np.where(pair[:, 0] >= 0, blob[:, 0], -1)
    return first, int(blk_words.numpy()[0]), record.numpy()


class TestBlockArena(unittest.TestCase):
    """The blocks' staged words are placed in the worlds' order, whatever order the worlds run in, and an arena short of
    a listing leaves its last worlds without blocks and says so."""

    def _check(self, device):
        rng = np.random.default_rng(7)
        # Worlds over several tiles, some with too few regions for a block (at most BLOCK_CLOSING).
        sizes = rng.integers(3, 40, size=301)
        from newton._src.solvers.feather_pgs import region_sweep as rs  # noqa: PLC0415

        words = np.where(sizes > rs.BLOCK_CLOSING, (int(rs.STAGE_UNIT) + int(rs.STAGE_CAND)) * sizes, 0)
        starts = np.concatenate([[0], np.cumsum(words)])
        first, total, record = placed_blocks(device, sizes, int(starts[-1]))
        self.assertEqual(total, int(starts[-1]))
        np.testing.assert_array_equal(first, np.where(words > 0, starts[:-1], -1))
        self.assertEqual(list(record), [total, 0])
        # Short by one word: the worlds whose words end past it list no block, the others the same places.
        first, _total, record = placed_blocks(device, sizes, int(starts[-1]) - 1)
        fits = starts[1:] <= starts[-1] - 1
        np.testing.assert_array_equal(first, np.where((words > 0) & fits, starts[:-1], -1))
        self.assertEqual(list(record), [total, 1])

    def test_cpu(self):
        self._check("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "needs CUDA")
    def test_cuda(self):
        self._check("cuda:0")


if __name__ == "__main__":
    unittest.main()

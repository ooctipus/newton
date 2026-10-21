# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's kernels for convex hulls and ellipsoids (rules F, E, V and R of the law for solid cores).

The kernels take the per-type arguments ``(begin, slot_begin, inputs, data, out)``
(:class:`~newton._src.geometry.terrain_law.TerrainPairInputs`), their launch argument and the solid classes' data
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`), on the shadow's shared launches
(:data:`~newton._src.geometry.terrain_shadow.TERRAIN_LAW_KERNELS`): :func:`terrain_hull_rows` runs one thread per pair
over the whole class and queues the tiled-lane pairs whose windows (as the pose gives them) hold more than
:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_THREAD_VERTICES` vertices in its
:class:`~newton._src.geometry.terrain_law.TerrainPairWork`, :func:`terrain_hull_rows_tiled` runs ``wp.block_dim()``
lanes per queued pair on the work list's fixed grid (the lanes mark the window vertices that yield a row, the first lane
walks those in window order and finishes the pair, which gives it the one-thread kernel's rows), and
:func:`terrain_hull_rows_full`, on the fixed full-finish grid of its
:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch` striding over the class's deferred list, the pairs the
other two deferred (more distinct rows than the class's 24 walk slots). The walk, the rules and the finish are
:mod:`~newton._src.geometry.terrain_law_solid`'s; the module is compiled without floating-point contraction, so the CPU
decides every boundary as the GPU does.
"""

from __future__ import annotations

import warp as wp

from .terrain_contacts import TERRAIN_PARTNER_HULL, TERRAIN_PARTNER_WALK_SLOTS
from .terrain_law import (
    TerrainFeatureData,
    TerrainFinishScratch,
    TerrainLawRows,
    TerrainPairInputs,
    TerrainPairWork,
    terrain_pair_exit,
)
from .terrain_law_cores import TerrainHullCore
from .terrain_law_solid import (
    TERRAIN_SOLID_MASK_WORDS,
    TerrainSolidData,
    solid_mark,
    solid_pair_read,
    solid_pair_window,
    solid_replay,
    solid_rows_full,
    solid_rows_thread,
    solid_unsupported,
)
from .types import GeoType

_ELLIPSOID = wp.constant(int(GeoType.ELLIPSOID))
_CONVEX_MESH = wp.constant(int(GeoType.CONVEX_MESH))

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

TERRAIN_HULL_WALK_SLOTS = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_HULL]
"""Walk slots of a pair of the class: its distinct rows while it walks its window, and its rows when they fit: 24."""

_ROWS = wp.constant(TERRAIN_HULL_WALK_SLOTS)
_CLASS = wp.constant(TERRAIN_PARTNER_HULL)
_MASK_WORDS = wp.constant(TERRAIN_SOLID_MASK_WORDS)

_Mask = wp.types.vector(length=TERRAIN_SOLID_MASK_WORDS, dtype=wp.int32)


@wp.func
def _core(local: wp.transform, size: wp.vec4, kind: int, partner: int, solids: TerrainSolidData):
    """The pair's convex hull (its points and faces from the hull table, scaled) or ellipsoid (semi-axes) in the
    heightfield frame; ``False`` for a shape of the class with no core yet (a barrel cylinder)."""
    core = TerrainHullCore()
    core.c = wp.transform_get_translation(local)
    core.rot = wp.quat_to_matrix(wp.normalize(wp.transform_get_rotation(local)))
    core.s = wp.vec3(size[0], size[1], size[2])
    core.table = solids.hulls
    if kind == _ELLIPSOID:
        core.ellipsoid = 1
        core.radius = wp.max(core.s[0], wp.max(core.s[1], core.s[2]))
        return True, core
    if kind != _CONVEX_MESH:
        return False, core
    hull = solids.hulls.shape_hull[partner]
    if hull < 0:
        return False, core
    points = solids.hulls.hull_points[hull]
    faces = solids.hulls.hull_faces[hull]
    triangles = solids.hulls.hull_triangles[hull]
    core.point_begin = points[0]
    core.point_count = points[1]
    core.face_begin = faces[0]
    core.face_count = faces[1]
    core.triangle_begin = triangles[0]
    core.triangle_count = triangles[1]
    # the radius, and the point lowest along the heightfield's up axis (the support searches' start)
    up = wp.cw_mul(wp.vec3(core.rot[2, 0], core.rot[2, 1], core.rot[2, 2]), core.s)
    radius = float(0.0)
    core.start = core.point_begin
    low = wp.dot(solids.hulls.points[core.point_begin], up)
    for i in range(core.point_begin, core.point_begin + core.point_count):
        p = solids.hulls.points[i]
        radius = wp.max(radius, wp.length(wp.cw_mul(p, core.s)))
        value = wp.dot(p, up)
        if value < low:
            low = value
            core.start = i
    core.radius = radius
    return True, core


@wp.kernel(enable_backward=False)
def terrain_hull_rows(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    work: TerrainPairWork,
    solids: TerrainSolidData,
):
    """The law's rows of the class's pairs, one thread per pair (``begin`` and ``slot_begin`` are the class's first
    pair and walk slot); a pair from ``work.first`` on whose window holds more than ``solids.thread_vertices`` vertices
    is queued in ``work`` for the tiled kernel.

    Launch: ``wp.launch`` with one thread per pair of the class and ``block_dim``
    :func:`~newton._src.geometry.terrain_law.terrain_law_block_dim`.
    """
    i = wp.tid()
    pair = begin + i
    state, frame, heightfield, local, partner, size, reach = solid_pair_read(inputs, data, pair)
    if state != 0:
        terrain_pair_exit(out, pair, state)
        return
    supported, core = _core(local, size, inputs.shape_type[partner], partner, solids)
    if not supported:
        solid_unsupported(out, pair)
        return
    solid_rows_thread(
        core, data, out, work, solids, pair, slot_begin + i * _ROWS, _ROWS, _CLASS, frame, heightfield, reach
    )


@wp.kernel(enable_backward=False)
def terrain_hull_rows_tiled(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    work: TerrainPairWork,
    solids: TerrainSolidData,
):
    """The class's pairs the one-thread kernel queued for their large windows, ``wp.block_dim()`` lanes per pair
    (``begin`` and ``slot_begin`` are the tiled segment's first pair and walk slot): the grid's ``work.blocks`` blocks
    stride over the queue; a block's lanes mark the window vertices that yield a row, and its first lane walks those in
    window order and finishes the pair, which gives it the one-thread walk's rows.

    Launch: ``wp.launch_tiled`` with ``dim=work.blocks`` and ``block_dim``
    :func:`~newton._src.geometry.terrain_law.terrain_law_block_dim`. A block past ``work.blocks`` returns at once (it
    would take queued pairs a second time); with fewer blocks the missing blocks' pairs would get no rows.
    """
    block, lane = wp.tid()
    if block >= work.blocks:
        return
    lanes = wp.block_dim()
    for entry in range(block, work.count[0], work.blocks):
        # every lane reads the same pair; the one-thread kernel found it current, cored and in its window
        pair = work.pairs[entry]
        _state, frame, heightfield, local, partner, size, reach = solid_pair_read(inputs, data, pair)
        _supported, core = _core(local, size, inputs.shape_type[partner], partner, solids)
        _state, cells, window, core, setup = solid_pair_window(core, data, heightfield, reach)
        width = cells[3] - cells[2] + 2
        vertices = (cells[1] - cells[0] + 2) * width
        words = (vertices + 31) // 32
        marked = _Mask()
        if words <= _MASK_WORDS:
            marked = solid_mark(core, data, window, setup, cells, lane, lanes)
            # lanes mark disjoint vertices, so a sum of their words is their union
            for word in range(words):
                marked[word] = wp.tile_extract(wp.tile_sum(wp.tile(marked[word])), 0)
        if lane == 0:
            solid_replay(
                core,
                data,
                out,
                window,
                setup,
                cells,
                marked,
                words,
                pair,
                slot_begin + (pair - begin) * _ROWS,
                _ROWS,
                _CLASS,
                frame,
            )


@wp.kernel(enable_backward=False)
def terrain_hull_rows_full(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    scratch: TerrainFinishScratch,
    solids: TerrainSolidData,
):
    """Finish the class's deferred pairs (``begin`` and ``slot_begin`` are the class's first pair and walk slot): the
    grid's ``scratch.blocks`` blocks stride over the class's deferred list, and each block's first lane walks its pairs
    again, one at a time, into the block's scratch (every distinct candidate a window of the class can give), runs the
    finish and writes every row it keeps, in the pair's walk slots when they fit, else in rows of the spill region. A
    pair's rows depend on the pair alone: not on the order of the deferral, the number of deferred pairs or the block
    that finishes it.

    Launch: ``wp.launch_tiled`` with ``dim=scratch.blocks`` and ``block_dim``
    :func:`~newton._src.geometry.terrain_law.terrain_law_block_dim`. A block past ``scratch.blocks`` returns at once
    (it has no scratch, and would finish deferred pairs a second time); with fewer blocks the missing blocks' pairs
    would keep no rows.
    """
    block, lane = wp.tid()
    if block >= scratch.blocks or lane != 0:
        return
    for entry in range(block, out.deferred_count[_CLASS], scratch.blocks):
        pair = out.deferred[begin + entry]
        i = pair - begin
        _state, frame, heightfield, local, partner, size, reach = solid_pair_read(inputs, data, pair)
        _supported, core = _core(local, size, inputs.shape_type[partner], partner, solids)
        solid_rows_full(core, data, out, scratch, block, pair, slot_begin + i * _ROWS, _ROWS, frame, heightfield, reach)

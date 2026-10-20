# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's kernels for boxes (rules F, E, V and R of the law for solid cores).

The kernels have the per-type signature ``(begin, slot_begin, inputs, data, out, solids)``
(:class:`~newton._src.geometry.terrain_law.TerrainPairInputs`, :class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData`):
:func:`terrain_box_rows` runs one thread per pair over the whole class and lists the pairs whose windows (as the
pose gives them) hold more than :data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_THREAD_VERTICES`
vertices, :func:`terrain_box_rows_tiled` runs ``wp.block_dim()`` lanes per listed pair on a fixed grid striding over
the list (the lanes mark the window vertices that yield a row, the first lane walks those in window order and finishes
the pair, which gives it the one-thread kernel's rows), and :func:`terrain_box_rows_full`, on a fixed grid striding
over the class's deferred list, the pairs the other two deferred
(more distinct rows than the class's 16 slots). The walk, the rules and the finish are
:mod:`~newton._src.geometry.terrain_law_solid`'s; the module is compiled without floating-point contraction, so the
CPU decides every boundary as the GPU does.
"""

from __future__ import annotations

import warp as wp

from .terrain_contacts import TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_ROWS
from .terrain_law import TerrainFeatureData, TerrainLawRows, TerrainPairInputs
from .terrain_law_cores import TerrainBoxCore
from .terrain_law_solid import (
    TERRAIN_SOLID_MASK_WORDS,
    TerrainSolidData,
    solid_mark,
    solid_pair_exit,
    solid_pair_read,
    solid_pair_window,
    solid_replay,
    solid_rows_full,
    solid_rows_thread,
    solid_unsupported,
)

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

TERRAIN_BOX_ROWS = TERRAIN_PARTNER_ROWS[TERRAIN_PARTNER_BOX]
"""Rows a pair of the class keeps (its slots): 16."""

_ROWS = wp.constant(TERRAIN_BOX_ROWS)
_CLASS = wp.constant(TERRAIN_PARTNER_BOX)
_MASK_WORDS = wp.constant(TERRAIN_SOLID_MASK_WORDS)

_Mask = wp.types.vector(length=TERRAIN_SOLID_MASK_WORDS, dtype=wp.int32)


@wp.func
def _core(local: wp.transform, size: wp.vec4, kind: int, partner: int, solids: TerrainSolidData):
    """The pair's box in the heightfield frame (centre, rotation, half extents)."""
    core = TerrainBoxCore()
    core.c = wp.transform_get_translation(local)
    core.rot = wp.quat_to_matrix(wp.normalize(wp.transform_get_rotation(local)))
    core.h = wp.vec3(size[0], size[1], size[2])
    return True, core


@wp.kernel(enable_backward=False)
def terrain_box_rows(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    solids: TerrainSolidData,
):
    """The law's rows of the class's pairs, one thread per pair; a window of more than ``solids.thread_vertices``
    vertices is listed for the tiled kernel."""
    i = wp.tid()
    pair = begin + i
    state, frame, heightfield, local, partner, size, reach = solid_pair_read(inputs, data, pair)
    if state != 0:
        solid_pair_exit(out, pair, state)
        return
    supported, core = _core(local, size, inputs.shape_type[partner], partner, solids)
    if not supported:
        solid_unsupported(out, pair)
        return
    solid_rows_thread(core, data, out, solids, pair, slot_begin + i * _ROWS, _ROWS, _CLASS, frame, heightfield, reach)


@wp.kernel(enable_backward=False)
def terrain_box_rows_tiled(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    solids: TerrainSolidData,
):
    """The class's pairs the one-thread kernel listed for their large windows, ``wp.block_dim()`` lanes per pair
    (``begin`` and ``slot_begin`` are the class's first pair and slot): the grid's ``solids.tile_blocks`` blocks stride
    over the list; a block's lanes mark the window vertices that yield a row, and its first lane walks those in window
    order and finishes the pair, which gives it the one-thread walk's rows."""
    block, lane = wp.tid()
    lanes = wp.block_dim()
    for entry in range(block, solids.wide_count[_CLASS], solids.tile_blocks):
        # every lane reads the same pair; the one-thread kernel found it current, cored and in its window
        pair = solids.wide[begin + entry]
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
def terrain_box_rows_full(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    solids: TerrainSolidData,
):
    """Finish the class's deferred pairs (``begin`` and ``slot_begin`` are the class's first pair and slot): its
    ``solids.full_threads`` threads stride over the class's deferred list, and each walks its pairs again, one at a
    time, into its own scratch list (:data:`~newton._src.geometry.terrain_law.TERRAIN_ROW_CANDIDATES` rows), runs the
    finish and writes the kept rows. A pair's rows depend on the pair alone: not on the order of the deferral, the
    number of deferred pairs or the thread that finishes it."""
    thread = wp.tid()
    if thread >= solids.full_threads:
        return
    for entry in range(thread, out.deferred_count[_CLASS], solids.full_threads):
        pair = out.deferred[begin + entry]
        i = pair - begin
        _state, frame, heightfield, local, partner, size, reach = solid_pair_read(inputs, data, pair)
        _supported, core = _core(local, size, inputs.shape_type[partner], partner, solids)
        solid_rows_full(
            core, data, out, solids, thread, pair, slot_begin + i * _ROWS, _ROWS, _CLASS, frame, heightfield, reach
        )

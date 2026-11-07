# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The streamed rows' records: what the sweeps' row loops load a row ahead, written by the rows' writers.

The rows are the walk's (:mod:`.walk_rows`: the one writer walks every row's seeds into its whitened form ``Z``). Every
row whose metadata a sweep streams has a record (:func:`.row_metadata.pack_row_record`, which the sweeps' row loop
decodes: its sides' coordinates and widths, its stored diagonal, right-hand side, kind, parent and friction
coefficient), which the writer of the row stores once the row is final (:func:`write_row_record`): the one writer
after its closure (:func:`.walk_rows.walked_rows_kernel`), the free bodies' writer as it writes theirs
(:func:`.walk_rows.write_free_body_rows`), and the patch link the parent it links (:func:`store_record_parent`). The
records are the dense rows' where a world's line holds more of them than the sweep stages, and the free bodies' rows
(:func:`.sweep_routes.free_body_rows`, rows over at most two free bodies; their ``A_ii`` is the walk's). The bilateral
closure is the factor's (:mod:`.factor.closure`).
"""

from __future__ import annotations

import warp as wp

from .device_preludes import CPU_MATH, dual_native
from .row_metadata import RECORD_WORDS, pack_row_record
from .row_records import DenseRows

# ---------------------------------------------------------------------------------------------------------------------
# Free-body rows
# ---------------------------------------------------------------------------------------------------------------------
#
# A free-body row acts on at most two free rigid bodies, ``a`` and ``b`` (a negative body for none), through six
# entries each. A free body is a one-node tree; the one writer walks each side's seed into ``Z = L^-1 J^T`` (its walk's
# factor, armature included), so ``A_ii = cfm + |Z_a|^2 + |Z_b|^2`` and the sweeps hold the change of the bodies'
# velocities in factor coordinates ``u = L^T dv``. A body's six velocity coordinates start at its articulation's first
# coordinate, in the addressing the mapping reads: global (``articulation_dof_start``) or world
# (``articulation_world_dof_offset``).


@wp.struct
class RowRecords:
    """Where a line's rows' records sit (:data:`.row_records.ROW_RECORDS`) and which worlds' a sweep reads: the records
    (``record``, a line's from its row ``first``; ``first`` the line's pitch where no row has one), the free bodies'
    first row (``base``: their records keep the inverse of their diagonal and count their parents from it), the
    sweep's skip of the tiers' worlds (``skip_rows``, with the line's ``speed`` rows from ``speed_first``, and each
    world's free bodies' rows ``mf_count``: :func:`swept`), and each articulation's first world coordinate and response
    coordinates (``offset``, ``dofs``: a row's sides as the record holds them)."""

    record: wp.array2d[wp.int32]
    first: int
    base: int
    skip_rows: int
    speed: int
    mf_count: wp.array[int]
    speed_first: wp.array[int]
    offset: wp.array[int]
    dofs: wp.array[int]


@wp.func_native(
    dual_native(
        pack_row_record(
            "&record.data[at]", "dof_a", "dof_b", "n_a", "n_b", "diagonal", "rhs", "kind", "parent", "mu", "    "
        ),
        preludes=(CPU_MATH,),
    )
)
def store_row_record(
    record: wp.array2d[wp.int32],
    at: int,
    dof_a: int,
    dof_b: int,
    n_a: int,
    n_b: int,
    diagonal: float,
    rhs: float,
    kind: int,
    parent: int,
    mu: float,
): ...


@wp.func_native(
    """    // The kind and parent word's high half (pack_mf_header: the parent above the kind), stored alone: no read of
    // the kind, which the row's writer stored (little-endian on both devices).
    reinterpret_cast<unsigned short*>(&record.data[at + 3])[1] = static_cast<unsigned short>(parent);
"""
)
def store_parent_word(record: wp.array2d[wp.int32], at: int, parent: int): ...


@wp.func
def record_at(records: RowRecords, line: int, slot: int) -> int:
    """The flat index of row ``slot``'s record in line ``line``'s records (-1 for a row without one)."""
    if slot < records.first:
        return -1
    return line * records.record.shape[1] + (slot - records.first) * RECORD_WORDS


@wp.func
def swept(records: RowRecords, world: int, count: int) -> bool:
    """Whether the matrix-free sweep visits ``world`` (``count`` dense rows), so its rows' records are read: every
    world but those the tiers own (no free body's row, at most ``skip_rows`` dense rows and, with the line's speed rows,
    none from ``speed_first``: :func:`.sweep_routes.mf_gs_source`'s skip)."""
    if records.mf_count[world] > 0:
        return True
    if wp.min(count, records.base) > records.skip_rows:
        return True
    return records.speed != 0 and count > records.speed_first[world]


@wp.func
def write_row_record(records: RowRecords, rows: DenseRows, incident: float, line: int, slot: int):
    """Store row ``slot``'s record (:func:`.row_metadata.pack_row_record`) from the row as its writer left it: its sides'
    first world coordinates and widths, its diagonal (a free body's row its inverse, cached for its step's form of the
    law), its residual at ``du = 0`` (rhs plus ``incident``), kind, parent (a free body's row's from ``base``, as the
    sweeps stage the free bodies' rows) and friction coefficient. A row before ``first`` has none."""
    at = record_at(records, line, slot)
    if at < 0:
        return
    sides = rows.sides[line, slot]
    dof_a = int(-1)
    dof_b = int(-1)
    n_a = int(0)
    n_b = int(0)
    if sides[0] >= 0:
        dof_a = records.offset[sides[0]]
        n_a = records.dofs[sides[0]]
    if sides[1] >= 0:
        dof_b = records.offset[sides[1]]
        n_b = records.dofs[sides[1]]
    diagonal = rows.diag[line, slot]
    parent = rows.row_parent[line, slot]
    if slot >= records.base:
        inverse = float(0.0)
        if diagonal > 0.0:
            inverse = 1.0 / diagonal
        diagonal = inverse
        if parent >= 0:
            parent = parent - records.base
    store_row_record(
        records.record, at, dof_a, dof_b, n_a, n_b, diagonal, rows.rhs[line, slot] + incident,
        rows.row_type[line, slot], parent, rows.row_mu[line, slot],
    )  # fmt: skip


@wp.func
def store_record_parent(records: RowRecords, line: int, slot: int, parent: int):
    """Store row ``slot``'s linked ``parent`` (a line row) in its record's kind and parent word, where it has a record:
    the patch link's, after the row's writer stored the rest."""
    at = record_at(records, line, slot)
    if at < 0:
        return
    stored = parent
    if slot >= records.base:
        stored = parent - records.base
    store_parent_word(records.record, at, stored)


def no_row_records() -> RowRecords:
    """Records of a store that has none: no row is past ``first`` (a link outside a solver's step)."""
    records = RowRecords()
    records.first = 1 << 30
    return records


def bind_row_records(placement) -> RowRecords:
    """The placement's records (:meth:`.placement.Placement.allocate_row_records`) as their writers take them."""
    records = RowRecords()
    records.record = placement.row_record
    records.first = int(placement.record_first) if placement.writes_records else int(placement.row_pitch)
    records.base = int(placement.body_base)
    records.skip_rows = int(placement.record_skip_rows)
    records.speed = int(placement.record_speed_rows)
    records.mf_count = placement.mf_constraint_count
    records.speed_first = placement.speed_first
    records.offset = placement.forest.articulation_world_dof_offset
    records.dofs = placement.forest.articulation_response_dof_count
    return records

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The streamed rows' records (:func:`.row_metadata.pack_row_record`), which the sweeps' row loops load a row ahead:
each row's writer stores its record once the row is final (:func:`write_row_record`), the patch link the parent it
links (:func:`store_record_parent`).
"""

from __future__ import annotations

import warp as wp

from .device_preludes import CPU_MATH, dual_native
from .row_metadata import HEADER_WORDS, RECORD_WORDS, pack_row_header, pack_row_record
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
    """Where a line's rows' records sit (:func:`.row_metadata.record_layout`: ``record`` from row ``first``, the free
    rows' headers from row ``base`` at word ``free``), which worlds' a sweep reads (:func:`swept`: ``skip_rows``,
    ``speed``, ``speed_first``, ``mf_count``) and each articulation's first world coordinate and coordinates
    (``offset``, ``dofs``)."""

    record: wp.array2d[wp.int32]
    first: int
    base: int
    free: int
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
    dual_native(
        pack_row_header("&record.data[at]", "dof_a", "dof_b", "inverse", "rhs", "kind", "parent", "    "),
        preludes=(CPU_MATH,),
    )
)
def store_row_header(
    record: wp.array2d[wp.int32],
    at: int,
    dof_a: int,
    dof_b: int,
    inverse: float,
    rhs: float,
    kind: int,
    parent: int,
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
    """The flat index of row ``slot``'s record in line ``line``'s records (-1 for a row without one): a dense row's of
    :data:`.row_metadata.RECORD_WORDS` words, a free row's header from ``free``."""
    if slot < records.first:
        return -1
    at = line * records.record.shape[1]
    if slot >= records.base:
        return at + records.free + (slot - records.base) * HEADER_WORDS
    return at + (slot - records.first) * RECORD_WORDS


@wp.func
def swept(records: RowRecords, world: int, count: int) -> bool:
    """Whether the matrix-free sweep visits ``world`` (``count`` dense rows): every world but the tiers'
    (:func:`.sweep_levels.claim_worlds`' predicate)."""
    if records.mf_count[world] > 0:
        return True
    if wp.min(count, records.base) > records.skip_rows:
        return True
    return records.speed != 0 and count > records.speed_first[world]


@wp.func
def record_firsts(records: RowRecords, sides: wp.vec2i) -> wp.vec2i:
    """The first world coordinates of a row's ``sides`` as its record holds them (-1 for no side)."""
    firsts = wp.vec2i(-1, -1)
    if sides[0] >= 0:
        firsts[0] = records.offset[sides[0]]
    if sides[1] >= 0:
        firsts[1] = records.offset[sides[1]]
    return firsts


@wp.func
def write_row_record(records: RowRecords, rows: DenseRows, incident: float, line: int, slot: int):
    """Store row ``slot``'s record from the row as its writer left it (:func:`store_row_record_of`)."""
    if record_at(records, line, slot) < 0:
        return
    sides = rows.sides[line, slot]
    store_row_record_of(records, rows, incident, line, slot, sides, record_firsts(records, sides))


@wp.func
def store_row_record_of(
    records: RowRecords, rows: DenseRows, incident: float, line: int, slot: int, sides: wp.vec2i, firsts: wp.vec2i
):
    """Store row ``slot``'s record (:func:`.row_metadata.pack_row_record`), its ``sides`` at ``firsts``
    (:func:`record_firsts`), the rest from the row as its writer left it (its rhs plus ``incident``). A free row's is its
    header (:func:`.row_metadata.pack_row_header`), with its diagonal's inverse and its parent from ``base``."""
    at = record_at(records, line, slot)
    if at < 0:
        return
    diagonal = rows.diag[line, slot]
    parent = rows.row_parent[line, slot]
    rhs = rows.rhs[line, slot] + incident
    kind = rows.row_type[line, slot]
    if slot >= records.base:
        inverse = float(0.0)
        if diagonal > 0.0:
            inverse = 1.0 / diagonal
        if parent >= 0:
            parent = parent - records.base
        store_row_header(records.record, at, firsts[0], firsts[1], inverse, rhs, kind, parent)
        return
    n_a = int(0)
    n_b = int(0)
    if sides[0] >= 0:
        n_a = records.dofs[sides[0]]
    if sides[1] >= 0:
        n_b = records.dofs[sides[1]]
    store_row_record(
        records.record, at, firsts[0], firsts[1], n_a, n_b, diagonal, rhs, kind, parent, rows.row_mu[line, slot]
    )


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
    records.base = 1 << 30
    return records


def bind_row_records(placement) -> RowRecords:
    """The placement's records (:meth:`.placement.Placement.allocate_row_records`) as their writers take them."""
    records = RowRecords()
    records.record = placement.row_record
    records.first = int(placement.record_first) if placement.writes_records else int(placement.row_pitch)
    records.base = int(placement.body_base)
    records.free = int(placement.free_records)
    records.skip_rows = int(placement.record_skip_rows)
    records.speed = int(placement.record_speed_rows)
    records.mf_count = placement.mf_constraint_count
    records.speed_first = placement.speed_first
    records.offset = placement.forest.articulation_world_dof_offset
    records.dofs = placement.forest.articulation_response_dof_count
    return records

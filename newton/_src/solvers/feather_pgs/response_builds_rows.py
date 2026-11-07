# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The streamed rows' records (:func:`.row_metadata.pack_row_record`), which the sweeps' row loops load a row ahead:
one pass stores the records of the worlds the general sweep and the unit builder read, once their rows are final
(:func:`store_row_records`).
"""

from __future__ import annotations

import warp as wp

from ...core.execution import Buckets, bucket_item
from .device_preludes import CPU_MATH, CPU_WORDS, dual_native
from .row_metadata import HEADER_WORDS, RECORD_WORDS, pack_row_record
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
    rows' headers from row ``base`` at word ``free``) and each articulation's first world coordinate and coordinates
    (``offset``, ``dofs``)."""

    record: wp.array2d[wp.int32]
    first: int
    base: int
    free: int
    offset: wp.array[int]
    dofs: wp.array[int]


_RECORD_WORDS, _HEADER_WORDS = wp.constant(RECORD_WORDS), wp.constant(HEADER_WORDS)


@wp.func_native(dual_native(pack_row_record("&record.data[at]"), preludes=(CPU_MATH, CPU_WORDS)))
def store_row_record(
    record: wp.array2d[wp.int32],
    at: int,
    words: int,
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
    """Store row ``slot``'s record (:func:`.row_metadata.pack_row_record`) from the row as its writers left it (its rhs
    plus ``incident``). A free row's is its header, with its diagonal's inverse and its parent from ``base``."""
    at = record_at(records, line, slot)
    if at < 0:
        return
    diagonal = rows.diag[line, slot]
    parent = rows.row_parent[line, slot]
    rhs = rows.rhs[line, slot] + incident
    kind = rows.row_type[line, slot]
    sides = rows.sides[line, slot]
    firsts = record_firsts(records, sides)
    words = int(_RECORD_WORDS)
    n_a = int(0)
    n_b = int(0)
    mu = float(0.0)
    if slot >= records.base:
        words = _HEADER_WORDS
        inverse = float(0.0)
        if diagonal > 0.0:
            inverse = 1.0 / diagonal
        diagonal = inverse
        if parent >= 0:
            parent = parent - records.base
    else:
        if sides[0] >= 0:
            n_a = records.dofs[sides[0]]
        if sides[1] >= 0:
            n_b = records.dofs[sides[1]]
        mu = rows.row_mu[line, slot]
    store_row_record(records.record, at, words, firsts[0], firsts[1], n_a, n_b, diagonal, rhs, kind, parent, mu)


@wp.kernel(enable_backward=False)
def store_row_records(
    worlds: Buckets,
    grid: int,
    lanes: int,
    records: RowRecords,
    rows: DenseRows,
    incident: wp.array2d[float],
    count: wp.array[int],
    mf_count: wp.array[int],
):
    """Store the records of the rows of ``worlds`` (the claims' worlds the general sweep and the unit builder read,
    :func:`.sweep_levels.claim_worlds`), once their writers and the patch link have left them final: a world's dense
    rows (``count``) from ``first``, then its free rows (``mf_count``) from ``base``, ``lanes`` lanes a world
    and ``grid`` worlds at once."""
    item, lane = wp.tid()
    pitch = rows.rhs.shape[1]
    world = bucket_item(worlds, item)
    while world >= 0:
        line = rows.first[world]
        for slot in range(records.first + lane, wp.min(count[world], records.base), lanes):
            write_row_record(records, rows, incident[line, slot], line, slot)
        for slot in range(records.base + lane, wp.min(records.base + mf_count[world], pitch), lanes):
            write_row_record(records, rows, incident[line, slot], line, slot)
        item += grid
        world = bucket_item(worlds, item)


def bind_row_records(placement) -> RowRecords:
    """The placement's records (:meth:`.placement.Placement.allocate_row_records`) as their pass takes them."""
    records = RowRecords()
    records.record = placement.row_record
    records.first = int(placement.record_first) if placement.writes_records else int(placement.row_pitch)
    records.base = int(placement.body_base)
    records.free = int(placement.free_records)
    records.offset = placement.forest.articulation_world_dof_offset
    records.dofs = placement.forest.articulation_response_dof_count
    return records

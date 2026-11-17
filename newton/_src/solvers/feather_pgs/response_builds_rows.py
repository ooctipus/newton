# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The streamed rows' records (:func:`.row_metadata.pack_row_record`), which the sweeps' row loops load a row ahead:
one pass stores the records of the worlds the general sweep and the unit builder read, once their rows are final
(:func:`store_row_records`).
"""

from __future__ import annotations

import warp as wp

from ...core.execution import Buckets
from .device_preludes import CPU_MATH, CPU_WORDS, dual_native, span_item, span_total
from .projection_laws import idle_key, idle_root
from .row_laws import ROW_CONTACT, ROW_FRICTION, ROW_JOINT_LIMIT, ROW_JOINT_VELOCITY_LIMIT, ROW_TORSION
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
    (``offset``, ``dofs``), or where the rows take support classes (``classed``) each row's word (``classes``,
    :func:`.packed_rows.row_classes`), which its record holds in place of its sides' widths."""

    record: wp.array2d[wp.int32]
    first: int
    base: int
    free: int
    offset: wp.array[int]
    dofs: wp.array[int]
    classed: int
    classes: wp.array2d[int]


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
    key: float,
    root: float,
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
def ring_key(rows: DenseRows, incident: wp.array2d[float], line: int, first: int, hops: int) -> float:
    """The least idle key of the patch ring of normal row ``first`` (the rows the live order's test walks for a first
    tangent, :class:`.sweep.LiveOrder`): over its contact normals and joint limits of a positive diagonal; minus
    infinity where it holds a row of another kind of a positive diagonal (it acts whatever the bound), infinity where
    it holds none. The test's own part is the ring's impulses. ``hops`` at least the test's guard: a ring that does not
    close is walked as far, or further (a lower key: more visits)."""
    key = float(wp.inf)
    r = first
    for _hop in range(hops + 1):
        if r < 0:
            break
        kind = rows.row_type[line, r]
        diagonal = rows.diag[line, r]
        if not (diagonal <= 0.0) and kind != ROW_JOINT_VELOCITY_LIMIT and kind != ROW_TORSION:
            if kind == ROW_CONTACT or kind == ROW_JOINT_LIMIT:
                key = wp.min(key, idle_key(rows.rhs[line, r] + incident[line, r], diagonal))
            else:
                key = -wp.inf
        r = rows.row_parent[line, r]
        if r == first:
            break
    return key


@wp.func
def write_row_record(records: RowRecords, rows: DenseRows, incidents: wp.array2d[float], line: int, slot: int):
    """Store row ``slot``'s record (:func:`.row_metadata.pack_row_record`) from the row as its writers left it (its rhs
    plus its incident velocity). A free row's is its header, with its diagonal's inverse and its parent from ``base``.
    A dense row's idle key is its own, a first tangent's its patch ring's (:func:`ring_key`)."""
    at = record_at(records, line, slot)
    if at < 0:
        return
    incident = incidents[line, slot]
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
    key = float(0.0)
    root = float(0.0)
    if slot >= records.base:
        words = _HEADER_WORDS
        inverse = float(0.0)
        if diagonal > 0.0:
            inverse = 1.0 / diagonal
        diagonal = inverse
        if parent >= 0:
            parent = parent - records.base
    else:
        if records.classed != 0:
            n_a = records.classes[line, slot]
        else:
            if sides[0] >= 0:
                n_a = records.dofs[sides[0]]
            if sides[1] >= 0:
                n_b = records.dofs[sides[1]]
        mu = rows.row_mu[line, slot]
        key = idle_key(rhs, diagonal)
        if kind == ROW_FRICTION and parent >= 0 and slot == parent + 1:
            key = ring_key(rows, incidents, line, parent, rows.rhs.shape[1])
        root = idle_root(diagonal)
    store_row_record(
        records.record, at, words, firsts[0], firsts[1], n_a, n_b, diagonal, rhs, kind, parent, mu, key, root
    )


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
    mf_contact_end: wp.array[int],
    regions: int,
):
    """Store the records of the rows of ``worlds`` (the claims' worlds the general sweep and the unit builder read,
    :func:`.sweep_levels.claim_worlds`), once their writers and the patch link have left them final: a world's dense
    rows (``count``) from ``first``, then its free rows (``mf_count``) from ``base``, ``lanes`` lanes a world
    and ``grid`` worlds at once. Where the free bodies' contacts are regions (``regions``) no sweep reads their
    rows' records: the free rows' start after their contacts' (``mf_contact_end``), at the speed rows."""
    item, lane = wp.tid()
    pitch = rows.rhs.shape[1]
    # A world's lanes read the list together, a warp's at most (``lanes`` is a multiple of 32 or below it).
    readers = wp.min(lanes, 32)
    total = span_total(worlds, readers)
    while item < total:
        world = span_item(worlds, item, readers)
        line = rows.first[world]
        for slot in range(records.first + lane, wp.min(count[world], records.base), lanes):
            write_row_record(records, rows, incident, line, slot)
        free_first = records.base + lane
        if regions != 0:
            free_first += mf_contact_end[world]
        for slot in range(free_first, wp.min(records.base + mf_count[world], pitch), lanes):
            write_row_record(records, rows, incident, line, slot)
        item += grid


def bind_row_records(placement) -> RowRecords:
    """The placement's records (:meth:`.placement.Placement.allocate_row_records`) as their pass takes them."""
    records = RowRecords()
    records.record = placement.row_record
    records.first = int(placement.record_first) if placement.writes_records else int(placement.row_pitch)
    records.base = int(placement.body_base)
    records.free = int(placement.free_records)
    records.offset = placement.forest.articulation_world_dof_offset
    records.dofs = placement.forest.articulation_response_dof_count
    records.classed = int(placement.classed)
    records.classes = placement.row_classes.row_class
    return records

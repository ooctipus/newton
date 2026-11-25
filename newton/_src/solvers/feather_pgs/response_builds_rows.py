# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The streamed rows' records (:func:`.row_metadata.pack_row_record`), which the sweeps' row loops load a row ahead:
each row's writer stores its record once the row is final, from the values it wrote (:func:`write_row_record`): the
general placement's writer a world's dense rows after its last part (:func:`write_world_records`), the free bodies'
writer each free row after its rhs.
"""

from __future__ import annotations

import functools

import warp as wp

from .device_preludes import CPU_MATH, CPU_WORDS, dual_native
from .projection_laws import idle_key, idle_root
from .row_laws import ROW_CONTACT, ROW_JOINT_LIMIT, ROW_TORSION
from .row_metadata import pack_row_record, record_words
from .row_records import DenseRows, line_size

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
    """Where a line's rows' records sit (:func:`.row_metadata.record_layout`: ``record``, the dense rows' from row 0
    where ``dense``, then the free rows' headers from the line's body base ``base`` where ``free``) and each
    articulation's first world coordinate and coordinates (``offset``, ``dofs``), or where the rows take support
    classes (``classed``) each row's word (``classes``, :func:`.packed_rows.row_classes`), which its record holds in
    place of its sides' widths."""

    record: wp.array2d[wp.int32]
    dense: int
    free: int
    base: wp.array[int]
    offset: wp.array[int]
    dofs: wp.array[int]
    classed: int
    classes: wp.array2d[int]


def _name(func, split: bool):
    """``func`` named for its record form: its own name for the packed form, ``_split`` after it for the split one."""
    func.__name__ = func.__qualname__ = func.__name__ + ("_split" if split else "")
    return func


@functools.cache
def _record_store(split: bool) -> wp.Function:
    """Store a row's record of ``words`` words at word ``at`` of line ``line``'s records
    (:func:`.row_metadata.pack_row_record`), packed or ``split`` (:func:`.row_metadata.split_records`): the form is the
    line's, fixed where the writer is built."""

    def store_row_record(
        record: wp.array2d[wp.int32],
        line: int,
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

    _name(store_row_record, split)
    return wp.func_native(
        dual_native(
            pack_row_record(f"&record.data[(size_t)line * {line_size('record')} + at]", split=split),
            preludes=(CPU_MATH, CPU_WORDS),
        )
    )(store_row_record)


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
def row_idle_key(kind: int, rhs: float, diagonal: float) -> float:
    """A dense row's idle key as the live order's tests take it (:class:`.sweep.LiveOrder`), a patch ring's least key
    over its rows included: a contact normal's or joint limit's :func:`.projection_laws.idle_key`; minus infinity for
    another row of a positive diagonal (it acts whatever the bound); infinity for a row the tests pass (a spin row, a
    diagonal that is not positive)."""
    key = float(wp.inf)
    if not (diagonal <= 0.0) and kind != ROW_TORSION:
        key = -wp.inf
        if kind == ROW_CONTACT or kind == ROW_JOINT_LIMIT:
            key = idle_key(rhs, diagonal)
    return key


@functools.cache
def record_writers(split: bool = False) -> tuple[wp.Function, wp.Function]:
    """The records' writers of a line whose records are packed or ``split`` (:func:`.row_metadata.split_records`, fixed
    where a writer is built, so a packed line's writer keeps the packed form's code): ``write_row_record`` and
    ``write_world_records``."""
    store = _record_store(split)
    header, dense = record_words(True, split), record_words(False, split)

    def record_at(records: RowRecords, line: int, slot: int) -> int:
        """The word of row ``slot``'s record in line ``line``'s records (-1 for a row without one): a dense row's of
        :func:`.row_metadata.record_words` words, a free row's header after the dense rows' records."""
        base = records.base[line]
        if slot >= base:
            if records.free == 0:
                return -1
            return records.dense * base * wp.static(dense) + (slot - base) * wp.static(header)
        if records.dense == 0:
            return -1
        return slot * wp.static(dense)

    record_at = wp.func(_name(record_at, split))

    def write_row_record(records: RowRecords, rows: DenseRows, incident: float, line: int, slot: int):
        """Store row ``slot``'s record (:func:`.row_metadata.pack_row_record`) from the row as its writers left it (its
        rhs plus ``incident``). A free row's is its header, with its diagonal's inverse and its parent from ``base``."""
        at = record_at(records, line, slot)
        if at < 0:
            return
        diagonal = rows.diag[line, slot]
        parent = rows.row_parent[line, slot]
        rhs = rows.rhs[line, slot] + incident
        kind = rows.row_type[line, slot]
        sides = rows.sides[line, slot]
        firsts = record_firsts(records, sides)
        words = int(wp.static(dense))
        n_a = int(0)
        n_b = int(0)
        mu = float(0.0)
        key = float(0.0)
        root = float(0.0)
        if slot >= records.base[line]:
            words = wp.static(header)
            inverse = float(0.0)
            if diagonal > 0.0:
                inverse = 1.0 / diagonal
            diagonal = inverse
            if parent >= 0:
                parent = parent - records.base[line]
        else:
            if records.classed != 0:
                n_a = records.classes[line, slot]
            else:
                if sides[0] >= 0:
                    n_a = records.dofs[sides[0]]
                if sides[1] >= 0:
                    n_b = records.dofs[sides[1]]
            mu = rows.row_mu[line, slot]
            key = row_idle_key(kind, rhs, diagonal)
            root = idle_root(diagonal)
        store(
            records.record, line, at, words, firsts[0], firsts[1], n_a, n_b, diagonal, rhs, kind, parent, mu, key, root
        )

    write_row_record = wp.func(_name(write_row_record, split))

    def write_world_records(
        records: RowRecords, rows: DenseRows, incident: wp.array2d[float], line: int, lane: int, lanes: int, count: int
    ):
        """Store the records of a world's ``count`` dense rows where they hold records, ``lanes`` lanes striding over
        them (:func:`write_row_record`), once its writer and the closure left them final: the general placement's
        writer, after the part that writes the world's last rows. A world none of whose rows a sweep reads (no row can
        act, or a tier walks them) stores none."""
        if records.dense != 0:
            for slot in range(lane, wp.min(count, records.base[line]), lanes):
                write_row_record(records, rows, incident[line, slot], line, slot)

    return write_row_record, wp.func(_name(write_world_records, split))


def bind_row_records(placement) -> RowRecords:
    """The placement's records (:meth:`.placement.Placement.allocate_row_records`) as their pass takes them."""
    records = RowRecords()
    records.record = placement.row_record
    records.dense = int(placement.writes_records and placement.record_first == 0)
    records.free = int(placement.writes_records)
    records.base = placement.body_base
    records.offset = placement.forest.articulation_world_dof_offset
    records.dofs = placement.forest.articulation_response_dof_count
    records.classed = int(placement.classed)
    records.classes = placement.row_classes.row_class
    return records

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense rows stored packed by support: the matrix-free sweeps' placement of whitened rows.

A dense row touches at most two articulations, its sides (:func:`.row_records.row_sides`): the first by index, then
the second or none. Packed, a row holds its sides' coordinates in order, the first side's, then the second's: entry
``e`` of a side of ``n`` coordinates is the side's coordinate ``e``, world coordinate ``offset + e``, so a row's
entries ascend with the world's coordinates. An entry's world coordinate is not stored: it follows from the row's
sides (``row_sides``, :class:`.row_records.RowSupport`, :data:`PACKED_SIDES`). A row holds one array of entries,
``packed_Z``: the one writer stores each row's whitened form ``Z = L^-1 P J^T``, walked from its seeds at the row's
sides (:mod:`.walk_rows`), with its incident velocity ``J v_hat`` (``row_incident``).

Whitening keeps a side's support: ``J`` vanishes off the ancestors of the row's bodies, a set closed under ancestors,
so ``Z`` does too. A row's diagonal is ``A_ii = |Z|^2``, its entries' squares summed in entry order (the row law adds
``cfm``), and its velocity term at ``v = v_hat + P^T L^-T du`` is ``J_i v_hat + Z_i^T du``: the sweeps hold the velocity
change ``du`` on every articulation they touch, which the step's one leave walks down
(:func:`.factor.walk.walk_leave_kernel`).

The rows' width ``S`` (``support``) comes from the data: the most coordinates two articulations of one world hold
(:func:`support_width`). Where that is every coordinate of the widest world (worlds of one or two articulations), the
rows are laid out ``dense``: entry ``e`` is world coordinate ``e`` and the coordinates outside the row's sides hold
zero (coordinate -1), so a row is also its dense row over the world, the form the routes that hold a world's
coordinates on lanes read. Bilateral elimination widens them (:data:`WIDE`): a corrected row reaches every tree of its
block, past its two sides, so the layout is dense and the sides no longer bound a row's coordinates. The writer
writes each row's sides whole, zeros included, and records its sides: the rows are never cleared (the keyed tiers walk
their worlds' rows in the kernel). Bilateral pre-elimination corrects them (:mod:`.factor.closure`).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .device_preludes import DevicePrelude
from .field_group import Field, FieldGroup

__all__ = ["PACKED_ROWS", "PACKED_SIDES", "WIDE", "support_width"]

WIDE = 2
"""The ``dense`` layout of rows that bilateral elimination corrected: every entry is a world coordinate's, sides or
not."""

PACKED_ROWS = FieldGroup(Field("packed_Z", wp.float32, ("world", "row", "support"), owner="placement"))
"""Each dense row's packed entries: its whitened value."""


_PACKED_SIDES = """    // A packed row's entries (packed_rows): PACKED_COORDINATE(sides, e) is the world coordinate of entry e of the
    // row with these sides, -1 past them. Dense: entry e is coordinate e, held inside a side's coordinates; else the
    // first side's entries, then the second's.
    const auto PACKED_COORDINATE = [&](const wp::vec2i sides, const int e) {
        const int a0 = sides[0], a1 = sides[1];
        const int n0 = a0 >= 0 ? response_dofs.data[a0] : 0, n1 = a1 >= 0 ? response_dofs.data[a1] : 0;
        const int off0 = a0 >= 0 ? articulation_world_dof_offset.data[a0] : 0;
        const int off1 = a1 >= 0 ? articulation_world_dof_offset.data[a1] : 0;
        if (packed_dense != 0) return ((e >= off0 && e < off0 + n0) || (e >= off1 && e < off1 + n1)) ? e : -1;
        return e < n0 ? off0 + e : (e < n0 + n1 ? off1 + e - n0 : -1);
    };
    // PACKED_FIRST(sides, art): the entry where articulation art's side of a row with these sides starts, -1 when it
    // is not one of them (dense_rows.jacobian_row).
    const auto PACKED_FIRST = [&](const wp::vec2i sides, const int art) {
        if (art < 0 || (art != sides[0] && art != sides[1])) return -1;
        if (packed_dense != 0) return articulation_world_dof_offset.data[art];
        return art == sides[0] ? 0 : response_dofs.data[sides[0]];
    };"""

PACKED_SIDES = DevicePrelude("packed_sides", _PACKED_SIDES, _PACKED_SIDES)
"""``PACKED_COORDINATE(sides, e)`` and ``PACKED_FIRST(sides, art)`` for native sources that take ``response_dofs``,
``articulation_world_dof_offset`` and ``packed_dense``: a packed row's coordinates and sides from its sides (no stored
coordinates)."""


def support_width(response_dofs: np.ndarray, art_to_world: np.ndarray, world_count: int) -> int:
    """The packed rows' width: the most coordinates two articulations of one world hold (one, in a world of one);
    the rows are ``dense`` where this is the widest world's coordinates.

    Args:
        response_dofs: Each articulation's coordinates.
        art_to_world: Each articulation's world (-1 for none).
        world_count: The worlds.
    """
    largest = np.zeros((max(int(world_count), 1), 2), dtype=np.int64)
    for size, world in zip(np.asarray(response_dofs).tolist(), np.asarray(art_to_world).tolist(), strict=True):
        if world < 0:
            continue
        if size > largest[world, 0]:
            largest[world, 1] = largest[world, 0]
            largest[world, 0] = size
        elif size > largest[world, 1]:
            largest[world, 1] = size
    return max(int(largest.sum(axis=1).max()), 1)

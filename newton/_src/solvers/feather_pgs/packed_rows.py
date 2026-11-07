# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense rows stored packed by support: the matrix-free sweeps' placement of whitened rows.

A dense row touches at most two articulations, its sides (:func:`.row_records.row_sides`), the first by index. The one
writer stores its whitened form ``Z = L^-1 P J^T`` (``packed_Z``), walked from its seeds (:mod:`.walk_rows`), with its
incident velocity ``J v_hat``. ``Z`` keeps ``J``'s support, the ancestors of the row's bodies; a row's diagonal is
``|Z|^2`` (the row law adds ``cfm``) and its velocity term ``J v_hat + Z^T du``, ``du`` the sweeps' velocity change,
which the step's one leave walks down (:func:`.factor.walk.walk_leave_kernel`).

An entry's world coordinate is not stored; it follows from the layout, chosen from the data:

- by side: the first side's coordinates, then the second's, ``S`` the most two articulations of a world hold
  (:func:`support_width`); ``dense`` where that is the widest world's, entry ``e`` at coordinate ``e`` (the form the
  routes that hold a world's coordinates on lanes read); :data:`WIDE` where bilateral elimination corrects a row past
  its sides (:mod:`.factor.closure`);
- by support class (:func:`row_classes`): each side its seeds' root paths, where only the general sweep reads the rows.

Entries ascend with the world's coordinates in every layout; the rows are never cleared (the keyed tiers walk their
worlds' rows in the kernel).
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np
import warp as wp

from . import field_group, host_index
from .device_preludes import DevicePrelude
from .field_group import Field, FieldGroup

__all__ = [
    "CLASS_SIDES",
    "PACKED_ROWS",
    "PACKED_SIDES",
    "WIDE",
    "RowClasses",
    "SupportTable",
    "class_rank",
    "row_classes",
    "support_width",
]

WIDE = 2
"""The ``dense`` layout of rows that bilateral elimination corrected: every entry is a world coordinate's, sides or
not."""

PACKED_ROWS = FieldGroup(Field("packed_Z", wp.float32, ("world", "row", "support"), owner="placement"))
"""Each dense row's packed entries: its whitened value."""


_PACKED_SIDES = """    // A packed row's entries (packed_rows): SIDES_COORDINATE(off0, n0, off1, n1, e) is the world coordinate of entry
    // e of a row whose sides start at world coordinates off0 and off1 and hold n0 and n1 coordinates, -1 past them.
    // Dense: entry e is coordinate e, held inside a side's coordinates; else the first side's entries, then the second's.
    const auto SIDES_COORDINATE = [&](const int off0, const int n0, const int off1, const int n1, const int e) {
        if (packed_dense != 0) return ((e >= off0 && e < off0 + n0) || (e >= off1 && e < off1 + n1)) ? e : -1;
        return e < n0 ? off0 + e : (e < n0 + n1 ? off1 + e - n0 : -1);
    };
    // PACKED_COORDINATE(sides, e): from the row's sides (-1 for none).
    const auto PACKED_COORDINATE = [&](const wp::vec2i sides, const int e) {
        const int a0 = sides[0], a1 = sides[1];
        const int n0 = a0 >= 0 ? response_dofs.data[a0] : 0, n1 = a1 >= 0 ? response_dofs.data[a1] : 0;
        const int off0 = a0 >= 0 ? articulation_world_dof_offset.data[a0] : 0;
        const int off1 = a1 >= 0 ? articulation_world_dof_offset.data[a1] : 0;
        return SIDES_COORDINATE(off0, n0, off1, n1, e);
    };
    // RECORD_COORDINATE(firsts, widths, e): from the words of a streamed row's record that hold its sides
    // (row_metadata.pack_row_record: their first world coordinates and widths, 16-bit halves).
    const auto RECORD_COORDINATE = [&](const int firsts, const int widths, const int e) {
        return SIDES_COORDINATE(firsts >> 16, widths & 0xFFFF, (firsts << 16) >> 16, (widths >> 16) & 0xFFFF, e);
    };
    // PACKED_FIRST(sides, art): the entry where articulation art's side of a row with these sides starts, -1 when it
    // is not one of them (dense_rows.jacobian_row).
    const auto PACKED_FIRST = [&](const wp::vec2i sides, const int art) {
        if (art < 0 || (art != sides[0] && art != sides[1])) return -1;
        if (packed_dense != 0) return articulation_world_dof_offset.data[art];
        return art == sides[0] ? 0 : response_dofs.data[sides[0]];
    };"""

PACKED_SIDES = DevicePrelude("packed_sides", _PACKED_SIDES, _PACKED_SIDES)
"""``PACKED_COORDINATE(sides, e)``, ``RECORD_COORDINATE(firsts, widths, e)`` and ``PACKED_FIRST(sides, art)`` for native
sources that take ``response_dofs``, ``articulation_world_dof_offset`` and ``packed_dense``: a packed row's coordinates
from its sides or from its record's words (no stored coordinates), and where a side starts."""


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


ROW_CLASSES = (
    Field("body_class", wp.int32, ("body",)),
    Field("limit_class", wp.int32, ("dof",)),
    Field("pair_class", wp.int32, ("endpoint", "endpoint")),
    Field("class_dof", wp.int32, ("class", "entry")),
    Field("class_count", wp.int32, ("class",)),
    Field("row_class", wp.int32, ("world", "row")),
)
"""The rows' support classes (:func:`row_classes`): each body's and coordinate's endpoint (its root path; a
coordinate's is its body's), two endpoints' union, each class's tree coordinates (ascending) and count, and each row's
two classes (16-bit halves, ``0xFFFF`` none)."""

RowClasses = field_group.struct("RowClasses", ROW_CLASSES, __name__)

NO_CLASS = 0xFFFF

_CLASS_SIDES = """    // A row by its support classes (packed_rows.row_classes): each side's entries its class's tree coordinates,
    // ascending, the first side's then the second's. CLASS_ENTRY(off0, off1, classes, e): entry e's world coordinate
    // (-1 past them) for sides at off0 and off1 whose classes are the 16-bit halves of classes (0xFFFF none).
    const auto CLASS_ENTRY = [&](const int off0, const int off1, const int classes, const int e) {
        const int c0 = classes & 0xFFFF, c1 = (classes >> 16) & 0xFFFF;
        const int n0 = c0 != 0xFFFF ? class_count.data[c0] : 0;
        if (e < n0) return off0 + class_dof.data[c0 * class_dof.shape[1] + e];
        const int n1 = c1 != 0xFFFF ? class_count.data[c1] : 0;
        return e - n0 < n1 ? off1 + class_dof.data[c1 * class_dof.shape[1] + e - n0] : -1;
    };
    // From the row's sides, or from its record's words (its sides' first world coordinates and its classes).
    const auto CLASS_COORDINATE = [&](const wp::vec2i sides, const int classes, const int e) {
        return CLASS_ENTRY(sides[0] >= 0 ? articulation_world_dof_offset.data[sides[0]] : 0,
                           sides[1] >= 0 ? articulation_world_dof_offset.data[sides[1]] : 0, classes, e);
    };
    const auto CLASS_RECORD_COORDINATE = [&](const int firsts, const int classes, const int e) {
        return CLASS_ENTRY(firsts >> 16, (firsts << 16) >> 16, classes, e);
    };"""

CLASS_SIDES = DevicePrelude("class_sides", _CLASS_SIDES, _CLASS_SIDES)

_CLASS_RANK = """    // The writer's classes: CLASS_OF(classes, e0, e1) the class of a side whose seeds' endpoints are e0 and e1 (e0
    // twice for one seed); CLASS_RANK(classes, cls, c) tree coordinate c's entry in class cls, -1 off it;
    // CLASS_NEXT(classes, cls, at, c) the same for a walk's coordinates (each seed's in descending order, all in the
    // class) from the cursor at (-1 to start); CLASS_WORD(c0, c1) a row's two classes (-1 none).
    const auto CLASS_OF = [](const auto& classes, const int e0, const int e1) -> int {
        return classes.pair_class.data[(size_t)e0 * classes.pair_class.shape[1] + e1];
    };
    const auto CLASS_RANK = [](const auto& classes, const int cls, const int c) -> int {
        const int* const dofs = classes.class_dof.data + (size_t)cls * classes.class_dof.shape[1];
        const int n = classes.class_count.data[cls];
        int lo = 0, hi = n;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (dofs[mid] < c)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo < n && dofs[lo] == c ? lo : -1;
    };
    const auto CLASS_NEXT = [](const auto& classes, const int cls, int& at, const int c) -> int {
        const int* const dofs = classes.class_dof.data + (size_t)cls * classes.class_dof.shape[1];
        if (at < 0 || dofs[at] < c) at = classes.class_count.data[cls] - 1;
        while (at > 0 && dofs[at] > c) --at;
        return at;
    };
    const auto CLASS_WORD = [](const int c0, const int c1) -> int {
        return (int)((unsigned)(c0 & 0xFFFF) | ((unsigned)(c1 & 0xFFFF) << 16));
    };"""


@cache
def class_rank() -> DevicePrelude:
    """The one writer's ``CLASS_OF``, ``CLASS_RANK``, ``CLASS_NEXT`` and ``CLASS_WORD``, beside its walk's preludes."""
    return DevicePrelude("class_rank", _CLASS_RANK, _CLASS_RANK)


@dataclasses.dataclass(frozen=True, eq=False)
class SupportTable:
    """:data:`ROW_CLASSES` on the host but the rows' classes, and the widest row's entries."""

    body_class: np.ndarray
    limit_class: np.ndarray
    pair_class: np.ndarray
    class_dof: np.ndarray
    class_count: np.ndarray
    width: int

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, f.name).nbytes for f in ROW_CLASSES[:-1])


def row_classes(walks, art_class, art_world, world_count, joints, contact_body, mimic_dofs=None) -> SupportTable | None:
    """The rows' support classes (:data:`ROW_CLASSES`), ``None`` past 16-bit ids: a side lives on its seeds' root
    paths. A body's endpoint is its path in its walking class (``walks``' ``(parent, first)``, by ``art_class``); two
    seeds on one tree (``contact_body`` pairs, ``mimic_dofs``) take their union. The widest row: two trees' widest
    contact paths in a world, or one side's widest class. ``joints``: the articulations' first and end joints, each
    joint's child and first coordinate."""
    paths = []
    for parent, first in walks:
        held: list[tuple[int, ...]] = []
        for b, up in enumerate(parent):
            held.append((held[up] if up >= 0 else ()) + tuple(range(first[b], first[b + 1])))
        paths.append(held)
    base = host_index.offsets([len(p) for p in paths])
    endpoints = max(int(base[-1]), 1)
    classes: dict[tuple[int, tuple[int, ...]], int] = {}
    pair = np.full((endpoints, endpoints), -1, dtype=np.int32)
    for t, held in enumerate(paths):
        for b, path in enumerate(held):
            pair[base[t] + b, base[t] + b] = classes.setdefault((t, path), len(classes))

    def union(e0: int, e1: int) -> None:
        t = int(np.searchsorted(base, e0, side="right")) - 1
        path = tuple(sorted(set(paths[t][e0 - base[t]]) | set(paths[t][e1 - base[t]])))
        pair[e0, e1] = pair[e1, e0] = classes.setdefault((t, path), len(classes))

    start, end, child, qd_start = joints
    arts = np.flatnonzero(art_class >= 0)
    art, local = host_index.expand(end[arts] - start[arts])
    joint = start[arts][art] + local
    endpoint = base[art_class[arts][art]] + local
    body_class = np.full(max(contact_body.size, 1), -1, dtype=np.int32)
    held = child[joint] >= 0
    body_class[child[joint][held]] = endpoint[held]
    at, rank = host_index.expand(qd_start[joint + 1] - qd_start[joint])
    limit_class = np.full(max(int(qd_start[-1]), 1), -1, dtype=np.int32)
    limit_class[qd_start[joint][at] + rank] = endpoint[at]
    contacts = np.unique(body_class[: contact_body.size][contact_body & (body_class[: contact_body.size] >= 0)])
    owner = np.searchsorted(base, contacts, side="right") - 1
    widest = np.zeros(len(paths), dtype=np.int64)
    for t in range(len(paths)):
        mine = contacts[owner == t].tolist()
        widest[t] = max((len(paths[t][e - base[t]]) for e in mine), default=0)
        for i, e0 in enumerate(mine):
            for e1 in mine[i + 1 :]:
                union(e0, e1)
    if mimic_dofs is not None:
        for d0, d1 in zip(*(np.asarray(d).tolist() for d in mimic_dofs), strict=True):
            e0, e1 = int(limit_class[d0]), int(limit_class[d1])
            if e0 >= 0 and e1 >= 0 and pair[e0, e1] < 0:
                union(e0, e1)
    if len(classes) >= NO_CLASS:
        return None
    entries = max((len(path) for _, path in classes), default=1)
    class_dof = np.full((max(len(classes), 1), max(entries, 1)), -1, dtype=np.int32)
    class_count = np.zeros(max(len(classes), 1), dtype=np.int32)
    for (_, path), k in classes.items():
        class_dof[k, : len(path)] = path
        class_count[k] = len(path)
    art_widest = np.zeros(art_class.size, dtype=np.int64)
    art_widest[arts] = widest[art_class[arts]]
    two = support_width(art_widest, art_world, world_count)
    return SupportTable(body_class, limit_class, pair, class_dof, class_count, max(entries, two, 1))

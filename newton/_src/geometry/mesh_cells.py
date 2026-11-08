# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""An exact multi-level sparse cell index of a triangle mesh, for the mesh-vs-convex midphase.

Each mesh gets one index in its unscaled local frame (the frame of its vertices), so static and moving meshes and
every shape scale share it. Space is cut into cubic cells of edge ``h * 2**l`` at level ``l``. A triangle sits at the
finest level where its bounding box covers at most :data:`MAX_BOX_CELLS` cells, and registers in the cells of that
box that its surface meets (an exact, closed triangle-box test in float64). Every triangle therefore costs at most
:data:`MAX_BOX_CELLS` entries whatever its size, so the index is O(triangles): a 100 m ground made of two triangles, a
1 m wall and a 0.1 m lattice face each take a few. A long, thin stair tread stays in cells about its width, not its
length. Floors, ceilings, walls and rooms are cells like any other: nothing assumes a height function.

Only occupied cells are stored. Their keys ``(level, x, y, z)`` sit in an open-addressing hash table (linear probing,
load at most one half), each slot holding the key and the cell's ``(start, count)`` into a CSR of entries. An entry is
a triangle id with the triangle's bounding box relative to the cell (16-bit fixed point, rounded outward), so a query
rejects most entries without touching the triangle. Each triangle keeps its vertices and its *registration*: its
box's base cell and spans and a mask of the box cells it registered in. A query that meets a triangle in several
cells emits it from the first registered one inside the query's cell range only (the reference-cell rule, by the
mask: no memory).

A query visits, at every occupied level, the cells under its bound's box (level-0 cell coordinates shifted right by
``l``: ``floor(x / (h 2^l)) = floor(floor(x / h) / 2^l)``, the same identity the build uses). A point of the bound
that lies on a triangle lies in a visited cell where that triangle is registered, so no triangle that meets the bound
is missed. The box is padded by more than the float32 rounding of the cell coordinates. A level whose occupied cells
are fewer than the cells of the range walks those cells instead of looking the range up.

The index is built once on the host from the mesh's own vertex buffers (:func:`build_mesh_cells`) and never changes:
it is the rigid mesh's, in the mesh's own frame, like its vertices.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .types import GeoType

# Keys pack (level, x, y, z) as 4 + 3 * 20 bits. The level-0 grid is kept under 2**14 cells per axis (the cell edge
# grows for larger meshes), so the top level holds the whole mesh in one cell and no key is all ones.
_LEVEL_BITS = 4
_MAX_LEVEL = (1 << _LEVEL_BITS) - 1
_MAX_AXIS_CELLS = 1 << 14

MAX_BOX_CELLS = 32
"""Most cells a triangle's bounding box may cover at its level (the registration mask's width)."""

MESH_CELLS_EMPTY_KEY = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))
_HASH_MULTIPLIER = wp.constant(wp.uint64(0xFF51AFD7ED558CCD))
_KEY_SHIFT_LEVEL = wp.constant(wp.uint64(60))
_KEY_SHIFT_X = wp.constant(wp.uint64(40))
_KEY_SHIFT_Y = wp.constant(wp.uint64(20))
_KEY_AXIS_MASK = wp.constant(wp.uint64(0xFFFFF))
_HASH_SHIFT = wp.constant(wp.uint64(33))

# The grid origin sits a fraction of a cell below the mesh's lower corner, so lattice-aligned data (a terrain on a
# 0.1 m lattice, integer room coordinates) does not land on cell faces, where it would register on both sides.
_ORIGIN_OFFSET = 0.3183098861837907

# Relative padding of the build's closed cell boxes (float64 rounding only).
_BUILD_PAD = 1.0e-9

# Entry boxes: 16-bit coordinates relative to the cell's lower corner, in cell / 512 (a triangle's box spans at most
# MAX_BOX_CELLS = 32 cells, well inside the +-64 cells they reach).
_BOX_STEPS = 512.0
MESH_CELLS_BOX_STEPS = wp.constant(_BOX_STEPS)


@wp.struct
class MeshGrid:
    """One mesh's cell index, in the mesh's unscaled local frame."""

    origin: wp.vec3
    """Lower corner of cell (0, 0, 0) [m]."""
    cell: float
    """Level-0 cell edge ``h`` [m]; level ``l`` cells have edge ``h * 2**l``."""
    inv_cell: float
    """``1 / h`` [1/m]."""
    levels: int
    """Bit ``l`` is set when level ``l`` holds a triangle."""
    dims: wp.vec3i
    """Level-0 cell count per axis."""
    table_start: int
    """Start of this mesh's hash slots in :attr:`MeshCells.slots`."""
    table_mask: int
    """Slot count minus one (a power of two minus one)."""
    triangle_start: int
    """Start of this mesh's triangles in the per-triangle arrays."""
    level_start: int
    """Start of this mesh's 17 level offsets in :attr:`MeshCells.level_start`."""


@wp.struct
class MeshCellSlot:
    """A hash slot (or a sorted cell): the cell's key, its entries and the box of their triangles."""

    key: wp.uint64
    """The cell key, or :data:`MESH_CELLS_EMPTY_KEY` for an empty slot."""
    start: int
    """The cell's first entry in :attr:`MeshCells.entries`."""
    count: int
    """The cell's entry count."""
    box_lo: int
    """Lower corner of the union of the cell's triangle boxes, relative to the cell in cell / 32, 8 bits a coordinate
    (``128 + v``; 0 stands for any lower value)."""
    box_hi: int
    """Upper corner, likewise (255 stands for any higher value)."""


@wp.struct
class MeshCellTriangle:
    """One triangle as the midphase reads it: its vertices and its registration."""

    v0: wp.vec3
    """First vertex, unscaled mesh frame [m]."""
    v1: wp.vec3
    """Second vertex [m]."""
    v2: wp.vec3
    """Third vertex [m]."""
    registration: wp.vec4i
    """``(mask, x | sx << 20, y | sy << 20, z | sz << 20)``: the base cell ``(x, y, z)`` and spans of the
    triangle's box at its level, and bit ``dx + (sx + 1) (dy + (sy + 1) dz)`` set for each box cell it registered
    in."""


@wp.struct
class MeshCells:
    """The cell indices of every collision mesh of a scene, packed, and the shape-to-index map."""

    grids: wp.array[MeshGrid]
    """One per indexed mesh."""
    shape_grid: wp.array[wp.int32]
    """Per shape: its mesh's index in :attr:`grids`, or -1."""
    slots: wp.array[MeshCellSlot]
    """The hash slots of every mesh."""
    entries: wp.array[wp.vec4i]
    """Per cell, by ascending triangle: ``(triangle, lo_x | lo_y << 16, lo_z | hi_x << 16, hi_y | hi_z << 16)``, the
    triangle id (local to the mesh) and its box relative to the cell's lower corner, each ``32768 + 512 x / edge``."""
    triangles: wp.array[MeshCellTriangle]
    """Per triangle, by mesh: its vertices and registration."""
    sorted_slots: wp.array[MeshCellSlot]
    """Every mesh's occupied cells, by ascending key (so grouped by level)."""
    level_start: wp.array[wp.int32]
    """Per mesh, 17 offsets: level ``l``'s cells are ``sorted_slots[level_start[g + l] : level_start[g + l + 1]]``."""


# ----------------------------------------------------------------------------------------------------------------
# Device functions


@wp.func
def mesh_cells_key(level: int, x: int, y: int, z: int) -> wp.uint64:
    """The hash key of cell ``(x, y, z)`` at ``level``."""
    return (
        (wp.uint64(level) << _KEY_SHIFT_LEVEL)
        | (wp.uint64(x) << _KEY_SHIFT_X)
        | (wp.uint64(y) << _KEY_SHIFT_Y)
        | wp.uint64(z)
    )


@wp.func
def mesh_cells_key_cell(key: wp.uint64) -> wp.vec3i:
    """The cell coordinates of a key."""
    return wp.vec3i(
        int((key >> _KEY_SHIFT_X) & _KEY_AXIS_MASK),
        int((key >> _KEY_SHIFT_Y) & _KEY_AXIS_MASK),
        int(key & _KEY_AXIS_MASK),
    )


@wp.func
def _mesh_cells_hash(key: wp.uint64) -> wp.uint64:
    h = key ^ (key >> _HASH_SHIFT)
    h = h * _HASH_MULTIPLIER
    return h ^ (h >> _HASH_SHIFT)


@wp.func
def mesh_cells_find(cells: MeshCells, grid: MeshGrid, key: wp.uint64) -> MeshCellSlot:
    """Cell ``key``'s slot in ``grid``; its ``count`` is zero when the cell is empty."""
    mask = grid.table_mask
    slot = int(_mesh_cells_hash(key) & wp.uint64(mask))
    empty = MeshCellSlot()
    empty.count = 0
    for _probe in range(mask + 1):
        stored = cells.slots[grid.table_start + slot]
        if stored.key == key:
            return stored
        if stored.key == MESH_CELLS_EMPTY_KEY:
            return empty
        slot = (slot + 1) & mask
    return empty


@wp.func
def mesh_cells_slot_overlaps(
    slot: MeshCellSlot, lower: wp.vec3, upper: wp.vec3, cell_lower: wp.vec3, inv_size: float
) -> bool:
    """Whether the box ``[lower, upper]`` overlaps the union of the cell's triangle boxes."""
    for k in range(3):
        lo = wp.clamp(int(wp.floor((lower[k] - cell_lower[k]) * inv_size * 32.0)), -128, 127) + 128
        hi = wp.clamp(int(wp.floor((upper[k] - cell_lower[k]) * inv_size * 32.0)), -128, 127) + 128
        if hi < ((slot.box_lo >> (8 * k)) & 255) or lo > ((slot.box_hi >> (8 * k)) & 255):
            return False
    return True


@wp.func
def mesh_cells_candidate(pair: int, triangle: int, level: int, cell: wp.vec3i) -> wp.vec4i:
    """A midphase candidate: the pair, the triangle and the cell it was met in."""
    return wp.vec4i(pair, triangle, cell[0] | (cell[1] << 15), cell[2] | (level << 15))

@wp.func
def mesh_cells_box_in_cell(
    lower: wp.vec3, upper: wp.vec3, cell_lower: wp.vec3, inv_size: float
) -> tuple[wp.vec3i, wp.vec3i]:
    """A box on a cell's entry-box lattice (clamped to its 16-bit range)."""
    lo = wp.vec3i(0)
    hi = wp.vec3i(0)
    for k in range(3):
        lo[k] = wp.clamp(int(wp.floor((lower[k] - cell_lower[k]) * inv_size * MESH_CELLS_BOX_STEPS)), -32768, 32767)
        hi[k] = wp.clamp(int(wp.floor((upper[k] - cell_lower[k]) * inv_size * MESH_CELLS_BOX_STEPS)), -32768, 32767)
    return lo, hi


@wp.func
def mesh_cells_entry_overlaps(entry: wp.vec4i, lo: wp.vec3i, hi: wp.vec3i) -> bool:
    """Whether an entry's box overlaps the query box ``[lo, hi]`` on the cell's lattice."""
    if hi[0] < (entry[1] & 0xFFFF) - 32768 or lo[0] > ((entry[2] >> 16) & 0xFFFF) - 32768:
        return False
    if hi[1] < ((entry[1] >> 16) & 0xFFFF) - 32768 or lo[1] > (entry[3] & 0xFFFF) - 32768:
        return False
    return hi[2] >= (entry[2] & 0xFFFF) - 32768 and lo[2] <= ((entry[3] >> 16) & 0xFFFF) - 32768


@wp.func
def mesh_cells_range(grid: MeshGrid, lower: wp.vec3, upper: wp.vec3) -> tuple[wp.vec3i, wp.vec3i, bool]:
    """Level-0 cell range ``[lo, hi]`` under the box ``[lower, upper]`` (unscaled mesh frame), clamped to the grid.

    Returns ``False`` when the box misses the grid. The caller pads the box for float32 rounding.
    """
    lo = wp.vec3i(0)
    hi = wp.vec3i(0)
    inside = bool(True)
    for k in range(3):
        a = (lower[k] - grid.origin[k]) * grid.inv_cell
        b = (upper[k] - grid.origin[k]) * grid.inv_cell
        n = grid.dims[k]
        if b < 0.0 or a >= float(n):
            inside = False
        lo[k] = wp.clamp(int(wp.floor(a)), 0, n - 1)
        hi[k] = wp.clamp(int(wp.floor(b)), 0, n - 1)
    return lo, hi, inside


@wp.func
def mesh_cells_first_in_range(registration: wp.vec4i, cell: wp.vec3i, qlo: wp.vec3i, qhi: wp.vec3i) -> bool:
    """Whether ``cell`` is its triangle's first registered cell inside the query range ``[qlo, qhi]``.

    ``registration`` is the triangle's :attr:`MeshCellTriangle.registration`; box cells are ordered by mask bit.
    """
    bx = registration[1] & 0xFFFFF
    by = registration[2] & 0xFFFFF
    bz = registration[3] & 0xFFFFF
    nx = (registration[1] >> 20) + 1
    ny = (registration[2] >> 20) + 1
    bit = (cell[0] - bx) + nx * ((cell[1] - by) + ny * (cell[2] - bz))
    below = wp.int64(registration[0]) & wp.int64(0xFFFFFFFF) & ((wp.int64(1) << wp.int64(bit)) - wp.int64(1))
    if below == wp.int64(0):
        return True
    # The query range inside the box, as runs of bits along x.
    x0 = wp.max(qlo[0], bx) - bx
    x1 = wp.min(qhi[0], bx + nx - 1) - bx
    run = ((wp.int64(1) << wp.int64(x1 - x0 + 1)) - wp.int64(1)) << wp.int64(x0)
    inside = wp.int64(0)
    for z in range(wp.max(qlo[2], bz) - bz, wp.min(qhi[2], bz + (registration[3] >> 20)) - bz + 1):
        for y in range(wp.max(qlo[1], by) - by, wp.min(qhi[1], by + ny - 1) - by + 1):
            inside = inside | (run << wp.int64(nx * (y + ny * z)))
    return (below & inside) == wp.int64(0)


@wp.func
def mesh_cells_boxes_overlap(lower_a: wp.vec3, upper_a: wp.vec3, lower_b: wp.vec3, upper_b: wp.vec3) -> bool:
    """Whether two closed boxes overlap."""
    return (
        lower_a[0] <= upper_b[0]
        and lower_b[0] <= upper_a[0]
        and lower_a[1] <= upper_b[1]
        and lower_b[1] <= upper_a[1]
        and lower_a[2] <= upper_b[2]
        and lower_b[2] <= upper_a[2]
    )


@wp.struct
class MeshQueryBound:
    """A convex shape's bound in the scaled mesh frame, inflated by the pair's contact threshold.

    An oriented box always; for shapes with a symmetry axis (sphere, capsule, cylinder, cone) also a segment with a
    radius. A triangle is a candidate when it meets both.
    """

    center: wp.vec3
    axes: wp.mat33  # columns: the box axes
    half: wp.vec3
    seg_a: wp.vec3
    seg_b: wp.vec3
    radius: float
    has_segment: bool


@wp.func
def _axis_separates(axis: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, half: wp.vec3) -> bool:
    d0 = wp.dot(axis, p0)
    d1 = wp.dot(axis, p1)
    d2 = wp.dot(axis, p2)
    r = half[0] * wp.abs(axis[0]) + half[1] * wp.abs(axis[1]) + half[2] * wp.abs(axis[2])
    return wp.min(wp.min(d0, d1), d2) > r or wp.max(wp.max(d0, d1), d2) < -r


@wp.func
def triangle_meets_box(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, half: wp.vec3) -> bool:
    """Separating-axis test of a triangle (box frame) against the closed box ``[-half, half]``."""
    for k in range(3):
        if wp.min(wp.min(p0[k], p1[k]), p2[k]) > half[k] or wp.max(wp.max(p0[k], p1[k]), p2[k]) < -half[k]:
            return False
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2
    if _axis_separates(wp.cross(e0, e1), p0, p1, p2, half):
        return False
    for k in range(3):
        u = wp.vec3(0.0)
        u[k] = 1.0
        if _axis_separates(wp.cross(u, e0), p0, p1, p2, half):
            return False
        if _axis_separates(wp.cross(u, e1), p0, p1, p2, half):
            return False
        if _axis_separates(wp.cross(u, e2), p0, p1, p2, half):
            return False
    return True


@wp.func
def _segment_segment_distance_sq(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3) -> float:
    """Squared distance between segments ``p1q1`` and ``p2q2`` (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    eps = 1.0e-20
    s = float(0.0)
    t = float(0.0)
    if a <= eps and e <= eps:
        return wp.dot(r, r)
    if a <= eps:
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= eps:
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denom = a * e - b * b
            if denom > eps * a * e:
                s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)
    d = (p1 + d1 * s) - (p2 + d2 * t)
    return wp.dot(d, d)


@wp.func
def _point_triangle_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    # Closest point by Voronoi regions (Ericson 5.1.5); degenerate triangles reduce to their edges.
    ab = b - a
    ac = c - a
    n = wp.cross(ab, ac)
    if wp.dot(n, n) <= 1.0e-30 * (wp.dot(ab, ab) * wp.dot(ac, ac) + 1.0e-30):
        d = wp.min(_segment_segment_distance_sq(p, p, a, b), _segment_segment_distance_sq(p, p, b, c))
        return wp.min(d, _segment_segment_distance_sq(p, p, c, a))
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.dot(ap, ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.dot(bp, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        q = a + ab * (d1 / (d1 - d3))
        return wp.length_sq(p - q)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.dot(cp, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        q = a + ac * (d2 / (d2 - d6))
        return wp.length_sq(p - q)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        q = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        return wp.length_sq(p - q)
    denom = 1.0 / (va + vb + vc)
    q = a + ab * (vb * denom) + ac * (vc * denom)
    return wp.length_sq(p - q)


@wp.func
def segment_triangle_distance_sq(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    """Squared distance between segment ``pq`` and triangle ``abc``.

    The closest pair has an endpoint of the segment (point-triangle), or a triangle edge (segment-segment), or the
    segment pierces the triangle (zero).
    """
    n = wp.cross(b - a, c - a)
    sp = wp.dot(n, p - a)
    sq = wp.dot(n, q - a)
    if (sp <= 0.0 and sq >= 0.0) or (sp >= 0.0 and sq <= 0.0):
        denom = sp - sq
        if denom != 0.0:
            x = p + (q - p) * (sp / denom)
            # Inside test by edge orientations against the normal.
            if (
                wp.dot(wp.cross(b - a, x - a), n) >= 0.0
                and wp.dot(wp.cross(c - b, x - b), n) >= 0.0
                and wp.dot(wp.cross(a - c, x - c), n) >= 0.0
            ):
                return 0.0
    d = wp.min(_point_triangle_distance_sq(p, a, b, c), _point_triangle_distance_sq(q, a, b, c))
    d = wp.min(d, _segment_segment_distance_sq(p, q, a, b))
    d = wp.min(d, _segment_segment_distance_sq(p, q, b, c))
    return wp.min(d, _segment_segment_distance_sq(p, q, c, a))


@wp.func
def mesh_query_bound_meets_triangle(bound: MeshQueryBound, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> bool:
    """Whether triangle ``v0 v1 v2`` (scaled mesh frame) meets the bound: the box, and the segment's capsule."""
    rt = wp.transpose(bound.axes)
    if not triangle_meets_box(rt * (v0 - bound.center), rt * (v1 - bound.center), rt * (v2 - bound.center), bound.half):
        return False
    if bound.has_segment:
        return segment_triangle_distance_sq(bound.seg_a, bound.seg_b, v0, v1, v2) <= bound.radius * bound.radius
    return True


@wp.func
def mesh_query_bound_box(bound: MeshQueryBound) -> tuple[wp.vec3, wp.vec3]:
    """Axis-aligned box of the bound (scaled mesh frame): the box's, cut by the segment capsule's."""
    a = bound.axes
    ext = wp.vec3(
        wp.abs(a[0, 0]) * bound.half[0] + wp.abs(a[0, 1]) * bound.half[1] + wp.abs(a[0, 2]) * bound.half[2],
        wp.abs(a[1, 0]) * bound.half[0] + wp.abs(a[1, 1]) * bound.half[1] + wp.abs(a[1, 2]) * bound.half[2],
        wp.abs(a[2, 0]) * bound.half[0] + wp.abs(a[2, 1]) * bound.half[1] + wp.abs(a[2, 2]) * bound.half[2],
    )
    lower = bound.center - ext
    upper = bound.center + ext
    if bound.has_segment:
        r = wp.vec3(bound.radius)
        lower = wp.max(lower, wp.min(bound.seg_a, bound.seg_b) - r)
        upper = wp.min(upper, wp.max(bound.seg_a, bound.seg_b) + r)
    return lower, upper


# ----------------------------------------------------------------------------------------------------------------
# Host build


def _hash_np(keys: np.ndarray) -> np.ndarray:
    h = keys ^ (keys >> np.uint64(33))
    with np.errstate(over="ignore"):
        h = h * np.uint64(0xFF51AFD7ED558CCD)
    return h ^ (h >> np.uint64(33))


def _triangles_meet_boxes(a, b, c, center, half) -> np.ndarray:
    """Vectorized separating-axis test of triangles against closed boxes (float64), rows paired."""
    p0, p1, p2 = a - center, b - center, c - center
    lo = np.minimum(np.minimum(p0, p1), p2)
    hi = np.maximum(np.maximum(p0, p1), p2)
    hit = np.all((lo <= half) & (hi >= -half), axis=1)

    def separates(axis):
        d0 = np.einsum("ij,ij->i", axis, p0)
        d1 = np.einsum("ij,ij->i", axis, p1)
        d2 = np.einsum("ij,ij->i", axis, p2)
        r = np.einsum("ij,ij->i", np.abs(axis), half)
        return (np.minimum(np.minimum(d0, d1), d2) > r) | (np.maximum(np.maximum(d0, d1), d2) < -r)

    e = (p1 - p0, p2 - p1, p0 - p2)
    hit &= ~separates(np.cross(e[0], e[1]))
    for k in range(3):
        u = np.zeros(3)
        u[k] = 1.0
        for ej in e:
            hit &= ~separates(np.cross(u, ej))
    return hit


def _slots(keys, starts, counts, box_lo=None, box_hi=None) -> np.ndarray:
    out = np.zeros(keys.size, dtype=MeshCellSlot.numpy_dtype())
    out["key"], out["start"], out["count"] = keys, starts, counts
    out["box_lo"] = 0 if box_lo is None else box_lo
    out["box_hi"] = 0xFFFFFF if box_hi is None else box_hi
    return out


def default_cell_size(points: np.ndarray, indices: np.ndarray) -> float:
    """The level-0 cell edge chosen from the triangles: twice their median bounding-box span [m].

    Most triangles then sit at level 0 inside one or two cells per axis; larger ones climb levels.
    """
    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if tri.shape[0] == 0:
        return 1.0
    p = np.asarray(points, dtype=np.float64)
    v = p[tri]
    span = (v.max(axis=1) - v.min(axis=1)).max(axis=1)
    h = 2.0 * float(np.median(span))
    return h if h > 0.0 else 1.0


def build_mesh_grid(points: np.ndarray, indices: np.ndarray, cell: float | None = None) -> dict:
    """Build one mesh's cell index on the host.

    Args:
        points: Vertex positions in the mesh's unscaled frame, shape ``(n, 3)`` [m].
        indices: Triangle vertex indices, flat or ``(t, 3)``.
        cell: Level-0 cell edge [m]; :func:`default_cell_size` when ``None``. Grown so the grid stays under
            ``2**14`` cells per axis.

    Returns:
        A dict of the grid's fields and arrays, and statistics.
    """
    p32 = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    nt = tri.shape[0]
    if cell is None:
        cell = default_cell_size(p32, tri)
    if nt == 0:
        return {
            "origin": np.zeros(3, dtype=np.float32),
            "cell": np.float32(1.0),
            "levels": 0,
            "dims": np.ones(3, dtype=np.int32),
            "slots": _slots(np.full(1, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64), np.zeros(1), np.zeros(1)),
            "entries": np.zeros((0, 4), dtype=np.int32),
            "triangles": np.zeros(0, dtype=MeshCellTriangle.numpy_dtype()),
            "sorted_slots": np.zeros(0, dtype=MeshCellSlot.numpy_dtype()),
            "level_start": np.zeros(_MAX_LEVEL + 2, dtype=np.int32),
            "stats": {"triangles": 0, "cells": 0, "entries": 0, "levels": {}},
        }
    v32 = p32[tri]  # (t, 3, 3)
    a, b, c = (v32[:, i].astype(np.float64) for i in range(3))
    tmin = v32.min(axis=1).astype(np.float64)
    tmax = v32.max(axis=1).astype(np.float64)
    vmin, vmax = tmin.min(axis=0), tmax.max(axis=0)
    extent = float((vmax - vmin).max())
    h = float(np.float32(max(float(cell), extent / (_MAX_AXIS_CELLS - 2), 1.0e-6)))
    origin = (vmin - _ORIGIN_OFFSET * h).astype(np.float32)
    o = origin.astype(np.float64)
    # Level-0 integer cell coordinates of each triangle's box; level l divides them by 2**l.
    dims = (np.floor((vmax - o) / h).astype(np.int64) + 1).astype(np.int32)
    lo0 = np.clip(np.floor((tmin - o) / h).astype(np.int64), 0, dims.astype(np.int64) - 1)
    hi0 = np.clip(np.floor((tmax - o) / h).astype(np.int64), 0, dims.astype(np.int64) - 1)

    level = np.full(nt, -1, dtype=np.int64)
    for lvl in range(_MAX_LEVEL + 1):
        open_ = np.flatnonzero(level < 0)
        if open_.size == 0:
            break
        span = (hi0[open_] >> lvl) - (lo0[open_] >> lvl) + 1
        level[open_[span.prod(axis=1) <= MAX_BOX_CELLS]] = lvl
    assert (level >= 0).all(), "the top level holds every triangle"

    base = lo0 >> level[:, None]
    span = (hi0 >> level[:, None]) - base  # per axis, cells - 1
    mask = np.zeros(nt, dtype=np.int64)
    keys_chunks, tri_chunks = [], []
    level_counts = {int(lv): int(cnt) for lv, cnt in zip(*np.unique(level, return_counts=True), strict=True)}
    # Every box offset (dx, dy, dz) a triangle's box can hold.
    offsets = [
        (dx, dy, dz)
        for dz in range(MAX_BOX_CELLS)
        for dy in range(MAX_BOX_CELLS)
        for dx in range(MAX_BOX_CELLS)
        if (dx + 1) * (dy + 1) * (dz + 1) <= MAX_BOX_CELLS
    ]
    for dx, dy, dz in offsets:
        d = np.array([dx, dy, dz], dtype=np.int64)
        t = np.flatnonzero(np.all(span >= d, axis=1))
        if t.size == 0:
            continue
        k = base[t] + d
        size = h * np.left_shift(1, level[t]).astype(np.float64)
        center = o + (k + 0.5) * size[:, None]
        half = np.repeat((0.5 * size * (1.0 + _BUILD_PAD))[:, None], 3, axis=1)
        hit = _triangles_meet_boxes(a[t], b[t], c[t], center, half)
        t, k = t[hit], k[hit]
        bit = dx + (span[t, 0] + 1) * (dy + (span[t, 1] + 1) * dz)
        mask[t] |= np.left_shift(1, bit)
        keys_chunks.append(
            (level[t].astype(np.uint64) << np.uint64(60))
            | (k[:, 0].astype(np.uint64) << np.uint64(40))
            | (k[:, 1].astype(np.uint64) << np.uint64(20))
            | k[:, 2].astype(np.uint64)
        )
        tri_chunks.append(t)
    # Rounding can only lose a touching cell; a triangle that met none keeps every cell of its box.
    lost = np.flatnonzero(mask == 0)
    for dx, dy, dz in offsets:
        if lost.size == 0:
            break
        d = np.array([dx, dy, dz], dtype=np.int64)
        t = lost[np.all(span[lost] >= d, axis=1)]
        if t.size == 0:
            continue
        k = base[t] + d
        bit = dx + (span[t, 0] + 1) * (dy + (span[t, 1] + 1) * dz)
        mask[t] |= np.left_shift(1, bit)
        keys_chunks.append(
            (level[t].astype(np.uint64) << np.uint64(60))
            | (k[:, 0].astype(np.uint64) << np.uint64(40))
            | (k[:, 1].astype(np.uint64) << np.uint64(20))
            | k[:, 2].astype(np.uint64)
        )
        tri_chunks.append(t)

    keys = np.concatenate(keys_chunks)
    tris = np.concatenate(tri_chunks)
    order = np.lexsort((tris, keys))
    keys, tris = keys[order], tris[order]
    # Each entry's triangle box relative to its cell, rounded outward by a step.
    entry_level = (keys >> np.uint64(60)).astype(np.int64)
    entry_cell = np.stack(
        [((keys >> np.uint64(s)) & np.uint64(0xFFFFF)).astype(np.int64) for s in (40, 20, 0)], axis=1
    )
    entry_size = h * np.left_shift(1, entry_level).astype(np.float64)
    cell_lo = o + entry_cell * entry_size[:, None]
    q_lo = np.floor((tmin[tris] - cell_lo) / entry_size[:, None] * _BOX_STEPS) - 1
    q_hi = np.ceil((tmax[tris] - cell_lo) / entry_size[:, None] * _BOX_STEPS) + 1
    q_lo = (np.clip(q_lo, -32768, 32767) + 32768).astype(np.int64)
    q_hi = (np.clip(q_hi, -32768, 32767) + 32768).astype(np.int64)

    def pack(a_, b_):
        return (a_ | (b_ << 16)).astype(np.uint32).view(np.int32)

    # Each cell's union of triangle boxes, in cell / 32 (saturating), for the slot.
    u_lo = np.clip(np.floor((tmin[tris] - cell_lo) / entry_size[:, None] * 32.0) - 1, -128, 127).astype(np.int64)
    u_hi = np.clip(np.ceil((tmax[tris] - cell_lo) / entry_size[:, None] * 32.0) + 1, -128, 127).astype(np.int64)

    entries = np.stack(
        [tris.astype(np.int32), pack(q_lo[:, 0], q_lo[:, 1]), pack(q_lo[:, 2], q_hi[:, 0]), pack(q_hi[:, 1], q_hi[:, 2])],
        axis=1,
    )

    cell_keys, starts, counts = np.unique(keys, return_index=True, return_counts=True)
    ncell = cell_keys.size
    c_lo = np.minimum.reduceat(u_lo, starts, axis=0) + 128
    c_hi = np.maximum.reduceat(u_hi, starts, axis=0) + 128
    cell_box_lo = c_lo[:, 0] | (c_lo[:, 1] << 8) | (c_lo[:, 2] << 16)
    cell_box_hi = c_hi[:, 0] | (c_hi[:, 1] << 8) | (c_hi[:, 2] << 16)
    capacity = 1
    while capacity < 2 * ncell:
        capacity <<= 1
    table_mask = capacity - 1
    table_keys = np.full(capacity, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
    table_cells = np.zeros((capacity, 4), dtype=np.int64)
    table_cells[:, 3] = 0xFFFFFF
    slot = (_hash_np(cell_keys) & np.uint64(table_mask)).astype(np.int64)
    pending = np.arange(ncell)
    while pending.size:
        s = slot[pending]
        free = table_keys[s] == np.uint64(0xFFFFFFFFFFFFFFFF)
        cand, cs = pending[free], s[free]
        _, first = np.unique(cs, return_index=True)
        win = cand[first]
        table_keys[slot[win]] = cell_keys[win]
        table_cells[slot[win], 0] = starts[win]
        table_cells[slot[win], 1] = counts[win]
        table_cells[slot[win], 2] = cell_box_lo[win]
        table_cells[slot[win], 3] = cell_box_hi[win]
        placed = np.zeros(ncell, dtype=bool)
        placed[win] = True
        pending = pending[~placed[pending]]
        slot[pending] = (slot[pending] + 1) & table_mask

    mask32 = mask.astype(np.uint32).view(np.int32)
    registration = np.stack(
        [mask32, base[:, 0] | (span[:, 0] << 20), base[:, 1] | (span[:, 1] << 20), base[:, 2] | (span[:, 2] << 20)],
        axis=1,
    ).astype(np.int32)
    levels = 0
    for lvl in level_counts:
        levels |= 1 << lvl
    records = np.zeros(nt, dtype=MeshCellTriangle.numpy_dtype())
    records["v0"], records["v1"], records["v2"] = v32[:, 0], v32[:, 1], v32[:, 2]
    records["registration"] = registration
    return {
        "origin": origin,
        "cell": np.float32(h),
        "levels": levels,
        "dims": dims,
        "slots": _slots(table_keys, table_cells[:, 0], table_cells[:, 1], table_cells[:, 2], table_cells[:, 3]),
        "entries": entries,
        "triangles": records,
        "sorted_slots": _slots(cell_keys, starts, counts, cell_box_lo, cell_box_hi),
        "level_start": np.searchsorted(
            cell_keys >> np.uint64(60), np.arange(_MAX_LEVEL + 2, dtype=np.uint64), side="left"
        ).astype(np.int32),
        "stats": {
            "triangles": int(nt),
            "cells": int(ncell),
            "entries": int(entries.shape[0]),
            "slots": int(capacity),
            "levels": level_counts,
            "entries_per_cell_mean": float(counts.mean()),
            "entries_per_cell_p99": float(np.percentile(counts, 99)),
            "entries_per_cell_max": int(counts.max()),
        },
    }


@wp.kernel(enable_backward=False)
def _mesh_sizes_kernel(mesh_ids: wp.array[wp.uint64], sizes: wp.array[wp.vec2i]):
    i = wp.tid()
    mesh = wp.mesh_get(mesh_ids[i])
    sizes[i] = wp.vec2i(mesh.points.shape[0], mesh.indices.shape[0])


@wp.kernel(enable_backward=False)
def _mesh_copy_kernel(mesh_id: wp.uint64, points: wp.array[wp.vec3], indices: wp.array[wp.int32]):
    i = wp.tid()
    mesh = wp.mesh_get(mesh_id)
    if i < mesh.points.shape[0]:
        points[i] = mesh.points[i]
    if i < mesh.indices.shape[0]:
        indices[i] = mesh.indices[i]


def _read_meshes(mesh_ids: list[int], device) -> list[tuple[np.ndarray, np.ndarray]]:
    """The vertex and index buffers the collision kernels read, fetched by mesh id."""
    if not mesh_ids:
        return []
    ids = wp.array(np.asarray(mesh_ids, dtype=np.uint64), dtype=wp.uint64, device=device)
    sizes = wp.zeros(len(mesh_ids), dtype=wp.vec2i, device=device)
    wp.launch(_mesh_sizes_kernel, dim=len(mesh_ids), inputs=[ids], outputs=[sizes], device=device, record_tape=False)
    out = []
    for mesh_id, (npoints, nindices) in zip(mesh_ids, sizes.numpy(), strict=True):
        points = wp.empty(int(npoints), dtype=wp.vec3, device=device)
        indices = wp.empty(int(nindices), dtype=wp.int32, device=device)
        wp.launch(
            _mesh_copy_kernel,
            dim=int(max(npoints, nindices, 1)),
            inputs=[wp.uint64(mesh_id)],
            outputs=[points, indices],
            device=device,
            record_tape=False,
        )
        out.append((points.numpy(), indices.numpy()))
    return out


def build_mesh_cells(
    shape_type,
    shape_source_ptr,
    device=None,
    *,
    shape_flags=None,
    cell: float | None = None,
) -> MeshCells:
    """Build the cell index of every triangle mesh shape (``GeoType.MESH``), one per distinct mesh.

    Args:
        shape_type: Per-shape :class:`GeoType` values (host sequence or array).
        shape_source_ptr: Per-shape source pointers; a mesh shape's is its ``wp.Mesh`` id.
        device: Device of the returned arrays (and of the meshes).
        shape_flags: Optional per-shape flags; when given, only shapes that collide with shapes are indexed.
        cell: Level-0 cell edge for every mesh [m]; chosen per mesh from its triangles when ``None``.

    Returns:
        The packed :class:`MeshCells`.
    """
    from .flags import ShapeFlags  # noqa: PLC0415

    def host(values, dtype):
        return np.asarray(values.numpy() if hasattr(values, "numpy") else values, dtype=dtype).reshape(-1)

    types = host(shape_type, np.int64)
    sources = host(shape_source_ptr, np.uint64)
    flags = host(shape_flags, np.int64) if shape_flags is not None else None
    shape_grid = np.full(max(types.size, 1), -1, dtype=np.int32)
    mesh_ids: list[int] = []
    grid_of: dict[int, int] = {}
    for s in range(types.size):
        if types[s] != int(GeoType.MESH) or sources[s] == 0:
            continue
        if flags is not None and not flags[s] & int(ShapeFlags.COLLIDE_SHAPES):
            continue
        mid = int(sources[s])
        if mid not in grid_of:
            grid_of[mid] = len(mesh_ids)
            mesh_ids.append(mid)
        shape_grid[s] = grid_of[mid]

    names = ("slots", "entries", "triangles", "sorted_slots", "level_start")
    chunks: dict[str, list] = {name: [] for name in names}
    grids = []
    table_offset = entry_offset = triangle_offset = sorted_offset = 0
    for points, indices in _read_meshes(mesh_ids, device):
        g = build_mesh_grid(points, indices, cell)
        info = MeshGrid()
        info.origin = wp.vec3(*[float(x) for x in g["origin"]])
        info.cell = float(g["cell"])
        info.inv_cell = float(np.float32(1.0) / np.float32(g["cell"]))
        info.levels = int(g["levels"])
        info.dims = wp.vec3i(*[int(x) for x in g["dims"]])
        info.table_start = table_offset
        info.table_mask = int(g["slots"].size - 1)
        info.triangle_start = triangle_offset
        info.level_start = len(grids) * (_MAX_LEVEL + 2)
        grids.append(info)
        slots = g["slots"].copy()
        slots["start"] += entry_offset
        sorted_slots = g["sorted_slots"].copy()
        sorted_slots["start"] += entry_offset
        chunks["slots"].append(slots)
        chunks["sorted_slots"].append(sorted_slots)
        chunks["level_start"].append(g["level_start"] + sorted_offset)
        chunks["entries"].append(g["entries"])
        chunks["triangles"].append(g["triangles"])
        table_offset += g["slots"].size
        entry_offset += g["entries"].shape[0]
        triangle_offset += g["triangles"].shape[0]
        sorted_offset += g["sorted_slots"].size

    def packed(name, dtype, width=None):
        parts = [x for x in chunks[name] if len(x)]
        if parts:
            return wp.array(np.concatenate(parts), dtype=dtype, device=device)
        if dtype in (MeshCellTriangle, MeshCellSlot):
            empty = np.zeros(1, dtype=dtype.numpy_dtype())
            if dtype == MeshCellSlot:
                empty["key"] = 0xFFFFFFFFFFFFFFFF
            return wp.array(empty, dtype=dtype, device=device)
        shape = (1,) if width is None else (1, width)
        return wp.array(np.zeros(shape, dtype=np.int64), dtype=dtype, device=device)

    result = MeshCells()
    result.grids = wp.array(grids if grids else [MeshGrid()], dtype=MeshGrid, device=device)
    result.shape_grid = wp.array(shape_grid, dtype=wp.int32, device=device)
    result.slots = packed("slots", MeshCellSlot)
    result.entries = packed("entries", wp.vec4i, 4)
    result.triangles = packed("triangles", MeshCellTriangle)
    result.sorted_slots = packed("sorted_slots", MeshCellSlot)
    result.level_start = packed("level_start", wp.int32)
    return result

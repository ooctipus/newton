# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Terrain cells of a static triangle mesh: the lattice, the cell classes and the cell table.

A static terrain mesh is mostly a height function sampled on a lattice: heightfield patches converted to triangles,
horizontal treads and box tops, and axis-aligned risers between them. :func:`build_terrain_mesh_cells` recovers that
structure from the triangles alone, never from the task, and writes it as a cell table the terrain law can read where
it reads a :class:`~newton.Heightfield` today:

* **The lattice.** The cell size of each axis is the most frequent length of the up-facing triangles' axis-aligned
  edges, refined with the origin by a least-squares fit of the coordinates on it (:func:`terrain_mesh_lattice`).
  Coordinates within the float32 tolerance of a lattice line are moved onto it, and other coordinates within it of
  each other are welded (two boxes' shared side, rounded one ulp apart).
* **Cells.** Cell ``(r, c)`` covers ``[x0 + c dx, x0 + (c + 1) dx] x [y0 + r dy, y0 + (r + 1) dy]`` in the mesh frame,
  row-major like a heightfield's cells. Every cell is one of:

  - **grid**: the up-facing surface over the cell is one plane, or two half-cell planes meeting along a diagonal
    (``(0, 0)-(1, 1)`` as in a :class:`~newton.Heightfield`, or ``(1, 0)-(0, 1)``, bit
    :data:`TERRAIN_CELL_DIAGONAL`); its four corner heights give the surface exactly;
  - **step**: horizontal levels on a 2 x 2 split by axis-aligned risers at the cell-local offsets ``(s, t)``: the
    riser ``x = s dx``, the riser ``y = t dy``, or both crossing (a box corner); an offset of 0 is no riser. Its
    corners hold the quadrants' levels (``z00`` for ``u < s, v < t``, ``z10`` for ``u > s, v < t`` ...), its record
    the offsets;
  - **mixed**: anything else (two parallel risers in one cell, a sloped piece beside a riser, a wall standing above
    both sides, an overhang, a partly covered cell); it keeps the mesh's own triangles as a CSR list with their
    z-range, for the triangle narrow phase without a BVH;
  - **empty**: no up-facing surface over the cell.

  Grid and step cells are the law's cells. The terrain is a solid under them, ``z <= h(x, y)``: a vertical triangle
  in a law cell must not reach above the solid's top along its line (on a riser, the higher side), and a down-facing
  triangle must lie below the cell's lowest corner (a box's bottom). An up-facing piece under a cell's one top plane
  with every down-facing piece lower still is inside the solid (overlapping boxes). Faces inside the solid (buried
  walls and tops, bottoms) are reproduced by the solid itself.
* **Risers on cell edges.** Each cell has its own four corners, so a riser on a lattice line (a stair's riser) is
  the difference between two neighbours' corners along their shared edge. The four edge bits
  (:data:`TERRAIN_CELL_EDGE_X0` ...) mark the edges whose neighbour's surface differs by more than
  :data:`TERRAIN_CELL_RISER_EPS` along the edge, or whose neighbour is not a law cell (mixed, empty, or outside the
  table): the edges where the law adds a riser, as it adds the border walls of a heightfield.

A heightfield is the special case with grid cells only, diagonal ``(0, 0)-(1, 1)``, no records and no edge bits
inside its footprint: ``vertex_z`` is its height grid.

The table (:class:`TerrainMeshCells`), a heightfield's vertex grid plus what it cannot say:

* ``vertex_z``: float32 ``(rows + 1, columns + 1)``, the shared lattice-vertex heights: the solid's top at each vertex
  (the highest corner of the law cells around it);
* ``cell_flags``: uint32 per cell, the kind (bits 0-1), the diagonal (bit 2), the record bit (bit 3), the edge bits
  (bits 4-7) and the payload (bits 8-31: a law cell's record, a mixed cell's index);
* ``record_z``, ``record_offsets``: float32 ``(records, 4)`` and ``(records, 2)``, the corners and riser offsets
  ``(s, t)`` of every step cell and of every grid cell a corner of which differs from the shared vertex (the low side
  of a riser on a cell edge). A cell without a record reads its corners from ``vertex_z``;
* ``mixed_cell``, ``mixed_start``, ``mixed_triangles``, ``mixed_z``: the mixed cells' cell ids, CSR offsets, triangle
  ids and z-ranges ``(lowest, highest)`` of their triangles clipped to the cell.

The layout was chosen by the read cost of the law's windows (:attr:`TerrainMeshCells.cell_z` rebuilds per-cell corners
for the host): on the rough terrain it is 24 MB against 48 MB of per-cell corners, and no slower to read.

The build runs where the model lives, as Warp kernels compiled without floating-point contraction, so every device
(the CPU included) builds the same bits:

1. the device reduces the mesh to its distinct up-facing edge lengths and vertex coordinates (with counts), on which
   the host fits the lattice, and to its distinct coordinates per axis, which the host snaps and welds and the device
   maps back to every vertex;
2. each triangle lists the cells it can touch, laid out in triangle order, and one thread per incidence clips its
   piece to the cell in cell-local float64 (a face's area, extents, z-range and plane at the corners; a wall's run and
   top); a stable radix sort groups the incidences by cell;
3. one thread per cell classifies it from its pieces (stacked solids, up to four planes, grid or step), copies the
   corner vertices' heights, and checks its bottoms and the walls inside it; one more pass checks the walls on its
   edges against both neighbours, and the last ones write the edge bits, the step records and the mixed CSR.

Decisions are taken in float64 (the mesh's float32 vertices are exact in it); corners copy the heights of the mesh's
vertices at the cell corners bit for bit, and planes evaluated at the corners round once to float32.
:func:`verify_terrain_mesh_cells` (numpy) checks the table against every triangle independently of the classifier.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import warp as wp

# Every float operation of the build rounds as on the CPU: the table is the same bits on every device.
wp.set_module_options({"fuse_fp": False})

TERRAIN_CELL_GRID = 0
"""Cell kind: one plane, or two half-cell planes, given by the four corners."""
TERRAIN_CELL_STEP = 1
"""Cell kind: horizontal levels on a 2 x 2 split by axis-aligned risers at the cell-local offsets ``(s, t)`` of the
step's record (one riser, or two crossing)."""
TERRAIN_CELL_MIXED = 2
"""Cell kind: the mesh's own triangles (CSR list), for the triangle narrow phase."""
TERRAIN_CELL_EMPTY = 3
"""Cell kind: no up-facing surface over the cell."""
TERRAIN_CELL_KIND_MASK = 0x3
TERRAIN_CELL_DIAGONAL = 0x4
"""Grid cell: the halves meet on the diagonal ``(1, 0)-(0, 1)`` instead of the heightfield's ``(0, 0)-(1, 1)``."""
TERRAIN_CELL_RECORD = 0x8
"""The cell has a record: the corners the shared vertex heights cannot give (a riser on or inside the cell) and a step's
offsets. The payload is the record's index."""
TERRAIN_CELL_EDGE_X0 = 0x10
"""Riser on the cell's edge ``x = 0`` (shared with column ``c - 1``)."""
TERRAIN_CELL_EDGE_X1 = 0x20
"""Riser on the cell's edge ``x = dx`` (shared with column ``c + 1``)."""
TERRAIN_CELL_EDGE_Y0 = 0x40
"""Riser on the cell's edge ``y = 0`` (shared with row ``r - 1``)."""
TERRAIN_CELL_EDGE_Y1 = 0x80
"""Riser on the cell's edge ``y = dy`` (shared with row ``r + 1``)."""
TERRAIN_CELL_PAYLOAD_SHIFT = 8
"""The payload (bits 8-31): a law cell's record index (with :data:`TERRAIN_CELL_RECORD`), a mixed cell's index in the
mixed CSR."""
TERRAIN_CELL_RISER_EPS = 1.0e-6
"""Height difference [m] along a shared edge above which the edge is a riser (the law's flatness tolerance)."""

_EPS32 = float(np.finfo(np.float32).eps)
_PLANE_TOL = 1.0e-6  # [m] two pieces are one plane when they agree at the cell corners within this (the law's 1 um)
_MAX_POLY = 8  # a triangle clipped by four half-planes keeps at most 7 vertices
_CHUNK = 1 << 20  # incidences per clipping batch

_KIND_NAMES = ("grid", "step", "mixed", "empty")


@dataclass
class TerrainMeshCells:
    """The cell table of one static terrain mesh (see the module docstring for the layout)."""

    origin: tuple[float, float]
    """Lower-left corner ``(x0, y0)`` [m] of cell ``(0, 0)`` in the mesh frame (float64)."""
    cell_size: tuple[float, float]
    """Cell edge lengths ``(dx, dy)`` [m]."""
    shape: tuple[int, int]
    """``(rows, columns)`` of cells."""
    vertex_z: np.ndarray
    """float32 ``(rows + 1, columns + 1)``: the shared lattice-vertex heights, the solid's top at each vertex (the
    highest corner of the law cells around it; 0 where none is)."""
    cell_flags: np.ndarray
    mixed_cell: np.ndarray
    mixed_start: np.ndarray
    mixed_triangles: np.ndarray
    mixed_z: np.ndarray
    record_z: np.ndarray
    """float32 ``(records, 4)``: the corners of each law cell with a record."""
    record_offsets: np.ndarray
    """float32 ``(records, 2)``: a record's riser offsets ``(s, t)`` in cell units, 0 where it has no riser."""
    stats: dict = field(default_factory=dict)
    """Build counters: mixed reasons, wall and bottom checks, lattice shares (diagnostics, not read by the law)."""

    @property
    def kind(self) -> np.ndarray:
        return (self.cell_flags & TERRAIN_CELL_KIND_MASK).astype(np.int32)

    def counts(self) -> dict[str, int]:
        """Number of cells of each kind."""
        k = np.bincount(self.kind, minlength=4)
        return {name: int(k[i]) for i, name in enumerate(_KIND_NAMES)}

    def memory_bytes(self) -> dict[str, int]:
        """Bytes of each array of the table."""
        out = {
            "vertex_z": self.vertex_z.nbytes,
            "cell_flags": self.cell_flags.nbytes,
            "record_z": self.record_z.nbytes,
            "record_offsets": self.record_offsets.nbytes,
            "mixed_cell": self.mixed_cell.nbytes,
            "mixed_start": self.mixed_start.nbytes,
            "mixed_triangles": self.mixed_triangles.nbytes,
            "mixed_z": self.mixed_z.nbytes,
        }
        out["total"] = sum(out.values())
        return out

    def _record_of(self) -> tuple[np.ndarray, np.ndarray]:
        has = (self.cell_flags & TERRAIN_CELL_RECORD) != 0
        has &= self.kind != TERRAIN_CELL_MIXED
        return has, (self.cell_flags[has] >> TERRAIN_CELL_PAYLOAD_SHIFT).astype(np.int64)

    def offsets(self) -> np.ndarray:
        """Each cell's riser offsets ``(s, t)`` in cell units (float32; zeros for a cell without a record)."""
        out = np.zeros((self.cell_flags.size, 2), dtype=np.float32)
        has, idx = self._record_of()
        out[has] = self.record_offsets[idx]
        return out

    @functools.cached_property
    def cell_z(self) -> np.ndarray:
        """Each cell's corners ``z00, z10, z01, z11`` as the law reads them: its record's, else the shared vertex
        heights; a mixed cell's four corners hold its top, an empty cell's zeros."""
        vz = self.vertex_z
        out = np.stack([vz[:-1, :-1], vz[:-1, 1:], vz[1:, :-1], vz[1:, 1:]], axis=-1).reshape(-1, 4).copy()
        has, idx = self._record_of()
        out[has] = self.record_z[idx]
        kind = self.kind
        mixed = kind == TERRAIN_CELL_MIXED
        out[mixed] = self.mixed_z[(self.cell_flags[mixed] >> TERRAIN_CELL_PAYLOAD_SHIFT).astype(np.int64), 1][:, None]
        out[kind == TERRAIN_CELL_EMPTY] = 0.0
        return out

    def surface_z(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reference decode: the surface height at mesh-frame points, and each point's cell kind.

        Cell-local coordinates are taken in float64 from the exact lattice, the height is formed in float32 from the
        corners as the law forms it. On a riser line the solid's top (the higher side) is returned; mixed and empty
        cells, and points outside the table, return NaN.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rows, cols = self.shape
        qx = (x - self.origin[0]) / self.cell_size[0]
        qy = (y - self.origin[1]) / self.cell_size[1]
        c = np.floor(qx).astype(np.int64)
        r = np.floor(qy).astype(np.int64)
        inside = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
        cell = np.where(inside, r * cols + c, 0)
        u = (qx - c).astype(np.float32)
        v = (qy - r).astype(np.float32)
        z = self.cell_z[cell]
        z00, z10, z01, z11 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        flags = self.cell_flags[cell]
        kind = (flags & TERRAIN_CELL_KIND_MASK).astype(np.int32)
        one = np.float32(1.0)
        # diagonal (0,0)-(1,1): triangles (00,10,11) for u >= v and (00,11,01) for u < v
        lo_d0 = z00 + (z10 - z00) * u + (z11 - z10) * v
        hi_d0 = z00 + (z11 - z01) * u + (z01 - z00) * v
        g0 = np.where(u >= v, lo_d0, hi_d0)
        # diagonal (1,0)-(0,1): triangles (00,10,01) for u + v <= 1 and (10,11,01) beyond
        lo_d1 = z00 + (z10 - z00) * u + (z01 - z00) * v
        hi_d1 = z11 + (z11 - z01) * (u - one) + (z11 - z10) * (v - one)
        g1 = np.where(u + v <= one, lo_d1, hi_d1)
        grid = np.where((flags & TERRAIN_CELL_DIAGONAL) != 0, g1, g0)
        off = self.offsets()[cell]
        step = _step_levels(z, off[:, 0], off[:, 1], u, v, np.float32(0.0))
        out = np.where(kind == TERRAIN_CELL_GRID, grid, np.where(kind == TERRAIN_CELL_STEP, step, np.float32(np.nan)))
        out = np.where(inside, out, np.float32(np.nan))
        return out, np.where(inside, kind, -1)


def _step_levels(z, s, t, u, v, band):
    """A step cell's level at cell-local ``(u, v)``: the quadrant's corner, the higher side on a riser line (within
    ``band``); an offset of 0 is no riser."""
    hi_u = u > s + band
    on_u = (s > 0) & (np.abs(u - s) <= band)
    hi_v = v > t + band
    on_v = (t > 0) & (np.abs(v - t) <= band)
    out = None
    for iu in (0, 1):
        for iv in (0, 1):
            takes = ((hi_u == bool(iu)) | on_u) & ((hi_v == bool(iv)) | on_v)
            zq = np.where(takes, z[:, iu + 2 * iv], -np.inf)
            out = zq if out is None else np.maximum(out, zq)
    return out.astype(z.dtype)


# ----------------------------------------------------------------------------------------------------------------
# geometry helpers (float64, vectorized)
# ----------------------------------------------------------------------------------------------------------------


def _clip_half(px, py, n, axis: int, bound, sign: float):
    """Clip convex polygons ``(px, py)`` with ``n`` vertices to ``sign * (coord - bound) <= 0`` (Sutherland-Hodgman)."""
    k, m = px.shape
    coord = px if axis == 0 else py
    d = sign * (coord - bound[:, None])
    slot = np.arange(m)[None, :]
    valid = slot < n[:, None]
    nxt = np.where(slot + 1 < n[:, None], slot + 1, 0)
    pxn = np.take_along_axis(px, nxt, 1)
    pyn = np.take_along_axis(py, nxt, 1)
    dn = np.take_along_axis(d, nxt, 1)
    inside = d <= 0.0
    keep = valid & inside
    cross = valid & (inside != (dn <= 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(cross, d / (d - dn), 0.0)
    ix = px + t * (pxn - px)
    iy = py + t * (pyn - py)
    if axis == 0:
        ix = np.where(cross, bound[:, None], ix)  # the cut lies exactly on the bound
    else:
        iy = np.where(cross, bound[:, None], iy)
    emit = keep.astype(np.int64) + cross.astype(np.int64)
    start = np.cumsum(emit, axis=1) - emit
    ox = np.zeros_like(px)
    oy = np.zeros_like(py)
    rows = np.broadcast_to(np.arange(k)[:, None], (k, m))
    pk = start[keep]
    ox[rows[keep], pk] = px[keep]
    oy[rows[keep], pk] = py[keep]
    pc = (start + keep)[cross]
    ox[rows[cross], pc] = ix[cross]
    oy[rows[cross], pc] = iy[cross]
    return ox, oy, emit.sum(axis=1)


def _clip_box(px, py, n, xlo, xhi, ylo, yhi):
    px, py, n = _clip_half(px, py, n, 0, xlo, -1.0)
    px, py, n = _clip_half(px, py, n, 0, xhi, 1.0)
    px, py, n = _clip_half(px, py, n, 1, ylo, -1.0)
    px, py, n = _clip_half(px, py, n, 1, yhi, 1.0)
    return px, py, n


def _area(px, py, n):
    m = px.shape[1]
    slot = np.arange(m)[None, :]
    nxt = np.where(slot + 1 < n[:, None], slot + 1, 0)
    cross = px * np.take_along_axis(py, nxt, 1) - np.take_along_axis(px, nxt, 1) * py
    return 0.5 * np.where(slot < n[:, None], cross, 0.0).sum(axis=1)


def _mode(values: np.ndarray, quantum: float) -> tuple[float, float]:
    """Most frequent value after rounding to ``quantum``, and its share."""
    if values.size == 0:
        return 0.0, 0.0
    keys, counts = np.unique(np.round(values / quantum).astype(np.int64), return_counts=True)
    i = int(np.argmax(counts))
    return float(keys[i] * quantum), float(counts[i] / values.size)


def _segment_reduce(keys: np.ndarray, values: np.ndarray, size: int, op, fill: float) -> np.ndarray:
    out = np.full(size, fill, dtype=np.float64)
    op.at(out, keys, values)
    return out


def _snap_coords(x: np.ndarray, origin: float, size: float, tol: float) -> np.ndarray:
    """Move coordinates within ``tol`` of a lattice line onto it (the float32 rounding of lattice positions), and weld
    the other coordinates that lie within ``tol`` of each other to their smallest (two boxes' shared side, rounded
    differently in float32)."""
    q = (x - origin) / size
    qr = np.round(q)
    on = np.abs(q - qr) * size <= tol
    out = np.where(on, origin + qr * size, x)
    off = np.flatnonzero(~on)
    if off.size:
        vals, inv = np.unique(out[off], return_inverse=True)
        start = np.r_[True, np.diff(vals) > tol]
        group = np.cumsum(start) - 1
        first = vals[start][group]
        width = _segment_reduce(group, vals, int(group[-1]) + 1, np.maximum, -np.inf) - vals[start]
        weld = width[group] <= tol  # a chain wider than the tolerance is distinct features: left as it is
        out[off] = np.where(weld, first, vals)[inv]
    return out


def _snap_to_lattice(v: np.ndarray, origin, size, tol: float) -> tuple[np.ndarray, float]:
    """:func:`_snap_coords` on both axes of vertices; returns the moved vertices and the largest move [m]."""
    vs = v.copy()
    for axis in (0, 1):
        vs[:, axis] = _snap_coords(v[:, axis], origin[axis], size[axis], tol)
    return vs, float(np.max(np.abs(vs[:, :2] - v[:, :2]))) if v.size else 0.0


# ----------------------------------------------------------------------------------------------------------------
# the device build: Warp kernels, the same on every device
# ----------------------------------------------------------------------------------------------------------------

_F64_ZERO = wp.constant(wp.float64(0.0))
_F64_ONE = wp.constant(wp.float64(1.0))
_F64_TOL_PLANE = wp.constant(wp.float64(_PLANE_TOL))
_F64_BIG = wp.constant(wp.float64(1.0e300))
_UP = wp.constant(0)
_DOWN = wp.constant(1)
_WALL = wp.constant(2)
_DEGENERATE = wp.constant(3)
# incidence kinds: a face piece, a wall inside the cell, a wall on the cell's high edge (x = dx or y = dy), a wall on
# its low edge, a wall along no axis, and a dead incidence (no piece in the cell)
_INC_FACE = wp.constant(0)
_INC_WALL_IN = wp.constant(1)
_INC_WALL_HI = wp.constant(2)
_INC_WALL_LO = wp.constant(3)
_INC_WALL_GENERAL = wp.constant(4)
_INC_DEAD = wp.constant(5)
_GRID = wp.constant(TERRAIN_CELL_GRID)
_STEP = wp.constant(TERRAIN_CELL_STEP)
_MIXED = wp.constant(TERRAIN_CELL_MIXED)
_EMPTY = wp.constant(TERRAIN_CELL_EMPTY)

vec8d = wp.types.vector(length=8, dtype=wp.float64)
mat48d = wp.types.matrix(shape=(4, 8), dtype=wp.float64)


@wp.struct
class _Lattice:
    x0: wp.float64
    y0: wp.float64
    dx: wp.float64
    dy: wp.float64
    rows: wp.int32
    cols: wp.int32
    tol: wp.float64
    tu: wp.float64
    area_tol: wp.float64
    piece_min: wp.float64


@wp.kernel(enable_backward=False)
def _abs_max_xy(vertices: wp.array[wp.vec3], out: wp.array[wp.float32]):
    i = wp.tid()
    v = vertices[i]
    wp.atomic_max(out, 0, wp.max(wp.abs(v[0]), wp.abs(v[1])))


@wp.kernel(enable_backward=False)
def _classify_triangles(
    vertices: wp.array[wp.vec3],
    triangles: wp.array[wp.vec3i],
    tol: wp.float64,
    tri_class: wp.array[wp.int32],
    span_x: wp.array[wp.float64],
    span_y: wp.array[wp.float64],
    up_coord_x: wp.array[wp.float32],
    up_coord_y: wp.array[wp.float32],
):
    """Each triangle's class on the mesh's own float32 vertices; an up-facing triangle's axis-aligned edge lengths
    (NaN where an edge is not axis-aligned) and its vertex coordinates, for the lattice."""
    t = wp.tid()
    ia = triangles[t][0]
    ib = triangles[t][1]
    ic = triangles[t][2]
    a = wp.vec3d(vertices[ia])
    b = wp.vec3d(vertices[ib])
    c = wp.vec3d(vertices[ic])
    n = wp.cross(b - a, c - a)
    nlen = wp.length(n)
    cls = int(_DEGENERATE)
    if nlen > _F64_ZERO:
        nz = n[2] / nlen
        if nz > wp.float64(1.0e-4):
            cls = _UP
        elif nz < wp.float64(-1.0e-4):
            cls = _DOWN
        else:
            cls = _WALL
    tri_class[t] = cls
    nan = wp.float64(wp.nan)
    for j in range(3):
        p = a
        q = b
        if j == 1:
            p = b
            q = c
        elif j == 2:
            p = c
            q = a
        d = q - p
        sx = nan
        sy = nan
        if cls == _UP:
            if wp.abs(d[1]) <= tol and wp.abs(d[0]) > tol:
                sx = wp.abs(d[0])
            if wp.abs(d[0]) <= tol and wp.abs(d[1]) > tol:
                sy = wp.abs(d[1])
        span_x[3 * t + j] = sx
        span_y[3 * t + j] = sy
        vx = wp.float32(nan)
        vy = wp.float32(nan)
        if cls == _UP:
            vx = wp.float32(p[0])
            vy = wp.float32(p[1])
        up_coord_x[3 * t + j] = vx
        up_coord_y[3 * t + j] = vy


@wp.kernel(enable_backward=False)
def _sorted_heads(keys: wp.array[wp.float64], count: int, heads: wp.array[wp.int32], valid: wp.array[wp.int32]):
    i = wp.tid()
    h = int(0)
    if i < count and keys[i] == keys[i]:  # NaN keys (sorted last) are no value
        wp.atomic_add(valid, 0, 1)
        if i == 0:
            h = 1
        elif keys[i] != keys[i - 1]:
            h = 1
    heads[i] = h


@wp.kernel(enable_backward=False)
def _iota(out: wp.array[wp.int32]):
    i = wp.tid()
    out[i] = i


@wp.kernel(enable_backward=False)
def _compact_unique(
    keys: wp.array[wp.float64],
    heads: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    count: int,
    values: wp.array[wp.float64],
    starts: wp.array[wp.int32],
):
    i = wp.tid()
    if i < count and heads[i] == 1:
        values[rank[i] - 1] = keys[i]
        starts[rank[i] - 1] = i


def _unique_counts(values: wp.array, device) -> tuple[np.ndarray, np.ndarray, wp.array, wp.array]:
    """Distinct values of a device array (NaN skipped) with their counts, sorted; also each sorted position's
    1-based rank and the sort permutation, for mapping values back."""
    n = values.shape[0]
    keys = wp.empty(2 * n, dtype=wp.float64, device=device)
    wp.copy(keys, values.flatten() if values.dtype == wp.float64 else _to_f64(values, device), count=n)
    perm = wp.empty(2 * n, dtype=wp.int32, device=device)
    wp.launch(_iota, dim=n, outputs=[perm], device=device)
    wp.utils.radix_sort_pairs(keys, perm, n)
    heads = wp.empty(n, dtype=wp.int32, device=device)
    valid_n = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(_sorted_heads, dim=n, inputs=[keys, n], outputs=[heads, valid_n], device=device)
    rank = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_scan(heads, rank, inclusive=True)
    nu = int(rank[n - 1 : n].numpy()[0]) if n else 0
    uvals = wp.empty(max(nu, 1), dtype=wp.float64, device=device)
    ustart = wp.empty(max(nu, 1), dtype=wp.int32, device=device)
    wp.launch(_compact_unique, dim=n, inputs=[keys, heads, rank, n], outputs=[uvals, ustart], device=device)
    vals = uvals.numpy()[:nu]
    starts = ustart.numpy()[:nu]
    counts = np.diff(np.r_[starts, int(valid_n.numpy()[0])]).astype(np.int64)
    return vals, counts, rank, perm


@wp.kernel(enable_backward=False)
def _widen(src: wp.array[wp.float32], dst: wp.array[wp.float64]):
    i = wp.tid()
    dst[i] = wp.float64(src[i])


def _to_f64(values: wp.array, device) -> wp.array:
    out = wp.empty(values.shape[0], dtype=wp.float64, device=device)
    wp.launch(_widen, dim=values.shape[0], inputs=[values], outputs=[out], device=device)
    return out


def _weighted_mode(values: np.ndarray, counts: np.ndarray, quantum: float) -> tuple[float, float]:
    """``_mode`` of the multiset ``values`` repeated ``counts`` times."""
    if values.size == 0:
        return 0.0, 0.0
    keys, inv = np.unique(np.round(values / quantum).astype(np.int64), return_inverse=True)
    weight = np.bincount(inv, weights=counts, minlength=keys.size)
    i = int(np.argmax(weight))
    return float(keys[i] * quantum), float(weight[i] / counts.sum())


def _weighted_median(values: np.ndarray, counts: np.ndarray) -> float:
    """``np.median`` of sorted ``values`` repeated ``counts`` times."""
    total = int(counts.sum())
    cum = np.cumsum(counts)
    lo = values[int(np.searchsorted(cum, (total - 1) // 2, side="right"))]
    hi = values[int(np.searchsorted(cum, total // 2, side="right"))]
    return float(np.mean([lo, hi]))


def _lattice_axis(spans, span_counts, coords, coord_counts, tol: float) -> dict:
    """One axis of the lattice from the distinct up-facing edge lengths and vertex coordinates (with their counts)."""
    size, share = _weighted_mode(spans, span_counts, tol)
    if size <= 0.0:
        return {"size": 0.0, "origin": 0.0, "edge_share": 0.0, "vertex_share": 0.0}
    near = np.abs(spans - size) <= tol
    size = _weighted_median(spans[near], span_counts[near])
    residue = np.mod(coords, size)
    origin, _ = _mode(np.where(size - residue <= tol, 0.0, residue), tol)
    # least squares over the coordinates on the lattice: float32 rounding averages out over the whole extent
    for _ in range(3):
        k = np.round((coords - origin) / size)
        on = np.abs(coords - origin - k * size) <= tol
        kc = k[on] - k[on].mean()
        size = float(np.sum(kc * (coords[on] - coords[on].mean())) / np.sum(kc * kc)) if kc.size > 1 else size
        origin = float(np.mean(coords[on] - k[on] * size))
    origin = float(origin - size * np.round(origin / size))
    q = (coords - origin) / size
    on = np.abs(q - np.round(q)) * size <= tol
    share_v = float(coord_counts[on].sum() / max(coord_counts.sum(), 1))
    return {"size": size, "origin": origin, "edge_share": share, "vertex_share": share_v}


def _lattice_device(v32: wp.array, tri: wp.array, device) -> tuple[dict, wp.array]:
    """The lattice (:func:`terrain_mesh_lattice`) from device summaries; also each triangle's class."""
    m = tri.shape[0]
    amax = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(_abs_max_xy, dim=v32.shape[0], inputs=[v32], outputs=[amax], device=device)
    extent = float(amax.numpy()[0]) if v32.shape[0] else 1.0
    tol = 8.0 * _EPS32 * max(extent, 1.0)
    tri_class = wp.empty(m, dtype=wp.int32, device=device)
    span = [wp.empty(3 * m, dtype=wp.float64, device=device) for _ in range(2)]
    coord = [wp.empty(3 * m, dtype=wp.float32, device=device) for _ in range(2)]
    wp.launch(
        _classify_triangles,
        dim=m,
        inputs=[v32, tri, tol],
        outputs=[tri_class, span[0], span[1], coord[0], coord[1]],
        device=device,
    )
    out = {"tolerance": tol}
    for axis, name in ((0, "x"), (1, "y")):
        sv, sc, _, _ = _unique_counts(span[axis], device)
        cv, cc, _, _ = _unique_counts(coord[axis], device)
        out[name] = _lattice_axis(sv, sc, cv, cc, tol)
    return out, tri_class


@wp.kernel(enable_backward=False)
def _snap_scatter(
    perm: wp.array[wp.int32],
    rank: wp.array[wp.int32],
    mapped: wp.array[wp.float64],
    count: int,
    axis: int,
    out: wp.array[wp.vec3d],
):
    i = wp.tid()
    if i < count:
        p = perm[i]
        v = out[p]
        v[axis] = mapped[rank[i] - 1]
        out[p] = v


@wp.kernel(enable_backward=False)
def _set_z(vertices: wp.array[wp.vec3], out: wp.array[wp.vec3d]):
    i = wp.tid()
    out[i] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(vertices[i][2]))


@wp.kernel(enable_backward=False)
def _coord_of(vertices: wp.array[wp.vec3], axis: int, out: wp.array[wp.float32]):
    i = wp.tid()
    out[i] = vertices[i][axis]


def _snap_device(v32: wp.array, origin, size, tol: float, device):
    """Snap and weld every vertex coordinate (:func:`_snap_coords` on the distinct coordinates, mapped back); returns
    the snapped vertices, the largest move and each axis's snapped extent."""
    n = v32.shape[0]
    vs = wp.empty(n, dtype=wp.vec3d, device=device)
    wp.launch(_set_z, dim=n, inputs=[v32], outputs=[vs], device=device)
    moved = 0.0
    extent = []
    for axis in (0, 1):
        c = wp.empty(n, dtype=wp.float32, device=device)
        wp.launch(_coord_of, dim=n, inputs=[v32, axis], outputs=[c], device=device)
        vals, _, rank, perm = _unique_counts(c, device)
        snapped = _snap_coords(vals, origin[axis], size[axis], tol)
        moved = max(moved, float(np.max(np.abs(snapped - vals))) if vals.size else 0.0)
        extent.append((float(snapped.min()), float(snapped.max())))
        mapped = wp.array(snapped, dtype=wp.float64, device=device)
        wp.launch(_snap_scatter, dim=n, inputs=[perm, rank, mapped, n, axis], outputs=[vs], device=device)
    return vs, moved, extent


# --- incidences ------------------------------------------------------------------------------------------------


@wp.func
def _cell_range(lo: wp.float64, hi: wp.float64, o: wp.float64, size: wp.float64, shrink: wp.float64, n: int):
    """Cells ``[k0, k1]`` whose interior a footprint ``[lo, hi]`` overlaps (shrunk by ``shrink`` cell units)."""
    k0 = wp.clamp(int(wp.floor((lo - o) / size + shrink)), 0, n - 1)
    k1 = wp.clamp(int(wp.ceil((hi - o) / size - shrink)) - 1, 0, n - 1)
    return k0, wp.max(k1, k0)


@wp.func
def _wall_line(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, tol: wp.float64):
    """A vertical triangle's line: 0 for ``x = const``, 1 for ``y = const``, -1 for neither."""
    xs = wp.max(wp.max(a[0], b[0]), c[0]) - wp.min(wp.min(a[0], b[0]), c[0])
    ys = wp.max(wp.max(a[1], b[1]), c[1]) - wp.min(wp.min(a[1], b[1]), c[1])
    if xs <= tol:
        return 0
    if ys <= tol:
        return 1
    return -1


@wp.kernel(enable_backward=False)
def _incidence_counts(
    vs: wp.array[wp.vec3d],
    triangles: wp.array[wp.vec3i],
    tri_class: wp.array[wp.int32],
    lat: _Lattice,
    counts: wp.array[wp.int32],
):
    """The cells each triangle can touch: a face's footprint box; an axis-aligned wall's run of cells (twice on a
    lattice line, one per side); a general wall's footprint box."""
    t = wp.tid()
    a = vs[triangles[t][0]]
    b = vs[triangles[t][1]]
    c = vs[triangles[t][2]]
    cls = tri_class[t]
    n = int(0)
    if cls == _UP or cls == _DOWN:
        shrink = lat.tol / lat.dx
        c0, c1 = _cell_range(
            wp.min(wp.min(a[0], b[0]), c[0]), wp.max(wp.max(a[0], b[0]), c[0]), lat.x0, lat.dx, shrink, lat.cols
        )
        shrink = lat.tol / lat.dy
        r0, r1 = _cell_range(
            wp.min(wp.min(a[1], b[1]), c[1]), wp.max(wp.max(a[1], b[1]), c[1]), lat.y0, lat.dy, shrink, lat.rows
        )
        n = (c1 - c0 + 1) * (r1 - r0 + 1)
    elif cls == _WALL:
        line = _wall_line(a, b, c, lat.tol)
        if line < 0:
            c0, c1 = _cell_range(
                wp.min(wp.min(a[0], b[0]), c[0]), wp.max(wp.max(a[0], b[0]), c[0]), lat.x0, lat.dx, lat.tu, lat.cols
            )
            r0, r1 = _cell_range(
                wp.min(wp.min(a[1], b[1]), c[1]), wp.max(wp.max(a[1], b[1]), c[1]), lat.y0, lat.dy, lat.tu, lat.rows
            )
            n = (c1 - c0 + 1) * (r1 - r0 + 1)
        else:
            k0 = int(0)
            k1 = int(0)
            if line == 0:
                k0, k1 = _cell_range(
                    wp.min(wp.min(a[1], b[1]), c[1]), wp.max(wp.max(a[1], b[1]), c[1]), lat.y0, lat.dy, lat.tu, lat.rows
                )
            else:
                k0, k1 = _cell_range(
                    wp.min(wp.min(a[0], b[0]), c[0]), wp.max(wp.max(a[0], b[0]), c[0]), lat.x0, lat.dx, lat.tu, lat.cols
                )
            n = 2 * (k1 - k0 + 1)
    counts[t] = n


@wp.func
def _clip_side(px: vec8d, py: vec8d, n: int, axis: int, bound: wp.float64, sign: wp.float64):
    """Clip a convex polygon to ``sign * (coord - bound) <= 0`` (Sutherland-Hodgman), as the reference clips."""
    ox = vec8d()
    oy = vec8d()
    m = int(0)
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        ci = px[i]
        cj = px[j]
        if axis == 1:
            ci = py[i]
            cj = py[j]
        di = sign * (ci - bound)
        dj = sign * (cj - bound)
        if di <= _F64_ZERO:
            ox[m] = px[i]
            oy[m] = py[i]
            m += 1
        if (di <= _F64_ZERO) != (dj <= _F64_ZERO):
            tt = di / (di - dj)
            x = px[i] + tt * (px[j] - px[i])
            y = py[i] + tt * (py[j] - py[i])
            if axis == 0:
                x = bound
            else:
                y = bound
            ox[m] = x
            oy[m] = y
            m += 1
    return ox, oy, m


@wp.func
def _clip_cell(px: vec8d, py: vec8d, n: int, w: wp.float64, h: wp.float64):
    px, py, n = _clip_side(px, py, n, 0, _F64_ZERO, wp.float64(-1.0))
    px, py, n = _clip_side(px, py, n, 0, w, _F64_ONE)
    px, py, n = _clip_side(px, py, n, 1, _F64_ZERO, wp.float64(-1.0))
    px, py, n = _clip_side(px, py, n, 1, h, _F64_ONE)
    return px, py, n


@wp.func
def _shoelace(px: vec8d, py: vec8d, n: int) -> wp.float64:
    s = wp.float64(0.0)
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        s += px[i] * py[j] - px[j] * py[i]
    return wp.abs(wp.float64(0.5) * s)


@wp.func
def _upper_bound(offsets: wp.array[wp.int32], count: int, x: int) -> int:
    """First index in ``offsets[0:count]`` holding a value greater than ``x``."""
    lo = int(0)
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel(enable_backward=False)
def _incidence_pieces(
    vs: wp.array[wp.vec3d],
    triangles: wp.array[wp.vec3i],
    tri_class: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    tri_count: int,
    lat: _Lattice,
    inc_cell: wp.array[wp.int32],
    inc_tri: wp.array[wp.int32],
    inc_kind: wp.array[wp.int32],
    inc_area: wp.array[wp.float64],
    inc_corner: wp.array[wp.vec4d],
    inc_ext: wp.array[vec8d],
    inc_z: wp.array[wp.vec2d],
):
    """One incidence: its cell, kind and piece. A face's piece is the triangle clipped to the cell in cell-local
    coordinates (area, extents in cell units, z range, its plane at the cell's corners); a wall's piece its run
    ``[t0, t1]`` in the row of cells and its top over it (``ext[0:2]``, ``z[1]``), with its line's offset in cell
    units (``ext[2]``) for a wall inside the cell."""
    i = wp.tid()
    t = _upper_bound(offsets, tri_count + 1, i) - 1
    local = i - offsets[t]
    a = vs[triangles[t][0]]
    b = vs[triangles[t][1]]
    c = vs[triangles[t][2]]
    cls = tri_class[t]
    inc_tri[i] = t
    kind = int(_INC_DEAD)
    cell = int(lat.rows * lat.cols)
    area = wp.float64(0.0)
    corner = wp.vec4d()
    ext = vec8d()
    zz = wp.vec2d()
    if cls == _UP or cls == _DOWN:
        c0, c1 = _cell_range(
            wp.min(wp.min(a[0], b[0]), c[0]),
            wp.max(wp.max(a[0], b[0]), c[0]),
            lat.x0,
            lat.dx,
            lat.tol / lat.dx,
            lat.cols,
        )
        r0, _r1 = _cell_range(
            wp.min(wp.min(a[1], b[1]), c[1]),
            wp.max(wp.max(a[1], b[1]), c[1]),
            lat.y0,
            lat.dy,
            lat.tol / lat.dy,
            lat.rows,
        )
        ncx = c1 - c0 + 1
        cc = c0 + local % ncx
        rr = r0 + local // ncx
        bx0 = lat.x0 + wp.float64(cc) * lat.dx
        by0 = lat.y0 + wp.float64(rr) * lat.dy
        px = vec8d()
        py = vec8d()
        px[0] = a[0] - bx0
        px[1] = b[0] - bx0
        px[2] = c[0] - bx0
        py[0] = a[1] - by0
        py[1] = b[1] - by0
        py[2] = c[1] - by0
        px, py, n = _clip_cell(px, py, 3, lat.dx, lat.dy)
        area = _shoelace(px, py, n)
        if n >= 3 and area > lat.piece_min:
            kind = _INC_FACE
            cell = rr * lat.cols + cc
            nrm = wp.cross(b - a, c - a)
            gx = -nrm[0] / nrm[2]
            gy = -nrm[1] / nrm[2]
            zlo = wp.float64(1.0e300)
            zhi = wp.float64(-1.0e300)
            e = vec8d(_F64_BIG, -_F64_BIG, _F64_BIG, -_F64_BIG, _F64_BIG, -_F64_BIG, _F64_BIG, -_F64_BIG)
            for k in range(n):
                u = px[k] / lat.dx
                w = py[k] / lat.dy
                e[0] = wp.min(e[0], u)
                e[1] = wp.max(e[1], u)
                e[2] = wp.min(e[2], w)
                e[3] = wp.max(e[3], w)
                e[4] = wp.min(e[4], u - w)
                e[5] = wp.max(e[5], u - w)
                e[6] = wp.min(e[6], u + w)
                e[7] = wp.max(e[7], u + w)
                z = a[2] + gx * ((px[k] + bx0) - a[0]) + gy * ((py[k] + by0) - a[1])
                zlo = wp.min(zlo, z)
                zhi = wp.max(zhi, z)
            ext = e
            zz = wp.vec2d(zlo, zhi)
            for q in range(4):
                cx = bx0
                cy = by0
                if q == 1 or q == 3:
                    cx = bx0 + lat.dx
                if q >= 2:
                    cy = by0 + lat.dy
                corner[q] = a[2] + gx * (cx - a[0]) + gy * (cy - a[1])
    elif cls == _WALL:
        line = _wall_line(a, b, c, lat.tol)
        if line < 0:
            c0, c1 = _cell_range(
                wp.min(wp.min(a[0], b[0]), c[0]), wp.max(wp.max(a[0], b[0]), c[0]), lat.x0, lat.dx, lat.tu, lat.cols
            )
            r0, _r1 = _cell_range(
                wp.min(wp.min(a[1], b[1]), c[1]), wp.max(wp.max(a[1], b[1]), c[1]), lat.y0, lat.dy, lat.tu, lat.rows
            )
            ncx = c1 - c0 + 1
            kind = _INC_WALL_GENERAL
            cell = (r0 + local // ncx) * lat.cols + c0 + local % ncx
        else:
            # the line in cell units, the run along it
            lv = (a[0] + b[0] + c[0]) / wp.float64(3.0)
            o_l = lat.x0
            s_l = lat.dx
            o_r = lat.y0
            s_r = lat.dy
            n_l = lat.cols
            n_r = lat.rows
            ra = a[1]
            rb = b[1]
            rc = c[1]
            if line == 1:
                lv = (a[1] + b[1] + c[1]) / wp.float64(3.0)
                o_l = lat.y0
                s_l = lat.dy
                o_r = lat.x0
                s_r = lat.dx
                n_l = lat.rows
                n_r = lat.cols
                ra = a[0]
                rb = b[0]
                rc = c[0]
            ql = (lv - o_l) / s_l
            on_edge = wp.abs(ql - wp.round(ql)) * s_l <= lat.tol
            k0, _k1 = _cell_range(wp.min(wp.min(ra, rb), rc), wp.max(wp.max(ra, rb), rc), o_r, s_r, lat.tu, n_r)
            kk = k0 + local // 2
            side = local % 2
            lo_r = o_r + wp.float64(kk) * s_r
            px = vec8d()
            pz = vec8d()
            px[0] = ra
            px[1] = rb
            px[2] = rc
            pz[0] = a[2]
            pz[1] = b[2]
            pz[2] = c[2]
            px, pz, n = _clip_side(px, pz, 3, 0, lo_r, wp.float64(-1.0))
            px, pz, n = _clip_side(px, pz, n, 0, lo_r + s_r, _F64_ONE)
            tmin = wp.float64(1.0e300)
            tmax = wp.float64(-1.0e300)
            top = wp.float64(-1.0e300)
            for k in range(n):
                tmin = wp.min(tmin, px[k])
                tmax = wp.max(tmax, px[k])
                top = wp.max(top, pz[k])
            t0 = (tmin - lo_r) / s_r
            t1 = (tmax - lo_r) / s_r
            if n >= 2 and (t1 - t0) > lat.tu:
                ext[0] = t0
                ext[1] = t1
                zz = wp.vec2d(_F64_ZERO, top)
                lc = int(0)
                ok = int(1)
                if on_edge:
                    le = int(wp.round(ql))
                    # side 0: the cell below the line (the wall on its high edge), side 1: the cell above
                    lc = le - 1 + side
                    kind = _INC_WALL_HI
                    if side == 1:
                        kind = _INC_WALL_LO
                    if lc < 0 or lc >= n_l:
                        ok = 0
                else:
                    lc = wp.clamp(int(wp.floor(ql)), 0, n_l - 1)
                    ext[2] = ql - wp.floor(ql)
                    kind = _INC_WALL_IN
                    if side == 1:
                        ok = 0
                if ok == 1:
                    if line == 0:
                        cell = kk * lat.cols + lc
                    else:
                        cell = lc * lat.cols + kk
                else:
                    kind = int(_INC_DEAD)
    if kind == _INC_DEAD:
        cell = int(lat.rows * lat.cols)
    inc_cell[i] = cell
    inc_kind[i] = kind
    inc_area[i] = area
    inc_corner[i] = corner
    inc_ext[i] = ext
    inc_z[i] = zz


# --- per cell --------------------------------------------------------------------------------------------------


@wp.func
def _step_level64(z: wp.vec4d, s: wp.float64, t: wp.float64, u: wp.float64, v: wp.float64, band: wp.float64):
    """A step cell's level at ``(u, v)``: the quadrant's corner; on a riser line (within ``band``) the higher side."""
    best = wp.float64(-1.0e300)
    for iu in range(2):
        for iv in range(2):
            ok_u = bool(False)
            if iu == 1:
                ok_u = u > s + band
            else:
                ok_u = not (u > s + band)
            if s > _F64_ZERO and wp.abs(u - s) <= band:
                ok_u = bool(True)
            ok_v = bool(False)
            if iv == 1:
                ok_v = v > t + band
            else:
                ok_v = not (v > t + band)
            if t > _F64_ZERO and wp.abs(v - t) <= band:
                ok_v = bool(True)
            if ok_u and ok_v:
                best = wp.max(best, z[iu + 2 * iv])
    return best


@wp.func
def _surface64(
    kind: int, diag: int, z: wp.vec4d, s: wp.float64, t: wp.float64, u: wp.float64, v: wp.float64, band: wp.float64
) -> wp.float64:
    """The surface of a law cell at cell-local ``(u, v)`` in float64."""
    if kind == _STEP:
        return _step_level64(z, s, t, u, v, band)
    if diag == 1:
        if u + v <= _F64_ONE:
            return z[0] + (z[1] - z[0]) * u + (z[2] - z[0]) * v
        return z[3] + (z[3] - z[2]) * (u - _F64_ONE) + (z[3] - z[1]) * (v - _F64_ONE)
    if u >= v:
        return z[0] + (z[1] - z[0]) * u + (z[3] - z[1]) * v
    return z[0] + (z[3] - z[2]) * u + (z[2] - z[0]) * v


@wp.func
def _edge_min64(
    kind: int, z: wp.vec4d, s: wp.float64, t: wp.float64, line_axis: int, high: int, t0: wp.float64, t1: wp.float64
) -> wp.float64:
    """Lowest surface along a cell's edge (``line_axis`` 0: ``u = 0`` or ``1``) over the run ``[t0, t1]``."""
    zs = z[0]
    ze = z[2]
    b = t
    if line_axis == 0:
        if high == 1:
            zs = z[1]
            ze = z[3]
    else:
        ze = z[1]
        b = s
        if high == 1:
            zs = z[2]
            ze = z[3]
    if kind == _STEP and b > _F64_ZERO:
        out = wp.float64(1.0e300)
        if t0 < b:
            out = zs
        if t1 > b:
            out = wp.min(out, ze)
        return out
    return wp.min(zs + (ze - zs) * t0, zs + (ze - zs) * t1)


@wp.kernel(enable_backward=False)
def _classify_cells(
    vs: wp.array[wp.vec3d],
    triangles: wp.array[wp.vec3i],
    tri_class: wp.array[wp.int32],
    cell_start: wp.array[wp.int32],
    order: wp.array[wp.int32],
    inc_tri: wp.array[wp.int32],
    inc_kind: wp.array[wp.int32],
    inc_area: wp.array[wp.float64],
    inc_corner: wp.array[wp.vec4d],
    inc_ext: wp.array[vec8d],
    inc_z: wp.array[wp.vec2d],
    lat: _Lattice,
    buried: wp.array[wp.int32],
    cell_kind: wp.array[wp.int32],
    cell_diag: wp.array[wp.int32],
    cell_z: wp.array[wp.vec4d],
    cell_off: wp.array[wp.vec2d],
    stats: wp.array[wp.int32],
    thickness: wp.array[wp.float64],
):
    """One cell: its kind, corners and offsets from its pieces, then the checks of its bottoms and its walls inside
    it. ``order[cell_start[c]:cell_start[c + 1]]`` lists the cell's incidences in triangle order."""
    cid = wp.tid()
    r = cid // lat.cols
    cc = cid - r * lat.cols
    bx0 = lat.x0 + wp.float64(cc) * lat.dx
    by0 = lat.y0 + wp.float64(r) * lat.dy
    cell_area = lat.dx * lat.dy
    tol = lat.tol
    tu = lat.tu
    i0 = cell_start[cid]
    i1 = cell_start[cid + 1]
    # stacked closed solids: an up-facing piece under the cell's one top plane, every bottom lower still, is buried
    up_area = wp.float64(0.0)
    top_i = int(-1)
    for p in range(i0, i1):
        i = order[p]
        if inc_kind[i] == _INC_FACE and tri_class[inc_tri[i]] == _UP:
            up_area += inc_area[i]
            if top_i < 0 or inc_z[i][0] > inc_z[top_i][0]:
                top_i = i
    if up_area > cell_area + lat.area_tol:
        top = inc_corner[top_i]
        tmin = wp.min(wp.min(top[0], top[1]), wp.min(top[2], top[3]))
        top_area = wp.float64(0.0)
        all_below = bool(True)
        lowest_buried = wp.float64(1.0e300)
        highest_bottom = wp.float64(-1.0e300)
        for p in range(i0, i1):
            i = order[p]
            if inc_kind[i] != _INC_FACE:
                continue
            if tri_class[inc_tri[i]] == _UP:
                d = wp.max(
                    wp.max(wp.abs(inc_corner[i][0] - top[0]), wp.abs(inc_corner[i][1] - top[1])),
                    wp.max(wp.abs(inc_corner[i][2] - top[2]), wp.abs(inc_corner[i][3] - top[3])),
                )
                if d <= _F64_TOL_PLANE:
                    top_area += inc_area[i]
                else:
                    if not (inc_z[i][1] < tmin - _F64_TOL_PLANE):
                        all_below = bool(False)
                    lowest_buried = wp.min(lowest_buried, inc_z[i][0])
            else:
                highest_bottom = wp.max(highest_bottom, inc_z[i][1])
        if (
            wp.abs(top_area - cell_area) <= lat.area_tol
            and all_below
            and highest_bottom < lowest_buried - _F64_TOL_PLANE
        ):
            wp.atomic_add(stats, 1, 1)
            for p in range(i0, i1):
                i = order[p]
                if inc_kind[i] == _INC_FACE and tri_class[inc_tri[i]] == _UP:
                    d = wp.max(
                        wp.max(wp.abs(inc_corner[i][0] - top[0]), wp.abs(inc_corner[i][1] - top[1])),
                        wp.max(wp.abs(inc_corner[i][2] - top[2]), wp.abs(inc_corner[i][3] - top[3])),
                    )
                    if d > _F64_TOL_PLANE:
                        buried[i] = 1
        wp.atomic_add(stats, 0, 1)
    # up to four planes, each led by the largest piece not yet grouped (the first in triangle order on ties)
    plane = wp.mat44d()
    gext = mat48d()
    garea = wp.vec4d()
    nplanes = int(0)
    extra = int(0)
    area_up = wp.float64(0.0)
    for _k in range(4):
        lead = int(-1)
        for p in range(i0, i1):
            i = order[p]
            if inc_kind[i] != _INC_FACE or tri_class[inc_tri[i]] != _UP or buried[i] == 1:
                continue
            g = _group_of(plane, nplanes, inc_corner[i])
            if g >= 0:
                continue
            if lead < 0 or inc_area[i] > inc_area[lead]:
                lead = i
        if lead >= 0:
            for q in range(4):
                plane[nplanes, q] = inc_corner[lead][q]
            for e in range(8):
                if e % 2 == 0:
                    gext[nplanes, e] = _F64_BIG
                else:
                    gext[nplanes, e] = -_F64_BIG
            nplanes += 1
    for p in range(i0, i1):
        i = order[p]
        if inc_kind[i] != _INC_FACE or tri_class[inc_tri[i]] != _UP or buried[i] == 1:
            continue
        area_up += inc_area[i]
        g = _group_of(plane, nplanes, inc_corner[i])
        if g < 0:
            extra += 1
            continue
        garea[g] = garea[g] + inc_area[i]
        for e in range(8):
            if e % 2 == 0:
                gext[g, e] = wp.min(gext[g, e], inc_ext[i][e])
            else:
                gext[g, e] = wp.max(gext[g, e], inc_ext[i][e])
    # A group's corner at a mesh vertex takes that vertex's height bit for bit (the first member in triangle order with
    # a vertex there): its plane evaluated there rounds (a height of 0 becomes 5.6e-17, a float32 height its neighbour).
    grouped = plane
    exact = int(0)
    for p in range(i0, i1):
        i = order[p]
        if inc_kind[i] != _INC_FACE or tri_class[inc_tri[i]] != _UP or buried[i] == 1:
            continue
        g = _group_of(grouped, nplanes, inc_corner[i])
        if g < 0:
            continue
        tv = triangles[inc_tri[i]]
        for j in range(3):
            v = vs[tv[j]]
            for q in range(4):
                cx = bx0
                cy = by0
                if q == 1 or q == 3:
                    cx = bx0 + lat.dx
                if q >= 2:
                    cy = by0 + lat.dy
                bit = 1 << (4 * g + q)
                if (exact & bit) == 0 and wp.abs(v[0] - cx) <= tol and wp.abs(v[1] - cy) <= tol:
                    exact = exact | bit
                    plane[g, q] = v[2]
    kind = int(_EMPTY)
    diag = int(0)
    z = wp.vec4d()
    s_cut = wp.float64(0.0)
    t_cut = wp.float64(0.0)
    if nplanes > 0:
        kind = int(_MIXED)
        z = wp.vec4d(plane[0, 0], plane[0, 1], plane[0, 2], plane[0, 3])
        covered = wp.abs(area_up - cell_area) <= lat.area_tol and extra == 0
        if covered and nplanes == 1:
            kind = _GRID
        elif covered and nplanes == 2:
            a_lr = gext[0, 4] >= -tu
            b_ul = gext[1, 5] <= tu
            a_ul = gext[0, 5] <= tu
            b_lr = gext[1, 4] >= -tu
            cont0 = (
                wp.abs(plane[0, 0] - plane[1, 0]) <= _F64_TOL_PLANE
                and wp.abs(plane[0, 3] - plane[1, 3]) <= _F64_TOL_PLANE
            )
            a_ll = gext[0, 7] <= _F64_ONE + tu
            b_ur = gext[1, 6] >= _F64_ONE - tu
            a_ur = gext[0, 6] >= _F64_ONE - tu
            b_ll = gext[1, 7] <= _F64_ONE + tu
            cont1 = (
                wp.abs(plane[0, 1] - plane[1, 1]) <= _F64_TOL_PLANE
                and wp.abs(plane[0, 2] - plane[1, 2]) <= _F64_TOL_PLANE
            )
            if cont0 and ((a_lr and b_ul) or (a_ul and b_lr)):
                kind = _GRID
                if not (a_lr and b_ul):
                    z[1] = plane[1, 1]
                    z[2] = plane[0, 2]
                else:
                    z[2] = plane[1, 2]
            elif cont1 and ((a_ll and b_ur) or (a_ur and b_ll)):
                kind = _GRID
                diag = 1
                if a_ll and b_ur:
                    z[3] = plane[1, 3]
                else:
                    z[0] = plane[1, 0]
        if kind == _MIXED and covered and nplanes >= 2:
            kind = _step_of(
                plane,
                gext,
                garea,
                nplanes,
                i0,
                i1,
                order,
                inc_tri,
                inc_kind,
                inc_ext,
                inc_corner,
                tri_class,
                buried,
                cell_area,
                tu,
            )
            if kind == _STEP:
                z, s_cut, t_cut = _step_levels_of(
                    plane, gext, nplanes, i0, i1, order, inc_tri, inc_kind, inc_ext, inc_corner, tri_class, buried, tu
                )
    law = kind == _GRID or kind == _STEP
    # vertices on the cell's corners give the corners' float32 bits
    if law:
        for p in range(i0, i1):
            i = order[p]
            if inc_kind[i] != _INC_FACE:
                continue
            t = inc_tri[i]
            for j in range(3):
                vtx = vs[triangles[t][j]]
                u = (vtx[0] - bx0) / lat.dx
                w = (vtx[1] - by0) / lat.dy
                if (u == _F64_ZERO or u == _F64_ONE) and (w == _F64_ZERO or w == _F64_ONE):
                    q = int(0)
                    if u == _F64_ONE:
                        q += 1
                    if w == _F64_ONE:
                        q += 2
                    if wp.abs(z[q] - vtx[2]) <= _F64_TOL_PLANE:
                        z[q] = vtx[2]
    # bottoms below the lowest corner; general walls; walls inside the cell not above the solid's top on their line
    if law:
        lowest = wp.min(wp.min(z[0], z[1]), wp.min(z[2], z[3]))
        for p in range(i0, i1):
            i = order[p]
            ik = inc_kind[i]
            if ik == _INC_FACE and tri_class[inc_tri[i]] == _DOWN:
                if inc_z[i][1] > lowest - _F64_TOL_PLANE:
                    kind = int(_MIXED)
                else:
                    wp.atomic_min(thickness, 0, lowest - inc_z[i][1])
            elif ik == _INC_WALL_GENERAL:
                kind = int(_MIXED)
            elif ik == _INC_WALL_IN and kind != _MIXED:
                t = inc_tri[i]
                line = _wall_line(vs[triangles[t][0]], vs[triangles[t][1]], vs[triangles[t][2]], tol)
                s_line = inc_ext[i][2]
                t0 = inc_ext[i][0]
                t1 = inc_ext[i][1]
                diag_t = s_line
                if diag == 1:
                    diag_t = _F64_ONE - s_line
                cross = t_cut
                if line == 1:
                    cross = s_cut
                gap = wp.float64(2.0) * tu
                lo = wp.float64(1.0e300)
                for k in range(5):
                    tt = t0
                    if k == 1:
                        tt = t1
                    elif k == 2:
                        tt = wp.clamp(diag_t, t0, t1)
                    elif k == 3:
                        tt = wp.clamp(cross - gap, t0, t1)
                    elif k == 4:
                        tt = wp.clamp(cross + gap, t0, t1)
                    u = s_line
                    w = tt
                    if line == 1:
                        u = tt
                        w = s_line
                    lo = wp.min(lo, _surface64(kind, diag, z, s_cut, t_cut, u, w, tu))
                if not (inc_z[i][1] <= lo + _F64_TOL_PLANE):
                    kind = int(_MIXED)
                    wp.atomic_add(stats, 4, 1)
                else:
                    on_riser = kind == _STEP and wp.abs(s_cut - s_line) <= tu
                    if line == 1:
                        on_riser = kind == _STEP and wp.abs(t_cut - s_line) <= tu
                    if on_riser:
                        wp.atomic_add(stats, 2, 1)
                    else:
                        wp.atomic_add(stats, 3, 1)
    cell_kind[cid] = kind
    cell_diag[cid] = diag
    cell_z[cid] = z
    cell_off[cid] = wp.vec2d(s_cut, t_cut)


@wp.func
def _group_of(plane: wp.mat44d, nplanes: int, corner: wp.vec4d) -> int:
    for g in range(nplanes):
        d = wp.max(
            wp.max(wp.abs(corner[0] - plane[g, 0]), wp.abs(corner[1] - plane[g, 1])),
            wp.max(wp.abs(corner[2] - plane[g, 2]), wp.abs(corner[3] - plane[g, 3])),
        )
        if d <= _F64_TOL_PLANE:
            return g
    return -1


@wp.func
def _cuts(gext: mat48d, nplanes: int, lo_col: int, hi_col: int, tu: wp.float64):
    """The one interior cut of the levels' extents along an axis (0 if none), and whether there is at most one."""
    cmin = wp.float64(1.0e300)
    cmax = wp.float64(-1.0e300)
    any_in = bool(False)
    for g in range(nplanes):
        for e in range(2):
            val = gext[g, lo_col]
            if e == 1:
                val = gext[g, hi_col]
            if val > tu and val < _F64_ONE - tu:
                any_in = bool(True)
                cmin = wp.min(cmin, val)
                cmax = wp.max(cmax, val)
    if not any_in:
        return _F64_ZERO, True
    return cmin, (cmax - cmin) <= tu


@wp.func
def _quadrant_bounds(q: int, su: wp.float64, tv: wp.float64):
    u0 = wp.float64(0.0)
    u1 = su
    v0 = wp.float64(0.0)
    v1 = tv
    if q == 1 or q == 3:
        u0 = su
        u1 = _F64_ONE
    if q >= 2:
        v0 = tv
        v1 = _F64_ONE
    return u0, u1, v0, v1


@wp.func
def _step_of(
    plane: wp.mat44d,
    gext: mat48d,
    garea: wp.vec4d,
    nplanes: int,
    i0: int,
    i1: int,
    order: wp.array[wp.int32],
    inc_tri: wp.array[wp.int32],
    inc_kind: wp.array[wp.int32],
    inc_ext: wp.array[vec8d],
    inc_corner: wp.array[wp.vec4d],
    tri_class: wp.array[wp.int32],
    buried: wp.array[wp.int32],
    cell_area: wp.float64,
    tu: wp.float64,
) -> int:
    """Whether the levels form a 2 x 2 step: horizontal, one cut per axis at most, each quadrant one level's, each
    level's quadrants adding up to its area."""
    for g in range(nplanes):
        lo = wp.min(wp.min(plane[g, 0], plane[g, 1]), wp.min(plane[g, 2], plane[g, 3]))
        hi = wp.max(wp.max(plane[g, 0], plane[g, 1]), wp.max(plane[g, 2], plane[g, 3]))
        if hi - lo > _F64_TOL_PLANE:
            return _MIXED
    s_cut, s_one = _cuts(gext, nplanes, 0, 1, tu)
    t_cut, t_one = _cuts(gext, nplanes, 2, 3, tu)
    if not (s_one and t_one and (s_cut > _F64_ZERO or t_cut > _F64_ZERO)):
        return _MIXED
    su = s_cut
    if su <= _F64_ZERO:
        su = _F64_ONE
    tv = t_cut
    if tv <= _F64_ZERO:
        tv = _F64_ONE
    held = wp.vec4d()
    for q in range(4):
        u0, u1, v0, v1 = _quadrant_bounds(q, su, tv)
        qarea = (u1 - u0) * (v1 - v0)
        q_lo = int(4)
        q_hi = int(-1)
        for p in range(i0, i1):
            i = order[p]
            if inc_kind[i] != _INC_FACE or tri_class[inc_tri[i]] != _UP or buried[i] == 1:
                continue
            g = _group_of(plane, nplanes, inc_corner[i])
            if g < 0:
                continue
            e = inc_ext[i]
            if wp.min(e[1], u1) - wp.max(e[0], u0) > tu and wp.min(e[3], v1) - wp.max(e[2], v0) > tu:
                q_lo = wp.min(q_lo, g)
                q_hi = wp.max(q_hi, g)
        if qarea > tu:
            if q_lo != q_hi or q_hi < 0:
                return _MIXED
            held[q_lo] = held[q_lo] + qarea
    for g in range(nplanes):
        if wp.abs(held[g] - garea[g] / cell_area) > wp.float64(1.0e-6):
            return _MIXED
    return _STEP


@wp.func
def _step_levels_of(
    plane: wp.mat44d,
    gext: mat48d,
    nplanes: int,
    i0: int,
    i1: int,
    order: wp.array[wp.int32],
    inc_tri: wp.array[wp.int32],
    inc_kind: wp.array[wp.int32],
    inc_ext: wp.array[vec8d],
    inc_corner: wp.array[wp.vec4d],
    tri_class: wp.array[wp.int32],
    buried: wp.array[wp.int32],
    tu: wp.float64,
):
    """A step's corners (each quadrant's own level: the highest piece over it) and offsets."""
    s_cut, _s_one = _cuts(gext, nplanes, 0, 1, tu)
    t_cut, _t_one = _cuts(gext, nplanes, 2, 3, tu)
    su = s_cut
    if su <= _F64_ZERO:
        su = _F64_ONE
    tv = t_cut
    if tv <= _F64_ZERO:
        tv = _F64_ONE
    z = wp.vec4d()
    for q in range(4):
        u0, u1, v0, v1 = _quadrant_bounds(q, su, tv)
        lev = wp.float64(-1.0e300)
        for p in range(i0, i1):
            i = order[p]
            if inc_kind[i] != _INC_FACE or tri_class[inc_tri[i]] != _UP or buried[i] == 1:
                continue
            e = inc_ext[i]
            if wp.min(e[1], u1) - wp.max(e[0], u0) > tu and wp.min(e[3], v1) - wp.max(e[2], v0) > tu:
                lev = wp.max(lev, inc_corner[i][0])
        if lev > -_F64_BIG:
            z[q] = lev
    if s_cut <= _F64_ZERO:
        z[1] = z[0]
        z[3] = z[2]
    if t_cut <= _F64_ZERO:
        z[2] = z[0]
        z[3] = z[1]
    return z, s_cut, t_cut


@wp.kernel(enable_backward=False)
def _check_edge_walls(
    cell_start: wp.array[wp.int32],
    order: wp.array[wp.int32],
    inc_kind: wp.array[wp.int32],
    inc_ext: wp.array[vec8d],
    inc_z: wp.array[wp.vec2d],
    inc_tri: wp.array[wp.int32],
    vs: wp.array[wp.vec3d],
    triangles: wp.array[wp.vec3i],
    lat: _Lattice,
    kind_in: wp.array[wp.int32],
    cell_z: wp.array[wp.vec4d],
    cell_off: wp.array[wp.vec2d],
    kind_out: wp.array[wp.int32],
    stats: wp.array[wp.int32],
):
    """A wall on a cell edge must not reach above the higher side's lowest surface along its run; beside a cell that
    is not a law cell it is that cell's triangle. Each side checks the wall on its own edge."""
    cid = wp.tid()
    kind = kind_in[cid]
    r = cid // lat.cols
    c = cid - r * lat.cols
    if kind == _GRID or kind == _STEP:
        for p in range(cell_start[cid], cell_start[cid + 1]):
            i = order[p]
            ik = inc_kind[i]
            if ik != _INC_WALL_HI and ik != _INC_WALL_LO:
                continue
            t = inc_tri[i]
            line = _wall_line(vs[triangles[t][0]], vs[triangles[t][1]], vs[triangles[t][2]], lat.tol)
            # the neighbour across the edge
            nb = int(-1)
            if line == 0:
                if ik == _INC_WALL_HI and c + 1 < lat.cols:
                    nb = cid + 1
                elif ik == _INC_WALL_LO and c > 0:
                    nb = cid - 1
            else:
                if ik == _INC_WALL_HI and r + 1 < lat.rows:
                    nb = cid + lat.cols
                elif ik == _INC_WALL_LO and r > 0:
                    nb = cid - lat.cols
            t0 = inc_ext[i][0]
            t1 = inc_ext[i][1]
            own_high = int(0)
            if ik == _INC_WALL_HI:
                own_high = 1
            floor_z = _edge_min64(kind, cell_z[cid], cell_off[cid][0], cell_off[cid][1], line, own_high, t0, t1)
            beside_mixed = bool(False)
            if nb >= 0:
                kn = kind_in[nb]
                if kn == _GRID or kn == _STEP:
                    floor_z = wp.max(
                        floor_z,
                        _edge_min64(kn, cell_z[nb], cell_off[nb][0], cell_off[nb][1], line, 1 - own_high, t0, t1),
                    )
                else:
                    beside_mixed = bool(True)
            if beside_mixed:
                wp.atomic_add(stats, 7, 1)
            elif inc_z[i][1] <= floor_z + _F64_TOL_PLANE:
                wp.atomic_add(stats, 5, 1)
            else:
                wp.atomic_add(stats, 6, 1)
                kind = int(_MIXED)
    kind_out[cid] = kind


@wp.func
def _profile(kind: int, z: wp.vec4f, off: wp.vec2f, line_axis: int, high: int):
    zs = wp.float64(z[0])
    ze = wp.float64(z[2])
    b = wp.float64(off[1])
    if line_axis == 0:
        if high == 1:
            zs = wp.float64(z[1])
            ze = wp.float64(z[3])
    else:
        ze = wp.float64(z[1])
        b = wp.float64(off[0])
        if high == 1:
            zs = wp.float64(z[2])
            ze = wp.float64(z[3])
    if not (kind == _STEP and b > _F64_ZERO):
        b = wp.float64(-1.0)
    return zs, ze, b


@wp.kernel(enable_backward=False)
def _edge_bits(
    kind: wp.array[wp.int32],
    cell_z: wp.array[wp.vec4f],
    cell_off: wp.array[wp.vec2f],
    lat: _Lattice,
    bits: wp.array[wp.int32],
):
    """Each law cell's edges whose neighbour's surface along the edge differs, or that border a cell that is no law
    cell or the table's outside."""
    cid = wp.tid()
    k = kind[cid]
    out = int(0)
    if k == _GRID or k == _STEP:
        r = cid // lat.cols
        c = cid - r * lat.cols
        eps = wp.float64(TERRAIN_CELL_RISER_EPS)
        for e in range(4):
            line_axis = e // 2
            high = e % 2
            nb = int(-1)
            if e == 0 and c > 0:
                nb = cid - 1
            elif e == 1 and c + 1 < lat.cols:
                nb = cid + 1
            elif e == 2 and r > 0:
                nb = cid - lat.cols
            elif e == 3 and r + 1 < lat.rows:
                nb = cid + lat.cols
            riser = bool(True)
            if nb >= 0:
                kn = kind[nb]
                if kn == _GRID or kn == _STEP:
                    a0, a1, ba = _profile(k, cell_z[cid], cell_off[cid], line_axis, high)
                    b0, b1, bb = _profile(kn, cell_z[nb], cell_off[nb], line_axis, 1 - high)
                    same_break = (ba < _F64_ZERO and bb < _F64_ZERO) or wp.abs(ba - bb) <= wp.float64(1.0e-6)
                    if same_break and wp.abs(a0 - b0) <= eps and wp.abs(a1 - b1) <= eps:
                        riser = bool(False)
            if riser:
                out = out | (TERRAIN_CELL_EDGE_X0 << e)
    bits[cid] = out


@wp.kernel(enable_backward=False)
def _mixed_ranges(
    kind: wp.array[wp.int32],
    cell_start: wp.array[wp.int32],
    order: wp.array[wp.int32],
    inc_kind: wp.array[wp.int32],
    inc_z: wp.array[wp.vec2d],
    inc_tri: wp.array[wp.int32],
    vs: wp.array[wp.vec3d],
    triangles: wp.array[wp.vec3i],
    is_mixed: wp.array[wp.int32],
    entries: wp.array[wp.int32],
    zrange: wp.array[wp.vec2d],
):
    """Per cell: whether it is mixed, a mixed cell's triangle count and the z-range of its triangles
    (faces clipped to the cell, walls whole)."""
    cid = wp.tid()
    k = kind[cid]
    is_mixed[cid] = int(k == _MIXED)
    n = int(0)
    lo = wp.float64(1.0e300)
    hi = wp.float64(-1.0e300)
    if k == _MIXED:
        for p in range(cell_start[cid], cell_start[cid + 1]):
            i = order[p]
            n += 1
            if inc_kind[i] == _INC_FACE:
                lo = wp.min(lo, inc_z[i][0])
                hi = wp.max(hi, inc_z[i][1])
            else:
                t = inc_tri[i]
                for j in range(3):
                    zj = vs[triangles[t][j]][2]
                    lo = wp.min(lo, zj)
                    hi = wp.max(hi, zj)
    entries[cid] = n
    zrange[cid] = wp.vec2d(lo, hi)


@wp.kernel(enable_backward=False)
def _finalize_cells(
    kind: wp.array[wp.int32],
    zc: wp.array[wp.vec4d],
    off: wp.array[wp.vec2d],
    zrange: wp.array[wp.vec2d],
    cell_z: wp.array[wp.vec4f],
    off32: wp.array[wp.vec2f],
):
    """The float32 corners (a mixed cell's top in all four, an empty cell's zeros) and offsets."""
    cid = wp.tid()
    k = kind[cid]
    z = zc[cid]
    if k == _MIXED:
        top = zrange[cid][1]
        z = wp.vec4d(top, top, top, top)
    elif k == _EMPTY:
        z = wp.vec4d()
    cell_z[cid] = wp.vec4f(wp.float32(z[0]), wp.float32(z[1]), wp.float32(z[2]), wp.float32(z[3]))
    o = wp.vec2f()
    if k == _STEP:
        o = wp.vec2f(wp.float32(off[cid][0]), wp.float32(off[cid][1]))
    off32[cid] = o


@wp.kernel(enable_backward=False)
def _vertex_top(kind: wp.array[wp.int32], cell_z: wp.array[wp.vec4f], lat: _Lattice, vz: wp.array[wp.float32]):
    """The shared vertex heights: the highest corner of the law cells around each vertex (the solid's top there)."""
    cid = wp.tid()
    k = kind[cid]
    if k == _GRID or k == _STEP:
        r = cid // lat.cols
        c = cid - r * lat.cols
        vc = lat.cols + 1
        z = cell_z[cid]
        wp.atomic_max(vz, r * vc + c, z[0])
        wp.atomic_max(vz, r * vc + c + 1, z[1])
        wp.atomic_max(vz, (r + 1) * vc + c, z[2])
        wp.atomic_max(vz, (r + 1) * vc + c + 1, z[3])


@wp.kernel(enable_backward=False)
def _vertex_fill(vz: wp.array[wp.float32]):
    i = wp.tid()
    if vz[i] < wp.float32(-1.0e30):
        vz[i] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def _needs_record(
    kind: wp.array[wp.int32],
    cell_z: wp.array[wp.vec4f],
    vz: wp.array[wp.float32],
    lat: _Lattice,
    has_record: wp.array[wp.int32],
):
    """A law cell needs a record when it is a step or a corner differs from the shared vertex height."""
    cid = wp.tid()
    k = kind[cid]
    need = int(0)
    if k == _STEP:
        need = 1
    elif k == _GRID:
        r = cid // lat.cols
        c = cid - r * lat.cols
        vc = lat.cols + 1
        z = cell_z[cid]
        if z[0] != vz[r * vc + c] or z[1] != vz[r * vc + c + 1]:
            need = 1
        if z[2] != vz[(r + 1) * vc + c] or z[3] != vz[(r + 1) * vc + c + 1]:
            need = 1
    has_record[cid] = need


@wp.kernel(enable_backward=False)
def _write_table(
    kind: wp.array[wp.int32],
    diag: wp.array[wp.int32],
    bits: wp.array[wp.int32],
    off32: wp.array[wp.vec2f],
    zrange: wp.array[wp.vec2d],
    cell_z: wp.array[wp.vec4f],
    has_record: wp.array[wp.int32],
    record_rank: wp.array[wp.int32],
    mixed_rank: wp.array[wp.int32],
    entry_start: wp.array[wp.int32],
    cell_start: wp.array[wp.int32],
    order: wp.array[wp.int32],
    inc_tri: wp.array[wp.int32],
    cell_flags: wp.array[wp.uint32],
    record_z: wp.array[wp.vec4f],
    record_offsets: wp.array[wp.vec2f],
    mixed_cell: wp.array[wp.int32],
    mixed_start: wp.array[wp.int32],
    mixed_triangles: wp.array[wp.int32],
    mixed_z: wp.array[wp.vec2f],
):
    cid = wp.tid()
    k = kind[cid]
    flags = wp.uint32(k)
    if k == _GRID:
        flags = flags | wp.uint32(diag[cid] * TERRAIN_CELL_DIAGONAL)
    if has_record[cid] == 1:
        ri = record_rank[cid] - 1
        flags = flags | wp.uint32(TERRAIN_CELL_RECORD) | (wp.uint32(ri) << wp.uint32(TERRAIN_CELL_PAYLOAD_SHIFT))
        record_z[ri] = cell_z[cid]
        record_offsets[ri] = off32[cid]
    if k == _MIXED:
        mi = mixed_rank[cid] - 1
        flags = flags | (wp.uint32(mi) << wp.uint32(TERRAIN_CELL_PAYLOAD_SHIFT))
        top = zrange[cid][1]
        mixed_cell[mi] = cid
        e0 = entry_start[cid]
        mixed_start[mi] = e0
        mixed_z[mi] = wp.vec2f(wp.float32(zrange[cid][0]), wp.float32(top))
        n = int(0)
        for p in range(cell_start[cid], cell_start[cid + 1]):
            mixed_triangles[e0 + n] = inc_tri[order[p]]
            n += 1
    if k == _GRID or k == _STEP:
        flags = flags | wp.uint32(bits[cid])
    cell_flags[cid] = flags


# --- the host side: launches, scans and the lattice's few distinct values ---------------------------------------


def _scan(counts: wp.array, device) -> tuple[wp.array, int]:
    """Exclusive prefix sum with the total appended (``count + 1`` entries)."""
    n = counts.shape[0]
    out = wp.zeros(n + 1, dtype=wp.int32, device=device)
    if n:
        wp.utils.array_scan(counts, out[1:], inclusive=True)
    return out, int(out[n : n + 1].numpy()[0])


@wp.kernel(enable_backward=False)
def _cell_counts(inc_cell: wp.array[wp.int32], ncell: int, counts: wp.array[wp.int32]):
    i = wp.tid()
    c = inc_cell[i]
    if c < ncell:
        wp.atomic_add(counts, c, 1)


def terrain_mesh_lattice(vertices: np.ndarray, triangles: np.ndarray, device=None) -> dict:
    """The mesh's lattice: per axis, the most frequent axis-aligned edge length of the up-facing triangles (the cell
    size) and the most frequent vertex coordinate modulo it (the origin), refined together by a least-squares fit of
    the distinct coordinates on the lattice, with the shares that support them. The device reduces the mesh to its
    distinct edge lengths and coordinates (with counts); the fit runs on those."""
    device = wp.get_device(device)
    v32 = wp.array(np.asarray(vertices, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)
    tri = wp.array(np.asarray(triangles, dtype=np.int32).reshape(-1, 3), dtype=wp.vec3i, device=device)
    return _lattice_device(v32, tri, device)[0]


def build_terrain_mesh_cells(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    cell_size: tuple[float, float] | None = None,
    origin: tuple[float, float] | None = None,
    device=None,
) -> TerrainMeshCells:
    """Classify a static mesh's lattice cells and write its cell table, on ``device`` (the model's; the default
    device when omitted). Every device runs the same kernels: the CPU build is the same build.

    Args:
        vertices: Vertex positions [m] in the mesh's (scaled) frame, shape ``(N, 3)``.
        triangles: Vertex indices, shape ``(M, 3)`` or ``(3 M,)``.
        cell_size: Cell edge lengths ``(dx, dy)`` [m]; the lattice's (:func:`terrain_mesh_lattice`) when omitted.
        origin: A lattice point ``(x0, y0)`` [m]; the lattice's when omitted.
        device: The Warp device to build on.

    Returns:
        The cell table (host arrays).
    """
    device = wp.get_device(device)
    v32 = wp.array(np.asarray(vertices, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)
    tri = wp.array(np.asarray(triangles, dtype=np.int32).reshape(-1, 3), dtype=wp.vec3i, device=device)
    m = tri.shape[0]
    lattice, tri_class = _lattice_device(v32, tri, device)
    tol = lattice["tolerance"]
    if cell_size is None:
        cell_size = (lattice["x"]["size"], lattice["y"]["size"])
    if origin is None:
        origin = (lattice["x"]["origin"], lattice["y"]["origin"])
    dx, dy = float(cell_size[0]), float(cell_size[1])
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("the mesh has no lattice: no axis-aligned up-facing edges")
    vs, moved, extent = _snap_device(v32, origin, (dx, dy), tol, device)
    c_lo = int(np.floor((extent[0][0] - origin[0]) / dx + tol / dx))
    r_lo = int(np.floor((extent[1][0] - origin[1]) / dy + tol / dy))
    x0 = origin[0] + c_lo * dx
    y0 = origin[1] + r_lo * dy
    cols = max(1, int(np.ceil((extent[0][1] - x0) / dx - tol / dx)))
    rows = max(1, int(np.ceil((extent[1][1] - y0) / dy - tol / dy)))
    ncell = rows * cols
    lat = _Lattice()
    lat.x0, lat.y0, lat.dx, lat.dy = x0, y0, dx, dy
    lat.rows, lat.cols = rows, cols
    lat.tol = tol
    lat.tu = tol / min(dx, dy)
    lat.area_tol = 1.0e-7 * dx * dy  # coverage sums of exact float64 pieces in cell-local coordinates
    lat.piece_min = 1.0e-9 * dx * dy  # float64 slivers along a cell edge are no piece

    # incidences: per triangle its candidate cells, laid out in triangle order, then grouped by cell (stable)
    tri_counts = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(_incidence_counts, dim=m, inputs=[vs, tri, tri_class, lat], outputs=[tri_counts], device=device)
    offsets, total = _scan(tri_counts, device)
    inc_cell = wp.empty(2 * max(total, 1), dtype=wp.int32, device=device)
    inc_tri = wp.empty(total, dtype=wp.int32, device=device)
    inc_kind = wp.empty(total, dtype=wp.int32, device=device)
    inc_area = wp.empty(total, dtype=wp.float64, device=device)
    inc_corner = wp.empty(total, dtype=wp.vec4d, device=device)
    inc_ext = wp.empty(total, dtype=vec8d, device=device)
    inc_z = wp.empty(total, dtype=wp.vec2d, device=device)
    wp.launch(
        _incidence_pieces,
        dim=total,
        inputs=[vs, tri, tri_class, offsets, m, lat],
        outputs=[inc_cell, inc_tri, inc_kind, inc_area, inc_corner, inc_ext, inc_z],
        device=device,
    )
    counts = wp.zeros(ncell, dtype=wp.int32, device=device)
    wp.launch(_cell_counts, dim=total, inputs=[inc_cell, ncell], outputs=[counts], device=device)
    cell_start, live = _scan(counts, device)
    order = wp.empty(2 * max(total, 1), dtype=wp.int32, device=device)
    wp.launch(_iota, dim=total, outputs=[order], device=device)
    if total:
        wp.utils.radix_sort_pairs(inc_cell, order, total)

    # per cell: classify, then the walls on its edges, then the table
    buried = wp.zeros(total, dtype=wp.int32, device=device)
    kind1 = wp.empty(ncell, dtype=wp.int32, device=device)
    diag = wp.empty(ncell, dtype=wp.int32, device=device)
    zc = wp.empty(ncell, dtype=wp.vec4d, device=device)
    off = wp.empty(ncell, dtype=wp.vec2d, device=device)
    stats_dev = wp.zeros(8, dtype=wp.int32, device=device)
    thickness = wp.array([np.inf], dtype=wp.float64, device=device)
    wp.launch(
        _classify_cells,
        dim=ncell,
        inputs=[vs, tri, tri_class, cell_start, order, inc_tri, inc_kind, inc_area, inc_corner, inc_ext, inc_z, lat],
        outputs=[buried, kind1, diag, zc, off, stats_dev, thickness],
        device=device,
    )
    kind = wp.empty(ncell, dtype=wp.int32, device=device)
    wp.launch(
        _check_edge_walls,
        dim=ncell,
        inputs=[cell_start, order, inc_kind, inc_ext, inc_z, inc_tri, vs, tri, lat, kind1, zc, off],
        outputs=[kind, stats_dev],
        device=device,
    )
    is_mixed = wp.empty(ncell, dtype=wp.int32, device=device)
    entries = wp.empty(ncell, dtype=wp.int32, device=device)
    zrange = wp.empty(ncell, dtype=wp.vec2d, device=device)
    wp.launch(
        _mixed_ranges,
        dim=ncell,
        inputs=[kind, cell_start, order, inc_kind, inc_z, inc_tri, vs, tri],
        outputs=[is_mixed, entries, zrange],
        device=device,
    )
    cell_z = wp.empty(ncell, dtype=wp.vec4f, device=device)
    off32 = wp.empty(ncell, dtype=wp.vec2f, device=device)
    wp.launch(_finalize_cells, dim=ncell, inputs=[kind, zc, off, zrange], outputs=[cell_z, off32], device=device)
    bits = wp.empty(ncell, dtype=wp.int32, device=device)
    wp.launch(_edge_bits, dim=ncell, inputs=[kind, cell_z, off32, lat], outputs=[bits], device=device)
    nvert = (rows + 1) * (cols + 1)
    vertex_z = wp.full(nvert, -np.inf, dtype=wp.float32, device=device)
    wp.launch(_vertex_top, dim=ncell, inputs=[kind, cell_z, lat], outputs=[vertex_z], device=device)
    wp.launch(_vertex_fill, dim=nvert, outputs=[vertex_z], device=device)
    has_record = wp.empty(ncell, dtype=wp.int32, device=device)
    wp.launch(_needs_record, dim=ncell, inputs=[kind, cell_z, vertex_z, lat], outputs=[has_record], device=device)
    record_rank = wp.empty(ncell, dtype=wp.int32, device=device)
    mixed_rank = wp.empty(ncell, dtype=wp.int32, device=device)
    wp.utils.array_scan(has_record, record_rank, inclusive=True)
    wp.utils.array_scan(is_mixed, mixed_rank, inclusive=True)
    entry_start, nentries = _scan(entries, device)
    nrec = int(record_rank[ncell - 1 : ncell].numpy()[0])
    nmixed = int(mixed_rank[ncell - 1 : ncell].numpy()[0])
    cell_flags = wp.empty(ncell, dtype=wp.uint32, device=device)
    record_z = wp.empty(max(nrec, 1), dtype=wp.vec4f, device=device)
    record_offsets = wp.empty(max(nrec, 1), dtype=wp.vec2f, device=device)
    mixed_cell = wp.empty(max(nmixed, 1), dtype=wp.int32, device=device)
    mixed_start = wp.empty(nmixed + 1, dtype=wp.int32, device=device)
    mixed_triangles = wp.empty(max(nentries, 1), dtype=wp.int32, device=device)
    mixed_z = wp.empty(max(nmixed, 1), dtype=wp.vec2f, device=device)
    wp.launch(
        _write_table,
        dim=ncell,
        inputs=[
            kind,
            diag,
            bits,
            off32,
            zrange,
            cell_z,
            has_record,
            record_rank,
            mixed_rank,
            entry_start,
            cell_start,
            order,
            inc_tri,
        ],
        outputs=[cell_flags, record_z, record_offsets, mixed_cell, mixed_start, mixed_triangles, mixed_z],
        device=device,
    )
    stacked = stats_dev.numpy()
    mstart = mixed_start.numpy()
    if nmixed:
        mstart[nmixed] = nentries
    else:
        mstart = np.zeros(1, dtype=np.int32)
    cls = np.bincount(tri_class.numpy(), minlength=4)
    stats = {
        "lattice": lattice,
        "triangles": int(m),
        "up": int(cls[0]),
        "down": int(cls[1]),
        "walls": int(cls[2]),
        "degenerate": int(cls[3]),
        "incidences": int(live),
        "stacked_cells": {"cells": int(stacked[0]), "one_top_plane_over_buried": int(stacked[1])},
        "buried_pieces": int(buried.numpy().sum()),
        "wall_checks": dict(
            zip(
                ("interior_riser", "interior_buried", "interior_bad", "edge_ok", "edge_bad", "edge_beside_mixed"),
                (int(x) for x in stacked[2:8]),
                strict=True,
            )
        ),
        "bottoms": {"min_thickness": float(thickness.numpy()[0]) if np.isfinite(thickness.numpy()[0]) else None},
        "max_lattice_snap": moved,
        "device": str(device),
    }
    return TerrainMeshCells(
        origin=(float(x0), float(y0)),
        cell_size=(dx, dy),
        shape=(rows, cols),
        vertex_z=vertex_z.numpy().reshape(rows + 1, cols + 1),
        cell_flags=cell_flags.numpy(),
        mixed_cell=mixed_cell.numpy()[:nmixed],
        mixed_start=mstart.astype(np.int32),
        mixed_triangles=mixed_triangles.numpy()[:nentries],
        mixed_z=mixed_z.numpy().reshape(-1, 2)[:nmixed],
        record_z=record_z.numpy().reshape(-1, 4)[:nrec],
        record_offsets=record_offsets.numpy().reshape(-1, 2)[:nrec],
        stats=stats,
    )


# ----------------------------------------------------------------------------------------------------------------
# verification (independent of the classifier) and the qualifying rule
# ----------------------------------------------------------------------------------------------------------------


def verify_terrain_mesh_cells(table: TerrainMeshCells, vertices: np.ndarray, triangles: np.ndarray) -> dict:
    """Check every triangle against the table's decoded float32 surface (:meth:`TerrainMeshCells.surface_z`).

    Test points are the triangle's pieces clipped to each cell its footprint covers, pulled ``1e-4`` of the way to
    the piece's centroid (strictly inside one cell and one side of a diagonal or riser). An up-facing point in a law
    cell must decode to the triangle's height, or lie inside the solid below it (a buried face of stacked solids,
    counted apart); a down-facing point must lie below the surface; a vertical triangle's
    points (its corners and its edges' midpoints) must not lie above the surface on both sides of the wall. Points
    in mixed cells are the triangle path's: their triangle must be in that cell's CSR list.

    Returns:
        Per-triangle counts (reproduced by law cells only, partly or wholly in mixed cells, failed), the largest
        height error of up-facing points [m] and in float32 ulps of the cell's largest corner, the CSR misses and the
        largest move of a vertex onto the lattice [m].
    """
    (x0, y0), (dx, dy), (rows, cols) = table.origin, table.cell_size, table.shape
    v = np.asarray(vertices, dtype=np.float32).astype(np.float64).reshape(-1, 3)
    f = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    tol = table.stats["lattice"]["tolerance"]
    v, snap = _snap_to_lattice(v, (x0, y0), (dx, dy), tol)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nrm = np.cross(b - a, c - a)
    nlen = np.linalg.norm(nrm, axis=1)
    nz = nrm[:, 2] / np.maximum(nlen, 1.0e-300)
    up = nz > 1.0e-4
    down = nz < -1.0e-4
    mc = table.mixed_cell.astype(np.int64)
    counts = np.diff(table.mixed_start.astype(np.int64))
    csr_keys = np.repeat(mc, counts) * (f.shape[0] + 1) + table.mixed_triangles.astype(np.int64)
    csr = np.sort(csr_keys)

    fail = np.zeros(f.shape[0], dtype=bool)
    in_mixed = np.zeros(f.shape[0], dtype=bool)
    buried = np.zeros(f.shape[0], dtype=bool)
    csr_miss = 0
    max_err = 0.0
    max_ulp = 0.0
    tol_z = 4.0 * _EPS32 * max(1.0, float(np.abs(v[:, 2]).max()))

    def check_mixed(tri, x, y):
        nonlocal csr_miss
        cq = np.floor((x - x0) / dx).astype(np.int64)
        rq = np.floor((y - y0) / dy).astype(np.int64)
        cell = np.clip(rq, 0, rows - 1) * cols + np.clip(cq, 0, cols - 1)
        key = cell * (f.shape[0] + 1) + tri
        pos = np.minimum(np.searchsorted(csr, key), max(csr.size - 1, 0))
        hit = (csr[pos] == key) if csr.size else np.zeros(key.size, dtype=bool)
        csr_miss += int((~hit).sum())
        fail[tri[~hit]] = True

    # faces: shrunk clipped pieces
    faces = np.flatnonzero(up | down)
    fx = np.stack([a[faces, 0], b[faces, 0], c[faces, 0]], 1)
    fy = np.stack([a[faces, 1], b[faces, 1], c[faces, 1]], 1)
    c0 = np.clip(np.floor((fx.min(1) - x0) / dx).astype(np.int64), 0, cols - 1)
    c1 = np.clip(np.floor((fx.max(1) - x0) / dx).astype(np.int64), 0, cols - 1)
    r0 = np.clip(np.floor((fy.min(1) - y0) / dy).astype(np.int64), 0, rows - 1)
    r1 = np.clip(np.floor((fy.max(1) - y0) / dy).astype(np.int64), 0, rows - 1)
    ncx = c1 - c0 + 1
    span = ncx * (r1 - r0 + 1)
    gx = -nrm[faces, 0] / nrm[faces, 2]
    gy = -nrm[faces, 1] / nrm[faces, 2]
    bounds = np.cumsum(span)
    start = 0
    while start < faces.size:
        stop = max(int(np.searchsorted(bounds, (bounds[start - 1] if start else 0) + _CHUNK, side="right")), start + 1)
        cnt = span[start:stop]
        idx = np.repeat(np.arange(start, stop), cnt)
        local = np.arange(idx.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cc = c0[idx] + local % ncx[idx]
        rr = r0[idx] + local // ncx[idx]
        bx0 = x0 + cc * dx
        by0 = y0 + rr * dy
        px = np.zeros((idx.size, _MAX_POLY))
        py = np.zeros((idx.size, _MAX_POLY))
        px[:, :3] = fx[idx] - bx0[:, None]
        py[:, :3] = fy[idx] - by0[:, None]
        n = np.full(idx.size, 3, dtype=np.int64)
        zero = np.zeros(idx.size)
        px, py, n = _clip_box(px, py, n, zero, zero + dx, zero, zero + dy)
        area = np.abs(_area(px, py, n))
        keep = (n >= 3) & (area > 1.0e-9 * dx * dy)
        idx, px, py, n = idx[keep], px[keep], py[keep], n[keep]
        slot = np.arange(_MAX_POLY)[None, :]
        valid = slot < n[:, None]
        cx = np.where(valid, px, 0.0).sum(1) / n
        cy = np.where(valid, py, 0.0).sum(1) / n
        qx = px + 1.0e-4 * (cx[:, None] - px) + bx0[keep][:, None]
        qy = py + 1.0e-4 * (cy[:, None] - py) + by0[keep][:, None]
        tri = faces[idx]
        rep = np.broadcast_to(tri[:, None], qx.shape)[valid]
        sx, sy = qx[valid], qy[valid]
        fi = np.broadcast_to(idx[:, None], qx.shape)[valid]
        zt = a[tri, 2][:, None].repeat(_MAX_POLY, 1)[valid] + gx[fi] * (sx - a[rep, 0]) + gy[fi] * (sy - a[rep, 1])
        zs, kd = table.surface_z(sx, sy)
        zs = zs.astype(np.float64)
        law = (kd == TERRAIN_CELL_GRID) | (kd == TERRAIN_CELL_STEP)
        is_up = up[rep]
        err = np.abs(zs - zt)
        exact = err <= tol_z * np.maximum(1.0, np.abs(zt))
        under = zt < zs - tol_z  # an up-facing piece inside the solid (stacked boxes)
        bad_up = law & is_up & ~exact & ~under
        bad_dn = law & ~is_up & ~(zt <= zs + tol_z)
        fail[rep[bad_up | bad_dn]] = True
        buried[rep[law & is_up & under]] = True
        if np.any(law & is_up & exact):
            e = err[law & is_up & exact]
            max_err = max(max_err, float(e.max()))
            # in float32 ulps of the cell's largest corner: the decode's terms are that large
            cq = np.clip(np.floor((sx[law & is_up & exact] - x0) / dx).astype(np.int64), 0, cols - 1)
            rq = np.clip(np.floor((sy[law & is_up & exact] - y0) / dy).astype(np.int64), 0, rows - 1)
            zmax = np.abs(table.cell_z[rq * cols + cq]).max(axis=1)
            ulp = e / np.spacing(np.maximum(zmax, np.float32(2.0**-20))).astype(np.float64)
            max_ulp = max(max_ulp, float(ulp.max()))
        mixed = (kd == TERRAIN_CELL_MIXED) | (kd == TERRAIN_CELL_EMPTY)
        in_mixed[rep[mixed]] = True
        if mixed.any():
            check_mixed(rep[mixed], sx[mixed], sy[mixed])
        start = stop

    # walls: corners and edge midpoints, pulled inside, tested on both sides of the wall
    wi = np.flatnonzero(~(up | down) & (nlen > 0.0))
    if wi.size:
        pts = np.concatenate(
            [np.stack([a[wi], b[wi], c[wi]], 1), 0.5 * np.stack([a[wi] + b[wi], b[wi] + c[wi], c[wi] + a[wi]], 1)], 1
        )
        cen = pts[:, :3].mean(1, keepdims=True)
        pts = pts + 1.0e-4 * (cen - pts)
        nh = nrm[wi, :2] / np.maximum(np.linalg.norm(nrm[wi, :2], axis=1, keepdims=True), 1.0e-300)
        delta = 1.0e-4 * min(dx, dy)
        tri = np.repeat(wi, 6)
        p = pts.reshape(-1, 3)
        nh6 = np.repeat(nh, 6, axis=0)
        za_, ka = table.surface_z(p[:, 0] + delta * nh6[:, 0], p[:, 1] + delta * nh6[:, 1])
        zb_, kb = table.surface_z(p[:, 0] - delta * nh6[:, 0], p[:, 1] - delta * nh6[:, 1])
        za_ = za_.astype(np.float64)
        zb_ = zb_.astype(np.float64)
        law_a = (ka == TERRAIN_CELL_GRID) | (ka == TERRAIN_CELL_STEP)
        law_b = (kb == TERRAIN_CELL_GRID) | (kb == TERRAIN_CELL_STEP)
        floor_z = np.maximum(np.where(law_a, za_, -np.inf), np.where(law_b, zb_, -np.inf))
        # beside a mixed cell the wall is that cell's triangle; elsewhere it must not stand above the law surface
        beside_mixed = (ka == TERRAIN_CELL_MIXED) | (kb == TERRAIN_CELL_MIXED)
        bad = (law_a | law_b) & ~beside_mixed & ~(p[:, 2] <= floor_z + tol_z)
        fail[tri[bad]] = True
        side_mixed = (ka == TERRAIN_CELL_MIXED) | (kb == TERRAIN_CELL_MIXED)
        in_mixed[tri[side_mixed]] = True
        for side, k in ((1.0, ka), (-1.0, kb)):
            m = k == TERRAIN_CELL_MIXED
            if m.any():
                check_mixed(tri[m], p[m, 0] + side * delta * nh6[m, 0], p[m, 1] + side * delta * nh6[m, 1])

    total = f.shape[0]
    return {
        "triangles": int(total),
        "reproduced": int((~fail & ~in_mixed).sum()),
        "in_mixed": int((~fail & in_mixed).sum()),
        "buried_up_facing": int((~fail & buried).sum()),
        "failed": int(fail.sum()),
        "failed_triangles": np.flatnonzero(fail),
        "max_height_error": max_err,
        "max_height_error_ulp": max_ulp,
        "csr_misses": csr_miss,
        "tolerance": tol_z,
        "max_lattice_snap": snap,
    }


TERRAIN_CELL_BVH_BYTES_PER_TRIANGLE = 64
"""Bytes per triangle of the mesh BVH the table retires (two 32-byte node halves per triangle)."""
TERRAIN_CELL_MIN_LATTICE_SHARE = 0.5
"""Least share of the up-facing vertices on the lattice, per axis, for the cells to follow the mesh's own structure."""


def terrain_mesh_cells_qualify(table: TerrainMeshCells, triangle_count: int) -> tuple[bool, str]:
    """The qualifying rule: whether a static mesh's cell table replaces its BVH (else the mesh keeps the generic path).

    A mixed cell is the triangle path through the cell's CSR list, and a law cell is exact (the table reproduces every
    triangle, :func:`verify_terrain_mesh_cells`), so the table never loses a pair to the BVH route: pairs that touch
    nothing leave at the cell z-range instead of a tree traversal. The rule therefore asks two things of the data:

    1. **Structure**: the lattice holds at least :data:`TERRAIN_CELL_MIN_LATTICE_SHARE` of the up-facing vertices on
       each axis, so that the cells follow the mesh rather than resample it (the Isaac Lab rough terrain: 97.8%; an
       unstructured soup: near 0, its "lattice" the most common edge length by chance);
    2. **Size**: the table is no larger than the BVH it retires,
       :data:`TERRAIN_CELL_BVH_BYTES_PER_TRIANGLE` bytes per triangle (the rough terrain: 50.3 MB against 76.4 MB). A
       lattice much finer than the mesh's triangles (few long sloped triangles) or triangles much smaller than a cell
       (long CSR lists) fail it.

    The law's share of the cells is reported, never required: it sets the time (law cells at the heightfield's cost,
    mixed cells at the triangle path's), not the choice.
    """
    lat = table.stats.get("lattice", {})
    share = min(lat.get("x", {}).get("vertex_share", 0.0), lat.get("y", {}).get("vertex_share", 0.0))
    if share < TERRAIN_CELL_MIN_LATTICE_SHARE:
        return (
            False,
            f"the lattice holds {share:.1%} of the up-facing vertices (< {TERRAIN_CELL_MIN_LATTICE_SHARE:.0%})",
        )
    size = table.memory_bytes()["total"]
    bvh = TERRAIN_CELL_BVH_BYTES_PER_TRIANGLE * triangle_count
    if size > bvh:
        return False, f"the table ({size / 2**20:.1f} MiB) is larger than the BVH it retires ({bvh / 2**20:.1f} MiB)"
    counts = table.counts()
    ncell = max(table.shape[0] * table.shape[1], 1)
    law = counts["grid"] + counts["step"]
    return True, (
        f"lattice share {share:.1%}, table {size / 2**20:.1f} MiB <= BVH {bvh / 2**20:.1f} MiB; "
        f"law cells {law / ncell:.2%}, mixed {counts['mixed'] / ncell:.2%}"
    )


def terrain_mesh_cells_law_surface(table: TerrainMeshCells) -> tuple[bool, str]:
    """Whether the terrain law reads the whole table as its one surface type, a lattice of vertex heights.

    The law reads ``vertex_z`` as it reads a heightfield's heights: every cell must be a grid cell on the heightfield's
    diagonal ``(0, 0)-(1, 1)`` whose corners are the shared vertex heights (no record), with no riser on an edge inside
    the table (edge bits only on the table's border, where the law's border walls stand). Step cells, risers on cell
    edges, the other diagonal, mixed and empty cells are what the law's riser features and the cell CSR add; until then
    a table with any of them keeps its mesh on the generic path.
    """
    flags = table.cell_flags
    rows, cols = table.shape
    counts = table.counts()
    if counts["grid"] != rows * cols:
        return False, f"cells the law does not read yet: {counts['step']} step, {counts['mixed']} mixed, " + (
            f"{counts['empty']} empty"
        )
    if np.any((flags & TERRAIN_CELL_DIAGONAL) != 0):
        return False, f"{int(np.count_nonzero(flags & TERRAIN_CELL_DIAGONAL))} grid cells on the diagonal (1, 0)-(0, 1)"
    if np.any((flags & TERRAIN_CELL_RECORD) != 0):
        return False, f"{int(np.count_nonzero(flags & TERRAIN_CELL_RECORD))} cells with corners of their own (risers)"
    grid = flags.reshape(rows, cols)
    inner = (
        np.count_nonzero(grid[:, 1:] & TERRAIN_CELL_EDGE_X0)
        + np.count_nonzero(grid[:, :-1] & TERRAIN_CELL_EDGE_X1)
        + np.count_nonzero(grid[1:, :] & TERRAIN_CELL_EDGE_Y0)
        + np.count_nonzero(grid[:-1, :] & TERRAIN_CELL_EDGE_Y1)
    )
    if inner:
        return False, f"{inner} cell edges with a riser inside the table"
    return True, f"{rows} x {cols} grid cells, the heightfield's diagonal, no riser"


def terrain_mesh_surface(
    vertices: np.ndarray, triangles: np.ndarray, device=None
) -> tuple[TerrainMeshCells | None, str]:
    """A static mesh's cell table when the terrain law takes the mesh as a surface, else ``None``; and why.

    The data decide, never the task: the lattice must hold the mesh's structure (:func:`terrain_mesh_cells_qualify`'s
    first test, checked on the lattice before anything is built), the table's size bound by the BVH it retires (checked
    on the lattice's extent first, so that a chance lattice of a mesh without one allocates nothing), the rule itself on
    the built table, and the law must read every cell (:func:`terrain_mesh_cells_law_surface`).

    Args:
        vertices: Vertex positions [m] in the shape's (scaled) frame, shape ``(N, 3)``.
        triangles: Vertex indices, shape ``(M, 3)`` or ``(3 M,)``.
        device: The Warp device to build on (the model's).
    """
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    m = len(triangles)
    if m == 0 or len(vertices) == 0:
        return None, "no triangles"
    lattice = terrain_mesh_lattice(vertices, triangles, device=device)
    share = min(lattice["x"]["vertex_share"], lattice["y"]["vertex_share"])
    dx, dy = lattice["x"]["size"], lattice["y"]["size"]
    if dx <= 0.0 or dy <= 0.0 or share < TERRAIN_CELL_MIN_LATTICE_SHARE:
        return None, f"the lattice holds {share:.1%} of the up-facing vertices (< {TERRAIN_CELL_MIN_LATTICE_SHARE:.0%})"
    lo = vertices[:, :2].min(axis=0).astype(np.float64)
    hi = vertices[:, :2].max(axis=0).astype(np.float64)
    cells = (np.floor((hi[0] - lo[0]) / dx) + 2.0) * (np.floor((hi[1] - lo[1]) / dy) + 2.0)
    bvh = TERRAIN_CELL_BVH_BYTES_PER_TRIANGLE * m
    if 8.0 * cells > bvh:
        return None, f"the lattice's {cells:.0f} cells outweigh the BVH they would retire ({bvh / 2**20:.1f} MiB)"
    table = build_terrain_mesh_cells(vertices, triangles, device=device)
    ok, reason = terrain_mesh_cells_qualify(table, m)
    if not ok:
        return None, reason
    law, why = terrain_mesh_cells_law_surface(table)
    if not law:
        return None, f"{reason}; the law does not read it whole: {why}"
    return table, f"{reason}; {why}"

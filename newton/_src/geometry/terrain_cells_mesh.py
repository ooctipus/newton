# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Terrain cells of a static triangle mesh: the lattice, the cell classes and the cell table.

A static terrain mesh is mostly a height function sampled on a lattice: heightfield patches converted to triangles,
horizontal treads and box tops, and axis-aligned risers between them. :func:`build_terrain_mesh_cells` recovers that
structure from the triangles alone, never from the task, and writes it as a cell table the terrain law can read where
it reads a :class:`~newton.Heightfield` today:

* **The lattice.** The cell size of each axis is the most frequent length of the up-facing triangles' axis-aligned
  edges, and the origin the most frequent vertex coordinate modulo that size (:func:`terrain_mesh_lattice`).
* **Cells.** Cell ``(r, c)`` covers ``[x0 + c dx, x0 + (c + 1) dx] x [y0 + r dy, y0 + (r + 1) dy]`` in the mesh frame,
  row-major like a heightfield's cells. Every cell is one of:

  - **grid**: the up-facing surface over the cell is one plane, or two half-cell planes meeting along a diagonal
    (``(0, 0)-(1, 1)`` as in a :class:`~newton.Heightfield`, or ``(1, 0)-(0, 1)``, bit
    :data:`TERRAIN_CELL_DIAGONAL`); its four corner heights give the surface exactly;
  - **step**: two horizontal levels split by one axis-aligned riser at the cell-local offset ``s`` (the line
    ``x = s dx``, or ``y = s dy`` with :data:`TERRAIN_CELL_STEP_Y`); its corners hold each side's level, ``s`` is a
    24-bit fraction of the cell in the payload;
  - **mixed**: anything else (a corner where two risers meet, a sloped piece beside a riser, a wall standing above
    both sides, an overhang, a partly covered cell); it keeps the mesh's own triangles as a CSR list with their
    z-range, for the triangle narrow phase without a BVH;
  - **empty**: no up-facing surface over the cell.

  Grid and step cells are the law's cells. The terrain is a solid under them, ``z <= h(x, y)``: a vertical triangle
  in a law cell must lie on its riser or on a cell edge and not reach above the surface beside it, and a down-facing
  triangle must lie below the cell's lowest corner (a box's bottom). Faces inside the solid (buried walls, bottoms)
  are reproduced by the solid itself.
* **Risers on cell edges.** Each cell holds its own four corners, so a riser on a lattice line (a stair's riser) is
  the difference between two neighbours' corners along their shared edge. The four edge bits
  (:data:`TERRAIN_CELL_EDGE_X0` ...) mark the edges whose neighbour's surface differs by more than
  :data:`TERRAIN_CELL_RISER_EPS` along the edge, or whose neighbour is not a law cell (mixed, empty, or outside the
  table): the edges where the law adds a riser, as it adds the border walls of a heightfield.

A heightfield is the special case with grid cells only, diagonal ``(0, 0)-(1, 1)`` and no edge bits inside its
footprint: its corners are its vertex heights.

The table (:class:`TerrainMeshCells`):

* ``cell_z``: float32 ``(cells, 4)``, the corners ``z00, z10, z01, z11`` (``z10`` at ``x + dx``); a mixed cell's four
  corners all hold its top, so the largest corner bounds every cell from above;
* ``cell_flags``: uint32 per cell, the kind (bits 0-1), the diagonal (bit 2), the step axis (bit 3), the edge bits
  (bits 4-7) and the payload (bits 8-31: a step's offset ``round(s 2^24)``, a mixed cell's index);
* ``mixed_cell``, ``mixed_start``, ``mixed_triangles``, ``mixed_z``: the mixed cells' cell ids, CSR offsets, triangle
  ids and z-ranges ``(lowest, highest)`` of their triangles clipped to the cell.

The build runs once on the CPU in float64 (the mesh's float32 vertices are exact in it); corners copy the heights of
the mesh's vertices at the cell corners bit for bit, and planes evaluated at the corners round once to float32.
:func:`verify_terrain_mesh_cells` checks the table against every triangle independently of the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TERRAIN_CELL_GRID = 0
"""Cell kind: one plane, or two half-cell planes, given by the four corners."""
TERRAIN_CELL_STEP = 1
"""Cell kind: two horizontal levels split by one axis-aligned riser at the payload's offset."""
TERRAIN_CELL_MIXED = 2
"""Cell kind: the mesh's own triangles (CSR list), for the triangle narrow phase."""
TERRAIN_CELL_EMPTY = 3
"""Cell kind: no up-facing surface over the cell."""
TERRAIN_CELL_KIND_MASK = 0x3
TERRAIN_CELL_DIAGONAL = 0x4
"""Grid cell: the halves meet on the diagonal ``(1, 0)-(0, 1)`` instead of the heightfield's ``(0, 0)-(1, 1)``."""
TERRAIN_CELL_STEP_Y = 0x8
"""Step cell: the riser is the line ``y = s dy`` (normal along y); without it, the line ``x = s dx``."""
TERRAIN_CELL_EDGE_X0 = 0x10
"""Riser on the cell's edge ``x = 0`` (shared with column ``c - 1``)."""
TERRAIN_CELL_EDGE_X1 = 0x20
"""Riser on the cell's edge ``x = dx`` (shared with column ``c + 1``)."""
TERRAIN_CELL_EDGE_Y0 = 0x40
"""Riser on the cell's edge ``y = 0`` (shared with row ``r - 1``)."""
TERRAIN_CELL_EDGE_Y1 = 0x80
"""Riser on the cell's edge ``y = dy`` (shared with row ``r + 1``)."""
TERRAIN_CELL_PAYLOAD_SHIFT = 8
TERRAIN_CELL_STEP_ONE = 1 << 24
"""A step's offset ``s`` is stored as ``round(s * TERRAIN_CELL_STEP_ONE)``, ``0 < s < 1`` of the cell."""
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
    cell_z: np.ndarray
    cell_flags: np.ndarray
    mixed_cell: np.ndarray
    mixed_start: np.ndarray
    mixed_triangles: np.ndarray
    mixed_z: np.ndarray
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
            "cell_z": self.cell_z.nbytes,
            "cell_flags": self.cell_flags.nbytes,
            "mixed_cell": self.mixed_cell.nbytes,
            "mixed_start": self.mixed_start.nbytes,
            "mixed_triangles": self.mixed_triangles.nbytes,
            "mixed_z": self.mixed_z.nbytes,
        }
        out["total"] = sum(out.values())
        return out

    def step_offset(self) -> np.ndarray:
        """Each cell's step offset ``s`` as a fraction of the cell (meaningful for step cells only)."""
        return (self.cell_flags >> TERRAIN_CELL_PAYLOAD_SHIFT).astype(np.float64) / TERRAIN_CELL_STEP_ONE

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
        s = self.step_offset()[cell].astype(np.float32)
        along = np.where((flags & TERRAIN_CELL_STEP_Y) != 0, v, u)
        far = np.where((flags & TERRAIN_CELL_STEP_Y) != 0, z01, z10)
        step = np.where(along < s, z00, np.where(along > s, far, np.maximum(z00, far)))
        out = np.where(kind == TERRAIN_CELL_GRID, grid, np.where(kind == TERRAIN_CELL_STEP, step, np.float32(np.nan)))
        out = np.where(inside, out, np.float32(np.nan))
        return out, np.where(inside, kind, -1)


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


def _masked(values, n, fill):
    slot = np.arange(values.shape[1])[None, :]
    return np.where(slot < n[:, None], values, fill)


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


def _snap_to_lattice(v: np.ndarray, origin, size, tol: float) -> tuple[np.ndarray, float]:
    """Move coordinates within ``tol`` of a lattice line onto it (the float32 rounding of lattice positions), and weld
    the other coordinates that lie within ``tol`` of each other to their smallest (two boxes' shared riser, rounded
    differently in float32); returns the moved vertices and the largest move [m]."""
    vs = v.copy()
    for axis in (0, 1):
        q = (v[:, axis] - origin[axis]) / size[axis]
        qr = np.round(q)
        on = np.abs(q - qr) * size[axis] <= tol
        vs[:, axis] = np.where(on, origin[axis] + qr * size[axis], v[:, axis])
        off = np.flatnonzero(~on)
        if off.size:
            vals, inv = np.unique(vs[off, axis], return_inverse=True)
            start = np.r_[True, np.diff(vals) > tol]
            group = np.cumsum(start) - 1
            first = vals[start][group]
            width = _segment_reduce(group, vals, int(group[-1]) + 1, np.maximum, -np.inf) - vals[start]
            weld = width[group] <= tol  # a chain wider than the tolerance is distinct features: left as it is
            vs[off, axis] = np.where(weld, first, vals)[inv]
    return vs, float(np.max(np.abs(vs[:, :2] - v[:, :2]))) if v.size else 0.0


# ----------------------------------------------------------------------------------------------------------------
# lattice
# ----------------------------------------------------------------------------------------------------------------


def terrain_mesh_lattice(vertices: np.ndarray, triangles: np.ndarray) -> dict:
    """The mesh's lattice: per axis, the most frequent axis-aligned edge length of the up-facing triangles (the cell
    size) and the most frequent vertex coordinate modulo it (the origin), with the shares that support them."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nrm = np.cross(b - a, c - a)
    up = nrm[:, 2] > 1.0e-4 * np.linalg.norm(nrm, axis=1)
    extent = float(np.max(np.abs(v[:, :2]))) if v.size else 1.0
    tol = 8.0 * _EPS32 * max(extent, 1.0)
    out = {"tolerance": tol}
    tri = np.stack([a[up], b[up], c[up]], 1)
    edges = np.concatenate([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]])
    verts = tri.reshape(-1, 3)
    for axis, name in ((0, "x"), (1, "y")):
        other = 1 - axis
        span = np.abs(edges[:, axis])
        aligned = (np.abs(edges[:, other]) <= tol) & (span > tol)
        size, share = _mode(span[aligned], tol)
        if size <= 0.0:
            out[name] = {"size": 0.0, "origin": 0.0, "edge_share": 0.0, "vertex_share": 0.0}
            continue
        size = float(np.median(span[aligned][np.abs(span[aligned] - size) <= tol]))
        coords = np.unique(verts[:, axis])
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
        q = (verts[:, axis] - origin) / size
        on = np.abs(q - np.round(q)) * size <= tol
        out[name] = {"size": size, "origin": origin, "edge_share": share, "vertex_share": float(on.mean())}
    return out


# ----------------------------------------------------------------------------------------------------------------
# the builder
# ----------------------------------------------------------------------------------------------------------------


def build_terrain_mesh_cells(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    cell_size: tuple[float, float] | None = None,
    origin: tuple[float, float] | None = None,
) -> TerrainMeshCells:
    """Classify a static mesh's lattice cells and write its cell table.

    Args:
        vertices: Vertex positions [m] in the mesh's (scaled) frame, shape ``(N, 3)``.
        triangles: Vertex indices, shape ``(M, 3)`` or ``(3 M,)``.
        cell_size: Cell edge lengths ``(dx, dy)`` [m]; the lattice's (:func:`terrain_mesh_lattice`) when omitted.
        origin: A lattice point ``(x0, y0)`` [m]; the lattice's when omitted.

    Returns:
        The cell table.
    """
    v32 = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    v = v32.astype(np.float64)
    f = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    lattice = terrain_mesh_lattice(v, f)
    tol = lattice["tolerance"]
    if cell_size is None:
        cell_size = (lattice["x"]["size"], lattice["y"]["size"])
    if origin is None:
        origin = (lattice["x"]["origin"], lattice["y"]["origin"])
    dx, dy = float(cell_size[0]), float(cell_size[1])
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("the mesh has no lattice: no axis-aligned up-facing edges")

    vs, _ = _snap_to_lattice(v, origin, (dx, dy), tol)
    lo = vs[:, :2].min(axis=0)
    hi = vs[:, :2].max(axis=0)
    c_lo = int(np.floor((lo[0] - origin[0]) / dx + tol / dx))
    r_lo = int(np.floor((lo[1] - origin[1]) / dy + tol / dy))
    x0 = origin[0] + c_lo * dx
    y0 = origin[1] + r_lo * dy
    cols = max(1, int(np.ceil((hi[0] - x0) / dx - tol / dx)))
    rows = max(1, int(np.ceil((hi[1] - y0) / dy - tol / dy)))
    ncell = rows * cols

    a, b, c = vs[f[:, 0]], vs[f[:, 1]], vs[f[:, 2]]
    nrm = np.cross(b - a, c - a)
    nlen = np.linalg.norm(nrm, axis=1)
    nz = nrm[:, 2] / np.maximum(nlen, 1.0e-300)
    up = nz > 1.0e-4
    down = nz < -1.0e-4
    wall = ~(up | down) & (nlen > 0.0)
    tri_z = np.stack([a[:, 2], b[:, 2], c[:, 2]], 1)

    stats: dict = {"lattice": lattice, "triangles": int(f.shape[0]), "up": int(up.sum()), "down": int(down.sum())}
    stats["walls"] = int(wall.sum())
    stats["degenerate"] = int((nlen == 0.0).sum())
    area_tol = 1.0e-7 * dx * dy  # coverage sums of exact float64 pieces in cell-local coordinates
    cell_area = dx * dy

    # --- incidences of up- and down-facing triangles with the cells their footprint covers -----------------------
    faces = np.flatnonzero(up | down)
    fx = np.stack([a[faces, 0], b[faces, 0], c[faces, 0]], 1)
    fy = np.stack([a[faces, 1], b[faces, 1], c[faces, 1]], 1)
    c0 = np.clip(np.floor((fx.min(1) - x0) / dx + tol / dx).astype(np.int64), 0, cols - 1)
    c1 = np.clip(np.ceil((fx.max(1) - x0) / dx - tol / dx).astype(np.int64) - 1, 0, cols - 1)
    r0 = np.clip(np.floor((fy.min(1) - y0) / dy + tol / dy).astype(np.int64), 0, rows - 1)
    r1 = np.clip(np.ceil((fy.max(1) - y0) / dy - tol / dy).astype(np.int64) - 1, 0, rows - 1)
    c1 = np.maximum(c1, c0)
    r1 = np.maximum(r1, r0)
    ncx = c1 - c0 + 1
    span = ncx * (r1 - r0 + 1)
    # plane z = pz + gx (x - xa) + gy (y - ya) of each face
    fn = nrm[faces]
    gx = -fn[:, 0] / fn[:, 2]
    gy = -fn[:, 1] / fn[:, 2]
    fa = a[faces]

    inc_cell, inc_face, inc_area, inc_zlo, inc_zhi = [], [], [], [], []
    inc_corner = []  # plane at the cell's four corners
    inc_ext = []  # u min/max, v min/max, (u - v) min/max, (u + v) min/max of the piece in cell units
    snap_cell, snap_corner, snap_z = [], [], []
    bounds = np.cumsum(span)
    start = 0
    while start < faces.size:
        stop = int(np.searchsorted(bounds, bounds[start - 1] + _CHUNK if start else _CHUNK, side="right"))
        stop = max(stop, start + 1)
        sl = slice(start, stop)
        cnt = span[sl]
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
        keep = (n >= 3) & (area > 1.0e-9 * dx * dy)  # drop float64 slivers along a cell edge
        idx, cc, rr, bx0, by0, px, py, n, area = (t[keep] for t in (idx, cc, rr, bx0, by0, px, py, n, area))
        u = px / dx
        w = py / dy
        px = px + bx0[:, None]  # back to the mesh frame for the planes
        py = py + by0[:, None]
        pz = (
            fa[idx, 2][:, None]
            + gx[idx][:, None] * (px - fa[idx, 0][:, None])
            + gy[idx][:, None] * (py - fa[idx, 1][:, None])
        )
        ext = np.stack(
            [
                _masked(u, n, np.inf).min(1),
                _masked(u, n, -np.inf).max(1),
                _masked(w, n, np.inf).min(1),
                _masked(w, n, -np.inf).max(1),
                _masked(u - w, n, np.inf).min(1),
                _masked(u - w, n, -np.inf).max(1),
                _masked(u + w, n, np.inf).min(1),
                _masked(u + w, n, -np.inf).max(1),
            ],
            1,
        )
        cxs = np.stack([bx0, bx0 + dx, bx0, bx0 + dx], 1)
        cys = np.stack([by0, by0, by0 + dy, by0 + dy], 1)
        corner = (
            fa[idx, 2][:, None]
            + gx[idx][:, None] * (cxs - fa[idx, 0][:, None])
            + gy[idx][:, None] * (cys - fa[idx, 1][:, None])
        )
        cell = rr * cols + cc
        inc_cell.append(cell)
        inc_face.append(idx)
        inc_area.append(area)
        inc_zlo.append(_masked(pz, n, np.inf).min(1))
        inc_zhi.append(_masked(pz, n, -np.inf).max(1))
        inc_corner.append(corner)
        inc_ext.append(ext)
        # mesh vertices that sit exactly on a cell corner: their float32 heights are the corners' bits
        for j in range(3):
            ux = (fx[idx, j] - bx0) / dx
            wy = (fy[idx, j] - by0) / dy
            at = ((ux == 0.0) | (ux == 1.0)) & ((wy == 0.0) | (wy == 1.0))
            snap_cell.append(cell[at])
            snap_corner.append((ux[at] == 1.0).astype(np.int64) + 2 * (wy[at] == 1.0).astype(np.int64))
            snap_z.append(v32[f[faces[idx[at]], j], 2])
        start = stop
    inc_cell = np.concatenate(inc_cell) if inc_cell else np.zeros(0, np.int64)
    inc_face = faces[np.concatenate(inc_face)] if inc_face else np.zeros(0, np.int64)
    inc_area = np.concatenate(inc_area) if inc_area else np.zeros(0)
    inc_zlo = np.concatenate(inc_zlo) if inc_zlo else np.zeros(0)
    inc_zhi = np.concatenate(inc_zhi) if inc_zhi else np.zeros(0)
    inc_corner = np.concatenate(inc_corner) if inc_corner else np.zeros((0, 4))
    inc_ext = np.concatenate(inc_ext) if inc_ext else np.zeros((0, 8))
    inc_up = up[inc_face]
    stats["face_incidences"] = int(inc_cell.size)

    # --- stacked closed solids: up-facing pieces inside the solid under one top plane --------------------------------
    # Going down a column from the top plane, the solid ends at the first down-facing face. An up-facing piece
    # below the top plane, with every down-facing piece of the cell lower still, lies inside the solid (boxes that
    # overlap, a platform over a box grid): it is buried, and the top plane alone is the cell's surface.
    buried = np.zeros(inc_cell.size, dtype=bool)
    upi = np.flatnonzero(inc_up)
    up_area = np.bincount(inc_cell[upi], weights=inc_area[upi], minlength=ncell)
    si = upi[up_area[inc_cell[upi]] > cell_area + area_tol]
    if si.size:
        so = si[np.lexsort((-inc_zlo[si], inc_cell[si]))]
        sc = inc_cell[so]
        sfirst = np.r_[True, sc[1:] != sc[:-1]]
        sseg = np.cumsum(sfirst) - 1
        top = inc_corner[so[sfirst]]
        on_top = np.max(np.abs(inc_corner[so] - top[sseg]), axis=1) <= _PLANE_TOL
        top_area = np.bincount(sseg[on_top], weights=inc_area[so][on_top], minlength=top.shape[0])
        below = inc_zhi[so] < top.min(axis=1)[sseg] - _PLANE_TOL
        all_below = np.bincount(sseg[~on_top & ~below], minlength=top.shape[0]) == 0
        lowest_buried = _segment_reduce(sseg[~on_top], inc_zlo[so][~on_top], top.shape[0], np.minimum, np.inf)
        di = np.flatnonzero(~inc_up)
        seg_of_cell = np.full(ncell, -1, dtype=np.int64)
        seg_of_cell[sc[sfirst]] = np.arange(top.shape[0])
        ds = seg_of_cell[inc_cell[di]]
        dmask = ds >= 0
        highest_bottom = _segment_reduce(ds[dmask], inc_zhi[di][dmask], top.shape[0], np.maximum, -np.inf)
        ok = (np.abs(top_area - cell_area) <= area_tol) & all_below & (highest_bottom < lowest_buried - _PLANE_TOL)
        buried[so[~on_top & ok[sseg]]] = True
        stats["stacked_cells"] = {"cells": int(top.shape[0]), "one_top_plane_over_buried": int(ok.sum())}
    stats["buried_pieces"] = int(buried.sum())

    # --- per cell: the up-facing planes ---------------------------------------------------------------------------
    upi = np.flatnonzero(inc_up & ~buried)
    order = upi[np.lexsort((-inc_area[upi], inc_cell[upi]))]
    oc = inc_cell[order]
    first = np.r_[True, oc[1:] != oc[:-1]] if oc.size else np.zeros(0, bool)
    head = np.flatnonzero(first)
    seg = np.cumsum(first) - 1
    plane_a = inc_corner[order[head]][seg]
    in_a = np.max(np.abs(inc_corner[order] - plane_a), axis=1) <= _PLANE_TOL
    # plane B: the largest piece not on plane A
    cand = np.where(~in_a, np.arange(order.size), order.size)
    first_b = np.full(head.size, order.size, dtype=np.int64)
    np.minimum.at(first_b, seg, cand)
    has_b = first_b < order.size
    plane_b = np.where(has_b[:, None], inc_corner[order[np.minimum(first_b, order.size - 1)]], np.nan)[seg]
    in_b = ~in_a & (np.max(np.abs(inc_corner[order] - plane_b), axis=1) <= _PLANE_TOL)
    third = ~(in_a | in_b)

    cells = oc[head]
    nseg = head.size
    area_up = np.zeros(nseg)
    np.add.at(area_up, seg, inc_area[order])
    area_a = np.zeros(nseg)
    np.add.at(area_a, seg[in_a], inc_area[order][in_a])
    n_third = np.bincount(seg[third], minlength=nseg)
    ext = inc_ext[order]

    def group_ext(mask, col, op, fill):
        return _segment_reduce(seg[mask], ext[mask, col], nseg, op, fill)

    ea = [
        group_ext(in_a, k, np.minimum if k % 2 == 0 else np.maximum, np.inf if k % 2 == 0 else -np.inf)
        for k in range(8)
    ]
    eb = [
        group_ext(in_b, k, np.minimum if k % 2 == 0 else np.maximum, np.inf if k % 2 == 0 else -np.inf)
        for k in range(8)
    ]
    za = plane_a[head]
    zb = plane_b[head]
    flat_a = (za.max(1) - za.min(1)) <= _PLANE_TOL
    flat_b = (zb.max(1) - zb.min(1)) <= _PLANE_TOL
    tu = tol / min(dx, dy)

    covered = np.abs(area_up - cell_area) <= area_tol
    one_plane = covered & ~has_b
    two = covered & has_b & (n_third == 0)

    def near(x, y):
        return np.abs(x - y) <= tu

    # two half-cell planes on diagonal (0,0)-(1,1): one group in u >= v, the other in u <= v, continuous at 00 and 11
    a_lr = ea[4] >= -tu
    b_ul = eb[5] <= tu
    a_ul = ea[5] <= tu
    b_lr = eb[4] >= -tu
    cont0 = (np.abs(za[:, 0] - zb[:, 0]) <= _PLANE_TOL) & (np.abs(za[:, 3] - zb[:, 3]) <= _PLANE_TOL)
    diag0 = two & cont0 & ((a_lr & b_ul) | (a_ul & b_lr))
    # diagonal (1,0)-(0,1): one group in u + v <= 1, the other in u + v >= 1, continuous at 10 and 01
    a_ll = ea[7] <= 1.0 + tu
    b_ur = eb[6] >= 1.0 - tu
    a_ur = ea[6] >= 1.0 - tu
    b_ll = eb[7] <= 1.0 + tu
    cont1 = (np.abs(za[:, 1] - zb[:, 1]) <= _PLANE_TOL) & (np.abs(za[:, 2] - zb[:, 2]) <= _PLANE_TOL)
    diag1 = two & ~diag0 & cont1 & ((a_ll & b_ur) | (a_ur & b_ll))
    # step: two horizontal levels split by a line u = s or v = s
    levels = two & ~diag0 & ~diag1 & flat_a & flat_b
    sx_ab = levels & near(ea[0], 0.0) & near(ea[1], eb[0]) & near(eb[1], 1.0)
    sx_ba = levels & near(eb[0], 0.0) & near(eb[1], ea[0]) & near(ea[1], 1.0)
    sy_ab = levels & ~(sx_ab | sx_ba) & near(ea[2], 0.0) & near(ea[3], eb[2]) & near(eb[3], 1.0)
    sy_ba = levels & ~(sx_ab | sx_ba | sy_ab) & near(eb[2], 0.0) & near(eb[3], ea[2]) & near(ea[3], 1.0)
    s_val = np.select([sx_ab, sx_ba, sy_ab, sy_ba], [ea[1], eb[1], ea[3], eb[3]], default=0.0)
    is_step = (sx_ab | sx_ba | sy_ab | sy_ba) & (s_val > tu) & (s_val < 1.0 - tu)

    kind = np.full(ncell, TERRAIN_CELL_EMPTY, dtype=np.int64)
    flags = np.zeros(ncell, dtype=np.uint64)
    corners = np.zeros((ncell, 4), dtype=np.float64)
    seg_kind = np.full(nseg, TERRAIN_CELL_MIXED, dtype=np.int64)
    seg_kind[one_plane | diag0 | diag1] = TERRAIN_CELL_GRID
    seg_kind[is_step] = TERRAIN_CELL_STEP
    seg_corner = za.copy()
    lr_is_a = a_lr & b_ul
    seg_corner[diag0, 1] = np.where(lr_is_a, za[:, 1], zb[:, 1])[diag0]
    seg_corner[diag0, 2] = np.where(lr_is_a, zb[:, 2], za[:, 2])[diag0]
    ll_is_a = a_ll & b_ur
    seg_corner[diag1, 0] = np.where(ll_is_a, za[:, 0], zb[:, 0])[diag1]
    seg_corner[diag1, 3] = np.where(ll_is_a, zb[:, 3], za[:, 3])[diag1]
    # step corners: each side's level (z00 holds the side below s)
    low_a = sx_ab | sy_ab
    lev_lo = np.where(low_a, za[:, 0], zb[:, 0])
    lev_hi = np.where(low_a, zb[:, 0], za[:, 0])
    on_x = sx_ab | sx_ba
    st = is_step
    seg_corner[st, 0] = lev_lo[st]
    seg_corner[st, 1] = np.where(on_x, lev_hi, lev_lo)[st]
    seg_corner[st, 2] = np.where(on_x, lev_lo, lev_hi)[st]
    seg_corner[st, 3] = lev_hi[st]
    seg_flags = np.zeros(nseg, dtype=np.uint64)
    seg_flags[diag1] |= TERRAIN_CELL_DIAGONAL
    seg_flags[st & ~on_x] |= TERRAIN_CELL_STEP_Y
    payload = np.clip(np.round(s_val * TERRAIN_CELL_STEP_ONE), 1, TERRAIN_CELL_STEP_ONE - 1).astype(np.uint64)
    seg_flags[st] |= payload[st] << np.uint64(TERRAIN_CELL_PAYLOAD_SHIFT)
    kind[cells] = seg_kind
    flags[cells] = seg_flags
    corners[cells] = seg_corner

    reasons = {
        "partial_or_stacked_cover": int((~covered).sum()),
        "three_or_more_planes": int((covered & has_b & (n_third > 0)).sum()),
        "two_planes_not_halves_or_step": int((two & ~diag0 & ~diag1 & ~is_step).sum()),
    }

    # vertices on corners: copy their float32 heights where they lie on the cell's surface plane
    snap_cell = np.concatenate(snap_cell) if snap_cell else np.zeros(0, np.int64)
    snap_corner = np.concatenate(snap_corner) if snap_corner else np.zeros(0, np.int64)
    snap_z = np.concatenate(snap_z).astype(np.float64) if snap_z else np.zeros(0)
    law = (kind == TERRAIN_CELL_GRID) | (kind == TERRAIN_CELL_STEP)
    ok = law[snap_cell] & (np.abs(corners[snap_cell, snap_corner] - snap_z) <= _PLANE_TOL)
    corners[snap_cell[ok], snap_corner[ok]] = snap_z[ok]
    stats["corners_from_vertices"] = int(np.unique(snap_cell[ok] * 4 + snap_corner[ok]).size)

    mixed_extra: list[tuple[np.ndarray, np.ndarray]] = []  # (cell, triangle) entries beyond the face incidences

    # --- down-facing pieces: below the lowest corner of a law cell (a solid's bottom), else the cell is mixed -------
    dn = np.flatnonzero(~inc_up)
    dcell = inc_cell[dn]
    lowest = corners[dcell].min(axis=1)
    bad = law[dcell] & (inc_zhi[dn] > lowest - _PLANE_TOL)
    thick = (lowest - inc_zhi[dn])[law[dcell] & ~bad]
    stats["bottoms"] = {
        "pieces": int(dn.size),
        "above_surface": int(bad.sum()),
        "min_thickness": float(thick.min()) if thick.size else None,
    }
    over = np.unique(dcell[bad])
    reasons["bottom_within_or_above_surface"] = int(law[over].sum())
    kind[over] = TERRAIN_CELL_MIXED
    law = (kind == TERRAIN_CELL_GRID) | (kind == TERRAIN_CELL_STEP)

    # --- walls ----------------------------------------------------------------------------------------------------
    widx = np.flatnonzero(wall)
    wx = np.stack([a[widx, 0], b[widx, 0], c[widx, 0]], 1)
    wy = np.stack([a[widx, 1], b[widx, 1], c[widx, 1]], 1)
    wz = tri_z[widx]
    along_y = (wx.max(1) - wx.min(1)) <= tol  # the line x = const
    along_x = ~along_y & ((wy.max(1) - wy.min(1)) <= tol)  # the line y = const
    general = ~(along_y | along_x)
    stats["walls_general"] = int(general.sum())
    # a general (not axis-aligned) wall makes every cell its footprint box overlaps mixed
    if general.any():
        gi = np.flatnonzero(general)
        gc0 = np.clip(np.floor((wx[gi].min(1) - x0) / dx + tu).astype(np.int64), 0, cols - 1)
        gc1 = np.clip(np.ceil((wx[gi].max(1) - x0) / dx - tu).astype(np.int64) - 1, 0, cols - 1)
        gr0 = np.clip(np.floor((wy[gi].min(1) - y0) / dy + tu).astype(np.int64), 0, rows - 1)
        gr1 = np.clip(np.ceil((wy[gi].max(1) - y0) / dy - tu).astype(np.int64) - 1, 0, rows - 1)
        for k in range(gi.size):
            rr, cc = np.meshgrid(np.arange(gr0[k], max(gr1[k], gr0[k]) + 1), np.arange(gc0[k], max(gc1[k], gc0[k]) + 1))
            cl = (rr * cols + cc).ravel()
            kind[cl] = TERRAIN_CELL_MIXED
            mixed_extra.append((cl, np.full(cl.size, widx[gi[k]], dtype=np.int64)))
    reasons["general_wall"] = int(general.sum())

    wall_checks = {"interior_ok": 0, "interior_bad": 0, "edge_ok": 0, "edge_bad": 0, "edge_beside_mixed": 0}
    pieces = [
        _wall_pieces(wx, wy, wz, widx, mask, line_axis, x0, y0, dx, dy, rows, cols, tol, tu)
        for mask, line_axis in ((along_y, 0), (along_x, 1))
    ]
    # interior walls first (they can only demote cells): the cell must be a step whose riser is this line, and the
    # wall must not reach above its high level
    for line_axis, p in enumerate(pieces):
        if p is None:
            continue
        ii = ~p["edge"]
        ci = np.floor(p["line"][ii]).astype(np.int64)
        s_in = p["line"][ii] - ci
        cell_i = p["run"][ii] * cols + ci if line_axis == 0 else ci * cols + p["run"][ii]
        fl = flags[cell_i]
        axis_ok = (kind[cell_i] == TERRAIN_CELL_STEP) & (((fl & TERRAIN_CELL_STEP_Y) != 0) == (line_axis == 1))
        s_cell = (fl >> np.uint64(TERRAIN_CELL_PAYLOAD_SHIFT)).astype(np.float64) / TERRAIN_CELL_STEP_ONE
        high = corners[cell_i].max(axis=1)
        riser = axis_ok & (np.abs(s_cell - s_in) <= tu) & (p["top"][ii] <= high + _PLANE_TOL)
        lawi = (kind[cell_i] == TERRAIN_CELL_GRID) | (kind[cell_i] == TERRAIN_CELL_STEP)
        inside = _line_min(corners[cell_i], flags[cell_i], kind[cell_i], line_axis, s_in, p["t0"][ii], p["t1"][ii])
        good = riser | (
            lawi & ~riser & (p["top"][ii] <= inside + _PLANE_TOL) & ~(axis_ok & (np.abs(s_cell - s_in) <= tu))
        )
        wall_checks["interior_buried"] = wall_checks.get("interior_buried", 0) + int((good & ~riser).sum())
        wall_checks["interior_ok"] += int(good.sum())
        wall_checks["interior_bad"] += int((~good).sum())
        kind[np.unique(cell_i[~good])] = TERRAIN_CELL_MIXED
        mixed_extra.append((cell_i, p["tri"][ii]))
    reasons["wall_inside_cell_off_riser"] = int(wall_checks["interior_bad"])
    # edge walls: between two law cells (or a law cell and the table's border) the wall must not reach above the
    # higher side's lowest surface along its run; beside a mixed cell it is that cell's triangle
    law = (kind == TERRAIN_CELL_GRID) | (kind == TERRAIN_CELL_STEP)
    for line_axis, p in enumerate(pieces):
        if p is None:
            continue
        ee = p["edge"]
        le = np.round(p["line"][ee]).astype(np.int64)
        run = p["run"][ee]
        t0, t1 = p["t0"][ee], p["t1"][ee]
        floor_z = np.full(le.size, -np.inf)
        beside_mixed = np.zeros(le.size, dtype=bool)
        sides = []
        for col_off, high in ((-1, True), (0, False)):
            ce = le + col_off
            exists = (ce >= 0) & (ce < (cols if line_axis == 0 else rows))
            ce = np.clip(ce, 0, (cols if line_axis == 0 else rows) - 1)
            cell_e = run * cols + ce if line_axis == 0 else ce * cols + run
            low = _edge_min(corners[cell_e], flags[cell_e], kind[cell_e], line_axis, high, t0, t1)
            floor_z = np.maximum(floor_z, np.where(exists & law[cell_e], low, -np.inf))
            beside_mixed |= exists & ~law[cell_e]
            sides.append((cell_e, exists))
        good = beside_mixed | (p["top"][ee] <= floor_z + _PLANE_TOL)
        wall_checks["edge_ok"] += int((good & ~beside_mixed).sum())
        wall_checks["edge_beside_mixed"] += int(beside_mixed.sum())
        wall_checks["edge_bad"] += int((~good).sum())
        for cell_e, exists in sides:
            kind[np.unique(cell_e[~good & exists])] = TERRAIN_CELL_MIXED
            mixed_extra.append((cell_e[exists], p["tri"][ee][exists]))
    reasons["wall_on_edge_above_both_sides"] = int(wall_checks["edge_bad"])
    stats["wall_checks"] = wall_checks
    law = (kind == TERRAIN_CELL_GRID) | (kind == TERRAIN_CELL_STEP)
    stats["mixed_reasons"] = reasons

    # --- mixed cells: CSR of the triangles touching them, z-range of those triangles clipped to the cell ---------
    mixed = np.flatnonzero(kind == TERRAIN_CELL_MIXED)
    mixed_index = np.full(ncell, -1, dtype=np.int64)
    mixed_index[mixed] = np.arange(mixed.size)
    ent_cell = [inc_cell]
    ent_tri = [inc_face]
    ent_lo = [inc_zlo]
    ent_hi = [inc_zhi]
    for cl, tr in mixed_extra:
        ent_cell.append(cl)
        ent_tri.append(tr)
        ent_lo.append(tri_z[tr].min(1))
        ent_hi.append(tri_z[tr].max(1))
    ent_cell = np.concatenate(ent_cell)
    ent_tri = np.concatenate(ent_tri)
    ent_lo = np.concatenate(ent_lo)
    ent_hi = np.concatenate(ent_hi)
    sel = mixed_index[ent_cell] >= 0
    mi = mixed_index[ent_cell[sel]]
    key = np.unique(mi * (f.shape[0] + 1) + ent_tri[sel])
    m_of = key // (f.shape[0] + 1)
    t_of = key % (f.shape[0] + 1)
    mixed_start = np.zeros(mixed.size + 1, dtype=np.int64)
    np.add.at(mixed_start, m_of + 1, 1)
    mixed_start = np.cumsum(mixed_start)
    mz_lo = np.full(mixed.size, np.inf)
    mz_hi = np.full(mixed.size, -np.inf)
    np.minimum.at(mz_lo, mi, ent_lo[sel])
    np.maximum.at(mz_hi, mi, ent_hi[sel])
    corners[mixed] = mz_hi[:, None]
    flags[mixed] = mixed_index[mixed].astype(np.uint64) << np.uint64(TERRAIN_CELL_PAYLOAD_SHIFT)
    flags[kind == TERRAIN_CELL_EMPTY] = 0
    flags[kind == TERRAIN_CELL_GRID] &= np.uint64(TERRAIN_CELL_DIAGONAL)
    flags |= kind.astype(np.uint64)

    cell_z = corners.astype(np.float32)
    table = TerrainMeshCells(
        origin=(float(x0), float(y0)),
        cell_size=(dx, dy),
        shape=(rows, cols),
        cell_z=cell_z,
        cell_flags=flags.astype(np.uint32),
        mixed_cell=mixed.astype(np.int32),
        mixed_start=mixed_start.astype(np.int32),
        mixed_triangles=t_of.astype(np.int32),
        mixed_z=np.stack([mz_lo, mz_hi], 1).astype(np.float32),
        stats=stats,
    )
    _write_edge_bits(table)
    return table


def _wall_pieces(wx, wy, wz, widx, mask, line_axis, x0, y0, dx, dy, rows, cols, tol, tu):
    """Axis-aligned walls cut into the rows (or columns) of cells they run through.

    ``line_axis`` 0: walls on a line ``x = X`` running along y; 1: on ``y = Y`` running along x. Returns per piece the
    wall's triangle id, its line in cell units (``line``; an integer on a lattice line), the run index of the cell row
    it lies in (``run``), whether the line is a lattice line (``edge``), the run range ``[t0, t1]`` in cell units and
    the wall's top over that range (the triangle clipped to the run, in the run-height plane).
    """
    wi = np.flatnonzero(mask)
    if wi.size == 0:
        return None
    line = (wx if line_axis == 0 else wy)[wi].mean(1)
    run = (wy if line_axis == 0 else wx)[wi]
    size_l, size_r = (dx, dy) if line_axis == 0 else (dy, dx)
    o_l, o_r = (x0, y0) if line_axis == 0 else (y0, x0)
    n_r = rows if line_axis == 0 else cols
    ql = (line - o_l) / size_l
    on_edge = np.abs(ql - np.round(ql)) * size_l <= tol
    k0 = np.clip(np.floor((run.min(1) - o_r) / size_r + tu).astype(np.int64), 0, n_r - 1)
    k1 = np.clip(np.ceil((run.max(1) - o_r) / size_r - tu).astype(np.int64) - 1, 0, n_r - 1)
    k1 = np.maximum(k1, k0)
    cnt = k1 - k0 + 1
    rep = np.repeat(np.arange(wi.size), cnt)
    kk = k0[rep] + (np.arange(rep.size) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    px = np.zeros((rep.size, _MAX_POLY))
    pz = np.zeros((rep.size, _MAX_POLY))
    px[:, :3] = run[rep]
    pz[:, :3] = wz[wi][rep]
    nn = np.full(rep.size, 3, dtype=np.int64)
    lo_r = o_r + kk * size_r
    big = np.full(rep.size, 1.0e30)
    px, pz, nn = _clip_box(px, pz, nn, lo_r, lo_r + size_r, -big, big)
    t0 = (_masked(px, nn, np.inf).min(1) - lo_r) / size_r
    t1 = (_masked(px, nn, -np.inf).max(1) - lo_r) / size_r
    alive = (nn >= 2) & ((t1 - t0) > tu)
    top = _masked(pz, nn, -np.inf).max(1)
    return {
        "tri": widx[wi[rep]][alive],
        "line": ql[rep][alive],
        "run": kk[alive],
        "edge": on_edge[rep][alive],
        "t0": t0[alive],
        "t1": t1[alive],
        "top": top[alive],
    }


def _surface64(z: np.ndarray, flags: np.ndarray, kind: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Float64 surface of law cells at cell-local ``(u, v)`` in cell units (a riser line takes the higher side)."""
    z00, z10, z01, z11 = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    g0 = np.where(u >= v, z00 + (z10 - z00) * u + (z11 - z10) * v, z00 + (z11 - z01) * u + (z01 - z00) * v)
    g1 = np.where(
        u + v <= 1.0, z00 + (z10 - z00) * u + (z01 - z00) * v, z11 + (z11 - z01) * (u - 1.0) + (z11 - z10) * (v - 1.0)
    )
    grid = np.where((flags & TERRAIN_CELL_DIAGONAL) != 0, g1, g0)
    s = (flags >> np.uint64(TERRAIN_CELL_PAYLOAD_SHIFT)).astype(np.float64) / TERRAIN_CELL_STEP_ONE
    ry = (flags & TERRAIN_CELL_STEP_Y) != 0
    along = np.where(ry, v, u)
    far = np.where(ry, z01, z10)
    step = np.where(along < s, z00, np.where(along > s, far, np.maximum(z00, far)))
    return np.where(kind == TERRAIN_CELL_STEP, step, grid)


def _line_min(z, flags, kind, line_axis: int, s_line, t0, t1) -> np.ndarray:
    """Lowest surface of a law cell along the line ``u = s_line`` (``line_axis`` 0) or ``v = s_line`` over the run
    ``[t0, t1]``: the surface is linear between the run's ends, the diagonal and a riser, so those points bound it."""
    s = (flags >> np.uint64(TERRAIN_CELL_PAYLOAD_SHIFT)).astype(np.float64) / TERRAIN_CELL_STEP_ONE
    diag = np.where((flags & TERRAIN_CELL_DIAGONAL) != 0, 1.0 - s_line, s_line)
    eps = 1.0e-9
    samples = [t0, t1, np.clip(diag, t0, t1), np.clip(s - eps, t0, t1), np.clip(s + eps, t0, t1)]
    out = np.full(t0.shape, np.inf)
    for t in samples:
        u, v = (s_line, t) if line_axis == 0 else (t, s_line)
        out = np.minimum(out, _surface64(z, flags, kind, np.broadcast_to(u, t.shape), np.broadcast_to(v, t.shape)))
    return out


def _edge_profile(z: np.ndarray, flags: np.ndarray, kind: np.ndarray, line_axis: int, high: bool):
    """A cell's surface along one of its edges: values at the edge's two ends and a breakpoint (a step crossing it).

    ``line_axis`` 0 is an edge ``x = const`` (running along y), 1 an edge ``y = const``; ``high`` the edge at ``x = dx``
    (or ``y = dy``). Returns ``(z_start, z_end, s)`` with ``s`` NaN where the profile is linear, else the run
    parameter where it jumps from ``z_start`` to ``z_end``.
    """
    if line_axis == 0:
        zs = np.where(high, z[:, 1], z[:, 0])
        ze = np.where(high, z[:, 3], z[:, 2])
    else:
        zs = np.where(high, z[:, 2], z[:, 0])
        ze = np.where(high, z[:, 3], z[:, 1])
    step = kind == TERRAIN_CELL_STEP
    riser_y = (flags & TERRAIN_CELL_STEP_Y) != 0
    # the step crosses edges parallel to its riser's normal: a riser x = s crosses the edges y = const
    crosses = step & (riser_y == (line_axis == 0))
    s = (flags >> np.uint64(TERRAIN_CELL_PAYLOAD_SHIFT)).astype(np.float64) / TERRAIN_CELL_STEP_ONE
    return zs, ze, np.where(crosses, s, np.nan)


def _edge_min(z, flags, kind, line_axis: int, high: bool, t0, t1):
    """Lowest surface height along a cell's edge over the run ``[t0, t1]`` (cell units)."""
    zs, ze, s = _edge_profile(z, flags, kind, line_axis, high)
    lin0 = zs + (ze - zs) * t0
    lin1 = zs + (ze - zs) * t1
    linear = np.isnan(s)
    # stepped: the side below s holds zs, beyond holds ze
    has_lo = t0 < np.where(linear, 0.0, s)
    has_hi = t1 > np.where(linear, 1.0, s)
    stepped = np.minimum(np.where(has_lo, zs, np.inf), np.where(has_hi, ze, np.inf))
    return np.where(linear, np.minimum(lin0, lin1), stepped)


def _write_edge_bits(table: TerrainMeshCells) -> None:
    """Mark each law cell's edges whose neighbour's surface differs along the edge or is not a law cell."""
    rows, cols = table.shape
    kind = table.kind.reshape(rows, cols)
    flags = table.cell_flags.reshape(rows, cols).astype(np.uint64)
    z = table.cell_z.astype(np.float64).reshape(rows, cols, 4)
    law = (kind == TERRAIN_CELL_GRID) | (kind == TERRAIN_CELL_STEP)
    bits = np.zeros((rows, cols), dtype=np.uint64)
    for line_axis, bit_lo, bit_hi in (
        (0, TERRAIN_CELL_EDGE_X0, TERRAIN_CELL_EDGE_X1),
        (1, TERRAIN_CELL_EDGE_Y0, TERRAIN_CELL_EDGE_Y1),
    ):
        # pairs (left/lower cell A, right/upper cell B) across every interior lattice line of this axis
        if line_axis == 0:
            za, fa, ka, la = z[:, :-1].reshape(-1, 4), flags[:, :-1].ravel(), kind[:, :-1].ravel(), law[:, :-1].ravel()
            zb, fb, kb, lb = z[:, 1:].reshape(-1, 4), flags[:, 1:].ravel(), kind[:, 1:].ravel(), law[:, 1:].ravel()
        else:
            za, fa, ka, la = z[:-1].reshape(-1, 4), flags[:-1].ravel(), kind[:-1].ravel(), law[:-1].ravel()
            zb, fb, kb, lb = z[1:].reshape(-1, 4), flags[1:].ravel(), kind[1:].ravel(), law[1:].ravel()
        sa0, sa1, ssa = _edge_profile(za, fa, ka, line_axis, True)
        sb0, sb1, ssb = _edge_profile(zb, fb, kb, line_axis, False)
        same_shape = (np.isnan(ssa) & np.isnan(ssb)) | (
            np.abs(np.nan_to_num(ssa, nan=-9.0) - np.nan_to_num(ssb, nan=-9.0)) <= 1.0 / TERRAIN_CELL_STEP_ONE
        )
        same = (
            same_shape & (np.abs(sa0 - sb0) <= TERRAIN_CELL_RISER_EPS) & (np.abs(sa1 - sb1) <= TERRAIN_CELL_RISER_EPS)
        )
        riser = ~(same & la & lb)
        if line_axis == 0:
            r = riser.reshape(rows, cols - 1)
            bits[:, :-1] |= np.where(r, np.uint64(bit_hi), np.uint64(0))
            bits[:, 1:] |= np.where(r, np.uint64(bit_lo), np.uint64(0))
        else:
            r = riser.reshape(rows - 1, cols)
            bits[:-1, :] |= np.where(r, np.uint64(bit_hi), np.uint64(0))
            bits[1:, :] |= np.where(r, np.uint64(bit_lo), np.uint64(0))
    # the table's outer border is a riser of its own (the block's walls)
    bits[:, 0] |= np.uint64(TERRAIN_CELL_EDGE_X0)
    bits[:, -1] |= np.uint64(TERRAIN_CELL_EDGE_X1)
    bits[0, :] |= np.uint64(TERRAIN_CELL_EDGE_Y0)
    bits[-1, :] |= np.uint64(TERRAIN_CELL_EDGE_Y1)
    bits[~law] = 0
    table.cell_flags = (flags | bits).astype(np.uint32).ravel()


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

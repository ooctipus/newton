# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Float64 reference of the terrain contact law, the oracle of its float32 kernels (numpy only).

The law emits the support set of one (heightfield, convex shape) pair directly, one row per local minimum of the
core-to-surface distance within the writer's reach plus the corners of flat support regions, each row owned by one
surface feature:

* P: point and segment cores (sphere centre, capsule ends and axis interior): face, crease-edge and vertex minima
  under one shared set of predicate values; G2 keeps one exit for a core point inside the solid, G3 the face
  vertically above it when no exit lies in the window.
* F: solid cores against plane clusters: the analytic plane manifold of the core against the cluster plane,
  emitted once per cluster by the triangle holding the first supported point.
* E: solid cores against convex creases: the maximiser of the separation over the crease's normal arc, then the
  core's flat support feature (interior) or its reference feature (arc end, G1) crossed with the edge.
* V: solid cores against vertex features (vertices whose incident cluster normals span 3-D).
* R: solid cores below the surface with no row: up to four of their deepest boundary points (G3).
* C: optional concave anticipation of a moving core.
* The finish: coincident rows merged, rows implied by the others of their plane group dropped (the lower convex
  hull of tangent position and distance), at most four rows per plane group.

This is a port of the design study's float64 prototype with four changes, each measured on the CPU:

* Plane clusters use robust planes. A cluster is grown over the flat edges as before, but a component whose
  vertices all lie within ``FLAT_EPS`` of the plane through three extreme vertices, or of a horizontal plane,
  each with its offset centred on the residual range, is one cluster with that plane; other components keep the
  candidate (those two or the plane of their lowest-id triangle) that claims the most triangles. The plane of a
  single 0.1 m triangle carries the float32 height decode noise (a tilt up to about 2e-6 rad), so far members of
  a large cluster left its 1 um band: a 12.7 m planar 12 deg down-slope split into 4,072 clusters, and cropping
  the recorded suite terrain by three rows and seven columns split its flat border into over a million clusters
  (1,562,181 in all, against 566,737 uncropped); with robust planes both crops have 539,984 +- 2.
* Edge classes follow the cluster ids alone: two clusters always meet at a crease.
* A vertex is a feature when three of its incident cluster normals have a triple product above
  ``VERTEX_SPAN = 1e-6`` (the prototype: 1e-10), the resolution of that product in float32.
* Rows of a plane group are implied when they lie on or above the lower convex hull of (tangent position,
  distance) of the others, within ``IMPLIED_RESIDUAL``; the prototype also required them to lie inside the 3-D
  hull of the others' core points.

Heights are decoded as the narrow phase decodes them (float32 ``min_z + e * (max_z - min_z)``), then promoted to
float64. Vertex (r, c) lies at x = -hx + c * dx, y = -hy + r * dy; cell (r, c) splits along p00-p11 into
sub 0 = (p00, p10, p11) and sub 1 = (p00, p11, p01); triangle id = 2 * (r * cols + c) + sub.
"""

from __future__ import annotations

import math

import numpy as np

FLAT_EPS = 1.0e-6
"""Plane-cluster membership tolerance [m] (absolute; it absorbs the ~0.24 um float32 height decode)."""
CLUSTER_ROUNDS = 8
"""Rounds of the cluster build; triangles still unassigned afterwards are clusters of their own."""
NORMAL_SAME = 1.0e-5
"""Unit normals closer than this are one direction (vertex features)."""
VERTEX_SPAN = 1.0e-6
"""A vertex is a feature when some three incident cluster normals have a triple product above this."""
IMPLIED_RESIDUAL = 1.0e-6
"""A row no deeper than this below the lower hull of its plane group's other rows is implied [m]."""

EPS_PRED = 1.0e-12
"""Predicate band [m]: a value within it counts as on the boundary."""
EPS_MERGE_P = 1.0e-7
EPS_MERGE_N = 1.0e-6
EPS_THETA = 1.0e-7
CYLINDER_FLAT_MODE_COS = math.cos(math.radians(22.5))

RING = ((0, 1), (1, 1), (1, 0), (0, -1), (-1, -1), (-1, 0))
"""One-ring neighbour offsets (dr, dc) of a vertex, counter-clockwise."""


# ----------------------------------------------------------------------------------------------------------------
# heights and clusters
# ----------------------------------------------------------------------------------------------------------------


def decode_heights(elevations, nrow: int, ncol: int, min_z: float, max_z: float) -> np.ndarray:
    """Decode normalized elevations as the narrow phase does (float32), promoted to float64, shape (nrow, ncol)."""
    e = np.asarray(elevations, dtype=np.float32).reshape(nrow, ncol)
    lo = np.float32(min_z)
    span = np.float32(np.float32(max_z) - lo)
    return (lo + e * span).astype(np.float32).astype(np.float64)


def connected_roots(count: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Union-find over ``count`` nodes and the edges (a, b): the root of every node is its component's minimum."""
    parent = np.arange(count, dtype=np.int64)
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    while True:
        ra, rb = parent[a], parent[b]
        lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)
        merge = lo != hi
        if not np.any(merge):
            return parent
        np.minimum.at(parent, hi[merge], lo[merge])
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped


def _triangle_vertex_ids(rows: int, cols: int, ncol: int) -> np.ndarray:
    """Vertex ids (r * ncol + c) of every triangle, shape (2 * rows * cols, 3), in Newton order."""
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    v00 = r * ncol + c
    v10 = v00 + 1
    v01 = v00 + ncol
    v11 = v01 + 1
    out = np.empty((rows, cols, 2, 3), np.int64)
    out[:, :, 0] = np.stack([v00, v10, v11], -1)
    out[:, :, 1] = np.stack([v00, v11, v01], -1)
    return out.reshape(-1, 3)


def _plane_gradient(dc: np.ndarray, dr: np.ndarray, dz: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Gradients (gx, gy) of planes through the origin and two points (dc[i] dx, dr[i] dy, dz[i]), i = 0, 1."""
    a00, a01 = dc[..., 0] * dx, dr[..., 0] * dy
    a10, a11 = dc[..., 1] * dx, dr[..., 1] * dy
    det = a00 * a11 - a01 * a10
    gx = (dz[..., 0] * a11 - a01 * dz[..., 1]) / det
    gy = (a00 * dz[..., 1] - dz[..., 0] * a10) / det
    return np.stack([gx, gy], -1)


def _normal_scale(gradient: np.ndarray) -> np.ndarray:
    """Normal z-component 1 / |(-gx, -gy, 1)|: the factor from vertical residual to normal distance."""
    return 1.0 / np.sqrt(1.0 + gradient[..., 0] ** 2 + gradient[..., 1] ** 2)


def build_reference_clusters(Z: np.ndarray, dx: float, dy: float, *, flat_eps: float = FLAT_EPS):
    """Plane clusters of a heightfield (float64 reference of the Warp build).

    Round by round, the triangles not yet assigned form components over the flat edges (an edge is flat when
    the far vertex of one side lies within ``flat_eps`` of the other side's plane). Each component has three
    candidate planes: the extreme plane (through its lowest vertex ``a``, the vertex ``b`` farthest from ``a`` and
    the vertex ``c`` spanning the largest triangle with them, in grid units) and the level plane (horizontal),
    each with its offset centred on the component's residual range, and the seed plane (its lowest-id
    triangle's). A component whose vertices all lie within ``flat_eps`` of its extreme or level plane is one
    cluster with the closer of the two. Otherwise the candidate claiming the most triangles (all three vertices
    within ``flat_eps``) claims them, the seed plane first and the extreme plane second on ties, and the rest
    wait for the next round.

    Plane gradients and residuals are rounded to float32, as the Warp build stores and compares them.

    Returns:
        ``(root, records)``: every triangle's cluster root (the cluster's lowest triangle id), and for every root
        of a multi-triangle cluster its plane ``(anchor vertex id, gx, gy, offset)``.
    """
    nrow, ncol = Z.shape
    rows, cols = nrow - 1, ncol - 1
    count = 2 * rows * cols
    zf = Z.reshape(-1)
    tv = _triangle_vertex_ids(rows, cols, ncol)
    vr, vc = tv // ncol, tv % ncol
    # unit normals of every triangle (relative coordinates)
    e1 = np.stack([(vc[:, 1] - vc[:, 0]) * dx, (vr[:, 1] - vr[:, 0]) * dy, zf[tv[:, 1]] - zf[tv[:, 0]]], -1)
    e2 = np.stack([(vc[:, 2] - vc[:, 0]) * dx, (vr[:, 2] - vr[:, 0]) * dy, zf[tv[:, 2]] - zf[tv[:, 0]]], -1)
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n *= np.sign(n[:, 2:3])
    # flat edges: (triangle a, triangle b, far vertex of b, plane point of a = a's p00)
    tri = np.arange(count).reshape(rows, cols, 2)
    pa, pb, far = [], [], []
    pa.append(tri[:, :, 0].ravel())
    pb.append(tri[:, :, 1].ravel())
    far.append((np.arange(rows)[:, None] + 1) * ncol + np.arange(cols)[None, :])  # D: p01 of the cell
    pa.append(tri[1:, :, 0].ravel())
    pb.append(tri[:-1, :, 1].ravel())
    far.append((np.arange(rows - 1)[:, None]) * ncol + np.arange(cols)[None, :])  # H: p00 of the cell below
    pa.append(tri[:, 1:, 1].ravel())
    pb.append(tri[:, :-1, 0].ravel())
    far.append(np.arange(rows)[:, None] * ncol + np.arange(cols - 1)[None, :])  # V: p00 of the cell left
    pa = np.concatenate(pa)
    pb = np.concatenate(pb)
    far = np.concatenate([f.ravel() for f in far])
    anchor = tv[pa, 0]
    w = np.stack(
        [(far % ncol - anchor % ncol) * dx, (far // ncol - anchor // ncol) * dy, zf[far] - zf[anchor]],
        -1,
    )
    deviation = np.abs(np.einsum("ij,ij->i", n[pa], w)).astype(np.float32).astype(np.float64)
    flat = deviation <= flat_eps
    ea, eb = pa[flat], pb[flat]

    root_of = np.full(count, -1, np.int64)
    records = {}
    active = np.ones(count, bool)
    for _round in range(CLUSTER_ROUNDS):
        if not np.any(active):
            break
        keep = active[ea] & active[eb]
        parent = connected_roots(count, ea[keep], eb[keep])
        idx = np.flatnonzero(active)
        root = parent[idx]
        verts = tv[idx]  # (m, 3)
        vroot = np.repeat(root, 3)
        vid = verts.reshape(-1)
        big = np.iinfo(np.int64).max
        # extreme vertices: a = lowest id, b = farthest from a (grid units), c = largest area with (a, b)
        va = np.full(count, big, np.int64)
        np.minimum.at(va, vroot, vid)
        ar, ac = va[vroot] // ncol, va[vroot] % ncol
        d2 = (vid // ncol - ar) ** 2 + (vid % ncol - ac) ** 2
        best = np.full(count, -1, np.int64)
        np.maximum.at(best, vroot, d2)
        vb = np.full(count, big, np.int64)
        hit = d2 == best[vroot]
        np.minimum.at(vb, vroot[hit], vid[hit])
        br, bc = vb[vroot] // ncol - ar, vb[vroot] % ncol - ac
        area = np.abs(bc * (vid // ncol - ar) - br * (vid % ncol - ac))
        best_area = np.full(count, -1, np.int64)
        np.maximum.at(best_area, vroot, area)
        vcx = np.full(count, big, np.int64)
        hit = area == best_area[vroot]
        np.minimum.at(vcx, vroot[hit], vid[hit])
        roots = np.unique(root)
        # extreme-vertex planes of the roots
        a, b, c = va[roots], vb[roots], vcx[roots]
        dcs = np.stack([b % ncol - a % ncol, c % ncol - a % ncol], -1).astype(np.float64)
        drs = np.stack([b // ncol - a // ncol, c // ncol - a // ncol], -1).astype(np.float64)
        dzs = np.stack([zf[b] - zf[a], zf[c] - zf[a]], -1)
        degenerate = best_area[roots] <= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = _plane_gradient(dcs, drs, dzs, dx, dy)
        grad = np.where(degenerate[:, None], 0.0, grad).astype(np.float32).astype(np.float64)
        root_index = np.full(count, -1, np.int64)
        root_index[roots] = np.arange(len(roots))
        g = grad[root_index[vroot]]
        anchor_of = va[vroot]
        res = (
            (zf[vid] - zf[anchor_of])
            - g[:, 0] * ((vid % ncol - anchor_of % ncol) * dx)
            - g[:, 1] * ((vid // ncol - anchor_of // ncol) * dy)
        )
        res = res.astype(np.float32).astype(np.float64)
        hres = (zf[vid] - zf[anchor_of]).astype(np.float32).astype(np.float64)
        rmin, rmax = np.full(count, np.inf), np.full(count, -np.inf)
        hmin, hmax = np.full(count, np.inf), np.full(count, -np.inf)
        np.minimum.at(rmin, vroot, res)
        np.maximum.at(rmax, vroot, res)
        np.minimum.at(hmin, vroot, hres)
        np.maximum.at(hmax, vroot, hres)
        half_e = np.where(degenerate, np.inf, 0.5 * (rmax[roots] - rmin[roots]) * _normal_scale(grad))
        half_h = 0.5 * (hmax[roots] - hmin[roots])
        # seed planes (the lowest-id triangle) of the roots
        seed_vertices = tv[roots]
        sdc = (seed_vertices[:, 1:] % ncol - (seed_vertices[:, :1] % ncol)).astype(np.float64)
        sdr = (seed_vertices[:, 1:] // ncol - (seed_vertices[:, :1] // ncol)).astype(np.float64)
        sdz = zf[seed_vertices[:, 1:]] - zf[seed_vertices[:, :1]]
        seed_grad = _plane_gradient(sdc, sdr, sdz, dx, dy).astype(np.float32).astype(np.float64)
        sg = seed_grad[root_index[vroot]]
        seed_anchor_of = tv[vroot, 0]
        sres = (
            (zf[vid] - zf[seed_anchor_of])
            - sg[:, 0] * ((vid % ncol - seed_anchor_of % ncol) * dx)
            - sg[:, 1] * ((vid // ncol - seed_anchor_of // ncol) * dy)
        )
        sres = np.abs(sres.astype(np.float32).astype(np.float64)) * _normal_scale(sg)
        seed_member = np.all((sres <= flat_eps).reshape(-1, 3), axis=1)
        # the extreme-vertex plane and the horizontal plane, offsets centred on their residual ranges
        off_e, off_h = np.zeros(count), np.zeros(count)
        off_e[roots] = (0.5 * (rmin[roots] + rmax[roots])).astype(np.float32)
        off_h[roots] = (0.5 * (hmin[roots] + hmax[roots])).astype(np.float32)
        eres = np.abs((res - off_e[vroot]).astype(np.float32).astype(np.float64)) * _normal_scale(g)
        extreme_member = np.all((eres <= flat_eps).reshape(-1, 3), axis=1)
        hres = np.abs((hres - off_h[vroot]).astype(np.float32).astype(np.float64))
        level_member = np.all((hres <= flat_eps).reshape(-1, 3), axis=1)
        counts = np.stack(
            [
                np.bincount(root, weights=seed_member, minlength=count)[roots],
                np.bincount(root, weights=extreme_member, minlength=count)[roots],
                np.bincount(root, weights=level_member, minlength=count)[roots],
            ],
            1,
        )
        # a planar component takes the closer of its extreme and level planes; another component the plane that
        # claims the most triangles, the seed's first and then the extreme one's on ties
        choice = np.argmax(counts, axis=1)
        planar = np.minimum(half_e, half_h) <= flat_eps
        choice = np.where(planar, np.where(half_e <= half_h, 1, 2), choice)
        root_choice = np.zeros(count, np.int64)
        root_choice[roots] = choice
        member = np.choose(root_choice[root], [seed_member, extreme_member, level_member])
        # a cluster is named by its lowest member triangle
        label = np.full(count, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(label, root[member], idx[member])
        root_of[idx[member]] = label[root[member]]
        active[idx[member]] = False
        for k, rt in enumerate(roots):
            name = int(label[rt])
            if name == np.iinfo(np.int64).max:
                continue
            if choice[k] == 1:
                records[name] = (int(va[rt]), np.float32(grad[k, 0]), np.float32(grad[k, 1]), np.float32(off_e[rt]))
            elif choice[k] == 2:
                records[name] = (int(va[rt]), np.float32(0.0), np.float32(0.0), np.float32(off_h[rt]))
            else:
                records[name] = (
                    int(tv[rt, 0]),
                    np.float32(seed_grad[k, 0]),
                    np.float32(seed_grad[k, 1]),
                    np.float32(0.0),
                )
    rest = np.flatnonzero(active)
    root_of[rest] = rest
    sizes = np.bincount(root_of, minlength=count)
    records = {rt: rec for rt, rec in records.items() if sizes[rt] > 1}
    return root_of, records


def build_prototype_clusters(Z: np.ndarray, hx: float, hy: float, *, flat_eps: float = FLAT_EPS) -> np.ndarray:
    """Cluster roots of the design study's prototype (the plane of the lowest-id triangle claims its members)."""
    nrow, ncol = Z.shape
    rows, cols = nrow - 1, ncol - 1
    dx, dy = 2.0 * hx / cols, 2.0 * hy / rows
    x = -hx + np.arange(ncol) * dx
    y = -hy + np.arange(nrow) * dy
    X, Y = np.meshgrid(x, y)
    P = np.stack([X, Y, Z], -1)
    p00, p10, p01, p11 = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    n0 = np.cross(p10 - p00, p11 - p00)
    n1 = np.cross(p11 - p00, p01 - p00)
    n0 /= np.linalg.norm(n0, axis=-1, keepdims=True)
    n1 /= np.linalg.norm(n1, axis=-1, keepdims=True)
    n0 *= np.sign(n0[..., 2:3])
    n1 *= np.sign(n1[..., 2:3])
    ids = np.arange(rows * cols).reshape(rows, cols) * 2

    def flat(n_a, p_a, o_b):
        return np.abs(np.einsum("...i,...i->...", n_a, o_b - p_a)) <= flat_eps

    ra, rb = [], []
    m = flat(n0, p00, p01)
    ra.append(ids[m])
    rb.append(ids[m] + 1)
    m = flat(n0[1:], p00[1:], p00[:-1])
    ra.append(ids[1:][m])
    rb.append(ids[:-1][m] + 1)
    m = flat(n1[:, 1:], p00[:, 1:], p00[:, :-1])
    ra.append(ids[:, 1:][m] + 1)
    rb.append(ids[:, :-1][m])
    ra, rb = np.concatenate(ra), np.concatenate(rb)
    count = 2 * rows * cols
    V = np.empty((rows, cols, 2, 3, 3))
    V[:, :, 0] = np.stack([p00, p10, p11], 2)
    V[:, :, 1] = np.stack([p00, p11, p01], 2)
    V = V.reshape(count, 3, 3)
    N = np.stack([n0, n1], 2).reshape(count, 3)
    root_of = np.full(count, -1, np.int64)
    active = np.ones(count, bool)
    for _round in range(CLUSTER_ROUNDS):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        keep = active[ra] & active[rb]
        parent = connected_roots(count, ra[keep], rb[keep])
        canon = parent[idx]
        dev = np.abs(np.einsum("ni,nki->nk", N[canon], V[idx] - V[canon][:, :1, :])).max(1)
        ok = dev <= flat_eps
        root_of[idx[ok]] = canon[ok]
        active[idx[ok]] = False
    rest = np.flatnonzero(active)
    root_of[rest] = rest
    return root_of


def cluster_ids(root_of: np.ndarray) -> np.ndarray:
    """Newton's per-triangle cluster ids: the dense index of a multi-triangle cluster (by root), else -(1 + id)."""
    count = len(root_of)
    sizes = np.bincount(root_of, minlength=count)
    multi = sizes > 1
    dense = np.cumsum(multi) - 1
    return np.where(multi[root_of], dense[root_of], -(1 + np.arange(count))).astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------
# surface
# ----------------------------------------------------------------------------------------------------------------


class ReferenceHeightfield:
    """The heightfield surface of the law: vertices, triangles, plane clusters, edges and vertex features.

    Args:
        Z: Decoded heights (nrow, ncol) [m], float64 (see :func:`decode_heights`).
        hx: Half-extent along x [m].
        hy: Half-extent along y [m].
        prototype_clusters: Use the prototype's cluster planes (the plane of the lowest-id triangle).
    """

    def __init__(self, Z, hx: float, hy: float, *, prototype_clusters: bool = False):
        self.Z = np.asarray(Z, np.float64)
        self.nrow, self.ncol = self.Z.shape
        self.rows, self.cols = self.nrow - 1, self.ncol - 1
        self.hx, self.hy = float(hx), float(hy)
        self.dx = 2.0 * self.hx / self.cols
        self.dy = 2.0 * self.hy / self.rows
        self.prototype = prototype_clusters
        if prototype_clusters:
            self.cluster = build_prototype_clusters(self.Z, self.hx, self.hy)
            self.records = {}
        else:
            self.cluster, self.records = build_reference_clusters(self.Z, self.dx, self.dy)
        self.sizes = np.bincount(self.cluster, minlength=len(self.cluster))
        self._plane_cache = {}

    # ---- vertices and triangles ------------------------------------------------------------------------------
    def vx(self, c):
        return -self.hx + c * self.dx

    def vy(self, r):
        return -self.hy + r * self.dy

    def valid_vertex(self, r, c):
        return 0 <= r < self.nrow and 0 <= c < self.ncol

    def V(self, r, c):
        return np.array([self.vx(c), self.vy(r), self.Z[r, c]])

    def tri_id(self, r, c, sub):
        return 2 * (r * self.cols + c) + sub

    def tri_rc(self, t):
        cell, sub = divmod(int(t), 2)
        r, c = divmod(cell, self.cols)
        return r, c, sub

    def tri_vidx(self, t):
        r, c, sub = self.tri_rc(t)
        if sub == 0:
            return ((r, c), (r, c + 1), (r + 1, c + 1))
        return ((r, c), (r + 1, c + 1), (r + 1, c))

    def tri_verts(self, t):
        return [self.V(*rc) for rc in self.tri_vidx(t)]

    def valid_cell(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def triangle_plane(self, t):
        """(unit upward normal, a vertex) of triangle t's own plane."""
        a, b, c = self.tri_verts(t)
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        return (n if n[2] > 0 else -n), a

    def plane(self, t):
        """(unit upward normal, a point) of the plane of triangle t's cluster."""
        t = int(t)
        root = int(self.cluster[t])
        cached = self._plane_cache.get(root)
        if cached is not None:
            return cached
        if self.prototype or self.sizes[root] == 1:
            out = self.triangle_plane(root)
        else:
            anchor, gx, gy, off = self.records[root]
            r, c = divmod(anchor, self.ncol)
            n = np.array([-float(gx), -float(gy), 1.0])
            n /= np.linalg.norm(n)
            out = (n, self.V(r, c) + np.array([0.0, 0.0, float(off)]))
        self._plane_cache[root] = out
        return out

    def same_cluster(self, t, s):
        return t == s or self.cluster[t] == self.cluster[s]

    def locate(self, x, y):
        """Triangle containing the footprint (x, y) under the half-open partition (None outside the footprint)."""
        u = (x + self.hx) / self.dx
        v = (y + self.hy) / self.dy
        c = int(np.floor(u))
        r = int(np.floor(v))
        if c == self.cols and abs(u - self.cols) < 1e-9:
            c -= 1
        if r == self.rows and abs(v - self.rows) < 1e-9:
            r -= 1
        if not self.valid_cell(r, c):
            return None
        fx, fy = u - c, v - r
        return self.tri_id(r, c, 0 if fx >= fy else 1)

    def height(self, x, y):
        t = self.locate(x, y)
        if t is None:
            return None
        n, a = self.plane(t)
        return a[2] - (n[0] * (x - a[0]) + n[1] * (y - a[1])) / n[2]

    # ---- window ------------------------------------------------------------------------------------------------
    def cells_in_box(self, lo, hi):
        c0 = int(np.floor((lo[0] + self.hx) / self.dx))
        c1 = int(np.floor((hi[0] + self.hx) / self.dx))
        r0 = int(np.floor((lo[1] + self.hy) / self.dy))
        r1 = int(np.floor((hi[1] + self.hy) / self.dy))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, self.cols - 1), min(r1, self.rows - 1)
        if c1 < c0 or r1 < r0:
            return None
        return r0, r1, c0, c1

    def window_features(self, box):
        """Triangles, edges and vertices of the cells [r0..r1] x [c0..c1].

        Edges are (kind, r, c, v0, v1, tri_a, tri_b, opp_a, opp_b) with tri_a/tri_b the two adjacent triangles
        (None outside the heightfield) and opp_a/opp_b their vertices opposite the edge.
        """
        r0, r1, c0, c1 = box
        tris = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                tris.append(self.tri_id(r, c, 0))
                tris.append(self.tri_id(r, c, 1))
        edges = []
        for r in range(r0, r1 + 2):
            for c in range(c0, c1 + 1):
                ta = self.tri_id(r - 1, c, 1) if self.valid_cell(r - 1, c) else None
                tb = self.tri_id(r, c, 0) if self.valid_cell(r, c) else None
                edges.append(("H", r, c, (r, c), (r, c + 1), ta, tb, (r - 1, c), (r + 1, c + 1)))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 2):
                ta = self.tri_id(r, c - 1, 0) if self.valid_cell(r, c - 1) else None
                tb = self.tri_id(r, c, 1) if self.valid_cell(r, c) else None
                edges.append(("V", r, c, (r, c), (r + 1, c), ta, tb, (r, c - 1), (r + 1, c + 1)))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                edges.append(
                    (
                        "D",
                        r,
                        c,
                        (r, c),
                        (r + 1, c + 1),
                        self.tri_id(r, c, 0),
                        self.tri_id(r, c, 1),
                        (r, c + 1),
                        (r + 1, c),
                    )
                )
        verts = [(r, c) for r in range(r0, r1 + 2) for c in range(c0, c1 + 2)]
        return tris, edges, verts

    def edge_class(self, e):
        """'convex', 'concave', 'flat' or 'border' of a window edge tuple: flat inside one cluster."""
        _k, _r, _c, _i0, _i1, ta, tb, _oa, ob = e
        if ta is None or tb is None:
            return "border"
        if self.cluster[ta] == self.cluster[tb]:
            return "flat"
        n, a = self.plane(ta)
        s = float(n @ (self.V(*ob) - a))
        return "concave" if s > 0.0 else "convex"

    def ring(self, r, c):
        """One-ring vertex indices of (r, c), or None at the heightfield border."""
        out = []
        for dr, dc in RING:
            if not self.valid_vertex(r + dr, c + dc):
                return None
            out.append((r + dr, c + dc))
        return out

    def incident_triangles(self, r, c):
        """The 6 triangles around vertex (r, c), counter-clockwise, matching the one-ring order."""
        return [
            self.tri_id(r, c, 0),
            self.tri_id(r, c, 1),
            self.tri_id(r, c - 1, 0),
            self.tri_id(r - 1, c - 1, 1),
            self.tri_id(r - 1, c - 1, 0),
            self.tri_id(r - 1, c, 1),
        ]

    def vertex_feature(self, r, c):
        """Whether vertex (r, c) is a feature of its own: its incident cluster normals span 3-D."""
        if self.ring(r, c) is None:
            return False
        return spans3([self.plane(t)[0] for t in self.incident_triangles(r, c)])


def spans3(normals) -> bool:
    """Whether unit normals span 3-D: three distinct directions with a triple product above ``VERTEX_SPAN``."""
    uniq = []
    for nn in normals:
        if all(np.linalg.norm(nn - u) > NORMAL_SAME for u in uniq):
            uniq.append(nn)
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            for k in range(j + 1, len(uniq)):
                if abs(np.linalg.det(np.stack([uniq[i], uniq[j], uniq[k]]))) > VERTEX_SPAN:
                    return True
    return False


# ----------------------------------------------------------------------------------------------------------------
# convex cores (heightfield-local frame)
# ----------------------------------------------------------------------------------------------------------------


def quat_rotate(q, v):
    """Rotate vector v (3,) by unit quaternion q (x, y, z, w)."""
    q = np.asarray(q, np.float64)
    v = np.asarray(v, np.float64)
    u, w = q[:3], q[3]
    return v * (2.0 * w * w - 1.0) + np.cross(u, v) * 2.0 * w + u * (2.0 * float(v @ u))


def quat_matrix(q):
    return np.stack([quat_rotate(q, e) for e in np.eye(3)], 1)


def reduce4(points, heights, normal):
    """Indices of at most 4 points: the deepest, the farthest from it, the largest triangle, then the point adding
    the most area outside that triangle (areas in the plane normal to ``normal``). Lowest index wins ties."""
    n = len(points)
    if n <= 4:
        return list(range(n))
    P = np.asarray(points, np.float64)
    P = P - np.outer(P @ normal, normal)
    h = np.asarray(heights, np.float64)
    i0 = int(np.argmin(h))
    i1 = int(np.argmax(np.linalg.norm(P - P[i0], axis=1)))
    a2 = np.abs(np.cross(P[i1] - P[i0], P - P[i0]) @ normal)
    i2 = int(np.argmax(a2))
    best, i3 = 1e-15, None
    tri = (i0, i1, i2)
    for j in range(n):
        if j in tri:
            continue
        gain = 0.0
        for a, b, c in ((i0, i1, i2), (i1, i2, i0), (i2, i0, i1)):
            side_c = np.cross(P[b] - P[a], P[c] - P[a]) @ normal
            side_j = np.cross(P[b] - P[a], P[j] - P[a]) @ normal
            if side_c * side_j < 0:
                gain = max(gain, abs(side_j))
        if gain > best:
            best, i3 = gain, j
    out = [i0, i1, i2] + ([i3] if i3 is not None else [])
    return list(dict.fromkeys(out))


def convex_hull_faces(verts, *, tol: float = 1.0e-9):
    """Faces of the convex hull of ``verts``: vertex index lists and outward unit normals (brute force, small sets)."""
    V = np.asarray(verts, np.float64)
    n = len(V)
    faces = {}
    scale = max(float(np.abs(V).max()), 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                normal = np.cross(V[j] - V[i], V[k] - V[i])
                length = float(np.linalg.norm(normal))
                if length < 1e-12 * scale * scale:
                    continue
                normal /= length
                s = (V - V[i]) @ normal
                if np.all(s <= tol * scale):
                    pass
                elif np.all(s >= -tol * scale):
                    normal, s = -normal, -s
                else:
                    continue
                key = (*np.round(normal, 7).tolist(), round(float(normal @ V[i]), 7))
                if key not in faces:
                    faces[key] = (normal, sorted(int(m) for m in np.flatnonzero(np.abs(s) <= tol * scale)))
    normals = [f[0] for f in faces.values()]
    lists = [f[1] for f in faces.values()]
    return lists, normals


class Core:
    r_eff = 0.0
    kind = "core"


class Sphere(Core):
    kind = "sphere"

    def __init__(self, c, radius):
        self.c = np.asarray(c, np.float64)
        self.r_eff = float(radius)

    def lowest(self, n):
        return self.c.copy()

    def aabb(self):
        return self.c.copy(), self.c.copy()

    def plane_manifold(self, n, a, D):
        h = float(n @ (self.c - a))
        return [(self.c.copy(), h)] if h <= D else []


class Capsule(Core):
    kind = "capsule"

    def __init__(self, c, axis, half_height, radius):
        self.c = np.asarray(c, np.float64)
        self.axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
        self.h = float(half_height)
        self.r_eff = float(radius)
        self.a = self.c + self.h * self.axis
        self.b = self.c - self.h * self.axis

    def lowest(self, n):
        return self.a.copy() if n @ self.a <= n @ self.b else self.b.copy()

    def aabb(self):
        return np.minimum(self.a, self.b), np.maximum(self.a, self.b)

    def plane_manifold(self, n, a, D):
        out = []
        for k in (self.a, self.b):
            h = float(n @ (k - a))
            if h <= D:
                out.append((k.copy(), h))
        return out


class Polytope(Core):
    """Convex polyhedron core: vertices and faces (vertex index lists, outward unit normals)."""

    kind = "polytope"

    def __init__(self, verts, faces, normals):
        self.verts = np.asarray(verts, np.float64)
        self.faces = faces
        self.fnormals = np.asarray(normals, np.float64)
        self.foffsets = np.array([self.fnormals[i] @ self.verts[f[0]] for i, f in enumerate(faces)])

    def lowest(self, n):
        return self.verts[int(np.argmin(self.verts @ n))].copy()

    def aabb(self):
        return self.verts.min(0), self.verts.max(0)

    def lift(self, u, n):
        """Entry point of the line u + t n into the polyhedron (smallest t)."""
        t_lo, t_hi = -np.inf, np.inf
        for f_n, f_d in zip(self.fnormals, self.foffsets, strict=True):
            denom = f_n @ n
            num = f_d - f_n @ u
            if abs(denom) < 1e-14:
                if num < -1e-12:
                    return None
                continue
            t = num / denom
            if denom > 0:
                t_hi = min(t_hi, t)
            else:
                t_lo = max(t_lo, t)
        if t_lo > t_hi + 1e-12 or not np.isfinite(t_lo):
            return None
        return u + t_lo * n, t_lo

    def closest(self, v):
        """Closest point of the polyhedron to v when v is outside (None inside)."""
        if np.all(self.fnormals @ v - self.foffsets <= 1e-12):
            return None
        best, best_d = None, np.inf
        for f, fn in zip(self.faces, self.fnormals, strict=True):
            p = v - (fn @ v - fn @ self.verts[f[0]]) * fn
            if _in_face(self.verts[f], fn, p):
                d = float(np.linalg.norm(p - v))
                if d < best_d:
                    best, best_d = p, d
        for i, j in self.edges():
            a, b = self.verts[i], self.verts[j]
            ab = b - a
            s = float(np.clip((v - a) @ ab / (ab @ ab), 0.0, 1.0))
            p = a + s * ab
            d = float(np.linalg.norm(p - v))
            if d < best_d:
                best, best_d = p, d
        return best

    def edges(self):
        if not hasattr(self, "_edges"):
            count = {}
            for f in self.faces:
                for a in range(len(f)):
                    for b in range(a + 1, len(f)):
                        key = (f[a], f[b])
                        count[key] = count.get(key, 0) + 1
            self._edges = [key for key, k in count.items() if k >= 2]
        return self._edges


def _in_face(points, normal, p, *, tol=1e-12):
    """Whether p (on the face plane) lies in the convex polygon of ``points``."""
    e1 = points[1] - points[0] if len(points) > 1 else np.array([1.0, 0.0, 0.0])
    e1 = e1 - (e1 @ normal) * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    P = np.stack([(points - points[0]) @ e1, (points - points[0]) @ e2], 1)
    H = hull2d(P)
    q = np.array([(p - points[0]) @ e1, (p - points[0]) @ e2])
    m = len(H)
    for i in range(m):
        a, b = H[i], H[(i + 1) % m]
        if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) < -tol:
            return False
    return True


class Box(Polytope):
    kind = "box"

    def __init__(self, c, rot, half):
        self.c = np.asarray(c, np.float64)
        self.R = np.asarray(rot, np.float64)
        self.half = np.asarray(half, np.float64)
        corners = []
        for i in range(8):
            loc = np.array(
                [
                    self.half[0] if i & 1 else -self.half[0],
                    self.half[1] if i & 2 else -self.half[1],
                    self.half[2] if i & 4 else -self.half[2],
                ]
            )
            corners.append(self.c + self.R @ loc)
        faces, normals = [], []
        for axis in range(3):
            for sign in (-1, 1):
                faces.append([i for i in range(8) if ((i >> axis) & 1) == (1 if sign > 0 else 0)])
                normals.append(sign * self.R[:, axis])
        super().__init__(corners, faces, normals)

    def plane_manifold(self, n, a, D):
        """Newton's plane-box manifold: up to 4 deepest corners within reach."""
        kept = []
        for i in range(8):
            h = float(n @ (self.verts[i] - a))
            if h <= D:
                kept.append((h, i))
        kept.sort(key=lambda x: (x[0], x[1]))
        return [(self.verts[i].copy(), h) for h, i in kept[:4]]


class Hull(Polytope):
    kind = "hull"

    def __init__(self, c, rot, verts_local):
        self.c = np.asarray(c, np.float64)
        self.R = np.asarray(rot, np.float64)
        v = self.c + np.asarray(verts_local, np.float64) @ self.R.T
        faces, normals = convex_hull_faces(v)
        super().__init__(v, faces, normals)

    def plane_manifold(self, n, a, D):
        """The reference face (most opposed to n) and the deepest vertex, within reach, reduced to 4 by area."""
        h = (self.verts - a) @ n
        ref = int(np.argmin(self.fnormals @ n))
        cand = set(self.faces[ref])
        cand.add(int(np.argmin(h)))
        idx = sorted(i for i in cand if h[i] <= D)
        if not idx:
            return []
        keep = reduce4(self.verts[idx], h[idx], n)
        sel = sorted((idx[k] for k in keep), key=lambda i: (h[i], i))
        return [(self.verts[i].copy(), float(h[i])) for i in sel]


class Cylinder(Core):
    kind = "cylinder"

    def __init__(self, c, axis, radius, half_height):
        self.c = np.asarray(c, np.float64)
        self.axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
        self.r = float(radius)
        self.h = float(half_height)

    def lowest(self, n):
        ax = self.axis
        s = -1.0 if n @ ax > 0 else 1.0
        cap = self.c + s * self.h * ax
        perp = -n + ax * (n @ ax)
        ln = np.linalg.norm(perp)
        if ln < 1e-12:
            return cap
        return cap + self.r * perp / ln

    def aabb(self):
        ax = self.axis
        ext = self.r * np.sqrt(np.maximum(0.0, 1.0 - ax * ax)) + self.h * np.abs(ax)
        return self.c - ext, self.c + ext

    def plane_manifold(self, n, a, D):
        """Newton's plane-cylinder manifold (deepest rim point plus a tripod or the rolling points) within reach."""
        axis = self.axis.copy()
        dot_na = n @ axis
        if dot_na > 0.0:
            axis = -axis
            dot_na = -dot_na
        cap = self.c + axis * self.h
        perp_align = -n + axis * dot_na
        la = perp_align @ perp_align
        has_align = la > 1e-10
        if has_align:
            perp_align = perp_align / math.sqrt(la)
        flat = -dot_na >= CYLINDER_FLAT_MODE_COS
        perp_fixed = np.zeros(3)
        if flat or not has_align:
            ref = np.array([1.0, 0.0, 0.0])
            if abs(axis @ ref) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            perp_fixed = ref - axis * (axis @ ref)
            perp_fixed /= np.linalg.norm(perp_fixed)
        deepest_perp = perp_align if has_align else perp_fixed
        pts = [cap + deepest_perp * self.r]
        merge = (0.01 * max(self.r, self.h)) ** 2
        if flat:
            u = perp_fixed * self.r
            v = np.cross(axis, perp_fixed) * self.r
            for p in (cap + u, cap - 0.5 * u + 0.8660254 * v, cap - 0.5 * u - 0.8660254 * v):
                if len(pts) < 4 and np.sum((p - pts[0]) ** 2) > merge:
                    pts.append(p)
        else:
            perp_roll = perp_align if has_align else perp_fixed
            u = perp_roll * self.r
            v = np.cross(axis, perp_roll) * self.r
            p = self.c - axis * self.h + u
            if np.sum((p - pts[0]) ** 2) > merge:
                pts.append(p)
            pp, pm = cap + v, cap - v
            p = pp if n @ (pp - a) <= n @ (pm - a) else pm
            if len(pts) < 4 and np.sum((p - pts[0]) ** 2) > merge:
                pts.append(p)
        out = []
        for p in pts:
            hh = float(n @ (p - a))
            if hh <= D:
                out.append((p, hh))
        return out

    def lift(self, u, n):
        """Entry of the line u + t n into the solid cylinder (smallest t)."""
        ax = self.axis
        w = u - self.c
        t_lo, t_hi = -np.inf, np.inf
        dn = n @ ax
        dw = w @ ax
        if abs(dn) < 1e-14:
            if abs(dw) > self.h:
                return None
        else:
            t1, t2 = (-self.h - dw) / dn, (self.h - dw) / dn
            t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))
        wp_ = w - dw * ax
        npp = n - dn * ax
        A = npp @ npp
        B = 2.0 * (wp_ @ npp)
        C = wp_ @ wp_ - self.r * self.r
        if A < 1e-14:
            if C > 0:
                return None
        else:
            disc = B * B - 4 * A * C
            if disc < 0:
                return None
            sq = math.sqrt(disc)
            t1, t2 = (-B - sq) / (2 * A), (-B + sq) / (2 * A)
            t_lo, t_hi = max(t_lo, t1), min(t_hi, t2)
        if t_lo > t_hi + 1e-12 or not np.isfinite(t_lo):
            return None
        return u + t_lo * n, t_lo


def make_core(kind: str, dims: dict, pose_local, *, verts=None) -> Core:
    """A core in the heightfield-local frame: ``pose_local`` = (px, py, pz, qx, qy, qz, qw)."""
    p = np.asarray(pose_local[:3], np.float64)
    R = quat_matrix(np.asarray(pose_local[3:7], np.float64))
    z = R[:, 2]
    if kind == "sphere":
        return Sphere(p, dims["radius"])
    if kind == "capsule":
        return Capsule(p, z, dims["half_height"], dims["radius"])
    if kind == "box":
        return Box(p, R, dims["half"])
    if kind == "cylinder":
        return Cylinder(p, z, dims["radius"], dims["half_height"])
    if kind == "hull":
        return Hull(p, R, verts)
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------------------------------------------------


class Row:
    """One contact row: terrain point q, core point k, unit normal n (terrain to shape), core distance d."""

    __slots__ = ("d", "feat", "group", "k", "n", "q")

    def __init__(self, q, k, n, d, feat, *, group=None):
        self.q, self.k, self.n, self.d, self.feat, self.group = q, k, n, float(d), feat, group

    def __repr__(self):
        return f"Row({self.feat}, d={self.d * 1e3:.3f}mm, n={np.round(self.n, 4)}, q={np.round(self.q, 4)})"


class Stats:
    """Work counters of one or more pairs."""

    def __init__(self):
        self.tris = self.edges = self.verts = 0
        self.convex_edges = self.convex_verts = 0
        self.face_tests = self.edge_solves = self.vertex_solves = 0
        self.emitted = self.merged = self.implied = self.reduced = 0


def _outward_edge_normals(hf, t):
    """In-plane outward normals of triangle t's edges (cluster plane), keyed by the edge's sorted vertex pair."""
    vid = hf.tri_vidx(t)
    vs = [hf.V(*x) for x in vid]
    n, _ = hf.plane(t)
    out = {}
    for i in range(3):
        a, b, c = vs[i], vs[(i + 1) % 3], vs[(i + 2) % 3]
        m = np.cross(b - a, n)
        if m @ (c - a) > 0:
            m = -m
        m = m / np.linalg.norm(m)
        out[tuple(sorted((vid[i], vid[(i + 1) % 3])))] = (m, a)
    return out


def _below_surface(hf, p):
    h = hf.height(p[0], p[1])
    return h is not None and p[2] < h


class PointLaw:
    """Local minima of the distance from a point to the surface (exact, shared predicates)."""

    def __init__(self, hf: ReferenceHeightfield, tris, edges, verts):
        self.hf = hf
        self.tris = tris
        self.edges = edges
        self.verts = verts
        self.m = {t: _outward_edge_normals(hf, t) for t in tris}
        # a face's boundary point belongs to the face at convex, flat and border edges (closed test) and to the
        # neighbour's own projection at concave edges (strict test)
        self.cls = {tuple(sorted((e[3], e[4]))): hf.edge_class(e) for e in edges}

    def face_pred(self, t, key, p):
        if t not in self.m:
            self.m[t] = _outward_edge_normals(self.hf, t)
        m, a = self.m[t][key]
        return float((p - a) @ m)

    def minima(self, p, D, tag, *, stats=None):
        hf = self.hf
        out = []
        below = _below_surface(hf, p)
        for t in self.tris:
            n, a = hf.plane(t)
            d = float(n @ (p - a))
            if d > D:
                continue
            if stats:
                stats.face_tests += 1
            # front-facing only
            if (d > EPS_PRED and below) or (d < -EPS_PRED and not below):
                continue
            inside = True
            on_boundary = 0
            for key in self.m[t]:
                s = self.face_pred(t, key, p)
                if s > EPS_PRED or (s >= -EPS_PRED and self.cls.get(key) == "concave"):
                    inside = False
                    break
                if s >= -EPS_PRED:
                    on_boundary += 1
            # a projection on a corner of the face is the vertex's to decide
            if inside and on_boundary < 2:
                out.append(Row(p - d * n, p.copy(), n.copy(), d, (tag, "F", t)))
        for e in self.edges:
            _k, _r, _c, i0, i1, ta, tb, _oa, _ob = e
            if ta is None or tb is None:
                continue
            key = tuple(sorted((i0, i1)))
            sa = self.face_pred(ta, key, p)
            sb = self.face_pred(tb, key, p)
            if not (sa > EPS_PRED and sb > EPS_PRED):
                continue
            v0, v1 = hf.V(*i0), hf.V(*i1)
            u = v1 - v0
            w0 = float((p - v0) @ u)
            w1 = float((p - v1) @ (-u))
            if not (w0 > EPS_PRED and w1 > EPS_PRED):
                continue
            q = v0 + (w0 / float(u @ u)) * u
            dist = float(np.linalg.norm(p - q))
            if dist < 1e-12:
                n = 0.5 * (hf.plane(ta)[0] + hf.plane(tb)[0])
                n /= np.linalg.norm(n)
                d = 0.0
            elif below:
                n, d = (q - p) / dist, -dist
            else:
                n, d = (p - q) / dist, dist
            if d > D:
                continue
            out.append(Row(q, p.copy(), n, d, (tag, "E", e[0], e[1], e[2])))
        for r, c in self.verts:
            ring = hf.ring(r, c)
            if ring is None:
                continue
            v = hf.V(r, c)
            w = p - v
            if any(float(w @ (hf.V(rr, cc) - v)) > EPS_PRED for rr, cc in ring):
                continue
            dist = float(np.linalg.norm(w))
            if dist < 1e-12:
                continue
            n, d = (-w / dist, -dist) if below else (w / dist, dist)
            if d > D:
                continue
            out.append(Row(v, p.copy(), n, d, (tag, "V", r, c)))
        if below:
            # G2: a core point inside the solid keeps one exit, the minimum translation
            exits = [ct for ct in out if ct.d <= EPS_PRED]
            rest = [ct for ct in out if ct.d > EPS_PRED]
            if exits:
                out = [*rest, min(exits, key=lambda ct: (-ct.d, str(ct.feat)))]
            else:
                # G3: the face vertically above a core point with no exit in the window
                rec = _recovery_row(hf, p, (tag, "R"))
                if rec is not None:
                    out = [*rest, rec]
        return out


def _recovery_row(hf, p, feat):
    """G3: the face vertically above a core point below the surface, plane normal, plane depth."""
    t = hf.locate(p[0], p[1])
    if t is None:
        return None
    n, a = hf.plane(t)
    d = float(n @ (p - a))
    if d >= 0.0:
        return None
    return Row(p - d * n, p.copy(), n.copy(), d, feat)


def _solid_recovery(hf, core, *, kmax=4):
    """G3 for solid cores with no row: up to 4 of the core's deepest boundary points below the surface."""
    if hasattr(core, "verts"):
        pts = core.verts
    else:
        ax = core.axis
        ref = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = ref - ax * float(ax @ ref)
        u /= np.linalg.norm(u)
        w = np.cross(ax, u)
        th = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        ring = core.r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), w))
        pts = np.concatenate([core.c + core.h * ax + ring, core.c - core.h * ax + ring])
    cands = []
    for i, x in enumerate(pts):
        t = hf.locate(x[0], x[1])
        if t is None:
            continue
        n, a = hf.plane(t)
        d = float(n @ (x - a))
        if d < 0.0:
            cands.append((d, i, x, n, t))
    cands.sort(key=lambda z: (z[0], z[1]))
    return [Row(x - d * n, x.copy(), n.copy(), d, ("R", i), group=("P", t)) for d, i, x, n, t in cands[:kmax]]


def _closest_seg_seg(p0, p1, q0, q1):
    """Parameters (s, t) of the closest points of segments p0p1 and q0q1 (None when parallel)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c = d1 @ r
    b = d1 @ d2
    denom = a * e - b * b
    if denom <= 1e-14 * a * e:
        return None
    s = (b * f - c * e) / denom
    t = (b * s + f) / e
    return float(s), float(t)


def sphere_law(hf, core, D, window, *, stats=None):
    tris, edges, verts = window
    return PointLaw(hf, tris, edges, verts).minima(core.c, D, "c", stats=stats)


def capsule_law(hf, core, D, window, *, stats=None):
    tris, edges, verts = window
    law = PointLaw(hf, tris, edges, verts)
    out = law.minima(core.a, D, "a", stats=stats) + law.minima(core.b, D, "b", stats=stats)
    a, b = core.a, core.b
    ab = b - a
    # axis-interior minima against edges
    for e in edges:
        _k, _r, _c, i0, i1, ta, tb, _oa, _ob = e
        if ta is None or tb is None:
            continue
        v0, v1 = hf.V(*i0), hf.V(*i1)
        st = _closest_seg_seg(a, b, v0, v1)
        if st is None:
            continue
        s, t = st
        if not (0.0 < s < 1.0 and 0.0 < t < 1.0):
            continue
        p = a + s * ab
        key = tuple(sorted((i0, i1)))
        if not (law.face_pred(ta, key, p) > 0.0 and law.face_pred(tb, key, p) > 0.0):
            continue
        q = v0 + t * (v1 - v0)
        dist = float(np.linalg.norm(p - q))
        if dist < 1e-12:
            continue
        below = _below_surface(hf, p)
        n, d = ((q - p) / dist, -dist) if below else ((p - q) / dist, dist)
        if d > D:
            continue
        out.append(Row(q, p, n, d, ("s", "E", e[0], e[1], e[2])))
    # axis-interior minima against vertices
    L2 = float(ab @ ab)
    for r, c in verts:
        ring = hf.ring(r, c)
        if ring is None:
            continue
        v = hf.V(r, c)
        s = float((v - a) @ ab) / L2
        if not (0.0 < s < 1.0):
            continue
        p = a + s * ab
        w = p - v
        if any(float(w @ (hf.V(rr, cc) - v)) > 0.0 for rr, cc in ring):
            continue
        dist = float(np.linalg.norm(w))
        if dist < 1e-12:
            continue
        below = _below_surface(hf, p)
        n, d = (-w / dist, -dist) if below else (w / dist, dist)
        if d > D:
            continue
        out.append(Row(v, p, n, d, ("s", "V", r, c)))
    return out


def _sigma(core, n, x0):
    k = core.lowest(n)
    return float(n @ (k - x0)), k


def _maximize_arc(core, nA, m, theta_ab, x0, *, samples=9, iters=40):
    """argmax over theta in [0, theta_ab] of sigma(n(theta)), n(theta) = cos nA + sin m (coarse scan + golden)."""
    thetas = np.linspace(0.0, theta_ab, samples)
    vals = [_sigma(core, math.cos(t) * nA + math.sin(t) * m, x0)[0] for t in thetas]
    j = int(np.argmax(vals))
    lo = thetas[max(j - 1, 0)]
    hi = thetas[min(j + 1, samples - 1)]
    g = (math.sqrt(5.0) - 1.0) / 2.0

    def f(t):
        return _sigma(core, math.cos(t) * nA + math.sin(t) * m, x0)[0]

    c = hi - g * (hi - lo)
    d = lo + g * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = f(d)
    t = 0.5 * (lo + hi)
    best_t, best_v = t, f(t)
    for tt, vv in ((0.0, vals[0]), (theta_ab, vals[-1])):
        if vv >= best_v - 1e-15:
            if vv > best_v + 1e-15 or abs(tt - best_t) < 1e-12:
                best_t, best_v = tt, vv
    if vals[j] > best_v:
        best_t, best_v = thetas[j], vals[j]
    return best_t, best_v


def _support_feature(core, n, band):
    """The core's flat support feature along -n within ``band``: ('pts', P) for a polytope's face, edge or vertex
    or a cylinder's lowest point or generator, ('disk', centre, axis, r) for a cylinder cap within the band."""
    if hasattr(core, "verts"):
        h = core.verts @ n
        return ("pts", core.verts[h <= h.min() + band])
    ax = core.axis
    k0 = core.lowest(n)
    s = -1.0 if n @ ax > 0 else 1.0
    tilt = math.sqrt(max(0.0, 1.0 - float(n @ ax) ** 2))
    if 2.0 * core.r * tilt <= band:
        return ("disk", core.c + s * core.h * ax, ax, core.r)
    k1 = k0 - 2.0 * s * core.h * ax
    if float(n @ (k1 - k0)) <= band:
        return ("pts", np.stack([k0, k1]))
    return ("pts", k0[None, :])


def _reference_feature(core, n):
    """The core's reference feature at normal n, as the plane manifold sees it (G1): a polytope's face through its
    deepest vertex that is most opposed to n; a cylinder's near cap inside the 22.5 deg flat mode, else its lowest
    generator."""
    if hasattr(core, "verts"):
        h = core.verts @ n
        deepest = int(np.argmin(h))
        best, k = -np.inf, -1
        for q, f in enumerate(core.faces):
            if deepest in f:
                sc = -float(core.fnormals[q] @ n)
                if sc > best:
                    best, k = sc, q
        return ("pts", core.verts[core.faces[k]])
    axis = core.axis if float(n @ core.axis) < 0.0 else -core.axis
    cap = core.c + core.h * axis
    if -float(n @ axis) >= CYLINDER_FLAT_MODE_COS:
        return ("disk", cap, axis, core.r)
    perp = -n + axis * float(n @ axis)
    ln = float(np.linalg.norm(perp))
    if ln < 1e-12:
        return ("disk", cap, axis, core.r)
    perp = perp / ln
    return ("pts", np.stack([core.c - axis * core.h + perp * core.r, core.c + axis * core.h + perp * core.r]))


def _feature_interval(feat, v0, e1, e2, tol, *, need_straddle=False):
    """Interval of the edge-line coordinate x (along e1 from v0) where the feature, projected along n = e1 x e2,
    meets the line y = 0; None when it misses (or, with need_straddle, when it does not cross the line)."""
    if feat[0] == "pts":
        S = feat[1]
        P = np.stack([(S - v0) @ e1, (S - v0) @ e2], 1)
        if need_straddle and not (P[:, 1].min() < -1e-9 and P[:, 1].max() > 1e-9):
            return None
        return _hull_line_interval(hull2d(P), tol)
    _kind, C, axis, r = feat
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - axis * float(axis @ ref)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    yc, xc = float((C - v0) @ e2), float((C - v0) @ e1)
    A_, B_ = float(u @ e2), float(w @ e2)
    R = r * math.hypot(A_, B_)
    if R < 1e-15:
        if abs(yc) > tol or need_straddle:
            return None
        ext = r * math.hypot(float(u @ e1), float(w @ e1))
        return (xc - ext, xc + ext)
    if abs(yc) > R + tol:
        return None
    if need_straddle and abs(yc) >= R - 1e-9:
        return None
    phi = math.atan2(B_, A_)
    dl = math.acos(max(-1.0, min(1.0, -yc / R)))
    xs = [xc + r * (math.cos(th) * float(u @ e1) + math.sin(th) * float(w @ e1)) for th in (phi + dl, phi - dl)]
    return (min(xs), max(xs))


def hull2d(P):
    """Convex hull (counter-clockwise) of 2-D points (monotone chain); returns the hull points."""
    pts = sorted({(round(float(x), 15), round(float(y), 15)) for x, y in P})
    if len(pts) <= 2:
        return np.asarray(pts, np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], np.float64)


def _hull_line_interval(H, tol):
    """Interval of x where the 2-D convex polygon H meets the line y = 0 (None if it misses)."""
    if len(H) == 0:
        return None
    if len(H) == 1:
        return (H[0, 0], H[0, 0]) if abs(H[0, 1]) <= tol else None
    xs = []
    m = len(H)
    for i in range(m):
        a, b = H[i], H[(i + 1) % m] if m > 2 else H[1 - i]
        ya, yb = a[1], b[1]
        if abs(ya) <= tol:
            xs.append(a[0])
        if (ya < -tol and yb > tol) or (ya > tol and yb < -tol):
            s = ya / (ya - yb)
            xs.append(a[0] + s * (b[0] - a[0]))
        if m == 2 and i == 1:
            break
    if not xs:
        return None
    return min(xs), max(xs)


def solid_law(hf, core, D, window, *, stats=None, flat_band=None):
    tris, edges, verts = window
    out = []
    size = float(np.linalg.norm(core.aabb()[1] - core.aabb()[0]))
    band = flat_band if flat_band is not None else max(1e-4, 1e-3 * size)
    # ---- face clusters ---------------------------------------------------------------------------------------
    for t in tris:
        n, a = hf.plane(t)
        cands = core.plane_manifold(n, a, D)
        if not cands:
            continue
        if stats:
            stats.face_tests += 1
        sup = []
        for k, h in cands:
            q = k - h * n
            s = hf.locate(q[0], q[1])
            ok = s is not None and hf.same_cluster(t, s)
            if ok and abs(h) > EPS_PRED:
                # front-facing only
                ok = (h < 0.0) == _below_surface(hf, k)
            sup.append(ok)
        try:
            j = sup.index(True)
        except ValueError:
            continue
        k0, h0 = cands[j]
        q0 = k0 - h0 * n
        if hf.locate(q0[0], q0[1]) != t:
            continue
        for (k, h), ok in zip(cands, sup, strict=True):
            if ok:
                out.append(Row(k - h * n, k.copy(), n.copy(), h, ("F", t), group=("P", t)))
    # ---- convex edges ----------------------------------------------------------------------------------------
    vcache = {}

    def vfeat(rc):
        if rc not in vcache:
            vcache[rc] = hf.vertex_feature(*rc)
        return vcache[rc]

    for e in edges:
        if hf.edge_class(e) != "convex":
            continue
        _k, er, ec, i0, i1, ta, tb, _oa, _ob = e
        if stats:
            stats.convex_edges += 1
        v0, v1 = hf.V(*i0), hf.V(*i1)
        dvec = v1 - v0
        L = float(np.linalg.norm(dvec))
        dh = dvec / L
        nA, _ = hf.plane(ta)
        nB, _ = hf.plane(tb)
        # the edge's normal arc, exactly perpendicular to the edge
        nA = nA - (nA @ dh) * dh
        nA /= np.linalg.norm(nA)
        nB = nB - (nB @ dh) * dh
        nB /= np.linalg.norm(nB)
        cosab = float(np.clip(nA @ nB, -1.0, 1.0))
        th_ab = math.acos(cosab)
        m = nB - cosab * nA
        if th_ab < 1e-9 or np.linalg.norm(m) < 1e-12:
            continue
        m /= np.linalg.norm(m)
        lo_s = max(_sigma(core, nA, v0)[0], _sigma(core, nB, v0)[0])
        if lo_s > D:
            continue
        if stats:
            stats.edge_solves += 1
        th, sig = _maximize_arc(core, nA, m, th_ab, v0)
        if sig > D:
            continue
        nst = math.cos(th) * nA + math.sin(th) * m
        interior = EPS_THETA < th < th_ab - EPS_THETA
        e1, e2 = dh, np.cross(nst, dh)
        # G1: interior theta*: the flat support feature along -n(theta*); arc end: the reference feature
        feat = _support_feature(core, nst, band) if interior else _reference_feature(core, nst)
        iv = _feature_interval(feat, v0, e1, e2, 1e-7 + 1e-6 * size, need_straddle=not interior)
        xs = []
        if iv is not None:
            x0, x1 = max(iv[0], 0.0), min(iv[1], L)
            if x1 >= x0 - 1e-12:
                xs = [x0, x1] if x1 - x0 >= 1e-9 else [0.5 * (x0 + x1)]
        # an end of the edge on a vertex feature belongs to that vertex
        xs = [x for x in xs if not ((x <= 1e-12 and vfeat(i0)) or (x >= L - 1e-12 and vfeat(i1)))]
        if not xs:
            continue
        grp = ("E", e[0], er, ec, round(th, 9)) if interior else ("P", ta if th < 0.5 * th_ab else tb)
        for x in xs:
            u = v0 + x * dh
            hit = core.lift(u, nst)
            if hit is None:
                continue
            k, tt = hit
            if tt > D:
                continue
            out.append(Row(u, k, nst.copy(), tt, ("E", e[0], er, ec, round(x / L, 6)), group=grp))
    # ---- vertex features -------------------------------------------------------------------------------------
    for r, c in verts:
        ring = hf.ring(r, c)
        if ring is None:
            continue
        if not hf.vertex_feature(r, c):
            continue
        inc = hf.incident_triangles(r, c)
        normals = [hf.plane(t)[0] for t in inc]
        v = hf.V(r, c)
        W = [hf.V(rr, cc) - v for rr, cc in ring]
        if stats:
            stats.vertex_solves += 1
        res = _nearest(core, v)
        if res is None:
            continue
        nst, k, d = res
        if any(float(nst @ w) > 1e-12 for w in W):
            continue
        if d > D:
            continue
        grp = ("V", r, c)
        for t_i, nn in zip(inc, normals, strict=True):
            if float(np.linalg.norm(nn - nst)) < 1e-9:
                grp = ("P", t_i)
                break
        if stats:
            stats.convex_verts += 1
        out.append(Row(v.copy(), k, nst.copy(), d, ("V", r, c), group=grp))
    return out


def _nearest(core, v):
    """(n, k, d): the direction from v to the core's nearest point k and the distance d, or, with v inside the
    core, the direction that expels v fastest with d = -depth."""
    kc = _closest_point(core, v)
    if kc is not None:
        w = kc - v
        dist = float(np.linalg.norm(w))
        if dist > 1e-12:
            return w / dist, kc, dist
        return None
    if core.kind in ("box", "hull"):
        depth = core.foffsets - core.fnormals @ v
        i = int(np.argmin(depth))
        f = core.fnormals[i]
        return -f, v + depth[i] * f, -float(depth[i])
    if core.kind == "cylinder":
        w = v - core.c
        z = float(w @ core.axis)
        rad = w - z * core.axis
        lr = float(np.linalg.norm(rad))
        d_cap = core.h - abs(z)
        d_side = core.r - lr
        if d_cap <= d_side or lr < 1e-12:
            f = core.axis * (1.0 if z >= 0 else -1.0)
            return -f, v + d_cap * f, -d_cap
        f = rad / lr
        return -f, v + d_side * f, -d_side
    return None


def _closest_point(core, v):
    """Closest core point to v when separated (None if v is inside the core)."""
    if core.kind == "box":
        loc = core.R.T @ (v - core.c)
        cl = np.clip(loc, -core.half, core.half)
        if np.allclose(cl, loc):
            return None
        return core.c + core.R @ cl
    if core.kind == "cylinder":
        w = v - core.c
        z = float(np.clip(w @ core.axis, -core.h, core.h))
        rad = w - (w @ core.axis) * core.axis
        lr = np.linalg.norm(rad)
        if lr > core.r:
            rad = rad * (core.r / lr)
        p = core.c + z * core.axis + rad
        if np.linalg.norm(p - v) < 1e-14:
            return None
        return p
    if core.kind == "hull":
        return core.closest(v)
    return None


# ----------------------------------------------------------------------------------------------------------------
# finish
# ----------------------------------------------------------------------------------------------------------------


def finish(rows, *, stats=None, max_per_group=4, implied="lower_hull"):
    """Merge coincident rows, drop rows implied within their plane group, keep at most ``max_per_group`` per group.

    Args:
        rows: Rows of one pair in feature order.
        stats: Optional counters.
        max_per_group: Rows kept per plane group.
        implied: ``"lower_hull"`` (rows on or above the lower hull of the others' (tangent position, distance))
            or ``"hull3d"`` (the prototype: core point inside the others' 3-D hull and not deeper).
    """
    kept = []
    for ct in rows:
        if any(np.linalg.norm(o.q - ct.q) <= EPS_MERGE_P and np.linalg.norm(o.n - ct.n) <= EPS_MERGE_N for o in kept):
            if stats:
                stats.merged += 1
            continue
        kept.append(ct)
    groups = {}
    for ct in kept:
        groups.setdefault(plane_group(ct), []).append(ct)
    final = []
    for members in groups.values():
        group = _drop_implied(members, stats, implied)
        if len(group) > max_per_group:
            idx = reduce4([ct.k for ct in group], [ct.d for ct in group], group[0].n)[:max_per_group]
            if stats:
                stats.reduced += len(group) - len(idx)
            group = [group[i] for i in idx]
        final.extend(group)
    if stats:
        stats.emitted += len(final)
    return final


def plane_group(ct):
    """Rows on one terrain plane with one normal: (normal, plane offset) rounded."""
    return (tuple(np.round(ct.n, 7)), round(float(ct.n @ ct.q), 8))


def _drop_implied(g, stats, implied):
    if len(g) < 3:
        return g
    g = list(g)
    changed = True
    while changed and len(g) >= 3:
        changed = False
        K = np.array([ct.k for ct in g])
        d = np.array([ct.d for ct in g])
        for j in sorted(range(len(g)), key=lambda i: (-d[i], i)):
            others = [i for i in range(len(g)) if i != j]
            if implied == "hull3d":
                val = _convex_interp3(K[others], d[others], K[j])
                drop = val is not None and d[j] >= val - 1e-9
            else:
                val = lower_hull_value(K[others], d[others], K[j], g[0].n)
                drop = val is not None and d[j] >= val - IMPLIED_RESIDUAL
            if drop:
                if stats:
                    stats.implied += 1
                g.pop(j)
                changed = True
                break
    return g


def lower_hull_value(P, d, p, n, *, tol=1e-9):
    """Lowest convex interpolation of d at p's tangent position over the others' tangent positions (None outside)."""
    e1 = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    U = np.stack([P @ e1, P @ e2], 1)
    u = np.array([p @ e1, p @ e2])
    best = None
    m = len(U)
    for i in range(m):
        if np.linalg.norm(U[i] - u) <= tol:
            best = d[i] if best is None else min(best, d[i])
        for j in range(i + 1, m):
            a, b = U[i], U[j]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-20:
                continue
            s = float((u - a) @ ab) / L2
            perp = abs(ab[0] * (u - a)[1] - ab[1] * (u - a)[0]) / math.sqrt(L2)
            if -tol <= s <= 1 + tol and perp <= tol:
                v = d[i] + s * (d[j] - d[i])
                best = v if best is None else min(best, v)
            for k in range(j + 1, m):
                c = U[k]
                T = np.stack([b - a, c - a], 1)
                det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
                if abs(det) < 1e-18:
                    continue
                uv = np.linalg.solve(T, u - a)
                if uv[0] >= -tol and uv[1] >= -tol and uv[0] + uv[1] <= 1 + tol:
                    v = d[i] + uv[0] * (d[j] - d[i]) + uv[1] * (d[k] - d[i])
                    best = v if best is None else min(best, v)
    return best


def _convex_interp3(P, d, p, *, tol=1e-9, res=1e-7):
    """Lowest convex interpolation of d at p over segments or triangles of the 3-D points P that contain p."""
    best = None
    m = len(P)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = P[i], P[j]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-20:
                continue
            s = float((p - a) @ ab) / L2
            if -tol <= s <= 1 + tol and np.linalg.norm(a + s * ab - p) <= res:
                v = d[i] + s * (d[j] - d[i])
                best = v if best is None else min(best, v)
            for k in range(j + 1, m):
                c = P[k]
                T = np.stack([b - a, c - a], 1)
                uv, *_ = np.linalg.lstsq(T, p - a, rcond=None)
                if np.linalg.norm(T @ uv - (p - a)) > res:
                    continue
                if uv[0] >= -tol and uv[1] >= -tol and uv[0] + uv[1] <= 1 + tol:
                    v = d[i] + uv[0] * (d[j] - d[i]) + uv[1] * (d[k] - d[i])
                    best = v if best is None else min(best, v)
    return best


# ----------------------------------------------------------------------------------------------------------------
# concave anticipation and the pair entry point
# ----------------------------------------------------------------------------------------------------------------


def _closest_on_tri(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + d1 / (d1 - d3) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
    den = 1.0 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


def _concave_neighbours(hf, t):
    """Triangles across t's concave edges (the faces whose surface lies above t's extended plane there)."""
    out = []
    vid = hf.tri_vidx(t)
    _n, _a = hf.plane(t)
    r, c, _sub = hf.tri_rc(t)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            for s in (0, 1):
                if not hf.valid_cell(r + dr, c + dc):
                    continue
                u = hf.tri_id(r + dr, c + dc, s)
                if u == t:
                    continue
                sv = hf.tri_vidx(u)
                shared = set(vid) & set(sv)
                if len(shared) != 2:
                    continue
                e = (None, None, None, None, None, t, u, None, next(x for x in sv if x not in shared))
                if hf.edge_class(e) == "concave":
                    out.append(u)
    return out


def anticipate(hf, core, D, window, motion, *, stats=None):
    """C: face rows a moving core will need before the next collision, at concave creases only."""
    tris = window[0]
    out = []
    if core.kind == "sphere":
        pts = [("c", core.c)]
    elif core.kind == "capsule":
        pts = [("a", core.a), ("b", core.b)]
    else:
        pts = None
    for t in tris:
        n, a = hf.plane(t)
        ring = None
        if pts is not None:
            cands = [(tag, p, float(n @ (p - a))) for tag, p in pts]
        else:
            cands = [("k", k, h) for k, h in core.plane_manifold(n, a, D)]
        for tag, p, d in cands:
            if d > D or d < -D:
                continue
            if abs(d) > EPS_PRED and (d < 0.0) != _below_surface(hf, p):
                continue
            q = p - d * n
            s_now = hf.locate(q[0], q[1])
            if s_now is None or hf.same_cluster(t, s_now):
                continue
            p1 = p + motion(p)
            d1 = float(n @ (p1 - a))
            q1 = p1 - d1 * n
            s1 = hf.locate(q1[0], q1[1])
            if s1 is None or not hf.same_cluster(t, s1):
                continue
            if ring is None:
                ring = _concave_neighbours(hf, t)
            if s_now not in ring:
                continue
            va, vb, vc = hf.tri_verts(t)
            out.append(Row(_closest_on_tri(q, va, vb, vc), p.copy(), n.copy(), d, ("A", tag, t), group=("P", t)))
    return out


def pair_rows(hf: ReferenceHeightfield, core: Core, D: float, *, stats=None, motion=None, implied="lower_hull"):
    """The law's rows of one pair. ``D``: reach on the core distance (rounding radius + margins + gaps).

    ``motion`` optionally maps a core point to its predicted displacement over the collision interval (rule C).
    """
    lo, hi = core.aabb()
    box = hf.cells_in_box(lo[:2] - D, hi[:2] + D)
    if box is None:
        return []
    window = hf.window_features(box)
    if stats:
        stats.tris += len(window[0])
        stats.edges += len(window[1])
        stats.verts += len(window[2])
    if core.kind == "sphere":
        raw = sphere_law(hf, core, D, window, stats=stats)
    elif core.kind == "capsule":
        raw = capsule_law(hf, core, D, window, stats=stats)
    else:
        raw = solid_law(hf, core, D, window, stats=stats)
        if not raw:
            raw = _solid_recovery(hf, core)
    if motion is not None:
        raw = raw + anticipate(hf, core, D, window, motion, stats=stats)
    return finish(raw, stats=stats, implied=implied)

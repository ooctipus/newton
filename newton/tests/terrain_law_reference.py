# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Float64 reference of the terrain contact law, the oracle of its float32 kernels (numpy only).

The law emits the support set of one (heightfield, convex shape) pair directly, one row per local minimum of the
core-to-surface distance within the writer's reach plus the corners of flat support regions, each row owned by one
surface feature:

* P: point and segment cores (sphere centre, capsule ends and axis interior): face, crease-edge and vertex minima
  under one shared set of predicate values; G2 keeps one exit for a core point inside the solid, G3 the face
  vertically above it when no exit lies in the window. An axis passing below a ridge exits over it (the segment's
  minimum translation), and an axis inside the solid with no exit gets G3 at its deepest crossing under a crease
  (:func:`capsule_law`).
* F: solid cores against plane clusters: the analytic plane manifold of the core against the cluster plane,
  emitted once per cluster by the triangle holding the first supported point.
* E: solid cores against convex creases: the maximiser of the separation over the crease's normal arc, then the
  core's flat support feature (interior) or its reference feature (arc end, G1) crossed with the edge.
* V: solid cores against vertex features (vertices whose incident cluster normals span 3-D).
* R: solid cores below the surface with no row: up to four of their deepest boundary points (G3).
* The finish: coincident rows merged, rows implied by the others of their plane group dropped (the lower convex
  hull of tangent position and distance), at most four rows per plane group and, given a row limit, at most that
  many per pair (every group's deepest row first, then its extremes).
* The border: the heightfield is a solid block whose footprint has vertical walls from its border edges down
  ``WALL_DEPTH``. A border edge is a ridge between its one triangle and its wall, a border vertex's ring takes the wall
  below it, two walls meet at a vertical corner line below each corner of the footprint, and the walls and corner
  lines are one-sided (they bound the block for points beside it; a point inside the footprint below the surface exits
  through the top). A solid's face is clipped to the footprint: the core's part on the face's side of the clip plane
  through a border edge (containing the face's normal) gives the face its deepest points there (rule F), and rule R
  takes the core's boundary points inside the footprint.

This is a port of the design study's float64 prototype with three changes, each measured on the CPU:

* Plane clusters use robust planes. A cluster is grown over the flat edges as before, but a component whose
  vertices all lie within ``FLAT_EPS`` of the plane through three extreme vertices, or of a horizontal plane,
  each with its offset centred on the residual range, is one cluster with that plane; other components keep the
  candidate (those two or the plane of their lowest-id triangle) that claims the most triangles. The plane of a
  single 0.1 m triangle carries the float32 height decode noise (a tilt up to about 2e-6 rad), so far members of
  a large cluster left its 1 um band: a 12.7 m planar 12 deg down-slope split into 4,072 clusters, and cropping
  the recorded suite terrain by three rows and seven columns split its flat border into over a million clusters
  (1,562,181 in all, against 566,737 uncropped); with robust planes both crops have 539,984 +- 2.
* Edge classes follow the cluster ids alone: two clusters always meet at a crease.
* Rows of a plane group are implied when they lie on or above the lower convex hull of (tangent position,
  distance) of the others, within ``IMPLIED_RESIDUAL``; the prototype also required them to lie inside the 3-D
  hull of the others' core points.

Heights are decoded as the narrow phase decodes them (float32 ``min_z + e * (max_z - min_z)``), then promoted to
float64. Vertex (r, c) lies at x = -hx + c * dx, y = -hy + r * dy; cell (r, c) splits along p00-p11 into
sub 0 = (p00, p10, p11) and sub 1 = (p00, p11, p01); triangle id = 2 * (r * cols + c) + sub.
"""

from __future__ import annotations

import functools
import math

import numpy as np

FLAT_EPS = 1.0e-6
"""Plane-cluster membership tolerance [m] (absolute; it absorbs the ~0.24 um float32 height decode)."""
CLUSTER_ROUNDS = 8
"""Rounds of the cluster build; triangles still unassigned afterwards are clusters of their own."""
NORMAL_SAME = 1.0e-5
"""Unit normals closer than this are one direction (vertex features)."""
VERTEX_SPAN = 1.0e-10
"""A vertex is a feature when some three incident cluster normals have a triple product above this.

The prototype's value, ``NORMAL_SAME`` squared. At 1e-6 (the float32 resolution of the plain triple product), 7,685
more vertices of the recorded suite terrain were not features, 1,800 of them convex, and solids whose support sat
over one of those lost it in 84-87% of poses: no face or edge claims a support whose normal lies strictly inside a
vertex's normal cone.
"""
IMPLIED_RESIDUAL = 1.0e-6
"""A row no deeper than this below the lower hull of its plane group's other rows is implied [m]."""
GROUP_NORMAL = 1.0e-5
"""Largest normal difference within a plane group (with ``GROUP_OFFSET``)."""
GROUP_OFFSET = 1.0e-6
"""Largest distance of a row's terrain point from its plane group's plane [m].

Planes closer than the implication residual are one plane at the finish's resolution; the float32 finish uses
four predicate bands, 0.4-1.5 um on windows of 0.1-0.4 m.
"""

EPS_PRED = 1.0e-12
"""Predicate band [m]: a value within it counts as on the boundary."""
EPS_MERGE_P = 1.0e-7
EPS_MERGE_N = 1.0e-6
EPS_THETA = 1.0e-5
"""Arc-end tolerance of rule E [rad] (DESIGN 1.4): a ``theta*`` within it of an arc's end takes the end's reference
feature (G1)."""
CYLINDER_FLAT_MODE_COS = math.cos(math.radians(22.5))
TIE = 1.0e-6
"""Tie band of the plane manifolds' and reference features' discrete choices (DESIGN 1.4): of candidates within
``TIE`` times the core's bounding radius in height (or distance; area: its square), or within ``TIE`` in a normal's
score, the first in index order wins. Two of them are ties by construction: a rolling cylinder's two cap-rim side
points lie at one height, and a lying cylinder's caps face the plane equally."""
HULL_FACE_TOL = 1.0e-6
"""A hull's vertices within this of a face's plane lie on that face [m] (hulls under 1 m; local frame).

Mesh vertices are float32: the vertices of one planar face lie within about 1e-8 m of its plane, and a tighter test
splits the face into slivers (a sole's flat bottom into triangles).
"""
LIFT_SLACK = 1.0e-8
"""How far a line may miss a core and still graze it [m] (rule E's lifts).

An interior ``theta*`` puts the crease line exactly through the core's support point; at a support edge sharper than
a right angle (a cone's base rim or apex, a sharp hull vertex) the line along ``n(theta*)`` only touches the core
there, and the golden-section ``theta*`` places it within about 1e-9 m. A tighter test drops that support.
"""

RING = ((0, 1), (1, 1), (1, 0), (0, -1), (-1, -1), (-1, 0))
"""One-ring neighbour offsets (dr, dc) of a vertex, counter-clockwise."""
WALL_DEPTH = 1.0
"""Depth of the side walls below the border edges [m] (the float32 law's ``TERRAIN_WALL_DEPTH``)."""
WALL_NORMALS = (
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
)
"""Outward normals of the walls ``y = -hy``, ``y = hy``, ``x = -hx`` and ``x = hx`` (sides 0 to 3)."""


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
    candidate planes: the extreme plane (through its lowest-id vertex ``a``, the vertex ``b`` farthest from ``a`` and
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

    def vertex_span(self, r, c):
        """The largest triple product of three distinct incident cluster normals of vertex (r, c), at the border its
        triangles' and its walls'."""
        if self.ring(r, c) is None:
            normals = [self.plane(t)[0] for t in self.incident_in_grid(r, c)]
            normals += [WALL_NORMALS[side] for side in self.vertex_sides(r, c)]
            return largest_triple_product(normals)
        return largest_triple_product([self.plane(t)[0] for t in self.incident_triangles(r, c)])

    def vertex_feature(self, r, c):
        """Whether vertex (r, c) is a feature of its own: its incident cluster normals span 3-D."""
        return self.vertex_span(r, c) > VERTEX_SPAN

    # ---- the border ------------------------------------------------------------------------------------------
    def border_side(self, kind, r, c):
        """The wall (0 to 3, :data:`WALL_NORMALS`) of window edge kind ``"H"`` (x-edge) or ``"V"`` (y-edge) at vertex
        (r, c) on the border, else None."""
        if kind == "H" and 0 <= c < self.cols:
            if r == 0:
                return 0
            if r == self.rows:
                return 1
        if kind == "V" and 0 <= r < self.rows:
            if c == 0:
                return 2
            if c == self.cols:
                return 3
        return None

    def vertex_sides(self, r, c):
        """The walls through vertex (r, c): none inside, one on the border, two at a corner (the y-wall first)."""
        out = []
        if r == 0:
            out.append(0)
        elif r == self.rows:
            out.append(1)
        if c == 0:
            out.append(2)
        elif c == self.cols:
            out.append(3)
        return out

    def border_ring(self, r, c):
        """The one-ring vertices of vertex (r, c) in the grid."""
        return [(r + dr, c + dc) for dr, dc in RING if self.valid_vertex(r + dr, c + dc)]

    def incident_in_grid(self, r, c):
        """The triangles around vertex (r, c) in the grid, in the one-ring's order."""
        cells = ((r, c, 0), (r, c, 1), (r, c - 1, 0), (r - 1, c - 1, 1), (r - 1, c - 1, 0), (r - 1, c, 1))
        return [self.tri_id(rr, cc, sub) for rr, cc, sub in cells if self.valid_cell(rr, cc)]

    @staticmethod
    def wall_value(k, v, side):
        """(k - v) . n of wall ``side``: k's distance beyond the wall through v."""
        return float((np.asarray(k) - np.asarray(v)) @ WALL_NORMALS[side])

    @staticmethod
    def wall_along(k, v, side):
        """(k - v) . t, t the wall's axis toward higher vertex indices (x for walls 0 and 1, y for 2 and 3)."""
        return float(k[0] - v[0]) if side < 2 else float(k[1] - v[1])

    @staticmethod
    def wall_frame(v0, v1, side):
        """The in-wall unit normal of border edge v0 v1 pointing up."""
        m = np.cross(v1 - v0, WALL_NORMALS[side])
        if m[2] < 0.0:
            m = -m
        return m / np.linalg.norm(m)

    def wall_holds(self, side, v0, v1, k, *, tol=0.0):
        """Whether the wall below border edge v0 v1 holds k's projection: below the edge, between its vertices' vertical
        lines and above the base (``WALL_DEPTH`` below the edge)."""
        m = self.wall_frame(v0, v1, side)
        s = float((k - v0) @ m)
        if s > tol or s < -WALL_DEPTH * m[2] - tol:
            return False
        return self.wall_along(k, v0, side) >= -tol and self.wall_along(k, v1, side) <= tol


def largest_triple_product(normals) -> float:
    """The largest |det| of three distinct unit normals (closer than ``NORMAL_SAME`` is one direction), else 0."""
    uniq = []
    for nn in normals:
        if all(np.linalg.norm(nn - u) > NORMAL_SAME for u in uniq):
            uniq.append(nn)
    best = 0.0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            for k in range(j + 1, len(uniq)):
                best = max(best, abs(float(np.linalg.det(np.stack([uniq[i], uniq[j], uniq[k]])))))
    return best


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


def first_within(values, band, *, largest=False):
    """The first index whose value lies within ``band`` of the smallest (or largest) value."""
    v = np.asarray(values, np.float64)
    if largest:
        return int(np.flatnonzero(v >= v.max() - band)[0])
    return int(np.flatnonzero(v <= v.min() + band)[0])


def reduce4(points, heights, normal, *, tie=0.0):
    """Indices of at most 4 points: the deepest, the farthest from it, the largest triangle, then the point adding
    the most area outside that triangle (areas in the plane normal to ``normal``). The first in index order within
    ``tie`` (a length; areas within its square times... the largest area's scale) of the best wins."""
    n = len(points)
    if n <= 4:
        return list(range(n))
    P = np.asarray(points, np.float64)
    P = P - np.outer(P @ normal, normal)
    h = np.asarray(heights, np.float64)
    i0 = first_within(h, tie)
    i1 = first_within(np.linalg.norm(P - P[i0], axis=1), tie, largest=True)
    a2 = np.abs(np.cross(P[i1] - P[i0], P - P[i0]) @ normal)
    scale = max(float(np.linalg.norm(P[i1] - P[i0])), 1.0e-12)
    i2 = first_within(a2, tie * scale, largest=True)
    tri = (i0, i1, i2)
    gains = np.zeros(n)
    for j in range(n):
        if j in tri:
            continue
        gain = 0.0
        for a, b, c in ((i0, i1, i2), (i1, i2, i0), (i2, i0, i1)):
            side_c = np.cross(P[b] - P[a], P[c] - P[a]) @ normal
            side_j = np.cross(P[b] - P[a], P[j] - P[a]) @ normal
            if side_c * side_j < 0:
                gain = max(gain, abs(side_j))
        gains[j] = gain
    i3 = None
    if gains.max() > 1e-15:
        i3 = int(np.flatnonzero(gains >= max(gains.max() - tie * scale, 1e-15 * (1 + 1e-12)))[0])
    out = [i0, i1, i2] + ([i3] if i3 is not None else [])
    return list(dict.fromkeys(out))


def _order_by_depth(indices, h, band):
    """Indices deepest first: an index passes an earlier one only when deeper by more than ``band``."""
    out = []
    for i in sorted(indices):
        pos = len(out)
        while pos > 0 and h[out[pos - 1]] > h[i] + band:
            pos -= 1
        out.insert(pos, i)
    return out


def convex_hull_faces(verts, *, tol: float = 1.0e-9):
    """Faces of the convex hull of ``verts``: vertex index lists and outward unit normals (brute force, small sets).

    A face is a plane through three vertices with every vertex on one side (within ``tol`` times the vertices' extent,
    at least 1 m); its vertices are those within that of the plane, and planes with the same vertices are one face.
    """
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
                key = tuple(int(m) for m in np.flatnonzero(np.abs(s) <= tol * scale))
                if key not in faces:
                    faces[key] = (normal, list(key))
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
                if num < -LIFT_SLACK:
                    return None
                continue
            t = num / denom
            if denom > 0:
                t_hi = min(t_hi, t)
            else:
                t_lo = max(t_lo, t)
        if t_lo > t_hi + LIFT_SLACK or not np.isfinite(t_lo):
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
            if float(ab @ ab) == 0.0:
                # a zero-thickness box's coincident corners
                continue
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
        """Newton's plane-box manifold: up to 4 deepest corners within reach, deepest first; a corner passes a kept one
        only when deeper by more than the tie band (else corner order holds). Each corner once: along a zero half
        extent (a finite plane's thickness) only the corner on its lower side."""
        band = TIE * float(np.linalg.norm(self.half))
        kept = []
        for i in range(8):
            if any(self.half[axis] == 0.0 and (i >> axis) & 1 for axis in range(3)):
                continue
            h = float(n @ (self.verts[i] - a))
            if h > D:
                continue
            pos = len(kept)
            while pos > 0 and kept[pos - 1][0] > h + band:
                pos -= 1
            kept.insert(pos, (h, i))
            del kept[4:]
        return [(self.verts[i].copy(), h) for h, i in kept]


@functools.lru_cache(maxsize=64)
def _local_hull_faces(data: bytes, shape: tuple):
    """:func:`convex_hull_faces` of local hull vertices (memoized: poses of one hull share them)."""
    return convex_hull_faces(np.frombuffer(data, np.float64).reshape(shape), tol=HULL_FACE_TOL)


class Hull(Polytope):
    kind = "hull"

    def __init__(self, c, rot, verts_local):
        self.c = np.asarray(c, np.float64)
        self.R = np.asarray(rot, np.float64)
        local = np.asarray(verts_local, np.float64)
        # Faces from the local vertices: the decision does not depend on where the hull is.
        faces, normals = _local_hull_faces(local.tobytes(), local.shape)
        super().__init__(self.c + local @ self.R.T, faces, [self.R @ nn for nn in normals])

    def radius(self):
        """The largest distance of a vertex from the centre (the tie band's scale)."""
        return float(np.linalg.norm(self.verts - self.c, axis=1).max())

    def plane_manifold(self, n, a, D):
        """The reference face (DESIGN 1.2: the face through the deepest vertex most opposed to n, the face G1's
        crossings use) and the deepest vertex, within reach, reduced to 4 by area, deepest first (ties within the band
        in index order)."""
        band = TIE * self.radius()
        h = (self.verts - a) @ n
        deepest = first_within(h, band)
        through = [q for q, f in enumerate(self.faces) if deepest in f]
        cand = {deepest}
        if through:
            cand.update(self.faces[through[first_within(self.fnormals[through] @ n, TIE)]])
        idx = sorted(i for i in cand if h[i] <= D)
        if not idx:
            return []
        keep = reduce4(self.verts[idx], h[idx], n, tie=band)
        sel = _order_by_depth([idx[k] for k in keep], h, band)
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
        if dot_na > TIE:
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
            p = pm if n @ (pm - a) < n @ (pp - a) - TIE * self.r else pp
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
            if abs(dw) > self.h + LIFT_SLACK:
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
            if C > 2.0 * self.r * LIFT_SLACK:
                return None
        else:
            # disc / (4 A) = r^2 - rho^2 for a line at distance rho from the axis: a graze within the slack counts
            disc = B * B - 4 * A * C
            if disc < 0:
                if disc / (4.0 * A) < -2.0 * self.r * LIFT_SLACK:
                    return None
                disc = 0.0
            sq = math.sqrt(disc)
            t1, t2 = (-B - sq) / (2 * A), (-B + sq) / (2 * A)
            t_lo, t_hi = max(t_lo, t1), min(t_hi, t2)
        if t_lo > t_hi + LIFT_SLACK or not np.isfinite(t_lo):
            return None
        return u + t_lo * n, t_lo


class Barrel(Cylinder):
    """A barrel cylinder (Newton's ``scale[2]``): caps of radius ``r`` at ``+-h`` along the axis and a side of
    revolution of a circular arc of radius ``R >= h`` through the caps' rims (:meth:`rho`). Its plane manifold is its
    lowest point (Newton's support point) with the cylinder's tripod within 22.5 deg of upright; its curved side has no
    generator."""

    kind = "barrel"

    def __init__(self, c, axis, radius, half_height, barrel):
        super().__init__(c, axis, radius, half_height)
        self.R = float(barrel)

    def rho(self, z):
        """The side's radius at axial height ``z`` (``|z| <= h``), in Newton's support function's form."""
        denom = math.sqrt(max(self.R**2 - z * z, 0.0)) + math.sqrt(max(self.R**2 - self.h**2, 0.0))
        return self.r if denom <= 0.0 else self.r + (self.h**2 - z * z) / denom

    def lowest(self, n):
        ax = self.axis
        z = min(max(-self.R * float(n @ ax), -self.h), self.h)
        perp = -n + ax * float(n @ ax)
        ln2 = float(perp @ perp)
        if ln2 <= 1e-10:
            return self.c + z * ax
        return self.c + z * ax + self.rho(z) * perp / math.sqrt(ln2)

    def aabb(self):
        lo, hi = np.zeros(3), np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = 1.0
            lo[i] = self.lowest(e)[i]
            hi[i] = self.lowest(-e)[i]
        return lo, hi

    def plane_manifold(self, n, a, D):
        axis = self.axis.copy()
        dot_na = float(n @ axis)
        if dot_na > TIE:
            axis = -axis
            dot_na = -dot_na
        cap = self.c + axis * self.h
        pts = [self.lowest(n)]
        if -dot_na >= CYLINDER_FLAT_MODE_COS:
            ref = np.array([1.0, 0.0, 0.0])
            if abs(axis @ ref) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            perp_fixed = ref - axis * (axis @ ref)
            perp_fixed /= np.linalg.norm(perp_fixed)
            u = perp_fixed * self.r
            v = np.cross(axis, perp_fixed) * self.r
            merge = (0.01 * max(self.r, self.h)) ** 2
            for p in (cap + u, cap - 0.5 * u + 0.8660254 * v, cap - 0.5 * u - 0.8660254 * v):
                if len(pts) < 4 and np.sum((p - pts[0]) ** 2) > merge:
                    pts.append(p)
        out = []
        for p in pts:
            hh = float(n @ (p - a))
            if hh <= D:
                out.append((p, hh))
        return out

    def excess(self, x):
        """How far ``x`` lies outside the side at its axial height (clamped to the caps): convex along any line."""
        w = x - self.c
        z = float(w @ self.axis)
        return float(np.linalg.norm(w - z * self.axis)) - self.rho(min(max(z, -self.h), self.h))

    def lift(self, u, n):
        """Entry of the line u + t n into the barrel (smallest t): the caps' slab, then the side's excess along the
        line, convex there: a golden section to its least value, a bisection to its first zero before it."""
        ax = self.axis
        w = u - self.c
        dn, dw = float(n @ ax), float(w @ ax)
        if abs(dn) < 1e-14:
            if abs(dw) > self.h + LIFT_SLACK:
                return None
            # parallel to the caps: the side's circle at the line's height
            return Cylinder(self.c, ax, self.rho(min(max(dw, -self.h), self.h)), self.h).lift(u, n)
        t1, t2 = (-self.h - dw) / dn, (self.h - dw) / dn
        t_lo, t_hi = min(t1, t2), max(t1, t2)
        if self.excess(u + t_lo * n) > 0.0:
            g = (math.sqrt(5.0) - 1.0) / 2.0
            a, b = t_lo, t_hi
            for _ in range(120):
                c, d = b - g * (b - a), a + g * (b - a)
                if self.excess(u + c * n) <= self.excess(u + d * n):
                    b = d
                else:
                    a = c
            least = 0.5 * (a + b)
            if self.excess(u + least * n) > LIFT_SLACK:
                return None
            lo, hi = t_lo, least
            for _ in range(120):
                mid = 0.5 * (lo + hi)
                if self.excess(u + mid * n) > 0.0:
                    lo = mid
                else:
                    hi = mid
            t_lo = hi
        return u + t_lo * n, t_lo

    def contains(self, x):
        w = x - self.c
        z = float(w @ self.axis)
        return abs(z) <= self.h and float(np.linalg.norm(w - z * self.axis)) <= self.rho(z)

    def nearest(self, v):
        """:func:`_nearest`'s ``(n, k, d)``, in the meridian half-plane of ``v`` (radial ``x``, axial ``z``): outside,
        the nearest of the near cap's projection and the side arc's (the circle of radius ``R`` about ``(x0, 0)``,
        ``x0 = r - sqrt(R^2 - h^2)``, where it lies on the side); inside, the nearer of the cap and the arc."""
        ax = self.axis
        w = v - self.c
        z = float(w @ ax)
        rad = w - z * ax
        x = float(np.linalg.norm(rad))
        radial = rad / x if x > 1e-12 else _disk_basis(ax)[0]
        x0 = self.r - math.sqrt(max(self.R**2 - self.h**2, 0.0))
        ox = x - x0
        od = math.hypot(ox, z)
        arc = od > 0.0 and ox >= 0.0 and abs(self.R * z) <= self.h * od
        if abs(z) <= self.h and x <= self.rho(z):
            d_cap = self.h - abs(z)
            if arc and self.R - od < d_cap:
                f = (ox * radial + z * ax) / od
                d = self.R - od
                return -f, v + d * f, -d
            f = ax * (1.0 if z >= 0.0 else -1.0)
            return -f, v + d_cap * f, -d_cap
        sz = 1.0 if z >= 0.0 else -1.0
        cands = [(min(x, self.r), sz * self.h)]
        if arc:
            cands.append((x0 + self.R * ox / od, self.R * z / od))
        bx, bz = min(cands, key=lambda p: (x - p[0]) ** 2 + (z - p[1]) ** 2)
        k = self.c + bz * ax + bx * radial
        dist = float(np.linalg.norm(k - v))
        if dist <= 0.0:
            return None
        return (k - v) / dist, k, dist


def _disk_basis(axis):
    """Two unit vectors spanning the plane normal to ``axis`` (x unless the axis is within 26 deg of x, then y)."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - axis * float(axis @ ref)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


class Cone(Core):
    """A cone: apex at ``c + h axis``, base disk of radius ``r`` centred at ``c - h axis`` (Newton's convention).

    Its plane manifold mirrors the cylinder's with the base as the only cap: within 22.5 deg of base-down the deepest
    base rim point plus the fixed base tripod; otherwise the two ends of the lowest generator (the base rim point
    under the plane's normal and the apex), the lower one first.
    """

    kind = "cone"

    def __init__(self, c, axis, radius, half_height):
        self.c = np.asarray(c, np.float64)
        self.axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
        self.r = float(radius)
        self.h = float(half_height)
        self.apex = self.c + self.h * self.axis
        self.base = self.c - self.h * self.axis

    def _perp(self, n):
        """The base-plane direction of the rim point lowest along ``n`` (a fixed one when the base is level)."""
        ax = self.axis
        perp = -n + ax * float(n @ ax)
        ln = float(perp @ perp)
        if ln > 1e-10:
            return perp / math.sqrt(ln)
        return _disk_basis(-ax)[0]

    def generator(self, n):
        """The base rim point under ``n`` and the apex: the lowest generator's ends."""
        return self.base + self.r * self._perp(n), self.apex.copy()

    def _band(self):
        return TIE * math.hypot(self.r, self.h)

    def lowest(self, n):
        g, p = self.generator(n)
        return p if float(n @ p) < float(n @ g) - self._band() else g

    def aabb(self):
        ext = self.r * np.sqrt(np.maximum(0.0, 1.0 - self.axis * self.axis))
        return np.minimum(self.base - ext, self.apex), np.maximum(self.base + ext, self.apex)

    def plane_manifold(self, n, a, D):
        ax = self.axis
        g, p = self.generator(n)
        merge = (0.01 * max(self.r, self.h)) ** 2
        if float(n @ ax) >= CYLINDER_FLAT_MODE_COS:
            down = -ax
            u, v = _disk_basis(down)
            u, v = u * self.r, v * self.r
            pts = [g]
            for q in (self.base + u, self.base - 0.5 * u + 0.8660254 * v, self.base - 0.5 * u - 0.8660254 * v):
                if len(pts) < 4 and np.sum((q - pts[0]) ** 2) > merge:
                    pts.append(q)
        else:
            first, second = (p, g) if float(n @ p) < float(n @ g) - self._band() else (g, p)
            pts = [first]
            if np.sum((second - first) ** 2) > merge:
                pts.append(second)
        out = []
        for q in pts:
            hh = float(n @ (q - a))
            if hh <= D:
                out.append((q.copy(), hh))
        return out

    def lift(self, u, n):
        """Entry of the line u + t n into the solid cone (smallest t)."""
        ax = self.axis
        w = u - self.apex
        # z(t) = a . (x(t) - apex) in [-2h, 0]; |P x|^2 <= k^2 z^2 with k = r / (2 h)
        k2 = (self.r / (2.0 * self.h)) ** 2
        dz, wz = float(n @ ax), float(w @ ax)
        t_lo, t_hi = -np.inf, np.inf
        if abs(dz) < 1e-14:
            if wz > LIFT_SLACK or wz < -2.0 * self.h - LIFT_SLACK:
                return None
        else:
            t1, t2 = (-2.0 * self.h - wz) / dz, (0.0 - wz) / dz
            t_lo, t_hi = max(t_lo, min(t1, t2)), min(t_hi, max(t1, t2))
        wp_ = w - wz * ax
        np_ = n - dz * ax
        A = float(np_ @ np_) - k2 * dz * dz
        B = 2.0 * (float(wp_ @ np_) - k2 * wz * dz)
        C = float(wp_ @ wp_) - k2 * wz * wz
        # The lower nappe: within the slab z <= 0 the quadratic's solution set is one interval.
        if abs(A) < 1e-14:
            if abs(B) < 1e-14:
                if C > 2.0 * self.r * LIFT_SLACK:
                    return None
            elif B > 0.0:
                t_hi = min(t_hi, -C / B)
            else:
                t_lo = max(t_lo, -C / B)
        else:
            disc = B * B - 4.0 * A * C
            if A > 0.0 and disc < 0.0 and disc / (4.0 * A) >= -2.0 * self.r * LIFT_SLACK:
                disc = 0.0
            if disc < 0.0:
                if A > 0.0:
                    return None
            else:
                sq = math.sqrt(disc)
                r1, r2 = sorted(((-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)))
                if A > 0.0:
                    t_lo, t_hi = max(t_lo, r1), min(t_hi, r2)
                elif dz > 0.0:
                    # A line steeper than the wall leaves the lower nappe at r1 (and enters the upper one at r2).
                    t_hi = min(t_hi, r1)
                else:
                    t_lo = max(t_lo, r2)
        if t_lo > t_hi + LIFT_SLACK or not np.isfinite(t_lo):
            return None
        return u + t_lo * n, t_lo

    def contains(self, x):
        z = float((x - self.apex) @ self.axis)
        rho = float(np.linalg.norm((x - self.apex) - z * self.axis))
        return -2.0 * self.h <= z <= 0.0 and rho <= self.r * (-z) / (2.0 * self.h)


class Ellipsoid(Core):
    """An ellipsoid of semi-axes ``s`` (local x, y, z) centred at ``c`` with rotation ``rot``; its plane manifold is
    its support point (Newton's analytic plane query)."""

    kind = "ellipsoid"

    def __init__(self, c, rot, semi):
        self.c = np.asarray(c, np.float64)
        self.R = np.asarray(rot, np.float64)
        self.s = np.asarray(semi, np.float64)

    def lowest(self, n):
        m = self.R.T @ n
        y = -(self.s**2) * m / math.sqrt(float(np.sum((self.s * m) ** 2)))
        return self.c + self.R @ y

    def aabb(self):
        ext = np.sqrt(np.sum((self.R * self.s[None, :]) ** 2, axis=1))
        return self.c - ext, self.c + ext

    def plane_manifold(self, n, a, D):
        k = self.lowest(n)
        h = float(n @ (k - a))
        return [(k, h)] if h <= D else []

    def lift(self, u, n):
        y0 = (self.R.T @ (u - self.c)) / self.s
        dy = (self.R.T @ n) / self.s
        A = float(dy @ dy)
        B = 2.0 * float(y0 @ dy)
        C = float(y0 @ y0) - 1.0
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            if disc / (4.0 * A) < -2.0 * LIFT_SLACK / float(self.s.max()):
                return None
            disc = 0.0
        t = (-B - math.sqrt(disc)) / (2.0 * A)
        return u + t * n, t

    def contains(self, x):
        y = (self.R.T @ (x - self.c)) / self.s
        return float(y @ y) <= 1.0

    def boundary_nearest(self, x):
        """The boundary point nearest ``x`` (inside or outside), from the secular equation (float64 bisection).

        The stationary points are ``x_i = s_i^2 y_i / (s_i^2 + t)`` with ``sum((s_i y_i / (s_i^2 + t))^2) = 1``; the
        nearest one has ``t > 0`` outside and ``t`` in ``(-min s^2, 0]`` inside. An interior point on the smallest
        axis's plane may instead be nearest to the points ``t = -min s^2`` off that plane.
        """
        y = self.R.T @ (x - self.c)
        s2 = self.s**2

        def residual(t):
            return float(np.sum((self.s * y / (t + s2)) ** 2)) - 1.0

        inside = float(np.sum((y / self.s) ** 2)) <= 1.0
        if inside:
            a, b = -float(s2.min()), 0.0
            a = a + 1e-15 * max(1.0, abs(a))
        else:
            a, b = 0.0, float(np.linalg.norm(y) * self.s.max()) + 1e-12
        candidates = []
        if residual(a) >= 0.0:
            for _ in range(200):
                mid = 0.5 * (a + b)
                if residual(mid) > 0.0:
                    a = mid
                else:
                    b = mid
            t = 0.5 * (a + b)
            candidates.append(s2 * y / (t + s2))
        if inside:
            k = int(np.argmin(s2))
            others = [j for j in range(3) if j != k]
            if all(abs(s2[j] - s2[k]) > 1e-15 for j in others):
                xs = np.zeros(3)
                for j in others:
                    xs[j] = s2[j] * y[j] / (s2[j] - s2[k])
                rest = 1.0 - sum((xs[j] / self.s[j]) ** 2 for j in others)
                if rest >= 0.0:
                    for sign in (1.0, -1.0):
                        cand = xs.copy()
                        cand[k] = sign * self.s[k] * math.sqrt(rest)
                        candidates.append(cand)
        if not candidates:
            # Equal smallest axes with the point on their plane: the minimum translation over dense directions.
            i = np.arange(4096) + 0.5
            phi = np.arccos(1.0 - 2.0 * i / 4096)
            theta = math.pi * (1.0 + 5.0**0.5) * i
            d = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
            support = np.sqrt(np.sum((d * self.s) ** 2, axis=1))
            j = int(np.argmin(support - d @ y))
            candidates.append(s2 * d[j] / support[j])
        best = min(candidates, key=lambda cand: float(np.linalg.norm(cand - y)))
        return self.c + self.R @ best

    def outward_normal(self, p):
        g = (self.R.T @ (p - self.c)) / self.s**2
        g = self.R @ g
        return g / np.linalg.norm(g)


def make_core(kind: str, dims: dict, pose_local, *, verts=None) -> Core:
    """A core in the heightfield-local frame: ``pose_local`` = (px, py, pz, qx, qy, qz, qw).

    ``dims``: sphere ``radius``; capsule, cylinder and cone ``radius`` and ``half_height``; barrel ``radius``,
    ``half_height`` and ``barrel`` (its side's profile radius); box ``half``; plane ``size`` (width and length: a box of
    half its width and length and no thickness); ellipsoid ``semi`` (semi-axes); hull ``verts`` (its local vertices,
    scale applied).
    """
    p = np.asarray(pose_local[:3], np.float64)
    q = np.asarray(pose_local[3:7], np.float64)
    # A float32 pose's quaternion is unit only to about 6e-8; a polytope's face normals must stay its corners' planes.
    R = quat_matrix(q / np.linalg.norm(q))
    z = R[:, 2]
    if kind == "sphere":
        return Sphere(p, dims["radius"])
    if kind == "capsule":
        return Capsule(p, z, dims["half_height"], dims["radius"])
    if kind == "box":
        return Box(p, R, dims["half"])
    if kind == "cylinder":
        return Cylinder(p, z, dims["radius"], dims["half_height"])
    if kind == "barrel":
        return Barrel(p, z, dims["radius"], dims["half_height"], dims["barrel"])
    if kind == "plane":
        return Box(p, R, (0.5 * dims["size"][0], 0.5 * dims["size"][1], 0.0))
    if kind == "cone":
        return Cone(p, z, dims["radius"], dims["half_height"])
    if kind == "ellipsoid":
        return Ellipsoid(p, R, dims["semi"])
    if kind == "hull":
        return Hull(p, R, dims["verts"] if verts is None else verts)
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
        self.limited = 0
        """Rows dropped by the per-pair row limit."""


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
                if not below:
                    row = self.border_edge(e, p, D, tag)
                    if row is not None:
                        out.append(row)
                continue
            key = tuple(sorted((i0, i1)))
            if not below and self.cls.get(key) == "concave":
                # A valley's minima outside the solid are its faces'; beyond both faces lies below its line, where
                # the normal from the edge would leave the faces' cone (a ghost pushing the core down).
                continue
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
                if not below:
                    row = self.border_vertex(r, c, p, D, tag)
                    if row is not None:
                        out.append(row)
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

    def border_edge(self, e, p, D, tag):
        """A border edge's claim on point p outside the solid: the edge (strictly beyond its triangle and its wall at
        the edge, inside its slab) above the wall's edge line, the wall below it (p beside the wall, its projection
        held), or None."""
        hf = self.hf
        kind, r, c, i0, i1, ta, tb, _oa, _ob = e
        side = hf.border_side(kind, r, c)
        if side is None:
            return None
        t = ta if ta is not None else tb
        v0, v1 = hf.V(*i0), hf.V(*i1)
        n_wall = WALL_NORMALS[side]
        s_wall = float((p - v0) @ hf.wall_frame(v0, v1, side))
        if s_wall > 0.0:
            s_top = self.face_pred(t, tuple(sorted((i0, i1))), p)
            u = v1 - v0
            w0 = float((p - v0) @ u)
            w1 = float((p - v1) @ (-u))
            if not (s_top > EPS_PRED and s_wall > EPS_PRED and w0 > EPS_PRED and w1 > EPS_PRED):
                return None
            q = v0 + (w0 / float(u @ u)) * u
            dist = float(np.linalg.norm(p - q))
            if dist < 1e-12:
                n = hf.plane(t)[0] + n_wall
                n, d = n / np.linalg.norm(n), 0.0
            else:
                n, d = (p - q) / dist, dist
            if d > D:
                return None
            return Row(q, p.copy(), n, d, (tag, "E", kind, r, c))
        d = hf.wall_value(p, v0, side)
        if d < -EPS_PRED or d > D or not hf.wall_holds(side, v0, v1, p, tol=EPS_PRED):
            return None
        return Row(p - d * n_wall, p.copy(), n_wall.copy(), d, (tag, "W", kind, r, c))

    def border_vertex(self, r, c, p, D, tag):
        """A border vertex's claim on point p outside the solid: its Voronoi region (its ring's vertices in the grid
        and the wall below it: p not below the vertex), or below a corner of the footprint the corner line (p beyond
        both walls, within ``WALL_DEPTH`` of the corner); None otherwise."""
        hf = self.hf
        v = hf.V(r, c)
        if v[2] - p[2] > 0.0:
            sides = hf.vertex_sides(r, c)
            if len(sides) < 2 or v[2] - p[2] <= EPS_PRED or v[2] - p[2] > WALL_DEPTH:
                return None
            if any(hf.wall_value(p, v, side) <= EPS_PRED for side in sides):
                return None
            q = np.array([v[0], v[1], p[2]])
            w = p - q
            dist = float(np.linalg.norm(w))
            n = w / dist if dist >= 1e-12 else WALL_NORMALS[sides[0]] + WALL_NORMALS[sides[1]]
            n = n / np.linalg.norm(n)
            if dist > D:
                return None
            return Row(q, p.copy(), n, dist, (tag, "L", r, c))
        w = p - v
        if any(float(w @ (hf.V(rr, cc) - v)) > EPS_PRED for rr, cc in hf.border_ring(r, c)):
            return None
        dist = float(np.linalg.norm(w))
        if dist < 1e-12 or dist > D:
            return None
        return Row(v, p.copy(), w / dist, dist, (tag, "V", r, c))


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


def solid_recovery_points(core):
    """The boundary points G3 tries for a solid core: polytope vertices; a cylinder's 16 rim points per cap (a barrel's
    then 16 around its side's middle); a cone's 16 base rim points and its apex; an ellipsoid's 14 points on its local
    axes and diagonals."""
    if hasattr(core, "verts"):
        return core.verts
    th = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    if core.kind == "ellipsoid":
        local = [np.array([sx, 0.0, 0.0]) for sx in (1.0, -1.0)]
        local += [np.array([0.0, sy, 0.0]) for sy in (1.0, -1.0)]
        local += [np.array([0.0, 0.0, sz]) for sz in (1.0, -1.0)]
        third = 1.0 / math.sqrt(3.0)
        local += [np.array([sx, sy, sz]) * third for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)]
        return np.stack([core.c + core.R @ (core.s * p) for p in local])
    ax = core.axis
    u, w = _disk_basis(ax)
    ring = core.r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), w))
    if core.kind == "cone":
        return np.concatenate([core.base + ring, core.apex[None, :]])
    rims = [core.c + core.h * ax + ring, core.c - core.h * ax + ring]
    if core.kind == "barrel":
        rims.append(core.c + (core.rho(0.0) / core.r) * ring)
    return np.concatenate(rims)


def _solid_recovery(hf, core, *, kmax=4):
    """G3 for solid cores with no row: up to 4 of the core's deepest boundary points below the surface, clipped to the
    footprint (a point outside it has no face above it; the core's part inside it on a border edge's clip plane below
    the edge's face is rule F's clip row, :func:`_clip_points`, so a pair left to rule R has none)."""
    if hasattr(core, "verts") or core.kind in ("cone", "ellipsoid", "barrel"):
        pts = solid_recovery_points(core)
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


EDGE_ORDER = {"H": 0, "V": 1, "D": 2}
"""Window edge kinds in the kernels' walk order (a vertex's x-edge, y-edge, diagonal)."""


def _footprint_crossing(a, b, v0, v1):
    """Parameters ``(s, t)`` where segment ``ab``'s footprint (xy) crosses segment ``v0 v1``'s, or None."""
    d1 = (b - a)[:2]
    d2 = (v1 - v0)[:2]
    den = float(d1[0] * d2[1] - d1[1] * d2[0])
    if den == 0.0:
        return None
    r = (v0 - a)[:2]
    s = float(r[0] * d2[1] - r[1] * d2[0]) / den
    t = float(r[0] * d1[1] - r[1] * d1[0]) / den
    if not (0.0 <= s <= 1.0 and 0.0 <= t <= 1.0):
        return None
    return s, t


def capsule_law(hf, core, D, window, *, stats=None):
    """The rows of a segment core: its ends as point cores (G2, G3) and its axis interior.

    The axis interior's rows are the closest points of the axis and a crease (both segment parameters inside) or a
    vertex, the normal from the surface toward the axis point in the feature's normal cone:

    * outside the solid: at a ridge (convex crease) beyond both faces, at a vertex in its Voronoi region;
    * inside it (an exit): below a ridge within both faces, so that the normal from the axis point to the crease
      lies in the ridge's arc and the exit lifts the axis over the crease, the segment's minimum translation out of
      the ridge (a point exit through a face would push the segment sideways); at a valley (concave crease) beyond
      both faces and at a pit vertex in its Voronoi region, the axis point's closest surface point.

    G3 for the axis interior: with both ends outside and no axis exit, the deepest crossing of the axis's footprint
    under a crease (ties to the first in the kernels' walk order) gets the plane of the crease's first triangle,
    its normal and depth.
    """
    tris, edges, verts = window
    law = PointLaw(hf, tris, edges, verts)
    out = law.minima(core.a, D, "a", stats=stats) + law.minima(core.b, D, "b", stats=stats)
    a, b = core.a, core.b
    ab = b - a
    axis = []
    ends_outside = not _below_surface(hf, a) and not _below_surface(hf, b)
    witness = None
    # axis-interior minima and exits against creases (a border edge is a ridge between its triangle and its wall)
    for e in edges:
        kind, r, c, i0, i1, ta, tb, _oa, _ob = e
        side = None
        if ta is None or tb is None:
            side = hf.border_side(kind, r, c)
            if side is None:
                continue
            cls = "convex"
        else:
            cls = hf.edge_class(e)
        if cls == "flat":
            continue
        v0, v1 = hf.V(*i0), hf.V(*i1)
        row = None
        st = _closest_seg_seg(a, b, v0, v1)
        if st is not None and 0.0 < st[0] < 1.0 and 0.0 < st[1] < 1.0:
            s, t = st
            p = a + s * ab
            q = v0 + t * (v1 - v0)
            key = tuple(sorted((i0, i1)))
            if side is None:
                sa, sb = law.face_pred(ta, key, p), law.face_pred(tb, key, p)
            else:
                sa = law.face_pred(ta if ta is not None else tb, key, p)
                sb = float((p - v0) @ hf.wall_frame(v0, v1, side))
            dist = float(np.linalg.norm(p - q))
            if dist >= 1e-12:
                if _below_surface(hf, p):
                    # a border edge is one-sided: an axis below it exits through the top (G3 at its crossing)
                    claims = side is None and ((sa > 0.0 and sb > 0.0) if cls == "concave" else (sa < 0.0 and sb < 0.0))
                    n, d = (q - p) / dist, -dist
                else:
                    claims = cls == "convex" and sa > 0.0 and sb > 0.0
                    n, d = (p - q) / dist, dist
                if claims and d <= D:
                    row = Row(q, p, n, d, ("s", "E", e[0], e[1], e[2]))
        if row is not None:
            axis.append(row)
        elif ends_outside:
            crossing = _footprint_crossing(a, b, v0, v1)
            if crossing is not None:
                s, t = crossing
                depth = float((v0[2] + t * (v1[2] - v0[2])) - (a[2] + s * ab[2]))
                order = (r, c, EDGE_ORDER[kind])
                if depth > 0.0 and (witness is None or (-depth, order) < (-witness[0], witness[1])):
                    witness = (depth, order, a + s * ab, ta if ta is not None else tb)
    # axis-interior minima against vertices
    L2 = float(ab @ ab)
    for r, c in verts:
        ring = hf.ring(r, c)
        if ring is None:
            row = _axis_border_vertex(hf, law, a, b, r, c, D)
            if row is not None:
                axis.append(row)
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
        axis.append(Row(v, p, n, d, ("s", "V", r, c)))
    if witness is not None and not any(row.d < 0.0 for row in axis):
        _depth, _order, p, ta = witness
        n, pa = hf.plane(ta)
        d = float(n @ (p - pa))
        if d < 0.0:
            axis.append(Row(p - d * n, p.copy(), n.copy(), d, ("s", "R", ta)))
    return out + axis


def _axis_border_vertex(hf, law, a, b, r, c, D):
    """The axis interior at a border vertex, outside the solid only: the axis point over the vertex in its Voronoi
    region (its ring in the grid and the wall below it), else at a corner of the footprint the axis point closest to
    the corner line below it, beyond both walls."""
    ab = b - a
    v = hf.V(r, c)
    s = float((v - a) @ ab) / float(ab @ ab)
    if 0.0 < s < 1.0:
        p = a + s * ab
        row = law.border_vertex(r, c, p, D, "s") if not _below_surface(hf, p) and p[2] >= v[2] else None
        if row is not None:
            return row
    sides = hf.vertex_sides(r, c)
    if len(sides) < 2:
        return None
    st = _closest_seg_seg(a, b, v, v - np.array([0.0, 0.0, WALL_DEPTH]))
    if st is None or not (0.0 < st[0] < 1.0 and 0.0 < st[1] < 1.0):
        return None
    return law.border_vertex(r, c, a + st[0] * ab, D, "s")


def _sigma(core, n, x0):
    """The core's separation from x0 along n, ``min n . (k - x0)`` (exact: a cone's lowest point without its tie
    band), and that point."""
    k = core.lowest(n)
    if core.kind == "cone":
        g, p = core.generator(n)
        k = p if float(n @ p) < float(n @ g) else g
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
    or a cylinder's lowest point or generator, ('disk', centre, axis, r) for a cylinder cap within the band.

    A cone's is its base disk (base down within the band), the lowest generator (both ends within the band) or its
    lowest point; an ellipsoid's is its support point; a barrel's a cap within the band or its lowest point."""
    if hasattr(core, "verts"):
        h = core.verts @ n
        return ("pts", core.verts[h <= h.min() + band])
    if core.kind == "ellipsoid":
        return ("pts", core.lowest(n)[None, :])
    if core.kind == "barrel":
        # a level cap within the band, else the lowest point (the curved side has no generator)
        ax = core.axis
        tilt = math.sqrt(max(0.0, 1.0 - float(n @ ax) ** 2))
        if 2.0 * core.r * tilt <= band:
            return ("disk", core.c + (-1.0 if n @ ax > 0 else 1.0) * core.h * ax, ax, core.r)
        return ("pts", core.lowest(n)[None, :])
    if core.kind == "cone":
        ax = core.axis
        tilt = math.sqrt(max(0.0, 1.0 - float(n @ ax) ** 2))
        if float(n @ ax) > 0.0 and 2.0 * core.r * tilt <= band:
            return ("disk", core.base.copy(), ax, core.r)
        g, p = core.generator(n)
        k0, k1 = (p, g) if float(n @ p) < float(n @ g) - core._band() else (g, p)
        if float(n @ (k1 - k0)) <= band:
            return ("pts", np.stack([k0, k1]))
        return ("pts", k0[None, :])
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
    generator (a barrel's lowest point)."""
    if hasattr(core, "verts"):
        h = core.verts @ n
        radius = float(np.linalg.norm(core.verts - core.c, axis=1).max())
        deepest = first_within(h, TIE * radius)
        through = [q for q, f in enumerate(core.faces) if deepest in f]
        k = through[first_within([-float(core.fnormals[q] @ n) for q in through], TIE, largest=True)]
        return ("pts", core.verts[core.faces[k]])
    if core.kind == "ellipsoid":
        return ("pts", core.lowest(n)[None, :])
    if core.kind == "barrel":
        # The near cap within the flat mode, else the lowest point (which straddles no line).
        axis = core.axis if float(n @ core.axis) < 0.0 else -core.axis
        if -float(n @ axis) >= CYLINDER_FLAT_MODE_COS:
            return ("disk", core.c + core.h * axis, axis, core.r)
        return ("pts", core.lowest(n)[None, :])
    if core.kind == "cone":
        # The base within the flat mode, else the lowest generator (the apex alone when the axis is along n).
        if float(n @ core.axis) >= CYLINDER_FLAT_MODE_COS:
            return ("disk", core.base.copy(), core.axis, core.r)
        perp = -n + core.axis * float(n @ core.axis)
        if float(np.linalg.norm(perp)) < 1e-12:
            return ("pts", core.apex[None, :])
        return ("pts", np.stack(core.generator(n)))
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
    crossing = _feature_crossing(feat, v0, e1, e2, tol, need_straddle=need_straddle)
    return None if crossing is None else crossing[:2]


def _feature_crossing(feat, v0, e1, e2, tol, *, need_straddle=False):
    """:func:`_feature_interval` with a feature point over each end: ``(lo, hi, k_lo, k_hi)`` or None.

    A point set's ends are the extreme ones among its points within ``tol`` of the line and the crossings of the
    line by the chords between its points on either side (the line meets their convex hull in that interval); a
    disk's are its rim points over the ends."""
    if feat[0] == "pts":
        S = feat[1]
        P = np.stack([(S - v0) @ e1, (S - v0) @ e2], 1)
        if need_straddle and not (P[:, 1].min() < -1e-9 and P[:, 1].max() > 1e-9):
            return None
        ends = [(float(P[i, 0]), S[i]) for i in range(len(S)) if abs(P[i, 1]) <= tol]
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                ya, yb = P[i, 1], P[j, 1]
                if (ya < -tol and yb > tol) or (ya > tol and yb < -tol):
                    f = ya / (ya - yb)
                    ends.append((float(P[i, 0] + f * (P[j, 0] - P[i, 0])), S[i] + f * (S[j] - S[i])))
        if not ends:
            return None
        lo = min(ends, key=lambda end: end[0])
        hi = max(ends, key=lambda end: end[0])
        return lo[0], hi[0], lo[1].copy(), hi[1].copy()
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
        along = (u * float(u @ e1) + w * float(w @ e1)) / max(ext / r, 1e-300)
        return xc - ext, xc + ext, C - r * along, C + r * along
    if abs(yc) > R + tol:
        return None
    if need_straddle and abs(yc) >= R - 1e-9:
        return None
    phi = math.atan2(B_, A_)
    dl = math.acos(max(-1.0, min(1.0, -yc / R)))
    rim = [C + r * (math.cos(th) * u + math.sin(th) * w) for th in (phi + dl, phi - dl)]
    xs = [float((k - v0) @ e1) for k in rim]
    i_lo, i_hi = (0, 1) if xs[0] <= xs[1] else (1, 0)
    return xs[i_lo], xs[i_hi], rim[i_lo], rim[i_hi]


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


def _crease_rows(core, D, band, size, v0, v1, nA, nB, end_feature, tag, groups, *, one_sided=False, stats=None):
    """Rule E's rows of a convex crease from ``v0`` to ``v1`` with face normals ``nA`` and ``nB`` (a border edge: its
    triangle's and its wall's; a corner line: its walls'): theta* on the normal arc, then the core's flat support
    feature (interior) or reference feature (an arc end, G1) crossed with the crease's line, its ends lifted onto the
    core. ``end_feature(j)``: whether the edge's end ``j`` is a vertex feature (an end there is the vertex's); ``groups``: the
    plane groups of the arc's ends; ``one_sided``: no row for a core inside both faces (a corner line's walls)."""
    out = []
    dvec = v1 - v0
    L = float(np.linalg.norm(dvec))
    dh = dvec / L
    # the edge's normal arc, exactly perpendicular to the edge
    nA = nA - (nA @ dh) * dh
    nA /= np.linalg.norm(nA)
    nB = nB - (nB @ dh) * dh
    nB /= np.linalg.norm(nB)
    cosab = float(np.clip(nA @ nB, -1.0, 1.0))
    th_ab = math.acos(cosab)
    m = nB - cosab * nA
    if th_ab < 1e-9 or np.linalg.norm(m) < 1e-12:
        return out
    m /= np.linalg.norm(m)
    lo_s = max(_sigma(core, nA, v0)[0], _sigma(core, nB, v0)[0])
    if lo_s > D:
        return out
    if stats:
        stats.edge_solves += 1
    th, sig = _maximize_arc(core, nA, m, th_ab, v0)
    if sig > D or (one_sided and sig < 0.0):
        return out
    nst = math.cos(th) * nA + math.sin(th) * m
    interior = EPS_THETA < th < th_ab - EPS_THETA
    e1, e2 = dh, np.cross(nst, dh)
    # G1: interior theta*: the flat support feature along -n(theta*); arc end: the reference feature
    feat = _support_feature(core, nst, band) if interior else _reference_feature(core, nst)
    iv = _feature_crossing(feat, v0, e1, e2, 1e-7 + 1e-6 * size, need_straddle=not interior)
    if iv is None and interior:
        # inside the arc the feature meets the line; the float32 search's rounding may leave a single fast-turning
        # support point off it: within the flatness band
        iv = _feature_crossing(feat, v0, e1, e2, band)
    xs = []
    if iv is not None:
        x0, x1 = max(iv[0], 0.0), min(iv[1], L)
        if x1 >= x0 - 1e-12:
            xs = [x0, x1] if x1 - x0 >= 1e-9 else [0.5 * (x0 + x1)]
    # an end of the edge on a vertex feature belongs to that vertex
    xs = [x for x in xs if not ((x <= 1e-12 and end_feature(0)) or (x >= L - 1e-12 and end_feature(1)))]
    grp = (*tag, round(th, 9)) if interior else (groups[0] if th < 0.5 * th_ab else groups[1])
    for x in xs:
        u = v0 + x * dh
        # The feature's chord over the interval lies in the core on the line, so the lift is no deeper than its
        # point over u; at an end, where the line can only touch the core, that point is the lift (a tangent
        # lift that rounding made miss, or carried past the chord, is the chord's point).
        lo, hi, k_lo, k_hi = iv
        k = k_lo if x == lo else k_hi if x == hi else k_lo + ((x - lo) / (hi - lo)) * (k_hi - k_lo)
        tt = float(nst @ (k - u))
        hit = core.lift(u, nst)
        if hit is not None and hit[1] <= tt + LIFT_SLACK:
            tt = hit[1]
        if tt > D:
            continue
        out.append(Row(u, u + tt * nst, nst.copy(), tt, (*tag, round(x / L, 6)), group=grp))
    return out


def _feature_min(feat, m, v0):
    """The smallest ``m . (k - v0)`` over a support feature (:func:`_support_feature`)."""
    if feat[0] == "pts":
        return float(np.min((feat[1] - v0) @ m))
    _kind, centre, axis, r = feat
    return float(m @ (centre - v0)) - r * math.sqrt(max(0.0, 1.0 - float(m @ axis) ** 2))


def _clip_normal(core, n, m, v0, band):
    """Rule F's clip at a border edge (face normal ``n``, clip plane through ``v0`` of outward normal ``m``): the normal
    on the quarter arc from ``n`` toward ``m`` at which the core's support first reaches the clip plane (``n`` itself
    when its flat support feature does), or None when no point of the core lies on the face's side of the plane."""
    if _feature_min(_support_feature(core, n, band), m, v0) <= 0.0:
        return n
    if float(m @ (core.lowest(m) - v0)) >= 0.0:
        return None
    lo, hi = 0.0, 0.5 * math.pi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        k = core.lowest(math.cos(mid) * n + math.sin(mid) * m)
        if float(m @ (k - v0)) > 0.0:
            lo = mid
        else:
            hi = mid
    return math.cos(hi) * n + math.sin(hi) * m


def _clip_points(hf, core, D, window, band, tol):
    """Rule F's footprint clipping: per border edge of the window, the core's part on its triangle's side of the clip
    plane through the edge (containing the face's normal, its outward normal the face's in-plane normal at the edge),
    deepest along the face's normal: the ends of its support feature's crossing with the clip plane at
    :func:`_clip_normal`, an end past a corner of the footprint replaced by the core's point on the corner line (the
    lift along the normal), an end past another vertex left to the next edge. Returns ``(k, h, n, t, order)`` per point:
    core point, depth along the face's normal ``n``, the face's triangle and the edge's place (row, column, x/y, end),
    within the reach and front-facing as a face's supported point is (below the face inside every wall and below the
    surface, above it not below the surface)."""
    out = []
    for e in window[1]:
        kind, r, c, i0, i1, ta, tb, oa, ob = e
        if ta is not None and tb is not None:
            continue
        side = hf.border_side(kind, r, c)
        if side is None:
            continue
        t = ta if ta is not None else tb
        far = oa if ta is not None else ob
        n, a = hf.plane(t)
        v0, v1 = hf.V(*i0), hf.V(*i1)
        m = np.cross(v1 - v0, n)
        if m @ (hf.V(*far) - v0) > 0.0:
            m = -m
        m = m / np.linalg.norm(m)
        nc = _clip_normal(core, n, m, v0, band)
        if nc is None:
            continue
        L = float(np.linalg.norm(v1 - v0))
        dh = (v1 - v0) / L
        feat = _support_feature(core, nc, band)
        iv = _feature_crossing(feat, v0, dh, m, tol)
        if iv is None and nc is not n:
            iv = _feature_crossing(feat, v0, dh, m, band)
        if iv is None:
            continue
        lo, hi, k_lo, k_hi = iv
        ends = [(lo, k_lo), (hi, k_hi)] if hi - lo >= 1e-9 else [(0.5 * (lo + hi), 0.5 * (k_lo + k_hi))]
        for slot, (x, k_end) in enumerate(ends):
            k = k_end
            corner = i0 if x < -1e-12 else i1 if x > L + 1e-12 else None
            if corner is not None:
                if len(hf.vertex_sides(*corner)) < 2:
                    continue
                hit = core.lift(hf.V(*corner), n)
                if hit is None:
                    continue
                k = hit[0]
            else:
                # the core's point on the clip plane over the end (its lift along the normal)
                u = v0 + x * dh
                hit = core.lift(u, n)
                if hit is not None and hit[1] <= float(n @ (k - u)) + LIFT_SLACK:
                    k = hit[0]
            h = float(n @ (k - a))
            if h > D:
                continue
            if h < 0.0 and max(abs(k[0]) - hf.hx, abs(k[1]) - hf.hy) > 1e-12:
                # below the face: inside every wall (near a corner the point may lie beyond the other one)
                continue
            # front-facing, as a face's supported point: below the face inside the solid (the point taken within the
            # footprint's closure), above it outside
            if h < -EPS_PRED:
                if not _below_surface(hf, np.r_[np.clip(k[:2], (-hf.hx, -hf.hy), (hf.hx, hf.hy)), k[2]]):
                    continue
            elif h > EPS_PRED and _below_surface(hf, k):
                continue
            out.append((k.copy(), h, n.copy(), t, (r, c, 0 if kind == "H" else 1, slot)))
    return out


def _wall_rows(hf, core, D, window):
    """Rule F on the walls: per wall of the window, its plane manifold's points beside it (the walls are one-sided)
    whose projection a piece of it in the window holds (below a border edge, between its vertices' vertical lines,
    above the base)."""
    pieces = {}
    for e in window[1]:
        kind, r, c, i0, i1, ta, tb, _oa, _ob = e
        if ta is not None and tb is not None:
            continue
        side = hf.border_side(kind, r, c)
        if side is not None:
            pieces.setdefault(side, []).append((hf.V(*i0), hf.V(*i1)))
    out = []
    for side, segments in sorted(pieces.items()):
        n = WALL_NORMALS[side]
        for k, h in core.plane_manifold(n, segments[0][0], D):
            if h < -1e-12:
                continue
            if any(hf.wall_holds(side, v0, v1, k, tol=1e-12) for v0, v1 in segments):
                out.append(Row(k - h * n, k.copy(), n.copy(), h, ("W", side), group=("W", side)))
    return out


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
    # ---- the border: the faces clipped to the footprint, the walls ---------------------------------------------
    for k, h, n, t, order in _clip_points(hf, core, D, window, band, 1e-7 + 1e-6 * size):
        out.append(Row(k - h * n, k, n, h, ("F", t, "clip", order), group=("P", t)))
    out.extend(_wall_rows(hf, core, D, window))
    # ---- convex edges: creases, border edges (triangle and wall), corner lines (two walls) --------------------
    vcache = {}

    def vfeat(rc):
        if rc not in vcache:
            vcache[rc] = hf.vertex_feature(*rc)
        return vcache[rc]

    for e in edges:
        kind, er, ec, i0, i1, ta, tb, _oa, _ob = e
        if ta is None or tb is None:
            side = hf.border_side(kind, er, ec)
            if side is None:
                continue
            t = ta if ta is not None else tb
            nA, nB = hf.plane(t)[0], WALL_NORMALS[side]
            groups = (("P", t), ("W", side))
        else:
            if hf.edge_class(e) != "convex":
                continue
            nA, nB = hf.plane(ta)[0], hf.plane(tb)[0]
            groups = (("P", ta), ("P", tb))
        if stats:
            stats.convex_edges += 1
        out.extend(
            _crease_rows(
                core,
                D,
                band,
                size,
                hf.V(*i0),
                hf.V(*i1),
                nA,
                nB,
                lambda j, i0=i0, i1=i1: vfeat(i1 if j else i0),
                ("E", kind, er, ec),
                groups,
                stats=stats,
            )
        )
    for r, c in verts:
        sides = hf.vertex_sides(r, c)
        if len(sides) < 2:
            continue
        v = hf.V(r, c)
        out.extend(
            _crease_rows(
                core,
                D,
                band,
                size,
                v,
                v - np.array([0.0, 0.0, WALL_DEPTH]),
                WALL_NORMALS[sides[0]],
                WALL_NORMALS[sides[1]],
                lambda j, r=r, c=c: j == 0 and vfeat((r, c)),
                ("L", r, c),
                (("W", sides[0]), ("W", sides[1])),
                one_sided=True,
                stats=stats,
            )
        )
    # ---- vertex features -------------------------------------------------------------------------------------
    for r, c in verts:
        ring = hf.ring(r, c)
        if not hf.vertex_feature(r, c):
            continue
        v = hf.V(r, c)
        if ring is None:
            # a border vertex: its triangles and walls, its ring in the grid and the wall below it
            inc = hf.incident_in_grid(r, c)
            normals = [hf.plane(t)[0] for t in inc]
            W = [hf.V(rr, cc) - v for rr, cc in hf.border_ring(r, c)] + [np.array([0.0, 0.0, -1.0])]
        else:
            inc = hf.incident_triangles(r, c)
            normals = [hf.plane(t)[0] for t in inc]
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
    if core.kind == "barrel":
        return core.nearest(v)
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
    if core.kind == "cone":
        # The base plane or the lateral surface, whichever is nearer (both feet lie on the boundary from inside).
        w = v - core.apex
        z = float(w @ core.axis)
        rad = w - z * core.axis
        lr = float(np.linalg.norm(rad))
        d_base = z + 2.0 * core.h
        slant = math.hypot(core.r, 2.0 * core.h)
        d_side = (2.0 * core.h * lr + core.r * z) / -slant
        if d_base <= d_side:
            return core.axis.copy(), v - d_base * core.axis, -d_base
        radial = rad / lr if lr > 1e-12 else _disk_basis(core.axis)[0]
        f = (2.0 * core.h * radial + core.r * core.axis) / slant
        return -f, v + d_side * f, -d_side
    if core.kind == "ellipsoid":
        k = core.boundary_nearest(v)
        f = core.outward_normal(k)
        return -f, k, -float(np.linalg.norm(k - v))
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
    if core.kind == "cone":
        if core.contains(v):
            return None
        # The nearest point of the meridian section (a triangle: apex, base rim, opposite base rim) in its plane.
        w = v - core.apex
        z = float(w @ core.axis)
        rad = w - z * core.axis
        lr = float(np.linalg.norm(rad))
        radial = rad / lr if lr > 1e-12 else _disk_basis(core.axis)[0]
        p = np.array([lr, z, 0.0])
        apex2, rim2, far2 = np.zeros(3), np.array([core.r, -2.0 * core.h, 0.0]), np.array([-core.r, -2.0 * core.h, 0.0])
        q = _closest_on_tri(p, apex2, rim2, far2)
        return core.apex + q[1] * core.axis + q[0] * radial
    if core.kind == "ellipsoid":
        if core.contains(v):
            return None
        return core.boundary_nearest(v)
    return None


# ----------------------------------------------------------------------------------------------------------------
# finish
# ----------------------------------------------------------------------------------------------------------------


def finish(
    rows,
    *,
    stats=None,
    max_per_group=4,
    row_limit=None,
    implied="lower_hull",
    group_normal=GROUP_NORMAL,
    group_offset=GROUP_OFFSET,
):
    """The per-pair finish: merge, implied rows, ``max_per_group`` rows per plane group, ``row_limit`` per pair.

    1. Coincident rows (terrain points within ``EPS_MERGE_P``, normals within ``EPS_MERGE_N``) merge into the
       first in feature order.
    2. Plane groups form in feature order: a row joins the first group whose first row's normal lies within
       ``group_normal`` of its own and whose plane lies within ``group_offset`` of its terrain point.
    3. A group drops the rows the others imply (see ``implied``).
    4. A group keeps at most ``max_per_group`` rows by :func:`group_ranks`.
    5. With ``row_limit``, a pair keeps at most that many rows: every group's rank-0 row (its deepest) first,
       then the rank-1 rows (the farthest), and so on; within a rank the deepest first, ties in feature order.

    Args:
        rows: Rows of one pair in feature order.
        stats: Optional counters.
        max_per_group: Rows kept per plane group.
        row_limit: Rows kept per pair, or ``None`` for no limit.
        implied: ``"lower_hull"`` (rows on or above the lower hull of the others' (tangent position, distance))
            or ``"hull3d"`` (the prototype: core point inside the others' 3-D hull and not deeper).
        group_normal: Largest normal difference within a plane group.
        group_offset: Largest distance [m] of a row's terrain point from its group's plane.

    Returns:
        The kept rows in feature order.
    """
    kept = []
    for ct in rows:
        if any(np.linalg.norm(o.q - ct.q) <= EPS_MERGE_P and np.linalg.norm(o.n - ct.n) <= EPS_MERGE_N for o in kept):
            if stats:
                stats.merged += 1
            continue
        kept.append(ct)
    groups = []
    for ct in kept:
        for members in groups:
            first = members[0]
            if (
                np.linalg.norm(ct.n - first.n) <= group_normal
                and abs(float(first.n @ (ct.q - first.q))) <= group_offset
            ):
                members.append(ct)
                break
        else:
            groups.append([ct])
    ranked = []
    for members in groups:
        group = _drop_implied(members, stats, implied)
        order, chosen = group_ranks(group, normal=members[0].n)
        keep = order if len(group) <= max_per_group else order[: min(chosen, max_per_group)]
        if stats:
            stats.reduced += len(group) - len(keep)
        ranked.extend((rank, group[i]) for rank, i in enumerate(keep))
    position = {id(ct): i for i, ct in enumerate(rows)}
    if row_limit is not None and len(ranked) > row_limit:
        ranked.sort(key=lambda item: (item[0], item[1].d, position[id(item[1])]))
        if stats:
            stats.limited += len(ranked) - row_limit
        ranked = ranked[:row_limit]
    final = sorted((ct for _rank, ct in ranked), key=lambda ct: position[id(ct)])
    if stats:
        stats.emitted += len(final)
    return final


def group_ranks(group, *, normal=None):
    """Rank the rows of one plane group, in tangent positions on its plane; ties in feature order.

    Returns ``(order, chosen)``: rank 0 is the deepest row, rank 1 the farthest from it, rank 2 the one spanning
    the largest triangle with those two and rank 3 the one adding the most area (more than 1e-15) outside that
    triangle; these ``chosen`` rows come first, the others after them in feature order (a group of at most four
    rows keeps them all, in this order). ``normal`` is the group's normal, by default its first row's.
    """
    n = group[0].n if normal is None else normal
    P = np.array([ct.k - float(ct.k @ n) * n for ct in group]).reshape(-1, 3)
    rest = list(range(len(group)))
    if not rest:
        return [], 0
    order = [min(rest, key=lambda i: (group[i].d, i))]
    rest.remove(order[0])
    p0 = P[order[0]]
    if rest:
        order.append(max(rest, key=lambda i: (float(np.linalg.norm(P[i] - p0)), -i)))
        rest.remove(order[1])
    if rest:
        p1 = P[order[1]]
        order.append(max(rest, key=lambda i: (abs(float(np.cross(p1 - p0, P[i] - p0) @ n)), -i)))
        rest.remove(order[2])
    if rest:
        tri = [P[i] for i in order]
        best, pick = 1.0e-15, None
        for j in rest:
            gain = 0.0
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                side_c = float(np.cross(tri[b] - tri[a], tri[c] - tri[a]) @ n)
                side_j = float(np.cross(tri[b] - tri[a], P[j] - tri[a]) @ n)
                if side_c * side_j < 0.0:
                    gain = max(gain, abs(side_j))
            if gain > best:
                best, pick = gain, j
        if pick is not None:
            order.append(pick)
            rest.remove(pick)
    return order + rest, len(order)


def _drop_implied(g, stats, implied):
    """Drop a plane group's rows its other rows imply, the shallowest first, one at a time: in one pass, which drops
    what testing again from the shallowest after every drop would (a drop only raises the others' lower hull)."""
    if len(g) < 3:
        return g
    K = np.array([ct.k for ct in g])
    d = np.array([ct.d for ct in g])
    alive = [True] * len(g)
    size = len(g)
    for j in sorted(range(len(g)), key=lambda i: (-d[i], i)):
        if size < 3:
            break
        others = [i for i in range(len(g)) if i != j and alive[i]]
        if implied == "hull3d":
            val = _convex_interp3(K[others], d[others], K[j])
            drop = val is not None and d[j] >= val - 1e-9
        else:
            drop = lower_hull_reaches(K[others], d[others], K[j], g[0].n, d[j], IMPLIED_RESIDUAL)
        if drop:
            if stats:
                stats.implied += 1
            alive[j] = False
            size -= 1
    return [ct for ct, keep in zip(g, alive, strict=True) if keep]


def lower_hull_reaches(P, d, p, n, depth, residual, *, tol=1e-9):
    """Whether ``depth >= lower_hull_value(P, d, p, n) - residual`` (:func:`lower_hull_value`): the first interpolation
    that reaches ``depth`` decides it, as the lowest would (a rounded subtraction keeps order)."""
    e1 = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    U = np.stack([P @ e1, P @ e2], 1) if len(P) else np.zeros((0, 2))
    u = np.array([p @ e1, p @ e2])
    m = len(U)
    for i in range(m):
        if np.linalg.norm(U[i] - u) <= tol and depth >= d[i] - residual:
            return True
        for j in range(i + 1, m):
            a, b = U[i], U[j]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-20:
                continue
            s = float((u - a) @ ab) / L2
            perp = abs(ab[0] * (u - a)[1] - ab[1] * (u - a)[0]) / math.sqrt(L2)
            if -tol <= s <= 1 + tol and perp <= tol and depth >= (d[i] + s * (d[j] - d[i])) - residual:
                return True
            for k in range(j + 1, m):
                c = U[k]
                T = np.stack([b - a, c - a], 1)
                det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
                if abs(det) < 1e-18:
                    continue
                uv = np.linalg.solve(T, u - a)
                if uv[0] >= -tol and uv[1] >= -tol and uv[0] + uv[1] <= 1 + tol:
                    if depth >= (d[i] + uv[0] * (d[j] - d[i]) + uv[1] * (d[k] - d[i])) - residual:
                        return True
    return False


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
# the pair entry point
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


def pair_rows(
    hf: ReferenceHeightfield,
    core: Core,
    D: float,
    *,
    stats=None,
    implied="lower_hull",
    group_offset=GROUP_OFFSET,
):
    """The law's rows of one pair. ``D``: reach on the core distance (rounding radius + margins + gaps).

    ``group_offset`` is the finish's (:func:`finish`; a comparison with the float32 kernels passes their four bands).
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
    return finish(raw, stats=stats, implied=implied, group_offset=group_offset)

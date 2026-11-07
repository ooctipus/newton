# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static data of the per-pair terrain contact law: surface features and the terrain pair table.

:class:`TerrainFeatures` builds, for every heightfield of a model, the per-triangle plane clusters (with the
plane records of clusters of several triangles), the tile-max mip and the version stamps; the law's kernels read
them through :class:`~newton._src.geometry.terrain_law.TerrainFeatureData`. :class:`TerrainPairTable` lists every
(heightfield, convex partner) pair the collision filter admits, grouped into one contiguous segment per partner
class and lane class. Nothing routes contacts through the law yet: building either object changes no contact.

Plane clusters are built by Warp kernels on the model's device, round by round over the triangles not yet
assigned:

1. Components over the flat edges (the far vertex of one side within ``1 um`` of the other side's plane).
2. Three candidate planes per component: the extreme plane through its lowest-id vertex ``a``, the vertex ``b``
   farthest from ``a`` and the vertex ``c`` spanning the largest triangle with them (grid units, lowest id on
   ties), and the level (horizontal) plane, each with its offset centred on the component's residual range; and
   the seed plane of its lowest-id triangle.
3. A component whose vertices all lie within ``1 um`` of its extreme or level plane is one cluster with the
   closer of the two. Otherwise the candidate claiming the most triangles (all three vertices within ``1 um``;
   the seed plane, then the extreme plane on ties) forms a cluster of them, and the rest wait for the next round.
4. After eight rounds the triangles left are clusters of their own.

The plane of a single 0.1 m triangle carries the float32 height decode noise (a tilt up to about ``2e-6`` rad),
so it cannot carry a large cluster on its own: a 12.7 m planar 12 deg slope decoded in float32 split into 4,072
clusters under the plane of the lowest-id triangle, and cropping the recorded suite terrain by three rows and
seven columns split its flat border (decoded heights within 2.9 um) into over a million clusters; with these
candidates the slope is one cluster and the suite terrain has 539,984 clusters (539,982 cropped). Gradients and
residuals are rounded to float32 before every comparison, as the float64 reference rounds them. A cluster is
named after its lowest triangle; its id is the dense index of its plane record, or ``-(1 + triangle id)`` for a
triangle alone.

This module's kernels are compiled without floating-point contraction (``fuse_fp=False``) and the heights are
decoded with explicitly rounded operations, so every operation of a build rounds as it does on the CPU: the
ids, plane records and mip are the same bits on every device. The build runs at construction and after edits,
never per step; the law's per-pair kernels belong in a module of their own.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import warp as wp

from ..sim.shape_contact_pairs import _shape_contact_pairs_host
from .terrain_law import (
    TERRAIN_FINE_CELLS,
    TERRAIN_FLAT_EPS,
    TERRAIN_TILE_CELLS,
    TerrainClusterPlane,
    TerrainFeatureData,
    TerrainSurfaceInfo,
    terrain_decode,
    terrain_vertex_bits_of_clusters,
    terrain_window,
)
from .types import GeoType

# CUDA would otherwise fuse the build's float64 products and sums (fma.rn.f64) where the CPU and the float64
# reference round each one, and a float32 rounding of a residual or gradient could then flip.
wp.set_module_options({"fuse_fp": False})

TERRAIN_CLUSTER_ROUNDS = 8
"""Rounds of the cluster build; triangles still unassigned afterwards are clusters of their own."""

TERRAIN_PARTNER_POINT = 0
"""Partner class of point and segment cores: spheres and capsules."""
TERRAIN_PARTNER_BOX = 1
"""Partner class of boxes and finite planes (a box of no thickness)."""
TERRAIN_PARTNER_CYLINDER = 2
"""Partner class of cylinders (flat-sided and barrel) and cones."""
TERRAIN_PARTNER_HULL = 3
"""Partner class of the support-function cores: convex meshes and ellipsoids."""
TERRAIN_PARTNER_CLASSES = 4
TERRAIN_PARTNER_WALK_SLOTS = (8, 16, 16, 24)
"""Walk slots of a pair of each partner class in the law's side buffer, the fast path's layout: its walk holds its
distinct rows there and, when the finish keeps them as they are (or, for a solid, when they fit), writes its rows there.
A pair with more distinct rows is deferred to its class's finish of deferred pairs, which holds every candidate its
window can give (:func:`terrain_pair_candidates`) and writes all its rows to the side buffer's spill region: the slots
bound no row, they choose which kernel finishes a pair and where its rows go. Of the recorded suite states' walking
pairs (16,384 worlds, warm and after a run), the walks finish all but a hull pair with 30 rows. Each fits the in-place
finish of a walk, which takes up to :data:`~newton._src.geometry.terrain_law.TERRAIN_FINISH_MASK_ROWS` rows (24)."""
TERRAIN_PARTNER_SCAN_RUN = (0, 8, 32, 8)
"""Window vertices one scan job of a solid class takes (the point class has no scan): a solid pair's window is scanned
in runs of this many vertices, one job each, at most
:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_SCAN_VERTICES`. A shorter run lists the same features with
more threads and repeats a run's setup (its core, record, window frame and list reservations) more often. Measured on
the recorded after-run states (scan kernel per collide, runs of 32, 16 and 8 vertices): hulls 0.216, 0.134, 0.103 ms
(h1) and 0.140, 0.091, 0.080 (g1); boxes 0.096, 0.069, 0.041 (anymald) and 0.093, 0.071, 0.043 (go2); cylinders 0.215,
0.263, 0.331 (anymald), 0.083, 0.083, 0.082 (go2) and 0.067, 0.065, 0.058 (cassie)."""
TERRAIN_WALK_ROWS_PER_VERTEX = (6, 16, 15)
"""Rows a window vertex's features can give a pair at most: a sphere's centre (its two faces, three edges and itself,
one row each), a capsule (both ends' six, the axis's four), a solid (its two faces' plane manifolds of four points
each, its three edges' interval ends, itself)."""
TERRAIN_WALK_ROWS_PER_PAIR = (1, 3, 4)
"""Rows a pair can gain after its walk at most: a sphere's exit, a capsule's two ends' exits and its axis's recovery,
a solid's four recovery rows."""
TERRAIN_BORDER_ROWS_PER_SIDE = 4
"""Rows a solid pair's window can give at the heightfield's border, per vertex along its sides at most
(:func:`terrain_pair_candidates`): two clipped rows of each border edge's face (two border edges per window vertex along
a side, at most, its rows and columns on both borders). A sphere's or capsule's border claims take its edges' and
vertices' items, which :data:`TERRAIN_WALK_ROWS_PER_VERTEX` counts."""
TERRAIN_BORDER_ROWS_PER_PAIR = 24
"""Rows a solid pair's window can give at the heightfield's border beyond :data:`TERRAIN_BORDER_ROWS_PER_SIDE`: a wall's
plane manifold (four rows) on each of four walls and two rows on each of four corner lines, at most."""

TERRAIN_LANE_THREAD = 0
"""Lane class of pairs whose largest window holds at most :data:`TERRAIN_LANE_CELLS` cells: the class's one-thread
kernel walks them."""
TERRAIN_LANE_TILE = 1
"""Lane class of larger windows: the class's one-thread kernel takes such a pair through the whole-pair rejects and,
when it reaches its window, queues it for the class's tiled kernel, a tile of
:data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK` lanes per pair (a solid class's only when its window, as
the pose gives it, holds more than :data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_THREAD_VERTICES`
vertices; it walks the others in one thread)."""
TERRAIN_LANE_CLASSES = 2
TERRAIN_LANE_CELLS = 16
"""Cells of a partner's largest window (:func:`terrain_window_cells`) up to which its pairs are of
:data:`TERRAIN_LANE_THREAD`."""

_FLAT_EPS = wp.constant(TERRAIN_FLAT_EPS)
_INT_MAX = wp.constant(2147483647)
_FLOAT_MAX = wp.constant(3.402823466e38)


# ----------------------------------------------------------------------------------------------------------------
# cluster build kernels (float64, one heightfield per launch)
# ----------------------------------------------------------------------------------------------------------------


@wp.struct
class _Grid:
    elevations: wp.array[wp.float32]
    data_offset: int
    nrow: int
    ncol: int
    min_z: float
    max_z: float
    dx: wp.float64
    dy: wp.float64


@wp.func
def _z(grid: _Grid, vertex: int) -> wp.float64:
    return wp.float64(terrain_decode(grid.min_z, grid.max_z, grid.elevations[grid.data_offset + vertex]))


@wp.func
def _triangle_vertex(grid: _Grid, triangle: int, j: int) -> int:
    """Vertex id ``r * ncol + c`` of corner ``j`` of a triangle (sub 0: p00, p10, p11; sub 1: p00, p11, p01)."""
    cell = triangle // 2
    sub = triangle - 2 * cell
    cols = grid.ncol - 1
    r = cell // cols
    c = cell - r * cols
    v = r * grid.ncol + c
    if j == 0:
        return v
    if sub == 0:
        if j == 1:
            return v + 1
        return v + grid.ncol + 1
    if j == 1:
        return v + grid.ncol + 1
    return v + grid.ncol


@wp.func
def _offset(grid: _Grid, vertex: int, anchor: int) -> wp.vec3d:
    """Vertex minus anchor: integer grid differences times the cell size, height difference."""
    vr = vertex // grid.ncol
    ar = anchor // grid.ncol
    return wp.vec3d(
        wp.float64(vertex - vr * grid.ncol - (anchor - ar * grid.ncol)) * grid.dx,
        wp.float64(vr - ar) * grid.dy,
        _z(grid, vertex) - _z(grid, anchor),
    )


@wp.func
def _triangle_normal(grid: _Grid, triangle: int) -> wp.vec3d:
    v0 = _triangle_vertex(grid, triangle, 0)
    n = wp.cross(
        _offset(grid, _triangle_vertex(grid, triangle, 1), v0), _offset(grid, _triangle_vertex(grid, triangle, 2), v0)
    )
    length = wp.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    return wp.vec3d(n[0] / length, n[1] / length, n[2] / length)


@wp.func
def _flat_edge(grid: _Grid, triangle: int, far: int) -> bool:
    """Whether vertex ``far`` lies within the flat tolerance of a triangle's own plane (rounded to float32)."""
    n = _triangle_normal(grid, triangle)
    w = _offset(grid, far, _triangle_vertex(grid, triangle, 0))
    deviation = wp.float32(wp.abs(n[0] * w[0] + n[1] * w[1] + n[2] * w[2]))
    return wp.float64(deviation) <= wp.float64(_FLAT_EPS)


@wp.func
def _find(parent: wp.array[wp.int32], x: int) -> int:
    """Root of ``x``'s tree as the union pass sees it, halving the path on the way (union pass only)."""
    while True:
        p = parent[x]
        if p == x:
            return x
        grand = parent[p]
        if grand != p:
            # Path halving stores into other triangles' slots. In the union pass that is safe: a node only ever
            # points further up its own tree (a store computed from an older parent still points at an ancestor),
            # and nothing reads a parent as final there. A pass that stores final parents must not halve.
            parent[x] = grand
        x = grand


@wp.func
def _root(parent: wp.array[wp.int32], x: int) -> int:
    """Root of ``x``'s tree, by loads alone: parents only point to lower ids, so the walk ends at the root."""
    p = parent[x]
    while p != x:
        x = p
        p = parent[x]
    return x


@wp.func
def _union(parent: wp.array[wp.int32], a: int, b: int):
    ra = _find(parent, a)
    rb = _find(parent, b)
    while ra != rb:
        lo = wp.min(ra, rb)
        hi = wp.max(ra, rb)
        # Hook the higher root under the lower one: every root is its component's lowest id.
        old = wp.atomic_cas(parent, hi, hi, lo)
        if old == hi:
            return
        # Another thread hooked hi first. Continue from its parent as the atomic returned it (lower than hi, so the
        # loop ends) rather than from a plain reread, which a non-coherent cache may serve stale on CUDA.
        if ra == hi:
            ra = old
        else:
            rb = old


@wp.kernel(enable_backward=False)
def _cluster_init(active: wp.array[wp.int32], label: wp.array[wp.int32]):
    t = wp.tid()
    active[t] = 1
    label[t] = -1


@wp.kernel(enable_backward=False)
def _cluster_parents(parent: wp.array[wp.int32]):
    t = wp.tid()
    parent[t] = t


@wp.kernel(enable_backward=False)
def _cluster_union(grid: _Grid, active: wp.array[wp.int32], parent: wp.array[wp.int32]):
    cell = wp.tid()
    cols = grid.ncol - 1
    r = cell // cols
    c = cell - r * cols
    t0 = 2 * cell
    t1 = t0 + 1
    v00 = r * grid.ncol + c
    # Diagonal: sub 0 against the far vertex p01 of sub 1.
    if active[t0] != 0 and active[t1] != 0:
        if _flat_edge(grid, t0, v00 + grid.ncol):
            _union(parent, t0, t1)
    # x-edge (r, c)-(r, c + 1): sub 0 against p00 of the cell below.
    if r > 0:
        below = t1 - 2 * cols
        if active[t0] != 0 and active[below] != 0:
            if _flat_edge(grid, t0, v00 - grid.ncol):
                _union(parent, t0, below)
    # y-edge (r, c)-(r + 1, c): sub 1 against p00 of the cell to the left.
    if c > 0:
        left = t0 - 2
        if active[t1] != 0 and active[left] != 0:
            if _flat_edge(grid, t1, v00 - 1):
                _union(parent, t1, left)


@wp.kernel(enable_backward=False)
def _cluster_compress(active: wp.array[wp.int32], parent: wp.array[wp.int32]):
    """Point every active triangle at its component's root, storing into its own slot only.

    A load of another triangle's slot returns its parent from the union pass or the root its own thread stored;
    either leads to the root. A path-halving store into another triangle's slot, computed from its old parent,
    could land after that triangle's thread stored the root (it did on CUDA) and leave a non-root parent, which
    splits a component's statistics over several ids and indexes scratch that no root reset.
    """
    t = wp.tid()
    if active[t] != 0:
        parent[t] = _root(parent, t)


@wp.kernel(enable_backward=False)
def _cluster_check_roots(active: wp.array[wp.int32], parent: wp.array[wp.int32], counts: wp.array[wp.int32]):
    """Count in ``counts[1]`` the active triangles whose parent is not a root after compression (none, unless raced)."""
    t = wp.tid()
    if active[t] != 0:
        p = parent[t]
        if parent[p] != p:
            wp.atomic_add(counts, 1, 1)


@wp.kernel(enable_backward=False)
def _cluster_reset_roots(
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    vb: wp.array[wp.int32],
    vc: wp.array[wp.int32],
    best_d: wp.array[wp.int32],
    best_a: wp.array[wp.int32],
    rmin: wp.array[wp.float32],
    rmax: wp.array[wp.float32],
    hmin: wp.array[wp.float32],
    hmax: wp.array[wp.float32],
    count_e: wp.array[wp.int32],
    count_s: wp.array[wp.int32],
    count_h: wp.array[wp.int32],
    name: wp.array[wp.int32],
):
    t = wp.tid()
    if active[t] != 0 and parent[t] == t:
        va[t] = _INT_MAX
        vb[t] = _INT_MAX
        vc[t] = _INT_MAX
        best_d[t] = -1
        best_a[t] = -1
        rmin[t] = _FLOAT_MAX
        rmax[t] = -_FLOAT_MAX
        hmin[t] = _FLOAT_MAX
        hmax[t] = -_FLOAT_MAX
        count_e[t] = 0
        count_s[t] = 0
        count_h[t] = 0
        name[t] = _INT_MAX


@wp.kernel(enable_backward=False)
def _cluster_extreme_a(grid: _Grid, active: wp.array[wp.int32], parent: wp.array[wp.int32], va: wp.array[wp.int32]):
    t = wp.tid()
    if active[t] != 0:
        # The lowest-id vertex of a triangle is its p00.
        wp.atomic_min(va, parent[t], _triangle_vertex(grid, t, 0))


@wp.func
def _grid_d2(grid: _Grid, v: int, a: int) -> int:
    vr = v // grid.ncol
    ar = a // grid.ncol
    dr = vr - ar
    dc = (v - vr * grid.ncol) - (a - ar * grid.ncol)
    return dr * dr + dc * dc


@wp.func
def _grid_area(grid: _Grid, v: int, a: int, b: int) -> int:
    vr = v // grid.ncol
    ar = a // grid.ncol
    br = b // grid.ncol
    dcb = (b - br * grid.ncol) - (a - ar * grid.ncol)
    drb = br - ar
    dcv = (v - vr * grid.ncol) - (a - ar * grid.ncol)
    drv = vr - ar
    return wp.abs(dcb * drv - drb * dcv)


@wp.kernel(enable_backward=False)
def _cluster_extreme_b_distance(
    grid: _Grid,
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    best_d: wp.array[wp.int32],
):
    t = wp.tid()
    if active[t] != 0:
        root = parent[t]
        for j in range(3):
            wp.atomic_max(best_d, root, _grid_d2(grid, _triangle_vertex(grid, t, j), va[root]))


@wp.kernel(enable_backward=False)
def _cluster_extreme_b(
    grid: _Grid,
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    best_d: wp.array[wp.int32],
    vb: wp.array[wp.int32],
):
    t = wp.tid()
    if active[t] != 0:
        root = parent[t]
        for j in range(3):
            v = _triangle_vertex(grid, t, j)
            if _grid_d2(grid, v, va[root]) == best_d[root]:
                wp.atomic_min(vb, root, v)


@wp.kernel(enable_backward=False)
def _cluster_extreme_c_area(
    grid: _Grid,
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    vb: wp.array[wp.int32],
    best_a: wp.array[wp.int32],
):
    t = wp.tid()
    if active[t] != 0:
        root = parent[t]
        for j in range(3):
            wp.atomic_max(best_a, root, _grid_area(grid, _triangle_vertex(grid, t, j), va[root], vb[root]))


@wp.kernel(enable_backward=False)
def _cluster_extreme_c(
    grid: _Grid,
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    vb: wp.array[wp.int32],
    best_a: wp.array[wp.int32],
    vc: wp.array[wp.int32],
):
    t = wp.tid()
    if active[t] != 0:
        root = parent[t]
        for j in range(3):
            v = _triangle_vertex(grid, t, j)
            if _grid_area(grid, v, va[root], vb[root]) == best_a[root]:
                wp.atomic_min(vc, root, v)


@wp.func
def _plane_gradient(grid: _Grid, a: int, b: int, c: int) -> wp.vec2:
    """Gradient (gx, gy) of the plane through vertices a, b and c, rounded to float32 (Cramer's rule, float64)."""
    ob = _offset(grid, b, a)
    oc = _offset(grid, c, a)
    det = ob[0] * oc[1] - ob[1] * oc[0]
    gx = (ob[2] * oc[1] - ob[1] * oc[2]) / det
    gy = (ob[0] * oc[2] - ob[2] * oc[0]) / det
    return wp.vec2(wp.float32(gx), wp.float32(gy))


@wp.kernel(enable_backward=False)
def _cluster_root_planes(
    grid: _Grid,
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    vb: wp.array[wp.int32],
    vc: wp.array[wp.int32],
    best_a: wp.array[wp.int32],
    gradient: wp.array[wp.vec2],
    seed_gradient: wp.array[wp.vec2],
):
    t = wp.tid()
    if active[t] != 0 and parent[t] == t:
        if best_a[t] > 0:
            gradient[t] = _plane_gradient(grid, va[t], vb[t], vc[t])
        else:
            gradient[t] = wp.vec2(0.0, 0.0)
        seed_gradient[t] = _plane_gradient(
            grid, _triangle_vertex(grid, t, 0), _triangle_vertex(grid, t, 1), _triangle_vertex(grid, t, 2)
        )


@wp.func
def _residual(grid: _Grid, v: int, anchor: int, g: wp.vec2) -> float:
    """Vertical residual of vertex v from the plane z = z(anchor) + g . (x - x_a), rounded to float32."""
    o = _offset(grid, v, anchor)
    return wp.float32(o[2] - wp.float64(g[0]) * o[0] - wp.float64(g[1]) * o[1])


@wp.func
def _normal_scale(g: wp.vec2) -> wp.float64:
    gx = wp.float64(g[0])
    gy = wp.float64(g[1])
    return wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + gx * gx + gy * gy)


@wp.kernel(enable_backward=False)
def _cluster_residual_range(
    grid: _Grid,
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    gradient: wp.array[wp.vec2],
    rmin: wp.array[wp.float32],
    rmax: wp.array[wp.float32],
    hmin: wp.array[wp.float32],
    hmax: wp.array[wp.float32],
):
    t = wp.tid()
    if active[t] != 0:
        root = parent[t]
        for j in range(3):
            v = _triangle_vertex(grid, t, j)
            res = _residual(grid, v, va[root], gradient[root])
            wp.atomic_min(rmin, root, res)
            wp.atomic_max(rmax, root, res)
            level = wp.float32(_z(grid, v) - _z(grid, va[root]))
            wp.atomic_min(hmin, root, level)
            wp.atomic_max(hmax, root, level)


@wp.func
def _midrange(rmin: float, rmax: float) -> float:
    return wp.float32(wp.float64(0.5) * (wp.float64(rmin) + wp.float64(rmax)))


_PLANE_SEED = wp.constant(1)
_PLANE_EXTREME = wp.constant(2)
_PLANE_LEVEL = wp.constant(4)


@wp.func
def _choice(
    rmin: float,
    rmax: float,
    g: wp.vec2,
    best_area: int,
    hmin: float,
    hmax: float,
    count_e: int,
    count_s: int,
    count_h: int,
) -> int:
    """The plane a component's cluster takes: its closer planar candidate, else the one claiming the most."""
    half_h = wp.float64(0.5) * (wp.float64(hmax) - wp.float64(hmin))
    half_e = wp.float64(_FLOAT_MAX)
    if best_area > 0:
        half_e = wp.float64(0.5) * (wp.float64(rmax) - wp.float64(rmin)) * _normal_scale(g)
    if wp.min(half_e, half_h) <= wp.float64(_FLAT_EPS):
        if half_e <= half_h:
            return _PLANE_EXTREME
        return _PLANE_LEVEL
    plane = _PLANE_SEED
    claimed = count_s
    if count_e > claimed:
        plane = _PLANE_EXTREME
        claimed = count_e
    if count_h > claimed:
        plane = _PLANE_LEVEL
    return plane


@wp.kernel(enable_backward=False)
def _cluster_members(
    grid: _Grid,
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    gradient: wp.array[wp.vec2],
    seed_gradient: wp.array[wp.vec2],
    rmin: wp.array[wp.float32],
    rmax: wp.array[wp.float32],
    hmin: wp.array[wp.float32],
    hmax: wp.array[wp.float32],
    count_e: wp.array[wp.int32],
    count_s: wp.array[wp.int32],
    count_h: wp.array[wp.int32],
    member: wp.array[wp.int32],
):
    """Count the triangles each candidate plane claims and record the claims (seed, extreme and level bits)."""
    t = wp.tid()
    if active[t] == 0:
        return
    root = parent[t]
    anchor = va[root]
    g = gradient[root]
    scale = _normal_scale(g)
    offset = _midrange(rmin[root], rmax[root])
    level_offset = _midrange(hmin[root], hmax[root])
    sg = seed_gradient[root]
    seed_scale = _normal_scale(sg)
    seed_anchor = _triangle_vertex(grid, root, 0)
    extreme = bool(True)
    level = bool(True)
    seed = bool(True)
    for j in range(3):
        v = _triangle_vertex(grid, t, j)
        res = _residual(grid, v, anchor, g)
        deviation = wp.float32(wp.float64(res) - wp.float64(offset))
        if wp.float64(wp.abs(deviation)) * scale > wp.float64(_FLAT_EPS):
            extreme = False
        height = wp.float32(_z(grid, v) - _z(grid, anchor))
        if wp.float64(wp.abs(wp.float32(wp.float64(height) - wp.float64(level_offset)))) > wp.float64(_FLAT_EPS):
            level = False
        seed_res = _residual(grid, v, seed_anchor, sg)
        if wp.float64(wp.abs(seed_res)) * seed_scale > wp.float64(_FLAT_EPS):
            seed = False
    flags = int(0)
    if seed:
        flags = flags | _PLANE_SEED
        wp.atomic_add(count_s, root, 1)
    if extreme:
        flags = flags | _PLANE_EXTREME
        wp.atomic_add(count_e, root, 1)
    if level:
        flags = flags | _PLANE_LEVEL
        wp.atomic_add(count_h, root, 1)
    member[t] = flags


@wp.kernel(enable_backward=False)
def _cluster_names(
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    gradient: wp.array[wp.vec2],
    best_a: wp.array[wp.int32],
    rmin: wp.array[wp.float32],
    rmax: wp.array[wp.float32],
    hmin: wp.array[wp.float32],
    hmax: wp.array[wp.float32],
    count_e: wp.array[wp.int32],
    count_s: wp.array[wp.int32],
    count_h: wp.array[wp.int32],
    member: wp.array[wp.int32],
    name: wp.array[wp.int32],
):
    t = wp.tid()
    if active[t] == 0:
        return
    root = parent[t]
    plane = _choice(
        rmin[root],
        rmax[root],
        gradient[root],
        best_a[root],
        hmin[root],
        hmax[root],
        count_e[root],
        count_s[root],
        count_h[root],
    )
    if (member[t] & plane) != 0:
        wp.atomic_min(name, root, t)


@wp.kernel(enable_backward=False)
def _cluster_assign(
    grid: _Grid,
    active: wp.array[wp.int32],
    parent: wp.array[wp.int32],
    va: wp.array[wp.int32],
    gradient: wp.array[wp.vec2],
    seed_gradient: wp.array[wp.vec2],
    best_a: wp.array[wp.int32],
    rmin: wp.array[wp.float32],
    rmax: wp.array[wp.float32],
    hmin: wp.array[wp.float32],
    hmax: wp.array[wp.float32],
    count_e: wp.array[wp.int32],
    count_s: wp.array[wp.int32],
    count_h: wp.array[wp.int32],
    member: wp.array[wp.int32],
    name: wp.array[wp.int32],
    label: wp.array[wp.int32],
    records: wp.array[TerrainClusterPlane],
    counts: wp.array[wp.int32],
):
    """Assign the triangles their component's chosen plane claims; count the others in ``counts[0]``."""
    t = wp.tid()
    if active[t] == 0:
        return
    root = parent[t]
    plane = _choice(
        rmin[root],
        rmax[root],
        gradient[root],
        best_a[root],
        hmin[root],
        hmax[root],
        count_e[root],
        count_s[root],
        count_h[root],
    )
    if (member[t] & plane) == 0:
        wp.atomic_add(counts, 0, 1)
        return
    cluster = name[root]
    label[t] = cluster
    active[t] = 0
    if t == cluster:
        record = TerrainClusterPlane()
        if plane == _PLANE_EXTREME:
            g = gradient[root]
            record.anchor = va[root]
            record.gx = g[0]
            record.gy = g[1]
            record.offset = _midrange(rmin[root], rmax[root])
        elif plane == _PLANE_LEVEL:
            record.anchor = va[root]
            record.gx = 0.0
            record.gy = 0.0
            record.offset = _midrange(hmin[root], hmax[root])
        else:
            g = seed_gradient[root]
            record.anchor = _triangle_vertex(grid, root, 0)
            record.gx = g[0]
            record.gy = g[1]
            record.offset = 0.0
        records[t] = record


@wp.kernel(enable_backward=False)
def _cluster_leftovers(active: wp.array[wp.int32], label: wp.array[wp.int32], size: wp.array[wp.int32]):
    t = wp.tid()
    if active[t] != 0:
        label[t] = t
        active[t] = 0
    wp.atomic_add(size, label[t], 1)


@wp.kernel(enable_backward=False)
def _cluster_flag_records(label: wp.array[wp.int32], size: wp.array[wp.int32], flag: wp.array[wp.int32]):
    t = wp.tid()
    flag[t] = int(label[t] == t and size[t] > 1)


@wp.kernel(enable_backward=False)
def _cluster_write(
    label: wp.array[wp.int32],
    size: wp.array[wp.int32],
    dense: wp.array[wp.int32],
    records: wp.array[TerrainClusterPlane],
    clusters: wp.array[wp.int32],
    cluster_offset: int,
    planes: wp.array[TerrainClusterPlane],
    plane_offset: int,
):
    t = wp.tid()
    name = label[t]
    if size[name] > 1:
        clusters[cluster_offset + t] = dense[name]
        if name == t:
            planes[plane_offset + dense[t]] = records[t]
    else:
        clusters[cluster_offset + t] = -(1 + t)


@wp.kernel(enable_backward=False)
def _block_max(
    grid: _Grid, side: int, block_r0: int, block_c0: int, block_cols: int, block_offset: int, out: wp.array[wp.float32]
):
    """The largest decoded height of the vertices of every block of ``side`` x ``side`` cells (a mip level)."""
    i, j = wp.tid()
    br = block_r0 + i
    bc = block_c0 + j
    r_end = wp.min((br + 1) * side, grid.nrow - 1)
    c_end = wp.min((bc + 1) * side, grid.ncol - 1)
    value = float(-_FLOAT_MAX)
    for r in range(br * side, r_end + 1):
        for c in range(bc * side, c_end + 1):
            e = grid.elevations[grid.data_offset + r * grid.ncol + c]
            value = wp.max(value, terrain_decode(grid.min_z, grid.max_z, e))
    out[block_offset + br * block_cols + bc] = value


@wp.kernel(enable_backward=False)
def _vertex_bits(data: TerrainFeatureData, heightfield: int, bits: wp.array[wp.uint8]):
    """Every vertex's bits of one heightfield (:func:`~newton._src.geometry.terrain_law.terrain_vertex_bits_of_clusters`,
    from its clusters and heights, which ``data`` holds), one thread per vertex."""
    r, c = wp.tid()
    window = terrain_window(data, heightfield, r, c)
    surface = window.surface
    bits[surface.data_offset + r * surface.ncol + c] = wp.uint8(terrain_vertex_bits_of_clusters(data, window, r, c))


@wp.kernel(enable_backward=False)
def _bump_version(version: wp.array[wp.int32], index: int):
    version[index] = version[index] + 1


@wp.kernel(enable_backward=False)
def _copy_version(source: wp.array[wp.int32], target: wp.array[wp.int32], index: int):
    target[index] = source[index]


# ----------------------------------------------------------------------------------------------------------------
# surface features
# ----------------------------------------------------------------------------------------------------------------


class _ClusterScratch:
    """Per-triangle scratch of one cluster build (freed afterwards)."""

    def __init__(self, count: int, device):
        def ints():
            return wp.empty(count, dtype=wp.int32, device=device)

        self.active, self.label, self.parent = ints(), ints(), ints()
        self.va, self.vb, self.vc = ints(), ints(), ints()
        self.best_d, self.best_a = ints(), ints()
        self.count_e, self.count_s, self.count_h, self.member, self.name = ints(), ints(), ints(), ints(), ints()
        self.rmin = wp.empty(count, dtype=wp.float32, device=device)
        self.rmax = wp.empty(count, dtype=wp.float32, device=device)
        self.hmin = wp.empty(count, dtype=wp.float32, device=device)
        self.hmax = wp.empty(count, dtype=wp.float32, device=device)
        self.gradient = wp.empty(count, dtype=wp.vec2, device=device)
        self.seed_gradient = wp.empty(count, dtype=wp.vec2, device=device)
        self.records = wp.empty(count, dtype=TerrainClusterPlane, device=device)
        # Per round: triangles left unassigned, and triangles under a non-root parent after compression.
        self.counts = wp.zeros(2, dtype=wp.int32, device=device)


def _cluster_bytes(count: int) -> int:
    """Peak scratch bytes of one cluster build of ``count`` triangles."""
    return count * (14 * 4 + 4 * 4 + 2 * 8 + 16)


def _merge_regions(a, b):
    """Bounding box of two vertex regions ``(row_begin, row_end, col_begin, col_end)``, either of them ``None``."""
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


class TerrainFeatures:
    """Plane clusters, tile-max mip and version stamps of every heightfield of a model.

    Built on the model's device by Warp kernels, deterministically. The law's kernels read the arrays through
    :meth:`data`. Edited heights are reported with :meth:`mark_elevations_changed` and absorbed by
    :meth:`rebuild_terrain_features`; until then the version stamps differ and a kernel that checks them
    (:func:`~newton._src.geometry.terrain_law.terrain_features_current`) reports stale data instead of
    reading it. So does a kernel given a struct from :meth:`data` that predates a rebuild which reallocated the
    plane records.

    Args:
        model: A finalized model. Its heightfield elevations are read in place, never copied.
    """

    def __init__(self, model):
        self.model = model
        self.device = model.device
        types = model.shape_type.numpy() if model.shape_count else np.zeros(0, np.int32)
        index = model.shape_heightfield_index.numpy() if model.shape_count else np.zeros(0, np.int32)
        self.heightfield_shapes = [int(s) for s in np.flatnonzero(types == int(GeoType.HFIELD)) if index[s] >= 0]
        self.heightfield_of_shape = {s: int(index[s]) for s in self.heightfield_shapes}
        count = len(self.heightfield_shapes)
        self.heightfield_count = count
        layouts = model.heightfield_data.numpy() if count else None
        self._grids = []
        infos = []
        triangle_offset = 0
        tile_offset = 0
        fine_offset = 0
        self._tiles = []
        self._fines = []
        for h in range(count):
            row = layouts[h]
            nrow, ncol = int(row["nrow"]), int(row["ncol"])
            hx, hy = float(row["hx"]), float(row["hy"])
            grid = _Grid()
            grid.elevations = model.heightfield_elevations
            grid.data_offset = int(row["data_offset"])
            grid.nrow, grid.ncol = nrow, ncol
            grid.min_z, grid.max_z = float(row["min_z"]), float(row["max_z"])
            grid.dx = 2.0 * hx / (ncol - 1)
            grid.dy = 2.0 * hy / (nrow - 1)
            self._grids.append(grid)
            tile_rows = -(-(nrow - 1) // TERRAIN_TILE_CELLS)
            tile_cols = -(-(ncol - 1) // TERRAIN_TILE_CELLS)
            self._tiles.append((tile_rows, tile_cols))
            fine_rows = -(-(nrow - 1) // TERRAIN_FINE_CELLS)
            fine_cols = -(-(ncol - 1) // TERRAIN_FINE_CELLS)
            self._fines.append((fine_rows, fine_cols))
            info = TerrainSurfaceInfo()
            info.data_offset = grid.data_offset
            info.nrow, info.ncol = nrow, ncol
            info.hx, info.hy = hx, hy
            info.min_z, info.max_z = grid.min_z, grid.max_z
            info.dx = float(np.float32(np.float32(2.0) * np.float32(hx) / np.float32(ncol - 1)))
            info.dy = float(np.float32(np.float32(2.0) * np.float32(hy) / np.float32(nrow - 1)))
            # the device's float64 steps of the float32 half extents, bit for bit (IEEE float64 on both)
            info.dx64 = float(np.float64(2.0) * np.float64(np.float32(hx)) / np.float64(ncol - 1))
            info.dy64 = float(np.float64(2.0) * np.float64(np.float32(hy)) / np.float64(nrow - 1))
            info.triangle_offset = triangle_offset
            info.plane_offset = 0
            info.tile_offset = tile_offset
            info.tile_cols = tile_cols
            info.fine_offset = fine_offset
            info.fine_cols = fine_cols
            infos.append(info)
            triangle_offset += 2 * (nrow - 1) * (ncol - 1)
            tile_offset += tile_rows * tile_cols
            fine_offset += fine_rows * fine_cols
        self._infos = infos
        self.clusters = wp.zeros(max(triangle_offset, 1), dtype=wp.int32, device=self.device)
        self.tile_max = wp.zeros(max(tile_offset, 1), dtype=wp.float32, device=self.device)
        self.fine_max = wp.zeros(max(fine_offset, 1), dtype=wp.float32, device=self.device)
        self.vertex_bits = wp.zeros(
            max(model.heightfield_elevations.shape[0] if count else 0, 1), dtype=wp.uint8, device=self.device
        )
        self.elevation_version = wp.zeros(max(count, 1), dtype=wp.int32, device=self.device)
        self.feature_version = wp.zeros(max(count, 1), dtype=wp.int32, device=self.device)
        self._dirty = [None] * count
        self._plane_counts = [0] * count
        self._plane_capacity = [0] * count
        self.planes = wp.zeros(1, dtype=TerrainClusterPlane, device=self.device)
        self._surfaces = wp.array(
            self._infos if count else [TerrainSurfaceInfo()], dtype=TerrainSurfaceInfo, device=self.device
        )
        self.layout_version = 0
        """Incremented whenever a build reallocates the plane records: fetch :meth:`data` again."""
        self._layout = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.cluster_build_seconds = 0.0
        """Host time spent building clusters so far [s] (device synchronized)."""
        self.cluster_scratch_bytes = 0
        """Largest device scratch of one cluster build [bytes]."""
        built = [self._build_clusters(h) for h in range(count)]
        self._place_planes(built)
        for h in range(count):
            self._build_vertex_bits(h)
            self._build_tiles(h, 0, self._tiles[h][0], 0, self._tiles[h][1])
            self._build_fines(h, 0, self._fines[h][0], 0, self._fines[h][1])

    # ---- queries ---------------------------------------------------------------------------------------------
    def data(self) -> TerrainFeatureData:
        """Return the arrays the law's kernels read.

        A rebuild that reallocates the plane records (:attr:`layout_version` changes) leaves a struct fetched
        before it, or baked into a captured graph, pointing at the old records: kernels that check
        :func:`~newton._src.geometry.terrain_law.terrain_features_current` then report stale features instead of
        reading them. Fetch the struct again, and capture again, after such a rebuild.
        """
        data = TerrainFeatureData()
        data.elevations = self.model.heightfield_elevations
        data.clusters = self.clusters
        data.planes = self.planes
        data.tile_max = self.tile_max
        data.fine_max = self.fine_max
        data.vertex_bits = self.vertex_bits
        data.surfaces = self._surfaces
        data.elevation_version = self.elevation_version
        data.feature_version = self.feature_version
        data.layout = self.layout_version
        data.layout_version = self._layout
        return data

    def surface(self, shape: int) -> TerrainSurfaceInfo:
        """Return the layout of a heightfield shape."""
        return self._infos[self._heightfield(shape)]

    def plane_count(self, shape: int) -> int:
        """Return the number of plane records (clusters of several triangles) of a heightfield shape."""
        return self._plane_counts[self._heightfield(shape)]

    def memory_bytes(self) -> dict[str, int]:
        """Return the device bytes the features hold, by array."""
        return {
            "clusters": self.clusters.capacity,
            "planes": self.planes.capacity,
            "tile_max": self.tile_max.capacity,
            "fine_max": self.fine_max.capacity,
            "vertex_bits": self.vertex_bits.capacity,
            "surfaces": self._surfaces.capacity,
            "versions": self.elevation_version.capacity + self.feature_version.capacity + self._layout.capacity,
        }

    def check_current(self) -> None:
        """Raise at a safe host boundary if any heightfield's features are older than its heights."""
        if not self.heightfield_count:
            return
        if self.device.is_capturing:
            raise RuntimeError("Terrain feature versions must be read outside graph capture")
        heights = self.elevation_version.numpy()[: self.heightfield_count]
        features = self.feature_version.numpy()[: self.heightfield_count]
        stale = [self.heightfield_shapes[h] for h in np.flatnonzero(heights != features)]
        if stale:
            raise RuntimeError(f"Terrain features are older than the heights of heightfield shapes {stale}")

    # ---- edits -----------------------------------------------------------------------------------------------
    def mark_elevations_changed(self, shape: int, *, region: tuple[int, int, int, int] | None = None) -> None:
        """Report edited heights of a heightfield shape; its features are stale until rebuilt.

        Args:
            shape: The heightfield's shape index.
            region: Edited vertices as ``(row_begin, row_end, col_begin, col_end)`` (ends exclusive), or ``None``
                for the whole heightfield.
        """
        h = self._heightfield(shape)
        self._dirty[h] = _merge_regions(self._dirty[h], self._region(h, region))
        wp.launch(_bump_version, dim=1, inputs=[self.elevation_version, h], device=self.device, record_tape=False)

    def rebuild_terrain_features(self, shape: int, *, region: tuple[int, int, int, int] | None = None) -> None:
        """Rebuild a heightfield shape's clusters and mip after its heights changed, and mark them current.

        The clusters of the whole heightfield are rebuilt, so the result equals a fresh build: a cluster's plane
        and membership depend on its lowest triangle and extreme vertices, which an edit may move far from the
        edited cells. The mip is rebuilt for the tiles over the bounding box of ``region`` and of every region
        reported since the last rebuild, so that no tile is marked current over stale heights.

        Args:
            shape: The heightfield's shape index.
            region: Edited vertices as ``(row_begin, row_end, col_begin, col_end)`` (ends exclusive), added to the
                regions reported since the last rebuild; with neither, the whole heightfield.
        """
        h = self._heightfield(shape)
        if self.device.is_capturing:
            raise RuntimeError("Terrain features must be rebuilt outside graph capture")
        region = _merge_regions(self._dirty[h], None if region is None else self._region(h, region))
        if region is None:
            region = self._region(h, None)
        built = [None] * self.heightfield_count
        built[h] = self._build_clusters(h)
        self._place_planes(built)
        # every cluster may have changed: every vertex's bits
        self._build_vertex_bits(h)
        r0, r1, c0, c1 = region
        info = self._infos[h]
        rows, cols = info.nrow - 1, info.ncol - 1
        tr0 = max(r0 - 1, 0) // TERRAIN_TILE_CELLS
        tr1 = min(max(r1 - 1, 0), rows - 1) // TERRAIN_TILE_CELLS + 1
        tc0 = max(c0 - 1, 0) // TERRAIN_TILE_CELLS
        tc1 = min(max(c1 - 1, 0), cols - 1) // TERRAIN_TILE_CELLS + 1
        self._build_tiles(h, tr0, tr1, tc0, tc1)
        fr0 = max(r0 - 1, 0) // TERRAIN_FINE_CELLS
        fr1 = min(max(r1 - 1, 0), rows - 1) // TERRAIN_FINE_CELLS + 1
        fc0 = max(c0 - 1, 0) // TERRAIN_FINE_CELLS
        fc1 = min(max(c1 - 1, 0), cols - 1) // TERRAIN_FINE_CELLS + 1
        self._build_fines(h, fr0, fr1, fc0, fc1)
        self._dirty[h] = None
        wp.launch(
            _copy_version,
            dim=1,
            inputs=[self.elevation_version, self.feature_version, h],
            device=self.device,
            record_tape=False,
        )

    # ---- build -----------------------------------------------------------------------------------------------
    def _heightfield(self, shape: int) -> int:
        if shape not in self.heightfield_of_shape:
            raise ValueError(f"Shape {shape} is not a heightfield")
        return self.heightfield_of_shape[shape]

    def _region(self, h: int, region):
        info = self._infos[h]
        if region is None:
            return (0, info.nrow, 0, info.ncol)
        r0, r1, c0, c1 = (int(v) for v in region)
        if not (0 <= r0 < r1 <= info.nrow and 0 <= c0 < c1 <= info.ncol):
            raise ValueError(f"Region {region} is outside the {info.nrow} x {info.ncol} heightfield")
        return (r0, r1, c0, c1)

    def _build_clusters(self, h: int):
        """Build heightfield h's clusters into ``self.clusters``; return its plane records (device) and their count."""
        started = time.perf_counter()
        grid = self._grids[h]
        info = self._infos[h]
        device = self.device
        cells = (grid.nrow - 1) * (grid.ncol - 1)
        count = 2 * cells
        s = _ClusterScratch(count, device)
        self.cluster_scratch_bytes = max(self.cluster_scratch_bytes, _cluster_bytes(count))

        def launch(kernel, dim, inputs):
            wp.launch(kernel, dim=dim, inputs=inputs, device=device, record_tape=False)

        launch(_cluster_init, count, [s.active, s.label])
        for _round in range(TERRAIN_CLUSTER_ROUNDS):
            s.counts.zero_()
            launch(_cluster_parents, count, [s.parent])
            launch(_cluster_union, cells, [grid, s.active, s.parent])
            launch(_cluster_compress, count, [s.active, s.parent])
            launch(_cluster_check_roots, count, [s.active, s.parent, s.counts])
            launch(
                _cluster_reset_roots,
                count,
                [
                    s.active,
                    s.parent,
                    s.va,
                    s.vb,
                    s.vc,
                    s.best_d,
                    s.best_a,
                    s.rmin,
                    s.rmax,
                    s.hmin,
                    s.hmax,
                    s.count_e,
                    s.count_s,
                    s.count_h,
                    s.name,
                ],
            )
            launch(_cluster_extreme_a, count, [grid, s.active, s.parent, s.va])
            launch(_cluster_extreme_b_distance, count, [grid, s.active, s.parent, s.va, s.best_d])
            launch(_cluster_extreme_b, count, [grid, s.active, s.parent, s.va, s.best_d, s.vb])
            launch(_cluster_extreme_c_area, count, [grid, s.active, s.parent, s.va, s.vb, s.best_a])
            launch(_cluster_extreme_c, count, [grid, s.active, s.parent, s.va, s.vb, s.best_a, s.vc])
            launch(
                _cluster_root_planes,
                count,
                [grid, s.active, s.parent, s.va, s.vb, s.vc, s.best_a, s.gradient, s.seed_gradient],
            )
            launch(
                _cluster_residual_range,
                count,
                [grid, s.active, s.parent, s.va, s.gradient, s.rmin, s.rmax, s.hmin, s.hmax],
            )
            launch(
                _cluster_members,
                count,
                [
                    grid,
                    s.active,
                    s.parent,
                    s.va,
                    s.gradient,
                    s.seed_gradient,
                    s.rmin,
                    s.rmax,
                    s.hmin,
                    s.hmax,
                    s.count_e,
                    s.count_s,
                    s.count_h,
                    s.member,
                ],
            )
            launch(
                _cluster_names,
                count,
                [
                    s.active,
                    s.parent,
                    s.gradient,
                    s.best_a,
                    s.rmin,
                    s.rmax,
                    s.hmin,
                    s.hmax,
                    s.count_e,
                    s.count_s,
                    s.count_h,
                    s.member,
                    s.name,
                ],
            )
            launch(
                _cluster_assign,
                count,
                [
                    grid,
                    s.active,
                    s.parent,
                    s.va,
                    s.gradient,
                    s.seed_gradient,
                    s.best_a,
                    s.rmin,
                    s.rmax,
                    s.hmin,
                    s.hmax,
                    s.count_e,
                    s.count_s,
                    s.count_h,
                    s.member,
                    s.name,
                    s.label,
                    s.records,
                    s.counts,
                ],
            )
            remaining, misparented = (int(n) for n in s.counts.numpy())
            if misparented:
                raise RuntimeError(
                    f"The cluster build of heightfield shape {self.heightfield_shapes[h]} left {misparented} "
                    f"triangles under a parent that is not their component's root on {device}"
                )
            if remaining == 0:
                break
        size = s.parent
        size.zero_()
        launch(_cluster_leftovers, count, [s.active, s.label, size])
        flag = s.va
        launch(_cluster_flag_records, count, [s.label, size, flag])
        dense = s.vb
        wp.utils.array_scan(flag, dense, inclusive=False)
        records_count = int(dense[count - 1 : count].numpy()[0] + flag[count - 1 : count].numpy()[0])
        planes = wp.zeros(max(records_count, 1), dtype=TerrainClusterPlane, device=device)
        launch(
            _cluster_write,
            count,
            [s.label, size, dense, s.records, self.clusters, info.triangle_offset, planes, 0],
        )
        wp.synchronize_device(device)
        self.cluster_build_seconds += time.perf_counter() - started
        return planes, records_count

    def _place_planes(self, built) -> None:
        """Lay the plane records of every heightfield out in ``self.planes`` (reusing the array when they fit)."""
        counts = list(self._plane_counts)
        for h, item in enumerate(built):
            if item is not None:
                counts[h] = item[1]
        fits = all(counts[h] <= self._plane_capacity[h] for h in range(self.heightfield_count))
        if not fits:
            capacity = [max(c + max(c // 4, 64), self._plane_capacity[h]) for h, c in enumerate(counts)]
            old_planes, old_offsets = self.planes, [i.plane_offset for i in self._infos]
            self.planes = wp.zeros(max(sum(capacity), 1), dtype=TerrainClusterPlane, device=self.device)
            offset = 0
            for h in range(self.heightfield_count):
                if built[h] is None and counts[h]:
                    wp.copy(self.planes, old_planes, dest_offset=offset, src_offset=old_offsets[h], count=counts[h])
                self._infos[h].plane_offset = offset
                offset += capacity[h]
            self._plane_capacity = capacity
            self.layout_version += 1
            # Structs fetched before now point at the old records and read as stale on the device.
            self._layout.fill_(self.layout_version)
        for h, item in enumerate(built):
            if item is not None and item[1]:
                wp.copy(self.planes, item[0], dest_offset=self._infos[h].plane_offset, count=item[1])
        self._plane_counts = counts
        if self.heightfield_count:
            self._surfaces.assign(wp.array(self._infos, dtype=TerrainSurfaceInfo, device="cpu"))

    def _build_vertex_bits(self, h: int) -> None:
        info = self._infos[h]
        wp.launch(
            _vertex_bits,
            dim=(info.nrow, info.ncol),
            inputs=[self.data(), h, self.vertex_bits],
            device=self.device,
            record_tape=False,
        )

    def _build_tiles(self, h: int, tr0: int, tr1: int, tc0: int, tc1: int) -> None:
        if tr1 <= tr0 or tc1 <= tc0:
            return
        info = self._infos[h]
        wp.launch(
            _block_max,
            dim=(tr1 - tr0, tc1 - tc0),
            inputs=[self._grids[h], TERRAIN_TILE_CELLS, tr0, tc0, info.tile_cols, info.tile_offset, self.tile_max],
            device=self.device,
            record_tape=False,
        )

    def _build_fines(self, h: int, fr0: int, fr1: int, fc0: int, fc1: int) -> None:
        if fr1 <= fr0 or fc1 <= fc0:
            return
        info = self._infos[h]
        wp.launch(
            _block_max,
            dim=(fr1 - fr0, fc1 - fc0),
            inputs=[self._grids[h], TERRAIN_FINE_CELLS, fr0, fc0, info.fine_cols, info.fine_offset, self.fine_max],
            device=self.device,
            record_tape=False,
        )


# ----------------------------------------------------------------------------------------------------------------
# terrain pair table
# ----------------------------------------------------------------------------------------------------------------


def heightfield_pair_table(
    shape_types: np.ndarray,
    shape_world: np.ndarray | None,
    colliding: np.ndarray,
    *,
    explicit_pairs: np.ndarray | None = None,
    shape_collision_group: np.ndarray | None = None,
    filter_mask: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Return every ``(heightfield, partner)`` shape pair the collision filter admits, once.

    The pairs are those of exactly one heightfield and a partner other than a triangle mesh. With an explicit
    pair list (an empty one included) they are its pairs of that kind, in either order. Otherwise the filter is
    applied here as the model builder applies it: every colliding heightfield with every colliding partner of a
    compatible world (the same world, or either shape global) whose collision groups interact and that no filter
    pair excludes.

    Args:
        shape_types: Geometry type of every shape.
        shape_world: World of every shape (-1 for global shapes), or ``None`` for a single world.
        colliding: Whether every shape takes part in shape-shape collision.
        explicit_pairs: The model's admitted shape pairs, or ``None`` to apply the filter here.
        shape_collision_group: Collision group of every shape, or ``None`` when every group interacts. Unused
            with ``explicit_pairs``.
        filter_mask: Returns, for ``(pair_count, 2)`` shape pairs, which ones a filter pair excludes
            (:meth:`~newton.Model.shape_collision_filter_mask`), or ``None`` without filter pairs. Unused with
            ``explicit_pairs``.

    Returns:
        Unique ``(heightfield, partner)`` rows, sorted.
    """
    shape_types = np.asarray(shape_types)
    hfield, mesh = int(GeoType.HFIELD), int(GeoType.MESH)
    if explicit_pairs is not None:
        pairs = np.asarray(explicit_pairs, dtype=np.int64).reshape(-1, 2)
        pairs = pairs[np.all((pairs >= 0) & (pairs < len(shape_types)), axis=1)]
        first, second = shape_types[pairs[:, 0]], shape_types[pairs[:, 1]]
        partner = np.where(first == hfield, second, first)
        pairs = pairs[((first == hfield) != (second == hfield)) & (partner != mesh)]
        swap = shape_types[pairs[:, 0]] != hfield
        table = np.where(swap[:, None], pairs[:, ::-1], pairs)
    else:
        colliding = np.asarray(colliding, dtype=bool)
        heightfields = np.flatnonzero(colliding & (shape_types == hfield))
        partners = np.flatnonzero(colliding & (shape_types != hfield) & (shape_types != mesh))
        world = np.full(len(shape_types), -1, np.int64) if shape_world is None else np.asarray(shape_world)
        pieces = [_cross(heightfields[world[heightfields] < 0], partners)]
        local = heightfields[world[heightfields] >= 0]
        pieces.append(_cross(local, partners[world[partners] < 0]))
        # Each world's heightfields with the partners of the same world.
        order = np.argsort(world[partners], kind="stable")
        sorted_partners, sorted_worlds = partners[order], world[partners][order]
        begin = np.searchsorted(sorted_worlds, world[local], side="left")
        lengths = np.searchsorted(sorted_worlds, world[local], side="right") - begin
        starts = np.repeat(begin - (np.cumsum(lengths) - lengths), lengths)
        positions = starts + np.arange(int(lengths.sum()))
        pieces.append(np.stack([np.repeat(local, lengths), sorted_partners[positions]], axis=1))
        table = np.concatenate(pieces)
        if shape_collision_group is not None:
            group = np.asarray(shape_collision_group, dtype=np.int64)
            table = table[_groups_interact(group[table[:, 0]], group[table[:, 1]])]
        if filter_mask is not None and len(table):
            table = table[~np.asarray(filter_mask(table), dtype=bool)]
    return np.unique(np.asarray(table, dtype=np.int64).reshape(-1, 2), axis=0).astype(np.int32)


def _groups_interact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether shapes of collision groups ``a`` and ``b`` collide, elementwise (as the broad phase decides).

    Group 0 collides with nothing, a positive group with itself and the negative groups, a negative group with
    every group but itself.
    """
    return (a != 0) & (b != 0) & (((a > 0) & ((a == b) | (b < 0))) | ((a < 0) & (a != b)))


def _cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Every pair of an element of ``first`` with an element of ``second``, as rows."""
    return np.stack([np.repeat(first, len(second)), np.tile(second, len(first))], axis=1).reshape(-1, 2)


def terrain_partner_classes(shape_type: np.ndarray, shape_scale: np.ndarray) -> np.ndarray:
    """Return the law's partner class of every shape (-1 when the law has none: meshes, infinite planes,
    heightfields).

    Cylinders, barrel cylinders (a nonzero barrel radius in ``scale[2]``) among them, take the cylinder class with
    cones; a finite plane (:func:`terrain_finite_planes`) takes the box class.

    Args:
        shape_type: Geometry type of every shape.
        shape_scale: Scale of every shape, ``(shape_count, 3)``.

    Returns:
        The partner class of every shape, int32.
    """
    kind = np.asarray(shape_type)
    result = np.full(len(kind), -1, np.int32)
    result[(kind == int(GeoType.SPHERE)) | (kind == int(GeoType.CAPSULE))] = TERRAIN_PARTNER_POINT
    result[(kind == int(GeoType.BOX)) | terrain_finite_planes(kind, shape_scale)] = TERRAIN_PARTNER_BOX
    result[(kind == int(GeoType.CYLINDER)) | (kind == int(GeoType.CONE))] = TERRAIN_PARTNER_CYLINDER
    result[(kind == int(GeoType.CONVEX_MESH)) | (kind == int(GeoType.ELLIPSOID))] = TERRAIN_PARTNER_HULL
    return result


def terrain_finite_planes(shape_type: np.ndarray, shape_scale: np.ndarray) -> np.ndarray:
    """Return which shapes are planes of finite width and length (``scale[0]`` and ``scale[1]``, both nonzero; a
    zero one is infinite, as :meth:`~newton.ModelBuilder.add_shape_plane` and the collision radius take it)."""
    scale = np.asarray(shape_scale, dtype=np.float64).reshape(-1, 3)
    plane = np.asarray(shape_type) == int(GeoType.PLANE)
    return plane & (scale[:, 0] != 0.0) & (scale[:, 1] != 0.0)


def terrain_window_cells(radius, reach, dx, dy) -> np.ndarray:
    """Return the cells of the largest window a partner can need: its bounding sphere plus reach, any alignment.

    Takes scalars or arrays of equal shape [m]; computes in float64.
    """
    extent = 2.0 * (np.asarray(radius, np.float64) + np.asarray(reach, np.float64))
    return (np.ceil(extent / dx) + 1.0) * (np.ceil(extent / dy) + 1.0)


def terrain_window_vertices(radius, reach, dx, dy, nrow, ncol) -> np.ndarray:
    """Return a bound on the vertices of a pair's window, any pose: its bounding sphere plus reach on the grid.

    A window spans the cells a box of extent ``e = 2 (radius + reach)`` touches, widened by
    :data:`~newton._src.geometry.terrain_law.TERRAIN_CELL_WIDEN` cells per side and its float32 cell coordinates
    rounded: at most ``floor(e / dx) + 3`` cells along x (``ceil(e / dx + 0.003) + 1``), so at most
    ``floor(e / dx) + 4`` vertex columns, and no more than the heightfield's. Takes scalars or arrays of equal shape
    [m]; computes in float64.
    """
    extent = 2.0 * (np.asarray(radius, np.float64) + np.asarray(reach, np.float64))
    columns = np.minimum(np.floor(extent / np.asarray(dx, np.float64)) + 4.0, np.asarray(ncol, np.float64))
    rows = np.minimum(np.floor(extent / np.asarray(dy, np.float64)) + 4.0, np.asarray(nrow, np.float64))
    return (columns * rows).astype(np.int64)


def terrain_window_sides(radius, reach, dx, dy, nrow, ncol) -> np.ndarray:
    """Return a bound on the vertex columns plus rows of a pair's window, any pose (the sides of
    :func:`terrain_window_vertices`'s window). Takes scalars or arrays of equal shape [m]; computes in float64."""
    extent = 2.0 * (np.asarray(radius, np.float64) + np.asarray(reach, np.float64))
    columns = np.minimum(np.floor(extent / np.asarray(dx, np.float64)) + 4.0, np.asarray(ncol, np.float64))
    rows = np.minimum(np.floor(extent / np.asarray(dy, np.float64)) + 4.0, np.asarray(nrow, np.float64))
    return (columns + rows).astype(np.int64)


def terrain_pair_candidates(shape_type, vertices, sides=None) -> np.ndarray:
    """Return the most distinct candidates a pair's walk can give its finish: every feature of each of its window's
    ``vertices`` (:func:`terrain_window_vertices`) giving its partner's most rows
    (:data:`TERRAIN_WALK_ROWS_PER_VERTEX`), the rows added after the walk (:data:`TERRAIN_WALK_ROWS_PER_PAIR`) and for
    a solid the rows its window can give at the heightfield's border (:data:`TERRAIN_BORDER_ROWS_PER_SIDE` per vertex
    along its ``sides``, :func:`terrain_window_sides`, and :data:`TERRAIN_BORDER_ROWS_PER_PAIR`).

    The walks visit every feature of a window once (a window vertex owns its cell's faces and diagonal, its x- and
    y-edges and itself; at the border the walls, the corner lines and the clipped faces of its border edges), and
    each feature gives a partner at most the rows counted there, so no window can give more: a scratch block that holds
    this many holds every candidate of the pair.

    Args:
        shape_type: Geometry type of every partner.
        vertices: The bound on every partner's window vertices.
        sides: The bound on every partner's window columns plus rows, or ``None`` for a window that reaches no border.
    """
    kind = np.asarray(shape_type)
    per_vertex = np.full(kind.shape, TERRAIN_WALK_ROWS_PER_VERTEX[2], np.int64)
    per_pair = np.full(kind.shape, TERRAIN_WALK_ROWS_PER_PAIR[2], np.int64)
    border = np.zeros(kind.shape, np.int64)
    if sides is not None:
        border = TERRAIN_BORDER_ROWS_PER_SIDE * np.asarray(sides, np.int64) + TERRAIN_BORDER_ROWS_PER_PAIR
    sphere = kind == int(GeoType.SPHERE)
    capsule = kind == int(GeoType.CAPSULE)
    per_vertex[sphere], per_pair[sphere] = TERRAIN_WALK_ROWS_PER_VERTEX[0], TERRAIN_WALK_ROWS_PER_PAIR[0]
    per_vertex[capsule], per_pair[capsule] = TERRAIN_WALK_ROWS_PER_VERTEX[1], TERRAIN_WALK_ROWS_PER_PAIR[1]
    border = np.broadcast_to(border, kind.shape).copy()
    border[sphere | capsule] = 0
    return per_vertex * np.asarray(vertices, np.int64) + per_pair + border


class TerrainPairTable:
    """The static (heightfield, partner) pairs of the terrain law, one contiguous segment per partner and lane class.

    Pairs are exactly those the model's collision filter admits, since the law's pairs bypass the broad phase:
    the model's pair list (:attr:`~newton.Model.shape_contact_pairs`; an empty list gives an empty table) or,
    when the model has none, the same filter applied here (worlds, collision groups, filter pairs). They are
    restricted to partners the law has a class for and sorted by (partner class, lane class, world, heightfield,
    partner). A pair's lane class and its bound on candidates (:func:`terrain_pair_candidates`) follow from its largest
    window over all orientations (bounding radius plus margins and gaps, on its heightfield's cells) at construction.

    Args:
        model: A finalized model.
        pairs: Shape pairs to take instead of the model's list (a collision pipeline's explicit pairs), as an
            array of ``(shape, shape)`` rows or a Warp array of ``vec2i``; the heightfield pairs among them are kept.
    """

    def __init__(self, model, *, pairs=None):
        self.device = model.device
        count = model.shape_count
        types = model.shape_type.numpy() if count else np.zeros(0, np.int32)
        worlds = model.shape_world.numpy() if count else np.zeros(0, np.int32)
        flags = model.shape_flags.numpy() if count else np.zeros(0, np.int32)
        from .flags import ShapeFlags  # noqa: PLC0415

        colliding = (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        listed = _shape_contact_pairs_host(model) if count else None
        explicit = pairs
        if not count:
            pairs = np.zeros((0, 2), np.int32)
        elif explicit is not None:
            explicit = explicit.numpy() if hasattr(explicit, "numpy") else np.asarray(explicit)
            pairs = heightfield_pair_table(types, worlds, colliding, explicit_pairs=explicit.reshape(-1, 2))
        elif listed is not None:
            pairs = heightfield_pair_table(types, worlds, colliding, explicit_pairs=listed)
        else:
            pairs = heightfield_pair_table(
                types,
                worlds,
                colliding,
                shape_collision_group=model.shape_collision_group.numpy(),
                filter_mask=model.shape_collision_filter_mask,
            )
        scale = model.shape_scale.numpy() if count else np.zeros((0, 3), np.float32)
        if count:
            # An infinite plane on a body meets the whole heightfield: no window holds its contacts.
            body = model.shape_body.numpy()[pairs[:, 1]]
            plane = types[pairs[:, 1]] == int(GeoType.PLANE)
            infinite = plane & ~terrain_finite_planes(types, scale)[pairs[:, 1]] & (body >= 0)
            if np.any(infinite):
                heightfield, partner = (int(s) for s in pairs[np.argmax(infinite)])
                raise ValueError(
                    f"Shape {partner}, a plane of infinite extent on body {int(body[np.argmax(infinite)])}, collides "
                    f"with heightfield shape {heightfield}: the terrain law has no contact for a plane that meets the "
                    "whole heightfield. Give the plane a finite width and length (ModelBuilder.add_shape_plane), "
                    "filter the pair, or set NEWTON_TERRAIN_LAW=none before constructing the collision pipeline to "
                    "keep the terrain route."
                )
        partner_class = terrain_partner_classes(types, scale)[pairs[:, 1]]
        pairs = pairs[partner_class >= 0]
        partner_class = partner_class[partner_class >= 0]
        lane = np.zeros(len(pairs), np.int32)
        world = np.zeros(len(pairs), np.int32)
        vertices = np.zeros(len(pairs), np.int64)
        sides = np.zeros(len(pairs), np.int64)
        candidates = np.zeros(len(pairs), np.int64)
        if len(pairs):
            first, second = pairs[:, 0], pairs[:, 1]
            margin = model.shape_margin.numpy()
            gap = model.shape_gap.numpy()
            layout = model.heightfield_data.numpy()[model.shape_heightfield_index.numpy()[first]]
            nrow, ncol = layout["nrow"].astype(np.int64), layout["ncol"].astype(np.int64)
            dx = 2.0 * layout["hx"].astype(np.float64) / (ncol - 1)
            dy = 2.0 * layout["hy"].astype(np.float64) / (nrow - 1)
            reach = margin[first] + margin[second] + gap[first] + gap[second]
            radius = model.shape_collision_radius.numpy()[second]
            cells = terrain_window_cells(radius, reach, dx, dy)
            lane = np.where(cells <= TERRAIN_LANE_CELLS, TERRAIN_LANE_THREAD, TERRAIN_LANE_TILE).astype(np.int32)
            world = np.where(worlds[second] >= 0, worlds[second], worlds[first]).astype(np.int32)
            vertices = terrain_window_vertices(radius, reach, dx, dy, nrow, ncol)
            sides = terrain_window_sides(radius, reach, dx, dy, nrow, ncol)
            candidates = terrain_pair_candidates(types[second], vertices, sides)
        order = np.lexsort((pairs[:, 1], pairs[:, 0], world, lane, partner_class))
        self.pairs_host = pairs[order].astype(np.int32).reshape(-1, 2)
        """``(heightfield, partner)`` shape pairs in table order, int32 ``(pair_count, 2)`` (host)."""
        self.partner_class = partner_class[order]
        """Partner class of every pair, int32 (host)."""
        self.lane_class = lane[order]
        """Lane class of every pair, int32 (host)."""
        self.world = world[order]
        """World of every pair (the partner's, or the heightfield's for a global partner), int32 (host)."""
        self.vertices = vertices[order]
        """Bound on every pair's window vertices, any pose (:func:`terrain_window_vertices`), int64 (host)."""
        self.sides = sides[order]
        """Bound on every pair's window columns plus rows, any pose (:func:`terrain_window_sides`), int64 (host)."""
        self.candidates = candidates[order]
        """Bound on every pair's distinct candidates, any pose (:func:`terrain_pair_candidates`), int64 (host)."""
        barrel = (types == int(GeoType.CYLINDER)) & (scale[:, 2] > 0.0)
        self.barrels = bool(np.any(barrel[self.pairs_host[:, 1]]))
        """Whether a pair's partner is a barrel cylinder (a cylinder whose scale's third entry, its side's radius, is
        not zero): the cylinder class then takes the kernels that hold the barrels' paths
        (:data:`~newton._src.geometry.terrain_shadow.TERRAIN_LAW_BARREL_KERNELS`)."""
        segment = self.partner_class * TERRAIN_LANE_CLASSES + self.lane_class
        segments = TERRAIN_PARTNER_CLASSES * TERRAIN_LANE_CLASSES
        sizes = np.bincount(segment, minlength=segments) if len(segment) else np.zeros(segments, np.int64)
        self.segment_offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        """First pair of every (partner class, lane class) segment, and the pair count last (host)."""
        self.pairs = wp.array(
            self.pairs_host if len(self.pairs_host) else np.zeros((1, 2), np.int32), dtype=wp.vec2i, device=self.device
        )
        """The pairs on the model's device (one unused pair when the table is empty)."""
        self.pair_count = len(self.pairs_host)
        """Pairs in the table."""

    def segment(self, partner_class: int, lane_class: int) -> tuple[int, int]:
        """Return the ``[begin, end)`` pair range of one partner class and lane class."""
        s = partner_class * TERRAIN_LANE_CLASSES + lane_class
        return int(self.segment_offsets[s]), int(self.segment_offsets[s + 1])

    def memory_bytes(self) -> dict[str, int]:
        """Return the device bytes the table holds."""
        return {"pairs": self.pairs.capacity}

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's kernels for point and segment cores: spheres and capsules.

A sphere is a point core (its centre) and a capsule a segment core (its axis), each with its radius as rounding.
Per pair the kernels emit one row per local minimum of the core-to-surface distance within the reach ``D`` (the
radius plus both margins and both gaps, the contact writer's admission limit), owned by one surface feature
(rule P of the law):

* face: the core point's projection onto the triangle's cluster plane lies in the triangle (front-facing);
* crease edge (convex, or concave for a core point inside the solid): the core point lies beyond both faces at
  the edge and inside the edge's slab;
* vertex: the core point lies in the vertex's Voronoi region;
* capsule interior: the axis point closest to a crease edge (both segment parameters inside), and the axis point
  over a vertex in its Voronoi region. An axis point inside the solid below a ridge exits over it: within both
  faces the normal from it to the crease lies in the ridge's arc, and lifting the axis along it by the distance is
  the segment's minimum translation out of the ridge (a point exit through either face would push it sideways).

Every boundary is decided by one canonical value that both sides read, closed by the window's band
(:func:`~newton._src.geometry.terrain_law.terrain_eps`); duplicates of one constraint merge in the finish. A core
point inside the solid keeps one exit, the one with the smallest translation (G2); with no exit in its window it
gets the face vertically above it, with that plane's normal and depth (G3). The axis interior gets G3 too: with
both ends outside the solid and no axis exit, the deepest crossing of the axis's footprint under a crease gets the
plane of the crease's first triangle (:func:`_axis_witness`). Concave anticipation (rule C) is off.

Work per pair, cheapest rejects first: the tile-max exit (the core's lowest point lies above every 8 x 8-cell mip
block under its window by more than ``D``); the vertex drop (:func:`_vertex_dropped`: the core lies farther than
``D`` from the box its vertex's features lie in, a sphere centre in 3-D, a capsule by height; off while a core point
lies inside the solid); then per vertex its faces by projection, its crease edges in closed form (flat and border
edges dropped on cluster ids) and the vertex; then
:func:`~newton._src.geometry.terrain_law.terrain_finish` with at most :data:`TERRAIN_POINT_ROWS` rows. Each feature
is rejected on geometry before its canonical values are read: a face whose projection lies off its cell
(:func:`_off_cell`, conservative), an edge whose slab misses the core point or whose closest point is out of reach
(exact) before its class and planes. On 4,096 recorded quadruped foot spheres the drop keeps 4.0 of a window's
11.7 vertices; per pair, 2.3 of the 7.7 front-facing faces within reach get to their canonical values, and of 12.1
edges 3.4 are creases and 1.0 gets to its planes. The rows are bit for bit those of the walk without the drop, the
vertex mask and the off-cell reject (the point-kernel tests check it). A window's cluster planes are evaluated once
(:class:`~newton._src.geometry.terrain_law.TerrainPlanes`).

The window is walked vertex by vertex: vertex ``(r, c)`` owns the features :func:`_point_item` lists (its cell's
faces and diagonal, its x- and y-edges, itself), each evaluated for the sphere centre or for both capsule ends and
the axis. An item yields at most one row (:class:`_Claim`).

Thread mapping (the pair table's lane classes): :func:`terrain_point_rows` runs one thread per pair of the class, in
blocks of :data:`~newton._src.geometry.terrain_shadow.TERRAIN_LAW_BLOCK` (so that its warps spread evenly over the SMs).
For a pair whose window holds at most 16 cells it marks a word of window vertices the drop keeps in a bit mask and then
walks the marked ones in window order, so that a warp's lanes evaluate their k-th kept vertices together instead of one
window position that most lanes drop (a model of the warps' issued instructions on recorded quadruped foot spheres,
per-stage SASS counts weighted by what any lane of a warp runs: -24% and -27% on two suite scenes against walking every
window position). A sphere walks its centre's six items in code of its own (:func:`_walk_sphere`; a kernel of it alone
is 12,300 SASS instructions on sm_120, of the capsule's walk (:func:`_walk_segment`, every item) alone 27,700), so that
a warp of spheres never runs the capsule's. A pair with a larger window the kernel only takes through the whole-pair
rejects, the tile-max exit above all (97% of the recorded suite's such pairs), and queues the rest
(:class:`~newton._src.geometry.terrain_shadow.TerrainPairWork`): :func:`terrain_point_rows_tiled`, a fixed grid of
:data:`~newton._src.geometry.terrain_shadow.TERRAIN_LAW_BLOCK` lanes per pair (``wp.launch_tiled``), takes the queued
pairs in turn, so that a pair that leaves at the tile max never occupies a tile. The lanes split the window's heights
and vertices and mark, per vertex, the items that yield anything; the first lane evaluates those items in window order,
so a pair gets the rows the one-thread kernel gives it. Both hold a pair's distinct rows in its own slots (up to six,
merged as the finish merges) and its ends' exits, and write them when the finish keeps them as they are: no plane group
holds three of them (the finish's implied-row and per-group steps act on larger groups), they fit the eight slots and
the candidates fit the finish's list. That is every recorded pair of the suite's rough scenes; any other pair is
appended to a deferred list, and :func:`terrain_point_rows_full`, launched after them with a fixed grid of lane blocks
that strides over the list, marks each pair's window as the tiled kernel does, walks the marked items again into the
full candidate list and runs the finish.

Inlined into the walks, the finish took every thread to 255 registers and a 10 KB stack on sm_120. Apart, the one-thread
kernel takes 128 registers (its launch bounds: 16 blocks of 32 threads per SM) and a 176-byte stack (a sphere's walk
alone would need 140 registers), the tiled kernel 226 registers and a 32-byte stack, and only deferred pairs pay for the
finish. The full-finish kernel keeps the candidate list and the finish's input in global scratch
(:class:`~newton._src.geometry.terrain_shadow.TerrainFinishScratch`), so that its stack, which sets the driver's
local-memory reservation for every thread the device holds, is the finish's own frame: 6.0 KB (10.1 KB with the list on
the stack). The kernels take the per-type kernel arguments ``(begin, slot_begin, inputs, data, out)`` of
:class:`~newton._src.geometry.terrain_law.TerrainPairInputs` and a launch argument: the walks the class's work list, the
full-finish kernel its scratch.

The module is compiled without floating-point contraction, as the cluster build is: every float32 operation
rounds as on the CPU, so CPU tests decide every boundary, and a CPU census counts every row, as the GPU does.
"""

from __future__ import annotations

import warp as wp

from .broad_phase_common import is_shape_pair_immovable_filtered
from .terrain_contacts import TERRAIN_PARTNER_POINT, TERRAIN_PARTNER_ROWS
from .terrain_law import (
    TERRAIN_EDGE_CONCAVE,
    TERRAIN_EPS32,
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_FEATURE_VERTEX,
    TERRAIN_FLAT_EPS,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_WIDE,
    TERRAIN_ROW_CANDIDATES,
    TERRAIN_WINDOW_WIDE,
    TerrainFeatureData,
    TerrainLawRows,
    TerrainPairInputs,
    TerrainPlanes,
    TerrainWindow,
    terrain_beyond,
    terrain_box_cells,
    terrain_claims,
    terrain_crease_class,
    terrain_defer_pair,
    terrain_edge_kind,
    terrain_edge_sides,
    terrain_eps,
    terrain_face_edge_value,
    terrain_face_values,
    terrain_feature_id,
    terrain_features_current,
    terrain_finish,
    terrain_finish_state,
    terrain_height,
    terrain_hold_row,
    terrain_interior_vertex,
    terrain_locate,
    terrain_pair_exit,
    terrain_plane,
    terrain_plane_group_of_three,
    terrain_planes,
    terrain_point_below_cached,
    terrain_report,
    terrain_store_row,
    terrain_tile_exit,
    terrain_vertex_edge_value,
    terrain_vertex_point,
    terrain_window,
    terrain_window_extent,
    terrain_window_point,
    terrain_window_vertex,
)
from .terrain_shadow import TERRAIN_LAW_BLOCK, TerrainFinishScratch, TerrainPairWork
from .types import GeoType

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

TERRAIN_POINT_ROWS = TERRAIN_PARTNER_ROWS[TERRAIN_PARTNER_POINT]
"""Rows a sphere or capsule pair keeps (the point class's row limit, the slots per pair): 8."""
TERRAIN_POINT_MASK_WORDS = 8
"""Words of the tiled kernels' productive-vertex mask: windows of up to 256 vertices; larger windows are walked by
the first lane alone."""
TERRAIN_POINT_MASK_VERTICES = 32 * TERRAIN_POINT_MASK_WORDS
"""Window vertices the tiled kernels mark (each vertex's productive items, in shared memory)."""
TERRAIN_POINT_ITEMS = 16
"""Items a window vertex owns for a capsule (:func:`_point_item`); a sphere's are the first six."""

_ROWS = wp.constant(TERRAIN_POINT_ROWS)
_CLASS = wp.constant(TERRAIN_PARTNER_POINT)
_CANDIDATES = wp.constant(TERRAIN_ROW_CANDIDATES)
_MASK_VERTICES = wp.constant(TERRAIN_POINT_MASK_VERTICES)
_ITEMS = wp.constant(TERRAIN_POINT_ITEMS)
_SPHERE_ITEMS = wp.constant(6)
_END_A_ITEMS = wp.constant(0x003F)
"""Items 0-5 (:func:`_point_item`): the sphere centre's or capsule end ``a``'s."""
_END_B_ITEMS = wp.constant(0x0FC0)
"""Items 6-11: capsule end ``b``'s."""
_AXIS_ITEMS = wp.constant(0xF000)
"""Items 12-15: the capsule axis's."""
_CAPSULE = wp.constant(int(GeoType.CAPSULE))
_WIDE = wp.constant(TERRAIN_WINDOW_WIDE)
_EPS32 = wp.constant(TERRAIN_EPS32)
_ON_FEATURE = wp.constant(1.0e-9)
"""Core distance [m] below which a core point lies on an edge (bisector normal) or vertex (no normal: no row)."""
_FOOTPRINT_BANDS = wp.constant(16.0)
"""Bands a face's canonical values may exceed their band by through rounding, generously (see :func:`_off_cell`)."""
_FOOTPRINT_FLAT = wp.constant(4.0 * TERRAIN_FLAT_EPS)
_FLAT = wp.constant(TERRAIN_FLAT_EPS)
"""How far a cluster plane may move a triangle edge's xy footprint [m], generously (see :func:`_off_cell`)."""
_PARALLEL = wp.constant(1.0e-8)
"""Relative squared sine below which a capsule axis is parallel to an edge (its ends and vertices own the minima)."""
_SLOT_AXIS = wp.constant(2)
_ROW = wp.constant(1)
_EXIT_A = wp.constant(2)
_EXIT_B = wp.constant(3)
_WITNESS = _EXIT_B
"""An axis crossing under a crease shares end ``b``'s exit kind, slot and keep-the-largest rule: it arises only with both
ends outside the solid, when end ``b`` has no exit (one walk branch less, which cost the tiled kernel 75 registers)."""

_Mask = wp.types.vector(length=TERRAIN_POINT_MASK_WORDS, dtype=wp.int32)


@wp.struct
class _PointCore:
    """A pair's core in the window frame."""

    a: wp.vec3
    """Sphere centre, or the capsule's axis end on its local +z side."""
    b: wp.vec3
    """The capsule's other axis end (the centre again for a sphere)."""
    segment: int
    """1 for a capsule."""
    reach: float
    """Reach ``D`` on the core distance [m]."""
    below_a: int
    """1 when end ``a`` lies inside the solid."""
    below_b: int
    """1 when end ``b`` lies inside the solid."""
    low: float
    """The core's lowest point minus the reach [m]; the vertex drop compares vertex heights with it."""
    skip: int
    """1 when the vertex drop applies (no end inside the solid, whose exits the reach does not bound)."""
    eps: float
    """The window's closed-predicate band [m]."""


@wp.struct
class _Claim:
    """What one item yields: nothing, a row, an exit candidate of an end inside the solid, or an axis crossing."""

    kind: int
    """0 nothing, 1 a row, 2 or 3 an exit candidate of end ``a`` or ``b``; with both ends outside the solid, 3 is a
    crossing of the axis under a crease instead (:func:`_axis_witness`: ``q`` the axis point, ``d`` its depth,
    ``feature`` the crease)."""
    q: wp.vec3
    k: wp.vec3
    n: wp.vec3
    d: float
    feature: int


@wp.func
def _offer(end: int, below: int, q: wp.vec3, k: wp.vec3, n: wp.vec3, d: float, reach: float, feature: int) -> _Claim:
    """A claim of a core end: an exit candidate when the end lies inside the solid, else a row within reach."""
    claim = _Claim()
    if below != 0:
        claim.kind = _EXIT_A + end
    elif d <= reach:
        claim.kind = _ROW
    claim.q = q
    claim.k = k
    claim.n = n
    claim.d = d
    claim.feature = feature
    return claim


# ----------------------------------------------------------------------------------------------------------------
# features of one end (sphere centre or capsule end)
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _off_cell(window: TerrainWindow, core: _PointCore, r: int, c: int, q: wp.vec3) -> bool:
    """Whether plane point ``q`` lies off cell ``(r, c)`` by more than its triangles' canonical values can claim.

    A face value is the in-plane distance to an edge's line, no smaller than the xy distance to the line's footprint
    (projecting the plane onto xy shortens distances and keeps sides), and a cluster plane moves an edge's footprint
    by at most :data:`TERRAIN_FLAT_EPS`. A projection the face claims thus lies within ``e`` (the band, rounding
    and the cluster tolerance, :data:`_FOOTPRINT_BANDS` and :data:`_FOOTPRINT_FLAT`) of each edge of the half
    cell in xy, which the half cell's corners stretch to ``e (2 dx + dy) / dy`` beyond the cell along x and
    ``e (2 dy + dx) / dx`` along y. A conservative test: it refuses no projection the values claim.
    """
    surface = window.surface
    dx = surface.dx
    dy = surface.dy
    e = _FOOTPRINT_BANDS * core.eps + _FOOTPRINT_FLAT
    ex = e * (2.0 * dx + dy)
    ey = e * (2.0 * dy + dx)
    x = q[0] - float(c - window.c0) * dx
    y = q[1] - float(r - window.r0) * dy
    return -x * dy > ex or (x - dx) * dy > ex or -y * dx > ey or (y - dy) * dx > ey


@wp.func
def _end_face(
    data: TerrainFeatureData,
    window: TerrainWindow,
    core: _PointCore,
    end: int,
    k: wp.vec3,
    below: int,
    r: int,
    c: int,
    sub: int,
    n: wp.vec3,
    a: wp.vec3,
) -> _Claim:
    """Rule P, face: the end's projection onto the cluster plane lies in the triangle, front-facing.

    A projection off the triangle's cell is refused before the canonical values are evaluated (:func:`_off_cell`).
    """
    d = wp.dot(n, k - a)
    if d > core.reach:
        return _Claim()
    # Front-facing: an end outside the solid above the plane, an end inside below it (closed by the band).
    if below != 0:
        if d > core.eps:
            return _Claim()
    elif d < -core.eps:
        return _Claim()
    q = k - d * n
    if _off_cell(window, core, r, c, q):
        return _Claim()
    s = terrain_face_values(data, window, r, c, sub, n, k)
    if not (terrain_claims(s[0], core.eps) and terrain_claims(s[1], core.eps) and terrain_claims(s[2], core.eps)):
        return _Claim()
    return _offer(end, below, q, k, n, d, core.reach, terrain_feature_id(r, c, TERRAIN_FEATURE_FACE, sub, end))


@wp.func
def _end_edge(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    end: int,
    k: wp.vec3,
    below: int,
    r: int,
    c: int,
    which: int,
    v0: wp.vec3,
    v1: wp.vec3,
) -> tuple[TerrainPlanes, _Claim]:
    """Rule P, crease edge: the end lies beyond both faces at the edge and inside its slab; q is its closest point.

    For a crease (:func:`_point_item` drops flat and border edges); the slab and the reach are tested before the
    edge's class and planes are read.
    """
    u = v1 - v0
    length = wp.length(u)
    eps = core.eps
    w0 = terrain_vertex_edge_value(k, v0, v1)
    if not (terrain_beyond(w0, eps * length) and terrain_beyond(terrain_vertex_edge_value(k, v1, v0), eps * length)):
        return planes, _Claim()
    q = v0 + wp.clamp(w0 / wp.dot(u, u), 0.0, 1.0) * u
    w = k - q
    dist = wp.length(w)
    if below == 0 and dist > wp.max(core.reach, _ON_FEATURE):
        # Out of reach: exactly the row test of _offer (d = dist above _ON_FEATURE).
        return planes, _Claim()
    _v0, _v1, ta, tb, wa, wb = terrain_edge_sides(r, c, which)
    planes, na, pa = terrain_plane(data, window, planes, ta[0], ta[1], ta[2])
    if below == 0 and terrain_crease_class(data, window, r, c, which, na, pa) == TERRAIN_EDGE_CONCAVE:
        # A valley's minima outside the solid are its faces' (two half-spaces); only an inside end exits there.
        return planes, _Claim()
    planes, nb, _pb = terrain_plane(data, window, planes, tb[0], tb[1], tb[2])
    sa = terrain_face_edge_value(k, v0, v1, na, terrain_window_vertex(data, window, wa[0], wa[1]))
    sb = terrain_face_edge_value(k, v0, v1, nb, terrain_window_vertex(data, window, wb[0], wb[1]))
    if not (terrain_beyond(sa, eps) and terrain_beyond(sb, eps)):
        return planes, _Claim()
    n = wp.normalize(na + nb)
    d = float(0.0)
    if dist > _ON_FEATURE:
        n = w / dist
        d = dist
        if below != 0:
            n = -n
            d = -dist
    feature = terrain_feature_id(r, c, TERRAIN_FEATURE_EDGE, which, end)
    return planes, _offer(end, below, q, k, n, d, core.reach, feature)


@wp.func
def _end_vertex(
    data: TerrainFeatureData, window: TerrainWindow, core: _PointCore, end: int, k: wp.vec3, below: int, r: int, c: int
) -> _Claim:
    """Rule P, vertex: the end lies in the vertex's Voronoi region; the row's normal points from the vertex to it."""
    if not terrain_interior_vertex(window.surface, r, c):
        return _Claim()
    v = terrain_window_vertex(data, window, r, c)
    w = k - v
    dist = wp.length(w)
    if dist <= _ON_FEATURE or (below == 0 and dist > core.reach):
        return _Claim()
    if not terrain_vertex_point(data, window, r, c, k):
        return _Claim()
    n = w / dist
    d = dist
    if below != 0:
        n = -n
        d = -dist
    return _offer(end, below, v, k, n, d, core.reach, terrain_feature_id(r, c, TERRAIN_FEATURE_VERTEX, 0, end))


# ----------------------------------------------------------------------------------------------------------------
# the capsule's axis interior
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _axis_crease(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    r: int,
    c: int,
    which: int,
    v0: wp.vec3,
    v1: wp.vec3,
) -> tuple[TerrainPlanes, _Claim]:
    """The axis point closest to a crease edge, both segment parameters inside, with the normal from the crease toward
    it (outside the solid) or from it to the crease (inside) in the crease's normal cone.

    Outside the solid a ridge claims it beyond both faces at the edge. Inside it a valley claims it beyond both faces
    (the edge is its closest surface point), and a ridge within both faces: the axis passes below the ridge, and the
    exit lifts it over the crease by the distance of the two lines, the segment's minimum translation out of the
    ridge. For a crease (:func:`_point_item` drops flat and border edges).
    """
    d2 = v1 - v0
    d1 = core.b - core.a
    offset = core.a - v0
    aa = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, offset)
    cc = wp.dot(d1, offset)
    bb = wp.dot(d1, d2)
    denom = aa * e - bb * bb
    if denom <= _PARALLEL * aa * e:
        return planes, _Claim()
    s = (bb * f - cc * e) / denom
    t = (bb * s + f) / e
    if s < 0.0 or s > 1.0 or t < 0.0 or t > 1.0:
        return planes, _Claim()
    p = core.a + s * d1
    q = v0 + t * d2
    w = p - q
    dist = wp.length(w)
    if dist <= _ON_FEATURE:
        return planes, _Claim()
    planes, below = terrain_point_below_cached(data, window, planes, p)
    if not below and dist > core.reach:
        return planes, _Claim()
    _v0, _v1, ta, tb, wa, wb = terrain_edge_sides(r, c, which)
    planes, na, pa = terrain_plane(data, window, planes, ta[0], ta[1], ta[2])
    concave = terrain_crease_class(data, window, r, c, which, na, pa) == TERRAIN_EDGE_CONCAVE
    if not below and concave:
        return planes, _Claim()
    planes, nb, _pb = terrain_plane(data, window, planes, tb[0], tb[1], tb[2])
    sa = terrain_face_edge_value(p, v0, v1, na, terrain_window_vertex(data, window, wa[0], wa[1]))
    sb = terrain_face_edge_value(p, v0, v1, nb, terrain_window_vertex(data, window, wb[0], wb[1]))
    if below and not concave:
        # Below a ridge: the normal from the axis point to the crease lies in the ridge's arc (within both faces).
        if not (terrain_claims(sa, core.eps) and terrain_claims(sb, core.eps)):
            return planes, _Claim()
    elif not (terrain_beyond(sa, core.eps) and terrain_beyond(sb, core.eps)):
        return planes, _Claim()
    n = w / dist
    d = dist
    if below:
        n = -n
        d = -dist
    # Every axis row is a row, never an end's exit candidate (G2 holds for the ends).
    feature = terrain_feature_id(r, c, TERRAIN_FEATURE_EDGE, which, _SLOT_AXIS)
    return planes, _offer(0, 0, q, p, n, d, core.reach, feature)


@wp.func
def _axis_witness(
    window: TerrainWindow, core: _PointCore, r: int, c: int, which: int, v0: wp.vec3, v1: wp.vec3
) -> _Claim:
    """Where the axis's footprint crosses under crease ``which`` of vertex ``(r, c)``: the axis point there and its
    depth below the crease (the edge's height there minus the axis's), a candidate for the axis interior's G3.

    The axis's vertical depth is piecewise linear along it, with its interior maxima where the footprint crosses a
    ridge, so the deepest such crossing is the axis interior's deepest point.
    """
    claim = _Claim()
    a = core.a
    d1 = core.b - a
    d2 = v1 - v0
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if den == 0.0:
        return claim
    rx = v0[0] - a[0]
    ry = v0[1] - a[1]
    s = (rx * d2[1] - ry * d2[0]) / den
    t = (rx * d1[1] - ry * d1[0]) / den
    if s < 0.0 or s > 1.0 or t < 0.0 or t > 1.0:
        return claim
    depth = (v0[2] + t * d2[2]) - (a[2] + s * d1[2])
    if depth > 0.0:
        claim.kind = _WITNESS
        claim.q = a + s * d1
        claim.d = depth
        claim.feature = (r * window.surface.ncol + c) * 3 + which
    return claim


@wp.func
def _axis_edge(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    r: int,
    c: int,
    which: int,
    v0: wp.vec3,
    v1: wp.vec3,
) -> tuple[TerrainPlanes, _Claim]:
    """The axis's claim at a crease (:func:`_axis_crease`); without one, and with both ends outside the solid, the
    axis's crossing under the crease (:func:`_axis_witness`)."""
    planes, claim = _axis_crease(data, window, planes, core, r, c, which, v0, v1)
    if claim.kind == 0 and core.below_a == 0 and core.below_b == 0:
        claim = _axis_witness(window, core, r, c, which, v0, v1)
    return planes, claim


@wp.func
def _axis_recovery(
    data: TerrainFeatureData, window: TerrainWindow, planes: TerrainPlanes, p: wp.vec3, crease: int
) -> tuple[TerrainPlanes, _Claim]:
    """G3 for the axis interior: the plane of the first triangle of ``crease`` (:func:`_axis_witness`'s id) under
    axis point ``p``, its normal and depth."""
    which = crease % 3
    vertex = crease // 3
    r = vertex // window.surface.ncol
    c = vertex - r * window.surface.ncol
    _v0, _v1, ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    planes, n, a = terrain_plane(data, window, planes, ta[0], ta[1], ta[2])
    d = wp.dot(n, p - a)
    claim = _Claim()
    if d < 0.0:
        claim.kind = _ROW
        claim.q = p - d * n
        claim.k = p
        claim.n = n
        claim.d = d
        claim.feature = terrain_feature_id(ta[0], ta[1], TERRAIN_FEATURE_RECOVERY, ta[2], _SLOT_AXIS)
    return planes, claim


@wp.func
def _axis_vertex(
    data: TerrainFeatureData, window: TerrainWindow, planes: TerrainPlanes, core: _PointCore, r: int, c: int
) -> tuple[TerrainPlanes, _Claim]:
    """The axis point over a vertex (parameter inside the segment) in the vertex's Voronoi region."""
    if not terrain_interior_vertex(window.surface, r, c):
        return planes, _Claim()
    v = terrain_window_vertex(data, window, r, c)
    d1 = core.b - core.a
    s = wp.dot(v - core.a, d1) / wp.dot(d1, d1)
    if s < 0.0 or s > 1.0:
        return planes, _Claim()
    p = core.a + s * d1
    w = p - v
    dist = wp.length(w)
    if dist <= _ON_FEATURE:
        return planes, _Claim()
    planes, below = terrain_point_below_cached(data, window, planes, p)
    if not below and dist > core.reach:
        return planes, _Claim()
    if not terrain_vertex_point(data, window, r, c, p):
        return planes, _Claim()
    n = w / dist
    d = dist
    if below:
        n = -n
        d = -dist
    return planes, _offer(0, 0, v, p, n, d, core.reach, terrain_feature_id(r, c, TERRAIN_FEATURE_VERTEX, 0, _SLOT_AXIS))


# ----------------------------------------------------------------------------------------------------------------
# the items a window vertex owns
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _end_item(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    end: int,
    k: wp.vec3,
    below: int,
    r: int,
    c: int,
    r1: int,
    c1: int,
    feature: int,
) -> tuple[TerrainPlanes, _Claim]:
    """Evaluate feature ``feature`` of window vertex ``(r, c)`` for core end ``end`` at ``k`` (``below``: inside the
    solid): the face of sub 0 or 1, the x-, y- or diagonal edge, or the vertex (0-5; see :func:`_point_item`)."""
    inside = r <= r1 and c <= c1
    if feature < 2:
        if not inside:
            return planes, _Claim()
        planes, n, a = terrain_plane(data, window, planes, r, c, feature)
        return planes, _end_face(data, window, core, end, k, below, r, c, feature, n, a)
    if feature == 5:
        return planes, _end_vertex(data, window, core, end, k, below, r, c)
    which = feature - 2
    if (which == 0 and c > c1) or (which == 1 and r > r1) or (which == 2 and not inside):
        return planes, _Claim()
    if terrain_edge_kind(data, window.surface, r, c, which) >= 0:
        # Flat inside one cluster, or on the heightfield's border: no edge minimum (decided from cluster ids).
        return planes, _Claim()
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    return _end_edge(data, window, planes, core, end, k, below, r, c, which, v0, v1)


@wp.func
def _axis_item(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    r: int,
    c: int,
    r1: int,
    c1: int,
    feature: int,
) -> tuple[TerrainPlanes, _Claim]:
    """Evaluate feature ``feature`` of window vertex ``(r, c)`` for the capsule's axis: the x-, y- or diagonal edge or
    the vertex (2-5, numbered as :func:`_end_item` numbers them)."""
    if feature == 5:
        return _axis_vertex(data, window, planes, core, r, c)
    which = feature - 2
    if (which == 0 and c > c1) or (which == 1 and r > r1) or (which == 2 and not (r <= r1 and c <= c1)):
        return planes, _Claim()
    if terrain_edge_kind(data, window.surface, r, c, which) >= 0:
        return planes, _Claim()
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    return _axis_edge(data, window, planes, core, r, c, which, v0, v1)


@wp.func
def _point_item(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    r: int,
    c: int,
    r1: int,
    c1: int,
    item: int,
) -> tuple[TerrainPlanes, _Claim]:
    """Evaluate one item of window vertex ``(r, c)`` in a window of cells ``[r0, r1] x [c0, c1]``.

    Vertex ``(r, c)`` owns its cell's two faces and diagonal (inside the window), its x-edge (``c <= c1``), its
    y-edge (``r <= r1``) and itself, so that every feature of the window is visited once. Items 0-5 are the face of
    sub 0, sub 1, the x-edge, the y-edge, the diagonal and the vertex for the sphere centre or capsule end ``a``
    (:func:`_end_item`), items 6-11 the same for end ``b``, items 12-15 the edges and the vertex for the capsule's
    axis (:func:`_axis_item`).
    """
    if item < 12:
        end = item // 6
        k = core.a
        below = core.below_a
        if end == 1:
            k = core.b
            below = core.below_b
        return _end_item(data, window, planes, core, end, k, below, r, c, r1, c1, item - 6 * end)
    return _axis_item(data, window, planes, core, r, c, r1, c1, item - 10)


@wp.func
def _vertex_items(core: _PointCore) -> int:
    if core.segment != 0:
        return _ITEMS
    return _SPHERE_ITEMS


@wp.func
def _cell_margins(window: TerrainWindow, core: _PointCore) -> wp.vec3:
    """How far beyond its cell, along x and y, a face row's terrain point may lie (:func:`_off_cell`'s bounds), and
    ``1 / dx + 1 / dy`` (for the slope bound of :func:`_vertex_dropped`)."""
    surface = window.surface
    e = _FOOTPRINT_BANDS * core.eps + _FOOTPRINT_FLAT
    return wp.vec3(
        e * (2.0 * surface.dx + surface.dy) / surface.dy,
        e * (2.0 * surface.dy + surface.dx) / surface.dx,
        1.0 / surface.dx + 1.0 / surface.dy,
    )


@wp.func
def _vertex_box(
    data: TerrainFeatureData, window: TerrainWindow, core: _PointCore, margins: wp.vec3, r: int, c: int
) -> wp.vec3:
    """The heights of the box vertex ``(r, c)``'s rows lie in (see :func:`_vertex_dropped`): its cell's lowest and
    highest vertex (those in the grid), window frame, and how far a row's terrain point may leave them, ``zm``."""
    surface = window.surface
    zlo = float(1.0e30)
    zhi = float(-1.0e30)
    for dr in range(2):
        for dc in range(2):
            rr = r + dr
            cc = c + dc
            if rr < surface.nrow and cc < surface.ncol:
                z = terrain_height(data, surface, rr, cc) - window.z_ref
                zlo = wp.min(zlo, z)
                zhi = wp.max(zhi, z)
    slope = (zhi - zlo + 2.0 * _FLAT) * margins[2] + 1.0
    zm = 4.0 * (_FLAT * slope + (margins[0] + margins[1]) * slope) + _FOOTPRINT_BANDS * core.eps
    return wp.vec3(zlo, zhi, zm)


@wp.func
def _far_from_box(
    window: TerrainWindow, core: _PointCore, margins: wp.vec3, r: int, c: int, box: wp.vec3, k: wp.vec3
) -> bool:
    """Whether core point ``k`` lies farther than the reach from vertex ``(r, c)``'s box (:func:`_vertex_box`), widened
    in xy by ``margins``."""
    surface = window.surface
    x = k[0] - float(c - window.c0) * surface.dx
    y = k[1] - float(r - window.r0) * surface.dy
    ox = wp.max(wp.max(-margins[0] - x, x - surface.dx - margins[0]), 0.0)
    oy = wp.max(wp.max(-margins[1] - y, y - surface.dy - margins[1]), 0.0)
    oz = wp.max(wp.max(box[0] - box[2] - k[2], k[2] - box[1] - box[2]), 0.0)
    return ox * ox + oy * oy + oz * oz > core.reach * core.reach


@wp.func
def _vertex_dropped(
    data: TerrainFeatureData, window: TerrainWindow, core: _PointCore, margins: wp.vec3, r: int, c: int
) -> bool:
    """Whether no feature vertex ``(r, c)`` owns can yield a row, so that the walk passes the vertex by.

    The features lie in the vertex's cell (its faces and diagonal, its x- and y-edges, itself), and a row's terrain
    point in the box of the cell's vertices (those in the grid), widened in xy by ``margins`` (a face's point may lie
    off its triangle within the band, :func:`_cell_margins`) and in z by how far a cluster plane may leave the
    vertices there (the cluster tolerance, steepened by the cell's slope, and the plane over the xy widening), all
    four times over plus the band (:func:`_vertex_box`). A row of a core point outside the solid has its core point
    within the reach of its terrain point, so a core farther than the reach from the box yields none: a sphere centre
    by its 3-D distance, a capsule by height alone (every vertex below the core's lowest point by more than the
    reach; :func:`_kept_items` drops a far end's items too). Neither applies while a core point lies inside the solid
    (its exits are not bounded by the reach).
    """
    if core.skip == 0:
        return False
    box = _vertex_box(data, window, core, margins, r, c)
    if core.segment != 0:
        return box[1] + box[2] < core.low
    return _far_from_box(window, core, margins, r, c, box, core.a)


@wp.func
def _kept_items(
    data: TerrainFeatureData, window: TerrainWindow, core: _PointCore, margins: wp.vec3, r: int, c: int
) -> int:
    """The items of window vertex ``(r, c)`` that can yield anything, a bit per item (:func:`_point_item`'s numbers).

    None of a vertex the drop passes by (:func:`_vertex_dropped`). Of a capsule's kept vertex, with both ends outside
    the solid, its axis's items and the items of each end within the reach of the vertex's box: an end's rows, like a
    sphere centre's, have their terrain points within the reach of the end (an axis point's exit, below a ridge, is
    not bounded by the reach, so the axis keeps its items).
    """
    if core.skip == 0:
        return (1 << _vertex_items(core)) - 1
    box = _vertex_box(data, window, core, margins, r, c)
    if core.segment == 0:
        if _far_from_box(window, core, margins, r, c, box, core.a):
            return 0
        return _END_A_ITEMS
    if box[1] + box[2] < core.low:
        return 0
    m = _AXIS_ITEMS
    if not _far_from_box(window, core, margins, r, c, box, core.a):
        m = m | _END_A_ITEMS
    if not _far_from_box(window, core, margins, r, c, box, core.b):
        m = m | _END_B_ITEMS
    return m


_MASK_BITS = wp.constant(31)
"""Window vertices per word of the one-thread kernel's vertex mask (positive int32 words)."""


@wp.func
def _lowest_bit(m: int) -> int:
    """The index of the lowest set bit of a positive mask."""
    x = m & (-m)
    v = int(0)
    if (x & 0x0000FFFF) == 0:
        v += 16
    if (x & 0x00FF00FF) == 0:
        v += 8
    if (x & 0x0F0F0F0F) == 0:
        v += 4
    if (x & 0x33333333) == 0:
        v += 2
    if (x & 0x55555555) == 0:
        v += 1
    return v


# ----------------------------------------------------------------------------------------------------------------
# per pair
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _pair_setup(inputs: TerrainPairInputs, data: TerrainFeatureData, pair: int):
    """Read a pair and decide its early exits.

    Returns ``(state, frame, heightfield, cells, a, b, segment, reach)``: ``state`` nonzero when the pair has no
    row (stale features, the immovable-pair policy, a window off the heightfield, the tile-max exit), the
    heightfield's world transform, its index, the window's cells ``(r0, r1, c0, c1)``, the core's ends in the
    heightfield frame, 1 for a capsule, and the reach.
    """
    shapes = inputs.pairs[pair]
    hfield = shapes[0]
    partner = shapes[1]
    heightfield = int(inputs.shape_heightfield_index[hfield])
    frame = inputs.shape_transform[hfield]
    cells = wp.vec4i(0, -1, 0, -1)
    a = wp.vec3(0.0)
    b = wp.vec3(0.0)
    segment = int(0)
    reach = float(0.0)
    if not terrain_features_current(data, heightfield):
        return TERRAIN_PAIR_STALE, frame, heightfield, cells, a, b, segment, reach
    if is_shape_pair_immovable_filtered(
        hfield, partner, inputs.shape_body, inputs.body_flags, inputs.include_static_kinematic_pairs != 0
    ):
        return TERRAIN_PAIR_FILTERED, frame, heightfield, cells, a, b, segment, reach
    local = wp.transform_multiply(wp.transform_inverse(frame), inputs.shape_transform[partner])
    size = inputs.shape_data[partner]
    reach = size[0] + inputs.shape_data[hfield][3] + size[3] + inputs.shape_gap[hfield] + inputs.shape_gap[partner]
    a = wp.transform_get_translation(local)
    b = a
    if inputs.shape_type[partner] == _CAPSULE:
        segment = 1
        axis = wp.transform_vector(local, wp.vec3(0.0, 0.0, size[1]))
        b = a - axis
        a = a + axis
    lo = wp.min(a, b)
    hi = wp.max(a, b)
    surface = data.surfaces[heightfield]
    cells = terrain_box_cells(surface, lo[0] - reach, lo[1] - reach, hi[0] + reach, hi[1] + reach)
    if cells[1] < cells[0] or cells[3] < cells[2]:
        return TERRAIN_PAIR_OUTSIDE, frame, heightfield, cells, a, b, segment, reach
    # The lowest point minus the reach, lowered by its own float32 rounding and by the cluster tolerance (a face row's
    # terrain point lies on its cluster plane, up to TERRAIN_FLAT_EPS above its triangle's highest vertex). The exit
    # drops no row but one claimed within the band beyond its triangle, where a steep face's plane may rise a band times
    # its slope higher, and then only within that of the reach limit.
    low = lo[2] - reach
    low = low - 4.0 * _EPS32 * (wp.abs(lo[2]) + reach) - _FLAT
    if terrain_tile_exit(data, surface, cells[0], cells[1], cells[2], cells[3], low):
        return TERRAIN_PAIR_TILE_EXIT, frame, heightfield, cells, a, b, segment, reach
    return 0, frame, heightfield, cells, a, b, segment, reach


@wp.func
def _window_points(window: TerrainWindow, a: wp.vec3, b: wp.vec3, segment: int):
    """A core's ends in the window frame, each rounded once from float64 (a sphere's centre converted once)."""
    ka = terrain_window_point(window, a)
    kb = ka
    if segment != 0:
        kb = terrain_window_point(window, b)
    return ka, kb


@wp.func
def _point_extent(ka: wp.vec3, kb: wp.vec3) -> float:
    """The largest coordinate magnitude of a core's ends in the window frame (part of the window's band)."""
    return wp.max(
        wp.max(wp.max(wp.abs(ka[0]), wp.abs(ka[1])), wp.abs(ka[2])),
        wp.max(wp.max(wp.abs(kb[0]), wp.abs(kb[1])), wp.abs(kb[2])),
    )


@wp.func
def _pair_core(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    ka: wp.vec3,
    kb: wp.vec3,
    segment: int,
    reach: float,
    eps: float,
) -> tuple[TerrainPlanes, _PointCore]:
    """The core from its ends in the window frame (:func:`_window_points`), with its inside tests and drop level."""
    core = _PointCore()
    core.a = ka
    core.b = kb
    core.segment = segment
    core.reach = reach
    core.eps = eps
    planes, below_a = terrain_point_below_cached(data, window, planes, core.a)
    below_b = below_a
    if segment != 0:
        planes, below_b = terrain_point_below_cached(data, window, planes, core.b)
    core.skip = 1
    if below_a:
        core.below_a = 1
        core.skip = 0
    if below_b:
        core.below_b = 1
        core.skip = 0
    core.low = wp.min(core.a[2], core.b[2]) - reach
    return planes, core


@wp.func
def _exit_row(
    data: TerrainFeatureData, window: TerrainWindow, planes: TerrainPlanes, k: wp.vec3, end: int
) -> tuple[TerrainPlanes, _Claim]:
    """G3: the face vertically above an end inside the solid that has no exit, with its plane's normal and depth."""
    t = terrain_locate(window, k[0], k[1])
    if t[0] < 0:
        return planes, _Claim()
    planes, n, a = terrain_plane(data, window, planes, t[0], t[1], t[2])
    d = wp.dot(n, k - a)
    if d >= 0.0:
        return planes, _Claim()
    claim = _Claim()
    claim.kind = _ROW
    claim.q = k - d * n
    claim.k = k
    claim.n = n
    claim.d = d
    claim.feature = terrain_feature_id(t[0], t[1], TERRAIN_FEATURE_RECOVERY, t[2], end)
    return planes, claim


@wp.func
def _wide(cells: wp.vec4i) -> int:
    if cells[1] - cells[0] + 1 >= _WIDE or cells[3] - cells[2] + 1 >= _WIDE:
        return TERRAIN_PAIR_WIDE
    return 0


# A pair's slots hold, while its window is walked, its distinct rows (slots 0-5) and the best exit of each end inside
# the solid (slots 6 and 7, feature -1 for none), in the window frame: rows are rare against the features walked, so
# this keeps them out of the registers every feature evaluation needs. With both ends outside, slot 7 holds the axis's
# deepest crossing under a crease instead (_WITNESS), for the axis interior's G3.
_EXIT_SLOT = wp.constant(6)
_WITNESS_SLOT = wp.constant(7)
_HELD = wp.constant(6)


@wp.func
def _hold(out: TerrainLawRows, slot: int, held: int, claim: _Claim, eps: float, limit: int) -> int:
    """Hold a claim's row (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`)."""
    return terrain_hold_row(out, slot, held, claim.q, claim.n, claim.d, claim.feature, eps, limit)


@wp.func
def _offer_exit(out: TerrainLawRows, slot: int, end: int, claim: _Claim):
    """G2: keep an end's exit with the smallest translation (the largest core distance; ties: the first in order)."""
    s = slot + _EXIT_SLOT + end
    if out.feature[s] < 0 or claim.d > out.d[s]:
        out.q[s] = claim.q
        out.n[s] = claim.n
        out.d[s] = claim.d
        out.feature[s] = claim.feature


@wp.func
def _walk_claim(out: TerrainLawRows, slot: int, held: int, candidates: int, claim: _Claim, eps: float):
    """Take one item's claim: hold a row (counting the finish's candidates), or offer an exit (or, in end ``b``'s slot
    with both ends outside, the axis's crossing: the deepest is kept)."""
    if claim.kind == _ROW:
        return _hold(out, slot, held, claim, eps, _HELD), candidates + 1
    if claim.kind == _EXIT_A or claim.kind == _EXIT_B:
        _offer_exit(out, slot, claim.kind - _EXIT_A, claim)
    return held, candidates


@wp.func
def _axis_exit_held(out: TerrainLawRows, slot: int, held: int) -> bool:
    """Whether a pair's held rows include an exit of its axis interior (an axis row inside the solid)."""
    for h in range(held):
        if (out.feature[slot + h] & 7) == _SLOT_AXIS and out.d[slot + h] < 0.0:
            return True
    return False


@wp.func
def _close_pair(
    out: TerrainLawRows,
    pair: int,
    slot: int,
    state: int,
    held: int,
    candidates: int,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    frame: wp.transform,
):
    """Add the ends' exits (G2, G3) and the axis interior's G3, then write the held rows in the world frame, or defer
    the pair to the finish.

    The finish keeps a pair's distinct rows as they are when no plane group holds three of them (its implied-row
    and per-group steps act on larger groups), they fit the pair's :data:`TERRAIN_POINT_ROWS` slots and its
    candidates fit its list; any other pair is appended to the deferred list for :func:`terrain_point_rows_full`.
    """
    full = held > _HELD
    if not full:
        for end in range(2):
            if (end == 0 and core.below_a != 0) or (end == 1 and core.segment != 0 and core.below_b != 0):
                # The exit is read before the slot it may land in is written: end 0 lands at most in slot 6,
                # end 1 at most in slot 7.
                s = slot + _EXIT_SLOT + end
                claim = _Claim()
                if out.feature[s] >= 0:
                    claim.kind = _ROW
                    claim.q = out.q[s]
                    claim.n = out.n[s]
                    claim.d = out.d[s]
                    claim.feature = out.feature[s]
                else:
                    k = core.a
                    if end == 1:
                        k = core.b
                    planes, claim = _exit_row(data, window, planes, k, end)
                if claim.kind != 0:
                    candidates += 1
                    held = _hold(out, slot, held, claim, core.eps, _ROWS)
        if core.segment != 0 and core.below_a == 0 and core.below_b == 0:
            # G3 for the axis interior: read before its row lands (at most in slot 6, the ends adding none).
            s = slot + _WITNESS_SLOT
            if out.feature[s] >= 0 and not _axis_exit_held(out, slot, held):
                planes, claim = _axis_recovery(data, window, planes, out.q[s], out.feature[s])
                if claim.kind != 0:
                    candidates += 1
                    held = _hold(out, slot, held, claim, core.eps, _ROWS)
        full = held > _ROWS or candidates > _CANDIDATES or terrain_plane_group_of_three(out, slot, held, core.eps)
    if full:
        # Left to terrain_point_rows_full, which writes every slot of the pair.
        terrain_defer_pair(out, _CLASS, pair, state)
        return
    for h in range(held):
        terrain_store_row(
            out, slot + h, out.q[slot + h], out.n[slot + h], out.d[slot + h], out.feature[slot + h], window, frame
        )
    out.count[pair] = held
    out.state[pair] = state


@wp.func
def _open_pair(out: TerrainLawRows, slot: int):
    """Mark a pair's exit slots empty before its window is walked."""
    out.feature[slot + _EXIT_SLOT] = -1
    out.feature[slot + _EXIT_SLOT + 1] = -1


@wp.func
def _kept_word(
    data: TerrainFeatureData, window: TerrainWindow, core: _PointCore, margins: wp.vec3, cells: wp.vec4i, word: int
) -> int:
    """The window vertices of mask word ``word`` (window indices from ``31 word``, one bit each) the drop keeps."""
    r0 = cells[0]
    c0 = cells[2]
    width = cells[3] + 2 - c0
    vertices = (cells[1] + 2 - r0) * width
    first = word * _MASK_BITS
    m = int(0)
    for v in range(first, wp.min(vertices, first + _MASK_BITS)):
        if not _vertex_dropped(data, window, core, margins, r0 + v // width, c0 + v % width):
            m = m | (1 << (v - first))
    return m


@wp.func
def _walk_sphere(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    out: TerrainLawRows,
    slot: int,
    cells: wp.vec4i,
):
    """A sphere's walk: its centre's six items of every window vertex the drop keeps, in window order, into the pair's
    slots (see :func:`terrain_point_rows`). Returns the plane cache, the distinct rows held and the candidates."""
    r0 = cells[0]
    r1 = cells[1]
    c0 = cells[2]
    c1 = cells[3]
    width = c1 + 2 - c0
    vertices = (r1 + 2 - r0) * width
    margins = _cell_margins(window, core)
    held = int(0)
    candidates = int(0)
    for word in range((vertices + _MASK_BITS - 1) // _MASK_BITS):
        if held > _HELD:
            break
        m = _kept_word(data, window, core, margins, cells, word)
        while m != 0 and held <= _HELD:
            v = word * _MASK_BITS + _lowest_bit(m)
            m = m & (m - 1)
            for feature in range(_SPHERE_ITEMS):
                planes, claim = _end_item(
                    data,
                    window,
                    planes,
                    core,
                    0,
                    core.a,
                    core.below_a,
                    r0 + v // width,
                    c0 + v % width,
                    r1,
                    c1,
                    feature,
                )
                held, candidates = _walk_claim(out, slot, held, candidates, claim, core.eps)
    return planes, held, candidates


@wp.func
def _walk_segment(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    out: TerrainLawRows,
    slot: int,
    cells: wp.vec4i,
):
    """A capsule's walk: the items (both ends' and the axis's) each window vertex the drop keeps can yield anything at
    (:func:`_kept_items`), in window order, into the pair's slots (see :func:`terrain_point_rows`). Returns the plane
    cache, the rows held and the candidates."""
    r0 = cells[0]
    r1 = cells[1]
    c0 = cells[2]
    c1 = cells[3]
    width = c1 + 2 - c0
    vertices = (r1 + 2 - r0) * width
    margins = _cell_margins(window, core)
    held = int(0)
    candidates = int(0)
    for word in range((vertices + _MASK_BITS - 1) // _MASK_BITS):
        if held > _HELD:
            break
        m = _kept_word(data, window, core, margins, cells, word)
        while m != 0 and held <= _HELD:
            v = word * _MASK_BITS + _lowest_bit(m)
            m = m & (m - 1)
            r = r0 + v // width
            c = c0 + v % width
            kept = _kept_items(data, window, core, margins, r, c)
            for item in range(_ITEMS):
                if ((kept >> item) & 1) != 0:
                    planes, claim = _point_item(data, window, planes, core, r, c, r1, c1, item)
                    held, candidates = _walk_claim(out, slot, held, candidates, claim, core.eps)
    return planes, held, candidates


@wp.kernel(enable_backward=False, launch_bounds=(TERRAIN_LAW_BLOCK, 16))
def terrain_point_rows(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    work: TerrainPairWork,
):
    """The law's rows of sphere and capsule pairs, one thread per pair: it walks the windows of up to 16 cells and
    queues the larger ones for :func:`terrain_point_rows_tiled`.

    A walked pair holds its distinct rows in its slots and writes them when the finish would keep them as they are;
    any other pair is left to :func:`terrain_point_rows_full` (the finish's candidate list and code would otherwise
    cost every thread their registers and stack). A pair from ``work.first`` on (the tiled segment) leaves here
    through the whole-pair rejects, or is queued in ``work`` to be walked by a tile of lanes.
    """
    i = wp.tid()
    pair = begin + i
    state, frame, heightfield, cells, a, b, segment, reach = _pair_setup(inputs, data, pair)
    if state != 0:
        terrain_pair_exit(out, pair, state)
        return
    if pair >= work.first:
        # A larger window that the rejects keep: a tile of lanes walks it.
        work.pairs[wp.atomic_add(work.count, 0, 1)] = pair
        return
    slot = slot_begin + i * _ROWS
    window = terrain_window(data, heightfield, cells[0], cells[2])
    ka, kb = _window_points(window, a, b, segment)
    eps = terrain_eps(terrain_window_extent(data, window, cells[1], cells[3], _point_extent(ka, kb)))
    planes = terrain_planes()
    planes, core = _pair_core(data, window, planes, ka, kb, segment, reach, eps)
    _open_pair(out, slot)
    held = int(0)
    candidates = int(0)
    # A sphere walks its centre's items alone, in code of its own: its warps never run the capsule's.
    if segment == 0:
        planes, held, candidates = _walk_sphere(data, window, planes, core, out, slot, cells)
    else:
        planes, held, candidates = _walk_segment(data, window, planes, core, out, slot, cells)
    _close_pair(out, pair, slot, _wide(cells), held, candidates, data, window, planes, core, frame)


@wp.func
def _vertex_marks(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: _PointCore,
    margins: wp.vec3,
    r: int,
    c: int,
    r1: int,
    c1: int,
    items: int,
):
    """Bit ``item`` for every item of window vertex ``(r, c)`` whose claim is not nothing, of the items it can yield
    anything at all (:func:`_kept_items`), and the updated plane cache."""
    m = int(0)
    kept = _kept_items(data, window, core, margins, r, c)
    for item in range(items):
        if ((kept >> item) & 1) != 0:
            planes, claim = _point_item(data, window, planes, core, r, c, r1, c1, item)
            if claim.kind != 0:
                m = m | (1 << item)
    return planes, m


@wp.func
def _tiled_window(inputs: TerrainPairInputs, data: TerrainFeatureData, pair: int, lane: int, lanes: int):
    """A pair's whole-pair state and, past the rejects, its window, plane cache and core, the block's lanes splitting
    the window's heights for its band (every lane returns the same values).

    Returns ``(state, frame, cells, window, planes, core)``; past a reject (``state`` nonzero) only the state and the
    cells are set.
    """
    state, frame, heightfield, cells, a, b, segment, reach = _pair_setup(inputs, data, pair)
    window = TerrainWindow()
    planes = terrain_planes()
    core = _PointCore()
    if state != 0:
        return state, frame, cells, window, planes, core
    r0 = cells[0]
    c0 = cells[2]
    window = terrain_window(data, heightfield, r0, c0)
    width = cells[3] - c0 + 2
    vertices = (cells[1] - r0 + 2) * width
    surface = window.surface
    extent = float(0.0)
    for v in range(lane, vertices, lanes):
        extent = wp.max(extent, wp.abs(terrain_height(data, surface, r0 + v // width, c0 + v % width) - window.z_ref))
    extent = wp.tile_extract(wp.tile_max(wp.tile(extent)), 0)
    size = wp.max(float(cells[3] + 1 - c0) * surface.dx, float(cells[1] + 1 - r0) * surface.dy)
    ka, kb = _window_points(window, a, b, segment)
    eps = terrain_eps(wp.max(wp.max(extent, size), _point_extent(ka, kb)))
    planes, core = _pair_core(data, window, planes, ka, kb, segment, reach, eps)
    return state, frame, cells, window, planes, core


@wp.func
def _tiled_pair(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    pair: int,
    lane: int,
):
    """Walk one pair's window with the block's lanes (see :func:`terrain_point_rows_tiled`); every lane returns
    together, ready for the next pair."""
    lanes = wp.block_dim()
    slot = slot_begin + (pair - begin) * _ROWS
    state, frame, cells, window, planes, core = _tiled_window(inputs, data, pair, lane, lanes)
    if state != 0:
        # A queued pair passed these rejects in the one-thread kernel; every lane reads the same pair regardless.
        if lane == 0:
            terrain_pair_exit(out, pair, state)
        return
    r0 = cells[0]
    r1 = cells[1]
    c0 = cells[2]
    c1 = cells[3]
    width = c1 - c0 + 2
    vertices = (r1 - r0 + 2) * width
    items = _vertex_items(core)
    margins = _cell_margins(window, core)
    masked = vertices <= _MASK_VERTICES
    marks = wp.tile_zeros(shape=_MASK_VERTICES, dtype=wp.int32, storage="shared")
    marked = _Mask()
    if masked:
        # The lanes split the vertices: per vertex its items that yield anything (in ``marks``), and as bit words
        # the vertices with any.
        for start in range(0, vertices, lanes):
            v = start + lane
            m = int(0)
            if v < vertices:
                planes, m = _vertex_marks(
                    data, window, planes, core, margins, r0 + v // width, c0 + v % width, r1, c1, items
                )
                if m != 0:
                    marked[v // 32] = marked[v // 32] | (1 << (v % 32))
            wp.tile_scatter_masked(marks, wp.min(v, _MASK_VERTICES - 1), m, v < vertices)
        # Lanes mark disjoint vertices, so a sum of their words is their union.
        for word in range((vertices + 31) // 32):
            marked[word] = wp.tile_extract(wp.tile_sum(wp.tile(marked[word])), 0)
    if lane == 0:
        # The marked items in window order: the nonzero claims the one-thread walk takes, in its order.
        _open_pair(out, slot)
        held = int(0)
        candidates = int(0)
        v = int(0)
        while v < vertices and held <= _HELD:
            if masked:
                if marked[v // 32] == 0:
                    v = (v // 32 + 1) * 32
                    continue
            # The items to evaluate: the vertex's marks in a marked window, else every item of a kept vertex.
            m = int(0)
            if masked:
                if ((marked[v // 32] >> (v % 32)) & 1) != 0:
                    m = wp.tile_extract(marks, v)
            else:
                m = _kept_items(data, window, core, margins, r0 + v // width, c0 + v % width)
            if m != 0:
                for item in range(items):
                    if ((m >> item) & 1) != 0:
                        planes, claim = _point_item(
                            data, window, planes, core, r0 + v // width, c0 + v % width, r1, c1, item
                        )
                        held, candidates = _walk_claim(out, slot, held, candidates, claim, core.eps)
            v += 1
        _close_pair(out, pair, slot, _wide(cells), held, candidates, data, window, planes, core, frame)


@wp.kernel(enable_backward=False)
def terrain_point_rows_tiled(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    work: TerrainPairWork,
):
    """The law's rows of the sphere and capsule pairs with long windows that the one-thread kernel queued,
    ``wp.block_dim()`` lanes per pair.

    Launched with ``work.blocks`` blocks whatever the segment's size: block ``b`` takes the queued pairs ``b``,
    ``b + blocks``, ... For each, the lanes split the window's heights (its band) and its vertices, marking those
    whose items yield a row or an exit; the first lane evaluates the marked ones in window order, which gives a pair
    the rows (or the deferral) :func:`terrain_point_rows` gives it.
    """
    block, lane = wp.tid()
    for entry in range(block, work.count[0], work.blocks):
        _tiled_pair(begin, slot_begin, inputs, data, out, work.pairs[entry], lane)


@wp.func
def _finish_candidate(scratch: TerrainFinishScratch, base: int, count: int, dropped: int, claim: _Claim):
    """Append a claim's row to a full-finish block's candidates (from ``base``; counted as dropped when full)."""
    if count < _CANDIDATES:
        scratch.q[base + count] = claim.q
        scratch.k[base + count] = claim.k
        scratch.n[base + count] = claim.n
        scratch.d[base + count] = claim.d
        scratch.feature[base + count] = claim.feature
        return count + 1, dropped
    return count, dropped + 1


@wp.func
def _finish_input(scratch: TerrainFinishScratch, block: int, count: int, dropped: int):
    """Copy a full-finish block's candidates into its :class:`~newton._src.geometry.terrain_law.TerrainRows` record,
    the finish's input (entries past ``count``, never read, keep earlier pairs' values)."""
    base = block * _CANDIDATES
    scratch.rows[block].count = count
    scratch.rows[block].dropped = dropped
    for row in range(count):
        q = scratch.q[base + row]
        k = scratch.k[base + row]
        n = scratch.n[base + row]
        for j in range(3):
            scratch.rows[block].q[row, j] = q[j]
            scratch.rows[block].k[row, j] = k[j]
            scratch.rows[block].n[row, j] = n[j]
        scratch.rows[block].d[row] = scratch.d[base + row]
        scratch.rows[block].feature[row] = scratch.feature[base + row]


@wp.func
def _finish_deferred(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    scratch: TerrainFinishScratch,
    block: int,
    lane: int,
    pair: int,
):
    """Walk a deferred pair's window into the full candidate list (``block``'s scratch) with the block's lanes, and
    run the law's finish on the first lane (see :func:`terrain_point_rows_full`); every lane returns together."""
    lanes = wp.block_dim()
    state, frame, cells, window, planes, core = _tiled_window(inputs, data, pair, lane, lanes)
    if state != 0:
        # A deferred pair passed these rejects in its walk; every lane reads the same pair regardless.
        if lane == 0:
            terrain_pair_exit(out, pair, state)
        return
    r0 = cells[0]
    r1 = cells[1]
    c0 = cells[2]
    c1 = cells[3]
    width = c1 - c0 + 2
    vertices = (r1 - r0 + 2) * width
    items = _vertex_items(core)
    margins = _cell_margins(window, core)
    masked = vertices <= _MASK_VERTICES
    marks = wp.tile_zeros(shape=_MASK_VERTICES, dtype=wp.int32, storage="shared")
    marked = _Mask()
    if masked:
        # The lanes split the vertices: per vertex its items that yield anything (in ``marks``), and as bit words
        # the vertices with any.
        for start in range(0, vertices, lanes):
            v = start + lane
            m = int(0)
            if v < vertices:
                planes, m = _vertex_marks(
                    data, window, planes, core, margins, r0 + v // width, c0 + v % width, r1, c1, items
                )
                if m != 0:
                    marked[v // 32] = marked[v // 32] | (1 << (v % 32))
            wp.tile_scatter_masked(marks, wp.min(v, _MASK_VERTICES - 1), m, v < vertices)
        # Lanes mark disjoint vertices, so a sum of their words is their union.
        for word in range((vertices + 31) // 32):
            marked[word] = wp.tile_extract(wp.tile_sum(wp.tile(marked[word])), 0)
    if lane == 0:
        base = block * _CANDIDATES
        count = int(0)
        dropped = int(0)
        exit_a = _Claim()
        exit_b = _Claim()
        # The marked items in window order: the nonzero claims of a walk over every vertex the drop keeps.
        v = int(0)
        while v < vertices:
            if masked:
                if marked[v // 32] == 0:
                    v = (v // 32 + 1) * 32
                    continue
            # The items to evaluate: the vertex's marks in a marked window, else every item of a kept vertex.
            m = int(0)
            if masked:
                if ((marked[v // 32] >> (v % 32)) & 1) != 0:
                    m = wp.tile_extract(marks, v)
            else:
                m = _kept_items(data, window, core, margins, r0 + v // width, c0 + v % width)
            if m != 0:
                for item in range(items):
                    if ((m >> item) & 1) != 0:
                        planes, claim = _point_item(
                            data, window, planes, core, r0 + v // width, c0 + v % width, r1, c1, item
                        )
                        # G2: an end's exit with the smallest translation, the first in window order on ties.
                        if claim.kind == _EXIT_A and (exit_a.kind == 0 or claim.d > exit_a.d):
                            exit_a = claim
                        elif claim.kind == _EXIT_B and (exit_b.kind == 0 or claim.d > exit_b.d):
                            # With both ends outside: the axis's deepest crossing under a crease (_WITNESS).
                            exit_b = claim
                        elif claim.kind == _ROW:
                            count, dropped = _finish_candidate(scratch, base, count, dropped, claim)
            v += 1
        # G2 and G3: one exit per end inside the solid, after the window's rows.
        for end in range(2):
            if (end == 0 and core.below_a != 0) or (end == 1 and core.segment != 0 and core.below_b != 0):
                claim = exit_a
                k = core.a
                if end == 1:
                    claim = exit_b
                    k = core.b
                if claim.kind != 0:
                    claim.kind = _ROW
                    claim.k = k
                else:
                    planes, claim = _exit_row(data, window, planes, k, end)
                if claim.kind != 0:
                    count, dropped = _finish_candidate(scratch, base, count, dropped, claim)
        # G3 for the axis interior: both ends outside, no axis exit, a crossing of the axis under a crease.
        witness = exit_b
        if core.segment != 0 and core.below_a == 0 and core.below_b == 0 and witness.kind != 0:
            axis_exit = bool(False)
            for row in range(count):
                if (scratch.feature[base + row] & 7) == _SLOT_AXIS and scratch.d[base + row] < 0.0:
                    axis_exit = True
            if not axis_exit:
                planes, claim = _axis_recovery(data, window, planes, witness.q, witness.feature)
                if claim.kind != 0:
                    count, dropped = _finish_candidate(scratch, base, count, dropped, claim)
        _finish_input(scratch, block, count, dropped)
        order, kept, flags = terrain_finish(scratch.rows[block], core.eps, _ROWS)
        slot = slot_begin + (pair - begin) * _ROWS
        for kept_row in range(kept):
            row = base + order[kept_row]
            terrain_store_row(
                out,
                slot + kept_row,
                scratch.q[row],
                scratch.n[row],
                scratch.d[row],
                scratch.feature[row],
                window,
                frame,
            )
        out.count[pair] = kept
        out.state[pair] = _wide(cells) | terrain_finish_state(flags)
        terrain_report(out.status, out.overflow, flags, True)


@wp.kernel(enable_backward=False)
def terrain_point_rows_full(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    scratch: TerrainFinishScratch,
):
    """Finish the sphere and capsule pairs the other two kernels deferred (a plane group of three rows, more rows than
    the slots, or more candidates than the list), ``wp.block_dim()`` lanes per pair.

    Launched after them with :attr:`~newton._src.geometry.terrain_shadow.TerrainFinishScratch.blocks` blocks
    (``wp.launch_tiled``), whatever the class's size: block ``b`` takes the deferred pairs ``b``, ``b + blocks``, ...
    (the order is free: a pair's rows depend on the pair alone), so that an empty list costs a near-empty launch. The
    lanes mark the pair's window as the tiled kernel's do; the first lane walks the marked items again into the full
    candidate list, in the block's scratch, and runs the law's finish
    (:func:`~newton._src.geometry.terrain_law.terrain_finish`: merge, implied rows, four rows per plane group, at most
    :data:`TERRAIN_POINT_ROWS` rows, each group's deepest first on overflow).
    """
    block, lane = wp.tid()
    for entry in range(block, out.deferred_count[_CLASS], scratch.blocks):
        _finish_deferred(begin, slot_begin, inputs, data, out, scratch, block, lane, out.deferred[begin + entry])

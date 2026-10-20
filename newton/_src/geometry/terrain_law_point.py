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

Thread mapping (the pair table's lane classes): :func:`terrain_point_rows` runs one thread per pair; it first marks
the window's vertices the drop keeps in a bit mask and then walks the marked ones in window order, so that a warp's
lanes evaluate their k-th kept vertices together instead of one window position that most lanes drop (a model of
the warps' issued instructions on recorded quadruped foot spheres, per-stage SASS counts weighted by what any lane of
a warp runs: -24% and -27% on two suite scenes against walking every window position).
:func:`terrain_point_rows_tiled` runs
:data:`~newton._src.geometry.terrain_contacts.TERRAIN_TILE_LANES` lanes per pair (``wp.launch_tiled``) for windows
of more cells: the lanes split the window's heights and vertices, mark the vertices whose items yield anything, and
the first lane evaluates those in window order, so a pair gets the rows the one-thread kernel gives it. Both hold a
pair's distinct rows in its own slots (up to six, merged as the finish merges) and its ends' exits, and write them
when the finish keeps them as they are: no plane group holds three of them (the finish's implied-row and per-group
steps act on larger groups), they fit the eight slots and the candidates fit the finish's list. That is every
recorded pair of the suite's rough scenes; any other pair is appended to a deferred list, and
:func:`terrain_point_rows_full`, launched after them, walks it again into the full candidate list and runs the
finish. Inlined into the walks, the finish took every thread to 255 registers and a 10 KB stack on sm_120; apart,
the one-thread walk takes 128 registers (launch bounds of two 256-thread blocks per SM) and a 104-byte stack, the
tiled walk 156 registers, and only deferred pairs, in full warps, pay for the finish. All three have the per-type
kernel signature ``(begin, slot_begin, inputs, data, out)`` of
:class:`~newton._src.geometry.terrain_law.TerrainPairInputs`.

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
    TerrainRows,
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
from .types import GeoType

# Every float32 operation rounds as on the CPU (see the module docstring).
wp.set_module_options({"fuse_fp": False})

TERRAIN_POINT_ROWS = TERRAIN_PARTNER_ROWS[TERRAIN_PARTNER_POINT]
"""Rows a sphere or capsule pair keeps (the point class's row limit, the slots per pair): 8."""
TERRAIN_POINT_MASK_WORDS = 8
"""Words of the tiled kernel's productive-vertex mask: windows of up to 256 vertices; larger windows are walked by
the first lane alone."""
TERRAIN_POINT_ITEMS = 16
"""Items a window vertex owns for a capsule (:func:`_point_item`); a sphere's are the first six."""

_ROWS = wp.constant(TERRAIN_POINT_ROWS)
_CLASS = wp.constant(TERRAIN_PARTNER_POINT)
_CANDIDATES = wp.constant(TERRAIN_ROW_CANDIDATES)
_MASK_WORDS = wp.constant(TERRAIN_POINT_MASK_WORDS)
_ITEMS = wp.constant(TERRAIN_POINT_ITEMS)
_SPHERE_ITEMS = wp.constant(6)
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
    sub 0, sub 1, the x-edge, the y-edge, the diagonal and the vertex for the sphere centre or capsule end ``a``,
    items 6-11 the same for end ``b``, items 12-15 the edges and the vertex for the capsule's axis.
    """
    end = item // 6
    feature = item - 6 * end
    if end == 2:
        feature = item - 10
    k = core.a
    below = core.below_a
    if end == 1:
        k = core.b
        below = core.below_b
    inside = r <= r1 and c <= c1
    if feature < 2:
        if not inside:
            return planes, _Claim()
        planes, n, a = terrain_plane(data, window, planes, r, c, feature)
        return planes, _end_face(data, window, core, end, k, below, r, c, feature, n, a)
    if feature == 5:
        if end == 2:
            return _axis_vertex(data, window, planes, core, r, c)
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
    if end == 2:
        return _axis_edge(data, window, planes, core, r, c, which, v0, v1)
    return _end_edge(data, window, planes, core, end, k, below, r, c, which, v0, v1)


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
def _vertex_dropped(
    data: TerrainFeatureData, window: TerrainWindow, core: _PointCore, margins: wp.vec3, r: int, c: int
) -> bool:
    """Whether no feature vertex ``(r, c)`` owns can yield a row, so that the walk passes the vertex by.

    The features lie in the vertex's cell (its faces and diagonal, its x- and y-edges, itself), and a row's terrain
    point in the box of the cell's vertices (those in the grid), widened in xy by ``margins`` (a face's point may lie
    off its triangle within the band, :func:`_cell_margins`) and in z by how far a cluster plane may leave the
    vertices there (the cluster tolerance, steepened by the cell's slope, and the plane over the xy widening), all
    four times over plus the band. A row of a core point outside the solid has its core point within the reach of
    its terrain point, so a core farther than the reach from the box yields none: a sphere centre by its 3-D distance,
    a capsule by height alone (every vertex below the core's lowest point by more than the reach). Neither applies
    while a core point lies inside the solid (its exits are not bounded by the reach).
    """
    if core.skip == 0:
        return False
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
    if core.segment != 0:
        return zhi + zm < core.low
    k = core.a
    x = k[0] - float(c - window.c0) * surface.dx
    y = k[1] - float(r - window.r0) * surface.dy
    ox = wp.max(wp.max(-margins[0] - x, x - surface.dx - margins[0]), 0.0)
    oy = wp.max(wp.max(-margins[1] - y, y - surface.dy - margins[1]), 0.0)
    oz = wp.max(wp.max(zlo - zm - k[2], k[2] - zhi - zm), 0.0)
    return ox * ox + oy * oy + oz * oz > core.reach * core.reach


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
def _pair_core(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    a: wp.vec3,
    b: wp.vec3,
    segment: int,
    reach: float,
    eps: float,
) -> tuple[TerrainPlanes, _PointCore]:
    """The core in the window frame, with its inside tests and cell-skip level."""
    core = _PointCore()
    core.a = terrain_window_point(window, a)
    core.b = terrain_window_point(window, b)
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
def _core_extent(window: TerrainWindow, a: wp.vec3, b: wp.vec3) -> float:
    ka = terrain_window_point(window, a)
    kb = terrain_window_point(window, b)
    return wp.max(
        wp.max(wp.max(wp.abs(ka[0]), wp.abs(ka[1])), wp.abs(ka[2])),
        wp.max(wp.max(wp.abs(kb[0]), wp.abs(kb[1])), wp.abs(kb[2])),
    )


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


@wp.kernel(enable_backward=False, launch_bounds=(256, 2))
def terrain_point_rows(
    begin: int, slot_begin: int, inputs: TerrainPairInputs, data: TerrainFeatureData, out: TerrainLawRows
):
    """The law's rows of sphere and capsule pairs, one thread per pair (windows of up to 16 cells).

    A pair holds its distinct rows in its slots and writes them when the finish would keep them as they are; any
    other pair is left to :func:`terrain_point_rows_full` (the finish's candidate list and code would otherwise cost
    every thread their registers and stack).
    """
    i = wp.tid()
    pair = begin + i
    slot = slot_begin + i * _ROWS
    state, frame, heightfield, cells, a, b, segment, reach = _pair_setup(inputs, data, pair)
    if state != 0:
        terrain_pair_exit(out, pair, state)
        return
    r1 = cells[1]
    c1 = cells[3]
    window = terrain_window(data, heightfield, cells[0], cells[2])
    eps = terrain_eps(terrain_window_extent(data, window, r1, c1, _core_extent(window, a, b)))
    planes = terrain_planes()
    planes, core = _pair_core(data, window, planes, a, b, segment, reach, eps)
    _open_pair(out, slot)
    held = int(0)
    candidates = int(0)
    items = _vertex_items(core)
    # The vertices to walk as bit masks first, so that a warp's lanes walk their k-th vertices together rather than
    # one window position, which most lanes would pass by. Vertices past two words (none for windows of 16 cells)
    # are walked after them.
    r0 = cells[0]
    c0 = cells[2]
    width = c1 + 2 - c0
    vertices = (r1 + 2 - r0) * width
    margins = _cell_margins(window, core)
    low_word = int(0)
    high_word = int(0)
    for v in range(wp.min(vertices, 2 * _MASK_BITS)):
        if not _vertex_dropped(data, window, core, margins, r0 + v // width, c0 + v % width):
            if v < _MASK_BITS:
                low_word = low_word | (1 << v)
            else:
                high_word = high_word | (1 << (v - _MASK_BITS))
    for word in range(2):
        m = low_word
        if word == 1:
            m = high_word
        while m != 0 and held <= _HELD:
            v = word * _MASK_BITS + _lowest_bit(m)
            m = m & (m - 1)
            for item in range(items):
                planes, claim = _point_item(data, window, planes, core, r0 + v // width, c0 + v % width, r1, c1, item)
                held, candidates = _walk_claim(out, slot, held, candidates, claim, eps)
    for v in range(2 * _MASK_BITS, vertices):
        if held <= _HELD and not _vertex_dropped(data, window, core, margins, r0 + v // width, c0 + v % width):
            for item in range(items):
                planes, claim = _point_item(data, window, planes, core, r0 + v // width, c0 + v % width, r1, c1, item)
                held, candidates = _walk_claim(out, slot, held, candidates, claim, eps)
    _close_pair(out, pair, slot, _wide(cells), held, candidates, data, window, planes, core, frame)


@wp.kernel(enable_backward=False)
def terrain_point_rows_tiled(
    begin: int, slot_begin: int, inputs: TerrainPairInputs, data: TerrainFeatureData, out: TerrainLawRows
):
    """The law's rows of sphere and capsule pairs with long windows, ``wp.block_dim()`` lanes per pair.

    The lanes split the window's heights (its band) and its vertices, marking those whose items yield a row or an
    exit; the first lane evaluates the marked ones in window order, which gives a pair the rows (or the deferral)
    :func:`terrain_point_rows` gives it.
    """
    i, lane = wp.tid()
    lanes = wp.block_dim()
    pair = begin + i
    slot = slot_begin + i * _ROWS
    state, frame, heightfield, cells, a, b, segment, reach = _pair_setup(inputs, data, pair)
    if state != 0:
        # Every lane reads the same pair: the exit is uniform.
        if lane == 0:
            terrain_pair_exit(out, pair, state)
        return
    r0 = cells[0]
    r1 = cells[1]
    c0 = cells[2]
    c1 = cells[3]
    window = terrain_window(data, heightfield, r0, c0)
    width = c1 - c0 + 2
    vertices = (r1 - r0 + 2) * width
    surface = window.surface
    extent = float(0.0)
    for v in range(lane, vertices, lanes):
        extent = wp.max(extent, wp.abs(terrain_height(data, surface, r0 + v // width, c0 + v % width) - window.z_ref))
    extent = wp.tile_extract(wp.tile_max(wp.tile(extent)), 0)
    size = wp.max(float(c1 + 1 - c0) * surface.dx, float(r1 + 1 - r0) * surface.dy)
    eps = terrain_eps(wp.max(wp.max(extent, size), _core_extent(window, a, b)))
    planes = terrain_planes()
    planes, core = _pair_core(data, window, planes, a, b, segment, reach, eps)
    items = _vertex_items(core)
    margins = _cell_margins(window, core)
    words = (vertices + 31) // 32
    marked = _Mask()
    if words <= _MASK_WORDS:
        for v in range(lane, vertices, lanes):
            r = r0 + v // width
            c = c0 + v % width
            if _vertex_dropped(data, window, core, margins, r, c):
                continue
            for item in range(items):
                planes, claim = _point_item(data, window, planes, core, r, c, r1, c1, item)
                if claim.kind != 0:
                    marked[v // 32] = marked[v // 32] | (1 << (v % 32))
        # Lanes mark disjoint vertices, so a sum of their words is their union.
        for word in range(words):
            marked[word] = wp.tile_extract(wp.tile_sum(wp.tile(marked[word])), 0)
    if lane != 0:
        return
    _open_pair(out, slot)
    held = int(0)
    candidates = int(0)
    for v in range(vertices):
        if held > _HELD:
            break
        if words <= _MASK_WORDS and (marked[v // 32] & (1 << (v % 32))) == 0:
            continue
        r = r0 + v // width
        c = c0 + v % width
        if _vertex_dropped(data, window, core, margins, r, c):
            continue
        for item in range(items):
            planes, claim = _point_item(data, window, planes, core, r, c, r1, c1, item)
            held, candidates = _walk_claim(out, slot, held, candidates, claim, eps)
    _close_pair(out, pair, slot, _wide(cells), held, candidates, data, window, planes, core, frame)


@wp.kernel(enable_backward=False)
def terrain_point_rows_full(
    begin: int, slot_begin: int, inputs: TerrainPairInputs, data: TerrainFeatureData, out: TerrainLawRows
):
    """Finish the sphere and capsule pairs the other two kernels deferred (a plane group of three rows, more rows than
    the slots, or more candidates than the list).

    Launched after them with one thread per pair of the point class; thread ``j`` takes the ``j``-th pair they
    appended to the deferred list (the order is free: a pair's rows depend on the pair alone), so that the few
    deferred pairs run in full warps. One thread walks the pair's window again into the full candidate list and runs
    the law's finish (:func:`~newton._src.geometry.terrain_law.terrain_finish`: merge, implied rows, four rows per
    plane group, at most :data:`TERRAIN_POINT_ROWS` rows, each group's deepest first on overflow).
    """
    entry = wp.tid()
    if entry >= out.deferred_count[_CLASS]:
        return
    pair = out.deferred[begin + entry]
    i = pair - begin
    _state, frame, heightfield, cells, a, b, segment, reach = _pair_setup(inputs, data, pair)
    r1 = cells[1]
    c1 = cells[3]
    window = terrain_window(data, heightfield, cells[0], cells[2])
    eps = terrain_eps(terrain_window_extent(data, window, r1, c1, _core_extent(window, a, b)))
    planes = terrain_planes()
    planes, core = _pair_core(data, window, planes, a, b, segment, reach, eps)
    rows = TerrainRows()
    exit_a = _Claim()
    exit_b = _Claim()
    items = _vertex_items(core)
    margins = _cell_margins(window, core)
    for r in range(cells[0], r1 + 2):
        for c in range(cells[2], c1 + 2):
            if _vertex_dropped(data, window, core, margins, r, c):
                continue
            for item in range(items):
                planes, claim = _point_item(data, window, planes, core, r, c, r1, c1, item)
                # G2: an end's exit with the smallest translation, the first in window order on ties.
                if claim.kind == _EXIT_A and (exit_a.kind == 0 or claim.d > exit_a.d):
                    exit_a = claim
                elif claim.kind == _EXIT_B and (exit_b.kind == 0 or claim.d > exit_b.d):
                    # With both ends outside: the axis's deepest crossing under a crease (_WITNESS).
                    exit_b = claim
                elif claim.kind == _ROW:
                    n = rows.count
                    if n < _CANDIDATES:
                        for j in range(3):
                            rows.q[n, j] = claim.q[j]
                            rows.k[n, j] = claim.k[j]
                            rows.n[n, j] = claim.n[j]
                        rows.d[n] = claim.d
                        rows.feature[n] = claim.feature
                        rows.count = n + 1
                    else:
                        rows.dropped = rows.dropped + 1
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
                n = rows.count
                if n < _CANDIDATES:
                    for j in range(3):
                        rows.q[n, j] = claim.q[j]
                        rows.k[n, j] = claim.k[j]
                        rows.n[n, j] = claim.n[j]
                    rows.d[n] = claim.d
                    rows.feature[n] = claim.feature
                    rows.count = n + 1
                else:
                    rows.dropped = rows.dropped + 1
    # G3 for the axis interior: both ends outside, no axis exit, a crossing of the axis under a crease.
    witness = exit_b
    if core.segment != 0 and core.below_a == 0 and core.below_b == 0 and witness.kind != 0:
        axis_exit = bool(False)
        for row in range(rows.count):
            if (rows.feature[row] & 7) == _SLOT_AXIS and rows.d[row] < 0.0:
                axis_exit = True
        if not axis_exit:
            planes, claim = _axis_recovery(data, window, planes, witness.q, witness.feature)
            if claim.kind != 0:
                n = rows.count
                if n < _CANDIDATES:
                    for j in range(3):
                        rows.q[n, j] = claim.q[j]
                        rows.k[n, j] = claim.k[j]
                        rows.n[n, j] = claim.n[j]
                    rows.d[n] = claim.d
                    rows.feature[n] = claim.feature
                    rows.count = n + 1
                else:
                    rows.dropped = rows.dropped + 1
    order, kept, flags = terrain_finish(rows, eps, _ROWS)
    slot = slot_begin + i * _ROWS
    for kept_row in range(kept):
        row = order[kept_row]
        terrain_store_row(
            out,
            slot + kept_row,
            wp.vec3(rows.q[row, 0], rows.q[row, 1], rows.q[row, 2]),
            wp.vec3(rows.n[row, 0], rows.n[row, 1], rows.n[row, 2]),
            rows.d[row],
            rows.feature[row],
            window,
            frame,
        )
    out.count[pair] = kept
    out.state[pair] = _wide(cells) | terrain_finish_state(flags)
    terrain_report(out.status, out.overflow, flags, True)

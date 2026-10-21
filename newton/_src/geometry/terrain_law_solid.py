# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law's per-pair walk for solid cores: boxes, cylinders and cones, convex hulls and ellipsoids.

A solid core has no rounding radius; its reach ``D`` is both margins and both gaps (the contact writer's admission
limit). Per pair the walk visits every surface feature of the window once (window vertex ``(r, c)`` owns its cell's
two faces and diagonal, its x- and y-edges and itself) and offers the rows of the law's rules for solids:

* F, faces: the analytic plane manifold of the core against the triangle's cluster plane
  (:func:`~newton._src.geometry.terrain_law_cores.core_manifold`); a manifold point is supported when its footprint
  along the normal lies in a triangle of the cluster and it is front-facing, and the triangle holding the first
  supported point emits every supported point (one emission per plane);
* E, convex creases: ``theta* = argmax sigma(theta)`` over the crease's normal arc, the float64 law's search: nine
  equally spaced normals, then a bisection on the sign of the derivative between the best one's neighbours
  (:func:`~newton._src.geometry.terrain_law_cores.core_slope`, closed form for boxes and cylinders, the support
  function for hulls; 22 steps); inside the arc, the core's flat support feature within the band
  ``beta = max(0.1 mm, 1e-3 size)``, at an arc end its reference feature (G1); the interval where it meets the
  crease's line emits its ends lifted onto the core (no deeper than the feature's chord over them), ends on vertex
  features left to the vertex;
* V, vertex features: the direction to the core's nearest point (or the minimum translation from inside) inside the
  vertex's normal cone, within reach;
* R, recovery (G3): a pair with no row whose core lies below the surface gets up to four of the core's deepest
  boundary points against the face vertically above them.

Rule C (concave anticipation) is off. Cheapest rejects first: the tile-max exit, a feature skipped when its highest
vertex lies below the core's lowest point minus the reach, a face's plane rejected by one support call before its
manifold, an edge rejected by the core's bounding sphere (a row's line along its normal passes through the sphere; out
of reach only where the edge rises nowhere above the core's lowest point: a row deeper than the reach can stand far
from the core) and by its arc ends' support values before the search, a vertex by the bounding sphere and one support
call before its feature test and nearest point.

The kernels follow the per-pair output protocol of :mod:`~newton._src.geometry.terrain_law`: a pair with no row is
recorded by :func:`~newton._src.geometry.terrain_law.terrain_pair_exit`, and rows are held in the pair's own slots
(window frame) as the finish merges them (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`): a row
coinciding with a held one is dropped, the first in window order staying. The finish runs in place on the held rows
(:func:`~newton._src.geometry.terrain_law.terrain_finish_slots`, a row's core point being ``q + d n``), and the kept
rows are rewritten in the world frame at the front of the slots. A pair with more distinct candidates than its slots
is left to its class's full kernel (:func:`~newton._src.geometry.terrain_law.terrain_defer_pair`): the shadow's fixed
full-finish grid (:data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_FULL_BLOCKS` blocks) strides over the class's
deferred list, and each block's first lane walks its pairs again, one at a time, into the block's scratch list of
:data:`~newton._src.geometry.terrain_law.TERRAIN_ROW_CANDIDATES` rows
(:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`). Every deferred pair is thus finished from its own
candidates, whatever the number of deferred pairs, the order of the deferral or the block that finishes it; a pair
with more distinct candidates than the scratch list keeps the first in window order and is flagged (a candidate
overflow, as the point class's finish flags it). The kernels themselves (one module per partner class) build the
class's core and call the generic bodies here; every module that compiles them disables floating-point contraction,
so the CPU decides every boundary as the GPU does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .broad_phase_common import is_shape_pair_immovable_filtered
from .terrain_law import (
    TERRAIN_EDGE_CONCAVE,
    TERRAIN_EPS32,
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_FEATURE_VERTEX,
    TERRAIN_FLAT_EPS,
    TERRAIN_NORMAL_SAME,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_UNSUPPORTED,
    TERRAIN_PAIR_WIDE,
    TERRAIN_ROW_CANDIDATES,
    TERRAIN_VERTEX_SPAN,
    TERRAIN_WINDOW_WIDE,
    TerrainFeatureData,
    TerrainFinishScratch,
    TerrainLawRows,
    TerrainPairInputs,
    TerrainPairWork,
    TerrainPlanes,
    TerrainWindow,
    terrain_box_cells,
    terrain_cluster,
    terrain_crease_class,
    terrain_defer_pair,
    terrain_edge_kind,
    terrain_edge_sides,
    terrain_eps,
    terrain_feature_id,
    terrain_features_current,
    terrain_finish_slots,
    terrain_finish_state,
    terrain_height,
    terrain_hold_row,
    terrain_incident_triangle,
    terrain_interior_vertex,
    terrain_locate,
    terrain_pair_exit,
    terrain_plane,
    terrain_planes,
    terrain_point_below_cached,
    terrain_report,
    terrain_store_row,
    terrain_tile_exit,
    terrain_triangle_vertices,
    terrain_vertex_in_cone,
    terrain_window,
    terrain_window_extent,
    terrain_window_point,
    terrain_window_vertex,
)
from .terrain_law_cores import (
    TerrainCrossing,
    TerrainHullTable,
    core_bounds,
    core_flat_crossing,
    core_lift,
    core_manifold,
    core_nearest,
    core_recovery_count,
    core_recovery_point,
    core_reference_crossing,
    core_sigma,
    core_slope,
    core_sphere,
)

TERRAIN_SOLID_ARC_SAMPLES = 9
"""Equally spaced normals of rule E's coarse ``theta*`` scan, the arc's ends included (the float64 law's)."""
TERRAIN_SOLID_BISECTION = 22
"""Bisection steps of rule E's ``theta*`` search between the best sample's neighbours (on the derivative's sign):
an eighth of the arc and 22 halvings resolve ``theta*`` to ``2^-24`` of the arc."""
TERRAIN_SOLID_ARC_END = 1.0e-5
"""A ``theta*`` within this of an arc's end [rad] is at the end (DESIGN 1.4; the float64 law's): far above the float32
rounding of ``theta*`` (about 1e-7 rad), so that a body resting flat on a crease's face takes the end in float32 and
float64 alike."""
TERRAIN_SOLID_MASK_WORDS = 8
"""Words of the tiled kernels' productive-vertex mask: windows of up to 256 vertices (larger: the first lane walks)."""
TERRAIN_SOLID_THREAD_VERTICES = 64
"""Window vertices up to which one thread walks a solid pair (its window as the pose gives it, at run time); a larger
window of a tiled-lane pair goes to the class's tiled kernel (queued in its
:class:`~newton._src.geometry.terrain_law.TerrainPairWork`). The pair table's lane classes bound a partner's window
over every orientation, and most solids of a tiled lane walk far smaller windows (a lying cylinder: about 24
vertices), which a warp of one-thread pairs walks at full width while a tile of lanes per pair replays its marks
serially."""
TERRAIN_HULL_FACE_TOL = 1.0e-6
"""A hull's points within this of a face's plane lie on that face [m] (hulls under 1 m; their local frame): a planar
face of a float32 mesh stays one face (its points lie within about 1e-8 m of its plane)."""

_BISECTION = wp.constant(TERRAIN_SOLID_BISECTION)
_ArcNormals = wp.types.matrix(shape=(TERRAIN_SOLID_ARC_SAMPLES, 3), dtype=wp.float32)
_ARC_END = wp.constant(TERRAIN_SOLID_ARC_END)
_CANDIDATES = wp.constant(TERRAIN_ROW_CANDIDATES)
_WIDE = wp.constant(TERRAIN_WINDOW_WIDE)
_EPS32 = wp.constant(TERRAIN_EPS32)
_FLAT = wp.constant(TERRAIN_FLAT_EPS)
_HUGE = wp.constant(3.0e38)
_MASK_WORDS = wp.constant(TERRAIN_SOLID_MASK_WORDS)
_ITEMS = wp.constant(6)
_NORMAL_SAME = wp.constant(TERRAIN_NORMAL_SAME)
_VERTEX_SPAN = wp.constant(TERRAIN_VERTEX_SPAN)

_Mask = wp.types.vector(length=TERRAIN_SOLID_MASK_WORDS, dtype=wp.int32)
_RingNormals = wp.types.matrix(shape=(6, 3), dtype=wp.float32)


@wp.struct
class TerrainSolidData:
    """What the solid classes' kernels read besides the pair inputs, the terrain and their launch argument (the work
    list or the full finish's scratch): hull tables and the one-thread kernels' lane choice."""

    hulls: TerrainHullTable
    """The hulls of the hull class (empty for the other classes)."""
    thread_vertices: int
    """Window vertices up to which the one-thread kernels walk a pair; 0: they walk every window."""


@wp.struct
class _Sink:
    """A pair's held rows while its window is walked."""

    slot: int
    """First slot of the pair's store."""
    capacity: int
    """Slots of the store."""
    held: int
    """Distinct rows offered (held while within the capacity)."""
    raw: int
    """Rows offered, coincident ones included."""
    hits: int
    """Rows offered while probing (the tiled kernels' marking pass)."""
    probe: int
    """1: count rows, hold none."""
    eps: float
    """The window's closed-predicate band [m]."""


@wp.struct
class _Pair:
    """A pair's window-frame quantities."""

    reach: float
    eps: float
    band: float
    """Flatness band of support features [m]: ``max(0.1 mm, 1e-3 size)``."""
    tol: float
    """Crossing and lift tolerance [m]: ``1e-7 + 1e-6 size`` (the float64 law's), at least the band ``eps``."""
    straddle: float
    """How far a reference feature must reach past a crease's line to straddle it [m]."""
    low: float
    """The core's lowest point minus the reach, lowered by its rounding and the cluster tolerance [m]: a feature
    whose highest vertex lies below it gives no row."""
    sphere_c: wp.vec3
    sphere_r: float


@wp.struct
class _Emitters:
    """A window's cache of which triangle emits a cluster's plane manifold (clusters of several triangles).

    Every triangle of a cluster computes the same manifold and support tests, which name one emitting triangle, so a
    triangle of a cached cluster that is not its emitter yields nothing without recomputing them. Four entries, named
    fields (registers), replaced round robin.
    """

    cluster0: int
    cluster1: int
    cluster2: int
    cluster3: int
    emitter0: wp.vec3i
    emitter1: wp.vec3i
    emitter2: wp.vec3i
    emitter3: wp.vec3i
    next: int


@wp.func
def _emitters() -> _Emitters:
    cache = _Emitters()
    cache.cluster0 = -1
    cache.cluster1 = -1
    cache.cluster2 = -1
    cache.cluster3 = -1
    return cache


@wp.func
def _emitter_lookup(cache: _Emitters, cluster: int) -> tuple[bool, wp.vec3i]:
    if cache.cluster0 == cluster:
        return True, cache.emitter0
    if cache.cluster1 == cluster:
        return True, cache.emitter1
    if cache.cluster2 == cluster:
        return True, cache.emitter2
    if cache.cluster3 == cluster:
        return True, cache.emitter3
    return False, wp.vec3i(-1, -1, -1)


@wp.func
def _emitter_store(cache: _Emitters, cluster: int, emitter: wp.vec3i) -> _Emitters:
    if cache.next == 0:
        cache.cluster0 = cluster
        cache.emitter0 = emitter
    elif cache.next == 1:
        cache.cluster1 = cluster
        cache.emitter1 = emitter
    elif cache.next == 2:
        cache.cluster2 = cluster
        cache.emitter2 = emitter
    else:
        cache.cluster3 = cluster
        cache.emitter3 = emitter
    cache.next = (cache.next + 1) % 4
    return cache


# ----------------------------------------------------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _offer(store: TerrainLawRows, sink: _Sink, q: wp.vec3, n: wp.vec3, d: float, feature: int) -> _Sink:
    """Offer a row: count it while probing; else hold it unless it coincides with a held row, as the finish merges
    (:func:`~newton._src.geometry.terrain_law.terrain_hold_row` on the sink's slots of ``store``).

    The capacity bounds the distinct rows, not the offers: a crease's interval ends shared by the next crease merge
    before they count (a pair keeps every distinct candidate its slots or its scratch list hold).
    """
    if sink.probe != 0:
        sink.hits = sink.hits + 1
        return sink
    sink.raw = sink.raw + 1
    sink.held = terrain_hold_row(store, sink.slot, sink.held, q, n, d, feature, sink.eps, sink.capacity)
    return sink


# ----------------------------------------------------------------------------------------------------------------
# the rules
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _vertex_z(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int) -> float:
    return terrain_height(data, window.surface, r, c) - window.z_ref


@wp.func
def _solid_face(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    emitters: _Emitters,
    pair: _Pair,
    sink: _Sink,
    store: TerrainLawRows,
    r: int,
    c: int,
    sub: int,
) -> tuple[TerrainPlanes, _Emitters, _Sink]:
    """Rule F for triangle ``sub`` of cell ``(r, c)``: emit the cluster plane's supported manifold points when this
    triangle holds the first one.

    A cluster's manifold, support tests and emitting triangle are the same for every one of its triangles: the first
    of them evaluated records the emitter (:class:`_Emitters`), and the others pass by.
    """
    surface = window.surface
    cluster = terrain_cluster(data, surface, r, c, sub)
    cached = bool(False)
    if cluster >= 0:
        cached, emitter = _emitter_lookup(emitters, cluster)
        if cached and (emitter[0] != r or emitter[1] != c or emitter[2] != sub):
            return planes, emitters, sink
    va, vb, vd = terrain_triangle_vertices(r, c, sub)
    top = wp.max(
        _vertex_z(data, window, va[0], va[1]),
        wp.max(_vertex_z(data, window, vb[0], vb[1]), _vertex_z(data, window, vd[0], vd[1])),
    )
    if top < pair.low:
        # Never the emitter: a supported point's terrain point lies on the cluster plane over this triangle, at most
        # TERRAIN_FLAT_EPS above its highest vertex, and within the reach below the core's lowest point.
        return planes, emitters, sink
    planes, n, a = terrain_plane(data, window, planes, r, c, sub)
    if core_sigma(core, n, a) > pair.reach:
        if cluster >= 0 and not cached:
            emitters = _emitter_store(emitters, cluster, wp.vec3i(-1, -1, -1))
        return planes, emitters, sink
    count, points, depth = core_manifold(core, n, a, pair.reach)
    supported = int(0)
    first = wp.vec3i(-1, -1, -1)
    for j in range(count):
        k = wp.vec3(points[j, 0], points[j, 1], points[j, 2])
        h = depth[j]
        q = k - h * n
        t = terrain_locate(window, q[0], q[1])
        if t[0] < 0:
            continue
        if terrain_cluster(data, surface, t[0], t[1], t[2]) != cluster:
            continue
        if wp.abs(h) > pair.eps:
            # front-facing: a point above the plane outside the solid, below it inside
            planes, below = terrain_point_below_cached(data, window, planes, k)
            if (h < 0.0) != below:
                continue
        supported = supported | (1 << j)
        if first[0] < 0:
            first = t
    if cluster >= 0 and not cached:
        emitters = _emitter_store(emitters, cluster, first)
    if supported == 0 or first[0] != r or first[1] != c or first[2] != sub:
        return planes, emitters, sink
    for j in range(count):
        if ((supported >> j) & 1) != 0:
            k = wp.vec3(points[j, 0], points[j, 1], points[j, 2])
            sink = _offer(
                store, sink, k - depth[j] * n, n, depth[j], terrain_feature_id(r, c, TERRAIN_FEATURE_FACE, sub, j)
            )
    return planes, emitters, sink


@wp.func
def _arc_sample(samples: _ArcNormals, i: int) -> wp.vec3:
    return wp.vec3(samples[i, 0], samples[i, 1], samples[i, 2])


@wp.func
def _arc_set(samples: _ArcNormals, i: int, n: wp.vec3) -> _ArcNormals:
    samples[i, 0] = n[0]
    samples[i, 1] = n[1]
    samples[i, 2] = n[2]
    return samples


@wp.func
def _arc_samples(n_a: wp.vec3, n_b: wp.vec3) -> _ArcNormals:
    """The arc's normals at eighths of its angle, by halving: the normalized sum of two unit normals bisects their
    angle (no trigonometry, which CUDA evaluates with fused operations)."""
    samples = _ArcNormals()
    samples = _arc_set(samples, 0, n_a)
    samples = _arc_set(samples, 8, n_b)
    for level in range(3):
        half = 4 >> level
        for i in range(half, 8, 2 * half):
            samples = _arc_set(
                samples, i, wp.normalize(_arc_sample(samples, i - half) + _arc_sample(samples, i + half))
            )
    return samples


@wp.func
def _theta_star(
    core: Any, n_a: wp.vec3, n_b: wp.vec3, v0: wp.vec3, sigma_a: float, sigma_b: float, tie: float
) -> tuple[wp.vec3, float, int]:
    """``argmax sigma(n)`` over the crease's normal arc from ``n_a`` to ``n_b`` (less than a half turn): ``(n*,
    sigma(n*), end)`` with ``end`` 1 or 2 at the arc's start or end, 0 inside.

    The float64 law's search, for the same answer where ``sigma`` has several maxima (a core through the crease's
    sides, ``sigma < 0``, can turn ``sigma`` convex): a scan of nine equally spaced normals, then a bisection on the
    sign of the derivative between the best one's neighbours (22 steps, the same count for every lane; the lowest
    support point decides, so a kink is found where two support points cross), then the arc's ends and the
    best sample where they are higher. Within ``tie`` of the refined maximum (float32 cannot order such values: a
    maximum within about 1e-3 rad of an end), an end wins where ``sigma`` does not rise from it into the arc.
    """
    samples = _arc_samples(n_a, n_b)
    best = int(0)
    best_value = sigma_a
    for i in range(1, 9):
        value = sigma_b
        if i < 8:
            value = core_sigma(core, _arc_sample(samples, i), v0)
        if value > best_value:
            best = i
            best_value = value
    lo = _arc_sample(samples, wp.max(best - 1, 0))
    hi = _arc_sample(samples, wp.min(best + 1, 8))
    # the arc turns about this axis from n_a toward n_b: its tangent at n is axis x n
    axis = wp.normalize(wp.cross(n_a, n_b))
    for _step in range(_BISECTION):
        mid = wp.normalize(lo + hi)
        if core_slope(core, mid, wp.cross(axis, mid), v0, 1.0) > 0.0:
            lo = mid
        else:
            hi = mid
    nst = wp.normalize(lo + hi)
    value = core_sigma(core, nst, v0)
    end = int(0)
    if sigma_a > value + tie or (sigma_a >= value - tie and core_slope(core, n_a, wp.cross(axis, n_a), v0, 1.0) <= 0.0):
        nst = n_a
        value = sigma_a
        end = 1
    if sigma_b > value + tie or (
        end == 0 and sigma_b >= value - tie and core_slope(core, n_b, wp.cross(axis, n_b), v0, -1.0) >= 0.0
    ):
        nst = n_b
        value = sigma_b
        end = 2
    if best_value > value + tie:
        nst = _arc_sample(samples, best)
        value = best_value
        end = 0
        if best == 0:
            end = 1
        elif best == 8:
            end = 2
    return nst, value, end


@wp.func
def _solid_edge(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: TerrainLawRows,
    r: int,
    c: int,
    which: int,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule E for edge ``which`` of vertex ``(r, c)``: a convex crease's rows where the core's flat support feature
    (interior ``theta*``) or reference feature (arc end) crosses the crease's line."""
    surface = window.surface
    if terrain_edge_kind(data, surface, r, c, which) >= 0:
        # flat inside one cluster, or on the border: no crease
        return planes, sink
    v0i, v1i, ta, tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    top = wp.max(v0[2], v1[2])
    if top < pair.low:
        return planes, sink
    e = v1 - v0
    length2 = wp.dot(e, e)
    length = wp.sqrt(length2)
    dh = e / length
    toward = pair.sphere_c - v0
    # A row's line along its normal meets the core, so passes within the core's sphere: the sphere's centre lies
    # within its radius of the crease's segment along the crease, and across the line for some normal of the arc.
    along = wp.dot(toward, dh)
    if along < -pair.sphere_r - pair.eps or along > length + pair.sphere_r + pair.eps:
        return planes, sink
    if top <= pair.low + pair.reach:
        # Nowhere above the core's lowest point, the crease holds no row deeper than the reach (whose terrain point
        # lies above the core along its normal), so one out of reach holds none.
        closest = wp.clamp(wp.dot(toward, e) / length2, 0.0, 1.0)
        if wp.length(v0 + closest * e - pair.sphere_c) - pair.sphere_r > pair.reach:
            return planes, sink
    planes, na, pa = terrain_plane(data, window, planes, ta[0], ta[1], ta[2])
    if terrain_crease_class(data, window, r, c, which, na, pa) == TERRAIN_EDGE_CONCAVE:
        return planes, sink
    planes, nb, _pb = terrain_plane(data, window, planes, tb[0], tb[1], tb[2])
    # the normal arc, exactly perpendicular to the edge
    n_a = wp.normalize(na - wp.dot(na, dh) * dh)
    n_b = wp.normalize(nb - wp.dot(nb, dh) * dh)
    if wp.length(n_b - n_a) < 1.0e-9:
        return planes, sink
    # The centre's offset across the line at normal n is dh . (n x (c - v0)), a sinusoid along the arc (less than a
    # half turn): beyond the radius at both ends with one sign, it is beyond it on the whole arc.
    across_a = wp.dot(dh, wp.cross(n_a, toward))
    across_b = wp.dot(dh, wp.cross(n_b, toward))
    if across_a * across_b > 0.0 and wp.min(wp.abs(across_a), wp.abs(across_b)) > pair.sphere_r + pair.eps:
        return planes, sink
    # the core lies in its sphere: sigma(n) >= n . (c - v0) - radius (a bound cheaper than sigma)
    if wp.max(wp.dot(n_a, toward), wp.dot(n_b, toward)) - pair.sphere_r > pair.reach:
        return planes, sink
    sigma_a = core_sigma(core, n_a, v0)
    sigma_b = core_sigma(core, n_b, v0)
    if wp.max(sigma_a, sigma_b) > pair.reach:
        return planes, sink
    nst, sigma, end = _theta_star(core, n_a, n_b, v0, sigma_a, sigma_b, pair.eps)
    if sigma > pair.reach:
        return planes, sink
    # within the arc-end tolerance of an end, the end's reference feature (G1)
    interior = end == 0 and wp.length(nst - n_a) > _ARC_END and wp.length(nst - n_b) > _ARC_END
    e2 = wp.cross(nst, dh)
    found = bool(False)
    crossing = TerrainCrossing()
    if interior:
        found, crossing = core_flat_crossing(core, nst, pair.band, v0, dh, e2, pair.tol)
    else:
        found, crossing = core_reference_crossing(core, nst, v0, dh, e2, pair.tol, pair.straddle)
    if not found:
        return planes, sink
    x0 = wp.max(crossing.lo, 0.0)
    x1 = wp.min(crossing.hi, length)
    if x1 < x0:
        return planes, sink
    ends = 2
    if x1 - x0 < 1.0e-9:
        x0 = 0.5 * (x0 + x1)
        ends = 1
    span = crossing.hi - crossing.lo
    for slot in range(ends):
        x = x0
        if slot == 1:
            x = x1
        # an interval end on a vertex feature belongs to that vertex
        if x <= 0.0:
            planes, feature = _vertex_feature(data, window, planes, v0i[0], v0i[1])
            if feature:
                continue
        if x >= length:
            planes, feature = _vertex_feature(data, window, planes, v1i[0], v1i[1])
            if feature:
                continue
        u = v0 + x * dh
        # The feature's chord over the interval lies in the core on the line, so the lift is no deeper than its
        # point over u, and at an end it is that point: the feature's boundary, where the line can only graze
        # the core's sides and rounding can carry the lift up a side or past the core. A lift beyond the chord
        # by more than the tolerance, or none, is that rounding: the chord's point stands.
        k = crossing.k_lo
        if x == crossing.hi:
            k = crossing.k_hi
        elif x != crossing.lo:
            k = 0.5 * (crossing.k_lo + crossing.k_hi)
            if span > 0.0:
                k = crossing.k_lo + ((x - crossing.lo) / span) * (crossing.k_hi - crossing.k_lo)
        t = wp.dot(nst, k - u)
        hit, _k, lift = core_lift(core, u, nst, pair.tol)
        if hit and lift <= t + pair.tol:
            t = lift
        if t > pair.reach:
            continue
        sink = _offer(store, sink, u, nst, t, terrain_feature_id(r, c, TERRAIN_FEATURE_EDGE, which, slot))
    return planes, sink


@wp.func
def _solid_vertex(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: TerrainLawRows,
    r: int,
    c: int,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule V for vertex ``(r, c)``: a vertex feature whose normal cone holds the direction to the core's nearest
    point (or its minimum translation) emits a row within reach."""
    if not terrain_interior_vertex(window.surface, r, c):
        return planes, sink
    v = terrain_window_vertex(data, window, r, c)
    if v[2] < pair.low:
        return planes, sink
    toward = pair.sphere_c - v
    distance = wp.length(toward)
    if distance - pair.sphere_r > pair.reach:
        return planes, sink
    if distance > 0.0 and core_sigma(core, toward / distance, v) > pair.reach:
        # the core's support plane facing the vertex lies beyond the reach, and so does the core
        return planes, sink
    planes, feature = _vertex_feature(data, window, planes, r, c)
    if not feature:
        return planes, sink
    ok, nst, _k, d = core_nearest(core, v)
    if not ok or d > pair.reach:
        return planes, sink
    if not terrain_vertex_in_cone(data, window, r, c, nst):
        return planes, sink
    sink = _offer(store, sink, v, nst, d, terrain_feature_id(r, c, TERRAIN_FEATURE_VERTEX, 0, 0))
    return planes, sink


@wp.func
def _ring_normal(normals: _RingNormals, i: int) -> wp.vec3:
    return wp.vec3(normals[i, 0], normals[i, 1], normals[i, 2])


@wp.func
def _vertex_feature(
    data: TerrainFeatureData, window: TerrainWindow, planes: TerrainPlanes, r: int, c: int
) -> tuple[TerrainPlanes, bool]:
    """:func:`~newton._src.geometry.terrain_law.terrain_vertex_feature` through the window's plane cache (the same
    planes, bit for bit)."""
    if not terrain_interior_vertex(window.surface, r, c):
        return planes, False
    normals = _RingNormals()
    count = int(0)
    for j in range(6):
        t = terrain_incident_triangle(r, c, j)
        planes, n, _a = terrain_plane(data, window, planes, t[0], t[1], t[2])
        fresh = bool(True)
        for i in range(count):
            if wp.length(n - _ring_normal(normals, i)) <= _NORMAL_SAME:
                fresh = False
        if fresh:
            normals[count, 0] = n[0]
            normals[count, 1] = n[1]
            normals[count, 2] = n[2]
            count += 1
    for i in range(count):
        ni = _ring_normal(normals, i)
        for j in range(i + 1, count):
            dj = _ring_normal(normals, j) - ni
            for k in range(j + 1, count):
                if wp.abs(wp.dot(ni, wp.cross(dj, _ring_normal(normals, k) - ni))) > _VERTEX_SPAN:
                    return planes, True
    return planes, False


@wp.func
def solid_item(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    emitters: _Emitters,
    pair: _Pair,
    sink: _Sink,
    store: TerrainLawRows,
    r: int,
    c: int,
    r1: int,
    c1: int,
    item: int,
) -> tuple[TerrainPlanes, _Emitters, _Sink]:
    """Evaluate item ``item`` of window vertex ``(r, c)`` in a window of cells ``[r0, r1] x [c0, c1]``: 0 and 1 the
    cell's faces, 2 the x-edge, 3 the y-edge, 4 the diagonal, 5 the vertex (every feature of the window once)."""
    inside = r <= r1 and c <= c1
    if item < 2:
        if not inside:
            return planes, emitters, sink
        return _solid_face(core, data, window, planes, emitters, pair, sink, store, r, c, item)
    if item == 5:
        planes, sink = _solid_vertex(core, data, window, planes, pair, sink, store, r, c)
        return planes, emitters, sink
    which = item - 2
    if (which == 0 and c > c1) or (which == 1 and r > r1) or (which == 2 and not inside):
        return planes, emitters, sink
    planes, sink = _solid_edge(core, data, window, planes, pair, sink, store, r, c, which)
    return planes, emitters, sink


@wp.func
def solid_recovery(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    sink: _Sink,
    store: TerrainLawRows,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule R (G3): up to four of the core's boundary points below the surface, deepest first (then in order), each
    against the face vertically above it with that plane's normal and depth.

    A row's feature id takes the pick's rank (0-3) as its point slot, unique within the pair: a boundary point's index
    (up to 32 for a cylinder) would wrap in the slot's three bits.
    """
    count = core_recovery_count(core)
    previous_d = float(-_HUGE)
    previous = int(-1)
    for pick in range(4):
        best = int(-1)
        best_d = float(0.0)
        for i in range(count):
            x = core_recovery_point(core, i)
            t = terrain_locate(window, x[0], x[1])
            if t[0] < 0:
                continue
            planes, n, a = terrain_plane(data, window, planes, t[0], t[1], t[2])
            d = wp.dot(n, x - a)
            if d >= 0.0:
                continue
            # strictly after the previous pick in (depth, index) order
            if d < previous_d or (d == previous_d and i <= previous):
                continue
            if best < 0 or d < best_d:
                best = i
                best_d = d
        if best < 0:
            break
        x = core_recovery_point(core, best)
        t = terrain_locate(window, x[0], x[1])
        planes, n, a = terrain_plane(data, window, planes, t[0], t[1], t[2])
        sink = _offer(
            store, sink, x - best_d * n, n, best_d, terrain_feature_id(t[0], t[1], TERRAIN_FEATURE_RECOVERY, t[2], pick)
        )
        previous_d = best_d
        previous = best
    return planes, sink


# ----------------------------------------------------------------------------------------------------------------
# per pair
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def solid_pair_read(inputs: TerrainPairInputs, data: TerrainFeatureData, pair: int):
    """Read a pair: ``(state, frame, heightfield, local, partner, size, reach)``.

    ``state`` is nonzero for a pair with no row (stale features, the immovable-pair policy); ``local`` is the
    partner's transform in the heightfield frame, ``size`` its scale and margin, ``reach`` both margins and gaps.
    """
    shapes = inputs.pairs[pair]
    hfield = shapes[0]
    partner = shapes[1]
    heightfield = int(inputs.shape_heightfield_index[hfield])
    frame = inputs.shape_transform[hfield]
    local = wp.transform_identity()
    size = wp.vec4(0.0)
    reach = float(0.0)
    if not terrain_features_current(data, heightfield):
        return TERRAIN_PAIR_STALE, frame, heightfield, local, partner, size, reach
    if is_shape_pair_immovable_filtered(
        hfield, partner, inputs.shape_body, inputs.body_flags, inputs.include_static_kinematic_pairs != 0
    ):
        return TERRAIN_PAIR_FILTERED, frame, heightfield, local, partner, size, reach
    local = wp.transform_multiply(wp.transform_inverse(frame), inputs.shape_transform[partner])
    size = inputs.shape_data[partner]
    reach = inputs.shape_data[hfield][3] + size[3] + inputs.shape_gap[hfield] + inputs.shape_gap[partner]
    return 0, frame, heightfield, local, partner, size, reach


@wp.func
def solid_pair_window(core: Any, data: TerrainFeatureData, heightfield: int, reach: float):
    """The pair's window: ``(state, cells, window, core, pair)`` with the core moved into the window frame.

    ``state`` is nonzero when the window misses the heightfield or the tile-max exit applies.
    """
    surface = data.surfaces[heightfield]
    lo, hi = core_bounds(core)
    cells = terrain_box_cells(surface, lo[0] - reach, lo[1] - reach, hi[0] + reach, hi[1] + reach)
    window = TerrainWindow()
    pair = _Pair()
    if cells[1] < cells[0] or cells[3] < cells[2]:
        return TERRAIN_PAIR_OUTSIDE, cells, window, core, pair
    # The lowest point minus the reach, lowered by its own float32 rounding and by the cluster tolerance (a face row's
    # terrain point lies on its cluster plane, up to TERRAIN_FLAT_EPS above its triangle's highest vertex): the exit
    # drops no row but one whose terrain point rounding carries off its triangle on a steep face, and then only within
    # that of the reach limit.
    low = lo[2] - reach
    low = low - 4.0 * _EPS32 * (wp.abs(lo[2]) + reach) - _FLAT
    if terrain_tile_exit(data, surface, cells[0], cells[1], cells[2], cells[3], low):
        return TERRAIN_PAIR_TILE_EXIT, cells, window, core, pair
    window = terrain_window(data, heightfield, cells[0], cells[2])
    centre = core.c
    core.c = terrain_window_point(window, centre)
    shift = core.c - centre
    lo = lo + shift
    hi = hi + shift
    extent = wp.max(
        wp.max(wp.max(wp.abs(lo[0]), wp.abs(lo[1])), wp.abs(lo[2])),
        wp.max(wp.max(wp.abs(hi[0]), wp.abs(hi[1])), wp.abs(hi[2])),
    )
    eps = terrain_eps(terrain_window_extent(data, window, cells[1], cells[3], extent))
    size = wp.length(hi - lo)
    pair.reach = reach
    pair.eps = eps
    pair.band = wp.max(1.0e-4, 1.0e-3 * size)
    pair.tol = wp.max(1.0e-7 + 1.0e-6 * size, eps)
    pair.straddle = eps
    pair.low = lo[2] - reach - 4.0 * _EPS32 * (wp.abs(lo[2]) + reach) - eps - _FLAT
    sphere_c, sphere_r = core_sphere(core)
    pair.sphere_c = sphere_c
    pair.sphere_r = sphere_r
    return 0, cells, window, core, pair


@wp.func
def _wide(cells: wp.vec4i) -> int:
    if cells[1] - cells[0] + 1 >= _WIDE or cells[3] - cells[2] + 1 >= _WIDE:
        return TERRAIN_PAIR_WIDE
    return 0


@wp.func
def solid_sink(slot: int, capacity: int, eps: float, probe: int) -> _Sink:
    sink = _Sink()
    sink.slot = slot
    sink.capacity = capacity
    sink.eps = eps
    sink.probe = probe
    return sink


@wp.func
def solid_scratch_store(scratch: TerrainFinishScratch) -> TerrainLawRows:
    """A full-finish launch's scratch rows as a store of held rows (its row fields only; the in-place finish reads a
    row's core point as ``q + d n``)."""
    store = TerrainLawRows()
    store.q = scratch.q
    store.n = scratch.n
    store.d = scratch.d
    store.feature = scratch.feature
    return store


@wp.func
def solid_walk(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: TerrainLawRows,
    cells: wp.vec4i,
) -> tuple[TerrainPlanes, _Sink]:
    """Walk the window's features in window order (vertex by vertex, its items in order)."""
    r1 = cells[1]
    c1 = cells[3]
    emitters = _emitters()
    for r in range(cells[0], r1 + 2):
        for c in range(cells[2], c1 + 2):
            for item in range(_ITEMS):
                planes, emitters, sink = solid_item(
                    core, data, window, planes, emitters, pair, sink, store, r, c, r1, c1, item
                )
    return planes, sink


@wp.func
def solid_close(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: TerrainLawRows,
    out: TerrainLawRows,
    pair_index: int,
    out_slot: int,
    row_limit: int,
    partner_class: int,
    frame: wp.transform,
    cells: wp.vec4i,
    deferrable: int,
):
    """Finish a walked pair: defer it when its distinct rows overflow its slots, else add the recovery rows when it
    has none, run the finish in place and write the kept rows (world frame) to the pair's output slots."""
    dropped = int(0)
    if sink.held > sink.capacity:
        if deferrable != 0:
            terrain_defer_pair(out, partner_class, pair_index, _wide(cells))
            return
        # more distinct candidates than the scratch list: the first in window order, flagged
        sink.held = sink.capacity
        dropped = 1
    if sink.raw == 0:
        planes, sink = solid_recovery(core, data, window, planes, sink, store)
    keep, kept, flags = terrain_finish_slots(
        store.q, store.n, store.d, sink.slot, sink.held, dropped, pair.eps, row_limit
    )
    written = int(0)
    for i in range(sink.held):
        if ((keep >> i) & 1) != 0:
            s = sink.slot + i
            terrain_store_row(
                out, out_slot + written, store.q[s], store.n[s], store.d[s], store.feature[s], window, frame
            )
            written += 1
    out.count[pair_index] = kept
    out.state[pair_index] = _wide(cells) | terrain_finish_state(flags)
    terrain_report(out.status, out.overflow, flags, True)


@wp.func
def solid_rows_thread(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    work: TerrainPairWork,
    solids: TerrainSolidData,
    pair_index: int,
    slot: int,
    row_limit: int,
    partner_class: int,
    frame: wp.transform,
    heightfield: int,
    reach: float,
):
    """One thread's walk of a pair whose core is built (heightfield frame): window, walk, close; a pair from
    ``work.first`` on (the tiled segment) whose window holds more than ``solids.thread_vertices`` vertices is queued in
    ``work`` for the class's tiled kernel instead."""
    state, cells, window, core, pair = solid_pair_window(core, data, heightfield, reach)
    if state != 0:
        terrain_pair_exit(out, pair_index, state)
        return
    if solids.thread_vertices > 0 and pair_index >= work.first:
        if (cells[1] - cells[0] + 2) * (cells[3] - cells[2] + 2) > solids.thread_vertices:
            work.pairs[wp.atomic_add(work.count, 0, 1)] = pair_index
            return
    planes = terrain_planes()
    sink = solid_sink(slot, row_limit, pair.eps, 0)
    planes, sink = solid_walk(core, data, window, planes, pair, sink, out, cells)
    solid_close(
        core, data, window, planes, pair, sink, out, out, pair_index, slot, row_limit, partner_class, frame, cells, 1
    )


@wp.func
def solid_rows_full(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    scratch: TerrainFinishScratch,
    block: int,
    pair_index: int,
    slot: int,
    row_limit: int,
    partner_class: int,
    frame: wp.transform,
    heightfield: int,
    reach: float,
):
    """A deferred pair walked again into block ``block``'s scratch list, finished and written to its output slots."""
    _state, cells, window, core, pair = solid_pair_window(core, data, heightfield, reach)
    planes = terrain_planes()
    sink = solid_sink(block * _CANDIDATES, _CANDIDATES, pair.eps, 0)
    store = solid_scratch_store(scratch)
    planes, sink = solid_walk(core, data, window, planes, pair, sink, store, cells)
    solid_close(
        core, data, window, planes, pair, sink, store, out, pair_index, slot, row_limit, partner_class, frame, cells, 0
    )


@wp.func
def solid_mark(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    pair: _Pair,
    cells: wp.vec4i,
    lane: int,
    lanes: int,
) -> _Mask:
    """The tiled kernels' marking pass: this lane's productive window vertices (windows of up to 256 vertices)."""
    planes = terrain_planes()
    emitters = _emitters()
    store = TerrainLawRows()
    marked = _Mask()
    r0 = cells[0]
    c0 = cells[2]
    width = cells[3] - c0 + 2
    vertices = (cells[1] - r0 + 2) * width
    for v in range(lane, vertices, lanes):
        sink = solid_sink(0, 0, pair.eps, 1)
        for item in range(_ITEMS):
            planes, emitters, sink = solid_item(
                core,
                data,
                window,
                planes,
                emitters,
                pair,
                sink,
                store,
                r0 + v // width,
                c0 + v % width,
                cells[1],
                cells[3],
                item,
            )
        if sink.hits > 0:
            marked[v // 32] = marked[v // 32] | (1 << (v % 32))
    return marked


@wp.func
def solid_replay(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    window: TerrainWindow,
    pair: _Pair,
    cells: wp.vec4i,
    marked: _Mask,
    words: int,
    pair_index: int,
    slot: int,
    row_limit: int,
    partner_class: int,
    frame: wp.transform,
):
    """The tiled kernels' first lane: the marked vertices' items in window order (every vertex when the window is too
    large to mark), then the pair's close: the rows the one-thread walk gives the pair."""
    planes = terrain_planes()
    emitters = _emitters()
    sink = solid_sink(slot, row_limit, pair.eps, 0)
    r0 = cells[0]
    c0 = cells[2]
    width = cells[3] - c0 + 2
    vertices = (cells[1] - r0 + 2) * width
    for v in range(vertices):
        if words <= _MASK_WORDS and (marked[v // 32] & (1 << (v % 32))) == 0:
            continue
        for item in range(_ITEMS):
            planes, emitters, sink = solid_item(
                core,
                data,
                window,
                planes,
                emitters,
                pair,
                sink,
                out,
                r0 + v // width,
                c0 + v % width,
                cells[1],
                cells[3],
                item,
            )
    solid_close(
        core, data, window, planes, pair, sink, out, out, pair_index, slot, row_limit, partner_class, frame, cells, 1
    )


@wp.func
def solid_unsupported(out: TerrainLawRows, pair_index: int):
    """Record a pair whose partner has no core in the law (no row,
    :func:`~newton._src.geometry.terrain_law.terrain_pair_exit`)."""
    terrain_pair_exit(out, pair_index, TERRAIN_PAIR_UNSUPPORTED)


# ----------------------------------------------------------------------------------------------------------------
# host: hull tables and the solid classes' data
# ----------------------------------------------------------------------------------------------------------------


def hull_faces(points: np.ndarray, *, tol: float = TERRAIN_HULL_FACE_TOL) -> tuple[list[list[int]], np.ndarray]:
    """Planar faces of the convex hull of ``points``: vertex index lists (increasing) and outward unit normals.

    A face is a plane through three points with every point on one side (within ``tol`` times the points' extent, at
    least 1 m); its vertices are the points within that of the plane, and planes with the same vertices are one face.
    Points inside the hull belong to no face (they never support it).
    """
    p = np.asarray(points, np.float64).reshape(-1, 3)
    count = len(p)
    scale = max(float(np.abs(p).max()) if count else 0.0, 1.0)
    faces: dict[tuple, np.ndarray] = {}
    for i in range(count):
        for j in range(i + 1, count - 1):
            k = np.arange(j + 1, count)
            normal = np.cross(p[j] - p[i], p[k] - p[i])
            length = np.linalg.norm(normal, axis=1)
            good = length > 1.0e-12 * scale * scale
            if not np.any(good):
                continue
            normal = normal[good] / length[good, None]
            side = (p - p[i]) @ normal.T
            below = np.all(side <= tol * scale, axis=0)
            above = np.all(side >= -tol * scale, axis=0)
            for m in np.flatnonzero(below | above):
                on = side[:, m] if below[m] else -side[:, m]
                key = tuple(int(v) for v in np.flatnonzero(np.abs(on) <= tol * scale))
                if key not in faces:
                    faces[key] = normal[m] if below[m] else -normal[m]
    lists = [list(key) for key in faces]
    normals = np.asarray(list(faces.values()), np.float64).reshape(-1, 3)
    return lists, normals


def _fan(face: list[int], points: np.ndarray, normal: np.ndarray) -> list[tuple[int, int, int]]:
    """Fan triangles of a planar convex face, its vertices ordered around its normal."""
    p = points[face]
    centre = p.mean(axis=0)
    u = p[0] - centre
    if np.linalg.norm(u) < 1.0e-12:
        u = np.cross(normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1.0e-6:
            u = np.cross(normal, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    angle = np.arctan2((p - centre) @ w, (p - centre) @ u)
    loop = [face[i] for i in np.argsort(angle, kind="stable")]
    return [(loop[0], loop[i], loop[i + 1]) for i in range(1, len(loop) - 1)]


def build_hull_table(meshes: list[np.ndarray], shape_hull: np.ndarray, device) -> TerrainHullTable:
    """The hull table of ``meshes`` (each an ``(n, 3)`` array of unscaled local points) and the per-shape map."""
    points, planes, face_vertices, face_begin = [], [], [], [0]
    vertex_faces_per_point: list[list[int]] = []
    triangles, hull_points, hull_faces_range, hull_triangles = [], [], [], []
    for mesh in meshes:
        p = np.asarray(mesh, np.float64).reshape(-1, 3)
        point_base = len(points)
        face_base = len(planes)
        tri_base = len(triangles)
        lists, normals = hull_faces(p)
        incident = [[] for _ in range(len(p))]
        for f, (face, normal) in enumerate(zip(lists, normals, strict=True)):
            planes.append((*normal.tolist(), float(normal @ p[face[0]])))
            face_vertices.extend(point_base + i for i in face)
            face_begin.append(len(face_vertices))
            for i in face:
                incident[i].append(face_base + f)
            triangles.extend(tuple(point_base + i for i in tri) for tri in _fan(face, p, normal))
        points.extend(p.tolist())
        vertex_faces_per_point.extend(incident)
        hull_points.append((point_base, len(p)))
        hull_faces_range.append((face_base, len(lists)))
        hull_triangles.append((tri_base, len(triangles) - tri_base))
    vertex_faces, vertex_face_begin = [], [0]
    for incident in vertex_faces_per_point:
        vertex_faces.extend(sorted(incident))
        vertex_face_begin.append(len(vertex_faces))
    # the point graph: every point's face-sharing points (a support search descends it)
    neighbors, neighbor_begin = [], [0]
    for i, incident in enumerate(vertex_faces_per_point):
        near = {v for f in incident for v in face_vertices[face_begin[f] : face_begin[f + 1]]}
        near.discard(i)
        neighbors.extend(sorted(near))
        neighbor_begin.append(len(neighbors))

    def array(values, dtype, empty):
        values = values if len(values) else empty
        return wp.array(np.asarray(values), dtype=dtype, device=device)

    table = TerrainHullTable()
    table.points = array(np.asarray(points, np.float32).reshape(-1, 3), wp.vec3, np.zeros((1, 3), np.float32))
    table.planes = array(np.asarray(planes, np.float32).reshape(-1, 4), wp.vec4, np.zeros((1, 4), np.float32))
    table.face_vertices = array(np.asarray(face_vertices, np.int32), wp.int32, np.zeros(1, np.int32))
    table.face_begin = array(np.asarray(face_begin, np.int32), wp.int32, np.zeros(1, np.int32))
    table.vertex_faces = array(np.asarray(vertex_faces, np.int32), wp.int32, np.zeros(1, np.int32))
    table.vertex_face_begin = array(np.asarray(vertex_face_begin, np.int32), wp.int32, np.zeros(1, np.int32))
    table.neighbors = array(np.asarray(neighbors, np.int32), wp.int32, np.zeros(1, np.int32))
    table.neighbor_begin = array(np.asarray(neighbor_begin, np.int32), wp.int32, np.zeros(1, np.int32))
    table.triangles = array(np.asarray(triangles, np.int32).reshape(-1, 3), wp.vec3i, np.zeros((1, 3), np.int32))
    table.hull_points = array(np.asarray(hull_points, np.int32).reshape(-1, 2), wp.vec2i, np.zeros((1, 2), np.int32))
    table.hull_faces = array(
        np.asarray(hull_faces_range, np.int32).reshape(-1, 2), wp.vec2i, np.zeros((1, 2), np.int32)
    )
    table.hull_triangles = array(
        np.asarray(hull_triangles, np.int32).reshape(-1, 2), wp.vec2i, np.zeros((1, 2), np.int32)
    )
    table.shape_hull = array(np.asarray(shape_hull, np.int32), wp.int32, np.full(1, -1, np.int32))
    return table


def terrain_solid_unsupported(partner_class: np.ndarray, shape_type: np.ndarray) -> np.ndarray:
    """Whether solid partners have no core in the law yet (their pairs get no row and state
    :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_UNSUPPORTED`): in the hull class, a shape that is neither a
    convex hull nor an ellipsoid (a barrel cylinder).

    Args:
        partner_class: Partner class of every partner
            (:func:`~newton._src.geometry.terrain_contacts.terrain_partner_classes`).
        shape_type: Geometry type of every partner.
    """
    from .terrain_contacts import TERRAIN_PARTNER_HULL  # noqa: PLC0415
    from .types import GeoType  # noqa: PLC0415

    kind = np.asarray(shape_type)
    cored = (kind == int(GeoType.CONVEX_MESH)) | (kind == int(GeoType.ELLIPSOID))
    return (np.asarray(partner_class) == TERRAIN_PARTNER_HULL) & ~cored


def terrain_solid_data(
    model,
    partners: np.ndarray,
    *,
    thread_vertices: int = TERRAIN_SOLID_THREAD_VERTICES,
) -> TerrainSolidData:
    """The solid classes' data for a model: the hull table of the convex-mesh partners in ``partners`` (shape
    indices) and the one-thread kernels' lane choice (``thread_vertices`` 0: they walk every window). Their launch
    arguments, the work lists and the full finish's scratch, are the shadow's
    (:func:`~newton._src.geometry.terrain_law.terrain_pair_work`,
    :func:`~newton._src.geometry.terrain_law.terrain_finish_scratch`)."""
    from .types import GeoType  # noqa: PLC0415

    device = model.device
    types = model.shape_type.numpy() if model.shape_count else np.zeros(0, np.int32)
    shape_hull = np.full(max(model.shape_count, 1), -1, np.int32)
    meshes, index_of = [], {}
    for s in np.unique(np.asarray(partners, np.int64)):
        if int(types[s]) != int(GeoType.CONVEX_MESH):
            continue
        mesh = model.shape_source[int(s)]
        key = id(mesh)
        if key not in index_of:
            index_of[key] = len(meshes)
            meshes.append(np.asarray(mesh.vertices, np.float32).reshape(-1, 3))
        shape_hull[s] = index_of[key]
    solids = TerrainSolidData()
    solids.hulls = build_hull_table(meshes, shape_hull, device)
    solids.thread_vertices = int(thread_vertices)
    return solids

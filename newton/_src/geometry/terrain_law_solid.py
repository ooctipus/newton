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
  boundary points against the face vertically above them;
* X, deepest exit: the exit (rule P's, a sphere centre's: the minimum with the smallest translation, over the cells
  within the point's bound, where its closest surface point lies, inside the window or not) of the deepest of the
  core's candidate boundary points, when it lies deeper than every other row of the pair by more than
  :data:`TERRAIN_SOLID_EXIT_MARGIN` (:func:`solid_exits`). A deepest point whose closest surface point is unique is a
  local maximum of its distance to that one feature: rule F's support for a face, else an extreme point of the core,
  which the candidates hold (a box's corners, a hull's points, a cone's apex; per valley the core's support points at
  the local minima of ``sigma`` over its normal arc; per vertex the local maxima of the distance from it: each rim's
  farthest point, an ellipsoid's farthest point and the near end of an elongated one).
  A deepest point equally far from two or more surface points (a tie) can lie inside a core face, where no candidate
  is: there X takes the deepest candidate's exit, which may fall short of the core's depth. The exit row joins the
  pair's rows before the finish, which treats it as any row: it can merge with a coinciding row, or imply rows of
  its plane group, which the finish then drops.

Rule C (concave anticipation) is off. Cheapest rejects first: the tile-max exit, a feature skipped when its highest
vertex lies below the core's lowest point minus the reach, a face's plane rejected by one support call before its
manifold, an edge rejected by the core's bounding sphere (a row's line along its normal passes through the sphere; out
of reach only where the edge rises nowhere above the core's lowest point: a row deeper than the reach can stand far
from the core) and by its arc ends' support values before the search, a vertex by the bounding sphere and one support
call before its feature test and nearest point.

Rule X runs after the walk (and rule R), only for a core reaching below its window's highest vertex (read in the
pass that sets the window's band): the walk leaves such a pair, its rows held in its walk slots, to its class's exit
kernel (:func:`solid_close`, :func:`solid_rows_exit`), so that rule X's work runs on warps of its own, off the walks'
registers and stacks; a deferred pair's full finish runs it itself. A candidate's bound is its depth below the plane of the face under it when its
projection stays in that face's cluster, else its distance to that triangle, or its distance to the feature that
proposed it, the smallest, lowered by a descent over blocks of 3 x 3 cells toward its closest surface point
(:func:`_near_bound`). A first pass over the candidates takes the two largest bounds above the pair's depth, and the
largest's exit is searched; a second pass searches the exits of the candidates that can still beat it (or tie it from
earlier in window order) only when the second largest bound can; a valley is skipped where the wedge below it cannot
hold such a point. An exit search covers the cells within its candidate's bound, so that its work follows the
candidate's depth, not the window's size: a pair's rule X work is its candidates' (a few per window vertex for
cylinders, cones and ellipsoids, a box's or hull's points) and, on buried solids under rough slopes, one exit search.

The kernels follow the per-pair output protocol of :mod:`~newton._src.geometry.terrain_law`: a pair with no row is
recorded by :func:`~newton._src.geometry.terrain_law.terrain_pair_exit`, and rows are held in the pair's own walk
slots (window frame) as the finish merges them (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`): a row
coinciding with a held one is dropped, the first in window order staying. The finish runs in place on the held rows
(:func:`~newton._src.geometry.terrain_law.terrain_finish_slots`, a row's core point being ``q + d n``), and the kept
rows are rewritten in the world frame at the front of the slots. A pair with more distinct candidates than its slots
is left to its class's full kernel (:func:`~newton._src.geometry.terrain_law.terrain_defer_pair`): the shadow's fixed
full-finish grid strides over the class's deferred list, and each block's first lane walks its pairs again, one at a
time, into the block's scratch (:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`), which holds every
distinct candidate a window of the class can give
(:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`). The finish runs on all of them
(:func:`~newton._src.geometry.terrain_law.terrain_finish_list`) and every row it keeps is written, in the pair's walk
slots when they fit, else in rows of the spill region it reserves
(:func:`~newton._src.geometry.terrain_law.terrain_spill_rows`). Every deferred pair is thus finished from its own candidates, whatever
the number of deferred pairs, the order of the deferral or the block that finishes it. The kernels themselves (one
module per partner class) build the class's core and call the generic bodies here; every module that compiles them
disables floating-point contraction, so the CPU decides every boundary as the GPU does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .broad_phase_common import is_shape_pair_immovable_filtered
from .terrain_law import (
    TERRAIN_CELL_WIDEN,
    TERRAIN_EDGE_CONCAVE,
    TERRAIN_EPS32,
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_FEATURE_VERTEX,
    TERRAIN_FINISH_CANDIDATE_OVERFLOW,
    TERRAIN_FINISH_ROW_OVERFLOW,
    TERRAIN_FLAT_EPS,
    TERRAIN_NORMAL_SAME,
    TERRAIN_PAIR_EXIT,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_UNSUPPORTED,
    TERRAIN_PAIR_WIDE,
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
    terrain_finish_list,
    terrain_finish_overflow,
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
    terrain_spill_rows,
    terrain_store_row,
    terrain_tile_exit,
    terrain_triangle_vertices,
    terrain_vertex_in_cone,
    terrain_window,
    terrain_window_point,
    terrain_window_vertex,
)
from .terrain_law_cores import (
    TerrainCrossing,
    TerrainHullTable,
    closest_on_triangle,
    core_bounds,
    core_exit_count,
    core_exit_point,
    core_far_count,
    core_far_point,
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
    core_support_count,
    core_support_point,
)
from .terrain_law_point import terrain_point_exit

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
TERRAIN_SOLID_EXIT_MARGIN = 1.0e-5
"""Rule X offers a pair's deepest exit only where it lies deeper than every other row of the pair by more than this
[m] (the float64 law's): a pair whose rows reach its depth keeps them as they are."""
TERRAIN_SOLID_EXIT_SLOT = 7
"""Point slot of rule X's row in its feature id: no other solid rule uses it (F 0-3, E 0-1, V 0, R 0-3)."""
TERRAIN_SOLID_EXIT_DESCENT = 4
"""Blocks of 3 x 3 cells rule X's near bound visits at most per candidate, each centred on the cell of the closest
point the last one found (:func:`_near_bound`): a descent toward the candidate's closest surface point, which only
tightens the bound that decides whether its exit is searched."""
TERRAIN_SOLID_EXIT_SLOTS = 64
"""Candidate slots of one window vertex in rule X's candidate keys (window order): a valley's two support points at
each of its nine arc samples, for its three edges (54), then the vertex's local maxima of the distance from it."""

_BISECTION = wp.constant(TERRAIN_SOLID_BISECTION)
_ArcNormals = wp.types.matrix(shape=(TERRAIN_SOLID_ARC_SAMPLES, 3), dtype=wp.float32)
_ARC_END = wp.constant(TERRAIN_SOLID_ARC_END)
_WIDE = wp.constant(TERRAIN_WINDOW_WIDE)
_EPS32 = wp.constant(TERRAIN_EPS32)
_FLAT = wp.constant(TERRAIN_FLAT_EPS)
_HUGE = wp.constant(3.0e38)
_MASK_WORDS = wp.constant(TERRAIN_SOLID_MASK_WORDS)
_ITEMS = wp.constant(6)
_NORMAL_SAME = wp.constant(TERRAIN_NORMAL_SAME)
_VERTEX_SPAN = wp.constant(TERRAIN_VERTEX_SPAN)
_EXIT_MARGIN = wp.constant(TERRAIN_SOLID_EXIT_MARGIN)
_EXIT_SLOT = wp.constant(TERRAIN_SOLID_EXIT_SLOT)
_DESCENT = wp.constant(TERRAIN_SOLID_EXIT_DESCENT)
_EXIT_SLOTS = wp.constant(TERRAIN_SOLID_EXIT_SLOTS)
_CELL_WIDEN = wp.constant(TERRAIN_CELL_WIDEN)

_Mask = wp.types.vector(length=TERRAIN_SOLID_MASK_WORDS, dtype=wp.int32)
_RingNormals = wp.types.matrix(shape=(6, 3), dtype=wp.float32)
_ArcValues = wp.types.vector(length=TERRAIN_SOLID_ARC_SAMPLES, dtype=wp.float32)


@wp.struct
class TerrainSolidData:
    """What the solid classes' kernels read besides the pair inputs, the terrain and their launch argument (the work
    list or the full finish's scratch): hull tables, the one-thread kernels' lane choice and rule X's list."""

    hulls: TerrainHullTable
    """The hulls of the hull class (empty for the other classes)."""
    thread_vertices: int
    """Window vertices up to which the one-thread kernels walk a pair; 0: they walk every window."""
    exits: wp.array[wp.int32]
    """The walked pairs left to their class's exit kernel (rule X; :func:`solid_close`): partner class ``c``'s list
    starts at the side buffer's ``deferred_first[c]`` (its first pair in the table) and holds ``exit_count[c]`` pairs,
    appended in any order (a pair's rows depend on it alone)."""
    exit_count: wp.array[wp.int32]
    """Per partner class: the pairs in its exit list (zeroed before every launch)."""
    exit_work: wp.array[wp.int32]
    """Rule X's exit searches since the launch began (``[0]``) and the window vertices they covered (``[1]``)."""


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
    extent: float
    """The window's extent (:func:`~newton._src.geometry.terrain_law.terrain_window_extent`), whose band ``eps``
    is [m]."""
    top: float
    """The window's highest vertex [m, window frame]: a core whose lowest point lies no lower has no rule X
    candidate."""


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


@wp.struct
class _Exit:
    """Rule X's search (:func:`solid_exits`): the depth an exit must pass, the best exit found, and the two passes over
    the candidates (window order, :func:`_exit_candidates`): the first takes their largest bounds, the second the exits
    of those that can still beat the best one."""

    target: float
    """The pair's deepest row's depth (0 for none deeper than the surface) plus :data:`TERRAIN_SOLID_EXIT_MARGIN`."""
    threshold: float
    """The larger of ``target`` and the best exit's depth: a candidate whose bound is less cannot do better."""
    found: int
    q: wp.vec3
    n: wp.vec3
    d: float
    feature: int
    key: int
    """The best exit's candidate key (window order, :func:`_exit_candidates`): the first wins a tie."""
    walk: int
    """0 in the first pass (bounds), 1 in the second (exits)."""
    first: float
    """The first pass's largest bound above ``target`` (``target`` for none)."""
    first_key: int
    first_x: wp.vec3
    second: float
    """The first pass's largest bound of every other candidate above ``target`` (``target`` for none)."""
    searches: int
    """Exit searches so far."""
    cells: int
    """Window vertices the exit searches covered so far."""


@wp.func
def _exit_bound(
    data: TerrainFeatureData, window: TerrainWindow, planes: TerrainPlanes, x: wp.vec3
) -> tuple[TerrainPlanes, float]:
    """An upper bound of point ``x``'s exit translation, or -1 above the surface (the float64 law's): its depth below
    the cluster plane of the triangle under it when its projection onto that plane stays in the cluster (a face
    minimum), else its distance to that triangle."""
    t = terrain_locate(window, x[0], x[1])
    if t[0] < 0:
        return planes, -1.0
    planes, n, a = terrain_plane(data, window, planes, t[0], t[1], t[2])
    d = wp.dot(n, x - a)
    if d >= 0.0:
        return planes, -1.0
    q = x - d * n
    s = terrain_locate(window, q[0], q[1])
    if s[0] >= 0:
        same = s[0] == t[0] and s[1] == t[1] and s[2] == t[2]
        if not same:
            cluster = terrain_cluster(data, window.surface, t[0], t[1], t[2])
            same = cluster >= 0 and terrain_cluster(data, window.surface, s[0], s[1], s[2]) == cluster
        if same:
            return planes, -d
    va, vb, vd = terrain_triangle_vertices(t[0], t[1], t[2])
    k = closest_on_triangle(
        x,
        terrain_window_vertex(data, window, va[0], va[1]),
        terrain_window_vertex(data, window, vb[0], vb[1]),
        terrain_window_vertex(data, window, vd[0], vd[1]),
    )
    return planes, wp.length(x - k)


@wp.func
def _near_bound(data: TerrainFeatureData, window: TerrainWindow, x: wp.vec3, bound: float, stop: float) -> float:
    """``bound`` lowered to ``x``'s distance to the triangles of the 3 x 3 cells around the one under it, then around
    the cell under ``x``'s projection onto the plane of the block's nearest triangle, while that cell moves and the
    bound exceeds ``stop`` (at most :data:`TERRAIN_SOLID_EXIT_DESCENT` blocks, within the heightfield): a descent
    toward ``x``'s closest surface point, whose distance every block's is at least (on a planar stretch the second
    block holds it). The second stage of rule X's bound, before an exit search; it decides which exits are searched,
    never the exit kept."""
    surface = window.surface
    t = terrain_locate(window, x[0], x[1])
    if t[0] < 0:
        return bound
    r = t[0]
    c = t[1]
    for _step in range(_DESCENT):
        near = wp.vec3i(-1, -1, -1)
        nearest = float(_HUGE)
        for rr in range(wp.max(r - 1, 0), wp.min(r + 1, surface.nrow - 2) + 1):
            for cc in range(wp.max(c - 1, 0), wp.min(c + 1, surface.ncol - 2) + 1):
                for sub in range(2):
                    va, vb, vd = terrain_triangle_vertices(rr, cc, sub)
                    k = closest_on_triangle(
                        x,
                        terrain_window_vertex(data, window, va[0], va[1]),
                        terrain_window_vertex(data, window, vb[0], vb[1]),
                        terrain_window_vertex(data, window, vd[0], vd[1]),
                    )
                    distance = wp.length(x - k)
                    if distance < nearest:
                        nearest = distance
                        near = wp.vec3i(rr, cc, sub)
        bound = wp.min(bound, nearest)
        if bound <= stop:
            break
        va, vb, vd = terrain_triangle_vertices(near[0], near[1], near[2])
        a = terrain_window_vertex(data, window, va[0], va[1])
        n = wp.normalize(
            wp.cross(
                terrain_window_vertex(data, window, vb[0], vb[1]) - a,
                terrain_window_vertex(data, window, vd[0], vd[1]) - a,
            )
        )
        q = x - wp.dot(x - a, n) * n
        s = terrain_locate(window, q[0], q[1])
        if s[0] < 0 or (s[0] == r and s[1] == c):
            break
        r = s[0]
        c = s[1]
    return bound


@wp.func
def _exit_cells(window: TerrainWindow, x: wp.vec3, radius: float) -> wp.vec4i:
    """The cells ``(r0, r1, c0, c1)`` within ``radius`` of window-frame point ``x`` along x and y, clamped to the
    heightfield (cell coordinates widened as :func:`~newton._src.geometry.terrain_law.terrain_box_cells` widens
    them)."""
    surface = window.surface
    c0 = wp.max(window.c0 + int(wp.floor((x[0] - radius) / surface.dx - _CELL_WIDEN)), 0)
    c1 = wp.min(window.c0 + int(wp.floor((x[0] + radius) / surface.dx + _CELL_WIDEN)), surface.ncol - 2)
    r0 = wp.max(window.r0 + int(wp.floor((x[1] - radius) / surface.dy - _CELL_WIDEN)), 0)
    r1 = wp.min(window.r0 + int(wp.floor((x[1] + radius) / surface.dy + _CELL_WIDEN)), surface.nrow - 2)
    return wp.vec4i(r0, r1, c0, c1)


@wp.func
def _exit_eps(data: TerrainFeatureData, window: TerrainWindow, pair: _Pair, cells: wp.vec4i, region: wp.vec4i) -> float:
    """The closed-predicate band of an exit search over ``region``: the pair's own inside its window ``cells``, else
    the band of the window's extent grown by the region's (its size and heights about the window's reference
    vertex)."""
    if region[0] >= cells[0] and region[1] <= cells[1] and region[2] >= cells[2] and region[3] <= cells[3]:
        return pair.eps
    surface = window.surface
    extent = wp.max(pair.extent, wp.abs(float(region[2] - window.c0) * surface.dx))
    extent = wp.max(extent, wp.abs(float(region[3] + 1 - window.c0) * surface.dx))
    extent = wp.max(extent, wp.abs(float(region[0] - window.r0) * surface.dy))
    extent = wp.max(extent, wp.abs(float(region[1] + 1 - window.r0) * surface.dy))
    for r in range(region[0], region[1] + 2):
        for c in range(region[2], region[3] + 2):
            extent = wp.max(extent, wp.abs(terrain_height(data, surface, r, c) - window.z_ref))
    return terrain_eps(extent)


@wp.func
def _exit_take(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    cells: wp.vec4i,
    best: _Exit,
    x: wp.vec3,
    bound: float,
    key: int,
) -> tuple[TerrainPlanes, _Exit]:
    """Search candidate ``x``'s exit (:func:`~newton._src.geometry.terrain_law_point.terrain_point_exit`) over the cells
    within ``bound`` of it, clamped to the heightfield, not to the window: its closest surface point lies there, since
    ``bound`` is at least its distance (widened by the band and the cluster tolerance, so that every claim at that
    distance is in), and the search is the same, bit for bit, as one over a window holding those cells. The exit is
    kept when deeper than the best (the first in window order on ties)."""
    radius = bound + 2.0 * _FLAT + 4.0 * pair.eps
    region = _exit_cells(window, x, radius)
    eps = _exit_eps(data, window, pair, cells, region)
    planes, found, q, n, d, feature = terrain_point_exit(data, window, planes, x, region, eps, pair.reach)
    best.searches = best.searches + 1
    best.cells = best.cells + (region[1] - region[0] + 2) * (region[3] - region[2] + 2)
    if found and (best.found == 0 or d < best.d or (d == best.d and key < best.key)):
        best.found = 1
        best.q = q
        best.n = n
        best.d = d
        best.feature = feature
        best.key = key
        best.threshold = wp.max(best.target, -d)
    return planes, best


@wp.func
def _exit_open(best: _Exit, key: int, bound: float, scale: float) -> bool:
    """Whether candidates of bound ``bound / scale`` (``scale`` positive) from key ``key`` on can still matter: in the
    first pass, whether one can join the two largest bounds; in the second, whether one can beat the best exit, or tie
    it from earlier in window order."""
    if best.walk == 0:
        return bound > wp.max(best.target, best.second) * scale
    if key == best.first_key:
        return False
    limit = best.threshold * scale
    return bound > limit or (bound == limit and best.found != 0 and key < best.key)


@wp.func
def _exit_try(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    cells: wp.vec4i,
    best: _Exit,
    x: wp.vec3,
    bound: float,
    key: int,
) -> tuple[TerrainPlanes, _Exit]:
    """Take candidate ``x`` of bound ``bound`` (key ``key``) while its bound, and still once lowered by
    :func:`_near_bound`, can matter (:func:`_exit_open`): in the first pass it joins the two largest bounds (the
    first in window order on ties), in the second its exit is searched (:func:`_exit_take`)."""
    if not _exit_open(best, key, bound, 1.0):
        return planes, best
    stop = best.threshold
    if best.walk == 0:
        stop = wp.max(best.target, best.second)
    bound = _near_bound(data, window, x, bound, stop)
    if not _exit_open(best, key, bound, 1.0):
        return planes, best
    if best.walk != 0:
        return _exit_take(data, window, planes, pair, cells, best, x, bound, key)
    if bound > best.first:
        best.second = best.first
        best.first = bound
        best.first_key = key
        best.first_x = x
    else:
        best.second = bound
    return planes, best


@wp.func
def _exit_valley(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    cells: wp.vec4i,
    best: _Exit,
    low: float,
    r: int,
    c: int,
    which: int,
    key: int,
) -> tuple[TerrainPlanes, _Exit]:
    """Rule X at edge ``which`` of vertex ``(r, c)`` when it is a valley (concave crease): the core's support points at
    the local minima of ``sigma`` over the valley's normal arc, the core's deepest points in the wedge below it (keys
    from ``key``: two per arc sample).

    The search: nine equally spaced normals; each no higher than its neighbours (the first of a run of equal ones)
    refined between them by bisection on the sign of the derivative (rule E's 22 steps), the sample kept where the
    refinement ends higher (the float64 law's search, :func:`newton.tests.terrain_law_reference._arc_angles`, in
    float64 and to its precision). Skipped where no core point in the wedge can matter (:func:`_exit_open`): within it
    a point's distance to the edge is at most its height under the edge's higher end over the arc's smaller normal
    ``z`` (``low`` is the core's lowest height).
    """
    surface = window.surface
    if terrain_edge_kind(data, surface, r, c, which) >= 0:
        return planes, best
    v0i, v1i, ta, tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    top = wp.max(v0[2], v1[2])
    if top <= low:
        return planes, best
    planes, na, pa = terrain_plane(data, window, planes, ta[0], ta[1], ta[2])
    if terrain_crease_class(data, window, r, c, which, na, pa) != TERRAIN_EDGE_CONCAVE:
        return planes, best
    planes, nb, _pb = terrain_plane(data, window, planes, tb[0], tb[1], tb[2])
    e = v1 - v0
    length2 = wp.dot(e, e)
    dh = e / wp.sqrt(length2)
    # the valley's normal arc, exactly perpendicular to the edge
    n_a = wp.normalize(na - wp.dot(na, dh) * dh)
    n_b = wp.normalize(nb - wp.dot(nb, dh) * dh)
    if wp.length(n_b - n_a) < 1.0e-9:
        return planes, best
    if not _exit_open(best, key, top - low, wp.min(n_a[2], n_b[2])):
        return planes, best
    samples = _arc_samples(n_a, n_b)
    values = _ArcValues()
    for i in range(9):
        values[i] = core_sigma(core, _arc_sample(samples, i), v0)
    axis = wp.normalize(wp.cross(n_a, n_b))
    for i in range(9):
        if (i > 0 and values[i] >= values[i - 1]) or (i < 8 and values[i] > values[i + 1]):
            continue
        lo = _arc_sample(samples, wp.max(i - 1, 0))
        hi = _arc_sample(samples, wp.min(i + 1, 8))
        for _step in range(_BISECTION):
            mid = wp.normalize(lo + hi)
            if core_slope(core, mid, wp.cross(axis, mid), v0, 1.0) < 0.0:
                lo = mid
            else:
                hi = mid
        nst = wp.normalize(lo + hi)
        if core_sigma(core, nst, v0) > values[i]:
            nst = _arc_sample(samples, i)
        for j in range(core_support_count(core)):
            x = core_support_point(core, nst, j)
            if x[2] >= top:
                # closest to the valley only below its line
                continue
            planes, bound = _exit_bound(data, window, planes, x)
            if bound > 0.0:
                s = wp.clamp(wp.dot(x - v0, e) / length2, 0.0, 1.0)
                bound = wp.min(bound, wp.length(x - (v0 + s * e)))
            planes, best = _exit_try(data, window, planes, pair, cells, best, x, bound, key + 2 * i + j)
    return planes, best


@wp.func
def _exit_candidates(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    cells: wp.vec4i,
    best: _Exit,
    low: float,
) -> tuple[TerrainPlanes, _Exit]:
    """One pass over rule X's candidates in window order (:func:`solid_exits`), each with its key: the core's points
    tried as they are first (keys 0 to their count), then per window vertex (keys from its window index times
    :data:`TERRAIN_SOLID_EXIT_SLOTS`) its valleys' support points (18 keys per edge) and its local maxima of the
    distance (from key 54)."""
    r0 = cells[0]
    r1 = cells[1]
    c0 = cells[2]
    c1 = cells[3]
    points = core_exit_count(core)
    for i in range(points):
        x = core_exit_point(core, i)
        planes, bound = _exit_bound(data, window, planes, x)
        planes, best = _exit_try(data, window, planes, pair, cells, best, x, bound, i)
    far = core_far_count(core)
    if core_support_count(core) == 0 and far == 0:
        return planes, best
    width = c1 + 2 - c0
    for r in range(r0, r1 + 2):
        for c in range(c0, c1 + 2):
            base = points + ((r - r0) * width + (c - c0)) * _EXIT_SLOTS
            inside = r <= r1 and c <= c1
            for which in range(3):
                if (which == 0 and c > c1) or (which == 1 and r > r1) or (which == 2 and not inside):
                    continue
                planes, best = _exit_valley(
                    core, data, window, planes, pair, cells, best, low, r, c, which, base + 18 * which
                )
            if far == 0 or not terrain_interior_vertex(window.surface, r, c):
                continue
            v = terrain_window_vertex(data, window, r, c)
            if v[2] <= low:
                continue
            for i in range(far):
                has, x = core_far_point(core, i, v)
                if not has or x[2] >= v[2]:
                    # closest to the vertex only below it
                    continue
                planes, bound = _exit_bound(data, window, planes, x)
                if bound > 0.0:
                    bound = wp.min(bound, wp.length(x - v))
                planes, best = _exit_try(data, window, planes, pair, cells, best, x, bound, base + 54 + i)
    return planes, best


@wp.func
def solid_exits(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: TerrainLawRows,
    cells: wp.vec4i,
    work: wp.array[wp.int32],
) -> tuple[TerrainPlanes, _Sink]:
    """Rule X: the exit of the deepest of the core's candidate boundary points, offered when it lies deeper than every
    row the pair holds by more than :data:`TERRAIN_SOLID_EXIT_MARGIN` (point slot :data:`TERRAIN_SOLID_EXIT_SLOT`);
    its exit searches and the window vertices they covered are added to ``work``.

    A deepest boundary point whose closest surface point is unique is a local maximum of its distance to that one
    feature: a support point of rule F's plane manifold for a face, an extreme point of the core for a valley edge or
    a vertex. The candidates, in window order (:func:`_exit_candidates`): the core's points tried as they are (a box's
    corners, a hull's points, a cone's apex; :func:`~newton._src.geometry.terrain_law_cores.core_exit_point`), then,
    window vertex by window vertex, each valley's (:func:`_exit_valley`) and, at an interior vertex above the core's
    lowest point, the local maxima of the distance from it (each rim's farthest point, an ellipsoid's one or two;
    :func:`~newton._src.geometry.terrain_law_cores.core_far_point`). A candidate's bound is its exit bound or its
    distance to the feature that proposed it, the smaller, lowered by :func:`_near_bound`; the rule keeps the deepest
    exit, the first in window order on ties, whatever order the exits are searched in. A first pass finds the two
    largest bounds above the pair's depth and the largest's exit is searched first; a second pass searches the exits
    of the candidates that can still beat it (or tie it from earlier in window order) only when the second largest
    bound can. Each exit search covers the cells within the candidate's bound (:func:`_exit_take`), not the window:
    its cost depends on the candidate's depth, not the window's size. A pair whose core lies wholly above its window's
    highest vertex has no candidate (``pair.top``).
    """
    lo, _hi = core_bounds(core)
    low = lo[2]
    if low >= pair.top:
        return planes, sink
    deepest = float(0.0)
    for h in range(wp.min(sink.held, sink.capacity)):
        deepest = wp.max(deepest, -store.d[sink.slot + h])
    best = _Exit()
    best.target = deepest + _EXIT_MARGIN
    best.threshold = best.target
    best.first = best.target
    best.second = best.target
    best.first_key = -1
    for walk in range(2):
        if walk == 1:
            if best.first <= best.target:
                break
            planes, best = _exit_take(data, window, planes, pair, cells, best, best.first_x, best.first, best.first_key)
            # every other candidate of the largest bound comes later in window order: it cannot win a tie
            if best.second < best.threshold or (best.second == best.threshold and best.threshold >= best.first):
                break
            best.walk = 1
        planes, best = _exit_candidates(core, data, window, planes, pair, cells, best, low)
    if best.searches > 0:
        wp.atomic_add(work, 0, best.searches)
        wp.atomic_add(work, 1, best.cells)
    if best.found != 0 and -best.d > best.target:
        feature = best.feature - (best.feature & 7) + _EXIT_SLOT
        sink = _offer(store, sink, best.q, best.n, best.d, feature)
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
def _window_extent_top(
    data: TerrainFeatureData, window: TerrainWindow, r1: int, c1: int, extent: float
) -> tuple[float, float]:
    """:func:`~newton._src.geometry.terrain_law.terrain_window_extent` (the same value, bit for bit) and, read in the
    same pass, the window's highest vertex (window frame), which rule X's gate takes."""
    surface = window.surface
    extent = wp.max(extent, wp.max(float(c1 + 1 - window.c0) * surface.dx, float(r1 + 1 - window.r0) * surface.dy))
    top = float(-_HUGE)
    for r in range(window.r0, r1 + 2):
        for c in range(window.c0, c1 + 2):
            z = terrain_height(data, surface, r, c) - window.z_ref
            extent = wp.max(extent, wp.abs(z))
            top = wp.max(top, z)
    return extent, top


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
    extent, top = _window_extent_top(data, window, cells[1], cells[3], extent)
    eps = terrain_eps(extent)
    size = wp.length(hi - lo)
    pair.extent = extent
    pair.top = top
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
def solid_finish(
    window: TerrainWindow,
    pair: _Pair,
    sink: _Sink,
    out: TerrainLawRows,
    pair_index: int,
    partner_class: int,
    frame: wp.transform,
    cells: wp.vec4i,
):
    """Finish a pair's held rows in its walk slots (``sink.slot``): defer it when its distinct rows overflow them, else
    run the finish in place and write the kept rows (world frame) at the front of its slots."""
    if sink.held > sink.capacity:
        terrain_defer_pair(out, partner_class, pair_index, _wide(cells))
        return
    keep, kept = terrain_finish_slots(out.q, out.n, out.d, sink.slot, sink.held, pair.eps)
    written = int(0)
    for i in range(sink.held):
        if ((keep >> i) & 1) != 0:
            s = sink.slot + i
            terrain_store_row(out, sink.slot + written, out.q[s], out.n[s], out.d[s], out.feature[s], window, frame)
            written += 1
    out.first[pair_index] = sink.slot
    out.count[pair_index] = kept
    out.state[pair_index] = _wide(cells)


@wp.func
def _exit_gate(core: Any, pair: _Pair) -> bool:
    """Whether rule X can give a pair a row: its core's lowest point lies below its window's highest vertex."""
    lo, _hi = core_bounds(core)
    return lo[2] < pair.top


@wp.func
def solid_close(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    out: TerrainLawRows,
    solids: TerrainSolidData,
    pair_index: int,
    partner_class: int,
    frame: wp.transform,
    cells: wp.vec4i,
):
    """Close a walked pair in its walk slots (``sink.slot``): defer it when its distinct rows overflow them, else add
    the recovery rows when it has none; a pair whose core reaches below its window's highest vertex is left, its rows
    held (count in ``out.count``, state :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_EXIT`), to its class's
    exit kernel (rule X, :func:`solid_rows_exit`), any other finished here (:func:`solid_finish`). Rule X's work thus
    stays off the walks: their warps, registers and stacks."""
    if sink.held > sink.capacity:
        terrain_defer_pair(out, partner_class, pair_index, _wide(cells))
        return
    if sink.raw == 0:
        planes, sink = solid_recovery(core, data, window, planes, sink, out)
    if _exit_gate(core, pair):
        out.first[pair_index] = sink.slot
        out.count[pair_index] = sink.held
        out.state[pair_index] = _wide(cells) | TERRAIN_PAIR_EXIT
        entry = wp.atomic_add(solids.exit_count, partner_class, 1)
        solids.exits[out.deferred_first[partner_class] + entry] = pair_index
        return
    solid_finish(window, pair, sink, out, pair_index, partner_class, frame, cells)


@wp.func
def solid_rows_exit(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    solids: TerrainSolidData,
    pair_index: int,
    slot: int,
    walk_slots: int,
    partner_class: int,
    frame: wp.transform,
    heightfield: int,
    reach: float,
):
    """Rule X for a pair its walk left to the class's exit kernel, its rows held in its walk slots from ``slot``
    (``out.count`` of them): its window again (the walk's, bit for bit), the deepest exit (:func:`solid_exits`), then
    the finish (:func:`solid_finish`), which gives the pair the rows a walk that ran rule X itself would."""
    _state, cells, window, core, pair = solid_pair_window(core, data, heightfield, reach)
    planes = terrain_planes()
    sink = solid_sink(slot, walk_slots, pair.eps, 0)
    sink.held = out.count[pair_index]
    sink.raw = sink.held
    planes, sink = solid_exits(core, data, window, planes, pair, sink, out, cells, solids.exit_work)
    solid_finish(window, pair, sink, out, pair_index, partner_class, frame, cells)


@wp.func
def solid_close_full(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    scratch: TerrainFinishScratch,
    out: TerrainLawRows,
    work: wp.array[wp.int32],
    pair_index: int,
    slot: int,
    walk_slots: int,
    frame: wp.transform,
    cells: wp.vec4i,
):
    """Finish a deferred pair walked into a full-finish block's scratch (``sink.slot``): add the recovery rows when it
    has none and rule X's exit (its searches counted in ``work``), run the finish on every distinct candidate and write
    the kept rows (world frame) to its walk slots from ``slot`` when they fit, else to rows of the spill region, or,
    when that has no room left, the first of them by priority to its walk slots (flagged
    :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_ROW_OVERFLOW`).

    The scratch holds every candidate a window of the class can give; a pair with more has a window its shape data no
    longer bounds (changed since construction): it keeps the first in window order and is flagged."""
    store = solid_scratch_store(scratch)
    if sink.raw == 0:
        planes, sink = solid_recovery(core, data, window, planes, sink, store)
    planes, sink = solid_exits(core, data, window, planes, pair, sink, store, cells, work)
    flags = int(0)
    if sink.held > sink.capacity:
        sink.held = sink.capacity
        flags = TERRAIN_FINISH_CANDIDATE_OVERFLOW
    for i in range(sink.held):
        s = sink.slot + i
        # a row's core point, as the in-place finish of a walk reads it
        scratch.k[s] = scratch.q[s] + scratch.d[s] * scratch.n[s]
    kept = terrain_finish_list(scratch, sink.slot, sink.held, pair.eps)
    if kept > walk_slots:
        first = terrain_spill_rows(out, kept)
        if first >= 0:
            slot = first
        else:
            # no room left: the finish's first rows by priority, in the pair's own walk slots
            kept = terrain_finish_overflow(scratch, sink.slot, sink.held, walk_slots)
            flags = flags | TERRAIN_FINISH_ROW_OVERFLOW
    written = int(0)
    for i in range(sink.held):
        s = sink.slot + i
        if scratch.mark[s] != 0:
            terrain_store_row(
                out, slot + written, scratch.q[s], scratch.n[s], scratch.d[s], scratch.feature[s], window, frame
            )
            written += 1
    out.first[pair_index] = slot
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
    walk_slots: int,
    partner_class: int,
    frame: wp.transform,
    heightfield: int,
    reach: float,
):
    """One thread's walk of a pair whose core is built (heightfield frame) in its walk slots from ``slot``: window,
    walk, close; a pair from ``work.first`` on (the tiled segment) whose window holds more than
    ``solids.thread_vertices`` vertices is queued in ``work`` for the class's tiled kernel instead."""
    state, cells, window, core, pair = solid_pair_window(core, data, heightfield, reach)
    if state != 0:
        terrain_pair_exit(out, pair_index, state)
        return
    if solids.thread_vertices > 0 and pair_index >= work.first:
        if (cells[1] - cells[0] + 2) * (cells[3] - cells[2] + 2) > solids.thread_vertices:
            work.pairs[wp.atomic_add(work.count, 0, 1)] = pair_index
            return
    planes = terrain_planes()
    sink = solid_sink(slot, walk_slots, pair.eps, 0)
    planes, sink = solid_walk(core, data, window, planes, pair, sink, out, cells)
    solid_close(core, data, window, planes, pair, sink, out, solids, pair_index, partner_class, frame, cells)


@wp.func
def solid_rows_full(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    scratch: TerrainFinishScratch,
    solids: TerrainSolidData,
    block: int,
    pair_index: int,
    slot: int,
    walk_slots: int,
    frame: wp.transform,
    heightfield: int,
    reach: float,
):
    """A deferred pair walked again into block ``block``'s scratch, finished and written (:func:`solid_close_full`;
    its walk slots from ``slot``)."""
    _state, cells, window, core, pair = solid_pair_window(core, data, heightfield, reach)
    planes = terrain_planes()
    sink = solid_sink(block * scratch.capacity, scratch.capacity, pair.eps, 0)
    planes, sink = solid_walk(core, data, window, planes, pair, sink, solid_scratch_store(scratch), cells)
    solid_close_full(
        core,
        data,
        window,
        planes,
        pair,
        sink,
        scratch,
        out,
        solids.exit_work,
        pair_index,
        slot,
        walk_slots,
        frame,
        cells,
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
    solids: TerrainSolidData,
    pair_index: int,
    slot: int,
    walk_slots: int,
    partner_class: int,
    frame: wp.transform,
):
    """The tiled kernels' first lane: the marked vertices' items in window order (every vertex when the window is too
    large to mark) in the pair's walk slots from ``slot``, then the pair's close: the rows (or the deferral, or the
    exit list) the one-thread walk gives the pair."""
    planes = terrain_planes()
    emitters = _emitters()
    sink = solid_sink(slot, walk_slots, pair.eps, 0)
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
    solid_close(core, data, window, planes, pair, sink, out, solids, pair_index, partner_class, frame, cells)


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
    pairs: int | None = None,
    exit_count: wp.array | None = None,
    exit_work: wp.array | None = None,
) -> TerrainSolidData:
    """The solid classes' data for a model: the hull table of the convex-mesh partners in ``partners`` (shape
    indices), the one-thread kernels' lane choice (``thread_vertices`` 0: they walk every window) and rule X's exit
    lists. Their launch arguments, the work lists and the full finish's scratch, are the shadow's
    (:func:`~newton._src.geometry.terrain_law.terrain_pair_work`,
    :func:`~newton._src.geometry.terrain_law.terrain_finish_scratch`).

    Args:
        model: A finalized model.
        partners: Partner shapes of the solid classes' pairs.
        thread_vertices: Window vertices up to which the one-thread kernels walk a pair.
        pairs: Pairs of the table (the exit lists' room), or ``None`` for one per entry of ``partners``.
        exit_count: The four per-class counters of the exit lists (zeroed before every launch), or ``None`` for its
            own.
        exit_work: The two counters of rule X's exit searches (zeroed before every launch), or ``None`` for its own.
    """
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
    room = len(partners) if pairs is None else int(pairs)
    solids.exits = wp.zeros(max(room, 1), dtype=wp.int32, device=device)
    solids.exit_count = exit_count if exit_count is not None else wp.zeros(4, dtype=wp.int32, device=device)
    solids.exit_work = exit_work if exit_work is not None else wp.zeros(2, dtype=wp.int32, device=device)
    return solids

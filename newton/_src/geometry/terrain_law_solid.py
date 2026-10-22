# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law for solid cores (boxes, cylinders and cones, convex hulls and ellipsoids): its rules, and the kernels
that run them on a class's pairs.

A solid core has no rounding radius; its reach ``D`` is both margins and both gaps (the contact writer's admission
limit). Per pair the law visits every surface feature of the window once (window vertex ``(r, c)`` owns its cell's
two faces and diagonal, its x- and y-edges and itself) and offers the rows of the law's rules for solids, in window
order:

* F, faces: the analytic plane manifold of the core against the triangle's cluster plane
  (:func:`~newton._src.geometry.terrain_law_cores.core_manifold`); a manifold point is supported when its footprint
  along the normal lies in a triangle of the cluster and it is front-facing, and the triangle holding the first
  supported point emits every supported point (one emission per plane);
* E, convex creases: ``theta* = argmax sigma(theta)`` over the crease's normal arc: nine equally spaced normals, then
  a bisection on the sign of the derivative between the best one's neighbours
  (:func:`~newton._src.geometry.terrain_law_cores.core_slope`, closed form for boxes and cylinders, the support
  function for hulls; 22 steps; the float64 law brackets the same samples and refines by golden section on
  ``sigma``); inside the arc, the core's flat support feature within the band
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

Rule X runs after the walk's rows (and rule R), only for a core reaching below its window's highest vertex (read in
the pass that sets the window's band). A candidate's bound is its depth below the plane of the face under it when its
projection stays in that face's cluster, else its distance to that triangle, or its distance to the feature that
proposed it, the smallest, lowered by a descent over blocks of 3 x 3 cells toward its closest surface point
(:func:`_near_bound`). A first pass over the candidates takes the two largest bounds above the pair's depth, and the
largest's exit is searched; a second pass searches the exits of the candidates that can still beat it (or tie it from
earlier in window order) only when the second largest bound can; a valley is skipped where the wedge below it cannot
hold such a point. An exit search covers the cells within its candidate's bound, so that its work follows the
candidate's depth, not the window's size: a pair's rule X work is its candidates' (a few per window vertex for
cylinders, cones and ellipsoids, a box's or hull's points) and, on buried solids under rough slopes, one exit search.

The kernels (:func:`solid_kernels`, one module per partner class, in the order of :data:`TERRAIN_SOLID_LAUNCHES`) take
the shape of today's terrain route: a cheap scan that lists the features that survive the cheapest rejects, one thread
per listed feature, and a per-pair close that holds their rows in window order, so that a warp runs one rule on many
pairs' features at once instead of one pair's whole window, and no thread walks a large window alone:

1. The whole-pair rejects, one thread per pair; a pair past them keeps its core and window frame
   (:class:`TerrainSolidRecord`) for the scan and the jobs and is listed live, its window in runs of its class's
   :data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN` window vertices. The scan
   (:func:`solid_scan`), one thread per run: the run's
   features through their cheapest rejects in window order: rule F's faces (one per cluster plane and run: its
   manifold, support tests and emitting triangle are its cluster's alone, :func:`_face_support`), rule E's edges
   (:func:`_edge_cheap`) and rule V's vertices (:func:`_vertex_cheap`), each survivor listed for its job kernel.
2. The job kernels, a fixed grid striding over each list: :func:`solid_face_job`, :func:`solid_edge_job` (its crease
   tests, :func:`_edge_arc`; a crease whose arc can hold a row is listed for ``theta*``), :func:`solid_theta_job`
   (``theta*``, :func:`_edge_theta`, kept per listed crease), :func:`solid_crease_job` (the crossing at ``theta*``
   and its ends, :func:`_edge_rows_at`; apart, so that the search runs in a lean kernel) and :func:`solid_vertex_job`
   (:func:`_vertex_rows`).
   A job adds its rows to the class's candidate pool, each with its place in window order (its key: its item's index,
   window vertex times six plus its item, times eight plus its point slot), the row the walk would offer there.
3. The close (:func:`solid_close_jobs`), one thread per live pair: the pair's candidates held in its walk slots in key
   order (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`, the walk's held rows bit for bit, however the
   jobs ran; a candidate listed twice is one offer), then rule R's kernel for a pair without a row, a block of lanes
   per pair splitting its core points (:func:`solid_recover_pair`); when the law runs rule X
   (:attr:`TerrainSolidJobs.rule_x`, ``NEWTON_TERRAIN_LAW_RULE_X``; off by default), a pair whose core reaches below
   its window's highest vertex is left, its rows held, to rule X, with one item per core point and window vertex that
   can give a candidate listed, and any other is finished (step 5).
4. Rule X, when the law runs it: its gate (:func:`solid_xbound_job`, an exit bound per cell), items
   (:func:`solid_xitem_job`: a core point, or a window vertex that can give a candidate) and valleys
   (:func:`solid_xvalley_job`), the job threads: each lists the candidates whose bounds pass the pair's target, the
   test the rule's passes open with, and a descent kernel takes each listed candidate's near-bound descent
   (:func:`solid_xdescent_job`), so that the exit kernel (:func:`solid_exit_pair`, one thread per pair left to it)
   runs the rule's passes over the listed candidates alone (:func:`solid_exits`, :func:`_exit_listed`) and searches
   their exits: any other candidate is a no-op in both passes, and the rows are the passes' over every candidate. A
   pair some of whose rule X work found no room is left to the full-finish kernels, which run rule X over every
   candidate.
5. The finish: a pair whose held rows hold no plane group of three is written as it is by the thread that holds it
   (its close, rule R's first lane or the exit kernel); any other is listed for the small-pair finish kernel, one
   thread per pair (:func:`solid_finish_pair`). A block of lanes per pair (the finish kernel) would take a pair of
   :data:`TERRAIN_SOLID_LANE_ROWS` rows, which no walk holds.

The kernels follow the per-pair output protocol of :mod:`~newton._src.geometry.terrain_law`: a pair with no row is
recorded by :func:`~newton._src.geometry.terrain_law.terrain_pair_exit`, and rows are held in the pair's own walk slots
(window frame) as the finish merges them (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`): a row coinciding
with a held one is dropped, the first in window order staying. The finish runs in place on the held rows
(:func:`~newton._src.geometry.terrain_law.terrain_finish_slots`, a row's core point being ``q + d n``; the same rows
whatever the number of lanes), and the kept
rows are rewritten in the world frame at the front of the slots. A pair with more distinct candidates than its slots,
or whose work found no room in a job list or the candidate pool (scene-level buffers sized from the pairs' bounds,
:func:`terrain_solid_job_capacities`), is left to its class's full-finish kernels
(:func:`~newton._src.geometry.terrain_law.terrain_defer_pair`): two launches on the shadow's fixed full-finish grid
stride over the class's deferred list, each block's first lane taking its pairs one at a time into the block's scratch
(:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`), which holds every distinct candidate a window of the
class can give (:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`): the pooled kernel holds a pair
every job of which found room from its candidates in the pool, in window order (:func:`solid_pooled`), and the
full-finish kernel walks the others again (:func:`solid_rows_full`, :func:`solid_walk`). The finish runs on all of
them (:func:`~newton._src.geometry.terrain_law.terrain_finish_list`) and every row it keeps is written,
in the pair's walk slots when they fit, else in rows of the spill region it reserves
(:func:`~newton._src.geometry.terrain_law.terrain_spill_rows`). Every pair is thus finished from its own candidates,
whatever the order of the jobs, the number of deferred pairs, the order of the deferral or the block that finishes it.
Every module that compiles the kernels disables floating-point contraction, so the CPU decides every boundary as the GPU
does.
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
    TERRAIN_LAW_BLOCK,
    TERRAIN_PAIR_EXIT,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_FULL,
    TERRAIN_PAIR_HELD,
    TERRAIN_PAIR_INCOMPLETE,
    TERRAIN_PAIR_JOBS,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_UNSUPPORTED,
    TERRAIN_PAIR_WIDE,
    TERRAIN_STATUS_JOBS,
    TERRAIN_VERTEX_FEATURE_BIT,
    TERRAIN_WINDOW_WIDE,
    TerrainFeatureData,
    TerrainFinishScratch,
    TerrainLawRows,
    TerrainPairInputs,
    TerrainPlanes,
    TerrainSurfaceInfo,
    TerrainWindow,
    terrain_box_cells,
    terrain_cluster,
    terrain_crease_class,
    terrain_defer_pair,
    terrain_edge_open_crease,
    terrain_edge_open_valley,
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
    terrain_interior_vertex,
    terrain_locate,
    terrain_pair_exit,
    terrain_plane,
    terrain_plane_group_of_three,
    terrain_planes,
    terrain_point_below_cached,
    terrain_report,
    terrain_spill_rows,
    terrain_store_row,
    terrain_tile_exit,
    terrain_triangle_vertices,
    terrain_vertex_bits,
    terrain_vertex_feature,
    terrain_vertex_in_cone,
    terrain_window,
    terrain_window_point,
    terrain_window_vertex,
)
from .terrain_law_cores import (
    TerrainBoxCore,
    TerrainCrossing,
    TerrainCylinderCore,
    TerrainHullCoreRecord,
    TerrainHullTable,
    closest_on_triangle,
    core_bounds,
    core_capsule,
    core_column,
    core_exit_count,
    core_exit_point,
    core_exits_gated,
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
    core_sigma_bound,
    core_sigma_near,
    core_slope,
    core_slope_near,
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
TERRAIN_SOLID_SEEN = 8
"""Clusters of several triangles a scan job remembers (round robin): it lists the first face of each it meets for
rule F, whose emitting triangle and rows are its cluster's alone. A cluster forgotten and met again, or met by another
of the pair's scan jobs, is listed again; its rows, the same and at the same place in window order, merge in the
close."""
TERRAIN_SOLID_JOB_LISTS = ("faces", "edges", "thetas", "vertices", "xitems", "xvalleys")
"""The job lists of a solid class, in their counters' order (the candidate pool's counter follows): rule F's faces, rule
E's edges, rule E's creases whose arcs can hold a row (``theta*``), rule V's vertices, rule X's items (a core point and
a window vertex each) and rule X's valleys."""
TERRAIN_SOLID_JOB_SHARES = (8, 16, 32, 8, 4, 32)
"""Per job list (:data:`TERRAIN_SOLID_JOB_LISTS`), the share of its pairs' bounds it holds by default
(:func:`terrain_solid_job_capacities`): the largest pair's bound plus the sum of every pair's bound over this. A pair's
bound on a list is its window's features of the list's kind at most (two faces, three edges or one vertex per window
vertex, :func:`~newton._src.geometry.terrain_contacts.terrain_window_vertices`; rule X's items: its window's vertices
or its core's points, the more)."""
TERRAIN_SOLID_CANDIDATE_SHARE = 4
"""The candidate pool of a solid class holds, by default, its pairs' walk slots over this, plus the largest pair's bound
on its candidates (:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`)."""
TERRAIN_SOLID_JOB_FLOOR = 1 << 14
"""Entries a solid class's job list holds by default at least, or all its pairs' bounds when they sum to less: the work
of a scene with few pairs always finds room (16,384 entries, 128 KiB)."""
TERRAIN_SOLID_POOL_FLOOR = 1 << 16
"""Entries a solid class's candidate pool holds by default at least, or all its pairs' bounds when they sum to less
(65,536 entries, 2.5 MiB)."""
TERRAIN_SOLID_EXIT_SHARE = 256
"""Rule X's listed candidates of a solid class: by default the largest pair's bound on them (every rule X key of its
window, :data:`TERRAIN_SOLID_EXIT_SLOTS` per window vertex) plus its pairs' rule X items over this (one candidate per
item for one pair in 256 at once; the spill region's share, :data:`~newton._src.geometry.terrain_shadow.
TERRAIN_LAW_SPILL_SHARE`), at least the class's keys up to :data:`TERRAIN_SOLID_JOB_FLOOR`. Only a pair whose core
reaches below its window's highest vertex lists any, and only those deeper than every row it holds: the recorded suite
states list at most 587 per class and collide."""
TERRAIN_SOLID_JOB_THREADS = 1 << 17
"""Threads of a job kernel's launch on CUDA at most (fewer with a smaller list): a fixed grid that strides over its
list."""
TERRAIN_SOLID_JOB_THREADS_CPU = 4
"""Threads of a job kernel's launch on the CPU, which runs a launch's threads one after another: a few, so that each
job's cost is the job's (they still stride over the list)."""
TERRAIN_SOLID_FINISH_BLOCKS = TERRAIN_SOLID_JOB_THREADS // TERRAIN_LAW_BLOCK
"""Blocks of rule R's and the finish kernel's launches on CUDA at most (fewer with fewer pairs): the job threads' worth
of :data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK`-lane blocks, each taking one pair of its list at a time
with its lanes."""
TERRAIN_SOLID_RECOVERY_CLEAR = 1.0e-4
"""Height [m] by which a core without a candidate must clear its window's highest vertex for its close to leave it
without a row instead of listing it for rule R: every core point then lies above the plane of the face under it (a
cluster plane lies within :data:`~newton._src.geometry.terrain_law.TERRAIN_FLAT_EPS` of its triangles' vertices) by
far more than the float32 rounding of its plane distance, so rule R would find none of them below the surface."""
TERRAIN_SOLID_LANE_ROWS = 32
"""Held rows from which a pair's finish takes a block of lanes (the finish kernel); a pair with fewer is written as it is
or finished by one thread of the small-pair finish kernel. Above every class's walk slots, so that no walking pair takes
the lanes and a class launches no finish kernel: on the law's own operating states one thread per pair (the block's
every lane staging the pair, zero-filling its arrays, otherwise) was faster for every class, by 0.04 to 0.34 ms per env
step."""
TERRAIN_SOLID_SCAN_VERTICES = 32
"""Window vertices one scan job takes at most (the bits of a run's survivor masks; a class's run is
:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`), and the window vertices of one run of rule
X's item listing (one reservation each): a large window is scanned by several threads, so that no thread scans a large
window alone. A cluster plane met by two runs is listed twice; its rows, the same at the same place in window order,
merge in the close."""
TERRAIN_SOLID_FEW = 4
"""Candidates of a pair the close sorts in registers (:func:`_hold_few`); a longer list is sorted in chunks of
:data:`TERRAIN_SOLID_SORTED` on the stack."""
TERRAIN_SOLID_SORTED = 32
"""Keys of a pair's pool list sorted in registers per walk of the list: the close (:func:`_hold_candidates`) and the
exit kernel (:func:`_exit_listed`) take a list of ``n`` candidates in key order in ``ceil(n / 32)`` walks, each keeping
the smallest keys past the last one taken (a full chunk drops its largest, which a later walk takes)."""
TERRAIN_HULL_FACE_TOL = 1.0e-6
"""A hull's points within this of a face's plane lie on that face [m] (hulls under 1 m; their local frame): a planar
face of a float32 mesh stays one face (its points lie within about 1e-8 m of its plane)."""
TERRAIN_SOLID_EXIT_MARGIN = 1.0e-5
"""Rule X offers a pair's deepest exit only where it lies deeper than every other row of the pair by more than this
[m] (the float64 law's): a pair whose rows reach its depth keeps them as they are."""
TERRAIN_SOLID_EXIT_SLOT = 7
"""Point slot of rule X's row in its feature id: no other solid rule uses it (F 0-3, E 0-1, V 0, R 0-3)."""
TERRAIN_SOLID_EXIT_DESCENT = 4
"""Blocks of 3 x 3 cells rule X's near bound visits at most per candidate, each after the first centred on the cell
under the candidate's projection onto the plane of the last block's nearest triangle (:func:`_near_bound`): a descent
toward the candidate's closest surface point, which only tightens the bound that decides whether its exit is
searched."""
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
_ITEMS = wp.constant(6)
_EXIT_MARGIN = wp.constant(TERRAIN_SOLID_EXIT_MARGIN)
_EXIT_SLOT = wp.constant(TERRAIN_SOLID_EXIT_SLOT)
_DESCENT = wp.constant(TERRAIN_SOLID_EXIT_DESCENT)
_EXIT_SLOTS = wp.constant(TERRAIN_SOLID_EXIT_SLOTS)
_CELL_WIDEN = wp.constant(TERRAIN_CELL_WIDEN)
_SEEN = wp.constant(TERRAIN_SOLID_SEEN)
_SORTED = wp.constant(TERRAIN_SOLID_SORTED)
_FEW = wp.constant(TERRAIN_SOLID_FEW)
_NO_KEY = wp.constant(2147483647)
_SCAN_VERTICES = wp.constant(TERRAIN_SOLID_SCAN_VERTICES)
_RECOVERY_CLEAR = wp.constant(TERRAIN_SOLID_RECOVERY_CLEAR)
_LANE_ROWS = wp.constant(TERRAIN_SOLID_LANE_ROWS)
_KEY_SLOTS = wp.constant(8)
_FACES = wp.constant(0)
_EDGES = wp.constant(1)
_THETAS = wp.constant(2)
_VERTICES = wp.constant(3)
_XITEMS = wp.constant(4)
_XVALLEYS = wp.constant(5)
_CANDIDATES = wp.constant(6)
_LIVE = wp.constant(7)
_RECOVER = wp.constant(8)
_SCANS = wp.constant(9)
_XDESCENTS = wp.constant(10)
_WALKS = wp.constant(11)
_LANES = wp.constant(12)
_SMALL = wp.constant(13)
_NO_POINT = wp.constant(1 << 30)

_Mat43 = wp.types.matrix(shape=(4, 3), dtype=wp.float32)
_ArcValues = wp.types.vector(length=TERRAIN_SOLID_ARC_SAMPLES, dtype=wp.float32)
_Seen = wp.types.vector(length=TERRAIN_SOLID_SEEN, dtype=wp.int32)
_Sorted = wp.types.vector(length=TERRAIN_SOLID_SORTED, dtype=wp.int32)

TERRAIN_SOLID_COUNTERS = len(TERRAIN_SOLID_JOB_LISTS) + 8
"""Counters of a solid class's jobs: one per job list, then the candidate pool's, the live pairs', the pairs left to
rule R, the scan jobs', the listed rule X candidates' descents, the deferred pairs the full-finish kernel walked
again, the pairs given to the finish kernel's lanes and those given to the small-pair finish kernel (zeroed before
every launch)."""


@wp.struct
class TerrainSolidData:
    """What the solid classes' kernels read besides the pair inputs, the terrain and their launch arguments (their
    class's jobs, the full finish's scratch): hull tables and rule X's lists."""

    hulls: TerrainHullTable
    """The hulls of the hull class (empty for the other classes)."""
    exits: wp.array[wp.int32]
    """The walked pairs left to their class's exit kernel (rule X; :func:`solid_close_jobs`): partner class ``c``'s
    list starts at the side buffer's ``deferred_first[c]`` (its first pair in the table) and holds ``exit_count[c]``
    pairs, appended in any order (a pair's rows depend on it alone)."""
    exit_count: wp.array[wp.int32]
    """Per partner class: the pairs in its exit list (zeroed before every launch)."""
    exit_work: wp.array[wp.int32]
    """Rule X's exit searches since the launch began (``[0]``) and the window vertices they covered (``[1]``)."""


@wp.struct
class _Sink:
    """A pair's held rows while its window is walked, or a job's rows on their way to its class's candidate pool."""

    slot: int
    """First slot of the pair's store; a job's: its pair's index in its class."""
    capacity: int
    """Slots of the store."""
    held: int
    """Distinct rows offered (held while within the capacity)."""
    raw: int
    """Rows offered, coincident ones included."""
    eps: float
    """The window's closed-predicate band [m]."""
    key: int
    """A job's place in window order: its item's index (window vertex times six plus its item) times eight; a row's
    key adds its point slot."""
    lost: int
    """1 when the candidate pool had no room for a job's row."""


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
class TerrainSolidRecord:
    """What a solid pair's whole-pair kernel keeps for its scan and job kernels: its window's cells and heightfield and
    its window-frame quantities (its core, window frame shift included, is in its class's core records)."""

    cells: wp.vec4i
    """The window's cells ``(r0, r1, c0, c1)``."""
    heightfield: int
    """The pair's heightfield index."""
    pair: _Pair
    """The pair's window-frame quantities."""
    low: float
    """The core's lowest point [m, window frame] (its bounds' in the window frame, as rule X takes them)."""


@wp.struct
class TerrainSolidJobs:
    """A solid class's jobs: the scan's runs and records, the job lists, the candidate pool and rule X's lists.

    Every list and the pool have fixed capacities (:func:`terrain_solid_jobs`); their counters (``counts``) are zeroed
    before every launch. Entries are appended in any order, and every row carries its place in window order (its key),
    so that a pair's rows depend on the pair alone. A pair whose work finds no room is left to its class's full-finish
    kernel, which walks it again (rule X's items, valleys or candidates without room: it takes every candidate of the
    window), and raises the narrow phase's ``terrain_jobs`` (:func:`_jobs_overflowed`).
    """

    first: int
    """The class's first table pair: pair ``first + i`` is the class's pair ``i``."""
    records: wp.array[TerrainSolidRecord]
    """Per class pair: its record (a live pair's, from the whole-pair kernel)."""
    box_cores: wp.array[TerrainBoxCore]
    """Per class pair of the box class: its core, window frame (one unused entry for another class)."""
    cylinder_cores: wp.array[TerrainCylinderCore]
    """Per class pair of the cylinder class: its core, window frame (one unused entry for another class)."""
    hull_cores: wp.array[TerrainHullCoreRecord]
    """Per class pair of the hull class: its core without its table, window frame (one unused entry for another
    class)."""
    faces: wp.array[wp.vec2i]
    """Rule F's jobs: ``(table pair, item)``, ``item`` its window vertex times six plus the face's triangle (0 or 1),
    one per cluster plane a scan job meets (:data:`TERRAIN_SOLID_SEEN`) and per triangle alone."""
    edges: wp.array[wp.vec2i]
    """Rule E's jobs: ``(table pair, item)``, ``item`` its window vertex times six plus 2, 3 or 4 (x-, y-, diagonal
    edge)."""
    thetas: wp.array[wp.vec2i]
    """Rule E's creases whose normal arcs can hold a row, for ``theta*``: ``(table pair, item)`` as the edges'."""
    theta_n: wp.array[wp.vec3]
    """Per entry of :attr:`thetas`: its crease's ``theta*`` (:func:`_edge_theta`), for :func:`solid_crease_job`."""
    theta_kind: wp.array[wp.int32]
    """Per entry of :attr:`thetas`: its ``theta*``'s kind (:func:`_edge_theta`: -1 no row, 0 an arc end, 1 inside)."""
    vertices: wp.array[wp.vec2i]
    """Rule V's jobs: ``(table pair, item)``, ``item`` its window vertex times six plus 5."""
    xitems: wp.array[wp.vec2i]
    """Rule X's items: ``(table pair, item)``, ``item`` a window vertex whose valleys or local maxima of the distance
    can give a candidate, or ``-1 - j`` for the core's point ``j``."""
    xvalleys: wp.array[wp.vec2i]
    """Rule X's valleys: ``(table pair, item)``, ``item`` its window vertex times three plus its edge."""
    live: wp.array[wp.int32]
    """The class pairs past the whole-pair rejects (class pair indices, in any order): the close's."""
    scans: wp.array[wp.vec2i]
    """The scan's jobs: ``(class pair, run)``, run ``k`` the window vertices ``[k, k + 1)`` times the class's
    :data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN` of the pair's window (room for every pair's
    runs at its largest window)."""
    recover: wp.array[wp.int32]
    """The closed pairs without a row, left to rule R (class pair indices, in any order)."""
    lanes: wp.array[wp.int32]
    """The closed pairs whose finish takes a block of lanes (:data:`TERRAIN_SOLID_LANE_ROWS` held rows or more; class
    pair indices, in any order)."""
    small: wp.array[wp.int32]
    """The closed pairs holding fewer than :data:`TERRAIN_SOLID_LANE_ROWS` rows with a plane group of three, whose finish
    takes the small-pair finish kernel, one thread per pair (class pair indices, in any order)."""
    counts: wp.array[wp.int32]
    """Entries asked for per job list (:data:`TERRAIN_SOLID_JOB_LISTS`), then the candidates asked for, the live pairs,
    the pairs left to rule R, the scan jobs, the descents and the deferred pairs walked again
    (:data:`TERRAIN_SOLID_COUNTERS`, zeroed before every launch; past a list's capacity, the entries that found no
    room)."""
    threads: int
    """Threads of the job kernels' launches: each strides over its list."""
    pair_threads: int
    """Threads of the launches that stride over lists of the class's pairs (the close and the exit kernel: the live
    pairs, the pairs left to rule X): the job threads, at most the class's pairs."""
    finish_blocks: int
    """Blocks of the launches that take a list of the class's pairs a pair per block at a time (``wp.launch_tiled``:
    rule R's kernel over the pairs without a row, the finish kernel over the lane list):
    :data:`TERRAIN_SOLID_FINISH_BLOCKS` on CUDA, the job threads on the CPU, at most the class's pairs."""
    q: wp.array[wp.vec3]
    """The candidate pool: terrain point of every candidate [m, window frame]. Its first entries hold the jobs' rows
    (:func:`_offer`), its last ``descent.shape[0]`` rule X's listed candidates (:func:`_x_list`), each region with its
    own counter and room."""
    n: wp.array[wp.vec3]
    """Unit normal of every candidate."""
    d: wp.array[float]
    """Core distance of every candidate [m]."""
    feature: wp.array[wp.int32]
    """Feature id of every candidate."""
    key: wp.array[wp.int32]
    """Place in window order of every candidate: its item's index times eight plus its point slot."""
    next: wp.array[wp.int32]
    """The next candidate of the same pair (-1: none)."""
    head: wp.array[wp.int32]
    """Per class pair: its first candidate (-1: none)."""
    target: wp.array[float]
    """Per class pair left to rule X: the depth its exit must pass (its deepest row's, plus
    :data:`TERRAIN_SOLID_EXIT_MARGIN`)."""
    descent: wp.array[wp.vec4]
    """Per listed rule X candidate ``k`` (pool entry :func:`_pool_rows` plus ``k``): its bound after each block of
    :func:`_near_bound`'s descent, the last repeated (:func:`_near_descent`), taken by :func:`solid_xdescent_job` so
    that the exit kernel's passes need not. Its length is the room of rule X's region of the pool."""
    xpair: wp.array[wp.int32]
    """Per listed rule X candidate ``k``: its class pair, whose window :func:`solid_xdescent_job` reads."""
    xhead: wp.array[wp.int32]
    """Per class pair left to rule X: its first listed candidate in the candidate pool (-1: none). Rule X's jobs list
    the candidates whose bounds pass the pair's target there: the candidate point in ``q``, its bound in ``d`` and its
    descent's in ``descent``, its key in ``key``, and for a valley's its valley's first key in ``feature`` (-1 for
    another's) and the valley's gate (its height over the core's lowest point, its arc's lower normal ``z``) in
    ``n``."""
    xall: wp.array[wp.int32]
    """Per class pair left to rule X: nonzero when some of its rule X work found no room, so that its exit kernel
    leaves it to the class's full-finish kernels, which take every candidate of its window."""
    xhot: wp.array[wp.int32]
    """Per class pair left to rule X: nonzero when an exit of its core can lie deeper than its target over some cell
    under its footprint (:func:`solid_xbound_job`; always for a class whose exits are not gated,
    :func:`~newton._src.geometry.terrain_law_cores.core_exits_gated`). Rule X's items of a pair without it list
    nothing."""
    rule_x: int
    """Nonzero when the law runs rule X (``NEWTON_TERRAIN_LAW_RULE_X``); else every closed pair is finished by its close
    or rule R's kernel, and rule X's kernels are not launched."""


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
    """Offer a row: hold it unless it coincides with a held row, as the finish merges
    (:func:`~newton._src.geometry.terrain_law.terrain_hold_row` on the sink's slots of ``store``).

    The capacity bounds the distinct rows, not the offers: a crease's interval ends shared by the next crease merge
    before they count (a pair keeps every distinct candidate its slots or its scratch list hold).
    """
    sink.raw = sink.raw + 1
    sink.held = terrain_hold_row(store, sink.slot, sink.held, q, n, d, feature, sink.eps, sink.capacity)
    return sink


@wp.func
def _pool_rows(jobs: TerrainSolidJobs) -> int:
    """The candidate pool's room for the jobs' rows: its entries but rule X's region (:attr:`TerrainSolidJobs.q`)."""
    return jobs.key.shape[0] - jobs.descent.shape[0]


@wp.func
def _offer(store: TerrainSolidJobs, sink: _Sink, q: wp.vec3, n: wp.vec3, d: float, feature: int) -> _Sink:
    """Offer a job's row: add it to the candidate pool, in its pair's list (``sink.slot``), with its place in window
    order (``sink.key`` plus its point slot); the close holds them in that order (:func:`solid_close_jobs`). With no
    room left, the row is lost and ``sink.lost`` set (the close then leaves the pair to the full-finish kernel)."""
    sink.raw = sink.raw + 1
    entry = wp.atomic_add(store.counts, _CANDIDATES, 1)
    if entry >= _pool_rows(store):
        sink.lost = 1
        return sink
    store.q[entry] = q
    store.n[entry] = n
    store.d[entry] = d
    store.feature[entry] = feature
    store.key[entry] = sink.key + (feature & 7)
    store.next[entry] = wp.atomic_exch(store.head, sink.slot, entry)
    return sink


# ----------------------------------------------------------------------------------------------------------------
# the rules
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _vertex_z(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int) -> float:
    return terrain_height(data, window.surface, r, c) - window.z_ref


@wp.func
def _face_top(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, sub: int) -> float:
    """The highest vertex of triangle ``sub`` of cell ``(r, c)`` [m, window frame]: a face whose highest vertex lies
    below the pair's ``low`` gives no row (rule F's cheapest reject)."""
    va, vb, vd = terrain_triangle_vertices(r, c, sub)
    return wp.max(
        _vertex_z(data, window, va[0], va[1]),
        wp.max(_vertex_z(data, window, vb[0], vb[1]), _vertex_z(data, window, vd[0], vd[1])),
    )


@wp.func
def _face_support(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    r: int,
    c: int,
    sub: int,
    cluster: int,
) -> tuple[TerrainPlanes, int, wp.vec3i, int, _Mat43, wp.vec4, wp.vec3]:
    """Rule F's manifold of the cluster plane of triangle ``sub`` of cell ``(r, c)`` (cluster id ``cluster``):
    ``(planes, supported, first, count, points, depth, n)``, ``supported`` the bits of the supported manifold points
    and ``first`` the triangle holding the first of them (``(-1, -1, -1)`` for none, or a plane beyond the reach).

    A function of the cluster alone: every triangle of a cluster evaluates its plane, manifold and support tests alike,
    so whichever of them computes it names the same emitting triangle and rows."""
    surface = window.surface
    first = wp.vec3i(-1, -1, -1)
    planes, n, a = terrain_plane(data, window, planes, r, c, sub)
    if core_sigma(core, n, a) > pair.reach:
        return planes, 0, first, 0, _Mat43(), wp.vec4(0.0), n
    count, points, depth = core_manifold(core, n, a, pair.reach)
    supported = int(0)
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
    return planes, supported, first, count, points, depth, n


@wp.func
def _face_offer(
    store: Any,
    sink: _Sink,
    supported: int,
    count: int,
    points: _Mat43,
    depth: wp.vec4,
    n: wp.vec3,
    emitter: wp.vec3i,
) -> _Sink:
    """Offer rule F's supported manifold points, emitted by triangle ``emitter``."""
    for j in range(count):
        if ((supported >> j) & 1) != 0:
            k = wp.vec3(points[j, 0], points[j, 1], points[j, 2])
            sink = _offer(
                store,
                sink,
                k - depth[j] * n,
                n,
                depth[j],
                terrain_feature_id(emitter[0], emitter[1], TERRAIN_FEATURE_FACE, emitter[2], j),
            )
    return sink


@wp.func
def _solid_face(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    emitters: _Emitters,
    pair: _Pair,
    sink: _Sink,
    store: Any,
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
    if _face_top(data, window, r, c, sub) < pair.low:
        # Never the emitter: a supported point's terrain point lies on the cluster plane over this triangle, at most
        # TERRAIN_FLAT_EPS above its highest vertex, and within the reach below the core's lowest point.
        return planes, emitters, sink
    planes, supported, first, count, points, depth, n = _face_support(
        core, data, window, planes, pair, r, c, sub, cluster
    )
    if cluster >= 0 and not cached:
        emitters = _emitter_store(emitters, cluster, first)
    if supported == 0 or first[0] != r or first[1] != c or first[2] != sub:
        return planes, emitters, sink
    sink = _face_offer(store, sink, supported, count, points, depth, n, first)
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

    The float64 law's samples and bracket, for the same answer where ``sigma`` has several maxima (a core through the
    crease's sides, ``sigma < 0``, can turn ``sigma`` convex): a scan of nine equally spaced normals, then a bisection
    on the sign of the derivative between the best one's neighbours (the float64 law: golden section on ``sigma``;
    here 22 steps, the same count for every lane; the lowest support point decides, so a kink is found where two
    support points cross), then the arc's ends and the best sample where they are higher. Within ``tie`` of the
    refined maximum (float32 cannot order such values: a maximum within about 1e-3 rad of an end), an end wins where
    ``sigma`` does not rise from it into the arc. Each support search starts where the last one ended
    (:func:`~newton._src.geometry.terrain_law_cores.core_sigma_near`): the normals move little from one to the next.
    """
    samples = _arc_samples(n_a, n_b)
    best = int(0)
    best_value = sigma_a
    for i in range(1, 9):
        value = sigma_b
        if i < 8:
            core, value = core_sigma_near(core, _arc_sample(samples, i), v0)
        if value > best_value:
            best = i
            best_value = value
    lo = _arc_sample(samples, wp.max(best - 1, 0))
    hi = _arc_sample(samples, wp.min(best + 1, 8))
    # the arc turns about this axis from n_a toward n_b: its tangent at n is axis x n
    axis = wp.normalize(wp.cross(n_a, n_b))
    for _step in range(_BISECTION):
        mid = wp.normalize(lo + hi)
        core, slope = core_slope_near(core, mid, wp.cross(axis, mid), v0, 1.0)
        if slope > 0.0:
            lo = mid
        else:
            hi = mid
    nst = wp.normalize(lo + hi)
    core, value = core_sigma_near(core, nst, v0)
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
def _edge_cheap(data: TerrainFeatureData, window: TerrainWindow, pair: _Pair, r: int, c: int, which: int) -> bool:
    """Rule E's cheapest rejects of edge ``which`` of vertex ``(r, c)``, from its vertices and the core's sphere alone:
    ``False`` for an edge that gives no row (not a crease, below the core, the sphere off the crease's segment)."""
    if not terrain_edge_open_crease(data, window, r, c, which):
        # flat inside one cluster, or on the border: no crease; or a valley in this window's frame by its static class,
        # which the crease tests reject (:func:`_edge_arc`)
        return False
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    top = wp.max(v0[2], v1[2])
    if top < pair.low:
        return False
    e = v1 - v0
    length2 = wp.dot(e, e)
    length = wp.sqrt(length2)
    dh = e / length
    toward = pair.sphere_c - v0
    # A row's line along its normal meets the core, so passes within the core's sphere: the sphere's centre lies
    # within its radius of the crease's segment along the crease, and across the line for some normal of the arc.
    along = wp.dot(toward, dh)
    if along < -pair.sphere_r - pair.eps or along > length + pair.sphere_r + pair.eps:
        return False
    if top <= pair.low + pair.reach:
        # Nowhere above the core's lowest point, the crease holds no row deeper than the reach (whose terrain point
        # lies above the core along its normal), so one out of reach holds none.
        closest = wp.clamp(wp.dot(toward, e) / length2, 0.0, 1.0)
        if wp.length(v0 + closest * e - pair.sphere_c) - pair.sphere_r > pair.reach:
            return False
    return True


@wp.func
def _edge_arc(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    r: int,
    c: int,
    which: int,
) -> tuple[TerrainPlanes, bool, wp.vec3, wp.vec3, float, float]:
    """Rule E's crease tests of edge ``which`` of vertex ``(r, c)`` past :func:`_edge_cheap`: ``(planes, ok, n_a,
    n_b, sigma_a, sigma_b)``, ``ok`` for a convex crease whose normal arc (``n_a`` to ``n_b``) can hold a row within
    reach (its ends' support values ``sigma_a`` and ``sigma_b``)."""
    v0i, v1i, ta, tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    e = v1 - v0
    length2 = wp.dot(e, e)
    length = wp.sqrt(length2)
    dh = e / length
    toward = pair.sphere_c - v0
    no = wp.vec3(0.0)
    planes, na, pa = terrain_plane(data, window, planes, ta[0], ta[1], ta[2])
    if terrain_crease_class(data, window, r, c, which, na, pa) == TERRAIN_EDGE_CONCAVE:
        return planes, False, no, no, 0.0, 0.0
    planes, nb, _pb = terrain_plane(data, window, planes, tb[0], tb[1], tb[2])
    # the normal arc, exactly perpendicular to the edge
    n_a = wp.normalize(na - wp.dot(na, dh) * dh)
    n_b = wp.normalize(nb - wp.dot(nb, dh) * dh)
    if wp.length(n_b - n_a) < 1.0e-9:
        return planes, False, no, no, 0.0, 0.0
    # The centre's offset across the line at normal n is dh . (n x (c - v0)), a sinusoid along the arc (less than a
    # half turn): beyond the radius at both ends with one sign, it is beyond it on the whole arc.
    across_a = wp.dot(dh, wp.cross(n_a, toward))
    across_b = wp.dot(dh, wp.cross(n_b, toward))
    if across_a * across_b > 0.0 and wp.min(wp.abs(across_a), wp.abs(across_b)) > pair.sphere_r + pair.eps:
        return planes, False, no, no, 0.0, 0.0
    # the core lies in its sphere: sigma(n) >= n . (c - v0) - radius (a bound cheaper than sigma)
    if wp.max(wp.dot(n_a, toward), wp.dot(n_b, toward)) - pair.sphere_r > pair.reach:
        return planes, False, no, no, 0.0, 0.0
    sigma_a = core_sigma(core, n_a, v0)
    sigma_b = core_sigma(core, n_b, v0)
    if wp.max(sigma_a, sigma_b) > pair.reach:
        return planes, False, no, no, 0.0, 0.0
    return planes, True, n_a, n_b, sigma_a, sigma_b


@wp.func
def _edge_reach(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    pair: _Pair,
    r: int,
    c: int,
    which: int,
    n_a: wp.vec3,
    n_b: wp.vec3,
) -> bool:
    """Rule E's pre-bounds of a crease past its arc tests (:func:`_edge_arc`, its normal arc ``n_a`` to ``n_b``), from
    closed-form lower bounds of the core's support values
    (:func:`~newton._src.geometry.terrain_law_cores.core_sigma_bound`: the values themselves for boxes, cylinders,
    cones and ellipsoids, a hull's bounding box's, so that no support search runs) before the ``theta*`` search:
    ``False`` for a crease that holds no row at any normal of its arc.

    Every row of a crease lies where a point of the core's support feature at ``theta*`` meets the crease's line, in
    the plane through the line and ``theta*`` (within ``tol``; a reference feature at an arc end straddles it):

    * along the crease, at ``x`` in ``[0, length]``, ``x`` the projection of core points on the edge's direction
      ``dh`` (an interval end, or a point between two): a core whose extent along ``dh`` misses the segment by more than
      the rounding of both holds none;
    * across, at a core point ``k`` with ``e2 . (k - v0)`` within ``tol`` of 0, ``e2 = n x dh`` the normal of that plane:
      every ``e2`` of the arc is a combination ``alpha e2_a + beta e2_b`` of its ends' with ``alpha, beta >= 0`` and
      ``alpha + beta >= 1``, so a core lying beyond ``tol`` on one side of both ends' planes lies beyond it on that side
      of every plane of the arc, and no point of it meets the line;
    * both at once: such a point lies over the segment, within the radius of the core's capsule
      (:func:`~newton._src.geometry.terrain_law_cores.core_capsule`) of a point of the capsule's axis over the segment
      widened by that radius; that clipped axis lying beyond ``tol`` plus the radius on one side of both ends' planes
      leaves no such point.

    Each test is exact: it rejects only creases that the ``theta*`` search and the crossing (:func:`_edge_rows_at`)
    would leave without a row, with every float32 rounding of both sides inside its margin.
    """
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    e = v1 - v0
    length = wp.sqrt(wp.dot(e, e))
    dh = e / length
    # float32 rounding of the support values, the crossing's coordinates and the arc's normals: far below 4 bands
    margin = pair.tol + 4.0 * pair.eps
    if core_sigma_bound(core, dh, v0) > length + margin or core_sigma_bound(core, -dh, v0) > margin:
        return False
    e_a = wp.cross(n_a, dh)
    e_b = wp.cross(n_b, dh)
    if core_sigma_bound(core, e_a, v0) > margin and core_sigma_bound(core, e_b, v0) > margin:
        return False
    if core_sigma_bound(core, -e_a, v0) > margin and core_sigma_bound(core, -e_b, v0) > margin:
        return False
    p0, p1, radius = core_capsule(core)
    x0 = wp.dot(p0 - v0, dh)
    x1 = wp.dot(p1 - v0, dh)
    lo = -radius - margin
    hi = length + radius + margin
    t0 = float(0.0)
    t1 = float(1.0)
    run = x1 - x0
    if run > 0.0:
        t0 = wp.max(t0, (lo - x0) / run)
        t1 = wp.min(t1, (hi - x0) / run)
    elif run < 0.0:
        t0 = wp.max(t0, (hi - x0) / run)
        t1 = wp.min(t1, (lo - x0) / run)
    elif x0 < lo or x0 > hi:
        return False
    if t0 > t1:
        return False
    q0 = p0 + t0 * (p1 - p0) - v0
    q1 = p0 + t1 * (p1 - p0) - v0
    bound = margin + radius
    ya0 = wp.dot(q0, e_a)
    ya1 = wp.dot(q1, e_a)
    yb0 = wp.dot(q0, e_b)
    yb1 = wp.dot(q1, e_b)
    if wp.min(ya0, ya1) > bound and wp.min(yb0, yb1) > bound:
        return False
    if wp.max(ya0, ya1) < -bound and wp.max(yb0, yb1) < -bound:
        return False
    return True


@wp.func
def _edge_rows(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: Any,
    r: int,
    c: int,
    which: int,
    n_a: wp.vec3,
    n_b: wp.vec3,
    sigma_a: float,
    sigma_b: float,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule E's rows of a crease past :func:`_edge_arc` (its normal arc ``n_a`` to ``n_b``, its ends' support values
    ``sigma_a`` and ``sigma_b``): ``theta*`` (:func:`_edge_theta`) and the rows at it (:func:`_edge_rows_at`)."""
    nst, kind = _edge_theta(core, data, window, pair, r, c, which, n_a, n_b, sigma_a, sigma_b)
    if kind < 0:
        return planes, sink
    return _edge_rows_at(core, data, window, planes, pair, sink, store, r, c, which, nst, kind)


@wp.func
def _edge_theta(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    pair: _Pair,
    r: int,
    c: int,
    which: int,
    n_a: wp.vec3,
    n_b: wp.vec3,
    sigma_a: float,
    sigma_b: float,
) -> tuple[wp.vec3, int]:
    """A crease's ``theta*`` (:func:`_theta_star`): ``(n*, kind)``, ``kind`` -1 when the core's support there lies
    beyond the reach (no row), 1 inside the arc, 0 at an arc end (within :data:`TERRAIN_SOLID_ARC_END` of one: its
    reference feature, G1)."""
    v0i, _v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    nst, sigma, end = _theta_star(core, n_a, n_b, v0, sigma_a, sigma_b, pair.eps)
    if sigma > pair.reach:
        return nst, -1
    if end == 0 and wp.length(nst - n_a) > _ARC_END and wp.length(nst - n_b) > _ARC_END:
        return nst, 1
    return nst, 0


@wp.func
def _edge_rows_at(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: Any,
    r: int,
    c: int,
    which: int,
    nst: wp.vec3,
    kind: int,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule E's rows of a crease at its ``theta*`` ``nst`` (``kind`` from :func:`_edge_theta`): the crossing of the
    core's flat support feature (inside the arc) or reference feature (an arc end) with the crease's line, and its
    ends lifted onto the core."""
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    e = v1 - v0
    length2 = wp.dot(e, e)
    length = wp.sqrt(length2)
    dh = e / length
    interior = kind == 1
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
            if terrain_vertex_feature(data, window, v0i[0], v0i[1]):
                continue
        if x >= length:
            if terrain_vertex_feature(data, window, v1i[0], v1i[1]):
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
def _solid_edge(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: Any,
    r: int,
    c: int,
    which: int,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule E for edge ``which`` of vertex ``(r, c)``: a convex crease's rows where the core's flat support feature
    (interior ``theta*``) or reference feature (arc end) crosses the crease's line."""
    if not _edge_cheap(data, window, pair, r, c, which):
        return planes, sink
    planes, ok, n_a, n_b, sigma_a, sigma_b = _edge_arc(core, data, window, planes, pair, r, c, which)
    if not ok or not _edge_reach(core, data, window, pair, r, c, which, n_a, n_b):
        return planes, sink
    return _edge_rows(core, data, window, planes, pair, sink, store, r, c, which, n_a, n_b, sigma_a, sigma_b)


@wp.func
def _vertex_cheap(core: Any, data: TerrainFeatureData, window: TerrainWindow, pair: _Pair, r: int, c: int):
    """Rule V's rejects of vertex ``(r, c)``: ``(core, listed)``, ``listed`` ``False`` for a vertex that gives no row
    (not a vertex feature, from its bits: on the border or with incident normals that do not span 3-D; below the core;
    out of the reach of its sphere or of its support plane facing the vertex). The core keeps its support search's end
    for the next vertex's (:func:`~newton._src.geometry.terrain_law_cores.core_sigma_near`: a run's vertices lie near
    each other)."""
    if not terrain_vertex_feature(data, window, r, c):
        return core, False
    v = terrain_window_vertex(data, window, r, c)
    if v[2] < pair.low:
        return core, False
    toward = pair.sphere_c - v
    distance = wp.length(toward)
    if distance - pair.sphere_r > pair.reach:
        return core, False
    if distance > 0.0:
        core, sigma = core_sigma_near(core, toward / distance, v)
        if sigma > pair.reach:
            # the core's support plane facing the vertex lies beyond the reach, and so does the core
            return core, False
    return core, True


@wp.func
def _vertex_rows(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: Any,
    r: int,
    c: int,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule V's row of a vertex past :func:`_vertex_cheap` (a vertex feature): its normal cone holds the direction to
    the core's nearest point (or its minimum translation), within reach."""
    v = terrain_window_vertex(data, window, r, c)
    ok, nst, _k, d = core_nearest(core, v)
    if not ok or d > pair.reach:
        return planes, sink
    if not terrain_vertex_in_cone(data, window, r, c, nst):
        return planes, sink
    sink = _offer(store, sink, v, nst, d, terrain_feature_id(r, c, TERRAIN_FEATURE_VERTEX, 0, 0))
    return planes, sink


@wp.func
def _solid_vertex(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    sink: _Sink,
    store: Any,
    r: int,
    c: int,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule V for vertex ``(r, c)``: a vertex feature whose normal cone holds the direction to the core's nearest
    point (or its minimum translation) emits a row within reach."""
    _core, listed = _vertex_cheap(core, data, window, pair, r, c)
    if not listed:
        return planes, sink
    return _vertex_rows(core, data, window, planes, pair, sink, store, r, c)


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
    lane: int,
    lanes: int,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule R (G3): up to four of the core's boundary points below the surface, deepest first (then in order), each
    against the face vertically above it with that plane's normal and depth, by a block's ``lanes`` lanes (lane
    ``lane``; one thread passes ``0, 1``).

    Each lane takes every ``lanes``-th point and keeps its first four in (depth, index) order, each point's depth taken
    once; four block minima then pick the pair's first four in that order, and the first lane offers their rows. A
    row's feature id takes the pick's rank (0-3) as its point slot, unique within the pair: a boundary point's index (up
    to 32 for a cylinder) would wrap in the slot's three bits.
    """
    depth = wp.vec4(0.0)
    index = wp.vec4i(-1)
    kept = int(0)
    for i in range(lane, core_recovery_count(core), lanes):
        x = core_recovery_point(core, i)
        t = terrain_locate(window, x[0], x[1])
        if t[0] < 0:
            continue
        planes, n, a = terrain_plane(data, window, planes, t[0], t[1], t[2])
        d = wp.dot(n, x - a)
        if d >= 0.0:
            continue
        # its place in (depth, index) order: after every kept point no deeper (each has a smaller index)
        k = kept
        while k > 0 and d < depth[k - 1]:
            k -= 1
        if k >= 4:
            continue
        j = wp.min(kept, 3)
        while j > k:
            depth[j] = depth[j - 1]
            index[j] = index[j - 1]
            j -= 1
        depth[k] = d
        index[k] = i
        kept = wp.min(kept + 1, 4)
    if lanes > 1:
        # merge the lanes' lists: each round's least depth, then the least index at it (each index is one lane's)
        mine = int(0)
        picked = wp.vec4(0.0)
        picks = wp.vec4i(-1)
        count = int(0)
        for _round in range(4):
            d = float(_HUGE)
            i = int(_NO_POINT)
            if mine < kept:
                d = depth[mine]
                i = index[mine]
            least = wp.tile_extract(wp.tile_min(wp.tile(d)), 0)
            first = wp.tile_extract(wp.tile_min(wp.tile(wp.where(d == least, i, _NO_POINT))), 0)
            if first != _NO_POINT:
                if i == first:
                    mine += 1
                picked[count] = least
                picks[count] = first
                count += 1
        depth = picked
        index = picks
        kept = count
    if lane != 0:
        return planes, sink
    for pick in range(kept):
        best_d = depth[pick]
        x = core_recovery_point(core, index[pick])
        t = terrain_locate(window, x[0], x[1])
        planes, n, a = terrain_plane(data, window, planes, t[0], t[1], t[2])
        sink = _offer(
            store, sink, x - best_d * n, n, best_d, terrain_feature_id(t[0], t[1], TERRAIN_FEATURE_RECOVERY, t[2], pick)
        )
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
def _near_block(data: TerrainFeatureData, window: TerrainWindow, x: wp.vec3, r: int, c: int) -> tuple[float, wp.vec3i]:
    """``x``'s distance to the triangles of the 3 x 3 cells around cell ``(r, c)`` (within the heightfield), and the
    nearest triangle."""
    surface = window.surface
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
    return nearest, near


@wp.func
def _near_next(data: TerrainFeatureData, window: TerrainWindow, x: wp.vec3, near: wp.vec3i) -> wp.vec3i:
    """The cell under ``x``'s projection onto the plane of triangle ``near`` (``(-1, -1, -1)`` off the
    heightfield): the next block's centre."""
    va, vb, vd = terrain_triangle_vertices(near[0], near[1], near[2])
    a = terrain_window_vertex(data, window, va[0], va[1])
    n = wp.normalize(
        wp.cross(
            terrain_window_vertex(data, window, vb[0], vb[1]) - a,
            terrain_window_vertex(data, window, vd[0], vd[1]) - a,
        )
    )
    q = x - wp.dot(x - a, n) * n
    return terrain_locate(window, q[0], q[1])


@wp.func
def _near_bound(data: TerrainFeatureData, window: TerrainWindow, x: wp.vec3, bound: float, stop: float) -> float:
    """``bound`` lowered to ``x``'s distance to the triangles of the 3 x 3 cells around the one under it, then around
    the cell under ``x``'s projection onto the plane of the block's nearest triangle, while that cell moves and the
    bound exceeds ``stop`` (at most :data:`TERRAIN_SOLID_EXIT_DESCENT` blocks, within the heightfield): a descent
    toward ``x``'s closest surface point, whose distance every block's is at least (on a planar stretch the second
    block holds it). The second stage of rule X's bound, before an exit search; it decides which exits are searched,
    never the exit kept."""
    t = terrain_locate(window, x[0], x[1])
    if t[0] < 0:
        return bound
    r = t[0]
    c = t[1]
    for _step in range(_DESCENT):
        nearest, near = _near_block(data, window, x, r, c)
        bound = wp.min(bound, nearest)
        if bound <= stop:
            break
        s = _near_next(data, window, x, near)
        if s[0] < 0 or (s[0] == r and s[1] == c):
            break
        r = s[0]
        c = s[1]
    return bound


@wp.func
def _near_descent(data: TerrainFeatureData, window: TerrainWindow, x: wp.vec3, bound: float) -> wp.vec4:
    """:func:`_near_bound`'s bound after each block of its whole descent, the last repeated: its bound at any stop
    is the first of them no greater than the stop, else the last (:func:`_near_stop`), bit for bit."""
    steps = wp.vec4(bound)
    t = terrain_locate(window, x[0], x[1])
    if t[0] < 0:
        return steps
    r = t[0]
    c = t[1]
    for step in range(_DESCENT):
        nearest, near = _near_block(data, window, x, r, c)
        bound = wp.min(bound, nearest)
        for k in range(step, _DESCENT):
            steps[k] = bound
        s = _near_next(data, window, x, near)
        if s[0] < 0 or (s[0] == r and s[1] == c):
            break
        r = s[0]
        c = s[1]
    return steps


@wp.func
def _near_stop(steps: wp.vec4, stop: float) -> float:
    """A listed candidate's bound lowered by :func:`_near_bound` at ``stop``, from its descent's bounds
    (:func:`_near_descent`)."""
    for k in range(_DESCENT):
        if steps[k] <= stop:
            return steps[k]
    return steps[_DESCENT - 1]


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
    it from earlier in window order (an exit deeper than the target: only such a one is offered)."""
    if best.walk == 0:
        return bound > wp.max(best.target, best.second) * scale
    if key == best.first_key:
        return False
    limit = best.threshold * scale
    tie = bound == limit and best.found != 0 and key < best.key and best.threshold > best.target
    return bound > limit or tie


@wp.func
def _exit_stop(best: _Exit) -> float:
    """The bound a candidate's must pass at the pass's state: in the first pass the target or the second largest
    bound, in the second the threshold."""
    if best.walk == 0:
        return wp.max(best.target, best.second)
    return best.threshold


@wp.func
def _exit_lowered(
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
    """Take candidate ``x`` of bound ``bound`` (key ``key``), lowered by :func:`_near_bound`, while it can matter
    (:func:`_exit_open`): in the first pass it joins the two largest bounds (the first in window order on ties), in
    the second its exit is searched (:func:`_exit_take`)."""
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
    :func:`_near_bound`, can matter (:func:`_exit_open`, :func:`_exit_lowered`)."""
    if not _exit_open(best, key, bound, 1.0):
        return planes, best
    bound = _near_bound(data, window, x, bound, _exit_stop(best))
    return _exit_lowered(data, window, planes, pair, cells, best, x, bound, key)


@wp.func
def _exit_try_listed(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    cells: wp.vec4i,
    best: _Exit,
    x: wp.vec3,
    bound: float,
    steps: wp.vec4,
    key: int,
) -> tuple[TerrainPlanes, _Exit]:
    """:func:`_exit_try` for a listed candidate, its descent's bounds ``steps`` taken by its job
    (:func:`_near_descent`, :func:`_near_stop`): the same bounds, bit for bit."""
    if not _exit_open(best, key, bound, 1.0):
        return planes, best
    bound = _near_stop(steps, _exit_stop(best))
    return _exit_lowered(data, window, planes, pair, cells, best, x, bound, key)


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
    refinement ends higher. The float64 law (:func:`newton.tests.terrain_law_reference._arc_angles`) brackets the same
    samples but refines by golden section on ``sigma`` (60 steps): both find a bracket's minimum when it holds one, and
    could stop at different ones where ``sigma`` has two within an eighth of the arc of a sample. Skipped where no core
    point in the wedge can matter (:func:`_exit_open`): within it a point's distance to the edge is at most its height
    under the edge's higher end over the arc's smaller normal ``z`` (``low`` is the core's lowest height).
    """
    planes, ok, v0, e, length2, n_a, n_b, top = _valley_arc(data, window, planes, low, r, c, which)
    if not ok:
        return planes, best
    if not _exit_open(best, key, top - low, wp.min(n_a[2], n_b[2])):
        return planes, best
    samples, values, axis = _valley_samples(core, v0, n_a, n_b)
    for i in range(9):
        if not _valley_minimum(values, i):
            continue
        nst = _valley_refine(core, samples, values, axis, v0, i)
        for j in range(core_support_count(core)):
            x = core_support_point(core, nst, j)
            if x[2] >= top:
                # closest to the valley only below its line
                continue
            planes, bound = _valley_bound(data, window, planes, x, v0, e, length2)
            planes, best = _exit_try(data, window, planes, pair, cells, best, x, bound, key + 2 * i + j)
    return planes, best


@wp.func
def _valley_arc(
    data: TerrainFeatureData, window: TerrainWindow, planes: TerrainPlanes, low: float, r: int, c: int, which: int
) -> tuple[TerrainPlanes, bool, wp.vec3, wp.vec3, float, wp.vec3, wp.vec3, float]:
    """Rule X at edge ``which`` of vertex ``(r, c)``, before its gate: ``(planes, ok, v0, e, length2, n_a, n_b, top)``,
    ``ok`` for a valley (concave crease) rising above the core's lowest height ``low``, its lower vertex ``v0``, its
    vector ``e`` (squared length ``length2``), its normal arc ``n_a`` to ``n_b`` and its higher end's height ``top``."""
    no = wp.vec3(0.0)
    if not terrain_edge_open_valley(data, window, r, c, which):
        # flat, on the border, or a ridge in this window's frame by its static class: no valley
        return planes, False, no, no, 0.0, no, no, 0.0
    v0i, v1i, ta, tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    top = wp.max(v0[2], v1[2])
    if top <= low:
        return planes, False, no, no, 0.0, no, no, 0.0
    planes, na, pa = terrain_plane(data, window, planes, ta[0], ta[1], ta[2])
    if terrain_crease_class(data, window, r, c, which, na, pa) != TERRAIN_EDGE_CONCAVE:
        return planes, False, no, no, 0.0, no, no, 0.0
    planes, nb, _pb = terrain_plane(data, window, planes, tb[0], tb[1], tb[2])
    e = v1 - v0
    length2 = wp.dot(e, e)
    dh = e / wp.sqrt(length2)
    # the valley's normal arc, exactly perpendicular to the edge
    n_a = wp.normalize(na - wp.dot(na, dh) * dh)
    n_b = wp.normalize(nb - wp.dot(nb, dh) * dh)
    if wp.length(n_b - n_a) < 1.0e-9:
        return planes, False, no, no, 0.0, no, no, 0.0
    return planes, True, v0, e, length2, n_a, n_b, top


@wp.func
def _valley_samples(core: Any, v0: wp.vec3, n_a: wp.vec3, n_b: wp.vec3) -> tuple[_ArcNormals, _ArcValues, wp.vec3]:
    """A valley's nine equally spaced normals, ``sigma`` at each, and the axis its arc turns about."""
    samples = _arc_samples(n_a, n_b)
    values = _ArcValues()
    for i in range(9):
        values[i] = core_sigma(core, _arc_sample(samples, i), v0)
    axis = wp.normalize(wp.cross(n_a, n_b))
    return samples, values, axis


@wp.func
def _valley_minimum(values: _ArcValues, i: int) -> bool:
    """Whether sample ``i`` is no higher than its neighbours (the first of a run of equal ones)."""
    return not ((i > 0 and values[i] >= values[i - 1]) or (i < 8 and values[i] > values[i + 1]))


@wp.func
def _valley_refine(core: Any, samples: _ArcNormals, values: _ArcValues, axis: wp.vec3, v0: wp.vec3, i: int) -> wp.vec3:
    """Sample ``i``'s local minimum of ``sigma`` refined between its neighbours by bisection on the sign of the
    derivative (rule E's 22 steps), the sample kept where the refinement ends higher."""
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
    return nst


@wp.func
def _valley_bound(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    x: wp.vec3,
    v0: wp.vec3,
    e: wp.vec3,
    length2: float,
) -> tuple[TerrainPlanes, float]:
    """A valley candidate's bound: its exit bound, or its distance to the valley's edge (from ``v0`` along ``e``),
    the smaller."""
    planes, bound = _exit_bound(data, window, planes, x)
    if bound > 0.0:
        s = wp.clamp(wp.dot(x - v0, e) / length2, 0.0, 1.0)
        bound = wp.min(bound, wp.length(x - (v0 + s * e)))
    return planes, bound


@wp.func
def _x_list(
    jobs: TerrainSolidJobs, i: int, x: wp.vec3, bound: float, key: int, valley: int, gate: float, scale: float
) -> int:
    """List a rule X candidate of class pair ``i`` whose bound passes its target (:attr:`TerrainSolidJobs.xhead`), in
    rule X's region of the candidate pool: point ``x``, bound ``bound``, key ``key``; for a valley's, its valley's
    first key ``valley`` and the valley's gate (``gate`` over ``scale``; ``valley`` -1 for another's); and its pair for
    its descent (:func:`solid_xdescent_job`). 0 when the region had no room."""
    k = wp.atomic_add(jobs.counts, _XDESCENTS, 1)
    if k >= jobs.descent.shape[0]:
        return 0
    entry = _pool_rows(jobs) + k
    jobs.xpair[k] = i
    jobs.q[entry] = x
    jobs.d[entry] = bound
    jobs.n[entry] = wp.vec3(gate, scale, 0.0)
    jobs.feature[entry] = valley
    jobs.key[entry] = key
    jobs.next[entry] = wp.atomic_exch(jobs.xhead, i, entry)
    return 1


@wp.struct
class _EveryCandidate:
    """Rule X's passes over every candidate of the window (:func:`_exit_candidates`): the full-finish kernel's."""

    low: float
    """The core's lowest point [m, window frame]."""


@wp.struct
class _ListedCandidates:
    """Rule X's passes over class pair ``row``'s listed candidates (:func:`_exit_listed`): the exit kernel's."""

    row: int


@wp.func
def _exit_listed(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    cells: wp.vec4i,
    best: _Exit,
    jobs: TerrainSolidJobs,
    how: _ListedCandidates,
) -> tuple[TerrainPlanes, _Exit]:
    """One pass of rule X over class pair ``how.row``'s listed candidates (:attr:`TerrainSolidJobs.xhead`) in key
    order, as :func:`_exit_candidates` takes them: a valley's gate (:func:`_exit_open`) at its first listed candidate,
    which takes or skips the valley's candidates. A candidate left off the list has a bound no higher than the target,
    and a valley off it a gate closed at the target and at every later threshold: both are no-ops in either pass, and
    the state at a listed candidate is the full pass's. The list is taken in chunks of :data:`TERRAIN_SOLID_SORTED` keys
    sorted in registers, one walk of it per chunk (:data:`TERRAIN_SOLID_SORTED`): one walk per pass on the recorded
    states."""
    keys = _Sorted()
    entries = _Sorted()
    last = int(-1)
    valley = int(-1)
    passing = bool(True)
    more = bool(True)
    while more:
        count = int(0)
        more = bool(False)
        e = jobs.xhead[how.row]
        while e >= 0:
            k = jobs.key[e]
            if k > last:
                if count < _SORTED or k < keys[_SORTED - 1]:
                    more = more or count == _SORTED
                    j = wp.min(count, _SORTED - 1)
                    while j > 0 and keys[j - 1] > k:
                        keys[j] = keys[j - 1]
                        entries[j] = entries[j - 1]
                        j -= 1
                    keys[j] = k
                    entries[j] = e
                    count = wp.min(count + 1, _SORTED)
                else:
                    more = True
            e = jobs.next[e]
        for j in range(count):
            key = keys[j]
            if key == last:
                continue
            last = key
            entry = entries[j]
            first = jobs.feature[entry]
            if first >= 0:
                if first != valley:
                    valley = first
                    gate = jobs.n[entry]
                    passing = _exit_open(best, first, gate[0], gate[1])
                if not passing:
                    continue
            planes, best = _exit_try_listed(
                data,
                window,
                planes,
                pair,
                cells,
                best,
                jobs.q[entry],
                jobs.d[entry],
                jobs.descent[entry - _pool_rows(jobs)],
                key,
            )
    return planes, best


@wp.func
def _exit_pass(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    cells: wp.vec4i,
    best: _Exit,
    jobs: TerrainSolidJobs,
    how: _ListedCandidates,
) -> tuple[TerrainPlanes, _Exit]:
    """One pass over class pair ``how.row``'s listed candidates (:func:`_exit_listed`)."""
    return _exit_listed(data, window, planes, pair, cells, best, jobs, how)


@wp.func
def _exit_pass(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    pair: _Pair,
    cells: wp.vec4i,
    best: _Exit,
    jobs: TerrainSolidJobs,
    how: _EveryCandidate,
) -> tuple[TerrainPlanes, _Exit]:
    """One pass over every candidate of the window (:func:`_exit_candidates`)."""
    return _exit_candidates(core, data, window, planes, pair, cells, best, how.low)


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
    jobs: TerrainSolidJobs,
    how: Any,
) -> tuple[TerrainPlanes, _Sink]:
    """Rule X: the exit of the deepest of the core's candidate boundary points, offered when it lies deeper than every
    row the pair holds by more than :data:`TERRAIN_SOLID_EXIT_MARGIN` (point slot :data:`TERRAIN_SOLID_EXIT_SLOT`);
    its exit searches and the window vertices they covered are added to ``work``. ``how`` picks the passes' candidates
    at compile time: every candidate of the window (:class:`_EveryCandidate`, the full-finish kernel's) or class pair
    ``row``'s listed ones (:class:`_ListedCandidates`, the exit kernel's: rule X's items and valleys,
    :func:`solid_xitem_job`, those whose bounds pass the target; any other is a no-op in both passes, so the rows are
    those of the passes over every candidate).

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
    its cost depends on the candidate's depth, not the window's size.
    """
    deepest = float(0.0)
    for h in range(wp.min(sink.held, sink.capacity)):
        deepest = wp.max(deepest, -store.d[sink.slot + h])
    best = _Exit()
    best.target = deepest + _EXIT_MARGIN
    best.threshold = best.target
    best.first = best.target
    best.second = best.target
    best.first_key = -1
    planes, best = _exit_pass(core, data, window, planes, pair, cells, best, jobs, how)
    if best.first > best.target:
        planes, best = _exit_take(data, window, planes, pair, cells, best, best.first_x, best.first, best.first_key)
        # The second pass only when another candidate's bound can beat the exit found, or tie it from earlier in
        # window order (every other candidate of the largest bound comes later; an exit no deeper than the target is
        # never offered).
        beat = best.second > best.threshold
        tie = best.second == best.threshold and best.threshold < best.first and best.threshold > best.target
        if beat or tie:
            best.walk = 1
            planes, best = _exit_pass(core, data, window, planes, pair, cells, best, jobs, how)
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
def _vertex_quiet(data: TerrainFeatureData, surface: TerrainSurfaceInfo, r: int, c: int) -> bool:
    """Whether window vertex ``(r, c)`` lists nothing for rules E and V in any window: its three edges are flat or on
    the border (two-bit codes 1 or 2 of its bits, whose bits then differ) and it is no vertex feature."""
    bits = terrain_vertex_bits(data, surface, r, c)
    return ((bits ^ (bits >> 1)) & 21) == 21 and (bits & TERRAIN_VERTEX_FEATURE_BIT) == 0


@wp.func
def _window_extent_top(
    data: TerrainFeatureData, window: TerrainWindow, r1: int, c1: int, extent: float
) -> tuple[float, float, bool]:
    """:func:`~newton._src.geometry.terrain_law.terrain_window_extent` (the same value, bit for bit) and, read in the
    same pass, the window's highest vertex (window frame), which rule X's gate takes, and whether every window vertex
    is quiet (:func:`_vertex_quiet`, read until one is not)."""
    surface = window.surface
    extent = wp.max(extent, wp.max(float(c1 + 1 - window.c0) * surface.dx, float(r1 + 1 - window.r0) * surface.dy))
    top = float(-_HUGE)
    quiet = bool(True)
    for r in range(window.r0, r1 + 2):
        for c in range(window.c0, c1 + 2):
            z = terrain_height(data, surface, r, c) - window.z_ref
            extent = wp.max(extent, wp.abs(z))
            top = wp.max(top, z)
            if quiet:
                quiet = _vertex_quiet(data, surface, r, c)
    return extent, top, quiet


@wp.func
def solid_pair_cells(core: Any, data: TerrainFeatureData, heightfield: int, reach: float):
    """The pair's window cells and its whole-window rejects: ``(state, cells, lo, hi)``, the core's bounds
    (heightfield frame) and ``state`` nonzero when the window misses the heightfield or the tile-max exit applies."""
    surface = data.surfaces[heightfield]
    lo, hi = core_bounds(core)
    cells = terrain_box_cells(surface, lo[0] - reach, lo[1] - reach, hi[0] + reach, hi[1] + reach)
    if cells[1] < cells[0] or cells[3] < cells[2]:
        return TERRAIN_PAIR_OUTSIDE, cells, lo, hi
    # The lowest point minus the reach, lowered by its own float32 rounding and by the cluster tolerance (a face row's
    # terrain point lies on its cluster plane, up to TERRAIN_FLAT_EPS above its triangle's highest vertex): the exit
    # drops no row but one whose terrain point rounding carries off its triangle on a steep face, and then only within
    # that of the reach limit.
    low = lo[2] - reach
    low = low - 4.0 * _EPS32 * (wp.abs(lo[2]) + reach) - _FLAT
    if terrain_tile_exit(data, surface, cells[0], cells[1], cells[2], cells[3], low):
        return TERRAIN_PAIR_TILE_EXIT, cells, lo, hi
    return 0, cells, lo, hi


@wp.func
def solid_pair_frame(
    core: Any, data: TerrainFeatureData, heightfield: int, cells: wp.vec4i, lo: wp.vec3, hi: wp.vec3, reach: float
):
    """A pair's window frame past its rejects (:func:`solid_pair_cells`, its cells and the core's bounds ``lo`` and
    ``hi``): ``(window, core, pair, quiet)`` with the core moved into the window frame, ``quiet`` when every window
    vertex is quiet (:func:`_vertex_quiet`)."""
    pair = _Pair()
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
    extent, top, quiet = _window_extent_top(data, window, cells[1], cells[3], extent)
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
    return window, core, pair, quiet


@wp.func
def solid_pair_window(core: Any, data: TerrainFeatureData, heightfield: int, reach: float):
    """The pair's window: ``(state, cells, window, core, pair)`` with the core moved into the window frame.

    ``state`` is nonzero when the window misses the heightfield or the tile-max exit applies.
    """
    state, cells, lo, hi = solid_pair_cells(core, data, heightfield, reach)
    if state != 0:
        return state, cells, TerrainWindow(), core, _Pair()
    window, core, pair, _quiet = solid_pair_frame(core, data, heightfield, cells, lo, hi, reach)
    return 0, cells, window, core, pair


@wp.func
def _wide(cells: wp.vec4i) -> int:
    if cells[1] - cells[0] + 1 >= _WIDE or cells[3] - cells[2] + 1 >= _WIDE:
        return TERRAIN_PAIR_WIDE
    return 0


@wp.func
def solid_sink(slot: int, capacity: int, eps: float) -> _Sink:
    sink = _Sink()
    sink.slot = slot
    sink.capacity = capacity
    sink.eps = eps
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
    """Walk the window's features in window order (vertex by vertex, its items in order): the full finish's walk of a
    deferred pair, in one thread."""
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
    slot: int,
    held: int,
    capacity: int,
    out: TerrainLawRows,
    pair_index: int,
    partner_class: int,
    frame: wp.transform,
    cells: wp.vec4i,
    lane: int,
    lanes: int,
):
    """Finish a pair's ``held`` rows in its walk slots from ``slot`` (``capacity`` of them) with a block's ``lanes``
    lanes (lane ``lane``; one thread passes ``0, 1``): defer it when its distinct rows overflow its slots, else run the
    finish in place (:func:`~newton._src.geometry.terrain_law.terrain_finish_slots`) and write the kept rows (world
    frame) at the front of its slots, the first lane writing once every lane has read them."""
    if held > capacity:
        if lane == 0:
            terrain_defer_pair(out, partner_class, pair_index, _wide(cells))
        return
    # the held rows are distinct: every one was held with terrain_hold_row at the pair's eps
    keep, kept = terrain_finish_slots(out.q, out.n, out.d, slot, held, pair.eps, lane, lanes, True)
    if lanes > 1:
        # the lanes agree on ``kept``; the reduction makes every lane's reads of the rows precede the writes below
        kept = wp.tile_extract(wp.tile_max(wp.tile(kept)), 0)
    if lane != 0:
        return
    written = int(0)
    for i in range(held):
        if ((keep >> i) & 1) != 0:
            s = slot + i
            terrain_store_row(out, slot + written, out.q[s], out.n[s], out.d[s], out.feature[s], window, frame)
            written += 1
    out.first[pair_index] = slot
    out.count[pair_index] = kept
    out.state[pair_index] = _wide(cells)


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
    jobs: TerrainSolidJobs,
    pair_index: int,
    slot: int,
    walk_slots: int,
    frame: wp.transform,
    cells: wp.vec4i,
):
    """Finish a deferred pair walked into a full-finish block's scratch (``sink.slot``): add the recovery rows when it
    has none and, when the law runs rule X (``jobs.rule_x``), its exit over every candidate (its searches counted in
    ``work``), run the finish on every distinct candidate and write the kept rows (world frame) to its walk slots from
    ``slot`` when they fit, else to rows of the spill region, or, when that has no room left, the first of them by
    priority to its walk slots (flagged :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_ROW_OVERFLOW`).

    The scratch holds every candidate a window of the class can give; a pair with more has a window its shape data no
    longer bounds (changed since construction): it keeps the first in window order and is flagged."""
    store = solid_scratch_store(scratch)
    if sink.raw == 0:
        planes, sink = solid_recovery(core, data, window, planes, sink, store, 0, 1)
    # Rule X can give a row only to a core whose lowest point lies below its window's highest vertex.
    lo, _hi = core_bounds(core)
    if jobs.rule_x != 0 and lo[2] < pair.top:
        every = _EveryCandidate()
        every.low = lo[2]
        planes, sink = solid_exits(core, data, window, planes, pair, sink, store, cells, work, jobs, every)
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
def solid_rows_full(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    scratch: TerrainFinishScratch,
    solids: TerrainSolidData,
    jobs: TerrainSolidJobs,
    block: int,
    pair_index: int,
    slot: int,
    walk_slots: int,
    frame: wp.transform,
    heightfield: int,
    reach: float,
):
    """A deferred pair some of whose work found no room (``core`` in the heightfield frame), walked again into block
    ``block``'s scratch (counted in its class's walks), finished and written (:func:`solid_close_full`; its walk slots
    from ``slot``)."""
    wp.atomic_add(jobs.counts, _WALKS, 1)
    _state, cells, window, core, pair = solid_pair_window(core, data, heightfield, reach)
    planes = terrain_planes()
    sink = solid_sink(block * scratch.capacity, scratch.capacity, pair.eps)
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
        jobs,
        pair_index,
        slot,
        walk_slots,
        frame,
        cells,
    )


@wp.func
def solid_pooled(
    data: TerrainFeatureData,
    scratch: TerrainFinishScratch,
    jobs: TerrainSolidJobs,
    block: int,
    i: int,
) -> tuple[bool, TerrainWindow, _Pair, wp.vec4i, _Sink]:
    """A deferred pair every job of which found room (class pair ``i``: more distinct rows than its walk slots, or
    rule X's lists without room), its candidates held into block ``block``'s scratch from the candidate pool in window
    order as the walk offers them (:func:`_hold_pooled`, the walk's held rows bit for bit), not by a second walk of its
    window: ``(pooled, window, pair, cells, sink)`` for :func:`solid_close_full` (its kept core is the scan's, window
    frame). ``pooled`` is ``False`` for a pair whose list holds more entries than the block (its window past the bound
    of its shape data at construction): it is walked instead."""
    record, window = solid_job_window(data, jobs, i)
    sink = solid_sink(block * scratch.capacity, scratch.capacity, record.pair.eps)
    pooled, held, raw = _hold_pooled(solid_scratch_store(scratch), scratch, jobs, i, sink.slot, record.pair.eps)
    sink.held = held
    sink.raw = raw
    return pooled, window, record.pair, record.cells, sink


# ----------------------------------------------------------------------------------------------------------------
# the job pipeline: a scan per pair, a thread per listed feature, a close per pair, rule X's items and valleys
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _job_vertex(cells: wp.vec4i, vertex: int) -> wp.vec2i:
    """Window vertex ``vertex`` (window order: row by row from ``(r0, c0)``) of a window of cells ``cells``, as
    ``(r, c)``."""
    width = cells[3] + 2 - cells[2]
    row = vertex // width
    return wp.vec2i(cells[0] + row, cells[2] + vertex - row * width)


@wp.func
def _job_append(jobs: TerrainSolidJobs, kind: int, entries: wp.array[wp.vec2i], job: wp.vec2i) -> int:
    """Append a job to a list (kind ``kind`` of :data:`TERRAIN_SOLID_JOB_LISTS`): 0 when it found no room."""
    entry = wp.atomic_add(jobs.counts, kind, 1)
    if entry < entries.shape[0]:
        entries[entry] = job
        return 1
    return 0


@wp.func
def _job_reserve(jobs: TerrainSolidJobs, kind: int, entries: wp.array[wp.vec2i], count: int) -> tuple[int, int]:
    """Reserve ``count`` consecutive entries of a list (kind ``kind`` of :data:`TERRAIN_SOLID_JOB_LISTS`) with one
    atomic: ``(first, kept)``, ``kept`` 0 when some of them found no room (they are still counted)."""
    if count == 0:
        return 0, 1
    first = wp.atomic_add(jobs.counts, kind, count)
    if first + count > entries.shape[0]:
        return first, 0
    return first, 1


@wp.func
def _job_put(entries: wp.array[wp.vec2i], entry: int, job: wp.vec2i) -> int:
    """Write a reserved job (:func:`_job_reserve`) where its list has room; returns the next entry."""
    if entry < entries.shape[0]:
        entries[entry] = job
    return entry + 1


@wp.func
def _bit(mask: wp.uint32, v: int) -> bool:
    """Whether bit ``v`` (0-31) of a mask is set."""
    return ((mask >> wp.uint32(v)) & wp.uint32(1)) != wp.uint32(0)


@wp.func
def _with_bit(mask: wp.uint32, v: int) -> wp.uint32:
    """A mask with bit ``v`` (0-31) set."""
    return mask | (wp.uint32(1) << wp.uint32(v))


@wp.func
def solid_scan(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    record: TerrainSolidRecord,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    run: int,
    run_vertices: int,
):
    """Run ``run`` of a live pair's scan (class pair ``i``, its kept core and record; the window vertices ``[run, run +
    1)`` times ``run_vertices``, its class's run of at most :data:`TERRAIN_SOLID_SCAN_VERTICES`, in window order): its
    vertices' features in window order through the
    cheapest rejects of their rules, every survivor listed for its job kernel: rule F's faces (:func:`_face_top`; of a cluster of several triangles, the first that passes, its
    cluster's manifold and emitting triangle being its cluster's alone), rule E's edges (:func:`_edge_cheap`) and rule
    V's vertices (:func:`_vertex_cheap`). The run marks its survivors in one bit per vertex and item, then reserves
    each list's entries with one atomic and writes them in window order (:func:`_job_reserve`): consecutive entries of a
    list are one run's, so that the threads of a job kernel's warp take one pair's features. A job without room flags
    the pair :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_INCOMPLETE` (its close then defers it)."""
    pair = record.pair
    cells = record.cells
    surface = window.surface
    r0 = cells[0]
    r1 = cells[1]
    c0 = cells[2]
    c1 = cells[3]
    width = c1 + 2 - c0
    first = run * run_vertices
    last = wp.min(first + run_vertices, (r1 + 2 - r0) * width)
    seen = _Seen()
    for k in range(_SEEN):
        seen[k] = -1
    oldest = int(0)
    # the survivors: bit v of a mask for the run's vertex first + v (a face of triangle 0 or 1, an x-, y- or diagonal
    # edge, the vertex)
    face0 = wp.uint32(0)
    face1 = wp.uint32(0)
    edge0 = wp.uint32(0)
    edge1 = wp.uint32(0)
    edge2 = wp.uint32(0)
    point = wp.uint32(0)
    faces = int(0)
    edges = int(0)
    points = int(0)
    for vertex in range(first, last):
        v = vertex - first
        row = vertex // width
        r = r0 + row
        c = c0 + vertex - row * width
        inside = r <= r1 and c <= c1
        if inside:
            for sub in range(2):
                cluster = terrain_cluster(data, surface, r, c, sub)
                known = bool(False)
                if cluster >= 0:
                    for k in range(_SEEN):
                        if seen[k] == cluster:
                            known = True
                if known:
                    continue
                if _face_top(data, window, r, c, sub) < pair.low:
                    continue
                if cluster >= 0:
                    seen[oldest] = cluster
                    oldest = (oldest + 1) % _SEEN
                if sub == 0:
                    face0 = _with_bit(face0, v)
                else:
                    face1 = _with_bit(face1, v)
                faces += 1
        for which in range(3):
            if (which == 0 and c > c1) or (which == 1 and r > r1) or (which == 2 and not inside):
                continue
            if _edge_cheap(data, window, pair, r, c, which):
                if which == 0:
                    edge0 = _with_bit(edge0, v)
                elif which == 1:
                    edge1 = _with_bit(edge1, v)
                else:
                    edge2 = _with_bit(edge2, v)
                edges += 1
        core, listed = _vertex_cheap(core, data, window, pair, r, c)
        if listed:
            point = _with_bit(point, v)
            points += 1
    entry, kept = _job_reserve(jobs, _FACES, jobs.faces, faces)
    if faces > 0:
        for v in range(last - first):
            item = (first + v) * _ITEMS
            if _bit(face0, v):
                entry = _job_put(jobs.faces, entry, wp.vec2i(pair_index, item))
            if _bit(face1, v):
                entry = _job_put(jobs.faces, entry, wp.vec2i(pair_index, item + 1))
    entry, room = _job_reserve(jobs, _EDGES, jobs.edges, edges)
    kept = kept & room
    if edges > 0:
        for v in range(last - first):
            item = (first + v) * _ITEMS
            if _bit(edge0, v):
                entry = _job_put(jobs.edges, entry, wp.vec2i(pair_index, item + 2))
            if _bit(edge1, v):
                entry = _job_put(jobs.edges, entry, wp.vec2i(pair_index, item + 3))
            if _bit(edge2, v):
                entry = _job_put(jobs.edges, entry, wp.vec2i(pair_index, item + 4))
    entry, room = _job_reserve(jobs, _VERTICES, jobs.vertices, points)
    kept = kept & room
    if points > 0:
        for v in range(last - first):
            if _bit(point, v):
                entry = _job_put(jobs.vertices, entry, wp.vec2i(pair_index, (first + v) * _ITEMS + 5))
    if kept == 0:
        wp.atomic_or(out.state, pair_index, TERRAIN_PAIR_INCOMPLETE)


@wp.func
def solid_job_window(
    data: TerrainFeatureData, jobs: TerrainSolidJobs, i: int
) -> tuple[TerrainSolidRecord, TerrainWindow]:
    """Class pair ``i``'s scan record and window frame (the scan's, bit for bit)."""
    record = jobs.records[i]
    return record, terrain_window(data, record.heightfield, record.cells[0], record.cells[2])


@wp.func
def _job_sink(i: int, item: int, eps: float) -> _Sink:
    """The sink of a job of class pair ``i`` on item ``item`` (its window vertex times six plus its item)."""
    sink = _Sink()
    sink.slot = i
    sink.key = item * _KEY_SLOTS
    sink.eps = eps
    return sink


@wp.func
def _jobs_overflowed(out: TerrainLawRows):
    """Record a pair some of whose work found no room in its class's job lists or candidate pool (it takes the
    full-finish kernel, its rows unchanged): the narrow phase's sticky ``terrain_jobs`` bit
    (:data:`~newton._src.geometry.terrain_law.TERRAIN_STATUS_JOBS`) and the side buffer's job overflow counter
    (``overflow[2]``, accumulated over collides)."""
    wp.atomic_or(out.status, 0, TERRAIN_STATUS_JOBS)
    wp.atomic_add(out.overflow, 2, 1)


@wp.func
def _job_lost(out: TerrainLawRows, pair_index: int, lost: int):
    """A job that lost a row (no room in the candidate pool, or for a crease in the list of creases): its pair goes to
    the full-finish kernel (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_INCOMPLETE`)."""
    if lost != 0:
        wp.atomic_or(out.state, pair_index, TERRAIN_PAIR_INCOMPLETE)


@wp.func
def solid_face_job(
    core: Any, data: TerrainFeatureData, out: TerrainLawRows, jobs: TerrainSolidJobs, pair_index: int, i: int, item: int
):
    """Rule F for the cluster of a listed face: its manifold, support tests and emitting triangle
    (:func:`_face_support`), and its rows, at the emitting triangle's place in window order, when the walk would emit
    them there: the triangle lies in the window and its highest vertex is not below the pair's ``low``."""
    record, window = solid_job_window(data, jobs, i)
    cells = record.cells
    pair = record.pair
    vertex = item // _ITEMS
    sub = item - vertex * _ITEMS
    rc = _job_vertex(cells, vertex)
    cluster = terrain_cluster(data, window.surface, rc[0], rc[1], sub)
    planes = terrain_planes()
    planes, supported, first, count, points, depth, n = _face_support(
        core, data, window, planes, pair, rc[0], rc[1], sub, cluster
    )
    if supported == 0:
        return
    if first[0] < cells[0] or first[0] > cells[1] or first[1] < cells[2] or first[1] > cells[3]:
        return
    if _face_top(data, window, first[0], first[1], first[2]) < pair.low:
        return
    width = cells[3] + 2 - cells[2]
    sink = _job_sink(i, ((first[0] - cells[0]) * width + first[1] - cells[2]) * _ITEMS + first[2], pair.eps)
    sink = _face_offer(jobs, sink, supported, count, points, depth, n, first)
    _job_lost(out, pair_index, sink.lost)


@wp.func
def solid_edge_job(
    core: Any, data: TerrainFeatureData, out: TerrainLawRows, jobs: TerrainSolidJobs, pair_index: int, i: int, item: int
):
    """Rule E's crease tests of a listed edge (:func:`_edge_arc`) and its pre-bounds (:func:`_edge_reach`); a convex
    crease whose arc can hold a row within reach is listed for ``theta*`` (:func:`solid_theta_job`)."""
    record, window = solid_job_window(data, jobs, i)
    vertex = item // _ITEMS
    rc = _job_vertex(record.cells, vertex)
    which = item - vertex * _ITEMS - 2
    planes = terrain_planes()
    planes, ok, n_a, n_b, _sigma_a, _sigma_b = _edge_arc(core, data, window, planes, record.pair, rc[0], rc[1], which)
    if not ok or not _edge_reach(core, data, window, record.pair, rc[0], rc[1], which, n_a, n_b):
        return
    entry = wp.atomic_add(jobs.counts, _THETAS, 1)
    if entry < jobs.thetas.shape[0]:
        jobs.thetas[entry] = wp.vec2i(pair_index, item)
    else:
        _job_lost(out, pair_index, 1)


@wp.func
def solid_theta_job(core: Any, data: TerrainFeatureData, jobs: TerrainSolidJobs, i: int, item: int, entry: int):
    """Rule E's ``theta*`` of a crease whose arc can hold a row (entry ``entry`` of :attr:`TerrainSolidJobs.thetas`):
    its arc again (:func:`_edge_arc`, the edge job's, bit for bit) and :func:`_edge_theta`, kept for
    :func:`solid_crease_job`."""
    record, window = solid_job_window(data, jobs, i)
    pair = record.pair
    vertex = item // _ITEMS
    which = item - vertex * _ITEMS - 2
    rc = _job_vertex(record.cells, vertex)
    planes = terrain_planes()
    planes, _ok, n_a, n_b, sigma_a, sigma_b = _edge_arc(core, data, window, planes, pair, rc[0], rc[1], which)
    nst, kind = _edge_theta(core, data, window, pair, rc[0], rc[1], which, n_a, n_b, sigma_a, sigma_b)
    jobs.theta_n[entry] = nst
    jobs.theta_kind[entry] = kind


@wp.func
def solid_crease_job(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    item: int,
    entry: int,
):
    """Rule E's rows of a crease at its ``theta*`` (entry ``entry`` of :attr:`TerrainSolidJobs.thetas`,
    :func:`solid_theta_job`'s): the crossing and its ends (:func:`_edge_rows_at`)."""
    kind = jobs.theta_kind[entry]
    if kind < 0:
        return
    record, window = solid_job_window(data, jobs, i)
    pair = record.pair
    vertex = item // _ITEMS
    which = item - vertex * _ITEMS - 2
    rc = _job_vertex(record.cells, vertex)
    planes = terrain_planes()
    sink = _job_sink(i, item, pair.eps)
    planes, sink = _edge_rows_at(
        core, data, window, planes, pair, sink, jobs, rc[0], rc[1], which, jobs.theta_n[entry], kind
    )
    _job_lost(out, pair_index, sink.lost)


@wp.func
def solid_vertex_job(
    core: Any, data: TerrainFeatureData, out: TerrainLawRows, jobs: TerrainSolidJobs, pair_index: int, i: int, item: int
):
    """Rule V's row of a listed vertex (:func:`_vertex_rows`)."""
    record, window = solid_job_window(data, jobs, i)
    pair = record.pair
    rc = _job_vertex(record.cells, item // _ITEMS)
    planes = terrain_planes()
    sink = _job_sink(i, item, pair.eps)
    planes, sink = _vertex_rows(core, data, window, planes, pair, sink, jobs, rc[0], rc[1])
    _job_lost(out, pair_index, sink.lost)


@wp.func
def _hold_candidate(
    out: TerrainLawRows, jobs: TerrainSolidJobs, entry: int, slot: int, held: int, eps: float, limit: int
) -> int:
    return terrain_hold_row(
        out, slot, held, jobs.q[entry], jobs.n[entry], jobs.d[entry], jobs.feature[entry], eps, limit
    )


@wp.func
def _hold_few(
    out: TerrainLawRows, jobs: TerrainSolidJobs, i: int, slot: int, limit: int, eps: float
) -> tuple[bool, int, int]:
    """:func:`_hold_candidates` of a list of at most :data:`TERRAIN_SOLID_FEW` candidates, its keys sorted in named
    registers (no array on the stack, nothing to zero): ``(few, held, raw)``, ``few`` ``False`` (nothing read past the
    list's fifth entry, nothing held) for a longer list. The same rows: one walk of the list, a stable insertion by
    key (equal keys after the first in list order, as :func:`_hold_sorted` inserts them), then the first entry of
    every distinct key held in key order."""
    k0 = int(_NO_KEY)
    k1 = int(_NO_KEY)
    k2 = int(_NO_KEY)
    k3 = int(_NO_KEY)
    e0 = int(-1)
    e1 = int(-1)
    e2 = int(-1)
    e3 = int(-1)
    count = int(0)
    entry = jobs.head[i]
    while entry >= 0:
        if count == _FEW:
            return False, 0, 0
        key = jobs.key[entry]
        if key < k0:
            k3 = k2
            e3 = e2
            k2 = k1
            e2 = e1
            k1 = k0
            e1 = e0
            k0 = key
            e0 = entry
        elif key < k1:
            k3 = k2
            e3 = e2
            k2 = k1
            e2 = e1
            k1 = key
            e1 = entry
        elif key < k2:
            k3 = k2
            e3 = e2
            k2 = key
            e2 = entry
        else:
            k3 = key
            e3 = entry
        count += 1
        entry = jobs.next[entry]
    held = int(0)
    raw = int(0)
    if count > 0:
        raw = 1
        held = _hold_candidate(out, jobs, e0, slot, held, eps, limit)
    if count > 1 and k1 != k0:
        raw += 1
        held = _hold_candidate(out, jobs, e1, slot, held, eps, limit)
    if count > 2 and k2 != k1:
        raw += 1
        held = _hold_candidate(out, jobs, e2, slot, held, eps, limit)
    if count > 3 and k3 != k2:
        raw += 1
        held = _hold_candidate(out, jobs, e3, slot, held, eps, limit)
    return True, held, raw


@wp.func
def _hold_candidates(
    out: TerrainLawRows, jobs: TerrainSolidJobs, i: int, slot: int, limit: int, eps: float
) -> tuple[int, int]:
    """Hold class pair ``i``'s candidates in its walk slots from ``slot`` in window order (by key), as the walk offers
    them (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`, at most ``limit``): ``(held, raw)``, its
    distinct rows and its offers. Candidates of one key (a cluster listed twice) are one offer; the list's order (its
    jobs') never matters. A list of at most :data:`TERRAIN_SOLID_FEW` candidates (most pairs' on the recorded states)
    is sorted in registers (:func:`_hold_few`), a longer one in chunks on the stack (:func:`_hold_sorted`)."""
    few, held, raw = _hold_few(out, jobs, i, slot, limit, eps)
    if few:
        return held, raw
    return _hold_sorted(out, jobs, i, slot, limit, eps)


@wp.func
def _hold_sorted(
    out: TerrainLawRows, jobs: TerrainSolidJobs, i: int, slot: int, limit: int, eps: float
) -> tuple[int, int]:
    """:func:`_hold_candidates` of any list: taken in chunks of :data:`TERRAIN_SOLID_SORTED` keys sorted in a stack
    array, one walk of it per chunk (:data:`TERRAIN_SOLID_SORTED`): one walk on the recorded states."""
    keys = _Sorted()
    entries = _Sorted()
    held = int(0)
    raw = int(0)
    last = int(-1)
    more = bool(True)
    while more:
        count = int(0)
        more = bool(False)
        entry = jobs.head[i]
        while entry >= 0:
            key = jobs.key[entry]
            if key > last:
                if count < _SORTED or key < keys[_SORTED - 1]:
                    more = more or count == _SORTED
                    j = wp.min(count, _SORTED - 1)
                    while j > 0 and keys[j - 1] > key:
                        keys[j] = keys[j - 1]
                        entries[j] = entries[j - 1]
                        j -= 1
                    keys[j] = key
                    entries[j] = entry
                    count = wp.min(count + 1, _SORTED)
                else:
                    more = True
            entry = jobs.next[entry]
        for j in range(count):
            if keys[j] == last:
                continue
            last = keys[j]
            raw += 1
            held = _hold_candidate(out, jobs, entries[j], slot, held, eps, limit)
    return held, raw


@wp.func
def _hold_pooled(
    store: TerrainLawRows,
    scratch: TerrainFinishScratch,
    jobs: TerrainSolidJobs,
    i: int,
    base: int,
    eps: float,
) -> tuple[bool, int, int]:
    """:func:`_hold_candidates` into a full-finish block's scratch from ``base`` (its distinct rows held in ``store``,
    the scratch's rows): ``(pooled, held, raw)``. The pair's list is sorted by key (equal keys in list order) in the
    block's finish state arrays, free until its finish (``group`` the keys, ``rank`` the entries), not on the stack;
    ``pooled`` is ``False`` for a list of more entries than the block holds (nothing held: walk the pair instead)."""
    count = int(0)
    entry = jobs.head[i]
    while entry >= 0:
        if count < scratch.capacity:
            key = jobs.key[entry]
            j = count
            while j > 0 and scratch.group[base + j - 1] > key:
                scratch.group[base + j] = scratch.group[base + j - 1]
                scratch.rank[base + j] = scratch.rank[base + j - 1]
                j -= 1
            scratch.group[base + j] = key
            scratch.rank[base + j] = entry
        count += 1
        entry = jobs.next[entry]
    if count > scratch.capacity:
        return False, 0, 0
    held = int(0)
    raw = int(0)
    last = int(-1)
    for j in range(count):
        key = scratch.group[base + j]
        if key == last:
            continue
        last = key
        raw += 1
        held = _hold_candidate(store, jobs, scratch.rank[base + j], base, held, eps, scratch.capacity)
    return True, held, raw


@wp.func
def _xjobs_open(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    record: TerrainSolidRecord,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    sink: _Sink,
    pair_index: int,
    i: int,
):
    """Open rule X for class pair ``i`` (left to its class's exit kernel, its rows held): its target (its deepest
    row's depth plus :data:`TERRAIN_SOLID_EXIT_MARGIN`, as :func:`solid_exits` takes it), its candidate list cleared,
    and its items listed for :func:`solid_xitem_job`: every core point, and every window vertex that can give a
    candidate (when its core takes candidates at window vertices: a valley needs an edge rising above the core's
    lowest point, a local maximum of the distance a vertex above it), in window order, each run of
    :data:`TERRAIN_SOLID_SCAN_VERTICES` window vertices reserving its items with one atomic (:func:`_job_reserve`).
    Items without room take every candidate."""
    deepest = float(0.0)
    for h in range(wp.min(sink.held, sink.capacity)):
        deepest = wp.max(deepest, -out.d[sink.slot + h])
    jobs.target[i] = deepest + _EXIT_MARGIN
    jobs.xhead[i] = -1
    jobs.xall[i] = 0
    # the gate's cells open the pair (solid_xbound_job); a class whose exits are not gated is open at once
    jobs.xhot[i] = wp.where(core_exits_gated(core), 0, 1)
    cells = record.cells
    points = core_exit_count(core)
    entry, kept = _job_reserve(jobs, _XITEMS, jobs.xitems, points)
    for j in range(points):
        entry = _job_put(jobs.xitems, entry, wp.vec2i(pair_index, -1 - j))
    vertices = int(0)
    if core_support_count(core) > 0 or core_far_count(core) > 0:
        vertices = (cells[1] - cells[0] + 2) * (cells[3] + 2 - cells[2])
    low = record.low
    for start in range(0, vertices, _SCAN_VERTICES):
        stop = wp.min(start + _SCAN_VERTICES, vertices)
        listed = wp.uint32(0)
        count = int(0)
        for v in range(start, stop):
            rc = _job_vertex(cells, v)
            r = rc[0]
            c = rc[1]
            above = _vertex_z(data, window, r, c) > low
            if c <= cells[3]:
                above = above or _vertex_z(data, window, r, c + 1) > low
            if r <= cells[1]:
                above = above or _vertex_z(data, window, r + 1, c) > low
            if r <= cells[1] and c <= cells[3]:
                above = above or _vertex_z(data, window, r + 1, c + 1) > low
            if above:
                listed = _with_bit(listed, v - start)
                count += 1
        entry, room = _job_reserve(jobs, _XITEMS, jobs.xitems, count)
        kept = kept & room
        if count > 0:
            for v in range(start, stop):
                if _bit(listed, v - start):
                    entry = _job_put(jobs.xitems, entry, wp.vec2i(pair_index, v))
    if kept == 0:
        jobs.xall[i] = 1


@wp.func
def _solid_settle(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    record: TerrainSolidRecord,
    sink: _Sink,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    solids: TerrainSolidData,
    pair_index: int,
    i: int,
    partner_class: int,
    frame: wp.transform,
):
    """A closed pair with its rows held (rule R's included): left to rule X (count in ``out.count``, state
    :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_EXIT`, its items listed, :func:`_xjobs_open`) when its core
    reaches below its window's highest vertex, else finished (:func:`_finish_or_lanes`)."""
    cells = record.cells
    pair = record.pair
    # Rule X can give a row only to a core whose lowest point lies below its window's highest vertex.
    if jobs.rule_x != 0 and record.low < pair.top:
        out.first[pair_index] = sink.slot
        out.count[pair_index] = sink.held
        out.state[pair_index] = _wide(cells) | TERRAIN_PAIR_EXIT
        entry = wp.atomic_add(solids.exit_count, partner_class, 1)
        solids.exits[out.deferred_first[partner_class] + entry] = pair_index
        _xjobs_open(core, data, window, record, out, jobs, sink, pair_index, i)
        return
    _finish_or_lanes(
        window, record, sink.slot, sink.held, sink.capacity, out, jobs, pair_index, i, partner_class, frame
    )


@wp.func
def _store_held(
    window: TerrainWindow,
    cells: wp.vec4i,
    slot: int,
    held: int,
    out: TerrainLawRows,
    pair_index: int,
    frame: wp.transform,
):
    """The finish of ``held`` distinct rows in a pair's walk slots from ``slot`` that hold no plane group of three:
    it keeps every one in order (no group can lose a row to its implied-row or per-group steps), so that they are
    written in the world frame in place, as :func:`solid_finish` writes them."""
    for h in range(held):
        s = slot + h
        terrain_store_row(out, s, out.q[s], out.n[s], out.d[s], out.feature[s], window, frame)
    out.first[pair_index] = slot
    out.count[pair_index] = held
    out.state[pair_index] = _wide(cells)


@wp.func
def _finish_or_lanes(
    window: TerrainWindow,
    record: TerrainSolidRecord,
    slot: int,
    held: int,
    capacity: int,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    partner_class: int,
    frame: wp.transform,
):
    """Finish a pair holding ``held`` rows in its walk slots from ``slot``: here, by this thread, when it holds fewer
    than :data:`TERRAIN_SOLID_LANE_ROWS` and no plane group of three
    (:func:`~newton._src.geometry.terrain_law.terrain_plane_group_of_three`), which the finish keeps as they are: they
    are written in the world frame in place (:func:`_store_held`); any other pair in the finish kernel by a block's
    lanes (state :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_HELD`, the class's lane list), so that no
    one-thread kernel holds the general finish's code and state."""
    if held > capacity:
        terrain_defer_pair(out, partner_class, pair_index, _wide(record.cells))
        return
    small = held < _LANE_ROWS
    if small:
        if not terrain_plane_group_of_three(out, slot, held, record.pair.eps):
            _store_held(window, record.cells, slot, held, out, pair_index, frame)
            return
    out.first[pair_index] = slot
    out.count[pair_index] = held
    out.state[pair_index] = _wide(record.cells) | TERRAIN_PAIR_HELD
    # the small and lane lists hold every pair of the class at most
    if small:
        jobs.small[wp.atomic_add(jobs.counts, _SMALL, 1)] = i
    else:
        jobs.lanes[wp.atomic_add(jobs.counts, _LANES, 1)] = i


@wp.func
def solid_close_jobs(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    solids: TerrainSolidData,
    pair_index: int,
    i: int,
    slot: int,
    walk_slots: int,
    partner_class: int,
    frame: wp.transform,
):
    """Close a scanned pair (class pair ``i``, its walk slots from ``slot``): its candidates held in window order
    (:func:`_hold_candidates`), which gives it the walk's held rows. A pair that lost work, or holds more distinct rows
    than its slots, is left to the class's full-finish kernel; a pair without a row to rule R
    (:func:`solid_recover_pair`, its list); any other is settled (:func:`_solid_settle`)."""
    state = out.state[pair_index]
    if (state & TERRAIN_PAIR_INCOMPLETE) != 0:
        # some of its work found no room (its record may be missing: a window past its bound lists no first run), so
        # that the full-finish kernel walks it again
        terrain_defer_pair(out, partner_class, pair_index, state & (TERRAIN_PAIR_WIDE | TERRAIN_PAIR_INCOMPLETE))
        _jobs_overflowed(out)
        return
    record, window = solid_job_window(data, jobs, i)
    cells = record.cells
    pair = record.pair
    held, raw = _hold_candidates(out, jobs, i, slot, walk_slots, pair.eps)
    sink = solid_sink(slot, walk_slots, pair.eps)
    sink.held = held
    sink.raw = raw
    if sink.held > sink.capacity:
        terrain_defer_pair(out, partner_class, pair_index, _wide(cells))
        return
    if sink.raw == 0:
        if record.low > pair.top + _RECOVERY_CLEAR:
            # The core lies above every vertex of its window: rule R finds none of its points below the surface, and
            # rule X needs a core below the window's highest vertex, so the pair keeps no row.
            out.first[pair_index] = slot
            out.count[pair_index] = 0
            out.state[pair_index] = _wide(cells)
            return
        # the recovery list holds every pair of the class at most
        jobs.recover[wp.atomic_add(jobs.counts, _RECOVER, 1)] = i
        return
    _solid_settle(core, data, window, record, sink, out, jobs, solids, pair_index, i, partner_class, frame)


@wp.func
def solid_recover_pair(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    solids: TerrainSolidData,
    pair_index: int,
    i: int,
    slot: int,
    walk_slots: int,
    partner_class: int,
    frame: wp.transform,
    lane: int,
    lanes: int,
):
    """Rule R for a closed pair without a row (class pair ``i``, its walk slots from ``slot``) by a block's ``lanes``
    lanes (lane ``lane``, :func:`solid_recovery`), then its settling by the first lane (:func:`_solid_settle`)."""
    record, window = solid_job_window(data, jobs, i)
    sink = solid_sink(slot, walk_slots, record.pair.eps)
    planes = terrain_planes()
    planes, sink = solid_recovery(core, data, window, planes, sink, out, lane, lanes)
    if lane == 0:
        _solid_settle(core, data, window, record, sink, out, jobs, solids, pair_index, i, partner_class, frame)


@wp.func
def _exit_footprint(core: Any, window: TerrainWindow) -> wp.vec4i:
    """The cells ``(r0, r1, c0, c1)`` under the core's footprint (its bounds widened as
    :func:`~newton._src.geometry.terrain_law.terrain_box_cells` widens them, clamped to the heightfield): every core
    point lies over one of them or over no cell of the heightfield."""
    surface = window.surface
    lo, hi = core_bounds(core)
    c0 = wp.max(window.c0 + int(wp.floor(lo[0] / surface.dx - _CELL_WIDEN)), 0)
    c1 = wp.min(window.c0 + int(wp.floor(hi[0] / surface.dx + _CELL_WIDEN)), surface.ncol - 2)
    r0 = wp.max(window.r0 + int(wp.floor(lo[1] / surface.dy - _CELL_WIDEN)), 0)
    r1 = wp.min(window.r0 + int(wp.floor(hi[1] / surface.dy + _CELL_WIDEN)), surface.nrow - 2)
    return wp.vec4i(r0, r1, c0, c1)


@wp.func
def _xy_apart(core: Any, va: wp.vec3, vb: wp.vec3, vc: wp.vec3) -> bool:
    """Whether an edge of the triangle ``va vb vc`` separates the core from it in x and y (the core wholly beyond the
    edge's line, away from the third vertex)."""
    for e in range(3):
        p = va
        q = vb
        o = vc
        if e == 1:
            p = vb
            q = vc
            o = va
        elif e == 2:
            p = vc
            q = va
            o = vb
        m = wp.vec3(q[1] - p[1], p[0] - q[0], 0.0)
        if m[0] * (o[0] - p[0]) + m[1] * (o[1] - p[1]) > 0.0:
            m = -m
        if core_sigma(core, wp.normalize(m), p) > 0.0:
            return True
    return False


@wp.func
def _cluster_around(data: TerrainFeatureData, surface: TerrainSurfaceInfo, r: int, c: int, sub: int) -> bool:
    """Whether every triangle of the 3 x 3 cells around cell ``(r, c)`` lies in the cluster of its triangle ``sub``
    (cells off the heightfield: no)."""
    cluster = terrain_cluster(data, surface, r, c, sub)
    if cluster < 0 or r < 1 or c < 1 or r > surface.nrow - 3 or c > surface.ncol - 3:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            for s2 in range(2):
                if terrain_cluster(data, surface, rr, cc, s2) != cluster:
                    return False
    return True


@wp.func
def _exit_cell_open(
    core: Any, data: TerrainFeatureData, window: TerrainWindow, pair: _Pair, target: float, r: int, c: int
) -> bool:
    """Whether an exit of a core point over cell ``(r, c)`` can lie deeper than ``target``: rule X's gate per cell
    (:func:`solid_xbound_job`). ``False`` only where none can.

    Every rule X candidate is a point of the core whose exit is its closest surface point. A candidate over a triangle
    of the cell lies below the triangle's plane (its cluster's, the law's surface there) by no more than the core's
    part over it (:func:`~newton._src.geometry.terrain_law_cores.core_column`) does, ``delta`` along the plane's
    normal, and below the triangle's highest vertex by no more than that part's lowest point. Its exit is no deeper
    than the surface point straight above it: ``delta`` over the normal's ``z``, or that rise. Where the 3 x 3 cells
    around the triangle are all of its cluster, the plane's point along the normal is a surface point too (its offset
    of ``delta`` times the normal's slope stays within a cell), so the exit is no deeper than ``delta`` itself. A
    triangle the part's projection misses (one of its edges separates them in x and y) holds no candidate. Both bounds
    are widened by the pair's band and the cluster tolerance (over the normal's ``z`` for the vertical one)."""
    surface = window.surface
    guard = 2.0 * _FLAT + 4.0 * pair.eps
    up = wp.vec3(0.0, 0.0, 1.0)
    planes = terrain_planes()
    for sub in range(2):
        ia, ib, ic = terrain_triangle_vertices(r, c, sub)
        va = terrain_window_vertex(data, window, ia[0], ia[1])
        vb = terrain_window_vertex(data, window, ib[0], ib[1])
        vc = terrain_window_vertex(data, window, ic[0], ic[1])
        over, part = core_column(
            core,
            wp.vec2(wp.min(wp.min(va[0], vb[0]), vc[0]), wp.min(wp.min(va[1], vb[1]), vc[1])),
            wp.vec2(wp.max(wp.max(va[0], vb[0]), vc[0]), wp.max(wp.max(va[1], vb[1]), vc[1])),
        )
        if not over or _xy_apart(part, va, vb, vc):
            continue
        planes, n, a = terrain_plane(data, window, planes, r, c, sub)
        if n[2] <= 0.0:
            return True
        delta = -core_sigma(part, n, a)
        rise = wp.max(wp.max(va[2], vb[2]), vc[2]) - core_sigma(part, up, wp.vec3(0.0))
        if wp.min(delta / n[2], rise) + guard / n[2] <= target:
            continue
        slope = wp.sqrt(wp.max(1.0 - n[2] * n[2], 0.0))
        if delta + guard > target or delta * slope > wp.min(surface.dx, surface.dy):
            return True
        if not _cluster_around(data, surface, r, c, sub):
            return True
    return False


@wp.func
def solid_xbound_job(core: Any, data: TerrainFeatureData, jobs: TerrainSolidJobs, i: int, item: int):
    """Rule X's gate at item ``item`` of class pair ``i`` (:attr:`TerrainSolidJobs.xitems`): it opens the pair
    (:attr:`TerrainSolidJobs.xhot`) when an exit over its cell can pass the pair's target (:func:`_exit_cell_open`).
    A core point's cell is the one under it; a window vertex's is its own when it lies under the core's footprint
    (:func:`_exit_footprint`).

    A core point tried as it is (a box's corner, a cone's apex) is a candidate of its own, its exit bounded over the
    cell under it. The others (valleys' support points, local maxima of the distance) lie over the footprint, and the
    close lists a window vertex when a corner of its cell rises above the core's lowest point; over any other cell the
    surface lies below every core point (by the cluster tolerance at most, far below the exit margin), which then has
    no exit that can pass the target. So a pair no item opens has no exit deeper than its target: rule X gives it
    nothing, and its items list nothing (:func:`solid_xitem_job`)."""
    if jobs.xhot[i] != 0:
        return
    record, window = solid_job_window(data, jobs, i)
    r = int(-1)
    c = int(-1)
    if item < 0:
        x = core_exit_point(core, -1 - item)
        t = terrain_locate(window, x[0], x[1])
        if t[0] < 0:
            # over no cell of the heightfield: no exit
            return
        r = t[0]
        c = t[1]
    else:
        rc = _job_vertex(record.cells, item)
        foot = _exit_footprint(core, window)
        if rc[0] < foot[0] or rc[0] > foot[1] or rc[1] < foot[2] or rc[1] > foot[3]:
            return
        r = rc[0]
        c = rc[1]
    if _exit_cell_open(core, data, window, record.pair, jobs.target[i], r, c):
        jobs.xhot[i] = 1


@wp.func
def solid_xitem_job(core: Any, data: TerrainFeatureData, jobs: TerrainSolidJobs, i: int, item: int):
    """Rule X's item ``item`` of class pair ``i`` (:attr:`TerrainSolidJobs.xitems`): a core point, or a window
    vertex's valleys and local maxima of the distance, at the target: a candidate whose bound passes it is listed
    (:attr:`TerrainSolidJobs.xhead`; the test rule X's passes open with, :func:`_exit_open`, before any candidate is
    taken); a valley whose gate passes it goes to :func:`solid_xvalley_job`. Work without room takes every candidate
    (:attr:`TerrainSolidJobs.xall`). A pair whose gate no cell opened (:attr:`TerrainSolidJobs.xhot`) lists nothing:
    no exit of its core can pass the target."""
    if jobs.xhot[i] == 0:
        return
    record, window = solid_job_window(data, jobs, i)
    cells = record.cells
    target = jobs.target[i]
    planes = terrain_planes()
    if item < 0:
        j = -1 - item
        x = core_exit_point(core, j)
        planes, bound = _exit_bound(data, window, planes, x)
        if bound > target:
            if _x_list(jobs, i, x, bound, j, -1, 0.0, 0.0) == 0:
                jobs.xall[i] = 1
        return
    low = record.low
    far = core_far_count(core)
    base = core_exit_count(core) + item * _EXIT_SLOTS
    rc = _job_vertex(cells, item)
    r = rc[0]
    c = rc[1]
    inside = r <= cells[1] and c <= cells[3]
    valleys = wp.uint32(0)
    count = int(0)
    for which in range(3):
        if (which == 0 and c > cells[3]) or (which == 1 and r > cells[1]) or (which == 2 and not inside):
            continue
        planes, ok, _v0, _e, _length2, n_a, n_b, top = _valley_arc(data, window, planes, low, r, c, which)
        if not ok:
            continue
        # The valley's gate (_exit_open) at the lowest threshold either pass can take, the target: a valley below it
        # stays below every later threshold (a nonpositive scale makes the gate pass, as here).
        if top - low < target * wp.min(n_a[2], n_b[2]):
            continue
        valleys = _with_bit(valleys, which)
        count += 1
    # the item's valleys reserved at once (:func:`_job_reserve`)
    entry, kept = _job_reserve(jobs, _XVALLEYS, jobs.xvalleys, count)
    if kept == 0:
        jobs.xall[i] = 1
    for which in range(3):
        if _bit(valleys, which):
            entry = _job_put(jobs.xvalleys, entry, wp.vec2i(i + jobs.first, 3 * item + which))
    if far == 0 or not terrain_interior_vertex(window.surface, r, c):
        return
    v = terrain_window_vertex(data, window, r, c)
    if v[2] <= low:
        return
    for f in range(far):
        has, x = core_far_point(core, f, v)
        if not has or x[2] >= v[2]:
            continue
        planes, bound = _exit_bound(data, window, planes, x)
        if bound > 0.0:
            bound = wp.min(bound, wp.length(x - v))
        if bound > target:
            if _x_list(jobs, i, x, bound, base + 54 + f, -1, 0.0, 0.0) == 0:
                jobs.xall[i] = 1


@wp.func
def solid_xvalley_job(core: Any, data: TerrainFeatureData, jobs: TerrainSolidJobs, i: int, item: int):
    """Rule X at a listed valley of class pair ``i`` (``item``: its window vertex times three plus its edge): the
    core's support points at the local minima of ``sigma`` over its arc (:func:`_exit_valley`'s); one whose bound
    passes the target is listed with the valley's gate (:func:`_x_list`)."""
    record, window = solid_job_window(data, jobs, i)
    vertex = item // 3
    which = item - 3 * vertex
    rc = _job_vertex(record.cells, vertex)
    target = jobs.target[i]
    low = record.low
    key = core_exit_count(core) + vertex * _EXIT_SLOTS + 18 * which
    planes = terrain_planes()
    planes, _ok, v0, e, length2, n_a, n_b, top = _valley_arc(data, window, planes, low, rc[0], rc[1], which)
    samples, values, axis = _valley_samples(core, v0, n_a, n_b)
    for k in range(9):
        if not _valley_minimum(values, k):
            continue
        nst = _valley_refine(core, samples, values, axis, v0, k)
        for s in range(core_support_count(core)):
            x = core_support_point(core, nst, s)
            if x[2] >= top:
                continue
            planes, bound = _valley_bound(data, window, planes, x, v0, e, length2)
            if bound > target:
                gate = top - low
                scale = wp.min(n_a[2], n_b[2])
                if _x_list(jobs, i, x, bound, key + 2 * k + s, key, gate, scale) == 0:
                    jobs.xall[i] = 1


@wp.func
def solid_xdescent_job(data: TerrainFeatureData, jobs: TerrainSolidJobs, k: int):
    """The near bound's descent of listed rule X candidate ``k`` (:func:`_near_descent`), for the exit kernel's
    passes."""
    _record, window = solid_job_window(data, jobs, jobs.xpair[k])
    entry = _pool_rows(jobs) + k
    jobs.descent[k] = _near_descent(data, window, jobs.q[entry], jobs.d[entry])


@wp.func
def solid_exit_pair(
    core: Any,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    solids: TerrainSolidData,
    pair_index: int,
    i: int,
    slot: int,
    walk_slots: int,
    partner_class: int,
    frame: wp.transform,
):
    """Rule X for a pair its close left to the class's exit kernel, its rows held in its walk slots from ``slot``
    (``out.count`` of them): the deepest exit over its listed candidates (:func:`solid_exits`) joins its rows, which
    are then finished (:func:`_finish_or_lanes`). A pair some of whose rule X work found no room is left to the
    class's full-finish kernels (its candidates all pooled: the pooled kernel), which take every candidate."""
    if jobs.xall[i] != 0:
        terrain_defer_pair(out, partner_class, pair_index, out.state[pair_index] & TERRAIN_PAIR_WIDE)
        _jobs_overflowed(out)
        return
    record, window = solid_job_window(data, jobs, i)
    pair = record.pair
    held = out.count[pair_index]
    # with no listed candidate both passes are no-ops
    if jobs.xhead[i] >= 0:
        planes = terrain_planes()
        sink = solid_sink(slot, walk_slots, pair.eps)
        sink.held = held
        sink.raw = held
        listed = _ListedCandidates()
        listed.row = i
        planes, sink = solid_exits(
            core, data, window, planes, pair, sink, out, record.cells, solids.exit_work, jobs, listed
        )
        held = sink.held
    _finish_or_lanes(window, record, slot, held, walk_slots, out, jobs, pair_index, i, partner_class, frame)


@wp.func
def solid_finish_pair(
    data: TerrainFeatureData,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    slot: int,
    walk_slots: int,
    partner_class: int,
    frame: wp.transform,
    lane: int,
    lanes: int,
):
    """The finish of a closed pair (class pair ``i``, its walk slots from ``slot``) by a block's ``lanes`` lanes: a
    pair holding its rows for it (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_HELD`, from its close or rule
    R) or kept by the exit kernel (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_EXIT`, rule X's row among
    its rows; one the exit kernel left to the full-finish kernels no longer carries it) is finished in place
    (:func:`solid_finish`), the rows of a walk that finished it itself; any other pair is passed by. Every lane reads
    the same state, so that the block takes or passes the pair together."""
    if (out.state[pair_index] & (TERRAIN_PAIR_HELD | TERRAIN_PAIR_EXIT)) == 0:
        return
    record, window = solid_job_window(data, jobs, i)
    solid_finish(
        window,
        record.pair,
        slot,
        out.count[pair_index],
        walk_slots,
        out,
        pair_index,
        partner_class,
        frame,
        record.cells,
        lane,
        lanes,
    )


@wp.func
def solid_unsupported(out: TerrainLawRows, pair_index: int):
    """Record a pair whose partner has no core in the law (no row,
    :func:`~newton._src.geometry.terrain_law.terrain_pair_exit`)."""
    terrain_pair_exit(out, pair_index, TERRAIN_PAIR_UNSUPPORTED)


# ----------------------------------------------------------------------------------------------------------------
# the kernels of one solid class
# ----------------------------------------------------------------------------------------------------------------

TERRAIN_SOLID_LAUNCHES = (
    ("pairs", "rows"),
    ("jobs", "scan"),
    ("jobs", "faces"),
    ("jobs", "edges"),
    ("jobs", "thetas"),
    ("jobs", "creases"),
    ("jobs", "vertices"),
    ("pair_jobs", "close"),
    ("pair_tiles", "recovery"),
    ("jobs", "xbound"),
    ("jobs", "xitems"),
    ("jobs", "xvalleys"),
    ("jobs", "xdescents"),
    ("pair_jobs", "exit"),
    ("pair_jobs", "small"),
    ("pair_tiles", "finish"),
    ("full", "pooled"),
    ("full", "full"),
)
"""A solid class's launches in order, by kind and kernel (:func:`solid_kernels`): ``pairs`` one thread per pair of the
class, ``jobs`` the class's job threads (:attr:`TerrainSolidJobs.threads`, striding over a list; a thread past them
returns at once, so that a larger launch takes no entry twice), ``pair_jobs`` the same over a list of the class's pairs
with at most one thread per pair (:attr:`TerrainSolidJobs.pair_threads`), ``pair_tiles`` blocks of lanes striding over
a list of the class's pairs, one pair per block at a time (:attr:`TerrainSolidJobs.finish_blocks`, ``wp.launch_tiled``),
``full`` the full-finish grid (:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`,
``wp.launch_tiled``). A job kernel whose list has no
room by construction (:func:`terrain_solid_job_capacities`: none of the class's pairs can list an entry, as rule X's
valleys of a class without vertex candidates) is not launched (:data:`TERRAIN_SOLID_LIST_OF_KERNEL`)."""
TERRAIN_SOLID_LIST_OF_KERNEL = {
    "faces": "faces",
    "edges": "edges",
    "thetas": "thetas",
    "creases": "thetas",
    "vertices": "vertices",
    "xbound": "xitems",
    "xitems": "xitems",
    "xvalleys": "xvalleys",
    "xdescents": "xdescents",
}
"""The job kernels that take a sized list (:func:`terrain_solid_job_capacities`), by the list they take: a kernel whose
list has no room is not launched."""


def solid_kernels(
    name: str, partner_class: int, walk_slots: int, scan_run: int, make_core, save_core, load_core, module: str
) -> dict:
    """Build one solid class's kernels in module ``module`` (whose options, contraction off, they take), named
    ``terrain_<name>_rows`` (the scan) and ``terrain_<name>_rows_<kernel>`` for the others of
    :data:`TERRAIN_SOLID_LAUNCHES`.

    Every kernel takes ``(begin, slot_begin, inputs, data, out)`` and the class's
    :class:`TerrainSolidJobs` and the solid classes' :class:`TerrainSolidData` (the full-finish kernel: its
    :class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`, the data, then the jobs): table pair ``begin + i``
    is the class's pair ``i``, its walk slots ``slot_begin + i * walk_slots``.

    Args:
        name: The class's name in kernel names.
        partner_class: The class (``TERRAIN_PARTNER_*``).
        walk_slots: Walk slots of a pair of the class.
        scan_run: Window vertices of one scan job of the class
            (:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`), at most
            :data:`TERRAIN_SOLID_SCAN_VERTICES`.
        make_core: ``wp.func`` ``(local, size, kind, partner, solids) -> (supported, core)``: a pair's core in the
            heightfield frame.
        save_core: ``wp.func`` ``(jobs, i, core)``: keep class pair ``i``'s core (window frame) for its jobs.
        load_core: ``wp.func`` ``(jobs, i, solids) -> core``: class pair ``i``'s kept core.
        module: The module the kernels join.
    """
    if not 0 < scan_run <= TERRAIN_SOLID_SCAN_VERTICES:
        raise ValueError(f"A scan run takes 1 to {TERRAIN_SOLID_SCAN_VERTICES} window vertices, not {scan_run}")
    rows_ = wp.constant(walk_slots)
    class_ = wp.constant(partner_class)
    scan_ = wp.constant(scan_run)

    def rows(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """The whole-pair rejects, one thread per pair of the class (stale features, the pair policy, a partner without
        a core, a window off the heightfield, the tile-max exit): a pair past them keeps its core and record (window
        frame) for its scan and jobs, is listed live (:attr:`TerrainSolidJobs.live`) for the close, its window's runs
        for the scan (:attr:`TerrainSolidJobs.scans`), and takes state
        :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_JOBS`; any other is recorded with no row."""
        i = wp.tid()
        pair = begin + i
        state, _frame, heightfield, local, partner, size, reach = solid_pair_read(inputs, data, pair)
        if state != 0:
            terrain_pair_exit(out, pair, state)
            return
        supported, core = wp.static(make_core)(local, size, inputs.shape_type[partner], partner, solids)
        if not supported:
            solid_unsupported(out, pair)
            return
        state, cells, lo, hi = solid_pair_cells(core, data, heightfield, reach)
        if state != 0:
            terrain_pair_exit(out, pair, state)
            return
        _window, core, setup, quiet = solid_pair_frame(core, data, heightfield, cells, lo, hi, reach)
        wp.static(save_core)(jobs, i, core)
        record = TerrainSolidRecord()
        record.cells = cells
        record.heightfield = heightfield
        record.pair = setup
        low, _high = core_bounds(core)
        record.low = low[2]
        jobs.records[i] = record
        out.first[pair] = -1
        out.count[pair] = 0
        out.state[pair] = _wide(cells) | TERRAIN_PAIR_JOBS
        jobs.head[i] = -1
        # the live list holds every pair of the class at most
        jobs.live[wp.atomic_add(jobs.counts, _LIVE, 1)] = i
        if quiet:
            # A quiet window (:func:`_vertex_quiet`, every vertex): its edges list no crease and its vertices no
            # feature, and its triangles, joined by flat edges, are one cluster, whose rows are its own. Its scan
            # would list that cluster alone, once per run, when a triangle's highest vertex is not below the pair's
            # low: when the window's highest vertex is not. One face job takes it (its first triangle; the cluster's
            # rows at the same place in window order), and no run is scanned.
            if setup.top >= setup.low:
                if _job_append(jobs, _FACES, jobs.faces, wp.vec2i(pair, 0)) == 0:
                    out.state[pair] = _wide(cells) | TERRAIN_PAIR_JOBS | TERRAIN_PAIR_INCOMPLETE
            return
        vertices = (cells[1] - cells[0] + 2) * (cells[3] - cells[2] + 2)
        runs = (vertices + scan_ - 1) // scan_
        first = wp.atomic_add(jobs.counts, _SCANS, runs)
        for k in range(runs):
            if first + k < jobs.scans.shape[0]:
                jobs.scans[first + k] = wp.vec2i(i, k)
            else:
                # a window larger than its shape data bounded at construction: the full-finish kernel walks it
                out.state[pair] = _wide(cells) | TERRAIN_PAIR_JOBS | TERRAIN_PAIR_INCOMPLETE

    def scan(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """The scan, one run of a live pair's window per thread at a time: the run's features through their cheapest
        rejects, the survivors listed for the job kernels (:func:`solid_scan`; the pair's core and record kept by the
        whole-pair kernel)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_SCANS], jobs.scans.shape[0]), jobs.threads):
            job = jobs.scans[entry]
            i = job[0]
            record, window = solid_job_window(data, jobs, i)
            solid_scan(
                wp.static(load_core)(jobs, i, solids), data, window, record, out, jobs, begin + i, i, job[1], scan_
            )

    def faces(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule F, one listed face per thread at a time (the grid strides over the list, :func:`solid_face_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_FACES], jobs.faces.shape[0]), jobs.threads):
            job = jobs.faces[entry]
            i = job[0] - begin
            solid_face_job(wp.static(load_core)(jobs, i, solids), data, out, jobs, job[0], i, job[1])

    def edges(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule E's crease tests, one listed edge per thread at a time (:func:`solid_edge_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_EDGES], jobs.edges.shape[0]), jobs.threads):
            job = jobs.edges[entry]
            i = job[0] - begin
            solid_edge_job(wp.static(load_core)(jobs, i, solids), data, out, jobs, job[0], i, job[1])

    def thetas(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule E's ``theta*``, one listed crease per thread at a time (:func:`solid_theta_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_THETAS], jobs.thetas.shape[0]), jobs.threads):
            job = jobs.thetas[entry]
            i = job[0] - begin
            solid_theta_job(wp.static(load_core)(jobs, i, solids), data, jobs, i, job[1], entry)

    def creases(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule E's rows, one listed crease per thread at a time, at its ``theta*`` (:func:`solid_crease_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_THETAS], jobs.thetas.shape[0]), jobs.threads):
            job = jobs.thetas[entry]
            i = job[0] - begin
            solid_crease_job(wp.static(load_core)(jobs, i, solids), data, out, jobs, job[0], i, job[1], entry)

    def vertices(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule V, one listed vertex per thread at a time (:func:`solid_vertex_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_VERTICES], jobs.vertices.shape[0]), jobs.threads):
            job = jobs.vertices[entry]
            i = job[0] - begin
            solid_vertex_job(wp.static(load_core)(jobs, i, solids), data, out, jobs, job[0], i, job[1])

    def close(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """The close, one live pair per thread at a time: a scanned pair's rows held in window order for the finish
        kernel, its rule X's items or rule R's list (:func:`solid_close_jobs`); a pair the scan deferred is passed
        by."""
        t = wp.tid()
        if t >= jobs.pair_threads:
            return
        for entry in range(t, jobs.counts[_LIVE], jobs.pair_threads):
            i = jobs.live[entry]
            pair = begin + i
            if (out.state[pair] & TERRAIN_PAIR_JOBS) != 0:
                solid_close_jobs(
                    wp.static(load_core)(jobs, i, solids),
                    data,
                    out,
                    jobs,
                    solids,
                    pair,
                    i,
                    slot_begin + i * rows_,
                    rows_,
                    class_,
                    inputs.shape_transform[inputs.pairs[pair][0]],
                )

    def recovery(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule R, ``wp.block_dim()`` lanes per closed pair without a row (the class's recovery list, block ``b`` its
        pairs ``b``, ``b + finish_blocks``, ...; ``wp.launch_tiled`` with ``dim=jobs.finish_blocks``): the lanes split
        the pair's core points, and the first lane offers its rows and finishes it, holds them for the finish kernel or
        lists its rule X's items (:func:`solid_recover_pair`)."""
        block, lane = wp.tid()
        if block >= jobs.finish_blocks:
            return
        lanes = wp.block_dim()
        for entry in range(block, jobs.counts[_RECOVER], jobs.finish_blocks):
            i = jobs.recover[entry]
            pair = begin + i
            solid_recover_pair(
                wp.static(load_core)(jobs, i, solids),
                data,
                out,
                jobs,
                solids,
                pair,
                i,
                slot_begin + i * rows_,
                rows_,
                class_,
                inputs.shape_transform[inputs.pairs[pair][0]],
                lane,
                lanes,
            )

    def xbound(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule X's gate, one item per thread at a time (:func:`solid_xbound_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_XITEMS], jobs.xitems.shape[0]), jobs.threads):
            job = jobs.xitems[entry]
            i = job[0] - begin
            solid_xbound_job(wp.static(load_core)(jobs, i, solids), data, jobs, i, job[1])

    def xitems(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule X's items, one per thread at a time: candidates and valleys (:func:`solid_xitem_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_XITEMS], jobs.xitems.shape[0]), jobs.threads):
            job = jobs.xitems[entry]
            i = job[0] - begin
            solid_xitem_job(wp.static(load_core)(jobs, i, solids), data, jobs, i, job[1])

    def xvalleys(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule X's valleys, one per thread at a time: candidates (:func:`solid_xvalley_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for entry in range(t, wp.min(jobs.counts[_XVALLEYS], jobs.xvalleys.shape[0]), jobs.threads):
            job = jobs.xvalleys[entry]
            i = job[0] - begin
            solid_xvalley_job(wp.static(load_core)(jobs, i, solids), data, jobs, i, job[1])

    def xdescents(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule X's listed candidates' near-bound descents, one per thread at a time (:func:`solid_xdescent_job`)."""
        t = wp.tid()
        if t >= jobs.threads:
            return
        for k in range(t, wp.min(jobs.counts[_XDESCENTS], jobs.descent.shape[0]), jobs.threads):
            solid_xdescent_job(data, jobs, k)

    def exit_(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """Rule X, one pair of the class's exit list per thread at a time (the job threads stride over it): a listed
        pair's deepest exit over its listed candidates (:func:`solid_exit_pair`)."""
        t = wp.tid()
        if t >= jobs.pair_threads:
            return
        for entry in range(t, solids.exit_count[class_], jobs.pair_threads):
            pair = solids.exits[out.deferred_first[class_] + entry]
            i = pair - begin
            solid_exit_pair(
                wp.static(load_core)(jobs, i, solids),
                data,
                out,
                jobs,
                solids,
                pair,
                i,
                slot_begin + i * rows_,
                rows_,
                class_,
                inputs.shape_transform[inputs.pairs[pair][0]],
            )

    def small(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """The finish of the class's pairs holding fewer than :data:`TERRAIN_SOLID_LANE_ROWS` rows with a plane group of
        three (its small list, from its close, rule R and rule X's exit kernel), one pair per thread at a time, so that
        a warp runs the general finish on as many pairs as it has lanes and no other kernel holds its code and stack
        (:func:`solid_finish_pair` with one lane)."""
        t = wp.tid()
        if t >= jobs.pair_threads:
            return
        for entry in range(t, jobs.counts[_SMALL], jobs.pair_threads):
            i = jobs.small[entry]
            pair = begin + i
            solid_finish_pair(
                data,
                out,
                jobs,
                pair,
                i,
                slot_begin + i * rows_,
                rows_,
                class_,
                inputs.shape_transform[inputs.pairs[pair][0]],
                0,
                1,
            )

    def finish(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """The finish of the class's pairs that hold :data:`TERRAIN_SOLID_LANE_ROWS` rows or more (its lane list,
        from its close, rule R and rule X's exit kernel), ``wp.block_dim()`` lanes per pair (:func:`solid_finish_pair`):
        block ``b`` takes the listed pairs ``b``, ``b + finish_blocks``, ... ``wp.launch_tiled`` with
        ``dim=jobs.finish_blocks``; a block past it returns at once."""
        block, lane = wp.tid()
        if block >= jobs.finish_blocks:
            return
        lanes = wp.block_dim()
        for entry in range(block, jobs.counts[_LANES], jobs.finish_blocks):
            i = jobs.lanes[entry]
            pair = begin + i
            solid_finish_pair(
                data,
                out,
                jobs,
                pair,
                i,
                slot_begin + i * rows_,
                rows_,
                class_,
                inputs.shape_transform[inputs.pairs[pair][0]],
                lane,
                lanes,
            )

    def full(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        scratch: TerrainFinishScratch,
        solids: TerrainSolidData,
        jobs: TerrainSolidJobs,
    ):
        """Finish the class's deferred pairs some of whose work found no room, and those the pooled kernel could not hold
        (both still carry :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_FULL`): the grid's ``scratch.blocks``
        blocks stride over the class's deferred list, and each block's first lane walks its pairs again, one at a time,
        into the block's scratch (every distinct candidate a window of the class can give, :func:`solid_rows_full`), runs
        rules R and X and the finish and writes every row it keeps, in the pair's walk slots when they fit, else in rows
        of the spill region (:func:`solid_close_full`). A pair's rows depend on the pair alone: not on the order of the
        deferral, the number of deferred pairs or the block that finishes it. ``wp.launch_tiled`` with
        ``dim=scratch.blocks``; a block past it returns at once."""
        block, lane = wp.tid()
        if block >= scratch.blocks or lane != 0:
            return
        for entry in range(block, out.deferred_count[class_], scratch.blocks):
            pair = out.deferred[begin + entry]
            if (out.state[pair] & TERRAIN_PAIR_FULL) == 0:
                # the pooled kernel finished it
                continue
            i = pair - begin
            _state, frame, heightfield, local, partner, size, reach = solid_pair_read(inputs, data, pair)
            _supported, core = wp.static(make_core)(local, size, inputs.shape_type[partner], partner, solids)
            solid_rows_full(
                core,
                data,
                out,
                scratch,
                solids,
                jobs,
                block,
                pair,
                slot_begin + i * rows_,
                rows_,
                frame,
                heightfield,
                reach,
            )

    def pooled(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        scratch: TerrainFinishScratch,
        solids: TerrainSolidData,
        jobs: TerrainSolidJobs,
    ):
        """Finish the class's deferred pairs every job of which found room (more distinct rows than their walk slots,
        or rule X's lists without room) from their candidates in the candidate pool, not by a second walk of their
        windows (:func:`solid_pooled`), before the full-finish kernel walks the others: the full-finish grid strides
        over the class's deferred list, each block's first lane taking its pairs one at a time into the block's
        scratch, and runs rules R and X and the finish and writes the rows (:func:`solid_close_full`), as the walk
        would. A pair whose list holds more entries than a block is marked
        :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_INCOMPLETE` for the full-finish kernel. A kernel of its
        own, so that neither kernel holds both paths' state. ``wp.launch_tiled`` with ``dim=scratch.blocks``; a block
        past it returns at once."""
        block, lane = wp.tid()
        if block >= scratch.blocks or lane != 0:
            return
        for entry in range(block, out.deferred_count[class_], scratch.blocks):
            pair = out.deferred[begin + entry]
            if (out.state[pair] & TERRAIN_PAIR_INCOMPLETE) != 0:
                continue
            i = pair - begin
            found, window, setup, cells, sink = solid_pooled(data, scratch, jobs, block, i)
            if not found:
                wp.atomic_or(out.state, pair, TERRAIN_PAIR_INCOMPLETE)
                continue
            solid_close_full(
                wp.static(load_core)(jobs, i, solids),
                data,
                window,
                terrain_planes(),
                setup,
                sink,
                scratch,
                out,
                solids.exit_work,
                jobs,
                pair,
                slot_begin + i * rows_,
                rows_,
                inputs.shape_transform[inputs.pairs[pair][0]],
                cells,
            )

    kernels = {}
    for kernel_name, function in (
        ("rows", rows),
        ("scan", scan),
        ("faces", faces),
        ("edges", edges),
        ("thetas", thetas),
        ("creases", creases),
        ("vertices", vertices),
        ("close", close),
        ("recovery", recovery),
        ("xbound", xbound),
        ("xitems", xitems),
        ("xvalleys", xvalleys),
        ("xdescents", xdescents),
        ("exit", exit_),
        ("small", small),
        ("finish", finish),
        ("pooled", pooled),
        ("full", full),
    ):
        key = f"terrain_{name}_rows" if kernel_name == "rows" else f"terrain_{name}_rows_{kernel_name}"
        function.__name__ = key
        function.__qualname__ = key
        kernels[kernel_name] = wp.kernel(function, enable_backward=False, module=module)
    return kernels


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
    pairs: int | None = None,
    exit_count: wp.array | None = None,
    exit_work: wp.array | None = None,
) -> TerrainSolidData:
    """The solid classes' data for a model: the hull table of the convex-mesh partners in ``partners`` (shape
    indices) and rule X's exit lists. Their launch arguments, each class's jobs and the full finish's scratch, are the
    shadow's (:func:`terrain_solid_jobs`, :func:`~newton._src.geometry.terrain_law.terrain_finish_scratch`).

    Args:
        model: A finalized model.
        partners: Partner shapes of the solid classes' pairs.
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
    room = len(partners) if pairs is None else int(pairs)
    solids.exits = wp.zeros(max(room, 1), dtype=wp.int32, device=device)
    solids.exit_count = exit_count if exit_count is not None else wp.zeros(4, dtype=wp.int32, device=device)
    solids.exit_work = exit_work if exit_work is not None else wp.zeros(2, dtype=wp.int32, device=device)
    return solids


def terrain_solid_exit_items(model, partners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per partner: rule X's core points (the candidates it tries as they are: a box's 8 corners, a hull's points, a
    cone's apex) and whether its core takes candidates at window vertices (its valleys' support points and its local
    maxima of the distance: cylinders, cones and ellipsoids).

    Args:
        model: A finalized model.
        partners: Partner shapes.
    """
    from .types import GeoType  # noqa: PLC0415

    partners = np.asarray(partners, np.int64)
    types = model.shape_type.numpy()[partners] if len(partners) else np.zeros(0, np.int32)
    points = np.zeros(len(partners), np.int64)
    points[types == int(GeoType.BOX)] = 8
    points[types == int(GeoType.CONE)] = 1
    for k in np.flatnonzero(types == int(GeoType.CONVEX_MESH)):
        points[k] = len(np.asarray(model.shape_source[int(partners[k])].vertices).reshape(-1, 3))
    takes = (types == int(GeoType.CYLINDER)) | (types == int(GeoType.CONE)) | (types == int(GeoType.ELLIPSOID))
    return points, takes


def terrain_solid_job_capacities(
    vertices: np.ndarray,
    points: np.ndarray,
    takes_vertices: np.ndarray,
    candidates: np.ndarray,
    walk_slots: int,
    *,
    scan_run: int,
    scale: float = 1.0,
) -> dict[str, int]:
    """Return a solid class's job list capacities by default, from its pairs' own bounds: per list
    (:data:`TERRAIN_SOLID_JOB_LISTS`) the largest pair's bound plus the sum of every pair's bound over the list's
    share (:data:`TERRAIN_SOLID_JOB_SHARES`), the candidate pool's rows (``candidates``): the largest pair's bound on
    its candidates plus the class's walk slots over :data:`TERRAIN_SOLID_CANDIDATE_SHARE`, and rule X's listed
    candidates (``xdescents``, the pool's region of their own): the largest pair's rule X keys plus its pairs' rule X
    items over :data:`TERRAIN_SOLID_EXIT_SHARE`; at least the sum of every pair's bound up to
    :data:`TERRAIN_SOLID_JOB_FLOOR` (:data:`TERRAIN_SOLID_POOL_FLOOR` for the pool's rows), so that a scene with few
    pairs never lacks room; each times ``scale``. The scan's runs (``scans``) are exact: every pair's at its largest
    window (:data:`TERRAIN_SOLID_SCAN_VERTICES` vertices each). Every list and region is a scene-level buffer, as
    ``rigid_contact_max`` sizes contacts: a pair whose work finds no room takes its class's full-finish kernel (its rows
    unchanged, at a slower path's cost), raises the narrow phase's ``terrain_jobs`` and is counted
    (:meth:`~newton._src.geometry.terrain_shadow.TerrainLawShadow.overflow_counts`), and the job check prints the
    entries asked for against the room. The shares are defaults with room to spare on the recorded suite states
    (16,384 worlds): there the most used list holds 43% of its room, rule X's region 4%.

    Args:
        vertices: Per pair, the bound on its window's vertices
            (:func:`~newton._src.geometry.terrain_contacts.terrain_window_vertices`).
        points: Per pair, its core's rule X points (:func:`terrain_solid_exit_items`).
        takes_vertices: Per pair, whether its core takes rule X candidates at window vertices.
        candidates: Per pair, the bound on its distinct candidates
            (:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`).
        walk_slots: Walk slots of a pair of the class.
        scan_run: Window vertices of one scan job of the class
            (:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`): the scan's runs are sized
            exactly from the pairs' windows.
        scale: Factor on every capacity.
    """
    v = np.asarray(vertices, np.int64)
    takes = np.asarray(takes_vertices, bool)
    points = np.asarray(points, np.int64)
    bounds = (
        2 * v,
        3 * v,
        3 * v,
        v,
        points + np.where(takes, v, 0),
        np.where(takes, 3 * v, 0),
    )
    capacities = {}
    for name, bound, share in zip(TERRAIN_SOLID_JOB_LISTS, bounds, TERRAIN_SOLID_JOB_SHARES, strict=True):
        total = int(bound.sum())
        default = (int(bound.max()) + -(-total // share)) if len(bound) else 0
        default = max(default, min(total, TERRAIN_SOLID_JOB_FLOOR))
        # a list none of the pairs can fill has no room (its kernel is not launched)
        capacities[name] = max(1, int(np.ceil(scale * default))) if total > 0 else 0
    # a pair's pool entries: up to eight rows (its key's point slots) per listed face, crease and vertex
    total = int(np.sum(48 * v))
    default = (int(np.max(candidates)) if len(v) else 0) + walk_slots * len(v) // TERRAIN_SOLID_CANDIDATE_SHARE
    default = max(default, min(total, TERRAIN_SOLID_POOL_FLOOR))
    capacities["candidates"] = max(1, int(np.ceil(scale * default)))
    # rule X's listed candidates, the pool's region of their own: every key of the largest pair's window (its core's
    # points and, taking candidates at window vertices, their key slots), and one per item for one pair in 256
    keys = points + np.where(takes, TERRAIN_SOLID_EXIT_SLOTS * v, 0)
    items = points + np.where(takes, v, 0)
    default = (int(keys.max()) + -(-int(items.sum()) // TERRAIN_SOLID_EXIT_SHARE)) if len(v) else 0
    default = max(default, min(int(keys.sum()), TERRAIN_SOLID_JOB_FLOOR))
    capacities["xdescents"] = max(1, int(np.ceil(scale * default)))
    # the scan's runs: every pair's at its largest window, exact (a window past its bound takes the full finish)
    capacities["scans"] = max(1, int(np.sum(-(-v // int(scan_run)))))
    return capacities


def terrain_solid_jobs(
    device,
    partner_class: int,
    first: int,
    pairs: int,
    capacities: dict[str, int],
    *,
    counts: wp.array | None = None,
    rule_x: bool = True,
) -> TerrainSolidJobs:
    """Allocate a solid class's :class:`TerrainSolidJobs`: per pair its record, core, candidate list heads and rule
    X's target; the job lists, the scan's runs and the candidate pool with ``capacities``
    (:func:`terrain_solid_job_capacities`); the job kernels' threads (:data:`TERRAIN_SOLID_JOB_THREADS` on CUDA,
    :data:`TERRAIN_SOLID_JOB_THREADS_CPU` on the CPU, at most the longest list).

    Args:
        device: The device.
        partner_class: The class (``TERRAIN_PARTNER_*``): which core records it keeps.
        first: The class's first table pair.
        pairs: Pairs of the class.
        capacities: Entries of every job list, the scan's runs and the candidate pool.
        counts: The class's :data:`TERRAIN_SOLID_COUNTERS` counters (zeroed before every launch), or ``None`` for its
            own.
        rule_x: Whether the class's close leaves pairs to rule X (:attr:`TerrainSolidJobs.rule_x`).
    """
    from .terrain_contacts import TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL  # noqa: PLC0415

    room = max(int(pairs), 1)
    jobs = TerrainSolidJobs()
    jobs.first = int(first)
    jobs.records = wp.zeros(room, dtype=TerrainSolidRecord, device=device)
    own = {
        TERRAIN_PARTNER_BOX: "box_cores",
        TERRAIN_PARTNER_CYLINDER: "cylinder_cores",
        TERRAIN_PARTNER_HULL: "hull_cores",
    }
    for name, dtype in (
        ("box_cores", TerrainBoxCore),
        ("cylinder_cores", TerrainCylinderCore),
        ("hull_cores", TerrainHullCoreRecord),
    ):
        setattr(jobs, name, wp.zeros(room if own.get(int(partner_class)) == name else 1, dtype=dtype, device=device))
    for name in TERRAIN_SOLID_JOB_LISTS:
        setattr(jobs, name, wp.zeros(max(int(capacities[name]), 1), dtype=wp.vec2i, device=device))
    jobs.theta_n = wp.zeros(max(int(capacities["thetas"]), 1), dtype=wp.vec3, device=device)
    jobs.theta_kind = wp.zeros(max(int(capacities["thetas"]), 1), dtype=wp.int32, device=device)
    jobs.live = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.scans = wp.zeros(max(int(capacities["scans"]), 1), dtype=wp.vec2i, device=device)
    jobs.recover = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.lanes = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.small = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.counts = counts if counts is not None else wp.zeros(TERRAIN_SOLID_COUNTERS, dtype=wp.int32, device=device)
    longest = max(int(capacities[name]) for name in (*TERRAIN_SOLID_JOB_LISTS, "scans"))
    cuda = wp.get_device(device).is_cuda
    threads = TERRAIN_SOLID_JOB_THREADS if cuda else TERRAIN_SOLID_JOB_THREADS_CPU
    jobs.threads = max(1, min(threads, longest))
    jobs.pair_threads = max(1, min(jobs.threads, room))
    jobs.finish_blocks = max(1, min(TERRAIN_SOLID_FINISH_BLOCKS if cuda else TERRAIN_SOLID_JOB_THREADS_CPU, room))
    exits = max(int(capacities["xdescents"]), 1)
    pool = max(int(capacities["candidates"]), 1) + exits
    jobs.q = wp.zeros(pool, dtype=wp.vec3, device=device)
    jobs.n = wp.zeros(pool, dtype=wp.vec3, device=device)
    jobs.d = wp.zeros(pool, dtype=wp.float32, device=device)
    jobs.feature = wp.zeros(pool, dtype=wp.int32, device=device)
    jobs.key = wp.zeros(pool, dtype=wp.int32, device=device)
    jobs.next = wp.zeros(pool, dtype=wp.int32, device=device)
    jobs.descent = wp.zeros(exits, dtype=wp.vec4, device=device)
    jobs.xpair = wp.zeros(exits, dtype=wp.int32, device=device)
    jobs.head = wp.full(room, -1, dtype=wp.int32, device=device)
    jobs.target = wp.zeros(room, dtype=wp.float32, device=device)
    jobs.xhead = wp.full(room, -1, dtype=wp.int32, device=device)
    jobs.xall = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.xhot = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.rule_x = int(bool(rule_x))
    return jobs


def terrain_solid_jobs_bytes(jobs: TerrainSolidJobs) -> int:
    """Device bytes of a solid class's jobs."""
    arrays = (
        jobs.records,
        jobs.box_cores,
        jobs.cylinder_cores,
        jobs.hull_cores,
        jobs.faces,
        jobs.edges,
        jobs.thetas,
        jobs.theta_n,
        jobs.theta_kind,
        jobs.vertices,
        jobs.xitems,
        jobs.xvalleys,
        jobs.live,
        jobs.scans,
        jobs.recover,
        jobs.lanes,
        jobs.small,
        jobs.q,
        jobs.n,
        jobs.d,
        jobs.feature,
        jobs.key,
        jobs.next,
        jobs.descent,
        jobs.xpair,
        jobs.head,
        jobs.target,
        jobs.xhead,
        jobs.xall,
        jobs.xhot,
    )
    return int(sum(array.capacity for array in arrays))

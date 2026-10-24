# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The terrain law for solid cores (boxes, cylinders and cones, convex hulls and ellipsoids): its rules, and the kernels
that run them on a class's pairs.

A solid core has no rounding radius; its reach ``D`` is both margins and both gaps (the contact writer's admission
limit). Per pair the law visits every surface feature of the window once (window vertex ``(r, c)`` owns its cell's
two faces and diagonal, its x- and y-edges and itself, and on the heightfield's border the walls below its border edges,
those edges' clips of their faces and at a corner of the footprint the corner line below it,
:data:`TERRAIN_SOLID_ITEMS`) and offers the rows of the law's rules for solids, in window order:

* F, faces: the analytic plane manifold of the core against the triangle's cluster plane
  (:func:`~newton._src.geometry.terrain_law_cores.core_manifold`); a manifold point is supported when its footprint
  along the normal lies in a triangle of the cluster and it is front-facing, and the triangle holding the first
  supported point emits every supported point (one emission per plane). The faces are clipped to the heightfield's
  footprint: at a border edge, the core's part on its face's side of the clip plane through the edge (containing the
  face's normal) gives the face that part's deepest points on the plane (:func:`_clip_theta`,
  :func:`_clip_rows_at`; past a corner of the footprint, the core's point on the corner line). The walls (one plane
  per side, from the border edges down :data:`~newton._src.geometry.terrain_law.TERRAIN_WALL_DEPTH`) are faces too,
  one-sided: their manifold points beside them whose projection a piece of wall holds (:func:`_wall_support`);
* E, convex creases: ``theta* = argmax sigma(theta)`` over the crease's normal arc: nine equally spaced normals, then
  a bisection on the sign of the derivative between the best one's neighbours
  (:func:`~newton._src.geometry.terrain_law_cores.core_slope`, closed form for boxes and cylinders, the support
  function for hulls; 22 steps; the float64 law brackets the same samples and refines by golden section on
  ``sigma``); inside the arc, the core's flat support feature within the band
  ``beta = max(0.1 mm, 1e-3 size)``, at an arc end its reference feature (G1); the interval where it meets the
  crease's line emits its ends lifted onto the core (no deeper than the feature's chord over them), ends on vertex
  features left to the vertex. A border edge is a convex crease between its triangle and its wall; a corner line,
  one-sided as its walls are, between its two walls;
* V, vertex features: the direction to the core's nearest point (or the minimum translation from inside) inside the
  vertex's normal cone, within reach; a border vertex's cone takes its walls' normals, and its ring the wall below it;
* R, recovery (G3): a pair with no row whose core lies below the surface gets up to four of the core's deepest
  boundary points against the face vertically above them, clipped to the footprint: its points inside it (its part
  inside it on a border edge's clip plane below the edge's face is rule F's clip row, so a pair left to rule R has
  none).

Rule C (concave anticipation) is off. Cheapest rejects first: the tile-max exit, a feature skipped when its highest
vertex lies below the core's lowest point minus the reach, a face's plane rejected by one support call before its
manifold, an edge rejected by the core's bounding sphere (a row's line along its normal passes through the sphere; out
of reach only where the edge rises nowhere above the core's lowest point: a row deeper than the reach can stand far
from the core) and by its arc ends' support values before the search, a vertex by the bounding sphere and one support
call before its feature test and nearest point.

The kernels (:func:`solid_kernels`, one module per partner class, in the order of :data:`TERRAIN_SOLID_LAUNCHES`) take
the shape of today's terrain route: a cheap scan that lists the features that survive the cheapest rejects, one thread
per listed feature, and a per-pair close that holds their rows in window order, so that a warp runs one rule on many
pairs' features at once instead of one pair's whole window, and no thread walks a large window alone:

1. The whole-pair rejects, one thread per pair; a pair past them keeps its core and window frame
   (:class:`TerrainSolidRecord`) for the scan and the jobs and is listed live, its window in runs of its class's
   :data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN` window vertices. The scan
   (:func:`solid_scan`), one thread per run: the run's
   features through their cheapest rejects in window order: rule F's faces (one per cluster plane and run: its
   manifold, support tests and emitting triangle are its cluster's alone, :func:`_face_support`) and walls (one per wall
   and run, :func:`_wall_cheap`), rule E's edges and corner lines (:func:`_edge_cheap`) and rule V's vertices
   (:func:`_vertex_cheap`), each survivor listed for its job kernel.
2. The job kernels, a fixed grid striding over each list: :func:`solid_face_job`, :func:`solid_edge_job` (its crease
   tests, :func:`_edge_arc`; a crease whose arc can hold a row is listed for ``theta*``, a border edge whose clip plane
   the core may cross for its clip, :func:`_clip_cheap`), :func:`solid_theta_job` (``theta*``, :func:`_edge_theta`,
   kept per listed crease), :func:`solid_crease_job` (the crossing at ``theta*`` and its ends, :func:`_edge_rows_at`;
   apart, so that the search runs in a lean kernel), :func:`solid_vertex_job` (:func:`_vertex_rows`) and
   :func:`solid_border_job`, the border's walls (:func:`solid_wall_job`) and clips (:func:`_clip_theta` and
   :func:`_clip_rows_at` in one thread): its own kernel, so that the others hold none of the border's rare paths.
   A job adds its rows to the class's candidate pool, each with its place in window order (its key: its item's index,
   window vertex times :data:`TERRAIN_SOLID_ITEMS` plus its item, times eight plus its point slot), the row the walk
   would offer there.
3. The close (:func:`solid_close_jobs`), one thread per live pair: the pair's candidates held in its walk slots in key
   order (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`, the rows a walk of its window holds, bit for bit,
   however the jobs ran; a candidate listed twice is one offer), then rule R's kernel for a pair without a row, a block
   of lanes per pair splitting its core points (:func:`solid_recover_pair`); either then finishes the pair (step 4). A
   pair holding more distinct rows than its walk slots is deferred (step 5).
4. The finish: a pair whose held rows hold no plane group of three is written as it is by the thread that holds it
   (its close or rule R's first lane); any other is listed for the small-pair finish kernel, one thread per pair
   (:func:`solid_finish_pair`).
5. The finish of the deferred pairs (:func:`solid_finish_deferred`), a block of lanes per pair on the fixed grid of the
   class's scratch (:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`, which holds every candidate a
   window of the class can give, :func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`): the first
   lane places the pair's candidates from the pool in key order, one per key as its close takes them, the lanes merge
   and finish them, every step split among them
   (:func:`~newton._src.geometry.terrain_law.terrain_finish_list`), and write every row it keeps in the pair's walk
   slots when they fit, else in rows of the spill region it reserves
   (:func:`~newton._src.geometry.terrain_law.terrain_finish_deferred`).

The kernels follow the per-pair output protocol of :mod:`~newton._src.geometry.terrain_law`: a pair with no row is
recorded by :func:`~newton._src.geometry.terrain_law.terrain_pair_exit`, and rows are held in the pair's own walk slots
(window frame) as the finish merges them (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`): a row coinciding
with a held one is dropped, the first in window order staying. The finish runs in place on the held rows
(:func:`~newton._src.geometry.terrain_law.terrain_finish_slots`, a row's core point being ``q + d n``), and the kept
rows are rewritten in the world frame at the front of the slots; a deferred pair's finish keeps the same rows of its
candidates. The job lists and the candidate pool are scene-level buffers sized from the pairs' bounds
(:func:`terrain_solid_job_capacities`), each with a reserve beyond its room (:func:`terrain_solid_job_reserves`): work
past a list's room takes the reserve, the same kernels running it, so that its pair's rows are unchanged; a pair some
of whose work finds no room even there keeps no row. Every pair is thus finished from its own candidates, whatever the
order of the jobs, the number of deferred pairs, the order of the deferral or the block that finishes it.
Every module that compiles the kernels disables floating-point contraction, so the CPU decides every boundary as the GPU
does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .broad_phase_common import is_shape_pair_immovable_filtered
from .terrain_law import (
    TERRAIN_EDGE_CONCAVE,
    TERRAIN_EDGE_CORNER,
    TERRAIN_EPS32,
    TERRAIN_FACE_WALL,
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_FEATURE_VERTEX,
    TERRAIN_FLAT_EPS,
    TERRAIN_LAW_BLOCK,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_HELD,
    TERRAIN_PAIR_INCOMPLETE,
    TERRAIN_PAIR_JOBS,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_RESERVE,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_WIDE,
    TERRAIN_STATUS_JOBS,
    TERRAIN_VERTEX_FEATURE_BIT,
    TERRAIN_WALL_DEPTH,
    TERRAIN_WINDOW_WIDE,
    TerrainFeatureData,
    TerrainFinishScratch,
    TerrainLawRows,
    TerrainPairInputs,
    TerrainPlanes,
    TerrainSurfaceInfo,
    TerrainWindow,
    terrain_beyond_walls,
    terrain_border_far,
    terrain_border_side,
    terrain_border_triangle,
    terrain_box_cells,
    terrain_cluster,
    terrain_cluster_plane,
    terrain_crease_class,
    terrain_defer_pair,
    terrain_edge_open_crease,
    terrain_edge_sides,
    terrain_eps,
    terrain_feature_id,
    terrain_features_current,
    terrain_finish_deferred,
    terrain_finish_slots,
    terrain_height,
    terrain_hold_row,
    terrain_interior_vertex,
    terrain_locate,
    terrain_pair_exit,
    terrain_place_candidate,
    terrain_plane,
    terrain_plane_group_of_three,
    terrain_planes,
    terrain_point_below,
    terrain_point_below_cached,
    terrain_point_below_inside,
    terrain_store_row,
    terrain_tile_exit,
    terrain_triangle_vertices,
    terrain_vertex_bits,
    terrain_vertex_feature,
    terrain_vertex_in_cone,
    terrain_vertex_sides,
    terrain_wall_holds,
    terrain_wall_normal,
    terrain_wall_value,
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
    core_bounds,
    core_capsule,
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
TERRAIN_SOLID_SEEN = 8
"""Clusters of several triangles a scan job remembers (round robin): it lists the first face of each it meets for
rule F, whose emitting triangle and rows are its cluster's alone. A cluster forgotten and met again, or met by another
of the pair's scan jobs, is listed again; its rows, the same and at the same place in window order, merge in the
close."""
TERRAIN_SOLID_JOB_LISTS = ("faces", "edges", "thetas", "vertices", "border")
"""The job lists of a solid class, in their counters' order (the candidate pool's counter follows): rule F's faces, rule
E's edges, rule E's creases whose arcs can hold a row (``theta*``), rule V's vertices and the heightfield border's
walls and clipped faces (:func:`solid_border_job`)."""
TERRAIN_SOLID_JOB_SHARES = (8, 16, 32, 8, 8)
"""Per job list (:data:`TERRAIN_SOLID_JOB_LISTS`), the share of its pairs' bounds it holds by default
(:func:`terrain_solid_job_capacities`): the largest pair's bound plus the sum of every pair's bound over this. A pair's
bound on a list is its window's features of the list's kind at most (two faces, three edges or one vertex per window
vertex, :func:`~newton._src.geometry.terrain_contacts.terrain_window_vertices`; four walls per scan run and the clips of
two border edges per window vertex along its sides)."""
TERRAIN_SOLID_CANDIDATE_SHARE = 4
"""The candidate pool of a solid class holds, by default, its pairs' walk slots over this, plus the largest pair's bound
on its candidates (:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`)."""
TERRAIN_SOLID_JOB_FLOOR = 1 << 14
"""Entries a solid class's job list holds by default at least, or all its pairs' bounds when they sum to less: the work
of a scene with few pairs always finds room (16,384 entries, 128 KiB)."""
TERRAIN_SOLID_POOL_FLOOR = 1 << 16
"""Entries a solid class's candidate pool holds by default at least, or all its pairs' bounds when they sum to less
(65,536 entries, 2.5 MiB)."""
TERRAIN_SOLID_RESERVE_PAIRS = 64
"""Pairs of a solid class's largest bounds whose work each of its job lists, its candidate pool and its scan's runs hold
beyond their room (:func:`terrain_solid_job_reserves`), whatever ``NEWTON_TERRAIN_LAW_SHADOW_JOBS``: work that
outgrows the room takes the reserve, the same kernels running it, so that its pair's rows are unchanged (it raises
``terrain_jobs``); only work past the reserve is lost (its pair keeps no row). Fewer pairs when the reserve would
exceed :data:`TERRAIN_SOLID_RESERVE_BYTES`."""
TERRAIN_SOLID_RESERVE_BYTES = 1 << 24
"""Bytes of a solid class's reserve (:data:`TERRAIN_SOLID_RESERVE_PAIRS`) up to which it holds its full count of pairs:
a class whose windows can give more work holds fewer (at least one)."""
TERRAIN_SOLID_JOB_THREADS = 1 << 17
"""Threads of a job kernel's launch on CUDA at most (fewer with a smaller list): a fixed grid that strides over its
list."""
TERRAIN_SOLID_BORDER_THREADS = 1 << 12
"""Threads of the border kernel's launch on CUDA at most (:func:`solid_border_job`, fewer with a smaller list): the
border's jobs are rare, and an empty launch of a smaller grid costs less (the job kernels' grid,
:data:`TERRAIN_SOLID_JOB_THREADS`, took 8 us of the GB300's per collide)."""
TERRAIN_SOLID_JOB_THREADS_CPU = 4
"""Threads of a job kernel's launch on the CPU, which runs a launch's threads one after another: a few, so that each
job's cost is the job's (they still stride over the list)."""
TERRAIN_SOLID_FINISH_BLOCKS = TERRAIN_SOLID_JOB_THREADS // TERRAIN_LAW_BLOCK
"""Blocks of rule R's launch on CUDA at most (fewer with fewer pairs): the job threads' worth of
:data:`~newton._src.geometry.terrain_law.TERRAIN_LAW_BLOCK`-lane blocks, each taking one pair of its list at a time
with its lanes."""
TERRAIN_SOLID_RECOVERY_CLEAR = 1.0e-4
"""Height [m] by which a core without a candidate must clear its window's highest vertex for its close to leave it
without a row instead of listing it for rule R: every core point then lies above the plane of the face under it (a
cluster plane lies within :data:`~newton._src.geometry.terrain_law.TERRAIN_FLAT_EPS` of its triangles' vertices) by
far more than the float32 rounding of its plane distance, so rule R would find none of them below the surface."""
TERRAIN_SOLID_SCAN_VERTICES = 32
"""Window vertices one scan job takes at most (the bits of a run's survivor masks; a class's run is
:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`): a large window is scanned by several threads,
so that no thread scans a large window alone. A cluster plane met by two runs is listed twice; its rows, the same at
the same place in window order, merge in the close."""
TERRAIN_SOLID_FEW = 4
"""Candidates of a pair the close sorts in registers (:func:`_hold_few`); a longer list is sorted in chunks of
:data:`TERRAIN_SOLID_SORTED` on the stack."""
TERRAIN_SOLID_SORTED = 32
"""Keys of a pair's pool list sorted in registers per walk of the list: the close (:func:`_hold_candidates`) takes a
list of ``n`` candidates in key order in ``ceil(n / 32)`` walks, each keeping the smallest keys past the last one taken
(a full chunk drops its largest, which a later walk takes)."""
TERRAIN_HULL_FACE_TOL = 1.0e-6
"""A hull's points within this of a face's plane lie on that face [m] (hulls under 1 m; their local frame): a planar
face of a float32 mesh stays one face (its points lie within about 1e-8 m of its plane)."""

_BISECTION = wp.constant(TERRAIN_SOLID_BISECTION)
_ArcNormals = wp.types.matrix(shape=(TERRAIN_SOLID_ARC_SAMPLES, 3), dtype=wp.float32)
_ARC_END = wp.constant(TERRAIN_SOLID_ARC_END)
_WIDE = wp.constant(TERRAIN_WINDOW_WIDE)
_EPS32 = wp.constant(TERRAIN_EPS32)
_FLAT = wp.constant(TERRAIN_FLAT_EPS)
_HUGE = wp.constant(3.0e38)
TERRAIN_SOLID_ITEMS = 9
"""Items a window vertex ``(r, c)`` owns in a solid pair's window order: its cell's faces (0, 1), its x-, y- and diagonal
edges (2, 3, 4), itself (5), the corner line below it at a corner of the footprint (6) and on the border, the wall below
its x-edge or y-edge and that edge's clip of its triangle's face (7, 8). A row's key is its item's index (window vertex
times this plus its item) times eight plus its point slot."""
_ITEMS = wp.constant(TERRAIN_SOLID_ITEMS)
_ITEM_VERTEX = wp.constant(5)
_ITEM_CORNER = wp.constant(6)
_ITEM_BORDER = wp.constant(7)
_WALL_DEPTH = wp.constant(TERRAIN_WALL_DEPTH)
_CLIP_SLOT = wp.constant(4)
"""Point slot of a clip row's first end (:func:`_clip_rows_at`; a wall's manifold points take slots 0-3)."""
_SEEN = wp.constant(TERRAIN_SOLID_SEEN)
_SORTED = wp.constant(TERRAIN_SOLID_SORTED)
_FEW = wp.constant(TERRAIN_SOLID_FEW)
_NO_KEY = wp.constant(2147483647)
_SCAN_VERTICES = wp.constant(TERRAIN_SOLID_SCAN_VERTICES)
_RECOVERY_CLEAR = wp.constant(TERRAIN_SOLID_RECOVERY_CLEAR)
_KEY_SLOTS = wp.constant(8)
_FACES = wp.constant(0)
_EDGES = wp.constant(1)
_THETAS = wp.constant(2)
_VERTICES = wp.constant(3)
_BORDER = wp.constant(4)
_CANDIDATES = wp.constant(5)
_LIVE = wp.constant(6)
_RECOVER = wp.constant(7)
_SCANS = wp.constant(8)
_SMALL = wp.constant(9)
_CLIP_JOB = wp.constant(1 << 30)
_NO_POINT = wp.constant(1 << 30)

_Mat43 = wp.types.matrix(shape=(4, 3), dtype=wp.float32)
_Seen = wp.types.vector(length=TERRAIN_SOLID_SEEN, dtype=wp.int32)
_Sorted = wp.types.vector(length=TERRAIN_SOLID_SORTED, dtype=wp.int32)

TERRAIN_SOLID_COUNTERS = len(TERRAIN_SOLID_JOB_LISTS) + 5
"""Counters of a solid class's jobs: one per job list, then the candidate pool's, the live pairs', the pairs left to
rule R, the scan jobs' and the pairs given to the small-pair finish kernel (zeroed before every launch)."""
_Room = wp.types.vector(length=TERRAIN_SOLID_COUNTERS, dtype=wp.int32)


@wp.struct
class TerrainSolidData:
    """What the solid classes' kernels read besides the pair inputs, the terrain and their launch arguments (their
    class's jobs, its scratch): the hull tables."""

    hulls: TerrainHullTable
    """The hulls of the hull class (empty for the other classes)."""


@wp.struct
class _Sink:
    """A pair's held rows (rule R's), or a job's rows on their way to its class's candidate pool."""

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
    reserve: int
    """1 when a job's row took the candidate pool's reserve."""
    lost: int
    """1 when the candidate pool had no room for a job's row, its reserve included."""


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
    """The window's highest vertex [m, window frame]: a core whose lowest point clears it has no point below the
    surface (rule R)."""


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
    """The core's lowest point [m, window frame] (its bounds' in the window frame)."""


@wp.struct
class TerrainSolidJobs:
    """A solid class's jobs: the scan's runs and records, the job lists and the candidate pool.

    Every list and the pool have fixed capacities (:func:`terrain_solid_jobs`): their room (:attr:`room`, sized from
    the pairs' bounds at the scene level) and a reserve beyond it (:func:`terrain_solid_job_reserves`); their counters
    (``counts``) are zeroed before every launch. Entries are appended in any order, and every row carries its place in
    window order (its key), so that a pair's rows depend on the pair alone. Work past a list's room takes its reserve,
    the same kernels running it: its pair's rows are unchanged, and it is counted and raises the narrow phase's
    ``terrain_jobs`` (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_RESERVE`, :func:`_jobs_overflowed`); a
    pair some of whose work found no room even there keeps no row
    (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_INCOMPLETE`).
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
    """Rule F's jobs: ``(table pair, item)``, ``item`` its window vertex times :data:`TERRAIN_SOLID_ITEMS` plus the
    face's triangle (0 or 1), one per cluster plane a scan job meets (:data:`TERRAIN_SOLID_SEEN`) and per triangle
    alone."""
    edges: wp.array[wp.vec2i]
    """Rule E's jobs: ``(table pair, item)``, ``item`` its window vertex times :data:`TERRAIN_SOLID_ITEMS` plus 2, 3 or
    4 (x-, y-, diagonal edge) or 6 (its corner line)."""
    thetas: wp.array[wp.vec2i]
    """Rule E's creases whose normal arcs can hold a row, for ``theta*``: ``(table pair, item)`` as the edges'."""
    theta_n: wp.array[wp.vec3]
    """Per entry of :attr:`thetas`: its crease's ``theta*`` (:func:`_edge_theta`), for :func:`solid_crease_job`."""
    theta_kind: wp.array[wp.int32]
    """Per entry of :attr:`thetas`: its ``theta*``'s kind (:func:`_edge_theta`: -1 no row, 0 an arc end, 1 inside)."""
    vertices: wp.array[wp.vec2i]
    """Rule V's jobs: ``(table pair, item)``, ``item`` its window vertex times :data:`TERRAIN_SOLID_ITEMS` plus 5."""
    border: wp.array[wp.vec2i]
    """The border's jobs (:func:`solid_border_job`): ``(table pair, item)``, ``item`` its window vertex times
    :data:`TERRAIN_SOLID_ITEMS` plus 7 or 8 (its x- or y-edge): the wall of that border edge, or with
    :data:`_CLIP_JOB` added the clip of its triangle's face."""
    live: wp.array[wp.int32]
    """The class pairs past the whole-pair rejects (class pair indices, in any order): the close's."""
    scans: wp.array[wp.vec2i]
    """The scan's jobs: ``(class pair, run)``, run ``k`` the window vertices ``[k, k + 1)`` times the class's
    :data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN` of the pair's window (room for every pair's
    runs at its largest window)."""
    recover: wp.array[wp.int32]
    """The closed pairs without a row, left to rule R (class pair indices, in any order)."""
    small: wp.array[wp.int32]
    """The closed pairs whose held rows (at most their walk slots) hold a plane group of three, whose finish takes the
    small-pair finish kernel, one thread per pair (class pair indices, in any order)."""
    counts: wp.array[wp.int32]
    """Entries asked for per job list (:data:`TERRAIN_SOLID_JOB_LISTS`), then the candidates asked for, the live pairs,
    the pairs left to rule R, the scan jobs and the pairs given to the small-pair finish kernel
    (:data:`TERRAIN_SOLID_COUNTERS`, zeroed before every launch; past a list's room, the entries that took its reserve
    or found no room)."""
    room: _Room
    """Per counter of :data:`TERRAIN_SOLID_COUNTERS` with a list (the job lists, the candidate pool and the scan's
    runs): its room, its entries past it its reserve (0 for a counter without a list)."""
    threads: int
    """Threads of the job kernels' launches: each strides over its list."""
    border_threads: int
    """Threads of the border kernel's launch (:data:`TERRAIN_SOLID_BORDER_THREADS` on CUDA, the job threads on the CPU,
    at most the border list's room): it strides over the border list."""
    pair_threads: int
    """Threads of the launches that stride over lists of the class's pairs (the close over the live pairs, the
    small-pair finish over its list): the job threads, at most the class's pairs."""
    finish_blocks: int
    """Blocks of rule R's launch, which takes the class's pairs without a row a pair per block at a time
    (``wp.launch_tiled``): :data:`TERRAIN_SOLID_FINISH_BLOCKS` on CUDA, the job threads on the CPU, at most the class's
    pairs."""
    q: wp.array[wp.vec3]
    """The candidate pool: terrain point of every candidate [m, window frame], the jobs' rows (:func:`_offer`)."""
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
def _offer(store: TerrainSolidJobs, sink: _Sink, q: wp.vec3, n: wp.vec3, d: float, feature: int) -> _Sink:
    """Offer a job's row: add it to the candidate pool, in its pair's list (``sink.slot``), with its place in window
    order (``sink.key`` plus its point slot); the close holds them in that order (:func:`solid_close_jobs`). A row past
    the pool's room takes its reserve (``sink.reserve`` set); with no room left there either, the row is lost
    (``sink.lost`` set: the pair keeps no row)."""
    sink.raw = sink.raw + 1
    entry = wp.atomic_add(store.counts, _CANDIDATES, 1)
    if entry >= store.room[_CANDIDATES]:
        sink.reserve = 1
        if entry >= store.key.shape[0]:
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
def _item_edge(item: int) -> int:
    """The edge of a window vertex's item (:data:`TERRAIN_SOLID_ITEMS`): 0, 1, 2 its x-, y- and diagonal edge, 3 the
    corner line below it; -1 for an item that is no edge."""
    if item >= 2 and item <= 4:
        return item - 2
    if item == _ITEM_CORNER:
        return TERRAIN_EDGE_CORNER
    return -1


@wp.func
def _edge_ends(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int
) -> tuple[wp.vec2i, wp.vec2i, wp.vec3, wp.vec3]:
    """An edge's end vertices ``(v0i, v1i, v0, v1)`` (window frame): edge ``which`` of vertex ``(r, c)`` from its lower
    vertex, or for the corner line (``which`` :data:`~newton._src.geometry.terrain_law.TERRAIN_EDGE_CORNER`) from the
    corner vertex down to the walls' base (``v1i`` off the grid)."""
    if which == TERRAIN_EDGE_CORNER:
        v = terrain_window_vertex(data, window, r, c)
        return wp.vec2i(r, c), wp.vec2i(-1, -1), v, wp.vec3(v[0], v[1], v[2] - _WALL_DEPTH)
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    return (
        v0i,
        v1i,
        terrain_window_vertex(data, window, v0i[0], v0i[1]),
        terrain_window_vertex(data, window, v1i[0], v1i[1]),
    )


@wp.func
def _edge_cheap(data: TerrainFeatureData, window: TerrainWindow, pair: _Pair, r: int, c: int, which: int) -> bool:
    """Rule E's cheapest rejects of edge ``which`` of vertex ``(r, c)`` (a corner line below it for ``which``
    :data:`~newton._src.geometry.terrain_law.TERRAIN_EDGE_CORNER`), from its vertices and the core's sphere alone:
    ``False`` for an edge that gives no row (not a crease, below the core, the sphere off the crease's segment). A border
    edge is a ridge between its triangle and its wall."""
    if which < TERRAIN_EDGE_CORNER and terrain_border_side(window.surface, r, c, which) < 0:
        if not terrain_edge_open_crease(data, window, r, c, which):
            # flat inside one cluster: no crease; or a valley in this window's frame by its static class, which the
            # crease tests reject (:func:`_edge_arc`)
            return False
    _v0i, _v1i, v0, v1 = _edge_ends(data, window, r, c, which)
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
    reach (its ends' support values ``sigma_a`` and ``sigma_b``). A border edge's arc runs from its triangle's normal
    to its wall's, a corner line's from one wall's normal to the other's."""
    _v0i, _v1i, v0, v1 = _edge_ends(data, window, r, c, which)
    e = v1 - v0
    length2 = wp.dot(e, e)
    length = wp.sqrt(length2)
    dh = e / length
    toward = pair.sphere_c - v0
    no = wp.vec3(0.0)
    na = wp.vec3(0.0)
    nb = wp.vec3(0.0)
    side = int(-1)
    if which < TERRAIN_EDGE_CORNER:
        side = terrain_border_side(window.surface, r, c, which)
    if which == TERRAIN_EDGE_CORNER:
        sides = terrain_vertex_sides(window.surface, r, c)
        na = terrain_wall_normal(sides[0])
        nb = terrain_wall_normal(sides[1])
    elif side >= 0:
        t = terrain_border_triangle(r, c, which, side)
        planes, na, _pa = terrain_plane(data, window, planes, t[0], t[1], t[2])
        nb = terrain_wall_normal(side)
    else:
        _v0i, _v1i, ta, tb, _wa, _wb = terrain_edge_sides(r, c, which)
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
    _v0i, _v1i, v0, v1 = _edge_ends(data, window, r, c, which)
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
    reference feature, G1). A corner line is one-sided, as its walls are: a core inside both walls there
    (``sigma* < 0`` beyond the band) has no row."""
    _v0i, _v1i, v0, _v1 = _edge_ends(data, window, r, c, which)
    nst, sigma, end = _theta_star(core, n_a, n_b, v0, sigma_a, sigma_b, pair.eps)
    if sigma > pair.reach:
        return nst, -1
    if which == TERRAIN_EDGE_CORNER and sigma < -pair.eps:
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
    v0i, v1i, v0, v1 = _edge_ends(data, window, r, c, which)
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
        if not found:
            # Inside the arc the feature meets the line (theta* is sigma's stationary point), but a single support point
            # that turns fast with the normal (a cylinder's rim near its axis, a barrel's side) the search's rounding
            # leaves off it: within the flatness band.
            found, crossing = core_flat_crossing(core, nst, pair.band, v0, dh, e2, pair.band)
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
def _clip_frame(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int, side: int
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    """Rule F's clip at border edge ``which`` of vertex ``(r, c)`` (wall ``side``): ``(v0, v1, n, a, m)``, the edge's
    ends, its triangle's cluster plane (normal ``n`` through ``a``) and the outward normal ``m`` of the clip plane
    through the edge containing ``n``: the triangle's in-plane outward normal there, as
    :func:`~newton._src.geometry.terrain_law.terrain_face_edge_value` takes it, so that the plane bounds the face's
    claims (a manifold point is supported where its projection lies on the face's side). The clip's planes are
    evaluated afresh, not through the window's plane cache: the border's rare path then keeps the cache's registers out
    of the kernels that hold it beside rule E's."""
    _v0i, _v1i, v0, v1 = _edge_ends(data, window, r, c, which)
    t = terrain_border_triangle(r, c, which, side)
    far = terrain_border_far(r, c, which, side)
    n, a = terrain_cluster_plane(data, window, t[0], t[1], t[2])
    m = wp.cross(v1 - v0, n)
    if wp.dot(m, terrain_window_vertex(data, window, far[0], far[1]) - v0) > 0.0:
        m = -m
    return v0, v1, n, a, wp.normalize(m)


@wp.func
def _clip_cheap(
    core: Any, data: TerrainFeatureData, window: TerrainWindow, pair: _Pair, r: int, c: int, which: int, side: int
) -> bool:
    """Rule F's clip at a border edge past :func:`_edge_cheap`: ``False`` when the core's sphere lies wholly on one side
    of the clip plane (:func:`_clip_frame`; nothing of the core is clipped there, or nothing lies on the face's side) or
    above the face beyond the reach."""
    v0, _v1, n, a, m = _clip_frame(data, window, r, c, which, side)
    across = wp.dot(m, pair.sphere_c - v0)
    if wp.abs(across) > pair.sphere_r + pair.eps:
        return False
    return wp.dot(n, pair.sphere_c - a) - pair.sphere_r <= pair.reach


@wp.func
def _clip_theta(core: Any, n: wp.vec3, m: wp.vec3, v0: wp.vec3):
    """Rule F's clip at a border edge (its face's normal ``n``, its clip plane through ``v0`` of outward normal ``m``,
    :func:`_clip_frame`): ``(core, n_c, kind)``, the normal ``n_c`` on the quarter arc from ``n`` toward ``m`` at which
    the core's support first reaches the clip plane, where its support feature holds the deepest points along ``n`` of
    its part on the face's side: ``kind`` 0 at ``n`` itself (the support there reaches the plane), 1 inside the arc, -1
    when no point of the core lies on the face's side of the plane.

    The depth of that part along ``n`` is ``max over lambda >= 0`` of ``min over the core of (n + lambda m) . (k - v0)``
    (Lagrange duality, up to the face's offset), a concave function of ``lambda`` whose slope is ``m . (k* - v0)`` at
    its support point ``k*``: the bisection (:data:`TERRAIN_SOLID_BISECTION` steps on the arc, ``lambda`` its tangent)
    follows the sign of that slope, the smallest of tied support points
    (:func:`~newton._src.geometry.terrain_law_cores.core_slope_near` along ``m``)."""
    core, slope = core_slope_near(core, n, m, v0, 1.0)
    if slope <= 0.0:
        return core, n, 0
    core, inner = core_sigma_near(core, m, v0)
    if inner >= 0.0:
        return core, m, -1
    lo = n
    hi = m
    for _step in range(_BISECTION):
        mid = wp.normalize(lo + hi)
        core, slope = core_slope_near(core, mid, m, v0, 1.0)
        if slope > 0.0:
            lo = mid
        else:
            hi = mid
    return core, wp.normalize(lo + hi), 1


@wp.func
def _clip_rows_at(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    pair: _Pair,
    sink: _Sink,
    store: Any,
    r: int,
    c: int,
    which: int,
    side: int,
    n_c: wp.vec3,
    kind: int,
) -> _Sink:
    """Rule F's clipped rows at border edge ``which`` of vertex ``(r, c)`` (wall ``side``) at :func:`_clip_theta`'s
    ``n_c``: rows of its triangle's face (normal ``n`` through ``a``, :func:`_clip_frame`) at the ends (0 and 1, or 0
    alone for one point) of the core's part on the face's side of the clip plane deepest along ``n``: the terrain point
    on the edge (the point's projection along ``n``), ``n`` and the point's depth along it.

    The core's support feature at ``n_c`` crossed with the clip plane
    (:func:`~newton._src.geometry.terrain_law_cores.core_flat_crossing` across the plane's normal; inside the arc, a
    single support point the search left off the plane by its rounding, within the flatness band) gives the ends of
    that part's deepest region on the plane, each the core's point on the plane over its place on the edge (its lift
    along ``n``, which a chord's point beyond it by the tolerance or a miss leaves to the chord). An end within this
    edge is its own, an end past a corner of the footprint becomes the core's point on the corner line along ``n``, an
    end past another vertex is the next edge's (no row). Each end's depth along ``n`` is within the reach, and the point
    is front-facing as a face's supported point is (:func:`_face_support`): below the face it lies inside every wall
    and below the surface, above the face not below it. Each core function has one call site here, so that the crease
    kernel, which holds this rare path beside rule E's, keeps fewer registers."""
    if kind < 0:
        return sink
    v0, v1, n, a, m = _clip_frame(data, window, r, c, which, side)
    e = v1 - v0
    length = wp.length(e)
    dh = e / length
    found = bool(False)
    crossing = TerrainCrossing()
    tol = pair.tol
    attempt = int(0)
    while attempt <= kind and not found:
        found, crossing = core_flat_crossing(core, n_c, pair.band, v0, dh, m, tol)
        tol = pair.band
        attempt += 1
    if not found:
        return sink
    ends = 2
    if crossing.hi - crossing.lo < 1.0e-9:
        ends = 1
    v0i, v1i, _v0, _v1 = _edge_ends(data, window, r, c, which)
    for slot in range(ends):
        x = crossing.lo
        k = crossing.k_lo
        if ends == 1:
            x = 0.5 * (crossing.lo + crossing.hi)
            k = 0.5 * (crossing.k_lo + crossing.k_hi)
        elif slot == 1:
            x = crossing.hi
            k = crossing.k_hi
        corner = wp.vec2i(-1, -1)
        if x < -pair.eps:
            corner = v0i
        elif x > length + pair.eps:
            corner = v1i
        u = v0 + x * dh
        if corner[0] >= 0:
            # past a vertex: the next edge's, or at a corner of the footprint the core's point on the corner line
            if terrain_vertex_sides(window.surface, corner[0], corner[1])[1] < 0:
                continue
            u = terrain_window_vertex(data, window, corner[0], corner[1])
        hit, k_lift, t = core_lift(core, u, n, pair.tol)
        if corner[0] >= 0:
            if not hit:
                continue
            k = k_lift
        elif hit and t <= wp.dot(n, k - u) + pair.tol:
            k = k_lift
        h = wp.dot(n, k - a)
        if h > pair.reach:
            continue
        if h < -pair.eps:
            # below the face: inside every wall (the block's footprint; near a corner the point on this edge's clip
            # plane may lie beyond the other wall) and below the surface (the footprint's closure): inside the solid
            if terrain_beyond_walls(window, k) > pair.eps or not terrain_point_below_inside(data, window, k):
                continue
        elif h > pair.eps:
            # above the face: not below the surface, outside the solid
            if terrain_point_below(data, window, k):
                continue
        sink = _offer(
            store,
            sink,
            k - h * n,
            n,
            h,
            terrain_feature_id(r, c, TERRAIN_FEATURE_FACE, TERRAIN_FACE_WALL + which, _CLIP_SLOT + slot),
        )
    return sink


@wp.func
def _wall_cheap(data: TerrainFeatureData, window: TerrainWindow, pair: _Pair, r: int, c: int, which: int, side: int):
    """Rule F's cheapest rejects of the wall below border edge ``which`` of vertex ``(r, c)`` (wall ``side``): ``False``
    when the core's lowest point clears the edge by the reach, or its sphere lies wholly inside the wall's plane (the
    wall is one-sided) or beyond the reach of it."""
    _v0i, _v1i, v0, v1 = _edge_ends(data, window, r, c, which)
    if wp.max(v0[2], v1[2]) < pair.low:
        return False
    beyond = terrain_wall_value(pair.sphere_c, v0, side)
    if beyond + pair.sphere_r < -pair.eps:
        return False
    return beyond - pair.sphere_r <= pair.reach


@wp.func
def _wall_piece(window: TerrainWindow, cells: wp.vec4i, r: int, c: int, side: int, k: wp.vec3) -> wp.vec2i:
    """The window vertex whose border edge's piece of wall ``side`` lies along point ``k`` (the edge's row ``r`` or
    column ``c`` kept, the other coordinate from ``k``, clamped to the window's edges)."""
    surface = window.surface
    if side < 2:
        return wp.vec2i(r, wp.clamp(window.c0 + int(wp.floor(k[0] / surface.dx)), cells[2], cells[3]))
    return wp.vec2i(wp.clamp(window.r0 + int(wp.floor(k[1] / surface.dy)), cells[0], cells[1]), c)


@wp.func
def _wall_support(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    pair: _Pair,
    cells: wp.vec4i,
    r: int,
    c: int,
    which: int,
    side: int,
) -> tuple[int, wp.vec2i, int, _Mat43, wp.vec4]:
    """Rule F's manifold of wall ``side`` (below border edge ``which`` of vertex ``(r, c)``): ``(supported, emitter,
    count, points, depth)``, ``supported`` the bits of the manifold points beside the wall (it is one-sided) whose
    projection a piece of it in the window holds (:func:`~newton._src.geometry.terrain_law.terrain_wall_holds`) and
    ``emitter`` the window vertex of the piece holding the first of them (``(-1, -1)`` for none). A function of the
    wall alone: whichever of its pieces lists it gives the same emitter and rows."""
    _v0i, _v1i, v0, _v1 = _edge_ends(data, window, r, c, which)
    n = terrain_wall_normal(side)
    emitter = wp.vec2i(-1, -1)
    if core_sigma(core, n, v0) > pair.reach:
        return 0, emitter, 0, _Mat43(), wp.vec4(0.0)
    count, points, depth = core_manifold(core, n, v0, pair.reach)
    supported = int(0)
    for j in range(count):
        if depth[j] < -pair.eps:
            continue
        k = wp.vec3(points[j, 0], points[j, 1], points[j, 2])
        piece = _wall_piece(window, cells, r, c, side, k)
        _u0i, _u1i, u0, u1 = _edge_ends(data, window, piece[0], piece[1], which)
        if not terrain_wall_holds(pair.eps, side, u0, u1, k):
            continue
        supported = supported | (1 << j)
        if emitter[0] < 0:
            emitter = piece
    return supported, emitter, count, points, depth


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
    to 32 for a cylinder) would wrap in the slot's three bits. Rule R is clipped to the heightfield's footprint: a point
    outside it has no face above it. The core's part inside it on a border edge's clip plane below the edge's face is
    rule F's clip row (:func:`_clip_rows_at`), so that a pair left to rule R has none.
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
    """Whether window vertex ``(r, c)`` lists nothing for rules E and V and no border feature in any window: it is
    interior, its three edges are flat (two-bit codes 1 or 2 of its bits, whose bits then differ; an interior vertex's
    edges have both triangles) and it is no vertex feature."""
    bits = terrain_vertex_bits(data, surface, r, c)
    return (
        ((bits ^ (bits >> 1)) & 21) == 21
        and (bits & TERRAIN_VERTEX_FEATURE_BIT) == 0
        and terrain_interior_vertex(surface, r, c)
    )


@wp.func
def _window_extent_top(
    data: TerrainFeatureData, window: TerrainWindow, r1: int, c1: int, extent: float
) -> tuple[float, float, bool]:
    """:func:`~newton._src.geometry.terrain_law.terrain_window_extent` (the same value, bit for bit) and, read in the
    same pass, the window's highest vertex (window frame), which the close's rule R test takes, and whether every window
    vertex is quiet (:func:`_vertex_quiet`, read until one is not)."""
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
def solid_finish(
    window: TerrainWindow,
    pair: _Pair,
    slot: int,
    held: int,
    out: TerrainLawRows,
    pair_index: int,
    frame: wp.transform,
    cells: wp.vec4i,
):
    """Finish a pair's ``held`` distinct rows in its walk slots from ``slot`` in place
    (:func:`~newton._src.geometry.terrain_law.terrain_finish_slots`) and write the kept rows (world frame) at the front
    of its slots."""
    # the held rows are distinct: every one was held with terrain_hold_row at the pair's eps
    keep, kept = terrain_finish_slots(out.q, out.n, out.d, slot, held, pair.eps, True)
    written = int(0)
    for i in range(held):
        if ((keep >> i) & 1) != 0:
            s = slot + i
            terrain_store_row(out, slot + written, out.q[s], out.n[s], out.d[s], out.feature[s], window, frame)
            written += 1
    out.first[pair_index] = slot
    out.count[pair_index] = kept
    out.state[pair_index] = _wide(cells)


# ----------------------------------------------------------------------------------------------------------------
# the job pipeline: a scan per pair, a thread per listed feature, a close per pair
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _job_vertex(cells: wp.vec4i, vertex: int) -> wp.vec2i:
    """Window vertex ``vertex`` (window order: row by row from ``(r0, c0)``) of a window of cells ``cells``, as
    ``(r, c)``."""
    width = cells[3] + 2 - cells[2]
    row = vertex // width
    return wp.vec2i(cells[0] + row, cells[2] + vertex - row * width)


@wp.func
def _job_reserve(
    jobs: TerrainSolidJobs, out: TerrainLawRows, pair_index: int, kind: int, entries: wp.array[wp.vec2i], count: int
) -> tuple[int, int]:
    """Reserve ``count`` consecutive entries of table pair ``pair_index``'s jobs in a list (counter ``kind``) with one
    atomic: ``(first, kept)``, ``kept`` 0 when some of them found no room (they are still counted). Entries past the
    list's room take its reserve and flag the pair
    (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_RESERVE`)."""
    if count == 0:
        return 0, 1
    first = wp.atomic_add(jobs.counts, kind, count)
    if first + count > jobs.room[kind]:
        wp.atomic_or(out.state, pair_index, TERRAIN_PAIR_RESERVE)
        if first + count > entries.shape[0]:
            return first, 0
    return first, 1


@wp.func
def _job_append(
    jobs: TerrainSolidJobs, out: TerrainLawRows, pair_index: int, kind: int, entries: wp.array[wp.vec2i], job: wp.vec2i
) -> int:
    """Append a job of table pair ``pair_index`` to a list (counter ``kind``, :func:`_job_reserve`): 0 when it found no
    room."""
    entry, kept = _job_reserve(jobs, out, pair_index, kind, entries, 1)
    if kept != 0:
        entries[entry] = job
    return kept


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
    vertices' features in window order through the cheapest rejects of their rules, every survivor listed for its job
    kernel: rule F's faces (:func:`_face_top`; of a cluster of several triangles, the first that passes, its cluster's
    manifold and emitting triangle being its cluster's alone) and walls (:func:`_wall_cheap`; each wall once, its
    manifold and emitting piece being its own), rule E's edges, border edges among them, and corner lines
    (:func:`_edge_cheap`) and rule V's vertices (:func:`_vertex_cheap`). The run marks its survivors in one bit per
    vertex and item, then reserves
    each list's entries with one atomic and writes them in window order (:func:`_job_reserve`): consecutive entries of a
    list are one run's, so that the threads of a job kernel's warp take one pair's features. A job without room, the
    list's reserve included, flags the pair :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_INCOMPLETE` (its
    close then leaves it without a row)."""
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
    wall0 = wp.uint32(0)
    wall1 = wp.uint32(0)
    edge0 = wp.uint32(0)
    edge1 = wp.uint32(0)
    edge2 = wp.uint32(0)
    corner = wp.uint32(0)
    point = wp.uint32(0)
    faces = int(0)
    edges = int(0)
    points = int(0)
    walls = int(0)
    walls_seen = int(0)
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
            if which < 2:
                side = terrain_border_side(surface, r, c, which)
                if side >= 0 and ((walls_seen >> side) & 1) == 0:
                    if _wall_cheap(data, window, pair, r, c, which, side):
                        walls_seen = walls_seen | (1 << side)
                        if which == 0:
                            wall0 = _with_bit(wall0, v)
                        else:
                            wall1 = _with_bit(wall1, v)
                        walls += 1
        if terrain_vertex_sides(surface, r, c)[1] >= 0:
            if _edge_cheap(data, window, pair, r, c, TERRAIN_EDGE_CORNER):
                corner = _with_bit(corner, v)
                edges += 1
        core, listed = _vertex_cheap(core, data, window, pair, r, c)
        if listed:
            point = _with_bit(point, v)
            points += 1
    entry, kept = _job_reserve(jobs, out, pair_index, _FACES, jobs.faces, faces)
    if faces > 0:
        for v in range(last - first):
            item = (first + v) * _ITEMS
            if _bit(face0, v):
                entry = _job_put(jobs.faces, entry, wp.vec2i(pair_index, item))
            if _bit(face1, v):
                entry = _job_put(jobs.faces, entry, wp.vec2i(pair_index, item + 1))
    entry, room = _job_reserve(jobs, out, pair_index, _BORDER, jobs.border, walls)
    kept = kept & room
    if walls > 0:
        for v in range(last - first):
            item = (first + v) * _ITEMS + _ITEM_BORDER
            if _bit(wall0, v):
                entry = _job_put(jobs.border, entry, wp.vec2i(pair_index, item))
            if _bit(wall1, v):
                entry = _job_put(jobs.border, entry, wp.vec2i(pair_index, item + 1))
    entry, room = _job_reserve(jobs, out, pair_index, _EDGES, jobs.edges, edges)
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
            if _bit(corner, v):
                entry = _job_put(jobs.edges, entry, wp.vec2i(pair_index, item + _ITEM_CORNER))
    entry, room = _job_reserve(jobs, out, pair_index, _VERTICES, jobs.vertices, points)
    kept = kept & room
    if points > 0:
        for v in range(last - first):
            if _bit(point, v):
                entry = _job_put(jobs.vertices, entry, wp.vec2i(pair_index, (first + v) * _ITEMS + _ITEM_VERTEX))
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
    """Record a pair some of whose work found no room in its class's job lists or candidate pool (it took their reserve,
    its rows unchanged, or it lost work and keeps no row): the narrow phase's sticky ``terrain_jobs`` bit
    (:data:`~newton._src.geometry.terrain_law.TERRAIN_STATUS_JOBS`) and the side buffer's job overflow counter
    (``overflow[2]``, accumulated over collides)."""
    wp.atomic_or(out.status, 0, TERRAIN_STATUS_JOBS)
    wp.atomic_add(out.overflow, 2, 1)


@wp.func
def _job_flags(out: TerrainLawRows, pair_index: int, sink: _Sink):
    """Flag the pair of a job whose rows took the candidate pool's reserve
    (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_RESERVE`) or lost a row, no room left there either
    (:data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_INCOMPLETE`: the pair keeps no row)."""
    if sink.reserve != 0:
        wp.atomic_or(out.state, pair_index, TERRAIN_PAIR_RESERVE)
    if sink.lost != 0:
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
    _job_flags(out, pair_index, sink)


@wp.func
def solid_wall_job(
    core: Any,
    data: TerrainFeatureData,
    window: TerrainWindow,
    record: TerrainSolidRecord,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    r: int,
    c: int,
    which: int,
):
    """Rule F for a listed wall (below border edge ``which`` of vertex ``(r, c)``): its manifold, support tests and
    emitting piece (:func:`_wall_support`), and its rows at that piece's place in window order (its vertex's item 7 or
    8), when its edge is not below the pair's ``low``: terrain points on the wall, its normal, the manifold points'
    distances beyond it."""
    cells = record.cells
    pair = record.pair
    side = terrain_border_side(window.surface, r, c, which)
    supported, emitter, count, points, depth = _wall_support(core, data, window, pair, cells, r, c, which, side)
    if supported == 0:
        return
    _u0i, _u1i, u0, u1 = _edge_ends(data, window, emitter[0], emitter[1], which)
    if wp.max(u0[2], u1[2]) < pair.low:
        return
    width = cells[3] + 2 - cells[2]
    item = ((emitter[0] - cells[0]) * width + emitter[1] - cells[2]) * _ITEMS + _ITEM_BORDER + which
    sink = _job_sink(i, item, pair.eps)
    n = terrain_wall_normal(side)
    for j in range(count):
        if ((supported >> j) & 1) != 0:
            k = wp.vec3(points[j, 0], points[j, 1], points[j, 2])
            sink = _offer(
                jobs,
                sink,
                k - depth[j] * n,
                n,
                depth[j],
                terrain_feature_id(emitter[0], emitter[1], TERRAIN_FEATURE_FACE, TERRAIN_FACE_WALL + which, j),
            )
    _job_flags(out, pair_index, sink)


@wp.func
def solid_edge_job(
    core: Any, data: TerrainFeatureData, out: TerrainLawRows, jobs: TerrainSolidJobs, pair_index: int, i: int, item: int
):
    """Rule E's crease tests of a listed edge (:func:`_edge_arc`) and its pre-bounds (:func:`_edge_reach`); a convex
    crease whose arc can hold a row within reach is listed for ``theta*`` (:func:`solid_theta_job`). A border edge is
    also listed for the border's jobs, for rule F's clip of its triangle's face, when the core may lie across the clip
    plane (:func:`_clip_cheap`; item :data:`TERRAIN_SOLID_ITEMS` 7 or 8 with :data:`_CLIP_JOB`,
    :func:`solid_border_job`)."""
    record, window = solid_job_window(data, jobs, i)
    vertex = item // _ITEMS
    rc = _job_vertex(record.cells, vertex)
    which = _item_edge(item - vertex * _ITEMS)
    planes = terrain_planes()
    if which < TERRAIN_EDGE_CORNER:
        side = terrain_border_side(window.surface, rc[0], rc[1], which)
        if side >= 0:
            if _clip_cheap(core, data, window, record.pair, rc[0], rc[1], which, side):
                job = wp.vec2i(pair_index, vertex * _ITEMS + _ITEM_BORDER + which + _CLIP_JOB)
                if _job_append(jobs, out, pair_index, _BORDER, jobs.border, job) == 0:
                    wp.atomic_or(out.state, pair_index, TERRAIN_PAIR_INCOMPLETE)
    planes, ok, n_a, n_b, _sigma_a, _sigma_b = _edge_arc(core, data, window, planes, record.pair, rc[0], rc[1], which)
    if not ok or not _edge_reach(core, data, window, record.pair, rc[0], rc[1], which, n_a, n_b):
        return
    if _job_append(jobs, out, pair_index, _THETAS, jobs.thetas, wp.vec2i(pair_index, item)) == 0:
        wp.atomic_or(out.state, pair_index, TERRAIN_PAIR_INCOMPLETE)


@wp.func
def solid_theta_job(core: Any, data: TerrainFeatureData, jobs: TerrainSolidJobs, i: int, item: int, entry: int):
    """Rule E's ``theta*`` of a crease whose arc can hold a row (entry ``entry`` of :attr:`TerrainSolidJobs.thetas`):
    its arc again (:func:`_edge_arc`, the edge job's, bit for bit) and :func:`_edge_theta`, kept for
    :func:`solid_crease_job`."""
    record, window = solid_job_window(data, jobs, i)
    pair = record.pair
    vertex = item // _ITEMS
    rc = _job_vertex(record.cells, vertex)
    planes = terrain_planes()
    which = _item_edge(item - vertex * _ITEMS)
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
    rc = _job_vertex(record.cells, vertex)
    planes = terrain_planes()
    sink = _job_sink(i, item, pair.eps)
    planes, sink = _edge_rows_at(
        core,
        data,
        window,
        planes,
        pair,
        sink,
        jobs,
        rc[0],
        rc[1],
        _item_edge(item - vertex * _ITEMS),
        jobs.theta_n[entry],
        kind,
    )
    _job_flags(out, pair_index, sink)


@wp.func
def solid_border_job(
    core: Any, data: TerrainFeatureData, out: TerrainLawRows, jobs: TerrainSolidJobs, pair_index: int, i: int, job: int
):
    """The border's job ``job`` of a listed pair (:attr:`TerrainSolidJobs.border`), in one thread: the rows of a wall
    (:func:`solid_wall_job`, listed by the scan) or of a border edge's clip of its triangle's face (listed by the edge
    job, :data:`_CLIP_JOB` added to its item): its normal on the quarter arc (:func:`_clip_theta`) and its rows
    (:func:`_clip_rows_at`). Its own kernel, so that the face, theta and crease kernels hold none of the border's rare
    paths (a clip's support searches and lift, a wall's manifold)."""
    item = job
    if job >= _CLIP_JOB:
        item = job - _CLIP_JOB
    record, window = solid_job_window(data, jobs, i)
    vertex = item // _ITEMS
    which = item - vertex * _ITEMS - _ITEM_BORDER
    rc = _job_vertex(record.cells, vertex)
    if job < _CLIP_JOB:
        solid_wall_job(core, data, window, record, out, jobs, pair_index, i, rc[0], rc[1], which)
        return
    pair = record.pair
    side = terrain_border_side(window.surface, rc[0], rc[1], which)
    v0, _v1, n, _a, m = _clip_frame(data, window, rc[0], rc[1], which, side)
    # the arc's search may update its own copy of the core (a hull's support cache); the rows take the core as listed
    _searched, n_c, kind = _clip_theta(core, n, m, v0)
    if kind < 0:
        return
    sink = _job_sink(i, item, pair.eps)
    sink = _clip_rows_at(core, data, window, pair, sink, jobs, rc[0], rc[1], which, side, n_c, kind)
    _job_flags(out, pair_index, sink)


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
    _job_flags(out, pair_index, sink)


@wp.func
def _hold_candidate(
    out: TerrainLawRows, jobs: TerrainSolidJobs, entry: int, slot: int, held: int, eps: float, limit: int
) -> int:
    """Hold a pool candidate in a pair's walk slots from ``slot``, merged as the finish merges
    (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`, at most ``limit``)."""
    return terrain_hold_row(
        out, slot, held, jobs.q[entry], jobs.n[entry], jobs.d[entry], jobs.feature[entry], eps, limit
    )


@wp.func
def _hold_candidate(
    scratch: TerrainFinishScratch, jobs: TerrainSolidJobs, entry: int, slot: int, held: int, eps: float, limit: int
) -> int:
    """Place a pool candidate after the ``held`` ones of a scratch block from ``slot``, with its core point ``q + d n``
    (the row's definition, as the in-place finish reads it), unmerged: the finish of deferred pairs merges
    (:func:`~newton._src.geometry.terrain_law.terrain_place_candidate`, up to the block's capacity; ``eps`` and
    ``limit`` serve the walk slots' overload)."""
    q = jobs.q[entry]
    n = jobs.n[entry]
    d = jobs.d[entry]
    return terrain_place_candidate(scratch, slot, held, q, q + d * n, n, d, jobs.feature[entry])


@wp.func
def _hold_few(store: Any, jobs: TerrainSolidJobs, i: int, slot: int, limit: int, eps: float) -> tuple[bool, int, int]:
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
        held = _hold_candidate(store, jobs, e0, slot, held, eps, limit)
    if count > 1 and k1 != k0:
        raw += 1
        held = _hold_candidate(store, jobs, e1, slot, held, eps, limit)
    if count > 2 and k2 != k1:
        raw += 1
        held = _hold_candidate(store, jobs, e2, slot, held, eps, limit)
    if count > 3 and k3 != k2:
        raw += 1
        held = _hold_candidate(store, jobs, e3, slot, held, eps, limit)
    return True, held, raw


@wp.func
def _hold_candidates(store: Any, jobs: TerrainSolidJobs, i: int, slot: int, limit: int, eps: float) -> tuple[int, int]:
    """Hold class pair ``i``'s candidates from ``slot`` of ``store`` in window order (by key), as the walk offers them:
    ``(held, raw)``, its rows held and its offers. In a pair's walk slots
    (:class:`~newton._src.geometry.terrain_law.TerrainLawRows`) they merge as the finish merges
    (:func:`~newton._src.geometry.terrain_law.terrain_hold_row`, at most ``limit``), so that ``held`` counts its
    distinct rows; in a scratch block of the finish of deferred pairs
    (:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`) each is placed as it is, the finish merging them
    (:func:`_hold_candidate`). Candidates of one key (a cluster listed twice) are one offer; the list's order (its
    jobs') never matters. A list of at most :data:`TERRAIN_SOLID_FEW` candidates (most pairs' on the recorded states)
    is sorted in registers (:func:`_hold_few`), a longer one in chunks on the stack (:func:`_hold_sorted`)."""
    few, held, raw = _hold_few(store, jobs, i, slot, limit, eps)
    if few:
        return held, raw
    return _hold_sorted(store, jobs, i, slot, limit, eps)


@wp.func
def _hold_sorted(store: Any, jobs: TerrainSolidJobs, i: int, slot: int, limit: int, eps: float) -> tuple[int, int]:
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
            held = _hold_candidate(store, jobs, entries[j], slot, held, eps, limit)
    return held, raw


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
def _finish_or_hold(
    window: TerrainWindow,
    record: TerrainSolidRecord,
    slot: int,
    held: int,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    frame: wp.transform,
):
    """Finish a pair holding ``held`` distinct rows (at most its walk slots) in its walk slots from ``slot``: here, by
    this thread, when no plane group holds three of them
    (:func:`~newton._src.geometry.terrain_law.terrain_plane_group_of_three`), which the finish keeps as they are: they
    are written in the world frame in place (:func:`_store_held`); any other pair by the small-pair finish kernel
    (state :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_HELD`, the class's small list), so that the close and
    rule R's kernel hold none of the general finish's code and state."""
    if not terrain_plane_group_of_three(out, slot, held, record.pair.eps):
        _store_held(window, record.cells, slot, held, out, pair_index, frame)
        return
    out.first[pair_index] = slot
    out.count[pair_index] = held
    out.state[pair_index] = _wide(record.cells) | TERRAIN_PAIR_HELD
    # the small list holds every pair of the class at most
    jobs.small[wp.atomic_add(jobs.counts, _SMALL, 1)] = i


@wp.func
def solid_close_jobs(
    data: TerrainFeatureData,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    slot: int,
    walk_slots: int,
    partner_class: int,
    frame: wp.transform,
):
    """Close a scanned pair (class pair ``i``, its walk slots from ``slot``): its candidates held in window order
    (:func:`_hold_candidates`), which gives it the walk's held rows. A pair holding more distinct rows than its slots is
    left to the class's finish of deferred pairs (:func:`solid_finish_deferred`); a pair without a row to rule R
    (:func:`solid_recover_pair`, its list); any other is finished (:func:`_finish_or_hold`). A pair some of whose work
    took its class's job reserve is counted (:func:`_jobs_overflowed`); one some of whose work found no room even
    there keeps no row."""
    state = out.state[pair_index]
    if (state & TERRAIN_PAIR_INCOMPLETE) != 0:
        # Some of its work found no room, the reserve included (its record may be missing: a window past its bound
        # lists no first run): its candidates are incomplete, and it keeps no row.
        terrain_pair_exit(out, pair_index, state & (TERRAIN_PAIR_WIDE | TERRAIN_PAIR_INCOMPLETE))
        _jobs_overflowed(out)
        return
    if (state & TERRAIN_PAIR_RESERVE) != 0:
        # some of its work took the reserve, its rows unchanged
        _jobs_overflowed(out)
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
            # The core lies above every vertex of its window: rule R finds none of its points below the surface, so
            # the pair keeps no row.
            out.first[pair_index] = slot
            out.count[pair_index] = 0
            out.state[pair_index] = _wide(cells)
            return
        # the recovery list holds every pair of the class at most
        jobs.recover[wp.atomic_add(jobs.counts, _RECOVER, 1)] = i
        return
    _finish_or_hold(window, record, slot, held, out, jobs, pair_index, i, frame)


@wp.func
def solid_recover_pair(
    core: Any,
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
    """Rule R for a closed pair without a row (class pair ``i``, its walk slots from ``slot``) by a block's ``lanes``
    lanes (lane ``lane``, :func:`solid_recovery`), then its finish by the first lane (:func:`_finish_or_hold`)."""
    record, window = solid_job_window(data, jobs, i)
    sink = solid_sink(slot, walk_slots, record.pair.eps)
    planes = terrain_planes()
    planes, sink = solid_recovery(core, data, window, planes, sink, out, lane, lanes)
    if lane == 0:
        _finish_or_hold(window, record, sink.slot, sink.held, out, jobs, pair_index, i, frame)


@wp.func
def solid_finish_pair(
    data: TerrainFeatureData,
    out: TerrainLawRows,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    slot: int,
    frame: wp.transform,
):
    """The finish of a closed pair holding its rows for it (class pair ``i``, its ``out.count`` rows in its walk slots
    from ``slot``, :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_HELD`, from its close or rule R), in place
    (:func:`solid_finish`)."""
    record, window = solid_job_window(data, jobs, i)
    solid_finish(window, record.pair, slot, out.count[pair_index], out, pair_index, frame, record.cells)


@wp.func
def _place_deferred(
    scratch: TerrainFinishScratch, jobs: TerrainSolidJobs, i: int, base: int, eps: float, lane: int, lanes: int
) -> int:
    """Place class pair ``i``'s candidates in a scratch block from ``base`` in window order, one per key, as
    :func:`_hold_candidates` places them, with a block's ``lanes`` lanes (lane ``lane``; one thread passes ``0, 1``):
    the first lane walks the pair's pool list into the block's state words (an entry's key and index), then each lane
    takes every ``lanes``-th entry, and the first of its key in the list places its candidate at its key's place among
    the list's keys. Returns the candidates placed (the first lane's count is the one read). A list longer than the
    block (a window its shape data no longer bounds) is placed by the first lane alone (:func:`_hold_candidates`)."""
    length = int(0)
    if lane == 0:
        entry = jobs.head[i]
        while entry >= 0:
            if length < scratch.capacity:
                scratch.alive[base + length] = jobs.key[entry]
                scratch.group[base + length] = entry
            length += 1
            entry = jobs.next[entry]
    if lanes > 1:
        # every lane takes the first lane's length, after its writes
        length = wp.tile_extract(wp.tile(length), 0)
    if length > scratch.capacity:
        held = int(0)
        if lane == 0:
            held, _raw = _hold_candidates(scratch, jobs, i, base, scratch.capacity, eps)
        return held
    # the first entry of each key in the list (its candidates are one offer)
    for t in range(lane, length, lanes):
        key = scratch.alive[base + t]
        first = int(1)
        for u in range(t):
            if scratch.alive[base + u] == key:
                first = 0
        scratch.rank[base + t] = first
    if lanes > 1:
        # every lane reads the others' first entries
        wp.tile_extract(wp.tile_max(wp.tile(0)), 0)
    held = int(0)
    for t in range(lane, length, lanes):
        if scratch.rank[base + t] == 0:
            continue
        key = scratch.alive[base + t]
        place = int(0)
        for u in range(length):
            if scratch.rank[base + u] != 0 and scratch.alive[base + u] < key:
                place += 1
        entry = scratch.group[base + t]
        q = jobs.q[entry]
        n = jobs.n[entry]
        d = jobs.d[entry]
        terrain_place_candidate(scratch, base, place, q, q + d * n, n, d, jobs.feature[entry])
        held += 1
    if lanes > 1:
        held = wp.tile_extract(wp.tile_sum(wp.tile(held)), 0)
    return held


@wp.func
def solid_finish_deferred(
    data: TerrainFeatureData,
    out: TerrainLawRows,
    scratch: TerrainFinishScratch,
    jobs: TerrainSolidJobs,
    pair_index: int,
    i: int,
    slot: int,
    walk_slots: int,
    frame: wp.transform,
    block: int,
    lane: int,
    lanes: int,
):
    """Finish a deferred pair (class pair ``i``: more distinct rows than its ``walk_slots`` walk slots from ``slot``)
    with a block's ``lanes`` lanes (lane ``lane``; one thread passes ``0, 1``): its candidates placed from the candidate
    pool into block ``block``'s scratch in window order, one per key as its close takes them (:func:`_place_deferred`;
    the scratch holds every candidate a window of the class can give), then merged and finished there by every lane and
    written (:func:`~newton._src.geometry.terrain_law.terrain_finish_deferred`)."""
    record, window = solid_job_window(data, jobs, i)
    pair = record.pair
    base = block * scratch.capacity
    held = _place_deferred(scratch, jobs, i, base, pair.eps, lane, lanes)
    terrain_finish_deferred(
        scratch,
        base,
        held,
        pair.eps,
        out,
        pair_index,
        slot,
        walk_slots,
        _wide(record.cells),
        window,
        frame,
        lane,
        lanes,
    )


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
    ("border_jobs", "border"),
    ("pair_jobs", "close"),
    ("pair_tiles", "recovery"),
    ("pair_jobs", "small"),
    ("deferred", "finish"),
)
"""A solid class's launches in order, by kind and kernel (:func:`solid_kernels`): ``pairs`` one thread per pair of the
class, ``jobs`` the class's job threads (:attr:`TerrainSolidJobs.threads`, striding over a list; a thread past them
returns at once, so that a larger launch takes no entry twice), ``border_jobs`` the same with the border kernel's
threads (:attr:`TerrainSolidJobs.border_threads`), ``pair_jobs`` the same over a list of the class's pairs
with at most one thread per pair (:attr:`TerrainSolidJobs.pair_threads`), ``pair_tiles`` blocks of lanes striding over
a list of the class's pairs, one pair per block at a time (:attr:`TerrainSolidJobs.finish_blocks`, ``wp.launch_tiled``),
``deferred`` blocks of lanes striding over the class's deferred list on the fixed grid of its scratch
(:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`, ``wp.launch_tiled``)."""


def solid_kernels(
    name: str, partner_class: int, walk_slots: int, scan_run: int, make_core, save_core, load_core, module: str
) -> dict:
    """Build one solid class's kernels in module ``module`` (whose options, contraction off, they take), named
    ``terrain_<name>_rows`` (the scan) and ``terrain_<name>_rows_<kernel>`` for the others of
    :data:`TERRAIN_SOLID_LAUNCHES`.

    Every kernel takes ``(begin, slot_begin, inputs, data, out)`` and the class's
    :class:`TerrainSolidJobs` and the solid classes' :class:`TerrainSolidData` (the finish of deferred pairs: its
    :class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`, the data, then the jobs): table pair ``begin + i``
    is the class's pair ``i``, its walk slots ``slot_begin + i * walk_slots``.

    Args:
        name: The class's name in kernel names.
        partner_class: The class (``TERRAIN_PARTNER_*``).
        walk_slots: Walk slots of a pair of the class.
        scan_run: Window vertices of one scan job of the class
            (:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`), at most
            :data:`TERRAIN_SOLID_SCAN_VERTICES`.
        make_core: ``wp.func`` ``(local, size, kind, partner, solids) -> core``: a pair's core in the heightfield
            frame.
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
        """The whole-pair rejects, one thread per pair of the class (stale features, the pair policy, a window off the
        heightfield, the tile-max exit): a pair past them keeps its core and record (window frame) for its scan and
        jobs, is listed live (:attr:`TerrainSolidJobs.live`) for the close, its window's runs
        for the scan (:attr:`TerrainSolidJobs.scans`), and takes state
        :data:`~newton._src.geometry.terrain_law.TERRAIN_PAIR_JOBS`; any other is recorded with no row."""
        i = wp.tid()
        pair = begin + i
        state, _frame, heightfield, local, partner, size, reach = solid_pair_read(inputs, data, pair)
        if state != 0:
            terrain_pair_exit(out, pair, state)
            return
        core = wp.static(make_core)(local, size, inputs.shape_type[partner], partner, solids)
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
                if _job_append(jobs, out, pair, _FACES, jobs.faces, wp.vec2i(pair, 0)) == 0:
                    wp.atomic_or(out.state, pair, TERRAIN_PAIR_INCOMPLETE)
            return
        vertices = (cells[1] - cells[0] + 2) * (cells[3] - cells[2] + 2)
        runs = (vertices + scan_ - 1) // scan_
        # Past the scan's room (every pair's runs at its largest window) only a window larger than its shape data
        # bounded at construction: its runs take the reserve.
        entry, kept = _job_reserve(jobs, out, pair, _SCANS, jobs.scans, runs)
        for k in range(runs):
            entry = _job_put(jobs.scans, entry, wp.vec2i(i, k))
        if kept == 0:
            wp.atomic_or(out.state, pair, TERRAIN_PAIR_INCOMPLETE)

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

    def border(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """The border's walls and clipped faces, one listed job per thread at a time (:func:`solid_border_job`), on the
        border kernel's grid (:attr:`TerrainSolidJobs.border_threads`)."""
        t = wp.tid()
        if t >= jobs.border_threads:
            return
        for entry in range(t, wp.min(jobs.counts[_BORDER], jobs.border.shape[0]), jobs.border_threads):
            job = jobs.border[entry]
            i = job[0] - begin
            solid_border_job(wp.static(load_core)(jobs, i, solids), data, out, jobs, job[0], i, job[1])

    def close(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        jobs: TerrainSolidJobs,
        solids: TerrainSolidData,
    ):
        """The close, one live pair per thread at a time: a scanned pair's rows held in window order and finished, held
        for the small-pair finish kernel or listed for rule R (:func:`solid_close_jobs`); a pair the scan deferred is
        passed by."""
        t = wp.tid()
        if t >= jobs.pair_threads:
            return
        for entry in range(t, jobs.counts[_LIVE], jobs.pair_threads):
            i = jobs.live[entry]
            pair = begin + i
            if (out.state[pair] & TERRAIN_PAIR_JOBS) != 0:
                solid_close_jobs(
                    data,
                    out,
                    jobs,
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
        the pair's core points, and the first lane offers its rows and finishes it or holds them for a finish kernel
        (:func:`solid_recover_pair`)."""
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
                pair,
                i,
                slot_begin + i * rows_,
                rows_,
                class_,
                inputs.shape_transform[inputs.pairs[pair][0]],
                lane,
                lanes,
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
        """The finish of the class's pairs whose held rows hold a plane group of three (its small list, from its close
        and rule R), one pair per thread at a time, so that a warp runs the general finish on as many pairs as it has
        lanes and no other kernel holds its code and stack (:func:`solid_finish_pair`)."""
        t = wp.tid()
        if t >= jobs.pair_threads:
            return
        for entry in range(t, jobs.counts[_SMALL], jobs.pair_threads):
            i = jobs.small[entry]
            pair = begin + i
            solid_finish_pair(
                data, out, jobs, pair, i, slot_begin + i * rows_, inputs.shape_transform[inputs.pairs[pair][0]]
            )

    def finish(
        begin: int,
        slot_begin: int,
        inputs: TerrainPairInputs,
        data: TerrainFeatureData,
        out: TerrainLawRows,
        scratch: TerrainFinishScratch,
        solids: TerrainSolidData,
        jobs: TerrainSolidJobs,
    ):
        """The finish of the class's deferred pairs (more distinct rows than their walk slots), ``wp.block_dim()``
        lanes per pair (:func:`solid_finish_deferred`): the grid's ``scratch.blocks`` blocks stride over the class's
        deferred list, each block taking its pairs one at a time into its scratch, which holds every candidate a
        window of the class can give, so that a pair's rows depend on the pair alone: not on the order of the
        deferral, the number of deferred pairs or the block that finishes it. ``wp.launch_tiled`` with
        ``dim=scratch.blocks``; a block past it returns at once."""
        block, lane = wp.tid()
        if block >= scratch.blocks:
            return
        lanes = wp.block_dim()
        for entry in range(block, out.deferred_count[class_], scratch.blocks):
            pair = out.deferred[begin + entry]
            i = pair - begin
            solid_finish_deferred(
                data,
                out,
                scratch,
                jobs,
                pair,
                i,
                slot_begin + i * rows_,
                rows_,
                inputs.shape_transform[inputs.pairs[pair][0]],
                block,
                lane,
                lanes,
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
        ("border", border),
        ("close", close),
        ("recovery", recovery),
        ("small", small),
        ("finish", finish),
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


def terrain_solid_data(model, partners: np.ndarray) -> TerrainSolidData:
    """The solid classes' data for a model: the hull table of the convex-mesh partners in ``partners`` (shape
    indices). Their launch arguments, each class's jobs and scratch, are the shadow's
    (:func:`terrain_solid_jobs`, :func:`~newton._src.geometry.terrain_law.terrain_finish_scratch`).

    Args:
        model: A finalized model.
        partners: Partner shapes of the solid classes' pairs.
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
    return solids


def _job_bounds(vertices, sides, scan_run: int) -> tuple:
    """Per job list (:data:`TERRAIN_SOLID_JOB_LISTS`), a pair's bound from its window's vertices and sides: two faces per
    window vertex, three edges per window vertex and four corner lines, three creases per window vertex and four corner
    lines, one vertex per window vertex, and for the border four walls per scan run and the clips of the border edges
    along its sides (two per vertex there)."""
    v = np.asarray(vertices, np.int64)
    runs = -(-v // int(scan_run))
    border = 2 * np.asarray(sides, np.int64)
    return (2 * v, 3 * v + 4, 3 * v + 4, v, 4 * runs + border)


def terrain_solid_job_capacities(
    vertices: np.ndarray,
    candidates: np.ndarray,
    walk_slots: int,
    *,
    scan_run: int,
    sides: np.ndarray,
    scale: float = 1.0,
) -> dict[str, int]:
    """Return a solid class's job list rooms by default, from its pairs' own bounds: per list
    (:data:`TERRAIN_SOLID_JOB_LISTS`) the largest pair's bound plus the sum of every pair's bound over the list's share
    (:data:`TERRAIN_SOLID_JOB_SHARES`), the candidate pool's rows (``candidates``): the largest pair's bound on its
    candidates plus the class's walk slots over :data:`TERRAIN_SOLID_CANDIDATE_SHARE`; at least the sum of every pair's
    bound up to :data:`TERRAIN_SOLID_JOB_FLOOR` (:data:`TERRAIN_SOLID_POOL_FLOOR` for the pool's rows), so that a scene
    with few pairs never lacks room; each times ``scale``. The scan's runs (``scans``) are exact: every pair's at its
    largest window (:data:`TERRAIN_SOLID_SCAN_VERTICES` vertices each). Every list is a scene-level buffer, as
    ``rigid_contact_max`` sizes contacts, with a reserve beyond its room (:func:`terrain_solid_job_reserves`): work
    that finds no room takes the reserve (its pair's rows unchanged), raises the narrow phase's ``terrain_jobs`` and
    is counted (:meth:`~newton._src.geometry.terrain_shadow.TerrainLawShadow.overflow_counts`), and the job check
    prints the entries asked for against the room. The shares are defaults with room to spare on the recorded suite
    states (16,384 worlds): there the most used list holds 43% of its room.

    Args:
        vertices: Per pair, the bound on its window's vertices
            (:func:`~newton._src.geometry.terrain_contacts.terrain_window_vertices`).
        candidates: Per pair, the bound on its distinct candidates
            (:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`).
        walk_slots: Walk slots of a pair of the class.
        scan_run: Window vertices of one scan job of the class
            (:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`): the scan's runs are sized
            exactly from the pairs' windows.
        sides: Per pair, the bound on its window's columns plus rows
            (:func:`~newton._src.geometry.terrain_contacts.terrain_window_sides`).
        scale: Factor on every capacity.
    """
    v = np.asarray(vertices, np.int64)
    bounds = _job_bounds(v, sides, scan_run)
    capacities = {}
    for name, bound, share in zip(TERRAIN_SOLID_JOB_LISTS, bounds, TERRAIN_SOLID_JOB_SHARES, strict=True):
        total = int(bound.sum())
        default = (int(bound.max()) + -(-total // share)) if len(bound) else 0
        default = max(default, min(total, TERRAIN_SOLID_JOB_FLOOR))
        capacities[name] = max(1, int(np.ceil(scale * default)))
    # a pair's pool entries: up to eight rows (its key's point slots) per item of a window vertex
    total = int(np.sum(8 * TERRAIN_SOLID_ITEMS * v))
    default = (int(np.max(candidates)) if len(v) else 0) + walk_slots * len(v) // TERRAIN_SOLID_CANDIDATE_SHARE
    default = max(default, min(total, TERRAIN_SOLID_POOL_FLOOR))
    capacities["candidates"] = max(1, int(np.ceil(scale * default)))
    # the scan's runs: every pair's at its largest window, exact (a window past its bound takes the reserve)
    capacities["scans"] = max(1, int(np.sum(-(-v // int(scan_run)))))
    return capacities


def terrain_solid_job_reserves(
    vertices: np.ndarray, candidates: np.ndarray, *, scan_run: int, sides: np.ndarray
) -> dict[str, int]:
    """Return a solid class's job reserves: per job list, the candidate pool and the scan's runs, the work of
    :data:`TERRAIN_SOLID_RESERVE_PAIRS` pairs of the class's largest bounds (:func:`_job_bounds`, the most distinct
    candidates, the runs of the largest window), whatever
    ``NEWTON_TERRAIN_LAW_SHADOW_JOBS``, or of fewer pairs (at least one) when that would exceed
    :data:`TERRAIN_SOLID_RESERVE_BYTES`: the room a collide's work takes beyond the lists' room
    (:func:`terrain_solid_job_capacities`) with its rows unchanged.

    Args:
        vertices: Per pair, the bound on its window's vertices
            (:func:`~newton._src.geometry.terrain_contacts.terrain_window_vertices`).
        candidates: Per pair, the bound on its distinct candidates
            (:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates`).
        scan_run: Window vertices of one scan job of the class
            (:data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`).
        sides: Per pair, the bound on its window's columns plus rows
            (:func:`~newton._src.geometry.terrain_contacts.terrain_window_sides`).
    """
    v = int(np.max(vertices)) if len(vertices) else 0
    largest = [int(np.max(bound)) if len(vertices) else 0 for bound in _job_bounds(vertices, sides, scan_run)]
    bounds = dict(zip(TERRAIN_SOLID_JOB_LISTS, largest, strict=True))
    bounds["candidates"] = int(np.max(candidates)) if len(candidates) else 0
    bounds["scans"] = -(-v // int(scan_run))
    # bytes of a pair's reserve: 8 per job and run, 16 more per crease (its theta*), 40 per candidate
    per_pair = 8 * sum(bounds[name] for name in (*TERRAIN_SOLID_JOB_LISTS, "scans")) + 16 * bounds["thetas"]
    per_pair += 40 * bounds["candidates"]
    pairs = max(1, min(TERRAIN_SOLID_RESERVE_PAIRS, TERRAIN_SOLID_RESERVE_BYTES // max(per_pair, 1)))
    return {name: pairs * bound for name, bound in bounds.items()}


def terrain_solid_jobs(
    device,
    partner_class: int,
    first: int,
    pairs: int,
    capacities: dict[str, int],
    reserves: dict[str, int] | None = None,
    *,
    counts: wp.array | None = None,
) -> TerrainSolidJobs:
    """Allocate a solid class's :class:`TerrainSolidJobs`: per pair its record, core and candidate list head; the job
    lists, the scan's runs and the candidate pool with ``capacities`` as their room
    (:func:`terrain_solid_job_capacities`) and ``reserves`` beyond it (:func:`terrain_solid_job_reserves`); the job
    kernels' threads (:data:`TERRAIN_SOLID_JOB_THREADS` on CUDA, :data:`TERRAIN_SOLID_JOB_THREADS_CPU` on the CPU, at
    most the longest list's room).

    Args:
        device: The device.
        partner_class: The class (``TERRAIN_PARTNER_*``): which core records it keeps.
        first: The class's first table pair.
        pairs: Pairs of the class.
        capacities: Room of every job list, the scan's runs and the candidate pool.
        reserves: Their reserves, or ``None`` for none.
        counts: The class's :data:`TERRAIN_SOLID_COUNTERS` counters (zeroed before every launch), or ``None`` for its
            own.
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
    reserves = reserves or {}
    names = (*TERRAIN_SOLID_JOB_LISTS, "candidates", "scans")
    size = {name: max(int(capacities[name]) + int(reserves.get(name, 0)), 1) for name in names}
    rooms = [0] * TERRAIN_SOLID_COUNTERS
    for name, counter in zip(names, (_FACES, _EDGES, _THETAS, _VERTICES, _BORDER, _CANDIDATES, _SCANS), strict=True):
        rooms[int(counter)] = int(capacities[name])
    jobs.room = _Room(*rooms)
    for name in TERRAIN_SOLID_JOB_LISTS:
        setattr(jobs, name, wp.zeros(size[name], dtype=wp.vec2i, device=device))
    jobs.theta_n = wp.zeros(size["thetas"], dtype=wp.vec3, device=device)
    jobs.theta_kind = wp.zeros(size["thetas"], dtype=wp.int32, device=device)
    jobs.live = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.scans = wp.zeros(size["scans"], dtype=wp.vec2i, device=device)
    jobs.recover = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.small = wp.zeros(room, dtype=wp.int32, device=device)
    jobs.counts = counts if counts is not None else wp.zeros(TERRAIN_SOLID_COUNTERS, dtype=wp.int32, device=device)
    # the job kernels' grid is the room's: they stride over a list's reserve too
    longest = max(int(capacities[name]) for name in (*TERRAIN_SOLID_JOB_LISTS, "scans"))
    cuda = wp.get_device(device).is_cuda
    threads = TERRAIN_SOLID_JOB_THREADS if cuda else TERRAIN_SOLID_JOB_THREADS_CPU
    jobs.threads = max(1, min(threads, longest))
    border = TERRAIN_SOLID_BORDER_THREADS if cuda else TERRAIN_SOLID_JOB_THREADS_CPU
    jobs.border_threads = max(1, min(border, int(capacities["border"])))
    jobs.pair_threads = max(1, min(jobs.threads, room))
    jobs.finish_blocks = max(1, min(TERRAIN_SOLID_FINISH_BLOCKS if cuda else TERRAIN_SOLID_JOB_THREADS_CPU, room))
    pool = size["candidates"]
    jobs.q = wp.zeros(pool, dtype=wp.vec3, device=device)
    jobs.n = wp.zeros(pool, dtype=wp.vec3, device=device)
    jobs.d = wp.zeros(pool, dtype=wp.float32, device=device)
    jobs.feature = wp.zeros(pool, dtype=wp.int32, device=device)
    jobs.key = wp.zeros(pool, dtype=wp.int32, device=device)
    jobs.next = wp.zeros(pool, dtype=wp.int32, device=device)
    jobs.head = wp.full(room, -1, dtype=wp.int32, device=device)
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
        jobs.live,
        jobs.scans,
        jobs.recover,
        jobs.small,
        jobs.q,
        jobs.n,
        jobs.d,
        jobs.feature,
        jobs.key,
        jobs.next,
        jobs.head,
    )
    return int(sum(array.capacity for array in arrays))

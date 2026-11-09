# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared float32 primitives of the per-pair terrain contact law.

The law emits the support set of one (heightfield, convex shape) pair directly: one row per local minimum of the
core-to-surface distance within the writer's reach, plus the corners of flat support regions, each owned by one
surface feature (a plane cluster, a crease edge or a vertex). This module holds what every per-type kernel shares:
the surface predicates, the per-pair window (:func:`terrain_box_cells`, :func:`terrain_tile_exit`,
:func:`terrain_window_extent`), a window's cache of cluster planes (:class:`TerrainPlanes`), the finish
(:func:`terrain_finish_slots`, :func:`terrain_finish_list`, :func:`terrain_finish_deferred`), the per-pair inputs and
side buffer of the shadow route (:class:`TerrainPairInputs`, :class:`TerrainLawRows`), the launch arguments of its fixed
grids (:class:`TerrainPairWork`, :class:`TerrainFinishScratch`) and the per-pair output protocol
(:func:`terrain_pair_exit`, :func:`terrain_spill_rows`, :func:`terrain_hold_row`, :func:`terrain_place_candidate`,
:func:`terrain_plane_group_of_three`, :func:`terrain_defer_pair`, :func:`terrain_store_row`). The static
data live in :mod:`terrain_contacts`, the per-type kernels in modules of their own (``terrain_law_<type>``), and
:mod:`terrain_shadow` runs them after the narrow phase and writes their rows as the contacts of the pairs the narrow
phase leaves to the law (by default; ``NEWTON_TERRAIN_LAW``), or runs them beside the terrain route without feeding any
contact (``NEWTON_TERRAIN_LAW_SHADOW``).

Surface model:

* Window-local frame. A pair works in the frame of one window vertex ``(r0, c0)``: vertex ``(r, c)`` lies at
  ``((c - c0) dx, (r - r0) dy, z(r, c) - z_ref)``, computed from integer index differences, so a vertex shared by
  several cells is bit-identical in all of them (the narrow phase's per-cell frame ``x1 = x0 + dx`` leaves cracks
  of up to 9.5 um on a 200 m terrain). The window's origin is the exact grid position in float64 and a core
  point enters the frame rounded once, so its position against the grid is exact to a float32 rounding of a
  window-sized number (a float32 origin rounds by up to 4 um 100 m out). The law reads one surface type, vertex
  heights in float32 on a lattice (:attr:`TerrainFeatureData.heights`), a heightfield's decoded once at build as the
  narrow phase decodes them on the CPU (float32 ``min_z + e * (max_z - min_z)``, each operation rounded once, on
  every device: :func:`terrain_decode` uses explicitly rounded operations, which no floating-point contraction
  fuses).
* Canonical edge functions. The face test of a triangle at an edge and the edge's own test read one value
  ``s = (k - v0) . m`` (``m`` the triangle's in-plane outward normal at the edge, ``v0`` the edge's lower vertex),
  and the edge's slab test and the vertex's Voronoi test read one value ``(k - v) . (w - v)``. Both sides of every
  boundary therefore decide on the same bits on the CPU, and on CUDA in kernels compiled without contraction.
  Under CUDA's default contraction two call sites of one function may be fused differently; their values then
  differ by a few float32 ulps of the window extent, well inside the closed band below.
* Closed predicates. Every side claims its region up to ``eps = 8 eps32 max(window extent, 0.1 m)`` beyond its
  boundary, so every support is claimed at least once in float32; two inclusive claims of one constraint coincide
  and the per-pair finish merges them (``4 eps`` in position, ``1e-3`` in normal).
* Plane clusters (built by :class:`~newton._src.geometry.terrain_contacts.TerrainFeatures`): every triangle of a
  cluster uses the cluster's plane, a cluster of one triangle its own plane. Edge classes (flat inside a cluster,
  else convex or concave) and vertex features (incident cluster normals spanning 3-D) follow from the cluster ids
  and the heights.
* The border: the heightfield is a solid block. Its footprint ``[-hx, hx] x [-hy, hy]`` has vertical side walls from
  the border edges down :data:`TERRAIN_WALL_DEPTH` (the terrain route's prism extrusion), so that the block's border
  features are its walls (one plane per side, :func:`terrain_border_side`), its border edges (convex creases between
  their one triangle and the wall, :func:`terrain_wall_edge_value`), its border vertices (whose rings take the wall
  below them) and the vertical corner lines where two walls meet. The walls are one-sided: they bound the block for
  points beside it, while a point inside the footprint below the surface exits through the top, as everywhere.

Rows carry a window-local feature id in the 23-bit ``sort_sub_key`` of the contact contract (19 bits:
``r mod 64``, ``c mod 64``, kind, sub-feature, point slot), unique within any window under 64 cells across. The kinds'
sub-features: a face's triangle (0, 1) or the wall of its vertex's x- or y-edge (2, 3: :data:`TERRAIN_FACE_WALL`); an
edge's x-, y- or diagonal edge (0, 1, 2) or its vertex's corner line (3: :data:`TERRAIN_EDGE_CORNER`).

The per-pair finish merges coincident rows, drops the rows the others of their plane group imply (the lower convex
hull of tangent position and distance, residual ``1e-6 m``) and keeps at most four rows per plane group (deepest,
farthest, largest triangle, largest quadrilateral): every row it keeps is a support, and no count bounds them. It
runs in place, on a walking pair's own slots (:func:`terrain_finish_slots`, up to :data:`TERRAIN_FINISH_MASK_ROWS`
rows, one thread), or for a deferred pair on its candidates in a block's scratch (:func:`terrain_finish_list`, every
candidate a pair's window can give, :class:`TerrainFinishScratch`; a block's lanes splitting every step); both keep the
same rows, bit for bit. A pair's rows live in its walk slots, or when they are more, in a spill
region sized at the scene level (:class:`TerrainLawRows`). A spill region too small for a collide's rows, or a window
larger than the one its pair's shape data gave at construction, sets a sticky bit of the narrow phase's buffer status
word, which ``record_narrow_phase_buffers`` keeps and ``NarrowPhase.check_buffer_capacity`` raises on
(``terrain_rows``; stale features: ``terrain_stale``; a solid class's job lists without room for a pair's work:
``terrain_jobs``).
"""

from __future__ import annotations

import numpy as np
import warp as wp

# ----------------------------------------------------------------------------------------------------------------
# tolerances and layout
# ----------------------------------------------------------------------------------------------------------------

TERRAIN_FLAT_EPS = 1.0e-6
"""Plane-cluster membership tolerance [m], absolute: every vertex of a cluster lies within it of the cluster plane."""
TERRAIN_EPS32 = 1.1920928955078125e-7
"""Float32 machine epsilon."""
TERRAIN_EPS_FACTOR = 8.0
"""Closed-predicate band in float32 epsilons of the window extent."""
TERRAIN_EPS_MIN_EXTENT = 0.1
"""Smallest window extent [m] the predicate band scales with."""
TERRAIN_MERGE_SCALE = 4.0
"""Rows closer than this many predicate bands (with normals within :data:`TERRAIN_MERGE_NORMAL`) coincide."""
TERRAIN_MERGE_NORMAL = 1.0e-3
"""Largest normal difference of coincident rows."""
TERRAIN_GROUP_NORMAL = 1.0e-5
"""Largest normal difference of rows on one plane (one plane group)."""
TERRAIN_IMPLIED_RESIDUAL = 1.0e-6
"""A row no deeper than this below the lower hull of its plane group's other rows is implied [m]."""
TERRAIN_BARYCENTRIC_TOLERANCE = 1.0e-5
"""Relative tolerance of the tangent-position containment of implied rows."""
TERRAIN_NORMAL_SAME = 1.0e-5
"""Unit normals closer than this are one direction (vertex features)."""
TERRAIN_VERTEX_SPAN = 1.0e-10
"""A vertex is a feature when three of its incident cluster normals have a triple product above this.

The square of :data:`TERRAIN_NORMAL_SAME`: a normal cone at least that wide both ways is a feature. A support whose
normal lies inside a cone that is not a feature misses the faces and edges by at most the cone's width times the
reach (0.4 um at 4 cm). At 1e-6, 7,685 more vertices of the recorded suite terrain were not features, 1,800 of them
convex; a box, cylinder or toe whose support sat over one of those lost that support in 84-87% of poses.
"""
TERRAIN_CONE_BAND = 1.0e-6
"""Relative band of the closed normal-cone test ``n . (w - v) <= band |w - v|``."""
TERRAIN_GROUP_ROWS = 4
"""Rows kept per plane group."""
TERRAIN_FINISH_MASK_ROWS = 24
"""Rows the in-place finish of a walking pair (:func:`terrain_finish_slots`) takes: the length of its arrays, which every
call zero-fills, so the largest class's walk slots (a hull's 24); a deferred pair's finish takes any number
(:func:`terrain_finish_list`)."""
TERRAIN_TILE_CELLS = 8
"""Cells per side of a tile of the tile-max mip."""
TERRAIN_FINE_CELLS = 2
"""Cells per side of a block of the fine height mip: the window exit of a pair that the tiles keep
(:func:`terrain_tile_exit`)."""
TERRAIN_FINE_MARGIN = 1.0e-5
"""Height [m] by which every fine block under a window must lie below a pair's ``low`` for the fine exit: a feature
within it of the reach limit still reaches the pair's own rejects, which decide it as they did without the fine
exit."""

TERRAIN_EDGE_FLAT = wp.constant(0)
"""Edge class: both triangles in one plane cluster."""
TERRAIN_EDGE_CONVEX = wp.constant(1)
"""Edge class: a ridge between two clusters."""
TERRAIN_EDGE_CONCAVE = wp.constant(2)
"""Edge class: a valley between two clusters."""
TERRAIN_EDGE_BORDER = wp.constant(3)
"""Edge class: on the heightfield's border."""
TERRAIN_VERTEX_EDGE_FLAT = wp.constant(1)
"""Two-bit code of a flat edge in a vertex's bits (:attr:`TerrainFeatureData.vertex_bits`; edge ``which`` at bits
``2 which`` and ``2 which + 1``, 0 for a crease)."""
TERRAIN_VERTEX_EDGE_BORDER = wp.constant(2)
"""Two-bit code of a border edge in a vertex's bits."""
TERRAIN_VERTEX_EDGE_CONCAVE = wp.constant(3)
"""Two-bit code of a crease in a vertex's bits that every window frame of :data:`TERRAIN_CREASE_STATIC_CELLS` cells
classes concave (:func:`terrain_crease_class`): its far vertex lies above its first triangle's cluster plane by more
than every float32 rounding such a frame can give that test (:func:`terrain_crease_static`); 0 for any other crease."""
TERRAIN_CREASE_STATIC_CELLS = 64
"""Window-frame offset [cells] up to which a crease's static class (:data:`TERRAIN_VERTEX_EDGE_CONCAVE`) holds: every
point the class test reads lies within this many cells of the window's reference vertex."""
TERRAIN_CREASE_STATIC_ROUNDING = 64.0
"""Float32 epsilons of the window-frame magnitudes by which a crease's far vertex must clear its first triangle's
cluster plane to be concave in every frame (:func:`terrain_crease_static`): over six times the rounding a frame's test
can take."""
TERRAIN_VERTEX_FEATURE_BIT = wp.constant(64)
"""Bit of a vertex feature in a vertex's bits (:func:`terrain_vertex_feature`)."""
TERRAIN_VERTEX_CREASE_UNSURE_BIT = wp.constant(128)
"""Bit of a vertex in its bits when some of its creases of code 0 is not convex in every window frame of
:data:`TERRAIN_CREASE_STATIC_CELLS` cells (:func:`terrain_crease_static`): without it, every crease of code 0 of the
vertex is convex in every such frame."""

TERRAIN_EDGE_X = wp.constant(0)
"""Edge ``(r, c)-(r, c + 1)``, between cell ``(r - 1, c)`` sub 1 and cell ``(r, c)`` sub 0."""
TERRAIN_EDGE_Y = wp.constant(1)
"""Edge ``(r, c)-(r + 1, c)``, between cell ``(r, c - 1)`` sub 0 and cell ``(r, c)`` sub 1."""
TERRAIN_EDGE_DIAGONAL = wp.constant(2)
"""Edge ``(r, c)-(r + 1, c + 1)``, between the two triangles of cell ``(r, c)``."""

TERRAIN_FEATURE_FACE = wp.constant(0)
TERRAIN_FEATURE_EDGE = wp.constant(1)
TERRAIN_FEATURE_VERTEX = wp.constant(2)
TERRAIN_FEATURE_RECOVERY = wp.constant(3)
TERRAIN_FEATURE_BITS = 19
"""Bits of a feature id: ``r mod 64`` (6), ``c mod 64`` (6), kind (2), sub-feature (2), point slot (3)."""
TERRAIN_FACE_WALL = wp.constant(2)
"""Sub-feature of a face row on a wall: the wall below vertex ``(r, c)``'s x-edge (this code) or y-edge (one more)."""
TERRAIN_EDGE_CORNER = wp.constant(3)
"""Sub-feature of an edge row on a corner line: the vertical edge where two walls meet below vertex ``(r, c)``."""

TERRAIN_ON_FEATURE = 1.0e-9
"""Core distance [m] below which a core point lies on an edge or a corner line: its row takes the bisector of their
normals."""
TERRAIN_WALL_DEPTH = 1.0
"""Depth of the heightfield's side walls below its border edges [m, heightfield-local]: the block's footprint has
vertical walls from every border edge down this far, the terrain route's prism extrusion (``TRIANGLE_PRISM_EXTRUSION``),
and a wall or corner line claims nothing below them."""
TERRAIN_SIDE_COUNT = 4
"""Walls of a heightfield (:func:`terrain_border_side`)."""

TERRAIN_FINISH_CANDIDATE_OVERFLOW = wp.constant(1)
"""Report flag: the pair's window gave more candidates than its scratch holds (:class:`TerrainFinishScratch`),
which only a window larger than its pair's shape data gave at construction can (the shape's scale, margin or gap
changed since)."""
TERRAIN_FINISH_ROW_OVERFLOW = wp.constant(2)
"""Report flag: the spill region had no room left for the pair's rows (:attr:`TerrainLawRows.capacity`)."""
TERRAIN_STATUS_ROWS = wp.constant(1 << 13)
"""Sticky bit of the narrow phase's buffer status word (``terrain_rows``): the terrain law's spill region overflowed,
or a pair's window outgrew the bound its shape data gave at construction."""
TERRAIN_STATUS_STALE = wp.constant(1 << 14)
"""Sticky bit of the narrow phase's buffer status word (``terrain_stale``): terrain features older than the heights
were read."""
TERRAIN_STATUS_JOBS = wp.constant(1 << 15)
"""Sticky bit of the narrow phase's buffer status word (``terrain_jobs``): a solid class's job lists or candidate pool
(scene-level buffers) had no room for some of a pair's work, which took the lists' reserve (its rows unchanged) or,
past the reserve, was lost (the pair keeps no row, :data:`TERRAIN_PAIR_INCOMPLETE`)."""

TERRAIN_PAIR_TILE_EXIT = wp.constant(1)
"""Pair state: the core lies above every tile-max block under its window by more than its reach (no row)."""
TERRAIN_PAIR_STALE = wp.constant(2)
"""Pair state: the heightfield's features are older than its heights (no row; ``terrain_stale`` is raised)."""
TERRAIN_PAIR_FILTERED = wp.constant(4)
"""Pair state: the pipeline's immovable-pair policy drops the pair (no row)."""
TERRAIN_PAIR_OUTSIDE = wp.constant(8)
"""Pair state: the core's window misses the heightfield's footprint (no row)."""
TERRAIN_PAIR_ROW_OVERFLOW = wp.constant(16)
"""Pair state: the pair's rows exceed its walk slots and the spill region had no room for them (no row;
``terrain_rows`` is raised)."""
TERRAIN_PAIR_CANDIDATE_OVERFLOW = wp.constant(32)
"""Pair state: the pair's window gave more candidates than its scratch holds (it keeps the first in window
order; ``terrain_rows`` is raised): its shape data changed after the shadow was built."""
TERRAIN_PAIR_WIDE = wp.constant(64)
"""Pair state: the window spans 64 cells or more, so feature ids may repeat within it (rows are still emitted)."""
TERRAIN_PAIR_DEFERRED = wp.constant(128)
"""Pair state, transient: the pair's rows need more than its walk slots (or, for the point class, the general finish),
and it awaits its class's finish of deferred pairs (:func:`terrain_defer_pair`)."""
TERRAIN_PAIR_JOBS = wp.constant(1024)
"""Pair state, transient: a solid pair's scan listed its window's features for its class's job kernels, and its close
kernel takes their rows."""
TERRAIN_PAIR_INCOMPLETE = wp.constant(2048)
"""Pair state: a job list or the candidate pool of a solid pair's class had no room for some of its work, its reserve
included, so that the pair keeps no row (``terrain_jobs`` is raised)."""
TERRAIN_PAIR_HELD = wp.constant(4096)
"""Pair state, transient: a solid pair holds its rows in its walk slots (window frame, ``count`` of them) for its
class's small-pair finish kernel."""
TERRAIN_PAIR_RESERVE = wp.constant(8192)
"""Pair state, transient: some of a solid pair's work found no room in its class's job lists or candidate pool and took
their reserve: its rows are unchanged, and its close counts it and raises ``terrain_jobs``."""
TERRAIN_WINDOW_WIDE = 64
"""Cells across a window from which feature ids are no longer unique."""
TERRAIN_PLANE_CACHE = 4
"""Cluster planes a window's cache holds (:class:`TerrainPlanes`)."""
TERRAIN_CELL_WIDEN = 1.0e-3
"""Widening of a footprint box's float32 cell coordinates [cells], so that rounding never drops a touched cell."""

_MaskInts = wp.types.vector(length=TERRAIN_FINISH_MASK_ROWS, dtype=wp.int32)
_MaskFloats = wp.types.vector(length=TERRAIN_FINISH_MASK_ROWS, dtype=wp.float32)

_MASK_ROWS = wp.constant(TERRAIN_FINISH_MASK_ROWS)
_CELL_WIDEN = wp.constant(TERRAIN_CELL_WIDEN)
_TILE_SIDE = wp.constant(TERRAIN_TILE_CELLS)
_FINE_SIDE = wp.constant(TERRAIN_FINE_CELLS)
_FINE_MARGIN = wp.constant(TERRAIN_FINE_MARGIN)
_EPS32 = wp.constant(TERRAIN_EPS32)
_EPS_FACTOR = wp.constant(TERRAIN_EPS_FACTOR)
_EPS_MIN_EXTENT = wp.constant(TERRAIN_EPS_MIN_EXTENT)
_MERGE_SCALE = wp.constant(TERRAIN_MERGE_SCALE)
_MERGE_NORMAL = wp.constant(TERRAIN_MERGE_NORMAL)
_GROUP_NORMAL = wp.constant(TERRAIN_GROUP_NORMAL)
_IMPLIED_RESIDUAL = wp.constant(TERRAIN_IMPLIED_RESIDUAL)
_BARYCENTRIC_TOLERANCE = wp.constant(TERRAIN_BARYCENTRIC_TOLERANCE)
_NORMAL_SAME = wp.constant(TERRAIN_NORMAL_SAME)
_VERTEX_SPAN = wp.constant(TERRAIN_VERTEX_SPAN)
_CONE_BAND = wp.constant(TERRAIN_CONE_BAND)
_CREASE_CELLS = wp.constant(TERRAIN_CREASE_STATIC_CELLS)
_CREASE_ROUNDING = wp.constant(TERRAIN_CREASE_STATIC_ROUNDING)
_CREASE_FLAT = wp.constant(TERRAIN_FLAT_EPS)
_GROUP_ROWS = wp.constant(TERRAIN_GROUP_ROWS)
_WALL_DEPTH = wp.constant(TERRAIN_WALL_DEPTH)
_ON_FEATURE = wp.constant(TERRAIN_ON_FEATURE)
_NO_ROW = wp.constant(2147483647)
"""A lane's answer when it has no candidate (the finish of deferred pairs): past every candidate, so that the least
answer of a block's lanes is the first candidate any lane has."""


# ----------------------------------------------------------------------------------------------------------------
# static data as the kernels see it
# ----------------------------------------------------------------------------------------------------------------


@wp.struct
class TerrainSurfaceInfo:
    """One surface's layout in the law's static arrays: a heightfield, a lattice of vertex heights in the shape's
    frame."""

    data_offset: int
    """First vertex height of the surface in :attr:`TerrainFeatureData.heights`."""
    nrow: int
    """Vertex rows."""
    ncol: int
    """Vertex columns."""
    x0: float
    """Position of vertex column 0 along x [m] in the shape's frame (a heightfield's ``-hx``), float32: the cell
    lookup's (:func:`terrain_box_cells`)."""
    y0: float
    """Position of vertex row 0 along y [m] (a heightfield's ``-hy``), float32."""
    x0_64: wp.float64
    """:attr:`x0` in float64, exact (the window origin's: a heightfield's float32 ``-hx``)."""
    y0_64: wp.float64
    """:attr:`y0` in float64, exact."""
    min_z: float
    """Lowest height the surface's vertices can take [m] (a heightfield's height of elevation 0), the scale of the
    static crease classes' rounding bound."""
    max_z: float
    """Highest height the surface's vertices can take [m] (a heightfield's height of elevation 1)."""
    dx: float
    """Cell size along x [m] (float32; a heightfield's ``2 hx / (ncol - 1)``, as the narrow phase computes it)."""
    dy: float
    """Cell size along y [m]."""
    dx64: wp.float64
    """Cell size along x [m] in float64: the window origin's and the cluster planes' grid step (a heightfield's
    ``2 hx / (ncol - 1)`` from the float32 ``hx``, computed once on the host, bit for bit the device's float64
    division)."""
    dy64: wp.float64
    """Cell size along y [m] in float64."""
    triangle_offset: int
    """First cluster id of the heightfield."""
    plane_offset: int
    """First cluster plane record of the heightfield."""
    tile_offset: int
    """First tile-max value of the heightfield."""
    tile_cols: int
    """Tiles per tile row."""
    fine_offset: int
    """First fine-mip value of the heightfield."""
    fine_cols: int
    """Fine blocks per fine row."""


@wp.struct
class TerrainClusterPlane:
    """The plane of a cluster of several triangles: ``z = z(anchor) + offset + gx (x - x_a) + gy (y - y_a)``."""

    anchor: int
    """Vertex id ``r * ncol + c`` the plane is anchored at: the cluster's lowest-id vertex, often far from a window."""
    gx: float
    """Plane gradient along x."""
    gy: float
    """Plane gradient along y."""
    offset: float
    """Plane height above the anchor vertex [m]."""


@wp.struct
class TerrainFeatureData:
    """Everything the law's kernels read about the terrain (all heightfields of a model)."""

    heights: wp.array[wp.float32]
    """Every surface's vertex heights [m], concatenated: a heightfield's elevations decoded once at build
    (:func:`terrain_decode`, the narrow phase's CPU decode)."""
    clusters: wp.array[wp.int32]
    """Per triangle: the dense index of its cluster's plane record, or ``-(1 + local triangle id)`` alone."""
    planes: wp.array[TerrainClusterPlane]
    """Plane records of the clusters of several triangles."""
    tile_max: wp.array[wp.float32]
    """Per tile of 8 x 8 cells: the largest height of its vertices [m]."""
    fine_max: wp.array[wp.float32]
    """Per block of 2 x 2 cells: the largest height of its vertices [m]."""
    vertex_bits: wp.array[wp.uint8]
    """Per vertex, indexed as the heights: the kinds of the three edges it owns and whether it is a vertex feature
    (:func:`terrain_vertex_bits_of_clusters`), static data of the clusters and heights read by
    :func:`terrain_edge_kind` and :func:`terrain_vertex_feature`."""
    surfaces: wp.array[TerrainSurfaceInfo]
    """Layout of every heightfield, by heightfield index."""
    elevation_version: wp.array[wp.int32]
    """Version of the heights of every heightfield, bumped by every reported edit."""
    feature_version: wp.array[wp.int32]
    """Version of the heights the clusters and the mip were built from."""
    layout: int
    """Layout of the plane records this struct points at (the features' layout version when it was fetched)."""
    layout_version: wp.array[wp.int32]
    """The features' current layout (one entry), bumped whenever a rebuild reallocates the plane records."""


@wp.struct
class TerrainWindow:
    """The window-local frame of one pair."""

    surface: TerrainSurfaceInfo
    """The heightfield's layout."""
    heightfield: int
    """Heightfield index."""
    r0: int
    """Row of the window's reference vertex."""
    c0: int
    """Column of the window's reference vertex."""
    z_ref: float
    """Decoded height of the reference vertex [m]."""
    origin: wp.vec3d
    """Heightfield-local position of the reference vertex at height ``z_ref`` [m], in float64."""
    eps: float
    """Closed-predicate band [m] (see :func:`terrain_eps`)."""


@wp.struct
class TerrainPlanes:
    """A window's cache of the planes of clusters of several triangles (see :func:`terrain_plane`).

    Evaluating such a plane over the window's reference vertex takes about ten float64 operations, which some GPUs
    run at 1/64 of the float32 rate; a window reads each cluster's plane many times (its faces, both sides of its
    edges, the point-below test), so it evaluates it once. A cached plane has the bits of a fresh evaluation. The
    :data:`TERRAIN_PLANE_CACHE` entries are named fields, not an indexed array, so that the cache can stay in
    registers.
    """

    id0: int
    """Cluster id (plane record index) of entry 0 (-1: empty)."""
    id1: int
    id2: int
    id3: int
    plane0: wp.vec4
    """Entry 0: the unit upward normal and the plane's height over the window's reference vertex [m]."""
    plane1: wp.vec4
    plane2: wp.vec4
    plane3: wp.vec4
    next: int
    """Entry the next miss replaces (round robin)."""


@wp.struct
class TerrainPairInputs:
    """What the law's per-type kernels read of a pair besides the terrain: its shapes and the pipeline's pair policy.

    Every per-type kernel takes ``(begin, slot_begin, inputs, data, out)`` and a launch argument (the point class's
    walks its :class:`TerrainPairWork`, a solid class's kernels its jobs, every class's finish of deferred pairs its
    :class:`TerrainFinishScratch`), a solid class's kernel also the solid classes'
    :class:`~newton._src.geometry.terrain_law_solid.TerrainSolidData` (see
    :data:`~newton._src.geometry.terrain_shadow.TERRAIN_LAW_KERNELS`): pair ``begin + i`` of the static pair table
    (:class:`~newton._src.geometry.terrain_contacts.TerrainPairTable`) walks its window in its walk slots
    ``slot_begin + i * K``, ``K`` its partner class's walk slots, and writes its rows there, or when they are more,
    to rows of the spill region (:func:`terrain_spill_rows`).
    """

    pairs: wp.array[wp.vec2i]
    """``(heightfield shape, partner shape)`` of every pair, in table order."""
    shape_type: wp.array[wp.int32]
    """Geometry type of every shape."""
    shape_transform: wp.array[wp.transform]
    """World transform of every shape (the collision pipeline's per-collide shape transforms)."""
    shape_data: wp.array[wp.vec4]
    """Per shape: scale ``xyz`` (sphere and capsule radius in ``x``, capsule half-height in ``y``) and margin ``w``."""
    shape_gap: wp.array[float]
    """Contact gap of every shape [m]."""
    shape_surface: wp.array[wp.int32]
    """Surface index of every surface shape (:class:`~newton._src.geometry.terrain_contacts.TerrainSurfaces`)."""
    shape_body: wp.array[int]
    """The broad phase's immovable-pair test: body of every shape (immobile shapes as static)."""
    body_flags: wp.array[int]
    """The broad phase's immovable-pair test: body flags (empty: dynamic and kinematic bodies are not told apart)."""
    include_static_kinematic_pairs: int
    """The broad phase's immovable-pair test: nonzero keeps pairs of two immovable shapes."""


@wp.struct
class TerrainLawRows:
    """The rows the law's per-type kernels write: per pair a static range of walk slots, then one spill region, and
    per pair its first row, row count and state.

    A pair's rows are rows ``first`` to ``first + count``: read them per pair, through ``first`` and ``count``. A pair
    that walks its window holds its distinct rows in its walk slots (window frame), and when they are its rows, they are
    rewritten at the front in the world frame. A pair with more rows than its walk slots is deferred to its class's
    finish of deferred pairs, which writes all of them to rows of the spill region it reserves
    (:func:`terrain_spill_rows`): its rows are bounded by its window, not by its slots. Reservations take spill rows in
    the order the pairs reach them (the launch's order, which varies between runs on a GPU), so a pair's rows depend on
    the pair alone, their place in the spill region does not. A pair that finds no room keeps its finish's first rows by
    priority in its walk slots and is flagged (:func:`terrain_finish_overflow`).

    Rows are in the world frame: ``q`` on the terrain, ``n`` the unit normal from the terrain to the partner and
    ``d = n . (k - q)`` the core distance, so that the partner's core point is ``q + d n``.
    """

    q: wp.array[wp.vec3]
    """Terrain point of every row [m]."""
    n: wp.array[wp.vec3]
    """Unit normal of every row, from the terrain to the partner."""
    d: wp.array[float]
    """Core distance of every row [m] (negative: the core point lies inside the solid)."""
    feature: wp.array[wp.int32]
    """Window-local feature id of every row (:func:`terrain_feature_id`)."""
    slots: int
    """Walk slots of every pair together: the spill region's first row."""
    capacity: int
    """Rows in all: the walk slots and the spill region."""
    spilled: wp.array[wp.int32]
    """Spill rows asked for since the launch began (one entry, zeroed before every launch), room found or not: the
    spill region a launch needs."""
    reserved: wp.array[wp.int32]
    """Spill rows reserved since the launch began (one entry, zeroed before every launch), at most the spill region's:
    a reservation that does not fit takes none, so that a later, smaller one still can."""
    first: wp.array[wp.int32]
    """First row of every pair's rows (-1: none)."""
    count: wp.array[wp.int32]
    """Rows of every pair."""
    state: wp.array[wp.int32]
    """``TERRAIN_PAIR_*`` bits of every pair."""
    status: wp.array[wp.int32]
    """The narrow phase's sticky buffer status word (``terrain_rows``, ``terrain_stale``, ``terrain_jobs``)."""
    overflow: wp.array[wp.int32]
    """Pairs whose rows found no room in the spill region (``[0]``), whose windows outgrew their scratch (``[1]``) and
    whose work found no room in their solid class's job lists or candidate pool (``[2]``: those that took the lists'
    reserve and those that lost work), accumulated over collides."""
    deferred: wp.array[wp.int32]
    """Pairs a class's walks left to its finish of deferred pairs (:data:`TERRAIN_PAIR_DEFERRED`): partner class
    ``c``'s list starts at ``deferred_first[c]`` and holds ``deferred_count[c]`` pairs, appended in any order."""
    deferred_first: wp.array[wp.int32]
    """Per partner class: the start of its deferred list (its first pair in the table)."""
    deferred_count: wp.array[wp.int32]
    """Per partner class: the pairs in its deferred list (zeroed before every launch)."""


# ----------------------------------------------------------------------------------------------------------------
# launches of the per-type kernels: block size, the tiled kernels' work list, the deferred pairs' scratch
# ----------------------------------------------------------------------------------------------------------------

TERRAIN_LAW_BLOCK = 32
"""Threads per block of every terrain-law launch on CUDA: lanes per pair of the tiled kernels (one warp), and the
one-thread kernel's block (several blocks per SM, so that its warps spread evenly), one module build per device."""
TERRAIN_LAW_TILE_BLOCKS = 2048
"""Blocks of a tiled launch at most, whatever the class's size: one per queued pair up to this many (about one
resident wave of one-warp blocks on a 188-SM GPU), more queued pairs taking turns."""
TERRAIN_LAW_DEFERRED_BLOCKS = 64
"""Blocks of a class's finish of deferred pairs (:data:`TERRAIN_LAW_BLOCK` lanes each), whatever the class's size: they
take the class's deferred pairs in turn."""


@wp.struct
class TerrainPairWork:
    """The point class's pairs with larger windows that reach their windows, which its one-thread kernel queues for its
    tiled kernel, in any order (a pair's rows depend on the pair alone)."""

    first: int
    """First table pair of the class's tiled segment: the one-thread kernel walks the pairs before it, and of those from
    it on queues the ones that reach their windows."""
    pairs: wp.array[wp.int32]
    """The queued table pairs, room for every pair of the tiled segment."""
    count: wp.array[wp.int32]
    """The queued pairs (one entry, zeroed before every launch)."""
    blocks: int
    """Blocks of the tiled launch: block ``b`` takes the queued pairs ``b``, ``b + blocks``, ..."""


def terrain_pair_work(device, first: int, pairs: int, *, count: wp.array | None = None) -> TerrainPairWork:
    """Allocate the :class:`TerrainPairWork` of a tiled segment of ``pairs`` pairs from table pair ``first``.

    Args:
        device: The device.
        first: First table pair of the tiled segment; the one-thread kernel walks the pairs before it.
        pairs: Pairs of the tiled segment.
        count: The one-entry counter of queued pairs (zeroed before every launch), or ``None`` for one of its own.
    """
    work = TerrainPairWork()
    work.first = first
    work.pairs = wp.zeros(max(pairs, 1), dtype=wp.int32, device=device)
    work.count = count if count is not None else wp.zeros(1, dtype=wp.int32, device=device)
    work.blocks = max(min(pairs, TERRAIN_LAW_TILE_BLOCKS), 1)
    return work


def terrain_law_block_dim(device) -> int:
    """Return the block size of every terrain-law launch on ``device``: :data:`TERRAIN_LAW_BLOCK` on CUDA, one on the
    CPU (where ``wp.launch_tiled`` runs one lane per pair whatever it is given, and one size builds the module once)."""
    return TERRAIN_LAW_BLOCK if wp.get_device(device).is_cuda else 1


TERRAIN_LAW_SCRATCH_BYTES = 1 << 24
"""Bytes of a class's scratch up to which its finish of deferred pairs takes :data:`TERRAIN_LAW_DEFERRED_BLOCKS`
blocks: a class whose windows can give more candidates finishes its deferred pairs on fewer blocks (at least one), so
that its scratch stays near this size (:func:`terrain_finish_blocks`). A rare path's parallelism, not a bound on any
pair."""
TERRAIN_FINISH_ROW_BYTES = 60
"""Bytes of one candidate in a scratch: terrain point, core point, normal, distance, feature id and the finish's four
words of state."""


@wp.struct
class TerrainFinishScratch:
    """Global scratch of a class's finish of deferred pairs: per block, the candidates of the pair it finishes (its
    walk's rows, unmerged) and the finish's state of each.

    Block ``b`` holds candidate ``j`` at ``b * capacity + j``, off the per-thread stack: the largest stack a context
    launches sets the driver's local-memory reservation for every thread the device can hold. ``capacity`` bounds the
    candidates of every pair of the class (:func:`~newton._src.geometry.terrain_contacts.terrain_pair_candidates` from
    each pair's shape extent, reach and cells at construction), so that the finish (:func:`terrain_finish_list`) sees
    all of them.
    """

    blocks: int
    """Blocks of the launch: its grid stride over the deferred list."""
    capacity: int
    """Candidates per block."""
    q: wp.array[wp.vec3]
    """Terrain point of every candidate [m] (window frame)."""
    k: wp.array[wp.vec3]
    """Core point of every candidate [m] (window frame)."""
    n: wp.array[wp.vec3]
    """Unit normal of every candidate."""
    d: wp.array[float]
    """Core distance of every candidate [m]."""
    feature: wp.array[wp.int32]
    """Feature id of every candidate."""
    alive: wp.array[wp.int32]
    """The finish's state: whether a candidate is still a row (not merged, not implied)."""
    group: wp.array[wp.int32]
    """The finish's state: the candidate representing a candidate's plane group (-1: none)."""
    rank: wp.array[wp.int32]
    """The finish's state: a candidate's rank in its plane group (0 deepest, 1 farthest, 2 largest triangle, 3 largest
    quadrilateral, 4 not kept); before its last step, a plane group's size at the candidate representing it."""
    mark: wp.array[wp.int32]
    """The finish's state: the first coincident or coplanar candidate of one the merge or the groups leave undecided,
    then the candidates tested for implied-row removal, and on return the kept ones (1)."""


def terrain_finish_blocks(capacity: int) -> int:
    """Return the blocks of a finish of deferred pairs whose blocks hold ``capacity`` candidates each:
    :data:`TERRAIN_LAW_DEFERRED_BLOCKS`, fewer (at least one) when that scratch would exceed
    :data:`TERRAIN_LAW_SCRATCH_BYTES`."""
    per_block = max(int(capacity), 1) * TERRAIN_FINISH_ROW_BYTES
    return max(1, min(TERRAIN_LAW_DEFERRED_BLOCKS, TERRAIN_LAW_SCRATCH_BYTES // per_block))


def terrain_finish_scratch(device, capacity: int, *, blocks: int | None = None) -> TerrainFinishScratch:
    """Allocate the :class:`TerrainFinishScratch` of a finish of deferred pairs whose blocks hold ``capacity``
    candidates.

    Args:
        device: The device.
        capacity: Distinct candidates per block (the most any pair of the class can give).
        blocks: Blocks of the launch, or ``None`` for :func:`terrain_finish_blocks`.
    """
    scratch = TerrainFinishScratch()
    scratch.capacity = max(int(capacity), 1)
    scratch.blocks = terrain_finish_blocks(scratch.capacity) if blocks is None else int(blocks)
    candidates = scratch.blocks * scratch.capacity
    scratch.q = wp.zeros(candidates, dtype=wp.vec3, device=device)
    scratch.k = wp.zeros(candidates, dtype=wp.vec3, device=device)
    scratch.n = wp.zeros(candidates, dtype=wp.vec3, device=device)
    scratch.d = wp.zeros(candidates, dtype=wp.float32, device=device)
    scratch.feature = wp.zeros(candidates, dtype=wp.int32, device=device)
    scratch.alive = wp.zeros(candidates, dtype=wp.int32, device=device)
    scratch.group = wp.zeros(candidates, dtype=wp.int32, device=device)
    scratch.rank = wp.zeros(candidates, dtype=wp.int32, device=device)
    scratch.mark = wp.zeros(candidates, dtype=wp.int32, device=device)
    return scratch


def terrain_law_rows(
    device,
    pair_count: int,
    slots: int,
    spill: int,
    deferred_first,
    *,
    status: wp.array | None = None,
    spilled: wp.array | None = None,
    reserved: wp.array | None = None,
    deferred_count: wp.array | None = None,
) -> TerrainLawRows:
    """Allocate a :class:`TerrainLawRows`: ``slots`` walk slots, a spill region of ``spill`` rows and per-pair arrays
    for ``pair_count`` pairs.

    Args:
        device: The device.
        pair_count: Pairs of the table (per-pair arrays and the deferred lists).
        slots: Walk slots of every pair together.
        spill: Rows of the spill region.
        deferred_first: Per partner class, the start of its deferred list (four entries).
        status: The narrow phase's sticky buffer status word, or ``None`` for one of its own.
        spilled: The one-entry counter of spill rows asked for (zeroed before every launch), or ``None`` for one of its
            own.
        reserved: The one-entry counter of spill rows reserved (zeroed before every launch), or ``None`` for one of
            its own.
        deferred_count: The per-class counters of deferred pairs (zeroed before every launch), or ``None`` for its own.
    """
    out = TerrainLawRows()
    rows = max(int(slots) + int(spill), 1)
    pairs = max(int(pair_count), 1)
    out.q = wp.zeros(rows, dtype=wp.vec3, device=device)
    out.n = wp.zeros(rows, dtype=wp.vec3, device=device)
    out.d = wp.zeros(rows, dtype=wp.float32, device=device)
    out.feature = wp.zeros(rows, dtype=wp.int32, device=device)
    out.slots = int(slots)
    out.capacity = int(slots) + int(spill)
    out.spilled = spilled if spilled is not None else wp.zeros(1, dtype=wp.int32, device=device)
    out.reserved = reserved if reserved is not None else wp.zeros(1, dtype=wp.int32, device=device)
    out.first = wp.full(pairs, -1, dtype=wp.int32, device=device)
    out.count = wp.zeros(pairs, dtype=wp.int32, device=device)
    out.state = wp.zeros(pairs, dtype=wp.int32, device=device)
    out.status = status if status is not None else wp.zeros(1, dtype=wp.int32, device=device)
    out.overflow = wp.zeros(3, dtype=wp.int32, device=device)
    out.deferred = wp.zeros(pairs, dtype=wp.int32, device=device)
    out.deferred_first = wp.array(np.asarray(deferred_first, np.int32), dtype=wp.int32, device=device)
    out.deferred_count = (
        deferred_count if deferred_count is not None else wp.zeros(len(deferred_first), dtype=wp.int32, device=device)
    )
    return out


# ----------------------------------------------------------------------------------------------------------------
# heights and the window-local frame
# ----------------------------------------------------------------------------------------------------------------


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __fadd_rn(min_z, __fmul_rn(elevation, __fsub_rn(max_z, min_z)));
#else
volatile float span = max_z - min_z;
volatile float scaled = elevation * span;
return min_z + scaled;
#endif
""")
def terrain_decode(min_z: float, max_z: float, elevation: float) -> float:
    """Return ``min_z + elevation * (max_z - min_z)`` in float32, each operation rounded once, on every device.

    This is the narrow phase's CPU decode. CUDA's default contraction would fuse the product and the sum into
    one rounding (and does so even through a float64 round trip, which the compiler narrows back to float32),
    so the CUDA path uses explicitly rounded intrinsics, which are never fused.
    """
    ...


@wp.func
def terrain_height(data: TerrainFeatureData, surface: TerrainSurfaceInfo, r: int, c: int) -> float:
    """Return the height of vertex ``(r, c)`` [m] (surface-local)."""
    return data.heights[surface.data_offset + r * surface.ncol + c]


@wp.func
def terrain_eps(extent: float) -> float:
    """Return the closed-predicate band for a window whose frame coordinates stay within ``extent`` [m].

    ``extent`` must bound the magnitude of every window-frame coordinate a predicate reads: the core point and
    the window's vertices, heights relative to the reference vertex included.
    """
    return _EPS_FACTOR * _EPS32 * wp.max(extent, _EPS_MIN_EXTENT)


@wp.func
def terrain_window(data: TerrainFeatureData, heightfield: int, r0: int, c0: int) -> TerrainWindow:
    """Return the window frame of reference vertex ``(r0, c0)`` of a heightfield (band unset, see :func:`terrain_eps`).

    The origin is the exact grid position ``(x0 + c0 dx, y0 + r0 dy)`` in float64: in float32 it would round
    by up to 4 um on a 200 m terrain and shift every core point against the grid by as much.
    """
    window = TerrainWindow()
    surface = data.surfaces[heightfield]
    window.surface = surface
    window.heightfield = heightfield
    window.r0 = r0
    window.c0 = c0
    window.z_ref = terrain_height(data, surface, r0, c0)
    window.origin = wp.vec3d(
        surface.x0_64 + wp.float64(c0) * surface.dx64,
        surface.y0_64 + wp.float64(r0) * surface.dy64,
        wp.float64(window.z_ref),
    )
    window.eps = 0.0
    return window


@wp.func
def terrain_window_point(window: TerrainWindow, p: wp.vec3) -> wp.vec3:
    """Return heightfield-local point ``p`` in the window frame, rounded once (from float64)."""
    return wp.vec3(
        wp.float32(wp.float64(p[0]) - window.origin[0]),
        wp.float32(wp.float64(p[1]) - window.origin[1]),
        wp.float32(wp.float64(p[2]) - window.origin[2]),
    )


@wp.func
def terrain_window_to_local(window: TerrainWindow, k: wp.vec3) -> wp.vec3:
    """Return window-frame point ``k`` in the heightfield-local frame, rounded once (from float64)."""
    return wp.vec3(
        wp.float32(wp.float64(k[0]) + window.origin[0]),
        wp.float32(wp.float64(k[1]) + window.origin[1]),
        wp.float32(wp.float64(k[2]) + window.origin[2]),
    )


@wp.func
def terrain_window_vertex(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int) -> wp.vec3:
    """Return vertex ``(r, c)`` in the window frame, from integer index differences."""
    z = terrain_height(data, window.surface, r, c) - window.z_ref
    return wp.vec3(float(c - window.c0) * window.surface.dx, float(r - window.r0) * window.surface.dy, z)


@wp.func
def terrain_valid_cell(surface: TerrainSurfaceInfo, r: int, c: int) -> bool:
    """Return whether cell ``(r, c)`` exists."""
    return r >= 0 and c >= 0 and r < surface.nrow - 1 and c < surface.ncol - 1


@wp.func
def terrain_triangle_vertices(r: int, c: int, sub: int) -> tuple[wp.vec2i, wp.vec2i, wp.vec2i]:
    """Return the vertices ``(row, column)`` of triangle ``sub`` of cell ``(r, c)`` in Newton's order.

    Sub 0 is ``(p00, p10, p11)`` and sub 1 is ``(p00, p11, p01)``, with ``p10 = (r, c + 1)``.
    """
    if sub == 0:
        return wp.vec2i(r, c), wp.vec2i(r, c + 1), wp.vec2i(r + 1, c + 1)
    return wp.vec2i(r, c), wp.vec2i(r + 1, c + 1), wp.vec2i(r + 1, c)


@wp.func
def terrain_triangle_id(surface: TerrainSurfaceInfo, r: int, c: int, sub: int) -> int:
    """Return the local triangle id ``2 * (r * cols + c) + sub`` (the contact fingerprints' numbering)."""
    return 2 * (r * (surface.ncol - 1) + c) + sub


@wp.func
def terrain_cluster(data: TerrainFeatureData, surface: TerrainSurfaceInfo, r: int, c: int, sub: int) -> int:
    """Return the cluster id of triangle ``sub`` of cell ``(r, c)`` (equal ids: one cluster)."""
    return data.clusters[surface.triangle_offset + terrain_triangle_id(surface, r, c, sub)]


@wp.func
def terrain_cluster_plane(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, sub: int
) -> tuple[wp.vec3, wp.vec3]:
    """Return the unit upward normal and a point of the plane of a triangle's cluster, in the window frame.

    A cluster of several triangles reads its plane record and evaluates the plane over the window's reference
    vertex, in float64 from exact grid steps, so the point is window-local however far the cluster's anchor (its
    lowest-id vertex) lies: a point at a far anchor would carry the float32 rounding of that distance into every
    plane distance. A triangle alone uses its own plane, from edge vectors of integer index differences.
    """
    surface = window.surface
    cluster = terrain_cluster(data, surface, r, c, sub)
    if cluster >= 0:
        plane = data.planes[surface.plane_offset + cluster]
        ra = plane.anchor // surface.ncol
        ca = plane.anchor - ra * surface.ncol
        n = wp.normalize(wp.vec3(-plane.gx, -plane.gy, 1.0))
        # z(anchor) + offset + g . (reference - anchor) - z_ref; a gradient times a grid step count is exact.
        dx = surface.dx64
        dy = surface.dy64
        z = (
            (wp.float64(terrain_height(data, surface, ra, ca)) - wp.float64(window.z_ref))
            + wp.float64(plane.offset)
            + wp.float64(plane.gx) * wp.float64(window.c0 - ca) * dx
            + wp.float64(plane.gy) * wp.float64(window.r0 - ra) * dy
        )
        return n, wp.vec3(0.0, 0.0, wp.float32(z))
    a, b, d = terrain_triangle_vertices(r, c, sub)
    za = terrain_height(data, surface, a[0], a[1])
    e1 = wp.vec3(
        float(b[1] - a[1]) * surface.dx,
        float(b[0] - a[0]) * surface.dy,
        terrain_height(data, surface, b[0], b[1]) - za,
    )
    e2 = wp.vec3(
        float(d[1] - a[1]) * surface.dx,
        float(d[0] - a[0]) * surface.dy,
        terrain_height(data, surface, d[0], d[1]) - za,
    )
    n = wp.normalize(wp.cross(e1, e2))
    if n[2] < 0.0:
        n = -n
    return n, terrain_window_vertex(data, window, a[0], a[1])


@wp.func
def terrain_planes() -> TerrainPlanes:
    """Return an empty plane cache for a window."""
    planes = TerrainPlanes()
    planes.id0 = -1
    planes.id1 = -1
    planes.id2 = -1
    planes.id3 = -1
    return planes


@wp.func
def terrain_plane(
    data: TerrainFeatureData, window: TerrainWindow, planes: TerrainPlanes, r: int, c: int, sub: int
) -> tuple[TerrainPlanes, wp.vec3, wp.vec3]:
    """Return :func:`terrain_cluster_plane` of a triangle, evaluating a cluster of several triangles once per window.

    Returns the updated cache (:func:`terrain_planes`), the plane's unit upward normal and a point of it (window
    frame). A triangle alone is evaluated afresh (float32 only); a cluster's plane is read from the cache when held,
    with the bits of a fresh evaluation.
    """
    cluster = terrain_cluster(data, window.surface, r, c, sub)
    if cluster >= 0:
        held = wp.vec4(0.0)
        found = bool(True)
        if planes.id0 == cluster:
            held = planes.plane0
        elif planes.id1 == cluster:
            held = planes.plane1
        elif planes.id2 == cluster:
            held = planes.plane2
        elif planes.id3 == cluster:
            held = planes.plane3
        else:
            found = False
        if found:
            return planes, wp.vec3(held[0], held[1], held[2]), wp.vec3(0.0, 0.0, held[3])
    n, a = terrain_cluster_plane(data, window, r, c, sub)
    if cluster >= 0:
        entry = wp.vec4(n[0], n[1], n[2], a[2])
        if planes.next == 0:
            planes.id0 = cluster
            planes.plane0 = entry
        elif planes.next == 1:
            planes.id1 = cluster
            planes.plane1 = entry
        elif planes.next == 2:
            planes.id2 = cluster
            planes.plane2 = entry
        else:
            planes.id3 = cluster
            planes.plane3 = entry
        planes.next = (planes.next + 1) % 4
    return planes, n, a


# ----------------------------------------------------------------------------------------------------------------
# canonical edge functions and closed predicates
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_face_edge_value(k: wp.vec3, v0: wp.vec3, v1: wp.vec3, n: wp.vec3, w: wp.vec3) -> float:
    """Return ``s = (k - v0) . m``, ``m`` the in-plane outward normal of a triangle at its edge ``v0 v1``.

    ``n`` is the triangle's (cluster) plane normal and ``w`` its third vertex. ``v0`` must be the edge's lower
    vertex: the triangles on both sides of an edge then read the same ``cross(v1 - v0, n)``, and inside one
    cluster their values are exact negatives.
    """
    m = wp.cross(v1 - v0, n)
    if wp.dot(m, w - v0) > 0.0:
        m = -m
    return wp.dot(k - v0, wp.normalize(m))


@wp.func
def terrain_vertex_edge_value(k: wp.vec3, v: wp.vec3, w: wp.vec3) -> float:
    """Return ``(k - v) . (w - v)``: the value vertex ``v``'s Voronoi test and edge ``v w``'s slab test share."""
    return wp.dot(k - v, w - v)


@wp.func
def terrain_claims(value: float, band: float) -> bool:
    """Return whether a region bounded by ``value <= 0`` claims ``value`` (closed by ``band``)."""
    return value <= band


@wp.func
def terrain_beyond(value: float, band: float) -> bool:
    """Return whether the region beyond a boundary ``value > 0`` claims ``value`` (closed by ``band``)."""
    return value > -band


@wp.func
def terrain_rows_coincide(q_a: wp.vec3, n_a: wp.vec3, q_b: wp.vec3, n_b: wp.vec3, eps: float) -> bool:
    """Return whether two rows are one constraint: terrain points within ``4 eps``, normals within ``1e-3``."""
    return wp.length(q_a - q_b) <= _MERGE_SCALE * eps and wp.length(n_a - n_b) <= _MERGE_NORMAL


@wp.func
def terrain_rows_same_plane(q_a: wp.vec3, n_a: wp.vec3, q_b: wp.vec3, n_b: wp.vec3, eps: float) -> bool:
    """Return whether row ``a`` lies in row ``b``'s plane (the finish's plane groups): normals within
    :data:`TERRAIN_GROUP_NORMAL`, ``a``'s terrain point within ``4 eps`` of ``b``'s plane."""
    return wp.length(n_a - n_b) <= _GROUP_NORMAL and wp.abs(wp.dot(n_b, q_a - q_b)) <= _MERGE_SCALE * eps


@wp.func
def terrain_face_values(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, sub: int, n: wp.vec3, k: wp.vec3
) -> wp.vec3:
    """Return the canonical edge values of a triangle for point ``k`` (plane normal ``n``).

    Components follow the triangle's edges ``(p0, p1)``, ``(p1, p2)``, ``(p2, p0)``; ``k`` projects into the
    triangle when all three are at most zero.
    """
    a, b, d = terrain_triangle_vertices(r, c, sub)
    pa = terrain_window_vertex(data, window, a[0], a[1])
    pb = terrain_window_vertex(data, window, b[0], b[1])
    pd = terrain_window_vertex(data, window, d[0], d[1])
    # Each edge with its lower vertex first: sub 0 (p00, p10), (p10, p11), (p00, p11); sub 1 (p00, p11),
    # (p01, p11), (p00, p01).
    s0 = terrain_face_edge_value(k, pa, pb, n, pd)
    s1 = float(0.0)
    if sub == 0:
        s1 = terrain_face_edge_value(k, pb, pd, n, pa)
    else:
        s1 = terrain_face_edge_value(k, pd, pb, n, pa)
    s2 = terrain_face_edge_value(k, pa, pd, n, pb)
    return wp.vec3(s0, s1, s2)


@wp.func
def terrain_edge_sides(r: int, c: int, which: int) -> tuple[wp.vec2i, wp.vec2i, wp.vec3i, wp.vec3i, wp.vec2i, wp.vec2i]:
    """Return an edge's lower and upper vertex, its two triangles ``(r, c, sub)`` and their third vertices."""
    if which == 0:
        return (
            wp.vec2i(r, c),
            wp.vec2i(r, c + 1),
            wp.vec3i(r - 1, c, 1),
            wp.vec3i(r, c, 0),
            wp.vec2i(r - 1, c),
            wp.vec2i(r + 1, c + 1),
        )
    if which == 1:
        return (
            wp.vec2i(r, c),
            wp.vec2i(r + 1, c),
            wp.vec3i(r, c - 1, 0),
            wp.vec3i(r, c, 1),
            wp.vec2i(r, c - 1),
            wp.vec2i(r + 1, c + 1),
        )
    return (
        wp.vec2i(r, c),
        wp.vec2i(r + 1, c + 1),
        wp.vec3i(r, c, 0),
        wp.vec3i(r, c, 1),
        wp.vec2i(r, c + 1),
        wp.vec2i(r + 1, c),
    )


@wp.func
def _edge_kind_of_clusters(data: TerrainFeatureData, surface: TerrainSurfaceInfo, r: int, c: int, which: int) -> int:
    """:func:`terrain_edge_kind` from the cluster ids (what the vertex bits record)."""
    _v0, _v1, ta, tb, _wa, _wb = terrain_edge_sides(r, c, which)
    if not terrain_valid_cell(surface, ta[0], ta[1]) or not terrain_valid_cell(surface, tb[0], tb[1]):
        return TERRAIN_EDGE_BORDER
    if terrain_cluster(data, surface, ta[0], ta[1], ta[2]) == terrain_cluster(data, surface, tb[0], tb[1], tb[2]):
        return TERRAIN_EDGE_FLAT
    return -1


@wp.func
def terrain_vertex_bits(data: TerrainFeatureData, surface: TerrainSurfaceInfo, r: int, c: int) -> int:
    """Return vertex ``(r, c)``'s bits (:attr:`TerrainFeatureData.vertex_bits`)."""
    return int(data.vertex_bits[surface.data_offset + r * surface.ncol + c])


@wp.func
def terrain_edge_kind(data: TerrainFeatureData, surface: TerrainSurfaceInfo, r: int, c: int, which: int) -> int:
    """Return :data:`TERRAIN_EDGE_BORDER` or :data:`TERRAIN_EDGE_FLAT` of edge ``which`` of vertex ``(r, c)``, else -1.

    Decided from the cluster ids alone, as the vertex's bits record it; a crease (-1) needs its first triangle's plane
    (:func:`terrain_crease_class`). A vertex off the grid decides it from the cluster ids directly.
    """
    if r < 0 or c < 0 or r >= surface.nrow or c >= surface.ncol:
        return _edge_kind_of_clusters(data, surface, r, c, which)
    code = (terrain_vertex_bits(data, surface, r, c) >> (2 * which)) & 3
    if code == TERRAIN_VERTEX_EDGE_FLAT:
        return TERRAIN_EDGE_FLAT
    if code == TERRAIN_VERTEX_EDGE_BORDER:
        return TERRAIN_EDGE_BORDER
    return -1


@wp.func
def terrain_crease_class(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int, n: wp.vec3, a: wp.vec3
) -> int:
    """Return whether crease ``which`` of vertex ``(r, c)`` is concave or convex, given its first triangle's plane.

    The far vertex of the second triangle above the first triangle's cluster plane (normal ``n`` through ``a``)
    makes a valley (concave).
    """
    _v0, _v1, _ta, _tb, _wa, wb = terrain_edge_sides(r, c, which)
    if wp.dot(n, terrain_window_vertex(data, window, wb[0], wb[1]) - a) > 0.0:
        return TERRAIN_EDGE_CONCAVE
    return TERRAIN_EDGE_CONVEX


@wp.func
def terrain_edge_class(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int) -> int:
    """Return the class of edge ``which`` of vertex ``(r, c)``: flat inside one cluster, else convex or concave.

    The far vertex of the second triangle above the first triangle's cluster plane makes a valley (concave);
    two different clusters always meet at a crease, however small.
    """
    kind = terrain_edge_kind(data, window.surface, r, c, which)
    if kind >= 0:
        return kind
    _v0, _v1, ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    n, a = terrain_cluster_plane(data, window, ta[0], ta[1], ta[2])
    return terrain_crease_class(data, window, r, c, which, n, a)


@wp.func
def terrain_ring_vertex(r: int, c: int, j: int) -> wp.vec2i:
    """Return one-ring neighbour ``j`` (0..5, counter-clockwise) of vertex ``(r, c)``."""
    if j == 0:
        return wp.vec2i(r, c + 1)
    if j == 1:
        return wp.vec2i(r + 1, c + 1)
    if j == 2:
        return wp.vec2i(r + 1, c)
    if j == 3:
        return wp.vec2i(r, c - 1)
    if j == 4:
        return wp.vec2i(r - 1, c - 1)
    return wp.vec2i(r - 1, c)


@wp.func
def terrain_incident_triangle(r: int, c: int, j: int) -> wp.vec3i:
    """Return incident triangle ``j`` (0..5) of vertex ``(r, c)`` as ``(row, column, sub)``, matching the ring."""
    if j == 0:
        return wp.vec3i(r, c, 0)
    if j == 1:
        return wp.vec3i(r, c, 1)
    if j == 2:
        return wp.vec3i(r, c - 1, 0)
    if j == 3:
        return wp.vec3i(r - 1, c - 1, 1)
    if j == 4:
        return wp.vec3i(r - 1, c - 1, 0)
    return wp.vec3i(r - 1, c, 1)


@wp.func
def terrain_interior_vertex(surface: TerrainSurfaceInfo, r: int, c: int) -> bool:
    """Return whether vertex ``(r, c)`` has a full one-ring."""
    return r >= 1 and c >= 1 and r < surface.nrow - 1 and c < surface.ncol - 1


@wp.func
def terrain_grid_vertex(surface: TerrainSurfaceInfo, r: int, c: int) -> bool:
    """Return whether vertex ``(r, c)`` exists."""
    return r >= 0 and c >= 0 and r < surface.nrow and c < surface.ncol


# ----------------------------------------------------------------------------------------------------------------
# the border: walls, border edges, border vertices and corner lines
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_border_side(surface: TerrainSurfaceInfo, r: int, c: int, which: int) -> int:
    """Return the wall of edge ``which`` of vertex ``(r, c)`` when the edge lies on the heightfield's border, else -1:
    0 and 1 the walls ``y = -hy`` and ``y = hy`` (the x-edges of rows 0 and ``nrow - 1``), 2 and 3 the walls
    ``x = -hx`` and ``x = hx`` (the y-edges of columns 0 and ``ncol - 1``)."""
    if which == 0:
        if c >= 0 and c < surface.ncol - 1:
            if r == 0:
                return 0
            if r == surface.nrow - 1:
                return 1
    elif which == 1:
        if r >= 0 and r < surface.nrow - 1:
            if c == 0:
                return 2
            if c == surface.ncol - 1:
                return 3
    return -1


@wp.func
def terrain_wall_normal(side: int) -> wp.vec3:
    """Return the outward unit normal of wall ``side`` (:func:`terrain_border_side`)."""
    if side == 0:
        return wp.vec3(0.0, -1.0, 0.0)
    if side == 1:
        return wp.vec3(0.0, 1.0, 0.0)
    if side == 2:
        return wp.vec3(-1.0, 0.0, 0.0)
    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def terrain_wall_value(k: wp.vec3, v: wp.vec3, side: int) -> float:
    """Return ``(k - v) . n``, ``n`` wall ``side``'s outward normal: point ``k``'s distance beyond the wall through
    ``v``, one coordinate difference (exact negations of each other for opposite directions)."""
    if side == 0:
        return v[1] - k[1]
    if side == 1:
        return k[1] - v[1]
    if side == 2:
        return v[0] - k[0]
    return k[0] - v[0]


@wp.func
def terrain_beyond_walls(window: TerrainWindow, k: wp.vec3) -> float:
    """Return how far window-frame point ``k`` lies beyond the heightfield's walls: the largest of its four walls'
    values (:func:`terrain_wall_value`, from the border vertices' coordinates), at most 0 over the footprint."""
    surface = window.surface
    x_lo = float(-window.c0) * surface.dx
    x_hi = float(surface.ncol - 1 - window.c0) * surface.dx
    y_lo = float(-window.r0) * surface.dy
    y_hi = float(surface.nrow - 1 - window.r0) * surface.dy
    return wp.max(wp.max(x_lo - k[0], k[0] - x_hi), wp.max(y_lo - k[1], k[1] - y_hi))


@wp.func
def terrain_wall_along(k: wp.vec3, v: wp.vec3, side: int) -> float:
    """Return ``(k - v) . t``, ``t`` the axis along wall ``side`` toward higher vertex indices (x for walls 0 and 1, y
    for walls 2 and 3): the value both wall pieces at a border vertex, or a wall piece and a corner line, share."""
    if side < 2:
        return k[0] - v[0]
    return k[1] - v[1]


@wp.func
def terrain_border_triangle(r: int, c: int, which: int, side: int) -> wp.vec3i:
    """Return the one triangle ``(row, column, sub)`` of border edge ``which`` of vertex ``(r, c)`` on wall ``side``."""
    _v0, _v1, ta, tb, _wa, _wb = terrain_edge_sides(r, c, which)
    if side == 0 or side == 2:
        return tb
    return ta


@wp.func
def terrain_border_far(r: int, c: int, which: int, side: int) -> wp.vec2i:
    """Return the third vertex of a border edge's triangle (:func:`terrain_border_triangle`)."""
    _v0, _v1, _ta, _tb, wa, wb = terrain_edge_sides(r, c, which)
    if side == 0 or side == 2:
        return wb
    return wa


@wp.func
def terrain_wall_frame(v0: wp.vec3, v1: wp.vec3, n_wall: wp.vec3) -> wp.vec3:
    """Return the in-wall unit normal of border edge ``v0 v1`` pointing up, away from the wall below the edge."""
    m = wp.cross(v1 - v0, n_wall)
    if m[2] < 0.0:
        m = -m
    return wp.normalize(m)


@wp.func
def terrain_wall_edge_value(k: wp.vec3, v0: wp.vec3, v1: wp.vec3, n_wall: wp.vec3) -> float:
    """Return ``s = (k - v0) . m`` (:func:`terrain_wall_frame`) of border edge ``v0 v1`` (``v0`` its lower vertex): the
    value the border edge's claim above the wall and the wall's claim below the edge share, as
    :func:`terrain_face_edge_value` is a face's and its edge's."""
    return wp.dot(k - v0, terrain_wall_frame(v0, v1, n_wall))


@wp.func
def terrain_wall_holds(eps: float, side: int, v0: wp.vec3, v1: wp.vec3, k: wp.vec3) -> bool:
    """Return whether the wall below border edge ``v0 v1`` (wall ``side``) holds point ``k``'s projection on it (closed
    by ``eps``): below the edge, between the vertical lines through its vertices and above the wall's base,
    :data:`TERRAIN_WALL_DEPTH` below the edge. The tests are along the wall, so that ``k`` and its projection decide
    alike."""
    m = terrain_wall_frame(v0, v1, terrain_wall_normal(side))
    s = wp.dot(k - v0, m)
    if not terrain_claims(s, eps):
        return False
    if s < -_WALL_DEPTH * m[2] - eps:
        return False
    return terrain_beyond(terrain_wall_along(k, v0, side), eps) and terrain_claims(terrain_wall_along(k, v1, side), eps)


@wp.func
def terrain_vertex_sides(surface: TerrainSurfaceInfo, r: int, c: int) -> wp.vec2i:
    """Return the walls through vertex ``(r, c)`` (:func:`terrain_border_side`): ``(-1, -1)`` for an interior vertex,
    one of them for a border vertex, two for a corner of the footprint (the y-wall first)."""
    a = int(-1)
    b = int(-1)
    if r == 0:
        a = 0
    elif r == surface.nrow - 1:
        a = 1
    if c == 0:
        b = 2
    elif c == surface.ncol - 1:
        b = 3
    if a < 0:
        return wp.vec2i(b, -1)
    return wp.vec2i(a, b)


_RingNormals = wp.types.matrix(shape=(6, 3), dtype=wp.float32)


@wp.func
def _ring_normal(normals: _RingNormals, i: int) -> wp.vec3:
    return wp.vec3(normals[i, 0], normals[i, 1], normals[i, 2])


@wp.func
def _ring_add(normals: _RingNormals, count: int, n: wp.vec3) -> tuple[_RingNormals, int]:
    """Add a normal to a vertex's distinct incident normals (closer than :data:`TERRAIN_NORMAL_SAME`: one)."""
    fresh = bool(True)
    for i in range(count):
        if wp.length(n - _ring_normal(normals, i)) <= _NORMAL_SAME:
            fresh = False
    if fresh:
        normals[count, 0] = n[0]
        normals[count, 1] = n[1]
        normals[count, 2] = n[2]
        count += 1
    return normals, count


@wp.func
def _vertex_feature_of_planes(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int) -> bool:
    """:func:`terrain_vertex_feature` from the incident clusters' planes (what the vertex bits record). The cluster
    normals do not depend on the window: a cluster's from its gradient, a lone triangle's from edge vectors of integer
    index differences and height differences. A border vertex's incident normals are its triangles' and its walls'
    (at most three triangles and one wall, or two triangles and two walls at a corner of the footprint)."""
    surface = window.surface
    if not terrain_grid_vertex(surface, r, c):
        return False
    normals = _RingNormals()
    count = int(0)
    for j in range(6):
        t = terrain_incident_triangle(r, c, j)
        if terrain_valid_cell(surface, t[0], t[1]):
            n, _a = terrain_cluster_plane(data, window, t[0], t[1], t[2])
            normals, count = _ring_add(normals, count, n)
    sides = terrain_vertex_sides(surface, r, c)
    for j in range(2):
        if sides[j] >= 0:
            normals, count = _ring_add(normals, count, terrain_wall_normal(sides[j]))
    for i in range(count):
        ni = _ring_normal(normals, i)
        for j in range(i + 1, count):
            dj = _ring_normal(normals, j) - ni
            for k in range(j + 1, count):
                if wp.abs(wp.dot(ni, wp.cross(dj, _ring_normal(normals, k) - ni))) > _VERTEX_SPAN:
                    return True
    return False


@wp.func
def _grid_point64(surface: TerrainSurfaceInfo, z: float, r: int, c: int) -> wp.vec3d:
    """Vertex ``(r, c)`` at height ``z`` on the exact grid (heightfield-local, float64)."""
    return wp.vec3d(wp.float64(c) * surface.dx64, wp.float64(r) * surface.dy64, wp.float64(z))


@wp.func
def terrain_crease_static(data: TerrainFeatureData, surface: TerrainSurfaceInfo, r: int, c: int, which: int) -> int:
    """Return crease ``which`` of vertex ``(r, c)``'s class in every window frame whose points it reads lie within
    :data:`TERRAIN_CREASE_STATIC_CELLS` cells of the window's reference vertex: 1 concave in all of them (the static
    class of :data:`TERRAIN_VERTEX_EDGE_CONCAVE`), -1 convex in all of them, 0 when frames may disagree.

    :func:`terrain_crease_class` takes ``n . (w_b - a)`` in float32 in a window frame: ``n`` the unit normal of the
    first triangle's cluster plane, ``a`` its point over the reference vertex (a lone triangle's first vertex), ``w_b``
    the second triangle's far vertex. Its exact value, ``n_z`` times the height of ``w_b`` over the plane, is the same
    in every frame and is taken here in float64. The frame's coordinates lie within ``W = (cells + 1) max(dx, dy)``
    across, its heights within ``H = |max_z - min_z| + flat`` of the reference vertex's, and the plane's point over the
    reference vertex within ``H + sqrt(2) s W`` (``s`` the plane's slope): every rounding of the frame's test (its
    inputs, its normal's normalization, a lone triangle's normal from rounded edges, its differences, products and
    sums) stays below ``20 eps32 (1 + s) (W + H)``. The far vertex must clear the plane by
    :data:`TERRAIN_CREASE_STATIC_ROUNDING` ``eps32 2 (1 + s) (W + H)``, over six times that, above the plane for the
    concave class, below it for the convex one (the frame's test classes a value of 0 convex)."""
    _v0i, _v1i, ta, tb, _wa, wb = terrain_edge_sides(r, c, which)
    if not terrain_valid_cell(surface, ta[0], ta[1]) or not terrain_valid_cell(surface, tb[0], tb[1]):
        return 0
    w = _grid_point64(surface, terrain_height(data, surface, wb[0], wb[1]), wb[0], wb[1])
    cluster = terrain_cluster(data, surface, ta[0], ta[1], ta[2])
    value = wp.float64(0.0)
    slope = wp.float64(0.0)
    if cluster >= 0:
        plane = data.planes[surface.plane_offset + cluster]
        ra = plane.anchor // surface.ncol
        ca = plane.anchor - ra * surface.ncol
        anchor = _grid_point64(surface, terrain_height(data, surface, ra, ca), ra, ca)
        gx = wp.float64(plane.gx)
        gy = wp.float64(plane.gy)
        height = anchor[2] + wp.float64(plane.offset) + gx * (w[0] - anchor[0]) + gy * (w[1] - anchor[1])
        norm = wp.sqrt(gx * gx + gy * gy + wp.float64(1.0))
        value = (w[2] - height) / norm
        slope = wp.sqrt(gx * gx + gy * gy)
    else:
        a, b, d = terrain_triangle_vertices(ta[0], ta[1], ta[2])
        pa = _grid_point64(surface, terrain_height(data, surface, a[0], a[1]), a[0], a[1])
        pb = _grid_point64(surface, terrain_height(data, surface, b[0], b[1]), b[0], b[1])
        pd = _grid_point64(surface, terrain_height(data, surface, d[0], d[1]), d[0], d[1])
        n = wp.cross(pb - pa, pd - pa)
        if n[2] < wp.float64(0.0):
            n = -n
        length = wp.length(n)
        if length <= wp.float64(0.0):
            return 0
        value = wp.dot(n, w - pa) / length
        slope = wp.sqrt(n[0] * n[0] + n[1] * n[1]) / n[2]
    cells = wp.float64(_CREASE_CELLS + 1)
    span = cells * wp.max(wp.float64(surface.dx), wp.float64(surface.dy))
    heights = wp.abs(wp.float64(surface.max_z) - wp.float64(surface.min_z)) + wp.float64(_CREASE_FLAT)
    margin = (
        wp.float64(_CREASE_ROUNDING)
        * wp.float64(_EPS32)
        * wp.float64(2.0)
        * (wp.float64(1.0) + slope)
        * (span + heights)
    )
    if value > margin:
        return 1
    if value < -margin:
        return -1
    return 0


@wp.func
def terrain_edge_open_crease(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int) -> bool:
    """Return whether edge ``which`` of window vertex ``(r, c)`` is a crease that may be convex in the window's frame:
    not flat, not on the border (:func:`terrain_edge_kind`), and not of the static class
    :data:`TERRAIN_VERTEX_EDGE_CONCAVE`, which every frame of :data:`TERRAIN_CREASE_STATIC_CELLS` cells classes concave
    (:func:`terrain_crease_static`; farther from the reference vertex the frame's test decides). One read of the
    vertex's bits."""
    surface = window.surface
    if r < 0 or c < 0 or r >= surface.nrow or c >= surface.ncol:
        return terrain_edge_kind(data, surface, r, c, which) < 0
    code = (terrain_vertex_bits(data, surface, r, c) >> (2 * which)) & 3
    if code == TERRAIN_VERTEX_EDGE_CONCAVE:
        return r - window.r0 > _CREASE_CELLS - 2 or c - window.c0 > _CREASE_CELLS - 2
    return code == 0


@wp.func
def terrain_vertex_bits_of_clusters(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int) -> int:
    """Return vertex ``(r, c)``'s bits from the clusters and heights (:attr:`TerrainFeatureData.vertex_bits`): per
    edge it owns (x-, y-, diagonal edge), two bits from bit ``2 which``: :data:`TERRAIN_VERTEX_EDGE_FLAT`,
    :data:`TERRAIN_VERTEX_EDGE_BORDER`, :data:`TERRAIN_VERTEX_EDGE_CONCAVE` (:func:`terrain_crease_static`) or 0 for
    any other crease; :data:`TERRAIN_VERTEX_FEATURE_BIT` for a vertex feature; and
    :data:`TERRAIN_VERTEX_CREASE_UNSURE_BIT` when a crease of code 0 is not convex in every frame."""
    bits = int(0)
    for which in range(3):
        kind = _edge_kind_of_clusters(data, window.surface, r, c, which)
        if kind == TERRAIN_EDGE_FLAT:
            bits = bits | (TERRAIN_VERTEX_EDGE_FLAT << (2 * which))
        elif kind == TERRAIN_EDGE_BORDER:
            bits = bits | (TERRAIN_VERTEX_EDGE_BORDER << (2 * which))
        else:
            static = terrain_crease_static(data, window.surface, r, c, which)
            if static > 0:
                bits = bits | (TERRAIN_VERTEX_EDGE_CONCAVE << (2 * which))
            elif static == 0:
                bits = bits | TERRAIN_VERTEX_CREASE_UNSURE_BIT
    if _vertex_feature_of_planes(data, window, r, c):
        bits = bits | TERRAIN_VERTEX_FEATURE_BIT
    return bits


@wp.func
def terrain_vertex_feature(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int) -> bool:
    """Return whether vertex ``(r, c)`` is a feature of its own: its incident cluster normals span 3-D.

    Normals closer than :data:`TERRAIN_NORMAL_SAME` are one direction; three distinct directions span 3-D when
    their triple product exceeds :data:`TERRAIN_VERTEX_SPAN`. The product is taken of their differences, which
    float32 resolves to about 1e-13 for normals close together (the plain product only to about 1e-7). Read from the
    vertex's bits, built with the clusters (:func:`terrain_vertex_bits_of_clusters`); a vertex off the grid is none.
    """
    surface = window.surface
    if r < 0 or c < 0 or r >= surface.nrow or c >= surface.ncol:
        return False
    return (terrain_vertex_bits(data, surface, r, c) & TERRAIN_VERTEX_FEATURE_BIT) != 0


@wp.func
def terrain_vertex_in_cone(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, n: wp.vec3) -> bool:
    """Return whether direction ``n`` lies in vertex ``(r, c)``'s normal cone (closed): ``n . (w - v) <= 0`` for its
    ring's vertices in the grid, and at a border vertex for the wall below it (``n . (-z) <= 0``)."""
    surface = window.surface
    v = terrain_window_vertex(data, window, r, c)
    for j in range(6):
        w = terrain_ring_vertex(r, c, j)
        if terrain_grid_vertex(surface, w[0], w[1]):
            e = terrain_window_vertex(data, window, w[0], w[1]) - v
            if wp.dot(n, e) > _CONE_BAND * wp.length(e):
                return False
    if not terrain_interior_vertex(surface, r, c):
        if -n[2] > _CONE_BAND:
            return False
    return True


# ----------------------------------------------------------------------------------------------------------------
# point claims (the predicates of a point core: sphere centre, capsule end, a solid's manifold point)
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_face_point(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, sub: int, k: wp.vec3
) -> tuple[bool, wp.vec3, wp.vec3, float]:
    """Return whether a triangle claims point ``k``'s projection on its cluster plane, and the row.

    Returns ``(claims, q, n, d)``: the projection ``q`` lies in the triangle up to the band, ``n`` is the
    cluster normal and ``d = n . (k - q)``. Reach and facing are left to the caller.
    """
    n, a = terrain_cluster_plane(data, window, r, c, sub)
    s = terrain_face_values(data, window, r, c, sub, n, k)
    claims = terrain_claims(s[0], window.eps) and terrain_claims(s[1], window.eps) and terrain_claims(s[2], window.eps)
    d = wp.dot(n, k - a)
    return claims, k - d * n, n, d


@wp.func
def terrain_edge_values(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int, k: wp.vec3
) -> wp.vec4:
    """Return an edge's predicate values for point ``k``: ``(s_a, s_b, w_0, w_1)``.

    ``s_a`` and ``s_b`` are the values the two faces read at this edge (:func:`terrain_face_values`), ``w_0``
    and ``w_1`` the values the edge's end vertices read for each other (:func:`terrain_vertex_edge_value`).
    The edge must have both triangles.
    """
    v0i, v1i, ta, tb, wa, wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    na, _pa = terrain_cluster_plane(data, window, ta[0], ta[1], ta[2])
    nb, _pb = terrain_cluster_plane(data, window, tb[0], tb[1], tb[2])
    sa = terrain_face_edge_value(k, v0, v1, na, terrain_window_vertex(data, window, wa[0], wa[1]))
    sb = terrain_face_edge_value(k, v0, v1, nb, terrain_window_vertex(data, window, wb[0], wb[1]))
    return wp.vec4(sa, sb, terrain_vertex_edge_value(k, v0, v1), terrain_vertex_edge_value(k, v1, v0))


@wp.func
def terrain_edge_point(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int, k: wp.vec3
) -> tuple[bool, wp.vec3]:
    """Return whether an edge claims point ``k`` (beyond both faces, inside the edge's slab), and the closest point.

    The face values are the ones :func:`terrain_face_point` reads; a border edge claims nothing. Edge classes
    are left to the caller.
    """
    v0i, v1i, ta, tb, _wa, _wb = terrain_edge_sides(r, c, which)
    surface = window.surface
    if not terrain_valid_cell(surface, ta[0], ta[1]) or not terrain_valid_cell(surface, tb[0], tb[1]):
        return False, wp.vec3(0.0)
    values = terrain_edge_values(data, window, r, c, which, k)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    u = terrain_window_vertex(data, window, v1i[0], v1i[1]) - v0
    length = wp.length(u)
    eps = window.eps
    claims = (
        terrain_beyond(values[0], eps)
        and terrain_beyond(values[1], eps)
        and terrain_beyond(values[2], eps * length)
        and terrain_beyond(values[3], eps * length)
    )
    t = wp.clamp(values[2] / wp.dot(u, u), 0.0, 1.0)
    return claims, v0 + t * u


@wp.func
def terrain_border_edge_values(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int, side: int, k: wp.vec3
) -> wp.vec4:
    """Return a border edge's predicate values for point ``k``: ``(s_top, s_wall, w_0, w_1)``, the value its triangle
    reads at the edge (:func:`terrain_face_values`), the value its wall reads at the edge
    (:func:`terrain_wall_edge_value`) and the values its end vertices read for each other, as
    :func:`terrain_edge_values` gives an edge's with the wall as its second face."""
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    t = terrain_border_triangle(r, c, which, side)
    far = terrain_border_far(r, c, which, side)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    n, _a = terrain_cluster_plane(data, window, t[0], t[1], t[2])
    s_top = terrain_face_edge_value(k, v0, v1, n, terrain_window_vertex(data, window, far[0], far[1]))
    s_wall = terrain_wall_edge_value(k, v0, v1, terrain_wall_normal(side))
    return wp.vec4(s_top, s_wall, terrain_vertex_edge_value(k, v0, v1), terrain_vertex_edge_value(k, v1, v0))


@wp.func
def terrain_border_point(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int, side: int, k: wp.vec3
) -> tuple[int, wp.vec3, wp.vec3, float]:
    """Return the claim of border edge ``which`` of vertex ``(r, c)`` (wall ``side``) and its wall on point ``k``
    outside the solid: ``(kind, q, n, d)``, ``kind`` 1 for the edge (beyond its triangle and its wall at the edge,
    inside its slab: ``q`` its closest point), 2 for the wall below it (``k`` beside the wall, its projection held,
    :func:`terrain_wall_holds`: ``q`` the projection, ``n`` the wall's normal) and 0 for none. One claim per point:
    the edge above the wall's edge line, the wall below it. ``n`` points from ``q`` to ``k`` (the bisector of the
    triangle's and the wall's normals for ``k`` on the edge) and ``d = |k - q|``."""
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    eps = window.eps
    n_wall = terrain_wall_normal(side)
    values = terrain_border_edge_values(data, window, r, c, which, side, k)
    if values[1] <= 0.0:
        # below the wall's edge line: the wall, from beside it only
        d = terrain_wall_value(k, v0, side)
        if not terrain_beyond(d, eps) or not terrain_wall_holds(eps, side, v0, v1, k):
            return 0, wp.vec3(0.0), n_wall, 0.0
        return 2, k - d * n_wall, n_wall, d
    u = v1 - v0
    length = wp.length(u)
    claims = (
        terrain_beyond(values[0], eps)
        and terrain_beyond(values[2], eps * length)
        and terrain_beyond(values[3], eps * length)
    )
    if not claims:
        return 0, wp.vec3(0.0), n_wall, 0.0
    q = v0 + wp.clamp(values[2] / wp.dot(u, u), 0.0, 1.0) * u
    w = k - q
    dist = wp.length(w)
    if dist <= _ON_FEATURE:
        t = terrain_border_triangle(r, c, which, side)
        n_top, _a = terrain_cluster_plane(data, window, t[0], t[1], t[2])
        return 1, q, wp.normalize(n_top + n_wall), 0.0
    return 1, q, w / dist, dist


@wp.func
def terrain_corner_point(
    data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, k: wp.vec3
) -> tuple[bool, wp.vec3, wp.vec3, float]:
    """Return whether the corner line below footprint corner ``(r, c)`` claims point ``k``, and the row ``(q, n, d)``:
    ``k`` beyond both walls (the values the walls' pieces at the corner share, :func:`terrain_wall_along`), below the
    corner vertex (the value its claim shares, :func:`terrain_vertex_point`) and above the walls' base; ``q`` the corner
    line's point at ``k``'s height, ``n`` horizontal from it to ``k`` (the bisector of the walls' normals on the line)."""
    surface = window.surface
    sides = terrain_vertex_sides(surface, r, c)
    if sides[1] < 0:
        return False, wp.vec3(0.0), wp.vec3(0.0), 0.0
    v = terrain_window_vertex(data, window, r, c)
    eps = window.eps
    below = v[2] - k[2]
    claims = (
        terrain_beyond(below, eps)
        and below <= _WALL_DEPTH + eps
        and terrain_beyond(terrain_wall_value(k, v, sides[0]), eps)
        and terrain_beyond(terrain_wall_value(k, v, sides[1]), eps)
    )
    if not claims:
        return False, wp.vec3(0.0), wp.vec3(0.0), 0.0
    q = wp.vec3(v[0], v[1], k[2])
    w = k - q
    dist = wp.length(w)
    if dist <= _ON_FEATURE:
        return True, q, wp.normalize(terrain_wall_normal(sides[0]) + terrain_wall_normal(sides[1])), 0.0
    return True, q, w / dist, dist


@wp.func
def terrain_vertex_point(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, k: wp.vec3) -> bool:
    """Return whether vertex ``(r, c)`` claims point ``k``: ``(k - v) . (w - v) <= 0`` for its one-ring's vertices in
    the grid, and at a border vertex ``k`` not below it (the wall below it: ``v_z - k_z <= 0``, the value a corner
    line's claim below a corner of the footprint shares, :func:`terrain_corner_point`)."""
    surface = window.surface
    if not terrain_grid_vertex(surface, r, c):
        return False
    v = terrain_window_vertex(data, window, r, c)
    for j in range(6):
        wi = terrain_ring_vertex(r, c, j)
        if terrain_grid_vertex(surface, wi[0], wi[1]):
            w = terrain_window_vertex(data, window, wi[0], wi[1])
            if not terrain_claims(terrain_vertex_edge_value(k, v, w), window.eps * wp.length(w - v)):
                return False
    if not terrain_interior_vertex(surface, r, c):
        if not terrain_claims(v[2] - k[2], window.eps):
            return False
    return True


@wp.func
def terrain_locate(window: TerrainWindow, x: float, y: float) -> wp.vec3i:
    """Return the triangle ``(row, column, sub)`` under window-frame point ``(x, y)`` (half-open partition).

    Points on the heightfield's far border belong to its last cells; outside the footprint the row is -1.
    """
    surface = window.surface
    u = x / surface.dx
    v = y / surface.dy
    fc = wp.floor(u)
    fr = wp.floor(v)
    c = window.c0 + int(fc)
    r = window.r0 + int(fr)
    fx = u - fc
    fy = v - fr
    if c == surface.ncol - 1 and fx == 0.0:
        c -= 1
        fx = 1.0
    if r == surface.nrow - 1 and fy == 0.0:
        r -= 1
        fy = 1.0
    if not terrain_valid_cell(surface, r, c):
        return wp.vec3i(-1, -1, -1)
    if fx >= fy:
        return wp.vec3i(r, c, 0)
    return wp.vec3i(r, c, 1)


@wp.func
def terrain_point_below(data: TerrainFeatureData, window: TerrainWindow, k: wp.vec3) -> bool:
    """Return whether window-frame point ``k`` lies below the surface (its cluster plane under it)."""
    t = terrain_locate(window, k[0], k[1])
    if t[0] < 0:
        return False
    n, a = terrain_cluster_plane(data, window, t[0], t[1], t[2])
    return k[2] < a[2] - (n[0] * (k[0] - a[0]) + n[1] * (k[1] - a[1])) / n[2]


@wp.func
def terrain_point_below_cached(
    data: TerrainFeatureData, window: TerrainWindow, planes: TerrainPlanes, k: wp.vec3
) -> tuple[TerrainPlanes, bool]:
    """Return :func:`terrain_point_below` through a window's plane cache (and the updated cache)."""
    t = terrain_locate(window, k[0], k[1])
    if t[0] < 0:
        return planes, False
    planes, n, a = terrain_plane(data, window, planes, t[0], t[1], t[2])
    return planes, k[2] < a[2] - (n[0] * (k[0] - a[0]) + n[1] * (k[1] - a[1])) / n[2]


@wp.func
def terrain_point_below_inside(data: TerrainFeatureData, window: TerrainWindow, k: wp.vec3) -> bool:
    """Return :func:`terrain_point_below` for a point of the footprint's closure: a point past its border by the
    predicates' rounding is taken over the border's cells (their planes extended to it), where the half-open partition
    of :func:`terrain_locate` has no triangle."""
    surface = window.surface
    u = k[0] / surface.dx
    v = k[1] / surface.dy
    c = wp.clamp(window.c0 + int(wp.floor(u)), 0, surface.ncol - 2)
    r = wp.clamp(window.r0 + int(wp.floor(v)), 0, surface.nrow - 2)
    fx = wp.clamp(u - float(c - window.c0), 0.0, 1.0)
    fy = wp.clamp(v - float(r - window.r0), 0.0, 1.0)
    sub = int(0)
    if fx < fy:
        sub = 1
    n, a = terrain_cluster_plane(data, window, r, c, sub)
    return k[2] < a[2] - (n[0] * (k[0] - a[0]) + n[1] * (k[1] - a[1])) / n[2]


# ----------------------------------------------------------------------------------------------------------------
# per-pair window
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_box_cells(surface: TerrainSurfaceInfo, x0: float, y0: float, x1: float, y1: float) -> wp.vec4i:
    """Return the cells ``(r0, r1, c0, c1)`` a heightfield-local footprint box ``[x0, x1] x [y0, y1]`` touches.

    Clamped to the grid; the box misses the heightfield when ``r1 < r0`` or ``c1 < c0``. The float32 cell
    coordinates are widened by :data:`TERRAIN_CELL_WIDEN` cells, so that their rounding never drops a cell the box
    touches (a box edge that close to a grid line may take one more row or column).
    """
    cols = float(surface.ncol - 1)
    rows = float(surface.nrow - 1)
    u0 = wp.clamp((x0 - surface.x0) / surface.dx - _CELL_WIDEN, -1.0, cols + 1.0)
    u1 = wp.clamp((x1 - surface.x0) / surface.dx + _CELL_WIDEN, -1.0, cols + 1.0)
    v0 = wp.clamp((y0 - surface.y0) / surface.dy - _CELL_WIDEN, -1.0, rows + 1.0)
    v1 = wp.clamp((y1 - surface.y0) / surface.dy + _CELL_WIDEN, -1.0, rows + 1.0)
    c0 = wp.max(int(wp.floor(u0)), 0)
    c1 = wp.min(int(wp.floor(u1)), surface.ncol - 2)
    r0 = wp.max(int(wp.floor(v0)), 0)
    r1 = wp.min(int(wp.floor(v1)), surface.nrow - 2)
    return wp.vec4i(r0, r1, c0, c1)


@wp.func
def terrain_fine_bound(data: TerrainFeatureData, surface: TerrainSurfaceInfo, fine_r: int, fine_c: int) -> float:
    """Return an upper bound of every height of a fine block [m]: its maximum plus four float32 ulps of slack."""
    value = data.fine_max[surface.fine_offset + fine_r * surface.fine_cols + fine_c]
    return value + 4.0 * _EPS32 * wp.max(wp.abs(value), 1.0e-3)


@wp.func
def terrain_fine_exit(
    data: TerrainFeatureData, surface: TerrainSurfaceInfo, r0: int, r1: int, c0: int, c1: int, low: float
) -> bool:
    """Return whether every height under cells ``[r0, r1] x [c0, c1]`` lies below ``low`` [m, heightfield-local], from
    the fine mip's blocks under the cells (:func:`terrain_fine_bound`)."""
    for fr in range(r0 // _FINE_SIDE, r1 // _FINE_SIDE + 1):
        for fc in range(c0 // _FINE_SIDE, c1 // _FINE_SIDE + 1):
            if terrain_fine_bound(data, surface, fr, fc) >= low:
                return False
    return True


@wp.func
def terrain_tile_exit(
    data: TerrainFeatureData, surface: TerrainSurfaceInfo, r0: int, r1: int, c0: int, c1: int, low: float
) -> bool:
    """Return whether every height under cells ``[r0, r1] x [c0, c1]`` lies below ``low`` [m, heightfield-local].

    Reads the tile-max mip blocks under the cells (:func:`terrain_tile_bound`), and when a tile reaches ``low``, the
    fine mip's 2 x 2-cell blocks under the cells against ``low`` minus :data:`TERRAIN_FINE_MARGIN`
    (:func:`terrain_fine_exit`): a tile spans 8 x 8 cells, so a core above its own window often sits under a tile some
    far cell of which rises past it. With ``low`` a core's lowest point minus its reach, the pair has no row: every
    terrain point under its window is farther than the reach.
    """
    for tr in range(r0 // _TILE_SIDE, r1 // _TILE_SIDE + 1):
        for tc in range(c0 // _TILE_SIDE, c1 // _TILE_SIDE + 1):
            if terrain_tile_bound(data, surface, tr, tc) >= low:
                return terrain_fine_exit(data, surface, r0, r1, c0, c1, low - _FINE_MARGIN)
    return True


@wp.func
def terrain_window_extent(data: TerrainFeatureData, window: TerrainWindow, r1: int, c1: int, extent: float) -> float:
    """Return the largest coordinate magnitude of a window's frame: ``extent`` (the core's), its size, its heights.

    The window spans cells ``[window.r0, r1] x [window.c0, c1]``; the result is what :func:`terrain_eps` takes.
    """
    surface = window.surface
    extent = wp.max(extent, wp.max(float(c1 + 1 - window.c0) * surface.dx, float(r1 + 1 - window.r0) * surface.dy))
    for r in range(window.r0, r1 + 2):
        for c in range(window.c0, c1 + 2):
            extent = wp.max(extent, wp.abs(terrain_height(data, surface, r, c) - window.z_ref))
    return extent


# ----------------------------------------------------------------------------------------------------------------
# tile-max mip and version stamps
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_tile_bound(data: TerrainFeatureData, surface: TerrainSurfaceInfo, tile_r: int, tile_c: int) -> float:
    """Return an upper bound of every height of a tile [m]: its maximum plus four float32 ulps of slack."""
    value = data.tile_max[surface.tile_offset + tile_r * surface.tile_cols + tile_c]
    return value + 4.0 * _EPS32 * wp.max(wp.abs(value), 1.0e-3)


@wp.func
def terrain_features_current(data: TerrainFeatureData, heightfield: int) -> bool:
    """Return whether a heightfield's clusters and mip were built from its current heights, and ``data`` is current.

    A struct fetched before a rebuild that reallocated the plane records (one baked into a captured graph, say)
    still points at the old records while the cluster ids index the new ones; it reads as stale.
    """
    return (
        data.feature_version[heightfield] == data.elevation_version[heightfield]
        and data.layout_version[0] == data.layout
    )


@wp.func
def terrain_report(status: wp.array[wp.int32], overflow: wp.array[wp.int32], flags: int, current: bool):
    """Record a pair's report flags and staleness in the narrow phase's sticky status word and counters.

    ``overflow[0]`` counts pairs whose rows found no room in the spill region, ``overflow[1]`` pairs whose candidates
    outgrew their scratch.
    """
    if (flags & TERRAIN_FINISH_ROW_OVERFLOW) != 0:
        wp.atomic_or(status, 0, TERRAIN_STATUS_ROWS)
        wp.atomic_add(overflow, 0, 1)
    if (flags & TERRAIN_FINISH_CANDIDATE_OVERFLOW) != 0:
        wp.atomic_or(status, 0, TERRAIN_STATUS_ROWS)
        wp.atomic_add(overflow, 1, 1)
    if not current:
        wp.atomic_or(status, 0, TERRAIN_STATUS_STALE)


# ----------------------------------------------------------------------------------------------------------------
# feature ids
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_feature_id(r: int, c: int, kind: int, sub: int, slot: int) -> int:
    """Return the 19-bit window-local feature id of a row (the contact's ``sort_sub_key``).

    Bits 18-13 hold ``r mod 64``, 12-7 ``c mod 64``, 6-5 the kind (face, edge, vertex, recovery), 4-3 the
    sub-feature (triangle 0/1, or edge x/y/diagonal) and 2-0 the point slot.
    """
    return ((((r & 63) * 64 + (c & 63)) * 4 + (kind & 3)) * 4 + (sub & 3)) * 8 + (slot & 7)


@wp.func
def terrain_feature_parts(feature: int) -> wp.vec4i:
    """Return ``(r mod 64, c mod 64, kind, sub-feature)`` of a feature id (the slot is ``feature & 7``)."""
    return wp.vec4i((feature >> 13) & 63, (feature >> 7) & 63, (feature >> 5) & 3, (feature >> 3) & 3)


# ----------------------------------------------------------------------------------------------------------------
# per-pair finish
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _tangent_frame(n: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    reference = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) >= 0.9:
        reference = wp.vec3(0.0, 1.0, 0.0)
    e1 = wp.normalize(reference - wp.dot(reference, n) * n)
    return e1, wp.cross(n, e1)


@wp.func
def _cross2(a: wp.vec2, b: wp.vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def _slot_k(q: wp.array[wp.vec3], n: wp.array[wp.vec3], d: wp.array[float], i: int) -> wp.vec3:
    """A held row's core point ``q + d n`` (the row's definition)."""
    return q[i] + d[i] * n[i]


@wp.func
def _slot_member(alive: int, group: _MaskInts, i: int, g: int) -> bool:
    return ((alive >> i) & 1) != 0 and group[i] == g


@wp.func
def _staged_implied(
    px: _MaskFloats,
    py: _MaskFloats,
    pd: _MaskFloats,
    members: int,
    live: int,
    skip: int,
    eps: float,
) -> bool:
    """Whether the other live members of a plane group imply member ``skip`` at its tangent position: whether the
    lowest convex interpolation of their distances there (members within ``4 eps`` of the position coincide with it; a
    position on a segment or in a triangle of members interpolates their distances) reaches its distance within
    :data:`TERRAIN_IMPLIED_RESIDUAL`. The group's ``members`` are staged in member order (row order): tangent positions
    ``(px, py)`` of their core points and distances ``pd``, live where ``live`` has their bit.

    The tests are the points and segments of the other live members, each segment with its triangles (a third member
    after it). The test stops at the first interpolation that reaches the distance: the lowest one does exactly when
    one does (a rounded subtraction keeps order)."""
    point = wp.vec2(px[skip], py[skip])
    limit = pd[skip]
    tolerance = _MERGE_SCALE * eps
    for a in range(members):
        if a == skip or ((live >> a) & 1) == 0:
            continue
        pa = wp.vec2(px[a], py[a])
        da = pd[a]
        # test (a, a): the point alone; tests (a, b): the segment and its triangles
        for b in range(a, members):
            if b != a and (b == skip or ((live >> b) & 1) == 0):
                continue
            if b == a:
                if wp.length(point - pa) <= tolerance:
                    if limit >= da - _IMPLIED_RESIDUAL:
                        return True
                continue
            pb = wp.vec2(px[b], py[b])
            db = pd[b]
            ab = pb - pa
            length2 = wp.dot(ab, ab)
            if length2 > 0.0:
                s = wp.dot(point - pa, ab) / length2
                perpendicular = wp.abs(_cross2(ab, point - pa)) / wp.sqrt(length2)
                if s >= -_BARYCENTRIC_TOLERANCE and s <= 1.0 + _BARYCENTRIC_TOLERANCE and perpendicular <= tolerance:
                    if limit >= (da + s * (db - da)) - _IMPLIED_RESIDUAL:
                        return True
            for c in range(b + 1, members):
                if c == skip or ((live >> c) & 1) == 0:
                    continue
                pc = wp.vec2(px[c], py[c])
                det = _cross2(pb - pa, pc - pa)
                scale = wp.length(pb - pa) * wp.length(pc - pa)
                if wp.abs(det) > _BARYCENTRIC_TOLERANCE * scale and scale > 0.0:
                    beta = _cross2(point - pa, pc - pa) / det
                    gamma = _cross2(pb - pa, point - pa) / det
                    if (
                        beta >= -_BARYCENTRIC_TOLERANCE
                        and gamma >= -_BARYCENTRIC_TOLERANCE
                        and beta + gamma <= 1.0 + _BARYCENTRIC_TOLERANCE
                    ):
                        value = da + beta * (db - da) + gamma * (pd[c] - da)
                        if limit >= value - _IMPLIED_RESIDUAL:
                            return True
    return False


@wp.func
def terrain_finish_slots(
    q: wp.array[wp.vec3],
    n: wp.array[wp.vec3],
    d: wp.array[float],
    base: int,
    count: int,
    eps: float,
    distinct: bool,
) -> tuple[int, int]:
    """The law's finish of a walking pair's held rows in place: ``count`` rows (at most
    :data:`TERRAIN_FINISH_MASK_ROWS`) at ``[base, base + count)`` of row arrays, in feature order.

    A row's core point is ``q + d n``. The steps: coincident rows merge (the first in feature order stays); the first
    row of a plane represents its plane group (:func:`terrain_rows_same_plane`); the rows the others of their group
    imply are dropped, the shallowest first, one at a time (the lower convex hull of tangent position and distance,
    residual :data:`TERRAIN_IMPLIED_RESIDUAL`); a group keeps at most four rows: its deepest, the farthest from it, the
    one spanning the largest triangle with them and the one adding the most area. Ties are broken by row order, never
    by memory order. :func:`terrain_finish_list` is the same finish on any number of candidates, bit for bit.

    ``distinct`` rows have no coincident pair to merge (rows held with :func:`terrain_hold_row` at ``eps``: it drops a
    row coinciding with a held one by the same test), so the first step is skipped.

    Returns ``(keep, kept)``: bit ``i`` of ``keep`` marks row ``i`` kept (kept rows in row order), and their count.
    """
    count = wp.min(count, _MASK_ROWS)
    alive = int(0)
    group = _MaskInts()
    rank = _MaskInts()
    # 1. coincident rows: the first in feature order stays
    for i in range(count):
        survives = bool(True)
        if not distinct:
            qi = q[base + i]
            ni = n[base + i]
            for j in range(i):
                if ((alive >> j) & 1) != 0 and terrain_rows_coincide(q[base + j], n[base + j], qi, ni, eps):
                    survives = False
                    break
        if survives:
            alive = alive | (1 << i)
    # 2. plane groups: the first row of a plane represents it
    for i in range(count):
        group[i] = -1
        if ((alive >> i) & 1) == 0:
            continue
        qi = q[base + i]
        ni = n[base + i]
        group[i] = i
        for j in range(i):
            if ((alive >> j) & 1) != 0 and group[j] == j:
                if terrain_rows_same_plane(qi, ni, q[base + j], n[base + j], eps):
                    group[i] = j
                    break
    # 3. implied rows, the shallowest first, one at a time. One pass in that order drops what testing again from the
    # shallowest after every drop would: a drop removes interpolation candidates, so it only raises the lower hull of
    # a group's other rows, and no row found not implied before a drop is implied after it. A group's members are
    # staged once (tangent position of the core point, distance), in row order.
    px = _MaskFloats()
    py = _MaskFloats()
    pd = _MaskFloats()
    for g in range(count):
        if ((alive >> g) & 1) == 0 or group[g] != g:
            continue
        e1, e2 = _tangent_frame(n[base + g])
        size = int(0)
        for i in range(count):
            if _slot_member(alive, group, i, g):
                k = _slot_k(q, n, d, base + i)
                px[size] = wp.dot(k, e1)
                py[size] = wp.dot(k, e2)
                pd[size] = d[base + i]
                rank[size] = i
                size += 1
        members = size
        live = int(0)
        for s in range(members):
            live = live | (1 << s)
        tested = int(0)
        while size >= 3:
            pick = int(-1)
            for s in range(members):
                if ((live >> s) & 1) != 0 and ((tested >> s) & 1) == 0:
                    if pick < 0 or pd[s] > pd[pick]:
                        pick = s
            if pick < 0:
                break
            tested = tested | (1 << pick)
            implied = _staged_implied(px, py, pd, members, live, pick, eps)
            if implied:
                live = live & ~(1 << pick)
                alive = alive & ~(1 << rank[pick])
                size -= 1
    # 4. at most four rows per group: deepest, farthest, largest triangle, largest quadrilateral. A group of at most
    # four rows keeps them all (its picks would rank every one), so that only larger groups pick.
    sizes = _MaskInts()
    for i in range(count):
        rank[i] = 4
        if ((alive >> i) & 1) != 0:
            sizes[group[i]] = sizes[group[i]] + 1
    for g in range(count):
        # A group keeps its first row's id even when that row was implied.
        if group[g] != g:
            continue
        if sizes[g] <= _GROUP_ROWS:
            continue
        ng = n[base + g]
        i0 = int(-1)
        for i in range(count):
            if _slot_member(alive, group, i, g):
                if i0 < 0 or d[base + i] < d[base + i0]:
                    i0 = i
        p0 = _slot_k(q, n, d, base + i0)
        p0 = p0 - wp.dot(p0, ng) * ng
        rank[i0] = 0
        i1 = int(-1)
        far = float(-1.0)
        for j in range(count):
            if not _slot_member(alive, group, j, g) or rank[j] == 0:
                continue
            pj = _slot_k(q, n, d, base + j)
            distance = wp.length(pj - wp.dot(pj, ng) * ng - p0)
            if distance > far:
                far = distance
                i1 = j
        if i1 < 0:
            continue
        rank[i1] = 1
        p1 = _slot_k(q, n, d, base + i1)
        p1 = p1 - wp.dot(p1, ng) * ng
        i2 = int(-1)
        largest = float(-1.0)
        for j in range(count):
            if not _slot_member(alive, group, j, g) or rank[j] <= 1:
                continue
            pj = _slot_k(q, n, d, base + j)
            area = wp.abs(wp.dot(wp.cross(p1 - p0, pj - wp.dot(pj, ng) * ng - p0), ng))
            if area > largest:
                largest = area
                i2 = j
        if i2 < 0:
            continue
        rank[i2] = 2
        p2 = _slot_k(q, n, d, base + i2)
        p2 = p2 - wp.dot(p2, ng) * ng
        i3 = int(-1)
        gain_best = float(1.0e-15)
        for j in range(count):
            if not _slot_member(alive, group, j, g) or rank[j] <= 2:
                continue
            pj = _slot_k(q, n, d, base + j)
            pj = pj - wp.dot(pj, ng) * ng
            gain = float(0.0)
            for side in range(3):
                pa = p0
                pb = p1
                pc = p2
                if side == 1:
                    pa = p1
                    pb = p2
                    pc = p0
                elif side == 2:
                    pa = p2
                    pb = p0
                    pc = p1
                side_c = wp.dot(wp.cross(pb - pa, pc - pa), ng)
                side_j = wp.dot(wp.cross(pb - pa, pj - pa), ng)
                if side_c * side_j < 0.0:
                    gain = wp.max(gain, wp.abs(side_j))
            if gain > gain_best:
                gain_best = gain
                i3 = j
        if i3 >= 0:
            rank[i3] = 3
    keep = int(0)
    kept = int(0)
    for i in range(count):
        if ((alive >> i) & 1) != 0:
            if rank[i] < _GROUP_ROWS or sizes[group[i]] <= _GROUP_ROWS:
                keep = keep | (1 << i)
                kept += 1
    return keep, kept


@wp.func
def _list_member(scratch: TerrainFinishScratch, i: int, g: int) -> bool:
    return scratch.alive[i] != 0 and scratch.group[i] == g


@wp.func
def _list_implied(
    scratch: TerrainFinishScratch,
    base: int,
    count: int,
    g: int,
    skip: int,
    e1: wp.vec3,
    e2: wp.vec3,
    point: wp.vec2,
    eps: float,
    lane: int,
    lanes: int,
) -> bool:
    """:func:`_staged_implied` over a scratch's candidates (their core points read, not rebuilt), from this lane's share
    of the tests: the points and segments of the other live members of plane group ``g``, each segment with its
    triangles; lane ``lane`` of ``lanes`` takes every ``lanes``-th of them in that order, so that the lanes' answers
    ORed are the answer of one lane taking all. A group's members follow its first candidate ``g``."""
    limit = scratch.d[base + skip]
    tolerance = _MERGE_SCALE * eps
    test = int(0)
    for a in range(g, count):
        if a == skip or not _list_member(scratch, base + a, g):
            continue
        ka = scratch.k[base + a]
        pa = wp.vec2(wp.dot(ka, e1), wp.dot(ka, e2))
        da = scratch.d[base + a]
        mine = test % lanes == lane
        test += 1
        if mine:
            if wp.length(point - pa) <= tolerance:
                if limit >= da - _IMPLIED_RESIDUAL:
                    return True
        for b in range(a + 1, count):
            if b == skip or not _list_member(scratch, base + b, g):
                continue
            mine = test % lanes == lane
            test += 1
            if not mine:
                continue
            kb = scratch.k[base + b]
            pb = wp.vec2(wp.dot(kb, e1), wp.dot(kb, e2))
            db = scratch.d[base + b]
            ab = pb - pa
            length2 = wp.dot(ab, ab)
            if length2 > 0.0:
                s = wp.dot(point - pa, ab) / length2
                perpendicular = wp.abs(_cross2(ab, point - pa)) / wp.sqrt(length2)
                if s >= -_BARYCENTRIC_TOLERANCE and s <= 1.0 + _BARYCENTRIC_TOLERANCE and perpendicular <= tolerance:
                    if limit >= (da + s * (db - da)) - _IMPLIED_RESIDUAL:
                        return True
            for c in range(b + 1, count):
                if c == skip or not _list_member(scratch, base + c, g):
                    continue
                kc = scratch.k[base + c]
                pc = wp.vec2(wp.dot(kc, e1), wp.dot(kc, e2))
                det = _cross2(pb - pa, pc - pa)
                scale = wp.length(pb - pa) * wp.length(pc - pa)
                if wp.abs(det) > _BARYCENTRIC_TOLERANCE * scale and scale > 0.0:
                    beta = _cross2(point - pa, pc - pa) / det
                    gamma = _cross2(pb - pa, point - pa) / det
                    if (
                        beta >= -_BARYCENTRIC_TOLERANCE
                        and gamma >= -_BARYCENTRIC_TOLERANCE
                        and beta + gamma <= 1.0 + _BARYCENTRIC_TOLERANCE
                    ):
                        value = da + beta * (db - da) + gamma * (scratch.d[base + c] - da)
                        if limit >= value - _IMPLIED_RESIDUAL:
                            return True
    return False


@wp.func
def _lanes_max(value: int) -> int:
    """The largest ``value`` of a block's lanes: a reduction across the block, which also orders every lane's memory
    accesses before it before every lane's after it (a block barrier)."""
    return wp.tile_extract(wp.tile_max(wp.tile(value)), 0)


@wp.func
def _lanes_max(value: float) -> float:
    """The largest ``value`` of a block's lanes (a block barrier)."""
    return wp.tile_extract(wp.tile_max(wp.tile(value)), 0)


@wp.func
def _lanes_min(value: int) -> int:
    """The least ``value`` of a block's lanes (a block barrier)."""
    return wp.tile_extract(wp.tile_min(wp.tile(value)), 0)


@wp.func
def _lanes_sum(value: int) -> int:
    """The sum of a block's lanes' ``value`` (a block barrier)."""
    return wp.tile_extract(wp.tile_sum(wp.tile(value)), 0)


@wp.func
def _lanes_first(value: int) -> int:
    """The first lane's ``value``, for every lane of a block (a block barrier)."""
    return wp.tile_extract(wp.tile(value), 0)


@wp.func
def _list_merge(scratch: TerrainFinishScratch, base: int, count: int, eps: float, lane: int, lanes: int):
    """Step 1 of :func:`terrain_finish_list`: coincident candidates merged, the first in candidate order staying
    (``alive`` 1, the others 0); the group sizes of step 2 zeroed (``rank``).

    Each lane takes every ``lanes``-th candidate: one that no earlier candidate coincides with stays, and one whose
    first coincident candidate stays merges into it; any other (a chain of coincident candidates, rows within a few
    bands of each other) is decided by the first lane in candidate order, as one lane decides them all."""
    for i in range(lane, count, lanes):
        qi = scratch.q[base + i]
        ni = scratch.n[base + i]
        first = int(-1)
        for j in range(i):
            # every test, so that a lane's loads overlap
            coincide = terrain_rows_coincide(scratch.q[base + j], scratch.n[base + j], qi, ni, eps)
            if coincide and first < 0:
                first = j
        alive = int(1)
        if first >= 0:
            # undecided: its first coincident candidate in ``mark``
            alive = -1
            scratch.mark[base + i] = first
        scratch.alive[base + i] = alive
        scratch.rank[base + i] = 0
    if lanes > 1:
        _lanes_max(0)
    undecided = int(0)
    for i in range(lane, count, lanes):
        if scratch.alive[base + i] < 0:
            if scratch.alive[base + scratch.mark[base + i]] == 1:
                scratch.alive[base + i] = 0
            else:
                undecided = 1
    if lanes > 1:
        undecided = _lanes_max(undecided)
    if undecided == 0:
        return
    if lane == 0:
        for i in range(count):
            if scratch.alive[base + i] >= 0:
                continue
            qi = scratch.q[base + i]
            ni = scratch.n[base + i]
            survives = int(1)
            for j in range(i):
                if scratch.alive[base + j] == 1 and terrain_rows_coincide(
                    scratch.q[base + j], scratch.n[base + j], qi, ni, eps
                ):
                    survives = 0
                    break
            scratch.alive[base + i] = survives
    if lanes > 1:
        _lanes_max(0)


@wp.func
def _list_groups(scratch: TerrainFinishScratch, base: int, count: int, eps: float, lane: int, lanes: int):
    """Step 2 of :func:`terrain_finish_list`: the first live candidate of a plane represents its plane group
    (``group``), then each group's size at its first candidate (``rank``) and the implied-row step's marks cleared
    (``mark``).

    As step 1 (:func:`_list_merge`): a live candidate that shares a plane with no earlier live one represents its
    group, and one whose first such candidate represents a group joins it; any other is decided by the first lane in
    candidate order, as one lane decides them all."""
    for i in range(lane, count, lanes):
        group = int(-1)
        if scratch.alive[base + i] != 0:
            qi = scratch.q[base + i]
            ni = scratch.n[base + i]
            first = int(-1)
            for j in range(i):
                same = terrain_rows_same_plane(qi, ni, scratch.q[base + j], scratch.n[base + j], eps)
                if same and first < 0 and scratch.alive[base + j] != 0:
                    first = j
            group = i
            if first >= 0:
                # undecided: the first live candidate on its plane in ``mark``
                group = -2
                scratch.mark[base + i] = first
        scratch.group[base + i] = group
    if lanes > 1:
        _lanes_max(0)
    undecided = int(0)
    for i in range(lane, count, lanes):
        if scratch.group[base + i] == -2:
            first = scratch.mark[base + i]
            if scratch.group[base + first] == first:
                scratch.group[base + i] = first
            else:
                undecided = 1
    if lanes > 1:
        undecided = _lanes_max(undecided)
    if undecided != 0:
        if lane == 0:
            for i in range(count):
                if scratch.group[base + i] != -2:
                    continue
                qi = scratch.q[base + i]
                ni = scratch.n[base + i]
                group = i
                for j in range(i):
                    if scratch.alive[base + j] != 0 and scratch.group[base + j] == j:
                        if terrain_rows_same_plane(qi, ni, scratch.q[base + j], scratch.n[base + j], eps):
                            group = j
                            break
                scratch.group[base + i] = group
        if lanes > 1:
            _lanes_max(0)
    for i in range(lane, count, lanes):
        scratch.mark[base + i] = 0
        if scratch.alive[base + i] != 0:
            wp.atomic_add(scratch.rank, base + scratch.group[base + i], 1)
    if lanes > 1:
        _lanes_max(0)


@wp.func
def _list_next_group(scratch: TerrainFinishScratch, base: int, count: int, after: int, lane: int, lanes: int) -> int:
    """The first plane group past candidate ``after`` with three live candidates or more (its size at its first
    candidate, ``rank``), or -1: the groups whose rows the implied-row step tests, in candidate order."""
    found = int(_NO_ROW)
    for i in range(lane, count, lanes):
        if i > after and scratch.group[base + i] == i and scratch.rank[base + i] >= 3:
            found = i
            break
    if lanes > 1:
        found = _lanes_min(found)
    if found == _NO_ROW:
        return -1
    return found


@wp.func
def _list_pick(scratch: TerrainFinishScratch, base: int, count: int, g: int, lane: int, lanes: int) -> int:
    """The implied-row step's next candidate of plane group ``g``: its shallowest live candidate not yet tested
    (``mark`` 0), the first in candidate order on ties, or -1. Each lane reads its own candidates' marks."""
    pick = int(-1)
    for i in range(lane, count, lanes):
        if _list_member(scratch, base + i, g) and scratch.mark[base + i] == 0:
            if pick < 0 or scratch.d[base + i] > scratch.d[base + pick]:
                pick = i
    if lanes > 1:
        mine = float(-wp.inf)
        if pick >= 0:
            mine = scratch.d[base + pick]
        shallowest = _lanes_max(mine)
        first = int(_NO_ROW)
        if pick >= 0 and mine == shallowest:
            first = pick
        pick = _lanes_min(first)
        if pick == _NO_ROW:
            pick = -1
    return pick


@wp.func
def _list_keep_group(scratch: TerrainFinishScratch, base: int, count: int, g: int):
    """Step 4 of :func:`terrain_finish_list` for plane group ``g`` (its size at its first candidate, ``rank``): its live
    candidates ranked (``rank``: 0 deepest, 1 farthest, 2 largest triangle, 3 largest quadrilateral, 4 not kept; a
    group of at most four keeps every one, ranking 3 the one it does not pick)."""
    size = scratch.rank[base + g]
    if size <= 1:
        # its first candidate alone: no step drops a row of a group of fewer than three
        scratch.rank[base + g] = 0
        return
    rest = int(_GROUP_ROWS)
    if size <= _GROUP_ROWS:
        rest = _GROUP_ROWS - 1
    ng = scratch.n[base + g]
    i0 = int(-1)
    for i in range(count):
        if _list_member(scratch, base + i, g):
            scratch.rank[base + i] = rest
            if i0 < 0 or scratch.d[base + i] < scratch.d[base + i0]:
                i0 = i
    scratch.rank[base + i0] = 0
    p0 = scratch.k[base + i0]
    p0 = p0 - wp.dot(p0, ng) * ng
    i1 = int(-1)
    far = float(-1.0)
    for j in range(count):
        if j == i0 or not _list_member(scratch, base + j, g):
            continue
        pj = scratch.k[base + j]
        distance = wp.length(pj - wp.dot(pj, ng) * ng - p0)
        if distance > far:
            far = distance
            i1 = j
    if i1 < 0:
        return
    scratch.rank[base + i1] = 1
    p1 = scratch.k[base + i1]
    p1 = p1 - wp.dot(p1, ng) * ng
    i2 = int(-1)
    largest = float(-1.0)
    for j in range(count):
        if j == i0 or j == i1 or not _list_member(scratch, base + j, g):
            continue
        pj = scratch.k[base + j]
        area = wp.abs(wp.dot(wp.cross(p1 - p0, pj - wp.dot(pj, ng) * ng - p0), ng))
        if area > largest:
            largest = area
            i2 = j
    if i2 < 0:
        return
    scratch.rank[base + i2] = 2
    p2 = scratch.k[base + i2]
    p2 = p2 - wp.dot(p2, ng) * ng
    i3 = int(-1)
    gain_best = float(1.0e-15)
    for j in range(count):
        if j == i0 or j == i1 or j == i2 or not _list_member(scratch, base + j, g):
            continue
        pj = scratch.k[base + j]
        pj = pj - wp.dot(pj, ng) * ng
        gain = float(0.0)
        for side in range(3):
            pa = p0
            pb = p1
            pc = p2
            if side == 1:
                pa = p1
                pb = p2
                pc = p0
            elif side == 2:
                pa = p2
                pb = p0
                pc = p1
            side_c = wp.dot(wp.cross(pb - pa, pc - pa), ng)
            side_j = wp.dot(wp.cross(pb - pa, pj - pa), ng)
            if side_c * side_j < 0.0:
                gain = wp.max(gain, wp.abs(side_j))
        if gain > gain_best:
            gain_best = gain
            i3 = j
    if i3 >= 0:
        scratch.rank[base + i3] = 3


@wp.func
def terrain_finish_list(scratch: TerrainFinishScratch, base: int, count: int, eps: float, lane: int, lanes: int) -> int:
    """The law's finish (:func:`terrain_finish_slots`) of ``count`` candidates at ``[base, base + count)`` of a
    scratch, any number of them, coincident ones included, with their core points from ``scratch.k``, by a block's
    ``lanes`` lanes (lane ``lane``; one thread passes ``0, 1``).

    The finish's state lives in the scratch (``alive``, ``group``, ``rank``, ``mark``), not on the stack, so that it
    holds as many candidates as the scratch does: on the rows :func:`terrain_finish_slots` takes, with core points
    ``q + d n``, it keeps the same rows, bit for bit (the same steps, ties and float32 operations). The lanes split
    every step: each takes every ``lanes``-th candidate of the merge and the plane groups (:func:`_list_merge`,
    :func:`_list_groups`: a candidate's earlier ones, the order's few chains decided by the first lane), the
    implied-row tests of a row (:func:`_list_implied`, the next row picked across the block, :func:`_list_pick`) and
    the groups whose rows it ranks (:func:`_list_keep_group`); their answers combine across the block (block
    reductions, which also order the lanes' writes before the reads that follow), so that any number of lanes keeps the
    rows one lane keeps. With ``lanes > 1`` every lane of the block must call it with the same candidates.

    Returns the kept candidates' count (every lane); ``scratch.mark`` is 1 at the kept ones (kept rows in candidate
    order).
    """
    _list_merge(scratch, base, count, eps, lane, lanes)
    _list_groups(scratch, base, count, eps, lane, lanes)
    # 3. implied rows (see terrain_finish_slots), the shallowest first, one at a time: the groups of three rows or more,
    # each with every lane (a group's tests touch its own rows alone, so that their order is free)
    g = _list_next_group(scratch, base, count, -1, lane, lanes)
    while g >= 0:
        e1, e2 = _tangent_frame(scratch.n[base + g])
        size = scratch.rank[base + g]
        while size >= 3:
            pick = _list_pick(scratch, base, count, g, lane, lanes)
            if pick < 0:
                break
            kj = scratch.k[base + pick]
            point = wp.vec2(wp.dot(kj, e1), wp.dot(kj, e2))
            implied = _list_implied(scratch, base, count, g, pick, e1, e2, point, eps, lane, lanes)
            if lanes > 1:
                implied = _lanes_max(int(implied)) != 0
            if pick % lanes == lane:
                # the pick's lane, which alone reads its mark; the next pick's reductions order its writes
                scratch.mark[base + pick] = 1
                if implied:
                    scratch.alive[base + pick] = 0
            if implied:
                size -= 1
        if lane == 0:
            scratch.rank[base + g] = size
        g = _list_next_group(scratch, base, count, g, lane, lanes)
    # 4. at most four rows per plane group: each lane ranks the groups of its candidates that represent one
    for g in range(lane, count, lanes):
        if scratch.group[base + g] == g:
            _list_keep_group(scratch, base, count, g)
    if lanes > 1:
        _lanes_max(0)
    kept = int(0)
    for i in range(lane, count, lanes):
        keep = int(0)
        if scratch.alive[base + i] == 0:
            scratch.rank[base + i] = _GROUP_ROWS
        elif scratch.rank[base + i] < _GROUP_ROWS:
            keep = 1
        scratch.mark[base + i] = keep
        kept += keep
    if lanes > 1:
        kept = _lanes_sum(kept)
    return kept


# ----------------------------------------------------------------------------------------------------------------
# per-pair output: the protocol every per-type kernel follows (see terrain_shadow.TERRAIN_LAW_KERNELS)
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def terrain_pair_exit(out: TerrainLawRows, pair: int, state: int):
    """Record a pair that has no row, with its ``TERRAIN_PAIR_*`` bits, raising ``terrain_stale`` for stale features.

    Every per-type kernel's early exit (:data:`TERRAIN_PAIR_STALE`, :data:`TERRAIN_PAIR_FILTERED`,
    :data:`TERRAIN_PAIR_OUTSIDE`, :data:`TERRAIN_PAIR_TILE_EXIT`), and rows the spill region has no room for
    (:func:`terrain_spill_rows`).
    """
    out.first[pair] = -1
    out.count[pair] = 0
    out.state[pair] = state
    if state == TERRAIN_PAIR_STALE:
        terrain_report(out.status, out.overflow, 0, False)


@wp.func
def terrain_spill_rows(out: TerrainLawRows, rows: int) -> int:
    """Reserve ``rows`` consecutive rows of the spill region for a pair whose rows exceed its walk slots.

    Returns the first reserved row, or -1 when the spill region has no room left for them: then nothing is reserved
    (a compare-and-swap takes the rows only when they fit, so that a later, smaller pair still can), and the caller
    keeps the pair's first rows in its walk slots (:func:`terrain_finish_overflow`). The rows asked for are counted
    either way (``out.spilled``: the region a launch needs). Rows are taken in the order the pairs ask, so their
    place, and which pairs find room when the region is too small, depend on the launch; a pair's rows do not.
    """
    wp.atomic_add(out.spilled, 0, rows)
    room = out.capacity - out.slots
    taken = out.reserved[0]
    while taken + rows <= room:
        seen = wp.atomic_cas(out.reserved, 0, taken, taken + rows)
        if seen == taken:
            return out.slots + taken
        taken = seen
    return -1


@wp.func
def terrain_finish_overflow(scratch: TerrainFinishScratch, base: int, count: int, limit: int) -> int:
    """Keep at most ``limit`` of a scratch finish's kept candidates (``scratch.mark`` 1 at the kept ones, on return at
    the ones still kept), when the spill region has no room for them all: each plane group's deepest row first
    (rank 0), then each group's farthest, largest-triangle and largest-quadrilateral rows (ranks 1 to 3), the deepest
    first within a rank, ties to the earlier candidate. Returns the count kept."""
    taken = int(0)
    for level in range(_GROUP_ROWS):
        while taken < limit:
            pick = int(-1)
            for i in range(count):
                if scratch.mark[base + i] == 1 and scratch.rank[base + i] == level:
                    if pick < 0 or scratch.d[base + i] < scratch.d[base + pick]:
                        pick = i
            if pick < 0:
                break
            scratch.mark[base + pick] = 2
            taken += 1
    for i in range(count):
        keep = int(0)
        if scratch.mark[base + i] == 2:
            keep = 1
        scratch.mark[base + i] = keep
    return taken


@wp.func
def terrain_place_candidate(
    scratch: TerrainFinishScratch,
    base: int,
    held: int,
    q: wp.vec3,
    k: wp.vec3,
    n: wp.vec3,
    d: float,
    feature: int,
) -> int:
    """Place a candidate (window frame) with its core point ``k`` after the ``held`` ones a scratch block holds from
    ``base``, unmerged (the finish merges coincident candidates, :func:`terrain_finish_list`): past the block's
    capacity it is only counted. Returns the new number of candidates."""
    if held < scratch.capacity:
        scratch.q[base + held] = q
        scratch.k[base + held] = k
        scratch.n[base + held] = n
        scratch.d[base + held] = d
        scratch.feature[base + held] = feature
    return held + 1


@wp.func
def terrain_hold_row(
    out: TerrainLawRows,
    slot: int,
    held: int,
    q: wp.vec3,
    n: wp.vec3,
    d: float,
    feature: int,
    eps: float,
    limit: int,
) -> int:
    """Add a row (window frame) to the distinct rows a pair holds in its slots from ``slot``, as the finish merges.

    A row that coincides with a held one (:func:`terrain_rows_coincide`) is dropped, the first in walk order staying;
    past ``limit`` held rows a distinct row is only counted. Returns the new number of distinct rows, which a kernel
    compares with its limit to tell whether the finish could keep the rows as they are.
    """
    for h in range(wp.min(held, limit)):
        if terrain_rows_coincide(out.q[slot + h], out.n[slot + h], q, n, eps):
            return held
    if held < limit:
        out.q[slot + held] = q
        out.n[slot + held] = n
        out.d[slot + held] = d
        out.feature[slot + held] = feature
    return held + 1


@wp.func
def terrain_plane_group_of_three(out: TerrainLawRows, slot: int, held: int, eps: float) -> bool:
    """Whether three of a pair's held rows share a plane group, as the finish groups them (its first row of a plane
    represents it, :func:`terrain_rows_same_plane`): only such a group can lose rows to the finish's implied-row and
    per-group steps. Up to 32 held rows (a group's members counted in 4 bits, two 64-bit words)."""
    representatives = int(0)
    low = wp.int64(0)
    high = wp.int64(0)
    for i in range(held):
        qi = out.q[slot + i]
        ni = out.n[slot + i]
        group = i
        for j in range(i):
            if (representatives >> j) & 1 != 0:
                if terrain_rows_same_plane(qi, ni, out.q[slot + j], out.n[slot + j], eps):
                    group = j
                    break
        if group == i:
            representatives = representatives | (1 << i)
        # 4-bit member counts: groups 0-15 in the low word, 16-31 in the high one
        if group < 16:
            shift = wp.int64(4 * group)
            if ((low >> shift) & wp.int64(15)) >= wp.int64(2):
                return True
            low = low + (wp.int64(1) << shift)
        else:
            shift = wp.int64(4 * (group - 16))
            if ((high >> shift) & wp.int64(15)) >= wp.int64(2):
                return True
            high = high + (wp.int64(1) << shift)
    return False


@wp.func
def terrain_defer_pair(out: TerrainLawRows, partner_class: int, pair: int, state: int):
    """Leave a pair to its partner class's finish of deferred pairs: no row yet, :data:`TERRAIN_PAIR_DEFERRED` added to
    ``state``, and the pair appended to the class's deferred list (in any order; a pair's rows depend on it alone)."""
    out.count[pair] = 0
    out.state[pair] = state | TERRAIN_PAIR_DEFERRED
    out.deferred[out.deferred_first[partner_class] + wp.atomic_add(out.deferred_count, partner_class, 1)] = pair


@wp.func
def terrain_finish_deferred(
    scratch: TerrainFinishScratch,
    base: int,
    held: int,
    eps: float,
    out: TerrainLawRows,
    pair: int,
    slot: int,
    walk_slots: int,
    state: int,
    window: TerrainWindow,
    frame: wp.transform,
    lane: int,
    lanes: int,
):
    """Finish a deferred pair's ``held`` candidates, which the first lane placed in a block's scratch from ``base``
    (window frame, their core points in ``scratch.k``; the other lanes' ``held`` is not read), with the block's
    ``lanes`` lanes (lane ``lane``; one thread passes ``0, 1``; :func:`terrain_finish_list`), and write every row it
    keeps in the world frame, each lane its own candidates' in candidate order: in the pair's ``walk_slots`` walk slots
    from ``slot`` when they fit, else in rows of the spill region the first lane reserves (:func:`terrain_spill_rows`),
    or, when that has no room left, the first of them by priority in its walk slots (flagged
    :data:`TERRAIN_PAIR_ROW_OVERFLOW`); its state ``state`` with the report's bits. The scratch holds every candidate a
    window of the class can give; a pair with more has a window its shape data no longer bounds (changed since
    construction): it keeps the first ``scratch.capacity`` in window order and is flagged. Every lane of the block must
    call it with the same pair."""
    if lanes > 1:
        # every lane takes the first lane's count, after its writes
        held = _lanes_first(held)
    flags = int(0)
    count = held
    if held > scratch.capacity:
        flags = TERRAIN_FINISH_CANDIDATE_OVERFLOW
        count = scratch.capacity
    kept = terrain_finish_list(scratch, base, count, eps, lane, lanes)
    if lane == 0:
        if kept > walk_slots:
            first = terrain_spill_rows(out, kept)
            if first >= 0:
                slot = first
            else:
                # no room left: the finish's first rows by priority, in the pair's own walk slots
                kept = terrain_finish_overflow(scratch, base, count, walk_slots)
                flags = flags | TERRAIN_FINISH_ROW_OVERFLOW
        out.first[pair] = slot
        out.count[pair] = kept
        out.state[pair] = state | terrain_finish_state(flags)
        terrain_report(out.status, out.overflow, flags, True)
    if lanes > 1:
        # every lane takes the first lane's rows, after its marks
        slot = _lanes_first(slot)
    # a kept row's place: the kept candidates before it, the lanes' counted across the block
    written = int(0)
    for start in range(0, count, lanes):
        i = start + lane
        keep = int(0)
        if i < count:
            keep = scratch.mark[base + i]
        before = int(0)
        if lanes > 1:
            before = wp.untile(wp.tile_scan_exclusive(wp.tile(keep)))
        if keep != 0:
            terrain_store_row(
                out,
                slot + written + before,
                scratch.q[base + i],
                scratch.n[base + i],
                scratch.d[base + i],
                scratch.feature[base + i],
                window,
                frame,
            )
        if lanes > 1:
            written += _lanes_sum(keep)
        else:
            written += keep


@wp.func
def terrain_store_row(
    out: TerrainLawRows,
    slot: int,
    q: wp.vec3,
    n: wp.vec3,
    d: float,
    feature: int,
    window: TerrainWindow,
    frame: wp.transform,
):
    """Write one kept row (window frame) to a pool row, in the world frame.

    ``frame`` is the heightfield's world transform; the terrain point leaves the window frame rounded once.
    """
    out.q[slot] = wp.transform_point(frame, terrain_window_to_local(window, q))
    out.n[slot] = wp.transform_vector(frame, n)
    out.d[slot] = d
    out.feature[slot] = feature


@wp.func
def terrain_finish_state(flags: int) -> int:
    """Return the ``TERRAIN_PAIR_*`` bits of a pair's report flags."""
    state = int(0)
    if (flags & TERRAIN_FINISH_ROW_OVERFLOW) != 0:
        state = state | TERRAIN_PAIR_ROW_OVERFLOW
    if (flags & TERRAIN_FINISH_CANDIDATE_OVERFLOW) != 0:
        state = state | TERRAIN_PAIR_CANDIDATE_OVERFLOW
    return state

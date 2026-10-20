# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared float32 primitives of the per-pair terrain contact law.

The law emits the support set of one (heightfield, convex shape) pair directly: one row per local minimum of the
core-to-surface distance within the writer's reach, plus the corners of flat support regions, each owned by one
surface feature (a plane cluster, a crease edge or a vertex). This module holds what every per-type kernel shares;
the kernels themselves and the static data they read live in :mod:`terrain_contacts`. Nothing routes contacts
through the law yet.

Surface model:

* Window-local frame. A pair works in the frame of one window vertex ``(r0, c0)``: vertex ``(r, c)`` lies at
  ``((c - c0) dx, (r - r0) dy, z(r, c) - z_ref)``, computed from integer index differences, so a vertex shared by
  several cells is bit-identical in all of them (the narrow phase's per-cell frame ``x1 = x0 + dx`` leaves cracks
  of up to 9.5 um on a 200 m terrain). The window's origin is the exact grid position in float64 and a core
  point enters the frame rounded once, so its position against the grid is exact to a float32 rounding of a
  window-sized number (a float32 origin rounds by up to 4 um 100 m out). Heights are decoded as the narrow
  phase decodes them on the CPU (float32 ``min_z + e * (max_z - min_z)``, each operation rounded once), on
  every device and in every kernel (:func:`terrain_decode` uses explicitly rounded operations, which no
  floating-point contraction fuses).
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

Rows carry a window-local feature id in the 23-bit ``sort_sub_key`` of the contact contract (19 bits:
``r mod 64``, ``c mod 64``, kind, sub-feature, point slot), unique within any window under 64 cells across.

The per-pair finish (:func:`terrain_finish`) merges coincident rows, drops the rows the others of their plane
group imply (the lower convex hull of tangent position and distance, residual ``1e-6 m``), keeps at most four rows
per plane group (deepest, farthest, largest triangle, largest quadrilateral) and at most ``row_limit`` rows per
pair, each group's deepest row first and then its extremes; an overflow sets a sticky bit of the narrow phase's
buffer status word, which ``verify_narrow_phase_buffers`` keeps.
"""

from __future__ import annotations

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
TERRAIN_VERTEX_SPAN = 1.0e-6
"""A vertex is a feature when three of its incident cluster normals have a triple product above this."""
TERRAIN_CONE_BAND = 1.0e-6
"""Relative band of the closed normal-cone test ``n . (w - v) <= band |w - v|``."""
TERRAIN_GROUP_ROWS = 4
"""Rows kept per plane group."""
TERRAIN_ROW_CANDIDATES = 32
"""Candidate rows one pair's finish holds."""
TERRAIN_TILE_CELLS = 8
"""Cells per side of a tile of the tile-max mip."""

TERRAIN_EDGE_FLAT = wp.constant(0)
"""Edge class: both triangles in one plane cluster."""
TERRAIN_EDGE_CONVEX = wp.constant(1)
"""Edge class: a ridge between two clusters."""
TERRAIN_EDGE_CONCAVE = wp.constant(2)
"""Edge class: a valley between two clusters."""
TERRAIN_EDGE_BORDER = wp.constant(3)
"""Edge class: on the heightfield's border."""

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

TERRAIN_FINISH_CANDIDATE_OVERFLOW = wp.constant(1)
"""Finish flag: the pair produced more candidates than :data:`TERRAIN_ROW_CANDIDATES`."""
TERRAIN_FINISH_ROW_OVERFLOW = wp.constant(2)
"""Finish flag: more rows survived than the pair's row limit."""
TERRAIN_STATUS_ROWS = wp.constant(1 << 13)
"""Sticky bit of the narrow phase's buffer status word: a terrain pair overflowed its candidates or rows."""
TERRAIN_STATUS_STALE = wp.constant(1 << 14)
"""Sticky bit of the narrow phase's buffer status word: terrain features older than the heights were read."""

TerrainRowsVec3 = wp.types.matrix(shape=(TERRAIN_ROW_CANDIDATES, 3), dtype=wp.float32)
TerrainRowsFloat = wp.types.vector(length=TERRAIN_ROW_CANDIDATES, dtype=wp.float32)
TerrainRowsInt = wp.types.vector(length=TERRAIN_ROW_CANDIDATES, dtype=wp.int32)

_CANDIDATES = wp.constant(TERRAIN_ROW_CANDIDATES)
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
_GROUP_ROWS = wp.constant(TERRAIN_GROUP_ROWS)


# ----------------------------------------------------------------------------------------------------------------
# static data as the kernels see it
# ----------------------------------------------------------------------------------------------------------------


@wp.struct
class TerrainSurfaceInfo:
    """One heightfield's layout in the law's static arrays."""

    data_offset: int
    """First elevation of the heightfield in the model's concatenated elevations."""
    nrow: int
    """Vertex rows."""
    ncol: int
    """Vertex columns."""
    hx: float
    """Half-extent along x [m], scale applied."""
    hy: float
    """Half-extent along y [m], scale applied."""
    min_z: float
    """Height of elevation 0 [m], scale applied."""
    max_z: float
    """Height of elevation 1 [m], scale applied."""
    dx: float
    """Cell size along x [m] (float32 ``2 hx / (ncol - 1)``, as the narrow phase computes it)."""
    dy: float
    """Cell size along y [m]."""
    triangle_offset: int
    """First cluster id of the heightfield."""
    plane_offset: int
    """First cluster plane record of the heightfield."""
    tile_offset: int
    """First tile-max value of the heightfield."""
    tile_cols: int
    """Tiles per tile row."""


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

    elevations: wp.array[wp.float32]
    """The model's concatenated normalized elevations."""
    clusters: wp.array[wp.int32]
    """Per triangle: the dense index of its cluster's plane record, or ``-(1 + local triangle id)`` alone."""
    planes: wp.array[TerrainClusterPlane]
    """Plane records of the clusters of several triangles."""
    tile_max: wp.array[wp.float32]
    """Per tile of 8 x 8 cells: the largest decoded height of its vertices [m]."""
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
class TerrainRows:
    """Candidate rows of one pair, in feature order (window frame)."""

    count: int
    """Rows held."""
    dropped: int
    """Rows refused because the candidates were full."""
    q: TerrainRowsVec3
    """Terrain points [m]."""
    k: TerrainRowsVec3
    """Core points [m]."""
    n: TerrainRowsVec3
    """Unit normals from the terrain to the shape."""
    d: TerrainRowsFloat
    """Core distances ``n . (k - q)`` [m]."""
    feature: TerrainRowsInt
    """Feature ids (:func:`terrain_feature_id`)."""


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
    """Return the decoded height of vertex ``(r, c)`` [m] (heightfield-local)."""
    e = data.elevations[surface.data_offset + r * surface.ncol + c]
    return terrain_decode(surface.min_z, surface.max_z, e)


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

    The origin is the exact grid position ``(-hx + c0 dx, -hy + r0 dy)`` in float64: in float32 it would round
    by up to 4 um on a 200 m terrain and shift every core point against the grid by as much.
    """
    window = TerrainWindow()
    surface = data.surfaces[heightfield]
    window.surface = surface
    window.heightfield = heightfield
    window.r0 = r0
    window.c0 = c0
    window.z_ref = terrain_height(data, surface, r0, c0)
    hx = wp.float64(surface.hx)
    hy = wp.float64(surface.hy)
    window.origin = wp.vec3d(
        -hx + wp.float64(c0) * (wp.float64(2.0) * hx / wp.float64(surface.ncol - 1)),
        -hy + wp.float64(r0) * (wp.float64(2.0) * hy / wp.float64(surface.nrow - 1)),
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
        dx = wp.float64(2.0) * wp.float64(surface.hx) / wp.float64(surface.ncol - 1)
        dy = wp.float64(2.0) * wp.float64(surface.hy) / wp.float64(surface.nrow - 1)
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
def terrain_edge_class(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, which: int) -> int:
    """Return the class of edge ``which`` of vertex ``(r, c)``: flat inside one cluster, else convex or concave.

    The far vertex of the second triangle above the first triangle's cluster plane makes a valley (concave);
    two different clusters always meet at a crease, however small.
    """
    _v0, _v1, ta, tb, _wa, wb = terrain_edge_sides(r, c, which)
    surface = window.surface
    if not terrain_valid_cell(surface, ta[0], ta[1]) or not terrain_valid_cell(surface, tb[0], tb[1]):
        return TERRAIN_EDGE_BORDER
    if terrain_cluster(data, surface, ta[0], ta[1], ta[2]) == terrain_cluster(data, surface, tb[0], tb[1], tb[2]):
        return TERRAIN_EDGE_FLAT
    n, a = terrain_cluster_plane(data, window, ta[0], ta[1], ta[2])
    if wp.dot(n, terrain_window_vertex(data, window, wb[0], wb[1]) - a) > 0.0:
        return TERRAIN_EDGE_CONCAVE
    return TERRAIN_EDGE_CONVEX


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


_RingNormals = wp.types.matrix(shape=(6, 3), dtype=wp.float32)


@wp.func
def _ring_normal(normals: _RingNormals, i: int) -> wp.vec3:
    return wp.vec3(normals[i, 0], normals[i, 1], normals[i, 2])


@wp.func
def terrain_vertex_feature(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int) -> bool:
    """Return whether vertex ``(r, c)`` is a feature of its own: its incident cluster normals span 3-D.

    Normals closer than :data:`TERRAIN_NORMAL_SAME` are one direction; three distinct directions span 3-D when
    their triple product exceeds :data:`TERRAIN_VERTEX_SPAN`.
    """
    surface = window.surface
    if not terrain_interior_vertex(surface, r, c):
        return False
    normals = _RingNormals()
    count = int(0)
    for j in range(6):
        t = terrain_incident_triangle(r, c, j)
        n, _a = terrain_cluster_plane(data, window, t[0], t[1], t[2])
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
            nj = _ring_normal(normals, j)
            for k in range(j + 1, count):
                if wp.abs(wp.dot(ni, wp.cross(nj, _ring_normal(normals, k)))) > _VERTEX_SPAN:
                    return True
    return False


@wp.func
def terrain_vertex_in_cone(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, n: wp.vec3) -> bool:
    """Return whether direction ``n`` lies in vertex ``(r, c)``'s normal cone (closed): ``n . (w - v) <= 0``."""
    v = terrain_window_vertex(data, window, r, c)
    for j in range(6):
        w = terrain_ring_vertex(r, c, j)
        e = terrain_window_vertex(data, window, w[0], w[1]) - v
        if wp.dot(n, e) > _CONE_BAND * wp.length(e):
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
def terrain_vertex_point(data: TerrainFeatureData, window: TerrainWindow, r: int, c: int, k: wp.vec3) -> bool:
    """Return whether vertex ``(r, c)`` claims point ``k``: ``(k - v) . (w - v) <= 0`` for its whole one-ring."""
    if not terrain_interior_vertex(window.surface, r, c):
        return False
    v = terrain_window_vertex(data, window, r, c)
    for j in range(6):
        wi = terrain_ring_vertex(r, c, j)
        w = terrain_window_vertex(data, window, wi[0], wi[1])
        if not terrain_claims(terrain_vertex_edge_value(k, v, w), window.eps * wp.length(w - v)):
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
    """Record a pair's finish flags and staleness in the narrow phase's sticky status word and counters.

    ``overflow[0]`` counts pairs whose rows overflowed, ``overflow[1]`` pairs whose candidates did.
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
def terrain_rows_push(rows: TerrainRows, q: wp.vec3, k: wp.vec3, n: wp.vec3, d: float, feature: int) -> TerrainRows:
    """Append a candidate row (counted as dropped when the candidates are full)."""
    i = rows.count
    if i < _CANDIDATES:
        for j in range(3):
            rows.q[i, j] = q[j]
            rows.k[i, j] = k[j]
            rows.n[i, j] = n[j]
        rows.d[i] = d
        rows.feature[i] = feature
        rows.count = i + 1
    else:
        rows.dropped = rows.dropped + 1
    return rows


@wp.func
def terrain_row_q(rows: TerrainRows, i: int) -> wp.vec3:
    """Return candidate ``i``'s terrain point."""
    return wp.vec3(rows.q[i, 0], rows.q[i, 1], rows.q[i, 2])


@wp.func
def terrain_row_k(rows: TerrainRows, i: int) -> wp.vec3:
    """Return candidate ``i``'s core point."""
    return wp.vec3(rows.k[i, 0], rows.k[i, 1], rows.k[i, 2])


@wp.func
def terrain_row_n(rows: TerrainRows, i: int) -> wp.vec3:
    """Return candidate ``i``'s normal."""
    return wp.vec3(rows.n[i, 0], rows.n[i, 1], rows.n[i, 2])


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
def _lower_hull_value(
    u: TerrainRowsVec3, members: TerrainRowsInt, count: int, skip: int, point: wp.vec2, eps: float
) -> tuple[bool, float]:
    """Lowest convex interpolation of the members' distances (``u[i, 2]``) at a tangent position, ``skip`` excluded.

    ``u[i]`` holds a member's tangent coordinates and distance. Positions within ``4 eps`` coincide; a position
    on a segment or in a triangle of other members interpolates their distances.
    """
    found = bool(False)
    best = float(0.0)
    tolerance = _MERGE_SCALE * eps
    for ai in range(count):
        if ai == skip:
            continue
        a = members[ai]
        pa = wp.vec2(u[a, 0], u[a, 1])
        if wp.length(point - pa) <= tolerance:
            if not found or u[a, 2] < best:
                best = u[a, 2]
                found = True
        for bi in range(ai + 1, count):
            if bi == skip:
                continue
            b = members[bi]
            pb = wp.vec2(u[b, 0], u[b, 1])
            ab = pb - pa
            length2 = wp.dot(ab, ab)
            if length2 > 0.0:
                s = wp.dot(point - pa, ab) / length2
                perpendicular = wp.abs(_cross2(ab, point - pa)) / wp.sqrt(length2)
                if s >= -_BARYCENTRIC_TOLERANCE and s <= 1.0 + _BARYCENTRIC_TOLERANCE and perpendicular <= tolerance:
                    value = u[a, 2] + s * (u[b, 2] - u[a, 2])
                    if not found or value < best:
                        best = value
                        found = True
            for ci in range(bi + 1, count):
                if ci == skip:
                    continue
                cc = members[ci]
                pc = wp.vec2(u[cc, 0], u[cc, 1])
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
                        value = u[a, 2] + beta * (u[b, 2] - u[a, 2]) + gamma * (u[cc, 2] - u[a, 2])
                        if not found or value < best:
                            best = value
                            found = True
    return found, best


@wp.func
def _group_members(
    alive: TerrainRowsInt, group: TerrainRowsInt, count: int, representative: int
) -> tuple[TerrainRowsInt, int]:
    members = TerrainRowsInt()
    size = int(0)
    for i in range(count):
        if alive[i] != 0 and group[i] == representative:
            members[size] = i
            size += 1
    return members, size


@wp.func
def terrain_finish(rows: TerrainRows, eps: float, row_limit: int) -> tuple[TerrainRowsInt, int, int]:
    """Select one pair's rows: merge, implied-row removal, four rows per plane group, ``row_limit`` per pair.

    Returns ``(order, kept, flags)``: the kept candidates' indices in candidate order, their count, and
    :data:`TERRAIN_FINISH_CANDIDATE_OVERFLOW` / :data:`TERRAIN_FINISH_ROW_OVERFLOW` bits. Ties are broken by
    candidate (feature) order, never by memory order.
    """
    count = wp.min(rows.count, _CANDIDATES)
    flags = int(0)
    if rows.dropped > 0:
        flags = flags | TERRAIN_FINISH_CANDIDATE_OVERFLOW
    alive = TerrainRowsInt()
    group = TerrainRowsInt()
    rank = TerrainRowsInt()
    # 1. coincident rows: the first in feature order stays
    for i in range(count):
        alive[i] = 1
        qi = terrain_row_q(rows, i)
        ni = terrain_row_n(rows, i)
        for j in range(i):
            if alive[j] != 0 and terrain_rows_coincide(terrain_row_q(rows, j), terrain_row_n(rows, j), qi, ni, eps):
                alive[i] = 0
                break
    # 2. plane groups: the first row of a plane represents it
    for i in range(count):
        group[i] = -1
        if alive[i] == 0:
            continue
        qi = terrain_row_q(rows, i)
        ni = terrain_row_n(rows, i)
        group[i] = i
        for j in range(i):
            if alive[j] != 0 and group[j] == j:
                nj = terrain_row_n(rows, j)
                if wp.length(ni - nj) <= _GROUP_NORMAL and wp.abs(wp.dot(nj, qi - terrain_row_q(rows, j))) <= (
                    _MERGE_SCALE * eps
                ):
                    group[i] = j
                    break
    # 3. implied rows, the shallowest first, one at a time
    u = TerrainRowsVec3()
    for g in range(count):
        if alive[g] == 0 or group[g] != g:
            continue
        e1, e2 = _tangent_frame(terrain_row_n(rows, g))
        changed = bool(True)
        while changed:
            changed = False
            members, size = _group_members(alive, group, count, g)
            if size < 3:
                break
            for mi in range(size):
                i = members[mi]
                ki = terrain_row_k(rows, i)
                u[i, 0] = wp.dot(ki, e1)
                u[i, 1] = wp.dot(ki, e2)
                u[i, 2] = rows.d[i]
            tested = TerrainRowsInt()
            for _pass in range(size):
                pick = int(-1)
                for mi in range(size):
                    if tested[mi] == 0:
                        if pick < 0 or rows.d[members[mi]] > rows.d[members[pick]]:
                            pick = mi
                tested[pick] = 1
                j = members[pick]
                found, value = _lower_hull_value(u, members, size, pick, wp.vec2(u[j, 0], u[j, 1]), eps)
                if found and rows.d[j] >= value - _IMPLIED_RESIDUAL:
                    alive[j] = 0
                    changed = True
                    break
    # 4. at most four rows per group: deepest, farthest, largest triangle, largest quadrilateral
    for i in range(count):
        rank[i] = 4
    total = int(0)
    for g in range(count):
        # A group keeps its first row's id even when that row was implied.
        if group[g] != g:
            continue
        members, size = _group_members(alive, group, count, g)
        if size == 0:
            continue
        n = terrain_row_n(rows, g)
        i0 = int(-1)
        for mi in range(size):
            if i0 < 0 or rows.d[members[mi]] < rows.d[i0]:
                i0 = members[mi]
        p0 = terrain_row_k(rows, i0)
        p0 = p0 - wp.dot(p0, n) * n
        rank[i0] = 0
        total += 1
        i1 = int(-1)
        far = float(-1.0)
        for mi in range(size):
            j = members[mi]
            if rank[j] == 0:
                continue
            pj = terrain_row_k(rows, j)
            distance = wp.length(pj - wp.dot(pj, n) * n - p0)
            if distance > far:
                far = distance
                i1 = j
        if i1 < 0:
            continue
        rank[i1] = 1
        total += 1
        p1 = terrain_row_k(rows, i1)
        p1 = p1 - wp.dot(p1, n) * n
        i2 = int(-1)
        largest = float(-1.0)
        for mi in range(size):
            j = members[mi]
            if rank[j] <= 1:
                continue
            pj = terrain_row_k(rows, j)
            area = wp.abs(wp.dot(wp.cross(p1 - p0, pj - wp.dot(pj, n) * n - p0), n))
            if area > largest:
                largest = area
                i2 = j
        if i2 < 0:
            continue
        rank[i2] = 2
        total += 1
        p2 = terrain_row_k(rows, i2)
        p2 = p2 - wp.dot(p2, n) * n
        i3 = int(-1)
        gain_best = float(1.0e-15)
        for mi in range(size):
            j = members[mi]
            if rank[j] <= 2:
                continue
            pj = terrain_row_k(rows, j)
            pj = pj - wp.dot(pj, n) * n
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
                side_c = wp.dot(wp.cross(pb - pa, pc - pa), n)
                side_j = wp.dot(wp.cross(pb - pa, pj - pa), n)
                if side_c * side_j < 0.0:
                    gain = wp.max(gain, wp.abs(side_j))
            if gain > gain_best:
                gain_best = gain
                i3 = j
        if i3 >= 0:
            rank[i3] = 3
            total += 1
        if size <= _GROUP_ROWS:
            # A group of at most four rows keeps them all.
            for mi in range(size):
                j = members[mi]
                if rank[j] == _GROUP_ROWS:
                    rank[j] = _GROUP_ROWS - 1
                    total += 1
    # 5. at most row_limit rows: the groups' deepest rows first, then their extremes
    keep = TerrainRowsInt()
    if total <= row_limit:
        for i in range(count):
            if alive[i] != 0 and rank[i] < _GROUP_ROWS:
                keep[i] = 1
    else:
        flags = flags | TERRAIN_FINISH_ROW_OVERFLOW
        taken = int(0)
        for level in range(_GROUP_ROWS):
            while taken < row_limit:
                pick = int(-1)
                for i in range(count):
                    if alive[i] != 0 and rank[i] == level and keep[i] == 0:
                        if pick < 0 or rows.d[i] < rows.d[pick]:
                            pick = i
                if pick < 0:
                    break
                keep[pick] = 1
                taken += 1
    order = TerrainRowsInt()
    kept = int(0)
    for i in range(count):
        if keep[i] != 0:
            order[kept] = i
            kept += 1
    return order, kept, flags

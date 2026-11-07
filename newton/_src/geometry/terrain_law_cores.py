# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solid cores of the terrain law (rounding radius 0): boxes, cylinders (barrel cylinders too) and cones, convex hulls
and ellipsoids.

One struct per partner class, each with overloads of the same functions, which the law's generic per-pair walk
(:mod:`terrain_law_solid`) calls:

* :func:`core_sigma` (``min n . (k - p0)`` over the core) and :func:`core_slope` (its one-sided derivative along a
  turning normal, tied support points counted on the side asked);
* :func:`core_manifold`: the analytic plane manifold of rule F, at most four points in order;
* :func:`core_flat_crossing` and :func:`core_reference_crossing`: rule E's interval where the core's flat support
  feature (an interior ``theta*``) or its reference feature (an arc end, G1) meets a crease's line;
* :func:`core_lift` (where a line enters the core), :func:`core_nearest` (rule V: the direction to the nearest point,
  or the minimum translation from inside) and :func:`core_recovery_point` (rule R's boundary points);
* :func:`core_bounds` (axis-aligned bounds), :func:`core_sphere` (a bounding sphere), :func:`core_capsule` (a bounding
  capsule: a box's longest axis, a cylinder's or cone's axis, a hull's or ellipsoid's bounding box's longest axis) and
  :func:`core_sigma_bound` (a lower bound of :func:`core_sigma` in closed form: the value itself, or for a hull its
  bounding box's).

Every definition is the float64 law's (``newton/tests/terrain_law_reference.py``), in float32: the plane manifolds are
Newton's analytic plane queries (a box's four deepest corners, the cylinder's deepest rim point with a tripod or the
rolling points, a hull's face through its deepest vertex most opposed to the normal with that vertex, reduced to four
by area, an ellipsoid's support point); a cone's mirrors the cylinder's with its base as the only cap, a barrel
cylinder's takes its lowest point (Newton's support point) with the cylinder's tripod within 22.5 deg of upright. Ties
are broken as the float64 law breaks them (lowest index first), and boundary decisions are closed by the caller's
tolerances.
"""

from __future__ import annotations

import math

import warp as wp

from .collision_primitive import CYLINDER_FLAT_MODE_COS

TERRAIN_SOLID_TIE = 1.0e-6
"""Tie band of the plane manifolds' and reference features' discrete choices (DESIGN 1.4): of candidates within this
times the core's bounding radius in height or distance (area: times a length of the face), or within it in a normal's
score, the first in index order wins, as in the float64 law. Two are ties by construction: a rolling cylinder's two
cap-rim side points lie at one height, and a lying cylinder's caps face a plane equally."""
TERRAIN_HULL_FACE_VERTICES = 64
"""Vertices of one hull face the plane manifold takes (a larger face is truncated in index order)."""
TERRAIN_HULL_BAND_POINTS = 32
"""Points of a hull's flat support feature (within the flatness band of its lowest along a crease's normal) that rule
E's crossing lists to pair them; a feature of more points takes every point's pairs from the whole hull (the same
pairs, in the same order)."""

_FLAT_COS = wp.constant(CYLINDER_FLAT_MODE_COS)
_TIE = wp.constant(TERRAIN_SOLID_TIE)
_HUGE = wp.constant(3.0e38)
_FACE_CAP = wp.constant(TERRAIN_HULL_FACE_VERTICES)
_BAND_POINTS = wp.constant(TERRAIN_HULL_BAND_POINTS)
_TRIPOD_S = wp.constant(0.8660254)
_ALIGN = wp.constant(1.0e-10)
"""Squared length below which a cylinder's or cone's radial direction is degenerate (the axis along the normal)."""
_THIRD = wp.constant(1.0 / math.sqrt(3.0))
_RING = wp.constant(16)
"""Rim points per cap that rule R tries for cylinders and cones."""
_RingTable = wp.types.matrix(shape=(16, 2), dtype=wp.float32)
_RING_ANGLES = wp.constant(
    _RingTable(*[v for i in range(16) for v in (math.cos(2.0 * math.pi * i / 16), math.sin(2.0 * math.pi * i / 16))])
)
"""``(cos, sin)`` of the rim points' angles ``2 pi i / 16`` (a table: CUDA's trigonometry fuses operations)."""
_ELLIPSOID_STEPS = wp.constant(48)
"""Bisection steps of the ellipsoid's nearest-point (secular) equation."""
_BARREL_STEPS = wp.constant(48)
"""Golden-section steps of a barrel's lift to the least excess along the line, then bisection steps to its entry."""
_GOLDEN = wp.constant(0.6180339887498949)

_Vec4i = wp.types.vector(length=4, dtype=wp.int32)
_Mat43 = wp.types.matrix(shape=(4, 3), dtype=wp.float32)
_FaceList = wp.types.vector(length=TERRAIN_HULL_FACE_VERTICES + 1, dtype=wp.int32)
_BandPoints = wp.types.vector(length=TERRAIN_HULL_BAND_POINTS, dtype=wp.int32)


# ----------------------------------------------------------------------------------------------------------------
# core types
# ----------------------------------------------------------------------------------------------------------------


@wp.struct
class TerrainBoxCore:
    """A box: centre ``c``, rotation ``rot`` (columns: its local axes) and half extents ``h``."""

    c: wp.vec3
    rot: wp.mat33
    h: wp.vec3


@wp.struct
class TerrainCylinderCore:
    """A flat-capped cylinder (``cone`` 0) or a cone (``cone`` 1): centre ``c``, unit axis ``a`` (local z), radius ``r``
    and half height ``h``; a cone's apex is ``c + h a`` and its base disk is centred at ``c - h a``. A barrel cylinder
    (``barrel`` the radius ``R >= h`` of its side's profile, 0 for a straight side) has caps of radius ``r`` and a side
    of revolution of a circular arc through their rims (:func:`_barrel_radius`), Newton's barrel cylinder."""

    c: wp.vec3
    a: wp.vec3
    r: float
    h: float
    cone: int
    barrel: float


@wp.struct
class TerrainHullTable:
    """Static data of the convex hulls a shadow's hull class meets, all hulls concatenated (global indices).

    Points are the meshes' own vertices, unscaled, in mesh order (the narrow phase's support set); faces are the
    hull's planar faces, each with its vertex indices in increasing order; every point lists its incident faces in
    increasing order (the Gauss map's vertex cells) and the points it shares a face with (the hull's point graph,
    which a support search descends); every face is fanned into triangles for nearest points.
    """

    points: wp.array[wp.vec3]
    """Unscaled local points of every hull."""
    planes: wp.array[wp.vec4]
    """Per face: its unit outward normal (unscaled frame) and offset (``n . x <= w`` inside)."""
    face_vertices: wp.array[wp.int32]
    """Per face: its point indices in increasing order; face ``f`` spans ``face_begin[f]`` to ``face_begin[f + 1]``."""
    face_begin: wp.array[wp.int32]
    vertex_faces: wp.array[wp.int32]
    """Per point: its incident faces in increasing order; point ``p`` spans ``vertex_face_begin[p]`` and on."""
    vertex_face_begin: wp.array[wp.int32]
    neighbors: wp.array[wp.int32]
    """Per point: the points it shares a face with, in increasing order; point ``p`` spans ``neighbor_begin[p]`` and
    on."""
    neighbor_begin: wp.array[wp.int32]
    triangles: wp.array[wp.vec3i]
    """Fan triangles of every face (point indices)."""
    hull_points: wp.array[wp.vec2i]
    """Per hull: its first point and point count."""
    hull_faces: wp.array[wp.vec2i]
    """Per hull: its first face and face count."""
    hull_triangles: wp.array[wp.vec2i]
    """Per hull: its first triangle and triangle count."""
    shape_hull: wp.array[wp.int32]
    """Per shape: its hull's index in this table, or -1."""


@wp.struct
class TerrainHullCore:
    """A convex hull (``ellipsoid`` 0) or an ellipsoid (``ellipsoid`` 1): centre ``c``, rotation ``rot``, and the mesh
    scale or the semi-axes ``s``; a hull reads its points and faces from ``table``."""

    c: wp.vec3
    rot: wp.mat33
    s: wp.vec3
    ellipsoid: int
    point_begin: int
    point_count: int
    face_begin: int
    face_count: int
    triangle_begin: int
    triangle_count: int
    radius: float
    """The largest distance of a point from the centre (scaled), or the largest semi-axis."""
    start: int
    """A hull's point lowest along the heightfield's up axis (global index): where support searches start."""
    box_c: wp.vec3
    """Centre of the core's bounding box in its frame (scaled; an ellipsoid's 0): the box ``c + rot (box_c + [-box_h,
    box_h])`` contains the core (:func:`core_sigma_bound`, :func:`core_capsule`)."""
    box_h: wp.vec3
    """Half extents of the core's bounding box in its frame (scaled; an ellipsoid's semi-axes)."""
    table: TerrainHullTable


@wp.struct
class TerrainHullCoreRecord:
    """A :class:`TerrainHullCore` without its table: how a hull pair's scan keeps its core for its job kernels."""

    c: wp.vec3
    rot: wp.mat33
    s: wp.vec3
    ellipsoid: int
    point_begin: int
    point_count: int
    face_begin: int
    face_count: int
    triangle_begin: int
    triangle_count: int
    radius: float
    start: int
    box_c: wp.vec3
    box_h: wp.vec3


# ----------------------------------------------------------------------------------------------------------------
# shared helpers: crossings of a feature with a crease's line
# ----------------------------------------------------------------------------------------------------------------


@wp.struct
class TerrainCrossing:
    """A feature's crossing with a crease's line ``y = 0``: the interval ``[lo, hi]`` and the feature's points at its
    ends (the core points of rows at unclipped ends: a line along the normal through one only grazes the core)."""

    lo: float
    hi: float
    k_lo: wp.vec3
    k_hi: wp.vec3


@wp.func
def _crossing() -> TerrainCrossing:
    crossing = TerrainCrossing()
    crossing.lo = _HUGE
    crossing.hi = -_HUGE
    return crossing


@wp.func
def _cross_point(crossing: TerrainCrossing, x: float, y: float, p: wp.vec3, tol: float) -> TerrainCrossing:
    """Add feature point ``p`` (at ``(x, y)``) when within ``tol`` of the line ``y = 0``."""
    if wp.abs(y) <= tol:
        if x < crossing.lo:
            crossing.lo = x
            crossing.k_lo = p
        if x > crossing.hi:
            crossing.hi = x
            crossing.k_hi = p
    return crossing


@wp.func
def _cross_segment(
    crossing: TerrainCrossing, xa: float, ya: float, pa: wp.vec3, xb: float, yb: float, pb: wp.vec3, tol: float
) -> TerrainCrossing:
    """Add the crossing of feature segment ``ab`` with the line ``y = 0`` (ends strictly on both sides)."""
    if (ya < -tol and yb > tol) or (ya > tol and yb < -tol):
        s = ya / (ya - yb)
        x = xa + s * (xb - xa)
        p = pa + s * (pb - pa)
        if x < crossing.lo:
            crossing.lo = x
            crossing.k_lo = p
        if x > crossing.hi:
            crossing.hi = x
            crossing.k_hi = p
    return crossing


@wp.func
def _disk_basis(axis: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """Two unit vectors spanning the plane normal to ``axis``: from x, or y within 26 deg of x (the float64 law's)."""
    ref = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(axis[0]) >= 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    u = wp.normalize(ref - axis * wp.dot(axis, ref))
    return u, wp.cross(axis, u)


@wp.func
def _disk_crossing(
    centre: wp.vec3,
    axis: wp.vec3,
    r: float,
    v0: wp.vec3,
    e1: wp.vec3,
    e2: wp.vec3,
    tol: float,
    straddle: int,
    straddle_tol: float,
) -> tuple[bool, TerrainCrossing]:
    """Where a disk, projected along ``e1 x e2``, meets the line ``y = 0`` (closed-form chord of its ellipse).

    With ``straddle`` the disk must cross the line strictly (by ``straddle_tol``).
    """
    crossing = _crossing()
    u, w = _disk_basis(axis)
    yc = wp.dot(centre - v0, e2)
    xc = wp.dot(centre - v0, e1)
    a_ = wp.dot(u, e2)
    b_ = wp.dot(w, e2)
    radius = r * wp.sqrt(a_ * a_ + b_ * b_)
    u1 = wp.dot(u, e1)
    w1 = wp.dot(w, e1)
    if radius <= 1.0e-7 * r:
        if wp.abs(yc) > tol or straddle != 0:
            return False, crossing
        # edge-on: the disk projects onto its diameter along the line
        ext = wp.sqrt(u1 * u1 + w1 * w1)
        along = wp.normalize(u1 * u + w1 * w)
        crossing.lo = xc - r * ext
        crossing.hi = xc + r * ext
        crossing.k_lo = centre - r * along
        crossing.k_hi = centre + r * along
        return True, crossing
    if wp.abs(yc) > radius + tol:
        return False, crossing
    if straddle != 0 and wp.abs(yc) >= radius - straddle_tol:
        return False, crossing
    # The rim point at angle phi crosses where a cos phi + b sin phi = -yc / r: with (a, b) = R (ca, sa), the two
    # solutions are (cos phi, sin phi) = k (ca, sa) +- sqrt(1 - k^2) (-sa, ca), k = -yc / R (no trigonometry, which
    # CUDA evaluates with fused operations).
    k = wp.clamp(-yc / radius, -1.0, 1.0)
    ca = r * a_ / radius
    sa = r * b_ / radius
    root = wp.sqrt(wp.max(1.0 - k * k, 0.0))
    cos_a = k * ca - root * sa
    sin_a = k * sa + root * ca
    cos_b = k * ca + root * sa
    sin_b = k * sa - root * ca
    xa = xc + r * (cos_a * u1 + sin_a * w1)
    xb = xc + r * (cos_b * u1 + sin_b * w1)
    pa = centre + r * (cos_a * u + sin_a * w)
    pb = centre + r * (cos_b * u + sin_b * w)
    if xa <= xb:
        crossing.lo = xa
        crossing.hi = xb
        crossing.k_lo = pa
        crossing.k_hi = pb
    else:
        crossing.lo = xb
        crossing.hi = xa
        crossing.k_lo = pb
        crossing.k_hi = pa
    return True, crossing


@wp.func
def _two_point_crossing(
    pa: wp.vec3, pb: wp.vec3, v0: wp.vec3, e1: wp.vec3, e2: wp.vec3, tol: float, straddle: int, straddle_tol: float
) -> tuple[bool, TerrainCrossing]:
    """Where segment ``pa pb``, projected along ``e1 x e2``, meets the line ``y = 0``."""
    crossing = _crossing()
    xa = wp.dot(pa - v0, e1)
    ya = wp.dot(pa - v0, e2)
    xb = wp.dot(pb - v0, e1)
    yb = wp.dot(pb - v0, e2)
    if straddle != 0 and not (wp.min(ya, yb) < -straddle_tol and wp.max(ya, yb) > straddle_tol):
        return False, crossing
    crossing = _cross_point(crossing, xa, ya, pa, tol)
    crossing = _cross_point(crossing, xb, yb, pb, tol)
    crossing = _cross_segment(crossing, xa, ya, pa, xb, yb, pb, tol)
    return crossing.lo <= crossing.hi, crossing


@wp.func
def _point_crossing(p: wp.vec3, v0: wp.vec3, e1: wp.vec3, e2: wp.vec3, tol: float) -> tuple[bool, TerrainCrossing]:
    """Where one point, projected along ``e1 x e2``, lies on the line ``y = 0`` (within ``tol``)."""
    crossing = _crossing()
    crossing = _cross_point(crossing, wp.dot(p - v0, e1), wp.dot(p - v0, e2), p, tol)
    return crossing.lo <= crossing.hi, crossing


@wp.func
def _quadratic_entry(
    qa: float, qb: float, qc: float, t_lo: float, t_hi: float, scale: float, tol: float
) -> tuple[bool, float, float]:
    """Intersect ``[t_lo, t_hi]`` with ``{t : qa t^2 + qb t + qc <= 0}`` for ``qa > 0`` (a convex quadric's chord).

    ``scale`` converts the discriminant to a squared distance (``disc / (4 qa) = r^2 - rho^2`` for a cylinder), so a
    line that misses by at most ``tol`` still grazes at its closest approach.
    """
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        if disc / (4.0 * qa) < -2.0 * scale * tol:
            return False, t_lo, t_hi
        disc = 0.0
    sq = wp.sqrt(disc)
    return True, wp.max(t_lo, (-qb - sq) / (2.0 * qa)), wp.min(t_hi, (-qb + sq) / (2.0 * qa))


@wp.func
def _slab(num: float, denom: float, tol: float, t_lo: float, t_hi: float) -> tuple[bool, float, float]:
    """Intersect ``[t_lo, t_hi]`` with ``{t : denom t <= num}`` (a half-space along a line; ``tol`` for a parallel
    line just outside)."""
    if wp.abs(denom) <= 1.0e-12:
        return num >= -tol, t_lo, t_hi
    t = num / denom
    if denom > 0.0:
        return True, t_lo, wp.min(t_hi, t)
    return True, wp.max(t_lo, t), t_hi


# ----------------------------------------------------------------------------------------------------------------
# boxes
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _box_local(core: TerrainBoxCore, i: int) -> wp.vec3:
    """Corner ``i`` in the box frame: bit 0, 1, 2 select ``+h`` along x, y, z (the float64 law's order)."""
    return wp.vec3(
        wp.where((i & 1) != 0, core.h[0], -core.h[0]),
        wp.where((i & 2) != 0, core.h[1], -core.h[1]),
        wp.where((i & 4) != 0, core.h[2], -core.h[2]),
    )


@wp.func
def _box_corner(core: TerrainBoxCore, i: int) -> wp.vec3:
    return core.c + core.rot @ _box_local(core, i)


@wp.func
def core_bounds(core: TerrainBoxCore) -> tuple[wp.vec3, wp.vec3]:
    """Axis-aligned bounds of the core."""
    r = core.rot
    ext = wp.vec3(
        wp.abs(r[0, 0]) * core.h[0] + wp.abs(r[0, 1]) * core.h[1] + wp.abs(r[0, 2]) * core.h[2],
        wp.abs(r[1, 0]) * core.h[0] + wp.abs(r[1, 1]) * core.h[1] + wp.abs(r[1, 2]) * core.h[2],
        wp.abs(r[2, 0]) * core.h[0] + wp.abs(r[2, 1]) * core.h[1] + wp.abs(r[2, 2]) * core.h[2],
    )
    return core.c - ext, core.c + ext


@wp.func
def core_sphere(core: TerrainBoxCore) -> tuple[wp.vec3, float]:
    """A sphere containing the core: centre and radius."""
    return core.c, wp.length(core.h)


@wp.func
def core_sigma_bound(core: TerrainBoxCore, n: wp.vec3, p0: wp.vec3) -> float:
    """A lower bound of :func:`core_sigma` in closed form (a box's: the value itself)."""
    return core_sigma(core, n, p0)


@wp.func
def core_capsule(core: TerrainBoxCore) -> tuple[wp.vec3, wp.vec3, float]:
    """A capsule containing the core: its axis's ends and radius (a box: its longest axis through the centre, the
    radius its cross-section's half diagonal)."""
    k = int(0)
    if core.h[1] > core.h[k]:
        k = 1
    if core.h[2] > core.h[k]:
        k = 2
    axis = wp.vec3(core.rot[0, k], core.rot[1, k], core.rot[2, k]) * core.h[k]
    radius = wp.sqrt(wp.max(wp.dot(core.h, core.h) - core.h[k] * core.h[k], 0.0))
    return core.c - axis, core.c + axis, radius


@wp.func
def core_sigma(core: TerrainBoxCore, n: wp.vec3, p0: wp.vec3) -> float:
    """``min n . (k - p0)`` over the core (closed form)."""
    m = wp.transpose(core.rot) @ n
    return wp.dot(n, core.c - p0) - (core.h[0] * wp.abs(m[0]) + core.h[1] * wp.abs(m[1]) + core.h[2] * wp.abs(m[2]))


@wp.func
def core_slope(core: TerrainBoxCore, n: wp.vec3, dn: wp.vec3, p0: wp.vec3, side: float) -> float:
    """One-sided derivative of :func:`core_sigma` as ``n`` turns along ``dn``: ``side`` +1 from the right (tied lowest
    corners count at their smallest slope), -1 from the left (their largest)."""
    m = wp.transpose(core.rot) @ n
    dm = wp.transpose(core.rot) @ dn
    s = wp.dot(dn, core.c - p0)
    for i in range(3):
        if core.h[i] * wp.abs(m[i]) <= 0.0:
            s = s - side * core.h[i] * wp.abs(dm[i])
        elif m[i] > 0.0:
            s = s - core.h[i] * dm[i]
        else:
            s = s + core.h[i] * dm[i]
    return s


@wp.func
def core_sigma_near(core: TerrainBoxCore, n: wp.vec3, p0: wp.vec3) -> tuple[TerrainBoxCore, float]:
    """:func:`core_sigma` in a search over nearby normals (closed form: nothing to keep)."""
    return core, core_sigma(core, n, p0)


@wp.func
def core_slope_near(
    core: TerrainBoxCore, n: wp.vec3, dn: wp.vec3, p0: wp.vec3, side: float
) -> tuple[TerrainBoxCore, float]:
    """:func:`core_slope` in a search over nearby normals (closed form: nothing to keep)."""
    return core, core_slope(core, n, dn, p0, side)


@wp.func
def core_manifold(core: TerrainBoxCore, n: wp.vec3, a: wp.vec3, reach: float) -> tuple[int, _Mat43, wp.vec4]:
    """Rule F's plane manifold: the four deepest corners within ``reach`` of the plane ``(n, a)``, deepest first
    (ties within the band in corner order), each corner once: along a zero half extent (a finite plane's thickness)
    only the corner on its lower side."""
    order = _Vec4i(-1, -1, -1, -1)
    depth = wp.vec4(_HUGE)
    count = int(0)
    band = _TIE * wp.length(core.h)
    for i in range(8):
        if (
            (core.h[0] == 0.0 and (i & 1) != 0)
            or (core.h[1] == 0.0 and (i & 2) != 0)
            or (core.h[2] == 0.0 and (i & 4) != 0)
        ):
            continue
        h = wp.dot(n, _box_corner(core, i) - a)
        if h > reach:
            continue
        # insert after every kept corner not shallower by more than the tie band (stable in corner order)
        j = wp.min(count, 4)
        while j > 0 and depth[j - 1] > h + band:
            if j < 4:
                depth[j] = depth[j - 1]
                order[j] = order[j - 1]
            j -= 1
        if j < 4:
            depth[j] = h
            order[j] = i
        count = wp.min(count + 1, 4)
    points = _Mat43()
    for j in range(count):
        p = _box_corner(core, order[j])
        points[j, 0] = p[0]
        points[j, 1] = p[1]
        points[j, 2] = p[2]
    return count, points, depth


@wp.func
def core_flat_crossing(
    core: TerrainBoxCore, n: wp.vec3, band: float, v0: wp.vec3, e1: wp.vec3, e2: wp.vec3, tol: float
) -> tuple[bool, TerrainCrossing]:
    """Rule E, interior ``theta*``: where the corners within ``band`` of the lowest along ``n`` (a vertex, edge or
    face), projected along ``n``, meet the crease's line ``y = 0``."""
    low = float(_HUGE)
    for i in range(8):
        low = wp.min(low, wp.dot(n, _box_corner(core, i)))
    crossing = _crossing()
    for i in range(8):
        pi = _box_corner(core, i)
        if wp.dot(n, pi) > low + band:
            continue
        xi = wp.dot(pi - v0, e1)
        yi = wp.dot(pi - v0, e2)
        crossing = _cross_point(crossing, xi, yi, pi, tol)
        for j in range(i + 1, 8):
            pj = _box_corner(core, j)
            if wp.dot(n, pj) > low + band:
                continue
            crossing = _cross_segment(crossing, xi, yi, pi, wp.dot(pj - v0, e1), wp.dot(pj - v0, e2), pj, tol)
    return crossing.lo <= crossing.hi, crossing


@wp.func
def core_reference_crossing(
    core: TerrainBoxCore, n: wp.vec3, v0: wp.vec3, e1: wp.vec3, e2: wp.vec3, tol: float, straddle_tol: float
) -> tuple[bool, TerrainCrossing]:
    """Rule E at an arc end (G1): where the face through the deepest corner most opposed to ``n`` meets the crease's
    line, when it straddles the line."""
    m = wp.transpose(core.rot) @ n
    band = _TIE * wp.length(core.h)
    low = float(_HUGE)
    for i in range(8):
        low = wp.min(low, wp.dot(n, _box_corner(core, i)))
    deepest = int(0)
    for i in range(8):
        if wp.dot(n, _box_corner(core, i)) <= low + band:
            deepest = i
            break
    # the faces through the deepest corner, in the float64 law's face order (axis, then - before +): the first
    # within the tie band of the most opposed
    best = float(-_HUGE)
    for i in range(3):
        sign = wp.where(((deepest >> i) & 1) != 0, 1.0, -1.0)
        best = wp.max(best, -sign * m[i])
    axis = int(0)
    for i in range(3):
        sign = wp.where(((deepest >> i) & 1) != 0, 1.0, -1.0)
        if -sign * m[i] >= best - _TIE:
            axis = i
            break
    bit = (deepest >> axis) & 1
    ymin = float(_HUGE)
    ymax = float(-_HUGE)
    for i in range(8):
        if ((i >> axis) & 1) == bit:
            y = wp.dot(_box_corner(core, i) - v0, e2)
            ymin = wp.min(ymin, y)
            ymax = wp.max(ymax, y)
    crossing = _crossing()
    if not (ymin < -straddle_tol and ymax > straddle_tol):
        return False, crossing
    for i in range(8):
        if ((i >> axis) & 1) != bit:
            continue
        pi = _box_corner(core, i)
        xi = wp.dot(pi - v0, e1)
        yi = wp.dot(pi - v0, e2)
        crossing = _cross_point(crossing, xi, yi, pi, tol)
        for j in range(i + 1, 8):
            if ((j >> axis) & 1) != bit:
                continue
            pj = _box_corner(core, j)
            crossing = _cross_segment(crossing, xi, yi, pi, wp.dot(pj - v0, e1), wp.dot(pj - v0, e2), pj, tol)
    return crossing.lo <= crossing.hi, crossing


@wp.func
def core_lift(core: TerrainBoxCore, u: wp.vec3, n: wp.vec3, tol: float) -> tuple[bool, wp.vec3, float]:
    """Where the line ``u + t n`` enters the core: ``(hit, point, t)`` with the smallest ``t`` (slabs)."""
    rt = wp.transpose(core.rot)
    ul = rt @ (u - core.c)
    nl = rt @ n
    t_lo = float(-_HUGE)
    t_hi = float(_HUGE)
    for i in range(3):
        ok, t_lo, t_hi = _slab(core.h[i] - ul[i], nl[i], tol, t_lo, t_hi)
        if not ok:
            return False, u, 0.0
        ok, t_lo, t_hi = _slab(core.h[i] + ul[i], -nl[i], tol, t_lo, t_hi)
        if not ok:
            return False, u, 0.0
    if t_lo > t_hi + tol or t_lo <= -_HUGE:
        return False, u, 0.0
    return True, u + t_lo * n, t_lo


@wp.func
def core_nearest(core: TerrainBoxCore, v: wp.vec3) -> tuple[bool, wp.vec3, wp.vec3, float]:
    """Rule V: ``(ok, n, k, d)``: the unit direction from ``v`` to the core's nearest point ``k`` and the distance, or
    with ``v`` inside, the minimum translation (``n`` minus the nearest face's normal, ``d`` minus its depth)."""
    p = wp.transpose(core.rot) @ (v - core.c)
    cl = wp.vec3(
        wp.clamp(p[0], -core.h[0], core.h[0]),
        wp.clamp(p[1], -core.h[1], core.h[1]),
        wp.clamp(p[2], -core.h[2], core.h[2]),
    )
    if cl[0] != p[0] or cl[1] != p[1] or cl[2] != p[2]:
        k = core.c + core.rot @ cl
        w = k - v
        dist = wp.length(w)
        if dist <= 0.0:
            return False, w, k, 0.0
        return True, w / dist, k, dist
    # inside: the face of least depth, in the float64 law's face order (axis, then - before +)
    best = float(_HUGE)
    face = wp.vec3(0.0)
    for i in range(3):
        axis = wp.vec3(core.rot[0, i], core.rot[1, i], core.rot[2, i])
        if core.h[i] + p[i] < best:
            best = core.h[i] + p[i]
            face = -axis
        if core.h[i] - p[i] < best:
            best = core.h[i] - p[i]
            face = axis
    return True, -face, v + best * face, -best


@wp.func
def core_recovery_count(core: TerrainBoxCore) -> int:
    """Boundary points rule R tries."""
    return 8


@wp.func
def core_recovery_point(core: TerrainBoxCore, i: int) -> wp.vec3:
    """Rule R's boundary point ``i``: the corners in order."""
    return _box_corner(core, i)


# ----------------------------------------------------------------------------------------------------------------
# cylinders and cones
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _radial(n: wp.vec3, a: wp.vec3) -> wp.vec3:
    """``n`` without its component along the unit axis ``a``."""
    return n - wp.dot(n, a) * a


@wp.func
def _cone_perp(core: TerrainCylinderCore, n: wp.vec3) -> wp.vec3:
    """The base-plane direction of a cone's rim point lowest along ``n`` (fixed when the base is level)."""
    perp = -_radial(n, core.a)
    length2 = wp.dot(perp, perp)
    if length2 > _ALIGN:
        return perp / wp.sqrt(length2)
    u, _w = _disk_basis(-core.a)
    return u


@wp.func
def _cone_ends(core: TerrainCylinderCore, n: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """A cone's lowest generator: the base rim point under ``n`` and the apex."""
    return core.c - core.h * core.a + core.r * _cone_perp(core, n), core.c + core.h * core.a


@wp.func
def _cone_order(core: TerrainCylinderCore, n: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """A cone's lowest generator's ends, the lower first: the apex only when lower by more than the tie band."""
    g, p = _cone_ends(core, n)
    if wp.dot(n, p) < wp.dot(n, g) - _TIE * wp.sqrt(core.r * core.r + core.h * core.h):
        return p, g
    return g, p


@wp.func
def _barrel_radius(core: TerrainCylinderCore, z: float) -> float:
    """A barrel's side radius at axial height ``z`` (``|z| <= h``): ``r + (h^2 - z^2) / (sqrt(R^2 - z^2) +
    sqrt(R^2 - h^2))``, the arc through the caps' rims in Newton's support function's form (no cancellation)."""
    r2 = core.barrel * core.barrel
    denom = wp.sqrt(wp.max(r2 - z * z, 0.0)) + wp.sqrt(wp.max(r2 - core.h * core.h, 0.0))
    if denom <= 0.0:
        return core.r
    return core.r + (core.h * core.h - z * z) / denom


@wp.func
def _barrel_lowest(core: TerrainCylinderCore, n: wp.vec3) -> wp.vec3:
    """A barrel's point lowest along ``n``, Newton's support point along ``-n``: at axial height ``-R n . a`` on its
    side, clamped to a cap's rim (its centre when the axis lies along ``n`` within :data:`_ALIGN`). The radial
    direction is made normal to the axis again: near the axis rounding leaves ``n``'s radial part along it."""
    z = wp.clamp(-core.barrel * wp.dot(n, core.a), -core.h, core.h)
    perp = -_radial(n, core.a)
    perp = perp - wp.dot(perp, core.a) * core.a
    length2 = wp.dot(perp, perp)
    if length2 <= _ALIGN:
        return core.c + z * core.a
    return core.c + z * core.a + _barrel_radius(core, z) * perp / wp.sqrt(length2)


@wp.func
def _barrel_side(core: TerrainCylinderCore, n: wp.vec3) -> bool:
    """Whether a barrel's lowest point along ``n`` lies on its side's arc between the caps (a single point, its
    derivative along a turning normal the point's)."""
    return core.barrel * wp.abs(wp.dot(n, core.a)) < core.h


@wp.func
def _barrel_excess(core: TerrainCylinderCore, w: wp.vec3) -> float:
    """How far point ``c + w`` lies outside a barrel's side at its axial height (clamped to the caps): its distance from
    the axis less the side's radius there, convex along any line."""
    z = wp.dot(w, core.a)
    return wp.length(w - z * core.a) - _barrel_radius(core, wp.clamp(z, -core.h, core.h))


@wp.func
def core_bounds(core: TerrainCylinderCore) -> tuple[wp.vec3, wp.vec3]:
    if core.barrel > 0.0:
        lo = wp.vec3(0.0)
        hi = wp.vec3(0.0)
        for i in range(3):
            e = wp.vec3(0.0)
            e[i] = 1.0
            lo[i] = _barrel_lowest(core, e)[i]
            hi[i] = _barrel_lowest(core, -e)[i]
        return lo, hi
    a = core.a
    ext = core.r * wp.vec3(
        wp.sqrt(wp.max(0.0, 1.0 - a[0] * a[0])),
        wp.sqrt(wp.max(0.0, 1.0 - a[1] * a[1])),
        wp.sqrt(wp.max(0.0, 1.0 - a[2] * a[2])),
    )
    if core.cone != 0:
        base = core.c - core.h * a
        apex = core.c + core.h * a
        return wp.min(base - ext, apex), wp.max(base + ext, apex)
    ext = ext + core.h * wp.vec3(wp.abs(a[0]), wp.abs(a[1]), wp.abs(a[2]))
    return core.c - ext, core.c + ext


@wp.func
def core_sphere(core: TerrainCylinderCore) -> tuple[wp.vec3, float]:
    radius = wp.sqrt(core.r * core.r + core.h * core.h)
    if core.barrel > 0.0:
        radius = wp.max(radius, _barrel_radius(core, 0.0))
    return core.c, radius


@wp.func
def core_sigma_bound(core: TerrainCylinderCore, n: wp.vec3, p0: wp.vec3) -> float:
    """A lower bound of :func:`core_sigma` in closed form (a cylinder's or cone's: the value itself)."""
    return core_sigma(core, n, p0)


@wp.func
def core_capsule(core: TerrainCylinderCore) -> tuple[wp.vec3, wp.vec3, float]:
    """A capsule containing the core: its axis (a cone's from its base centre to its apex) and radius (a barrel's
    side's at its middle)."""
    radius = core.r
    if core.barrel > 0.0:
        radius = _barrel_radius(core, 0.0)
    return core.c - core.h * core.a, core.c + core.h * core.a, radius


@wp.func
def _cylinder_lowest(core: TerrainCylinderCore, n: wp.vec3) -> wp.vec3:
    """The core's point lowest along ``n``: a cone's lowest generator end, a cylinder's lowest rim point (its cap's
    centre when the axis lies along ``n``), a barrel's :func:`_barrel_lowest`."""
    if core.barrel > 0.0:
        return _barrel_lowest(core, n)
    if core.cone != 0:
        k0, _k1 = _cone_order(core, n)
        return k0
    s = wp.where(wp.dot(n, core.a) > 0.0, -1.0, 1.0)
    cap = core.c + s * core.h * core.a
    perp = -_radial(n, core.a)
    length = wp.length(perp)
    if length < 1.0e-12:
        return cap
    return cap + core.r * perp / length


@wp.func
def core_sigma(core: TerrainCylinderCore, n: wp.vec3, p0: wp.vec3) -> float:
    if core.barrel > 0.0:
        # n . k of the lowest point k = c + z a - rho(z) P n / |P n| (no division: robust along the axis)
        g = wp.dot(n, core.a)
        z = wp.clamp(-core.barrel * g, -core.h, core.h)
        return wp.dot(n, core.c - p0) + z * g - _barrel_radius(core, z) * wp.length(_radial(n, core.a))
    rim = core.r * wp.length(_radial(n, core.a))
    if core.cone != 0:
        apex = wp.dot(n, core.c + core.h * core.a - p0)
        base = wp.dot(n, core.c - core.h * core.a - p0) - rim
        return wp.min(apex, base)
    return wp.dot(n, core.c - p0) - core.h * wp.abs(wp.dot(n, core.a)) - rim


@wp.func
def _rim_slope(core: TerrainCylinderCore, n: wp.vec3, dn: wp.vec3, side: float) -> float:
    """Derivative of ``-r |P n|`` (the rim's term) as ``n`` turns along ``dn``, one-sided at a level cap."""
    pn = _radial(n, core.a)
    length = wp.length(pn)
    if core.r * length <= 0.0:
        return -side * core.r * wp.length(_radial(dn, core.a))
    return -core.r * wp.dot(pn, dn) / length


@wp.func
def core_slope(core: TerrainCylinderCore, n: wp.vec3, dn: wp.vec3, p0: wp.vec3, side: float) -> float:
    if core.barrel > 0.0 and _barrel_side(core, n):
        # a single lowest point on the side: the derivative is its own (on a cap's rim or disk, the cylinder's)
        return wp.dot(dn, _barrel_lowest(core, n) - p0)
    rim = _rim_slope(core, n, dn, side)
    if core.cone != 0:
        apex = core.c + core.h * core.a
        base = core.c - core.h * core.a
        f_apex = wp.dot(n, apex - p0)
        f_base = wp.dot(n, base - p0) - core.r * wp.length(_radial(n, core.a))
        d_apex = wp.dot(dn, apex - p0)
        d_base = wp.dot(dn, base - p0) + rim
        if f_apex < f_base:
            return d_apex
        if f_base < f_apex:
            return d_base
        if side > 0.0:
            return wp.min(d_apex, d_base)
        return wp.max(d_apex, d_base)
    g = wp.dot(n, core.a)
    dg = wp.dot(dn, core.a)
    cap = float(0.0)
    if core.h * wp.abs(g) <= 0.0:
        cap = -side * core.h * wp.abs(dg)
    elif g > 0.0:
        cap = -core.h * dg
    else:
        cap = core.h * dg
    return wp.dot(dn, core.c - p0) + cap + rim


@wp.func
def _manifold_add(count: int, points: _Mat43, depth: wp.vec4, p: wp.vec3, merge: float) -> tuple[int, _Mat43, wp.vec4]:
    """Append a manifold point unless it lies within ``merge`` (squared) of the first one or four are held."""
    first = wp.vec3(points[0, 0], points[0, 1], points[0, 2])
    if count < 4 and wp.length_sq(p - first) > merge:
        points[count, 0] = p[0]
        points[count, 1] = p[1]
        points[count, 2] = p[2]
        count += 1
    return count, points, depth


@wp.func
def _manifold_reach(count: int, points: _Mat43, n: wp.vec3, a: wp.vec3, reach: float) -> tuple[int, _Mat43, wp.vec4]:
    """Keep the manifold points within ``reach`` of the plane ``(n, a)``, in order, with their heights."""
    kept = _Mat43()
    depth = wp.vec4(_HUGE)
    out = int(0)
    for j in range(count):
        p = wp.vec3(points[j, 0], points[j, 1], points[j, 2])
        h = wp.dot(n, p - a)
        if h <= reach:
            kept[out, 0] = p[0]
            kept[out, 1] = p[1]
            kept[out, 2] = p[2]
            depth[out] = h
            out += 1
    return out, kept, depth


@wp.func
def core_sigma_near(core: TerrainCylinderCore, n: wp.vec3, p0: wp.vec3) -> tuple[TerrainCylinderCore, float]:
    """:func:`core_sigma` in a search over nearby normals (closed form: nothing to keep)."""
    return core, core_sigma(core, n, p0)


@wp.func
def core_slope_near(
    core: TerrainCylinderCore, n: wp.vec3, dn: wp.vec3, p0: wp.vec3, side: float
) -> tuple[TerrainCylinderCore, float]:
    """:func:`core_slope` in a search over nearby normals (closed form: nothing to keep)."""
    return core, core_slope(core, n, dn, p0, side)


@wp.func
def core_manifold(core: TerrainCylinderCore, n: wp.vec3, a: wp.vec3, reach: float) -> tuple[int, _Mat43, wp.vec4]:
    """Rule F's plane manifold: Newton's plane-cylinder manifold (the deepest rim point with the fixed tripod within
    22.5 deg of upright, else the rolling points); for a cone, the deepest base rim point with the base tripod within
    22.5 deg of base-down, else the lowest generator's two ends, the lower first; for a barrel, its lowest point with
    the cylinder's tripod within 22.5 deg of upright (its curved side has no rolling points)."""
    points = _Mat43()
    depth = wp.vec4(_HUGE)
    merge = 0.01 * wp.max(core.r, core.h)
    merge = merge * merge
    count = int(1)
    if core.cone != 0:
        g, _p = _cone_ends(core, n)
        if wp.dot(n, core.a) >= _FLAT_COS:
            down = -core.a
            base = core.c - core.h * core.a
            u, v = _disk_basis(down)
            u = u * core.r
            v = v * core.r
            points[0, 0] = g[0]
            points[0, 1] = g[1]
            points[0, 2] = g[2]
            count, points, depth = _manifold_add(count, points, depth, base + u, merge)
            count, points, depth = _manifold_add(count, points, depth, base - 0.5 * u + _TRIPOD_S * v, merge)
            count, points, depth = _manifold_add(count, points, depth, base - 0.5 * u - _TRIPOD_S * v, merge)
        else:
            first, second = _cone_order(core, n)
            points[0, 0] = first[0]
            points[0, 1] = first[1]
            points[0, 2] = first[2]
            count, points, depth = _manifold_add(count, points, depth, second, merge)
        return _manifold_reach(count, points, n, a, reach)
    axis = core.a
    dot_na = wp.dot(n, axis)
    if dot_na > _TIE:
        axis = -axis
        dot_na = -dot_na
    cap = core.c + axis * core.h
    perp_align = -n + axis * dot_na
    la = wp.dot(perp_align, perp_align)
    has_align = la > _ALIGN
    if has_align:
        perp_align = perp_align / wp.sqrt(la)
    flat = -dot_na >= _FLAT_COS
    perp_fixed = wp.vec3(0.0)
    if flat or not has_align:
        ref = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(wp.dot(axis, ref)) > 0.9:
            ref = wp.vec3(0.0, 1.0, 0.0)
        perp_fixed = wp.normalize(ref - axis * wp.dot(axis, ref))
    deepest_perp = perp_fixed
    if has_align:
        deepest_perp = perp_align
    first = cap + deepest_perp * core.r
    if core.barrel > 0.0:
        first = _barrel_lowest(core, n)
    points[0, 0] = first[0]
    points[0, 1] = first[1]
    points[0, 2] = first[2]
    if flat:
        u = perp_fixed * core.r
        v = wp.cross(axis, perp_fixed) * core.r
        count, points, depth = _manifold_add(count, points, depth, cap + u, merge)
        count, points, depth = _manifold_add(count, points, depth, cap - 0.5 * u + _TRIPOD_S * v, merge)
        count, points, depth = _manifold_add(count, points, depth, cap - 0.5 * u - _TRIPOD_S * v, merge)
    elif core.barrel <= 0.0:
        perp_roll = perp_fixed
        if has_align:
            perp_roll = perp_align
        u = perp_roll * core.r
        v = wp.cross(axis, perp_roll) * core.r
        count, points, depth = _manifold_add(count, points, depth, core.c - axis * core.h + u, merge)
        # The cap's side points at +-v tie by construction (v is normal to n and the axis), so the float64 law's
        # choice, the lower one by more than its tie band, is always +v; float32 rounding of the two depths
        # (about 1e-7 of the distance to the plane) would otherwise choose.
        count, points, depth = _manifold_add(count, points, depth, cap + v, merge)
    return _manifold_reach(count, points, n, a, reach)


@wp.func
def core_flat_crossing(
    core: TerrainCylinderCore, n: wp.vec3, band: float, v0: wp.vec3, e1: wp.vec3, e2: wp.vec3, tol: float
) -> tuple[bool, TerrainCrossing]:
    """Rule E, interior ``theta*``: the flat support feature along ``n`` within ``band`` (a level cap or base disk,
    a generator, or the lowest point; a barrel's curved side has no generator) crossed with the crease's line."""
    tilt = wp.length(_radial(n, core.a))
    if core.cone != 0:
        if wp.dot(n, core.a) > 0.0 and 2.0 * core.r * tilt <= band:
            return _disk_crossing(core.c - core.h * core.a, core.a, core.r, v0, e1, e2, tol, 0, 0.0)
        k0, k1 = _cone_order(core, n)
        if wp.dot(n, k1 - k0) <= band:
            return _two_point_crossing(k0, k1, v0, e1, e2, tol, 0, 0.0)
        return _point_crossing(k0, v0, e1, e2, tol)
    s = wp.where(wp.dot(n, core.a) > 0.0, -1.0, 1.0)
    if 2.0 * core.r * tilt <= band:
        return _disk_crossing(core.c + s * core.h * core.a, core.a, core.r, v0, e1, e2, tol, 0, 0.0)
    k0 = _cylinder_lowest(core, n)
    k1 = k0 - 2.0 * s * core.h * core.a
    if core.barrel <= 0.0 and wp.dot(n, k1 - k0) <= band:
        return _two_point_crossing(k0, k1, v0, e1, e2, tol, 0, 0.0)
    return _point_crossing(k0, v0, e1, e2, tol)


@wp.func
def core_reference_crossing(
    core: TerrainCylinderCore, n: wp.vec3, v0: wp.vec3, e1: wp.vec3, e2: wp.vec3, tol: float, straddle_tol: float
) -> tuple[bool, TerrainCrossing]:
    """Rule E at an arc end (G1): the near cap (base) within 22.5 deg of level, else the lowest generator, crossed
    with the crease's line when it straddles it; a barrel's curved side offers its lowest point alone, which straddles
    no line."""
    if core.cone != 0:
        if wp.dot(n, core.a) >= _FLAT_COS:
            return _disk_crossing(core.c - core.h * core.a, core.a, core.r, v0, e1, e2, tol, 1, straddle_tol)
        if wp.length(_radial(n, core.a)) < 1.0e-12:
            return False, _crossing()
        g, p = _cone_ends(core, n)
        return _two_point_crossing(g, p, v0, e1, e2, tol, 1, straddle_tol)
    axis = core.a
    if wp.dot(n, core.a) >= 0.0:
        axis = -core.a
    cap = core.c + core.h * axis
    if -wp.dot(n, axis) >= _FLAT_COS:
        return _disk_crossing(cap, axis, core.r, v0, e1, e2, tol, 1, straddle_tol)
    perp = -n + axis * wp.dot(n, axis)
    length = wp.length(perp)
    if length < 1.0e-12:
        return _disk_crossing(cap, axis, core.r, v0, e1, e2, tol, 1, straddle_tol)
    if core.barrel > 0.0:
        return False, _crossing()
    perp = perp / length
    pa = core.c - axis * core.h + perp * core.r
    pb = core.c + axis * core.h + perp * core.r
    return _two_point_crossing(pa, pb, v0, e1, e2, tol, 1, straddle_tol)


@wp.func
def core_lift(core: TerrainCylinderCore, u: wp.vec3, n: wp.vec3, tol: float) -> tuple[bool, wp.vec3, float]:
    ax = core.a
    if core.cone != 0:
        apex = core.c + core.h * ax
        w = u - apex
        dz = wp.dot(n, ax)
        wz = wp.dot(w, ax)
        t_lo = float(-_HUGE)
        t_hi = float(_HUGE)
        ok, t_lo, t_hi = _slab(-wz, dz, tol, t_lo, t_hi)
        if not ok:
            return False, u, 0.0
        ok, t_lo, t_hi = _slab(2.0 * core.h + wz, -dz, tol, t_lo, t_hi)
        if not ok:
            return False, u, 0.0
        k2 = core.r / (2.0 * core.h)
        k2 = k2 * k2
        wr = w - wz * ax
        nr = n - dz * ax
        qa = wp.dot(nr, nr) - k2 * dz * dz
        qb = 2.0 * (wp.dot(wr, nr) - k2 * wz * dz)
        qc = wp.dot(wr, wr) - k2 * wz * wz
        if wp.abs(qa) < 1.0e-12:
            if wp.abs(qb) < 1.0e-12:
                if qc > 2.0 * core.r * tol:
                    return False, u, 0.0
            elif qb > 0.0:
                t_hi = wp.min(t_hi, -qc / qb)
            else:
                t_lo = wp.max(t_lo, -qc / qb)
        elif qa > 0.0:
            ok, t_lo, t_hi = _quadratic_entry(qa, qb, qc, t_lo, t_hi, core.r, tol)
            if not ok:
                return False, u, 0.0
        else:
            disc = qb * qb - 4.0 * qa * qc
            if disc >= 0.0:
                sq = wp.sqrt(disc)
                r1 = (-qb - sq) / (2.0 * qa)
                r2 = (-qb + sq) / (2.0 * qa)
                lo_root = wp.min(r1, r2)
                hi_root = wp.max(r1, r2)
                if dz > 0.0:
                    t_hi = wp.min(t_hi, lo_root)
                else:
                    t_lo = wp.max(t_lo, hi_root)
        if t_lo > t_hi + tol or t_lo <= -_HUGE:
            return False, u, 0.0
        return True, u + t_lo * n, t_lo
    w = u - core.c
    dn = wp.dot(n, ax)
    dw = wp.dot(w, ax)
    t_lo = float(-_HUGE)
    t_hi = float(_HUGE)
    ok, t_lo, t_hi = _slab(core.h - dw, dn, tol, t_lo, t_hi)
    if not ok:
        return False, u, 0.0
    ok, t_lo, t_hi = _slab(core.h + dw, -dn, tol, t_lo, t_hi)
    if not ok:
        return False, u, 0.0
    radius = core.r
    if core.barrel > 0.0:
        if t_lo > -_HUGE and t_hi < _HUGE:
            return _barrel_lift(core, u, n, tol, t_lo, t_hi)
        # parallel to the caps: the side's circle at the line's height
        radius = _barrel_radius(core, wp.clamp(dw, -core.h, core.h))
    wr = w - dw * ax
    nr = n - dn * ax
    qa = wp.dot(nr, nr)
    qb = 2.0 * wp.dot(wr, nr)
    qc = wp.dot(wr, wr) - radius * radius
    if qa < 1.0e-12:
        if qc > 2.0 * radius * tol:
            return False, u, 0.0
    else:
        ok, t_lo, t_hi = _quadratic_entry(qa, qb, qc, t_lo, t_hi, radius, tol)
        if not ok:
            return False, u, 0.0
    if t_lo > t_hi + tol or t_lo <= -_HUGE:
        return False, u, 0.0
    return True, u + t_lo * n, t_lo


@wp.func
def _barrel_lift(
    core: TerrainCylinderCore, u: wp.vec3, n: wp.vec3, tol: float, t_lo: float, t_hi: float
) -> tuple[bool, wp.vec3, float]:
    """Where line ``u + t n`` enters a barrel between its caps' planes (``[t_lo, t_hi]``): its side's excess along the
    line (:func:`_barrel_excess`) is convex there, so the entry is its first zero, before its least value (a golden
    section) when the line starts outside; a line whose least excess exceeds ``tol`` misses."""
    w = u - core.c
    if t_lo > t_hi + tol:
        return False, u, 0.0
    if _barrel_excess(core, w + t_lo * n) > 0.0:
        a = t_lo
        b = wp.max(t_hi, t_lo)
        x1 = b - _GOLDEN * (b - a)
        x2 = a + _GOLDEN * (b - a)
        f1 = _barrel_excess(core, w + x1 * n)
        f2 = _barrel_excess(core, w + x2 * n)
        for _step in range(_BARREL_STEPS):
            if f1 <= f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - _GOLDEN * (b - a)
                f1 = _barrel_excess(core, w + x1 * n)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + _GOLDEN * (b - a)
                f2 = _barrel_excess(core, w + x2 * n)
        least = 0.5 * (a + b)
        if _barrel_excess(core, w + least * n) > tol:
            return False, u, 0.0
        lo = t_lo
        hi = least
        for _step in range(_BARREL_STEPS):
            mid = 0.5 * (lo + hi)
            if _barrel_excess(core, w + mid * n) > 0.0:
                lo = mid
            else:
                hi = mid
        t_lo = hi
    return True, u + t_lo * n, t_lo


@wp.func
def _closest_on_triangle_2d(p: wp.vec2, a: wp.vec2, b: wp.vec2, c: wp.vec2) -> wp.vec2:
    """The point of triangle ``abc`` nearest ``p`` (Voronoi regions, 2-D)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + d1 / (d1 - d3) * ab
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
    den = 1.0 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


@wp.func
def _barrel_nearest(core: TerrainCylinderCore, v: wp.vec3) -> tuple[bool, wp.vec3, wp.vec3, float]:
    """:func:`core_nearest` of a barrel, in the meridian half-plane of ``v`` (radial ``x``, axial ``z``): outside, the
    nearest of the caps' projections and the side arc's (the circle of radius ``R`` about ``(x0, 0)``, ``x0 = r -
    sqrt(R^2 - h^2)``, where it lies between the caps); inside, the nearer of the caps and the arc."""
    ax = core.a
    w = v - core.c
    z = wp.dot(w, ax)
    rad = w - z * ax
    x = wp.length(rad)
    radial = wp.vec3(0.0)
    if x > 1.0e-12:
        radial = rad / x
    else:
        radial, _w = _disk_basis(ax)
    x0 = core.r - wp.sqrt(wp.max(core.barrel * core.barrel - core.h * core.h, 0.0))
    ox = x - x0
    od = wp.sqrt(ox * ox + z * z)
    # the side arc's point along the ray from its circle's centre, when on the side (between the caps, on the far
    # side of the centre: from a point nearer the axis the side's nearest point is a rim, no nearer than a cap)
    arc = od > 0.0 and ox >= 0.0 and wp.abs(core.barrel * z) <= core.h * od
    ax2 = x0 + core.barrel * ox / wp.max(od, 1.0e-30)
    az2 = core.barrel * z / wp.max(od, 1.0e-30)
    if wp.abs(z) <= core.h and x <= _barrel_radius(core, z):
        # inside: the cap or the side, whichever is nearer
        d_cap = core.h - wp.abs(z)
        if arc and core.barrel - od < d_cap:
            f = (ox * radial + z * ax) / od
            d = core.barrel - od
            return True, -f, v + d * f, -d
        f = ax * wp.where(z >= 0.0, 1.0, -1.0)
        return True, -f, v + d_cap * f, -d_cap
    # outside: the caps' nearest points and the arc's
    sz = wp.where(z >= 0.0, 1.0, -1.0)
    bx = wp.min(x, core.r)
    bz = sz * core.h
    best = (x - bx) * (x - bx) + (z - bz) * (z - bz)
    if arc:
        d2 = (x - ax2) * (x - ax2) + (z - az2) * (z - az2)
        if d2 < best:
            best = d2
            bx = ax2
            bz = az2
    k = core.c + bz * ax + bx * radial
    dist = wp.length(k - v)
    if dist <= 0.0:
        return False, ax, k, 0.0
    return True, (k - v) / dist, k, dist


@wp.func
def core_nearest(core: TerrainCylinderCore, v: wp.vec3) -> tuple[bool, wp.vec3, wp.vec3, float]:
    ax = core.a
    if core.barrel > 0.0:
        return _barrel_nearest(core, v)
    if core.cone != 0:
        apex = core.c + core.h * ax
        w = v - apex
        z = wp.dot(w, ax)
        rad = w - z * ax
        lr = wp.length(rad)
        radial = wp.vec3(0.0)
        if lr > 1.0e-12:
            radial = rad / lr
        else:
            radial, _w = _disk_basis(ax)
        inside = z >= -2.0 * core.h and z <= 0.0 and lr <= core.r * (-z) / (2.0 * core.h)
        if inside:
            d_base = z + 2.0 * core.h
            slant = wp.sqrt(core.r * core.r + 4.0 * core.h * core.h)
            d_side = -(2.0 * core.h * lr + core.r * z) / slant
            if d_base <= d_side:
                return True, ax, v - d_base * ax, -d_base
            f = (2.0 * core.h * radial + core.r * ax) / slant
            return True, -f, v + d_side * f, -d_side
        q = _closest_on_triangle_2d(
            wp.vec2(lr, z), wp.vec2(0.0, 0.0), wp.vec2(core.r, -2.0 * core.h), wp.vec2(-core.r, -2.0 * core.h)
        )
        k = apex + q[1] * ax + q[0] * radial
        dist = wp.length(k - v)
        if dist <= 0.0:
            return False, ax, k, 0.0
        return True, (k - v) / dist, k, dist
    w = v - core.c
    zz = wp.dot(w, ax)
    rad = w - zz * ax
    lr = wp.length(rad)
    if wp.abs(zz) > core.h or lr > core.r:
        z = wp.clamp(zz, -core.h, core.h)
        if lr > core.r:
            rad = rad * (core.r / lr)
        k = core.c + z * ax + rad
        dist = wp.length(k - v)
        if dist <= 0.0:
            return False, ax, k, 0.0
        return True, (k - v) / dist, k, dist
    d_cap = core.h - wp.abs(zz)
    d_side = core.r - lr
    if d_cap <= d_side or lr < 1.0e-12:
        f = ax * wp.where(zz >= 0.0, 1.0, -1.0)
        return True, -f, v + d_cap * f, -d_cap
    f = rad / lr
    return True, -f, v + d_side * f, -d_side


@wp.func
def core_recovery_count(core: TerrainCylinderCore) -> int:
    if core.cone != 0:
        return _RING + 1
    if core.barrel > 0.0:
        return 3 * _RING
    return 2 * _RING


@wp.func
def core_recovery_point(core: TerrainCylinderCore, i: int) -> wp.vec3:
    """Rule R's boundary point ``i``: 16 rim points per cap (the top cap first), then a barrel's 16 points around its
    side's middle; a cone's 16 base rim points, then its apex."""
    u, w = _disk_basis(core.a)
    j = i % _RING
    ring = core.r * (_RING_ANGLES[j, 0] * u + _RING_ANGLES[j, 1] * w)
    if core.cone != 0:
        if i >= _RING:
            return core.c + core.h * core.a
        return core.c - core.h * core.a + ring
    if i < _RING:
        return core.c + core.h * core.a + ring
    if i >= 2 * _RING:
        return core.c + (_barrel_radius(core, 0.0) / core.r) * ring
    return core.c - core.h * core.a + ring


# ----------------------------------------------------------------------------------------------------------------
# convex hulls and ellipsoids
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _hull_point(core: TerrainHullCore, i: int) -> wp.vec3:
    """Point ``i`` (global index) of the hull, scaled, in the core's frame."""
    return core.c + core.rot @ wp.cw_mul(core.table.points[i], core.s)


@wp.func
def _hull_face_normal(core: TerrainHullCore, f: int) -> wp.vec3:
    """Outward unit normal of face ``f`` (global index), scaled, in the core's frame."""
    p = core.table.planes[f]
    return core.rot @ wp.normalize(wp.cw_div(wp.vec3(p[0], p[1], p[2]), core.s))


@wp.func
def _hull_deepest(core: TerrainHullCore, n: wp.vec3) -> int:
    """The point lowest along ``n`` (global index; the first in mesh order within the tie band)."""
    dir_local = wp.cw_mul(wp.transpose(core.rot) @ n, core.s)
    low = float(_HUGE)
    for i in range(core.point_begin, core.point_begin + core.point_count):
        low = wp.min(low, wp.dot(core.table.points[i], dir_local))
    band = _TIE * core.radius
    for i in range(core.point_begin, core.point_begin + core.point_count):
        if wp.dot(core.table.points[i], dir_local) <= low + band:
            return i
    return core.point_begin


@wp.func
def _ellipsoid_support(core: TerrainHullCore, n: wp.vec3) -> wp.vec3:
    m = wp.transpose(core.rot) @ n
    sm = wp.cw_mul(core.s, m)
    y = -wp.cw_mul(core.s, sm) / wp.length(sm)
    return core.c + core.rot @ y


@wp.func
def core_bounds(core: TerrainHullCore) -> tuple[wp.vec3, wp.vec3]:
    if core.ellipsoid != 0:
        r = core.rot
        s = core.s
        a0 = wp.vec3(r[0, 0] * s[0], r[0, 1] * s[1], r[0, 2] * s[2])
        a1 = wp.vec3(r[1, 0] * s[0], r[1, 1] * s[1], r[1, 2] * s[2])
        a2 = wp.vec3(r[2, 0] * s[0], r[2, 1] * s[1], r[2, 2] * s[2])
        ext = wp.vec3(wp.length(a0), wp.length(a1), wp.length(a2))
        return core.c - ext, core.c + ext
    lo = wp.vec3(_HUGE)
    hi = wp.vec3(-_HUGE)
    for i in range(core.point_begin, core.point_begin + core.point_count):
        p = _hull_point(core, i)
        lo = wp.min(lo, p)
        hi = wp.max(hi, p)
    return lo, hi


@wp.func
def core_sphere(core: TerrainHullCore) -> tuple[wp.vec3, float]:
    return core.c, core.radius


@wp.func
def core_capsule(core: TerrainHullCore) -> tuple[wp.vec3, wp.vec3, float]:
    """A capsule containing the core: its bounding box's longest axis through the box's centre, the radius the box's
    cross-section's half diagonal."""
    h = core.box_h
    k = int(0)
    if h[1] > h[k]:
        k = 1
    if h[2] > h[k]:
        k = 2
    centre = core.c + core.rot @ core.box_c
    axis = wp.vec3(core.rot[0, k], core.rot[1, k], core.rot[2, k]) * h[k]
    radius = wp.sqrt(wp.max(wp.dot(h, h) - h[k] * h[k], 0.0))
    return centre - axis, centre + axis, radius


@wp.func
def core_sigma_bound(core: TerrainHullCore, n: wp.vec3, p0: wp.vec3) -> float:
    """A lower bound of :func:`core_sigma` in closed form: an ellipsoid's value itself, a hull's its bounding box's
    (``core.box_c``, ``core.box_h``), which contains it, so that no support search runs."""
    if core.ellipsoid != 0:
        return core_sigma(core, n, p0)
    m = wp.transpose(core.rot) @ n
    h = core.box_h
    return wp.dot(n, core.c + core.rot @ core.box_c - p0) - (
        h[0] * wp.abs(m[0]) + h[1] * wp.abs(m[1]) + h[2] * wp.abs(m[2])
    )


@wp.func
def _hull_support(core: TerrainHullCore, dir_local: wp.vec3) -> tuple[int, float]:
    """A hull's point lowest along ``dir_local`` (scaled local frame) and its value: a descent over the hull's point
    graph from ``core.start``, each step to the lowest neighbour while one is lower (a point of a convex hull with no
    lower neighbour is the lowest; terrain normals keep the descent within a few steps of the start)."""
    best = core.start
    low = wp.dot(core.table.points[best], dir_local)
    moved = bool(True)
    while moved:
        moved = bool(False)
        current = best
        for j in range(core.table.neighbor_begin[current], core.table.neighbor_begin[current + 1]):
            u = core.table.neighbors[j]
            value = wp.dot(core.table.points[u], dir_local)
            if value < low:
                low = value
                best = u
                moved = bool(True)
    return best, low


@wp.func
def core_sigma(core: TerrainHullCore, n: wp.vec3, p0: wp.vec3) -> float:
    if core.ellipsoid != 0:
        sm = wp.cw_mul(core.s, wp.transpose(core.rot) @ n)
        return wp.dot(n, core.c - p0) - wp.length(sm)
    _best, low = _hull_support(core, wp.cw_mul(wp.transpose(core.rot) @ n, core.s))
    return wp.dot(n, core.c - p0) + low


@wp.func
def _hull_slope(core: TerrainHullCore, n: wp.vec3, dn: wp.vec3, p0: wp.vec3, side: float) -> tuple[int, float]:
    """A hull's :func:`core_slope` and its lowest point along ``n`` (global index)."""
    dir_local = wp.cw_mul(wp.transpose(core.rot) @ n, core.s)
    ddir_local = wp.cw_mul(wp.transpose(core.rot) @ dn, core.s)
    lowest, low = _hull_support(core, dir_local)
    # the lowest point and its neighbours tied with it (exact ties: a face along n)
    best = wp.dot(core.table.points[lowest], ddir_local)
    for j in range(core.table.neighbor_begin[lowest], core.table.neighbor_begin[lowest + 1]):
        p = core.table.points[core.table.neighbors[j]]
        if wp.dot(p, dir_local) <= low:
            d = wp.dot(p, ddir_local)
            if side > 0.0:
                best = wp.min(best, d)
            else:
                best = wp.max(best, d)
    return lowest, wp.dot(dn, core.c - p0) + best


@wp.func
def core_slope(core: TerrainHullCore, n: wp.vec3, dn: wp.vec3, p0: wp.vec3, side: float) -> float:
    if core.ellipsoid != 0:
        m = wp.transpose(core.rot) @ n
        dm = wp.transpose(core.rot) @ dn
        sm = wp.cw_mul(core.s, m)
        sdm = wp.cw_mul(core.s, dm)
        return wp.dot(dn, core.c - p0) - wp.dot(sm, sdm) / wp.length(sm)
    _lowest, slope = _hull_slope(core, n, dn, p0, side)
    return slope


@wp.func
def core_sigma_near(core: TerrainHullCore, n: wp.vec3, p0: wp.vec3) -> tuple[TerrainHullCore, float]:
    """:func:`core_sigma` in a search over nearby normals: a hull's support search starts at the point the last one
    ended on (kept as the core's start), a step or none away for a nearby normal. The value is a search's from any
    start: a descent ends at a point without a lower neighbour, on a convex hull the lowest (the points of a face
    tied along the normal all neighbour each other)."""
    if core.ellipsoid != 0:
        return core, core_sigma(core, n, p0)
    best, low = _hull_support(core, wp.cw_mul(wp.transpose(core.rot) @ n, core.s))
    core.start = best
    return core, wp.dot(n, core.c - p0) + low


@wp.func
def core_slope_near(
    core: TerrainHullCore, n: wp.vec3, dn: wp.vec3, p0: wp.vec3, side: float
) -> tuple[TerrainHullCore, float]:
    """:func:`core_slope` in a search over nearby normals (:func:`core_sigma_near`)."""
    if core.ellipsoid != 0:
        return core, core_slope(core, n, dn, p0, side)
    lowest, slope = _hull_slope(core, n, dn, p0, side)
    core.start = lowest
    return core, slope


@wp.func
def _hull_vertex_face(core: TerrainHullCore, vertex: int, n: wp.vec3) -> int:
    """The face through point ``vertex`` most opposed to ``n`` (global indices; of the faces within the tie band of
    the most opposed, the first in face order): the Gauss map's cell of the vertex, read from its incident faces. -1
    when the point lies on no face (inside the hull)."""
    best = float(-_HUGE)
    for j in range(core.table.vertex_face_begin[vertex], core.table.vertex_face_begin[vertex + 1]):
        best = wp.max(best, -wp.dot(_hull_face_normal(core, core.table.vertex_faces[j]), n))
    for j in range(core.table.vertex_face_begin[vertex], core.table.vertex_face_begin[vertex + 1]):
        f = core.table.vertex_faces[j]
        if -wp.dot(_hull_face_normal(core, f), n) >= best - _TIE:
            return f
    return -1


@wp.func
def core_manifold(core: TerrainHullCore, n: wp.vec3, a: wp.vec3, reach: float) -> tuple[int, _Mat43, wp.vec4]:
    """Rule F's plane manifold: a hull's reference face (the face through its deepest vertex most opposed to ``n``,
    DESIGN 1.2, the face rule E's G1 crossings use) with that vertex, within ``reach``, reduced to four by area
    (deepest, farthest, largest triangle, largest quadrilateral) and ordered by depth (then index); an ellipsoid's
    support point."""
    points = _Mat43()
    depth = wp.vec4(_HUGE)
    if core.ellipsoid != 0:
        k = _ellipsoid_support(core, n)
        h = wp.dot(n, k - a)
        if h > reach:
            return 0, points, depth
        points[0, 0] = k[0]
        points[0, 1] = k[1]
        points[0, 2] = k[2]
        depth[0] = h
        return 1, points, depth
    deepest = _hull_deepest(core, n)
    face = _hull_vertex_face(core, deepest, n)
    # the candidates in index order: the face's vertices and the deepest vertex, within reach
    cand = _FaceList()
    count = int(0)
    inserted = bool(False)
    begin = int(0)
    end = int(0)
    if face >= 0:
        begin = core.table.face_begin[face]
        end = wp.min(core.table.face_begin[face + 1], begin + _FACE_CAP)
    for j in range(begin, end):
        i = core.table.face_vertices[j]
        if not inserted and deepest <= i:
            if deepest < i and wp.dot(n, _hull_point(core, deepest) - a) <= reach:
                cand[count] = deepest
                count += 1
            inserted = True
        if wp.dot(n, _hull_point(core, i) - a) <= reach:
            cand[count] = i
            count += 1
    if not inserted and wp.dot(n, _hull_point(core, deepest) - a) <= reach:
        cand[count] = deepest
        count += 1
    if count == 0:
        return 0, points, depth
    keep = _Vec4i(-1, -1, -1, -1)
    kept = int(0)
    if count <= 4:
        for j in range(count):
            keep[j] = cand[j]
        kept = count
    else:
        # reduce4 of the float64 law, in tangent positions on the plane normal to n; of candidates within the tie
        # band of the best, the first in index order
        band = _TIE * core.radius
        low = float(_HUGE)
        for j in range(count):
            low = wp.min(low, wp.dot(n, _hull_point(core, cand[j]) - a))
        i0 = int(0)
        for j in range(count):
            if wp.dot(n, _hull_point(core, cand[j]) - a) <= low + band:
                i0 = j
                break
        p0 = _hull_point(core, cand[i0])
        p0 = p0 - wp.dot(p0, n) * n
        far = float(-1.0)
        for j in range(count):
            pj = _hull_point(core, cand[j])
            far = wp.max(far, wp.length(pj - wp.dot(pj, n) * n - p0))
        i1 = int(0)
        for j in range(count):
            pj = _hull_point(core, cand[j])
            if wp.length(pj - wp.dot(pj, n) * n - p0) >= far - band:
                i1 = j
                break
        p1 = _hull_point(core, cand[i1])
        p1 = p1 - wp.dot(p1, n) * n
        span = wp.max(wp.length(p1 - p0), 1.0e-12)
        largest = float(-1.0)
        for j in range(count):
            pj = _hull_point(core, cand[j])
            largest = wp.max(largest, wp.abs(wp.dot(wp.cross(p1 - p0, pj - wp.dot(pj, n) * n - p0), n)))
        i2 = int(0)
        for j in range(count):
            pj = _hull_point(core, cand[j])
            if wp.abs(wp.dot(wp.cross(p1 - p0, pj - wp.dot(pj, n) * n - p0), n)) >= largest - band * span:
                i2 = j
                break
        p2 = _hull_point(core, cand[i2])
        p2 = p2 - wp.dot(p2, n) * n
        gain_best = float(0.0)
        for pass_ in range(2):
            i3 = int(-1)
            for j in range(count):
                if j == i0 or j == i1 or j == i2:
                    continue
                pj = _hull_point(core, cand[j])
                pj = pj - wp.dot(pj, n) * n
                gain = float(0.0)
                for edge in range(3):
                    pa = p0
                    pb = p1
                    pc = p2
                    if edge == 1:
                        pa = p1
                        pb = p2
                        pc = p0
                    elif edge == 2:
                        pa = p2
                        pb = p0
                        pc = p1
                    side_c = wp.dot(wp.cross(pb - pa, pc - pa), n)
                    side_j = wp.dot(wp.cross(pb - pa, pj - pa), n)
                    if side_c * side_j < 0.0:
                        gain = wp.max(gain, wp.abs(side_j))
                if pass_ == 0:
                    gain_best = wp.max(gain_best, gain)
                elif gain_best > 1.0e-15 and gain >= wp.max(gain_best - band * span, 1.0e-15) and i3 < 0:
                    i3 = j
        keep[0] = cand[i0]
        kept = 1
        if i1 != i0:
            keep[kept] = cand[i1]
            kept += 1
        if i2 != i0 and i2 != i1:
            keep[kept] = cand[i2]
            kept += 1
        if i3 >= 0:
            keep[kept] = cand[i3]
            kept += 1
        # back to index order before ordering by depth
        for j in range(kept):
            for l in range(j + 1, kept):
                if keep[l] < keep[j]:
                    tmp = keep[j]
                    keep[j] = keep[l]
                    keep[l] = tmp
    # deepest first: a point passes an earlier one only when deeper by more than the tie band
    band_depth = _TIE * core.radius
    for j in range(1, kept):
        moving = keep[j]
        hm = wp.dot(n, _hull_point(core, moving) - a)
        pos = j
        while pos > 0 and wp.dot(n, _hull_point(core, keep[pos - 1]) - a) > hm + band_depth:
            keep[pos] = keep[pos - 1]
            pos -= 1
        keep[pos] = moving
    for j in range(kept):
        p = _hull_point(core, keep[j])
        points[j, 0] = p[0]
        points[j, 1] = p[1]
        points[j, 2] = p[2]
        depth[j] = wp.dot(n, p - a)
    return kept, points, depth


@wp.func
def core_flat_crossing(
    core: TerrainHullCore, n: wp.vec3, band: float, v0: wp.vec3, e1: wp.vec3, e2: wp.vec3, tol: float
) -> tuple[bool, TerrainCrossing]:
    """Rule E, interior ``theta*``: the hull's points within ``band`` of the lowest along ``n`` (a face, edge or
    vertex), or the ellipsoid's support point, crossed with the crease's line."""
    if core.ellipsoid != 0:
        return _point_crossing(_ellipsoid_support(core, n), v0, e1, e2, tol)
    dir_local = wp.cw_mul(wp.transpose(core.rot) @ n, core.s)
    low = float(_HUGE)
    for i in range(core.point_begin, core.point_begin + core.point_count):
        low = wp.min(low, wp.dot(core.table.points[i], dir_local))
    crossing = _crossing()
    # The points within the band, in mesh order: the feature's points and segments are theirs alone, so a list of them
    # takes the pairs below in the same order with one pass over the hull instead of one per point.
    members = _BandPoints()
    count = int(0)
    for i in range(core.point_begin, core.point_begin + core.point_count):
        if wp.dot(core.table.points[i], dir_local) <= low + band:
            if count < _BAND_POINTS:
                members[count] = i
            count += 1
    if count <= _BAND_POINTS:
        for j in range(count):
            pi = _hull_point(core, members[j])
            xi = wp.dot(pi - v0, e1)
            yi = wp.dot(pi - v0, e2)
            crossing = _cross_point(crossing, xi, yi, pi, tol)
            for l in range(j + 1, count):
                pl = _hull_point(core, members[l])
                crossing = _cross_segment(crossing, xi, yi, pi, wp.dot(pl - v0, e1), wp.dot(pl - v0, e2), pl, tol)
        return crossing.lo <= crossing.hi, crossing
    for i in range(core.point_begin, core.point_begin + core.point_count):
        if wp.dot(core.table.points[i], dir_local) > low + band:
            continue
        pi = _hull_point(core, i)
        xi = wp.dot(pi - v0, e1)
        yi = wp.dot(pi - v0, e2)
        crossing = _cross_point(crossing, xi, yi, pi, tol)
        for j in range(i + 1, core.point_begin + core.point_count):
            if wp.dot(core.table.points[j], dir_local) > low + band:
                continue
            pj = _hull_point(core, j)
            crossing = _cross_segment(crossing, xi, yi, pi, wp.dot(pj - v0, e1), wp.dot(pj - v0, e2), pj, tol)
    return crossing.lo <= crossing.hi, crossing


@wp.func
def core_reference_crossing(
    core: TerrainHullCore, n: wp.vec3, v0: wp.vec3, e1: wp.vec3, e2: wp.vec3, tol: float, straddle_tol: float
) -> tuple[bool, TerrainCrossing]:
    """Rule E at an arc end (G1): the face through the deepest vertex most opposed to ``n`` (the Gauss map's cell of
    that vertex), crossed with the crease's line when it straddles it; an ellipsoid's single point never does."""
    if core.ellipsoid != 0:
        return False, _crossing()
    face = _hull_vertex_face(core, _hull_deepest(core, n), n)
    if face < 0:
        return False, _crossing()
    begin = core.table.face_begin[face]
    end = core.table.face_begin[face + 1]
    ymin = float(_HUGE)
    ymax = float(-_HUGE)
    for j in range(begin, end):
        y = wp.dot(_hull_point(core, core.table.face_vertices[j]) - v0, e2)
        ymin = wp.min(ymin, y)
        ymax = wp.max(ymax, y)
    crossing = _crossing()
    if not (ymin < -straddle_tol and ymax > straddle_tol):
        return False, crossing
    for j in range(begin, end):
        pi = _hull_point(core, core.table.face_vertices[j])
        xi = wp.dot(pi - v0, e1)
        yi = wp.dot(pi - v0, e2)
        crossing = _cross_point(crossing, xi, yi, pi, tol)
        for l in range(j + 1, end):
            pl = _hull_point(core, core.table.face_vertices[l])
            crossing = _cross_segment(crossing, xi, yi, pi, wp.dot(pl - v0, e1), wp.dot(pl - v0, e2), pl, tol)
    return crossing.lo <= crossing.hi, crossing


@wp.func
def core_lift(core: TerrainHullCore, u: wp.vec3, n: wp.vec3, tol: float) -> tuple[bool, wp.vec3, float]:
    """Where the line enters the core: a hull's slabs in its unscaled frame (where the line keeps its parameter);
    an ellipsoid's quadric."""
    rt = wp.transpose(core.rot)
    us = wp.cw_div(rt @ (u - core.c), core.s)
    ns = wp.cw_div(rt @ n, core.s)
    if core.ellipsoid != 0:
        qa = wp.dot(ns, ns)
        qb = 2.0 * wp.dot(us, ns)
        qc = wp.dot(us, us) - 1.0
        smax = wp.max(core.s[0], wp.max(core.s[1], core.s[2]))
        ok, t_lo, t_hi = _quadratic_entry(qa, qb, qc, -_HUGE, _HUGE, 1.0, tol / smax)
        if not ok or t_lo > t_hi + tol:
            return False, u, 0.0
        return True, u + t_lo * n, t_lo
    t_lo = float(-_HUGE)
    t_hi = float(_HUGE)
    for f in range(core.face_begin, core.face_begin + core.face_count):
        p = core.table.planes[f]
        nf = wp.vec3(p[0], p[1], p[2])
        # a distance tol in the scaled frame is |nf / s| tol along nf in the unscaled one
        ok, t_lo, t_hi = _slab(
            p[3] - wp.dot(nf, us), wp.dot(nf, ns), tol * wp.length(wp.cw_div(nf, core.s)), t_lo, t_hi
        )
        if not ok:
            return False, u, 0.0
    if t_lo > t_hi + tol or t_lo <= -_HUGE:
        return False, u, 0.0
    return True, u + t_lo * n, t_lo


@wp.func
def _ellipsoid_nearest(core: TerrainHullCore, v: wp.vec3) -> tuple[bool, wp.vec3, wp.vec3, float]:
    """The ellipsoid's nearest boundary point to ``v`` from the secular equation, bisected a fixed number of steps;
    with ``v`` inside, the minimum translation."""
    y = wp.transpose(core.rot) @ (v - core.c)
    s = core.s
    s2 = wp.cw_mul(s, s)
    q = wp.cw_div(y, s)
    inside = wp.dot(q, q) <= 1.0
    smin2 = wp.min(s2[0], wp.min(s2[1], s2[2]))
    lo = float(0.0)
    hi = wp.length(y) * wp.max(s[0], wp.max(s[1], s[2]))
    if inside:
        lo = -smin2
        hi = 0.0
    for _step in range(_ELLIPSOID_STEPS):
        t = 0.5 * (lo + hi)
        g = wp.cw_div(wp.cw_mul(s, y), s2 + wp.vec3(t))
        if wp.dot(g, g) > 1.0:
            lo = t
        else:
            hi = t
    t = 0.5 * (lo + hi)
    x = wp.cw_div(wp.cw_mul(s2, y), s2 + wp.vec3(t))
    xq = wp.cw_div(x, s)
    on_boundary = wp.abs(wp.dot(xq, xq) - 1.0) <= 1.0e-3
    # an interior point on the smallest axis's plane: the stationary points off that plane may be nearer
    if inside:
        k = int(0)
        if s2[1] < s2[k]:
            k = 1
        if s2[2] < s2[k]:
            k = 2
        alt = wp.vec3(0.0)
        rest = float(1.0)
        valid = bool(True)
        for j in range(3):
            if j != k:
                gap = s2[j] - s2[k]
                if gap <= 0.0:
                    valid = False
                else:
                    alt[j] = s2[j] * y[j] / gap
                    rest = rest - alt[j] * alt[j] / s2[j]
        if valid and rest >= 0.0:
            alt[k] = s[k] * wp.sqrt(rest)
            if y[k] < 0.0:
                alt[k] = -alt[k]
            if not on_boundary or wp.length(alt - y) < wp.length(x - y):
                x = alt
    k3 = core.c + core.rot @ x
    if inside:
        f = wp.normalize(core.rot @ wp.cw_div(x, s2))
        return True, -f, k3, -wp.length(x - y)
    dist = wp.length(k3 - v)
    if dist <= 0.0:
        return False, k3, k3, 0.0
    return True, (k3 - v) / dist, k3, dist


@wp.func
def closest_on_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """The point of triangle ``abc`` nearest ``p`` (Voronoi regions)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + d1 / (d1 - d3) * ab
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
    den = 1.0 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


@wp.func
def core_nearest(core: TerrainHullCore, v: wp.vec3) -> tuple[bool, wp.vec3, wp.vec3, float]:
    """Rule V: a hull's nearest point over its face triangles (the same count for every pair of one hull), or the
    face of least depth from inside; an ellipsoid's from its secular equation."""
    if core.ellipsoid != 0:
        return _ellipsoid_nearest(core, v)
    rt = wp.transpose(core.rot)
    pl = rt @ (v - core.c)
    xs = wp.cw_div(pl, core.s)
    inside = bool(True)
    best = float(_HUGE)
    face = int(core.face_begin)
    for f in range(core.face_begin, core.face_begin + core.face_count):
        p = core.table.planes[f]
        nf = wp.vec3(p[0], p[1], p[2])
        scale = wp.length(wp.cw_div(nf, core.s))
        depth = (p[3] - wp.dot(nf, xs)) / scale
        if depth < 0.0:
            inside = False
        if depth < best:
            best = depth
            face = f
    if inside:
        fn = _hull_face_normal(core, face)
        return True, -fn, v + best * fn, -best
    nearest = pl
    dist2 = float(_HUGE)
    for j in range(core.triangle_begin, core.triangle_begin + core.triangle_count):
        tri = core.table.triangles[j]
        q = closest_on_triangle(
            pl,
            wp.cw_mul(core.table.points[tri[0]], core.s),
            wp.cw_mul(core.table.points[tri[1]], core.s),
            wp.cw_mul(core.table.points[tri[2]], core.s),
        )
        d2 = wp.length_sq(q - pl)
        if d2 < dist2:
            dist2 = d2
            nearest = q
    k = core.c + core.rot @ nearest
    dist = wp.length(k - v)
    if dist <= 0.0:
        return False, k, k, 0.0
    return True, (k - v) / dist, k, dist


@wp.func
def core_recovery_count(core: TerrainHullCore) -> int:
    if core.ellipsoid != 0:
        return 14
    return core.point_count


@wp.func
def core_recovery_point(core: TerrainHullCore, i: int) -> wp.vec3:
    """Rule R's boundary point ``i``: a hull's points in mesh order; an ellipsoid's points on its local axes (+x,
    -x, +y, -y, +z, -z), then on its eight diagonals."""
    if core.ellipsoid == 0:
        return _hull_point(core, core.point_begin + i)
    local = wp.vec3(0.0)
    if i < 6:
        axis = i // 2
        sign = wp.where((i % 2) == 0, 1.0, -1.0)
        local[axis] = sign
    else:
        j = i - 6
        local = wp.vec3(
            wp.where((j & 4) == 0, _THIRD, -_THIRD),
            wp.where((j & 2) == 0, _THIRD, -_THIRD),
            wp.where((j & 1) == 0, _THIRD, -_THIRD),
        )
    return core.c + core.rot @ wp.cw_mul(core.s, local)

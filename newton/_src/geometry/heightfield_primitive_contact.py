# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Closed-form primitive-vs-heightfield-cell contacts.

Each admitted primitive type owns one query with the signature of
``compute_gjk_mpr_contacts`` that writes the generic ``ContactData`` contract,
so the reducer, export and solvers are shared. The table at the bottom is the
only place a new primitive has to be registered.

* Sphere: the distance from the center to the triangle prism
  (:mod:`heightfield_sphere_contact`).
* Capsule: the segment against the prism. Each end cap is a sphere query; when
  the segment's deepest penetrating point (or, outside, its nearest approach)
  lies strictly between the ends, a third contact is written there, so a
  capsule bridging a ridge still rests on it.
"""

from __future__ import annotations

import math
import os
from typing import Any

import warp as wp

from .contact_data import ContactData
from .heightfield_sphere_contact import (
    _ADMISSION_SLACK,
    _xy_inside_triangle,
    create_point_prism_writer,
    create_sphere_prism_contacts,
    point_prism_query,
    prism_normal,
)
from .support_function import TRIANGLE_PRISM_EXTRUSION, GenericShapeData, unpack_mesh_ptr
from .types import GeoType


def analytic_primitives_default() -> bool:
    """Return whether typed heightfield primitive queries use the closed forms by default."""
    value = os.environ.get("NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES", "1")
    if value not in ("0", "1"):
        raise ValueError("NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES must be 0 or 1")
    return value == "1"


def analytic_convex_default() -> bool:
    """Return whether heightfield convex-hull pairs use the closed-form row by default.

    On by default, like the other rows; ``NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX=0``
    keeps hulls on the generic query. (Measured neutral on Go2 rough once the
    per-cell manifolds were capped at four points.)
    """
    value = os.environ.get("NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX", "1")
    if value not in ("0", "1"):
        raise ValueError("NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX must be 0 or 1")
    return value == "1"


@wp.func
def _closest_segment_segment(p0: wp.vec3, p1: wp.vec3, q0: wp.vec3, q1: wp.vec3) -> tuple[float, float]:
    """Parameters of the closest points between two segments (Ericson 5.1.9)."""
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    eps = 1.0e-20
    s = float(0.0)
    t = float(0.0)
    if a <= eps and e <= eps:
        return 0.0, 0.0
    if a <= eps:
        t = wp.clamp(f / e, 0.0, 1.0)
        return 0.0, t
    c = wp.dot(d1, r)
    if e <= eps:
        s = wp.clamp(-c / a, 0.0, 1.0)
        return s, 0.0
    b = wp.dot(d1, d2)
    denom = a * e - b * b
    if denom > 0.0:
        s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = wp.clamp(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = wp.clamp((b - c) / a, 0.0, 1.0)
    return s, t


@wp.func
def _segment_edge(
    p0: wp.vec3,
    p1: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    best_t: float,
    best_q: wp.vec3,
    best_sq: float,
) -> tuple[float, wp.vec3, float]:
    s, u = _closest_segment_segment(p0, p1, a, b)
    q = a + u * (b - a)
    d_sq = wp.length_sq(p0 + s * (p1 - p0) - q)
    if d_sq < best_sq:
        return s, q, d_sq
    return best_t, best_q, best_sq


@wp.func
def _segment_prism_outside(
    p0: wp.vec3,
    p1: wp.vec3,
    e1: wp.vec3,
    e2: wp.vec3,
    n: wp.vec3,
    reach: float,
) -> tuple[float, wp.vec3, float]:
    """Closest segment parameter, prism point and squared distance for a non-penetrating segment.

    The closest pair of a segment and a convex polytope that do not intersect
    is attained at a segment end against a face or at the segment against an
    edge, so the end-point queries and the segment-edge queries over the top
    triangle and the vertical edges suffice. The bottom edges are included
    only within ``reach``.
    """
    down = wp.vec3(0.0, 0.0, TRIANGLE_PRISM_EXTRUSION)
    a0 = wp.vec3(0.0)
    q0, _n0, d0 = point_prism_query(p0, e1, e2, n, reach)
    best_t = float(0.0)
    best_q = q0
    best_sq = d0 * d0
    q1, _n1, d1 = point_prism_query(p1, e1, e2, n, reach)
    if d1 * d1 < best_sq:
        best_t = 1.0
        best_q = q1
        best_sq = d1 * d1
    best_t, best_q, best_sq = _segment_edge(p0, p1, a0, e1, best_t, best_q, best_sq)
    best_t, best_q, best_sq = _segment_edge(p0, p1, e1, e2, best_t, best_q, best_sq)
    best_t, best_q, best_sq = _segment_edge(p0, p1, e2, a0, best_t, best_q, best_sq)
    best_t, best_q, best_sq = _segment_edge(p0, p1, a0, a0 - down, best_t, best_q, best_sq)
    best_t, best_q, best_sq = _segment_edge(p0, p1, e1, e1 - down, best_t, best_q, best_sq)
    best_t, best_q, best_sq = _segment_edge(p0, p1, e2, e2 - down, best_t, best_q, best_sq)
    top_z = wp.max(0.0, wp.max(e1[2], e2[2]))
    low_z = wp.min(0.0, wp.min(e1[2], e2[2]))
    lowest = wp.min(p0[2], p1[2])
    highest = wp.max(p0[2], p1[2])
    if wp.max(lowest - (top_z - TRIANGLE_PRISM_EXTRUSION), (low_z - TRIANGLE_PRISM_EXTRUSION) - highest) <= reach:
        best_t, best_q, best_sq = _segment_edge(p0, p1, a0 - down, e1 - down, best_t, best_q, best_sq)
        best_t, best_q, best_sq = _segment_edge(p0, p1, e1 - down, e2 - down, best_t, best_q, best_sq)
        best_t, best_q, best_sq = _segment_edge(p0, p1, e2 - down, a0 - down, best_t, best_q, best_sq)
    return best_t, best_q, best_sq


@wp.func
def _clip_half_space(
    p0: wp.vec3, d: wp.vec3, point: wp.vec3, normal: wp.vec3, t_enter: float, t_exit: float
) -> tuple[float, float]:
    """Clip the parametric segment to the half-space ``dot(x - point, normal) <= 0``."""
    num = wp.dot(p0 - point, normal)
    den = wp.dot(d, normal)
    if wp.abs(den) < 1.0e-20:
        if num > 0.0:
            return 1.0, 0.0
        return t_enter, t_exit
    t = -num / den
    if den > 0.0:
        return t_enter, wp.min(t_exit, t)
    return wp.max(t_enter, t), t_exit


@wp.func
def _segment_prism_inside(p0: wp.vec3, p1: wp.vec3, e1: wp.vec3, e2: wp.vec3, n: wp.vec3) -> tuple[float, float]:
    """Parameter of the deepest penetrating segment point and its plane depth (negative), or depth 0."""
    d = p1 - p0
    a0 = wp.vec3(0.0)
    up = wp.vec3(0.0, 0.0, 1.0)
    t_enter = float(0.0)
    t_exit = float(1.0)
    t_enter, t_exit = _clip_half_space(p0, d, a0, n, t_enter, t_exit)
    for k in range(3):
        a = a0
        b = e1
        third = e2
        if k == 1:
            a = e1
            b = e2
            third = a0
        elif k == 2:
            a = e2
            b = a0
            third = e1
        outward = wp.cross(b - a, up)
        if wp.dot(third - a, outward) > 0.0:
            outward = -outward
        t_enter, t_exit = _clip_half_space(p0, d, a, outward, t_enter, t_exit)
    bottom = -TRIANGLE_PRISM_EXTRUSION * n[2]
    t_enter, t_exit = _clip_half_space(p0, d, a0 - up * TRIANGLE_PRISM_EXTRUSION, -n, t_enter, t_exit)
    if t_enter > t_exit:
        return 0.0, 0.0
    s_enter = wp.max(wp.dot(p0 + t_enter * d, n), bottom)
    s_exit = wp.max(wp.dot(p0 + t_exit * d, n), bottom)
    if s_exit < s_enter:
        return t_exit, wp.min(s_exit, 0.0)
    return t_enter, wp.min(s_enter, 0.0)


def create_capsule_prism_contacts(writer_func: Any, query_filter: Any = None):
    """Create the closed-form capsule-vs-prism query with the signature of ``compute_gjk_mpr_contacts``."""
    write_point = create_point_prism_writer(writer_func, query_filter)

    @wp.func
    def capsule_prism_contacts(
        shape_a_data: GenericShapeData,
        shape_b_data: GenericShapeData,
        rot_a: wp.quat,
        rot_b: wp.quat,
        pos_a_adjusted: wp.vec3,
        pos_b_adjusted: wp.vec3,
        rigid_gap: float,
        shape_a: int,
        shape_b: int,
        margin_a: float,
        margin_b: float,
        writer_data: Any,
        sort_sub_key: int = 0,
        query_context: Any = 0,
    ):
        e1 = shape_a_data.scale
        e2 = shape_a_data.auxiliary
        n = prism_normal(e1, e2)
        if n[2] == 0.0:
            return
        radius = shape_b_data.scale[0]
        half_height = shape_b_data.scale[1]
        threshold = rigid_gap + radius + margin_a + margin_b
        reach = threshold + _ADMISSION_SLACK
        # Capsule segment in the triangle-origin heightfield-local frame.
        c = wp.quat_rotate_inv(rot_a, pos_b_adjusted - pos_a_adjusted)
        axis = wp.quat_rotate_inv(rot_a, wp.quat_rotate(rot_b, wp.vec3(0.0, 0.0, half_height)))
        p0 = c - axis
        p1 = c + axis
        key = sort_sub_key << 3

        # End caps.
        for end in range(2):
            p = p0
            if end == 1:
                p = p1
            point_a, normal, d = point_prism_query(p, e1, e2, n, reach)
            write_point(
                p,
                point_a,
                normal,
                d,
                threshold,
                radius,
                rot_a,
                pos_a_adjusted,
                shape_a_data,
                margin_a,
                margin_b,
                shape_a,
                shape_b,
                rigid_gap,
                key | end,
                writer_data,
                query_context,
            )

        # Interior contact: the deepest penetrating axis point keeps the plane depth even
        # when it sits on a wall; outside, the nearest approach of the axis.
        t, depth = _segment_prism_inside(p0, p1, e1, e2, n)
        point_a = wp.vec3(0.0)
        normal = n
        d = depth
        if depth < 0.0:
            p = p0 + t * (p1 - p0)
            point_a = p - depth * n
        else:
            t, point_a, d_sq = _segment_prism_outside(p0, p1, e1, e2, n, reach)
            if d_sq > reach * reach:
                return
            p = p0 + t * (p1 - p0)
            d = wp.sqrt(d_sq)
            if d > 1.0e-12:
                normal = (p - point_a) / d
        if t > 1.0e-3 and t < 1.0 - 1.0e-3:
            write_point(
                p0 + t * (p1 - p0),
                point_a,
                normal,
                d,
                threshold,
                radius,
                rot_a,
                pos_a_adjusted,
                shape_a_data,
                margin_a,
                margin_b,
                shape_a,
                shape_b,
                rigid_gap,
                key | 2,
                writer_data,
                query_context,
            )

    return capsule_prism_contacts


@wp.func
def _clip_xy_segment(p0: wp.vec3, p1: wp.vec3, e1: wp.vec3, e2: wp.vec3) -> tuple[float, float]:
    """Parameter range of a segment whose vertical projection lies over the cell triangle.

    The prism walls are vertical, so clipping in the xy plane is exact. Returns
    ``t_enter > t_exit`` when the segment misses the footprint.
    """
    d = p1 - p0
    t_enter = float(0.0)
    t_exit = float(1.0)
    orientation = e1[0] * e2[1] - e1[1] * e2[0]
    for k in range(3):
        ax = float(0.0)
        ay = float(0.0)
        bx = e1[0]
        by = e1[1]
        if k == 1:
            ax = e1[0]
            ay = e1[1]
            bx = e2[0]
            by = e2[1]
        elif k == 2:
            ax = e2[0]
            ay = e2[1]
            bx = 0.0
            by = 0.0
        # Inward side of edge a->b for a counter-clockwise triangle (flipped otherwise).
        nx = -(by - ay)
        ny = bx - ax
        if orientation < 0.0:
            nx = -nx
            ny = -ny
        num = (p0[0] - ax) * nx + (p0[1] - ay) * ny
        den = d[0] * nx + d[1] * ny
        if wp.abs(den) < 1.0e-20:
            if num < 0.0:
                return 1.0, 0.0
        else:
            t = -num / den
            if den < 0.0:
                t_exit = wp.min(t_exit, t)
            else:
                t_enter = wp.max(t_enter, t)
    return t_enter, t_exit


_AXIAL_TOLERANCE = math.sin(2.0 * math.pi / 180.0)
"""Axis-to-face alignment within two degrees selects the flat-cap or rolling manifolds (matches the axial post-processing)."""


def create_cylinder_prism_contacts(writer_func: Any, query_filter: Any = None):
    """Create the closed-form cylinder-vs-prism query with the signature of ``compute_gjk_mpr_contacts``.

    The contact set of a flat-capped cylinder against a cell is one of three
    manifolds, selected by the angle between the axis and the top-face
    normal: a cap resting on the face (four rim points), the side line of a
    rolling cylinder (two points, which already lie in the plane the axial
    post-processing projects onto), or the single lowest rim point of a
    tilted cylinder. Every candidate surface point is resolved against the
    prism by the shared point query, so cell boundaries and penetration keep
    the sphere semantics.
    """
    has_query_filter = query_filter is not None

    @wp.func
    def write_surface_point(
        p: wp.vec3,
        e1: wp.vec3,
        e2: wp.vec3,
        n: wp.vec3,
        reach: float,
        threshold: float,
        rot_a: wp.quat,
        pos_a: wp.vec3,
        prism: GenericShapeData,
        margin_a: float,
        margin_b: float,
        shape_a: int,
        shape_b: int,
        rigid_gap: float,
        sort_sub_key: int,
        writer_data: Any,
        query_context: Any,
    ):
        point_a, normal, d = point_prism_query(p, e1, e2, n, reach)
        if d > threshold + _ADMISSION_SLACK:
            return
        if wp.static(has_query_filter):
            if d >= 0.0:
                point_a, normal, d = query_filter(prism, point_a, normal, d, query_context)
        contact = ContactData()
        contact.radius_eff_a = 0.0
        contact.radius_eff_b = 0.0
        contact.margin_a = margin_a
        contact.margin_b = margin_b
        contact.shape_a = shape_a
        contact.shape_b = shape_b
        contact.gap_sum = rigid_gap
        contact.sort_sub_key = sort_sub_key
        contact.contact_point_center = wp.quat_rotate(rot_a, 0.5 * (point_a + p)) + pos_a
        contact.contact_normal_a_to_b = wp.quat_rotate(rot_a, normal)
        contact.contact_distance = d
        writer_func(contact, writer_data, -1)

    @wp.func
    def cylinder_prism_contacts(
        shape_a_data: GenericShapeData,
        shape_b_data: GenericShapeData,
        rot_a: wp.quat,
        rot_b: wp.quat,
        pos_a_adjusted: wp.vec3,
        pos_b_adjusted: wp.vec3,
        rigid_gap: float,
        shape_a: int,
        shape_b: int,
        margin_a: float,
        margin_b: float,
        writer_data: Any,
        sort_sub_key: int = 0,
        query_context: Any = 0,
    ):
        e1 = shape_a_data.scale
        e2 = shape_a_data.auxiliary
        n = prism_normal(e1, e2)
        if n[2] == 0.0:
            return
        radius = shape_b_data.scale[0]
        half_height = shape_b_data.scale[1]
        threshold = rigid_gap + margin_a + margin_b
        reach = threshold + _ADMISSION_SLACK
        # Cylinder center and unit axis in the triangle-origin heightfield-local frame.
        c = wp.quat_rotate_inv(rot_a, pos_b_adjusted - pos_a_adjusted)
        axis = wp.quat_rotate_inv(rot_a, wp.quat_rotate(rot_b, wp.vec3(0.0, 0.0, 1.0)))
        along = wp.dot(axis, n)
        # Unit direction of steepest descent on the cap plane (toward -n), when defined.
        lateral = n - along * axis
        lateral_length = wp.length(lateral)
        if lateral_length > 1.0e-6:
            lateral = lateral / lateral_length
        else:
            lateral = wp.vec3(0.0)
            if wp.abs(axis[0]) < 0.9:
                lateral = wp.normalize(wp.cross(axis, wp.vec3(1.0, 0.0, 0.0)))
            else:
                lateral = wp.normalize(wp.cross(axis, wp.vec3(0.0, 1.0, 0.0)))
        key = sort_sub_key << 3

        if wp.abs(along) < wp.static(_AXIAL_TOLERANCE):
            # Rolling: the side line nearest the face, clipped to the cell footprint.
            base = c - radius * lateral
            line0 = base - half_height * axis
            line1 = base + half_height * axis
            t_enter, t_exit = _clip_xy_segment(line0, line1, e1, e2)
            if t_enter > t_exit:
                return
            for end in range(2):
                t = t_enter
                if end == 1:
                    t = t_exit
                write_surface_point(
                    line0 + t * (line1 - line0),
                    e1,
                    e2,
                    n,
                    reach,
                    threshold,
                    rot_a,
                    pos_a_adjusted,
                    shape_a_data,
                    margin_a,
                    margin_b,
                    shape_a,
                    shape_b,
                    rigid_gap,
                    key | end,
                    writer_data,
                    query_context,
                )
            return

        # The lower cap faces the terrain.
        cap_sign = 1.0
        if along > 0.0:
            cap_sign = -1.0
        cap = c + cap_sign * half_height * axis
        if wp.abs(along) > wp.static(math.sqrt(1.0 - _AXIAL_TOLERANCE * _AXIAL_TOLERANCE)):
            # Flat cap: four rim points, two along the descent direction and two across it,
            # each pulled back along its radius to the farthest cap point over the cell.
            across = wp.cross(axis, lateral)
            for corner in range(4):
                direction = lateral
                if corner == 1:
                    direction = -lateral
                elif corner == 2:
                    direction = across
                elif corner == 3:
                    direction = -across
                rim = cap + radius * direction
                t_enter, t_exit = _clip_xy_segment(cap, rim, e1, e2)
                if t_enter > t_exit:
                    continue
                write_surface_point(
                    cap + t_exit * (rim - cap),
                    e1,
                    e2,
                    n,
                    reach,
                    threshold,
                    rot_a,
                    pos_a_adjusted,
                    shape_a_data,
                    margin_a,
                    margin_b,
                    shape_a,
                    shape_b,
                    rigid_gap,
                    key | corner,
                    writer_data,
                    query_context,
                )
            return

        # Tilted: the lowest rim point, pulled back along its radius to the cell.
        rim = cap - radius * lateral
        t_enter, t_exit = _clip_xy_segment(cap, rim, e1, e2)
        if t_enter > t_exit:
            return
        write_surface_point(
            cap + t_exit * (rim - cap),
            e1,
            e2,
            n,
            reach,
            threshold,
            rot_a,
            pos_a_adjusted,
            shape_a_data,
            margin_a,
            margin_b,
            shape_a,
            shape_b,
            rigid_gap,
            key,
            writer_data,
            query_context,
        )

    return cylinder_prism_contacts


_Polygon = wp.types.matrix(shape=(8, 3), dtype=wp.float32)


@wp.func
def _clip_polygon_xy(points: _Polygon, count: int, e1: wp.vec3, e2: wp.vec3) -> tuple[_Polygon, int]:
    """Sutherland-Hodgman clip of a convex polygon against the cell footprint in the xy plane.

    A box face has four vertices; clipping against a triangle yields at most
    seven. Points keep their z so they stay on the box face.
    """
    orientation = e1[0] * e2[1] - e1[1] * e2[0]
    current = points
    current_count = count
    for k in range(3):
        ax = float(0.0)
        ay = float(0.0)
        bx = e1[0]
        by = e1[1]
        if k == 1:
            ax = e1[0]
            ay = e1[1]
            bx = e2[0]
            by = e2[1]
        elif k == 2:
            ax = e2[0]
            ay = e2[1]
            bx = 0.0
            by = 0.0
        nx = -(by - ay)
        ny = bx - ax
        if orientation < 0.0:
            nx = -nx
            ny = -ny
        output = _Polygon()
        output_count = int(0)
        if current_count == 0:
            return output, 0
        previous = wp.vec3(current[current_count - 1, 0], current[current_count - 1, 1], current[current_count - 1, 2])
        previous_inside = (previous[0] - ax) * nx + (previous[1] - ay) * ny >= 0.0
        for i in range(current_count):
            point = wp.vec3(current[i, 0], current[i, 1], current[i, 2])
            inside = (point[0] - ax) * nx + (point[1] - ay) * ny >= 0.0
            if inside != previous_inside:
                dp = (previous[0] - ax) * nx + (previous[1] - ay) * ny
                dc = (point[0] - ax) * nx + (point[1] - ay) * ny
                t = dp / (dp - dc)
                crossing = previous + t * (point - previous)
                if output_count < 8:
                    output[output_count, 0] = crossing[0]
                    output[output_count, 1] = crossing[1]
                    output[output_count, 2] = crossing[2]
                    output_count += 1
            if inside and output_count < 8:
                output[output_count, 0] = point[0]
                output[output_count, 1] = point[1]
                output[output_count, 2] = point[2]
                output_count += 1
            previous = point
            previous_inside = inside
        current = output
        current_count = output_count
    return current, current_count


@wp.func
def _footprint_point_query(p: wp.vec3, n: wp.vec3) -> tuple[wp.vec3, wp.vec3, float]:
    """Plane projection, normal and signed distance of a point known to lie over the cell.

    Vertices produced by :func:`_clip_polygon_xy` are inside the footprint by
    construction, often exactly on its boundary; the general point query may
    classify such a boundary point as outside and answer with a side wall,
    which turns a resting face contact into a lateral push. The clipped
    polygon always belongs to the top face, so its depth is the plane depth.
    """
    s = wp.dot(p, n)
    return p - s * n, n, s


_MAX_SUPPORT = 4


@wp.func
def _deepest_support(points: _Polygon, count: int, n: wp.vec3) -> tuple[_Polygon, int]:
    """Keep the ``_MAX_SUPPORT`` deepest points along the cell normal.

    A convex face on a plane needs at most four support points. Clipping a
    face that crosses a cell yields up to seven (box) or eight (hull)
    vertices, and every extra one is a contact row the solver visits: Go2
    rough measured +2% with the full clipped polygon.
    """
    if count <= _MAX_SUPPORT:
        return points, count
    kept = _Polygon()
    taken = wp.vector(length=8, dtype=wp.int32)
    for i in range(8):
        taken[i] = 0
    for slot in range(_MAX_SUPPORT):
        best = int(-1)
        best_depth = float(1.0e30)
        for i in range(count):
            if taken[i] == 0:
                depth = points[i, 0] * n[0] + points[i, 1] * n[1] + points[i, 2] * n[2]
                if depth < best_depth:
                    best_depth = depth
                    best = i
        taken[best] = 1
        for k in range(3):
            kept[slot, k] = points[best, k]
    return kept, _MAX_SUPPORT


def create_box_prism_contacts(writer_func: Any, query_filter: Any = None):
    """Create the closed-form box-vs-prism query with the signature of ``compute_gjk_mpr_contacts``.

    The box face most opposed to the cell's face normal is the one touching
    the terrain whether the box rests flat (four points), on an edge (that
    face contains the edge) or on a corner (it contains the corner). Its
    polygon is clipped to the cell footprint and every clipped vertex is
    resolved against the prism by the shared point query, so a face covering
    a small cell contributes that cell's triangle and penetration keeps the
    plane depth.
    """
    has_query_filter = query_filter is not None

    @wp.func
    def box_prism_contacts(
        shape_a_data: GenericShapeData,
        shape_b_data: GenericShapeData,
        rot_a: wp.quat,
        rot_b: wp.quat,
        pos_a_adjusted: wp.vec3,
        pos_b_adjusted: wp.vec3,
        rigid_gap: float,
        shape_a: int,
        shape_b: int,
        margin_a: float,
        margin_b: float,
        writer_data: Any,
        sort_sub_key: int = 0,
        query_context: Any = 0,
    ):
        e1 = shape_a_data.scale
        e2 = shape_a_data.auxiliary
        n = prism_normal(e1, e2)
        if n[2] == 0.0:
            return
        half = shape_b_data.scale
        threshold = rigid_gap + margin_a + margin_b
        # Box center and axes in the triangle-origin heightfield-local frame.
        relative = wp.quat_inverse(rot_a) * rot_b
        c = wp.quat_rotate_inv(rot_a, pos_b_adjusted - pos_a_adjusted)
        u0 = wp.quat_rotate(relative, wp.vec3(1.0, 0.0, 0.0))
        u1 = wp.quat_rotate(relative, wp.vec3(0.0, 1.0, 0.0))
        u2 = wp.quat_rotate(relative, wp.vec3(0.0, 0.0, 1.0))
        # The face most opposed to the terrain normal.
        d0 = wp.dot(u0, n)
        d1 = wp.dot(u1, n)
        d2 = wp.dot(u2, n)
        face_axis = u2
        face_sign = -1.0
        span_a = u0 * half[0]
        span_b = u1 * half[1]
        face_half = half[2]
        if wp.abs(d0) >= wp.abs(d1) and wp.abs(d0) >= wp.abs(d2):
            face_axis = u0
            span_a = u1 * half[1]
            span_b = u2 * half[2]
            face_half = half[0]
            if d0 > 0.0:
                face_sign = -1.0
            else:
                face_sign = 1.0
        elif wp.abs(d1) >= wp.abs(d2):
            face_axis = u1
            span_a = u2 * half[2]
            span_b = u0 * half[0]
            face_half = half[1]
            if d1 > 0.0:
                face_sign = -1.0
            else:
                face_sign = 1.0
        else:
            if d2 > 0.0:
                face_sign = -1.0
            else:
                face_sign = 1.0
        center = c + face_sign * face_half * face_axis
        polygon = _Polygon()
        for corner in range(4):
            sa = 1.0
            sb = 1.0
            if corner == 1:
                sb = -1.0
            elif corner == 2:
                sa = -1.0
                sb = -1.0
            elif corner == 3:
                sa = -1.0
            vertex = center + sa * span_a + sb * span_b
            polygon[corner, 0] = vertex[0]
            polygon[corner, 1] = vertex[1]
            polygon[corner, 2] = vertex[2]
        clipped, count = _clip_polygon_xy(polygon, 4, e1, e2)
        clipped, count = _deepest_support(clipped, count, n)
        key = sort_sub_key << 3
        for i in range(count):
            p = wp.vec3(clipped[i, 0], clipped[i, 1], clipped[i, 2])
            point_a, normal, d = _footprint_point_query(p, n)
            if d > threshold + _ADMISSION_SLACK:
                continue
            if wp.static(has_query_filter):
                if d >= 0.0:
                    point_a, normal, d = query_filter(shape_a_data, point_a, normal, d, query_context)
            contact = ContactData()
            contact.radius_eff_a = 0.0
            contact.radius_eff_b = 0.0
            contact.margin_a = margin_a
            contact.margin_b = margin_b
            contact.shape_a = shape_a
            contact.shape_b = shape_b
            contact.gap_sum = rigid_gap
            contact.sort_sub_key = key | i
            contact.contact_point_center = wp.quat_rotate(rot_a, 0.5 * (point_a + p)) + pos_a_adjusted
            contact.contact_normal_a_to_b = wp.quat_rotate(rot_a, normal)
            contact.contact_distance = d
            writer_func(contact, writer_data, -1)

    return box_prism_contacts


_BAND_ABSOLUTE = 1.0e-3
_BAND_RELATIVE = 0.01
_SUPPORT_DIRECTIONS = 8


def create_convex_prism_contacts(writer_func: Any, query_filter: Any = None):
    """Create the closed-form convex-hull-vs-prism query with the signature of ``compute_gjk_mpr_contacts``.

    The hull vertices within admission distance of the cell plane (and never
    fewer than those within a small band of the lowest one) form the touching
    feature: a face when the hull rests flat, an edge or a single vertex
    otherwise. Up to eight of them, the extremes along eight fixed in-plane
    directions, are taken in that angular order, which is a convex polygon in
    the cell's xy plane by construction. Like the box row the polygon is then
    clipped to the cell footprint and every clipped vertex is resolved at its
    plane depth, so a face covering several small cells gives each cell its
    own support points. No face or edge tables are needed, only the hull
    points the support function already reads.
    """
    has_query_filter = query_filter is not None

    @wp.func
    def convex_prism_contacts(
        shape_a_data: GenericShapeData,
        shape_b_data: GenericShapeData,
        rot_a: wp.quat,
        rot_b: wp.quat,
        pos_a_adjusted: wp.vec3,
        pos_b_adjusted: wp.vec3,
        rigid_gap: float,
        shape_a: int,
        shape_b: int,
        margin_a: float,
        margin_b: float,
        writer_data: Any,
        sort_sub_key: int = 0,
        query_context: Any = 0,
    ):
        e1 = shape_a_data.scale
        e2 = shape_a_data.auxiliary
        n = prism_normal(e1, e2)
        if n[2] == 0.0:
            return
        threshold = rigid_gap + margin_a + margin_b
        mesh = wp.mesh_get(unpack_mesh_ptr(shape_b_data.auxiliary))
        count = mesh.points.shape[0]
        if count == 0:
            return
        relative = wp.quat_inverse(rot_a) * rot_b
        c = wp.quat_rotate_inv(rot_a, pos_b_adjusted - pos_a_adjusted)
        scale = shape_b_data.scale
        # Heights along the cell normal as one dot product per vertex in the
        # mesh frame: dot(c + R(s*p), n) = dot(c, n) + dot(p, s*(R^T n)).
        height_axis = wp.cw_mul(wp.quat_rotate_inv(relative, n), scale)
        height_base = wp.dot(c, n)
        lowest = float(1.0e30)
        highest = float(-1.0e30)
        lowest_index = int(0)
        for i in range(count):
            h = height_base + wp.dot(mesh.points[i], height_axis)
            if h < lowest:
                lowest = h
                lowest_index = i
            highest = wp.max(highest, h)
        if lowest - threshold - _ADMISSION_SLACK > 0.0 and lowest > 0.0:
            # The whole hull is farther from the plane than the admission threshold.
            # (Points over neighbouring cells can only be farther from this prism.)
            return
        band = lowest + wp.max(_BAND_ABSOLUTE, _BAND_RELATIVE * (highest - lowest))
        limit = wp.max(band, threshold + _ADMISSION_SLACK)
        # The touching polygon: extreme touching vertices along eight in-plane
        # directions, taken in angular order so they arrive counter-clockwise.
        # One pass over the vertices with eight running maxima; the directions
        # are expressed in the mesh frame so each extreme is one dot product.
        directions = _Polygon()
        for k in range(_SUPPORT_DIRECTIONS):
            angle = float(k) * (wp.pi / float(_SUPPORT_DIRECTIONS)) * 2.0
            direction = wp.cw_mul(wp.quat_rotate_inv(relative, wp.vec3(wp.cos(angle), wp.sin(angle), 0.0)), scale)
            directions[k, 0] = direction[0]
            directions[k, 1] = direction[1]
            directions[k, 2] = direction[2]
        best = wp.vector(length=8, dtype=wp.int32)
        best_value = wp.vector(length=8, dtype=wp.float32)
        for k in range(_SUPPORT_DIRECTIONS):
            best[k] = -1
            best_value[k] = -1.0e30
        for i in range(count):
            vertex = mesh.points[i]
            if height_base + wp.dot(vertex, height_axis) <= limit:
                for k in range(_SUPPORT_DIRECTIONS):
                    value = vertex[0] * directions[k, 0] + vertex[1] * directions[k, 1] + vertex[2] * directions[k, 2]
                    if value > best_value[k]:
                        best_value[k] = value
                        best[k] = i
        chosen = wp.vector(length=8, dtype=wp.int32)
        selected = int(0)
        lowest_chosen = int(0)
        for k in range(_SUPPORT_DIRECTIONS):
            index = best[k]
            if index < 0:
                continue
            duplicate = int(0)
            for j in range(selected):
                if chosen[j] == index:
                    duplicate = 1
            if duplicate == 0:
                chosen[selected] = index
                selected += 1
                if index == lowest_index:
                    lowest_chosen = 1
        points = _Polygon()
        for j in range(selected):
            p = c + wp.quat_rotate(relative, wp.cw_mul(mesh.points[chosen[j]], scale))
            points[j, 0] = p[0]
            points[j, 1] = p[1]
            points[j, 2] = p[2]
        clipped, clipped_count = _clip_polygon_xy(points, selected, e1, e2)
        # A hull resting on a vertex may project that vertex inside the
        # touching polygon, where no direction selects it; the cell over it
        # still has to report its exact depth, so it goes first and the
        # polygon fills the remaining slots of the eight-point budget.
        if lowest_chosen == 0:
            p = c + wp.quat_rotate(relative, wp.cw_mul(mesh.points[lowest_index], scale))
            if _xy_inside_triangle(p, e1, e2):
                clipped_count = wp.min(clipped_count, 7)
                for i in range(clipped_count, 0, -1):
                    for k in range(3):
                        clipped[i, k] = clipped[i - 1, k]
                clipped[0, 0] = p[0]
                clipped[0, 1] = p[1]
                clipped[0, 2] = p[2]
                clipped_count += 1
        clipped, clipped_count = _deepest_support(clipped, clipped_count, n)
        key = sort_sub_key << 3
        for i in range(clipped_count):
            p = wp.vec3(clipped[i, 0], clipped[i, 1], clipped[i, 2])
            point_a, normal, d = _footprint_point_query(p, n)
            if d > threshold + _ADMISSION_SLACK:
                continue
            if wp.static(has_query_filter):
                if d >= 0.0:
                    point_a, normal, d = query_filter(shape_a_data, point_a, normal, d, query_context)
            contact = ContactData()
            contact.radius_eff_a = 0.0
            contact.radius_eff_b = 0.0
            contact.margin_a = margin_a
            contact.margin_b = margin_b
            contact.shape_a = shape_a
            contact.shape_b = shape_b
            contact.gap_sum = rigid_gap
            contact.sort_sub_key = key | i
            contact.contact_point_center = wp.quat_rotate(rot_a, 0.5 * (point_a + p)) + pos_a_adjusted
            contact.contact_normal_a_to_b = wp.quat_rotate(rot_a, normal)
            contact.contact_distance = d
            writer_func(contact, writer_data, -1)

    return convex_prism_contacts


ANALYTIC_PRIMITIVE_QUERIES = {
    int(GeoType.SPHERE): create_sphere_prism_contacts,
    int(GeoType.CAPSULE): create_capsule_prism_contacts,
    int(GeoType.CYLINDER): create_cylinder_prism_contacts,
    int(GeoType.BOX): create_box_prism_contacts,
    int(GeoType.CONVEX_MESH): create_convex_prism_contacts,
}
"""Primitive types with a closed-form heightfield-cell query, by factory."""

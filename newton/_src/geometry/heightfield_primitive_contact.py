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
    PROXY_RADIUS,
    create_sphere_prism_contacts,
    point_prism_query,
    prism_normal,
)
from .support_function import TRIANGLE_PRISM_EXTRUSION, GenericShapeData
from .types import GeoType


def analytic_primitives_default() -> bool:
    """Return whether typed heightfield primitive queries use the closed forms by default."""
    value = os.environ.get("NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES", "1")
    if value not in ("0", "1"):
        raise ValueError("NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES must be 0 or 1")
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
    has_query_filter = query_filter is not None

    @wp.func
    def write_point(
        p: wp.vec3,
        point_a: wp.vec3,
        normal: wp.vec3,
        d: float,
        threshold: float,
        radius: float,
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
        """Write one axis point's contact from its prism query (``point_a``, outward ``normal``, signed ``d``)."""
        point_b = p - normal * PROXY_RADIUS
        signed_distance = d - PROXY_RADIUS
        if signed_distance > threshold + _ADMISSION_SLACK:
            return
        if wp.static(has_query_filter):
            if signed_distance >= 0.0 and not query_filter(prism, point_a, normal, query_context):
                return
        normal_ws = wp.quat_rotate(rot_a, normal)
        point = wp.quat_rotate(rot_a, 0.5 * (point_a + point_b)) + pos_a
        contact = ContactData()
        contact.radius_eff_a = 0.0
        contact.radius_eff_b = radius
        contact.margin_a = margin_a
        contact.margin_b = margin_b
        contact.shape_a = shape_a
        contact.shape_b = shape_b
        contact.gap_sum = rigid_gap
        contact.sort_sub_key = sort_sub_key
        contact.contact_point_center = point - normal_ws * (radius * 0.5)
        contact.contact_normal_a_to_b = normal_ws
        contact.contact_distance = signed_distance - radius
        writer_func(contact, writer_data, -1)

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
        reach = threshold + PROXY_RADIUS + _ADMISSION_SLACK
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
            if d >= 0.0 and not query_filter(prism, point_a, normal, query_context):
                return
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


ANALYTIC_PRIMITIVE_QUERIES = {
    int(GeoType.SPHERE): create_sphere_prism_contacts,
    int(GeoType.CAPSULE): create_capsule_prism_contacts,
    int(GeoType.CYLINDER): create_cylinder_prism_contacts,
}
"""Primitive types with a closed-form heightfield-cell query, by factory."""

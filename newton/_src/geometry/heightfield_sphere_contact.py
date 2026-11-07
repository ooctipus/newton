# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Closed-form sphere-vs-heightfield-prism contact.

The typed heightfield sphere query shrank the sphere to a 0.1 mm proxy and
ran MPR, GJK, the prism penetration refiner and the single-contact writer for
every (sphere, triangle) pair. For a sphere all of that reduces to the distance
from its center to the triangle prism, which this module evaluates directly:

* center above the triangle plane: the nearest point on the prism, which is
  the nearest point on the triangle when the center projects onto its face
  and otherwise the nearest point among the triangle, its three extruded side
  walls and the bottom face;
* center inside the prism: the top-face normal with the plane depth, exactly
  the result of the triangle-prism penetration refiner.

The emitted :class:`ContactData` keeps the proxy radius so that every field,
including the 0.1 mm witness bias of the generic query, is reproduced and the
reducer, export and solver see the same contacts.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .contact_data import ContactData
from .support_function import (
    TRIANGLE_PRISM_EXTRUSION,
    GenericShapeData,
    closest_point_on_triangle,
)

PROXY_RADIUS = 1.0e-4
"""Sphere proxy radius of the generic query (``small_radius`` in ``create_compute_gjk_mpr_contacts``)."""

_ADMISSION_SLACK = 1.0e-4
"""Separation beyond the query threshold at which a pair is dropped without consulting the writer."""


@wp.func
def _xy_inside_triangle(p: wp.vec3, b: wp.vec3, c: wp.vec3) -> bool:
    """Test the vertical projection of ``p`` against the triangle (0, b, c)."""
    det = b[0] * c[1] - b[1] * c[0]
    u = p[0] * c[1] - p[1] * c[0]
    v = b[0] * p[1] - b[1] * p[0]
    if det < 0.0:
        det = -det
        u = -u
        v = -v
    return u >= 0.0 and v >= 0.0 and u + v <= det


@wp.func
def _nearer(c: wp.vec3, candidate: wp.vec3, best: wp.vec3, best_sq: float) -> tuple[wp.vec3, float]:
    d_sq = wp.length_sq(c - candidate)
    if d_sq < best_sq:
        return candidate, d_sq
    return best, best_sq


@wp.func
def _closest_point_on_prism_sides(
    c: wp.vec3,
    e1: wp.vec3,
    e2: wp.vec3,
    best: wp.vec3,
    best_sq: float,
    reach: float,
) -> tuple[wp.vec3, float]:
    """Refine the nearest prism point with the side walls and, within reach, the bottom face."""
    down = wp.vec3(0.0, 0.0, TRIANGLE_PRISM_EXTRUSION)
    a0 = wp.vec3(0.0)
    a1 = a0 - down
    b1 = e1 - down
    c1 = e2 - down
    best, best_sq = _nearer(c, closest_point_on_triangle(c, a0, e1, b1), best, best_sq)
    best, best_sq = _nearer(c, closest_point_on_triangle(c, a0, b1, a1), best, best_sq)
    best, best_sq = _nearer(c, closest_point_on_triangle(c, e1, e2, c1), best, best_sq)
    best, best_sq = _nearer(c, closest_point_on_triangle(c, e1, c1, b1), best, best_sq)
    best, best_sq = _nearer(c, closest_point_on_triangle(c, e2, a0, a1), best, best_sq)
    best, best_sq = _nearer(c, closest_point_on_triangle(c, e2, a1, c1), best, best_sq)
    top_z = wp.max(0.0, wp.max(e1[2], e2[2]))
    low_z = wp.min(0.0, wp.min(e1[2], e2[2]))
    # The bottom face lies in z within [low_z - extrusion, top_z - extrusion].
    below = c[2] - (top_z - TRIANGLE_PRISM_EXTRUSION)
    above = (low_z - TRIANGLE_PRISM_EXTRUSION) - c[2]
    if wp.max(below, above) <= reach:
        best, best_sq = _nearer(c, closest_point_on_triangle(c, a1, b1, c1), best, best_sq)
    return best, best_sq


@wp.func
def prism_normal(e1: wp.vec3, e2: wp.vec3) -> wp.vec3:
    """Unit top-face normal with positive Z, or zero for a degenerate cell."""
    n = wp.cross(e1, e2)
    n_sq = wp.length_sq(n)
    if n_sq < 1.0e-24:
        return wp.vec3(0.0)
    n = n / wp.sqrt(n_sq)
    if n[2] < 0.0:
        n = -n
    return n


@wp.func
def point_prism_query(c: wp.vec3, e1: wp.vec3, e2: wp.vec3, n: wp.vec3, reach: float) -> tuple[wp.vec3, wp.vec3, float]:
    """Nearest prism point, outward direction and signed distance of a point.

    Inside the prism the direction is the top-face normal and the distance is
    the (negative) plane depth, as the triangle-prism penetration refiner
    reports; outside it is the nearest point among the top triangle, the
    side walls and, within ``reach``, the bottom face.
    """
    s = wp.dot(c, n)
    projected = c - s * n
    top = closest_point_on_triangle(projected, wp.vec3(0.0), e1, e2)
    on_face = wp.length_sq(projected - top) < 1.0e-10
    if s < 0.0 and n[2] > 0.0:
        if -s <= TRIANGLE_PRISM_EXTRUSION * n[2] and _xy_inside_triangle(c, e1, e2):
            return projected, n, s
    best = top
    best_sq = wp.length_sq(c - top)
    if not (s >= 0.0 and on_face):
        best, best_sq = _closest_point_on_prism_sides(c, e1, e2, best, best_sq, reach)
    d = wp.sqrt(best_sq)
    direction = n
    if d > 1.0e-12:
        direction = (c - best) / d
    return best, direction, d


def create_sphere_prism_contacts(writer_func: Any, query_filter: Any = None):
    """Create the closed-form sphere-vs-prism query with the signature of ``compute_gjk_mpr_contacts``.

    Args:
        writer_func: Contact writer ``(ContactData, writer_data, output_index)``.
        query_filter: Optional predicate for nonnegative-distance witnesses, as
            accepted by :func:`create_compute_gjk_mpr_contacts`.
    """
    has_query_filter = query_filter is not None

    @wp.func
    def sphere_prism_contacts(
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
        threshold = rigid_gap + radius + margin_a + margin_b
        # Sphere center in the triangle-origin heightfield-local frame.
        c = wp.quat_rotate_inv(rot_a, pos_b_adjusted - pos_a_adjusted)
        point_a, normal, d = point_prism_query(c, e1, e2, n, threshold + PROXY_RADIUS + _ADMISSION_SLACK)
        point_b = c - normal * PROXY_RADIUS
        signed_distance = d - PROXY_RADIUS

        if signed_distance > threshold + _ADMISSION_SLACK:
            return
        if wp.static(has_query_filter):
            if signed_distance >= 0.0 and not query_filter(shape_a_data, point_a, normal, query_context):
                return

        normal_ws = wp.quat_rotate(rot_a, normal)
        point = wp.quat_rotate(rot_a, 0.5 * (point_a + point_b)) + pos_a_adjusted

        contact = ContactData()
        contact.radius_eff_a = 0.0
        contact.radius_eff_b = radius
        contact.margin_a = margin_a
        contact.margin_b = margin_b
        contact.shape_a = shape_a
        contact.shape_b = shape_b
        contact.gap_sum = rigid_gap
        contact.sort_sub_key = sort_sub_key << 3
        # Minkowski post-processing of the sphere proxy (post_process_axial_on_discrete_contact).
        contact.contact_point_center = point - normal_ws * (radius * 0.5)
        contact.contact_normal_a_to_b = normal_ws
        contact.contact_distance = signed_distance - radius
        writer_func(contact, writer_data, -1)

    return sphere_prism_contacts

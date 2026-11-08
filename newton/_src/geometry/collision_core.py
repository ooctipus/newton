# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import warp as wp

from ..core.types import vec5
from .broad_phase_common import binary_search
from .collision_convex import (
    create_plane_convex_query,
    create_solve_convex_multi_contact,
    create_solve_convex_single_contact,
    create_write_convex_query_result,
)
from .contact_data import ContactData
from .flags import MeshProperties
from .mesh_cells import (
    MESH_CELLS_LEVEL_SHIFT,
    MeshCells,
    MeshCellSlot,
    MeshGrid,
    MeshQueryBound,
    mesh_cells_box_in_cell,
    mesh_cells_box_meets_slabs,
    mesh_cells_candidate,
    mesh_cells_entry_box,
    mesh_cells_entry_first_in_range,
    mesh_cells_entry_overlaps,
    mesh_cells_find,
    mesh_cells_first_in_range,
    mesh_cells_key,
    mesh_cells_key_cell,
    mesh_cells_range,
    mesh_cells_slot_box,
    mesh_cells_slot_overlaps,
    mesh_query_bound_box,
    mesh_query_bound_meets_triangle,
    triangle_meets_box,
)
from .mpr import create_support_map_function
from .multicontact import create_build_manifold
from .support_function import (
    GenericShapeData,
    GeoTypeEx,
    SupportMapDataProvider,
    closest_point_on_triangle,
    pack_mesh_ptr,
    support_map,
    support_map_lean,
    support_map_lean_accelerated,
    unpack_mesh_ptr,
)
from .types import GeoType

# Configuration flag for multi-contact generation
ENABLE_MULTI_CONTACT = True

# GJK/MPR see a sphere or capsule as its core (point or segment) inflated by this radius; the rest of the radius
# is restored by the post-processors below, so a reported distance is the true surface distance.
MINKOWSKI_CORE_RADIUS = wp.constant(1.0e-4)

# Type definitions for multi-contact manifolds
_mat53f = wp.types.matrix((5, 3), wp.float32)

# Type definitions for single-contact mode
_vec1 = wp.types.vector(1, wp.float32)


@wp.func
def is_discrete_shape(shape_type: int) -> bool:
    """A discrete shape can be represented with a finite amount of flat polygon faces."""
    return (
        shape_type == GeoType.BOX
        or shape_type == GeoType.CONVEX_MESH
        or shape_type == GeoTypeEx.TRIANGLE
        or shape_type == GeoTypeEx.TRIANGLE_PRISM
        or shape_type == GeoType.PLANE
    )


@wp.func
def project_point_onto_plane(point: wp.vec3, plane_point: wp.vec3, plane_normal: wp.vec3) -> wp.vec3:
    """
    Project a point onto a plane defined by a point and normal.

    Args:
        point: The point to project
        plane_point: A point on the plane
        plane_normal: Normal vector of the plane (should be normalized)

    Returns:
        The projected point on the plane
    """
    to_point = point - plane_point
    distance_to_plane = wp.dot(to_point, plane_normal)
    projected_point = point - plane_normal * distance_to_plane
    return projected_point


@wp.func
def compute_plane_normal_from_contacts(
    points: _mat53f,
    normal: wp.vec3,
    signed_distances: vec5,
    count: int,
) -> wp.vec3:
    """
    Compute plane normal from reconstructed plane points.

    Reconstructs the plane points from contact data and computes the plane normal
    using fan triangulation to find the largest area triangle for numerical stability.

    Args:
        points: Contact points matrix (5x3)
        normal: Initial contact normal (used for reconstruction)
        signed_distances: Signed distances vector (5 elements)
        count: Number of contact points

    Returns:
        Normalized plane normal from the contact points
    """
    if count < 3:
        # Not enough points to form a triangle, return original normal
        return normal

    # Reconstruct plane points from contact data
    # Use first point as anchor for fan triangulation
    # Contact points are at midpoint, move to discrete surface (plane)
    p0 = points[0] + normal * (signed_distances[0] * 0.5)

    # Find the triangle with the largest area for numerical stability
    # This avoids issues with nearly collinear points
    best_normal = wp.vec3(0.0, 0.0, 0.0)
    max_area_sq = float(0.0)

    for i in range(1, count - 1):
        # Reconstruct plane points for this triangle
        pi = points[i] + normal * (signed_distances[i] * 0.5)
        pi_next = points[i + 1] + normal * (signed_distances[i + 1] * 0.5)

        # Compute cross product for triangle (p0, pi, pi_next)
        edge1 = pi - p0
        edge2 = pi_next - p0
        cross = wp.cross(edge1, edge2)
        area_sq = wp.dot(cross, cross)

        if area_sq > max_area_sq:
            max_area_sq = area_sq
            best_normal = cross

    # Normalize, avoid zero
    len_n = wp.sqrt(wp.max(1.0e-12, max_area_sq))
    plane_normal = best_normal / len_n

    # Ensure normal points in same direction as original normal
    if wp.dot(plane_normal, normal) < 0.0:
        plane_normal = -plane_normal

    return plane_normal


@wp.func
def no_post_process_contact(
    contact_data: ContactData,
    shape_a: GenericShapeData,
    pos_a_adjusted: wp.vec3,
    rot_a: wp.quat,
    shape_b: GenericShapeData,
    pos_b_adjusted: wp.vec3,
    rot_b: wp.quat,
) -> ContactData:
    return contact_data


@wp.func
def post_process_minkowski_only(
    contact_data: ContactData,
    shape_a: GenericShapeData,
    pos_a_adjusted: wp.vec3,
    rot_a: wp.quat,
    shape_b: GenericShapeData,
    pos_b_adjusted: wp.vec3,
    rot_b: wp.quat,
) -> ContactData:
    """Lean post-processor: Minkowski sphere/capsule adjustment only, no axial rolling."""
    type_a = shape_a.shape_type
    type_b = shape_b.shape_type
    normal = contact_data.contact_normal_a_to_b
    radius_eff_a = contact_data.radius_eff_a
    radius_eff_b = contact_data.radius_eff_b

    if type_a == GeoType.SPHERE or type_a == GeoType.CAPSULE:
        restored_a = radius_eff_a - MINKOWSKI_CORE_RADIUS
        contact_data.contact_point_center = contact_data.contact_point_center + normal * (restored_a * 0.5)
        contact_data.contact_distance = contact_data.contact_distance - restored_a

    if type_b == GeoType.SPHERE or type_b == GeoType.CAPSULE:
        restored_b = radius_eff_b - MINKOWSKI_CORE_RADIUS
        contact_data.contact_point_center = contact_data.contact_point_center - normal * (restored_b * 0.5)
        contact_data.contact_distance = contact_data.contact_distance - restored_b

    return contact_data


@wp.func
def post_process_axial_on_discrete_contact(
    contact_data: ContactData,
    shape_a: GenericShapeData,
    pos_a_adjusted: wp.vec3,
    rot_a: wp.quat,
    shape_b: GenericShapeData,
    pos_b_adjusted: wp.vec3,
    rot_b: wp.quat,
) -> ContactData:
    """
    Post-process a single contact for minkowski objects and axial shape rolling.

    This function handles:
    1. Minkowski objects (spheres/capsules): Adjusts contact point and distance for rounded geometry
    2. Axial shapes on discrete surfaces: Projects contact point for rolling stabilization

    Args:
        contact_data: Contact data to post-process
        shape_a: Shape data for shape A
        pos_a_adjusted: Position of shape A
        rot_a: Orientation of shape A
        shape_b: Shape data for shape B
        pos_b_adjusted: Position of shape B
        rot_b: Orientation of shape B

    Returns:
        Post-processed contact data
    """
    type_a = shape_a.shape_type
    type_b = shape_b.shape_type
    normal = contact_data.contact_normal_a_to_b
    radius_eff_a = contact_data.radius_eff_a
    radius_eff_b = contact_data.radius_eff_b

    # 1. Minkowski object processing for spheres and capsules
    # Adjust contact point and distance for sphere/capsule A
    if type_a == GeoType.SPHERE or type_a == GeoType.CAPSULE:
        restored_a = radius_eff_a - MINKOWSKI_CORE_RADIUS
        contact_data.contact_point_center = contact_data.contact_point_center + normal * (restored_a * 0.5)
        contact_data.contact_distance = contact_data.contact_distance - restored_a

    # Adjust contact point and distance for sphere/capsule B
    if type_b == GeoType.SPHERE or type_b == GeoType.CAPSULE:
        restored_b = radius_eff_b - MINKOWSKI_CORE_RADIUS
        contact_data.contact_point_center = contact_data.contact_point_center - normal * (restored_b * 0.5)
        contact_data.contact_distance = contact_data.contact_distance - restored_b

    # 2. Axial shape rolling stabilization (cylinders and cones on discrete surfaces)
    is_discrete_a = is_discrete_shape(type_a)
    is_discrete_b = is_discrete_shape(type_b)
    is_axial_a = type_a == GeoType.CYLINDER or type_a == GeoType.CONE
    is_axial_b = type_b == GeoType.CYLINDER or type_b == GeoType.CONE

    # Only process if we have discrete vs axial configuration
    if (is_discrete_a and is_axial_b) or (is_discrete_b and is_axial_a):
        # Extract the axial shape parameters
        if is_discrete_a and is_axial_b:
            shape_axis = wp.quat_rotate(rot_b, wp.vec3(0.0, 0.0, 1.0))
            shape_radius = shape_b.scale[0]
            shape_half_height = shape_b.scale[1]
            is_cone = type_b == GeoType.CONE
            shape_pos = pos_b_adjusted
            axial_normal = normal
        else:  # is_discrete_b and is_axial_a
            shape_axis = wp.quat_rotate(rot_a, wp.vec3(0.0, 0.0, 1.0))
            shape_radius = shape_a.scale[0]
            shape_half_height = shape_a.scale[1]
            is_cone = type_a == GeoType.CONE
            shape_pos = pos_a_adjusted
            axial_normal = -normal  # Flip normal for shape A

        # Check if shape is in rolling configuration
        axis_normal_dot = wp.abs(wp.dot(shape_axis, axial_normal))

        # Compute threshold based on shape type
        is_rolling = False
        if is_cone:
            # For a cone rolling on its base, the axis makes an angle with the normal
            cone_half_angle = wp.atan2(shape_radius, 2.0 * shape_half_height)
            tolerance_angle = wp.static(2.0 * wp.pi / 180.0)  # 2 degrees
            lower_threshold = wp.sin(cone_half_angle - tolerance_angle)
            upper_threshold = wp.sin(cone_half_angle + tolerance_angle)

            if axis_normal_dot >= lower_threshold and axis_normal_dot <= upper_threshold:
                is_rolling = True
        else:
            # For cylinder: axis should be perpendicular to normal (dot product ≈ 0)
            perpendicular_threshold = wp.static(math.sin(2.0 * math.pi / 180.0))
            if axis_normal_dot <= perpendicular_threshold:
                is_rolling = True

        # If rolling, project contact point onto the projection plane
        if is_rolling:
            projection_plane_normal = wp.normalize(wp.cross(shape_axis, axial_normal))
            point_on_projection_plane = shape_pos

            # Project the contact point
            projected_point = project_point_onto_plane(
                contact_data.contact_point_center, point_on_projection_plane, projection_plane_normal
            )

            # Update the contact with the projected point
            contact_data.contact_point_center = projected_point

    return contact_data


def create_compute_gjk_mpr_contacts(
    writer_func: Any,
    post_process_contact: Any = post_process_axial_on_discrete_contact,
    support_func: Any = None,
    use_precomputed_center: bool = False,
    penetration_refiner: Any = None,
    query_filter: Any = None,
    point_filter: Any = None,
    manifold_shape: Any = None,
    deepest_once: Any = None,
):
    """
    Factory function to create a compute_gjk_mpr_contacts function with a specific writer function.

    Args:
        writer_func: Function to write contact data (signature: (ContactData, writer_data) -> None)
        post_process_contact: Function to post-process contact data
        support_func: Support mapping function (defaults to support_map)
        use_precomputed_center: Whether the geometry data supplies a cached center.
        penetration_refiner: Optional physical-proxy result refinement function.
        query_filter: Optional surface witness of nonnegative-distance queries (see
            :func:`create_solve_convex_multi_contact`).
        point_filter: Optional admission of each written contact, ``(ContactData, query_context) -> bool``.
        manifold_shape: Optional manifold geometry of shape A (see :func:`create_solve_convex_multi_contact`).
        deepest_once: Optional ``(context) -> bool``: the manifold's deepest point is not written twice (see
            :func:`create_solve_convex_multi_contact`).

    Returns:
        A compute_gjk_mpr_contacts function with the writer function baked in
    """
    if support_func is None:
        support_func = support_map
    has_point_filter = point_filter is not None

    @wp.func
    def compute_gjk_mpr_contacts(
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
        data_provider: Any,
        writer_data: Any,
        sort_sub_key: int = 0,
        query_context: Any = 0,
    ):
        """
        Compute contacts between two shapes using GJK/MPR algorithm and write them.

        Args:
            shape_a_data: Generic shape data for shape A (contains shape_type)
            shape_b_data: Generic shape data for shape B (contains shape_type)
            rot_a: Orientation of shape A
            rot_b: Orientation of shape B
            pos_a_adjusted: Adjusted position of shape A
            pos_b_adjusted: Adjusted position of shape B
            rigid_gap: Contact gap for rigid bodies
            shape_a: Index of shape A
            shape_b: Index of shape B
            margin_a: Per-shape margin offset for shape A (signed distance padding)
            margin_b: Per-shape margin offset for shape B (signed distance padding)
            data_provider: Support-map data provider passed to the configured
                support function. Accelerated providers carry cooked convex
                directional seeds and vertex adjacency.
            writer_data: Data structure for contact writer
            sort_sub_key: Sub-key for deterministic contact sorting (e.g. triangle/edge index)
            query_context: Current geometry metadata for the optional query filter.
        """
        if shape_a_data.shape_type == int(GeoTypeEx.TRIANGLE):
            face_normal = wp.quat_rotate(rot_a, wp.cross(shape_a_data.scale, shape_a_data.auxiliary))
            face_length = wp.length(face_normal)
            if face_length > 1.0e-12:
                face_normal /= face_length
                # Shape B's interior point: a cached center (a hull's interior point), else its origin.
                center_b = pos_b_adjusted
                if wp.static(use_precomputed_center):
                    center_b = pos_b_adjusted + wp.quat_rotate(rot_b, shape_b_data.center)
                if wp.dot(center_b - pos_a_adjusted, face_normal) < 0.0:
                    # Only watertight nearest-face overlaps survive midphase with
                    # the center behind a triangle. GJK would reverse the normal;
                    # retain the authored outward face and its physical depth.
                    support_b = support_func(shape_b_data, wp.quat_rotate_inv(rot_b, -face_normal), data_provider)
                    point_b = wp.quat_rotate(rot_b, support_b) + pos_b_adjusted
                    distance = wp.dot(point_b - pos_a_adjusted, face_normal)
                    point_a = point_b - distance * face_normal
                    projection = wp.quat_rotate_inv(rot_a, point_a - pos_a_adjusted)
                    closest = closest_point_on_triangle(
                        projection, wp.vec3(0.0), shape_a_data.scale, shape_a_data.auxiliary
                    )
                    if wp.length_sq(projection - closest) < 1.0e-10:
                        contact = ContactData()
                        contact.contact_point_center = 0.5 * (point_a + point_b)
                        contact.contact_normal_a_to_b = face_normal
                        contact.contact_distance = distance
                        contact.margin_a = margin_a
                        contact.margin_b = margin_b
                        contact.shape_a = shape_a
                        contact.shape_b = shape_b
                        contact.gap_sum = rigid_gap
                        contact.sort_sub_key = sort_sub_key
                        if wp.static(has_point_filter):
                            if point_filter(contact, query_context):
                                writer_func(contact, writer_data, -1)
                        else:
                            writer_func(contact, writer_data, -1)
                    return

        radius_eff_a = float(0.0)
        radius_eff_b = float(0.0)

        # Get shape types from shape data
        type_a = shape_a_data.shape_type
        type_b = shape_b_data.shape_type

        # Special treatment for minkowski objects
        if type_a == GeoType.SPHERE or type_a == GeoType.CAPSULE:
            radius_eff_a = shape_a_data.scale[0]
            shape_a_data.scale[0] = MINKOWSKI_CORE_RADIUS

        if type_b == GeoType.SPHERE or type_b == GeoType.CAPSULE:
            radius_eff_b = shape_b_data.scale[0]
            shape_b_data.scale[0] = MINKOWSKI_CORE_RADIUS

        # Pre-pack ContactData template with static information
        contact_template = ContactData()
        contact_template.radius_eff_a = radius_eff_a
        contact_template.radius_eff_b = radius_eff_b
        contact_template.margin_a = margin_a
        contact_template.margin_b = margin_b
        contact_template.shape_a = shape_a
        contact_template.shape_b = shape_b
        contact_template.gap_sum = rigid_gap
        contact_template.sort_sub_key = sort_sub_key

        if wp.static(ENABLE_MULTI_CONTACT):
            wp.static(
                create_solve_convex_multi_contact(
                    support_func,
                    writer_func,
                    post_process_contact,
                    use_precomputed_center,
                    penetration_refiner,
                    query_filter,
                    point_filter,
                    manifold_shape,
                    deepest_once,
                )
            )(
                shape_a_data,
                shape_b_data,
                rot_a,
                rot_b,
                pos_a_adjusted,
                pos_b_adjusted,
                data_provider,
                rigid_gap + radius_eff_a + radius_eff_b + margin_a + margin_b,
                type_a == GeoType.SPHERE
                or type_b == GeoType.SPHERE
                or type_a == GeoType.ELLIPSOID
                or type_b == GeoType.ELLIPSOID,
                writer_data,
                contact_template,
                query_context,
            )
        else:
            wp.static(
                create_solve_convex_single_contact(
                    support_func,
                    writer_func,
                    post_process_contact,
                    use_precomputed_center,
                    penetration_refiner,
                    query_filter,
                    point_filter,
                )
            )(
                shape_a_data,
                shape_b_data,
                rot_a,
                rot_b,
                pos_a_adjusted,
                pos_b_adjusted,
                data_provider,
                rigid_gap + radius_eff_a + radius_eff_b + margin_a + margin_b,
                writer_data,
                contact_template,
                query_context,
            )

    return compute_gjk_mpr_contacts


FACE_CASE_BAND = 4.0e-6
"""Relative band [-] by which a support point's foot must lie inside its triangle for the closed-form face case.

A foot within the band of an edge, or on it, goes to GJK/MPR, which decides edge contacts with the feature rule
(as the rule's own classification of manifold points does, with the same world-coordinate rounding allowance).
"""


@wp.func
def triangle_face_case(
    triangle: GenericShapeData,
    origin: wp.vec3,
    support_point: wp.vec3,
    center: wp.vec3,
    threshold: float,
) -> tuple[bool, wp.vec3, float]:
    """Whether a convex shape's contact with a mesh triangle is the face's, decided in closed form.

    ``support_point`` is the shape's world support point s along minus the face normal n, ``origin`` the triangle's
    first world corner p0 and d = n . (s - p0). With s's foot inside the triangle at in-plane distance delta from its
    edges:

    - separated (d >= 0): dist(shape, triangle) = d and (s, its foot) is a closest pair, since every point x of the
      shape has n . (x - p0) >= d and |x - y| >= n . (x - y) for every point y of the plane;
    - penetrating (d < 0) with delta >= |d|: |d| along n is the prism's minimum translation (any separating
      translation must move s out of the prism, which takes at least min(|d|, delta), and |d| n separates).

    Everything else is not decided here: a foot outside the triangle or within ``FACE_CASE_BAND`` of an edge (an edge
    or vertex witness), a deep thin overlap (delta < |d|), a distance beyond ``threshold``, and a shape whose
    interior point lies behind the face (the GJK/MPR query's own one-sided case).

    Returns:
        ``(face, unit, distance)``: whether the face case holds, the outward unit face normal and d.
    """
    face = wp.cross(triangle.scale, triangle.auxiliary)
    length = wp.length(face)
    if length <= 1.0e-30:
        return False, wp.vec3(0.0), 0.0
    unit = face / length
    distance = wp.dot(support_point - origin, unit)
    if distance > threshold:
        return False, unit, distance
    if wp.dot(center - origin, unit) < 0.0:
        return False, unit, distance
    foot = support_point - distance * unit - origin
    band = FACE_CASE_BAND * (
        1.0 + wp.max(wp.abs(support_point[0]), wp.max(wp.abs(support_point[1]), wp.abs(support_point[2])))
    )
    c1 = triangle.scale
    c2 = triangle.auxiliary
    # Inward in-plane distances to the edges p0p1, p1p2 and p2p0 (the corners run counterclockwise about n).
    delta = wp.dot(foot, wp.normalize(wp.cross(unit, c1)))
    delta = wp.min(delta, wp.dot(foot - c1, wp.normalize(wp.cross(unit, c2 - c1))))
    delta = wp.min(delta, wp.dot(foot - c2, wp.normalize(wp.cross(unit, -c2))))
    if delta <= band:
        return False, unit, distance
    if distance < 0.0 and delta < -distance:
        return False, unit, distance
    return True, unit, distance


def create_compute_face_contacts(
    writer_func: Any,
    support_func: Any,
    point_filter: Any,
    manifold_shape: Any,
    post_process_contact: Any = post_process_axial_on_discrete_contact,
    deepest_once: Any = None,
):
    """Create the face case of a mesh triangle and a convex shape: the face-normal manifold, without GJK/MPR.

    Where :func:`triangle_face_case` holds, GJK/MPR's witness is the support point and its foot, along the face
    normal (up to rounding), so the manifold built here is the one the GJK/MPR query builds after it: the shape's
    support feature clipped against the triangle (grown across coplanar edges by ``manifold_shape``), each point
    admitted by ``point_filter`` (a face contact needs no cone test: the triangle containing the point owns it).

    Args:
        writer_func: Function that writes generated contacts.
        support_func: Support mapping function (as the GJK/MPR query's).
        point_filter: Admission of each written contact, ``(ContactData, context) -> bool``.
        manifold_shape: The manifold's triangle, ``(geom_a, point_a, normal, context) -> (geom, origin, moved)``.
        post_process_contact: Function that post-processes generated contacts.
        deepest_once: Optional ``(context) -> bool``: the manifold's deepest point is not written twice.
    """
    support_funcs = create_support_map_function(support_func, True)
    build_manifold = create_build_manifold(
        support_func,
        writer_func,
        post_process_contact,
        _support_funcs=support_funcs,
        point_filter=point_filter,
        deepest_once=deepest_once,
    )

    @wp.func
    def compute_face_contacts(
        triangle: GenericShapeData,
        geom_b: GenericShapeData,
        rot_b: wp.quat,
        pos_a: wp.vec3,
        pos_b: wp.vec3,
        unit: wp.vec3,
        support_point: wp.vec3,
        distance: float,
        rigid_gap: float,
        shape_a: int,
        shape_b: int,
        margin_a: float,
        margin_b: float,
        data_provider: Any,
        writer_data: Any,
        sort_sub_key: int,
        query_context: Any,
    ) -> int:
        """Write the face case's manifold: ``triangle`` in the world's orientation with its first corner at ``pos_a``."""
        template = ContactData()
        template.margin_a = margin_a
        template.margin_b = margin_b
        template.shape_a = shape_a
        template.shape_b = shape_b
        template.gap_sum = rigid_gap
        template.sort_sub_key = sort_sub_key
        relative_b = pos_b - pos_a
        point_b = support_point - pos_a
        point_a = point_b - distance * unit
        manifold_a = triangle
        manifold_origin = pos_a
        grown, shift, moved = manifold_shape(triangle, point_a, unit, query_context)
        if moved:
            manifold_a = grown
            manifold_origin = pos_a + shift
            relative_b = relative_b - shift
            point_a = point_a - shift
            point_b = point_b - shift
        return build_manifold(
            manifold_a,
            geom_b,
            wp.quat_identity(),
            manifold_origin,
            rot_b,
            relative_b,
            point_a,
            point_b,
            unit,
            data_provider,
            writer_data,
            template,
            query_context,
        )

    return compute_face_contacts


@wp.func
def compute_tight_aabb_from_support(
    shape_data: GenericShapeData,
    orientation: wp.quat,
    center_pos: wp.vec3,
    data_provider: Any,
) -> tuple[wp.vec3, wp.vec3]:
    """
    Compute tight AABB for a shape using support function.

    Args:
        shape_data: Generic shape data
        orientation: Shape orientation (quaternion)
        center_pos: Center position of the shape
        data_provider: Support map data provider

    Returns:
        Tuple of (aabb_min, aabb_max) in world space
    """
    # Transpose orientation matrix to transform world axes to local space
    # Convert quaternion to 3x3 rotation matrix and transpose (inverse rotation)
    rot_mat = wp.quat_to_matrix(orientation)
    rot_mat_t = wp.transpose(rot_mat)

    # Transform world axes to local space (multiply by transposed rotation = inverse rotation)
    local_x = wp.vec3(rot_mat_t[0, 0], rot_mat_t[1, 0], rot_mat_t[2, 0])
    local_y = wp.vec3(rot_mat_t[0, 1], rot_mat_t[1, 1], rot_mat_t[2, 1])
    local_z = wp.vec3(rot_mat_t[0, 2], rot_mat_t[1, 2], rot_mat_t[2, 2])

    # Compute AABB extents by evaluating support function in local space
    # Dot products are done in local space to avoid expensive rotations

    min_x = float(0.0)
    max_x = float(0.0)
    min_y = float(0.0)
    max_y = float(0.0)
    min_z = float(0.0)
    max_z = float(0.0)

    if shape_data.shape_type == GeoType.CONVEX_MESH:
        # Single-pass AABB: iterate over vertices once, project onto all 3 axes.
        # This replaces 6 separate support_map calls (each iterating all vertices)
        # with 1 pass that computes min/max projections simultaneously.
        mesh_ptr = unpack_mesh_ptr(shape_data.auxiliary)
        mesh = wp.mesh_get(mesh_ptr)
        mesh_scale = shape_data.scale
        num_verts = mesh.points.shape[0]

        # Pre-scale axes: dot(local_axis, scale*v) == dot(scale*local_axis, v)
        scaled_x = wp.cw_mul(local_x, mesh_scale)
        scaled_y = wp.cw_mul(local_y, mesh_scale)
        scaled_z = wp.cw_mul(local_z, mesh_scale)

        min_x = float(1.0e10)
        max_x = float(-1.0e10)
        min_y = float(1.0e10)
        max_y = float(-1.0e10)
        min_z = float(1.0e10)
        max_z = float(-1.0e10)

        for i in range(num_verts):
            p = mesh.points[i]
            vx = wp.dot(p, scaled_x)
            vy = wp.dot(p, scaled_y)
            vz = wp.dot(p, scaled_z)
            min_x = wp.min(min_x, vx)
            max_x = wp.max(max_x, vx)
            min_y = wp.min(min_y, vy)
            max_y = wp.max(max_y, vy)
            min_z = wp.min(min_z, vz)
            max_z = wp.max(max_z, vz)
    else:
        # Generic path: 6 support evaluations for other shape types (all O(1))
        support_point = support_map(shape_data, local_x, data_provider)
        max_x = wp.dot(local_x, support_point)

        support_point = support_map(shape_data, local_y, data_provider)
        max_y = wp.dot(local_y, support_point)

        support_point = support_map(shape_data, local_z, data_provider)
        max_z = wp.dot(local_z, support_point)

        support_point = support_map(shape_data, -local_x, data_provider)
        min_x = wp.dot(local_x, support_point)

        support_point = support_map(shape_data, -local_y, data_provider)
        min_y = wp.dot(local_y, support_point)

        support_point = support_map(shape_data, -local_z, data_provider)
        min_z = wp.dot(local_z, support_point)

    # AABB in world space (add world position to extents)
    aabb_min = wp.vec3(min_x, min_y, min_z) + center_pos
    aabb_max = wp.vec3(max_x, max_y, max_z) + center_pos

    return aabb_min, aabb_max


@wp.func
def compute_bounding_sphere_from_aabb(aabb_lower: wp.vec3, aabb_upper: wp.vec3) -> tuple[wp.vec3, float]:
    """
    Compute a bounding sphere from an AABB.

    Returns:
        Tuple of (center, radius) where center is the AABB center and radius is half the diagonal.
    """
    center = 0.5 * (aabb_lower + aabb_upper)
    half_extents = 0.5 * (aabb_upper - aabb_lower)
    radius = wp.length(half_extents)
    return center, radius


@wp.func
def convert_infinite_plane_to_cube(
    shape_data: GenericShapeData,
    plane_rotation: wp.quat,
    plane_position: wp.vec3,
    other_position: wp.vec3,
    other_radius: float,
) -> tuple[GenericShapeData, wp.vec3]:
    """
    Convert an infinite plane into a cube proxy for GJK/MPR collision detection.

    Since GJK/MPR cannot handle infinite planes, we create a finite cube where:
    - The cube is positioned with its top face at the plane surface
    - The cube's lateral dimensions are sized based on the other object's bounding sphere
    - The cube extends only 'downward' from the plane (half-space in -Z direction in plane's local frame)

    Args:
        shape_data: The plane's shape data (should have shape_type == GeoType.PLANE)
        plane_rotation: The plane's orientation (plane normal is along local +Z)
        plane_position: The plane's position in world space
        other_position: The other object's position in world space
        other_radius: Bounding sphere radius of the colliding object

    Returns:
        Tuple of (modified_shape_data, adjusted_position):
        - modified_shape_data: GenericShapeData configured as a BOX
        - adjusted_position: The cube's center position (centered on other object projected to plane)
    """
    result = GenericShapeData()
    result.shape_type = GeoType.BOX

    # Size the cube based on the other object's bounding sphere radius
    # Make it large enough to always contain potential contact points
    # The lateral dimensions (x, y) should be at least 2x the radius to ensure coverage
    lateral_size = other_radius * 10.0

    # The depth (z) should be large enough to encompass the potential collision region
    # Half-space behavior: cube extends only below the plane surface (negative Z)
    depth = other_radius * 10.0

    # Set the box half-extents
    # x, y: lateral coverage (parallel to plane)
    # z: depth perpendicular to plane
    result.scale = wp.vec3(lateral_size, lateral_size, depth)
    result.auxiliary = wp.vec3(0.0, 0.0, 0.0)
    result.center = wp.vec3(0.0, 0.0, 0.0)

    # Position the cube center at the plane surface, directly under/over the other object
    # Project the other object's position onto the plane
    plane_normal = wp.quat_rotate(plane_rotation, wp.vec3(0.0, 0.0, 1.0))
    to_other = other_position - plane_position
    distance_along_normal = wp.dot(to_other, plane_normal)

    # Point on plane surface closest to the other object
    plane_surface_point = other_position - plane_normal * distance_along_normal

    # Position cube center slightly below the plane surface so the top face is at the surface
    # Since the cube has half-extent 'depth', its top face is at center + depth*normal
    # We want: center + depth*normal = plane_surface, so center = plane_surface - depth*normal
    adjusted_position = plane_surface_point - plane_normal * depth

    return result, adjusted_position


@wp.func
def check_infinite_plane_bsphere_overlap(
    shape_data_a: GenericShapeData,
    shape_data_b: GenericShapeData,
    pos_a: wp.vec3,
    pos_b: wp.vec3,
    quat_a: wp.quat,
    quat_b: wp.quat,
    bsphere_center_a: wp.vec3,
    bsphere_center_b: wp.vec3,
    bsphere_radius_a: float,
    bsphere_radius_b: float,
) -> bool:
    """
    Check if an infinite plane overlaps with another shape's bounding sphere.
    Treats the plane as a half-space: objects on or below the plane (negative side of the normal)
    are considered to overlap and will generate contacts.
    Returns True if they overlap, False otherwise.
    Uses data already extracted by extract_shape_data.
    """
    type_a = shape_data_a.shape_type
    type_b = shape_data_b.shape_type
    scale_a = shape_data_a.scale
    scale_b = shape_data_b.scale

    # Check if either shape is an infinite plane
    is_infinite_plane_a = (type_a == GeoType.PLANE) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
    is_infinite_plane_b = (type_b == GeoType.PLANE) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

    # If neither is an infinite plane, return True (no culling)
    if not (is_infinite_plane_a or is_infinite_plane_b):
        return True

    # Determine which is the plane and which is the other shape
    if is_infinite_plane_a:
        plane_pos = pos_a
        plane_quat = quat_a
        other_center = bsphere_center_b
        other_radius = bsphere_radius_b
    else:
        plane_pos = pos_b
        plane_quat = quat_b
        other_center = bsphere_center_a
        other_radius = bsphere_radius_a

    # Compute plane normal (plane's local +Z axis in world space)
    plane_normal = wp.quat_rotate(plane_quat, wp.vec3(0.0, 0.0, 1.0))

    # Distance from sphere center to plane (positive = above plane, negative = below plane)
    center_dist = wp.dot(other_center - plane_pos, plane_normal)

    # Treat plane as a half-space: objects on or below the plane (negative side) generate contacts
    # Remove absolute value to only check penetration side
    return center_dist <= other_radius


def create_find_contacts(writer_func: Any, support_func: Any = None, post_process_contact: Any = None):
    """
    Factory function to create a find_contacts function with a specific writer function.

    The generated function uses precomputed centers. Callers must populate
    :attr:`GenericShapeData.center` for every ``CONVEX_MESH`` shape before
    invoking it.

    Args:
        writer_func: Function to write contact data (signature: (ContactData, writer_data) -> None)
        support_func: Support mapping function (defaults to support_map)
        post_process_contact: Post-processing function (defaults to post_process_axial_on_discrete_contact)

    Returns:
        A find_contacts function with the writer function baked in
    """
    if support_func is None:
        support_func = support_map
    if post_process_contact is None:
        post_process_contact = post_process_axial_on_discrete_contact

    plane_query = create_plane_convex_query(support_func)
    # Keep the full support-map kernel's instruction footprint unchanged.
    use_plane_convex_query = support_func in (support_map_lean, support_map_lean_accelerated)
    write_query = create_write_convex_query_result(
        support_func, writer_func, post_process_contact, use_precomputed_center=True
    )

    @wp.func
    def find_contacts(
        pos_a: wp.vec3,
        pos_b: wp.vec3,
        quat_a: wp.quat,
        quat_b: wp.quat,
        shape_data_a: GenericShapeData,
        shape_data_b: GenericShapeData,
        is_infinite_plane_a: bool,
        is_infinite_plane_b: bool,
        bsphere_radius_a: float,
        bsphere_radius_b: float,
        rigid_gap: float,
        shape_a: int,
        shape_b: int,
        margin_a: float,
        margin_b: float,
        data_provider: Any,
        writer_data: Any,
    ):
        """
        Find contacts between two shapes using GJK/MPR algorithm and write them using the writer function.

        Args:
            pos_a: Position of shape A in world space
            pos_b: Position of shape B in world space
            quat_a: Orientation of shape A
            quat_b: Orientation of shape B
            shape_data_a: Generic shape data for shape A (contains shape_type)
            shape_data_b: Generic shape data for shape B (contains shape_type)
            is_infinite_plane_a: Whether shape A is an infinite plane
            is_infinite_plane_b: Whether shape B is an infinite plane
            bsphere_radius_a: Bounding sphere radius of shape A
            bsphere_radius_b: Bounding sphere radius of shape B
            rigid_gap: Contact gap for rigid bodies
            shape_a: Index of shape A
            shape_b: Index of shape B
            margin_a: Per-shape margin offset for shape A (signed distance padding)
            margin_b: Per-shape margin offset for shape B (signed distance padding)
            data_provider: Support-map data provider passed to the configured
                support function. Accelerated providers carry cooked convex
                directional seeds and vertex adjacency.
            writer_data: Data structure for contact writer
        """
        if writer_data.contact_count[0] >= writer_data.contact_max:
            return

        # Convert infinite planes to cube proxies for GJK/MPR compatibility
        # Use the OTHER object's radius to properly size the cube
        # Only convert if it's an infinite plane (finite planes can be handled normally)
        pos_a_adjusted = pos_a
        if is_infinite_plane_a:
            # Position the cube based on the OTHER object's position (pos_b)
            # Note: convert_infinite_plane_to_cube modifies shape_data_a.shape_type to BOX
            shape_data_a, pos_a_adjusted = convert_infinite_plane_to_cube(
                shape_data_a, quat_a, pos_a, pos_b, bsphere_radius_b + rigid_gap
            )

            if wp.static(use_plane_convex_query):
                if shape_data_b.shape_type == GeoType.CONVEX_MESH:
                    # An infinite plane has a known normal; only the hull support point is needed.
                    result = wp.static(plane_query)(
                        shape_data_b, quat_a, quat_b, pos_a_adjusted, pos_b, pos_a, data_provider
                    )
                    if result.signed_distance <= rigid_gap + margin_a + margin_b:
                        contact = ContactData()
                        contact.shape_a = shape_a
                        contact.shape_b = shape_b
                        contact.margin_a = margin_a
                        contact.margin_b = margin_b
                        contact.gap_sum = rigid_gap
                        wp.static(write_query)(
                            shape_data_a,
                            shape_data_b,
                            quat_a,
                            quat_b,
                            pos_a_adjusted,
                            pos_b,
                            wp.quat_inverse(quat_a) * quat_b,
                            wp.quat_rotate_inv(quat_a, pos_b - pos_a_adjusted),
                            result,
                            data_provider,
                            wp.static(not ENABLE_MULTI_CONTACT),
                            writer_data,
                            contact,
                        )
                    return

        pos_b_adjusted = pos_b
        if is_infinite_plane_b:
            # Position the cube based on the OTHER object's position (pos_a)
            # Note: convert_infinite_plane_to_cube modifies shape_data_b.shape_type to BOX
            shape_data_b, pos_b_adjusted = convert_infinite_plane_to_cube(
                shape_data_b, quat_b, pos_b, pos_a, bsphere_radius_a + rigid_gap
            )

        # Compute and write contacts using GJK/MPR
        wp.static(
            create_compute_gjk_mpr_contacts(
                writer_func,
                post_process_contact=post_process_contact,
                support_func=support_func,
                use_precomputed_center=True,
            )
        )(
            shape_data_a,
            shape_data_b,
            quat_a,
            quat_b,
            pos_a_adjusted,
            pos_b_adjusted,
            rigid_gap,
            shape_a,
            shape_b,
            margin_a,
            margin_b,
            data_provider,
            writer_data,
        )

    return find_contacts


@wp.func
def pre_contact_check(
    shape_a: int,
    shape_b: int,
    pos_a: wp.vec3,
    pos_b: wp.vec3,
    quat_a: wp.quat,
    quat_b: wp.quat,
    shape_data_a: GenericShapeData,
    shape_data_b: GenericShapeData,
    aabb_a_lower: wp.vec3,
    aabb_a_upper: wp.vec3,
    aabb_b_lower: wp.vec3,
    aabb_b_upper: wp.vec3,
    pair: wp.vec2i,
    mesh_id_a: wp.uint64,
    mesh_id_b: wp.uint64,
    shape_pairs_mesh: wp.array[wp.vec2i],
    shape_pairs_mesh_count: wp.array[int],
    shape_pairs_mesh_plane: wp.array[wp.vec2i],
    shape_pairs_mesh_plane_cumsum: wp.array[int],
    shape_pairs_mesh_plane_count: wp.array[int],
    mesh_plane_vertex_total_count: wp.array[int],
    shape_pairs_mesh_mesh: wp.array[wp.vec2i],
    shape_pairs_mesh_mesh_count: wp.array[int],
):
    """
    Perform pre-contact checks for early rejection and special case handling.

    Args:
        shape_a: Index of shape A
        shape_b: Index of shape B
        pos_a: Position of shape A in world space
        pos_b: Position of shape B in world space
        quat_a: Orientation of shape A
        quat_b: Orientation of shape B
        shape_data_a: Generic shape data for shape A (contains shape_type and scale)
        shape_data_b: Generic shape data for shape B (contains shape_type and scale)
        aabb_a_lower: Lower bound of AABB for shape A
        aabb_a_upper: Upper bound of AABB for shape A
        aabb_b_lower: Lower bound of AABB for shape B
        aabb_b_upper: Upper bound of AABB for shape B
        pair: Shape pair indices
        mesh_id_a: Mesh ID pointer for shape A (wp.uint64(0) if not a mesh)
        mesh_id_b: Mesh ID pointer for shape B (wp.uint64(0) if not a mesh)
        shape_pairs_mesh: Output array for mesh collision pairs
        shape_pairs_mesh_count: Counter for mesh collision pairs
        shape_pairs_mesh_plane: Output array for mesh-plane collision pairs
        shape_pairs_mesh_plane_cumsum: Cumulative sum array for mesh-plane vertices
        shape_pairs_mesh_plane_count: Counter for mesh-plane collision pairs
        mesh_plane_vertex_total_count: Total vertex count for mesh-plane collisions
        shape_pairs_mesh_mesh: Output array for mesh-mesh collision pairs
        shape_pairs_mesh_mesh_count: Counter for mesh-mesh collision pairs

    Returns:
        Tuple of (skip_pair, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b)
    """
    # Get shape types from shape data
    type_a = shape_data_a.shape_type
    type_b = shape_data_b.shape_type

    # Check if shapes are infinite planes (scale.x == 0 and scale.y == 0)
    # Scale is already in shape_data, no need for array lookup
    is_infinite_plane_a = (type_a == GeoType.PLANE) and (shape_data_a.scale[0] == 0.0 and shape_data_a.scale[1] == 0.0)
    is_infinite_plane_b = (type_b == GeoType.PLANE) and (shape_data_b.scale[0] == 0.0 and shape_data_b.scale[1] == 0.0)

    # Early return: both shapes are infinite planes
    if is_infinite_plane_a and is_infinite_plane_b:
        return True, is_infinite_plane_a, is_infinite_plane_b, float(0.0), float(0.0)

    # Compute bounding spheres from AABBs instead of using mesh bounding spheres
    bsphere_center_a, bsphere_radius_a = compute_bounding_sphere_from_aabb(aabb_a_lower, aabb_a_upper)
    bsphere_center_b, bsphere_radius_b = compute_bounding_sphere_from_aabb(aabb_b_lower, aabb_b_upper)

    # Check if infinite plane vs bounding sphere overlap - early rejection
    if not check_infinite_plane_bsphere_overlap(
        shape_data_a,
        shape_data_b,
        pos_a,
        pos_b,
        quat_a,
        quat_b,
        bsphere_center_a,
        bsphere_center_b,
        bsphere_radius_a,
        bsphere_radius_b,
    ):
        return True, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b

    # Check for mesh vs infinite plane collision - special handling
    # After sorting, type_a <= type_b, so we only need to check one direction
    if type_a == GeoType.PLANE and type_b == GeoType.MESH:
        # Check if plane is infinite (scale x and y are zero) - use scale from shape_data
        if shape_data_a.scale[0] == 0.0 and shape_data_a.scale[1] == 0.0:
            # Get mesh vertex count using the provided mesh_id
            if mesh_id_b != wp.uint64(0):
                mesh_obj = wp.mesh_get(mesh_id_b)
                vertex_count = mesh_obj.points.shape[0]

                # Add to mesh-plane collision buffer with cumulative vertex count
                mesh_plane_idx = wp.atomic_add(shape_pairs_mesh_plane_count, 0, 1)
                if mesh_plane_idx < shape_pairs_mesh_plane.shape[0]:
                    # Store shape indices (mesh, plane)
                    shape_pairs_mesh_plane[mesh_plane_idx] = wp.vec2i(shape_b, shape_a)
                    # Store inclusive cumulative vertex count in separate array for better cache locality
                    cumulative_count_before = wp.atomic_add(mesh_plane_vertex_total_count, 0, vertex_count)
                    cumulative_count_inclusive = cumulative_count_before + vertex_count
                    shape_pairs_mesh_plane_cumsum[mesh_plane_idx] = cumulative_count_inclusive
            return True, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b

    # Check for mesh-mesh collisions - add to separate buffer for specialized handling
    if type_a == GeoType.MESH and type_b == GeoType.MESH:
        # Add to mesh-mesh collision buffer using atomic counter
        mesh_mesh_pair_idx = wp.atomic_add(shape_pairs_mesh_mesh_count, 0, 1)
        if mesh_mesh_pair_idx < shape_pairs_mesh_mesh.shape[0]:
            shape_pairs_mesh_mesh[mesh_mesh_pair_idx] = pair
        return True, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b

    # Check for other mesh collisions (mesh vs non-mesh) - add to separate buffer for specialized handling
    if type_a == GeoType.MESH or type_b == GeoType.MESH:
        # Add to mesh collision buffer using atomic counter
        mesh_pair_idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)
        if mesh_pair_idx < shape_pairs_mesh.shape[0]:
            shape_pairs_mesh[mesh_pair_idx] = pair
        return True, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b

    return False, is_infinite_plane_a, is_infinite_plane_b, bsphere_radius_a, bsphere_radius_b


@wp.func
def aabb_to_unscaled(
    aabb_lower: wp.vec3,
    aabb_upper: wp.vec3,
    scale: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Convert an axis-aligned bounding box from scaled local space to unscaled local space.

    Given an AABB ``[aabb_lower, aabb_upper]`` expressed in a frame where geometry has been
    pre-multiplied component-wise by ``scale``, return the equivalent AABB in the unscaled
    frame (i.e. divided component-wise). Negative scale components flip the axis, so per-axis
    min/max are swapped to keep ``lower <= upper``. Zero/near-zero components are guarded with
    a small epsilon, but in practice ``scale`` should be non-zero whenever this is called.

    Returns:
        The unscaled lower bounds, unscaled upper bounds, and component-wise
        inverse scale. Zero or near-zero scale components use the epsilon-guarded
        reciprocal applied by this function.
    """
    eps = float(1.0e-12)
    inv_x = 1.0 / wp.where(wp.abs(scale[0]) > eps, scale[0], wp.where(scale[0] >= 0.0, eps, -eps))
    inv_y = 1.0 / wp.where(wp.abs(scale[1]) > eps, scale[1], wp.where(scale[1] >= 0.0, eps, -eps))
    inv_z = 1.0 / wp.where(wp.abs(scale[2]) > eps, scale[2], wp.where(scale[2] >= 0.0, eps, -eps))

    lx0 = aabb_lower[0] * inv_x
    lx1 = aabb_upper[0] * inv_x
    ly0 = aabb_lower[1] * inv_y
    ly1 = aabb_upper[1] * inv_y
    lz0 = aabb_lower[2] * inv_z
    lz1 = aabb_upper[2] * inv_z

    out_lower = wp.vec3(wp.min(lx0, lx1), wp.min(ly0, ly1), wp.min(lz0, lz1))
    out_upper = wp.vec3(wp.max(lx0, lx1), wp.max(ly0, ly1), wp.max(lz0, lz1))
    return out_lower, out_upper, wp.vec3(inv_x, inv_y, inv_z)


@wp.func
def transform_normal_with_scale(
    transform: wp.transform,
    scale: wp.vec3,
    normal_local: wp.vec3,
) -> wp.vec3:
    """Transform a unit normal from a (translated, rotated, component-wise scaled) local frame
    to world space.

    Under a non-uniform component-wise scale ``S = diag(scale)``, surface normals do **not**
    transform like vectors: the correct rule is ``n_world ∝ R · S^{-T} · n_local`` which, for a
    diagonal scale, reduces to ``R · (n_local / scale)``. The translation component of
    ``transform`` is irrelevant for normals. The returned normal is normalized; if the scaled
    normal is degenerate (zero length), the rotation-only transform of ``normal_local`` is
    returned as a fallback.

    This is the analog of ``wp.transform_vector`` for normals when the local frame includes a
    non-uniform scale (e.g. a triangle mesh shape with ``shape_data.scale = (sx, sy, sz)``).
    """
    eps = float(1.0e-12)
    sx = wp.where(wp.abs(scale[0]) > eps, scale[0], wp.where(scale[0] >= 0.0, eps, -eps))
    sy = wp.where(wp.abs(scale[1]) > eps, scale[1], wp.where(scale[1] >= 0.0, eps, -eps))
    sz = wp.where(wp.abs(scale[2]) > eps, scale[2], wp.where(scale[2] >= 0.0, eps, -eps))

    n_scaled = wp.vec3(normal_local[0] / sx, normal_local[1] / sy, normal_local[2] / sz)
    len_n = wp.length(n_scaled)
    if len_n > eps:
        n_scaled = n_scaled / len_n
    else:
        # Degenerate (e.g. a normal aligned with an axis collapsed to zero scale): fall
        # back to rotating the unscaled local normal so the result is still well-defined.
        n_scaled = normal_local

    return wp.transform_vector(transform, n_scaled)


@wp.func
def convex_mesh_interior_point(mesh_id: wp.uint64, scale: wp.vec3) -> wp.vec3:
    """Return the scaled volume centroid of a convex hull, which lies inside it.

    Unlike the vertex mean, the centroid does not depend on how densely each region
    of the hull is tessellated. Degenerate hulls fall back to the vertex mean.
    """
    hull = wp.mesh_get(mesh_id)
    num_verts = hull.points.shape[0]
    mean = wp.vec3(0.0, 0.0, 0.0)
    for i in range(num_verts):
        mean += hull.points[i]
    mean = mean / float(wp.max(num_verts, 1))
    # Fan the faces from the interior vertex mean into tetrahedra.
    volume = float(0.0)
    weighted = wp.vec3(0.0, 0.0, 0.0)
    for tri in range(hull.indices.shape[0] // 3):
        a = hull.points[hull.indices[3 * tri + 0]] - mean
        b = hull.points[hull.indices[3 * tri + 1]] - mean
        c = hull.points[hull.indices[3 * tri + 2]] - mean
        tet = wp.abs(wp.dot(a, wp.cross(b, c)))
        volume += tet
        weighted += tet * (a + b + c)
    center = mean
    if volume > 0.0:
        center = mean + weighted / (4.0 * volume)
    return wp.cw_mul(center, scale)


@wp.func
def _mesh_query_bound(
    convex: GenericShapeData,
    local_lower: wp.vec3,
    local_upper: wp.vec3,
    X_mesh_shape: wp.transform,
    contact_threshold: float,
) -> MeshQueryBound:
    """The convex shape's bound in the scaled mesh frame, inflated by ``contact_threshold``.

    The box is the shape's own local box ``[local_lower, local_upper]`` (:func:`mesh_vs_convex_local_box`) placed in
    the mesh frame. Shapes with a
    symmetry axis also carry the segment-plus-radius that contains them: a sphere's center, a capsule's core, and the
    axis of a cylinder or cone with its largest radius. A contact needs a triangle within the threshold of the shape,
    hence inside both. A small relative pad covers float32 rounding; it only admits more triangles.
    """
    rot = wp.quat_to_matrix(wp.transform_get_rotation(X_mesh_shape))
    pos = wp.transform_get_translation(X_mesh_shape)
    local_center = 0.5 * (local_lower + local_upper)
    local_half = 0.5 * (local_upper - local_lower)
    size = wp.max(wp.max(local_half[0], local_half[1]), local_half[2])
    reach = wp.max(wp.max(wp.abs(pos[0]), wp.abs(pos[1])), wp.abs(pos[2])) + size + contact_threshold
    pad = contact_threshold + 4.0e-6 * reach

    bound = MeshQueryBound()
    bound.center = pos + rot * local_center
    bound.axes = rot
    bound.half = local_half + wp.vec3(pad)
    bound.has_segment = False
    shape = convex.shape_type
    if shape == GeoType.SPHERE or shape == GeoType.CAPSULE or shape == GeoType.CYLINDER or shape == GeoType.CONE:
        half_length = float(0.0)
        radius = local_half[0]
        if shape == GeoType.CAPSULE:
            half_length = convex.scale[1]
            radius = convex.scale[0]
        elif shape != GeoType.SPHERE:
            half_length = local_half[2]
        axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
        bound.seg_a = bound.center - half_length * axis
        bound.seg_b = bound.center + half_length * axis
        bound.radius = radius + pad
        bound.has_segment = True
    return bound


@wp.func
def mesh_vs_convex_local_box(
    convex: GenericShapeData,
    shape: int,
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
) -> tuple[wp.vec3, wp.vec3]:
    """The convex shape's local box: a hull's stored local AABB (as the narrow phase reads it for hulls), a
    primitive's support extremes along its axes."""
    if convex.shape_type == int(GeoType.CONVEX_MESH):
        return shape_collision_aabb_lower[shape], shape_collision_aabb_upper[shape]
    return compute_tight_aabb_from_support(convex, wp.quat_identity(), wp.vec3(0.0), SupportMapDataProvider())


@wp.func
def _mesh_triangle_is_front_facing_local(
    mesh_id: wp.uint64,
    center_in_bvh: wp.vec3,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    watertight: bool,
    convex: GenericShapeData,
    convex_in_mesh: wp.transform,
    mesh_scale: wp.vec3,
) -> bool:
    """Preserve one-sided sheets; recover straddling nearest faces of closed meshes.

    ``v0``, ``v1``, ``v2`` are the triangle's vertices in the unscaled mesh frame (the mesh's own points).
    """
    face_normal = wp.cross(v1 - v0, v2 - v0)
    center_dist = wp.dot(face_normal, center_in_bvh - v0)
    if center_dist >= 0.0:
        return True
    if not watertight:
        return False

    # Reject wholly embedded supports before the more expensive inside query.
    # Mesh scaling transforms normals by its inverse, including reflections.
    scaled_normal = wp.cw_div(face_normal, mesh_scale)
    direction = wp.transform_vector(wp.transform_inverse(convex_in_mesh), scaled_normal)
    support = support_map(convex, direction, SupportMapDataProvider())
    support_in_bvh = wp.cw_div(wp.transform_point(convex_in_mesh, support), mesh_scale)
    if wp.dot(face_normal, support_in_bvh - v0) <= 0.0:
        return False

    # An exterior center must not acquire contacts on the opposite side of a
    # thin shell. For an interior center, only its nearest surface is eligible,
    # avoiding opposing constraints when a convex spans both shell faces.
    nearest = wp.mesh_query_point_sign_parity(mesh_id, center_in_bvh, 1.0e6)
    if not nearest.result or nearest.sign >= 0.0:
        return False
    nearest_point = wp.mesh_eval_position(mesh_id, nearest.face, nearest.u, nearest.v)
    nearest_normal = wp.mesh_eval_face_normal(mesh_id, nearest.face)
    normal_length = wp.length(face_normal)
    if (
        wp.dot(face_normal, nearest_normal) < 0.9999 * normal_length
        or wp.abs(wp.dot(face_normal, nearest_point - v0)) > 1.0e-6 * normal_length
    ):
        return False

    return True


@wp.func
def mesh_vs_convex_pair_shapes(pair: wp.vec2i, shape_type: wp.array[int]) -> tuple[int, int]:
    """The ``(mesh, convex)`` shapes of a mesh-like pair, or ``(-1, -1)`` when it is not mesh-vs-convex."""
    type_a = shape_type[pair[0]]
    type_b = shape_type[pair[1]]
    if type_a == GeoType.MESH and type_b != GeoType.MESH:
        return pair[0], pair[1]
    if type_b == GeoType.MESH and type_a != GeoType.MESH:
        return pair[1], pair[0]
    return -1, -1


@wp.func
def _mesh_vs_convex_shape_data(
    non_mesh_shape: int,
    shape_type: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_source_ptr: wp.array[wp.uint64],
) -> GenericShapeData:
    convex = GenericShapeData()
    convex.shape_type = shape_type[non_mesh_shape]
    convex_scale = shape_data[non_mesh_shape]
    convex.scale = wp.vec3(convex_scale[0], convex_scale[1], convex_scale[2])
    if convex.shape_type == int(GeoType.CONVEX_MESH):
        convex.auxiliary = pack_mesh_ptr(shape_source_ptr[non_mesh_shape])
    return convex


@wp.func
def mesh_vs_convex_query_box(
    mesh_shape: int,
    non_mesh_shape: int,
    X_mesh_ws: wp.transform,
    X_ws: wp.transform,
    shape_type: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_source_ptr: wp.array[wp.uint64],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    contact_threshold: float,
) -> tuple[wp.vec3, wp.vec3]:
    """The box, in the mesh's unscaled frame, of the convex shape's midphase bound (:func:`_mesh_query_bound`).

    Padded past the float32 rounding of the cell coordinates (relative to the coordinates' size).
    """
    convex = _mesh_vs_convex_shape_data(non_mesh_shape, shape_type, shape_data, shape_source_ptr)
    convex_in_mesh = wp.transform_multiply(wp.transform_inverse(X_mesh_ws), X_ws)
    local_lower, local_upper = mesh_vs_convex_local_box(
        convex, non_mesh_shape, shape_collision_aabb_lower, shape_collision_aabb_upper
    )
    bound = _mesh_query_bound(convex, local_lower, local_upper, convex_in_mesh, contact_threshold)
    mesh_data = shape_data[mesh_shape]
    return mesh_query_bound_unscaled_box(bound, wp.vec3(mesh_data[0], mesh_data[1], mesh_data[2]))


@wp.func
def mesh_query_bound_unscaled_box(bound: MeshQueryBound, mesh_scale: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """The bound's box (:func:`mesh_query_bound_box`) in the mesh's unscaled frame.

    Padded past the float32 rounding of the cell coordinates (relative to the coordinates' size).
    """
    lower, upper = mesh_query_bound_box(bound)
    lower_bvh, upper_bvh, _inv_scale = aabb_to_unscaled(lower, upper, mesh_scale)
    reach = wp.max(wp.max(wp.abs(lower_bvh[0]), wp.abs(lower_bvh[1])), wp.abs(lower_bvh[2]))
    reach = wp.max(reach, wp.max(wp.max(wp.abs(upper_bvh[0]), wp.abs(upper_bvh[1])), wp.abs(upper_bvh[2])))
    pad = 1.0e-5 * (reach + (upper_bvh[0] - lower_bvh[0]) + 1.0e-3)
    return lower_bvh - wp.vec3(pad), upper_bvh + wp.vec3(pad)


@wp.func
def mesh_query_bound_slab(bound: MeshQueryBound, k: int, mesh_scale: wp.vec3) -> wp.vec4:
    """The bound's box as its ``k``-th slab in the mesh's unscaled frame: ``(n, d)`` with ``|n . x - d| <= half[k]``.

    A point ``x`` of the unscaled frame is ``scale * x`` in the bound's; so ``n = scale * axis_k``, ``d = axis_k .
    center``.
    """
    axis = wp.vec3(bound.axes[0, k], bound.axes[1, k], bound.axes[2, k])
    normal = wp.cw_mul(axis, mesh_scale)
    return wp.vec4(normal[0], normal[1], normal[2], wp.dot(axis, bound.center))


@wp.func
def mesh_vs_convex_query_range(
    mesh_cells: MeshCells, grid_index: int, query_lower: wp.vec3, query_upper: wp.vec3
) -> tuple[wp.vec3i, wp.vec3i, bool]:
    """The level-0 cell range under a query box (:func:`mesh_cells_range`), padded for the grid origin's rounding."""
    grid = mesh_cells.grids[grid_index]
    origin_size = wp.max(wp.max(wp.abs(grid.origin[0]), wp.abs(grid.origin[1])), wp.abs(grid.origin[2]))
    pad = wp.vec3(1.0e-5 * (origin_size + grid.cell))
    return mesh_cells_range(grid, query_lower - pad, query_upper + pad)


@wp.func
def mesh_cells_entry_is_candidate(
    entry: wp.vec4i,
    cell: wp.vec3i,
    qlo: wp.vec3i,
    qhi: wp.vec3i,
    box_lo: wp.vec3i,
    box_hi: wp.vec3i,
    cell_lower: wp.vec3,
    size: float,
    slab0: wp.vec4,
    slab1: wp.vec4,
    slab2: wp.vec4,
    slab_half: wp.vec3,
) -> bool:
    """Whether a cell's entry is a candidate: its box meets the query box (``box_lo..box_hi`` on the cell's lattice) and
    the bound's oriented box (its slabs), and, for a small triangle, this is its first registered cell in range."""
    if not (
        mesh_cells_entry_overlaps(entry, box_lo, box_hi) and mesh_cells_entry_first_in_range(entry, cell, qlo, qhi)
    ):
        return False
    tri_lo, tri_hi, tri_bounded = mesh_cells_entry_box(entry, cell_lower, size)
    return not tri_bounded or mesh_cells_box_meets_slabs(tri_lo, tri_hi, slab0, slab1, slab2, slab_half)


@wp.func
def mesh_vs_convex_cell_hits(
    lane: int,
    lanes: int,
    pair_index: int,
    grid_index: int,
    query_lower: wp.vec3,
    query_upper: wp.vec3,
    lo0: wp.vec3i,
    hi0: wp.vec3i,
    slab0: wp.vec4,
    slab1: wp.vec4,
    slab2: wp.vec4,
    slab_half: wp.vec3,
    mesh_cells: MeshCells,
    hit_head: wp.array[wp.vec4i],
    hit_span: wp.array[wp.vec2i],
    hit_count: wp.array[int],
):
    """Append a hit for each occupied cell of the query's range whose triangles may meet the bound.

    At every occupied level of the mesh's index, the cells of the level-0 range ``[lo0, hi0]`` are looked up, or the
    level's occupied cells walked when they are fewer. A cell is a hit unless its triangles' box misses the query box
    ``[query_lower, query_upper]`` (unscaled mesh frame) or the bound's oriented box (its three slabs, ``slab0``..
    ``slab2`` with half-widths ``slab_half``, :func:`mesh_query_bound_slab`). A hit is ``(pair, level, x | y << 15, z)``
    with its entries ``(start, count)``; :func:`mesh_vs_convex_hit_candidates` scans them, one thread a hit. ``lanes``
    threads share the pair; each takes every ``lanes``-th cell.
    """
    grid = mesh_cells.grids[grid_index]
    visited = int(0)
    for level in range(16):
        if (grid.levels >> level) & 1 == 0:
            continue
        shift = MESH_CELLS_LEVEL_SHIFT * level
        qlo = wp.vec3i(lo0[0] >> shift, lo0[1] >> shift, lo0[2] >> shift)
        qhi = wp.vec3i(hi0[0] >> shift, hi0[1] >> shift, hi0[2] >> shift)
        nx = qhi[0] - qlo[0] + 1
        ny = qhi[1] - qlo[1] + 1
        nz = qhi[2] - qlo[2] + 1
        list_start = mesh_cells.level_start[grid.level_start + level]
        list_count = mesh_cells.level_start[grid.level_start + level + 1] - list_start
        # Look the range's cells up, or walk the level's occupied cells when they are fewer.
        by_list = float(nx) * float(ny) * float(nz) > float(list_count)
        n = list_count
        if not by_list:
            n = nx * ny * nz
        size = grid.cell * float(1 << shift)
        inv_size = grid.inv_cell / float(1 << shift)
        k = ((lane - visited) % lanes + lanes) % lanes
        while k < n:
            cell = wp.vec3i(0)
            slot = MeshCellSlot()
            slot.count = 0
            if by_list:
                slot = mesh_cells.sorted_slots[list_start + k]
                cell = mesh_cells_key_cell(slot.key)
                if (
                    cell[0] < qlo[0]
                    or cell[0] > qhi[0]
                    or cell[1] < qlo[1]
                    or cell[1] > qhi[1]
                    or cell[2] < qlo[2]
                    or cell[2] > qhi[2]
                ):
                    slot.count = 0
            else:
                cell = wp.vec3i(qlo[0] + k % nx, qlo[1] + (k // nx) % ny, qlo[2] + k // (nx * ny))
                slot = mesh_cells_find(mesh_cells, grid, mesh_cells_key(level, cell[0], cell[1], cell[2]))
            if slot.count > 0:
                cell_lower = grid.origin + size * wp.vec3(float(cell[0]), float(cell[1]), float(cell[2]))
                union_lo, union_hi, union_bounded = mesh_cells_slot_box(slot, cell_lower, size)
                if mesh_cells_slot_overlaps(slot, query_lower, query_upper, cell_lower, inv_size) and (
                    not union_bounded or mesh_cells_box_meets_slabs(union_lo, union_hi, slab0, slab1, slab2, slab_half)
                ):
                    out = wp.atomic_add(hit_count, 0, 1)
                    if out < hit_head.shape[0]:
                        hit_head[out] = wp.vec4i(pair_index, level, cell[0] | (cell[1] << 15), cell[2])
                        hit_span[out] = wp.vec2i(slot.start, slot.count)
            k += lanes
        visited += n


@wp.func
def mesh_vs_convex_hit_candidates(
    head: wp.vec4i,
    span: wp.vec2i,
    query_lower: wp.vec3,
    query_upper: wp.vec3,
    lo0: wp.vec3i,
    hi0: wp.vec3i,
    slab0: wp.vec4,
    slab1: wp.vec4,
    slab2: wp.vec4,
    slab_half: wp.vec3,
    grid: MeshGrid,
    mesh_cells: MeshCells,
    candidates: wp.array[wp.vec4i],
    candidate_count: wp.array[int],
):
    """Append a candidate (:func:`mesh_cells_candidate`) for each triangle of a hit cell
    (:func:`mesh_vs_convex_cell_hits`) whose own box meets the query box and the bound's oriented box, from the
    triangle's first registered cell in the query's range only.

    The oriented box test is the exact pass's own necessary condition (a triangle that meets the oriented box has a
    box that meets it), so it drops nothing the exact pass keeps. A small triangle (box of at most two cells per axis)
    is tested for its first cell by its entry's block bits; a larger one by its registration
    (:func:`mesh_cells_first_in_range`), loaded only for entries that passed the box tests.
    """
    pair_index = head[0]
    level = head[1]
    cell = wp.vec3i(head[2] & 0x7FFF, head[2] >> 15, head[3])
    shift = MESH_CELLS_LEVEL_SHIFT * level
    qlo = wp.vec3i(lo0[0] >> shift, lo0[1] >> shift, lo0[2] >> shift)
    qhi = wp.vec3i(hi0[0] >> shift, hi0[1] >> shift, hi0[2] >> shift)
    size = grid.cell * float(1 << shift)
    inv_size = grid.inv_cell / float(1 << shift)
    cell_lower = grid.origin + size * wp.vec3(float(cell[0]), float(cell[1]), float(cell[2]))
    box_lo, box_hi = mesh_cells_box_in_cell(query_lower, query_upper, cell_lower, inv_size)
    # A cell's candidates take one reservation, 32 entries at a time (marked, then written): one append per candidate
    # costs one atomic on the shared count.
    for chunk in range(span[0], span[0] + span[1], 32):
        marks = int(0)
        found = int(0)
        for e in range(chunk, wp.min(chunk + 32, span[0] + span[1])):
            if mesh_cells_entry_is_candidate(
                mesh_cells.entries[e],
                cell,
                qlo,
                qhi,
                box_lo,
                box_hi,
                cell_lower,
                size,
                slab0,
                slab1,
                slab2,
                slab_half,
            ):
                entry = mesh_cells.entries[e]
                first = bool(True)
                if (entry[3] >> 31) & 1 == 0:
                    # A larger triangle: this cell must be its first registered cell in range (its registration).
                    registration = mesh_cells.triangles[grid.triangle_start + entry[0]]
                    first = mesh_cells_first_in_range(registration.cells_mask, registration.cells_box, cell, qlo, qhi)
                if first:
                    marks = marks | (1 << (e - chunk))
                    found += 1
        if found > 0:
            out = wp.atomic_add(candidate_count, 0, found)
            for j in range(32):
                if (marks >> j) & 1 != 0:
                    if out < candidates.shape[0]:
                        candidates[out] = mesh_cells_candidate(
                            pair_index, mesh_cells.entries[chunk + j][0], level, cell
                        )
                    out += 1


@wp.func
def mesh_vs_convex_midphase(
    mesh_shape: int,
    non_mesh_shape: int,
    candidate: wp.vec4i,
    query_lower: wp.vec3,
    query_upper: wp.vec3,
    X_mesh_ws: wp.transform,
    X_ws: wp.transform,
    mesh_id: wp.uint64,
    shape_type: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[wp.int32],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    contact_threshold: float,
    non_mesh_center: wp.vec3,
    triangle_pairs: wp.array[wp.vec3i],
    triangle_pairs_count: wp.array[int],
):
    """
    Finish the mesh vs convex shape midphase for one candidate triangle of a pair.

    The candidates come from the mesh's cell index (:func:`mesh_vs_convex_hit_candidates`): every triangle whose
    box meets the shape's query box and oriented box, once, from its first registered cell in the query's range (the
    reference-cell rule, applied there). A candidate is kept when it meets the shape's bound
    exactly (an oriented box, and a segment plus a radius for spheres, capsules, cylinders and cones, all inflated by
    the threshold) and faces the shape. Kept triangles are appended to the triangle pairs of the triangle narrow
    phase.

    Args:
        mesh_shape: Index of the mesh shape
        non_mesh_shape: Index of the non-mesh (convex) shape
        candidate: The candidate (:func:`mesh_cells_candidate`)
        query_lower: The pair's query box (:func:`mesh_vs_convex_query_box`), lower corner [m]
        query_upper: The pair's query box, upper corner [m]
        X_mesh_ws: Mesh world-space transform
        X_ws: Non-mesh shape world-space transform
        mesh_id: Mesh ID
        shape_type: Array of shape types
        shape_data: Array of shape data (vec4: scale.xyz, margin.w)
        shape_source_ptr: Array of mesh/SDF source pointers
        shape_mesh_properties: Per-shape watertightness metadata
        shape_collision_aabb_lower: Per-shape local AABB lower bounds (a hull's local box)
        shape_collision_aabb_upper: Per-shape local AABB upper bounds
        contact_threshold: Contact candidate distance [m], including margin and gap
        non_mesh_center: Interior point of the convex shape in its scaled local frame [m]
        triangle_pairs: Output array for triangle pairs (mesh_shape, non_mesh_shape, tri_index)
        triangle_pairs_count: Counter for triangle pairs
    """
    tri_index = candidate[1]
    mesh = wp.mesh_get(mesh_id)
    v0 = mesh.points[mesh.indices[3 * tri_index + 0]]
    v1 = mesh.points[mesh.indices[3 * tri_index + 1]]
    v2 = mesh.points[mesh.indices[3 * tri_index + 2]]
    # The triangle meets the query box (which holds the bound), so it is met in a cell of the query's range.
    query_center = 0.5 * (query_lower + query_upper)
    if not triangle_meets_box(
        v0 - query_center, v1 - query_center, v2 - query_center, 0.5 * (query_upper - query_lower)
    ):
        return
    watertight = (shape_mesh_properties[mesh_shape] & MeshProperties.WATERTIGHT) != 0
    convex = _mesh_vs_convex_shape_data(non_mesh_shape, shape_type, shape_data, shape_source_ptr)
    convex_in_mesh = wp.transform_multiply(wp.transform_inverse(X_mesh_ws), X_ws)
    mesh_data = shape_data[mesh_shape]
    mesh_scale = wp.vec3(mesh_data[0], mesh_data[1], mesh_data[2])
    local_lower, local_upper = mesh_vs_convex_local_box(
        convex, non_mesh_shape, shape_collision_aabb_lower, shape_collision_aabb_upper
    )
    bound = _mesh_query_bound(convex, local_lower, local_upper, convex_in_mesh, contact_threshold)
    if not mesh_query_bound_meets_triangle(
        bound,
        wp.cw_mul(v0, mesh_scale),
        wp.cw_mul(v1, mesh_scale),
        wp.cw_mul(v2, mesh_scale),
    ):
        return
    _lower, _upper, inv_scale = aabb_to_unscaled(wp.vec3(0.0), wp.vec3(0.0), mesh_scale)
    center_in_bvh = wp.cw_mul(wp.transform_point(convex_in_mesh, non_mesh_center), inv_scale)
    if not _mesh_triangle_is_front_facing_local(
        mesh_id,
        center_in_bvh,
        v0,
        v1,
        v2,
        watertight,
        convex,
        convex_in_mesh,
        mesh_scale,
    ):
        return
    # Pair order is unconstrained; append without a scan.
    out_idx = wp.atomic_add(triangle_pairs_count, 0, 1)
    if out_idx < triangle_pairs.shape[0]:
        triangle_pairs[out_idx] = wp.vec3i(mesh_shape, non_mesh_shape, tri_index)


@wp.func
def find_pair_from_cumulative_index(
    global_idx: int,
    cumulative_sums: wp.array[int],
    pair_count: int,
) -> tuple[int, int]:
    """
    Binary search to find which pair a global index belongs to.

    This function is useful for mapping a flat global index to a (pair_index, local_index)
    tuple when work is distributed across multiple pairs with varying sizes.

    Args:
        global_idx: Global index to search for
        cumulative_sums: Array of inclusive cumulative sums (end indices for each pair)
        pair_count: Number of pairs

    Returns:
        Tuple of (pair_index, local_index_within_pair)
    """
    # Use binary_search to find first index where cumulative_sums[i] > global_idx
    # This gives us the bucket that contains global_idx
    pair_idx = binary_search(cumulative_sums, global_idx, 0, pair_count)

    # Get cumulative start for this pair to calculate local index
    cumulative_start = int(0)
    if pair_idx > 0:
        cumulative_start = int(cumulative_sums[pair_idx - 1])

    local_idx = global_idx - cumulative_start

    return pair_idx, local_idx


@wp.func
def get_triangle_shape_from_mesh(
    mesh_id: wp.uint64,
    mesh_scale: wp.vec3,
    X_mesh_ws: wp.transform,
    tri_idx: int,
) -> tuple[GenericShapeData, wp.vec3]:
    """
    Extract triangle shape data from a mesh.

    This function retrieves a specific triangle from a mesh and creates a GenericShapeData
    structure for collision detection. The triangle is represented in world space with
    vertex A as the origin.

    Args:
        mesh_id: The mesh ID (use wp.mesh_get to retrieve the mesh object)
        mesh_scale: Scale to apply to mesh vertices
        X_mesh_ws: Mesh world-space transform
        tri_idx: Triangle index in the mesh

    Returns:
        Tuple of (shape_data, v0_world) where:
        - shape_data: GenericShapeData with triangle geometry (type=TRIANGLE, scale=B-A, auxiliary=C-A)
        - v0_world: First vertex position in world space (used as triangle origin)
    """
    # Get the mesh object from the ID
    mesh = wp.mesh_get(mesh_id)

    # Extract triangle vertices from mesh (indices are stored as flat array: i0, i1, i2, i0, i1, i2, ...)
    idx0 = mesh.indices[tri_idx * 3 + 0]
    idx1 = mesh.indices[tri_idx * 3 + 1]
    idx2 = mesh.indices[tri_idx * 3 + 2]

    # Mirror parity (det(scale) < 0) reflects the geometry, which would invert
    # triangle winding and flip the face-normal sign. Swap the second and third
    # indices so downstream code (back-face culling, GJK/MPR triangle support)
    # always sees a consistently-wound (outward-facing) triangle.
    if mesh_scale[0] * mesh_scale[1] * mesh_scale[2] < 0.0:
        tmp = idx1
        idx1 = idx2
        idx2 = tmp

    # Get vertex positions in mesh local space (with scale applied)
    v0_local = wp.cw_mul(mesh.points[idx0], mesh_scale)
    v1_local = wp.cw_mul(mesh.points[idx1], mesh_scale)
    v2_local = wp.cw_mul(mesh.points[idx2], mesh_scale)

    # Transform vertices to world space
    v0_world = wp.transform_point(X_mesh_ws, v0_local)
    v1_world = wp.transform_point(X_mesh_ws, v1_local)
    v2_world = wp.transform_point(X_mesh_ws, v2_local)

    # Create triangle shape data: vertex A at origin, B-A in scale, C-A in auxiliary
    shape_data = GenericShapeData()
    shape_data.shape_type = int(GeoTypeEx.TRIANGLE)
    shape_data.scale = v1_world - v0_world  # B - A
    shape_data.auxiliary = v2_world - v0_world  # C - A
    shape_data.center = wp.vec3(0.0, 0.0, 0.0)

    return shape_data, v0_world


# OBB collisions by Separating Axis Theorem
@wp.func
def get_box_axes(q: wp.quat) -> wp.mat33:
    """Get the 3 local axes of a box from its quaternion rotation"""
    # Box local axes (x, y, z)
    local_x = wp.vec3(1.0, 0.0, 0.0)
    local_y = wp.vec3(0.0, 1.0, 0.0)
    local_z = wp.vec3(0.0, 0.0, 1.0)

    # Rotate local axes to world space using warp's built-in method
    axis_x = wp.quat_rotate(q, local_x)
    axis_y = wp.quat_rotate(q, local_y)
    axis_z = wp.quat_rotate(q, local_z)

    return wp.matrix_from_rows(axis_x, axis_y, axis_z)


@wp.func
def project_box_onto_axis(transform: wp.transform, extents: wp.vec3, axis: wp.vec3) -> wp.vec2:
    """Project a box onto an axis and return [min, max] projection values"""
    # Get box axes and extents
    axes = get_box_axes(wp.transform_get_rotation(transform))

    # Project box center onto axis
    center_proj = wp.dot(wp.transform_get_translation(transform), axis)

    # Project each axis of the box onto the separating axis and get the extent
    extent = 0.0
    extent += extents[0] * wp.abs(wp.dot(axes[0], axis))  # x-axis contribution
    extent += extents[1] * wp.abs(wp.dot(axes[1], axis))  # y-axis contribution
    extent += extents[2] * wp.abs(wp.dot(axes[2], axis))  # z-axis contribution

    return wp.vec2(center_proj - extent, center_proj + extent)


@wp.func
def test_axis_separation(
    transform_a: wp.transform, extents_a: wp.vec3, transform_b: wp.transform, extents_b: wp.vec3, axis: wp.vec3
) -> bool:
    """Test if two boxes are separated along a given axis. Returns True if separated."""
    # Normalize the axis (handle zero-length axes)
    axis_len = wp.length(axis)
    if axis_len < 1e-8:
        return False  # Invalid axis, assume no separation

    normalized_axis = axis / axis_len

    # Project both boxes onto the axis
    proj_a = project_box_onto_axis(transform_a, extents_a, normalized_axis)
    proj_b = project_box_onto_axis(transform_b, extents_b, normalized_axis)

    # Check if projections overlap - if no overlap, boxes are separated
    return proj_a[1] < proj_b[0] or proj_b[1] < proj_a[0]


@wp.func
def sat_box_intersection(
    transform_a: wp.transform, extents_a: wp.vec3, transform_b: wp.transform, extents_b: wp.vec3
) -> bool:
    """
    Test if two oriented boxes intersect using the Separating Axis Theorem.

    Args:
        transform_a: Transform of first box (position and rotation)
        extents_a: Half-extents of first box
        transform_b: Transform of second box (position and rotation)
        extents_b: Half-extents of second box

    Returns:
        bool: True if boxes intersect, False if separated
    """
    # Get the axes for both boxes
    axes_a = get_box_axes(wp.transform_get_rotation(transform_a))
    axes_b = get_box_axes(wp.transform_get_rotation(transform_b))

    # Test the 15 potential separating axes

    # Test face normals of box A (3 axes)
    for i in range(3):
        if test_axis_separation(transform_a, extents_a, transform_b, extents_b, axes_a[i]):
            return False  # Boxes are separated

    # Test face normals of box B (3 axes)
    for i in range(3):
        if test_axis_separation(transform_a, extents_a, transform_b, extents_b, axes_b[i]):
            return False  # Boxes are separated

    # Test cross products of edge directions (9 axes: 3x3 combinations)
    for i in range(3):
        for j in range(3):
            cross_axis = wp.cross(axes_a[i], axes_b[j])
            if test_axis_separation(transform_a, extents_a, transform_b, extents_b, cross_axis):
                return False  # Boxes are separated

    # If no separating axis found, boxes intersect
    return True

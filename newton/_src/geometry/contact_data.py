# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Contact data structures for collision detection.

This module defines the core contact data structures used throughout the collision detection system.
"""

import warp as wp

# Bit flag and mask used to encode heightfield shape indices in collision pair buffers.
SHAPE_PAIR_HFIELD_BIT = wp.int32(1 << 30)
SHAPE_PAIR_INDEX_MASK = wp.int32((1 << 30) - 1)

# Contact feature keys use at most 23 bits in the public sort key. Reserve four
# higher internal bits to carry strict-guard membership, which shape owns a
# generated contact normal, and canonical-endpoint provenance through the global
# reducer without enlarging the hot by-value ContactData struct. Predictive status
# is derived from guard membership and current separation when the contact is written.
CONTACT_NORMAL_OWNER_WORLD = wp.constant(-1)
CONTACT_NORMAL_OWNER_SHAPE_A = wp.constant(0)
CONTACT_NORMAL_OWNER_SHAPE_B = wp.constant(1)
CONTACT_STRICT_GUARD_NONE = wp.constant(0)
CONTACT_STRICT_GUARD_PAIRED = wp.constant(1)
CONTACT_STRICT_GUARD_ONLY = wp.constant(2)
_CONTACT_IS_STRICT_GUARD_SHIFT = wp.constant(28)
_CONTACT_IS_STRICT_GUARD_MASK = wp.constant(1 << 28)
_CONTACT_NORMAL_OWNER_SHIFT = wp.constant(29)
_CONTACT_NORMAL_OWNER_MASK = wp.constant(0x3)
_CONTACT_NORMAL_OWNER_BITS_MASK = wp.constant(0x3 << 29)
_CONTACT_IS_CANONICAL_ENDPOINT_MASK = wp.constant(wp.int32(-2147483648))
_CONTACT_FEATURE_KEY_MASK = wp.constant((1 << 28) - 1)


@wp.struct
class ContactData:
    """
    Internal contact representation for collision detection.

    This struct stores contact information between two colliding shapes before conversion
    to solver-specific formats. It serves as an intermediate representation passed between
    collision detection algorithms and contact writer functions.

    Attributes:
        contact_point_center: Center point of the contact region in world space
        contact_normal_a_to_b: Unit normal vector pointing from shape A to shape B
        contact_distance: Signed distance between shapes (negative indicates penetration)
        radius_eff_a: Effective radius of shape A (for rounded shapes like spheres/capsules)
        radius_eff_b: Effective radius of shape B (for rounded shapes like spheres/capsules)
        margin_a: Collision surface margin offset for shape A
        margin_b: Collision surface margin offset for shape B
        shape_a: Index of the first shape in the collision pair
        shape_b: Index of the second shape in the collision pair
        gap_sum: Pairwise summed contact gap threshold that determines if a contact should be written
        contact_stiffness: Contact stiffness. 0.0 means no stiffness was set.
        contact_damping: Contact damping scale. 0.0 means no damping was set.
        contact_friction_scale: Friction scaling factor. 0.0 means no friction was set.
        sort_sub_key: Sub-key for deterministic contact sorting (encodes edge/triangle/vertex index).
        strict_guard_provenance_finalized: Strict-guard decision state. Zero lets a direct manifold select a paired
            guard, one means ``sort_sub_key`` contains the final paired-guard decision, and two denotes a dedicated
            guard-only row.
    """

    contact_point_center: wp.vec3
    contact_normal_a_to_b: wp.vec3
    contact_distance: float
    radius_eff_a: float
    radius_eff_b: float
    margin_a: float
    margin_b: float
    shape_a: int
    shape_b: int
    gap_sum: float
    contact_stiffness: float
    contact_damping: float
    contact_friction_scale: float
    sort_sub_key: int
    strict_guard_provenance_finalized: int


@wp.func
def make_contact_sort_key(shape_a: int, shape_b: int, sort_sub_key: int) -> wp.int64:
    """Build a 64-bit sort key for deterministic contact ordering.

    Layout (bit 63 kept zero so int64 order matches uint64 order)::

        [62:43] shape_a      (20 bits, max 1,048,575 shapes)
        [42:23] shape_b      (20 bits, max 1,048,575 shapes)
        [22:0]  folded sort_sub_key (23 bits, max 8,388,607)

    Values exceeding these bit widths are silently masked.  The effective
    limits depend on upstream bit consumption in each contact path:

    - Mesh-triangle contacts: ``(tri_idx << 1) | 1`` — 22 effective bits
      for ``tri_idx`` (~4M triangles).  When expanded by the multi-contact
      path (``<< 3 | i``), this drops to 19 effective bits (~524K triangles).
    - SDF edge minima: ``(edge_idx << 2) | (mode << 1)`` — 21 effective
      sort-key bits (~2M edges), or 20 after global reduction's 22-bit
      fingerprint packing (~1M edges).
    - SDF owned endpoints: ``(edge_idx << 3) | (mode << 2) | {1, 3}`` — 20
      effective sort-key bits (~1M edges), or 19 after global reduction's
      22-bit fingerprint packing (~524K edges).
    - Hydroelastic contacts: ``((voxel_idx * 5 + face_idx) << 1) | source``,
      with bit 0 distinguishing normal-bin and voxel-bin winners and bit 22
      reserved for reduction anchors — 21 effective fingerprint bits (~419K
      iso voxels).
    """
    feature_key = sort_sub_key & _CONTACT_FEATURE_KEY_MASK
    leaf_slot = (feature_key >> 23) & 0x1F
    folded_sub_key = ((feature_key & 0x7FFFFF) ^ (leaf_slot << 3)) & 0x7FFFFF
    return (
        ((wp.int64(shape_a) & wp.int64(0xFFFFF)) << wp.int64(43))
        | ((wp.int64(shape_b) & wp.int64(0xFFFFF)) << wp.int64(23))
        | wp.int64(folded_sub_key)
    )


@wp.func
def pack_contact_normal_owner(feature_key: int, normal_owner: int) -> int:
    """Pack contact-normal ownership into an internal feature key."""
    owner_code = normal_owner + 1
    return (
        feature_key & (_CONTACT_FEATURE_KEY_MASK | _CONTACT_IS_STRICT_GUARD_MASK | _CONTACT_IS_CANONICAL_ENDPOINT_MASK)
    ) | ((owner_code & _CONTACT_NORMAL_OWNER_MASK) << _CONTACT_NORMAL_OWNER_SHIFT)


@wp.func
def pack_contact_is_strict_guard(packed_feature_key: int, is_strict_guard: bool) -> int:
    """Pack strict nonpenetration-guard membership into an internal feature key."""
    return (
        packed_feature_key
        & (_CONTACT_FEATURE_KEY_MASK | _CONTACT_NORMAL_OWNER_BITS_MASK | _CONTACT_IS_CANONICAL_ENDPOINT_MASK)
    ) | (int(is_strict_guard) << _CONTACT_IS_STRICT_GUARD_SHIFT)


@wp.func
def pack_contact_is_canonical_endpoint(packed_feature_key: int, is_canonical_endpoint: bool) -> int:
    """Pack canonical mesh-endpoint provenance into an internal feature key."""
    result = packed_feature_key & (
        _CONTACT_FEATURE_KEY_MASK | _CONTACT_IS_STRICT_GUARD_MASK | _CONTACT_NORMAL_OWNER_BITS_MASK
    )
    if is_canonical_endpoint:
        result = result | _CONTACT_IS_CANONICAL_ENDPOINT_MASK
    return result


@wp.func
def unpack_contact_feature_key(packed_feature_key: int) -> int:
    """Return the public geometric feature key without internal metadata."""
    return packed_feature_key & _CONTACT_FEATURE_KEY_MASK


@wp.func
def unpack_contact_is_strict_guard(packed_feature_key: int) -> bool:
    """Return whether an internal feature key belongs to the final strict-guard manifold."""
    return (packed_feature_key & _CONTACT_IS_STRICT_GUARD_MASK) != 0


@wp.func
def unpack_contact_is_canonical_endpoint(packed_feature_key: int) -> bool:
    """Return whether an internal feature key identifies a canonical mesh endpoint."""
    return (packed_feature_key & _CONTACT_IS_CANONICAL_ENDPOINT_MASK) != 0


@wp.func
def unpack_contact_normal_owner(packed_feature_key: int) -> int:
    """Return the shape-relative owner encoded in an internal feature key."""
    owner_code = (packed_feature_key >> _CONTACT_NORMAL_OWNER_SHIFT) & _CONTACT_NORMAL_OWNER_MASK
    return owner_code - 1


@wp.func
def compute_contact_signed_approach_speed(
    shape_a: int,
    shape_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal_a_to_b: wp.vec3,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
) -> float:
    """Return signed closing speed of two shape points along a contact normal [m/s]."""
    origin_a = wp.transform_get_translation(shape_transform[shape_a])
    origin_b = wp.transform_get_translation(shape_transform[shape_b])
    velocity_a = shape_linear_velocity[shape_a] + wp.cross(shape_angular_velocity[shape_a], point_a - origin_a)
    velocity_b = shape_linear_velocity[shape_b] + wp.cross(shape_angular_velocity[shape_b], point_b - origin_b)
    return -wp.dot(velocity_b - velocity_a, normal_a_to_b)


@wp.func
def compute_contact_approach_speed(
    shape_a: int,
    shape_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal_a_to_b: wp.vec3,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
) -> float:
    """Return non-negative closing speed of two shape points along a contact normal [m/s]."""
    return wp.max(
        compute_contact_signed_approach_speed(
            shape_a,
            shape_b,
            point_a,
            point_b,
            normal_a_to_b,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
        ),
        0.0,
    )


@wp.func
def predict_constant_velocity_anchor(
    shape: int,
    anchor_world: wp.vec3,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    prediction_dt: float,
) -> wp.vec3:
    """Predict a rigid anchor with constant center velocity and exact exponential-map rotation [m]."""
    origin = wp.transform_get_translation(shape_transform[shape])
    angular_velocity = shape_angular_velocity[shape]
    rotation_center_offset = wp.vec3(0.0)
    if shape < shape_rotation_center_offset.shape[0]:
        rotation_center_offset = shape_rotation_center_offset[shape]
    rotation_center = origin + rotation_center_offset
    center_velocity = shape_linear_velocity[shape] + wp.cross(angular_velocity, rotation_center_offset)
    offset = anchor_world - rotation_center
    angular_speed = wp.length(angular_velocity)
    if angular_speed > 0.0:
        rotation = wp.quat_from_axis_angle(angular_velocity / angular_speed, angular_speed * prediction_dt)
        offset = wp.quat_rotate(rotation, offset)
    return rotation_center + center_velocity * prediction_dt + offset


@wp.func
def predict_contact_end_geometry(
    shape_a: int,
    shape_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal_a_to_b: wp.vec3,
    normal_owner: int,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    prediction_dt: float,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Predict contact anchors and their owned normal at the end of a constant-velocity horizon."""
    point_a_end = predict_constant_velocity_anchor(
        shape_a,
        point_a,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        prediction_dt,
    )
    point_b_end = predict_constant_velocity_anchor(
        shape_b,
        point_b,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        prediction_dt,
    )

    normal_end = normal_a_to_b
    owner_angular_velocity = wp.vec3(0.0)
    if normal_owner == CONTACT_NORMAL_OWNER_SHAPE_A:
        owner_angular_velocity = shape_angular_velocity[shape_a]
    elif normal_owner == CONTACT_NORMAL_OWNER_SHAPE_B:
        owner_angular_velocity = shape_angular_velocity[shape_b]
    owner_angular_speed = wp.length(owner_angular_velocity)
    if owner_angular_speed > 0.0:
        owner_rotation = wp.quat_from_axis_angle(
            owner_angular_velocity / owner_angular_speed,
            owner_angular_speed * prediction_dt,
        )
        normal_end = wp.quat_rotate(owner_rotation, normal_end)

    return point_a_end, point_b_end, normal_end


@wp.func
def compute_contact_end_separation(
    shape_a: int,
    shape_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal_a_to_b: wp.vec3,
    normal_owner: int,
    total_separation_needed: float,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    prediction_dt: float,
) -> float:
    """Predict physical separation using constant center velocities and exact exponential-map rotations [m]."""
    point_a_end, point_b_end, normal_end = predict_contact_end_geometry(
        shape_a,
        shape_b,
        point_a,
        point_b,
        normal_a_to_b,
        normal_owner,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        prediction_dt,
    )
    return wp.dot(point_b_end - point_a_end, normal_end) - total_separation_needed


@wp.func
def compute_contact_swept_separation_lower_bound(
    shape_a: int,
    shape_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal_a_to_b: wp.vec3,
    normal_owner: int,
    total_separation_needed: float,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    prediction_dt: float,
) -> float:
    """Bound separation from below over a constant-velocity prediction horizon [m].

    The exact separation is sampled at the start, midpoint, and end. A
    rigid-motion curvature bound then limits how far the continuous trajectory
    can fall below either half-horizon chord.
    """
    start_separation = wp.dot(point_b - point_a, normal_a_to_b) - total_separation_needed
    if prediction_dt <= 0.0:
        return start_separation

    point_a_mid, point_b_mid, normal_mid = predict_contact_end_geometry(
        shape_a,
        shape_b,
        point_a,
        point_b,
        normal_a_to_b,
        normal_owner,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.5 * prediction_dt,
    )
    midpoint_separation = wp.dot(point_b_mid - point_a_mid, normal_mid) - total_separation_needed
    point_a_end, point_b_end, normal_end = predict_contact_end_geometry(
        shape_a,
        shape_b,
        point_a,
        point_b,
        normal_a_to_b,
        normal_owner,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        prediction_dt,
    )
    end_separation = wp.dot(point_b_end - point_a_end, normal_end) - total_separation_needed

    origin_a = wp.transform_get_translation(shape_transform[shape_a])
    origin_b = wp.transform_get_translation(shape_transform[shape_b])
    center_offset_a = wp.vec3(0.0)
    center_offset_b = wp.vec3(0.0)
    if shape_a < shape_rotation_center_offset.shape[0]:
        center_offset_a = shape_rotation_center_offset[shape_a]
    if shape_b < shape_rotation_center_offset.shape[0]:
        center_offset_b = shape_rotation_center_offset[shape_b]
    center_a = origin_a + center_offset_a
    center_b = origin_b + center_offset_b

    omega_a = shape_angular_velocity[shape_a]
    omega_b = shape_angular_velocity[shape_b]
    center_velocity_a = shape_linear_velocity[shape_a] + wp.cross(omega_a, center_offset_a)
    center_velocity_b = shape_linear_velocity[shape_b] + wp.cross(omega_b, center_offset_b)
    center_delta = center_b - center_a
    center_velocity_delta = center_velocity_b - center_velocity_a
    radius_a = point_a - center_a
    radius_b = point_b - center_b

    curvature_bound = float(0.0)
    if normal_owner == CONTACT_NORMAL_OWNER_SHAPE_A:
        normal_speed = wp.length(wp.cross(omega_a, normal_a_to_b))
        normal_acceleration = wp.length(omega_a) * normal_speed
        relative_angular_speed = wp.length(omega_b - omega_a)
        curvature_bound = (
            normal_acceleration * (wp.length(center_delta) + wp.length(center_velocity_delta) * prediction_dt)
            + 2.0 * normal_speed * wp.length(center_velocity_delta)
            + relative_angular_speed * (wp.length(omega_a) + wp.length(omega_b)) * wp.length(radius_b)
        )
    elif normal_owner == CONTACT_NORMAL_OWNER_SHAPE_B:
        normal_speed = wp.length(wp.cross(omega_b, normal_a_to_b))
        normal_acceleration = wp.length(omega_b) * normal_speed
        relative_angular_speed = wp.length(omega_a - omega_b)
        curvature_bound = (
            normal_acceleration * (wp.length(center_delta) + wp.length(center_velocity_delta) * prediction_dt)
            + 2.0 * normal_speed * wp.length(center_velocity_delta)
            + relative_angular_speed * (wp.length(omega_a) + wp.length(omega_b)) * wp.length(radius_a)
        )
    else:
        curvature_bound = wp.length(omega_a) * wp.length(wp.cross(omega_a, radius_a)) + wp.length(omega_b) * wp.length(
            wp.cross(omega_b, radius_b)
        )

    sampled_minimum = wp.min(start_separation, wp.min(midpoint_separation, end_separation))
    return sampled_minimum - curvature_bound * prediction_dt * prediction_dt * 0.03125


@wp.func
def compute_speculative_admission_margin(
    physical_separation: float,
    predicted_closing_distance: float,
    max_speculative_extension: float,
) -> float:
    """Return remaining capped predictive reach beyond a physical separation [m]."""
    return wp.min(predicted_closing_distance, max_speculative_extension) - physical_separation


@wp.func
def compute_contact_predictive_score(
    shape_a: int,
    shape_b: int,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal_a_to_b: wp.vec3,
    normal_owner: int,
    physical_separation: float,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    collision_update_dt: float,
    max_speculative_extension: float,
) -> float:
    """Return the capped swept-motion admission margin over one collision-update horizon [m]."""
    total_separation_needed = wp.dot(point_b - point_a, normal_a_to_b) - physical_separation
    swept_separation = compute_contact_swept_separation_lower_bound(
        shape_a,
        shape_b,
        point_a,
        point_b,
        normal_a_to_b,
        normal_owner,
        total_separation_needed,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        collision_update_dt,
    )
    return compute_speculative_admission_margin(
        physical_separation,
        physical_separation - swept_separation,
        max_speculative_extension,
    )


@wp.func
def _contact_passes_gap_check_precomputed(
    contact_data: ContactData,
    contact_normal_a_to_b: wp.vec3,
    total_separation_needed: float,
) -> bool:
    """Check a contact gap using manifold-invariant precomputed values."""
    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    return d <= contact_data.gap_sum


@wp.func
def contact_passes_gap_check(
    contact_data: ContactData,
) -> bool:
    """
    Check if a contact passes the gap threshold check and should be written.

    Args:
        contact_data: ContactData struct containing contact information

    Returns:
        True if the contact distance is within the contact gap threshold, False otherwise
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )

    # Distance calculation matching box_plane_collision
    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    return _contact_passes_gap_check_precomputed(
        contact_data,
        contact_normal_a_to_b,
        total_separation_needed,
    )


@wp.func
def prepare_speculative_contact(contact_data: ContactData) -> tuple[wp.vec3, wp.vec3, wp.vec3, float]:
    """Return the normalized contact geometry used for speculative admission and storage."""
    normal = wp.normalize(contact_data.contact_normal_a_to_b)
    point_a = contact_data.contact_point_center - normal * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    point_b = contact_data.contact_point_center + normal * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )
    separation = wp.dot(point_b - point_a, normal) - total_separation_needed
    return normal, point_a, point_b, separation


@wp.func
def contact_passes_speculative_gap_check(
    contact_data: ContactData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    collision_update_dt: float,
    max_speculative_extension: float,
) -> bool:
    """Return whether a contact is present now or predicted before the next collision pass."""
    normal, point_a, point_b, physical_separation = prepare_speculative_contact(contact_data)
    if physical_separation <= contact_data.gap_sum:
        return True
    return (
        compute_contact_predictive_score(
            contact_data.shape_a,
            contact_data.shape_b,
            point_a,
            point_b,
            normal,
            unpack_contact_normal_owner(contact_data.sort_sub_key),
            physical_separation,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            shape_rotation_center_offset,
            collision_update_dt,
            max_speculative_extension,
        )
        >= 0.0
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for velocity-expanded rigid-contact candidate generation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import (
    CONTACT_NORMAL_OWNER_SHAPE_A,
    CONTACT_NORMAL_OWNER_SHAPE_B,
    ContactData,
    pack_contact_is_canonical_endpoint,
    pack_contact_is_strict_guard,
    pack_contact_normal_owner,
)
from newton._src.geometry.contact_reduction import FACE_NORMALS, get_face_normal
from newton._src.geometry.contact_reduction_global import (
    EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    PREDICTIVE_BIN_ID,
    PREDICTIVE_CONTACT_SLOTS,
    PREDICTIVE_CURRENT_SEPARATION_LOCAL_SLOT,
    PREDICTIVE_MOMENT_BIN_ID,
    VALUES_PER_KEY,
    GlobalContactReducer,
    GlobalContactReducerData,
    create_export_reduced_contacts_kernel,
    export_and_reduce_contact_centered_two_spatial_depths,
    export_and_reduce_predictive_contact,
    export_contact_to_buffer,
    make_contact_value,
    make_spatial_contact_value,
    reclaim_contact_id,
    reduce_buffered_contacts_speculative_kernel,
    reduction_finalize_slot,
    reduction_finish_contact_materialization,
    reduction_rollback_slot,
    reduction_try_update_slot,
)
from newton._src.geometry.contact_sort import ContactSorter
from newton._src.geometry.narrow_phase import (
    ContactWriterData,
    NarrowPhase,
    create_prepare_convex_pair,
    write_contact_simple,
)
from newton._src.geometry.sdf_contact import (
    mesh_sdf_contact_is_owned_endpoint,
    mesh_sdf_contact_voxel_owner,
)
from newton._src.geometry.types import GeoType
from newton._src.sim.collide import ContactWriterData as FullContactWriterData
from newton._src.sim.collide import write_contact_speculative
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

_prepare_speculative_convex_pair = create_prepare_convex_pair(
    external_aabb=True,
    speculative=True,
)


@wp.kernel(enable_backward=False)
def _write_finalized_guard_provenance_cases(writer_data: FullContactWriterData):
    """Write finalized and direct guard cases, including canonical endpoints."""
    case = wp.tid()
    contact = ContactData()
    contact.contact_normal_a_to_b = wp.vec3(1.0, 0.0, 0.0)
    contact.contact_distance = wp.where(case == 0, 0.01, -0.01)
    contact.shape_a = 0
    contact.shape_b = 1
    contact.sort_sub_key = pack_contact_is_strict_guard(case, case != 2)
    if case >= 3:
        contact.sort_sub_key = pack_contact_is_canonical_endpoint(contact.sort_sub_key, True)
    contact.strict_guard_provenance_finalized = 0 if case == 4 else 1
    write_contact_speculative(contact, writer_data, case)


@wp.kernel
def _classify_mesh_sdf_owned_endpoints(results: wp.array[wp.int32]):
    results[0] = int(mesh_sdf_contact_is_owned_endpoint(0, 0, 0, 1))
    results[1] = int(mesh_sdf_contact_is_owned_endpoint(0, 1, 0, 1))
    results[2] = int(mesh_sdf_contact_is_owned_endpoint(0, 2, 0, 1))
    results[3] = int(mesh_sdf_contact_is_owned_endpoint(1, 0, 0, 1))
    results[4] = int(mesh_sdf_contact_is_owned_endpoint(2, 0, 0, 1))


@wp.kernel
def _select_mesh_sdf_voxel_owners(
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
    results: wp.array[wp.int32],
):
    results[0] = mesh_sdf_contact_voxel_owner(
        0, 1, shape_collision_aabb_lower, shape_collision_aabb_upper, shape_voxel_resolution
    )
    results[1] = mesh_sdf_contact_voxel_owner(
        1, 0, shape_collision_aabb_lower, shape_collision_aabb_upper, shape_voxel_resolution
    )
    results[2] = mesh_sdf_contact_voxel_owner(
        2, 3, shape_collision_aabb_lower, shape_collision_aabb_upper, shape_voxel_resolution
    )
    results[3] = mesh_sdf_contact_voxel_owner(
        3, 2, shape_collision_aabb_lower, shape_collision_aabb_upper, shape_voxel_resolution
    )
    results[4] = mesh_sdf_contact_voxel_owner(
        4, 1, shape_collision_aabb_lower, shape_collision_aabb_upper, shape_voxel_resolution
    )


@wp.kernel
def _extract_speculative_plane_proxy_scale(
    shape_types: wp.array[wp.int32],
    shape_data: wp.array[wp.vec4],
    shape_transform: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_gap: wp.array[wp.float32],
    shape_collision_radius: wp.array[wp.float32],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    proxy_scale: wp.array[wp.vec3],
    valid_result: wp.array[wp.int32],
):
    valid, query = wp.static(_prepare_speculative_convex_pair)(
        wp.vec2i(0, 1),
        shape_types,
        shape_data,
        shape_transform,
        shape_source,
        shape_gap,
        shape_collision_radius,
        shape_aabb_lower,
        shape_aabb_upper,
        shape_collision_aabb_lower,
        shape_collision_aabb_upper,
    )
    valid_result[0] = int(valid)
    proxy_scale[0] = query.geom_a.scale


@wp.kernel
def _register_regular_and_predictive_contact(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    position = wp.vec3(0.0)
    normal = wp.vec3(1.0, 0.0, 0.0)
    contact_id = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        position,
        normal,
        0.05,
        17,
        position,
        0.0,
        0.1,
        False,
        False,
        1.0,
        position,
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[0] = contact_id
    contact_ids[1] = export_and_reduce_predictive_contact(
        0,
        1,
        position,
        normal,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
        17,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        contact_id,
        False,
        reducer_data,
    )
    contact_ids[2] = export_and_reduce_predictive_contact(
        0,
        1,
        position,
        normal,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
        17,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        contact_id,
        False,
        reducer_data,
    )


@wp.kernel
def _register_inside_fixed_gap_impact_guard(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    """Fill spatial slots, then register a translated non-extreme contact inside the fixed gap."""
    normal = wp.vec3(1.0, 0.0, 0.0)
    for candidate_idx in range(6):
        angle = float(candidate_idx) * wp.pi / 3.0
        position = wp.vec3(0.0, wp.cos(angle), wp.sin(angle))
        export_and_reduce_contact_centered_two_spatial_depths(
            0,
            1,
            position,
            normal,
            0.04,
            candidate_idx + 1,
            position,
            0.0,
            0.05,
            False,
            False,
            1.0,
            position,
            wp.vec3(-2.0),
            wp.vec3(2.0),
            wp.vec3i(1),
            reducer_data,
        )

    position = wp.vec3(0.0)
    regular_contact_id = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        position,
        normal,
        0.045,
        7,
        position,
        0.0,
        0.05,
        True,
        False,
        1.0,
        position,
        wp.vec3(-2.0),
        wp.vec3(2.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[0] = regular_contact_id
    contact_ids[1] = export_and_reduce_predictive_contact(
        0,
        1,
        position,
        normal,
        0.045,
        0.0,
        0.05,
        0.0,
        0.0,
        7,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.01,
        0.1,
        regular_contact_id,
        False,
        reducer_data,
    )


@wp.kernel
def _rank_rotating_anchor_end_separation(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    """Fill spatial slots, then rank two interior contacts whose linear and exact predictions disagree."""
    normal = wp.vec3(1.0, 0.0, 0.0)
    for candidate_idx in range(6):
        angle = float(candidate_idx) * wp.pi / 3.0
        position = wp.vec3(0.0, 2.0 * wp.cos(angle), 2.0 * wp.sin(angle))
        export_and_reduce_contact_centered_two_spatial_depths(
            0,
            1,
            position,
            normal,
            0.01,
            candidate_idx + 1,
            position,
            0.0,
            0.3,
            True,
            False,
            1.0,
            position,
            wp.vec3(-3.0),
            wp.vec3(3.0),
            wp.vec3i(1),
            reducer_data,
        )

    position_a = wp.vec3(0.0, 0.5, 0.0)
    fingerprint_a = pack_contact_normal_owner(7, CONTACT_NORMAL_OWNER_SHAPE_A)
    regular_a = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        position_a,
        normal,
        0.0497,
        fingerprint_a,
        position_a,
        0.0,
        0.3,
        True,
        False,
        1.0,
        position_a,
        wp.vec3(-3.0),
        wp.vec3(3.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[0] = regular_a
    contact_ids[1] = export_and_reduce_predictive_contact(
        0,
        1,
        position_a,
        normal,
        0.0497,
        0.0,
        0.3,
        0.0,
        0.0,
        fingerprint_a,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.01,
        0.3,
        regular_a,
        False,
        reducer_data,
    )

    position_b = wp.vec3(0.0, -0.5, 0.0)
    fingerprint_b = pack_contact_normal_owner(8, CONTACT_NORMAL_OWNER_SHAPE_A)
    regular_b = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        position_b,
        normal,
        0.25,
        fingerprint_b,
        position_b,
        0.0,
        0.3,
        True,
        False,
        1.0,
        position_b,
        wp.vec3(-3.0),
        wp.vec3(3.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[2] = regular_b
    contact_ids[3] = export_and_reduce_predictive_contact(
        0,
        1,
        position_b,
        normal,
        0.25,
        0.0,
        0.3,
        0.0,
        0.0,
        fingerprint_b,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.01,
        0.3,
        regular_b,
        False,
        reducer_data,
    )


@wp.kernel
def _register_future_feature_leading_candidate(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    """Register a rotating candidate that loses current support but leads at the horizon."""
    normal = wp.vec3(1.0, 0.0, 0.0)
    for candidate_idx in range(6):
        angle = float(candidate_idx) * wp.pi / 3.0
        position = wp.vec3(0.0, 0.001 * wp.cos(angle), 0.001 * wp.sin(angle))
        export_and_reduce_contact_centered_two_spatial_depths(
            0,
            1,
            position,
            normal,
            0.001,
            candidate_idx + 1,
            position,
            0.0,
            0.1,
            False,
            False,
            1.0,
            position,
            wp.vec3(-1.0),
            wp.vec3(1.0),
            wp.vec3i(1),
            reducer_data,
        )

    guard_position = wp.vec3(0.0)
    guard_fingerprint = pack_contact_normal_owner(7, CONTACT_NORMAL_OWNER_SHAPE_B)
    guard_regular_id = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        guard_position,
        normal,
        0.01,
        guard_fingerprint,
        guard_position,
        0.0,
        0.1,
        True,
        False,
        1.0,
        guard_position,
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[0] = export_and_reduce_predictive_contact(
        0,
        1,
        guard_position,
        normal,
        0.01,
        0.0,
        0.1,
        0.0,
        0.0,
        guard_fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.01,
        0.1,
        guard_regular_id,
        False,
        reducer_data,
    )

    leading_position = wp.vec3(0.0)
    leading_fingerprint = pack_contact_normal_owner(8, CONTACT_NORMAL_OWNER_SHAPE_B)
    leading_regular_id = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        leading_position,
        normal,
        0.05,
        leading_fingerprint,
        leading_position,
        0.0,
        0.1,
        True,
        False,
        1.0,
        leading_position,
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[1] = leading_regular_id
    contact_ids[2] = export_and_reduce_predictive_contact(
        0,
        1,
        leading_position,
        normal,
        0.05,
        0.0,
        0.1,
        0.0,
        0.0,
        leading_fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.01,
        0.1,
        leading_regular_id,
        False,
        reducer_data,
    )


@wp.kernel
def _register_outer_ring_around_closest_contact(
    reducer_data: GlobalContactReducerData,
    contact_ids: wp.array[wp.int32],
):
    normal = wp.vec3(0.0, 0.0, 1.0)
    contact_ids[0] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        wp.vec3(0.0, 0.0, 0.001865),
        normal,
        0.001865,
        1,
        wp.vec3(0.0),
        0.001,
        0.01,
        False,
        False,
        1.0,
        wp.vec3(0.0),
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(1),
        reducer_data,
    )
    for candidate_idx in range(6):
        angle = float(candidate_idx) * wp.pi / 3.0
        centered_position = wp.vec3(0.01 * wp.cos(angle), 0.01 * wp.sin(angle), 0.0)
        contact_ids[candidate_idx + 1] = export_and_reduce_contact_centered_two_spatial_depths(
            0,
            1,
            centered_position + wp.vec3(0.0, 0.0, 0.005),
            normal,
            0.005,
            candidate_idx + 2,
            centered_position,
            0.001,
            0.01,
            False,
            False,
            1.0,
            wp.vec3(0.0),
            wp.vec3(-1.0),
            wp.vec3(1.0),
            wp.vec3i(1),
            reducer_data,
        )


@wp.kernel
def _register_inner_and_rotating_leading_contact(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    normal = wp.vec3(1.0, 0.0, 0.0)
    inner_position = wp.vec3(0.0)
    contact_ids[0] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        inner_position,
        normal,
        -0.01,
        11,
        inner_position,
        0.01,
        0.1,
        False,
        False,
        1.0,
        inner_position,
        wp.vec3(-2.0),
        wp.vec3(2.0),
        wp.vec3i(1),
        reducer_data,
    )

    leading_position = wp.vec3(0.0, 1.0, 0.0)
    regular_contact_id = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        leading_position,
        normal,
        0.05,
        23,
        leading_position,
        0.01,
        0.1,
        True,
        False,
        1.0,
        leading_position,
        wp.vec3(-2.0),
        wp.vec3(2.0),
        wp.vec3i(1),
        reducer_data,
    )
    contact_ids[1] = regular_contact_id
    contact_ids[2] = export_and_reduce_predictive_contact(
        0,
        1,
        leading_position,
        normal,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
        23,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        regular_contact_id,
        False,
        reducer_data,
    )


@wp.func
def _predictive_axis_position(candidate_idx: int) -> wp.vec3:
    """Return one of the six unit world-axis support positions."""
    if candidate_idx == 0:
        return wp.vec3(1.0, 0.0, 0.0)
    if candidate_idx == 1:
        return wp.vec3(-1.0, 0.0, 0.0)
    if candidate_idx == 2:
        return wp.vec3(0.0, 1.0, 0.0)
    if candidate_idx == 3:
        return wp.vec3(0.0, -1.0, 0.0)
    if candidate_idx == 4:
        return wp.vec3(0.0, 0.0, 1.0)
    return wp.vec3(0.0, 0.0, -1.0)


@wp.func
def _predictive_wrench_position(candidate_idx: int) -> wp.vec3:
    """Return the pair-centered position for one predictive wrench winner."""
    diagonal = wp.sqrt(0.5)
    if candidate_idx == 0:
        return wp.vec3(0.0, diagonal, -diagonal)
    if candidate_idx == 1:
        return wp.vec3(0.0, -diagonal, diagonal)
    if candidate_idx == 2:
        return wp.vec3(-diagonal, 0.0, diagonal)
    if candidate_idx == 3:
        return wp.vec3(diagonal, 0.0, -diagonal)
    if candidate_idx == 4:
        return wp.vec3(diagonal, -diagonal, 0.0)
    if candidate_idx == 5:
        return wp.vec3(-diagonal, diagonal, 0.0)
    return wp.vec3(0.0)


@wp.func
def _predictive_wrench_normal(candidate_idx: int) -> wp.vec3:
    """Return a normal producing one normal or lever-arm moment extreme."""
    diagonal = wp.sqrt(0.5)
    if candidate_idx < 2:
        return wp.vec3(0.0, diagonal, diagonal)
    if candidate_idx < 4:
        return wp.vec3(diagonal, 0.0, diagonal)
    if candidate_idx < 6:
        return wp.vec3(diagonal, diagonal, 0.0)
    if candidate_idx < 12:
        return _predictive_axis_position(candidate_idx - 6)
    return wp.normalize(wp.vec3(1.0, 1.0, 1.0))


@wp.kernel
def _register_predictive_wrench_candidates(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    for candidate_idx in range(14):
        clearance = float(0.05)
        fingerprint = candidate_idx + 1
        if candidate_idx == 12:
            clearance = 0.01
        elif candidate_idx == 13:
            clearance = 0.08
        contact_ids[candidate_idx] = export_and_reduce_predictive_contact(
            0,
            1,
            _predictive_wrench_position(candidate_idx),
            _predictive_wrench_normal(candidate_idx),
            clearance,
            0.0,
            0.1,
            0.0,
            0.0,
            fingerprint,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            shape_rotation_center_offset,
            0.1,
            0.1,
            -1,
            False,
            reducer_data,
        )


@wp.kernel
def _register_predictive_moment_extreme_with_colliding_fingerprints(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
):
    normal = wp.vec3(1.0, 0.0, 0.0)
    for candidate_idx in range(3):
        clearance = 0.01 + float(candidate_idx) * 0.01
        position_y = float(candidate_idx) * 0.05
        fingerprint = int(12)
        if candidate_idx == 1:
            fingerprint = 18
        elif candidate_idx == 2:
            clearance = 0.08
            position_y = 1.0
            fingerprint = 7
        export_and_reduce_predictive_contact(
            0,
            1,
            wp.vec3(0.0, position_y, 0.0),
            normal,
            clearance,
            0.0,
            0.0,
            0.0,
            0.0,
            fingerprint,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            shape_rotation_center_offset,
            0.1,
            0.1,
            -1,
            False,
            reducer_data,
        )


@wp.kernel
def _register_predictive_wrench_candidates_contended(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    candidate_order: int,
):
    candidate_idx = wp.tid()
    logical_idx = candidate_idx
    if candidate_order == 1:
        logical_idx = 255 - candidate_idx
    elif candidate_order == 2:
        logical_idx = (candidate_idx * 73) % 256

    clearance = 0.09
    fingerprint = logical_idx + 1
    if logical_idx < 12:
        clearance = 0.05
    elif logical_idx == 12:
        clearance = 0.01
    export_and_reduce_predictive_contact(
        0,
        1,
        _predictive_wrench_position(logical_idx),
        _predictive_wrench_normal(logical_idx),
        clearance,
        0.0,
        0.1,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        -1,
        False,
        reducer_data,
    )


@wp.kernel
def _register_predictive_wrench_adversary(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
    target_fingerprint: wp.array[wp.int32],
):
    """Surround an interior lever-arm extreme with position and separation winners."""
    for candidate_idx in range(6):
        position = _predictive_axis_position(candidate_idx)
        contact_ids[candidate_idx] = export_and_reduce_predictive_contact(
            0,
            1,
            position,
            position,
            0.05,
            0.0,
            0.1,
            0.0,
            0.0,
            candidate_idx + 1,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            shape_rotation_center_offset,
            0.1,
            0.1,
            -1,
            False,
            reducer_data,
        )

    contact_ids[6] = export_and_reduce_predictive_contact(
        0,
        1,
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.01,
        0.0,
        0.1,
        0.0,
        0.0,
        7,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        -1,
        False,
        reducer_data,
    )

    # Edge 37, SDF mode 1, high endpoint. Its position loses all six
    # position supports and its clearance loses the guard, but r x n is a
    # unique negative-Z wrench feature. Keep the packed owner/mode/endpoint
    # fingerprint intact when the candidate becomes a winner.
    fingerprint = pack_contact_normal_owner((37 << 3) | (1 << 2) | 3, CONTACT_NORMAL_OWNER_SHAPE_A)
    target_fingerprint[0] = fingerprint
    contact_ids[7] = export_and_reduce_predictive_contact(
        0,
        1,
        wp.vec3(0.5, 0.5, 0.0),
        wp.normalize(wp.vec3(1.0, -1.0, 0.0)),
        0.05,
        0.0,
        0.1,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        -1,
        False,
        reducer_data,
    )


@wp.kernel
def _register_predictive_crossing_voxel_adversary(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
    target_fingerprint: wp.array[wp.int32],
):
    """Surround an outer crossing contact in every pair-level support direction."""
    normal = get_face_normal(2)
    ref = wp.vec3(0.0, 1.0, 0.0)
    basis_u = wp.normalize(ref - wp.dot(ref, normal) * normal)
    basis_v = wp.cross(normal, basis_u)
    aabb_lower = wp.vec3(-1.2)
    aabb_upper = wp.vec3(1.2)
    voxel_res = wp.vec3i(5, 5, 4)

    # These six contacts beat the target in every ordinary spatial direction
    # and in every component of its lever-arm moment. Their radius remains
    # smaller than the target's radius between two sampled directions, so they
    # do not geometrically cover it.
    for candidate_idx in range(6):
        angle = float(candidate_idx) * wp.pi / 3.0
        position = 0.9 * (wp.cos(angle) * basis_u + wp.sin(angle) * basis_v)
        fingerprint = pack_contact_normal_owner(100 + candidate_idx, CONTACT_NORMAL_OWNER_SHAPE_A)
        regular_id = export_and_reduce_contact_centered_two_spatial_depths(
            0,
            1,
            position,
            normal,
            0.05,
            fingerprint,
            position,
            0.0,
            0.1,
            True,
            False,
            -0.01,
            position,
            aabb_lower,
            aabb_upper,
            voxel_res,
            reducer_data,
        )
        contact_ids[candidate_idx] = export_and_reduce_predictive_contact(
            0,
            1,
            position,
            normal,
            0.05,
            0.0,
            0.1,
            0.0,
            0.0,
            fingerprint,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            shape_rotation_center_offset,
            1.0,
            0.1,
            regular_id,
            False,
            reducer_data,
        )

    # Win the single least-swept-separation slot without adding a spatial or
    # moment extreme.
    guard_position = wp.vec3(0.0)
    guard_fingerprint = pack_contact_normal_owner(200, CONTACT_NORMAL_OWNER_SHAPE_A)
    guard_regular_id = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        guard_position,
        normal,
        0.001,
        guard_fingerprint,
        guard_position,
        0.0,
        0.1,
        True,
        False,
        -0.059,
        guard_position,
        aabb_lower,
        aabb_upper,
        voxel_res,
        reducer_data,
    )
    contact_ids[6] = export_and_reduce_predictive_contact(
        0,
        1,
        guard_position,
        normal,
        0.001,
        0.0,
        0.1,
        0.0,
        0.0,
        guard_fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        1.0,
        0.1,
        guard_regular_id,
        False,
        reducer_data,
    )

    # This endpoint is predicted to cross, is an exposed support point between
    # two sampled directions, and occupies a different existing voxel. The
    # pair-level normal/moment/depth manifold nevertheless drops it.
    target_position = wp.cos(wp.pi / 6.0) * basis_u + wp.sin(wp.pi / 6.0) * basis_v
    fingerprint = pack_contact_normal_owner(1, CONTACT_NORMAL_OWNER_SHAPE_A)
    target_fingerprint[0] = fingerprint
    contact_ids[7] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        target_position,
        normal,
        0.05,
        fingerprint,
        target_position,
        0.0,
        0.1,
        True,
        True,
        -0.01,
        target_position,
        aabb_lower,
        aabb_upper,
        voxel_res,
        reducer_data,
    )
    contact_ids[8] = export_and_reduce_predictive_contact(
        0,
        1,
        target_position,
        normal,
        0.05,
        0.0,
        0.1,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        1.0,
        0.1,
        contact_ids[7],
        False,
        reducer_data,
    )


@wp.kernel
def _register_authored_shell_voxel_adversary(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    is_owned_endpoint: bool,
    contact_ids: wp.array[wp.int32],
    target_fingerprint: wp.array[wp.int32],
):
    """Suppress an authored-shell candidate from every non-voxel reduction lane."""
    normal = get_face_normal(2)
    ref = wp.vec3(0.0, 1.0, 0.0)
    basis_u = wp.normalize(ref - wp.dot(ref, normal) * normal)
    basis_v = wp.cross(normal, basis_u)
    aabb_lower = wp.vec3(-1.2)
    aabb_upper = wp.vec3(1.2)
    voxel_res = wp.vec3i(5, 5, 4)

    # Fill all ordinary spatial lanes without claiming voxel coverage.
    for candidate_idx in range(6):
        angle = float(candidate_idx) * wp.pi / 3.0
        position = 0.9 * (wp.cos(angle) * basis_u + wp.sin(angle) * basis_v)
        export_and_reduce_contact_centered_two_spatial_depths(
            0,
            1,
            position,
            normal,
            0.05,
            100 + candidate_idx,
            position,
            0.0,
            0.1,
            False,
            False,
            1.0,
            position,
            aabb_lower,
            aabb_upper,
            voxel_res,
            reducer_data,
        )

    # Fill every pair-level normal and moment support, plus the minimum
    # swept-separation guard, with scores strictly ahead of the target.
    for candidate_idx in range(13):
        clearance = 0.05
        if candidate_idx == 12:
            clearance = 0.01
        export_and_reduce_predictive_contact(
            0,
            1,
            _predictive_wrench_position(candidate_idx),
            _predictive_wrench_normal(candidate_idx),
            clearance,
            0.0,
            0.1,
            0.0,
            0.0,
            200 + candidate_idx,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            shape_rotation_center_offset,
            0.1,
            0.1,
            -1,
            False,
            reducer_data,
        )

    # This non-crossing candidate lies inside the authored gap and occupies a
    # distinct voxel, but loses all ordinary and pair-level support scores.
    target_position = wp.cos(wp.pi / 6.0) * basis_u + wp.sin(wp.pi / 6.0) * basis_v
    fingerprint = pack_contact_normal_owner(1, CONTACT_NORMAL_OWNER_SHAPE_A)
    target_fingerprint[0] = fingerprint
    contact_ids[0] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        target_position,
        normal,
        0.06,
        fingerprint,
        target_position,
        0.0,
        0.1,
        True,
        is_owned_endpoint,
        0.06,
        target_position,
        aabb_lower,
        aabb_upper,
        voxel_res,
        reducer_data,
    )
    contact_ids[1] = export_and_reduce_predictive_contact(
        0,
        1,
        target_position,
        normal,
        0.06,
        0.0,
        0.1,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        contact_ids[0],
        not is_owned_endpoint,
        reducer_data,
    )


@wp.kernel
def _replace_dedicated_endpoint_with_ordinary_equivalent(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    """Replace a dedicated endpoint winner after saturating pair guard lanes."""
    normal = get_face_normal(2)
    ref = wp.vec3(0.0, 1.0, 0.0)
    basis_u = wp.normalize(ref - wp.dot(ref, normal) * normal)
    basis_v = wp.cross(normal, basis_u)
    aabb_lower = wp.vec3(-1.2)
    aabb_upper = wp.vec3(1.2)
    voxel_res = wp.vec3i(5, 5, 4)

    for candidate_idx in range(6):
        angle = float(candidate_idx) * wp.pi / 3.0
        position = 0.9 * (wp.cos(angle) * basis_u + wp.sin(angle) * basis_v)
        export_and_reduce_contact_centered_two_spatial_depths(
            0,
            1,
            position,
            normal,
            0.05,
            100 + candidate_idx,
            position,
            0.0,
            0.1,
            False,
            False,
            1.0,
            position,
            aabb_lower,
            aabb_upper,
            voxel_res,
            reducer_data,
        )

    for candidate_idx in range(13):
        clearance = 0.01 if candidate_idx == 12 else 0.05
        export_and_reduce_predictive_contact(
            0,
            1,
            _predictive_wrench_position(candidate_idx),
            _predictive_wrench_normal(candidate_idx),
            clearance,
            0.0,
            0.1,
            0.0,
            0.0,
            200 + candidate_idx,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            shape_rotation_center_offset,
            0.1,
            0.1,
            -1,
            False,
            reducer_data,
        )

    position = wp.cos(wp.pi / 6.0) * basis_u + wp.sin(wp.pi / 6.0) * basis_v
    ordinary_fingerprint = pack_contact_normal_owner(1, CONTACT_NORMAL_OWNER_SHAPE_A)
    ordinary_fingerprint = pack_contact_is_canonical_endpoint(ordinary_fingerprint, True)
    dedicated_fingerprint = pack_contact_is_strict_guard(ordinary_fingerprint, True)
    contact_ids[0] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        position,
        normal,
        0.06,
        dedicated_fingerprint,
        position,
        0.0,
        0.0,
        True,
        True,
        0.06,
        position,
        aabb_lower,
        aabb_upper,
        voxel_res,
        reducer_data,
    )
    contact_ids[2] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        position,
        normal,
        0.059,
        ordinary_fingerprint,
        position,
        0.0,
        0.1,
        True,
        True,
        0.059,
        position,
        aabb_lower,
        aabb_upper,
        voxel_res,
        reducer_data,
    )


@wp.kernel
def _register_inner_and_crossing_voxel_competitors(
    reducer_data: GlobalContactReducerData,
    contact_ids: wp.array[wp.int32],
):
    """Compete a crossing outer candidate with inner coverage in the same and a distinct voxel."""
    normal = wp.vec3(0.0, 0.0, 1.0)
    aabb_lower = wp.vec3(-1.0)
    aabb_upper = wp.vec3(1.0)
    voxel_res = wp.vec3i(2, 1, 1)
    inner_position = wp.vec3(0.0)
    contact_ids[0] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        inner_position,
        normal,
        0.01,
        11,
        inner_position,
        0.02,
        0.04,
        False,
        False,
        1.0,
        inner_position,
        aabb_lower,
        aabb_upper,
        voxel_res,
        reducer_data,
    )
    contact_ids[1] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        inner_position,
        normal,
        0.05,
        12,
        inner_position,
        0.02,
        0.04,
        True,
        False,
        -0.5,
        inner_position,
        aabb_lower,
        aabb_upper,
        voxel_res,
        reducer_data,
    )
    crossing_position = wp.vec3(-0.8, 0.0, 0.0)
    contact_ids[2] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        crossing_position,
        normal,
        0.05,
        13,
        crossing_position,
        0.02,
        0.04,
        True,
        False,
        -0.02,
        crossing_position,
        aabb_lower,
        aabb_upper,
        voxel_res,
        reducer_data,
    )


@wp.kernel
def _buffer_one_contact(reducer_data: GlobalContactReducerData):
    export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.25, -0.5, 0.75),
        wp.vec3(1.0, 0.0, 0.0),
        -0.01,
        7,
        reducer_data,
    )


@wp.kernel
def _buffer_separated_axial_contact(reducer_data: GlobalContactReducerData):
    """Buffer one separated axial-shape contact produced by a mesh triangle."""
    export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.05,
        7,
        reducer_data,
    )


@wp.kernel
def _replace_full_buffer_predictive_winner(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
    contact_ids: wp.array[wp.int32],
):
    normal = wp.vec3(1.0, 0.0, 0.0)
    fingerprint = int(7)
    original_clearance = 0.08
    original_contact_id = export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.0, 0.0, original_clearance),
        normal,
        original_clearance,
        fingerprint,
        reducer_data,
    )
    contact_ids[0] = export_and_reduce_predictive_contact(
        0,
        1,
        wp.vec3(0.0, 0.0, original_clearance),
        normal,
        original_clearance,
        0.0,
        0.0,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        original_contact_id,
        False,
        reducer_data,
    )
    contact_ids[1] = export_and_reduce_predictive_contact(
        0,
        1,
        wp.vec3(0.0, 0.0, 0.05),
        normal,
        0.05,
        0.0,
        0.0,
        0.0,
        0.0,
        fingerprint,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
        shape_rotation_center_offset,
        0.1,
        0.1,
        -1,
        False,
        reducer_data,
    )


@wp.kernel
def _replace_validated_predictive_claims(
    reducer_data: GlobalContactReducerData,
    allocated_ids: wp.array[wp.int32],
):
    normal_entry_idx = int(0)
    impact_entry_idx = int(1)
    spatial_slot = int(0)
    impact_slot = int(6)
    fingerprint = int(7)
    provisional_spatial = make_contact_value(-0.1, fingerprint, 0, reducer_data.deterministic)
    provisional_impact = make_contact_value(0.1, fingerprint, 0, reducer_data.deterministic)
    spatial_idx = spatial_slot * reducer_data.ht_capacity + normal_entry_idx
    impact_idx = impact_slot * reducer_data.ht_capacity + impact_entry_idx
    reducer_data.ht_values[spatial_idx] = provisional_spatial
    reducer_data.ht_values[impact_idx] = provisional_impact

    contact_id = export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.1,
        fingerprint,
        reducer_data,
    )

    # Reproduce two stronger contenders replacing both claims after validation.
    reducer_data.ht_values[spatial_idx] = make_contact_value(0.0, fingerprint + 1, 0, reducer_data.deterministic)
    reducer_data.ht_values[impact_idx] = make_contact_value(0.2, fingerprint + 1, 0, reducer_data.deterministic)
    spatial_final = make_contact_value(-0.1, fingerprint, contact_id, reducer_data.deterministic)
    impact_final = make_contact_value(0.1, fingerprint, contact_id, reducer_data.deterministic)
    retained = reduction_finalize_slot(
        normal_entry_idx,
        spatial_slot,
        provisional_spatial,
        spatial_final,
        reducer_data.ht_values,
        reducer_data.ht_capacity,
    )
    if reduction_finalize_slot(
        impact_entry_idx,
        impact_slot,
        provisional_impact,
        impact_final,
        reducer_data.ht_values,
        reducer_data.ht_capacity,
    ):
        retained = True
    if not retained:
        reclaim_contact_id(contact_id, reducer_data)

    for i in range(reducer_data.capacity):
        allocated_ids[i] = export_contact_to_buffer(
            0,
            1,
            wp.vec3(float(i), 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            0.0,
            i,
            reducer_data,
        )


@wp.kernel
def _rollback_replaced_provisional_claims(
    reducer_data: GlobalContactReducerData,
    rolled_back_values: wp.array[wp.uint64],
):
    """Stage two rollback interleavings without relying on thread scheduling."""
    predecessor_provisional = make_spatial_contact_value(-0.1, True, 7, 0, reducer_data.deterministic)
    predecessor_materialized = make_spatial_contact_value(-0.1, True, 7, 3, reducer_data.deterministic)
    replacement = make_spatial_contact_value(0.0, True, 8, 0, reducer_data.deterministic)

    reducer_data.ht_values[0] = predecessor_provisional
    previous_value = reduction_try_update_slot(0, 0, replacement, reducer_data.ht_values, reducer_data.ht_capacity)
    reduction_rollback_slot(
        0,
        0,
        replacement,
        previous_value,
        reducer_data.ht_values,
        reducer_data.ht_capacity,
        reducer_data.deterministic,
    )
    rolled_back_values[0] = reducer_data.ht_values[0]

    reducer_data.ht_values[reducer_data.ht_capacity] = predecessor_materialized
    previous_value = reduction_try_update_slot(0, 1, replacement, reducer_data.ht_values, reducer_data.ht_capacity)
    reduction_rollback_slot(
        0,
        1,
        replacement,
        previous_value,
        reducer_data.ht_values,
        reducer_data.ht_capacity,
        reducer_data.deterministic,
    )
    rolled_back_values[1] = reducer_data.ht_values[reducer_data.ht_capacity]


@wp.kernel
def _replace_validated_two_depth_claim(
    reducer_data: GlobalContactReducerData,
    completed_id: wp.array[wp.int32],
    allocated_ids: wp.array[wp.int32],
):
    """Replace a spatial claim after allocation, then complete the transaction."""
    fingerprint = int(7)
    provisional = make_spatial_contact_value(-0.1, True, fingerprint, 0, reducer_data.deterministic)
    reducer_data.ht_values[0] = provisional
    contact_id = export_contact_to_buffer(
        0,
        1,
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.1,
        fingerprint,
        reducer_data,
    )

    # A stronger candidate replaces the only claim after allocation but
    # before the materialized ID is published.
    reducer_data.ht_values[0] = make_spatial_contact_value(0.0, True, fingerprint + 1, 0, reducer_data.deterministic)
    final_value = make_spatial_contact_value(-0.1, True, fingerprint, contact_id, reducer_data.deterministic)
    retained = reduction_finalize_slot(
        0,
        0,
        provisional,
        final_value,
        reducer_data.ht_values,
        reducer_data.ht_capacity,
    )
    completed_id[0] = reduction_finish_contact_materialization(contact_id, retained, reducer_data)

    for i in range(reducer_data.capacity):
        allocated_ids[i] = export_contact_to_buffer(
            0,
            1,
            wp.vec3(float(i), 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            0.0,
            i,
            reducer_data,
        )


@wp.kernel
def _register_crossing_with_full_hashtable(
    reducer_data: GlobalContactReducerData,
    contact_id: wp.array[wp.int32],
):
    position = wp.vec3(0.0)
    contact_id[0] = export_and_reduce_contact_centered_two_spatial_depths(
        0,
        1,
        position,
        wp.vec3(1.0, 0.0, 0.0),
        0.2,
        7,
        position,
        0.0,
        0.1,
        True,
        False,
        -0.01,
        position,
        wp.vec3(-1.0),
        wp.vec3(1.0),
        wp.vec3i(1),
        reducer_data,
    )


@wp.kernel
def _reclaim_and_allocate_contact_ids(
    reducer_data: GlobalContactReducerData,
    allocated_ids: wp.array[wp.int32],
):
    """Reclaim one ID per thread and immediately contend for the available IDs."""
    tid = wp.tid()
    reclaim_contact_id(tid + 1, reducer_data)
    allocated_ids[tid] = export_contact_to_buffer(
        0,
        1,
        wp.vec3(float(tid), 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.0,
        tid,
        reducer_data,
    )


@wp.kernel
def _register_predictive_wrench_candidates_sequential(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_rotation_center_offset: wp.array[wp.vec3],
):
    candidate_idx = int(0)
    while candidate_idx < 256:
        clearance = 0.09
        fingerprint = candidate_idx + 1
        if candidate_idx < 12:
            clearance = 0.05
        elif candidate_idx == 12:
            clearance = 0.01
        export_and_reduce_predictive_contact(
            0,
            1,
            _predictive_wrench_position(candidate_idx),
            _predictive_wrench_normal(candidate_idx),
            clearance,
            0.0,
            0.1,
            0.0,
            0.0,
            fingerprint,
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            shape_rotation_center_offset,
            0.1,
            0.1,
            -1,
            False,
            reducer_data,
        )
        candidate_idx = candidate_idx + 1


def _assert_predictive_wrench_manifold(test, exported_count: int, positions: np.ndarray):
    """Require twelve wrench-feature extrema and one swept-separation guard."""
    test.assertEqual(exported_count, 13)
    radial_distance = np.linalg.norm(positions[:exported_count], axis=1)
    test.assertEqual(int(np.count_nonzero(np.isclose(radial_distance, 1.0, rtol=0.0, atol=1.0e-6))), 6)
    test.assertEqual(int(np.count_nonzero(np.isclose(radial_distance, 0.0, rtol=0.0, atol=1.0e-6))), 7)


_export_reduced_contacts = create_export_reduced_contacts_kernel(write_contact_simple)
_export_reduced_contacts_full = create_export_reduced_contacts_kernel(write_contact_speculative)


def _make_predictive_reducer(capacity, device, deterministic):
    """Create reducer storage for predictive reservation tests."""
    return GlobalContactReducer(
        capacity=capacity,
        device=device,
        deterministic=deterministic,
        enable_contact_reclamation=True,
    )


def _predictive_slot_contact_id(reducer: GlobalContactReducer, slot: int) -> int:
    """Return the one-based contact ID stored in a pair's predictive slot."""
    keys = reducer.hashtable.keys.numpy()
    bins = (keys >> np.uint64(55)) & np.uint64(0xFF)
    predictive_entries = np.flatnonzero(bins == PREDICTIVE_BIN_ID)
    if len(predictive_entries) != 1:
        raise AssertionError(f"expected one predictive entry, found {len(predictive_entries)}")
    value = reducer.ht_values.numpy()[slot * reducer.hashtable.capacity + int(predictive_entries[0])]
    contact_id_mask = (1 << (20 if reducer.deterministic else 32)) - 1
    return int(value & np.uint64(contact_id_mask))


def _build_spheres(device, velocity: float, separation: float = 0.3, gap: float = 0.0):
    """Build two spheres separated along X with the first sphere moving."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = gap
    body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
    builder.add_shape_sphere(body_a, radius=0.1)
    builder.body_qd[body_a] = (velocity, 0.0, 0.0, 0.0, 0.0, 0.0)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(separation, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    model = builder.finalize(device=device)
    return model, model.state()


def _collide(model, state, speculative: bool):
    """Run one collision pass and return the populated contact buffer."""
    config = None
    if speculative:
        config = newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.25,
        )
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", speculative_config=config)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, dt=0.02)
    return contacts


def _export_reducer_contacts(reducer: GlobalContactReducer, device):
    """Export reducer winners and return their count and world positions."""
    capacity = 32
    contact_count = wp.zeros(1, dtype=wp.int32, device=device)
    contact_position = wp.zeros(capacity, dtype=wp.vec3, device=device)
    writer_data = ContactWriterData()
    writer_data.contact_max = capacity
    writer_data.contact_count = contact_count
    writer_data.contact_pair = wp.zeros(capacity, dtype=wp.vec2i, device=device)
    writer_data.contact_position = contact_position
    writer_data.contact_normal = wp.zeros(capacity, dtype=wp.vec3, device=device)
    writer_data.contact_penetration = wp.zeros(capacity, dtype=wp.float32, device=device)
    writer_data.contact_tangent = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.contact_sort_key = wp.empty(0, dtype=wp.int64, device=device)
    shape_gap = wp.full(2, 0.1, dtype=wp.float32, device=device)
    writer_data.shape_gap = shape_gap
    writer_data.shape_transform = wp.empty(0, dtype=wp.transform, device=device)
    writer_data.shape_linear_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.shape_angular_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.shape_rotation_center_offset = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.collision_update_dt = 0.0
    writer_data.max_speculative_extension = 0.0
    reducer.exported_flags.zero_()
    total_blocks = 128
    wp.launch_tiled(
        _export_reduced_contacts,
        dim=total_blocks,
        inputs=[
            reducer.hashtable.keys,
            reducer.ht_values,
            reducer.hashtable.active_slots,
            reducer.position_depth,
            reducer.normal,
            reducer.shape_pairs,
            reducer.contact_fingerprints,
            reducer.exported_flags,
            wp.zeros(2, dtype=wp.int32, device=device),
            wp.zeros(2, dtype=wp.vec4, device=device),
            shape_gap,
            writer_data,
            total_blocks,
            int(not device.is_cpu),
            int(reducer.deterministic),
        ],
        device=device,
        block_dim=EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    )
    return int(contact_count.numpy()[0]), contact_position.numpy()


def _export_reducer_provenance(reducer: GlobalContactReducer, device):
    """Export reducer winners through the full writer and return provenance and sort keys."""
    capacity = 32
    contacts = newton.Contacts(capacity, 0, device=device)
    writer_data = FullContactWriterData()
    writer_data.contact_max = capacity
    writer_data.body_q = wp.empty(0, dtype=wp.transform, device=device)
    writer_data.shape_body = wp.full(2, -1, dtype=wp.int32, device=device)
    writer_data.shape_gap = wp.full(2, 0.1, dtype=wp.float32, device=device)
    writer_data.contact_count = contacts.rigid_contact_count
    writer_data.out_shape0 = contacts.rigid_contact_shape0
    writer_data.out_shape1 = contacts.rigid_contact_shape1
    writer_data.out_point0 = contacts.rigid_contact_point0
    writer_data.out_point1 = contacts.rigid_contact_point1
    writer_data.out_offset0 = contacts.rigid_contact_offset0
    writer_data.out_offset1 = contacts.rigid_contact_offset1
    writer_data.out_normal = contacts.rigid_contact_normal
    writer_data.out_normal_owner = contacts.rigid_contact_normal_owner
    writer_data.out_is_predictive = contacts.rigid_contact_is_predictive
    writer_data.out_is_strict_guard = contacts.rigid_contact_is_strict_guard
    writer_data.out_margin0 = contacts.rigid_contact_margin0
    writer_data.out_margin1 = contacts.rigid_contact_margin1
    writer_data.out_tids = contacts.rigid_contact_tids
    writer_data.out_stiffness = wp.empty(0, dtype=wp.float32, device=device)
    writer_data.out_damping = wp.empty(0, dtype=wp.float32, device=device)
    writer_data.out_friction = wp.empty(0, dtype=wp.float32, device=device)
    contact_sort_key = wp.zeros(capacity, dtype=wp.int64, device=device)
    writer_data.out_sort_key = contact_sort_key
    writer_data.shape_transform = wp.empty(0, dtype=wp.transform, device=device)
    writer_data.shape_linear_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.shape_angular_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.shape_rotation_center_offset = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.collision_update_dt = 0.0
    writer_data.max_speculative_extension = 0.0

    shape_gap = writer_data.shape_gap
    reducer.exported_flags.zero_()
    total_blocks = 128
    wp.launch_tiled(
        _export_reduced_contacts_full,
        dim=total_blocks,
        inputs=[
            reducer.hashtable.keys,
            reducer.ht_values,
            reducer.hashtable.active_slots,
            reducer.position_depth,
            reducer.normal,
            reducer.shape_pairs,
            reducer.contact_fingerprints,
            reducer.exported_flags,
            wp.zeros(2, dtype=wp.int32, device=device),
            wp.zeros(2, dtype=wp.vec4, device=device),
            shape_gap,
            writer_data,
            total_blocks,
            int(not device.is_cpu),
            int(reducer.deterministic),
        ],
        device=device,
        block_dim=EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    )
    count = int(contacts.rigid_contact_count.numpy()[0])
    return (
        count,
        contacts.rigid_contact_is_predictive.numpy()[:count],
        contacts.rigid_contact_is_strict_guard.numpy()[:count],
        contact_sort_key.numpy()[:count],
    )


def test_mesh_sdf_candidate_endpoint_tagging(test, device):
    """Tag explicit endpoints and endpoint minima without tagging edge interiors."""
    results = wp.zeros(5, dtype=wp.int32, device=device)
    wp.launch(_classify_mesh_sdf_owned_endpoints, dim=1, inputs=[results], device=device)
    np.testing.assert_array_equal(results.numpy(), (0, 1, 1, 1, 1))


def test_mesh_sdf_voxel_owner_is_fine_and_pair_order_invariant(test, device):
    """Choose the finer pair grid with stable ties and a safe degenerate fallback."""
    aabb_lower = wp.zeros(5, dtype=wp.vec3, device=device)
    aabb_upper = wp.array(
        [
            wp.vec3(2.0),
            wp.vec3(0.4),
            wp.vec3(1.0),
            wp.vec3(2.0),
            wp.vec3(0.0),
        ],
        dtype=wp.vec3,
        device=device,
    )
    voxel_resolution = wp.array(
        [
            wp.vec3i(2),
            wp.vec3i(4),
            wp.vec3i(2),
            wp.vec3i(4),
            wp.vec3i(4),
        ],
        dtype=wp.vec3i,
        device=device,
    )
    results = wp.full(5, -1, dtype=wp.int32, device=device)
    wp.launch(
        _select_mesh_sdf_voxel_owners,
        dim=1,
        inputs=[aabb_lower, aabb_upper, voxel_resolution, results],
        device=device,
    )
    np.testing.assert_array_equal(results.numpy(), (1, 1, 2, 2, 4))


def test_speculative_candidates_are_opt_in(test, device):
    """Verify separated approaching shapes only emit a candidate when enabled."""
    model, state = _build_spheres(device, velocity=10.0)
    test.assertEqual(int(_collide(model, state, speculative=False).rigid_contact_count.numpy()[0]), 0)
    test.assertGreater(int(_collide(model, state, speculative=True).rigid_contact_count.numpy()[0]), 0)


def test_speculative_provenance_overwrites_reused_contact_rows(test, device):
    """Replace predictive provenance when a reused output row becomes physical."""
    model, state = _build_spheres(device, velocity=10.0)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(max_speculative_extension=0.25),
    )
    contacts = pipeline.contacts()

    pipeline.collide(state, contacts, dt=0.02)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(contacts.rigid_contact_is_predictive.numpy()[:count], 1)
    np.testing.assert_array_equal(contacts.rigid_contact_is_strict_guard.numpy()[:count], 1)

    body_q = state.body_q.numpy()
    body_q[1, :3] = (0.19, 0.0, 0.0)
    state.body_q.assign(body_q)
    state.body_qd.zero_()
    pipeline.collide(state, contacts, dt=0.02)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(contacts.rigid_contact_is_predictive.numpy()[:count], 0)
    np.testing.assert_array_equal(contacts.rigid_contact_is_strict_guard.numpy()[:count], 1)


def test_speculative_writer_separates_predictive_and_strict_guard_provenance(test, device):
    """Keep physical guard winners distinct from predictive and ordinary physical rows."""
    capacity = 5
    contacts = newton.Contacts(capacity, 0, device=device)
    writer_data = FullContactWriterData()
    writer_data.contact_max = capacity
    writer_data.body_q = wp.empty(0, dtype=wp.transform, device=device)
    writer_data.shape_body = wp.full(2, -1, dtype=wp.int32, device=device)
    writer_data.shape_gap = wp.zeros(2, dtype=wp.float32, device=device)
    writer_data.contact_count = contacts.rigid_contact_count
    writer_data.out_shape0 = contacts.rigid_contact_shape0
    writer_data.out_shape1 = contacts.rigid_contact_shape1
    writer_data.out_point0 = contacts.rigid_contact_point0
    writer_data.out_point1 = contacts.rigid_contact_point1
    writer_data.out_offset0 = contacts.rigid_contact_offset0
    writer_data.out_offset1 = contacts.rigid_contact_offset1
    writer_data.out_normal = contacts.rigid_contact_normal
    writer_data.out_normal_owner = contacts.rigid_contact_normal_owner
    writer_data.out_is_predictive = contacts.rigid_contact_is_predictive
    writer_data.out_is_strict_guard = contacts.rigid_contact_is_strict_guard
    writer_data.out_margin0 = contacts.rigid_contact_margin0
    writer_data.out_margin1 = contacts.rigid_contact_margin1
    writer_data.out_tids = contacts.rigid_contact_tids
    writer_data.out_stiffness = wp.empty(0, dtype=wp.float32, device=device)
    writer_data.out_damping = wp.empty(0, dtype=wp.float32, device=device)
    writer_data.out_friction = wp.empty(0, dtype=wp.float32, device=device)
    writer_data.out_sort_key = wp.empty(0, dtype=wp.int64, device=device)
    writer_data.shape_transform = wp.empty(0, dtype=wp.transform, device=device)
    writer_data.shape_linear_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.shape_angular_velocity = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.shape_rotation_center_offset = wp.empty(0, dtype=wp.vec3, device=device)
    writer_data.collision_update_dt = 0.0
    writer_data.max_speculative_extension = 0.0

    wp.launch(_write_finalized_guard_provenance_cases, dim=capacity, inputs=[writer_data], device=device)

    np.testing.assert_array_equal(contacts.rigid_contact_is_predictive.numpy(), (1, 0, 0, 0, 0))
    np.testing.assert_array_equal(contacts.rigid_contact_is_strict_guard.numpy(), (1, 1, 0, 1, 0))


def test_contact_sort_preserves_contact_provenance(test, device):
    """Permute predictive and strict-guard provenance with their full contact record."""
    capacity = 3
    count = wp.array([capacity], dtype=wp.int32, device=device)
    sort_keys = wp.array([30, 10, 20], dtype=wp.int64, device=device)
    shape0 = wp.array([30, 10, 20], dtype=wp.int32, device=device)
    is_predictive = wp.array([1, 0, 1], dtype=wp.uint8, device=device)
    is_strict_guard = wp.array([1, 1, 0], dtype=wp.uint8, device=device)

    ContactSorter(capacity, device=device).sort_full(
        sort_keys,
        count,
        shape0=shape0,
        shape1=wp.zeros(capacity, dtype=wp.int32, device=device),
        point0=wp.zeros(capacity, dtype=wp.vec3, device=device),
        point1=wp.zeros(capacity, dtype=wp.vec3, device=device),
        offset0=wp.zeros(capacity, dtype=wp.vec3, device=device),
        offset1=wp.zeros(capacity, dtype=wp.vec3, device=device),
        normal=wp.zeros(capacity, dtype=wp.vec3, device=device),
        normal_owner=wp.full(capacity, -1, dtype=wp.int32, device=device),
        is_predictive=is_predictive,
        is_strict_guard=is_strict_guard,
        margin0=wp.zeros(capacity, dtype=wp.float32, device=device),
        margin1=wp.zeros(capacity, dtype=wp.float32, device=device),
        tids=wp.zeros(capacity, dtype=wp.int32, device=device),
        device=device,
    )

    np.testing.assert_array_equal(shape0.numpy(), [10, 20, 30])
    np.testing.assert_array_equal(is_predictive.numpy(), [0, 1, 1])
    np.testing.assert_array_equal(is_strict_guard.numpy(), [1, 0, 1])


def test_speculative_candidates_require_approach(test, device):
    """Verify separated stationary and diverging shapes emit no candidate."""
    for velocity in (0.0, -10.0):
        model, state = _build_spheres(device, velocity=velocity)
        contacts = _collide(model, state, speculative=True)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_strict_speculative_candidates_retain_stationary_shell(test, device):
    """Retain stationary separated primitives inside the strict geometric shell."""
    model, state = _build_spheres(device, velocity=0.0, separation=0.22)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.05,
            enforce_nonpenetration=True,
        ),
    )

    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, dt=0.02)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, 1)
    np.testing.assert_array_equal(contacts.rigid_contact_is_predictive.numpy()[:count], 1)
    np.testing.assert_array_equal(contacts.rigid_contact_is_strict_guard.numpy()[:count], 1)


def test_strict_speculative_plane_box_retains_stationary_shell(test, device):
    """Retain a stationary box support face inside the strict geometric shell."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    builder.add_shape_plane(width=0.0, length=0.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.12)))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.05,
            enforce_nonpenetration=True,
        ),
    )

    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.02)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, 4)
    np.testing.assert_array_equal(contacts.rigid_contact_is_predictive.numpy()[:count], 1)
    np.testing.assert_array_equal(contacts.rigid_contact_is_strict_guard.numpy()[:count], 1)


def test_speculative_candidates_require_dt(test, device):
    """Require a current horizon and suppress candidates that cannot reach it."""
    model, state = _build_spheres(device, velocity=10.0)
    config = newton.CollisionPipeline.SpeculativeContactConfig(
        max_speculative_extension=0.25,
    )
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", speculative_config=config)

    contacts = pipeline.contacts()
    with test.assertRaisesRegex(ValueError, "dt must be provided"):
        pipeline.collide(state, contacts)

    pipeline.collide(state, contacts, dt=0.02)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    pipeline.collide(state, contacts, dt=0.005)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_gap_uses_larger_fixed_or_velocity_distance(test, device):
    """Use the larger fixed or velocity-based gap without adding them."""
    model, state = _build_spheres(device, velocity=1.0, separation=0.33, gap=0.05)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(max_speculative_extension=0.25),
    )

    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, dt=0.05)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

    pipeline.collide(state, contacts, dt=0.15)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_reject_invalid_dt_override(test, device):
    """Reject negative and non-finite per-call speculative horizons."""
    model, state = _build_spheres(device, velocity=10.0)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
    )
    contacts = pipeline.contacts()
    for dt in (-0.01, float("nan"), float("inf"), float("-inf")):
        with test.subTest(dt=dt), test.assertRaisesRegex(ValueError, "dt must be a non-negative finite number"):
            pipeline.collide(state, contacts, dt=dt)


def test_speculative_candidates_reject_common_motion(test, device):
    """Reject separated shapes whose large common motion makes their swept unions overlap."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body_a = builder.add_body(xform=wp.transform_identity())
    builder.add_shape_sphere(body_a, radius=0.1)
    builder.body_qd[body_a] = (20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.4, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    builder.body_qd[body_b] = (20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    model = builder.finalize(device=device)
    shape_pairs = wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device)
    config = newton.CollisionPipeline.SpeculativeContactConfig(
        max_speculative_extension=0.25,
    )

    for broad_phase in ("nxn", "sap", "explicit"):
        with test.subTest(broad_phase=broad_phase):
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase=broad_phase,
                shape_pairs_filtered=shape_pairs if broad_phase == "explicit" else None,
                speculative_config=config,
            )
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts, dt=0.1)
            test.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
            test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_candidates_preserve_physical_geometry(test, device):
    """Verify candidate generation does not enlarge stored physical margins."""
    model, state = _build_spheres(device, velocity=10.0)
    contacts = _collide(model, state, speculative=True)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertAlmostEqual(float(contacts.rigid_contact_margin0.numpy()[0]), 0.1, places=6)
    test.assertAlmostEqual(float(contacts.rigid_contact_margin1.numpy()[0]), 0.1, places=6)


def test_speculative_candidates_include_angular_motion(test, device):
    """Verify an offset shape's angular motion expands and filters candidates."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(
        xform=wp.transform_identity(),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body, radius=0.1, xform=wp.transform(wp.vec3(0.0, 1.0, 0.0)))
    builder.body_qd[body] = (0.0, 0.0, 0.0, 0.0, 0.0, -10.0)
    builder.add_shape_sphere(-1, radius=0.1, xform=wp.transform(wp.vec3(0.3, 1.0, 0.0)))
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.25,
        ),
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.02)
    test.assertGreater(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_cone_reaches_infinite_plane(test, device):
    """Retain a swept GJK candidate before its current bounds reach an infinite plane."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder.add_shape_cone(body, radius=0.1, half_height=0.1)
    builder.body_qd[body] = (0.0, 0.0, -20.0, 0.0, 0.0, 0.0)
    builder.add_shape_plane(width=0.0, length=0.0)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.75,
        ),
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.03)

    test.assertGreater(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_plane_proxy_adds_gap_once(test, device):
    """Size an infinite-plane proxy from the base radius plus one pair gap."""
    shape_types = wp.array([int(GeoType.PLANE), int(GeoType.CONE)], dtype=wp.int32, device=device)
    shape_data = wp.array(
        [wp.vec4(0.0), wp.vec4(0.1, 0.1, 0.0, 0.0)],
        dtype=wp.vec4,
        device=device,
    )
    shape_transform = wp.array(
        [wp.transform_identity(), wp.transform(wp.vec3(0.0, 0.0, 0.5))],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.array([wp.vec3(0.0), wp.vec3(-0.1, -0.1, 0.4)], dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.array([wp.vec3(0.0), wp.vec3(0.1, 0.1, 0.6)], dtype=wp.vec3, device=device)
    shape_collision_aabb_lower = wp.array(
        [wp.vec3(0.0), wp.vec3(-0.1)],
        dtype=wp.vec3,
        device=device,
    )
    shape_collision_aabb_upper = wp.array(
        [wp.vec3(0.0), wp.vec3(0.1)],
        dtype=wp.vec3,
        device=device,
    )
    proxy_scale = wp.zeros(1, dtype=wp.vec3, device=device)
    valid_result = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _extract_speculative_plane_proxy_scale,
        dim=1,
        inputs=[
            shape_types,
            shape_data,
            shape_transform,
            wp.zeros(2, dtype=wp.uint64, device=device),
            wp.array([0.2, 0.3], dtype=wp.float32, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            shape_aabb_lower,
            shape_aabb_upper,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
        ],
        outputs=[proxy_scale, valid_result],
        device=device,
    )

    test.assertEqual(int(valid_result.numpy()[0]), 1)
    base_radius = np.linalg.norm([0.1, 0.1, 0.1])
    expected_half_extent = 10.0 * (base_radius + 0.2 + 0.3)
    np.testing.assert_allclose(proxy_scale.numpy()[0], expected_half_extent, rtol=1.0e-6, atol=1.0e-6)


def test_stationary_contacts_match_non_speculative_pipeline(test, device):
    """Match contacts for non-moving shapes with speculative generation on and off."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body_a = builder.add_body()
    builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.15, 0.0, 0.0)))
    builder.add_shape_box(
        body_b,
        hx=0.1,
        hy=0.1,
        hz=0.1,
    )
    model = builder.finalize(device=device)
    state = model.state()

    pipelines = (
        newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True),
        newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.25,
            ),
        ),
    )
    outputs = []
    for pipeline in pipelines:
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.03)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        outputs.append(
            (
                contacts.rigid_contact_shape0.numpy()[:count],
                contacts.rigid_contact_shape1.numpy()[:count],
                contacts.rigid_contact_point0.numpy()[:count],
                contacts.rigid_contact_point1.numpy()[:count],
                contacts.rigid_contact_normal.numpy()[:count],
                contacts.rigid_contact_margin0.numpy()[:count],
                contacts.rigid_contact_margin1.numpy()[:count],
            )
        )

    for regular, speculative in zip(*outputs, strict=True):
        np.testing.assert_allclose(speculative, regular, rtol=0.0, atol=1.0e-6)


def test_speculative_contacts_prevent_dynamic_tunneling(test, device):
    """Prevent a fast dynamic sphere from crossing an infinite plane in one XPBD step."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)))
    builder.add_shape_sphere(body, radius=0.05)
    builder.body_qd[body] = (0.0, 0.0, -20.0, 0.0, 0.0, 0.0)
    builder.add_shape_plane(width=0.0, length=0.0)
    model = builder.finalize(device=device)
    dt = 0.03

    def step(speculative):
        config = None
        if speculative:
            config = newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.75,
            )
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", speculative_config=config)
        contacts = pipeline.contacts()
        state_in = model.state()
        state_out = model.state()
        pipeline.collide(state_in, contacts, dt=dt)
        newton.solvers.SolverXPBD(model, iterations=5).step(state_in, state_out, None, contacts, dt)
        return float(state_out.body_q.numpy()[body, 2])

    test.assertLess(step(False), 0.0)
    test.assertGreaterEqual(step(True), 0.04)


def test_speculative_narrow_phase_launch(test, device):
    """Verify the public narrow-phase convenience API performs exact admission."""
    shape_transform = wp.array(
        [wp.transform_identity(), wp.transform(wp.vec3(0.3, 0.0, 0.0))],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.full(2, wp.vec3(-0.1), dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.full(2, wp.vec3(0.1), dtype=wp.vec3, device=device)
    narrow_phase = NarrowPhase(
        max_candidate_pairs=1,
        reduce_contacts=False,
        device=device,
        shape_aabb_lower=shape_aabb_lower,
        shape_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        has_meshes=False,
        contact_max=4,
        verify_buffers=False,
        speculative=True,
    )

    candidate_pair = wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device)
    candidate_pair_count = wp.array([1], dtype=wp.int32, device=device)
    shape_types = wp.array([int(GeoType.SPHERE), int(GeoType.SPHERE)], dtype=wp.int32, device=device)
    shape_data = wp.array(
        [wp.vec4(0.1, 0.1, 0.1, 0.0), wp.vec4(0.1, 0.1, 0.1, 0.0)],
        dtype=wp.vec4,
        device=device,
    )
    shape_gap = wp.array([0.2, 0.0], dtype=wp.float32, device=device)
    shape_base_gap = wp.zeros(2, dtype=wp.float32, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)

    for velocity, expected_count in ((10.0, 1), (-10.0, 0)):
        contact_count = wp.zeros(1, dtype=wp.int32, device=device)
        narrow_phase.launch(
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            shape_types=shape_types,
            shape_data=shape_data,
            shape_transform=shape_transform,
            shape_source=wp.zeros(2, dtype=wp.uint64, device=device),
            shape_sdf_index=wp.full(2, -1, dtype=wp.int32, device=device),
            shape_gap=shape_gap,
            shape_base_gap=shape_base_gap,
            shape_collision_radius=wp.full(2, 0.1, dtype=wp.float32, device=device),
            shape_flags=wp.zeros(2, dtype=wp.int32, device=device),
            shape_collision_aabb_lower=shape_aabb_lower,
            shape_collision_aabb_upper=shape_aabb_upper,
            shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
            contact_pair=wp.zeros(4, dtype=wp.vec2i, device=device),
            contact_position=wp.zeros(4, dtype=wp.vec3, device=device),
            contact_normal=wp.zeros(4, dtype=wp.vec3, device=device),
            contact_penetration=wp.zeros(4, dtype=wp.float32, device=device),
            contact_count=contact_count,
            contact_tangent=wp.empty(0, dtype=wp.vec3, device=device),
            shape_linear_velocity=wp.array([wp.vec3(velocity, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
            shape_angular_velocity=shape_angular_velocity,
            collision_update_dt=0.02,
            max_speculative_extension=0.25,
            device=device,
        )
        test.assertEqual(int(contact_count.numpy()[0]), expected_count)


def test_speculative_plane_box_retains_rotating_future_face(test, device):
    """Retain both endpoints of the box face that rotates through a plane."""
    angle = np.deg2rad(10.0)
    half_extents = np.array((0.08, 0.012, 0.01), dtype=np.float32)
    initial_gap = 0.008
    center_height = initial_gap + half_extents[0] * np.sin(angle) + half_extents[2] * np.cos(angle)
    shape_transform = wp.array(
        [
            wp.transform_identity(),
            wp.transform(
                wp.vec3(0.0, 0.0, center_height),
                wp.quat(0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)),
            ),
        ],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.array([wp.vec3(-1.0), wp.vec3(*-half_extents)], dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.array([wp.vec3(1.0), wp.vec3(*half_extents)], dtype=wp.vec3, device=device)
    narrow_phase = NarrowPhase(
        max_candidate_pairs=1,
        reduce_contacts=False,
        device=device,
        shape_aabb_lower=shape_aabb_lower,
        shape_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        has_meshes=False,
        contact_max=4,
        verify_buffers=False,
        speculative=True,
    )

    contact_count = wp.zeros(1, dtype=wp.int32, device=device)
    contact_position = wp.zeros(4, dtype=wp.vec3, device=device)
    contact_normal = wp.zeros(4, dtype=wp.vec3, device=device)
    contact_separation = wp.zeros(4, dtype=wp.float32, device=device)
    narrow_phase.launch(
        candidate_pair=wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device),
        candidate_pair_count=wp.array([1], dtype=wp.int32, device=device),
        shape_types=wp.array([int(GeoType.PLANE), int(GeoType.BOX)], dtype=wp.int32, device=device),
        shape_data=wp.array(
            [wp.vec4(0.0), wp.vec4(half_extents[0], half_extents[1], half_extents[2], 0.0)],
            dtype=wp.vec4,
            device=device,
        ),
        shape_transform=shape_transform,
        shape_source=wp.zeros(2, dtype=wp.uint64, device=device),
        shape_sdf_index=wp.full(2, -1, dtype=wp.int32, device=device),
        shape_gap=wp.array([0.02, 0.0], dtype=wp.float32, device=device),
        shape_base_gap=wp.full(2, 0.005, dtype=wp.float32, device=device),
        shape_collision_radius=wp.array([1.0, np.linalg.norm(half_extents)], dtype=wp.float32, device=device),
        shape_flags=wp.zeros(2, dtype=wp.int32, device=device),
        shape_collision_aabb_lower=shape_aabb_lower,
        shape_collision_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        contact_pair=wp.zeros(4, dtype=wp.vec2i, device=device),
        contact_position=contact_position,
        contact_normal=contact_normal,
        contact_penetration=contact_separation,
        contact_count=contact_count,
        contact_tangent=wp.empty(0, dtype=wp.vec3, device=device),
        shape_linear_velocity=wp.zeros(2, dtype=wp.vec3, device=device),
        shape_angular_velocity=wp.array([wp.vec3(0.0), wp.vec3(0.0, -120.0, 0.0)], dtype=wp.vec3, device=device),
        collision_update_dt=0.01,
        max_speculative_extension=0.02,
        device=device,
    )

    count = int(contact_count.numpy()[0])
    test.assertEqual(count, 4)
    positions = contact_position.numpy()[:count]
    normals = contact_normal.numpy()[:count]
    separations = contact_separation.numpy()[:count]
    box_points = positions + 0.5 * separations[:, None] * normals
    test.assertTrue(np.all(box_points[:, 0] < 0.0))
    np.testing.assert_array_equal(np.sort(np.sign(box_points[:, 1])), np.array([-1.0, -1.0, 1.0, 1.0]))


def test_speculative_plane_box_admits_translating_future_face_within_cap(test, device):
    """Admit a separated box face that translates through a plane within the scalar cap."""
    half_extents = np.array((0.04, 0.03, 0.02), dtype=np.float32)
    initial_clearance = 0.03
    shape_transform = wp.array(
        [
            wp.transform_identity(),
            wp.transform(wp.vec3(0.0, 0.0, initial_clearance + half_extents[2])),
        ],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.array([wp.vec3(-1.0), wp.vec3(*-half_extents)], dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.array([wp.vec3(1.0), wp.vec3(*half_extents)], dtype=wp.vec3, device=device)
    narrow_phase = NarrowPhase(
        max_candidate_pairs=1,
        reduce_contacts=False,
        device=device,
        shape_aabb_lower=shape_aabb_lower,
        shape_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        has_meshes=False,
        contact_max=4,
        verify_buffers=False,
        speculative=True,
    )

    contact_count = wp.zeros(1, dtype=wp.int32, device=device)
    contact_separation = wp.zeros(4, dtype=wp.float32, device=device)
    narrow_phase.launch(
        candidate_pair=wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device),
        candidate_pair_count=wp.array([1], dtype=wp.int32, device=device),
        shape_types=wp.array([int(GeoType.PLANE), int(GeoType.BOX)], dtype=wp.int32, device=device),
        shape_data=wp.array(
            [wp.vec4(0.0), wp.vec4(half_extents[0], half_extents[1], half_extents[2], 0.0)],
            dtype=wp.vec4,
            device=device,
        ),
        shape_transform=shape_transform,
        shape_source=wp.zeros(2, dtype=wp.uint64, device=device),
        shape_sdf_index=wp.full(2, -1, dtype=wp.int32, device=device),
        shape_gap=wp.array([0.04, 0.0], dtype=wp.float32, device=device),
        shape_base_gap=wp.full(2, 0.005, dtype=wp.float32, device=device),
        shape_collision_radius=wp.array([1.0, np.linalg.norm(half_extents)], dtype=wp.float32, device=device),
        shape_flags=wp.zeros(2, dtype=wp.int32, device=device),
        shape_collision_aabb_lower=shape_aabb_lower,
        shape_collision_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        contact_pair=wp.zeros(4, dtype=wp.vec2i, device=device),
        contact_position=wp.zeros(4, dtype=wp.vec3, device=device),
        contact_normal=wp.zeros(4, dtype=wp.vec3, device=device),
        contact_penetration=contact_separation,
        contact_count=contact_count,
        contact_tangent=wp.empty(0, dtype=wp.vec3, device=device),
        shape_linear_velocity=wp.array([wp.vec3(0.0), wp.vec3(0.0, 0.0, -4.0)], dtype=wp.vec3, device=device),
        shape_angular_velocity=wp.zeros(2, dtype=wp.vec3, device=device),
        collision_update_dt=0.01,
        max_speculative_extension=0.04,
        device=device,
    )

    test.assertEqual(int(contact_count.numpy()[0]), 4)
    np.testing.assert_allclose(contact_separation.numpy(), initial_clearance, rtol=0.0, atol=1.0e-6)


def test_speculative_narrow_phase_rotates_offset_shape_about_body_com(test, device):
    """Predict an offset shape around its body COM instead of tangent-extrapolating its origin."""
    prediction_dt = 0.1
    angle = 10.0 * prediction_dt
    moving_origin = wp.vec3(0.0, 1.0, 0.0)
    exact_origin_end = wp.vec3(np.sin(angle), np.cos(angle), 0.0)
    static_origin = exact_origin_end + wp.vec3(0.0, -0.17, 0.0)
    shape_transform = wp.array(
        [wp.transform(moving_origin, wp.quat_identity()), wp.transform(static_origin, wp.quat_identity())],
        dtype=wp.transform,
        device=device,
    )
    shape_aabb_lower = wp.full(2, wp.vec3(-0.1), dtype=wp.vec3, device=device)
    shape_aabb_upper = wp.full(2, wp.vec3(0.1), dtype=wp.vec3, device=device)
    narrow_phase = NarrowPhase(
        max_candidate_pairs=1,
        reduce_contacts=False,
        device=device,
        shape_aabb_lower=shape_aabb_lower,
        shape_aabb_upper=shape_aabb_upper,
        shape_voxel_resolution=wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        has_meshes=False,
        contact_max=4,
        verify_buffers=False,
        speculative=True,
    )

    common_args = {
        "candidate_pair": wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device),
        "candidate_pair_count": wp.array([1], dtype=wp.int32, device=device),
        "shape_types": wp.full(2, int(GeoType.SPHERE), dtype=wp.int32, device=device),
        "shape_data": wp.full(2, wp.vec4(0.1, 0.1, 0.1, 0.0), dtype=wp.vec4, device=device),
        "shape_transform": shape_transform,
        "shape_source": wp.zeros(2, dtype=wp.uint64, device=device),
        "shape_sdf_index": wp.full(2, -1, dtype=wp.int32, device=device),
        "shape_gap": wp.array([1.2, 0.0], dtype=wp.float32, device=device),
        "shape_base_gap": wp.zeros(2, dtype=wp.float32, device=device),
        "shape_collision_radius": wp.full(2, 0.1, dtype=wp.float32, device=device),
        "shape_flags": wp.zeros(2, dtype=wp.int32, device=device),
        "shape_collision_aabb_lower": shape_aabb_lower,
        "shape_collision_aabb_upper": shape_aabb_upper,
        "shape_voxel_resolution": wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
        "contact_pair": wp.zeros(4, dtype=wp.vec2i, device=device),
        "contact_position": wp.zeros(4, dtype=wp.vec3, device=device),
        "contact_normal": wp.zeros(4, dtype=wp.vec3, device=device),
        "contact_penetration": wp.zeros(4, dtype=wp.float32, device=device),
        "contact_tangent": wp.empty(0, dtype=wp.vec3, device=device),
        "shape_linear_velocity": wp.array([wp.vec3(10.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
        "shape_angular_velocity": wp.array([wp.vec3(0.0, 0.0, -10.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
        "collision_update_dt": prediction_dt,
        "max_speculative_extension": 1.2,
        "device": device,
    }

    tangent_count = wp.zeros(1, dtype=wp.int32, device=device)
    narrow_phase.launch(contact_count=tangent_count, **common_args)
    test.assertEqual(int(tangent_count.numpy()[0]), 0)

    com_centered_count = wp.zeros(1, dtype=wp.int32, device=device)
    narrow_phase.launch(
        contact_count=com_centered_count,
        shape_rotation_center_offset=wp.array([wp.vec3(0.0, -1.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
        **common_args,
    )
    test.assertEqual(int(com_centered_count.numpy()[0]), 1)


def test_speculative_narrow_phase_rejects_hydroelastic(test, device):
    """Verify indexed hydroelastic writers cannot silently bypass exact admission."""
    with test.assertRaisesRegex(NotImplementedError, "does not yet support hydroelastic"):
        NarrowPhase(
            max_candidate_pairs=1,
            device=device,
            hydroelastic_sdf=object(),
            speculative=True,
        )


def test_speculative_narrow_phase_rejects_unmarked_custom_writer(test, device):
    """Reject custom writers that do not explicitly implement speculative admission."""
    with test.assertRaisesRegex(ValueError, "contact_writer_supports_speculative=True"):
        NarrowPhase(
            max_candidate_pairs=1,
            device=device,
            contact_writer_warp_func=write_contact_simple,
            speculative=True,
        )


def test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction(test, device):
    """Reject speculative hydroelastic pairs before constructing their SDF pipeline."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body_a = builder.add_body()
    builder.add_shape_sphere(body_a, radius=0.1)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.15, 0.0, 0.0)))
    builder.add_shape_sphere(body_b, radius=0.1)
    model = builder.finalize(device=device)
    shape_flags = model.shape_flags.numpy()
    shape_flags |= int(newton.ShapeFlags.HYDROELASTIC)
    model.shape_flags.assign(shape_flags)

    with test.assertRaisesRegex(NotImplementedError, "does not yet support hydroelastic"):
        newton.CollisionPipeline(
            model,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
        )


def test_speculative_pipeline_allows_missing_explicit_pairs(test, device):
    """Allow non-explicit speculative pipelines when the model pair array is absent."""
    model, _state = _build_spheres(device, velocity=0.0)
    model.shape_contact_pairs = None
    newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
    )


def test_speculative_mesh_sdf_candidates(test, device):
    """Verify separated approaching mesh SDFs retain leading candidates."""
    projectile = newton.Mesh.create_box(0.05, compute_normals=False, compute_uvs=False)
    projectile.build_sdf(device=device, max_resolution=32)
    wall = newton.Mesh.create_box(0.02, 0.3, 0.3, compute_normals=False, compute_uvs=False)
    wall.build_sdf(device=device, max_resolution=32)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0)))
    builder.add_shape_mesh(body, mesh=projectile)
    builder.body_qd[body] = (20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    builder.add_shape_mesh(-1, mesh=wall)
    model = builder.finalize(device=device)

    contacts = _collide(model, model.state(), speculative=True)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(contacts.rigid_contact_is_predictive.numpy()[:count], 1)


def test_speculative_axial_shapes_reach_triangle_mesh(test, device):
    """Retain separated sphere and capsule contacts against a triangle mesh."""
    for shape_type in ("sphere", "capsule"):
        with test.subTest(shape_type=shape_type):
            wall = newton.Mesh.create_box(0.02, 0.3, 0.3, compute_normals=False, compute_uvs=False)
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            builder.rigid_gap = 0.0
            body = builder.add_body(xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0)))
            if shape_type == "sphere":
                builder.add_shape_sphere(body, radius=0.1)
            else:
                builder.add_shape_capsule(body, radius=0.1, half_height=0.1)
            builder.body_qd[body] = (10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            builder.add_shape_mesh(-1, mesh=wall)
            model = builder.finalize(device=device)

            contacts = _collide(model, model.state(), speculative=True)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_mesh_sdf_manifold_is_bounded(test, device):
    """Bound this mesh-SDF fixture by its expanded strict-guard budget."""
    projectile = newton.Mesh.create_sphere(
        0.2,
        num_latitudes=32,
        num_longitudes=32,
        compute_normals=False,
        compute_uvs=False,
    )
    projectile.build_sdf(device=device, max_resolution=64)
    wall = newton.Mesh.create_box(0.02, 0.5, 0.5, compute_normals=False, compute_uvs=False)
    wall.build_sdf(device=device, max_resolution=64)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    body = builder.add_body(xform=wp.transform(wp.vec3(-0.35, 0.0, 0.0)))
    builder.add_shape_mesh(body, mesh=projectile)
    builder.body_qd[body] = (10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    builder.add_shape_mesh(-1, mesh=wall)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.25,
        ),
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.03)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertLessEqual(count, PREDICTIVE_CONTACT_SLOTS + VALUES_PER_KEY)
    np.testing.assert_array_equal(contacts.rigid_contact_is_predictive.numpy()[:count], 1)
    np.testing.assert_array_equal(contacts.rigid_contact_is_strict_guard.numpy()[:count], 1)


def test_speculative_mesh_sdf_retains_rotating_leading_feature(test, device):
    """Verify an inner SDF contact cannot hide the rod end rotating toward the board."""
    rod = newton.Mesh.create_box(0.5, 0.04, 0.04, compute_normals=False, compute_uvs=False)
    rod.build_sdf(device=device, max_resolution=64)
    board = newton.Mesh.create_box(0.7, 0.3, 0.02, compute_normals=False, compute_uvs=False)
    board.build_sdf(device=device, max_resolution=64)

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.rigid_gap = 0.0
    rod_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.095),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.1),
        )
    )
    builder.add_shape_mesh(rod_body, mesh=rod)
    builder.body_qd[rod_body] = (0.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    builder.add_shape_mesh(-1, mesh=board)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.15,
        ),
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts, dt=0.03)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 1)
    rod_points = contacts.rigid_contact_point0.numpy()[:count]
    test.assertLess(float(rod_points[:, 0].min()), -0.3)
    test.assertGreater(float(rod_points[:, 0].max()), 0.3)


def test_predictive_reducer_reuses_regular_contact(test, device, deterministic):
    """Verify one candidate winning regular and predictive keys exports once."""
    reducer = GlobalContactReducer(capacity=8, device=device, deterministic=deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    contact_ids = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        _register_regular_and_predictive_contact,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertGreaterEqual(int(ids[0]), 0)
    test.assertEqual(int(ids[1]), int(ids[0]))
    test.assertEqual(int(ids[2]), int(ids[0]))
    test.assertEqual(int(reducer.contact_count.numpy()[0]), 1)
    exported_count, _ = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 1)
    provenance_count, provenance, strict_guards, _sort_keys = _export_reducer_provenance(reducer, device)
    test.assertEqual(provenance_count, 1)
    np.testing.assert_array_equal(provenance, 1)
    np.testing.assert_array_equal(strict_guards, 1)


def test_predictive_reducer_retains_inside_fixed_gap_impact_guard(test, device, deterministic):
    """Retain a translated fixed-gap candidate that stays separated and wins no ordinary spatial slot."""
    reducer = _make_predictive_reducer(16, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    contact_ids = wp.full(2, -1, dtype=wp.int32, device=device)
    wp.launch(
        _register_inside_fixed_gap_impact_guard,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertEqual(int(ids[0]), -1)
    test.assertGreater(int(ids[1]), 0)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 7)
    radial_distance = np.linalg.norm(positions[:exported_count, 1:3], axis=1)
    test.assertEqual(int(np.count_nonzero(np.isclose(radial_distance, 0.0, rtol=0.0, atol=1.0e-6))), 1)
    provenance_count, provenance, strict_guards, _sort_keys = _export_reducer_provenance(reducer, device)
    test.assertEqual(provenance_count, exported_count)
    test.assertEqual(int(np.count_nonzero(provenance)), 1)
    np.testing.assert_array_equal(provenance, strict_guards)


def test_predictive_reducer_ranks_swept_rotating_separation(test, device, deterministic):
    """Rank by swept separation when the exact end state would choose the other contact."""
    angle = 0.2
    linear_end_a = 0.0497 + angle * 0.5
    linear_end_b = 0.25 - angle * 0.5
    exact_end_a = 0.5 * 0.0497 * (1.0 + np.cos(angle)) + np.sin(angle) * 0.5
    exact_end_b = 0.5 * 0.25 * (1.0 + np.cos(angle)) - np.sin(angle) * 0.5
    test.assertLess(linear_end_a, linear_end_b)
    test.assertGreater(exact_end_a, exact_end_b)
    test.assertGreater(exact_end_b, 0.0)

    reducer = _make_predictive_reducer(16, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.array([wp.vec3(0.0, 0.0, 20.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    contact_ids = wp.full(4, -1, dtype=wp.int32, device=device)
    wp.launch(
        _rank_rotating_anchor_end_separation,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertEqual(int(ids[0]), -1)
    test.assertEqual(int(ids[2]), -1)
    test.assertEqual(_predictive_slot_contact_id(reducer, 6), int(ids[1]))
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 8)
    radial_distance = np.linalg.norm(positions[:exported_count, 1:3], axis=1)
    guard_positions = positions[:exported_count][radial_distance < 1.0]
    test.assertEqual(len(guard_positions), 2)
    test.assertTrue(np.any(np.isclose(guard_positions[:, 1], -0.5, rtol=0.0, atol=1.0e-6)))


def test_predictive_reducer_retains_future_feature_leading_candidate(test, device, deterministic):
    """Retain a horizon-leading feature that loses current support and swept-separation ranking."""
    reducer = _make_predictive_reducer(16, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.array([wp.vec3(0.0, 0.0, 20.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    contact_ids = wp.full(3, -1, dtype=wp.int32, device=device)
    wp.launch(
        _register_future_feature_leading_candidate,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertGreater(int(ids[0]), 0)
    test.assertEqual(int(ids[1]), -1)
    test.assertGreater(int(ids[2]), 0)
    exported_count, _positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 8)
    provenance_count, provenance, strict_guards, _sort_keys = _export_reducer_provenance(reducer, device)
    test.assertEqual(provenance_count, exported_count)
    test.assertEqual(int(np.count_nonzero(provenance)), 2)
    np.testing.assert_array_equal(provenance, strict_guards)


def test_outer_reducer_retains_minimum_normal_clearance(test, device, deterministic):
    """Retain the nearest outer-tier contact when tangent-plane projections coincide."""
    reducer = GlobalContactReducer(capacity=8, device=device, deterministic=deterministic)
    contact_ids = wp.full(7, -1, dtype=wp.int32, device=device)
    wp.launch(
        _register_outer_ring_around_closest_contact,
        dim=1,
        inputs=[reducer.get_data_struct(), contact_ids],
        device=device,
    )

    exported_count, positions = _export_reducer_contacts(reducer, device)
    clearances = positions[:exported_count, 2]
    test.assertTrue(np.any(np.isclose(clearances, 0.001865, rtol=0.0, atol=1.0e-7)))


def test_speculative_buffered_reducer_uses_one_based_ids(test, device, deterministic):
    """Verify buffered speculative reduction processes every real contact and skips reserved ID zero."""
    reducer = GlobalContactReducer(capacity=8, device=device, deterministic=deterministic)
    wp.launch(_buffer_one_contact, dim=1, inputs=[reducer.get_data_struct()], device=device)

    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    wp.launch(
        reduce_buffered_contacts_speculative_kernel,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            wp.full(2, int(GeoType.BOX), dtype=wp.int32, device=device),
            wp.zeros(2, dtype=wp.vec4, device=device),
            wp.zeros(2, dtype=wp.float32, device=device),
            shape_transform,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3(-1.0), dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3(1.0), dtype=wp.vec3, device=device),
            wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
            0.1,
            0.1,
            1,
        ],
        device=device,
    )

    test.assertEqual(int(reducer.contact_count.numpy()[0]), 1)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 1)
    for actual, expected in zip(positions[0], (0.25, -0.5, 0.75), strict=True):
        test.assertAlmostEqual(float(actual), expected, places=6)


def test_speculative_buffered_axial_contacts_use_predictive_manifold(test, device):
    """Route separated axial contacts only through the bounded predictive manifold."""
    for shape_type in (GeoType.SPHERE, GeoType.CAPSULE):
        with test.subTest(shape_type=shape_type):
            reducer = GlobalContactReducer(capacity=8, device=device)
            wp.launch(_buffer_separated_axial_contact, dim=1, inputs=[reducer.get_data_struct()], device=device)

            wp.launch(
                reduce_buffered_contacts_speculative_kernel,
                dim=1,
                inputs=[
                    reducer.get_data_struct(),
                    wp.array([int(shape_type), int(GeoType.MESH)], dtype=wp.int32, device=device),
                    wp.array([wp.vec4(0.1, 0.1, 0.1, 0.0), wp.vec4(0.0)], dtype=wp.vec4, device=device),
                    wp.zeros(2, dtype=wp.float32, device=device),
                    wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
                    wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device),
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3(-1.0), dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3(1.0), dtype=wp.vec3, device=device),
                    wp.full(2, wp.vec3i(1), dtype=wp.vec3i, device=device),
                    0.1,
                    0.1,
                    1,
                ],
                device=device,
            )

            bins = (reducer.hashtable.keys.numpy() >> np.uint64(55)) & np.uint64(0xFF)
            test.assertIn(PREDICTIVE_BIN_ID, bins)
            test.assertFalse(np.any(bins < PREDICTIVE_BIN_ID))


def test_predictive_reducer_reclaims_replaced_reservation(test, device, deterministic):
    """Reclaim capacity when both validated predictive claims are replaced."""
    capacity = 8
    reducer = _make_predictive_reducer(capacity, device, deterministic)
    allocated_ids = wp.full(capacity, -1, dtype=wp.int32, device=device)
    wp.launch(
        _replace_validated_predictive_claims,
        dim=1,
        inputs=[reducer.get_data_struct(), allocated_ids],
        device=device,
    )

    test.assertEqual(sorted(int(contact_id) for contact_id in allocated_ids.numpy()), list(range(1, capacity + 1)))
    test.assertEqual(int(reducer.contact_count.numpy()[0]), capacity)


def test_reduction_rollback_does_not_resurrect_provisional_claim(test, device, deterministic):
    """Clear an unowned provisional predecessor while restoring materialized predecessors."""
    reducer = _make_predictive_reducer(8, device, deterministic)
    rolled_back_values = wp.zeros(2, dtype=wp.uint64, device=device)
    wp.launch(
        _rollback_replaced_provisional_claims,
        dim=1,
        inputs=[reducer.get_data_struct(), rolled_back_values],
        device=device,
    )

    values = rolled_back_values.numpy()
    contact_id_mask = (1 << (20 if deterministic else 32)) - 1
    test.assertEqual(int(values[0]), 0)
    test.assertEqual(int(values[1] & np.uint64(contact_id_mask)), 3)


def test_two_depth_reducer_reclaims_replaced_materialization(test, device, deterministic):
    """Reuse an allocation whose only spatial claim was replaced before publication."""
    capacity = 8
    reducer = _make_predictive_reducer(capacity, device, deterministic)
    completed_id = wp.zeros(1, dtype=wp.int32, device=device)
    allocated_ids = wp.full(capacity, -1, dtype=wp.int32, device=device)
    wp.launch(
        _replace_validated_two_depth_claim,
        dim=1,
        inputs=[reducer.get_data_struct(), completed_id, allocated_ids],
        device=device,
    )

    test.assertEqual(int(completed_id.numpy()[0]), -1)
    test.assertEqual(sorted(int(contact_id) for contact_id in allocated_ids.numpy()), list(range(1, capacity + 1)))
    test.assertEqual(int(reducer.contact_count.numpy()[0]), capacity)


def test_crossing_only_reducer_counts_full_hashtable_failure(test, device, deterministic):
    """Report a failed voxel insertion when a crossing cannot use normal slots."""
    reducer = _make_predictive_reducer(8, device, deterministic)
    reducer.hashtable.keys.fill_(0)
    contact_id = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _register_crossing_with_full_hashtable,
        dim=1,
        inputs=[reducer.get_data_struct(), contact_id],
        device=device,
    )

    test.assertEqual(int(contact_id.numpy()[0]), -1)
    test.assertEqual(int(reducer.ht_insert_failures.numpy()[0]), 1)
    test.assertEqual(int(reducer.contact_count.numpy()[0]), 0)


def test_predictive_reducer_retains_rotating_leading_contact(test, device, deterministic):
    """Reuse full-shell directional support as the predictive angular guard."""
    reducer = _make_predictive_reducer(8, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.array([wp.vec3(0.0, 0.0, -1.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    contact_ids = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        _register_inner_and_rotating_leading_contact,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertGreater(int(ids[0]), 0)
    test.assertGreater(int(ids[1]), 0)
    test.assertEqual(int(ids[2]), int(ids[1]))
    test.assertNotEqual(int(ids[2]), int(ids[0]))
    test.assertEqual(int(reducer.contact_fingerprints.numpy()[ids[1]]), 23)
    test.assertEqual(int(reducer.contact_count.numpy()[0]), 2)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 2)
    test.assertAlmostEqual(float(max(positions[:exported_count, 1])), 1.0, places=6)
    provenance_count, provenance, strict_guards, sort_keys = _export_reducer_provenance(reducer, device)
    target_rows = (sort_keys & np.int64(0x7FFFFF)) == 23
    test.assertEqual(provenance_count, 2)
    test.assertEqual(int(np.count_nonzero(target_rows)), 1)
    test.assertEqual(int(provenance[target_rows][0]), 1)
    test.assertEqual(int(strict_guards[target_rows][0]), 1)


def test_predictive_reducer_retains_wrench_supports_and_separation_guard(test, device, deterministic):
    """Retain normal and lever-arm extrema plus the least-swept-separation guard."""
    reducer = _make_predictive_reducer(16, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    contact_ids = wp.full(14, -1, dtype=wp.int32, device=device)
    wp.launch(
        _register_predictive_wrench_candidates,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
        ],
        device=device,
    )

    test.assertEqual(int(reducer.contact_count.numpy()[0]), 13)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    _assert_predictive_wrench_manifold(test, exported_count, positions)


def test_predictive_reducer_retains_moment_extreme_despite_fingerprint_collision(test, device, deterministic):
    """Retain a lever-arm moment extreme when nearer candidates share its fingerprint shard."""
    reducer = _make_predictive_reducer(8, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    wp.launch(
        _register_predictive_moment_extreme_with_colliding_fingerprints,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
        ],
        device=device,
    )

    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertGreater(exported_count, 1)
    test.assertAlmostEqual(float(positions[:exported_count, 1].max()), 1.0, places=6)


def test_predictive_reducer_retains_wrench_manifold_under_contention(test, device, deterministic):
    """Retain wrench supports and the swept-separation guard under concurrent candidate orderings."""
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    for candidate_order in (0, 1, 2):
        with test.subTest(candidate_order=candidate_order):
            reducer = _make_predictive_reducer(16, device, deterministic)
            wp.launch(
                _register_predictive_wrench_candidates_contended,
                dim=256,
                inputs=[
                    reducer.get_data_struct(),
                    shape_transform,
                    shape_linear_velocity,
                    shape_angular_velocity,
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    candidate_order,
                ],
                block_dim=32,
                device=device,
            )

            test.assertLessEqual(int(reducer.contact_count.numpy()[0]), reducer.capacity)
            exported_count, positions = _export_reducer_contacts(reducer, device)
            _assert_predictive_wrench_manifold(test, exported_count, positions)


def test_predictive_reducer_retains_interior_wrench_extreme(test, device, deterministic):
    """Retain an interior contact whose lever arm is the only extreme feature."""
    reducer = _make_predictive_reducer(16, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    contact_ids = wp.full(8, -1, dtype=wp.int32, device=device)
    target_fingerprint = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _register_predictive_wrench_adversary,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
            target_fingerprint,
        ],
        device=device,
    )

    target_id = int(contact_ids.numpy()[7])
    test.assertGreater(target_id, 0)
    test.assertEqual(
        int(reducer.contact_fingerprints.numpy()[target_id]),
        int(target_fingerprint.numpy()[0]),
    )
    np.testing.assert_allclose(
        reducer.position_depth.numpy()[target_id],
        (0.5, 0.5, 0.0, 0.05),
        rtol=0.0,
        atol=1.0e-7,
    )
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertTrue(
        np.any(np.all(np.isclose(positions[:exported_count], (0.5, 0.5, 0.0), rtol=0.0, atol=1.0e-6), axis=1))
    )


def test_predictive_reducer_retains_crossing_outer_voxel(test, device, deterministic):
    """Retain a crossing endpoint that loses every pair-level support score."""
    reducer = _make_predictive_reducer(32, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    normal = np.asarray(FACE_NORMALS[2], dtype=np.float32)
    shape_linear_velocity = wp.array(
        [wp.vec3(*(0.06 * normal)), wp.vec3(0.0)],
        dtype=wp.vec3,
        device=device,
    )
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    contact_ids = wp.full(9, -1, dtype=wp.int32, device=device)
    target_fingerprint = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _register_predictive_crossing_voxel_adversary,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
            target_fingerprint,
        ],
        device=device,
    )

    target_id = int(contact_ids.numpy()[7])
    test.assertGreater(target_id, 0)
    stored_fingerprint = int(reducer.contact_fingerprints.numpy()[target_id])
    test.assertEqual(stored_fingerprint, int(target_fingerprint.numpy()[0]) | (1 << 28))

    contact_id_mask = np.uint64((1 << (20 if reducer.deterministic else 32)) - 1)
    retained_ids = reducer.ht_values.numpy() & contact_id_mask
    test.assertTrue(np.any(retained_ids == target_id))
    provenance_count, provenance, strict_guards, sort_keys = _export_reducer_provenance(reducer, device)
    test.assertGreater(provenance_count, 0)
    target_rows = (sort_keys & np.int64(0x7FFFFF)) == 1
    test.assertEqual(int(np.count_nonzero(target_rows)), 1)
    test.assertEqual(int(provenance[target_rows][0]), 1)
    test.assertEqual(int(strict_guards[target_rows][0]), 1)


def test_authored_shell_reducer_retains_bounded_guards(test, device, deterministic):
    """Retain ordinary owned-endpoint coverage or one pair-wide interior shell guard."""
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    for is_owned_endpoint in (False, True):
        with test.subTest(is_owned_endpoint=is_owned_endpoint):
            reducer = _make_predictive_reducer(32, device, deterministic)
            contact_ids = wp.full(2, -1, dtype=wp.int32, device=device)
            target_fingerprint = wp.zeros(1, dtype=wp.int32, device=device)
            wp.launch(
                _register_authored_shell_voxel_adversary,
                dim=1,
                inputs=[
                    reducer.get_data_struct(),
                    shape_transform,
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    is_owned_endpoint,
                    contact_ids,
                    target_fingerprint,
                ],
                device=device,
            )

            regular_id, predictive_id = (int(contact_id) for contact_id in contact_ids.numpy())
            if is_owned_endpoint:
                test.assertGreater(regular_id, 0)
                test.assertEqual(predictive_id, -1)
                target_id = regular_id
            else:
                test.assertEqual(regular_id, -1)
                test.assertGreater(predictive_id, 0)
                target_id = predictive_id

                keys = reducer.hashtable.keys.numpy()
                bins = (keys >> np.uint64(55)) & np.uint64(0xFF)
                predictive_entry = int(np.flatnonzero(bins == PREDICTIVE_BIN_ID)[0])
                moment_entry = int(np.flatnonzero(bins == PREDICTIVE_MOMENT_BIN_ID)[0])
                values = reducer.ht_values.numpy().reshape(VALUES_PER_KEY, reducer.hashtable.capacity)
                contact_id_mask = np.uint64((1 << (20 if reducer.deterministic else 32)) - 1)
                existing_lane_ids = (
                    np.concatenate((values[:, predictive_entry], values[: VALUES_PER_KEY - 1, moment_entry]))
                    & contact_id_mask
                )
                test.assertFalse(np.any(existing_lane_ids == target_id))
                test.assertEqual(
                    int(values[PREDICTIVE_CURRENT_SEPARATION_LOCAL_SLOT, moment_entry] & contact_id_mask), target_id
                )

            provenance_count, provenance, strict_guards, sort_keys = _export_reducer_provenance(reducer, device)
            test.assertGreater(provenance_count, 0)
            target_rows = (sort_keys & np.int64(0x7FFFFF)) == 1
            if is_owned_endpoint:
                stored_fingerprint = int(reducer.contact_fingerprints.numpy()[target_id])
                test.assertEqual(stored_fingerprint, int(target_fingerprint.numpy()[0]))
            else:
                stored_fingerprint = int(reducer.contact_fingerprints.numpy()[target_id])
                test.assertEqual(stored_fingerprint, int(target_fingerprint.numpy()[0]))
            test.assertEqual(int(np.count_nonzero(target_rows)), 1)
            test.assertEqual(int(provenance[target_rows][0]), int(not is_owned_endpoint))
            test.assertEqual(int(strict_guards[target_rows][0]), int(not is_owned_endpoint))


def test_reduced_canonical_endpoint_guard_survives_source_replacement(test, device, deterministic):
    """Classify a final canonical voxel winner as paired after it replaces a dedicated row."""
    reducer = _make_predictive_reducer(64, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    contact_ids = wp.full(4, -1, dtype=wp.int32, device=device)
    wp.launch(
        _replace_dedicated_endpoint_with_ordinary_equivalent,
        dim=1,
        inputs=[
            reducer.get_data_struct(True),
            shape_transform,
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
        ],
        device=device,
    )

    dedicated_id, dedicated_predictive_id, ordinary_id, ordinary_predictive_id = (
        int(contact_id) for contact_id in contact_ids.numpy()
    )
    test.assertGreater(dedicated_id, 0)
    test.assertEqual(dedicated_predictive_id, -1)
    test.assertGreater(ordinary_id, 0)
    test.assertEqual(ordinary_predictive_id, -1)
    ordinary_stored_fingerprint = np.uint32(reducer.contact_fingerprints.numpy()[ordinary_id])
    test.assertNotEqual(int(ordinary_stored_fingerprint & np.uint32(1 << 31)), 0)
    test.assertEqual(int(ordinary_stored_fingerprint & np.uint32(1 << 28)), 0)

    provenance_count, provenance, strict_guards, sort_keys = _export_reducer_provenance(reducer, device)
    test.assertGreater(provenance_count, 0)
    target_rows = (sort_keys & np.int64(0x7FFFFF)) == 1
    test.assertEqual(int(np.count_nonzero(target_rows)), 1)
    test.assertEqual(int(provenance[target_rows][0]), 1)
    test.assertEqual(int(strict_guards[target_rows][0]), 1)


def test_predictive_reducer_preserves_inner_voxel_priority(test, device, deterministic):
    """Retain swept voxel coverage without letting an outer candidate replace an inner winner."""
    reducer = _make_predictive_reducer(8, device, deterministic)
    contact_ids = wp.full(3, -1, dtype=wp.int32, device=device)
    wp.launch(
        _register_inner_and_crossing_voxel_competitors,
        dim=1,
        inputs=[reducer.get_data_struct(), contact_ids],
        device=device,
    )

    inner_id, blocked_outer_id, crossing_id = (int(contact_id) for contact_id in contact_ids.numpy())
    test.assertGreater(inner_id, 0)
    test.assertEqual(blocked_outer_id, -1)
    test.assertGreater(crossing_id, 0)
    test.assertEqual(int(reducer.contact_fingerprints.numpy()[inner_id]), 11)
    test.assertEqual(int(reducer.contact_fingerprints.numpy()[crossing_id]), 13 | (1 << 28))


def test_predictive_reducer_preserves_winners_with_small_buffer(test, device, deterministic):
    """Preserve wrench supports and the swept-separation guard in a small contact buffer."""
    reducer = _make_predictive_reducer(16, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    wp.launch(
        _register_predictive_wrench_candidates_sequential,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
        ],
        device=device,
    )

    allocated_count = int(reducer.contact_count.numpy()[0])
    test.assertLessEqual(allocated_count, reducer.capacity)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertLessEqual(exported_count, allocated_count)
    _assert_predictive_wrench_manifold(test, exported_count, positions)


def test_predictive_reducer_restores_winner_when_buffer_is_full(test, device, deterministic):
    """Preserve displaced winners when a stronger candidate cannot allocate storage."""
    reducer = _make_predictive_reducer(1, device, deterministic)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    shape_linear_velocity = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0)], dtype=wp.vec3, device=device)
    shape_angular_velocity = wp.zeros(2, dtype=wp.vec3, device=device)
    contact_ids = wp.full(2, -1, dtype=wp.int32, device=device)
    wp.launch(
        _replace_full_buffer_predictive_winner,
        dim=1,
        inputs=[
            reducer.get_data_struct(),
            shape_transform,
            shape_linear_velocity,
            shape_angular_velocity,
            wp.zeros(2, dtype=wp.vec3, device=device),
            contact_ids,
        ],
        device=device,
    )

    ids = contact_ids.numpy()
    test.assertEqual(int(ids[0]), 1)
    test.assertEqual(int(ids[1]), -1)
    exported_count, positions = _export_reducer_contacts(reducer, device)
    test.assertEqual(exported_count, 1)
    test.assertAlmostEqual(float(positions[0, 2]), 0.08, places=6)


def test_predictive_reducer_reclamation_storage_is_opt_in(test, device):
    """Avoid allocating predictive reservation storage in the default reducer."""
    reducer = GlobalContactReducer(capacity=32, device=device)
    test.assertEqual(reducer.reclaimed_contact_bits.shape[0], 0)
    test.assertEqual(reducer.reclaimed_contact_cursor.shape[0], 0)


def test_predictive_reducer_reclaims_ids_without_duplicates(test, device):
    """Allocate each concurrently reclaimed contact ID exactly once."""
    capacity = 1024
    reducer = _make_predictive_reducer(capacity, device, deterministic=False)
    reducer.contact_count.fill_(capacity)
    allocated_ids = wp.full(capacity, -1, dtype=wp.int32, device=device)

    wp.launch(
        _reclaim_and_allocate_contact_ids,
        dim=capacity,
        inputs=[reducer.get_data_struct(), allocated_ids],
        device=device,
    )

    ids = np.sort(allocated_ids.numpy())
    np.testing.assert_array_equal(ids, np.arange(1, capacity + 1, dtype=np.int32))
    test.assertEqual(int(reducer.contact_count.numpy()[0]), capacity)
    test.assertEqual(int(np.count_nonzero(reducer.reclaimed_contact_bits.numpy())), 0)


class TestSpeculativeContacts(unittest.TestCase):
    """Test inexpensive speculative-candidate behavior."""


class TestSpeculativeMeshContacts(unittest.TestCase):
    """Test speculative mesh/SDF candidate generation on CUDA."""


for _name, _test in (
    ("test_mesh_sdf_candidate_endpoint_tagging", test_mesh_sdf_candidate_endpoint_tagging),
    (
        "test_mesh_sdf_voxel_owner_is_fine_and_pair_order_invariant",
        test_mesh_sdf_voxel_owner_is_fine_and_pair_order_invariant,
    ),
    ("test_speculative_candidates_are_opt_in", test_speculative_candidates_are_opt_in),
    (
        "test_speculative_provenance_overwrites_reused_contact_rows",
        test_speculative_provenance_overwrites_reused_contact_rows,
    ),
    (
        "test_speculative_writer_separates_predictive_and_strict_guard_provenance",
        test_speculative_writer_separates_predictive_and_strict_guard_provenance,
    ),
    ("test_contact_sort_preserves_contact_provenance", test_contact_sort_preserves_contact_provenance),
    ("test_speculative_candidates_require_approach", test_speculative_candidates_require_approach),
    (
        "test_strict_speculative_candidates_retain_stationary_shell",
        test_strict_speculative_candidates_retain_stationary_shell,
    ),
    (
        "test_strict_speculative_plane_box_retains_stationary_shell",
        test_strict_speculative_plane_box_retains_stationary_shell,
    ),
    ("test_speculative_candidates_require_dt", test_speculative_candidates_require_dt),
    (
        "test_speculative_gap_uses_larger_fixed_or_velocity_distance",
        test_speculative_gap_uses_larger_fixed_or_velocity_distance,
    ),
    ("test_speculative_candidates_reject_invalid_dt_override", test_speculative_candidates_reject_invalid_dt_override),
    ("test_speculative_candidates_reject_common_motion", test_speculative_candidates_reject_common_motion),
    ("test_speculative_candidates_preserve_physical_geometry", test_speculative_candidates_preserve_physical_geometry),
    ("test_speculative_candidates_include_angular_motion", test_speculative_candidates_include_angular_motion),
    ("test_speculative_cone_reaches_infinite_plane", test_speculative_cone_reaches_infinite_plane),
    ("test_speculative_plane_proxy_adds_gap_once", test_speculative_plane_proxy_adds_gap_once),
    (
        "test_stationary_contacts_match_non_speculative_pipeline",
        test_stationary_contacts_match_non_speculative_pipeline,
    ),
    ("test_speculative_contacts_prevent_dynamic_tunneling", test_speculative_contacts_prevent_dynamic_tunneling),
    ("test_speculative_narrow_phase_launch", test_speculative_narrow_phase_launch),
    (
        "test_speculative_plane_box_retains_rotating_future_face",
        test_speculative_plane_box_retains_rotating_future_face,
    ),
    (
        "test_speculative_plane_box_admits_translating_future_face_within_cap",
        test_speculative_plane_box_admits_translating_future_face_within_cap,
    ),
    (
        "test_speculative_narrow_phase_rotates_offset_shape_about_body_com",
        test_speculative_narrow_phase_rotates_offset_shape_about_body_com,
    ),
    (
        "test_speculative_narrow_phase_rejects_hydroelastic",
        test_speculative_narrow_phase_rejects_hydroelastic,
    ),
    (
        "test_speculative_narrow_phase_rejects_unmarked_custom_writer",
        test_speculative_narrow_phase_rejects_unmarked_custom_writer,
    ),
    (
        "test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction",
        test_speculative_pipeline_rejects_hydroelastic_before_sdf_construction,
    ),
    (
        "test_speculative_pipeline_allows_missing_explicit_pairs",
        test_speculative_pipeline_allows_missing_explicit_pairs,
    ),
    (
        "test_predictive_reducer_reclamation_storage_is_opt_in",
        test_predictive_reducer_reclamation_storage_is_opt_in,
    ),
    (
        "test_speculative_buffered_axial_contacts_use_predictive_manifold",
        test_speculative_buffered_axial_contacts_use_predictive_manifold,
    ),
    (
        "test_predictive_reducer_reclaims_ids_without_duplicates",
        test_predictive_reducer_reclaims_ids_without_duplicates,
    ),
):
    add_function_test(TestSpeculativeContacts, _name, _test, devices=get_test_devices())

for _deterministic in (False, True):
    _suffix = "deterministic" if _deterministic else "fast"
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_reuses_regular_contact_{_suffix}",
        test_predictive_reducer_reuses_regular_contact,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_inside_fixed_gap_impact_guard_{_suffix}",
        test_predictive_reducer_retains_inside_fixed_gap_impact_guard,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_ranks_swept_rotating_separation_{_suffix}",
        test_predictive_reducer_ranks_swept_rotating_separation,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_future_feature_leading_candidate_{_suffix}",
        test_predictive_reducer_retains_future_feature_leading_candidate,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_outer_reducer_retains_minimum_normal_clearance_{_suffix}",
        test_outer_reducer_retains_minimum_normal_clearance,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_speculative_buffered_reducer_uses_one_based_ids_{_suffix}",
        test_speculative_buffered_reducer_uses_one_based_ids,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_reclaims_replaced_reservation_{_suffix}",
        test_predictive_reducer_reclaims_replaced_reservation,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_reduction_rollback_does_not_resurrect_provisional_claim_{_suffix}",
        test_reduction_rollback_does_not_resurrect_provisional_claim,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_two_depth_reducer_reclaims_replaced_materialization_{_suffix}",
        test_two_depth_reducer_reclaims_replaced_materialization,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_crossing_only_reducer_counts_full_hashtable_failure_{_suffix}",
        test_crossing_only_reducer_counts_full_hashtable_failure,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_rotating_leading_contact_{_suffix}",
        test_predictive_reducer_retains_rotating_leading_contact,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_wrench_supports_and_separation_guard_{_suffix}",
        test_predictive_reducer_retains_wrench_supports_and_separation_guard,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_moment_extreme_despite_fingerprint_collision_{_suffix}",
        test_predictive_reducer_retains_moment_extreme_despite_fingerprint_collision,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_wrench_manifold_under_contention_{_suffix}",
        test_predictive_reducer_retains_wrench_manifold_under_contention,
        devices=get_cuda_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_interior_wrench_extreme_{_suffix}",
        test_predictive_reducer_retains_interior_wrench_extreme,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_retains_crossing_outer_voxel_{_suffix}",
        test_predictive_reducer_retains_crossing_outer_voxel,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_authored_shell_reducer_retains_bounded_guards_{_suffix}",
        test_authored_shell_reducer_retains_bounded_guards,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_reduced_canonical_endpoint_guard_survives_source_replacement_{_suffix}",
        test_reduced_canonical_endpoint_guard_survives_source_replacement,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_preserves_inner_voxel_priority_{_suffix}",
        test_predictive_reducer_preserves_inner_voxel_priority,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_preserves_winners_with_small_buffer_{_suffix}",
        test_predictive_reducer_preserves_winners_with_small_buffer,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )
    add_function_test(
        TestSpeculativeContacts,
        f"test_predictive_reducer_restores_winner_when_buffer_is_full_{_suffix}",
        test_predictive_reducer_restores_winner_when_buffer_is_full,
        devices=get_test_devices(),
        deterministic=_deterministic,
    )

add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_axial_shapes_reach_triangle_mesh",
    test_speculative_axial_shapes_reach_triangle_mesh,
    devices=get_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_candidates",
    test_speculative_mesh_sdf_candidates,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_manifold_is_bounded",
    test_speculative_mesh_sdf_manifold_is_bounded,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSpeculativeMeshContacts,
    "test_speculative_mesh_sdf_retains_rotating_leading_feature",
    test_speculative_mesh_sdf_retains_rotating_leading_feature,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

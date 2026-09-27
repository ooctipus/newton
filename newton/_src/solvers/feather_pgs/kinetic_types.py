# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared private Kuka kinetic descriptors for one bounded scratch experiment.

No canonical primary spatial/factor arrays are aliases of these fields. Current,
geometric and held generations are distinct; only held arrays intentionally
persist through refresh/reuse. Public external wrench convention is force first,
torque second, exactly matching the literal original source indexing.
"""

import warp as wp


@wp.struct
class KineticPlan:
    body_ids: wp.array2d[int]
    body_parent: wp.array2d[int]
    body_dof: wp.array2d[int]
    body_lane: wp.array[int]
    joint_ids: wp.array2d[int]
    dof_ids: wp.array2d[int]
    root_slots: wp.array2d[int]
    primary_group: wp.array[int]
    secondary_group: wp.array[int]
    body_local_dof: wp.array2d[int]
    dof_joint: wp.array2d[int]
    q_index: wp.array2d[int]


@wp.struct
class CurrentKineticCache:
    bias: wp.array2d[float]
    axes: wp.array2d[wp.spatial_vector]
    com_offset: wp.array2d[wp.vec3]
    origin: wp.array2d[wp.vec3]
    free_bias: wp.array[wp.spatial_vector]
    free_axes: wp.array2d[wp.spatial_vector]
    generation: wp.array[wp.int64]
    valid: wp.array[int]


@wp.struct
class GeometricCache:
    geometric: wp.array2d[float]
    generation: wp.array[wp.int64]
    valid: wp.array[int]


@wp.struct
class HeldKineticOperator:
    T: wp.array2d[float]
    augmented: wp.array2d[float]
    generation: wp.array[wp.int64]
    valid: wp.array[int]
    lower6: wp.array3d[float]
    inverse6: wp.array3d[float]


@wp.struct
class CurrentForceInput:
    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    predictor_qd: wp.array[float]
    joint_f: wp.array[float]
    body_f: wp.array[wp.spatial_vector]
    body_flags: wp.array[int]
    joint_spring_stiffness: wp.array[float]
    joint_spring_ref: wp.array[float]
    joint_damping: wp.array[float]
    joint_target_ke: wp.array[float]
    joint_target_kd: wp.array[float]
    joint_target_pos: wp.array[float]
    joint_target_vel: wp.array[float]
    joint_effort_limit: wp.array[float]
    drive_row_by_dof: wp.array[int]
    drive_q_index_by_dof: wp.array[int]
    kinematic_dof: wp.array[int]
    kinematic_joint: wp.array[int]
    joint_qd_start: wp.array[int]
    joint_q_start: wp.array[int]
    free_root_joints: wp.array[int]
    prescribed_body_v_s: wp.array[wp.spatial_vector]
    state_generation: wp.array[wp.int64]
    expected_held_generation: wp.array[wp.int64]
    dt: float


@wp.struct
class KineticPredictorOutput:
    v_hat: wp.array[float]
    joint_qdd: wp.array[float]
    endpoint_twists: wp.array[wp.spatial_vector]
    status: wp.array[int]
    external_nonzero: wp.array[int]


@wp.struct
class KineticSchedule:
    generation: wp.array[wp.int64]
    geometry_requested: wp.array[int]
    status: wp.array[int]


@wp.struct
class RefreshInput:
    """Actual held update request; current R/K are added only on request."""

    requested: wp.array[int]
    state_generation: wp.array[wp.int64]
    held_generation: wp.array[wp.int64]
    R: wp.array[float]
    joint_target_ke: wp.array[float]
    joint_target_kd: wp.array[float]
    drive_row_by_dof: wp.array[int]
    dt: float
    status: wp.array[int]


def allocate_current(worlds, device):
    """Allocate distinct current storage; caller owns generation publication."""
    if worlds <= 0:
        raise ValueError("Kinetic cache needs positive world capacity")
    value = CurrentKineticCache()
    value.bias = wp.zeros((worlds, 23), dtype=float, device=device)
    value.axes = wp.zeros((worlds, 23), dtype=wp.spatial_vector, device=device)
    value.com_offset = wp.zeros((worlds, 32), dtype=wp.vec3, device=device)
    value.origin = wp.zeros((worlds, 3), dtype=wp.vec3, device=device)
    value.free_bias = wp.zeros(worlds, dtype=wp.spatial_vector, device=device)
    value.free_axes = wp.zeros((worlds, 6), dtype=wp.spatial_vector, device=device)
    value.generation = wp.zeros(worlds, dtype=wp.int64, device=device)
    value.valid = wp.zeros(worlds, dtype=int, device=device)
    return value


def allocate_geometric(worlds, device):
    """Allocate geometry storage independently of current and held epochs."""
    if worlds <= 0:
        raise ValueError("Geometric cache needs positive world capacity")
    value = GeometricCache()
    value.geometric = wp.zeros((worlds, 180), dtype=float, device=device)
    value.generation = wp.zeros(worlds, dtype=wp.int64, device=device)
    value.valid = wp.zeros(worlds, dtype=int, device=device)
    return value


def allocate_held(worlds, lower6, inverse6, device):
    """Allocate robot held state while retaining original grouped free factors."""
    if worlds <= 0:
        raise ValueError("Held cache needs positive world capacity")
    value = HeldKineticOperator()
    value.T = wp.zeros((worlds, 180), dtype=float, device=device)
    value.augmented = wp.zeros((worlds, 180), dtype=float, device=device)
    value.generation = wp.zeros(worlds, dtype=wp.int64, device=device)
    value.valid = wp.zeros(worlds, dtype=int, device=device)
    value.lower6 = lower6
    value.inverse6 = inverse6
    return value

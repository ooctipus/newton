# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current dense-row descriptors; existing response backing is reused once."""

import warp as wp


@wp.struct
class RawRowInput:
    count: wp.array[int]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    margin0: wp.array[float]
    margin1: wp.array[float]
    shape0: wp.array[int]
    shape1: wp.array[int]
    world: wp.array[int]
    slot: wp.array[int]
    path: wp.array[int]
    art_a: wp.array[int]
    art_b: wp.array[int]
    slots_needed: wp.array[int]
    shape_body: wp.array[int]
    body_to_articulation: wp.array[int]
    art_to_world: wp.array[int]
    response_count: wp.array[int]
    body_response_mask: wp.array[wp.uint32]
    prescribed_articulation: wp.array[int]
    is_free_rigid: wp.array[int]
    shape_mu: wp.array[float]
    shape_restitution: wp.array[float]
    body_q: wp.array[wp.transform]


@wp.struct
class RowState:
    active_worlds: wp.array[int]
    active_count: wp.array[int]
    resolved: wp.array[int]
    predictor_status: wp.array[int]
    state_generation: wp.array[wp.int64]
    held_generation: wp.array[wp.int64]
    v_hat: wp.array[float]
    endpoint_twists: wp.array[wp.spatial_vector]
    dense_count: wp.array[int]
    mf_count: wp.array[int]
    primary_offset: wp.array[int]
    secondary_offset: wp.array[int]
    raw_invalid: wp.array[int]
    capacity_status: wp.array[int]


@wp.struct
class PrefixInput:
    q: wp.array[float]
    q_index: wp.array[int]
    lower: wp.array[float]
    upper: wp.array[float]


@wp.struct
class RowSettings:
    dt: float
    beta: float
    cfm: float
    bias_scale: float
    contact_speculative_scale: float
    joint_limit_speculative_scale: float
    activation_gap: float
    restitution_velocity_threshold: float
    friction_scale: float
    friction_gap_threshold: float
    shared_anchor: int
    friction_shared_anchor: int
    friction_anchor_limit: int
    friction_articulation_pairs_only: int
    enable_friction: int
    contact_w: float
    raw_capacity: int
    dense_capacity: int
    mf_capacity: int
    workers: int


@wp.struct
class ArmMap:
    C: wp.array2d[wp.vec3]
    state_generation: wp.array[wp.int64]
    held_generation: wp.array[wp.int64]
    valid: wp.array[int]


@wp.struct
class DenseRowOutput:
    response: wp.array3d[float]
    physical_J: wp.array3d[float]
    r0: wp.array2d[float]
    rhs: wp.array2d[float]
    diag: wp.array2d[float]
    row_type: wp.array2d[int]
    row_parent: wp.array2d[int]
    row_mu: wp.array2d[float]
    row_beta: wp.array2d[float]
    row_cfm: wp.array2d[float]
    phi: wp.array2d[float]
    target_velocity: wp.array2d[float]
    row_restitution: wp.array2d[float]
    row_w: wp.array2d[float]
    impulses: wp.array2d[float]
    valid: wp.array2d[int]
    slot_counter: wp.array[int]
    phase_bounds: wp.array2d[int]
    secondary_nonzero: wp.array[int]
    status: wp.array[int]
    global_status: wp.array[int]


def allocate_arm_map(worlds, device):
    """Allocate only the small current active-world common-arm map."""
    if worlds <= 0:
        raise ValueError("Arm map requires positive world capacity")
    value = ArmMap()
    value.C = wp.zeros((worlds, 7), dtype=wp.vec3, device=device)
    value.state_generation = wp.zeros(worlds, dtype=wp.int64, device=device)
    value.held_generation = wp.zeros(worlds, dtype=wp.int64, device=device)
    value.valid = wp.zeros(worlds, dtype=int, device=device)
    return value

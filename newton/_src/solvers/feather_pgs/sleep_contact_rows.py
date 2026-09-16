# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dormant raw-contact allocation and held-force consumers.

The caller owns admission, current mask construction and lease lifetime. Only
the original dense/sparse-diagonal contact route may acquire a force lease.
These kernels neither decide sleeping nor change the original allocator.
"""

import warp as wp

from . import kernels


@wp.kernel
def allocate_awake_world_contact_slots(
    dormant: wp.array[int],
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    propagation_articulated_contacts: int,
    propagation_same_articulation: int,
    propagation_free_free: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    propagation_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_anchor_limit: int,
    contact_friction_articulation_pairs_only: int,
    row_capacity_telemetry: int,
    resolved_worlds: wp.array[int],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    propagation_slot_counter: wp.array[int],
    dense_contact_world_flag: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    propagation_dropped_contact_rows: wp.array[int],
    capacity_status: wp.array[int],
):
    """Prepend only a raw dormant mask to the original allocation ABI.

    Awake entries use the original function, including overflow, route, gap and
    friction-anchor rules. Original raw-count overflow remains authoritative.
    Dormant entries retain current articulation/world metadata but reserve no
    row. The mask must cover the original raw-contact capacity.
    """
    thread = wp.tid()
    total_contacts = contact_count[0]
    capacity = contact_shape0.shape[0]
    if total_contacts > capacity:
        if thread == 0:
            wp.atomic_max(capacity_status, 3, 1)
        for c in range(thread, capacity, total_num_threads):
            contact_slot[c] = -1
            contact_path[c] = -1
            contact_slots_needed[c] = 0
        return

    for c in range(thread, total_contacts, total_num_threads):
        if dormant[c] != 0 or resolved_worlds.shape[0] > 0:
            shape_a = contact_shape0[c]
            shape_b = contact_shape1[c]
            body_a = shape_body[shape_a] if shape_a >= 0 else -1
            body_b = shape_body[shape_b] if shape_b >= 0 else -1
            art_a = body_to_articulation[body_a] if body_a >= 0 else -1
            art_b = body_to_articulation[body_b] if body_b >= 0 else -1
            world_a = art_to_world[art_a] if art_a >= 0 else -1
            world_b = art_to_world[art_b] if art_b >= 0 else -1
            world = world_a if world_a >= 0 else world_b
            same_world = world_a < 0 or world_b < 0 or world_a == world_b
            skip = dormant[c] != 0
            if same_world and world >= 0 and world < resolved_worlds.shape[0]:
                skip = skip or resolved_worlds[world] != 0
            if skip:
                contact_world[c] = world if same_world else -1
                contact_art_a[c] = art_a
                contact_art_b[c] = art_b
                contact_slot[c] = -1
                contact_path[c] = -1
                contact_slots_needed[c] = 0
                continue
        kernels._allocate_world_contact_slot(
            c,
            total_contacts,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            body_q,
            shape_transform,
            shape_body,
            body_to_articulation,
            art_to_world,
            articulation_response_dof_count,
            body_flags,
            body_has_response_dofs,
            is_free_rigid,
            has_free_rigid,
            propagation_articulated_contacts,
            propagation_same_articulation,
            propagation_free_free,
            contact_gap_gate,
            same_articulation_contact_gap_gate,
            articulation_pair_contact_gap_gate,
            max_constraints,
            mf_max_constraints,
            propagation_max_constraints,
            enable_friction,
            contact_friction_gap_threshold,
            contact_friction_anchor_limit,
            contact_friction_articulation_pairs_only,
            row_capacity_telemetry,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            world_slot_counter,
            contact_path,
            mf_slot_counter,
            propagation_slot_counter,
            dense_contact_world_flag,
            contact_slots_needed,
            dense_dropped_contact_rows,
            mf_dropped_contact_rows,
            propagation_dropped_contact_rows,
            capacity_status,
        )


@wp.func
def _dense_force(
    c: int,
    world: int,
    slot: int,
    count: int,
    normal: wp.array[wp.vec3],
    impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    enable_friction: int,
    inv_dt: float,
) -> wp.vec3:
    """Apply the existing dense contact-force sign and tangent convention."""
    force = wp.vec3()
    if inv_dt > 0.0:
        n = -normal[c]
        lam_n = impulses[world, slot]
        lam_t0 = float(0.0)
        lam_t1 = float(0.0)
        if (
            enable_friction != 0
            and slot + 2 < count
            and row_type[world, slot + 1] == kernels.PGS_CONSTRAINT_TYPE_FRICTION
            and row_parent[world, slot + 1] == slot
            and row_type[world, slot + 2] == kernels.PGS_CONSTRAINT_TYPE_FRICTION
            and row_parent[world, slot + 2] == slot
        ):
            lam_t0 = impulses[world, slot + 1]
            lam_t1 = impulses[world, slot + 2]
        force = lam_n * n
        if enable_friction != 0:
            tangent0, tangent1 = kernels.contact_tangent_basis(n)
            force += lam_t0 * tangent0 + lam_t1 * tangent1
        force *= inv_dt
    return force


@wp.kernel
def capture_dormant_contact_forces(
    contact_count: wp.array[int],
    dormant: wp.array[int],
    contact_normal: wp.array[wp.vec3],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_path: wp.array[int],
    world_impulses: wp.array2d[float],
    world_constraint_count: wp.array[int],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    enable_friction: int,
    inv_dt: float,
    held_force: wp.array[wp.vec3],
    held_valid: wp.array[int],
    status: wp.array[int],
):
    """Capture newly leased dense forces; retain already skipped valid forces.

    Call after the real solve, before granting a lease or replacing its routing
    metadata. ``dormant`` selects proposed/current leases, not just contacts
    already skipped during allocation. The caller clears status per pass.
    Status bits: unsupported route (1), row/world bounds (2), nonfinite force
    (4), invalid raw count (8). Invalid captures cannot acquire a valid lease.
    """
    c = wp.tid()
    total = contact_count[0]
    if total < 0 or total > held_force.shape[0]:
        if c == 0:
            wp.atomic_or(status, 0, 8)
        if c < held_valid.shape[0]:
            held_valid[c] = 0
        return
    if c >= total:
        return
    if dormant[c] == 0:
        return
    slot = contact_slot[c]
    path = contact_path[c]
    if slot == -1 and path == -1:
        # No row also describes an original gap-filtered zero-force contact.
        # Only previously captured skipped contacts keep a nonzero force.
        if held_valid[c] == 0:
            held_force[c] = wp.vec3()
            held_valid[c] = 1
        return
    if path != 0:
        held_valid[c] = 0
        wp.atomic_or(status, 0, 1)
        return
    world = contact_world[c]
    if (
        world < 0
        or world >= world_impulses.shape[0]
        or world >= world_constraint_count.shape[0]
        or world >= world_row_type.shape[0]
        or world >= world_row_parent.shape[0]
    ):
        held_valid[c] = 0
        wp.atomic_or(status, 0, 2)
        return
    count = world_constraint_count[world]
    if (
        slot < 0
        or slot >= count
        or count > world_impulses.shape[1]
        or count > world_row_type.shape[1]
        or count > world_row_parent.shape[1]
    ):
        held_valid[c] = 0
        wp.atomic_or(status, 0, 2)
        return
    force = _dense_force(
        c, world, slot, count, contact_normal, world_impulses, world_row_type, world_row_parent, enable_friction, inv_dt
    )
    if not wp.isfinite(force[0]) or not wp.isfinite(force[1]) or not wp.isfinite(force[2]):
        held_valid[c] = 0
        wp.atomic_or(status, 0, 4)
        return
    held_force[c] = force
    held_valid[c] = 1


@wp.kernel
def publish_dormant_contact_forces(
    contact_count: wp.array[int],
    dormant: wp.array[int],
    held_force: wp.array[wp.vec3],
    held_valid: wp.array[int],
    rigid_contact_force: wp.array[wp.vec3],
):
    """Overwrite only valid dormant slots after the ordinary public-force export.

    The caller must reject invalid leases before publication, and runs the
    original optional spatial-force pack after this kernel.
    """
    c = wp.tid()
    total = contact_count[0]
    if total < 0 or total > held_force.shape[0]:
        return
    if c < total:
        if dormant[c] != 0 and held_valid[c] != 0:
            rigid_contact_force[c] = held_force[c]

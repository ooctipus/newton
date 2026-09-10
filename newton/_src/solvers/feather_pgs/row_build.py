# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact row build with one thread per (contact, dof).

The legacy ``populate_world_J_for_size`` runs one thread per contact and walks the kinematic chain six times per
contact (normal and two friction rows, two bodies), accumulating into a pre-cleared Jacobian with read-modify-write
stores. It is latency bound (65k threads, 8% issue utilisation). This kernel assigns one thread per (contact, dof):
every thread recomputes the contact-level scalars (cheap, all reads of one contact are L1-resident across the 32
threads of the contact) and produces the three Jacobian entries of its dof directly, using the per-body response-dof
bitmask that the solver already builds, so the whole Jacobian row is written once with coalesced stores. Results are
bit-identical to the legacy kernel: each entry is one signed projection (two when both bodies belong to the same
articulation, summed in the same order).
"""

from __future__ import annotations

import warp as wp

from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    contact_tangent_basis,
    mixed_contact_restitution,
    prescribed_relative_contact_target,
)


@wp.func
def _projection(point_world: wp.vec3, origin: wp.vec3, direction: wp.vec3, S: wp.spatial_vector) -> float:
    lin = wp.vec3(S[0], S[1], S[2])
    ang = wp.vec3(S[3], S[4], S[5])
    v = lin + wp.cross(ang, point_world - origin)
    return wp.dot(direction, v)


@wp.kernel(enable_backward=False)
def contact_row_prelude(
    contact_count: wp.array[int],
    total_threads: int,
    world_constraint_count: wp.array[int],
    mf_constraint_count: wp.array[int],
    owned_rows_le: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    size_a: int,
    size_b: int,
    articulation_response_dof_count: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_shared_anchor: int,
    contact_friction_anchor_limit: int,
    contact_friction_articulation_pairs_only: int,
    is_free_rigid: wp.array[int],
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    pre_flags: wp.array[int],  # bit0 active, bit1 friction rows, bit2 A in size, bit3 B in size
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Per contact row (grid-stride, 3 threads per contact): geometry, friction decisions and row metadata.

    Thread ``r`` of a contact writes the flags (r == 0) and the metadata of row ``slot + r`` only, so each thread's
    dependency chain carries one prescribed-target evaluation and eight stores instead of three and twenty-four.
    """
    tid = wp.tid()
    r = tid % 3
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(tid // 3, total_contacts, total_threads // 3):
        if owned_rows_le > 0:
            # Worlds the sweep kernel owns build their contact rows in-kernel (world_rows); skip them here.
            w = contact_world[c]
            if world_constraint_count[w] <= owned_rows_le and mf_constraint_count[w] == 0:
                continue
        _contact_row_prelude_one(
            c,
            r,
            total_contacts,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            contact_thickness0,
            contact_thickness1,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            size_a,
            size_b,
            articulation_response_dof_count,
            articulation_origin,
            shape_body,
            body_q,
            body_v_s,
            prescribed_articulation,
            shape_material_mu,
            shape_material_restitution,
            enable_friction,
            contact_friction_gap_threshold,
            contact_friction_shared_anchor,
            contact_friction_anchor_limit,
            contact_friction_articulation_pairs_only,
            is_free_rigid,
            contact_friction_scale,
            contact_shared_anchor,
            pgs_beta,
            pgs_cfm,
            pre_flags,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_beta,
            world_row_cfm,
            world_phi,
            world_target_velocity,
            world_row_restitution,
        )


@wp.func
def _contact_row_prelude_one(
    c: int,
    r: int,
    total_contacts: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    size_a: int,
    size_b: int,
    articulation_response_dof_count: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_shared_anchor: int,
    contact_friction_anchor_limit: int,
    contact_friction_articulation_pairs_only: int,
    is_free_rigid: wp.array[int],
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    pre_flags: wp.array[int],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    if r == 0:
        pre_flags[c] = 0
    if contact_path[c] != 0:
        return
    slot = contact_slot[c]
    if slot < 0:
        return
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    a_in = False
    b_in = False
    if art_a >= 0:
        sa = articulation_response_dof_count[art_a]
        a_in = sa == size_a or sa == size_b
    if art_b >= 0:
        sb = articulation_response_dof_count[art_b]
        b_in = sb == size_a or sb == size_b
    if not a_in and not b_in:
        return

    world = contact_world[c]
    normal = -contact_normal[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)
    if body_a >= 0:
        point_a_world = wp.transform_point(body_q[body_a], point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal
    if body_b >= 0:
        point_b_world = wp.transform_point(body_q[body_b], point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal
    phi = wp.dot(normal, point_a_world - point_b_world)

    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)
    restitution = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    apply_friction_filter = contact_friction_articulation_pairs_only == 0 or (a_non_free and b_non_free)
    effective_friction_anchor_limit = int(0)
    if apply_friction_filter:
        effective_friction_anchor_limit = contact_friction_anchor_limit
    friction_anchor_rank = int(0)
    same_next_contact = int(0)
    if effective_friction_anchor_limit > 0:
        for lookback in range(1, 9):
            prev = c - lookback
            if prev < 0:
                break
            if prev >= total_contacts:
                break
            if contact_shape0[prev] == shape_a and contact_shape1[prev] == shape_b:
                friction_anchor_rank += int(1)
            else:
                break
        next = c + 1
        if next < total_contacts and contact_shape0[next] == shape_a and contact_shape1[next] == shape_b:
            same_next_contact = int(1)
    friction_anchor_scale = 1.0
    if effective_friction_anchor_limit > 0 and (friction_anchor_rank > 0 or same_next_contact != 0):
        friction_anchor_scale = 0.5
    friction_mu = mu * contact_friction_scale * friction_anchor_scale

    t0, t1 = contact_tangent_basis(normal)
    will_add_friction = enable_friction != 0 and (not apply_friction_filter or phi <= contact_friction_gap_threshold)
    if effective_friction_anchor_limit > 0 and friction_anchor_rank >= effective_friction_anchor_limit:
        will_add_friction = False
    contact_anchor_world = 0.5 * (point_a_world + point_b_world)
    point_a_normal = point_a_world
    point_b_normal = point_b_world
    if contact_shared_anchor != 0:
        point_a_normal = contact_anchor_world
        point_b_normal = contact_anchor_world
    point_a_friction = point_a_world
    point_b_friction = point_b_world
    if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
        point_a_friction = contact_anchor_world
        point_b_friction = contact_anchor_world

    if r == 0:
        flags = 1
        if will_add_friction:
            flags |= 2
        if a_in:
            flags |= 4
        if b_in:
            flags |= 8
        pre_flags[c] = flags
        normal_target = prescribed_relative_contact_target(
            body_a,
            art_a,
            body_b,
            art_b,
            point_a_normal,
            point_b_normal,
            normal,
            prescribed_articulation,
            articulation_origin,
            body_v_s,
        )
        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = phi
        world_target_velocity[world, slot] = normal_target
        world_row_restitution[world, slot] = restitution
    elif will_add_friction:
        direction = t0
        if r == 2:
            direction = t1
        friction_target = prescribed_relative_contact_target(
            body_a,
            art_a,
            body_b,
            art_b,
            point_a_friction,
            point_b_friction,
            direction,
            prescribed_articulation,
            articulation_origin,
            body_v_s,
        )
        world_row_type[world, slot + r] = PGS_CONSTRAINT_TYPE_FRICTION
        world_row_parent[world, slot + r] = slot
        world_row_mu[world, slot + r] = friction_mu
        world_row_beta[world, slot + r] = 0.0
        world_row_cfm[world, slot + r] = pgs_cfm
        world_phi[world, slot + r] = 0.0
        world_target_velocity[world, slot + r] = friction_target
        world_row_restitution[world, slot + r] = 0.0


@wp.kernel(enable_backward=False)
def populate_world_J_masked(
    contact_count: wp.array[int],
    total_threads: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    size_a: int,
    size_b: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_response_dof_mask: wp.array[wp.uint32],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    contact_friction_shared_anchor: int,
    contact_shared_anchor: int,
    pre_flags: wp.array[int],
    contact_world: wp.array[int],
    world_constraint_count: wp.array[int],
    mf_constraint_count: wp.array[int],
    owned_rows_le: int,
    # outputs: Jacobian blocks of the two size groups (the second may alias the first when only one group exists)
    J_group_a: wp.array3d[float],
    J_group_b: wp.array3d[float],
):
    """One thread per (contact, local dof), grid-stride over contacts: the Jacobian entries of that dof, written once.

    ``lanes`` threads per contact (the larger group size, not 32): idle lanes were 30% of the launch on 22-DOF worlds.
    """
    tid = wp.tid()
    lanes = wp.max(size_a, size_b)
    lane = tid % lanes
    total_contacts = wp.min(contact_count[0], pre_flags.shape[0])
    for c in range(tid // lanes, total_contacts, total_threads // lanes):
        if owned_rows_le > 0:
            # Worlds the sweep kernel owns build their contact rows in-kernel (world_rows); skip them here.
            w = contact_world[c]
            if world_constraint_count[w] <= owned_rows_le and mf_constraint_count[w] == 0:
                continue
        _populate_world_J_masked_one(
            c,
            lane,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            contact_thickness0,
            contact_thickness1,
            contact_slot,
            contact_art_a,
            contact_art_b,
            size_a,
            articulation_response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            body_response_dof_mask,
            joint_S_s,
            shape_body,
            body_q,
            contact_friction_shared_anchor,
            contact_shared_anchor,
            pre_flags,
            J_group_a,
            J_group_b,
        )


@wp.func
def _populate_world_J_masked_one(
    c: int,
    lane: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    size_a: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_response_dof_mask: wp.array[wp.uint32],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    contact_friction_shared_anchor: int,
    contact_shared_anchor: int,
    pre_flags: wp.array[int],
    J_group_a: wp.array3d[float],
    J_group_b: wp.array3d[float],
):
    flags = pre_flags[c]
    if (flags & 1) == 0:
        return
    slot = contact_slot[c]
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    a_in = (flags & 4) != 0
    b_in = (flags & 8) != 0
    will_add_friction = (flags & 2) != 0
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    # Geometry, identical arithmetic to the prelude / legacy builder.
    normal = -contact_normal[c]
    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)
    if body_a >= 0:
        point_a_world = wp.transform_point(body_q[body_a], contact_point0[c]) - thickness_a * normal
    else:
        point_a_world = contact_point0[c] - thickness_a * normal
    if body_b >= 0:
        point_b_world = wp.transform_point(body_q[body_b], contact_point1[c]) + thickness_b * normal
    else:
        point_b_world = contact_point1[c] + thickness_b * normal
    t0, t1 = contact_tangent_basis(normal)
    contact_anchor_world = 0.5 * (point_a_world + point_b_world)
    pan = point_a_world
    pbn = point_b_world
    if contact_shared_anchor != 0:
        pan = contact_anchor_world
        pbn = contact_anchor_world
    paf = point_a_world
    pbf = point_b_world
    if contact_shared_anchor != 0 or contact_friction_shared_anchor != 0:
        paf = contact_anchor_world
        pbf = contact_anchor_world
    bit = wp.uint32(1 << lane)

    if a_in and lane < articulation_response_dof_count[art_a]:
        group_a = art_group_idx[art_a]
        origin_a = articulation_origin[art_a]
        S_a = joint_S_s[art_dof_start[art_a] + lane]
        jn = float(0.0)
        jt0 = float(0.0)
        jt1 = float(0.0)
        if body_a >= 0 and (body_response_dof_mask[body_a] & bit) != wp.uint32(0):
            jn += _projection(pan, origin_a, normal, S_a)
            jt0 += _projection(paf, origin_a, t0, S_a)
            jt1 += _projection(paf, origin_a, t1, S_a)
        if b_in and art_b == art_a:
            if body_b >= 0 and (body_response_dof_mask[body_b] & bit) != wp.uint32(0):
                jn += -_projection(pbn, origin_a, normal, S_a)
                jt0 += -_projection(pbf, origin_a, t0, S_a)
                jt1 += -_projection(pbf, origin_a, t1, S_a)
        if articulation_response_dof_count[art_a] == size_a:
            J_group_a[group_a, slot, lane] = jn
            if will_add_friction:
                J_group_a[group_a, slot + 1, lane] = jt0
                J_group_a[group_a, slot + 2, lane] = jt1
        else:
            J_group_b[group_a, slot, lane] = jn
            if will_add_friction:
                J_group_b[group_a, slot + 1, lane] = jt0
                J_group_b[group_a, slot + 2, lane] = jt1
    if b_in and (not a_in or art_b != art_a) and lane < articulation_response_dof_count[art_b]:
        group_b = art_group_idx[art_b]
        origin_b = articulation_origin[art_b]
        S_b = joint_S_s[art_dof_start[art_b] + lane]
        jn = float(0.0)
        jt0 = float(0.0)
        jt1 = float(0.0)
        if body_b >= 0 and (body_response_dof_mask[body_b] & bit) != wp.uint32(0):
            jn = -_projection(pbn, origin_b, normal, S_b)
            jt0 = -_projection(pbf, origin_b, t0, S_b)
            jt1 = -_projection(pbf, origin_b, t1, S_b)
        if articulation_response_dof_count[art_b] == size_a:
            J_group_a[group_b, slot, lane] = jn
            if will_add_friction:
                J_group_a[group_b, slot + 1, lane] = jt0
                J_group_a[group_b, slot + 2, lane] = jt1
        else:
            J_group_b[group_b, slot, lane] = jn
            if will_add_friction:
                J_group_b[group_b, slot + 1, lane] = jt0
                J_group_b[group_b, slot + 2, lane] = jt1

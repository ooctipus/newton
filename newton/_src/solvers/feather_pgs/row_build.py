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

Persistent patch friction has its own prelude and Jacobian pass (``*_patch``): the allocation's extent decides which
contacts own tangent rows, and those rows follow the patch row law (:class:`PatchRowLaw`), as the generic per-contact
builder writes them.
"""

from __future__ import annotations

import warp as wp

from .contact_laws import (
    contact_friction_points,
    contact_geometry,
    contact_materials,
    contact_row_frame,
    contact_tangent_basis,
    jacobian_entry,
    prescribed_target,
)
from .friction_patches import PatchRowLaw, patch_friction_location
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION


@wp.kernel(enable_backward=False)
def contact_row_prelude(
    contact_count: wp.array[int],
    total_threads: int,
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
    contact_slots_needed: wp.array[int],
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
    contact_friction_shared_anchor: int,
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
    total_contacts = contact_count[0]
    for c in range(tid // 3, total_contacts, total_threads // 3):
        _contact_row_prelude_one(
            c,
            r,
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
            contact_slots_needed,
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
            contact_friction_shared_anchor,
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
    contact_slots_needed: wp.array[int],
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
    contact_friction_shared_anchor: int,
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
    body_a, body_b, normal, point_a, point_b, phi = contact_geometry(
        c,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        shape_body,
        body_q,
    )
    mu, restitution = contact_materials(
        contact_shape0[c], contact_shape1[c], shape_material_mu, shape_material_restitution
    )
    tangent0, tangent1 = contact_tangent_basis(normal)
    friction_a, friction_b = contact_friction_points(
        point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
    )
    # Allocation owns the row extent; recomputing eligibility can cross a
    # floating-point threshold and overwrite the following contact.
    will_add_friction = contact_slots_needed[c] >= 3
    if r == 0:
        flags = 1
        if will_add_friction:
            flags |= 2
        if a_in:
            flags |= 4
        if b_in:
            flags |= 8
        pre_flags[c] = flags
    if r == 0 or will_add_friction:
        direction, p0, p1 = contact_row_frame(
            r, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
        )
        row = slot + r
        if r == 0:
            world_row_type[world, row] = PGS_CONSTRAINT_TYPE_CONTACT
            world_row_parent[world, row] = -1
            world_row_mu[world, row] = mu
            world_row_beta[world, row] = pgs_beta
            world_phi[world, row] = phi
            world_row_restitution[world, row] = restitution
        else:
            world_row_type[world, row] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, row] = slot
            world_row_mu[world, row] = mu * contact_friction_scale
            world_row_beta[world, row] = 0.0
            world_phi[world, row] = 0.0
            world_row_restitution[world, row] = 0.0
        world_row_cfm[world, row] = pgs_cfm
        world_target_velocity[world, row] = prescribed_target(
            body_a, art_a, body_b, art_b, p0, p1, direction, prescribed_articulation, articulation_origin, body_v_s
        )


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
    total_contacts = contact_count[0]
    for c in range(tid // lanes, total_contacts, total_threads // lanes):
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
    body_a, body_b, normal, point_a, point_b, _phi = contact_geometry(
        c,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        shape_body,
        body_q,
    )
    tangent0, tangent1 = contact_tangent_basis(normal)
    friction_a, friction_b = contact_friction_points(
        point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
    )
    _write_contact_jacobian_lane(
        lane,
        flags,
        contact_slot[c],
        contact_art_a[c],
        contact_art_b[c],
        body_a,
        body_b,
        normal,
        tangent0,
        tangent1,
        point_a,
        point_b,
        friction_a,
        friction_b,
        contact_shared_anchor,
        size_a,
        articulation_response_dof_count,
        art_group_idx,
        art_dof_start,
        articulation_origin,
        body_response_dof_mask,
        joint_S_s,
        J_group_a,
        J_group_b,
    )


@wp.kernel(enable_backward=False)
def contact_row_prelude_patch(
    contact_count: wp.array[int],
    total_threads: int,
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
    contact_slots_needed: wp.array[int],
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
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    law: PatchRowLaw,
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
    """Per contact row (grid-stride, 3 threads per contact): the metadata of the persistent patch row law.

    A contact owns tangent rows when the allocation reserved them (``contact_slots_needed`` of 3). They
    act at the patch friction location, carry the anchor's tangential displacement as phi with
    ``law.beta`` and keep the undivided friction coefficient: the patch link, which runs after every
    row producer, divides it and links the normal rows into rings.
    """
    tid = wp.tid()
    r = tid % 3
    total_contacts = contact_count[0]
    for c in range(tid // 3, total_contacts, total_threads // 3):
        _contact_row_prelude_patch_one(
            c,
            r,
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
            contact_slots_needed,
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
            contact_friction_scale,
            contact_shared_anchor,
            pgs_beta,
            pgs_cfm,
            law,
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
def _contact_row_prelude_patch_one(
    c: int,
    r: int,
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
    contact_slots_needed: wp.array[int],
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
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    law: PatchRowLaw,
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
    will_add_friction = contact_slots_needed[c] == 3
    if r > 0 and not will_add_friction:
        return

    world = contact_world[c]
    body_a, body_b, normal, point_a, point_b, phi = contact_geometry(
        c,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        shape_body,
        body_q,
    )
    mu, restitution = contact_materials(
        contact_shape0[c], contact_shape1[c], shape_material_mu, shape_material_restitution
    )
    tangent0, tangent1 = contact_tangent_basis(normal)
    friction_a, friction_b = patch_friction_location(law.patches, c)
    direction, p0, p1 = contact_row_frame(
        r, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
    )
    row = slot + r
    if r == 0:
        flags = 1
        if will_add_friction:
            flags |= 2
        if a_in:
            flags |= 4
        if b_in:
            flags |= 8
        pre_flags[c] = flags
        world_row_type[world, row] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, row] = -1
        world_row_mu[world, row] = mu
        world_row_beta[world, row] = pgs_beta
        world_phi[world, row] = phi
        world_row_restitution[world, row] = restitution
    else:
        world_row_type[world, row] = PGS_CONSTRAINT_TYPE_FRICTION
        world_row_parent[world, row] = slot
        world_row_mu[world, row] = mu * contact_friction_scale
        world_row_beta[world, row] = law.beta
        world_phi[world, row] = law.patches.phi[c][r - 1]
        world_row_restitution[world, row] = 0.0
    world_row_cfm[world, row] = pgs_cfm
    world_target_velocity[world, row] = prescribed_target(
        body_a, art_a, body_b, art_b, p0, p1, direction, prescribed_articulation, articulation_origin, body_v_s
    )


@wp.kernel(enable_backward=False)
def populate_world_J_masked_patch(
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
    contact_shared_anchor: int,
    law: PatchRowLaw,
    pre_flags: wp.array[int],
    # outputs: Jacobian blocks of the two size groups (the second may alias the first when only one group exists)
    J_group_a: wp.array3d[float],
    J_group_b: wp.array3d[float],
):
    """One thread per (contact, local dof): the patch rows' Jacobian entries of that dof, written once.

    Tangent rows act at the patch friction location; everything else is :func:`populate_world_J_masked`'s.
    """
    tid = wp.tid()
    lanes = wp.max(size_a, size_b)
    lane = tid % lanes
    total_contacts = contact_count[0]
    for c in range(tid // lanes, total_contacts, total_threads // lanes):
        flags = pre_flags[c]
        if (flags & 1) == 0:
            continue
        body_a, body_b, normal, point_a, point_b, _phi = contact_geometry(
            c,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            shape_body,
            body_q,
        )
        tangent0, tangent1 = contact_tangent_basis(normal)
        friction_a, friction_b = patch_friction_location(law.patches, c)
        _write_contact_jacobian_lane(
            lane,
            flags,
            contact_slot[c],
            contact_art_a[c],
            contact_art_b[c],
            body_a,
            body_b,
            normal,
            tangent0,
            tangent1,
            point_a,
            point_b,
            friction_a,
            friction_b,
            contact_shared_anchor,
            size_a,
            articulation_response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            body_response_dof_mask,
            joint_S_s,
            J_group_a,
            J_group_b,
        )


@wp.func
def _write_contact_jacobian_lane(
    lane: int,
    flags: int,
    slot: int,
    art_a: int,
    art_b: int,
    body_a: int,
    body_b: int,
    normal: wp.vec3,
    tangent0: wp.vec3,
    tangent1: wp.vec3,
    point_a: wp.vec3,
    point_b: wp.vec3,
    friction_a: wp.vec3,
    friction_b: wp.vec3,
    shared_anchor: int,
    size_a: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_response_dof_mask: wp.array[wp.uint32],
    joint_S_s: wp.array[wp.spatial_vector],
    J_group_a: wp.array3d[float],
    J_group_b: wp.array3d[float],
):
    """Write one dof of a contact's normal row and, with flag bit 1, its two tangent rows.

    Each entry is one signed :func:`.contact_laws.jacobian_entry`, or two summed (A's first) when both bodies
    belong to the same articulation. Rows act where :func:`.contact_laws.contact_row_frame` puts them.
    """
    a_in = (flags & 4) != 0
    b_in = (flags & 8) != 0
    rows = 1
    if (flags & 2) != 0:
        rows = 3
    bit = wp.uint32(1 << lane)
    if a_in and lane < articulation_response_dof_count[art_a]:
        group = art_group_idx[art_a]
        origin = articulation_origin[art_a]
        S = joint_S_s[art_dof_start[art_a] + lane]
        a_moves = body_a >= 0 and (body_response_dof_mask[body_a] & bit) != wp.uint32(0)
        b_moves = b_in and art_b == art_a and body_b >= 0 and (body_response_dof_mask[body_b] & bit) != wp.uint32(0)
        for row in range(rows):
            direction, p0, p1 = contact_row_frame(
                row, shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
            )
            value = float(0.0)
            if a_moves:
                value += jacobian_entry(S, direction, p0, origin)
            if b_moves:
                value += -jacobian_entry(S, direction, p1, origin)
            if articulation_response_dof_count[art_a] == size_a:
                J_group_a[group, slot + row, lane] = value
            else:
                J_group_b[group, slot + row, lane] = value
    if b_in and (not a_in or art_b != art_a) and lane < articulation_response_dof_count[art_b]:
        group = art_group_idx[art_b]
        origin = articulation_origin[art_b]
        S = joint_S_s[art_dof_start[art_b] + lane]
        b_moves = body_b >= 0 and (body_response_dof_mask[body_b] & bit) != wp.uint32(0)
        for row in range(rows):
            direction, p0, p1 = contact_row_frame(
                row, shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
            )
            value = float(0.0)
            if b_moves:
                value = -jacobian_entry(S, direction, p1, origin)
            if articulation_response_dof_count[art_b] == size_a:
                J_group_a[group, slot + row, lane] = value
            else:
                J_group_b[group, slot + row, lane] = value

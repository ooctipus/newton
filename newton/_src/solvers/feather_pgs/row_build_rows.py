# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact row build with one thread per contact row.

The two-pass builder (``row_build.py``: a per-contact prelude writing flags and metadata, then a per-(contact, dof)
Jacobian pass) is bound by the per-contact dependency chain of the prelude, not by arithmetic or threads: with every
thread resident at once, the prelude takes as long as one contact's chain of dependent loads and 24 scattered stores.

This kernel gives each contact three threads, one per row (normal, friction 0, friction 1). Each thread evaluates the
contact-level scalars (identical arithmetic to the prelude), writes the metadata of its own row and the Jacobian
entries of its own row for the articulation(s) of its size group(s) in a contiguous loop over the DOFs. Nothing is
staged between passes and the geometry is computed three times per contact instead of once per DOF. Results are
bit-identical to the two-pass builder (same projections, same summation order).
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
from .row_build import _projection


@wp.func
def _write_row_J(
    art: int,
    body_self: int,
    point_self: wp.vec3,
    sign_self: float,
    body_other: int,
    point_other: wp.vec3,
    other_same_art: bool,
    direction: wp.vec3,
    row: int,
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
    """Jacobian entries of one row for one articulation: sign * projection at the body's DOFs (both bodies if same art)."""
    n = articulation_response_dof_count[art]
    group = art_group_idx[art]
    origin = articulation_origin[art]
    dof_start = art_dof_start[art]
    mask_self = wp.uint32(0)
    if body_self >= 0:
        mask_self = body_response_dof_mask[body_self]
    mask_other = wp.uint32(0)
    if other_same_art and body_other >= 0:
        mask_other = body_response_dof_mask[body_other]
    in_a = n == size_a
    for d in range(n):
        bit = wp.uint32(1 << d)
        S = joint_S_s[dof_start + d]
        v = float(0.0)
        if (mask_self & bit) != wp.uint32(0):
            v += sign_self * _projection(point_self, origin, direction, S)
        if (mask_other & bit) != wp.uint32(0):
            v += -sign_self * _projection(point_other, origin, direction, S)
        if in_a:
            J_group_a[group, row, d] = v
        else:
            J_group_b[group, row, d] = v


@wp.kernel(enable_backward=False)
def build_contact_rows(
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
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    J_group_a: wp.array3d[float],
    J_group_b: wp.array3d[float],
):
    """One thread per (contact, row), grid-stride over contacts: metadata and Jacobian of that row, written once."""
    tid = wp.tid()
    r = tid % 3
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(tid // 3, total_contacts, total_threads // 3):
        if contact_path[c] != 0:
            continue
        slot = contact_slot[c]
        if slot < 0:
            continue
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
            continue
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
        will_add_friction = enable_friction != 0 and (
            not apply_friction_filter or phi <= contact_friction_gap_threshold
        )
        if effective_friction_anchor_limit > 0 and friction_anchor_rank >= effective_friction_anchor_limit:
            will_add_friction = False
        if r > 0 and not will_add_friction:
            continue
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
        # This row's direction, anchors and metadata.
        row = slot + r
        direction = normal
        pa = point_a_normal
        pb = point_b_normal
        if r == 1:
            direction = t0
            pa = point_a_friction
            pb = point_b_friction
        elif r == 2:
            direction = t1
            pa = point_a_friction
            pb = point_b_friction
        target = prescribed_relative_contact_target(
            body_a, art_a, body_b, art_b, pa, pb, direction, prescribed_articulation, articulation_origin, body_v_s
        )
        if r == 0:
            world_row_type[world, row] = PGS_CONSTRAINT_TYPE_CONTACT
            world_row_parent[world, row] = -1
            world_row_mu[world, row] = mu
            world_row_beta[world, row] = pgs_beta
            world_row_cfm[world, row] = pgs_cfm
            world_phi[world, row] = phi
            world_target_velocity[world, row] = target
            world_row_restitution[world, row] = mixed_contact_restitution(shape_a, shape_b, shape_material_restitution)
        else:
            world_row_type[world, row] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, row] = slot
            world_row_mu[world, row] = friction_mu
            world_row_beta[world, row] = 0.0
            world_row_cfm[world, row] = pgs_cfm
            world_phi[world, row] = 0.0
            world_target_velocity[world, row] = target
            world_row_restitution[world, row] = 0.0
        # Jacobian of this row. Articulation A gets +projection at body A (and -projection at body B when B is the
        # same articulation); articulation B (different) gets -projection at body B. Same order as the two-pass builder.
        if a_in:
            _write_row_J(
                art_a,
                body_a,
                pa,
                1.0,
                body_b,
                pb,
                b_in and art_b == art_a,
                direction,
                row,
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
        if b_in and (not a_in or art_b != art_a):
            _write_row_J(
                art_b,
                body_b,
                pb,
                -1.0,
                -1,
                pb,
                False,
                direction,
                row,
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

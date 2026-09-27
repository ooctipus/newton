# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact row build: one warp per contact, native CUDA.

Every lane of the warp evaluates the contact-level prelude (the loads are warp-uniform, so they are served once and
the instruction count per contact is that of a single thread), lanes below the articulation DOF count write the
three Jacobian entries of their dof from the per-body response-dof bitmask, and the 24 row-metadata scalars are
written by 24 lanes in parallel. No scratch buffers, no per-capacity launches, no Jacobian clear needed for the rows
it writes. Arithmetic mirrors ``populate_world_J_for_size`` operation for operation.
"""

from __future__ import annotations

import warp as wp

from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION


def get_row_build_native_kernel() -> wp.Kernel:
    T_CONTACT = int(PGS_CONSTRAINT_TYPE_CONTACT)
    T_FRICTION = int(PGS_CONSTRAINT_TYPE_FRICTION)
    snippet = f"""
#if defined(__CUDA_ARCH__)
    using vec3 = wp::vec_t<3, float>;
    using vec6 = wp::vec_t<6, float>;
    const int lane = threadIdx.x;
    const int total = min(contact_count.data[0], (int)contact_point0.shape.dims[0]);
    for (int c = block; c < total; c += nblocks) {{
        if (contact_path.data[c] != 0) continue;
        const int slot = contact_slot.data[c];
        if (slot < 0) continue;
        const int art_a = contact_art_a.data[c];
        const int art_b = contact_art_b.data[c];
        const bool a_in = art_a >= 0 && articulation_response_dof_count.data[art_a] == target_size;
        const bool b_in = art_b >= 0 && articulation_response_dof_count.data[art_b] == target_size;
        if (!a_in && !b_in) continue;

        const int world = contact_world.data[c];
        const vec3 normal = -contact_normal.data[c];
        const int shape_a = contact_shape0.data[c];
        const int shape_b = contact_shape1.data[c];
        const int body_a = shape_a >= 0 ? shape_body.data[shape_a] : -1;
        const int body_b = shape_b >= 0 ? shape_body.data[shape_b] : -1;
        const float thickness_a = contact_thickness0.data[c];
        const float thickness_b = contact_thickness1.data[c];
        vec3 point_a_world, point_b_world;
        if (body_a >= 0) point_a_world = wp::transform_point(body_q.data[body_a], contact_point0.data[c]) - thickness_a * normal;
        else point_a_world = contact_point0.data[c] - thickness_a * normal;
        if (body_b >= 0) point_b_world = wp::transform_point(body_q.data[body_b], contact_point1.data[c]) + thickness_b * normal;
        else point_b_world = contact_point1.data[c] + thickness_b * normal;
        const float phi = wp::dot(normal, point_a_world - point_b_world);

        float mu = 0.0f; int mat_count = 0;
        if (shape_a >= 0) {{ mu += shape_material_mu.data[shape_a]; mat_count += 1; }}
        if (shape_b >= 0) {{ mu += shape_material_mu.data[shape_b]; mat_count += 1; }}
        if (mat_count > 0) mu /= (float)mat_count;
        // mixed_contact_restitution
        float restitution = 0.0f; int material_count = 0;
        if (shape_a >= 0) {{ const float va = shape_material_restitution.data[shape_a]; if (isfinite(va)) restitution += wp::clamp(va, 0.0f, 1.0f); material_count += 1; }}
        if (shape_b >= 0) {{ const float vb = shape_material_restitution.data[shape_b]; if (isfinite(vb)) restitution += wp::clamp(vb, 0.0f, 1.0f); material_count += 1; }}
        if (material_count > 0) restitution /= (float)material_count;
        const bool a_non_free = art_a >= 0 && is_free_rigid.data[art_a] == 0;
        const bool b_non_free = art_b >= 0 && is_free_rigid.data[art_b] == 0;
        const bool apply_friction_filter = contact_friction_articulation_pairs_only == 0 || (a_non_free && b_non_free);
        int effective_friction_anchor_limit = 0;
        if (apply_friction_filter) effective_friction_anchor_limit = contact_friction_anchor_limit;
        int friction_anchor_rank = 0; int same_next_contact = 0;
        if (effective_friction_anchor_limit > 0) {{
            for (int lookback = 1; lookback < 9; ++lookback) {{
                const int prev = c - lookback;
                if (prev < 0) break;
                if (prev >= total) break;
                if (contact_shape0.data[prev] == shape_a && contact_shape1.data[prev] == shape_b) friction_anchor_rank += 1;
                else break;
            }}
            const int next = c + 1;
            if (next < total && contact_shape0.data[next] == shape_a && contact_shape1.data[next] == shape_b) same_next_contact = 1;
        }}
        float friction_anchor_scale = 1.0f;
        if (effective_friction_anchor_limit > 0 && (friction_anchor_rank > 0 || same_next_contact != 0)) friction_anchor_scale = 0.5f;
        const float friction_mu = mu * contact_friction_scale * friction_anchor_scale;

        // contact_tangent_basis
        vec3 t0 = wp::cross(normal, vec3(1.0f, 0.0f, 0.0f));
        if (wp::length_sq(t0) < 1.0e-12f) t0 = wp::cross(normal, vec3(0.0f, 1.0f, 0.0f));
        t0 = wp::normalize(t0);
        const vec3 t1 = wp::normalize(wp::cross(normal, t0));
        bool will_add_friction = enable_friction != 0 && (!apply_friction_filter || phi <= contact_friction_gap_threshold);
        if (effective_friction_anchor_limit > 0 && friction_anchor_rank >= effective_friction_anchor_limit) will_add_friction = false;
        const vec3 contact_anchor_world = 0.5f * (point_a_world + point_b_world);
        vec3 pan = point_a_world, pbn = point_b_world;
        if (contact_shared_anchor != 0) {{ pan = contact_anchor_world; pbn = contact_anchor_world; }}
        vec3 paf = point_a_world, pbf = point_b_world;
        if (contact_shared_anchor != 0 || contact_friction_shared_anchor != 0) {{ paf = contact_anchor_world; pbf = contact_anchor_world; }}

        // prescribed_relative_contact_target for the three directions
        auto prescribed = [&](int body, int art, float sign, const vec3& p, const vec3& dir) -> float {{
            float value = 0.0f;
            if (body >= 0 && art >= 0 && prescribed_articulation.data[art] != 0) {{
                const vec6 tw = body_v_s.data[body];
                const vec3 lin(tw.c[0], tw.c[1], tw.c[2]);
                const vec3 ang(tw.c[3], tw.c[4], tw.c[5]);
                const vec3 pv = lin + wp::cross(ang, p - articulation_origin.data[art]);
                value = sign * wp::dot(dir, pv);
            }}
            return value;
        }};
        const float normal_target = -(prescribed(body_a, art_a, 1.0f, pan, normal) + prescribed(body_b, art_b, -1.0f, pbn, normal));
        const float friction0_target = -(prescribed(body_a, art_a, 1.0f, paf, t0) + prescribed(body_b, art_b, -1.0f, pbf, t0));
        const float friction1_target = -(prescribed(body_a, art_a, 1.0f, paf, t1) + prescribed(body_b, art_b, -1.0f, pbf, t1));

        // Jacobian entries of dof `lane`
        auto projection = [&](const vec3& p, const vec3& origin, const vec3& dir, const vec6& S) -> float {{
            const vec3 lin(S.c[0], S.c[1], S.c[2]);
            const vec3 ang(S.c[3], S.c[4], S.c[5]);
            const vec3 v = lin + wp::cross(ang, p - origin);
            return wp::dot(dir, v);
        }};
        if (lane < target_size) {{
            const unsigned bit = 1u << lane;
            if (a_in) {{
                const int group_a = art_group_idx.data[art_a];
                const vec3 origin_a = articulation_origin.data[art_a];
                const vec6 S_a = joint_S_s.data[art_dof_start.data[art_a] + lane];
                float jn = 0.0f, jt0 = 0.0f, jt1 = 0.0f;
                if (body_a >= 0 && (body_response_dof_mask.data[body_a] & bit) != 0u) {{
                    jn += projection(pan, origin_a, normal, S_a);
                    jt0 += projection(paf, origin_a, t0, S_a);
                    jt1 += projection(paf, origin_a, t1, S_a);
                }}
                if (b_in && art_b == art_a) {{
                    if (body_b >= 0 && (body_response_dof_mask.data[body_b] & bit) != 0u) {{
                        jn += -projection(pbn, origin_a, normal, S_a);
                        jt0 += -projection(pbf, origin_a, t0, S_a);
                        jt1 += -projection(pbf, origin_a, t1, S_a);
                    }}
                }}
                const size_t base = ((size_t)group_a * (size_t)J_group.shape.dims[1] + (size_t)slot) * (size_t)J_group.shape.dims[2] + (size_t)lane;
                J_group.data[base] = jn;
                if (will_add_friction) {{
                    J_group.data[base + (size_t)J_group.shape.dims[2]] = jt0;
                    J_group.data[base + 2 * (size_t)J_group.shape.dims[2]] = jt1;
                }}
            }}
            if (b_in && (!a_in || art_b != art_a)) {{
                const int group_b = art_group_idx.data[art_b];
                const vec3 origin_b = articulation_origin.data[art_b];
                const vec6 S_b = joint_S_s.data[art_dof_start.data[art_b] + lane];
                float jn = 0.0f, jt0 = 0.0f, jt1 = 0.0f;
                if (body_b >= 0 && (body_response_dof_mask.data[body_b] & bit) != 0u) {{
                    jn = -projection(pbn, origin_b, normal, S_b);
                    jt0 = -projection(pbf, origin_b, t0, S_b);
                    jt1 = -projection(pbf, origin_b, t1, S_b);
                }}
                const size_t base = ((size_t)group_b * (size_t)J_group.shape.dims[1] + (size_t)slot) * (size_t)J_group.shape.dims[2] + (size_t)lane;
                J_group.data[base] = jn;
                if (will_add_friction) {{
                    J_group.data[base + (size_t)J_group.shape.dims[2]] = jt0;
                    J_group.data[base + 2 * (size_t)J_group.shape.dims[2]] = jt1;
                }}
            }}
        }}
        // Row metadata: lane = row * 8 + field (rows 1 and 2 only with friction)
        {{
            const int row = lane >> 3;
            const int field = lane & 7;
            if (row < 3 && (row == 0 || will_add_friction)) {{
                const size_t idx = (size_t)world * (size_t)world_row_type.shape.dims[1] + (size_t)(slot + row);
                const bool fr = row > 0;
                switch (field) {{
                    case 0: world_row_type.data[idx] = fr ? {T_FRICTION} : {T_CONTACT}; break;
                    case 1: world_row_parent.data[idx] = fr ? slot : -1; break;
                    case 2: world_row_mu.data[idx] = fr ? friction_mu : mu; break;
                    case 3: world_row_beta.data[idx] = fr ? 0.0f : pgs_beta; break;
                    case 4: world_row_cfm.data[idx] = pgs_cfm; break;
                    case 5: world_phi.data[idx] = fr ? 0.0f : phi; break;
                    case 6: world_target_velocity.data[idx] = (row == 0) ? normal_target : (row == 1 ? friction0_target : friction1_target); break;
                    default: world_row_restitution.data[idx] = fr ? 0.0f : restitution; break;
                }}
            }}
        }}
    }}
#endif
"""

    @wp.func_native(snippet)
    def row_build_native(
        block: int,
        nblocks: int,
        contact_count: wp.array[int],
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
        target_size: int,
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
        J_group: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_row_beta: wp.array2d[float],
        world_row_cfm: wp.array2d[float],
        world_phi: wp.array2d[float],
        world_target_velocity: wp.array2d[float],
        world_row_restitution: wp.array2d[float],
    ): ...

    def row_build_native_template(
        nblocks: int,
        contact_count: wp.array[int],
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
        target_size: int,
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
        J_group: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_row_beta: wp.array2d[float],
        world_row_cfm: wp.array2d[float],
        world_phi: wp.array2d[float],
        world_target_velocity: wp.array2d[float],
        world_row_restitution: wp.array2d[float],
    ):
        block, _lane = wp.tid()
        row_build_native(
            block,
            nblocks,
            contact_count,
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
            target_size,
            articulation_response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            body_response_dof_mask,
            joint_S_s,
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
            J_group,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_beta,
            world_row_cfm,
            world_phi,
            world_target_velocity,
            world_row_restitution,
        )

    name = "populate_world_J_native"
    row_build_native_template.__name__ = name
    row_build_native_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(row_build_native_template)

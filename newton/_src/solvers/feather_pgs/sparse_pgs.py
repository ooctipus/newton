# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ordered PGS over ancestor-sparse, mass-whitened constraint rows."""

from functools import cache

import warp as wp

from .friction import FRICTION_PAIR_CUDA
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from .sparse_mass_matrix import _SparseRowSupport


@cache
def _get_pgs_solve_sparse_kernel(
    max_constraints: int,
    max_world_dofs: int,
    max_row_dofs: int,
    *,
    mf_max_constraints: int = 0,
    free_row_dofs: int = 0,
    active_rows: bool = False,
    branch_groups: tuple[int, ...] = (),
    use_support_table: bool = False,
) -> wp.Kernel:
    """Build a CUDA sweep over factored articulations and physical free bodies.

    Rows contain unique factor-coordinate indices, padded with -1. Both
    tangent rows of a contact must have identical, ordered support. The caller
    supplies ``row_incident = J v_hat``. Factor blocks store ``Z = J P^T L^-T``
    once; physical free blocks store J and their response separately. All
    coordinates accumulate deltas from v_hat; only factor blocks need decoding.

    This uses the existing interleaved row order and paired friction law,
    including linked patch normal loads. The caller admits only augmented
    drives, immediate response, and no warm start, torsion or ordinary joint
    velocity-limit rows. Seeded patch impulses retain the ordinary delta semantics.
    Independent free-body rows follow the articulated rows in every sweep,
    retaining their own response, capacity, friction law, and impulse buffer.
    ``active_rows`` selects point-contact normal/limit rows at sweep entry,
    then retains their ordered GS updates. Newly violated zero rows wait until
    the next sweep; mixed responses and linked patches retain the full sweep.

    ``use_support_table`` reads identical padded indices from a shared
    topology table. It requires a single factored articulation per world,
    at offset zero, with no physical free-body response.

    ``branch_groups`` experimentally partitions factor DOFs into two to four
    branches, with -1 identifying the shared separator. Its first sweep stays
    serial; later sweeps retain each branch's own separator response and combine
    branch increments at sweep boundaries. This changes finite-iteration
    convergence, not the contact law. Separator-only, cross-branch, seeded and
    linked-patch worlds retain serial sweeps. This mode requires active rows,
    no free bodies, and at most 64 factor DOFs. A point contact's normal and
    tangent rows must belong to the same branch, as in the sparse row producer.

    Launch the default with ``ceil(world_count / 2)`` tiles and 64 threads per
    tile. Branch sweeps use one 32-thread tile per world.
    """
    M, D, S = int(max_constraints), int(max_world_dofs), int(max_row_dofs)
    F = int(mf_max_constraints)
    P = int(free_row_dofs)
    if M <= 0 or D <= 0 or S <= 0 or S > D or F < 0 or P < 0 or P > S:
        raise ValueError("sparse PGS requires positive capacities and max_row_dofs <= max_world_dofs")
    if use_support_table and (F or P or D > 64):
        raise ValueError("sparse support tables require at most 64 factored DOFs and no free rows")
    index_read = (
        f"support.class_dof.data[support.row_class.data[offset + row] * {S} + entry]"
        if use_support_table
        else f"row_dof.data[row_base + row * {S} + entry]"
    )
    parallel_branches = bool(branch_groups)
    if parallel_branches:
        groups = set(branch_groups)
        branch_count = int(max(groups)) + 1
        if (
            len(branch_groups) != D
            or D > 64
            or not 2 <= branch_count <= 4
            or groups - {-1} != set(range(branch_count))
            or min(groups) < -1
            or any(int(group) != group for group in branch_groups)
            or not active_rows
            or F
            or P
        ):
            raise ValueError("branch sweeps require two to four factor-coordinate groups and no free rows")
    W = 1 if parallel_branches else 2
    L = 8 if parallel_branches else 32
    Q = (S + L - 1) // L
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    active_rows = bool(active_rows) and F == 0 and P == 0
    active_storage = active_initialize = active_scan = ""
    row_loop = "        for (int row = 0; row < count; ++row) {"
    row_loop_end = "        }"
    if active_rows:
        words = (M + 31) // 32
        scan_lane = "thread" if parallel_branches else "lane"
        scan_mask = "FULL_MASK" if parallel_branches else "MASK"
        scan_delta = "global_delta" if parallel_branches else "delta"
        active_storage = f"""
    __shared__ unsigned normal_words_storage[{W * words}];
    __shared__ unsigned selected_words_storage[{W * words}];
    unsigned* normal_words = normal_words_storage + slot * {words};
    unsigned* selected_words = selected_words_storage + slot * {words};
"""
        active_initialize = f"""
    bool linked = false;
    for (int row = lane; row < count; row += 32)
        if (type[row] == {contact_type} && world_row_parent.data[offset + row] >= 0) linked = true;
    const bool select_active = __ballot_sync(MASK, linked) == 0u && isfinite(omega) && omega >= 0.0f;
"""
        active_scan = f"""
        for (int first = 0; first < count; first += 32) {{
            const int row = first + {scan_lane};
            bool selected = false;
            if (row < count) {{
                const int kind = static_cast<int>(type[row]);
                if (kind != {friction_type}) {{
                    selected = !select_active || (kind != {contact_type} && kind != {limit_type})
                        || impulse[row] != 0.0f;
                    if (!selected && diagonal[row] > 0.0f) {{
                        float dot = 0.0f;
                        // Delta starts at zero even when the impulse buffer is seeded.
                        if (iteration != 0) {{
                            #pragma unroll
                            for (int entry = 0; entry < {S}; ++entry) {{
                                const int index = {index_read};
                                if (index >= 0)
                                    dot += row_factor.data[row_base + row * {S} + entry] * {scan_delta}[index];
                            }}
                        }}
                        const float residual = dot + rhs[row];
                        selected = residual < 0.0f || !isfinite(residual);
                    }}
                }} else if (select_active && global_iteration < friction_start_iteration) {{
                    // The active list visits only first tangents; clear both delayed rows here.
                    impulse[row] = 0.0f;
                }}
            }}
            const unsigned mask = __ballot_sync({scan_mask}, selected);
            if ({scan_lane} == 0) normal_words[first / 32] = mask;
        }}
        __syncwarp({scan_mask});
        for (int first = 0; first < count; first += 32) {{
            const int row = first + {scan_lane};
            bool selected = false;
            if (row < count) {{
                selected = !select_active || ((normal_words[first / 32] >> {scan_lane}) & 1u) != 0u;
                if (select_active && type[row] == {friction_type}
                    && global_iteration >= friction_start_iteration) {{
                    const int parent = world_row_parent.data[offset + row];
                    selected = row == parent + 1 && (
                        ((normal_words[parent / 32] >> (parent & 31)) & 1u) != 0u
                        || impulse[row] != 0.0f || impulse[parent + 2] != 0.0f);
                }}
            }}
            const unsigned mask = __ballot_sync({scan_mask}, selected);
            if ({scan_lane} == 0) selected_words[first / 32] = mask;
        }}
        __syncwarp({scan_mask});
"""
        row_loop = """
        for (int word = 0; word < (count + 31) / 32; ++word) {
            unsigned pending = selected_words[word];
            while (pending != 0u) {
                const int row = word * 32 + __ffs(pending) - 1;
                pending &= pending - 1u;"""
        row_loop_end = "            }\n        }"
    mf_storage = mf_initialize = mf_sweep = mf_writeback = ""
    if F:
        mf_storage = f"""
    __shared__ float mf_impulse_storage[{W * F}];
    __shared__ float mf_incident_storage[{W * F}];
    const int mf_offset = world * {F};
    const int mf_count = min(mf_constraint_count.data[world], {F});
    float* mf_impulse = mf_impulse_storage + slot * {F};
    float* mf_incident = mf_incident_storage + slot * {F};
"""
        mf_initialize = f"""
    for (int i = lane; i < mf_count; i += 32) {{
        const int global_row = mf_offset + i;
        mf_impulse[i] = mf_impulses.data[global_row];
        const int dofs = mf_meta.data[global_row * 4];
        const int first = dofs >> 16;
        const int second = (dofs << 16) >> 16;
        float incident = 0.0f;
        for (int k = 0; k < 6; ++k) {{
            if (first >= 0)
                incident += mf_J_a.data[global_row * 6 + k]
                    * v_hat.data[world_dof_indices.data[world * {D} + first + k]];
            if (second >= 0)
                incident += mf_J_b.data[global_row * 6 + k]
                    * v_hat.data[world_dof_indices.data[world * {D} + second + k]];
        }}
        mf_incident[i] = incident;
    }}
"""
        mf_sweep = f"""
        for (int row = 0; row < mf_count; ++row) {{
            const int global_row = mf_offset + row;
            const int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[global_row * 4]);
            const int kind = meta.w & 0xffff;
            const bool friction = kind == {friction_type};
            if (friction && global_iteration < friction_start_iteration) {{
                if (lane == 0) mf_impulse[row] = 0.0f;
                __syncwarp(MASK);
                continue;
            }}
            const int parent = meta.w >> 16;
            if (friction && row != parent + 1) continue;
            const int sibling = friction ? parent + 2 : -1;
            const float inverse_diagonal = __int_as_float(meta.y);
            if (!friction && inverse_diagonal <= 0.0f) continue;
            const int first = meta.x >> 16;
            const int second = (meta.x << 16) >> 16;
            int index = -1;
            if (lane < 6 && first >= 0) index = first + lane;
            if (lane >= 6 && lane < 12 && second >= 0) index = second + lane - 6;
            float left = 0.0f, right = 0.0f, sibling_left = 0.0f, sibling_right = 0.0f;
            if (index >= 0 && lane < 6) {{
                left = mf_J_a.data[global_row * 6 + lane];
                right = mf_MiJt_a.data[global_row * 6 + lane];
                if (friction) {{
                    sibling_left = mf_J_a.data[(mf_offset + sibling) * 6 + lane];
                    sibling_right = mf_MiJt_a.data[(mf_offset + sibling) * 6 + lane];
                }}
            }}
            if (index >= 0 && lane >= 6 && lane < 12) {{
                left = mf_J_b.data[global_row * 6 + lane - 6];
                right = mf_MiJt_b.data[global_row * 6 + lane - 6];
                if (friction) {{
                    sibling_left = mf_J_b.data[(mf_offset + sibling) * 6 + lane - 6];
                    sibling_right = mf_MiJt_b.data[(mf_offset + sibling) * 6 + lane - 6];
                }}
            }}
            // A coordinate has one writer, including coincident endpoint blocks.
            if (first >= 0 && first == second) {{
                if (lane < 6) {{
                    left += mf_J_b.data[global_row * 6 + lane];
                    right += mf_MiJt_b.data[global_row * 6 + lane];
                    if (friction) {{
                        sibling_left += mf_J_b.data[(mf_offset + sibling) * 6 + lane];
                        sibling_right += mf_MiJt_b.data[(mf_offset + sibling) * 6 + lane];
                    }}
                }} else {{
                    index = -1;
                    left = right = sibling_left = sibling_right = 0.0f;
                }}
            }}
            float dot = index >= 0 ? left * delta[index] : 0.0f;
            float sibling_dot = index >= 0 ? sibling_left * delta[index] : 0.0f;
            float cross = left * sibling_right;
            for (int shift = 16; shift > 0; shift >>= 1) {{
                dot += __shfl_down_sync(MASK, dot, shift);
                sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift);
                cross += __shfl_down_sync(MASK, cross, shift);
            }}
            const float residual = __shfl_sync(MASK, dot, 0)
                + mf_incident[row] + __int_as_float(meta.z);
            const float old_impulse = mf_impulse[row];
            float new_impulse = old_impulse;
            float sibling_change = 0.0f;
            if (friction) {{
                float normal_load = mf_impulse[parent];
                for (int next = mf_meta.data[(mf_offset + parent) * 4 + 3] >> 16;
                     next >= 0 && next != parent; next = mf_meta.data[(mf_offset + next) * 4 + 3] >> 16)
                    normal_load += mf_impulse[next];
                const float radius = fmaxf(mf_row_mu.data[global_row] * normal_load, 0.0f);
                const float old_sibling = mf_impulse[sibling];
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    const float sibling_inverse = __int_as_float(mf_meta.data[(mf_offset + sibling) * 4 + 1]);
                    const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0)
                        + mf_incident[sibling]
                        + __int_as_float(mf_meta.data[(mf_offset + sibling) * 4 + 2]);
                    pair = friction_pair_candidate(inverse_diagonal > 0.0f ? 1.0f / inverse_diagonal : 0.0f,
                        __shfl_sync(MASK, cross, 0), sibling_inverse > 0.0f ? 1.0f / sibling_inverse : 0.0f,
                        residual, sibling_residual, old_impulse, old_sibling, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                const float new_sibling = pair.y * scale;
                sibling_change = new_sibling - old_sibling;
                if (lane == 0) mf_impulse[sibling] = new_sibling;
            }} else if (kind == {velocity_limit_type}) {{
                // Velocity limits apply a stateless, unrelaxed correction.
                new_impulse = residual < 0.0f ? -residual * inverse_diagonal : 0.0f;
            }} else {{
                new_impulse = old_impulse - omega * residual * inverse_diagonal;
                if (kind == {contact_type}) new_impulse = fmaxf(new_impulse, 0.0f);
            }}
            const float change = kind == {velocity_limit_type} ? new_impulse : new_impulse - old_impulse;
            if (lane == 0) mf_impulse[row] = new_impulse;
            if (change != 0.0f || sibling_change != 0.0f) {{
                changed = 1;
                if (index >= 0) {{
                    float value = delta[index];
                    if (sibling_change != 0.0f) value += sibling_right * sibling_change;
                    if (change != 0.0f) value += right * change;
                    delta[index] = value;
                }}
            }}
            __syncwarp(MASK);
        }}
"""
        mf_writeback = """
    for (int i = lane; i < mf_count; i += 32) mf_impulses.data[mf_offset + i] = mf_impulse[i];
"""
    setup = f"""
    __shared__ float delta_storage[{W * D}];
    __shared__ float impulse_storage[{W * M}];
    __shared__ float rhs_storage[{W * M}];
    __shared__ float diagonal_storage[{W * M}];
    __shared__ unsigned char type_storage[{W * M}];
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int slot = threadIdx.x >> 5;
    const int offset = world * {M};
    const int row_base = offset * {S};
    const int count = min(world_constraint_count.data[world], {M});
    float* delta = delta_storage + slot * {D};
    float* impulse = impulse_storage + slot * {M};
    float* rhs = rhs_storage + slot * {M};
    float* diagonal = diagonal_storage + slot * {M};
    unsigned char* type = type_storage + slot * {M};
{mf_storage}
{active_storage}

    for (int d = lane; d < {D}; d += 32) delta[d] = 0.0f;
    for (int i = lane; i < count; i += 32) {{
        impulse[i] = world_impulses.data[offset + i];
        rhs[i] = rhs_bias.data[offset + i] + row_incident.data[offset + i];
        diagonal[i] = world_diag.data[offset + i];
        type[i] = static_cast<unsigned char>(world_row_type.data[offset + i]);
    }}
{mf_initialize}
    __syncwarp(MASK);
{active_initialize}
"""
    row_update = f"""
            const int row_type = static_cast<int>(type[row]);
            const bool friction = row_type == {friction_type};
            if (friction && global_iteration < friction_start_iteration) {{
                if (lane == 0) impulse[row] = 0.0f;
                __syncwarp(MASK);
                continue;
            }}
            const int parent = friction ? world_row_parent.data[offset + row] : -1;
            if (friction && row != parent + 1) continue;
            const int sibling = friction ? parent + 2 : -1;
            const float denominator = diagonal[row];
            if (!friction && denominator <= 0.0f) continue;
            const float old_impulse = impulse[row];
            float old_sibling = 0.0f;
            float radius = 0.0f;
            if (friction) {{
                float normal_load = impulse[parent];
                for (int next = world_row_parent.data[offset + parent]; next >= 0 && next != parent;
                     next = world_row_parent.data[offset + next])
                    normal_load += impulse[next];
                radius = fmaxf(world_row_mu.data[offset + row] * normal_load, 0.0f);
                old_sibling = impulse[sibling];
                // A zero-radius, zero-impulse pair cannot change the velocity.
                if (radius == 0.0f && old_impulse == 0.0f && old_sibling == 0.0f) continue;
            }}

            int indices[{Q}];
            float responses[{Q}];
            float sibling_responses[{Q}];
            float dot = 0.0f;
            float sibling_dot = 0.0f;
            float cross = 0.0f;
            #pragma unroll
            for (int q = 0; q < {Q}; ++q) {{
                const int entry = lane + q * {L};
                const int index = entry < {S} ? {index_read} : -1;
                const float value = index >= 0 ? row_factor.data[row_base + row * {S} + entry] : 0.0f;
                const float sibling_value = friction && index >= 0
                    ? row_factor.data[row_base + sibling * {S} + entry] : 0.0f;
                float response = value;
                float sibling_response = sibling_value;
                if ({str(bool(P)).lower()} && index >= 0 && world_free_dof_mask.data[world * {D} + index] != 0) {{
                    const int width = row_free_response.shape[2];
                    response = row_free_response.data[(offset + row) * width + entry];
                    if (friction)
                        sibling_response = row_free_response.data[(offset + sibling) * width + entry];
                }}
                indices[q] = index;
                responses[q] = response;
                sibling_responses[q] = sibling_response;
                if (index >= 0) {{
                    const float velocity = delta[index];
                    dot += value * velocity;
                    sibling_dot += sibling_value * velocity;
                    cross += value * sibling_response;
                }}
            }}
            for (int shift = {L // 2}; shift > 0; shift >>= 1) {{
                dot += __shfl_down_sync(MASK, dot, shift, {L});
                sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift, {L});
                cross += __shfl_down_sync(MASK, cross, shift, {L});
            }}
            const float residual = __shfl_sync(MASK, dot, 0, {L}) + rhs[row];
            float new_impulse = old_impulse;
            float sibling_change = 0.0f;
            if (friction) {{
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0, {L}) + rhs[sibling];
                    pair = friction_pair_candidate(denominator, __shfl_sync(MASK, cross, 0, {L}),
                        diagonal[sibling], residual, sibling_residual, old_impulse, old_sibling, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                const float new_sibling = pair.y * scale;
                sibling_change = new_sibling - old_sibling;
                if (lane == 0) impulse[sibling] = new_sibling;
            }} else {{
                new_impulse = old_impulse - omega * residual / denominator;
                if (row_type == {contact_type} || row_type == {limit_type})
                    new_impulse = fmaxf(new_impulse, 0.0f);
            }}
            const float change = new_impulse - old_impulse;
            if (lane == 0) impulse[row] = new_impulse;
            if (change != 0.0f || sibling_change != 0.0f) {{
                changed = 1;
                #pragma unroll
                for (int q = 0; q < {Q}; ++q) {{
                    if (indices[q] >= 0) {{
                        // Match the existing paired-tangent response order.
                        float value = delta[indices[q]];
                        if (sibling_change != 0.0f) value += sibling_responses[q] * sibling_change;
                        if (change != 0.0f) value += responses[q] * change;
                        delta[indices[q]] = value;
                    }}
                }}
            }}
            __syncwarp(MASK);
"""
    sweep_start = ""
    sweep_finish = """
        if (global_iteration >= friction_start_iteration && __ballot_sync(MASK, changed != 0) == 0u) break;
"""
    writeback = f"""
    for (int d = lane; d < {D}; d += 32) factor_velocity_delta.data[world * {D} + d] = delta[d];
    for (int i = lane; i < count; i += 32) world_impulses.data[offset + i] = impulse[i];
{mf_writeback}
"""
    branch_lookup = ""
    if parallel_branches:
        # Literal masks avoid a dynamically indexed per-thread topology table.
        masks = [sum(1 << dof for dof, group in enumerate(branch_groups) if group == b) for b in range(branch_count)]
        membership = "".join(f"(bit & {mask}ull) != 0ull ? {b} : " for b, mask in enumerate(masks)) + "-1"
        branch_lookup = f"""
    const auto sparse_pgs_branch_group = [](int index) {{
        const unsigned long long bit = 1ull << index;
        return {membership};
    }};
"""
        setup = f"""
    constexpr unsigned FULL_MASK = 0xffffffffu;
    const int thread = threadIdx.x;
    const int lane = thread & 7;
    const int branch = thread >> 3;
    const unsigned MASK = 0xffu << (branch * 8);
    const int offset = world * {M};
    const int row_base = offset * {S};
    const int count = min(world_constraint_count.data[world], {M});
    __shared__ float delta_storage[{4 * D}], global_delta[{D}];
    __shared__ float impulse[{M}], rhs[{M}], diagonal[{M}];
    __shared__ unsigned char type[{M}];
    __shared__ unsigned normal_words[{words}], selected_words[{words}], owner_words[{4 * words}];
    __shared__ int serial_flags, branch_changed[4];
    float* delta = delta_storage + branch * {D};
    if (thread == 0) serial_flags = 0;
    for (int d = lane; d < {D}; d += 8) delta[d] = 0.0f;
    for (int d = thread; d < {D}; d += 32) global_delta[d] = 0.0f;
    __syncwarp(FULL_MASK);
    for (int first = 0; first < count; first += 32) {{
        const int row = first + thread;
        int group = -1;
        if (row < count) {{
            impulse[row] = world_impulses.data[offset + row];
            rhs[row] = rhs_bias.data[offset + row] + row_incident.data[offset + row];
            diagonal[row] = world_diag.data[offset + row];
            type[row] = static_cast<unsigned char>(world_row_type.data[offset + row]);
            if (impulse[row] != 0.0f) atomicOr(&serial_flags, 1);
            if (type[row] == {contact_type} && world_row_parent.data[offset + row] >= 0)
                atomicOr(&serial_flags, 3);
            #pragma unroll
            for (int entry = 0; entry < {S}; ++entry) {{
                const int index = {index_read};
                if (index >= 0) {{
                    const int next = sparse_pgs_branch_group(index);
                    if (next >= 0) {{
                        if (group >= 0 && group != next) atomicOr(&serial_flags, 1);
                        group = next;
                    }}
                }}
            }}
            // Separator-only and cross-branch rows retain the complete serial recurrence.
            if (group < 0) atomicOr(&serial_flags, 1);
        }}
        #pragma unroll
        for (int b = 0; b < 4; ++b) {{
            const unsigned mask = __ballot_sync(FULL_MASK, row < count && group == b);
            if (thread == 0) owner_words[b * {words} + first / 32] = mask;
        }}
    }}
    __syncwarp(FULL_MASK);
    const bool select_active = (serial_flags & 2) == 0 && isfinite(omega) && omega >= 0.0f;
"""
        # Every call starts from zero delta, including phased calls with an iteration offset.
        sweep_start = "        const bool serial_sweep = serial_flags != 0 || iteration == 0;"
        row_loop = f"""
        for (int word = 0; word < (count + 31) / 32; ++word) {{
            unsigned pending = selected_words[word] & (serial_sweep
                ? (branch == 0 ? 0xffffffffu : 0u) : owner_words[branch * {words} + word]);
            while (pending != 0u) {{
                const int row = word * 32 + __ffs(pending) - 1;
                pending &= pending - 1u;
"""
        row_loop_end = "            }\n        }"
        sweep_finish = f"""
        if (lane == 0) branch_changed[branch] = changed;
        __syncwarp(FULL_MASK);
        for (int d = thread; d < {D}; d += 32) {{
            float value = global_delta[d];
            const int group = sparse_pgs_branch_group(d);
            if (serial_sweep) value = delta_storage[d];
            else if (group >= 0) value = delta_storage[group * {D} + d];
            else {{
                const float previous = value;
                #pragma unroll
                for (int b = 0; b < 4; ++b)
                    value += delta_storage[b * {D} + d] - previous;
            }}
            global_delta[d] = value;
        }}
        __syncwarp(FULL_MASK);
        for (int d = lane; d < {D}; d += 8) delta[d] = global_delta[d];
        bool any_changed = false;
        #pragma unroll
        for (int b = 0; b < 4; ++b) any_changed |= branch_changed[b] != 0;
        __syncwarp(FULL_MASK);
        if (global_iteration >= friction_start_iteration && !any_changed) break;
"""
        writeback = f"""
    for (int d = thread; d < {D}; d += 32)
        factor_velocity_delta.data[world * {D} + d] = global_delta[d];
    for (int row = thread; row < count; row += 32) world_impulses.data[offset + row] = impulse[row];
"""
    snippet = (
        "#if defined(__CUDA_ARCH__)\n"
        + FRICTION_PAIR_CUDA
        + branch_lookup
        + setup
        + f"""
    for (int iteration = 0; iteration < iterations; ++iteration) {{
        const int global_iteration = iteration_offset + iteration;
        int changed = 0;
{sweep_start}
{active_scan}
{row_loop}
{row_update}
{row_loop_end}
{mf_sweep}
{sweep_finish}
    }}
{writeback}
#endif
"""
    )

    @wp.func_native(snippet)
    def solve_native(
        world: int,
        world_constraint_count: wp.array[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_free_dof_mask: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        v_hat: wp.array[float],
        mf_constraint_count: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
        support: _SparseRowSupport,
    ): ...

    def solve(
        world_count: int,
        world_constraint_count: wp.array[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_free_dof_mask: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        v_hat: wp.array[float],
        mf_constraint_count: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
        support: _SparseRowSupport,
    ):
        block, lane = wp.tid()
        world = block * W + lane // 32
        if world < world_count:
            solve_native(
                world,
                world_constraint_count,
                rhs_bias,
                world_diag,
                world_impulses,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                world_row_type,
                world_row_parent,
                world_row_mu,
                world_free_dof_mask,
                world_dof_indices,
                v_hat,
                mf_constraint_count,
                mf_meta,
                mf_impulses,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_row_mu,
                iterations,
                omega,
                friction_start_iteration,
                iteration_offset,
                factor_velocity_delta,
                support,
            )

    suffix = "_active" if active_rows else ""
    if parallel_branches:
        suffix += "_branches"
    if use_support_table:
        suffix += "_support_table"
    solve.__name__ = f"pgs_solve_sparse_{M}_{D}_{S}_{F}_{P}{suffix}"
    solve.__qualname__ = solve.__name__
    return wp.kernel(enable_backward=False, module="unique")(solve)

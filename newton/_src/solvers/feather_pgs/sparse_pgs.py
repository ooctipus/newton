# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ordered PGS over ancestor-sparse, mass-whitened constraint rows."""

from functools import cache

import warp as wp

from .friction import FRICTION_PAIR_CACHED_CUDA, FRICTION_PAIR_CUDA
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from .sparse_mass_matrix import _SparseRowSupport

_SPARSE_CONTACT_METRIC_CACHE_SIZE = 12
_SPARSE_LOCAL_WORKSPACE_BYTES = 8 * 1024


@cache
def _get_pgs_solve_sparse_kernel(
    max_constraints: int,
    max_world_dofs: int,
    max_row_dofs: int,
    *,
    mf_max_constraints: int = 0,
    free_row_dofs: int = 0,
    active_rows: bool = False,
    use_support_table: bool = False,
    contact_triplets: bool = False,
    velocity_limits: bool = False,
    lanes_per_world: int = 32,
    local_factors: bool = False,
) -> wp.Kernel:
    """Build a CUDA sweep over factored articulations and physical free bodies.

    Rows contain unique factor-coordinate indices, padded with -1. Both
    tangent rows of a contact must have identical, ordered support. The caller
    supplies ``row_incident = J v_hat``. Factor blocks store ``Z = J P^T L^-T``
    once; physical free blocks store J and their response separately. All
    coordinates accumulate deltas from v_hat; only factor blocks need decoding.

    This uses the existing interleaved row order and paired friction law,
    including linked patch normal loads. The caller admits only augmented
    drives, immediate response, and no warm start or torsion. ``velocity_limits``
    adds a final articulated limit pass to each sweep, with stateless,
    unrelaxed corrections accumulated in the factor delta. It requires full
    ordered rows and no independent free-body response. Seeded patch impulses
    retain the ordinary delta semantics.
    Independent free-body rows follow the articulated rows in every sweep,
    retaining their own response, capacity, friction law, and impulse buffer.
    ``active_rows`` selects point-contact normal/limit rows at sweep entry,
    then retains their ordered GS updates. Newly violated zero rows wait until
    the next sweep; mixed responses and linked patches retain the full sweep.

    ``use_support_table`` reads identical padded indices from a shared
    topology table. It requires a single factored articulation per world,
    at offset zero, with no physical free-body response.

    ``contact_triplets`` optionally retains tangent metrics within this call
    and publishes an admitted normal/tangent transaction once. It requires
    active rows and support tables; unsupported owners retain the ordinary
    sweep. The bounded metric cache recomputes uncached pairs when full;
    runtime metadata guards retain the ordinary update for other rows.

    Small complete articulated sweeps may use 8 or 16 lanes per world, as in
    the local response/solve owner. Launch with ``ceil(world_count / (64 /
    lanes_per_world))`` tiles and 64 threads per tile. Active-row bitsets and
    independent free-body work retain whole warps.

    ``local_factors`` retains small factor rows in private shared storage,
    like the local response/solve owner. It expands sparse indices once per
    call; it neither applies a second factor transform nor stores physical Y.
    """
    M, D, S = int(max_constraints), int(max_world_dofs), int(max_row_dofs)
    F = int(mf_max_constraints)
    P = int(free_row_dofs)
    if M <= 0 or D <= 0 or S <= 0 or S > D or F < 0 or P < 0 or P > S:
        raise ValueError("sparse PGS requires positive capacities and max_row_dofs <= max_world_dofs")
    if use_support_table and (F or P or D > 64):
        raise ValueError("sparse support tables require at most 64 factored DOFs and no free rows")
    if velocity_limits and (F or P or active_rows or contact_triplets):
        raise ValueError("sparse joint velocity limits require full articulated row sweeps")
    L = int(lanes_per_world)
    if L not in (8, 16, 32):
        raise ValueError("sparse PGS requires 8, 16 or 32 lanes per world")
    if L < 32 and (D > L or F or P or active_rows or contact_triplets):
        raise ValueError("subwarp sparse PGS requires a small complete articulated sweep")
    if local_factors and (D > 16 or M > 64 or F or P or active_rows or contact_triplets):
        raise ValueError("local factor storage requires a small complete articulated sweep")
    index_read = (
        f"support.class_dof.data[support.row_class.data[offset + row] * {S} + entry]"
        if use_support_table
        else f"row_dof.data[row_base + row * {S} + entry]"
    )
    W = 64 // L
    row_width = D if local_factors else S
    Q = (row_width + L - 1) // L
    factor_read = f"row_factor.data[row_base + row * {S} + entry]"
    sibling_read = f"row_factor.data[row_base + sibling * {S} + entry]"
    factor_storage = factor_initialize = ""
    if local_factors:
        factor_storage = f"""
    __shared__ float local_factor_storage[{W * M * D}];
    float* local_factor = local_factor_storage + slot * {M * D};
"""
        factor_initialize = f"""
    for (int i = lane; i < count * {D}; i += {L}) local_factor[i] = 0.0f;
    __syncwarp(MASK);
    for (int i = lane; i < count * {S}; i += {L}) {{
        const int row = i / {S}, entry = i % {S};
        const int index = {index_read};
        if (index >= 0) local_factor[row * {D} + index] = {factor_read};
    }}
"""
        index_read = "entry"
        factor_read = f"local_factor[row * {D} + entry]"
        sibling_read = f"local_factor[sibling * {D} + entry]"
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    active_rows = bool(active_rows) and F == 0 and P == 0
    contact_triplets = bool(contact_triplets) and active_rows and use_support_table and M >= 3
    active_storage = active_initialize = active_scan = ""
    row_loop = "        for (int row = 0; row < count; ++row) {"
    row_loop_end = "        }"
    if active_rows:
        words = (M + 31) // 32
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
            const int row = first + lane;
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
                                    dot += row_factor.data[row_base + row * {S} + entry] * delta[index];
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
            const unsigned mask = __ballot_sync(MASK, selected);
            if (lane == 0) normal_words[first / 32] = mask;
        }}
        __syncwarp(MASK);
        for (int first = 0; first < count; first += 32) {{
            const int row = first + lane;
            bool selected = false;
            if (row < count) {{
                selected = !select_active || ((normal_words[first / 32] >> lane) & 1u) != 0u;
                if (select_active && type[row] == {friction_type}
                    && global_iteration >= friction_start_iteration) {{
                    const int parent = world_row_parent.data[offset + row];
                    selected = row == parent + 1 && (
                        ((normal_words[parent / 32] >> (parent & 31)) & 1u) != 0u
                        || impulse[row] != 0.0f || impulse[parent + 2] != 0.0f);
                }}
            }}
            const unsigned mask = __ballot_sync(MASK, selected);
            if (lane == 0) selected_words[first / 32] = mask;
        }}
        __syncwarp(MASK);
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
                if (!(fabsf(change) <= change_tolerance && fabsf(sibling_change) <= change_tolerance)) changed = 1;
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
    const unsigned MASK = {hex((1 << L) - 1)}u << (threadIdx.x & {32 - L});
    const int lane = threadIdx.x & {L - 1};
    const int slot = threadIdx.x / {L};
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
{factor_storage}

    for (int d = lane; d < {D}; d += {L}) delta[d] = 0.0f;
    for (int i = lane; i < count; i += {L}) {{
        impulse[i] = world_impulses.data[offset + i];
        rhs[i] = rhs_bias.data[offset + i] + row_incident.data[offset + i];
        diagonal[i] = world_diag.data[offset + i];
        type[i] = static_cast<unsigned char>(world_row_type.data[offset + i]);
    }}
{mf_initialize}
{factor_initialize}
    __syncwarp(MASK);
{active_initialize}
"""
    velocity_filter = velocity_sweep = ""
    if velocity_limits:
        # Storage puts velocity rows before contacts, but their original law
        # owns the final pass. Scalar rows need only one residual reduction.
        velocity_filter = f"""
            if (row_type == {velocity_limit_type}) continue;
"""
        velocity_sweep = f"""
        for (int row = 0; row < count; ++row) {{
            if (type[row] != {velocity_limit_type}) continue;
            const float denominator = diagonal[row];
            if (denominator <= 0.0f) continue;
            int indices[{Q}];
            float responses[{Q}];
            float dot = 0.0f;
            #pragma unroll
            for (int q = 0; q < {Q}; ++q) {{
                const int entry = lane + q * {L};
                const int index = entry < {row_width} ? {index_read} : -1;
                const float response = index >= 0 ? {factor_read} : 0.0f;
                indices[q] = index;
                responses[q] = response;
                if (index >= 0) dot += response * delta[index];
            }}
            for (int shift = {L // 2}; shift > 0; shift >>= 1)
                dot += __shfl_down_sync(MASK, dot, shift, {L});
            const float residual = __shfl_sync(MASK, dot, 0, {L}) + rhs[row];
            // Limits report this correction without undoing earlier ones.
            const float correction = residual < 0.0f ? -residual / denominator : 0.0f;
            if (lane == 0) impulse[row] = correction;
            if (correction != 0.0f) {{
                if (!(fabsf(correction) <= change_tolerance)) changed = 1;
                #pragma unroll
                for (int q = 0; q < {Q}; ++q)
                    if (indices[q] >= 0) delta[indices[q]] += responses[q] * correction;
            }}
            __syncwarp(MASK);
        }}
"""
    row_update = f"""
            const int row_type = static_cast<int>(type[row]);
{velocity_filter}
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
                const int index = entry < {row_width} ? {index_read} : -1;
                const float value = index >= 0 ? {factor_read} : 0.0f;
                const float sibling_value = friction && index >= 0
                    ? {sibling_read} : 0.0f;
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
                if (!(fabsf(change) <= change_tolerance && fabsf(sibling_change) <= change_tolerance)) changed = 1;
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
    sweep_finish = """
        if (global_iteration >= friction_start_iteration && __ballot_sync(MASK, changed != 0) == 0u) break;
"""
    friction_source = FRICTION_PAIR_CUDA
    if contact_triplets:
        triplet_setup, triplet_update = _contact_triplet_code(M, S, Q, contact_type, friction_type)
        setup += triplet_setup
        active_scan += "\n        int consumed_tangent = -1;\n"
        row_update = row_update.replace(
            "            int indices[",
            "            const int metric_slot = friction ? contact_metric_slot(parent) : -1;\n"
            "            const bool metric_ready = metric_slot >= 0 && metric_map[metric_slot] >= 0;\n"
            "            int indices[",
            1,
        ).replace("cross += value * sibling_response;", "if (!metric_ready) cross += value * sibling_response;")
        row_update = row_update.replace(
            "cross += __shfl_down_sync(MASK, cross, shift, 32);",
            "if (!metric_ready) cross += __shfl_down_sync(MASK, cross, shift, 32);",
        ).replace(
            "pair = friction_pair_candidate(denominator,",
            "pair = friction_pair_cached(metric_slot, denominator,",
        )
        row_update = triplet_update + row_update
        friction_source = FRICTION_PAIR_CACHED_CUDA
    writeback = f"""
    for (int d = lane; d < {D}; d += {L}) factor_velocity_delta.data[world * {D} + d] = delta[d];
    for (int i = lane; i < count; i += {L}) world_impulses.data[offset + i] = impulse[i];
{mf_writeback}
"""
    snippet = (
        "#if defined(__CUDA_ARCH__)\n"
        + friction_source
        + setup
        + f"""
    for (int iteration = 0; iteration < iterations; ++iteration) {{
        const int global_iteration = iteration_offset + iteration;
        int changed = 0;
{active_scan}
{row_loop}
{row_update}
{row_loop_end}
{mf_sweep}
{velocity_sweep}
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
        change_tolerance: float,
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
        change_tolerance: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
        support: _SparseRowSupport,
    ):
        block, lane = wp.tid()
        world = block * W + lane // L
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
                change_tolerance,
                friction_start_iteration,
                iteration_offset,
                factor_velocity_delta,
                support,
            )

    suffix = "_active" if active_rows else ""
    if use_support_table:
        suffix += "_support_table"
    if contact_triplets:
        suffix += "_contact_triplets"
    if velocity_limits:
        suffix += "_velocity_limits"
    if L != 32:
        suffix += f"_lanes{L}"
    if local_factors:
        suffix += "_local_factors"
    solve.__name__ = f"pgs_solve_sparse_{M}_{D}_{S}_{F}_{P}{suffix}"
    solve.__qualname__ = solve.__name__
    return wp.kernel(enable_backward=False, module="unique")(solve)


def _contact_triplet_code(M: int, S: int, Q: int, contact_type: int, friction_type: int) -> tuple[str, str]:
    """Native additions for one-call metrics and ordered register-held updates."""
    pairs = M // 3
    cache_size = min(_SPARSE_CONTACT_METRIC_CACHE_SIZE, pairs)
    setup = f"""
    __shared__ float metric_storage[{2 * cache_size * 5}];
    __shared__ signed char metric_map_storage[{2 * pairs}];
    __shared__ int metric_count_storage[2];
    float* metrics = metric_storage + slot * {cache_size * 5};
    signed char* metric_map = metric_map_storage + slot * {pairs};
    int* metric_count = metric_count_storage + slot;
    for (int id = lane; id < {pairs}; id += 32) metric_map[id] = -1;
    if (lane == 0) *metric_count = 0;
    __syncwarp(MASK);
    const auto contact_metric_slot = [&](int normal) {{
        if (!select_active || normal < 0 || normal + 2 >= count
            || type[normal] != {contact_type}
            || type[normal + 1] != {friction_type} || type[normal + 2] != {friction_type}
            || world_row_parent.data[offset + normal] >= 0
            || world_row_parent.data[offset + normal + 1] != normal
            || world_row_parent.data[offset + normal + 2] != normal
            || support.row_class.data[offset + normal + 1]
                != support.row_class.data[offset + normal + 2]) return -1;
        // Full, disjoint three-row records make normal/3 injective even when
        // another row kind precedes a contact or a triplet crosses a mask word.
        return normal / 3;
    }};
    const auto friction_pair_cached = [&](int id, float a, float c, float d,
        float r0, float r1, float old0, float old1, float radius, float relaxation) {{
        FrictionPairMetric metric;
        const int cached = id >= 0 ? metric_map[id] : -1;
        if (cached >= 0) {{
            metric = FrictionPairMetric{{metrics[cached * 5], metrics[cached * 5 + 1], metrics[cached * 5 + 2],
                metrics[cached * 5 + 3], metrics[cached * 5 + 4]}};
        }} else {{
            metric = friction_pair_metric(a, c, d);
            if (id >= 0 && lane == 0 && *metric_count < {cache_size}) {{
                // One warp owns this world's ordered visits. A full cache
                // leaves the map unset, retaining exact recomputation.
                const int entry = *metric_count;
                metrics[entry * 5] = metric.scale;
                metrics[entry * 5 + 1] = metric.largest;
                metrics[entry * 5 + 2] = metric.smallest;
                metrics[entry * 5 + 3] = metric.vx;
                metrics[entry * 5 + 4] = metric.vy;
                metric_map[id] = static_cast<signed char>(entry);
                *metric_count = entry + 1;
            }}
            __syncwarp(MASK);
        }}
        return friction_pair_from_metric(metric, r0, r1, old0, old1, radius, relaxation);
    }};
"""
    update = f"""
            if (row == consumed_tangent) continue;
            const int triplet_slot = contact_metric_slot(row);
            const bool fuse = triplet_slot >= 0 && global_iteration >= friction_start_iteration
                && isfinite(diagonal[row]) && diagonal[row] > 0.0f
                && support.row_class.data[offset + row] == support.row_class.data[offset + row + 1]
                && ((selected_words[(row + 1) / 32] >> ((row + 1) & 31)) & 1u) != 0u;
            if (fuse) {{
                const int tangent = row + 1, sibling = row + 2;
                int indices[{Q}];
                float velocities[{Q}], normal_responses[{Q}];
                float tangent_responses[{Q}], sibling_responses[{Q}];
                float normal_dot = 0.0f;
                #pragma unroll
                for (int q = 0; q < {Q}; ++q) {{
                    const int entry = lane + q * 32;
                    const int index = entry < {S}
                        ? support.class_dof.data[support.row_class.data[offset + row] * {S} + entry] : -1;
                    indices[q] = index;
                    normal_responses[q] = index >= 0 ? row_factor.data[row_base + row * {S} + entry] : 0.0f;
                    velocities[q] = index >= 0 ? delta[index] : 0.0f;
                    if (index >= 0) normal_dot += normal_responses[q] * velocities[q];
                }}
                for (int shift = 16; shift > 0; shift >>= 1)
                    normal_dot += __shfl_down_sync(MASK, normal_dot, shift, 32);
                const float normal_residual = __shfl_sync(MASK, normal_dot, 0, 32) + rhs[row];
                const float old_normal = impulse[row];
                const float new_normal = fmaxf(old_normal - omega * normal_residual / diagonal[row], 0.0f);
                const float normal_change = new_normal - old_normal;
                const float radius = fmaxf(world_row_mu.data[offset + tangent] * new_normal, 0.0f);
                const float old_tangent = impulse[tangent], old_sibling = impulse[sibling];
                float tangent_change = 0.0f, sibling_change = 0.0f;
                if (lane == 0) impulse[row] = new_normal;
                #pragma unroll
                for (int q = 0; q < {Q}; ++q)
                    if (indices[q] >= 0 && normal_change != 0.0f)
                        velocities[q] += normal_responses[q] * normal_change;
                if (!(radius == 0.0f && old_tangent == 0.0f && old_sibling == 0.0f)) {{
                    const bool ready = metric_map[triplet_slot] >= 0;
                    float dot = 0.0f, sibling_dot = 0.0f, cross = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < {Q}; ++q) {{
                        const int entry = lane + q * 32;
                        const int index = indices[q];
                        const float value = index >= 0 ? row_factor.data[row_base + tangent * {S} + entry] : 0.0f;
                        const float sibling_value = index >= 0
                            ? row_factor.data[row_base + sibling * {S} + entry] : 0.0f;
                        tangent_responses[q] = value;
                        sibling_responses[q] = sibling_value;
                        if (index >= 0) {{
                            // Use the actual post-normal coordinate, preserving
                            // the original rounding and sibling-then-first order.
                            dot += value * velocities[q];
                            sibling_dot += sibling_value * velocities[q];
                            if (!ready) cross += value * sibling_value;
                        }}
                    }}
                    for (int shift = 16; shift > 0; shift >>= 1) {{
                        dot += __shfl_down_sync(MASK, dot, shift, 32);
                        sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift, 32);
                        if (!ready) cross += __shfl_down_sync(MASK, cross, shift, 32);
                    }}
                    const float residual = __shfl_sync(MASK, dot, 0, 32) + rhs[tangent];
                    float2 pair = make_float2(0.0f, 0.0f);
                    if (radius > 0.0f) {{
                        const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0, 32) + rhs[sibling];
                        pair = friction_pair_cached(triplet_slot, diagonal[tangent], __shfl_sync(MASK, cross, 0, 32),
                            diagonal[sibling], residual, sibling_residual, old_tangent, old_sibling, radius, omega);
                    }}
                    const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                    const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                    const float new_tangent = pair.x * scale, new_sibling = pair.y * scale;
                    tangent_change = new_tangent - old_tangent;
                    sibling_change = new_sibling - old_sibling;
                    if (lane == 0) {{ impulse[tangent] = new_tangent; impulse[sibling] = new_sibling; }}
                    #pragma unroll
                    for (int q = 0; q < {Q}; ++q) {{
                        if (indices[q] >= 0) {{
                            if (sibling_change != 0.0f) velocities[q] += sibling_responses[q] * sibling_change;
                            if (tangent_change != 0.0f) velocities[q] += tangent_responses[q] * tangent_change;
                        }}
                    }}
                }}
                if (normal_change != 0.0f || tangent_change != 0.0f || sibling_change != 0.0f) {{
                    if (!(fabsf(normal_change) <= change_tolerance && fabsf(tangent_change) <= change_tolerance
                          && fabsf(sibling_change) <= change_tolerance)) changed = 1;
                    #pragma unroll
                    for (int q = 0; q < {Q}; ++q)
                        if (indices[q] >= 0) delta[indices[q]] = velocities[q];
                }}
                consumed_tangent = tangent;
                __syncwarp(MASK);
                continue;
            }}
"""
    return setup, update

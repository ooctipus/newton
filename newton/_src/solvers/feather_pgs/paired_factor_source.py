# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share the paired GS recurrence through explicit representation entry and exit."""

from .friction import FRICTION_PAIR_CUDA
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)

_KINETIC_ENTRY = r"""
    const int primary_lane=lane-primary_offset,secondary_lane=lane-secondary_offset;
    const int physical_dof=(primary_lane>=0&&primary_lane<23)?primary_lane:
        ((secondary_lane>=0&&secondary_lane<6)?23+secondary_lane:-1);
    const int global_dof=physical_dof>=0?plan.dof_ids.data[world*35+physical_dof]:-1;
    float factor_velocity=0.0f;
"""


_KINETIC_EXIT = r"""
    float physical_output=0.0f;
    for(int k=0;k<23;++k){const float du=__shfl_sync(MASK,factor_velocity,primary_offset+k);
        if(primary_lane>=0&&primary_lane<23)physical_output+=t(k,primary_lane)*du;}
    for(int k=0;k<6;++k){const float du=__shfl_sync(MASK,factor_velocity,secondary_offset+k);
        if(secondary_lane>=0&&secondary_lane<=k&&secondary_lane<6)
            physical_output+=held.inverse6.data[secondary_group*36+k*6+secondary_lane]*du;}
    const bool finite=global_dof<0||isfinite(physical_output+state.v_hat.data[global_dof]);
    if(!__all_sync(MASK,finite)){if(lane==0)solve.status.data[world]=2;return;}
    if(global_dof>=0&&(!split_component||(primary_lane>=0&&primary_lane<23)))
        v_out.data[global_dof]=state.v_hat.data[global_dof]+physical_output;
    if(lane==0)solve.status.data[world]=0;
"""


_KINETIC_ALIASES = r"""
    auto world_constraint_count=state.dense_count;
    auto dense_phase_bounds=rows.phase_bounds;
    auto world_impulses=rows.impulses;
    auto rhs_bias=rows.r0;
    auto world_diag=rows.diag;
    auto world_row_type=rows.row_type;
    auto world_row_parent=rows.row_parent;
    auto world_row_mu=rows.row_mu;
    auto factor_rows=rows.response;
    auto primary_group_by_world=plan.primary_group;
    auto secondary_group_by_world=plan.secondary_group;
    auto primary_offset_by_world=state.primary_offset;
    auto secondary_offset_by_world=state.secondary_offset;
    auto v_out=solve.v_out;
    const int iterations=solve.iterations,iteration_offset=solve.iteration_offset,friction_start_iteration=solve.friction_start_iteration;
    const float omega=solve.omega;
"""


def _physical_entry(D, P, S, independent_components):
    velocity_condition = "global_dof >= 0"
    if independent_components:
        velocity_condition += f" && (!split_component || (lane >= primary_offset && lane < primary_offset + {P}))"
    return f"""    const int global_dof = lane < {D} ? world_dof_indices.data[dof_map_base + lane] : -1;
    const float physical_velocity = {velocity_condition} ? v_out.data[global_dof] : 0.0f;
    const int primary_lane = lane - primary_offset;
    const int secondary_lane = lane - secondary_offset;
    float factor_velocity = 0.0f;
#pragma unroll
    for (int k = 0; k < {P}; ++k) {{
        const float xk = __shfl_sync(MASK, physical_velocity, primary_offset + k);
        if (primary_lane >= 0 && primary_lane <= k && primary_lane < {P})
            factor_velocity += primary_L.data[
                primary_matrix_base + k * {P} + primary_lane] * xk;
    }}
#pragma unroll
    for (int k = 0; k < {S}; ++k) {{
        const float xk = __shfl_sync(MASK, physical_velocity, secondary_offset + k);
        if (secondary_lane >= 0 && secondary_lane <= k && secondary_lane < {S})
            factor_velocity += secondary_L.data[
                secondary_matrix_base + k * {S} + secondary_lane] * xk;
    }}
"""


def _physical_exit(P, S, independent_components):
    output_condition = "global_dof >= 0"
    if independent_components:
        output_condition += f" && (!split_component || (primary_lane >= 0 && primary_lane < {P}))"
    return f"""    float physical_output = 0.0f;
#pragma unroll
    for (int k = 0; k < {P}; ++k) {{
        const float uk = __shfl_sync(MASK, factor_velocity, primary_offset + k);
        if (primary_lane >= 0 && primary_lane <= k && primary_lane < {P})
            physical_output += primary_Linv.data[
                primary_matrix_base + k * {P} + primary_lane] * uk;
    }}
#pragma unroll
    for (int k = 0; k < {S}; ++k) {{
        const float uk = __shfl_sync(MASK, factor_velocity, secondary_offset + k);
        if (secondary_lane >= 0 && secondary_lane <= k && secondary_lane < {S})
            physical_output += secondary_Linv.data[
                secondary_matrix_base + k * {S} + secondary_lane] * uk;
    }}
    if ({output_condition}) v_out.data[global_dof] = physical_output;
"""


def get_source(M, D, P, S, *, contact_triples=False, independent_components=False, kinetic=False):
    """Build physical or held-coordinate source without recovering a sibling definition."""
    if M <= 0 or P <= 0 or S <= 0 or P + S != D or D > 32:
        raise ValueError("paired factor source requires positive capacity and two components spanning at most 32 DOFs")
    if independent_components and (P, S) != (23, 6):
        raise ValueError("independent paired source requires the 23+6 pair")
    if kinetic and (M, D, P, S, contact_triples, independent_components) != (192, 29, 23, 6, True, False):
        raise ValueError("held kinetic source requires the 192-row 23+6 contact-triple contract")
    W = 2
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    joint_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    joint_velocity_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    fences = [""] * 6
    if kinetic:
        row_guard = "    if (m < 0 || m > 192) { if(lane==0) solve.status.data[world]=1; return; }"
        entry, exit_code = _KINETIC_ENTRY, _KINETIC_EXIT
        for index, indentation in enumerate(
            (
                "                ",
                "                    ",
                "                    ",
                "                ",
                "                ",
                "            ",
            )
        ):
            fences[index] = f"\n{indentation}__syncwarp(MASK); /* IMPULSE_READ_COMPLETE_{index} */"
    else:
        row_guard = f"    if (m > {M}) m = {M};\n"
        row_guard += "    if (m == 0 || world_mf_constraint_count.data[world] "
        row_guard += (
            "> 0) return;\n    const bool split_component = world_mf_constraint_count.data[world] < 0;"
            if independent_components
            else "!= 0) return;"
        )
        entry = _physical_entry(D, P, S, independent_components)
        exit_code = _physical_exit(P, S, independent_components)
    body = f"""    __shared__ float s_lam_storage[{W * M}];
    __shared__ float s_rhs_storage[{W * M}];
    __shared__ float s_diag_storage[{W * M}];
    __shared__ unsigned char s_type_storage[{W * M}];
    __shared__ float s_contact_mu_storage[{W * ((M + 2) // 3)}];
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int world_slot = threadIdx.x >> 5;

    int m = world_constraint_count.data[world];
{row_guard}

    const int off = world * {M};
    const int row_base = off * {D};
    const int dof_map_base = world * {D};
    const int primary_group = primary_group_by_world.data[world];
    const int secondary_group = secondary_group_by_world.data[world];
    const int primary_offset = primary_offset_by_world.data[world];
    const int secondary_offset = secondary_offset_by_world.data[world];
    const int primary_matrix_base = primary_group * {P * P};
    const int secondary_matrix_base = secondary_group * {S * S};
    float* s_lam = &s_lam_storage[world_slot * {M}];
    float* s_rhs = &s_rhs_storage[world_slot * {M}];
    float* s_diag = &s_diag_storage[world_slot * {M}];
    unsigned char* s_type = &s_type_storage[world_slot * {M}];
    float* s_contact_mu = &s_contact_mu_storage[world_slot * {((M + 2) // 3)}];

    int contact_start = dense_phase_bounds.data[world * 2 + 1];
    if (contact_start < 0) contact_start = 0;
    if (contact_start > m) contact_start = m;
    const bool triple_layout = {str(bool(contact_triples)).lower()}
        && dense_phase_bounds.data[world * 2] == contact_start
        && (m - contact_start) % 3 == 0;

    for (int i = lane; i < m; i += 32) {{
        s_lam[i] = world_impulses.data[off + i];
        s_rhs[i] = rhs_bias.data[off + i];
        s_diag[i] = world_diag.data[off + i];
        if (!triple_layout || i < contact_start)
            s_type[i] = static_cast<unsigned char>(world_row_type.data[off + i]);
    }}
    const int contact_count = (m - contact_start) / 3;
    for (int contact = lane; contact < contact_count; contact += 32)
        s_contact_mu[contact] = world_row_mu.data[off + contact_start + contact * 3 + 1];

{entry}    __syncwarp(MASK);

    for (int iter = 0; iter < iterations; ++iter) {{
        const int global_iter = iteration_offset + iter;
        int iteration_changed = 0;

        if (triple_layout) {{
            // The prefix contains equality and unilateral joint-limit rows;
            // contacts then own exact normal/tangent/tangent triples. Consume
            // the trailing structural contract directly instead of decoding
            // parent, sibling, and friction metadata for every contact row.
            for (int i = 0; i < contact_start; ++i) {{
                const float row_factor = lane < {D}
                    ? factor_rows.data[row_base + i * {D} + lane] : 0.0f;
                const float denom = s_diag[i];
                if (denom <= 0.0f) continue;
                float sum = lane < {D} ? row_factor * factor_velocity : 0.0f;
                sum += __shfl_down_sync(MASK, sum, 16);
                sum += __shfl_down_sync(MASK, sum, 8);
                sum += __shfl_down_sync(MASK, sum, 4);
                sum += __shfl_down_sync(MASK, sum, 2);
                sum += __shfl_down_sync(MASK, sum, 1);
                const float old_impulse = s_lam[i];
                float new_impulse = old_impulse
                    + omega * (-( __shfl_sync(MASK, sum, 0) + s_rhs[i]) / denom);
                const int row_type = static_cast<int>(s_type[i]);
                if ((row_type == {contact_type} || row_type == {joint_limit_type}) && new_impulse < 0.0f)
                    new_impulse = 0.0f;
                const float delta = new_impulse - old_impulse;{fences[0]}
                s_lam[i] = new_impulse;
                if (delta != 0.0f) {{
                    iteration_changed = 1;
                    factor_velocity += row_factor * delta;
                }}
                __syncwarp(MASK);
            }}

            for (int contact = 0; contact < contact_count; ++contact) {{
                const int normal = contact_start + contact * 3;
                const int tangent1 = normal + 1;
                const int tangent2 = normal + 2;
                const float normal_factor = lane < {D}
                    ? factor_rows.data[row_base + normal * {D} + lane] : 0.0f;
                float normal_sum = lane < {D} ? normal_factor * factor_velocity : 0.0f;
                normal_sum += __shfl_down_sync(MASK, normal_sum, 16);
                normal_sum += __shfl_down_sync(MASK, normal_sum, 8);
                normal_sum += __shfl_down_sync(MASK, normal_sum, 4);
                normal_sum += __shfl_down_sync(MASK, normal_sum, 2);
                normal_sum += __shfl_down_sync(MASK, normal_sum, 1);
                const float normal_denom = s_diag[normal];
                if (normal_denom > 0.0f) {{
                    const float old_normal = s_lam[normal];
                    float new_normal = old_normal
                        + omega * (-( __shfl_sync(MASK, normal_sum, 0) + s_rhs[normal]) / normal_denom);
                    if (new_normal < 0.0f) new_normal = 0.0f;
                    const float normal_delta = new_normal - old_normal;{fences[1]}
                    s_lam[normal] = new_normal;
                    if (normal_delta != 0.0f) {{
                        iteration_changed = 1;
                        factor_velocity += normal_factor * normal_delta;
                    }}
                }}
                __syncwarp(MASK);

                const float tangent1_factor = lane < {D}
                    ? factor_rows.data[row_base + tangent1 * {D} + lane] : 0.0f;
                const float tangent2_factor = lane < {D}
                    ? factor_rows.data[row_base + tangent2 * {D} + lane] : 0.0f;
                // Patch normal load: normal rows link the region's next normal (circular list, -1 ends a
                // point contact). Solve both tangents together on the friction disk at that load.
                float lambda_n = s_lam[normal];
                for (int patch_row = world_row_parent.data[off + normal];
                     patch_row >= 0 && patch_row != normal;
                     patch_row = world_row_parent.data[off + patch_row])
                    lambda_n += s_lam[patch_row];
                const float radius = fmaxf(s_contact_mu[contact] * lambda_n, 0.0f);
                const float old_tangent1 = s_lam[tangent1];
                const float old_tangent2 = s_lam[tangent2];
                if (radius <= 0.0f && old_tangent1 == 0.0f && old_tangent2 == 0.0f) {{
                    __syncwarp(MASK);
                    continue;
                }}
                if (global_iter < friction_start_iteration) {{{fences[2]}
                    s_lam[tangent1] = 0.0f;
                    s_lam[tangent2] = 0.0f;
                    __syncwarp(MASK);
                    continue;
                }}
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    float tangent1_sum = lane < {D} ? tangent1_factor * factor_velocity : 0.0f;
                    float tangent2_sum = lane < {D} ? tangent2_factor * factor_velocity : 0.0f;
                    float cross_sum = lane < {D} ? tangent1_factor * tangent2_factor : 0.0f;
                    for (int offset = 16; offset > 0; offset >>= 1) {{
                        tangent1_sum += __shfl_down_sync(MASK, tangent1_sum, offset);
                        tangent2_sum += __shfl_down_sync(MASK, tangent2_sum, offset);
                        cross_sum += __shfl_down_sync(MASK, cross_sum, offset);
                    }}
                    const float tangent1_residual = __shfl_sync(MASK, tangent1_sum, 0) + s_rhs[tangent1];
                    const float tangent2_residual = __shfl_sync(MASK, tangent2_sum, 0) + s_rhs[tangent2];
                    const float cross = __shfl_sync(MASK, cross_sum, 0);
                    pair = friction_pair_candidate(s_diag[tangent1], cross, s_diag[tangent2],
                        tangent1_residual, tangent2_residual, old_tangent1, old_tangent2, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                const float new_tangent1 = pair.x * scale;
                const float new_tangent2 = pair.y * scale;
                const float tangent1_delta = new_tangent1 - old_tangent1;
                const float tangent2_delta = new_tangent2 - old_tangent2;{fences[3]}
                s_lam[tangent1] = new_tangent1;
                s_lam[tangent2] = new_tangent2;
                if (tangent2_delta != 0.0f) {{
                    iteration_changed = 1;
                    factor_velocity += tangent2_factor * tangent2_delta;
                }}
                if (tangent1_delta != 0.0f) {{
                    iteration_changed = 1;
                    factor_velocity += tangent1_factor * tangent1_delta;
                }}
                __syncwarp(MASK);
            }}
        }} else {{
        float prefetched_factor = lane < {D} ? factor_rows.data[row_base + lane] : 0.0f;

        for (int i = 0; i < m; ++i) {{
            const float row_factor = prefetched_factor;
            if (i + 1 < m && lane < {D})
                prefetched_factor = factor_rows.data[row_base + (i + 1) * {D} + lane];

            const int row_type = static_cast<int>(s_type[i]);
            if (row_type == {friction_type} && global_iter < friction_start_iteration) {{
                s_lam[i] = 0.0f;
                __syncwarp(MASK);
                continue;
            }}

            int parent = -1;
            int sibling = -1;
            float sibling_factor = 0.0f;
            if (row_type == {friction_type}) {{
                parent = world_row_parent.data[off + i];
                if (i != parent + 1) {{
                    // The first tangent row solved both tangents; the second visit is a no-op.
                    __syncwarp(MASK);
                    continue;
                }}
                sibling = parent + 2;
                sibling_factor = lane < {D} ? factor_rows.data[row_base + sibling * {D} + lane] : 0.0f;
            }}

            const float denom = s_diag[i];
            if (denom <= 0.0f && row_type != {friction_type}) continue;
            float sum = lane < {D} ? row_factor * factor_velocity : 0.0f;
            sum += __shfl_down_sync(MASK, sum, 16);
            sum += __shfl_down_sync(MASK, sum, 8);
            sum += __shfl_down_sync(MASK, sum, 4);
            sum += __shfl_down_sync(MASK, sum, 2);
            sum += __shfl_down_sync(MASK, sum, 1);
            const float jv = __shfl_sync(MASK, sum, 0);

            const float residual = jv + s_rhs[i];
            const float raw_delta = denom > 0.0f ? -residual / denom : 0.0f;
            const float old_impulse = s_lam[i];
            float new_impulse = old_impulse + omega * raw_delta;
            float delta_impulse = 0.0f;
            float sibling_delta = 0.0f;
            if (row_type == {contact_type} || row_type == {joint_limit_type}) {{
                if (new_impulse < 0.0f) new_impulse = 0.0f;
                delta_impulse = new_impulse - old_impulse;
            }} else if (row_type == {joint_velocity_limit_type}) {{
                if (residual < 0.0f) {{
                    delta_impulse = raw_delta;
                    new_impulse = raw_delta;
                }} else {{
                    new_impulse = 0.0f;
                }}
            }} else if (row_type == {friction_type}) {{
                // Paired tangent solve at the patch normal load, matching the general owner.
                float lambda_n = s_lam[parent];
                for (int patch_row = world_row_parent.data[off + parent];
                     patch_row >= 0 && patch_row != parent;
                     patch_row = world_row_parent.data[off + patch_row])
                    lambda_n += s_lam[patch_row];
                const float radius = fmaxf(world_row_mu.data[off + i] * lambda_n, 0.0f);
                const float sibling_impulse = s_lam[sibling];
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    float sibling_sum = lane < {D} ? sibling_factor * factor_velocity : 0.0f;
                    float cross_sum = lane < {D} ? row_factor * sibling_factor : 0.0f;
                    for (int offset = 16; offset > 0; offset >>= 1) {{
                        sibling_sum += __shfl_down_sync(MASK, sibling_sum, offset);
                        cross_sum += __shfl_down_sync(MASK, cross_sum, offset);
                    }}
                    const float sibling_residual = __shfl_sync(MASK, sibling_sum, 0) + s_rhs[sibling];
                    const float cross = __shfl_sync(MASK, cross_sum, 0);
                    pair = friction_pair_candidate(denom, cross, s_diag[sibling],
                        residual, sibling_residual, old_impulse, sibling_impulse, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                const float new_sibling_impulse = pair.y * scale;
                sibling_delta = new_sibling_impulse - sibling_impulse;{fences[4]}
                s_lam[sibling] = new_sibling_impulse;
                delta_impulse = new_impulse - old_impulse;
            }} else {{
                delta_impulse = new_impulse - old_impulse;
            }}{fences[5]}
            s_lam[i] = new_impulse;

            if (sibling_delta != 0.0f) {{
                iteration_changed = 1;
                factor_velocity += sibling_factor * sibling_delta;
            }}
            if (delta_impulse != 0.0f) {{
                iteration_changed = 1;
                factor_velocity += row_factor * delta_impulse;
            }}
            __syncwarp(MASK);
        }}
        }}

        const unsigned changed = __ballot_sync(MASK, iteration_changed != 0);
        if (global_iter >= friction_start_iteration && changed == 0u) break;
    }}

{exit_code}    for (int i = lane; i < m; i += 32) world_impulses.data[off + i] = s_lam[i];
"""
    if kinetic:
        return FRICTION_PAIR_CUDA + _KINETIC_ALIASES + body
    return "\n#if defined(__CUDA_ARCH__)\n" + FRICTION_PAIR_CUDA + "\n" + body + "#endif\n"

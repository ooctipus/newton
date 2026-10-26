# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share the paired GS recurrence through explicit representation entry and exit."""

from .device_preludes import RING, cuda_native, lane_group
from .friction import FRICTION_PAIR
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from .solve_policy import EXACT_POLICY, SolvePolicy, sweep_exit, vote


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
        const float xk = READ(physical_velocity, primary_offset + k);
        if (primary_lane >= 0 && primary_lane <= k && primary_lane < {P})
            factor_velocity += primary_L.data[
                primary_matrix_base + k * {P} + primary_lane] * xk;
    }}
#pragma unroll
    for (int k = 0; k < {S}; ++k) {{
        const float xk = READ(physical_velocity, secondary_offset + k);
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
        const float uk = READ(factor_velocity, primary_offset + k);
        if (primary_lane >= 0 && primary_lane <= k && primary_lane < {P})
            physical_output += primary_Linv.data[
                primary_matrix_base + k * {P} + primary_lane] * uk;
    }}
#pragma unroll
    for (int k = 0; k < {S}; ++k) {{
        const float uk = READ(factor_velocity, secondary_offset + k);
        if (secondary_lane >= 0 && secondary_lane <= k && secondary_lane < {S})
            physical_output += secondary_Linv.data[
                secondary_matrix_base + k * {S} + secondary_lane] * uk;
    }}
    if ({output_condition}) v_out.data[global_dof] = physical_output;
"""


def _unit_walk(D, factor_rows, contact_type, friction_type, joint_limit_type, joint_velocity_limit_type, votes):
    """Sweep every row with the general branch's law, solving each unit's tangent pair once.

    A unit is a row and, when the next row is a tangent row naming it as its
    parent, that tangent pair. Patch contacts are units of one row, or of three
    when they own an anchor; the general branch visits a pair's second tangent
    row only to skip it. A tangent row outside a unit keeps that branch's law:
    zeroed while friction is delayed, and otherwise left alone. ``votes`` holds
    the change votes of the row and the two tangents (:func:`solve_policy.vote`).
    """
    row_vote, tangent1_vote, tangent2_vote = votes
    return f"""        for (int i = 0; i < m;) {{
            const int row_type = static_cast<int>(s_type[i]);
            if (row_type == {friction_type}) {{
                if (global_iter < friction_start_iteration) s_lam[i] = 0.0f;
                SYNC();
                ++i;
                continue;
            }}
            const int tangent1 = i + 1;
            const int tangent2 = i + 2;
            const bool owns_tangents = tangent2 < m
                && static_cast<int>(s_type[tangent1]) == {friction_type}
                && world_row_parent.data[off + tangent1] == i;
            const float row_factor = lane < {D}
                ? {factor_rows}.data[row_base + i * {D} + lane] : 0.0f;
            float tangent1_factor = 0.0f;
            float tangent2_factor = 0.0f;
            if (owns_tangents && lane < {D}) {{
                tangent1_factor = {factor_rows}.data[row_base + tangent1 * {D} + lane];
                tangent2_factor = {factor_rows}.data[row_base + tangent2 * {D} + lane];
            }}
            const float denom = s_diag[i];
            if (denom > 0.0f) {{
                float sum = lane < {D} ? row_factor * factor_velocity : 0.0f;
                SUM(sum);
                const float residual = sum + s_rhs[i];
                const float raw_delta = -residual / denom;
                const float old_impulse = s_lam[i];
                float new_impulse = old_impulse + omega * raw_delta;
                float delta_impulse = 0.0f;
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
                }} else {{
                    delta_impulse = new_impulse - old_impulse;
                }}
                s_lam[i] = new_impulse;
                if (delta_impulse != 0.0f) {{
{row_vote}
                    factor_velocity += row_factor * delta_impulse;
                }}
                SYNC();
            }}
            if (!owns_tangents) {{
                ++i;
                continue;
            }}
            if (global_iter < friction_start_iteration) {{
                s_lam[tangent1] = 0.0f;
                s_lam[tangent2] = 0.0f;
                SYNC();
                i += 3;
                continue;
            }}
            // Solve both tangents on the friction disk of the patch normal load: normal rows link
            // the region's next normal (circular list, -1 ends a point contact).
            const float lambda_n = RING_LOAD(s_lam[i], i, m, ring_parent, ring_impulse);
            const float radius = fmaxf(world_row_mu.data[off + tangent1] * lambda_n, 0.0f);
            const float old_tangent1 = s_lam[tangent1];
            const float old_tangent2 = s_lam[tangent2];
            if (radius <= 0.0f && old_tangent1 == 0.0f && old_tangent2 == 0.0f) {{
                // The pair solve would write zeros and change no velocity.
                s_lam[tangent1] = 0.0f;
                s_lam[tangent2] = 0.0f;
                SYNC();
                i += 3;
                continue;
            }}
            float2 pair = make_float2(0.0f, 0.0f);
            if (radius > 0.0f) {{
                float tangent1_sum = lane < {D} ? tangent1_factor * factor_velocity : 0.0f;
                float tangent2_sum = lane < {D} ? tangent2_factor * factor_velocity : 0.0f;
                float cross_sum = lane < {D} ? tangent1_factor * tangent2_factor : 0.0f;
                SUM(tangent1_sum, tangent2_sum, cross_sum);
                const float tangent1_residual = tangent1_sum + s_rhs[tangent1];
                const float tangent2_residual = tangent2_sum + s_rhs[tangent2];
                const float cross = cross_sum;
                pair = friction_pair_candidate(s_diag[tangent1], cross, s_diag[tangent2],
                    tangent1_residual, tangent2_residual, old_tangent1, old_tangent2, radius, omega);
            }}
            const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
            const float scale = magnitude > radius ? radius / magnitude : 1.0f;
            const float new_tangent1 = pair.x * scale;
            const float new_tangent2 = pair.y * scale;
            const float tangent1_delta = new_tangent1 - old_tangent1;
            const float tangent2_delta = new_tangent2 - old_tangent2;
            s_lam[tangent1] = new_tangent1;
            s_lam[tangent2] = new_tangent2;
            if (tangent2_delta != 0.0f) {{
{tangent2_vote}
                factor_velocity += tangent2_factor * tangent2_delta;
            }}
            if (tangent1_delta != 0.0f) {{
{tangent1_vote}
                factor_velocity += tangent1_factor * tangent1_delta;
            }}
            SYNC();
            i += 3;
        }}
"""


def get_source(
    M,
    D,
    P,
    S,
    *,
    contact_triples=False,
    independent_components=False,
    contact_units=False,
    solve_policy: SolvePolicy = EXACT_POLICY,
):
    """Build the physical-coordinate paired recurrence source.

    With ``independent_components`` a world whose signed selector is negative
    sweeps only its articulation's rows and coordinates; the free body's rows
    run concurrently in the matrix-free kernel.

    ``contact_triples`` consumes point contacts as normal/tangent/tangent
    triples when the row layout proves it. ``contact_units`` walks units
    instead: each row with the general branch's law, and each tangent pair once
    from the row it names as parent. Patch contacts are units of one row, or
    three when they own an anchor, so the walk visits the general branch's
    rows in its order and with its arithmetic.

    ``solve_policy`` compiles the sweep's stopping rule and readout.
    """
    if M <= 0 or P <= 0 or S <= 0 or P + S != D or D > 32:
        raise ValueError("paired factor source requires positive capacity and two components spanning at most 32 DOFs")
    if contact_triples and contact_units:
        raise ValueError("contact triples and contact units are exclusive paired factor schedules")
    W = 2
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    joint_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    joint_velocity_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    fences = [""] * 6
    row_guard = f"    if (m > {M}) m = {M};\n"
    row_guard += "    if (m == 0 || world_mf_constraint_count.data[world] "
    row_guard += (
        "> 0) return;\n    const bool split_component = world_mf_constraint_count.data[world] < 0;"
        if independent_components
        else "!= 0) return;"
    )
    entry = _physical_entry(D, P, S, independent_components)
    exit_code = _physical_exit(P, S, independent_components)
    # The unit walk reads each unit's coefficient from its tangent row, so it stores no per-contact mu.
    contact_mu_storage = "" if contact_units else f"    __shared__ float s_contact_mu_storage[{W * ((M + 2) // 3)}];\n"
    contact_mu_view = (
        "" if contact_units else f"    float* s_contact_mu = &s_contact_mu_storage[world_slot * {((M + 2) // 3)}];\n"
    )
    storage = f"""    __shared__ float s_lam_storage[{W * M}];
    __shared__ float s_rhs_storage[{W * M}];
    __shared__ float s_diag_storage[{W * M}];
    __shared__ unsigned char s_type_storage[{W * M}];
{contact_mu_storage}"""
    views = f"""    float* s_lam = &s_lam_storage[world_slot * {M}];
    float* s_rhs = &s_rhs_storage[world_slot * {M}];
    float* s_diag = &s_diag_storage[world_slot * {M}];
    unsigned char* s_type = &s_type_storage[world_slot * {M}];
{contact_mu_view}
"""
    world_slot = "    const int world_slot = SUB;\n"
    primary_group_by_world = "primary_group_by_world"
    secondary_group_by_world = "secondary_group_by_world"
    primary_offset_by_world = "primary_offset_by_world"
    secondary_offset_by_world = "secondary_offset_by_world"
    factor_rows = "factor_rows"
    rhs_bias = "rhs_bias"

    def product(factor, row):
        return f"{factor} * factor_velocity"

    def change_vote(indent, change, scale, kind):
        return vote(
            solve_policy,
            " " * indent + "iteration_changed = 1;",
            flag="iteration_changed",
            terms=((change, scale, str(kind)),),
        )

    friction = friction_type
    tangent_votes = [change_vote(16, f"tangent{k}_delta", f"s_diag[tangent{k}]", friction) for k in (1, 2)]
    triple_tangent1_vote, triple_tangent2_vote = (
        change_vote(20, f"tangent{k}_delta", f"s_diag[tangent{k}]", friction) for k in (1, 2)
    )
    prefix_vote = change_vote(20, "delta", "denom", "row_type")
    normal_vote = change_vote(24, "normal_delta", "normal_denom", contact_type)
    sibling_vote = change_vote(16, "sibling_delta", "s_diag[sibling]", friction)
    general_vote = change_vote(16, "delta_impulse", "denom", "row_type")
    if contact_units:
        loads = f"""    for (int i = lane; i < m; i += 32) {{
        s_lam[i] = world_impulses.data[off + i];
        s_rhs[i] = {rhs_bias}.data[off + i];
        s_diag[i] = world_diag.data[off + i];
        s_type[i] = static_cast<unsigned char>(world_row_type.data[off + i]);
    }}

"""
        votes = (change_vote(20, "delta_impulse", "denom", "row_type"), *tangent_votes)
        sweep = _unit_walk(
            D, factor_rows, contact_type, friction_type, joint_limit_type, joint_velocity_limit_type, votes
        )
    else:
        loads = f"""    int contact_start = dense_phase_bounds.data[world * 2 + 1];
    if (contact_start < 0) contact_start = 0;
    if (contact_start > m) contact_start = m;
    const bool triple_layout = {str(bool(contact_triples)).lower()}
        && dense_phase_bounds.data[world * 2] == contact_start
        && (m - contact_start) % 3 == 0;

    for (int i = lane; i < m; i += 32) {{
        s_lam[i] = world_impulses.data[off + i];
        s_rhs[i] = {rhs_bias}.data[off + i];
        s_diag[i] = world_diag.data[off + i];
        if (!triple_layout || i < contact_start)
            s_type[i] = static_cast<unsigned char>(world_row_type.data[off + i]);
    }}
    const int contact_count = (m - contact_start) / 3;
    for (int contact = lane; contact < contact_count; contact += 32)
        s_contact_mu[contact] = world_row_mu.data[off + contact_start + contact * 3 + 1];

"""
        sweep = f"""        if (triple_layout) {{
            // The prefix contains equality and unilateral joint-limit rows;
            // contacts then own exact normal/tangent/tangent triples. Consume
            // the trailing structural contract directly instead of decoding
            // parent, sibling, and friction metadata for every contact row.
            for (int i = 0; i < contact_start; ++i) {{
                const float row_factor = lane < {D}
                    ? {factor_rows}.data[row_base + i * {D} + lane] : 0.0f;
                const float denom = s_diag[i];
                if (denom <= 0.0f) continue;
                float sum = lane < {D} ? {product("row_factor", "i")} : 0.0f;
                SUM(sum);
                const float old_impulse = s_lam[i];
                float new_impulse = old_impulse
                    + omega * (-( sum + s_rhs[i]) / denom);
                const int row_type = static_cast<int>(s_type[i]);
                if ((row_type == {contact_type} || row_type == {joint_limit_type}) && new_impulse < 0.0f)
                    new_impulse = 0.0f;
                const float delta = new_impulse - old_impulse;{fences[0]}
                s_lam[i] = new_impulse;
                if (delta != 0.0f) {{
{prefix_vote}
                    factor_velocity += row_factor * delta;
                }}
                SYNC();
            }}

            for (int contact = 0; contact < contact_count; ++contact) {{
                const int normal = contact_start + contact * 3;
                const int tangent1 = normal + 1;
                const int tangent2 = normal + 2;
                const float normal_factor = lane < {D}
                    ? {factor_rows}.data[row_base + normal * {D} + lane] : 0.0f;
                float normal_sum = lane < {D} ? {product("normal_factor", "normal")} : 0.0f;
                SUM(normal_sum);
                const float normal_denom = s_diag[normal];
                if (normal_denom > 0.0f) {{
                    const float old_normal = s_lam[normal];
                    float new_normal = old_normal
                        + omega * (-( normal_sum + s_rhs[normal]) / normal_denom);
                    if (new_normal < 0.0f) new_normal = 0.0f;
                    const float normal_delta = new_normal - old_normal;{fences[1]}
                    s_lam[normal] = new_normal;
                    if (normal_delta != 0.0f) {{
{normal_vote}
                        factor_velocity += normal_factor * normal_delta;
                    }}
                }}
                SYNC();

                const float tangent1_factor = lane < {D}
                    ? {factor_rows}.data[row_base + tangent1 * {D} + lane] : 0.0f;
                const float tangent2_factor = lane < {D}
                    ? {factor_rows}.data[row_base + tangent2 * {D} + lane] : 0.0f;
                // Patch normal load: normal rows link the region's next normal (circular list, -1 ends a
                // point contact). Solve both tangents together on the friction disk at that load.
                const float lambda_n = RING_LOAD(s_lam[normal], normal, m, ring_parent, ring_impulse);
                const float radius = fmaxf(s_contact_mu[contact] * lambda_n, 0.0f);
                const float old_tangent1 = s_lam[tangent1];
                const float old_tangent2 = s_lam[tangent2];
                if (radius <= 0.0f && old_tangent1 == 0.0f && old_tangent2 == 0.0f) {{
                    SYNC();
                    continue;
                }}
                if (global_iter < friction_start_iteration) {{{fences[2]}
                    s_lam[tangent1] = 0.0f;
                    s_lam[tangent2] = 0.0f;
                    SYNC();
                    continue;
                }}
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    float tangent1_sum = lane < {D} ? {product("tangent1_factor", "tangent1")} : 0.0f;
                    float tangent2_sum = lane < {D} ? {product("tangent2_factor", "tangent2")} : 0.0f;
                    float cross_sum = lane < {D} ? tangent1_factor * tangent2_factor : 0.0f;
                    SUM(tangent1_sum, tangent2_sum, cross_sum);
                    const float tangent1_residual = tangent1_sum + s_rhs[tangent1];
                    const float tangent2_residual = tangent2_sum + s_rhs[tangent2];
                    const float cross = cross_sum;
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
{triple_tangent2_vote}
                    factor_velocity += tangent2_factor * tangent2_delta;
                }}
                if (tangent1_delta != 0.0f) {{
{triple_tangent1_vote}
                    factor_velocity += tangent1_factor * tangent1_delta;
                }}
                SYNC();
            }}
        }} else {{
        float prefetched_factor = lane < {D} ? {factor_rows}.data[row_base + lane] : 0.0f;

        for (int i = 0; i < m; ++i) {{
            const float row_factor = prefetched_factor;
            if (i + 1 < m && lane < {D})
                prefetched_factor = {factor_rows}.data[row_base + (i + 1) * {D} + lane];

            const int row_type = static_cast<int>(s_type[i]);
            if (row_type == {friction_type} && global_iter < friction_start_iteration) {{
                s_lam[i] = 0.0f;
                SYNC();
                continue;
            }}

            int parent = -1;
            int sibling = -1;
            float sibling_factor = 0.0f;
            if (row_type == {friction_type}) {{
                parent = world_row_parent.data[off + i];
                if (i != parent + 1) {{
                    // The first tangent row solved both tangents; the second visit is a no-op.
                    SYNC();
                    continue;
                }}
                sibling = parent + 2;
                sibling_factor = lane < {D} ? {factor_rows}.data[row_base + sibling * {D} + lane] : 0.0f;
            }}

            const float denom = s_diag[i];
            if (denom <= 0.0f && row_type != {friction_type}) continue;
            float sum = lane < {D} ? {product("row_factor", "i")} : 0.0f;
            SUM(sum);
            const float jv = sum;

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
                const float lambda_n = RING_LOAD(s_lam[parent], parent, m, ring_parent, ring_impulse);
                const float radius = fmaxf(world_row_mu.data[off + i] * lambda_n, 0.0f);
                const float sibling_impulse = s_lam[sibling];
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    float sibling_sum = lane < {D} ? sibling_factor * factor_velocity : 0.0f;
                    float cross_sum = lane < {D} ? row_factor * sibling_factor : 0.0f;
                    SUM(sibling_sum, cross_sum);
                    const float sibling_residual = sibling_sum + s_rhs[sibling];
                    const float cross = cross_sum;
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
{sibling_vote}
                factor_velocity += sibling_factor * sibling_delta;
            }}
            if (delta_impulse != 0.0f) {{
{general_vote}
                factor_velocity += row_factor * delta_impulse;
            }}
            SYNC();
        }}
        }}
"""
    sweep_end = sweep_exit(
        solve_policy,
        """        const unsigned changed = BALLOT(iteration_changed != 0);
        if (global_iter >= friction_start_iteration && changed == 0u) break;
""",
        gate="global_iter >= friction_start_iteration",
        quiet="BALLOT(iteration_changed != 0) == 0u",
        iteration="iter",
    )
    body = f"""{storage}    const int lane = LANE;
{world_slot}
    int m = world_constraint_count.data[world];
{row_guard}

    const int off = world * {M};
    const int row_base = off * {D};
    const int dof_map_base = world * {D};
    const int primary_group = {primary_group_by_world}.data[world];
    const int secondary_group = {secondary_group_by_world}.data[world];
    const int primary_offset = {primary_offset_by_world}.data[world];
    const int secondary_offset = {secondary_offset_by_world}.data[world];
    const int primary_matrix_base = primary_group * {P * P};
    const int secondary_matrix_base = secondary_group * {S * S};
{views}    // A contact normal's patch ring links the region's next normal (-1 ends a point contact).
    const auto ring_parent = [&](int r) {{ return world_row_parent.data[off + r]; }};
    const auto ring_impulse = [&](int r) {{ return s_lam[r]; }};
{loads}{entry}    SYNC();

    for (int iter = 0; iter < iterations; ++iter) {{
        const int global_iter = iteration_offset + iter;
        int iteration_changed = 0;

{sweep}
{sweep_end}    }}

{exit_code}    for (int i = lane; i < m; i += 32) world_impulses.data[off + i] = s_lam[i];
"""
    return cuda_native(body + "#endif\n", preludes=(lane_group(32), RING, FRICTION_PAIR, *solve_policy.preludes))

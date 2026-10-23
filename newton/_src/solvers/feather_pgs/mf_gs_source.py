# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construct common matrix-free arithmetic with explicit representation bindings."""

import re
from dataclasses import dataclass

from .contact_torsion import torque_sweep_source
from .device_preludes import cuda_native
from .friction import FRICTION_PAIR
from .kernels import (
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
)
from .row_metadata import (
    _DENSE_META_ROW_TYPE_BITS,
    _DENSE_META_ROW_TYPE_MASK,
    _mf_parent_expression,
    _validate_dense_metadata_encoding,
)
from .solve_policy import EXACT_POLICY, SolvePolicy, SolvePolicyUnsupported, sweep_exit, vote
from .velocity_limit_law import accumulated_update, check_velocity_limit_law


@dataclass(frozen=True)
class GeneratedSource:
    """Complete native body plus the shared held-owner MF phase and arena size."""

    native: str
    mf_phase: str | None
    arena_floats: int


def get_source(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    friction_mode: str = "current",
    *,
    software_pipeline: bool = True,
    shared_metadata: bool = True,
    resident_row_metadata: bool = False,
    stream_mf_impulses: bool = False,
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    fuse_vel_limits: bool = False,
    skip_local_internal_worlds: bool = False,
    local_internal_max_constraints: int = 0,
    contact_torsion: bool = False,
    factor_coordinates: bool = False,
    skip_incremental_rows: int = 0,
    independent_components: bool = False,
    held_policy: tuple[str, str, str] | None = None,
    shared_arena: bool = False,
    cpu_body: str = "",
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> GeneratedSource:
    """Build the shared recurrence with explicit physical or held bindings.

    ``resident_row_metadata`` keeps the packed dense row types and parents
    resident in a streamed-metadata (``shared_metadata=False``) source.

    ``velocity_limit_law`` selects how every visit of a velocity-limit row projects it
    (:mod:`.velocity_limit_law`): the end-of-iteration dense and matrix-free passes and
    the dense rows of a velocity-limit row phase. The stateless law keeps the source
    byte-identical.

    ``solve_policy`` compiles the sweep's stopping rule and readout; a readout
    needs the physical bindings and the per-world phases of the plain kernel.
    """
    if resident_row_metadata and shared_metadata:
        raise ValueError("resident_row_metadata applies to streamed metadata (shared_metadata=False)")
    accumulated_limits = check_velocity_limit_law(velocity_limit_law) == "accumulated"
    if fuse_vel_limits and not has_drive_rows:
        raise ValueError("fuse_vel_limits requires has_drive_rows")
    if stream_mf_impulses and friction_mode != "current":
        raise ValueError("Streamed MF impulses require the warp-uniform current-friction projection")
    if factor_coordinates and (
        friction_mode != "current"
        or has_drive_rows
        or has_dense_velocity_limit_rows
        or fuse_vel_limits
        or skip_local_internal_worlds
    ):
        raise ValueError("factor coordinates require the paired augmented-drive contact solve")
    if independent_components and not factor_coordinates:
        raise ValueError("Independent components require the paired factor-coordinate solve")
    if solve_policy.counts and (held_policy is not None or shared_arena):
        raise ValueError("A solve readout requires the physical bindings")
    if (not solve_policy.exact or solve_policy.traces) and (
        friction_mode != "current" or contact_torsion or held_policy is not None or shared_arena
    ):
        raise SolvePolicyUnsupported(
            "The velocity exit law and trace cover the current-friction physical sweep without torsion"
        )
    _validate_dense_metadata_encoding(max_constraints)
    M_D = max_constraints
    M_MF = mf_max_constraints
    D = max_world_dofs

    if held_policy is not None or shared_arena:
        if (
            M_D,
            M_MF,
            D,
            friction_mode,
            software_pipeline,
            shared_metadata,
            stream_mf_impulses,
            has_drive_rows,
            has_dense_velocity_limit_rows,
            fuse_vel_limits,
            skip_local_internal_worlds,
            local_internal_max_constraints,
            contact_torsion,
            factor_coordinates,
            skip_incremental_rows,
        ) != (
            192,
            64,
            29,
            "current",
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            0,
            False,
            True,
            0,
        ):
            raise ValueError(
                "Held coordinates and shared arena require the fixed 192/64-row 29D paired-component recipe"
            )
    if held_policy is not None and (len(held_policy) != 3 or not all(held_policy) or shared_arena):
        raise ValueError("Held coordinates require complete entry, staging and publication policies")
    if cpu_body and held_policy is None:
        raise ValueError("Only the held-coordinate owner supplies a scalar CPU recurrence")
    held_entry = "\n" + held_policy[0] if held_policy is not None else ""
    # Change votes under the solve policy; the exact law keeps each site's own text.
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    velocity_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    mf_sibling_inverse = "__int_as_float(mf_meta.data[off_meta + sib * 4 + 1])"
    dense_sibling_vote = vote(
        solve_policy,
        "                    if (sib_delta != 0.0f) iteration_changed = 1;",
        flag="iteration_changed",
        terms=(("sib_delta", "s_diag_dense[sib]", str(friction_type)),),
    )
    mf_sibling_vote = vote(
        solve_policy,
        "                    if (sib_delta != 0.0f) iteration_changed = 1;",
        flag="iteration_changed",
        terms=(("sib_delta", f"{mf_sibling_inverse} > 0.0f ? 1.0f / {mf_sibling_inverse} : 0.0f", str(friction_type)),),
    )
    dense_limit_vote = vote(
        solve_policy,
        "                    iteration_changed = 1;",
        flag="iteration_changed",
        terms=(("delta_impulse_vlim", "denom", str(velocity_limit_type)),),
    )
    # The fused clamp corrects a drive row's velocity limit, so the trace counts it as a velocity limit.
    drive_limit_vote = vote(
        solve_policy,
        "                    if (delta_fvl != 0.0f) iteration_changed = 1;",
        flag="iteration_changed",
        terms=(("delta_fvl", "denom_fvl", str(velocity_limit_type)),),
    )
    mf_vote = vote(
        solve_policy,
        "                iteration_changed = 1;",
        flag="iteration_changed",
        terms=(("delta_impulse", "mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f", "mf_rt"),),
    )
    mf_limit_vote = vote(
        solve_policy,
        "                    iteration_changed = 1;",
        flag="iteration_changed",
        terms=(("delta_impulse_mf_vlim", "1.0f / mf_diag", str(velocity_limit_type)),),
    )
    dense_vote = vote(
        solve_policy,
        "                iteration_changed = 1;",
        flag="iteration_changed",
        terms=(("delta_impulse", "denom", "row_type"),),
    )
    dense_rhs = "rows.r0.data[off_dense + i]" if held_policy is not None else "rhs_bias.data[off_dense + i]"
    velocity_load = (
        "s_v[d] = 0.0f;" if held_policy is not None else "s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;"
    )
    velocity_store = (
        held_policy[2]
        if held_policy is not None
        else f"""    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0) v_out.data[global_dof] = s_v[d];
    }}"""
    )
    cpu_section = "#else\n" + cpu_body + "\n" if cpu_body else ""
    arena_offset = 0

    def dense_read(array: str, index: str) -> str:
        if held_policy is not None:
            return ("hybrid_j" if array == "J_world" else "hybrid_y") + f"({index})"
        return f"{array}.data[{index}]"

    def mf_velocity(index: str, base: str) -> str:
        if held_policy is not None:
            return f"* (s_v[{index}] + hybrid_base[{base}])"
        return f"* s_v[{index}]"

    def shared(dtype: str, name: str, count: int, spacing: str) -> str:
        nonlocal arena_offset
        offset = arena_offset
        arena_offset += count
        if shared_arena:
            return f"    {dtype}* {name}=reinterpret_cast<{dtype}*>(arena_address)+{offset};"
        return f"    __shared__ {dtype}{spacing}{name}[{count}];"

    def dense_j_read(base: str, dof: str) -> str:
        return f"""{dense_read("J_world", f"{base} + {dof}")}"""

    factor_fallback_skip = "    if (m_mf == 0) return;" if factor_coordinates else ""
    if held_policy is not None:
        factor_fallback_skip += "\n" + held_policy[1]

    local_internal_skip = (
        f"""
    if (local_solve_owner.data[world] != {PGS_LOCAL_SOLVE_OWNER_GENERAL}) return;"""
        if skip_local_internal_worlds
        else ""
    )
    if skip_incremental_rows > 0:
        # Worlds with only dense rows within the incremental kernel's row budget are
        # solved by _get_pgs_solve_mf_gs_incremental_kernel in the same launch pair.
        local_internal_skip += f"""
    if (m_mf == 0 && m_dense <= {int(skip_incremental_rows)} && defer_dense_response == 0) return;"""
    # How many DOF elements each lane handles (ceil(D/32))
    ELEMS_PER_LANE = (D + 31) // 32

    # --- Code generation for dense phase (D-wide dot/update, software-pipelined) ---

    # Pipeline register declarations
    dense_pipe_decl = "\n".join([f"        float pre_dJ_{k} = 0.0f, pre_dY_{k} = 0.0f;" for k in range(ELEMS_PER_LANE)])

    # Initial prefetch (constraint dense_lo, the first row of this phase's range)
    dense_prefetch_init_parts = [f"            const int init_jy_base = jy_world_base + dense_lo * {D};"]
    for k in range(ELEMS_PER_LANE):
        d_expr = f"lane + {k * 32}" if k > 0 else "lane"
        dense_prefetch_init_parts.append(f"""
            if ({d_expr} < {D}) {{
                pre_dJ_{k} = {dense_j_read("init_jy_base", d_expr)};
                pre_dY_{k} = {dense_read("Y_world", f"init_jy_base + {d_expr}")};
            }}""")
    dense_prefetch_init_code = "\n".join(dense_prefetch_init_parts)

    # Consume prefetched values into cur_ variables
    dense_consume_code = "\n".join(
        [f"                float cur_dJ_{k} = pre_dJ_{k}, cur_dY_{k} = pre_dY_{k};" for k in range(ELEMS_PER_LANE)]
    )

    # Prefetch next constraint (i+1)
    dense_prefetch_next_parts = []
    for k in range(ELEMS_PER_LANE):
        d_expr = f"lane + {k * 32}" if k > 0 else "lane"
        dense_prefetch_next_parts.append(f"""
                if ({d_expr} < {D}) {{
                    pre_dJ_{k} = {dense_j_read("next_jy_base", d_expr)};
                    pre_dY_{k} = {dense_read("Y_world", f"next_jy_base + {d_expr}")};
                }}""")
    dense_prefetch_next_code = "\n".join(dense_prefetch_next_parts)

    # Dense dot product using prefetched J: cur_dJ_k * s_v[d]
    dense_dot_parts = []
    for k in range(ELEMS_PER_LANE):
        d_expr = f"lane + {k * 32}" if k > 0 else "lane"
        dense_dot_parts.append(f"""
            if ({d_expr} < {D}) {{
                my_sum += cur_dJ_{k} * s_v[{d_expr}];
            }}""")
    dense_dot_code = "\n".join(["float my_sum = 0.0f;", *dense_dot_parts])

    # Dense v update using prefetched Y: s_v[d] += cur_dY_k * delta
    dense_v_update_parts = []
    for k in range(ELEMS_PER_LANE):
        d_expr = f"lane + {k * 32}" if k > 0 else "lane"
        dense_v_update_parts.append(f"""
            if ({d_expr} < {D} &&
                (defer_dense_response == 0 ||
                 world_deferred_dof_mask.data[deferred_mask_base + {d_expr}] == 0)) {{
                s_v[{d_expr}] += cur_dY_{k} * delta_impulse;
            }}""")
    dense_v_update_code = "\n".join(dense_v_update_parts)

    if not software_pipeline:
        # Direct-load variant: no double-buffer registers, J/Y read at use.
        dense_pipe_decl = ""
        dense_prefetch_init_code = ";"
        dense_prefetch_next_code = ";"
        dense_consume_code = f"const int cur_jy_base = jy_world_base + i * {D};"
        dot_parts = []
        upd_parts = []
        for k in range(ELEMS_PER_LANE):
            d_expr = f"lane + {k * 32}" if k > 0 else "lane"
            dot_parts.append(f"""
            if ({d_expr} < {D}) {{
                my_sum += {dense_j_read("cur_jy_base", d_expr)} * s_v[{d_expr}];
            }}""")
            upd_parts.append(f"""
            if ({d_expr} < {D} &&
                (defer_dense_response == 0 ||
                 world_deferred_dof_mask.data[deferred_mask_base + {d_expr}] == 0)) {{
                s_v[{d_expr}] += {dense_read("Y_world", f"cur_jy_base + {d_expr}")} * delta_impulse;
            }}""")
        dense_dot_code = "\n".join(["float my_sum = 0.0f;", *dot_parts])
        dense_v_update_code = "\n".join(upd_parts)

    # Dense sibling v update — NOT pipelined (random sib index)
    dense_sib_v_parts = []
    for k in range(ELEMS_PER_LANE):
        d_expr = f"lane + {k * 32}" if k > 0 else "lane"
        dense_sib_v_parts.append(f"""
                if ({d_expr} < {D} &&
                    (defer_dense_response == 0 ||
                     world_deferred_dof_mask.data[deferred_mask_base + {d_expr}] == 0)) {{
                    s_v[{d_expr}] += {dense_read("Y_world", f"sib_row_base + {d_expr}")} * sib_delta;
                }}""")
    dense_sib_v_code = "\n".join(dense_sib_v_parts)
    dense_sibling_dot_code = f"""
                    for (int d = lane; d < {D}; d += 32)
                        sibling_residual += {dense_j_read("sib_row_base", "d")} * s_v[d];"""
    dense_tangent_cross_code = f"""
                    for (int d = lane; d < {D}; d += 32)
                        cross += {dense_j_read(f"jy_world_base + i * {D}", "d")} * {dense_read("Y_world", "sib_row_base + d")};"""

    # Dense friction-row projection.  The dense rows carry articulated
    # contacts such as finger/cube; leaving them on the legacy per-row
    # projection while MF rows use a block Coulomb solve leaves the most
    # important grasp contacts in the old formulation.
    if friction_mode in ("bisection", "bisection_desaxce"):
        dense_friction_block = f"""
                int parent_idx = (s_meta_dense[i] >> __DENSE_META_ROW_TYPE_BITS__) - 1;
                int i_t1 = parent_idx + 1;
                int i_t2 = parent_idx + 2;

                if (i != i_t1) {{
                    // The block solve runs at the first friction row and
                    // pre-writes t2.  Collapse this row's delta to zero.
                    new_impulse = s_lam_dense[i];
                }} else {{
                    int n_row_base = jy_world_base + parent_idx * __D__;
                    int t1_row_base = jy_world_base + i_t1 * __D__;
                    int t2_row_base = jy_world_base + i_t2 * __D__;

                    float target_vel_n = -s_rhs_dense[parent_idx];
                    float mu = s_mu_dense[i];
                    float old_lambda_n = s_lam_dense[parent_idx];
                    float old_lambda_t1 = s_lam_dense[i_t1];
                    float old_lambda_t2 = s_lam_dense[i_t2];

                    float new_lambda_n = old_lambda_n;
                    float new_lambda_t1 = old_lambda_t1;
                    float new_lambda_t2 = old_lambda_t2;
                    float d_n_total = 0.0f;
                    float d_t2_total = 0.0f;

                    if (lane == 0) {{
                        float u_n = 0.0f, u_t1 = 0.0f, u_t2 = 0.0f;
                        float G_nn = 0.0f, G_nt1 = 0.0f, G_nt2 = 0.0f;
                        float G_t1t1 = 0.0f, G_t1t2 = 0.0f, G_t2t2 = 0.0f;

                        for (int k = 0; k < __D__; k++) {{
                            float vk = s_v[k];
                            float Jn = {dense_read("J_world", "n_row_base + k")};
                            float Jt1 = {dense_read("J_world", "t1_row_base + k")};
                            float Jt2 = {dense_read("J_world", "t2_row_base + k")};
                            float Yn = {dense_read("Y_world", "n_row_base + k")};
                            float Yt1 = {dense_read("Y_world", "t1_row_base + k")};
                            float Yt2 = {dense_read("Y_world", "t2_row_base + k")};

                            u_n += Jn * vk;
                            u_t1 += Jt1 * vk;
                            u_t2 += Jt2 * vk;

                            G_nn += Jn * Yn;
                            G_nt1 += Jn * Yt1;
                            G_nt2 += Jn * Yt2;
                            G_t1t1 += Jt1 * Yt1;
                            G_t1t2 += Jt1 * Yt2;
                            G_t2t2 += Jt2 * Yt2;
                        }}

                        // __DENSE_DESAXCE_BIAS__

                        if (G_nn >= 1.0e-20f) {{
                            float u_n_at_zero = u_n + G_nn * (0.0f - old_lambda_n);
                            if (u_n_at_zero >= target_vel_n) {{
                                new_lambda_n = 0.0f;
                                new_lambda_t1 = 0.0f;
                                new_lambda_t2 = 0.0f;
                            }} else {{
                                float lo = 0.0f;
                                float hi = fmaxf(
                                    old_lambda_n * 2.0f,
                                    (target_vel_n - u_n) / G_nn + old_lambda_n);
                                hi = fmaxf(hi, 1.0f);

                                for (int _bi = 0; _bi < 20; _bi++) {{
                                    float mid = 0.5f * (lo + hi);
                                    float d_n = mid - old_lambda_n;
                                    float ut1_eff = u_t1 + G_nt1 * d_n;
                                    float ut2_eff = u_t2 + G_nt2 * d_n;

                                    float det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                    float d_t1 = 0.0f, d_t2 = 0.0f;
                                    if (fabsf(det) > 1.0e-20f) {{
                                        d_t1 = (-ut1_eff * G_t2t2 + ut2_eff * G_t1t2) / det;
                                        d_t2 = ( ut1_eff * G_t1t2 - ut2_eff * G_t1t1) / det;
                                    }}

                                    float trial_t1 = old_lambda_t1 + d_t1;
                                    float trial_t2 = old_lambda_t2 + d_t2;
                                    float flimit = mu * mid;
                                    float tmag = sqrtf(trial_t1 * trial_t1 + trial_t2 * trial_t2);
                                    if (tmag > flimit && tmag > 1.0e-20f) {{
                                        float sc = flimit / tmag;
                                        trial_t1 *= sc;
                                        trial_t2 *= sc;
                                    }}

                                    float d_t1_actual = trial_t1 - old_lambda_t1;
                                    float d_t2_actual = trial_t2 - old_lambda_t2;
                                    float u_n_trial = u_n + G_nn * d_n
                                        + G_nt1 * d_t1_actual
                                        + G_nt2 * d_t2_actual;
                                    if (u_n_trial < target_vel_n) lo = mid;
                                    else hi = mid;
                                }}

                                new_lambda_n = 0.5f * (lo + hi);
                                float d_n_final = new_lambda_n - old_lambda_n;
                                float ut1_f = u_t1 + G_nt1 * d_n_final;
                                float ut2_f = u_t2 + G_nt2 * d_n_final;
                                float det_f = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                float d_t1_f = 0.0f, d_t2_f = 0.0f;
                                if (fabsf(det_f) > 1.0e-20f) {{
                                    d_t1_f = (-ut1_f * G_t2t2 + ut2_f * G_t1t2) / det_f;
                                    d_t2_f = ( ut1_f * G_t1t2 - ut2_f * G_t1t1) / det_f;
                                }}

                                new_lambda_t1 = old_lambda_t1 + d_t1_f;
                                new_lambda_t2 = old_lambda_t2 + d_t2_f;
                                float flimit_f = mu * new_lambda_n;
                                float tmag_f = sqrtf(new_lambda_t1 * new_lambda_t1 + new_lambda_t2 * new_lambda_t2);
                                if (tmag_f > flimit_f && tmag_f > 1.0e-20f) {{
                                    float sc_f = flimit_f / tmag_f;
                                    new_lambda_t1 *= sc_f;
                                    new_lambda_t2 *= sc_f;
                                }}
                            }}

                            d_n_total = new_lambda_n - old_lambda_n;
                            d_t2_total = new_lambda_t2 - old_lambda_t2;
                            s_lam_dense[parent_idx] = new_lambda_n;
                            s_lam_dense[i_t2] = new_lambda_t2;
                        }}
                    }}

                    __syncwarp();
                    new_lambda_t1 = __shfl_sync(MASK, new_lambda_t1, 0);
                    d_n_total = __shfl_sync(MASK, d_n_total, 0);
                    d_t2_total = __shfl_sync(MASK, d_t2_total, 0);
                    if (d_n_total != 0.0f || d_t2_total != 0.0f) iteration_changed = 1;

                    if (d_n_total != 0.0f) {{
                        for (int d = lane; d < __D__; d += 32) {{
                            if (defer_dense_response == 0 ||
                                world_deferred_dof_mask.data[deferred_mask_base + d] == 0) {{
                                s_v[d] += {dense_read("Y_world", "n_row_base + d")} * d_n_total;
                            }}
                        }}
                    }}
                    if (d_t2_total != 0.0f) {{
                        for (int d = lane; d < __D__; d += 32) {{
                            if (defer_dense_response == 0 ||
                                world_deferred_dof_mask.data[deferred_mask_base + d] == 0) {{
                                s_v[d] += {dense_read("Y_world", "t2_row_base + d")} * d_t2_total;
                            }}
                        }}
                    }}
                    __syncwarp();
                    new_impulse = new_lambda_t1;
                }}
"""
        if friction_mode == "bisection_desaxce":
            dense_friction_block = dense_friction_block.replace(
                "// __DENSE_DESAXCE_BIAS__",
                "// Dense contacts use block Coulomb only; dense de Saxce\n"
                "                            // normal amplification is not enabled until validated.",
            )
        else:
            dense_friction_block = dense_friction_block.replace(
                "// __DENSE_DESAXCE_BIAS__",
                '// (no de Saxce bias; friction_mode="bisection".)',
            )
        dense_friction_block = dense_friction_block.replace("__D__", str(D))
    else:
        dense_friction_block = f"""
                int parent_idx = (s_meta_dense[i] >> __DENSE_META_ROW_TYPE_BITS__) - 1;
                if (i != parent_idx + 1) {{
                    new_impulse = old_impulse;
                }} else {{
                    int sib = parent_idx + 2;
                    int sib_row_base = jy_world_base + sib * {D};
                    float lambda_n = s_lam_dense[parent_idx];
                    for (int patch_row = ((s_meta_dense[parent_idx] >> __DENSE_META_ROW_TYPE_BITS__) - 1); patch_row >= 0 && patch_row != parent_idx; patch_row = ((s_meta_dense[patch_row] >> __DENSE_META_ROW_TYPE_BITS__) - 1))
                        lambda_n += s_lam_dense[patch_row];
                    float radius = fmaxf(s_mu_dense[i] * lambda_n, 0.0f);
                    float sibling_residual = 0.0f;
                    {dense_sibling_dot_code}
                    for (int offset = 16; offset > 0; offset >>= 1)
                        sibling_residual += __shfl_down_sync(MASK, sibling_residual, offset);
                    sibling_residual = __shfl_sync(MASK, sibling_residual, 0) + s_rhs_dense[sib];
                    float cross = 0.0f;
                    {dense_tangent_cross_code}
                    for (int offset = 16; offset > 0; offset >>= 1)
                        cross += __shfl_down_sync(MASK, cross, offset);
                    cross = __shfl_sync(MASK, cross, 0);
                    const float sibling_impulse = s_lam_dense[sib];
                    float2 pair = friction_pair_candidate(denom, cross, s_diag_dense[sib],
                        residual, sibling_residual, old_impulse, sibling_impulse, radius, omega);
                    float a = pair.x;
                    float b = pair.y;
                    float mag = sqrtf(a * a + b * b);
                    float scale = mag > radius ? radius / mag : 1.0f;
                    new_impulse = a * scale;
                    float sib_delta = b * scale - sibling_impulse;
                    __syncwarp();
                    if (lane == 0) s_lam_dense[sib] = b * scale;
                    __syncwarp();
{dense_sibling_vote}
                    {dense_sib_v_code}
                }}
"""

    dense_friction_block = dense_friction_block.replace("__DENSE_META_ROW_TYPE_BITS__", str(_DENSE_META_ROW_TYPE_BITS))
    # Both modes using the scalar dense disk law share this ownership rule;
    # coulomb_newton differs only in its MF projector below. The existing
    # end-of-row barrier publishes each scalar result to the next dense row.
    dense_scalar_owner = friction_mode in ("current", "coulomb_newton")
    dense_old_impulse = (
        "__shfl_sync(MASK, lane == 0 ? s_lam_dense[i] : 0.0f, 0)" if dense_scalar_owner else "s_lam_dense[i]"
    )
    dense_impulse_store = "if (lane == 0) " if dense_scalar_owner else ""

    # --- MF friction-row projection -------------------------------------
    # ``friction_mode="current"`` solves the tangent block on its friction
    # disk at fixed normal load (matches :func:`friction_step_current`).
    # ``friction_mode="bisection"`` runs the RAISim bisection on λ_n
    # (matches :func:`friction_step_bisection`).
    # ``friction_mode="bisection_desaxce"`` runs the same bisection
    # with the de Saxce max-dissipation bias (``μ · ‖c_T‖`` added to
    # the normal target velocity) — FPGS Friction Modes 6/13.
    # ``friction_mode="coulomb_newton"`` runs Gilles Daviet's scalar
    # bracketed-Newton on alpha (matches
    # :func:`friction_step_coulomb_newton`) — FPGS Friction Modes
    # 7/13.  We inject one of these CUDA blocks into the MF phase;
    # the outer row loop handles the standard ``delta_impulse``
    # accounting for the first-friction row (and t2's delta
    # collapses to zero because ``s_lam_mf[i_t2]`` is pre-written
    # by the per-contact solve).
    if friction_mode in ("bisection", "bisection_desaxce"):
        mf_friction_block = """
                // __BISECTION_LABEL__
                int mf_par = packed_tp >> 16;
                int i_t1 = mf_par + 1;
                int i_t2 = mf_par + 2;

                if (i != i_t1) {
                    // Second friction row of the triple: the bisection
                    // that ran at i == i_t1 already wrote s_lam_mf[i_t2].
                    // Collapse delta_impulse to zero so no further
                    // v_out correction is applied for this row.
                    new_impulse = s_lam_mf[i];
                } else {
                    // First friction row of the triple. Run bisection on
                    // (λ_n, λ_t1, λ_t2), apply v_out corrections for the
                    // normal and t2 siblings in-place, and return
                    // new_λ_t1 as the ``new_impulse`` so the outer kernel
                    // applies the t1 v_out delta via its usual path.
                    int n_mf6 = mf6_base + mf_par * 6;
                    int t1_mf6 = mf6_base + i_t1 * 6;
                    int t2_mf6 = mf6_base + i_t2 * 6;

                    // Body indices are shared across the triple: read
                    // from the parent row's packed meta.
                    int parent_packed_dofs = mf_meta.data[off_meta + mf_par * 4];
                    int dof_a_par = parent_packed_dofs >> 16;
                    int dof_b_par = (parent_packed_dofs << 16) >> 16;
                    // FPGS stores rhs = beta * phi / dt (negative on
                    // penetration).  RAISim's bisection targets
                    // ``u_n >= b_n`` with ``b_n = -erp * gap / dt``
                    // (positive on penetration).  Flip the sign once
                    // so the rest of this block mirrors RAISim.
                    float target_vel_n = -__int_as_float(
                        mf_meta.data[off_meta + mf_par * 4 + 2]);
                    float mu = mf_row_mu.data[off_mf + i];

                    float old_lambda_n = s_lam_mf[mf_par];
                    float old_lambda_t1 = s_lam_mf[i_t1];
                    float old_lambda_t2 = s_lam_mf[i_t2];

                    float new_lambda_n = old_lambda_n;
                    float new_lambda_t1 = old_lambda_t1;
                    float new_lambda_t2 = old_lambda_t2;
                    float d_n_total = 0.0f;
                    float d_t2_total = 0.0f;

                    // Lane 0 runs the serial bisection; other lanes
                    // wait and then consume the broadcast results.
                    if (lane == 0) {
                        float u_n = 0.0f, u_t1 = 0.0f, u_t2 = 0.0f;
                        if (dof_a_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float va = s_v[dof_a_par + k];
                                u_n  += mf_J_a.data[n_mf6  + k] * va;
                                u_t1 += mf_J_a.data[t1_mf6 + k] * va;
                                u_t2 += mf_J_a.data[t2_mf6 + k] * va;
                            }
                        }
                        if (dof_b_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float vb = s_v[dof_b_par + k];
                                u_n  += mf_J_b.data[n_mf6  + k] * vb;
                                u_t1 += mf_J_b.data[t1_mf6 + k] * vb;
                                u_t2 += mf_J_b.data[t2_mf6 + k] * vb;
                            }
                        }

                        // __DESAXCE_BIAS__

                        float G_nn = 0.0f, G_nt1 = 0.0f, G_nt2 = 0.0f;
                        float G_t1t1 = 0.0f, G_t1t2 = 0.0f, G_t2t2 = 0.0f;
                        if (dof_a_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float Jna = mf_J_a.data[n_mf6  + k];
                                float Jt1a = mf_J_a.data[t1_mf6 + k];
                                float Jt2a = mf_J_a.data[t2_mf6 + k];
                                float Mna = mf_MiJt_a.data[n_mf6  + k];
                                float Mt1a = mf_MiJt_a.data[t1_mf6 + k];
                                float Mt2a = mf_MiJt_a.data[t2_mf6 + k];
                                G_nn   += Jna * Mna;
                                G_nt1  += Jna * Mt1a;
                                G_nt2  += Jna * Mt2a;
                                G_t1t1 += Jt1a * Mt1a;
                                G_t1t2 += Jt1a * Mt2a;
                                G_t2t2 += Jt2a * Mt2a;
                            }
                        }
                        if (dof_b_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float Jnb = mf_J_b.data[n_mf6  + k];
                                float Jt1b = mf_J_b.data[t1_mf6 + k];
                                float Jt2b = mf_J_b.data[t2_mf6 + k];
                                float Mnb = mf_MiJt_b.data[n_mf6  + k];
                                float Mt1b = mf_MiJt_b.data[t1_mf6 + k];
                                float Mt2b = mf_MiJt_b.data[t2_mf6 + k];
                                G_nn   += Jnb * Mnb;
                                G_nt1  += Jnb * Mt1b;
                                G_nt2  += Jnb * Mt2b;
                                G_t1t1 += Jt1b * Mt1b;
                                G_t1t2 += Jt1b * Mt2b;
                                G_t2t2 += Jt2b * Mt2b;
                            }
                        }

                        if (G_nn >= 1.0e-20f) {
                            // Check separating contact (λ_n = 0 solves it)
                            float u_n_at_zero = u_n + G_nn * (0.0f - old_lambda_n);
                            if (u_n_at_zero >= target_vel_n) {
                                new_lambda_n = 0.0f;
                                new_lambda_t1 = 0.0f;
                                new_lambda_t2 = 0.0f;
                            } else {
                                float lo = 0.0f;
                                float hi = fmaxf(
                                    old_lambda_n * 2.0f,
                                    (target_vel_n - u_n) / G_nn + old_lambda_n);
                                hi = fmaxf(hi, 1.0f);

                                for (int _bi = 0; _bi < 20; _bi++) {
                                    float mid = 0.5f * (lo + hi);
                                    float d_n = mid - old_lambda_n;
                                    float ut1_eff = u_t1 + G_nt1 * d_n;
                                    float ut2_eff = u_t2 + G_nt2 * d_n;

                                    float det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                    float d_t1 = 0.0f, d_t2 = 0.0f;
                                    if (fabsf(det) > 1.0e-20f) {
                                        d_t1 = (-ut1_eff * G_t2t2 + ut2_eff * G_t1t2) / det;
                                        d_t2 = ( ut1_eff * G_t1t2 - ut2_eff * G_t1t1) / det;
                                    }
                                    float trial_t1 = old_lambda_t1 + d_t1;
                                    float trial_t2 = old_lambda_t2 + d_t2;

                                    float flimit = mu * mid;
                                    float tmag = sqrtf(
                                        trial_t1 * trial_t1 + trial_t2 * trial_t2);
                                    if (tmag > flimit && tmag > 1.0e-20f) {
                                        float sc = flimit / tmag;
                                        trial_t1 *= sc;
                                        trial_t2 *= sc;
                                    }
                                    float d_t1_actual = trial_t1 - old_lambda_t1;
                                    float d_t2_actual = trial_t2 - old_lambda_t2;
                                    float u_n_trial = u_n + G_nn * d_n
                                        + G_nt1 * d_t1_actual
                                        + G_nt2 * d_t2_actual;
                                    if (u_n_trial < target_vel_n) lo = mid;
                                    else hi = mid;
                                }
                                new_lambda_n = 0.5f * (lo + hi);

                                float d_n_final = new_lambda_n - old_lambda_n;
                                float ut1_f = u_t1 + G_nt1 * d_n_final;
                                float ut2_f = u_t2 + G_nt2 * d_n_final;
                                float det_f = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                float d_t1_f = 0.0f, d_t2_f = 0.0f;
                                if (fabsf(det_f) > 1.0e-20f) {
                                    d_t1_f = (-ut1_f * G_t2t2 + ut2_f * G_t1t2) / det_f;
                                    d_t2_f = ( ut1_f * G_t1t2 - ut2_f * G_t1t1) / det_f;
                                }
                                new_lambda_t1 = old_lambda_t1 + d_t1_f;
                                new_lambda_t2 = old_lambda_t2 + d_t2_f;

                                float flimit_f = mu * new_lambda_n;
                                float tmag_f = sqrtf(
                                    new_lambda_t1 * new_lambda_t1
                                    + new_lambda_t2 * new_lambda_t2);
                                if (tmag_f > flimit_f && tmag_f > 1.0e-20f) {
                                    float sc_f = flimit_f / tmag_f;
                                    new_lambda_t1 *= sc_f;
                                    new_lambda_t2 *= sc_f;
                                }
                            }
                            d_n_total = new_lambda_n - old_lambda_n;
                            d_t2_total = new_lambda_t2 - old_lambda_t2;
                            s_lam_mf[mf_par] = new_lambda_n;
                            s_lam_mf[i_t2]   = new_lambda_t2;
                        }
                    }

                    __syncwarp();
                    new_lambda_t1 = __shfl_sync(MASK, new_lambda_t1, 0);
                    d_n_total = __shfl_sync(MASK, d_n_total, 0);
                    d_t2_total = __shfl_sync(MASK, d_t2_total, 0);
                    if (d_n_total != 0.0f || d_t2_total != 0.0f) iteration_changed = 1;

                    // Apply the normal-row v_out delta (all lanes).
                    if (d_n_total != 0.0f) {
                        if (lane < 6 && dof_a_par >= 0) {
                            s_v[dof_a_par + lane] +=
                                mf_MiJt_a.data[n_mf6 + lane] * d_n_total;
                        }
                        if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                            s_v[dof_b_par + lane - 6] +=
                                mf_MiJt_b.data[n_mf6 + lane - 6] * d_n_total;
                        }
                    }
                    // Apply the t2-row v_out delta (all lanes).
                    if (d_t2_total != 0.0f) {
                        if (lane < 6 && dof_a_par >= 0) {
                            s_v[dof_a_par + lane] +=
                                mf_MiJt_a.data[t2_mf6 + lane] * d_t2_total;
                        }
                        if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                            s_v[dof_b_par + lane - 6] +=
                                mf_MiJt_b.data[t2_mf6 + lane - 6] * d_t2_total;
                        }
                    }
                    __syncwarp();
                    new_impulse = new_lambda_t1;
                }
"""
        # Inject the per-mode label and the optional de Saxce bias
        # correction (``μ · ‖c_T‖`` added to ``target_vel_n`` once per
        # contact before the bisection) into the shared CUDA body.
        if friction_mode == "bisection_desaxce":
            mf_friction_block = mf_friction_block.replace(
                "// __BISECTION_LABEL__",
                '// friction_mode="bisection_desaxce": RAISim bisection + de Saxce bias (μ·‖c_T‖).',
            )
            mf_friction_block = mf_friction_block.replace(
                "// __DESAXCE_BIAS__",
                "{\n"
                "                                float c_T_mag = sqrtf(u_t1 * u_t1 + u_t2 * u_t2);\n"
                "                                target_vel_n = target_vel_n + mu * c_T_mag;\n"
                "                            }",
            )
        else:
            mf_friction_block = mf_friction_block.replace(
                "// __BISECTION_LABEL__",
                '// friction_mode="bisection": RAISim-style bisection on λ_n.',
            )
            # Pure RAISim bisection: no de Saxce bias.
            mf_friction_block = mf_friction_block.replace(
                "// __DESAXCE_BIAS__",
                '// (no de Saxce bias; friction_mode="bisection".)',
            )
    elif friction_mode == "coulomb_newton":
        # Gilles Daviet's 1D Coulomb Newton (7/13).  CUDA port of
        # :func:`friction_step_coulomb_newton`; mirrors the
        # bisection block's structure — lane 0 runs the serial
        # scalar Newton, then broadcasts (new_t1, d_n, d_t2) to
        # every lane via ``__shfl_sync`` so the v_out deltas for
        # the normal / t2 rows are applied cooperatively.
        mf_friction_block = """
                // friction_mode="coulomb_newton": Gilles Daviet's
                // scalar bracketed-Newton on alpha.  Ported from
                // ``coulomb_root_finding_warp.py::solve_coulomb`` —
                // see :func:`friction_step_coulomb_newton` for the
                // matrix-free row-data mapping.
                int mf_par = packed_tp >> 16;
                int i_t1 = mf_par + 1;
                int i_t2 = mf_par + 2;

                if (i != i_t1) {
                    // Second friction row of the triple: the
                    // Newton solve that ran at i == i_t1 already
                    // wrote s_lam_mf[i_t2].  Collapse
                    // delta_impulse to zero for this row.
                    new_impulse = s_lam_mf[i];
                } else {
                    int n_mf6 = mf6_base + mf_par * 6;
                    int t1_mf6 = mf6_base + i_t1 * 6;
                    int t2_mf6 = mf6_base + i_t2 * 6;

                    int parent_packed_dofs = mf_meta.data[off_meta + mf_par * 4];
                    int dof_a_par = parent_packed_dofs >> 16;
                    int dof_b_par = (parent_packed_dofs << 16) >> 16;
                    float target_vel_n = -__int_as_float(
                        mf_meta.data[off_meta + mf_par * 4 + 2]);
                    float mu = mf_row_mu.data[off_mf + i];

                    float old_lambda_n = s_lam_mf[mf_par];
                    float old_lambda_t1 = s_lam_mf[i_t1];
                    float old_lambda_t2 = s_lam_mf[i_t2];

                    float new_lambda_n = old_lambda_n;
                    float new_lambda_t1 = old_lambda_t1;
                    float new_lambda_t2 = old_lambda_t2;
                    float d_n_total = 0.0f;
                    float d_t2_total = 0.0f;

                    if (lane == 0) {
                        float u_n = 0.0f, u_t1 = 0.0f, u_t2 = 0.0f;
                        if (dof_a_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float va = s_v[dof_a_par + k];
                                u_n  += mf_J_a.data[n_mf6  + k] * va;
                                u_t1 += mf_J_a.data[t1_mf6 + k] * va;
                                u_t2 += mf_J_a.data[t2_mf6 + k] * va;
                            }
                        }
                        if (dof_b_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float vb = s_v[dof_b_par + k];
                                u_n  += mf_J_b.data[n_mf6  + k] * vb;
                                u_t1 += mf_J_b.data[t1_mf6 + k] * vb;
                                u_t2 += mf_J_b.data[t2_mf6 + k] * vb;
                            }
                        }

                        float G_nn = 0.0f, G_nt1 = 0.0f, G_nt2 = 0.0f;
                        float G_t1t1 = 0.0f, G_t1t2 = 0.0f, G_t2t2 = 0.0f;
                        if (dof_a_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float Jna = mf_J_a.data[n_mf6  + k];
                                float Jt1a = mf_J_a.data[t1_mf6 + k];
                                float Jt2a = mf_J_a.data[t2_mf6 + k];
                                float Mna = mf_MiJt_a.data[n_mf6  + k];
                                float Mt1a = mf_MiJt_a.data[t1_mf6 + k];
                                float Mt2a = mf_MiJt_a.data[t2_mf6 + k];
                                G_nn   += Jna * Mna;
                                G_nt1  += Jna * Mt1a;
                                G_nt2  += Jna * Mt2a;
                                G_t1t1 += Jt1a * Mt1a;
                                G_t1t2 += Jt1a * Mt2a;
                                G_t2t2 += Jt2a * Mt2a;
                            }
                        }
                        if (dof_b_par >= 0) {
                            for (int k = 0; k < 6; k++) {
                                float Jnb = mf_J_b.data[n_mf6  + k];
                                float Jt1b = mf_J_b.data[t1_mf6 + k];
                                float Jt2b = mf_J_b.data[t2_mf6 + k];
                                float Mnb = mf_MiJt_b.data[n_mf6  + k];
                                float Mt1b = mf_MiJt_b.data[t1_mf6 + k];
                                float Mt2b = mf_MiJt_b.data[t2_mf6 + k];
                                G_nn   += Jnb * Mnb;
                                G_nt1  += Jnb * Mt1b;
                                G_nt2  += Jnb * Mt2b;
                                G_t1t1 += Jt1b * Mt1b;
                                G_t1t2 += Jt1b * Mt2b;
                                G_t2t2 += Jt2b * Mt2b;
                            }
                        }

                        if (G_nn >= 1.0e-20f) {
                            // u_free = u_current - G * lambda_old
                            float u_free_n = u_n - (G_nn * old_lambda_n
                                + G_nt1 * old_lambda_t1
                                + G_nt2 * old_lambda_t2);
                            float u_free_t1 = u_t1 - (G_nt1 * old_lambda_n
                                + G_t1t1 * old_lambda_t1
                                + G_t1t2 * old_lambda_t2);
                            float u_free_t2 = u_t2 - (G_nt2 * old_lambda_n
                                + G_t1t2 * old_lambda_t1
                                + G_t2t2 * old_lambda_t2);

                            if (u_free_n < target_vel_n) {
                                // Reference b vector shifted into the
                                // positive-target frame (b_N < 0 on
                                // penetrating contacts).
                                float bN = u_free_n - target_vel_n;
                                float bT0 = u_free_t1;
                                float bT1 = u_free_t2;

                                float WN = G_nn;
                                float wNT0 = G_nt1;
                                float wNT1 = G_nt2;
                                // A_T = W_T - wNT wNT^T / WN.
                                float AT00 = G_t1t1 - (wNT0 * wNT0) / WN;
                                float AT01 = G_t1t2 - (wNT0 * wNT1) / WN;
                                float AT10 = G_t1t2 - (wNT1 * wNT0) / WN;
                                float AT11 = G_t2t2 - (wNT1 * wNT1) / WN;
                                // c_T = bT - (bN / WN) * wNT.
                                float cT0 = bT0 - (bN / WN) * wNT0;
                                float cT1 = bT1 - (bN / WN) * wNT1;

                                // Sticking check at alpha = 0.
                                float inv_det0 = 1.0f /
                                    (AT00 * AT11 - AT01 * AT10);
                                float s0_0 = (AT11 * cT0 - AT01 * cT1) * inv_det0;
                                float s0_1 = (AT00 * cT1 - AT10 * cT0) * inv_det0;
                                float norm_s0 = sqrtf(s0_0 * s0_0 + s0_1 * s0_1);
                                float phi0 = norm_s0 - mu *
                                    (wNT0 * s0_0 + wNT1 * s0_1 - bN) / WN;

                                float last_s0 = 0.0f, last_s1 = 0.0f;
                                float alpha = 0.0f;

                                if (phi0 <= 0.0f) {
                                    // Sticking: alpha = 0, s = s0.
                                    last_s0 = s0_0;
                                    last_s1 = s0_1;
                                    alpha = 0.0f;
                                } else {
                                    // Bracket: double hi until phi(hi) < 0.
                                    float hi = 1.0f;
                                    for (int _e = 0; _e < 30; _e++) {
                                        float a_hi = AT00 + hi;
                                        float d_hi = AT11 + hi;
                                        float idet_hi = 1.0f /
                                            (a_hi * d_hi - AT01 * AT10);
                                        float sh0 = (d_hi * cT0 - AT01 * cT1) * idet_hi;
                                        float sh1 = (a_hi * cT1 - AT10 * cT0) * idet_hi;
                                        float ns_hi = sqrtf(sh0 * sh0 + sh1 * sh1);
                                        float phi_hi = ns_hi - mu *
                                            (wNT0 * sh0 + wNT1 * sh1 - bN) / WN;
                                        if (phi_hi < 0.0f) break;
                                        hi *= 2.0f;
                                    }

                                    float lo = 0.0f;
                                    float x = 0.5f * (lo + hi);
                                    float tol = 1.0e-6f + 1.0e-6f * phi0;

                                    for (int _it = 0; _it < 50; _it++) {
                                        float ax = AT00 + x;
                                        float dx = AT11 + x;
                                        float idet = 1.0f /
                                            (ax * dx - AT01 * AT10);
                                        float s0 = (dx * cT0 - AT01 * cT1) * idet;
                                        float s1 = (ax * cT1 - AT10 * cT0) * idet;
                                        float t0 = (dx * s0 - AT01 * s1) * idet;
                                        float t1 = (ax * s1 - AT10 * s0) * idet;
                                        float ns = sqrtf(s0 * s0 + s1 * s1);
                                        float fx = ns - mu *
                                            (wNT0 * s0 + wNT1 * s1 - bN) / WN;
                                        float dfx = -(s0 * t0 + s1 * t1) / ns
                                            + mu * (wNT0 * t0 + wNT1 * t1) / WN;

                                        last_s0 = s0;
                                        last_s1 = s1;

                                        if (fabsf(fx) < tol
                                            || fabsf(hi - lo) < 1.0e-6f * (1.0f + hi)) {
                                            alpha = x;
                                            break;
                                        }

                                        if (fx > 0.0f) lo = x;
                                        else hi = x;

                                        float x_new = 0.5f * (lo + hi);
                                        if (dfx != 0.0f) {
                                            float x_newton = x - fx / dfx;
                                            if (x_newton > lo && x_newton < hi) {
                                                x_new = x_newton;
                                            }
                                        }
                                        x = x_new;
                                        alpha = x;
                                    }
                                }

                                // Recover r_T, r_N from the last s.
                                float rT0 = -last_s0;
                                float rT1 = -last_s1;
                                float rN = -(wNT0 * rT0 + wNT1 * rT1 + bN) / WN;
                                new_lambda_n = rN;
                                new_lambda_t1 = rT0;
                                new_lambda_t2 = rT1;
                            } else {
                                // Separating contact: zero impulse.
                                new_lambda_n = 0.0f;
                                new_lambda_t1 = 0.0f;
                                new_lambda_t2 = 0.0f;
                            }

                            d_n_total = new_lambda_n - old_lambda_n;
                            d_t2_total = new_lambda_t2 - old_lambda_t2;
                            s_lam_mf[mf_par] = new_lambda_n;
                            s_lam_mf[i_t2]   = new_lambda_t2;
                        }
                    }

                    __syncwarp();
                    new_lambda_t1 = __shfl_sync(MASK, new_lambda_t1, 0);
                    d_n_total = __shfl_sync(MASK, d_n_total, 0);
                    d_t2_total = __shfl_sync(MASK, d_t2_total, 0);
                    if (d_n_total != 0.0f || d_t2_total != 0.0f) iteration_changed = 1;

                    // Apply normal-row v_out delta (all lanes).
                    if (d_n_total != 0.0f) {
                        if (lane < 6 && dof_a_par >= 0) {
                            s_v[dof_a_par + lane] +=
                                mf_MiJt_a.data[n_mf6 + lane] * d_n_total;
                        }
                        if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                            s_v[dof_b_par + lane - 6] +=
                                mf_MiJt_b.data[n_mf6 + lane - 6] * d_n_total;
                        }
                    }
                    // Apply t2-row v_out delta (all lanes).
                    if (d_t2_total != 0.0f) {
                        if (lane < 6 && dof_a_par >= 0) {
                            s_v[dof_a_par + lane] +=
                                mf_MiJt_a.data[t2_mf6 + lane] * d_t2_total;
                        }
                        if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                            s_v[dof_b_par + lane - 6] +=
                                mf_MiJt_b.data[t2_mf6 + lane - 6] * d_t2_total;
                        }
                    }
                    __syncwarp();
                    new_impulse = new_lambda_t1;
                }
"""
    else:
        mf_friction_block = f"""
                int mf_par = packed_tp >> 16;
                if (i != mf_par + 1) {{
                    new_impulse = old_impulse;
                }} else {{
                    int sib = mf_par + 2;
                    int sib_mf6 = mf6_base + sib * 6;
                    float radius = mf_friction_radius;
                    float2 pair = make_float2(0.0f, 0.0f);
                    // Zero load gives a zero disk; avoid unused tangent reductions.
                    if (radius > 0.0f) {{
                        float sibling_residual = 0.0f;
                        if (lane < 6 && dof_a >= 0)
                            sibling_residual = mf_J_a.data[sib_mf6 + lane] {mf_velocity("dof_a + lane", "lane")};
                        if (lane >= 6 && lane < 12 && dof_b >= 0)
                            sibling_residual = mf_J_b.data[sib_mf6 + lane - 6] {mf_velocity("dof_b + lane - 6", "lane - 6")};
                        for (int offset = 16; offset > 0; offset >>= 1)
                            sibling_residual += __shfl_down_sync(MASK, sibling_residual, offset);
                        sibling_residual = __shfl_sync(MASK, sibling_residual, 0)
                            + __int_as_float(mf_meta.data[off_meta + sib * 4 + 2]);
                        float inv_sib = __int_as_float(mf_meta.data[off_meta + sib * 4 + 1]);
                        float cross = 0.0f;
                        if (lane < 6 && dof_a >= 0)
                            cross = mf_J_a.data[mf6_base + i * 6 + lane] * mf_MiJt_a.data[sib_mf6 + lane];
                        if (lane >= 6 && lane < 12 && dof_b >= 0)
                            cross = mf_J_b.data[mf6_base + i * 6 + lane - 6] * mf_MiJt_b.data[sib_mf6 + lane - 6];
                        for (int offset = 16; offset > 0; offset >>= 1)
                            cross += __shfl_down_sync(MASK, cross, offset);
                        cross = __shfl_sync(MASK, cross, 0);
                        pair = friction_pair_candidate(mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f, cross, inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f,
                            residual, sibling_residual, old_impulse, s_lam_mf[sib], radius, omega);
                    }}
                    float a = pair.x;
                    float b = pair.y;
                    float mag = sqrtf(a * a + b * b);
                    float scale = mag > radius ? radius / mag : 1.0f;
                    new_impulse = a * scale;
                    float sib_delta = b * scale - s_lam_mf[sib];
                    s_lam_mf[sib] = b * scale;
{mf_sibling_vote}
                    if (lane < 6 && dof_a >= 0)
                        s_v[dof_a + lane] += mf_MiJt_a.data[sib_mf6 + lane] * sib_delta;
                    if (lane >= 6 && lane < 12 && dof_b >= 0)
                        s_v[dof_b + lane - 6] += mf_MiJt_b.data[sib_mf6 + lane - 6] * sib_delta;
                }}
"""

    mf_friction_precheck = (
        """
            float mf_friction_radius = 0.0f;
            if (mf_rt == 2) {
                int parent = packed_tp >> 16;
                float lambda_n = s_lam_mf[parent];
                for (int patch_row = (mf_meta.data[off_meta + parent * 4 + 3] >> 16); patch_row >= 0 && patch_row != parent; patch_row = (mf_meta.data[off_meta + patch_row * 4 + 3] >> 16))
                    lambda_n += s_lam_mf[patch_row];
                mf_friction_radius = fmaxf(mf_row_mu.data[off_mf + i] * lambda_n, 0.0f);
                // A zero disk with no carried impulse cannot change velocity.
                if (mf_friction_radius == 0.0f && s_lam_mf[i] == 0.0f && s_lam_mf[i + 1] == 0.0f)
                    continue;
            }
"""
        if friction_mode == "current"
        else ""
    )

    drive_shared_declarations = (
        f"""
    __shared__ float s_drive_target_dense[{M_D}];
    __shared__ float s_drive_vel_mul_dense[{M_D}];
    __shared__ float s_drive_imp_mul_dense[{M_D}];
    __shared__ float s_drive_max_imp_dense[{M_D}];"""
        if has_drive_rows
        else ""
    )
    drive_loads = (
        """
        s_drive_target_dense[i] = world_drive_target_vel_bias.data[off_dense + i];
        s_drive_vel_mul_dense[i] = world_drive_vel_multiplier.data[off_dense + i];
        s_drive_imp_mul_dense[i] = world_drive_impulse_multiplier.data[off_dense + i];
        s_drive_max_imp_dense[i] = world_drive_max_impulse.data[off_dense + i];"""
        if has_drive_rows
        else ""
    )
    if fuse_vel_limits:
        drive_shared_declarations += f"""
    __shared__ float s_drive_vel_limit_dense[{M_D}];"""
        drive_loads += """
        s_drive_vel_limit_dense[i] = world_drive_vel_limit.data[off_dense + i];"""
    drive_projection = (
        """
            if (row_type == 1) {
                new_impulse = old_impulse * s_drive_imp_mul_dense[i]
                    + jv * s_drive_vel_mul_dense[i]
                    + s_drive_target_dense[i];
                float max_imp = s_drive_max_imp_dense[i];
                if (new_impulse > max_imp) new_impulse = max_imp;
                if (new_impulse < -max_imp) new_impulse = -max_imp;
                delta_impulse = new_impulse - old_impulse;
            }"""
        if has_drive_rows
        else """
            if (row_type == 1) {
                continue;
            }"""
    )
    # Every visit of a velocity-limit row: the end-of-iteration dense and free-body passes, the
    # dense rows of a velocity-limit row phase (row_phase 5), and the contact loop's free-body
    # branch, which the contact-first row order never reaches.
    if accumulated_limits:
        # Lane 0 owns a dense total, as it owns the scalar dense rows' impulses.
        dense_velocity_limit_law = (
            "                const float old_vlim = __shfl_sync(MASK, lane == 0 ? s_lam_dense[i] : 0.0f, 0);\n"
            + accumulated_update(
                "old_vlim",
                "residual_vlim / denom",
                new="new_vlim",
                delta="delta_impulse_vlim",
                indent=" " * 16,
                declare="float ",
            )
            + "                __syncwarp();\n"
            + "                if (lane == 0) s_lam_dense[i] = new_vlim;"
        )
        # Every lane reads a free-body total before lane 0 stores the new one: a plain read stays
        # warp-uniform when streamed impulses rewrite it into a lane-0 broadcast.
        mf_velocity_limit_law = (
            "                const float old_mf_vlim = s_lam_mf[i];\n"
            + accumulated_update(
                "old_mf_vlim",
                "residual_mf_vlim * mf_diag",
                new="new_mf_vlim",
                delta="delta_impulse_mf_vlim",
                indent=" " * 16,
                declare="float ",
            )
            + "                __syncwarp();\n"
            + "                if (lane == 0) s_lam_mf[i] = new_mf_vlim;"
        )
        dense_velocity_limit_row = (
            "            // row_type 4=JOINT_VELOCITY_LIMIT: an accumulated unilateral row,\n"
            "            // unrelaxed, as in the end-of-iteration velocity-limit pass.\n"
            "            else if (row_type == 4) {\n"
            + accumulated_update(
                "old_impulse", "residual / denom", new="new_impulse", delta="delta_impulse", indent=" " * 16
            )
            + "            }"
        )
        mf_velocity_limit_row = accumulated_update(
            "old_impulse", "residual * mf_diag", new="new_impulse", delta="delta_impulse", indent=" " * 16
        ).rstrip("\n")
    else:
        dense_velocity_limit_law = """                float delta_vlim = -residual_vlim / denom;
                float delta_impulse_vlim = 0.0f;
                if (residual_vlim < 0.0f) delta_impulse_vlim = delta_vlim;
                if (lane == 0) s_lam_dense[i] = delta_impulse_vlim;"""
        mf_velocity_limit_law = """                float delta_mf_vlim = -residual_mf_vlim * mf_diag;
                float delta_impulse_mf_vlim = 0.0f;
                if (residual_mf_vlim < 0.0f) delta_impulse_mf_vlim = delta_mf_vlim;
                if (lane == 0) s_lam_mf[i] = delta_impulse_mf_vlim;"""
        dense_velocity_limit_row = """            // row_type 4=JOINT_VELOCITY_LIMIT:
            // PhysX-style stateless bilateral speed projection. The row
            // uses a signed Jacobian and target velocity so residual < 0
            // means that side of the velocity box is currently violated.
            // Do not accumulate lambda: each visit recomputes jointV and
            // applies only the impulse needed for the current overshoot.
            else if (row_type == 4) {
                if (residual < 0.0f) {
                    // PhysX applies the clamp delta directly; it is not
                    // a relaxed PGS correction.
                    delta_impulse = delta;
                    new_impulse = delta_impulse;
                } else {
                    new_impulse = 0.0f;
                }
            }"""
        mf_velocity_limit_row = """                if (residual < 0.0f) {
                    delta_impulse = delta;
                    new_impulse = delta_impulse;
                } else {
                    delta_impulse = 0.0f;
                    new_impulse = 0.0f;
                }"""
    dense_velocity_limit_phase = (
        f"""
        if (row_phase == 0 || row_phase == 2) {{
            for (int i = 0; i < m_dense; i++) {{
                if ((s_meta_dense[i] & {_DENSE_META_ROW_TYPE_MASK}) != {int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)}) continue;
                float denom = s_diag_dense[i];
                if (denom <= 0.0f) continue;

                float my_sum_vlim = 0.0f;
                int row_base_vlim = jy_world_base + i * {D};
                for (int d = lane; d < {D}; d += 32) {{
                    my_sum_vlim += {dense_read("J_world", "row_base_vlim + d")} * s_v[d];
                }}
                my_sum_vlim += __shfl_down_sync(MASK, my_sum_vlim, 16);
                my_sum_vlim += __shfl_down_sync(MASK, my_sum_vlim, 8);
                my_sum_vlim += __shfl_down_sync(MASK, my_sum_vlim, 4);
                my_sum_vlim += __shfl_down_sync(MASK, my_sum_vlim, 2);
                my_sum_vlim += __shfl_down_sync(MASK, my_sum_vlim, 1);
                float jv_vlim = __shfl_sync(MASK, my_sum_vlim, 0);

                float residual_vlim = jv_vlim + s_rhs_dense[i];
{dense_velocity_limit_law}

                if (delta_impulse_vlim != 0.0f) {{
{dense_limit_vote}
                    for (int d = lane; d < {D}; d += 32) {{
                        if (defer_dense_response == 0 ||
                            world_deferred_dof_mask.data[deferred_mask_base + d] == 0) {{
                            s_v[d] += {dense_read("Y_world", "row_base_vlim + d")} * delta_impulse_vlim;
                        }}
                    }}
                }}
                __syncwarp();
            }}
        }}"""
        if has_dense_velocity_limit_rows
        else ""
    )

    # Fused stateless velocity clamp on driven DOFs: the same PhysX PxClamp /
    # unitResponse math as _FVL_LAST_WORD_CLAMP_SNIPPET (propagation
    # full-iteration kernel), in this kernel's idiom (packed s_meta_dense,
    # deferred-response masking). Runs where the dedicated velocity-limit
    # rows run — after the contact rows of the same iteration — so the limit
    # keeps the last word even when the sweep is under-converged (the
    # ordering PhysX documents for gripping scenarios via
    # PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST). Velocity-only: the
    # persistent drive lambda keeps the drive-only value.
    fused_drive_vel_limit_phase = (
        f"""
        if (row_phase == 0 || row_phase == 2 || row_phase == 5) {{
            int drive_hi_fvl = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
            for (int i = 0; i < drive_hi_fvl; i++) {{
                if ((s_meta_dense[i] & {_DENSE_META_ROW_TYPE_MASK}) != {int(PGS_CONSTRAINT_TYPE_JOINT_TARGET)}) continue;
                float qdot_max_fvl = s_drive_vel_limit_dense[i];
                float denom_fvl = s_diag_dense[i];
                if (denom_fvl <= 0.0f) continue;

                float my_sum_fvl = 0.0f;
                int row_base_fvl = jy_world_base + i * {D};
                for (int d = lane; d < {D}; d += 32) {{
                    my_sum_fvl += {dense_read("J_world", "row_base_fvl + d")} * s_v[d];
                }}
                my_sum_fvl += __shfl_down_sync(MASK, my_sum_fvl, 16);
                my_sum_fvl += __shfl_down_sync(MASK, my_sum_fvl, 8);
                my_sum_fvl += __shfl_down_sync(MASK, my_sum_fvl, 4);
                my_sum_fvl += __shfl_down_sync(MASK, my_sum_fvl, 2);
                my_sum_fvl += __shfl_down_sync(MASK, my_sum_fvl, 1);
                float jv_fvl = __shfl_sync(MASK, my_sum_fvl, 0);

                if (fabsf(jv_fvl) > qdot_max_fvl) {{
                    float delta_fvl = (fminf(fmaxf(jv_fvl, -qdot_max_fvl), qdot_max_fvl) - jv_fvl) / denom_fvl;
{drive_limit_vote}
                    for (int d = lane; d < {D}; d += 32) {{
                        if (defer_dense_response == 0 ||
                            world_deferred_dof_mask.data[deferred_mask_base + d] == 0) {{
                            s_v[d] += {dense_read("Y_world", "row_base_fvl + d")} * delta_fvl;
                        }}
                    }}
                }}
                __syncwarp();
            }}
        }}"""
        if fuse_vel_limits
        else ""
    )

    # The end-of-iteration fused clamp revisits the drive-row prefix, so a
    # phase-5 launch (physx_grasp schedule stages only the vel-limit segment)
    # must extend its dense staging window down to row 0.
    fused_load_lo_decl = (
        """
    int load_lo = (row_phase == 5) ? 0 : dense_lo;"""
        if fuse_vel_limits
        else ""
    )
    dense_load_start = "load_lo" if fuse_vel_limits else "dense_lo"
    torsion_skip = f"if (row_type == {PGS_CONSTRAINT_TYPE_TORSION}) continue;" if contact_torsion else ""
    torsion_sweep = torque_sweep_source(D) if contact_torsion else ""
    # A limit-only launch owns no contact impulses. Do not stage/store the
    # potentially much larger contact prefix, or stage any world velocities
    # when neither dense nor MF limits exist. Fused drive clamps remain work
    # even when the explicit dense limit segment is empty.
    empty_limit_phase = (
        ""
        if fuse_vel_limits
        else """
    if (row_phase == 5 && dense_lo == dense_hi && mf_contact_end == m_mf) return;"""
    )

    mf_phase = f"""        // ── Phase 2: MF constraints (6-DOF per body, software-pipelined) ──
        // MF contact/friction rows only exist in phases 0/1/4. The guard sits
        // OUTSIDE the loop: phase-3/5 launches previously prefetch-scanned
        // every MF row (meta + J/MiJt) only to discard it, which dominated
        // split-propagation scheduling cost on free-object-heavy scenes.
        if (row_phase == 0 || row_phase == 1 || row_phase == 4) {{

        // Pipeline registers: prefetch next constraint's global data
        int4 pre_meta;
        float pre_Ja = 0.0f, pre_Jb = 0.0f;
        float pre_MiJta = 0.0f, pre_MiJtb = 0.0f;

        // Prefetch constraint 0 (contact/friction rows occupy [0, mf_contact_end))
        if (mf_contact_end > 0) {{
            pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta]);
            if (lane < 6) {{
                pre_Ja = mf_J_a.data[mf6_base + lane];
                pre_MiJta = mf_MiJt_a.data[mf6_base + lane];
            }}
            if (lane >= 6 && lane < 12) {{
                pre_Jb = mf_J_b.data[mf6_base + lane - 6];
                pre_MiJtb = mf_MiJt_b.data[mf6_base + lane - 6];
            }}
        }}

        for (int i = 0; i < mf_contact_end; i++) {{
            // Consume prefetched data for constraint i
            int4 meta = pre_meta;
            float cur_Ja = pre_Ja;
            float cur_Jb = pre_Jb;
            float cur_MiJta = pre_MiJta;
            float cur_MiJtb = pre_MiJtb;

            // Prefetch constraint i+1 (loads issued now, complete during compute)
            if (i + 1 < mf_contact_end) {{
                int next_mf6 = mf6_base + (i + 1) * 6;
                pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + (i + 1) * 4]);
                if (lane < 6) {{
                    pre_Ja = mf_J_a.data[next_mf6 + lane];
                    pre_MiJta = mf_MiJt_a.data[next_mf6 + lane];
                }}
                if (lane >= 6 && lane < 12) {{
                    pre_Jb = mf_J_b.data[next_mf6 + lane - 6];
                    pre_MiJtb = mf_MiJt_b.data[next_mf6 + lane - 6];
                }}
            }}

            // Process constraint i
            int packed_dofs = meta.x;
            int dof_a = packed_dofs >> 16;
            int dof_b = (packed_dofs << 16) >> 16;
            float mf_diag = __int_as_float(meta.y);
            int packed_tp = meta.w;
            int mf_rt = packed_tp & 0xFFFF;

            if ((row_phase == 1 || row_phase == 4) && mf_rt != 0 && mf_rt != 2) continue;
            if (row_phase == 0 && mf_rt == 4) continue;
            if (mf_rt == 2 && global_iter < friction_start_iteration) {{
                s_lam_mf[i] = 0.0f;
                __syncwarp();
                continue;
            }}

            if (mf_rt == 2 && i != (packed_tp >> 16) + 1) continue;
            if (mf_diag <= 0.0f{" && mf_rt != 2" if friction_mode == "current" else ""}) continue;

            {mf_friction_precheck}

            // J · v using prefetched J values
            float my_sum = 0.0f;
            if (lane < 6 && dof_a >= 0) {{
                my_sum = cur_Ja {mf_velocity("dof_a + lane", "lane")};
            }}
            if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                my_sum = cur_Jb {mf_velocity("dof_b + lane - 6", "lane - 6")};
            }}
            my_sum += __shfl_down_sync(MASK, my_sum, 16);
            my_sum += __shfl_down_sync(MASK, my_sum, 8);
            my_sum += __shfl_down_sync(MASK, my_sum, 4);
            my_sum += __shfl_down_sync(MASK, my_sum, 2);
            my_sum += __shfl_down_sync(MASK, my_sum, 1);
            float jv = __shfl_sync(MASK, my_sum, 0);

            float residual = jv + __int_as_float(meta.z);
            float old_impulse = s_lam_mf[i];
            float delta = -residual * mf_diag;
            if (mf_rt == 0) {{
                // Relative diagonal regularization (residual + divisor): makes
                // indeterminate normal-force splits unique. Identity at g = 0.
                float w = regularize ? mf_row_w.data[off_mf + i] : 1.0f;
                delta = -residual * mf_diag * w - (1.0f - w) * old_impulse;
            }}
            float new_impulse = old_impulse + omega * delta;
            float delta_impulse = 0.0f;

            if (mf_rt == 0) {{
                if (new_impulse < 0.0f) new_impulse = 0.0f;
            }} else if (mf_rt == 4) {{
{mf_velocity_limit_row}
            }} else if (mf_rt == 2) {{
                {mf_friction_block}
            }}

            if (mf_rt != 4) delta_impulse = new_impulse - old_impulse;
            s_lam_mf[i] = new_impulse;

            // V update using prefetched MiJt values
            if (delta_impulse != 0.0f) {{
{mf_vote}
                if (lane < 6 && dof_a >= 0) {{
                    s_v[dof_a + lane] += cur_MiJta * delta_impulse;
                }}
                if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                    s_v[dof_b + lane - 6] += cur_MiJtb * delta_impulse;
                }}
            }}
            __syncwarp();
        }}
        }}

        // ── Final velocity-limit phase ──
        // Default/interleaved solves skip row_type=4 above and visit both
        // dense articulated limits and MF rigid limits here, after all
        // drive/contact/friction/position-limit rows. Split schedules use
        // row_phase 2/5 as their explicit final velocity-limit pass.
{dense_velocity_limit_phase}
{fused_drive_vel_limit_phase}

        if (row_phase == 0 || row_phase == 2 || row_phase == 5) {{
            for (int i = mf_contact_end; i < m_mf; i++) {{
                int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + i * 4]);
                int packed_tp = meta.w;
                int mf_rt = packed_tp & 0xFFFF;
                if (mf_rt != 4) continue;

                int packed_dofs = meta.x;
                int dof_a = packed_dofs >> 16;
                int dof_b = (packed_dofs << 16) >> 16;
                float mf_diag = __int_as_float(meta.y);
                if (mf_diag <= 0.0f) continue;

                int row_mf6 = mf6_base + i * 6;
                float my_sum_mf_vlim = 0.0f;
                if (lane < 6 && dof_a >= 0) {{
                    my_sum_mf_vlim = mf_J_a.data[row_mf6 + lane] {mf_velocity("dof_a + lane", "lane")};
                }}
                if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                    my_sum_mf_vlim = mf_J_b.data[row_mf6 + lane - 6] {mf_velocity("dof_b + lane - 6", "lane - 6")};
                }}
                my_sum_mf_vlim += __shfl_down_sync(MASK, my_sum_mf_vlim, 16);
                my_sum_mf_vlim += __shfl_down_sync(MASK, my_sum_mf_vlim, 8);
                my_sum_mf_vlim += __shfl_down_sync(MASK, my_sum_mf_vlim, 4);
                my_sum_mf_vlim += __shfl_down_sync(MASK, my_sum_mf_vlim, 2);
                my_sum_mf_vlim += __shfl_down_sync(MASK, my_sum_mf_vlim, 1);
                float jv_mf_vlim = __shfl_sync(MASK, my_sum_mf_vlim, 0);

                float residual_mf_vlim = jv_mf_vlim + __int_as_float(meta.z);
{mf_velocity_limit_law}

                if (delta_impulse_mf_vlim != 0.0f) {{
{mf_limit_vote}
                    if (lane < 6 && dof_a >= 0) {{
                        s_v[dof_a + lane] += mf_MiJt_a.data[row_mf6 + lane] * delta_impulse_mf_vlim;
                    }}
                    if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                        s_v[dof_b + lane - 6] += mf_MiJt_b.data[row_mf6 + lane - 6] * delta_impulse_mf_vlim;
                    }}
                }}
                __syncwarp();
            }}
        }}

        {torsion_sweep}
        // Friction rows may intentionally remain inactive until a later
        // iteration. Once they are active, an exactly stationary full sweep
        // is a fixed point, so subsequent sweeps are redundant.
"""

    sweep_end = sweep_exit(
        solve_policy,
        """        if (global_iter >= friction_start_iteration) {
            unsigned changed = __ballot_sync(MASK, iteration_changed != 0);
            if (changed == 0u) break;
        }
""",
        gate="global_iter >= friction_start_iteration",
        quiet="__ballot_sync(MASK, iteration_changed != 0) == 0u",
        iteration="iter",
    )
    snippet = f"""    const unsigned MASK = 0xFFFFFFFF;
    int lane = threadIdx.x;{held_entry}

    int m_dense = world_constraint_count.data[world];
    int m_mf = mf_constraint_count.data[world];
    if (m_dense == 0 && m_mf == 0) return;
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
{factor_fallback_skip}
{local_internal_skip}
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;

    // The dense slot layout is [drive][joint-limit][joint-vel-limit]
    // [contact/friction], so each PhysX-grasp row phase owns one contiguous
    // dense row range and only needs to load/scan/store that range:
    //   row_phase 3: [0, bounds[0])          drive + position-limit rows
    //   row_phase 5: [bounds[0], bounds[1])  joint velocity-limit rows
    //   row_phase 4: [bounds[1], m_dense)    contact + friction rows
    // Other phases scan every dense row. Shared arrays keep absolute
    // indexing and the per-row type filters below remain as a safety net.
    int dense_lo = 0;
    int dense_hi = m_dense;
    if (row_phase == 3) {{
        dense_hi = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
    }} else if (row_phase == 5) {{
        dense_lo = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
        dense_hi = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }} else if (row_phase == 4) {{
        dense_lo = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }}{fused_load_lo_decl}{empty_limit_phase}

    int dof_map_base = world * {D};
    int off_dense = world * {M_D};
    int off_mf = world * {M_MF};
    int off_meta = off_mf * 4;
    int jy_world_base = world * {M_D} * {D};
    int deferred_mask_base = world * {D};
    int mf6_base = world * {M_MF} * 6;
    int mf_load_lo = (row_phase == 5) ? mf_contact_end : 0;

    // ═══════════════════════════════════════════════════════
    // SHARED MEMORY
    // ═══════════════════════════════════════════════════════
{shared("float", "s_v", D, " ")}
{shared("float", "s_lam_dense", M_D, " ")}
{shared("float", "s_rhs_dense", M_D, " ")}
{shared("float", "s_diag_dense", M_D, " ")}
    // Low 3 bits: row type (0..4). Remaining bits: parent + 1 (-1 maps to 0).
{shared("int", "s_meta_dense", M_D, "   ")}
{shared("float", "s_mu_dense", M_D, " ")}
{drive_shared_declarations}
{shared("float", "s_lam_mf", M_MF, " ")}

    // ═══════════════════════════════════════════════════════
    // LOAD PHASE
    // ═══════════════════════════════════════════════════════
    for (int i = {dense_load_start} + lane; i < dense_hi; i += 32) {{
        s_lam_dense[i] = world_impulses.data[off_dense + i];
        s_rhs_dense[i] = {dense_rhs};
        s_diag_dense[i] = world_diag.data[off_dense + i];
        int row_type = world_row_type.data[off_dense + i];
        int row_parent = world_row_parent.data[off_dense + i];
        s_meta_dense[i] = (row_type & {_DENSE_META_ROW_TYPE_MASK}) | ((row_parent + 1) << {_DENSE_META_ROW_TYPE_BITS});
        s_mu_dense[i] = world_row_mu.data[off_dense + i];
{drive_loads}
    }}
    if (row_phase != 3 && row_phase != 6) {{
        // Dense-only phases touch no MF rows; leave their impulses unstaged.
        for (int i = mf_load_lo + lane; i < m_mf; i += 32) {{
            s_lam_mf[i] = mf_impulses.data[off_mf + i];
        }}
    }}
    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        {velocity_load}
    }}
    __syncwarp();

    // ═══════════════════════════════════════════════════════
    // SOLVE PHASE
    // ═══════════════════════════════════════════════════════
    // Dense pipeline registers
{dense_pipe_decl}

    for (int iter = 0; iter < iterations; iter++) {{
        int global_iter = iteration_offset + iter;
        int iteration_changed = 0;

        // ── Phase 1: Dense constraints (D-DOF warp-parallel, software-pipelined) ──

        // Prefetch the first constraint of this phase's dense range
        if (dense_hi > dense_lo) {{
            {dense_prefetch_init_code}
        }}

        for (int i = dense_lo; i < dense_hi; i++) {{
            // Consume prefetched J/Y for constraint i
            {dense_consume_code}

            // Prefetch constraint i+1
            if (i + 1 < dense_hi) {{
                int next_jy_base = jy_world_base + (i + 1) * {D};
                {dense_prefetch_next_code}
            }}

            int row_type = s_meta_dense[i] & {_DENSE_META_ROW_TYPE_MASK};
            {torsion_skip}
            // row_phase 0: all rows.
            // row_phase 1: contact/friction rows.
            // row_phase 2: internal articulation rows.
            // row_phase 3: PhysX-grasp first pass, drive/position-limit rows.
            // row_phase 4: PhysX-grasp contact pass, contact/friction rows.
            // row_phase 5: PhysX-grasp final pass, joint velocity-limit rows.
            // row_phase 6: dense rows before the multi-block free-contact pass.
            if (row_phase == 1 && row_type != 0 && row_type != 2) continue;
            if (row_phase == 2 && row_type != 1 && row_type != 3 && row_type != 4 && row_type != 5 && row_type != 6) continue;
            if (row_phase == 3 && row_type != 1 && row_type != 3 && row_type != 5 && row_type != 6) continue;
            if (row_phase == 4 && row_type != 0 && row_type != 2) continue;
            if (row_phase == 5 && row_type != 4) continue;
            if ((row_phase == 0 || row_phase == 2 || row_phase == 6) && row_type == 4) continue;
            if (freeze_drive_rows != 0 && row_type == 1) continue;
            if (row_type == 2 && global_iter < friction_start_iteration) {{
                {dense_impulse_store}s_lam_dense[i] = 0.0f;
                __syncwarp();
                continue;
            }}

            float denom = s_diag_dense[i];
            if (denom <= 0.0f{" && row_type != 2" if friction_mode == "current" else ""}) continue;

            // J_i · v (using prefetched J)
            {dense_dot_code}

            // Warp reduce
            my_sum += __shfl_down_sync(MASK, my_sum, 16);
            my_sum += __shfl_down_sync(MASK, my_sum, 8);
            my_sum += __shfl_down_sync(MASK, my_sum, 4);
            my_sum += __shfl_down_sync(MASK, my_sum, 2);
            my_sum += __shfl_down_sync(MASK, my_sum, 1);
            float jv = __shfl_sync(MASK, my_sum, 0);

            float old_impulse = {dense_old_impulse};
            float w = regularize ? world_row_w.data[off_dense + i] : 1.0f;
            float residual = jv + s_rhs_dense[i];
            float delta = denom > 0.0f ? -residual / denom * w - (1.0f - w) * old_impulse : 0.0f;
            float new_impulse = old_impulse + omega * delta;
            float delta_impulse = 0.0f;

            // row_type 1=JOINT_TARGET: PhysX PGS force-drive row.
            // computeDriveImpulse(lambda, jointVel, jointDeltaPos=0,
            // elapsedTime=0, driveDesc), then clamp to maxForce * dt.
{drive_projection}
{dense_velocity_limit_row}
            // row_type 0=CONTACT, 3=JOINT_LIMIT: unilateral lambda >= 0
            // projector with accumulated impulse.
            else if (row_type == 0 || row_type == 3) {{
                if (new_impulse < 0.0f) new_impulse = 0.0f;
                delta_impulse = new_impulse - old_impulse;
            }} else if (row_type == 2) {{
                {dense_friction_block}
                delta_impulse = new_impulse - old_impulse;
            }} else {{
                delta_impulse = new_impulse - old_impulse;
            }}

            {dense_impulse_store}s_lam_dense[i] = new_impulse;

            // V update using prefetched Y
            if (delta_impulse != 0.0f) {{
{dense_vote}
                {dense_v_update_code}
            }}
            __syncwarp();
        }}

{mf_phase}{sweep_end}    }}

    // ═══════════════════════════════════════════════════════
    // STORE PHASE
    // ═══════════════════════════════════════════════════════
{velocity_store}
    for (int i = dense_lo + lane; i < dense_hi; i += 32) {{
        world_impulses.data[off_dense + i] = s_lam_dense[i];
    }}
    if (row_phase != 3 && row_phase != 6) {{
        for (int i = mf_load_lo + lane; i < m_mf; i += 32) {{
            mf_impulses.data[off_mf + i] = s_lam_mf[i];
        }}
    }}
{cpu_section}#endif
"""

    snippet = snippet.replace("packed_tp >> 16", _mf_parent_expression("packed_tp", M_MF))
    for parent_row in ("parent", "patch_row"):
        packed_parent = f"mf_meta.data[off_meta + {parent_row} * 4 + 3]"
        snippet = snippet.replace(f"{packed_parent} >> 16", _mf_parent_expression(packed_parent, M_MF))
    if not shared_metadata:
        # Shared-memory diet: the nine read-only per-row metadata arrays are
        # ~9/11ths of this kernel's shared footprint and pin occupancy to
        # floor(smem_per_sm / (10*M_D*4B)). Stream them from global/L2 instead;
        # values are identical, only the storage class changes.
        # s_meta_dense cannot alias a single global stream (it is packed at
        # load time from two arrays), so its reads are rewritten back to the
        # source arrays below. Keeping the packed array resident costs 4*M_D
        # bytes of static shared memory per block; at M_D=1024 that is +4 KiB
        # (e.g. 22.9 -> 27.0 KiB with M_MF=4096, D=604), dropping occupancy
        # from 4 to 3 blocks/SM on 100-KiB-smem parts (sm_86/89/120) and
        # regressing dense-contact scenes by the lost concurrency. Replay of a
        # captured launch confirms the streamed variant is bitwise-identical
        # in v_out, impulses, and mf_impulses. A launch with at most one block
        # per SM has no concurrency to lose; resident_row_metadata keeps the
        # packed array there (execution_plan._select_mfgs_storage).
        meta_map = {
            "s_rhs_dense": ("float", "rhs_bias"),
            "s_diag_dense": ("float", "world_diag"),
            "s_mu_dense": ("float", "world_row_mu"),
            "s_drive_target_dense": ("float", "world_drive_target_vel_bias"),
            "s_drive_vel_mul_dense": ("float", "world_drive_vel_multiplier"),
            "s_drive_imp_mul_dense": ("float", "world_drive_impulse_multiplier"),
            "s_drive_max_imp_dense": ("float", "world_drive_max_impulse"),
        }
        if fuse_vel_limits:
            meta_map["s_drive_vel_limit_dense"] = ("float", "world_drive_vel_limit")
        for sname, (ctype, gname) in meta_map.items():
            snippet = re.sub(rf"    __shared__ {ctype}\s+{sname}\[{M_D}\];\n", "", snippet)
            snippet = re.sub(rf"\s*{sname}\[i\] = {gname}\.data\[off_dense \+ i\];", "", snippet, count=1)
            snippet = re.sub(rf"{sname}\[([^\]]*)\]", rf"{gname}.data[off_dense + (\1)]", snippet)

        # Stream the packed dense-row metadata: drop the resident array and
        # its load-time packing, and read row type/parent straight from their
        # global source arrays (identical values, S2-era storage classes).
        # resident_row_metadata keeps the array, its packing and its reads in
        # the dense row loop and the friction projectors: row types, friction
        # parents and the patch-ring walks, chains of dependent loads that
        # only visit rows the launch staged.
        if not resident_row_metadata:
            snippet = re.sub(
                r"\s*// Low 3 bits[^\n]*\n\s*__shared__ int   s_meta_dense\[\d+\];\n",
                "\n",
                snippet,
            )
            snippet = re.sub(
                r"\s*int row_type = world_row_type\.data\[off_dense \+ i\];"
                r"\s*int row_parent = world_row_parent\.data\[off_dense \+ i\];"
                r"\s*s_meta_dense\[i\] = [^\n]*\n",
                "\n",
                snippet,
            )
            snippet = re.sub(
                r"int parent_idx = \(s_meta_dense\[i\] >> \d+\) - 1;",
                "int parent_idx = world_row_parent.data[off_dense + i];",
                snippet,
            )
            snippet = re.sub(
                r"\(\(s_meta_dense\[(\w+)\] >> \d+\) - 1\)",
                r"world_row_parent.data[off_dense + \1]",
                snippet,
            )
            snippet = re.sub(
                r"int row_type = s_meta_dense\[i\] & \d+;",
                "int row_type = world_row_type.data[off_dense + i];",
                snippet,
            )
        # The end-of-iteration limit scans keep the global row-type reads in
        # both variants, so resident_row_metadata changes only the row loop
        # and the friction projectors. Packed reads would be exact there too:
        # the scans run in row phases 0 and 2, which stage every dense row,
        # and 5, which stages [0, bounds[1]) and scans [0, bounds[0]).
        snippet = re.sub(
            r"\(s_meta_dense\[i\] & \d+\)",
            "(world_row_type.data[off_dense + i])",
            snippet,
        )
        if not resident_row_metadata and "s_meta_dense" in snippet:
            raise RuntimeError(
                "MF-GS shared-memory diet failed to stream s_meta_dense; "
                "the generated snippet still references the resident array"
            )

    if stream_mf_impulses:
        # One lane owns every global impulse access. Reads broadcast registers,
        # including friction-parent/sibling values; no memory fence or race
        # between independent lane loads and stores is needed. All these read
        # sites are in warp-uniform row/projector branches, not lane-0 branches.
        for statement in (
            "s_lam_mf[i] = mf_impulses.data[off_mf + i];",
            "mf_impulses.data[off_mf + i] = s_lam_mf[i];",
        ):
            if snippet.count(statement) != 1:
                raise RuntimeError("MF-GS impulse staging source changed")
            snippet = snippet.replace(statement, "")
        snippet = snippet.replace(
            f"    __shared__ float s_lam_mf[{M_MF}];",
            "    volatile float* g_lam_mf = (volatile float*)(&mf_impulses.data[off_mf]);",
        )
        snippet = re.sub(
            r"s_lam_mf\[([^\]]*)\](\s*=(?!=)\s*[^;]+;)",
            r"if (lane == 0) g_lam_mf[\1]\2",
            snippet,
        )
        snippet = re.sub(
            r"s_lam_mf\[([^\]]*)\]",
            r"__shfl_sync(MASK, lane == 0 ? g_lam_mf[\1] : 0.0f, 0)",
            snippet,
        )

    if independent_components:
        # The paired owner's signed selector arrives in the otherwise unused
        # local-owner argument: negative means this world's matrix-free rows
        # are the free body's alone, so the sweep reads and writes only its
        # six coordinates and leaves the articulation's to the paired kernel.
        replacements = {
            "int m_dense = world_constraint_count.data[world];": "const bool split_component = local_solve_owner.data[world] < 0;\n"
            "    int m_dense = split_component ? 0 : world_constraint_count.data[world];",
            "int off_meta = off_mf * 4;": "int off_meta = off_mf * 4;\n"
            "    const int first_dofs = split_component ? mf_meta.data[off_meta] : 0;\n"
            "    const int component_offset = split_component ? max(first_dofs >> 16, static_cast<int>(static_cast<short>(first_dofs & 65535))) : 0;",
            "s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;": "s_v[d] = global_dof >= 0 && (!split_component || (d >= component_offset && d < component_offset + 6)) "
            "? v_out.data[global_dof] : 0.0f;",
            "if (global_dof >= 0) v_out.data[global_dof] = s_v[d];": "if (global_dof >= 0 && (!split_component || (d >= component_offset && d < component_offset + 6))) "
            "v_out.data[global_dof] = s_v[d];",
        }
        for old, new in replacements.items():
            if snippet.count(old) != 1:
                raise RuntimeError("Independent MF ownership source seam changed")
            snippet = snippet.replace(old, new)

    # Fixed signature (has_drive_rows precedent): world_drive_vel_limit is
    # always a parameter; launch sites pass the solver's drive_vel_limit
    # array — a (1, 1) dummy when the fused clamp is off — and
    # fuse_vel_limits only gates the clamp code emission above.

    snippet = cuda_native(snippet, preludes=(FRICTION_PAIR, *solve_policy.preludes))
    return GeneratedSource(snippet, mf_phase if held_policy is not None else None, arena_offset if shared_arena else 0)

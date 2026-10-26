# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate the native source of the matrix-free Gauss-Seidel sweep."""

import re

from .contact_torsion import torque_sweep_source
from .device_preludes import RING, cuda_native, lane_group
from .kernels import (
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
)
from .law_source import law_preludes
from .projection_laws import COULOMB_LAW, PROJECTION_LAW
from .row_metadata import (
    _DENSE_META_ROW_TYPE_BITS,
    _DENSE_META_ROW_TYPE_MASK,
    _mf_parent_expression,
    _validate_dense_metadata_encoding,
)
from .solve_policy import EXACT_POLICY, SolvePolicy, SolvePolicyUnsupported, sweep_exit, vote
from .velocity_limit_law import check_velocity_limit_law


def get_source(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    friction_mode: str = "current",
    *,
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
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> str:
    """Return the native body of the matrix-free Gauss-Seidel sweep.

    ``resident_row_metadata`` keeps the packed dense row types and parents
    resident in a streamed-metadata (``shared_metadata=False``) source.

    ``velocity_limit_law`` selects how every visit of a velocity-limit row projects it
    (:mod:`.velocity_limit_law`): the end-of-iteration dense and matrix-free passes and
    the dense rows of a velocity-limit row phase. The stateless law keeps the source
    byte-identical.

    ``solve_policy`` compiles the sweep's stopping rule and readout.
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
    if (not solve_policy.exact or solve_policy.traces) and (friction_mode != "current" or contact_torsion):
        raise SolvePolicyUnsupported(
            "The velocity exit law and trace cover the current-friction physical sweep without torsion"
        )
    _validate_dense_metadata_encoding(max_constraints)
    M_D = max_constraints
    M_MF = mf_max_constraints
    D = max_world_dofs

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
        terms=(("speed_row[1]", "1.0f / mf_diag", str(velocity_limit_type)),),
    )
    dense_vote = vote(
        solve_policy,
        "                iteration_changed = 1;",
        flag="iteration_changed",
        terms=(("delta_impulse", "denom", "row_type"),),
    )
    dense_rhs = "rhs_bias.data[off_dense + i]"
    velocity_load = "s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;"
    velocity_store = f"""    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0) v_out.data[global_dof] = s_v[d];
    }}"""

    def dense_read(array: str, index: str) -> str:
        return f"{array}.data[{index}]"

    def mf_velocity(index: str) -> str:
        return f"* s_v[{index}]"

    def shared(dtype: str, name: str, count: int, spacing: str) -> str:
        return f"    __shared__ {dtype}{spacing}{name}[{count}];"

    def dense_j_read(base: str, dof: str) -> str:
        return f"""{dense_read("J_world", f"{base} + {dof}")}"""

    factor_fallback_skip = "    if (m_mf == 0) return;" if factor_coordinates else ""

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
    if (m_mf == 0 && m_dense <= {int(skip_incremental_rows)}) return;"""
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
            if ({d_expr} < {D}) {{
                s_v[{d_expr}] += cur_dY_{k} * delta_impulse;
            }}""")
    dense_v_update_code = "\n".join(dense_v_update_parts)

    # Dense sibling v update — NOT pipelined (random sib index)
    dense_sib_v_parts = []
    for k in range(ELEMS_PER_LANE):
        d_expr = f"lane + {k * 32}" if k > 0 else "lane"
        dense_sib_v_parts.append(f"""
                if ({d_expr} < {D}) {{
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

                        const wp::vec3 lambda = CoulombLaw::coulomb_bisection(
                            wp::mat33(G_nn, G_nt1, G_nt2, G_nt1, G_t1t1, G_t1t2, G_nt2, G_t1t2, G_t2t2),
                            wp::vec3(u_n, u_t1, u_t2), wp::vec3(old_lambda_n, old_lambda_t1, old_lambda_t2), mu, target_vel_n);
                        new_lambda_n = lambda[0];
                        new_lambda_t1 = lambda[1];
                        new_lambda_t2 = lambda[2];
                        d_n_total = new_lambda_n - old_lambda_n;
                        d_t2_total = new_lambda_t2 - old_lambda_t2;
                        s_lam_dense[parent_idx] = new_lambda_n;
                        s_lam_dense[i_t2] = new_lambda_t2;
                    }}

                    SYNC();
                    new_lambda_t1 = READ(new_lambda_t1, 0);
                    d_n_total = READ(d_n_total, 0);
                    d_t2_total = READ(d_t2_total, 0);
                    if (d_n_total != 0.0f || d_t2_total != 0.0f) iteration_changed = 1;

                    if (d_n_total != 0.0f) {{
                        for (int d = lane; d < __D__; d += 32) {{
                            s_v[d] += {dense_read("Y_world", "n_row_base + d")} * d_n_total;
                        }}
                    }}
                    if (d_t2_total != 0.0f) {{
                        for (int d = lane; d < __D__; d += 32) {{
                            s_v[d] += {dense_read("Y_world", "t2_row_base + d")} * d_t2_total;
                        }}
                    }}
                    SYNC();
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
                    const float lambda_n = RING_LOAD(s_lam_dense[parent_idx], parent_idx, {M_D},
                        [&](int patch_row) {{ return ((s_meta_dense[patch_row] >> __DENSE_META_ROW_TYPE_BITS__) - 1); }},
                        [&](int patch_row) {{ return s_lam_dense[patch_row]; }});
                    float radius = fmaxf(s_mu_dense[i] * lambda_n, 0.0f);
                    const float sibling_impulse = s_lam_dense[sib];
                    wp::vec2 pair(0.0f);
                    // Zero load gives a zero disk; avoid unused tangent reductions.
                    if (radius > 0.0f) {{
                        float sibling_residual = 0.0f;
                        {dense_sibling_dot_code}
                        SUM(sibling_residual);
                        sibling_residual += s_rhs_dense[sib];
                        float cross = 0.0f;
                        {dense_tangent_cross_code}
                        SUM(cross);
                        pair = ProjectionLaw::disk(ProjectionLaw::friction_pair(denom, cross, s_diag_dense[sib],
                            wp::vec2(residual, sibling_residual), wp::vec2(old_impulse, sibling_impulse), radius, omega), radius);
                    }}
                    new_impulse = pair[0];
                    float sib_delta = pair[1] - sibling_impulse;
                    SYNC();
                    if (lane == 0) s_lam_dense[sib] = pair[1];
                    SYNC();
{dense_sibling_vote}
                    {dense_sib_v_code}
                }}
"""

    dense_friction_block = dense_friction_block.replace("__DENSE_META_ROW_TYPE_BITS__", str(_DENSE_META_ROW_TYPE_BITS))
    # Both modes using the scalar dense disk law share this ownership rule;
    # coulomb_newton differs only in its MF projector below. The existing
    # end-of-row barrier publishes each scalar result to the next dense row.
    dense_scalar_owner = friction_mode in ("current", "coulomb_newton")
    dense_old_impulse = "READ(lane == 0 ? s_lam_dense[i] : 0.0f, 0)" if dense_scalar_owner else "s_lam_dense[i]"
    dense_impulse_store = "if (lane == 0) " if dense_scalar_owner else ""

    # --- MF friction-row projection -------------------------------------
    # ``friction_mode="current"`` solves the tangent block on its friction
    # disk at fixed normal load (matches :func:`friction_step_current`).
    # ``friction_mode="bisection"`` runs the RAISim bisection on λ_n
    # (``CoulombLaw::coulomb_bisection``, as :func:`friction_step_coulomb`).
    # ``friction_mode="bisection_desaxce"`` runs the same bisection
    # with the de Saxce max-dissipation bias (``μ · ‖c_T‖`` added to
    # the normal target velocity) — FPGS Friction Modes 6/13.
    # ``friction_mode="coulomb_newton"`` runs Gilles Daviet's scalar
    # bracketed-Newton on alpha (``CoulombLaw::coulomb_newton``) — FPGS
    # Friction Modes 7/13.  We inject one of these CUDA blocks into the MF phase;
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

                        const wp::vec3 lambda = CoulombLaw::coulomb_bisection(
                            wp::mat33(G_nn, G_nt1, G_nt2, G_nt1, G_t1t1, G_t1t2, G_nt2, G_t1t2, G_t2t2),
                            wp::vec3(u_n, u_t1, u_t2), wp::vec3(old_lambda_n, old_lambda_t1, old_lambda_t2), mu, target_vel_n);
                        new_lambda_n = lambda[0];
                        new_lambda_t1 = lambda[1];
                        new_lambda_t2 = lambda[2];
                        d_n_total = new_lambda_n - old_lambda_n;
                        d_t2_total = new_lambda_t2 - old_lambda_t2;
                        s_lam_mf[mf_par] = new_lambda_n;
                        s_lam_mf[i_t2] = new_lambda_t2;
                    }

                    SYNC();
                    new_lambda_t1 = READ(new_lambda_t1, 0);
                    d_n_total = READ(d_n_total, 0);
                    d_t2_total = READ(d_t2_total, 0);
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
                    SYNC();
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
                "target_vel_n = CoulombLaw::de_saxce_target(target_vel_n, wp::vec3(u_n, u_t1, u_t2), mu);",
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
        # Gilles Daviet's 1D Coulomb Newton (7/13), as
        # :func:`friction_step_coulomb` runs it; mirrors the
        # bisection block's structure — lane 0 runs the serial
        # scalar Newton, then broadcasts (new_t1, d_n, d_t2) to
        # every lane via ``READ`` so the v_out deltas for
        # the normal / t2 rows are applied cooperatively.
        mf_friction_block = """
                // friction_mode="coulomb_newton": Gilles Daviet's
                // scalar bracketed-Newton on alpha (CoulombLaw::coulomb_newton;
                // friction_step_coulomb documents the matrix-free row data).
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

                        const wp::vec3 lambda = CoulombLaw::coulomb_newton(
                            wp::mat33(G_nn, G_nt1, G_nt2, G_nt1, G_t1t1, G_t1t2, G_nt2, G_t1t2, G_t2t2),
                            wp::vec3(u_n, u_t1, u_t2), wp::vec3(old_lambda_n, old_lambda_t1, old_lambda_t2), mu, target_vel_n);
                        new_lambda_n = lambda[0];
                        new_lambda_t1 = lambda[1];
                        new_lambda_t2 = lambda[2];
                        d_n_total = new_lambda_n - old_lambda_n;
                        d_t2_total = new_lambda_t2 - old_lambda_t2;
                        s_lam_mf[mf_par] = new_lambda_n;
                        s_lam_mf[i_t2] = new_lambda_t2;
                    }

                    SYNC();
                    new_lambda_t1 = READ(new_lambda_t1, 0);
                    d_n_total = READ(d_n_total, 0);
                    d_t2_total = READ(d_t2_total, 0);
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
                    SYNC();
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
                    wp::vec2 pair(0.0f);
                    // Zero load gives a zero disk; avoid unused tangent reductions.
                    if (radius > 0.0f) {{
                        float sibling_residual = 0.0f;
                        if (lane < 6 && dof_a >= 0)
                            sibling_residual = mf_J_a.data[sib_mf6 + lane] {mf_velocity("dof_a + lane")};
                        if (lane >= 6 && lane < 12 && dof_b >= 0)
                            sibling_residual = mf_J_b.data[sib_mf6 + lane - 6] {mf_velocity("dof_b + lane - 6")};
                        SUM(sibling_residual);
                        sibling_residual += __int_as_float(mf_meta.data[off_meta + sib * 4 + 2]);
                        float inv_sib = __int_as_float(mf_meta.data[off_meta + sib * 4 + 1]);
                        float cross = 0.0f;
                        if (lane < 6 && dof_a >= 0)
                            cross = mf_J_a.data[mf6_base + i * 6 + lane] * mf_MiJt_a.data[sib_mf6 + lane];
                        if (lane >= 6 && lane < 12 && dof_b >= 0)
                            cross = mf_J_b.data[mf6_base + i * 6 + lane - 6] * mf_MiJt_b.data[sib_mf6 + lane - 6];
                        SUM(cross);
                        pair = ProjectionLaw::friction_pair(mf_diag > 0.0f ? 1.0f / mf_diag : 0.0f, cross, inv_sib > 0.0f ? 1.0f / inv_sib : 0.0f,
                            wp::vec2(residual, sibling_residual), wp::vec2(old_impulse, s_lam_mf[sib]), radius, omega);
                    }}
                    pair = ProjectionLaw::disk(pair, radius);
                    new_impulse = pair[0];
                    float sib_delta = pair[1] - s_lam_mf[sib];
                    s_lam_mf[sib] = pair[1];
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
                const float lambda_n = RING_LOAD(s_lam_mf[parent], parent, __M_MF__,
                    [&](int patch_row) { return mf_meta.data[off_meta + patch_row * 4 + 3] >> 16; },
                    [&](int patch_row) { return s_lam_mf[patch_row]; });
                mf_friction_radius = fmaxf(mf_row_mu.data[off_mf + i] * lambda_n, 0.0f);
                // A zero disk with no carried impulse cannot change velocity.
                if (mf_friction_radius == 0.0f && s_lam_mf[i] == 0.0f && s_lam_mf[i + 1] == 0.0f)
                    continue;
            }
""".replace("__M_MF__", str(M_MF))
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
                new_impulse = ProjectionLaw::physx_drive(old_impulse, jv, s_drive_imp_mul_dense[i],
                    s_drive_vel_mul_dense[i], s_drive_target_dense[i], s_drive_max_imp_dense[i]);
                delta_impulse = new_impulse - old_impulse;
            }"""
        if has_drive_rows
        else """
            if (row_type == 1) {
                continue;
            }"""
    )
    # Every visit of a velocity-limit row: the end-of-iteration dense and free-body passes and the
    # dense rows of a velocity-limit row phase (row_phase 5), under the law projection_laws.velocity_limit_update
    # names (stateless: the correction of the current overshoot; accumulated: a unilateral total). A free
    # body's velocity-limit row is a speed row (projection_laws.speed_row): its residual is the limit minus
    # the speed of the velocity it bounds, and its correction acts along that velocity.
    law = int(accumulated_limits)
    if accumulated_limits:
        # Lane 0 owns a dense total, as it owns the scalar dense rows' impulses. Every lane reads a free-body
        # total before lane 0 stores the new one: a plain read stays warp-uniform when streamed impulses
        # rewrite it into a lane-0 broadcast.
        dense_old_vlim, mf_old_vlim = "old_vlim", "old_mf_vlim"
        dense_total = "                const float old_vlim = READ(lane == 0 ? s_lam_dense[i] : 0.0f, 0);\n"
        mf_total = "                const float old_mf_vlim = s_lam_mf[i];\n"
        total_fence = "                SYNC();\n"
    else:
        # The stateless law stores the correction, not a total: it never reads the old one.
        dense_old_vlim = mf_old_vlim = "0.0f"
        dense_total = mf_total = total_fence = ""
    dense_velocity_limit_law = (
        f"{dense_total}"
        f"                const auto vlim = ProjectionLaw::velocity_limit_update({dense_old_vlim}, residual_vlim, residual_vlim / denom, {law});\n"
        "                const float delta_impulse_vlim = vlim.v1;\n"
        f"{total_fence}"
        "                if (lane == 0) s_lam_dense[i] = vlim.v0;"
    )
    mf_velocity_limit_law = (
        f"{mf_total}"
        f"                const wp::vec3 speed_row = ProjectionLaw::speed_row(speed_mf_vlim, __int_as_float(meta.z), mf_diag, {mf_old_vlim}, {law});\n"
        f"{total_fence}"
        "                if (lane == 0) s_lam_mf[i] = speed_row[0];"
    )
    dense_velocity_limit_row = f"""            // row_type 4=JOINT_VELOCITY_LIMIT: a signed Jacobian and target velocity, so residual < 0
            // means that side of the velocity box is violated; the unrelaxed velocity-limit law.
            else if (row_type == 4) {{
                const auto vlim = ProjectionLaw::velocity_limit_update(old_impulse, residual, residual / denom, {law});
                new_impulse = vlim.v0;
                delta_impulse = vlim.v1;
            }}"""
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
                SUM(my_sum_vlim);
                float jv_vlim = my_sum_vlim;

                float residual_vlim = jv_vlim + s_rhs_dense[i];
{dense_velocity_limit_law}

                if (delta_impulse_vlim != 0.0f) {{
{dense_limit_vote}
                    for (int d = lane; d < {D}; d += 32) {{
                        s_v[d] += {dense_read("Y_world", "row_base_vlim + d")} * delta_impulse_vlim;
                    }}
                }}
                SYNC();
            }}
        }}"""
        if has_dense_velocity_limit_rows
        else ""
    )

    # Fused stateless velocity clamp on driven DOFs: PhysX's PxClamp /
    # unitResponse math (packed s_meta_dense).
    # Runs where the dedicated velocity-limit
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
                SUM(my_sum_fvl);
                float jv_fvl = my_sum_fvl;

                if (fabsf(jv_fvl) > qdot_max_fvl) {{
                    const float delta_fvl = ProjectionLaw::velocity_clamp_impulse(jv_fvl, qdot_max_fvl, denom_fvl);
{drive_limit_vote}
                    for (int d = lane; d < {D}; d += 32) {{
                        s_v[d] += {dense_read("Y_world", "row_base_fvl + d")} * delta_fvl;
                    }}
                }}
                SYNC();
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
            if (ProjectionLaw::friction_gate(mf_rt, global_iter, friction_start_iteration)) {{
                s_lam_mf[i] = 0.0f;
                SYNC();
                continue;
            }}

            if (mf_rt == 2 && i != (packed_tp >> 16) + 1) continue;
            if (mf_diag <= 0.0f{" && mf_rt != 2" if friction_mode == "current" else ""}) continue;

            {mf_friction_precheck}

            // J · v using prefetched J values
            float my_sum = 0.0f;
            if (lane < 6 && dof_a >= 0) {{
                my_sum = cur_Ja {mf_velocity("dof_a + lane")};
            }}
            if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                my_sum = cur_Jb {mf_velocity("dof_b + lane - 6")};
            }}
            SUM(my_sum);
            float jv = my_sum;

            float residual = jv + __int_as_float(meta.z);
            float old_impulse = s_lam_mf[i];
            float delta = -residual * mf_diag;
            // A contact row's regularization weight makes indeterminate normal-force splits unique.
            if (mf_rt == 0 && regularize)
                delta = ProjectionLaw::regularized_delta(delta, mf_row_w.data[off_mf + i], old_impulse);
            float new_impulse = ProjectionLaw::relaxed_step(old_impulse, delta, omega);

            if (mf_rt == 0) {{
                new_impulse = ProjectionLaw::unilateral(new_impulse);
            }} else if (mf_rt == 2) {{
                {mf_friction_block}
            }}

            float delta_impulse = new_impulse - old_impulse;
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
            SYNC();
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
                float mf_diag = __int_as_float(meta.y);
                if (mf_diag <= 0.0f || dof_a < 0) continue;

                // A free body's speed row bounds the magnitude of its linear (J = e_0) or angular
                // (J = e_3) velocity: lanes 0-2 hold its components.
                int row_mf6 = mf6_base + i * 6;
                const int speed_dof = dof_a + (mf_J_a.data[row_mf6 + 3] != 0.0f ? 3 : 0);
                const float speed_component = lane < 3 ? s_v[speed_dof + lane] : 0.0f;
                const float speed_sq = speed_component * speed_component;
                // In lane order, (s0 + s1) + s2: not the group's widest-first SUM.
                const float speed_mf_vlim = sqrtf((READ(speed_sq, 0) + READ(speed_sq, 1)) + READ(speed_sq, 2));

{mf_velocity_limit_law}

                if (speed_row[2] != 0.0f) {{
{mf_limit_vote}
                    // The correction acts along the velocity: it removes the overshoot, keeping the direction.
                    if (lane < 3) {{
                        s_v[speed_dof + lane] += speed_row[2] * speed_component;
                    }}
                }}
                SYNC();
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
            unsigned changed = BALLOT(iteration_changed != 0);
            if (changed == 0u) break;
        }
""",
        gate="global_iter >= friction_start_iteration",
        quiet="BALLOT(iteration_changed != 0) == 0u",
        iteration="iter",
    )
    snippet = f"""    const int lane = LANE;

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
    SYNC();

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
            if (ProjectionLaw::friction_gate(row_type, global_iter, friction_start_iteration)) {{
                {dense_impulse_store}s_lam_dense[i] = 0.0f;
                SYNC();
                continue;
            }}

            float denom = s_diag_dense[i];
            if (denom <= 0.0f{" && row_type != 2" if friction_mode == "current" else ""}) continue;

            // J_i · v (using prefetched J)
            {dense_dot_code}

            // Warp reduce
            SUM(my_sum);
            float jv = my_sum;

            float old_impulse = {dense_old_impulse};
            float residual = jv + s_rhs_dense[i];
            float delta = denom > 0.0f ? -residual / denom : 0.0f;
            if (regularize && denom > 0.0f)
                delta = ProjectionLaw::regularized_delta(delta, world_row_w.data[off_dense + i], old_impulse);
            float new_impulse = ProjectionLaw::relaxed_step(old_impulse, delta, omega);
            float delta_impulse = 0.0f;

            // row_type 1=JOINT_TARGET: PhysX PGS force-drive row.
            // computeDriveImpulse(lambda, jointVel, jointDeltaPos=0,
            // elapsedTime=0, driveDesc), then clamp to maxForce * dt.
{drive_projection}
{dense_velocity_limit_row}
            // row_type 0=CONTACT, 3=JOINT_LIMIT: unilateral lambda >= 0
            // projector with accumulated impulse.
            else if (row_type == 0 || row_type == 3) {{
                new_impulse = ProjectionLaw::unilateral(new_impulse);
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
            SYNC();
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
#endif
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
            r"READ(lane == 0 ? g_lam_mf[\1] : 0.0f, 0)",
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

    # The research friction modes' Coulomb laws compile only into the sweeps that run them.
    laws = law_preludes(PROJECTION_LAW, *((COULOMB_LAW,) if friction_mode != "current" else ()))
    snippet = cuda_native(snippet, preludes=(lane_group(32), RING, *laws, *solve_policy.preludes))
    return snippet

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Response-block Gauss-Seidel sweep for the FeatherPGS matrix-free path.

Three kernel kinds replace one launch of the warp-per-world ``pgs_solve_mf_gs`` kernel for
worlds whose dense plus matrix-free row count fits a compile-time budget:

1. ``build`` (warp per world): forms the response block ``A[r][c] = J_r . Y_c`` over the
   world's rows (dense articulated rows followed by matrix-free rigid rows, all expressed on
   the same world velocity vector), the initial residual ``r = rhs + J v``, the inverse
   diagonal and the per-row projection metadata, and appends the world to a class list:
   lane class (``m <= lane_rows``, bucketed by row count), 32-row warp class, 64-row warp
   class. Worlds above 64 rows, with deferred dense response or outside the general solve
   queue are left to the legacy kernel.
2. ``sweep_lane`` (lane per world): each lane runs its world's sequential sweep in the legacy
   row order (dense rows, matrix-free contact rows, dense velocity-limit rows, matrix-free
   velocity-limit rows) with the packed symmetric response block in shared memory, then
   reconstructs ``v`` and writes the impulses.
3. ``sweep_warp`` (warp per world, persistent grid over the class list): each lane owns one
   or two rows with their response rows in registers; each row update is one shuffle plus
   the friction parent/sibling reads.

Row semantics (types, phases, friction gating, regularization, PhysX drive and
velocity-limit projections, stationary-sweep exit) follow ``pgs_solve_mf_gs``; only the
order of floating-point operations differs.
"""

from __future__ import annotations

import warp as wp

from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
)

# Per-row vector slots in the blob (row-major per world, stride RB_STRIDE).
RB_V_RHS = 0
RB_V_INVD = 1
RB_V_W = 2
RB_V_MU = 3
RB_V_LAM = 4
RB_V_RES = 5
RB_V_APP = 6
RB_V_DRV_T = 7
RB_V_DRV_VM = 8
RB_V_DRV_IM = 9
RB_V_DRV_MAX = 10
RB_NV = 11
# Per-row int slots.
RB_M_KIND = 0
RB_M_PARENT = 1
RB_M_DOFS = 2
RB_NM = 3
# World class.
RB_CLASS_NONE = 0
RB_CLASS_LANE = 1
RB_CLASS_WARP32 = 2
RB_CLASS_WARP64 = 3
# Row-major blob stride and per-world response-block slot.
RB_STRIDE = 64
RB_A_SLOT = RB_STRIDE * RB_STRIDE

# Kind packing: bits 0-2 projection kind, bit 3 matrix-free row, bit 4 first friction row of
# its triple, bit 5 matrix-free contact-section row, bits 8-11 raw row type.
_K_UNI = 0
_K_FRIC = 1
_K_VLIM = 2
_K_DRIVE = 3
_K_BILAT = 4
_KB_MF = 1 << 3
_KB_FIRST = 1 << 4
_KB_MFCONTACT = 1 << 5
_KB_TYPE_SHIFT = 8

_T_CONTACT = int(PGS_CONSTRAINT_TYPE_CONTACT)
_T_DRIVE = int(PGS_CONSTRAINT_TYPE_JOINT_TARGET)
_T_FRICTION = int(PGS_CONSTRAINT_TYPE_FRICTION)
_T_LIMIT = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
_T_VLIM = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)


def list_count_slots(lane_rows: int) -> int:
    """Number of counters: one per lane bucket (m = 1..lane_rows), then warp32, warp64."""
    return int(lane_rows) + 2


# The admission predicate, emitted inside the native bodies as a lambda. ``main`` selects the
# main-loop pass, otherwise the velocity-limit pass. ``r`` is the unified row index.
_ADMIT_LAMBDA = f"""
    auto rb_admit = [](int kind, int r, int m_dense, int dense_lo, int dense_hi, int mf_contact_end,
                       int row_phase, int freeze_drive_rows, int has_drive_rows, bool main) -> bool {{
        const int raw = (kind >> {_KB_TYPE_SHIFT}) & 0xF;
        const bool is_mf = (kind & {_KB_MF}) != 0;
        if (!is_mf) {{
            if (main) {{
                if (r < dense_lo || r >= dense_hi) return false;
                if (row_phase == 1 && raw != {_T_CONTACT} && raw != {_T_FRICTION}) return false;
                if (row_phase == 2 && raw != {_T_DRIVE} && raw != {_T_LIMIT} && raw != {_T_VLIM} && raw != 5 && raw != 6) return false;
                if (row_phase == 3 && raw != {_T_DRIVE} && raw != {_T_LIMIT} && raw != 5 && raw != 6) return false;
                if (row_phase == 4 && raw != {_T_CONTACT} && raw != {_T_FRICTION}) return false;
                if (row_phase == 5 && raw != {_T_VLIM}) return false;
                if ((row_phase == 0 || row_phase == 2) && raw == {_T_VLIM}) return false;
                if (raw == {_T_DRIVE} && (freeze_drive_rows != 0 || has_drive_rows == 0)) return false;
                return true;
            }}
            return raw == {_T_VLIM} && (row_phase == 0 || row_phase == 2);
        }}
        const bool contact = (kind & {_KB_MFCONTACT}) != 0;
        if (main) {{
            if (!contact) return false;
            if (row_phase != 0 && row_phase != 1 && row_phase != 4) return false;
            if ((row_phase == 1 || row_phase == 4) && raw != {_T_CONTACT} && raw != {_T_FRICTION}) return false;
            if (row_phase == 0 && raw == {_T_VLIM}) return false;
            return true;
        }}
        return !contact && raw == {_T_VLIM} && (row_phase == 0 || row_phase == 2 || row_phase == 5);
    }};
"""


def _phase_bounds_code() -> str:
    return """
    int dense_lo = 0;
    int dense_hi = m_dense;
    if (row_phase == 3) {
        dense_hi = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
    } else if (row_phase == 5) {
        dense_lo = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
        dense_hi = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    } else if (row_phase == 4) {
        dense_lo = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build kernel
# ─────────────────────────────────────────────────────────────────────────────


def get_response_block_build_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
    lane_rows: int,
    has_drive_rows: bool,
    skip_local_internal_worlds: bool = False,
) -> wp.Kernel:
    M_D = int(max_constraints)
    M_MF = int(mf_max_constraints)
    D = int(max_world_dofs)
    RS = int(lane_rows)
    ST = RB_STRIDE
    if RS < 0 or RS > 32:
        raise ValueError("lane_rows must be in [0, 32]")
    drive_loads = (
        """
            drv_t = world_drive_target_vel_bias.data[off_dense + r];
            drv_vm = world_drive_vel_multiplier.data[off_dense + r];
            drv_im = world_drive_impulse_multiplier.data[off_dense + r];
            drv_max = world_drive_max_impulse.data[off_dense + r];"""
        if has_drive_rows
        else ""
    )
    owner_check = (
        f"ok = ok && local_solve_owner.data[world] == {PGS_LOCAL_SOLVE_OWNER_GENERAL};"
        if skip_local_internal_worlds
        else ""
    )
    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
    const int m_dense = world_constraint_count.data[world];
    const int m_mf = mf_constraint_count.data[world];
    const int m_tot = m_dense + m_mf;
    bool ok = m_tot > 0 && m_tot <= {ST} && defer_dense_response == 0;
    {owner_check}
    if (!ok) {{
        if (lane == 0) rb_class.data[world] = {RB_CLASS_NONE};
        return;
    }}
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;
    const int cls = (m_tot <= {RS}) ? {RB_CLASS_LANE} : ((m_tot <= 32) ? {RB_CLASS_WARP32} : {RB_CLASS_WARP64});
    const int a_stride = (cls == {RB_CLASS_LANE}) ? {RS} : ((cls == {RB_CLASS_WARP32}) ? 32 : 64);
    if (lane == 0) {{
        rb_class.data[world] = cls;
        const int slot = (cls == {RB_CLASS_LANE}) ? (m_tot - 1) : ((cls == {RB_CLASS_WARP32}) ? {RS} : {RS} + 1);
        const int idx = atomicAdd(&rb_counts.data[slot], 1);
        rb_lists.data[slot * world_count + idx] = world;
    }}

    const int off_dense = world * {M_D};
    const int off_mf = world * {M_MF};
    const int off_meta = off_mf * 4;
    const int jy_world_base = world * {M_D} * {D};
    const int mf6_base = world * {M_MF} * 6;
    const int dof_map_base = world * {D};
    const int vb = world * {RB_NV} * {ST};
    const int mb = world * {RB_NM} * {ST};
    const int a_base = world * {RB_A_SLOT};

    __shared__ float s_J[{ST} * {D}];
    __shared__ float s_v[{D}];

    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_in.data[global_dof] : 0.0f;
    }}
    // Every row's Jacobian on the world velocity vector. Dense rows: one flat coalesced
    // copy (independent loads, pipelined). Matrix-free rows: two 6-blocks.
    for (int e = lane; e < m_dense * {D}; e += 32) {{
        s_J[e] = J_world.data[jy_world_base + e];
    }}
    for (int e = lane; e < m_mf * {D}; e += 32) {{
        s_J[m_dense * {D} + e] = 0.0f;
    }}
    __syncwarp();
    for (int j = 0; j < m_mf; ++j) {{
        const int r = m_dense + j;
        const int packed_dofs = mf_meta.data[off_meta + j * 4];
        const int dof_a = packed_dofs >> 16;
        const int dof_b = (packed_dofs << 16) >> 16;
        const int mf6 = mf6_base + j * 6;
        if (lane < 6 && dof_a >= 0) s_J[r * {D} + dof_a + lane] = mf_J_a.data[mf6 + lane];
        if (lane >= 6 && lane < 12 && dof_b >= 0) s_J[r * {D} + dof_b + lane - 6] = mf_J_b.data[mf6 + lane - 6];
    }}
    __syncwarp();

    // Per-row scalars and metadata (lane r handles rows r, r + 32).
    for (int r = lane; r < m_tot; r += 32) {{
        float rhs, invd, w, mu, lam;
        float drv_t = 0.0f, drv_vm = 0.0f, drv_im = 0.0f, drv_max = 0.0f;
        int kind, parent, dofs = 0;
        if (r < m_dense) {{
            rhs = rhs_bias.data[off_dense + r];
            const float diag = world_diag.data[off_dense + r];
            invd = diag > 0.0f ? 1.0f / diag : 0.0f;
            w = regularize ? world_row_w.data[off_dense + r] : 1.0f;
            mu = world_row_mu.data[off_dense + r];
            lam = world_impulses.data[off_dense + r];
            const int raw = world_row_type.data[off_dense + r] & 7;
            parent = world_row_parent.data[off_dense + r];
            int k = {_K_BILAT};
            if (raw == {_T_CONTACT} || raw == {_T_LIMIT}) k = {_K_UNI};
            else if (raw == {_T_FRICTION}) k = {_K_FRIC};
            else if (raw == {_T_VLIM}) k = {_K_VLIM};
            else if (raw == {_T_DRIVE}) k = {_K_DRIVE};
            kind = k | (raw << {_KB_TYPE_SHIFT});
            if (k == {_K_FRIC} && r == parent + 1) kind |= {_KB_FIRST};{drive_loads}
        }} else {{
            const int j = r - m_dense;
            const int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + j * 4]);
            dofs = meta.x;
            invd = __int_as_float(meta.y);
            if (invd <= 0.0f) invd = 0.0f;
            rhs = __int_as_float(meta.z);
            const int raw = meta.w & 0xFFFF;
            const int mf_par = meta.w >> 16;
            parent = (raw == {_T_FRICTION}) ? (m_dense + mf_par) : -1;
            w = (raw == {_T_CONTACT} && regularize) ? mf_row_w.data[off_mf + j] : 1.0f;
            mu = mf_row_mu.data[off_mf + j];
            lam = mf_impulses.data[off_mf + j];
            int k = {_K_BILAT};
            if (raw == {_T_CONTACT}) k = {_K_UNI};
            else if (raw == {_T_FRICTION}) k = {_K_FRIC};
            else if (raw == {_T_VLIM}) k = {_K_VLIM};
            kind = k | {_KB_MF} | (raw << {_KB_TYPE_SHIFT});
            if (k == {_K_FRIC} && j == mf_par + 1) kind |= {_KB_FIRST};
            if (j < mf_contact_end) kind |= {_KB_MFCONTACT};
        }}
        // Residual r_r = rhs_r + J_r . v.
        float res = rhs;
        for (int d = 0; d < {D}; ++d) res += s_J[r * {D} + d] * s_v[d];
        rb_vec.data[vb + {RB_V_RHS} * {ST} + r] = rhs;
        rb_vec.data[vb + {RB_V_INVD} * {ST} + r] = invd;
        rb_vec.data[vb + {RB_V_W} * {ST} + r] = w;
        rb_vec.data[vb + {RB_V_MU} * {ST} + r] = mu;
        rb_vec.data[vb + {RB_V_LAM} * {ST} + r] = lam;
        rb_vec.data[vb + {RB_V_RES} * {ST} + r] = res;
        rb_vec.data[vb + {RB_V_APP} * {ST} + r] = 0.0f;
        rb_vec.data[vb + {RB_V_DRV_T} * {ST} + r] = drv_t;
        rb_vec.data[vb + {RB_V_DRV_VM} * {ST} + r] = drv_vm;
        rb_vec.data[vb + {RB_V_DRV_IM} * {ST} + r] = drv_im;
        rb_vec.data[vb + {RB_V_DRV_MAX} * {ST} + r] = drv_max;
        rb_meta.data[mb + {RB_M_KIND} * {ST} + r] = kind;
        rb_meta.data[mb + {RB_M_PARENT} * {ST} + r] = parent;
        rb_meta.data[mb + {RB_M_DOFS} * {ST} + r] = dofs;
    }}

    // A[r][c] = J_r . Y_c with lane-owned columns c = lane, lane + 32 (Y_c in registers,
    // J_r broadcast from shared).
    {{
        float yc0[{D}], yc1[{D}];
        const int c0 = lane, c1 = lane + 32;
        #pragma unroll
        for (int d = 0; d < {D}; ++d) {{ yc0[d] = 0.0f; yc1[d] = 0.0f; }}
        for (int h = 0; h < 2; ++h) {{
            const int c = (h == 0) ? c0 : c1;
            if (c >= m_tot) continue;
            if (c < m_dense) {{
                const int base = jy_world_base + c * {D};
                #pragma unroll
                for (int d = 0; d < {D}; ++d) {{
                    const float y = Y_world.data[base + d];
                    if (h == 0) yc0[d] = y; else yc1[d] = y;
                }}
            }} else {{
                const int j = c - m_dense;
                const int packed_dofs = mf_meta.data[off_meta + j * 4];
                const int dof_a = packed_dofs >> 16;
                const int dof_b = (packed_dofs << 16) >> 16;
                const int mf6 = mf6_base + j * 6;
                #pragma unroll
                for (int d = 0; d < {D}; ++d) {{
                    float y = 0.0f;
                    if (dof_a >= 0 && d >= dof_a && d < dof_a + 6) y = mf_MiJt_a.data[mf6 + d - dof_a];
                    if (dof_b >= 0 && d >= dof_b && d < dof_b + 6) y = mf_MiJt_b.data[mf6 + d - dof_b];
                    if (h == 0) yc0[d] = y; else yc1[d] = y;
                }}
            }}
        }}
        for (int r = 0; r < m_tot; ++r) {{
            float acc0 = 0.0f, acc1 = 0.0f;
            #pragma unroll
            for (int d = 0; d < {D}; ++d) {{
                const float jd = s_J[r * {D} + d];
                acc0 += jd * yc0[d];
                acc1 += jd * yc1[d];
            }}
            if (cls == {RB_CLASS_LANE}) {{
                // Symmetric block, packed lower triangle (r >= c).
                if (c0 <= r) rb_A.data[a_base + r * (r + 1) / 2 + c0] = acc0;
            }} else {{
                if (c0 < m_tot) rb_A.data[a_base + r * a_stride + c0] = acc0;
                if (c1 < m_tot) rb_A.data[a_base + r * a_stride + c1] = acc1;
            }}
        }}
    }}
#endif
"""

    @wp.func_native(snippet)
    def rb_build_native(
        world: int,
        world_count: int,
        world_constraint_count: wp.array[int],
        local_solve_owner: wp.array[int],
        world_dof_indices: wp.array2d[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_row_w: wp.array2d[float],
        world_impulses: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_drive_target_vel_bias: wp.array2d[float],
        world_drive_vel_multiplier: wp.array2d[float],
        world_drive_impulse_multiplier: wp.array2d[float],
        world_drive_max_impulse: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        regularize: int,
        defer_dense_response: int,
        v_in: wp.array[float],
        rb_class: wp.array[int],
        rb_counts: wp.array[int],
        rb_lists: wp.array[int],
        rb_A: wp.array[float],
        rb_vec: wp.array[float],
        rb_meta: wp.array[int],
    ): ...

    def rb_build_template(
        general_world_count: wp.array[int],
        general_worlds: wp.array[int],
        general_world_grid_stride: int,
        use_general_world_queue: int,
        world_count: int,
        world_constraint_count: wp.array[int],
        local_solve_owner: wp.array[int],
        world_dof_indices: wp.array2d[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_row_w: wp.array2d[float],
        world_impulses: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_drive_target_vel_bias: wp.array2d[float],
        world_drive_vel_multiplier: wp.array2d[float],
        world_drive_impulse_multiplier: wp.array2d[float],
        world_drive_max_impulse: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        regularize: int,
        defer_dense_response: int,
        v_in: wp.array[float],
        rb_class: wp.array[int],
        rb_counts: wp.array[int],
        rb_lists: wp.array[int],
        rb_A: wp.array[float],
        rb_vec: wp.array[float],
        rb_meta: wp.array[int],
    ):
        # Persistent grid: block ``candidate`` strides over the candidate worlds.
        candidate, _lane = wp.tid()
        general_index = candidate
        general_count = general_world_count[0]
        if use_general_world_queue == 0:
            general_count = world_count
        while general_index < general_count:
            world = general_index
            if use_general_world_queue != 0:
                world = general_worlds[general_index]
            rb_build_native(
                world,
                world_count,
                world_constraint_count,
                local_solve_owner,
                world_dof_indices,
                rhs_bias,
                world_diag,
                world_row_w,
                world_impulses,
                J_world,
                Y_world,
                world_row_type,
                world_row_parent,
                world_row_mu,
                world_drive_target_vel_bias,
                world_drive_vel_multiplier,
                world_drive_impulse_multiplier,
                world_drive_max_impulse,
                mf_constraint_count,
                mf_contact_rows_end,
                mf_meta,
                mf_impulses,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_row_mu,
                mf_row_w,
                regularize,
                defer_dense_response,
                v_in,
                rb_class,
                rb_counts,
                rb_lists,
                rb_A,
                rb_vec,
                rb_meta,
            )
            general_index += general_world_grid_stride

    name = f"pgs_rb_build_{M_D}_{M_MF}_{D}_lane{RS}_drive{int(has_drive_rows)}"
    if skip_local_internal_worlds:
        name += "_local_fallback"
    rb_build_template.__name__ = name
    rb_build_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(rb_build_template)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lane-per-world sweep (packed symmetric response block in shared memory)
# ─────────────────────────────────────────────────────────────────────────────


def _pk(r: int, c: int) -> int:
    """Packed lower-triangle index of the symmetric response block."""
    a, b = (r, c) if r >= c else (c, r)
    return a * (a + 1) // 2 + b


def _lane_res_update(RS: int, col: int, delta: str) -> str:
    return " ".join(f"res[{j}] += sA[{_pk(j, col)}][lane] * {delta};" for j in range(RS))


def _lane_row_code(i: int, RS: int, has_drive_rows: bool) -> str:
    """Branch-free row update for unified row ``i`` (lane per world, compile-time indices).

    Row-uniform flags (``u_*``) are ballots taken once before the sweeps; inside a row every
    lane runs the same straight-line code with per-lane selects, so the shared loads of the
    residual update can be scheduled early and no divergence barriers are emitted.
    """
    par_first = f"lam[{i - 1}]" if i >= 1 else "0.0f"
    par_second = f"lam[{i - 2}]" if i >= 2 else "0.0f"
    sib_first = i + 1 if i + 1 < RS else None
    sib_second = i - 1 if i >= 1 else None
    sib_first_expr = f"lam[{sib_first}]" if sib_first is not None else "0.0f"
    sib_second_expr = f"lam[{sib_second}]" if sib_second is not None else "0.0f"
    apply_first = (
        f"""
                    if ((u_first >> {i}) & 1u) {{
                        const float sd = first ? sib_delta : 0.0f;
                        lam[{sib_first}] = (sd != 0.0f) ? sib_new : lam[{sib_first}];
                        app[{sib_first}] += sd;
                        {_lane_res_update(RS, sib_first, "sd")}
                    }}"""
        if sib_first is not None
        else ""
    )
    apply_second = (
        f"""
                    if ((u_second >> {i}) & 1u) {{
                        const float sd = first ? 0.0f : sib_delta;
                        lam[{sib_second}] = (sd != 0.0f) ? sib_new : lam[{sib_second}];
                        app[{sib_second}] += sd;
                        {_lane_res_update(RS, sib_second, "sd")}
                    }}"""
        if sib_second is not None
        else ""
    )
    drive = (
        f"""
            if ((u_drive >> {i}) & 1u) {{
                const float rhs = rb_vec.data[vb + {RB_V_RHS} * {RB_STRIDE} + {i}];
                float drive_impulse = old_impulse * rb_vec.data[vb + {RB_V_DRV_IM} * {RB_STRIDE} + {i}]
                    + (residual - rhs) * rb_vec.data[vb + {RB_V_DRV_VM} * {RB_STRIDE} + {i}]
                    + rb_vec.data[vb + {RB_V_DRV_T} * {RB_STRIDE} + {i}];
                const float max_imp = rb_vec.data[vb + {RB_V_DRV_MAX} * {RB_STRIDE} + {i}];
                drive_impulse = fminf(fmaxf(drive_impulse, -max_imp), max_imp);
                if (k == {_K_DRIVE}) {{ new_impulse = drive_impulse; delta_impulse = drive_impulse - old_impulse; }}
            }}"""
        if has_drive_rows
        else ""
    )
    return f"""
        if ((u_act >> {i}) & 1u) {{
            const bool act = (admit_main >> {i}) & 1u;
            const int kind = kind_{i};
            const int k = kind & 7;
            const float residual = res[{i}];
            const float old_impulse = lam[{i}];
            const float delta = -residual * ca[{i}] - cb[{i}] * old_impulse;
            const float relaxed = old_impulse + omega * delta;
            float new_impulse = (k == {_K_UNI}) ? fmaxf(relaxed, 0.0f) : relaxed;
            float delta_impulse = new_impulse - old_impulse;
            if (k == {_K_VLIM}) {{
                delta_impulse = (residual < 0.0f) ? delta : 0.0f;
                new_impulse = delta_impulse;
            }}{drive}
            float sib_new = 0.0f, sib_delta = 0.0f;
            const bool first = (kind & {_KB_FIRST}) != 0;
            if ((u_fric >> {i}) & 1u) {{
                const float lambda_n = first ? {par_first} : {par_second};
                const float radius = fmaxf(mu[{i}] * lambda_n, 0.0f);
                const float b_val = first ? {sib_first_expr} : {sib_second_expr};
                const float mag = sqrtf(relaxed * relaxed + b_val * b_val);
                const bool scale_it = (radius > 0.0f) && (mag > radius);
                const float scale = scale_it ? (radius / mag) : 1.0f;
                const float nf = (radius > 0.0f) ? relaxed * scale : 0.0f;
                const float sn = b_val * scale;
                const float sd = scale_it ? (sn - b_val) : 0.0f;
                if (k == {_K_FRIC}) {{
                    if (global_iter < friction_start_iteration) {{
                        new_impulse = 0.0f; delta_impulse = 0.0f;
                    }} else {{
                        new_impulse = nf; delta_impulse = nf - old_impulse; sib_new = sn; sib_delta = sd;
                    }}
                }}
            }}
            if (!act) {{ new_impulse = old_impulse; delta_impulse = 0.0f; sib_delta = 0.0f; }}
            lam[{i}] = new_impulse;
            app[{i}] += delta_impulse;
            changed |= (delta_impulse != 0.0f) | (sib_delta != 0.0f);
            {_lane_res_update(RS, i, "delta_impulse")}
            if ((u_fric >> {i}) & 1u) {{
                if (__ballot_sync(AMASK, sib_delta != 0.0f)) {{{apply_first}{apply_second}
                }}
            }}
        }}"""


def _lane_vlim_code(i: int, RS: int) -> str:
    return f"""
        if ((u_vlim >> {i}) & 1u) {{
            const bool act = (admit_vlim >> {i}) & 1u;
            const float residual = res[{i}];
            float d = (residual < 0.0f) ? -residual * ca[{i}] : 0.0f;
            if (!act) d = 0.0f;
            lam[{i}] = act ? d : lam[{i}];
            app[{i}] += d;
            changed |= (d != 0.0f);
            {_lane_res_update(RS, i, "d")}
        }}"""


def get_response_block_sweep_lane_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
    lane_rows: int,
    has_drive_rows: bool,
) -> wp.Kernel:
    """Lane-per-world sweep: 32 worlds per block, each lane runs its world's sequential sweep.

    Per-lane state lives in registers (rows are unrolled at codegen time) and the packed
    symmetric response block in shared memory, lane-minor. Row kind flags are warp-uniform
    ballots over the lanes that own a world; inside a row every lane runs the same
    straight-line code with per-lane selects.
    """
    M_D = int(max_constraints)
    M_MF = int(mf_max_constraints)
    D = int(max_world_dofs)
    RS = int(lane_rows)
    ST = RB_STRIDE
    if RS <= 0 or RS > 32:
        raise ValueError("lane_rows must be in [1, 32]")
    P = RS * (RS + 1) // 2
    hd = int(has_drive_rows)
    kind_loads = "\n".join(
        f"    const int kind_{i} = ({i} < m_tot) ? rb_meta.data[mb + {RB_M_KIND} * {ST} + {i}] : 0;" for i in range(RS)
    )
    admit_code = "\n".join(
        f"""    if ({i} < m_tot && ca[{i}] > 0.0f) {{
        if (rb_admit(kind_{i}, {i}, m_dense, dense_lo, dense_hi, mf_contact_end, row_phase, freeze_drive_rows, {hd}, true)) admit_main |= (1u << {i});
        if (rb_admit(kind_{i}, {i}, m_dense, dense_lo, dense_hi, mf_contact_end, row_phase, freeze_drive_rows, {hd}, false)) admit_vlim |= (1u << {i});
    }}"""
        for i in range(RS)
    )
    uniform_code = "\n".join(
        f"""    {{
        const bool a_ = (admit_main >> {i}) & 1u;
        const bool f_ = a_ && (kind_{i} & 7) == {_K_FRIC};
        if (__ballot_sync(AMASK, a_)) u_act |= (1u << {i});
        if (__ballot_sync(AMASK, f_)) u_fric |= (1u << {i});
        if (__ballot_sync(AMASK, f_ && (kind_{i} & {_KB_FIRST}) != 0)) u_first |= (1u << {i});
        if (__ballot_sync(AMASK, f_ && (kind_{i} & {_KB_FIRST}) == 0)) u_second |= (1u << {i});
        if (__ballot_sync(AMASK, a_ && (kind_{i} & 7) == {_K_DRIVE})) u_drive |= (1u << {i});
        if (__ballot_sync(AMASK, (admit_vlim >> {i}) & 1u)) u_vlim |= (1u << {i});
    }}"""
        for i in range(RS)
    )
    main_pass = "\n".join(_lane_row_code(i, RS, has_drive_rows) for i in range(RS))
    vlim_pass = "\n".join(_lane_vlim_code(i, RS) for i in range(RS))
    load_a_lines = []
    for r in range(RS):
        for c in range(r + 1):
            load_a_lines.append(f"    sA[{_pk(r, c)}][lane] = ({r} < m_tot) ? rb_A.data[a_base + {_pk(r, c)}] : 0.0f;")
    load_a = "\n".join(load_a_lines)
    load_vec = "\n".join(
        f"""    lam[{i}] = ({i} < m_tot) ? rb_vec.data[vb + {RB_V_LAM} * {ST} + {i}] : 0.0f;
    res[{i}] = ({i} < m_tot) ? rb_vec.data[vb + {RB_V_RES} * {ST} + {i}] : 0.0f;
    {{
        const float invd = ({i} < m_tot) ? rb_vec.data[vb + {RB_V_INVD} * {ST} + {i}] : 0.0f;
        const float w = (regularize && {i} < m_tot) ? rb_vec.data[vb + {RB_V_W} * {ST} + {i}] : 1.0f;
        ca[{i}] = invd * w;
        cb[{i}] = 1.0f - w;
        mu[{i}] = ({i} < m_tot) ? rb_vec.data[vb + {RB_V_MU} * {ST} + {i}] : 0.0f;
    }}
    app[{i}] = 0.0f;"""
        for i in range(RS)
    )
    sel_app = " ".join(f"if (r == {i}) a = app[{i}];" for i in range(RS))
    sel_lam = " ".join(f"if (r == {i}) l = lam[{i}];" for i in range(RS))
    snippet = f"""
#if defined(__CUDA_ARCH__)
{_ADMIT_LAMBDA}
    const unsigned MASK = 0xFFFFFFFF;
    const int lane = threadIdx.x;
    __shared__ float sA[{P}][32];
    // Lane -> world through the bucketed lane-class lists (bucket k holds worlds with k+1 rows).
    int world = -1;
    {{
        const int g = block * 32 + lane;
        int acc = 0;
        for (int k = 0; k < {RS}; ++k) {{
            const int n = rb_counts.data[k];
            if (world < 0 && g >= acc && g < acc + n) world = rb_lists.data[k * world_count + (g - acc)];
            acc += n;
        }}
        if (block * 32 >= acc) return;  // whole block past the end (uniform)
    }}
    // Lanes past the end of the lists leave; every warp-sync below uses the mask of the lanes
    // that stay (all lanes are present when the mask is taken).
    const unsigned AMASK = __ballot_sync(MASK, world >= 0);
    if (world < 0) return;
    const int m_dense = world_constraint_count.data[world];
    const int m_mf = mf_constraint_count.data[world];
    const int m_tot = m_dense + m_mf;
{_phase_bounds_code()}
    const int vb = world * {RB_NV} * {ST};
    const int mb = world * {RB_NM} * {ST};
    const int a_base = world * {RB_A_SLOT};
{load_a}
    float lam[{RS}], res[{RS}], ca[{RS}], cb[{RS}], mu[{RS}], app[{RS}];
{load_vec}
{kind_loads}
    unsigned admit_main = 0u, admit_vlim = 0u;
{admit_code}
    unsigned u_act = 0u, u_fric = 0u, u_first = 0u, u_second = 0u, u_drive = 0u, u_vlim = 0u;
{uniform_code}
    for (int iter = 0; iter < iterations; ++iter) {{
        const int global_iter = iteration_offset + iter;
        int changed = 0;
{main_pass}
{vlim_pass}
        // Stationary worlds stop contributing; the warp leaves when every lane is stationary.
        const bool stop = (global_iter >= friction_start_iteration) && (changed == 0);
        if (__ballot_sync(AMASK, !stop) == 0u) break;
        if (stop) admit_main = 0u, admit_vlim = 0u;
    }}
    // v = v_in + sum_r Y_r * app_r ; impulses back to the solver arrays.
    const int dof_map_base = world * {D};
    const int jy_world_base = world * {M_D} * {D};
    const int off_dense = world * {M_D};
    const int off_mf = world * {M_MF};
    const int mf6_base = world * {M_MF} * 6;
    for (int d = 0; d < {D}; ++d) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof < 0) continue;
        float v = v_out.data[global_dof];
        for (int r = 0; r < m_dense; ++r) {{
            float a = 0.0f;
            {sel_app}
            if (a != 0.0f) v += Y_world.data[jy_world_base + r * {D} + d] * a;
        }}
        v_out.data[global_dof] = v;
    }}
    for (int j = 0; j < m_mf; ++j) {{
        const int r = m_dense + j;
        float a = 0.0f;
        {sel_app}
        if (a == 0.0f) continue;
        const int packed_dofs = rb_meta.data[mb + {RB_M_DOFS} * {ST} + r];
        const int dof_a = packed_dofs >> 16;
        const int dof_b = (packed_dofs << 16) >> 16;
        const int mf6 = mf6_base + j * 6;
        for (int k = 0; k < 6; ++k) {{
            if (dof_a >= 0) {{
                const int gd = world_dof_indices.data[dof_map_base + dof_a + k];
                if (gd >= 0) v_out.data[gd] += mf_MiJt_a.data[mf6 + k] * a;
            }}
            if (dof_b >= 0) {{
                const int gd = world_dof_indices.data[dof_map_base + dof_b + k];
                if (gd >= 0) v_out.data[gd] += mf_MiJt_b.data[mf6 + k] * a;
            }}
        }}
    }}
    for (int r = dense_lo; r < dense_hi; ++r) {{
        float l = 0.0f;
        {sel_lam}
        world_impulses.data[off_dense + r] = l;
    }}
    if (row_phase != 3) {{
        for (int j = 0; j < m_mf; ++j) {{
            const int r = m_dense + j;
            float l = 0.0f;
            {sel_lam}
            mf_impulses.data[off_mf + j] = l;
        }}
    }}
#endif
"""

    @wp.func_native(snippet)
    def rb_sweep_lane_native(
        block: int,
        world_count: int,
        rb_counts: wp.array[int],
        rb_lists: wp.array[int],
        rb_A: wp.array[float],
        rb_vec: wp.array[float],
        rb_meta: wp.array[int],
        world_constraint_count: wp.array[int],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        dense_phase_bounds: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        Y_world: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        world_impulses: wp.array2d[float],
        mf_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        regularize: int,
        row_phase: int,
        friction_start_iteration: int,
        iteration_offset: int,
        freeze_drive_rows: int,
        v_out: wp.array[float],
    ): ...

    def rb_sweep_lane_template(
        world_count: int,
        rb_counts: wp.array[int],
        rb_lists: wp.array[int],
        rb_A: wp.array[float],
        rb_vec: wp.array[float],
        rb_meta: wp.array[int],
        world_constraint_count: wp.array[int],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        dense_phase_bounds: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        Y_world: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        world_impulses: wp.array2d[float],
        mf_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        regularize: int,
        row_phase: int,
        friction_start_iteration: int,
        iteration_offset: int,
        freeze_drive_rows: int,
        v_out: wp.array[float],
    ):
        block, _lane = wp.tid()
        rb_sweep_lane_native(
            block,
            world_count,
            rb_counts,
            rb_lists,
            rb_A,
            rb_vec,
            rb_meta,
            world_constraint_count,
            mf_constraint_count,
            mf_contact_rows_end,
            dense_phase_bounds,
            world_dof_indices,
            Y_world,
            mf_MiJt_a,
            mf_MiJt_b,
            world_impulses,
            mf_impulses,
            iterations,
            omega,
            regularize,
            row_phase,
            friction_start_iteration,
            iteration_offset,
            freeze_drive_rows,
            v_out,
        )

    name = f"pgs_rb_sweep_lane_{M_D}_{M_MF}_{D}_rows{RS}_drive{int(has_drive_rows)}"
    rb_sweep_lane_template.__name__ = name
    rb_sweep_lane_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(rb_sweep_lane_template)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Warp-per-world sweep (one or two rows per lane), persistent grid over a class list
# ─────────────────────────────────────────────────────────────────────────────


def get_response_block_sweep_warp_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
    rows: int,
    lane_rows: int,
    has_drive_rows: bool,
) -> wp.Kernel:
    """Warp-per-world sweep for ``rows`` in {32, 64} over the matching class list.

    The response block lives in shared memory (row-minor so a column read is conflict-free)
    and the row loop is a runtime loop: row ``i`` is owned by lane ``i & 31`` in half
    ``i >> 5``. Every lane computes the projection from a few broadcasts of the owner's state,
    so the only branches are warp-uniform.
    """
    M_D = int(max_constraints)
    M_MF = int(mf_max_constraints)
    D = int(max_world_dofs)
    RB = int(rows)
    RS = int(lane_rows)
    ST = RB_STRIDE
    if RB not in (32, 64):
        raise ValueError("rows must be 32 or 64")
    two = RB == 64
    list_slot = RS if RB == 32 else RS + 1
    hd = int(has_drive_rows)
    # Second-half state exists only for the 64-row class; the 32-row class aliases it to
    # constants so the same body compiles for both.
    if two:
        second_state = f"""
        float l1 = r1_ok ? rb_vec.data[vb + {RB_V_LAM} * {ST} + lane + 32] : 0.0f;
        float res1 = r1_ok ? rb_vec.data[vb + {RB_V_RES} * {ST} + lane + 32] : 0.0f;
        const float invd1 = r1_ok ? rb_vec.data[vb + {RB_V_INVD} * {ST} + lane + 32] : 0.0f;
        const float w1 = (r1_ok && regularize) ? rb_vec.data[vb + {RB_V_W} * {ST} + lane + 32] : 1.0f;
        const float ca1 = invd1 * w1;
        const float cb1 = 1.0f - w1;
        const float mu1 = r1_ok ? rb_vec.data[vb + {RB_V_MU} * {ST} + lane + 32] : 0.0f;
        const float rhs1 = r1_ok ? rb_vec.data[vb + {RB_V_RHS} * {ST} + lane + 32] : 0.0f;
        const int kind1 = r1_ok ? rb_meta.data[mb + {RB_M_KIND} * {ST} + lane + 32] : 0;
        float app1 = 0.0f;"""
        second_drive = f"""
        const float dt1 = r1_ok ? rb_vec.data[vb + {RB_V_DRV_T} * {ST} + lane + 32] : 0.0f;
        const float dvm1 = r1_ok ? rb_vec.data[vb + {RB_V_DRV_VM} * {ST} + lane + 32] : 0.0f;
        const float dim1 = r1_ok ? rb_vec.data[vb + {RB_V_DRV_IM} * {ST} + lane + 32] : 0.0f;
        const float dmax1 = r1_ok ? rb_vec.data[vb + {RB_V_DRV_MAX} * {ST} + lane + 32] : 0.0f;"""
        second_masks = f"""
        const unsigned admit_main1 = __ballot_sync(MASK, r1_ok && invd1 > 0.0f && rb_admit(kind1, lane + 32, m_dense, dense_lo, dense_hi, mf_contact_end, row_phase, freeze_drive_rows, {hd}, true));
        const unsigned admit_vlim1 = __ballot_sync(MASK, r1_ok && invd1 > 0.0f && rb_admit(kind1, lane + 32, m_dense, dense_lo, dense_hi, mf_contact_end, row_phase, freeze_drive_rows, {hd}, false));
        const unsigned first_mask1 = __ballot_sync(MASK, (kind1 & {_KB_FIRST}) != 0);
        const unsigned kind_fric1 = __ballot_sync(MASK, (kind1 & 7) == {_K_FRIC});
        const unsigned kind_uni1 = __ballot_sync(MASK, (kind1 & 7) == {_K_UNI});
        const unsigned kind_vlim1 = __ballot_sync(MASK, (kind1 & 7) == {_K_VLIM});
        const unsigned kind_drive1 = __ballot_sync(MASK, (kind1 & 7) == {_K_DRIVE});"""
        load_a = f"""
        for (int e = lane; e < m_tot * {RB}; e += 32) {{
            const int r = e / {RB};
            const int c = e - r * {RB};
            sA[c * {RB + 1} + r] = (c < m_tot) ? rb_A.data[a_base + e] : 0.0f;
        }}"""
        store_second = "        s_app[lane + 32] = app1;\n        s_lam[lane + 32] = l1;"
    else:
        second_state = """
        float l1 = 0.0f, res1 = 0.0f, app1 = 0.0f;
        const float ca1 = 0.0f, cb1 = 0.0f, mu1 = 0.0f, rhs1 = 0.0f;"""
        second_drive = """
        const float dt1 = 0.0f, dvm1 = 0.0f, dim1 = 0.0f, dmax1 = 0.0f;"""
        second_masks = """
        const unsigned admit_main1 = 0u, admit_vlim1 = 0u, first_mask1 = 0u;
        const unsigned kind_fric1 = 0u, kind_uni1 = 0u, kind_vlim1 = 0u, kind_drive1 = 0u;"""
        load_a = f"""
        for (int e = lane; e < m_tot * {RB}; e += 32) {{
            const int r = e / {RB};
            const int c = e - r * {RB};
            sA[c * {RB + 1} + r] = (c < m_tot) ? rb_A.data[a_base + e] : 0.0f;
        }}"""
        store_second = ""
    drive_state = (
        f"""
        const float dt0 = r0_ok ? rb_vec.data[vb + {RB_V_DRV_T} * {ST} + lane] : 0.0f;
        const float dvm0 = r0_ok ? rb_vec.data[vb + {RB_V_DRV_VM} * {ST} + lane] : 0.0f;
        const float dim0 = r0_ok ? rb_vec.data[vb + {RB_V_DRV_IM} * {ST} + lane] : 0.0f;
        const float dmax0 = r0_ok ? rb_vec.data[vb + {RB_V_DRV_MAX} * {ST} + lane] : 0.0f;{second_drive}"""
        if has_drive_rows
        else ""
    )
    drive_update = (
        """
                }} else if ((kdrive >> o) & 1u) {{
                    const float rhs = __shfl_sync(MASK, h ? rhs1 : rhs0, o);
                    const float dim_ = __shfl_sync(MASK, h ? dim1 : dim0, o);
                    const float dvm_ = __shfl_sync(MASK, h ? dvm1 : dvm0, o);
                    const float dt_ = __shfl_sync(MASK, h ? dt1 : dt0, o);
                    const float dmax_ = __shfl_sync(MASK, h ? dmax1 : dmax0, o);
                    new_impulse = fminf(fmaxf(old_impulse * dim_ + (residual - rhs) * dvm_ + dt_, -dmax_), dmax_);
                    delta_impulse = new_impulse - old_impulse;"""
        if has_drive_rows
        else ""
    )
    RP = RB + 1
    res_upd = (
        f"res0 += sA[i * {RP} + lane] * d_; res1 += sA[i * {RP} + lane + 32] * d_;"
        if two
        else f"res0 += sA[i * {RP} + lane] * d_;"
    )
    sib_upd = (
        f"res0 += sA[sib * {RP} + lane] * sib_delta; res1 += sA[sib * {RP} + lane + 32] * sib_delta;"
        if two
        else f"res0 += sA[sib * {RP} + lane] * sib_delta;"
    )
    snippet = f"""
#if defined(__CUDA_ARCH__)
{_ADMIT_LAMBDA}
    const unsigned MASK = 0xFFFFFFFF;
    const int lane = threadIdx.x;
    const int m_dense = world_constraint_count.data[world];
    const int m_mf = mf_constraint_count.data[world];
    const int m_tot = m_dense + m_mf;
{_phase_bounds_code()}
    const int dof_map_base = world * {D};
    const int jy_world_base = world * {M_D} * {D};
    const int off_dense = world * {M_D};
    const int off_mf = world * {M_MF};
    const int mf6_base = world * {M_MF} * 6;
    const int vb = world * {RB_NV} * {ST};
    const int mb = world * {RB_NM} * {ST};
    const int a_base = world * {RB_A_SLOT};
    __shared__ float sA[{RB} * {RB + 1}];
    __shared__ float s_app[{ST}];
    __shared__ float s_lam[{ST}];
    {{
        const bool r0_ok = lane < m_tot;
        const bool r1_ok = {"lane + 32 < m_tot" if two else "false"};
        // Response block, column-major (sA[c * RB + r]) so the update reads one column per lane.
{load_a}
        float l0 = r0_ok ? rb_vec.data[vb + {RB_V_LAM} * {ST} + lane] : 0.0f;
        float res0 = r0_ok ? rb_vec.data[vb + {RB_V_RES} * {ST} + lane] : 0.0f;
        const float invd0 = r0_ok ? rb_vec.data[vb + {RB_V_INVD} * {ST} + lane] : 0.0f;
        const float w0 = (r0_ok && regularize) ? rb_vec.data[vb + {RB_V_W} * {ST} + lane] : 1.0f;
        const float ca0 = invd0 * w0;
        const float cb0 = 1.0f - w0;
        const float mu0 = r0_ok ? rb_vec.data[vb + {RB_V_MU} * {ST} + lane] : 0.0f;
        const float rhs0 = r0_ok ? rb_vec.data[vb + {RB_V_RHS} * {ST} + lane] : 0.0f;
        const int kind0 = r0_ok ? rb_meta.data[mb + {RB_M_KIND} * {ST} + lane] : 0;
        float app0 = 0.0f;{second_state}{drive_state}
        const unsigned admit_main0 = __ballot_sync(MASK, r0_ok && invd0 > 0.0f && rb_admit(kind0, lane, m_dense, dense_lo, dense_hi, mf_contact_end, row_phase, freeze_drive_rows, {hd}, true));
        const unsigned admit_vlim0 = __ballot_sync(MASK, r0_ok && invd0 > 0.0f && rb_admit(kind0, lane, m_dense, dense_lo, dense_hi, mf_contact_end, row_phase, freeze_drive_rows, {hd}, false));
        const unsigned first_mask0 = __ballot_sync(MASK, (kind0 & {_KB_FIRST}) != 0);
        const unsigned kind_fric0 = __ballot_sync(MASK, (kind0 & 7) == {_K_FRIC});
        const unsigned kind_uni0 = __ballot_sync(MASK, (kind0 & 7) == {_K_UNI});
        const unsigned kind_vlim0 = __ballot_sync(MASK, (kind0 & 7) == {_K_VLIM});
        const unsigned kind_drive0 = __ballot_sync(MASK, (kind0 & 7) == {_K_DRIVE});{second_masks}
        __syncwarp();
        for (int iter = 0; iter < iterations; ++iter) {{
            const int global_iter = iteration_offset + iter;
            int changed = 0;
            for (int i = 0; i < m_tot; ++i) {{
                const int o = i & 31;
                const bool h = i >= 32;
                const unsigned admit = h ? admit_main1 : admit_main0;
                if (!((admit >> o) & 1u)) continue;
                const unsigned kfric = h ? kind_fric1 : kind_fric0;
                const unsigned kuni = h ? kind_uni1 : kind_uni0;
                const unsigned kvlim = h ? kind_vlim1 : kind_vlim0;
                const unsigned kdrive = h ? kind_drive1 : kind_drive0;
                const unsigned fmask = h ? first_mask1 : first_mask0;
                const float residual = __shfl_sync(MASK, h ? res1 : res0, o);
                const float old_impulse = __shfl_sync(MASK, h ? l1 : l0, o);
                const float ca_ = __shfl_sync(MASK, h ? ca1 : ca0, o);
                const float cb_ = __shfl_sync(MASK, h ? cb1 : cb0, o);
                const float delta = -residual * ca_ - cb_ * old_impulse;
                const float relaxed = old_impulse + omega * delta;
                float new_impulse = relaxed;
                float delta_impulse = 0.0f, sib_new = 0.0f, sib_delta = 0.0f;
                int sib = -1;
                if ((kfric >> o) & 1u) {{
                    if (global_iter < friction_start_iteration) {{
                        new_impulse = 0.0f;
                    }} else {{
                        const bool first = (fmask >> o) & 1u;
                        const int par = first ? (i - 1) : (i - 2);
                        sib = first ? (i + 1) : (i - 1);
                        const float mu_ = __shfl_sync(MASK, h ? mu1 : mu0, o);
                        const float pl = __shfl_sync(MASK, (par >= 32) ? l1 : l0, par & 31);
                        const float sl = __shfl_sync(MASK, (sib >= 32) ? l1 : l0, sib & 31);
                        const float radius = fmaxf(mu_ * pl, 0.0f);
                        if (radius <= 0.0f) {{
                            new_impulse = 0.0f;
                        }} else {{
                            const float mag = sqrtf(relaxed * relaxed + sl * sl);
                            if (mag > radius) {{
                                const float scale = radius / mag;
                                new_impulse = relaxed * scale;
                                sib_new = sl * scale;
                                sib_delta = sib_new - sl;
                            }}
                        }}
                    }}
                    delta_impulse = new_impulse - old_impulse;
                }} else if ((kuni >> o) & 1u) {{
                    new_impulse = fmaxf(relaxed, 0.0f);
                    delta_impulse = new_impulse - old_impulse;
                }} else if ((kvlim >> o) & 1u) {{
                    delta_impulse = (residual < 0.0f) ? delta : 0.0f;
                    new_impulse = delta_impulse;{drive_update}
                }} else {{
                    delta_impulse = new_impulse - old_impulse;
                }}
                if (lane == o) {{
                    if (h) {{ l1 = new_impulse; app1 += delta_impulse; }}
                    else {{ l0 = new_impulse; app0 += delta_impulse; }}
                }}
                if (delta_impulse != 0.0f) {{
                    changed = 1;
                    const float d_ = delta_impulse;
                    {res_upd}
                }}
                if (sib_delta != 0.0f) {{
                    changed = 1;
                    if (lane == (sib & 31)) {{
                        if (sib >= 32) {{ l1 = sib_new; app1 += sib_delta; }}
                        else {{ l0 = sib_new; app0 += sib_delta; }}
                    }}
                    {sib_upd}
                }}
            }}
            // Velocity-limit pass (dense rows in row order, then matrix-free rows).
            for (int i = 0; i < m_tot; ++i) {{
                const int o = i & 31;
                const bool h = i >= 32;
                const unsigned admit = h ? admit_vlim1 : admit_vlim0;
                if (!((admit >> o) & 1u)) continue;
                const float residual = __shfl_sync(MASK, h ? res1 : res0, o);
                const float ca_ = __shfl_sync(MASK, h ? ca1 : ca0, o);
                const float d_ = (residual < 0.0f) ? -residual * ca_ : 0.0f;
                if (lane == o) {{
                    if (h) {{ l1 = d_; app1 += d_; }}
                    else {{ l0 = d_; app0 += d_; }}
                }}
                if (d_ != 0.0f) {{
                    changed = 1;
                    {res_upd}
                }}
            }}
            if (global_iter >= friction_start_iteration) {{
                const unsigned any = __ballot_sync(MASK, changed != 0);
                if (any == 0u) break;
            }}
        }}
        s_app[lane] = app0;
        s_lam[lane] = l0;
{store_second}
    }}
    __syncwarp();

    // v = v_in + sum_r Y_r * app_r ; lanes over DOFs for dense rows, body blocks for mf rows.
    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof < 0) continue;
        float v = v_out.data[global_dof];
        for (int r = 0; r < m_dense; ++r) {{
            const float a = s_app[r];
            if (a != 0.0f) v += Y_world.data[jy_world_base + r * {D} + d] * a;
        }}
        v_out.data[global_dof] = v;
    }}
    __syncwarp();
    for (int j = 0; j < m_mf; ++j) {{
        const float a = s_app[m_dense + j];
        if (a == 0.0f) continue;
        const int packed_dofs = rb_meta.data[mb + {RB_M_DOFS} * {ST} + m_dense + j];
        const int dof_a = packed_dofs >> 16;
        const int dof_b = (packed_dofs << 16) >> 16;
        const int mf6 = mf6_base + j * 6;
        if (lane < 6 && dof_a >= 0) {{
            const int gd = world_dof_indices.data[dof_map_base + dof_a + lane];
            if (gd >= 0) v_out.data[gd] += mf_MiJt_a.data[mf6 + lane] * a;
        }}
        if (lane >= 6 && lane < 12 && dof_b >= 0) {{
            const int gd = world_dof_indices.data[dof_map_base + dof_b + lane - 6];
            if (gd >= 0) v_out.data[gd] += mf_MiJt_b.data[mf6 + lane - 6] * a;
        }}
        __syncwarp();
    }}
    for (int r = dense_lo + lane; r < dense_hi; r += 32) {{
        world_impulses.data[off_dense + r] = s_lam[r];
    }}
    if (row_phase != 3) {{
        for (int j = lane; j < m_mf; j += 32) {{
            mf_impulses.data[off_mf + j] = s_lam[m_dense + j];
        }}
    }}
#endif
"""

    @wp.func_native(snippet)
    def rb_sweep_warp_native(
        world: int,
        rb_A: wp.array[float],
        rb_vec: wp.array[float],
        rb_meta: wp.array[int],
        world_constraint_count: wp.array[int],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        dense_phase_bounds: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        Y_world: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        world_impulses: wp.array2d[float],
        mf_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        regularize: int,
        row_phase: int,
        friction_start_iteration: int,
        iteration_offset: int,
        freeze_drive_rows: int,
        v_out: wp.array[float],
    ): ...

    def rb_sweep_warp_template(
        world_count: int,
        grid_stride: int,
        rb_counts: wp.array[int],
        rb_lists: wp.array[int],
        rb_A: wp.array[float],
        rb_vec: wp.array[float],
        rb_meta: wp.array[int],
        world_constraint_count: wp.array[int],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        dense_phase_bounds: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        Y_world: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        world_impulses: wp.array2d[float],
        mf_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        regularize: int,
        row_phase: int,
        friction_start_iteration: int,
        iteration_offset: int,
        freeze_drive_rows: int,
        v_out: wp.array[float],
    ):
        # Persistent grid over this class's world list.
        block, _lane = wp.tid()
        count = rb_counts[list_slot]
        idx = block
        while idx < count:
            world = rb_lists[list_slot * world_count + idx]
            rb_sweep_warp_native(
                world,
                rb_A,
                rb_vec,
                rb_meta,
                world_constraint_count,
                mf_constraint_count,
                mf_contact_rows_end,
                dense_phase_bounds,
                world_dof_indices,
                Y_world,
                mf_MiJt_a,
                mf_MiJt_b,
                world_impulses,
                mf_impulses,
                iterations,
                omega,
                regularize,
                row_phase,
                friction_start_iteration,
                iteration_offset,
                freeze_drive_rows,
                v_out,
            )
            idx += grid_stride

    name = f"pgs_rb_sweep_warp_{M_D}_{M_MF}_{D}_rows{RB}_lane{RS}_drive{int(has_drive_rows)}"
    rb_sweep_warp_template.__name__ = name
    rb_sweep_warp_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(rb_sweep_warp_template)

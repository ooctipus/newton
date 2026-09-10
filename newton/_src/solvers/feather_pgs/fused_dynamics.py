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

"""Fused world-dynamics kernel (K1, class S): one warp per articulation, state in shared memory.

One launch computes, from the joint state alone: forward kinematics, motion subspaces, twists
and accelerations, bias forces, joint torques (inverse dynamics), and on mass-update steps the
composite inertias, the mass matrix and its Cholesky factor. Every per-joint formula is a
line-by-line port of the Warp functions in ``kernels.py`` using the same ``wp::`` primitives, so
the outputs match the serial kernels to the bit (the factor matches the tiled Cholesky to
floating-point order). Per-template kinematic constants are read once per warp; body mass
properties are live per body.

Supported joint types: prismatic, revolute, ball, fixed, free, distance. Templates with other
joint types keep the legacy chain.
"""

from __future__ import annotations

import warp as wp

from ...sim.enums import JointType
from .kernels import BodyFlags

_SUPPORTED_JOINT_TYPES = frozenset(
    {
        int(JointType.PRISMATIC),
        int(JointType.REVOLUTE),
        int(JointType.BALL),
        int(JointType.FIXED),
        int(JointType.FREE),
        int(JointType.DISTANCE),
    }
)


def template_supported(joint_types) -> bool:
    return all(int(t) in _SUPPORTED_JOINT_TYPES for t in joint_types)


def fused_dynamics_launch_shape(lanes: int, with_mass: bool) -> tuple[int, int]:
    """Return ``(warps_per_block, articulation_slots_per_block)`` for the fused kernel.

    Static shared memory is capped at 48 KB per block: the kinematics-only variant keeps 8
    articulation slots per block (~3 KB each), the mass variant 4 (~8 KB each).
    """
    G = int(lanes)
    if G not in (4, 8, 16, 32):
        raise ValueError("lanes must be 4, 8, 16 or 32")
    if with_mass and G < 8:
        raise ValueError("the mass variant needs at least 8 lanes per articulation")
    apb = 32 // G
    slots = 4 if with_mass else 8
    wpb = max(1, slots // apb)
    return wpb, wpb * apb


def get_fused_dynamics_kernel(
    n_dofs: int, max_joints: int, lanes: int = 8, *, with_mass: bool = False, tpl_shared: bool = False
) -> wp.Kernel:
    N = int(n_dofs)
    MJ = int(max_joints)
    G = int(lanes)
    WPB, _ = fused_dynamics_launch_shape(G, with_mass)
    APB = 32 // G
    WM = 1 if with_mass else 0
    TS = 1 if tpl_shared else 0  # one template per block: stage the template tables in shared memory
    T_PRISMATIC = int(JointType.PRISMATIC)
    T_REVOLUTE = int(JointType.REVOLUTE)
    T_BALL = int(JointType.BALL)
    T_FREE = int(JointType.FREE)
    T_DISTANCE = int(JointType.DISTANCE)
    KINEMATIC = int(BodyFlags.KINEMATIC)
    snippet = f"""
#if defined(__CUDA_ARCH__)
    using vec3 = wp::vec_t<3, float>;
    using vec6 = wp::vec_t<6, float>;
    using quat = wp::quat_t<float>;
    using mat3 = wp::mat_t<3, 3, float>;
    using xform = wp::transform_t<float>;
    constexpr int N_ = {N};
    constexpr int MJ_ = {MJ};
    constexpr int WPB = {WPB};    // warps per block
    constexpr int G = {G};        // lanes per articulation
    constexpr int APB = {APB};    // articulations per warp
    constexpr int SLOTS = WPB * APB;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int grp = lane / G;
    const int local = lane - grp * G;
    const int slot = warp * APB + grp;
    const int gidx = block * SLOTS + slot;
    constexpr int WITH_MASS = {WM};
    __shared__ xform s_bq_all[SLOTS * MJ_];
    __shared__ vec6 s_v_all[SLOTS * MJ_];
    __shared__ vec6 s_a_all[SLOTS * MJ_];
    __shared__ vec6 s_f_all[SLOTS * MJ_];     // bias force f_s (body_f_s)
    __shared__ vec6 s_S_all[SLOTS * N_];
    // Mass-variant scratch (composite inertia, joint-space inertia, factor); 1 element when unused.
    constexpr int MS = WITH_MASS ? SLOTS : 0;
    __shared__ vec3 s_com_all[MS * MJ_ + 1];
    __shared__ mat3 s_io_all[MS * MJ_ + 1];
    __shared__ float s_Ic_all[MS * MJ_ * 36 + 1];
    __shared__ float s_F_all[MS * N_ * 6 + 1];
    __shared__ float s_H_all[MS * N_ * N_ + 1];
    xform* s_bq = s_bq_all + slot * MJ_;
    vec6* s_v = s_v_all + slot * MJ_;
    vec6* s_a = s_a_all + slot * MJ_;
    vec6* s_f = s_f_all + slot * MJ_;
    vec6* s_fn = s_a;   // net wrench passed to the parent (accelerations are dead after pass 1)
    vec6* s_S = s_S_all + slot * N_;
    vec3* s_com = s_com_all + (WITH_MASS ? slot * MJ_ : 0);
    mat3* s_io = s_io_all + (WITH_MASS ? slot * MJ_ : 0);
    float* s_Ic = s_Ic_all + (WITH_MASS ? slot * MJ_ * 36 : 0);
    float* s_F = s_F_all + (WITH_MASS ? slot * N_ * 6 : 0);
    float* s_H = s_H_all + (WITH_MASS ? slot * N_ * N_ : 0);
    constexpr int TPL_SHARED = {TS};
    __shared__ int s_tpl_int[TPL_SHARED ? MJ_ * 9 : 1];
    __shared__ xform s_tpl_Xp[TPL_SHARED ? MJ_ : 1];
    __shared__ xform s_tpl_Xc[TPL_SHARED ? MJ_ : 1];
    __shared__ vec3 s_tpl_axis[TPL_SHARED ? N_ : 1];
    __shared__ int s_tpl_lvl[TPL_SHARED ? MJ_ + 2 : 1];
    __shared__ int s_tpl_lvlj[TPL_SHARED ? MJ_ : 1];
    __shared__ int s_tpl_ch[TPL_SHARED ? MJ_ : 1];
    int c_base = 0, l_base = 0;
    if (TPL_SHARED) {{
        // Every articulation of this block shares one template (checked on the host).
        const int t_b = art_template.data[group_to_art.data[block * SLOTS]];
        const int tj_b = tpl_joint_offset.data[t_b];
        const int td_b = tpl_dof_offset.data[t_b];
        const int cnt = tpl_joint_offset.data[t_b + 1] - tj_b;
        const int dof_count = tpl_dof_offset.data[t_b + 1] - td_b;
        c_base = tpl_children_offsets.data[tj_b];
        l_base = tpl_level_offsets.data[t_b * (tpl_max_levels + 1)];
        const int nch = tpl_children_offsets.data[tj_b + cnt] - c_base;
        for (int e = threadIdx.x; e < cnt; e += blockDim.x) {{
            const int g = tj_b + e;
            s_tpl_int[e * 9 + 0] = tpl_parent_local.data[g];
            s_tpl_int[e * 9 + 1] = tpl_type.data[g];
            s_tpl_int[e * 9 + 2] = tpl_q_off.data[g];
            s_tpl_int[e * 9 + 3] = tpl_qd_off.data[g];
            s_tpl_int[e * 9 + 4] = tpl_child_off.data[g];
            s_tpl_int[e * 9 + 5] = tpl_lin.data[g];
            s_tpl_int[e * 9 + 6] = tpl_ang.data[g];
            s_tpl_int[e * 9 + 7] = tpl_children_offsets.data[g];
            s_tpl_int[e * 9 + 8] = tpl_children_offsets.data[g + 1];
            s_tpl_Xp[e] = tpl_X_p.data[g];
            s_tpl_Xc[e] = tpl_X_c.data[g];
            s_tpl_lvlj[e] = tpl_level_joints.data[l_base + e];
        }}
        for (int e = threadIdx.x; e < nch; e += blockDim.x) s_tpl_ch[e] = tpl_children.data[c_base + e];
        for (int e = threadIdx.x; e < dof_count; e += blockDim.x) s_tpl_axis[e] = tpl_axis.data[td_b + e];
        for (int e = threadIdx.x; e <= tpl_max_levels; e += blockDim.x) s_tpl_lvl[e] = tpl_level_offsets.data[t_b * (tpl_max_levels + 1) + e];
        __syncthreads();
    }}
#if {TS}
#define TPL_PL(jl) s_tpl_int[(jl) * 9 + 0]
#define TPL_TYPE(jl) s_tpl_int[(jl) * 9 + 1]
#define TPL_QOFF(jl) s_tpl_int[(jl) * 9 + 2]
#define TPL_QDOFF(jl) s_tpl_int[(jl) * 9 + 3]
#define TPL_CHILD(jl) s_tpl_int[(jl) * 9 + 4]
#define TPL_LIN(jl) s_tpl_int[(jl) * 9 + 5]
#define TPL_ANG(jl) s_tpl_int[(jl) * 9 + 6]
#define TPL_CLO(jl) s_tpl_int[(jl) * 9 + 7]
#define TPL_CHI(jl) s_tpl_int[(jl) * 9 + 8]
#define TPL_XP(jl) s_tpl_Xp[jl]
#define TPL_XC(jl) s_tpl_Xc[jl]
#define TPL_AXIS(as) s_tpl_axis[(as) - td]
#define TPL_LVL(level) s_tpl_lvl[level]
#define TPL_LVLJ(k) s_tpl_lvlj[(k) - l_base]
#define TPL_CHILDREN(c) s_tpl_ch[(c) - c_base]
#else
#define TPL_PL(jl) tpl_parent_local.data[tj + (jl)]
#define TPL_TYPE(jl) tpl_type.data[tj + (jl)]
#define TPL_QOFF(jl) tpl_q_off.data[tj + (jl)]
#define TPL_QDOFF(jl) tpl_qd_off.data[tj + (jl)]
#define TPL_CHILD(jl) tpl_child_off.data[tj + (jl)]
#define TPL_LIN(jl) tpl_lin.data[tj + (jl)]
#define TPL_ANG(jl) tpl_ang.data[tj + (jl)]
#define TPL_CLO(jl) tpl_children_offsets.data[tj + (jl)]
#define TPL_CHI(jl) tpl_children_offsets.data[tj + (jl) + 1]
#define TPL_XP(jl) tpl_X_p.data[tj + (jl)]
#define TPL_XC(jl) tpl_X_c.data[tj + (jl)]
#define TPL_AXIS(as) tpl_axis.data[as]
#define TPL_LVL(level) tpl_level_offsets.data[t * (tpl_max_levels + 1) + (level)]
#define TPL_LVLJ(k) tpl_level_joints.data[k]
#define TPL_CHILDREN(c) tpl_children.data[c]
#endif
    if (gidx >= n_arts) return;
    const int art = group_to_art.data[gidx];
    const int t = art_template.data[art];
    const int start = articulation_start.data[art];
    const int body_base = art_body_base.data[art];
    const int q_base = art_q_base.data[art];
    const int qd_base = art_qd_base.data[art];
    const int dof_start = articulation_dof_start.data[art];
    const int tj = tpl_joint_offset.data[t];
    const int td = tpl_dof_offset.data[t];
    const int count = tpl_joint_offset.data[t + 1] - tj;
    const int n_levels = tpl_max_levels;
    const vec3 gravity_s = gravity.data[0];

    // Newton-layout spatial helpers (top = linear, bottom = angular).
    auto vtop = [](const vec6& x) {{ return vec3(x.c[0], x.c[1], x.c[2]); }};
    auto vbot = [](const vec6& x) {{ return vec3(x.c[3], x.c[4], x.c[5]); }};
    auto mk6 = [](const vec3& a, const vec3& b) {{ return vec6(a, b); }};
    auto scross = [&](const vec6& a, const vec6& b) {{
        const vec3 w_a = vbot(a), v_a = vtop(a), w_b = vbot(b), v_b = vtop(b);
        const vec3 w = wp::cross(w_a, w_b);
        const vec3 v = wp::cross(w_a, v_b) + wp::cross(v_a, w_b);
        return mk6(v, w);
    }};
    auto scross_dual = [&](const vec6& a, const vec6& b) {{
        const vec3 w_a = vbot(a), v_a = vtop(a), w_b = vbot(b), v_b = vtop(b);
        const vec3 w = wp::cross(w_a, w_b) + wp::cross(v_a, v_b);
        const vec3 v = wp::cross(w_a, v_b);
        return mk6(v, w);
    }};
    // Newton transform_twist: swap to Warp layout, apply, swap back.
    auto twist = [&](const xform& tf, const vec6& x) {{
        const quat q = tf.q;
        const vec3 p = tf.p;
        vec3 w = vbot(x);
        vec3 v = vtop(x);
        w = wp::quat_rotate(q, w);
        v = wp::quat_rotate(q, v) + wp::cross(p, w);
        return mk6(v, w);
    }};
    auto mul_com = [&](float mass, const vec3& com, const mat3& io, const vec6& vel) {{
        const vec3 linear = vtop(vel);
        const vec3 angular = vbot(vel);
        return mk6(mass * (linear - wp::cross(com, angular)), mass * wp::cross(com, linear) + wp::mul(io, angular));
    }};

    // ── Pass 1: poses, motion subspaces, twists, accelerations and bias forces, level by level ──
    vec3 origin = vec3();
    auto do_pose = [&](int k) {{
        const int jl = TPL_LVLJ(k);
        const int tjl = tj + jl;
        const int pl = TPL_PL(jl);
        const xform X_pj = (pl >= 0) ? TPL_XP(jl) : joint_X_p.data[start + jl];
        const xform X_cj = TPL_XC(jl);
        xform X_wpj = X_pj;
        if (pl >= 0) X_wpj = wp::mul(s_bq[pl], X_pj);
        const int type = TPL_TYPE(jl);
        const int qs = q_base + TPL_QOFF(jl);
        const int as = td + TPL_QDOFF(jl);
        xform X_j = wp::transform_identity<float>();
        if (type == {T_PRISMATIC}) {{
            const float qv = joint_q.data[qs];
            const vec3 axis = TPL_AXIS(as);
            X_j = xform(axis * qv, wp::quat_identity<float>());
        }} else if (type == {T_REVOLUTE}) {{
            const float qv = joint_q.data[qs];
            const vec3 axis = TPL_AXIS(as);
            X_j = xform(vec3(), wp::quat_from_axis_angle(axis, qv));
        }} else if (type == {T_BALL}) {{
            X_j = xform(vec3(), quat(joint_q.data[qs + 0], joint_q.data[qs + 1], joint_q.data[qs + 2], joint_q.data[qs + 3]));
        }} else if (type == {T_FREE} || type == {T_DISTANCE}) {{
            X_j = xform(vec3(joint_q.data[qs + 0], joint_q.data[qs + 1], joint_q.data[qs + 2]),
                        quat(joint_q.data[qs + 3], joint_q.data[qs + 4], joint_q.data[qs + 5], joint_q.data[qs + 6]));
        }}
        const xform X_wcj = wp::mul(X_wpj, X_j);
        const xform X_wc = wp::mul(X_wcj, wp::transform_inverse(X_cj));
        const int child = body_base + TPL_CHILD(jl);
        const xform X_sm = wp::mul(X_wc, body_X_com.data[child]);
        s_bq[jl] = X_wc;
        body_q.data[child] = X_wc;
        body_q_com.data[child] = X_sm;
    }};
    auto do_motion = [&](int k) {{
        const int jl = TPL_LVLJ(k);
        const int tjl = tj + jl;
        const int pl = TPL_PL(jl);
        const int child = body_base + TPL_CHILD(jl);
        vec6 parent_v = vec6();
        vec6 parent_a = vec6();
        xform X_wpj = (pl >= 0) ? TPL_XP(jl) : joint_X_p.data[start + jl];
        if (pl >= 0) {{
            parent_v = s_v[pl];
            parent_a = s_a[pl];
            X_wpj = wp::mul(s_bq[pl], X_wpj);
        }}
        const xform X_wpj_local(X_wpj.p - origin, X_wpj.q);
        const int type = TPL_TYPE(jl);
        const int qds = qd_base + TPL_QDOFF(jl);
        const int ldof = TPL_QDOFF(jl);
        const int as = td + ldof;
        const int lin = TPL_LIN(jl);
        const int ang = TPL_ANG(jl);
        vec6 v_j_s = vec6();
        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f, qd4 = 0.0f, qd5 = 0.0f;
        if (type == {T_PRISMATIC} || type == {T_REVOLUTE}) {{
            qd0 = joint_qd.data[qds];
        }} else if (type == {T_BALL}) {{
            qd0 = joint_qd.data[qds + 0]; qd1 = joint_qd.data[qds + 1]; qd2 = joint_qd.data[qds + 2];
        }} else if (type == {T_FREE} || type == {T_DISTANCE}) {{
            qd0 = joint_qd.data[qds + 0]; qd1 = joint_qd.data[qds + 1]; qd2 = joint_qd.data[qds + 2];
            qd3 = joint_qd.data[qds + 3]; qd4 = joint_qd.data[qds + 4]; qd5 = joint_qd.data[qds + 5];
        }}
        if (type == {T_PRISMATIC}) {{
            const vec3 axis = TPL_AXIS(as);
            const vec6 S = twist(X_wpj_local, mk6(axis, vec3()));
            v_j_s = S * qd0;
            s_S[ldof] = S; joint_S_s.data[qds] = S;
        }} else if (type == {T_REVOLUTE}) {{
            const vec3 axis = TPL_AXIS(as);
            const vec6 S = twist(X_wpj_local, mk6(vec3(), axis));
            v_j_s = S * qd0;
            s_S[ldof] = S; joint_S_s.data[qds] = S;
        }} else if (type == {T_BALL}) {{
            const vec6 S0 = twist(X_wpj_local, mk6(vec3(), vec3(1.0f, 0.0f, 0.0f)));
            const vec6 S1 = twist(X_wpj_local, mk6(vec3(), vec3(0.0f, 1.0f, 0.0f)));
            const vec6 S2 = twist(X_wpj_local, mk6(vec3(), vec3(0.0f, 0.0f, 1.0f)));
            s_S[ldof + 0] = S0; s_S[ldof + 1] = S1; s_S[ldof + 2] = S2;
            joint_S_s.data[qds + 0] = S0; joint_S_s.data[qds + 1] = S1; joint_S_s.data[qds + 2] = S2;
            v_j_s = S0 * qd0 + S1 * qd1 + S2 * qd2;
        }} else if (type == {T_FREE} || type == {T_DISTANCE}) {{
            const quat q_sc = X_wpj_local.q;
            const vec3 v_local(qd0, qd1, qd2);
            const vec3 w_local(qd3, qd4, qd5);
            v_j_s = mk6(wp::quat_rotate(q_sc, v_local), wp::quat_rotate(q_sc, w_local));
            const vec3 ex = wp::quat_rotate(q_sc, vec3(1.0f, 0.0f, 0.0f));
            const vec3 ey = wp::quat_rotate(q_sc, vec3(0.0f, 1.0f, 0.0f));
            const vec3 ez = wp::quat_rotate(q_sc, vec3(0.0f, 0.0f, 1.0f));
            const vec6 S0 = mk6(ex, vec3()), S1 = mk6(ey, vec3()), S2 = mk6(ez, vec3());
            const vec6 S3 = mk6(vec3(), ex), S4 = mk6(vec3(), ey), S5 = mk6(vec3(), ez);
            s_S[ldof + 0] = S0; s_S[ldof + 1] = S1; s_S[ldof + 2] = S2; s_S[ldof + 3] = S3; s_S[ldof + 4] = S4; s_S[ldof + 5] = S5;
            joint_S_s.data[qds + 0] = S0; joint_S_s.data[qds + 1] = S1; joint_S_s.data[qds + 2] = S2;
            joint_S_s.data[qds + 3] = S3; joint_S_s.data[qds + 4] = S4; joint_S_s.data[qds + 5] = S5;
        }}
        (void)lin; (void)ang;
        // Issue the body-property loads before any global store so they overlap the twist math.
        const xform X_com_c = body_X_com.data[child];
        const float mass = body_mass.data[child];
        const mat3 I_c = body_inertia.data[child];
        const vec6 v_s = parent_v + v_j_s;
        const vec6 a_s = parent_a + scross(v_s, v_j_s);
        s_v[jl] = v_s;
        s_a[jl] = a_s;
        body_v_s.data[child] = v_s;
        if (publish_aux != 0) body_a_s.data[child] = a_s;
        // Body forces (compute_link_velocity).
        const xform X_sm = wp::mul(s_bq[jl], X_com_c);
        const xform X_sm_local(X_sm.p - origin, X_sm.q);
        const vec3 f_g = mass * gravity_s;
        const mat3 rotation = wp::quat_to_matrix(X_sm_local.q);
        const vec3 com = X_sm_local.p;
        const mat3 com_cross = wp::skew(com);
        const mat3 inertia_origin = wp::mul(wp::mul(rotation, I_c), wp::transpose(rotation))
            - wp::mul(wp::mul(mass, com_cross), com_cross);
        const vec6 f_g_s = mk6(f_g, wp::cross(com, f_g));
        if (WITH_MASS) {{
            s_com[jl] = com;
            s_io[jl] = inertia_origin;
        }}
        if (is_free_rigid.data[art] != 0) {{
            // assemble_com_spatial_inertia: [[m I, -m [c]x], [m [c]x, I_o]] (row-major 6x6)
            const mat3 mcx = wp::mul(mass, wp::skew(com));
            float* I6 = reinterpret_cast<float*>(&body_I_s.data[child]);
            for (int r = 0; r < 3; ++r) {{
                for (int c2 = 0; c2 < 3; ++c2) {{
                    I6[r * 6 + c2] = (r == c2) ? mass : 0.0f;
                    I6[r * 6 + 3 + c2] = -mcx.data[r][c2];
                    I6[(3 + r) * 6 + c2] = mcx.data[r][c2];
                    I6[(3 + r) * 6 + 3 + c2] = inertia_origin.data[r][c2];
                }}
            }}
        }}
        {{
            float* terms = &body_inertia_terms.data[child * 12];
            terms[0] = com[0]; terms[1] = com[1]; terms[2] = com[2];
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) terms[3 + 3 * r + c] = inertia_origin.data[r][c];
        }}
        const vec6 coriolis = scross_dual(v_s, mul_com(mass, com, inertia_origin, v_s));
        const vec6 f_b_s = mul_com(mass, com, inertia_origin, a_s) + coriolis;
        const vec6 f_s = f_b_s - f_g_s;
        s_f[jl] = f_s;
        if (publish_aux != 0) body_f_s.data[child] = f_s;  // read only by the legacy chain and the checks
    }};
    // Poses level by level, then the articulation origin, then motion level by level
    // (a merged single sweep measured slower: 177 us against 156 us at 16k articulations).
    for (int level = 0; level < n_levels; ++level) {{
        const int lo = TPL_LVL(level);
        const int hi = TPL_LVL(level + 1);
        for (int k = lo + local; k < hi; k += G) do_pose(k);
        __syncwarp();
    }}
    {{
        const int root_body = body_base + TPL_CHILD(0);
        if (count > 0 && root_body >= 0) origin = wp::transform_point(s_bq[0], body_com.data[root_body]);
        if (local == 0) articulation_origin.data[art] = origin;
    }}
    for (int level = 0; level < n_levels; ++level) {{
        const int lo = TPL_LVL(level);
        const int hi = TPL_LVL(level + 1);
        for (int k = lo + local; k < hi; k += G) do_motion(k);
        __syncwarp();
    }}

    // ── Pass 3: inverse dynamics (and composite inertia), leaves to root ──
    for (int step = 0; step < n_levels; ++step) {{
        const int level = n_levels - 1 - step;
        const int lo = TPL_LVL(level);
        const int hi = TPL_LVL(level + 1);
        for (int k = lo + local; k < hi; k += G) {{
            const int jl = TPL_LVLJ(k);
            const int tjl = tj + jl;
            const int child = body_base + TPL_CHILD(jl);
            const int c_lo = TPL_CLO(jl);
            const int c_hi = TPL_CHI(jl);
            vec6 f_t_s = vec6();
            for (int c = c_lo; c < c_hi; ++c) f_t_s = f_t_s + s_fn[TPL_CHILDREN(c)];
            if (publish_aux != 0) body_ft_s.data[child] = f_t_s;
            vec6 f_ext_com = vec6();
            if ((body_flags.data[child] & {KINEMATIC}) == 0) f_ext_com = body_f_ext.data[child];
            const vec3 f_ext_f = vbot(f_ext_com);
            const vec3 f_ext_t = vtop(f_ext_com);
            const vec3 com_world = wp::transform_point(s_bq[jl], body_com.data[child]);
            const vec3 com_rel = com_world - origin;
            const vec3 tau_origin = f_ext_f + wp::cross(com_rel, f_ext_t);
            const vec6 f_ext_origin = mk6(f_ext_t, tau_origin);
            const vec6 f_s = s_f[jl] + f_t_s - f_ext_origin;
            s_fn[jl] = f_s;
            // jcalc_tau
            const int type = TPL_TYPE(jl);
            const int qds = qd_base + TPL_QDOFF(jl);
            const int qs = q_base + TPL_QOFF(jl);
            const int ldof = TPL_QDOFF(jl);
            // Loads first (joint_f, existing tau, passive terms), stores last, so the loads overlap.
            if (type == {T_BALL} || type == {T_FREE} || type == {T_DISTANCE}) {{
                const int n = (type == {T_BALL}) ? 3 : 6;
                float value[6];
                for (int i = 0; i < 6; ++i) {{
                    if (i < n) {{
                        value[i] = joint_f.data[qds + i];
                        if (add_existing_tau != 0) value[i] += tau.data[qds + i];
                    }}
                }}
                for (int i = 0; i < 6; ++i) if (i < n) tau.data[qds + i] = value[i] - wp::dot(s_S[ldof + i], f_s);
            }} else if (type == {T_PRISMATIC} || type == {T_REVOLUTE}) {{
                const int axis_count = TPL_LIN(jl) + TPL_ANG(jl);
                float value[3];
                for (int i = 0; i < 3; ++i) {{
                    if (i < axis_count) {{
                        const int j = qds + i;
                        float passive_f = joint_spring_stiffness.data[j] * (joint_spring_ref.data[j] - joint_q.data[qs + i]);
                        passive_f -= joint_damping.data[j] * joint_qd.data[j];
                        value[i] = joint_f.data[j] + passive_f;
                        if (add_existing_tau != 0) value[i] += tau.data[j];
                    }}
                }}
                for (int i = 0; i < 3; ++i) if (i < axis_count) tau.data[qds + i] = -wp::dot(s_S[ldof + i], f_s) + value[i];
            }}
            if (WITH_MASS && do_mass != 0) {{
                // Composite inertia from the compact terms, children added in descending joint order.
                const float mass = body_mass.data[child];
                const vec3 com = s_com[jl];
                const mat3& io = s_io[jl];
                float* Ic = s_Ic + jl * 36;
                for (int element = 0; element < 36; ++element) {{
                    const int row = element / 6;
                    const int col = element - row * 6;
                    float value = 0.0f;
                    if (row < 3 && col < 3) {{
                        if (row == col) value = mass;
                    }} else if (row >= 3 && col >= 3) {{
                        value = io.data[row - 3][col - 3];
                    }} else {{
                        const int cross_row = row < 3 ? row : row - 3;
                        const int cross_col = col < 3 ? col : col - 3;
                        float cross = 0.0f;
                        if (cross_row == 0 && cross_col == 1) cross = -com[2];
                        if (cross_row == 0 && cross_col == 2) cross = com[1];
                        if (cross_row == 1 && cross_col == 0) cross = com[2];
                        if (cross_row == 1 && cross_col == 2) cross = -com[0];
                        if (cross_row == 2 && cross_col == 0) cross = -com[1];
                        if (cross_row == 2 && cross_col == 1) cross = com[0];
                        value = (row < 3 ? -mass : mass) * cross;
                    }}
                    for (int c = c_lo; c < c_hi; ++c) value += s_Ic[TPL_CHILDREN(c) * 36 + element];
                    Ic[element] = value;
                }}
            }}
        }}
        __syncwarp();
    }}
    if (!WITH_MASS || do_mass == 0) return;
    // Publish body_I_c.
    for (int e = local; e < count * 36; e += G) {{
        const int jl = e / 36;
        const int element = e - jl * 36;
        reinterpret_cast<float*>(&body_I_c.data[body_base + TPL_CHILD(jl)])[element] = s_Ic[jl * 36 + element];
    }}
    // ── Pass 4: mass matrix and Cholesky ──
    for (int d = local; d < N_; d += G) {{
        const int jl = dof_joint_offset.data[d];
        const vec6 S = s_S[d];
        for (int r = 0; r < 6; ++r) {{
            float acc = 0.0f;
            for (int c = 0; c < 6; ++c) acc += s_Ic[jl * 36 + r * 6 + c] * S.c[c];
            s_F[d * 6 + r] = acc;
        }}
    }}
    __syncwarp();
    for (int e = local; e < N_ * (N_ + 1) / 2; e += G) {{
        int row = (int)((sqrtf(8.0f * (float)e + 1.0f) - 1.0f) * 0.5f);
        while (row * (row + 1) / 2 > e) --row;
        while ((row + 1) * (row + 2) / 2 <= e) ++row;
        const int col = e - row * (row + 1) / 2;
        const int source = source_dof.data[row * N_ + col];
        float value = 0.0f;
        if (source >= 0) {{
            const int projection = (source == col) ? row : col;
            const vec6 motion = s_S[projection];
            for (int c = 0; c < 6; ++c) value += motion.c[c] * s_F[source * 6 + c];
        }}
        if (row == col) {{
            value += R_group.data[gidx * N_ + row];
            if (fused_augmented_drive != 0) {{
                const int drive_row = drive_row_by_dof.data[dof_start + row];
                if (drive_row >= 0) {{
                    const float K = row_K.data[drive_row];
                    if (K > 0.0f) value += K;
                }}
            }}
        }}
        s_H[row * N_ + col] = value;
    }}
    __syncwarp();
    for (int col = 0; col < N_; ++col) {{
        float diagonal = s_H[col * N_ + col];
        for (int k = 0; k < col; ++k) {{
            const float v = s_H[col * N_ + k];
            diagonal -= v * v;
        }}
        diagonal = sqrtf(diagonal);
        const float inverse_diagonal = 1.0f / diagonal;
        __syncwarp();
        if (local == 0) s_H[col * N_ + col] = diagonal;
        for (int row = col + 1 + local; row < N_; row += G) {{
            float v = s_H[row * N_ + col];
            for (int k = 0; k < col; ++k) v -= s_H[row * N_ + k] * s_H[col * N_ + k];
            s_H[row * N_ + col] = v * inverse_diagonal;
        }}
        __syncwarp();
    }}
    for (int e = local; e < N_ * N_; e += G) {{
        const int row = e / N_;
        const int col = e - row * N_;
        L_group.data[(size_t)gidx * N_ * N_ + e] = (col <= row) ? s_H[e] : 0.0f;
    }}
    if (local == 0) fk_id_cache_valid.data[art] = 1;
#undef TPL_PL
#undef TPL_TYPE
#undef TPL_QOFF
#undef TPL_QDOFF
#undef TPL_CHILD
#undef TPL_LIN
#undef TPL_ANG
#undef TPL_CLO
#undef TPL_CHI
#undef TPL_XP
#undef TPL_XC
#undef TPL_AXIS
#undef TPL_LVL
#undef TPL_LVLJ
#undef TPL_CHILDREN
#endif
"""

    @wp.func_native(snippet)
    def fused_dynamics_native(
        block: int,
        warps_per_block: int,
        n_arts: int,
        do_mass: int,
        add_existing_tau: int,
        fused_augmented_drive: int,
        publish_aux: int,
        tpl_max_levels: int,
        group_to_art: wp.array[int],
        art_template: wp.array[int],
        art_body_base: wp.array[int],
        art_q_base: wp.array[int],
        art_qd_base: wp.array[int],
        tpl_joint_offset: wp.array[int],
        tpl_dof_offset: wp.array[int],
        tpl_level_offsets: wp.array2d[int],
        tpl_level_joints: wp.array[int],
        tpl_children_offsets: wp.array[int],
        tpl_children: wp.array[int],
        tpl_type: wp.array[int],
        tpl_lin: wp.array[int],
        tpl_ang: wp.array[int],
        tpl_q_off: wp.array[int],
        tpl_qd_off: wp.array[int],
        tpl_parent_local: wp.array[int],
        tpl_child_off: wp.array[int],
        tpl_X_p: wp.array[wp.transform],
        tpl_X_c: wp.array[wp.transform],
        tpl_axis: wp.array[wp.vec3],
        articulation_start: wp.array[int],
        articulation_dof_start: wp.array[int],
        joint_q: wp.array[float],
        joint_qd: wp.array[float],
        joint_f: wp.array[float],
        joint_spring_stiffness: wp.array[float],
        joint_spring_ref: wp.array[float],
        joint_damping: wp.array[float],
        body_X_com: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_mass: wp.array[float],
        body_inertia: wp.array[wp.mat33],
        body_f_ext: wp.array[wp.spatial_vector],
        body_flags: wp.array[wp.int32],
        is_free_rigid: wp.array[wp.int32],
        gravity: wp.array[wp.vec3],
        R_group: wp.array2d[float],
        dof_joint_offset: wp.array[int],
        source_dof: wp.array2d[int],
        drive_row_by_dof: wp.array[int],
        row_K: wp.array[float],
        body_q: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        articulation_origin: wp.array[wp.vec3],
        joint_S_s: wp.array[wp.spatial_vector],
        body_v_s: wp.array[wp.spatial_vector],
        body_a_s: wp.array[wp.spatial_vector],
        body_inertia_terms: wp.array2d[float],
        body_I_s: wp.array[wp.spatial_matrix],
        body_f_s: wp.array[wp.spatial_vector],
        body_ft_s: wp.array[wp.spatial_vector],
        tau: wp.array[float],
        body_I_c: wp.array[wp.spatial_matrix],
        L_group: wp.array3d[float],
        fk_id_cache_valid: wp.array[int],
        joint_X_p: wp.array[wp.transform],
    ): ...

    def fused_dynamics_template(
        warps_per_block: int,
        n_arts: int,
        do_mass: int,
        add_existing_tau: int,
        fused_augmented_drive: int,
        publish_aux: int,
        tpl_max_levels: int,
        group_to_art: wp.array[int],
        art_template: wp.array[int],
        art_body_base: wp.array[int],
        art_q_base: wp.array[int],
        art_qd_base: wp.array[int],
        tpl_joint_offset: wp.array[int],
        tpl_dof_offset: wp.array[int],
        tpl_level_offsets: wp.array2d[int],
        tpl_level_joints: wp.array[int],
        tpl_children_offsets: wp.array[int],
        tpl_children: wp.array[int],
        tpl_type: wp.array[int],
        tpl_lin: wp.array[int],
        tpl_ang: wp.array[int],
        tpl_q_off: wp.array[int],
        tpl_qd_off: wp.array[int],
        tpl_parent_local: wp.array[int],
        tpl_child_off: wp.array[int],
        tpl_X_p: wp.array[wp.transform],
        tpl_X_c: wp.array[wp.transform],
        tpl_axis: wp.array[wp.vec3],
        articulation_start: wp.array[int],
        articulation_dof_start: wp.array[int],
        joint_q: wp.array[float],
        joint_qd: wp.array[float],
        joint_f: wp.array[float],
        joint_spring_stiffness: wp.array[float],
        joint_spring_ref: wp.array[float],
        joint_damping: wp.array[float],
        body_X_com: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_mass: wp.array[float],
        body_inertia: wp.array[wp.mat33],
        body_f_ext: wp.array[wp.spatial_vector],
        body_flags: wp.array[wp.int32],
        is_free_rigid: wp.array[wp.int32],
        gravity: wp.array[wp.vec3],
        R_group: wp.array2d[float],
        dof_joint_offset: wp.array[int],
        source_dof: wp.array2d[int],
        drive_row_by_dof: wp.array[int],
        row_K: wp.array[float],
        body_q: wp.array[wp.transform],
        body_q_com: wp.array[wp.transform],
        articulation_origin: wp.array[wp.vec3],
        joint_S_s: wp.array[wp.spatial_vector],
        body_v_s: wp.array[wp.spatial_vector],
        body_a_s: wp.array[wp.spatial_vector],
        body_inertia_terms: wp.array2d[float],
        body_I_s: wp.array[wp.spatial_matrix],
        body_f_s: wp.array[wp.spatial_vector],
        body_ft_s: wp.array[wp.spatial_vector],
        tau: wp.array[float],
        body_I_c: wp.array[wp.spatial_matrix],
        L_group: wp.array3d[float],
        fk_id_cache_valid: wp.array[int],
        joint_X_p: wp.array[wp.transform],
    ):
        block, _lane = wp.tid()
        fused_dynamics_native(
            block,
            warps_per_block,
            n_arts,
            do_mass,
            add_existing_tau,
            fused_augmented_drive,
            publish_aux,
            tpl_max_levels,
            group_to_art,
            art_template,
            art_body_base,
            art_q_base,
            art_qd_base,
            tpl_joint_offset,
            tpl_dof_offset,
            tpl_level_offsets,
            tpl_level_joints,
            tpl_children_offsets,
            tpl_children,
            tpl_type,
            tpl_lin,
            tpl_ang,
            tpl_q_off,
            tpl_qd_off,
            tpl_parent_local,
            tpl_child_off,
            tpl_X_p,
            tpl_X_c,
            tpl_axis,
            articulation_start,
            articulation_dof_start,
            joint_q,
            joint_qd,
            joint_f,
            joint_spring_stiffness,
            joint_spring_ref,
            joint_damping,
            body_X_com,
            body_com,
            body_mass,
            body_inertia,
            body_f_ext,
            body_flags,
            is_free_rigid,
            gravity,
            R_group,
            dof_joint_offset,
            source_dof,
            drive_row_by_dof,
            row_K,
            body_q,
            body_q_com,
            articulation_origin,
            joint_S_s,
            body_v_s,
            body_a_s,
            body_inertia_terms,
            body_I_s,
            body_f_s,
            body_ft_s,
            tau,
            body_I_c,
            L_group,
            fk_id_cache_valid,
            joint_X_p,
        )

    name = f"fused_dynamics_{N}_j{MJ}_g{G}_m{WM}_t{TS}"
    fused_dynamics_template.__name__ = name
    fused_dynamics_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(fused_dynamics_template)

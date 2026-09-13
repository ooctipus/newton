# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in complete root-six/four-leg-three held-factor representation.

The 117 coefficients are a leaf-first Cholesky factor, not an inverse. Current
forces, contact geometry and all public state remain owned by their original
producers. Large-row response uses this same held factor, never a second L.
"""

from functools import cache

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType, ModelFlags


@wp.struct
class BranchPlan:
    index: wp.array2d[int]
    row: wp.array[int]
    col: wp.array[int]
    dof_joint: wp.array[int]
    group_to_art: wp.array[int]
    art_to_world: wp.array[int]
    art_dof_start: wp.array[int]
    art_joint_start: wp.array[int]
    joint_child: wp.array[int]


@wp.struct
class BranchData:
    L: wp.array3d[float]
    valid: wp.array[int]
    status: wp.array[int]


def factor_pattern():
    """Return the exact structural lower triangle in reversed DOF order."""
    pattern = np.zeros((18, 18), bool)
    pattern[:6, :] = True
    pattern[:, :6] = True
    for leg in range(4):
        a = 6 + 3 * leg
        pattern[a : a + 3, a : a + 3] = True
    row, col = np.nonzero(np.tril(pattern[::-1, ::-1]))
    index = np.full((18, 18), -1, np.int32)
    index[row, col] = np.arange(len(row))
    return index, row.astype(np.int32), col.astype(np.int32)


def make_plan(model, solver=None):
    """Prove replicated active ancestry, allowing any fixed descendants."""
    worlds = int(model.world_count)
    if worlds < 1 or model.articulation_count != worlds or model.joint_dof_count != 18 * worlds:
        raise ValueError("Branch response requires one 18-DOF articulation per world")
    starts = model.articulation_start.numpy()
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    types, qd = model.joint_type.numpy(), model.joint_qd_start.numpy()
    if np.any(model.body_flags.numpy() & int(BodyFlags.KINEMATIC)):
        raise ValueError("Branch response excludes prescribed bodies")
    template = None
    dof_joint = None
    for world in range(worlds):
        lo, hi = int(starts[world]), int(starts[world + 1])
        body_to_joint = {int(children[j]): j for j in range(lo, hi)}
        ancestry = {}
        owned_joint = np.full(18, -1, np.int32)
        for j in range(lo, hi):
            start, end = int(qd[j] - 18 * world), int(qd[j + 1] - 18 * world)
            n = end - start
            if j == lo:
                if types[j] != int(JointType.FREE) or n != 6 or parents[j] != -1 or start != 0:
                    raise ValueError("Branch response requires a free root in the first six DOFs")
                path = []
            else:
                if n not in (0, 1) or types[j] != int(JointType.FIXED if n == 0 else JointType.REVOLUTE):
                    raise ValueError("Branch response admits fixed/revolute descendants only")
                parent = body_to_joint.get(int(parents[j]), -1)
                if parent not in ancestry:
                    raise ValueError("Branch response requires topologically ordered single-tree bodies")
                path = ancestry[parent].copy()
            path.extend(range(start, end))
            ancestry[j] = path
            owned_joint[start:end] = j - lo
            if n == 1:
                leg, depth = divmod(start - 6, 3)
                expected = list(range(6)) + list(range(6 + 3 * leg, 7 + 3 * leg + depth))
                if path != expected:
                    raise ValueError("Branch response requires four independent contiguous three-DOF legs")
        if np.any(owned_joint < 0):
            raise ValueError("Incomplete branch generalized-coordinate ownership")
        signature = (types[lo:hi].copy(), owned_joint)
        if template is None:
            template, dof_joint = signature, owned_joint
        elif any(not np.array_equal(a, b) for a, b in zip(template, signature, strict=True)):
            raise ValueError("Branch response requires identical replicated joint layouts")
    index, row, col = factor_pattern()
    host = {"index": index, "row": row, "col": col, "dof_joint": dof_joint}
    if solver is None:
        return host
    plan = BranchPlan()
    for name, value in host.items():
        setattr(plan, name, wp.array(value, dtype=int, device=model.device))
    plan.group_to_art = solver.group_to_art[18]
    plan.art_to_world = solver.art_to_world
    plan.art_dof_start = solver.articulation_dof_start
    plan.art_joint_start = model.articulation_start
    plan.joint_child = model.joint_child
    return plan, host


@cache
def get_refresh_kernel():
    """Form only structural H entries from current CRBA data and factor in place."""
    source = r"""
    const int art = p.group_to_art.data[group], world = p.art_to_world.data[art];
    if (!mask.data[art]) return;
    const int ds = p.art_dof_start.data[art], js = p.art_joint_start.data[art];
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x, nt = blockDim.x;
    __shared__ float a[117], force[108];
    __shared__ int bad;
#define BR_SYNC() __syncthreads()
#else
    const int lane = 0, nt = 1;
    float a[117], force[108]; int bad;
#define BR_SYNC()
#endif
    if (lane == 0) { bad = 0; d.valid.data[world] = 0; }
    for (int col = lane; col < 18; col += nt) {
        const int body = p.joint_child.data[js+p.dof_joint.data[col]];
        const float* inertia = reinterpret_cast<const float*>(&I.data[body]);
        const float* screw = reinterpret_cast<const float*>(&S.data[ds+col]);
        for (int r = 0; r < 6; ++r) {
            float value = 0.0f;
            for (int c = 0; c < 6; ++c) value += inertia[6*r+c]*screw[c];
            force[6*col+r] = value;
        }
    }
    BR_SYNC();
    for (int e = lane; e < 117; e += nt) {
        const int row = p.row.data[e], col = p.col.data[e];
        const int original_row = 17-row, original_col = 17-col;
        const int src = original_row > original_col ? original_row : original_col;
        const int proj = src == original_col ? original_row : original_col;
        const float* screw = reinterpret_cast<const float*>(&S.data[ds+proj]);
        float value = 0.0f;
        for (int k = 0; k < 6; ++k) value += screw[k]*force[6*src+k];
        if (row == col) {
            value += R.data[group*18+original_row];
            int drive = drive_row.data[ds+original_row];
            if (drive < 0 && !parallel_drives) {
                for (int r = 0; r < drive_counts.data[art]; ++r) {
                    const int entry = art*drive_stride+r;
                    if (drive_dofs.data[entry] == ds+original_row) { drive = entry; break; }
                }
            }
            if (drive >= 0 && K.data[drive] > 0.0f) value += K.data[drive];
        }
        a[e] = value;
    }
    BR_SYNC();
    for (int k = 0; k < 18; ++k) {
        const int diag = p.index.data[k*18+k];
        if (lane == 0) {
            if (!(a[diag] > 0.0f) || !isfinite(a[diag])) bad = 1;
            a[diag] = sqrtf(a[diag]);
        }
        BR_SYNC();
        for (int row = k+1+lane; row < 18; row += nt) {
            const int entry = p.index.data[row*18+k];
            if (entry >= 0) a[entry] /= a[diag];
        }
        BR_SYNC();
        for (int e = lane; e < 117; e += nt) {
            const int row = p.row.data[e], col = p.col.data[e];
            if (col <= k) continue;
            const int x = p.index.data[row*18+k], y = p.index.data[col*18+k];
            if (x >= 0 && y >= 0) a[e] -= a[x]*a[y];
        }
        BR_SYNC();
    }
    for (int e = lane; e < 117; e += nt) d.L.data[group*117+e] = a[e];
    BR_SYNC();
    if (lane == 0) { d.status.data[world] |= bad; d.valid.data[world] = bad == 0; }
#undef BR_SYNC
"""

    @wp.func_native(source)
    def native(
        group: int,
        p: BranchPlan,
        d: BranchData,
        mask: wp.array[int],
        S: wp.array[wp.spatial_vector],
        I: wp.array[wp.spatial_matrix],
        R: wp.array2d[float],
        drive_row: wp.array[int],
        K: wp.array[float],
        drive_counts: wp.array[int],
        drive_dofs: wp.array[int],
        drive_stride: int,
        parallel_drives: int,
    ): ...

    def refresh(
        p: BranchPlan,
        d: BranchData,
        mask: wp.array[int],
        S: wp.array[wp.spatial_vector],
        I: wp.array[wp.spatial_matrix],
        R: wp.array2d[float],
        drive_row: wp.array[int],
        K: wp.array[float],
        drive_counts: wp.array[int],
        drive_dofs: wp.array[int],
        drive_stride: int,
        parallel_drives: int,
    ):
        group, _ = wp.tid()
        native(group, p, d, mask, S, I, R, drive_row, K, drive_counts, drive_dofs, drive_stride, parallel_drives)

    refresh.__name__ = refresh.__qualname__ = "branch_refresh18_L117"
    return wp.kernel(enable_backward=False, module="unique")(refresh)


def _solve_source(value="values", factor="factor"):
    index, _, _ = factor_pattern()
    lines = []
    for row in range(18):
        for col in range(row):
            if index[row, col] >= 0:
                lines.append(f"{value}[{row}] -= {factor}[{index[row, col]}]*{value}[{col}];")
        lines.append(f"{value}[{row}] /= {factor}[{index[row, row]}];")
    for col in range(17, -1, -1):
        for row in range(col + 1, 18):
            if index[row, col] >= 0:
                lines.append(f"{value}[{col}] -= {factor}[{index[row, col]}]*{value}[{row}];")
        lines.append(f"{value}[{col}] /= {factor}[{index[col, col]}];")
    return "\n".join(lines)


@cache
def get_predictor_kernel():
    """Apply the held packed factor to current force, in original output order."""
    source = (
        """
    const int art = p.group_to_art.data[group], world = p.art_to_world.data[art];
    const int start = p.art_dof_start.data[art];
    const volatile float* factor = d.L.data+group*117;
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
    const auto li = [](int r, int c) {
        return r<12 ? 6*(r/3)+(r%3)*((r%3)+1)/2+c%3 : 24+12*(r-12)+(r-12)*(r-11)/2+c;
    };
    float value = lane<18 ? tau.data[start+17-lane] : 0.0f;
    for (int pivot=0; pivot<18; ++pivot) {
        if (lane==pivot) value/=factor[li(pivot,pivot)];
        const float solved=__shfl_sync(0xffffffff,value,pivot);
        if (lane>pivot && lane<18 && (lane>=12 || lane/3==pivot/3))
            value-=factor[li(lane,pivot)]*solved;
    }
    for (int pivot=17; pivot>=0; --pivot) {
        if (lane==pivot) value/=factor[li(pivot,pivot)];
        const float solved=__shfl_sync(0xffffffff,value,pivot);
        if (lane<pivot && (pivot>=12 || lane/3==pivot/3))
            value-=factor[li(pivot,lane)]*solved;
    }
    if (lane<18) qdd.data[start+17-lane]=d.valid.data[world] && !d.status.data[world] ? value : NAN;
#else
    float values[18];
    for (int r = 0; r < 18; ++r) values[r] = tau.data[start+17-r];
"""
        + _solve_source()
        + """
    for (int r = 0; r < 18; ++r)
        qdd.data[start+17-r] = d.valid.data[world] && !d.status.data[world] ? values[r] : NAN;
#endif
"""
    )

    @wp.func_native(source)
    def native(group: int, p: BranchPlan, d: BranchData, tau: wp.array[float], qdd: wp.array[float]): ...

    def predict(p: BranchPlan, d: BranchData, tau: wp.array[float], qdd: wp.array[float]):
        group, _ = wp.tid()
        native(group, p, d, tau, qdd)

    predict.__name__ = predict.__qualname__ = "branch_predict18_L117"
    return wp.kernel(enable_backward=False, module="unique")(predict)


@cache
def get_response_kernel(capacity):
    """Materialize physical J/Y only for the original non-cooperative route."""
    source = f"""
    const int art = p.group_to_art.data[group], world = p.art_to_world.data[art];
    const int count = counts.data[world];
    if (count <= skip && mf_count.data[world] == 0) return;
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x, nt = blockDim.x;
#else
    const int lane = 0, nt = 1;
#endif
    const volatile float* factor = d.L.data+group*117;
    for (int row = lane; row < count && row < {capacity}; row += nt) {{
        float values[18];
        for (int r = 0; r < 18; ++r) {{
            const float v = J.data[(group*{capacity}+row)*18+17-r];
            values[r] = v;
            Jw.data[(world*{capacity}+row)*18+17-r] = v;
        }}
        {_solve_source()}
        float diagonal = 0.0f;
        for (int r = 0; r < 18; ++r) {{
            const float v = d.valid.data[world] && !d.status.data[world] ? values[r] : NAN;
            Y.data[(group*{capacity}+row)*18+17-r] = v;
            Yw.data[(world*{capacity}+row)*18+17-r] = v;
            diagonal += Jw.data[(world*{capacity}+row)*18+17-r]*v;
        }}
        diag.data[group*{capacity}+row] = diagonal;
    }}
"""

    @wp.func_native(source)
    def native(
        group: int,
        p: BranchPlan,
        d: BranchData,
        J: wp.array3d[float],
        counts: wp.array[int],
        mf_count: wp.array[int],
        skip: int,
        Y: wp.array3d[float],
        Jw: wp.array3d[float],
        Yw: wp.array3d[float],
        diag: wp.array2d[float],
    ): ...

    def response(
        p: BranchPlan,
        d: BranchData,
        J: wp.array3d[float],
        counts: wp.array[int],
        mf_count: wp.array[int],
        skip: int,
        Y: wp.array3d[float],
        Jw: wp.array3d[float],
        Yw: wp.array3d[float],
        diag: wp.array2d[float],
    ):
        group, _ = wp.tid()
        native(group, p, d, J, counts, mf_count, skip, Y, Jw, Yw, diag)

    response.__name__ = response.__qualname__ = f"branch_fallback18_L117_c{capacity}"
    return wp.kernel(enable_backward=False, module="unique")(response)


def compact_source(source, ink_stage):
    """Replace representation operations, preserving the original finite recurrence."""
    index, _, _ = factor_pattern()

    def once(text, old, new, count=1):
        if text.count(old) != count:
            raise RuntimeError(f"Branch source seam changed: {old[:90]!r}")
        return text.replace(old, new)

    stage = ink_stage[: ink_stage.index("        s_rhs[i] += bjv;") + len("        s_rhs[i] += bjv;")]
    stage = once(
        stage,
        "e < 18 * 18; e += NT) s_L[e] = ink_L_a.data[(size_t)ink_ga * 18 * 18 + e]",
        "e < 117; e += NT) s_L[e] = ink_L_a.data[(size_t)ink_ga * 117 + e]",
    )
    stage = once(stage, "s_L[lane * 18 + lane]", "s_L[BR_LI(lane, lane)]")
    select_tags = r"""
        tag0 = -1; tag1 = -1;
        for (int leg = 0; leg < 4; ++leg) if ((branch_mask & (7u << (6+3*(3-leg)))) != 0u) {
            if (tag0 < 0) tag0 = leg; else if (tag1 == -1) tag1 = leg;
            else tag1 = -2;
        }
"""
    stage = once(
        stage,
        "for (int d = 0; d < 18; ++d) Zi[d * ZS] = 0.0f;",
        "for (int d = 0; d < 12; ++d) Zi[d * ZS] = 0.0f;\n        unsigned branch_mask = 0u;\n        int tag0 = -1, tag1 = -1;",
    )
    stage = once(
        stage,
        "const unsigned mask_o = (side == 0 && art_b == art_a && bo >= 0) ? body_response_dof_mask.data[bo] : 0u;",
        "const unsigned mask_o = (side == 0 && art_b == art_a && bo >= 0) ? body_response_dof_mask.data[bo] : 0u;\n                branch_mask |= mask_s | mask_o;"
        + select_tags,
    )
    compact_store = r"""
                    const int at = d < 6 ? 5-d : tag0 == (17-d)/3 ? 6+(17-d)%3 : tag1 == (17-d)/3 ? 9+(17-d)%3 : -1;
                    if (at >= 0) Zi[at*ZS] = v;
"""
    stage = once(stage, "Zi[(off + d) * ZS] = v;", compact_store)
    stage = once(
        stage,
        "for (int d = 0; d < 18; ++d) Zi[(ink_oa + d) * ZS] = Jg[d];",
        "for (int d = 0; d < 18; ++d) if (Jg[d] != 0.0f) branch_mask |= 1u << d;"
        + select_tags
        + "\n            for (int d = 0; d < 18; ++d) { const float v = Jg[d];"
        + compact_store
        + "\n            }",
    )
    stage = once(
        stage,
        "for (int d = 0; d < 18; ++d) bjv += Zi[d * ZS] * s_v[d];",
        "for (int d = 0; d < 18; ++d) { const int at = d < 6 ? 5-d : tag0 == (17-d)/3 ? 6+(17-d)%3 : tag1 == (17-d)/3 ? 9+(17-d)%3 : -1; const float jv = at >= 0 ? Zi[at*ZS] : 0.0f; bjv += jv*s_v[d]; }",
    )
    stage += r"""
        s_tag0[i] = tag0; s_tag1[i] = tag1;
        for (int slot = 0; slot < 2; ++slot) {
            const int tag = slot == 0 ? tag0 : tag1;
            for (int r = 0; r < 3; ++r) {
                float v = tag >= 0 ? Zi[(6+3*slot+r)*ZS] : 0.0f;
                if (tag >= 0) {
                    for (int c = 0; c < r; ++c) v -= s_L[BR_LI(3*tag+r,3*tag+c)]*Jr[6+3*slot+c];
                    v *= s_Dinv[3*tag+r];
                }
                Jr[6+3*slot+r] = tag1 == -2 ? NAN : v;
            }
        }
        for (int r = 0; r < 6; ++r) {
            float v = Zi[r*ZS];
            for (int slot = 0; slot < 2; ++slot) {
                const int tag = slot == 0 ? tag0 : tag1;
                if (tag >= 0) for (int c = 0; c < 3; ++c)
                    v -= s_L[BR_LI(12+r,3*tag+c)]*Jr[6+3*slot+c];
            }
            for (int c = 0; c < r; ++c) v -= s_L[BR_LI(12+r,12+c)]*Jr[c];
            Jr[r] = v*s_Dinv[12+r];
        }
        for (int d = 0; d < 12; ++d) s_Yt[d*YS+i] = Jr[d];
    }
    for (int leg = 0; leg < 4; ++leg) {
        const bool member = lane < n_rows && (s_tag0[lane] == leg || s_tag1[lane] == leg);
        const unsigned bits = __ballot_sync(MASK, member);
        if ((lane & 31) == 0) s_members[leg*2+(lane >> 5)] = bits;
    }
    SYNC();
"""
    source = once(source, ink_stage, stage, count=2)
    macros = r"""
#define BR_LI(r,c) ((r)<12 ? 6*((r)/3)+((r)%3)*(((r)%3)+1)/2+(c)%3 : 24+12*((r)-12)+((r)-12)*((r)-11)/2+(c))
#define BR_SLOT(d,i) ((d)>=12 ? (d)-12 : (s_tag0[i]==(d)/3 ? 6+(d)%3 : (s_tag1[i]==(d)/3 ? 9+(d)%3 : -1)))
#define BR_Z(d,i) (BR_SLOT(d,i)>=0 ? s_Yt[BR_SLOT(d,i)*YS+(i)] : 0.0f)
    __shared__ int s_tag0[AM], s_tag1[AM];
    __shared__ unsigned s_members[8];
    for (int e = lane; e < 8; e += NT) s_members[e] = 0u;
"""
    source = once(source, "    __shared__ float s_v[18];", macros + "\n    __shared__ float s_v[18];")
    source = once(source, "float Jr[18];", "float Jr[12];")
    source = once(source, "for (int d = 0; d < 18; ++d) Jr[d] = 0.0f;", "for (int d = 0; d < 12; ++d) Jr[d] = 0.0f;")
    source = once(source, "float s_Yt[1 ? YS * 18 : 4]", "float s_Yt[YS * 12]")
    source = once(source, "float s_L[1 ? (18 * 18 + (0 > 0 ? 0 * 0 : 1)) : 1]", "float s_L[117]")
    source = once(source, "for (int e = lane; e < 18 * 4; e += NT)", "for (int e = lane; e < 12 * 4; e += NT)")
    source = once(source, "const float yv = s_Yt[d * YS + j];", "const float yv = BR_Z(d, j);")
    for dest in ("b", "r"):
        old = f"for (int d = 0; d < 18; ++d) {dest} += Ji[d] * s_dv[d];"
        new = f"for (int d = 0; d < 6; ++d) {dest} += Ji[d] * s_dv[12+d];\n"
        new += f"        for (int slot = 0; slot < 2; ++slot) {{ const int tag = slot == 0 ? s_tag0[i] : s_tag1[i]; if (tag >= 0) for (int d = 0; d < 3; ++d) {dest} += Ji[6+3*slot+d]*s_dv[3*tag+d]; }}"
        source = once(source, old, new)
    old = "for (int d = 0; d < 18; ++d) acc += Ji[d] * s_Yt[d * YS + j];"
    new = """for (int d = 0; d < 6; ++d) acc += Ji[d]*s_Yt[d*YS+j];
                for (int slot = 0; slot < 2; ++slot) {
                    const int tag = slot == 0 ? s_tag0[i] : s_tag1[i];
                    if (tag >= 0) {
                        const int other = s_tag0[j] == tag ? 6 : s_tag1[j] == tag ? 9 : -1;
                        if (other >= 0) for (int d = 0; d < 3; ++d) acc += Ji[6+3*slot+d]*s_Yt[(other+d)*YS+j];
                    }
                }"""
    source = once(source, old, new)
    start = source.index("            // dv = Y^T y with the rows split")
    end = source.index("\n        }\n#endif", start)
    source = (
        source[:start]
        + r"""
            constexpr int NCH = NT/18;
            const int per = (((n_rows+3)/4)+NCH-1)/NCH;
            if (lane < NCH*18) {
                const int d = lane%18, ch = lane/18;
                float acc = 0.0f;
                if (d >= 12) {
                    const float4* y4 = reinterpret_cast<const float4*>(s_y);
                    const float4* z4 = reinterpret_cast<const float4*>(&s_Yt[(d-12)*YS]);
                    float a0=0.0f,a1=0.0f,a2=0.0f,a3=0.0f;
                    const int end = min((n_rows+3)/4,(ch+1)*per);
                    for (int j = ch*per; j < end; ++j) {
                        const float4 y=y4[j], z=z4[j];
                        a0+=z.x*y.x; a1+=z.y*y.y; a2+=z.z*y.z; a3+=z.w*y.w;
                    }
                    acc=(a0+a1)+(a2+a3);
                } else {
                    const int leg = d/3;
                    unsigned long long members = s_members[leg*2] | ((unsigned long long)s_members[leg*2+1]<<32);
                    const int lo=ch*per*4, hi=min(n_rows,(ch+1)*per*4);
                    if (lo>=64 || lo>=hi) members=0;
                    else { members &= ~0ull<<lo; if (hi<64) members &= (1ull<<hi)-1; }
                    while (members) {
                        const int j = __ffsll(members)-1;
                        acc += BR_Z(d,j)*s_y[j];
                        members &= members-1;
                    }
                }
                s_dvp[ch*18+d] = acc;
            }
            SYNC();
            for (int d=lane; d<18; d+=NT) {
                float acc=0.0f;
                for (int ch=0; ch<NCH; ++ch) acc+=s_dvp[ch*18+d];
                s_dv[d]=acc;
            }
            SYNC();"""
        + source[end:]
    )
    source = once(source, "u += ZAT(d, i) * a;", "u += BR_Z(d, i) * a;")
    start = source.index("    if (lane == 0) {\n        for (int a = 18 - 1; a >= 0; --a)")
    end = source.index("    if (0 > 0 && ink_gb >= 0 && lane", start)
    backward = ["    if (lane == 0) {"]
    for col in range(17, -1, -1):
        for row in range(col + 1, 18):
            if index[row, col] >= 0:
                backward.append(f"        s_dv[{col}] -= s_L[{index[row, col]}]*s_dv[{row}];")
        backward.append(f"        s_dv[{col}] *= s_Dinv[{col}];")
    backward.append("    }\n")
    source = source[:start] + "\n".join(backward) + source[end:]
    source = once(source, "v_out.data[global_dof] = s_v[d] + s_dv[d];", "v_out.data[global_dof] = s_v[d] + s_dv[17-d];")
    return source.replace("#undef A_AT", "#undef BR_LI\n#undef BR_SLOT\n#undef BR_Z\n#undef A_AT")


def supported(s):
    """Admit only the complete existing EX1/WR parallel and ordinary fallback recipe."""
    from . import solver_feather_pgs as original  # noqa: PLC0415

    return bool(
        s.model.device.is_cuda
        and not s.model.requires_grad
        and tuple(s.size_groups) == (18,)
        and s.max_world_dofs == 18
        and s.n_arts_by_size[18] == s.world_count
        and s._is_one_solve_art_per_world
        and s._ink_sizes == (18, 0, 0, 0)
        and s._ink_rows == 48
        and s._wr_world_contacts is not None
        and original._MF_EXACT_ROWSUM
        and not (
            original._FPGS_CAPTURE
            or original._GROUPED_CHECK
            or original._CHECK_ROWS
            or original._CHECK_ROWS_FUSED
            or original._DEBUG_CACHE_CMP
            or original._INK_CHECK
            or original._WR_CHECK
            or original._WR_WARM
        )
        and not (
            s._has_free_rigid_bodies
            or s._preelim_active
            or s._regularization_enabled
            or s._local_internal_fast_path
            or s._paired_factor_coordinates
            or s._sparse_diagonal_contact_solve
            or s._mf_warmstart_enabled
            or s._mimic_count
            or s._connect_count
            or s._debug_buffers_enabled
            or s._grouped_tau_mass
            or s._grouped_mass
            or s._fused_k1
        )
        and s._joint_world is None
        and s._row_packets is None
        and s._sparse_factor is None
        and s._single_factor is None
        and (s._jy_world_aliased or s._hinv_jt_writes_world)
        and s._execution_plan.use_tiled_hinv_jt(18)
        and not s._execution_plan.use_diagonal_mass(18)
        and s.pgs_mode == "matrix_free"
        and s.pgs_schedule == "interleaved"
        and s.pgs_iterations == 8
        and s.pgs_velocity_iterations == 0
        and not (s.pgs_warmstart or s.pgs_debug or s.enable_joint_velocity_limits or s.fuse_joint_velocity_limits)
        and s.drive_mode == "augmented"
        and s.friction_mode == "current"
        and s.articulated_contact_response == "immediate"
        and s.mf_gs_response_block_rows == 0
        and s.mf_gs_incremental_rows == 0
        and s.mf_gs_parallel_rows == 48
        and s.mf_gs_parallel_sweeps == 24
        and s.mf_gs_parallel_nesterov
        and s.mf_gs_parallel_matrix_free
        and not s._propagation_contacts_enabled()
    )


def create_owner(solver):
    """Leave unsupported constructors entirely on their original producers."""
    if not supported(solver):
        return None
    try:
        plan, host = make_plan(solver.model, solver)
    except ValueError:
        return None
    return BranchResponse(solver, plan, host)


class BranchResponse:
    """One held factor owner for prediction, compact solve and large-row response."""

    def __init__(self, solver, plan, host):
        from .solver_feather_pgs import _get_pgs_solve_parallel_kernel  # noqa: PLC0415

        self.solver, self.plan, self.host = solver, plan, host
        device, worlds = solver.model.device, solver.world_count
        self.data = BranchData()
        self.data.L = wp.empty((worlds, 1, 117), dtype=float, device=device)
        self.data.valid = wp.zeros(worlds, dtype=int, device=device)
        self.data.status = wp.zeros(worlds, dtype=int, device=device)
        self.refresh_kernel = get_refresh_kernel()
        self.predict_kernel = get_predictor_kernel()
        self.response_kernel = get_response_kernel(solver.dense_max_constraints)
        kernels = []
        for tier, (rows, lo) in enumerate(solver._par_tiers):
            kernel = _get_pgs_solve_parallel_kernel(
                solver.dense_max_constraints,
                solver.mf_max_constraints,
                18,
                device.arch,
                rows=rows,
                sweeps=solver.mf_gs_parallel_sweeps,
                min_rows=lo,
                nesterov=solver.mf_gs_parallel_nesterov,
                tol=solver.mf_gs_parallel_tol,
                matrix_free=True,
                inkernel_response=(18, 0, 0, 0),
                exact_row_sums=True,
                world_rows=True,
                branch_response=True,
            )
            kernel._fpgs_tier = tier
            kernels.append(kernel)
        solver._pgs_solve_mf_gs_incremental_kernels = kernels
        # No canonical H or original-order L survives admission. Current row
        # storage stays allocated for the real >48 fallback, not a shadow factor.
        dummy = wp.empty((1, 1, 1), dtype=float, device=device)
        solver.H_by_size[18] = dummy
        solver.L_by_size[18] = dummy
        solver._H_bufs = None
        solver._J_bufs = None
        solver._memset_stream = None

    def begin(self):
        """Reject new unsupported ownership before retired storage can be touched."""
        if not supported(self.solver):
            raise RuntimeError("Branch response configuration changed; reconstruct and recapture")

    def validate_notification(self, flags):
        """Revalidate topology before the original numeric notification publishes."""
        if flags & (
            ModelFlags.BODY_PROPERTIES
            | ModelFlags.JOINT_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.CONSTRAINT_PROPERTIES
        ):
            host = make_plan(self.solver.model)
            if any(not np.array_equal(host[k], self.host[k]) for k in host):
                raise RuntimeError("Branch response topology changed; reconstruct and recapture")
            if self.solver._mimic_count or self.solver._connect_count:
                raise RuntimeError("Branch response constraint ownership changed; reconstruct and recapture")

    def check(self):
        """Report an invalid factor only at the caller's existing checked boundary."""
        if np.any(self.data.status.numpy()):
            raise RuntimeError("Branch response held factor is invalid")

    def refresh(self, state_aug):
        """Use the original device mass-update mask and enabled augmented K."""
        s = self.solver
        wp.launch_tiled(
            self.refresh_kernel,
            dim=[s.world_count],
            inputs=[
                self.plan,
                self.data,
                s.mass_update_mask,
                state_aug.joint_S_s,
                s.body_I_c,
                s.R_by_size[18],
                s._augmented_drive_row_by_dof,
                s.aug_row_K,
                s.aug_row_counts,
                s.aug_row_dof_index,
                s.articulation_max_dofs,
                int(s._parallel_augmented_drive_topology),
            ],
            block_dim=32,
            device=s.model.device,
        )

    def predict(self, state_aug):
        """Cooperatively apply both packed triangular solves in one world warp."""
        s = self.solver
        wp.launch_tiled(
            self.predict_kernel,
            dim=[s.world_count],
            inputs=[self.plan, self.data, state_aug.joint_tau, state_aug.joint_qdd],
            device=s.model.device,
            block_dim=32,
        )

    def response(self):
        """Produce only actual fallback physical responses in existing buffers."""
        s = self.solver
        wp.launch_tiled(
            self.response_kernel,
            dim=[s.world_count],
            inputs=[
                self.plan,
                self.data,
                s.J_by_size[18],
                s.constraint_count,
                s.mf_constraint_count,
                s._ink_skip_rows(),
                s.Y_by_size[18],
                s.J_world,
                s.Y_world,
                s.diag_by_size[18],
            ],
            block_dim=32,
            device=s.model.device,
        )

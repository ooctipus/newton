# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental register-residual owners for intrinsic32/64-row G1 classes."""

from functools import cache
from types import MethodType

import warp as wp

from .sparse_factor import SparseData, SparsePlan


def _value(name, row, rows):
    """Read a statically addressed owned scalar from its lane."""
    if not 0 <= row < rows:
        return "0.0f"
    return f"__shfl_sync(0xffffffffu,{name}{row // 32},{row % 32})"


def _projection(row, rows):
    """Generate one original ordered projection with constant Gram columns."""
    lane, bank = row % 32, row // 32

    def gram(owned, col):
        """Name a constant Gram column, including unreachable boundary siblings."""
        return f"g{owned}_{col}" if 0 <= col < rows else "0.0f"

    before = _value("lam", row - 1, rows)
    before2 = _value("lam", row - 2, rows)
    after = _value("lam", row + 1, rows)
    sibling_updates = []
    for first, sibling in ((True, row + 1), (False, row - 1)):
        if not 0 <= sibling < rows:
            continue
        updates = "\n".join(f"res{k} += {gram(k, sibling)}*sibling_delta;" for k in range(rows // 32))
        sibling_updates.append(
            f"""if ({"first" if first else "!first"}) {{
                if (lane=={sibling % 32}) {{ lam{sibling // 32}+=sibling_delta; applied{sibling // 32}+=sibling_delta; }}
                {updates}
            }}"""
        )
    updates = "\n".join(f"res{k} += {gram(k, row)}*delta;" for k in range(rows // 32))
    return f"""
    if (count>{row}) {{
        const bool friction=(friction{bank}&(1u<<{lane}))!=0;
        const bool first=(first{bank}&(1u<<{lane}))!=0;
        if (friction && iteration<friction_start) {{ if(lane=={lane})lam{bank}=0.0f; }}
        else {{
            const float denom={_value("denom", row, rows)};
            if (denom>0.0f) {{
                const float old={_value("lam", row, rows)};
                float next=old+omega*(-{_value("res", row, rows)}/denom);
                float sibling_delta=0.0f;
                if (!friction) next=fmaxf(next,0.0f);
                else {{
                    const float parent_lam=first?{before}:{before2};
                    const float sibling_lam=first?{after}:{before};
                    const float radius=fmaxf({_value("mu", row, rows)}*parent_lam,0.0f);
                    if(radius<=0.0f) next=0.0f;
                    else {{
                        const float mag=sqrtf(next*next+sibling_lam*sibling_lam);
                        if(mag>radius) {{
                            const float scale=radius/mag;
                            next*=scale;
                            sibling_delta=sibling_lam*scale-sibling_lam;
                        }}
                    }}
                }}
                const float delta=next-old;
                if(sibling_delta!=0.0f) {{
                    {"".join(sibling_updates)}
                    changed=1;
                }}
                if(lane=={lane}) {{lam{bank}=next;applied{bank}+=delta;}}
                if(delta!=0.0f) {{ {updates} changed=1; }}
            }}
        }}
    }}
    """


def cuda_source(capacity, rows):
    """Build named Gram scalars; no runtime-indexed local Gram array exists."""
    if rows not in (32, 64) or capacity < rows:
        raise ValueError("Register residual supports intrinsic32/64 classes")
    banks = rows // 32
    stride = rows + 1
    initialization = []
    for k in range(banks):
        initialization.append(
            f"""
    const int row{k}=lane+{32 * k};
    const bool live{k}=row{k}<count;
    const int type{k}=live{k}?row_type.data[base+row{k}]:0;
    const int par{k}=live{k}?parent.data[base+row{k}]:-1;
    float lam{k}=live{k}?impulses.data[base+row{k}]:0.0f;
    float applied{k}=0.0f;
    float res{k}=live{k}?d.incident.data[base+row{k}]+rhs.data[base+row{k}]:0.0f;
    const float denom{k}=live{k}?diagonal.data[base+row{k}]:1.0f;
    const float mu{k}=live{k}?mu.data[base+row{k}]:0.0f;
    if(live{k} && type{k}!=0 && type{k}!=2 && type{k}!=3)bad=1;
    if(live{k} && type{k}==2) {{
        if(par{k}<0 || par{k}+2>=count || (row{k}!=par{k}+1 && row{k}!=par{k}+2))bad=1;
        else if(row_type.data[base+par{k}]!=0 || row_type.data[base+par{k}+1]!=2 || row_type.data[base+par{k}+2]!=2 || parent.data[base+par{k}+1]!=par{k} || parent.data[base+par{k}+2]!=par{k})bad=1;
    }}
    if(live{k} && (lam{k}!=0.0f || !isfinite(res{k}) || !isfinite(denom{k}) || (type{k}!=2 && res{k}<0.0f)))stationary=0;
    const unsigned friction{k}=__ballot_sync(0xffffffffu,live{k} && type{k}==2);
    const unsigned first{k}=__ballot_sync(0xffffffffu,live{k} && type{k}==2 && par{k}==row{k}-1);
    """
        )
    scatter = "\n".join(
        f"""if(live{k}) {{
        const int tpl=d.support.data[base+row{k}], length=p.support_count.data[tpl];
        for(int a=0;a<length;++a) {{
            const int node=p.support_nodes.data[tpl*18+a];
            z[node*{stride}+row{k}]=d.Z.data[(base+row{k})*18+a];
        }}
    }}"""
        for k in range(banks)
    )
    declarations = "\n".join(f"float g{k}_{j}=0.0f;" for k in range(banks) for j in range(rows))
    formation = []
    for k in range(banks):
        dots = "\n".join(f"g{k}_{j} += value*z[node*{stride}+{j}];" for j in range(rows))
        formation.append(
            f"""if(live{k}) {{
            const int tpl=d.support.data[base+row{k}], length=p.support_count.data[tpl];
            for(int a=0;a<length;++a) {{
                const int node=p.support_nodes.data[tpl*18+a];
                const float value=z[node*{stride}+row{k}];
                {dots}
            }}
        }}"""
        )
    store = "\n".join(
        f"if(live{k}) {{impulses.data[base+row{k}]=lam{k};applied[row{k}]=applied{k};}}" for k in range(banks)
    )
    return f"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31, art=p.group_to_art.data[group], world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art], count=counts.data[world], base=world*{capacity};
    if(count<={0 if rows == 32 else 32} || count>{rows})return;
    if(!d.valid.data[world] || d.status.data[world] || count>{capacity})return;
    int bad=0, stationary=isfinite(omega) && omega>=0.0f;
    {"".join(initialization)}
    if(__ballot_sync(0xffffffffu,bad)) {{if(lane==0)atomicOr(&d.status.data[world],4);return;}}
    if(__all_sync(0xffffffffu,stationary)) {{
        for(int col=lane;col<43;col+=32)vout.data[start+col]=vhat.data[start+col];
        return;
    }}
    __shared__ float z[43*{stride}], applied[{rows}], du[43];
    for(int i=lane;i<43*{stride};i+=32)z[i]=0.0f;
    __syncwarp();
    {scatter}
    __syncwarp();
    {declarations}
    {"".join(formation)}
    for(int iteration=0;iteration<iterations;++iteration) {{
        int changed=0;
        {"".join(_projection(j, rows) for j in range(rows))}
        if(iteration>=friction_start && !changed)break;
    }}
    {store}
    __syncwarp();
    for(int node=lane;node<43;node+=32) {{
        float value=0.0f;
        for(int row=0;row<count;++row)value+=z[node*{stride}+row]*applied[row];
        du[node]=value;
    }}
    __syncwarp();
    for(int col=lane;col<43;col+=32) {{
        float value=0.0f;
        for(int k=0;k<p.inverse_count.data[col];++k) {{
            const int row=p.inverse_nodes.data[col*18+k];
            value+=d.W.data[group*434+p.index.data[row*43+col]]*du[row];
        }}
        vout.data[start+42-col]=vhat.data[start+42-col]+value;
    }}
#endif
"""


def cpu_source(capacity, rows):
    """Exercise the same applied-delta recurrence on CPU for physical controls."""
    return f"""
#if !defined(__CUDA_ARCH__)
    const int art=p.group_to_art.data[group], world=p.art_to_world.data[art];
    const int start=p.art_dof_start.data[art], count=counts.data[world], base=world*{capacity};
    if(count<={0 if rows == 32 else 32} || count>{rows})return;
    if(!d.valid.data[world] || d.status.data[world] || count>{capacity})return;
    float z[43][{rows}]={{}}, gram[{rows}][{rows}]={{}};
    float lam[{rows}]={{}}, residual[{rows}]={{}}, applied[{rows}]={{}}, du[43]={{}};
    bool stationary=isfinite(omega) && omega>=0.0f;
    for(int r=0;r<count;++r) {{
        const int type=row_type.data[base+r], par=parent.data[base+r];
        bool bad=type!=0 && type!=2 && type!=3;
        if(type==2) {{
            if(par<0 || par+2>=count || (r!=par+1 && r!=par+2))bad=true;
            else if(row_type.data[base+par]!=0 || row_type.data[base+par+1]!=2 || row_type.data[base+par+2]!=2 || parent.data[base+par+1]!=par || parent.data[base+par+2]!=par)bad=true;
        }}
        if(bad) {{d.status.data[world]|=4;return;}}
        lam[r]=impulses.data[base+r];
        residual[r]=d.incident.data[base+r]+rhs.data[base+r];
        if(lam[r]!=0.0f || !isfinite(residual[r]) || !isfinite(diagonal.data[base+r]) || (type!=2 && residual[r]<0.0f))stationary=false;
    }}
    if(stationary) {{for(int k=0;k<43;++k)vout.data[start+k]=vhat.data[start+k];return;}}
    for(int r=0;r<count;++r) {{
        const int tpl=d.support.data[base+r], length=p.support_count.data[tpl];
        for(int k=0;k<length;++k)z[p.support_nodes.data[tpl*18+k]][r]=d.Z.data[(base+r)*18+k];
    }}
    for(int r=0;r<count;++r) {{
        const int tpl=d.support.data[base+r], length=p.support_count.data[tpl];
        for(int k=0;k<length;++k) {{
            const int node=p.support_nodes.data[tpl*18+k];
            for(int j=0;j<count;++j)gram[r][j]+=z[node][r]*z[node][j];
        }}
    }}
    for(int iteration=0;iteration<iterations;++iteration) {{
        bool changed=false;
        for(int r=0;r<count;++r) {{
            const int type=row_type.data[base+r];
            if(type==2 && iteration<friction_start) {{lam[r]=0.0f;continue;}}
            const float denom=diagonal.data[base+r];
            if(!(denom>0.0f))continue;
            const float old=lam[r];
            float next=old+omega*(-residual[r]/denom), sibling_delta=0.0f;
            int sibling=-1;
            if(type!=2)next=wp::max(next,0.0f);
            else {{
                const int par=parent.data[base+r];
                const float radius=wp::max(mu.data[base+r]*lam[par],0.0f);
                if(radius<=0.0f)next=0.0f;
                else {{
                    sibling=r==par+1?par+2:par+1;
                    const float other=lam[sibling], mag=sqrtf(next*next+other*other);
                    if(mag>radius) {{const float scale=radius/mag;next*=scale;sibling_delta=other*scale-other;}}
                }}
            }}
            const float delta=next-old;
            if(sibling_delta!=0.0f) {{
                lam[sibling]+=sibling_delta;applied[sibling]+=sibling_delta;
                for(int j=0;j<count;++j)residual[j]+=gram[j][sibling]*sibling_delta;
                changed=true;
            }}
            lam[r]=next;applied[r]+=delta;
            if(delta!=0.0f) {{for(int j=0;j<count;++j)residual[j]+=gram[j][r]*delta;changed=true;}}
        }}
        if(iteration>=friction_start && !changed)break;
    }}
    for(int r=0;r<count;++r)impulses.data[base+r]=lam[r];
    for(int node=0;node<43;++node)for(int r=0;r<count;++r)du[node]+=z[node][r]*applied[r];
    for(int col=0;col<43;++col) {{
        float value=0.0f;
        for(int k=0;k<p.inverse_count.data[col];++k) {{const int row=p.inverse_nodes.data[col*18+k];value+=d.W.data[group*434+p.index.data[row*43+col]]*du[row];}}
        vout.data[start+42-col]=vhat.data[start+42-col]+value;
    }}
#endif
"""


@cache
def get_solve_kernel(capacity, rows):
    """Compile one constant-size residual class with original sparse arguments."""
    source = cuda_source(capacity, rows) + cpu_source(capacity, rows)

    @wp.func_native(source)
    def native(
        group: int,
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
        impulses: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        vhat: wp.array[float],
        vout: wp.array[float],
    ): ...

    def solve(
        p: SparsePlan,
        d: SparseData,
        counts: wp.array[int],
        rhs: wp.array2d[float],
        diagonal: wp.array2d[float],
        impulses: wp.array2d[float],
        row_type: wp.array2d[int],
        parent: wp.array2d[int],
        mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start: int,
        vhat: wp.array[float],
        vout: wp.array[float],
    ):
        group, _ = wp.tid()
        native(
            group,
            p,
            d,
            counts,
            rhs,
            diagonal,
            impulses,
            row_type,
            parent,
            mu,
            iterations,
            omega,
            friction_start,
            vhat,
            vout,
        )

    solve.__name__ = solve.__qualname__ = f"sparse_register_residual43_r{rows}_c{capacity}"
    return wp.kernel(enable_backward=False, module="unique")(solve)


def install(owner):
    """Install three count-filtered owners without a queue or new allocation."""
    from .sparse_factor_rows import get_solve_kernel as get_original  # noqa: PLC0415

    if getattr(owner, "register_residual_kernels", None) is not None:
        return
    capacity = owner.solver.dense_max_constraints
    if capacity < 64:
        raise ValueError("Register residual requires the complete64-row class capacity")
    owner.register_residual_kernels = (
        get_solve_kernel(capacity, 32),
        get_solve_kernel(capacity, 64),
        get_original(capacity, minimum_count=64),
    )
    owner.solve = MethodType(_solve, owner)


def _solve(owner, rhs, iterations, omega, friction_start):
    """Charge all three disjoint launches while keeping original arguments."""
    s = owner.solver
    arguments = [
        owner.plan,
        owner.data,
        s.constraint_count,
        rhs,
        s.diag,
        s.impulses,
        s.row_type,
        s.row_parent,
        s.row_mu,
        iterations,
        omega,
        friction_start,
        s.v_hat,
        s.v_out,
    ]
    for kernel in owner.register_residual_kernels:
        wp.launch_tiled(kernel, dim=[s.world_count], inputs=arguments, block_dim=32, device=s.model.device)

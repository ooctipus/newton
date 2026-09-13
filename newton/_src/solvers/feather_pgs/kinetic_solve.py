# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original dense-eight recurrence in offset kinetic coordinates.

The CUDA recurrence and independent-component metadata predicates are recovered
from pinned original source. Only representation entry/exit and held-operator
qualification change. No canonical L23 or physical Y is built for private rows.
The opt-in CUDA path also orders shared-impulse transactions with eight warp
fences and removes three dead temporary stores; its arithmetic is unchanged.
"""

import ast
import functools
import re

import warp as wp

from .kinetic_rows_types import DenseRowOutput, RowState
from .kinetic_solve_types import KineticMFData, KineticSolveData
from .kinetic_source import checked_definitions
from .kinetic_types import HeldKineticOperator, KineticPlan


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return false;
#else
    return true;
#endif
""")
def _is_cpu() -> bool: ...


def snippet(filename, function, values=None):
    """Recover one literal original native body under exact source identity."""
    target = checked_definitions(filename, (function,))[function]
    item = next(
        node
        for node in target.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "snippet" for t in node.targets)
    )
    return eval(
        compile(ast.Expression(item.value), filename, "eval"),
        {"__builtins__": {"str": str, "bool": bool}},
        values or {},
    )


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Original source seam no longer unique: " + old[:80])
    return source.replace(old, new)


def _impulse_transaction_edits():
    """The eleven reviewed, reversible offset shared-impulse corrections."""
    temporary = (
        "\n                        s_lam[tangent1] = new_tangent1;",
        "\n                        s_lam[tangent2] = new_tangent2;",
        "\n                    s_lam[i] = new_impulse;",
    )
    siblings = (
        "                        const float old_tangent2 = s_lam[tangent2];",
        "                        const float old_tangent1 = s_lam[tangent1];",
        "                    const float sibling_impulse = s_lam[sibling];",
    )
    final = (
        "\n                s_lam[i] = new_impulse;",
        "\n                    s_lam[normal] = new_normal;",
        "\n                    s_lam[tangent1] = new_tangent1;",
        "\n                    s_lam[tangent2] = new_tangent2;",
        "\n            s_lam[i] = new_impulse;",
    )
    edits = [(old, f"\n                        /* UNUSED_OWN_STORE_{i} */") for i, old in enumerate(temporary)]
    sync = "\n                        __syncwarp(MASK); /* SIBLING_READ_COMPLETE */"
    edits.extend((old, old + sync) for old in siblings)
    for index, old in enumerate(final):
        indentation = old[1 : old.index("s_lam")]
        barrier = f"\n{indentation}__syncwarp(MASK); /* IMPULSE_READ_COMPLETE_{index} */"
        edits.append((old, barrier + old))
    return tuple(edits)


def recover_impulse_transactions(source):
    """Undo only the declared read-complete fences and dead-store removals."""
    for old, new in reversed(_impulse_transaction_edits()):
        if source.count(new) != 1:
            raise ValueError("Ambiguous impulse transaction recovery")
        source = source.replace(new, old)
    return source


def repair_impulse_transactions(source):
    """Complete every old-impulse read before any lane publishes its update."""
    if any(marker in source for marker in ("IMPULSE_READ_COMPLETE", "SIBLING_READ_COMPLETE", "UNUSED_OWN_STORE")):
        raise ValueError("Require the original uncorrected impulse source")
    result = source
    for old, new in _impulse_transaction_edits():
        if result.count(old) != 1:
            raise ValueError("Changed impulse transaction seam: " + old)
        result = result.replace(old, new)
    if recover_impulse_transactions(result) != source:
        raise ValueError("Transaction correction changed an unrecorded statement")
    return result


_T = r"""
    auto t=[&](int row,int col){
        if(row<7){if(col<7)return col<=row?held.T.data[world*180+40+row*(row+1)/2+col]:0.0f;
            return held.T.data[world*180+68+((col-7)/4)*28+row*4+(col-7)%4];}
        if(col<7 || (row-7)/4!=(col-7)/4 || (col-7)%4>(row-7)%4)return 0.0f;
        const int r=(row-7)%4;return held.T.data[world*180+((row-7)/4)*10+r*(r+1)/2+(col-7)%4];
    };
"""

_GUARD = r"""
    if(world<0||world>=state.dense_count.shape[0])return;
    if(rows.global_status.data[0]!=0||rows.status.data[world]!=0||held.valid.data[world]==0||
       held.generation.data[world]!=state.held_generation.data[world]||state.predictor_status.data[world]!=0||
       solve.iterations!=8||solve.omega!=1.0f||solve.iteration_offset!=0||solve.friction_start_iteration<0){
#if defined(__CUDA_ARCH__)
        if((threadIdx.x&31)==0)solve.status.data[world]=1;
#else
        solve.status.data[world]=1;
#endif
        return;
    }
    if(solve.selector.data[world]>0)return;
    const bool split_component=solve.selector.data[world]<0;
    const int po=state.primary_offset.data[world],so=state.secondary_offset.data[world];
    if(!((po==0&&so==23)||(po==6&&so==0))){
#if defined(__CUDA_ARCH__)
        if((threadIdx.x&31)==0)solve.status.data[world]=1;
#else
        solve.status.data[world]=1;
#endif
        return;
    }
"""


def cuda_recurrence():
    """Original offset recurrence with the explicit shared-transaction repair."""
    checked_definitions(
        "kinetic_solve.py",
        ("_impulse_transaction_edits", "recover_impulse_transactions", "repair_impulse_transactions"),
    )
    original = snippet(
        "solver_feather_pgs.py",
        "_get_pgs_solve_paired_factor_kernel",
        dict(  # noqa: C408 - preserve the checked original source-construction AST
            W=2,
            M=192,
            D=29,
            P=23,
            S=6,
            contact_type=0,
            friction_type=2,
            joint_limit_type=3,
            joint_velocity_limit_type=4,
            contact_triples=True,
        ),
    )
    body = original.split("#if defined(__CUDA_ARCH__)\n", 1)[1].rsplit("#endif", 1)[0]
    body = replace_once(
        body,
        "    if (m > 192) m = 192;\n    if (m == 0 || world_mf_constraint_count.data[world] != 0) return;",
        "    if (m < 0 || m > 192) { if(lane==0) solve.status.data[world]=1; return; }",
    )
    begin = body.index("    const int global_dof =")
    end = body.index("    __syncwarp(MASK);", begin)
    body = (
        body[:begin]
        + r"""
    const int primary_lane=lane-primary_offset,secondary_lane=lane-secondary_offset;
    const int physical_dof=(primary_lane>=0&&primary_lane<23)?primary_lane:
        ((secondary_lane>=0&&secondary_lane<6)?23+secondary_lane:-1);
    const int global_dof=physical_dof>=0?plan.dof_ids.data[world*35+physical_dof]:-1;
    float factor_velocity=0.0f;
"""
        + body[end:]
    )
    begin = body.index("    float physical_output = 0.0f;")
    end = body.index("    for (int i = lane; i < m; i += 32) world_impulses", begin)
    body = (
        body[:begin]
        + r"""
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
        + body[end:]
    )
    aliases = r"""
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
    return repair_impulse_transactions(aliases + body)


_CPU = r"""
    const int m=state.dense_count.data[world];
    if(m<0||m>192){solve.status.data[world]=1;return;}
    float du[29]={},lambda[192];
    for(int i=0;i<m;++i)lambda[i]=rows.impulses.data[world*192+i];
    for(int iter=0;iter<solve.iterations;++iter){bool changed=false;const int global_iter=iter+solve.iteration_offset;
        for(int i=0;i<m;++i){const int id=world*192+i,type=rows.row_type.data[id];
            if(type==2&&global_iter<solve.friction_start_iteration){lambda[i]=0.0f;continue;}
            int parent=-1,sibling=-1;if(type==2){parent=rows.row_parent.data[id];sibling=i==parent+1?parent+2:parent+1;
                if(parent<0||sibling<0||sibling>=m){solve.status.data[world]=1;return;}
                if(lambda[parent]<=0.0f&&lambda[i]==0.0f&&lambda[sibling]==0.0f)continue;}
            const float denom=rows.diag.data[id];if(denom<=0.0f)continue;
            float residual=0.0f;for(int d=0;d<29;++d)residual+=rows.response.data[id*29+d]*du[d];residual+=rows.r0.data[id];
            const float raw_delta=-residual/denom,old=lambda[i];float next=old+solve.omega*raw_delta,delta=0.0f,sibling_delta=0.0f;
            if(type==0||type==3){if(next<0.0f)next=0.0f;delta=next-old;}
            else if(type==4){if(residual<0.0f){delta=raw_delta;next=raw_delta;}else next=0.0f;}
            else if(type==2){const float radius=wp::max(rows.row_mu.data[id]*lambda[parent],0.0f);
                if(radius<=0.0f)next=0.0f;
                else{lambda[i]=next;const float sibling_impulse=lambda[sibling],length=sqrtf(next*next+sibling_impulse*sibling_impulse);
                    if(length>radius){const float scale=radius/length;next*=scale;const float other=sibling_impulse*scale;sibling_delta=other-sibling_impulse;lambda[sibling]=other;}}
                delta=next-old;
            }else delta=next-old;
            lambda[i]=next;
            if(sibling_delta!=0.0f){changed=true;for(int d=0;d<29;++d)du[d]+=rows.response.data[(world*192+sibling)*29+d]*sibling_delta;}
            if(delta!=0.0f){changed=true;for(int d=0;d<29;++d)du[d]+=rows.response.data[id*29+d]*delta;}
        }
        if(global_iter>=solve.friction_start_iteration&&!changed)break;
    }
    float result[29];bool good=true;
    for(int col=0;col<23;++col){float v=0.0f;for(int row=0;row<23;++row)v+=t(row,col)*du[po+row];result[col]=v+state.v_hat.data[plan.dof_ids.data[world*35+col]];good=good&&isfinite(result[col]);}
    const int group=plan.secondary_group.data[world];
    for(int col=0;col<6;++col){float v=0.0f;for(int row=col;row<6;++row)v+=held.inverse6.data[group*36+row*6+col]*du[so+row];result[23+col]=v+state.v_hat.data[plan.dof_ids.data[world*35+23+col]];good=good&&isfinite(result[23+col]);}
    if(!good){solve.status.data[world]=2;return;}
    for(int d=0;d<(split_component?23:29);++d)solve.v_out.data[plan.dof_ids.data[world*35+d]]=result[d];
    for(int i=0;i<m;++i)rows.impulses.data[world*192+i]=lambda[i];
    solve.status.data[world]=0;
"""


@functools.cache
def get_solve_kernel(arch):
    """Original two-warp CTA; ceil(active-capacity/2) tiles, block_dim64."""
    source = _GUARD + _T + "\n#if defined(__CUDA_ARCH__)\n" + cuda_recurrence() + "\n#else\n" + _CPU + "\n#endif\n"

    @wp.func_native(source)
    def solve_native(
        world: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_offset_eight(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
    ):
        block, logical = wp.tid()
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            return
        if _is_cpu():
            if logical != 0:
                return
            for half in range(2):
                cpu_index = block * 2 + half
                if cpu_index < state.active_count[0]:
                    cpu_world = state.active_worlds[cpu_index]
                    solve_native(cpu_world, plan, held, state, rows, solve)
            return
        index = block * 2 + logical // 32
        if index >= state.active_count[0]:
            return
        world = state.active_worlds[index]
        solve_native(world, plan, held, state, rows, solve)

    return kinetic_offset_eight


def qualifier_source():
    """Retain original row/MF predicate; substitute directly held T and Z."""
    source = snippet("independent_components.py", "get_prepare_kernel")
    source = re.sub(r"\bmf\b", "mf_count", source)
    source = replace_once(
        source,
        "|| group < 0 || group >= data.L.shape[0]) continue;",
        ") continue;\n        if (held.valid.data[world]==0 || held.generation.data[world]!=state.held_generation.data[world] || rows.status.data[world]!=0 || rows.global_status.data[0]!=0) { if(tid==0) solve.status.data[world]=1; continue; }",
    )
    source = replace_once(
        source,
        "for (int k = tid; k < 23 * 23; k += nt)",
        "for (int k = tid; k < 180; k += nt)",
    )
    source = replace_once(source, "data.L.data[group * 23 * 23 + k]", "held.T.data[world * 180 + k]")
    begin = source.index("            for (int row = warp; row < m; row += nw)")
    end = source.index("            if (tid == 0) data.selector.data[world] = -1;", begin)
    source = source[:begin] + source[end:]
    source = source.replace("data.world_count", "state.dense_count.shape[0]")
    names = {
        "counts": "state.dense_count",
        "mf_counts": "state.mf_count",
        "primary_offset": "state.primary_offset",
        "secondary_offset": "state.secondary_offset",
        "primary_group": "plan.primary_group",
        "selector": "solve.selector",
        "J": "rows.physical_J",
        "Y": "rows.response",
        "rhs": "rows.rhs",
        "diag": "rows.diag",
        "mu": "rows.row_mu",
        "row_type": "rows.row_type",
        "row_parent": "rows.row_parent",
        "mf_meta": "mf.meta",
        "mf_mu": "mf.mu",
        "mf_J_a": "mf.J_a",
        "mf_J_b": "mf.J_b",
        "mf_MiJt_a": "mf.MiJt_a",
        "mf_MiJt_b": "mf.MiJt_b",
    }
    for old, new in sorted(names.items(), key=lambda item: -len(item[0])):
        source = source.replace("data." + old + ".", new + ".")
    return source


@functools.cache
def get_qualify_kernel(arch):
    """Same original128-thread qualifier workers, without L/Y materialization."""

    @wp.func_native(qualifier_source())
    def qualify(
        worker: int,
        logical_lane: int,
        workers: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        mf: KineticMFData,
        solve: KineticSolveData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_independent_qualify(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        mf: KineticMFData,
        solve: KineticSolveData,
        workers: int,
    ):
        worker, logical = wp.tid()
        qualify(worker, logical, workers, plan, held, state, rows, mf, solve)

    return kinetic_independent_qualify


_MATERIALIZE = (
    _T
    + r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
    __shared__ float z[32];
#else
    if(logical!=0)return;
    const int lane=0,stride=1;
    float z[32];
#endif
    if(solve.selector.data[world]<=0)return;
    if(rows.status.data[world]!=0||rows.global_status.data[0]!=0||held.valid.data[world]==0||held.generation.data[world]!=state.held_generation.data[world]){
        if(lane==0)solve.status.data[world]=1;return;}
    const int count=state.dense_count.data[world],po=state.primary_offset.data[world],so=state.secondary_offset.data[world],group=plan.secondary_group.data[world];
    if(count<0||count>192||!((po==0&&so==23)||(po==6&&so==0))){if(lane==0)solve.status.data[world]=1;return;}
    for(int row=0;row<count;++row){const int id=world*192+row;
        for(int d=lane;d<29;d+=stride)z[d]=rows.response.data[id*29+d];
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
        bool good=true;
        for(int d=lane;d<29;d+=stride){float y=0.0f;int coord=0;
            if(d<23){coord=po+d;for(int k=0;k<23;++k)y+=t(k,d)*z[po+k];}
            else{const int col=d-23;coord=so+col;for(int k=col;k<6;++k)y+=held.inverse6.data[group*36+k*6+col]*z[so+k];}
            rows.response.data[id*29+coord]=y;good=good&&isfinite(y);
        }
#if defined(__CUDA_ARCH__)
        good=__all_sync(0xffffffffu,good);__syncwarp(0xffffffffu);
#endif
        if(!good){if(lane==0)solve.status.data[world]=2;return;}
        if(lane==0){float diag=rows.row_cfm.data[id];for(int d=0;d<29;++d)diag+=rows.physical_J.data[id*29+d]*rows.response.data[id*29+d];rows.diag.data[id]=diag;
            z[31]=isfinite(diag)?1.0f:0.0f;}
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
        if(z[31]==0.0f){if(lane==0)solve.status.data[world]=2;return;}
    }
    if(lane==0)solve.status.data[world]=0;
"""
)


@functools.cache
def get_materialize_kernel(arch):
    """In-place physicalY for positive-selector original generic worlds only."""

    @wp.func_native(_MATERIALIZE)
    def materialize(
        world: int,
        logical: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_coupled_materialize(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
    ):
        index, logical = wp.tid()
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            return
        if index >= state.active_count[0]:
            return
        world = state.active_worlds[index]
        if world >= 0 and world < state.dense_count.shape[0]:
            materialize(world, logical, plan, held, state, rows, solve)

    return kinetic_coupled_materialize

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original dense-eight recurrence in offset kinetic coordinates.

Shared numerical factories supply the CUDA recurrence and independent-component
metadata predicates. Only representation entry/exit and held-operator
qualification change. No canonical L23 or physical Y is built for private rows.
The CUDA path orders shared-impulse publication with warp fences. Both CPU
and CUDA use the current solver's paired tangent projection.
"""

import functools

import warp as wp

from .friction import get_pair_source
from .kinetic_rows_types import DenseRowOutput, RowState
from .kinetic_solve_types import KineticMFData, KineticSolveData
from .kinetic_types import HeldKineticOperator, KineticPlan

# Retain the checked source view for the compact owner.
FRICTION_PAIR_CUDA = get_pair_source()


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return false;
#else
    return true;
#endif
""")
def _is_cpu() -> bool: ...


def _impulse_transaction_edits():
    """Fence shared publication after all lanes read the previous impulses."""
    final = (
        "\n                s_lam[i] = new_impulse;",
        "\n                    s_lam[normal] = new_normal;",
        "\n                    s_lam[tangent1] = 0.0f;",
        "\n                s_lam[tangent1] = new_tangent1;",
        "\n                s_lam[sibling] = new_sibling_impulse;",
        "\n            s_lam[i] = new_impulse;",
    )
    edits = []
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
    """Build the common paired recurrence with explicit held-coordinate entry and exit."""
    from .paired_factor_source import get_source  # noqa: PLC0415

    return get_source(192, 29, 23, 6, contact_triples=True, kinetic=True)


_CPU = (
    get_pair_source(cpu=True)
    + r"""
    const int m=state.dense_count.data[world];
    if(m<0||m>192){solve.status.data[world]=1;return;}
    float du[29]={},lambda[192];
    for(int i=0;i<m;++i)lambda[i]=rows.impulses.data[world*192+i];
    for(int iter=0;iter<solve.iterations;++iter){bool changed=false;const int global_iter=iter+solve.iteration_offset;
        for(int i=0;i<m;++i){const int id=world*192+i,type=rows.row_type.data[id];
            if(type==2&&global_iter<solve.friction_start_iteration){lambda[i]=0.0f;continue;}
            int parent=-1,sibling=-1;if(type==2){parent=rows.row_parent.data[id];sibling=parent+2;
                if(parent<0||sibling<0||sibling>=m){solve.status.data[world]=1;return;}
                if(i!=parent+1)continue;}
            const float denom=rows.diag.data[id];if(denom<=0.0f&&type!=2)continue;
            float residual=0.0f;for(int d=0;d<29;++d)residual+=rows.response.data[id*29+d]*du[d];residual+=rows.r0.data[id];
            const float raw_delta=denom>0.0f?-residual/denom:0.0f,old=lambda[i];float next=old+solve.omega*raw_delta,delta=0.0f,sibling_delta=0.0f;
            if(type==0||type==3){if(next<0.0f)next=0.0f;delta=next-old;}
            else if(type==4){if(residual<0.0f){delta=raw_delta;next=raw_delta;}else next=0.0f;}
            else if(type==2){float normal_load=lambda[parent];
                for(int patch=rows.row_parent.data[world*192+parent];patch>=0&&patch!=parent;patch=rows.row_parent.data[world*192+patch])normal_load+=lambda[patch];
                const float radius=wp::max(rows.row_mu.data[id]*normal_load,0.0f);
                const int sid=world*192+sibling;float other_residual=rows.r0.data[sid],cross=0.0f;
                for(int d=0;d<29;++d){const float z=rows.response.data[sid*29+d];other_residual+=z*du[d];cross+=rows.response.data[id*29+d]*z;}
                const auto pair=friction_pair_candidate(denom,cross,rows.diag.data[sid],residual,other_residual,old,lambda[sibling],radius,solve.omega);
                const float length=sqrtf(pair[0]*pair[0]+pair[1]*pair[1]);const float scale=length>radius?radius/length:1.0f;
                next=pair[0]*scale;const float other=pair[1]*scale;sibling_delta=other-lambda[sibling];lambda[sibling]=other;
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
)


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
    """Qualify current held rows through the shared component predicates."""
    from .component_qualification import get_source  # noqa: PLC0415

    return get_source("kinetic")


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

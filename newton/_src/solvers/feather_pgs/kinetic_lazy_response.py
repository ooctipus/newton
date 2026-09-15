# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full-visit physical increments with solve-local first-use response storage.

Current J and incident velocity remain eager. Positive mixed worlds retain
the original numerical qualification, bridge and general solver. Only MF0
and independent dense components defer held actions until a row needs them.
"""

import functools
import inspect
import math
import textwrap
from types import SimpleNamespace

import warp as wp

from . import kinetic_compact_coupled as compact
from . import kinetic_current_contact as contact
from . import kinetic_hybrid as hybrid
from . import kinetic_rows as rows_source
from . import kinetic_rows_triplet as triplet
from . import kinetic_solve as original
from .kinetic_rows_types import ArmMap, DenseRowOutput, PrefixInput, RawRowInput, RowSettings, RowState
from .kinetic_solve_types import KineticMFData, KineticSolveData
from .kinetic_types import CurrentKineticCache, HeldKineticOperator, KineticPlan

replace = original.replace_once


@functools.cache
def get_arm_kernel(arch):
    """Retain row reset and epoch validation, without the unused arm response."""
    kernel = rows_source.get_arm_kernel(arch)
    source = textwrap.dedent(inspect.getsource(kernel.func))
    source = replace(source, "def kinetic_row_arm(", "def kinetic_lazy_row_begin(")
    begin = source.index("    for row in range(lane, 7, stride):")
    end = source.index("    if lane == 0:\n        arm.state_generation", begin)
    source = source[:begin] + source[end:]
    return contact._compile(source, "kinetic_lazy_row_begin", dict(rows_source.__dict__))


def prefix_source():
    """Publish the exact scalar prefix J/RHS without touching response/diag."""
    source = rows_source._PREFIX
    begin = source.index("        for(int row=lane;row<29;row+=stride){")
    end = source.index("        sync();", begin)
    source = (
        source[:begin]
        + r"""
        for(int row=lane;row<29;row+=stride){const int d=row-po;
            out.physical_J.data[(world*settings.dense_capacity+slot)*29+row]=(d==col?sign:0.0f);}
"""
        + source[end:]
    )
    source = replace(source, "float diag=settings.cfm;for(int k=0;k<29;++k)diag+=s[k];", "float diag=settings.cfm;")
    source = replace(source, "out.diag.data[id]=diag;", "")
    return source


@functools.cache
def get_prefix_kernel(arch):
    """Keep the original prefix descriptor ABI and launch ownership."""

    @wp.func_native(prefix_source())
    def emit(
        address: wp.uint64,
        index: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        prefix: PrefixInput,
        state: RowState,
        settings: RowSettings,
        out: DenseRowOutput,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_lazy_prefix(
        plan: KineticPlan,
        held: HeldKineticOperator,
        prefix: PrefixInput,
        state: RowState,
        settings: RowSettings,
        out: DenseRowOutput,
    ):
        index, logical = wp.tid()
        lanes = rows_source._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = rows_source._storage()
        emit(address, index, plan, held, prefix, state, settings, out)
        rows_source._release(address)

    return kinetic_lazy_prefix


def contact_source():
    """Retain cached geometry and three current physical rows, not held actions."""
    source = contact._contact_source()
    source = replace(source, "const bool physical=state.mf_count.data[world]>0;", "const bool physical=true;")
    begin = source.index("        for(int d=lane;d<29;d+=stride){\n            auto z=")
    end = source.index("        sync();good=all(good);", begin)
    source = source[:begin] + source[end:]
    source = replace(source, "out.response.data[id*29+coord]=s[192+component*32+d];", "")
    source = replace(source, "const float z=s[192+component*32+d];diag+=z*z;", "")
    source = replace(source, "out.diag.data[id]=diag;", "")
    return source


@functools.cache
def get_contact_kernel(arch):
    """Keep the original raw-striped current-contact launch and argument ABI."""

    @wp.func_native(contact_source())
    def emit(
        address: wp.uint64,
        worker: int,
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        raw: RawRowInput,
        state: RowState,
        settings: RowSettings,
        arm: ArmMap,
        out: DenseRowOutput,
        cache: contact.ContactGeometry,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_lazy_contact_J(
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        raw: RawRowInput,
        state: RowState,
        settings: RowSettings,
        arm: ArmMap,
        out: DenseRowOutput,
        cache: contact.ContactGeometry,
    ):
        worker, logical = wp.tid()
        lanes = rows_source._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = triplet._storage()
        emit(address, worker, plan, current, held, raw, state, settings, arm, out, cache)
        rows_source._release(address)

    return kinetic_lazy_contact_J


def qualifier_source(*, numerical=False):
    """Separate structural independence from the positive fallback's old proof."""
    if numerical:
        source = hybrid.qualify_source()
        return replace(
            source,
            "for (int world = worker; world < state.dense_count.shape[0]; world += workers) {",
            "for (int world = worker; world < state.dense_count.shape[0]; world += workers) {\n"
            "        if(solve.selector.data[world]<=0)continue;",
        )
    source = original.qualifier_source()
    source = replace(
        source,
        "if (tid == 0) solve.selector.data[world] = mf_count == 0 ? 0 : wp::max(mf_count, 1);",
        "if (tid == 0) { solve.selector.data[world] = mf_count == 0 ? 0 : wp::max(mf_count, 1); hybrid_data.eligible.data[world]=0; }",
    )
    source = replace(source, "const float diag = rows.diag.data[r];", "const float diag = rows.row_cfm.data[r];")
    source = replace(
        source,
        "const float j = rows.physical_J.data[r * 29 + d], y = rows.response.data[r * 29 + d];",
        "const float j = rows.physical_J.data[r * 29 + d], y = j;",
    )
    return source


@functools.cache
def get_qualify_kernel(arch, numerical=False):
    """Use the existing worker/128 scheduling for both disjoint proof visits."""
    source = qualifier_source(numerical=numerical)
    if numerical:
        source = source.replace("hybrid.", "hybrid_data.")

    @wp.func_native(source)
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
        hybrid_data: hybrid.HybridCoupledData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_lazy_qualify(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        mf: KineticMFData,
        solve: KineticSolveData,
        hybrid_data: hybrid.HybridCoupledData,
        workers: int,
    ):
        worker, logical = wp.tid()
        qualify(worker, logical, workers, plan, held, state, rows, mf, solve, hybrid_data)

    return kinetic_lazy_qualify


# The three arrays are warp-private scratch, not a capacity-sized row panel.
_ACTION = r"""
    auto action=[&](int row,bool physical){
        const int id=world*192+row;
        for(int d=lane;d<29;d+=stride)js[d]=rows.physical_J.data[id*29+d];
        sync();
        for(int d=lane;d<29;d+=stride){
            const int local=d-po;float z=0.0f;
            if(local>=0&&local<23){
                if(local<7){for(int k=0;k<=local;++k)z+=t(local,k)*js[po+k];
                    for(int k=7;k<23;++k)z+=t(local,k)*js[po+k];}
                else{const int begin=7+((local-7)/4)*4;for(int k=begin;k<=local;++k)z+=t(local,k)*js[po+k];}
            }else{const int r=d-so;for(int k=0;k<=r;++k)z+=held.inverse6.data[group*36+r*6+k]*js[so+k];}
            zs[d]=z;
        }
        sync();
        if(lane==0){float diag=rows.row_cfm.data[id];for(int d=0;d<29;++d)diag+=zs[d]*zs[d];ds[row]=diag;}
        for(int d=lane;d<29;d+=stride){
            float value=zs[d];
            if(physical){value=0.0f;const int local=d-po;
                if(local>=0&&local<23){
                    if(local<7){for(int k=local;k<7;++k)value+=t(k,local)*zs[po+k];}
                    else{for(int k=0;k<7;++k)value+=t(k,local)*zs[po+k];
                        const int end=7+((local-7)/4+1)*4;for(int k=local;k<end;++k)value+=t(k,local)*zs[po+k];}
                }else{const int c=d-so;for(int k=c;k<6;++k)value+=held.inverse6.data[group*36+k*6+c]*zs[so+k];}
            }
            rs[d]=value;
        }
        sync();
        bool good=isfinite(ds[row])&&ds[row]>=0.0f;
        for(int d=lane;d<29;d+=stride)good=good&&isfinite(zs[d])&&isfinite(rs[d]);
        good=all(good);
        if(!good){invalid=true;if(lane==0)rows.status.data[world]=3;return;}
        for(int d=lane;d<29;d+=stride)rows.response.data[id*29+d]=rs[d];
        if(lane==0)rows.diag.data[id]=ds[row];
        sync();
    };
"""


@functools.cache
def get_eager_kernel(arch):
    """Materialize Z/diag only after the J-only route declined a mixed world."""
    source = (
        original._T
        + r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
    __shared__ float scratch[288];
#else
    if(logical!=0)return;const int lane=0,stride=1;float scratch[288];
#endif
    auto sync=[&](){
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
    };
    auto all=[&](bool x){
#if defined(__CUDA_ARCH__)
        return __all_sync(0xffffffffu,x)!=0;
#else
        return x;
#endif
    };
    if(world<0||world>=state.dense_count.shape[0]||solve.selector.data[world]<=0)return;
    if(rows.status.data[world]!=0||rows.global_status.data[0]!=0)return;
    const int m=state.dense_count.data[world],po=state.primary_offset.data[world],so=state.secondary_offset.data[world];
    const int group=plan.secondary_group.data[world];
    if(m<0||m>192||!((po==0&&so==23)||(po==6&&so==0))){if(lane==0)rows.status.data[world]=4;return;}
    float* js=scratch,*zs=scratch+32,*rs=scratch+64,*ds=scratch+96;bool invalid=false;
"""
        + _ACTION
        + "\n    for(int row=0;row<m;++row){action(row,false);if(invalid)return;}\n"
    )

    @wp.func_native(source)
    def eager(
        world: int,
        logical: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_lazy_positive_response(
        plan: KineticPlan, held: HeldKineticOperator, state: RowState, rows: DenseRowOutput, solve: KineticSolveData
    ):
        world, logical = wp.tid()
        eager(world, logical, plan, held, state, rows, solve)

    return kinetic_lazy_positive_response


# Producer/epoch admission is the original _GUARD. Only first-use response
# state is local to this solve; current J/metadata are already validated by
# row_begin, prefix/contact emission and the unchanged row validator.
_RESET_READY = r"""
    for(int k=lane;k<6;k+=stride)ready[k]=0u;
    sync();
"""

_CACHE = (
    _ACTION
    + r"""
    auto ensure=[&](int row){
        const unsigned bit=1u<<(row&31);
        if((ready[row>>5]&bit)==0u){
            action(row,true);
            if(!invalid){if(lane==0)ready[row>>5]|=bit;sync();}
            else if(lane==0)solve.status.data[world]=3;
        }
    };
    auto denominator=[&](int row,float residual,float old,bool unilateral){
        if(!isfinite(residual)){invalid=true;if(lane==0)solve.status.data[world]=3;return 0.0f;}
        if(unilateral&&old==0.0f&&residual>=0.0f)return 0.0f;
        ensure(row);return invalid?0.0f:ds[row];
    };
    auto response=[&](int row){return lane<29?rows.response.data[(world*192+row)*29+lane]:0.0f;};
"""
)


def cuda_source():
    """Adapt only response access/entry/exit in the original repaired eight law."""
    source = original.cuda_recurrence()
    source = replace(source, "auto factor_rows=rows.response;", "auto factor_rows=rows.physical_J;")
    source = replace(source, "        s_diag[i] = world_diag.data[off + i];", "")
    source = replace(
        source,
        "    float factor_velocity=0.0f;",
        r"""
    float factor_velocity=0.0f;
    __shared__ float lazy_scratch[192];
    __shared__ unsigned lazy_ready[12];
    const int stride=32,group=secondary_group;
    float* js=lazy_scratch+world_slot*96,*zs=js+32,*rs=js+64,*ds=s_diag;
    unsigned* ready=lazy_ready+world_slot*6;bool invalid=false;
    auto sync=[&](){__syncwarp(MASK);};
    auto all=[&](bool x){return __all_sync(MASK,x)!=0;};
"""
        + _RESET_READY
        + _CACHE,
    )
    # Prefix scalar rows: dot first, denominator only when a nonzero update is possible.
    prefix_denom = "                const float denom = s_diag[i];\n                if (denom <= 0.0f) continue;\n"
    source = replace(source, prefix_denom, "")
    source = replace(
        source,
        "                const float old_impulse = s_lam[i];",
        "                const float old_impulse = s_lam[i];\n"
        "                const float denom=denominator(i,__shfl_sync(MASK,sum,0)+s_rhs[i],old_impulse,"
        "s_type[i]==0||s_type[i]==3);\n                if(denom<=0.0f)continue;",
    )
    source = replace(
        source,
        "                const float normal_denom = s_diag[normal];",
        "                const float normal_denom=denominator(normal,__shfl_sync(MASK,normal_sum,0)+s_rhs[normal],s_lam[normal],true);",
    )
    # Both tangent responses precede either sibling transaction. Their original
    # residuals, disk projections, store fences and update order remain intact.
    source = replace(
        source,
        "                const float tangent1_factor = lane < 29",
        "                ensure(tangent1);ensure(tangent2);if(invalid)return;\n"
        "                const float tangent1_factor = lane < 29",
    )
    for tangent in ("tangent1", "tangent2"):
        source = replace(
            source,
            f"                const float {tangent}_denom = s_diag[{tangent}];",
            f"                const float {tangent}_denom=denominator({tangent},__shfl_sync(MASK,{tangent}_sum,0)+s_rhs[{tangent}],s_lam[{tangent}],false);",
        )
    # General mixed one/three-row layout: same original scalar visit sequence.
    source = replace(
        source, "            const float denom = s_diag[i];\n            if (denom <= 0.0f) continue;\n", ""
    )
    source = replace(
        source,
        "            const float residual = jv + s_rhs[i];",
        "            const float residual = jv + s_rhs[i];\n"
        "            const float denom=denominator(i,residual,s_lam[i],row_type==0||row_type==3);\n"
        "            if(denom<=0.0f)continue;\n"
        "            if(row_type==2){ensure(sibling);if(invalid)return;}",
    )
    # Each physical increment uses cached R, never its residual coefficient J.
    updates = {
        "row_factor * delta": "response(i) * delta",
        "normal_factor * normal_delta": "response(normal) * normal_delta",
        "tangent2_factor * sibling_delta": "response(tangent2) * sibling_delta",
        "tangent1_factor * tangent1_delta": "response(tangent1) * tangent1_delta",
        "tangent1_factor * sibling_delta": "response(tangent1) * sibling_delta",
        "tangent2_factor * tangent2_delta": "response(tangent2) * tangent2_delta",
        "sibling_factor * sibling_delta": "response(sibling) * sibling_delta",
    }
    for old, new in updates.items():
        if old == "row_factor * delta":
            if source.count("factor_velocity += " + old) != 2:
                raise ValueError("Changed original scalar update sites")
            source = source.replace("factor_velocity += " + old, "factor_velocity += " + new)
        else:
            source = replace(source, "factor_velocity += " + old, "factor_velocity += " + new)
    begin = source.index("    float physical_output=0.0f;")
    end = source.index("    const bool finite=", begin)
    source = (
        source[:begin] + "    if(invalid)return;\n    const float physical_output=factor_velocity;\n" + source[end:]
    )
    return source


def cpu_source():
    """Keep the existing sequential CPU oracle law in physical increments."""
    source = original._CPU
    source = replace(
        source,
        "    float du[29]={},lambda[192];",
        "    const int lane=0,stride=1,group=plan.secondary_group.data[world];\n"
        "    float scratch[288];float* js=scratch,*zs=scratch+32,*rs=scratch+64,*ds=scratch+96;\n"
        "    unsigned ready[6];bool invalid=false;auto sync=[](){};auto all=[](bool x){return x;};\n"
        + _RESET_READY
        + _CACHE
        + "\n    float du[29]={},lambda[192];",
    )
    source = replace(source, "            const float denom=rows.diag.data[id];if(denom<=0.0f)continue;", "")
    source = replace(
        source,
        "residual+=rows.response.data[id*29+d]*du[d];residual+=rows.r0.data[id];",
        "residual+=rows.physical_J.data[id*29+d]*du[d];residual+=rows.r0.data[id];\n"
        "            const float denom=denominator(i,residual,lambda[i],type==0||type==3);if(denom<=0.0f)continue;\n"
        "            if(type==2){ensure(sibling);if(invalid)return;}",
    )
    begin = source.index("    float result[29];")
    end = source.index("    for(int i=0;i<m;++i)rows.impulses", begin)
    source = (
        source[:begin]
        + r"""
    if(invalid)return;
    for(int d=0;d<29;++d){const int coord=d<23?po+d:so+d-23;
        const float result=du[coord]+state.v_hat.data[plan.dof_ids.data[world*35+d]];
        if(!isfinite(result)){solve.status.data[world]=2;return;}}
    for(int d=0;d<(split_component?23:29);++d){const int coord=d<23?po+d:so+d-23;
        solve.v_out.data[plan.dof_ids.data[world*35+d]]=du[coord]+state.v_hat.data[plan.dof_ids.data[world*35+d]];}
"""
        + source[end:]
    )
    return source


@functools.cache
def get_solve_kernel(arch):
    """Run original two-world scheduling with per-invocation response readiness."""
    source = (
        original._GUARD
        + original._T
        + "\n#if defined(__CUDA_ARCH__)\n"
        + cuda_source()
        + "\n#else\n"
        + cpu_source()
        + "\n#endif\n"
    )

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
    def kinetic_lazy_physical_eight(
        plan: KineticPlan, held: HeldKineticOperator, state: RowState, rows: DenseRowOutput, solve: KineticSolveData
    ):
        block, logical = wp.tid()
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            return
        if original._is_cpu():
            if logical != 0:
                return
            for half in range(2):
                index = block * 2 + half
                if index < state.active_count[0]:
                    solve_native(state.active_worlds[index], plan, held, state, rows, solve)
            return
        index = block * 2 + logical // 32
        if index < state.active_count[0]:
            solve_native(state.active_worlds[index], plan, held, state, rows, solve)

    return kinetic_lazy_physical_eight


@functools.cache
def get_materialize_kernel(arch):
    """Complete positive response, original numerical route and bridge together."""
    eager = inspect.getclosurevars(get_eager_kernel(arch).func).nonlocals["eager"]
    qualify = inspect.getclosurevars(get_qualify_kernel(arch, True).func).nonlocals["qualify"]
    bridge = inspect.getclosurevars(compact.get_materialize_kernel(arch).func).nonlocals["materialize"]

    boundary_source = r"""
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
    """

    @wp.func_native(boundary_source)
    def boundary(): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_lazy_positive_materialize(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
        hybrid_data: hybrid.HybridCoupledData,
        mf: KineticMFData,
    ):
        index, logical = wp.tid()
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            return
        if index >= state.active_count[0]:
            return
        world = state.active_worlds[index]
        if world < 0 or world >= state.dense_count.shape[0]:
            return
        if solve.selector[world] <= 0:
            return
        eager(world, logical, plan, held, state, rows, solve)
        boundary()
        qualify(world, logical, state.dense_count.shape[0], plan, held, state, rows, mf, solve, hybrid_data)
        boundary()
        bridge(world, logical, plan, held, state, rows, solve, hybrid_data)

    return kinetic_lazy_positive_materialize


def install(binding, call):
    """Replace live closures without changing the original solver stream fork."""
    if binding.current_contact is None:
        raise ValueError("Lazy response requires the current-contact owner")
    rows, owner = call.rows, call.solve
    if not math.isfinite(rows.settings.cfm) or rows.settings.cfm <= 0.0:
        call.lazy_response = SimpleNamespace(active=False, reason="nonpositive_or_nonfinite_cfm")
        return
    arch = str(binding.device.arch)
    if rows.kernels[2] is not contact.get_contact_kernel(arch):
        raise ValueError("Lazy response requires the unchanged cached-triplet owner")
    rows.kernels[0] = get_arm_kernel(arch)
    rows.kernels[1] = get_prefix_kernel(arch)
    rows.kernels[2] = get_contact_kernel(arch)
    owner.kernels["qualify"] = get_qualify_kernel(arch)
    owner.kernels["offset_eight"] = get_solve_kernel(arch)
    owner.kernels["materialize"] = get_materialize_kernel(arch)
    owner.arguments["materialize"].append(owner.mf_descriptor)
    call.lazy_response = SimpleNamespace(
        active=True,
        materialize_kernels=(owner.kernels["materialize"],),
        materialize_arguments=(owner.arguments["materialize"],),
        materialize_dimensions=(owner.dimensions["materialize"],),
    )

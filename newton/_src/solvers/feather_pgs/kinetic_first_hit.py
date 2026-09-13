# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental current contact packets and first-demand physical responses.

MF worlds retain the original eager/factor path. MF0 packet storage reuses J;
action storage reuses Y. No state or response capacity is allocated here.
"""

import functools
import math

import warp as wp

from . import kinetic_rows as rows_native
from . import kinetic_rows_triplet as triplet
from . import kinetic_solve as original
from .kinetic_rows_types import ArmMap, DenseRowOutput, PrefixInput, RawRowInput, RowSettings, RowState
from .kinetic_solve_types import KineticSolveData
from .kinetic_types import CurrentKineticCache, HeldKineticOperator, KineticPlan

_PACKET = r"""
        if(!physical){
            if(nr!=3){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
            for(int component=0;component<3;++component){
                const auto direction=component==0?normal:(component==1?tangent0:tangent1);
                const auto a=component==0?xa:xta,b=component==0?xb:xtb;
                const int oa=la<30?0:(la==30?1:2),ob=lb<30?0:(lb==30?1:2);
                const auto ra=wp::cross(a-current.origin.data[world*3+oa],direction);
                const auto rb=-wp::cross(b-current.origin.data[world*3+ob],direction);
                const int id=world*settings.dense_capacity+slot+component;
                for(int k=lane;k<16;k+=stride){
                    float value=0.0f;
                    if(k<3)value=k==0?direction[0]:(k==1?direction[1]:direction[2]);
                    else if(k<6)value=k==3?ra[0]:(k==4?ra[1]:ra[2]);
                    else if(k<9)value=-(k==6?direction[0]:(k==7?direction[1]:direction[2]));
                    else if(k<12)value=k==9?rb[0]:(k==10?rb[1]:rb[2]);
                    else if(k==12)value=static_cast<float>(la);else if(k==13)value=static_cast<float>(lb);
                    else if(k==14)value=static_cast<float>(ma);else value=static_cast<float>(mb);
                    out.physical_J.data[id*29+k]=value;good=good&&isfinite(value);
                }
            }
            sync();good=all(good);
            if(!good){if(lane==0)wp::atomic_max(&out.status.data[world],3);continue;}
"""


def contact_source():
    """Keep original rounded geometry and all incident/material equations."""
    split = triplet._TRIPLET.index("        for(int d=lane;d<29;d+=stride){")
    begin = triplet._TRIPLET.index("        if(lane==0){\n            auto relative")
    metadata = triplet._TRIPLET[begin : triplet._TRIPLET.rfind("    }\n")]
    loop = """                for(int d=0;d<29;++d){const float z=s[192+component*32+d];diag+=z*z;
                    if(d>=23&&s[96+component*32+d]!=0.0f)wp::atomic_max(&out.secondary_nonzero.data[world],1);}"""
    metadata = original.replace_once(metadata, loop, "                /* Diagonal is unobserved until first demand. */")
    return (
        triplet._PRELUDE
        + triplet._TRIPLET[:split]
        + _PACKET
        + metadata
        + "            continue;\n        }\n"
        + triplet._TRIPLET[split:]
    )


@functools.cache
def get_contact_kernel(arch):
    """Publish current packets for MF0 and original eager triplets otherwise."""

    @wp.func_native(contact_source())
    def native(
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
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_first_hit_packets(
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        raw: RawRowInput,
        state: RowState,
        settings: RowSettings,
        arm: ArmMap,
        out: DenseRowOutput,
    ):
        worker, logical = wp.tid()
        lanes = rows_native._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = triplet._storage()
        native(address, worker, plan, current, held, raw, state, settings, arm, out)
        rows_native._release(address)

    return kinetic_first_hit_packets


def prefix_source():
    """Retain current prefix Z/diagonal and publish its scalar physical row key."""
    old = "if(state.mf_count.data[world]>0)out.physical_J.data[(world*settings.dense_capacity+slot)*29+row]=(d==col?sign:0.0f);"
    new = (
        old
        + "\n            else if(row==0){out.physical_J.data[(world*settings.dense_capacity+slot)*29]=static_cast<float>(po+col);out.physical_J.data[(world*settings.dense_capacity+slot)*29+1]=sign;}"
    )
    return original.replace_once(rows_native._PREFIX, old, new)


@functools.cache
def get_prefix_kernel(arch):
    """Bind the unchanged prefix signature without another allocation."""

    @wp.func_native(prefix_source())
    def native(
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
    def kinetic_first_hit_prefix(
        plan: KineticPlan,
        held: HeldKineticOperator,
        prefix: PrefixInput,
        state: RowState,
        settings: RowSettings,
        out: DenseRowOutput,
    ):
        index, logical = wp.tid()
        lanes = rows_native._lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = rows_native._storage()
        native(address, index, plan, held, prefix, state, settings, out)
        rows_native._release(address)

    return kinetic_first_hit_prefix


_HELPERS = r"""
    const bool lazy=state.mf_count.data[world]==0;
    if(lazy&&!triple_layout){if(lane==0)solve.status.data[world]=1;return;}
    __shared__ float motion_storage[2*192];
    float* motion=&motion_storage[world_slot*192];
    bool motion_dirty=true,action_good=true;
    auto rebuild_motion=[&](){
        if(!lazy||!motion_dirty)return;
        float value[6]={0,0,0,0,0,0};
        const int d=plan.body_local_dof.data[world*32+lane];
        const float velocity=__shfl_sync(MASK,factor_velocity,d>=0&&d<23?primary_offset+d:0);
        if(lane<30&&d>=0)for(int k=0;k<6;++k)value[k]=current.axes.data[world*23+d][k]*velocity;
        int parent=plan.body_parent.data[world*32+lane];
        for(int it=0;it<5;++it){const int source=parent>=0?parent:lane;
            for(int k=0;k<6;++k){const float v=__shfl_sync(MASK,value[k],source);if(parent>=0)value[k]+=v;}
            const int next=__shfl_sync(MASK,parent,source);parent=parent>=0?next:-1;
        }
        for(int q=0;q<6;++q){const float v=__shfl_sync(MASK,factor_velocity,secondary_offset+q);
            if(lane==30)for(int k=0;k<6;++k)value[k]+=current.free_axes.data[world*6+q][k]*v;}
        for(int k=0;k<6;++k)motion[lane*6+k]=value[k];
        __syncwarp(MASK);motion_dirty=false;
    };
    auto physical_residual=[&](int row){
        float sum=0.0f;
        const int id=off+row;
        if(row<contact_start){
            const int coord=static_cast<int>(rows.physical_J.data[id*29]);
            sum=__shfl_sync(MASK,factor_velocity,coord)*rows.physical_J.data[id*29+1];
        }else{
            rebuild_motion();
            if(lane<12){const int endpoint=lane/6,k=lane%6;
                const int b=static_cast<int>(rows.physical_J.data[id*29+12+endpoint]);
                if(b>=0&&b<32)sum=rows.physical_J.data[id*29+lane]*motion[b*6+k];}
            sum+=__shfl_down_sync(MASK,sum,16);sum+=__shfl_down_sync(MASK,sum,8);
            sum+=__shfl_down_sync(MASK,sum,4);sum+=__shfl_down_sync(MASK,sum,2);sum+=__shfl_down_sync(MASK,sum,1);
            sum=__shfl_sync(MASK,sum,0);
        }
        return sum;
    };
    auto action=[&](int row){
        const int id=off+row;
        if(!lazy)return lane<29?rows.response.data[id*29+lane]:0.0f;
        if(rows.valid.data[id]!=2){
            float j=0.0f,z=0.0f;
            if(row<contact_start){z=lane<29?rows.response.data[id*29+lane]:0.0f;}
            else{
                const int d=physical_dof;
                if(d>=0){
                    for(int endpoint=0;endpoint<2;++endpoint){
                        const int b=static_cast<int>(rows.physical_J.data[id*29+12+endpoint]);
                        const unsigned mask=static_cast<unsigned>(rows.physical_J.data[id*29+14+endpoint]);
                        const int q=d<23?d:d-23;
                        if(((d<23&&b>=0&&b<30)||(d>=23&&b==30))&&(mask&(1u<<q))){
                            const auto axis=d<23?current.axes.data[world*23+d]:current.free_axes.data[world*6+q];
                            float dot=0.0f;for(int k=0;k<6;++k)dot+=axis[k]*rows.physical_J.data[id*29+endpoint*6+k];
                            j+=dot;
                        }
                    }
                }
                for(int col=0;col<23;++col){const float jk=__shfl_sync(MASK,j,primary_offset+col);
                    if(primary_lane>=0&&primary_lane<23){
                        const bool support=primary_lane<7?(col>=7||col<=primary_lane):
                            (col>=7&&(col-7)/4==(primary_lane-7)/4&&col<=primary_lane);
                        if(support)z+=t(primary_lane,col)*jk;}}
                for(int col=0;col<6;++col){const float jk=__shfl_sync(MASK,j,secondary_offset+col);
                    if(secondary_lane>=col&&secondary_lane<6)z+=held.inverse6.data[secondary_group*36+secondary_lane*6+col]*jk;}
                float diag=rows.row_cfm.data[id];
                for(int col=0;col<29;++col){const float zk=__shfl_sync(MASK,z,col);diag+=zk*zk;}
                if(lane==0){s_diag[row]=diag;rows.diag.data[id]=diag;}
                __syncwarp(MASK);
            }
            float y=0.0f;
            for(int k=0;k<23;++k){const float zk=__shfl_sync(MASK,z,primary_offset+k);
                if(primary_lane>=0&&primary_lane<23){
                    const bool support=k<7?(primary_lane>=7||primary_lane<=k):
                        (primary_lane>=7&&(primary_lane-7)/4==(k-7)/4&&primary_lane<=k);
                    if(support)y+=t(k,primary_lane)*zk;}}
            for(int k=0;k<6;++k){const float zk=__shfl_sync(MASK,z,secondary_offset+k);
                if(secondary_lane>=0&&secondary_lane<=k&&secondary_lane<6)y+=held.inverse6.data[secondary_group*36+k*6+secondary_lane]*zk;}
            const bool finite=__all_sync(MASK,isfinite(y)&&isfinite(z))&&isfinite(s_diag[row])&&s_diag[row]>=0.0f;
            if(!finite)action_good=false;
            if(lane<29)rows.response.data[id*29+lane]=y;
            __syncwarp(MASK);if(lane==0)rows.valid.data[id]=2;__syncwarp(MASK);
        }
        return lane<29?rows.response.data[id*29+lane]:0.0f;
    };
"""


def cuda_source():
    """Transform row access/action coordinates, preserving the corrected projections."""
    source = original.cuda_recurrence()
    insert = "    for (int iter = 0; iter < iterations; ++iter) {"
    source = original.replace_once(source, insert, _HELPERS + "\n" + insert)
    # Existing scalar prefix remains eager, but its Z is converted exactly once.
    old = """                const float row_factor = lane < 29
                    ? factor_rows.data[row_base + i * 29 + lane] : 0.0f;"""
    source = original.replace_once(source, old, "                const float row_factor = action(i);")
    # Retain the original factor dot for MF; lazy rows use endpoint motion.
    source = original.replace_once(
        source,
        "                const float old_impulse = s_lam[i];",
        "                if(lazy)sum=physical_residual(i);\n                const float old_impulse = s_lam[i];",
    )
    old = """                const float normal_factor = lane < 29
                    ? factor_rows.data[row_base + normal * 29 + lane] : 0.0f;"""
    source = original.replace_once(source, old, "                float normal_factor = lazy?0.0f:action(normal);")
    source = original.replace_once(
        source,
        "                const float normal_denom = s_diag[normal];",
        r"""
                if(lazy)normal_sum=physical_residual(normal);
                float normal_denom=s_diag[normal];
                if(lazy){
                    if(s_lam[normal]!=0.0f || normal_sum+s_rhs[normal]<0.0f){normal_factor=action(normal);normal_denom=s_diag[normal];}
                    else normal_denom=0.0f;
                }
""",
    )
    for name in ("tangent1", "tangent2"):
        old = f"""                const float {name}_factor = lane < 29
                    ? factor_rows.data[row_base + {name} * 29 + lane] : 0.0f;"""
        source = original.replace_once(source, old, f"                const float {name}_factor = action({name});")
        source = original.replace_once(
            source,
            f"                const float {name}_denom = s_diag[{name}];",
            f"                if(lazy){name}_sum=physical_residual({name});\n                const float {name}_denom = s_diag[{name}];",
        )
    # A scan is requested only by actual velocity-changing updates; siblings
    # and this cursor's own delta are combined before the next residual call.
    import re  # noqa: PLC0415 -- source transformation only

    source = re.sub(r"(factor_velocity \+= [^;]+;)", r"\1 motion_dirty=true;", source)
    # Replace rather than ALSO execute the old factor-coordinate dot on lazy
    # worlds. MF retains every original multiply/reduction in its original order.
    for variable in ("sum", "normal_sum", "tangent1_sum", "tangent2_sum"):
        begin_dot = source.index(f"                float {variable} = lane < 29")
        end_dot = source.index(f"                {variable} += __shfl_down_sync(MASK, {variable}, 1);", begin_dot)
        end_dot += len(f"                {variable} += __shfl_down_sync(MASK, {variable}, 1);")
        dot = source[begin_dot:end_dot]
        source = (
            source[:begin_dot]
            + f"                float {variable}=0.0f;\n                if(!lazy){{\n"
            + dot.replace(f"float {variable} =", f"{variable} =", 1)
            + "\n                }"
            + source[end_dot:]
        )
    begin = source.index("    float physical_output=0.0f;")
    end = source.index("    const bool finite=", begin)
    conversion = source[begin:end]
    source = (
        source[:begin]
        + "    float physical_output=factor_velocity;\n    if(!lazy){\n"
        + conversion.replace("    float physical_output=0.0f;", "    physical_output=0.0f;")
        + "    }\n    if(!action_good){if(lane==0)solve.status.data[world]=2;return;}\n"
        + source[end:]
    )
    return source


_CPU_LAZY = r"""
    const int m=state.dense_count.data[world],off=world*192,group=plan.secondary_group.data[world];
    const int first=rows.phase_bounds.data[world*2+1];
    if(m<0||m>192||first<0||first>m||rows.phase_bounds.data[world*2]!=first||(m-first)%3!=0){solve.status.data[world]=1;return;}
    float dv[29]={},lambda[192],motion[192]={};bool dirty=true,good=true;
    for(int i=0;i<m;++i)lambda[i]=rows.impulses.data[off+i];
    auto rebuild=[&](){
        if(!dirty)return;
        for(int b=0;b<32;++b){const int d=plan.body_local_dof.data[world*32+b],p=plan.body_parent.data[world*32+b];
            for(int k=0;k<6;++k)motion[b*6+k]=(b<30&&d>=0?current.axes.data[world*23+d][k]*dv[po+d]:0.0f)+(p>=0?motion[p*6+k]:0.0f);}
        for(int k=0;k<6;++k){float v=0.0f;for(int d=0;d<6;++d)v+=current.free_axes.data[world*6+d][k]*dv[so+d];motion[30*6+k]=v;}
        dirty=false;
    };
    auto residual=[&](int row){const int id=off+row;
        if(row<first){const int coord=static_cast<int>(rows.physical_J.data[id*29]);return dv[coord]*rows.physical_J.data[id*29+1]+rows.r0.data[id];}
        rebuild();float value=0.0f;
        for(int endpoint=0;endpoint<2;++endpoint){const int b=static_cast<int>(rows.physical_J.data[id*29+12+endpoint]);
            if(b>=0)for(int k=0;k<6;++k)value+=rows.physical_J.data[id*29+endpoint*6+k]*motion[b*6+k];}
        return value+rows.r0.data[id];
    };
    auto ensure=[&](int row){const int id=off+row;if(rows.valid.data[id]==2)return;
        float j[29]={},z[29]={};
        if(row<first){for(int d=0;d<29;++d)z[d]=rows.response.data[id*29+d];}
        else{
            for(int d=0;d<29;++d){const int q=d<23?d:d-23,coord=d<23?po+d:so+q;
                for(int endpoint=0;endpoint<2;++endpoint){const int b=static_cast<int>(rows.physical_J.data[id*29+12+endpoint]);
                    const unsigned mask=static_cast<unsigned>(rows.physical_J.data[id*29+14+endpoint]);
                    if(((d<23&&b>=0&&b<30)||(d>=23&&b==30))&&(mask&(1u<<q))){
                        const auto axis=d<23?current.axes.data[world*23+d]:current.free_axes.data[world*6+q];
                        float dot=0.0f;for(int k=0;k<6;++k)dot+=axis[k]*rows.physical_J.data[id*29+endpoint*6+k];j[coord]+=dot;}}}
            for(int d=0;d<23;++d)for(int k=0;k<23;++k)z[po+d]+=t(d,k)*j[po+k];
            for(int d=0;d<6;++d)for(int k=0;k<=d;++k)z[so+d]+=held.inverse6.data[group*36+d*6+k]*j[so+k];
            float diag=rows.row_cfm.data[id];for(int d=0;d<29;++d)diag+=z[d]*z[d];rows.diag.data[id]=diag;
        }
        for(int d=0;d<29;++d){float y=0.0f;
            if(d<23)for(int k=0;k<23;++k)y+=t(k,d)*z[po+k];
            else for(int k=d-23;k<6;++k)y+=held.inverse6.data[group*36+k*6+d-23]*z[so+k];
            rows.response.data[id*29+(d<23?po+d:so+d-23)]=y;good=good&&isfinite(y);}
        good=good&&isfinite(rows.diag.data[id])&&rows.diag.data[id]>=0.0f;rows.valid.data[id]=2;
    };
    for(int iter=0;iter<solve.iterations;++iter){bool changed=false;const int global_iter=iter+solve.iteration_offset;
        for(int row=0;row<m;++row){const int id=off+row,type=rows.row_type.data[id];
            if(type==2&&global_iter<solve.friction_start_iteration){lambda[row]=0.0f;continue;}
            int parent=-1,sibling=-1;
            if(type==2){parent=rows.row_parent.data[id];sibling=row==parent+1?parent+2:parent+1;
                if(parent<0||sibling<0||sibling>=m){solve.status.data[world]=1;return;}
                if(lambda[parent]<=0.0f&&lambda[row]==0.0f&&lambda[sibling]==0.0f)continue;}
            const float r=residual(row),old=lambda[row];
            if(row>=first&&type==0&&old==0.0f&&r>=0.0f)continue;
            ensure(row);if(type==2)ensure(sibling);
            const float denom=rows.diag.data[id];if(denom<=0.0f)continue;
            const float raw_delta=-r/denom;float next=old+solve.omega*raw_delta,delta=0.0f,other_delta=0.0f;
            if(type==0||type==3){if(next<0.0f)next=0.0f;delta=next-old;}
            else if(type==4){if(r<0.0f){delta=raw_delta;next=raw_delta;}else next=0.0f;}
            else if(type==2){const float radius=wp::max(rows.row_mu.data[id]*lambda[parent],0.0f);
                if(radius<=0.0f)next=0.0f;
                else{const float other=lambda[sibling],length=sqrtf(next*next+other*other);
                    if(length>radius){const float scale=radius/length;next*=scale;const float projected=other*scale;other_delta=projected-other;lambda[sibling]=projected;}}
                delta=next-old;
            }else delta=next-old;
            lambda[row]=next;
            if(other_delta!=0.0f){changed=true;dirty=true;for(int d=0;d<29;++d)dv[d]+=rows.response.data[(off+sibling)*29+d]*other_delta;}
            if(delta!=0.0f){changed=true;dirty=true;for(int d=0;d<29;++d)dv[d]+=rows.response.data[id*29+d]*delta;}
        }
        if(global_iter>=solve.friction_start_iteration&&!changed)break;
    }
    for(int d=0;d<29;++d){const int coord=d<23?po+d:so+d-23;good=good&&isfinite(dv[coord]+state.v_hat.data[plan.dof_ids.data[world*35+d]]);}
    if(!good){solve.status.data[world]=2;return;}
    for(int d=0;d<29;++d){const int g=plan.dof_ids.data[world*35+d];solve.v_out.data[g]=state.v_hat.data[g]+dv[d<23?po+d:so+d-23];}
    for(int row=0;row<m;++row)rows.impulses.data[off+row]=lambda[row];solve.status.data[world]=0;
"""


@functools.cache
def get_solve_kernel(arch):
    """Keep the original two-warp launch and eager MF branch in one owner."""
    current_guard = r"""
    if(state.mf_count.data[world]==0&&(current.valid.data[world]==0||current.generation.data[world]!=state.state_generation.data[world])){
#if defined(__CUDA_ARCH__)
        if((threadIdx.x&31)==0)solve.status.data[world]=1;
#else
        solve.status.data[world]=1;
#endif
        return;
    }
"""
    source = (
        original._GUARD
        + current_guard
        + original._T
        + "\n#if defined(__CUDA_ARCH__)\n"
        + cuda_source()
        + "\n#else\nif(state.mf_count.data[world]!=0){\n"
        + original._CPU
        + "\nreturn;\n}\n"
        + _CPU_LAZY
        + "\n#endif\n"
    )

    @wp.func_native(source)
    def native(
        world: int,
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
        current: CurrentKineticCache,
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_first_hit_eight(
        plan: KineticPlan,
        held: HeldKineticOperator,
        state: RowState,
        rows: DenseRowOutput,
        solve: KineticSolveData,
        current: CurrentKineticCache,
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
                    native(state.active_worlds[index], plan, held, state, rows, solve, current)
            return
        index = block * 2 + logical // 32
        if index < state.active_count[0]:
            native(state.active_worlds[index], plan, held, state, rows, solve, current)

    return kinetic_first_hit_eight


def install(call):
    """Replace both sides before launch; retain allocations, MF and force closure."""
    rows, solve = call.rows, call.solve
    settings = rows.settings
    if not (
        settings.enable_friction == 1
        and math.isinf(settings.friction_gap_threshold)
        and settings.friction_gap_threshold > 0
        and settings.friction_anchor_limit == 0
    ):
        return False
    arch = str(rows.device.arch)
    rows.kernels[1] = get_prefix_kernel(arch)
    rows.kernels[2] = get_contact_kernel(arch)
    solve.kernels["offset_eight"] = get_solve_kernel(arch)
    solve.arguments["offset_eight"] = [rows.plan, rows.held, rows.state, rows.out, solve.solve_descriptor, rows.current]
    call.first_hit = True
    return True

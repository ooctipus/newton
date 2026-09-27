# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current rows in held kinetic coordinates; allocator and MF remain external.

All arrays must be contiguous with the exact descriptor shapes. A nonzero
status rejects the world before any solver consumes its rows; global_status
rejects the entire frame. No numerical fallback is silently run here.
"""

import functools

import warp as wp

from .kinetic_rows_types import (
    ArmMap,
    DenseRowOutput,
    PrefixInput,
    RawRowInput,
    RowSettings,
    RowState,
)
from .kinetic_types import CurrentKineticCache, HeldKineticOperator, KineticPlan


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return wp::vec2i(threadIdx.x&31,32);
#else
    return wp::vec2i(0,1);
#endif
""")
def _lane() -> wp.vec2i: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[192];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(192*sizeof(float)));
#endif
""")
def _storage() -> wp.uint64: ...


@wp.func_native(r"""
#if !defined(__CUDA_ARCH__)
    free(reinterpret_cast<void*>(address));
#endif
""")
def _release(address: wp.uint64): ...


@wp.func
def _t(held: HeldKineticOperator, world: int, row: int, col: int):
    value = float(0.0)
    if row < 7:
        if col < 7:
            if col <= row:
                value = held.T[world, 40 + row * (row + 1) // 2 + col]
        else:
            value = held.T[world, 68 + ((col - 7) // 4) * 28 + row * 4 + (col - 7) % 4]
    elif col >= 7:
        finger = (row - 7) // 4
        r = (row - 7) % 4
        c = (col - 7) % 4
        if (col - 7) // 4 == finger and c <= r:
            value = held.T[world, finger * 10 + r * (r + 1) // 2 + c]
    return value


@functools.cache
def get_arm_kernel(arch):
    """Reset active used slots and build one seven-by-three current arm map."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_row_arm(
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        state: RowState,
        arm: ArmMap,
        out: DenseRowOutput,
    ):
        index, logical = wp.tid()
        lanes = _lane()
        lane, stride = lanes[0], lanes[1]
        if stride == 1 and logical != 0:
            return
        count = state.active_count[0]
        if count < 0 or count > state.active_worlds.shape[0]:
            if index == 0 and lane == 0:
                out.global_status[0] = 1
            return
        if index >= count:
            return
        world = state.active_worlds[index]
        if world < 0 or world >= out.status.shape[0]:
            wp.atomic_max(out.global_status, 0, 1)
            return
        if lane == 0:
            out.status[world] = 0
            out.secondary_nonzero[world] = 0
            out.slot_counter[world] = 0
            arm.valid[world] = 0
        rows = state.dense_count[world]
        good = rows >= 0 and rows <= out.valid.shape[1]
        good = good and state.resolved[world] == 0 and state.predictor_status[world] == 0
        good = good and current.valid[world] != 0 and held.valid[world] != 0
        good = good and current.generation[world] == state.state_generation[world]
        good = good and held.generation[world] == state.held_generation[world]
        if not good:
            if lane == 0:
                out.status[world] = 2
            return
        for row in range(lane, rows, stride):
            out.valid[world, row] = 0
            out.impulses[world, row] = 0.0
        for row in range(lane, 7, stride):
            value = wp.vec3(0.0)
            for col in range(row + 1):
                value += _t(held, world, row, col) * wp.spatial_bottom(current.axes[world, col])
            arm.C[world, row] = value
        if lane == 0:
            arm.state_generation[world] = state.state_generation[world]
            arm.held_generation[world] = state.held_generation[world]
            arm.valid[world] = 1

    return kinetic_row_arm


_COMMON = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
#else
    const int lane=0,stride=1;
#endif
    auto sync=[&](){
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
    };
    auto all=[&](bool good){
#if defined(__CUDA_ARCH__)
        return __all_sync(0xffffffffu,good)!=0;
#else
        return good;
#endif
    };
    auto t=[&](int world,int row,int col){
        if(row<7){if(col<7)return col<=row?held.T.data[world*180+40+row*(row+1)/2+col]:0.0f;
            return held.T.data[world*180+68+((col-7)/4)*28+row*4+(col-7)%4];}
        if(col<7 || (row-7)/4!=(col-7)/4 || (col-7)%4>(row-7)%4)return 0.0f;
        const int r=(row-7)%4;return held.T.data[world*180+((row-7)/4)*10+r*(r+1)/2+(col-7)%4];
    };
"""

_PREFIX = (
    _COMMON
    + r"""
    if(state.active_count.data[0]<0 || state.active_count.data[0]>state.active_worlds.shape[0] || out.global_status.data[0]!=0)return;
    if(index>=state.active_count.data[0])return;
    const int world=state.active_worlds.data[index];
    if(world<0 || world>=out.status.shape[0])return;
    if(out.status.data[world]!=0)return;
    bool good=settings.contact_w==1.0f && settings.dt>0.0f && isfinite(settings.dt)
        && isfinite(settings.cfm) && settings.cfm>=0.0f;
    for(int code=lane;code<46;code+=stride){
        const int d=code/2,g=plan.dof_ids.data[world*35+d],qi=prefix.q_index.data[g];
        int active=0;float phi=0.0f;
        if(qi>=0){const float q=prefix.q.data[qi],bound=(code&1)?prefix.upper.data[g]:prefix.lower.data[g];
            good=good&&isfinite(q);phi=(code&1)?bound-q:q-bound;
            active=isfinite(bound)&&((code&1)?q>=bound-settings.activation_gap:q<=bound+settings.activation_gap);}
        s[code]=static_cast<float>(active);s[64+code]=phi;
    }
    sync();good=all(good);
    if(!good){if(lane==0)out.status.data[world]=3;return;}
    if(lane==0){int n=0;for(int code=0;code<46;++code)if(s[code]!=0.0f)s[128+n++]=static_cast<float>(code);s[191]=static_cast<float>(n);}
    sync();const int n=static_cast<int>(s[191]);
    if(n>state.dense_count.data[world] || n>settings.dense_capacity){if(lane==0)out.status.data[world]=4;return;}
    const int po=state.primary_offset.data[world],so=state.secondary_offset.data[world];
    if(!((po==0&&so==23)||(po==6&&so==0))){if(lane==0)out.status.data[world]=5;return;}
    for(int slot=0;slot<n;++slot){
        const int code=static_cast<int>(s[128+slot]),col=code/2,g=plan.dof_ids.data[world*35+col];
        const float sign=(code&1)?-1.0f:1.0f,phi=s[64+code];
        const float bias=(phi<0.0f?settings.bias_scale*settings.beta:settings.joint_limit_speculative_scale)*phi/settings.dt;
        for(int row=lane;row<29;row+=stride){const int d=row-po;float z=0.0f;
            if(d>=0&&d<23)z=sign*t(world,d,col);
            out.response.data[(world*settings.dense_capacity+slot)*29+row]=z;s[row]=z*z;
            if(state.mf_count.data[world]>0)out.physical_J.data[(world*settings.dense_capacity+slot)*29+row]=(d==col?sign:0.0f);
        }
        sync();
        if(lane==0){float diag=settings.cfm;for(int k=0;k<29;++k)diag+=s[k];const int id=world*settings.dense_capacity+slot;
            out.r0.data[id]=sign*state.v_hat.data[g]+bias;out.rhs.data[id]=bias;out.diag.data[id]=diag;
            out.row_type.data[id]=3;out.row_parent.data[id]=-1;out.row_mu.data[id]=0.0f;out.row_beta.data[id]=settings.beta;
            out.row_cfm.data[id]=settings.cfm;out.phi.data[id]=phi;out.target_velocity.data[id]=0.0f;
            out.row_restitution.data[id]=0.0f;if(out.row_w.shape[0]>1)out.row_w.data[id]=1.0f;
            out.valid.data[id]=1;
            if(!isfinite(diag)||!isfinite(out.r0.data[id]))out.status.data[world]=3;
        }
        sync();
    }
    if(lane==0){out.slot_counter.data[world]=n;out.phase_bounds.data[world*2]=n;out.phase_bounds.data[world*2+1]=n;}
"""
)


@wp.func_native(_PREFIX)
def _prefix(
    address: wp.uint64,
    index: int,
    plan: KineticPlan,
    held: HeldKineticOperator,
    prefix: PrefixInput,
    state: RowState,
    settings: RowSettings,
    out: DenseRowOutput,
): ...


@functools.cache
def get_prefix_kernel(arch):
    """Emit original ordered scalar-limit prefix directly in kinetic coordinates."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_row_prefix(
        plan: KineticPlan,
        held: HeldKineticOperator,
        prefix: PrefixInput,
        state: RowState,
        settings: RowSettings,
        out: DenseRowOutput,
    ):
        index, logical = wp.tid()
        lanes = _lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = _storage()
        _prefix(address, index, plan, held, prefix, state, settings, out)
        _release(address)

    return kinetic_row_prefix


_CONTACT = (
    _COMMON
    + r"""
    const int count=raw.count.data[0];
    if(count<0 || count>settings.raw_capacity || count>raw.shape0.shape[0] || settings.workers<=0){
        if(worker==0&&lane==0)out.global_status.data[0]=1;return;}
    if(state.raw_invalid.data[0]!=0){if(worker==0&&lane==0)out.global_status.data[0]=1;return;}
    for(int c=worker;c<count;c+=settings.workers){
        if(raw.path.data[c]!=0 || raw.slot.data[c]<0)continue;
        const int world=raw.world.data[c],slot=raw.slot.data[c];
        if(world<0 || world>=out.status.shape[0]){if(lane==0)wp::atomic_max(&out.global_status.data[0],1);continue;}
        if(!all(state.resolved.data[world]==0 && out.status.data[world]==0))continue;
        const int aa=raw.art_a.data[c],ab=raw.art_b.data[c];
        if(aa>=raw.response_count.shape[0] || ab>=raw.response_count.shape[0]){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
        if((aa<0 || (raw.response_count.data[aa]!=23&&raw.response_count.data[aa]!=6)) &&
           (ab<0 || (raw.response_count.data[ab]!=23&&raw.response_count.data[ab]!=6)))continue;
        if(arm.valid.data[world]==0 || arm.state_generation.data[world]!=state.state_generation.data[world] ||
           arm.held_generation.data[world]!=state.held_generation.data[world]){if(lane==0)wp::atomic_max(&out.status.data[world],2);continue;}
        const int sa=raw.shape0.data[c],sb=raw.shape1.data[c];
        if(sa>=raw.shape_body.shape[0]||sb>=raw.shape_body.shape[0]){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
        const int ba=sa>=0?raw.shape_body.data[sa]:-1,bb=sb>=0?raw.shape_body.data[sb]:-1;
        if(ba>=raw.body_q.shape[0]||bb>=raw.body_q.shape[0]){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
        const int la=ba>=0?plan.body_lane.data[ba]:-1,lb=bb>=0?plan.body_lane.data[bb]:-1;
        bool good=(ba<0||(la>=0&&la<32&&plan.body_ids.data[world*32+la]==ba)) &&
                  (bb<0||(lb>=0&&lb<32&&plan.body_ids.data[world*32+lb]==bb));
        good=good&&(ba<0?aa<0:aa==raw.body_to_articulation.data[ba])&&(bb<0?ab<0:ab==raw.body_to_articulation.data[bb]);
        good=good&&(aa<0||raw.art_to_world.data[aa]==world)&&(ab<0||raw.art_to_world.data[ab]==world);
        if(!good){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
        if(lane==0){
            const auto n=-raw.normal.data[c];
            const auto xa=(ba>=0?wp::transform_point(raw.body_q.data[ba],raw.point0.data[c]):raw.point0.data[c])-n*raw.margin0.data[c];
            const auto xb=(bb>=0?wp::transform_point(raw.body_q.data[bb],raw.point1.data[c]):raw.point1.data[c])+n*raw.margin1.data[c];
            auto t0=wp::cross(n,wp::vec3(1.0f,0.0f,0.0f));if(wp::dot(t0,t0)<1.0e-12f)t0=wp::cross(n,wp::vec3(0.0f,1.0f,0.0f));
            t0=wp::normalize(t0);const auto t1=wp::normalize(wp::cross(n,t0));
            for(int k=0;k<3;++k){s[64+k]=n[k];s[67+k]=xa[k];s[70+k]=xb[k];s[73+k]=t0[k];s[76+k]=t1[k];}
            const float phi=wp::dot(n,xa-xb);float mu=0.0f,e=0.0f;int nm=0;
            for(int endpoint=0;endpoint<2;++endpoint){const int shape=endpoint==0?sa:sb;if(shape>=0){mu+=raw.shape_mu.data[shape];const float ev=raw.shape_restitution.data[shape];if(isfinite(ev))e+=wp::clamp(ev,0.0f,1.0f);++nm;}}
            if(nm){mu/=static_cast<float>(nm);e/=static_cast<float>(nm);}
            const bool filter=settings.friction_articulation_pairs_only==0 ||
                (aa>=0&&ab>=0&&raw.is_free_rigid.data[aa]==0&&raw.is_free_rigid.data[ab]==0);
            const int limit=filter?settings.friction_anchor_limit:0;int rank=0,next=0;
            if(limit>0){for(int look=1;look<=8;++look){const int prev=c-look;if(prev<0)break;if(raw.shape0.data[prev]!=sa||raw.shape1.data[prev]!=sb)break;++rank;}
                if(c+1<count&&raw.shape0.data[c+1]==sa&&raw.shape1.data[c+1]==sb)next=1;}
            bool friction=settings.enable_friction!=0&&(!filter||phi<=settings.friction_gap_threshold);
            if(limit>0&&rank>=limit)friction=false;
            s[79]=phi;s[80]=mu;s[81]=e;s[82]=mu*settings.friction_scale*((limit>0&&(rank>0||next))?0.5f:1.0f);s[83]=friction?3.0f:1.0f;
        }
        sync();const int nr=static_cast<int>(s[83]);
        if(slot<out.slot_counter.data[world]||slot+nr>state.dense_count.data[world]||slot+nr>settings.dense_capacity||nr!=raw.slots_needed.data[c]){
            if(lane==0)wp::atomic_max(&out.status.data[world],4);continue;}
        const unsigned ma=ba>=0?raw.body_response_mask.data[ba]:0u,mb=bb>=0?raw.body_response_mask.data[bb]:0u;
        const bool primary_a=la>=0&&la<30&&aa>=0&&raw.response_count.data[aa]==23;
        const bool primary_b=lb>=0&&lb<30&&ab>=0&&raw.response_count.data[ab]==23;
        const bool common=primary_a&&primary_b&&aa==ab&&(ma&mb&127u)==127u;
        const bool physical=state.mf_count.data[world]>0;
        const int po=state.primary_offset.data[world],so=state.secondary_offset.data[world];
        if(!((po==0&&so==23)||(po==6&&so==0))){if(lane==0)wp::atomic_max(&out.status.data[world],5);continue;}
        for(int component=0;component<nr;++component){
            const auto direction=wp::vec3(s[component==0?64:(component==1?73:76)],s[(component==0?64:(component==1?73:76))+1],s[(component==0?64:(component==1?73:76))+2]);
            auto xa=wp::vec3(s[67],s[68],s[69]),xb=wp::vec3(s[70],s[71],s[72]);
            if(settings.shared_anchor!=0||(component!=0&&settings.friction_shared_anchor!=0)){const auto mid=(xa+xb)*0.5f;xa=mid;xb=mid;}
            const auto moment=wp::cross(xa-xb,direction);
            for(int d=lane;d<29;d+=stride){float j=0.0f;
                if(d<23){
                    const auto axis=current.axes.data[world*23+d];const auto lin=wp::vec3(axis[0],axis[1],axis[2]),ang=wp::vec3(axis[3],axis[4],axis[5]);
                    if(common&&d<7){if(physical)j=wp::dot(ang,moment);}
                    else {if(primary_a&&(ma&(1u<<d)))j+=wp::dot(direction,lin+wp::cross(ang,xa-current.origin.data[world*3]));
                          if(primary_b&&(mb&(1u<<d)))j-=wp::dot(direction,lin+wp::cross(ang,xb-current.origin.data[world*3]));}
                }else{const int k=d-23;const auto axis=current.free_axes.data[world*6+k];const auto lin=wp::vec3(axis[0],axis[1],axis[2]),ang=wp::vec3(axis[3],axis[4],axis[5]);
                    if(la==30&&(ma&(1u<<k)))j+=wp::dot(direction,lin+wp::cross(ang,xa-current.origin.data[world*3+1]));
                    if(lb==30&&(mb&(1u<<k)))j-=wp::dot(direction,lin+wp::cross(ang,xb-current.origin.data[world*3+1]));}
                s[d]=j;good=good&&isfinite(j);
            }
            sync();
            for(int d=lane;d<29;d+=stride){float z=0.0f;
                if(d<7){if(common)z=wp::dot(arm.C.data[world*7+d],moment);else for(int k=0;k<=d;++k)z+=t(world,d,k)*s[k];
                    for(int k=7;k<23;++k)if((ma|mb)&(1u<<k))z+=t(world,d,k)*s[k];}
                else if(d<23){const int begin=7+((d-7)/4)*4;for(int k=begin;k<=d;++k)z+=t(world,d,k)*s[k];}
                else{const int row=d-23,group=plan.secondary_group.data[world];for(int k=0;k<=row;++k)z+=held.inverse6.data[group*36+row*6+k]*s[23+k];}
                s[32+d]=z;good=good&&isfinite(z);
            }
            sync();good=all(good);
            if(!good){if(lane==0)wp::atomic_max(&out.status.data[world],3);break;}
            const int id=world*settings.dense_capacity+slot+component;
            for(int d=lane;d<29;d+=stride){const int coord=d<23?po+d:so+d-23;
                out.response.data[id*29+coord]=s[32+d];if(physical)out.physical_J.data[id*29+coord]=s[d];}
            if(lane==0){float relative=0.0f,known=0.0f,diag=settings.cfm;
                for(int endpoint=0;endpoint<2;++endpoint){const int body=endpoint==0?ba:bb,art=endpoint==0?aa:ab,local=endpoint==0?la:lb;
                    if(body>=0){const auto velocity=state.endpoint_twists.data[body];const auto lin=wp::vec3(velocity[0],velocity[1],velocity[2]),ang=wp::vec3(velocity[3],velocity[4],velocity[5]);
                        const auto anchor=endpoint==0?xa:xb;const int oi=local<30?0:(local==30?1:2);
                        const float value=(endpoint==0?1.0f:-1.0f)*wp::dot(direction,lin+wp::cross(ang,anchor-current.origin.data[world*3+oi]));
                        relative+=value;if(art>=0&&raw.prescribed_articulation.data[art]!=0)known+=value;}}
                for(int d=0;d<29;++d){diag+=s[32+d]*s[32+d];if(d>=23&&s[d]!=0.0f)wp::atomic_max(&out.secondary_nonzero.data[world],1);}
                const float phi=component==0?s[79]:0.0f,e=component==0?s[81]:0.0f;
                float bias=component==0?(phi<=0.0f?settings.bias_scale*settings.beta:settings.contact_speculative_scale)*phi/settings.dt:0.0f;
                if(e>0.0f&&relative < -settings.restitution_velocity_threshold && (phi<=1.0e-6f||phi+settings.dt*relative<=1.0e-6f))bias=e*relative;
                out.r0.data[id]=relative+bias;out.rhs.data[id]=known+bias;out.diag.data[id]=diag;
                out.row_type.data[id]=component==0?0:2;out.row_parent.data[id]=component==0?-1:slot;
                out.row_mu.data[id]=component==0?s[80]:s[82];out.row_beta.data[id]=component==0?settings.beta:0.0f;
                out.row_cfm.data[id]=settings.cfm;out.phi.data[id]=phi;out.target_velocity.data[id]=-known;out.row_restitution.data[id]=e;
                if(out.row_w.shape[0]>1)out.row_w.data[id]=1.0f;
                if(!isfinite(relative)||!isfinite(bias)||!isfinite(diag)||!isfinite(out.row_mu.data[id]))wp::atomic_max(&out.status.data[world],3);
                out.valid.data[id]=1;
            }
            sync();
        }
    }
"""
)


@wp.func_native(_CONTACT)
def _contact(
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


@functools.cache
def get_contact_kernel(arch):
    """Bounded-grid raw-order traversal; one warp emits a contact's actual rows."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_contact_rows(
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
        lanes = _lane()
        if lanes[1] == 1 and logical != 0:
            return
        address = _storage()
        _contact(address, worker, plan, current, held, raw, state, settings, arm, out)
        _release(address)

    return kinetic_contact_rows


@functools.cache
def get_validate_kernel(arch):
    """Reject malformed capacity/counts or any missing consumed current row."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_rows_validate(state: RowState, settings: RowSettings, out: DenseRowOutput):
        index, logical = wp.tid()
        lanes = _lane()
        if lanes[1] == 1 and logical != 0:
            return
        if index == 0:
            if state.raw_invalid[0] != 0:
                wp.atomic_max(out.global_status, 0, 1)
            for i in range(lanes[0], state.capacity_status.shape[0], lanes[1]):
                if state.capacity_status[i] != 0:
                    wp.atomic_max(out.global_status, 0, 1)
        if state.active_count[0] < 0 or state.active_count[0] > state.active_worlds.shape[0]:
            if index == 0 and lanes[0] == 0:
                out.global_status[0] = 1
            return
        if index >= state.active_count[0]:
            return
        world = state.active_worlds[index]
        if world < 0 or world >= out.status.shape[0]:
            wp.atomic_max(out.global_status, 0, 1)
            return
        count = state.dense_count[world]
        if (
            count < 0
            or count > settings.dense_capacity
            or state.mf_count[world] < 0
            or state.mf_count[world] > settings.mf_capacity
        ):
            wp.atomic_max(out.status, world, 4)
            return
        for row in range(lanes[0], count, lanes[1]):
            if out.valid[world, row] != 1:
                wp.atomic_max(out.status, world, 6)

    return kinetic_rows_validate

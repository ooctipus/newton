# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Preserve the checked original host/source-construction expressions.
# ruff: noqa: RUF005

"""Current Kuka forcing and held kinetic action; no rows or state integration."""

import functools

import numpy as np
import warp as wp

from .kinetic_types import (
    CurrentForceInput,
    CurrentKineticCache,
    GeometricCache,
    HeldKineticOperator,
    KineticPlan,
    KineticPredictorOutput,
    RefreshInput,
)


def unpack_augmented(packed):
    """Decode unrounded held block coefficients in original physical DOF order."""
    packed = np.asarray(packed)
    h = np.zeros(packed.shape[:-1] + (23, 23), dtype=packed.dtype)
    for finger in range(4):
        off = 7 + 4 * finger
        for i in range(4):
            for j in range(i + 1):
                h[..., off + i, off + j] = h[..., off + j, off + i] = packed[..., 10 * finger + i * (i + 1) // 2 + j]
        b = packed[..., 68 + 28 * finger : 96 + 28 * finger].reshape(packed.shape[:-1] + (7, 4))
        h[..., :7, off : off + 4] = b
        h[..., off : off + 4, :7] = b.swapaxes(-1, -2)
    for i in range(7):
        for j in range(i + 1):
            h[..., i, j] = h[..., j, i] = packed[..., 40 + i * (i + 1) // 2 + j]
    return h


def pack_augmented(h):
    """Pack a structurally qualified block matrix; do not infer physical zeros."""
    h = np.asarray(h)
    if h.shape[-2:] != (23, 23) or not np.all(np.isfinite(h)):
        raise ValueError("Expected finite held23 matrix")
    packed = np.empty(h.shape[:-2] + (180,), dtype=h.dtype)
    for finger in range(4):
        off = 7 + 4 * finger
        for i in range(4):
            for j in range(i + 1):
                packed[..., 10 * finger + i * (i + 1) // 2 + j] = h[..., off + i, off + j]
        packed[..., 68 + 28 * finger : 96 + 28 * finger] = h[..., :7, off : off + 4].reshape(h.shape[:-2] + (28,))
    for i in range(7):
        for j in range(i + 1):
            packed[..., 40 + i * (i + 1) // 2 + j] = h[..., i, j]
    if np.max(np.abs(unpack_augmented(packed) - h)) > 1e-8:
        raise ValueError("Unsupported cross-finger or nonsymmetric held operator")
    return packed


def unpack_T(packed):
    """Decode inverse-whitener blocks without introducing cross-finger fill."""
    packed = np.asarray(packed)
    t = np.zeros(packed.shape[:-1] + (23, 23), dtype=packed.dtype)
    for finger in range(4):
        off = 7 + 4 * finger
        for i in range(4):
            for j in range(i + 1):
                t[..., off + i, off + j] = packed[..., 10 * finger + i * (i + 1) // 2 + j]
        t[..., :7, off : off + 4] = packed[..., 68 + 28 * finger : 96 + 28 * finger].reshape(packed.shape[:-1] + (7, 4))
    for i in range(7):
        for j in range(i + 1):
            t[..., i, j] = packed[..., 40 + i * (i + 1) // 2 + j]
    return t


def factor_from_augmented(packed):
    """Create a CPU fixture inverse action; native refresh is separately charged."""
    h = unpack_augmented(np.asarray(packed, dtype=np.float64))
    out = np.empty(h.shape[:-2] + (180,), dtype=np.float64)
    schur = h[..., :7, :7].copy()
    blocks = []
    for finger in range(4):
        off = 7 + 4 * finger
        ti = np.linalg.inv(np.linalg.cholesky(h[..., off : off + 4, off : off + 4]))
        e = h[..., :7, off : off + 4] @ ti.swapaxes(-1, -2)
        schur -= e @ e.swapaxes(-1, -2)
        blocks.append((ti, e))
        for i in range(4):
            for j in range(i + 1):
                out[..., finger * 10 + i * (i + 1) // 2 + j] = ti[..., i, j]
    ta = np.linalg.inv(np.linalg.cholesky(schur))
    for i in range(7):
        for j in range(i + 1):
            out[..., 40 + i * (i + 1) // 2 + j] = ta[..., i, j]
    for finger, (ti, e) in enumerate(blocks):
        out[..., 68 + 28 * finger : 96 + 28 * finger] = (-ta @ e @ ti).reshape(out.shape[:-1] + (28,))
    return out.astype(np.float32)


def operator_from_held_lower(lower):
    """Seed only from captured HELD L, never current or next-state geometry."""
    lower = np.asarray(lower, dtype=np.float64)
    packed = pack_augmented(lower @ lower.swapaxes(-1, -2))
    return factor_from_augmented(packed), packed.astype(np.float32)


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[544];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(544*sizeof(float)));
#endif
""")
def _storage() -> wp.uint64: ...


@wp.func_native(r"""
#if !defined(__CUDA_ARCH__)
    free(reinterpret_cast<void*>(address));
#endif
""")
def _release(address: wp.uint64): ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return wp::vec2i(threadIdx.x & 31, 32);
#else
    return wp::vec2i(0, 1);
#endif
""")
def _lane() -> wp.vec2i: ...


_PREDICTOR = r"""
    float* s = reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31, stride=32;
#else
    const int lane=0, stride=1;
#endif
    if (current.valid.data[world]==0 || current.generation.data[world]!=inputs.state_generation.data[world]) {
        if(lane==0) output.status.data[world]=1;
        return;
    }
    if (held.valid.data[world]==0 || held.generation.data[world]!=inputs.expected_held_generation.data[world]) {
        if(lane==0) output.status.data[world]=2;
        return;
    }
    bool good=isfinite(inputs.dt)&&inputs.dt>0.0f, external=false;
    for(int b=lane;b<32;b+=stride) {
        const int body=plan.body_ids.data[world*32+b];
        float v[6]={0,0,0,0,0,0};
        if(b<31 && (inputs.body_flags.data[body]&2)==0) {
            const auto f=inputs.body_f.data[body];
            const auto r=current.com_offset.data[world*32+b];
            for(int k=0;k<6;++k) {good=good&&isfinite(f[k]);external=external||f[k]!=0.0f;}
            for(int k=0;k<3;++k) good=good&&isfinite(r[k]);
            bool nonzero=false;for(int k=0;k<6;++k)nonzero=nonzero||f[k]!=0.0f;
            if(nonzero){
                const wp::vec3 force(f[0],f[1],f[2]);
                const wp::vec3 moment=wp::vec3(f[3],f[4],f[5])+wp::cross(r,force);
                for(int k=0;k<3;++k){v[k]=force[k];v[k+3]=moment[k];}
            }
        }
        for(int k=0;k<6;++k)s[b*6+k]=v[k];
    }
    for(int i=lane;i<35;i+=stride) {
        const int d=plan.dof_ids.data[world*35+i];
        good=good&&isfinite(inputs.joint_qd.data[d])&&isfinite(inputs.predictor_qd.data[d])&&isfinite(inputs.joint_f.data[d]);
    }
#if defined(__CUDA_ARCH__)
    external=__any_sync(0xffffffffu,external);
    good=__all_sync(0xffffffffu,good);
    __syncwarp(0xffffffffu);
#endif
    if(!good){if(lane==0)output.status.data[world]=3;return;}
    if(lane==0)output.external_nonzero.data[world]=external?1:0;
    // Separate finger suffixes, then palm and trunk. No prefix subtraction.
    if(external) {
#if defined(__CUDA_ARCH__)
        float value[6];for(int k=0;k<6;++k)value[k]=s[lane*6+k];
        const int end=lane>=10&&lane<30?10+5*((lane-10)/5+1):0;
        for(int offset=1;offset<=4;offset*=2) {
            for(int k=0;k<6;++k){const float other=__shfl_down_sync(0xffffffffu,value[k],offset);if(lane>=10&&lane<30&&lane+offset<end)value[k]+=other;}
        }
        for(int k=0;k<6;++k) {
            const float a=__shfl_sync(0xffffffffu,value[k],10),b=__shfl_sync(0xffffffffu,value[k],15);
            const float c=__shfl_sync(0xffffffffu,value[k],20),d=__shfl_sync(0xffffffffu,value[k],25);
            if(lane==9)value[k]+=((a+b)+c)+d;
        }
        for(int offset=1;offset<=8;offset*=2) {
            for(int k=0;k<6;++k){const float other=__shfl_down_sync(0xffffffffu,value[k],offset);if(lane<10&&lane+offset<10)value[k]+=other;}
        }
        for(int k=0;k<6;++k)s[lane*6+k]=value[k];
#else
        for(int b=29;b>=0;--b){const int p=plan.body_parent.data[world*32+b];if(p>=0)for(int k=0;k<6;++k)s[p*6+k]+=s[b*6+k];}
#endif
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    // Scalar primary force/control law; one body owns each physical DOF.
    for(int b=lane;b<30;b+=stride) {
        const int i=plan.body_local_dof.data[world*32+b];
        if(i<0)continue;
        const int d=plan.dof_ids.data[world*35+i];
        const auto axis=current.axes.data[world*23+i];
        float ext=0.0f;
        for(int k=0;k<6;++k){good=good&&isfinite(axis[k]);if(external)ext+=axis[k]*s[b*6+k];}
        const float q=inputs.joint_q.data[plan.q_index.data[world*23+i]],qd=inputs.joint_qd.data[d];
        const float passive=inputs.joint_spring_stiffness.data[d]*(inputs.joint_spring_ref.data[d]-q)-inputs.joint_damping.data[d]*qd;
        float u0=0.0f;
        if(inputs.drive_row_by_dof.data[d]>=0) {
            const float ke=inputs.joint_target_ke.data[d],kd=inputs.joint_target_kd.data[d];
            const float K=ke*inputs.dt*inputs.dt+kd*inputs.dt;
            good=good&&isfinite(K);
            if(K>0.0f) {
                const float dq=inputs.joint_q.data[inputs.drive_q_index_by_dof.data[d]];
                u0=-(ke*(dq-inputs.joint_target_pos.data[d]+inputs.dt*qd)+kd*(qd-inputs.joint_target_vel.data[d]));
                const float cap=inputs.joint_effort_limit.data[d];
                if(cap>0.0f)u0=wp::clamp(u0,-cap,cap);
            }
        }
        const float tau=-current.bias.data[world*23+i]+ext+inputs.joint_f.data[d]+passive+u0;
        s[192+i]=tau;
        good=good&&isfinite(q)&&isfinite(tau);
    }
    for(int i=lane;i<6;i+=stride) {
        const auto axis=current.free_axes.data[world*6+i],bias=current.free_bias.data[world];
        float value=0.0f;
        for(int k=0;k<6;++k){good=good&&isfinite(axis[k])&&isfinite(bias[k]);value-=axis[k]*(bias[k]-s[30*6+k]);}
        value+=inputs.joint_f.data[plan.dof_ids.data[world*35+23+i]];
        s[192+23+i]=value;good=good&&isfinite(value);
    }
#if defined(__CUDA_ARCH__)
    good=__all_sync(0xffffffffu,good);
    __syncwarp(0xffffffffu);
#endif
    if(!good){if(lane==0)output.status.data[world]=3;return;}
    // Sparse T*tau. Four fingers are independent; all enter the arm action.
    const float* t=held.T.data+world*180;
    const int sg=plan.secondary_group.data[world];
    const float* w6=held.inverse6.data+sg*36;
    bool good6=true;
    for(int i=lane;i<29;i+=stride) {
        float value=0.0f;
        if(i<7) {
            for(int k=0;k<=i;++k){const float a=t[40+i*(i+1)/2+k];good=good&&isfinite(a);value+=a*s[192+k];}
            for(int f=0;f<4;++f)for(int k=0;k<4;++k){const float a=t[68+f*28+i*4+k];good=good&&isfinite(a);value+=a*s[192+7+f*4+k];}
        } else if(i<23) {
            const int f=(i-7)/4,row=(i-7)%4;
            for(int k=0;k<=row;++k){const float a=t[f*10+row*(row+1)/2+k];good=good&&isfinite(a);value+=a*s[192+7+f*4+k];}
        } else {
            const int row=i-23;
            for(int k=0;k<=row;++k){const float a=w6[row*6+k];good6=good6&&isfinite(a);value+=a*s[192+23+k];}
        }
        s[227+i]=value;if(i<23)good=good&&isfinite(value);else good6=good6&&isfinite(value);
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int i=lane;i<29;i+=stride) {
        float value=0.0f;
        if(i<7)for(int k=i;k<7;++k)value+=t[40+k*(k+1)/2+i]*s[227+k];
        else if(i<23) {
            const int f=(i-7)/4,col=(i-7)%4;
            for(int k=col;k<4;++k)value+=t[f*10+k*(k+1)/2+col]*s[227+7+f*4+k];
            for(int k=0;k<7;++k)value+=t[68+f*28+k*4+col]*s[227+k];
        } else for(int k=i-23;k<6;++k)value+=w6[k*6+i-23]*s[227+23+k];
        s[262+i]=value;if(i<23)good=good&&isfinite(value);else good6=good6&&isfinite(value);
    }
#if defined(__CUDA_ARCH__)
    good=__all_sync(0xffffffffu,good);good6=__all_sync(0xffffffffu,good6);
    __syncwarp(0xffffffffu);
#endif
    if(!good6&&lane==0) {
        const float* l=held.lower6.data+sg*36;float x[6];
        for(int i=0;i<6;++i){float v=s[192+23+i];for(int k=0;k<i;++k)v-=l[i*6+k]*x[k];x[i]=l[i*6+i]!=0.0f?v/l[i*6+i]:0.0f;}
        for(int i=5;i>=0;--i){float v=x[i];for(int k=i+1;k<6;++k)v-=l[k*6+i]*x[k];x[i]=l[i*6+i]!=0.0f?v/l[i*6+i]:0.0f;s[262+23+i]=x[i];}
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int i=lane;i<35;i+=stride) {
        const int d=plan.dof_ids.data[world*35+i];
        float a=i<29?s[262+i]:0.0f;
        if(inputs.kinematic_dof.data[d]!=0)a=0.0f;
        s[262+i]=a;s[297+i]=inputs.predictor_qd.data[d]+inputs.dt*a;
        good=good&&isfinite(a)&&isfinite(s[297+i]);
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int root=lane;root<2;root+=stride) {
        const int joint=inputs.free_root_joints.data[plan.root_slots.data[world*2+root]];
        if(inputs.kinematic_joint.data[joint]!=0)continue;
        const int d=inputs.joint_qd_start.data[joint],off=root==0?23:29;
        const wp::vec3 v(inputs.predictor_qd.data[d],inputs.predictor_qd.data[d+1],inputs.predictor_qd.data[d+2]);
        const wp::vec3 w(inputs.predictor_qd.data[d+3],inputs.predictor_qd.data[d+4],inputs.predictor_qd.data[d+5]);
        const auto c=wp::cross(w,v);for(int k=0;k<3;++k)s[297+off+k]+=c[k]*inputs.dt;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
    float value[6]={0,0,0,0,0,0};
    const int dof=plan.body_local_dof.data[world*32+lane];
    if(lane<30&&dof>=0)for(int k=0;k<6;++k)value[k]=current.axes.data[world*23+dof][k]*s[297+dof];
    int parent=plan.body_parent.data[world*32+lane];
    for(int it=0;it<5;++it) {
        const int source=parent>=0?parent:lane;
        for(int k=0;k<6;++k){const float a=__shfl_sync(0xffffffffu,value[k],source);if(parent>=0)value[k]+=a;}
        const int next=__shfl_sync(0xffffffffu,parent,source);parent=parent>=0?next:-1;
    }
    for(int k=0;k<6;++k)s[332+lane*6+k]=value[k];
#else
    for(int b=0;b<32;++b){const int d=plan.body_local_dof.data[world*32+b],p=plan.body_parent.data[world*32+b];for(int k=0;k<6;++k)s[332+b*6+k]=(b<30&&d>=0?current.axes.data[world*23+d][k]*s[297+d]:0.0f)+(p>=0?s[332+p*6+k]:0.0f);}
#endif
    for(int b=lane;b<32;b+=stride) {
        if(b==30)for(int k=0;k<6;++k){float v=0.0f;for(int d=0;d<6;++d)v+=current.free_axes.data[world*6+d][k]*s[297+23+d];s[332+b*6+k]=v;}
        if(b==31){const auto v=inputs.prescribed_body_v_s.data[plan.body_ids.data[world*32+b]];for(int k=0;k<6;++k)s[332+b*6+k]=v[k];}
        for(int k=0;k<6;++k)good=good&&isfinite(s[332+b*6+k]);
    }
    for(int i=lane;i<35;i+=stride)good=good&&isfinite(s[297+i]);
#if defined(__CUDA_ARCH__)
    good=__all_sync(0xffffffffu,good);__syncwarp(0xffffffffu);
#endif
    if(!good){if(lane==0)output.status.data[world]=4;return;}
    // All checks precede numeric publication. No q/qd or public pose is advanced.
    for(int i=lane;i<35;i+=stride){const int d=plan.dof_ids.data[world*35+i];output.joint_qdd.data[d]=s[262+i];output.v_hat.data[d]=s[297+i];}
    for(int b=lane;b<32;b+=stride){wp::spatial_vector v;for(int k=0;k<6;++k)v[k]=s[332+b*6+k];output.endpoint_twists.data[plan.body_ids.data[world*32+b]]=v;}
    if(lane==0)output.status.data[world]=0;
"""


@wp.func_native(_PREDICTOR)
def _predictor(
    address: wp.uint64,
    world: int,
    plan: KineticPlan,
    current: CurrentKineticCache,
    held: HeldKineticOperator,
    inputs: CurrentForceInput,
    output: KineticPredictorOutput,
): ...


@functools.cache
def get_predictor_kernel(arch):
    """Return a fixed one-warp world owner with CPU single-owner parity."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_current_predictor(
        plan: KineticPlan,
        current: CurrentKineticCache,
        held: HeldKineticOperator,
        inputs: CurrentForceInput,
        output: KineticPredictorOutput,
    ):
        world, logical_lane = wp.tid()
        lanes = _lane()
        if lanes[1] == 1 and logical_lane != 0:
            return
        if world >= output.status.shape[0]:
            return
        address = _storage()
        _predictor(address, world, plan, current, held, inputs, output)
        _release(address)

    return kinetic_current_predictor


_REFRESH = r"""
    float* h=reinterpret_cast<float*>(address);float* l=h+180;float* t=h+360;
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
#else
    const int lane=0,stride=1;
#endif
    if(inputs.requested.data[world]==0){if(lane==0)inputs.status.data[world]=0;return;}
    if(lane==0)held.valid.data[world]=0;
    if(geometric.valid.data[world]==0||geometric.generation.data[world]!=inputs.state_generation.data[world]){
        if(lane==0)inputs.status.data[world]=5;return;
    }
    bool good=isfinite(inputs.dt)&&inputs.dt>0.0f;
    for(int i=lane;i<180;i+=stride){h[i]=geometric.geometric.data[world*180+i];l[i]=0.0f;t[i]=0.0f;good=good&&isfinite(h[i]);}
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int i=lane;i<23;i+=stride){
        const int d=plan.dof_ids.data[world*35+i];
        float r=inputs.R.data[d];
        if(inputs.drive_row_by_dof.data[d]>=0){const float k=inputs.joint_target_ke.data[d]*inputs.dt*inputs.dt+inputs.joint_target_kd.data[d]*inputs.dt;good=good&&isfinite(k);if(k>0.0f)r+=k;}
        const int f=(i-7)/4,row=(i-7)%4;
        const int index=i<7?40+i*(i+1)/2+i:f*10+row*(row+1)/2+row;
        h[index]+=r;good=good&&isfinite(h[index]);
    }
#if defined(__CUDA_ARCH__)
    good=__all_sync(0xffffffffu,good);__syncwarp(0xffffffffu);
#endif
    if(!good){if(lane==0)inputs.status.data[world]=6;return;}
    // Four independent4x4 Choleskys, column phases with row ownership.
    for(int col=0;col<4;++col){
        for(int i=lane;i<16;i+=stride){const int f=i/4,row=i%4;if(row==col){float v=h[f*10+row*(row+1)/2+col];for(int k=0;k<col;++k)v-=l[f*10+row*(row+1)/2+k]*l[f*10+row*(row+1)/2+k];good=good&&isfinite(v)&&v>0.0f;l[f*10+row*(row+1)/2+col]=v>0.0f?sqrtf(v):0.0f;}}
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
        for(int i=lane;i<16;i+=stride){const int f=i/4,row=i%4;if(row>col){float v=h[f*10+row*(row+1)/2+col];for(int k=0;k<col;++k)v-=l[f*10+row*(row+1)/2+k]*l[f*10+col*(col+1)/2+k];const float diag=l[f*10+col*(col+1)/2+col];l[f*10+row*(row+1)/2+col]=diag>0.0f?v/diag:0.0f;}}
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
    }
    for(int job=lane;job<28;job+=stride){
        const int f=job/7,row=job%7;
        for(int col=0;col<4;++col){float v=h[68+f*28+row*4+col];for(int k=0;k<col;++k)v-=l[68+f*28+row*4+k]*l[f*10+col*(col+1)/2+k];const float diag=l[f*10+col*(col+1)/2+col];l[68+f*28+row*4+col]=diag>0.0f?v/diag:0.0f;}
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int row=lane;row<7;row+=stride)for(int col=0;col<=row;++col){float v=h[40+row*(row+1)/2+col];for(int f=0;f<4;++f)for(int k=0;k<4;++k)v-=l[68+f*28+row*4+k]*l[68+f*28+col*4+k];t[40+row*(row+1)/2+col]=v;}
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int col=0;col<7;++col){
        if(lane==0){float v=t[40+col*(col+1)/2+col];for(int k=0;k<col;++k)v-=l[40+col*(col+1)/2+k]*l[40+col*(col+1)/2+k];good=good&&isfinite(v)&&v>0.0f;l[40+col*(col+1)/2+col]=v>0.0f?sqrtf(v):0.0f;}
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
        for(int row=lane;row<7;row+=stride)if(row>col){float v=t[40+row*(row+1)/2+col];for(int k=0;k<col;++k)v-=l[40+row*(row+1)/2+k]*l[40+col*(col+1)/2+k];const float diag=l[40+col*(col+1)/2+col];l[40+row*(row+1)/2+col]=diag>0.0f?v/diag:0.0f;}
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
    }
#if defined(__CUDA_ARCH__)
    good=__all_sync(0xffffffffu,good);
#endif
    if(!good){if(lane==0)inputs.status.data[world]=6;return;}
    // Each inverse column is independent; no full23 factor or inverse is built.
    for(int job=lane;job<16;job+=stride){const int f=job/4,col=job%4;for(int row=col;row<4;++row){float v=row==col?1.0f:0.0f;for(int k=col;k<row;++k)v-=l[f*10+row*(row+1)/2+k]*t[f*10+k*(k+1)/2+col];t[f*10+row*(row+1)/2+col]=v/l[f*10+row*(row+1)/2+row];}}
    for(int col=lane;col<7;col+=stride)for(int row=col;row<7;++row){float v=row==col?1.0f:0.0f;for(int k=col;k<row;++k)v-=l[40+row*(row+1)/2+k]*t[40+k*(k+1)/2+col];t[40+row*(row+1)/2+col]=v/l[40+row*(row+1)/2+row];}
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int job=lane;job<112;job+=stride){const int f=job/28,row=(job%28)/4,col=job%4;float v=0.0f;for(int k=col;k<4;++k)v+=l[68+f*28+row*4+k]*t[f*10+k*(k+1)/2+col];t[68+job]=v;}
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int job=lane;job<112;job+=stride){const int f=job/28,row=(job%28)/4,col=job%4;float v=0.0f;for(int k=0;k<=row;++k)v-=t[40+row*(row+1)/2+k]*t[68+f*28+k*4+col];l[68+job]=v;}
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int job=lane;job<112;job+=stride)t[68+job]=l[68+job];
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for(int i=lane;i<180;i+=stride)good=good&&isfinite(t[i]);
#if defined(__CUDA_ARCH__)
    good=__all_sync(0xffffffffu,good);__syncwarp(0xffffffffu);
#endif
    if(!good){if(lane==0)inputs.status.data[world]=6;return;}
    for(int i=lane;i<180;i+=stride){held.augmented.data[world*180+i]=h[i];held.T.data[world*180+i]=t[i];}
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    if(lane==0){held.generation.data[world]=inputs.held_generation.data[world];held.valid.data[world]=1;inputs.status.data[world]=0;}
"""


@wp.func_native(_REFRESH)
def _refresh(
    address: wp.uint64,
    world: int,
    plan: KineticPlan,
    geometric: GeometricCache,
    held: HeldKineticOperator,
    inputs: RefreshInput,
): ...


@functools.cache
def get_refresh_kernel(arch):
    """Refresh primary180 from current geometric180 and actual requested R/K."""

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_held_refresh(
        plan: KineticPlan,
        geometric: GeometricCache,
        held: HeldKineticOperator,
        inputs: RefreshInput,
    ):
        world, logical_lane = wp.tid()
        lanes = _lane()
        if lanes[1] == 1 and logical_lane != 0:
            return
        if world >= inputs.status.shape[0]:
            return
        address = _storage()
        _refresh(address, world, plan, geometric, held, inputs)
        _release(address)

    return kinetic_held_refresh

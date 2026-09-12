# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental light predictor/ZERO ownership for the existing 23+6 recipe.

The inverse inputs are held lower factors W=L^-1, never physical H^-1.
This module deliberately contains no constraint-row or GS shared panel.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap
from dataclasses import dataclass

import numpy as np
import warp as wp

from . import kernels, simple_world


@wp.struct
class JointWorldPlan:
    body_ids: wp.array2d[int]
    body_parent: wp.array2d[int]
    body_dof: wp.array2d[int]
    body_lane: wp.array[int]
    joint_ids: wp.array2d[int]
    dof_ids: wp.array2d[int]
    root_slots: wp.array2d[int]
    primary_group: wp.array[int]
    secondary_group: wp.array[int]


@wp.struct
class PredictorData:
    lower23: wp.array3d[float]
    lower6: wp.array3d[float]
    inverse23: wp.array3d[float]
    inverse6: wp.array3d[float]
    tau: wp.array[float]
    qd: wp.array[float]
    qdd: wp.array[float]
    kinematic_dof: wp.array[int]
    kinematic_joint: wp.array[int]
    free_root_joints: wp.array[int]
    joint_qd_start: wp.array[int]


@wp.struct
class LightOutput:
    resolved: wp.array[int]
    active_worlds: wp.array[int]
    active_count: wp.array[int]
    endpoint_twists: wp.array[wp.spatial_vector]
    predictor_fallback: wp.array[int]
    v_out: wp.array[float]


@wp.struct
class IntegrationData:
    joint_type: wp.array[int]
    joint_parent: wp.array[int]
    joint_child: wp.array[int]
    joint_q_start: wp.array[int]
    joint_qd_start: wp.array[int]
    kinematic_joint_mask: wp.array[int]
    joint_dof_dim: wp.array2d[int]
    body_com: wp.array[wp.vec3]
    joint_X_c: wp.array[wp.transform]
    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    joint_q_new: wp.array[float]
    joint_qd_new: wp.array[float]
    angular_damping: float


@dataclass
class HostPlan:
    """Static, checked physical ownership; arrays retain actual source indices."""

    body_ids: np.ndarray
    body_parent: np.ndarray
    body_dof: np.ndarray
    body_lane: np.ndarray
    joint_ids: np.ndarray
    dof_ids: np.ndarray
    root_slots: np.ndarray
    primary_group: np.ndarray
    secondary_group: np.ndarray

    def device_data(self, device):
        """Allocate immutable map storage once, outside graph capture."""
        result = JointWorldPlan()
        for name in self.__dataclass_fields__:
            setattr(result, name, wp.array(getattr(self, name), dtype=int, device=device))
        return result


def build_plan(
    *,
    world_count,
    art_world,
    art_start,
    art_end,
    dof_start,
    dof_count,
    response_count,
    primary_arts,
    secondary_arts,
    prescribed,
    joint_parent,
    joint_child,
    joint_qd_start,
    joint_dof_dim,
    body_art,
    body_response_mask,
    free_root_joints,
):
    """Require exact world35/body32 closure and parent-mask identities before dispatch."""
    worlds, bodies = int(world_count), len(body_art)
    fields = {
        "body_ids": np.full((worlds, 32), -1, np.int32),
        "body_parent": np.full((worlds, 32), -1, np.int32),
        "body_dof": np.full((worlds, 32), -1, np.int32),
        "body_lane": np.full(bodies, -1, np.int32),
        "joint_ids": np.full((worlds, 32), -1, np.int32),
        "dof_ids": np.full((worlds, 35), -1, np.int32),
        "root_slots": np.full((worlds, 2), -1, np.int32),
        "primary_group": np.full(worlds, -1, np.int32),
        "secondary_group": np.full(worlds, -1, np.int32),
    }
    result = HostPlan(**fields)
    primary_group = {int(art): i for i, art in enumerate(primary_arts)}
    secondary_group = {int(art): i for i, art in enumerate(secondary_arts)}
    roots = {int(joint): i for i, joint in enumerate(free_root_joints)}
    body_counts = np.bincount(np.asarray(body_art)[np.asarray(body_art) >= 0], minlength=len(art_world))
    by_world = [[] for _ in range(worlds)]
    for art, world in enumerate(art_world):
        if not 0 <= world < worlds:
            raise ValueError("Joint-world ownership requires all articulations inside current worlds")
        by_world[int(world)].append(art)
    for world, arts in enumerate(by_world):
        primary = [art for art in arts if art in primary_group]
        secondary = [art for art in arts if art in secondary_group]
        prescribed_arts = [art for art in arts if prescribed[art] != 0]
        if len(arts) != 3 or len(primary) != 1 or len(secondary) != 1 or len(prescribed_arts) != 1:
            raise ValueError("Joint-world ownership requires primary23/free6/prescribed6")
        ordered = [primary[0], secondary[0], prescribed_arts[0]]
        if len(set(ordered)) != 3 or [int(dof_count[a]) for a in ordered] != [23, 6, 6]:
            raise ValueError("Joint-world ownership requires 35 physical DOFs")
        if [int(response_count[a]) for a in ordered] != [23, 6, 0]:
            raise ValueError("Joint-world response must remain 23+6 with a prescribed root")
        result.primary_group[world] = primary_group[ordered[0]]
        result.secondary_group[world] = secondary_group[ordered[1]]
        joints_by_art = [list(range(int(art_start[a]), int(art_end[a]))) for a in ordered]
        if [len(joints) for joints in joints_by_art] != [30, 1, 1]:
            raise ValueError("Joint-world body ownership requires a 30-link primary and two roots")
        joints = np.asarray([joint for family in joints_by_art for joint in family], dtype=np.int32)
        children = np.asarray(joint_child)[joints]
        if np.any(children < 0) or np.any(children >= bodies) or len(set(children)) != 32:
            raise ValueError("Joint-world joints must publish 32 distinct bodies")
        expected_arts = np.repeat(ordered, [30, 1, 1])
        if not np.array_equal(np.asarray(body_art)[children], expected_arts):
            raise ValueError("Joint-world joint/body ownership mismatch")
        if int(np.sum(body_counts[ordered])) != 32:
            raise ValueError("Joint-world plan cannot omit an articulation body")
        if np.any(result.body_lane[children] >= 0):
            raise ValueError("Joint-world body ownership overlaps")
        result.joint_ids[world], result.body_ids[world] = joints, children
        result.body_lane[children] = np.arange(32, dtype=np.int32)
        dofs = np.concatenate([np.arange(dof_start[a], dof_start[a] + dof_count[a]) for a in ordered])
        result.dof_ids[world] = dofs
        local_body = {int(body): i for i, body in enumerate(children)}
        masks = np.zeros(30, dtype=np.uint32)
        seen_dofs = []
        for lane, joint in enumerate(joints[:30]):
            parent = int(joint_parent[joint])
            parent_lane = local_body.get(parent, -1)
            if parent >= 0 and (parent_lane < 0 or parent_lane >= lane):
                raise ValueError("Joint-world parent order must be local and parent-first")
            result.body_parent[world, lane] = parent_lane
            mask = int(masks[parent_lane]) if parent_lane >= 0 else 0
            dimension = int(np.sum(joint_dof_dim[joint]))
            if dimension not in (0, 1):
                raise ValueError("Joint-world primary joints must be fixed or scalar")
            if dimension:
                dof = int(joint_qd_start[joint])
                local = dof - int(dof_start[ordered[0]])
                if not 0 <= local < 23:
                    raise ValueError("Joint-world primary DOF is outside held factor")
                mask |= 1 << local
                result.body_dof[world, lane] = dof
                seen_dofs.append(local)
            masks[lane] = mask
            if mask != int(body_response_mask[children[lane]]):
                raise ValueError("Joint-world ancestor sum differs from original response mask")
        if sorted(seen_dofs) != list(range(23)):
            raise ValueError("Joint-world primary must cover every held DOF exactly once")
        for slot, joint in enumerate(joints[30:]):
            if int(joint_parent[joint]) >= 0 or int(joint) not in roots:
                raise ValueError("Joint-world secondary and prescribed owners must be free roots")
            result.root_slots[world, slot] = roots[int(joint)]
        if int(body_response_mask[children[30]]) != 63:
            raise ValueError("Joint-world free response requires all six current axes")
    if np.any(result.body_lane < 0) or len(np.unique(result.dof_ids)) != worlds * 35:
        raise ValueError("Joint-world plan must cover disjoint complete current physical owners")
    return result


def bind_plan(solver):
    """Bind actual constructor metadata once; response29 is not global35."""
    model = solver.model
    starts = solver.articulation_dof_start.numpy()
    return build_plan(
        world_count=solver.world_count,
        art_world=solver.art_to_world.numpy(),
        art_start=model.articulation_start.numpy(),
        art_end=solver.articulation_joint_end.numpy(),
        dof_start=starts,
        dof_count=np.diff(np.r_[starts, model.joint_dof_count]),
        response_count=solver.articulation_response_dof_count.numpy(),
        primary_arts=solver.group_to_art[23].numpy(),
        secondary_arts=solver.group_to_art[6].numpy(),
        prescribed=solver._prescribed_articulation.numpy(),
        joint_parent=model.joint_parent.numpy(),
        joint_child=model.joint_child.numpy(),
        joint_qd_start=model.joint_qd_start.numpy(),
        joint_dof_dim=model.joint_dof_dim.numpy(),
        body_art=solver.body_to_articulation.numpy(),
        body_response_mask=solver.body_response_dof_mask.numpy(),
        free_root_joints=solver._free_root_joint_indices.numpy(),
    )


def predictor_reference(inverse, tau):
    """Emulate the two FP32 row-dot phases without an extra inverse producer."""
    n = inverse.shape[-1]
    first, result = np.zeros_like(tau), np.zeros_like(tau)
    for i in range(n):
        for k in range(i + 1):
            first[..., i] = np.float32(first[..., i] + inverse[..., i, k] * tau[..., k])
    for i in range(n):
        for k in range(i, n):
            result[..., i] = np.float32(result[..., i] + inverse[..., k, i] * first[..., k])
    return result


def prefix_reference(plan, axes, velocity):
    """Emulate the five simultaneous parent-pointer rounds for primary bodies."""
    worlds = plan.body_ids.shape[0]
    values = np.zeros((worlds, 32, 6), np.float32)
    active = plan.body_dof >= 0
    values[active] = axes[plan.body_dof[active]] * velocity[plan.body_dof[active], None]
    parent = plan.body_parent.copy()
    world_ids = np.arange(worlds)[:, None]
    for _ in range(5):
        valid = parent >= 0
        previous = values.copy()
        values = np.float32(previous + np.where(valid[..., None], previous[world_ids, np.maximum(parent, 0)], 0.0))
        parent = np.where(valid, parent[world_ids, np.maximum(parent, 0)], -1)
    return values


_STORAGE = r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[544];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(544 * sizeof(float)));
#endif
"""


@wp.func_native(_STORAGE)
def _storage() -> wp.uint64: ...


@wp.func_native(r"""
#if !defined(__CUDA_ARCH__)
    free(reinterpret_cast<void*>(address));
#endif
""")
def _release(address: wp.uint64): ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
""")
def _sync(): ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return wp::vec2i(threadIdx.x & 31, 32);
#else
    return wp::vec2i(0, 1);
#endif
""")
def _lane() -> wp.vec2i: ...


_PREDICT = r"""
    float* s = reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31, stride = 32;
#else
    const int lane = 0, stride = 1;
#endif
    const int pg = plan.primary_group.data[world], sg = plan.secondary_group.data[world];
    bool good = invalid == 0;
    // Stage 1: W times tau. Row ownership is physical23 followed by free6.
    for (int i = lane; i < 29; i += stride) {
        const int n = i < 23 ? 23 : 6, off = i < 23 ? 0 : 23, row = i - off;
        const float* W = i < 23 ? pred.inverse23.data + pg*529 : pred.inverse6.data + sg*36;
        float value = 0.0f;
        for (int k = 0; k <= row; ++k) {
            const float w = W[row*n+k], t = pred.tau.data[plan.dof_ids.data[world*35+off+k]];
            good = good && isfinite(w) && isfinite(t);
            value += w*t;
        }
        s[i] = value;
        good = good && isfinite(value);
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    // Stage 2: W transpose times the complete first vector.
    for (int i = lane; i < 29; i += stride) {
        const int n = i < 23 ? 23 : 6, off = i < 23 ? 0 : 23, row = i - off;
        const float* W = i < 23 ? pred.inverse23.data + pg*529 : pred.inverse6.data + sg*36;
        float value = 0.0f;
        for (int k = row; k < n; ++k) value += W[k*n+row]*s[off+k];
        s[35+i] = value;
        good = good && isfinite(value);
    }
#if defined(__CUDA_ARCH__)
    good = __all_sync(0xffffffffu, good);
    __syncwarp(0xffffffffu);
#endif
    if (lane == 0) s[513] = good ? 0.0f : 1.0f;
    // Only unsupported/nonfinite inverse action pays the old triangular path.
    if (!good && lane == 0) {
        float value[23];
        for (int block = 0; block < 2; ++block) {
            const int n = block == 0 ? 23 : 6, off = block == 0 ? 0 : 23;
            const float* L = block == 0 ? pred.lower23.data + pg*529 : pred.lower6.data + sg*36;
            for (int i = 0; i < n; ++i) {
                float rhs = pred.tau.data[plan.dof_ids.data[world*35+off+i]];
                for (int k = 0; k < i; ++k) rhs -= L[i*n+k]*value[k];
                value[i] = L[i*n+i] != 0.0f ? rhs/L[i*n+i] : 0.0f;
            }
            for (int i = n-1; i >= 0; --i) {
                float rhs = value[i];
                for (int k = i+1; k < n; ++k) rhs -= L[k*n+i]*value[k];
                value[i] = L[i*n+i] != 0.0f ? rhs/L[i*n+i] : 0.0f;
            }
            for (int i = 0; i < n; ++i) s[35+off+i] = value[i];
        }
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for (int i = lane; i < 35; i += stride) {
        const int d = plan.dof_ids.data[world*35+i];
        float a = i < 29 ? s[35+i] : 0.0f;
        if (pred.kinematic_dof.data[d] != 0) a = 0.0f;
        pred.qdd.data[d] = a;
        data.v_hat.data[d] = pred.qd.data[d] + data.dt*a;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    for (int root = lane; root < 2; root += stride) {
        const int joint = pred.free_root_joints.data[plan.root_slots.data[world*2+root]];
        if (pred.kinematic_joint.data[joint] != 0) continue;
        const int d = pred.joint_qd_start.data[joint];
        const wp::vec3 v(pred.qd.data[d], pred.qd.data[d+1], pred.qd.data[d+2]);
        const wp::vec3 w(pred.qd.data[d+3], pred.qd.data[d+4], pred.qd.data[d+5]);
        const wp::vec3 c = wp::cross(w,v);
        for (int k=0;k<3;++k) data.v_hat.data[d+k] += c[k]*data.dt;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    if (lane == 0) output.predictor_fallback.data[world] = s[513] != 0.0f;
"""


@wp.func_native(_PREDICT)
def _predict(
    address: wp.uint64,
    world: int,
    plan: JointWorldPlan,
    pred: PredictorData,
    data: simple_world._SimpleWorldInput,
    output: LightOutput,
    invalid: int,
): ...


_TWISTS = r"""
    float* s = reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31, stride = 32;
    float value[6] = {0,0,0,0,0,0};
    const int dof = plan.body_dof.data[world*32+lane];
    if (dof >= 0) for (int k=0;k<6;++k) value[k]=data.joint_S_s.data[dof][k]*data.v_hat.data[dof];
    int parent = plan.body_parent.data[world*32+lane];
    for (int iteration=0;iteration<5;++iteration) {
        const int source=parent>=0?parent:lane;
        float ancestor[6];
        for (int k=0;k<6;++k) ancestor[k]=__shfl_sync(0xffffffffu,value[k],source);
        const int next=__shfl_sync(0xffffffffu,parent,source);
        if (parent>=0) for (int k=0;k<6;++k) value[k]+=ancestor[k];
        parent=parent>=0?next:-1;
    }
    for(int k=0;k<6;++k) s[96+lane*6+k]=value[k];
#else
    const int lane=0,stride=1;
    float values[32][6] = {}, previous[32][6];
    int parents[32],next[32];
    for(int b=0;b<32;++b) {
        parents[b]=plan.body_parent.data[world*32+b];
        const int dof=plan.body_dof.data[world*32+b];
        if(dof>=0) for(int k=0;k<6;++k) values[b][k]=data.joint_S_s.data[dof][k]*data.v_hat.data[dof];
    }
    for(int it=0;it<5;++it) {
        for(int b=0;b<32;++b) {next[b]=parents[b]>=0?parents[parents[b]]:-1;for(int k=0;k<6;++k) previous[b][k]=values[b][k];}
        for(int b=0;b<32;++b) {if(parents[b]>=0) for(int k=0;k<6;++k) values[b][k]+=previous[parents[b]][k];parents[b]=next[b];}
    }
    for(int b=0;b<32;++b) for(int k=0;k<6;++k) s[96+b*6+k]=values[b][k];
#endif
    for(int b=lane;b<32;b+=stride) {
        const int body=plan.body_ids.data[world*32+b], art=data.body_to_articulation.data[body];
        const auto pose=data.body_q.data[body];
        const auto p=wp::transform_get_translation(pose);
        const auto q=wp::transform_get_rotation(pose);
        bool valid=true; float norm=0.0f;
        for(int k=0;k<3;++k) valid=valid&&isfinite(p[k])&&isfinite(data.articulation_origin.data[art][k]);
        for(int k=0;k<4;++k) {valid=valid&&isfinite(q[k]);norm+=q[k]*q[k];}
        valid=valid&&isfinite(norm)&&norm>0.0f;
        const bool responding=data.body_has_response_dofs.data[body]!=0&&(data.body_flags.data[body]&2)==0;
        if(data.prescribed_articulation.data[art]!=0) {
            for(int k=0;k<6;++k) s[96+b*6+k]=data.body_v_s.data[body][k];
        } else if(responding) {
            if(b>=30) {
                const int first=data.articulation_dof_start.data[art];
                for(int k=0;k<6;++k) {
                    float value=0.0f;
                    for(int d=0;d<6;++d) value+=data.joint_S_s.data[first+d][k]*data.v_hat.data[first+d];
                    s[96+b*6+k]=value;
                }
            }
            if(data.is_free_rigid.data[art]!=0) {
                const int first=data.articulation_dof_start.data[art];
                for(int k=0;k<6;++k) {
                    const float limit=k<3?data.max_linear_velocity.data[body]:data.max_angular_velocity.data[body];
                    valid=valid&&!isnan(limit);
                    if(isfinite(limit)&&limit>0.0f) valid=valid&&fabsf(data.v_hat.data[first+k])<=limit;
                }
            }
        } else {
            for(int k=0;k<6;++k) s[96+b*6+k]=0.0f;
            if(data.articulation_response_dof_count.data[art]>0&&(data.body_flags.data[body]&2)!=0) valid=false;
        }
        for(int k=0;k<6;++k) valid=valid&&isfinite(s[96+b*6+k]);
        s[288+b]=valid?1.0f:0.0f;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
"""


@wp.func_native(_TWISTS)
def _twists(address: wp.uint64, world: int, plan: JointWorldPlan, data: simple_world._SimpleWorldInput): ...


_LIMITS = r"""
    float* s = reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
#else
    const int lane=0,stride=1;
#endif
    bool valid=s[513]==0.0f&&isfinite(data.dt)&&data.dt>0.0f;
    valid=valid&&isfinite(data.beta)&&data.beta>=0.0f;
    valid=valid&&isfinite(data.speculative_scale)&&data.speculative_scale>=0.0f;
    valid=valid&&isfinite(data.activation_gap)&&data.activation_gap>=0.0f;
    valid=valid&&isfinite(data.restitution_threshold)&&data.restitution_threshold>=0.0f;
    valid=valid&&isfinite(data.absolute_margin)&&data.absolute_margin>=0.0f;
    valid=valid&&isfinite(data.relative_margin)&&data.relative_margin>=0.0f;
    for(int b=lane;b<32;b+=stride) valid=valid&&s[288+b]!=0.0f;
    const int count=data.world_dof_count.data[world];
    valid=valid&&count>=0&&count<=data.world_dof_indices.shape[1];
    if(valid) for(int local=lane;local<count;local+=stride) {
        const int d=data.world_dof_indices.data[world*data.world_dof_indices.shape[1]+local];
        if(d<0||d>=data.v_hat.shape[0]||d>=data.limit_q_index.shape[0]) {valid=false;continue;}
        const float velocity=data.v_hat.data[d];
        if(!isfinite(velocity)) {valid=false;continue;}
        const int qi=data.limit_q_index.data[d];
        if(qi<0) continue;
        if(qi>=data.q.shape[0]||d>=data.lower.shape[0]||d>=data.upper.shape[0]) {valid=false;continue;}
        const float q=data.q.data[qi],lower=data.lower.data[d],upper=data.upper.data[d];
        if(!isfinite(q)||isnan(lower)||isnan(upper)||lower>upper) {valid=false;continue;}
        for(int side=0;side<2;++side) {
            const float bound=side==0?lower:upper;
            const bool active=isfinite(bound)&&(side==0?q<=lower+data.activation_gap:q>=upper-data.activation_gap);
            if(active) {
                const float phi=side==0?q-lower:upper-q, velocity_signed=side==0?velocity:-velocity;
                const float bias=phi<0.0f?data.beta*phi/data.dt:phi/data.dt;
                const float residual=velocity_signed+bias;
                const float margin=data.absolute_margin+data.relative_margin*(fabsf(velocity_signed)+fabsf(bias));
                valid=valid&&isfinite(residual)&&isfinite(margin)&&residual>=margin;
            }
        }
    }
#if defined(__CUDA_ARCH__)
    valid=__all_sync(0xffffffffu,valid);
#endif
    if(lane==0) s[512]=valid?1.0f:0.0f;
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
"""


@wp.func_native(_LIMITS)
def _limits(address: wp.uint64, world: int, data: simple_world._SimpleWorldInput): ...


@wp.func_native("return reinterpret_cast<float*>(address)[512] != 0.0f;")
def _ready(address: wp.uint64) -> bool: ...


@wp.func_native("return reinterpret_cast<float*>(address)[288+lane] != 0.0f;")
def _body_valid(address: wp.uint64, lane: int) -> bool: ...


@wp.func_native(
    "float* p=reinterpret_cast<float*>(address)+96+lane*6; wp::spatial_vector result; for(int k=0;k<6;++k) result[k]=p[k]; return result;"
)
def _body_twist(address: wp.uint64, lane: int) -> wp.spatial_vector: ...


def normal_source():
    """Reuse the original geometric law, replacing only scratch ownership and result publication."""
    endpoint = textwrap.dedent(inspect.getsource(simple_world._endpoint_speed.func))
    endpoint = endpoint.replace("_endpoint_speed(", "_light_endpoint_speed(")
    endpoint = endpoint.replace("scratch: _SimpleWorldScratch,", "address: wp.uint64, plan: JointWorldPlan,")
    endpoint = endpoint.replace("scratch.body_twist[body]", "_body_twist(address, plan.body_lane[body])")
    normal = textwrap.dedent(inspect.getsource(simple_world._check_normal.func))
    normal = normal.replace("_check_normal(", "_light_check_normal(")
    normal = normal.replace("scratch: _SimpleWorldScratch", "address: wp.uint64, plan: JointWorldPlan")
    normal = normal.replace("scratch.resolved.shape[0]", "plan.body_ids.shape[0]")
    normal = normal.replace("scratch.body_valid[body_a] != 0", "_body_valid(address, plan.body_lane[body_a])")
    normal = normal.replace("scratch.body_valid[body_b] != 0", "_body_valid(address, plan.body_lane[body_b])")
    normal = normal.replace("_endpoint_speed(", "_light_endpoint_speed(").replace(
        ", data, scratch)", ", data, address, plan)"
    )
    tree = ast.parse(normal)

    class Results(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr in ("atomic_max", "atomic_add", "atomic_min"):
                    return None
            return self.generic_visit(node)

        def visit_If(self, node):
            text = ast.unparse(node.test)
            if text == "scratch.limit_ok[world] == 0":
                return None
            if ast.dump(node.test) == ast.dump(ast.parse("not responds_a and not responds_b", mode="eval").body):
                node.body = [ast.Return(value=ast.Constant(value=True))]
                return node
            return self.generic_visit(node)

        def visit_Return(self, node):
            if node.value is None:
                node.value = ast.Constant(value=False)
            return node

    tree = Results().visit(tree)
    function = tree.body[0]
    # The final diagnostic rejection had no return. Its body is now empty.
    final = function.body[-1]
    if not isinstance(final, ast.If) or ast.unparse(final.test) != "not valid" or final.body:
        raise RuntimeError("Original normal publication changed; review the private selector")
    function.body[-1] = ast.Return(value=ast.Name(id="valid", ctx=ast.Load()))
    ast.fix_missing_locations(tree)
    return endpoint + "\n" + ast.unparse(tree) + "\n"


def operation_source(name):
    """Keep the complete original generalized operation with an explicit physical index."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(kernels, name).func)))
    function = tree.body[0]
    function.name = "_light_" + name
    function.decorator_list = ast.parse("@wp.func\ndef placeholder():\n    pass").body[0].decorator_list
    function.args.args.insert(0, ast.arg(arg="physical_index", annotation=ast.Name(id="int", ctx=ast.Load())))

    class Index(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node.func) == "wp.tid":
                return ast.copy_location(ast.Name(id="physical_index", ctx=ast.Load()), node)
            return self.generic_visit(node)

    tree = Index().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


_FINISH = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
    good=__all_sync(0xffffffffu,good);
#else
    const int lane=0,stride=1;
#endif
    const bool selected=good&&s[512]!=0.0f&&invalid==0;
    if(lane==0) {
        output.resolved.data[world]=selected?1:0;
        if(!selected) {
#if defined(__CUDA_ARCH__)
            const int index=atomicAdd(output.active_count.data,1);
#else
            const int index=output.active_count.data[0]++;
#endif
            output.active_worlds.data[index]=world;
        }
    }
    if(!selected) for(int b=lane;b<32;b+=stride) {
        const int body=plan.body_ids.data[world*32+b];
        wp::spatial_vector twist;
        for(int k=0;k<6;++k) twist[k]=s[96+b*6+k];
        output.endpoint_twists.data[body]=twist;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
#endif
    return selected;
"""


@wp.func_native(_FINISH)
def _finish(
    address: wp.uint64, world: int, good: bool, invalid: int, plan: JointWorldPlan, output: LightOutput
) -> bool: ...


@functools.cache
def get_kernel(arch):
    """One warp per current world, no dynamic row-panel reservation."""
    from .raw_world_contacts import RawWorldContacts  # noqa: PLC0415

    source = normal_source() + "\n".join(
        operation_source(name)
        for name in (
            "update_qdd_from_velocity",
            "remove_free_root_transport_from_qdd",
            "integrate_generalized_joints",
        )
    )
    filename = f"<light-joint-world-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(kernels.__dict__)
    namespace.update(simple_world.__dict__)
    namespace.update(globals())
    exec(compile(source, filename, "exec"), namespace)
    check_normal = namespace["_light_check_normal"]
    update_qdd = namespace["_light_update_qdd_from_velocity"]
    remove_transport = namespace["_light_remove_free_root_transport_from_qdd"]
    integrate = namespace["_light_integrate_generalized_joints"]

    @wp.kernel(module="unique", enable_backward=False)
    def light_joint_world(
        plan: JointWorldPlan,
        predictor: PredictorData,
        data: simple_world._SimpleWorldInput,
        raw: simple_world._SimpleRawContacts,
        buckets: RawWorldContacts,
        output: LightOutput,
        integration: IntegrationData,
    ):
        world, _logical_lane = wp.tid()
        lanes = _lane()
        lane, stride = lanes[0], lanes[1]
        if stride == 1 and _logical_lane != 0:
            return
        address = _storage()
        _predict(address, world, plan, predictor, data, output, buckets.invalid[0])
        _twists(address, world, plan, data)
        _limits(address, world, data)
        good = bool(True)
        if _ready(address):
            for index in range(buckets.offsets[world] + lane, buckets.offsets[world + 1], stride):
                contact = buckets.ids[index]
                good = check_normal(contact, raw, data, address, plan) and good
        selected = _finish(address, world, good, buckets.invalid[0], plan, output)
        if selected:
            for local in range(lane, 35, stride):
                dof = plan.dof_ids[world, local]
                output.v_out[dof] = data.v_hat[dof]
                update_qdd(
                    dof, integration.joint_qd, predictor.kinematic_dof, 1.0 / data.dt, output.v_out, predictor.qdd
                )
            _sync()
            for root in range(lane, 2, stride):
                remove_transport(
                    plan.root_slots[world, root],
                    predictor.free_root_joints,
                    predictor.joint_qd_start,
                    predictor.kinematic_joint,
                    integration.joint_qd,
                    predictor.qdd,
                )
            _sync()
            for local in range(lane, 32, stride):
                integrate(
                    plan.joint_ids[world, local],
                    integration.joint_type,
                    integration.joint_parent,
                    integration.joint_child,
                    integration.joint_q_start,
                    integration.joint_qd_start,
                    integration.kinematic_joint_mask,
                    integration.joint_dof_dim,
                    integration.body_com,
                    integration.joint_X_c,
                    integration.joint_q,
                    integration.joint_qd,
                    predictor.qdd,
                    data.dt,
                    integration.angular_damping,
                    integration.joint_q_new,
                    integration.joint_qd_new,
                )
        _release(address)

    return light_joint_world

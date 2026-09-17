# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete compact Franka state and live-force owner; original PGS remains external."""

import functools
import hashlib
import inspect
import linecache
import math
import os
import textwrap

import numpy as np
import warp as wp

from ...sim import ModelFlags
from . import (  # noqa: F401 - names in the checked generated state factory
    kernels,
    kinetic_state,
    kuka_joint_world,
    world_scan_owner,
    world_scan_publication,
)

PublicationData = world_scan_publication.PublicationData


@wp.struct
class KineticPlan:
    body_ids: wp.array2d[int]
    body_parent: wp.array2d[int]
    joint_ids: wp.array2d[int]
    dof_ids: wp.array2d[int]
    root_slots: wp.array2d[int]
    primary_group: wp.array[int]
    secondary_group: wp.array[int]
    arts: wp.array2d[int]
    body_local_dof: wp.array2d[int]
    dof_body: wp.array[int]
    q_index: wp.array2d[int]
    source: wp.array2d[int]


@wp.struct
class KineticData:
    bias: wp.array2d[float]
    com_offset: wp.array2d[wp.vec3]
    free_bias: wp.array[wp.spatial_vector]
    geometric: wp.array2d[float]
    current_valid: wp.array[int]
    geometry_valid: wp.array[int]
    generation: wp.array[wp.int64]
    geometry_generation: wp.array[wp.int64]
    source_q: wp.array[wp.uint64]
    source_qd: wp.array[wp.uint64]
    status: wp.array[int]


@wp.struct
class ForceInput:
    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    predictor_qd: wp.array[float]
    joint_f: wp.array[float]
    body_f: wp.array[wp.spatial_vector]
    body_flags: wp.array[int]
    stiffness: wp.array[float]
    reference: wp.array[float]
    damping: wp.array[float]
    u0: wp.array[float]
    S: wp.array[wp.spatial_vector]
    lower9: wp.array3d[float]
    lower6: wp.array3d[float]
    kinematic_dof: wp.array[int]
    kinematic_joint: wp.array[int]
    qdd: wp.array[float]
    vhat: wp.array[float]
    dt: float


_release = world_scan_publication._release
_store_pose = world_scan_publication._store_pose
_load_pose = world_scan_publication._load_pose
_store_motion = world_scan_publication._store_motion
_load_motion = world_scan_publication._load_motion
_store_origin = world_scan_publication._store_origin
_load_origin = world_scan_publication._load_origin
_store_shift = kinetic_state._store_shift
_load_shift = kinetic_state._load_shift
_store_axis = kinetic_state._store_axis


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return wp::vec2i(threadIdx.x&15,16);
#else
    return wp::vec2i(0,1);
#endif
""")
def _lane() -> wp.vec2i: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return 2*group+((threadIdx.x&31)>>4);
#else
    return group;
#endif
""")
def _world(group: int) -> int: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return 16;
#else
    return 32;
#endif
""")
def _scan_width() -> int: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffu<<(threadIdx.x&16));
#endif
""")
def _sync(): ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[2*896];
    return reinterpret_cast<uint64_t>(values+896*((threadIdx.x&31)>>4));
#else
    return reinterpret_cast<uint64_t>(malloc(896*sizeof(float)));
#endif
""")
def _storage() -> wp.uint64: ...


def _paired_scan(source):
    """Keep the original parent-jump arithmetic, with independent CUDA half-warps."""
    cuda, cpu = source.split("#else\n", 1)
    original = "const int lane=threadIdx.x&31;"
    if cuda.count(original) != 1 or cuda.count(",source);") != cuda.count("__shfl_sync"):
        raise RuntimeError("Original world scan lane/shuffle seam changed")
    cuda = cuda.replace(original, "const int lane=threadIdx.x&15; const unsigned mask=0xffffu<<(threadIdx.x&16);")
    cuda = cuda.replace("__shfl_sync(0xffffffffu,", "__shfl_sync(mask,")
    cuda = cuda.replace(",source);", ",source,16);")
    cuda = cuda.replace("__syncwarp(0xffffffffu);", "__syncwarp(mask);")
    return cuda + "#else\n" + cpu


@wp.func_native(_paired_scan(world_scan_publication._POSE_SCAN))
def _scan_poses(address: wp.uint64, world: int, plan: KineticPlan): ...


@wp.func_native(_paired_scan(world_scan_publication._MOTION_SCAN))
def _scan_motion(address: wp.uint64, world: int, plan: KineticPlan): ...


@wp.func_native(r"""
    return cache.source_q.data[world]==reinterpret_cast<uint64_t>(q.data)
        &&cache.source_qd.data[world]==reinterpret_cast<uint64_t>(qd.data);
""")
def _same_source(cache: KineticData, world: int, q: wp.array[float], qd: wp.array[float]) -> int: ...


@wp.func_native(r"""
    cache.source_q.data[world]=reinterpret_cast<uint64_t>(q.data);
    cache.source_qd.data[world]=reinterpret_cast<uint64_t>(qd.data);
""")
def _stamp_source(cache: KineticData, world: int, q: wp.array[float], qd: wp.array[float]): ...


@wp.func_native(r"""
    float* s=reinterpret_cast<float*>(address);
    for(int k=0;k<6;++k)s[686+6*body+k]=wrench[k];
    if(refresh) {
        float* t=s+752+13*body;t[0]=mass;
        for(int k=0;k<3;++k)t[1+k]=mass*radius[k];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            t[4+3*i+j]=inertia.data[i][j]+mass*((i==j?wp::dot(radius,radius):0.0f)-radius[i]*radius[j]);
    }
""")
def _store_terms(
    address: wp.uint64,
    body: int,
    mass: float,
    radius: wp.vec3,
    inertia: wp.mat33,
    wrench: wp.spatial_vector,
    refresh: int,
): ...


@wp.func
def _primary_terms(
    address: wp.uint64,
    local: int,
    world: int,
    body: int,
    pose: wp.transform,
    pose_com: wp.transform,
    origin: wp.vec3,
    velocity: wp.spatial_vector,
    acceleration: wp.spatial_vector,
    data: PublicationData,
    cache: KineticData,
    refresh: int,
):
    radius = wp.transform_get_translation(pose_com) - origin
    rotation = wp.transform_get_rotation(pose_com)
    mass, inertia_local = data.body_mass[body], data.body_inertia[body]
    linear, omega = wp.spatial_top(velocity), wp.spatial_bottom(velocity)
    a, alpha = wp.spatial_top(acceleration), wp.spatial_bottom(acceleration)
    inertia = wp.mat33()
    momentum, acceleration_force = wp.vec3(), wp.vec3()
    if refresh != 0:
        matrix = wp.quat_to_matrix(rotation)
        inertia = matrix * inertia_local * wp.transpose(matrix)
        momentum, acceleration_force = inertia * omega, inertia * alpha
    else:
        momentum = wp.quat_rotate(rotation, inertia_local * wp.quat_rotate_inv(rotation, omega))
        acceleration_force = wp.quat_rotate(rotation, inertia_local * wp.quat_rotate_inv(rotation, alpha))
    force = mass * (a + wp.cross(alpha, radius) + wp.cross(omega, linear + wp.cross(omega, radius)) - data.gravity[0])
    torque = acceleration_force + wp.cross(omega, momentum) + wp.cross(radius, force)
    public_radius = wp.transform_point(pose, data.body_com[body]) - origin
    data.body_qd[body] = wp.spatial_vector(linear + wp.cross(omega, public_radius), omega)
    cache.com_offset[world, local] = public_radius
    _store_terms(address, local, mass, radius, inertia, wp.spatial_vector(force, torque), refresh)


@wp.func_native(r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&15,stride=16;
    const unsigned mask=0xffffu<<(threadIdx.x&16);
#else
    const int lane=0,stride=1;
#endif
    // Each component owns its complete eleven-body subtree reduction.
    // No atomics, prefix subtraction, or spatial matrix materialization.
    for(int component=lane;component<(refresh?19:6);component+=stride) {
        const int width=component<6?6:13;
        float* base=s+(component<6?686+component:752+component-6);
        for(int body=10;body>0;--body) {
            const int parent=plan.body_parent.data[world*32+body];
            base[width*parent]+=base[width*body];
        }
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    bool good=true;
    for(int dof=lane;dof<9;dof+=stride) {
        const int body=plan.dof_body.data[dof];
        const float* axis=s+632+6*dof;
        float bias=0.0f;for(int k=0;k<6;++k)bias+=axis[k]*s[686+6*body+k];
        cache.bias.data[world*9+dof]=bias;good&=wp::isfinite(bias);
        if(refresh) {
            const float* t=s+752+13*body;
            const wp::vec3 l(axis[0],axis[1],axis[2]),w(axis[3],axis[4],axis[5]),h(t[1],t[2],t[3]);
            const wp::vec3 linear=t[0]*l+wp::cross(w,h);
            wp::vec3 angular=wp::cross(h,l);
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)angular[i]+=t[4+3*i+j]*w[j];
            for(int k=0;k<3;++k){s[6*dof+k]=linear[k];s[6*dof+3+k]=angular[k];}
        }
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    if(refresh)for(int e=lane;e<81;e+=stride) {
        const int row=e/9,col=e%9,src=plan.source.data[e];
        float value=0.0f;
        if(src>=0){const int projection=src==col?row:col;for(int k=0;k<6;++k)value+=s[632+6*projection+k]*s[6*src+k];}
        cache.geometric.data[plan.primary_group.data[world]*81+e]=value;good&=wp::isfinite(value);
    }
    for(int local=lane;local<13;local+=stride) {
        const int body=plan.body_ids.data[world*32+local];
        const auto q=data.body_q.data[body];const auto qd=data.body_qd.data[body];
        for(int k=0;k<7;++k)good&=wp::isfinite(q[k]);
        for(int k=0;k<6;++k)good&=wp::isfinite(qd[k]);
    }
#if defined(__CUDA_ARCH__)
    good=__all_sync(mask,good);
#endif
    if(lane==0) {
        if(!good)cache.status.data[world]|=1;
        cache.generation.data[world]+=1;
        cache.geometry_valid.data[world]=refresh&&cache.status.data[world]==0;
        if(refresh)cache.geometry_generation.data[world]=cache.generation.data[world];
    }
    return good?1:0;
""")
def _collect(
    address: wp.uint64, world: int, plan: KineticPlan, data: PublicationData, cache: KineticData, refresh: int
) -> int: ...


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise RuntimeError(f"Franka original state seam changed: {old[:80]!r}")
    return source.replace(old, new)


def state_source():
    """Retain the checked original integration/free-body law, replacing primary ownership."""
    source = textwrap.dedent(inspect.getsource(kinetic_state._get_kernel))
    source = source.removeprefix("@functools.cache\n")
    source = _replace(source, "def _get_kernel(arch, finish):", "def _factory(finish):")
    source = _replace(source, "    checked_source()\n", "")
    source = _replace(source, '    @wp.kernel(module="unique", enable_backward=False)\n', "")
    source = _replace(
        source,
        "        world, logical_lane = wp.tid()\n",
        """        group, logical_lane = wp.tid()
        world = _world(group)
        if world >= plan.body_ids.shape[0]:
            return
""",
    )
    source = _replace(
        source,
        "        schedule: KineticSchedule,\n        current: CurrentKineticCache,\n        geometric: GeometricCache,",
        "        cache: KineticData,\n        requests: wp.array[int],\n        global_refresh: int,",
    )
    source = _replace(
        source,
        "        if _admit(world, int(wp.static(finish)), plan, data, schedule, current, geometric) == 0:\n            return\n        data.gravity = _world_gravity(data.gravity, world)\n",
        """        refresh = int(global_refresh != 0)
        if not wp.static(finish):
            refresh = int(refresh != 0 or requests[0] != 0)
            valid = int(1)
            for root in range(3):
                valid = valid & cache.current_valid[plan.arts[world, root]]
            if valid != 0 and _same_source(cache, world, data.joint_q, data.joint_qd) != 0:
                if refresh == 0 or (cache.geometry_valid[world] != 0 and cache.geometry_generation[world] == cache.generation[world]):
                    return
""",
    )
    source = source.replace("range(lane, 35, stride)", "range(lane, 21, stride)")
    # Only pose/motion scans use padding; physical accesses retain all13 bodies.
    source = source.replace("range(lane, 32, stride)", "range(lane, 13, stride)")
    source = (
        source.replace("local >= 30", "local >= 11")
        .replace("local < 30", "local < 11")
        .replace("local == 30", "local == 11")
    )
    source = source.replace("local - 29", "local - 10").replace("29 + root", "10 + root")
    source = source.replace("current.", "cache.")
    source = _replace(
        source,
        "        _scan_poses(address, world, plan)",
        """        for padding in range(lane, _scan_width(), stride):
            if padding >= 13:
                _store_pose(address, padding, wp.transform_identity())
        _sync()
        _scan_poses(address, world, plan)""",
    )
    source = _replace(
        source,
        "            cache.origin[world, root] = origin + _load_shift(address, root)\n            if root > 0:\n                data.articulation_origin[data.body_to_articulation[body]] = cache.origin[world, root]",
        "            data.articulation_origin[data.body_to_articulation[body]] = origin + _load_shift(address, root)",
    )
    first = source.index("            if local < 11:\n                if local_dof >= 0:")
    end = source.index("            _store_motion(address, local, velocity)", first)
    source = (
        source[:first]
        + """            velocity = kernels.jcalc_motion(
                data.joint_type[joint], data.joint_axis,
                data.joint_dof_dim[joint, 0], data.joint_dof_dim[joint, 1],
                anchor_local, qd, data.joint_qd_start[joint], data.joint_S_s,
            )
            if local < 11 and local_dof >= 0:
                axis = data.joint_S_s[data.joint_qd_start[joint]]
                _store_axis(address, local_dof, axis)
"""
        + source[end:]
    )
    source = _replace(
        source,
        "        _scan_motion(address, world, plan)",
        """        for padding in range(lane, _scan_width(), stride):
            if padding >= 13:
                _store_motion(address, padding, wp.spatial_vector())
        _sync()
        _scan_motion(address, world, plan)""",
    )
    source = _replace(
        source,
        "                    schedule,\n                    current,",
        "                    cache,\n                    refresh,",
    )
    source = _replace(
        source,
        "        success = _collect(address, world, plan, data, schedule, current, geometric)",
        "        success = _collect(address, world, plan, data, cache, refresh)",
    )
    source = _replace(
        source,
        "                cache.com_offset[world, local] = wp.transform_get_translation(pose_com) - origin",
        "                cache.com_offset[world, local] = wp.transform_point(pose, data.body_com[body]) - origin",
    )
    source = _replace(
        source, "                root > 0 and success != 0", "                success != 0 and cache.status[world] == 0"
    )
    source = _replace(
        source,
        "        _release(address)",
        "        if lane == 0:\n            _stamp_source(cache, world, q, qd)\n        _release(address)",
    )
    source = _replace(
        source,
        "    return kinetic_state",
        """    name = "franka_kinetic_finish13_h81_p16" if finish else "franka_kinetic_repair13_h81_p16"
    kinetic_state.__name__ = name
    kinetic_state.__qualname__ = name
    return wp.kernel(module="unique", enable_backward=False)(kinetic_state)""",
    )
    return source


@functools.cache
def get_state_kernel(finish):
    source = state_source()
    filename = f"<franka-kinetic-state-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(globals())
    exec(compile(source, filename, "exec"), namespace)
    return namespace["_factory"](finish)


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[2*120];
    return reinterpret_cast<uint64_t>(values+120*((threadIdx.x&31)>>4));
#else
    return reinterpret_cast<uint64_t>(malloc(120*sizeof(float)));
#endif
""")
def _force_storage() -> wp.uint64: ...


@wp.func_native(r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&15,stride=16;
    const unsigned mask=0xffffu<<(threadIdx.x&16);
#else
    const int lane=0,stride=1;
#endif
    bool good=wp::isfinite(f.dt)&&f.dt>0.0f&&cache.status.data[world]==0;
    for(int root=0;root<3;++root)good&=cache.current_valid.data[plan.arts.data[world*3+root]]!=0;
    for(int b=lane;b<13;b+=stride) {
        const int body=plan.body_ids.data[world*32+b];
        float v[6]={};
        if(b<12&&(f.body_flags.data[body]&2)==0) {
            const auto external=f.body_f.data[body];const auto r=cache.com_offset.data[world*13+b];
            const wp::vec3 force(external[0],external[1],external[2]);
            const wp::vec3 torque=wp::vec3(external[3],external[4],external[5])+wp::cross(r,force);
            for(int k=0;k<3;++k){v[k]=force[k];v[k+3]=torque[k];}
        }
        for(int k=0;k<6;++k){s[6*b+k]=v[k];good&=wp::isfinite(v[k]);}
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    for(int component=lane;component<6;component+=stride)
        for(int b=10;b>0;--b)s[6*plan.body_parent.data[world*32+b]+component]+=s[6*b+component];
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    for(int i=lane;i<15;i+=stride) {
        const int dof=plan.dof_ids.data[world*21+i];
        float tau=f.joint_f.data[dof]+f.u0.data[dof];
        if(i<9) {
            const auto axis=f.S.data[dof];const int body=plan.dof_body.data[i];
            float ext=0.0f;for(int k=0;k<6;++k)ext+=axis[k]*s[6*body+k];
            const float q=f.joint_q.data[plan.q_index.data[world*9+i]],qd=f.joint_qd.data[dof];
            tau+=-cache.bias.data[world*9+i]+ext+f.stiffness.data[dof]*(f.reference.data[dof]-q)-f.damping.data[dof]*qd;
        } else {
            const auto axis=f.S.data[dof],bias=cache.free_bias.data[world];
            for(int k=0;k<6;++k)tau-=axis[k]*(bias[k]-s[6*11+k]);
        }
        s[78+i]=tau;good&=wp::isfinite(tau);
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    // Two independent original held-L triangular actions; no new factor or inverse.
    for(int family=lane;family<2;family+=stride) {
        const int n=family==0?9:6,offset=family==0?0:9;
        const int group=family==0?plan.primary_group.data[world]:plan.secondary_group.data[world];
        const float* l=family==0?f.lower9.data+group*81:f.lower6.data+group*36;
        float* x=s+93+offset;
        for(int row=0;row<n;++row) {
            float value=s[78+offset+row];
            for(int k=0;k<row;++k)value-=l[row*n+k]*x[k];
            const float diagonal=l[row*n+row];x[row]=diagonal!=0.0f?value/diagonal:0.0f;
        }
        for(int row=n-1;row>=0;--row) {
            float value=x[row];for(int k=row+1;k<n;++k)value-=l[k*n+row]*x[k];
            const float diagonal=l[row*n+row];x[row]=diagonal!=0.0f?value/diagonal:0.0f;
            good&=wp::isfinite(x[row]);
        }
    }
#if defined(__CUDA_ARCH__)
    good=__all_sync(mask,good);__syncwarp(mask);
#endif
    if(!good){if(lane==0)cache.status.data[world]|=2;return;}
    for(int i=lane;i<21;i+=stride) {
        const int dof=plan.dof_ids.data[world*21+i];
        const float acceleration=i<15&&f.kinematic_dof.data[dof]==0?s[93+i]:0.0f;
        f.qdd.data[dof]=acceleration;
        f.vhat.data[dof]=f.predictor_qd.data[dof]+acceleration*f.dt;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    for(int root=lane;root<2;root+=stride) {
        const int joint=plan.joint_ids.data[world*32+11+root];
        if(f.kinematic_joint.data[joint]!=0)continue;
        const int d=plan.dof_ids.data[world*21+9+6*root];
        const wp::vec3 v(f.predictor_qd.data[d],f.predictor_qd.data[d+1],f.predictor_qd.data[d+2]);
        const wp::vec3 w(f.predictor_qd.data[d+3],f.predictor_qd.data[d+4],f.predictor_qd.data[d+5]);
        const auto c=wp::cross(w,v);for(int k=0;k<3;++k)f.vhat.data[d+k]+=c[k]*f.dt;
    }
""")
def _predict(address: wp.uint64, world: int, plan: KineticPlan, cache: KineticData, f: ForceInput): ...


@functools.cache
def get_predictor_kernel():
    def predictor(plan: KineticPlan, cache: KineticData, force: ForceInput):
        group, logical_lane = wp.tid()
        world = _world(group)
        if world >= plan.body_ids.shape[0]:
            return
        lanes = _lane()
        if lanes[1] == 1 and logical_lane != 0:
            return
        address = _force_storage()
        _predict(address, world, plan, cache, force)
        _release(address)

    predictor.__name__ = "franka_kinetic_current_force_held9_6_p16"
    predictor.__qualname__ = predictor.__name__
    return wp.kernel(module="unique", enable_backward=False)(predictor)


def build_plan(solver):
    """Build complete physical IDs, rejecting omissions and changed scalar topology."""
    model, worlds = solver.model, int(solver.world_count)
    art_world = solver.art_to_world.numpy()
    starts, ends = model.articulation_start.numpy(), solver.articulation_joint_end.numpy()
    dof_starts = solver.articulation_dof_start.numpy()
    counts = np.diff(np.r_[dof_starts, model.joint_dof_count])
    responses = solver.articulation_response_dof_count.numpy()
    prescribed = solver._prescribed_articulation.numpy()
    types, parents, children = model.joint_type.numpy(), model.joint_parent.numpy(), model.joint_child.numpy()
    qstarts, dstarts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    dimensions = model.joint_dof_dim.numpy()
    body_art, masks = solver.body_to_articulation.numpy(), solver.body_response_dof_mask.numpy()
    group9, group6 = solver.group_to_art[9].numpy(), solver.group_to_art[6].numpy()
    primary = {int(a): g for g, a in enumerate(group9)}
    secondary = {int(a): g for g, a in enumerate(group6)}
    roots = {int(j): k for k, j in enumerate(solver._free_root_joint_indices.numpy())}
    fields = {
        name: np.full((worlds, width), -1, np.int32)
        for name, width in (
            ("body_ids", 32),
            ("body_parent", 32),
            ("joint_ids", 32),
            ("dof_ids", 21),
            ("root_slots", 2),
            ("arts", 3),
            ("body_local_dof", 32),
            ("q_index", 9),
        )
    }
    fields.update(primary_group=np.full(worlds, -1, np.int32), secondary_group=np.full(worlds, -1, np.int32))
    expected_parent = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8], np.int32)
    dof_body = np.array([1, 2, 3, 4, 5, 6, 7, 9, 10], np.int32)
    if len(art_world) != 3 * worlds or model.body_count != 13 * worlds or model.joint_dof_count != 21 * worlds:
        raise ValueError("Franka kinetic requires exact 13-body/21-coordinate/three-articulation worlds")
    by_world = [[] for _ in range(worlds)]
    for art, world in enumerate(art_world):
        if not 0 <= world < worlds:
            raise ValueError("Franka articulation outside complete world ownership")
        by_world[int(world)].append(art)
    for world, arts in enumerate(by_world):
        a = [art for art in arts if art in primary]
        b = [art for art in arts if art in secondary]
        c = [art for art in arts if prescribed[art] != 0]
        if len(arts) != 3 or (len(a), len(b), len(c)) != (1, 1, 1):
            raise ValueError("Franka kinetic requires primary9/free6/prescribed6")
        ordered = [a[0], b[0], c[0]]
        if len(set(ordered)) != 3 or list(counts[ordered]) != [9, 6, 6] or list(responses[ordered]) != [9, 6, 0]:
            raise ValueError("Franka kinetic physical and response coordinates do not match")
        if list(ends[ordered] - starts[ordered]) != [11, 1, 1]:
            raise ValueError("Franka kinetic requires all11 primary joints and both free roots")
        joints = np.concatenate([np.arange(starts[art], ends[art]) for art in ordered])
        bodies = children[joints]
        if (
            np.any(bodies < 0)
            or len(np.unique(bodies)) != 13
            or not np.array_equal(body_art[bodies], np.repeat(ordered, [11, 1, 1]))
        ):
            raise ValueError("Franka joint/body ownership mismatch")
        lookup = {int(body): i for i, body in enumerate(bodies)}
        parent = np.array([lookup.get(int(parents[joint]), -1) for joint in joints], np.int32)
        if not np.array_equal(parent[:11], expected_parent) or np.any(parents[joints[11:]] >= 0):
            raise ValueError("Franka kinetic requires the checked primary tree and independent free roots")
        if np.any((parents[joints] >= 0) & (parent < 0)):
            raise ValueError("Franka kinetic parent escapes its world")
        # Existing enum: PRISMATIC0, REVOLUTE1, FIXED3, FREE4.
        if list(types[joints]) != [3, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 4, 4]:
            raise ValueError("Franka kinetic supports the exact fixed/revolute/prismatic/free joint laws")
        dofs = np.concatenate([np.arange(dof_starts[art], dof_starts[art] + counts[art]) for art in ordered])
        ancestor_mask = np.zeros(11, np.uint32)
        for local in range(11):
            own = -1
            if local in dof_body:
                own = int(np.flatnonzero(dof_body == local)[0])
                if dstarts[joints[local]] != dofs[own] or dimensions[joints[local]].sum() != 1:
                    raise ValueError("Franka scalar coordinate mapping changed")
                fields["body_local_dof"][world, local] = own
                fields["q_index"][world, own] = qstarts[joints[local]]
            elif dimensions[joints[local]].sum() != 0:
                raise ValueError("Franka weld gained physical coordinates")
            inherited = int(ancestor_mask[parent[local]]) if parent[local] >= 0 else 0
            ancestor_mask[local] = inherited | (1 << own if own >= 0 else 0)
        if not np.array_equal(masks[bodies[:11]], ancestor_mask) or masks[bodies[11]] != 63:
            raise ValueError("Franka kinetic must preserve original ancestor response masks")
        fields["arts"][world] = ordered
        fields["body_ids"][world, :13], fields["joint_ids"][world, :13] = bodies, joints
        fields["body_parent"][world, :13], fields["dof_ids"][world] = parent, dofs
        fields["primary_group"][world], fields["secondary_group"][world] = primary[a[0]], secondary[b[0]]
        fields["root_slots"][world] = [roots[int(j)] for j in joints[11:]]
    if (
        len(np.unique(fields["body_ids"][:, :13])) != model.body_count
        or len(np.unique(fields["dof_ids"])) != model.joint_dof_count
    ):
        raise ValueError("Franka kinetic must cover every public body and physical coordinate exactly once")
    fields["dof_body"] = dof_body
    fields["source"] = solver._crba_source_dof_by_size[9].numpy().astype(np.int32)
    expected = np.full((9, 9), -1, np.int32)
    for row in range(9):
        for col in range(9):
            if row == col or row < 7 or col < 7:
                expected[row, col] = max(row, col)
    if not np.array_equal(fields["source"], expected):
        raise ValueError("Franka kinetic CRBA source schedule changed")
    return fields


def supported(solver):
    """Require the unchanged immediate matrix-free publication and factor consumers."""
    from . import solver_feather_pgs as source  # noqa: PLC0415

    return bool(
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and not solver.model.particle_count
        and solver.size_groups == [9, 6]
        and solver.pgs_mode == "matrix_free"
        and solver.articulated_contact_response == "immediate"
        and solver.drive_mode == "augmented"
        and solver.pgs_velocity_iterations == 0
        and not solver.enable_joint_velocity_limits
        and not solver.grouped_dynamics
        and not solver.lazy_kinematics
        and not solver._fused_k1
        and solver._fk_id_cache_enabled
        and not solver._fk_id_cache_uses_snapshot
        and solver._fk_id_cache is None
        and solver._async_augmented_drives
        and solver._parallel_augmented_drive_topology
        and solver._articulation_dynamics_stream is not None
        and not solver._debug_buffers_enabled
        and solver._crba_cholesky_warp_kernels_by_size.get(9) is not None
        and solver._crba_cholesky_warp_kernels_by_size.get(6) is not None
        and all(
            getattr(solver, name, None) is None
            for name in (
                "_sparse_factor",
                "_world_scan_publication",
                "_kinetic_world",
                "_joint_world",
                "_prismatic_publication",
            )
        )
        and not any(
            (
                source._DEBUG_CACHE,
                source._DEBUG_CACHE_MODE,
                source._DEBUG_CACHE_CMP,
                source._GROUPED_CHECK,
                source._FPGS_CAPTURE,
            )
        )
    )


def _fingerprint(array):
    value = array.numpy()
    return value.shape, value.dtype.str, hashlib.sha256(value.tobytes()).digest()


@wp.kernel(module="unique", enable_backward=False)
def _compare_proof_words(current: wp.array[wp.uint32], frozen: wp.array[wp.uint32], mismatch: wp.array[int], bit: int):
    index = wp.tid()
    if current[index] != frozen[index]:
        wp.atomic_or(mismatch, 0, bit)


def _plan_dimensions(solver):
    """Include scalar proof inputs and the dimensions of the retained storage owners."""
    return (
        solver.world_count,
        solver.model.world_count,
        solver.model.body_count,
        solver.model.joint_count,
        solver.model.articulation_count,
        solver.model.joint_coord_count,
        solver.model.joint_dof_count,
        solver.n_arts_by_size.get(9),
        solver.n_arts_by_size.get(6),
    )


@wp.kernel
def _invalidate(world_mask: wp.array[wp.bool], plan: KineticPlan, cache: KineticData):
    world = wp.tid()
    if not world_mask or world_mask[world]:
        for root in range(3):
            cache.current_valid[plan.arts[world, root]] = 0
        cache.geometry_valid[world] = 0


class FrankaKineticState:
    """Live current/bias/moment lifetime without taking ownership of constraint solves."""

    def __init__(self, solver):
        self.solver = solver
        model, device, worlds = solver.model, solver.model.device, int(solver.world_count)
        if model.requires_grad or model.particle_count:
            raise ValueError("Franka kinetic state excludes gradient and particle ownership")
        self.host_plan = build_plan(solver)
        self.plan = KineticPlan()
        for name, value in self.host_plan.items():
            setattr(self.plan, name, wp.array(value, dtype=int, device=device))
        self.data = KineticData()
        for name, shape, dtype in (
            ("bias", (worlds, 9), float),
            ("com_offset", (worlds, 13), wp.vec3),
            ("free_bias", worlds, wp.spatial_vector),
            ("geometric", (solver.n_arts_by_size[9], 81), float),
            ("geometry_valid", worlds, int),
            ("generation", worlds, wp.int64),
            ("geometry_generation", worlds, wp.int64),
            ("source_q", worlds, wp.uint64),
            ("source_qd", worlds, wp.uint64),
            ("status", worlds, int),
        ):
            setattr(self.data, name, wp.zeros(shape, dtype=dtype, device=device))
        self.data.current_valid = solver._fk_id_cache_valid
        self.data.current_valid.zero_()
        self.geometry_valid, self.status = self.data.geometry_valid, self.data.status
        self.geometric, self.bias = self.data.geometric, self.data.bias
        self.compact_workspace = os.environ.get("FEATHER_PGS_FRANKA_COMPACT_WORKSPACE", "0") == "1"
        state_factory = get_state_kernel
        if self.compact_workspace:
            from .franka_compact_workspace import get_state_kernel as state_factory  # noqa: PLC0415
        self.repair_kernel, self.finish_kernel = state_factory(False), state_factory(True)
        self.predictor_kernel = get_predictor_kernel()
        fields = (*world_scan_owner.PLAN_FIELDS, "joint_axis", "joint_X_p", "joint_X_c")
        self._model_plan = {name: _fingerprint(getattr(model, name)) for name in fields}
        self._solver_plan = {
            name: _fingerprint(getattr(solver, name))
            for name in (
                "art_to_world",
                "articulation_joint_end",
                "articulation_dof_start",
                "articulation_response_dof_count",
                "body_to_articulation",
                "_prescribed_articulation",
                "body_response_dof_mask",
                "_free_root_joint_indices",
            )
        }
        self._group_plan = {
            (name, size): _fingerprint(getattr(solver, name)[size])
            for name, size in (("group_to_art", 9), ("group_to_art", 6), ("_crba_source_dof_by_size", 9))
        }
        self._plan_dimensions = _plan_dimensions(solver)
        self._device_proof = None
        if device.is_cuda:
            self._initialize_device_proof()

    def _proof_arrays(self):
        """Yield the original proof operands and errors in their original order."""
        for name, expected in self._model_plan.items():
            yield f"model {name}", getattr(self.solver.model, name), expected
        for name, expected in self._solver_plan.items():
            yield f"mapping {name}", getattr(self.solver, name), expected
        for (name, size), expected in self._group_plan.items():
            yield f"mapping {name}[{size}]", getattr(self.solver, name).get(size), expected

    def _initialize_device_proof(self):
        """Freeze exact-sized proof words without changing the host fallback."""
        device = self.solver.model.device
        views = []
        for _, value, _ in self._proof_arrays():
            if value is None or value.device != device or not value.is_contiguous:
                return
            try:
                view = value.view(wp.uint32).flatten()
            except (TypeError, RuntimeError):
                return
            views.append((value, view))
        # Each view retains its source allocation. Only these exact proof
        # words are duplicated; no maximum-capacity or physics buffers enter.
        self._device_proof = [(wp.clone(view), value, view, value.shape, value.dtype) for value, view in views]
        self._proof_device = device
        self._proof_status = wp.zeros(1, dtype=int, device=device)

    def _validate_device_proof(self):
        """Compare all pinned bits before one compact readback and cache mutation."""
        arrays = list(self._proof_arrays())
        self._proof_status.zero_()
        for index, (label, value, expected) in enumerate(arrays):
            frozen, source, view, shape, dtype = self._device_proof[index]
            if (
                value is None
                or value.shape != shape
                or value.dtype != dtype
                or value.device != self._proof_device
                or not value.is_contiguous
            ):
                # Preserve logical NumPy shape/dtype/content semantics for
                # unusual layouts or bindings; do not silently reject them.
                if value is None or _fingerprint(value) != expected:
                    raise RuntimeError(f"Franka kinetic {label} changed; reconstruct and recapture")
                continue
            if value is not source or value.ptr != view.ptr:
                view = value.view(wp.uint32).flatten()
                self._device_proof[index] = frozen, value, view, shape, dtype
            # Identity only reuses a zero-copy descriptor. Every current word
            # is compared even when the source object and pointer are stable.
            wp.launch(
                _compare_proof_words,
                dim=frozen.size,
                inputs=[view, frozen, self._proof_status, 1 << index],
                device=self._proof_device,
            )
        mismatch = int(self._proof_status.numpy()[0])
        for index, (label, _, _) in enumerate(arrays):
            if mismatch & (1 << index):
                raise RuntimeError(f"Franka kinetic {label} changed; reconstruct and recapture")

    def validate_model(self, flags=None):
        """Check all immutable ownership before caller mutates validity or held epochs."""
        numeric = int(
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        if flags is not None and int(flags) != 0 and int(flags) & ~numeric == 0:
            # Match the already qualified notification contract: these flags
            # change numeric bindings, not parent/axis/anchor/body membership.
            return
        if _plan_dimensions(self.solver) != self._plan_dimensions:
            raise RuntimeError("Franka kinetic dimensions changed; reconstruct and recapture")
        if self._device_proof is not None:
            # Match array.numpy(): order reads on the null stream after the
            # caller stream, then restore it without a new global sync API.
            with wp.ScopedStream(self._proof_device.null_stream):
                self._validate_device_proof()
            return
        for name, expected in self._model_plan.items():
            if _fingerprint(getattr(self.solver.model, name)) != expected:
                raise RuntimeError(f"Franka kinetic model {name} changed; reconstruct and recapture")
        for name, expected in self._solver_plan.items():
            if _fingerprint(getattr(self.solver, name)) != expected:
                raise RuntimeError(f"Franka kinetic mapping {name} changed; reconstruct and recapture")
        for (name, size), expected in self._group_plan.items():
            value = getattr(self.solver, name).get(size)
            if value is None or _fingerprint(value) != expected:
                raise RuntimeError(f"Franka kinetic mapping {name}[{size}] changed; reconstruct and recapture")
        # build_plan is a pure function of the complete inputs checked above.
        # Equal shape/dtype/content and scalar dimensions preserve its original
        # proof; rebuilding its per-world Python loop on every unchanged root
        # notification adds no validation and scales with the environment count.

    def invalidate(self, world_mask=None):
        if world_mask is not None and (
            world_mask.dtype != wp.bool
            or world_mask.shape != (self.solver.world_count,)
            or world_mask.device != self.solver.model.device
        ):
            raise ValueError("Franka kinetic reset requires the original per-world bool mask")
        wp.launch(
            _invalidate,
            dim=self.solver.world_count,
            inputs=[world_mask, self.plan, self.data],
            device=self.solver.model.device,
        )

    def _validate_call(self, state, state_aug, dt):
        if state.requires_grad or state_aug is not self.solver:
            raise RuntimeError("Franka kinetic state excludes gradient/alternate augmented state")
        if self.solver.model.device.is_cuda and not supported(self.solver):
            raise RuntimeError("Franka kinetic configuration changed; reconstruct and recapture")
        if dt is None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Franka kinetic requires positive finite dt")
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            value = getattr(state, name)
            if value.device != self.solver.model.device or not value.is_contiguous:
                raise RuntimeError("Franka kinetic state requires contiguous same-device input")

    def _publication(self, state_in, state_aug, state_out, dt):
        solver, model = self.solver, self.solver.model
        data = PublicationData()
        for name in world_scan_owner.MODEL_FIELDS:
            setattr(data, name, getattr(model, name))
        for name, source in world_scan_owner.SOLVER_FIELDS.items():
            setattr(data, name, getattr(solver, source))
        data.joint_q, data.joint_qd = state_in.joint_q, state_in.joint_qd
        data.joint_q_new, data.joint_qd_new = state_out.joint_q, state_out.joint_qd
        data.joint_qdd = state_aug.joint_qdd
        data.body_q, data.body_qd = state_out.body_q, state_out.body_qd
        for name in world_scan_owner.CACHE_FIELDS:
            if name == "articulation_origin":
                value = solver.articulation_origin
            elif name == "body_inertia_terms":
                value = solver._body_inertia_terms
            else:
                value = getattr(state_aug, name)
            setattr(data, name, value)
        data.dt, data.angular_damping = dt, solver.angular_damping
        # The original free finalizer materializes I_s for is_free_rigid even on
        # held steps. Primary full I_s/terms are intentionally not produced.
        data.materialize_all_body_inertia = 0
        data.materialize_body_inertia_terms = 0
        return data

    def _launch_count(self):
        """Use two independent CUDA worlds per CTA; retain serial CPU ownership."""
        worlds = self.solver.world_count
        return (worlds + 1) // 2 if self.solver.model.device.is_cuda else worlds

    def begin(self, state_in, state_aug, dt, global_refresh):
        self._validate_call(state_in, state_aug, dt)
        wp.launch_tiled(
            self.repair_kernel,
            dim=[self._launch_count()],
            inputs=[
                self.plan,
                self._publication(state_in, state_aug, state_in, dt),
                self.data,
                self.solver._mass_update_requested,
                int(global_refresh),
            ],
            block_dim=32,
            device=self.solver.model.device,
        )

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        self._validate_call(state_in, state_aug, dt)
        if stage3_qd.ptr != state_in.joint_qd.ptr:
            raise RuntimeError("Franka kinetic excludes alternate prescaled predictor velocity")
        solver = self.solver
        force = ForceInput()
        force.joint_q, force.joint_qd, force.predictor_qd = state_in.joint_q, state_in.joint_qd, stage3_qd
        force.joint_f, force.body_f, force.body_flags = control.joint_f, state_in.body_f, solver.model.body_flags
        force.stiffness, force.reference, force.damping = (
            solver._passive_spring_stiffness,
            solver._passive_spring_ref,
            solver._passive_joint_damping,
        )
        force.u0 = state_aug.joint_tau
        force.S = state_aug.joint_S_s
        force.lower9, force.lower6 = solver.L_by_size[9], solver.L_by_size[6]
        force.kinematic_dof, force.kinematic_joint = solver._kinematic_dof_mask, solver._kinematic_joint_mask
        force.qdd, force.vhat, force.dt = state_aug.joint_qdd, solver.v_hat, dt
        wp.launch_tiled(
            self.predictor_kernel,
            dim=[self._launch_count()],
            inputs=[self.plan, self.data, force],
            block_dim=32,
            device=solver.model.device,
        )

    def finish(self, state_in, state_aug, state_out, dt, next_refresh):
        self._validate_call(state_in, state_aug, dt)
        self._validate_call(state_out, state_aug, dt)
        if any(
            a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity
            for a in (state_in.joint_q, state_in.joint_qd)
            for b in (state_out.joint_q, state_out.joint_qd)
        ):
            raise RuntimeError("Franka kinetic integration requires disjoint generalized state banks")
        wp.launch_tiled(
            self.finish_kernel,
            dim=[self._launch_count()],
            inputs=[
                self.plan,
                self._publication(state_in, state_aug, state_out, dt),
                self.data,
                self.solver._mass_update_requested,
                int(next_refresh),
            ],
            block_dim=32,
            device=self.solver.model.device,
        )

    def check(self):
        values = self.status.numpy()
        if np.any(values):
            raise RuntimeError(f"Franka kinetic native failure in {np.count_nonzero(values)} worlds")

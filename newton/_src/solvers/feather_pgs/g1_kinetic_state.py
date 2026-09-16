# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current/next G1 moments and bias with unchanged held sparse W consumers."""

import functools
import hashlib
import linecache

import warp as wp

from . import kernels, kuka_joint_world, sparse_factor
from .world_scan_publication import PublicationData


@wp.struct
class KineticPlan:
    parent: wp.array[int]
    depth: wp.array[int]
    jump: wp.array2d[int]
    children_offsets: wp.array[int]
    children: wp.array[int]
    q_index: wp.array[int]
    root_slot: wp.array[int]


@wp.struct
class KineticData:
    bias: wp.array2d[float]
    com_offset: wp.array2d[wp.vec3]
    geometric: wp.array2d[float]
    current_valid: wp.array[int]
    geometry_valid: wp.array[int]
    generation: wp.array[wp.int64]
    geometry_generation: wp.array[wp.int64]
    status: wp.array[int]
    source_q: wp.array[wp.uint64]
    source_qd: wp.array[wp.uint64]


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
    qdd: wp.array[float]
    vhat: wp.array[float]
    dt: float


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    return wp::vec_t<2,int>(threadIdx.x,blockDim.x);
#else
    return wp::vec_t<2,int>(0,1);
#endif
""")
def _lanes() -> wp.vec2i: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
""")
def _sync(): ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[1936];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(1936*sizeof(float)));
#endif
""")
def _storage() -> wp.uint64: ...


@wp.func_native(r"""
#if !defined(__CUDA_ARCH__)
    free(reinterpret_cast<void*>(address));
#endif
""")
def _release(address: wp.uint64): ...


@wp.func_native(
    "return cache.source_q.data[group]==reinterpret_cast<uint64_t>(q.data)&&cache.source_qd.data[group]==reinterpret_cast<uint64_t>(qd.data);"
)
def _same_source(cache: KineticData, group: int, q: wp.array[float], qd: wp.array[float]) -> int: ...


@wp.func_native(
    "cache.source_q.data[group]=reinterpret_cast<uint64_t>(q.data);cache.source_qd.data[group]=reinterpret_cast<uint64_t>(qd.data);"
)
def _stamp_source(cache: KineticData, group: int, q: wp.array[float], qd: wp.array[float]): ...


@wp.func_native("float* s=reinterpret_cast<float*>(address);for(int k=0;k<7;++k)s[7*body+k]=pose[k];")
def _store_pose(address: wp.uint64, body: int, pose: wp.transform): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address);wp::transform p;for(int k=0;k<7;++k)p[k]=s[7*body+k];return p;"
)
def _load_pose(address: wp.uint64, body: int) -> wp.transform: ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+308;for(int k=0;k<6;++k)s[6*body+k]=v[k];")
def _store_motion(address: wp.uint64, body: int, v: wp.spatial_vector): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+308+264*acceleration;wp::spatial_vector v;for(int k=0;k<6;++k)v[k]=s[6*body+k];return v;"
)
def _load_motion(address: wp.uint64, body: int, acceleration: int) -> wp.spatial_vector: ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+1930+3*shift;for(int k=0;k<3;++k)s[k]=v[k];")
def _store_origin(address: wp.uint64, shift: int, v: wp.vec3): ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+1930+3*shift;return wp::vec3(s[0],s[1],s[2]);")
def _load_origin(address: wp.uint64, shift: int) -> wp.vec3: ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+836;for(int k=0;k<6;++k)s[6*dof+k]=v[k];")
def _store_axis(address: wp.uint64, dof: int, v: wp.spatial_vector): ...


_POSE = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int body=threadIdx.x;
    wp::transform value;
    if(body<44)for(int k=0;k<7;++k)value[k]=s[7*body+k];
    for(int round=0;round<4;++round) {
        const int parent=body<44?p.jump.data[round*44+body]:-1;
        wp::transform ancestor;
        if(parent>=0)for(int k=0;k<7;++k)ancestor[k]=s[7*parent+k];
        __syncthreads();
        if(parent>=0)value=wp::transform_multiply(ancestor,value);
        if(body<44)for(int k=0;k<7;++k)s[7*body+k]=value[k];
        __syncthreads();
    }
#else
    wp::transform old[44];
    for(int round=0;round<4;++round) {
        for(int body=0;body<44;++body)for(int k=0;k<7;++k)old[body][k]=s[7*body+k];
        for(int body=0;body<44;++body) {
            const int parent=p.jump.data[round*44+body];
            const wp::transform v=parent>=0?wp::transform_multiply(old[parent],old[body]):old[body];
            for(int k=0;k<7;++k)s[7*body+k]=v[k];
        }
    }
#endif
"""


@wp.func_native(_POSE)
def _scan_poses(address: wp.uint64, p: KineticPlan): ...


_MOTION = r"""
    float* s=reinterpret_cast<float*>(address)+308;
#if defined(__CUDA_ARCH__)
    const int body=threadIdx.x;
    float v[6]={},a[6]={};
    if(body<44)for(int k=0;k<6;++k){v[k]=s[6*body+k];s[264+6*body+k]=0.0f;}
    __syncthreads();
    for(int round=0;round<4;++round) {
        const int parent=body<44?p.jump.data[round*44+body]:-1;
        float pv[6],pa[6];
        if(parent>=0)for(int k=0;k<6;++k){pv[k]=s[6*parent+k];pa[k]=s[264+6*parent+k];}
        __syncthreads();
        if(parent>=0) {
            const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
            const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
            const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
            for(int k=0;k<3;++k){a[k]=pa[k]+a[k]+c[k];a[k+3]=pa[k+3]+a[k+3]+d[k];}
            for(int k=0;k<6;++k)v[k]=pv[k]+v[k];
        }
        if(body<44)for(int k=0;k<6;++k){s[6*body+k]=v[k];s[264+6*body+k]=a[k];}
        __syncthreads();
    }
#else
    float ov[44][6],oa[44][6];
    for(int body=0;body<44;++body)for(int k=0;k<6;++k)s[264+6*body+k]=0.0f;
    for(int round=0;round<4;++round) {
        for(int body=0;body<44;++body)for(int k=0;k<6;++k){ov[body][k]=s[6*body+k];oa[body][k]=s[264+6*body+k];}
        for(int body=0;body<44;++body) {
            const int parent=p.jump.data[round*44+body];
            if(parent<0)continue;
            const float* pv=ov[parent],*pa=oa[parent],*v=ov[body];
            const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
            const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
            const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
            for(int k=0;k<3;++k){s[264+6*body+k]=pa[k]+oa[body][k]+c[k];s[267+6*body+k]=pa[k+3]+oa[body][k+3]+d[k];}
            for(int k=0;k<6;++k)s[6*body+k]=pv[k]+v[k];
        }
    }
#endif
"""


@wp.func_native(_MOTION)
def _scan_motion(address: wp.uint64, p: KineticPlan): ...


@wp.func_native(r"""
    float* s=reinterpret_cast<float*>(address);
    for(int k=0;k<6;++k)s[1094+6*body+k]=wrench[k];
    if(refresh) {
        float* t=s+1358+13*body;
        t[0]=mass;
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
def _body_terms(
    address: wp.uint64,
    body: int,
    local: int,
    group: int,
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
    mass = data.body_mass[body]
    inertia_local = data.body_inertia[body]
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
    cache.com_offset[group, local] = radius
    _store_terms(address, local, mass, radius, inertia, wp.spatial_vector(force, torque), refresh)


_COLLECT = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x,stride=blockDim.x;
#else
    const int lane=0,stride=1;
#endif
    for(int level=9;level>=0;--level) {
        for(int body=lane;body<44;body+=stride)if(plan.depth.data[body]==level) {
            for(int pos=plan.children_offsets.data[body];pos<plan.children_offsets.data[body+1];++pos) {
                const int child=plan.children.data[pos];
                for(int k=0;k<6;++k)s[1094+6*body+k]+=s[1094+6*child+k];
                if(refresh)for(int k=0;k<13;++k)s[1358+13*body+k]+=s[1358+13*child+k];
            }
        }
#if defined(__CUDA_ARCH__)
        __syncthreads();
#endif
    }
    for(int dof=lane;dof<43;dof+=stride) {
        const int body=p.dof_joint.data[dof];
        float bias=0.0f;
        for(int k=0;k<6;++k)bias+=s[836+6*dof+k]*s[1094+6*body+k];
        cache.bias.data[group*43+dof]=bias;
        if(!wp::isfinite(bias)) {
#if defined(__CUDA_ARCH__)
            atomicOr(&cache.status.data[group],1);
#else
            cache.status.data[group]|=1;
#endif
        }
        if(refresh) {
            const float* t=s+1358+13*body,*axis=s+836+6*dof;
            const wp::vec3 l(axis[0],axis[1],axis[2]),w(axis[3],axis[4],axis[5]),h(t[1],t[2],t[3]);
            const wp::vec3 linear=t[0]*l+wp::cross(w,h);
            wp::vec3 angular=wp::cross(h,l);
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)angular[i]+=t[4+3*i+j]*w[j];
            for(int k=0;k<3;++k){s[6*dof+k]=linear[k];s[6*dof+3+k]=angular[k];}
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if(refresh)for(int e=lane;e<434;e+=stride) {
        const int row=42-p.row.data[e],col=42-p.col.data[e],src=p.source.data[e];
        const int projection=src==col?row:col;
        float value=0.0f;
        for(int k=0;k<6;++k)value+=s[836+6*projection+k]*s[6*src+k];
        cache.geometric.data[group*434+e]=value;
        if(!wp::isfinite(value)) {
#if defined(__CUDA_ARCH__)
            atomicOr(&cache.status.data[group],2);
#else
            cache.status.data[group]|=2;
#endif
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if(lane==0) {
        cache.generation.data[group]+=1;
        cache.current_valid.data[p.group_to_art.data[group]]=cache.status.data[group]==0;
        cache.geometry_valid.data[group]=refresh&&cache.status.data[group]==0;
        if(refresh)cache.geometry_generation.data[group]=cache.generation.data[group];
    }
"""


@wp.func_native(_COLLECT)
def _collect(
    address: wp.uint64, group: int, p: sparse_factor.SparsePlan, plan: KineticPlan, cache: KineticData, refresh: int
): ...


@functools.cache
def get_state_kernel(finish: bool, chain_scan: bool = False):
    """Build complete repair/finish using original generalized integration laws."""
    plan_type = KineticPlan
    scan_poses, scan_motion, collect = _scan_poses, _scan_motion, _collect
    if chain_scan:
        from . import g1_chain_scan  # noqa: PLC0415

        plan_type = g1_chain_scan.ChainPlan
        scan_poses, scan_motion = g1_chain_scan.scan_poses, g1_chain_scan.scan_motion
        collect = g1_chain_scan.get_collect(_COLLECT)
    source = "\n".join(
        kuka_joint_world.operation_source(name)
        for name in ("update_qdd_from_velocity", "remove_free_root_transport_from_qdd", "integrate_generalized_joints")
    )
    filename = f"<g1-kinetic-generalized-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(kernels.__dict__)
    exec(compile(source, filename, "exec"), namespace)
    update_qdd = namespace["_light_update_qdd_from_velocity"]
    remove_transport = namespace["_light_remove_free_root_transport_from_qdd"]
    integrate = namespace["_light_integrate_generalized_joints"]

    def state(
        p: sparse_factor.SparsePlan,
        plan: plan_type,
        data: PublicationData,
        cache: KineticData,
        requests: wp.array[int],
        global_refresh: int,
    ):
        group, logical_lane = wp.tid()
        lanes = _lanes()
        lane, stride = lanes[0], lanes[1]
        if stride == 1 and logical_lane != 0:
            return
        art = p.group_to_art[group]
        ds, js = p.art_dof_start[art], p.art_joint_start[art]
        refresh = int(global_refresh != 0)
        if not wp.static(finish):
            refresh = int(refresh != 0 or requests[0] != 0)
            if (
                cache.current_valid[art] != 0
                and _same_source(cache, group, data.joint_q, data.joint_qd) != 0
                and (
                    refresh == 0
                    or (
                        cache.geometry_valid[group] != 0 and cache.geometry_generation[group] == cache.generation[group]
                    )
                )
            ):
                return
        if lane == 0:
            cache.current_valid[art] = 0
        address = _storage()
        q = data.joint_q
        qd = data.joint_qd
        if wp.static(finish):
            for local in range(lane, 43, stride):
                update_qdd(
                    ds + local, data.joint_qd, data.kinematic_dof_mask, 1.0 / data.dt, data.v_out, data.joint_qdd
                )
            _sync()
            if lane == 0:
                remove_transport(
                    plan.root_slot[group],
                    data.free_root_joint_indices,
                    data.joint_qd_start,
                    data.kinematic_joint_mask,
                    data.joint_qd,
                    data.joint_qdd,
                )
            _sync()
            for local in range(lane, 44, stride):
                integrate(
                    js + local,
                    data.joint_type,
                    data.joint_parent,
                    data.joint_child,
                    data.joint_q_start,
                    data.joint_qd_start,
                    data.kinematic_joint_mask,
                    data.joint_dof_dim,
                    data.body_com,
                    data.joint_X_c,
                    data.joint_q,
                    data.joint_qd,
                    data.joint_qdd,
                    data.dt,
                    data.angular_damping,
                    data.joint_q_new,
                    data.joint_qd_new,
                )
            _sync()
            q = data.joint_q_new
            qd = data.joint_qd_new
        for local in range(lane, 44, stride):
            joint = js + local
            transform = kernels.jcalc_transform(
                data.joint_type[joint],
                data.joint_axis,
                data.joint_qd_start[joint],
                data.joint_dof_dim[joint, 0],
                data.joint_dof_dim[joint, 1],
                q,
                data.joint_q_start[joint],
            )
            relative = data.joint_X_p[joint] * transform * wp.transform_inverse(data.joint_X_c[joint])
            if local == 0:
                _store_origin(address, 1, wp.transform_get_translation(relative))
                relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
            _store_pose(address, local, relative)
        _sync()
        scan_poses(address, plan)
        if lane == 0:
            root_body = data.joint_child[js]
            origin = wp.transform_point(_load_pose(address, 0), data.body_com[root_body])
            _store_origin(address, 0, origin)
            data.articulation_origin[art] = origin + _load_origin(address, 1)
        _sync()
        origin = _load_origin(address, 0)
        shift = _load_origin(address, 1)
        for local in range(lane, 44, stride):
            joint = js + local
            parent = plan.parent[local]
            anchor = data.joint_X_p[joint]
            if parent >= 0:
                anchor = _load_pose(address, parent) * anchor
            else:
                anchor = wp.transform(wp.transform_get_translation(anchor) - shift, wp.transform_get_rotation(anchor))
            anchor = wp.transform(wp.transform_get_translation(anchor) - origin, wp.transform_get_rotation(anchor))
            velocity = kernels.jcalc_motion(
                data.joint_type[joint],
                data.joint_axis,
                data.joint_dof_dim[joint, 0],
                data.joint_dof_dim[joint, 1],
                anchor,
                qd,
                data.joint_qd_start[joint],
                data.joint_S_s,
            )
            start = data.joint_qd_start[joint]
            count = data.joint_dof_dim[joint, 0] + data.joint_dof_dim[joint, 1]
            for k in range(count):
                _store_axis(address, start - ds + k, data.joint_S_s[start + k])
            _store_motion(address, local, velocity)
        _sync()
        scan_motion(address, plan)
        for local in range(lane, 44, stride):
            body = data.joint_child[js + local]
            pose = _load_pose(address, local)
            data.body_q[body] = wp.transform(
                wp.transform_get_translation(pose) + shift, wp.transform_get_rotation(pose)
            )
            _body_terms(
                address,
                body,
                local,
                group,
                pose,
                pose * data.body_X_com[body],
                origin,
                _load_motion(address, local, 0),
                _load_motion(address, local, 1),
                data,
                cache,
                refresh,
            )
        _sync()
        collect(address, group, p, plan, cache, refresh)
        if lane == 0:
            _stamp_source(cache, group, q, qd)
        _release(address)

    state.__name__ = state.__qualname__ = "g1_kinetic_finish44" if finish else "g1_kinetic_repair44"
    if chain_scan:
        state.__name__ = state.__qualname__ = state.__name__ + "_chain"
    return wp.kernel(module="unique", enable_backward=False)(state)


_PREDICT = r"""
    const int art=p.group_to_art.data[group],world=p.art_to_world.data[art];
    const int ds=p.art_dof_start.data[art],js=p.art_joint_start.data[art];
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x,stride=blockDim.x;
    __shared__ float s[350];
    __shared__ int nonzero;
#else
    const int lane=0,stride=1;
    float s[350];int nonzero=0;
#endif
    if(lane==0)nonzero=0;
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    for(int local=lane;local<44;local+=stride) {
        const int body=p.joint_child.data[js+local];
        const auto input=f.body_f.data[body];
        const wp::vec3 force(input[0],input[1],input[2]);
        const wp::vec3 torque=wp::vec3(input[3],input[4],input[5])+wp::cross(cache.com_offset.data[group*44+local],force);
        for(int k=0;k<3;++k){s[6*local+k]=force[k];s[6*local+3+k]=torque[k];}
        bool active=false;for(int k=0;k<6;++k)active|=input[k]!=0.0f;
        if(active) {
#if defined(__CUDA_ARCH__)
            atomicOr(&nonzero,1);
#else
            nonzero=1;
#endif
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if(nonzero)for(int level=9;level>=0;--level) {
        for(int local=lane;local<44;local+=stride)if(plan.depth.data[local]==level) {
            for(int pos=plan.children_offsets.data[local];pos<plan.children_offsets.data[local+1];++pos) {
                const int child=plan.children.data[pos];
                for(int k=0;k<6;++k)s[6*local+k]+=s[6*child+k];
            }
        }
#if defined(__CUDA_ARCH__)
        __syncthreads();
#endif
    }
    for(int i=lane;i<43;i+=stride) {
        const int dof=ds+i,body=p.dof_joint.data[i];
        float value=-cache.bias.data[group*43+i]+f.joint_f.data[dof]+f.u0.data[dof];
        if(nonzero)for(int k=0;k<6;++k)value+=f.S.data[dof][k]*s[6*body+k];
        if(i>=6)value+=f.stiffness.data[dof]*(f.reference.data[dof]-f.joint_q.data[plan.q_index.data[group*43+i]])-f.damping.data[dof]*f.joint_qd.data[dof];
        s[264+i]=value;
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    for(int row=lane;row<43;row+=stride) {
        float value=0.0f;
        for(int col=0;col<=row;++col) {
            const int entry=p.index.data[row*43+col];
            if(entry>=0)value+=d.W.data[group*434+entry]*s[264+42-col];
        }
        s[307+row]=value;
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    for(int col=lane;col<43;col+=stride) {
        float value=0.0f;
        for(int k=0;k<p.inverse_count.data[col];++k) {
            const int row=p.inverse_nodes.data[col*18+k];
            value+=d.W.data[group*434+p.index.data[row*43+col]]*s[307+row];
        }
        const int dof=ds+42-col;
        const bool valid=cache.current_valid.data[art]&&d.valid.data[world]&&!d.status.data[world]&&!cache.status.data[group]&&wp::isfinite(value);
        f.qdd.data[dof]=valid?value:NAN;
        f.vhat.data[dof]=f.predictor_qd.data[dof]+f.dt*f.qdd.data[dof];
        if(!valid) {
#if defined(__CUDA_ARCH__)
            atomicOr(&cache.status.data[group],4);
#else
            cache.status.data[group]|=4;
#endif
        }
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    if(lane==0) {
        const wp::vec3 v(f.predictor_qd.data[ds],f.predictor_qd.data[ds+1],f.predictor_qd.data[ds+2]);
        const wp::vec3 w(f.predictor_qd.data[ds+3],f.predictor_qd.data[ds+4],f.predictor_qd.data[ds+5]);
        const wp::vec3 transport=f.dt*wp::cross(w,v);
        for(int k=0;k<3;++k)f.vhat.data[ds+k]+=transport[k];
    }
"""


@functools.cache
def get_predictor_kernel(chain_scan: bool = False):
    """Apply current external/control force and both unchanged W actions."""
    plan_type, source = KineticPlan, _PREDICT
    if chain_scan:
        from . import g1_chain_scan  # noqa: PLC0415

        plan_type = g1_chain_scan.ChainPlan
        source = g1_chain_scan.predictor_source(source)

    @wp.func_native(source)
    def native(
        group: int,
        p: sparse_factor.SparsePlan,
        d: sparse_factor.SparseData,
        plan: plan_type,
        cache: KineticData,
        f: ForceInput,
    ): ...

    def predict(
        p: sparse_factor.SparsePlan, d: sparse_factor.SparseData, plan: plan_type, cache: KineticData, f: ForceInput
    ):
        group, _ = wp.tid()
        native(group, p, d, plan, cache, f)

    predict.__name__ = predict.__qualname__ = "g1_kinetic_predict43"
    if chain_scan:
        predict.__name__ = predict.__qualname__ = predict.__name__ + "_chain"
    return wp.kernel(module="unique", enable_backward=False)(predict)


from .g1_kinetic_owner import G1KineticState, supported  # noqa: E402,F401 -- preserve the single private owner API

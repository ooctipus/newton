# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers

# SPDX-License-Identifier: Apache-2.0

"""Generated current/next tree moments, bias and fused state publication."""

import functools
import hashlib
import linecache
from types import SimpleNamespace

import warp as wp

from . import kernels, kuka_joint_world, sparse_factor, tree_chain_scan
from .tree_kinetic_layout import G1_LAYOUT, TreeLayout
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


_POSE = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int body=threadIdx.x;
    wp::transform value;
    if(body<$BODIES)for(int k=0;k<7;++k)value[k]=s[7*body+k];
    for(int round=0;round<$ROUNDS;++round) {
        const int parent=body<$BODIES?p.jump.data[round*$BODIES+body]:-1;
        wp::transform ancestor;
        if(parent>=0)for(int k=0;k<7;++k)ancestor[k]=s[7*parent+k];
        __syncthreads();
        if(parent>=0)value=wp::transform_multiply(ancestor,value);
        if(body<$BODIES)for(int k=0;k<7;++k)s[7*body+k]=value[k];
        __syncthreads();
    }
#else
    wp::transform old[$BODIES];
    for(int round=0;round<$ROUNDS;++round) {
        for(int body=0;body<$BODIES;++body)for(int k=0;k<7;++k)old[body][k]=s[7*body+k];
        for(int body=0;body<$BODIES;++body) {
            const int parent=p.jump.data[round*$BODIES+body];
            const wp::transform v=parent>=0?wp::transform_multiply(old[parent],old[body]):old[body];
            for(int k=0;k<7;++k)s[7*body+k]=v[k];
        }
    }
#endif
"""

_MOTION = r"""
    float* s=reinterpret_cast<float*>(address)+$MOTION;
#if defined(__CUDA_ARCH__)
    const int body=threadIdx.x;
    float v[6]={},a[6]={};
    if(body<$BODIES)for(int k=0;k<6;++k){v[k]=s[6*body+k];s[$SIX_BODIES+6*body+k]=0.0f;}
    __syncthreads();
    for(int round=0;round<$ROUNDS;++round) {
        const int parent=body<$BODIES?p.jump.data[round*$BODIES+body]:-1;
        float pv[6],pa[6];
        if(parent>=0)for(int k=0;k<6;++k){pv[k]=s[6*parent+k];pa[k]=s[$SIX_BODIES+6*parent+k];}
        __syncthreads();
        if(parent>=0) {
            const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
            const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
            const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
            for(int k=0;k<3;++k){a[k]=pa[k]+a[k]+c[k];a[k+3]=pa[k+3]+a[k+3]+d[k];}
            for(int k=0;k<6;++k)v[k]=pv[k]+v[k];
        }
        if(body<$BODIES)for(int k=0;k<6;++k){s[6*body+k]=v[k];s[$SIX_BODIES+6*body+k]=a[k];}
        __syncthreads();
    }
#else
    float ov[$BODIES][6],oa[$BODIES][6];
    for(int body=0;body<$BODIES;++body)for(int k=0;k<6;++k)s[$SIX_BODIES+6*body+k]=0.0f;
    for(int round=0;round<$ROUNDS;++round) {
        for(int body=0;body<$BODIES;++body)for(int k=0;k<6;++k){ov[body][k]=s[6*body+k];oa[body][k]=s[$SIX_BODIES+6*body+k];}
        for(int body=0;body<$BODIES;++body) {
            const int parent=p.jump.data[round*$BODIES+body];
            if(parent<0)continue;
            const float* pv=ov[parent],*pa=oa[parent],*v=ov[body];
            const wp::vec3 pl(pv[0],pv[1],pv[2]),pw(pv[3],pv[4],pv[5]);
            const wp::vec3 vl(v[0],v[1],v[2]),vw(v[3],v[4],v[5]);
            const wp::vec3 c=wp::cross(pw,vl)+wp::cross(pl,vw),d=wp::cross(pw,vw);
            for(int k=0;k<3;++k){s[$SIX_BODIES+6*body+k]=pa[k]+oa[body][k]+c[k];s[$SIX_BODIES_PLUS3+6*body+k]=pa[k+3]+oa[body][k+3]+d[k];}
            for(int k=0;k<6;++k)s[6*body+k]=pv[k]+v[k];
        }
    }
#endif
"""

_COLLECT = r"""    $PHYSICAL_DECL

    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x,stride=blockDim.x;
#else
    const int lane=0,stride=1;
#endif
$REDUCE
    for(int dof=lane;dof<$DOFS;dof+=stride) {
        const int body=p.dof_joint.data[dof];
        float bias=0.0f;
        for(int k=0;k<6;++k)bias+=s[$AXIS+6*dof+k]*s[$WRENCH+6*body+k];
        cache.bias.data[group*$DOFS+dof]=bias;
        if(!wp::isfinite(bias)) {
#if defined(__CUDA_ARCH__)
            atomicOr(&cache.status.data[group],1);
#else
            cache.status.data[group]|=1;
#endif
        }
        if(refresh) {
            const float* t=s+$MOMENT+13*body,*axis=s+$AXIS+6*dof;
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
    if(refresh)for(int e=lane;e<$NNZ;e+=stride) {
        const int row=$PHYSICAL_ROW,col=$PHYSICAL_COL,src=p.source.data[e];
        const int projection=src==col?row:col;
        float value=0.0f;
        for(int k=0;k<6;++k)value+=s[$AXIS+6*projection+k]*s[6*src+k];
        cache.geometric.data[group*$NNZ+e]=value;
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

_PREDICT = r"""    $PHYSICAL_DECL

    const int art=p.group_to_art.data[group],world=p.art_to_world.data[art];
    const int ds=p.art_dof_start.data[art],js=p.art_joint_start.data[art];
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x,stride=blockDim.x;
    __shared__ float s[$PRED_SCRATCH];
    __shared__ int nonzero;
#else
    const int lane=0,stride=1;
    float s[$PRED_SCRATCH];int nonzero=0;
#endif
    if(lane==0)nonzero=0;
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    for(int local=lane;local<$BODIES;local+=stride) {
        const int body=p.joint_child.data[js+local];
        const auto input=f.body_f.data[body];
        const wp::vec3 force(input[0],input[1],input[2]);
        const wp::vec3 torque=wp::vec3(input[3],input[4],input[5])+wp::cross(cache.com_offset.data[group*$BODIES+local],force);
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
$REDUCE
    for(int i=lane;i<$DOFS;i+=stride) {
        const int dof=ds+i,body=p.dof_joint.data[i];
        float value=-cache.bias.data[group*$DOFS+i]+f.joint_f.data[dof]+f.u0.data[dof];
        if(nonzero)for(int k=0;k<6;++k)value+=f.S.data[dof][k]*s[6*body+k];
        if(i>=$ROOT_DOFS)value+=f.stiffness.data[dof]*(f.reference.data[dof]-f.joint_q.data[plan.q_index.data[group*$DOFS+i]])-f.damping.data[dof]*f.joint_qd.data[dof];
        s[$SIX_BODIES+i]=value;
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    for(int row=lane;row<$DOFS;row+=stride) {
        float value=0.0f;
        for(int col=0;col<=row;++col) {
            const int entry=p.index.data[row*$DOFS+col];
            if(entry>=0)value+=d.W.data[group*$NNZ+entry]*s[$SIX_BODIES+$PHYSICAL_INDEX];
        }
        s[$WHITE+row]=value;
    }
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
    for(int col=lane;col<$DOFS;col+=stride) {
        float value=0.0f;
        for(int k=0;k<p.inverse_count.data[col];++k) {
            const int row=p.inverse_nodes.data[col*$INVERSE_WIDTH+k];
            value+=d.W.data[group*$NNZ+p.index.data[row*$DOFS+col]]*s[$WHITE+row];
        }
        const int dof=ds+$PHYSICAL_INDEX;
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
    if($FREE_ROOT&&lane==0) {
        const wp::vec3 v(f.predictor_qd.data[ds],f.predictor_qd.data[ds+1],f.predictor_qd.data[ds+2]);
        const wp::vec3 w(f.predictor_qd.data[ds+3],f.predictor_qd.data[ds+4],f.predictor_qd.data[ds+5]);
        const wp::vec3 transport=f.dt*wp::cross(w,v);
        for(int k=0;k<3;++k)f.vhat.data[ds+k]+=transport[k];
    }
"""

_LEVEL_REDUCE = r"""    for(int level=$LAST_PARENT;level>=0;--level) {
        for(int body=lane;body<$BODIES;body+=stride)if(plan.depth.data[body]==level) {
            for(int pos=plan.children_offsets.data[body];pos<plan.children_offsets.data[body+1];++pos) {
                const int child=plan.children.data[pos];
                for(int k=0;k<6;++k)s[$WRENCH+6*body+k]+=s[$WRENCH+6*child+k];
                if(refresh)for(int k=0;k<13;++k)s[$MOMENT+13*body+k]+=s[$MOMENT+13*child+k];
            }
        }
#if defined(__CUDA_ARCH__)
        __syncthreads();
#endif
    }
"""

_EXTERNAL_REDUCE = r"""    if(nonzero)for(int level=$LAST_PARENT;level>=0;--level) {
        for(int local=lane;local<$BODIES;local+=stride)if(plan.depth.data[local]==level) {
            for(int pos=plan.children_offsets.data[local];pos<plan.children_offsets.data[local+1];++pos) {
                const int child=plan.children.data[pos];
                for(int k=0;k<6;++k)s[6*local+k]+=s[6*child+k];
            }
        }
#if defined(__CUDA_ARCH__)
        __syncthreads();
#endif
    }
"""


_STORAGE_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[$TOTAL];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc($TOTAL*sizeof(float)));
#endif
"""

_STORE_TERMS_SOURCE = r"""
    float* s=reinterpret_cast<float*>(address);
    for(int k=0;k<6;++k)s[$WRENCH+6*body+k]=wrench[k];
    if(refresh) {
        float* t=s+$MOMENT+13*body;
        t[0]=mass;
        for(int k=0;k<3;++k)t[1+k]=mass*radius[k];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            t[4+3*i+j]=inertia.data[i][j]+mass*((i==j?wp::dot(radius,radius):0.0f)-radius[i]*radius[j]);
    }
"""


@functools.cache
def get_helpers(layout: TreeLayout = G1_LAYOUT, chain_scan: bool = False, body_velocity: bool = False):
    """Bind one physical tree evaluation to its storage and traversal schedule."""
    plan_type = tree_chain_scan.ChainPlan if chain_scan else KineticPlan
    pose_source = layout.render(tree_chain_scan._POSE if chain_scan else _POSE)
    motion_source = layout.render(tree_chain_scan._MOTION if chain_scan else _MOTION)
    reduction = (
        tree_chain_scan.reduction_source(moments=True, layout=layout) if chain_scan else layout.render(_LEVEL_REDUCE)
    )
    collect_source = layout.render(_COLLECT.replace("$REDUCE", reduction))

    @wp.func_native(layout.render(_STORAGE_SOURCE))
    def _storage() -> wp.uint64: ...

    @wp.func_native(
        layout.render("float* s=reinterpret_cast<float*>(address)+$MOTION;for(int k=0;k<6;++k)s[6*body+k]=v[k];")
    )
    def _store_motion(address: wp.uint64, body: int, v: wp.spatial_vector): ...

    @wp.func_native(
        layout.render(
            "float* s=reinterpret_cast<float*>(address)+$MOTION+$SIX_BODIES*acceleration;wp::spatial_vector v;for(int k=0;k<6;++k)v[k]=s[6*body+k];return v;"
        )
    )
    def _load_motion(address: wp.uint64, body: int, acceleration: int) -> wp.spatial_vector: ...

    @wp.func_native(
        layout.render("float* s=reinterpret_cast<float*>(address)+$ORIGIN+3*shift;for(int k=0;k<3;++k)s[k]=v[k];")
    )
    def _store_origin(address: wp.uint64, shift: int, v: wp.vec3): ...

    @wp.func_native(
        layout.render("float* s=reinterpret_cast<float*>(address)+$ORIGIN+3*shift;return wp::vec3(s[0],s[1],s[2]);")
    )
    def _load_origin(address: wp.uint64, shift: int) -> wp.vec3: ...

    @wp.func_native(
        layout.render("float* s=reinterpret_cast<float*>(address)+$AXIS;for(int k=0;k<6;++k)s[6*dof+k]=v[k];")
    )
    def _store_axis(address: wp.uint64, dof: int, v: wp.spatial_vector): ...

    @wp.func_native(pose_source)
    def _scan_poses(address: wp.uint64, p: plan_type): ...

    @wp.func_native(motion_source)
    def _scan_motion(address: wp.uint64, p: plan_type): ...

    @wp.func_native(layout.render(_STORE_TERMS_SOURCE))
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
        force = mass * (
            a + wp.cross(alpha, radius) + wp.cross(omega, linear + wp.cross(omega, radius)) - data.gravity[0]
        )
        torque = acceleration_force + wp.cross(omega, momentum) + wp.cross(radius, force)
        public_radius = wp.transform_point(pose, data.body_com[body]) - origin
        data.body_qd[body] = wp.spatial_vector(linear + wp.cross(omega, public_radius), omega)
        if wp.static(body_velocity):
            data.body_v_s[body] = velocity
        cache.com_offset[group, local] = radius
        _store_terms(address, local, mass, radius, inertia, wp.spatial_vector(force, torque), refresh)

    @wp.func_native(collect_source)
    def _collect(
        address: wp.uint64, group: int, p: sparse_factor.SparsePlan, plan: plan_type, cache: KineticData, refresh: int
    ): ...

    return SimpleNamespace(
        plan_type=plan_type,
        _body_terms=_body_terms,
        _collect=_collect,
        _load_motion=_load_motion,
        _load_origin=_load_origin,
        _scan_motion=_scan_motion,
        _scan_poses=_scan_poses,
        _storage=_storage,
        _store_axis=_store_axis,
        _store_motion=_store_motion,
        _store_origin=_store_origin,
        _store_terms=_store_terms,
    )


@functools.cache
def get_state_kernel(
    finish: bool, chain_scan: bool = False, *, layout: TreeLayout = G1_LAYOUT, body_velocity: bool = False
):
    """Build complete repair/finish using original generalized integration laws."""
    h = get_helpers(layout, chain_scan, body_velocity)
    plan_type = h.plan_type
    scan_poses, scan_motion, collect = h._scan_poses, h._scan_motion, h._collect
    _storage, _store_motion, _load_motion = h._storage, h._store_motion, h._load_motion
    _store_origin, _load_origin, _store_axis = h._store_origin, h._load_origin, h._store_axis
    _body_terms = h._body_terms
    bodies, dofs, free_root = layout.bodies, layout.dofs, layout.root_dofs == 6
    source = "\n".join(
        kuka_joint_world.operation_source(name)
        for name in ("update_qdd_from_velocity", "remove_free_root_transport_from_qdd", "integrate_generalized_joints")
    )
    filename = f"<tree-kinetic-generalized-{hashlib.sha256(source.encode()).hexdigest()}>"
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
            refresh = int(refresh != 0 or requests[art] != 0)
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
            for local in range(lane, dofs, stride):
                update_qdd(
                    ds + local, data.joint_qd, data.kinematic_dof_mask, 1.0 / data.dt, data.v_out, data.joint_qdd
                )
            _sync()
            if wp.static(free_root) and lane == 0:
                remove_transport(
                    plan.root_slot[group],
                    data.free_root_joint_indices,
                    data.joint_qd_start,
                    data.kinematic_joint_mask,
                    data.joint_qd,
                    data.joint_qdd,
                )
            _sync()
            for local in range(lane, bodies, stride):
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
        for local in range(lane, bodies, stride):
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
        for local in range(lane, bodies, stride):
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
        for local in range(lane, bodies, stride):
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

    state.__name__ = state.__qualname__ = f"tree_kinetic_finish{bodies}" if finish else f"tree_kinetic_repair{bodies}"
    if chain_scan:
        state.__name__ = state.__qualname__ = state.__name__ + "_chain"
    return wp.kernel(module="unique", enable_backward=False)(state)


@functools.cache
def get_predictor_kernel(chain_scan: bool = False, *, layout: TreeLayout = G1_LAYOUT):
    """Apply current external/control force and both unchanged W actions."""
    plan_type = tree_chain_scan.ChainPlan if chain_scan else KineticPlan
    reduction = (
        tree_chain_scan.reduction_source(moments=False, layout=layout)
        if chain_scan
        else layout.render(_EXTERNAL_REDUCE)
    )
    if chain_scan:
        reduction = "if(nonzero) {\n" + reduction + "}\n"
    source = layout.render(_PREDICT.replace("$REDUCE", reduction))

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

    predict.__name__ = predict.__qualname__ = f"tree_kinetic_predict{layout.dofs}"
    if chain_scan:
        predict.__name__ = predict.__qualname__ = predict.__name__ + "_chain"
    return wp.kernel(module="unique", enable_backward=False)(predict)

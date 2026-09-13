# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact current/next Kuka kinetic producer, with complete public state finish.

The primary spatial cache is private and transient. Joint bias and sparse180
geometric mass are produced by exact subtree moment reductions, not body-matrix
materialization. The two free roots retain the original canonical body service.
"""

# Warp uses explicit scalar constructors for values mutated inside device loops.

import functools
import hashlib
import linecache

import warp as wp

from . import kernels, kuka_joint_world
from . import world_scan_publication as previous
from .kinetic_source import checked_definitions
from .kinetic_types import (
    CurrentKineticCache,
    GeometricCache,
    KineticPlan,
    KineticSchedule,
)

PublicationData = previous.PublicationData


def checked_source():
    """Check only the retained publication/generalized source definitions."""
    checked_definitions(
        "world_scan_publication.py",
        (
            "PublicationData",
            "_POSE_SCAN",
            "_MOTION_SCAN",
            "_release",
            "_sync",
            "_lane",
            "_store_pose",
            "_load_pose",
            "_store_motion",
            "_load_motion",
            "_store_origin",
            "_load_origin",
        ),
    )
    checked_definitions("kuka_joint_world.py", ("operation_source",))
    checked_definitions(
        "kernels.py",
        (
            "update_qdd_from_velocity",
            "remove_free_root_transport_from_qdd",
            "integrate_generalized_joints",
            "jcalc_transform",
            "jcalc_motion",
            "finalize_body_dynamics_body",
        ),
    )


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
#else
    const int lane=0,stride=1;
#endif
    bool bad=!(data.dt>0.0f)||!wp::isfinite(data.dt)
        ||schedule.generation.data[world]<0
        ||(schedule.geometry_requested.data[world]!=0&&schedule.geometry_requested.data[world]!=1);
    for(int local=lane;local<35;local+=stride) {
        const int dof=plan.dof_ids.data[world*35+local];
        bad|=!wp::isfinite(data.joint_qd.data[dof]);
        if(finish)bad|=!wp::isfinite(data.v_out.data[dof]);
        if(local<23)bad|=data.kinematic_dof_mask.data[dof]!=0;
    }
    for(int local=lane;local<37;local+=stride) {
        int qid=0;
        if(local<23)qid=plan.q_index.data[world*23+local];
        else {
            const int root=(local-23)/7;
            const int joint=plan.joint_ids.data[world*32+30+root];
            qid=data.joint_q_start.data[joint]+(local-23)%7;
        }
        bad|=!wp::isfinite(data.joint_q.data[qid]);
    }
#if defined(__CUDA_ARCH__)
    bad=__ballot_sync(0xffffffffu,bad)!=0;
#endif
    if(lane==0) {
        current.valid.data[world]=0;
        schedule.status.data[world]=bad?1:0;
        if(schedule.geometry_requested.data[world]!=0)geometric.valid.data[world]=0;
    }
    return bad?0:1;
""")
def _admit(
    world: int,
    finish: int,
    plan: KineticPlan,
    data: PublicationData,
    schedule: KineticSchedule,
    current: CurrentKineticCache,
    geometric: GeometricCache,
) -> int: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[1320];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(1320*sizeof(float)));
#endif
""")
def _storage() -> wp.uint64: ...


_release = previous._release
_sync = previous._sync
_lane = previous._lane
_store_pose = previous._store_pose
_load_pose = previous._load_pose
_store_motion = previous._store_motion
_load_motion = previous._load_motion
_store_origin = previous._store_origin
_load_origin = previous._load_origin


@wp.func_native(previous._POSE_SCAN)
def _scan_poses(address: wp.uint64, world: int, plan: KineticPlan): ...


@wp.func_native(previous._MOTION_SCAN)
def _scan_motion(address: wp.uint64, world: int, plan: KineticPlan): ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+617+3*root;for(int k=0;k<3;++k)s[k]=shift[k];")
def _store_shift(address: wp.uint64, root: int, shift: wp.vec3): ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+617+3*root;return wp::vec3(s[0],s[1],s[2]);")
def _load_shift(address: wp.uint64, root: int) -> wp.vec3: ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+632+6*dof;for(int k=0;k<6;++k)s[k]=axis[k];")
def _store_axis(address: wp.uint64, dof: int, axis: wp.spatial_vector): ...


@wp.func_native(r"""
    float* s=reinterpret_cast<float*>(address);
    for(int k=0;k<3;++k)s[770+3*body+k]=radius[k];
    for(int k=0;k<6;++k)s[1130+6*body+k]=wrench[k];
    if(refresh)for(int i=0;i<3;++i)for(int j=0;j<3;++j)s[860+9*body+3*i+j]=inertia.data[i][j];
""")
def _store_terms(
    address: wp.uint64,
    body: int,
    radius: wp.vec3,
    inertia: wp.mat33,
    wrench: wp.spatial_vector,
    refresh: int,
): ...


@wp.func
def _inertia_action(rotation: wp.quat, inertia: wp.mat33, value: wp.vec3):
    return wp.quat_rotate(rotation, inertia * wp.quat_rotate_inv(rotation, value))


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
    schedule: KineticSchedule,
    current: CurrentKineticCache,
):
    radius = wp.transform_get_translation(pose_com) - origin
    rotation = wp.transform_get_rotation(pose_com)
    mass = data.body_mass[body]
    inertia_local = data.body_inertia[body]
    linear, omega = wp.spatial_top(velocity), wp.spatial_bottom(velocity)
    a, alpha = wp.spatial_top(acceleration), wp.spatial_bottom(acceleration)
    inertia_com = wp.mat33()
    angular_momentum = wp.vec3()
    angular_acceleration_force = wp.vec3()
    if schedule.geometry_requested[world] != 0:
        matrix = wp.quat_to_matrix(rotation)
        inertia_com = matrix * inertia_local * wp.transpose(matrix)
        angular_momentum = inertia_com * omega
        angular_acceleration_force = inertia_com * alpha
    else:
        angular_momentum = _inertia_action(rotation, inertia_local, omega)
        angular_acceleration_force = _inertia_action(rotation, inertia_local, alpha)
    velocity_com = linear + wp.cross(omega, radius)
    force = mass * (a + wp.cross(alpha, radius) + wp.cross(omega, velocity_com) - data.gravity[0])
    torque = angular_acceleration_force + wp.cross(omega, angular_momentum) + wp.cross(radius, force)
    wrench = wp.spatial_vector(force, torque)
    public_radius = wp.transform_point(pose, data.body_com[body]) - origin
    data.body_qd[body] = wp.spatial_vector(linear + wp.cross(omega, public_radius), omega)
    current.com_offset[world, local] = radius
    _store_terms(address, local, radius, inertia_com, wrench, schedule.geometry_requested[world])


# Exact group suffixes: the trunk owns all four branch totals; each branch owns
# only its own suffix. Never subtract a large outside prefix to obtain a leaf.
_SUM_CUDA = r"""
        float value=own;
        const int end=lane<10?10:(lane<30?10+5*((lane-10)/5+1):32);
        for(int offset=1;offset<16;offset*=2) {
            const float other=__shfl_down_sync(0xffffffffu,value,offset);
            if(lane+offset<end)value+=other;
        }
        const float f0=__shfl_sync(0xffffffffu,value,10);
        const float f1=__shfl_sync(0xffffffffu,value,15);
        const float f2=__shfl_sync(0xffffffffu,value,20);
        const float f3=__shfl_sync(0xffffffffu,value,25);
        if(lane<10)value+=((f0+f1)+f2)+f3;
        const int body=lane<7?lane+1:(lane<23?10+5*((lane-7)/4)+(lane-7)%4:0);
        const float subtree=__shfl_sync(0xffffffffu,value,body);
"""
_SUM_CPU = r"""
        float value[32],old[32];
        for(int local=0;local<32;++local)value[local]=own[local];
        for(int offset=1;offset<16;offset*=2) {
            for(int local=0;local<32;++local)old[local]=value[local];
            for(int local=0;local<32;++local) {
                const int end=local<10?10:(local<30?10+5*((local-10)/5+1):32);
                if(local+offset<end)value[local]+=old[local+offset];
            }
        }
        const float fingers=((value[10]+value[15])+value[20])+value[25];
        for(int local=0;local<10;++local)value[local]+=fingers;
"""


_COLLECT = r"""
    float* s=reinterpret_cast<float*>(address);
    bool invalid=false;
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31,stride=32;
    float bias=0.0f;
    for(int k=0;k<6;++k) {
        const float own=lane<30?s[1130+6*lane+k]:0.0f;
        CUDA_SUM
        if(lane<23)bias+=s[632+6*lane+k]*subtree;
    }
    if(lane<23){current.bias.data[world*23+lane]=bias;invalid|=!wp::isfinite(bias);}
#else
    const int lane=0,stride=1;
    float bias[23]={};
    for(int k=0;k<6;++k) {
        float own[32];for(int local=0;local<32;++local)own[local]=local<30?s[1130+6*local+k]:0.0f;
        CPU_SUM
        for(int d=0;d<23;++d) {
            const int body=d<7?d+1:10+5*((d-7)/4)+(d-7)%4;
            bias[d]+=s[632+6*d+k]*value[body];
        }
    }
    for(int d=0;d<23;++d){current.bias.data[world*23+d]=bias[d];invalid|=!wp::isfinite(bias[d]);}
#endif
    for(int local=lane;local<32;local+=stride) {
        const wp::vec3 r=current.com_offset.data[world*32+local];
        for(int k=0;k<3;++k)invalid|=!wp::isfinite(r[k]);
        if(local<23)for(int k=0;k<6;++k)invalid|=!wp::isfinite(current.axes.data[world*23+local][k]);
        if(local<3)for(int k=0;k<3;++k)invalid|=!wp::isfinite(current.origin.data[world*3+local][k]);
        if(local==30) {
            for(int k=0;k<6;++k)invalid|=!wp::isfinite(current.free_bias.data[world][k]);
            for(int d=0;d<6;++d)for(int k=0;k<6;++k)invalid|=!wp::isfinite(current.free_axes.data[world*6+d][k]);
        }
    }
    if(schedule.geometry_requested.data[world]!=0) {
        // Pose/motion scratch is dead. Preserve all13 moment scalars, including
        // the tiny original model-inertia asymmetry, until lower-H assembly.
        for(int component=0;component<13;++component) {
#if defined(__CUDA_ARCH__)
            float own=0.0f;
            if(lane<30) {
                const float m=data.body_mass.data[plan.body_ids.data[world*32+lane]];
                const wp::vec3 r(s[770+3*lane],s[771+3*lane],s[772+3*lane]);
                if(component==0)own=m;
                else if(component<4)own=m*r[component-1];
                else {
                    const int i=(component-4)/3,j=(component-4)%3;
                    own=s[860+9*lane+component-4]+m*((i==j?wp::dot(r,r):0.0f)-r[i]*r[j]);
                }
            }
            CUDA_SUM
            if(lane<23)s[13*lane+component]=subtree;
#else
            float own[32]={};
            for(int local=0;local<30;++local) {
                const float m=data.body_mass.data[plan.body_ids.data[world*32+local]];
                const wp::vec3 r(s[770+3*local],s[771+3*local],s[772+3*local]);
                if(component==0)own[local]=m;
                else if(component<4)own[local]=m*r[component-1];
                else {
                    const int i=(component-4)/3,j=(component-4)%3;
                    own[local]=s[860+9*local+component-4]+m*((i==j?wp::dot(r,r):0.0f)-r[i]*r[j]);
                }
            }
            CPU_SUM
            for(int d=0;d<23;++d) {
                const int body=d<7?d+1:10+5*((d-7)/4)+(d-7)%4;
                s[13*d+component]=value[body];
            }
#endif
        }
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
        for(int d=lane;d<23;d+=stride) {
            const float* t=s+13*d,*axis=s+632+6*d;
            const wp::vec3 l(axis[0],axis[1],axis[2]),w(axis[3],axis[4],axis[5]);
            const wp::vec3 moment(t[1],t[2],t[3]);
            const wp::vec3 linear=t[0]*l+wp::cross(w,moment);
            wp::vec3 angular=wp::cross(moment,l);
            for(int k=0;k<3;++k)for(int j=0;j<3;++j)angular[k]+=t[4+3*k+j]*w[j];
            for(int k=0;k<3;++k){s[320+6*d+k]=linear[k];s[323+6*d+k]=angular[k];}
        }
#if defined(__CUDA_ARCH__)
        __syncwarp(0xffffffffu);
#endif
        for(int entry=lane;entry<180;entry+=stride) {
            int i=0,j=0;
            if(entry<40) {
                const int finger=entry/10,index=entry%10;
                while((i+1)*(i+2)/2<=index)++i;
                j=index-i*(i+1)/2;i+=7+4*finger;j+=7+4*finger;
            } else if(entry<68) {
                const int index=entry-40;
                while((i+1)*(i+2)/2<=index)++i;
                j=index-i*(i+1)/2;
            } else {
                const int index=entry-68,finger=index/28;
                j=(index%28)/4;i=7+4*finger+index%4;
            }
            float value=0.0f;
            for(int k=0;k<6;++k)value+=s[632+6*j+k]*s[320+6*i+k];
            geometric.geometric.data[world*180+entry]=value;
            invalid|=!wp::isfinite(value);
        }
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffffu);
    invalid=__ballot_sync(0xffffffffu,invalid)!=0;
#endif
    if(lane==0) {
        if(invalid)schedule.status.data[world]=2;
        else {
            current.generation.data[world]=schedule.generation.data[world];
            current.valid.data[world]=1;
            if(schedule.geometry_requested.data[world]!=0) {
                geometric.generation.data[world]=schedule.generation.data[world];
                geometric.valid.data[world]=1;
            }
        }
    }
    return invalid?0:1;
""".replace("CUDA_SUM", _SUM_CUDA).replace("CPU_SUM", _SUM_CPU)


@wp.func_native(_COLLECT)
def _collect(
    address: wp.uint64,
    world: int,
    plan: KineticPlan,
    data: PublicationData,
    schedule: KineticSchedule,
    current: CurrentKineticCache,
    geometric: GeometricCache,
) -> int: ...


@functools.cache
def _get_kernel(arch, finish):
    """Share exact kinetic equations between cold construction and next finish."""
    checked_source()
    source = "\n".join(
        kuka_joint_world.operation_source(name)
        for name in (
            "update_qdd_from_velocity",
            "remove_free_root_transport_from_qdd",
            "integrate_generalized_joints",
        )
    )
    filename = f"<kinetic-generalized-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = dict(kernels.__dict__)
    exec(compile(source, filename, "exec"), namespace)
    update_qdd = namespace["_light_update_qdd_from_velocity"]
    remove_transport = namespace["_light_remove_free_root_transport_from_qdd"]
    integrate = namespace["_light_integrate_generalized_joints"]

    @wp.kernel(module="unique", enable_backward=False)
    def kinetic_state(
        plan: KineticPlan,
        data: PublicationData,
        schedule: KineticSchedule,
        current: CurrentKineticCache,
        geometric: GeometricCache,
    ):
        world, logical_lane = wp.tid()
        lanes = _lane()
        lane, stride = lanes[0], lanes[1]
        if stride == 1 and logical_lane != 0:
            return
        if _admit(world, int(wp.static(finish)), plan, data, schedule, current, geometric) == 0:
            return
        address = _storage()
        q = data.joint_q
        qd = data.joint_qd
        if wp.static(finish):
            for local in range(lane, 35, stride):
                dof = plan.dof_ids[world, local]
                update_qdd(
                    dof,
                    data.joint_qd,
                    data.kinematic_dof_mask,
                    1.0 / data.dt,
                    data.v_out,
                    data.joint_qdd,
                )
            _sync()
            for root in range(lane, 2, stride):
                remove_transport(
                    plan.root_slots[world, root],
                    data.free_root_joint_indices,
                    data.joint_qd_start,
                    data.kinematic_joint_mask,
                    data.joint_qd,
                    data.joint_qdd,
                )
            _sync()
            for local in range(lane, 32, stride):
                joint = plan.joint_ids[world, local]
                integrate(
                    joint,
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
        for local in range(lane, 32, stride):
            joint = plan.joint_ids[world, local]
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
                _store_shift(address, 0, wp.transform_get_translation(relative))
                relative = wp.transform(wp.vec3(), wp.transform_get_rotation(relative))
            elif local >= 30:
                _store_shift(address, local - 29, wp.vec3())
            _store_pose(address, local, relative)
        _sync()
        _scan_poses(address, world, plan)
        for root in range(lane, 3, stride):
            root_lane = int(0)
            if root > 0:
                root_lane = 29 + root
            body = plan.body_ids[world, root_lane]
            origin = wp.transform_point(_load_pose(address, root_lane), data.body_com[body])
            _store_origin(address, root, origin)
            current.origin[world, root] = origin + _load_shift(address, root)
            if root > 0:
                data.articulation_origin[data.body_to_articulation[body]] = current.origin[world, root]
        _sync()
        for local in range(lane, 32, stride):
            joint = plan.joint_ids[world, local]
            root = int(0)
            if local >= 30:
                root = local - 29
            origin = _load_origin(address, root)
            parent = plan.body_parent[world, local]
            anchor = data.joint_X_p[joint]
            if parent >= 0:
                anchor = _load_pose(address, parent) * anchor
            else:
                anchor = wp.transform(
                    wp.transform_get_translation(anchor) - _load_shift(address, root),
                    wp.transform_get_rotation(anchor),
                )
            anchor_local = wp.transform(
                wp.transform_get_translation(anchor) - origin,
                wp.transform_get_rotation(anchor),
            )
            velocity = wp.spatial_vector()
            local_dof = plan.body_local_dof[world, local]
            if local < 30:
                if local_dof >= 0:
                    global_dof = data.joint_qd_start[joint]
                    axis = kernels.transform_twist(
                        anchor_local,
                        wp.spatial_vector(wp.vec3(), data.joint_axis[global_dof]),
                    )
                    current.axes[world, local_dof] = axis
                    _store_axis(address, local_dof, axis)
                    velocity = axis * qd[global_dof]
            else:
                velocity = kernels.jcalc_motion(
                    data.joint_type[joint],
                    data.joint_axis,
                    data.joint_dof_dim[joint, 0],
                    data.joint_dof_dim[joint, 1],
                    anchor_local,
                    qd,
                    data.joint_qd_start[joint],
                    data.joint_S_s,
                )
                if local == 30:
                    for component in range(6):
                        current.free_axes[world, component] = data.joint_S_s[data.joint_qd_start[joint] + component]
            _store_motion(address, local, velocity)
        _sync()
        _scan_motion(address, world, plan)
        for local in range(lane, 32, stride):
            body = plan.body_ids[world, local]
            root = int(0)
            if local >= 30:
                root = local - 29
            pose = _load_pose(address, local)
            pose_com = pose * data.body_X_com[body]
            origin = _load_origin(address, root)
            velocity = _load_motion(address, local, 0)
            acceleration = _load_motion(address, local, 1)
            shift = _load_shift(address, root)
            data.body_q[body] = wp.transform(
                wp.transform_get_translation(pose) + shift,
                wp.transform_get_rotation(pose),
            )
            if local < 30:
                _primary_terms(
                    address,
                    local,
                    world,
                    body,
                    pose,
                    pose_com,
                    origin,
                    velocity,
                    acceleration,
                    data,
                    schedule,
                    current,
                )
            else:
                data.body_q_com[body] = pose_com
                data.body_v_s[body] = velocity
                data.body_a_s[body] = acceleration
                kernels.finalize_body_dynamics_body(
                    body,
                    data.body_to_articulation,
                    data.body_q,
                    data.body_q_com,
                    data.body_com,
                    data.body_mass,
                    data.body_inertia,
                    data.is_free_rigid,
                    data.articulation_origin,
                    data.materialize_all_body_inertia,
                    data.materialize_body_inertia_terms,
                    data.gravity,
                    data.body_v_s,
                    data.body_a_s,
                    data.body_I_s,
                    data.body_inertia_terms,
                    data.body_f_s,
                    data.body_qd,
                )
                current.com_offset[world, local] = wp.transform_get_translation(pose_com) - origin
                if local == 30:
                    current.free_bias[world] = data.body_f_s[body]
        _sync()
        success = _collect(address, world, plan, data, schedule, current, geometric)
        for root in range(lane, 3, stride):
            root_lane = int(0)
            if root > 0:
                root_lane = 29 + root
            data.fk_id_cache_valid[data.body_to_articulation[plan.body_ids[world, root_lane]]] = int(
                root > 0 and success != 0
            )
        _release(address)

    return kinetic_state


def get_construct_kernel(arch):
    """Construct current compact kinetics without advancing generalized state."""
    return _get_kernel(arch, False)


def get_finish_kernel(arch):
    """Advance all35 physical DOFs and publish complete next private/public state."""
    return _get_kernel(arch, True)

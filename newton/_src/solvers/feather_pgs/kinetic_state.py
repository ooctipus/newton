# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact current/next Kuka kinetic producer, with complete public state finish.

The primary spatial cache is private and transient. Joint bias and sparse180
geometric mass are produced by exact subtree moment reductions, not body-matrix
materialization. The two free roots retain the original canonical body service.
"""

# Warp uses explicit scalar constructors for values mutated inside device loops.

import functools

import warp as wp

from . import world_scan_publication as previous
from .cached_state import make_body_terms
from .cached_state_forest import make_forest_state
from .kinetic_source import checked_definitions
from .kinetic_types import (
    CurrentKineticCache,
    GeometricCache,
    KineticPlan,
    KineticSchedule,
)

PublicationData = previous.PublicationData


@wp.func_native(r"""
    const int index=gravity.shape[0]==1?0:world;
    return wp::array_t<wp::vec3>(gravity.data+index,1);
""")
def _world_gravity(gravity: wp.array[wp.vec3], world: int) -> wp.array[wp.vec3]:
    """Provide a register-only one-element view for both original force laws."""
    ...


def checked_source():
    """Check only the retained publication/generalized source definitions."""
    checked_definitions("cached_state_scan.py", ("_POSE_SCAN_TEMPLATE", "_MOTION_SCAN_TEMPLATE", "scan_source"))
    checked_definitions(
        "cached_state.py",
        ("_INTEGRATION", "_RELATIVE", "_BODY_TERMS", "generalized_operations", "make_state_kernel", "make_body_terms"),
    )
    checked_definitions(
        "cached_state_forest.py", ("_RELATIVE", "_ORIGINS", "_MOTION", "_BODIES", "_COLLECT", "make_forest_state")
    )
    checked_definitions("kinetic_state.py", ("_world_gravity",))
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
            "finalize_body_dynamics_values",
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


_primary_terms = make_body_terms(
    "address: wp.uint64,\n    local: int,\n    world: int,\n    body: int,\n    pose: wp.transform,\n    pose_com: wp.transform,\n    origin: wp.vec3,\n    velocity: wp.spatial_vector,\n    acceleration: wp.spatial_vector,\n    data: PublicationData,\n    schedule: KineticSchedule,\n    current: CurrentKineticCache,",
    globals(),
    refresh="schedule.geometry_requested[world]",
    cache="current",
    com_offset="radius",
    store="_store_terms(address, local, radius, inertia_com, wrench, schedule.geometry_requested[world])",
)


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
def _get_kernel(arch, finish, entry="plain"):
    """Bind the complete forest to compact moments and guarded cache lifetime."""
    checked_source()
    signature = "plan: KineticPlan, data: PublicationData, schedule: KineticSchedule, current: CurrentKineticCache, geometric: GeometricCache"
    namespace = globals()
    entry_setup = ""
    if entry == "repair" and not finish:
        signature += ", repair: wp.array[int]"
        entry_setup = "if repair[world] == 0:\n    return\n"
    elif entry == "guarded_finish" and finish:
        from .kinetic_guard import BoundaryGuard, _guard_world_check  # noqa: PLC0415

        signature += ", guard: BoundaryGuard"
        namespace = dict(namespace, BoundaryGuard=BoundaryGuard, _guard_world_check=_guard_world_check)
        entry_setup = """
if _guard_world_check(world, guard) != 0:
    if lane == 0:
        current.valid[world] = 0
        if schedule.geometry_requested[world] != 0:
            geometric.valid[world] = 0
        schedule.status[world] = 3
    return
"""
    elif entry != "plain":
        raise ValueError("Entry policy does not match the state operation")
    return make_forest_state(
        "kinetic_state",
        namespace,
        finish=finish,
        primary_bodies=30,
        free_roots=2,
        dofs=35,
        signature=signature,
        setup="""
world, logical_lane = wp.tid()
lanes = _lane()
lane, stride = lanes[0], lanes[1]
if stride == 1 and logical_lane != 0:
    return
"""
        + entry_setup
        + """
if _admit(world, int(wp.static(finish)), plan, data, schedule, current, geometric) == 0:
    return
data.gravity = _world_gravity(data.gravity, world)
""",
        joint_motion="""
velocity = wp.spatial_vector()
local_dof = plan.body_local_dof[world, local]
if local < $PRIMARY_BODIES:
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
    if local == $PRIMARY_BODIES:
        for component in range(6):
            current.free_axes[world, component] = data.joint_S_s[data.joint_qd_start[joint] + component]
""",
        origin_publication="    current.origin[world, root] = origin + _load_shift(address, root)\n    if root > 0:\n        data.articulation_origin[data.body_to_articulation[body]] = current.origin[world, root]",
        terms_args="        schedule,\n        current,",
        cache="current",
        free_com_offset="wp.transform_get_translation(pose_com) - origin",
        collect="_collect(address, world, plan, data, schedule, current, geometric)",
        valid="root > 0 and success != 0",
    )


def get_construct_kernel(arch):
    """Construct current compact kinetics without advancing generalized state."""
    return _get_kernel(arch, False)


def get_finish_kernel(arch):
    """Advance all35 physical DOFs and publish complete next private/public state."""
    return _get_kernel(arch, True)


def get_repair_kernel(arch):
    """Construct only explicitly requested worlds before reading their plans."""
    return _get_kernel(arch, False, "repair")


def get_guarded_finish_kernel(arch):
    """Publish next state only after the complete world's boundary checks pass."""
    return _get_kernel(arch, True, "guarded_finish")

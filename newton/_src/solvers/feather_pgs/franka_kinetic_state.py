# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete compact Franka state and live-force owner; original PGS remains external."""

import functools
import hashlib
import math
import os
from string import Template

import numpy as np
import warp as wp

from ...sim import ModelFlags
from . import (  # noqa: F401 - names in the checked generated state factory
    cached_force,
    cached_state_binding,
    forest_kinetic_plan,
    kernels,
    kinetic_state,
    response_operator,
    world_scan_owner,
    world_scan_publication,
)
from .cached_state import make_body_terms
from .cached_state_forest import make_forest_state
from .cached_state_scan import scan_source
from .forest_kinetic_layout import DEFAULT_LAYOUT, from_solver

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


@wp.func_native(scan_source(False, cuda_width=16))
def _scan_poses(address: wp.uint64, world: int, plan: KineticPlan): ...


@wp.func_native(scan_source(True, cuda_width=16))
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


def make_primary_terms(store_terms):
    """Bind body arithmetic to the selected scratch layout and public COM offset."""
    return make_body_terms(
        "address: wp.uint64, local: int, world: int, body: int, pose: wp.transform, pose_com: wp.transform, origin: wp.vec3, velocity: wp.spatial_vector, acceleration: wp.spatial_vector, data: PublicationData, cache: KineticData, refresh: int",
        dict(globals(), _store_terms=store_terms, _inertia_action=kinetic_state._inertia_action),
        refresh="refresh",
        cache="cache",
        com_offset="public_radius",
        store="_store_terms(address, local, mass, radius, inertia_com, wrench, refresh)",
    )


_primary_terms = make_primary_terms(_store_terms)


def free_geometry_source(layout):
    """Produce current free geometry when scalar and free factors share a width."""
    if layout.primary_dofs != 6:
        return ""
    return Template(r"""
    if(refresh) {
        const int body=plan.body_ids.data[world*32+$BODY];
        const auto rotation=data.body_q.data[body].q;
        const auto inertia=data.body_inertia.data[body];
        const float mass=data.body_mass.data[body];
        const int first=plan.dof_ids.data[world*$DOFS+6];
        for(int e=lane;e<36;e+=stride) {
            const auto a=data.joint_S_s.data[first+e/6],b=data.joint_S_s.data[first+e%6];
            const wp::vec3 wa=wp::quat_rotate_inv(rotation,wp::vec3(a[3],a[4],a[5]));
            const wp::vec3 wb=wp::quat_rotate_inv(rotation,wp::vec3(b[3],b[4],b[5]));
            const float value=mass*(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])+wp::dot(wa,wp::mul(inertia,wb));
            cache.geometric.data[plan.secondary_group.data[world]*36+e]=value;
            good&=wp::isfinite(value);
        }
    }
""").substitute(BODY=layout.primary_bodies, DOFS=layout.physical_dofs)


def collect_source(layout=DEFAULT_LAYOUT):
    """Bind reduction and projection extents while preserving the original arithmetic."""
    return Template(r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&15,stride=16;
    const unsigned mask=0xffffu<<(threadIdx.x&16);
#else
    const int lane=0,stride=1;
#endif
    // Each component owns its complete primary subtree reduction.
    // No atomics, prefix subtraction, or spatial matrix materialization.
    for(int component=lane;component<(refresh?19:6);component+=stride) {
        const int width=component<6?6:13;
        float* base=s+(component<6?686+component:752+component-6);
        for(int body=$PRIMARY_LAST;body>0;--body) {
            const int parent=plan.body_parent.data[world*32+body];
            base[width*parent]+=base[width*body];
        }
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    bool good=true;
    for(int dof=lane;dof<$N;dof+=stride) {
        const int body=plan.dof_body.data[dof];
        const float* axis=s+632+6*dof;
        float bias=0.0f;for(int k=0;k<6;++k)bias+=axis[k]*s[686+6*body+k];
        cache.bias.data[world*$N+dof]=bias;good&=wp::isfinite(bias);
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
    if(refresh)for(int e=lane;e<$N2;e+=stride) {
        const int row=e/$N,col=e%$N,src=plan.source.data[e];
        float value=0.0f;
        if(src>=0){const int projection=src==col?row:col;for(int k=0;k<6;++k)value+=s[632+6*projection+k]*s[6*src+k];}
        cache.geometric.data[plan.primary_group.data[world]*$N2+e]=value;good&=wp::isfinite(value);
    }
    $FREE_GEOMETRY
    for(int local=lane;local<$BODIES;local+=stride) {
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
""").substitute(
        PRIMARY_LAST=layout.primary_bodies - 1,
        N=layout.primary_dofs,
        N2=layout.primary_dofs**2,
        BODIES=layout.bodies,
        FREE_GEOMETRY=free_geometry_source(layout),
    )


@wp.func_native(collect_source())
def _collect(
    address: wp.uint64, world: int, plan: KineticPlan, data: PublicationData, cache: KineticData, refresh: int
) -> int: ...


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise RuntimeError(f"Franka original state seam changed: {old[:80]!r}")
    return source.replace(old, new)


def make_state_kernel(finish, namespace, *, suffix="", layout=DEFAULT_LAYOUT):
    """Bind the shared forest law to explicit storage helpers and cache lifetime."""
    prefix = "franka_kinetic" if layout == DEFAULT_LAYOUT else "scalar_forest"
    name = f"{prefix}_{'finish' if finish else 'repair'}{layout.bodies}_h{layout.primary_dofs**2}{suffix}_p16"
    return make_forest_state(
        name,
        namespace,
        finish=finish,
        primary_bodies=layout.primary_bodies,
        free_roots=2,
        dofs=layout.physical_dofs,
        signature="plan: KineticPlan, data: PublicationData, cache: KineticData, requests: wp.array[int], global_refresh: int",
        setup="""
group, logical_lane = wp.tid()
world = _world(group)
if world >= plan.body_ids.shape[0]:
    return
lanes = _lane()
lane, stride = lanes[0], lanes[1]
if stride == 1 and logical_lane != 0:
    return
refresh = int(global_refresh != 0)
if not wp.static(finish):
    valid = int(1)
    for root in range(3):
        art = plan.arts[world, root]
        refresh = int(refresh != 0 or requests[art] != 0)
        valid = valid & cache.current_valid[art]
    if valid != 0 and _same_source(cache, world, data.joint_q, data.joint_qd) != 0:
        if refresh == 0 or (cache.geometry_valid[world] != 0 and cache.geometry_generation[world] == cache.generation[world]):
            return
""",
        joint_motion="""
local_dof = plan.body_local_dof[world, local]
velocity = kernels.jcalc_motion(
    data.joint_type[joint], data.joint_axis,
    data.joint_dof_dim[joint, 0], data.joint_dof_dim[joint, 1],
    anchor_local, qd, data.joint_qd_start[joint], data.joint_S_s,
)
if local < $PRIMARY_BODIES and local_dof >= 0:
    axis = data.joint_S_s[data.joint_qd_start[joint]]
    _store_axis(address, local_dof, axis)
""",
        origin_publication="    data.articulation_origin[data.body_to_articulation[body]] = origin + _load_shift(address, root)",
        terms_args="        cache,\n        refresh,",
        cache="cache",
        free_com_offset="wp.transform_point(pose, data.body_com[body]) - origin",
        collect="_collect(address, world, plan, data, cache, refresh)",
        valid="success != 0 and cache.status[world] == 0",
        stamp="if lane == 0:\n    _stamp_source(cache, world, q, qd)\n",
        padded=True,
    )


@functools.cache
def get_state_kernel(finish):
    """Construct a paired-world state kernel without inspecting another owner."""
    return make_state_kernel(finish, globals())


_FORCE = cached_force.ForceOperands(
    q="f.joint_q.data",
    state_qd="f.joint_qd.data",
    predictor_qd="f.predictor_qd.data",
    stiffness="f.stiffness.data",
    reference="f.reference.data",
    damping="f.damping.data",
    dt="f.dt",
)


def predictor_source(layout=DEFAULT_LAYOUT):
    """Bind force extents and original held-factor actions to the complete layout."""
    return Template(r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&15,stride=16;
    const unsigned mask=0xffffu<<(threadIdx.x&16);
#else
    const int lane=0,stride=1;
#endif
    bool good=wp::isfinite(f.dt)&&f.dt>0.0f&&cache.status.data[world]==0;
    for(int root=0;root<3;++root)good&=cache.current_valid.data[plan.arts.data[world*3+root]]!=0;
    for(int b=lane;b<$BODIES;b+=stride) {
        const int body=plan.body_ids.data[world*32+b];
        float v[6]={};
        if(b<$DYNAMIC_BODIES&&(f.body_flags.data[body]&2)==0) {
            const auto external=f.body_f.data[body];const auto r=cache.com_offset.data[world*$BODIES+b];
            $SHIFT
        }
        for(int k=0;k<6;++k){s[6*b+k]=v[k];good&=wp::isfinite(v[k]);}
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    for(int component=lane;component<6;component+=stride)
        for(int b=$PRIMARY_LAST;b>0;--b)s[6*plan.body_parent.data[world*32+b]+component]+=s[6*b+component];
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    for(int i=lane;i<$RESPONSE;i+=stride) {
        const int dof=plan.dof_ids.data[world*$PHYSICAL+i];
        float tau=f.joint_f.data[dof]+f.u0.data[dof];
        if(i<$N) {
            const auto axis=f.S.data[dof];const int body=plan.dof_body.data[i];
            float ext=0.0f;$PROJECT
            tau+=-cache.bias.data[world*$N+i]+ext+$PASSIVE;
        } else {
            const auto axis=f.S.data[dof],bias=cache.free_bias.data[world];
            $PROJECT_FREE
        }
        s[$TORQUE+i]=tau;good&=wp::isfinite(tau);
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    // Two independent original held-L triangular actions; no new factor or inverse.
    for(int family=lane;family<2;family+=stride) {
        const int n=family==0?$N:6,offset=family==0?0:$N;
        const int group=family==0?plan.primary_group.data[world]:plan.secondary_group.data[world];
        const float* l=family==0?f.lower9.data+group*$N2:f.lower6.data+group*36;
        float* x=s+$ACCELERATION+offset;
        $FORWARD_ACTION
        $TRANSPOSE_ACTION
    }
#if defined(__CUDA_ARCH__)
    good=__all_sync(mask,good);__syncwarp(mask);
#endif
    if(!good){if(lane==0)cache.status.data[world]|=2;return;}
    for(int i=lane;i<$PHYSICAL;i+=stride) {
        const int dof=plan.dof_ids.data[world*$PHYSICAL+i];
        const float acceleration=i<$RESPONSE&&f.kinematic_dof.data[dof]==0?s[$ACCELERATION+i]:0.0f;
        f.qdd.data[dof]=acceleration;
        f.vhat.data[dof]=$VELOCITY;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp(mask);
#endif
    for(int root=lane;root<2;root+=stride) {
        const int joint=plan.joint_ids.data[world*32+$PRIMARY_BODIES+root];
        if(f.kinematic_joint.data[joint]!=0)continue;
        const int d=plan.dof_ids.data[world*$PHYSICAL+$N+6*root];
        $TRANSPORT
    }
""").substitute(
        BODIES=layout.bodies,
        DYNAMIC_BODIES=layout.primary_bodies + 1,
        PRIMARY_LAST=layout.primary_bodies - 1,
        PRIMARY_BODIES=layout.primary_bodies,
        RESPONSE=layout.response_dofs,
        PHYSICAL=layout.physical_dofs,
        N=layout.primary_dofs,
        N2=layout.primary_dofs**2,
        TORQUE=layout.torque,
        ACCELERATION=layout.acceleration,
        SHIFT=cached_force.shift_wrench("external", "r", "v[k]", "v[k+3]"),
        PROJECT=cached_force.project_wrench("axis", "s[6*body+k]", "ext"),
        PASSIVE=_FORCE.passive(f"plan.q_index.data[world*{layout.primary_dofs}+i]", "dof"),
        PROJECT_FREE=cached_force.project_wrench("axis", f"s[6*{layout.primary_bodies}+k]", "tau", bias="bias"),
        VELOCITY=_FORCE.velocity("dof", "acceleration"),
        TRANSPORT=_FORCE.transport("d", "f.vhat.data[d+k]"),
        FORWARD_ACTION=response_operator.CholeskySolve("n", "l").forward(
            f"s[{layout.torque}+offset+$index]", "x[$index]"
        ),
        TRANSPOSE_ACTION=response_operator.CholeskySolve("n", "l").transpose(
            "x[$index]", "x[$index]", post_row="good&=wp::isfinite(x[row]);"
        ),
    )


def get_predictor_kernel(layout=None):
    return _get_predictor_kernel(DEFAULT_LAYOUT if layout is None else layout)


@functools.cache
def _get_predictor_kernel(layout):
    storage_source = f"""
#if defined(__CUDA_ARCH__)
        __shared__ float values[2*{layout.force_floats}];
        return reinterpret_cast<uint64_t>(values+{layout.force_floats}*((threadIdx.x&31)>>4));
#else
        return reinterpret_cast<uint64_t>(malloc({layout.force_floats}*sizeof(float)));
#endif
    """

    @wp.func_native(storage_source)
    def force_storage() -> wp.uint64: ...

    @wp.func_native(predictor_source(layout))
    def predict(address: wp.uint64, world: int, plan: KineticPlan, cache: KineticData, f: ForceInput): ...

    def predictor(plan: KineticPlan, cache: KineticData, force: ForceInput):
        group, logical_lane = wp.tid()
        world = _world(group)
        if world >= plan.body_ids.shape[0]:
            return
        lanes = _lane()
        if lanes[1] == 1 and logical_lane != 0:
            return
        address = force_storage()
        predict(address, world, plan, cache, force)
        _release(address)

    predictor.__name__ = (
        "franka_kinetic_current_force_held9_6_p16"
        if layout == DEFAULT_LAYOUT
        else f"scalar_forest_current_force_held{layout.primary_dofs}_6_b{layout.bodies}_p16"
    )
    predictor.__qualname__ = predictor.__name__
    return wp.kernel(module="unique", enable_backward=False)(predictor)


def build_plan(solver):
    """Bind topology-derived ownership to complete physical and response dimensions."""
    return forest_kinetic_plan.build_plan(solver)


def supported(solver):
    """Require the unchanged immediate matrix-free publication and factor consumers."""
    from . import solver_feather_pgs as source  # noqa: PLC0415

    try:
        layout = from_solver(solver)
    except ValueError:
        return False
    return bool(
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and not solver.model.particle_count
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
        and (
            (
                layout.primary_dofs == 6
                and solver.cholesky_kernel == "auto"
                and solver.L_by_size.get(6) is not None
                and solver._compact_diagonal_mass_size != 6
            )
            or (
                solver._crba_cholesky_warp_kernels_by_size.get(layout.primary_dofs) is not None
                and solver._crba_cholesky_warp_kernels_by_size.get(6) is not None
            )
        )
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
        tuple(sorted(solver.n_arts_by_size.items())),
        tuple(sorted(solver._crba_source_dof_by_size)),
        tuple(solver.size_groups),
    )


class FrankaKineticState:
    """Live current/bias/moment lifetime without taking ownership of constraint solves."""

    def __init__(self, solver):
        self.solver = solver
        model, device, worlds = solver.model, solver.model.device, int(solver.world_count)
        if model.requires_grad or model.particle_count:
            raise ValueError("Franka kinetic state excludes gradient and particle ownership")
        self.layout = from_solver(solver)
        n, bodies = self.layout.primary_dofs, self.layout.bodies
        self.host_plan = forest_kinetic_plan.build_plan(solver, self.layout)
        self.same_width = n == 6
        if self.same_width:
            # Geometric assembly does not read joint/source projection metadata.
            # Keep its plain element schedule separate from ordinary CRBA tables.
            self.factor_offsets = wp.zeros(n, dtype=int, device=device)
            self.factor_schedule = wp.array(
                [row * n + col for row in range(n) for col in range(row + 1)], dtype=int, device=device
            )
        self.plan = KineticPlan()
        for name, value in self.host_plan.items():
            setattr(self.plan, name, wp.array(value, dtype=int, device=device))
        self.data = KineticData()
        for name, shape, dtype in (
            ("bias", (worlds, n), float),
            ("com_offset", (worlds, bodies), wp.vec3),
            ("free_bias", worlds, wp.spatial_vector),
            ("geometric", (solver.n_arts_by_size[n], n * n), float),
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
        self.compact_workspace = (
            self.layout != DEFAULT_LAYOUT or os.environ.get("FEATHER_PGS_FRANKA_COMPACT_WORKSPACE", "0") == "1"
        )
        state_factory = get_state_kernel
        if self.compact_workspace:
            from .franka_compact_workspace import get_state_kernel as state_factory  # noqa: PLC0415

            self.repair_kernel, self.finish_kernel = (state_factory(finish, self.layout) for finish in (False, True))
        else:
            self.repair_kernel, self.finish_kernel = state_factory(False), state_factory(True)
        self.predictor_kernel = get_predictor_kernel(self.layout)
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
            for name, size in (("group_to_art", n), ("group_to_art", 6), ("_crba_source_dof_by_size", n))
            if size in getattr(solver, name)
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
        """Invalidate current state for every owned articulation in selected worlds."""
        cached_state_binding.invalidate(
            self.solver.world_count,
            world_mask,
            self.plan.arts,
            self.solver.art_to_world,
            self.data.current_valid,
            self.data.geometry_valid,
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
        """Bind live current and next-state publication descriptors."""
        return cached_state_binding.publication(self.solver, state_in, state_aug, state_out, dt)

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
        force.lower9, force.lower6 = solver.L_by_size[self.layout.primary_dofs], solver.L_by_size[6]
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
        if not cached_state_binding.disjoint_states(state_in, state_out):
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

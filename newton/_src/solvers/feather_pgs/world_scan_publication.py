# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental late all-world generalized/FK/dynamics publication.

One body per lane composes current transforms and twist/bias-acceleration
segments. This owns no mass factor, constraint solve, or early-publication stream.
"""

import functools
import hashlib
import linecache

import numpy as np
import warp as wp

from . import kernels, kuka_joint_world
from .cached_state_scan import scan_source


@wp.struct
class PublicationData:
    joint_type: wp.array[int]
    joint_parent: wp.array[int]
    joint_child: wp.array[int]
    joint_q_start: wp.array[int]
    joint_qd_start: wp.array[int]
    joint_dof_dim: wp.array2d[int]
    joint_axis: wp.array[wp.vec3]
    joint_X_p: wp.array[wp.transform]
    joint_X_c: wp.array[wp.transform]
    kinematic_dof_mask: wp.array[int]
    kinematic_joint_mask: wp.array[int]
    free_root_joint_indices: wp.array[int]
    body_X_com: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    body_mass: wp.array[float]
    body_inertia: wp.array[wp.mat33]
    body_to_articulation: wp.array[int]
    is_free_rigid: wp.array[int]
    gravity: wp.array[wp.vec3]
    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    v_out: wp.array[float]
    joint_qdd: wp.array[float]
    joint_q_new: wp.array[float]
    joint_qd_new: wp.array[float]
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_q_com: wp.array[wp.transform]
    joint_S_s: wp.array[wp.spatial_vector]
    articulation_origin: wp.array[wp.vec3]
    body_v_s: wp.array[wp.spatial_vector]
    body_a_s: wp.array[wp.spatial_vector]
    body_f_s: wp.array[wp.spatial_vector]
    body_I_s: wp.array[wp.spatial_matrix]
    body_inertia_terms: wp.array2d[float]
    fk_id_cache_valid: wp.array[int]
    dt: float
    angular_damping: float
    materialize_all_body_inertia: int
    materialize_body_inertia_terms: int


def validate_scan_plan(plan: kuka_joint_world.HostPlan):
    """Require complete 32-body parent-first ownership within four doubling rounds."""
    if plan.body_parent.shape != plan.body_ids.shape or plan.body_parent.shape[1] != 32:
        raise ValueError("World publication requires a complete 32-body plan")
    depth = np.zeros_like(plan.body_parent)
    for lane in range(32):
        parent = plan.body_parent[:, lane]
        if np.any(parent < -1) or np.any(parent >= lane):
            raise ValueError("World publication requires local parent-first bodies")
        valid = parent >= 0
        depth[:, lane] = np.where(valid, depth[np.arange(len(depth)), np.maximum(parent, 0)] + 1, 0)
    if np.any(depth >= 16):
        raise ValueError("World publication supports at most fifteen parent edges")


_STORAGE = r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[624];
    return reinterpret_cast<uint64_t>(values);
#else
    return reinterpret_cast<uint64_t>(malloc(624*sizeof(float)));
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
    return wp::vec2i(threadIdx.x&31,32);
#else
    return wp::vec2i(0,1);
#endif
""")
def _lane() -> wp.vec2i: ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+7*lane; for(int k=0;k<3;++k)s[k]=pose.p[k]; for(int k=0;k<4;++k)s[3+k]=pose.q[k];"
)
def _store_pose(address: wp.uint64, lane: int, pose: wp.transform): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+7*lane; return wp::transform(wp::vec3(s[0],s[1],s[2]),wp::quat(s[3],s[4],s[5],s[6]));"
)
def _load_pose(address: wp.uint64, lane: int) -> wp.transform: ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+224+6*lane; for(int k=0;k<6;++k){s[k]=velocity[k];s[192+k]=0.0f;}"
)
def _store_motion(address: wp.uint64, lane: int, velocity: wp.spatial_vector): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+224+6*lane+192*acceleration; wp::spatial_vector result;for(int k=0;k<6;++k)result[k]=s[k];return result;"
)
def _load_motion(address: wp.uint64, lane: int, acceleration: int) -> wp.spatial_vector: ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+608+3*root;for(int k=0;k<3;++k)s[k]=origin[k];")
def _store_origin(address: wp.uint64, root: int, origin: wp.vec3): ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+608+3*root;return wp::vec3(s[0],s[1],s[2]);")
def _load_origin(address: wp.uint64, root: int) -> wp.vec3: ...


_POSE_SCAN = scan_source(False)


@wp.func_native(_POSE_SCAN)
def _scan_poses(address: wp.uint64, world: int, plan: kuka_joint_world.JointWorldPlan): ...


_MOTION_SCAN = scan_source(True)


@wp.func_native(_MOTION_SCAN)
def _scan_motion(address: wp.uint64, world: int, plan: kuka_joint_world.JointWorldPlan): ...


@functools.cache
def _publication_kernels(arch):
    """Compile the complete late generalized, pose, twist and dynamics owner."""
    source = "\n".join(
        kuka_joint_world.operation_source(name)
        for name in (
            "update_qdd_from_velocity",
            "remove_free_root_transport_from_qdd",
            "integrate_generalized_joints",
        )
    )
    filename = f"<world-scan-publication-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(kernels.__dict__)
    exec(compile(source, filename, "exec"), namespace)
    update_qdd = namespace["_light_update_qdd_from_velocity"]
    remove_transport = namespace["_light_remove_free_root_transport_from_qdd"]
    integrate = namespace["_light_integrate_generalized_joints"]
    finalize = kernels.finalize_body_dynamics_values

    @wp.func
    def publish_world(
        world: int,
        logical_lane: int,
        plan: kuka_joint_world.JointWorldPlan,
        data: PublicationData,
        already_integrated: int,
    ):
        lanes = _lane()
        lane, stride = lanes[0], lanes[1]
        if stride == 1 and logical_lane != 0:
            return
        address = _storage()
        if already_integrated == 0:
            for local in range(lane, 35, stride):
                dof = plan.dof_ids[world, local]
                update_qdd(dof, data.joint_qd, data.kinematic_dof_mask, 1.0 / data.dt, data.v_out, data.joint_qdd)
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
        for local in range(lane, 32, stride):
            joint = plan.joint_ids[world, local]
            transform = kernels.jcalc_transform(
                data.joint_type[joint],
                data.joint_axis,
                data.joint_qd_start[joint],
                data.joint_dof_dim[joint, 0],
                data.joint_dof_dim[joint, 1],
                data.joint_q_new,
                data.joint_q_start[joint],
            )
            relative = data.joint_X_p[joint] * transform * wp.transform_inverse(data.joint_X_c[joint])
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
            data.articulation_origin[data.body_to_articulation[body]] = origin
        _sync()
        for local in range(lane, 32, stride):
            joint = plan.joint_ids[world, local]
            body = plan.body_ids[world, local]
            root = int(0)
            if local >= 30:
                root = local - 29
            origin = _load_origin(address, root)
            parent = plan.body_parent[world, local]
            anchor = data.joint_X_p[joint]
            if parent >= 0:
                anchor = _load_pose(address, parent) * anchor
            anchor_local = wp.transform(
                wp.transform_get_translation(anchor) - origin, wp.transform_get_rotation(anchor)
            )
            velocity = kernels.jcalc_motion(
                data.joint_type[joint],
                data.joint_axis,
                data.joint_dof_dim[joint, 0],
                data.joint_dof_dim[joint, 1],
                anchor_local,
                data.joint_qd_new,
                data.joint_qd_start[joint],
                data.joint_S_s,
            )
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
            data.body_q[body] = pose
            data.body_q_com[body] = pose_com
            data.body_v_s[body] = velocity
            data.body_a_s[body] = acceleration
            finalize(
                body,
                data.body_to_articulation[body],
                pose,
                pose_com,
                data.body_com,
                data.body_mass,
                data.body_inertia,
                data.is_free_rigid,
                origin,
                data.materialize_all_body_inertia,
                data.materialize_body_inertia_terms,
                data.gravity,
                velocity,
                acceleration,
                data.body_I_s,
                data.body_inertia_terms,
                data.body_f_s,
                data.body_qd,
            )
        _sync()
        for root in range(lane, 3, stride):
            root_lane = int(0)
            if root > 0:
                root_lane = 29 + root
            data.fk_id_cache_valid[data.body_to_articulation[plan.body_ids[world, root_lane]]] = 1
        _release(address)

    @wp.kernel(module="unique", enable_backward=False)
    def world_scan_publication(plan: kuka_joint_world.JointWorldPlan, data: PublicationData):
        world, logical_lane = wp.tid()
        publish_world(world, logical_lane, plan, data, 0)

    @wp.kernel(module="unique", enable_backward=False)
    def world_scan_publication_resolved(
        plan: kuka_joint_world.JointWorldPlan, data: PublicationData, resolved: wp.array[int]
    ):
        world, logical_lane = wp.tid()
        publish_world(world, logical_lane, plan, data, resolved[world])

    return world_scan_publication, world_scan_publication_resolved


def get_kernel(arch):
    """Return the standalone generalized and body publication owner."""
    return _publication_kernels(arch)[0]


def get_resolved_kernel(arch):
    """Publish bodies and integrate only worlds not handled by the current light owner.

    The caller must pass the current light owner's resolved array only after that
    invocation has integrated selected worlds. Ordinary ZERO classification does
    not establish this ownership and must use the standalone kernel instead.
    """
    return _publication_kernels(arch)[1]

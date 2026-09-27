# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental late all-world generalized/FK/dynamics publication.

One body per lane composes current transforms and twist/bias-acceleration
segments. This owns no mass factor, constraint solve, or early-publication stream.
"""

import ast
import functools
import hashlib
import inspect
import linecache
import textwrap

import numpy as np
import warp as wp

from . import kernels, kuka_joint_world


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


_POSE_SCAN = r"""
    float* s=reinterpret_cast<float*>(address);
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
    wp::transform value;
    for(int k=0;k<7;++k)value[k]=s[7*lane+k];
    int parent=plan.body_parent.data[world*32+lane];
    for(int round=0;round<4;++round) {
        const int source=parent>=0?parent:lane;
        wp::transform ancestor;
        for(int k=0;k<7;++k)ancestor[k]=__shfl_sync(0xffffffffu,value[k],source);
        const int next=__shfl_sync(0xffffffffu,parent,source);
        if(parent>=0)value=wp::transform_multiply(ancestor,value);
        parent=parent>=0?next:-1;
    }
    for(int k=0;k<7;++k)s[7*lane+k]=value[k];
    __syncwarp(0xffffffffu);
#else
    wp::transform value[32],previous[32];
    int parent[32],next[32];
    for(int lane=0;lane<32;++lane) {
        parent[lane]=plan.body_parent.data[world*32+lane];
        for(int k=0;k<7;++k)value[lane][k]=s[7*lane+k];
    }
    for(int round=0;round<4;++round) {
        for(int lane=0;lane<32;++lane) {
            previous[lane]=value[lane];
            next[lane]=parent[lane]>=0?parent[parent[lane]]:-1;
        }
        for(int lane=0;lane<32;++lane) {
            if(parent[lane]>=0)value[lane]=wp::transform_multiply(previous[parent[lane]],previous[lane]);
            parent[lane]=next[lane];
        }
    }
    for(int lane=0;lane<32;++lane)for(int k=0;k<7;++k)s[7*lane+k]=value[lane][k];
#endif
"""


@wp.func_native(_POSE_SCAN)
def _scan_poses(address: wp.uint64, world: int, plan: kuka_joint_world.JointWorldPlan): ...


_MOTION_SCAN = r"""
    float* s=reinterpret_cast<float*>(address)+224;
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x&31;
    float v[6],a[6];
    for(int k=0;k<6;++k){v[k]=s[6*lane+k];a[k]=0.0f;}
    int parent=plan.body_parent.data[world*32+lane];
    for(int round=0;round<4;++round) {
        const int source=parent>=0?parent:lane;
        float pv[6],pa[6];
        for(int k=0;k<6;++k){pv[k]=__shfl_sync(0xffffffffu,v[k],source);pa[k]=__shfl_sync(0xffffffffu,a[k],source);}
        const int next=__shfl_sync(0xffffffffu,parent,source);
        if(parent>=0) {
            const wp::vec3 p_lin(pv[0],pv[1],pv[2]),p_ang(pv[3],pv[4],pv[5]);
            const wp::vec3 c_lin(v[0],v[1],v[2]),c_ang(v[3],v[4],v[5]);
            const wp::vec3 c0=wp::cross(p_ang,c_lin),c1=wp::cross(p_lin,c_ang),c2=wp::cross(p_ang,c_ang);
            for(int k=0;k<3;++k){a[k]=pa[k]+a[k]+(c0[k]+c1[k]);a[k+3]=pa[k+3]+a[k+3]+c2[k];}
            for(int k=0;k<6;++k)v[k]=pv[k]+v[k];
        }
        parent=parent>=0?next:-1;
    }
    for(int k=0;k<6;++k){s[6*lane+k]=v[k];s[192+6*lane+k]=a[k];}
    __syncwarp(0xffffffffu);
#else
    float v[32][6],a[32][6]={},old_v[32][6],old_a[32][6];
    int parent[32],next[32];
    for(int lane=0;lane<32;++lane){parent[lane]=plan.body_parent.data[world*32+lane];for(int k=0;k<6;++k)v[lane][k]=s[6*lane+k];}
    for(int round=0;round<4;++round) {
        for(int lane=0;lane<32;++lane) {
            next[lane]=parent[lane]>=0?parent[parent[lane]]:-1;
            for(int k=0;k<6;++k){old_v[lane][k]=v[lane][k];old_a[lane][k]=a[lane][k];}
        }
        for(int lane=0;lane<32;++lane) {
            if(parent[lane]>=0) {
                const float* pv=old_v[parent[lane]];const float* pa=old_a[parent[lane]];const float* cv=old_v[lane];
                const wp::vec3 p_lin(pv[0],pv[1],pv[2]),p_ang(pv[3],pv[4],pv[5]);
                const wp::vec3 c_lin(cv[0],cv[1],cv[2]),c_ang(cv[3],cv[4],cv[5]);
                const wp::vec3 c0=wp::cross(p_ang,c_lin),c1=wp::cross(p_lin,c_ang),c2=wp::cross(p_ang,c_ang);
                for(int k=0;k<3;++k){a[lane][k]=pa[k]+old_a[lane][k]+(c0[k]+c1[k]);a[lane][k+3]=pa[k+3]+old_a[lane][k+3]+c2[k];}
                for(int k=0;k<6;++k)v[lane][k]=pv[k]+cv[k];
            }
            parent[lane]=next[lane];
        }
    }
    for(int lane=0;lane<32;++lane)for(int k=0;k<6;++k){s[6*lane+k]=v[lane][k];s[192+6*lane+k]=a[lane][k];}
#endif
"""


@wp.func_native(_MOTION_SCAN)
def _scan_motion(address: wp.uint64, world: int, plan: kuka_joint_world.JointWorldPlan): ...


def finalizer_source():
    """Retain the original body law while replacing five canonical reloads with private values."""
    source = textwrap.dedent(inspect.getsource(kernels.finalize_body_dynamics_body.func))
    tree = ast.parse(source)
    function = tree.body[0]
    function.name = "_scan_finalize_body_values"
    extra = (
        ast.parse(
            "def extra(pose: wp.transform, pose_com: wp.transform, origin_value: wp.vec3, "
            "velocity: wp.spatial_vector, acceleration: wp.spatial_vector):\n    pass"
        )
        .body[0]
        .args.args
    )
    function.args.args.extend(extra)
    replacements = {
        "body_q[body]": "pose",
        "body_q_com[body]": "pose_com",
        "articulation_origin[articulation]": "origin_value",
        "body_v_s[body]": "velocity",
        "body_a_s[body]": "acceleration",
    }
    counts = dict.fromkeys(replacements, 0)

    class ReplaceValues(ast.NodeTransformer):
        def visit_Subscript(self, node):
            key = ast.unparse(node)
            if key in replacements and isinstance(node.ctx, ast.Load):
                counts[key] += 1
                return ast.copy_location(ast.Name(id=replacements[key], ctx=ast.Load()), node)
            return self.generic_visit(node)

    tree = ReplaceValues().visit(tree)
    if counts != {
        "body_q[body]": 2,
        "body_q_com[body]": 1,
        "articulation_origin[articulation]": 1,
        "body_v_s[body]": 1,
        "body_a_s[body]": 1,
    }:
        raise RuntimeError("Original finalizer ownership changed; review the private value adapter")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


@functools.cache
def get_kernel(arch):
    """Compile the complete late generalized, pose, twist and dynamics owner."""
    source = finalizer_source() + "\n".join(
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
    finalize = namespace["_scan_finalize_body_values"]

    @wp.kernel(module="unique", enable_backward=False)
    def world_scan_publication(plan: kuka_joint_world.JointWorldPlan, data: PublicationData):
        world, logical_lane = wp.tid()
        lanes = _lane()
        lane, stride = lanes[0], lanes[1]
        if stride == 1 and logical_lane != 0:
            return
        address = _storage()
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
                pose,
                pose_com,
                origin,
                velocity,
                acceleration,
            )
        _sync()
        for root in range(lane, 3, stride):
            root_lane = int(0)
            if root > 0:
                root_lane = 29 + root
            data.fk_id_cache_valid[data.body_to_articulation[plan.body_ids[world, root_lane]]] = 1
        _release(address)

    return world_scan_publication


def _resolved_source(source):
    """Guard only generalized work already published by the current light owner."""
    signature = "    def world_scan_publication(plan: kuka_joint_world.JointWorldPlan, data: PublicationData):\n"
    first = "        for local in range(lane, 35, stride):\n"
    body = (
        "        for local in range(lane, 32, stride):\n"
        "            joint = plan.joint_ids[world, local]\n"
        "            transform = kernels.jcalc_transform(\n"
    )
    for seam in (signature, first, body, "def get_kernel(arch):\n", "    return world_scan_publication\n"):
        if source.count(seam) != 1:
            raise ValueError("Original publication ownership seam changed; review resolved composition")
    start, end = source.index(first), source.index(body)
    generalized = source[start:end]
    if start >= end or generalized.count("        _sync()\n") != 3:
        raise ValueError("Original generalized publication boundaries changed")
    result = source[:start] + "        if resolved[world] == 0:\n" + textwrap.indent(generalized, "    ") + source[end:]
    result = result.replace("def get_kernel(arch):\n", "def _get_resolved_factory(arch):\n")
    result = result.replace(
        signature,
        "    def world_scan_publication_resolved(\n"
        "        plan: kuka_joint_world.JointWorldPlan, data: PublicationData, resolved: wp.array[int]\n"
        "    ):\n",
    )
    return result.replace("    return world_scan_publication\n", "    return world_scan_publication_resolved\n")


@functools.cache
def get_resolved_kernel(arch):
    """Return the late body owner with a current light-resolved generalized skip.

    The caller must pass the current light owner's resolved array only after that
    invocation has integrated selected worlds. Ordinary ZERO classification does
    not establish this ownership and must use the standalone kernel instead.
    """
    source = _resolved_source(inspect.getsource(get_kernel))
    filename = f"<world-scan-resolved-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(globals())
    exec(compile(source, filename, "exec"), namespace)
    return namespace["_get_resolved_factory"](arch)

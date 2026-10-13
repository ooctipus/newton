# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared generalized laws with explicit kernel or physical-index entry."""

import hashlib
import linecache
from string import Template

from .kinetic_source import checked_definitions

_OPERATIONS = {
    "apply_free_root_velocity_corrections": r'''@wp.${decorator}
def ${name}(${index_argument}free_root_joint_indices: wp.array[int], joint_qd_start: wp.array[int], joint_child: wp.array[int], body_to_articulation: wp.array[int], is_free_rigid: wp.array[int], art_group_index: wp.array[int], kinematic_joint_mask: wp.array[int], body_q: wp.array[wp.transform], body_inertia: wp.array[wp.mat33], cholesky: wp.array3d[float], joint_qd: wp.array[float], dt: float, requires_grad: bool, apply_transport: bool, v_hat: wp.array[float]):
    """Fuse free-root transport with the isolated rigid-body gyroscopic update."""
    root_index = ${index}
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    if apply_transport:
        v = wp.vec3(joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2])
        c = wp.cross(w, v)
        v_hat[d + 0] = v_hat[d + 0] + c[0] * dt
        v_hat[d + 1] = v_hat[d + 1] + c[1] * dt
        v_hat[d + 2] = v_hat[d + 2] + c[2] * dt
    body = joint_child[free_root_joint_indices[root_index]]
    art = body_to_articulation[body]
    if is_free_rigid[art] == 0:
        return
    predicted_world = wp.vec3(v_hat[d + 3], v_hat[d + 4], v_hat[d + 5])
    if not requires_grad:
        if w[0] == 0.0 and w[1] == 0.0 and (w[2] == 0.0) and (predicted_world[0] == 0.0) and (predicted_world[1] == 0.0) and (predicted_world[2] == 0.0):
            return
    inertia = body_inertia[body]
    if not requires_grad:
        if inertia[0, 0] == inertia[1, 1] and inertia[1, 1] == inertia[2, 2] and (inertia[0, 1] == 0.0) and (inertia[0, 2] == 0.0) and (inertia[1, 0] == 0.0) and (inertia[1, 2] == 0.0) and (inertia[2, 0] == 0.0) and (inertia[2, 1] == 0.0):
            return
    group = art_group_index[art]
    lower = wp.mat33(0.0)
    for r in range(3):
        for c in range(r + 1):
            lower[r, c] = cholesky[group, r + 3, c + 3]
    rotation = wp.transform_get_rotation(body_q[body])
    basis = wp.quat_to_matrix(rotation)
    effective = wp.transpose(basis) * (lower * wp.transpose(lower)) * basis
    omega = wp.quat_rotate_inv(rotation, w)
    predicted = wp.quat_rotate_inv(rotation, predicted_world)
    corrected = wp.quat_rotate(rotation, _gyroscopic_velocity(inertia, effective, omega, predicted, dt))
    for k in range(3):
        v_hat[d + 3 + k] = corrected[k]
''',
    "remove_free_root_transport_from_qdd": r'''@wp.${decorator}
def ${name}(${index_argument}free_root_joint_indices: wp.array[int], joint_qd_start: wp.array[int], kinematic_joint_mask: wp.array[int], joint_qd: wp.array[float], joint_qdd: wp.array[float]):
    """Make ``jcalc_integrate`` reproduce the solved velocity exactly.

    The solver commits to ``v_out``; ``qdd = (v_out - qd) / dt`` alone would let
    the integrator's transport term push the realized root velocity to
    ``v_out + dt * (omega x v)``. Subtracting the term here closes the loop, and
    in the contact-free case recovers the dynamics' own ``qdd`` bit for bit.
    """
    root_index = ${index}
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    v = wp.vec3(joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2])
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    c = wp.cross(w, v)
    joint_qdd[d + 0] = joint_qdd[d + 0] - c[0]
    joint_qdd[d + 1] = joint_qdd[d + 1] - c[1]
    joint_qdd[d + 2] = joint_qdd[d + 2] - c[2]
''',
    "integrate_generalized_joints": r"""@wp.${decorator}
def ${name}(${index_argument}joint_type: wp.array[int], joint_parent: wp.array[int], joint_child: wp.array[int], joint_q_start: wp.array[int], joint_qd_start: wp.array[int], kinematic_joint_mask: wp.array[int], joint_dof_dim: wp.array2d[int], body_com: wp.array[wp.vec3], joint_X_c: wp.array[wp.transform], joint_q: wp.array[float], joint_qd: wp.array[float], joint_qdd: wp.array[float], dt: float, angular_damping: float, joint_q_new: wp.array[float], joint_qd_new: wp.array[float]):
    index = ${index}
    type = joint_type[index]
    parent = joint_parent[index]
    child = joint_child[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    if kinematic_joint_mask[index] != 0:
        for coord in range(coord_start, joint_q_start[index + 1]):
            joint_q_new[coord] = joint_q[coord]
        for dof in range(dof_start, joint_qd_start[index + 1]):
            joint_qd_new[dof] = joint_qd[dof]
        return
    lin_axis_count = joint_dof_dim[index, 0]
    ang_axis_count = joint_dof_dim[index, 1]
    jcalc_integrate(type, child, body_com, joint_X_c[index], joint_q, joint_qd, joint_qdd, coord_start, dof_start, lin_axis_count, ang_axis_count, dt, angular_damping, parent, joint_q_new, joint_qd_new)
""",
    "update_qdd_from_velocity": r"""@wp.${decorator}
def ${name}(${index_argument}joint_qd: wp.array[float], kinematic_dof_mask: wp.array[int], inv_dt: float, v_new: wp.array[float], joint_qdd: wp.array[float]):
    tid = ${index}
    if kinematic_dof_mask[tid] != 0:
        v_new[tid] = joint_qd[tid]
        joint_qdd[tid] = 0.0
    else:
        joint_qdd[tid] = (v_new[tid] - joint_qd[tid]) * inv_dt
""",
}


def get_source(name: str, *, indexed: bool = False) -> str:
    """Bind one complete law to a kernel lane or an explicit physical index."""
    if name not in _OPERATIONS:
        raise ValueError("Unsupported generalized operation: " + name)
    checked_definitions("generalized_operation_source.py", ("_OPERATIONS", "get_source", "bind"))
    return Template(_OPERATIONS[name]).substitute(
        decorator="func" if indexed else "kernel",
        name="_light_" + name if indexed else name,
        index_argument="physical_index: int, " if indexed else "",
        index="physical_index" if indexed else "wp.tid()",
    )


def bind(name: str, namespace: dict, *, indexed: bool = False):
    """Compile the shared law with its owning module's existing dependencies."""
    source = get_source(name, indexed=indexed)
    filename = f"<generalized-operation-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    scope = dict(namespace)
    exec(compile(source, filename, "exec"), scope)
    return scope["_light_" + name if indexed else name]

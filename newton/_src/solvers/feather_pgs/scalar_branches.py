# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scalar branches: fixed-root articulations whose leaves each carry one coordinate.

A world-fixed root (a FIXED, or zero-coordinate D6, joint to the world) whose
children are one-body prismatic or revolute leaves has a diagonal generalized
inertia, and every leaf is a scalar system: on a fixed root the velocity
product of a one-coordinate body projects to zero on its own axis, and the
leaf's pose, motion axis and body velocity follow from its coordinate and the
model frames alone. This module evaluates such a size group with one thread per
coordinate and no tree plan, so no body or coordinate cap applies. The held-tree
owner composes it in place of a tree group (``mixed_tree_kinetic.py``):

* ``begin`` publishes the input state of articulations whose cache is invalid:
  the root pose, the articulation origin, and each leaf's pose and motion axis.
* ``predict`` forms each leaf's generalized force (augmented drive, passive
  spring and damping, control force, gravity and external wrench), refreshes
  the inverse mass of articulations the mass mask selects, and writes the
  acceleration and the velocity predictor.
* ``finish`` converts the solved velocity, integrates, and publishes the next
  state: coordinates, root and leaf poses, motion axes, body velocities and the
  articulation origin.

Prismatic leaves follow the prismatic linear state's law
(``prismatic_linear_state.py``) operation for operation; revolute leaves follow
ordinary dynamics without evaluating the velocity product, which is zero for
them. The family never writes ``body_q_com``, ``body_v_s``, ``body_a_s``,
``body_f_s``, ``body_I_s`` or the compact inertia terms of its bodies, which no
consumer on its route reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...math.spatial import transform_twist
from ...sim import BodyFlags, JointType
from . import cached_state_binding
from .device_proof import DeviceProof
from .kernels import mul_com_spatial_inertia, transform_com_inertia_terms
from .tree_kinetic_owner import _NUMERIC_FLAGS

# Lanes of the repair launch: one warp per articulation, so a valid cache exits per warp.
_REPAIR_LANES = wp.constant(32)

# Topology arrays the plan and the kernels' index arithmetic rely on.
_TOPOLOGY_FIELDS = (
    "articulation_start",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_q_start",
    "joint_qd_start",
    "joint_dof_dim",
)
# Frames and axes are read live, so they are watched, not frozen: a notification
# that moves one invalidates the caches. Inertial values arrive with numeric
# notifications, which invalidate them anyway.
_LIVE_FIELDS = ("joint_X_p", "joint_X_c", "joint_axis")


@wp.func
def _root_pose(joint_X_p: wp.array[wp.transform], joint_X_c: wp.array[wp.transform], root_joint: int):
    """Pose of a world-fixed root: its joint frames with the identity joint transform between them."""
    return joint_X_p[root_joint] * wp.transform_inverse(joint_X_c[root_joint])


@wp.func
def _prismatic_pose(X_wpj: wp.transform, axis: wp.vec3, q: float, X_c: wp.transform):
    return (X_wpj * wp.transform(axis * q, wp.quat_identity())) * wp.transform_inverse(X_c)


@wp.func
def _revolute_pose(X_wpj: wp.transform, axis: wp.vec3, q: float, X_c: wp.transform):
    return (X_wpj * wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))) * wp.transform_inverse(X_c)


@wp.func
def _prismatic_motion(X_wpj: wp.transform, axis: wp.vec3):
    """Prismatic motion axis: translation only, so it does not depend on the articulation origin."""
    return wp.spatial_vector(wp.quat_rotate(wp.transform_get_rotation(X_wpj), axis), wp.vec3())


@wp.func
def _revolute_motion(X_wpj: wp.transform, origin: wp.vec3, axis: wp.vec3):
    """Revolute motion axis about the articulation origin, as ``jcalc_motion`` forms it."""
    local = wp.transform(wp.transform_get_translation(X_wpj) - origin, wp.transform_get_rotation(X_wpj))
    return transform_twist(local, wp.spatial_vector(wp.vec3(), axis))


@wp.func
def _external_at_origin(
    body: int,
    com_relative: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
    body_flags: wp.array[int],
):
    """External COM wrench moved to the articulation origin; a kinematic body takes none."""
    external = wp.spatial_vector()
    if (body_flags[body] & BodyFlags.KINEMATIC) == 0:
        external = body_f[body]
    force = wp.spatial_top(external)
    torque = wp.spatial_bottom(external)
    return wp.spatial_vector(force, torque + wp.cross(com_relative, force))


@wp.func
def _drive_terms(
    dof: int,
    q: float,
    qd: float,
    dt: float,
    drive_row_by_dof: wp.array[int],
    drive_target_index_by_dof: wp.array[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
):
    """Explicit force ``u0`` and implicit stiffness ``K`` of the augmented drive law.

    The law of ``_prepare_augmented_joint_drives_by_dof``: an undriven
    coordinate and a nonpositive ``K`` contribute neither.
    """
    u0 = float(0.0)
    stiffness = float(0.0)
    if drive_row_by_dof[dof] >= 0:
        ke = joint_target_ke[dof]
        kd = joint_target_kd[dof]
        K = ke * dt * dt + kd * dt
        if K <= 0.0:
            stiffness = 0.0
        else:
            u0 = -(
                ke * (q - joint_target_pos[drive_target_index_by_dof[dof]] + dt * qd)
                + kd * (qd - joint_target_vel[dof])
            )
            effort_limit = joint_effort_limit[dof]
            if effort_limit > 0.0:
                u0 = wp.clamp(u0, -effort_limit, effort_limit)
            stiffness = K
    return u0, stiffness


@wp.func
def _leaf_force(
    motion: wp.spatial_vector,
    body_force: wp.spatial_vector,
    dof: int,
    q: float,
    qd: float,
    joint_f: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_ref: wp.array[float],
    damping: wp.array[float],
):
    """``jcalc_tau`` of one scalar coordinate: body force, control force and passive spring and damping."""
    passive = spring_stiffness[dof] * (spring_ref[dof] - q)
    passive -= damping[dof] * qd
    return -wp.dot(motion, body_force) + joint_f[dof] + passive


@wp.func
def _publish_root(
    art: int,
    root_joint: int,
    X_root: wp.transform,
    joint_child: wp.array[int],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    current_valid: wp.array[int],
):
    root = joint_child[root_joint]
    body_q[root] = X_root
    body_qd[root] = wp.spatial_vector()
    articulation_origin[art] = wp.transform_point(X_root, body_com[root])
    current_valid[art] = 1


@wp.kernel(enable_backward=False)
def repair_scalar_branches(
    n: int,
    group_to_art: wp.array[int],
    dof_joint_offset: wp.array[int],
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    current_valid: wp.array[int],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    joint_q: wp.array[float],
    # outputs
    body_q: wp.array[wp.transform],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
):
    """Publish the input state of articulations with an invalid cache, one warp per articulation.

    Validity is read only: ``finish`` publishes the next state and marks it valid.
    """
    tid = wp.tid()
    group = tid // _REPAIR_LANES
    lane = tid - group * _REPAIR_LANES
    art = group_to_art[group]
    if current_valid[art] != 0:
        return
    root_joint = articulation_start[art]
    root = joint_child[root_joint]
    X_root = _root_pose(joint_X_p, joint_X_c, root_joint)
    origin = wp.transform_point(X_root, body_com[root])
    if lane == 0:
        body_q[root] = X_root
        articulation_origin[art] = origin
    coordinate_start = joint_q_start[root_joint]
    dof_start = articulation_dof_start[art]
    for local in range(lane, n, _REPAIR_LANES):
        joint = root_joint + dof_joint_offset[local]
        dof = dof_start + local
        X_wpj = X_root * joint_X_p[joint]
        axis = joint_axis[dof]
        q = joint_q[coordinate_start + local]
        child = joint_child[joint]
        if joint_type[joint] == JointType.PRISMATIC:
            body_q[child] = _prismatic_pose(X_wpj, axis, q, joint_X_c[joint])
            joint_S_s[dof] = _prismatic_motion(X_wpj, axis)
        else:
            body_q[child] = _revolute_pose(X_wpj, axis, q, joint_X_c[joint])
            joint_S_s[dof] = _revolute_motion(X_wpj, origin, axis)


@wp.kernel(enable_backward=False)
def predict_prismatic_branches(
    n: int,
    group_to_art: wp.array[int],
    dof_joint_offset: wp.array[int],
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_flags: wp.array[int],
    body_f: wp.array[wp.spatial_vector],
    gravity: wp.array[wp.vec3],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    predictor_qd: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_ref: wp.array[float],
    damping: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    mass_update_mask: wp.array[int],
    armature: wp.array2d[float],
    drive_row_by_dof: wp.array[int],
    drive_target_index_by_dof: wp.array[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    dt: float,
    # in/out
    diagonal_inverse_mass: wp.array[float],
    # outputs
    joint_qdd: wp.array[float],
    v_hat: wp.array[float],
):
    """Predict prismatic leaves: the force, the masked inverse-mass refresh, ``qdd`` and ``v_hat``.

    A prismatic motion axis has no angular part, so the gravity and external
    moment arms drop out and the leaf's pose is not read.
    """
    element = wp.tid()
    group = element // n
    local = element - group * n
    art = group_to_art[group]
    root_joint = articulation_start[art]
    joint = root_joint + dof_joint_offset[local]
    dof = articulation_dof_start[art] + local
    child = joint_child[joint]
    motion = joint_S_s[dof]
    mass = body_mass[child]
    force = mass * gravity[0]
    body_force = -wp.spatial_vector(force, wp.vec3()) - _external_at_origin(child, wp.vec3(), body_f, body_flags)
    q = joint_q[joint_q_start[root_joint] + local]
    qd = joint_qd[dof]
    u0, stiffness = _drive_terms(
        dof,
        q,
        qd,
        dt,
        drive_row_by_dof,
        drive_target_index_by_dof,
        joint_target_ke,
        joint_target_kd,
        joint_target_pos,
        joint_target_vel,
        joint_effort_limit,
    )
    value = _leaf_force(motion, body_force, dof, q, qd, joint_f, spring_stiffness, spring_ref, damping)
    value += u0
    inverse = float(0.0)
    if mass_update_mask[art] != 0:
        linear = wp.spatial_top(motion)
        diagonal = mass * wp.dot(linear, linear) + armature[group, local]
        if stiffness > 0.0:
            diagonal += stiffness
        inverse = 1.0 / diagonal
        diagonal_inverse_mass[dof] = inverse
    else:
        inverse = diagonal_inverse_mass[dof]
    qdd = value * inverse
    if kinematic_dof_mask[dof] != 0:
        qdd = 0.0
    joint_qdd[dof] = qdd
    v_hat[dof] = predictor_qd[dof] + qdd * dt


@wp.kernel(enable_backward=False)
def predict_scalar_branches(
    n: int,
    group_to_art: wp.array[int],
    dof_joint_offset: wp.array[int],
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_flags: wp.array[int],
    body_f: wp.array[wp.spatial_vector],
    gravity: wp.array[wp.vec3],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    predictor_qd: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_ref: wp.array[float],
    damping: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    mass_update_mask: wp.array[int],
    armature: wp.array2d[float],
    drive_row_by_dof: wp.array[int],
    drive_target_index_by_dof: wp.array[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    dt: float,
    # in/out
    diagonal_inverse_mass: wp.array[float],
    # outputs
    joint_qdd: wp.array[float],
    v_hat: wp.array[float],
):
    """Predict prismatic and revolute leaves; a revolute inverse mass projects the COM inertia on its axis."""
    element = wp.tid()
    group = element // n
    local = element - group * n
    art = group_to_art[group]
    root_joint = articulation_start[art]
    joint = root_joint + dof_joint_offset[local]
    dof = articulation_dof_start[art] + local
    child = joint_child[joint]
    motion = joint_S_s[dof]
    mass = body_mass[child]
    pose = body_q[child]
    com_relative = wp.transform_point(pose, body_com[child]) - articulation_origin[art]
    force = mass * gravity[0]
    body_force = -wp.spatial_vector(force, wp.cross(com_relative, force)) - _external_at_origin(
        child, com_relative, body_f, body_flags
    )
    q = joint_q[joint_q_start[root_joint] + local]
    qd = joint_qd[dof]
    u0, stiffness = _drive_terms(
        dof,
        q,
        qd,
        dt,
        drive_row_by_dof,
        drive_target_index_by_dof,
        joint_target_ke,
        joint_target_kd,
        joint_target_pos,
        joint_target_vel,
        joint_effort_limit,
    )
    value = _leaf_force(motion, body_force, dof, q, qd, joint_f, spring_stiffness, spring_ref, damping)
    value += u0
    inverse = float(0.0)
    if mass_update_mask[art] != 0:
        projected = float(0.0)
        if joint_type[joint] == JointType.PRISMATIC:
            linear = wp.spatial_top(motion)
            projected = mass * wp.dot(linear, linear)
        else:
            com, inertia_origin = transform_com_inertia_terms(
                wp.transform(com_relative, wp.transform_get_rotation(pose)), mass, body_inertia[child]
            )
            projected = wp.dot(motion, mul_com_spatial_inertia(mass, com, inertia_origin, motion))
        diagonal = projected + armature[group, local]
        if stiffness > 0.0:
            diagonal += stiffness
        inverse = 1.0 / diagonal
        diagonal_inverse_mass[dof] = inverse
    else:
        inverse = diagonal_inverse_mass[dof]
    qdd = value * inverse
    if kinematic_dof_mask[dof] != 0:
        qdd = 0.0
    joint_qdd[dof] = qdd
    v_hat[dof] = predictor_qd[dof] + qdd * dt


@wp.kernel(enable_backward=False)
def finish_prismatic_branches(
    n: int,
    group_to_art: wp.array[int],
    dof_joint_offset: wp.array[int],
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    kinematic_dof_mask: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    dt: float,
    inv_dt: float,
    # in/out
    v_out: wp.array[float],
    # outputs
    joint_qdd: wp.array[float],
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    current_valid: wp.array[int],
):
    """Convert the solved velocity, integrate and publish prismatic leaves and their root."""
    element = wp.tid()
    group = element // n
    local = element - group * n
    art = group_to_art[group]
    root_joint = articulation_start[art]
    joint = root_joint + dof_joint_offset[local]
    dof = articulation_dof_start[art] + local
    coordinate = joint_q_start[root_joint] + local
    qd = joint_qd[dof]
    qdd = float(0.0)
    if kinematic_dof_mask[dof] != 0:
        v_out[dof] = qd
    else:
        qdd = (v_out[dof] - qd) * inv_dt
    joint_qdd[dof] = qdd
    q = joint_q[coordinate]
    q_new = q
    qd_new = qd
    if kinematic_joint_mask[joint] == 0:
        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt
    joint_q_new[coordinate] = q_new
    joint_qd_new[dof] = qd_new
    X_root = _root_pose(joint_X_p, joint_X_c, root_joint)
    X_wpj = X_root * joint_X_p[joint]
    axis = joint_axis[dof]
    child = joint_child[joint]
    body_q[child] = _prismatic_pose(X_wpj, axis, q_new, joint_X_c[joint])
    motion = _prismatic_motion(X_wpj, axis)
    joint_S_s[dof] = motion
    body_qd[child] = motion * qd_new
    if local == 0:
        _publish_root(
            art, root_joint, X_root, joint_child, body_com, body_q, body_qd, articulation_origin, current_valid
        )


@wp.kernel(enable_backward=False)
def finish_scalar_branches(
    n: int,
    group_to_art: wp.array[int],
    dof_joint_offset: wp.array[int],
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    kinematic_dof_mask: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    dt: float,
    inv_dt: float,
    # in/out
    v_out: wp.array[float],
    # outputs
    joint_qdd: wp.array[float],
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    current_valid: wp.array[int],
):
    """Convert, integrate and publish prismatic and revolute leaves and their root."""
    element = wp.tid()
    group = element // n
    local = element - group * n
    art = group_to_art[group]
    root_joint = articulation_start[art]
    joint = root_joint + dof_joint_offset[local]
    dof = articulation_dof_start[art] + local
    coordinate = joint_q_start[root_joint] + local
    qd = joint_qd[dof]
    qdd = float(0.0)
    if kinematic_dof_mask[dof] != 0:
        v_out[dof] = qd
    else:
        qdd = (v_out[dof] - qd) * inv_dt
    joint_qdd[dof] = qdd
    q = joint_q[coordinate]
    q_new = q
    qd_new = qd
    if kinematic_joint_mask[joint] == 0:
        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt
    joint_q_new[coordinate] = q_new
    joint_qd_new[dof] = qd_new
    X_root = _root_pose(joint_X_p, joint_X_c, root_joint)
    X_wpj = X_root * joint_X_p[joint]
    axis = joint_axis[dof]
    child = joint_child[joint]
    if joint_type[joint] == JointType.PRISMATIC:
        body_q[child] = _prismatic_pose(X_wpj, axis, q_new, joint_X_c[joint])
        motion = _prismatic_motion(X_wpj, axis)
        joint_S_s[dof] = motion
        body_qd[child] = motion * qd_new
    else:
        origin = wp.transform_point(X_root, body_com[joint_child[root_joint]])
        pose = _revolute_pose(X_wpj, axis, q_new, joint_X_c[joint])
        body_q[child] = pose
        motion = _revolute_motion(X_wpj, origin, axis)
        joint_S_s[dof] = motion
        velocity = motion * qd_new
        com_relative = wp.transform_point(pose, body_com[child]) - origin
        angular = wp.spatial_bottom(velocity)
        body_qd[child] = wp.spatial_vector(wp.spatial_top(velocity) + wp.cross(angular, com_relative), angular)
    if local == 0:
        _publish_root(
            art, root_joint, X_root, joint_child, body_com, body_q, body_qd, articulation_origin, current_valid
        )


@wp.kernel(enable_backward=False)
def prepare_listed_augmented_drives(
    dofs: wp.array[int],
    drive_row_by_dof: wp.array[int],
    drive_q_index_by_dof: wp.array[int],
    drive_target_index_by_dof: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    dt: float,
    # outputs
    row_K: wp.array[float],
    drive_tau: wp.array[float],
):
    """``_prepare_augmented_joint_drives_by_dof`` over listed DOFs: those whose drives no branch predict folds."""
    dof = dofs[wp.tid()]
    drive_tau[dof] = 0.0
    row = drive_row_by_dof[dof]
    if row < 0:
        return
    u0, stiffness = _drive_terms(
        dof,
        joint_q[drive_q_index_by_dof[dof]],
        joint_qd[dof],
        dt,
        drive_row_by_dof,
        drive_target_index_by_dof,
        joint_target_ke,
        joint_target_kd,
        joint_target_pos,
        joint_target_vel,
        joint_effort_limit,
    )
    row_K[row] = stiffness
    drive_tau[dof] = u0


def branch_topology(model, articulations: np.ndarray, articulation_joint_end: np.ndarray) -> tuple:
    """Check that every articulation is a world-fixed root over one-coordinate leaves.

    Every articulation must hold the same number ``n`` of leaves, laid out as
    joints ``start + 1 .. start + n`` whose coordinates and DOFs follow the
    root's in leaf order: a FIXED or zero-coordinate D6 root joint to the world,
    and prismatic or revolute leaves of one coordinate whose parent is the root
    body and whose body is no joint's parent.

    Returns:
        ``(kinds, reason)``: ``kinds`` is the ``(articulations, n)`` array of leaf
        joint types, or ``None`` with the reason the topology is declined.
    """
    articulations = np.asarray(articulations, dtype=np.int64)
    if articulations.size == 0:
        return None, "No articulations."
    starts = model.articulation_start.numpy().astype(np.int64)
    ends = np.asarray(articulation_joint_end, dtype=np.int64)
    if np.any(ends[articulations] != starts[articulations + 1]):
        return None, "Loop-closing joints keep their articulations on the tree route."
    joint_counts = starts[articulations + 1] - starts[articulations]
    n = int(joint_counts[0]) - 1
    if n < 1 or np.any(joint_counts != n + 1):
        return None, "Scalar branches require the same number of leaves on every root."
    types = model.joint_type.numpy()
    parents = model.joint_parent.numpy()
    children = model.joint_child.numpy()
    dims = model.joint_dof_dim.numpy()
    q_start = model.joint_q_start.numpy().astype(np.int64)
    qd_start = model.joint_qd_start.numpy().astype(np.int64)
    roots = starts[articulations]
    root_bodies = children[roots]
    fixed = types[roots] == int(JointType.FIXED)
    locked = (types[roots] == int(JointType.D6)) & np.all(dims[roots] == 0, axis=1)
    if (
        not np.all(fixed | locked)
        or np.any(parents[roots] != -1)
        or np.any(root_bodies < 0)
        or np.any(q_start[roots + 1] != q_start[roots])
        or np.any(qd_start[roots + 1] != qd_start[roots])
    ):
        return None, "Scalar branches require a FIXED or zero-coordinate D6 root joint to the world."
    leaves = roots[:, None] + 1 + np.arange(n)[None, :]
    leaf_bodies = children[leaves]
    if np.any(leaf_bodies < 0) or np.any(parents[leaves] != root_bodies[:, None]):
        return None, (
            "Scalar branches require every joint below the root to hang from the root body "
            "(chains and composite branches are not admitted)."
        )
    kinds = types[leaves]
    prismatic = kinds == int(JointType.PRISMATIC)
    revolute = kinds == int(JointType.REVOLUTE)
    leaf_dims = dims[leaves]
    one_axis = (prismatic & np.all(leaf_dims == (1, 0), axis=-1)) | (revolute & np.all(leaf_dims == (0, 1), axis=-1))
    if not np.all(one_axis):
        return None, "Scalar branches require one-coordinate prismatic or revolute leaves."
    owners = np.bincount(children[children >= 0], minlength=model.body_count)
    descendants = np.bincount(parents[parents >= 0], minlength=model.body_count)
    if np.any(owners[root_bodies] != 1) or np.any(owners[leaf_bodies] != 1) or np.any(descendants[leaf_bodies] != 0):
        return None, "Scalar branches require leaves without descendants (composite branches are not admitted)."
    offsets = np.arange(n)[None, :]
    if np.any(q_start[leaves] != q_start[roots][:, None] + offsets) or np.any(
        qd_start[leaves] != qd_start[roots][:, None] + offsets
    ):
        return None, "Scalar branches require leaf coordinates in leaf order."
    return kinds, ""


@dataclass(frozen=True)
class ScalarBranchPlan:
    """A size group whose articulations are all scalar branches."""

    size: int
    """Leaves (coordinates) per articulation."""
    group_count: int
    """Articulations in the group."""
    prismatic: bool
    """Every leaf is prismatic: the prismatic kernels apply."""
    kinds: tuple[str, ...]
    """Leaf joint kinds present, sorted."""
    dofs: np.ndarray
    """The group's DOFs, sorted."""


def plan_scalar_branches(solver, *, requested: bool) -> tuple[ScalarBranchPlan | None, str]:
    """Decide whether the solver's diagonal-mass group runs as scalar branches.

    The rule is topology only: the group must be the compact diagonal-mass size
    of the sparse-diagonal response, each of its coordinates must move one body,
    and each articulation must pass :func:`branch_topology`.

    Returns:
        ``(plan, reason)``; ``plan`` is ``None`` when the group stays as it is.
    """
    size = solver._compact_diagonal_mass_size
    if size is None:
        return None, "No diagonal-mass group shares the sparse-diagonal response."
    if not requested:
        return None, "Off: FEATHER_PGS_SCALAR_BRANCHES=1 opts the diagonal-mass group into scalar branches."
    model = solver.model
    if model.requires_grad or model.particle_count:
        return None, "Scalar branches exclude gradients and particles."
    if solver.enable_joint_velocity_limits:
        return None, "Joint velocity limits keep the diagonal-mass group on its ordinary dynamics."
    if size not in solver._direct_diagonal_inertia_sizes:
        return None, "The diagonal-mass group has coordinates that do not each move one body."
    articulations = solver.group_to_art[size].numpy()
    kinds, reason = branch_topology(model, articulations, solver.articulation_joint_end.numpy())
    if kinds is None:
        return None, reason
    offsets = solver._crba_dof_joint_offset_by_size[size].numpy()
    if kinds.shape[1] != size or not np.array_equal(offsets, np.arange(1, size + 1)):
        return None, "The diagonal-mass group's coordinates do not follow its leaves."
    if not solver._parallel_augmented_drive_topology:
        return None, "Scalar branches evaluate the augmented drives of the owned drive-row topology."
    roots = model.articulation_start.numpy()[articulations]
    leaf = np.arange(size)[None, :]
    dofs = model.joint_qd_start.numpy()[roots][:, None] + leaf
    coordinates = model.joint_q_start.numpy()[roots][:, None] + leaf
    driven = solver._augmented_drive_row_by_dof.numpy()[dofs] >= 0
    # The kernels read a driven leaf's coordinate directly; the drive rows must address the same one.
    if np.any(solver._augmented_drive_q_index_by_dof.numpy()[dofs][driven] != coordinates[driven]):
        return None, "The augmented drives of the diagonal-mass group do not read its leaf coordinates."
    names = {int(JointType.PRISMATIC): "prismatic", int(JointType.REVOLUTE): "revolute"}
    present = tuple(sorted({names[int(kind)] for kind in np.unique(kinds)}))
    plan = ScalarBranchPlan(
        size=int(size),
        group_count=len(articulations),
        prismatic=present == ("prismatic",),
        kinds=present,
        dofs=np.sort(dofs.reshape(-1)).astype(np.int32),
    )
    reason = (
        f"Admitted: {plan.group_count} world-fixed roots of {plan.size} one-coordinate {' and '.join(present)} "
        "leaves run as scalar branches; they supersede FEATHER_PGS_PRISMATIC_PUBLICATION and "
        "FEATHER_PGS_PRISMATIC_LINEAR_STATE for these articulations."
    )
    return plan, reason


class ScalarBranchState:
    """Own one size group of scalar branches with the held-tree owner's group interface.

    The owner calls ``begin`` before any factor or force consumer, ``refresh``
    at the mass cadence, ``bind_response`` and ``predict`` at stage 3, and
    ``finish`` to integrate and publish; ``invalidate`` and ``validate_model``
    follow resets and model notifications. ``sparse`` is the group itself: the
    inverse-mass refresh runs inside ``predict``, where the mass mask, the
    drive stiffness and the motion axes of this substep are all at hand.
    """

    # Comparing input banks needs device source tags; repairs here are cheap, so a
    # changed state bank invalidates the cache instead, as ordinary dynamics does
    # (the input validation by identity of TreeKineticState.bank_compare).
    bank_compare = False
    motion_velocity = False
    # Every repair launches: one warp per articulation exits on a valid cache.
    skip_captured_repairs = False

    def __init__(self, solver, plan: ScalarBranchPlan, *, admission=None):
        self.solver = solver
        self.plan = plan
        self.size = plan.size
        self.admission = admission
        self.sparse = self
        model, device = solver.model, solver.model.device
        self.group_count = plan.group_count
        self.group_to_art = solver.group_to_art[plan.size]
        self.dof_joint_offset = solver._crba_dof_joint_offset_by_size[plan.size]
        self._cache_to_art = self.group_to_art.reshape((self.group_count, 1))
        # The family holds no geometry cache; invalidation clears this flag only.
        self._geometry_valid = wp.zeros(self.group_count, dtype=int, device=device)
        self.repair_kernel = repair_scalar_branches
        self.predictor_kernel = predict_prismatic_branches if plan.prismatic else predict_scalar_branches
        self.finish_kernel = finish_prismatic_branches if plan.prismatic else finish_scalar_branches
        self.lane_timing = {}
        # The predict folds the branches' augmented drives, so the drive
        # preparation runs only over the other DOFs.
        others = np.setdiff1d(np.arange(model.joint_dof_count, dtype=np.int32), plan.dofs).astype(np.int32)
        self.other_drive_dofs = wp.array(others, dtype=int, device=device) if others.size else None
        self._plan_dimensions = self._dimensions()
        self._proof = DeviceProof(
            [(name, getattr(model, name)) for name in (*_TOPOLOGY_FIELDS, *_LIVE_FIELDS)]
            + [("group_to_art", self.group_to_art), ("art_to_world", solver.art_to_world)],
            device,
            watched=_LIVE_FIELDS,
        )

    def _dimensions(self):
        model = self.solver.model
        return (
            self.solver.world_count,
            model.articulation_count,
            model.body_count,
            model.joint_count,
            model.joint_coord_count,
            model.joint_dof_count,
        )

    def record(self) -> dict:
        """Host-only route record of this group, in the tree groups' schema (see ``path_selection``).

        Predict and finish run one thread per coordinate (``lanes``), repair one
        warp per articulation; there is no tree layout, lane timing or block packing.
        """
        size = int(self.size)
        return {
            "operator": type(self).__name__,
            "size": size,
            "group_count": int(self.group_count),
            "bodies": size + 1,
            "dofs": size,
            "lanes": size,
            "trees_per_block": None,
            "block_dim": None,
            "roots": [0],
            "root_coordinates": [0],
            # A diagonal-mass family, negated as in the tree groups' records.
            "root_families": [-size],
            "chain_scan": False,
            "chain_scan_requested": False,
            "response": "diagonal",
            "input_validation": "content" if self.bank_compare else "identity",
            # Every repair launches, one warp per articulation, and none is classified or compared.
            "captured_repair_skip": bool(self.skip_captured_repairs),
            "classify_first": False,
            "captured_repairs": [],
            "motion_velocity": False,
            "publishes_twists": False,
            "lane_choice": "static",
            "kinds": list(self.plan.kinds),
            # The predict evaluates the branches' drives; the drive kernel runs over the other DOFs only.
            "drive_dofs": 0 if self.other_drive_dofs is None else int(self.other_drive_dofs.shape[0]),
            "kernels": {
                "repair": self.repair_kernel.key,
                "finish": self.finish_kernel.key,
                "predict": self.predictor_kernel.key,
                "bank": None,
                "bank_repair": None,
                "classify": None,
                "listed_repair": None,
            },
        }

    def validate_model(self, flags=None):
        """Reject topology changes; frames, axes and inertial values are read live.

        Numeric notifications cannot change the topology and are accepted
        outright. Other notifications (Isaac Lab re-poses fixed roots through
        ``joint_X_p`` with ``JOINT_PROPERTIES`` on resets) compare the frozen
        topology arrays on the device with one small readback, which also
        watches the joint frames and axes.

        Returns:
            Whether the proof ran and found the topology, frames and axes
            unchanged (see :meth:`TreeKineticState.validate_model`); ``False``
            after a moved frame or axis, which the caller invalidates, and for
            numeric notifications, which skip the proof.
        """
        if flags is not None and int(flags) != 0 and int(flags) & ~_NUMERIC_FLAGS == 0:
            return False
        if self._dimensions() != self._plan_dimensions:
            raise RuntimeError("Scalar branch dimensions changed; reconstruct and recapture")

        def current(label):
            if label == "group_to_art":
                return self.group_to_art
            if label == "art_to_world":
                return self.solver.art_to_world
            return getattr(self.solver.model, label)

        return not self._proof.validate(current, "Scalar branch topology")

    def forget_continuity(self):
        """Nothing to forget: every repair launches (``skip_captured_repairs`` is ``False``)."""

    def invalidate(self, world_mask=None):
        """Invalidate the cached state of every owned articulation in selected worlds."""
        cached_state_binding.invalidate(
            self.solver.world_count,
            world_mask,
            self._cache_to_art,
            self.solver.art_to_world,
            self.solver._fk_id_cache_valid,
            self._geometry_valid,
        )

    def _validate_call(self, state, state_aug, dt):
        solver = self.solver
        if state.requires_grad or state_aug is not solver:
            raise RuntimeError("Scalar branches exclude gradient and alternate augmented ownership")
        if self.admission is not None and not self.admission(solver):
            raise RuntimeError("Scalar branch configuration changed; reconstruct and recapture")
        if solver.enable_joint_velocity_limits:
            raise RuntimeError("Scalar branches exclude joint velocity limits; reconstruct and recapture")
        if dt is None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Scalar branches require a positive finite step duration")
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            value = getattr(state, name)
            if value.device != solver.model.device or not value.is_contiguous:
                raise RuntimeError("Scalar branches require contiguous same-device state arrays")

    def begin(self, state_in, state_aug, stage3_qd, dt, global_refresh, *, state_out=None):
        """Publish the input state of articulations whose cache is invalid.

        ``state_out`` (the step's output bank) is not compared: the group
        validates its input bank by identity (``bank_compare`` is ``False``).
        """
        del stage3_qd, global_refresh, state_out
        self._validate_call(state_in, state_aug, dt)
        solver, model = self.solver, self.solver.model
        wp.launch(
            self.repair_kernel,
            dim=self.group_count * _REPAIR_LANES,
            inputs=[
                self.size,
                self.group_to_art,
                self.dof_joint_offset,
                model.articulation_start,
                solver.articulation_dof_start,
                solver._fk_id_cache_valid,
                model.joint_type,
                model.joint_child,
                model.joint_q_start,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.body_com,
                state_in.joint_q,
            ],
            outputs=[state_in.body_q, state_aug.joint_S_s, solver.articulation_origin],
            block_dim=128,
            device=model.device,
        )

    def launch_other_drives(self, state_in, state_aug, control, dt):
        """Prepare the augmented drives of the DOFs outside the branches, whose ``predict`` folds its own.

        Writes the drive rows' ``K`` and the drive forces into ``joint_tau`` as
        ``_prepare_augmented_joint_drives_by_dof`` does, for the listed DOFs only.
        """
        if self.other_drive_dofs is None:
            return
        solver, model = self.solver, self.solver.model
        wp.launch(
            prepare_listed_augmented_drives,
            dim=self.other_drive_dofs.shape[0],
            inputs=[
                self.other_drive_dofs,
                solver._augmented_drive_row_by_dof,
                solver._augmented_drive_q_index_by_dof,
                solver._augmented_drive_target_index_by_dof,
                state_in.joint_q,
                state_in.joint_qd,
                model.joint_target_ke,
                model.joint_target_kd,
                control.joint_target_q,
                control.joint_target_qd,
                model.joint_effort_limit,
                dt,
            ],
            outputs=[solver.aug_row_K, state_aug.joint_tau],
            block_dim=256,
            device=model.device,
        )

    def refresh(self, state_aug):
        """Nothing to do here: ``predict`` refreshes the inverse masses the mass mask selects."""

    def bind_response(self):
        """The inverse diagonal is the solver's and never rebinds."""

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        """Form each leaf's force and refresh, solve and predict its coordinate."""
        self._validate_call(state_in, state_aug, dt)
        solver, model = self.solver, self.solver.model
        inputs = [
            self.size,
            self.group_to_art,
            self.dof_joint_offset,
            model.articulation_start,
            solver.articulation_dof_start,
        ]
        if self.plan.prismatic:
            inputs += [model.joint_child, model.joint_q_start, state_aug.joint_S_s]
        else:
            inputs += [
                model.joint_type,
                model.joint_child,
                model.joint_q_start,
                state_aug.joint_S_s,
                solver.articulation_origin,
                state_in.body_q,
                model.body_com,
            ]
        inputs.append(model.body_mass)
        if not self.plan.prismatic:
            inputs.append(model.body_inertia)
        inputs += [
            model.body_flags,
            state_in.body_f,
            model.gravity,
            control.joint_f,
            state_in.joint_q,
            state_in.joint_qd,
            stage3_qd,
            solver._passive_spring_stiffness,
            solver._passive_spring_ref,
            solver._passive_joint_damping,
            solver._kinematic_dof_mask,
            solver.mass_update_mask,
            solver.R_by_size[self.size],
            solver._augmented_drive_row_by_dof,
            solver._augmented_drive_target_index_by_dof,
            model.joint_target_ke,
            model.joint_target_kd,
            control.joint_target_q,
            control.joint_target_qd,
            model.joint_effort_limit,
            dt,
            solver._diagonal_inverse_mass,
        ]
        wp.launch(
            self.predictor_kernel,
            dim=self.group_count * self.size,
            inputs=inputs,
            outputs=[state_aug.joint_qdd, solver.v_hat],
            device=model.device,
        )

    def finish(self, state_in, state_aug, state_out, dt, next_refresh):
        """Convert the solved velocity, integrate once and publish the next state."""
        del next_refresh
        self._validate_call(state_in, state_aug, dt)
        self._validate_call(state_out, state_aug, dt)
        if not cached_state_binding.disjoint_states(state_in, state_out):
            raise RuntimeError("Scalar branch integration requires non-overlapping generalized state banks")
        solver, model = self.solver, self.solver.model
        inputs = [
            self.size,
            self.group_to_art,
            self.dof_joint_offset,
            model.articulation_start,
            solver.articulation_dof_start,
        ]
        if not self.plan.prismatic:
            inputs.append(model.joint_type)
        inputs += [
            model.joint_child,
            model.joint_q_start,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.body_com,
            solver._kinematic_dof_mask,
            solver._kinematic_joint_mask,
            state_in.joint_q,
            state_in.joint_qd,
            dt,
            1.0 / dt,
            solver.v_out,
        ]
        wp.launch(
            self.finish_kernel,
            dim=self.group_count * self.size,
            inputs=inputs,
            outputs=[
                state_aug.joint_qdd,
                state_out.joint_q,
                state_out.joint_qd,
                state_out.body_q,
                state_out.body_qd,
                state_aug.joint_S_s,
                solver.articulation_origin,
                solver._fk_id_cache_valid,
            ],
            device=model.device,
        )

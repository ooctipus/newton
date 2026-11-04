# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scalar branches: fixed-root articulations whose leaves each carry one prismatic coordinate.

A world-fixed root (a FIXED, or zero-coordinate D6, joint to the world) whose
children are one-body prismatic leaves has a diagonal generalized inertia, and
every leaf is a scalar system: on a fixed root a prismatic body has no velocity
product, and the leaf's pose, motion axis and body velocity follow from its
coordinate and the model frames alone. This module evaluates such a size group with one thread per
coordinate and no tree plan, so no body or coordinate cap applies. By default the
held-tree owner composes it in place of a tree group (``held_tree.py``)
wherever :func:`plan_scalar_branches` admits the group:

* ``begin`` publishes the input state of articulations whose cache is invalid:
  the root pose, the articulation origin, and each leaf's pose and motion axis.
* ``predict`` forms each leaf's generalized force (augmented drive, passive
  spring and damping, control force, gravity and external wrench), refreshes
  the inverse mass of articulations the mass mask selects, and writes the
  acceleration and the velocity predictor.
* ``finish`` converts the solved velocity, integrates, and publishes the next
  state: coordinates, root and leaf poses, motion axes, body velocities and the
  articulation origin.

The leaves follow the linear-state law of a one-body prismatic branch operation
for operation (the scalar-branch tests keep that law as their oracle). Revolute
leaves, whose inertia depends on their axis and child frame, run in the held
tree's diagonal-mass group. The family never writes ``body_q_com``,
``body_v_s``, ``body_a_s``, ``body_f_s``, ``body_I_s`` or the compact inertia
terms of its bodies, which no consumer on its route reads.

A leaf's inverse mass, ``1 / (m |S|^2 + armature + K)``, depends on the
inertial values, the armature and the drive's ``K = ke dt^2 + kd dt``, never on
the pose, so a root re-pose leaves it unchanged. The inverse mass refreshes wherever the solver's
mass mask selects (the update cadence plus the requests that resets and
inertial, DOF and body notifications raise), as the linear-state law does:
that keeps prismatic leaves bitwise with that law and follows a changed step
duration, which no notification announces. M6 timed prototype
predict and finish kernels at 489 us per substep with this cadence over 16384
keyboard-sized worlds, against 500 us with a refresh on every substep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType
from . import cached_state_binding
from .factor.fields import DriveRows, hold_drive_row
from .factor.kernels import Clock, clock_changed
from .spatial_laws import (
    axis_transform,
    coordinate_force,
    drive_force,
    drive_step_force,
    drive_stiffness,
    free_drive_force,
    joint_frame_pose,
    limited_drive_force,
    motion_axis,
    spring_damper_stiffness,
    transform_compose,
    wrench_shift,
)
from .tree_kinetic_owner import proof_runs

# Lanes of the repair launch: one warp per articulation, so a valid cache exits per warp.
_REPAIR_LANES = wp.constant(32)
# Threads per block of the repair launch, four articulations per block; the predict,
# finish and listed drive launches take Warp's default block. tools/fpgs_bench/kernel_sass.py
# compiles the repair at this block.
_REPAIR_BLOCK_DIM = 128


@wp.func
def _root_pose(joint_X_p: wp.array[wp.transform], joint_X_c: wp.array[wp.transform], root_joint: int):
    """Pose of a world-fixed root: its joint frames with the identity joint transform between them."""
    return joint_frame_pose(joint_X_p[root_joint], wp.transform_identity(), joint_X_c[root_joint])


@wp.func
def _leaf_pose(X_root: wp.transform, X_p: wp.transform, angular: bool, axis: wp.vec3, q: float, X_c: wp.transform):
    """A one-coordinate leaf's pose: its relative pose composed with its root's, as the pose scan composes it."""
    return transform_compose(X_root, joint_frame_pose(X_p, axis_transform(angular, axis, q), X_c))


@wp.func
def _leaf_motion(X_wpj: wp.transform, origin: wp.vec3, angular: bool, axis: wp.vec3):
    """A one-coordinate leaf's motion column about the articulation origin (its joint anchor ``X_wpj``)."""
    local = wp.transform(wp.transform_get_translation(X_wpj) - origin, wp.transform_get_rotation(X_wpj))
    return motion_axis(local, axis, angular)


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
    return wrench_shift(external, com_relative)


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
    damping: wp.array[float],
):
    """Explicit force ``u0`` and implicit stiffness ``K`` of a coordinate's augmented drive
    (:func:`.spatial_laws.drive_stiffness`, :func:`.spatial_laws.drive_force`): a coordinate without a row or with a
    nonpositive ``K`` contributes neither."""
    u0 = float(0.0)
    stiffness = float(0.0)
    if drive_row_by_dof[dof] >= 0:
        ke = joint_target_ke[dof]
        kd = joint_target_kd[dof]
        K = drive_stiffness(ke, kd, damping[dof], dt)
        if K > 0.0:
            u0 = drive_force(ke, kd, q, qd, joint_target_pos[drive_target_index_by_dof[dof]], joint_target_vel[dof], dt)
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
    """One scalar coordinate's generalized force (:func:`.spatial_laws.coordinate_force`)."""
    return coordinate_force(
        motion, body_force, joint_f[dof], spring_stiffness[dof], spring_ref[dof], damping[dof], q, qd
    )


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
    clock: wp.array[Clock],
    dt: float,
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
    """Publish the input state of articulations with an invalid cache, one warp per articulation; a changed step
    duration ``dt`` (:func:`.factor.kernels.clock_changed`) invalidates every cache.

    Validity is read only: ``finish`` publishes the next state and marks it valid.
    """
    tid = wp.tid()
    group = tid // _REPAIR_LANES
    lane = tid - group * _REPAIR_LANES
    art = group_to_art[group]
    if current_valid[art] != 0 and clock_changed(clock[0], dt) == 0:
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
        body_q[child] = _leaf_pose(X_root, joint_X_p[joint], False, axis, q, joint_X_c[joint])
        joint_S_s[dof] = _leaf_motion(X_wpj, origin, False, axis)


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
    armature: wp.array[float],
    drive_row_by_dof: wp.array[int],
    drive_target_index_by_dof: wp.array[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    dt: float,
    v_out: wp.array[float],
    # in/out
    diagonal_inverse_mass: wp.array[float],
    row_K: wp.array[float],
    # outputs
    joint_qdd: wp.array[float],
    v_hat: wp.array[float],
):
    """Predict prismatic leaves: the force, the masked inverse-mass refresh, ``qdd`` and ``v_hat``.

    A prismatic motion axis has no angular part, so the gravity and external
    moment arms drop out and the leaf's pose is not read. A leaf is one coordinate, so it decides saturation here
    (:func:`.spatial_laws.drive_step_force`) with one more division: its ``K`` changing refreshes its own inverse mass.
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
    linear = wp.spatial_top(motion)
    leaf_inertia = mass * wp.dot(linear, linear) + armature[dof]
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
        damping,
    )
    value = _leaf_force(motion, body_force, dof, q, qd, joint_f, spring_stiffness, spring_ref, damping)
    refresh = mass_update_mask[art] != 0
    row = drive_row_by_dof[dof]
    if row >= 0:
        if row_K[row] != stiffness:
            refresh = True
    inverse = float(0.0)
    if refresh:
        inverse = 1.0 / (leaf_inertia + stiffness)
    else:
        inverse = diagonal_inverse_mass[dof]
    passive = spring_damper_stiffness(0.0, damping[dof], 0.0, dt)
    drive = stiffness - passive
    limit = joint_effort_limit[dof]
    if drive > 0.0 and limit > 0.0:
        limited = limited_drive_force(u0, limit)
        trial = (value + limited) * inverse
        step_force = drive_step_force(u0, limited, drive, trial, leaf_inertia + passive)
        if wp.abs(step_force) > limit:
            u0 = wp.sign(step_force) * limit
            stiffness = passive
            inverse = 1.0 / (leaf_inertia + stiffness)
            refresh = True
        elif u0 != limited:
            rows = (v_out[dof] - v_hat[dof]) / dt
            u0 = free_drive_force(u0, limited, drive, trial, rows, leaf_inertia + passive, limit)
    if refresh:
        diagonal_inverse_mass[dof] = inverse
    if row >= 0:
        row_K[row] = stiffness
    qdd = (value + u0) * inverse
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
    body_q[child] = _leaf_pose(X_root, joint_X_p[joint], False, axis, q_new, joint_X_c[joint])
    motion = _leaf_motion(X_wpj, wp.vec3(), False, axis)
    joint_S_s[dof] = motion
    body_qd[child] = motion * qd_new
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
    damping: wp.array[float],
    max_dofs: int,
    dt: float,
    # outputs
    rows: DriveRows,
    drive_tau: wp.array[float],
):
    """``tree_factor.prepare_augmented_drives`` over listed DOFs: those whose drives no branch predict folds."""
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
        damping,
    )
    passive = spring_damper_stiffness(0.0, damping[dof], 0.0, dt)
    drive_tau[dof] = hold_drive_row(rows, row, row // max_dofs, dof, u0, stiffness, passive, joint_effort_limit[dof])


def branch_topology(model, articulations: np.ndarray, articulation_joint_end: np.ndarray) -> tuple:
    """Check that every articulation is a world-fixed root over one-coordinate prismatic leaves.

    Every articulation must hold the same number ``n`` of leaves, laid out as
    joints ``start + 1 .. start + n`` whose coordinates and DOFs follow the
    root's in leaf order: a FIXED or zero-coordinate D6 root joint to the world,
    and prismatic leaves of one coordinate whose parent is the root body and
    whose body is no joint's parent.

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
        return None, "Joints outside every tree keep their articulations on the tree route."
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
    if not np.all((kinds == int(JointType.PRISMATIC)) & np.all(dims[leaves] == (1, 0), axis=-1)):
        return None, (
            "Scalar branches require one-coordinate prismatic leaves (revolute leaves run in the held tree's "
            "diagonal-mass group)."
        )
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


@dataclass(frozen=True, eq=False)
class ScalarBranchPlan:
    """A size group whose articulations are all scalar branches.

    Plans compare and hash by identity: ``dofs`` is an array.
    """

    size: int
    """Leaves (coordinates) per articulation."""
    group_count: int
    """Articulations in the group."""
    dofs: np.ndarray
    """The group's DOFs, sorted."""


def plan_scalar_branches(solver, *, size: int | None = None) -> tuple[ScalarBranchPlan | None, str]:
    """Decide whether the solver's diagonal-mass group runs as scalar branches.

    The rule is topology only: the group must be the compact diagonal-mass size
    of the sparse-diagonal response (``size``, by default the solver's; the
    response's admission passes its candidate), each of its coordinates must
    move one body, and each articulation must pass :func:`branch_topology`.

    Returns:
        ``(plan, reason)``; ``plan`` is ``None`` when the group stays as it is.
    """
    if size is None:
        size = solver._compact_diagonal_mass_size
    if size is None:
        return None, "No diagonal-mass group shares the sparse-diagonal response."
    if solver._route_pins.get("star") == "held":
        return None, "A test pin keeps the diagonal-mass group in the held tree (star='held')."
    model = solver.model
    if model.particle_count:
        return None, "Scalar branches exclude particles."
    if solver.options.enable_joint_velocity_limits:
        return None, "Joint velocity limits keep the diagonal-mass group on its ordinary dynamics."
    if size not in solver.forest.direct_diagonal_inertia_sizes:
        return None, "The diagonal-mass group has coordinates that do not each move one body."
    articulations = solver.forest.group_to_art[size].numpy()
    kinds, reason = branch_topology(model, articulations, solver.forest.articulation_joint_end.numpy())
    if kinds is None:
        return None, reason
    offsets = solver.forest.crba_dof_joint_offset_by_size[size].numpy()
    if kinds.shape[1] != size or not np.array_equal(offsets, np.arange(1, size + 1)):
        return None, "The diagonal-mass group's coordinates do not follow its leaves."
    if not solver.factor.parallel_drives:
        return None, "Scalar branches evaluate the augmented drives of the owned drive-row topology."
    roots = model.articulation_start.numpy()[articulations]
    leaf = np.arange(size)[None, :]
    dofs = model.joint_qd_start.numpy()[roots][:, None] + leaf
    coordinates = model.joint_q_start.numpy()[roots][:, None] + leaf
    driven = solver.factor.drive_row_by_dof.numpy()[dofs] >= 0
    # The kernels read a driven leaf's coordinate directly; the drive rows must address the same one.
    if np.any(solver.factor.drive_q_index_by_dof.numpy()[dofs][driven] != coordinates[driven]):
        return None, "The augmented drives of the diagonal-mass group do not read its leaf coordinates."
    plan = ScalarBranchPlan(
        size=int(size),
        group_count=len(articulations),
        dofs=np.sort(dofs.reshape(-1)).astype(np.int32),
    )
    reason = (
        f"Admitted: {plan.group_count} world-fixed roots of {plan.size} one-coordinate prismatic "
        "leaves run as scalar branches."
    )
    return plan, reason


class ScalarBranchState:
    """Own one size group of scalar branches with the held-tree owner's group interface.

    The owner calls ``begin`` before any factor or force consumer, ``refresh``
    at the mass cadence, ``predict`` at stage 3, and
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
        """Bind the group's kernels; the owner's tree groups hold the topology proof."""
        self.solver = solver
        self.plan = plan
        self.size = plan.size
        self.admission = admission
        self.sparse = self
        model, device = solver.model, solver.model.device
        self.group_count = plan.group_count
        self.group_to_art = solver.forest.group_to_art[plan.size]
        self.dof_joint_offset = solver.forest.crba_dof_joint_offset_by_size[plan.size]
        self._cache_to_art = self.group_to_art.reshape((self.group_count, 1))
        # The family holds no geometry cache; invalidation clears this flag only.
        self._geometry_valid = wp.zeros(self.group_count, dtype=int, device=device)
        self.repair_kernel = repair_scalar_branches
        self.predictor_kernel = predict_prismatic_branches
        self.finish_kernel = finish_prismatic_branches
        # The predict folds the branches' augmented drives, so the drive
        # preparation runs only over the other DOFs.
        other = np.ones(model.joint_dof_count, dtype=bool)
        other[plan.dofs[(plan.dofs >= 0) & (plan.dofs < model.joint_dof_count)]] = False
        others = np.flatnonzero(other).astype(np.int32)
        self.other_drive_dofs = wp.array(others, dtype=int, device=device) if others.size else None
        self._plan_dimensions = self._dimensions()

    def _dimensions(self):
        model = self.solver.model
        return (
            self.solver.forest.world_count,
            model.articulation_count,
            model.body_count,
            model.joint_count,
            model.joint_coord_count,
            model.joint_dof_count,
        )

    def validate_model(self, flags=None):
        """Reject dimension changes; frames and inertial values are read live.

        The held-tree owner that composes the group proves the topology, the
        joint axes and the joint frames through its tree groups beside this
        one (Isaac Lab re-poses fixed roots through ``joint_X_p`` with
        ``JOINT_PROPERTIES`` on resets). A changed axis raises there, as for
        the tree groups.

        Returns:
            ``False`` for numeric notifications, which skip every proof (see
            :meth:`TreeKineticState.validate_model`); otherwise ``True``: the
            owner's tree groups prove the same model arrays, and the owner keeps
            the caches only when every proof passes.
        """
        if not proof_runs(flags):
            return False
        if self._dimensions() != self._plan_dimensions:
            raise RuntimeError("Scalar branch dimensions changed; reconstruct and recapture")
        return True

    def forget_continuity(self):
        """Nothing to forget: every repair launches (``skip_captured_repairs`` is ``False``)."""

    def invalidate(self, world_mask=None):
        """Invalidate the cached state of every owned articulation in selected worlds."""
        cached_state_binding.invalidate(
            self.solver.forest.world_count,
            world_mask,
            self._cache_to_art,
            self.solver.forest.art_to_world,
            self.solver.factor.fk_id_cache_valid,
            self._geometry_valid,
        )

    def _validate_call(self, state, state_aug, dt):
        solver = self.solver
        if self.admission is not None and not self.admission(solver):
            raise RuntimeError("Scalar branch configuration changed; reconstruct and recapture")
        if solver.options.enable_joint_velocity_limits:
            raise RuntimeError("Scalar branches exclude joint velocity limits; reconstruct and recapture")
        if dt is None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Scalar branches require a positive finite step duration")
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            value = getattr(state, name)
            if value.device != solver.model.device or not value.is_contiguous:
                raise RuntimeError("Scalar branches require contiguous same-device state arrays")

    def begin(self, state_in, state_aug, stage3_qd, dt, *, state_out=None):
        """Publish the input state of articulations whose cache is invalid.

        ``state_out`` (the step's output bank) is not compared: the group
        validates its input bank by identity (``bank_compare`` is ``False``).
        """
        del stage3_qd, state_out
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
                solver.forest.articulation_dof_start,
                solver.factor.fk_id_cache_valid,
                solver.factor.clock,
                dt,
                model.joint_child,
                model.joint_q_start,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.body_com,
                state_in.joint_q,
            ],
            outputs=[state_in.body_q, state_aug.joint_S_s, solver.factor.articulation_origin],
            block_dim=_REPAIR_BLOCK_DIM,
            device=model.device,
        )

    def launch_other_drives(self, state_in, state_aug, control, dt):
        """Prepare the augmented drives of the DOFs outside the branches, whose ``predict`` folds its own.

        Writes the drive rows' ``K`` and the drive forces into ``joint_tau`` as
        :func:`.tree_factor.prepare_augmented_drives` does, for the listed DOFs only. The step's stage list runs it
        only where such DOFs exist (``other_drive_dofs`` is not ``None``).
        """
        solver, model = self.solver, self.solver.model
        wp.launch(
            prepare_listed_augmented_drives,
            dim=self.other_drive_dofs.shape[0],
            inputs=[
                self.other_drive_dofs,
                solver.factor.drive_row_by_dof,
                solver.factor.drive_q_index_by_dof,
                solver.factor.drive_target_index_by_dof,
                state_in.joint_q,
                state_in.joint_qd,
                model.joint_target_ke,
                model.joint_target_kd,
                control.joint_target_q,
                control.joint_target_qd,
                model.joint_effort_limit,
                solver.factor.passive_joint_damping,
                solver.forest.articulation_max_dofs,
                dt,
            ],
            outputs=[solver.factor.drive_rows(solver.v_hat, solver.v_out), state_aug.joint_tau],
            block_dim=256,
            device=model.device,
        )

    def refresh(self, state_aug):
        """Nothing to do here: ``predict`` refreshes the inverse masses the mass mask selects."""

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        """Form each leaf's force and refresh, solve and predict its coordinate, deciding its drive's saturation."""
        self._validate_call(state_in, state_aug, dt)
        solver, model = self.solver, self.solver.model
        inputs = [
            self.size,
            self.group_to_art,
            self.dof_joint_offset,
            model.articulation_start,
            solver.forest.articulation_dof_start,
        ]
        inputs += [
            model.joint_child,
            model.joint_q_start,
            state_aug.joint_S_s,
            model.body_mass,
            model.body_flags,
            state_in.body_f,
            model.gravity,
            control.joint_f,
            state_in.joint_q,
            state_in.joint_qd,
            stage3_qd,
            solver.factor.passive_spring_stiffness,
            solver.factor.passive_spring_ref,
            solver.factor.passive_joint_damping,
            solver.factor.kinematic_dof_mask,
            solver.factor.mass_update_mask,
            solver.factor.dof_armature,
            solver.factor.drive_row_by_dof,
            solver.factor.drive_target_index_by_dof,
            model.joint_target_ke,
            model.joint_target_kd,
            control.joint_target_q,
            control.joint_target_qd,
            model.joint_effort_limit,
            dt,
            solver.v_out,
            solver.factor.diagonal_inverse_mass,
            solver.factor.aug_row_K,
        ]
        wp.launch(
            self.predictor_kernel,
            dim=self.group_count * self.size,
            inputs=inputs,
            outputs=[state_aug.joint_qdd, solver.v_hat],
            device=model.device,
        )

    def finish(self, state_in, state_aug, state_out, dt):
        """Convert the solved velocity, integrate once and publish the next state."""
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
            solver.forest.articulation_dof_start,
        ]
        inputs += [
            model.joint_child,
            model.joint_q_start,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.body_com,
            solver.factor.kinematic_dof_mask,
            solver.factor.kinematic_joint_mask,
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
                solver.factor.articulation_origin,
                solver.factor.fk_id_cache_valid,
            ],
            device=model.device,
        )

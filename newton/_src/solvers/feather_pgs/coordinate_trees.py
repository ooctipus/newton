# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held tree's one-coordinate class: a compact diagonal family whose trees are one body and one coordinate each.

A world-fixed root (a FIXED, or zero-coordinate D6, joint to the world) whose children are one-body leaves of one
coordinate each (prismatic, revolute or a one-axis D6) is a compact diagonal family whose trees are each one body: every
leaf is its own one-coordinate tree on the root's fixed frame. Such a tree's mass ``H_ii = S^T I S`` is constant (a slide
along its own axis leaves ``m |S|^2``, a turn about its own axis leaves ``S^T I S``), its bias is gravity's and the
external wrench's projection (a one-coordinate tree has no velocity product: ``S^T (S qd x* I S qd) = 0``), and its walk
is one division (the compact inverse). So the class's kernels hold a thread a coordinate across every articulation of
the group, with no tree block, no scan and no collection, composed from the held tree's own laws
(:func:`.spatial_laws.joint_frame_pose`, :func:`.spatial_laws.axis_transform` and :func:`.spatial_laws.motion_axis`,
the one-axis cases of ``joint_transform`` and ``joint_motion``, :func:`.spatial_laws.coordinate_force`, the drive laws):

* ``repair`` publishes the input state of articulations whose cache is invalid: the root pose, the articulation origin,
  and each leaf's pose and motion column.
* ``predict`` forms each leaf's generalized force (augmented drive, passive spring and damping, control force, gravity
  and external wrench), refreshes the inverse mass of articulations the mass mask selects, decides its drive's
  saturation and writes the acceleration and the velocity predictor.
* ``finish`` converts the solved velocity, integrates, and publishes the next state: coordinates, root and leaf poses,
  motion columns, body velocities and the articulation origin.

The kernels are generated from the family's data (:func:`class_kernels`): a family whose leaves all slide reads no pose
in its predict (a slide's column has no turn, so the moment arms of gravity and the external wrench drop out and its
mass is ``m |S|^2``); a family with turning leaves reads each leaf's kind from its joint's angular axis count and forms
the moments about the articulation origin and ``S^T I S`` from the body's inertia there. Validity is a word an
articulation: one launch holds every coordinate of an articulation, the repair reads it, the finish sets it. The class
never writes ``body_q_com``, ``body_v_s``, ``body_a_s``, ``body_f_s``, ``body_I_s`` or the compact inertia terms of its
bodies, which no consumer on its route reads.

A leaf's inverse mass, ``1 / (H_ii + armature + K)``, depends on the inertial values, the armature and the drive's
``K = ke dt^2 + kd dt``, never on the pose, so a root re-pose leaves it unchanged. It refreshes wherever the solver's mass
mask selects (the update cadence plus the requests that resets and inertial, DOF and body notifications raise), and
wherever a leaf's ``K`` changed: a one-coordinate tree decides its drive in its predict, one more division, so the drive
stage takes the other coordinates (:attr:`.held_tree.HeldPlan.drive_dofs`). M6 timed prototype predict and finish
kernels at 489 us per substep with this cadence over 16384 keyboard-sized worlds, against 500 us with a refresh on every
substep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType
from . import cached_state_binding
from .factor.fields import drive_terms
from .factor.kernels import Clock, clock_changed
from .kernel_template import python_kernel
from .spatial_laws import (
    axis_transform,
    coordinate_force,
    drive_step_force,
    free_drive_force,
    inertia_apply,
    inertia_at,
    joint_frame_pose,
    limited_drive_force,
    motion_axis,
    point_velocity,
    spring_damper_stiffness,
    transform_compose,
    wrench_shift,
)
from .tree_kinetic_owner import proof_runs

# Lanes of the repair launch: one warp per articulation, so a valid cache exits per warp.
_REPAIR_LANES = wp.constant(32)
# Threads per block of the repair launch, four articulations per block; the predict and finish launches take Warp's
# default block. tools/fpgs_bench/kernel_sass.py compiles the repair at this block.
REPAIR_BLOCK_DIM = 128


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
    """A one-coordinate leaf's motion column about ``origin`` (its joint anchor ``X_wpj``)."""
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
    """Publish an articulation's root and mark its cache valid."""
    root = joint_child[root_joint]
    body_q[root] = X_root
    body_qd[root] = wp.spatial_vector()
    articulation_origin[art] = wp.transform_point(X_root, body_com[root])
    current_valid[art] = 1


def class_kernels(turns: bool) -> tuple[wp.Kernel, wp.Kernel, wp.Kernel]:
    """The class's repair, predict and finish for a family whose leaves all slide (``turns`` false) or of any
    one-coordinate kind (``turns``: a leaf turns where its joint has an angular axis).

    A slide's column has no turn: the slides' predict reads no pose (the moment arms of gravity and the external
    wrench drop out, ``H_ii = m |S|^2``) and their finish takes the column about the world's origin, the same column
    about every point. With turning leaves every column, moment and velocity is taken about the articulation origin, and
    ``H_ii`` is the body's inertia there applied to the column (:func:`.spatial_laws.inertia_at`,
    :func:`.spatial_laws.inertia_apply`).
    """

    def repair(
        n: int,
        group_to_art: wp.array[int],
        dof_joint_offset: wp.array[int],
        articulation_start: wp.array[int],
        articulation_dof_start: wp.array[int],
        current_valid: wp.array[int],
        source: wp.array[wp.uint64],
        bank: wp.uint64,
        clock: wp.array[Clock],
        dt: float,
        joint_child: wp.array[int],
        joint_q_start: wp.array[int],
        joint_dof_dim: wp.array2d[int],
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
        """Publish the input state of articulations whose cache does not stand for the input bank ``bank`` (its
        coordinates' address), one warp per articulation: an invalid cache, one stamped with another bank (``source``),
        or a changed step duration ``dt`` (:func:`.factor.kernels.clock_changed`).

        Validity is read only: ``finish`` publishes the next state, stamps it and marks it valid.
        """
        tid = wp.tid()
        group = tid // _REPAIR_LANES
        lane = tid - group * _REPAIR_LANES
        art = group_to_art[group]
        if current_valid[art] != 0 and source[group] == bank and clock_changed(clock[0], dt) == 0:
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
            angular = False
            if wp.static(turns):
                angular = joint_dof_dim[joint, 1] != 0
            body_q[child] = _leaf_pose(X_root, joint_X_p[joint], angular, axis, q, joint_X_c[joint])
            joint_S_s[dof] = _leaf_motion(X_wpj, origin, angular, axis)

    def predict(
        n: int,
        group_to_art: wp.array[int],
        dof_joint_offset: wp.array[int],
        articulation_start: wp.array[int],
        articulation_dof_start: wp.array[int],
        joint_child: wp.array[int],
        joint_q_start: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_mass: wp.array[float],
        body_inertia: wp.array[wp.mat33],
        body_flags: wp.array[int],
        body_f: wp.array[wp.spatial_vector],
        articulation_origin: wp.array[wp.vec3],
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
        """Predict each leaf: the force, the masked inverse-mass refresh, ``qdd`` and ``v_hat``.

        A leaf is one coordinate, so it decides its drive's saturation here (:func:`.spatial_laws.drive_step_force`)
        with one more division: its ``K`` changing refreshes its own inverse mass.
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
        body_force = wp.spatial_vector()
        leaf_inertia = float(0.0)
        if wp.static(turns):
            # The moments about the articulation origin, from the leaf's pose.
            pose = body_q[child]
            radius = wp.transform_point(pose, body_com[child]) - articulation_origin[art]
            body_force = -wrench_shift(wp.spatial_vector(force, wp.vec3()), radius) - _external_at_origin(
                child, radius, body_f, body_flags
            )
            total, moment, rotational = inertia_at(mass, body_inertia[child], wp.transform_get_rotation(pose), radius)
            leaf_inertia = wp.dot(motion, inertia_apply(total, moment, rotational, motion)) + armature[dof]
        else:
            body_force = -wp.spatial_vector(force, wp.vec3()) - _external_at_origin(
                child, wp.vec3(), body_f, body_flags
            )
            linear = wp.spatial_top(motion)
            leaf_inertia = mass * wp.dot(linear, linear) + armature[dof]
        q = joint_q[joint_q_start[root_joint] + local]
        qd = joint_qd[dof]
        row = drive_row_by_dof[dof]
        u0, stiffness = drive_terms(
            row,
            dof,
            joint_q_start[root_joint] + local,
            joint_q,
            joint_qd,
            dt,
            drive_target_index_by_dof,
            joint_target_ke,
            joint_target_kd,
            joint_target_pos,
            joint_target_vel,
            damping,
        )
        value = coordinate_force(
            motion, body_force, joint_f[dof], spring_stiffness[dof], spring_ref[dof], damping[dof], q, qd
        )
        refresh = mass_update_mask[art] != 0
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

    def finish(
        n: int,
        group_to_art: wp.array[int],
        dof_joint_offset: wp.array[int],
        articulation_start: wp.array[int],
        articulation_dof_start: wp.array[int],
        joint_child: wp.array[int],
        joint_q_start: wp.array[int],
        joint_dof_dim: wp.array2d[int],
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
        source: wp.array[wp.uint64],
        bank: wp.uint64,
    ):
        """Convert the solved velocity, integrate and publish each leaf and its root, stamped with the output bank
        ``bank`` (its coordinates' address)."""
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
        angular = False
        origin = wp.vec3()
        if wp.static(turns):
            angular = joint_dof_dim[joint, 1] != 0
            origin = wp.transform_point(X_root, body_com[joint_child[root_joint]])
        pose = _leaf_pose(X_root, joint_X_p[joint], angular, axis, q_new, joint_X_c[joint])
        body_q[child] = pose
        motion = _leaf_motion(X_wpj, origin, angular, axis)
        joint_S_s[dof] = motion
        velocity = motion * qd_new
        if wp.static(turns):
            # The centre of mass's velocity: the twist about the origin at the body's centre of mass.
            radius = wp.transform_point(pose, body_com[child]) - origin
            velocity = wp.spatial_vector(point_velocity(velocity, radius), wp.spatial_bottom(velocity))
        body_qd[child] = velocity
        if local == 0:
            _publish_root(
                art, root_joint, X_root, joint_child, body_com, body_q, body_qd, articulation_origin, current_valid
            )
            source[group] = bank

    kind = "turns" if turns else "slides"
    return (
        python_kernel(repair, f"repair_{kind}"),
        python_kernel(predict, f"predict_{kind}"),
        python_kernel(finish, f"finish_{kind}"),
    )


repair_slides, predict_slides, finish_slides = class_kernels(False)
repair_turns, predict_turns, finish_turns = class_kernels(True)


def coordinate_topology(model, articulations: np.ndarray, articulation_joint_end: np.ndarray) -> tuple:
    """Check that every articulation is a world-fixed root over one-body leaves of one coordinate each.

    Every articulation must hold the same number ``n`` of leaves, laid out as joints ``start + 1 .. start + n`` whose
    coordinates and DOFs follow the root's in leaf order: a FIXED or zero-coordinate D6 root joint to the world, and
    leaves of one coordinate (prismatic, revolute or a one-axis D6) whose parent is the root body and whose body is no
    joint's parent.

    Returns:
        ``(turns, reason)``: ``turns`` is whether a leaf turns (``None`` with the reason the topology is declined).
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
        return None, "One-coordinate trees require the same number of leaves on every root."
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
        return None, "One-coordinate trees require a FIXED or zero-coordinate D6 root joint to the world."
    leaves = roots[:, None] + 1 + np.arange(n)[None, :]
    leaf_bodies = children[leaves]
    if np.any(leaf_bodies < 0) or np.any(parents[leaves] != root_bodies[:, None]):
        return None, (
            "One-coordinate trees require every joint below the root to hang from the root body "
            "(chains and composite trees are not admitted)."
        )
    kinds = types[leaves]
    one_axis = (kinds == int(JointType.PRISMATIC)) | (kinds == int(JointType.REVOLUTE)) | (kinds == int(JointType.D6))
    if not np.all(one_axis & (dims[leaves].sum(axis=-1) == 1)):
        return None, "One-coordinate trees require one-coordinate leaves (prismatic, revolute or a one-axis D6)."
    owners = np.bincount(children[children >= 0], minlength=model.body_count)
    descendants = np.bincount(parents[parents >= 0], minlength=model.body_count)
    if np.any(owners[root_bodies] != 1) or np.any(owners[leaf_bodies] != 1) or np.any(descendants[leaf_bodies] != 0):
        return None, "One-coordinate trees require leaves without descendants (composite trees are not admitted)."
    offsets = np.arange(n)[None, :]
    if np.any(q_start[leaves] != q_start[roots][:, None] + offsets) or np.any(
        qd_start[leaves] != qd_start[roots][:, None] + offsets
    ):
        return None, "One-coordinate trees require leaf coordinates in leaf order."
    return bool(np.any(dims[leaves][..., 1] != 0)), ""


@dataclass(frozen=True, eq=False)
class CoordinateTreesPlan:
    """A compact diagonal family whose trees are one body and one coordinate each (the class's one-body test).

    Plans compare and hash by identity: ``dofs`` is an array.
    """

    size: int
    """Leaves (coordinates) per articulation."""
    group_count: int
    """Articulations in the group."""
    dofs: np.ndarray
    """The group's DOFs, sorted."""
    turns: bool
    """Whether a leaf turns: the family's kernels (:func:`class_kernels`)."""


def plan(inputs, *, size: int | None) -> tuple[CoordinateTreesPlan | None, str]:
    """Decide whether the compact diagonal family of ``size`` coordinates takes the one-coordinate class's kernels.

    The rule is topology: the family must be the compact diagonal's (``size``, ``None`` without one), each of its
    coordinates must move one body, and each articulation must pass :func:`coordinate_topology`. The held plan
    (:func:`.held_tree.plan`) reads it with the class's ``inputs`` (:class:`.class_plan.ClassInputs`).

    Returns:
        ``(plan, reason)``; ``plan`` is ``None`` when the family stays a held tree group.
    """
    if size is None:
        return None, "No compact diagonal family."
    if inputs.pins.star == "held":
        return None, "A test pin keeps the compact diagonal family in a held tree group (star='held')."
    model, forest, factor = inputs.model, inputs.forest, inputs.factor
    if model.particle_count:
        return None, "One-coordinate trees exclude particles."
    if inputs.options.enable_joint_velocity_limits:
        return None, "Joint velocity limits keep the compact diagonal family in a held tree group (motion velocity)."
    if size not in forest.direct_diagonal_inertia_sizes:
        return None, "The compact diagonal family has coordinates that do not each move one body."
    articulations = forest.group_to_art[size].numpy()
    turns, reason = coordinate_topology(model, articulations, forest.articulation_joint_end.numpy())
    if turns is None:
        return None, reason
    offsets = forest.crba_dof_joint_offset_by_size[size].numpy()
    if not np.array_equal(offsets, np.arange(1, size + 1)):
        return None, "The compact diagonal family's coordinates do not follow its leaves."
    if not factor.parallel_drives:
        return None, "One-coordinate trees evaluate the augmented drives of the owned drive-row topology."
    roots = model.articulation_start.numpy()[articulations]
    leaf = np.arange(size)[None, :]
    dofs = model.joint_qd_start.numpy()[roots][:, None] + leaf
    coordinates = model.joint_q_start.numpy()[roots][:, None] + leaf
    driven = factor.drive_row_by_dof.numpy()[dofs] >= 0
    # The kernels read a driven leaf's coordinate directly; the drive rows must address the same one.
    if np.any(factor.drive_q_index_by_dof.numpy()[dofs][driven] != coordinates[driven]):
        return None, "The augmented drives of the compact diagonal family do not read its leaf coordinates."
    chosen = CoordinateTreesPlan(
        size=int(size),
        group_count=len(articulations),
        dofs=np.sort(dofs.reshape(-1)).astype(np.int32),
        turns=turns,
    )
    kind = "turning and sliding" if turns else "sliding"
    reason = (
        f"Admitted: {chosen.group_count} world-fixed roots of {chosen.size} {kind} one-body leaves hold their trees "
        "a thread a coordinate."
    )
    return chosen, reason


class CoordinateTrees:
    """Own one compact diagonal family of one-coordinate trees with the held-tree owner's group interface.

    The owner calls ``begin`` before any factor or force consumer, ``refresh`` at the mass cadence, ``predict`` at
    stage 3, and ``finish`` to integrate and publish; ``invalidate`` and ``validate_model`` follow resets and model
    notifications. ``sparse`` is the group itself: the inverse-mass refresh runs inside ``predict``, where the mass mask,
    the drive stiffness and the motion columns of this substep are all at hand.
    """

    # The cache is stamped with the bank the finish published into; a repair compares the stamp, never the words.
    bank_compare = False
    motion_velocity = False

    def __init__(self, solver, group, family: CoordinateTreesPlan):
        """Bind the class kernels of the held plan's one-coordinate group ``group`` (:class:`.held_tree.HeldGroupPlan`)
        for its ``family``; the owner's tree groups hold the topology proof."""
        self.solver = solver
        self.plan = group
        self.size = group.size
        self.sparse = self
        device = solver.model.device
        self.group_count = int(group.trees.size)
        self.group_to_art = wp.array(group.trees, dtype=int, device=device)
        self.dof_joint_offset = solver.forest.crba_dof_joint_offset_by_size[group.size]
        self._cache_to_art = self.group_to_art.reshape((self.group_count, 1))
        # The family holds no geometry cache; invalidation clears this flag only.
        self._geometry_valid = wp.zeros(self.group_count, dtype=int, device=device)
        # Each articulation's stamp: the address of the coordinates its cache was published into (0: none).
        self._source = wp.zeros(self.group_count, dtype=wp.uint64, device=device)
        self.repair_kernel, self.predictor_kernel, self.finish_kernel = (
            (repair_turns, predict_turns, finish_turns)
            if family.turns
            else (repair_slides, predict_slides, finish_slides)
        )
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

        The held-tree owner that composes the group proves the topology, the joint axes and the joint frames through its
        tree groups beside this one (Isaac Lab re-poses fixed roots through ``joint_X_p`` with ``JOINT_PROPERTIES`` on
        resets). A changed axis raises there, as for the tree groups.

        Returns:
            ``False`` for numeric notifications, which skip every proof (see :meth:`TreeKineticState.validate_model`);
            otherwise ``True``: the owner's tree groups prove the same model arrays, and the owner keeps the caches only
            when every proof passes.
        """
        if not proof_runs(flags):
            return False
        if self._dimensions() != self._plan_dimensions:
            raise RuntimeError("One-coordinate tree dimensions changed; reconstruct and recapture")
        return True

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

    def begin(self, state_in, state_aug, stage3_qd, dt, *, state_out=None):
        """Publish the input state of articulations whose cache does not stand for ``state_in``.

        ``state_out`` (the step's output bank) is not compared: the group validates its input bank by its stamp alone
        (``bank_compare`` is ``False``).
        """
        del stage3_qd, state_out
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
                self._source,
                wp.uint64(state_in.joint_q.ptr),
                solver.factor.clock,
                dt,
                model.joint_child,
                model.joint_q_start,
                model.joint_dof_dim,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.body_com,
                state_in.joint_q,
            ],
            outputs=[state_in.body_q, state_aug.joint_S_s, solver.factor.articulation_origin],
            block_dim=REPAIR_BLOCK_DIM,
            device=model.device,
        )

    def refresh(self, state_aug):
        """Nothing to do here: ``predict`` refreshes the inverse masses the mass mask selects."""

    def predict(self, state_in, state_aug, control, stage3_qd, dt):
        """Form each leaf's force and refresh, solve and predict its coordinate, deciding its drive's saturation."""
        solver, model = self.solver, self.solver.model
        inputs = [
            self.size,
            self.group_to_art,
            self.dof_joint_offset,
            model.articulation_start,
            solver.forest.articulation_dof_start,
            model.joint_child,
            model.joint_q_start,
            state_aug.joint_S_s,
            state_in.body_q,
            model.body_com,
            model.body_mass,
            model.body_inertia,
            model.body_flags,
            state_in.body_f,
            solver.factor.articulation_origin,
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
        solver, model = self.solver, self.solver.model
        inputs = [
            self.size,
            self.group_to_art,
            self.dof_joint_offset,
            model.articulation_start,
            solver.forest.articulation_dof_start,
            model.joint_child,
            model.joint_q_start,
            model.joint_dof_dim,
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
                self._source,
                wp.uint64(state_out.joint_q.ptr),
            ],
            device=model.device,
        )

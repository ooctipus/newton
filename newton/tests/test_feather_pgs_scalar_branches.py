# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scalar branches: world-fixed roots over one-coordinate leaves (``scalar_branches.py``).

The CPU tests compare the scalar kernels, bound through ``ScalarBranchState``,
with the head chain they replace, launched on the same inputs: the prismatic
linear state route for prismatic leaves (bitwise) and ordinary dynamics for
revolute leaves.
"""

import dataclasses
import json
import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels as fk
from newton._src.solvers.feather_pgs import path_selection
from newton._src.solvers.feather_pgs import scalar_branches as sb
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.device_proof import DeviceProof
from newton._src.solvers.feather_pgs.mixed_tree_kinetic import MixedTreeKineticState
from newton._src.solvers.feather_pgs.prismatic_linear_state import (
    finalize_prismatic_linear_state,
    get_linear_inverse_mass_kernel,
    repair_prismatic_linear_state,
)
from newton._src.solvers.feather_pgs.prismatic_publication import PrismaticPublicationPlan
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_subtree_reduce import _assert_switch_within, _excess

DT = 1.0 / 240.0
GRAVITY = (0.7, -1.2, -9.1)
# Fields the scalar family stops writing for its bodies; no consumer on its route reads them.
DROPPED_FIELDS = ("body_q_com", "body_v_s", "body_a_s", "body_f_s", "body_I_s", "_body_inertia_terms")
# Fields the held owner writes only for free bodies, not for a fixed-base arm (it publishes body_v_s).
ARM_DROPPED_FIELDS = ("body_q_com", "body_a_s", "body_f_s", "body_I_s", "_body_inertia_terms")
# CUDA bounds against the prismatic linear state route: the keys' are that route's own A/B
# tolerances (FMA contraction); the arm leaves the serial route for the held owner's float order.
KEY_BOUND = {"rtol": 3.0e-5, "atol": 5.0e-6}
# Key-only contact impulses [N·s] (1e-4 to 1e-2 per step in these tests) at the keys' relative bound.
IMPULSE_BOUND = {"rtol": KEY_BOUND["rtol"], "atol": 1.0e-8}
# The held-solve bound (ae6baadb) on the arm: twice the ordinary solver's distance from the
# reference, at least 1e-5, plus the scalar route's own run-to-run spread. Every reference arm here
# runs ordinary dynamics (no cached owner takes these worlds, and cached_dynamics selects only that
# owner), so the ordinary distance is zero and the floor applies.
ARM_FLOOR = 1.0e-5
# Coordinates of the fixtures' arm. A star of as many leaves shares the arm's size group, which is
# then not diagonal: the recipe refuses its prismatic linear state and no scalar plan exists.
ARM_DOFS = 6
# Leaves of the CUDA owner fixtures: the key presses, the small stars and the keyboard-shaped worlds.
PRESS_LEAVES = 7
STAR_LEAVES = 12
KEYBOARD_LEAVES = 108


def build_branch_model(
    device="cpu",
    *,
    worlds=3,
    leaves=7,
    kinds=("prismatic",),
    root="fixed",
    arm=True,
    shapes=False,
    chain=False,
    multi=False,
    descendant=False,
    particle=False,
):
    """Per world: a tilted root over one-coordinate leaves (a keyboard's class) beside a fixed-base six-joint arm.

    Leaves carry nontrivial axes, anchors, COMs, drives, limits, armature,
    springs and damping. ``kinds`` cycles the leaves' joint kinds. ``root`` is
    ``"fixed"``, ``"d6"`` (zero coordinates), ``"revolute"`` or ``"free"``.
    ``chain`` hangs the second leaf from the first, ``multi`` makes the last
    leaf a two-coordinate D6 joint and ``descendant`` welds a body to the first
    leaf. ``shapes`` adds a box to each leaf and a sphere to the arm's tip;
    ``particle`` adds one particle to the model.
    """
    template = newton.ModelBuilder(gravity=GRAVITY)
    case = template.add_link(
        mass=1.2, com=wp.vec3(0.0, 0.01, -0.005), inertia=wp.mat33(0.02, 0.001, 0.0, 0.001, 0.05, 0.0, 0.0, 0.0, 0.06)
    )
    frame = wp.transform(wp.vec3(0.25, 0.0, 0.08), wp.quat_rpy(0.3, -0.2, 0.4))
    child_frame = wp.transform(wp.vec3(0.02, -0.01, 0.004), wp.quat_rpy(-0.1, 0.2, 0.0))
    if root == "fixed":
        joints = [template.add_joint_fixed(parent=-1, child=case, parent_xform=frame, child_xform=child_frame)]
    elif root == "d6":
        joints = [template.add_joint_d6(parent=-1, child=case, parent_xform=frame, child_xform=child_frame)]
    elif root == "revolute":
        joints = [
            template.add_joint_revolute(
                parent=-1, child=case, axis=newton.Axis.Z, parent_xform=frame, child_xform=child_frame
            )
        ]
    else:
        joints = [template.add_joint_free(child=case, parent_xform=frame)]
    leaf_bodies = []
    for index in range(leaves):
        kind = kinds[index % len(kinds)]
        body = template.add_link(
            mass=0.012 + 1.0e-3 * index,
            com=wp.vec3(0.0005, -0.0003, 0.002),
            inertia=wp.mat33(2.0e-6, 1.0e-7, 0.0, 1.0e-7, 2.0e-6, 0.0, 0.0, 0.0, 3.0e-6),
        )
        common = {
            "parent": leaf_bodies[0] if chain and index == 1 else case,
            "child": body,
            "parent_xform": wp.transform(
                wp.vec3(-0.17 + 0.019 * index, 0.01, 0.012), wp.quat_rpy(0.2, -0.1 * index, 0.3)
            ),
            "child_xform": wp.transform(wp.vec3(-0.009, 0.004, 0.008), wp.quat_rpy(0.4, 0.2, -0.1)),
        }
        drive = {
            "target_ke": 300.0 + 10.0 * index,
            "target_kd": 1.5,
            "armature": 1.0e-4,
            "damping": 0.02 + 0.001 * index,
            "spring_stiffness": 0.4,
            "spring_ref": -0.001,
        }
        if multi and index == leaves - 1:
            axis = newton.ModelBuilder.JointDofConfig
            joints.append(
                template.add_joint_d6(
                    linear_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Z)],
                    **common,
                )
            )
        elif kind == "prismatic":
            joints.append(
                template.add_joint_prismatic(
                    axis=wp.vec3(0.2 + 0.1 * index, 0.5, 0.9), limit_lower=-0.004, limit_upper=0.0, **common, **drive
                )
            )
        else:
            joints.append(
                template.add_joint_revolute(
                    axis=wp.vec3(0.3, -0.4 + 0.1 * index, 0.8), limit_lower=-0.5, limit_upper=0.5, **common, **drive
                )
            )
        leaf_bodies.append(body)
        if shapes:
            template.add_shape_box(body, hx=0.008, hy=0.008, hz=0.004)
    if descendant:
        tip = template.add_link(mass=0.004, inertia=wp.mat33(1.0e-7, 0.0, 0.0, 0.0, 1.0e-7, 0.0, 0.0, 0.0, 1.0e-7))
        joints.append(
            template.add_joint_fixed(
                parent=leaf_bodies[0], child=tip, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity())
            )
        )
    template.add_articulation(joints)
    if arm:
        base = template.add_link(mass=0.6, com=wp.vec3(0.0, 0.0, 0.02), inertia=wp.mat33(np.eye(3) * 1.0e-3))
        chain_joints = [
            template.add_joint_fixed(
                parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.25, 0.0), wp.quat_identity())
            )
        ]
        parent = base
        for index in range(6):
            link = template.add_link(
                mass=0.15,
                com=wp.vec3(0.01, 0.0, 0.03),
                inertia=wp.mat33(2.0e-4, 0.0, 0.0, 0.0, 2.0e-4, 0.0, 0.0, 0.0, 1.0e-4),
            )
            chain_joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=newton.Axis.Z if index % 2 == 0 else newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
                    target_ke=50.0,
                    target_kd=2.0,
                    armature=1.0e-3,
                    limit_lower=-2.0,
                    limit_upper=2.0,
                )
            )
            parent = link
        if shapes:
            template.add_shape_sphere(
                parent, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), radius=0.01
            )
        template.add_articulation(chain_joints)
    builder = newton.ModelBuilder(gravity=GRAVITY)
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    if particle:
        builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(), 0.01)
    model = builder.finalize(device=device)
    rng = np.random.default_rng(5)
    model.joint_q.assign(rng.uniform(-0.004, 0.0, model.joint_coord_count).astype(np.float32))
    model.joint_qd.assign(rng.uniform(-0.05, 0.05, model.joint_dof_count).astype(np.float32))
    return model


def branch_articulations(model):
    """The first articulation of every world: the branch root's."""
    return np.arange(0, model.articulation_count, model.articulation_count // model.world_count)


def cpu_solver(model, size):
    """A CPU solver that carries the CUDA route's drive-row topology and compact diagonal size.

    CPU keeps ordinary split dynamics, so it builds neither the asynchronous
    drive rows nor the sparse-diagonal response; give it both so that the
    scalar kernels and the head kernels bind the arrays the CUDA route binds.
    """
    solver = SolverFeatherPGS(model, pgs_mode="split", enable_joint_limits=True, update_mass_matrix_interval=2)
    solver._async_augmented_drives = True
    solver._setup_parallel_augmented_drive_topology(model)
    solver._compact_diagonal_mass_size = size
    solver._fk_id_cache_enabled = True
    # CPU does not allocate the compact inertia terms that the head kernels materialize.
    solver._body_inertia_terms = wp.zeros((model.body_count, 12), dtype=float, device=model.device)
    return solver


def branch_size(model):
    """Coordinates of the branch articulation (the first of each world)."""
    start = model.articulation_start.numpy()[:2]
    return int(np.diff(model.joint_qd_start.numpy()[start])[0])


def same(test, expected, actual, index, name):
    expected, actual = expected.numpy()[index], actual.numpy()[index]
    test.assertTrue(np.all(np.isfinite(actual)), name)
    np.testing.assert_array_equal(actual, expected, err_msg=name)


def recipe_solver(model, *, scalar, compact=True, prismatic=True, **options):
    """The keyboard recipe's solver: compact contacts and both prismatic opt-ins, plus scalar branches if asked."""
    arguments = {
        "pgs_mode": "matrix_free",
        "pgs_iterations": 8,
        "update_mass_matrix_interval": 2,
        "use_parallel_streams": True,
        "enable_joint_limits": True,
        "dense_max_constraints": 64,
        "mf_max_constraints": 64,
        **options,
    }
    with (
        mock.patch.object(solver_module, "_COMPACT_CONTACT_BOUNDARY", compact),
        mock.patch.object(solver_module, "_SPARSE_CONTACT_DIRECT", True),
        mock.patch.object(solver_module, "_PRISMATIC_PUBLICATION", prismatic),
        mock.patch.object(solver_module, "_PRISMATIC_LINEAR_STATE", prismatic),
        mock.patch.object(solver_module, "_SCALAR_BRANCHES", scalar),
    ):
        return SolverFeatherPGS(model, **arguments)


def install_held_contacts(model, state, contacts, pairs):
    """Hold one contact per ``(body, other body or -1 for the world)`` pair with fixed local witnesses.

    Both witnesses start at the first body's origin with normal ``-z`` (as
    ``install_contacts`` of the compact-contact tests), so a contact starts at
    zero gap and pushes back once its first body moves down against the other.
    """
    shape_body = model.shape_body.numpy()
    poses = state.body_q.numpy().astype(np.float64)
    capacity = contacts.rigid_contact_shape0.shape[0]
    shapes = np.full((2, capacity), -1, dtype=np.int32)
    points = np.zeros((2, capacity, 3), dtype=np.float32)
    for contact, bodies in enumerate(pairs):
        witness = poses[bodies[0], :3]
        for end, body in enumerate(bodies):
            if body < 0:
                points[end, contact] = witness
                continue
            shapes[end, contact] = int(np.flatnonzero(shape_body == body)[0])
            pose = poses[body]
            offset, rotation = witness - pose[:3], -pose[3:6]
            points[end, contact] = offset + 2.0 * np.cross(rotation, np.cross(rotation, offset) + pose[6] * offset)
    for end in range(2):
        getattr(contacts, f"rigid_contact_shape{end}").assign(shapes[end])
        getattr(contacts, f"rigid_contact_point{end}").assign(points[end])
        getattr(contacts, f"rigid_contact_margin{end}").zero_()
    contacts.rigid_contact_normal.assign(np.tile(np.array([0.0, 0.0, -1.0], np.float32), (capacity, 1)))
    contacts.rigid_contact_count.assign(np.array([len(pairs)], dtype=np.int32))


class _Layout:
    """Index sets of a branch model: the scalar family's coordinates, DOFs and bodies, and the arm's."""

    def __init__(self, model):
        starts = model.articulation_start.numpy()
        child, q_start, qd_start = model.joint_child.numpy(), model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        self.branches = branch_articulations(model)
        self.arms = self.branches + 1
        size = branch_size(model)
        leaves = [starts[a] + 1 + np.arange(size) for a in self.branches]
        self.key_coordinates = np.concatenate([q_start[joints] for joints in leaves])
        self.key_dofs = np.concatenate([qd_start[joints] for joints in leaves])
        self.key_bodies = np.concatenate([child[starts[a] : starts[a + 1]] for a in self.branches])
        self.key_leaves = np.concatenate([child[joints] for joints in leaves])
        self.arm_dofs = np.concatenate([np.arange(qd_start[starts[a]], qd_start[starts[a + 1]]) for a in self.arms])
        self.arm_bodies = np.concatenate([child[starts[a] : starts[a + 1]] for a in self.arms])
        self.arm_tips = child[starts[self.arms + 1] - 1]


class _PrismaticRoute:
    """The head chain of prismatic leaves (prismatic linear state), launched on one solver's arrays."""

    def __init__(self, solver):
        self.solver, model = solver, solver.model
        self.model, self.device = model, model.device
        self.size = solver._compact_diagonal_mass_size
        self.group = solver.group_to_art[self.size]
        self.offsets = solver._crba_dof_joint_offset_by_size[self.size]
        self.plan = PrismaticPublicationPlan.build(model, solver.articulation_joint_end)
        self.previous = wp.zeros(model.articulation_count, dtype=int, device=self.device)
        _, self.direct_tau = solver_module._get_partitioned_inverse_dynamics_kernels(self.size, "cpu")
        self.inverse_mass = get_linear_inverse_mass_kernel(self.size)

    def begin(self, state):
        s, model = self.solver, self.model
        wp.copy(self.previous, s._fk_id_cache_valid)
        wp.launch(
            fk.eval_rigid_fk_id,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.plan.joint_end,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                state.joint_q,
                state.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                s.body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                s.is_free_rigid,
                0,
                0,
                1,
                model.gravity,
            ],
            outputs=[
                state.body_q,
                s.body_q_com,
                s.articulation_origin,
                s.joint_S_s,
                s.body_I_s,
                s._body_inertia_terms,
                s.body_v_s,
                s.body_f_s,
                s.body_a_s,
                s._fk_id_cache_valid,
            ],
            device=self.device,
        )
        wp.launch(
            repair_prismatic_linear_state,
            dim=model.body_count,
            inputs=[
                self.plan.body_joint,
                self.previous,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state.joint_q,
                state.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                s.body_X_com,
                model.joint_axis,
                s.body_to_articulation,
                model.body_mass,
                s.articulation_origin,
                model.gravity,
            ],
            outputs=[state.body_q, s.body_q_com, s.joint_S_s, s.body_v_s, s.body_a_s, s.body_f_s],
            device=self.device,
        )

    def predict(self, state, control, dt):
        s, model, size = self.solver, self.model, self.size
        wp.launch(
            solver_module._prepare_augmented_joint_drives_by_dof,
            dim=model.joint_dof_count,
            inputs=[
                s._augmented_drive_row_by_dof,
                s._augmented_drive_q_index_by_dof,
                s._augmented_drive_target_index_by_dof,
                state.joint_q,
                state.joint_qd,
                model.joint_target_ke,
                model.joint_target_kd,
                control.joint_target_q,
                control.joint_target_qd,
                model.joint_effort_limit,
                dt,
            ],
            outputs=[s.aug_row_K, s.joint_tau],
            device=self.device,
        )
        count = self.group.shape[0] * size
        wp.launch(
            self.direct_tau,
            dim=count,
            inputs=[
                self.group,
                self.offsets,
                model.articulation_start,
                model.joint_type,
                model.joint_child,
                model.joint_qd_start,
                model.joint_q_start,
                model.joint_dof_dim,
                control.joint_f,
                state.joint_q,
                state.joint_qd,
                s._passive_spring_stiffness,
                s._passive_spring_ref,
                s._passive_joint_damping,
                s.joint_S_s,
                s.body_f_s,
                state.body_f,
                model.body_flags,
                state.body_q,
                model.body_com,
                s.articulation_origin,
                1,
            ],
            outputs=[s.joint_tau],
            device=self.device,
        )
        wp.launch(
            self.inverse_mass,
            dim=count,
            inputs=[
                model.articulation_start,
                s.articulation_dof_start,
                s.mass_update_mask,
                model.joint_child,
                s.joint_S_s,
                model.body_mass,
                s._body_inertia_terms,
                self.group,
                self.offsets,
                s.R_by_size[size],
                s._augmented_drive_row_by_dof,
                s.aug_row_K,
            ],
            outputs=[s._diagonal_inverse_mass],
            device=self.device,
        )
        wp.launch(
            fk.solve_compact_diagonal_mass,
            dim=count,
            inputs=[s._diagonal_inverse_mass, self.group, s.articulation_dof_start, size, s.joint_tau],
            outputs=[s.joint_qdd],
            device=self.device,
        )
        wp.launch(
            fk.compute_velocity_predictor,
            dim=model.joint_dof_count,
            inputs=[state.joint_qd, s._kinematic_dof_mask, dt],
            outputs=[s.joint_qdd, s.v_hat],
            device=self.device,
        )

    def finish(self, state_in, state_out, dt):
        s, model = self.solver, self.model
        wp.launch(
            fk.update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, s._kinematic_dof_mask, 1.0 / dt],
            outputs=[s.v_out, s.joint_qdd],
            device=self.device,
        )
        wp.launch(
            fk.integrate_generalized_joints,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                s._kinematic_joint_mask,
                model.joint_dof_dim,
                model.body_com,
                model.joint_X_c,
                state_in.joint_q,
                state_in.joint_qd,
                s.joint_qdd,
                dt,
                0.0,
            ],
            outputs=[state_out.joint_q, state_out.joint_qd],
            device=self.device,
        )
        wp.launch(
            fk.eval_rigid_fk_kinematics,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.plan.joint_end,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                state_out.joint_q,
                state_out.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                s.body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
            ],
            outputs=[
                state_out.body_q,
                s.body_q_com,
                s.articulation_origin,
                s.joint_S_s,
                s.body_v_s,
                s.body_a_s,
                s._fk_id_cache_valid,
            ],
            device=self.device,
        )
        wp.launch(
            finalize_prismatic_linear_state,
            dim=model.body_count,
            inputs=[
                self.plan.body_joint,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                state_out.joint_q,
                state_out.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                s.body_X_com,
                model.joint_axis,
                s.joint_S_s,
                s.body_to_articulation,
                state_out.body_q,
                s.body_q_com,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                s.is_free_rigid,
                s.articulation_origin,
                0,
                1,
                model.gravity,
                s.body_v_s,
                s.body_a_s,
            ],
            outputs=[s.body_I_s, s._body_inertia_terms, s.body_f_s, state_out.body_qd],
            device=self.device,
        )


class _OrdinaryRoute:
    """Ordinary dynamics of the branch articulations: complete FK/ID, direct tau and inverse mass, finalize."""

    def __init__(self, solver):
        self.solver, model = solver, solver.model
        self.model, self.device = model, model.device
        self.size = solver._compact_diagonal_mass_size
        self.group = solver.group_to_art[self.size]
        self.offsets = solver._crba_dof_joint_offset_by_size[self.size]
        _, self.direct_tau = solver_module._get_partitioned_inverse_dynamics_kernels(self.size, "cpu")
        self.inverse_mass = solver_module._get_direct_diagonal_inverse_mass_kernel(self.size, "cpu")

    def begin(self, state):
        s, model = self.solver, self.model
        s._fk_id_cache_valid.zero_()
        wp.launch(
            fk.eval_rigid_fk_id,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                s.articulation_joint_end,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                state.joint_q,
                state.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                s.body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                s.is_free_rigid,
                0,
                1,
                1,
                model.gravity,
            ],
            outputs=[
                state.body_q,
                s.body_q_com,
                s.articulation_origin,
                s.joint_S_s,
                s.body_I_s,
                s._body_inertia_terms,
                s.body_v_s,
                s.body_f_s,
                s.body_a_s,
                s._fk_id_cache_valid,
            ],
            device=self.device,
        )

    def predict(self, state, control, dt):
        s, model, size = self.solver, self.model, self.size
        count = self.group.shape[0] * size
        wp.launch(
            solver_module._prepare_augmented_joint_drives_by_dof,
            dim=model.joint_dof_count,
            inputs=[
                s._augmented_drive_row_by_dof,
                s._augmented_drive_q_index_by_dof,
                s._augmented_drive_target_index_by_dof,
                state.joint_q,
                state.joint_qd,
                model.joint_target_ke,
                model.joint_target_kd,
                control.joint_target_q,
                control.joint_target_qd,
                model.joint_effort_limit,
                dt,
            ],
            outputs=[s.aug_row_K, s.joint_tau],
            device=self.device,
        )
        wp.launch(
            self.direct_tau,
            dim=count,
            inputs=[
                self.group,
                self.offsets,
                model.articulation_start,
                model.joint_type,
                model.joint_child,
                model.joint_qd_start,
                model.joint_q_start,
                model.joint_dof_dim,
                control.joint_f,
                state.joint_q,
                state.joint_qd,
                s._passive_spring_stiffness,
                s._passive_spring_ref,
                s._passive_joint_damping,
                s.joint_S_s,
                s.body_f_s,
                state.body_f,
                model.body_flags,
                state.body_q,
                model.body_com,
                s.articulation_origin,
                1,
            ],
            outputs=[s.joint_tau],
            device=self.device,
        )
        wp.launch(
            self.inverse_mass,
            dim=count,
            inputs=[
                model.articulation_start,
                s.articulation_dof_start,
                s.mass_update_mask,
                model.joint_child,
                s.joint_S_s,
                model.body_mass,
                s._body_inertia_terms,
                self.group,
                self.offsets,
                s.R_by_size[size],
                s._augmented_drive_row_by_dof,
                s.aug_row_K,
            ],
            outputs=[s._diagonal_inverse_mass],
            device=self.device,
        )
        wp.launch(
            fk.solve_compact_diagonal_mass,
            dim=count,
            inputs=[s._diagonal_inverse_mass, self.group, s.articulation_dof_start, size, s.joint_tau],
            outputs=[s.joint_qdd],
            device=self.device,
        )
        wp.launch(
            fk.compute_velocity_predictor,
            dim=model.joint_dof_count,
            inputs=[state.joint_qd, s._kinematic_dof_mask, dt],
            outputs=[s.joint_qdd, s.v_hat],
            device=self.device,
        )

    def finish(self, state_in, state_out, dt):
        s, model = self.solver, self.model
        wp.launch(
            fk.update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, s._kinematic_dof_mask, 1.0 / dt],
            outputs=[s.v_out, s.joint_qdd],
            device=self.device,
        )
        wp.launch(
            fk.integrate_generalized_joints,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                s._kinematic_joint_mask,
                model.joint_dof_dim,
                model.body_com,
                model.joint_X_c,
                state_in.joint_q,
                state_in.joint_qd,
                s.joint_qdd,
                dt,
                0.0,
            ],
            outputs=[state_out.joint_q, state_out.joint_qd],
            device=self.device,
        )
        wp.launch(
            fk.eval_rigid_fk_kinematics,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                s.articulation_joint_end,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                state_out.joint_q,
                state_out.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                s.body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
            ],
            outputs=[
                state_out.body_q,
                s.body_q_com,
                s.articulation_origin,
                s.joint_S_s,
                s.body_v_s,
                s.body_a_s,
                s._fk_id_cache_valid,
            ],
            device=self.device,
        )
        wp.launch(
            fk.finalize_body_dynamics,
            dim=model.body_count,
            inputs=[
                s.body_to_articulation,
                state_out.body_q,
                s.body_q_com,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                s.is_free_rigid,
                s.articulation_origin,
                0,
                1,
                model.gravity,
                s.body_v_s,
                s.body_a_s,
            ],
            outputs=[s.body_I_s, s._body_inertia_terms, s.body_f_s, state_out.body_qd],
            device=self.device,
        )


class _Pair:
    """A head-route solver and a scalar-branch solver over one model, each with its own states and inputs."""

    def __init__(self, model, route=_PrismaticRoute, *, poison=False, prismatic=None):
        size = branch_size(model)
        self.model = model
        self.head = cpu_solver(model, size)
        self.mine = cpu_solver(model, size)
        self.route = route(self.head)
        self.plan, self.reason = sb.plan_scalar_branches(self.mine, requested=True)
        if self.plan is None:
            raise AssertionError(self.reason)
        if prismatic is not None:
            self.plan = dataclasses.replace(self.plan, prismatic=prismatic)
        self.state = sb.ScalarBranchState(self.mine, self.plan)
        self.states = {name: [model.state(), model.state()] for name in ("head", "mine")}
        self.controls = {name: model.control() for name in ("head", "mine")}
        if poison:
            for name in DROPPED_FIELDS:
                getattr(self.mine, name).fill_(float("nan"))
        starts = model.articulation_start.numpy()
        self.arts = self.mine.group_to_art[size].numpy()
        self.bodies = np.concatenate([model.joint_child.numpy()[starts[a] : starts[a + 1]] for a in self.arts])
        self.coordinates = np.concatenate(
            [model.joint_q_start.numpy()[starts[a] + 1] + np.arange(size) for a in self.arts]
        )
        self.dofs = self.plan.dofs
        self.rng = np.random.default_rng(17)
        self.step_index = 0

    def inputs(self):
        """Draw the same targets, control forces and external wrenches for both routes."""
        model, rng = self.model, self.rng
        target_q = rng.uniform(-0.003, 0.0, self.controls["head"].joint_target_q.shape[0]).astype(np.float32)
        target_qd = rng.uniform(-0.01, 0.01, self.controls["head"].joint_target_qd.shape[0]).astype(np.float32)
        joint_f = rng.uniform(-2.0e-3, 2.0e-3, model.joint_dof_count).astype(np.float32)
        body_f = rng.uniform(-2.0e-2, 2.0e-2, (model.body_count, 6)).astype(np.float32)
        for name in ("head", "mine"):
            self.controls[name].joint_target_q.assign(target_q)
            self.controls[name].joint_target_qd.assign(target_qd)
            self.controls[name].joint_f.assign(joint_f)
            self.states[name][0].body_f.assign(body_f)

    def mass_mask(self, refresh):
        """Select every articulation for the inverse-mass refresh, or none."""
        values = np.full(self.model.articulation_count, int(refresh), dtype=np.int32)
        for solver in (self.head, self.mine):
            solver.mass_update_mask.assign(values)

    def step(self, test, *, exact_predict=True):
        """Advance both routes one substep and compare what each phase publishes.

        Both routes integrate the same solved velocity: the head's predictor
        plus a random impulse response, standing in for the contact solve.
        """
        head_in, head_out = self.states["head"]
        mine_in, mine_out = self.states["mine"]
        tag = f"step {self.step_index}"
        self.route.begin(head_in)
        self.state.begin(mine_in, self.mine, mine_in.joint_qd, DT, True)
        same(test, head_in.body_q, mine_in.body_q, self.bodies, f"{tag} input body_q")
        same(test, self.head.joint_S_s, self.mine.joint_S_s, self.dofs, f"{tag} input joint_S_s")
        same(test, self.head.articulation_origin, self.mine.articulation_origin, self.arts, f"{tag} input origin")
        self.route.predict(head_in, self.controls["head"], DT)
        self.state.predict(mine_in, self.mine, self.controls["mine"], mine_in.joint_qd, DT)
        same(test, self.head._diagonal_inverse_mass, self.mine._diagonal_inverse_mass, self.dofs, f"{tag} inverse mass")
        for name in ("joint_qdd", "v_hat"):
            if exact_predict:
                same(test, getattr(self.head, name), getattr(self.mine, name), self.dofs, f"{tag} {name}")
            else:
                np.testing.assert_allclose(
                    getattr(self.mine, name).numpy()[self.dofs],
                    getattr(self.head, name).numpy()[self.dofs],
                    rtol=2.0e-5,
                    atol=1.0e-6,
                    err_msg=f"{tag} {name}",
                )
        solved = self.head.v_hat.numpy() + self.rng.uniform(-0.03, 0.03, self.model.joint_dof_count).astype(np.float32)
        for solver in (self.head, self.mine):
            solver.v_out.assign(solved)
        self.route.finish(head_in, head_out, DT)
        self.state.finish(mine_in, self.mine, mine_out, DT, False)
        for name, index in (("joint_q", self.coordinates), ("joint_qd", self.dofs)):
            same(test, getattr(head_out, name), getattr(mine_out, name), index, f"{tag} {name}")
        for name in ("body_q", "body_qd"):
            same(test, getattr(head_out, name), getattr(mine_out, name), self.bodies, f"{tag} {name}")
        for name in ("joint_qdd", "v_out", "joint_S_s"):
            same(test, getattr(self.head, name), getattr(self.mine, name), self.dofs, f"{tag} finish {name}")
        same(test, self.head.articulation_origin, self.mine.articulation_origin, self.arts, f"{tag} origin")
        np.testing.assert_array_equal(self.mine._fk_id_cache_valid.numpy()[self.arts], 1)
        self.states["head"].reverse()
        self.states["mine"].reverse()
        self.step_index += 1


class TestScalarBranchRule(unittest.TestCase):
    def test_world_fixed_one_coordinate_stars_are_admitted(self):
        """Admit prismatic, revolute and mixed leaf stars on FIXED and zero-coordinate D6 roots."""
        for kinds, root in (
            (("prismatic",), "fixed"),
            (("revolute",), "fixed"),
            (("prismatic", "revolute"), "fixed"),
            (("prismatic",), "d6"),
        ):
            with self.subTest(kinds=kinds, root=root):
                model = build_branch_model(kinds=kinds, root=root)
                solver = cpu_solver(model, 7)
                plan, reason = sb.plan_scalar_branches(solver, requested=True)
                self.assertIsNotNone(plan, reason)
                self.assertEqual((plan.size, plan.group_count), (7, model.world_count))
                self.assertEqual(plan.kinds, tuple(sorted(kinds)))
                self.assertEqual(plan.prismatic, kinds == ("prismatic",))
                self.assertIn("supersede FEATHER_PGS_PRISMATIC", reason)
                starts = model.articulation_start.numpy()
                expected = np.concatenate(
                    [model.joint_qd_start.numpy()[starts[a] + 1] + np.arange(7) for a in solver.group_to_art[7].numpy()]
                )
                np.testing.assert_array_equal(plan.dofs, np.sort(expected))
                # A plan holds an array, so it compares and hashes by identity.
                self.assertNotEqual(plan, dataclasses.replace(plan, dofs=plan.dofs.copy()))
                self.assertEqual(len({plan, plan}), 1)

    def test_admits_the_keyboard_shaped_fixture(self):
        """Admit the 108-key group of the two-world 108+6 fixture, whatever the arm's root."""
        from newton.tests.test_feather_pgs_compact_contact import build_solver_fixture  # noqa: PLC0415

        model = build_solver_fixture("cpu")
        kinds, reason = sb.branch_topology(model, np.array([0, 2]), model.articulation_start.numpy()[1:])
        self.assertIsNotNone(kinds, reason)
        self.assertEqual(kinds.shape, (2, 108))
        self.assertTrue(np.all(kinds == int(newton.JointType.PRISMATIC)))

    def test_owner_fixtures_give_the_keys_their_own_group(self):
        """Give the keys of every CUDA owner fixture a diagonal-mass size group apart from the arm's."""
        for leaves in (PRESS_LEAVES, STAR_LEAVES, KEYBOARD_LEAVES):
            with self.subTest(leaves=leaves):
                model = build_branch_model(worlds=2, leaves=leaves, shapes=True)
                solver = SolverFeatherPGS(model, pgs_mode="split", enable_joint_limits=True)
                self.assertEqual(sorted(solver.size_groups), sorted((leaves, ARM_DOFS)))
                self.assertEqual(set(solver._diagonal_mass_sizes), {leaves})
        # An arm-sized star joins the arm's group: no diagonal-mass size is left to admit.
        model = build_branch_model(worlds=2, leaves=ARM_DOFS, shapes=True)
        solver = SolverFeatherPGS(model, pgs_mode="split", enable_joint_limits=True)
        self.assertEqual((solver.size_groups, set(solver._diagonal_mass_sizes)), ([ARM_DOFS], set()))

    def test_other_topologies_are_declined(self):
        """Decline chains, multi-coordinate leaves, welded descendants and roots that move."""
        for variant, words in (
            ({"chain": True}, "hang from the root body"),
            ({"descendant": True}, "hang from the root body"),
            ({"multi": True}, "one-coordinate"),
            ({"root": "revolute"}, "FIXED or zero-coordinate D6 root"),
            ({"root": "free"}, "FIXED or zero-coordinate D6 root"),
        ):
            with self.subTest(**variant):
                model = build_branch_model(worlds=2, **variant)
                joint_end = model.articulation_start.numpy()[1:]
                kinds, reason = sb.branch_topology(model, branch_articulations(model), joint_end)
                self.assertIsNone(kinds)
                self.assertIn(words, reason)

    def test_plan_reasons_follow_the_solver(self):
        """Report the request, the diagonal group and each decline of the solver or its model, with its reason."""
        model = build_branch_model(worlds=2)
        solver = cpu_solver(model, 7)
        self.assertIn("Off: FEATHER_PGS_SCALAR_BRANCHES=1", sb.plan_scalar_branches(solver, requested=False)[1])
        with mock.patch.object(solver, "_compact_diagonal_mass_size", None):
            plan, reason = sb.plan_scalar_branches(solver, requested=True)
            self.assertIsNone(plan)
            self.assertIn("No diagonal-mass group", reason)
            # A model without a diagonal group gives the same reason with the request off.
            self.assertEqual(sb.plan_scalar_branches(solver, requested=False)[1], reason)
        plan, _reason = sb.plan_scalar_branches(solver, requested=True)
        self.assertIsNotNone(plan)
        # A drive row that reads another coordinate than its leaf's, and leaves out of coordinate order.
        q_index = solver._augmented_drive_q_index_by_dof.numpy()
        q_index[plan.dofs[0]] += 1
        q_index = wp.array(q_index, dtype=int, device=model.device)
        offsets = dict(solver._crba_dof_joint_offset_by_size)
        offsets[7] = wp.array(np.roll(np.arange(1, 8), 1), dtype=int, device=model.device)
        for attribute, value, words in (
            ("enable_joint_velocity_limits", True, "velocity limits"),
            ("_direct_diagonal_inertia_sizes", frozenset(), "do not each move one body"),
            ("_crba_dof_joint_offset_by_size", offsets, "do not follow its leaves"),
            ("_parallel_augmented_drive_topology", False, "owned drive-row topology"),
            ("_augmented_drive_q_index_by_dof", q_index, "do not read its leaf coordinates"),
        ):
            with self.subTest(attribute=attribute), mock.patch.object(solver, attribute, value):
                plan, reason = sb.plan_scalar_branches(solver, requested=True)
                self.assertIsNone(plan)
                self.assertIn(words, reason)
        with self.subTest("gradients"), mock.patch.object(model, "requires_grad", True):
            plan, reason = sb.plan_scalar_branches(solver, requested=True)
            self.assertIsNone(plan)
            self.assertIn("exclude gradients and particles", reason)
        with self.subTest("particles"):
            particles = cpu_solver(build_branch_model(worlds=2, particle=True), 7)
            self.assertEqual(particles.model.particle_count, 1)
            plan, reason = sb.plan_scalar_branches(particles, requested=True)
            self.assertIsNone(plan)
            self.assertIn("exclude gradients and particles", reason)

    def test_owner_composes_the_scalar_group(self):
        """Compose the scalar group beside the arm's tree group, keep the forest off it, and validate both.

        The owner is built directly on CPU, where the solver keeps ordinary dynamics.
        """
        model = build_branch_model(worlds=2)
        solver = cpu_solver(model, 7)
        solver._scalar_branches, _reason = sb.plan_scalar_branches(solver, requested=True)
        owner = MixedTreeKineticState(solver)
        self.assertIsInstance(owner.scalar, sb.ScalarBranchState)
        self.assertEqual([type(group).__name__ for group in owner.groups], ["TreeKineticState", "ScalarBranchState"])
        self.assertEqual([group.sparse.size for group in owner.groups], [6, 7])
        # The scalar group validates its input bank by identity, so the owner does.
        self.assertFalse(owner.bank_compare)
        # Every group proves its inputs unchanged: the notification keeps the caches.
        self.assertTrue(owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES))
        owner.invalidate()
        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
        with mock.patch.dict(os.environ, {"FEATHER_PGS_TREE_FOREST": "1"}):
            forced = MixedTreeKineticState(solver)
        self.assertIn("Scalar branches own", forced.forest_reason)
        self.assertEqual(len(forced.groups), 2)

    def test_tree_groups_beside_scalar_branches_validate_by_identity(self):
        """Bind and launch no input-bank comparison beside scalar branches, and record identity validation.

        The scalar group holds no bank stamps, so its owner validates a changed
        input bank by identity: the solver invalidates every cache of the owner,
        and the arm's tree group could only compare words whose cache was
        discarded. After a finish into ``s1`` a step that reads ``s0`` (a copy
        of ``s1`` after an odd substep count) launches the arm's ordinary repair
        and the scalar repair. Without scalar branches the same solver's tree
        groups compare that bank and record content validation. Launches are
        recorded, not run.
        """
        model = build_branch_model(worlds=2)
        solver = cpu_solver(model, 7)
        solver._scalar_branches, _reason = sb.plan_scalar_branches(solver, requested=True)
        owners = {"scalar": MixedTreeKineticState(solver)}
        with mock.patch.object(solver, "_scalar_branches", None):
            owners["plain"] = MixedTreeKineticState(solver)
        s0, s1 = model.state(), model.state()
        solver._last_step_dt = DT
        augmented = solver._prepare_augmented_state(s0, s1, model.control())
        kinds = {}
        for owner in owners.values():
            for group in owner.groups:
                kinds.update({group.repair_kernel: "repair", getattr(group, "bank_repair_kernel", None): "compare"})
        kinds.pop(None, None)

        def launches(call):
            launched = []

            def record(kernel, *args, **kwargs):
                launched.append(kernel)

            with mock.patch.multiple(wp, launch=record, launch_tiled=record):
                call()
            return [kinds[kernel] for kernel in launched if kernel in kinds]

        expected = {
            "scalar": (False, "identity", ["repair", "repair"]),
            "plain": (True, "content", ["compare", "compare"]),
        }
        for name, owner in owners.items():
            compare, validation, copy_back = expected[name]
            with self.subTest(owner=name):
                self.assertEqual(owner.bank_compare, compare)
                for group in owner.groups:
                    record = path_selection._tree_group(group)
                    self.assertEqual(record["input_validation"], validation)
                    self.assertEqual(group.bank_compare, compare)
                    # Nothing that compares is bound where nothing may compare.
                    bound = {key for key in ("bank", "bank_repair") if record["kernels"][key] is not None}
                    self.assertEqual(bound, {"bank_repair"} if compare else set())
                launches(lambda owner=owner: owner.finish(s0, augmented, s1, DT, False))
                self.assertEqual(
                    launches(lambda owner=owner: owner.begin(s0, augmented, s0.joint_qd, DT, False, state_out=s1)),
                    copy_back,
                )

    def test_owner_proves_the_scalar_topology_through_its_tree_groups(self):
        """Validate a notification with the owner's and the arm group's proofs alone, and still refuse new topology.

        The arm group's proof holds the model topology and art_to_world and
        watches every joint frame, and the owner's coverage proof every size's
        group map, so beside them the scalar group adds no proof (one launch and
        one readback per notification) of its own: a moved key root is reported
        through the arm group's proof, so the solver invalidates every cache. A
        scalar group built alone keeps its proof.
        """
        joint = newton.ModelFlags.JOINT_PROPERTIES
        model = build_branch_model(worlds=2)
        solver = cpu_solver(model, 7)
        solver._scalar_branches, _reason = sb.plan_scalar_branches(solver, requested=True)
        owner = MixedTreeKineticState(solver)
        validate = DeviceProof.validate
        with mock.patch.object(DeviceProof, "validate", autospec=True, side_effect=validate) as proofs:
            self.assertTrue(owner.validate_model(joint))
        self.assertEqual(proofs.call_count, 2)
        self.assertIsNotNone(sb.ScalarBranchState(solver, solver._scalar_branches)._proof)
        frames = model.joint_X_p.numpy()
        frames[model.articulation_start.numpy()[owner.scalar.group_to_art.numpy()[1]], 2] += 0.01
        model.joint_X_p.assign(frames)
        self.assertFalse(owner.validate_model(joint))
        self.assertTrue(owner.validate_model(joint))
        parents = model.joint_parent.numpy()
        parents[model.articulation_start.numpy()[owner.scalar.group_to_art.numpy()[0]] + 2] = -1
        model.joint_parent.assign(parents)
        with self.assertRaisesRegex(RuntimeError, "joint_parent changed"):
            owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)

    def test_scalar_group_record_follows_the_tree_schema(self):
        """Record the scalar group with every field of a tree group's record, plus its leaf kinds and drive DOFs.

        The owner is built directly on CPU and recorded as ``get_path_selection`` records it.
        """
        model = build_branch_model(worlds=2)
        solver = cpu_solver(model, 7)
        solver._scalar_branches, solver._scalar_branches_reason = sb.plan_scalar_branches(solver, requested=True)
        with mock.patch.object(solver, "_tree_kinetic_state", MixedTreeKineticState(solver)):
            held = path_selection._held_tree(solver)
        json.dumps(held, allow_nan=False)
        self.assertEqual(held["owner"], "mixed")
        self.assertTrue(held["scalar_branches"]["enabled"])
        tree, scalar = held["groups"]
        self.assertEqual((set(scalar) - set(tree), set(tree) - set(scalar)), ({"kinds", "drive_dofs"}, set()))
        self.assertEqual((scalar["operator"], scalar["dofs"], scalar["kinds"]), ("ScalarBranchState", 7, ["prismatic"]))
        self.assertEqual(scalar["drive_dofs"], model.joint_dof_count - 7 * model.world_count)
        self.assertEqual(scalar["kernels"]["finish"], sb.finish_prismatic_branches.key)
        # The tree layout's packing fields hold no value for a group without a tree layout.
        self.assertEqual((scalar["trees_per_block"], scalar["block_dim"]), (None, None))
        self.assertIsInstance(tree["trees_per_block"], int)

    def test_cpu_route_record_reports_the_rule(self):
        """Report scalar branches as off on CPU, where no diagonal-mass group shares a sparse-diagonal response."""
        record = SolverFeatherPGS(build_branch_model(worlds=2), pgs_mode="split").get_path_selection()
        json.dumps(record, allow_nan=False)
        self.assertEqual(
            record["routes"]["held_tree"]["scalar_branches"],
            {"enabled": False, "reason": "No diagonal-mass group shares the sparse-diagonal response."},
        )

    def _admitted(self, model):
        """A plan the rule admits for the model's branch group, as the CUDA route builds it."""
        solver = cpu_solver(model, branch_size(model))
        plan, reason = sb.plan_scalar_branches(solver, requested=True)
        self.assertIsNotNone(plan, reason)
        return plan, reason

    def test_admitted_branches_skip_the_prismatic_opt_ins(self):
        """Build no prismatic publication or linear state where the owner takes the scalar branches.

        CPU installs no owner, so the install is replaced by one that takes the plan.
        """
        model = build_branch_model(worlds=2)
        admitted = self._admitted(model)

        def install(solver):
            solver._tree_kinetic_state = mock.Mock(scalar=object(), bank_compare=False)
            solver._cached_dynamics_selection_reason = "A test owner takes the branches."

        with (
            mock.patch.object(sb, "plan_scalar_branches", return_value=admitted),
            mock.patch("newton._src.solvers.feather_pgs.cached_dynamics.install", install),
            mock.patch.object(solver_module, "_PRISMATIC_PUBLICATION", True),
        ):
            solver = SolverFeatherPGS(model, pgs_mode="split")
        self.assertIs(solver._scalar_branches, admitted[0])
        self.assertEqual(solver._scalar_branches_reason, admitted[1])
        self.assertIsNone(solver._prismatic_publication)
        self.assertFalse(solver._prismatic_linear_state)

    def test_declined_branches_restore_the_prismatic_opt_ins(self):
        """Drop an admitted plan the held-tree owner does not take and build the prismatic opt-ins as without it.

        The declined solver and a twin built without the plan under the same
        opt-in must hold the same publication plan and the same routes, apart
        from the scalar branches' reason.
        """
        model = build_branch_model(worlds=2)
        with mock.patch.object(solver_module, "_PRISMATIC_PUBLICATION", True):
            with mock.patch.object(sb, "plan_scalar_branches", return_value=self._admitted(model)):
                solver = SolverFeatherPGS(model, pgs_mode="split")
            plain = SolverFeatherPGS(model, pgs_mode="split")
        self.assertIsNone(solver._tree_kinetic_state)
        self.assertIsNone(solver._scalar_branches)
        self.assertIn("did not take them", solver._scalar_branches_reason)
        self.assertIn("CPU", solver._scalar_branches_reason)
        self.assertIn("No diagonal-mass group", plain._scalar_branches_reason)
        for name in ("joint_end", "body_joint"):
            np.testing.assert_array_equal(
                getattr(solver._prismatic_publication, name).numpy(),
                getattr(plain._prismatic_publication, name).numpy(),
            )
        routes = [twin.get_path_selection()["routes"] for twin in (solver, plain)]
        for record in routes:
            self.assertFalse(record["held_tree"]["scalar_branches"]["enabled"])
            del record["held_tree"]["scalar_branches"]["reason"]
        self.assertEqual(routes[0], routes[1])
        self.assertTrue(routes[0]["publication"]["prismatic_publication"])


class TestScalarBranchKernels(unittest.TestCase):
    def test_prismatic_branches_match_the_linear_state_route(self):
        """Match the prismatic linear state route bitwise over steps, mass cadence, a masked reset and a re-pose.

        Every field the family publishes is compared: input and output poses,
        coordinates, body velocities, motion axes, accelerations, predictors,
        inverse masses and the articulation origin. The dropped FK/ID fields of
        the scalar solver hold NaN throughout, so no scalar kernel reads them.
        """
        pair = _Pair(build_branch_model(), poison=True)
        self.assertTrue(pair.plan.prismatic)
        worlds = pair.model.world_count
        for step in range(8):
            pair.inputs()
            pair.mass_mask(step % 2 == 0)
            if step == 2:
                # A masked reset of world 0 after writing its coordinates (Isaac Lab's reset).
                mask = wp.array([True] + [False] * (worlds - 1), dtype=wp.bool, device=pair.model.device)
                for name in ("head", "mine"):
                    current = pair.states[name][0]
                    q = current.joint_q.numpy()
                    q[pair.coordinates[: pair.plan.size]] -= 0.001
                    current.joint_q.assign(q)
                pair.head._fk_id_cache_valid.assign(
                    np.where(pair.head.art_to_world.numpy() == 0, 0, 1).astype(np.int32)
                )
                pair.state.invalidate(mask)
                np.testing.assert_array_equal(pair.mine._fk_id_cache_valid.numpy()[pair.arts], [0] + [1] * (worlds - 1))
            if step == 4:
                # Re-pose the branch roots through their frames and notify, as Isaac Lab's resets do.
                roots = pair.model.articulation_start.numpy()[pair.arts]
                frames = pair.model.joint_X_p.numpy()
                frames[roots, :3] += (0.01, -0.02, 0.03)
                frames[roots, 3:] = np.asarray(wp.quat_rpy(0.25, -0.1, 0.5))
                pair.model.joint_X_p.assign(frames)
                # The moved frames are reported, not refused; the solver then invalidates every cache.
                self.assertFalse(pair.state.validate_model(newton.ModelFlags.JOINT_PROPERTIES))
                pair.state.invalidate()
                pair.head._fk_id_cache_valid.zero_()
                np.testing.assert_array_equal(pair.mine._fk_id_cache_valid.numpy()[pair.arts], 0)
            pair.step(self)

    def test_kinematic_and_held_mass_steps_match(self):
        """Match a kinematic leaf (a prescribed coordinate) and held inverse masses bitwise."""
        model = build_branch_model(worlds=2)
        pair = _Pair(model)
        flags = model.body_flags.numpy()
        flags[pair.bodies[3]] |= int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        for solver in (pair.head, pair.mine):
            solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
        kinematic = pair.dofs[np.flatnonzero(pair.mine._kinematic_dof_mask.numpy()[pair.dofs])]
        self.assertEqual(len(kinematic), 1)
        for step in range(4):
            pair.inputs()
            pair.mass_mask(step == 0)
            pair.step(self)
        np.testing.assert_array_equal(pair.mine.joint_qdd.numpy()[kinematic], 0.0)

    def test_general_kernels_match_the_linear_state_route(self):
        """Run prismatic leaves through the general (revolute-capable) kernels with the linear state's bits."""
        pair = _Pair(build_branch_model(worlds=2), prismatic=False)
        self.assertIs(pair.state.predictor_kernel, sb.predict_scalar_branches)
        self.assertIs(pair.state.finish_kernel, sb.finish_scalar_branches)
        for step in range(3):
            pair.inputs()
            pair.mass_mask(step != 1)
            pair.step(self)

    def test_revolute_branches_follow_ordinary_dynamics(self):
        """Publish revolute leaves as ordinary dynamics does, predicting within the velocity-product rounding.

        Poses, motion axes, origins, inverse masses, integration and body
        velocities are bitwise. On a fixed root a one-coordinate body's
        velocity product projects to zero on its own axis; ordinary dynamics
        evaluates it, so its torques carry that rounding and the scalar
        family's do not.
        """
        model = build_branch_model(kinds=("revolute",), worlds=2)
        pair = _Pair(model, route=_OrdinaryRoute)
        self.assertFalse(pair.plan.prismatic)
        self.assertEqual(pair.plan.kinds, ("revolute",))
        for step in range(4):
            pair.inputs()
            pair.mass_mask(step % 2 == 0)
            pair.step(self, exact_predict=False)
        # The leaves turned: the published body velocities carry angular parts.
        self.assertGreater(float(np.abs(pair.states["mine"][0].body_qd.numpy()[pair.bodies, 3:]).max()), 1.0e-4)

    def test_other_drives_match_the_drive_kernel(self):
        """Prepare the drives outside the branches as the full drive kernel does and leave the branches' alone.

        The branches' predict evaluates their own drives (the twins above), so
        drive preparation covers only the other DOFs: their stiffness and
        forces carry the full kernel's bits, including a clamped force and a
        nonpositive stiffness, and nothing writes the branches' entries.
        """
        model = build_branch_model(worlds=3)
        pair = _Pair(model)
        layout = _Layout(model)
        rng = np.random.default_rng(23)
        ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
        effort = model.joint_effort_limit.numpy()
        ke[layout.arm_dofs[0]] = kd[layout.arm_dofs[0]] = 0.0
        effort[layout.arm_dofs[1]] = 1.0e-3
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        model.joint_effort_limit.assign(effort)
        state, control = model.state(), model.control()
        state.joint_q.assign(rng.uniform(-0.3, 0.3, model.joint_coord_count).astype(np.float32))
        state.joint_qd.assign(rng.uniform(-0.5, 0.5, model.joint_dof_count).astype(np.float32))
        control.joint_target_q.assign(rng.uniform(-0.2, 0.2, control.joint_target_q.shape[0]).astype(np.float32))
        control.joint_target_qd.assign(rng.uniform(-0.1, 0.1, control.joint_target_qd.shape[0]).astype(np.float32))
        head = pair.head
        wp.launch(
            solver_module._prepare_augmented_joint_drives_by_dof,
            dim=model.joint_dof_count,
            inputs=[
                head._augmented_drive_row_by_dof,
                head._augmented_drive_q_index_by_dof,
                head._augmented_drive_target_index_by_dof,
                state.joint_q,
                state.joint_qd,
                model.joint_target_ke,
                model.joint_target_kd,
                control.joint_target_q,
                control.joint_target_qd,
                model.joint_effort_limit,
                DT,
            ],
            outputs=[head.aug_row_K, head.joint_tau],
            device=model.device,
        )
        pair.mine.aug_row_K.fill_(-7.0)
        pair.mine.joint_tau.fill_(-7.0)
        pair.state.launch_other_drives(state, pair.mine, control, DT)
        others = pair.state.other_drive_dofs.numpy()
        np.testing.assert_array_equal(np.sort(np.concatenate([others, pair.dofs])), np.arange(model.joint_dof_count))
        self.assertEqual(path_selection._tree_group(pair.state)["drive_dofs"], len(others))
        rows = pair.mine._augmented_drive_row_by_dof.numpy()
        np.testing.assert_array_equal(pair.mine.joint_tau.numpy()[others], head.joint_tau.numpy()[others])
        other_rows = rows[others][rows[others] >= 0]
        np.testing.assert_array_equal(pair.mine.aug_row_K.numpy()[other_rows], head.aug_row_K.numpy()[other_rows])
        self.assertIn(0.0, head.aug_row_K.numpy()[other_rows])
        self.assertIn(np.float32(1.0e-3), np.abs(head.joint_tau.numpy()[others]))
        np.testing.assert_array_equal(pair.mine.joint_tau.numpy()[pair.dofs], -7.0)
        np.testing.assert_array_equal(pair.mine.aug_row_K.numpy()[rows[pair.dofs]], -7.0)

    def test_validation_accepts_frames_and_rejects_topology(self):
        """Accept numeric and frame notifications; refuse a changed parent or group map.

        The group reads frames and axes live: a notification that moved one
        reports it (``False``, the caller invalidates), and the next that moves
        nothing proves them unchanged (``True``, the caller keeps the caches).
        Numeric notifications skip the proof (``False``).
        """
        joint = newton.ModelFlags.JOINT_PROPERTIES
        model = build_branch_model(worlds=2)
        pair = _Pair(model)
        self.assertTrue(pair.state.validate_model(joint))
        frames = model.joint_X_p.numpy()
        frames[:, 2] += 0.05
        model.joint_X_p.assign(frames)
        axes = model.joint_axis.numpy()
        axes[pair.dofs[0]] = (0.0, 0.0, 1.0)
        model.joint_axis.assign(axes)
        self.assertFalse(pair.state.validate_model(joint))
        self.assertTrue(pair.state.validate_model(newton.ModelFlags.ALL))
        self.assertFalse(pair.state.validate_model(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))
        parents = model.joint_parent.numpy()
        parents[model.articulation_start.numpy()[pair.arts[0]] + 2] = -1
        model.joint_parent.assign(parents)
        # A numeric notification cannot change topology and is not checked.
        self.assertFalse(pair.state.validate_model(newton.ModelFlags.JOINT_DOF_PROPERTIES))
        with self.assertRaisesRegex(RuntimeError, "joint_parent changed"):
            pair.state.validate_model(joint)


def _close(test, expected, actual, index, tag, *, rtol, atol):
    actual, expected = actual.numpy()[index], expected.numpy()[index]
    test.assertTrue(np.all(np.isfinite(actual)), tag)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=tag)


def _arm_within(test, reference, actual, bodies, tag, *, again=None):
    """Bound the arm bodies' A/B difference by the held-solve rule plus the scalar route's A/A spread ``again``."""
    actual = actual.numpy()[bodies]
    test.assertTrue(np.all(np.isfinite(actual)), tag)
    spread = 0.0 if again is None else _excess(again.numpy()[bodies], actual)
    _assert_switch_within(test, tag, _excess(actual, reference.numpy()[bodies]), spread, ARM_FLOOR)


def _normal_impulses(test, solver, count):
    """The solved normal impulse [N·s] of each of the first ``count`` contacts, in contact order."""
    world, slot = solver.contact_world.numpy()[:count], solver.contact_slot.numpy()[:count]
    test.assertTrue(np.all(slot >= 0), "every held contact owns a row")
    return solver.impulses.numpy()[world, slot].astype(np.float64)


def _key_law(model, leaves, dofs, axes, q, qd, steps, push=None):
    """Integrate keys by their scalar law in float64 and return their coordinates after each step.

    Without contacts or active limits a prismatic key of mass ``m`` and unit
    motion axis ``S`` on a fixed root follows the implicit drive of zero targets
    with its passive spring and damping and gravity along the axis:
    ``(m |S|^2 + armature + ke dt^2 + kd dt) (qd' - qd) / dt = m S.g + push
    + spring (ref - q) - damping qd - (ke (q + dt qd) + kd qd)``, then
    ``q' = q + dt qd'``. ``push(step)`` gives a step's external generalized force.
    """
    m = model.body_mass.numpy()[leaves].astype(np.float64)
    ke = model.joint_target_ke.numpy()[dofs].astype(np.float64)
    kd = model.joint_target_kd.numpy()[dofs].astype(np.float64)
    spring = model.joint_spring_stiffness.numpy()[dofs].astype(np.float64)
    reference = model.joint_spring_ref.numpy()[dofs].astype(np.float64)
    damping = model.joint_damping.numpy()[dofs].astype(np.float64)
    armature = model.joint_armature.numpy()[dofs].astype(np.float64)
    inertia = m * np.sum(axes * axes, axis=1) + armature + ke * DT * DT + kd * DT
    weight = m * (axes @ model.gravity.numpy()[0].astype(np.float64))
    q, qd = np.array(q, dtype=np.float64), np.array(qd, dtype=np.float64)
    coordinates = []
    for step in range(steps):
        force = 0.0 if push is None else push(step)
        drive = -(ke * (q + DT * qd) + kd * qd)
        qd = qd + DT * (weight + force + spring * (reference - q) - damping * qd + drive) / inertia
        q = q + DT * qd
        coordinates.append(q.copy())
    return np.asarray(coordinates)


@unittest.skipUnless(wp.is_cuda_available(), "Scalar branches run on the CUDA held-tree owner")
class TestScalarBranchOwner(unittest.TestCase):
    def _poison(self, solver, layout):
        """Fill what the scalar owner neither writes nor reads with NaN.

        For the keys: the FK/ID fields the scalar family stops writing, and the
        drive stiffness and forces its predict evaluates itself. For the arm:
        the FK/ID fields the head's serial route writes and the held owner does
        not for a fixed-base tree.
        """
        for names, bodies in ((DROPPED_FIELDS, layout.key_bodies), (ARM_DROPPED_FIELDS, layout.arm_bodies)):
            for name in names:
                array = getattr(solver, name)
                values = array.numpy()
                values[bodies] = np.nan
                array.assign(values)
        tau, stiffness = solver.joint_tau.numpy(), solver.aug_row_K.numpy()
        tau[layout.key_dofs] = np.nan
        rows = solver._augmented_drive_row_by_dof.numpy()[layout.key_dofs]
        # A coordinate without a drive row (-1) must not poison the last row, which is an arm's.
        stiffness[rows[rows >= 0]] = np.nan
        solver.joint_tau.assign(tau)
        solver.aug_row_K.assign(stiffness)

    def _assert_routes(self, head, scalar):
        """Require the routes an A/B compares: the head's prismatic linear state and the scalar owner.

        The keys must form their own size group: an arm-sized star would share
        the arm's group, which the recipe cannot run (see
        ``test_owner_fixtures_give_the_keys_their_own_group``).
        """
        self.assertIsNone(head._tree_kinetic_state)
        self.assertTrue(head._prismatic_linear_state)
        owner = scalar._tree_kinetic_state
        self.assertIsInstance(owner, MixedTreeKineticState)
        self.assertIsInstance(owner.scalar, sb.ScalarBranchState)
        self.assertIs(owner.scalar.predictor_kernel, sb.predict_prismatic_branches)
        self.assertNotEqual(owner.scalar.size, ARM_DOFS)
        self.assertEqual(sorted(group.sparse.size for group in owner.groups), sorted((ARM_DOFS, owner.scalar.size)))
        self.assertIsNone(scalar._prismatic_publication)
        self.assertFalse(scalar._prismatic_linear_state)
        return owner

    def test_owner_matches_the_prismatic_route(self):
        """Own 108-key worlds as scalar branches beside a held arm and track the prismatic linear state route.

        The held-tree plan caps a group at 64 bodies, so the recipe's route had no
        owner for this keyboard-shaped model. Keys and their floor-contact
        impulses stay within the FMA bound of the prismatic linear state route
        through the mass cadence, an inertial and drive notification, a masked
        reset after a coordinate write, a root re-pose, a state bank copied back
        into the input (which every cache must repair) and captured graphs with
        masked and full resets. The arm moves to the held owner's float order,
        within the held-solve bound (``ARM_FLOOR``) plus the twin's spread.
        An eager twin of the scalar route does the captured work step by step:
        each replay must give what it gives. The FK/ID fields that neither the
        family writes for the keys nor the held owner for the arm hold NaN
        throughout, so nothing on the route reads them.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=4, leaves=KEYBOARD_LEAVES, shapes=True)
        layout = _Layout(model)
        starts, child = model.articulation_start.numpy(), model.joint_child.numpy()
        floors = [int(child[starts[a] + 1 + k]) for a in layout.branches for k in (0, 17, 53, 107)]
        model.rigid_contact_max = len(floors)
        # The head, the scalar route (captured below) and the scalar route's eager twin.
        solvers = [
            recipe_solver(model, scalar=False),
            recipe_solver(model, scalar=True),
            recipe_solver(model, scalar=True),
        ]
        head, scalar, twin = solvers
        owners = [self._assert_routes(head, solver) for solver in (scalar, twin)]
        for solver in (scalar, twin):
            self._poison(solver, layout)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [newton.Contacts(model.rigid_contact_max, 0, device=device) for _ in solvers]
        for index, pair in enumerate(states):
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
            install_held_contacts(model, pair[0], contacts[index], [(body, -1) for body in floors])
        worlds = model.world_count

        def inputs(sequence):
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[floors[1::4], 2] = -0.3 * sequence
            force[layout.arm_tips, :3] = (0.05 * sequence, -0.1, 0.02)
            for index, pair in enumerate(states):
                for state in pair:
                    state.body_f.assign(force)
                controls[index].joint_f.assign(np.linspace(-0.01, 0.02, model.joint_dof_count, dtype=np.float32))
                controls[index].joint_target_q.fill_(-0.0004 * sequence)

        def write_coordinates(reset):
            for pair in states:
                q, qd = pair[0].joint_q.numpy(), pair[0].joint_qd.numpy()
                for world, selected in enumerate(reset):
                    if selected:
                        q[layout.key_coordinates.reshape(worlds, -1)[world]] -= 0.0005
                        qd[layout.arm_dofs.reshape(worlds, -1)[world]] *= -0.5
                pair[0].joint_q.assign(q)
                pair[0].joint_qd.assign(qd)

        def compare(tag):
            for name, index in (("joint_q", layout.key_coordinates), ("joint_qd", layout.key_dofs)):
                _close(
                    self, getattr(states[0][0], name), getattr(states[1][0], name), index, f"{tag} {name}", **KEY_BOUND
                )
            for name in ("body_q", "body_qd"):
                _close(
                    self,
                    getattr(states[0][0], name),
                    getattr(states[1][0], name),
                    layout.key_bodies,
                    f"{tag} key {name}",
                    **KEY_BOUND,
                )
                _arm_within(
                    self,
                    getattr(states[0][0], name),
                    getattr(states[1][0], name),
                    layout.arm_bodies,
                    f"{tag} arm {name}",
                    again=getattr(states[2][0], name),
                )
            _close(self, head.v_hat, scalar.v_hat, layout.key_dofs, f"{tag} v_hat", **KEY_BOUND)
            _close(
                self,
                head._diagonal_inverse_mass,
                scalar._diagonal_inverse_mass,
                layout.key_dofs,
                f"{tag} minv",
                **KEY_BOUND,
            )
            # The floor contacts are key-only rows: their impulses follow the head's within FMA contraction.
            impulses = [_normal_impulses(self, solver, len(floors)) for solver in solvers]
            np.testing.assert_allclose(impulses[1], impulses[0], err_msg=f"{tag} impulses", **IMPULSE_BOUND)
            loads.append(float(impulses[0].max()))
            # The twin does the scalar route's work eagerly: it must give the same state and impulses.
            for name, index in (("joint_q", layout.key_coordinates), ("joint_qd", layout.key_dofs)):
                _close(
                    self,
                    getattr(states[1][0], name),
                    getattr(states[2][0], name),
                    index,
                    f"{tag} twin {name}",
                    **KEY_BOUND,
                )
            for name in ("body_q", "body_qd"):
                bodies = np.concatenate([layout.key_bodies, layout.arm_bodies])
                _close(
                    self,
                    getattr(states[1][0], name),
                    getattr(states[2][0], name),
                    bodies,
                    f"{tag} twin {name}",
                    **KEY_BOUND,
                )
            np.testing.assert_allclose(impulses[2], impulses[1], err_msg=f"{tag} twin impulses", **IMPULSE_BOUND)
            for solver in solvers:
                solver.check_constraint_capacity()
                self.assertGreater(int(solver.constraint_count.numpy().sum()), 0)
            for owner in owners:
                self.assertEqual(int(sum(int(group.status.numpy().sum()) for group in owner.groups[:-1])), 0)

        loads = []
        mask = wp.array([True] + [False] * (worlds - 1), dtype=wp.bool, device=device)
        for step in range(10):
            inputs(step + 1)
            if step == 3:
                model.body_mass.assign(model.body_mass.numpy() * 1.1)
                model.joint_armature.assign(model.joint_armature.numpy() * 1.2)
                model.joint_target_ke.assign(model.joint_target_ke.numpy() * 1.15)
                for solver in solvers:
                    solver.notify_model_changed(
                        newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES
                    )
            if step == 5:
                write_coordinates((True,) + (False,) * (worlds - 1))
                for index, solver in enumerate(solvers):
                    solver.reset(states[index][0], mask)
            if step == 7:
                # Lift the keyboards through their fixed roots, as Isaac Lab's resets re-pose them.
                frames = model.joint_X_p.numpy()
                frames[starts[layout.branches], 2] += 0.001
                model.joint_X_p.assign(frames)
                for solver in solvers:
                    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            for index, solver in enumerate(solvers):
                current, following = states[index]
                solver.step(current, following, controls[index], contacts[index], DT)
                if step == 8:
                    # Copy the output back into the input bank, as Isaac Lab does after an odd number of
                    # substeps: the next step reads a bank no cache published, so every route repairs.
                    current.assign(following)
                else:
                    states[index] = [following, current]
            compare(f"step {step}")

        graphs = []
        for index, solver in enumerate(solvers[:2]):
            current, following = states[index]
            with wp.ScopedCapture(device=device) as capture:
                solver.seed_double_buffer_events()
                solver.reset(current, mask)
                solver.step(current, following, controls[index], contacts[index], DT)
                solver.step(following, current, controls[index], contacts[index], DT)
            graphs.append(capture.graph)
        # Mask contents, coordinates and forces change between replays without rebinding graph inputs.
        for replay, reset in enumerate(((True, False, False, False), (False, True, True, False), (True,) * worlds)):
            mask.assign(np.asarray(reset, dtype=np.bool_))
            inputs(replay + 11)
            write_coordinates(reset)
            for solver in (scalar, twin):
                self._poison(solver, layout)
            for graph in graphs:
                wp.capture_launch(graph)
            # The eager twin of the captured work.
            current, following = states[2]
            twin.reset(current, mask)
            twin.step(current, following, controls[2], contacts[2], DT)
            twin.step(following, current, controls[2], contacts[2], DT)
            compare(f"replay {replay}")
        # The comparison is not vacuous: the floors carried the keys.
        self.assertGreater(max(loads), 0.0)

    def test_drive_kernel_covers_only_the_other_dofs(self):
        """Launch the drive preparation over the DOFs outside the branches only; the branches fold theirs.

        Without this restriction the full drive kernel runs over every DOF,
        the branches' 1.77 M of keyboard-shaped worlds included.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=STAR_LEAVES, shapes=True)
        model.rigid_contact_max = 1
        solver = recipe_solver(model, scalar=True)
        states, control = [model.state(), model.state()], model.control()
        contacts = newton.Contacts(model.rigid_contact_max, 0, device=device)
        contacts.rigid_contact_count.zero_()
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        launched = []
        original = wp.launch

        def record(kernel, *args, **kwargs):
            launched.append((kernel, kwargs.get("dim")))
            return original(kernel, *args, **kwargs)

        with mock.patch.object(wp, "launch", record):
            solver.step(states[0], states[1], control, contacts, DT)
        kernels = [kernel for kernel, _dim in launched]
        self.assertNotIn(solver_module._prepare_augmented_joint_drives_by_dof, kernels)
        dims = [dim for kernel, dim in launched if kernel is sb.prepare_listed_augmented_drives]
        self.assertEqual(dims, [model.joint_dof_count - STAR_LEAVES * model.world_count])
        self.assertTrue(np.isfinite(states[1].joint_q.numpy()).all())

    def test_key_press_follows_the_drive_law(self):
        """Press keys with an external force and release them: both routes follow the drive's closed form.

        Without contacts or active limits each key is a scalar system: the
        implicit drive gives ``(m |S|^2 + armature + K) qdd = u0 + forces`` with
        ``K = ke dt^2 + kd dt`` and ``u0 = -(ke (q + dt qd) + kd qd)`` for zero
        targets, plus the passive spring and damping and gravity along the axis.
        The press and the spring-back are integrated in float64 from the model's
        values and compared with both routes, which also match each other.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=PRESS_LEAVES, shapes=True)
        layout = _Layout(model)
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        lower[layout.key_dofs], upper[layout.key_dofs] = -0.5, 0.5
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        model.joint_q.zero_()
        model.joint_qd.zero_()
        model.rigid_contact_max = 1
        solvers = [recipe_solver(model, scalar=False), recipe_solver(model, scalar=True)]
        self._assert_routes(*solvers)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [newton.Contacts(model.rigid_contact_max, 0, device=device) for _ in solvers]
        for control, contact, pair in zip(controls, contacts, states, strict=True):
            control.joint_target_q.zero_()
            control.joint_target_qd.zero_()
            contact.rigid_contact_count.zero_()
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
        press, release = 20, 60
        travel = [[], []]
        axes = None
        for step in range(press + release):
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            if 0 < step <= press:
                # Along each key's motion axis, which the first step published.
                force[layout.key_leaves, :3] = -1.5 * axes / np.linalg.norm(axes, axis=1, keepdims=True)
            for index, solver in enumerate(solvers):
                current, following = states[index]
                current.body_f.assign(force)
                solver.step(current, following, controls[index], contacts[index], DT)
                states[index] = [following, current]
                travel[index].append(following.joint_q.numpy()[layout.key_coordinates].copy())
            if step == 0:
                axes = solvers[1].joint_S_s.numpy()[layout.key_dofs, :3].astype(np.float64)
                np.testing.assert_allclose(
                    solvers[0].joint_S_s.numpy()[layout.key_dofs, :3], axes, rtol=1.0e-6, atol=1.0e-7
                )
        travel = np.asarray(travel, dtype=np.float64)
        np.testing.assert_allclose(travel[1], travel[0], rtol=1.0e-5, atol=1.0e-8)
        # The same discrete law in float64; the press is the force's projection on each axis.
        pressing = -1.5 * np.linalg.norm(axes, axis=1)
        rest = np.zeros(len(layout.key_dofs))
        expected = _key_law(
            model,
            layout.key_leaves,
            layout.key_dofs,
            axes,
            rest,
            rest,
            press + release,
            push=lambda step: pressing if 0 < step <= press else 0.0,
        )
        self.assertGreater(float(np.abs(expected).max()), 2.0e-3)
        for index in range(2):
            np.testing.assert_allclose(travel[index], expected, rtol=1.0e-4, atol=2.0e-6)

    def test_pusher_press_matches_the_prismatic_route(self):
        """Press a key with a kinematic pusher: travel and impulses follow the head, the spring-back the key law.

        The pusher is a frictionless world contact above one key per world; its
        witness descends at constant speed, holds and then leaves. The contact is
        key-only (one sparse coordinate), so both routes solve the same row from
        the same motion axis, origin and inverse mass and differ by FMA
        contraction alone: travel and normal impulses within 1e-5 relative. Once
        the pusher leaves, each key springs back by its scalar law, integrated in
        float64 from the state it was released in.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=PRESS_LEAVES, shapes=True)
        layout = _Layout(model)
        worlds = model.world_count
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        lower[layout.key_dofs], upper[layout.key_dofs] = -0.5, 0.5
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        model.joint_q.zero_()
        model.joint_qd.zero_()
        # Frictionless keys: the pusher's normal impulse alone moves a key along its axis.
        mu = model.shape_material_mu.numpy()
        mu[np.isin(model.shape_body.numpy(), layout.key_leaves)] = 0.0
        model.shape_material_mu.assign(mu)
        model.rigid_contact_max = worlds
        solvers = [recipe_solver(model, scalar=False), recipe_solver(model, scalar=True)]
        self._assert_routes(*solvers)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [newton.Contacts(model.rigid_contact_max, 0, device=device) for _ in solvers]
        for control, contact, pair in zip(controls, contacts, states, strict=True):
            control.joint_target_q.zero_()
            control.joint_target_qd.zero_()
            contact.rigid_contact_count.zero_()
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])

        def advance():
            for index, solver in enumerate(solvers):
                current, following = states[index]
                current.clear_forces()
                solver.step(current, following, controls[index], contacts[index], DT)
                states[index] = [following, current]

        # A free step publishes the motion axes; the pusher presses the key closest to vertical.
        advance()
        axes = solvers[1].joint_S_s.numpy()[layout.key_dofs, :3].astype(np.float64)
        np.testing.assert_allclose(solvers[0].joint_S_s.numpy()[layout.key_dofs, :3], axes, rtol=1.0e-6, atol=1.0e-7)
        key = int(np.argmax(np.abs(axes[:PRESS_LEAVES, 2])))
        self.assertGreater(abs(axes[key, 2]), 0.5)
        pressed = layout.key_leaves.reshape(worlds, -1)[:, key]
        coordinates = layout.key_coordinates.reshape(worlds, -1)[:, key]
        dofs = layout.key_dofs.reshape(worlds, -1)[:, key]
        # Both routes see one pusher: zero gap at the head's key, its normal from the key up to the pusher.
        for contact in contacts:
            install_held_contacts(model, states[0][0], contact, [(int(body), -1) for body in pressed])
            contact.rigid_contact_normal.assign(np.tile(np.array([0.0, 0.0, 1.0], np.float32), (worlds, 1)))
        top = contacts[0].rigid_contact_point1.numpy().copy()
        rate, descend, hold, release = 1.0e-4, 16, 8, 40
        travel, impulses = [[], []], [[], []]
        for step in range(descend + hold):
            depth = rate * min(step + 1, descend)
            for contact in contacts:
                contact.rigid_contact_point1.assign(top - np.array([0.0, 0.0, depth], np.float32))
            advance()
            for index, solver in enumerate(solvers):
                impulses[index].append(_normal_impulses(self, solver, worlds))
                travel[index].append(states[index][0].joint_q.numpy()[coordinates].copy())
        released = [(state[0].joint_q.numpy()[coordinates], state[0].joint_qd.numpy()[dofs]) for state in states]
        for contact in contacts:
            contact.rigid_contact_count.zero_()
        spring = [[], []]
        for _step in range(release):
            advance()
            for index in range(2):
                spring[index].append(states[index][0].joint_q.numpy()[coordinates].copy())
        travel, impulses, spring = (np.asarray(values, dtype=np.float64) for values in (travel, impulses, spring))
        for values in (travel, impulses, spring):
            self.assertTrue(np.all(np.isfinite(values)))
        # The pusher holds each key against its drive, deeper than the key sags under gravity.
        self.assertTrue(np.all(impulses[:, descend:] > 0.0))
        self.assertTrue(np.all(np.abs(travel[:, -1]) > 1.0e-3))
        np.testing.assert_allclose(impulses[1], impulses[0], rtol=1.0e-5, atol=1.0e-9)
        np.testing.assert_allclose(travel[1], travel[0], rtol=1.0e-5, atol=1.0e-9)
        np.testing.assert_allclose(spring[1], spring[0], rtol=1.0e-5, atol=1.0e-9)
        pressed_axes = axes.reshape(worlds, -1, 3)[:, key]
        for index in range(2):
            expected = _key_law(model, pressed, dofs, pressed_axes, *released[index], release)
            np.testing.assert_allclose(spring[index], expected, rtol=1.0e-4, atol=1.0e-7)

    def test_gripper_contact_matches_the_prismatic_route(self):
        """Press a key with the arm's tip through a contact: key travel and impulses track the head route.

        The arm takes the held owner's float order, so the bound is relative
        (1e-3) rather than the keys' FMA bound; a defect in the scalar family's
        contact response would miss it by orders of magnitude.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=STAR_LEAVES, shapes=True)
        layout = _Layout(model)
        keys = layout.key_leaves.reshape(model.world_count, -1)[:, 4]
        pairs = [(int(tip), int(key)) for tip, key in zip(layout.arm_tips, keys, strict=True)]
        model.rigid_contact_max = len(pairs)
        solvers = [recipe_solver(model, scalar=False), recipe_solver(model, scalar=True)]
        self._assert_routes(*solvers)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [newton.Contacts(model.rigid_contact_max, 0, device=device) for _ in solvers]
        for index, pair in enumerate(states):
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
            install_held_contacts(model, pair[0], contacts[index], pairs)
        coordinates = layout.key_coordinates.reshape(model.world_count, -1)[:, 4]
        travel, impulses = [[], []], [[], []]
        for _step in range(12):
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[layout.arm_tips, 2] = -2.0
            for index, solver in enumerate(solvers):
                current, following = states[index]
                current.body_f.assign(force)
                solver.step(current, following, controls[index], contacts[index], DT)
                states[index] = [following, current]
                world, slot = solver.contact_world.numpy()[: len(pairs)], solver.contact_slot.numpy()[: len(pairs)]
                self.assertTrue(np.all(slot >= 0))
                impulses[index].append(solver.impulses.numpy()[world, slot].copy())
                travel[index].append(following.joint_q.numpy()[coordinates].copy())
        travel, impulses = np.asarray(travel), np.asarray(impulses)
        self.assertTrue(np.all(np.isfinite(impulses)) and np.all(np.isfinite(travel)))
        self.assertGreater(float(impulses[0].max()), 0.0, "The tip must press the key")
        np.testing.assert_allclose(impulses[1], impulses[0], rtol=1.0e-3, atol=1.0e-7)
        np.testing.assert_allclose(travel[1], travel[0], rtol=1.0e-3, atol=1.0e-7)

    def test_mixed_leaves_follow_ordinary_dynamics(self):
        """Own a star of prismatic and revolute leaves end to end and track ordinary dynamics.

        The prismatic opt-ins do not admit revolute leaves, so the reference is
        ordinary dynamics (cached_dynamics='off'); the scalar family runs its
        general kernels: revolute motion axes about the published origin,
        their inverse mass from the COM inertia and their COM velocities. Floor
        contacts on both leaf kinds read those axes and the origin, and a root
        re-pose must republish them. Revolute torques differ by the velocity
        product's rounding, which ordinary dynamics evaluates and the family
        does not (it is zero on the leaf's own axis), so leaves compare at a
        physics tolerance; the held arm at the held-solve bound.
        """
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=STAR_LEAVES, kinds=("prismatic", "revolute"), shapes=True)
        layout = _Layout(model)
        starts, child = model.articulation_start.numpy(), model.joint_child.numpy()
        # Even leaves are prismatic and odd ones revolute.
        floors = [int(child[starts[a] + 1 + k]) for a in layout.branches for k in (0, 3, 6, 9)]
        model.rigid_contact_max = len(floors)
        solvers = [
            recipe_solver(model, scalar=False, prismatic=False, cached_dynamics="off"),
            recipe_solver(model, scalar=True, prismatic=False),
        ]
        self.assertIsNone(solvers[0]._tree_kinetic_state)
        owner = solvers[1]._tree_kinetic_state
        self.assertIsInstance(owner.scalar, sb.ScalarBranchState)
        self.assertEqual(owner.scalar.plan.kinds, ("prismatic", "revolute"))
        self.assertIs(owner.scalar.predictor_kernel, sb.predict_scalar_branches)
        self.assertIs(owner.scalar.finish_kernel, sb.finish_scalar_branches)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [newton.Contacts(model.rigid_contact_max, 0, device=device) for _ in solvers]
        for index, pair in enumerate(states):
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
            install_held_contacts(model, pair[0], contacts[index], [(body, -1) for body in floors])
        leaves = {"rtol": 1.0e-4, "atol": 1.0e-6}
        for step in range(12):
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[floors[1::2], 2] = -0.2 * (step + 1)
            for index, pair in enumerate(states):
                pair[0].body_f.assign(force)
                controls[index].joint_target_q.fill_(-0.0003 * (step + 1))
            if step == 6:
                frames = model.joint_X_p.numpy()
                frames[starts[layout.branches], :3] += (0.002, -0.001, 0.001)
                frames[starts[layout.branches], 3:] = np.asarray(wp.quat_rpy(0.32, -0.18, 0.41))
                model.joint_X_p.assign(frames)
                for solver in solvers:
                    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            for index, solver in enumerate(solvers):
                current, following = states[index]
                solver.step(current, following, controls[index], contacts[index], DT)
                states[index] = [following, current]
            tag = f"step {step}"
            for name, index in (("joint_q", layout.key_coordinates), ("joint_qd", layout.key_dofs)):
                _close(self, getattr(states[0][0], name), getattr(states[1][0], name), index, f"{tag} {name}", **leaves)
            for name in ("body_q", "body_qd"):
                _close(
                    self,
                    getattr(states[0][0], name),
                    getattr(states[1][0], name),
                    layout.key_bodies,
                    f"{tag} {name}",
                    **leaves,
                )
                _arm_within(
                    self,
                    getattr(states[0][0], name),
                    getattr(states[1][0], name),
                    layout.arm_bodies,
                    f"{tag} arm {name}",
                )
            # Same motion axes and origins: the contact producer reads both.
            _close(self, solvers[0].joint_S_s, solvers[1].joint_S_s, layout.key_dofs, f"{tag} joint_S_s", **leaves)
            _close(
                self,
                solvers[0].articulation_origin,
                solvers[1].articulation_origin,
                layout.branches,
                f"{tag} origin",
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            for solver in solvers:
                solver.check_constraint_capacity()
        self.assertGreater(float(np.abs(states[1][0].body_qd.numpy()[layout.key_leaves, 3:]).max()), 1.0e-4)

    def test_route_record_names_the_scalar_group(self):
        """Record the scalar group, its precedence over the prismatic opt-ins, and its decline without an owner."""
        device = wp.get_device("cuda:0")
        model = build_branch_model(device, worlds=2, leaves=STAR_LEAVES, shapes=True)
        model.rigid_contact_max = 1
        record = recipe_solver(model, scalar=True).get_path_selection()
        json.dumps(record, allow_nan=False)
        held, publication = record["routes"]["held_tree"], record["routes"]["publication"]
        self.assertEqual(held["owner"], "mixed")
        self.assertTrue(held["scalar_branches"]["enabled"])
        self.assertIn("supersede FEATHER_PGS_PRISMATIC", held["scalar_branches"]["reason"])
        groups = {group["operator"]: group for group in held["groups"]}
        self.assertEqual(set(groups), {"TreeHeldOperator", "ScalarBranchState"})
        self.assertEqual(
            (groups["ScalarBranchState"]["dofs"], groups["ScalarBranchState"]["kinds"]), (STAR_LEAVES, ["prismatic"])
        )
        self.assertEqual(groups["ScalarBranchState"]["kernels"]["predict"], sb.predict_prismatic_branches.key)
        self.assertEqual(groups["TreeHeldOperator"]["dofs"], ARM_DOFS)
        self.assertEqual(len(record["measurements"]["held_tree_lane_timing_us"]), 2)
        self.assertEqual((publication["owner"], publication["prismatic_publication"]), ("cached", False))
        self.assertFalse(publication["prismatic_linear_state"])
        head = recipe_solver(model, scalar=False).get_path_selection()["routes"]
        self.assertIsNone(head["held_tree"]["owner"])
        self.assertFalse(head["held_tree"]["scalar_branches"]["enabled"])
        self.assertIn("Off: FEATHER_PGS_SCALAR_BRANCHES=1", head["held_tree"]["scalar_branches"]["reason"])
        self.assertTrue(head["publication"]["prismatic_linear_state"])
        # Without the held-tree owner the branches stay on ordinary dynamics with the recipe's opt-ins:
        # every route is the one of the same solver built without the flag, apart from the reason.
        declined = recipe_solver(model, scalar=True, cached_dynamics="off")
        self.assertIsNone(declined._scalar_branches)
        routes = declined.get_path_selection()["routes"]
        self.assertFalse(routes["held_tree"]["scalar_branches"]["enabled"])
        self.assertIn("did not take them", routes["held_tree"]["scalar_branches"]["reason"])
        self.assertTrue(routes["publication"]["prismatic_linear_state"])
        plain = recipe_solver(model, scalar=False, cached_dynamics="off").get_path_selection()["routes"]
        for record in (routes, plain):
            del record["held_tree"]["scalar_branches"]["reason"]
        self.assertEqual(routes, plain)


if __name__ == "__main__":
    unittest.main()

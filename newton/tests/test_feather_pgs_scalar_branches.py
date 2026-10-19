# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scalar branches: world-fixed roots over one-coordinate leaves (``scalar_branches.py``).

The CPU tests compare the scalar kernels, bound through ``ScalarBranchState``,
with the head chain they replace, launched on the same inputs: the prismatic
linear state route for prismatic leaves (bitwise) and ordinary dynamics for
revolute leaves.
"""

import dataclasses
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels as fk
from newton._src.solvers.feather_pgs import scalar_branches as sb
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.prismatic_linear_state import (
    finalize_prismatic_linear_state,
    get_linear_inverse_mass_kernel,
    repair_prismatic_linear_state,
)
from newton._src.solvers.feather_pgs.prismatic_publication import PrismaticPublicationPlan
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 240.0
GRAVITY = (0.7, -1.2, -9.1)
# Fields the scalar family stops writing for its bodies; no consumer on its route reads them.
DROPPED_FIELDS = ("body_q_com", "body_v_s", "body_a_s", "body_f_s", "body_I_s", "_body_inertia_terms")


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
):
    """Per world: a tilted root over one-coordinate leaves (a keyboard's class) beside a fixed-base six-joint arm.

    Leaves carry nontrivial axes, anchors, COMs, drives, limits, armature,
    springs and damping. ``kinds`` cycles the leaves' joint kinds. ``root`` is
    ``"fixed"``, ``"d6"`` (zero coordinates), ``"revolute"`` or ``"free"``.
    ``chain`` hangs the second leaf from the first, ``multi`` makes the last
    leaf a two-coordinate D6 joint and ``descendant`` welds a body to the first
    leaf. ``shapes`` adds a box to each leaf and a sphere to the arm's tip.
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

    def test_admits_the_keyboard_shaped_fixture(self):
        """Admit the 108-key group of the two-world 108+6 fixture, whatever the arm's root."""
        from newton.tests.test_feather_pgs_compact_contact import build_solver_fixture  # noqa: PLC0415

        model = build_solver_fixture("cpu")
        kinds, reason = sb.branch_topology(model, np.array([0, 2]), model.articulation_start.numpy()[1:])
        self.assertIsNotNone(kinds, reason)
        self.assertEqual(kinds.shape, (2, 108))
        self.assertTrue(np.all(kinds == int(newton.JointType.PRISMATIC)))

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
        """Report the request, the diagonal group and velocity limits before the topology."""
        model = build_branch_model(worlds=2)
        solver = cpu_solver(model, 7)
        self.assertIn("Off: FEATHER_PGS_SCALAR_BRANCHES=1", sb.plan_scalar_branches(solver, requested=False)[1])
        solver._compact_diagonal_mass_size = None
        plan, reason = sb.plan_scalar_branches(solver, requested=True)
        self.assertIsNone(plan)
        self.assertIn("No diagonal-mass group", reason)
        # A model without a diagonal group gives the same reason with the request off.
        self.assertEqual(sb.plan_scalar_branches(solver, requested=False)[1], reason)
        solver._compact_diagonal_mass_size = 7
        solver.enable_joint_velocity_limits = True
        self.assertIn("velocity limits", sb.plan_scalar_branches(solver, requested=True)[1])


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


if __name__ == "__main__":
    unittest.main()

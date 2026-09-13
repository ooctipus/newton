# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live model/descriptor regressions without captured inputs or a GPU."""

import importlib
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kinetic_live_bindings import LiveBindings, _disjoint_states
from newton._src.solvers.feather_pgs.kinetic_live_plan import PARENTS, SCALAR_BODIES, build_live_plan
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS, _FeatherPGSModelPlan


def bound_call(worlds=2):
    """Construct real live arrays with the explicitly CPU-only response ABI shim."""
    from newton._src.solvers.feather_pgs.kinetic_live_owner import (  # noqa: PLC0415
        KineticWorldOwner,
    )

    solver = live_solver(worlds)
    owner = KineticWorldOwner(solver, build_live_plan(solver))
    first, second = solver.model.state(), solver.model.state()
    control = solver.model.control()
    contacts = newton.Contacts(rigid_contact_max=256, soft_contact_max=0, device="cpu")
    slots = owner._slots(first, second)
    slots.requested.fill_(1)
    slots.next_schedule.geometry_requested.fill_(0)
    slots.next_generation.fill_(1)
    call = owner.bindings.bind_step(first, second, control, contacts, 1.0 / 240.0, slots)
    return solver, owner, first, second, control, contacts, call


def live_model(worlds=2):
    """Construct the exact supported topology from public live builder inputs."""
    template = newton.ModelBuilder()
    bodies, joints = [], []
    for local in range(30):
        body = template.add_link(
            mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)), com=wp.vec3(0.01, -0.02, 0.03)
        )
        bodies.append(body)
        parent = -1 if PARENTS[local] < 0 else bodies[PARENTS[local]]
        kwargs = {
            "parent": parent,
            "child": body,
            "parent_xform": wp.transform(wp.vec3(0.02, 0.01, 0.05), wp.quat_identity()),
        }
        if local in SCALAR_BODIES:
            joint = template.add_joint_revolute(
                **kwargs,
                axis=wp.vec3(0.0, 0.0, 1.0),
                target_ke=2.0,
                target_kd=0.2,
                armature=0.01,
                limit_lower=-1.0,
                limit_upper=1.0,
            )
        else:
            joint = template.add_joint_fixed(**kwargs)
        joints.append(joint)
    template.add_articulation(joints)
    for prescribed in (False, True):
        body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)), is_kinematic=prescribed)
        template.add_articulation([template.add_joint_free(body)])
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds, spacing=(2.0, 2.0, 0.0))
    model = builder.finalize(device="cpu")
    model.rigid_contact_max = 256
    model.rigid_body_max_linear_velocity = wp.full(model.body_count, float("inf"), dtype=float, device="cpu")
    model.rigid_body_max_angular_velocity = wp.full(model.body_count, float("inf"), dtype=float, device="cpu")
    return model


def live_solver(worlds=2):
    """Use the unchanged live solver constructor and admitted fixed recipe."""
    # The original constructor explicitly forbids matrix_free on CPU. Split
    # construction supplies real model/topology/buffers for descriptor tests;
    # these tests never claim that the empty original CPU generic solves.
    original_plan = _FeatherPGSModelPlan.build

    def response_plan(model, kinematic_dof_mask, **kwargs):
        return original_plan(model, kinematic_dof_mask, enable_prescribed_response=True)

    # Select the exact original CUDA response plan while retaining a supported
    # CPU buffer constructor. This is an explicit ABI test, not CPU GS proof.
    with patch.object(_FeatherPGSModelPlan, "build", side_effect=response_plan):
        solver = SolverFeatherPGS(
            live_model(worlds),
            pgs_mode="split",
            pgs_iterations=8,
            dense_max_constraints=192,
            mf_max_constraints=64,
            enable_joint_limits=True,
            joint_limit_activation_gap=0.0,
            fuse_joint_velocity_limits=False,
            mf_gs_incremental_rows=0,
            update_mass_matrix_interval=2,
            use_parallel_streams=False,
        )
    solver._compute_world_response_dof_mapping(solver.model)
    # Allocate the original response29 buffers for the explicit CPU ABI arm.
    # No original step is run through this diagnostic-only constructor shim.
    solver.pgs_mode = "matrix_free"
    solver._allocate_world_buffers(solver.model)
    # Original CPU split construction does not request lower inverses. Provide
    # their ordinary exact shapes for the explicitly diagnostic native CPU arm.
    for size in (23, 6):
        solver.Linv_by_size[size] = wp.zeros_like(solver.L_by_size[size])
    return solver


class TestKineticLiveBindings(unittest.TestCase):
    """Require package-local live bindings before testing their ownership."""

    def test_live_api_exists(self):
        """Require live plan and binding entry points without fixture imports."""
        plan = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_live_plan")
        binding = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_live_bindings")
        self.assertTrue(callable(plan.build_live_plan))
        self.assertTrue(callable(binding.LiveBindings))

    def test_live_plan_complete_physical_ownership(self):
        """Bind actual model indices, including free and prescribed coordinates."""
        solver = live_solver()
        plan = build_live_plan(solver)
        self.assertEqual(plan.data.dof_ids.shape, (2, 35))
        self.assertEqual(plan.data.body_ids.shape, (2, 32))
        np.testing.assert_array_equal(plan.host["dof_ids"][:, :29], solver.world_dof_indices.numpy())
        self.assertEqual(len(np.unique(plan.host["dof_ids"])), 70)
        rows = plan.drive_row_by_dof.numpy()[plan.host["dof_ids"]]
        self.assertTrue(np.all(rows[:, :23] >= 0))
        np.testing.assert_array_equal(rows[:, 23:], -1)

    def test_constructor_reuses_large_solver_buffers(self):
        """Allocate only compact ownership metadata and reuse original panels."""
        solver = live_solver()
        binding = LiveBindings(solver, build_live_plan(solver))
        self.assertIs(binding.selector, solver._local_solve_owner)
        self.assertIs(binding.output.v_hat, solver.v_hat)
        self.assertEqual(binding.row_valid.shape, (2, 192))
        self.assertEqual(binding.endpoint_twists.shape, (64,))

    def test_cross_field_partial_alias_rejected(self):
        """Reject overlapping body or generalized source ranges before writes."""
        model = live_model()
        first, second = model.state(), model.state()
        self.assertTrue(_disjoint_states(first, second))
        second.body_qd = first.body_f
        self.assertFalse(_disjoint_states(first, second))

    def test_current_descriptors_and_rebind_no_allocations(self):
        """Alias all current inputs and original panels without bind-time allocation."""
        solver, owner, first, second, control, contacts, call = bound_call()
        self.assertIs(call.inputs.joint_q, first.joint_q)
        self.assertIs(call.inputs.body_f, first.body_f)
        self.assertIs(call.inputs.joint_f, control.joint_f)
        self.assertIs(call.current_publication.body_q, first.body_q)
        self.assertIs(call.next_publication.body_q, second.body_q)
        self.assertIs(call.rows.out.response, solver.Y_world)
        self.assertIs(call.rows.out.physical_J, solver.J_world)
        self.assertIs(call.services.body_Hinv, solver.mf_body_Hinv)
        self.assertIs(call.free.lower, solver.L_by_size[6])
        self.assertIs(call.free.R, solver.R_by_size[6])
        self.assertIs(call.output.joint_qdd, solver.joint_qdd)
        self.assertEqual(len(call.kernel_arguments["finish"]), 6)
        with (
            patch.object(wp, "empty", side_effect=AssertionError("bind allocation")),
            patch.object(wp, "zeros", side_effect=AssertionError("bind allocation")),
            patch.object(wp, "array", side_effect=AssertionError("bind allocation")),
        ):
            again = owner.bindings.bind_step(first, second, control, contacts, 1.0 / 240.0, call.slots)
        self.assertIs(again.inputs.joint_f, control.joint_f)

    def test_cold_current_refresh_predict_and_next_state(self):
        """Run the native kinetic lifetime with no captured bias or held seed."""
        _solver, _owner, first, second, _control, _contacts, call = bound_call(worlds=3)
        call.construct_launch()
        call.refresh_launch()
        call.free.launch()
        call.predict_launch()
        np.testing.assert_array_equal(call.slots.schedule.status.numpy(), 0)
        np.testing.assert_array_equal(call.refresh.status.numpy(), 0)
        np.testing.assert_array_equal(call.output.status.numpy(), 0)
        call.zero.launch()
        np.testing.assert_array_equal(call.rows.state.resolved.numpy(), 1)
        call.solve.setup()
        call.solve.allocate()
        call.solve.build_rows()
        call.solve.prepare_mf()
        call.solve.qualify()
        call.solve.materialize()
        call.solve.solve()
        call.finish_launch()
        np.testing.assert_array_equal(call.solve.guard.frame_status.numpy(), 0)
        np.testing.assert_array_equal(call.slots.next_schedule.status.numpy(), 0)
        self.assertTrue(np.isfinite(second.body_q.numpy()).all())
        self.assertTrue(np.isfinite(second.body_qd.numpy()).all())
        self.assertTrue(np.isfinite(second.joint_qd.numpy()).all())
        self.assertTrue(np.any(second.joint_qd.numpy() != first.joint_qd.numpy()))

    def test_current_force_and_control_reuse_held_operator(self):
        """Read changed current controls/body wrenches while retaining exact held mass."""
        from newton._src.solvers.feather_pgs.kinetic_predictor import unpack_T  # noqa: PLC0415

        solver, owner, first, _second, control, _contacts, call = bound_call()
        call.construct_launch()
        call.refresh_launch()
        call.free.launch()
        call.predict_launch()
        before = call.output.joint_qdd.numpy().astype(np.float64)
        held = call.slots.held.T.numpy().copy()
        ids = owner.host_plan.host["dof_ids"][0, :23]
        body = int(owner.host_plan.host["body_ids"][0, 10])
        tau = np.zeros(23, np.float64)
        applied = control.joint_f.numpy()
        applied[ids[0]] = 0.7
        control.joint_f.assign(applied)
        tau[0] += 0.7
        target = control.joint_target_q.numpy()
        target[ids[1]] += 0.03
        control.joint_target_q.assign(target)
        tau[1] += float(solver.model.joint_target_ke.numpy()[ids[1]]) * float(target[ids[1]])
        force = np.array([0.3, -0.2, 0.1, 0.07, 0.02, -0.03], np.float32)
        body_f = first.body_f.numpy()
        body_f[body] = force
        first.body_f.assign(body_f)
        offset = call.slots.current.com_offset.numpy()[0, 10].astype(np.float64)
        wrench = np.r_[force[:3], force[3:].astype(np.float64) + np.cross(offset, force[:3])]
        mask = int(solver.body_response_dof_mask.numpy()[body])
        axes = call.slots.current.axes.numpy()[0].astype(np.float64)
        for local in range(23):
            if mask & (1 << local):
                tau[local] += axes[local] @ wrench
        call.predict_launch()
        np.testing.assert_array_equal(call.output.status.numpy(), 0)
        np.testing.assert_array_equal(call.slots.held.T.numpy(), held)
        t = unpack_T(held[0].astype(np.float64))
        expected = t.T @ (t @ tau)
        scale = 1.0 + np.abs(t.T) @ (np.abs(t) @ np.abs(tau))
        actual = call.output.joint_qdd.numpy()[ids].astype(np.float64) - before[ids]
        self.assertLessEqual(float(np.max(np.abs(actual - expected) / scale)), 2.0**-17)

    def test_nonuniform_gravity_and_output_alias_rejected(self):
        """Reject unsupported gravity and internal-output aliases before dispatch."""
        solver, owner, first, second, control, contacts, call = bound_call()
        second.joint_qd = solver.v_hat
        with self.assertRaisesRegex(ValueError, "overlap"):
            owner.bindings.bind_step(first, second, control, contacts, 1.0 / 240.0, call.slots)
        gravity = solver.model.gravity.numpy()
        gravity[-1, 2] += 1.0
        solver.model.gravity.assign(gravity)
        with self.assertRaisesRegex(ValueError, "uniform gravity"):
            build_live_plan(solver)


if __name__ == "__main__":
    unittest.main()

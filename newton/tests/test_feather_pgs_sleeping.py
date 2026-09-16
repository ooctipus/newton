# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioral controls for topology-admitted, contact-free FPGS sleeping."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.feather_pgs.solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.sleep_topology import build_sleep_topology
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_compact_contact import build_solver_fixture, install_contacts
from newton.tests.test_feather_pgs_prismatic_publication import _build_model, _solver

DT = 1 / 240


def _physical_pair(test, device, *, leaves=108):
    """Reuse the admitted contact fixture, now at a gravity/PD equilibrium."""
    model = build_solver_fixture(device, leaves=leaves)
    model.gravity.assign(np.tile(np.array((0.0, 0.0, -1.0), np.float32), (model.gravity.shape[0], 1)))
    model.joint_q.zero_()
    model.joint_qd.zero_()
    model.joint_armature.fill_(0.02)
    model.joint_spring_stiffness.fill_(1.0)
    model.joint_spring_ref.zero_()
    model.joint_damping.fill_(0.03)
    joints = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.PRISMATIC))
    bodies = model.joint_child.numpy()[joints]
    dofs = model.joint_qd_start.numpy()[joints]
    targets = model.joint_target_q_start.numpy()[joints]
    cases = []
    for enabled in (False, True):
        with (
            mock.patch.object(solver_module, "_COMPACT_CONTACT_BOUNDARY", True),
            mock.patch.object(solver_module, "_PRISMATIC_PUBLICATION", True),
            mock.patch.object(solver_module, "_SPARSE_CONTACT_DIRECT", True),
            mock.patch.object(solver_module, "_PRISMATIC_LINEAR_STATE", True),
            mock.patch.object(solver_module, "_SLEEPING", enabled),
        ):
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_iterations=8,
                update_mass_matrix_interval=2,
                use_parallel_streams=True,
                enable_joint_limits=True,
                lazy_kinematics=True,
                dense_max_constraints=64,
                mf_max_constraints=64,
            )
        test.assertTrue(solver._prismatic_linear_state)
        test.assertTrue(solver._direct_compact_diagonal_inertia)
        state, out, control = model.state(), model.state(), model.control()
        target = control.joint_target_q.numpy()
        target[targets] = model.body_mass.numpy()[bodies] / model.joint_target_ke.numpy()[dofs]
        control.joint_target_q.assign(target)
        target_velocity = np.zeros(model.joint_dof_count, dtype=np.float32)
        target_velocity[dofs[2]] = 0.02  # This component must never acquire a static-target lease.
        control.joint_target_qd.assign(target_velocity)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        cases.append(
            SimpleNamespace(
                solver=solver,
                state=state,
                out=out,
                control=control,
                contacts=newton.Contacts(model.rigid_contact_max, 0, device=device),
            )
        )
    test.assertIsNone(cases[0].solver._sleeping)
    test.assertTrue(cases[1].solver._sleeping.enabled)
    return model, cases, joints, bodies, dofs, targets


def _step(case, *, inplace=False):
    output = case.state if inplace else case.out
    case.solver.step(case.state, output, case.control, case.contacts, DT)
    if not inplace:
        case.state, case.out = output, case.state


def _tick(cases, *, inplace=False):
    for case in cases:
        _step(case, inplace=inplace)
        _step(case, inplace=inplace)
        case.solver.publish_kinematics(case.state)


def _compare(test, model, cases):
    """Use the existing linear-state physical tolerances, including public FK."""
    for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
        np.testing.assert_allclose(
            getattr(cases[1].state, field).numpy(),
            getattr(cases[0].state, field).numpy(),
            rtol=3e-5,
            atol=5e-6,
            err_msg=field,
        )
    for field in ("_diagonal_inverse_mass",):
        np.testing.assert_allclose(
            getattr(cases[1].solver, field).numpy(),
            getattr(cases[0].solver, field).numpy(),
            rtol=3e-5,
            atol=5e-6,
            err_msg=field,
        )
    np.testing.assert_allclose(
        cases[1].solver.L_by_size[6].numpy(), cases[0].solver.L_by_size[6].numpy(), rtol=3e-5, atol=5e-6
    )
    for case in cases:
        case.solver.check_constraint_capacity()
        reference = model.state()
        newton.eval_fk(model, case.state.joint_q, case.state.joint_qd, reference)
        for field in ("body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(case.state, field).numpy(),
                getattr(reference, field).numpy(),
                rtol=3e-6,
                atol=3e-6,
                err_msg="public " + field,
            )


def _settle(test, model, cases, bodies, joints):
    for _ in range(10):
        _tick(cases)
    _compare(test, model, cases)
    controller = cases[1].solver._sleeping
    quiet = np.delete(np.arange(bodies.size), 2)
    np.testing.assert_array_equal(controller.body_awake.numpy()[bodies[quiet]], 0)
    np.testing.assert_array_equal(controller.joint_awake.numpy()[joints[quiet]], 0)
    np.testing.assert_array_equal(cases[1].state.joint_qd.numpy()[model.joint_qd_start.numpy()[joints[quiet]]], 0.0)
    test.assertEqual(int(controller.body_awake.numpy()[bodies[2]]), 1)
    coupled = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))
    np.testing.assert_array_equal(controller.body_awake.numpy()[model.joint_child.numpy()[coupled]], 1)


class TestFeatherPGSSleeping(unittest.TestCase):
    def test_authored_wake_preserves_scalar_mass_request_guards(self):
        """Keep authored-state wake writes inside the scalar mass-request flag."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual sleeping wake and guarded native writes require CUDA")
        for device in devices:
            model, cases, _joints, _bodies, _dofs, _targets = _physical_pair(self, device)
            candidate = cases[1]
            self.assertEqual(candidate.solver._mass_update_requested.shape, (1,))
            guarded = np.full(model.articulation_count + 4, -17, dtype=np.int32)
            guarded[0] = 0
            storage = wp.array(guarded, dtype=wp.int32, device=device)
            candidate.solver._mass_update_requested = storage[:1]
            # The first real step authors all components, including world1's
            # articulation2. Keep the owning allocation alive until the readback.
            _step(candidate)
            np.testing.assert_array_equal(storage.numpy()[1:], guarded[1:])

    def test_topology_independence_without_branch_count_specialization(self):
        """Split anchored siblings but retain shared responsive ancestry."""
        for leaves, locked in ((3, False), (7, True), (11, False)):
            with self.subTest(leaves=leaves, locked=locked):
                model, joints, bodies = _build_model(leaves=leaves, locked_d6=locked)
                model.joint_target_ke.zero_()
                model.joint_target_kd.zero_()
                plan = build_sleep_topology(_solver(model, enabled=True))
                components = plan.body_component_host[bodies]
                self.assertTrue(np.all(components >= 0))
                self.assertEqual(np.unique(components).size, leaves)
                self.assertEqual(int(plan.body_component_host[0]), -1)
                self.assertEqual(int(plan.joint_component_host[joints[0]]), -1)
                np.testing.assert_array_equal(plan.joint_component_host[joints[1:]], components)
                if leaves == 3:
                    # A normalized solver window must not grant sleep ownership
                    # to raw global (-1) articulations.
                    np.testing.assert_array_equal(model.articulation_world.numpy(), -1)
                    np.testing.assert_array_equal(plan.component_world_host, -1)
                    np.testing.assert_array_equal(plan.component_eligible_host, 0)
        model, _joints, _bodies = _build_model(leaves=3, worlds=2)
        plan = build_sleep_topology(_solver(model, enabled=True))
        prismatic = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.PRISMATIC))
        components = plan.joint_component_host[prismatic]
        np.testing.assert_array_equal(plan.component_world_host[components], np.repeat((0, 1), 3))
        np.testing.assert_array_equal(plan.component_eligible_host[components], 1)
        model, _joints, bodies = _build_model(leaves=7, chain=True)
        model.joint_target_ke.zero_()
        model.joint_target_kd.zero_()
        plan = build_sleep_topology(_solver(model, enabled=True))
        self.assertEqual(int(plan.body_component_host[bodies[0]]), int(plan.body_component_host[bodies[1]]))
        self.assertNotEqual(int(plan.body_component_host[bodies[0]]), int(plan.body_component_host[bodies[2]]))

    def test_pd_equilibrium_and_current_target_force_wake(self):
        """Keep stable PD equilibria asleep and consume changed targets/forces immediately."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual direct-diagonal sleeping consumers require CUDA")
        for device in devices:
            model, cases, joints, bodies, dofs, targets = _physical_pair(self, device)
            _settle(self, model, cases, bodies, joints)
            # Mutate current inputs without notification: all three must wake
            # before force/prediction, while their independent siblings stay asleep.
            for case in cases:
                target = case.control.joint_target_q.numpy()
                target[targets[0]] += 0.01
                case.control.joint_target_q.assign(target)
                force = np.zeros(model.joint_dof_count, np.float32)
                force[dofs[1]] = 0.3
                case.control.joint_f.assign(force)
                body_force = np.zeros((model.body_count, 6), np.float32)
                body_force[bodies[3], 2] = 0.25
                case.state.body_f.assign(body_force)
                _step(case)
            _compare(self, model, cases)
            awake = cases[1].solver._sleeping.body_awake.numpy()
            np.testing.assert_array_equal(awake[bodies[[0, 1, 2, 3]]], 1)
            np.testing.assert_array_equal(awake[bodies[4:]], 0)
            self.assertTrue(np.all(np.abs(cases[1].state.joint_qd.numpy()[dofs[[0, 1, 3]]]) > 1e-5))
            # Small valid timesteps must not turn a slowly accelerating drive
            # into a false equilibrium merely because qd remains below tolerance.
            model, cases, _joints, bodies, dofs, targets = _physical_pair(self, device)
            for case in cases:
                target = case.control.joint_target_q.numpy()
                target[targets[0]] += 1.0
                case.control.joint_target_q.assign(target)
                for _ in range(20):
                    case.solver.step(case.state, case.out, case.control, case.contacts, 1e-8)
                    case.state, case.out = case.out, case.state
            self.assertEqual(int(cases[1].solver._sleeping.body_awake.numpy()[bodies[0]]), 1)
            self.assertGreater(float(cases[1].state.joint_qd.numpy()[dofs[0]]), 0.0)
            _compare(self, model, cases)

    def test_loaded_contacts_wake_sleeping_components(self):
        """Wake from authored loaded buffers and actual first sphere-pair collision."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual current contacts and sleeping consumers require CUDA")
        for device in devices:
            model, cases, joints, bodies, _dofs, _targets = _physical_pair(self, device)
            _settle(self, model, cases, bodies, joints)
            for case in cases:
                install_contacts(model, case.state, case.contacts)
                margins = np.zeros(model.rigid_contact_max, np.float32)
                margins[:6] = 0.003
                case.contacts.rigid_contact_margin0.assign(margins)
                case.contacts.rigid_contact_margin1.assign(margins)
                _step(case)
                case.solver.update_contacts(case.contacts)
            _compare(self, model, cases)
            awake = cases[1].solver._sleeping.body_awake.numpy()
            np.testing.assert_array_equal(awake[bodies[[0, 1, 108, 109]]], 1)
            np.testing.assert_array_equal(awake[bodies[[4, 5, 110, 111]]], 0)
            force = cases[1].contacts.rigid_contact_force.numpy()[:6]
            self.assertGreater(float(np.max(np.abs(force))), 1e-5)
            np.testing.assert_allclose(force, cases[0].contacts.rigid_contact_force.numpy()[:6], rtol=3e-5, atol=5e-6)

            # Separate fresh existing spheres while settling, then generate the
            # first impact through CollisionPipeline, not install_contacts.
            model, cases, joints, bodies, dofs, targets = _physical_pair(self, device)
            model.shape_gap.zero_()
            pipelines = []
            moving = np.array((0, 108))
            for case in cases:
                q = case.state.joint_q.numpy()
                q[dofs[moving]] = 0.04
                case.state.joint_q.assign(q)
                target = case.control.joint_target_q.numpy()
                target[targets[moving]] += 0.04 * (1.0 + 1.0 / 12.0)
                case.control.joint_target_q.assign(target)
                newton.eval_fk(model, case.state.joint_q, case.state.joint_qd, case.state)
                pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase="explicit",
                    shape_pairs_filtered=wp.array(((0, 1), (4, 5)), dtype=wp.vec2i, device=device),
                    rigid_contact_max=8,
                )
                case.contacts = pipeline.contacts()
                pipeline.collide(case.state, case.contacts)
                self.assertEqual(int(case.contacts.rigid_contact_count.numpy()[0]), 0)
                pipelines.append(pipeline)
            _settle(self, model, cases, bodies, joints)
            for case, pipeline in zip(cases, pipelines, strict=True):
                q, qd = case.state.joint_q.numpy(), case.state.joint_qd.numpy()
                q[dofs[moving]], qd[dofs[moving]] = 0.012, -0.1
                case.state.joint_q.assign(q)
                case.state.joint_qd.assign(qd)
                newton.eval_fk(model, case.state.joint_q, case.state.joint_qd, case.state)
                pipeline.collide(case.state, case.contacts)
                self.assertGreater(int(case.contacts.rigid_contact_count.numpy()[0]), 0)
                _step(case)
            _compare(self, model, cases)
            np.testing.assert_array_equal(cases[1].solver._sleeping.body_awake.numpy()[bodies[[0, 1, 108, 109]]], 1)
            np.testing.assert_array_equal(cases[1].solver._sleeping.body_awake.numpy()[bodies[[4, 110]]], 0)

    def test_captured_masked_reset_inplace_and_model_wake(self):
        """Preserve sleep/state ownership across graph resets, in-place writes and notification."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Captured mixed-world sleeping lifecycle requires CUDA")
        for device in devices:
            model, cases, joints, bodies, dofs, _targets = _physical_pair(self, device)
            _settle(self, model, cases, bodies, joints)
            controller = cases[1].solver._sleeping
            sleeping_before = controller.sleeping.numpy().copy()
            counters_before = controller.counters.numpy().copy()
            for case in cases:
                case.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            np.testing.assert_array_equal(controller.sleeping.numpy(), sleeping_before)
            np.testing.assert_array_equal(controller.counters.numpy(), counters_before)
            _tick(cases)
            _compare(self, model, cases)
            np.testing.assert_array_equal(controller.body_awake.numpy()[bodies[4:]], 0)

            # A changed anchored root affects every independent branch below it,
            # but not the otherwise-identical articulation in the other world.
            root_body = int(model.joint_parent.numpy()[joints[0]])
            root_joint = int(np.flatnonzero(model.joint_child.numpy() == root_body)[0])
            previous_position = cases[1].state.body_q.numpy()[bodies[0], :3].copy()
            shift = np.array((0.007, -0.002, 0.0), dtype=np.float32)
            frames = model.joint_X_p.numpy()
            frames[root_joint, :3] += shift
            model.joint_X_p.assign(frames)
            for case in cases:
                case.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            np.testing.assert_array_equal(controller.body_awake.numpy()[bodies[:108]], 1)
            np.testing.assert_array_equal(controller.body_awake.numpy()[bodies[108:]], 0)
            _tick(cases)
            _compare(self, model, cases)
            np.testing.assert_allclose(
                cases[1].state.body_q.numpy()[bodies[0], :3] - previous_position, shift, rtol=3e-6, atol=3e-6
            )
            _settle(self, model, cases, bodies, joints)
            # In-place stepping must preserve the lease and all public state.
            _tick(cases, inplace=True)
            _compare(self, model, cases)
            np.testing.assert_array_equal(cases[1].solver._sleeping.body_awake.numpy()[bodies[4:]], 0)
            mask = wp.array((True, False), dtype=wp.bool, device=device)
            graphs = []
            for case in cases:
                with wp.ScopedCapture(device=device) as capture:
                    case.solver.seed_double_buffer_events()
                    case.solver.reset(case.state, mask)
                    case.solver.step(case.state, case.out, case.control, case.contacts, DT)
                    case.solver.step(case.out, case.state, case.control, case.contacts, DT)
                    case.solver.publish_kinematics(case.state)
                graphs.append(capture.graph)
            for world in (0, 1):
                mask.assign(np.array((world == 0, world == 1), np.bool_))
                for case in cases:
                    q = case.state.joint_q.numpy()
                    q[dofs[world * 108 + 4]] += 0.001
                    case.state.joint_q.assign(q)
                for graph in graphs:
                    wp.capture_launch(graph)
                _compare(self, model, cases)
                np.testing.assert_array_equal(
                    cases[1].solver._sleeping.body_awake.numpy()[bodies[world * 108 : (world + 1) * 108]], 1
                )
                if world == 0:
                    np.testing.assert_array_equal(cases[1].solver._sleeping.body_awake.numpy()[bodies[108:]], 0)
            # Captured steps leave capture-local event handles in the solver.
            # Join completed graph work before installing ordinary eager wait
            # targets; do not wait on those capture-only handles in eager code.
            wp.synchronize_device(device)
            for case in cases:
                case.solver.seed_double_buffer_events()
            # Exact authored values, not State pointer identity, govern wake-up.
            for case in cases:
                qd = case.state.joint_qd.numpy()
                qd[dofs[5]] = 0.03
                case.state.joint_qd.assign(qd)
                _step(case, inplace=True)
            _compare(self, model, cases)
            self.assertEqual(int(cases[1].solver._sleeping.body_awake.numpy()[bodies[5]]), 1)
            model.body_mass.assign(model.body_mass.numpy() * 1.1)
            model.joint_target_ke.assign(model.joint_target_ke.numpy() * 1.05)
            for case in cases:
                case.solver.notify_model_changed(
                    newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES
                )
            np.testing.assert_array_equal(cases[1].solver._sleeping.body_awake.numpy()[bodies], 1)
            _tick(cases)
            _compare(self, model, cases)


if __name__ == "__main__":
    unittest.main()

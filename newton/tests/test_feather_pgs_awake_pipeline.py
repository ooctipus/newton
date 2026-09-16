# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical integration controls for complete scalar-component ownership."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.feather_pgs.solver_feather_pgs as solver_module
import newton.tests.test_feather_pgs_sleeping as sleeping_tests
from newton.tests.test_feather_pgs_sleeping import (
    _compare,
    _physical_pair,
    _settle,
    _tick,
)


class TestFeatherPGSAwakePipeline(unittest.TestCase):
    def test_explicit_owner_switch(self):
        """Expose the experimental owner independently from the sleeping switch."""
        self.assertIsInstance(solver_module._AWAKE_PIPELINE, bool)

    def test_native_branch_counts_current_physics_and_held_mass(self):
        """Integrate two branch counts through actual scalar and articulated owners."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual scalar ownership and parallel articulated fallback require CUDA")
        for device in devices:
            for leaves in (37, 108):
                with (
                    self.subTest(device=str(device), leaves=leaves),
                    mock.patch.object(solver_module, "_AWAKE_PIPELINE", True),
                ):
                    model, cases, joints, bodies, dofs, targets = _physical_pair(self, device, leaves=leaves)
                    self.assertIsNone(cases[0].solver._awake_pipeline)
                    self.assertIsNotNone(cases[1].solver._awake_pipeline)
                    self.assertEqual(cases[1].solver._compact_diagonal_mass_size, leaves)
                    self.assertEqual(model.world_count, 2)
                    _settle(self, model, cases, bodies, joints)

                    # Exercise the scalar projection with nonunit, rotated axes,
                    # displaced COMs and finite actuator effort (not passive clamp).
                    axes = model.joint_axis.numpy()
                    axes[dofs[0]] = (0.3, -0.2, 1.4)
                    model.joint_axis.assign(axes)
                    frames = model.joint_X_p.numpy()
                    frames[joints[0], 3:] = np.asarray(wp.quat_rpy(0.2, -0.1, 0.3))
                    model.joint_X_p.assign(frames)
                    com = model.body_com.numpy()
                    com[bodies[0]] = (0.03, -0.02, 0.01)
                    model.body_com.assign(com)
                    model.body_mass.assign(model.body_mass.numpy() * 1.1)
                    effort = model.joint_effort_limit.numpy()
                    effort[dofs[0]] = 0.1
                    model.joint_effort_limit.assign(effort)
                    model.joint_target_ke.assign(model.joint_target_ke.numpy() * 1.05)
                    for case in cases:
                        case.solver.notify_model_changed(
                            newton.ModelFlags.JOINT_PROPERTIES
                            | newton.ModelFlags.JOINT_DOF_PROPERTIES
                            | newton.ModelFlags.BODY_INERTIAL_PROPERTIES
                        )
                        q, qd = case.state.joint_q.numpy(), case.state.joint_qd.numpy()
                        q[dofs[4]], qd[dofs[4]] = 0.049, 0.3  # Approach the actual upper limit.
                        qd[dofs[3]] = -0.02
                        case.state.joint_q.assign(q)
                        case.state.joint_qd.assign(qd)
                        target = case.control.joint_target_q.numpy()
                        target[targets[0]], target[targets[4]] = 0.2, 0.08
                        case.control.joint_target_q.assign(target)
                        joint_force = np.zeros(model.joint_dof_count, dtype=np.float32)
                        joint_force[dofs[1]] = 0.3
                        case.control.joint_f.assign(joint_force)
                        body_force = np.zeros((model.body_count, 6), dtype=np.float32)
                        body_force[bodies[0]] = (0.2, -0.1, 0.3, 0.0, 0.0, 0.0)
                        body_force[bodies[1]] = (0.0, 0.0, 0.0, 8.0, -3.0, 2.0)
                        case.state.body_f.assign(body_force)
                        case.out.body_f.assign(body_force)
                    held_seen = False
                    for _ in range(4):
                        _tick(cases)
                        _compare(self, model, cases)
                        held_seen |= np.any(cases[1].solver.mass_update_mask.numpy() == 0)
                    self.assertTrue(held_seen)
                    self.assertGreater(abs(float(cases[1].state.joint_qd.numpy()[dofs[0]])), 1e-5)

    def test_native_tiny_dt_refreshes_scalar_inverse_only(self):
        """Use current scalar inertia at each dt while holding articulated factors.

        The scalar owner intentionally evaluates its exact current coefficient
        even when the dt change is below the inherited mass-refresh threshold.
        This is not a bit-identity comparison with the old held scalar inverse.
        """
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual scalar ownership and held articulated factors require CUDA")
        for device in devices:
            with self.subTest(device=str(device)), mock.patch.object(solver_module, "_AWAKE_PIPELINE", True):
                model, cases, joints, bodies, dofs, targets = _physical_pair(self, device, leaves=37)
                case = cases[1]  # Execute only the candidate; the oracle is the scalar equation.
                solver = case.solver
                self.assertIsNotNone(solver._awake_pipeline)
                self.assertEqual(solver.update_mass_matrix_interval, 2)
                dof, body, joint = int(dofs[0]), int(bodies[0]), int(joints[0])
                ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
                ke[dof], kd[dof] = 1e18, 1e8
                model.joint_target_ke.assign(ke)
                model.joint_target_kd.assign(kd)
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                target = case.control.joint_target_q.numpy()
                target[targets[0]] = 0.0
                case.control.joint_target_q.assign(target)
                force = np.zeros(model.joint_dof_count, dtype=np.float32)
                force[dof] = 1.0  # Keep this component awake independently of its tiny motion.
                case.control.joint_f.assign(force)

                direct = solver._compact_diagonal_mass_size
                articulation = int(model.joint_articulation.numpy()[joint])
                group = int(np.flatnonzero(solver.group_to_art[direct].numpy() == articulation)[0])
                local = dof - int(solver._model_plan.articulation_dof_start[articulation])
                armature = float(solver.R_by_size[direct].numpy()[group, local])
                axis = model.joint_axis.numpy()[dof].astype(np.float64)
                mass = float(model.body_mass.numpy()[body])
                base_mass = mass * float(np.dot(axis, axis)) + armature
                dense_arts = solver.group_to_art[6].numpy()
                timesteps = (1e-9, 2e-9)
                self.assertLess(abs(timesteps[1] - timesteps[0]), 1e-8)
                scalar_inverses = []
                held_factor = None
                for index, dt in enumerate(timesteps):
                    solver.step(case.state, case.out, case.control, case.contacts, dt)
                    case.state, case.out = case.out, case.state
                    inverse = float(solver._diagonal_inverse_mass.numpy()[dof])
                    expected = 1.0 / (base_mass + float(ke[dof]) * dt * dt + float(kd[dof]) * dt)
                    np.testing.assert_allclose(inverse, expected, rtol=3e-5, atol=5e-6)
                    scalar_inverses.append(inverse)
                    self.assertEqual(int(solver._sleeping.body_awake.numpy()[body]), 1)
                    self.assertTrue(np.isfinite(case.state.joint_q.numpy()).all())
                    self.assertTrue(np.isfinite(case.state.joint_qd.numpy()).all())
                    if index == 0:
                        held_factor = solver.L_by_size[6].numpy()
                        np.testing.assert_array_equal(solver.mass_update_mask.numpy()[dense_arts], 1)
                    else:
                        np.testing.assert_array_equal(solver.mass_update_mask.numpy()[dense_arts], 0)
                        np.testing.assert_array_equal(solver.L_by_size[6].numpy(), held_factor)
                self.assertLess(scalar_inverses[1], 0.5 * scalar_inverses[0])

    def test_actual_contacts_and_existing_sleep_wake_contract(self):
        """Preserve loaded forces, real collision wake and tiny-dt drive behavior."""
        with mock.patch.object(solver_module, "_AWAKE_PIPELINE", True):
            sleeping_tests.TestFeatherPGSSleeping.test_loaded_contacts_wake_sleeping_components(self)
            sleeping_tests.TestFeatherPGSSleeping.test_pd_equilibrium_and_current_target_force_wake(self)

    def test_existing_graph_reset_and_notification_contract(self):
        """Preserve captured mixed resets, notifications and in-place publication."""
        with mock.patch.object(solver_module, "_AWAKE_PIPELINE", True):
            sleeping_tests.TestFeatherPGSSleeping.test_captured_masked_reset_inplace_and_model_wake(self)


if __name__ == "__main__":
    unittest.main()

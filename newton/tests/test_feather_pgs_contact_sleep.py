# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical controls for the complete contact-island sleeping owner."""

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.feather_pgs.solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.awake_component_kernels import ScalarParameters
from newton._src.solvers.feather_pgs.awake_pipeline import (
    ComponentData,
    ComponentState,
    ParameterSources,
    apply_sleep_grants,
    assess_sleep_ready,
    finish_components,
)


class TestFeatherPGSContactSleep(unittest.TestCase):
    def test_explicit_contact_island_switch(self):
        """Keep loaded-contact ownership separate from contact-free sleeping."""
        self.assertIsInstance(solver_module._CONTACT_ISLANDS, bool)

    def test_geometry_seal_total_acceleration_and_grant(self):
        """Require current-pose geometry and physical quietness before freezing."""
        device = "cpu"

        def integers(values):
            return wp.array(values, dtype=wp.int32, device=device)

        def floats(values):
            return wp.array(values, dtype=wp.float32, device=device)

        data = ComponentData()
        data.managed_sleep = 1
        for name in ("owned", "body", "joint", "coordinate", "dof"):
            setattr(data, name, integers([0, 1]))
        for name in ("sleeping", "expected_valid", "body_awake", "joint_awake"):
            setattr(data, name, integers([0, 0]))
        data.counters = integers([15, 15])
        data.can_sleep = integers([1, 1])
        data.contact = integers([1, 1])
        for name in ("expected_q", "expected_qd", "expected_target_q", "expected_target_qd"):
            setattr(data, name, floats([-77, -77]))
        data.begin_target_q = floats([0.2, -0.3])
        data.begin_target_qd = floats([0, 0])
        parameters = []
        for scale in (1.0, 2.0):
            parameter = ScalarParameters()
            parameter.axis = wp.vec3(0.0, 0.0, scale)
            parameters.append(parameter)
        data.parameters = wp.array(parameters, dtype=ScalarParameters, device=device)
        q, qd, velocity = floats([0, 0]), floats([0, 0]), floats([0, 0])
        kinematic, valid, ready = integers([0, 0]), integers([1, 1]), integers([0, 0])
        collision_q = floats([-0.001, 0])

        def assess(dt):
            wp.launch(
                assess_sleep_ready,
                dim=2,
                inputs=[data, q, qd, kinematic, velocity, dt, 1e-5, 16, collision_q, valid, ready],
                device=device,
            )

        assess(1 / 240)
        np.testing.assert_array_equal(data.counters.numpy(), [16, 16])
        np.testing.assert_array_equal(ready.numpy(), [0, 1])
        collision_q.assign(q)
        valid.assign(np.array([1, 0], np.int32))
        assess(1 / 240)
        np.testing.assert_array_equal(ready.numpy(), [1, 0])
        # Tiny dt must not turn a large physical acceleration into a quiet
        # lease simply because the one-step displacement is tiny.
        valid.fill_(1)
        velocity.assign(np.array([1e-7, 0.0], np.float32))
        assess(1e-9)
        np.testing.assert_array_equal(data.counters.numpy(), [0, 16])
        np.testing.assert_array_equal(ready.numpy(), [0, 1])
        data.counters.fill_(15)
        assess(1 / 240)
        np.testing.assert_array_equal(ready.numpy(), [1, 1])

        state = ComponentState()
        state.tau, state.qdd, state.v_hat = floats([4, 5]), floats([6, 7]), floats([8, 9])
        state.body_v = wp.array(
            [wp.spatial_vector(0.0, 0.0, 1e-7, 0.0, 0.0, 0.0)] * 2,
            dtype=wp.spatial_vector,
            device=device,
        )
        status = integers([1])
        arguments = [data, ParameterSources(), state, q, ready, status, velocity]
        wp.launch(apply_sleep_grants, dim=2, inputs=arguments, device=device)
        np.testing.assert_array_equal(data.sleeping.numpy(), [0, 0])
        np.testing.assert_array_equal(state.tau.numpy(), [4, 5])
        status.zero_()
        wp.launch(apply_sleep_grants, dim=2, inputs=arguments, device=device)
        for field in (data.sleeping, data.expected_valid):
            np.testing.assert_array_equal(field.numpy(), [1, 1])
        for field in (data.expected_qd, data.body_awake, data.joint_awake, state.tau, state.qdd, state.v_hat, velocity):
            np.testing.assert_array_equal(field.numpy(), [0, 0])
        np.testing.assert_array_equal(data.expected_q.numpy(), q.numpy())
        np.testing.assert_array_equal(data.expected_target_q.numpy(), data.begin_target_q.numpy())
        # A new grant enters finish's stationary shortcut, so it must also
        # establish the zero canonical velocity needed by cached consumers.
        np.testing.assert_array_equal(state.body_v.numpy(), np.zeros((2, 6), np.float32))

    def test_existing_sleep_response_is_not_erased(self):
        """Revoke an old lease and integrate its actual unexpected solved response."""
        device = "cpu"

        def integers(value):
            return wp.array([value], dtype=wp.int32, device=device)

        def floats(value):
            return wp.array([value], dtype=wp.float32, device=device)

        data = ComponentData()
        data.managed_sleep = 1
        for name in ("owned", "body", "joint", "coordinate", "dof"):
            setattr(data, name, integers(0))
        for name in ("sleeping", "can_sleep", "contact", "expected_valid"):
            setattr(data, name, integers(1))
        for name in ("body_awake", "joint_awake"):
            setattr(data, name, integers(0))
        data.counters = integers(16)
        for name in ("expected_q", "expected_qd", "expected_target_q", "expected_target_qd", "begin_q"):
            setattr(data, name, floats(0.0))
        data.begin_target_q, data.begin_target_qd = floats(0.2), floats(0.0)
        parameter = ScalarParameters()
        parameter.axis = wp.vec3(0.0, 0.0, 1.0)
        parameter.rest_body = wp.transform_identity()
        data.parameters = wp.array([parameter], dtype=ScalarParameters, device=device)
        source = ParameterSources()
        source.articulation = integers(0)
        state = ComponentState()
        state.tau, state.qdd, state.v_hat = floats(0.0), floats(0.0), floats(0.0)
        state.origin = wp.zeros(1, dtype=wp.vec3, device=device)
        state.body_q_com = wp.zeros(1, dtype=wp.transform, device=device)
        for name in ("joint_S", "body_v", "body_a", "body_f"):
            setattr(state, name, wp.zeros(1, dtype=wp.spatial_vector, device=device))
        q, qd, velocity = floats(0.125), floats(0.0), floats(0.0)
        kinematic, valid, ready, status = integers(0), integers(1), integers(0), integers(0)
        previous = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        q_new, qd_new = floats(-77.0), floats(-77.0)
        body_q_new = wp.zeros(1, dtype=wp.transform, device=device)
        body_qd_new = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        for response in (0.02, -0.03, np.nan, np.inf):
            with self.subTest(response=response):
                data.sleeping.fill_(1)
                data.counters.fill_(16)
                data.can_sleep.fill_(1)
                velocity.assign(np.array([response], np.float32))
                wp.launch(
                    assess_sleep_ready,
                    dim=1,
                    inputs=[data, q, qd, kinematic, velocity, 0.25, 1e-5, 16, q, valid, ready],
                    device=device,
                )
                np.testing.assert_array_equal(ready.numpy(), [0])
                wp.launch(
                    apply_sleep_grants,
                    dim=1,
                    inputs=[data, source, state, q, ready, status, velocity],
                    device=device,
                )
                for field in (data.sleeping, data.counters, data.can_sleep):
                    np.testing.assert_array_equal(field.numpy(), [0])
                for field in (data.body_awake, data.joint_awake):
                    np.testing.assert_array_equal(field.numpy(), [1])
                np.testing.assert_array_equal(velocity.numpy(), np.array([response], np.float32))
                if not np.isfinite(response):
                    continue
                wp.launch(
                    finish_components,
                    dim=1,
                    inputs=[
                        data,
                        source,
                        state,
                        q,
                        qd,
                        previous,
                        kinematic,
                        velocity,
                        0.25,
                        1e-5,
                        16,
                        q_new,
                        qd_new,
                        body_q_new,
                        body_qd_new,
                    ],
                    device=device,
                )
                np.testing.assert_allclose(q_new.numpy(), [0.125 + 0.25 * response], rtol=0.0, atol=1e-8)
                np.testing.assert_allclose(qd_new.numpy(), [response], rtol=0.0, atol=1e-8)
                np.testing.assert_allclose(state.qdd.numpy(), [response / 0.25], rtol=0.0, atol=1e-8)
                np.testing.assert_allclose(state.body_v.numpy()[0, :3], [0.0, 0.0, response], rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

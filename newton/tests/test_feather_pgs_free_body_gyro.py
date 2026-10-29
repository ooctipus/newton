# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the independent-free-body implicit gyroscopic predictor."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.free_body_gyro import (
    _implicit_gyro_velocity,
    correct_free_body_gyroscopic_predictor,
)
from newton.solvers import SolverFeatherPGS


@wp.kernel
def _correct_samples(
    inertia: wp.array[wp.mat33],
    effective: wp.array[wp.mat33],
    old: wp.array[wp.vec3],
    predicted: wp.array[wp.vec3],
    dt: float,
    result: wp.array[wp.vec3],
):
    i = wp.tid()
    result[i] = _implicit_gyro_velocity(inertia[i], effective[i], old[i], predicted[i], dt)


@wp.kernel
def _free_rotation(
    inertia: wp.mat33,
    initial: wp.vec3,
    dt: float,
    steps: int,
    result: wp.array[wp.vec3],
    energy: wp.array[float],
):
    omega = initial
    orientation = wp.quat_identity()
    peak = float(0.0)
    for _ in range(steps):
        rotation = wp.quat_to_matrix(orientation)
        world_inertia = rotation * inertia * wp.transpose(rotation)
        predicted = omega - dt * (wp.inverse(world_inertia * 1.0e7) * wp.cross(omega, (world_inertia * 1.0e7) * omega))
        omega = _implicit_gyro_velocity(world_inertia, world_inertia, omega, predicted, dt)
        orientation = wp.normalize(orientation + dt * 0.5 * (wp.quat(omega, 0.0) * orientation))
        body_omega = wp.quat_rotate_inv(orientation, omega)
        current = 0.5 * wp.dot(body_omega, inertia * body_omega)
        peak = wp.max(peak, current)
    result[0] = wp.quat_rotate_inv(orientation, omega)
    energy[0] = peak
    energy[1] = 0.5 * wp.dot(result[0], inertia * result[0])


def _reference(inertia, omega, duration):
    """Integrate body-frame Euler equations with a small RK4 reference step."""
    h = 1.0 / 24000.0
    inverse = np.linalg.inv(inertia)

    def rhs(w):
        return -inverse @ np.cross(w, inertia @ w)

    w = np.asarray(omega, dtype=np.float64).copy()
    for _ in range(round(duration / h)):
        a = rhs(w)
        b = rhs(w + 0.5 * h * a)
        c = rhs(w + 0.5 * h * b)
        d = rhs(w + h * c)
        w += h * (a + 2.0 * b + 2.0 * c + d) / 6.0
    return w


class TestFreeBodyGyro(unittest.TestCase):
    def test_solver_free_coin_energy(self):
        """Exercise the opt-in predictor through a contact-free full solver step."""
        inertia = np.diag([4.257809e-7, 4.257809e-7, 8.375535e-7])
        initial = np.array([20.0, 40.0, 60.0])
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=0.010720685, com=wp.vec3(), inertia=wp.mat33(inertia))
        joint = builder.add_joint_free(parent=-1, child=body)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        state, following = model.state(), model.state()
        velocity = state.joint_qd.numpy()
        velocity[3:6] = initial
        state.joint_qd.assign(velocity)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        with patch.dict(os.environ, {"FEATHER_PGS_FREE_BODY_IMPLICIT_GYRO": "1"}):
            solver = SolverFeatherPGS(
                model,
                angular_damping=0.0,
                pgs_iterations=24,
                use_parallel_streams=False,
            )
        self.assertIs(solver._free_body_gyro_kernel, correct_free_body_gyroscopic_predictor)
        control = model.control()
        for _ in range(600):
            solver.step(state, following, control, None, 1.0 / 600.0)
            state, following = following, state
        orientation = wp.quat(state.body_q.numpy()[body, 3:7])
        omega = wp.vec3(state.joint_qd.numpy()[3:6])
        body_omega = np.asarray(wp.quat_rotate_inv(orientation, omega))
        energy_ratio = (body_omega @ inertia @ body_omega) / (initial @ inertia @ initial)
        momentum_ratio = np.linalg.norm(inertia @ body_omega) / np.linalg.norm(inertia @ initial)
        self.assertTrue(np.isfinite(body_omega).all())
        self.assertAlmostEqual(energy_ratio, 1.0, delta=0.001)
        self.assertAlmostEqual(momentum_ratio, 1.0, delta=0.001)

    def test_zero_and_principal_axis(self):
        """Preserve zero spin, principal-axis spin and spherical forced motion."""
        inertia = np.array([np.diag([2.0, 3.0, 4.0]), np.diag([2.0, 3.0, 4.0]), np.eye(3)]) * 1.0e-8
        old = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0], [3.0, 5.0, 7.0]], dtype=np.float32)
        predicted = old.copy()
        predicted[2] += [1.0, -2.0, 3.0]
        output = wp.empty(3, dtype=wp.vec3, device="cpu")
        wp.launch(
            _correct_samples,
            3,
            inputs=[
                wp.array(inertia, dtype=wp.mat33, device="cpu"),
                wp.array(inertia, dtype=wp.mat33, device="cpu"),
                wp.array(old, dtype=wp.vec3, device="cpu"),
                wp.array(predicted, dtype=wp.vec3, device="cpu"),
                1.0 / 600.0,
            ],
            outputs=[output],
            device="cpu",
        )
        np.testing.assert_allclose(output.numpy(), predicted, atol=1.0e-6, rtol=0.0)

    def test_forcing_effective_mass_and_scale(self):
        """Retain external forcing and effective inertia at either inertia scale."""
        physical = np.diag([0.4, 0.5, 0.8])
        effective = physical + np.diag([0.1, 0.2, 0.3])
        old = np.array([20.0, 40.0, 60.0])
        dt = 1.0 / 600.0
        predicted = old + dt * np.linalg.solve(effective, np.array([3.0, -4.0, 5.0]) - np.cross(old, physical @ old))
        scales = np.array([1.0, 1.0e-7])
        output = wp.empty(2, dtype=wp.vec3, device="cpu")
        wp.launch(
            _correct_samples,
            2,
            inputs=[
                wp.array(scales[:, None, None] * physical, dtype=wp.mat33, device="cpu"),
                wp.array(scales[:, None, None] * effective, dtype=wp.mat33, device="cpu"),
                wp.array(np.tile(old, (2, 1)), dtype=wp.vec3, device="cpu"),
                wp.array(np.tile(predicted, (2, 1)), dtype=wp.vec3, device="cpu"),
                dt,
            ],
            outputs=[output],
            device="cpu",
        )
        actual = output.numpy()
        np.testing.assert_allclose(actual[0], actual[1], atol=1.0e-5, rtol=0.0)
        for w in actual:
            midpoint = 0.5 * (old + w)
            residual = effective @ (w - predicted) + dt * (
                np.cross(midpoint, physical @ midpoint) - np.cross(old, physical @ old)
            )
            self.assertLess(np.linalg.norm(residual), 5.0e-6)

    def test_coin_rotation_and_refinement(self):
        """Bound torque-free coin energy and improve angular accuracy when dt halves."""
        inertia = np.diag([4.257809e-7, 4.257809e-7, 8.375535e-7])
        initial = np.array([20.0, 40.0, 60.0])
        e0 = 0.5 * initial @ inertia @ initial
        errors = []
        reference = _reference(inertia, initial, 0.1)
        for dt in (1.0 / 600.0, 1.0 / 1200.0):
            output = wp.empty(1, dtype=wp.vec3, device="cpu")
            energy = wp.empty(2, dtype=float, device="cpu")
            wp.launch(
                _free_rotation,
                1,
                inputs=[wp.mat33(inertia), wp.vec3(initial), dt, round(0.1 / dt)],
                outputs=[output, energy],
                device="cpu",
            )
            np.testing.assert_allclose(energy.numpy() / e0, 1.0, atol=0.001, rtol=0.0)
            errors.append(np.linalg.norm(output.numpy()[0] - reference))
        self.assertLess(errors[1], 0.7 * errors[0])
        wp.launch(
            _free_rotation,
            1,
            inputs=[wp.mat33(inertia), wp.vec3(initial), 1.0 / 600.0, 1800],
            outputs=[output, energy],
            device="cpu",
        )
        self.assertTrue(np.isfinite(output.numpy()).all())
        np.testing.assert_allclose(energy.numpy() / e0, 1.0, atol=0.001, rtol=0.0)
        momentum_ratio = np.linalg.norm(inertia @ output.numpy()[0]) / np.linalg.norm(inertia @ initial)
        self.assertAlmostEqual(momentum_ratio, 1.0, delta=0.001)

    def test_rotated_parent_mapping_and_eligibility(self):
        """Rotate physical inertia into joint axes and leave ineligible DOFs unchanged."""
        q_parent = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7)
        q_relative = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.4)
        body_rotation = q_parent * q_relative
        rotation = np.asarray(wp.quat_to_matrix(q_relative)).reshape(3, 3)
        physical_body = np.diag([4.0, 5.0, 8.0]) * 1.0e-7
        physical_joint = rotation @ physical_body @ rotation.T
        effective = physical_joint + np.eye(3) * 2.0e-7
        # The leaf-first factor (P H P^T = L L^T, P the coordinates reversed): the angular block first.
        factors = np.tile(np.eye(6), (3, 1, 1))
        factors[:, :3, :3] = np.linalg.cholesky(effective[::-1, ::-1])
        # The Cholesky contract defines only its lower triangle.
        factors[:, np.triu_indices(6, k=1)[0], np.triu_indices(6, k=1)[1]] = 123.0
        old = np.tile([0.4, 0.5, 0.6, 20.0, 40.0, 60.0], 3).astype(np.float32)
        predicted = old + np.tile([0.01, -0.02, 0.03, 0.2, -0.3, 0.4], 3).astype(np.float32)
        result = wp.array(predicted, dtype=float, device="cpu")

        def ints(values):
            return wp.array(values, dtype=int, device="cpu")

        wp.launch(
            correct_free_body_gyroscopic_predictor,
            3,
            inputs=[
                ints([2, 0, 1]),
                ints([1, 1, 0]),
                ints([0, 1, 2, 3]),
                ints([0, 1, 2]),
                ints([0, 6, 12, 18]),
                ints([0, 1, 0]),
                wp.array([wp.transform(wp.vec3(3.0, 4.0, 5.0), body_rotation)] * 3, dtype=wp.transform, device="cpu"),
                wp.array([wp.transform(wp.vec3(1.0, 2.0, 3.0), q_parent)] * 3, dtype=wp.transform, device="cpu"),
                wp.array([physical_body] * 3, dtype=wp.mat33, device="cpu"),
                wp.array(factors, dtype=float, device="cpu"),
                wp.array(old, dtype=float, device="cpu"),
                1.0 / 600.0,
            ],
            outputs=[result],
            device="cpu",
        )
        actual = result.numpy()
        np.testing.assert_array_equal(actual[:3], predicted[:3])
        np.testing.assert_array_equal(actual[6:], predicted[6:])
        w = actual[3:6]
        midpoint = 0.5 * (old[3:6] + w)
        residual = (
            effective @ (w - predicted[3:6])
            + (np.cross(midpoint, physical_joint @ midpoint) - np.cross(old[3:6], physical_joint @ old[3:6])) / 600.0
        )
        self.assertLess(np.linalg.norm(residual) / np.max(np.abs(effective)), 1.0e-5)


if __name__ == "__main__":
    unittest.main()

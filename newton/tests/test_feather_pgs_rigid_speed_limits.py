# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bound the speeds of free rigid bodies in matrix-free mode, not their velocity components.

PhysX's ``maxLinearVelocity`` and ``maxAngularVelocity`` bound the magnitudes of a rigid body's linear and
angular velocity. Matrix-free mode solves each finite limit as a speed row at the end of every sweep: a body
launched past its limits ends the step on them, along its own direction, under either velocity-limit law,
also along a diagonal whose every component is within the limit.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 60.0
LINEAR_LIMIT = 2.0
"""Linear speed limit of every box [m/s]."""
ANGULAR_LIMIT = 3.0
"""Angular speed limit of every box [rad/s]."""
INERTIA = 0.1
"""Spherical inertia of every box [kg m^2]: free flight keeps its angular velocity."""
RTOL = 1.0e-5
"""Velocities agree to this fraction of their limit (float32 rounding)."""

LAUNCHES = (
    ((1.5, 1.5, 0.5), (2.2, -2.2, 0.5)),
    ((-3.0, 0.0, 0.0), (0.0, 0.0, 4.0)),
    ((1.0, -1.0, 0.5), (1.0, 1.0, 1.0)),
)
"""(linear [m/s], angular [rad/s]) launch velocity of the box of each world: past both limits along diagonals
whose every component is within them, past both along one axis, and within both."""


def _clamped(velocity: np.ndarray) -> np.ndarray:
    """Scale each world's linear and angular velocity onto its limit where it exceeds it: [world, 6]."""
    clamped = np.array(velocity, dtype=np.float64)
    for part, limit in ((slice(0, 3), LINEAR_LIMIT), (slice(3, 6), ANGULAR_LIMIT)):
        speed = np.linalg.norm(clamped[:, part], axis=1, keepdims=True)
        clamped[:, part] *= np.minimum(1.0, limit / np.maximum(speed, 1.0e-30))
    return clamped


def _step_boxes(device, **options):
    """Launch one free box per world without gravity, take one matrix-free step and return the solver and velocity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverFeatherPGS.register_custom_attributes(builder)
    bodies = []
    for _ in LAUNCHES:
        builder.begin_world()
        body = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * INERTIA),
            lock_inertia=True,
            custom_attributes={
                "rigid_body_max_linear_velocity": LINEAR_LIMIT,
                "rigid_body_max_angular_velocity": ANGULAR_LIMIT,
            },
        )
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        bodies.append(body)
        builder.end_world()
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, angular_damping=0.0, **options)
    state_0, state_1 = model.state(), model.state()
    starts = model.joint_qd_start.numpy()
    joints = [int(np.flatnonzero(model.joint_child.numpy() == body)[0]) for body in bodies]
    dofs = np.array([np.arange(starts[joint], starts[joint] + 6) for joint in joints])
    qd = state_0.joint_qd.numpy()
    qd[dofs] = np.array([np.concatenate(launch) for launch in LAUNCHES])
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    solver.step(state_0, state_1, model.control(), None, DT)
    return solver, state_1.joint_qd.numpy()[dofs]


@unittest.skipUnless(wp.is_cuda_available(), "Matrix-free FeatherPGS requires CUDA")
class TestFeatherPGSRigidSpeedLimits(unittest.TestCase):
    def test_step_ends_on_the_speed_limits(self):
        """End a step past the limits on them, along the launch direction, under either activation gate."""
        expected = _clamped(np.array([np.concatenate(launch) for launch in LAUNCHES]))
        limits = np.array([LINEAR_LIMIT] * 3 + [ANGULAR_LIMIT] * 3)
        for fraction in (0.0, 0.7):
            with self.subTest(fraction=fraction):
                solver, velocity = _step_boxes("cuda:0", velocity_limit_activation_fraction=fraction)
                np.testing.assert_allclose(velocity / limits, expected / limits, rtol=0.0, atol=RTOL)
                routes = solver.get_path_selection()["routes"]
                self.assertTrue(routes["velocity_limits"]["free_body_rows"])

    def test_speed_rows_bound_whole_velocities(self):
        """Allocate one speed row per finite limit of each box past its limits: linear (axis 0) and angular (axis 3).

        The box within its limits may resolve without rows, so only the first two worlds are read.
        """
        solver, _ = _step_boxes("cuda:0")
        count = solver.placement.mf_constraint_count.numpy()[:2]
        np.testing.assert_array_equal(count, 2)
        types = solver.placement.mf_row_type.numpy()[:2, :2]
        np.testing.assert_array_equal(types, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
        # A speed row keeps its axis unwhitened.
        jacobian = solver.placement.mf_Z_a.numpy()[:2, :2]
        np.testing.assert_array_equal(np.sort(np.argmax(np.abs(jacobian), axis=2), axis=1), [[0, 3]] * 2)
        np.testing.assert_array_equal(np.abs(jacobian).sum(axis=2), 1.0)


if __name__ == "__main__":
    unittest.main()

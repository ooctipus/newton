# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold the D6 joint's law to the free joint's.

A D6 joint is the chain of its axes that its transform composes: the linear axes slide along the parent anchor's,
then each angular axis, carried through the rotations before it, turns about the joint origin the linear
coordinates moved. A body in free flight on a D6 (no gravity, no contacts) therefore keeps its line, speed and spin
as the same body on a free joint does, and a tumbling one converges to the free body as the step shrinks (its angle
coordinates integrate at first order). Both evaluators run the law: a D6 root takes the ordinary passes, a D6 below a
fixed root the held tree.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

_AXES = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
_INERTIA = (0.001, 0.001, 0.001)
_HEIGHT = 1.0


def _devices():
    return ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",)


def _d6_body(builder, parent, angular, inertia):
    """A body of unit mass on a D6 joint (linear x, y and z, then the ``angular`` axes) to ``parent``, or to the
    world."""
    pose = wp.transform(wp.vec3(0.0, 0.0, _HEIGHT))
    body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.diag(inertia).astype(np.float32)))
    axis = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        parent,
        body,
        linear_axes=[axis(axis=a) for a in _AXES],
        angular_axes=[axis(axis=_AXES[i]) for i in angular],
        parent_xform=pose if parent < 0 else wp.transform_identity(),
    )
    return body, joint


def _floating_model(device, kind, angular=(2,), inertia=_INERTIA):
    """One body in free flight: on a ``"free"`` joint, on a D6 ``"root"``, or on a D6 ``"child"`` of a fixed body."""
    builder = newton.ModelBuilder(gravity=0.0)
    if kind == "free":
        builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, _HEIGHT)),
            mass=1.0,
            inertia=wp.mat33(np.diag(inertia).astype(np.float32)),
        )
        return builder.finalize(device=device)
    joints = []
    parent = -1
    if kind == "child":
        parent = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, _HEIGHT)), mass=1.0)
        joints.append(builder.add_joint_fixed(-1, parent, parent_xform=wp.transform(wp.vec3(0.0, 0.0, _HEIGHT))))
    _, joint = _d6_body(builder, parent, angular, inertia)
    builder.add_articulation([*joints, joint])
    return builder.finalize(device=device)


def _fly(model, velocity, dt, seconds):
    """Step ``model`` from the moving body's coordinates ``velocity`` (linear, then angular); return its final twist
    (COM velocity, angular velocity), position and kinetic energy before and after, and the evaluator that ran."""
    solver = SolverFeatherPGS(model, pgs_iterations=8, angular_damping=0.0)
    state, other, control = model.state(), model.state(), model.control()
    qd = np.zeros(model.joint_dof_count, np.float32)
    count = len(velocity)
    qd[model.joint_dof_count - count :] = velocity
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body = model.body_count - 1
    inertia = model.body_inertia.numpy()[body].astype(float)

    def energy(s):
        v = s.body_qd.numpy()[body].astype(float)
        q = s.body_q.numpy()[body].astype(float)
        R = np.array(wp.quat_to_matrix(wp.quat(*q[3:])), dtype=float).reshape(3, 3)
        return 0.5 * v[:3] @ v[:3] + 0.5 * v[3:] @ (R @ inertia @ R.T) @ v[3:]

    before = energy(state)
    for _ in range(round(seconds / dt)):
        solver.step(state, other, control, None, dt)
        state, other = other, state
    evaluator = "held" if solver.factor.held is not None else "ordinary"
    return (
        state.body_qd.numpy()[body].astype(float),
        state.body_q.numpy()[body].astype(float),
        before,
        energy(state),
        evaluator,
    )


class TestFeatherPGSD6Law(unittest.TestCase):
    def test_floating_d6_keeps_its_line_and_spin(self):
        """Slide at 1 m/s along x while spinning at 20 rad/s about z on a D6, as on a free joint, for 0.3 s.

        The body keeps its speed, spin, line and energy on the ordinary passes (a D6 root) and on the held tree (a D6
        below a fixed root), with the yaw axis alone and with all three angular axes.
        """
        for device in _devices():
            for kind, evaluator in (("root", "ordinary"), ("child", "held")):
                for angular in ((2,), (0, 1, 2)):
                    with self.subTest(device=device, kind=kind, angular=angular):
                        model = _floating_model(device, kind, angular)
                        velocity = [1.0, 0.0, 0.0] + [0.0] * (len(angular) - 1) + [20.0]
                        twist, pose, before, after, ran = _fly(model, velocity, 1.0 / 240.0, 0.3)
                        self.assertEqual(ran, evaluator)
                        np.testing.assert_allclose(twist, (1.0, 0.0, 0.0, 0.0, 0.0, 20.0), atol=1.0e-4)
                        np.testing.assert_allclose(pose[1:3], (0.0, _HEIGHT), atol=1.0e-5)
                        self.assertAlmostEqual(pose[0], 0.3, delta=1.0e-4)
                        self.assertAlmostEqual(after, before, delta=1.0e-5 * before)

    def test_tumbling_d6_converges_to_the_free_body(self):
        """Tumble an asymmetric body at (5, 7, 20) rad/s on a three-angle D6 root and on a free joint for 0.25 s.

        The D6's angle coordinates integrate at first order, so its angular velocity meets the free body's as the step
        shrinks: a quarter of the step cuts the difference by more than 3, to below 0.35 rad/s at 1/1920 s. (Both
        meet Euler's equations at first order.) Before the D6 law composed its chain, this tumble left the finite
        numbers within 0.5 s.
        """
        inertia = (0.0010, 0.0008, 0.0005)
        velocity = [1.0, 0.5, 0.0, 5.0, 7.0, 20.0]
        errors = []
        for hz in (480, 1920):
            free, _, _, _, _ = _fly(_floating_model("cpu", "free", inertia=inertia), velocity, 1.0 / hz, 0.25)
            d6, _, _, _, _ = _fly(_floating_model("cpu", "root", (0, 1, 2), inertia), velocity, 1.0 / hz, 0.25)
            self.assertTrue(np.all(np.isfinite(d6)))
            errors.append(float(np.linalg.norm(d6[3:] - free[3:])))
        self.assertGreater(errors[0], 3.0 * errors[1], errors)
        self.assertLess(errors[1], 0.35, errors)

    def test_sliding_yaw_d6_comes_to_rest(self):
        """Slide a sphere that only yaws (a D6 with linear axes and yaw) at 1 m/s on ground of friction 0.5, spinning
        at 20 rad/s: it cannot roll, so friction stops it within 0.3 s as it stops a sphere too stiff to roll."""
        for device in _devices():
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
                pose = wp.transform(wp.vec3(0.0, 0.0, 0.0505))
                body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.001))
                axis = newton.ModelBuilder.JointDofConfig
                joint = builder.add_joint_d6(
                    -1,
                    body,
                    linear_axes=[axis(axis=a) for a in _AXES],
                    angular_axes=[axis(axis=newton.Axis.Z)],
                    parent_xform=pose,
                )
                builder.add_shape_sphere(body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0))
                builder.add_articulation([joint])
                model = builder.finalize(device=device)
                solver = SolverFeatherPGS(model, pgs_iterations=32, friction_anchor_beta=0.0, angular_damping=0.0)
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                state, other, control = model.state(), model.state(), model.control()
                state.joint_qd.assign(np.array((1.0, 0.0, 0.0, 20.0), np.float32))
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                for _ in range(72):
                    state.clear_forces()
                    pipeline.collide(state, contacts)
                    solver.step(state, other, control, contacts, 1.0 / 240.0)
                    state, other = other, state
                twist = state.body_qd.numpy()[0]
                self.assertLess(float(np.hypot(twist[0], twist[1])), 1.0e-3)
                self.assertAlmostEqual(float(twist[5]), 20.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()

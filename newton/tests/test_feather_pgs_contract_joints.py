# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The joint laws of SolverFeatherPGS, on its public API.

D6 joints against the free joint, passive joint springs, and joint velocity limits: the pre-solve prescale and the
limit rows' last word over contacts. A contract module: :mod:`newton.tests.test_feather_pgs_contract_lint` keeps it
on the public API.
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

# ---------------------------------------------------------------------------------------------------------------------
# d6 law
#
# Hold the D6 joint's law to the free joint's.
#
# A D6 joint is the chain of its axes that its transform composes: the linear axes slide along the parent anchor's,
# then each angular axis, carried through the rotations before it, turns about the joint origin the linear
# coordinates moved. A body in free flight on a D6 (no gravity, no contacts) therefore keeps its line, speed and spin
# as the same body on a free joint does, and a tumbling one converges to the free body as the step shrinks (its angle
# coordinates integrate at first order).
# ---------------------------------------------------------------------------------------------------------------------

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
    (COM velocity, angular velocity), position and kinetic energy before and after."""
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
    return (
        state.body_qd.numpy()[body].astype(float),
        state.body_q.numpy()[body].astype(float),
        before,
        energy(state),
    )


class TestFeatherPGSD6Law(unittest.TestCase):
    def test_floating_d6_keeps_its_line_and_spin(self):
        """Slide at 1 m/s along x while spinning at 20 rad/s about z on a D6, as on a free joint, for 0.3 s.

        The body keeps its speed, spin, line and energy on a D6 root and on a D6 below a fixed root, with the yaw axis
        alone and with all three angular axes.
        """
        for device in _devices():
            for kind, angular in itertools.product(("root", "child"), ((2,), (0, 1, 2))):
                with self.subTest(device=device, kind=kind, angular=angular):
                    model = _floating_model(device, kind, angular)
                    velocity = [1.0, 0.0, 0.0] + [0.0] * (len(angular) - 1) + [20.0]
                    twist, pose, before, after = _fly(model, velocity, 1.0 / 240.0, 0.3)
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
            free, _, _, _ = _fly(_floating_model("cpu", "free", inertia=inertia), velocity, 1.0 / hz, 0.25)
            d6, _, _, _ = _fly(_floating_model("cpu", "root", (0, 1, 2), inertia), velocity, 1.0 / hz, 0.25)
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


# ---------------------------------------------------------------------------------------------------------------------
# springs
#
# Tests for passive joint springs and passive damping in SolverFeatherPGS.
# ---------------------------------------------------------------------------------------------------------------------


def _build_hinge(spring_k: float, spring_ref: float, damping: float, limit_upper: float = 10.0):
    """Build a fixed-base link on a revolute Z joint (gravity exerts no torque about the axis).

    The joint has no drive; any motion toward ``spring_ref`` comes from the passive spring
    alone, and any velocity decay beyond numerics comes from the passive damping alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    link = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
    j = b.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        limit_lower=-10.0,
        limit_upper=limit_upper,
    )
    b.add_articulation([j], label="hinge")
    dof = b.joint_qd_start[j]
    b.joint_spring_stiffness[dof] = spring_k
    b.joint_spring_ref[dof] = spring_ref
    b.joint_damping[dof] = damping
    return b


def _run(builder, steps: int = 1200, qd0: float = 0.0, **solver_kwargs):
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    if qd0 != 0.0:
        qd = state_0.joint_qd.numpy()
        qd[0] = qd0
        state_0.joint_qd.assign(qd)
    control = model.control()
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return state_0


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSSprings(unittest.TestCase):
    def test_spring_converges_to_reference(self):
        """Verify an undriven joint settles at the passive spring's reference position."""
        state = _run(_build_hinge(spring_k=0.5, spring_ref=0.5, damping=0.05))
        q = state.joint_q.numpy()
        self.assertAlmostEqual(float(q[0]), 0.5, delta=0.02)

    def test_springref_preloads_against_limit(self):
        """Verify a spring reference beyond the joint range preloads the joint at its limit.

        This is the Robotiq spring_link pattern: ``springref`` is unreachable, so the
        spring presses the joint against its upper stop with constant preload torque.
        """
        state = _run(
            _build_hinge(spring_k=0.5, spring_ref=2.62, damping=0.05, limit_upper=0.3),
            enable_joint_limits=True,
        )
        q = state.joint_q.numpy()
        self.assertAlmostEqual(float(q[0]), 0.3, delta=0.03)

    def test_passive_damping_decays_velocity(self):
        """Verify passive joint damping decays velocity that persists when damping is disabled."""
        damped = _run(_build_hinge(spring_k=0.0, spring_ref=0.0, damping=0.02), steps=480, qd0=5.0)
        undamped = _run(_build_hinge(spring_k=0.0, spring_ref=0.0, damping=0.0), steps=480, qd0=5.0)
        qd_damped = abs(float(damped.joint_qd.numpy()[0]))
        qd_undamped = abs(float(undamped.joint_qd.numpy()[0]))
        self.assertGreater(qd_undamped, 4.0, "undamped joint should keep coasting")
        self.assertLess(qd_damped, 0.2 * qd_undamped)

    def test_grouped_passive_damping_uses_unscaled_velocity(self):
        """Keep passive damping on the input velocity while motion uses the prescale."""
        for damping, velocity in ((0.2, 10.0), (0.0, 10.0), (0.2, 1.0)):
            with self.subTest(damping=damping, velocity=velocity):
                builder = _build_hinge(spring_k=0.0, spring_ref=0.0, damping=damping)
                builder.joint_velocity_limit[0] = 2.0
                model = builder.finalize()
                control = model.control()
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_iterations=8,
                    dense_max_constraints=32,
                    enable_joint_velocity_limits=True,
                )
                state_in, state_out = model.state(), model.state()
                state_in.joint_qd.fill_(velocity)
                solver.step(state_in, state_out, control, None, 0.001)
                solver.check_constraint_capacity()
                np.testing.assert_allclose(state_in.joint_qd.numpy(), [velocity], atol=1e-6)
                np.testing.assert_allclose(solver.qd_work.numpy(), [min(velocity, 2.0)], atol=1e-6)
                # Z-axis gravity and centripetal forces exert no hinge torque: the predictor adds the damping's
                # acceleration, read on the input velocity with its K in the mass, to the prescaled velocity.
                inertia = model.body_inertia.numpy()[0, 2, 2] + model.body_mass.numpy()[0] * 0.2**2
                inertia += model.joint_armature.numpy()[0] + 0.001 * damping
                np.testing.assert_allclose(
                    solver.v_hat.numpy(), [min(velocity, 2.0) - 0.001 * damping * velocity / inertia], atol=2e-5
                )


# ---------------------------------------------------------------------------------------------------------------------
# velocity limit ordering
#
# Ordering of the joint velocity-limit rows vs. contact rows.
#
# A joint velocity limit must get the last word of every solver iteration:
# PhysX solves joint max-velocity after link contacts (see
# PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST and the
# eAFTER_STATIC_CONSTRAINTS path in DyFeatherstoneArticulation.cpp), and the
# dedicated velocity-limit rows run in the final pass of each FPGS iteration
# for the same reason: a contact impulse landing earlier in the iteration
# cannot re-accelerate the limited DOF past its limit.
#
# The scene makes that failure concrete: a stiff PD-driven horizontal arm with
# a low joint velocity limit, and a heavy free box slamming into the arm's tip.
# The impact pushes the arm's joint speed far past the limit; the limit must
# still hold at the end of every step.
# ---------------------------------------------------------------------------------------------------------------------

QDOT_MAX = 0.5
# GS-convergence slack on the limit; the final clamp leaves the DOF at the
# limit, later float noise stays well under 5%.
LIMIT_TOL = 1.05


def _build_arm_and_box_model(device) -> newton.Model:
    """A driven revolute arm plus a heavy free box overlapping its tip.

    Gravity is zero and the box carries a large downward velocity, so the
    contact impulse into the arm tip is the only thing fighting the velocity
    limit — fully deterministic.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    # Arm spanning x in [0, 0.8] at z = 0.5, hinged at its left end about y.
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    j_arm = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_arm])

    # Heavy box, bottom face slightly penetrating the arm top near the tip.
    box = builder.add_link(xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])

    return builder.finalize(device=device)


def _cartesian_pair(builder: newton.ModelBuilder) -> None:
    """Append an articulation of two prismatic joints along x and then y: a diagonal mass matrix and no bias force."""
    slider = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    carriage = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation(
        [
            builder.add_joint_prismatic(parent=-1, child=slider, axis=newton.Axis.X),
            builder.add_joint_prismatic(parent=slider, child=carriage, axis=newton.Axis.Y),
        ]
    )


class TestFeatherPGSVelocityPrescale(unittest.TestCase):
    """The pre-solve prescale: an articulation past a joint's velocity limit scales all its joint velocities by one
    ratio, ``min_i(limit_i / |qd_i|)``, as PhysX does, so its motion keeps its direction; one within its limits is
    untouched. A per-coordinate clamp would end the first pair below at ``(1, 1)``."""

    @unittest.skipUnless(wp.is_cuda_available(), "joint velocity limits require CUDA")
    def test_one_ratio_scales_the_articulation(self):
        builder = newton.ModelBuilder(gravity=0.0)
        for _ in range(2):
            _cartesian_pair(builder)
        model = builder.finalize(device="cuda:0")
        model.joint_velocity_limit.assign(np.ones(model.joint_dof_count, dtype=np.float32))
        solver = SolverFeatherPGS(model, enable_joint_velocity_limits=True, angular_damping=0.0)
        state_0, state_1 = model.state(), model.state()
        state_0.joint_qd.assign(np.array([3.0, 1.0, 0.5, -0.2], dtype=np.float32))
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        np.testing.assert_allclose(state_1.joint_qd.numpy(), [1.0, 1.0 / 3.0, 0.5, -0.2], rtol=1.0e-5, atol=1.0e-6)


class TestFeatherPGSVelocityLimitOrdering(unittest.TestCase):
    def _max_arm_speed_over_impact(self, velocity_limits: bool = True) -> float:
        model = _build_arm_and_box_model(device="cuda:0")

        n = model.joint_dof_count
        target_ke = np.zeros(n, dtype=np.float32)
        target_kd = np.zeros(n, dtype=np.float32)
        vel_limit = np.full(n, np.inf, dtype=np.float32)
        target_ke[0] = 200.0
        target_kd[0] = 5.0
        vel_limit[0] = QDOT_MAX
        model.joint_target_ke.assign(target_ke)
        model.joint_target_kd.assign(target_kd)
        model.joint_velocity_limit.assign(vel_limit)

        solver = SolverFeatherPGS(
            model,
            enable_joint_velocity_limits=velocity_limits,
            enable_contact_friction=False,
            # Deliberately under-converged (franka-class training configs run
            # 8): ordering only matters when the sweep does NOT converge — at
            # convergence the last contact visits apply ~zero delta and any
            # limit placement passes.
            pgs_iterations=8,
            dense_max_constraints=32,
            mf_max_constraints=32,
        )

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[3] = -3.0  # box linear z: slam into the arm tip
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        contacts = model.contacts()
        max_arm_speed = 0.0
        for _ in range(20):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
            max_arm_speed = max(max_arm_speed, float(abs(state_0.joint_qd.numpy()[0])))
        wp.synchronize()
        return max_arm_speed

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free FPGS requires CUDA")
    def test_impact_drives_an_unlimited_arm_past_the_limit(self):
        """Drive the arm at least twice past the limit when no velocity limit acts.

        Every other test here holds the limit through the impact, which an impact
        that never reached the limit would pass vacuously.
        """
        speed = self._max_arm_speed_over_impact(velocity_limits=False)
        self.assertGreater(speed, 2.0 * QDOT_MAX)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free FPGS requires CUDA")
    def test_dedicated_rows_hold_limit_through_impact(self):
        # Reference behavior: dedicated velocity-limit rows run in the final
        # phase of every iteration and hold the limit through the impact.
        speed = self._max_arm_speed_over_impact()
        self.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)

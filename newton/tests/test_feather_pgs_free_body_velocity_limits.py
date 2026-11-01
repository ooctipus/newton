# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold free rigid bodies to the speed law on the CPU.

The matrix-free sweep visits a world's free-body rows after its dense rows, one thread per world on the CPU. Each
finite speed limit of a body owns a speed row at the tail of its world's rows: it bounds the magnitude of the body's
linear or angular velocity, as PhysX's ``maxLinearVelocity`` and ``maxAngularVelocity`` do, with ``J = e_0`` or
``e_3`` naming which and ``rhs = limit``. A row keeps a non-negative total, the velocity-limit law of every sweep: it
pushes while its speed exceeds the limit, along the velocity, by the whole correction whatever ``pgs_omega`` relaxes,
and releases its total while the speed is within the limit: a body past its limit keeps its direction and ends on
the limit. Per-axis rows would let a diagonal velocity reach ``sqrt(3)`` times the limit; a sweep that relaxed the
rows as bilateral rows would accelerate a body that moves below its limit.

The sweep leaves the predicted velocity alone, so the contact rows act on it first, and bounds the free rigid bodies
with the speed rows alone. The rows act wherever the predicted velocity or another row of the sweep sets a body past a
limit, as a force pushes a body along the ground or a contact row pushes it out of the ground.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes

DT = 1.0 / 60.0
# Speed limits of every free box [m/s, rad/s].
LINEAR_LIMIT = 2.0
ANGULAR_LIMIT = 3.0
LIMITS = np.array([LINEAR_LIMIT] * 3 + [ANGULAR_LIMIT] * 3)
# Registered speed attributes that limit nothing [m/s, rad/s].
INFINITE_LIMITS = (float("inf"), float("inf"))
# A 1 kg body with a spherical inertia: its axes decouple and free flight keeps its angular velocity.
MASS = 1.0
INERTIA = 0.1
# Velocities agree to this fraction of their axis's limit: the default pgs_cfm leaves 1e-6 of each correction
# on a 1 kg body (1e-7 on its 0.1 kg m^2 axes), and free flight keeps a velocity to float32 rounding.
RTOL = 1.0e-5

# Initial (linear [m/s], angular [rad/s]) velocity of one box per world, below both limits in magnitude.
BELOW = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((1.0, -1.5, 0.5), (0.5, -1.0, 2.0)),
    ((-1.5, 0.3, 0.9), (-2.4, 0.4, 1.1)),
)
# (initial linear, initial angular, linear velocity change [m/s], angular velocity change [rad/s]) of one box
# per world that ends the predictor past its limits: launched past them, or pushed past them by a wrench.
PAST = (
    ((-3.0, 1.0, 0.0), (0.0, -4.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((1.5, 0.0, 0.0), (0.0, 0.0, 1.0), (-5.0, 0.5, 0.0), (0.0, 0.0, -6.0)),
)

# Spheres pressed into the ground [m], one per world. A contact row pushes each out at its Baumgarte speed
# (pgs_beta * depth / DT: 0.06, 0.12 and 0.24 m/s), below or past the linear limit CONTACT_LIMIT [m/s].
RADIUS = 0.1
DEPTHS = (0.005, 0.01, 0.02)
CONTACT_LIMIT = 0.1
CONTACT_LIMITS = np.array([CONTACT_LIMIT] * 3 + [ANGULAR_LIMIT] * 3)


def _begin_model(limits, *, gravity=(0.0, 0.0, 0.0)):
    """Start a model, without gravity by default, that registers the speed attributes unless ``limits`` is None."""
    builder = newton.ModelBuilder(gravity=gravity)
    if limits is not None:
        SolverFeatherPGS.register_custom_attributes(builder)
    return builder


def _speed_attributes(limits):
    """Body attributes for (linear [m/s], angular [rad/s]) speed ``limits``; None sets none."""
    if limits is None:
        return None
    return {"rigid_body_max_linear_velocity": limits[0], "rigid_body_max_angular_velocity": limits[1]}


def _add_body(builder, xform, limits):
    """Add a 1 kg free body with a spherical inertia that its shapes do not change."""
    return builder.add_body(
        xform=xform,
        mass=MASS,
        inertia=wp.mat33(np.eye(3) * INERTIA),
        lock_inertia=True,
        custom_attributes=_speed_attributes(limits),
    )


def _add_arm(builder):
    """Add a one-joint arm to the current world, so the world's sweep visits dense rows before its free-body rows."""
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_articulation([builder.add_joint_revolute(-1, link, axis=newton.Axis.Y)])


def _run(builder, device, bodies, *, velocities=None, wrenches=None, **options):
    """Finalize ``builder`` on ``device`` with a solver and a collision pipeline.

    Args:
        builder: The model.
        device: Warp device of the model and solver.
        bodies: The free bodies whose velocities :func:`_step` returns.
        velocities: Initial [body, 6] (linear [m/s], angular [rad/s]) velocity of each of ``bodies``; zero if
            None.
        wrenches: Velocity change [m/s, rad/s] that a constant body wrench adds over one step, per body.
        options: Further :class:`~newton.solvers.SolverFeatherPGS` options.
    """
    model = builder.finalize(device=device)
    # Without angular damping, free flight keeps the angular velocity exactly.
    solver = SolverFeatherPGS(model, angular_damping=0.0, **options)
    state_0, state_1 = model.state(), model.state()
    starts = model.joint_qd_start.numpy()
    joints = [int(np.flatnonzero(model.joint_child.numpy() == body)[0]) for body in bodies]
    dofs = np.array([np.arange(starts[joint], starts[joint] + 6) for joint in joints])
    if velocities is not None:
        qd = state_0.joint_qd.numpy()
        qd[dofs] = velocities
        state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    body_f = None
    if wrenches is not None:
        # A world-frame wrench at the center of mass that changes the body velocity by ``wrenches`` over one step.
        scale = np.array([MASS] * 3 + [INERTIA] * 3) / DT
        body_f = np.zeros((model.body_count, 6), dtype=np.float32)
        body_f[bodies] = np.array(wrenches) * scale
    pipeline = newton.CollisionPipeline(model)
    return SimpleNamespace(
        model=model,
        solver=solver,
        states=(state_0, state_1),
        control=model.control(),
        pipeline=pipeline,
        contacts=pipeline.contacts(),
        bodies=bodies,
        dofs=dofs,
        body_f=body_f,
    )


def _free_boxes(
    device, worlds, *, articulated=False, limits=(LINEAR_LIMIT, ANGULAR_LIMIT), fraction=0.0, wrenches=None
):
    """One free box per world, launched at ``worlds`` (linear, angular) velocities.

    Args:
        device: Warp device of the model and solver.
        worlds: Initial (linear [m/s], angular [rad/s]) velocity of each world's box.
        articulated: Add a one-joint arm to every world, so its sweep visits dense rows before the free-body
            rows.
        limits: (linear [m/s], angular [rad/s]) speed limits of every box; None leaves the speed attributes
            unregistered.
        fraction: The solver's ``velocity_limit_activation_fraction``.
        wrenches: Velocity change [m/s, rad/s] that a constant body wrench adds over one step, per world.
    """
    builder = _begin_model(limits)
    boxes = []
    for world in range(len(worlds)):
        builder.begin_world()
        box = _add_body(builder, wp.transform(wp.vec3(0.0, 0.0, 1.0 + world), wp.quat_identity()), limits)
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
        boxes.append(box)
        if articulated:
            _add_arm(builder)
        builder.end_world()
    return _run(
        builder,
        device,
        boxes,
        velocities=np.array([np.concatenate(world) for world in worlds]),
        wrenches=wrenches,
        velocity_limit_activation_fraction=fraction,
    )


def _pressed_spheres(device, *, articulated=False, **options):
    """One sphere per world at rest, pressed ``DEPTHS`` into a ground plane, with a ``CONTACT_LIMIT`` speed limit.

    ``options`` are further :class:`~newton.solvers.SolverFeatherPGS` options.
    """
    limits = (CONTACT_LIMIT, ANGULAR_LIMIT)
    builder = _begin_model(limits)
    spheres = []
    for world, depth in enumerate(DEPTHS):
        builder.begin_world()
        sphere = _add_body(builder, wp.transform(wp.vec3(0.5 * world, 0.0, RADIUS - depth), wp.quat_identity()), limits)
        builder.add_shape_sphere(sphere, radius=RADIUS)
        spheres.append(sphere)
        if articulated:
            _add_arm(builder)
        builder.end_world()
    builder.add_ground_plane()
    return _run(builder, device, spheres, **options)


def _landing_boxes(device, *, articulated=False, worlds=8):
    """Three tilted boxes per world dropped onto a ground plane, with registered but infinite speed limits."""
    builder = _begin_model(INFINITE_LIMITS, gravity=(0.0, 0.0, -9.81))
    boxes = []
    for _ in range(worlds):
        builder.begin_world()
        for k in range(3):
            xform = wp.transform(wp.vec3(0.3 * k, 0.05 * k, 0.12 + 0.25 * k), wp.quat_rpy(0.1 * k, 0.2, 0.05))
            box = _add_body(builder, xform, INFINITE_LIMITS)
            builder.add_shape_box(box, hx=0.1, hy=0.08, hz=0.06)
            boxes.append(box)
        if articulated:
            _add_arm(builder)
        builder.end_world()
    builder.add_ground_plane()
    return _run(builder, device, boxes)


def _step(run):
    """Advance one step and return each body's (linear, angular) velocity."""
    state_0, state_1 = run.states
    state_0.clear_forces()
    if run.body_f is not None:
        state_0.body_f.assign(run.body_f)
    run.pipeline.collide(state_0, run.contacts)
    run.solver.step(state_0, state_1, run.control, run.contacts, DT)
    run.states = (state_1, state_0)
    return state_1.joint_qd.numpy()[run.dofs]


def _free_body_rows(run):
    """Return each world's free-body row types, axes, signs, right-hand sides and impulses: [world, row].

    A speed row's axis is the first DOF of the velocity it bounds: 0 for the linear, 3 for the angular.
    """
    solver = run.solver
    count = solver.placement.mf_constraint_count.numpy()
    rows = int(count.max())
    # A speed row keeps its axis unwhitened.
    jacobian = solver.placement.mf_Z_a.numpy()[:, :rows]
    axis = np.argmax(np.abs(jacobian), axis=2)
    return SimpleNamespace(
        count=count,
        type=solver.placement.mf_row_type.numpy()[:, :rows],
        axis=axis,
        sign=np.sign(np.take_along_axis(jacobian, axis[..., None], axis=2)[..., 0]),
        rhs=solver.placement.mf_rhs.numpy()[:, :rows],
        eff_mass_inv=solver.placement.mf_eff_mass_inv.numpy()[:, :rows],
        impulses=solver.placement.mf_impulses.numpy()[:, :rows],
    )


def _velocity_limit_impulses(run):
    """Return the impulses the last sweep left in velocity-limit rows, zero in every other entry: [world, row]."""
    rows = _free_body_rows(run)
    live = np.arange(rows.type.shape[1])[None, :] < rows.count[:, None]
    return np.where(live & (rows.type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT), rows.impulses, 0.0)


def _assert_on_limit_scale(actual, desired, limits=LIMITS):
    """Compare [world, 6] velocities to ``RTOL`` of each axis's limit."""
    np.testing.assert_allclose(np.asarray(actual) / limits, np.asarray(desired) / limits, rtol=0.0, atol=RTOL)


def _speeds(velocity):
    """Linear and angular speed of each world: [world, 2]."""
    velocity = np.asarray(velocity, dtype=np.float64)
    return np.stack((np.linalg.norm(velocity[:, :3], axis=1), np.linalg.norm(velocity[:, 3:], axis=1)), axis=1)


def _clamped_magnitudes(velocity, limits=(LINEAR_LIMIT, ANGULAR_LIMIT)):
    """Scale each world's linear and angular velocity onto its limit where it exceeds it: [world, 6]."""
    clamped = np.array(velocity, dtype=np.float64)
    for part, limit in zip((slice(0, 3), slice(3, 6)), limits, strict=True):
        speed = np.linalg.norm(clamped[:, part], axis=1, keepdims=True)
        clamped[:, part] *= np.minimum(1.0, limit / np.maximum(speed, 1.0e-30))
    return clamped


class _FreeBodyVelocityLimitChecks:
    """Checks of the free-body speed rows; ``device`` names the device under test (the CPU)."""

    device = "cpu"

    def check_rows_are_velocity_limits(self, run):
        """Every box owns its two speed rows: the linear (axis 0) and the angular (axis 3) one."""
        rows = _free_body_rows(run)
        np.testing.assert_array_equal(rows.count, 2)
        np.testing.assert_array_equal(rows.type, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
        for world in range(rows.type.shape[0]):
            pairs = sorted(zip(rows.axis[world].tolist(), rows.sign[world].tolist(), strict=True))
            self.assertEqual(pairs, [(0, 1.0), (3, 1.0)])
        np.testing.assert_allclose(rows.rhs, LIMITS[rows.axis], rtol=1.0e-6)

    def test_step_keeps_speeds_below_the_limits(self):
        """Keep the velocity of a free body that moves below its limits and feels no other force.

        Every speed row finds its limit satisfied and must not move the body: a bilateral relaxation would
        drive each axis onto a limit.
        """
        for articulated in (False, True):
            for fraction in (0.0, 0.7):
                with self.subTest(articulated=articulated, fraction=fraction):
                    run = _free_boxes(self.device, BELOW, articulated=articulated, fraction=fraction)
                    start = np.array([np.concatenate(world) for world in BELOW])
                    for _ in range(3):
                        velocity = _step(run)
                        if fraction == 0.0:
                            self.check_rows_are_velocity_limits(run)
                        _assert_on_limit_scale(velocity, start)
                        np.testing.assert_allclose(_velocity_limit_impulses(run), 0.0, rtol=0.0, atol=1.0e-6)
                    if fraction == 0.7:
                        # The activation gate reserved rows only for axes within 30% of their limit.
                        self.assertGreater(int(run.solver.placement.mf_constraint_count.numpy().sum()), 0)

    def test_step_clamps_speeds_past_the_limits(self):
        """Hold a free body launched or pushed past its limits on them, along its own direction.

        The speed rows scale the predicted velocity onto the limits in the sweep's first iteration and keep that
        total.
        """
        worlds = [(linear, angular) for linear, angular, _, _ in PAST]
        wrenches = [np.concatenate((force, torque)) for _, _, force, torque in PAST]
        predicted = np.array([np.concatenate(world) for world in worlds]) + np.array(wrenches)
        expected = _clamped_magnitudes(predicted)
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                run = _free_boxes(self.device, worlds, articulated=articulated, wrenches=wrenches)
                velocity = _step(run)
                self.check_rows_are_velocity_limits(run)
                _assert_on_limit_scale(velocity, expected)
                speeds = np.stack(
                    (np.linalg.norm(velocity[:, :3], axis=1), np.linalg.norm(velocity[:, 3:], axis=1)), axis=1
                )
                np.testing.assert_allclose(speeds, [[LINEAR_LIMIT, ANGULAR_LIMIT]] * len(PAST), rtol=RTOL)
                rows = _free_body_rows(run)
                speed = np.take_along_axis(_speeds(predicted), rows.axis // 3, axis=1)
                total = np.maximum(speed - rows.rhs, 0.0) * rows.eff_mass_inv
                np.testing.assert_allclose(_velocity_limit_impulses(run), total, rtol=1.0e-4, atol=1.0e-6)

    def test_step_holds_a_pushed_box_on_the_limit(self):
        """Hold a box that a force pushes along frictionless ground on its speed limit, along the push.

        Gravity's increment to the predicted velocity is the ground contact's to remove: a speed bound applied to
        the predicted velocity, before the contact acts, would leave the box 0.25% below its limit. One box is
        pushed along an axis, one along a diagonal (the conformance gate's scene).
        """
        scene = scenes.box_push()
        model = scene.build(self.device)
        solver = SolverFeatherPGS(model, angular_damping=0.0, friction_anchor_beta=0.0)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        for step in range(scene.steps):
            state_0.clear_forces()
            scene.control(model, control, state_0, step)
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, scene.dt)
            state_0, state_1 = state_1, state_0
        velocity = state_0.body_qd.numpy()[:, :3]
        np.testing.assert_allclose(np.linalg.norm(velocity, axis=1), scene.params["speed_limit"], rtol=1.0e-4)
        np.testing.assert_allclose(velocity[1, 0], velocity[1, 1], rtol=1.0e-4)

    def test_step_holds_a_contact_push_on_the_limit(self):
        """End a step on the limit that a contact row pushed a free body past.

        A sphere pressed into the ground leaves it at the Baumgarte speed ``-rhs`` of its contact row, which
        the predicted velocity does not hold. Past the limit the two rows' targets conflict: in every iteration
        the contact row pushes the sphere to that speed and the linear speed row behind it takes the overshoot
        back, so the step ends on the limit and the row's total grows by the overshoot each iteration.
        """
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                run = _pressed_spheres(self.device, articulated=articulated)
                velocity = _step(run)
                rows = _free_body_rows(run)
                contact = rows.type == PGS_CONSTRAINT_TYPE_CONTACT
                limit = rows.type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
                np.testing.assert_array_equal(contact.sum(axis=1), 1)
                np.testing.assert_array_equal(limit.sum(axis=1), 2)
                push = -rows.rhs[contact]
                # One sphere leaves below its limit; the others would overshoot it.
                self.assertTrue(np.any(push < CONTACT_LIMIT) and np.any(push > 1.1 * CONTACT_LIMIT))
                expected = np.zeros((len(DEPTHS), 6))
                expected[:, 2] = np.minimum(push, CONTACT_LIMIT)
                _assert_on_limit_scale(velocity, expected, CONTACT_LIMITS)
                linear = limit & (rows.axis == 0)
                overshoot = np.maximum(push - CONTACT_LIMIT, 0.0)[:, None]
                sweeps = run.solver.options.pgs_iterations
                total = np.where(linear, sweeps * overshoot * rows.eff_mass_inv, 0.0)
                np.testing.assert_allclose(
                    np.where(limit, rows.impulses, 0.0), total, rtol=0.0, atol=sweeps * RTOL * CONTACT_LIMIT * MASS
                )


class TestFreeBodyVelocityLimits(_FreeBodyVelocityLimitChecks, unittest.TestCase):
    device = "cpu"

    def test_rows_exist_wherever_the_builder_can_allocate_them(self):
        """Select the free-body velocity-limit rows wherever the row builder can allocate them.

        The selection follows the builder's structure, not the limits: registered but infinite limits
        select it although no row is built, and an infinite activation fraction declines it.
        """
        world = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
        for name, options, selected, built in (
            ("limits", {}, True, True),
            ("infinite_limits", {"limits": INFINITE_LIMITS}, True, False),
            ("unregistered", {"limits": None}, False, False),
            ("never_activated", {"fraction": float("inf")}, False, False),
        ):
            with self.subTest(name):
                run = _free_boxes(self.device, world, **options)
                self.assertEqual(run.solver.sweep.has_free_body_velocity_limit_rows, selected)
                _step(run)
                self.assertEqual(int(run.solver.placement.mf_constraint_count.numpy().sum()) > 0, built)


if __name__ == "__main__":
    unittest.main()

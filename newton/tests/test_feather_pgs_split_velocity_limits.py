# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold split-mode free rigid bodies to the stateless velocity-limit law on every device.

Split mode sweeps free-body rows after the dense rows: on the CPU in ``pgs_solve_mf_loop``, on CUDA in the
shared-memory sweep of ``_get_pgs_solve_mf_kernel``. Each finitely limited body axis owns a lower and an
upper velocity-limit row (``J = +e`` and ``-e``, ``rhs = limit``) at the tail of its world's rows. The
stateless law pushes a row only while its limit is violated (``J v + rhs < 0``), by the whole correction
whatever ``pgs_omega`` relaxes, and keeps that correction rather than an accumulated impulse. A sweep that
relaxes these rows as bilateral rows drives every limited axis onto its limit instead, so it accelerates a
body that moves below its limit.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 60.0
# Speed limits of every free box [m/s, rad/s].
LINEAR_LIMIT = 2.0
ANGULAR_LIMIT = 3.0
LIMITS = np.array([LINEAR_LIMIT] * 3 + [ANGULAR_LIMIT] * 3)
# Registered speed attributes that limit nothing [m/s, rad/s].
INFINITE_LIMITS = (float("inf"), float("inf"))
# A 1 kg box with a spherical inertia: its axes decouple and free flight keeps its angular velocity.
MASS = 1.0
INERTIA = 0.1
# Velocities agree to this fraction of their axis's limit: the default pgs_cfm leaves 1e-6 of each correction
# on a 1 kg box (1e-7 on its 0.1 kg m^2 axes), and free flight keeps a velocity to float32 rounding.
RTOL = 1.0e-5
# An under-relaxed sweep: the other rows take this fraction of their correction, the velocity-limit rows all of it.
RELAXED_OMEGA = 0.6

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
# Per-axis sweep inputs: every axis past its limit on one side or the other, or inside it.
SWEEP_LINEAR = ((3.0, -2.5, 1.0), (-1.0, 0.5, -2.0), (0.0, 4.0, -7.0))
SWEEP_ANGULAR = ((-3.5, 0.0, 2.9), (5.0, -1.0, -3.0), (0.2, -6.0, 3.1))


def _free_boxes(
    device, worlds, *, articulated=False, limits=(LINEAR_LIMIT, ANGULAR_LIMIT), fraction=0.0, omega=1.0, wrenches=None
):
    """One free box per world, in split mode, launched at ``worlds`` (linear, angular) velocities.

    Args:
        device: Warp device of the model and solver.
        worlds: Initial (linear [m/s], angular [rad/s]) velocity of each world's box.
        articulated: Add a one-joint arm to every world, so the split driver interleaves the dense and the
            free-body sweeps.
        limits: (linear [m/s], angular [rad/s]) speed limits of every box; None leaves the speed attributes
            unregistered.
        fraction: The solver's ``velocity_limit_activation_fraction``.
        omega: The solver's ``pgs_omega``.
        wrenches: Velocity change [m/s, rad/s] that a constant body wrench adds over one step, per world.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    attributes = None
    if limits is not None:
        SolverFeatherPGS.register_custom_attributes(builder)
        attributes = {"rigid_body_max_linear_velocity": limits[0], "rigid_body_max_angular_velocity": limits[1]}
    boxes = []
    for world in range(len(worlds)):
        builder.begin_world()
        box = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0 + world), wp.quat_identity()),
            mass=MASS,
            inertia=wp.mat33(np.eye(3) * INERTIA),
            lock_inertia=True,
            custom_attributes=attributes,
        )
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
        boxes.append(box)
        if articulated:
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
            builder.add_articulation([builder.add_joint_revolute(-1, link, axis=newton.Axis.Y)])
        builder.end_world()
    model = builder.finalize(device=device)
    # Without angular damping, free flight keeps the angular velocity exactly.
    solver = SolverFeatherPGS(
        model,
        pgs_mode="split",
        velocity_limit_activation_fraction=fraction,
        pgs_omega=omega,
        angular_damping=0.0,
    )
    state_0, state_1 = model.state(), model.state()
    starts = model.joint_qd_start.numpy()
    joints = [int(np.flatnonzero(model.joint_child.numpy() == box)[0]) for box in boxes]
    dofs = np.array([np.arange(starts[joint], starts[joint] + 6) for joint in joints])
    qd = state_0.joint_qd.numpy()
    qd[dofs] = np.array([np.concatenate(world) for world in worlds])
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    body_f = None
    if wrenches is not None:
        # A world-frame wrench at the center of mass that changes the box velocity by ``wrenches`` over one step.
        scale = np.array([MASS] * 3 + [INERTIA] * 3) / DT
        body_f = np.zeros((model.body_count, 6), dtype=np.float32)
        body_f[boxes] = np.array(wrenches) * scale
    pipeline = newton.CollisionPipeline(model)
    return SimpleNamespace(
        model=model,
        solver=solver,
        states=(state_0, state_1),
        control=model.control(),
        pipeline=pipeline,
        contacts=pipeline.contacts(),
        boxes=boxes,
        dofs=dofs,
        body_f=body_f,
    )


def _step(run):
    """Advance one step and return each box's (linear, angular) velocity."""
    state_0, state_1 = run.states
    state_0.clear_forces()
    if run.body_f is not None:
        state_0.body_f.assign(run.body_f)
    run.pipeline.collide(state_0, run.contacts)
    run.solver.step(state_0, state_1, run.control, run.contacts, DT)
    run.states = (state_1, state_0)
    return state_1.joint_qd.numpy()[run.dofs]


def _box_rows(run):
    """Return each world's free-body row types, axes and signs: [world, row]."""
    solver = run.solver
    count = solver.mf_constraint_count.numpy()
    rows = int(count.max())
    jacobian = solver.mf_J_a.numpy()[:, :rows]
    axis = np.argmax(np.abs(jacobian), axis=2)
    return SimpleNamespace(
        count=count,
        type=solver.mf_row_type.numpy()[:, :rows],
        axis=axis,
        sign=np.sign(np.take_along_axis(jacobian, axis[..., None], axis=2)[..., 0]),
        rhs=solver.mf_rhs.numpy()[:, :rows],
        eff_mass_inv=solver.mf_eff_mass_inv.numpy()[:, :rows],
    )


def _assert_on_limit_scale(actual, desired):
    """Compare [world, 6] velocities to ``RTOL`` of each axis's limit."""
    np.testing.assert_allclose(np.asarray(actual) / LIMITS, np.asarray(desired) / LIMITS, rtol=0.0, atol=RTOL)


def _clamped_magnitudes(velocity):
    """Scale each world's linear and angular velocity onto its limit where it exceeds it: [world, 6]."""
    clamped = np.array(velocity, dtype=np.float64)
    for part, limit in ((slice(0, 3), LINEAR_LIMIT), (slice(3, 6), ANGULAR_LIMIT)):
        speed = np.linalg.norm(clamped[:, part], axis=1, keepdims=True)
        clamped[:, part] *= np.minimum(1.0, limit / np.maximum(speed, 1.0e-30))
    return clamped


def _sweep(device, *, iterations, articulated=False, omega=1.0):
    """Sweep each box's velocity-limit rows ``iterations`` times from the ``SWEEP_*`` velocities.

    A step at rest builds the rows (``velocity_limit_activation_fraction=0`` allocates all twelve); the sweep
    then starts from velocities that the step's predictor clamp never saw, so the rows alone decide.
    """
    run = _free_boxes(
        device, [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))] * len(SWEEP_LINEAR), articulated=articulated, omega=omega
    )
    _step(run)
    solver = run.solver
    # The sweep reads the response DOFs the free-body map assigns, not the joint coordinates.
    articulation = solver.body_to_articulation.numpy()[run.boxes]
    dofs = solver.articulation_dof_start.numpy()[articulation][:, None] + np.arange(6)
    start = np.hstack([np.array(SWEEP_LINEAR), np.array(SWEEP_ANGULAR)]).astype(np.float32)
    velocity = solver.v_out.numpy()
    velocity[dofs] = start
    solver.v_out.assign(velocity)
    solver.mf_impulses.zero_()
    solver._mf_pgs_solve(iterations)
    rows = _box_rows(run)
    return SimpleNamespace(
        start=start,
        velocity=solver.v_out.numpy()[dofs],
        impulses=solver.mf_impulses.numpy()[:, : rows.type.shape[1]],
        rows=rows,
    )


class _SplitVelocityLimitChecks:
    """Checks shared by the CPU and CUDA cases; ``device`` names the device under test."""

    device = "cpu"

    def check_rows_are_velocity_limits(self, run):
        """Every box owns its twelve velocity-limit rows, on both sides of each axis."""
        rows = _box_rows(run)
        np.testing.assert_array_equal(rows.count, 12)
        np.testing.assert_array_equal(rows.type, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
        for world in range(rows.type.shape[0]):
            pairs = sorted(zip(rows.axis[world].tolist(), rows.sign[world].tolist(), strict=True))
            self.assertEqual(pairs, [(axis, sign) for axis in range(6) for sign in (-1.0, 1.0)])
        np.testing.assert_allclose(rows.rhs, LIMITS[rows.axis], rtol=1.0e-6)

    def test_sweep_pushes_only_violated_limits(self):
        """Push each violated limit back by its whole, unrelaxed correction, and leave satisfied rows at zero impulse."""
        for articulated in (False, True):
            for omega in (1.0, RELAXED_OMEGA):
                with self.subTest(articulated=articulated, omega=omega):
                    swept = _sweep(self.device, iterations=1, articulated=articulated, omega=omega)
                    rows = swept.rows
                    # One sweep: a violated axis lands on its limit, an axis inside both limits keeps its velocity.
                    _assert_on_limit_scale(swept.velocity, np.clip(swept.start, -LIMITS, LIMITS))
                    inside = np.abs(swept.start) <= LIMITS
                    np.testing.assert_allclose(swept.velocity[inside], swept.start[inside], rtol=0.0, atol=1.0e-6)
                    # Each row stores the stateless correction it applied from the starting velocity.
                    residual = rows.sign * np.take_along_axis(swept.start, rows.axis, axis=1) + rows.rhs
                    correction = np.where(residual < 0.0, -residual * rows.eff_mass_inv, 0.0)
                    np.testing.assert_allclose(swept.impulses, correction, rtol=RTOL, atol=0.0)
                    self.assertTrue(np.any(correction > 0.0))

    def test_sweep_keeps_corrections_not_accumulated_impulses(self):
        """Converge onto the limits and store each row's last correction, not the impulse it accumulated."""
        swept = _sweep(self.device, iterations=12)
        _assert_on_limit_scale(swept.velocity, np.clip(swept.start, -LIMITS, LIMITS))
        # The first sweep applied up to MASS * 5 m/s; the last finds the limits satisfied to cfm rounding.
        self.assertLess(np.abs(swept.impulses).max(), 1.0e-4)

    def test_step_keeps_speeds_below_the_limits(self):
        """Keep the velocity of a free body that moves below its limits and feels no other force."""
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
                    if fraction == 0.7:
                        # The activation gate reserved rows only for axes within 30% of their limit.
                        self.assertGreater(int(run.solver.mf_constraint_count.numpy().sum()), 0)

    def test_step_clamps_speeds_past_the_limits(self):
        """Hold a free body launched or pushed past its limits on them, along its own direction."""
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

    def test_route_records_the_velocity_limit_sweep(self):
        """Record the stateless free-body law, and on CUDA its ``_vlim`` sweep, wherever limit rows can exist."""
        world = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
        for name, options, law, words in (
            ("limits", {}, "stateless", "unrelaxed correction"),
            ("infinite_limits", {"limits": INFINITE_LIMITS}, "stateless", "unrelaxed correction"),
            ("unregistered", {"limits": None}, None, "not registered"),
            ("never_activated", {"fraction": float("inf")}, None, "fraction=inf"),
        ):
            with self.subTest(name):
                solver = _free_boxes(self.device, world, **options).solver
                routes = solver.get_path_selection()["routes"]
                self.assertEqual(routes["velocity_limits"]["free_body_sweep_law"], law)
                self.assertIn(words, routes["velocity_limits"]["free_body_sweep_law_reason"])
                key = routes["sweep"]["free_body_matrix_free"]
                if self.device == "cpu":
                    self.assertEqual(key, "pgs_solve_mf_loop")
                else:
                    suffix = "_vlim" if law else ""
                    self.assertEqual(key, f"pgs_solve_mf_{solver.mf_max_constraints}_{solver.max_mf_bodies}{suffix}")


class TestSplitFreeBodyVelocityLimits(_SplitVelocityLimitChecks, unittest.TestCase):
    device = "cpu"

    def test_sweep_compiles_the_law_wherever_rows_can_exist(self):
        """Select the velocity-limit sweep wherever the row builder can allocate free-body limit rows.

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
                self.assertEqual(run.solver._has_free_body_velocity_limit_rows, selected)
                _step(run)
                self.assertEqual(int(run.solver.mf_constraint_count.numpy().sum()) > 0, built)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSplitFreeBodyVelocityLimitsCUDA(_SplitVelocityLimitChecks, unittest.TestCase):
    device = "cuda:0"

    def test_cuda_sweep_and_step_match_the_cpu(self):
        """Reproduce the CPU free-body sweep and step on CUDA."""
        for iterations, omega in ((1, 1.0), (1, RELAXED_OMEGA), (12, 1.0)):
            with self.subTest(sweep_iterations=iterations, omega=omega):
                cpu = _sweep("cpu", iterations=iterations, omega=omega)
                cuda = _sweep(self.device, iterations=iterations, omega=omega)
                _assert_on_limit_scale(cuda.velocity, cpu.velocity)
                if iterations == 1:
                    np.testing.assert_allclose(cuda.impulses, cpu.impulses, rtol=RTOL, atol=1.0e-6)
        worlds = list(BELOW) + [(linear, angular) for linear, angular, _, _ in PAST]
        wrenches = [np.zeros(6)] * len(BELOW) + [np.concatenate((force, torque)) for _, _, force, torque in PAST]
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                cpu, cuda = (
                    _free_boxes(device, worlds, articulated=articulated, wrenches=wrenches)
                    for device in ("cpu", self.device)
                )
                for _ in range(3):
                    _assert_on_limit_scale(_step(cuda), _step(cpu))


if __name__ == "__main__":
    unittest.main()

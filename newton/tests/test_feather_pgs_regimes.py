# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The regime suite: FeatherPGS against absolute references, across regimes the tasks do not reach.

Conformance compares the solver with itself at a larger budget, so a wrong law is common-mode and passes. Here each
family Q1-Q20 has a reference that shares nothing with the solver: a closed form, a float64 discrete law, quadrature,
float64 RK4 (:mod:`newton.tests.feather_pgs_references`), a manufactured solution, or the same problem at the origin.

A known failure is an expected failure that names its phase or says it is a documented limit (:func:`known_failure`):
when the phase lands, its test reports an unexpected success and the mark goes. Where a family holds in one regime and
fails in another, the two are separate tests, so what passes stays guarded. Runs on the CPU in about three minutes.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.factor.walk import TreeWalk, local_frames
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_references as refs
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance
from newton.tests.feather_pgs_law_harness import walk_inverse
from newton.tests.test_feather_pgs_factor import newton_state_body_q, relative, transform_point

H = 1.0 / 240.0
G0 = refs.GRAVITY
DEVICE = "cpu"
BASE = {"angular_damping": 0.0, "pgs_iterations": 64}
"""The options of every regime: no artificial damping, a budget that converges small scenes."""


LIMIT = "documented limit"
"""A known failure no phase is meant to fix: what the approved method does at that scale, stated so nobody takes it for
a defect."""


def known_failure(phase: str, why: str):
    """Mark a test as an expected failure of main, fixed by ``phase`` (a roadmap phase or stream), or a :data:`LIMIT`.

    Only a failed assertion is the known failure: any other exception is a broken test and stays an error, which
    :func:`unittest.expectedFailure` alone would hide. A test that passes reports an unexpected success, which fails the
    run, so the mark goes with its phase. Such a test asserts outside ``subTest``, which keeps the exception from it.
    """

    def mark(test):
        @functools.wraps(test)
        def run(self):
            try:
                test(self)
            except AssertionError:
                raise
            except Exception:
                self._outcome.expecting_failure = False
                raise

        run.known_failure = (phase, why)
        return unittest.expectedFailure(run)

    return mark


def _hinge(
    *,
    inertia=0.01,
    ke=0.0,
    kd=0.0,
    effort=0.0,
    stiffness=0.0,
    damping=0.0,
    armature=0.0,
):
    """A hinge about z through a body's centre: its one coordinate has inertia ``inertia``, no gravity torque."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia))
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 0.0, 1.0),
        target_ke=ke,
        target_kd=kd,
        effort_limit=effort,
        armature=armature,
        limit_lower=-1.0e6,
        limit_upper=1.0e6,
        spring_stiffness=stiffness,
        spring_ref=0.0,
        damping=damping,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=DEVICE)


def _roll(model, steps, *, dt=H, setup=None, each=None, before=None, contacts=True, **options):
    """Step ``model``: ``setup(state)`` before the first step, ``before(state, step)`` before each one after its forces
    clear, ``each(state, solver, contacts, step)`` after it. Returns the solver and the last state."""
    settings = {**BASE, **options}
    solver = SolverFeatherPGS(model, **settings)
    state, out = model.state(), model.state()
    if setup is not None:
        setup(state)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    pipeline = newton.CollisionPipeline(model) if contacts else None
    stream = pipeline.contacts() if contacts else None
    for step in range(steps):
        state.clear_forces()
        if before is not None:
            before(state, step)
        if contacts:
            pipeline.collide(state, stream)
        solver.step(state, out, control, stream, dt)
        state, out = out, state
        if each is not None:
            each(state, solver, stream, step)
    return solver, state


def _set_qd(state, values: dict) -> None:
    qd = state.joint_qd.numpy()
    for index, value in values.items():
        qd[index] = value
    state.joint_qd.assign(qd)


def _free_box(
    *,
    half=(0.1, 0.1, 0.1),
    z=None,
    mu=0.5,
    density=1000.0,
    gravity=(0.0, 0.0, -G0),
    restitution=0.0,
    gap=0.01,
    cap=None,
    prismatic=False,
):
    """One box resting on (or overlapping) the ground, free or on a vertical prismatic joint."""
    builder = newton.ModelBuilder(gravity=gravity)
    SolverFeatherPGS.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(density=density, mu=mu, restitution=restitution, gap=gap)
    z = half[2] if z is None else z
    if prismatic:
        body = builder.add_link(xform=wp.transform((0.0, 0.0, z), wp.quat_identity()))
        builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
        joint = builder.add_joint_prismatic(
            parent=-1, child=body, axis=(0.0, 0.0, 1.0), parent_xform=wp.transform((0.0, 0.0, z), wp.quat_identity())
        )
        builder.add_articulation([joint])
    else:
        body = builder.add_body(xform=wp.transform((0.0, 0.0, z), wp.quat_identity()))
        builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=DEVICE)
    if cap is not None:
        model.rigid_body_max_depenetration_velocity.fill_(cap)
    return model


RATIOS = (0.1, 1.0, 10.0, 100.0, 1000.0)
"""The stiffness ``K / I`` of the saturated drives."""


class TestQ01Drive(unittest.TestCase):
    """Q1: a drive saturates at ``limit / I``, not ``limit / (I + K)``. Reference: the exact Huber step."""

    def _drive(self, ratio, *, effort, steps=40):
        inertia = 0.01
        stiffness = ratio * inertia
        ke, kd = 0.5 * stiffness / (H * H), 0.5 * stiffness / H
        model = _hinge(inertia=inertia, ke=ke, kd=kd, effort=effort)
        solver = SolverFeatherPGS(model, enable_joint_limits=False, **BASE)
        state, out = model.state(), model.state()
        control = model.control()
        control.joint_target_q = wp.array([10.0], dtype=wp.float32, device=DEVICE)
        for _step in range(steps):
            solver.step(state, out, control, None, H)
            state, out = out, state
        reference = refs.drive_rollout(inertia, ke, kd, effort, 10.0, dt=H, steps=steps)
        return float(state.joint_q.numpy()[0]), float(state.joint_qd.numpy()[0]), reference[-1]

    def test_unsaturated(self):
        """Below its limit a drive is its implicit step, at every stiffness."""
        for ratio in (1.0e-2, 1.0, 1.0e3):
            with self.subTest(ratio=ratio):
                q, v, (rq, rv) = self._drive(ratio, effort=0.0)
                self.assertAlmostEqual(v, rv, delta=2.0e-3 * max(abs(rv), 1.0))
                self.assertAlmostEqual(q, rq, delta=2.0e-3 * max(abs(rq), 1.0))

    def test_saturated_soft(self):
        """A soft saturated drive accelerates at ``limit / I`` within the 1% its ``K / I`` of 1e-2 costs."""
        _q, v, (_rq, rv) = self._drive(1.0e-2, effort=1.0)
        self.assertAlmostEqual(v, rv, delta=0.02 * abs(rv))

    def test_saturated_stiff(self):
        """A stiff saturated drive accelerates at ``limit / I``."""
        speeds = {ratio: (self._drive(ratio, effort=1.0)[1], self._drive(ratio, effort=1.0)[2][1]) for ratio in RATIOS}
        wrong = {ratio: pair for ratio, pair in speeds.items() if abs(pair[0] - pair[1]) > 0.02 * abs(pair[1])}
        self.assertEqual(wrong, {})


class TestQ02PassiveJoint(unittest.TestCase):
    """Q2: passive springs and dampers. Reference: the discrete oscillator; a damper must never gain energy."""

    def _spin(self, model, steps, *, q0=0.0, v0=0.0):
        qs, vs = [], []

        def setup(state):
            state.joint_q.assign([q0])
            state.joint_qd.assign([v0])

        def each(state, solver, contacts, step):
            qs.append(float(state.joint_q.numpy()[0]))
            vs.append(float(state.joint_qd.numpy()[0]))

        _roll(model, steps, setup=setup, each=each, contacts=False, enable_joint_limits=False)
        return np.asarray(qs), np.asarray(vs)

    def test_spring_is_the_symplectic_oscillator(self):
        """An undamped spring keeps symplectic Euler's bounded orbit up to ``omega h = 1.9``."""
        inertia = 0.01
        for omega_h in (0.1, 1.0, 1.9):
            with self.subTest(omega_h=omega_h):
                omega = omega_h / H
                model = _hinge(inertia=inertia, stiffness=inertia * omega * omega)
                q, _v = self._spin(model, 120, q0=0.1)
                reference = refs.spring_rollout(omega, 0.1, dt=H, steps=120)
                np.testing.assert_allclose(q, reference[:, 0], atol=2.0e-3)

    def test_damper_decays(self):
        """A damper within the explicit bound decays, within the first-order error of ``exp(-c t / I)``."""
        inertia = 0.01
        for ratio in (0.1, 0.5, 1.0):
            with self.subTest(ratio=ratio):
                c = ratio * inertia / H
                _q, v = self._spin(_hinge(inertia=inertia, damping=c), 20, v0=1.0)
                self.assertTrue(np.all(np.abs(v[1:]) <= np.abs(v[:-1]) + 1.0e-6))
                if ratio == 0.1:
                    self.assertAlmostEqual(float(v[-1]), math.exp(-2.0), delta=0.15 * math.exp(-2.0))

    def test_stiff_damper_stays_bounded(self):
        """A stiff damper decays at any ``h c / I``: a passive element cannot add energy."""
        inertia = 0.01
        peaks = {}
        for ratio in (2.5, 4.2, 10.0):
            _q, v = self._spin(_hinge(inertia=inertia, damping=ratio * inertia / H), 60, v0=1.0)
            peaks[ratio] = float(np.max(np.abs(v)))
        self.assertLess(max(peaks.values()), 1.0 + 1.0e-3, peaks)


class TestQ03GapLaw(unittest.TestCase):
    """Q3: a contact's gap law, ``beta phi / h``, capped by its bodies' depenetration speed, on both row stores."""

    def _push(self, depth, *, cap=None, prismatic=False):
        model = _free_box(z=0.1 - depth, gravity=(0.0, 0.0, 0.0), cap=cap, prismatic=prismatic)
        _solver, state = _roll(model, 1, pgs_iterations=24)
        return float(state.joint_qd.numpy()[2 if not prismatic else 0])

    def test_uncapped_on_both_stores(self):
        """Free-body rows and dense rows push at ``beta |phi| / h``."""
        for prismatic in (False, True):
            for depth in (0.002, 0.01, 0.05):
                with self.subTest(prismatic=prismatic, depth=depth):
                    self.assertAlmostEqual(
                        self._push(depth, prismatic=prismatic), 0.2 * depth / H, delta=0.01 * 0.2 * depth / H
                    )

    def test_cap_on_free_body_rows(self):
        """A free body's rows stop at its cap."""
        self.assertAlmostEqual(self._push(0.02, cap=0.5), 0.5, delta=0.005)

    def test_cap_on_dense_rows(self):
        """A jointed body's rows stop at its cap too: the law cannot depend on the row store."""
        self.assertAlmostEqual(self._push(0.02, cap=0.5, prismatic=True), 0.5, delta=0.005)


def _sphere_world(radius, *, spheres, restitution, gravity, mu=0.0, ground=True, gap=0.05):
    """Spheres at ``spheres`` positions, optionally above the ground."""
    builder = newton.ModelBuilder(gravity=gravity)
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu, restitution=restitution, gap=gap)
    for position in spheres:
        body = builder.add_body(xform=wp.transform(position, wp.quat_identity()))
        builder.add_shape_sphere(body, radius=radius, cfg=cfg)
    if ground:
        builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=DEVICE)


class TestQ04Restitution(unittest.TestCase):
    """Q4: Newton's restitution. A firing row targets ``-e u`` of its frozen incident velocity."""

    def _bounce(self, restitution, height, **options):
        """Drop a ball: its speed entering the first firing step and leaving it."""
        model = _sphere_world(0.1, spheres=[(0.0, 0.0, 0.1 + height)], restitution=restitution, gravity=(0.0, 0.0, -G0))
        history = []

        def each(state, solver, contacts, step):
            history.append(float(state.joint_qd.numpy()[2]))

        _roll(model, 120, each=each, **options)
        v = np.asarray(history)
        turn = int(np.argmax(v > -1.0e-4)) if np.any(v > -1.0e-4) else None
        self.assertIsNotNone(turn)
        return -(v[turn - 1] - G0 * H), v[turn]

    def test_coefficient(self):
        """The rebound is ``e`` of the incident speed, the step's gravity included."""
        for restitution in (0.25, 0.5, 0.75, 1.0):
            with self.subTest(e=restitution):
                incident, rebound = self._bounce(restitution, 0.2)
                self.assertAlmostEqual(rebound, restitution * incident, delta=0.02 * incident)

    def test_threshold(self):
        """Below the velocity threshold a contact does not bounce; above it, it does."""
        incident, rebound = self._bounce(0.8, 0.02, restitution_velocity_threshold=1.0)
        self.assertLess(incident, 1.0)
        self.assertLess(rebound, 0.01)
        incident, rebound = self._bounce(0.8, 0.2, restitution_velocity_threshold=1.0)
        self.assertGreater(incident, 1.0)
        self.assertAlmostEqual(rebound, 0.8 * incident, delta=0.02 * incident)

    def test_gapped_cradle(self):
        """Equal balls with gaps, ``e = 1``: each collision hands the momentum on, and the last ball leaves alone."""
        radius, gap = 0.05, 0.02
        positions = [(index * (2.0 * radius + gap), 0.0, 0.0) for index in range(3)]
        model = _sphere_world(
            radius, spheres=positions, restitution=1.0, gravity=(0.0, 0.0, 0.0), ground=False, gap=0.03
        )

        def setup(state):
            _set_qd(state, {0: 1.0})

        _solver, state = _roll(model, 60, setup=setup, restitution_velocity_threshold=0.01)
        speeds = state.joint_qd.numpy()[[0, 6, 12]]
        np.testing.assert_allclose(speeds, [0.0, 0.0, 1.0], atol=0.02)


class TestQ05Coulomb(unittest.TestCase):
    """Q5: a box on a slope, its angle around ``atan mu``. Reference: Coulomb's analytic stick and slide."""

    def _slope(self, mu, angle, seconds):
        gravity = (G0 * math.sin(angle), 0.0, -G0 * math.cos(angle))
        model = _free_box(half=(0.1, 0.1, 0.02), mu=mu, gravity=gravity, gap=0.005)
        _solver, state = _roll(model, round(seconds / H))
        return float(state.body_q.numpy()[0, 0]), float(state.body_qd.numpy()[0, 0])

    def test_sticks_below_the_angle(self):
        """Below ``atan mu`` the box holds: no creep."""
        for mu in (0.1, 0.5, 1.0, 2.0):
            with self.subTest(mu=mu):
                x, v = self._slope(mu, math.atan(mu) * 0.9, 1.0)
                self.assertLess(abs(x), 1.0e-3)
                self.assertLess(abs(v), 1.0e-3)

    def test_slides_above_the_angle(self):
        """Above ``atan mu`` the box accelerates at ``g (sin a - mu cos a)``."""
        for mu in (0.1, 0.5, 1.0):
            with self.subTest(mu=mu):
                angle = math.atan(mu) + math.radians(5.0)
                _x, v = self._slope(mu, angle, 0.5)
                expected = G0 * (math.sin(angle) - mu * math.cos(angle)) * 0.5
                self.assertAlmostEqual(v, expected, delta=0.02 * expected)


class TestQ06RegionWrench(unittest.TestCase):
    """Q6: a flat patch's friction wrench. Reference: Coulomb at every point of the footprint, by quadrature."""

    def _spin_slide(self, half_x, half_y, speed, spin):
        model = _free_box(half=(half_x, half_y, 0.02), gap=0.005)

        def setup(state):
            _set_qd(state, {0: speed, 5: spin})

        yaw = [0.0]

        def each(state, solver, contacts, step):
            yaw[0] += H * float(state.body_qd.numpy()[0, 5])

        _solver, state = _roll(model, round(1.2 / H), setup=setup, each=each)
        return float(np.linalg.norm(state.body_q.numpy()[0, :2])), yaw[0]

    def test_pure_slide(self):
        """A sliding patch without spin stops where the disk law says."""
        for half_x, half_y in ((0.05, 0.05), (0.15, 0.05)):
            with self.subTest(half=(half_x, half_y)):
                x, _yaw = self._spin_slide(half_x, half_y, 1.0, 0.0)
                expected, _theta, _stop = refs.patch_slide_spin(half_x, half_y, 1.0, 0.0, 0.5)
                self.assertAlmostEqual(x, expected, delta=0.03 * expected)

    def _yaw_errors(self, cases, tolerance):
        wrong = {}
        for half_x, half_y in ((0.05, 0.05), (0.15, 0.05)):
            for speed, spin in cases:
                _x, yaw = self._spin_slide(half_x, half_y, speed, spin)
                _distance, expected, _stop = refs.patch_slide_spin(half_x, half_y, speed, spin, 0.5)
                if abs(yaw - expected) > tolerance * abs(expected):
                    wrong[half_x, speed] = (math.degrees(yaw), math.degrees(expected))
        return wrong

    def test_spin_capacity(self):
        """A spinning patch sweeps the yaw quadrature gives: the region's one wrench holds its spin on the footprint's
        arm (``mu_t = 0.94 mu`` times the hull's radius of gyration), within 5%."""
        self.assertEqual(self._yaw_errors(((0.0, 30.0), (0.3, 30.0)), 0.05), {})

    @known_failure(
        LIMIT,
        "the region's ball over-holds mixed slide and spin: the exact limit surface lies 7-11% inside it between 30 "
        "and 60 degrees (core-docs/WRENCH/VERDICT.md L2), so a fast slide sheds its spin early (8-18% less yaw)",
    )
    def test_mixed_slide_spin(self):
        """A patch sliding fast while it spins sweeps the yaw quadrature gives, within 5%."""
        self.assertEqual(self._yaw_errors(((1.0, 10.0),), 0.05), {})


class TestQ07Rolling(unittest.TestCase):
    """Q7: rolling. A slider ends rolling at ``v0 / (1 + k)``, and a rolling body keeps its speed."""

    def test_sphere_slides_to_five_sevenths(self):
        radius, v0 = 0.05, 1.0
        model = _sphere_world(
            radius, spheres=[(0.0, 0.0, radius)], restitution=0.0, gravity=(0.0, 0.0, -G0), mu=0.3, gap=0.005
        )
        _solver, state = _roll(model, 240, setup=lambda state: _set_qd(state, {0: v0}))
        self.assertAlmostEqual(float(state.joint_qd.numpy()[0]), refs.rolling_speed(v0, 0.4), delta=0.02 * v0)

    def _disc(self, density=7800.0):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        cfg = newton.ModelBuilder.ShapeConfig(density=density, mu=0.3, gap=0.002)
        radius = 0.015
        turn = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
        body = builder.add_body(xform=wp.transform((0.0, 0.0, radius), turn))
        builder.add_shape_cylinder(body, radius=radius, half_height=0.0015, cfg=cfg)
        builder.add_ground_plane(cfg=cfg)
        return builder.finalize(device=DEVICE), radius

    def test_disc_slides_to_two_thirds(self):
        model, _radius = self._disc()
        v0 = 0.3
        _solver, state = _roll(model, 240, setup=lambda state: _set_qd(state, {0: v0}))
        self.assertAlmostEqual(float(state.body_qd.numpy()[0, 0]), refs.rolling_speed(v0, 0.5), delta=0.03 * v0)

    def test_rolling_keeps_its_speed(self):
        """A rolling disc has no drag: the stick holds the contact point, not the rolling."""
        model, radius = self._disc()
        v0 = 0.1
        _solver, state = _roll(model, 480, setup=lambda state: _set_qd(state, {0: v0, 4: v0 / radius}))
        self.assertAlmostEqual(float(np.linalg.norm(state.body_qd.numpy()[0, :2])), v0, delta=0.02 * v0)


class TestQ08Tipping(unittest.TestCase):
    """Q8: a force high on a box tips it past ``F h = m g w / 2``, about its far edge, on four corner contacts."""

    WIDTH, HEIGHT, TALL = 0.2, 0.4, 0.2

    def _tilt(self, fraction):
        half = (self.WIDTH / 2.0, self.WIDTH / 2.0, self.TALL)
        model = _free_box(half=half, mu=2.0, gap=0.005, density=1000.0)
        mass = float(model.body_mass.numpy()[0])
        force = fraction * refs.tipping_force(mass, self.WIDTH, self.HEIGHT)
        torque = force * (self.HEIGHT - self.TALL)

        def before(state, step):
            state.body_f.assign([[force, 0.0, 0.0, 0.0, torque, 0.0]])

        _solver, state = _roll(model, 120, before=before)
        quat = state.body_q.numpy()[0, 3:]
        return 2.0 * math.atan2(float(quat[1]), float(quat[3]))

    def test_threshold(self):
        """It holds at 95% of the force and goes over at 105%."""
        self.assertLess(abs(self._tilt(0.95)), 2.0e-3)
        self.assertGreater(self._tilt(1.05), 2.0e-2)


class TestQ09Gyroscopic(unittest.TestCase):
    """Q9: a tumbling body keeps its energy and momentum. Reference: the invariants."""

    INERTIA = (0.02, 0.03, 0.05)

    def _model(self, kind):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag(self.INERTIA)))
        if kind == "d6":
            axis = newton.ModelBuilder.JointDofConfig
            joint = builder.add_joint_d6(
                parent=-1,
                child=body,
                linear_axes=[axis(axis=a) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))],
                angular_axes=[axis(axis=a) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))],
            )
            builder.add_articulation([joint])
        elif kind == "free":
            builder.add_articulation([builder.add_joint_free(child=body)])
        else:
            root = builder.add_joint_free(child=body)
            child = builder.add_link(mass=0.2, inertia=wp.mat33(np.diag((0.002, 0.002, 0.003))))
            hinge = builder.add_joint_revolute(
                parent=body,
                child=child,
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform((-0.05, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-1.0e6,
                limit_upper=1.0e6,
            )
            builder.add_articulation([root, hinge])
        return builder.finalize(device=DEVICE)

    @staticmethod
    def _energy(model, state):
        q, v = state.body_q.numpy(), state.body_qd.numpy()
        mass, inertia = model.body_mass.numpy(), model.body_inertia.numpy()
        total = 0.0
        for k in range(model.body_count):
            rotation = np.array(wp.quat_to_matrix(wp.quat(*q[k, 3:]))).reshape(3, 3)
            total += (
                0.5 * mass[k] * v[k, :3] @ v[k, :3] + 0.5 * v[k, 3:] @ rotation @ inertia[k] @ rotation.T @ v[k, 3:]
            )
        return float(total)

    @staticmethod
    def _momentum(model, state):
        """The bodies' angular momentum about the origin."""
        q, v = state.body_q.numpy(), state.body_qd.numpy()
        mass, inertia, com = model.body_mass.numpy(), model.body_inertia.numpy(), model.body_com.numpy()
        total = np.zeros(3)
        for k in range(model.body_count):
            rotation = np.array(wp.quat_to_matrix(wp.quat(*q[k, 3:]))).reshape(3, 3)
            total += np.cross(q[k, :3] + rotation @ com[k], mass[k] * v[k, :3])
            total += rotation @ inertia[k] @ rotation.T @ v[k, 3:]
        return total

    def _drift(self, kind, wh, *, seconds=0.3, spin=(10.0, 5.0, 20.0)):
        """Relative change of a tumble's energy at ``|w| h = wh``."""
        return self._tumble(kind, wh, seconds, spin)[0]

    def _tumble(self, kind, wh, seconds=0.3, spin=(10.0, 5.0, 20.0)):
        """Relative change of a tumble's energy and angular momentum at ``|w| h = wh``."""
        h = wh / float(np.linalg.norm(spin))
        steps = round(seconds / h)
        model = self._model(kind)
        solver = SolverFeatherPGS(model, angular_damping=0.0)
        state, out = model.state(), model.state()
        _set_qd(state, {3: spin[0], 4: spin[1], 5: spin[2]})
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        start, momentum = self._energy(model, state), self._momentum(model, state)
        control = model.control()
        for _step in range(steps):
            solver.step(state, out, control, None, h)
            state, out = out, state
        moved = float(np.linalg.norm(self._momentum(model, state) - momentum) / np.linalg.norm(momentum))
        return (self._energy(model, state) - start) / start, moved

    def test_free_body_keeps_its_energy(self):
        """The Cayley update keeps a free body's energy at any step, a Dzhanibekov tumble included."""
        for wh in (0.01, 0.1, 0.5):
            with self.subTest(wh=wh):
                self.assertLess(abs(self._drift("free", wh, spin=(0.2, 30.0, 0.2))), 2.0e-3)

    def test_free_body_momentum_is_first_order(self):
        """The Cayley update keeps energy, not momentum: a free body's ``L`` drifts at first order in ``w h``."""
        drifts = {wh: self._tumble("free", wh, spin=(0.2, 30.0, 0.2))[1] for wh in (0.01, 0.1, 0.5)}
        self.assertLess(drifts[0.01], 1.0e-3, drifts)
        self.assertLess(drifts[0.5], 0.06, drifts)
        self.assertGreater(drifts[0.1] / drifts[0.01], 6.0, drifts)
        self.assertLess(drifts[0.1] / drifts[0.01], 14.0, drifts)

    def test_trees_are_first_order(self):
        """A tree's energy error halves with the step: the small-step path works (decision 10-27)."""
        for kind in ("tree", "d6"):
            with self.subTest(kind=kind):
                coarse, fine = self._drift(kind, 0.02), self._drift(kind, 0.01)
                self.assertGreater(abs(coarse / fine), 1.5)
                self.assertLess(abs(coarse / fine), 3.0)

    @known_failure(LIMIT, "gyroscopic trees are first order: the approved path is small steps (decision 10-27)")
    def test_trees_keep_their_energy(self):
        """A tumbling tree keeps its energy at a task's step."""
        drifts = {kind: self._drift(kind, 0.1) for kind in ("tree", "d6")}
        self.assertLess(max(abs(drift) for drift in drifts.values()), 1.0e-2, drifts)


class TestQ10MassRatio(unittest.TestCase):
    """Q10: heavy on light. The rigid claim holds to a mass ratio of about half the budget; past it the light box sinks
    and the stack topples, until load-path rows carry the heavy box's load through the light one."""

    @staticmethod
    def _overlap(ratio, sweeps):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        bodies = []
        for k, density in enumerate((1000.0, 1000.0 * ratio)):
            cfg = newton.ModelBuilder.ShapeConfig(density=density, mu=0.5, gap=0.005)
            body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05 + 0.1001 * k), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
            bodies.append(body)
        builder.add_ground_plane()
        model = builder.finalize(device=DEVICE)
        worst = [0.0]

        def each(state, solver, contacts, step):
            q = state.body_q.numpy()
            worst[0] = min(worst[0], float(q[1, 2] - q[0, 2]) - 0.1)

        _roll(model, 240, each=each, pgs_iterations=sweeps)
        return -worst[0]

    @staticmethod
    def _motion(densities, sweeps):
        """Boxes of 10 cm stacked on the ground, box ``k`` (from the bottom) of density ``1000 densities[k]``.

        Returns the bottom box's deepest sink into the ground [m], the top box's deepest overlap into the box under it
        while every box stands upright (tilt under 5 degrees) [m], and the largest tilt of any box [degrees].
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        for k, density in enumerate(densities):
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0 * density, mu=0.5, gap=0.005)
            body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05 + 0.1001 * k), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        builder.add_ground_plane()
        model = builder.finalize(device=DEVICE)
        worst = {"sink": 0.0, "overlap": 0.0, "tilt": 0.0}

        def each(state, solver, contacts, step):
            q = state.body_q.numpy().astype(np.float64)
            # A box's tilt is the angle between its z axis and the world's, from its quaternion (x, y, z, w).
            cosines = np.clip(1.0 - 2.0 * (q[:, 3] ** 2 + q[:, 4] ** 2), -1.0, 1.0)
            tilt = float(np.degrees(np.arccos(cosines)).max())
            worst["tilt"] = max(worst["tilt"], tilt)
            worst["sink"] = max(worst["sink"], 0.05 - q[0, 2])
            if tilt < 5.0:
                worst["overlap"] = max(worst["overlap"], 0.1 - (q[-1, 2] - q[-2, 2]))

        _roll(model, 240, each=each, pgs_iterations=sweeps)
        return worst["sink"], worst["overlap"], worst["tilt"]

    def test_ratio_within_the_budget(self):
        """A ratio up to half the budget holds to a millimetre."""
        self.assertLess(self._overlap(12.0, 24), 1.0e-3)

    def test_overlap_scale(self):
        """Inside the limit the overlap follows ``5e-5 (ratio / N)^1.4`` [m], about 0.05 mm at ``ratio = N``, and stays
        under the rigid bound ``3e-4 (ratio / N)^2``.

        The scale is the friction law's, measured over mass ratios 3 to 100 at 12 to 96 sweeps (each point within a
        factor 2.2 of it inside the budget): the stick holds the light box, so the overlap grows more slowly with
        ``ratio / N`` than the bound.
        """
        for ratio, sweeps in ((10.0, 24), (30.0, 24), (100.0, 24), (30.0, 96), (100.0, 96)):
            with self.subTest(ratio=ratio, sweeps=sweeps):
                scale = 5.0e-5 * (ratio / sweeps) ** 1.4
                overlap = self._overlap(ratio, sweeps)
                self.assertGreater(overlap, 0.3 * scale)
                self.assertLess(overlap, 3.0e-4 * (ratio / sweeps) ** 2)

    @known_failure(
        "H: heavy on light, the load-path rows",
        "Gauss-Seidel contracts the heavy box's six transmission modes through a light box by 1 - c / ratio per sweep "
        "in every row order, so a cold solve leaves exp(-c N / ratio) of the load unsolved: the light box sinks and the "
        "stack topples",
    )
    def test_ratio_past_the_budget(self):
        """Past the budget, to a mass ratio of 1e4 at 24 sweeps, the stack neither sinks nor tilts.

        Main sinks 17-79 mm from 300:1 and topples or squeezes the light box out; three boxes topple from 100:1.
        Load-path rows solve the transmission in about 8 sweeps at any ratio.
        """
        failures = {}
        for densities in ((1.0, 300.0), (1.0, 1.0e3), (1.0, 1.0e4), (1.0, 1.0, 1.0e4)):
            sink, overlap, tilt = self._motion(densities, 24)
            if not (sink < 1.0e-3 and overlap < 1.0e-3 and tilt < 1.0):
                failures[densities] = {"sink": sink, "overlap": overlap, "tilt": tilt}
        self.assertFalse(failures, failures)


class TestQ11Stack(unittest.TestCase):
    """Q11: a stack's load. Reference: the weight above each interface; here at the ground."""

    @staticmethod
    def _ground_load(count, sweeps):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        cfg = newton.ModelBuilder.ShapeConfig(density=125.0, mu=0.6, gap=0.002)
        half = 0.05
        for index in range(count):
            body = builder.add_body(xform=wp.transform((0.0, 0.0, half + 2.0 * half * index), wp.quat_identity()))
            builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)
        builder.add_ground_plane(cfg=cfg)
        model = builder.finalize(device=DEVICE)
        solver, state = _roll(model, 120, pgs_iterations=sweeps, mf_max_constraints=12 * count + 12)
        mass = float(model.body_mass.numpy()[0])
        counts = int(solver.placement.mf_constraint_count.numpy()[0])
        lam = solver.placement.impulses.numpy()[:, solver.placement.body_base :][0, :counts]
        kinds = solver.placement.row_type.numpy()[:, solver.placement.body_base :][0, :counts]
        sides = solver.placement.row_sides.numpy()[:, solver.placement.body_base :][0, :counts]
        ground = (kinds == 0) & ((sides[:, 0] < 0) | (sides[:, 1] < 0))
        return float(np.sum(lam[ground])) / (H * mass * G0 * count), float(state.body_q.numpy()[-1, 2]) - (
            half + 2.0 * half * (count - 1)
        )

    def test_shallow_stacks_carry_their_load(self):
        """A stack of up to 64 boxes puts its weight on the ground at a budget of ``8 n``."""
        for count in (1, 2, 4, 8, 16, 32, 64):
            with self.subTest(count=count):
                load, drop = self._ground_load(count, 8 * count)
                self.assertAlmostEqual(load, 1.0, delta=0.02)
                self.assertLess(abs(drop), 5.0e-3)


class TestQ19Wall(unittest.TestCase):
    """Q19: a staggered wall of coins at rest stays at rest. Reference: friction that holds does not creep, and solids do
    not pass through each other.

    The tower of the coins scene without its tray: 12 coins a layer, 38 layers, at the scene's step and budget. Without a
    position-level stick the coins creep at any budget (78, 8 and 4 mm in 2 s at 24, 48 and 96 sweeps), which only delays
    what follows: pairs lose their support and pass through each other. As their heights cross, the contact turns from
    axial to radial and its depth jumps to the whole radial overlap, about 15 mm, and one Baumgarte row sends both coins
    off at up to 1.9 m/s for a few substeps. A one-contact region with a stick would hold it; M5 as drafted, a neutral
    move, does not reach it.
    """

    LAYERS, RING, RING_RADIUS, RADIUS, THICKNESS, CLEARANCE = 38, 12, 0.052, 0.0125, 0.0028, 0.0003
    STEP = 1.0 / 600.0

    def _wall(self, beta):
        """The wall after 2 s: the largest radial creep [m] and the most pairs that had crossed at any check."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.4, gap=self.CLEARANCE)
        for layer in range(self.LAYERS):
            for slot in range(self.RING):
                angle = 2.0 * math.pi * (slot + 0.5 * (layer % 2)) / self.RING
                z = self.CLEARANCE + self.THICKNESS / 2.0 + layer * (self.THICKNESS + self.CLEARANCE)
                position = (self.RING_RADIUS * math.cos(angle), self.RING_RADIUS * math.sin(angle), z)
                body = builder.add_body(xform=wp.transform(position, wp.quat_identity()))
                builder.add_shape_cylinder(body, radius=self.RADIUS, half_height=self.THICKNESS / 2.0, cfg=cfg)
        builder.add_ground_plane(cfg=cfg)
        model = builder.finalize(device=DEVICE)
        solver = SolverFeatherPGS(
            model,
            pgs_iterations=24,
            friction_anchor_beta=beta,
            dense_max_constraints=1,
            mf_max_constraints=64 * model.body_count,
        )
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="sap",
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
            rigid_contact_max=48 * model.body_count,
        )
        contacts = pipeline.contacts()
        state, out = model.state(), model.state()
        control = model.control()
        crossed = 0
        for step in range(round(2.0 / self.STEP)):
            state.clear_forces()
            pipeline.collide(state, contacts, dt=self.STEP)
            solver.step(state, out, control, contacts, self.STEP)
            state, out = out, state
            if step % 30 == 29:
                crossed = max(crossed, self._crossed(state.body_q.numpy()))
        q = state.body_q.numpy()
        return float(np.abs(np.linalg.norm(q[:, :2], axis=1) - self.RING_RADIUS).max()), crossed

    def _crossed(self, q):
        """Pairs overlapping in plan by over a millimetre whose heights are within half a coin's thickness."""
        apart = np.linalg.norm(q[:, None, :2] - q[None, :, :2], axis=-1)
        heights = np.abs(q[:, None, 2] - q[None, :, 2])
        return int(np.triu((apart < 2.0 * self.RADIUS - 1.0e-3) & (heights < self.THICKNESS / 2.0), 1).sum())

    def test_stick_holds_the_wall(self):
        """Patch friction's position-level stick holds the wall at the scene's budget."""
        creep, crossed = self._wall(0.2)
        self.assertLess(creep, 1.0e-3)
        self.assertEqual(crossed, 0)


class TestQ12Chain(unittest.TestCase):
    """Q12: a chain's factor in every configuration, not only the one it was built at. The walk's pivots stay positive
    and a step answers a tip wrench within the deep-chain regime's bar (1e-4) of the float64 walk at the same float32
    forward kinematics. No floor is asserted: with no armature added a pivot is the chain's own (a straight 100-link
    chain's smallest is 3.5e-6 of its row), and the walk factors it in float32. The forward kinematics' own float32
    floor is not the factor's: on the closed 100-link coil (every joint at the same 2 pi / 100) its frames alone put the
    float64 walk 1.2e-4 off the float64 frames' answer."""

    @staticmethod
    def _chain(links):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        previous = -1
        joints = []
        half = 0.025
        for index in range(links):
            link = builder.add_link(mass=0.1, inertia=wp.mat33(np.diag((2.0e-5, 2.0e-5, 2.0e-5))), lock_inertia=True)
            xform = wp.transform((0.0, 0.0, 2.0 if index == 0 else -2.0 * half), wp.quat_identity())
            joint = builder.add_joint_revolute(
                parent=previous,
                child=link,
                axis=(0.0, 1.0, 0.0),
                parent_xform=xform,
                child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-1.0e6,
                limit_upper=1.0e6,
            )
            joints.append(joint)
            previous = link
        builder.add_articulation(joints)
        return builder.finalize(device=DEVICE)

    def _step(self, links, angles):
        """One step from rest at ``angles``, no gravity, under a wrench on the tip body's centre of mass: the walk's
        smallest pivot (``ell^2`` over its row of ``L L^T``) and the step's ``dv`` against the float64 walk at Newton's
        float32 forward kinematics, relative."""
        model = self._chain(links)
        model.set_gravity((0.0, 0.0, 0.0))
        solver = SolverFeatherPGS(model, enable_joint_limits=False, **BASE)
        state, out = model.state(), model.state()
        state.joint_q.assign(angles)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        tip = links - 1
        body = int(model.joint_child.numpy()[tip])
        force, torque = np.array([0.3, -0.2, 0.5]), np.array([0.0, 0.0, 0.01])
        wrench = np.zeros((model.body_count, 6), dtype=np.float32)
        wrench[body] = np.r_[force, torque]
        state.body_f.assign(wrench)
        solver.step(state, out, model.control(), None, H)
        dv = (out.joint_qd.numpy().astype(np.float64) - state.joint_qd.numpy()) / H
        L = np.linalg.inv(walk_inverse(solver, 0))
        pivots = np.diag(L) ** 2 / np.diag(L @ L.T)
        # The float64 walk at the step's float32 poses, the wrench at the tip's frame (its joint's anchor, world axes).
        joint_q = state.joint_q.numpy().astype(np.float64)
        poses = newton_state_body_q(model, joint_q)
        centre = transform_point(poses[body], model.body_com.numpy()[body].astype(np.float64))
        parent, anchor = int(model.joint_parent.numpy()[tip]), model.joint_X_p.numpy()[tip, :3].astype(np.float64)
        if parent >= 0:
            anchor = transform_point(poses[parent], anchor)
        truth = TreeWalk(*local_frames(model, joint_q, 0, poses), np.zeros(int(model.joint_dof_count)))
        dv64, _ = truth.down(truth.up(tip, np.r_[force, torque + np.cross(centre - anchor, force)])[0])
        return float(pivots.min()), relative(dv, dv64)

    def test_pivots_and_the_answer_hold_in_every_configuration(self):
        """Straight, coiled, folded and random, a chain's walk keeps every pivot positive and answers within 1e-4."""
        rng = np.random.default_rng(7)
        for links in (1, 8, 32, 64, 100):
            shapes = {
                "straight": np.zeros(links),
                "coil": np.full(links, 2.0 * math.pi / max(links, 2)),
                "fold": np.where(np.arange(links) % 2 == 0, 2.5, -2.5),
                "random": rng.uniform(-math.pi, math.pi, links),
            }
            for name, angles in shapes.items():
                with self.subTest(links=links, shape=name):
                    smallest, error = self._step(links, angles)
                    self.assertGreater(smallest, 0.0, f"{name} {links}: pivot {smallest:.3e}")
                    self.assertLess(error, 1.0e-4, f"{name} {links}: dv {error:.2e} off float64")


class TestQ13Indeterminate(unittest.TestCase):
    """Q13: a plate on four corners. The split is indeterminate: only ``v+`` and the total force are physics."""

    def test_rest_and_total_force(self):
        model = _free_box(half=(0.2, 0.1, 0.02), gap=0.005)
        solver, state = _roll(model, 240)
        v = state.body_qd.numpy()[0]
        self.assertLess(float(np.abs(v).max()), 1.0e-3)
        count = int(solver.placement.mf_constraint_count.numpy()[0])
        normal = solver.placement.row_type.numpy()[:, solver.placement.body_base :][0, :count] == 0
        load = float(np.sum(solver.placement.impulses.numpy()[:, solver.placement.body_base :][0, :count][normal]))
        weight = float(model.body_mass.numpy()[0]) * G0 * H
        self.assertAlmostEqual(load, weight, delta=1.0e-3 * weight)


class TestQ14Closure(unittest.TestCase):
    """Q14: a loop and a gear hold their closure. Reference: zero."""

    CPU = conformance.REFERENCE_CPU

    def test_four_bar(self):
        scene = scenes.four_bar()
        run = conformance.simulate(scene, self.CPU, DEVICE, options={"pgs_iterations": 64})
        _data, joints, _mass = conformance.model_data(scene)
        self.assertLess(conformance.loop_gap(run, joints), 1.0e-3)

    def test_mimic(self):
        scene = scenes.mimic_pair()
        run = conformance.simulate(scene, self.CPU, DEVICE, options={"pgs_iterations": 64})
        _data, joints, _mass = conformance.model_data(scene)
        self.assertLess(conformance.mimic_error(run, joints), 1.0e-3)


class TestQ15Speculative(unittest.TestCase):
    """Q15: a fast sphere meets a thin wall. Speculative rows stop it at the surface, in the step it arrives."""

    def test_no_tunneling_at_any_speed(self):
        thickness = 0.02
        radius = 0.05
        for ratio in (0.01, 0.1, 0.5, 1.0, 2.0):
            with self.subTest(ratio=ratio):
                speed = ratio * thickness / H
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0, restitution=0.0, gap=2.0 * speed * H)
                body = builder.add_body(xform=wp.transform((-0.3, 0.0, 0.0), wp.quat_identity()))
                builder.add_shape_sphere(body, radius=radius, cfg=cfg)
                wall = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, restitution=0.0, gap=2.0 * speed * H)
                builder.add_shape_box(
                    -1,
                    xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                    hx=thickness / 2.0,
                    hy=1.0,
                    hz=1.0,
                    cfg=wall,
                )
                model = builder.finalize(device=DEVICE)
                steps = round(0.4 / speed / H) + 4
                _solver, state = _roll(model, steps, setup=lambda state, speed=speed: _set_qd(state, {0: speed}))
                surface = -thickness / 2.0 - radius
                x = float(state.body_q.numpy()[0, 0])
                self.assertAlmostEqual(x, surface, delta=1.0e-3)
                self.assertLess(abs(float(state.joint_qd.numpy()[0])), 1.0e-3)


class TestQ16Limits(unittest.TestCase):
    """Q16: joint limits and velocity limits are complementarity rows. Reference: the audit's zero, and the bound."""

    def test_limits_are_complementary(self):
        """Two coupled hinges pressed against their limits: rows complementary, joints on the limits."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        first = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
        second = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.005))
        joints = [
            builder.add_joint_revolute(
                parent=-1,
                child=first,
                axis=(0.0, 1.0, 0.0),
                child_xform=wp.transform((-0.2, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-0.3,
                limit_upper=0.3,
            ),
            builder.add_joint_revolute(
                parent=first,
                child=second,
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform((0.2, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform((-0.2, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            ),
        ]
        builder.add_articulation(joints)
        model = builder.finalize(device=DEVICE)
        solver, state = _roll(
            model,
            480,
            contacts=False,
            enable_joint_limits=True,
            pgs_iterations=128,
            diagnostics=SolverFeatherPGS.Diagnostics(audit=True),
        )
        audit = solver.read_audit()
        self.assertLess(abs(float(audit["gap"][0])), 1.0e-5)
        self.assertLess(float(audit["violation"][0]), 1.0e-3)
        np.testing.assert_allclose(state.joint_qd.numpy(), [0.0, 0.0], atol=1.0e-3)
        np.testing.assert_allclose(state.joint_q.numpy(), [0.3, 0.2], atol=2.0e-3)

    def test_velocity_limit_is_the_bound(self):
        """A pushed hinge stops accelerating at its velocity limit, and its rows are complementary."""
        scene = scenes.pushed_arm()
        model = scene.build(DEVICE)
        peak = 0.0
        control = model.control()
        options = {
            **BASE,
            **scene.solver_options,
            "enable_joint_velocity_limits": True,
            "diagnostics": SolverFeatherPGS.Diagnostics(audit=True),
        }
        solver = SolverFeatherPGS(model, **options)
        state, out = model.state(), model.state()
        for step in range(scene.steps):
            state.clear_forces()
            scene.control(model, control, state, step)
            solver.step(state, out, control, None, scene.dt)
            state, out = out, state
            peak = max(peak, float(np.abs(state.joint_qd.numpy()).max()))
        self.assertLessEqual(peak, scene.params["velocity_limit"] * 1.001)
        self.assertLess(float(solver.read_audit()["violation_max"][0]), 1.0e-3)


class TestQ17TimeOrder(unittest.TestCase):
    """Q17: on smooth motion the integrator is first order. Reference: float64 RK4 of the point-mass chain."""

    LENGTH, SECONDS = 0.5, 0.4

    def _angles(self, links, start, h):
        """A chain of 1 kg point masses at the links' ends, stepped ``h``; absolute angles at the end."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        previous = -1
        joints = []
        for _link in range(links):
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 1.0e-7), lock_inertia=True)
            joints.append(
                builder.add_joint_revolute(
                    parent=previous,
                    child=link,
                    axis=(0.0, 1.0, 0.0),
                    child_xform=wp.transform((0.0, 0.0, self.LENGTH), wp.quat_identity()),
                    limit_lower=-1.0e6,
                    limit_upper=1.0e6,
                )
            )
            previous = link
        builder.add_articulation(joints)
        model = builder.finalize(device=DEVICE)
        relative = [start[0]] + [b - a for a, b in itertools.pairwise(start)]
        model.joint_q.assign(relative)
        solver = SolverFeatherPGS(model, enable_joint_limits=False, **BASE)
        state, out = model.state(), model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        for _step in range(round(self.SECONDS / h)):
            solver.step(state, out, control, None, h)
            state, out = out, state
        return np.cumsum(state.joint_q.numpy().astype(np.float64))

    def test_error_halves_with_the_step(self):
        """The error of a pendulum and of a double pendulum against RK4 falls as ``h``."""
        for start in ((0.6,), (0.6, 0.9)):
            with self.subTest(links=len(start)):
                exact = refs.pendulum_rollout(
                    [self.LENGTH] * len(start), [1.0] * len(start), list(start), t_end=self.SECONDS
                )
                errors = [
                    float(np.linalg.norm(self._angles(len(start), start, 1.0 / rate) - exact))
                    for rate in (240, 480, 960)
                ]
                orders = [math.log2(errors[0] / errors[1]), math.log2(errors[1] / errors[2])]
                self.assertGreater(min(orders), 0.8, errors)
                self.assertLess(max(orders), 1.25, errors)


def _tangent_basis(n):
    """The tangent pair every FeatherPGS friction row uses for the normal ``n`` (``contact_tangent_basis``)."""
    t0 = np.cross(n, (1.0, 0.0, 0.0))
    if t0 @ t0 < 1.0e-12:
        t0 = np.cross(n, (0.0, 1.0, 0.0))
    t0 /= np.linalg.norm(t0)
    t1 = np.cross(n, t0)
    return t0, t1 / np.linalg.norm(t1)


@dataclasses.dataclass(frozen=True)
class FrictionUnit:
    """The friction rows the law writes: a region's anchor.

    ``carrier`` is the contact that holds the rows (its normal gives the tangent pair), ``members`` the contacts whose
    normal impulses pool into the unit's capacity, ``share`` its part of that pool, ``point`` [m] where its tangent
    rows act, ``None`` for the carrier's own contact point, and ``arm`` [m] its region's spin arm: a region with a
    footprint (a positive arm) holds a spin row of coefficient ``mu arm``."""

    carrier: int
    members: tuple
    share: float = 1.0
    point: tuple | None = None
    arm: float = 0.0


def patch_units(solver, count):
    """The law's units on the step ``solver`` took (its patch view): each region's anchor at its friction
    location, on its region's pooled normal impulse, with its region's spin arm."""
    view = solver.laws.friction_patches.view
    weight, ring = view.weight.numpy()[:count], view.next_contact.numpy()[:count]
    location = view.point.numpy()[:count].astype(np.float64)
    arm = view.spin_arm.numpy()[:count].astype(np.float64)
    units = []
    for c in np.flatnonzero(weight > 0.0):
        members, m = [int(c)], ring[c]
        while m >= 0 and m != c:
            members.append(int(m))
            m = ring[m]
        units.append(FrictionUnit(int(c), tuple(members), float(weight[c]), tuple(location[c]), float(arm[c])))
    return units


class Manufactured:
    """Q18: a contact problem with a known solution, on a real step's Jacobian and mass matrix.

    The solution is chosen first, so it exists for any ``J`` (tangent rows of coplanar contacts have rank 3, and a
    target chosen per row need not be reachable):

    1. a velocity ``v*``, a status per contact (separated or loaded) and per friction unit (sticking or sliding); a
       sticking unit's rows are projected out of ``v*``;
    2. each loaded contact's normal load, and each unit's friction inside its cone, or on the cone's edge against its
       slip when it slides;
    3. the predictor ``v_hat = v* - M^-1 J^T lambda*`` and each normal row's ``b``, which makes ``v*`` satisfy every row.

    The rows are the law's (:mod:`newton._src.solvers.feather_pgs.contact_laws`). A contact's normal row acts at its one
    contact point, the midpoint of its surface points. A friction unit (:class:`FrictionUnit`) is a tangent pair, with
    contact torsion or a region's footprint also a spin row ``n . (w_a - w_b)``, on the elliptic cone ``|(t1 / mu, t2 /
    mu, s / mu_t)| <= share load``, and slides with maximal dissipation, ``f = -share load D^2 u / |D u|`` for ``D =
    diag(mu, mu, mu_t)``:

    one unit per region, at its friction location, on its region's pooled load, with ``mu_t = mu arm``
    (:func:`patch_units`). Regions and locations are the law's geometry and are read from the step; the rows on
      them, the pooled cone and the solution are the reference's own.

    ``J`` and ``M`` come from Newton's public :func:`newton.eval_jacobian` and :func:`newton.eval_mass_matrix`, ``b``
    reaches the rows as a contact margin (the row law's gap), and ``v_hat`` as the step's initial velocity, so every
    placement's own writers and producers run. Any placement must then return ``v*``.

    The margins move both surface points of a contact by the same distance, so its contact point and every friction
    location stay where the collision put them: the rows do not depend on ``b``, and the generalized force's lever arms
    are read once.

    A row holds ``b`` as a float32 gap over ``dt``, so its value sits up to about two float32 steps of a contact point
    from the stated one: 2.4e-5 m/s at 1/240 s, and a sliding pattern can carry that 4.45x into a contact's speed. The
    error is therefore read against the solution of the held rows (:meth:`held_offsets`, :meth:`response`), with the
    offsets bounded by the test so that no wrong ``b`` passes as rounding. That reading needs a pattern whose
    linearization is regular (:meth:`conditioning`); any other is judged on what every solution holds (:meth:`residual`).
    """

    SEPARATED, STICKING, SLIDING = 0, 1, 2

    def __init__(self, model, state, contacts, dt, rng, *, units, beta=0.2, spin=False, loaded=2.0 / 3.0):
        self.model, self.state, self.contacts, self.dt, self.beta, self.spin = model, state, contacts, dt, beta, spin
        self.count = int(contacts.rigid_contact_count.numpy()[0])
        self._read_model()
        self._read_contacts()
        # Loaded with probability ``loaded``, then sticking or sliding alike.
        self.status = np.where(rng.uniform(size=self.count) < loaded, rng.integers(1, 3, self.count), self.SEPARATED)
        self.draw_velocity = rng.normal(size=self.dofs) * 0.02
        self.draw_load = rng.uniform(0.3, 1.0, self.count)
        self.draw_tangent = rng.normal(size=(self.count, 3))
        self.draw_gap = rng.uniform(0.01, 0.05, self.count)
        # A unit whose carrier is separated still holds its region's pooled load: it slides or sticks by this draw.
        self.draw_slides = rng.uniform(size=self.count) < 0.5
        self.units = list(units)
        # The second pass reads the gaps the first pass's margins hold.
        for _ in range(2):
            self._solve()

    def _read_model(self):
        model = self.model
        joints_of = model.articulation_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        child = model.joint_child.numpy()
        self.dofs = int(model.joint_dof_count)
        self.body_at = {}
        self.first = np.zeros(model.articulation_count, dtype=int)
        self.size = np.zeros(model.articulation_count, dtype=int)
        for art in range(model.articulation_count):
            joints = range(joints_of[art], joints_of[art + 1])
            self.first[art] = qd_start[joints[0]]
            self.size[art] = qd_start[joints[-1] + 1] - qd_start[joints[0]]
            for k, joint in enumerate(joints):
                self.body_at[int(child[joint])] = (art, k)
        self.jacobian = newton.eval_jacobian(model, self.state).numpy().astype(np.float64)
        H = newton.eval_mass_matrix(model, self.state).numpy().astype(np.float64)
        self.M = np.zeros((self.dofs, self.dofs))
        for art in range(model.articulation_count):
            first, size = self.first[art], self.size[art]
            self.M[first : first + size, first : first + size] = H[art, :size, :size]
        stiffness = model.joint_target_ke.numpy().astype(np.float64) * self.dt * self.dt
        stiffness += model.joint_target_kd.numpy().astype(np.float64) * self.dt
        self.stiffness = stiffness
        self.M[np.diag_indices(self.dofs)] += model.joint_armature.numpy() + stiffness
        self.body_q = self.state.body_q.numpy().astype(np.float64)
        self.body_com = model.body_com.numpy().astype(np.float64)

    def _read_contacts(self):
        model, contacts, c = self.model, self.contacts, self.count
        shape_body = model.shape_body.numpy()
        mu = model.shape_material_mu.numpy().astype(np.float64)
        mu_t = model.shape_material_mu_torsional.numpy().astype(np.float64)
        shapes = contacts.rigid_contact_shape0.numpy()[:c], contacts.rigid_contact_shape1.numpy()[:c]
        self.body_a, self.body_b = shape_body[shapes[0]], shape_body[shapes[1]]
        self.mu = 0.5 * (mu[shapes[0]] + mu[shapes[1]])
        self.mu_t = 0.5 * (mu_t[shapes[0]] + mu_t[shapes[1]])
        self.normal = -contacts.rigid_contact_normal.numpy()[:c].astype(np.float64)
        self.point0 = contacts.rigid_contact_point0.numpy()[:c].astype(np.float64)
        self.point1 = contacts.rigid_contact_point1.numpy()[:c].astype(np.float64)
        self.margin0 = contacts.rigid_contact_margin0.numpy()[:c].astype(np.float64)
        self.margin1 = contacts.rigid_contact_margin1.numpy()[:c].astype(np.float64)

    def _world(self, body, point):
        """A point of ``body``'s frame in the world (a static shape's points are world points)."""
        if body < 0:
            return point
        pose = self.body_q[body]
        return pose[:3] + conformance._quat_rotate(pose[3:7], point)

    def _block(self, body):
        """``body``'s Jacobian block: its centre of mass's velocity over its angular velocity, 6 by its tree's size."""
        art, k = self.body_at[int(body)]
        return self.jacobian[art, 6 * k : 6 * k + 6, : self.size[art]], self.first[art], self.size[art]

    def _point_rows(self, body, point):
        """``J_p``: the 3 rows of the world velocity of the world point ``point`` of ``body``."""
        rows = np.zeros((3, self.dofs))
        if body < 0:
            return rows
        block, first, size = self._block(body)
        arm = point - self._world(body, self.body_com[body])
        cross = np.array([[0.0, -arm[2], arm[1]], [arm[2], 0.0, -arm[0]], [-arm[1], arm[0], 0.0]])
        rows[:, first : first + size] = block[:3] - cross @ block[3:]
        return rows

    def _spin_rows(self, body):
        """The 3 rows of ``body``'s angular velocity."""
        rows = np.zeros((3, self.dofs))
        if body >= 0:
            block, first, size = self._block(body)
            rows[:, first : first + size] = block[3:]
        return rows

    def _geometry(self):
        """Each contact's relative-velocity rows ``J`` (3 by dofs, A less B at its contact point), its gap ``phi`` and
        its contact point."""
        relative = np.zeros((self.count, 3, self.dofs))
        phi = np.zeros(self.count)
        self.points = np.zeros((self.count, 3))
        for c in range(self.count):
            n = self.normal[c]
            p_a = self._world(self.body_a[c], self.point0[c]) - self.margin0[c] * n
            p_b = self._world(self.body_b[c], self.point1[c]) + self.margin1[c] * n
            phi[c] = n @ (p_a - p_b)
            self.points[c] = 0.5 * (p_a + p_b)
            relative[c] = self._point_rows(self.body_a[c], self.points[c]) - self._point_rows(
                self.body_b[c], self.points[c]
            )
        return relative, phi

    def _unit_rows(self, unit):
        """A unit's rows: its tangent pair at its point, and its spin row where its region has a footprint or under
        contact torsion."""
        c = unit.carrier
        n = self.normal[c]
        t0, t1 = _tangent_basis(n)
        point = self.points[c] if unit.point is None else np.asarray(unit.point, dtype=np.float64)
        relative = self._point_rows(self.body_a[c], point) - self._point_rows(self.body_b[c], point)
        rows = [t0 @ relative, t1 @ relative]
        if self.spin or unit.arm > 0.0:
            rows.append(n @ (self._spin_rows(self.body_a[c]) - self._spin_rows(self.body_b[c])))
        return np.stack(rows)

    def _scale(self, k):
        """Unit ``k``'s cone scale ``D``: ``mu`` on the tangent pair, ``mu_t`` on the spin row: the larger of the
        region's ``mu arm`` and, under contact torsion, the authored torsional coefficient."""
        unit = self.units[k]
        c = unit.carrier
        mu_t = max(self.mu_t[c] if self.spin else 0.0, self.mu[c] * unit.arm)
        return np.array([self.mu[c], self.mu[c], mu_t][: len(self.rows[k])])

    def _slides(self, k):
        """Whether unit ``k`` slides: its carrier's status, or its own draw where the carrier is separated."""
        carrier = self.units[k].carrier
        if self.status[carrier] == self.SEPARATED:
            return bool(self.draw_slides[carrier])
        return self.status[carrier] == self.SLIDING

    def _sliding(self, k):
        """Whether unit ``k`` slides with a slip the law can read: on the cone's edge against it."""
        weighted = self._scale(k) * (self.rows[k] @ self.v_star)
        return self.capacity[k] > 0.0 and self._slides(k) and np.linalg.norm(weighted) > 1.0e-6 * self._scale(k)[0]

    def _friction(self, k):
        """Unit ``k``'s friction [N s] in its rows' basis: on the cone's edge against its slip, or inside the cone."""
        if self.capacity[k] <= 0.0:
            return np.zeros(len(self.rows[k]))
        scale = self._scale(k)
        if self._sliding(k):
            weighted = scale * (self.rows[k] @ self.v_star)
            return -self.capacity[k] * scale * weighted / np.linalg.norm(weighted)
        c, draw = self.units[k].carrier, self.draw_tangent[self.units[k].carrier]
        t0, t1 = _tangent_basis(self.normal[c])
        direction = np.array([t0 @ draw, t1 @ draw, self.normal[c] @ draw])[: len(scale)]
        return 0.5 * self.capacity[k] * scale * direction / max(float(np.linalg.norm(direction)), 1.0e-9)

    def _solve(self):
        """State the problem: ``v*``, the forces, ``v_hat`` and the margins that hold ``b``."""
        self.relative, phi = self._geometry()
        self.rows = [self._unit_rows(unit) for unit in self.units]
        n = self.normal
        solve = np.linalg.solve
        self.load = np.zeros(self.count)
        for c in np.flatnonzero(self.status != self.SEPARATED):
            row = n[c] @ self.relative[c]
            self.load[c] = self.draw_load[c] * 0.02 / max(float(row @ solve(self.M, row)), 1.0e-12)
        self.capacity = np.array([unit.share * self.load[list(unit.members)].sum() for unit in self.units])
        self.v_star = self.draw_velocity.copy()
        sticking = [self.rows[k] for k in range(len(self.units)) if self.capacity[k] > 0.0 and not self._slides(k)]
        if sticking:
            A = np.concatenate(sticking)
            self.v_star -= np.linalg.pinv(A, rcond=1.0e-10) @ (A @ self.v_star)
        impulse = np.zeros(self.dofs)
        for c in range(self.count):
            impulse += self.load[c] * (n[c] @ self.relative[c])
        self.friction = [self._friction(k) for k in range(len(self.units))]
        for rows, friction in zip(self.rows, self.friction, strict=True):
            impulse += rows.T @ friction
        self.v_hat = self.v_star - solve(self.M, impulse)
        for c in range(self.count):
            normal_velocity = float(n[c] @ self.relative[c] @ self.v_star)
            target = self.draw_gap[c] if self.status[c] == self.SEPARATED else 0.0
            bias = target - normal_velocity
            gap = bias * self.dt / self.beta if bias <= 0.0 else bias * self.dt
            self.margin0[c] += 0.5 * (phi[c] - gap)
            self.margin1[c] += 0.5 * (phi[c] - gap)

    def apply(self):
        """Write the margins that state the problem into the contacts."""
        for array, values in (
            (self.contacts.rigid_contact_margin0, self.margin0),
            (self.contacts.rigid_contact_margin1, self.margin1),
        ):
            margin = array.numpy()
            margin[: self.count] = values
            array.assign(margin)

    def _held(self, solver):
        """Each contact's row store, world and first row where the audit reads its world, else ``None``; each contact
        must hold the rows the law states, a normal and its unit's."""
        rows = solver.placement
        paths, slots, worlds = rows.contact_path.numpy(), rows.contact_slot.numpy(), rows.contact_world.numpy()
        needed = rows.contact_slots_needed.numpy()
        held = ~np.isnan(solver.read_audit()["gap"])
        stores = [
            (paths[c], worlds[c], slots[c]) if slots[c] >= 0 and paths[c] in (0, 1) and held[worlds[c]] else None
            for c in range(self.count)
        ]
        stated = np.ones(self.count, dtype=int)
        for unit, unit_rows in zip(self.units, self.rows, strict=True):
            stated[unit.carrier] = 1 + len(unit_rows)
        for c in range(self.count):
            if stores[c] is not None and needed[c] != stated[c]:
                raise RuntimeError(f"contact {c} holds {needed[c]} rows, the law states {stated[c]}")
        return stores

    def held_offsets(self, solver):
        """Each loaded contact's held ``b`` less the stated one [m/s], and each unit's held tangent ``b`` (stated 0: a
        new anchor carries no displacement); 0 where a world's rows stay inside its sweep."""
        # Both classes' rows in the one store, a contact's slot its first row in the world's line.
        store = solver.placement.rhs.numpy().astype(np.float64)
        held = self._held(solver)
        offsets = np.zeros(self.count)
        for c in np.flatnonzero(self.status != self.SEPARATED):
            if held[c] is not None:
                _path, world, slot = held[c]
                offsets[c] = store[world, slot] + float(self.normal[c] @ self.relative[c] @ self.v_star)
        tangent = 0.0
        for unit in self.units:
            if held[unit.carrier] is not None:
                _path, world, slot = held[unit.carrier]
                tangent = max(tangent, float(np.abs(store[world, slot + 1 : slot + 3]).max()))
        return offsets, tangent

    def pattern(self):
        """The status pattern linearized at ``v*`` in float64: ``(A, J, mobility, normals)``, with ``A x = target`` for
        the change ``x`` of the loaded normals' impulses (their rows' targets first, the held offsets) and of the loaded
        units' friction (targets 0), ``J`` those unknowns' rows and ``mobility x`` the velocity they make.

        A loaded normal holds its row, a sticking unit its rows' velocity, and a sliding unit its friction on the cone's
        edge against its slip: ``dx + share g dL + load dg du = 0`` with ``g = D^2 u / |D u|`` and ``dL`` the change of
        its pooled load."""
        normals = np.flatnonzero(self.load > 0.0)
        units = [k for k in range(len(self.units)) if self.capacity[k] > 0.0]
        J = [self.normal[c] @ self.relative[c] for c in normals] + [row for k in units for row in self.rows[k]]
        if not J:
            return np.zeros((0, 0)), np.zeros((0, self.dofs)), np.zeros((self.dofs, 0)), normals
        J = np.stack(J)
        mobility = np.linalg.solve(self.M, J.T)
        G = J @ mobility
        A = G.copy()
        index = {int(c): i for i, c in enumerate(normals)}
        at = len(normals)
        for k in units:
            block = slice(at, at + len(self.rows[k]))
            at += len(self.rows[k])
            if not self._sliding(k):
                continue
            scale = self._scale(k)
            weighted = scale * (self.rows[k] @ self.v_star)
            norm = float(np.linalg.norm(weighted))
            g = scale * weighted / norm
            dg = (np.diag(scale * scale) - np.outer(g, g)) / norm
            A[block] = self.capacity[k] * dg @ G[block]
            A[block, block] += np.eye(len(scale))
            for c in self.units[k].members:
                if int(c) in index:
                    A[block, index[int(c)]] += self.units[k].share * g
        return A, J, mobility, normals

    def response(self, offsets):
        """The change in ``v*`` that ``offsets`` make, in float64: the status pattern linearized at ``v*``."""
        A, _J, mobility, normals = self.pattern()
        target = np.zeros(A.shape[0])
        target[: len(normals)] = -offsets[normals]
        return mobility @ np.linalg.lstsq(A, target, rcond=1.0e-10)[0]

    def conditioning(self):
        """The smallest singular value of the linearized pattern with each condition scaled to a unit row: 0 for a
        singular pattern (coplanar loaded normals, redundant sticks), whose impulses the problem leaves open, or one
        without a loaded contact. Scaling the rows makes it independent of their units."""
        A = self.pattern()[0]
        if not A.size:
            return 0.0
        return float(np.linalg.svd(A / np.linalg.norm(A, axis=1, keepdims=True), compute_uv=False)[-1])

    def exposure(self):
        """How far a held ``b`` can move the answer: the largest contact-point speed change per unit offset of the
        loaded normals' ``b`` (the infinity norm of the linearized pattern's map). It bounds a reading where a placement
        keeps its rows inside its sweep, so the held offsets cannot be read and removed."""
        A, _J, mobility, normals = self.pattern()
        if not A.size:
            return 0.0
        moved = self.relative.reshape(-1, self.dofs) @ mobility @ np.linalg.pinv(A, rcond=1.0e-10)[:, : len(normals)]
        return float(np.abs(moved).sum(axis=1).max())

    def residual(self, solver):
        """How far the step's ``v+`` and stored impulses are from solving the stated problem, any of its solutions:
        ``(normal, friction, force)`` [m/s].

        ``normal`` is the natural residual ``|min(d lambda, J v+ + b)|`` of each normal row on the stated ``b`` (``d`` its
        Delassus diagonal). ``friction`` reads each unit on its stated cone: a unit inside it must not move, one on its
        edge must slip against its friction, ``f = -t D^2 u``, and none may leave it. ``force`` is how far the stored
        impulses, through the stated rows, move the contact points from where the step's ``v+`` put them. Never
        ``lambda`` against ``lambda*``: a pattern that leaves it open admits any split."""
        rows = solver.placement
        # Both classes' rows in the one store, a contact's slot its first row in the world's line.
        store = rows.impulses.numpy().astype(np.float64)
        held = self._held(solver)
        if any(h is None for h in held):
            return None
        velocity = solver.v_out.numpy().astype(np.float64)
        load = np.array([store[h[1], h[2]] for h in held])
        impulse = np.zeros(self.dofs)
        normal = 0.0
        for c in range(self.count):
            row = self.normal[c] @ self.relative[c]
            target = self.draw_gap[c] if self.status[c] == self.SEPARATED else 0.0
            speed = float(row @ velocity) + target - float(row @ self.v_star)
            diagonal = float(row @ np.linalg.solve(self.M, row))
            normal = max(normal, abs(min(diagonal * load[c], speed)))
            impulse += load[c] * row
        friction = 0.0
        for k, (unit, unit_rows) in enumerate(zip(self.units, self.rows, strict=True)):
            _path, world, slot = held[unit.carrier]
            force = store[world, slot + 1 : slot + 1 + len(unit_rows)]
            impulse += unit_rows.T @ force
            scale = self._scale(k)
            capacity = unit.share * load[list(unit.members)].sum()
            weighted_force = force / scale
            weighted_slip = scale * (unit_rows @ velocity) / scale[0]
            excess = np.linalg.norm(weighted_force) - capacity
            diagonal = float(np.max(np.diag(unit_rows @ np.linalg.solve(self.M, unit_rows.T))))
            friction = max(friction, diagonal * scale[0] * max(excess, 0.0))
            if capacity <= 1.0e-9 * load.max():
                continue
            if excess < -1.0e-3 * capacity:
                friction = max(friction, float(np.linalg.norm(weighted_slip)))
            else:
                against = -weighted_force / max(np.linalg.norm(weighted_force), 1.0e-30)
                along = float(weighted_slip @ against)
                friction = max(friction, float(np.linalg.norm(weighted_slip - along * against)), -along)
        moved = velocity - solver.v_hat.numpy().astype(np.float64) - np.linalg.solve(self.M, impulse)
        return normal, friction, float(np.abs(self.relative @ moved).max())

    def error(self, velocity, offsets=None):
        """How far ``velocity`` is from the solution of the held rows: the largest contact-point speed error, and in
        coordinates."""
        delta = velocity.astype(np.float64) - self.v_star
        if offsets is not None:
            delta -= self.response(offsets)
        return float(np.abs(self.relative @ delta).max()), float(np.abs(delta).max())


class ManufacturedScene:
    """A scene settled through a route, where Q18 states its problems: one settle serves every seed.

    ``spin`` is contact torsion at :data:`TORSION`, ``loaded`` the probability that a contact is loaded. A step at the settled pose says whether the
    placement keeps its rows (``held``) and gives the patch units: the problem's margins keep every contact point, so
    the solve's step must hold the same locations."""

    def __init__(self, scene, route, device, *, settle, spin=False, loaded=2.0 / 3.0, sweeps=1024):
        self.scene, self.route, self.device, self.spin, self.loaded = scene, route, device, spin, loaded
        self.model = scene.build(device)
        if spin:
            self.model.shape_material_mu_torsional.fill_(TORSION)
        options = {**conformance.BASE_OPTIONS, **scene.solver_options, **route.options}
        options.update(
            {
                "contact_torsion": spin,
                "pgs_iterations": sweeps,
                "diagnostics": SolverFeatherPGS.Diagnostics(audit=True),
            }
        )
        self.options = options
        pipeline_options = {"deterministic": True}
        if scene.rigid_contact_max is not None:
            pipeline_options["rigid_contact_max"] = scene.rigid_contact_max
        self.pipeline = newton.CollisionPipeline(self.model, **pipeline_options)
        self.contacts = self.pipeline.contacts()
        self.control = self.model.control()
        with conformance.route_context(route), wp.ScopedDevice(device):
            solver = SolverFeatherPGS(self.model, **options)
            state, self.out = self.model.state(), self.model.state()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            for step in range(settle):
                state.clear_forces()
                if scene.control is not None:
                    scene.control(self.model, self.control, state, step)
                self.pipeline.collide(state, self.contacts)
                solver.step(state, self.out, self.control, self.contacts, scene.dt)
                state, self.out = self.out, state
            wp.copy(self.out.joint_q, state.joint_q)
            state.joint_qd.zero_()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            self.state = state
            self.pipeline.collide(state, self.contacts)
            probe = self._step(np.zeros(self.model.joint_dof_count))
            self.held = bool(np.isfinite(probe.read_audit()["gap"]).all())
            count = int(self.contacts.rigid_contact_count.numpy()[0])
            self.units = patch_units(probe, count)

    def _step(self, qd):
        """A fresh solver's step from the settled pose at ``qd``, over the current contacts."""
        solver = SolverFeatherPGS(self.model, **self.options)
        start = self.model.state()
        wp.copy(start.joint_q, self.state.joint_q)
        start.joint_qd.assign(qd.astype(np.float32))
        newton.eval_fk(self.model, start.joint_q, start.joint_qd, start)
        solver.step(start, self.out, self.control, self.contacts, self.scene.dt)
        return solver

    def problem(self, seed):
        """The problem of ``seed`` at the settled pose, stated in float64 (no solve)."""
        with wp.ScopedDevice(self.device):
            self.pipeline.collide(self.state, self.contacts)
            return Manufactured(
                self.model,
                self.state,
                self.contacts,
                self.scene.dt,
                np.random.default_rng(seed),
                units=self.units,
                spin=self.spin,
                loaded=self.loaded,
            )

    def seeds(self, count, *, regular=True, scan=None):
        """The first ``count`` seeds whose pattern is regular (conditioning at least :data:`REGULAR`), or with
        ``regular`` false the first that load a contact and are not. Where the rows stay inside the sweep, a regular
        seed must also keep its unread offsets under half the tolerance (:meth:`Manufactured.exposure`, :data:`UNREAD`)."""
        chosen = []
        for seed in range(SCAN if scan is None else scan):
            problem = self.problem(seed)
            good = (problem.conditioning() >= REGULAR) == regular and problem.load.any()
            if good and regular and not self.held:
                good = problem.exposure() * UNREAD < 0.5 * TestQ18Manufactured.TOLERANCE
            if good:
                chosen.append(seed)
            if len(chosen) == count:
                break
        return chosen

    def solve(self, seed):
        """State ``seed``'s problem and solve it through the route: ``(problem, solver)``, with the problem's
        ``predictor_miss``, ``offsets`` (:meth:`Manufactured.held_offsets`), ``tangent_offset`` and ``drift``, how far
        the step put a patch location from the stated one [m]."""
        problem = self.problem(seed)
        qd = np.zeros(problem.dofs)
        # v_hat moves with qd by M^-1 (M - K): a drive's force reads qd. Newton on that map reaches v_hat in a few steps.
        response = problem.M - np.diag(problem.stiffness)
        with conformance.route_context(self.route), wp.ScopedDevice(self.device):
            for _ in range(4):
                problem.apply()
                final = self._step(qd)
                miss = problem.v_hat - final.v_hat.numpy().astype(np.float64)
                qd += np.linalg.solve(response, problem.M @ miss)
            problem.predictor_miss = float(np.abs(miss).max())
            units = patch_units(final, problem.count)
            if [(u.carrier, u.members, u.share) for u in units] != [
                (u.carrier, u.members, u.share) for u in problem.units
            ]:
                raise RuntimeError("the patch regions changed with the margins")
            problem.drift = max(
                (float(np.abs(np.subtract(u.point, v.point)).max()) for u, v in zip(units, problem.units, strict=True)),
                default=0.0,
            )
        problem.offsets, problem.tangent_offset = problem.held_offsets(final)
        return problem, final


REGULAR = 1.0e-2
"""The conditioning (:meth:`Manufactured.conditioning`) of a pattern whose impulses the problem fixes: a seed's error is
read against the held rows' solution only there. Measured on the scenes: singular patterns read below 1e-11, regular ones
from 0.010; between them a legged pattern at 0.0022 has a second solution 0.04 m/s from the stated one, which the step
finds."""
TORSION = 0.1
"""The torsional coefficient ``mu_t`` [m] of every shape under contact torsion. At Newton's default of 5 mm a spin of
0.03 rad/s weighs 1.5% of a 0.02 m/s slip at ``mu = 0.5`` in a unit's cone, and a doubled ``mu_t`` passed the spin test;
at 0.1 m it weighs 30%, and the doubled ``mu_t`` misses by up to 2.2e-3 m/s."""
UNREAD = 2.4e-5
"""The held offset [m/s] a reading cannot remove where a placement keeps its rows inside its sweep: about two float32
steps of a contact point over 1/240 s (:class:`Manufactured`). Legged's regular seed of conditioning 0.043 carries it
20x into a contact's speed, 2.6e-4 m/s on the keyed tier however converged."""
SCAN = 64
"""How many seeds a scene's choice reads: a problem is stated in milliseconds, without a solve."""
SPARSE, SPARSER = 0.25, 0.15
"""Probabilities that a contact is loaded where the default, 2/3, rarely leaves a regular pattern in :data:`SCAN` seeds
(measured: the gripper 0 of 200 under point friction, 21 of 64 at 0.25)."""


def check_regular(case, prepared, seed, *, audit=True):
    """Solve ``prepared``'s regular ``seed`` and read it against the held rows' solution, in ``case``."""
    problem, solver = prepared.solve(seed)
    point, _coordinate = problem.error(solver.v_out.numpy(), problem.offsets)
    bound = TestQ18Manufactured.HELD_B / prepared.scene.dt
    case.assertGreater(problem.count, 0)
    case.assertLess(problem.predictor_miss, 1.0e-5 if prepared.device == "cpu" else 1.0e-4)
    case.assertLess(np.abs(problem.offsets).max(), bound)
    case.assertLess(problem.tangent_offset, bound)
    case.assertLess(problem.drift, TestQ18Manufactured.DRIFT)
    case.assertLess(point, TestQ18Manufactured.TOLERANCE)
    if audit:
        reading = solver.read_audit()
        case.assertLess(float(np.nanmax(reading["gap"])), 1.0e-6)
        case.assertLess(float(np.nanmax(reading["violation"])), 1.0e-4)


def check_regular_scenes(case, entries, route, device, *, spin=False, count=3):
    """Choose ``count`` regular seeds on each scene's own settled pose and check each (:func:`check_regular`).

    An entry is ``(build, settle)`` or ``(build, settle, loaded)``."""
    for build, settle, *loaded in entries:
        scene = build()
        with wp.ScopedDevice(device):
            prepared = ManufacturedScene(
                scene, route, device, settle=settle, spin=spin, loaded=loaded[0] if loaded else 2.0 / 3.0
            )
            seeds = prepared.seeds(count)
        with case.subTest(scene=scene.name, route=route.name, spin=spin):
            case.assertEqual(len(seeds), count, f"{len(seeds)} regular seeds of {SCAN}")
        for seed in seeds:
            with case.subTest(scene=scene.name, route=route.name, spin=spin, seed=seed):
                with wp.ScopedDevice(device):
                    check_regular(case, prepared, seed, audit=device == "cpu")


class TestQ18Manufactured(unittest.TestCase):
    """Q18: every placement returns a known solution, on the CPU's core and on each CUDA route.

    Seeds are chosen by their pattern's conditioning (:meth:`Manufactured.conditioning`), never by number. A singular
    pattern proves nothing about the solve's sensitivity and can admit another solution, so it is read on ``v+`` and the
    total force only, never on ``lambda`` (:meth:`test_irregular_patterns_solve_the_stated_problem`)."""

    SCENES = (
        (scenes.box_drop, 60),
        (scenes.box_stack, 60),
        (scenes.legged, 30),
        (scenes.gripper, 60, SPARSE),
        (scenes.hand, 30),
        (scenes.keyboard, 30),
    )
    """``(build, settle[, loaded])``: the gripper's twelve contacts a world act on nine coordinates, so it needs fewer
    loaded contacts for a regular pattern (:data:`SPARSE`)."""
    SPIN_SCENES = (
        (scenes.box_drop, 60, SPARSE),
        (scenes.legged, 30, SPARSE),
        (scenes.hand, 30, SPARSE),
        (scenes.keyboard, 30, SPARSE),
    )
    """Scenes with sphere feet, fingertips and flat faces, where spin rows carry load; a spin row per unit needs fewer
    loaded contacts for a regular pattern."""
    BOX_SPIN_SCENES = ((scenes.box_drop, 60, SPARSER), (scenes.box_stack, 60, SPARSER))
    """Boxes, whose faces hold a region's footprint: spin rows on every unit need the sparsest loading."""
    IRREGULAR_SCENES = (
        (scenes.box_drop, 60, (True,)),
        (scenes.box_stack, 60, (False,)),
        (scenes.legged, 30, (True,)),
        (scenes.gripper, 60, (False,)),
        (scenes.hand, 30, (False,)),
    )
    """Irregular patterns, with spin rows or without: a second solution where the step finds one (legged, the hand, the
    stack)."""
    TOLERANCE = 1.0e-4
    """A contact point's speed error [m/s]: float32 on velocities of order 0.1 m/s, at a budget that converges."""
    HELD_B = 2.0**-21
    """The largest offset of a held ``b`` over ``1 / dt`` [m]: four float32 steps at 1 m, so a wrong ``b`` fails here."""
    DRIFT = 1.0e-7
    """How far a patch location may still move over the last solve [m]: the reference states the rows the step held."""

    def test_the_core_returns_the_solution(self):
        """Separated, sticking and sliding contacts on free bodies, trees and forests: anchors at their regions'
        friction locations, on shares of pooled loads."""
        check_regular_scenes(self, self.SCENES, conformance.CPU, DEVICE)

    def test_spin_rows_return_the_solution(self):
        """Contact torsion: each unit's pair and spin row on their elliptic cone."""
        check_regular_scenes(self, self.SPIN_SCENES + self.BOX_SPIN_SCENES, conformance.CPU, DEVICE, spin=True)

    def test_irregular_patterns_solve_the_stated_problem(self):
        """A pattern that is not regular leaves the impulses open (coplanar loaded normals, redundant sticks) and can
        admit a second solution: sliding friction on an open split, or another status. It is judged on what any solution
        must hold, never on ``lambda``: the step's ``v+`` with its stored impulses solves the stated rows, and those
        impulses through the stated rows are the total force that moved it (:meth:`Manufactured.residual`)."""
        for build, settle, spins in self.IRREGULAR_SCENES:
            scene = build()
            for spin in spins:
                with wp.ScopedDevice(DEVICE):
                    prepared = ManufacturedScene(scene, conformance.CPU, DEVICE, settle=settle, spin=spin)
                    seeds = prepared.seeds(2, regular=False)
                with self.subTest(scene=scene.name, spin=spin):
                    self.assertEqual(len(seeds), 2, f"{len(seeds)} irregular seeds of {SCAN}")
                for seed in seeds:
                    with self.subTest(scene=scene.name, spin=spin, seed=seed), wp.ScopedDevice(DEVICE):
                        problem, solver = prepared.solve(seed)
                        self.assertLess(np.abs(problem.offsets).max(), self.HELD_B / scene.dt)
                        self.assertLess(problem.tangent_offset, self.HELD_B / scene.dt)
                        self.assertLess(problem.drift, self.DRIFT)
                        self.assertLess(max(problem.residual(solver)), self.TOLERANCE)


@unittest.skipUnless(wp.is_cuda_available(), "needs CUDA")
class TestQ18ManufacturedCUDA(unittest.TestCase):
    """Q18 on every CUDA placement: each must return the solution the core returns, on seeds its own settled pose makes
    regular. The Jacobi tiers are read apart (:meth:`test_tiers_return_the_solution`)."""

    PLACEMENTS = (
        (scenes.box_stack, 60, 2.0 / 3.0, ("MF_HELD", "MF_DEFAULT")),
        (scenes.legged, 30, 2.0 / 3.0, ("MF_HELD", "SPARSE")),
        (scenes.gripper, 60, SPARSE, ("MF_DEFAULT", "MF_FOREST")),
        (scenes.keyboard, 30, 2.0 / 3.0, ("SPARSE_DIAGONAL", "SPARSE_DIAGONAL_TREE")),
        (scenes.hand, 30, 2.0 / 3.0, ("PAIRED", "PAIRED_FOREST")),
    )
    """``(build, settle, loaded, routes)``, as :attr:`TestQ18Manufactured.SCENES`."""
    TIERS = ("MF_PARALLEL", "KEYED_PARALLEL")
    """The Jacobi tiers (K4), on legged."""

    def _placements(self, spin, count):
        for build, settle, loaded, names in self.PLACEMENTS:
            entry = (build, settle, min(loaded, SPARSE) if spin else loaded)
            for name in names:
                check_regular_scenes(self, (entry,), getattr(conformance, name), "cuda:0", spin=spin, count=count)

    def test_every_placement_returns_the_solution(self):
        self._placements(False, 3)

    def test_every_placement_holds_spin_rows(self):
        self._placements(True, 1)

    @known_failure(
        "E2: the Jacobi tiers' momentum on a sliding Coulomb face (K4)",
        "seed 34 cycles: a sliding pair's radius moves with its normal's impulse, which makes the tier's linearization "
        "non-normal, and Nesterov's momentum loses its slow complex pair (0.9865 +- 0.0112i) above beta 0.92, which "
        "its schedule passes about 37 sweeps after each restart; Gauss-Seidel returns it within 3.1e-7",
    )
    def test_tiers_return_the_solution(self):
        """At their default budget and exit law, as users run them, the tiers return the solution on legged too.

        Legged's worlds hold sixteen joint-limit rows far from their limits beside their contacts. The tiers leave such
        idle rows out of every contact's step bound; counted, they cut the contacts' steps about tenfold and seed 1
        stopped short at 6.2e-4 m/s."""
        build, settle, loaded, _names = self.PLACEMENTS[1]
        wrong = {}
        for name in self.TIERS:
            with wp.ScopedDevice("cuda:0"):
                prepared = ManufacturedScene(
                    build(), getattr(conformance, name), "cuda:0", settle=settle, loaded=loaded
                )
                for seed in prepared.seeds(4):
                    problem, solver = prepared.solve(seed)
                    point = problem.error(solver.v_out.numpy(), problem.offsets)[0]
                    if point >= TestQ18Manufactured.TOLERANCE:
                        wrong[name, seed] = point
        self.assertEqual(wrong, {})


FAR = (158.75, 158.75, 0.0)
"""The benchmark's farthest env origin [m]: 16,384 worlds on a centred 128 by 128 grid at 2.5 m spacing put the corner
worlds 224.5 m from the origin, at 158.75 m on each axis, where a float32 coordinate's spacing is 2^-16 m (15.3 um)."""


def _move(builder, offset):
    """Move every world of ``builder`` by ``offset`` [m], as the cloner places a world at its env origin
    (:meth:`newton.ModelBuilder.replicate` with ``xforms``): static shapes, the parent frames of roots that are not free,
    and free roots' coordinates."""
    shift = wp.transform(wp.vec3(*offset), wp.quat_identity())
    for shape, body in enumerate(builder.shape_body):
        if body == -1:
            builder.shape_transform[shape] = wp.transform_multiply(shift, wp.transform(*builder.shape_transform[shape]))
    for joint, parent in enumerate(builder.joint_parent):
        if parent != -1:
            continue
        frame = wp.transform(*builder.joint_X_p[joint])
        if builder.joint_type[joint] == int(newton.JointType.FREE):
            start = builder.joint_q_start[joint]
            local = wp.transform_multiply(wp.transform_multiply(wp.transform_inverse(frame), shift), frame)
            moved = wp.transform_multiply(local, wp.transform(*builder.joint_q[start : start + 7]))
            builder.joint_q[start : start + 7] = [float(x) for x in moved]
        else:
            builder.joint_X_p[joint] = wp.transform_multiply(shift, frame)
    builder.body_q = [wp.transform_multiply(shift, wp.transform(*q)) for q in builder.body_q]


def far(scene):
    """``scene`` with every world moved to :data:`FAR`."""
    finalize = scenes._finalize

    def build(device):
        def moved(builder, device):
            _move(builder, FAR)
            return finalize(builder, device)

        with mock.patch.object(scenes, "_finalize", moved):
            return scene.build(device)

    return dataclasses.replace(scene, name=f"{scene.name}_far", build=build)


class TestQ20WorldOffset(unittest.TestCase):
    """Q20: a world's physics does not depend on where it sits. Reference: the same problem at the origin.

    Isaac Lab simulates each world at its env origin (``builder.replicate(..., xforms=env origins)``) and Newton's state
    holds float32 world positions. At the benchmark's farthest origin (:data:`FAR`) a coordinate's spacing is 15.3 um:

    - a contact's gap subtracts its endpoints' translations before their offsets (``contact_laws.contact_points``), so
      it carries no rounding of world points: a row whose normal is horizontal holds the ``b`` it holds at the origin
      (formed from rounded world points, the gripper's pads held one off by 2.2e-3 m/s);
    - a body's step ``v h`` under half the spacing rounds away: at 1/240 s a slide slower than 1.8 mm/s does not move.
      The fix is the owner's: a world simulated in its own frame, with env origins applied at the observation and
      render boundary.
    """

    CREEP = 1.0e-3
    """The slide [m/s]: 4.2 um a step at 1/240 s, under half the spacing at :data:`FAR`."""

    def test_manufactured_problems_far_from_the_origin(self):
        """Q18's problems whose normals are along z return the same solution at :data:`FAR`: all its scenes but the
        gripper."""
        entries = [
            (lambda build=entry[0]: far(build()), *entry[1:])
            for entry in TestQ18Manufactured.SCENES
            if entry[0] is not scenes.gripper
        ]
        check_regular_scenes(self, entries, conformance.CPU, DEVICE, count=2)

    def test_horizontal_normals_far_from_the_origin(self):
        """The gripper's pads, normals across x, hold the stated ``b`` at :data:`FAR` as they do at the origin."""
        with wp.ScopedDevice(DEVICE):
            prepared = ManufacturedScene(far(scenes.gripper()), conformance.CPU, DEVICE, settle=60)
            problem, _solver = prepared.solve(0)
        self.assertLess(np.abs(problem.offsets).max(), TestQ18Manufactured.HELD_B / prepared.scene.dt)

    def _creep(self, origin):
        """A frictionless box sliding at :attr:`CREEP` for 2 s on the ground, its world at ``origin``: how far it went."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G0))
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.0, gap=0.005)
        body = builder.add_body(xform=wp.transform((origin[0], origin[1], 0.05), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        builder.add_ground_plane(cfg=cfg)
        model = builder.finalize(device=DEVICE)
        _solver, state = _roll(model, round(2.0 / H), setup=lambda state: _set_qd(state, {0: self.CREEP}))
        return float(state.body_q.numpy()[0, 0]) - origin[0]

    def test_creep_at_the_origin(self):
        """At the origin the box slides ``2 CREEP``."""
        self.assertAlmostEqual(self._creep((0.0, 0.0)), 2.0 * self.CREEP, delta=0.01 * 2.0 * self.CREEP)

    @known_failure(
        LIMIT,
        "float32 world positions: a step under half the coordinate's spacing rounds away, so a 1 mm/s slide stops past "
        "128 m in a coordinate (the corner world sits at 158.75 m)",
    )
    def test_creep_far_from_the_origin(self):
        """At :data:`FAR` the box slides ``2 CREEP`` too."""
        self.assertAlmostEqual(self._creep(FAR), 2.0 * self.CREEP, delta=0.01 * 2.0 * self.CREEP)


class TestRegimeSuite(unittest.TestCase):
    """The suite's own rule: every known failure names its phase and its reason."""

    def test_known_failures_name_their_phase(self):
        marked = [
            (cls.__name__, name, getattr(cls, name))
            for cls in globals().values()
            if isinstance(cls, type) and issubclass(cls, unittest.TestCase)
            for name in dir(cls)
            if name.startswith("test_") and hasattr(getattr(cls, name), "known_failure")
        ]
        self.assertTrue(marked)
        for cls, name, test in marked:
            phase, why = test.known_failure
            self.assertTrue(phase and why, f"{cls}.{name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

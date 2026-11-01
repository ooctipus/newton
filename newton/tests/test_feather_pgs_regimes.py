# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The regime suite: FeatherPGS against absolute references, across regimes the tasks do not reach.

Conformance compares the solver with itself at a larger budget, so a wrong law is common-mode and passes. Here each
family Q1-Q19 has a reference that shares nothing with the solver: a closed form, a float64 discrete law, quadrature,
float64 RK4 (:mod:`newton.tests.feather_pgs_references`), or a manufactured solution.

A known failure is an expected failure that names its phase or says it is a documented limit (:func:`known_failure`):
when the phase lands, its test reports an unexpected success and the mark goes. Where a family holds in one regime and
fails in another, the two are separate tests, so what passes stays guarded. Runs on the CPU in about three minutes.
"""

from __future__ import annotations

import functools
import itertools
import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_references as refs
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance

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
        _solver, state = _roll(model, 1, pgs_iterations=24, friction_anchor_beta=0.0)
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

        _roll(model, 120, each=each, friction_anchor_beta=0.0, **options)
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

        _solver, state = _roll(model, 60, setup=setup, friction_anchor_beta=0.0, restitution_velocity_threshold=0.01)
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

    @known_failure("PH4", "two anchors at the footprint's extremes hold 1.85x a flat patch's spin capacity")
    def test_spin_capacity(self):
        """A spinning patch sweeps the yaw quadrature gives."""
        wrong = {}
        for half_x, half_y in ((0.05, 0.05), (0.15, 0.05)):
            for speed, spin in ((0.0, 30.0), (0.3, 30.0), (1.0, 10.0)):
                _x, yaw = self._spin_slide(half_x, half_y, speed, spin)
                _distance, expected, _stop = refs.patch_slide_spin(half_x, half_y, speed, spin, 0.5)
                if abs(yaw - expected) > 0.05 * abs(expected):
                    wrong[half_x, speed] = (math.degrees(yaw), math.degrees(expected))
        self.assertEqual(wrong, {})


class TestQ07Rolling(unittest.TestCase):
    """Q7: rolling. A slider ends rolling at ``v0 / (1 + k)``, and a rolling body keeps its speed."""

    def test_sphere_slides_to_five_sevenths(self):
        radius, v0 = 0.05, 1.0
        model = _sphere_world(
            radius, spheres=[(0.0, 0.0, radius)], restitution=0.0, gravity=(0.0, 0.0, -G0), mu=0.3, gap=0.005
        )
        _solver, state = _roll(model, 240, setup=lambda state: _set_qd(state, {0: v0}), friction_anchor_beta=0.0)
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
        _solver, state = _roll(model, 240, setup=lambda state: _set_qd(state, {0: v0}), friction_anchor_beta=0.0)
        self.assertAlmostEqual(float(state.body_qd.numpy()[0, 0]), refs.rolling_speed(v0, 0.5), delta=0.03 * v0)

    def test_rolling_keeps_its_speed(self):
        """A rolling disc has no drag, under either friction law."""
        for beta in (0.0, 0.2):
            with self.subTest(beta=beta):
                model, radius = self._disc()
                v0 = 0.1
                _solver, state = _roll(
                    model,
                    480,
                    setup=lambda state, v0=v0, radius=radius: _set_qd(state, {0: v0, 4: v0 / radius}),
                    friction_anchor_beta=beta,
                )
                self.assertAlmostEqual(float(np.linalg.norm(state.body_qd.numpy()[0, :2])), v0, delta=0.02 * v0)


class TestQ08Tipping(unittest.TestCase):
    """Q8: a force high on a box tips it past ``F h = m g w / 2``, about its far edge, on four corner contacts."""

    WIDTH, HEIGHT, TALL = 0.2, 0.4, 0.2

    def _tilt(self, fraction, beta):
        half = (self.WIDTH / 2.0, self.WIDTH / 2.0, self.TALL)
        model = _free_box(half=half, mu=2.0, gap=0.005, density=1000.0)
        mass = float(model.body_mass.numpy()[0])
        force = fraction * refs.tipping_force(mass, self.WIDTH, self.HEIGHT)
        torque = force * (self.HEIGHT - self.TALL)

        def before(state, step):
            state.body_f.assign([[force, 0.0, 0.0, 0.0, torque, 0.0]])

        _solver, state = _roll(model, 120, before=before, friction_anchor_beta=beta)
        quat = state.body_q.numpy()[0, 3:]
        return 2.0 * math.atan2(float(quat[1]), float(quat[3]))

    def test_threshold(self):
        """It holds at 95% of the force and goes over at 105%, under either friction law."""
        for beta in (0.0, 0.2):
            with self.subTest(beta=beta):
                self.assertLess(abs(self._tilt(0.95, beta)), 2.0e-3)
                self.assertGreater(self._tilt(1.05, beta), 2.0e-2)


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
    """Q10: heavy on light. The rigid claim holds to a mass ratio set by the budget."""

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

        _roll(model, 240, each=each, pgs_iterations=sweeps, friction_anchor_beta=0.0)
        return -worst[0]

    def test_ratio_within_the_budget(self):
        """A ratio up to half the budget holds to a millimetre."""
        self.assertLess(self._overlap(12.0, 24), 1.0e-3)

    def test_overlap_scale(self):
        """Inside the limit the overlap follows ``(ratio / N)^2``, about 0.1 mm at ``ratio = N``."""
        for ratio, sweeps in ((10.0, 24), (30.0, 24), (100.0, 24), (30.0, 96), (100.0, 96)):
            with self.subTest(ratio=ratio, sweeps=sweeps):
                scale = 1.0e-4 * (ratio / sweeps) ** 2
                overlap = self._overlap(ratio, sweeps)
                self.assertGreater(overlap, 0.3 * scale)
                self.assertLess(overlap, 3.0 * scale)

    @known_failure(
        LIMIT, "what N sweeps can do under PGS: past a mass ratio of about 10 N the box ends inside the other"
    )
    def test_ratio_past_the_budget(self):
        """Past ``ratio = 10 N`` the budget cannot carry the load: the heavy box ends inside the light one."""
        self.assertLess(self._overlap(300.0, 24), 5.0e-3)


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
        solver, state = _roll(
            model, 120, pgs_iterations=sweeps, friction_anchor_beta=0.0, mf_max_constraints=12 * count + 12
        )
        mass = float(model.body_mass.numpy()[0])
        counts = int(solver.placement.mf_constraint_count.numpy()[0])
        lam = solver.placement.mf_impulses.numpy()[0, :counts]
        kinds = solver.placement.mf_row_type.numpy()[0, :counts]
        body_b = solver.placement.mf_body_b.numpy()[0, :counts]
        body_a = solver.placement.mf_body_a.numpy()[0, :counts]
        ground = (kinds == 0) & ((body_a < 0) | (body_b < 0))
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

    @known_failure(
        "F: friction, decided with the one-wrench verdict",
        "velocity-level point friction creeps at any budget, until coins pass through each other",
    )
    def test_point_friction_holds_the_wall(self):
        """Point friction holds it too: nothing about the law of a contact at rest permits creep."""
        creep, crossed = self._wall(0.0)
        self.assertLess(creep, 1.0e-3, (creep, crossed))
        self.assertEqual(crossed, 0, (creep, crossed))


class TestQ12Chain(unittest.TestCase):
    """Q12: a chain's factor. The pivot floor holds in every configuration, not only the one it was read at."""

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

    def _floor(self, links, angles):
        """The smallest pivot over its diagonal at ``angles``, against the floor the factor claims. The solver reads its
        floor from the model's straight configuration, as a robot's construction does."""
        model = self._chain(links)
        solver = SolverFeatherPGS(model, enable_joint_limits=False, **BASE)
        state, out = model.state(), model.state()
        state.joint_q.assign(angles)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver.step(state, out, model.control(), None, H)
        L = np.tril(solver.factor.L_by_size[links].numpy()[0].astype(np.float64))
        rows = L @ L.T
        pivots = np.diag(L) ** 2 / np.diag(rows)
        return float(pivots.min()), 4.0 * links * 2.0**-24

    def test_pivots_hold_in_every_configuration(self):
        """Straight, coiled and folded, a chain's smallest pivot stays above the float32 floor."""
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
                    # The floor is ``4 n u``: float32 resolves such a pivot only to about ``n u``, so half is its margin.
                    smallest, floor = self._floor(links, angles)
                    self.assertGreaterEqual(smallest, floor * 0.5, f"{name} {links}: pivot {smallest:.3e}")


class TestQ13Indeterminate(unittest.TestCase):
    """Q13: a plate on four corners. The split is indeterminate: only ``v+`` and the total force are physics."""

    def test_rest_and_total_force(self):
        model = _free_box(half=(0.2, 0.1, 0.02), gap=0.005)
        solver, state = _roll(model, 240, friction_anchor_beta=0.0)
        v = state.body_qd.numpy()[0]
        self.assertLess(float(np.abs(v).max()), 1.0e-3)
        count = int(solver.placement.mf_constraint_count.numpy()[0])
        normal = solver.placement.mf_row_type.numpy()[0, :count] == 0
        load = float(np.sum(solver.placement.mf_impulses.numpy()[0, :count][normal]))
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
                _solver, state = _roll(
                    model, steps, setup=lambda state, speed=speed: _set_qd(state, {0: speed}), friction_anchor_beta=0.0
                )
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
        solver, state = _roll(model, 480, contacts=False, enable_joint_limits=True, pgs_iterations=128, pgs_audit=True)
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
        options = {**BASE, **scene.solver_options, "enable_joint_velocity_limits": True, "pgs_audit": True}
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


class Manufactured:
    """Q18: a contact problem with a known solution, on a real step's Jacobian and mass matrix.

    The solution is chosen first, so it exists for any ``J`` (tangent rows of coplanar contacts have rank 3, and a
    target chosen per row need not be reachable):

    1. a velocity ``v*`` and a status per contact: separated, sticking, or sliding; sticking contacts' tangent velocity
       is projected out of ``v*``;
    2. each loaded contact's force: a normal load and a tangent force inside the cone, or on its edge against
       ``r_t = J_t v*`` when sliding;
    3. the predictor ``v_hat = v* - M^-1 J^T lambda*`` and each normal row's ``b``, which makes ``v*`` satisfy every row.

    ``J`` and ``M`` come from Newton's public :func:`newton.eval_jacobian` and :func:`newton.eval_mass_matrix`, ``b`` reaches
    the rows as a contact margin (the row law's gap), and ``v_hat`` as the step's initial velocity, so every placement's
    own writers and producers run. Any placement must then return ``v*``.

    Lever arms follow the margins, so ``J`` and the problem are iterated to a fixed point; the correction contracts as
    the gap over the bodies' size.

    A row holds ``b`` as a float32 gap over ``dt``, so its value sits up to about two float32 steps of a contact point
    from the stated one: 2.4e-5 m/s at 1/240 s, and a sliding pattern can carry that 4.45x into a contact's speed. The
    error is therefore read against the solution of the held rows (:meth:`held_offsets`, :meth:`response`), with the
    offsets bounded by the test so that no wrong ``b`` passes as rounding.
    """

    SEPARATED, STICKING, SLIDING = 0, 1, 2

    def __init__(self, model, state, contacts, dt, rng, *, beta=0.2):
        self.model, self.state, self.contacts, self.dt, self.beta = model, state, contacts, dt, beta
        self.count = int(contacts.rigid_contact_count.numpy()[0])
        self._read_model()
        self._read_contacts()
        self.status = rng.integers(0, 3, self.count)
        self.draw_velocity = rng.normal(size=self.dofs) * 0.02
        self.draw_load = rng.uniform(0.3, 1.0, self.count)
        self.draw_tangent = rng.normal(size=(self.count, 3))
        self.draw_gap = rng.uniform(0.01, 0.05, self.count)
        for _ in range(5):
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
        shapes = contacts.rigid_contact_shape0.numpy()[:c], contacts.rigid_contact_shape1.numpy()[:c]
        self.body_a, self.body_b = shape_body[shapes[0]], shape_body[shapes[1]]
        self.mu = 0.5 * (mu[shapes[0]] + mu[shapes[1]])
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

    def _point_rows(self, body, point):
        """``J_p``: the 3 rows of the world velocity of the world point ``point`` of ``body``."""
        rows = np.zeros((3, self.dofs))
        if body < 0:
            return rows
        art, k = self.body_at[int(body)]
        first, size = self.first[art], self.size[art]
        block = self.jacobian[art, 6 * k : 6 * k + 6, :size]
        arm = point - self._world(body, self.body_com[body])
        cross = np.array([[0.0, -arm[2], arm[1]], [arm[2], 0.0, -arm[0]], [-arm[1], arm[0], 0.0]])
        rows[:, first : first + size] = block[:3] - cross @ block[3:]
        return rows

    def _geometry(self):
        """Each contact's relative-velocity rows ``J`` (3 by dofs, A's point less B's) and its gap ``phi``."""
        relative = np.zeros((self.count, 3, self.dofs))
        phi = np.zeros(self.count)
        for c in range(self.count):
            n = self.normal[c]
            p_a = self._world(self.body_a[c], self.point0[c]) - self.margin0[c] * n
            p_b = self._world(self.body_b[c], self.point1[c]) + self.margin1[c] * n
            phi[c] = n @ (p_a - p_b)
            relative[c] = self._point_rows(self.body_a[c], p_a) - self._point_rows(self.body_b[c], p_b)
        return relative, phi

    def _solve(self):
        """One pass of the fixed point: ``v*``, the forces, ``v_hat`` and the margins that hold ``b``."""
        relative, phi = self._geometry()
        self.relative = relative
        n = self.normal
        tangent = np.stack([np.eye(3) - np.outer(n[c], n[c]) for c in range(self.count)])
        tangent_rows = np.einsum("cij,cjk->cik", tangent, relative)
        sticking = np.flatnonzero(self.status == self.STICKING)
        self.v_star = self.draw_velocity.copy()
        if sticking.size:
            A = tangent_rows[sticking].reshape(-1, self.dofs)
            self.v_star -= np.linalg.pinv(A, rcond=1.0e-10) @ (A @ self.v_star)
        solve = np.linalg.solve
        impulse = np.zeros(self.dofs)
        self.force = np.zeros((self.count, 3))
        for c in range(self.count):
            if self.status[c] == self.SEPARATED:
                continue
            row = n[c] @ relative[c]
            load = self.draw_load[c] * 0.02 / max(float(row @ solve(self.M, row)), 1.0e-12)
            slip = tangent_rows[c] @ self.v_star
            speed = float(np.linalg.norm(slip))
            if self.status[c] == self.SLIDING and speed > 1.0e-6:
                force = -self.mu[c] * load * slip / speed
            else:
                direction = tangent[c] @ self.draw_tangent[c]
                force = 0.5 * self.mu[c] * load * direction / max(float(np.linalg.norm(direction)), 1.0e-9)
            self.force[c] = load * n[c] + force
            impulse += relative[c].T @ self.force[c]
        self.v_hat = self.v_star - solve(self.M, impulse)
        for c in range(self.count):
            normal_velocity = float(n[c] @ relative[c] @ self.v_star)
            target = self.draw_gap[c] if self.status[c] == self.SEPARATED else 0.0
            bias = target - normal_velocity
            gap = bias * self.dt / self.beta if bias <= 0.0 else bias * self.dt
            self.margin0[c] += phi[c] - gap

    def apply(self):
        """Write the margins that state the problem into the contacts."""
        margin = self.contacts.rigid_contact_margin0.numpy()
        margin[: self.count] = self.margin0
        self.contacts.rigid_contact_margin0.assign(margin)

    def held_offsets(self, solver):
        """Each loaded contact's held ``b`` less the stated one [m/s], 0 where a world's rows stay inside its sweep."""
        rows = solver.placement
        paths, slots, worlds = rows.contact_path.numpy(), rows.contact_slot.numpy(), rows.contact_world.numpy()
        stores = rows.rhs.numpy().astype(np.float64), rows.mf_rhs.numpy().astype(np.float64)
        held = ~np.isnan(solver.read_audit()["gap"])
        offsets = np.zeros(self.count)
        for c in np.flatnonzero(self.status != self.SEPARATED):
            if slots[c] >= 0 and held[worlds[c]]:
                row = self.normal[c] @ self.relative[c]
                offsets[c] = stores[paths[c]][worlds[c], slots[c]] + float(row @ self.v_star)
        return offsets

    def response(self, offsets):
        """The change in ``v*`` that ``offsets`` make, in float64: the status pattern linearized at ``v*``.

        A sticking contact holds its relative velocity, a sliding one its normal speed and its force on the cone against
        the slip. A redundant pattern takes the smallest forces, which leaves the velocity unique."""
        loaded = np.flatnonzero((self.status != self.SEPARATED) & (np.abs(self.force).max(axis=1) > 0.0))
        rows = np.concatenate([self.relative[c] for c in loaded])
        mobility = np.linalg.solve(self.M, rows.T)
        G = rows @ mobility
        A, target = np.zeros_like(G), np.zeros(G.shape[0])
        for index, c in enumerate(loaded):
            at, n = 3 * index, self.normal[c]
            t0 = np.cross(n, (1.0, 0.0, 0.0)) if abs(n[0]) < 0.9 else np.cross(n, (0.0, 1.0, 0.0))
            t0 /= np.linalg.norm(t0)
            t1 = np.cross(n, t0)
            A[at] = n @ G[at : at + 3]
            target[at] = -offsets[c]
            slip = (np.eye(3) - np.outer(n, n)) @ self.relative[c] @ self.v_star
            speed = float(np.linalg.norm(slip))
            for row, t in ((1, t0), (2, t1)):
                if self.status[c] == self.SLIDING and speed > 1.0e-6:
                    across = np.eye(3) - np.outer(slip, slip) / speed**2 - np.outer(n, n)
                    A[at + row, at : at + 3] = t + self.mu[c] * (t @ slip) / speed * n
                    A[at + row] += self.mu[c] * float(n @ self.force[c]) / speed * (t @ across) @ G[at : at + 3]
                else:
                    A[at + row] = t @ G[at : at + 3]
        return mobility @ np.linalg.lstsq(A, target, rcond=1.0e-10)[0]

    def error(self, velocity, offsets=None):
        """How far ``velocity`` is from the solution of the held rows: the largest contact-point speed error, and in
        coordinates."""
        delta = velocity.astype(np.float64) - self.v_star
        if offsets is not None:
            delta -= self.response(offsets)
        return float(np.abs(self.relative @ delta).max()), float(np.abs(delta).max())


def manufactured_run(scene, route, device, *, seed=0, settle=30, sweeps=1024):
    """Settle ``scene``, state a manufactured problem at that pose and solve it through ``route``.

    Returns the problem, the solver (with the audit on) and the error of its velocity."""
    model = scene.build(device)
    options = {**conformance.BASE_OPTIONS, **scene.solver_options, **route.options}
    options.update({"friction_anchor_beta": 0.0, "pgs_iterations": sweeps, "pgs_audit": True})
    pipeline_options = {"deterministic": True}
    if scene.rigid_contact_max is not None:
        pipeline_options["rigid_contact_max"] = scene.rigid_contact_max
    pipeline = newton.CollisionPipeline(model, **pipeline_options)
    contacts = pipeline.contacts()
    control = model.control()
    with conformance.route_context(route):
        solver = SolverFeatherPGS(model, **options)
        state, out = model.state(), model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for step in range(settle):
            state.clear_forces()
            if scene.control is not None:
                scene.control(model, control, state, step)
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, scene.dt)
            state, out = out, state
        wp.copy(out.joint_q, state.joint_q)
        state.joint_qd.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline.collide(state, contacts)
        problem = Manufactured(model, state, contacts, scene.dt, np.random.default_rng(seed))
        problem.apply()
        qd = np.zeros(problem.dofs)
        # v_hat moves with qd by M^-1 (M - K): a drive's force reads qd. Newton on that map reaches v_hat in a few steps.
        response = problem.M - np.diag(problem.stiffness)
        for _ in range(4):
            final = SolverFeatherPGS(model, **options)
            start = model.state()
            wp.copy(start.joint_q, state.joint_q)
            start.joint_qd.assign(qd.astype(np.float32))
            newton.eval_fk(model, start.joint_q, start.joint_qd, start)
            final.step(start, out, control, contacts, scene.dt)
            miss = problem.v_hat - final.v_hat.numpy().astype(np.float64)
            qd += np.linalg.solve(response, problem.M @ miss)
        problem.predictor_miss = float(np.abs(miss).max())
    problem.offsets = problem.held_offsets(final)
    return problem, final, problem.error(final.v_out.numpy(), problem.offsets)


class TestQ18Manufactured(unittest.TestCase):
    """Q18: every placement returns a known solution, on the CPU's core and on each CUDA route."""

    SCENES = (
        (scenes.box_drop, 60),
        (scenes.box_stack, 60),
        (scenes.legged, 30),
        (scenes.gripper, 60),
        (scenes.hand, 30),
        (scenes.keyboard, 30),
    )
    TOLERANCE = 1.0e-4
    """A contact point's speed error [m/s]: float32 on velocities of order 0.1 m/s, at a budget that converges."""
    HELD_B = 2.0**-21
    """The largest offset of a held ``b`` over ``1 / dt`` [m]: four float32 steps at 1 m, so a wrong ``b`` fails here."""

    def test_the_core_returns_the_solution(self):
        """Separated, sticking and sliding contacts on free bodies, trees and forests."""
        for build, settle in self.SCENES:
            scene = build()
            for seed in range(3):
                with self.subTest(scene=scene.name, seed=seed), wp.ScopedDevice(DEVICE):
                    problem, solver, (point, _coordinate) = manufactured_run(
                        scene, conformance.CPU, DEVICE, seed=seed, settle=settle
                    )
                    self.assertGreater(problem.count, 0)
                    self.assertLess(problem.predictor_miss, 1.0e-5)
                    self.assertLess(np.abs(problem.offsets).max(), self.HELD_B / scene.dt)
                    self.assertLess(point, self.TOLERANCE)
                    audit = solver.read_audit()
                    self.assertLess(float(np.nanmax(audit["gap"])), 1.0e-6)
                    self.assertLess(float(np.nanmax(audit["violation"])), 1.0e-4)


@unittest.skipUnless(wp.is_cuda_available(), "needs CUDA")
class TestQ18ManufacturedCUDA(unittest.TestCase):
    """Q18 on every CUDA placement: each must return the solution the core returns."""

    PLACEMENTS = (
        (scenes.box_stack, 60, ("MF_HELD", "MF_DEFAULT")),
        (scenes.legged, 30, ("MF_HELD", "SPARSE", "SPARSE_TRIPLETS", "MF_PARALLEL", "KEYED_PARALLEL")),
        (scenes.gripper, 60, ("LOCAL_PACKETS", "LOCAL_FOREST", "LOCAL_OWNED")),
        (scenes.keyboard, 30, ("SPARSE_DIAGONAL", "SPARSE_DIAGONAL_TREE", "COMPACT_BOUNDARY")),
        (scenes.hand, 30, ("PAIRED", "PAIRED_FOREST")),
    )

    def test_every_placement_returns_the_solution(self):
        for build, settle, names in self.PLACEMENTS:
            scene = build()
            for name in names:
                for seed in range(3):
                    with self.subTest(scene=scene.name, route=name, seed=seed), wp.ScopedDevice("cuda:0"):
                        route = getattr(conformance, name)
                        problem, _solver, (point, _coordinate) = manufactured_run(
                            scene, route, "cuda:0", seed=seed, settle=settle
                        )
                        self.assertGreater(problem.count, 0)
                        self.assertLess(problem.predictor_miss, 1.0e-4)
                        self.assertLess(np.abs(problem.offsets).max(), TestQ18Manufactured.HELD_B / scene.dt)
                        self.assertLess(point, TestQ18Manufactured.TOLERANCE)


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

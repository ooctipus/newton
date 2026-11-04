# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FeatherPGS's repaired laws against closed forms: each test failed on the solver the repair replaced.

Conformance compares routes with their own converged run, so a wrong law is common-mode there. A test here takes its
reference from the law as physics states it, and runs on the row stores or integrators that repair reached.
"""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance
from newton.tests import test_feather_pgs_scalar_branches as branch
from newton.tests import test_feather_pgs_sparse_solver as sparse

H = 1.0 / 240.0
HALF = 0.1
"""Half extent of the test's box [m]."""


def _caps(model, cap: float) -> None:
    """Give every body the depenetration speed ``cap`` [m/s]."""
    model.rigid_body_max_depenetration_velocity = wp.full(model.body_count, cap, dtype=wp.float32, device=model.device)


def _roll(model, steps, *, dt=H, **options):
    """Step ``model`` with collisions from its initial state; return the last state."""
    solver = SolverFeatherPGS(model, **{"angular_damping": 0.0, "pgs_iterations": 64, **options})
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, dt)
        state, out = out, state
    return state


def _overlapping_box(store: str, overlap: float, device="cpu"):
    """A 1 kg-per-liter box overlapping the ground by ``overlap``, gravity off, on the row store ``store``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=0.01)
    pose = wp.transform((0.0, 0.0, HALF - overlap), wp.quat_identity())
    if store == "free":
        body = builder.add_body(xform=pose)
        builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
    elif store == "prismatic":
        body = builder.add_link(xform=pose)
        builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
        joint = builder.add_joint_prismatic(parent=-1, child=body, axis=(0.0, 0.0, 1.0), parent_xform=pose)
        builder.add_articulation([joint])
    else:
        # A free root with a passive tail: its rows live in the dense store.
        body = builder.add_link(xform=pose)
        builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
        tail = builder.add_link(xform=pose, mass=1.0e-3, inertia=wp.mat33(np.eye(3) * 1.0e-6))
        root = builder.add_joint_free(body)
        hinge = builder.add_joint_revolute(parent=body, child=tail, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([root, hinge])
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device)


class TestDepenetrationCap(unittest.TestCase):
    """PH1a: a contact's Baumgarte speed is ``min(beta |phi| / h, cap)`` on every row store."""

    overlap = 0.02
    uncapped = 0.2 * overlap / H

    def _speed(self, store: str, cap: float, *, beta: float = 0.0) -> float:
        model = _overlapping_box(store, self.overlap)
        _caps(model, cap)
        state = _roll(model, 1, friction_anchor_beta=beta)
        return float(state.body_qd.numpy()[0, 2])

    def test_cap_holds_on_every_cpu_store(self):
        for store in ("free", "prismatic", "tree"):
            for beta in (0.0, 0.2):
                with self.subTest(store=store, friction_anchor_beta=beta):
                    self.assertAlmostEqual(self._speed(store, 0.5, beta=beta), 0.5, delta=2.0e-4)

    def test_no_cap_is_the_gap_law(self):
        for store in ("free", "prismatic", "tree"):
            with self.subTest(store=store):
                self.assertAlmostEqual(self._speed(store, float("inf")), self.uncapped, delta=5.0e-3 * self.uncapped)

    def test_a_cap_above_the_bias_does_nothing(self):
        for store in ("free", "prismatic", "tree"):
            with self.subTest(store=store):
                speed = self._speed(store, 2.0 * self.uncapped)
                self.assertAlmostEqual(speed, self.uncapped, delta=5.0e-3 * self.uncapped)


def _pressed_pair(store: str, overlap: float, *, device="cpu"):
    """Two 8 kg boxes overlapping by ``overlap`` along z, sliding oppositely along x, no gravity, on row store ``store``.

    Returns the model and the state's initial ``joint_qd``. The boxes' witness points sit ``overlap`` apart along the
    normal, so a tangent impulse applied at each would put the couple ``overlap * f`` on the pair.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.8, gap=0.02)
    roots = []
    for height in (0.0, 2.0 * HALF - overlap):
        pose = wp.transform((0.0, 0.0, height), wp.quat_identity())
        if store == "free":
            body = builder.add_body(xform=pose)
            builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
        else:
            body = builder.add_link(xform=pose)
            builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
            tail = builder.add_link(xform=pose, mass=1.0e-3, inertia=wp.mat33(np.eye(3) * 1.0e-6))
            root = builder.add_joint_free(body)
            hinge = builder.add_joint_revolute(parent=body, child=tail, axis=(0.0, 0.0, 1.0))
            builder.add_articulation([root, hinge])
        roots.append(body)
    return builder.finalize(device=device)


def _momenta(model, state) -> tuple[np.ndarray, np.ndarray]:
    """The system's linear and angular momentum about the world origin."""
    q, qd = state.body_q.numpy(), state.body_qd.numpy()
    mass, inertia = model.body_mass.numpy(), model.body_inertia.numpy()
    linear, angular = np.zeros(3), np.zeros(3)
    for body in range(model.body_count):
        rotation = np.array(wp.quat_to_matrix(wp.quat(*q[body, 3:]))).reshape(3, 3)
        linear += mass[body] * qd[body, :3]
        angular += np.cross(q[body, :3], mass[body] * qd[body, :3])
        angular += rotation @ np.asarray(inertia[body]) @ rotation.T @ qd[body, 3:]
    return linear, angular


class TestOneContactPoint(unittest.TestCase):
    """PH1b: a contact's rows act at one point, so its impulses are a pure force pair and conserve angular momentum."""

    def _change(self, store: str, overlap: float) -> tuple[float, float]:
        model = _pressed_pair(store, overlap)
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0, angular_damping=0.0, pgs_iterations=64)
        state, out = model.state(), model.state()
        velocity = state.joint_qd.numpy()
        half = model.joint_dof_count // 2
        velocity[0], velocity[half] = 1.0, -1.0
        state.joint_qd.assign(velocity)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        before = _momenta(model, state)
        pipeline.collide(state, contacts)
        self.assertGreater(contacts.rigid_contact_count.numpy()[0], 0)
        solver.step(state, out, model.control(), contacts, H)
        after = _momenta(model, out)
        return float(np.linalg.norm(after[0] - before[0])), float(np.linalg.norm(after[1] - before[1]))

    def test_friction_conserves_angular_momentum(self):
        for store in ("free", "tree"):
            with self.subTest(store=store):
                linear, angular = self._change(store, 0.01)
                self.assertLess(linear, 1.0e-3)
                self.assertLess(angular, 2.0e-5)


def _hinges(inertias, *, damping=(0.0,), stiffness=(0.0,), device="cpu"):
    """Coaxial hinges in a chain, no gravity: ``H`` is constant, ``H[i, j] = sum_{k >= max(i, j)} I_k``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent, joints = -1, []
    for index, inertia in enumerate(inertias):
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia))
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=(0.0, 0.0, 1.0),
                damping=damping[index % len(damping)],
                spring_stiffness=stiffness[index % len(stiffness)],
                spring_ref=0.0,
                armature=0.0,
                limit_lower=-1.0e6,
                limit_upper=1.0e6,
            )
        )
        parent = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _free_roll(model, q0, v0, steps, *, dt=H, **options):
    """Step ``model`` without contacts from coordinates ``q0`` and velocities ``v0``; return each step's ``(q, v)``."""
    solver = SolverFeatherPGS(model, **{"angular_damping": 0.0, **options})
    state, out = model.state(), model.state()
    state.joint_q.assign(np.asarray(q0, dtype=np.float32))
    state.joint_qd.assign(np.asarray(v0, dtype=np.float32))
    control = model.control()
    history = []
    for _ in range(steps):
        solver.step(state, out, control, None, dt)
        state, out = out, state
        history.append((state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()))
    return history


def _passive_rollout(mass, damping, stiffness, q0, v0, steps, dt=H):
    """The passive law in float64: ``(H + dt C) dv = dt (-K q - C v)`` then ``q += dt v``: damper implicit, spring not."""
    q, v = np.asarray(q0, dtype=np.float64), np.asarray(v0, dtype=np.float64)
    C, K = np.diag(damping), np.diag(stiffness)
    history = []
    for _ in range(steps):
        v = v + np.linalg.solve(mass + dt * C, dt * (-K @ q - C @ v))
        q = q + dt * v
        history.append((q.copy(), v.copy()))
    return history


class TestPassiveDamper(unittest.TestCase):
    """PH1d: a joint's passive damper is backward Euler in the metric of ``H``, its spring stays symplectic."""

    def test_stiff_damper_is_implicit(self):
        """Decay by ``1 / (1 + h c / I)`` each step, at any ``h c / I``: explicit damping diverges above 2."""
        inertia = 0.01
        for ratio in (0.1, 1.0, 2.5, 4.2, 10.0):
            with self.subTest(h_c_over_I=ratio):
                model = _hinges([inertia], damping=(ratio * inertia / H,))
                history = _free_roll(model, [0.0], [1.0], 8)
                expected = (1.0 + ratio) ** -np.arange(1, 9)
                measured = np.array([v[0] for _, v in history])
                np.testing.assert_allclose(measured, expected, rtol=1.0e-3, atol=1.0e-7)

    def test_coupled_chain_follows_the_passive_law(self):
        """Damp through the coupled metric ``H + K`` on a dense factor, with a spring on the same coordinates."""
        inertias = (0.02, 0.01, 0.005)
        size = len(inertias)
        mass = np.array([[sum(inertias[max(i, j) :]) for j in range(size)] for i in range(size)])
        damping, stiffness = (0.0, 2.0, 5.0), (3.0, 0.0, 8.0)
        model = _hinges(inertias, damping=damping, stiffness=stiffness)
        v0 = [1.0, -0.5, 2.0]
        q0 = [0.1, 0.2, -0.1]
        expected = _passive_rollout(mass, damping, stiffness, q0, v0, 60)
        for (q, v), (q_ref, v_ref) in zip(_free_roll(model, q0, v0, 60), expected, strict=True):
            np.testing.assert_allclose(q, q_ref, rtol=2.0e-3, atol=1.0e-4)
            np.testing.assert_allclose(v, v_ref, rtol=2.0e-3, atol=1.0e-3)

    def test_spring_stays_symplectic(self):
        """Keep the undamped spring at ``theta = 0``: its energy stays bounded, as symplectic Euler's does."""
        inertia = 0.01
        for omega_h in (0.1, 1.0, 1.9):
            with self.subTest(omega_h=omega_h):
                stiffness = inertia * (omega_h / H) ** 2
                model = _hinges([inertia], stiffness=(stiffness,))
                steps = 400
                expected = _passive_rollout(np.array([[inertia]]), (0.0,), (stiffness,), [0.1], [0.0], steps)
                measured = _free_roll(model, [0.1], [0.0], steps)
                np.testing.assert_allclose(measured[-1][0], expected[-1][0], rtol=2.0e-2, atol=2.0e-3)


def _rolling_sphere(store: str, *, speed: float, radius: float = 0.1, mu: float = 1.0, device="cpu"):
    """A solid sphere rolling along +x on the ground, without slip, at ``speed`` [m/s], on row store ``store``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu, gap=0.01)
    pose = wp.transform((0.0, 0.0, radius), wp.quat_identity())
    if store == "free":
        body = builder.add_body(xform=pose)
        builder.add_shape_sphere(body, radius=radius, cfg=cfg)
    else:
        body = builder.add_link(xform=pose)
        builder.add_shape_sphere(body, radius=radius, cfg=cfg)
        tail = builder.add_link(xform=pose, mass=1.0e-3, inertia=wp.mat33(np.eye(3) * 1.0e-6))
        root = builder.add_joint_free(body)
        hinge = builder.add_joint_revolute(parent=body, child=tail, axis=(0.0, 1.0, 0.0))
        builder.add_articulation([root, hinge])
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    state = model.state()
    velocity = state.joint_qd.numpy()
    velocity[0], velocity[4] = speed, speed / radius
    state.joint_qd.assign(velocity)
    return model, state


class TestAngularDampingInThePredictor(unittest.TestCase):
    """PH1f: angular damping acts before the rows, so a sticking contact holds at the step's end."""

    radius = 0.1
    damping = 5.0

    def _slip(self, store: str, beta: float) -> float:
        model, state = _rolling_sphere(store, speed=1.0, radius=self.radius)
        solver = SolverFeatherPGS(
            model,
            angular_damping=self.damping,
            friction_anchor_beta=beta,
            pgs_iterations=64,
            update_mass_matrix_interval=1,
        )
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        out = model.state()
        control = model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        for _ in range(24):
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, H)
            state, out = out, state
        velocity = state.joint_qd.numpy()
        return float(velocity[0] - velocity[4] * self.radius)

    def test_rolling_contact_holds_at_the_step_end(self):
        for store in ("free", "tree"):
            with self.subTest(store=store):
                self.assertLess(abs(self._slip(store, 0.0)), 1.0e-3)

    def test_free_spin_keeps_its_decay(self):
        """Decay a free spin by ``1 - a h`` per step, on a body and a tree root."""
        scale = 1.0 - self.damping * H
        for store in ("free", "tree"):
            with self.subTest(store=store):
                model, state = _rolling_sphere(store, speed=0.0, radius=self.radius)
                model.gravity.fill_(wp.vec3(0.0, 0.0, 0.0))
                velocity = state.joint_qd.numpy()
                velocity[5] = 3.0
                history = _free_roll(
                    model, state.joint_q.numpy(), velocity, 10, angular_damping=self.damping, pgs_iterations=4
                )
                self.assertAlmostEqual(float(history[-1][1][5]), 3.0 * scale**10, delta=1.0e-4)


def _driven_hinges(inertias, *, ke, kd, effort, target, device="cpu"):
    """:func:`_hinges` with an effort-limited drive on every coordinate; returns the model and its control."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent, joints = -1, []
    for inertia in inertias:
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia))
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=(0.0, 0.0, 1.0),
                target_ke=ke,
                target_kd=kd,
                effort_limit=effort,
                armature=0.0,
                limit_lower=-1.0e6,
                limit_upper=1.0e6,
            )
        )
        parent = link
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    control = model.control()
    control.joint_target_q.assign(np.asarray(target, dtype=np.float32))
    return model, control


def _launched(step) -> list[str]:
    """The kernels ``step()`` launches, in order."""
    names = []

    def record(run):
        def recorded(kernel, *args, **kwargs):
            names.append(kernel.key)
            return run(kernel, *args, **kwargs)

        return recorded

    with (
        mock.patch.object(wp, "launch", record(wp.launch)),
        mock.patch.object(wp, "launch_tiled", record(wp.launch_tiled)),
    ):
        step()
    return names


def _huber_step(mass, u0, stiffness, limit, dt=H):
    """The exact implicit step of effort-limited drives (float64): the drive energy is Huber, so
    ``H dv = dt clamp(u0 - (K / dt) dv, +-limit)`` is the minimum of a convex energy. Newton with a line search: an
    undamped Newton cycles between active sets at ``K >> H``."""

    def energy(dv):
        force = u0 - stiffness / dt * dv
        inside = np.abs(force) <= limit
        huber = np.where(inside, 0.5 * force * force, limit * np.abs(force) - 0.5 * limit * limit)
        return 0.5 * dv @ mass @ dv + (dt * dt / stiffness * huber).sum()

    dv = np.zeros_like(u0)
    for _ in range(500):
        force = u0 - stiffness / dt * dv
        saturated = np.abs(force) > limit
        residual = mass @ dv - dt * np.clip(force, -limit, limit)
        if np.abs(residual).max() < 1.0e-13:
            break
        update = np.linalg.solve(mass + np.diag(np.where(saturated, 0.0, stiffness)), -residual)
        scale, start = 1.0, energy(dv)
        while energy(dv + scale * update) > start and scale > 1.0e-8:
            scale *= 0.5
        dv = dv + scale * update
    return dv


def _huber_rollout(mass, ke, kd, limit, target, steps, dt=H):
    """Joint positions and velocities of :func:`_huber_step` from rest at zero."""
    q, v = np.zeros(len(target)), np.zeros(len(target))
    stiffness = np.full(len(target), dt * (dt * ke + kd))
    history = []
    for _ in range(steps):
        u0 = -(ke * (q - target + dt * v) + kd * v)
        v = v + _huber_step(mass, u0, stiffness, limit, dt)
        q = q + dt * v
        history.append(np.concatenate((q, v)))
    return np.asarray(history)


def _decided_K(solver) -> np.ndarray:
    """The ``K`` the last step's decision left each saturated drive (its passive damper's): the drives the border released
    from the factor, which holds every drive's ``K``."""
    factor = solver.factor
    if factor.walk_arts is None:
        return np.zeros(0, dtype=np.float32)
    rows = factor.drive_row_by_dof.numpy()
    rows = rows[rows >= 0]
    slipped = factor.walk_view().release.count.numpy()[rows // solver.forest.articulation_max_dofs] > 0
    Kd, K = factor.aug_row_Kd.numpy()[rows], factor.aug_row_K.numpy()[rows]
    return Kd[slipped & (Kd != K)]


def _decided_operator(solver, joint_q: np.ndarray) -> tuple[float, int]:
    """Every tree's release from the border against the decided operator in float64: ``R^T R = I + B C^-1 B^T``
    with ``B`` the float64 walk's columns of its slipping coordinates and ``C = K_S^-1 - B^T B`` (SAT_PHYS §1.2), the
    worst relative error over the trees that slipped, and how many did."""
    from newton.tests.test_feather_pgs_walk_rows import device_release  # noqa: PLC0415

    factor, model = solver.factor, solver.model
    view = factor.walk_view()
    starts, qd = model.articulation_start.numpy(), model.joint_qd_start.numpy()
    rows_by_dof = factor.drive_row_by_dof.numpy()
    K, Kd = factor.aug_row_K.numpy().astype(np.float64), factor.aug_row_Kd.numpy().astype(np.float64)
    slips = view.release.count.numpy()
    worst, trees = 0.0, 0
    for art in np.flatnonzero(slips > 0):
        d0, n = int(qd[starts[art]]), int(qd[starts[art + 1]] - qd[starts[art]])
        rows = rows_by_dof[d0 : d0 + n]
        released = np.where(rows >= 0, K[np.maximum(rows, 0)] - Kd[np.maximum(rows, 0)], 0.0)
        S = np.flatnonzero(released > 0.0)
        walk = factor.walk(joint_q, int(art))
        B = np.stack([walk.up(covector=np.eye(n)[i])[0] for i in S], axis=1)
        expected = np.eye(n) + B @ np.linalg.solve(np.diag(1.0 / released[S]) - B.T @ B, B.T)
        R = device_release(view, int(art), n)
        worst = max(worst, float(np.linalg.norm(R.T @ R - expected) / np.linalg.norm(expected)))
        trees += 1
    return worst, trees


def _drive_roll(model, control, steps, *, dt=H, **options):
    """Step the drives from rest; each step's coordinates and velocities."""
    solver = SolverFeatherPGS(model, **{"angular_damping": 0.0, "pgs_iterations": 4, **options})
    state, out = model.state(), model.state()
    history = []
    for _ in range(steps):
        solver.step(state, out, control, None, dt)
        state, out = out, state
        history.append(np.concatenate((state.joint_q.numpy(), state.joint_qd.numpy())))
    return np.asarray(history), solver


class TestDriveSaturation(unittest.TestCase):
    """PH1c: a saturated drive is a constant force, so its joint accelerates at ``limit / I`` and not ``limit / (I + K)``.

    The reference is the exact step of the drive's Huber energy.
    """

    def test_a_saturated_joint_accelerates_at_limit_over_inertia(self):
        """Hold ``|u0|`` far past the threshold: every step takes ``limit / I``."""
        inertia, limit = 0.01, 1.0
        for ke, kd in ((10.0, 0.1), (1.0e3, 10.0), (1.0e4, 100.0), (1.0e5, 1.8e3)):
            with self.subTest(K_over_I=H * (H * ke + kd) / inertia):
                model, control = _driven_hinges([inertia], ke=ke, kd=kd, effort=limit, target=[1.0e4])
                history, _ = _drive_roll(model, control, 5)
                np.testing.assert_allclose(np.diff(history[:, 1]), limit / inertia * H, rtol=1.0e-4)
                np.testing.assert_allclose(history[0, 1], limit / inertia * H, rtol=1.0e-4)

    def test_a_stiff_drive_on_a_light_joint_stays_implicit(self):
        """Allegro's gains: ``K / I`` of 50 and 500, the step's force within the limit though ``|u0|`` is past it.

        A test of the previous step's force, ``|u0 - K qdd| >= U``, reads rest as saturated here: the joint takes
        ``U / I`` in one step and limit-cycles at 416 rad/s while the exact step reaches the target at 10 rad/s.
        """
        ke, kd, limit, dt = 3.0, 0.1, 0.5, 1.0 / 120.0
        for inertia in (1.0e-5, 1.0e-4):
            with self.subTest(K_over_I=dt * (dt * ke + kd) / inertia):
                steps = 60
                reference = _huber_rollout(np.array([[inertia]]), ke, kd, limit, np.array([0.5]), steps, dt=dt)
                model, control = _driven_hinges([inertia], ke=ke, kd=kd, effort=limit, target=[0.5])
                history, _ = _drive_roll(model, control, steps, dt=dt)
                self.assertLess(float(np.abs(history[:, 1]).max()), 1.1 * float(np.abs(reference[:, 1]).max()) + 1.0)
                np.testing.assert_allclose(history[-1, 0], reference[-1, 0], rtol=0.0, atol=1.0e-4)

    def test_the_threshold_reads_the_armature(self):
        """``H_ii`` is the factor's own diagonal on every dynamics route: a joint whose link is light and whose armature
        is not stays implicit at a ``u0`` that would saturate the link alone."""
        inertia, armature, ke, kd, limit = 1.0e-4, 0.05, 40.0, 1.0, 1.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia))
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=(0.0, 0.0, 1.0),
            target_ke=ke,
            target_kd=kd,
            effort_limit=limit,
            armature=armature,
            limit_lower=-1.0e6,
            limit_upper=1.0e6,
        )
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        control = model.control()
        control.joint_target_q.assign(np.array([0.2], dtype=np.float32))
        history, _ = _drive_roll(model, control, 60)
        reference = _huber_rollout(np.array([[inertia + armature]]), ke, kd, limit, np.array([0.2]), 60)
        np.testing.assert_allclose(history[-1, 0], reference[-1, 0], rtol=0.0, atol=5.0e-4)

    def test_a_coupled_chain_stays_within_the_one_coordinate_bound(self):
        """Coaxial hinges, the strongest coupling: a drive saturates beside a free one, and ``u0`` is hundreds of limits.

        A trial solve with every ``u0`` unlimited charges the free joint with its neighbour's force past the limit and
        saturates it, four times this bound. ``H_ii`` overstates a coordinate's response to its own excess, so the law can
        call a step differently from the exact one near the threshold; the error stays within the bound.
        """
        for inertias, ke, kd, limit, bound in (
            ((0.02, 0.01), 1.0e3, 20.0, 1.5, 0.03),
            ((0.02, 0.01, 0.005), 1.0e3, 20.0, 1.2, 0.03),
            ((0.05, 0.02, 0.01, 0.005, 0.002, 0.001), 400.0, 10.0, 2.0, 0.06),
            ((1.0e-4, 5.0e-5, 2.0e-5, 1.0e-5), 3.0, 0.1, 0.5, 0.02),
        ):
            with self.subTest(joints=len(inertias), lightest=inertias[-1]):
                size = len(inertias)
                target = 0.6 * np.cos(np.arange(size))
                mass = np.array([[sum(inertias[max(i, j) :]) for j in range(size)] for i in range(size)])
                reference = _huber_rollout(mass, ke, kd, limit, target, 240)
                model, control = _driven_hinges(inertias, ke=ke, kd=kd, effort=limit, target=target)
                history, _ = _drive_roll(model, control, 240)
                self.assertLess(float(np.sqrt(np.mean((history[:, :size] - reference[:, :size]) ** 2))), bound)

    def test_collinear_hinges_stay_on_the_coupled_step(self):
        """A link of 0.02 under a load of 0.8 on the same axis, Franka's gains, a new target every step of 1/60 s.

        ``H_ii`` is 0.82 and the first joint's real response ``1 / (A^-1)_ii`` is 0.02, so a one-coordinate test takes ``K``
        out of drives whose coupled step is below the limit: the mode of inertia 0.02 is then driven bang-bang.
        """
        inertias, ke, kd, limit, dt, steps = (0.02, 0.8), 80.0, 4.0, 87.0, 1.0 / 60.0, 600
        mass = np.array([[0.82, 0.8], [0.8, 0.8]])
        stiffness = np.full(2, dt * (dt * ke + kd))
        targets = np.random.default_rng(0).uniform(-1.0, 1.0, (steps, 2))
        q, v, exact = np.zeros(2), np.zeros(2), []
        for target in targets:
            v = v + _huber_step(mass, -(ke * (q - target + dt * v) + kd * v), stiffness, limit, dt)
            q = q + dt * v
            exact.append(np.abs(v).max())
        model, control = _driven_hinges(inertias, ke=ke, kd=kd, effort=limit, target=[0.0, 0.0])
        solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=4)
        state, out = model.state(), model.state()
        speeds = []
        for target in targets:
            control.joint_target_q.assign(target.astype(np.float32))
            solver.step(state, out, control, None, dt)
            state, out = out, state
            speeds.append(np.abs(state.joint_qd.numpy()).max())
        self.assertLess(float(np.max(speeds)), 1.2 * max(exact))
        self.assertLess(float(np.sqrt(np.mean(np.square(speeds)))), 1.2 * float(np.sqrt(np.mean(np.square(exact)))))

    def test_rollout_follows_the_huber_step(self):
        """Saturate, reach the target and hold it on one coordinate and on a coupled chain."""
        chains = (
            ((0.01,), [1.0], 1.0e3, 10.0, 1.0),
            ((0.01,), [1.0], 1.0e4, 100.0, 1.0),
            ((0.02, 0.01), [0.6, -0.8], 1.0e3, 20.0, 1.5),
            ((0.02, 0.01, 0.005), [0.4, -0.5, 0.6], 1.0e3, 20.0, 1.2),
        )
        steps = 120
        for inertias, target, ke, kd, limit in chains:
            with self.subTest(inertias=inertias, ke=ke, kd=kd):
                size = len(inertias)
                mass = np.array([[sum(inertias[max(i, j) :]) for j in range(size)] for i in range(size)])
                reference = _huber_rollout(mass, ke, kd, limit, np.asarray(target), steps)
                model, control = _driven_hinges(inertias, ke=ke, kd=kd, effort=limit, target=target)
                history, _ = _drive_roll(model, control, steps)
                rms = np.sqrt(np.mean((history[:, :size] - reference[:, :size]) ** 2))
                self.assertLess(rms, 0.04)

    def test_a_load_past_the_limit_saturates_a_drive_at_rest(self):
        """A drive at rest has ``u0`` 0 and no excess, and the trial still reads it past its limit: the joint runs at
        ``(load - limit) / I`` from the first step."""
        inertia, ke, kd, limit, load = 0.01, 100.0, 10.0, 1.0, 5.0
        model, control = _driven_hinges([inertia], ke=ke, kd=kd, effort=limit, target=[0.0])
        control.joint_f.assign(np.array([load], dtype=np.float32))
        history, _ = _drive_roll(model, control, 6)
        velocity = np.r_[0.0, history[:, 1]]
        np.testing.assert_allclose(np.diff(velocity), H * (load - limit) / inertia, rtol=1.0e-3)

    def test_a_drive_below_its_limit_keeps_the_implicit_step(self):
        inertia, ke, kd = 0.01, 100.0, 1.0
        reference = _huber_rollout(np.array([[inertia]]), ke, kd, 1.0e6, np.array([0.1]), 40)
        model, control = _driven_hinges([inertia], ke=ke, kd=kd, effort=1.0e6, target=[0.1])
        history, _ = _drive_roll(model, control, 40)
        np.testing.assert_allclose(history, reference, rtol=2.0e-3, atol=1.0e-4)

    def test_articulations_without_coordinates_hold_no_drives(self):
        """The predictor binds the drive rows, which a model with no coordinates does not have."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_fixed(parent=-1, child=link)])
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=4)
        state, out = model.state(), model.state()
        solver.step(state, out, model.control(), None, H)
        np.testing.assert_array_equal(out.body_q.numpy(), state.body_q.numpy())

    def test_a_step_at_a_limit_launches_what_a_quiet_step_does(self):
        """The decision and the border's decided step are the predictor's: no launch of their own, so a world with
        nothing to do pays no launch and a launch does not wait on another for its slowest world."""
        launches, moved = [], []
        for limit in (1.0e6, 0.5):
            model, control = _driven_hinges((0.02, 0.01), ke=1.0e3, kd=20.0, effort=limit, target=[0.6, -0.8])
            solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=4)
            state, out = model.state(), model.state()
            solver.step(state, out, control, None, H)
            launches.append(_launched(lambda: solver.step(out, state, control, None, H)))  # noqa: B023
            moved.append(_decided_K(solver))
        self.assertEqual(launches[0], launches[1])
        for stage in ("walk_step", "tree_kinetic_predict"):
            self.assertEqual(sum(name.startswith(stage) for name in launches[0]), 1, stage)
        self.assertEqual(moved[0].size, 0)
        self.assertLess(float(moved[1].min()), 1.0e-6)

    def test_a_stiff_drive_slips_without_losing_digits(self):
        """A saturated drive's ``K`` leaves the step by the border, whose capacitance takes the pivot with the decided
        ``K`` alone (the walk's ``P0 + Kd``), never ``1 / K - |B|^2``: at ``K`` of 5e5 times the light mode's inertia,
        where that subtraction keeps about one digit in float32, the frame is the decided operator to 1e-4."""
        for saturated in ((1.0e4, -1.0e4), (1.0e4, 0.0)):
            with self.subTest(targets=saturated):
                model, control = _driven_hinges((0.02, 0.8), ke=0.0, kd=6.0e5, effort=1.0, target=[0.0, 0.0])
                control.joint_target_qd.assign(np.asarray(saturated, dtype=np.float32))
                history, solver = _drive_roll(model, control, 3, update_mass_matrix_interval=2)
                decided = _decided_K(solver)
                self.assertEqual(int((decided < 1.0e-6).sum()), int(np.count_nonzero(saturated)))
                joint_q = history[-2, : model.joint_coord_count].astype(np.float64)
                worst, trees = _decided_operator(solver, joint_q)
                self.assertEqual(trees, 1)
                self.assertLess(worst, 1.0e-4)

    def test_a_branched_tree_slips_along_its_own_paths(self):
        """Each slip's column walks up its own root path: from a finger to the palm, then the palm's coordinates when the
        palm is free, never the fingers between. On each device the border is the decided operator to 1e-4 at every
        step, its slips in several fingers at once."""
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            for floating in (False, True):
                with self.subTest(device=device, floating=floating):
                    model, control = _driven_hand(device, floating)
                    solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=4)
                    state, out = model.state(), model.state()
                    rng = np.random.default_rng(1)
                    for _ in range(12):
                        control.joint_target_q.assign(rng.uniform(-2.0, 2.0, model.joint_dof_count).astype(np.float32))
                        joint_q = solver.cut.mirror(state).joint_q.numpy().astype(np.float64)
                        solver.step(state, out, control, None, 1.0 / 60.0)
                        state, out = out, state
                        worst, trees = _decided_operator(solver, joint_q)
                        self.assertEqual(trees, 2)
                        self.assertLess(worst, 1.0e-4)
                    self.assertGreater(int(solver.factor.walk_view().release.count.numpy().min()), 3)

    def test_slips_past_a_trees_frame_flag_its_capacity(self):
        """The border holds a tree's slips in its frame, as many as the tree has limited drives up to
        ``factor.walk.RELEASE_SLIPS``: a chain of 40 drives all at their limits passes it and flags ``slip_frames``, two
        hinges never do."""
        for scene, flagged in ((_driven_chain, True), (_driven_forest, False)):
            with self.subTest(scene=scene.__name__):
                model, control = scene("cpu")
                targets = np.random.default_rng(1).uniform(-1.0, 1.0, model.joint_dof_count).astype(np.float32)
                control.joint_target_q.assign(targets)
                _, solver = _drive_roll(model, control, 4, dt=1.0 / 60.0)
                self.assertEqual(solver.constraint_capacity_status()["slip_frames"], flagged)

    def test_a_limit_set_after_construction_is_flagged(self):
        """A tree with no limited drive when the solver is built has no room in the slips' frame (and, unless its class
        walks, no walk): a limit written later saturates its drive, the step keeps that drive's ``K`` and
        ``slip_frames`` says so, rather than reading past an array."""
        model, control = _driven_hinges((0.02, 0.01), ke=1.0e3, kd=20.0, effort=float("inf"), target=[0.6, -0.8])
        solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=4)
        self.assertEqual(solver.factor.slip_capacity([0]), 0)
        model.joint_effort_limit.assign(np.full(model.joint_dof_count, 0.5, dtype=np.float32))
        state, out = model.state(), model.state()
        solver.step(state, out, control, None, H)
        self.assertTrue(np.all(np.isfinite(out.joint_qd.numpy())))
        self.assertTrue(solver.constraint_capacity_status()["slip_frames"])

    def test_a_saturation_change_refreshes_a_stale_factor(self):
        """Hold ``update_mass_matrix_interval=2`` to the every-step result where ``H`` is constant: only ``K`` moves."""
        arms = []
        for interval in (1, 2):
            model, control = _driven_hinges((0.02, 0.01), ke=1.0e3, kd=20.0, effort=1.5, target=[0.6, -0.8])
            arms.append(_drive_roll(model, control, 120, update_mass_matrix_interval=interval)[0])
        np.testing.assert_allclose(arms[1], arms[0], rtol=1.0e-4, atol=2.0e-4)


@unittest.skipUnless(wp.is_cuda_available(), "scalar branches run on CUDA")
class TestScalarBranchJointLaws(unittest.TestCase):
    """PH1c and PH1d on the keys' route: their predict folds ``K`` and refreshes each leaf's inverse mass itself."""

    def _leaves(self, *, ke=0.0, kd=0.0, effort=0.0, damping=0.0, v0=0.0, interval=1):
        model = branch.build_branch_model("cuda:0", worlds=2, arm=True)
        model.gravity.fill_(wp.vec3(0.0, 0.0, 0.0))
        size = branch.branch_size(model)
        dofs = np.asarray(model.joint_qd_start.numpy()[model.articulation_start.numpy()[0] + 1]) + np.arange(size)
        child = model.joint_child.numpy()[model.articulation_start.numpy()[0] + 1 + np.arange(size)]
        inertia = model.body_mass.numpy()[child] + model.joint_armature.numpy()[dofs]
        for name, value in (("target_ke", ke), ("target_kd", kd), ("effort_limit", effort), ("damping", damping)):
            array = getattr(model, f"joint_{name}")
            values = array.numpy()
            values[dofs] = value
            array.assign(values)
        for name in ("joint_spring_stiffness",):
            getattr(model, name).zero_()
        for name in ("joint_limit_lower",):
            getattr(model, name).fill_(-1.0e3)
        model.joint_limit_upper.fill_(1.0e3)
        solver = branch.recipe_solver(model, scalar=True, angular_damping=0.0, update_mass_matrix_interval=interval)
        self.assertTrue(solver.get_path_selection()["routes"]["held_tree"]["scalar_branches"]["enabled"])
        state, out = model.state(), model.state()
        state.joint_q.zero_()
        velocity = np.zeros(model.joint_dof_count, dtype=np.float32)
        velocity[dofs] = v0
        state.joint_qd.assign(velocity)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        return solver, model, state, out, dofs, inertia

    def test_stiff_damper_is_implicit(self):
        damping = 50.0
        solver, model, state, out, dofs, inertia = self._leaves(damping=damping, v0=1.0)
        solver.step(state, out, model.control(), None, H)
        np.testing.assert_allclose(out.joint_qd.numpy()[dofs], 1.0 / (1.0 + H * damping / inertia), rtol=2.0e-3)

    def test_saturation_refreshes_a_stale_inverse_mass(self):
        """Accelerate at ``limit / I`` on a step without a factor refresh."""
        limit = 0.05
        solver, model, state, out, dofs, inertia = self._leaves(ke=1.0e4, kd=100.0, effort=limit, interval=2)
        control = model.control()
        target = np.zeros_like(control.joint_target_q.numpy())
        control.joint_target_q.assign(target)
        solver.step(state, out, control, None, H)
        state, out = out, state
        # The first step stays below the limit at rest on target; the second is a step without a mass update.
        target[dofs] = 1.0e3
        control.joint_target_q.assign(target)
        before = state.joint_qd.numpy()[dofs]
        solver.step(state, out, control, None, H)
        np.testing.assert_allclose(out.joint_qd.numpy()[dofs] - before, limit / inertia * H, rtol=2.0e-3)


_FINGER = {"ke": 3.0, "kd": 0.1, "limit": 0.5, "dt": 1.0 / 120.0, "lever": 0.05}
"""Allegro's gains on a link of 1e-4 about its hinge, ``K / I`` about 10."""


def _pressed_finger(*, stop: bool = False, gap: float = 0.0005, device="cpu"):
    """A hinge about y with a link along +x: ``q > 0`` turns its tip onto the ground, or onto a joint limit at ``stop``."""
    lever = _FINGER["lever"]
    radius = 0.008
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    pose = wp.transform((0.0, 0.0, radius + gap), wp.quat_identity())
    link = builder.add_link(xform=pose, mass=0.03, com=(lever, 0.0, 0.0), inertia=wp.mat33(np.eye(3) * 2.5e-5))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, gap=0.002)
    if not stop:
        builder.add_shape_sphere(
            link, xform=wp.transform((lever, 0.0, 0.0), wp.quat_identity()), radius=radius, cfg=cfg
        )
        builder.add_ground_plane(cfg=cfg)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 1.0, 0.0),
        parent_xform=pose,
        target_ke=_FINGER["ke"],
        target_kd=_FINGER["kd"],
        effort_limit=_FINGER["limit"],
        armature=0.0,
        limit_lower=-1.0e3,
        limit_upper=0.02 if stop else 1.0e3,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _press(targets, *, stop: bool = False, gap: float = 0.0005, device="cpu"):
    """Step the pressed finger through ``targets``; each step's contact moment about the hinge, drive force, ``q`` and
    ``qd``."""
    model = _pressed_finger(stop=stop, gap=gap, device=device)
    solver = SolverFeatherPGS(
        model, angular_damping=0.0, pgs_iterations=16, enable_joint_limits=stop, joint_limit_activation_gap=0.05
    )
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    pipeline = None if stop else newton.CollisionPipeline(model, rigid_contact_max=32)
    contacts = None if stop else pipeline.contacts()
    history = []
    for target in targets:
        control.joint_target_q.assign(np.array([target], dtype=np.float32))
        state.clear_forces()
        if pipeline is not None:
            pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, _FINGER["dt"])
        moment = float("nan")
        if pipeline is not None:
            solver.update_contacts(contacts)
            force = contacts.rigid_contact_force.numpy()[: int(contacts.rigid_contact_count.numpy()[0])]
            moment = float(np.abs(force[:, 2]).sum()) * _FINGER["lever"]
        drive = float(solver.factor.joint_tau.numpy()[0])
        history.append((moment, drive, float(out.joint_q.numpy()[0]), float(out.joint_qd.numpy()[0])))
        state, out = out, state
    return np.asarray(history)


def _rattle(dt, *, worlds=16, steps=480, device="cpu"):
    """Peak speed of three light links on a hinge chain, the tip on the ground under gravity, with targets at 60 Hz."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, gap=0.002)
    parent, joints, offset, x = -1, [], (0.0, 0.0, 0.06), 0.0
    links = ((0.03, 2.0e-5, 0.045), (0.02, 1.0e-5, 0.035), (0.01, 4.0e-6, 0.025))
    for index, (mass, inertia, length) in enumerate(links):
        link = template.add_link(
            xform=wp.transform((x, 0.0, 0.06), wp.quat_identity()),
            mass=mass,
            com=(length / 2.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3) * inertia),
        )
        if index == len(links) - 1:
            template.add_shape_sphere(
                link, xform=wp.transform((length, 0.0, 0.0), wp.quat_identity()), radius=0.008, cfg=cfg
            )
        joints.append(
            template.add_joint_revolute(
                parent=parent,
                child=link,
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform(offset, wp.quat_identity()),
                target_ke=_FINGER["ke"],
                target_kd=_FINGER["kd"],
                effort_limit=_FINGER["limit"],
                armature=1.0e-5,
                limit_lower=-1.0e3,
                limit_upper=1.0e3,
            )
        )
        parent, offset = link, (length, 0.0, 0.0)
        x += length
    template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.replicate(template, worlds, spacing=(0.3, 0.0, 0.0))
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=16)
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32 * worlds)
    contacts = pipeline.contacts()
    rng = np.random.default_rng(0)
    hold, peak = round(1.0 / (60.0 * dt)), 0.0
    for step in range(steps):
        if step % hold == 0:
            targets = rng.uniform(0.2, 1.4, (worlds, 3)) * np.array([1.0, 0.5, 0.5])
            control.joint_target_q.assign(targets.reshape(-1).astype(np.float32))
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, dt)
        state, out = out, state
        peak = max(peak, float(np.abs(state.joint_qd.numpy()).max()))
    return peak


class TestDrivePressedOnARow(unittest.TestCase):
    """A drive blocked by a contact or a limit moves nowhere, so its force on the step is ``u0``: past the limit it is
    the limit. The trial solves no rows and reads a blocked drive as free, so its excess follows last step's rows."""

    def test_a_drive_on_a_contact_carries_its_limit(self):
        """Target errors of 0.05, 0.3 and 1.0: ``u0`` of 0.15, 0.9 and 3 Nm against a limit of 0.5 Nm."""
        limit = _FINGER["limit"]
        for target in (0.05, 0.3, 1.0):
            with self.subTest(target=target):
                history = _press([target] * 120)
                expected = min(_FINGER["ke"] * (target - history[-1, 2]), limit)
                np.testing.assert_allclose(history[-20:, 0], expected, rtol=2.0e-3)
                self.assertLess(float(np.abs(history[-20:, 3]).max()), 1.0e-3)

    def test_a_drive_comes_off_its_limit_when_it_stops_pushing(self):
        history = _press([1.0] * 60 + [0.05] * 60)
        np.testing.assert_allclose(history[50:60, 0], _FINGER["limit"], rtol=2.0e-3)
        np.testing.assert_allclose(history[-20:, 0], _FINGER["ke"] * (0.05 - history[-1, 2]), rtol=2.0e-3)

    def test_a_drive_on_a_joint_limit_carries_its_limit(self):
        """The same on a limit row, read from the force the drive applies."""
        history = _press([1.0] * 120, stop=True)
        np.testing.assert_allclose(history[-20:, 1], _FINGER["limit"], rtol=2.0e-3)

    def test_a_drive_leaving_a_contact_takes_the_free_step(self):
        """Lift the tip off the ground: the rows' reading ends with the contact, and the drive takes its whole ``u0``."""
        history = _press([1.0] * 40 + [-1.0] * 20)
        free = _press([0.0] * 40 + [-1.0] * 20, gap=0.05)
        np.testing.assert_allclose(history[-1, 3], free[-1, 3], rtol=2.0e-2)

    def test_a_coupled_finger_on_a_contact_stays_calm(self):
        """Three light links, the tip on the ground, targets that jump at 60 Hz: taking ``K`` out on a lagged reading gives
        a joint ``limit / I`` beside a contact that has left, and this ran to 18,000 rad/s."""
        for dt in (1.0 / 120.0, 1.0 / 60.0):
            with self.subTest(dt=dt):
                self.assertLess(_rattle(dt), 40.0)


def _driven_forest(device):
    """Two trees of different sizes in each of two worlds, the first two collinear hinges, with drives that saturate."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for inertias in ((0.02, 0.8), (0.03, 0.01, 0.005)):
        parent, joints = -1, []
        for inertia in inertias:
            link = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia))
            joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=(0.0, 0.0, 1.0),
                    target_ke=80.0,
                    target_kd=4.0,
                    effort_limit=2.0,
                    armature=0.0,
                    limit_lower=-1.0e6,
                    limit_upper=1.0e6,
                )
            )
            parent = link
        template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    model = builder.finalize(device=device)
    return model, model.control()


def _driven_hand(device, floating: bool):
    """A palm, fixed or free, with three fingers of three hinges in each of two worlds, with drives that saturate: a
    slip's root path leaves its finger at the palm, past the other fingers' coordinates."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    palm = template.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.05))
    joints = [template.add_joint_free(child=palm) if floating else template.add_joint_fixed(parent=-1, child=palm)]
    for finger in range(3):
        parent = palm
        for k in range(3):
            link = template.add_link(mass=0.3, inertia=wp.mat33(np.eye(3) * (0.01 + 0.004 * k + 0.002 * finger)))
            joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=(0.0, 0.0, 1.0) if k else (1.0, 0.0, 0.0),
                    parent_xform=wp.transform((0.05 * (finger - 1), 0.0, 0.1 if k == 0 else 0.05), wp.quat_identity()),
                    target_ke=80.0,
                    target_kd=4.0,
                    effort_limit=1.0,
                    armature=0.0,
                    limit_lower=-1.0e6,
                    limit_upper=1.0e6,
                )
            )
            parent = link
    template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    model = builder.finalize(device=device)
    return model, model.control()


def _driven_chain(device):
    """A chain of 40 hinges in each of two worlds, past a warp of lanes, with drives that saturate."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent, joints = -1, []
    for index in range(40):
        link = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * (0.02 + 0.01 * (index % 3))))
        joints.append(
            template.add_joint_revolute(
                parent=parent,
                child=link,
                axis=(0.0, 0.0, 1.0),
                target_ke=80.0,
                target_kd=4.0,
                effort_limit=2.0,
                armature=0.01,
                limit_lower=-1.0e6,
                limit_upper=1.0e6,
            )
        )
        parent = link
    template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    model = builder.finalize(device=device)
    return model, model.control()


@unittest.skipUnless(wp.is_cuda_available(), "the held tree's natives run on CUDA")
class TestDriveSaturationOnRoutes(unittest.TestCase):
    """The decision on each evaluator of the trees: a release and a second solve that reach the right articulations."""

    def _roll(self, device, pins, *, scene=_driven_forest, **options):
        with conformance.route_context(conformance.Route("drives", device.split(":")[0], pins=pins)):
            model, control = scene(device)
            solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=4, **options)
        state, out = model.state(), model.state()
        history = []
        targets = np.random.default_rng(1).uniform(-1.0, 1.0, (120, model.joint_dof_count))
        # The second world's first tree stays quiet: only some articulations take the second solve.
        targets[:, 5] = 0.0
        targets[:, 6] = 0.0
        for target in targets:
            control.joint_target_q.assign(target.astype(np.float32))
            solver.step(state, out, control, None, 1.0 / 60.0)
            state, out = out, state
            history.append(state.joint_qd.numpy().copy())
        return np.asarray(history), solver

    def test_routes_agree_with_the_cpu(self):
        reference, _ = self._roll("cpu", {})
        self.assertGreater(float(np.abs(reference).max()), 2.0)
        for pins in ({}, {"grouping": "forest"}):
            with self.subTest(pins=pins):
                history, _ = self._roll("cuda:0", pins)
                np.testing.assert_allclose(history, reference, rtol=0.0, atol=1.0e-2)

    def test_a_long_chain_takes_the_decision(self):
        """A tree of two warps: the block's barriers and, past the shared factor's size, its factor in place."""
        reference, _ = self._roll("cpu", {}, scene=_driven_chain)
        history, solver = self._roll("cuda:0", {}, scene=_driven_chain)
        self.assertEqual(solver.factor.held.groups[0].layout.lanes, 64)
        self.assertLess(float(_decided_K(solver).min()), 1.0e-6)
        np.testing.assert_allclose(history[:20], reference[:20], rtol=0.0, atol=2.0e-2)

    def test_the_border_is_the_decided_operator(self):
        """On each evaluator the border's frame of the last step's slips is, to float32 rounding, the decided operator
        ``(H + R + K_F)^-1`` on the all-``K`` walk (the factor holds every drive's ``K``, untouched)."""
        for device, pins in (("cpu", {}), ("cuda:0", {}), ("cuda:0", {"grouping": "forest"})):
            with self.subTest(device=device, pins=pins):
                _, solver = self._roll(device, pins)
                self.assertLess(float(_decided_K(solver).min()), 1.0e-6)
                self.assertGreater(float(solver.factor.aug_row_K.numpy().max()), 1.0e-3)
                worst, trees = _decided_operator(solver, solver.model.joint_q.numpy().astype(np.float64))
                self.assertGreater(trees, 0)
                self.assertLess(worst, 1.0e-4)

    def test_a_captured_graph_decides_like_an_eager_step(self):
        """The decision in a graph: steps at their limits, then quiet steps, then at their limits again."""
        targets = np.zeros((60, 10))
        targets[:20] = targets[40:] = np.random.default_rng(2).uniform(-1.0, 1.0, (10,))
        for pins in ({}, {"grouping": "forest"}):
            with self.subTest(pins=pins), conformance.route_context(conformance.Route("drives", "cuda", pins=pins)):
                arms = []
                for captured in (False, True):
                    model, control = _driven_forest("cuda:0")
                    solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_iterations=4)
                    state, out = model.state(), model.state()
                    graph = None
                    if captured:
                        with wp.ScopedCapture("cuda:0") as capture:
                            solver.step(state, out, control, None, 1.0 / 60.0)
                            solver.step(out, state, control, None, 1.0 / 60.0)
                        graph = capture.graph
                    history = []
                    for target in targets[::2]:
                        control.joint_target_q.assign(target.astype(np.float32))
                        if graph is not None:
                            wp.capture_launch(graph)
                        else:
                            solver.step(state, out, control, None, 1.0 / 60.0)
                            solver.step(out, state, control, None, 1.0 / 60.0)
                        history.append(state.joint_qd.numpy().copy())
                    arms.append(np.asarray(history))
                np.testing.assert_allclose(arms[1], arms[0], rtol=0.0, atol=1.0e-3)

    def test_a_drive_on_a_contact_carries_its_limit(self):
        history = _press([1.0] * 120, device="cuda:0")
        np.testing.assert_allclose(history[-20:, 0], _FINGER["limit"], rtol=2.0e-3)

    def _sparse_roll(self, dense, steps, interval):
        model = sparse._build_model(device="cuda:0")
        dofs = np.arange(6, model.joint_dof_count)
        for name, value in (("target_ke", 1.0e3), ("target_kd", 20.0), ("effort_limit", 0.2)):
            array = getattr(model, f"joint_{name}")
            values = array.numpy()
            values[dofs] = value
            array.assign(values)
        control = model.control()
        control.joint_target_q.assign(0.5 * np.cos(np.arange(control.joint_target_q.shape[0])).astype(np.float32))
        solver = sparse._make_solver(model, not dense, update_mass_matrix_interval=interval)
        self.assertEqual(solver.placement.sparse_mass_matrix_size is not None, not dense)
        state, out = model.state(), model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        history = []
        for _ in range(steps):
            solver.step(state, out, control, None, H)
            state, out = out, state
            history.append(state.joint_qd.numpy().copy())
        self.assertLess(float(_decided_K(solver).min()), 1.0e-6)
        return np.asarray(history), solver

    def test_the_sparse_factor_takes_the_decision(self):
        arms = [self._sparse_roll(dense, 60, 1)[0] for dense in (False, True)]
        np.testing.assert_allclose(arms[1], arms[0], rtol=0.0, atol=2.0e-2)

    def test_a_compact_diagonal_decides_in_its_predictor(self):
        """A tree group of diagonal mass, the star pin keeping it from scalar branches: its inverse is the coordinate's own,
        on a step without a mass update."""
        limit = 0.05
        model = branch.build_branch_model("cuda:0", worlds=2, arm=True)
        model.gravity.fill_(wp.vec3(0.0, 0.0, 0.0))
        size = branch.branch_size(model)
        dofs = np.asarray(model.joint_qd_start.numpy()[model.articulation_start.numpy()[0] + 1]) + np.arange(size)
        child = model.joint_child.numpy()[model.articulation_start.numpy()[0] + 1 + np.arange(size)]
        inertia = model.body_mass.numpy()[child] + model.joint_armature.numpy()[dofs]
        for name, value in (("target_ke", 1.0e4), ("target_kd", 100.0), ("effort_limit", limit), ("damping", 0.0)):
            array = getattr(model, f"joint_{name}")
            values = array.numpy()
            values[dofs] = value
            array.assign(values)
        model.joint_spring_stiffness.zero_()
        model.joint_limit_lower.fill_(-1.0e3)
        model.joint_limit_upper.fill_(1.0e3)
        solver = branch.recipe_solver(model, scalar=False, angular_damping=0.0, update_mass_matrix_interval=2)
        self.assertTrue(
            any(getattr(group.sparse, "response", None) == "diagonal" for group in solver.factor.held.groups)
        )
        control = model.control()
        state, out = model.state(), model.state()
        state.joint_q.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        target = np.zeros_like(control.joint_target_q.numpy())
        control.joint_target_q.assign(target)
        solver.step(state, out, control, None, H)
        state, out = out, state
        # The first step stays below the limit at rest on target; the second is a step without a mass update.
        target[dofs] = 1.0e3
        control.joint_target_q.assign(target)
        before = state.joint_qd.numpy()[dofs]
        solver.step(state, out, control, None, H)
        np.testing.assert_allclose(out.joint_qd.numpy()[dofs] - before, limit / inertia * H, rtol=2.0e-3)


@unittest.skipUnless(wp.is_cuda_available(), "the specialized row stores run on CUDA")
class TestDepenetrationCapOnRoutes(unittest.TestCase):
    """PH1a on each contact-row producer: the normal law at the capped bias, from the public contact record."""

    cap = 0.01
    margin = 0.01
    bound = 2.0e-3
    """The natural residual a capped run keeps [m/s]: the solves are budget-bound, an uncapped row misses by far more."""

    def _residual(self, scene, law, route, iterations) -> tuple[float, float]:
        def build(device, base=scene.build):
            model = base(device)
            _caps(model, self.cap)
            # Thicker shapes overlap wherever they touch, so every contact starts deep, on any scene.
            model.shape_margin.fill_(self.margin)
            return model

        capped = dataclasses.replace(scene, build=build)
        options = {"friction_anchor_beta": 0.0 if law == "point" else 0.2, "pgs_iterations": iterations}
        run = conformance.simulate(capped, route, "cuda", options=options)
        for path, value in route.expect.items():
            self.assertEqual(conformance.route_value(run.routes, path), value, path)
        data, _joints, _mass = conformance.model_data(scene)
        worst, deepest = 0.0, 0.0
        for step, entry in enumerate(run.contacts):
            if len(entry["force"]) == 0:
                continue
            frame = conformance.contact_frame(entry, run.body_q[step], run.body_qd_solve[step], data)
            bias = np.where(frame.gap <= 0.0, np.maximum(0.2 * frame.gap / scene.dt, -self.cap), frame.gap / scene.dt)
            weight = scene.dt / conformance._reduced_mass(frame, data)
            residual = frame.separation_velocity + bias
            worst = max(worst, float(np.abs(np.minimum(np.maximum(frame.normal_force, 0.0) * weight, residual)).max()))
            deepest = max(deepest, float((-0.2 * frame.gap / scene.dt).max()))
        return worst, deepest

    def test_every_producer(self):
        cases = (
            (scenes.box_drop(), conformance.MF_HELD, 32),
            (scenes.legged(), conformance.MF_HELD, 32),
            (scenes.legged(), conformance.SPARSE, 32),
            (scenes.legged(), conformance.KEYED_PARALLEL, 32),
            (scenes.gripper(), conformance.MF_DEFAULT, 32),
            (scenes.keyboard(), conformance.SPARSE_DIAGONAL, 64),
            (scenes.keyboard(), conformance.COMPACT_BOUNDARY, 64),
            (scenes.hand(), conformance.PAIRED, 16),
        )
        for scene, route, iterations in cases:
            for law in ("point", "patch"):
                with self.subTest(scene=scene.name, route=route.name, law=law):
                    worst, deepest = self._residual(scene, law, route, iterations)
                    self.assertGreater(deepest, 2.0 * self.cap, "the scene never reached the cap")
                    self.assertLess(worst, self.bound)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FeatherPGS's repaired laws against closed forms: each test failed on the solver the repair replaced.

Conformance compares routes with their own converged run, so a wrong law is common-mode there. A test here takes its
reference from the law as physics states it, and runs on the row stores or integrators that repair reached.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance

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
            (scenes.legged(), conformance.MF_GENERAL, 32),
            (scenes.legged(), conformance.SPARSE, 32),
            (scenes.legged(), conformance.KEYED_PARALLEL, 32),
            (scenes.gripper(), conformance.LOCAL_PACKETS, 32),
            (scenes.gripper(), conformance.LOCAL_OWNED, 32),
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

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

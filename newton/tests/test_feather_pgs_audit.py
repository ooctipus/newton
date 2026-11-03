# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The audit (``pgs_audit``): the energy gap, the dual violation and the injected work against what they claim."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.audit import AUDIT_SLOTS
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests import test_feather_pgs_conformance as conformance
from newton.tests.feather_pgs_law_harness import walk_inverse

H = 1.0 / 240.0
G0 = 9.81


def _box_model(device, *, count=1, mu=0.5, height=None, spacing=0.3, half=0.1, ground=True, gap=0.01):
    """``count`` 1 kg boxes above the ground, in a stack by default; ``height`` overrides the first box's centre."""
    builder = newton.ModelBuilder(gravity=-G0)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=1000.0, restitution=0.0)
    cfg.gap = gap
    inertia = (1.0 / 6.0) * 4.0 * half * half
    bottom = half if height is None else height
    for index in range(count):
        z = bottom + 2.0 * half * index
        body = builder.add_body(
            xform=wp.transform((0.0, 0.0, z), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * inertia),
            lock_inertia=True,
        )
        builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)
    if ground:
        builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device)


def _hinge_model(device, *, limit=0.2):
    """One hinge about y carrying a 1 kg rod, swung into its limit."""
    builder = newton.ModelBuilder(gravity=-G0)
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform((0.25, 0.0, 0.0), wp.quat_identity()),
        limit_lower=-limit,
        limit_upper=limit,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _rollout(model, *, steps, dt=H, before=None, **options):
    """Step a model with the audit on; ``before(step, state)`` writes a step's initial state or forces."""
    settings = {"pgs_iterations": 64, "enable_joint_limits": True, "pgs_audit": True}
    settings.update(options)
    solver = SolverFeatherPGS(model, **settings)
    state, out = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    for step in range(steps):
        state.clear_forces()
        if before is not None:
            before(step, state)
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, dt)
        state, out = out, state
    return solver, state


def _free_coordinates(solver, velocity):
    """Each free body's ``u = L^T v`` through its walk (``H = L L^T``, :func:`walk_inverse`), by body: the metric of the
    kinetic energy."""
    start = solver.forest.articulation_dof_start.numpy()
    groups = solver.forest.group_to_art[6].numpy()
    u = np.zeros((groups.size, 6))
    for group, art in enumerate(groups):
        v = velocity[start[art] : start[art] + 6].astype(np.float64)
        u[group] = np.linalg.inv(walk_inverse(solver, int(art))).T @ v
    return u


def _store_sums(solver):
    """``(sum lambda r, sum lambda b)`` over both stores, ``r`` read back from the audit's velocity terms."""
    audit = solver._audit
    total, biased = 0.0, 0.0
    for impulses, velocity, rhs, counts in (
        (solver.placement.impulses, audit.velocity, solver.placement.rhs, solver.placement.constraint_count),
        (
            solver.placement.body_rows.impulses,
            audit.mf_velocity,
            solver.placement.body_rows.rhs,
            solver.placement.mf_constraint_count,
        ),
    ):
        lam, jv, b = impulses.numpy(), velocity.numpy(), rhs.numpy()
        for world, count in enumerate(counts.numpy()):
            total += float(np.sum(lam[world, :count].astype(np.float64) * (jv[world, :count] + b[world, :count])))
            biased += float(np.sum(lam[world, :count].astype(np.float64) * b[world, :count]))
    return total, biased


class TestAudit(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        self.scope = wp.ScopedDevice(self.device)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def test_off_allocates_and_launches_nothing(self):
        """Off, the solver holds no audit, reads none and reports it."""
        solver = SolverFeatherPGS(_box_model(self.device))
        self.assertIsNone(solver._audit)
        with self.assertRaisesRegex(RuntimeError, "pgs_audit=True"):
            solver.read_audit()
        self.assertFalse(solver.get_path_selection()["routes"]["diagnostics"]["audit"]["enabled"])

    def test_settled_box_has_no_gap(self):
        """A settled box's rows are solved: the gap and the violation sit at rounding."""
        solver, _state = _rollout(_box_model(self.device), steps=120)
        audit = solver.read_audit()
        self.assertEqual(set(audit), set(AUDIT_SLOTS))
        self.assertLess(abs(float(audit["gap"][0])), 1.0e-6)
        self.assertLess(float(audit["violation"][0]), 1.0e-4)
        self.assertEqual(float(audit["steps"][0]), 120.0)

    def test_gap_falls_with_the_budget(self):
        """An unconverged stack holds a gap the budget closes."""
        gaps = {}
        for sweeps in (2, 8, 32, 128):
            solver, _state = _rollout(_box_model(self.device, count=5), steps=30, pgs_iterations=sweeps)
            gaps[sweeps] = float(solver.read_audit()["gap_sum"][0])
        self.assertGreater(gaps[2], 5.0 * gaps[128], gaps)
        self.assertGreater(gaps[2], 1.0e-3, gaps)
        self.assertLess(gaps[128], gaps[8], gaps)
        self.assertLess(gaps[8], gaps[2], gaps)

    def test_energy_identity(self):
        """``sum lambda r = dT + sum lambda b + |Z^T lambda|^2 / 2`` on an impact, each side from its own data."""
        solver, _state = _rollout(
            _box_model(self.device, mu=0.0, height=0.1005), steps=1, before=self._downward(self.device)
        )
        v_hat, v_out = solver.v_hat.numpy(), solver.v_out.numpy()
        u_hat, u_out = _free_coordinates(solver, v_hat), _free_coordinates(solver, v_out)
        kinetic_change = 0.5 * float(np.sum(u_out * u_out) - np.sum(u_hat * u_hat))
        reaction = 0.5 * float(np.sum((u_out - u_hat) ** 2))
        rows, biased = _store_sums(solver)
        self.assertGreater(abs(kinetic_change), 1.0e-3)
        scale = max(abs(rows), abs(kinetic_change), abs(reaction), abs(biased))
        self.assertAlmostEqual(rows, kinetic_change + biased + reaction, delta=1.0e-4 * scale)
        audit = solver.read_audit()
        self.assertAlmostEqual(float(audit["work"][0]), -biased, delta=1.0e-6 * max(abs(biased), 1.0e-9))
        self.assertLess(float(audit["gap"][0]), 1.0e-5)

    @staticmethod
    def _downward(device, speed=1.0):
        """A step-zero write of the first body's velocity to ``-speed`` along z."""

        def before(step, state):
            if step == 0:
                qd = state.joint_qd.numpy()
                qd[2] = -speed
                wp.copy(state.joint_qd, wp.array(qd, dtype=wp.float32, device=device))

        return before

    def test_hinge_limit_identity(self):
        """A joint-limit row's law on the dense store: the impact's identity closes in the hinge's one-coordinate metric,
        and the rod that swings away has no gap."""
        limit = 0.02

        def before(step, state):
            if step == 0:
                wp.copy(state.joint_qd, wp.array([3.0], dtype=wp.float32, device=self.device))

        solver, _state = _rollout(_hinge_model(self.device, limit=limit), steps=3, before=before)
        self.assertGreater(float(solver.placement.impulses.numpy()[0, :2].sum()), 1.0e-2)
        factor = 1.0 / float(walk_inverse(solver, 0)[0, 0])
        u_hat, u_out = factor * float(solver.v_hat.numpy()[0]), factor * float(solver.v_out.numpy()[0])
        rows, biased = _store_sums(solver)
        kinetic_change = 0.5 * (u_out * u_out - u_hat * u_hat)
        reaction = 0.5 * (u_out - u_hat) ** 2
        scale = max(abs(rows), abs(kinetic_change), abs(reaction), abs(biased))
        self.assertAlmostEqual(rows, kinetic_change + biased + reaction, delta=1.0e-4 * scale)

        solver, _state = _rollout(_hinge_model(self.device, limit=limit), steps=40, before=before)
        audit = solver.read_audit()
        self.assertLess(abs(float(audit["gap"][0])), 1.0e-5)
        self.assertLess(float(audit["violation"][0]), 1.0e-3)

    def test_sliding_friction_term_cancels(self):
        """A sliding box's friction term closes: ``lambda_t r_t`` alone is about ``-mu N v``, and G nearly zero."""
        model = _box_model(self.device, mu=0.2)

        def before(step, state):
            if step == 0:
                qd = state.joint_qd.numpy()
                qd[0] = 1.0
                wp.copy(state.joint_qd, wp.array(qd, dtype=wp.float32, device=self.device))

        solver, _state = _rollout(model, steps=12, before=before, pgs_iterations=128)
        audit = solver.read_audit()
        rows, _biased = _store_sums(solver)
        sliding = 0.2 * 1.0 * G0 * H * float(solver.v_out.numpy()[0])
        self.assertLess(rows, -0.5 * sliding)
        self.assertLess(abs(float(audit["gap"][0])), 0.01 * sliding)

    def test_gap_reads_the_published_velocity(self):
        """A velocity written after the solve moves G by exactly the formula's change."""
        solver, _state = _rollout(_box_model(self.device, mu=0.5), steps=120)
        before = float(solver.read_audit()["gap"][0])
        count = int(solver.placement.mf_constraint_count.numpy()[0])
        load = float(
            np.sum(
                solver.placement.body_rows.impulses.numpy()[0, :count][
                    solver.placement.body_rows.row_type.numpy()[0, :count] == 0
                ]
            )
        )
        self.assertAlmostEqual(load, G0 * H, delta=1.0e-3 * G0 * H)
        dv = 0.01
        velocity = solver.v_out.numpy()
        velocity[2] += dv
        wp.copy(solver.v_out, wp.array(velocity, dtype=wp.float32, device=self.device))
        solver._audit.launch()
        self.assertAlmostEqual(float(solver.read_audit()["gap"][0]) - before, load * dv, delta=1.0e-4 * load * dv)

    def test_violation_reads_a_closing_row(self):
        """A closing velocity on a loaded contact shows as a dual violation of its size."""
        solver, _state = _rollout(_box_model(self.device, mu=0.5), steps=120)
        velocity = solver.v_out.numpy()
        velocity[2] -= 0.02
        wp.copy(solver.v_out, wp.array(velocity, dtype=wp.float32, device=self.device))
        solver._audit.launch()
        self.assertAlmostEqual(float(solver.read_audit()["violation"][0]), 0.02, delta=1.0e-4)

    def test_eliminated_rows_read_as_the_sweep_does(self):
        """Stewart's eliminated closures fold their rows into the contacts': ``D`` reads the closed problem, as a sweep."""
        scene = scenes.stewart_platform()
        with wp.ScopedDevice(self.device):
            model = scene.build(self.device)
            solver = SolverFeatherPGS(
                model,
                pgs_iterations=16,
                pgs_beta=0.1,
                dense_max_constraints=128,
                enable_bilateral_preelimination=True,
                pgs_audit=True,
            )
            pipeline = newton.CollisionPipeline(model, contact_matching=scene.contact_matching or "latest")
            contacts = pipeline.contacts()
            state, out, control = model.state(), model.state(), model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            worst = 0.0
            for step in range(240):
                pipeline.collide(state, contacts)
                state.clear_forces()
                if scene.control is not None:
                    scene.control(model, control, state, step)
                solver.step(state, out, control, contacts, scene.dt)
                state, out = out, state
                worst = max(worst, float(np.nanmax(solver.read_audit()["violation"])))
        self.assertTrue(solver.factor.eliminates)
        self.assertLess(worst, 1.0e-4)

    def test_regularized_rows_carry_their_compliance(self):
        """A regularized contact settles at ``r = 0`` only with its ``eps``; without it the gap would be the
        ``sum eps lambda^2`` this asserts it is far below."""
        solver, _state = _rollout(
            _box_model(self.device), steps=240, pgs_contact_regularization=0.1, pgs_iterations=128
        )
        audit = solver.read_audit()
        count = int(solver.placement.mf_constraint_count.numpy()[0])
        lam = solver.placement.body_rows.impulses.numpy()[0, :count]
        normal = solver.placement.body_rows.row_type.numpy()[0, :count] == 0
        diagonal = 1.0 / solver.placement.mf_eff_mass_inv.numpy()[0, :count]
        compliance = np.sum(0.1 * diagonal[normal] * lam[normal] ** 2)
        self.assertGreater(compliance, 1.0e-6)
        self.assertLess(abs(float(audit["gap"][0])), 0.02 * compliance)

    def test_patch_and_torsion_regions_close(self):
        """A patch region's pooled load and share of ``mu``, and a spin row's ``mu_t``, close G on a settled box."""
        for options in ({"friction_anchor_beta": 0.2}, {"friction_anchor_beta": 0.2, "contact_torsion": True}):
            with self.subTest(options=options):
                solver, _state = _rollout(_box_model(self.device), steps=120, pgs_iterations=128, **options)
                audit = solver.read_audit()
                self.assertLess(abs(float(audit["gap"][0])), 1.0e-5)

    def test_injected_work_prices_baumgarte(self):
        """A 5 cm overlap at ``beta = 0.2``: the rows push at ``beta phi / h`` = 2.4 m/s, and ``W`` is the work."""
        solver, _state = _rollout(_box_model(self.device, height=0.05, gap=0.1), steps=1, pgs_iterations=256)
        velocity = float(solver.v_out.numpy()[2])
        audit = solver.read_audit()
        self.assertAlmostEqual(velocity, 0.2 * 0.05 / H, delta=0.05)
        self.assertAlmostEqual(float(audit["work"][0]), velocity * velocity, delta=0.1)


@unittest.skipUnless(wp.is_cuda_available(), "needs CUDA")
class TestAuditCUDA(TestAudit):
    """The audit's laws on a GPU, and on the routes the CPU does not run."""

    device = "cuda:0"

    def _route(self, scene, route, sweeps):
        run = conformance.simulate(
            scene, route, self.device, options={"pgs_iterations": sweeps, "pgs_audit": True}, steps=60
        )
        return run.audit, run.routes["diagnostics"]["audit"]

    def test_routes_report_the_same_physics(self):
        """Each readable placement's gap falls with its budget, and closes where the budget converges."""
        for build, routes in (
            (scenes.box_stack, (conformance.MF_HELD,)),
            (scenes.legged, (conformance.MF_HELD, conformance.SPARSE, conformance.MF_PARALLEL)),
            (scenes.gripper, (conformance.LOCAL_OWNED,)),
            (scenes.hand, (conformance.PAIRED,)),
            (scenes.keyboard, (conformance.SPARSE_DIAGONAL, conformance.COMPACT_BOUNDARY)),
        ):
            scene = build()
            for route in routes:
                with self.subTest(scene=scene.name, route=route.name):
                    coarse, record = self._route(scene, route, 8)
                    fine, _record = self._route(scene, route, 512)
                    self.assertEqual(record["reason"], "Reads every row of every world.")
                    self.assertTrue(np.all(coarse["steps"] == 60.0))
                    self.assertTrue(np.all(fine["gap_sum"] <= coarse["gap_sum"] * 1.01 + 1.0e-7))
                    self.assertLess(float(fine["gap_sum"].max()), 1.0e-3)
                    self.assertLess(float(fine["violation_max"].max()), 1.0e-3)

    def test_worlds_without_stored_rows_read_nan(self):
        """A world whose rows exist only inside its sweep reads NaN and is named, not skipped in silence."""
        for build, route, phrase in (
            (scenes.legged, conformance.KEYED_PARALLEL, "keyed rows"),
            (scenes.gripper, conformance.LOCAL_PACKETS, "row packets"),
        ):
            with self.subTest(route=route.name):
                audit, record = self._route(build(), route, 16)
                self.assertTrue(np.all(np.isnan(audit["gap"])))
                self.assertIn(phrase, record["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

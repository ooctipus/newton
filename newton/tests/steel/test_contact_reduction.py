# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverSteel on body-pair-reduced contacts: the solver gate and settling conformance."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_contact_reduction_body_pairs import (
    DT,
    _cylinder_foot,
    _free_jointed_foot,
    _make_pipeline,
    _prismatic_jointed_foot,
    _speculative_ring_body,
)

STEEL_PROPAGATION_PATH = 2
RUN_TO_RUN = wp.DeterministicMode.RUN_TO_RUN


def _same_articulation_scissor(builder):
    """Build two overlapping sibling links in one non-free articulation."""
    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    base = builder.add_link()
    builder.add_shape_box(base, hx=0.04, hy=0.04, hz=0.04, cfg=no_collision)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=base,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
    ]
    for side in (-1.0, 1.0):
        link = builder.add_link()
        # The siblings overlap. Three slightly overlapping boxes on each link
        # make several box-box manifolds for one body pair, guaranteeing that
        # body-pair reduction has more than one manifold's worth of real work.
        for x in (-0.06, 0.0, 0.06):
            builder.add_shape_box(
                link,
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                hx=0.05,
                hy=0.04,
                hz=0.04,
            )
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=link,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.06, side * 0.02, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
                collision_filter_parent=False,
            )
        )
    builder.add_articulation(joints)
    return base


def _two_cluster_body(builder, spread=1.0):
    """One rigid body with two 9-sphere feet ``spread`` apart on the same plane.

    Nine contacts per cluster exceed the 7-slot budget, so each cluster is
    both fully represented (deepest + extremes) AND strictly reduced --
    the plank settle test asserts the latter to stay non-vacuous.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=2.0)
    for end in (-spread / 2.0, spread / 2.0):
        for i, j in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (0, 2), (1, 2), (2, 2)):
            builder.add_shape_sphere(
                body,
                xform=wp.transform(wp.vec3(end + 0.03 * i, 0.03 * j, 0.0), wp.quat_identity()),
                radius=0.01,
            )
    return body


def _undamped(solver):
    """Turn off the default angular damping of the free roots."""
    solver._rigid_body_angular_damping.zero_()
    return solver


def _foot_model(pos=(0.0, 0.0, 0.020)):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _cylinder_foot(builder, pos)
    builder.add_ground_plane()
    return builder.finalize(device=wp.get_device())


def _ring_model(device, penetration):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.rigid_gap = 0.005
    body = _speculative_ring_body(builder, (0.0, 0.0, 0.0015 - penetration))
    builder.add_ground_plane()
    return builder.finalize(device=device), body


class TestSteelBodyPairReductionGate(unittest.TestCase):
    """Steel configurations validated only on unreduced contacts reject reduced buffers."""

    def _foot_model(self):
        return _foot_model()

    def _foot_scene(self):
        return _foot_model((5.13, 5.07, 0.0149))

    def test_steel_compliance_rejects_reduced_contacts(self):
        """Reject reduced contacts when Steel's compliant contact law is on."""
        if not wp.get_device().is_cuda:
            self.skipTest("Steel contact compliance requires CUDA")
        model = self._foot_model()
        state_0, state_1 = model.state(), model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        self.assertTrue(contacts.rigid_contact_body_pair_reduced)
        solver = newton.solvers.SolverSteel(model, contact_compliance=True, friction_anchor_beta=0.0)
        with self.assertRaisesRegex(ValueError, "contact_compliance=True"):
            solver.step(state_0, state_1, model.control(), contacts, DT)

    def test_captured_warmstart_step_keeps_accepting_reduced_contacts(self):
        """Capture collide and a warm-started Steel step before any eager step, then step eagerly after replay.

        The reduced stamp is fixed at allocation, so replay after clear() cannot
        leave the buffer looking unreduced to the warm-start check.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        pipeline = _make_pipeline(model, True, contact_matching="latest")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverSteel(model, contact_warmstart=True, friction_anchor_beta=0.0)
        with wp.ScopedCapture(device) as capture:
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        contacts.clear()
        wp.capture_launch(capture.graph)

        raw = _make_pipeline(model, False)
        raw_contacts = raw.contacts()
        raw.collide(state_0, raw_contacts)
        reduced_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(reduced_count, 0)
        self.assertLess(reduced_count, int(raw_contacts.rigid_contact_count.numpy()[0]))
        solver.step(state_0, state_1, control, contacts, DT)
        self.assertTrue(np.isfinite(state_1.body_q.numpy()).all())


class TestSteelBodyPairReductionConformance(unittest.TestCase):
    """SolverSteel settles identically with and without body-pair reduction on every contact path."""

    def _settle(self, build_fn, make_solver, reduce_on, steps=240, *, deterministic=True):
        """Drop, settle, and return dynamics plus solver-path watermarks.

        The fell/contact guards make the comparison non-vacuous: a body that
        is accidentally kinematic, misconfigured, or not routed through the
        intended solver path neither falls nor proves anything by matching
        heights.
        The angular residual is the mean |omega| over the final quarter of the
        run -- ``body_qd`` is (linear, angular), so omega is components 3:6.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = build_fn(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        pipeline = _make_pipeline(model, reduce_on, deterministic=deterministic)
        contacts = pipeline.contacts()
        solver = make_solver(model)
        z0 = float(state_0.body_q.numpy()[body][2])
        peak_contacts = 0
        w_tail = []
        for k in range(steps):
            pipeline.collide(state_0, contacts)
            if k % 20 == 0:
                peak_contacts = max(peak_contacts, int(contacts.rigid_contact_count.numpy()[0]))
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            if k >= steps * 3 // 4 and k % 10 == 0:
                w_tail.append(float(np.abs(state_0.body_qd.numpy()[body][3:6]).max()))
        z_end = float(state_0.body_q.numpy()[body][2])
        watermarks = solver.constraint_row_watermarks() if hasattr(solver, "constraint_row_watermarks") else {}
        return z_end, z_end < z0 - 0.005, peak_contacts, float(np.mean(w_tail)), watermarks

    def _compare(
        self,
        build_fn,
        make_solver,
        tol=5e-4,
        w_tol=0.05,
        expected_row_path=None,
        *,
        deterministic=True,
    ):
        z_off, fell_off, contacts_off, w_off, rows_off = self._settle(
            build_fn,
            make_solver,
            False,
            deterministic=deterministic,
        )
        z_on, fell_on, contacts_on, w_on, rows_on = self._settle(
            build_fn,
            make_solver,
            True,
            deterministic=deterministic,
        )
        self.assertTrue(fell_off and fell_on, "body did not fall: the solver is not simulating it")
        self.assertGreater(contacts_off, 0)
        self.assertGreater(contacts_on, 0)
        self.assertLess(contacts_on, contacts_off, "the reduced run never actually compacted a contact manifold")
        if expected_row_path is not None:
            key = f"{expected_row_path}_high_water"
            self.assertGreater(rows_off[key], 0, f"unreduced run never exercised {expected_row_path} rows")
            self.assertGreater(rows_on[key], 0, f"reduced run never exercised {expected_row_path} rows")
            self.assertGreater(rows_on["contact_high_water"], 0, "solver saw no reduced rigid-contact rows")
        self.assertLess(abs(z_off - z_on), tol)
        # angular residual (true omega: body_qd components 3:6, tail-window
        # mean): reduced must settle as calm as unreduced up to a per-scene
        # floor. Individual supported scenes may pass an explicit measured
        # bound instead of the calm default.
        self.assertLess(w_on, max(2.0 * w_off, w_tol), f"reduced settle is rocking: |w| {w_on} vs {w_off}")

    def test_steel_nondeterministic_conformance(self):
        """Conform on the default unsorted collision pipeline.

        Sequential PGS observes row order, so the sorted conformance cases do
        not establish support for ``CollisionPipeline(deterministic=False)``.
        This exercises the serial solve with point friction and strict contact compaction.
        """
        self._compare(
            lambda b: _free_jointed_foot(b, (0.0, 0.0, 0.05)),
            lambda m: _undamped(newton.solvers.SolverSteel(m, friction_anchor_beta=0.0)),
            deterministic=False,
        )

    def test_steel_dynamic_dynamic_conformance(self):
        """Settle a free-jointed foot dropped onto another free-jointed foot.

        Covers articulated dynamic-dynamic contact: both the falling body and
        its support are simulated bodies, so the reduced set must preserve the
        body-body patch as well as the body-ground patches.
        """

        def build(b):
            _free_jointed_foot(b, (0.0, 0.0, 0.0175))
            return _free_jointed_foot(b, (0.0, 0.005, 0.09))

        self._compare(
            build, lambda m: _undamped(newton.solvers.SolverSteel(m, friction_anchor_beta=0.0)), tol=2e-3, w_tol=0.15
        )

    def test_steel_contact_mode_conformance(self):
        """Settle identically on/off in every Steel contact mode.

        supports_body_pair_reduced_contacts is class-wide, so the evidence must cover
        the dense articulated, free-body and propagation contact paths and both
        solve modes, not only the point-friction serial solve the other tests exercise.
        """
        if not wp.get_device().is_cuda:
            self.skipTest("Steel contact-mode conformance is a CUDA-only merge gate")

        # Use a non-free prismatic articulation for every propagation variant.
        configs = [
            ("serial", "immediate", 16, _prismatic_jointed_foot, "articulation"),
            ("serial", "immediate", None, _free_jointed_foot, "free_body"),
            ("serial", "propagation", 16, _prismatic_jointed_foot, "propagation"),
            ("colored", "propagation", 16, _prismatic_jointed_foot, "propagation"),
        ]
        # Dense, propagation, and colored propagation run 16 iterations. The
        # ``None`` entry deliberately omits iterations and asserts the
        # default (12), rather than testing a stale assumed default.
        # At 8 iterations the UNREDUCED redundant set does not converge (dense
        # settles 5 mm high, plain/colored propagation 4.8 mm high at 0.01980;
        # the REDUCED set lands at the true 0.01500 at every
        # iteration count) -- redundant near-parallel rows hurt PGS
        # conditioning, the same effect as the stack-collapse case, and
        # reduction itself is what removes it.  The comparison must be against
        # a converged baseline.
        for mode, response, iters, build_foot, row_path in configs:
            with self.subTest(solve_mode=mode, response=response):

                def make_solver(model, mode=mode, response=response, iters=iters, row_path=row_path):
                    kwargs = {}
                    if iters is not None:
                        kwargs["iterations"] = iters
                    solver = newton.solvers.SolverSteel(
                        model,
                        solve_mode=mode,
                        articulated_contact_response=response,
                        row_watermark=True,
                        **kwargs,
                    )
                    solver._rigid_body_angular_damping.zero_()
                    if iters is None:
                        self.assertEqual(solver._iterations, 12)
                    return solver

                self._compare(
                    lambda b, build_foot=build_foot: build_foot(b, (0.0, 0.0, 0.05)),
                    make_solver,
                    tol=2e-3 if row_path == "propagation" else 5e-4,
                    expected_row_path=row_path,
                )

    def test_steel_same_articulation_rows_are_nonvacuous(self):
        """Route a strictly reduced sibling-link manifold to propagation rows.

        The general propagation scenes are link-to-ground contacts. This pins
        the opt-in cross-response path for contacts whose two bodies belong to
        the same articulation, and proves both that reduction did work and
        that the resulting rows reached the intended CUDA kernel family.
        """
        if not wp.get_device().is_cuda:
            self.skipTest("the reduced same-articulation scene is a CUDA-only check")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        _same_articulation_scissor(builder)
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        raw_pipeline = _make_pipeline(model, False)
        raw_contacts = raw_pipeline.contacts()
        raw_pipeline.collide(state_0, raw_contacts)
        raw_count = int(raw_contacts.rigid_contact_count.numpy()[0])

        reduced_pipeline = _make_pipeline(model, True, body_pair_cell_size=10.0)
        reduced_contacts = reduced_pipeline.contacts()
        reduced_pipeline.collide(state_0, reduced_contacts)
        reduced_count = int(reduced_contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(raw_count, 0, "scissor scene emitted no sibling-link contacts")
        self.assertLess(reduced_count, raw_count, "same-articulation manifold was not reduced")

        solver = newton.solvers.SolverSteel(
            model,
            articulated_contact_response="propagation",
            propagation_same_articulation_rows=True,
            row_watermark=True,
        )
        solver._rigid_body_angular_damping.zero_()
        self.assertEqual(solver._iterations, 12)
        solver.step(state_0, state_1, model.control(), reduced_contacts, DT)
        self.assertTrue(np.isfinite(state_1.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_1.body_qd.numpy()).all())
        rows = solver.constraint_row_watermarks()
        self.assertGreater(rows["propagation_high_water"], 0)
        active_paths = solver._contact_path.numpy()[:reduced_count]
        routed = np.flatnonzero(active_paths == STEEL_PROPAGATION_PATH)
        self.assertGreater(routed.size, 0, "no reduced contact was routed to the propagation path")
        shape_body = model.shape_body.numpy()
        body_art = solver._body_to_articulation.numpy()
        shape0 = reduced_contacts.rigid_contact_shape0.numpy()[:reduced_count]
        shape1 = reduced_contacts.rigid_contact_shape1.numpy()[:reduced_count]
        for contact in routed:
            body0 = int(shape_body[shape0[contact]])
            body1 = int(shape_body[shape1[contact]])
            self.assertGreaterEqual(body0, 0)
            self.assertGreaterEqual(body1, 0)
            self.assertEqual(
                int(body_art[body0]),
                int(body_art[body1]),
                "the path assertion did not cover a same-articulation contact",
            )


class TestSteelBodyPairReductionSettling(unittest.TestCase):
    """Reduced contacts keep multi-patch, touchdown and pile settling on SolverSteel stable."""

    def _settle_plank(self, reduce_on):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = _two_cluster_body(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        pipeline = _make_pipeline(model, reduce_on)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverSteel(model, friction_anchor_beta=0.0)
        solver._rigid_body_angular_damping.zero_()
        for _ in range(240):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        stats = pipeline._body_pair_reducer.stats() if reduce_on else None
        return state_0.body_q.numpy()[body], stats

    def test_plank_settles_level(self):
        """Settle the two-cluster plank level, at the unreduced rest height.

        If either cluster lost its support points the plank would tilt about
        the other end. The invariant is against the unreduced pipeline: same
        rest height (within solver noise) and no pitch, on the same scene.
        """
        q_off, _ = self._settle_plank(False)
        q_on, stats = self._settle_plank(True)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "plank contacts were never reduced")
        self.assertLess(abs(float(q_on[4])), 0.02, "plank tilted -- a cluster lost support")
        self.assertLess(abs(float(q_on[2]) - float(q_off[2])), 1.5e-3)

    def test_steel_touchdown_capture(self):
        """Land a falling foot on Steel at the same rest height with and without reduction.

        This covers the exact defect class found on the walking humanoid: a
        misranked kept set leaves no load-bearing contact at touchdown and
        the body free-falls before a violent late landing.
        """

        def run(reduce_on):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            body = _free_jointed_foot(builder, (5.13, 5.07, 0.06))  # 3 cm drop
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            pipeline = _make_pipeline(model, reduce_on)
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverSteel(model, friction_anchor_beta=0.0)
            solver._rigid_body_angular_damping.zero_()
            zs = []
            for _ in range(240):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, DT)
                state_0, state_1 = state_1, state_0
                zs.append(float(state_0.body_q.numpy()[body][2]))
            stats = pipeline._body_pair_reducer.stats() if reduce_on else None
            return np.array(zs), stats

        z_off, _ = run(False)
        z_on, stats = run(True)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "touchdown contacts were never reduced")
        self.assertLess(float(z_off.min()), 0.04, "foot never fell: the solver is not simulating it")
        self.assertLess(abs(float(z_off[-30:].mean()) - float(z_on[-30:].mean())), 5e-4)
        # bounded touchdown transient: never punches through the resting height by > 2.5 mm
        self.assertGreater(float(z_on.min()), 0.015 - 2.5e-3)

    @unittest.skipUnless(wp.is_cuda_available(), "ring-body rest check runs on CUDA")
    def test_speculative_ring_body_rests_still_with_steel(self):
        """Rest the ring body under 4 x 16 Steel: reduced must be as still as unreduced."""
        device = wp.get_cuda_device()
        dt, substeps, steps = 0.05, 4, 60

        def settle(reduce_on):
            model, body = _ring_model(device, penetration=-0.0005)
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            pipeline = _make_pipeline(model, reduce_on, contact_matching="latest")
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverSteel(
                model,
                iterations=16,
                baumgarte=0.2,
                contact_regularization=0.01,
                contact_warmstart=True,
                friction_anchor_beta=0.2,
                contact_friction_gap_max=0.001,
            )
            speeds, rots = [], []
            q_ref = None
            for k in range(steps):
                for _ in range(substeps):
                    pipeline.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, dt / substeps)
                    state_0, state_1 = state_1, state_0
                if k == steps // 3:
                    q_ref = state_0.body_q.numpy()[body][3:7].copy()
                if k > steps // 3:
                    qd = state_0.body_qd.numpy()[body]
                    speeds.append(float(np.linalg.norm(qd[:3])))
                    q = state_0.body_q.numpy()[body][3:7]
                    cos_half = min(1.0, abs(float(np.dot(q, q_ref))))
                    rots.append(float(np.degrees(2.0 * np.arccos(cos_half))))
            stats = pipeline._body_pair_reducer.stats() if reduce_on else None
            return float(np.mean(speeds)), float(np.max(rots)), stats

        v_off, rot_off, _ = settle(False)
        v_on, rot_on, stats = settle(True)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "ring body contacts were never reduced")
        self.assertLess(v_off, 0.002, f"unreduced reference body is not at rest ({v_off * 1000:.1f} mm/s)")
        self.assertLess(v_on, 0.002, f"reduced body rocks at {v_on * 1000:.1f} mm/s (unreduced {v_off * 1000:.2f})")
        self.assertLess(rot_on, 0.5, f"reduced body drifted {rot_on:.2f} deg (unreduced {rot_off:.3f})")

    def test_verifier_zero_violations_settling(self):
        """Verify the implementation invariant over a full settling trajectory.

        Every collide re-checks: no discarded contact out-ranks a slot winner
        it was eligible for, and every kept registered contact matches a slot
        it won. Any nonzero counter is a slot race, clearing bug, or ranking
        regression.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (5.13, 5.07, 0.020))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        pipeline = _make_pipeline(model, True, body_pair_verify=True)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverSteel(model, friction_anchor_beta=0.0)
        solver._rigid_body_angular_damping.zero_()
        for _ in range(240):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        stats = pipeline._body_pair_reducer.stats()
        self.assertEqual(stats["invariant_violations"], 0)
        self.assertEqual(stats["probe_failures"], 0)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "verifier saw no discard")

    def test_property_random_piles(self):
        """Fuzz the invariant and stability on randomized multi-collider piles.

        Seeded random mixes of spheres, boxes, capsules, and cylinders are
        dropped into a pile -- geometry nobody hand-picked. Every free-jointed
        body has seven offset colliders, so the test cannot pass solely through
        the reducer's one-collider identity path. At least one frame must
        strictly reduce; the invariant verifier must stay clean; and the
        supported Steel consumer must remain finite and above the ground.
        """
        rng = np.random.default_rng(1234)
        for trial in range(1):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            bodies = []
            for k in range(10):
                # spawn strictly non-overlapping (max shape extent is 0.16 m):
                # interpenetrating spawns make BOTH pipelines ballistic and the
                # comparison chaotic rather than physical
                pos = (float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.15, 0.15)), 0.10 + 0.18 * k)
                body = builder.add_link(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
                bodies.append(body)
                kind = int(rng.integers(0, 4))
                radius = float(rng.uniform(0.012, 0.02))
                half_height = float(rng.uniform(0.012, 0.025))
                for j in range(7):
                    offset = wp.transform(wp.vec3((j - 3) * 2.2 * radius, 0.0, 0.0), wp.quat_identity())
                    if kind == 0:
                        builder.add_shape_sphere(body, xform=offset, radius=radius)
                    elif kind == 1:
                        builder.add_shape_box(body, xform=offset, hx=radius, hy=radius, hz=radius)
                    elif kind == 2:
                        builder.add_shape_capsule(body, xform=offset, radius=radius, half_height=half_height)
                    else:
                        builder.add_shape_cylinder(body, xform=offset, radius=radius, half_height=half_height)
                joint = builder.add_joint_free(parent=-1, child=body)
                builder.add_articulation([joint])
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            pipe_red = _make_pipeline(model, True, body_pair_verify=True)
            pipe_raw = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
            c_red, c_raw = pipe_red.contacts(), pipe_raw.contacts()
            # Seven colliders per body generate thousands of speculative rows.
            # Both consumers must retain them for this to test reduction stability.
            mf_capacity = 3 * max(c_red.rigid_contact_max, c_raw.rigid_contact_max)
            # Both piles are chaotic; run-to-run determinism keeps their peak speeds comparable between runs.
            solver = newton.solvers.SolverSteel(
                model, friction_anchor_beta=0.0, free_body_constraint_max=mf_capacity, deterministic=RUN_TO_RUN
            )
            solver._rigid_body_angular_damping.zero_()
            raw_not_less, strict_reduction_frames, peak_red = 0, 0, 0.0
            for _ in range(150):
                pipe_raw.collide(state_0, c_raw)
                pipe_red.collide(state_0, c_red)
                raw_count = int(c_raw.rigid_contact_count.numpy()[0])
                reduced_count = int(c_red.rigid_contact_count.numpy()[0])
                raw_not_less += int(raw_count >= reduced_count)
                strict_reduction_frames += int(raw_count > reduced_count)
                state_0.clear_forces()
                solver.step(state_0, state_1, control, c_red, DT)
                state_0, state_1 = state_1, state_0
                peak_red = max(peak_red, float(np.abs(state_0.body_qd.numpy()).max()))

            # Reference peak from the SAME pile and same solver family, driven
            # by unreduced contacts. An absolute bound flakes on a random pile,
            # whose peak speed can legitimately be large in either pipeline.
            raw_state_0, raw_state_1 = model.state(), model.state()
            newton.eval_fk(model, raw_state_0.joint_q, raw_state_0.joint_qd, raw_state_0)
            raw_control = model.control()
            raw_dynamics_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
            raw_dynamics_contacts = raw_dynamics_pipeline.contacts()
            raw_solver = newton.solvers.SolverSteel(
                model, friction_anchor_beta=0.0, free_body_constraint_max=mf_capacity, deterministic=RUN_TO_RUN
            )
            raw_solver._rigid_body_angular_damping.zero_()
            peak_raw = 0.0
            for _ in range(150):
                raw_dynamics_pipeline.collide(raw_state_0, raw_dynamics_contacts)
                raw_state_0.clear_forces()
                raw_solver.step(raw_state_0, raw_state_1, raw_control, raw_dynamics_contacts, DT)
                raw_state_0, raw_state_1 = raw_state_1, raw_state_0
                peak_raw = max(peak_raw, float(np.abs(raw_state_0.body_qd.numpy()).max()))

            solver.check_constraint_capacity()
            raw_solver.check_constraint_capacity()
            stats = pipe_red._body_pair_reducer.stats()
            self.assertEqual(stats["invariant_violations"], 0, f"trial {trial}")
            self.assertEqual(raw_not_less, 150, f"trial {trial}: reduction increased a count")
            self.assertGreater(strict_reduction_frames, 0, f"trial {trial}: reducer never removed a contact")
            body_q = state_0.body_q.numpy()
            qd = state_0.body_qd.numpy()
            self.assertTrue(np.isfinite(body_q).all() and np.isfinite(qd).all())
            self.assertLess(peak_red, max(3.0 * peak_raw, 10.0), f"trial {trial}: pile blew up")
            self.assertGreater(float(body_q[bodies, 2].min()), -0.06, f"trial {trial}: body tunneled")


if __name__ == "__main__":
    unittest.main()

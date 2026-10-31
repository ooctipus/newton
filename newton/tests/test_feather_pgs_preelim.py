# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for bilateral (mimic + connect) pre-elimination in SolverFeatherPGS."""

import dataclasses
import inspect
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_MIMIC,
)
from newton.tests import feather_pgs_scenes as scenes
from newton.tests.test_feather_pgs_connect import (
    _REL_P,
    _anchor_gaps,
    _build_carried_load,
    _build_four_bar,
    _loop_anchor_gap,
    _prescribe_carrier,
)


class TestPreeliminationSignature(unittest.TestCase):
    def test_legacy_joint_gap_position_is_preserved(self):
        """Bind a deprecated positional gap to the gap option without silently coercing it to a boolean."""
        parameters = list(inspect.signature(newton.solvers.SolverFeatherPGS).parameters.values())
        self.assertEqual(parameters[7].name, "joint_limit_activation_gap")
        arguments = [parameter.default for parameter in parameters[1:7]] + [0.123]
        model = newton.ModelBuilder().finalize(device="cpu")
        with self.assertWarnsRegex(DeprecationWarning, "'joint_limit_activation_gap' positionally"):
            solver = newton.solvers.SolverFeatherPGS(model, *arguments)
        self.assertEqual(solver.options.joint_limit_activation_gap, 0.123)
        self.assertIs(solver.options.enable_bilateral_preelimination, False)

    def test_every_bilateral_row_joins_its_block(self):
        """One law for every bilateral row: the selective mimic option is gone (pivoting drops dependent rows)."""
        parameters = inspect.signature(newton.solvers.SolverFeatherPGS).parameters
        self.assertNotIn("bilateral_preelimination_include_mimics", parameters)


def _run_four_bar(steps: int = 720, crank_target: float = 0.6, pgs_iterations: int = 2, **solver_kwargs):
    """Drive the connect-test four-bar and track the worst loop-anchor gap.

    Deliberately runs at a LOW iteration count: with iterative closure rows the
    anchor gap is convergence-limited (measured ~0.5 mm at 2 iterations), while
    pre-elimination enforces the closure independently of the sweep budget.
    """
    model = _build_four_bar().finalize()
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=pgs_iterations, pgs_beta=0.1, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    gap_max = 0.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
    return solver, model, state_0, gap_max


def _run_scene(scene: scenes.Scene, iterations: int, eliminate: bool = True):
    """Step a catalogue scene with the bilateral elimination (or the iterative closure rows); return the solver, every
    step's body positions and the largest contact impulse."""
    model = scene.build(wp.get_device())
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_iterations=iterations,
        pgs_beta=0.1,
        dense_max_constraints=128,
        enable_bilateral_preelimination=eliminate,
    )
    pipeline = newton.CollisionPipeline(model, contact_matching=scene.contact_matching or "latest")
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    positions, loaded = [], 0.0
    for step in range(scene.steps):
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        if scene.control is not None:
            scene.control(model, control, state_0, step)
        solver.step(state_0, state_1, control, contacts, scene.dt)
        state_0, state_1 = state_1, state_0
        positions.append(state_0.body_q.numpy()[:, :3].astype(np.float64))
        count = int(solver.placement.constraint_count.numpy()[0])
        contact_rows = solver.placement.row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_CONTACT
        if contact_rows.any():
            loaded = max(loaded, float(np.max(solver.placement.impulses.numpy()[0, :count][contact_rows])))
    return solver, np.array(positions), loaded


def _build_overcapacity_mixed_bilateral_model():
    """Build one articulation with three connect rows and six mimic rows."""
    builder = _build_four_bar()
    for index in range(6):
        builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0, label=f"mimic_{index}")
    return builder.finalize()


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSPreelimination(unittest.TestCase):
    def test_mimic_and_connect_rows_share_one_block(self):
        """An articulation's mimic and connect rows form one block, eliminated together."""
        builder = _build_four_bar()
        builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0)
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8, enable_bilateral_preelimination=True)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        for _ in range(32):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertTrue(solver._preelim_active)
        count = int(solver._preelim_nB.numpy()[0])
        slots = solver._preelim_slots.numpy()[:count]
        selected_types = solver.placement.row_type.numpy()[0, slots]
        self.assertEqual(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_MIMIC)), 1)
        self.assertGreater(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_CONNECT)), 0)

    def test_blocks_are_sized_from_the_model(self):
        """A block takes every bilateral row of its articulation, however many: three connect rows and six mimics
        (nine rows, past the former capacity of eight) are eliminated, and the pivot keeps one of the six identical
        mimics, which depend on each other."""
        model = _build_overcapacity_mixed_bilateral_model()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8, enable_bilateral_preelimination=True)
        self.assertTrue(solver._preelim_active)
        self.assertEqual(solver._elimination.rows, 9)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(16):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        count = int(solver._preelim_nB.numpy()[0])
        kept = solver.placement.row_type.numpy()[0, solver._preelim_slots.numpy()[:count]]
        self.assertEqual(int(np.count_nonzero(kept == PGS_CONSTRAINT_TYPE_MIMIC)), 1)
        self.assertGreater(int(np.count_nonzero(kept == PGS_CONSTRAINT_TYPE_CONNECT)), 0)

    def test_prescribed_parent_closures_are_eliminated(self):
        """A load pinned to a moving kinematic carrier by three ball closures (nine rows, the carrier's velocity in
        their right-hand sides) rides along with it under the elimination."""
        b, carrier, load, joints = _build_carried_load()
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(
            model, pgs_iterations=16, pgs_beta=0.2, enable_bilateral_preelimination=True
        )
        self.assertTrue(solver._preelim_active)
        articulation = wp.array([int(solver.forest.body_to_articulation.numpy()[carrier])], dtype=wp.int32)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        dt = 1.0 / 240.0
        v = np.array([0.3, 0.0, 0.1])
        omega = np.array([0.0, 0.0, 1.5])
        for i in range(240):
            _prescribe_carrier(model, solver, state_0, articulation, i * dt, v, omega)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
        _prescribe_carrier(model, solver, state_0, articulation, 240 * dt, v, omega)
        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
        expected = wp.transform_point(carrier_t, wp.vec3(*_REL_P))
        error = np.linalg.norm(bq[load, :3] - np.array([expected[0], expected[1], expected[2]]))
        self.assertLess(error, 3.0e-3, f"load lagged the carrier by {error:.4f} m")
        for gap in _anchor_gaps(model, state_0, joints, solver):
            self.assertLess(gap, 3.0e-3)

    def test_closure_exact_at_low_iterations(self):
        """Pre-elimination makes the loop closure iteration-independent.

        At 2 PGS iterations the iterative CONNECT rows cannot converge (the
        anchor gap plateaus near half a millimetre); the pre-eliminated solve
        must hold the closure 100x tighter through the same driven stroke.
        This is the regression-first differential: without the feature the
        ratio is 1 and the assertions below fail.
        """
        solver_off, _, _state_off, gap_off = _run_four_bar()
        self.assertFalse(solver_off._preelim_active)
        solver_on, _, state_on, gap_on = _run_four_bar(enable_bilateral_preelimination=True)
        self.assertTrue(solver_on._preelim_active)
        self.assertEqual(solver_on._preelim_count, 1)

        self.assertTrue(np.isfinite(state_on.body_q.numpy()).all())
        self.assertGreater(gap_off, 1.0e-4, "iterative baseline unexpectedly tight; differential is meaningless")
        self.assertLess(gap_on, 2.0e-5, f"pre-eliminated closure gap {gap_on:.2e} m not exact")
        self.assertLess(gap_on, 0.01 * gap_off, f"expected >100x tightening, got {gap_off / max(gap_on, 1e-12):.1f}x")

    @unittest.skip(
        "Obsolete drive-only fixture: dense non-contact rows deliberately cold-start; "
        "test_noncontact_dense_cache_is_cold_initialized covers that contract. "
        "A contact-bearing fixture is required to exercise warm-start projection ordering."
    )
    def test_dense_warmstart_preserves_projected_closure(self):
        """Project loaded contact history and an unsaturated drive kick at zero sweeps."""
        builder = _build_four_bar()
        # An unlimited drive stays in the augmented predictor, so reversing it
        # still changes velocity in the final step with no PGS drive sweeps.
        builder.joint_effort_limit[0] = float("inf")
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(-0.04, 0.0, 0.5), wp.quat_identity()),
            hx=0.02,
            hy=0.1,
            hz=0.1,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=2,
            pgs_beta=0.1,
            dense_max_constraints=128,
            enable_bilateral_preelimination=True,
            pgs_warmstart=True,
        )
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        for _ in range(120):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        count = int(solver.placement.constraint_count.numpy()[0])
        contact_rows = solver.placement.row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_CONTACT
        self.assertTrue(contact_rows.any())
        cache_peak = np.max(solver.placement.impulses.numpy()[0, :count][contact_rows])
        self.assertGreater(cache_peak, 1.0e-4, "loaded contact cache stayed empty")
        previous_predictor = solver.v_hat.numpy().copy()
        targets[0] = -0.6
        control.joint_target_q.assign(targets)
        solver.options = solver.laws.options = solver.factor.options = solver.placement.options = dataclasses.replace(
            solver.options, pgs_iterations=0
        )
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        gap = _loop_anchor_gap(model, state_1)
        self.assertTrue(solver._preelim_active)
        self.assertTrue(np.isfinite(state_1.body_q.numpy()).all())
        self.assertGreater(np.max(solver.placement.impulses.numpy()), 1.0e-4)
        self.assertGreater(np.linalg.norm(solver.v_hat.numpy() - previous_predictor), 1.0e-3)
        self.assertLess(gap, 2.0e-5, f"warm-started projection left a {gap:.2e} m closure gap")

    def test_loaded_contacts_keep_the_projected_closure(self):
        """Keep the closure through every sweep of loaded contacts, not only at the projection.

        The eliminated block corrects the other rows (``Z' = Z - Z_B S^-1 Z_B^T Z``): a contact's impulse then leaves
        ``J_B v`` alone, so two sweeps a step hold the closure as the projection set it, where uncorrected contact
        rows reopen it every step.
        """
        builder = _build_four_bar()
        builder.joint_effort_limit[0] = float("inf")
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(-0.04, 0.0, 0.5), wp.quat_identity()), hx=0.02, hy=0.1, hz=0.1
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=2,
            pgs_beta=0.1,
            dense_max_constraints=128,
            enable_bilateral_preelimination=True,
        )
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        worst = loaded = 0.0
        for _ in range(120):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
            worst = max(worst, _loop_anchor_gap(model, state_0))
            count = int(solver.placement.constraint_count.numpy()[0])
            contact_rows = solver.placement.row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_CONTACT
            if contact_rows.any():
                loaded = max(loaded, float(np.max(solver.placement.impulses.numpy()[0, :count][contact_rows])))
        self.assertTrue(solver._preelim_active)
        self.assertGreater(loaded, 0.1, "the contacts carried no load")
        self.assertLess(worst, 1.0e-5, f"loaded contacts opened the closure by {worst:.2e} m")

    def test_loaded_loops_converge_at_the_contacts_rate(self):
        """The elimination gives the contacts the closed-loop response, so a loop that carries contact load converges
        at the contacts' own rate, where closure rows in the sweep converge at the closure's coupling to them.

        The five-bar leg stands on its foot, which sits on the loop, and the Stewart platform's five closures
        (fifteen rows in one block) press it onto a table: 16 sweeps reach the converged trajectory (512 sweeps)
        within 1e-6 m (measured 0 and 1.5e-8 m; iterative closure rows 1.2e-3 and 7.3e-3 m). The four-bar is stopped
        at its rocker, so the crank's drive load (about 30 N m) crosses the closure into a four-point face contact
        whose split is statically indeterminate: that contact problem sets its 16-sweep error (3.1e-6 m, the same with
        the closure exact or regularized), 2000 times below the iterative closure rows' (7.3e-3 m). LC design §3.
        """
        for scene, bound in (
            (scenes.five_bar_leg(), 1.0e-6),
            (scenes.stewart_platform(), 1.0e-6),
            (scenes.loaded_four_bar(), 1.0e-5),
        ):
            with self.subTest(scene=scene.name):
                solver, fast, loaded = _run_scene(scene, 16)
                _, converged, _ = _run_scene(scene, 512)
                _, rows, _ = _run_scene(scene, 16, eliminate=False)
                self.assertTrue(solver._preelim_active)
                self.assertGreater(loaded, 1.0e-3, "the contact carried no load")
                error = float(np.abs(fast - converged).max())
                rows_error = float(np.abs(rows - converged).max())
                self.assertLess(error, bound, f"{scene.name}: 16 sweeps miss the converged trajectory by {error:.2e} m")
                self.assertGreater(rows_error, 100.0 * max(error, 1.0e-9), f"{scene.name}: rows {rows_error:.2e} m")

    def test_mechanism_behavior_preserved(self):
        """The rocker still tracks the crank through the closed loop (parallel four-bar)."""
        _, _, state, _ = _run_four_bar(enable_bilateral_preelimination=True, pgs_iterations=16)
        q = state.joint_q.numpy()
        self.assertAlmostEqual(q[2], q[0], delta=0.08)
        self.assertGreater(abs(q[2]), 0.3, "rocker did not move — loop not transmitting")

    def test_rows_remain_allocated(self):
        """B rows keep their dense slots (layout/warm-start stability); they are neutralized, not removed."""
        solver, _, _, _ = _run_four_bar(steps=60, enable_bilateral_preelimination=True)
        counts = solver.placement.constraint_count.numpy()
        rows = solver.placement.row_type.numpy()
        seen = set()
        for w in range(rows.shape[0]):
            seen.update(rows[w, : counts[w]].tolist())
        self.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)


class TestFeatherPGSPreeliminationCPU(unittest.TestCase):
    def test_cpu_pre_eliminates_the_closure(self):
        """Pre-eliminate the four-bar's closure on the CPU, whose matrix-free sweeps carry it as CUDA's do.

        Nothing warns, and at two sweeps the anchors stay within 1e-5 m over a second, against about 5e-4 m with
        the closure's iterative rows.
        """
        model = _build_four_bar().finalize(device="cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=16, enable_bilateral_preelimination=True)
        self.assertTrue(solver._preelim_active)
        self.assertFalse([w for w in caught if "requires CUDA" in str(w.message)])
        with wp.ScopedDevice("cpu"):
            _, _, _, gap = _run_four_bar(steps=240, enable_bilateral_preelimination=True)
        self.assertLess(gap, 1.0e-5)


if __name__ == "__main__":
    unittest.main()

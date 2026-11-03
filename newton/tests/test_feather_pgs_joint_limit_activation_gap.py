# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_SPEED,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import prefix_warp_rows


def _built_rows(q: float, *, gap: float, lower: float = -1.0, upper: float = 1.0, resolved: bool = False):
    """One revolute coordinate's joint-limit rows (the general route's warp, on the CPU): their J and phi."""
    rows = prefix_warp_rows(
        "cpu",
        1,
        1,
        8,
        resolved=[int(resolved)],
        lower=[lower],
        upper=[upper],
        q=[q],
        activation_gap=gap,
        beta=0.2,
        cfm=0.0,
    )
    count = int(rows["count"][0])
    return rows["J"][0, :count, 0].tolist(), rows["phi"][0, :count].tolist()


def _make_phase_layout_run(
    device,
    *,
    pgs_iterations=0,
    enable_joint_velocity_limits=True,
):
    """Build a deterministic scene containing every phase-bounded row family."""
    builder = newton.ModelBuilder(gravity=0.0)
    SolverFeatherPGS.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.05), wp.quat_identity()),
        limit_lower=-0.1,
        limit_upper=0.1,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = 0.2

    free_body = builder.add_body(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.05), wp.quat_identity()),
        custom_attributes={
            "rigid_body_max_linear_velocity": 1.0,
            "rigid_body_max_angular_velocity": 1.0,
        },
    )
    builder.add_shape_box(free_body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    velocity_limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
    velocity_limits[0] = 0.25
    model.joint_velocity_limit.assign(velocity_limits)
    solver = SolverFeatherPGS(
        model,
        enable_joint_limits=True,
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        velocity_limit_activation_fraction=0.5,
        dense_max_constraints=64,
        mf_max_constraints=64,
        pgs_iterations=pgs_iterations,
        pgs_warmstart=False,
    )
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0] = 0.5
    free_joint = int(np.flatnonzero(model.joint_type.numpy() == int(JointType.FREE))[0])
    free_dof = int(model.joint_qd_start.numpy()[free_joint])
    joint_qd[free_dof] = 1.5
    joint_qd[free_dof + 3] = 1.5
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    return model, solver, state_0, state_1, model.control(), model.contacts()


def _step_once(run):
    """Build all constraint rows for one deterministic step."""
    model, solver, state_0, state_1, control, contacts = run
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
    return solver


def _assert_dense_phase_layout(
    test_case: unittest.TestCase, solver: SolverFeatherPGS, *, expect_joint_velocity_limits: bool = True
):
    """Assert the three dense row-family partitions are ordered and populated."""
    dense_count = int(solver.placement.constraint_count.numpy()[0])
    dense_types = solver.placement.row_type.numpy()[0, :dense_count]
    phase_zero_end, phase_one_end = solver.placement.dense_phase_bounds.numpy()[0]
    limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    contact_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_CONTACT)
    friction_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_FRICTION)

    test_case.assertGreater(limit_rows.size, 0, "expected an active joint position-limit row")
    if expect_joint_velocity_limits:
        test_case.assertGreater(velocity_limit_rows.size, 0, "expected active joint velocity-limit rows")
    else:
        test_case.assertEqual(velocity_limit_rows.size, 0)
    test_case.assertGreater(contact_rows.size, 0, "expected dense contact rows")
    test_case.assertGreater(friction_rows.size, 0, "expected dense friction rows")
    test_case.assertTrue(np.all(limit_rows < phase_zero_end), (limit_rows, phase_zero_end))
    if expect_joint_velocity_limits:
        test_case.assertTrue(
            np.all((phase_zero_end <= velocity_limit_rows) & (velocity_limit_rows < phase_one_end)),
            (velocity_limit_rows, phase_zero_end, phase_one_end),
        )
    test_case.assertTrue(np.all(contact_rows >= phase_one_end), (contact_rows, phase_one_end))
    test_case.assertTrue(np.all(friction_rows >= phase_one_end), (friction_rows, phase_one_end))


def _assert_mf_phase_layout(test_case: unittest.TestCase, solver: SolverFeatherPGS):
    """Assert MF contacts precede the populated rigid velocity-limit tail."""
    mf_count = int(solver.placement.mf_constraint_count.numpy()[0])
    mf_types = solver.placement.body_rows.row_type.numpy()[0, :mf_count]
    mf_contact_end = int(solver.placement.mf_contact_rows_end.numpy()[0])
    contact_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_CONTACT)
    friction_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_FRICTION)
    velocity_limit_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_SPEED)

    test_case.assertGreater(contact_rows.size, 0, "expected matrix-free contact rows")
    test_case.assertGreater(friction_rows.size, 0, "expected matrix-free friction rows")
    test_case.assertGreater(velocity_limit_rows.size, 0, "expected active rigid-body velocity-limit rows")
    test_case.assertTrue(np.all(contact_rows < mf_contact_end), (contact_rows, mf_contact_end))
    test_case.assertTrue(np.all(friction_rows < mf_contact_end), (friction_rows, mf_contact_end))
    test_case.assertTrue(np.all(velocity_limit_rows >= mf_contact_end), (velocity_limit_rows, mf_contact_end))
    test_case.assertLess(mf_contact_end, mf_count)


def _limit_law(q: np.ndarray, lower: np.ndarray, upper: np.ndarray, gap: float) -> list:
    """The joint-limit rows' phi the row law gives (:func:`.row_laws.joint_limit_side`), in row order: each coordinate's
    lower side (``q - lower``), then its upper side (``upper - q``), where its bound is finite and ``q`` within ``gap``."""
    phi = []
    for value, low, high in zip(q.astype(np.float32), lower.astype(np.float32), upper.astype(np.float32), strict=True):
        if np.isfinite(low) and value <= low + np.float32(gap):
            phi.append(value - low)
        if np.isfinite(high) and value >= high - np.float32(gap):
            phi.append(high - value)
    return phi


def _run_joint_limit_trajectory():
    """Run a short trajectory; each step's joint-limit rows against the row law at the step's input coordinates."""
    model, solver, state_in, state_out, control, contacts = _make_phase_layout_run("cuda:0", pgs_iterations=8)
    revolute = np.flatnonzero(model.joint_type.numpy() == int(JointType.REVOLUTE))
    coordinates = model.joint_q_start.numpy()[revolute]
    dofs = model.joint_qd_start.numpy()[revolute]
    samples = []
    for _ in range(4):
        q = state_in.joint_q.numpy()[coordinates]
        state_in.clear_forces()
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        dense_count = int(solver.placement.constraint_count.numpy()[0])
        row_type = solver.placement.row_type.numpy()[0, :dense_count]
        limit_rows = np.flatnonzero(row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
        samples.append(
            {
                "phi": solver.placement.phi.numpy()[0, limit_rows].tolist(),
                "law": _limit_law(
                    q,
                    model.joint_limit_lower.numpy()[dofs],
                    model.joint_limit_upper.numpy()[dofs],
                    solver.options.joint_limit_activation_gap,
                ),
                "impulses": solver.placement.impulses.numpy()[0, limit_rows].copy(),
            }
        )
        state_in, state_out = state_out, state_in
    return samples


class TestFeatherPGSJointLimitActivationGap(unittest.TestCase):
    def test_resolved_world_skips_active_signed_limits(self):
        """The selector owns feasibility; the allocator must bypass either active side."""
        for q in (-1.2, 1.2):
            with self.subTest(q=q):
                self.assertEqual(len(_built_rows(q, gap=0.0)[0]), 1)
                self.assertEqual(_built_rows(q, gap=0.0, resolved=True), ([], []))

    def test_dense_row_families_respect_phase_bounds(self):
        """Keep active position limits before the CPU dense phase-0 bound."""
        solver = _step_once(_make_phase_layout_run("cpu", enable_joint_velocity_limits=False))
        _assert_dense_phase_layout(self, solver, expect_joint_velocity_limits=False)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free row-family layout requires CUDA")
    def test_combined_row_families_respect_phase_bounds(self):
        """Keep dense and matrix-free row families inside their phase ranges."""
        solver = _step_once(_make_phase_layout_run("cuda:0"))
        _assert_dense_phase_layout(self, solver)
        _assert_mf_phase_layout(self, solver)

    @unittest.skipUnless(wp.is_cuda_available(), "the matrix-free row families require CUDA")
    def test_limit_rows_follow_the_law_over_a_trajectory(self):
        """Give every step the joint-limit rows the row law gives at its input coordinates, with bitwise phi."""
        samples = _run_joint_limit_trajectory()
        for step, sample in enumerate(samples):
            self.assertEqual(sample["phi"], sample["law"], f"limit rows differed at step {step}")
        self.assertGreater(len(samples[0]["phi"]), 0)
        self.assertGreater(float(np.max(np.abs(samples[0]["impulses"]))), 0.0)

    def test_solver_rejects_invalid_joint_limit_activation_gap(self):
        model = newton.ModelBuilder().finalize()

        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "joint_limit_activation_gap"):
                    SolverFeatherPGS(model, joint_limit_activation_gap=value)

    def test_finite_gap_builds_only_near_limit_rows(self):
        self.assertEqual(_built_rows(0.0, gap=0.2), ([], []))

        jacobian, phi = _built_rows(-0.85, gap=0.2)
        self.assertEqual(jacobian, [1.0])
        self.assertAlmostEqual(phi[0], 0.15, places=6)

        jacobian, phi = _built_rows(0.85, gap=0.2)
        self.assertEqual(jacobian, [-1.0])
        self.assertAlmostEqual(phi[0], 0.15, places=6)

    def test_finite_gap_does_not_activate_unlimited_sentinel_limits(self):
        self.assertEqual(_built_rows(0.0, gap=0.2, lower=-MAXVAL, upper=MAXVAL), ([], []))

    def test_infinite_gap_preserves_historical_always_allocate_behavior(self):
        jacobian, phi = _built_rows(0.0, gap=float("inf"))

        self.assertEqual(jacobian, [1.0, -1.0])
        self.assertEqual(phi, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

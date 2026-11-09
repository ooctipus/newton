# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row-activation gates: which rows a step builds, read from the step's data.

A joint-limit row exists where its coordinate is within joint_limit_activation_gap of a finite bound, a velocity-limit
row where its speed is past velocity_limit_activation_fraction of its limit; a row that may act at the predictor marks
its world's acts word. The rows the general route's warp builds are checked against the row law on both devices.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.joint_table import FreeBodies
from newton._src.solvers.feather_pgs.row_kernels import allocate_rigid_velocity_limit_slots
from newton._src.solvers.feather_pgs.row_laws import ROW_JOINT_LIMIT
from newton._src.solvers.feather_pgs.row_records import DenseRows
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import PREFIX_RECORDS, prefix_warp_rows

# ---------------------------------------------------------------------------------------------------------------------
# joint_limit_activation_gap
# ---------------------------------------------------------------------------------------------------------------------


def _built_rows(q: float, *, gap: float, lower: float = -1.0, upper: float = 1.0):
    """One revolute coordinate's joint-limit rows (the general route's warp, on the CPU): their J and phi."""
    rows = prefix_warp_rows(
        "cpu",
        1,
        1,
        8,
        lower=[lower],
        upper=[upper],
        q=[q],
        activation_gap=gap,
        beta=0.2,
        cfm=0.0,
    )
    count = int(rows["count"][0])
    return rows["J"][0, :count, 0].tolist(), rows["phi"][0, :count].tolist()


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


def _limit_scene(
    device,
    *,
    pgs_iterations=0,
    enable_joint_velocity_limits=True,
):
    """A limited revolute link past its limit and a speed-limited free box on the ground: every row family."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
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


def _run_joint_limit_trajectory():
    """Run a short trajectory; each step's joint-limit rows against the row law at the step's input coordinates."""
    model, solver, state_in, state_out, control, contacts = _limit_scene("cuda:0", pgs_iterations=8)
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
        limit_rows = np.flatnonzero(row_type == ROW_JOINT_LIMIT)
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


# ---------------------------------------------------------------------------------------------------------------------
# joint_limit_warp
#
# The general route's joint-limit warp (:func:`.rows.joint_limit_warp_kernel`): one text on both devices, a warp per
# articulation on CUDA and a thread per articulation on the CPU (:func:`.rows.prefix_shape`).
# ---------------------------------------------------------------------------------------------------------------------


class TestFeatherPGSJointLimitWarp(unittest.TestCase):
    def test_acts_word_marks_worlds_whose_limits_may_act(self):
        """A world's acts word is set where an active side's residual at the predictor ``w v_hat + rhs`` is not above
        the separation margin (:func:`.row_laws.separates`), and the rows stay those of no word, on every device."""
        arts, size, dt = 257, 6, 0.01
        rng = np.random.default_rng(5)
        q = rng.uniform(-1.2, 1.2, arts * size)
        vhat = rng.uniform(-40.0, 40.0, arts * size) * (rng.random(arts * size) < 0.2)
        bounds = {"lower": np.full(arts * size, -1.0), "upper": np.full(arts * size, 1.0)}
        law = {**bounds, "q": q, "activation_gap": 0.25, "beta": 0.2, "cfm": 1.0e-6, "dt": dt}
        active, phi, sign = np.stack([q <= -0.75, q >= 0.75]), np.stack([q + 1.0, 1.0 - q]), np.array([[1.0], [-1.0]])
        rhs = np.where(phi < 0.0, 0.2 * phi, phi) / dt
        margin = 1.0e-5 + 2.0**-20 * (np.abs(vhat) + np.abs(rhs))
        acting = np.any(active & (sign * vhat + rhs < margin), axis=0).reshape(arts, size).any(axis=1)
        self.assertTrue(acting.any() and not acting.all())
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                plain = prefix_warp_rows(device, arts, size, 2 * size, **law)
                rows = prefix_warp_rows(device, arts, size, 2 * size, vhat=vhat, acts=np.zeros(arts), **law)
                np.testing.assert_array_equal(rows["acts"], acting)
                for name in ("count", "J", *PREFIX_RECORDS):
                    np.testing.assert_array_equal(rows[name], plain[name], err_msg=name)


# ---------------------------------------------------------------------------------------------------------------------
# velocity_limit_activation_fraction
# ---------------------------------------------------------------------------------------------------------------------


def _joint_velocity_rows(qd: float, *, fraction: float, qdot_max: float = 1.0):
    """One revolute coordinate's joint velocity-limit rows (the general route's warp, on the CPU): their J signs,
    targets and count."""
    rows = prefix_warp_rows(
        "cpu", 1, 1, 8, family="velocity", velocity_limit=[qdot_max], vhat=[qd], activation=fraction, cfm=0.0
    )
    count = int(rows["count"][0])
    return rows["J"][0, :count, 0].tolist(), rows["target"][0, :count].tolist(), count


def _allocated_rigid_velocity_slots(qd6, *, fraction: float, lin_limit: float = 1.0, ang_limit: float = 1.0, acts=None):
    device = "cpu"
    acts = wp.zeros(0 if acts is None else 1, dtype=wp.int32, device=device)
    rigid_velocity_limit_slot = wp.full((2,), 91, dtype=wp.int32, device=device)
    mf_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_rigid_velocity_limit_slots,
        dim=1,
        inputs=[
            field_group.upload(
                FreeBodies,
                device,
                free_rigid_body_indices=[0],
                body_to_articulation=[0],
                art_to_world=[0],
                is_free_rigid=[1],
                max_linear_velocity=[lin_limit],
                max_angular_velocity=[ang_limit],
            ),
            wp.array([int(BodyFlags.DYNAMIC)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array(list(qd6), dtype=wp.float32, device=device),
            fraction,
            64,
            0,
        ],
        outputs=[
            acts,
            rigid_velocity_limit_slot,
            mf_slot_counter,
            wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
            # The rows' records, one line of 64 (the speed law's record of each reserved row).
            field_group.upload(
                DenseRows,
                device,
                **{name: np.zeros((1, 64), np.int32) for name in ("row_type", "row_parent")},
                **{name: np.zeros((1, 64), np.float32) for name in ("row_mu", "row_beta", "phi", "target")},
                sides=np.zeros((1, 64, 2), np.int32),
                first=[0],
            ),
            wp.zeros((1, 64), dtype=wp.int32, device=device),
        ],
        device=device,
    )
    if acts.shape[0]:
        return rigid_velocity_limit_slot.numpy().tolist(), int(mf_slot_counter.numpy()[0]), int(acts.numpy()[0])
    return rigid_velocity_limit_slot.numpy().tolist(), int(mf_slot_counter.numpy()[0])


class TestFeatherPGSVelocityLimitActivationFraction(unittest.TestCase):
    def test_solver_rejects_invalid_velocity_limit_activation_fraction(self):
        model = newton.ModelBuilder().finalize()

        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "velocity_limit_activation_fraction"):
                    SolverFeatherPGS(model, velocity_limit_activation_fraction=value)

    def test_solver_accepts_and_defaults_velocity_limit_activation_fraction(self):
        model = newton.ModelBuilder().finalize()

        self.assertEqual(SolverFeatherPGS(model).options.velocity_limit_activation_fraction, 0.0)
        self.assertEqual(
            SolverFeatherPGS(model, velocity_limit_activation_fraction=0.7).options.velocity_limit_activation_fraction,
            0.7,
        )

    def test_fraction_zero_preserves_always_allocate_joint_rows(self):
        # fraction == 0.0 must match the historical behavior exactly: both
        # sides of every finitely limited DOF allocate, in the same order,
        # regardless of the current joint velocity.
        for qd in (0.0, 0.5, -2.0):
            with self.subTest(qd=qd):
                signs, targets, count = _joint_velocity_rows(qd, fraction=0.0)
                self.assertEqual(signs, [1.0, -1.0])
                self.assertEqual(targets, [-1.0, -1.0])
                self.assertEqual(count, 2)

    def test_fraction_gates_static_joint_dof_to_zero_rows(self):
        self.assertEqual(_joint_velocity_rows(0.0, fraction=0.9), ([], [], 0))
        self.assertEqual(_joint_velocity_rows(0.85, fraction=0.9), ([], [], 0))

    def test_joint_dof_past_threshold_allocates_rows_same_step(self):
        # The gate reads the same-step joint_qd input, so a DOF driven past
        # the threshold gets its row pair allocated in that very step.
        for qd in (0.95, -0.95, 1.5):
            with self.subTest(qd=qd):
                signs, _targets, count = _joint_velocity_rows(qd, fraction=0.9)
                self.assertEqual(signs, [1.0, -1.0])
                self.assertEqual(count, 2)

    def test_fraction_zero_allocates_every_limited_speed_row(self):
        """Allocate the linear and the angular speed row of a limited free body whatever its velocity."""
        slots, count = _allocated_rigid_velocity_slots([0.0] * 6, fraction=0.0)
        self.assertEqual(slots, [0, 1])
        self.assertEqual(count, 2)
        slots, count = _allocated_rigid_velocity_slots([0.0] * 6, fraction=0.0, lin_limit=float("inf"))
        self.assertEqual(slots, [-1, 0])
        self.assertEqual(count, 1)

    def test_acts_word_marks_a_speed_row_that_may_act(self):
        """A reserved speed row marks its world's acts word unless ``limit - |v|`` at the predictor is above the
        separation margin (:func:`.row_laws.separates`); a row the fraction leaves out marks nothing."""
        for qd6, fraction, acts in (
            ([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, 0),
            ([0.0, 0.0, 0.0, 0.0, 1.2, 0.0], 0.0, 1),
            ([0.6, 0.8, 0.0, 0.0, 0.0, 0.0], 0.0, 1),
            ([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], 0.95, 0),
        ):
            with self.subTest(qd6=qd6, fraction=fraction):
                self.assertEqual(_allocated_rigid_velocity_slots(qd6, fraction=fraction, acts=True)[2], acts)

    def test_fraction_gates_static_rigid_body_to_zero_rows(self):
        """Reserve no speed row for a free body at rest under a positive fraction."""
        slots, count = _allocated_rigid_velocity_slots([0.0] * 6, fraction=0.9)
        self.assertEqual(slots, [-1, -1])
        self.assertEqual(count, 0)

    def test_rigid_speed_past_threshold_allocates_row_same_step(self):
        """Gate a speed row on the magnitude of its velocity, not on one component.

        Each component of the linear velocity (0.6, 0.6, 0.2) is below 90% of the limit, its magnitude (0.87)
        is not; the angular velocity is at rest.
        """
        slots, count = _allocated_rigid_velocity_slots([0.6, 0.6, 0.2, 0.0, 0.0, 0.0], fraction=0.85)
        self.assertEqual(slots, [0, -1])
        self.assertEqual(count, 1)
        slots, count = _allocated_rigid_velocity_slots([0.0, 0.0, 0.0, -0.7, 0.0, 0.7], fraction=0.9)
        self.assertEqual(slots, [-1, 0])
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

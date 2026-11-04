# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row-activation gates: which rows a step builds, read from the step's data.

A joint-limit row exists where its coordinate is within joint_limit_activation_gap of a finite bound, a velocity-limit
row where its speed is past velocity_limit_activation_fraction of its limit, and a resolved world builds none. The
rows the general route's warp builds are checked against the row law on both devices.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.joint_table import FreeBodies
from newton._src.solvers.feather_pgs.row_kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    allocate_rigid_velocity_limit_slots,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import PREFIX_RECORDS, prefix_warp_rows

# ---------------------------------------------------------------------------------------------------------------------
# joint_limit_activation_gap
# ---------------------------------------------------------------------------------------------------------------------


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


def _rows(device, resolved=None, articulation_count=257, size=6):
    """One D6 joint per articulation and world, with positions drawn around its [-1, 1] limits: each world's row count,
    Jacobian rows and records."""
    dof_count = articulation_count * size
    rng = np.random.default_rng(42)
    rows = prefix_warp_rows(
        device,
        articulation_count,
        size,
        2 * size,
        resolved=resolved,
        lower=np.full(dof_count, -1.0),
        upper=np.full(dof_count, 1.0),
        q=rng.uniform(-1.2, 1.2, dof_count),
        activation_gap=0.25,
        beta=0.2,
        cfm=1.0e-6,
    )
    return {name: rows[name] for name in ("count", "J", *PREFIX_RECORDS)}


class TestFeatherPGSJointLimitWarp(unittest.TestCase):
    def test_resolved_worlds_take_no_rows(self):
        """An all-zero mask builds the same rows as the empty mask, and a resolved world builds none, on every device.

        Mixed decisions share a block: a resolved group must leave its neighbors' ballots and row order unchanged.
        """
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                unmasked = _rows(device)
                self.assertGreater(int(unmasked["count"].sum()), 0)
                for name, actual in _rows(device, np.zeros(257)).items():
                    np.testing.assert_array_equal(actual, unmasked[name], err_msg=name)
                resolved = np.arange(257) % 3 == 0
                for name, actual in _rows(device, resolved).items():
                    expected = unmasked[name].copy()
                    expected[resolved] = 0
                    np.testing.assert_array_equal(actual, expected, err_msg=name)


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


def _allocated_rigid_velocity_slots(
    qd6, *, fraction: float, lin_limit: float = 1.0, ang_limit: float = 1.0, resolved: bool = False
):
    device = "cpu"
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
            wp.array([int(resolved)], dtype=wp.int32, device=device),
        ],
        outputs=[rigid_velocity_limit_slot, mf_slot_counter, wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device)],
        device=device,
    )
    return rigid_velocity_limit_slot.numpy().tolist(), int(mf_slot_counter.numpy()[0])


class TestFeatherPGSVelocityLimitActivationFraction(unittest.TestCase):
    def test_resolved_world_clears_stale_rigid_slots_without_allocating(self):
        """Resolved worlds leave no stale velocity-limit publication metadata."""
        slots, count = _allocated_rigid_velocity_slots([0.0] * 6, fraction=0.0, resolved=True)
        self.assertEqual(slots, [-1, -1])
        self.assertEqual(count, 0)

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

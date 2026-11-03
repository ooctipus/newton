# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton._src.sim.enums import BodyFlags
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.joint_table import FreeBodies
from newton._src.solvers.feather_pgs.kernels import allocate_rigid_velocity_limit_slots
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import prefix_warp_rows


def _joint_velocity_rows(qd: float, *, fraction: float, qdot_max: float = 1.0):
    """One revolute coordinate's joint velocity-limit rows (the general route's warp, on the CPU): their J signs,
    targets and count."""
    rows = prefix_warp_rows(
        "cpu", 1, 1, 8, velocity=True, velocity_limit=[qdot_max], vhat=[qd], activation=fraction, cfm=0.0
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

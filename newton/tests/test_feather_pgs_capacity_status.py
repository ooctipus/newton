# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test sticky overflow visibility independently of opt-in row telemetry."""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels
from newton.solvers import SolverFeatherPGS


def _allocator_arrays(capacity, *, route=0, count=205, device="cpu"):
    """Create a complete contact prefix with three rows per contact."""

    def integers(values):
        return wp.array(values, dtype=wp.int32, device=device)

    def zeros(size):
        return wp.zeros(size, dtype=wp.int32, device=device)

    length = max(count, 1)
    return {
        "contact_count": integers([count]),
        "total_num_threads": 1,
        "contact_shape0": zeros(length),
        "contact_shape1": integers([-1] * length),
        "contact_point0": wp.zeros(length, dtype=wp.vec3, device=device),
        "contact_point1": wp.zeros(length, dtype=wp.vec3, device=device),
        "contact_normal": wp.array([[0.0, 0.0, -1.0]] * length, dtype=wp.vec3, device=device),
        "contact_thickness0": wp.zeros(length, dtype=float, device=device),
        "contact_thickness1": wp.zeros(length, dtype=float, device=device),
        "body_q": wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
        "shape_transform": wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
        "shape_body": integers([0]),
        "body_to_articulation": integers([0]),
        "art_to_world": integers([0]),
        "articulation_response_dof_count": integers([1]),
        "body_flags": zeros(1),
        "body_has_response_dofs": integers([1]),
        "is_free_rigid": integers([int(route == 1)]),
        "has_free_rigid": int(route == 1),
        "propagation_articulated_contacts": int(route == 2),
        "propagation_same_articulation": 0,
        "propagation_free_free": 0,
        "contact_gap_gate": 0.0,
        "same_articulation_contact_gap_gate": 0.0,
        "articulation_pair_contact_gap_gate": 0.0,
        "max_constraints": capacity,
        "mf_max_constraints": capacity,
        "propagation_max_constraints": capacity,
        "enable_friction": 1,
        "contact_friction_gap_threshold": float("inf"),
        "contact_friction_anchor_limit": 0,
        "contact_friction_articulation_pairs_only": 0,
        "row_capacity_telemetry": 0,
        "resolved_worlds": wp.empty(0, dtype=wp.int32, device=device),
        "contact_world": zeros(length),
        "contact_slot": integers([-9] * length),
        "contact_art_a": zeros(length),
        "contact_art_b": zeros(length),
        "world_slot_counter": zeros(1),
        "contact_path": integers([-9] * length),
        "mf_slot_counter": zeros(1),
        "propagation_slot_counter": zeros(1),
        "dense_contact_world_flag": zeros(1),
        "contact_slots_needed": zeros(length),
        "dense_dropped_contact_rows": zeros(1),
        "mf_dropped_contact_rows": zeros(1),
        "propagation_dropped_contact_rows": zeros(1),
        "capacity_status": zeros(4),
    }


def _allocate(arrays):
    """Run the real allocator without enabling detailed telemetry."""
    kernel = kernels.allocate_world_contact_slots
    wp.launch(
        kernel,
        dim=1,
        inputs=[arrays[name] for name in inspect.signature(kernel.func).parameters],
        device=arrays["contact_count"].device,
    )


class TestFeatherPGSCapacityStatus(unittest.TestCase):
    def test_graph_replay_overflow_empty_clear_overflow(self):
        """Keep overflow across graph replays until an explicit host clear."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph replay requires CUDA")
        device = devices[0]
        model = newton.ModelBuilder().finalize(device=device)
        solver = SolverFeatherPGS(model, row_watermark=False)
        arrays = _allocator_arrays(2, count=1, device=device)
        arrays["capacity_status"] = solver._constraint_capacity_status
        arrays["contact_count"].assign(np.array([0], dtype=np.int32))
        _allocate(arrays)  # Compile before capture without overflowing.
        with wp.ScopedCapture(device=device) as capture:
            _allocate(arrays)
        for count, clear, expected in ((1, False, True), (0, False, True), (0, True, False), (1, False, True)):
            if clear:
                self.assertTrue(solver.constraint_capacity_status(clear=True)["dense"])
            arrays["contact_count"].assign(np.array([count], dtype=np.int32))
            wp.capture_launch(capture.graph)
            self.assertEqual(solver.constraint_capacity_status()["dense"], expected)

    def test_sufficient_capacity_preserves_full_contact_law(self):
        """Solve all205 sphere-plane contacts at640 and compare a larger control."""
        count = 205
        dt = 0.01
        depths = np.full(count, 0.005, dtype=np.float32)
        depths[-1] = 0.015
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = 0.0
        shape_cfg.mu = 0.5
        shape_cfg.restitution = 0.0
        points = np.zeros((count, 3), dtype=np.float32)
        points[:, 0] = np.arange(count, dtype=np.float32) * 0.03
        points[:, 2] = -depths
        shapes = []
        for point in points:
            center = point + np.array([0.0, 0.0, 0.01], dtype=np.float32)
            shapes.append(
                builder.add_shape_sphere(
                    body, radius=0.01, xform=wp.transform(wp.vec3(*center), wp.quat_identity()), cfg=shape_cfg
                )
            )
        plane = builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        model.rigid_contact_max = count
        histories = {}
        for capacity in (192, 640, 768):
            solver = SolverFeatherPGS(
                model,
                pgs_mode="split",
                pgs_kernel="loop",
                dense_max_constraints=capacity,
                pgs_iterations=8,
                angular_damping=0.0,
                use_parallel_streams=False,
            )
            state_in, state_out = model.state(), model.state()
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            contacts = newton.Contacts(rigid_contact_max=count, soft_contact_max=0, device="cpu")
            contacts.rigid_contact_count.assign(np.array([count], dtype=np.int32))
            contacts.rigid_contact_shape0.assign(np.asarray(shapes, dtype=np.int32))
            contacts.rigid_contact_shape1.assign(np.full(count, plane, dtype=np.int32))
            contacts.rigid_contact_point0.assign(points)
            ground_points = points.copy()
            ground_points[:, 2] = 0.0
            contacts.rigid_contact_point1.assign(ground_points)
            contacts.rigid_contact_normal.assign(np.tile([0.0, 0.0, -1.0], (count, 1)).astype(np.float32))
            history = []
            for _ in range(3):
                old_velocity = float(state_in.joint_qd.numpy()[0])
                old_position = float(state_in.joint_q.numpy()[0])
                solver.step(state_in, state_out, model.control(), contacts, dt)
                solver.update_contacts(contacts)
                velocity = float(state_out.joint_qd.numpy()[0])
                forces = contacts.rigid_contact_force.numpy()
                self.assertTrue(np.isfinite(forces).all())
                self.assertTrue(np.isfinite(state_out.joint_q.numpy()).all())
                if capacity >= 640:
                    solver.check_constraint_capacity()
                    self.assertEqual(int(solver.constraint_count.numpy()[0]), count * 3)
                    residual = velocity + solver.pgs_beta * (old_position - depths.astype(np.float64)) / dt
                    self.assertGreaterEqual(float(residual.min()), -2.0e-6)
                    self.assertGreaterEqual(float(forces[:, 2].min()), -1.0e-7)
                    np.testing.assert_allclose(dt * forces[:, 2] * residual, 0.0, atol=1.0e-7)
                    np.testing.assert_allclose(forces[:, :2], 0.0, atol=1.0e-7)
                    # Force on contact A opposes the collision normal, so the
                    # upward published impulse accounts for the body's momentum.
                    np.testing.assert_allclose(
                        dt * forces[:, 2].sum(dtype=np.float64),
                        float(model.body_mass.numpy()[body]) * (velocity - old_velocity),
                        rtol=2.0e-5,
                        atol=2.0e-6,
                    )
                else:
                    self.assertTrue(solver.constraint_capacity_status()["dense"])
                history.append((state_out.joint_q.numpy().copy(), state_out.joint_qd.numpy().copy(), forces.copy()))
                state_in, state_out = state_out, state_in
            histories[capacity] = history
        for actual, expected in zip(histories[640], histories[768], strict=True):
            for actual_array, expected_array in zip(actual, expected, strict=True):
                np.testing.assert_allclose(actual_array, expected_array, rtol=2.0e-5, atol=2.0e-6)
        self.assertLess(float(histories[192][0][1][0]), float(histories[640][0][1][0]) - 1.0e-3)

    def test_full_triples_192_vs_640(self):
        """Expose dropped triples at192 while retaining all615 rows at640."""
        for capacity, accepted, overflow in ((192, 64, True), (640, 205, False)):
            with self.subTest(capacity=capacity):
                arrays = _allocator_arrays(capacity)
                _allocate(arrays)
                slots = arrays["contact_slot"].numpy()
                paths = arrays["contact_path"].numpy()
                self.assertEqual(int(np.sum(slots >= 0)), accepted)
                np.testing.assert_array_equal(slots[:accepted], np.arange(accepted) * 3)
                np.testing.assert_array_equal(slots[accepted:], -np.ones(205 - accepted, dtype=np.int32))
                np.testing.assert_array_equal(paths[:accepted], np.zeros(accepted, dtype=np.int32))
                np.testing.assert_array_equal(arrays["contact_slots_needed"].numpy(), np.full(205, 3))
                self.assertEqual(int(arrays["world_slot_counter"].numpy()[0]), accepted * 3)
                self.assertEqual(int(arrays["dense_dropped_contact_rows"].numpy()[0]), 0)
                np.testing.assert_array_equal(arrays["capacity_status"].numpy(), [int(overflow), 0, 0, 0])

    def test_every_contact_route_is_sticky(self):
        """Retain each failure flag across a later successful empty allocation."""
        for route in range(3):
            with self.subTest(route=route):
                arrays = _allocator_arrays(2, route=route, count=1)
                _allocate(arrays)
                expected = np.zeros(4, dtype=np.int32)
                expected[route] = 1
                np.testing.assert_array_equal(arrays["capacity_status"].numpy(), expected)
                arrays["contact_count"].assign(np.array([0], dtype=np.int32))
                _allocate(arrays)
                np.testing.assert_array_equal(arrays["capacity_status"].numpy(), expected)

    def test_raw_contact_prefix_overflow(self):
        """Report an incomplete contact prefix separately from row overflow."""
        arrays = _allocator_arrays(640, count=2)
        arrays["contact_count"].assign(np.array([3], dtype=np.int32))
        _allocate(arrays)
        np.testing.assert_array_equal(arrays["capacity_status"].numpy(), [0, 0, 0, 1])
        np.testing.assert_array_equal(arrays["contact_slot"].numpy(), [-1, -1])

    def test_internal_row_overflow(self):
        """Report unclamped internal demand even without contact reservations."""
        status = wp.zeros(4, dtype=wp.int32, device="cpu")
        counts = wp.zeros(2, dtype=wp.int32, device="cpu")
        kernel = kernels.finalize_constraint_counts_with_status
        for family in range(3):
            wp.launch(
                kernel,
                dim=2,
                inputs=[wp.array([3, 7], dtype=wp.int32, device="cpu"), 4, family],
                outputs=[counts, status],
                device="cpu",
            )
            np.testing.assert_array_equal(counts.numpy(), [3, 4])
        np.testing.assert_array_equal(status.numpy(), [1, 1, 1, 0])

    def test_public_check_and_explicit_clear(self):
        """Expose sticky status without row_watermark or changes to solver capacities."""
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = SolverFeatherPGS(model, dense_max_constraints=192, row_watermark=False)
        self.assertFalse(any(solver.constraint_capacity_status().values()))
        solver._constraint_capacity_status.assign(np.array([1, 0, 0, 0], dtype=np.int32))
        solver.reset(model.state())
        with self.assertRaisesRegex(RuntimeError, "dense_max_constraints=192"):
            solver.check_constraint_capacity()
        self.assertTrue(solver.constraint_capacity_status()["dense"])
        self.assertTrue(solver.constraint_capacity_status(clear=True)["dense"])
        self.assertFalse(any(solver.constraint_capacity_status().values()))
        solver.check_constraint_capacity()
        self.assertEqual(solver.dense_max_constraints, 192)

    def test_public_check_rejects_capture(self):
        """Reject host reads and explicit clears during CUDA graph capture."""
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = SolverFeatherPGS(model)
        capturing_device = SimpleNamespace(is_capturing=True)
        with mock.patch.object(wp, "get_device", return_value=capturing_device):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                solver.constraint_capacity_status(clear=True)
            with self.assertRaisesRegex(RuntimeError, "capture"):
                solver.check_constraint_capacity()


if __name__ == "__main__":
    unittest.main()

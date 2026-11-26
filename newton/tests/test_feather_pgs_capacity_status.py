# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test sticky overflow visibility independently of opt-in row telemetry."""

import contextlib
import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import (
    allocate_contacts,
    anchored_patches,
    contact_input,
    finalize_counts,
    row_order,
)


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
        "shape_body": integers([0]),
        "body_to_articulation": integers([0]),
        "art_to_world": integers([0]),
        "articulation_response_dof_count": integers([1]),
        "body_flags": zeros(1),
        "body_has_response_dofs": integers([1]),
        "is_free_rigid": integers([int(route == 1)]),
        "has_free_rigid": int(route == 1),
        "contact_gap_gate": 0.0,
        "same_articulation_contact_gap_gate": 0.0,
        "articulation_pair_contact_gap_gate": 0.0,
        "max_constraints": capacity,
        "mf_max_constraints": capacity,
        "enable_friction": 1,
        "contact_friction_gap_threshold": float("inf"),
        "row_capacity_telemetry": 0,
        "friction_patches": anchored_patches(length, device),
        "twists": wp.empty(0, dtype=wp.spatial_vector, device=device),
        "acts": wp.empty(0, dtype=wp.int32, device=device),
        "contact_world": zeros(length),
        "contact_slot": integers([-9] * length),
        "contact_art_a": zeros(length),
        "contact_art_b": zeros(length),
        "world_slot_counter": zeros(1),
        "contact_path": integers([-9] * length),
        "mf_slot_counter": zeros(1),
        "contact_slots_needed": zeros(length),
        "dense_dropped_contact_rows": zeros(1),
        "mf_dropped_contact_rows": zeros(1),
        "dense_first_rejected_slot": integers([2**31 - 1]),
        "mf_first_rejected_slot": integers([2**31 - 1]),
        "capacity_status": zeros(2),
        "units": wp.zeros((0, 0), dtype=wp.int32, device=device),
    }


def _allocate(arrays):
    """Run the real allocator without enabling detailed telemetry, then close both row families' reservations.

    Reservations record the first rejected slot; the families' count finalizers raise the status.
    """
    device = arrays["contact_count"].device
    data, _arrays = contact_input(arrays, device)
    allocate_contacts({**arrays, "data": data}, device)
    for family, counter, capacity, first_rejected in (
        (0, "world_slot_counter", "max_constraints", "dense_first_rejected_slot"),
        (1, "mf_slot_counter", "mf_max_constraints", "mf_first_rejected_slot"),
    ):
        wp.launch(
            finalize_counts,
            dim=1,
            inputs=[arrays[counter], arrays[capacity], family, arrays[first_rejected]],
            outputs=[wp.empty(1, dtype=wp.int32, device=device), arrays["capacity_status"]],
            device=device,
        )


def _clear_reservations(arrays):
    """Start a new step's reservations: empty counters, no rejected slot."""
    for name in ("world_slot_counter", "mf_slot_counter"):
        arrays[name].zero_()
    for name in ("dense_first_rejected_slot", "mf_first_rejected_slot"):
        arrays[name].fill_(2**31 - 1)


class TestFeatherPGSCapacityStatus(unittest.TestCase):
    def test_graph_replay_overflow_empty_clear_overflow(self):
        """Keep overflow across graph replays until an explicit host clear."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph replay requires CUDA")
        device = devices[0]
        model = newton.ModelBuilder().finalize(device=device)
        solver = SolverFeatherPGS(model)
        arrays = _allocator_arrays(2, count=1, device=device)
        arrays["capacity_status"] = solver.placement.constraint_capacity_status
        arrays["contact_count"].assign(np.array([0], dtype=np.int32))
        _allocate(arrays)  # Compile before capture without overflowing.
        with wp.ScopedCapture(device=device) as capture:
            _allocate(arrays)
        for count, clear, expected in ((1, False, True), (0, False, True), (0, True, False), (1, False, True)):
            if clear:
                self.assertTrue(solver.constraint_capacity_status(clear=True)["dense"])
            arrays["contact_count"].assign(np.array([count], dtype=np.int32))
            _clear_reservations(arrays)
            wp.capture_launch(capture.graph)
            self.assertEqual(solver.constraint_capacity_status()["dense"], expected)

    def test_dropped_contacts_fail_the_step(self):
        """A fixed contact buffer whose pass drops contacts fails the step; growable rows take every contact."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for z in (0.05, 0.149):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device="cpu")
        for growable in (False, True):
            pipeline = newton.CollisionPipeline(
                model, rigid_contact_max=None if growable else 2, growable_contacts=growable, publish_capacity=False
            )
            contacts = pipeline.contacts()
            solver = SolverFeatherPGS(model)
            state_0, state_1, control = model.state(), model.state(), model.control()
            pipeline.collide(state_0, contacts)
            if growable:
                solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
                self.assertEqual(pipeline.contact_drops().passes, 0)
                self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 2)
            else:
                with self.assertRaisesRegex(RuntimeError, "growable_contacts=True"):
                    solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)

    def test_status_without_articulations_reads_clear_flags(self):
        """Read, clear and check capacity status of a solver whose model has no articulation.

        Such a solver allocates no row buffers and no patch friction state, even
        under the default patch friction.
        """
        solver = SolverFeatherPGS(newton.ModelBuilder().finalize(device="cpu"))
        clear = dict.fromkeys(("dense", "matrix_free", "contact_launches", "slip_frames"), False)
        self.assertEqual(solver.constraint_capacity_status(), clear)
        self.assertEqual(solver.constraint_capacity_status(clear=True), clear)
        solver.check_constraint_capacity()

    def test_sufficient_capacity_preserves_full_contact_law(self):
        """Solve all 205 point-friction contacts at 640 rows and a larger control alike; from 192 rows the first step
        grows the capacity and repeats, so it solves them alike too."""
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
                dense_max_constraints=capacity,
                pgs_iterations=8,
                angular_damping=0.0,
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
            observables = solver.observables(contacts=contacts, kinds={solver.ObservableKind.CONTACT_F})
            for _ in range(3):
                old_velocity = float(state_in.joint_qd.numpy()[0])
                old_position = float(state_in.joint_q.numpy()[0])
                solver.step(state_in, state_out, model.control(), contacts, dt, observables=observables)
                velocity = float(state_out.joint_qd.numpy()[0])
                forces = contacts.rigid_contact_force.numpy()
                self.assertTrue(np.isfinite(forces).all())
                self.assertTrue(np.isfinite(state_out.joint_q.numpy()).all())
                if capacity < 640:
                    # The step outside capture grew the capacity to the rows it asked for (615: a rise from none
                    # predicts no more), in a power of two of L1 lines, and repeated: nothing dropped.
                    self.assertEqual(solver.placement.dense_max_constraints, 1024)
                    self.assertFalse(any(solver.constraint_capacity_status().values()))
                solver.check_constraint_capacity()
                # Every contact's three rows.
                self.assertEqual(int(solver.placement.constraint_count.numpy()[0]), count * 3)
                residual = velocity + solver.options.pgs_beta * (old_position - depths.astype(np.float64)) / dt
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
                history.append((state_out.joint_q.numpy().copy(), state_out.joint_qd.numpy().copy(), forces.copy()))
                state_in, state_out = state_out, state_in
            histories[capacity] = history
        for actual, expected in zip(histories[640], histories[768], strict=True):
            for actual_array, expected_array in zip(actual, expected, strict=True):
                np.testing.assert_allclose(actual_array, expected_array, rtol=2.0e-5, atol=2.0e-6)
        for actual, expected in zip(histories[192], histories[640], strict=True):
            for actual_array, expected_array in zip(actual, expected, strict=True):
                np.testing.assert_allclose(actual_array, expected_array, rtol=2.0e-5, atol=2.0e-6)

    def test_full_triples_192_vs_640(self):
        """Expose dropped triples at 192 rows while retaining all 615 rows at 640."""
        for capacity, accepted, overflow in ((192, 64, True), (640, 205, False)):
            with self.subTest(capacity=capacity):
                arrays = _allocator_arrays(capacity)
                _allocate(arrays)
                slots = arrays["contact_slot"].numpy()
                paths = arrays["contact_path"].numpy()
                self.assertEqual(int(np.sum(slots >= 0)), accepted)
                # The world's rows follow its row order: the first ``accepted`` contacts in it hold the rows.
                order = row_order(np.arange(205))
                np.testing.assert_array_equal(slots[order[:accepted]], np.arange(accepted) * 3)
                np.testing.assert_array_equal(slots[order[accepted:]], -np.ones(205 - accepted, dtype=np.int32))
                np.testing.assert_array_equal(paths[order[:accepted]], np.zeros(accepted, dtype=np.int32))
                np.testing.assert_array_equal(arrays["contact_slots_needed"].numpy(), np.full(205, 3))
                self.assertEqual(int(arrays["world_slot_counter"].numpy()[0]), 205 * 3)
                expected_first_rejected = accepted * 3 if overflow else 2**31 - 1
                self.assertEqual(int(arrays["dense_first_rejected_slot"].numpy()[0]), expected_first_rejected)
                self.assertEqual(int(arrays["dense_dropped_contact_rows"].numpy()[0]), 0)
                np.testing.assert_array_equal(arrays["capacity_status"].numpy(), [int(overflow), 0])

    def test_every_contact_route_is_sticky(self):
        """Retain each failure flag across a later successful empty allocation."""
        for route in range(2):
            with self.subTest(route=route):
                arrays = _allocator_arrays(2, route=route, count=1)
                _allocate(arrays)
                expected = np.zeros(2, dtype=np.int32)
                expected[route] = 1
                np.testing.assert_array_equal(arrays["capacity_status"].numpy(), expected)
                arrays["contact_count"].assign(np.array([0], dtype=np.int32))
                _clear_reservations(arrays)
                _allocate(arrays)
                np.testing.assert_array_equal(arrays["capacity_status"].numpy(), expected)

    def test_internal_row_overflow(self):
        """Report unclamped internal demand even without contact reservations."""
        status = wp.zeros(2, dtype=wp.int32, device="cpu")
        counts = wp.zeros(2, dtype=wp.int32, device="cpu")
        kernel = finalize_counts
        for family in range(2):
            wp.launch(
                kernel,
                dim=2,
                inputs=[
                    wp.array([3, 7], dtype=wp.int32, device="cpu"),
                    4,
                    family,
                    wp.full(2, 2**31 - 1, dtype=wp.int32, device="cpu"),
                ],
                outputs=[counts, status],
                device="cpu",
            )
            np.testing.assert_array_equal(counts.numpy(), [3, 4])
        np.testing.assert_array_equal(status.numpy(), [1, 1])

    def test_free_body_rows_past_the_packed_parent_split_their_records(self):
        """A free-body row capacity past the packed record's 16-bit parent (65,535 rows, 0xffff for none) takes split
        records (:func:`.row_metadata.split_records`: the parent in a word of its own), never a wrapped parent; one
        that fits keeps the packed record."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        self.assertFalse(SolverFeatherPGS(model, mf_max_constraints=1_000).placement.split_records)
        self.assertTrue(SolverFeatherPGS(model, mf_max_constraints=65_536).placement.split_records)

    def test_finalizer_truncates_rejected_contact_reservations(self):
        """Exclude a rejected triple even when its first row fits the capacity."""
        device = "cpu"
        counter = wp.array([6, 6], dtype=wp.int32, device=device)
        first_rejected = wp.array([3, 0], dtype=wp.int32, device=device)
        count = wp.empty(2, dtype=wp.int32, device=device)
        status = wp.zeros(2, dtype=wp.int32, device=device)
        for family in range(2):
            wp.launch(
                finalize_counts,
                dim=2,
                inputs=[counter, 4, family, first_rejected],
                outputs=[count, status],
                device=device,
            )
            np.testing.assert_array_equal(count.numpy(), [3, 0])
        np.testing.assert_array_equal(status.numpy(), [1, 1])

    def test_public_check_and_explicit_clear(self):
        """Keep historical audit flags separate from resettable per-world validity."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, dense_max_constraints=192)
        self.assertFalse(any(solver.constraint_capacity_status().values()))
        solver.placement.constraint_capacity_status.assign(np.array([1, 0], dtype=np.int32))
        solver.placement.constraint_overflow.fill_(True)
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            solver.check_constraint_capacity()
        solver.reset(model.state())
        solver.check_constraint_capacity()
        self.assertTrue(solver.constraint_capacity_status()["dense"])
        self.assertTrue(solver.constraint_capacity_status(clear=True)["dense"])
        self.assertFalse(any(solver.constraint_capacity_status().values()))
        solver.check_constraint_capacity()
        self.assertEqual(solver.placement.dense_max_constraints, 192)

    @staticmethod
    def _box_on_ball(device="cpu", height=0.25):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        anchor = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()))
        builder.add_shape_box(anchor, hx=0.2, hy=0.2, hz=0.05)
        limb = builder.add_link(xform=wp.transform(wp.vec3(0.3, 0.0, height), wp.quat_identity()))
        builder.add_shape_box(limb, hx=0.1, hy=0.2, hz=0.05)
        free = builder.add_joint_free(anchor)
        ball = builder.add_joint_ball(
            anchor,
            limb,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([free, ball])
        return builder.finalize(device=device)

    @staticmethod
    def _land(model, solver, steps=40):
        """Drop the box on its ball joint for ``steps`` steps, the solver's service after each; each step's state."""
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        history = []
        for _ in range(steps):
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)
            # The pipeline's contacts are fixed, so its service does not reach the solver's pool: serve it directly.
            solver.serve_rows()
            state_in, state_out = state_out, state_in
            history.append((state_in.body_q.numpy().copy(), state_in.body_qd.numpy().copy()))
        return history, state_in, state_out, contacts, control

    def test_a_step_past_its_capacity_grows_and_repeats(self):
        """A box on a ball joint that lands on the floor asks for more dense rows than the line holds: the step, outside
        capture, grows the capacity and repeats, so no pass drops a row, no status latches, and the run equals one
        built at the grown capacity, bit for bit."""
        model = self._box_on_ball()
        solver = SolverFeatherPGS(model, dense_max_constraints=6)
        history, *_ = self._land(model, solver)
        capacity = solver.placement.dense_max_constraints
        self.assertGreater(capacity, 6)
        self.assertEqual(solver.placement.rows_asked()[0], 0)
        self.assertFalse(solver.placement.constraint_overflow.numpy().any())
        self.assertFalse(any(solver.constraint_capacity_status().values()))
        solver.check_constraint_capacity()
        self.assertEqual(solver.get_path_selection()["options"]["dense_max_constraints"]["requested"], 6)
        built, *_ = self._land(model, SolverFeatherPGS(model, dense_max_constraints=capacity))
        for actual, expected in zip(history, built, strict=True):
            for actual_array, expected_array in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(actual_array, expected_array)

    @unittest.skipUnless(wp.is_cuda_available(), "graph replays are CUDA's")
    def test_a_replay_past_its_capacity_latches_and_the_service_grows(self):
        """Inside capture no step can stop: a graph replay whose world asks for more dense rows than its capacity holds
        drops them and latches the world's status; the next service grows the capacity and asks for a recapture, and
        the recaptured step's replays drop nothing: one replay dropped rows, and the status stays until reset."""
        device = wp.get_cuda_device()
        model = self._box_on_ball(device, height=1.0)
        # One row: the first contact's three rows pass it.
        solver = SolverFeatherPGS(model, dense_max_constraints=1)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        states, control = (model.state(), model.state()), model.control()

        def step(source, target):
            pipeline.collide(states[source], contacts)
            solver.step(states[source], states[target], control, contacts, 1.0 / 200.0)

        # Eager steps high in the air (the box falls about 95 cm to the floor): every kernel loads, no row is asked for.
        for k in range(10):
            step(k % 2, 1 - k % 2)
        self.assertEqual(solver.placement.dense_max_constraints, 1)

        def capture():
            graphs = []
            for source in (0, 1):
                with wp.ScopedCapture(device) as scope:
                    step(source, 1 - source)
                graphs.append(scope.graph)
            return graphs

        graphs, dropped_replays, grown = capture(), 0, False
        for k in range(160):
            wp.capture_launch(graphs[k % 2])
            wp.synchronize_device(device)
            if solver.placement.rows_asked()[0] > 0:
                dropped_replays += 1
                self.assertTrue(solver.placement.constraint_overflow.numpy()[0])
            if solver.serve_rows():
                grown = True
                graphs = capture()
        self.assertTrue(grown)
        self.assertEqual(dropped_replays, 1)
        self.assertGreater(solver.placement.dense_max_constraints, 1)
        self.assertEqual(solver.placement.rows_asked()[0], 0)
        with self.assertRaisesRegex(RuntimeError, "dense_max_constraints >= "):
            solver.check_constraint_capacity()
        self.assertTrue(np.isfinite(states[0].body_q.numpy()).all())
        solver.reset(model.state())
        solver.check_constraint_capacity()

    @staticmethod
    def _box_stack(levels=4):
        """A 2 x 2 x ``levels`` stack of free boxes resting on a ground plane."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for i in range(2):
            for j in range(2):
                for k in range(levels):
                    body = builder.add_body(
                        xform=wp.transform(wp.vec3(0.2 * i, 0.2 * j, 0.1 + 0.2 * k), wp.quat_identity())
                    )
                    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        return builder.finalize(device="cpu")

    def test_default_rows_hold_a_resting_box_stack(self):
        """A resting 2 x 2 x 4 box stack asks for more free-body rows than the default 512 a world: its steps grow the
        rows without dropping any, and it stays stacked."""
        model = self._box_stack()
        solver = SolverFeatherPGS(model, diagnostics=SolverFeatherPGS.Diagnostics(row_watermark=True))
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        for _ in range(30):
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 1.0 / 600.0)
            state_in, state_out = state_out, state_in
        marks = solver.constraint_row_watermarks()
        self.assertGreater(marks["mf_high_water"], 512)
        self.assertLessEqual(marks["mf_raw_high_water"], solver.placement.mf_max_constraints)
        self.assertEqual(marks["mf_overflow_world_steps"], 0)
        self.assertAlmostEqual(float(state_in.body_q.numpy()[:, 2].max()), 0.7, delta=1.0e-3)

    def test_default_dense_rows_hold_a_landing_limb(self):
        """The jointed box that lands on the floor (above) steps at the default dense rows, dropping none."""
        model = self._box_on_ball()
        solver = SolverFeatherPGS(model)
        self.assertEqual(solver.placement.dense_max_constraints, 32)
        _, state_in, *_ = self._land(model, solver)
        solver.check_constraint_capacity()
        self.assertFalse(any(solver.constraint_capacity_status().values()))
        self.assertTrue(np.isfinite(state_in.body_q.numpy()).all())

    def test_public_check_rejects_capture(self):
        """Reject host reads and explicit clears during CUDA graph capture."""
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = SolverFeatherPGS(model)
        capturing_device = SimpleNamespace(is_cuda=True, is_capturing=True)
        with (
            mock.patch.object(wp, "get_device", return_value=capturing_device),
            mock.patch.object(model, "device", capturing_device),
        ):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                solver.constraint_capacity_status(clear=True)
            with self.assertRaisesRegex(RuntimeError, "capture"):
                solver.check_constraint_capacity()


def _sprawl_run(device, capacity: int, heads=None, steps: int = 30, in_place: bool = False):
    """Step two worlds of the sprawled tree (scenes.sprawl: a free torso and its limbs resting on the ground, one contact
    a link: dense contact rows) at ``capacity`` dense rows; with ``heads`` (first, second), the general sweep's launches
    are dense heads at those rows and the capacity's (:meth:`.sweep_plan.Sweep.dense_head`, forced on any device); with
    ``in_place``, the capacity's dense impulses in place, as where its stage declines them (each head places its own).
    Returns the poses and velocities, the most live dense rows a world held and the solver."""
    from newton._src.solvers.feather_pgs import sweep_plan  # noqa: PLC0415
    from newton._src.solvers.feather_pgs.sweep_routes import Place  # noqa: PLC0415
    from newton.tests import feather_pgs_scenes as scenes  # noqa: PLC0415

    def forced(plan, config, items):
        full = plan._staged(config, items)
        if in_place:
            full = dataclasses.replace(full, places=dataclasses.replace(full.places, dense_impulses=Place.GLOBAL))
        if heads is None:
            return (full,)
        first, wide = heads
        return (
            plan.dense_head(full, items, first),
            plan.dense_head(full, items, wide, first),
            plan.dense_head(full, items, 0, wide),
        )

    scene = scenes.sprawl(fingers=2, finger_links=2, worlds=2)
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=scene.rigid_contact_max)
    contacts = pipeline.contacts()
    forcing = heads or in_place
    patch = mock.patch.object(sweep_plan.Sweep, "_dense_heads", forced) if forcing else contextlib.nullcontext()
    with patch:
        # The dense storage's route: the matrix-free sweep over the world's dense line.
        solver = SolverFeatherPGS(model, dense_max_constraints=capacity, pins=SolverFeatherPGS.Pins(storage="dense"))
    state, out, control = model.state(), model.state(), model.control()
    rows = 0
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, scene.dt)
        state, out = out, state
        rows = max(rows, int(solver.placement.constraint_count.numpy().max()))
    return state.body_q.numpy(), state.body_qd.numpy(), rows, solver


class TestDenseHeads(unittest.TestCase):
    def test_a_world_steps_alike_in_every_dense_head(self):
        """A capacity past the live dense rows keeps their stage: with the general sweep's launches at dense heads, a
        world swept by the launch whose head holds its live rows (the first head's, the wide one's, or past both the
        capacity's own launch) steps bit for bit as at the capacity alone, and each launch's kernel stages its head."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            q, qd, rows, _solver = _sprawl_run(device, 256, heads=None)
            self.assertTrue(32 < rows <= 192, f"the sprawl's live dense rows: {rows}")
            for heads in ((rows + 32, rows + 64), (32, rows + 32), (32, 64) if rows > 64 else (16, 32)):
                with self.subTest(device=device, heads=heads):
                    q_h, qd_h, _rows, solver = _sprawl_run(device, 256, heads=heads)
                    names = [launch.kernel.key for launch in solver.sweep.launches]
                    for first, wide in ((0, heads[0]), (heads[0], heads[1]), (heads[1], "all")):
                        self.assertTrue(any(f"_d{first}to{wide}" in name for name in names), (first, wide, names))
                    np.testing.assert_array_equal(q, q_h)
                    np.testing.assert_array_equal(qd, qd_h)

    def test_heads_stage_the_dense_impulses_a_capacity_keeps_in_place(self):
        """A capacity whose stage declines the dense impulses (in place) costs the worlds a head holds nothing: each
        head's launch stages them at its own length, and a world steps bit for bit as at the capacity alone, its
        impulses in place or staged."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            q, qd, rows, _solver = _sprawl_run(device, 256, heads=None)
            q_g, qd_g, _rows, solver = _sprawl_run(device, 256, heads=None, in_place=True)
            self.assertTrue(any("_gline" in launch.kernel.key for launch in solver.sweep.launches))
            np.testing.assert_array_equal(q, q_g)
            np.testing.assert_array_equal(qd, qd_g)
            heads = (32, rows + 32)
            with self.subTest(device=device, heads=heads):
                q_h, qd_h, _rows, solver = _sprawl_run(device, 256, heads=heads, in_place=True)
                keys = {
                    launch.config.places.dense_rows: launch.kernel.key
                    for launch in solver.sweep.launches
                    if launch.kernel.key.startswith("sweep_mf_gs") and "_records" not in launch.kernel.key
                }
                for head in heads:
                    self.assertNotIn("_gline", keys[head], keys)
                self.assertIn("_gline", keys[0], keys)
                np.testing.assert_array_equal(q, q_h)
                np.testing.assert_array_equal(qd, qd_h)

    def test_a_diagonal_world_steps_alike_in_every_dense_head(self):
        """The sparse diagonal's stage follows the live rows too: with its launches at dense heads (forced on any
        device), a world swept by the launch whose head holds its live rows (the first head's, the second's, or past
        both the capacity's own launch) steps bit for bit as at the capacity alone, and each launch stages its head."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            q, qd, rows, solver = _keyboard_run(device, 64, heads=None)
            self.assertIsNotNone(solver.sweep.diagonal_sweep, "the keyboard takes the sparse diagonal")
            self.assertTrue(2 < rows < 64, f"the keyboard's live rows: {rows}")
            for heads in ((rows, rows + 1), (rows - 1, rows), (1, 2)):
                with self.subTest(device=device, heads=heads):
                    q_h, qd_h, _rows, solver = _keyboard_run(device, 64, heads=heads)
                    names = [launch.kernel.key for launch in solver.sweep.launches]
                    for first, wide in ((0, heads[0]), (heads[0], heads[1]), (heads[1], "all")):
                        self.assertTrue(any(f"_d{first}to{wide}" in name for name in names), (first, wide, names))
                    np.testing.assert_array_equal(q, q_h)
                    np.testing.assert_array_equal(qd, qd_h)


def _keyboard_run(device, capacity: int, heads=None, steps: int = 30):
    """Step two worlds of the keyboard (scenes.keyboard: a fixed frame of keys and an arm pressing key 0: the sparse
    diagonal's rows) at ``capacity`` dense rows; with ``heads`` (first, second), the sparse diagonal's launches are
    dense heads at those rows and the capacity's (:meth:`.sweep_plan.Sweep.dense_head`, forced on any device).
    Returns the joint coordinates and velocities, the most live rows a world held and the solver."""
    from newton._src.solvers.feather_pgs import sweep_plan, sweep_routes_diag  # noqa: PLC0415
    from newton.tests import feather_pgs_scenes as scenes  # noqa: PLC0415

    def forced(plan, full, items):
        if not isinstance(full, sweep_routes_diag.DiagonalSweep):
            return (full,)
        first, wide = heads
        return (
            plan.dense_head(full, items, first),
            plan.dense_head(full, items, wide, first),
            plan.dense_head(full, items, 0, wide),
        )

    scene = scenes.keyboard(worlds=2)
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=scene.rigid_contact_max)
    contacts = pipeline.contacts()
    patch = mock.patch.object(sweep_plan.Sweep, "_dense_heads", forced) if heads else contextlib.nullcontext()
    with patch:
        solver = SolverFeatherPGS(model, **{**scene.solver_options, "dense_max_constraints": capacity})
    state, out, control = model.state(), model.state(), model.control()
    rows = 0
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, scene.dt)
        state, out = out, state
        rows = max(rows, int(solver.placement.constraint_count.numpy().max()))
    return state.joint_q.numpy(), state.joint_qd.numpy(), rows, solver


if __name__ == "__main__":
    unittest.main()

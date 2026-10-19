# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for mixed propagation-colored contact units, including the serial tail."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 240.0
RADIUS = 0.05
ROW_CAPACITY = 6
CONTACT_COUNT = 4
FRICTION_GAP_THRESHOLD = 0.0
# (x, y, gap) of the sphere feet of one body: three penetrate at the corners of a triangle
# and take three rows each, one hovers under the centre and takes one. Feet farther apart
# than two radii stay separate friction patches.
SHARED_FEET = ((0.15, 0.0, -0.002), (0.0, 0.0, 0.006), (-0.075, 0.13, -0.001), (-0.075, -0.13, -0.0015))
SHARED_ROW_COUNT = 10
TAIL_COLOR_CAP = 2


def _build_mixed_gap_model(device: str) -> newton.Model:
    """Build four disjoint spheres whose ground contacts use one or three rows."""
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    world = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=0.6,
        margin=0.0,
        gap=0.015,
    )
    for index, gap in enumerate((-0.001, 0.004, 0.007, 0.010)):
        body = world.add_link(
            xform=wp.transform(wp.vec3(0.25 * index, 0.0, RADIUS + gap), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001),
            lock_inertia=True,
        )
        world.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        world.add_articulation([world.add_joint_free(parent=-1, child=body)])
    scene.add_world(world)
    scene.add_ground_plane(cfg=cfg)
    return scene.finalize(device=device)


def _build_shared_body_model(device: str) -> newton.Model:
    """Build one body on four sphere feet, so every ground contact unit shares that body."""
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    world = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6, margin=0.0, gap=0.015)
    body = world.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02),
        lock_inertia=True,
    )
    for x, y, gap in SHARED_FEET:
        foot = wp.transform(wp.vec3(x, y, gap), wp.quat_identity())
        world.add_shape_sphere(body, xform=foot, radius=RADIUS, cfg=cfg)
    world.add_articulation([world.add_joint_free(parent=-1, child=body)])
    scene.add_world(world)
    scene.add_ground_plane(cfg=cfg)
    return scene.finalize(device=device)


def _make_solver(model: newton.Model, response: str, *, row_capacity: int = ROW_CAPACITY) -> SolverFeatherPGS:
    """Build a serial or colored propagation solver with ``row_capacity`` contact rows.

    Propagation takes the dense and the matrix-free row capacities together: the dense share
    stays at three rows and the matrix-free share takes the rest.
    """
    dense_rows = 3
    return SolverFeatherPGS(
        model,
        angular_damping=0.0,
        pgs_mode="matrix_free",
        articulated_contact_response=response,
        pgs_iterations=8,
        pgs_warmstart=False,
        mf_warmstart=False,
        propagation_cached_response=False,
        dense_max_constraints=dense_rows,
        mf_max_constraints=row_capacity - dense_rows,
        contact_friction_gap_threshold=FRICTION_GAP_THRESHOLD,
        contact_speculative_scale=0.0,
        use_parallel_streams=False,
    )


def _seed_state(model: newton.Model, state: newton.State) -> tuple[np.ndarray, np.ndarray]:
    """Give every free body the same downward speed and a distinct tangent speed."""
    joint_q = state.joint_q.numpy().copy()
    joint_qd = np.zeros(state.joint_qd.shape, dtype=np.float32)
    dof_starts = model.joint_qd_start.numpy()
    for index in range(model.joint_count):
        dof = int(dof_starts[index])
        joint_qd[dof] = 0.15 + 0.03 * index
        joint_qd[dof + 2] = -0.1
    state.joint_q.assign(joint_q)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return joint_q, joint_qd


def _unit_impulses(model: newton.Model, solver: SolverFeatherPGS, contacts: newton.Contacts):
    """Return propagation impulses keyed by the dynamic body of each contact."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    shape_body = model.shape_body.numpy()
    slots = solver.contact_slot.numpy()[:count]
    lengths = solver.contact_slots_needed.numpy()[:count]
    paths = solver.contact_path.numpy()[:count]
    rows = solver.propagation_impulses.numpy()[0]
    result = {}
    for contact in range(count):
        if int(paths[contact]) != 2:
            continue
        body_a = int(shape_body[int(shape0[contact])]) if shape0[contact] >= 0 else -1
        body_b = int(shape_body[int(shape1[contact])]) if shape1[contact] >= 0 else -1
        body = max(body_a, body_b)
        slot = int(slots[contact])
        length = int(lengths[contact])
        result[body] = rows[slot : slot + length].copy()
    return result


def _assert_mixed_topology(test: unittest.TestCase, solver: SolverFeatherPGS, contacts: newton.Contacts) -> None:
    """Assert every mixed unit is colored and owns contiguous rows in color order.

    The coloring keeps its unit state in global scratch, so no unit spills into the
    serial tail for lack of staging space. The four spheres share no body, so all
    four units, one three-row friction unit and three one-row units, take color 0
    in contact-index order, and their six rows fill the row capacity without gaps.
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, CONTACT_COUNT)
    paths = solver.contact_path.numpy()[:count]
    np.testing.assert_array_equal(paths, np.full(CONTACT_COUNT, 2, dtype=np.int32))
    lengths = solver.contact_slots_needed.numpy()[:count]
    test.assertEqual(sorted(lengths.tolist()), [1, 1, 1, 3])
    test.assertEqual(int(solver.propagation_constraint_count.numpy()[0]), ROW_CAPACITY)
    test.assertEqual(int(solver.color_world_unit_cursor.numpy()[0]), CONTACT_COUNT)

    # Entry -2 starts the serial tail and entry -1 holds the unit total.
    offsets = solver.color_world_offsets.numpy().reshape(solver.world_count, -1)[0]
    bucket_counts = np.diff(offsets)
    test.assertEqual(int(offsets[-1]), CONTACT_COUNT)
    test.assertEqual(int(offsets[-2]), CONTACT_COUNT)
    test.assertEqual(int(bucket_counts[0]), CONTACT_COUNT)
    np.testing.assert_array_equal(bucket_counts[1:], np.zeros_like(bucket_counts[1:]))

    ordered = solver.color_unit_sorted.numpy()[:CONTACT_COUNT]
    test.assertEqual(ordered.tolist(), list(range(CONTACT_COUNT)))
    slots = solver.contact_slot.numpy()[:count]
    starts = np.cumsum(lengths[ordered]) - lengths[ordered]
    np.testing.assert_array_equal(slots[ordered], starts)


def _assert_overflow_topology(
    test: unittest.TestCase, solver: SolverFeatherPGS, contacts: newton.Contacts, colors: int
) -> np.ndarray:
    """Assert one color each for the first ``colors`` shared-body units and the serial tail for the rest.

    The units share the body, so first-fit gives each unit the next color in contact-index
    order until the color cap runs out; the remaining units fill the tail in the same order,
    and every unit's rows follow the previous unit's rows. Returns each unit's row count.
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, CONTACT_COUNT)
    np.testing.assert_array_equal(solver.contact_path.numpy()[:count], np.full(CONTACT_COUNT, 2, dtype=np.int32))
    lengths = solver.contact_slots_needed.numpy()[:count]
    test.assertEqual(sorted(lengths.tolist()), [1, 3, 3, 3])
    test.assertEqual(int(solver.propagation_constraint_count.numpy()[0]), SHARED_ROW_COUNT)
    test.assertEqual(int(solver.color_world_unit_cursor.numpy()[0]), CONTACT_COUNT)

    # Entry -2 starts the serial tail and entry -1 holds the unit total.
    offsets = solver.color_world_offsets.numpy().reshape(solver.world_count, -1)[0]
    np.testing.assert_array_equal(offsets[: colors + 1], np.arange(colors + 1))
    test.assertEqual(int(offsets[-2]), colors)
    test.assertEqual(int(offsets[-1]), CONTACT_COUNT)

    test.assertEqual(solver.color_unit_sorted.numpy()[:CONTACT_COUNT].tolist(), list(range(CONTACT_COUNT)))
    np.testing.assert_array_equal(solver.contact_slot.numpy()[:count], np.cumsum(lengths) - lengths)
    return lengths


@unittest.skipUnless(wp.is_cuda_available(), "propagation-colored requires CUDA")
class TestFeatherPGSColoredUnitCapacity(unittest.TestCase):
    def test_mixed_gap_matches_serial_propagation(self):
        """Match impulses, body velocity, and a short trajectory with mixed one- and three-row units."""
        device = "cuda:0"
        model = _build_mixed_gap_model(device)
        colored = _make_solver(model, "propagation-colored")
        serial = _make_solver(model, "propagation")
        colored_in, colored_out = model.state(), model.state()
        serial_in, serial_out = model.state(), model.state()
        _seed_state(model, colored_in)
        _seed_state(model, serial_in)
        colored_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        serial_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        colored_contacts = colored_pipeline.contacts()
        serial_contacts = serial_pipeline.contacts()
        control = model.control()

        for step in range(4):
            colored_in.clear_forces()
            serial_in.clear_forces()
            colored_pipeline.collide(colored_in, colored_contacts)
            serial_pipeline.collide(serial_in, serial_contacts)
            colored.step(colored_in, colored_out, control, colored_contacts, DT)
            serial.step(serial_in, serial_out, control, serial_contacts, DT)

            if step == 0:
                _assert_mixed_topology(self, colored, colored_contacts)
            colored_impulses = _unit_impulses(model, colored, colored_contacts)
            serial_impulses = _unit_impulses(model, serial, serial_contacts)
            self.assertEqual(sorted(colored_impulses), sorted(serial_impulses))
            for body, expected in serial_impulses.items():
                np.testing.assert_allclose(colored_impulses[body], expected, rtol=2.0e-5, atol=2.0e-6)
            np.testing.assert_allclose(
                colored.propagation_body_qd.numpy(),
                serial.propagation_body_qd.numpy(),
                rtol=2.0e-5,
                atol=2.0e-6,
            )
            np.testing.assert_allclose(
                colored_out.joint_qd.numpy(), serial_out.joint_qd.numpy(), rtol=2.0e-5, atol=2.0e-6
            )
            np.testing.assert_allclose(
                colored_out.joint_q.numpy(), serial_out.joint_q.numpy(), rtol=2.0e-5, atol=2.0e-6
            )
            colored_in, colored_out = colored_out, colored_in
            serial_in, serial_out = serial_out, serial_in

    def test_mixed_gap_cuda_graph_matches_eager(self):
        """Replay a captured mixed-unit colored step with the same result as eager execution."""
        device = wp.get_device("cuda:0")
        model = _build_mixed_gap_model(str(device))
        graph_solver = _make_solver(model, "propagation-colored")
        eager_solver = _make_solver(model, "propagation-colored")
        graph_state, graph_out = model.state(), model.state()
        eager_state, eager_out = model.state(), model.state()
        initial_q, initial_qd = _seed_state(model, graph_state)
        _seed_state(model, eager_state)
        graph_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        eager_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        graph_contacts = graph_pipeline.contacts()
        eager_contacts = eager_pipeline.contacts()
        graph_control = model.control()
        eager_control = model.control()

        def one_step(solver, pipeline, contacts, state_in, state_out, control):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        one_step(graph_solver, graph_pipeline, graph_contacts, graph_state, graph_out, graph_control)
        one_step(eager_solver, eager_pipeline, eager_contacts, eager_state, eager_out, eager_control)
        for state in (graph_state, eager_state):
            state.joint_q.assign(initial_q)
            state.joint_qd.assign(initial_qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        with wp.ScopedCapture(device) as capture:
            one_step(graph_solver, graph_pipeline, graph_contacts, graph_state, graph_out, graph_control)
        wp.capture_launch(capture.graph)
        one_step(eager_solver, eager_pipeline, eager_contacts, eager_state, eager_out, eager_control)

        _assert_mixed_topology(self, graph_solver, graph_contacts)
        np.testing.assert_allclose(graph_state.joint_q.numpy(), eager_state.joint_q.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(graph_state.joint_qd.numpy(), eager_state.joint_qd.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            graph_solver.propagation_body_qd.numpy(),
            eager_solver.propagation_body_qd.numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_color_overflow_tail_matches_one_color_per_unit(self):
        """Sweep the units beyond the color cap in the serial tail as one color per unit would.

        Every unit shares the body, so first-fit gives each unit its own color in
        contact-index order. Under a two-color cap the last two units overflow into the
        ordered serial tail, which must sweep them in that same order: on the same contacts,
        the impulses, body velocity and a short trajectory match the uncapped solve.
        """
        device = "cuda:0"
        model = _build_shared_body_model(device)
        uncapped = _make_solver(model, "propagation-colored", row_capacity=SHARED_ROW_COUNT)
        with patch.object(solver_feather_pgs, "PROPAGATION_COLOR_TAIL", TAIL_COLOR_CAP):
            capped = _make_solver(model, "propagation-colored", row_capacity=SHARED_ROW_COUNT)
        arms = [(solver, model.state(), model.state()) for solver in (uncapped, capped)]
        for _, state_in, _ in arms:
            _seed_state(model, state_in)
        # One contact set feeds both arms, so both sweep the units in one contact-index order.
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        control = model.control()

        for step in range(4):
            pipeline.collide(arms[0][1], contacts)
            for solver, state_in, state_out in arms:
                state_in.clear_forces()
                solver.step(state_in, state_out, control, contacts, DT)

            if step == 0:
                _assert_overflow_topology(self, uncapped, contacts, CONTACT_COUNT)
                lengths = _assert_overflow_topology(self, capped, contacts, TAIL_COLOR_CAP)
                # Three of the four units take friction rows, so the two-unit tail holds one in
                # any contact order, and the tail's rows carry load: the comparison below
                # covers the tail's own sweep, friction pairs included.
                self.assertIn(3, lengths[TAIL_COLOR_CAP:].tolist())
                tail_start = int(np.sum(lengths[:TAIL_COLOR_CAP]))
                self.assertGreater(float(np.max(capped.propagation_impulses.numpy()[0, tail_start:])), 0.0)
            for name in ("propagation_impulses", "propagation_body_qd"):
                np.testing.assert_allclose(
                    getattr(capped, name).numpy(),
                    getattr(uncapped, name).numpy(),
                    rtol=2.0e-5,
                    atol=2.0e-6,
                    err_msg=name,
                )
            (_, _, uncapped_out), (_, _, capped_out) = arms
            for name in ("joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(capped_out, name).numpy(),
                    getattr(uncapped_out, name).numpy(),
                    rtol=2.0e-5,
                    atol=2.0e-6,
                    err_msg=name,
                )
            arms = [(solver, state_out, state_in) for solver, state_in, state_out in arms]
        for solver, _, _ in arms:
            solver.check_constraint_capacity()


if __name__ == "__main__":
    unittest.main(verbosity=2)

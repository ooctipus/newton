# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse diagonal's one writer on the compact diagonal (FL3.5): a real two-world model, a 108-key star beside a
six-coordinate chain.

The one writer walks every row from its unit source (:func:`.walk_rows.slot_rows_kernel`): the solver steps, resets,
takes a model change and replays a graph with the route record naming the row builders it launches, and under patch
friction its rows link their rings and keep the patch law. The rows' arithmetic is held to the walk's invariants in
``test_feather_pgs_walk_rows`` (:class:`TestSlotFrame`). The model is the fixture of the contact schedule and scalar
branch tests.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.feather_pgs_patch_parity import (
    assert_contact_impulses_close,
    assert_history_close,
    assert_ring_coulomb,
    assert_rows_close,
    contact_rows,
)
from newton.tests.feather_pgs_row_builders import assert_row_builders_named, launched_kernels


def build_solver_fixture(device):
    """Actual admitted two-world sparse108 + serial-six model with live contact shapes."""
    template = newton.ModelBuilder()
    inertia = wp.mat33(0.04, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.06)
    root = template.add_link(mass=2.0, inertia=inertia)
    joints = [template.add_joint_fixed(parent=-1, child=root)]
    for index in range(108):
        body = template.add_link(mass=0.2 + index / 300, inertia=inertia)
        joints.append(
            template.add_joint_prismatic(
                parent=root,
                child=body,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(index / 100, 0.0, 0.2), wp.quat_identity()),
                target_ke=12.0,
                target_kd=0.4,
                limit_lower=-0.05,
                limit_upper=0.05,
            )
        )
        if index < 2:
            template.add_shape_sphere(body=body, radius=0.01)
    template.add_articulation(joints)
    # The chain stands on a fixed base: the held tree, which alone evaluates the compact diagonal, takes fixed or
    # free roots.
    parent = template.add_link(mass=0.4, inertia=inertia)
    joints = [template.add_joint_fixed(parent=-1, child=parent)]
    for index in range(6):
        body = template.add_link(mass=0.4, com=wp.vec3(0.02, 0.01, 0.06), inertia=inertia)
        joints.append(
            template.add_joint_revolute(
                parent=parent,
                child=body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.03, 0.0, 0.04), wp.quat_identity()),
                target_ke=2.0,
                target_kd=0.1,
                limit_lower=-0.04,
                limit_upper=0.04,
            )
        )
        if index in (0, 5):
            template.add_shape_sphere(body=body, radius=0.01)
        parent = body
    template.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.4, -0.9, -9.2))
    builder.add_world(template)
    builder.add_world(template)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.linspace(-0.06, 0.06, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(-0.1, 0.1, model.joint_dof_count, dtype=np.float32))
    return model


def install_contacts(model, state, contacts):
    """Six concrete same-input contacts; local witnesses are held during the short step test."""
    pairs = np.array([[0, 3], [0, 1], [2, 3], [4, 7], [4, 5], [6, 7]], dtype=np.int32)
    shape_body = model.shape_body.numpy()
    poses = state.body_q.numpy().astype(np.float64)
    for endpoint in range(2):
        ids = np.full(8, -1, dtype=np.int32)
        ids[:6] = pairs[:, endpoint]
        getattr(contacts, f"rigid_contact_shape{endpoint}").assign(ids)
        points = np.zeros((8, 3), dtype=np.float32)
        for contact, pair in enumerate(pairs):
            midpoint = np.mean(poses[shape_body[pair], :3], axis=0)
            pose = poses[shape_body[pair[endpoint]]]
            p = midpoint - pose[:3]
            q = pose[3:6]
            points[contact] = p + 2 * np.cross(-q, np.cross(-q, p) + pose[6] * p)
        getattr(contacts, f"rigid_contact_point{endpoint}").assign(points)
        getattr(contacts, f"rigid_contact_margin{endpoint}").zero_()
    contacts.rigid_contact_normal.assign(np.tile(np.array([0.0, 0.0, -1.0], np.float32), (8, 1)))
    contacts.rigid_contact_count.assign(np.array([6], dtype=np.int32))


def install_patch_contacts(model, state, contacts):
    """Contacts that the patch builder links into rings, held during short steps.

    Per world, each star sphere rests on a static support at three contacts in a row: the outer two
    become the ring's anchors (three-row units) and the middle one a normal-only unit. A two-contact
    ring couples the first star sphere to the chain's last sphere, which keeps that sphere's
    coordinate out of the independent schedule.

    The stars slide along z only, so a level support's tangent rows have no Jacobian. The first star's
    support is tilted by 30 degrees and its contacts start 1 mm deep: the depth's correction slides the
    star, its friction reaches the pooled load of a ring whose normal-only unit carries load, and its
    anchors gain tangential phi. The second star rests on a level support at zero depth, where gravity
    keeps its anchors loaded and their history held. Deeper contacts would lift the stars off within a
    step. Only the second world's stars start inside their limits and stay supported.
    """
    shape_body = model.shape_body.numpy()
    poses = state.body_q.numpy().astype(np.float64)
    radius = 0.01
    down = np.array((0.0, 0.0, -1.0))
    tilted = np.array((-0.5, 0.0, -np.sqrt(0.75)))

    def local(shape, point):
        if shape < 0:
            return point
        pose = poses[shape_body[shape]]
        q = -pose[3:6]
        p = point - pose[:3]
        return p + 2 * np.cross(q, np.cross(q, p) + pose[6] * p)

    rows = []
    for world in range(2):
        star0, star1, _, chain = 4 * world, 4 * world + 1, 4 * world + 2, 4 * world + 3
        for shape, normal, depth in ((star0, tilted, 1.0e-3), (star1, down, 0.0)):
            bottom = poses[shape_body[shape], :3] - np.array((0.0, 0.0, radius))
            # The row runs along y, which both supports keep tangent.
            rows += [
                (shape, -1, bottom + np.array((0.0, offset, 0.0)), normal, depth) for offset in (-0.004, 0.0, 0.004)
            ]
        middle = 0.5 * (poses[shape_body[star0], :3] + poses[shape_body[chain], :3])
        rows += [(star0, chain, middle + np.array((0.0, offset, 0.0)), down, 0.0) for offset in (-0.004, 0.004)]
    capacity = contacts.rigid_contact_shape0.shape[0]
    shapes = np.full((2, capacity), -1, dtype=np.int32)
    points = np.zeros((2, capacity, 3), dtype=np.float32)
    normals = np.tile(down, (capacity, 1)).astype(np.float32)
    depths = np.zeros(capacity, dtype=np.float32)
    for contact, (shape0, shape1, point, normal, depth) in enumerate(rows):
        shapes[:, contact] = (shape0, shape1)
        points[0, contact] = local(shape0, point)
        points[1, contact] = local(shape1, point)
        normals[contact] = normal
        depths[contact] = depth
    for endpoint in range(2):
        getattr(contacts, f"rigid_contact_shape{endpoint}").assign(shapes[endpoint])
        getattr(contacts, f"rigid_contact_point{endpoint}").assign(points[endpoint])
    contacts.rigid_contact_margin0.assign(depths)
    contacts.rigid_contact_margin1.zero_()
    contacts.rigid_contact_normal.assign(normals)
    contacts.rigid_contact_count.assign(np.array([len(rows)], dtype=np.int32))


def assert_patch_tangent_law(solver, contacts):
    """Require each anchor's tangent rows to carry its patch phi and ``friction_anchor_beta``.

    Returns the number of anchors checked (with or without a spin row). ``contact_rows`` keeps phi for the normal row
    only.
    """
    count = min(int(contacts.rigid_contact_count.numpy()[0]), solver.placement.contact_slot.shape[0])
    world, slot = solver.placement.contact_world.numpy()[:count], solver.placement.contact_slot.numpy()[:count]
    path, needed = solver.placement.contact_path.numpy()[:count], solver.placement.contact_slots_needed.numpy()[:count]
    phi, beta, view = (
        solver.placement.phi.numpy(),
        solver.placement.row_beta.numpy(),
        solver.laws.friction_patches.view.phi.numpy(),
    )
    anchors = 0
    for c in range(count):
        if path[c] != 0 or slot[c] < 0 or needed[c] < 3:
            continue
        tangents = slice(int(slot[c]) + 1, int(slot[c]) + 3)
        np.testing.assert_array_equal(phi[world[c], tangents], view[c], err_msg=f"contact {c} phi")
        np.testing.assert_array_equal(
            beta[world[c], tangents], np.float32(solver.laws.friction_anchor_beta), err_msg=f"contact {c} beta"
        )
        anchors += 1
    return anchors


def check_star_records(test, solver):
    """Check and return the contact-row record of ``build_solver_fixture``'s solver: the one writer names the builders
    of the diagonal-mass size (108) and the dense six. The marker encodes schedule links into the sparse rows (the
    patch friction's units)."""
    record = solver.get_path_selection()["routes"]["contact_rows"]
    marker = solver.sweep.sparse_contact_marker.key
    expected = {"108": ("walk", ["write_slot_rows", marker]), "6": ("walk", ["write_slot_rows"])}
    test.assertEqual(record["producer"], "walk")
    test.assertEqual(
        {size: (entry["method"], entry["kernels"]) for size, entry in record["builder_by_dof"].items()}, expected
    )
    return record


def ring_carries_friction(records, first):
    """Whether contact ``first``'s patch ring, walked through ``contact_rows`` parents, holds friction."""
    contact = first
    for _ in range(len(records)):
        record = records[contact]
        if len(record["types"]) >= 3 and np.any(record["impulses"][1:] != 0.0):
            return True
        contact = int(record["parents"][0])
        if contact < 0 or contact == first:
            return False
    return False


class TestStarRows(unittest.TestCase):
    def test_actual_fixture_layout_and_cpu_fallback(self):
        """Exercise the real model/Contacts field binding, and the solver's construction on the CPU."""
        model = build_solver_fixture("cpu")
        self.assertEqual(model.joint_dof_count, 228)
        self.assertEqual(model.shape_count, 8)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts = newton.Contacts(8, 0, device=model.device)
        install_contacts(model, state, contacts)
        self.assertEqual(contacts.rigid_contact_count.numpy()[0], 6)
        newton.solvers.SolverFeatherPGS(model, dense_max_constraints=64)

    def test_cuda_actual_solver_step_reset_notify_and_graphs(self):
        """Use the actual allocator, scheduler, RHS and eight-sweep solve with loaded contacts.

        The contacts run the unit schedule; the route record names exactly the row builders the first step launches.
        """
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual sparse constructor and parallel-stream capture require CUDA")
        for device in devices:
            model = build_solver_fixture(device)
            solvers = [
                newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_iterations=8,
                    update_mass_matrix_interval=2,
                    enable_joint_limits=True,
                    dense_max_constraints=64,
                    mf_max_constraints=64,
                )
            ]
            for solver in solvers:
                self.assertTrue(solver.spec.classes.diagonal.units)
            records = [check_star_records(self, solver) for solver in solvers]
            states = [[model.state(), model.state()] for _ in solvers]
            controls = [model.control() for _ in solvers]
            contacts = [newton.Contacts(8, 0, device=device) for _ in solvers]
            for index in range(len(solvers)):
                state = states[index][0]
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                install_contacts(model, state, contacts[index])

            def compare(states=states, solvers=solvers):
                for pair in states[1:]:
                    for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(pair[0], field).numpy(),
                            getattr(states[0][0], field).numpy(),
                            rtol=3e-5,
                            atol=5e-6,
                        )
                for solver in solvers:
                    solver.check_constraint_capacity()
                    self.assertGreater(int(np.sum(solver.placement.constraint_count.numpy())), 0)
                    self.assertTrue(np.all(np.isfinite(solver.v_out.numpy())))

            for step in range(6):
                if step == 3:
                    model.body_mass.assign(model.body_mass.numpy() * 1.1)
                    for solver in solvers:
                        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                for index, solver in enumerate(solvers):
                    current, following = states[index]
                    if step == 2:
                        solver.reset(current, wp.array([True, False], dtype=bool, device=device))
                    with launched_kernels() as keys:
                        solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    if step == 0:
                        assert_row_builders_named(self, records[index], keys)
                    states[index] = [following, current]
                compare()
            graphs = []
            for index, solver in enumerate(solvers):
                current, following = states[index]
                solver.reset(current)
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    solver.step(following, current, controls[index], contacts[index], 1 / 240)
                graphs.append(capture.graph)
            for _ in range(2):
                for graph in graphs:
                    wp.capture_launch(graph)
                compare()

    def test_cuda_patch_rows_keep_the_patch_law(self):
        """Write persistent patch rows by the one writer under the sparse unit schedule and default patch friction:
        normal-only units link into their rings, loaded rings carry friction, the tangents keep the patch law, and
        the rings schedule independently or, on a coupled coordinate, serially.
        """
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual sparse constructor requires CUDA")
        for device in devices:
            model = build_solver_fixture(device)
            solvers = [
                newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_iterations=8,
                    enable_joint_limits=True,
                    dense_max_constraints=64,
                    mf_max_constraints=64,
                )
            ]
            for solver in solvers:
                self.assertTrue(solver.spec.classes.diagonal.units)
            records = [check_star_records(self, solver) for solver in solvers]
            states = [[model.state(), model.state()] for _ in solvers]
            controls = [model.control() for _ in solvers]
            contacts = [newton.Contacts(24, 0, device=device) for _ in solvers]
            for index in range(len(solvers)):
                state = states[index][0]
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                install_patch_contacts(model, state, contacts[index])
            linked = pooled = False
            independent = [False] * len(solvers)
            serial = [False] * len(solvers)
            for step in range(6):
                for index, solver in enumerate(solvers):
                    current, following = states[index]
                    with launched_kernels() as keys:
                        solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    if step == 0:
                        assert_row_builders_named(self, records[index], keys)
                    states[index] = [following, current]
                reference = contact_rows(solvers[0], contacts[0])
                for index in range(1, len(solvers)):
                    for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(states[index][0], field).numpy(),
                            getattr(states[0][0], field).numpy(),
                            rtol=3e-5,
                            atol=5e-6,
                        )
                    rows = contact_rows(solvers[index], contacts[index])
                    assert_rows_close(reference, rows)
                    assert_contact_impulses_close(reference, rows)
                    assert_history_close(solvers[0], solvers[index])
                for index, solver in enumerate(solvers):
                    solver.check_constraint_capacity()
                    assert_ring_coulomb(solver)
                    self.assertGreater(assert_patch_tangent_law(solver, contacts[index]), 0)
                    independent[index] |= bool(np.any(solver.placement.sparse_contact_level_count.numpy() > 0))
                    serial[index] |= bool(np.any(solver.placement.sparse_contact_serial_count.numpy() > 0))
                linked |= any(len(record["types"]) == 1 and record["parents"][0] >= 0 for record in reference.values())
                pooled |= any(
                    ring_carries_friction(reference, c)
                    for c, record in reference.items()
                    if len(record["types"]) == 1 and record["impulses"][0] > 0.0
                )
            self.assertTrue(linked, "A normal-only unit must link into a patch ring")
            self.assertTrue(pooled, "A loaded normal-only unit must share a ring that carries friction")
            # The level support's ring schedules independently; the coupled ring keeps the tilted support's ring
            # serial, in speculative batches.
            self.assertEqual(independent, [True] * len(solvers), "A patch ring must schedule independently")
            self.assertEqual(serial, [True] * len(solvers), "Patch rings on a coupled coordinate must stay serial")
            self.assertTrue(solvers[0].laws.friction_patches.previous.valid.numpy().any(), "Anchors must carry history")


class TestCoordinateLevels(unittest.TestCase):
    def test_levels_step_as_the_rows_in_order(self):
        """Under the general placement the star's units of one entry step by level before the rows in order (the
        coordinate levels): the limits and the level support's patch ring take levels, the ring coupled to the chain
        stays in order, and the steps are the rows in order bit for bit (the schedule emptied)."""
        for device in ("cpu", *(str(d) for d in wp.get_cuda_devices())):
            model = build_solver_fixture(device)
            runs = []
            for empty in (False, True):
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_iterations=8,
                    enable_joint_limits=True,
                    dense_max_constraints=320,
                    pins=newton.solvers.SolverFeatherPGS.Pins(placement="general"),
                )
                self.assertIn("_levels", solver.get_path_selection()["routes"]["sweep"]["mf_gs"])
                schedule, leveled = solver.sweep.schedule_diagonal, []

                def emptied(schedule=schedule, solver=solver, empty=empty, leveled=leveled):
                    schedule()
                    placement = solver.placement
                    counts = placement.sparse_contact_level_count.numpy()
                    offsets = placement.sparse_contact_level_offsets.numpy()
                    leveled.append([int(offsets[w, c]) for w, c in enumerate(counts)])
                    if empty:
                        placement.sparse_contact_level_count.zero_()

                solver.sweep.schedule_diagonal = emptied
                state, following, control = model.state(), model.state(), model.control()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                contacts = newton.Contacts(24, 0, device=device)
                install_patch_contacts(model, state, contacts)
                for _ in range(6):
                    solver.step(state, following, control, contacts, 1 / 240)
                    state, following = following, state
                solver.check_constraint_capacity()
                runs.append((leveled, {f: getattr(state, f).numpy() for f in ("joint_q", "joint_qd", "body_q")}))
            # Every world levels its limits and its level support's ring; the coupled ring's rows stay in order.
            self.assertTrue(all(min(step) > 0 for step in runs[0][0]), runs[0][0])
            for field, value in runs[0][1].items():
                np.testing.assert_array_equal(value, runs[1][1][field], err_msg=f"{device} {field}")


if __name__ == "__main__":
    unittest.main()

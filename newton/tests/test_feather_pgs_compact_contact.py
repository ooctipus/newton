# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The compact boundary against the sparse diagonal's one writer (FL3.5): one row law and walk, two mappings.

The compact boundary writes the sparse diagonal's contact rows where it reads the contacts
(:func:`.compact_contact.produce_contacts`); without it the one writer walks them from their unit sources
(:func:`.walk_rows.write_slot_rows`). Both call the same row functions, so on a real two-world model (a 108-key star
beside a six-coordinate chain) their steps, rows, impulses and anchor history agree. The rows' arithmetic is held to
the walk's invariants in ``test_feather_pgs_walk_rows`` (:class:`TestSlotFrame`).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import compact_contact as compact
from newton.tests.feather_pgs_patch_parity import (
    assert_contact_impulses_close,
    assert_history_close,
    assert_ring_coulomb,
    assert_rows_close,
    contact_rows,
)
from newton.tests.feather_pgs_row_builders import assert_row_builders_named, launched_kernels


def _unsupported(solver) -> bool:
    return False


def compact_boundary(enabled: bool):
    """Admit the compact contact boundary where the data supports it, or withhold it from the sparse-diagonal owner.

    Withheld, the one writer walks the rows from their unit sources, as for a dense articulation other than six
    coordinates.
    """
    return mock.patch.object(compact, "supported", compact.supported if enabled else _unsupported)


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

    Returns the number of anchors checked. ``contact_rows`` keeps phi for the normal row only.
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
        if path[c] != 0 or slot[c] < 0 or needed[c] != 3:
            continue
        tangents = slice(int(slot[c]) + 1, int(slot[c]) + 3)
        np.testing.assert_array_equal(phi[world[c], tangents], view[c], err_msg=f"contact {c} phi")
        np.testing.assert_array_equal(
            beta[world[c], tangents], np.float32(solver.laws.friction_anchor_beta), err_msg=f"contact {c} beta"
        )
        anchors += 1
    return anchors


# The two solver arms of the CUDA tests: the sparse diagonal's one writer, then the compact boundary.
_ARMS = ("walk", "compact")


def check_compact_records(test, solvers, law):
    """Check and return the contact-row records of the two arms of ``build_solver_fixture`` under ``law``.

    Each names the builders of the diagonal-mass size (108) and the dense six with the friction law's
    variants. The marker encodes schedule links into the sparse rows: of point friction's triples, or of
    patch friction's units.
    """
    records = []
    for solver, arm in zip(solvers, _ARMS, strict=True):
        record = solver.get_path_selection()["routes"]["contact_rows"]
        marker = solver.sweep.sparse_contact_marker.key
        # The one writer walks every row from its unit source; the boundary writes the contact rows and the writer
        # its limit prefix.
        producer = "walk" if arm == "walk" else "compact_contact_boundary"
        first = "write_contact_units" if arm == "walk" else "produce_contacts"
        expected = {
            "108": (producer, [first, "write_slot_rows", marker]),
            "6": (producer, [first, "write_slot_rows"]),
        }
        test.assertEqual(record["producer"], producer)
        test.assertEqual(record["law"], law)
        test.assertEqual(
            {size: (entry["method"], entry["kernels"]) for size, entry in record["builder_by_dof"].items()}, expected
        )
        records.append(record)
    return records


def ring_carries_friction(records, first):
    """Whether contact ``first``'s patch ring, walked through ``contact_rows`` parents, holds friction."""
    contact = first
    for _ in range(len(records)):
        record = records[contact]
        if len(record["types"]) == 3 and np.any(record["impulses"][1:] != 0.0):
            return True
        contact = int(record["parents"][0])
        if contact < 0 or contact == first:
            return False
    return False


class TestCompactContact(unittest.TestCase):
    def test_exact_mode_admission(self):
        """Reject incomplete mode compositions before any contact ownership changes.

        The sparse-diagonal owner's own admission (its fused limits: augmented drives, no velocity-limit, mimic or
        connect rows, no debug rows or pre-elimination) is the owner's, which its case (K2) stands for here.
        """
        valid = {
            "model": SimpleNamespace(device=SimpleNamespace(is_cuda=True)),
            "routes": SimpleNamespace(case="K2"),
            "_compact_diagonal_mass_size": 108,
            "sparse_diagonal_dense_size": 6,
            "sparse_diagonal_response_size": 108,
            "sparse_diagonal_contact_triples": True,
            "sparse_diagonal_contact_units": False,
            "dense_max_constraints": 40,
            "dense_internal_max_rows": 12,
        }

        def probe(fields):
            """The class's fields on the solver, the placement's on its placement."""
            owners = ("model", "routes", "_compact_diagonal_mass_size")
            placement = {name: value for name, value in fields.items() if name not in owners}
            return SimpleNamespace(placement=SimpleNamespace(**placement), **{name: fields[name] for name in owners})

        self.assertTrue(compact.supported(probe(valid)))
        # Patch friction schedules units instead of triples.
        units = {**valid, "sparse_diagonal_contact_triples": False, "sparse_diagonal_contact_units": True}
        self.assertTrue(compact.supported(probe(units)))
        for name, value in (
            ("routes", SimpleNamespace(case="K6")),
            ("sparse_diagonal_dense_size", 7),
            ("sparse_diagonal_contact_triples", False),
            ("dense_max_constraints", 11),
            ("dense_internal_max_rows", 13),
            ("_compact_diagonal_mass_size", None),
        ):
            self.assertFalse(compact.supported(probe({**valid, name: value})), name)

    def test_actual_fixture_layout_and_cpu_fallback(self):
        """Exercise the real model/Contacts field binding and exact unsupported CPU fallback."""
        model = build_solver_fixture("cpu")
        self.assertEqual(model.joint_dof_count, 228)
        self.assertEqual(model.shape_count, 8)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts = newton.Contacts(8, 0, device=model.device)
        install_contacts(model, state, contacts)
        self.assertEqual(contacts.rigid_contact_count.numpy()[0], 6)
        solver = newton.solvers.SolverFeatherPGS(model, dense_max_constraints=64)
        self.assertFalse(solver.placement.compact_contact_boundary)

    def test_cuda_actual_solver_step_reset_notify_and_graphs(self):
        """Use the actual allocator, scheduler, RHS and eight-sweep solve with loaded contacts.

        Point friction runs the triple schedule; each arm's route record names exactly the row
        builders its first step launches.
        """
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual sparse constructor and parallel-stream capture require CUDA")
        for device in devices:
            model = build_solver_fixture(device)
            solvers = []
            for enabled in (False, True):
                with (
                    compact_boundary(enabled),
                ):
                    solvers.append(
                        newton.solvers.SolverFeatherPGS(
                            model,
                            pgs_iterations=8,
                            friction_anchor_beta=0.0,
                            update_mass_matrix_interval=2,
                            enable_joint_limits=True,
                            dense_max_constraints=64,
                            mf_max_constraints=64,
                        )
                    )
            self.assertFalse(solvers[0].placement.compact_contact_boundary)
            self.assertTrue(solvers[1].placement.compact_contact_boundary)
            self.assertLessEqual(solvers[1].placement.dense_internal_max_rows, 12)
            for solver in solvers:
                self.assertTrue(solver.placement.sparse_diagonal_contact_triples)
            records = check_compact_records(self, solvers, "point")
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

    def test_cuda_patch_rows_match_the_original_producers(self):
        """Write persistent patch rows from the compact boundary as the original producers do.

        All arms run the sparse unit schedule under default patch friction; the compact arms replace only
        the row producer, so rows by contact, impulses, states and anchor history must agree.
        """
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual sparse constructor requires CUDA")
        for device in devices:
            model = build_solver_fixture(device)
            solvers = []
            for enabled in (False, True):
                with (
                    compact_boundary(enabled),
                ):
                    solvers.append(
                        newton.solvers.SolverFeatherPGS(
                            model,
                            pgs_iterations=8,
                            enable_joint_limits=True,
                            dense_max_constraints=64,
                            mf_max_constraints=64,
                        )
                    )
            for solver, enabled in zip(solvers, (False, True), strict=True):
                self.assertTrue(solver.laws.friction_anchors_enabled)
                self.assertTrue(solver.placement.sparse_diagonal_contact_units)
                self.assertEqual(solver.placement.compact_contact_boundary, enabled)
            records = check_compact_records(self, solvers, "patch")
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
                    independent[index] |= bool(np.any(solver.placement.sparse_contact_group_count.numpy() > 0))
                    serial[index] |= bool(np.any(solver.placement.sparse_contact_serial_count.numpy() > 0))
                linked |= any(len(record["types"]) == 1 and record["parents"][0] >= 0 for record in reference.values())
                pooled |= any(
                    ring_carries_friction(reference, c)
                    for c, record in reference.items()
                    if len(record["types"]) == 1 and record["impulses"][0] > 0.0
                )
            self.assertTrue(linked, "A normal-only unit must link into a patch ring")
            self.assertTrue(pooled, "A loaded normal-only unit must share a ring that carries friction")
            # In every arm the level support's ring schedules independently; the coupled ring keeps the tilted
            # support's ring serial, in speculative batches.
            self.assertEqual(independent, [True] * len(solvers), "A patch ring must schedule independently")
            self.assertEqual(serial, [True] * len(solvers), "Patch rings on a coupled coordinate must stay serial")
            self.assertTrue(solvers[0].laws.friction_patches.previous.valid.numpy().any(), "Anchors must carry history")


if __name__ == "__main__":
    unittest.main()

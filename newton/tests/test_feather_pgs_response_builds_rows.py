# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free-body row build of the response operators on its own: the row word the sweeps decode, the rows' whitening
by the dynamics' mass matrix, and the records the rows' writers store.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.core.memory import Place
from newton._src.solvers.feather_pgs.row_metadata import (
    HEADER_WORDS,
    pack_row_record,
    record_views,
    unpack_mf_header,
    validate_record_rows,
)
from newton.tests.feather_pgs_law_harness import unpacked_rows, walk_inverse

DEVICE = "cpu"


class TestFreeBodyRowsBuild(unittest.TestCase):
    """The free-body rows' build."""

    def test_meta_word_is_the_consumed_word(self):
        """The packed word's fields are the ones the sweeps' row loop decodes."""
        word = pack_row_record("w", HEADER_WORDS, dof_a="a", dof_b="b", diagonal="inv", indent="")
        self.assertIn("record_a.x = (a << 16) | (b & 0xFFFF);", word)
        self.assertIn("record_a.y = __float_as_int(inv);", word)
        consume = unpack_mf_header("w", "")
        self.assertIn("int dof_a = packed_dofs >> 16;", consume)
        self.assertIn("int dof_b = (packed_dofs << 16) >> 16;", consume)
        self.assertIn("float mf_diag = __int_as_float(w.y);", consume)
        with self.assertRaises(ValueError):
            validate_record_rows(70000)

    def _box_step(self, armature: float = 0.0, pins=None):
        """A 1 kg, 0.2 m box resting tilted on the ground (its lowest edge 5 mm into it), one step: the solver."""
        import newton  # noqa: PLC0415

        mass, half = 1.0, 0.1
        builder = newton.ModelBuilder()
        rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)
        principal = 2.0 * mass * half * half / 3.0
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), rotation),
            mass=mass,
            inertia=wp.mat33(principal, 0.0, 0.0, 0.0, principal, 0.0, 0.0, 0.0, principal),
        )
        builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.joint_armature[-6:] = [armature] * 6
        builder.add_ground_plane()
        model = builder.finalize(device=DEVICE)
        solver = newton.solvers.SolverFeatherPGS(model, pins=newton.solvers.SolverFeatherPGS.Pins(**(pins or {})))
        pipeline = newton.CollisionPipeline(model, growable_contacts=True)
        contacts = pipeline.contacts()
        state, out = model.state(), model.state()
        pipeline.collide(state, contacts)
        solver.step(state, out, model.control(), contacts, 1.0 / 240.0)
        return solver

    def test_writers_store_the_rows_records(self):
        """Every streamed row's record (:func:`.row_metadata.pack_row_record`), which the one writer stores, holds the
        row as the step left it: its sides' first coordinates and widths, its diagonal, its residual at ``du = 0``, its
        kind, parent and friction coefficient. Under the pin ``mf_storage='line'`` every row streams its record."""
        solver = self._box_step(pins={"mf_storage": "line"})
        p = solver.placement
        self.assertTrue(p.writes_records)
        self.assertEqual(p.record_first, 0)
        rows = int(p.constraint_count.numpy()[0])
        self.assertGreater(rows, 0, "the box must touch the ground")
        line = int(p.dense_first.numpy()[0])
        record = record_views(p.row_record.numpy(), p.free_records)[0][line][: rows - p.record_first]
        offset = p.forest.articulation_world_dof_offset.numpy()
        dofs = p.forest.articulation_response_dof_count.numpy()
        sides = p.row_sides.numpy()[line, p.record_first : rows]
        first_a = np.where(sides[:, 0] >= 0, offset[np.maximum(sides[:, 0], 0)], -1)
        first_b = np.where(sides[:, 1] >= 0, offset[np.maximum(sides[:, 1], 0)], -1)
        width_a = np.where(sides[:, 0] >= 0, dofs[np.maximum(sides[:, 0], 0)], 0)
        width_b = np.where(sides[:, 1] >= 0, dofs[np.maximum(sides[:, 1], 0)], 0)
        np.testing.assert_array_equal(record[:, 0] >> 16, first_a)
        np.testing.assert_array_equal((record[:, 0] << 16) >> 16, first_b)
        np.testing.assert_array_equal(record[:, 4], width_a | (width_b << 16))
        rows_at = slice(p.record_first, rows)
        np.testing.assert_array_equal(record[:, 1].view(np.float32), p.diag.numpy()[line, rows_at])
        residual = p.rhs.numpy()[line, rows_at] + p.row_incident.numpy()[line, rows_at]
        np.testing.assert_array_equal(record[:, 2].view(np.float32), residual)
        np.testing.assert_array_equal(record[:, 3] & 0xFFFF, p.row_type.numpy()[line, rows_at])
        np.testing.assert_array_equal(record[:, 3] >> 16, p.row_parent.numpy()[line, rows_at])
        np.testing.assert_array_equal(record[:, 5].view(np.float32), p.row_mu.numpy()[line, rows_at])

    def test_rows_see_the_dynamics_mass_matrix(self):
        """Whiten a free box's contact rows by the factor of its dynamics' mass matrix, armature included.

        A 1 kg, 0.2 m box rests tilted on the ground with a free-joint armature of 0.05 kg m^2, several times its
        principal inertia (1/150 kg m^2): every row's diagonal is ``J H^-1 J^T + cfm`` with ``H`` the box's spatial
        inertia at its center of mass plus the armature, and the rows hold ``Z = L^-1 J^T`` with ``L L^T = H``.
        """
        import newton  # noqa: PLC0415

        armature, mass, half = 0.05, 1.0, 0.1
        builder = newton.ModelBuilder()
        rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)
        principal = 2.0 * mass * half * half / 3.0
        body = builder.add_body(
            # Its lowest edge 5 mm into the ground: the rows exist (a world whose contacts separate needs none).
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), rotation),
            mass=mass,
            inertia=wp.mat33(principal, 0.0, 0.0, 0.0, principal, 0.0, 0.0, 0.0, principal),
        )
        builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.joint_armature[-6:] = [armature] * 6
        builder.add_ground_plane()
        model = builder.finalize(device=DEVICE)
        solver = newton.solvers.SolverFeatherPGS(model)
        pipeline = newton.CollisionPipeline(model, growable_contacts=True)
        contacts = pipeline.contacts()
        state, out = model.state(), model.state()
        pipeline.collide(state, contacts)
        solver.step(state, out, model.control(), contacts, 1.0 / 240.0)
        # A free body is a one-node tree: its contact rows are the line's, packed at the world's coordinates.
        rows = int(solver.placement.constraint_count.numpy()[0])
        self.assertGreater(rows, 0, "the box must touch the ground")
        # H at the center of mass: m on the linear block, the world-frame inertia on the angular one, plus armature.
        basis = np.array(wp.quat_to_matrix(rotation), dtype=np.float64).reshape(3, 3)
        inertia = basis @ (model.body_inertia.numpy()[body].astype(np.float64)) @ basis.T
        H = np.zeros((6, 6))
        H[:3, :3] = mass * np.eye(3)
        H[3:, 3:] = inertia
        H += armature * np.eye(6)
        L = np.linalg.inv(walk_inverse(solver, 0))  # the walk: H = L L^T
        np.testing.assert_allclose(L @ L.T, H, rtol=1e-5, atol=1e-6)
        # The box is each row's one body (the ground has none): the row's Jacobian over the world's six coordinates.
        J = unpacked_rows(solver)[0][0, :rows]
        expected = np.einsum("ri,ij,rj->r", J, np.linalg.inv(H), J) + solver.options.pgs_cfm
        np.testing.assert_allclose(solver.placement.diag.numpy()[0, :rows], expected, rtol=1e-5)


def _box_run(device, pins=None, steps: int = 12):
    """A tilted box on the ground (its lowest edge 5 mm into it), ``steps`` steps: the most rows of the line a step
    held, the solver and the box's state."""
    import newton  # noqa: PLC0415
    from newton.solvers import SolverFeatherPGS  # noqa: PLC0415

    builder = newton.ModelBuilder()
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), rotation))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pins=SolverFeatherPGS.Pins(**(pins or {})))
    pipeline = newton.CollisionPipeline(model, growable_contacts=True)
    contacts = pipeline.contacts()
    state, out = model.state(), model.state()
    rows = 0
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, out, model.control(), contacts, 1.0 / 240.0)
        rows = max(rows, int(solver.placement.constraint_count.numpy()[0]))
        state, out = out, state
    return rows, solver, state.body_q.numpy(), state.body_qd.numpy()


class TestLineImpulses(unittest.TestCase):
    """The line's impulses read and written in place, where the line is past the sweep's stage."""

    def test_line_impulses_in_place_hold_the_staged_values(self):
        """Every impulse in place (the test pin ``mf_storage='line'``) steps bit for bit as staged ones: both places
        hold the same values."""
        devices = ["cpu"] + (["cuda:0"] if wp.get_cuda_device_count() > 0 else [])
        for device in devices:
            with self.subTest(device=device):
                _, solver, q, qd = _box_run(device, {"mf_storage": "line"})
                self.assertIs(solver.sweep.general_sweep.places.dense_impulses, Place.GLOBAL)
                rows, _, q0, qd0 = _box_run(device)
                self.assertGreater(rows, 0, "the box must touch the ground")
                np.testing.assert_array_equal(q, q0)
                np.testing.assert_array_equal(qd, qd0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

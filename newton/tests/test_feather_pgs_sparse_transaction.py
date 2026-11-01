# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the sparse sweep's contact units: their block records, their kernels and their twins.

A contact unit solves a normal and its friction pair on their Gram block, in exact arithmetic the ordinary sweep's
row-by-row updates, so its CUDA twins match the ordinary sweep and the physical reference to rounding, not bitwise.
"""

import re
import unittest
from itertools import product

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import environment_flags
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_FRICTION as FRICTION
from newton._src.solvers.feather_pgs.sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_sparse_contact_triplets as triplets
from newton.tests.test_feather_pgs_native_prelude import _native_snippets
from newton.tests.test_feather_pgs_path_selection import _cpu_model
from newton.tests.test_feather_pgs_sparse_pgs import _sweep

TRIPLETS = {"active_rows": True, "use_support_table": True, "contact_triplets": True}
# Row capacities of the suite's two launch-bound families: shared memory admits 12 blocks of 256 rows on the
# RTX PRO 6000, and the register budget's 16 of 96.
CAPACITIES = (256, 96)


def _snippet(kernel):
    """Return the sweep body a sparse sweep kernel compiles."""
    return max(_native_snippets(kernel), key=len)


def _resized(data, capacity):
    """Return a DeviceSolve fixture re-padded to another row capacity."""
    if int(np.max(data["world_constraint_count"])) > capacity:
        raise ValueError("the fixture's rows exceed the capacity")
    resized = dict(data)
    for key in (
        "rhs_bias",
        "diagonal",
        "initial_impulses",
        "row_factor",
        "row_incident",
        "row_type",
        "row_parent",
        "row_mu",
        "support_row_class",
    ):
        value = data[key]
        rows = min(capacity, value.shape[1])
        resized[key] = np.zeros((value.shape[0], capacity, *value.shape[2:]), value.dtype)
        resized[key][:, :rows] = value[:, :rows]
    return resized


def _assert_twins(test, data, unit, ordinary, *, rtol=1.0e-5):
    """Match the ordinary sweep's velocity delta and live impulses to ``rtol`` of each array's magnitude."""
    counts = data["world_constraint_count"]
    live = np.arange(ordinary[1].shape[1])[None] < counts[:, None]
    for name, actual, expected in (
        ("delta", unit[0], ordinary[0]),
        ("impulses", unit[1][live], ordinary[1][live]),
    ):
        test.assertTrue(np.isfinite(actual).all(), name)
        magnitude = float(np.max(np.abs(expected), initial=0.0))
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * magnitude, err_msg=name)


def _ring_members(parents, normal):
    """Return the other normals of ``normal``'s ring in the order the pair update pools them."""
    members, row = [], int(parents[normal])
    while row >= 0 and row != normal:
        members.append(row)
        row = int(parents[row])
    return members


class TestContactRecordBuilder(unittest.TestCase):
    """The sparse contact builder writes each contact triplet's block record beside its rows."""


class TestContactUnits(unittest.TestCase):
    """The sparse sweep's contact units: factory, sweep body and route record."""

    def test_factory_reads_records_and_states_launch_bounds_cpu(self):
        """Solve fused triplets as contact units from their records under either law, with per-shape launch bounds."""
        for capacity in (256, 192, 128, 96, 48):
            with self.subTest(capacity=capacity):
                for law in ("patch", "point"):
                    kernel = sparse_kernel(capacity, 18, 12, linked_rows=law == "patch", **TRIPLETS)
                    linked = "_linked" if law == "patch" else ""
                    self.assertTrue(kernel.key.endswith(f"_contact_triplets{linked}"), kernel.key)
                    # The occupancy its shared memory admits, at most the register budget's 16 blocks.
                    threads, blocks = kernel.options["launch_bounds"]
                    self.assertEqual(threads, 64)
                    self.assertLessEqual(blocks, 16)
                    self.assertEqual(blocks == 16, capacity <= 128, (capacity, blocks))
                    # The records follow the scalar inputs, before the outputs.
                    names = list(kernel.adj.arg_types)
                    self.assertEqual(names[names.index("omega") + 1], "contact_records")
        with self.assertRaises(TypeError):
            sparse_kernel(64, 18, 12, **TRIPLETS, contact_transaction="single")

    def test_contact_unit_reduces_once_from_records_cpu(self):
        """Reduce a unit's three residuals once, its pair's moved through the record's couplings; no metric cache.

        Only rings of more than four members walk their parents on a visit. Lane 0 writes the
        impulses every lane read only after a warp barrier.
        """
        for linked in (False, True):
            with self.subTest(linked=linked):
                snippet = _snippet(sparse_kernel(256, 18, 12, linked_rows=linked, **TRIPLETS))
                for text in ("metric_map", "metric_count", "friction_pair_cached(", "contact_metric_slot("):
                    self.assertNotIn(text, snippet)
                width = CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS * linked
                self.assertRegex(snippet, rf"(normal|i) / 3\) \* {width}")
                start = snippet.index("const int tangent_parent = world_row_parent.data[offset + tangent];")
                unit = snippet[start : snippet.index("consumed_tangent = tangent;", start)]
                # One reduction of the three residuals together, and no other; the rows then run the row law on the
                # unit's operator (sweep.contact_unit): the pair's terms moved through the record's couplings, the
                # normal stepped by the record's 1/W_nn, the pair's law from the record's metric.
                self.assertEqual(unit.count("SUM("), 1)
                self.assertIn("SUM(unit_jv0, unit_jv1, unit_jv2);", unit)
                self.assertNotIn("__shfl", unit)
                self.assertIn("ProjectionLaw::friction_pair_from_metric(tangent_scale, basis.x, basis.y,", unit)
                self.assertIn(
                    "const float gram_n1 = coupling.x, gram_n2 = coupling.y, unit_inverse = coupling.w;", unit
                )
                self.assertIn("float jv = r == 0 ? unit_jv0 : unit_jv1 + gram_n1 * unit_dn;", unit)
                self.assertIn(
                    "r == 0 ? -residual * unit_inverse : -residual / (r == 0 ? unit_diag0 : unit_diag1)", unit
                )
                self.assertEqual("normal_parent < 0" in unit, not linked)
                # Linked visits pool listed members and walk only rings longer than four members.
                self.assertEqual("ring.w < 0" in unit, linked)
                self.assertEqual("make_int4(first, second, third, others)" in snippet, linked)
                self.assertEqual(unit.count("RING_LOAD("), int(linked))
                # Every lane reads the unit's staged values (impulses, right-hand sides, diagonals) with its rows,
                # before the reduction; each row's law then orders every lane's reads before lane 0's writes with a
                # warp barrier: shuffles order execution, not memory.
                reduction = unit.index("SUM(")
                for held in (
                    "unit_lam1 = s_impulse[head + 1]",
                    "unit_rhs2 = s_rhs[head + 2]",
                    "unit_diag1 = s_diag[head + 1]",
                ):
                    self.assertLess(unit.index(held), reduction)
                self.assertLess(unit.index("SYNC();"), unit.index("if (lane == 0) s_impulse[i] = new_impulse;"))
                # The rows read those registers and pool the ring from the normal's new impulse in a register, so
                # they read no store of the unit: only the normal's stores wait for the lanes (beside the gate's own
                # barrier), and the unit synchronizes once, after its apply. Nor do they read a staged right-hand side
                # or diagonal, which would wait for the normal's barrier.
                rows = unit[unit.index("for (int r = 0; r < 2; ++r) {") : unit.index("if (unit_dn != 0.0f ||")]
                self.assertEqual(re.findall(r"s_impulse\[[^\]]*\](?! = )", rows), [])
                self.assertEqual(re.findall(r"s_(?:rhs|diag)\[", rows), [])
                self.assertIn("if (r == 0) SYNC();", rows)
                self.assertEqual(rows.count("SYNC();"), 1)
                self.assertRegex(snippet[start:], r"consumed_tangent = tangent;\s+SYNC\(\);")

    def test_route_record_names_no_transaction_cpu(self):
        """Record the sparse route without a transaction field or switch: contact units are the one law."""
        solver = SolverFeatherPGS(_cpu_model())
        record = solver.get_path_selection()["routes"]["sparse_mass_matrix"]
        self.assertNotIn("contact_transaction", record)
        self.assertNotIn("launch_bounds", record)
        self.assertNotIn("FEATHER_PGS_SPARSE_TRANSACTION", environment_flags.FLAGS)
        self.assertNotIn("FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS", environment_flags.FLAGS)


@unittest.skipUnless(wp.is_cuda_available(), "The sparse sweep requires CUDA")
class TestContactUnitTwins(unittest.TestCase):
    """CUDA twins of the contact units: the ordinary sweep (the same law, row by row) and the physical reference, to
    rounding."""

    def _twins(self, data, *, linked=False, active=True):
        """Launch the sweep with contact units and the ordinary one on ``data``; return both results and the unit
        solve."""
        unit = triplets.DeviceSolve(data, candidate=True, active=active, linked=linked)
        ordinary = triplets.DeviceSolve(data, candidate=False, active=active, linked=linked)
        self.assertTrue(unit.units)
        self.assertFalse(ordinary.units)
        unit.launch()
        ordinary.launch()
        return unit.result(), ordinary.result(), unit

    def test_point_units_match_ordinary_and_reference_cuda(self):
        """Solve the point-law fixtures as the ordinary sweep and the physical reference do, at every bound."""
        cases = (
            [{"offset": value, "seeded": True} for value in (0, 29, 30)]
            + [{"activity": value} for value in ("late_activation", "deferred_activation", "seeded_retraction")]
            + [{"omega": value, "seeded": True} for value in (0.0, 0.7, 1.2, -0.5)]
        )
        for settings, capacity in product(cases, CAPACITIES):
            with self.subTest(capacity=capacity, **settings):
                data, physical = triplets.fixture(**settings)
                data = _resized(data, capacity)
                unit, ordinary, _ = self._twins(data)
                _assert_twins(self, data, unit, ordinary)
                triplets.assert_output(self, data, *unit, physical=physical)

    def test_near_singular_tangents_match_the_reference_cuda(self):
        """Solve a nearly rank-one tangent block to the reference tolerance; rounding is amplified there."""
        data, physical = triplets.fixture(activity="near_singular")
        for capacity in CAPACITIES:
            with self.subTest(capacity=capacity):
                resized = _resized(data, capacity)
                unit, ordinary, _ = self._twins(resized)
                triplets.assert_output(self, resized, *unit, physical=physical, reference=ordinary)

    def test_patch_rings_match_ordinary_and_reference_cuda(self):
        """Pool rings of one to six members, listed or walked, in the linked active scan and the full sweep."""
        settings = ({}, {"seeded": True, "omega": 0.7})
        for (name, layout), pad, options, active, capacity in product(
            triplets._LINKED_LAYOUTS.items(), (0, 30), settings, (True, False), CAPACITIES
        ):
            with self.subTest(layout=name, pad=pad, active=active, capacity=capacity, **options):
                data, physical = triplets.linked_fixture(layout, pad=pad, active=active, **options)
                data = _resized(data, capacity)
                unit, ordinary, _ = self._twins(data, linked=True, active=active)
                _assert_twins(self, data, unit, ordinary)
                triplets.assert_output(self, data, *unit, physical=physical)

    def test_ring_members_follow_the_pooling_walk_cuda(self):
        """List up to three other members of each anchor's ring in walk order, and mark longer rings to walk.

        Rings of one to six members load anchors whose radii pool the whole ring, so a
        missing, repeated or reordered member changes the impulses beyond rounding.
        """
        for name in (
            "self_loop",
            "two_member_ring",
            "three_member_ring",
            "far_ring",
            "loaded_ring",
            "long_loaded_ring",
        ):
            for capacity, active in product(CAPACITIES, (True, False)):
                with self.subTest(layout=name, capacity=capacity, active=active):
                    data, physical = triplets.linked_fixture(triplets._LINKED_LAYOUTS[name], pad=30, active=active)
                    data = _resized(data, capacity)
                    unit, ordinary, solve = self._twins(data, linked=True, active=active)
                    _assert_twins(self, data, unit, ordinary)
                    triplets.assert_output(self, data, *unit, physical=physical)
                    rings = solve.records.numpy()[..., CONTACT_RECORD_FLOATS:].view(np.int32)
                    for world, count in enumerate(data["world_constraint_count"]):
                        kinds, parents = data["row_type"][world], data["row_parent"][world]
                        for row in range(int(count) - 2):
                            if (kinds[row], kinds[row + 1], kinds[row + 2]) != (CONTACT, FRICTION, FRICTION):
                                continue
                            members = _ring_members(parents, row)
                            ring = rings[world, row // 3]
                            listed = len(members) if len(members) <= 3 else -1
                            self.assertEqual(int(ring[3]), listed)
                            self.assertEqual(list(ring[: max(listed, 0)]), members[: max(listed, 0)])

    def test_seventeen_loaded_triplets_cuda(self):
        """Solve 17 loaded triplets in one world from their records."""
        pairs = 17
        dofs = pairs * 3
        indices = np.arange(dofs, dtype=np.int32).reshape(pairs, 3)
        z = np.zeros((dofs, dofs), dtype=np.float32)
        bias = np.zeros(dofs, dtype=np.float32)
        kinds = np.tile(np.array([CONTACT, FRICTION, FRICTION], np.int32), pairs)
        parents = np.full(dofs, -1, dtype=np.int32)
        mu = np.zeros(dofs, dtype=np.float32)
        for pair in range(pairs):
            row = pair * 3
            z[row, row] = 1.0 + 0.01 * pair
            z[row + 1, row : row + 3] = (0.11, 0.6 + 0.08 * pair, 0.13)
            z[row + 2, row : row + 3] = (-0.07, 0.19, 1.8 - 0.05 * pair)
            bias[row : row + 3] = (-0.2 - 0.01 * pair, 0.4, 0.7 - 0.02 * pair)
            parents[row + 1 : row + 3] = row
            mu[row + 1 : row + 3] = 0.4 + 0.005 * pair
        diag = np.sum(z * z, axis=1) + 0.01
        initial = np.zeros(dofs, dtype=np.float32)
        velocity, impulses = _sweep(z, z, initial, bias, diag, kinds, parents, mu, initial, active_rows=True)
        self.assertEqual(np.count_nonzero(impulses[::3] > 0), pairs)
        physical = {
            "factor": np.eye(dofs),
            "initial": initial,
            "expected_velocity": velocity,
            "expected_impulse": impulses,
        }
        for capacity in CAPACITIES:
            with self.subTest(capacity=capacity):

                def padded(values, capacity=capacity):
                    result = np.zeros((3, capacity, *values.shape[1:]), dtype=values.dtype)
                    result[:, :dofs] = values
                    return result

                data = {
                    "world_constraint_count": np.array([dofs, 0, dofs], np.int32),
                    "rhs_bias": padded(bias),
                    "diagonal": padded(diag),
                    "initial_impulses": padded(initial),
                    "row_factor": padded(np.stack([z[row, indices[row // 3]] for row in range(dofs)])),
                    "row_incident": padded(initial),
                    "row_type": padded(kinds),
                    "row_parent": padded(parents),
                    "row_mu": padded(mu),
                    "support_row_class": padded(np.repeat(np.arange(pairs, dtype=np.int32), 3)),
                    "support_class_dof": indices,
                    "iterations": np.array(8),
                    "omega": np.array(1.0),
                    "reference_factor_delta": np.zeros((3, dofs), dtype=np.float32),
                }
                unit, ordinary, _ = self._twins(data)
                _assert_twins(self, data, unit, ordinary)
                triplets.assert_output(self, data, *unit, physical=physical)

    def test_seventeen_linked_anchors_cuda(self):
        """Pool 17 loaded anchors of self-loop and two-member rings in one world.

        Each ring owns its coordinates, so its anchor's block couples the normal and the tangents
        while rings stay independent. Odd anchors share their ring with a second loaded normal on
        a fourth coordinate, whose impulse joins the anchor's radius; every anchor slides, so a
        ring list or a record that missed an anchor past the twelfth would leave the ring-pooled
        physical sweep.
        """
        anchors = 17
        dofs = 3 * anchors + anchors // 2
        # Each ring's coordinates in one support class, padded to the two-member rings' four.
        indices = np.full((anchors, 4), -1, dtype=np.int32)
        kinds, parents, mu, bias, rows, units = [], [], [], [], [], []
        first = 0
        for anchor in range(anchors):
            ring = 4 if anchor % 2 else 3
            indices[anchor, :ring] = np.arange(first, first + ring)
            normal, columns = len(kinds), slice(first, first + 3)
            block = np.zeros((4, dofs))
            block[0, columns] = (1.0 + 0.01 * anchor, 0.0, 0.0)
            block[1, columns] = (0.11, 0.6 + 0.08 * anchor, 0.13)
            block[2, columns] = (-0.07, 0.19, 1.8 - 0.05 * anchor)
            kinds += [CONTACT, FRICTION, FRICTION]
            parents += [normal, normal, normal]
            mu += [0.0, 0.4 + 0.005 * anchor, 0.4 + 0.005 * anchor]
            bias += [-0.2 - 0.01 * anchor, 0.4, 0.7 - 0.02 * anchor]
            rows += list(block[:3])
            units += [anchor] * 3
            if anchor % 2:
                # A second loaded normal of the anchor's ring.
                block[3, first : first + 4] = (0.3, 0.1, 0.0, 1.0)
                parents[normal] = normal + 3
                kinds.append(CONTACT)
                parents.append(normal)
                mu.append(0.0)
                bias.append(-0.2)
                rows.append(block[3])
                units.append(anchor)
            first += ring
        self.assertEqual(first, dofs)
        count = len(kinds)
        kinds, parents = np.array(kinds, np.int32), np.array(parents, np.int32)
        z = np.array(rows, np.float32)
        bias, mu = np.array(bias, np.float32), np.array(mu, np.float32)
        diag = np.sum(z.astype(np.float64) ** 2, axis=1).astype(np.float32) + np.float32(0.01)
        normals = np.flatnonzero(kinds == CONTACT)
        anchor_rows = normals[kinds[np.minimum(normals + 1, count - 1)] == FRICTION]
        self.assertEqual(len(anchor_rows), anchors)
        oracle = {}
        for active in (True, False):
            velocity, impulses = _sweep(
                z,
                z,
                np.zeros(dofs),
                bias,
                diag,
                kinds,
                parents,
                mu,
                np.zeros(count),
                active_rows=active,
                linked_rows=True,
            )
            # Every normal holds load and every anchor slides on its pooled radius.
            self.assertTrue(np.all(impulses[normals] > 0.0))
            for row in anchor_rows:
                radius = mu[row + 1] * (impulses[row] + impulses[parents[row]] * (parents[row] != row))
                self.assertAlmostEqual(float(np.hypot(*impulses[row + 1 : row + 3])), float(radius), places=5)
            oracle[active] = {
                "factor": np.eye(dofs),
                "initial": np.zeros(dofs),
                "expected_velocity": velocity,
                "expected_impulse": impulses,
            }
        factor = np.stack(
            [np.where(indices[unit] >= 0, z[row, indices[unit]], 0.0) for row, unit in enumerate(units)]
        ).astype(np.float32)
        for capacity, active in product(CAPACITIES, (True, False)):
            with self.subTest(capacity=capacity, active=active):

                def padded(values, capacity=capacity):
                    result = np.zeros((3, capacity, *values.shape[1:]), dtype=values.dtype)
                    result[:, :count] = values
                    return result

                data = {
                    "world_constraint_count": np.array([count, 0, count], np.int32),
                    "rhs_bias": padded(bias),
                    "diagonal": padded(diag),
                    "initial_impulses": padded(np.zeros(count, np.float32)),
                    "row_factor": padded(factor),
                    "row_incident": padded(np.zeros(count, np.float32)),
                    "row_type": padded(kinds),
                    "row_parent": padded(parents),
                    "row_mu": padded(mu),
                    "support_row_class": padded(np.array(units, np.int32)),
                    "support_class_dof": indices,
                    "iterations": np.array(8),
                    "omega": np.array(1.0),
                    "reference_factor_delta": np.zeros((3, dofs), dtype=np.float32),
                }
                unit, ordinary, solve = self._twins(data, linked=True, active=active)
                _assert_twins(self, data, unit, ordinary)
                triplets.assert_output(self, data, *unit, physical=oracle[active])
                rings = solve.records.numpy()[..., CONTACT_RECORD_FLOATS:].view(np.int32)
                for row in anchor_rows:
                    members = _ring_members(parents, row)
                    self.assertEqual(list(rings[0, row // 3, : len(members)]), members)
                    self.assertEqual(int(rings[0, row // 3, 3]), len(members))

    def test_fused_visits_step_the_normal_with_the_record_cuda(self):
        """Step a fused normal with its record's ``1/W_nn``: zeroed, it holds the normal the ordinary sweep moves.

        A triplet the unit's guards decline takes the ordinary update, so matching the ordinary twin
        to rounding cannot show that the fused branch ran. Zeroing ``1/W_nn`` in the record of one
        loaded triplet holds that normal at its initial impulse through every sweep, which only the
        fused branch can do, for point triplets and linked anchors in the active scan and the full
        sweep.
        """
        fixtures = [("point", True, triplets.fixture(offset=30, seeded=True)[0])]
        for active in (True, False):
            layout = triplets._LINKED_LAYOUTS["loaded_ring"]
            fixtures.append(("patch", active, triplets.linked_fixture(layout, pad=30, seeded=True, active=active)[0]))
        for (law, active, fixture), capacity in product(fixtures, CAPACITIES):
            with self.subTest(law=law, active=active, capacity=capacity):
                data = _resized(fixture, capacity)
                linked = law == "patch"
                ordinary = triplets.DeviceSolve(data, candidate=False, active=active, linked=linked)
                ordinary.launch()
                initial = data["initial_impulses"]
                moved = np.abs(ordinary.result()[1] - initial)
                kinds, parents = data["row_type"][0], data["row_parent"][0]
                loaded = [
                    row
                    for row in range(int(data["world_constraint_count"][0]) - 2)
                    if (kinds[row], kinds[row + 1], kinds[row + 2]) == (CONTACT, FRICTION, FRICTION)
                    and parents[row + 1] == parents[row + 2] == row
                    and moved[0, row] > 1.0e-3
                ]
                self.assertTrue(loaded)
                row = loaded[0]
                unit = triplets.DeviceSolve(data, candidate=True, active=active, linked=linked)
                records = unit.records.numpy()
                records[:, row // 3, 3] = 0.0
                unit.records.assign(records)
                unit.launch()
                impulses = unit.result()[1]
                live = np.flatnonzero(data["world_constraint_count"])
                np.testing.assert_array_equal(impulses[live, row], initial[live, row])
                self.assertTrue(np.all(moved[live, row] > 1.0e-3))

    @staticmethod
    def _rollout(solver, model, *, steps=20, dt=1.0 / 400.0):
        """Step a resting world and a sliding world; return the first solve's impulses and each step's state."""
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
        contacts, control = pipeline.contacts(), model.control()
        state, output = model.state(), model.state()
        qd = state.joint_qd.numpy()
        qd[model.joint_qd_start.numpy()[model.articulation_start.numpy()[1]]] = 0.6
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        trajectory = []
        for step in range(steps):
            pipeline.collide(state, contacts)
            solver.step(state, output, control, contacts, dt)
            state.assign(output)
            if step == 0:
                counts = solver.placement.constraint_count.numpy()
                impulses = solver.placement.impulses.numpy()
                first = np.concatenate([impulses[world, :count] for world, count in enumerate(counts)])
            trajectory.append((first, state.body_q.numpy().copy(), state.body_qd.numpy().copy()))
        return trajectory


if __name__ == "__main__":
    unittest.main()

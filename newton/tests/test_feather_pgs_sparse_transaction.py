# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the single-stage sparse contact transaction: its block records, its kernels and their twins.

The transaction applies the paired friction law of the staged transaction, so its CUDA
twins match the staged sweep and the physical reference to rounding, not bitwise.
"""

import os
import re
import unittest
from itertools import product
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import environment_flags
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_FRICTION as FRICTION
from newton._src.solvers.feather_pgs.kernels import populate_world_J_for_compact_size
from newton._src.solvers.feather_pgs.response_builds_rows import factor_contact_rows_kernel
from newton._src.solvers.feather_pgs.sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from newton._src.solvers.feather_pgs.sparse_mass_matrix import row_supports
from newton._src.solvers.feather_pgs.sweep_routes_sparse import contact_transaction_launch_bounds, sparse_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_sparse_contact_triplets as triplets
from newton.tests.feather_pgs_patch_parity import row_law, standing_model
from newton.tests.test_feather_pgs_native_prelude import _native_snippets
from newton.tests.test_feather_pgs_path_selection import _cpu_model
from newton.tests.test_feather_pgs_sparse_contacts import _contact_input, _fixture
from newton.tests.test_feather_pgs_sparse_pgs import _sweep
from newton.tests.test_feather_pgs_sparse_solver import _make_solver

TRIPLETS = {"active_rows": True, "use_support_table": True, "contact_triplets": True}
# Row capacities of the suite's two launch-bound families: 12 blocks above 128 rows, 16 at or below.
CAPACITIES = (256, 96)
# Row capacities with the minimum blocks a single-stage sweep requests: the 14-block variant
# (FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS=14) compiles to another register budget at 128 rows or fewer.
SHAPES = ((256, 16), (96, 16), (96, 14))


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


def _assert_twins(test, data, single, staged, *, rtol=1.0e-5):
    """Match the staged transaction's velocity delta and live impulses to ``rtol`` of each array's magnitude."""
    counts = data["world_constraint_count"]
    live = np.arange(staged[1].shape[1])[None] < counts[:, None]
    for name, actual, expected in (
        ("delta", single[0], staged[0]),
        ("impulses", single[1][live], staged[1][live]),
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

    def _build(
        self, device, records, shared, friction_shared, patch, *, lanes=8, workers=1, width=CONTACT_RECORD_FLOATS
    ):
        """Run the prepared support-table builder on the two-branch fixture; return its outputs.

        ``lanes`` per contact and ``workers`` lane groups share the three contacts; ``width`` is
        the record stride, wider for a linked sweep's ring members.
        """

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        f = _fixture(device)
        plan = f["plan"]
        # A separately prescribed articulation on contact zero, as in the builder's preparation test.
        f["shape_body"] = array([1, 2, 3, -1, 0], int)
        f["shape1"] = array([4, 1, 3], int)
        f["art_b"] = array([1, 0, -1], int)
        f["group"] = array([0, 0], int)
        f["start"] = array([0, 6], int)
        f["origin"] = array([[0.3, -0.4, 0.7], [-0.2, 0.1, 0.4]], wp.vec3)
        f["patches"].phi = array([[0.01, -0.02], [-0.03, 0.04], [0.0, 0.0]], wp.vec2)
        f["patches"].enabled = patch
        response_dofs = array([6, 0], int)
        law = {
            "body_v": array(np.zeros((4, 6), dtype=np.float32), wp.spatial_vector),
            "prescribed": array([0, 1], int),
            "material_mu": array([0.8, 0.5, 0.3, 0.6, 0.4]),
            "material_restitution": array([0.4, 0.2, 0.5, 0.8, 0.6]),
            "beta": 0.05,
            "cfm": 0.03,
            "friction_scale": 0.8,
            "speculative_scale": 0.65,
            "restitution_threshold": 0.1,
            "dt": 0.01,
        }
        for name in ("row_type", "row_parent"):
            law[name] = full((1, 9), -9, int)
        for name in ("row_mu", "row_beta", "row_cfm", "phi", "target", "restitution", "rhs"):
            law[name] = full((1, 9), -9.0)
        support_plan = f["tree"].support_classes(plan.joint_ancestor_mask, 6)
        support = row_supports(support_plan, 1, 9, device)
        dense = full((1, 9, 6), 0.0)
        compact = _contact_input(
            f,
            device,
            response_dofs=response_dofs,
            body_mask=f["physical_mask"],
            shared_anchor=shared,
            friction_shared_anchor=friction_shared,
        )
        wp.launch(
            populate_world_J_for_compact_size,
            dim=(1, 32),
            inputs=[compact, 1, 6, f["group"], row_law(f["patches"])],
            outputs=[dense],
            device=device,
        )
        options = {"prepare_rows": True, "use_support_table": True}
        if records:
            options["contact_records"] = True
        record = full((1, 3, width), -7.0)
        outputs = {
            "row_factor": full((1, 9, 6), -9.0),
            "incident": full((1, 9), -9.0),
            "diagonal": full((1, 9), -9.0),
        }
        data = _contact_input(
            f,
            device,
            response_dofs=response_dofs,
            shared_anchor=shared,
            friction_shared_anchor=friction_shared,
            diag=outputs["diagonal"],
            **law,
        )
        wp.launch(
            factor_contact_rows_kernel(6, lanes_per_contact=lanes, **options),
            dim=workers * lanes,
            inputs=[
                data,
                workers,
                f["group"],
                full(response_dofs.shape, 0, int),
                full(response_dofs.shape, 0, int),
                f["sparse_mask"],
                row_law(f["patches"], 0.23),
                f["indices"].permutation,
                f["indices"].row_offsets,
                f["packed"],
                array(np.eye(6)[None]),
            ],
            outputs=[
                full((1, 1, 1), -9, int),
                outputs["row_factor"],
                record if records else full((1, 9, 12), -9.0),
                outputs["incident"],
                support,
            ],
            block_dim=128,
            device=device,
        )
        # The independent oracle: dense contact rows through the inverse factor, with the rows' CFM.
        expected = dense.numpy()[0, :7][:, plan.permutation] @ f["inverse"].T
        results = {name: value.numpy() for name, value in outputs.items()}
        results["rhs"] = law["rhs"].numpy()
        return results, record.numpy(), expected, law["cfm"]

    def _check_records(self, device, *, lanes=8, workers=1):
        """Check the records of the two full triplets against NumPy blocks, and the rows against the plain builder."""
        cases = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1))
        widths = (CONTACT_RECORD_FLOATS, CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS)
        for (shared, friction_shared, patch), width in product(cases, widths):
            with self.subTest(shared=shared, friction_shared=friction_shared, patch=patch, width=width):
                options = {"lanes": lanes, "workers": workers}
                results, record, expected, cfm = self._build(
                    device, True, shared, friction_shared, patch, width=width, **options
                )
                plain, _, _, _ = self._build(device, False, shared, friction_shared, patch, **options)
                for name, value in results.items():
                    np.testing.assert_array_equal(value, plain[name], err_msg=name)
                for triplet, row in ((0, 0), (1, 3)):
                    block = expected[row : row + 3] @ expected[row : row + 3].T + cfm * np.eye(3)
                    np.testing.assert_allclose(
                        record[0, triplet, :4],
                        (block[0, 1], block[0, 2], block[1, 2], 1.0 / block[0, 0]),
                        rtol=2.0e-5,
                        atol=2.0e-6,
                    )
                    # The eigenbasis relative to the larger tangent diagonal: largest along (vx, vy).
                    tangents = block[1:, 1:] / max(block[1, 1], block[2, 2])
                    largest, smallest, vx, vy = record[0, triplet, 4:8]
                    np.testing.assert_allclose(
                        (largest, smallest, vx, vy),
                        triplets.pair_metric(block[1, 1], block[1, 2], block[2, 2]),
                        rtol=2.0e-5,
                        atol=2.0e-6,
                    )
                    np.testing.assert_allclose(sorted((smallest, largest)), np.linalg.eigvalsh(tangents), atol=2e-6)
                    np.testing.assert_allclose(tangents @ (vx, vy), largest * np.array((vx, vy)), atol=2e-6)
                    self.assertGreater(abs(block[0, 1]) + abs(block[0, 2]), 1e-3)
                np.testing.assert_array_equal(record[0, 2], -7.0)
                # The builder leaves a linked sweep's ring members to the sweep.
                np.testing.assert_array_equal(record[0, :, CONTACT_RECORD_FLOATS:], -7.0)

    def test_records_hold_each_triplets_block_cpu(self):
        """Write W_nt1, W_nt2, W_t1t2, 1/W_nn and the pair law's tangent eigenbasis for every full triplet.

        Contacts at rows 0 and 3 carry three rows, records 0 and 1; the one-row contact at
        row 6 writes nothing. The rows the builder writes are those of the plain builder.
        """
        self._check_records("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Cooperative contact construction requires CUDA")
    def test_records_hold_each_triplets_block_cuda(self):
        """Reduce the cross terms in each lane group and write every record as the CPU builder does.

        On CUDA the lanes of a contact's group reduce the three cross products with shuffles and
        lane 0 writes the record; the rows stay bitwise those of the plain CUDA builder, across
        group widths and grid strides.
        """
        for lanes, workers in ((8, 1), (8, 2), (16, 2), (32, 2)):
            with self.subTest(lanes=lanes, workers=workers):
                self._check_records("cuda:0", lanes=lanes, workers=workers)

    def test_records_require_prepared_support_rows_cpu(self):
        """Build records only beside prepared rows with the support table, whose routes leave the slot free."""
        for options in ({"use_support_table": True}, {"prepare_rows": True}, {}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "records"):
                factor_contact_rows_kernel(6, contact_records=True, **options)
        kernel = factor_contact_rows_kernel(6, prepare_rows=True, use_support_table=True, contact_records=True)
        self.assertTrue(kernel.key.endswith("_prepared_support_table_contact_records"))


class TestSingleStageTransaction(unittest.TestCase):
    """The single-stage transaction's factory, sweep body, request and route record."""

    def test_factory_resolves_the_law_default_and_launch_bounds_cpu(self):
        """Take the single-stage transaction for linked rows by default, with per-shape launch bounds and records.

        Point friction keeps the staged transaction unless asked; either law reaches either form.
        """
        for capacity, blocks in product((256, 192, 128, 96, 48), (16, 14)):
            bounds = contact_transaction_launch_bounds(capacity, transaction_blocks=blocks)
            self.assertEqual(bounds, (64, 12 if capacity > 128 else blocks))
            with self.subTest(capacity=capacity, blocks=blocks):
                kernels = {
                    (law, transaction): sparse_kernel(
                        capacity,
                        18,
                        12,
                        linked_rows=law == "patch",
                        contact_transaction=transaction,
                        transaction_blocks=blocks,
                        **TRIPLETS,
                    )
                    for law, transaction in product(("patch", "point"), ("auto", "single", "staged"))
                }
                for (law, transaction), kernel in kernels.items():
                    single = transaction == "single" or (transaction == "auto" and law == "patch")
                    linked = "_linked" if law == "patch" else ""
                    suffix = f"_contact_triplets_single{bounds[1]}{linked}" if single else f"_contact_triplets{linked}"
                    self.assertTrue(kernel.key.endswith(suffix), kernel.key)
                    self.assertEqual(kernel.options.get("launch_bounds"), bounds if single else None)
                    names = list(kernel.adj.arg_types)
                    self.assertEqual("contact_records" in names, single)
                    if single:
                        # The records follow the scalar inputs, before the outputs.
                        self.assertEqual(names[names.index("iteration_offset") + 1], "contact_records")
                self.assertIs(kernels[("patch", "auto")], kernels[("patch", "auto")])
        for options in ({"contact_transaction": "fused"}, {"transaction_blocks": 12}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sparse_kernel(64, 18, 12, **TRIPLETS, **options)
        # The block request is an option, so it is keyword-only.
        with self.assertRaises(TypeError):
            contact_transaction_launch_bounds(96, 14)

    def test_single_stage_sweep_reduces_once_from_records_cpu(self):
        """Reduce three residuals once from the record, without the staged metric cache, its capacity or walk.

        The staged sweep keeps its cache, its map and its second reduction; only rings of
        more than four members walk their parents on a visit. Lane 0 writes the impulses
        every lane read only after a warp barrier.
        """
        for linked in (False, True):
            with self.subTest(linked=linked):
                single = _snippet(
                    sparse_kernel(256, 18, 12, linked_rows=linked, contact_transaction="single", **TRIPLETS)
                )
                staged = _snippet(
                    sparse_kernel(256, 18, 12, linked_rows=linked, contact_transaction="staged", **TRIPLETS)
                )
                for text in (
                    "metric_map",
                    "metric_count",
                    "friction_pair_cached(",
                    "contact_metric_slot(",
                ):
                    self.assertIn(text, staged)
                    self.assertNotIn(text, single)
                width = CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS * linked
                self.assertRegex(single, rf"(normal|i) / 3\) \* {width}")
                start = single.index("const int tangent_parent = world_row_parent.data[offset + tangent];")
                transaction = single[start : single.index("consumed_tangent = tangent;", start)]
                # One reduction of the three residuals together, and no other.
                self.assertEqual(transaction.count("SUM("), 1)
                self.assertIn("SUM(normal_dot, tangent_dot, sibling_dot);", transaction)
                self.assertNotIn("__shfl", transaction)
                self.assertIn("ProjectionLaw::friction_pair_from_metric(tangent_scale, basis.x, basis.y,", transaction)
                self.assertIn("wp::vec2(tangent_residual + coupling.x * normal_change,", transaction)
                self.assertIn("normal_residual * coupling.w", transaction)
                self.assertEqual("normal_parent < 0" in transaction, not linked)
                # Linked visits pool listed members and walk only rings longer than four members.
                self.assertEqual("ring.w >= 0" in transaction, linked)
                self.assertEqual("make_int4(first, second, third, others)" in single, linked)
                self.assertEqual(transaction.count("RING_LOAD("), int(linked))
                # Every lane reads the visit's impulses with its rows, before the reduction, and a
                # warp barrier orders all reads before lane 0's first write: shuffles order
                # execution, not memory.
                self.assertIn("SYNC();", transaction)
                reduction = transaction.index("SUM(")
                barrier = transaction.index("SYNC();")
                self.assertLess(transaction.index("old_tangent = s_impulse[tangent]"), reduction)
                self.assertLess(transaction.index("old_normal = s_impulse[i]"), reduction)
                self.assertLess(barrier, transaction.index("if (lane == 0) s_impulse[i] = new_normal;"))
                self.assertEqual(re.findall(r"impulse\[[^\]]+\](?!\s*=[^=])", transaction[barrier:]), [])

    def test_request_follows_the_law_and_the_flag_cpu(self):
        """Give patch friction the single-stage transaction and point friction the staged one unless the flag says."""
        for environment, patch, expected, phrase in (
            ({}, True, "single", "Patch friction"),
            ({}, False, "staged", "Point friction"),
            ({"FEATHER_PGS_SPARSE_TRANSACTION": "auto"}, True, "single", "by default"),
            ({"FEATHER_PGS_SPARSE_TRANSACTION": "staged"}, True, "staged", "=staged"),
            ({"FEATHER_PGS_SPARSE_TRANSACTION": "single"}, False, "single", "=single"),
            ({"FEATHER_PGS_SPARSE_TRANSACTION": ""}, False, "staged", "Point friction"),
        ):
            with (
                self.subTest(environment=environment, patch=patch),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                probe = SimpleNamespace(
                    _friction_anchors_enabled=patch,
                    _sparse_transaction_request=SolverFeatherPGS._sparse_transaction_flags(),
                )
                transaction, reason = SolverFeatherPGS._request_sparse_contact_transaction(probe)
                self.assertEqual(transaction, expected)
                self.assertIn(phrase, reason)
        for environment, blocks in (({}, 16), ({"FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS": "14"}, 14)):
            with mock.patch.dict(os.environ, environment, clear=True):
                self.assertEqual(SolverFeatherPGS._sparse_transaction_flags(), ("auto", blocks))
        for name, value in (
            ("FEATHER_PGS_SPARSE_TRANSACTION", "fused"),
            ("FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS", "12"),
        ):
            with (
                self.subTest(name=name),
                mock.patch.dict(os.environ, {name: value}, clear=True),
                self.assertRaisesRegex(ValueError, name),
            ):
                SolverFeatherPGS._sparse_transaction_flags()
        # =staged turns the patch default off; 14 blocks opts into a non-default launch bound.
        self.assertEqual(environment_flags.FLAGS["FEATHER_PGS_SPARSE_TRANSACTION"].kind, "policy")
        self.assertEqual(environment_flags.FLAGS["FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS"].kind, "experiment")

    def test_invalid_flags_raise_on_every_route_cpu(self):
        """Reject an invalid transaction flag at construction, also where no sparse sweep runs."""
        model = _cpu_model()
        for name, value in (
            ("FEATHER_PGS_SPARSE_TRANSACTION", "fused"),
            ("FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS", "12"),
        ):
            with (
                self.subTest(name=name),
                mock.patch.dict(os.environ, {name: value}),
                self.assertRaisesRegex(ValueError, name),
            ):
                SolverFeatherPGS(model)

    def test_route_record_names_no_transaction_without_sparse_factors_cpu(self):
        """Record the transaction fields of the sparse route even where the route is not selected."""
        solver = SolverFeatherPGS(_cpu_model())
        record = solver.get_path_selection()["routes"]["sparse_mass_matrix"]
        self.assertIsNone(record["contact_transaction"])
        self.assertIsNone(record["contact_transaction_launch_bounds"])
        self.assertEqual(record["contact_transaction_reason"], "Native sparse factors are not selected.")


@unittest.skipUnless(wp.is_cuda_available(), "The sparse sweep requires CUDA")
class TestSingleStageTwins(unittest.TestCase):
    """CUDA twins of the single-stage transaction: the staged sweep and the physical reference, to rounding."""

    def _twins(self, data, *, linked=False, active=True, blocks=16):
        """Launch the single-stage and staged transactions on ``data``; return both results and the single solve.

        ``blocks`` is the single-stage sweep's block request, which applies at 128 rows or fewer.
        """
        options = {"candidate": True, "active": active, "linked": linked}
        single = triplets.DeviceSolve(data, transaction="single", blocks=blocks, **options)
        staged = triplets.DeviceSolve(data, transaction="staged", **options)
        self.assertTrue(single.single_stage)
        self.assertFalse(staged.single_stage)
        self.assertIn(f"_single{12 if single.capacity > 128 else blocks}", single.kernel.key)
        single.launch()
        staged.launch()
        return single.result(), staged.result(), single

    def test_point_triplets_match_staged_and_reference_cuda(self):
        """Solve the point-law fixtures as the staged transaction and the physical reference do, at every bound."""
        cases = (
            [{"offset": value, "seeded": True} for value in (0, 29, 30)]
            + [{"activity": value} for value in ("late_activation", "deferred_activation", "seeded_retraction")]
            + [{"omega": value, "seeded": True} for value in (0.0, 0.7, 1.2, -0.5)]
            + [{"delay": 3, "iteration_offset": value, "seeded": True} for value in (0, 1, 4)]
        )
        for settings, (capacity, blocks) in product(cases, SHAPES):
            with self.subTest(capacity=capacity, blocks=blocks, **settings):
                data, physical = triplets.fixture(**settings)
                data = _resized(data, capacity)
                single, staged, _ = self._twins(data, blocks=blocks)
                _assert_twins(self, data, single, staged)
                triplets.assert_output(self, data, *single, physical=physical)

    def test_near_singular_tangents_match_the_reference_cuda(self):
        """Solve a nearly rank-one tangent block to the reference tolerance; rounding is amplified there."""
        data, physical = triplets.fixture(activity="near_singular")
        for capacity in CAPACITIES:
            with self.subTest(capacity=capacity):
                resized = _resized(data, capacity)
                single, staged, _ = self._twins(resized)
                triplets.assert_output(self, resized, *single, physical=physical, reference=staged)

    def test_patch_rings_match_staged_and_reference_cuda(self):
        """Pool rings of one to six members, listed or walked, in the linked active scan and the full sweep."""
        settings = ({}, {"seeded": True, "omega": 0.7}, {"seeded": True, "delay": 3, "iteration_offset": 1})
        for (name, layout), pad, options, active, (capacity, blocks) in product(
            triplets._LINKED_LAYOUTS.items(), (0, 30), settings, (True, False), SHAPES
        ):
            with self.subTest(layout=name, pad=pad, active=active, capacity=capacity, blocks=blocks, **options):
                data, physical = triplets.linked_fixture(layout, pad=pad, active=active, **options)
                data = _resized(data, capacity)
                single, staged, _ = self._twins(data, linked=True, active=active, blocks=blocks)
                _assert_twins(self, data, single, staged)
                triplets.assert_output(self, data, *single, physical=physical)

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
                    single, staged, solve = self._twins(data, linked=True, active=active)
                    _assert_twins(self, data, single, staged)
                    triplets.assert_output(self, data, *single, physical=physical)
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

    def test_more_triplets_than_the_staged_cache_cuda(self):
        """Solve 17 loaded triplets in one world, beyond the staged cache's 12, from their records."""
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
                    "friction_start_iteration": np.array(0),
                    "iteration_offset": np.array(0),
                    "reference_factor_delta": np.zeros((3, dofs), dtype=np.float32),
                }
                single, staged, _ = self._twins(data)
                _assert_twins(self, data, single, staged)
                triplets.assert_output(self, data, *single, physical=physical)

    def test_more_linked_anchors_than_the_staged_cache_cuda(self):
        """Pool 17 loaded anchors of self-loop and two-member rings in one world, beyond the staged cache's 12.

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
        for (capacity, blocks), active in product(SHAPES, (True, False)):
            with self.subTest(capacity=capacity, blocks=blocks, active=active):

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
                    "friction_start_iteration": np.array(0),
                    "iteration_offset": np.array(0),
                    "reference_factor_delta": np.zeros((3, dofs), dtype=np.float32),
                }
                single, staged, solve = self._twins(data, linked=True, active=active, blocks=blocks)
                _assert_twins(self, data, single, staged)
                triplets.assert_output(self, data, *single, physical=oracle[active])
                rings = solve.records.numpy()[..., CONTACT_RECORD_FLOATS:].view(np.int32)
                for row in anchor_rows:
                    members = _ring_members(parents, row)
                    self.assertEqual(list(rings[0, row // 3, : len(members)]), members)
                    self.assertEqual(int(rings[0, row // 3, 3]), len(members))

    def test_fused_visits_step_the_normal_with_the_record_cuda(self):
        """Step a fused normal with its record's ``1/W_nn``: zeroed, it holds the normal where staged moves it.

        A triplet the single-stage guards decline takes the ordinary update, whose sweep equals
        the staged one bit for bit, so matching the staged twin cannot show that the fused branch
        ran. Zeroing ``1/W_nn`` in the record of one loaded triplet holds that normal at its initial
        impulse through every sweep, which only the fused branch can do, for point triplets and
        linked anchors in the active scan and the full sweep.
        """
        fixtures = [("point", True, triplets.fixture(offset=30, seeded=True)[0])]
        for active in (True, False):
            layout = triplets._LINKED_LAYOUTS["loaded_ring"]
            fixtures.append(("patch", active, triplets.linked_fixture(layout, pad=30, seeded=True, active=active)[0]))
        for (law, active, fixture), (capacity, blocks) in product(fixtures, SHAPES):
            with self.subTest(law=law, active=active, capacity=capacity, blocks=blocks):
                data = _resized(fixture, capacity)
                linked = law == "patch"
                options = {"candidate": True, "active": active, "linked": linked}
                staged = triplets.DeviceSolve(data, transaction="staged", **options)
                staged.launch()
                initial = data["initial_impulses"]
                moved = np.abs(staged.result()[1] - initial)
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
                single = triplets.DeviceSolve(data, transaction="single", blocks=blocks, **options)
                records = single.records.numpy()
                records[:, row // 3, 3] = 0.0
                single.records.assign(records)
                single.launch()
                impulses = single.result()[1]
                live = np.flatnonzero(data["world_constraint_count"])
                np.testing.assert_array_equal(impulses[live, row], initial[live, row])
                self.assertTrue(np.all(moved[live, row] > 1.0e-3))

    def test_solver_steps_match_the_staged_transaction_cuda(self):
        """Build records in the contact builder and step with them as the staged transaction steps, both laws.

        One solve from the same state matches the staged sweep's impulses to rounding, through
        the builder's records; a resting and a sliding biped then keep the same trajectory,
        within bounds that leave room for rounding to grow through contact switches. The first
        solve must also differ from the staged one in some bit: the ordinary updates of declined
        triplets equal the staged sweep's, so a bitwise match would mean nothing was fused.
        """
        model = standing_model("cuda:0", worlds=2)
        for law in ("patch", "point"):
            solvers = {}
            for transaction in ("single", "staged"):
                environment = {
                    "FEATHER_PGS_SPARSE_CONTACT_TRIPLETS": "1",
                    "FEATHER_PGS_SPARSE_TRANSACTION": transaction,
                }
                options = {"dense_max_constraints": 96, "pgs_iterations": 8, "enable_joint_limits": False}
                if law == "point":
                    options["friction_anchor_beta"] = 0.0
                with mock.patch.dict(os.environ, environment):
                    solvers[transaction] = _make_solver(model, True, **options)
            with self.subTest(law=law):
                single, staged = solvers["single"], solvers["staged"]
                self.assertEqual(single._sparse_contact_transaction, "single")
                self.assertEqual(staged._sparse_contact_transaction, "staged")
                self.assertIn("_contact_triplets_single16", single._pgs_solve_sparse_kernel.key)
                self.assertIn("_contact_records", single._sparse_contact_response_kernel.key)
                self.assertNotIn("_contact_records", staged._sparse_contact_response_kernel.key)
                width = CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS * (law == "patch")
                self.assertEqual(single._sparse_contact_records.shape, (model.world_count, 32, width))
                self.assertIsNone(staged._sparse_contact_records)
                record = single.get_path_selection()["routes"]["sparse_mass_matrix"]
                self.assertEqual(record["contact_transaction"], "single")
                self.assertEqual(record["contact_transaction_launch_bounds"], [64, 16])
                trajectories = {name: self._rollout(solver, model) for name, solver in solvers.items()}
                # Each step holds (first solve's impulses, body_q, body_qd).
                first_single, first_staged = trajectories["single"][0][0], trajectories["staged"][0][0]
                magnitude = float(np.max(np.abs(first_staged)))
                self.assertGreater(magnitude, 0.0)
                np.testing.assert_allclose(first_single, first_staged, rtol=1e-4, atol=1e-4 * magnitude)
                self.assertFalse(np.array_equal(first_single, first_staged))
                for (_, q_single, qd_single), (_, q_staged, qd_staged) in zip(
                    trajectories["single"][1:], trajectories["staged"][1:], strict=True
                ):
                    self.assertTrue(np.isfinite(q_single).all() and np.isfinite(qd_single).all())
                    np.testing.assert_allclose(q_single, q_staged, atol=1e-3)
                    np.testing.assert_allclose(qd_single, qd_staged, atol=5e-2)

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
                counts = solver.constraint_count.numpy()
                impulses = solver.impulses.numpy()
                first = np.concatenate([impulses[world, :count] for world, count in enumerate(counts)])
            trajectory.append((first, state.body_q.numpy().copy(), state.body_qd.numpy().copy()))
        return trajectory


if __name__ == "__main__":
    unittest.main()

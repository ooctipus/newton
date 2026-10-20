# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the single-stage sparse contact transaction: its block records, its kernels and their twins.

The transaction applies the paired friction law of the staged transaction, so its CUDA
twins match the staged sweep and the physical reference to rounding, not bitwise.
"""

import os
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
from newton._src.solvers.feather_pgs.sparse_contact import (
    CONTACT_RECORD_FLOATS,
    CONTACT_RING_FLOATS,
    SparseContactRows,
    _get_sparse_contact_response_kernel,
)
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupportPlan
from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel, contact_transaction_launch_bounds
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_sparse_contact_triplets as triplets
from newton.tests.feather_pgs_patch_parity import standing_model
from newton.tests.test_feather_pgs_native_prelude import _native_snippets
from newton.tests.test_feather_pgs_path_selection import _cpu_model
from newton.tests.test_feather_pgs_sparse_contacts import _fixture
from newton.tests.test_feather_pgs_sparse_pgs import _sweep
from newton.tests.test_feather_pgs_sparse_solver import _make_solver

TRIPLETS = {"active_rows": True, "use_support_table": True, "contact_triplets": True}
# Row capacities of the suite's two launch-bound families: 12 blocks above 128 rows, 16 at or below.
CAPACITIES = (256, 96)


def _snippet(kernel):
    """Return the sweep body a sparse sweep kernel compiles."""
    return getattr(kernel, "_fpgs_row_native_source", None) or max(_native_snippets(kernel), key=len)


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

    def _build(self, device, records, shared, friction_shared, patch):
        """Run the prepared support-table builder on the two-branch fixture; return its outputs."""

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
        geometry = [f[name] for name in ("point0", "point1", "normal", "shape0", "shape1", "thickness0", "thickness1")]
        rows = SparseContactRows()
        rows.body_velocity = array(np.zeros((4, 6), dtype=np.float32), wp.spatial_vector)
        rows.prescribed_articulation = array([0, 1], int)
        rows.material_friction = array([0.8, 0.5, 0.3, 0.6, 0.4])
        rows.material_restitution = array([0.4, 0.2, 0.5, 0.8, 0.6])
        rows.beta, rows.cfm = 0.05, 0.03
        rows.friction_scale, rows.friction_beta = 0.8, 0.23
        rows.speculative_scale, rows.restitution_velocity_threshold = 0.65, 0.1
        for name in ("type", "parent"):
            setattr(rows, name, full((1, 9), -9, int))
        for name in ("mu", "beta_row", "cfm_row", "phi", "target_velocity", "restitution", "rhs"):
            setattr(rows, name, full((1, 9), -9.0))
        support_plan = _SparseRowSupportPlan.build(plan, plan.joint_ancestor_mask, max_row_dofs=6)
        support = support_plan.to_device(1, 9, device)
        dense = full((1, 9, 6), 0.0)
        wp.launch(
            populate_world_J_for_compact_size,
            dim=(1, 32),
            inputs=[
                f["count"],
                1,
                *geometry,
                f["slot"],
                f["art_a"],
                f["art_b"],
                f["path"],
                f["needed"],
                6,
                response_dofs,
                f["group"],
                f["start"],
                f["origin"],
                f["physical_mask"],
                f["motion"],
                f["shape_body"],
                f["body_q"],
                friction_shared,
                f["patches"],
                shared,
            ],
            outputs=[dense],
            device=device,
        )
        options = {"prepare_rows": True, "use_support_table": True}
        if records:
            options["contact_records"] = True
        record = full((1, 3, CONTACT_RECORD_FLOATS), -7.0)
        outputs = {
            "row_factor": full((1, 9, 6), -9.0),
            "incident": full((1, 9), -9.0),
            "diagonal": full((1, 9), -9.0),
        }
        wp.launch(
            _get_sparse_contact_response_kernel(6, **options),
            dim=8,
            inputs=[
                f["count"],
                1,
                *geometry,
                f["world"],
                f["slot"],
                f["art_a"],
                f["art_b"],
                f["path"],
                f["needed"],
                f["group"],
                response_dofs,
                full(response_dofs.shape, 0, int),
                full(response_dofs.shape, 0, int),
                f["start"],
                f["origin"],
                f["sparse_mask"],
                f["motion"],
                f["shape_body"],
                f["body_q"],
                friction_shared,
                f["patches"],
                shared,
                f["indices"].permutation,
                f["indices"].row_offsets,
                f["packed"],
                array(np.eye(6)[None]),
                f["velocity"],
                rows,
                0.01,
            ],
            outputs=[
                full((1, 1, 1), -9, int),
                outputs["row_factor"],
                record if records else full((1, 9, 12), -9.0),
                outputs["incident"],
                outputs["diagonal"],
                support,
            ],
            block_dim=128,
            device=device,
        )
        # The independent oracle: dense contact rows through the inverse factor, with the rows' CFM.
        expected = dense.numpy()[0, :7][:, plan.permutation] @ f["inverse"].T
        results = {name: value.numpy() for name, value in outputs.items()}
        results["rhs"] = rows.rhs.numpy()
        return results, record.numpy(), expected, float(rows.cfm)

    def test_records_hold_each_triplets_block_cpu(self):
        """Write W_nt1, W_nt2, W_t1t2, 1/W_nn and the pair law's tangent eigenbasis for every full triplet.

        Contacts at rows 0 and 3 carry three rows, records 0 and 1; the one-row contact at
        row 6 writes nothing. The rows the builder writes are those of the plain builder.
        """
        for shared, friction_shared, patch in ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)):
            with self.subTest(shared=shared, friction_shared=friction_shared, patch=patch):
                results, record, expected, cfm = self._build("cpu", True, shared, friction_shared, patch)
                plain, _, _, _ = self._build("cpu", False, shared, friction_shared, patch)
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

    def test_records_require_prepared_support_rows_cpu(self):
        """Build records only beside prepared rows with the support table, whose routes leave the slot free."""
        for options in ({"use_support_table": True}, {"prepare_rows": True}, {}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "records"):
                _get_sparse_contact_response_kernel(6, contact_records=True, **options)
        kernel = _get_sparse_contact_response_kernel(6, prepare_rows=True, use_support_table=True, contact_records=True)
        self.assertTrue(kernel.key.endswith("_prepared_support_table_contact_records"))


class TestSingleStageTransaction(unittest.TestCase):
    """The single-stage transaction's factory, sweep body, request and route record."""

    def test_factory_resolves_the_law_default_and_launch_bounds_cpu(self):
        """Take the single-stage transaction for linked rows by default, with per-shape launch bounds and records.

        Point friction keeps the staged transaction unless asked; either law reaches either form.
        """
        for capacity, blocks in product((256, 192, 128, 96, 48), (16, 14)):
            bounds = contact_transaction_launch_bounds(capacity, blocks)
            self.assertEqual(bounds, (64, 12 if capacity > 128 else blocks))
            with self.subTest(capacity=capacity, blocks=blocks):
                kernels = {
                    (law, transaction): _get_pgs_solve_sparse_kernel(
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
                _get_pgs_solve_sparse_kernel(64, 18, 12, **TRIPLETS, **options)

    def test_single_stage_sweep_reduces_once_from_records_cpu(self):
        """Reduce three residuals once from the record, without the staged metric cache, its capacity or walk.

        The staged sweep keeps its cache, its map and its second reduction; only rings of
        more than four members walk their parents on a visit.
        """
        for linked in (False, True):
            with self.subTest(linked=linked):
                single = _snippet(
                    _get_pgs_solve_sparse_kernel(
                        256, 18, 12, linked_rows=linked, contact_transaction="single", **TRIPLETS
                    )
                )
                staged = _snippet(
                    _get_pgs_solve_sparse_kernel(
                        256, 18, 12, linked_rows=linked, contact_transaction="staged", **TRIPLETS
                    )
                )
                for text in (
                    "metric_storage",
                    "metric_map",
                    "metric_count",
                    "friction_pair_cached(",
                    "contact_metric_slot(",
                ):
                    self.assertIn(text, staged)
                    self.assertNotIn(text, single)
                width = CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS * linked
                self.assertIn(f"normal / 3) * {width}" if linked else f"row / 3) * {width}", single)
                start = single.index("if (row == consumed_tangent) continue;")
                transaction = single[start : single.index("consumed_tangent = tangent;", start)]
                # One interleaved reduction of the three residuals, and no other.
                self.assertEqual(transaction.count("__shfl_down_sync"), 3)
                self.assertEqual(transaction.count("for (int shift = 16; shift > 0; shift >>= 1)"), 1)
                self.assertIn(
                    "friction_pair_from_metric(metric, tangent_residual + coupling.x * normal_change", transaction
                )
                self.assertIn("normal_residual * coupling.w", transaction)
                self.assertEqual("normal_parent < 0" in transaction, not linked)
                # Linked visits pool listed members and walk only rings longer than four members.
                self.assertEqual("ring.w >= 0" in transaction, linked)
                self.assertEqual("make_int4(first, second, third, others)" in single, linked)
                self.assertEqual(transaction.count("world_row_parent.data[offset + next]"), int(linked))

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
                probe = SimpleNamespace(_friction_anchors_enabled=patch)
                transaction, reason = SolverFeatherPGS._request_sparse_contact_transaction(probe)
                self.assertEqual(transaction, expected)
                self.assertIn(phrase, reason)
        for environment, blocks in (({}, 16), ({"FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS": "14"}, 14)):
            with mock.patch.dict(os.environ, environment, clear=True):
                self.assertEqual(SolverFeatherPGS._sparse_transaction_blocks(), blocks)
        for name, value in (
            ("FEATHER_PGS_SPARSE_TRANSACTION", "fused"),
            ("FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS", "12"),
        ):
            with (
                self.subTest(name=name),
                mock.patch.dict(os.environ, {name: value}, clear=True),
                self.assertRaisesRegex(ValueError, name),
            ):
                SolverFeatherPGS._request_sparse_contact_transaction(SimpleNamespace(_friction_anchors_enabled=True))
                SolverFeatherPGS._sparse_transaction_blocks()
        for name in ("FEATHER_PGS_SPARSE_TRANSACTION", "FEATHER_PGS_SPARSE_TRANSACTION_BLOCKS"):
            self.assertEqual(environment_flags.FLAGS[name].kind, "policy")

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

    def _twins(self, data, *, linked=False, active=True):
        """Launch the single-stage and staged transactions on ``data``; return both results and the single solve."""
        single = triplets.DeviceSolve(data, candidate=True, active=active, linked=linked, transaction="single")
        staged = triplets.DeviceSolve(data, candidate=True, active=active, linked=linked, transaction="staged")
        self.assertTrue(single.single_stage)
        self.assertFalse(staged.single_stage)
        single.launch()
        staged.launch()
        return single.result(), staged.result(), single

    def test_point_triplets_match_staged_and_reference_cuda(self):
        """Solve the point-law fixtures as the staged transaction and the physical reference do, at both bounds."""
        cases = (
            [{"offset": value, "seeded": True} for value in (0, 29, 30)]
            + [{"activity": value} for value in ("late_activation", "deferred_activation", "seeded_retraction")]
            + [{"omega": value, "seeded": True} for value in (0.0, 0.7, 1.2, -0.5)]
            + [{"delay": 3, "iteration_offset": value, "seeded": True} for value in (0, 1, 4)]
        )
        for settings, capacity in product(cases, CAPACITIES):
            with self.subTest(capacity=capacity, **settings):
                data, physical = triplets.fixture(**settings)
                data = _resized(data, capacity)
                single, staged, _ = self._twins(data)
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
        for (name, layout), pad, options, active, capacity in product(
            triplets._LINKED_LAYOUTS.items(), (0, 30), settings, (True, False), CAPACITIES
        ):
            with self.subTest(layout=name, pad=pad, active=active, capacity=capacity, **options):
                data, physical = triplets.linked_fixture(layout, pad=pad, active=active, **options)
                data = _resized(data, capacity)
                single, staged, _ = self._twins(data, linked=True, active=active)
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

    def test_solver_steps_match_the_staged_transaction_cuda(self):
        """Build records in the contact builder and step with them as the staged transaction steps, both laws.

        One solve from the same state matches the staged sweep's impulses to rounding, through
        the builder's records; a resting and a sliding biped then keep the same trajectory,
        within bounds that leave room for rounding to grow through contact switches.
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
                (first_single, *_), (first_staged, *_) = trajectories["single"], trajectories["staged"]
                magnitude = float(np.max(np.abs(first_staged)))
                self.assertGreater(magnitude, 0.0)
                np.testing.assert_allclose(first_single, first_staged, rtol=1e-4, atol=1e-4 * magnitude)
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

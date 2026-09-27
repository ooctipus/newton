# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test explicit broad-phase output capacity without truncating input pairs."""

import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.sim import collide as collide_module
from newton.tests.unittest_utils import StdOutCapture


def _model_and_pairs(*, all_overlap=False, boxes=False, device="cpu"):
    """Place the only overlapping pairs at the end of the explicit input list."""
    builder = newton.ModelBuilder()
    positions = np.arange(6) * 0.05 if all_overlap else [0.0, 100.0, 200.0, 300.0, 0.75, 100.75]
    for x in positions:
        body = builder.add_body(xform=wp.transform(wp.vec3(float(x), 0.0, 0.0)))
        if boxes:
            builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        else:
            builder.add_shape_sphere(body, radius=0.5)
    model = builder.finalize(device=device)
    near = [(0, 4), (1, 5)]
    pairs = [pair for pair in itertools.combinations(range(6), 2) if pair not in near] + near
    return model, wp.array(pairs, dtype=wp.vec2i, device=device)


def _pipeline(model, pairs, **kwargs):
    return newton.CollisionPipeline(
        model,
        shape_pairs_filtered=pairs,
        rigid_contact_max=32,
        max_triangle_pairs=16,
        **kwargs,
    )


def _contact_records(contacts):
    n = int(contacts.rigid_contact_count.numpy()[0])
    rows = np.column_stack(
        (
            contacts.rigid_contact_shape0.numpy()[:n],
            contacts.rigid_contact_shape1.numpy()[:n],
            contacts.rigid_contact_point0.numpy()[:n],
            contacts.rigid_contact_point1.numpy()[:n],
            contacts.rigid_contact_normal.numpy()[:n],
        )
    )
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


class TestCollisionOutputCapacity(unittest.TestCase):
    def test_generic_convex_output_matches_full_capacity(self):
        """Retain complete GJK/manifold results after bounding the broad-phase output."""
        model, pairs = _model_and_pairs(boxes=True)
        baseline = _pipeline(model, pairs, deterministic=True)
        bounded = _pipeline(model, pairs, broad_phase_output_max=2, deterministic=True)
        expected, actual = baseline.contacts(), bounded.contacts()
        baseline.collide(model.state(), expected)
        bounded.collide(model.state(), actual)
        self.assertEqual(int(bounded.broad_phase_pair_count.numpy()[0]), 2)
        self.assertEqual(int(bounded.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), 2)
        self.assertGreater(int(actual.rigid_contact_count.numpy()[0]), 0)
        np.testing.assert_allclose(_contact_records(actual), _contact_records(expected), atol=1.0e-6, rtol=1.0e-6)

    def test_cuda_split_buffers_follow_materialized_capacity(self):
        """Size CUDA split query/work storage from output capacity rather than the input list."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("split GJK/MPR buffers require CUDA")
        model, _ = _model_and_pairs(boxes=True, device=devices[0])
        pairs = wp.array(np.tile([0, 4], (30000, 1)), dtype=wp.vec2i, device=devices[0])
        pipeline = _pipeline(model, pairs, broad_phase_output_max=28000)
        self.assertEqual(len(pipeline.shape_pairs_filtered), 30000)
        self.assertTrue(pipeline.narrow_phase.split_gjk_mpr)
        self.assertEqual(len(pipeline.narrow_phase.split_query_results), 28000)
        self.assertEqual(len(pipeline.narrow_phase.split_gjk_work_items), 28000)
        self.assertEqual(len(pipeline.narrow_phase.split_manifold_work_items), 28000)

    def test_complete_input_traversal_with_small_output(self):
        """Find late overlapping pairs while allocating only their output budget."""
        model, pairs = _model_and_pairs()
        original_pairs = pairs.numpy().copy()
        baseline = _pipeline(model, pairs)
        bounded = _pipeline(model, pairs, broad_phase_output_max=2)
        self.assertIs(bounded.shape_pairs_filtered, pairs)
        self.assertEqual(len(bounded.shape_pairs_filtered), 15)
        self.assertEqual(bounded.shape_pairs_max, 2)
        self.assertEqual(len(bounded.broad_phase_shape_pairs), 2)
        self.assertEqual(len(bounded.narrow_phase.gjk_candidate_pairs), 2)
        state = model.state()
        expected, actual = baseline.contacts(), bounded.contacts()
        baseline.collide(state, expected)
        bounded.collide(state, actual)
        self.assertEqual(int(bounded.broad_phase_pair_count.numpy()[0]), 2)
        self.assertEqual(int(actual.rigid_contact_count.numpy()[0]), 2)
        self.assertEqual(set(map(tuple, bounded.broad_phase_shape_pairs.numpy())), {(0, 4), (1, 5)})
        np.testing.assert_allclose(_contact_records(actual), _contact_records(expected), atol=1.0e-6, rtol=1.0e-6)
        np.testing.assert_array_equal(pairs.numpy(), original_pairs)

    def test_raw_overflow_remains_visible_and_warns(self):
        """Retain unclamped demand and the existing warning after bounded writes."""
        model, pairs = _model_and_pairs(all_overlap=True)
        pipeline = _pipeline(model, pairs, broad_phase_output_max=2, deterministic=True)
        contacts = pipeline.contacts()
        captured = StdOutCapture()
        captured.begin()
        try:
            pipeline.collide(model.state(), contacts)
            count = int(pipeline.broad_phase_pair_count.numpy()[0])
        finally:
            output = captured.end()
        self.assertEqual(count, 15)
        self.assertEqual(len(pipeline.broad_phase_shape_pairs), 2)
        self.assertIn("Broad phase pair buffer overflowed 15 > 2", output)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
        self.assertTrue(np.isfinite(_contact_records(contacts)).all())
        # This is intentionally a current-frame counter, not a sticky status.
        state = model.state()
        body_q = state.body_q.numpy()
        body_q[:, 0] = np.arange(6, dtype=np.float32) * 10.0
        state.body_q.assign(body_q)
        pipeline.collide(state, contacts)
        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)

    def test_default_and_legacy_override_unchanged(self):
        """Keep full-list default capacity and ignore legacy shape_pairs_max in explicit mode."""
        model, pairs = _model_and_pairs()
        for kwargs in ({}, {"broad_phase_output_max": None}, {"shape_pairs_max": 1}):
            with self.subTest(kwargs=kwargs):
                pipeline = _pipeline(model, pairs, **kwargs)
                self.assertEqual(pipeline.shape_pairs_max, len(pairs))
                self.assertEqual(pipeline.narrow_phase.max_candidate_pairs, len(pairs))

    def test_reject_unsupported_modes_and_noninteger_values(self):
        """Reject ambiguous overrides rather than silently ignoring new capacity requests."""
        model, pairs = _model_and_pairs()
        for value in (0, -1, True, False, 1.5, float("nan"), float("inf"), "2"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "broad_phase_output_max"):
                _pipeline(model, pairs, broad_phase_output_max=value)
        for mode in ("nxn", "sap"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "explicit"):
                _pipeline(model, pairs, broad_phase=mode, broad_phase_output_max=2)
        original = _pipeline(model, pairs)
        with self.assertRaisesRegex(ValueError, "internal"):
            _pipeline(
                model,
                pairs,
                broad_phase=original.broad_phase,
                narrow_phase=original.narrow_phase,
                broad_phase_output_max=2,
            )

    def test_bound_clamps_to_input_length_and_preserves_empty_input(self):
        """Avoid allocating beyond the complete explicit input bound."""
        model, pairs = _model_and_pairs()
        oversized = _pipeline(model, pairs, broad_phase_output_max=100)
        self.assertEqual(oversized.shape_pairs_max, 15)
        integer = _pipeline(model, pairs, broad_phase_output_max=np.int32(2))
        self.assertEqual(integer.shape_pairs_max, 2)
        empty = wp.empty(0, dtype=wp.vec2i, device="cpu")
        pipeline = _pipeline(model, empty, broad_phase_output_max=10)
        self.assertEqual(pipeline.shape_pairs_max, 0)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

    def test_narrow_phase_derived_budgets_and_deterministic_contacts(self):
        """Pass the materialized budget downstream without shrinking contact sorting capacity."""
        model, pairs = _model_and_pairs()
        original_factory = collide_module.NarrowPhase
        with mock.patch.object(collide_module, "NarrowPhase", wraps=original_factory) as factory:
            bounded = _pipeline(model, pairs, broad_phase_output_max=2, deterministic=True)
        kwargs = factory.call_args.kwargs
        for key in (
            "max_candidate_pairs",
            "candidate_pair_work_estimate",
            "max_mesh_mesh_pairs",
            "max_mesh_plane_pairs",
        ):
            self.assertEqual(kwargs[key], 2)
        self.assertEqual(kwargs["contact_max"], 32)
        self.assertEqual(len(bounded._sort_key_array), 32)
        baseline = _pipeline(model, pairs, deterministic=True)
        contacts = bounded.contacts()
        expected = baseline.contacts()
        baseline.collide(model.state(), expected)
        for _ in range(3):
            bounded.collide(model.state(), contacts)
            np.testing.assert_allclose(_contact_records(contacts), _contact_records(expected), atol=1.0e-6, rtol=1.0e-6)


if __name__ == "__main__":
    unittest.main()

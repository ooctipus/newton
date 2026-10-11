# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check automatic collision choice without task-specific dispatch."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.coherent_convex import _ConvexQueryCache
from newton._src.geometry.coherent_convex_rejection import _create_rejection_query_kernels
from newton._src.geometry.convex_query_cost import _ConvexMethodChoice, _nonnegative_fit, _QueryCostModel
from newton.tests.test_coherent_convex_rejection import _accepted, _box_inputs, _buffers, _original, _selector_model
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_graph_choice_preserves_queries(test, device):
    """Change methods inside a captured graph as geometry changes; preserve cold outputs."""
    n, threads = 512, 128
    pairs, pair_array, values, inputs = _box_inputs(n, device)
    count = wp.array([n], dtype=int, device=device)
    baseline, candidate = _buffers(n, device), _buffers(n, device)
    original, kernels = _original(), _create_rejection_query_kernels()
    cache = _ConvexQueryCache(
        pairs=pairs,
        shape_types=values[0].numpy(),
        shape_world=np.zeros(2 * n, np.int32),
        world_count=1,
        query_capacity=n,
        device=device,
    )
    # Fixed independent-machine costs exercise both decisions reproducibly.
    model = _QueryCostModel(0, 100, 0, 1, 0, 1, 10, 0, ())
    with patch("newton._src.geometry.convex_query_cost._calibrate_query_cost", return_value=model):
        owner = _ConvexMethodChoice(device=device, slots=n)
    owner.period = 2

    def ordinary(buffers, query_count=None):
        wp.launch(
            original[0],
            threads,
            [pair_array, count if query_count is None else query_count, *values, threads, *buffers],
            device=device,
            block_dim=128,
        )
        wp.launch(original[1], threads, [pair_array, *values, threads, *buffers], device=device, block_dim=128)

    def rejection(query_count, gjk_count):
        cache.begin()
        rejection_buffers = [candidate[0], candidate[1], gjk_count, candidate[3], candidate[4]]
        for kernel in kernels:
            wp.launch(
                kernel,
                threads,
                [pair_array, query_count, inputs, cache.data, threads, *rejection_buffers],
                device=device,
                block_dim=128,
            )

    def run():
        for buffer in candidate:
            buffer.zero_()
        event = wp.get_stream(device).record_event()
        owner.launch(pair_array, count, inputs, lambda query_count: ordinary(candidate, query_count), rejection)
        wp.get_stream(device).wait_event(event)

    poses = values[2].numpy()
    run()  # Compile both branch factories before capture.
    with wp.ScopedCapture(device=device) as capture:
        run()
    for distance, expected, reset in ((1.2, 1, False), (0.8, 0, False), (1.2, 1, True), (1.002, 0, True)):
        with test.subTest(distance=distance, reset=reset):
            poses[1::2, 0] = distance
            values[2].assign(poses)
            if reset:
                cache.reset(wp.array([True, False], dtype=wp.bool, device=device))
            for buffer in baseline:
                buffer.zero_()
            ordinary(baseline)
            for _ in range(4):
                wp.capture_launch(capture.graph)
            test.assertEqual(int(owner.choice.numpy()[0]), expected)
            ids, results = _accepted(candidate)
            reference_ids, reference = _accepted(baseline)
            np.testing.assert_array_equal(np.sort(ids), np.sort(reference_ids))
            test.assertEqual(results[np.sort(ids)].tobytes(), reference[np.sort(ids)].tobytes())
            np.testing.assert_array_equal(values[2].numpy(), poses)
    count.zero_()
    for _ in range(2):
        wp.capture_launch(capture.graph)
    test.assertEqual(int(owner.choice.numpy()[0]), 0)
    test.assertEqual(len(_accepted(candidate)[0]), 0)


def test_dense_pipeline_uses_ordinary(test, device):
    """An admitted dense scene uses the real calibrated model and ordinary contacts."""
    model = _selector_model(device)
    pipelines = []
    for mode in ("ordinary", "auto", "rejection"):
        with patch.dict(
            os.environ,
            {
                "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0",
                "NEWTON_NARROW_PHASE_COHERENT_STATS": "0",
                "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": "0",
            },
        ):
            pipelines.append(
                newton.CollisionPipeline(model, broad_phase="explicit", rigid_contact_max=300000, convex_collision=mode)
            )
    state = model.state()
    counts = []
    for pipeline in pipelines:
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(state, contacts)
        wp.capture_launch(capture.graph)
        counts.append(int(contacts.rigid_contact_count.numpy()[0]))
    test.assertEqual(counts[0], counts[1])
    test.assertEqual(counts[0], counts[2])
    test.assertEqual(pipelines[2].get_collision_selection()["method"], "rejection")
    test.assertGreater(counts[0], 0)
    owner = pipelines[1].narrow_phase._convex_method_choice
    test.assertIsNotNone(owner)
    test.assertEqual(owner.describe()["method"], "ordinary")
    test.assertTrue(np.isfinite(owner.coefficients.numpy()).all())


class TestAutoConvexSelection(unittest.TestCase):
    def test_cell_cost_ignores_shared_output_overhead(self):
        """Preserve the predicted method difference when both primitives share fixed work."""
        from newton._src.geometry.convex_cell_cost import _fit_support_cost  # noqa: PLC0415

        records = []
        for calls in (1, 8, 32):
            for vertices in (8, 32, 64):
                candidates = calls * (1 + vertices / 16)
                records.append(
                    {
                        "calls": calls,
                        "vertices": vertices,
                        "candidates": candidates,
                        "scan_ns": 10 + calls + 2 * calls * vertices,
                        "cell_ns": 12 + 3 * calls + candidates,
                    }
                )
        costs, _ = _fit_support_cost(records)
        for row in records:
            row["scan_ns"] += 100
            row["cell_ns"] += 100
        shifted, _ = _fit_support_cost(records)
        np.testing.assert_allclose(costs, [1, 2, 3, 1, 2], atol=1e-10)
        np.testing.assert_allclose(shifted, costs, atol=1e-10)

    def test_public_collision_choice_overrides_environment(self):
        """Keep explicit instance choices independent of legacy process settings."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        with patch.dict(
            os.environ, {"NEWTON_NARROW_PHASE_COHERENT_CONVEX": "reject_only", "NEWTON_NARROW_PHASE_CONVEX_CELLS": "1"}
        ):
            ordinary = newton.CollisionPipeline(model, convex_collision="ordinary")
            automatic = newton.CollisionPipeline(model, convex_collision="auto")
        self.assertEqual(ordinary.get_collision_selection()["method"], "ordinary")
        self.assertEqual(automatic.get_collision_selection()["requested"], "auto")
        self.assertEqual(automatic.get_collision_selection()["reason"], "unsupported_pipeline")
        for choice in ("rejection", "convex_cells", "typo"):
            with self.subTest(choice=choice), self.assertRaisesRegex(ValueError, "convex_collision"):
                newton.CollisionPipeline(model, convex_collision=choice)

    def test_nonnegative_cost_fit(self):
        # Constraining a negative slope requires refitting the intercept.
        result = _nonnegative_fit([[1, 1], [1, 2], [1, 3]], [3, 2, 1])
        np.testing.assert_allclose(result, [2, 0])

    def test_unsupported_auto_preserves_collision(self):
        """Fall back to ordinary collision on a CPU pipeline without losing contacts."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        with patch.dict(os.environ, {"NEWTON_NARROW_PHASE_COHERENT_CONVEX": "auto"}):
            pipeline = newton.CollisionPipeline(model)
        self.assertIsNone(pipeline.narrow_phase._coherent_cache)
        self.assertEqual(pipeline._convex_selection_reason, "unsupported_pipeline")
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


for device in get_test_devices():
    if wp.get_device(device).is_cuda:
        add_function_test(
            TestAutoConvexSelection,
            "test_graph_choice_preserves_queries",
            test_graph_choice_preserves_queries,
            devices=[device],
        )
        add_function_test(
            TestAutoConvexSelection,
            "test_dense_pipeline_uses_ordinary",
            test_dense_pipeline_uses_ordinary,
            devices=[device],
        )

if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check rejection-only ownership and original retained-query semantics."""

import ast
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.coherent_convex import _ConvexQueryCache
from newton._src.geometry.coherent_convex_rejection import _create_rejection_query_kernels, _PairInputs
from newton._src.geometry.collision_convex import ConvexQueryResult
from newton._src.geometry.collision_core import post_process_minkowski_only
from newton._src.geometry.narrow_phase import create_narrow_phase_kernels_gjk_mpr_split, write_contact_simple
from newton._src.geometry.support_function import support_map_lean
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_test_devices


def _original():
    return create_narrow_phase_kernels_gjk_mpr_split(
        True, write_contact_simple, support_map_lean, post_process_minkowski_only, False
    )


def _buffers(n, device):
    return [
        wp.zeros(n, dtype=ConvexQueryResult, device=device),
        wp.zeros(n, dtype=int, device=device),
        wp.zeros(1, dtype=int, device=device),
        wp.zeros(n, dtype=int, device=device),
        wp.zeros(1, dtype=int, device=device),
    ]


def _box_inputs(n, device):
    pairs = np.arange(2 * n, dtype=np.int32).reshape(n, 2)
    poses = np.zeros((2 * n, 7), np.float32)
    poses[:, 6] = 1
    poses[1::2, 0] = np.resize(np.array([0.8, 1.002, 1.2], np.float32), n)
    values = [
        wp.array(np.full(2 * n, int(newton.GeoType.BOX)), dtype=int, device=device),
        wp.array(np.tile([0.5, 0.5, 0.5, 0.0], (2 * n, 1)), dtype=wp.vec4, device=device),
        wp.array(poses, dtype=wp.transform, device=device),
        wp.zeros(2 * n, dtype=wp.uint64, device=device),
        wp.full(2 * n, 0.005, dtype=float, device=device),
        wp.full(2 * n, 2.0, dtype=float, device=device),
        wp.full(2 * n, wp.vec3(-3.0), dtype=wp.vec3, device=device),
        wp.full(2 * n, wp.vec3(3.0), dtype=wp.vec3, device=device),
        wp.full(2 * n, wp.vec3(-0.5), dtype=wp.vec3, device=device),
        wp.full(2 * n, wp.vec3(0.5), dtype=wp.vec3, device=device),
    ]
    data = _PairInputs()
    for name, value in zip(data._cls.__annotations__, values, strict=True):
        setattr(data, name, value)
    return pairs, wp.array(pairs, dtype=wp.vec2i, device=device), values, data


def _accepted(buffers):
    count = int(buffers[4].numpy()[0])
    ids = buffers[3].numpy()[:count]
    assert 0 <= count <= buffers[3].shape[0] and len(np.unique(ids)) == count
    return ids, buffers[0].numpy()


def test_rejection_queries(test, device):
    """Preserve all cold outputs across current hints, stale hints and partial/all resets."""
    n, threads = 131, 128
    pairs, pair_array, values, inputs = _box_inputs(n, device)
    count = wp.array([n], dtype=int, device=device)
    original_buffers, candidate_buffers = _buffers(n, device), _buffers(n, device)
    original = _original()
    wp.launch(
        original[0],
        dim=threads,
        inputs=[pair_array, count, *values, threads, *original_buffers],
        device=device,
        block_dim=128,
    )
    wp.launch(
        original[1], dim=threads, inputs=[pair_array, *values, threads, *original_buffers], device=device, block_dim=128
    )
    original_ids, original_results = _accepted(original_buffers)
    test.assertEqual(len(original_ids), 88)
    cache = _ConvexQueryCache(
        pairs=pairs,
        shape_types=values[0].numpy(),
        shape_world=np.zeros(2 * n, np.int32),
        world_count=1,
        query_capacity=n,
        device=device,
    )
    kernels = _create_rejection_query_kernels(diagnostics=True)
    for case in (
        "cold",
        "current",
        "partial_reset",
        "all_reset",
        "source_changed",
        "epoch_changed",
        "retained_hint",
        "empty",
    ):
        if case == "partial_reset":
            cache.reset(wp.array([True, False], dtype=wp.bool, device=device))
        elif case == "all_reset":
            cache.reset()
        elif case == "source_changed":
            cache.data.source_a.fill_(123)
        elif case == "epoch_changed":
            cache.data.entry_epoch.fill_(123)
        elif case == "retained_hint":
            cache.data.mode.fill_(2)
        elif case == "empty":
            count.zero_()
        for buffer in candidate_buffers:
            buffer.zero_()
        cache.begin()
        for kernel in kernels:
            wp.launch(
                kernel,
                dim=threads,
                inputs=[pair_array, count, inputs, cache.data, threads, *candidate_buffers],
                device=device,
                block_dim=128,
            )
        ids, results = _accepted(candidate_buffers)
        with test.subTest(case=case):
            if case == "empty":
                test.assertEqual(len(ids), 0)
                test.assertEqual(int(cache.data.unresolved_work_count.numpy()[0]), 0)
                continue
            np.testing.assert_array_equal(np.sort(ids), np.sort(original_ids))
            # Controlled axis-aligned fixture is exact; broad live geometry has
            # a separate physical-equivalence gate for compiler ULP changes.
            test.assertEqual(results[np.sort(ids)].tobytes(), original_results[np.sort(ids)].tobytes())
            stats = cache.data.stats.numpy()
            np.testing.assert_array_equal(stats[3:6], [0, 0, 0])
            test.assertFalse(hasattr(cache.data, "warm_work_count"))
            done = cache.data.query_done.numpy()
            test.assertFalse(np.any(done == 2))
            unresolved_count = int(cache.data.unresolved_work_count.numpy()[0])
            unresolved = cache.data.unresolved_work_items.numpy()[:unresolved_count]
            test.assertEqual(len(np.unique(unresolved)), unresolved_count)
            np.testing.assert_array_equal(np.sort(unresolved), np.flatnonzero(done == 0))
            test.assertFalse(np.any(done[ids] != 0))
            if case in ("current", "retained_hint"):
                test.assertEqual(int(stats[2]), 43)
                test.assertEqual(unresolved_count, 88)
            else:
                test.assertEqual(int(stats[2]), 0)
                test.assertEqual(unresolved_count, n)


def test_current_motion_and_query_graph(test, device):
    """Recheck moved current shapes through eager and two actual graph replays."""
    n, threads = 131, 128
    pairs, pair_array, values, inputs = _box_inputs(n, device)
    original_poses = values[2].numpy().copy()
    count = wp.array([n], dtype=int, device=device)
    baseline, candidate = _buffers(n, device), _buffers(n, device)
    original = _original()
    kernels = _create_rejection_query_kernels(diagnostics=True)
    cache = _ConvexQueryCache(
        pairs=pairs,
        shape_types=values[0].numpy(),
        shape_world=np.zeros(2 * n, np.int32),
        world_count=1,
        query_capacity=n,
        device=device,
    )

    def run_original():
        for buffer in baseline:
            buffer.zero_()
        wp.launch(original[0], dim=threads, inputs=[pair_array, count, *values, threads, *baseline], device=device)
        wp.launch(original[1], dim=threads, inputs=[pair_array, *values, threads, *baseline], device=device)

    def run_candidate():
        for buffer in candidate:
            buffer.zero_()
        cache.begin()
        for kernel in kernels:
            wp.launch(
                kernel,
                dim=threads,
                inputs=[pair_array, count, inputs, cache.data, threads, *candidate],
                device=device,
                block_dim=128,
            )

    def compare():
        expected_ids, expected = _accepted(baseline)
        actual_ids, actual = _accepted(candidate)
        np.testing.assert_array_equal(np.sort(actual_ids), np.sort(expected_ids))
        test.assertEqual(actual[np.sort(actual_ids)].tobytes(), expected[np.sort(expected_ids)].tobytes())

    run_original()
    run_candidate()
    compare()
    graph = None
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            run_candidate()
        graph = capture.graph
    for moved in (True, False):
        poses = original_poses.copy()
        if moved:
            # A previously certified separated pair becomes penetrating;
            # retaining the cache without a reset must not lose its contact.
            poses[5, 0] = 0.8
        values[2].assign(poses)
        run_original()
        for _ in range(2):
            if graph is None:
                run_candidate()
            else:
                wp.capture_launch(graph)
            compare()
        test.assertEqual(len(_accepted(candidate)[0]), 89 if moved else 88)
    cache.reset(wp.array([False, True], dtype=wp.bool, device=device))
    if graph is None:
        run_candidate()
    else:
        wp.capture_launch(graph)
    compare()
    test.assertEqual(int(cache.data.stats.numpy()[2]), 0)


def _selector_model(device):
    """Build enough genuine pairs to admit the unmodified automatic CUDA selector."""
    world = newton.ModelBuilder()
    for _ in range(167):
        body = world.add_body()
        world.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    builder.add_world(world)
    builder.add_world(world, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0)))
    return builder.finalize(device=device)


def test_pipeline_rejection_selector(test, device):
    """Keep CUDA admission real and select exactly three kernels with public resets."""
    device = wp.get_device(device)
    if device.is_cpu:
        test.skipTest("Actual selector admission requires the lean CUDA split pipeline")
    model = _selector_model(device)
    for mode, kernel_count in (("0", 0), ("reject_only", 3)):
        with patch.dict(
            os.environ,
            {
                "NEWTON_NARROW_PHASE_COHERENT_CONVEX": mode,
                "NEWTON_NARROW_PHASE_COHERENT_STATS": "0",
                "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": "0",
            },
        ):
            pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
        test.assertTrue(pipeline.narrow_phase.split_gjk_mpr)
        cache = pipeline.narrow_phase._coherent_cache
        if mode == "0":
            test.assertIsNone(cache)
            test.assertIsNone(pipeline.narrow_phase._coherent_query_kernels)
        else:
            test.assertEqual(len(pipeline.narrow_phase._coherent_query_kernels), kernel_count)
            cache.data.mode.fill_(1)
            pipeline.reset_contact_matching(wp.array([True, False, False], dtype=wp.bool, device=device))
            np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [1, 0, 0])
            pipeline.reset_contact_matching(wp.array([False, False, True], dtype=wp.bool, device=device))
            np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [2, 1, 1])
            pipeline.reset_contact_matching()
            test.assertFalse(np.any(cache.data.mode.numpy()))


def test_pipeline_rejection_capacity_status(test, device):
    """Retain raw broad overflow and sticky history with an actually enabled cache."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("Actual selector admission requires the lean CUDA split pipeline")
    model = _selector_model(device)
    with patch.dict(
        os.environ,
        {
            "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "reject_only",
            "NEWTON_NARROW_PHASE_COHERENT_STATS": "0",
            "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": "0",
        },
    ):
        # Preserve the original automatic split-admission threshold while
        # undersizing the genuinely overlapping 28,056-pair input domain.
        pipeline = newton.CollisionPipeline(
            model, broad_phase_output_max=27_776, rigid_contact_max=16, verify_buffers=True
        )
    narrow, cache = pipeline.narrow_phase, pipeline.narrow_phase._coherent_cache
    test.assertIsNotNone(cache)
    test.assertEqual(cache.data.unresolved_work_items.shape[0], narrow.split_query_results.shape[0])
    test.assertEqual(cache.data.query_slot.shape[0], narrow.split_query_results.shape[0])
    contacts, state = pipeline.contacts(), model.state()
    messages = StdOutCapture()
    messages.begin()
    try:
        pipeline.collide(state, contacts)
        test.assertGreater(int(pipeline.broad_phase_pair_count.numpy()[0]), pipeline.shape_pairs_max)
        test.assertTrue(narrow.buffer_capacity_status()["broad_phase"])
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(state, contacts)
        narrow.buffer_capacity_status(clear=True)
        wp.capture_launch(capture.graph)
        test.assertTrue(narrow.buffer_capacity_status()["broad_phase"])
        # Keep the same captured graph and allocations, but remove all overlaps.
        poses = state.body_q.numpy()
        poses[:, 0] = 10.0 * (np.arange(len(poses)) + 1)
        state.body_q.assign(poses)
        wp.capture_launch(capture.graph)
        test.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
        test.assertTrue(narrow.buffer_capacity_status(clear=True)["broad_phase"])
        wp.capture_launch(capture.graph)
        narrow.check_buffer_capacity()
    finally:
        messages.end()


class TestCoherentConvexRejection(unittest.TestCase):
    def test_original_core_and_no_warm_factory(self):
        """Keep original MPR/GJK closures and original default cutoff/iteration arguments."""
        original, candidate = _original(), _create_rejection_query_kernels()
        self.assertEqual(
            [kernel.func.__name__ for kernel in candidate], ["classify_rejections", "cold_mpr", "cold_gjk"]
        )
        for left, right, name in ((original[0], candidate[1], "solve_mpr"), (original[1], candidate[2], "solve_gjk")):
            baseline = inspect.getclosurevars(left.func).nonlocals[name].core
            actual = inspect.getclosurevars(right.func).nonlocals[name]
            self.assertEqual(inspect.getsource(baseline.func), inspect.getsource(actual.func))
            self.assertEqual(inspect.signature(baseline.func), inspect.signature(actual.func))
            for key, value in inspect.getclosurevars(baseline.func).nonlocals.items():
                other = inspect.getclosurevars(actual.func).nonlocals[key]
                if hasattr(value, "func"):
                    self.assertEqual(inspect.getsource(value.func), inspect.getsource(other.func))
                elif value is None:
                    self.assertIsNone(other)
        source = inspect.getsource(_create_rejection_query_kernels)
        nodes = ast.parse(source)
        self.assertNotIn("_create_cached_gjk", source)
        self.assertNotIn("_CacheProvider", source)
        self.assertNotIn("warm_work", source)
        self.assertNotIn("feature_a", source)
        calls = [
            node
            for node in ast.walk(nodes)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("solve_mpr", "solve_gjk")
        ]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(len(call.args) == 6 and not call.keywords for call in calls))

    def test_cpu_selector_fails_closed(self):
        """Recognize all modes without admitting unsupported CPU collision execution."""
        builder = newton.ModelBuilder()
        builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        for mode in ("0", "1", "reject_only", "invalid"):
            with (
                self.subTest(mode=mode),
                patch.dict(
                    os.environ,
                    {
                        "NEWTON_NARROW_PHASE_COHERENT_CONVEX": mode,
                        "NEWTON_NARROW_PHASE_COHERENT_STATS": "0",
                        "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": "0",
                    },
                ),
            ):
                if mode == "0":
                    self.assertIsNone(newton.CollisionPipeline(model).narrow_phase._coherent_cache)
                else:
                    message = "experimental coherent" if mode == "reject_only" else "must be 0 or reject_only"
                    with self.assertRaisesRegex(ValueError, message):
                        newton.CollisionPipeline(model)


for _test in (
    test_rejection_queries,
    test_current_motion_and_query_graph,
    test_pipeline_rejection_selector,
    test_pipeline_rejection_capacity_status,
):
    add_function_test(TestCoherentConvexRejection, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()

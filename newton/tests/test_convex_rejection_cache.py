# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete-shape bounds used by the experimental coherent convex path."""

import gc
import itertools
import os
import types
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.coherent_convex import (
    _ConvexQueryCache,
    _plane_lower_bound,
    _support_feature,
)
from newton._src.geometry.support_function import GenericShapeData, pack_mesh_ptr
from newton._src.geometry.types import GeoType
from newton._src.sim.collide import (
    _SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD,
    _compute_generic_convex_pair_work_estimate,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _bound_kernel(
    types: wp.array2d[int],
    scales_a: wp.array[wp.vec3],
    scales_b: wp.array[wp.vec3],
    mesh_a: wp.uint64,
    mesh_b: wp.uint64,
    positions: wp.array[wp.vec3],
    orientations: wp.array[wp.quat],
    directions: wp.array[wp.vec3],
    bounds: wp.array[float],
    valid: wp.array[int],
    features: wp.array[int],
    feature_points: wp.array[wp.vec3],
):
    i = wp.tid()
    a = GenericShapeData()
    a.shape_type = types[i, 0]
    a.scale = scales_a[i]
    a.auxiliary = pack_mesh_ptr(mesh_a)
    b = GenericShapeData()
    b.shape_type = types[i, 1]
    b.scale = scales_b[i]
    b.auxiliary = pack_mesh_ptr(mesh_b)
    bound = _plane_lower_bound(a, b, orientations[i], positions[i], directions[i])
    bounds[i] = bound.lower
    valid[i] = bound.valid
    feature = _support_feature(a, directions[i])
    features[i] = feature.index
    feature_points[i] = feature.point


def _rotate(points, q):
    vector, scalar = q[:3], q[3]
    return (
        2 * (points @ vector)[:, None] * vector
        + (2 * scalar * scalar - 1) * points
        + 2 * scalar * np.cross(vector, points)
    )


def _launch(device, points_a, points_b, types, scales_a, scales_b, positions, orientations, directions):
    meshes = [
        wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device=device), indices=wp.array([0, 1, 2], dtype=int, device=device)
        )
        for points in (points_a, points_b)
    ]
    n = len(directions)
    inputs = [wp.array(types, dtype=int, device=device)]
    inputs += [wp.array(x, dtype=wp.vec3, device=device) for x in (scales_a, scales_b)]
    inputs += [meshes[0].id, meshes[1].id]
    inputs += [
        wp.array(positions, dtype=wp.vec3, device=device),
        wp.array(orientations, dtype=wp.quat, device=device),
        wp.array(directions, dtype=wp.vec3, device=device),
    ]
    outputs = [
        wp.zeros(n, dtype=float, device=device),
        wp.zeros(n, dtype=int, device=device),
        wp.zeros(n, dtype=int, device=device),
        wp.zeros(n, dtype=wp.vec3, device=device),
    ]
    wp.launch(_bound_kernel, dim=n, inputs=inputs, outputs=outputs, device=device)
    return [x.numpy() for x in outputs]


def test_random_full_shape_bound(test, device):
    """Compare native conservative bounds to every full-shape vertex in float64."""
    rng = np.random.default_rng(9731)
    n = 4096
    points_a = rng.uniform(-1, 1, (64, 3)).astype(np.float32)
    points_b = rng.uniform(-1, 1, (40, 3)).astype(np.float32)
    box = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    types = rng.choice([int(GeoType.BOX), int(GeoType.CONVEX_MESH)], (n, 2)).astype(np.int32)
    magnitudes = 10.0 ** rng.uniform(-6, 6, (n, 1))
    scales_a = (magnitudes * rng.uniform(0.01, 2, (n, 3))).astype(np.float32)
    scales_b = (magnitudes * rng.uniform(0.01, 2, (n, 3))).astype(np.float32)
    positions = (magnitudes * rng.uniform(-4, 4, (n, 3))).astype(np.float32)
    orientations = rng.normal(size=(n, 4))
    orientations /= np.linalg.norm(orientations, axis=1)[:, None]
    orientations *= rng.uniform(0.9, 1.1, (n, 1))
    orientations = orientations.astype(np.float32)
    directions = (rng.normal(size=(n, 3)) * 10.0 ** rng.uniform(-6, 6, (n, 1))).astype(np.float32)
    lower, valid, features, feature_points = _launch(
        device, points_a, points_b, types, scales_a, scales_b, positions, orientations, directions
    )
    np.testing.assert_array_equal(valid, np.ones(n, dtype=np.int32))
    for i in range(n):
        a = (box if types[i, 0] == int(GeoType.BOX) else points_a.astype(float)) * scales_a[i].astype(float)
        b = (box if types[i, 1] == int(GeoType.BOX) else points_b.astype(float)) * scales_b[i].astype(float)
        b = _rotate(b, orientations[i].astype(float)) + positions[i].astype(float)
        d = directions[i].astype(float)
        exact = (np.min(b @ d) - np.max(a @ d)) / np.linalg.norm(d)
        test.assertLessEqual(float(lower[i]), exact)
        if types[i, 0] == int(GeoType.CONVEX_MESH):
            np.testing.assert_array_equal(feature_points[i], (points_a[features[i]] * scales_a[i]).astype(np.float32))
        else:
            np.testing.assert_array_equal(np.abs(feature_points[i]), scales_a[i])


def test_invalid_and_shell_bound(test, device):
    """Refuse invalid inputs and never reject contact-shell equality."""
    n = 8
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    types = np.full((n, 2), int(GeoType.BOX), np.int32)
    scales_a = np.full((n, 3), 0.5, np.float32)
    scales_b = scales_a.copy()
    positions = np.zeros((n, 3), np.float32)
    positions[:, 0] = 2
    orientations = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
    directions = np.tile(np.array([1, 0, 0], np.float32), (n, 1))
    directions[1] = 0
    directions[2, 0] = np.nan
    directions[3, 0] = np.inf
    orientations[4, 0] = np.nan
    scales_a[5, 0] = -1
    positions[6, 0] = np.inf
    types[7, 0] = int(GeoType.SPHERE)
    lower, valid, _, _ = _launch(
        device, vertices, vertices, types, scales_a, scales_b, positions, orientations, directions
    )
    np.testing.assert_array_equal(valid, np.array([1, 0, 0, 0, 0, 0, 0, 0], np.int32))
    test.assertLess(float(lower[0]), 1.0)
    test.assertGreater(float(lower[0]), 0.99)


class TestConvexRejectionCache(unittest.TestCase):
    def test_cache_metadata_admission(self):
        """Refuse ambiguous ownership and malformed static cache metadata."""
        metadata = {
            "pairs": np.array([[0, 1], [1, 2]], dtype=np.int32),
            "shape_types": np.full(3, int(GeoType.BOX), dtype=np.int32),
            "shape_world": np.array([-1, 0, 0], dtype=np.int32),
            "world_count": 1,
            "query_capacity": 2,
            "device": "cpu",
        }
        for change in (
            {"pairs": np.array([[0, 1], [0, 1]])},
            {"pairs": np.array([[1, 0]])},
            {"pairs": np.array([[0, 0]])},
            {"pairs": np.array([[0, 3]])},
            {"pairs": np.array([[0.0, 1.0]])},
            {"shape_world": np.array([0, 0])},
            {"shape_world": np.array([-2, 0, 0])},
            {"shape_world": np.array([-1, 0, 1]), "world_count": 2},
            {"query_capacity": -1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                _ConvexQueryCache(**(metadata | change))
        cache = _ConvexQueryCache(**metadata)
        np.testing.assert_array_equal(cache.data.world.numpy(), [0, 0])
        np.testing.assert_array_equal(cache.data.keys.numpy(), [1, (1 << 32) | 2])

    def test_graph_owned_cache_lease(self):
        """Keep one capture lineage and release it before another graph owns the cache."""
        cache = _ConvexQueryCache.__new__(_ConvexQueryCache)
        cache._graph_token = None
        first, second = types.SimpleNamespace(), types.SimpleNamespace()
        cache._acquire_graph(None)
        cache._acquire_graph(first)
        cache._acquire_graph(first)
        self.assertEqual(len(first._newton_coherent_convex_leases), 1)
        self.assertIs(first._newton_coherent_convex_leases[0].owner, cache)
        with self.assertRaisesRegex(RuntimeError, "two live CUDA graphs"):
            cache._acquire_graph(second)
        first._newton_coherent_convex_leases[0].release()
        self.assertIsNone(cache._graph_token)
        cache._acquire_graph(second)
        self.assertIs(second._newton_coherent_convex_leases[0].owner, cache)


def test_world_epoch_reset(test, device):
    """Reset only selected worlds without reading beyond the public reset mask."""
    cache = _ConvexQueryCache(
        pairs=np.array([[0, 1], [0, 2], [0, 3]]),
        shape_types=np.full(4, int(GeoType.BOX)),
        shape_world=np.array([-1, -1, 0, 1]),
        world_count=2,
        query_capacity=3,
        device=device,
    )
    np.testing.assert_array_equal(cache.data.world.numpy(), [2, 0, 1])
    cache.data.mode.fill_(2)
    cache.reset(wp.array([False, True, False], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [0, 1, 0])
    np.testing.assert_array_equal(cache.data.mode.numpy(), [2, 2, 2])
    epochs_before = cache.data.world_epoch.numpy()[cache.data.world.numpy()]
    cache.reset(wp.array([False, False, True], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [1, 2, 1])
    test.assertTrue(np.all(cache.data.world_epoch.numpy()[cache.data.world.numpy()] != epochs_before))
    for mask in (
        wp.array([True], dtype=wp.bool, device=device),
        wp.array([False, True], dtype=wp.bool, device=device),
        wp.array([0, 1, 0], dtype=int, device=device),
    ):
        with test.assertRaises(ValueError):
            cache.reset(mask)
    cache.reset()
    np.testing.assert_array_equal(cache.data.mode.numpy(), [0, 0, 0])
    cache.begin()
    cache.begin()
    np.testing.assert_array_equal(cache.data.generation.numpy(), [2])


def test_pipeline_world_epoch_reset(test, device):
    """Forward canonical local/global/all resets through a pipeline with an enabled cache."""
    device = wp.get_device(device)
    world = newton.ModelBuilder()
    # Two 167-body worlds plus a global box yield 28,056 genuine generic pairs,
    # above the unchanged 27,776-pair threshold for the automatic lean split path.
    for _ in range(167):
        body = world.add_body()
        world.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.5)
    builder.add_world(world)
    builder.add_world(world, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0)))
    model = builder.finalize(device=device)
    estimate = _compute_generic_convex_pair_work_estimate(
        model,
        broad_phase_mode="explicit",
        shape_pairs_filtered=model.shape_contact_pairs,
        candidate_pair_work_estimate=model.shape_contact_pairs.shape[0],
    )
    test.assertEqual(estimate, 28_056)
    test.assertGreaterEqual(estimate, _SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD)
    with patch.dict(
        os.environ,
        {
            "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "reject_only" if device.is_cuda else "0",
            "NEWTON_NARROW_PHASE_COHERENT_STATS": "0",
            "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": "0",
        },
    ):
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    test.assertTrue(pipeline.narrow_phase._use_lean_gjk_mpr)
    if device.is_cuda:
        test.assertTrue(pipeline.narrow_phase.split_gjk_mpr)
        test.assertIs(pipeline.shape_pairs_filtered, model.shape_contact_pairs)
    if device.is_cpu:
        # Query execution is CUDA-only; exercise the actual public reset path on CPU.
        pipeline.narrow_phase._coherent_cache = _ConvexQueryCache(
            pairs=model.shape_contact_pairs.numpy(),
            shape_types=model.shape_type.numpy(),
            shape_world=model.shape_world.numpy(),
            world_count=model.world_count,
            query_capacity=model.shape_contact_pairs.shape[0],
            device=device,
        )
    cache = pipeline.narrow_phase._coherent_cache
    test.assertIsNotNone(cache)
    test.assertEqual(model.world_count, 2)
    test.assertGreater(cache.data.mode.shape[0], 0)
    cache.data.mode.fill_(2)
    before = cache.data.mode.numpy()
    pipeline.reset_contact_matching(wp.array([True, False, False], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [1, 0, 0])
    pipeline.reset_contact_matching(wp.array([False, False, False], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [1, 0, 0])
    # A global endpoint can belong to any local bucket: conservatively invalidate all.
    pipeline.reset_contact_matching(wp.array([False, False, True], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [2, 1, 1])
    np.testing.assert_array_equal(cache.data.mode.numpy(), before)
    with test.assertRaisesRegex(ValueError, "model.world_count"):
        pipeline.reset_contact_matching(wp.array([True, False], dtype=wp.bool, device=device))
    pipeline.reset_contact_matching()
    np.testing.assert_array_equal(cache.data.mode.numpy(), np.zeros_like(before))
    np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [2, 1, 1])


def test_global_only_epoch_reset(test, device):
    """Accept the one-entry canonical mask when every shape is global."""
    cache = _ConvexQueryCache(
        pairs=np.array([[0, 1]]),
        shape_types=np.full(2, int(GeoType.BOX)),
        shape_world=np.array([-1, -1]),
        world_count=0,
        query_capacity=1,
        device=device,
    )
    np.testing.assert_array_equal(cache.data.world.numpy(), [0])
    cache.reset(wp.array([False], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [0])
    cache.reset(wp.array([True], dtype=wp.bool, device=device))
    np.testing.assert_array_equal(cache.data.world_epoch.numpy(), [1])
    with test.assertRaises(ValueError):
        cache.reset(wp.zeros(0, dtype=wp.bool, device=device))


def test_graph_stream_handoff(test, device):
    """Retain cache storage through an ordered eager-to-nondefault capture handoff."""
    cache = _ConvexQueryCache(
        pairs=np.array([[0, 1], [0, 2]]),
        shape_types=np.full(3, int(GeoType.BOX)),
        shape_world=np.array([-1, 0, 1]),
        world_count=2,
        query_capacity=2,
        device=device,
    )
    mask = wp.array([False, True, False], dtype=wp.bool, device=device)
    cache.begin()
    cache.reset(mask)
    if not wp.get_device(device).is_cuda:
        np.testing.assert_array_equal(cache.data.generation.numpy(), [1])
        return
    stream = wp.Stream(device)
    cache_ref = weakref.ref(cache)
    with wp.ScopedStream(stream, sync_enter=True):
        with wp.ScopedCapture(stream=stream) as capture:
            cache.begin()
            cache.reset(mask)
        graph = capture.graph
        del capture
        del cache
        gc.collect()
        test.assertIsNotNone(cache_ref())
        wp.capture_launch(graph, stream=stream)
        wp.capture_launch(graph, stream=stream)
        retained = cache_ref()
        np.testing.assert_array_equal(retained.data.generation.numpy(), [3])
        np.testing.assert_array_equal(retained.data.world_epoch.numpy(), [0, 3, 0])
        test.assertIsNotNone(retained._graph_token)
        del graph
        gc.collect()
        test.assertIsNone(retained._graph_token)


for _test in (
    test_random_full_shape_bound,
    test_invalid_and_shell_bound,
    test_world_epoch_reset,
    test_pipeline_world_epoch_reset,
    test_global_only_epoch_reset,
    test_graph_stream_handoff,
):
    add_function_test(TestConvexRejectionCache, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()

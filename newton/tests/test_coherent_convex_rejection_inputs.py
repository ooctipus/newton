# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise changing current geometry without trusting prior rejection hints."""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.coherent_convex import _ConvexQueryCache
from newton._src.geometry.coherent_convex_rejection import _create_rejection_query_kernels
from newton.tests.test_coherent_convex_rejection import _accepted, _box_inputs, _buffers, _original
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_current_vertices_scale_and_identity(test, device):
    """Recompute complete current support after scale, vertex and source/type changes."""
    threads = 128
    pairs, pair_array, values, inputs = _box_inputs(1, device)
    vertices = np.array(list(itertools.product((-0.5, 0.5), repeat=3)), np.float32)
    meshes = [
        wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=device),
            indices=wp.array([0, 1, 2], dtype=int, device=device),
        )
        for _ in range(2)
    ]
    values[0].fill_(int(newton.GeoType.CONVEX_MESH))
    values[1].assign(np.tile([1.0, 1.0, 1.0, 0.0], (2, 1)).astype(np.float32))
    values[3].assign(np.array([mesh.id for mesh in meshes], np.uint64))
    poses = values[2].numpy().copy()
    poses[1, 0] = 1.2
    values[2].assign(poses)
    count = wp.array([1], dtype=int, device=device)
    baseline, candidate = _buffers(1, device), _buffers(1, device)
    original = _original()
    kernels = _create_rejection_query_kernels(diagnostics=True)
    cache = _ConvexQueryCache(
        pairs=pairs,
        shape_types=values[0].numpy(),
        shape_world=np.zeros(2, np.int32),
        world_count=1,
        query_capacity=1,
        device=device,
    )

    def run_original():
        for buffer in baseline:
            buffer.zero_()
        wp.launch(
            original[0],
            dim=threads,
            inputs=[pair_array, count, *values, threads, *baseline],
            device=device,
            block_dim=128,
        )
        wp.launch(
            original[1], dim=threads, inputs=[pair_array, *values, threads, *baseline], device=device, block_dim=128
        )

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
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_allclose(
            actual["signed_distance"][actual_ids], expected["signed_distance"][expected_ids], rtol=0, atol=0
        )
        for name in ("point_a", "point_b", "normal"):
            np.testing.assert_array_equal(actual[name][actual_ids], expected[name][expected_ids])

    run_original()
    run_candidate()
    compare()
    graph = None
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            run_candidate()
        graph = capture.graph
    for case in ("warm", "source_mismatch", "type_mismatch", "scale", "restored", "vertices", "restored_again"):
        if case == "source_mismatch":
            cache.data.source_a.fill_(123)
        elif case == "type_mismatch":
            cache.data.types.fill_(wp.vec2i(int(newton.GeoType.BOX)))
        elif case == "scale":
            values[1].assign(np.array([[1, 1, 1, 0], [1.6, 1, 1, 0]], np.float32))
            values[8].assign(np.array([[-0.5, -0.5, -0.5], [-0.8, -0.5, -0.5]], np.float32))
            values[9].assign(np.array([[0.5, 0.5, 0.5], [0.8, 0.5, 0.5]], np.float32))
        elif case in ("restored", "restored_again"):
            values[1].assign(np.tile([1.0, 1.0, 1.0, 0.0], (2, 1)).astype(np.float32))
            values[8].fill_(wp.vec3(-0.5))
            values[9].fill_(wp.vec3(0.5))
            meshes[1].points.assign(vertices)
        elif case == "vertices":
            expanded = vertices.copy()
            expanded[:, 0] *= 1.6
            meshes[1].points.assign(expanded)
            values[8].assign(np.array([[-0.5, -0.5, -0.5], [-0.8, -0.5, -0.5]], np.float32))
            values[9].assign(np.array([[0.5, 0.5, 0.5], [0.8, 0.5, 0.5]], np.float32))
        run_original()
        with test.subTest(case=case):
            for replay in range(2):
                if graph is None:
                    run_candidate()
                else:
                    wp.capture_launch(graph)
                compare()
                test.assertEqual(len(_accepted(candidate)[0]), int(case in ("scale", "vertices")))
                if replay == 0 and case in ("source_mismatch", "type_mismatch"):
                    test.assertEqual(int(cache.data.stats.numpy()[2]), 0)
            if case == "warm":
                test.assertEqual(int(cache.data.stats.numpy()[2]), 1)


class TestCoherentConvexRejectionInputs(unittest.TestCase):
    pass


add_function_test(
    TestCoherentConvexRejectionInputs,
    "test_current_vertices_scale_and_identity",
    test_current_vertices_scale_and_identity,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()

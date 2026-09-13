# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check certified BSP queries, original-law subtree scans and complete collision behavior."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.coherent_convex import _ConvexQueryCache
from newton._src.geometry.coherent_convex_rejection import _create_rejection_query_kernels
from newton._src.geometry.convex_bsp import (
    _bind_shape,
    _BspData,
    _BspOwner,
    _create_support,
    _filtered_sign,
    _query,
    _scan_mask,
    _subtree_masks,
)
from newton._src.geometry.convex_bsp_build import build_tree, dot, integer_points, pack_plane
from newton._src.geometry.convex_bsp_factories import coherent_kernels, source_successor, split_kernels
from newton._src.geometry.narrow_phase import (
    ContactWriterData,
    create_narrow_phase_kernels_gjk_mpr_split,
    write_contact_simple,
)
from newton._src.geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    create_shape_support_function,
    support_map_lean,
)
from newton.tests.test_coherent_convex_rejection import _accepted, _box_inputs, _buffers, _original

TEST_DEVICE = os.environ.get("FPGS_TEST_DEVICE", "cpu")
_TEST_SUPPORT = _create_support(True)


@wp.kernel
def _support_points(
    data: _BspData,
    shape: GenericShapeData,
    rays: wp.array[wp.vec3],
    original: wp.array[wp.vec3],
    candidate: wp.array[wp.vec3],
    path: wp.array[int],
):
    i = wp.tid()
    bound = _bind_shape(shape, data)
    original[i] = wp.static(create_shape_support_function(support_map_lean, True))(
        shape, rays[i], SupportMapDataProvider()
    )
    candidate[i] = wp.static(create_shape_support_function(_TEST_SUPPORT, True))(bound, rays[i], data)
    path[i] = -1
    if bound.root >= 0:
        path[i] = _query(data, bound.root, wp.cw_mul(rays[i], shape.scale))


@wp.kernel
def _signs(planes: wp.array[wp.vec3], directions: wp.array[wp.vec3], result: wp.array[int]):
    i = wp.tid()
    result[i] = _filtered_sign(planes[i], directions[i])


@wp.kernel
def _queries(
    data: _BspData,
    points: wp.array[wp.vec3],
    roots: wp.array[int],
    directions: wp.array[wp.vec3],
    result: wp.array[int],
    indices: wp.array[int],
):
    i = wp.tid()
    path = _query(data, roots[i], directions[i])
    result[i] = path
    indices[i] = path
    if path < -1:
        indices[i] = _scan_mask(points, directions[i], data.leaf_masks[-path - 2])


@wp.kernel
def _original_indices(vertices: wp.array[wp.vec3], directions: wp.array[wp.vec3], result: wp.array[int]):
    i = wp.tid()
    maximum = float(-1.0e10)
    winner = int(0)
    for j in range(vertices.shape[0]):
        score = wp.dot(vertices[j], directions[i])
        if score > maximum:
            maximum = score
            winner = j
    result[i] = winner


@wp.kernel
def _mask_indices(
    vertices: wp.array[wp.vec3],
    masks: wp.array[wp.uint64],
    rays: wp.array[wp.vec3],
    selected: wp.array[int],
    actual: wp.array[int],
    expected: wp.array[int],
    scores: wp.array[wp.vec2],
):
    i = wp.tid()
    actual[i] = _scan_mask(vertices, rays[i], masks[i])
    maximum = float(-1.0e10)
    winner = int(0)
    for vertex in range(vertices.shape[0]):
        if (masks[i] & (wp.uint64(1) << wp.uint64(vertex))) != wp.uint64(0):
            score = wp.dot(vertices[vertex], rays[i])
            if score > maximum:
                maximum = score
                winner = vertex
    expected[i] = winner
    scores[i] = wp.vec2(wp.dot(vertices[selected[i]], rays[i]), wp.dot(vertices[winner], rays[i]))


def packed_tree(tree):
    """Bind rounded static test descriptors without captured coefficients."""
    data = _BspData()
    data.plane = wp.array(tree["rounded"], dtype=wp.vec3, device=TEST_DEVICE)
    data.node_plane = wp.array([n[0] for n in tree["nodes"]], dtype=int, device=TEST_DEVICE)
    data.children = wp.array([n[1:] for n in tree["nodes"]], dtype=wp.vec2i, device=TEST_DEVICE)
    data.leaf_masks = wp.array(
        _subtree_masks(tree["nodes"], len(tree["vertices"])), dtype=wp.uint64, device=TEST_DEVICE
    )
    return data


def check_queries(test, tree, rays):
    """Require exact certified winners and complete retained sets, recording numerical scan quality."""
    rays = np.asarray(rays, dtype=np.float32)
    output = wp.zeros(len(rays), dtype=int, device=TEST_DEVICE)
    indices = wp.zeros_like(output)
    data = packed_tree(tree)
    wp.launch(
        _queries,
        dim=len(rays),
        inputs=[
            data,
            wp.array(tree["vertices"], dtype=wp.vec3, device=TEST_DEVICE),
            wp.full(len(rays), tree["root"], dtype=int, device=TEST_DEVICE),
            wp.array(rays, dtype=wp.vec3, device=TEST_DEVICE),
        ],
        outputs=[output, indices],
        device=TEST_DEVICE,
    )
    original = wp.zeros(len(rays), dtype=int, device=TEST_DEVICE)
    wp.launch(
        _original_indices,
        dim=len(rays),
        inputs=[
            wp.array(tree["vertices"], dtype=wp.vec3, device=TEST_DEVICE),
            wp.array(rays, dtype=wp.vec3, device=TEST_DEVICE),
        ],
        outputs=[original],
        device=TEST_DEVICE,
    )
    counts = {
        "queries": len(rays),
        "changed_winners": 0,
        "changed_geometric_ties": 0,
        "strict_geometric_improvements": 0,
        "fallback_queries": 0,
        "subset_queries": 0,
        "subset_dots": 0,
        "max_subset_normalized_geometric_deficit": 0.0,
        "max_original_normalized_geometric_deficit": 0.0,
    }
    masks = data.leaf_masks.numpy()
    paths, selected_indices = output.numpy(), indices.numpy()
    subset = np.flatnonzero(paths < -1)
    if len(subset):
        actual, expected = [wp.zeros(len(subset), dtype=int, device=TEST_DEVICE) for _ in range(2)]
        scores = wp.zeros(len(subset), dtype=wp.vec2, device=TEST_DEVICE)
        wp.launch(
            _mask_indices,
            dim=len(subset),
            inputs=[
                wp.array(tree["vertices"], dtype=wp.vec3, device=TEST_DEVICE),
                wp.array(masks[-paths[subset] - 2], dtype=wp.uint64, device=TEST_DEVICE),
                wp.array(rays[subset], dtype=wp.vec3, device=TEST_DEVICE),
                wp.array(selected_indices[subset], dtype=int, device=TEST_DEVICE),
            ],
            outputs=[actual, expected, scores],
            device=TEST_DEVICE,
        )
        # Validate actual selected support against the independent original-law
        # masked loop, without imposing its arbitrary winner index on ties.
        numerical_scores = scores.numpy()
        np.testing.assert_array_equal(numerical_scores[:, 0], numerical_scores[:, 1])
    for ray, path, winner, old in zip(rays, paths, selected_indices, original.numpy(), strict=True):
        dint = integer_points(ray.reshape(1, 3))[0]
        scores = [dot(point, dint) for point in tree["points"]]
        if path >= 0:
            test.assertEqual(scores[winner], max(scores))
        else:
            counts["fallback_queries"] += 1
        selected = winner if path != -1 else old
        values = tree["vertices"].astype(float) @ ray.astype(float)
        if path < -1:
            mask = int(masks[-int(path) - 2])
            test.assertTrue(all(mask & (1 << i) for i, score in enumerate(scores) if score == max(scores)))
            counts["subset_queries"] += 1
            counts["subset_dots"] += mask.bit_count()
            counts["max_subset_normalized_geometric_deficit"] = max(
                counts["max_subset_normalized_geometric_deficit"],
                float(values.max() - values[selected]) / max(float(np.linalg.norm(ray.astype(float))), 1e-300),
            )
        counts["max_original_normalized_geometric_deficit"] = max(
            counts["max_original_normalized_geometric_deficit"],
            float(values.max() - values[old]) / max(float(np.linalg.norm(ray.astype(float))), 1e-300),
        )
        counts["changed_winners"] += int(selected != old)
        counts["changed_geometric_ties"] += int(selected != old and scores[selected] == scores[old])
        counts["strict_geometric_improvements"] += int(scores[selected] > scores[old])
    return counts


class _RecoverFactory(ast.NodeTransformer):
    """Strip only the documented descriptor/binding substitution for source recovery."""

    def visit_FunctionDef(self, node):
        node.name = node.name.removesuffix("_bsp")
        node.args.args = [a for a in node.args.args if a.arg != "support_data"]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("geom_a", "geom_b"):
                return None
            if (
                node.targets[0].id == "provider"
                and isinstance(node.value, ast.Name)
                and node.value.id == "support_data"
            ):
                node.value = ast.Call(func=ast.Name(id="SupportMapDataProvider", ctx=ast.Load()), args=[], keywords=[])
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in ("geom_a", "geom_b"):
            return ast.Attribute(value=ast.Name(id="query", ctx=ast.Load()), attr=node.id, ctx=node.ctx)
        if node.id == "_bsp_support":
            node.id = "support_map_lean"
        return node


class TestConvexBsp(unittest.TestCase):
    def test_subtree_scan_order_and_fallback(self):
        """Retain the ambiguous node, duplicated leaf union, bit63 and original scan tie order."""
        nodes = [(0, 1, 2), (1, -2, -64), (1, -2, -1)]
        masks = _subtree_masks(nodes, 64)
        self.assertEqual(masks, [1 | 2 | (1 << 63), 2 | (1 << 63), 3])
        self.assertEqual(_subtree_masks(nodes, 65), [0, 0, 0])
        vertices = np.random.default_rng(411).normal(size=(64, 3)).astype(np.float32)
        vertices[1] = [1, 1, 0]
        vertices[63] = [1, -1, 0]
        # Independent dense masked loop has no bit-extraction helper in common.
        rays = np.array([[1, 0, 0], [1, -1, 0], [0, 0, 0], [-1e20, 0, 0]], dtype=np.float32)
        subset = 2 | (1 << 63)
        actual, expected = [wp.zeros(4, dtype=int, device=TEST_DEVICE) for _ in range(2)]
        scores = wp.zeros(4, dtype=wp.vec2, device=TEST_DEVICE)
        inputs = [
            wp.array(vertices, dtype=wp.vec3, device=TEST_DEVICE),
            wp.array([subset] * 4, dtype=wp.uint64, device=TEST_DEVICE),
            wp.array(rays, dtype=wp.vec3, device=TEST_DEVICE),
            actual,
        ]
        wp.launch(
            _mask_indices,
            dim=4,
            inputs=inputs,
            outputs=[actual, expected, scores],
            device=TEST_DEVICE,
        )
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        np.testing.assert_array_equal(actual.numpy(), [1, 63, 1, 0])
        np.testing.assert_array_equal(scores.numpy()[:, 0], scores.numpy()[:, 1])
        inputs[-1] = wp.zeros(4, dtype=int, device=TEST_DEVICE)
        wp.launch(_mask_indices, dim=4, inputs=inputs, outputs=[actual, expected, scores], device=TEST_DEVICE)
        self.assertNotEqual(scores.numpy()[0, 0], scores.numpy()[0, 1])
        data = packed_tree({"vertices": vertices, "rounded": [[1, 0, 0], [0, 1, 0]], "nodes": nodes})
        query_rays = np.array([[1, 0, 0], [1, -1, 0], [np.inf, 0, 0], [np.nan, 0, 0]], dtype=np.float32)
        path, winner = [wp.zeros(4, dtype=int, device=TEST_DEVICE) for _ in range(2)]
        inputs = [
            data,
            wp.array(vertices, dtype=wp.vec3, device=TEST_DEVICE),
            wp.zeros(4, dtype=int, device=TEST_DEVICE),
            wp.array(query_rays, dtype=wp.vec3, device=TEST_DEVICE),
        ]
        wp.launch(_queries, dim=4, inputs=inputs, outputs=[path, winner], device=TEST_DEVICE)
        np.testing.assert_array_equal(path.numpy(), [-3, 63, -1, -1])
        np.testing.assert_array_equal(winner.numpy(), [1, 63, -1, -1])
        data.leaf_masks.zero_()
        wp.launch(_queries, dim=4, inputs=inputs, outputs=[path, winner], device=TEST_DEVICE)
        np.testing.assert_array_equal(path.numpy(), [-1, 63, -1, -1])

        # A real 66-vertex convex prism exercises the owner-to-full-callback seam.
        n = 33
        angles = np.arange(n) * (2 * np.pi / n)
        ring = np.column_stack([np.cos(angles), np.sin(angles)])
        prism = np.concatenate([np.column_stack([ring, np.full(n, z)]) for z in (-1, 1)]).astype(np.float32)
        faces = [[0, j + 1, j] for j in range(1, n - 1)]
        faces += [[n, n + j, n + j + 1] for j in range(1, n - 1)]
        for i in range(n):
            j = (i + 1) % n
            faces.extend([[i, j, n + j], [i, n + j, n + i]])
        mesh = wp.Mesh(
            wp.array(prism, dtype=wp.vec3, device=TEST_DEVICE),
            wp.array(np.asarray(faces).reshape(-1), dtype=int, device=TEST_DEVICE),
        )
        owner = _BspOwner(
            SimpleNamespace(
                _mesh_keep_alive=[mesh],
                shape_source_ptr=wp.array([mesh.id], dtype=wp.uint64, device=TEST_DEVICE),
                shape_type=wp.array([int(newton.GeoType.CONVEX_MESH)], dtype=int, device=TEST_DEVICE),
                device=wp.get_device(TEST_DEVICE),
            )
        )
        self.assertEqual(len(owner.metadata["admitted_meshes"]), 1)
        self.assertFalse(owner.data.leaf_masks.numpy().any())
        shape = GenericShapeData()
        shape.shape_type = int(newton.GeoType.CONVEX_MESH)
        shape.scale = wp.vec3(1.0, 1.0, 1.0)
        shape.auxiliary = wp.vec3(
            float(mesh.id & 0x3FFFFF), float((mesh.id >> 22) & 0x3FFFFF), float((mesh.id >> 44) & 0xFFFFF)
        )
        rays = wp.array([[0, 0, 0], [0, 0, 1], [0, 0, -1], [np.inf, 0, 0]], dtype=wp.vec3, device=TEST_DEVICE)
        outputs = [wp.zeros(4, dtype=wp.vec3, device=TEST_DEVICE) for _ in range(2)]
        wp.launch(
            _support_points, dim=4, inputs=[owner.data, shape, rays], outputs=[*outputs, path], device=TEST_DEVICE
        )
        np.testing.assert_array_equal(path.numpy(), [-1, -1, -1, -1])
        np.testing.assert_array_equal(outputs[0].numpy().view(np.uint32), outputs[1].numpy().view(np.uint32))

    def test_fp32_descriptor_contract(self):
        """Exclude FP64 limbs and their loads from every hot-path descriptor."""
        self.assertEqual(
            set(_BspData.vars), {"mesh_ids", "roots", "plane", "node_plane", "children", "leaf_masks", "valid"}
        )
        self.assertEqual(_BspData.vars["plane"].type.dtype, wp.vec3)
        self.assertEqual(_BspData.vars["leaf_masks"].type.dtype, wp.uint64)

    def test_primitive_policy_and_graph_invalidation(self):
        """Retain primitive ties and disable old graph-visible trees before geometry mutation."""
        device = TEST_DEVICE
        vertices = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        mesh = wp.Mesh(
            wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces.reshape(-1), dtype=int, device=device)
        )
        model = SimpleNamespace(
            _mesh_keep_alive=[mesh],
            shape_source_ptr=wp.array([mesh.id], dtype=wp.uint64, device=device),
            shape_type=wp.array([int(newton.GeoType.CONVEX_MESH)], dtype=int, device=device),
            device=wp.get_device(device),
        )
        owner = _BspOwner(model)
        shape = GenericShapeData()
        shape.scale = wp.vec3(1.0, 0.7, 0.9)
        rays = wp.array(
            np.concatenate(
                [
                    np.eye(3),
                    -np.eye(3),
                    np.zeros((1, 3)),
                    np.random.default_rng(731).normal(size=(193, 3)),
                    np.array([[1.0, 1e-10, -1e-10]]),
                ]
            ),
            dtype=wp.vec3,
            device=device,
        )
        original, candidate = [wp.zeros(len(rays), dtype=wp.vec3, device=device) for _ in range(2)]
        paths = wp.zeros(len(rays), dtype=int, device=device)
        for kind in (newton.GeoType.BOX, newton.GeoType.SPHERE):
            shape.shape_type = int(kind)
            wp.launch(
                _support_points,
                dim=len(rays),
                inputs=[owner.data, shape, rays],
                outputs=[original, candidate, paths],
                device=device,
            )
            np.testing.assert_array_equal(candidate.numpy(), original.numpy())
        shape.shape_type = int(newton.GeoType.CONVEX_MESH)
        shape.auxiliary = wp.vec3(
            float(mesh.id & 0x3FFFFF), float((mesh.id >> 22) & 0x3FFFFF), float((mesh.id >> 44) & 0xFFFFF)
        )

        def launch():
            wp.launch(
                _support_points,
                dim=len(rays),
                inputs=[owner.data, shape, rays],
                outputs=[original, candidate, paths],
                device=device,
            )

        launch()
        fallback = paths.numpy() < -1
        self.assertTrue(fallback.any())
        self.assertTrue((paths.numpy() >= 0).any())
        # A zero mask deliberately disables only subset scanning, as for >64 hulls.
        old_masks = owner.data.leaf_masks.numpy()
        owner.data.leaf_masks.zero_()
        launch()
        fallback = paths.numpy() == -1
        self.assertTrue(fallback.any())
        np.testing.assert_array_equal(
            candidate.numpy()[fallback].view(np.uint32), original.numpy()[fallback].view(np.uint32)
        )
        owner.data.leaf_masks.assign(old_masks)
        graph = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                launch()
            graph = capture.graph
        calls = []
        pipeline = SimpleNamespace(
            _convex_bsp=owner,
            model=model,
            narrow_phase=SimpleNamespace(_coherent_cache=SimpleNamespace(reset=lambda: calls.append("reset"))),
        )
        newton.CollisionPipeline.invalidate_convex_bsp(pipeline)
        self.assertEqual(calls, ["reset"])
        old_pointer = owner.data.node_plane.ptr
        changed = vertices.copy()
        changed[0, 0] += 0.6
        mesh.points.assign(changed)
        for _ in range(2):
            if graph is None:
                launch()
            else:
                wp.capture_launch(graph)
            np.testing.assert_array_equal(candidate.numpy().view(np.uint32), original.numpy().view(np.uint32))
        self.assertEqual(owner.data.node_plane.ptr, old_pointer)
        self.assertEqual(owner.data.valid.numpy()[0], 0)

    def test_cpu_pipeline_falls_back_before_construction(self):
        """An unsupported CPU non-split pipeline never constructs or claims the private owner."""
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        with (
            patch.dict(os.environ, {"NEWTON_NARROW_PHASE_CONVEX_BSP": "1", "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0"}),
            patch("newton._src.geometry.convex_bsp._BspOwner", side_effect=AssertionError("not admitted")),
        ):
            pipeline = newton.CollisionPipeline(model)
        self.assertTrue(pipeline._convex_bsp_requested)
        self.assertIsNone(pipeline._convex_bsp)

    def test_split_and_coherent_manifold_complete(self):
        """Compile every private factory and preserve controlled box/hull contact geometry and fallback."""
        device = TEST_DEVICE
        n = 9
        block = 1 if wp.get_device(device).is_cpu else 128
        threads = 9 if block == 1 else 128
        pairs, pair_array, values, inputs = _box_inputs(n, device)
        vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
            ],
            dtype=np.int32,
        )
        mesh = wp.Mesh(
            wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces.reshape(-1), dtype=int, device=device)
        )
        shape_types = values[0].numpy()
        shape_types[::2] = int(newton.GeoType.CONVEX_MESH)
        values[0].assign(shape_types)
        source = np.zeros(n * 2, dtype=np.uint64)
        source[::2] = mesh.id
        values[3].assign(source)
        shape_data = values[1].numpy()
        shape_data[::2, :3] = 1.0
        values[1].assign(shape_data)
        model = SimpleNamespace(
            _mesh_keep_alive=[mesh], shape_source_ptr=values[3], shape_type=values[0], device=wp.get_device(device)
        )
        owner = _BspOwner(model)
        self.assertEqual(len(owner.metadata["admitted_meshes"]), 1)
        narrow = SimpleNamespace(_use_lean_gjk_mpr=True, external_aabb=True, _convex_writer_func=write_contact_simple)
        candidate_kernels = split_kernels(narrow)
        baseline_kernels = _original()
        count = wp.array([n], dtype=int, device=device)

        def writer():
            data = ContactWriterData()
            data.contact_max = n * 8
            data.contact_count = wp.zeros(1, dtype=int, device=device)
            for name, dtype in [
                ("contact_pair", wp.vec2i),
                ("contact_position", wp.vec3),
                ("contact_normal", wp.vec3),
                ("contact_penetration", float),
                ("contact_tangent", wp.vec3),
                ("contact_sort_key", wp.int64),
            ]:
                setattr(data, name, wp.zeros(n * 8, dtype=dtype, device=device))
            return data

        def run(kernels, private, coherent=False):
            buffers = _buffers(n, device)
            extension = [owner.data] if private else []
            if coherent:
                cache = _ConvexQueryCache(
                    pairs=pairs,
                    shape_types=values[0].numpy(),
                    shape_world=np.zeros(2 * n, np.int32),
                    world_count=1,
                    query_capacity=n,
                    device=device,
                )
                cold = coherent_kernels()
                for _ in range(2):
                    for buffer in buffers:
                        buffer.zero_()
                    cache.begin()
                    for kernel in cold:
                        wp.launch(
                            kernel,
                            dim=threads,
                            inputs=[pair_array, count, inputs, cache.data, threads, *buffers, *extension],
                            device=device,
                            block_dim=block,
                        )
            else:
                wp.launch(
                    kernels[0],
                    dim=threads,
                    inputs=[pair_array, count, *values, threads, *buffers, *extension],
                    device=device,
                    block_dim=block,
                )
                wp.launch(
                    kernels[1],
                    dim=threads,
                    inputs=[pair_array, *values, threads, *buffers, *extension],
                    device=device,
                    block_dim=block,
                )
            data = writer()
            wp.launch(
                kernels[2],
                dim=threads,
                inputs=[pair_array, *values, data, threads, buffers[0], buffers[3], buffers[4], *extension],
                device=device,
                block_dim=block,
            )
            return buffers, data

        # Full and lean keep distinct factory modules and original postprocessors.
        narrow._use_lean_gjk_mpr = False
        full_kernels = split_kernels(narrow)
        full_original = create_narrow_phase_kernels_gjk_mpr_split(True, write_contact_simple)
        for invalidated in (False, True):
            if invalidated:
                owner.invalidate()
                changed = vertices.copy()
                changed[:, 0] *= 1.1
                mesh.points.assign(changed)
            baseline, expected_writer = run(baseline_kernels, False)
            expected_ids, expected = _accepted(baseline)
            for coherent in (False, True):
                candidate, actual_writer = run(candidate_kernels, True, coherent)
                ids, actual = _accepted(candidate)
                np.testing.assert_array_equal(np.sort(ids), np.sort(expected_ids))
                # Controlled axial fixture; full geometric rather than tie-index comparison.
                np.testing.assert_allclose(
                    actual[ids]["signed_distance"], expected[ids]["signed_distance"], rtol=0, atol=5e-5
                )
                np.testing.assert_allclose(actual[ids]["normal"], expected[ids]["normal"], rtol=0, atol=1e-3)
                num = int(actual_writer.contact_count.numpy()[0])
                self.assertGreater(num, 0)
                self.assertLessEqual(num, actual_writer.contact_max)
                normals = actual_writer.contact_normal.numpy()[:num]
                depths = actual_writer.contact_penetration.numpy()[:num]
                positions = actual_writer.contact_position.numpy()[:num]
                self.assertTrue(np.isfinite(positions).all())
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=0, atol=1e-3)
                self.assertLessEqual(float(np.max(depths)), 0.011)
                contact_pairs = actual_writer.contact_pair.numpy()[:num]
                body_x = values[2].numpy()[contact_pairs[:, 1], 0]
                extent = 0.55 if invalidated else 0.5
                np.testing.assert_allclose(depths, body_x - extent - 0.5, rtol=0, atol=5e-5)
                np.testing.assert_allclose(positions[:, 0], (extent + body_x - 0.5) * 0.5, rtol=0, atol=5e-5)
                np.testing.assert_allclose(normals, np.tile([1.0, 0.0, 0.0], (num, 1)), rtol=0, atol=1e-3)
                self.assertGreater(int(expected_writer.contact_count.numpy()[0]), 0)
            full_baseline, _ = run(full_original, False)
            full_candidate, _ = run(full_kernels, True)
            old_ids, old_values = _accepted(full_baseline)
            new_ids, new_values = _accepted(full_candidate)
            np.testing.assert_array_equal(np.sort(new_ids), np.sort(old_ids))
            np.testing.assert_allclose(
                new_values[new_ids]["signed_distance"], old_values[new_ids]["signed_distance"], rtol=0, atol=5e-5
            )
            np.testing.assert_allclose(new_values[new_ids]["normal"], old_values[new_ids]["normal"], rtol=0, atol=1e-3)

    def test_source_recovery(self):
        """Recover both original factories exactly after private provider plumbing is stripped."""
        for original, coherent in [
            (create_narrow_phase_kernels_gjk_mpr_split, False),
            (_create_rejection_query_kernels, True),
        ]:
            changed, baseline = source_successor(original, coherent=coherent)
            recovered = _RecoverFactory().visit(ast.parse(changed))
            self.assertEqual(ast.dump(recovered), ast.dump(ast.parse(baseline)))

    def test_native_random_and_ambiguous_signs(self):
        """Certify every returned sign and reject ambiguous admitted exponent-range queries."""
        rng = np.random.default_rng(912)
        planes = []
        directions = []
        for bits in (1, 24, 53, 77, 106):
            for _ in range(512):
                plane = tuple(
                    int(rng.integers(-(2**30), 2**30)) * 2 ** max(bits - 30, 0) + int(rng.integers(-16, 16))
                    for _ in range(3)
                )
                # Match the actual constructor's at-most-106-bit admission.
                if max(abs(x).bit_length() for x in plane) > 106:
                    continue
                ray = rng.normal(size=3).astype(np.float32) * np.float32(2.0 ** int(rng.integers(-120, 121)))
                planes.append(plane)
                directions.append(ray)
        # Near cancelling dyadic components must not produce guessed signs.
        for exponent in (25, 54, 80, 104):
            for delta in (-1, 0, 1):
                planes.append((2**exponent + delta, -(2**exponent), 0))
                directions.append(np.ones(3, dtype=np.float32))
        for value in (np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max):
            for sign in (-1, 1):
                planes.append((2**105 + 1, -(2**105), 1))
                directions.append(np.array([value, value, sign * value], dtype=np.float32))
        packed = [pack_plane(p) for p in planes]
        output = wp.zeros(len(planes), dtype=int, device=TEST_DEVICE)
        wp.launch(
            _signs,
            dim=len(planes),
            inputs=[
                wp.array([p[0] for p in packed], dtype=wp.vec3, device=TEST_DEVICE),
                wp.array(directions, dtype=wp.vec3, device=TEST_DEVICE),
            ],
            outputs=[output],
            device=TEST_DEVICE,
        )
        exact = [
            dot(p, integer_points(np.asarray(d).reshape(1, 3))[0]) for p, d in zip(planes, directions, strict=True)
        ]
        actual = output.numpy()
        certified = actual != 2
        self.assertTrue(certified.any())
        self.assertTrue((~certified).any())
        np.testing.assert_array_equal(actual[certified], np.array([int(v > 0) - int(v < 0) for v in exact])[certified])

    def test_exact_tetrahedron_and_rejection(self):
        """Keep certified queries exact, fall back on boundaries and reject malformed hulls."""
        vertices = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        tree = build_tree(vertices, faces)
        rays = np.concatenate(
            [np.random.default_rng(13).normal(size=(2048, 3)), np.eye(3), -np.eye(3), np.zeros((1, 3))]
        )
        check_queries(self, tree, rays)
        with self.assertRaises(ValueError):
            build_tree(vertices, faces[:-1])
        with self.assertRaises(ValueError):
            build_tree(np.concatenate([vertices, vertices[:1]]), faces)
        with self.assertRaises(ValueError):
            pack_plane((2**107 + 1, 1, 0))

    @unittest.skipUnless(os.environ.get("BSP_TEST_ASSET"), "optional local actual-hull input")
    def test_actual_cooked_hulls(self):
        """Check actual cooked hull geometry with random, face-normal and edge-boundary queries."""
        builder = newton.ModelBuilder()
        builder.add_usd(os.environ["BSP_TEST_ASSET"], load_visual_shapes=False)
        model = builder.finalize(device=TEST_DEVICE)
        owner = _BspOwner(model)
        self.assertFalse(owner.metadata["unsupported_meshes"])
        self.assertGreaterEqual(len(owner.metadata["admitted_meshes"]), 1)
        if wp.get_device(TEST_DEVICE).is_cuda:
            # Constructor wiring unit only: the tiny model is below the unchanged
            # 27,776-pair live split threshold. Actual task admission is separate.
            with (
                patch.dict(
                    os.environ, {"NEWTON_NARROW_PHASE_CONVEX_BSP": "1", "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0"}
                ),
                patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0),
            ):
                pipeline = newton.CollisionPipeline(
                    model, broad_phase="explicit", rigid_contact_max=256, max_triangle_pairs=256
                )
            self.assertIsNotNone(pipeline._convex_bsp)
            self.assertIs(pipeline.narrow_phase._bsp_owner, pipeline._convex_bsp)
            self.assertIn("bsp_support", pipeline.narrow_phase.narrow_phase_mpr_kernel.module.name)
            self.assertEqual(
                len(pipeline._convex_bsp.metadata["admitted_meshes"]), len(owner.metadata["admitted_meshes"])
            )
        count = 0
        differences = []
        for mesh in model._mesh_keep_alive:
            vertices = mesh.points.numpy()
            faces = mesh.indices.numpy().reshape(-1, 3)
            tree = build_tree(vertices, faces)
            a, b, c = (vertices[faces[:, i]].astype(float) for i in range(3))
            rays = np.concatenate(
                [np.random.default_rng(71).normal(size=(1024, 3)), np.cross(b - a, c - a), np.zeros((1, 3))]
            ).astype(np.float32)
            differences.append(check_queries(self, tree, rays))
            shape = GenericShapeData()
            shape.shape_type = int(newton.GeoType.CONVEX_MESH)
            scale = np.array([0.7, 1.3, 0.9], dtype=np.float32)
            shape.scale = wp.vec3(*scale)
            shape.auxiliary = wp.vec3(
                float(mesh.id & 0x3FFFFF), float((mesh.id >> 22) & 0x3FFFFF), float((mesh.id >> 44) & 0xFFFFF)
            )
            outputs = [wp.zeros(len(rays), dtype=wp.vec3, device=TEST_DEVICE) for _ in range(2)]
            paths = wp.zeros(len(rays), dtype=int, device=TEST_DEVICE)
            wp.launch(
                _support_points,
                dim=len(rays),
                inputs=[owner.data, shape, wp.array(rays, dtype=wp.vec3, device=TEST_DEVICE)],
                outputs=[*outputs, paths],
                device=TEST_DEVICE,
            )
            scaled_vertices = (vertices * scale).astype(np.float32)
            path_values = paths.numpy()
            fallback = path_values == -1
            np.testing.assert_array_equal(
                outputs[1].numpy()[fallback].view(np.uint32), outputs[0].numpy()[fallback].view(np.uint32)
            )
            self.assertTrue((path_values < -1).any())
            self.assertTrue((path_values >= 0).any())
            masks = owner.data.leaf_masks.numpy()
            for ray, point, path in zip(
                (rays * scale).astype(np.float32), outputs[1].numpy(), path_values, strict=True
            ):
                matches = np.flatnonzero(np.all(scaled_vertices == point, axis=1))
                self.assertGreater(len(matches), 0)
                direction = integer_points(ray.reshape(1, 3))[0]
                scores = [dot(p, direction) for p in tree["points"]]
                if path >= 0:
                    self.assertEqual(max(scores[int(i)] for i in matches), max(scores))
                elif path < -1:
                    mask = int(masks[-int(path) - 2])
                    self.assertTrue(all(mask & (1 << i) for i, score in enumerate(scores) if score == max(scores)))
            count += len(rays)
        print("BSP_ACTUAL", {"queries": count, "native_winner_differences": differences, **owner.metadata})

    @unittest.skipUnless(os.environ.get("BSP_TEST_CORPUS"), "optional pinned historical CPU directions")
    def test_historical_cpu_directions(self):
        """Check historical query directions natively without claiming current live weighting."""
        folder = Path(os.environ["BSP_TEST_CORPUS"])
        for gpu, tag in enumerate(("rtx", "gb")):
            differences = []
            with np.load(folder / f"calls_{tag}_cpu.npz") as archive:
                calls = {k: archive[k] for k in archive.files}
            hull_path = Path(os.environ[f"BSP_TEST_HULLS_{gpu}"])
            with np.load(hull_path) as archive:
                hulls = {k: archive[k] for k in archive.files}
            for h, count in enumerate(hulls["counts"]):
                start, i0, ni = hulls["offsets"][h], hulls["index_offsets"][h], hulls["index_counts"][h]
                tree = build_tree(
                    hulls["vertices"][start : start + count], hulls["indices"][i0 : i0 + ni].reshape(-1, 3)
                )
                selected = calls["hull_index"] == h
                rays = (calls["directions"][selected] * calls["scales"][selected]).astype(np.float32)
                differences.append(check_queries(self, tree, rays))
            print("BSP_HISTORICAL", tag, len(calls["hull_index"]), differences)

    def test_ambiguous_signs_require_fallback(self):
        """Return fallback for cancellation, true zero, lost coefficient tails and nonfinite rays."""
        planes = [pack_plane((2**100 + delta, -(2**100), 0))[0] for delta in (-1, 0, 1)]
        planes.extend([[1.0, 0.0, 0.0]] * 4)
        rays = np.ones((7, 3), dtype=np.float32)
        rays[4, 0] = -1.0
        rays[5, 0] = np.inf
        rays[6, 0] = np.nan
        output = wp.zeros(7, dtype=int, device=TEST_DEVICE)
        wp.launch(
            _signs,
            dim=7,
            inputs=[
                wp.array(planes, dtype=wp.vec3, device=TEST_DEVICE),
                wp.array(rays, dtype=wp.vec3, device=TEST_DEVICE),
            ],
            outputs=[output],
            device=TEST_DEVICE,
        )
        np.testing.assert_array_equal(output.numpy(), [2, 2, 2, 1, -1, 2, 2])


if __name__ == "__main__":
    unittest.main()

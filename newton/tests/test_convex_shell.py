# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused current-geometry tests for the opt-in warm convex shell patch."""

import ast
import hashlib
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.coherent_convex_shell import _lane
from newton._src.geometry.convex_shell import _release, _shell_patch, _ShellData, _ShellOwner, _ShellResult, _storage

TEST_DEVICE = os.environ.get("FPGS_TEST_DEVICE", "cpu")


@wp.kernel
def _box_patch(data: _ShellData, position: wp.vec3, shell: float, results: wp.array[_ShellResult]):
    worker, logical = wp.tid()
    lane = _lane()
    if lane[1] == 1 and logical != 0:
        return
    address = _storage()
    result = _shell_patch(
        address,
        data,
        -2,
        -2,
        wp.vec3(0.5),
        wp.vec3(0.5),
        wp.quat_identity(),
        position,
        wp.vec3(1.0, 0.0, 0.0),
        shell,
        _ShellResult(),
    )
    if logical == 0:
        results[worker] = result
    _release(address)


@wp.kernel
def _mesh_patch(
    data: _ShellData,
    indices: wp.vec2i,
    scale_a: wp.vec3,
    scale_b: wp.vec3,
    ta: wp.transform,
    tb: wp.transform,
    normal: wp.vec3,
    shell: float,
    results: wp.array[_ShellResult],
):
    worker, logical = wp.tid()
    lane = _lane()
    if lane[1] == 1 and logical != 0:
        return
    qa, qb = wp.transform_get_rotation(ta), wp.transform_get_rotation(tb)
    relative_q = wp.quat_inverse(qa) * qb
    relative_p = wp.quat_rotate_inv(qa, wp.transform_get_translation(tb) - wp.transform_get_translation(ta))
    address = _storage()
    result = _shell_patch(
        address,
        data,
        indices[0],
        indices[1],
        scale_a,
        scale_b,
        relative_q,
        relative_p,
        normal,
        shell,
        _ShellResult(),
    )
    if logical == 0:
        results[worker] = result
    _release(address)


def _empty_data(device):
    data = _ShellData()
    for name in ("mesh_ids", "point_ptr"):
        setattr(data, name, wp.empty(0, dtype=wp.uint64, device=device))
    for name in ("point_count", "face_offsets", "edge_offsets"):
        setattr(data, name, wp.empty(0, dtype=int, device=device))
    data.faces = wp.empty(0, dtype=wp.vec3i, device=device)
    data.edges = wp.empty(0, dtype=wp.vec2i, device=device)
    data.valid = wp.full(1, 1, dtype=int, device=device)
    return data


def _warm_fixture(device):
    from newton._src.geometry.coherent_convex_shell import _create_shell_kernel  # noqa: PLC0415
    from newton._src.geometry.coherent_convex_warm import _ConvexQueryCache  # noqa: PLC0415
    from newton._src.geometry.coherent_convex_warm_queries import _create_query_kernels, _PairInputs  # noqa: PLC0415
    from newton._src.geometry.convex_bsp import _BspOwner  # noqa: PLC0415
    from newton._src.geometry.convex_bsp_factories import split_kernels  # noqa: PLC0415
    from newton._src.geometry.narrow_phase import ContactWriterData, write_contact_simple  # noqa: PLC0415
    from newton.tests.test_coherent_convex_rejection import _box_inputs, _buffers  # noqa: PLC0415

    n = 6
    pairs, pair_array, values, _ = _box_inputs(n, device)
    # The unchanged full-hull roundoff envelope intentionally refuses metre
    # boxes at the 1e-4 warm-query gap-width bound. Use actual hand-link scale.
    shape_data = values[1].numpy()
    shape_data[:, :3] *= 0.1
    values[1].assign(shape_data)
    poses = values[2].numpy()
    poses[:, :3] *= 0.1
    values[2].assign(poses)
    for index in range(4, 10):
        values[index].assign(values[index].numpy() * 0.1)
    inputs = _PairInputs()
    for name, value in zip(inputs._cls.__annotations__, values, strict=True):
        setattr(inputs, name, value)
    model = SimpleNamespace(_mesh_keep_alive=[], shape_source_ptr=values[3], shape_type=values[0], device=device)
    bsp = _BspOwner(model)
    shell = _ShellOwner(model, bsp)
    narrow = SimpleNamespace(_use_lean_gjk_mpr=True, external_aabb=True, _convex_writer_func=write_contact_simple)
    cold = split_kernels(narrow)
    cache = _ConvexQueryCache(
        pairs=pairs,
        shape_types=values[0].numpy(),
        shape_world=np.arange(n * 2) // 2,
        world_count=n,
        query_capacity=n,
        device=device,
    )
    kernels = _create_query_kernels(diagnostics=True)
    commit = _create_shell_kernel(write_contact_simple)
    buffers = _buffers(n, device)
    count = wp.array([n], dtype=int, device=device)
    writer = ContactWriterData()
    writer.contact_max = n * 5
    writer.contact_count = wp.zeros(1, dtype=int, device=device)
    for name, dtype in (
        ("contact_pair", wp.vec2i),
        ("contact_position", wp.vec3),
        ("contact_normal", wp.vec3),
        ("contact_penetration", float),
        ("contact_tangent", wp.vec3),
        ("contact_sort_key", wp.int64),
    ):
        setattr(writer, name, wp.zeros(n * 5, dtype=dtype, device=device))

    def launch():
        for buffer in buffers:
            buffer.zero_()
        writer.contact_count.zero_()
        cache.begin()
        for stage, kernel in enumerate(kernels):
            wp.launch(
                kernel,
                dim=128,
                inputs=[pair_array, count, inputs, cache.data, 128, *buffers, bsp.data],
                device=device,
                block_dim=128,
            )
            if stage == 1:
                wp.launch(
                    commit,
                    dim=(1, 32),
                    inputs=[pair_array, inputs, cache.data, 1, buffers[0], shell.data, writer],
                    device=device,
                    block_dim=32,
                )
        wp.launch(
            cold[2],
            dim=128,
            inputs=[pair_array, *values, writer, 128, buffers[0], buffers[3], buffers[4], bsp.data],
            device=device,
            block_dim=128,
        )

    def baseline_launch():
        for buffer in buffers:
            buffer.zero_()
        writer.contact_count.zero_()
        wp.launch(
            cold[0], dim=128, inputs=[pair_array, count, *values, 128, *buffers, bsp.data], device=device, block_dim=128
        )
        wp.launch(cold[1], dim=128, inputs=[pair_array, *values, 128, *buffers, bsp.data], device=device, block_dim=128)
        wp.launch(
            cold[2],
            dim=128,
            inputs=[pair_array, *values, writer, 128, buffers[0], buffers[3], buffers[4], bsp.data],
            device=device,
            block_dim=128,
        )

    return SimpleNamespace(
        launch=launch,
        baseline_launch=baseline_launch,
        cache=cache,
        buffers=buffers,
        writer=writer,
        values=values,
        bsp=bsp,
        shell=shell,
        count=count,
        kernels=kernels,
        commit=commit,
    )


class TestConvexShell(unittest.TestCase):
    def test_native_shell_api(self):
        """The complete native shell primitive is available without pipeline mutation."""
        from newton._src.geometry.convex_shell import _shell_patch, _ShellData, _ShellResult  # noqa: PLC0415

        self.assertIsNotNone(_ShellData)
        self.assertIsNotNone(_ShellResult)
        self.assertIsNotNone(_shell_patch)

    def test_cpu_current_box_patch(self):
        """Current positive boxes retain a broad complete shell and strict gaps."""
        data = _empty_data(TEST_DEVICE)
        out = wp.empty(1, dtype=_ShellResult, device=TEST_DEVICE)
        wp.launch(
            _box_patch,
            dim=(1, 32),
            block_dim=32,
            inputs=[data, wp.vec3(1.002, 0.0, 0.0), 0.005, out],
            device=TEST_DEVICE,
        )
        result = out.numpy()[0]
        self.assertEqual(int(result["reason"]), 0)
        self.assertEqual(int(result["count"]), 4)
        a, b = result["point_a"][:4], result["point_b"][:4]
        self.assertGreater(float(np.ptp(a[:, 1])), 0.999)
        self.assertGreater(float(np.ptp(a[:, 2])), 0.999)
        np.testing.assert_allclose(a[:, 0], 0.5, atol=1e-6)
        np.testing.assert_allclose(b[:, 0], 0.502, atol=1e-6)
        self.assertTrue(np.all((b - a)[:, 0] < 0.005))

    def test_cpu_refusal_no_partial_result(self):
        """Separated, penetrating and invalidated inputs publish no partial patch."""
        data = _empty_data(TEST_DEVICE)
        out = wp.empty(1, dtype=_ShellResult, device=TEST_DEVICE)
        for offset in (1.1, 0.9):
            wp.launch(
                _box_patch,
                dim=(1, 32),
                block_dim=32,
                inputs=[data, wp.vec3(offset, 0.0, 0.0), 0.005, out],
                device=TEST_DEVICE,
            )
            self.assertEqual(int(out.numpy()[0]["count"]), 0)
        data.valid.zero_()
        wp.launch(
            _box_patch,
            dim=(1, 32),
            block_dim=32,
            inputs=[data, wp.vec3(1.002, 0.0, 0.0), 0.005, out],
            device=TEST_DEVICE,
        )
        self.assertEqual(int(out.numpy()[0]["count"]), 0)

    def test_nonuniform_scale_and_reversed_pair(self):
        """Both pair orientations use current anisotropic geometry, not cached points."""
        data = _empty_data(TEST_DEVICE)
        out = wp.empty(1, dtype=_ShellResult, device=TEST_DEVICE)
        qa = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.6)
        ta = wp.transform(wp.vec3(0.4, -0.2, 0.1), qa)
        tb = wp.transform(wp.transform_get_translation(ta) + wp.quat_rotate(qa, wp.vec3(0.0305, 0.0, 0.0)), qa)
        for reverse in (False, True):
            sa, sb = wp.vec3(0.02, 0.03, 0.04), wp.vec3(0.01, 0.025, 0.035)
            first, second = ta, tb
            direction = 1.0
            if reverse:
                sa, sb, first, second, direction = sb, sa, tb, ta, -1.0
            wp.launch(
                _mesh_patch,
                dim=(1, 32),
                block_dim=32,
                inputs=[data, wp.vec2i(-2, -2), sa, sb, first, second, wp.vec3(direction, 0.0, 0.0), 0.002, out],
                device=TEST_DEVICE,
            )
            result = out.numpy()[0]
            self.assertEqual(int(result["reason"]), 0)
            self.assertEqual(int(result["count"]), 4)
            a, b = result["point_a"][:4], result["point_b"][:4]
            np.testing.assert_allclose(a[:, 0], direction * sa[0], atol=1e-6)
            np.testing.assert_allclose(direction * (b - a)[:, 0], 0.0005, atol=1e-6)
            self.assertGreater(float(np.ptp(a[:, 1])), 0.049)
            self.assertGreater(float(np.ptp(a[:, 2])), 0.069)

    def test_saved_current_shell_failures_cpu(self):
        """The same two old fitted-projector failures have valid broad current shell patches."""
        for gpu, pair, minimum_span, frame_pin in (
            (0, (1067, 1077), 0.02, "f6cdc2365126624618e0d68e40875eb7435f0dd4c0f608cc0f34516727ff497e"),
            (1, (2851, 2853), 0.08, "909a284adbacf299f71817123a90c9cc794dd65ca54a29809e147f570caa7053"),
        ):
            folder = Path(f"/tmp/fpgs-coherent-live-quality-gpu{gpu}-512-20260911-01")
            if not folder.exists():
                self.skipTest("the pre-existing paired contact-quality captures are not installed")
            self.assertEqual(hashlib.sha256((folder / "frame_00.npz").read_bytes()).hexdigest(), frame_pin)
            self.assertEqual(
                hashlib.sha256((folder / "model.npz").read_bytes()).hexdigest(),
                "e4a21145f3571549471b38453cb7bca729a818702c6ad2524cae30afa7901d2e",
            )
            model, frame = np.load(folder / "model.npz"), np.load(folder / "frame_00.npz")
            meshes, original_points, original_faces = [], [], []
            for shape in pair:
                pointer = int(model["shape_source_ptr"][shape])
                if pointer == 0:
                    meshes.append(None)
                    original_points.append(None)
                    original_faces.append(None)
                    continue
                points = model[f"mesh_{pointer}_points"]
                faces = model[f"mesh_{pointer}_indices"]
                original_points.append(points)
                original_faces.append(faces.reshape(-1, 3))
                meshes.append(
                    wp.Mesh(
                        points=wp.array(points, dtype=wp.vec3, device=TEST_DEVICE),
                        indices=wp.array(faces, dtype=int, device=TEST_DEVICE),
                    )
                )
            bsp = SimpleNamespace(
                metadata={"admitted_meshes": [{"mesh_id": mesh.id} for mesh in meshes if mesh is not None]},
                data=SimpleNamespace(valid=wp.full(1, 1, dtype=int, device=TEST_DEVICE)),
            )
            owner = _ShellOwner(
                SimpleNamespace(_mesh_keep_alive=[m for m in meshes if m is not None], device=TEST_DEVICE), bsp
            )
            ids = owner.data.mesh_ids.numpy().tolist()
            mesh_indices = wp.vec2i([ids.index(mesh.id) if mesh is not None else -2 for mesh in meshes])
            qi = int(np.flatnonzero(np.all(frame["actual_pairs"] == pair, axis=1))[0])
            normal = frame["actual_results"][qi]["normal"]
            transforms = [
                wp.transform(frame["actual_geom_transform"][shape, :3], frame["actual_geom_transform"][shape, 3:])
                for shape in pair
            ]
            scales = [frame["actual_geom_data"][shape, :3] for shape in pair]
            shell = float(
                sum(float(model["shape_gap"][shape]) + float(frame["actual_geom_data"][shape, 3]) for shape in pair)
            )
            out = wp.empty(1, dtype=_ShellResult, device=TEST_DEVICE)
            wp.launch(
                _mesh_patch,
                dim=(1, 32),
                block_dim=32,
                inputs=[
                    owner.data,
                    mesh_indices,
                    wp.vec3(scales[0]),
                    wp.vec3(scales[1]),
                    *transforms,
                    wp.vec3(normal),
                    shell,
                    out,
                ],
                device=TEST_DEVICE,
            )
            result = out.numpy()[0]
            with self.subTest(gpu=gpu, pair=pair, reason=int(result["reason"])):
                self.assertEqual(int(result["reason"]), 0)
                self.assertGreaterEqual(int(result["count"]), 3)
                a, b = result["point_a"][: result["count"]], result["point_b"][: result["count"]]
                span = np.linalg.norm(a[:, None] - a[None, :], axis=-1).max()
                self.assertGreater(float(span), minimum_span)
                self.assertTrue(np.all((b - a) @ normal < shell))
                # Independent FP64 planes from the original captured hulls,
                # not the native projected/slab representation.
                qa = wp.transform_get_rotation(transforms[0])
                qb = wp.transform_get_rotation(transforms[1])
                relative_q = wp.quat_inverse(qa) * qb
                relative_p = wp.quat_rotate_inv(
                    qa, wp.transform_get_translation(transforms[1]) - wp.transform_get_translation(transforms[0])
                )
                for side, witnesses in enumerate((a, b)):
                    if meshes[side] is None:
                        local = (
                            np.asarray([[(1.0 if i & (1 << k) else -1.0) for k in range(3)] for i in range(8)])
                            * scales[side]
                        )
                        from scipy.spatial import ConvexHull

                        triangles = ConvexHull(local).simplices
                    else:
                        local = original_points[side].astype(np.float64) * scales[side]
                        triangles = original_faces[side]
                    if side == 1:
                        local = np.asarray(
                            [np.asarray(wp.quat_rotate(relative_q, wp.vec3(p))) + np.asarray(relative_p) for p in local]
                        )
                    center = local.mean(axis=0)
                    x, y, z = (local[triangles[:, k]].astype(np.float64) for k in range(3))
                    planes = np.cross(y - x, z - x)
                    planes /= np.linalg.norm(planes, axis=1)[:, None]
                    planes[np.einsum("ij,ij->i", planes, x - center) < 0] *= -1
                    residual = witnesses.astype(np.float64) @ planes.T - np.einsum("ij,ij->i", x, planes)
                    self.assertLess(float(residual.max()), 1e-6)
                    self.assertLess(float(np.max(np.min(np.abs(residual), axis=1))), 1e-6)
                print("CURRENT_SHELL", gpu, int(result["cuts"]), int(result["polygon_count"]), float(span))

    def test_original_simplex_source_recovery(self):
        """Removing exactly the old optional callbacks recovers current cold simplex math."""
        from newton._src.geometry.coherent_convex_simplex import create_solve_closest_distance as warm  # noqa: PLC0415
        from newton._src.geometry.simplex_solver import create_solve_closest_distance as original  # noqa: PLC0415

        class Recover(ast.NodeTransformer):
            def visit_Expr(self, node):
                return (
                    None
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                    else self.generic_visit(node)
                )

            def visit_FunctionDef(self, node):
                if node.name == "create_solve_closest_distance":
                    node.args.kwonlyargs, node.args.kw_defaults = [], []
                return self.generic_visit(node)

            def visit_Assign(self, node):
                if any(isinstance(target, ast.Name) and target.id == "callbacks" for target in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                names = {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
                if names & {"callbacks", "_initial_simplex", "_record_vertex", "_final_simplex"}:
                    return None
                return self.generic_visit(node)

        before = Recover().visit(ast.parse(inspect.getsource(original)))
        after = Recover().visit(ast.parse(inspect.getsource(warm)))
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_complete_cold_warm_reset_and_fallback_cpu(self):
        """A warm success or pre-write refusal has exactly one complete publication owner."""
        fixture = _warm_fixture(TEST_DEVICE)
        for case in ("cold", "warm", "subset_reset", "source_changed", "invalidated", "empty"):
            if case == "subset_reset":
                fixture.cache.reset(
                    wp.array([False, True, False, False, False, False, False], dtype=wp.bool, device=TEST_DEVICE)
                )
            elif case == "source_changed":
                fixture.cache.data.source_a.fill_(123)
            elif case == "invalidated":
                fixture.bsp.invalidate()
            elif case == "empty":
                fixture.count.zero_()
            fixture.launch()
            done = fixture.cache.data.query_done.numpy()
            count = int(fixture.writer.contact_count.numpy()[0])
            self.assertLessEqual(count, fixture.writer.contact_max)
            if case == "empty":
                self.assertEqual(count, 0)
                continue
            self.assertGreater(count, 0)
            pairs = fixture.writer.contact_pair.numpy()[:count]
            depth = fixture.writer.contact_penetration.numpy()[:count]
            normal = fixture.writer.contact_normal.numpy()[:count]
            positions = fixture.writer.contact_position.numpy()[:count]
            self.assertTrue(np.isfinite(positions).all())
            np.testing.assert_allclose(normal, np.tile([1.0, 0.0, 0.0], (count, 1)), atol=1e-3)
            offset = fixture.values[2].numpy()[pairs[:, 1], 0]
            np.testing.assert_allclose(depth, offset - 0.1, atol=5e-5)
            self.assertEqual(set(map(tuple, pairs)), {(0, 1), (2, 3), (6, 7), (8, 9)})
            for pair in {(0, 1), (2, 3), (6, 7), (8, 9)}:
                self.assertLessEqual(int(np.count_nonzero(np.all(pairs == pair, axis=1))), 5)
            unresolved_count = int(fixture.cache.data.unresolved_work_count.numpy()[0])
            unresolved = fixture.cache.data.unresolved_work_items.numpy()[:unresolved_count]
            self.assertEqual(len(np.unique(unresolved)), unresolved_count)
            self.assertFalse(set(unresolved) & set(np.flatnonzero(done == 3)))
            if case == "warm":
                self.assertEqual(int(np.count_nonzero(done == 3)), 2)
            if case in ("cold", "source_changed", "invalidated"):
                self.assertFalse(np.any(done == 3))

    def test_flag_admission(self):
        """The new path is default off and refuses missing existing provider prerequisites."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device=TEST_DEVICE)
        for values in (
            {"NEWTON_NARROW_PHASE_COHERENT_SHELL": "bad"},
            {"NEWTON_NARROW_PHASE_COHERENT_SHELL": "1", "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0"},
            {
                "NEWTON_NARROW_PHASE_COHERENT_SHELL": "1",
                "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "reject_only",
                "NEWTON_NARROW_PHASE_CONVEX_BSP": "0",
            },
        ):
            with patch.dict(os.environ, values), self.assertRaises(ValueError):
                newton.CollisionPipeline(model, broad_phase="explicit")
        with patch.dict(
            os.environ,
            {
                "NEWTON_NARROW_PHASE_COHERENT_SHELL": "0",
                "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0",
                "NEWTON_NARROW_PHASE_CONVEX_BSP": "0",
            },
        ):
            pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
        self.assertIsNone(pipeline.narrow_phase._coherent_shell)

    def test_current_geometry_graph_and_original_fallback(self):
        """Current transforms/reset survive repeated graph replay; unsupported pairs keep original queries."""
        expected, actual = _warm_fixture(TEST_DEVICE), _warm_fixture(TEST_DEVICE)
        expected.baseline_launch()
        actual.launch()
        actual.launch()
        device = wp.get_device(TEST_DEVICE)
        graph = None
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                actual.launch()
            graph = capture.graph
        for ordinal, offset in enumerate((0.1004, 0.15, 0.09, 0.1003)):
            poses = actual.values[2].numpy()
            poses[3, 0] = offset
            for fixture in (expected, actual):
                fixture.values[2].assign(poses)
            if ordinal == 2:
                actual.cache.reset(
                    wp.array([False, True, False, False, False, False, False], dtype=wp.bool, device=TEST_DEVICE)
                )
            expected.baseline_launch()
            if graph is None:
                actual.launch()
            else:
                wp.capture_launch(graph)
            records = []
            for fixture in (expected, actual):
                count = int(fixture.writer.contact_count.numpy()[0])
                self.assertLessEqual(count, fixture.writer.contact_max)
                pair = fixture.writer.contact_pair.numpy()[:count]
                depth = fixture.writer.contact_penetration.numpy()[:count]
                normal = fixture.writer.contact_normal.numpy()[:count]
                self.assertTrue(np.isfinite(depth).all())
                records.append(
                    {
                        tuple(key): (
                            float(np.mean(depth[np.all(pair == key, axis=1)])),
                            np.mean(normal[np.all(pair == key, axis=1)], axis=0),
                        )
                        for key in np.unique(pair, axis=0)
                    }
                )
            self.assertEqual(records[0].keys(), records[1].keys())
            for key in records[0]:
                self.assertAlmostEqual(records[0][key][0], records[1][key][0], delta=5e-5)
                np.testing.assert_allclose(records[0][key][1], records[1][key][1], atol=1e-3)
        # A type outside the cached BOX/CONVEX set must still execute the
        # unchanged original support map and complete cold manifold.
        for fixture in (expected, actual):
            types = fixture.values[0].numpy()
            types[2:4] = int(newton.GeoType.SPHERE)
            fixture.values[0].assign(types)
        expected.baseline_launch()
        actual.launch()
        expected_count, actual_count = [int(f.writer.contact_count.numpy()[0]) for f in (expected, actual)]
        self.assertEqual(
            set(map(tuple, expected.writer.contact_pair.numpy()[:expected_count])),
            set(map(tuple, actual.writer.contact_pair.numpy()[:actual_count])),
        )

    def test_bounded_cap_refuses_before_publication(self):
        """A slab requiring more than 64 candidate vertices refuses without partial contacts."""
        from scipy.spatial import ConvexHull

        theta = np.arange(32) * (2.0 * np.pi / 32)
        circle = np.column_stack((np.cos(theta), np.sin(theta))) * 0.5
        lower_circle = np.column_stack((np.cos(theta + np.pi / 32), np.sin(theta + np.pi / 32))) * 0.5
        vertices = np.vstack(
            (np.column_stack((lower_circle, np.full(32, -0.5))), np.column_stack((circle, np.full(32, 0.5))))
        ).astype(np.float32)
        faces = ConvexHull(vertices).simplices.astype(np.int32)
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=TEST_DEVICE),
            indices=wp.array(faces.reshape(-1), dtype=int, device=TEST_DEVICE),
        )
        bsp = SimpleNamespace(
            metadata={"admitted_meshes": [{"mesh_id": mesh.id}]},
            data=SimpleNamespace(valid=wp.full(1, 1, dtype=int, device=TEST_DEVICE)),
        )
        owner = _ShellOwner(SimpleNamespace(_mesh_keep_alive=[mesh], device=TEST_DEVICE), bsp)
        out = wp.empty(1, dtype=_ShellResult, device=TEST_DEVICE)
        wp.launch(
            _mesh_patch,
            dim=(1, 32),
            block_dim=32,
            inputs=[
                owner.data,
                wp.vec2i(0, 0),
                wp.vec3(1.0),
                wp.vec3(1.0),
                wp.transform_identity(),
                wp.transform(wp.vec3(0.0, 0.0, 1.002), wp.quat_identity()),
                wp.vec3(0.0, 0.0, 1.0),
                0.005,
                out,
            ],
            device=TEST_DEVICE,
        )
        result = out.numpy()[0]
        self.assertEqual(int(result["count"]), 0)
        self.assertEqual(int(result["reason"]), 4)

    def test_graph_cache_lease_and_reset_mask(self):
        """Retain one graph lineage and the original explicit world-mask validation."""
        fixture = _warm_fixture(TEST_DEVICE)
        cache = fixture.cache
        first, second = SimpleNamespace(), SimpleNamespace()
        cache._acquire_graph(first)
        cache._acquire_graph(first)
        self.assertEqual(len(first._newton_coherent_convex_leases), 1)
        with self.assertRaisesRegex(RuntimeError, "two live CUDA graphs"):
            cache._acquire_graph(second)
        first._newton_coherent_convex_leases[0].release()
        cache._acquire_graph(second)
        second._newton_coherent_convex_leases[0].release()
        with self.assertRaises(ValueError):
            cache.reset(wp.zeros(6, dtype=wp.bool, device=TEST_DEVICE))
        with self.assertRaises(ValueError):
            cache.reset(wp.zeros(7, dtype=int, device=TEST_DEVICE))


if __name__ == "__main__":
    unittest.main()

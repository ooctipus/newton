# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check immutable cell coverage and all four unchanged query/manifold owners."""

import hashlib
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.coherent_convex import _ConvexQueryCache, _support_feature
from newton._src.geometry.coherent_convex_rejection import _create_rejection_query_kernels
from newton._src.geometry.convex_cells import (
    _bind_shape,
    _cell_support_feature,
    _CellData,
    _CellOwner,
    _create_support,
    _direction_cell,
    _query,
)
from newton._src.geometry.convex_cells_factories import coherent_kernels, manifold_kernel, source_successor
from newton._src.geometry.narrow_phase import ContactWriterData, write_contact_simple
from newton._src.geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    create_shape_support_function,
    support_map_lean,
)
from newton.tests.test_coherent_convex_rejection import _accepted, _box_inputs, _buffers, _original

VERTICES = np.array([[x, y, z] for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)], np.float32)
FACES = np.array(
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
    np.int32,
)
SUPPORT = _create_support(True)


def owner_fixture(device, slots=1, vertices=VERTICES, faces=FACES):
    """Bind an exact cooked mesh without allocating persistent pair hints."""
    mesh = wp.Mesh(
        wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces.reshape(-1), dtype=int, device=device)
    )
    model = SimpleNamespace(
        _mesh_keep_alive=[mesh],
        device=wp.get_device(device),
        shape_source_ptr=wp.array([mesh.id], dtype=wp.uint64, device=device),
        shape_type=wp.array([int(newton.GeoType.CONVEX_MESH)], dtype=int, device=device),
    )
    return _CellOwner(model), mesh


def cell_boundary_rays():
    """Include cell corners, adjacent FP32 values, face ties and signed directions."""
    grid = np.linspace(-1, 1, 17, dtype=np.float32)
    rays = []
    for axis in range(3):
        other = [k for k in range(3) if k != axis]
        for sign in (-1, 1):
            for x in grid:
                for y in grid:
                    for shift in (-np.inf, None, np.inf):
                        ray = np.zeros(3, np.float32)
                        ray[axis] = sign
                        ray[other[0]] = x if shift is None else np.nextafter(x, np.float32(shift))
                        ray[other[1]] = y
                        rays.append(ray)
    return np.asarray(rays)


@wp.kernel
def cell_check(rays: wp.array[wp.vec3], output: wp.array[int]):
    i = wp.tid()
    output[i] = _direction_cell(rays[i])


@wp.kernel
def support_check(
    data: _CellData,
    shape: GenericShapeData,
    rays: wp.array[wp.vec3],
    old: wp.array[wp.vec3],
    candidate: wp.array[wp.vec3],
    paths: wp.array[int],
    feature: wp.array[wp.vec2],
    magnitude: wp.array[wp.vec2],
):
    i = wp.tid()
    bound = _bind_shape(shape, data)
    old[i] = wp.static(create_shape_support_function(support_map_lean, True))(shape, rays[i], SupportMapDataProvider())
    candidate[i] = wp.static(create_shape_support_function(SUPPORT, True))(bound, rays[i], data)
    path, _score = _query(bound, wp.cw_mul(rays[i], shape.scale), data)
    paths[i] = path
    a = _support_feature(shape, rays[i])
    b = _cell_support_feature(bound, rays[i], data)
    feature[i] = wp.vec2(a.value, b.value)
    magnitude[i] = wp.vec2(a.magnitude, b.magnitude)


class TestConvexCells(unittest.TestCase):
    def test_direction_range_cpu(self):
        """Keep nonfinite/zero/extreme directions on the original support path."""
        rays = np.array(
            [
                [0, 0, 0],
                [np.inf, 0, 0],
                [np.nan, 1, 0],
                [2**-61, 0, 0],
                [2**61, 0, 0],
                [2**-60, 0, 0],
                [2**60, 0, 0],
                [1, 1e-40, 0],
            ],
            np.float32,
        )
        output = wp.zeros(len(rays), dtype=int, device="cpu")
        wp.launch(cell_check, len(rays), inputs=[wp.array(rays, dtype=wp.vec3, device="cpu"), output], device="cpu")
        np.testing.assert_array_equal(output.numpy()[:5], -np.ones(5, dtype=int))
        self.assertTrue(np.all(output.numpy()[5:] >= 0))

    def test_owner_api(self):
        """Require the distinct default-off cell support owner."""
        from newton._src.geometry.convex_cells import _CellOwner  # noqa: PLC0415

        self.assertTrue(callable(_CellOwner))

    def check_support_and_cells(self, device):
        """Check actual support, feature law, fallback and immutable table ownership."""
        rays = np.concatenate(
            [
                np.random.default_rng(31).normal(size=(256, 3)),
                np.eye(3),
                -np.eye(3),
                np.zeros((1, 3)),
                [[1, 1e-8, 1e-8], [1, 1e-40, 0], [0, 0, 1e30]],
            ]
        ).astype(np.float32)
        owner, mesh = owner_fixture(device, len(rays))
        self.assertEqual(len(owner.metadata["admitted_meshes"]), 1)
        shape = GenericShapeData()
        shape.scale = wp.vec3(1.0, 0.7, -0.9)
        shape.auxiliary = wp.vec3(
            float(mesh.id & 0x3FFFFF), float((mesh.id >> 22) & 0x3FFFFF), float((mesh.id >> 44) & 0xFFFFF)
        )
        rr = wp.array(rays, dtype=wp.vec3, device=device)
        old, new = [wp.zeros(len(rays), dtype=wp.vec3, device=device) for _ in range(2)]
        paths = wp.zeros(len(rays), dtype=int, device=device)
        feature, magnitude = [wp.zeros(len(rays), dtype=wp.vec2, device=device) for _ in range(2)]
        inputs = [owner.data, shape, rr, old, new, paths, feature, magnitude]
        for kind in (newton.GeoType.CONVEX_MESH, newton.GeoType.BOX, newton.GeoType.SPHERE):
            shape.shape_type = int(kind)
            shape.scale = wp.vec3(1.0, 0.7, -0.9 if kind == newton.GeoType.CONVEX_MESH else 0.9)
            wp.launch(support_check, len(rays), inputs=inputs, device=device)
            np.testing.assert_allclose(new.numpy(), old.numpy(), rtol=0, atol=0)
            np.testing.assert_array_equal(feature.numpy()[:, 0], feature.numpy()[:, 1])
            self.assertTrue(np.all(magnitude.numpy()[:, 1] >= magnitude.numpy()[:, 0]))
            if kind == newton.GeoType.CONVEX_MESH:
                self.assertGreater(np.count_nonzero(paths.numpy() >= 0), 200)
                self.assertGreater(np.count_nonzero(paths.numpy() < 0), 0)
        shape.shape_type = int(newton.GeoType.CONVEX_MESH)
        shape.scale = wp.vec3(1.0)
        owner.invalidate()
        mesh.points.assign(VERTICES * np.array([1.1, 1, 1], np.float32))
        wp.launch(support_check, len(rays), inputs=inputs, device=device)
        np.testing.assert_array_equal(new.numpy(), old.numpy())
        self.assertTrue(np.all(paths.numpy() == -1))

    def test_support_and_cells_cpu(self):
        """Exercise cell support/fallback and cell boundary semantics on CPU."""
        self.check_support_and_cells("cpu")

    def test_invalid_hull_and_mode(self):
        """Withdraw malformed hulls and reject a partial four-owner activation."""
        owner, _ = owner_fixture("cpu", faces=FACES[:-1])
        self.assertEqual(owner.metadata["admitted_meshes"], [])
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize(device="cpu")
        with patch.dict(
            os.environ, {"NEWTON_NARROW_PHASE_CONVEX_CELLS": "1", "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0"}
        ):
            with self.assertRaisesRegex(ValueError, "complete rejection-only"):
                newton.CollisionPipeline(model)
        changed, baseline = source_successor(_create_rejection_query_kernels, coherent=True)
        self.assertIn("_cell_plane_lower_bound", changed)
        self.assertNotIn("words_a", changed)
        self.assertNotIn("_cell_support_feature", baseline)

    def check_complete_queries(self, device):
        """Run all four original-law query owners across source/epoch/reset and empty epochs."""
        n = 9
        block = 1 if wp.get_device(device).is_cpu else 128
        threads = n if block == 1 else 128
        pairs, pair_array, values, inputs = _box_inputs(n, device)
        _, mesh = owner_fixture(device)
        types = values[0].numpy()
        types[::2] = int(newton.GeoType.CONVEX_MESH)
        values[0].assign(types)
        sources = np.zeros(2 * n, np.uint64)
        sources[::2] = mesh.id
        values[3].assign(sources)
        sizes = values[1].numpy()
        sizes[::2, :3] = 2.0
        values[1].assign(sizes)
        cache = _ConvexQueryCache(
            pairs=pairs,
            shape_types=types,
            shape_world=np.zeros(2 * n, np.int32),
            world_count=1,
            query_capacity=n,
            device=device,
        )
        model = SimpleNamespace(
            _mesh_keep_alive=[mesh], shape_source_ptr=values[3], shape_type=values[0], device=wp.get_device(device)
        )
        owner = _CellOwner(model)
        narrow = SimpleNamespace(_use_lean_gjk_mpr=True, external_aabb=True, _convex_writer_func=write_contact_simple)
        kernels = coherent_kernels(diagnostics=True)
        manifold = manifold_kernel(narrow)
        baseline = _original()
        count = wp.array([n], dtype=int, device=device)
        expected, actual = _buffers(n, device), _buffers(n, device)
        writer = ContactWriterData()
        writer.contact_max = n * 8
        writer.contact_count = wp.zeros(1, dtype=int, device=device)
        for name, dtype in [
            ("contact_pair", wp.vec2i),
            ("contact_position", wp.vec3),
            ("contact_normal", wp.vec3),
            ("contact_penetration", float),
            ("contact_tangent", wp.vec3),
            ("contact_sort_key", wp.int64),
        ]:
            setattr(writer, name, wp.zeros(n * 8, dtype=dtype, device=device))
        for case in (
            "cold",
            "current",
            "masked_reset",
            "source_changed",
            "generation_gap",
            "empty",
            "regrow",
            "invalidated",
        ):
            if case == "masked_reset":
                cache.reset(wp.array([True, False], dtype=wp.bool, device=device))
            if case == "source_changed":
                cache.data.source_a.fill_(123)
            if case == "generation_gap":
                cache.begin()
            if case == "empty":
                count.zero_()
            if case == "regrow":
                count.fill_(n)
            if case == "invalidated":
                owner.invalidate()
                mesh.points.assign(VERTICES * np.array([1.1, 1, 1], np.float32))
            for b in [*expected, *actual]:
                b.zero_()
            wp.launch(
                baseline[0],
                threads,
                inputs=[pair_array, count, *values, threads, *expected],
                device=device,
                block_dim=block,
            )
            wp.launch(
                baseline[1], threads, inputs=[pair_array, *values, threads, *expected], device=device, block_dim=block
            )
            cache.begin()
            for kernel in kernels:
                wp.launch(
                    kernel,
                    threads,
                    inputs=[pair_array, count, inputs, cache.data, threads, *actual, owner.data],
                    device=device,
                    block_dim=block,
                )
            ids, got = _accepted(actual)
            refids, ref = _accepted(expected)
            np.testing.assert_array_equal(np.sort(ids), np.sort(refids))
            np.testing.assert_allclose(got[ids]["signed_distance"], ref[ids]["signed_distance"], rtol=0, atol=5e-5)
            writer.contact_count.zero_()
            wp.launch(
                manifold,
                threads,
                inputs=[pair_array, *values, writer, threads, actual[0], actual[3], actual[4], owner.data],
                device=device,
                block_dim=block,
            )
            num = int(writer.contact_count.numpy()[0])
            self.assertLessEqual(num, n * 8)
            if case == "empty":
                self.assertEqual(num, 0)
            else:
                self.assertGreater(num, 0)
                self.assertTrue(np.isfinite(writer.contact_position.numpy()[:num]).all())
                np.testing.assert_allclose(
                    writer.contact_normal.numpy()[:num], np.tile([1, 0, 0], (num, 1)), rtol=0, atol=1e-3
                )

    def test_complete_queries_cpu(self):
        """Compile and check the complete classifier/MPR/GJK/manifold path on CPU."""
        self.check_complete_queries("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Root-owned paired GPU lease required")
    def test_support_and_cells_cuda(self):
        """Check native certificates, primitive policies and cell coverage on the leased GPU."""
        self.check_support_and_cells("cuda:0")

    def check_actual_cooked_hulls(self, device):
        """Exercise actual finalized Allegro hulls using random and supporting-face directions."""
        asset = Path(
            "/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd"
        )
        self.assertTrue(asset.is_file())
        self.assertEqual(
            hashlib.sha256(asset.read_bytes()).hexdigest(),
            "edc12a991d080730f78da83ea6ef0363e6dc997bff5861449e582e3e336a5bc8",
        )
        builder = newton.ModelBuilder()
        builder.add_usd(str(asset), load_visual_shapes=False)
        model = builder.finalize(device=device)
        owner = _CellOwner(model)
        self.assertEqual(len(owner.metadata["admitted_meshes"]), 10)
        self.assertEqual(owner.metadata["unsupported_meshes"], [])
        certified = 0
        for mesh in model._mesh_keep_alive:
            vertices = mesh.points.numpy()
            faces = mesh.indices.numpy().reshape(-1, 3)
            a, b, c = [vertices[faces[:, i]].astype(float) for i in range(3)]
            rays = np.concatenate(
                [
                    np.random.default_rng(41).normal(size=(256, 3)),
                    np.cross(b - a, c - a),
                    cell_boundary_rays() / np.array([0.7, -1.3, 0.9], np.float32),
                    np.zeros((1, 3)),
                ]
            ).astype(np.float32)

            shape = GenericShapeData()
            shape.shape_type = int(newton.GeoType.CONVEX_MESH)
            shape.scale = wp.vec3(0.7, -1.3, 0.9)
            shape.auxiliary = wp.vec3(
                float(mesh.id & 0x3FFFFF), float((mesh.id >> 22) & 0x3FFFFF), float((mesh.id >> 44) & 0xFFFFF)
            )
            outputs = [
                wp.zeros(len(rays), dtype=dtype, device=device) for dtype in (wp.vec3, wp.vec3, int, wp.vec2, wp.vec2)
            ]
            wp.launch(
                support_check,
                len(rays),
                inputs=[owner.data, shape, wp.array(rays, dtype=wp.vec3, device=device), *outputs],
                device=device,
            )
            old, candidate, paths, feature, magnitude = [out.numpy() for out in outputs]
            np.testing.assert_array_equal(candidate, old)
            np.testing.assert_array_equal(feature[:, 0], feature[:, 1])
            self.assertTrue(np.all(magnitude[:, 1] >= magnitude[:, 0]))
            certified += int(np.count_nonzero(paths >= 0))
        self.assertGreater(certified, 2000)
        self.assertEqual(owner.metadata["mask_bytes"], 122880)
        self.assertEqual(owner.metadata["pair_hint_bytes"], 0)
        print(
            "CELLS_ACTUAL_HULLS",
            {
                "meshes": 10,
                "certified": certified,
                "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "descriptor_bytes": owner.metadata["descriptor_bytes"],
            },
            flush=True,
        )

    def test_actual_cooked_hulls_cpu(self):
        """Validate support on all ten actual cooked Allegro hulls with CUDA hidden."""
        self.check_actual_cooked_hulls("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Root-owned paired GPU lease required")
    def test_actual_cooked_hulls_cuda(self):
        """Check the actual admitted cooked hulls and ambiguous face directions on CUDA."""
        self.check_actual_cooked_hulls("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Root-owned paired GPU lease required")
    def test_complete_queries_cuda(self):
        """Exercise all four native owners including every stale/empty/invalidation transition."""
        self.check_complete_queries("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Root-owned paired GPU lease required")
    def test_production_dispatch_graph_cuda(self):
        """Select the actual complete owner and retain current/reset/invalidation graph behavior."""
        device = "cuda:0"
        world = newton.ModelBuilder()
        hull = newton.Mesh(VERTICES, FACES.reshape(-1))
        for i in range(167):
            x = 0.8 if i == 1 else 10.0 * i
            body = world.add_body(xform=wp.transform(wp.vec3(x, 0, 0)))
            if i == 0:
                world.add_shape_convex_hull(body, mesh=hull, scale=wp.vec3(2.0))
            else:
                world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(-10, 0, 0)), hx=0.5, hy=0.5, hz=0.5)
        builder.add_world(world)
        builder.add_world(world, xform=wp.transform(wp.vec3(4000, 0, 0)))
        model = builder.finalize(device=device)
        state = model.state()
        flags = {
            "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "reject_only",
            "NEWTON_NARROW_PHASE_COHERENT_STATS": "0",
            "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": "0",
        }
        with patch.dict(os.environ, {**flags, "NEWTON_NARROW_PHASE_CONVEX_CELLS": "0"}):
            baseline = newton.CollisionPipeline(model, broad_phase="explicit")
        with patch.dict(os.environ, {**flags, "NEWTON_NARROW_PHASE_CONVEX_CELLS": "1"}):
            candidate = newton.CollisionPipeline(model, broad_phase="explicit")
        self.assertIsNotNone(candidate._convex_cells)
        self.assertIs(candidate.narrow_phase._cells_owner, candidate._convex_cells)
        self.assertIsNone(baseline._convex_cells)
        self.assertTrue(all("cells" in k.key for k in candidate.narrow_phase._coherent_query_kernels))
        self.assertIn("cells", candidate.narrow_phase.narrow_phase_manifold_kernel.key)
        left, right = baseline.contacts(), candidate.contacts()
        distance = [wp.zeros(c.rigid_contact_max, dtype=float, device=device) for c in (left, right)]

        def compare():
            baseline.collide(state, left)
            baseline.narrow_phase.check_buffer_capacity()
            candidate.narrow_phase.check_buffer_capacity()
            snapshots = []
            for contacts, out in zip((left, right), distance, strict=True):
                n = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(n, 0)
                newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=out)
                values = out.numpy()[:n]
                self.assertTrue(np.isfinite(values).all())
                snapshots.append(float(values.min()))
            self.assertAlmostEqual(*snapshots, delta=2e-4)

        candidate.collide(state, right)
        compare()
        poses = state.body_q.numpy().copy()
        with wp.ScopedCapture(device=device) as captured:
            candidate.collide(state, right)
        for moved in (True, False):
            changed = poses.copy()
            if moved:
                changed[[1, 168], 0] += 0.005
            state.body_q.assign(changed)
            wp.capture_launch(captured.graph)
            compare()
        candidate.reset_contact_matching(wp.array([True, False, False], dtype=wp.bool, device=device))
        wp.capture_launch(captured.graph)
        compare()
        candidate.invalidate_convex_cells()
        self.assertEqual(int(candidate._convex_cells.data.valid.numpy()[0]), 0)
        wp.capture_launch(captured.graph)
        compare()


if __name__ == "__main__":
    unittest.main()

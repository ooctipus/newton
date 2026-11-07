# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve real contacts when convex hulls have offset authoring origins."""

import importlib
import itertools
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.narrow_phase import create_prepare_convex_pair
from newton.tests.test_coherent_convex_rejection import _box_inputs

_prepare = create_prepare_convex_pair(True)


@wp.kernel(enable_backward=False)
def _admit(
    pair: wp.array[wp.vec2i],
    types: wp.array[int],
    data: wp.array[wp.vec4],
    transforms: wp.array[wp.transform],
    sources: wp.array[wp.uint64],
    gaps: wp.array[float],
    radii: wp.array[float],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    local_lower: wp.array[wp.vec3],
    local_upper: wp.array[wp.vec3],
    output: wp.array[int],
):
    valid, _query = _prepare(
        pair[0], types, data, transforms, sources, gaps, radii, lower, upper, local_lower, local_upper
    )
    output[0] = int(valid)


def check_offset_bounds(test, device):
    """Keep overlapping boxes independently of hull origin, scale and pair order."""
    center = np.array([2.0, -1.0, 0.5], np.float32)
    vertices = np.array(list(itertools.product((-0.05, 0.05), repeat=3)), np.float32) + center
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device), indices=wp.array([0, 1, 2], dtype=int, device=device)
    )
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.7)
    origin = wp.vec3(3.0, -4.0, 2.0)
    _, pairs, values, _ = _box_inputs(1, device)
    output = wp.zeros(1, dtype=int, device=device)
    for scale in ((1.0, 1.0, 1.0), (0.5, 2.0, 1.5), (-1.0, 0.5, 2.0)):
        scaled = vertices * np.asarray(scale)
        lo, hi = scaled.min(axis=0), scaled.max(axis=0)
        hull_center = 0.5 * (lo + hi)
        radius = 0.5 * np.linalg.norm(hi - lo)
        for separated in (False, True):
            # Overlapping aligned boxes have a known common interior volume.
            shift = np.array([2.0 if separated else 0.02, 0.0, 0.0])
            box_origin = origin + wp.quat_rotate(rotation, wp.vec3(*(hull_center + shift)))
            for reverse in (False, True):
                with test.subTest(scale=scale, separated=separated, reverse=reverse):
                    values[0].assign(np.array([int(newton.GeoType.CONVEX_MESH), int(newton.GeoType.BOX)], np.int32))
                    values[1].assign(np.array([[*scale, 0.0], [0.05, 0.05, 0.05, 0.0]], np.float32))
                    values[2].assign(
                        np.array(
                            [list(wp.transform(origin, rotation)), list(wp.transform(box_origin, rotation))], np.float32
                        )
                    )
                    values[3].assign(np.array([mesh.id, 0], np.uint64))
                    values[4].fill_(0.001)
                    values[5].assign(np.array([radius, np.sqrt(3) * 0.05], np.float32))
                    values[8].assign(np.array([lo, [-0.05] * 3], np.float32))
                    values[9].assign(np.array([hi, [0.05] * 3], np.float32))
                    pairs.assign(np.array([[1, 0] if reverse else [0, 1]], np.int32))
                    wp.launch(_admit, dim=1, inputs=[pairs, *values], outputs=[output], device=device)
                    test.assertEqual(int(output.numpy()[0]), int(not separated))


class TestSplitConvexBounds(unittest.TestCase):
    def test_offset_bounds_cpu(self):
        """Reject separation without discarding physically overlapping offset hulls."""
        check_offset_bounds(self, "cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_offset_bounds_cuda(self):
        """Preserve the same physical sphere contract in device pair preparation."""
        check_offset_bounds(self, "cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_offset_contacts_graph_and_rejection(self):
        """Retain real offset-hull contacts through split, rejection and graph replay."""
        vertices = np.array(list(itertools.product((-0.05, 0.05), repeat=3)), np.float32)
        vertices += np.array([2.0, -1.0, 0.5], np.float32)
        faces = np.array(
            [
                0,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                5,
                7,
                6,
                0,
                1,
                4,
                1,
                5,
                4,
                2,
                6,
                3,
                3,
                6,
                7,
                0,
                4,
                2,
                2,
                4,
                6,
                1,
                3,
                5,
                3,
                7,
                5,
            ],
            np.int32,
        )
        mesh = newton.Mesh(vertices, faces)
        builder = newton.ModelBuilder()
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.7)
        scale = (-1.0, 0.5, 2.0)
        center = np.array([2.0, -1.0, 0.5]) * scale
        origin = wp.vec3(3.0, -4.0, 2.0)
        body_a = builder.add_body(xform=wp.transform(origin, rotation))
        builder.add_shape_convex_hull(body_a, mesh=mesh, scale=scale)
        body_b = builder.add_body(
            xform=wp.transform(
                origin + wp.quat_rotate(rotation, wp.vec3(*(center + np.array([0.02, 0.0, 0.0])))), rotation
            )
        )
        builder.add_shape_box(body_b, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device="cuda:0")
        state = model.state()
        collision = importlib.import_module("newton._src.sim.collide")
        pipelines = []
        with (
            patch.object(collision, "_SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0),
            patch.object(collision, "_SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD", 0),
        ):
            for method in ("0", "0", "reject_only"):
                with patch.dict(
                    os.environ,
                    {"NEWTON_NARROW_PHASE_COHERENT_CONVEX": method, "NEWTON_NARROW_PHASE_COHERENT_STATS": "0"},
                ):
                    pipelines.append(
                        newton.CollisionPipeline(
                            model, broad_phase="explicit", deterministic=True, rigid_contact_max=64
                        )
                    )
        self.assertTrue(all(p.narrow_phase.split_gjk_mpr for p in pipelines))
        pipelines[0].narrow_phase.split_gjk_mpr = False
        self.assertIsNotNone(pipelines[2].narrow_phase._coherent_cache)
        contacts = [p.contacts() for p in pipelines]

        def collide():
            for pipeline, contact in zip(pipelines, contacts, strict=True):
                pipeline.collide(state, contact)

        collide()
        wp.synchronize()
        with wp.ScopedCapture(device="cuda:0") as capture:
            collide()
        original = state.body_q.numpy()
        for separated in (False, True, False):
            poses = original.copy()
            if separated:
                poses[body_b, 0] += 5.0
            state.body_q.assign(poses)
            for _ in range(2):
                wp.capture_launch(capture.graph)
                counts = [int(c.rigid_contact_count.numpy()[0]) for c in contacts]
                if separated:
                    self.assertEqual(counts, [0, 0, 0])
                else:
                    self.assertGreater(counts[0], 0)
                    self.assertEqual(counts, [counts[0]] * 3)
                    for field in ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1"):
                        values = [getattr(c, "rigid_contact_" + field).numpy()[: counts[0]] for c in contacts]
                        for value in values[1:]:
                            np.testing.assert_allclose(value, values[0], atol=2e-5, rtol=1e-5)
                self.assertTrue(all(not any(p.narrow_phase.buffer_capacity_status().values()) for p in pipelines))
        wp.synchronize()


if __name__ == "__main__":
    unittest.main()

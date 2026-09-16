# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise finite-shell spatial support without changing the collision envelope."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import contact_reduction_global as reduction
from newton._src.geometry import heightfield_finite as finite
from newton._src.geometry.types import GeoType


def reduce_points(
    points,
    *,
    candidate=True,
    deterministic=False,
    kind=GeoType.HFIELD,
    other=GeoType.BOX,
    extent=1.0,
    gap=0.02,
    source_changed=False,
    device="cpu",
    oct_normals=None,
):
    """Run the real buffered reducer and return its per-bin winning IDs."""
    reducer = reduction.GlobalContactReducer(capacity=64, device=device, deterministic=deterministic)
    count = len(points)
    pd = np.zeros((65, 4), dtype=np.float32)
    pd[1 : count + 1] = points
    pairs = np.zeros((65, 2), dtype=np.int32)
    pairs[1 : count + 1, 1] = 1
    reducer.position_depth.assign(pd)
    if oct_normals is not None:
        normals = np.zeros((65, 2), dtype=np.float32)
        normals[1 : count + 1] = oct_normals
        reducer.normal.assign(normals)
    reducer.shape_pairs.assign(pairs)
    reducer.contact_fingerprints.assign(np.arange(65, dtype=np.int32))
    reducer.contact_count.assign(np.array([count], dtype=np.int32))
    transform = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device)
    lower = wp.array([(-extent, -extent, -extent)] * 2, dtype=wp.vec3, device=device)
    upper = wp.array([(extent, extent, extent)] * 2, dtype=wp.vec3, device=device)
    voxels = wp.array([(1, 1, 1)] * 2, dtype=wp.vec3i, device=device)
    kernel = reduction.reduce_buffered_contacts_kernel
    inputs = [reducer.get_data_struct(), transform, lower, upper, voxels, 64]
    if candidate and hasattr(reduction, "reduce_heightfield_shell_contacts_kernel"):
        kernel = reduction.reduce_heightfield_shell_contacts_kernel
        inputs = [
            reducer.get_data_struct(),
            wp.array([kind, other], dtype=int, device=device),
            wp.array([(1.0, 1.0, 1.0, 0.0025)] * 2, dtype=wp.vec4, device=device),
            wp.array([gap, gap], dtype=float, device=device),
            wp.array([0, int(source_changed)], dtype=wp.uint64, device=device),
            wp.array(np.ones((2, 2, 3), dtype=np.float32), dtype=wp.vec3, ndim=2, device=device),
            wp.zeros(2, dtype=wp.uint64, device=device),
            transform,
            lower,
            upper,
            voxels,
            64,
        ]
    wp.launch(kernel, 64, inputs=inputs, device=device)
    keys = reducer.hashtable.keys.numpy()
    active = reducer.hashtable.active_slots.numpy()
    capacity = reducer.hashtable.capacity
    values = reducer.ht_values.numpy().reshape(reducer.values_per_key, capacity)
    mask = (1 << 20) - 1 if deterministic else 0xFFFFFFFF
    result = {}
    for entry in active[: active[capacity]]:
        result[(int(keys[entry]) >> 55) & 255] = tuple(int(x) & mask for x in values[:, entry])
    return result


def spatial_ids(result):
    """Extract directional winners without counting depth or voxel winners."""
    return {
        value
        for key, values in result.items()
        if key < int(reduction.NUM_NORMAL_BINS)
        for value in values[:-1]
        if value
    }


class TestHeightfieldShellSupport(unittest.TestCase):
    def test_positive_shell_keeps_opposed_support(self):
        """Retain both sides of a separated face instead of only depth ties."""
        points = [(x, y, 0.015, 0.03) for x in (-0.1, 0.1) for y in (-0.08, 0.08)]
        for deterministic in (False, True):
            self.assertEqual(spatial_ids(reduce_points(points, candidate=False, deterministic=deterministic)), set())
            ids = spatial_ids(reduce_points(points, deterministic=deterministic))
            self.assertTrue(ids, "Positive-shell manifold has no directional support")
            selected = np.array([points[index - 1] for index in ids])
            np.testing.assert_allclose(np.ptp(selected[:, :2], axis=0), [0.2, 0.16], atol=1e-7)

    def test_inner_priority_preserves_depth_and_voxels(self):
        """Keep every old inner extreme ahead of distant outer witnesses."""
        points = [(x, y, 0.0, -0.01) for x in (-0.1, 0.1) for y in (-0.08, 0.08)]
        points += [(x, y, 0.015, 0.03) for x in (-0.9, 0.9) for y in (-0.9, 0.9)]
        for deterministic in (False, True):
            original = reduce_points(points, candidate=False, deterministic=deterministic)
            current = reduce_points(points, deterministic=deterministic)
            self.assertTrue(spatial_ids(current))
            self.assertLessEqual(max(spatial_ids(current)), 4)
            for key, values in original.items():
                if key < int(reduction.NUM_NORMAL_BINS):
                    self.assertEqual(current[key][-1], values[-1])
                else:
                    self.assertEqual(current[key], values)

    def test_scaled_beta_and_finite_envelope(self):
        """Separate live finite-envelope eligibility from the legacy inner tier."""
        # beta * diagonal = 0.0003464: first witness is inner even though d>beta.
        points = [(0.0, 0.0, 0.0, 0.0003), (0.8, 0.8, 0.0, 0.0004)]
        self.assertEqual(spatial_ids(reduce_points(points)), {1})
        # Huge AABB makes both legacy-inner, but a discarded far witness may
        # not suppress the only contact within the unchanged gap + margins.
        points = [(0.0, 0.0, 0.0, 0.03), (0.8, 0.8, 0.0, 0.06)]
        self.assertEqual(spatial_ids(reduce_points(points, extent=1000.0)), {1})
        self.assertEqual(spatial_ids(reduce_points(points, gap=0.001)), set())

    def test_normal_bins_keep_independent_support(self):
        """Do not suppress an outer normal bin because another normal penetrates."""
        points = [(0.0, 0.0, 0.0, -0.01), (0.1, 0.2, 0.015, 0.03)]
        self.assertEqual(spatial_ids(reduce_points(points, oct_normals=[(0.0, 0.0), (1.0, 0.0)])), {1, 2})

    @unittest.skipUnless(wp.is_cuda_available(), "Requires root-owned CUDA lease")
    def test_native_mixed_inner_outer_cuda(self):
        """Exercise native atomic priority, finite eligibility and separate normal bins."""
        points = [(x, y, 0.0, -0.01) for x in (-0.1, 0.1) for y in (-0.08, 0.08)]
        points += [(x, y, 0.015, 0.03) for x in (-0.9, 0.9) for y in (-0.9, 0.9)]
        for deterministic in (False, True):
            ids = spatial_ids(reduce_points(points, deterministic=deterministic, device="cuda:0"))
            self.assertTrue(ids)
            self.assertLessEqual(max(ids), 4)
            far = [(0.0, 0.0, 0.0, 0.03), (0.8, 0.8, 0.0, 0.06)]
            self.assertEqual(
                spatial_ids(reduce_points(far, extent=1000.0, deterministic=deterministic, device="cuda:0")), {1}
            )
            mixed = [(0.0, 0.0, 0.0, -0.01), (0.1, 0.2, 0.015, 0.03)]
            self.assertEqual(
                spatial_ids(
                    reduce_points(
                        mixed, oct_normals=[(0.0, 0.0), (1.0, 0.0)], deterministic=deterministic, device="cuda:0"
                    )
                ),
                {1, 2},
            )

    def test_non_heightfield_keeps_original_encoding(self):
        """Leave unrelated mesh spatial, deepest and voxel competition unchanged."""
        points = [(0.1, 0.2, 0.0, -0.01), (-0.8, -0.8, 0.0, 0.03)]
        for deterministic in (False, True):
            self.assertEqual(
                reduce_points(points, kind=GeoType.MESH, deterministic=deterministic),
                reduce_points(points, candidate=False, kind=GeoType.MESH, deterministic=deterministic),
            )
            for other in (GeoType.SPHERE, GeoType.CAPSULE):
                self.assertEqual(
                    reduce_points(points, other=other, deterministic=deterministic),
                    reduce_points(points, candidate=False, other=other, deterministic=deterministic),
                )
            self.assertEqual(
                reduce_points(points, other=GeoType.CONVEX_MESH, source_changed=True, deterministic=deterministic),
                reduce_points(points, candidate=False, other=GeoType.CONVEX_MESH, deterministic=deterministic),
            )

    def test_paired_dispatch_only(self):
        """Enable the analytical shell only alongside its matching buffered reducer."""
        builder = newton.ModelBuilder()
        builder.add_shape_heightfield(
            heightfield=newton.Heightfield(np.zeros((3, 3), dtype=np.float32), nrow=3, ncol=3, hx=1.0, hy=1.0)
        )
        body = builder.add_body()
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        for reduce in (False, True):
            with patch.dict(os.environ, NEWTON_HEIGHTFIELD_FINITE_QUERY="1", NEWTON_HEIGHTFIELD_SHELL_SUPPORT="1"):
                pipeline = newton.CollisionPipeline(model, reduce_contacts=reduce, rigid_contact_max=64)
            narrow = pipeline.narrow_phase
            self.assertEqual(getattr(narrow, "_heightfield_shell_support", False), reduce)
            self.assertIs(narrow._finite_direct, finite.create_query_kernel(narrow._convex_writer_func))
            self.assertIn(narrow._finite_reducer.key, narrow._finite_reducer.module.kernels)
            self.assertEqual("shell" in narrow._finite_reducer.key, reduce)


if __name__ == "__main__":
    unittest.main()

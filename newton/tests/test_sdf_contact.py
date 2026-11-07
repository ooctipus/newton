# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_contact import (
    EDGE_CLUSTER_SIZE,
    _create_get_mesh_edge_bounding_sphere_func,
    _sdf_rsqrt_rn,
    build_mesh_edge_clusters,
    compute_block_counts_from_weights,
    mesh_sdf_contact_search_precision,
    texture_sdf_cluster_may_contact,
)
from newton._src.geometry.sdf_texture import TextureSDFData, _texture_sample_sdf_hw_clamped
from newton.tests.unittest_utils import get_cuda_test_devices, get_test_devices

_edge_bounding_sphere = _create_get_mesh_edge_bounding_sphere_func(True)


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_search_precision_kernel(out: wp.array[wp.float32]):
    out[0] = mesh_sdf_contact_search_precision(0.0, 1.0, 0.001, True)
    out[1] = mesh_sdf_contact_search_precision(0.01, 1.0, 0.001, True)
    out[2] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.1, True)
    out[3] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.001, False)


@wp.kernel(enable_backward=False)
def _sdf_rsqrt_rn_kernel(values: wp.array[wp.float32], out: wp.array[wp.float32]):
    tid = wp.tid()
    out[tid] = _sdf_rsqrt_rn(values[tid])


@wp.kernel(enable_backward=False)
def _cluster_certificate_kernel(
    sdf_table: wp.array[TextureSDFData],
    sdf_index: int,
    edge_indices: wp.array[wp.vec2i],
    edge_centers: wp.array[wp.vec4],
    edge_order: wp.array[wp.int32],
    edge_clusters: wp.array[wp.vec4],
    edge_reach: wp.array[wp.float32],
    edge_range: wp.vec2i,
    transforms: wp.array[wp.transform],
    contact_threshold: float,
    cluster_may_contact: wp.array2d[wp.int32],
    edge_passes: wp.array2d[wp.int32],
):
    """Cluster cull and, for each member, the mesh-SDF kernel's texture edge cull."""
    i, q = wp.tid()
    num_edges = edge_range[1]
    if q * EDGE_CLUSTER_SIZE >= num_edges:
        return
    sdf = sdf_table[sdf_index]
    X = transforms[i]
    unit = wp.vec3(1.0, 1.0, 1.0)
    bounds = edge_clusters[edge_range[0] + q]
    may = texture_sdf_cluster_may_contact(
        sdf,
        wp.vec3(bounds[0], bounds[1], bounds[2]),
        bounds[3],
        edge_reach[edge_range[0] + q],
        X,
        unit,
        1.0,
        contact_threshold,
    )
    cluster_may_contact[i, q] = wp.where(may, 1, 0)
    for lane in range(EDGE_CLUSTER_SIZE):
        slot = q * EDGE_CLUSTER_SIZE + lane
        if slot < num_edges:
            edge = edge_order[edge_range[0] + slot]
            center, radius = _edge_bounding_sphere(
                wp.uint64(0), edge_indices, edge_centers, edge_range, unit, X, unit, 1.0, edge
            )
            threshold = radius + contact_threshold
            clamped = wp.min(wp.max(center, sdf.sdf_box_lower), sdf.sdf_box_upper)
            aabb_dist_sq = wp.length_sq(center - clamped)
            passes = 0
            if aabb_dist_sq <= threshold * threshold:
                diff_mag = float(0.0)
                if aabb_dist_sq > 0.0:
                    diff_mag = wp.sqrt(aabb_dist_sq)
                if _texture_sample_sdf_hw_clamped(sdf, clamped, diff_mag) <= threshold:
                    passes = 1
            edge_passes[i, edge] = passes


class TestSDFContact(unittest.TestCase):
    def test_edge_clusters_partition_edges(self) -> None:
        """Clusters hold every edge once, all full but the last, inside their rounded-up spread."""
        rng = np.random.default_rng(3)
        for count in (1, EDGE_CLUSTER_SIZE, 3 * EDGE_CLUSTER_SIZE + 5, 1001):
            centers = rng.normal(size=(count, 3)).astype(np.float32)
            radii = rng.uniform(0.0, 0.1, count).astype(np.float32)
            order, bounds, reach = build_mesh_edge_clusters(centers, radii)
            np.testing.assert_array_equal(np.sort(order), np.arange(count))
            self.assertEqual(len(bounds), -(-count // EDGE_CLUSTER_SIZE))
            for q in range(len(bounds)):
                members = order[q * EDGE_CLUSTER_SIZE : (q + 1) * EDGE_CLUSTER_SIZE]
                if q < len(bounds) - 1:
                    self.assertEqual(len(members), EDGE_CLUSTER_SIZE)
                offsets = centers[members].astype(np.float64) - bounds[q, :3].astype(np.float64)
                self.assertLessEqual(np.linalg.norm(offsets, axis=1).max(), float(bounds[q, 3]))
                self.assertEqual(reach[q], radii[members].max())

    def test_cluster_cull_never_drops_an_edge_that_passes(self) -> None:
        """A cluster the SDF cell bounds reject has no member that passes the texture edge cull."""
        devices = get_cuda_test_devices()
        if not devices:
            self.skipTest("texture SDFs need CUDA")
        rng = np.random.default_rng(7)
        for device in devices:
            with self.subTest(device=device):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                meshes = (
                    newton.Mesh.create_cylinder(0.012, 0.004, segments=48),
                    newton.Mesh.create_box(0.010, 0.006, 0.003),
                )
                for k, mesh in enumerate(meshes):
                    mesh.build_sdf(
                        device=device, target_voxel_size=0.0002, narrow_band_range=(-0.002, 0.002), margin=0.001
                    )
                    body = builder.add_body(xform=wp.transform(wp.vec3(0.1 * k, 0.0, 0.0), wp.quat_identity()))
                    builder.add_shape_mesh(body, mesh=mesh)
                model = builder.finalize(device=device)
                edge_range = wp.vec2i(*model._shape_edge_range.numpy()[0])
                sdf_index = int(model._shape_sdf_index.numpy()[1])
                num_edges = int(edge_range[1])
                num_clusters = -(-num_edges // EDGE_CLUSTER_SIZE)
                self.assertGreater(num_clusters, 4)

                # Poses of the cylinder in the box's frame, from overlapping to a few sizes apart.
                count = 256
                axes = rng.normal(size=(count, 3))
                axes /= np.linalg.norm(axes, axis=1, keepdims=True)
                angles = rng.uniform(-np.pi, np.pi, count)
                directions = rng.normal(size=(count, 3))
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                distances = rng.uniform(0.0, 0.04, count)
                transforms = wp.array(
                    [
                        wp.transform(wp.vec3(*(d * u)), wp.quat_from_axis_angle(wp.vec3(*a), float(t)))
                        for d, u, a, t in zip(distances, directions, axes, angles, strict=True)
                    ],
                    dtype=wp.transform,
                    device=device,
                )
                for contact_threshold in (0.0, 0.0006, 0.003):
                    may = wp.full((count, num_clusters), -1, dtype=wp.int32, device=device)
                    passes = wp.full((count, num_edges), -1, dtype=wp.int32, device=device)
                    wp.launch(
                        _cluster_certificate_kernel,
                        dim=(count, num_clusters),
                        inputs=[
                            model._texture_sdf_data,
                            sdf_index,
                            model._mesh_edge_indices,
                            model._mesh_edge_centers,
                            model._mesh_edge_order,
                            model._mesh_edge_clusters,
                            model._mesh_edge_cluster_reach,
                            edge_range,
                            transforms,
                            contact_threshold,
                            may,
                        ],
                        outputs=[passes],
                        device=device,
                    )
                    may_np, passes_np = may.numpy(), passes.numpy()
                    order = model._mesh_edge_order.numpy()[edge_range[0] : edge_range[0] + num_edges]
                    self.assertTrue(np.all(may_np >= 0) and np.all(passes_np >= 0))
                    member_passes = np.zeros((count, num_clusters), dtype=bool)
                    for q in range(num_clusters):
                        members = order[q * EDGE_CLUSTER_SIZE : (q + 1) * EDGE_CLUSTER_SIZE]
                        member_passes[:, q] = passes_np[:, members].any(axis=1)
                    dropped = (may_np == 0) & member_passes
                    self.assertFalse(np.any(dropped), f"{int(dropped.sum())} rejected clusters hold passing edges")
                    # Not vacuous: edges pass in some poses, and many clusters are rejected.
                    self.assertGreater(int(passes_np.sum()), 0)
                    self.assertGreater(float(np.mean(may_np == 0)), 0.25)

    def test_block_count_scan_ignores_inactive_tail(self) -> None:
        """Keep active block offsets independent of stale inactive slots."""
        for device in get_test_devices():
            with self.subTest(device=device):
                weights = wp.array([1024, 2048, 1024, 99, 99, 99, 99, 99], dtype=wp.int32, device=device)
                pair_count = wp.array([3], dtype=wp.int32, device=device)
                total_weight = wp.array([4096], dtype=wp.int32, device=device)
                block_counts = wp.full(8, 77, dtype=wp.int32, device=device)
                offsets = wp.empty_like(block_counts)

                wp.launch(
                    compute_block_counts_from_weights,
                    dim=2,
                    inputs=[total_weight, weights, pair_count, len(weights), 4, block_counts, 2],
                    device=device,
                )
                wp.utils.array_scan(block_counts, offsets, inclusive=False)
                np.testing.assert_array_equal(offsets.numpy()[:4], [0, 1, 3, 4])

                pair_count.assign([1])
                total_weight.assign([1024])
                wp.launch(
                    compute_block_counts_from_weights,
                    dim=2,
                    inputs=[total_weight, weights, pair_count, len(weights), 4, block_counts, 2],
                    device=device,
                )
                wp.utils.array_scan(block_counts, offsets, inclusive=False)
                np.testing.assert_array_equal(offsets.numpy()[:2], [0, 4])

    def test_mesh_sdf_contact_search_precision_uses_inner_envelope(self) -> None:
        device = wp.get_preferred_device()
        values = wp.empty(4, dtype=wp.float32, device=device)

        wp.launch(_mesh_sdf_contact_search_precision_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_allclose(values.numpy(), np.array([0.0, 0.001, 0.005, 0.005], dtype=np.float32))

    def test_sdf_rsqrt_rn_on_all_devices(self) -> None:
        """Verify the native reciprocal square root on CPU and CUDA."""
        values_np = np.array([0.25, 1.0, 2.0, 4.0, 100.0], dtype=np.float32)
        expected = np.float32(1.0) / np.sqrt(values_np)

        for device in get_test_devices():
            with self.subTest(device=device):
                values = wp.array(values_np, device=device)
                result = wp.empty_like(values)
                wp.launch(_sdf_rsqrt_rn_kernel, dim=len(values_np), inputs=[values, result], device=device)

                np.testing.assert_allclose(result.numpy(), expected, rtol=np.finfo(np.float32).eps, atol=0.0)


if __name__ == "__main__":
    unittest.main()

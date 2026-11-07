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
    _create_sdf_contact_funcs,
    _sdf_rsqrt_rn,
    build_mesh_edge_clusters,
    compute_block_counts_from_weights,
    mesh_sdf_contact_search_precision,
    texture_sdf_cluster_may_contact,
)
from newton._src.geometry.sdf_texture import TextureSDFData, _texture_sample_sdf_hw_clamped
from newton._src.utils.heightfield import HeightfieldData
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

_edge_bounding_sphere = _create_get_mesh_edge_bounding_sphere_func(True)


@wp.func
def _two_sphere_distance(sdf: TextureSDFData, p: wp.vec3) -> float:
    scale = sdf.voxel_radius
    return wp.min(wp.length(p) - 0.2 * scale, wp.length(p - wp.vec3(0.55 * scale, 0.0, 0.0)) - 0.05 * scale)


@wp.func
def _two_sphere_pair(sdf: TextureSDFData, p: wp.vec3, q: wp.vec3) -> wp.vec2:
    return wp.vec2(_two_sphere_distance(sdf, p), _two_sphere_distance(sdf, q))


def _make_nonconvex_edge_kernel(paired):
    search = _create_sdf_contact_funcs(False, paired, _two_sphere_distance, _two_sphere_pair)

    @wp.kernel(enable_backward=False)
    def kernel(
        scales: wp.array[float],
        distances: wp.array[float],
        endpoints: wp.array[int],
        boundary: wp.array[wp.uint8],
        elevation: wp.array[float],
    ):
        i = wp.tid()
        scale = scales[i // 2]
        a = wp.vec3(0.0)
        b = wp.vec3(scale, 0.0, 0.0)
        if i % 2 == 1:
            a, b = b, a
        sdf = TextureSDFData()
        sdf.voxel_radius = scale
        distance, _point, endpoint = search(
            sdf,
            wp.uint64(0),
            a,
            b,
            _two_sphere_distance(sdf, (a + b) * 0.5),
            False,
            0,
            boundary,
            -1,
            False,
            HeightfieldData(),
            elevation,
            0.0,
        )
        distances[i] = distance
        endpoints[i] = endpoint

    return kernel


_nonconvex_edge_scalar = _make_nonconvex_edge_kernel(False)
_nonconvex_edge_paired = _make_nonconvex_edge_kernel(True)


def test_mesh_mesh_contained_bvh_contacts(test, device):
    """Retain deep interior edges after bounded closest-point misses."""
    with wp.ScopedDevice(device):
        outer = newton.Mesh.create_box(1.0, 1.0, 1.0)
        inner = newton.Mesh.create_box(0.02, 0.02, 0.02)
        for scale in (0.01, 1.0, 100.0):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, margin=0.0, gap=0.0)
            # Exercise the scaled shape-local bounds under a world transform.
            rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.37)
            outer_pose = wp.transform(wp.vec3(3.0 * scale, -2.0 * scale, scale), rotation)
            inner_pose = wp.transform_multiply(outer_pose, wp.transform(wp.vec3(0.2 * scale, 0.0, 0.0)))
            builder.add_shape_mesh(-1, mesh=outer, scale=(scale, 2.0 * scale, 3.0 * scale), xform=outer_pose, cfg=cfg)
            body = builder.add_body(xform=inner_pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_shape_mesh(body, mesh=inner, scale=(scale, scale, scale), cfg=cfg)
            model = builder.finalize()
            state = model.state()
            for reduced in (False, True):
                with test.subTest(scale=scale, reduced=reduced):
                    pipeline = newton.CollisionPipeline(model, reduce_contacts=reduced)
                    contacts = pipeline.contacts()
                    state.body_q.assign([inner_pose])
                    pipeline.collide(state, contacts)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    test.assertGreater(count, 0)
                    # The +x surface is the nearest exit, about 0.8 scaled units away.
                    normals = contacts.rigid_contact_normal.numpy()[:count]
                    test.assertTrue(np.all(np.isfinite(normals)))
                    exit_direction = np.array(wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0)))
                    np.testing.assert_allclose(normals @ exit_direction, 1.0, atol=2.0e-5)
                    points0 = contacts.rigid_contact_point0.numpy()[:count]
                    points1 = np.array(
                        [
                            wp.transform_point(inner_pose, wp.vec3(p))
                            for p in contacts.rigid_contact_point1.numpy()[:count]
                        ]
                    )
                    depths = -np.sum((points1 - points0) * normals, axis=1) / scale
                    test.assertTrue(np.all((depths > 0.77) & (depths < 0.83)), depths)
                    # Reuse the pipeline after moving outside the shape bounds.
                    exterior = wp.transform_multiply(outer_pose, wp.transform(wp.vec3(4.0 * scale, 0.0, 0.0)))
                    state.body_q.assign([exterior])
                    pipeline.collide(state, contacts)
                    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


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


def test_mesh_mesh_sdf_surface_witnesses(test, device, box_sdf, *, probe_sdf=False):
    """Keep SDF-based mesh-mesh witnesses on both surfaces with reduction and speculation."""
    # Parallel faces give exact witness planes. A common shift of both witnesses
    # preserves their separation, so checking only contact distance misses the bug.
    cases = (
        ("direct", False, -0.1),
        ("predictive", False, 0.1),
        ("reduced", True, -0.1),
        ("reduced_predictive", True, 0.1),
    )
    with wp.ScopedDevice(device):
        if probe_sdf:
            # A closed probe supplies the second SDF and exercises texture-only kernels.
            probe = newton.Mesh.create_box(0.2, 0.2, 0.2)
            probe.build_sdf(max_resolution=32, device=device)
        else:
            probe = newton.Mesh(
                np.array([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
                np.array([0, 2, 1], dtype=np.int32),
                compute_inertia=False,
            )
        probe_half_height = 0.2 if probe_sdf else 0.0
        box = newton.Mesh.create_box(1.0, 1.0, 0.5)
        if box_sdf:
            box.build_sdf(max_resolution=32, device=device)
        # Allow texture quantization error, well below the original 0.05 m shift.
        sdf_tolerance = 1.0e-3 if box_sdf else 2.0e-5
        for probe_first in (True, False):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, margin=0.0, gap=0.0)
            for is_probe in (probe_first, not probe_first):
                builder.add_shape_mesh(
                    body if is_probe else -1,
                    mesh=probe if is_probe else box,
                    cfg=cfg,
                )
            # Each shape order shares one model across all collision-only cases.
            model = builder.finalize(device=device)
            state = model.state()
            state.body_qd.assign([[0.0, 0.0, -2.0, 0.0, 0.0, 0.0]])
            for case, reduce_contacts, distance in cases:
                with test.subTest(probe_first=probe_first, case=case):
                    state.body_q.assign([wp.transform(wp.vec3(0.0, 0.0, 0.5 + distance + probe_half_height))])
                    pipeline = newton.CollisionPipeline(
                        model,
                        broad_phase="nxn",
                        reduce_contacts=reduce_contacts,
                        speculative_contact_gap_max=0.2 if distance > 0.0 else None,
                    )
                    if probe_sdf:
                        test.assertTrue(pipeline.narrow_phase.mesh_sdf_texture_only)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts, dt=0.1)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    test.assertGreater(count, 0)
                    points0 = contacts.rigid_contact_point0.numpy()[:count]
                    points1 = contacts.rigid_contact_point1.numpy()[:count]
                    probe_points, box_points = (points0, points1) if probe_first else (points1, points0)
                    # Dynamic-body witnesses are body-local; static ones are world-space.
                    if probe_sdf:
                        # Side edges can also produce contacts; accept any probe-box face.
                        np.testing.assert_allclose(np.max(np.abs(probe_points), axis=1), 0.2, atol=sdf_tolerance)
                    else:
                        np.testing.assert_allclose(probe_points[:, 2], 0.0, atol=2.0e-5)
                    np.testing.assert_allclose(box_points[:, 2], 0.5, atol=sdf_tolerance)


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
    def test_nonconvex_edge_keeps_endpoint_minima(self):
        """Keep deeper endpoint minima despite a shallower interior well."""
        for device in get_test_devices():
            for kernel in (_nonconvex_edge_scalar, _nonconvex_edge_paired):
                with self.subTest(device=device, kernel=kernel.key):
                    scales = wp.array([0.001, 1.0, 1000.0], dtype=float, device=device)
                    distances = wp.empty(6, dtype=float, device=device)
                    endpoints = wp.empty(6, dtype=int, device=device)
                    wp.launch(
                        kernel,
                        dim=6,
                        inputs=[
                            scales,
                            distances,
                            endpoints,
                            wp.empty(0, dtype=wp.uint8, device=device),
                            wp.empty(0, dtype=float, device=device),
                        ],
                        device=device,
                    )
                    np.testing.assert_allclose(distances.numpy(), np.repeat(-0.2 * scales.numpy(), 2), rtol=1e-6)
                    np.testing.assert_array_equal(endpoints.numpy(), [1, 2, 1, 2, 1, 2])

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
                edge_range = wp.vec2i(*model.shape_edge_range.numpy()[0])
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
                            model.mesh_edge_indices,
                            model.mesh_edge_centers,
                            model.mesh_edge_order,
                            model.mesh_edge_clusters,
                            model.mesh_edge_cluster_reach,
                            edge_range,
                            transforms,
                            contact_threshold,
                            may,
                        ],
                        outputs=[passes],
                        device=device,
                    )
                    may_np, passes_np = may.numpy(), passes.numpy()
                    order = model.mesh_edge_order.numpy()[edge_range[0] : edge_range[0] + num_edges]
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


add_function_test(
    TestSDFContact,
    "test_mesh_mesh_contained_bvh_contacts",
    test_mesh_mesh_contained_bvh_contacts,
    devices=get_test_devices(),
)

add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_bvh_bvh",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_test_devices(),
    box_sdf=False,
)
add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_bvh_texture",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_cuda_test_devices(),
    box_sdf=True,
)
add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_texture_texture",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_cuda_test_devices(),
    box_sdf=True,
    probe_sdf=True,
)


if __name__ == "__main__":
    unittest.main()

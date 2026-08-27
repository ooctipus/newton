# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import make_contact_sort_key
from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    _fold_contact_fingerprint,
    export_and_reduce_contact_centered_two_spatial_depths,
)
from newton._src.geometry.flags import MeshProperties
from newton._src.geometry.sdf_contact import (
    _create_sdf_contact_funcs,
    _mesh_sdf_nonpenetration_oracle_owns_pair,
    _sdf_rsqrt_rn,
    compute_block_counts_from_weights,
    mesh_sdf_contact_endpoint_owned,
    mesh_sdf_contact_passes_inner_cull_consistency,
    mesh_sdf_contact_search_precision,
    mesh_sdf_contact_segment_bounds,
    mesh_sdf_contact_segment_count,
    mesh_sdf_contact_segment_minimum_is_unique,
    mesh_sdf_contact_sort_sub_key,
    mesh_sdf_endpoint_guard_passes_admission,
)
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.sim.contact_oracle import (
    CONTACT_ORACLE_CAPACITY,
    CONTACT_ORACLE_INCOMPLETE,
    CONTACT_ORACLE_REFERENCE_INFEASIBLE,
    CONTACT_ORACLE_SWEEP_INCOMPLETE,
    CONTACT_ORACLE_VIOLATION,
    RIGID_BODY_PATH_BOUNDED,
    RIGID_BODY_PATH_CONSTANT_TWIST,
    RIGID_BODY_PATH_LINEAR_TRANSLATION,
    RIGID_BODY_PATH_STATIONARY,
    RIGID_BODY_PATH_UNKNOWN,
    RigidBodyPathCertificate,
)
from newton._src.utils.heightfield import HeightfieldData
from newton.tests.unittest_utils import get_cuda_test_devices, get_test_devices


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_search_precision_kernel(out: wp.array[wp.float32]):
    out[0] = mesh_sdf_contact_search_precision(0.0, 1.0, 0.001, True)
    out[1] = mesh_sdf_contact_search_precision(0.01, 1.0, 0.001, True)
    out[2] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.1, True)
    out[3] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.001, False)


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_segment_count_kernel(out: wp.array[wp.int32]):
    out[0] = mesh_sdf_contact_segment_count(0.04944, 0.00135, True)
    out[1] = mesh_sdf_contact_segment_count(0.0026, 0.00135, True)
    out[2] = mesh_sdf_contact_segment_count(1.0, 0.001, True)
    out[3] = mesh_sdf_contact_segment_count(0.04944, 0.00135, False)
    out[4] = mesh_sdf_contact_segment_count(0.04944, 0.0, True)


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_feature_key_kernel(
    edge_idx: int,
    public_keys: wp.array[wp.int64],
    reducer_keys: wp.array[wp.uint64],
):
    tid = wp.tid()
    segment_idx = tid
    endpoint = int(0)
    if tid >= 32:
        segment_idx = int(0)
        endpoint = tid - 31
    feature_key = mesh_sdf_contact_sort_sub_key(edge_idx, segment_idx, 0, endpoint)
    public_keys[tid] = make_contact_sort_key(0, 1, feature_key)
    reducer_keys[tid] = _fold_contact_fingerprint(feature_key)


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_internal_boundary_owner_kernel(out: wp.array[wp.int32]):
    out[0] = int(mesh_sdf_contact_segment_minimum_is_unique(2, 0))
    out[1] = int(mesh_sdf_contact_segment_minimum_is_unique(1, 1))
    out[2] = int(mesh_sdf_contact_segment_minimum_is_unique(0, 1))


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_endpoint_ownership_kernel(codes: wp.array[wp.int32], out: wp.array[wp.int32]):
    tid = wp.tid()
    out[2 * tid] = int(mesh_sdf_contact_endpoint_owned(codes[tid], 1))
    out[2 * tid + 1] = int(mesh_sdf_contact_endpoint_owned(codes[tid], 2))


@wp.kernel(enable_backward=False)
def _mesh_sdf_endpoint_guard_admission_kernel(out: wp.array[wp.int32]):
    margin_sum = 0.002
    base_gap_sum = 0.001
    max_speculative_extension = 0.003
    out[0] = int(mesh_sdf_endpoint_guard_passes_admission(0.001, margin_sum, base_gap_sum, max_speculative_extension))
    out[1] = int(mesh_sdf_endpoint_guard_passes_admission(0.002, margin_sum, base_gap_sum, max_speculative_extension))
    out[2] = int(mesh_sdf_endpoint_guard_passes_admission(0.005, margin_sum, base_gap_sum, max_speculative_extension))
    out[3] = int(
        mesh_sdf_endpoint_guard_passes_admission(0.005001, margin_sum, base_gap_sum, max_speculative_extension)
    )


@wp.kernel(enable_backward=False)
def _mesh_sdf_endpoint_guard_midpoint_bypass_kernel(out: wp.array[wp.int32]):
    out[0] = int(
        mesh_sdf_contact_passes_inner_cull_consistency(
            -0.001,
            0.0,
            1.0,
            wp.vec3(0.0),
            0.0,
            wp.vec3(-1.0),
            wp.vec3(1.0),
            1.0,
            False,
        )
    )
    out[1] = int(mesh_sdf_endpoint_guard_passes_admission(-0.001, 0.0, 0.0, 0.0))


@wp.kernel(enable_backward=False)
def _mesh_sdf_nonpenetration_ownership_kernel(
    shape_pairs: wp.array[wp.vec2i],
    eligible_shape: wp.array[wp.uint8],
    anchor_shape: wp.array[wp.uint8],
    shape_source: wp.array[wp.uint64],
    shape_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    out: wp.array[wp.int32],
):
    pair_index = wp.tid()
    pair = shape_pairs[pair_index]
    out[pair_index] = int(
        _mesh_sdf_nonpenetration_oracle_owns_pair(
            pair[0],
            pair[1],
            eligible_shape,
            anchor_shape,
            shape_source,
            shape_body,
            body_flags,
        )
    )


@wp.func
def _two_well_edge_sdf(_sdf: TextureSDFData, point: wp.vec3) -> float:
    """One-dimensional 1-Lipschitz SDF with two separated narrow wells."""
    edge_length = 0.04944
    witness_x = edge_length * 0.341502
    other_x = edge_length * 0.62
    witness_well = wp.abs(point[0] - witness_x) - 0.00035
    other_well = wp.abs(point[0] - other_x) - 0.00070
    return wp.min(witness_well, other_well)


@wp.func
def _two_well_edge_sdf_pair(sdf: TextureSDFData, point_a: wp.vec3, point_b: wp.vec3) -> wp.vec2f:
    return wp.vec2f(_two_well_edge_sdf(sdf, point_a), _two_well_edge_sdf(sdf, point_b))


_two_well_edge_collision, _ = _create_sdf_contact_funcs(
    False,
    True,
    _two_well_edge_sdf,
    _two_well_edge_sdf_pair,
)


@wp.kernel(enable_backward=False)
def _two_well_edge_candidates_kernel(
    heightfield_elevations: wp.array[wp.float32],
    candidate_mask: wp.array[wp.int32],
    candidate_t: wp.array[wp.float32],
    candidate_depth: wp.array[wp.float32],
    legacy_result: wp.array[wp.vec2f],
    reducer_data: GlobalContactReducerData,
):
    edge_length = 0.04944
    voxel_radius = 0.00135
    v0 = wp.vec3(0.0)
    v1 = wp.vec3(edge_length, 0.0, 0.0)
    sdf = TextureSDFData()
    midpoint_sdf = _two_well_edge_sdf(sdf, 0.5 * (v0 + v1))
    legacy_depth, legacy_point, _legacy_endpoint = _two_well_edge_collision(
        sdf,
        wp.uint64(0),
        v0,
        v1,
        midpoint_sdf,
        False,
        0,
        False,
        HeightfieldData(),
        heightfield_elevations,
        1.0e-6,
    )
    legacy_result[0] = wp.vec2f(legacy_point[0] / edge_length, legacy_depth)

    segment_count = mesh_sdf_contact_segment_count(edge_length, voxel_radius, True)
    for segment_idx in range(segment_count):
        segment_v0, segment_v1 = mesh_sdf_contact_segment_bounds(v0, v1, segment_idx, segment_count)
        segment_midpoint = 0.5 * (segment_v0 + segment_v1)
        segment_midpoint_sdf = _two_well_edge_sdf(sdf, segment_midpoint)
        segment_radius = 0.5 * wp.length(segment_v1 - segment_v0)
        if segment_midpoint_sdf <= segment_radius:
            depth, point, best_endpoint = _two_well_edge_collision(
                sdf,
                wp.uint64(0),
                segment_v0,
                segment_v1,
                segment_midpoint_sdf,
                False,
                0,
                False,
                HeightfieldData(),
                heightfield_elevations,
                1.0e-6,
            )
            if depth < 0.0 and mesh_sdf_contact_segment_minimum_is_unique(best_endpoint, segment_idx):
                candidate_mask[segment_idx] = 1
                candidate_t[segment_idx] = point[0] / edge_length
                candidate_depth[segment_idx] = depth
                fingerprint = mesh_sdf_contact_sort_sub_key(0, segment_idx, 0, 0)
                export_and_reduce_contact_centered_two_spatial_depths(
                    0,
                    1,
                    point,
                    wp.vec3(0.0, 0.0, 1.0),
                    depth,
                    fingerprint,
                    point - wp.vec3(0.5 * edge_length, 0.0, 0.0),
                    0.0,
                    0.0,
                    False,
                    False,
                    1.0,
                    point,
                    wp.vec3(0.0, -0.001, -0.001),
                    wp.vec3(edge_length, 0.001, 0.001),
                    wp.vec3i(segment_count, 1, 1),
                    reducer_data,
                )


@wp.kernel(enable_backward=False)
def _sdf_rsqrt_rn_kernel(values: wp.array[wp.float32], out: wp.array[wp.float32]):
    tid = wp.tid()
    out[tid] = _sdf_rsqrt_rn(values[tid])


class TestSDFContact(unittest.TestCase):
    @staticmethod
    def _combine_box_meshes(
        boxes: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
    ) -> newton.Mesh:
        """Combine translated boxes into one disconnected raw mesh."""
        vertices = []
        triangles = []
        for center, half_extents in boxes:
            box = newton.Mesh.create_box(*half_extents, compute_normals=False, compute_uvs=False)
            offset = len(vertices)
            vertices.extend(box.vertices + np.asarray(center, dtype=np.float32))
            triangles.extend(box.indices.reshape(-1, 3) + offset)
        return newton.Mesh(
            np.asarray(vertices, dtype=np.float32),
            np.asarray(triangles, dtype=np.int32),
            compute_inertia=False,
        )

    @staticmethod
    def _path_certificate(
        reference_body_q: wp.array,
        candidate_body_q: wp.array,
        *,
        origin_path_length: np.ndarray | None = None,
        angular_path_length: np.ndarray | None = None,
        motion_kind: np.ndarray | None = None,
    ) -> RigidBodyPathCertificate:
        """Declare a straight-origin, shortest-orientation path for an oracle test."""
        reference = reference_body_q.numpy()
        candidate = candidate_body_q.numpy()
        if origin_path_length is None:
            origin_path_length = np.linalg.norm(candidate[:, :3] - reference[:, :3], axis=1)
        if angular_path_length is None:
            reference_quat = reference[:, 3:].copy()
            candidate_quat = candidate[:, 3:].copy()
            reference_quat /= np.linalg.norm(reference_quat, axis=1, keepdims=True)
            candidate_quat /= np.linalg.norm(candidate_quat, axis=1, keepdims=True)
            negative_arc = np.sum(reference_quat * candidate_quat, axis=1) < 0.0
            candidate_quat[negative_arc] *= -1.0
            quaternion_delta = np.linalg.norm(candidate_quat - reference_quat, axis=1)
            angular_path_length = 4.0 * np.arcsin(np.clip(0.5 * quaternion_delta, 0.0, 1.0))
        origin_path_length = np.asarray(origin_path_length, dtype=np.float32)
        angular_path_length = np.asarray(angular_path_length, dtype=np.float32)
        if motion_kind is None:
            stationary = (origin_path_length <= 1.0e-7) & (angular_path_length <= 1.0e-7)
            linear = ~stationary & (angular_path_length <= 1.0e-7)
            motion_kind = np.full(candidate_body_q.shape[0], RIGID_BODY_PATH_CONSTANT_TWIST, dtype=np.uint8)
            motion_kind[stationary] = RIGID_BODY_PATH_STATIONARY
            motion_kind[linear] = RIGID_BODY_PATH_LINEAR_TRANSLATION
        device = candidate_body_q.device
        return RigidBodyPathCertificate(
            endpoint_body_q=candidate_body_q,
            origin_path_length=wp.array(origin_path_length, dtype=wp.float32, device=device),
            angular_path_length=wp.array(angular_path_length, dtype=wp.float32, device=device),
            motion_kind=wp.array(motion_kind, dtype=wp.uint8, device=device),
        )

    def _refine_nonpenetration_oracle(
        self,
        oracle,
        reference_body_q: wp.array,
        candidate_body_q: wp.array,
        **path_kwargs,
    ) -> None:
        """Refine one oracle using an explicit test-owned path certificate."""
        oracle.refine(reference_body_q, self._path_certificate(reference_body_q, candidate_body_q, **path_kwargs))

    def _scan_nonpenetration_oracle(
        self,
        oracle,
        reference_body_q: wp.array,
        candidate_body_q: wp.array,
        **path_kwargs,
    ) -> None:
        """Scan one oracle using an explicit test-owned path certificate."""
        oracle.scan(reference_body_q, self._path_certificate(reference_body_q, candidate_body_q, **path_kwargs))

    def _claim_nonpenetration_oracle(self, pipeline: newton.CollisionPipeline):
        """Claim one private oracle for focused geometry tests."""
        oracle, _token = pipeline._claim_strict_nonpenetration_oracle(object(), allow_oracle=True)
        self.assertIsNotNone(oracle)
        return oracle

    def _make_nonpenetration_oracle(
        self,
        source_mesh: newton.Mesh,
        target_mesh: newton.Mesh,
        *,
        reverse_shape_order: bool = False,
        source_xform: wp.transform | None = None,
        source_shape_xform: wp.transform | None = None,
        target_xform: wp.transform | None = None,
        source_scale: wp.vec3 | None = None,
        target_scale: wp.vec3 | None = None,
        source_margin: float = 0.0,
        target_margin: float = 0.0,
        device: wp.Device | None = None,
    ):
        """Build a CPU mesh pair with the private strict surface oracle enabled."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_cfg = newton.ModelBuilder.ShapeConfig(margin=source_margin)
        target_cfg = newton.ModelBuilder.ShapeConfig(margin=target_margin)
        if reverse_shape_order:
            target_shape = builder.add_shape_mesh(
                -1, mesh=target_mesh, xform=target_xform, scale=target_scale, cfg=target_cfg
            )
            source_body = builder.add_body(xform=source_xform)
            source_shape = builder.add_shape_mesh(
                source_body,
                mesh=source_mesh,
                xform=source_shape_xform,
                scale=source_scale,
                cfg=source_cfg,
            )
        else:
            source_body = builder.add_body(xform=source_xform)
            source_shape = builder.add_shape_mesh(
                source_body,
                mesh=source_mesh,
                xform=source_shape_xform,
                scale=source_scale,
                cfg=source_cfg,
            )
            target_shape = builder.add_shape_mesh(
                -1, mesh=target_mesh, xform=target_xform, scale=target_scale, cfg=target_cfg
            )
        model = builder.finalize(device=wp.get_device("cpu") if device is None else device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        self.assertIsNone(pipeline._strict_nonpenetration_oracle)
        oracle = self._claim_nonpenetration_oracle(pipeline)
        return model, pipeline, oracle, source_shape, target_shape

    def test_split_mesh_sdf_matches_overflow_fallback(self) -> None:
        """Preserve reduced contacts when split work exceeds its scratch capacity."""
        for device in get_cuda_test_devices():
            with self.subTest(device=device):
                mesh = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False, compute_inertia=False)
                mesh.build_sdf(max_resolution=32, device=device)
                builder = newton.ModelBuilder()
                mesh_body = builder.add_body(xform=wp.transform_identity())
                box_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()))
                builder.add_shape_mesh(mesh_body, mesh=mesh)
                builder.add_shape_box(box_body, cfg=newton.ModelBuilder.ShapeConfig(sdf_max_resolution=32))
                model = builder.finalize(device=device)
                pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase="nxn",
                    deterministic=True,
                    reduce_contacts=True,
                    rigid_contact_max=128,
                    max_triangle_pairs=4096,
                )
                self.assertTrue(pipeline.narrow_phase._use_mesh_sdf_split)
                contacts = pipeline.contacts()
                state = model.state()

                def collide(active_pipeline, active_state, active_contacts) -> tuple[np.ndarray, ...]:
                    active_pipeline.collide(active_state, active_contacts)
                    count = int(active_contacts.rigid_contact_count.numpy()[0])
                    self.assertGreater(count, 0)
                    values = (
                        active_contacts.rigid_contact_shape0.numpy()[:count],
                        active_contacts.rigid_contact_shape1.numpy()[:count],
                        active_contacts.rigid_contact_point0.numpy()[:count],
                        active_contacts.rigid_contact_point1.numpy()[:count],
                        active_contacts.rigid_contact_normal.numpy()[:count],
                    )
                    order = np.lexsort(tuple(np.column_stack(values).T[::-1]))
                    return tuple(value[order] for value in values)

                split_contacts = collide(pipeline, state, contacts)
                self.assertEqual(int(pipeline.narrow_phase.mesh_sdf_work_state.numpy()[1]), 0)

                pipeline.narrow_phase.mesh_sdf_segment_capacity = 0
                fallback_contacts = collide(pipeline, state, contacts)
                self.assertEqual(int(pipeline.narrow_phase.mesh_sdf_work_state.numpy()[1]), 1)

                for split, fallback in zip(split_contacts, fallback_contacts, strict=True):
                    if split.ndim == 1:
                        np.testing.assert_array_equal(split, fallback)
                    else:
                        np.testing.assert_allclose(split, fallback, rtol=1.0e-5, atol=1.0e-6)

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

    def test_mesh_sdf_contact_segments_are_voxel_scaled_and_bounded(self) -> None:
        device = wp.get_preferred_device()
        values = wp.empty(5, dtype=wp.int32, device=device)

        wp.launch(_mesh_sdf_contact_segment_count_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_array_equal(values.numpy(), (19, 1, 32, 1, 1))

    def test_mesh_sdf_contact_segment_keys_keep_minima_and_endpoints_distinct(self) -> None:
        device = wp.get_preferred_device()
        for edge_idx in (0, 17, 12345):
            with self.subTest(edge_idx=edge_idx):
                public_keys = wp.empty(34, dtype=wp.int64, device=device)
                reducer_keys = wp.empty(34, dtype=wp.uint64, device=device)
                wp.launch(
                    _mesh_sdf_contact_feature_key_kernel,
                    dim=34,
                    inputs=[edge_idx, public_keys, reducer_keys],
                    device=device,
                )

                self.assertEqual(len(np.unique(public_keys.numpy())), 34)
                self.assertEqual(len(np.unique(reducer_keys.numpy())), 34)

    def test_mesh_sdf_contact_internal_boundary_belongs_to_lower_segment(self) -> None:
        device = wp.get_preferred_device()
        values = wp.empty(3, dtype=wp.int32, device=device)

        wp.launch(_mesh_sdf_contact_internal_boundary_owner_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_array_equal(values.numpy(), (1, 0, 1))

    def test_mesh_sdf_contact_endpoint_ownership_supports_legacy_and_explicit_codes(self) -> None:
        device = wp.get_preferred_device()
        codes = wp.array([0, 4, 5, 6, 7], dtype=wp.int32, device=device)
        values = wp.empty(2 * codes.shape[0], dtype=wp.int32, device=device)

        wp.launch(
            _mesh_sdf_contact_endpoint_ownership_kernel, dim=codes.shape[0], inputs=[codes, values], device=device
        )

        np.testing.assert_array_equal(values.numpy().reshape(-1, 2), ((1, 1), (0, 0), (1, 0), (0, 1), (1, 1)))

    def test_mesh_sdf_endpoint_guard_admission_is_inclusive(self) -> None:
        device = wp.get_device("cpu")
        values = wp.empty(4, dtype=wp.int32, device=device)

        wp.launch(_mesh_sdf_endpoint_guard_admission_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_array_equal(values.numpy(), (1, 1, 1, 0))

    def test_mesh_sdf_endpoint_guard_bypasses_midpoint_cull(self) -> None:
        device = wp.get_device("cpu")
        values = wp.empty(2, dtype=wp.int32, device=device)

        wp.launch(_mesh_sdf_endpoint_guard_midpoint_bypass_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_array_equal(values.numpy(), (0, 1))

    def test_mesh_sdf_nonpenetration_ownership_uses_live_body_flags(self) -> None:
        """Classify movable--immovable pairs symmetrically from live metadata."""
        device = wp.get_device("cpu")
        pairs = wp.array(
            ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (0, 3), (3, 0)),
            dtype=wp.vec2i,
            device=device,
        )
        eligible = wp.ones(4, dtype=wp.uint8, device=device)
        anchor = wp.ones(4, dtype=wp.uint8, device=device)
        shape_source = wp.ones(4, dtype=wp.uint64, device=device)
        shape_body = wp.array((0, -1, 1, 2), dtype=wp.int32, device=device)
        body_flags = wp.array(
            (int(newton.BodyFlags.DYNAMIC), int(newton.BodyFlags.KINEMATIC), int(newton.BodyFlags.DYNAMIC)),
            dtype=wp.int32,
            device=device,
        )
        out = wp.empty(len(pairs), dtype=wp.int32, device=device)

        wp.launch(
            _mesh_sdf_nonpenetration_ownership_kernel,
            dim=len(pairs),
            inputs=[pairs, eligible, anchor, shape_source, shape_body, body_flags, out],
            device=device,
        )
        np.testing.assert_array_equal(out.numpy(), (1, 1, 1, 1, 0, 0, 0, 0))

        shape_source.assign(np.asarray((1, 0, 1, 1), dtype=np.uint64))
        wp.launch(
            _mesh_sdf_nonpenetration_ownership_kernel,
            dim=len(pairs),
            inputs=[pairs, eligible, anchor, shape_source, shape_body, body_flags, out],
            device=device,
        )
        np.testing.assert_array_equal(out.numpy(), (0, 0, 1, 1, 0, 0, 0, 0))
        shape_source.fill_(1)

        body_flags.assign(
            np.array(
                (int(newton.BodyFlags.KINEMATIC), int(newton.BodyFlags.DYNAMIC), int(newton.BodyFlags.DYNAMIC)),
                dtype=np.int32,
            )
        )
        wp.launch(
            _mesh_sdf_nonpenetration_ownership_kernel,
            dim=len(pairs),
            inputs=[pairs, eligible, anchor, shape_source, shape_body, body_flags, out],
            device=device,
        )
        np.testing.assert_array_equal(out.numpy(), (0, 0, 1, 1, 1, 1, 1, 1))

    def test_mesh_sdf_nonpenetration_oracle_rejects_reference_overlap(self) -> None:
        """Reject an owned pair that is already overlapping at the composition pose."""
        device = wp.get_device("cpu")
        held_mesh = newton.Mesh.create_box(0.05, 0.05, 0.1, compute_normals=False, compute_uvs=False)
        fixed_mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        held_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.19), wp.quat_identity()))
        fixed_body = builder.add_body()
        builder.body_flags[fixed_body] = int(newton.BodyFlags.KINEMATIC)
        builder.add_shape_mesh(held_body, mesh=held_mesh)
        builder.add_shape_mesh(fixed_body, mesh=fixed_mesh)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        self.assertIsNone(pipeline._strict_nonpenetration_oracle)
        self.assertFalse(hasattr(pipeline, "strict_nonpenetration_oracle"))
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()
        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(
            int(oracle.world_status.numpy()[0]),
            CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
        )
        self.assertAlmostEqual(float(oracle.world_min_separation.numpy()[0]), -0.01, delta=2.0e-5)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_materializes_reference_connected_edge_guard(self) -> None:
        """Materialize a solver guard when a candidate edge straddles a slab."""
        rod_mesh = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        slab_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            rod_mesh,
            slab_mesh,
            source_xform=wp.transform(wp.vec3(0.11, 0.04, 0.12), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        candidate = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate[source_body, 2] = 0.0
        candidate_body_q = wp.array(candidate, dtype=wp.transform, device=model.device)

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
        self.assertAlmostEqual(float(oracle.world_min_separation.numpy()[0]), -0.11, delta=2.0e-5)
        guard_count = int(oracle.guard_contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(guard_count, 1)
        point_ids = oracle.guard_contacts.rigid_contact_point_id.numpy()[:guard_count]
        self.assertTrue(np.all(((point_ids >> 2) & 1) == 1))
        self.assertTrue(np.all((point_ids & 3) == 0))

    def test_mesh_sdf_nonpenetration_oracle_materializes_interior_edge_guard(self) -> None:
        """Connect an interior edge sample when authored vertices stay outside."""
        source_mesh = newton.Mesh.create_box(0.2, 0.02, 0.005, compute_normals=False, compute_uvs=False)
        source_vertices = source_mesh.vertices
        target_mesh = newton.Mesh.create_box(0.1, 0.1, 0.05, compute_normals=False, compute_uvs=False)
        model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        source_body = int(model.shape_body.numpy()[source_shape])

        self._scan_nonpenetration_oracle(oracle, reference_body_q, reference_body_q)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

        candidate_np = reference_body_q.numpy().copy()
        candidate_np[source_body, 2] = 0.0
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        target_min = np.asarray((-0.1, -0.1, -0.05), dtype=np.float32)
        target_max = np.asarray((0.1, 0.1, 0.05), dtype=np.float32)
        reference_vertices = source_vertices + np.asarray((0.0, 0.0, 0.12), dtype=np.float32)
        reference_targets = np.clip(reference_vertices, target_min, target_max)
        reference_normals = reference_vertices - reference_targets
        reference_normals /= np.linalg.norm(reference_normals, axis=1, keepdims=True)
        endpoint_plane_separations = np.sum((source_vertices - reference_targets) * reference_normals, axis=1)
        self.assertTrue(np.all(endpoint_plane_separations > 0.01))

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        status = int(oracle.world_status.numpy()[0])
        guard_count = int(oracle.guard_contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(status, CONTACT_ORACLE_VIOLATION, f"unexpected guard count: {guard_count}")
        self.assertAlmostEqual(float(oracle.world_min_separation.numpy()[0]), -0.055, delta=2.0e-5)
        self.assertGreater(guard_count, 0)
        point_ids = oracle.guard_contacts.rigid_contact_point_id.numpy()[:guard_count]
        self.assertTrue(np.all(((point_ids >> 2) & 1) == 1))
        source_guard_indices = np.flatnonzero((point_ids & 3) == 0)
        self.assertEqual(source_guard_indices.size, 1)
        source_guard_index = int(source_guard_indices[0])
        source_points = oracle.guard_contacts.rigid_contact_point0.numpy()[:guard_count]
        target_points = oracle.guard_contacts.rigid_contact_point1.numpy()[:guard_count]
        normals = oracle.guard_contacts.rigid_contact_normal.numpy()[:guard_count]
        self.assertAlmostEqual(
            float(np.linalg.norm(source_points[source_guard_index] - target_points[source_guard_index])),
            0.055,
            delta=2.0e-5,
        )
        np.testing.assert_allclose(normals[source_guard_index], (0.0, 0.0, -1.0), rtol=0.0, atol=2.0e-5)

        first_point_ids = point_ids.copy()
        self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), guard_count)
        np.testing.assert_array_equal(
            oracle.guard_contacts.rigid_contact_point_id.numpy()[:guard_count],
            first_point_ids,
        )

    def test_mesh_sdf_nonpenetration_oracle_relinearizes_retained_feature_in_place(self) -> None:
        """Update one retained feature's anchor without duplicating its compact contact row."""
        device = wp.get_device("cpu")
        held_mesh = newton.Mesh.create_box(0.02, 0.02, 0.02, compute_normals=False, compute_uvs=False)
        fixed_mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        source_xform = wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity())
        model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            held_mesh,
            fixed_mesh,
            source_xform=source_xform,
            device=device,
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 2] = 0.0
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=device)

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertNotEqual(int(oracle.world_status.numpy()[0]) & CONTACT_ORACLE_VIOLATION, 0)
        self.assertAlmostEqual(float(oracle.world_min_separation.numpy()[0]), -0.08, delta=2.0e-5)
        oracle_count = int(oracle.guard_contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(oracle_count, 1)
        np.testing.assert_array_equal(oracle.guard_contacts.rigid_contact_is_strict_guard.numpy()[:oracle_count], 2)
        first_point = oracle.guard_contacts.rigid_contact_point1.numpy()[0].copy()
        first_shape0 = int(oracle.guard_contacts.rigid_contact_shape0.numpy()[0])
        first_shape1 = int(oracle.guard_contacts.rigid_contact_shape1.numpy()[0])
        first_point_id = int(oracle.guard_contacts.rigid_contact_point_id.numpy()[0])

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 1)
        np.testing.assert_array_equal(oracle.guard_contacts.rigid_contact_point1.numpy()[0], first_point)

        updated_candidate_np = candidate_np.copy()
        updated_candidate_np[source_body, 0] = 0.001
        updated_candidate_body_q = wp.array(updated_candidate_np, dtype=wp.transform, device=device)
        self._refine_nonpenetration_oracle(oracle, reference_body_q, updated_candidate_body_q)
        updated_count = int(oracle.guard_contacts.rigid_contact_count.numpy()[0])
        shape0 = oracle.guard_contacts.rigid_contact_shape0.numpy()[:updated_count]
        shape1 = oracle.guard_contacts.rigid_contact_shape1.numpy()[:updated_count]
        point_ids = oracle.guard_contacts.rigid_contact_point_id.numpy()[:updated_count]
        matching_feature = np.flatnonzero(
            (shape0 == first_shape0) & (shape1 == first_shape1) & (point_ids == first_point_id)
        )
        self.assertEqual(matching_feature.size, 1)
        feature_index = int(matching_feature[0])
        updated_points = oracle.guard_contacts.rigid_contact_point1.numpy()[:updated_count].copy()
        self.assertFalse(np.array_equal(updated_points[feature_index], first_point))

        self._refine_nonpenetration_oracle(oracle, reference_body_q, updated_candidate_body_q)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), updated_count)
        np.testing.assert_array_equal(
            oracle.guard_contacts.rigid_contact_point1.numpy()[:updated_count],
            updated_points,
        )

        oracle.guard_contacts.contact_generation.fill_(np.iinfo(np.int32).max)
        self._refine_nonpenetration_oracle(oracle, reference_body_q, updated_candidate_body_q)
        self.assertEqual(int(oracle.guard_contacts.contact_generation.numpy()[0]), 0)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), updated_count)
        np.testing.assert_array_equal(
            oracle.guard_contacts.rigid_contact_point1.numpy()[:updated_count],
            updated_points,
        )

    def test_mesh_sdf_nonpenetration_oracle_delegates_dynamic_dynamic_pairs(self) -> None:
        """Allocate structurally for two free bodies while delegating their live dynamic pair."""
        device = wp.get_device("cpu")
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.19), wp.quat_identity()))
        target_body = builder.add_body()
        builder.add_shape_mesh(source_body, mesh=mesh)
        builder.add_shape_mesh(target_body, mesh=mesh)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        owner = object()
        oracle, token = pipeline._claim_strict_nonpenetration_oracle(owner, allow_oracle=True)
        self.assertIsNotNone(oracle)
        self.assertIs(token, owner)
        second_oracle, second_token = pipeline._claim_strict_nonpenetration_oracle(owner, allow_oracle=True)
        self.assertIs(second_oracle, oracle)
        self.assertIs(second_token, token)
        state = model.state()

        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.01)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        self.assertGreater(np.count_nonzero(contacts.rigid_contact_is_strict_guard.numpy()[:count] == 1), 0)

    def test_mesh_sdf_nonpenetration_oracle_uses_only_direct_root_free_bodies_as_movable(self) -> None:
        """Exclude a descendant free joint from the immutable movable capability mask."""
        mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        direct_body = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0)))
        direct_shape = builder.add_shape_mesh(direct_body, mesh=mesh)
        parent = builder.add_link(mass=1.0)
        descendant = builder.add_link(mass=1.0)
        parent_joint = builder.add_joint_revolute(-1, parent)
        descendant_joint = builder.add_joint_free(parent=parent, child=descendant)
        builder.add_articulation([parent_joint, descendant_joint])
        descendant_shape = builder.add_shape_mesh(descendant, mesh=mesh)
        builder.add_shape_mesh(-1, mesh=mesh)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        movable_capable = oracle._eligible_shape.numpy()

        self.assertEqual(int(movable_capable[direct_shape]), 1)
        self.assertEqual(int(movable_capable[descendant_shape]), 0)

    def test_mesh_sdf_nonpenetration_oracle_does_not_suppress_heightfield_guards(self) -> None:
        """Keep a raw movable versus heightfield pair outside the oracle's raw-mesh anchor domain."""
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.11)))
        source_shape = builder.add_shape_mesh(source_body, mesh=mesh)
        heightfield_shape = builder.add_shape_heightfield(
            heightfield=newton.Heightfield(
                data=np.zeros((3, 3), dtype=np.float32),
                nrow=3,
                ncol=3,
                hx=1.0,
                hy=1.0,
                min_z=0.0,
                max_z=0.0,
            )
        )
        builder.add_shape_mesh(-1, mesh=mesh, xform=wp.transform(wp.vec3(10.0, 0.0, 0.0)))
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()

        self.assertEqual(int(oracle._anchor_capable_shape.numpy()[heightfield_shape]), 0)
        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)

        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.01)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        strict_guard = contacts.rigid_contact_is_strict_guard.numpy()[:count]
        pair = ((shape0 == source_shape) & (shape1 == heightfield_shape)) | (
            (shape0 == heightfield_shape) & (shape1 == source_shape)
        )
        self.assertGreater(np.count_nonzero(pair & (strict_guard != 0)), 0)

    def test_mesh_sdf_nonpenetration_oracle_does_not_suppress_convex_mesh_guards(self) -> None:
        """Keep a raw movable versus convex mesh pair under the ordinary strict guard path."""
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.21)))
        source_shape = builder.add_shape_mesh(source_body, mesh=mesh)
        convex_shape = builder.add_shape_convex_hull(-1, mesh=mesh)
        builder.add_shape_mesh(-1, mesh=mesh, xform=wp.transform(wp.vec3(10.0, 0.0, 0.0)))
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()

        self.assertEqual(int(oracle._anchor_capable_shape.numpy()[convex_shape]), 0)
        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)

        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.01)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        strict_guard = contacts.rigid_contact_is_strict_guard.numpy()[:count]
        pair = ((shape0 == source_shape) & (shape1 == convex_shape)) | (
            (shape0 == convex_shape) & (shape1 == source_shape)
        )
        self.assertGreater(np.count_nonzero(pair & (strict_guard != 0)), 0)

    def test_mesh_sdf_nonpenetration_oracle_excludes_zero_source_mesh_anchor(self) -> None:
        """Track live raw-source loss and recovery independently of immutable MESH capability."""
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2)))
        builder.add_shape_mesh(source_body, mesh=mesh)
        target_shape = builder.add_shape_mesh(-1, mesh=mesh)
        builder.add_shape_mesh(-1, mesh=mesh, xform=wp.transform(wp.vec3(10.0, 0.0, 0.0)))
        model = builder.finalize(device="cpu")
        original_shape_source = model.shape_source_ptr.numpy().copy()
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()

        self.assertEqual(int(oracle._anchor_capable_shape.numpy()[target_shape]), 1)
        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

        shape_source = original_shape_source.copy()
        shape_source[target_shape] = 0
        model.shape_source_ptr.assign(shape_source)
        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

        model.shape_source_ptr.assign(original_shape_source)
        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_delegates_uncertified_articulated_meshes(self) -> None:
        """Keep unrelated articulated meshes from invalidating certified free-body queries."""
        mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        free_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        builder.add_shape_mesh(free_body, mesh=mesh)
        builder.add_shape_mesh(-1, mesh=mesh)
        articulated_body = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0)), mass=1.0)
        builder.add_shape_mesh(articulated_body, mesh=mesh)
        articulated_joint = builder.add_joint_revolute(
            -1,
            articulated_body,
            parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0)),
        )
        builder.add_articulation([articulated_joint])
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        reference_body_q = model.state().body_q
        path = self._path_certificate(reference_body_q, reference_body_q)
        motion_kind = path.motion_kind.numpy()
        motion_kind[articulated_body] = RIGID_BODY_PATH_UNKNOWN
        path.motion_kind.assign(motion_kind)

        oracle.scan(reference_body_q, path)

        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_retains_guards_for_colliding_articulated_dynamic_mesh(self) -> None:
        """Delegate an overlapping direct-free versus articulated-dynamic pair without suppressing its guard."""
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        free_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.19)))
        free_shape = builder.add_shape_mesh(free_body, mesh=mesh)
        articulated_body = builder.add_link(mass=1.0)
        articulated_shape = builder.add_shape_mesh(articulated_body, mesh=mesh)
        articulated_joint = builder.add_joint_revolute(-1, articulated_body)
        builder.add_articulation([articulated_joint])
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()
        path = self._path_certificate(state.body_q, state.body_q)
        motion_kind = path.motion_kind.numpy()
        motion_kind[articulated_body] = RIGID_BODY_PATH_UNKNOWN
        path.motion_kind.assign(motion_kind)

        oracle.scan(state.body_q, path)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.01)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        strict_guard = contacts.rigid_contact_is_strict_guard.numpy()[:count]
        pair = ((shape0 == free_shape) & (shape1 == articulated_shape)) | (
            (shape0 == articulated_shape) & (shape1 == free_shape)
        )
        self.assertGreater(np.count_nonzero(pair & (strict_guard != 0)), 0)

    def test_mesh_sdf_nonpenetration_oracle_tracks_nonfree_kinematic_anchor_transitions(self) -> None:
        """Activate an articulated mesh only while its live body flag makes it an anchor."""
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2)))
        builder.add_shape_mesh(source_body, mesh=mesh)
        anchor_body = builder.add_link(mass=1.0)
        builder.add_shape_mesh(anchor_body, mesh=mesh)
        anchor_joint = builder.add_joint_revolute(-1, anchor_body)
        builder.add_articulation([anchor_joint])
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            include_static_kinematic_pairs=True,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()

        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

        flags = model.body_flags.numpy()
        flags[anchor_body] = int(newton.BodyFlags.KINEMATIC)
        model.body_flags.assign(flags)
        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

        flags[anchor_body] = int(newton.BodyFlags.DYNAMIC)
        model.body_flags.assign(flags)
        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_routes_dynamic_kinematic_pairs(self) -> None:
        """Keep a movable mesh against a kinematic anchor under raw-oracle ownership."""
        device = wp.get_device("cpu")
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.19), wp.quat_identity()))
        target_body = builder.add_body()
        builder.body_flags[target_body] = int(newton.BodyFlags.KINEMATIC)
        builder.add_shape_mesh(source_body, mesh=mesh)
        builder.add_shape_mesh(target_body, mesh=mesh)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            include_static_kinematic_pairs=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()

        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)

        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_tracks_live_ownership_across_broad_phases(self) -> None:
        """Switch ownership without a stale role mask in every broad-phase and shape order."""
        device = wp.get_device("cpu")
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        config = newton.CollisionPipeline.SpeculativeContactConfig(
            max_speculative_extension=0.05,
            enforce_nonpenetration=True,
        )
        for broad_phase in ("nxn", "sap", "explicit"):
            for reverse_shape_order in (False, True):
                with self.subTest(broad_phase=broad_phase, reverse_shape_order=reverse_shape_order):
                    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                    builder.rigid_gap = 0.0
                    if reverse_shape_order:
                        target_body = builder.add_body()
                        builder.body_flags[target_body] = int(newton.BodyFlags.KINEMATIC)
                        target_shape = builder.add_shape_mesh(target_body, mesh=mesh)
                        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
                        source_shape = builder.add_shape_mesh(source_body, mesh=mesh)
                    else:
                        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
                        source_shape = builder.add_shape_mesh(source_body, mesh=mesh)
                        target_body = builder.add_body()
                        builder.body_flags[target_body] = int(newton.BodyFlags.KINEMATIC)
                        target_shape = builder.add_shape_mesh(target_body, mesh=mesh)
                    model = builder.finalize(device=device)
                    explicit_pairs = None
                    if broad_phase == "explicit":
                        explicit_pairs = wp.array(((source_shape, target_shape),), dtype=wp.vec2i, device=device)
                    pipeline = newton.CollisionPipeline(
                        model,
                        broad_phase=broad_phase,
                        shape_pairs_filtered=explicit_pairs,
                        reduce_contacts=False,
                        include_static_kinematic_pairs=True,
                        speculative_config=config,
                    )
                    oracle = self._claim_nonpenetration_oracle(pipeline)
                    state = model.state()

                    self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
                    self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
                    self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

                    flags = model.body_flags.numpy()
                    flags[target_body] = int(newton.BodyFlags.DYNAMIC)
                    model.body_flags.assign(flags)
                    self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
                    self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)
                    self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts, dt=0.01)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    self.assertGreater(count, 0)
                    self.assertGreater(
                        np.count_nonzero(contacts.rigid_contact_is_strict_guard.numpy()[:count]),
                        0,
                    )

                    flags[source_body] = int(newton.BodyFlags.KINEMATIC)
                    model.body_flags.assign(flags)
                    self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
                    self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
                    self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

                    flags[target_body] = int(newton.BodyFlags.KINEMATIC)
                    model.body_flags.assign(flags)
                    self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)
                    self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 0)
                    self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_deduplicates_reversed_explicit_pairs(self) -> None:
        """Give one retained feature exactly one update owner per refinement generation."""
        device = wp.get_device("cpu")
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.02, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()))
        source_shape = builder.add_shape_mesh(source_body, mesh=source_mesh)
        target_shape = builder.add_shape_mesh(-1, mesh=target_mesh)
        model = builder.finalize(device=device)
        explicit_pairs = wp.array(
            ((source_shape, target_shape), (target_shape, source_shape), (source_shape, target_shape)),
            dtype=wp.vec2i,
            device=device,
        )
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=explicit_pairs,
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        reference_body_q = model.state().body_q
        candidate = reference_body_q.numpy().copy()
        candidate[source_body, 2] = 0.0
        candidate_body_q = wp.array(candidate, dtype=wp.transform, device=device)

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 3)
        self.assertNotEqual(int(oracle.world_status.numpy()[0]) & CONTACT_ORACLE_VIOLATION, 0)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 1)
        first = (
            oracle.guard_contacts.rigid_contact_point_id.numpy()[:1].copy(),
            oracle.guard_contacts.rigid_contact_point0.numpy()[:1].copy(),
            oracle.guard_contacts.rigid_contact_point1.numpy()[:1].copy(),
            oracle.guard_contacts.rigid_contact_normal.numpy()[:1].copy(),
        )

        for _ in range(3):
            self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)
            self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 1)
            np.testing.assert_array_equal(oracle.guard_contacts.rigid_contact_point_id.numpy()[:1], first[0])
            np.testing.assert_array_equal(oracle.guard_contacts.rigid_contact_point0.numpy()[:1], first[1])
            np.testing.assert_array_equal(oracle.guard_contacts.rigid_contact_point1.numpy()[:1], first[2])
            np.testing.assert_array_equal(oracle.guard_contacts.rigid_contact_normal.numpy()[:1], first[3])

    def test_mesh_sdf_nonpenetration_oracle_detects_vertex_free_surface_crossing(self) -> None:
        """Reject an edge-only reference overlap in either shape order."""
        rod_mesh = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        slab_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        for reverse_shape_order in (False, True):
            with self.subTest(reverse_shape_order=reverse_shape_order):
                model, pipeline, oracle, _source_shape, _target_shape = self._make_nonpenetration_oracle(
                    rod_mesh,
                    slab_mesh,
                    reverse_shape_order=reverse_shape_order,
                )
                state = model.state()

                oracle.begin_refinement()
                self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

                self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
                self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
                self.assertEqual(
                    int(oracle.world_status.numpy()[0]),
                    CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
                )
                self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 1)
                self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_traverses_more_than_eight_surface_hits(self) -> None:
        """Classify a segment crossing ten closed surfaces without an arbitrary traversal cap."""
        source_mesh = newton.Mesh.create_box(0.8, 0.005, 0.005, compute_normals=False, compute_uvs=False)
        target_mesh = self._combine_box_meshes(
            tuple(((x, 0.0, 0.0), (0.02, 0.05, 0.05)) for x in (-0.4, -0.2, 0.0, 0.2, 0.4))
        )
        model, pipeline, oracle, _source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
        )
        state = model.state()

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
        self.assertEqual(
            int(oracle.world_status.numpy()[0]),
            CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
        )
        self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 1)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_detects_vertex_free_surface_crossing_on_cuda(self) -> None:
        """Preserve edge-only reference infeasibility classification on CUDA."""
        rod_mesh = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        slab_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        devices = get_cuda_test_devices()
        if not devices:
            self.skipTest("CUDA is unavailable")
        for device in devices:
            with self.subTest(device=device):
                model, pipeline, oracle, _source_shape, _target_shape = self._make_nonpenetration_oracle(
                    rod_mesh,
                    slab_mesh,
                    device=device,
                )
                state = model.state()

                oracle.begin_refinement()
                self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

                self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
                self.assertEqual(
                    int(oracle.world_status.numpy()[0]),
                    CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
                )
                self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 1)
                self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_rejects_stationary_target_pure_translation_transit(self) -> None:
        """Reject a pure translation that crosses a slab between two disjoint endpoints."""
        source_box = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        target_box = newton.Mesh.create_box(0.3, 0.3, 0.01, compute_normals=False, compute_uvs=False)
        source_triangles = source_box.indices.reshape(-1, 3)
        target_triangles = target_box.indices.reshape(-1, 3)
        variants = (
            (source_triangles, target_triangles, False),
            (source_triangles[::-1, ::-1], target_triangles, False),
            (source_triangles, target_triangles[::-1, ::-1], False),
            (source_triangles, target_triangles, True),
        )
        source_xform = wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity())
        certified_fractions = []
        for variant, (source_indices, target_indices, reverse_shape_order) in enumerate(variants):
            with self.subTest(variant=variant):
                source_mesh = newton.Mesh(source_box.vertices, source_indices, compute_inertia=False)
                target_mesh = newton.Mesh(target_box.vertices, target_indices, compute_inertia=False)
                model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
                    source_mesh,
                    target_mesh,
                    reverse_shape_order=reverse_shape_order,
                    source_xform=source_xform,
                )
                reference_body_q = model.state().body_q
                candidate_np = reference_body_q.numpy().copy()
                source_body = int(model.shape_body.numpy()[source_shape])
                candidate_np[source_body, 2] = -0.12
                candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

                oracle.begin_refinement()
                self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

                self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
                self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
                self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)
                certified_fraction = float(oracle.certified_path_fraction.numpy()[0])
                self.assertAlmostEqual(certified_fraction, 1.0 / 24.0, delta=2.0e-5)
                certified_fractions.append(certified_fraction)

                prefix_np = reference_body_q.numpy().copy()
                prefix_np[source_body, :3] += certified_fraction * (
                    candidate_np[source_body, :3] - prefix_np[source_body, :3]
                )
                prefix_body_q = wp.array(prefix_np, dtype=wp.transform, device=model.device)
                self._scan_nonpenetration_oracle(oracle, reference_body_q, prefix_body_q)
                self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
                self.assertEqual(float(oracle.certified_path_fraction.numpy()[0]), 1.0)

        np.testing.assert_allclose(certified_fractions, certified_fractions[0], rtol=0.0, atol=2.0e-6)

    def test_mesh_sdf_nonpenetration_oracle_rejects_thin_pure_translation_transit(self) -> None:
        """Reject a transverse interval shorter than the former parameter cutoff."""
        source_mesh = newton.Mesh.create_box(0.01, 0.01, 2.0e-7, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 2.0e-7, compute_normals=False, compute_uvs=False)
        source_xform = wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
        model, _pipeline, oracle, source_shape, target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=source_xform,
        )
        mesh_properties = model._shape_mesh_properties.numpy()
        self.assertNotEqual(int(mesh_properties[target_shape]) & int(MeshProperties.WATERTIGHT), 0)
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 2] = -0.1
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_rejects_coplanar_translation_transit(self) -> None:
        """Reject an in-plane transit between clear coplanar endpoints."""
        source_mesh = newton.Mesh(
            np.asarray(((-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (0.0, 0.2, 0.0)), dtype=np.float32),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            compute_inertia=False,
            is_solid=False,
        )
        target_mesh = newton.Mesh(
            np.asarray(((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)), dtype=np.float32),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            compute_inertia=False,
            is_solid=False,
        )
        model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 0] = 1.0
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
        certified_fraction = float(oracle.certified_path_fraction.numpy()[0])
        self.assertGreater(certified_fraction, 0.0)
        self.assertLess(certified_fraction, 0.5)

    def test_mesh_sdf_nonpenetration_oracle_selects_earliest_translation_obstacle(self) -> None:
        """Select the first transit obstacle independently of target triangle order."""
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.05, compute_normals=False, compute_uvs=False)
        source_xform = wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_identity())
        obstacle_orders = (
            (((0.0, 0.0, -0.2), (0.2, 0.2, 0.01)), ((0.0, 0.0, 0.2), (0.2, 0.2, 0.01))),
            (((0.0, 0.0, 0.2), (0.2, 0.2, 0.01)), ((0.0, 0.0, -0.2), (0.2, 0.2, 0.01))),
        )
        certified_fractions = []
        for obstacle_order in obstacle_orders:
            with self.subTest(obstacle_order=obstacle_order):
                target_mesh = self._combine_box_meshes(obstacle_order)
                model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
                    source_mesh,
                    target_mesh,
                    source_xform=source_xform,
                )
                reference_body_q = model.state().body_q
                candidate_np = reference_body_q.numpy().copy()
                source_body = int(model.shape_body.numpy()[source_shape])
                candidate_np[source_body, 2] = -0.6
                candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

                self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

                self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
                certified_fractions.append(float(oracle.certified_path_fraction.numpy()[0]))

        expected_fraction = (0.6 - (0.2 + 0.01 + 0.05)) / 1.2
        np.testing.assert_allclose(certified_fractions, expected_fraction, rtol=0.0, atol=2.0e-5)

    def test_mesh_sdf_nonpenetration_oracle_selects_earliest_translation_shape_pair(self) -> None:
        """Reduce translation prefixes across separate obstacle shape pairs."""
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.05, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_identity()))
        builder.add_shape_mesh(source_body, mesh=source_mesh)
        builder.add_shape_mesh(
            -1,
            mesh=target_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
        )
        builder.add_shape_mesh(
            -1,
            mesh=target_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()),
        )
        model = builder.finalize(device=wp.get_device("cpu"))
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        candidate_np[source_body, 2] = -0.6
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 2)
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
        expected_fraction = (0.6 - (0.2 + 0.01 + 0.05)) / 1.2
        self.assertAlmostEqual(float(oracle.certified_path_fraction.numpy()[0]), expected_fraction, delta=2.0e-5)

    def test_mesh_sdf_nonpenetration_oracle_translation_prefix_includes_shape_margins(self) -> None:
        """Shorten translation by the summed collision margin before surface contact."""
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.02, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        margin_cases = ((0.0, 0.0), (0.01, 0.015))
        certified_fractions = []
        for source_margin, target_margin in margin_cases:
            with self.subTest(source_margin=source_margin, target_margin=target_margin):
                model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
                    source_mesh,
                    target_mesh,
                    source_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()),
                    source_margin=source_margin,
                    target_margin=target_margin,
                )
                reference_body_q = model.state().body_q
                candidate_np = reference_body_q.numpy().copy()
                source_body = int(model.shape_body.numpy()[source_shape])
                candidate_np[source_body, 2] = -0.3
                candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

                self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

                self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
                certified_fractions.append(float(oracle.certified_path_fraction.numpy()[0]))

        motion_distance = 0.6
        expected = [
            (0.3 - (0.01 + 0.02 + source_margin + target_margin)) / motion_distance
            for source_margin, target_margin in margin_cases
        ]
        np.testing.assert_allclose(certified_fractions, expected, rtol=0.0, atol=2.0e-5)
        self.assertLess(certified_fractions[1], certified_fractions[0])

    def test_mesh_sdf_nonpenetration_oracle_rejects_nonparallel_triangle_crossing(self) -> None:
        """Reject transit normal to one face even when motion is tangent to the other."""
        source_mesh = newton.Mesh(
            np.asarray(((0.0, -0.25, -0.25), (0.0, 0.25, -0.25), (0.0, 0.0, 0.25)), dtype=np.float32),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            compute_inertia=False,
            is_solid=False,
        )
        source_xform = wp.transform(wp.vec3(-2.0, 0.0, 0.0), wp.quat_identity())
        target_vertices = np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32)
        target_indices = (
            np.asarray(((0, 1, 2),), dtype=np.int32),
            np.asarray(((0, 1, 2), (2, 1, 0)), dtype=np.int32),
        )
        for duplicated_reverse_face, indices in enumerate(target_indices):
            with self.subTest(duplicated_reverse_face=bool(duplicated_reverse_face)):
                target_mesh = newton.Mesh(target_vertices, indices, compute_inertia=False, is_solid=False)
                model, _pipeline, oracle, source_shape, target_shape = self._make_nonpenetration_oracle(
                    source_mesh,
                    target_mesh,
                    source_xform=source_xform,
                )
                if duplicated_reverse_face:
                    mesh_properties = model._shape_mesh_properties.numpy()
                    self.assertNotEqual(int(mesh_properties[target_shape]) & int(MeshProperties.WATERTIGHT), 0)
                reference_body_q = model.state().body_q
                candidate_np = reference_body_q.numpy().copy()
                source_body = int(model.shape_body.numpy()[source_shape])
                candidate_np[source_body, 0] = 2.0
                candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

                self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

                self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
                self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_rejects_nonparallel_plane_intersection_transit(self) -> None:
        """Reject finite triangles sliding into intersection along their two planes."""
        source_mesh = newton.Mesh(
            np.asarray(((0.0, 0.0, -0.25), (0.0, 0.0, 0.25), (0.5, 0.0, 0.0)), dtype=np.float32),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            compute_inertia=False,
            is_solid=False,
        )
        target_mesh = newton.Mesh(
            np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            compute_inertia=False,
            is_solid=False,
        )
        source_xform = wp.transform(wp.vec3(-2.0, 0.0, 0.0), wp.quat_identity())
        model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=source_xform,
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 0] = 2.0
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_reports_spatially_significant_small_rotation(self) -> None:
        """Measure rotation in surface displacement instead of quaternion-dot units."""
        source_mesh = newton.Mesh.create_box(1.0, 5.0e-5, 0.01, compute_normals=False, compute_uvs=False)
        target_box = newton.Mesh.create_box(5.0e-5, 5.0e-5, 0.005, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh(
            target_box.vertices + np.asarray((0.8, 8.0e-4, 0.0), dtype=np.float32),
            target_box.indices,
            compute_inertia=False,
        )
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 2.0e-3), dtype=np.float32
        )
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)
        self.assertGreater(float(oracle.world_min_separation.numpy()[0]), 0.0)

    def test_mesh_sdf_nonpenetration_oracle_reports_body_offset_rotation_arc(self) -> None:
        """Use the body-origin radius when an offset shape arcs through an obstacle."""
        source_mesh = newton.Mesh.create_box(1.0e-5, 1.0e-5, 1.0e-5, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(5.0e-5, 5.0e-5, 5.0e-5, compute_normals=False, compute_uvs=False)
        reference_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.02)
        candidate_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.02)
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(0.0), reference_rotation),
            source_shape_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
            target_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 3:7] = np.asarray(candidate_rotation, dtype=np.float32)
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

        midpoint_np = reference_body_q.numpy().copy()
        midpoint_np[source_body, 3:7] = np.asarray(wp.quat_identity(), dtype=np.float32)
        midpoint_body_q = wp.array(midpoint_np, dtype=wp.transform, device=model.device)
        self._scan_nonpenetration_oracle(oracle, midpoint_body_q, midpoint_body_q)
        self.assertEqual(
            int(oracle.world_status.numpy()[0]),
            CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
        )
        self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 1)

    def test_mesh_sdf_nonpenetration_oracle_rejects_equal_endpoint_full_turn(self) -> None:
        """Reject a certified full turn whose equal endpoints hide an obstacle crossing."""
        source_mesh = newton.Mesh.create_box(1.0e-4, 1.0e-4, 1.0e-4, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(2.0e-4, 2.0e-4, 2.0e-4, compute_normals=False, compute_uvs=False)
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_shape_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
            target_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        source_body = int(model.shape_body.numpy()[source_shape])
        origin_path_length = np.zeros(model.body_count, dtype=np.float32)
        angular_path_length = np.zeros(model.body_count, dtype=np.float32)
        angular_path_length[source_body] = 2.0 * np.pi
        motion_kind = np.full(model.body_count, RIGID_BODY_PATH_STATIONARY, dtype=np.uint8)
        motion_kind[source_body] = RIGID_BODY_PATH_BOUNDED

        self._scan_nonpenetration_oracle(
            oracle,
            reference_body_q,
            reference_body_q,
            origin_path_length=origin_path_length,
            angular_path_length=angular_path_length,
            motion_kind=motion_kind,
        )

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)

    def test_mesh_sdf_nonpenetration_oracle_rejects_equal_endpoint_origin_excursion(self) -> None:
        """Reject an out-and-back origin path whose equal endpoints hide a crossing."""
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.02, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        source_body = int(model.shape_body.numpy()[source_shape])
        origin_path_length = np.zeros(model.body_count, dtype=np.float32)
        origin_path_length[source_body] = 2.0
        angular_path_length = np.zeros(model.body_count, dtype=np.float32)
        motion_kind = np.full(model.body_count, RIGID_BODY_PATH_STATIONARY, dtype=np.uint8)
        motion_kind[source_body] = RIGID_BODY_PATH_BOUNDED

        self._scan_nonpenetration_oracle(
            oracle,
            reference_body_q,
            reference_body_q,
            origin_path_length=origin_path_length,
            angular_path_length=angular_path_length,
            motion_kind=motion_kind,
        )

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)

    def test_mesh_sdf_nonpenetration_oracle_accepts_clear_tiny_rotation(self) -> None:
        """Keep a clear tiny rotation live with a path-scaled angular envelope."""
        source_mesh = newton.Mesh.create_box(1.0, 5.0e-5, 5.0e-5, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(5.0e-5, 5.0e-5, 5.0e-5, compute_normals=False, compute_uvs=False)
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            target_xform=wp.transform(wp.vec3(0.8, 1.0e-3, 0.0), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1.0e-4), dtype=np.float32
        )
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_certifies_near_board_constant_twist(self) -> None:
        """Prove a small clear rotation beside an oblique board despite overlapping world AABBs."""
        source_mesh = newton.Mesh.create_box(0.1, 0.02, 0.02, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.5, 0.02, 0.02, compute_normals=False, compute_uvs=False)
        board_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.25 * np.pi)
        board_normal = np.asarray(wp.quat_rotate(board_rotation, wp.vec3(0.0, 1.0, 0.0)), dtype=np.float32)
        candidate_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 2.0e-3) * board_rotation

        for common_translation in (np.zeros(3, dtype=np.float32), np.asarray((2048.0, -1024.0, 32.0))):
            with self.subTest(common_translation=tuple(common_translation)):
                source_position = common_translation + 0.05 * board_normal
                model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
                    source_mesh,
                    target_mesh,
                    source_xform=wp.transform(wp.vec3(*source_position), board_rotation),
                    target_xform=wp.transform(wp.vec3(*common_translation), board_rotation),
                )
                reference_body_q = model.state().body_q
                candidate_np = reference_body_q.numpy().copy()
                source_body = int(model.shape_body.numpy()[source_shape])
                candidate_np[source_body, 3:7] = np.asarray(candidate_rotation, dtype=np.float32)
                candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

                self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

                self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
                self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
                self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
                self.assertEqual(float(oracle.certified_path_fraction.numpy()[0]), 1.0)

    def test_mesh_sdf_nonpenetration_oracle_certifies_factory_scale_tangent_twist(self) -> None:
        """Keep a tangent assembly pair live under one Factory-scale constant-twist step."""
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.01, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(0.0, 0.0, 0.02), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 0] += 1.0e-4
        candidate_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1.8e-3), dtype=np.float32
        )
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        bounded_motion_kind = np.full(model.body_count, RIGID_BODY_PATH_STATIONARY, dtype=np.uint8)
        bounded_motion_kind[source_body] = RIGID_BODY_PATH_BOUNDED
        self._scan_nonpenetration_oracle(
            oracle,
            reference_body_q,
            candidate_body_q,
            motion_kind=bounded_motion_kind,
        )
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_certifies_tolerance_valid_board_contact_twist(self) -> None:
        """Keep a shallow tolerance-valid board contact live during tangential motion."""
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.01, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(0.0, 0.0, 0.02 - 5.0e-7), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 0] += 1.0e-4
        candidate_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1.8e-3), dtype=np.float32
        )
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 0)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
        self.assertEqual(float(oracle.certified_path_fraction.numpy()[0]), 1.0)

    def test_mesh_sdf_nonpenetration_oracle_certifies_rotation_about_separating_axis(self) -> None:
        """Certify a near-tangent rotation whose separating-axis projection is constant."""
        source_half_extents = np.asarray((0.04, 0.04, 0.01), dtype=np.float32)
        target_half_extents = np.asarray((0.2, 0.2, 0.01), dtype=np.float32)
        reference_clearance = 5.0e-7
        rotation_angle = 0.02
        source_mesh = newton.Mesh.create_box(
            *source_half_extents,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
        )
        target_mesh = newton.Mesh.create_box(
            *target_half_extents,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
        )
        reference_cases = (
            (wp.quat_identity(), 1.0),
            (wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.4), -1.0),
        )
        for reference_rotation, candidate_sign in reference_cases:
            with self.subTest(reference_rotation=tuple(reference_rotation), candidate_sign=candidate_sign):
                rotated_axes = np.asarray(
                    [
                        wp.quat_rotate(reference_rotation, wp.vec3(1.0, 0.0, 0.0)),
                        wp.quat_rotate(reference_rotation, wp.vec3(0.0, 1.0, 0.0)),
                        wp.quat_rotate(reference_rotation, wp.vec3(0.0, 0.0, 1.0)),
                    ],
                    dtype=np.float32,
                )
                source_vertical_extent = float(np.dot(np.abs(rotated_axes[:, 2]), source_half_extents))
                source_height = target_half_extents[2] + source_vertical_extent + reference_clearance
                model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
                    source_mesh,
                    target_mesh,
                    source_xform=wp.transform(wp.vec3(0.0, 0.0, source_height), reference_rotation),
                )
                reference_body_q = model.state().body_q
                source_body = int(model.shape_body.numpy()[source_shape])
                candidate_np = reference_body_q.numpy().copy()
                candidate_np[source_body, 0] += 6.77e-4
                candidate_rotation = (
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rotation_angle) * reference_rotation
                )
                candidate_np[source_body, 3:7] = candidate_sign * np.asarray(candidate_rotation, dtype=np.float32)
                candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

                self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

                self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
                self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
                self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_rejects_midarc_crossing_about_tangent_axis(self) -> None:
        """Keep a clear-endpoint rotation unsafe when its midarc crosses the separating plane."""
        source_half_extents = np.asarray((0.05, 0.01, 1.0e-4), dtype=np.float32)
        target_half_extents = np.asarray((0.2, 0.2, 0.01), dtype=np.float32)
        endpoint_angle_offset = 0.02
        endpoint_clearance = 1.0e-6
        reference_angle = 0.5 * np.pi - endpoint_angle_offset
        candidate_angle = 0.5 * np.pi + endpoint_angle_offset
        reference_extent = source_half_extents[0] * abs(np.sin(reference_angle)) + source_half_extents[2] * abs(
            np.cos(reference_angle)
        )
        source_height = target_half_extents[2] + reference_extent + endpoint_clearance
        source_mesh = newton.Mesh.create_box(
            *source_half_extents,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
        )
        target_mesh = newton.Mesh.create_box(
            *target_half_extents,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
        )
        reference_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), reference_angle)
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(0.0, 0.0, float(source_height)), reference_rotation),
        )
        reference_body_q = model.state().body_q
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np = reference_body_q.numpy().copy()
        candidate_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), candidate_angle),
            dtype=np.float32,
        )
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)
        certified_fraction = float(oracle.certified_path_fraction.numpy()[0])
        self.assertGreater(certified_fraction, 0.0)
        self.assertLess(certified_fraction, 0.5)

        prefix_angle = reference_angle + certified_fraction * (candidate_angle - reference_angle)
        prefix_np = reference_body_q.numpy().copy()
        prefix_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), prefix_angle),
            dtype=np.float32,
        )
        prefix_body_q = wp.array(prefix_np, dtype=wp.transform, device=model.device)
        self._scan_nonpenetration_oracle(oracle, reference_body_q, prefix_body_q)
        self.assertEqual(
            int(oracle.world_status.numpy()[0]),
            0,
            msg=f"constant-twist certified prefix {certified_fraction} was not clean",
        )
        self.assertEqual(float(oracle.certified_path_fraction.numpy()[0]), 1.0)

        midpoint_np = reference_body_q.numpy().copy()
        midpoint_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi),
            dtype=np.float32,
        )
        midpoint_body_q = wp.array(midpoint_np, dtype=wp.transform, device=model.device)
        self._scan_nonpenetration_oracle(oracle, midpoint_body_q, midpoint_body_q)
        self.assertEqual(
            int(oracle.world_status.numpy()[0]),
            CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
        )

    def test_mesh_sdf_nonpenetration_oracle_certifies_tolerance_valid_twist_with_bounded_subdivision(self) -> None:
        """Use configured tolerance for a physically clear subdivided twist."""
        source_half_extents = np.asarray((0.01, 0.01, 1.0), dtype=np.float32)
        target_half_extents = np.asarray((0.01, 0.2, 1.2), dtype=np.float32)
        reference_angle = 0.25 * np.pi - 0.006
        candidate_angle = 0.25 * np.pi + 0.014
        minimum_clearance = 5.0e-8
        maximum_x_extent = (source_half_extents[0] + source_half_extents[1]) / np.sqrt(2.0)
        source_x = target_half_extents[0] + maximum_x_extent + minimum_clearance
        source_mesh = newton.Mesh.create_box(
            *source_half_extents,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
        )
        target_mesh = newton.Mesh.create_box(
            *target_half_extents,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
        )
        reference_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), reference_angle)
        candidate_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), candidate_angle)

        for reverse_shape_order in (False, True):
            with self.subTest(reverse_shape_order=reverse_shape_order):
                model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
                    source_mesh,
                    target_mesh,
                    reverse_shape_order=reverse_shape_order,
                    source_xform=wp.transform(wp.vec3(float(source_x), 0.0, 0.0), reference_rotation),
                )
                reference_body_q = model.state().body_q
                source_body = int(model.shape_body.numpy()[source_shape])
                candidate_np = reference_body_q.numpy().copy()
                candidate_np[source_body, 3:7] = np.asarray(candidate_rotation, dtype=np.float32)
                candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

                self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

                self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
                self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
                self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_rejects_rotation_crossing_clear_endpoints(self) -> None:
        """Reject a square rotating through a tight frame although its symmetric endpoints are clear."""
        source_mesh = newton.Mesh.create_box(0.025, 0.025, 0.01, compute_normals=False, compute_uvs=False)
        target_mesh = self._combine_box_meshes(
            (
                ((-0.04, 0.0, 0.0), (0.01, 0.05, 0.01)),
                ((0.04, 0.0, 0.0), (0.01, 0.05, 0.01)),
                ((0.0, -0.04, 0.0), (0.03, 0.01, 0.01)),
                ((0.0, 0.04, 0.0), (0.03, 0.01, 0.01)),
            )
        )
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
        )
        reference_body_q = model.state().body_q
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np = reference_body_q.numpy().copy()
        candidate_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi), dtype=np.float32
        )
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)
        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)

        self._scan_nonpenetration_oracle(oracle, candidate_body_q, candidate_body_q)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)

        midpoint_np = reference_body_q.numpy().copy()
        midpoint_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.25 * np.pi), dtype=np.float32
        )
        midpoint_body_q = wp.array(midpoint_np, dtype=wp.transform, device=model.device)
        self._scan_nonpenetration_oracle(oracle, midpoint_body_q, midpoint_body_q)
        self.assertEqual(
            int(oracle.world_status.numpy()[0]),
            CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
        )

    def test_mesh_sdf_nonpenetration_oracle_rejects_underreported_path(self) -> None:
        """Fail closed when a path certificate is shorter than its endpoint chord."""
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.02, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 0] += 0.1
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)
        origin_path_length = np.zeros(model.body_count, dtype=np.float32)
        angular_path_length = np.zeros(model.body_count, dtype=np.float32)
        motion_kind = np.full(model.body_count, RIGID_BODY_PATH_STATIONARY, dtype=np.uint8)
        motion_kind[source_body] = RIGID_BODY_PATH_LINEAR_TRANSLATION

        self._scan_nonpenetration_oracle(
            oracle,
            reference_body_q,
            candidate_body_q,
            origin_path_length=origin_path_length,
            angular_path_length=angular_path_length,
            motion_kind=motion_kind,
        )

        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)

    def test_mesh_sdf_nonpenetration_oracle_rejects_underreported_tiny_body_rotation(self) -> None:
        """Validate constant-twist angles directly even when surface displacement is submicron."""
        source_mesh = newton.Mesh.create_box(1.0e-6, 1.0e-6, 1.0e-6, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(1.0e-6, 1.0e-6, 1.0e-6, compute_normals=False, compute_uvs=False)
        model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 3:7] = np.asarray(
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.1), dtype=np.float32
        )
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)
        origin_path_length = np.zeros(model.body_count, dtype=np.float32)
        angular_path_length = np.zeros(model.body_count, dtype=np.float32)
        motion_kind = np.full(model.body_count, RIGID_BODY_PATH_STATIONARY, dtype=np.uint8)
        motion_kind[source_body] = RIGID_BODY_PATH_CONSTANT_TWIST

        self._scan_nonpenetration_oracle(
            oracle,
            reference_body_q,
            candidate_body_q,
            origin_path_length=origin_path_length,
            angular_path_length=angular_path_length,
            motion_kind=motion_kind,
        )

        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_SWEEP_INCOMPLETE)

    def test_mesh_sdf_nonpenetration_oracle_certifies_clear_translation_through_hole(self) -> None:
        """Accept clear axial motion even though the shape-level swept AABBs overlap."""

        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.05, compute_normals=False, compute_uvs=False)
        target_mesh = self._combine_box_meshes(
            (
                ((-0.15, 0.0, 0.0), (0.05, 0.2, 0.01)),
                ((0.15, 0.0, 0.0), (0.05, 0.2, 0.01)),
                ((0.0, -0.15, 0.0), (0.1, 0.05, 0.01)),
                ((0.0, 0.15, 0.0), (0.1, 0.05, 0.01)),
            )
        )
        source_xform = wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity())
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=source_xform,
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 2] = -0.2
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle._endpoint_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
        self.assertEqual(float(oracle.certified_path_fraction.numpy()[0]), 1.0)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_surface_crossing_supports_reflected_nonuniform_scale(self) -> None:
        """Keep the finite-segment parameter and normal valid through an affine target scale."""
        rod_mesh = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        slab_mesh = newton.Mesh.create_box(0.1, 0.4, 0.02, compute_normals=False, compute_uvs=False)
        model, _pipeline, oracle, _source_shape, _target_shape = self._make_nonpenetration_oracle(
            rod_mesh,
            slab_mesh,
            target_scale=wp.vec3(2.0, 0.5, -0.5),
        )
        state = model.state()

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

        self.assertEqual(
            int(oracle.world_status.numpy()[0]),
            CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
        )
        self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 1)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_surface_crossing_is_topology_order_invariant(self) -> None:
        """Classify reference overlap independently of edge order and triangle winding."""
        source_box = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        target_box = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        source_triangles = source_box.indices.reshape(-1, 3)
        target_triangles = target_box.indices.reshape(-1, 3)
        variants = (
            (source_triangles, target_triangles),
            (source_triangles[::-1, ::-1], target_triangles),
            (source_triangles, target_triangles[:, ::-1]),
            (source_triangles[::-1, ::-1], target_triangles[:, ::-1]),
        )
        for variant, (source_indices, target_indices) in enumerate(variants):
            with self.subTest(variant=variant):
                source_mesh = newton.Mesh(
                    source_box.vertices,
                    source_indices,
                    compute_inertia=False,
                )
                target_mesh = newton.Mesh(
                    target_box.vertices,
                    target_indices,
                    compute_inertia=False,
                )
                model, _pipeline, oracle, _source_shape, _target_shape = self._make_nonpenetration_oracle(
                    source_mesh,
                    target_mesh,
                )
                state = model.state()

                oracle.begin_refinement()
                self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

                self.assertEqual(
                    int(oracle.world_status.numpy()[0]),
                    CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
                )
                self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 1)
                self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_uses_raw_mesh_when_texture_disagrees(self) -> None:
        """Certify vertex containment from raw geometry instead of an attached mismatched texture SDF."""
        devices = get_cuda_test_devices()
        if not devices:
            self.skipTest("Texture SDF construction requires CUDA")
        for device in devices:
            with self.subTest(device=device):
                source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.02, compute_normals=False, compute_uvs=False)
                target_mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
                mismatched_mesh = newton.Mesh.create_box(
                    0.005,
                    0.005,
                    0.005,
                    compute_normals=False,
                    compute_uvs=False,
                )
                mismatched_mesh.build_sdf(
                    device=device,
                    max_resolution=16,
                    narrow_band_range=(-0.05, 0.05),
                    margin=0.05,
                )
                target_mesh.sdf = mismatched_mesh.sdf
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                builder.rigid_gap = 0.0
                source_body = builder.add_body(xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()))
                builder.add_shape_mesh(source_body, mesh=source_mesh)
                target_shape = builder.add_shape_mesh(-1, mesh=target_mesh)
                model = builder.finalize(device=device)
                pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase="nxn",
                    reduce_contacts=False,
                    speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                        max_speculative_extension=0.05,
                        enforce_nonpenetration=True,
                    ),
                )
                oracle = self._claim_nonpenetration_oracle(pipeline)
                self.assertGreaterEqual(int(model._shape_sdf_index.numpy()[target_shape]), 0)
                state = model.state()
                candidate_np = state.body_q.numpy().copy()
                candidate_np[source_body, 0] = 0.0
                candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=device)

                oracle.begin_refinement()
                self._refine_nonpenetration_oracle(oracle, state.body_q, candidate_body_q)

                self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
                self.assertAlmostEqual(float(oracle.world_min_separation.numpy()[0]), -0.08, delta=2.0e-5)
                self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 1)

    def test_mesh_sdf_nonpenetration_oracle_surface_tangent_is_clean(self) -> None:
        """Do not classify a coplanar resting face as transverse penetration."""
        source_mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        slab_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        source_xform = wp.transform(wp.vec3(0.0, 0.0, 0.06), wp.quat_identity())
        model, pipeline, oracle, _source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            slab_mesh,
            source_xform=source_xform,
        )
        state = model.state()

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertGreaterEqual(float(oracle.world_min_separation.numpy()[0]), -1.0e-6)

    def test_mesh_sdf_nonpenetration_oracle_tangential_closed_box_slide_off_edge_is_clean(self) -> None:
        """Permit a closed box to slide off a slab without freezing on its side faces."""
        source_mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        source_xform = wp.transform(wp.vec3(0.1, 0.0, 0.06), wp.quat_identity())
        model, pipeline, oracle, source_shape, target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=source_xform,
        )
        mesh_properties = model._shape_mesh_properties.numpy()
        self.assertNotEqual(int(mesh_properties[source_shape]) & int(MeshProperties.WATERTIGHT), 0)
        self.assertNotEqual(int(mesh_properties[target_shape]) & int(MeshProperties.WATERTIGHT), 0)
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 0] = 0.3
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_rotated_support_frame_box_slide_is_clean(self) -> None:
        """Certify a near-support slide in the shapes' rotated frame."""
        source_mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.deg2rad(37.0)))
        source_position = wp.quat_rotate(rotation, wp.vec3(0.1, 0.0, 0.06001))
        target_xform = wp.transform(wp.vec3(0.0), rotation)
        model, pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=wp.transform(source_position, rotation),
            target_xform=target_xform,
        )
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, :3] = np.asarray(
            wp.quat_rotate(rotation, wp.vec3(0.3, 0.0, 0.06001)), dtype=np.float32
        )
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_rejects_scaled_projection_tolerance_alias(self) -> None:
        """Convert world tolerance along each reflected, nonuniform target-local axis."""
        source_mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        source_xform = wp.transform(wp.vec3(20.05 - 2.0e-5, 0.0, 0.12), wp.quat_identity())
        model, pipeline, oracle, source_shape, target_shape = self._make_nonpenetration_oracle(
            source_mesh,
            target_mesh,
            source_xform=source_xform,
            target_scale=wp.vec3(-100.0, 1.0, 1.0),
        )
        mesh_properties = model._shape_mesh_properties.numpy()
        self.assertNotEqual(int(mesh_properties[source_shape]) & int(MeshProperties.WATERTIGHT), 0)
        self.assertNotEqual(int(mesh_properties[target_shape]) & int(MeshProperties.WATERTIGHT), 0)
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        source_body = int(model.shape_body.numpy()[source_shape])
        candidate_np[source_body, 2] = -0.12
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_uses_raw_edges_not_sdf_simplified_edges(self) -> None:
        """Retain surface coverage when the ordinary SDF edge set omits the crossing edge."""
        rod_mesh = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        rod_mesh._collision_edges = np.empty((0, 2), dtype=np.int32)
        slab_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        model, _pipeline, oracle, source_shape, _target_shape = self._make_nonpenetration_oracle(
            rod_mesh,
            slab_mesh,
        )
        self.assertEqual(int(model.shape_edge_range.numpy()[source_shape, 1]), 0)
        state = model.state()

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

        self.assertEqual(
            int(oracle.world_status.numpy()[0]),
            CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
        )
        self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 1)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_surface_failure_is_bounded_and_deterministic(self) -> None:
        """Collapse many reference edge hits to one repeatable pair state."""
        rod_mesh = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        slab_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        model, _pipeline, oracle, _source_shape, _target_shape = self._make_nonpenetration_oracle(
            rod_mesh,
            slab_mesh,
        )
        state = model.state()
        snapshots = []
        for _ in range(2):
            oracle.begin_refinement()
            self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)
            self.assertEqual(
                int(oracle.world_status.numpy()[0]),
                CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_REFERENCE_INFEASIBLE,
            )
            self.assertEqual(int(oracle._reference_pair_state.numpy()[0]), 1)
            count = int(oracle.guard_contacts.rigid_contact_count.numpy()[0])
            self.assertEqual(count, 0)
            snapshots.append(
                (
                    oracle.world_status.numpy().copy(),
                    oracle.world_min_separation.numpy().copy(),
                    oracle.guard_contacts.rigid_contact_count.numpy().copy(),
                )
            )

        for first, second in zip(snapshots[0], snapshots[1], strict=True):
            np.testing.assert_array_equal(first, second)

    def test_mesh_sdf_nonpenetration_oracle_localizes_pure_translation_failure(self) -> None:
        """Localize a swept surface crossing to one world."""

        def make_tangent_stack(facet_count: int) -> newton.Mesh:
            vertices = []
            indices = []
            slope = 5.0e-6
            tangent_x = np.asarray((1.0, 0.0, -slope), dtype=np.float32)
            tangent_y = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
            for facet, x_position in enumerate(np.linspace(-0.8, 0.8, facet_count)):
                center = np.asarray((x_position, 0.0, 0.0), dtype=np.float32)
                triangle = np.stack(
                    (
                        center - 0.03 * tangent_x - 0.1 * tangent_y,
                        center + 0.03 * tangent_x - 0.1 * tangent_y,
                        center + 0.2 * tangent_y,
                    )
                )
                if facet == 0:
                    triangle = triangle[[0, 2, 1]]
                base = len(vertices)
                vertices.extend(triangle)
                indices.append((base, base + 1, base + 2))
            return newton.Mesh(
                np.asarray(vertices),
                np.asarray(indices, dtype=np.int32),
                compute_inertia=False,
                is_solid=False,
            )

        source_mesh = newton.Mesh(
            np.asarray(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.001, 0.0), (1.0, 0.001, 0.0))),
            np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32),
            compute_inertia=False,
            is_solid=False,
        )
        world_builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        world_builder.rigid_gap = 0.0
        source_body = world_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        world_builder.add_shape_mesh(source_body, mesh=source_mesh)
        world_builder.add_shape_mesh(-1, mesh=make_tangent_stack(8))
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_world(world_builder)
        builder.add_world(world_builder)
        model = builder.finalize(device=wp.get_device("cpu"))
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        reference_body_q = model.state().body_q
        reference_np = reference_body_q.numpy().copy()
        candidate_np = reference_np.copy()
        world_zero_body = int(np.flatnonzero(model.body_world.numpy() == 0)[0])
        candidate_np[world_zero_body, 2] = 0.0
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=model.device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        np.testing.assert_array_equal(
            oracle.world_status.numpy(),
            (CONTACT_ORACLE_VIOLATION, 0),
        )
        np.testing.assert_array_equal(reference_body_q.numpy(), reference_np)
        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)
        self.assertTrue(np.isinf(float(oracle.world_min_separation.numpy()[1])))

    def test_mesh_sdf_nonpenetration_oracle_fails_closed_on_singular_scale(self) -> None:
        """Reject a candidate pair whose mesh affine map cannot be inverted."""
        rod_mesh = newton.Mesh.create_box(0.02, 0.02, 0.1, compute_normals=False, compute_uvs=False)
        slab_mesh = newton.Mesh.create_box(0.2, 0.2, 0.01, compute_normals=False, compute_uvs=False)
        model, _pipeline, oracle, _source_shape, _target_shape = self._make_nonpenetration_oracle(
            rod_mesh,
            slab_mesh,
            target_scale=wp.vec3(1.0, 1.0, 0.0),
        )
        state = model.state()

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

        self.assertEqual(int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_INCOMPLETE)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

    def test_mesh_sdf_nonpenetration_oracle_transports_owner_normal_to_reference_pose(self) -> None:
        """Store candidate anchors locally while transporting the owner normal to the reference frame."""
        device = wp.get_device("cpu")
        source_mesh = newton.Mesh.create_box(0.02, 0.02, 0.02, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.2, 0.2, 0.05, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        source_shape = builder.add_shape_mesh(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity()),
            mesh=source_mesh,
        )
        target_body = builder.add_body()
        target_shape = builder.add_shape_mesh(target_body, mesh=target_mesh)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        reference_body_q = model.state().body_q
        candidate_body_q = wp.array(
            [
                wp.transform(
                    wp.vec3(0.0),
                    wp.quat(0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)),
                )
            ],
            dtype=wp.transform,
            device=device,
        )

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        self.assertEqual(
            int(oracle.world_status.numpy()[0]), CONTACT_ORACLE_VIOLATION | CONTACT_ORACLE_SWEEP_INCOMPLETE
        )
        self.assertAlmostEqual(float(oracle.world_min_separation.numpy()[0]), -0.03, delta=2.0e-5)
        count = int(oracle.guard_contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 1)
        owners = oracle.guard_contacts.rigid_contact_normal_owner.numpy()[:count]
        owner_matches = np.flatnonzero(owners == 1)
        self.assertEqual(len(owner_matches), 1)
        contact_index = int(owner_matches[0])
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_shape0.numpy()[contact_index]), source_shape)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_shape1.numpy()[contact_index]), target_shape)
        np.testing.assert_allclose(
            oracle.guard_contacts.rigid_contact_normal.numpy()[contact_index],
            (0.0, 0.0, -1.0),
            rtol=0.0,
            atol=2.0e-5,
        )
        point1_target = oracle.guard_contacts.rigid_contact_point1.numpy()[contact_index]
        self.assertAlmostEqual(float(point1_target[2]), 0.05, delta=2.0e-5)

        point0_world = oracle.guard_contacts.rigid_contact_point0.numpy()[contact_index]
        point1_world = np.array((point1_target[2], point1_target[1], -point1_target[0]), dtype=np.float32)
        candidate_normal = np.array((-1.0, 0.0, 0.0), dtype=np.float32)
        separation = float(np.dot(point1_world - point0_world, candidate_normal))
        self.assertAlmostEqual(separation, -0.03, delta=2.0e-5)

    def test_mesh_sdf_nonpenetration_oracle_ignores_inactive_shapes_in_candidate_broadphase(self) -> None:
        """Do not let non-mesh shapes create false pairs or exhaust oracle pair capacity."""
        device = wp.get_device("cpu")
        for broad_phase in ("nxn", "sap"):
            with self.subTest(broad_phase=broad_phase):
                mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                builder.rigid_gap = 0.0
                left_body = builder.add_body(xform=wp.transform(wp.vec3(-10.0, 0.0, 0.0), wp.quat_identity()))
                builder.add_shape_mesh(left_body, mesh=mesh)
                builder.add_shape_mesh(
                    -1,
                    mesh=mesh,
                    xform=wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity()),
                )
                for _ in range(16):
                    sphere_body = builder.add_body()
                    builder.add_shape_sphere(sphere_body, radius=0.1)
                model = builder.finalize(device=device)
                pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase=broad_phase,
                    shape_pairs_max=1,
                    speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                        max_speculative_extension=0.05,
                        enforce_nonpenetration=True,
                    ),
                )
                oracle = self._claim_nonpenetration_oracle(pipeline)
                state = model.state()

                self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)

                self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 0)
                self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
                self.assertTrue(np.isinf(float(oracle.world_min_separation.numpy()[0])))

    def test_mesh_sdf_nonpenetration_oracle_retains_non_domain_collision_path(self) -> None:
        """Leave mesh-primitive pairs to the ordinary collision pipeline."""
        device = wp.get_device("cpu")
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        near_mesh_body = builder.add_body()
        sphere_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity()))
        near_mesh_shape = builder.add_shape_mesh(near_mesh_body, mesh=mesh)
        builder.add_shape_mesh(
            -1,
            mesh=mesh,
            xform=wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity()),
        )
        sphere_shape = builder.add_shape_sphere(sphere_body, radius=0.05)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()

        oracle.begin_refinement()
        self._refine_nonpenetration_oracle(oracle, state.body_q, state.body_q)

        self.assertEqual(int(oracle.world_status.numpy()[0]), 0)
        self.assertEqual(int(oracle.guard_contacts.rigid_contact_count.numpy()[0]), 0)

        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.01)
        count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        pair_found = ((shape0 == near_mesh_shape) & (shape1 == sphere_shape)) | (
            (shape0 == sphere_shape) & (shape1 == near_mesh_shape)
        )
        self.assertTrue(np.any(pair_found))

    def test_mesh_sdf_nonpenetration_oracle_fails_closed_on_pair_capacity(self) -> None:
        """Mark every affected world incomplete when candidate-pair capacity is exhausted."""
        device = wp.get_device("cpu")
        mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        builder.add_shape_mesh(-1, mesh=mesh)
        for _ in range(2):
            body = builder.add_body()
            builder.add_shape_mesh(body, mesh=mesh)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            shape_pairs_max=1,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        state = model.state()

        self._scan_nonpenetration_oracle(oracle, state.body_q, state.body_q)

        status = int(oracle.world_status.numpy()[0])
        self.assertEqual(
            status & (CONTACT_ORACLE_INCOMPLETE | CONTACT_ORACLE_CAPACITY),
            CONTACT_ORACLE_INCOMPLETE | CONTACT_ORACLE_CAPACITY,
        )

    def test_mesh_sdf_nonpenetration_oracle_localizes_invalid_candidate_pose(self) -> None:
        """Fail only the world containing an invalid candidate body pose."""
        device = wp.get_device("cpu")
        source_mesh = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_normals=False, compute_uvs=False)
        target_mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        world_builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        world_builder.rigid_gap = 0.0
        source_body = world_builder.add_body(xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()))
        world_builder.add_shape_mesh(source_body, mesh=source_mesh)
        world_builder.add_shape_mesh(-1, mesh=target_mesh)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_world(world_builder)
        builder.add_world(world_builder)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=False,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        oracle = self._claim_nonpenetration_oracle(pipeline)
        reference_body_q = model.state().body_q
        candidate_np = reference_body_q.numpy().copy()
        world_zero_body = int(np.flatnonzero(model.body_world.numpy() == 0)[0])
        candidate_np[world_zero_body, 0] = np.nan
        candidate_body_q = wp.array(candidate_np, dtype=wp.transform, device=device)

        self._scan_nonpenetration_oracle(oracle, reference_body_q, candidate_body_q)

        status = oracle.world_status.numpy()
        self.assertEqual(int(status[0]), CONTACT_ORACLE_INCOMPLETE)
        self.assertEqual(int(status[1]), 0)
        self.assertTrue(np.isinf(float(oracle.world_min_separation.numpy()[0])))
        self.assertTrue(np.isinf(float(oracle.world_min_separation.numpy()[1])))

    def test_mesh_sdf_endpoint_guards_follow_oracle_ownership_with_reduction(self) -> None:
        """Keep reduced contacts compact and suppress endpoint guards for oracle-owned pairs."""
        device = wp.get_device("cpu")
        held_mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        fixed_mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_normals=False, compute_uvs=False)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.rigid_gap = 0.0
        held_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.19), wp.quat_identity()))
        builder.add_shape_mesh(held_body, mesh=held_mesh)
        builder.add_shape_mesh(-1, mesh=fixed_mesh)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=True,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(
                max_speculative_extension=0.05,
                enforce_nonpenetration=True,
            ),
        )
        contacts = pipeline.contacts()

        pipeline.collide(model.state(), contacts, dt=0.01)

        count = int(contacts.rigid_contact_count.numpy()[0])
        guard_kinds = contacts.rigid_contact_is_strict_guard.numpy()[:count]
        self.assertGreater(np.count_nonzero(guard_kinds == 1), 0)
        self.assertEqual(np.count_nonzero(guard_kinds == 2), 0)

        oracle = self._claim_nonpenetration_oracle(pipeline)
        self.assertEqual(int(oracle._eligible_shape.numpy()[0]), 1)
        self.assertEqual(int(oracle._anchor_capable_shape.numpy()[1]), 1)
        contacts.clear()
        pipeline.collide(model.state(), contacts, dt=0.01)

        self.assertEqual(int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0]), 1)
        count = int(contacts.rigid_contact_count.numpy()[0])
        guard_kinds = contacts.rigid_contact_is_strict_guard.numpy()[:count]
        self.assertEqual(np.count_nonzero(guard_kinds == 2), 0)

    def test_long_edge_finds_and_reduces_separated_narrow_sdf_wells(self) -> None:
        """Retain the measured long-edge witness when one whole-edge search misses both wells."""
        device = wp.get_device("cpu")
        reducer = GlobalContactReducer(capacity=64, device=device, deterministic=True)
        candidate_mask = wp.zeros(32, dtype=wp.int32, device=device)
        candidate_t = wp.zeros(32, dtype=wp.float32, device=device)
        candidate_depth = wp.zeros(32, dtype=wp.float32, device=device)
        legacy_result = wp.zeros(1, dtype=wp.vec2f, device=device)
        wp.launch(
            _two_well_edge_candidates_kernel,
            dim=1,
            inputs=[
                wp.empty(0, dtype=wp.float32, device=device),
                candidate_mask,
                candidate_t,
                candidate_depth,
                legacy_result,
                reducer.get_data_struct(),
            ],
            device=device,
        )

        witness_t = 0.341502
        other_t = 0.62
        legacy_t = float(legacy_result.numpy()[0, 0])
        legacy_depth = float(legacy_result.numpy()[0, 1])
        raw_t = candidate_t.numpy()[candidate_mask.numpy().astype(bool)]
        self.assertGreater(legacy_depth, 0.0)
        self.assertGreater(abs(legacy_t - witness_t), 0.01)
        self.assertEqual(len(raw_t), 2)
        self.assertTrue(np.any(np.isclose(raw_t, witness_t, rtol=0.0, atol=0.01)))
        self.assertTrue(np.any(np.isclose(raw_t, other_t, rtol=0.0, atol=0.01)))

        packed_winners = reducer.ht_values.numpy()
        retained_ids = np.unique(packed_winners[packed_winners != 0] & np.uint64((1 << 20) - 1))
        retained_ids = retained_ids[retained_ids != 0].astype(np.int64)
        retained_t = reducer.position_depth.numpy()[retained_ids, 0] / 0.04944
        self.assertEqual(len(retained_t), 2)
        self.assertTrue(np.any(np.isclose(retained_t, witness_t, rtol=0.0, atol=0.01)))
        self.assertTrue(np.any(np.isclose(retained_t, other_t, rtol=0.0, atol=0.01)))

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

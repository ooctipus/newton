# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_contact import (
    _sdf_rsqrt_rn,
    compute_block_counts_from_weights,
    mesh_sdf_contact_search_precision,
)
from newton.tests.unittest_utils import get_cuda_test_devices, get_test_devices


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


class TestSDFContact(unittest.TestCase):
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
                self.assertGreater(int(pipeline.narrow_phase.mesh_sdf_work_state.numpy()[4]), 0)
                segments_used = int(pipeline.narrow_phase.mesh_sdf_work_state.numpy()[0])
                self.assertGreater(segments_used, 1)

                def assert_same_contacts(expected, actual) -> None:
                    for lhs, rhs in zip(expected, actual, strict=True):
                        if lhs.ndim == 1:
                            np.testing.assert_array_equal(lhs, rhs)
                        else:
                            np.testing.assert_allclose(lhs, rhs, rtol=1.0e-5, atol=1.0e-6)

                # Hit records that do not fit trip the same fallback as segments that do not fit.
                hit_capacity = pipeline.narrow_phase.mesh_sdf_hit_capacity
                pipeline.narrow_phase.mesh_sdf_hit_capacity = 0
                assert_same_contacts(split_contacts, collide(pipeline, state, contacts))
                self.assertEqual(int(pipeline.narrow_phase.mesh_sdf_work_state.numpy()[1]), 1)
                pipeline.narrow_phase.mesh_sdf_hit_capacity = hit_capacity

                pipeline.narrow_phase.mesh_sdf_segment_capacity = 0
                assert_same_contacts(split_contacts, collide(pipeline, state, contacts))
                self.assertEqual(int(pipeline.narrow_phase.mesh_sdf_work_state.numpy()[1]), 1)

                # A partial overflow exports the complete contexts through the two-stage path and lets the
                # fallback redo only the incomplete ones: same contacts, hits from both paths.
                pipeline.narrow_phase.mesh_sdf_segment_capacity = max(1, segments_used // 2)
                assert_same_contacts(split_contacts, collide(pipeline, state, contacts))
                work_state = pipeline.narrow_phase.mesh_sdf_work_state.numpy()
                self.assertEqual(int(work_state[1]), 1)
                self.assertGreater(int(work_state[4]), 0)
                self.assertGreater(int(pipeline.narrow_phase.mesh_sdf_context_incomplete.numpy().sum()), 0)

    def test_split_mesh_sdf_dedicated_work_buffer(self) -> None:
        """A dedicated work-buffer capacity matches the aliased path and is independent of max_triangle_pairs."""
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
                common = {
                    "broad_phase": "nxn",
                    "deterministic": True,
                    "reduce_contacts": True,
                    "rigid_contact_max": 128,
                }

                def reduced(pipeline, active_model) -> np.ndarray:
                    contacts = pipeline.contacts()
                    pipeline.collide(active_model.state(), contacts)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    self.assertGreater(count, 0)
                    self.assertEqual(int(pipeline.narrow_phase.mesh_sdf_work_state.numpy()[1]), 0)
                    rows = np.column_stack(
                        (
                            contacts.rigid_contact_shape0.numpy()[:count],
                            contacts.rigid_contact_shape1.numpy()[:count],
                            contacts.rigid_contact_point0.numpy()[:count],
                            contacts.rigid_contact_normal.numpy()[:count],
                        )
                    )
                    return rows[np.lexsort(rows.T[::-1])]

                aliased = newton.CollisionPipeline(model, max_triangle_pairs=4096, **common)
                dedicated = newton.CollisionPipeline(model, max_triangle_pairs=64, mesh_sdf_work_segments=64, **common)
                self.assertEqual(dedicated.narrow_phase.mesh_sdf_segment_capacity, 64)
                self.assertGreater(
                    dedicated.narrow_phase.mesh_sdf_segment_capacity,
                    aliased.narrow_phase.mesh_sdf_segment_capacity * 64 // 4096,
                )
                np.testing.assert_allclose(reduced(dedicated, model), reduced(aliased, model), rtol=1.0e-5, atol=1.0e-6)
                with self.assertRaises(ValueError):
                    newton.CollisionPipeline(model, mesh_sdf_work_segments=-1, **common)

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

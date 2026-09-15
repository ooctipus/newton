# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check pair-private terrain ownership using the existing finite fixtures."""

import importlib
import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction import (
    compute_voxel_index,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from newton._src.geometry.contact_reduction_global import BETA_THRESHOLD, compute_effective_radius
from newton._src.sim.collide import write_contact
from tools.fpgs_bench import test_heightfield_finite_geometry as existing


def module():
    """Keep the missing-feature regression local to this focused suite."""
    return importlib.import_module("newton._src.geometry.heightfield_pair_terrain")


def small_fixture(device):
    """Reuse the existing finite lifecycle scene, with bounded mutable heights."""
    terrain = newton.Heightfield(
        np.full((3, 3), 0.5, np.float32), nrow=3, ncol=3, hx=1.0, hy=1.0, min_z=-0.02, max_z=0.02
    )
    builder = newton.ModelBuilder()
    config = builder.ShapeConfig(margin=0.01, gap=0.01)
    builder.add_shape_heightfield(heightfield=terrain, cfg=config)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.115), wp.quat_identity()))
    builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=config)
    other = builder.add_body(xform=wp.transform(wp.vec3(0.6, 0.0, 0.115), wp.quat_identity()))
    builder.add_shape_sphere(body=other, radius=0.1, cfg=config)
    odd = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.6, 0.115), wp.quat_identity()))
    builder.add_shape_box(body=odd, hx=0.1, hy=0.1, hz=0.1, cfg=config)
    model = builder.finalize(device=device)
    return model, model.state(), wp.array([[0, 1], [0, 2], [0, 3]], dtype=wp.vec2i, device=device)


def make_pipeline(model, pairs, enabled, *, local_capacity=64, deterministic=False, reduce=True):
    """Construct the ordinary production pipeline without a test-only launcher."""
    forced = nullcontext()
    if enabled and local_capacity != 64:
        factory = module().create_pair_kernels

        def limited(writer_func, local_capacity=64):
            return factory(writer_func, local_capacity=1)

        forced = patch.object(module(), "create_pair_kernels", side_effect=limited)
    with (
        patch.dict(
            os.environ,
            NEWTON_HEIGHTFIELD_CELL_REJECT="1",
            NEWTON_HEIGHTFIELD_FINITE_QUERY="1",
            NEWTON_HEIGHTFIELD_GEOMETRIC_CULL="1",
            NEWTON_HEIGHTFIELD_PAIR_REDUCER=str(int(enabled)),
        ),
        forced,
    ):
        return newton.CollisionPipeline(
            model,
            shape_pairs_filtered=pairs,
            reduce_contacts=reduce,
            deterministic=deterministic,
            rigid_contact_max=32768 if model.shape_count > 8 else 64,
            max_triangle_pairs=32768 if model.shape_count > 8 else 64,
        )


def public_snapshot(pipeline, state, contacts):
    """Reuse the existing physical reader, not retired triangle-buffer identity."""
    result = existing.snapshot(pipeline, state, contacts)
    result["envelope"] = score_envelope(pipeline, result)
    return result


def score_envelope(pipeline, snapshot):
    """Reuse original normal/spatial/voxel scoring on reconstructed public data."""
    model = pipeline.model
    lower = wp.vec3(*model.shape_collision_aabb_lower.numpy()[0])
    upper = wp.vec3(*model.shape_collision_aabb_upper.numpy()[0])
    resolution = wp.vec3i(*pipeline.narrow_phase.shape_voxel_resolution.numpy()[0])
    pose = pipeline.geom_transform.numpy()[0]
    inverse = wp.transform_inverse(wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])))
    types, data = model.shape_type.numpy(), pipeline.geom_data.numpy()
    radii = [compute_effective_radius(int(kind), wp.vec4(*scale)) for kind, scale in zip(types, data, strict=True)]
    beta = BETA_THRESHOLD * float(wp.length(upper - lower))
    scores = {}
    for shape, point, normal_values, distance in zip(
        snapshot["shape"], snapshot["point"], snapshot["normal"], snapshot["distance"], strict=True
    ):
        # Invert the original write_contact equations, including a sphere's
        # effective radius. Terrain is shape0 in both existing fixture builders.
        depth = float(distance) - radii[0] - radii[shape]
        normal = wp.vec3(*normal_values)
        center = wp.vec3(*point) + normal * (0.5 * depth + radii[0])
        normal_bin = int(get_slot(normal))
        voxel = max(0, min(99, int(compute_voxel_index(wp.transform_point(inverse, center), lower, upper, resolution))))
        values = [(normal_bin, 6, -depth), (20 + voxel // 7, voxel % 7, -depth)]
        if depth < beta:
            projected = project_point_to_plane(normal_bin, center)
            values.extend((normal_bin, k, float(wp.dot(projected, get_spatial_direction_2d(k)))) for k in range(6))
        for bin_id, slot, value in values:
            key = (int(shape), bin_id, slot)
            scores[key] = max(scores.get(key, -np.inf), value)
    return scores


def compare_public(test, before, after):
    """Keep original geometry tolerance while allowing order and score ties."""
    left, right = set(map(int, before["shape"])), set(map(int, after["shape"]))
    test.assertEqual(left, right, "Lost or invented contacting shapes")
    for shape in left:
        a, b = before["shape"] == shape, after["shape"] == shape
        test.assertLessEqual(abs(float(before["distance"][a].min() - after["distance"][b].min())), 2e-4)
    test.assertEqual(before["envelope"].keys(), after["envelope"].keys(), "Lost normal/spatial/voxel support")
    for key, score in before["envelope"].items():
        test.assertLessEqual(abs(score - after["envelope"][key]), 2e-4, f"Changed support/depth envelope {key}")


def collide_observed(test, pipeline, state, contacts, *, overflow=False):
    """Check actual dispatch and overflow publication order around one collision."""
    owner = pipeline.narrow_phase._pair_terrain
    launched, saw_exception = [], []
    original, original_tiled = wp.launch, wp.launch_tiled

    def observe(launch, *args, **kwargs):
        kernel = kwargs.get("kernel", args[0] if args else None)
        launched.append(kernel.key)
        if kernel is owner.exceptional:
            saw_exception.append(True)
            if overflow:
                test.assertGreater(int(np.count_nonzero(owner.overflow.numpy())), 0)
                test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0, "Overflow published partial contacts")
        return launch(*args, **kwargs)

    def plain(*args, **kwargs):
        return observe(original, *args, **kwargs)

    def tiled(*args, **kwargs):
        return observe(original_tiled, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=plain), patch.object(wp, "launch_tiled", side_effect=tiled):
        pipeline.collide(state, contacts)
    test.assertTrue(saw_exception, "Exceptional owner was not dispatched")
    test.assertTrue(any(name == owner.fast.key for name in launched), "Fast owner was not dispatched")
    for name in launched:
        test.assertFalse(
            any(
                token in name
                for token in ("_clear_active_kernel", "reduce_buffered_contacts", "export_reduced_contacts")
            ),
            "Admitted pair-private scene launched the retired global reducer: " + name,
        )
    np.testing.assert_array_equal(owner.status.numpy(), 0)
    pipeline.narrow_phase.check_buffer_capacity()
    return public_snapshot(pipeline, state, contacts)


def check_lifecycle(test, device):
    """Exercise finite/generic queries and same-buffer pose/height transitions."""
    model, state, pairs = small_fixture(device)
    original_pose = state.body_q.numpy().copy()
    original_heights = model.heightfield_elevations.numpy().copy()
    pipelines = [make_pipeline(model, pairs, enabled) for enabled in (False, True)]
    test.assertFalse(pipelines[0].narrow_phase._heightfield_pair_reducer)
    test.assertTrue(pipelines[1].narrow_phase._heightfield_pair_reducer)
    outputs = [pipeline.contacts() for pipeline in pipelines]
    initial = []
    for pipeline, contacts in zip(pipelines, outputs, strict=True):
        pipeline.collide(state, contacts)
        initial.append(public_snapshot(pipeline, state, contacts))
    compare_public(test, *initial)
    test.assertEqual(
        set(map(int, initial[1]["shape"])), {1, 2, 3}, "Finite box, generic sphere or odd pair disappeared"
    )
    collide_observed(test, pipelines[1], state, outputs[1])
    graphs = []
    if wp.get_device(device).is_cuda:
        for pipeline, contacts in zip(pipelines, outputs, strict=True):
            with wp.ScopedCapture(device=device) as captured:
                pipeline.collide(state, contacts)
            graphs.append(captured.graph)
    for stage in ("empty", "regrow", "height", "generic_face", "restore"):
        pose, height = original_pose.copy(), original_heights.copy()
        if stage == "empty":
            pose[:, 2] += 1000.0
        elif stage == "height":
            # Elevations are normalized within the fixed 40 mm terrain bounds.
            height += np.float32(0.003 / 0.04)
        elif stage == "generic_face":
            pose[0, 2] = 0.13
        state.body_q.assign(pose)
        model.heightfield_elevations.assign(height)
        current = []
        for index, (pipeline, contacts) in enumerate(zip(pipelines, outputs, strict=True)):
            if graphs:
                wp.capture_launch(graphs[index])
            else:
                pipeline.collide(state, contacts)
            current.append(public_snapshot(pipeline, state, contacts))
        compare_public(test, *current)
        np.testing.assert_array_equal(pipelines[1].narrow_phase._pair_terrain.status.numpy(), 0)
        if stage == "empty":
            test.assertEqual(current[1]["count"], 0)
        if stage == "height":
            test.assertGreater(
                float(np.max(current[1]["point"][:, 2])), float(np.max(initial[1]["point"][:, 2])) + 0.001
            )
        if stage in ("regrow", "restore"):
            compare_public(test, initial[1], current[1])


def check_saved(test, device):
    """Reuse both pinned96-pair scenes with the original physical geometry gates."""
    for gpu in (0, 1):
        model, state, pairs = existing.current_fixture(gpu, device)
        reference = make_pipeline(model, pairs, False)
        candidate = make_pipeline(model, pairs, True)
        test.assertTrue(candidate.narrow_phase._heightfield_pair_reducer)
        outputs = []
        for pipeline in (reference, candidate):
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            outputs.append(public_snapshot(pipeline, state, contacts))
        compare_public(test, *outputs)


class TestHeightfieldPairTerrainCPU(unittest.TestCase):
    def test_feature_module(self):
        """Require the new complete pair owner before its integration controls."""
        self.assertEqual(module().__name__, "newton._src.geometry.heightfield_pair_terrain")
        self.assertTrue(callable(module()._write_record), "Missing deferred record-only callback")
        fast, exceptional = module().create_pair_kernels(write_contact)
        self.assertIsNot(fast, exceptional)

    def test_duplicate_pairs_and_disabled_reduction_fall_back(self):
        """Keep unsupported duplicate/reversed pairs in the complete original path."""
        model, state, pairs = small_fixture("cpu")
        for values in ([[0, 1], [0, 1]], [[0, 1], [1, 0]]):
            duplicated = wp.array(values, dtype=wp.vec2i, device="cpu")
            pipeline = make_pipeline(model, duplicated, True)
            self.assertFalse(pipeline.narrow_phase._heightfield_pair_reducer)
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            pipeline.narrow_phase.check_buffer_capacity()
        pipeline = make_pipeline(model, pairs, True, reduce=False)
        self.assertFalse(pipeline.narrow_phase._heightfield_pair_reducer)
        pipeline = make_pipeline(model, pairs, True, deterministic=True)
        self.assertFalse(pipeline.narrow_phase._heightfield_pair_reducer)

    def test_current_complete_pipeline_cpu(self):
        """Run the actual pair owner and current geometry lifecycle on CPU."""
        check_lifecycle(self, "cpu")

    def test_forced_overflow_before_publication_cpu(self):
        """Route an entire overflowing pair before any partial public output."""
        check_overflow(self, "cpu")


def check_overflow(test, device):
    """Use the same production factory with a deliberately tiny private store."""
    model, state, pairs = small_fixture(device)
    baseline = make_pipeline(model, pairs, False)
    candidate = make_pipeline(model, pairs, True, local_capacity=1)
    test.assertTrue(candidate.narrow_phase._heightfield_pair_reducer)
    before = baseline.contacts()
    baseline.collide(state, before)
    expected = public_snapshot(baseline, state, before)
    after = candidate.contacts()
    actual = collide_observed(test, candidate, state, after, overflow=True)
    compare_public(test, expected, actual)


@unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned GPU lease")
class TestHeightfieldPairTerrainCUDA(unittest.TestCase):
    def test_current_lifecycle_and_forced_overflow(self):
        """Run production CUDA query/selection, replay and exceptional publication."""
        check_lifecycle(self, "cuda:0")
        check_overflow(self, "cuda:0")

    def test_saved96_physical_geometry(self):
        """Preserve saved finite/generic public geometry and contacting shapes."""
        check_saved(self, "cuda:0")


if __name__ == "__main__":
    unittest.main()

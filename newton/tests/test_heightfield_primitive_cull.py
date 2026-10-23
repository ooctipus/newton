# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check primitive heightfield rejection against the original generic callbacks."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import heightfield_geometric as geometric
from newton._src.geometry.contact_reduction_global import BETA_THRESHOLD
from newton._src.geometry.narrow_phase import NarrowPhase, write_contact_simple
from newton._src.utils.heightfield import HeightfieldData
from newton.tests import test_heightfield_geometric_cull as cuboid_checks
from newton.tests.test_heightfield_packed_pairs import logical

FLAGS = {
    "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
    "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
    "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
    # The triangle rejection is the terrain route's: keep the route for every heightfield pair.
    "NEWTON_TERRAIN_LAW": "none",
}


def primitive_model(device, kind, *, slope=False, border=False, deep=False, barrel=False):
    """Build a loaded primitive over a finite scaled, optionally rotated terrain."""
    builder = newton.ModelBuilder()
    config = builder.ShapeConfig(margin=0.001, gap=0.001)
    grid = np.linspace(-0.4, 0.4, 17, dtype=np.float32)
    heights = np.zeros((17, 17), dtype=np.float32)
    terrain_rotation = wp.quat_identity()
    terrain_scale = (1.0, 1.0, 1.0)
    local_z = -0.25 if deep else 0.015
    if slope:
        heights[:] = 0.13 * grid[None, :] + 0.07 * grid[:, None]
        terrain_rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.8, -0.2)), 0.45)
        terrain_scale = (1.25, 0.7, 1.4)
    heights[0, 0] = 1.0
    if deep:
        # Keep the whole-terrain broad phase overlapping the downward prism.
        heights[0, 1] = -1.0
    terrain_pose = wp.transform((20.0, -40.0, 0.0), terrain_rotation)
    terrain = newton.Heightfield(data=heights, nrow=17, ncol=17, hx=0.4, hy=0.4)
    builder.add_shape_heightfield(heightfield=terrain, xform=terrain_pose, scale=terrain_scale, cfg=config)
    shape_rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.4, -0.8, 0.7)), 0.8)
    local_pose = wp.transform((0.39 if border else 0.014, 0.009, local_z), shape_rotation)
    body = builder.add_body(xform=terrain_pose * local_pose)
    if kind == "sphere":
        builder.add_shape_sphere(body, radius=0.045, cfg=config)
    elif kind == "capsule":
        builder.add_shape_capsule(body, radius=0.022, half_height=0.1, cfg=config)
    else:
        builder.add_shape_cylinder(
            body, radius=0.025, half_height=0.1, barrel_radius=0.2 if barrel else 0.0, cfg=config
        )
    return builder.finalize(device=device)


def primitive_pipeline(model, enabled):
    """Keep the stock generic query and reducer while toggling triangle rejection."""
    with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": str(int(enabled))}):
        return newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=4096, max_triangle_pairs=4096)


@wp.kernel
def inspect_primitive_projection(
    transforms: wp.array[wp.transform],
    gaps: wp.array[float],
    data: wp.array[wp.vec4],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    heightfield_index: wp.array[int],
    heightfield: wp.array[HeightfieldData],
    elevations: wp.array[float],
    pairs: wp.array[wp.vec2i],
    pair_count: wp.array[int],
    total_num_threads: int,
    types: wp.array[int],
    sources: wp.array[wp.uint64],
    bounds: wp.array2d[wp.vec3],
    bound_source: wp.array[wp.uint64],
    output: wp.array[wp.vec4],
    threshold: wp.array[float],
):
    a, b = pairs[0][0], pairs[0][1]
    hfd = heightfield[heightfield_index[a]]
    value = geometric._prepare_pair(
        a, b, hfd, types, transforms, data, sources, gaps, lower, upper, bounds, bound_source
    )
    output[0] = wp.vec4(float(value.enabled), value.half[0], value.half[1], value.half[2])
    threshold[0] = value.threshold


class TestHeightfieldPrimitiveCull(unittest.TestCase):
    def test_generic_query_admission(self):
        """Admit geometric rejection independently of the finite-box contact query."""
        with patch.dict(
            os.environ,
            {
                "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
                "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
                "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
            },
        ):
            owner = NarrowPhase(
                max_candidate_pairs=4,
                max_triangle_pairs=16,
                device="cpu",
                has_heightfields=True,
                has_meshes=False,
            )
        self.assertTrue(owner._heightfield_geometric_cull)
        self.assertFalse(owner._heightfield_finite_query)
        with patch.dict(
            os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1", "NEWTON_HEIGHTFIELD_PAIR_CSR": "1"}
        ):
            owner = NarrowPhase(
                max_candidate_pairs=4, max_triangle_pairs=16, device="cpu", has_heightfields=True, has_meshes=False
            )
        self.assertFalse(owner._heightfield_pair_csr_requested)

    def test_generic_query_fallback_admission(self):
        """Retain speculative, mixed-mesh, custom-writer and unreduced fallbacks."""
        with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1"}):
            for kwargs in (
                {"speculative": True},
                {"has_meshes": True},
                {"has_heightfields": False},
                {"reduce_contacts": False},
                {"contact_writer_warp_func": write_contact_simple},
            ):
                with self.subTest(kwargs=kwargs):
                    options = {"has_heightfields": True, "has_meshes": False, **kwargs}
                    owner = NarrowPhase(max_candidate_pairs=4, max_triangle_pairs=16, device="cpu", **options)
                    self.assertFalse(owner._heightfield_geometric_cull)

    def assert_current_contacts(self, old, new, state, old_contacts, new_contacts):
        """Audit omitted native callbacks and retain finite, loaded contact support."""
        for owner in (old, new):
            owner.narrow_phase.check_buffer_capacity()
            self.assertFalse(owner.narrow_phase._heightfield_finite_query)
        before, after = logical(old.narrow_phase), logical(new.narrow_phase)
        report = cuboid_checks.TestHeightfieldGeometricCull.assert_omitted_callbacks(self, old, before, after)
        observations = []
        for owner, contacts in ((old, old_contacts), (new, new_contacts)):
            count = int(contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(count, 0, "A loaded primitive must retain support")
            self.assertLess(count, contacts.rigid_contact_max)
            distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=state.body_q.device)
            point = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=state.body_q.device)
            newton.eval_rigid_contact_kinematics(
                owner.model, state, contacts, out_distance=distance, out_point0_world=point
            )
            d, p = distance.numpy()[:count], point.numpy()[:count]
            normal = contacts.rigid_contact_normal.numpy()[:count]
            self.assertTrue(np.isfinite(d).all() and np.isfinite(p).all() and np.isfinite(normal).all())
            np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=2e-5)
            self.assertLess(float(d.min()), 0.005)
            observations.append(
                (
                    float(d.min()),
                    set(
                        zip(
                            contacts.rigid_contact_shape0.numpy()[:count],
                            contacts.rigid_contact_shape1.numpy()[:count],
                            strict=True,
                        )
                    ),
                )
            )
        self.assertAlmostEqual(observations[0][0], observations[1][0], delta=2e-4)
        self.assertEqual(observations[0][1], observations[1][1])
        return report

    def check_primitive_geometry(self, device):
        """Audit all three primitives across rotations, slopes, borders and penetration."""
        for kind in ("sphere", "capsule", "cylinder"):
            omitted = 0
            for case in ({}, {"slope": True}, {"border": True}, {"deep": True}):
                with self.subTest(device=device, kind=kind, case=case):
                    model = primitive_model(device, kind, **case)
                    state = model.state()
                    old, new = primitive_pipeline(model, False), primitive_pipeline(model, True)
                    self.assertTrue(new.narrow_phase._heightfield_geometric_cull)
                    old_contacts, new_contacts = old.contacts(), new.contacts()
                    old.collide(state, old_contacts)
                    new.collide(state, new_contacts)
                    report = self.assert_current_contacts(old, new, state, old_contacts, new_contacts)
                    omitted += report["original_triangles"] - report["surviving_triangles"]
            self.assertGreater(omitted, 0, f"The production append must reject some {kind} triangles")

    def test_primitive_geometry_cpu(self):
        """Audit primitive generic callbacks and loaded support on CPU."""
        self.check_primitive_geometry("cpu")

    def check_mutable_admission(self, device):
        """Read current dimensions and withdraw unsafe primitive metadata each call."""
        for kind in ("sphere", "capsule", "cylinder"):
            model = primitive_model(device, kind)
            # Replay the one-pass midphase's recorded inputs; the lane midphase takes the same.
            with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_LANE_MIDPHASE": "0"}):
                owner = primitive_pipeline(model, True)
            recorded = []
            launch = wp.launch

            def observe(*args, recorded=recorded, launch=launch, **kwargs):
                if kwargs.get("kernel") is geometric.heightfield_geometric_overlaps_kernel:
                    recorded.append(list(kwargs["inputs"]))
                return launch(*args, **kwargs)

            with patch.object(wp, "launch", side_effect=observe):
                owner.collide(model.state(), owner.contacts())
            self.assertEqual(len(recorded), 1)
            inputs = recorded[0]
            a, b = inputs[8].numpy()[0]
            result = wp.empty(1, dtype=wp.vec4, device=device)
            threshold = wp.empty(1, dtype=float, device=device)

            def evaluate(inputs=inputs, result=result, threshold=threshold):
                wp.launch(inspect_primitive_projection, 1, inputs=[*inputs, result, threshold], device=device)
                return result.numpy()[0].copy(), float(threshold.numpy()[0])

            original = inputs[2].numpy().copy()
            before, _ = evaluate()
            self.assertEqual(before[0], 1)
            r, h = original[b, :2]
            expected = (r, r, r) if kind == "sphere" else (r, r, h + r if kind == "capsule" else h)
            np.testing.assert_allclose(before[1:], expected, atol=1e-7)
            smaller = original.copy()
            smaller[b, :2] *= 0.75
            inputs[2].assign(smaller)
            shrunk, _ = evaluate()
            self.assertEqual(shrunk[0], 1)
            np.testing.assert_allclose(shrunk[1:], before[1:] * 0.75, atol=1e-7)
            for component, value in ((0, -1.0), (0, np.nan), (1, np.inf)):
                invalid = original.copy()
                invalid[b, component] = value
                inputs[2].assign(invalid)
                self.assertEqual(evaluate()[0][0], 0)
            inputs[2].assign(original)
            lower = inputs[3].numpy().copy()
            invalid_lower = lower.copy()
            invalid_lower[b] = 0.0
            inputs[3].assign(invalid_lower)
            self.assertEqual(evaluate()[0][0], 0)
            inputs[3].assign(lower)
            # The real buffered reducer scales beta by the current terrain AABB.
            upper = inputs[4].numpy().copy()
            enlarged = upper.copy()
            enlarged[a] += 1000.0
            inputs[4].assign(enlarged)
            enabled, current_threshold = evaluate()
            self.assertEqual(enabled[0], 1)
            self.assertGreaterEqual(current_threshold, BETA_THRESHOLD * float(np.linalg.norm(enlarged[a] - lower[a])))
            inputs[4].assign(upper)
            np.testing.assert_array_equal(evaluate()[0], before)

    def test_mutable_admission_cpu(self):
        """Verify current primitive dimensions, scaled beta and unknown-data fallback."""
        self.check_mutable_admission("cpu")

    def test_barrel_fallback_cpu(self):
        """Retain the complete generic append for an unsupported barrel cylinder."""
        model = primitive_model("cpu", "cylinder", barrel=True)
        state = model.state()
        old, new = primitive_pipeline(model, False), primitive_pipeline(model, True)
        old_contacts, new_contacts = old.contacts(), new.contacts()
        old.collide(state, old_contacts)
        new.collide(state, new_contacts)
        self.assertEqual(logical(old.narrow_phase), logical(new.narrow_phase))
        self.assert_current_contacts(old, new, state, old_contacts, new_contacts)

    def check_lifecycle(self, device):
        """Replay empty and loaded prefixes after current transforms and height refits."""
        model = primitive_model(device, "capsule")
        state = model.state()
        old, new = primitive_pipeline(model, False), primitive_pipeline(model, True)
        old_contacts, new_contacts = old.contacts(), new.contacts()
        old.collide(state, old_contacts)
        new.collide(state, new_contacts)
        self.assert_current_contacts(old, new, state, old_contacts, new_contacts)
        captured = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as captured:
                new.collide(state, new_contacts)

        def replay():
            if captured is None:
                new.collide(state, new_contacts)
            else:
                wp.capture_launch(captured.graph)

        poses = state.body_q.numpy().copy()
        away = poses.copy()
        away[:, 2] += 3.0
        state.body_q.assign(away)
        replay()
        self.assertEqual(logical(new.narrow_phase), [])
        self.assertEqual(int(new_contacts.rigid_contact_count.numpy()[0]), 0)
        for epoch in range(2):
            current = poses.copy()
            if epoch:
                current[:, 0] += 0.017
                current[:, 3:] = np.array(wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.8, 0.1, -0.3)), 1.0))
                heights = model.heightfield_elevations.numpy().copy()
                heights[8 * 17 + 8] += 0.01
                model.heightfield_elevations.assign(heights)
                scale = model.shape_scale.numpy().copy()
                scale[1, :2] *= 0.9
                model.shape_scale.assign(scale)
            state.body_q.assign(current)
            old.collide(state, old_contacts)
            replay()
            self.assert_current_contacts(old, new, state, old_contacts, new_contacts)

    def test_lifecycle_cpu(self):
        """Exercise empty/regrowing contacts, current transforms, dimensions and terrain."""
        self.check_lifecycle("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_primitives_cuda(self):
        """Audit the primitive native queries and metadata withdrawal on CUDA."""
        self.check_primitive_geometry("cuda:0")
        self.check_mutable_admission("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_graph_lifecycle_cuda(self):
        """Replay current primitive geometry through an empty and regrowing CUDA graph."""
        self.check_lifecycle("cuda:0")


if __name__ == "__main__":
    unittest.main()

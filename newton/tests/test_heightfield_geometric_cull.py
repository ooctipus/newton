# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise current geometric triangle rejection and its complete fallback."""

import json
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
from newton.tests.test_heightfield_cell_reject import make_model
from newton.tests.test_heightfield_packed_pairs import logical

FLAGS = {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1", "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1"}


def pipeline(model, enabled, **kwargs):
    """Select the complete stock collision owner with fixed capacities."""
    with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": str(int(enabled))}):
        return newton.CollisionPipeline(
            model, reduce_contacts=True, rigid_contact_max=32768, max_triangle_pairs=32768, **kwargs
        )


@wp.kernel
def check_radius(rotation: wp.quat, half: wp.vec3, axis: wp.vec3, out: wp.array[float]):
    pair = geometric.PairProjection()
    pair.half = half
    pair.basis = wp.matrix_from_rows(
        wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0)),
        wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0)),
        wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0)),
    )
    out[0] = geometric._radius(pair, axis)


@wp.kernel
def inspect_projection(
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
    output: wp.array[wp.vec2],
):
    a, b = pairs[0][0], pairs[0][1]
    hfd = heightfield[heightfield_index[a]]
    value = geometric._prepare_pair(
        a, b, hfd, types, transforms, data, sources, gaps, lower, upper, bounds, bound_source
    )
    output[0] = wp.vec2(float(value.enabled), value.threshold)


def snapshot(owner, state, contacts):
    """Read current contact distances without imposing reduction tie identity."""
    owner.narrow_phase.check_buffer_capacity()
    count = int(contacts.rigid_contact_count.numpy()[0])
    distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=state.body_q.device)
    newton.eval_rigid_contact_kinematics(owner.model, state, contacts, out_distance=distance)
    return logical(owner.narrow_phase), distance.numpy()[:count]


class TestHeightfieldGeometricCull(unittest.TestCase):
    def test_constructor_admission(self):
        """Require explicit finite, nonpredictive, stock-reducer admission."""
        flags = {
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
            "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1",
            "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
        }

        def create(**kwargs):
            kwargs.setdefault("has_meshes", False)
            return NarrowPhase(
                max_candidate_pairs=4,
                max_triangle_pairs=16,
                device="cpu",
                has_heightfields=True,
                **kwargs,
            )

        with patch.dict(os.environ, flags):
            self.assertTrue(create()._heightfield_geometric_cull)
            self.assertFalse(create(speculative=True)._heightfield_geometric_cull)
            self.assertFalse(create(has_meshes=True)._heightfield_geometric_cull)
            self.assertFalse(create(reduce_contacts=False)._heightfield_geometric_cull)
            # A known callable but nonstock writer must not inherit this proof.
            self.assertFalse(create(contact_writer_warp_func=write_contact_simple)._heightfield_geometric_cull)
        for flag in flags:
            with patch.dict(os.environ, {**flags, flag: "0"}):
                self.assertFalse(create()._heightfield_geometric_cull)
        with patch.dict(os.environ, {**flags, "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "yes"}):
            with self.assertRaises(ValueError):
                create()

    def check_current_admission_guards(self, device):
        """Withdraw unknown geometry and read the current reducer threshold every call."""
        model = make_model(device, z=0.015)
        owner = pipeline(model, True)
        state, contacts = model.state(), owner.contacts()
        recorded = []
        launch = wp.launch

        def observe(*args, **kwargs):
            if kwargs.get("kernel") is geometric.heightfield_geometric_overlaps_kernel:
                recorded.append(list(kwargs["inputs"]))
            return launch(*args, **kwargs)

        with patch.object(wp, "launch", side_effect=observe):
            owner.collide(state, contacts)
        self.assertEqual(len(recorded), 1)
        inputs = recorded[0]
        result = wp.empty(1, dtype=wp.vec2, device=device)

        def evaluate():
            wp.launch(inspect_projection, 1, inputs=[*inputs, result], device=device)
            return result.numpy()[0].copy()

        before = evaluate()
        self.assertEqual(before[0], 1)
        a, b = inputs[8].numpy()[0]
        for index, shape, component in ((1, a, None), (1, b, None), (2, a, 3), (2, b, 3)):
            original = inputs[index].numpy().copy()
            changed = original.copy()
            if component is None:
                changed[shape] = np.nan
            else:
                changed[shape, component] = np.nan
            inputs[index].assign(changed)
            self.assertEqual(evaluate()[0], 0)
            inputs[index].assign(original)
        original = inputs[3].numpy().copy()
        changed = original.copy()
        changed[b] = 0.0  # Does not enclose the certified cuboid.
        inputs[3].assign(changed)
        self.assertEqual(evaluate()[0], 0)
        inputs[3].assign(original)
        original = inputs[4].numpy().copy()
        changed = original.copy()
        changed[a] += 1000.0
        inputs[4].assign(changed)
        larger = evaluate()
        self.assertEqual(larger[0], 1)
        self.assertGreater(larger[1], before[1] + 0.1)
        inputs[4].assign(original)
        np.testing.assert_array_equal(evaluate(), before)

    def test_current_admission_guards(self):
        """Check unknown geometry and current reducer threshold withdrawal on CPU."""
        self.check_current_admission_guards("cpu")

    def test_rotated_projection_radius(self):
        """Match rotated support radius to all eight independent cuboid corners."""
        quaternion = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, -0.8, 0.4)), 0.7)
        half = np.array([0.13, 0.037, 0.011], dtype=np.float32)
        axis = np.array([0.5, -0.2, 0.8], dtype=np.float32)
        result = wp.zeros(1, dtype=float, device="cpu")
        wp.launch(check_radius, 1, inputs=[quaternion, wp.vec3(*half), wp.vec3(*axis), result], device="cpu")
        corners = [
            np.array(wp.quat_rotate(quaternion, wp.vec3(*(half * [x, y, z]))), dtype=np.float64)
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        ]
        self.assertAlmostEqual(
            float(result.numpy()[0]), max(abs(float(corner @ axis)) for corner in corners), delta=2e-7
        )

    def check_emission_and_lifecycle(self, device):
        """Keep supported geometry while compacting the actual emitted triangle prefix."""
        model = make_model(device, z=0.015)
        state = model.state()
        poses = state.body_q.numpy().copy()
        poses[0, 3:] = np.array(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.65))
        state.body_q.assign(poses)
        old, new = pipeline(model, False), pipeline(model, True)
        self.assertTrue(new.narrow_phase._heightfield_geometric_cull)
        old_contacts, new_contacts = old.contacts(), new.contacts()
        old.collide(state, old_contacts)
        new.collide(state, new_contacts)
        a, ad = snapshot(old, state, old_contacts)
        b, bd = snapshot(new, state, new_contacts)
        self.assertTrue(set(b).issubset(a))
        self.assertLess(len(b), len(a), "The production appender must actually retire triangle queries")
        self.assertGreater(len(ad), 0)
        self.assertGreater(len(bd), 0)
        self.assertTrue(np.isfinite(bd).all())
        self.assertAlmostEqual(float(ad.min()), float(bd.min()), delta=2e-4)
        self.assert_omitted_callbacks(old, a, b)
        # A stale/replaced immutable hull source withdraws this pair's admission.
        bound_sources = new.narrow_phase._finite_source.numpy().copy()
        new.narrow_phase._finite_source.assign(bound_sources + np.uint64(1))
        new.collide(state, new_contacts)
        self.assertEqual(logical(new.narrow_phase), a)
        new.narrow_phase._finite_source.assign(bound_sources)
        captured = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as captured:
                new.collide(state, new_contacts)

        def replay():
            if captured is None:
                new.collide(state, new_contacts)
            else:
                wp.capture_launch(captured.graph)

        away = poses.copy()
        away[:, 2] += 3.0
        state.body_q.assign(away)
        replay()
        self.assertEqual(logical(new.narrow_phase), [])
        state.body_q.assign(poses)
        replay()
        self.assertEqual(logical(new.narrow_phase), b)
        self.assertGreater(int(new_contacts.rigid_contact_count.numpy()[0]), 0)
        heights = model.heightfield_elevations.numpy().copy()
        heights[4 * 9 + 4] += 0.012
        model.heightfield_elevations.assign(heights)
        old.collide(state, old_contacts)
        replay()
        current_a, current_ad = snapshot(old, state, old_contacts)
        current_b, current_bd = snapshot(new, state, new_contacts)
        self.assertGreater(len(current_bd), 0)
        self.assertAlmostEqual(float(current_ad.min()), float(current_bd.min()), delta=2e-4)
        self.assert_omitted_callbacks(old, current_a, current_b)

    def assert_omitted_callbacks(self, old, before, after):
        """Audit every omitted original finite/generic callback in the real buffer."""
        removed = set(before) - set(after)
        self.assertTrue(set(after).issubset(before))
        reducer = old.narrow_phase.global_contact_reducer
        count = int(reducer.contact_count.numpy()[0])
        self.assertLess(count, reducer.capacity, "Audit may not inspect a clipped callback stream")
        shape_pairs = reducer.shape_pairs.numpy()[1 : count + 1]
        fingerprint = reducer.contact_fingerprints.numpy()[1 : count + 1].astype(np.uint32)
        depths = reducer.position_depth.numpy()[1 : count + 1, 3]
        # Both unchanged finite and generic multicontact writers encode
        # (((triangle << 1) | 1) << 3) | point. No callback is sampled away.
        keys = [(*pair, int(code >> 4)) for pair, code in zip(shape_pairs, fingerprint, strict=True)]
        before_keys = set(before)
        self.assertTrue(np.all((fingerprint & 8) == 8))
        self.assertTrue(all(key in before_keys for key in keys))
        omitted = np.array([key in removed for key in keys], dtype=bool)
        lo, hi = old.model.shape_collision_aabb_lower.numpy(), old.model.shape_collision_aabb_upper.numpy()
        gaps, margins = old.model.shape_gap.numpy(), old.model.shape_margin.numpy()
        a, b = shape_pairs[:, 0], shape_pairs[:, 1]
        bounds = np.maximum(
            gaps[a] + gaps[b] + margins[a] + margins[b], BETA_THRESHOLD * np.linalg.norm(hi[a] - lo[a], axis=1)
        )
        slack = depths[omitted] - bounds[omitted]
        self.assertTrue(np.isfinite(depths).all())
        if len(slack):
            self.assertGreater(float(slack.min()), 0.001, "An omitted callback could enter contact/reducer competition")
        return {
            "original_triangles": len(before),
            "surviving_triangles": len(after),
            "callbacks": count,
            "omitted_callbacks": int(omitted.sum()),
            "minimum_raw_depth_slack": float(slack.min()) if len(slack) else None,
        }

    def check_saved_current(self, device):
        """Check actual saved current pairs and every omitted original callback."""
        from tools.fpgs_bench.test_heightfield_finite_geometry import current_fixture  # noqa: PLC0415
        from tools.fpgs_bench.test_heightfield_finite_geometry import snapshot as current_snapshot  # noqa: PLC0415

        reports = []
        for gpu in (0, 1):
            model, state, pairs = current_fixture(gpu, device)
            old = pipeline(model, False, shape_pairs_filtered=pairs)
            new = pipeline(model, True, shape_pairs_filtered=pairs)
            old_contacts, new_contacts = old.contacts(), new.contacts()
            self.assertTrue(new.narrow_phase._heightfield_geometric_cull)
            for epoch in range(2):
                if epoch:
                    poses = state.body_q.numpy().copy()
                    poses[:, 2] += 0.003
                    state.body_q.assign(poses)
                old.collide(state, old_contacts)
                new.collide(state, new_contacts)
                before, after = current_snapshot(old, state, old_contacts), current_snapshot(new, state, new_contacts)
                a, b = logical(old.narrow_phase), logical(new.narrow_phase)
                self.assertGreater(len(a) - len(b), 0)
                report = self.assert_omitted_callbacks(old, a, b)
                self.assertGreater(report["omitted_callbacks"], 0)
                self.assertEqual(set(before["shape"].tolist()) - set(after["shape"].tolist()), set())
                for shape in np.unique(before["shape"]):
                    self.assertAlmostEqual(
                        float(before["distance"][before["shape"] == shape].min()),
                        float(after["distance"][after["shape"] == shape].min()),
                        delta=2e-4,
                    )
                reports.append({"gpu_fixture": gpu, "epoch": epoch, **report})
        print("GEOMETRIC_CULL_CURRENT " + json.dumps(reports, allow_nan=False), flush=True)

    def test_actual_emission_and_lifecycle_cpu(self):
        """Check actual current emission, source withdrawal and regrowth on CPU."""
        self.check_emission_and_lifecycle("cpu")

    def test_saved_current_cpu(self):
        """Audit both saved 96-pair scenes and changed current geometry on CPU."""
        self.check_saved_current("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_actual_emission_and_graph_cuda(self):
        """Check actual current emission, source withdrawal and graph regrowth on CUDA."""
        self.check_current_admission_guards("cuda:0")
        self.check_emission_and_lifecycle("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_saved_current_cuda(self):
        """Audit both saved current scenes against every original native callback."""
        self.check_saved_current("cuda:0")


if __name__ == "__main__":
    unittest.main()

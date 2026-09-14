# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the experimental finite triangle/cuboid query and complete dispatch."""

import inspect
import os
import textwrap
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import heightfield_finite as finite
from newton._src.geometry.contact_reduction_global import mesh_triangle_contacts_to_reducer_kernel
from newton._src.geometry.narrow_phase import create_narrow_phase_process_mesh_triangle_contacts_kernel
from newton._src.sim.collide import write_contact


@wp.kernel(enable_backward=False)
def evaluate(
    e1: wp.vec3,
    e2: wp.vec3,
    center: wp.vec3,
    rotation: wp.quat,
    half: wp.vec3,
    output: wp.array[finite.QueryResult],
):
    output[0] = finite.query(e1, e2, center, rotation, half, 0.04)


@wp.kernel(enable_backward=False)
def evaluate_contact_policy(center: wp.vec3, margin_sum: float, output: wp.array[finite.QueryResult]):
    """Exercise actual writer admission independently of pure geometry validity."""
    output[0] = finite.query_contacts(
        wp.vec3(2.0, 0.0, 0.0),
        wp.vec3(0.0, 2.0, 0.0),
        center,
        wp.quat_identity(),
        wp.vec3(0.1, 0.1, 0.1),
        0.04,
        margin_sum,
    )


@wp.kernel(enable_backward=False)
def evaluate_direct_face(
    e1: wp.vec3,
    e2: wp.vec3,
    center: wp.vec3,
    rotation: wp.quat,
    half: wp.vec3,
    output: wp.array[finite.QueryResult],
):
    output[0] = finite._query_contact_geometry(
        e1,
        e2,
        center,
        rotation,
        half,
        0.04,
        0.02,
    )


class TestHeightfieldFinite(unittest.TestCase):
    def test_direct_face_skips_only_discarded_patch(self):
        """Direct speculative refusal retains the existing contact policy."""
        self.assertTrue(callable(getattr(finite, "_query_contact_geometry", None)))
        device = os.environ.get("FPGS_TEST_DEVICE", "cpu")
        direct = wp.empty(1, dtype=finite.QueryResult, device=device)
        geometry = wp.empty(1, dtype=finite.QueryResult, device=device)
        for center in (
            (0.4, 0.4, 0.13),
            (0.4, 0.4, 0.12),
            (0.4, 0.4, 0.11),
            (0.4, 0.4, 0.17),
            (1.5, 1.5, 0.13),
            (0.4, 0.4, 0.09),
        ):
            inputs = [
                wp.vec3(2.0, 0.0, 0.0),
                wp.vec3(0.0, 2.0, 0.0),
                wp.vec3(*center),
                wp.quat_identity(),
                wp.vec3(0.1, 0.1, 0.1),
            ]
            wp.launch(evaluate_direct_face, 1, inputs=[*inputs, direct], device=device)
            wp.launch(evaluate, 1, inputs=[*inputs, geometry], device=device)
            a, b = direct.numpy()[0], geometry.numpy()[0]
            if a[0] < 0:
                self.assertGreater(b[0], 0)
                minimum = min(b[7 + 4 * k] for k in range(int(b[0])))
                self.assertGreater(minimum, 0.02)
                self.assertLessEqual(minimum, 0.04)
            else:
                np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)
            if center == (0.4, 0.4, 0.13):
                self.assertLess(a[0], 0)

    def test_direct_face_rotated_slopes_and_boundaries(self):
        """Keep discarded-patch selection independent of the active writer path."""
        device = os.environ.get("FPGS_TEST_DEVICE", "cpu")
        direct = wp.empty(1, dtype=finite.QueryResult, device=device)
        geometry = wp.empty(1, dtype=finite.QueryResult, device=device)
        refused = 0
        for slope in (0.0, 0.17, -0.31):
            e1, e2 = wp.vec3(2.0, 0.0, slope), wp.vec3(0.0, 2.0, -slope * 0.3)
            normal = wp.normalize(wp.cross(e1, e2))
            for angle in (0.0, 0.21, 0.73):
                q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 0.3)), angle)
                half = wp.vec3(0.1, 0.08, 0.15)
                axes = [wp.quat_rotate(q, wp.vec3(*(float(i == k) for i in range(3)))) for k in range(3)]
                radius = sum(abs(wp.dot(axis, normal)) * half[k] for k, axis in enumerate(axes))
                for gap in (-0.02, 0.01, 0.02, 0.020001, 0.03, 0.039999, 0.04, 0.06):
                    center = (e1 + e2) * 0.25 + normal * (radius + gap)
                    inputs = [e1, e2, center, q, half]
                    wp.launch(evaluate_direct_face, 1, inputs=[*inputs, direct], device=device)
                    wp.launch(evaluate, 1, inputs=[*inputs, geometry], device=device)
                    a, b = direct.numpy()[0], geometry.numpy()[0]
                    if a[0] < 0 and b[0] > 0:
                        refused += 1
                        np.testing.assert_allclose(b[1:4], np.asarray(normal), atol=2e-6)
                        minimum = min(b[7 + 4 * k] for k in range(int(b[0])))
                        self.assertGreater(minimum, 0.02)
                        self.assertLessEqual(minimum, 0.04)
                    else:
                        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)
        self.assertGreaterEqual(refused, 9)

    def run_query(self, e1, e2, center, rotation=(0.0, 0.0, 0.0, 1.0), half=(0.1, 0.1, 0.1)):
        """Evaluate one actual native CPU query without a saved response oracle."""
        output = wp.empty(1, dtype=finite.QueryResult, device="cpu")
        wp.launch(
            evaluate,
            1,
            inputs=[wp.vec3(*e1), wp.vec3(*e2), wp.vec3(*center), wp.quat(*rotation), wp.vec3(*half), output],
            device="cpu",
        )
        return output.numpy()[0]

    def test_query_api(self):
        """Expose only the new finite primitive and complete dispatch helpers."""
        self.assertTrue(callable(finite.query))
        self.assertTrue(callable(finite.bind_model))

    def test_positive_shell_and_finite_border(self):
        """Retain finite top contacts and avoid an infinite-plane boundary witness."""
        output = wp.empty(1, dtype=finite.QueryResult, device="cpu")
        for center in ((0.2, 0.2, 0.13), (1.5, 1.5, 0.13)):
            wp.launch(
                evaluate,
                1,
                inputs=[
                    wp.vec3(1, 0, 0),
                    wp.vec3(0, 1, 0),
                    wp.vec3(*center),
                    wp.quat_identity(),
                    wp.vec3(0.1, 0.1, 0.1),
                    output,
                ],
                device="cpu",
            )
            result = output.numpy()[0]
            self.assertGreaterEqual(result[0], 1)
            self.assertLessEqual(result[0], 5)
            self.assertTrue(np.isfinite(result).all())
            if center[0] < 1:
                self.assertAlmostEqual(float(result[7]), 0.03, delta=2e-6)
            else:
                self.assertGreater(float(result[7]), 0.5)

    def test_positive_face_uses_original_manifold(self):
        """Keep separated shell faces in the original manifold while retaining other queries."""
        output = wp.empty(1, dtype=finite.QueryResult, device="cpu")
        for center, fallback in (
            ((0.4, 0.4, 0.13), True),
            ((0.4, 0.4, 0.12), False),
            ((0.4, 0.4, 0.11), False),
            ((0.4, 0.4, 0.17), False),
            ((1.5, 1.5, 0.13), False),
        ):
            wp.launch(evaluate_contact_policy, 1, inputs=[wp.vec3(*center), 0.02, output], device="cpu")
            result = output.numpy()[0]
            self.assertEqual(result[0] < 0, fallback, msg=str((center, result.tolist())))

    def test_rotated_deep_and_parallel_edge(self):
        """Retain physical top penetration and both finite parallel-edge endpoints."""
        for depth in (-0.02, -0.6):
            q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.52)
            value = self.run_query((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.4, 0.4, depth), tuple(q))
            self.assertTrue(1 <= value[0] <= 5)
            self.assertLess(value[7], 0.0)
            np.testing.assert_allclose(value[1:4], [0.0, 0.0, 1.0], atol=2e-6)
            for k in range(int(value[0])):
                middle, distance = value[4 + 4 * k : 7 + 4 * k], value[7 + 4 * k]
                self.assertAlmostEqual(float(middle[2] - 0.5 * distance), 0.0, delta=2e-6)
        value = self.run_query((0.98, 0.5, -0.98), (0.0, 1.0, 0.0), (-1.02, 0.5, 1.02), half=(1.0, 1.0, 1.0))
        self.assertEqual(value[0], 2)
        self.assertAlmostEqual(float(value[7]), np.sqrt(2) * 0.02, delta=2e-6)
        self.assertAlmostEqual(float(np.linalg.norm(value[8:11] - value[4:7])), 1.0, delta=2e-6)

    def test_unsupported_and_side_fallback(self):
        """Keep ambiguous, invalid and artificial-side cases in the original query."""
        for e1, e2, center in (
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 0.2, 0.1)),
            ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.2, 0.2, 0.1)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 0.2, -1.3)),
            ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-0.03, 0.3, -0.2)),
        ):
            result = self.run_query(e1, e2, center, half=(0.01, 0.01, 0.01))
            self.assertEqual(result[0], -1)

    def test_original_fallback_recovery(self):
        """Recover every original fallback statement after removing only the marker skip."""
        originals = (
            mesh_triangle_contacts_to_reducer_kernel,
            create_narrow_phase_process_mesh_triangle_contacts_kernel(write_contact),
        )
        for original in originals:
            marked = finite.marked_fallback(original)
            before = textwrap.dedent(inspect.getsource(original.func)).splitlines()
            while before[0].startswith("@"):
                before.pop(0)
            after = inspect.getsource(marked.func).replace("_finite_fallback(", "(", 1)
            after = after.replace("        if tri_idx < 0:\n            continue\n", "", 1)
            self.assertEqual(after, "\n".join(before) + "\n")

    def test_complete_marker_lifetime(self):
        """Run real contacts through analytical plus retained fallback on a changing prefix."""
        heightfield = newton.Heightfield(np.zeros((3, 3), dtype=np.float32), nrow=3, ncol=3, hx=1.0, hy=1.0)
        builder = newton.ModelBuilder()
        config = builder.ShapeConfig(margin=0.01, gap=0.01)
        builder.add_shape_heightfield(heightfield=heightfield, cfg=config)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.115), wp.quat_identity()))
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, cfg=config)
        other = builder.add_body(xform=wp.transform(wp.vec3(0.6, 0.0, 0.115), wp.quat_identity()))
        builder.add_shape_sphere(body=other, radius=0.1, cfg=config)
        model = builder.finalize(device="cpu")
        state = model.state()
        for reduce in (False, True):
            with patch.dict(os.environ, NEWTON_HEIGHTFIELD_CELL_REJECT="1", NEWTON_HEIGHTFIELD_FINITE_QUERY="1"):
                pipeline = newton.CollisionPipeline(
                    model, reduce_contacts=reduce, rigid_contact_max=64, max_triangle_pairs=16
                )
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            pipeline.narrow_phase.check_buffer_capacity()
            self.assertGreater(contacts.rigid_contact_count.numpy()[0], 0)
            count = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
            first = pipeline.narrow_phase.triangle_pairs.numpy()[:count].copy()
            self.assertTrue((first[:, 2] < 0).any())
            self.assertTrue((first[:, 2] >= 0).any())
            pipeline.collide(state, contacts)
            pipeline.narrow_phase.check_buffer_capacity()
            second_count = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
            second = pipeline.narrow_phase.triangle_pairs.numpy()[:second_count]
            np.testing.assert_array_equal(first, second)
            pose = state.body_q.numpy().copy()
            changed = pose.copy()
            changed[body, 2] = 3.0
            state.body_q.assign(changed)
            pipeline.collide(state, contacts)
            pipeline.narrow_phase.check_buffer_capacity()
            self.assertLess(int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0]), count)
            self.assertGreater(contacts.rigid_contact_count.numpy()[0], 0)
            state.body_q.assign(pose)
            pipeline.collide(state, contacts)
            pipeline.narrow_phase.check_buffer_capacity()
            self.assertEqual(int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0]), count)


if __name__ == "__main__":
    unittest.main()

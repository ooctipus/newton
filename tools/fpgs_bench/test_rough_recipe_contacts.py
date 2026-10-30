# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare primitive terrain geometry for the cell-reject/packed-pair recipe.

These are collision controls, not a solver-convergence or timing certificate.
Run the CUDA class only under the benchmark owner's GPU lease.
"""

import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.test_heightfield_packed_pairs import logical

KINDS = ("sphere", "capsule", "cylinder", "box")
CASES = ("elevated", "near", "deep", "sloped", "rotated", "scaled")


def make_scene(device, kind, case):
    """Place each primitive at a known signed clearance from an interior plane."""
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.0, gap=0.02)
    scale = (1.3, 0.7, 1.6) if case == "scaled" else (1.0, 1.0, 1.0)
    slope = 0.25 if case in ("sloped", "scaled") else 0.0
    x = np.linspace(-0.8, 0.8, 17, dtype=np.float32)
    heights = np.broadcast_to(slope * x, (17, 17)).copy()
    # Force a broad-phase overlap even for the elevated case; this distant
    # corner is outside every tested primitive's cell window.
    heights[0, 0] = 1.0
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45 if case == "rotated" else 0.0)
    terrain_x = wp.transform((20.0, -40.0, 0.0), rotation)
    heightfield = newton.Heightfield(data=heights, nrow=17, ncol=17, hx=0.8, hy=0.8)
    builder.add_shape_heightfield(heightfield=heightfield, xform=terrain_x, scale=scale, cfg=cfg)
    angle = -math.atan(slope * scale[2] / scale[0])
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    local_normal = wp.quat_rotate(plane_q, wp.vec3(0.0, 0.0, 1.0))
    extent = {"sphere": 0.04, "capsule": 0.025, "cylinder": 0.03, "box": 0.025}[kind]
    clearance = 0.3 if case == "elevated" else -0.05 if case == "deep" else 0.005
    body_x = wp.transform_multiply(terrain_x, wp.transform(local_normal * (extent + clearance), plane_q))
    body = builder.add_body(xform=body_x)
    if kind == "sphere":
        builder.add_shape_sphere(body=body, radius=extent, cfg=cfg)
    elif kind == "capsule":
        horizontal = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2)
        builder.add_shape_capsule(
            body=body, radius=extent, half_height=0.06, xform=wp.transform(wp.vec3(), horizontal), cfg=cfg
        )
    elif kind == "cylinder":
        builder.add_shape_cylinder(body=body, radius=0.035, half_height=extent, cfg=cfg)
    else:
        builder.add_shape_box(body=body, hx=0.06, hy=0.035, hz=extent, cfg=cfg)
    model = builder.finalize(device=device)
    basis_q = wp.transform_get_rotation(body_x)
    basis = np.array([wp.quat_rotate(basis_q, wp.vec3(*axis)) for axis in np.eye(3)], dtype=np.float32)
    return model, basis, np.array((20.0, -40.0, 0.0)), clearance


def make_pipeline(model, enabled, reduced):
    """Isolate exactly the two recipe switches from unrelated ambient opt-ins, on the terrain route they tune (the
    terrain law is the default since d2530beca)."""
    flags = {
        "NEWTON_TERRAIN_LAW": "none",
        "NEWTON_HEIGHTFIELD_CELL_REJECT": str(int(enabled)),
        "NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS": str(int(enabled)),
        "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
        "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "0",
        "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
        "NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": "0",
        "NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS": "0",
    }
    with patch.dict(os.environ, flags):
        return newton.CollisionPipeline(model, reduce_contacts=reduced, rigid_contact_max=256, max_triangle_pairs=128)


def geometry(model, state, pipeline, contacts):
    """Read physical contact geometry only after verifying all observed capacities."""
    pipeline.narrow_phase.check_buffer_capacity()
    count = int(contacts.rigid_contact_count.numpy()[0])
    if not 0 <= count < contacts.rigid_contact_max:
        raise AssertionError("Contact pool full or overflowing")
    distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=model.device)
    point = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=model.device)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance, out_point0_world=point)
    return distance.numpy()[:count], contacts.rigid_contact_normal.numpy()[:count], point.numpy()[:count]


class TestRoughRecipeContactsCPU(unittest.TestCase):
    """Check raw witnesses and reduced support without relying on contact ordering."""

    device = "cpu"

    def assert_support(self, values, basis, origin, clearance):
        """Require finite terrain-supported witnesses and the expected deepest gap."""
        distance, normal, point = values
        self.assertTrue(all(np.isfinite(value).all() for value in values))
        self.assertGreater(len(distance), 0)
        deepest = int(np.argmin(distance))
        self.assertAlmostEqual(float(distance[deepest]), clearance, delta=0.002)
        self.assertGreater(float(normal[deepest] @ basis[2]), 0.99)
        # Positive-gap per-triangle side witnesses are also returned by the
        # original pipeline; compare all of them below, but anchor the deepest
        # support independently to the physical terrain surface.
        self.assertLess(float(abs((point[deepest] - origin) @ basis[2])), 0.0003)
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=0.002)

    def assert_equivalent(self, old, new, basis, *, reduced):
        """Compare penetration, normals and manifold coverage up to redundant witnesses."""
        if len(old[0]) == 0 or len(new[0]) == 0:
            self.assertEqual(len(old[0]), len(new[0]))
            return
        self.assertAlmostEqual(float(old[0].min()), float(new[0].min()), delta=0.0002)
        for source, target in ((old, new), (new, old)):
            normal_error = np.linalg.norm(source[1][:, None] - target[1][None, :], axis=2)
            self.assertLess(float(np.max(np.min(normal_error, axis=1))), 0.002)
            if not reduced:
                distance = np.abs(source[0][:, None] - target[0][None, :])
                position = np.linalg.norm(source[2][:, None] - target[2][None, :], axis=2)
                score = np.maximum.reduce((distance / 0.0002, normal_error / 0.002, position / 0.0002))
                self.assertLess(float(np.max(np.min(score, axis=1))), 1.0)
        # Reduced rows may select different redundant points. Compare support
        # in tangent directions instead of demanding identical row counts.
        directions = np.array(((1, 0), (0, 1), (1, 1), (1, -1)), dtype=float)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        projections = [values[2] @ basis[:2].T @ directions.T for values in (old, new)]
        for operation in (np.min, np.max):
            np.testing.assert_allclose(operation(projections[0], axis=0), operation(projections[1], axis=0), atol=0.002)

    def test_primitives_at_matched_terrain_states(self):
        """Compare all four primitive families over six physical terrain configurations."""
        for kind in KINDS:
            for case in CASES:
                for reduced in (False, True):
                    with self.subTest(kind=kind, case=case, reduced=reduced):
                        model, basis, origin, clearance = make_scene(self.device, kind, case)
                        state = model.state()
                        results, pipelines = [], []
                        for enabled in (False, True):
                            pipeline = make_pipeline(model, enabled, reduced)
                            contacts = pipeline.contacts()
                            pipeline.collide(state, contacts)
                            narrow = pipeline.narrow_phase
                            self.assertEqual(narrow._heightfield_cell_reject, enabled)
                            self.assertEqual(narrow._heightfield_packed_pairs, enabled)
                            self.assertFalse(narrow._heightfield_finite_query)
                            results.append(geometry(model, state, pipeline, contacts))
                            pipelines.append(pipeline)
                        self.assertTrue(
                            set(logical(pipelines[1].narrow_phase)).issubset(logical(pipelines[0].narrow_phase))
                        )
                        if case == "elevated":
                            self.assertGreater(len(logical(pipelines[0].narrow_phase)), 0)
                            self.assertEqual(logical(pipelines[1].narrow_phase), [])
                            self.assertEqual(len(results[0][0]), 0)
                            self.assertEqual(len(results[1][0]), 0)
                        else:
                            for result in results:
                                self.assert_support(result, basis, origin, clearance)
                            self.assert_equivalent(*results, basis, reduced=reduced)

    def test_pose_withdrawal_and_regrowth(self):
        """Exercise current poses and repeated CUDA replay without stale contact prefixes."""
        for kind in KINDS:
            with self.subTest(kind=kind):
                model, basis, origin, clearance = make_scene(self.device, kind, "rotated")
                state = model.state()
                pipelines = [make_pipeline(model, enabled, True) for enabled in (False, True)]
                contacts = [pipeline.contacts() for pipeline in pipelines]
                for pipeline, contact in zip(pipelines, contacts, strict=True):
                    pipeline.collide(state, contact)
                graphs = []
                if model.device.is_cuda:
                    for pipeline, contact in zip(pipelines, contacts, strict=True):
                        with wp.ScopedCapture(device=self.device) as capture:
                            pipeline.collide(state, contact)
                        graphs.append(capture.graph)
                initial = state.body_q.numpy().copy()
                for offset in (0.0, 0.4, 0.0, 0.4, 0.0):
                    moved = initial.copy()
                    moved[:, :3] += offset * basis[2]
                    state.body_q.assign(moved)
                    results = []
                    for index, (pipeline, contact) in enumerate(zip(pipelines, contacts, strict=True)):
                        if graphs:
                            wp.capture_launch(graphs[index])
                        else:
                            pipeline.collide(state, contact)
                        results.append(geometry(model, state, pipeline, contact))
                    if offset:
                        self.assertEqual([len(value[0]) for value in results], [0, 0])
                        self.assertEqual(logical(pipelines[1].narrow_phase), [])
                    else:
                        for result in results:
                            self.assert_support(result, basis, origin, clearance)
                        self.assert_equivalent(*results, basis, reduced=True)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required; use the benchmark owner's GPU lease")
class TestRoughRecipeContactsCUDA(TestRoughRecipeContactsCPU):
    """Run the same geometry cases and actual graph lifecycle on the selected GPU."""

    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()

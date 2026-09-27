# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check coin and dish layouts without launching GPU simulation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.robot._clutter_coins import build_coins
from newton.examples.robot._clutter_dishes import build_dishes


def _shape_bounds(builder, shape):
    """Compute a shape's tight world-space axis-aligned bounds."""
    pose = builder.shape_transform[shape]
    body = builder.shape_body[shape]
    if body >= 0:
        pose = wp.transform_multiply(builder.body_q[body], pose)
    position = np.asarray(wp.transform_get_translation(pose))
    rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(pose))).reshape(3, 3)
    scale = np.asarray(builder.shape_scale[shape])
    kind = builder.shape_type[shape]
    if kind == newton.GeoType.MESH:
        points = builder.shape_source[shape].vertices * scale
        points = points @ rotation.T + position
        return points.min(axis=0), points.max(axis=0)
    if kind == newton.GeoType.BOX:
        extent = np.abs(rotation) @ scale
    elif kind == newton.GeoType.CYLINDER:
        axis = np.abs(rotation[:, 2])
        extent = scale[0] * np.sqrt(np.maximum(0.0, 1.0 - axis * axis)) + scale[1] * axis
    elif kind == newton.GeoType.CAPSULE:
        extent = scale[0] + scale[1] * np.abs(rotation[:, 2])
    elif kind == newton.GeoType.ELLIPSOID:
        extent = np.linalg.norm(rotation * scale, axis=1)
    else:
        raise AssertionError(f"Unexpected primitive type: {kind}")
    return position - extent, position + extent


class TestClutterLayouts(unittest.TestCase):
    def test_counts_seed_and_mass(self):
        """Keep requested counts exact, seeded poses repeatable, and inertias positive."""
        with wp.ScopedDevice("cpu"):
            for build, default in ((build_coins, 520), (build_dishes, 40)):
                for count in (1, 7, 39, 65, default, 257):
                    with self.subTest(scene=build.__name__, count=count):
                        builder = newton.ModelBuilder()
                        scene = build(builder, count=count, seed=19)
                        repeated = newton.ModelBuilder()
                        again = build(repeated, count=count, seed=19)
                        self.assertEqual(scene.keys(), again.keys())
                        self.assertEqual(scene["object_bodies"], again["object_bodies"])
                        np.testing.assert_array_equal(scene["target"], again["target"])
                        np.testing.assert_array_equal(builder.body_q, repeated.body_q)
                        self.assertEqual(len(scene["object_bodies"]), count)
                        self.assertEqual(builder.body_count, count + ("container_body" in scene))
                        self.assertTrue(np.all(np.asarray(builder.body_mass) > 0.0))
                        inertia = np.asarray(builder.body_inertia).reshape(-1, 3, 3)
                        self.assertTrue(np.all(np.linalg.eigvalsh(inertia) > 0.0))
                        np.testing.assert_allclose(builder.shape_gap, 0.0003)
                changed = newton.ModelBuilder()
                build(changed, seed=20)
                original = newton.ModelBuilder()
                build(original, seed=19)
                self.assertFalse(np.array_equal(original.body_q, changed.body_q))
                for count in (0, -1):
                    builder = newton.ModelBuilder()
                    with self.assertRaisesRegex(ValueError, "positive"):
                        build(builder, count=count)
                    self.assertEqual(builder.shape_count, 0)

    def test_coins_start_separated_inside_tray(self):
        """Keep the thin coins separated in the bed, ring layers, and tray."""
        center = np.array((0.43, -0.07, 0.12))
        builder = newton.ModelBuilder()
        scene = build_coins(builder, center=tuple(center))
        poses = np.asarray(builder.body_q)[scene["object_bodies"]]
        radius, thickness = scene["coin_radius"], scene["coin_thickness"]
        count = len(scene["object_bodies"])
        np.testing.assert_allclose(poses[:, 3:], np.tile((0, 0, 0, 1), (count, 1)))
        delta = poses[:, None, :3] - poses[None, :, :3]
        planar_clearance = np.linalg.norm(delta[:, :, :2], axis=2) - 2 * radius
        vertical_clearance = np.abs(delta[:, :, 2]) - thickness
        clearance = np.maximum(planar_clearance, vertical_clearance)
        self.assertGreater(np.min(clearance[np.triu_indices(count, 1)]), 0.000299)
        self.assertTrue(np.all(np.abs(poses[:, :2] - center[:2]) + radius < (0.196, 0.146)))
        self.assertGreater(np.min(poses[:, 2]) - thickness / 2, center[2] + 0.005)
        for body in scene["object_bodies"]:
            (shape,) = builder.body_shapes[body]
            self.assertEqual(builder.shape_type[shape], newton.GeoType.CYLINDER)
            np.testing.assert_allclose(np.asarray(builder.shape_scale[shape])[:2], (0.0125, 0.0014))

    def test_dishes_start_separated_in_rack(self):
        """Keep loose objects separated; hollow-rack clearances have CAD tests."""
        center = np.array((0.43, -0.07, 0.12))
        builder = newton.ModelBuilder()
        scene = build_dishes(builder, center=tuple(center))
        bounds = []
        for body in scene["object_bodies"]:
            parts = np.asarray([_shape_bounds(builder, shape) for shape in builder.body_shapes[body]])
            bounds.append((parts[:, 0].min(axis=0), parts[:, 1].max(axis=0)))
        for index, (low, high) in enumerate(bounds):
            self.assertTrue(np.all(low[:2] > center[:2] - (0.175, 0.14)))
            self.assertTrue(np.all(high[:2] < center[:2] + np.array((0.175, 0.14))))
            for other_low, other_high in bounds[:index]:
                gap = np.maximum(low - other_high, other_low - high)
                self.assertGreater(np.max(gap), 0.0, f"Initial overlap involving {scene['object_kinds'][index]}")
        self.assertEqual(set(scene["object_kinds"]), {"plate", "bowl", "fork", "spoon", "knife", "chopstick"})
        self.assertLess(builder.shape_count, 50)


if __name__ == "__main__":
    unittest.main()

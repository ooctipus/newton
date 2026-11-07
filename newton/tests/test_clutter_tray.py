# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the movable coin tray's CAD topology and physical grasp geometry."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.robot._clutter_coins import build_coins
from newton.examples.robot._clutter_container import container_mesh, handle_mesh


class TestClutterTray(unittest.TestCase):
    def setUp(self):
        self._device_scope = wp.ScopedDevice("cpu")
        self._device_scope.__enter__()
        self.addCleanup(self._device_scope.__exit__, None, None, None)

    def test_closed_cad_meshes(self):
        """Keep low-polygon tray and handle meshes watertight with positive inertia."""
        with wp.ScopedDevice("cpu"):
            for mesh in (
                container_mesh(),
                container_mesh(half_extents=(0.12, 0.08), height=0.055, radius=0.012),
                handle_mesh(),
            ):
                triangles = mesh.indices.reshape(-1, 3)
                self.assertLess(len(triangles), 500)
                edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
                _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
                np.testing.assert_array_equal(counts, 2)
                directed = np.concatenate((edges, edges[:, ::-1]))
                _, inverse = np.unique(directed, axis=0, return_inverse=True)
                balance = np.bincount(inverse, weights=np.tile((1, -1), (len(edges), 1)).T.flatten())
                np.testing.assert_array_equal(balance, 0)
                self.assertGreater(mesh.mass, 0)
                self.assertTrue(np.all(np.linalg.eigvalsh(np.asarray(mesh.inertia).reshape(3, 3)) > 0))
                vertices = mesh.vertices[triangles]
                areas = np.linalg.norm(
                    np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1
                )
                self.assertGreater(areas.min(), 1.0e-10)

    def test_tray_is_a_free_body_not_a_prescribed_motion(self):
        """Keep the loaded tray free and attach its physical handle to that body."""
        builder = newton.ModelBuilder()
        scene = build_coins(builder)
        self.assertEqual(len(scene["object_bodies"]), 520)
        self.assertEqual(builder.body_count, 521)
        self.assertEqual(scene["bed_count"], 64)
        self.assertEqual(scene["tower_layers"], 38)
        tray = scene["container_body"]
        self.assertEqual(tray, scene["tray_body"])
        self.assertNotIn(tray, scene["object_bodies"])
        self.assertEqual(builder.joint_type[0], newton.JointType.FREE)
        self.assertEqual(scene["motion"], "container_slide")
        self.assertGreater(builder.body_mass[tray], 0.5)
        self.assertLess(builder.body_mass[tray], 1.5)
        self.assertTrue(all(builder.shape_body[shape] == tray for shape in scene["tray_shapes"]))
        self.assertEqual(len(scene["sdf_meshes"]), 2)
        self.assertTrue(all(builder.shape_type[shape] == newton.GeoType.MESH for shape in scene["tray_shapes"]))
        poses = np.asarray(builder.body_q)[scene["object_bodies"], :3]
        np.testing.assert_allclose(poses, scene["positions"])
        tray_position = np.asarray(builder.body_q[tray])[:3]
        np.testing.assert_allclose(tray_position, scene["container_initial_position"])
        self.assertGreater(np.min(poses[:, 2]) - scene["coin_thickness"] / 2, tray_position[2] + 0.005)

    def test_handle_clearance(self):
        """Leave room inside the loop for the real Franka fingertip collision pads."""
        builder = newton.ModelBuilder()
        scene = build_coins(builder)
        tray_pose = builder.body_q[scene["tray_body"]]
        handle_shape = scene["tray_shapes"][1]
        handle_pose = wp.transform_multiply(tray_pose, builder.shape_transform[handle_shape])
        np.testing.assert_allclose(np.asarray(handle_pose)[:3], scene["handle_target"])
        # The pad's Y thickness is 15.2 mm; 30 mm joint opening clears the
        # 12 mm half-width bar and fits before the rear bar begins at 53 mm.
        pad_thickness, open_q, closed_q = 0.0152, 0.030, 0.007
        self.assertGreater(open_q, scene["handle_width"] / 2)
        self.assertLess(open_q + pad_thickness, 0.053)
        self.assertLess(closed_q, scene["handle_width"] / 2)
        # The rear mounting bar spans z=.016..032 and y=-.162..-.138:
        # it physically meets the tray wall, rather than floating above it.
        center = np.asarray(scene["container_initial_position"])
        target = np.asarray(scene["handle_target"])
        self.assertLess(target[2] - 0.008, center[2] + 0.026)
        self.assertGreater(target[1] + 0.077, center[1] - 0.150)
        self.assertLess(target[1] + 0.053, center[1] - 0.146)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometry checks for the movable CAD hardware bin; no GPU SDF cook required."""

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton


@unittest.skipUnless(
    importlib.util.find_spec("trimesh") and importlib.util.find_spec("fast_simplification"),
    "Requires the examples mesh dependencies",
)
class TestClutterHardware(unittest.TestCase):
    def test_closed_low_poly_cad_surfaces(self):
        """Keep low-poly thread surfaces closed and preserve all through bores."""
        import trimesh

        from newton.examples.robot._clutter_hardware import _hardware_meshes  # noqa: PLC0415

        with wp.ScopedDevice("cpu"):
            meshes = _hardware_meshes()
        self.assertEqual(len(meshes), 8)
        for (kind, _size), mesh in meshes.items():
            surface = trimesh.Trimesh(mesh.vertices, np.asarray(mesh.indices).reshape(-1, 3), process=False)
            self.assertTrue(surface.is_watertight)
            self.assertTrue(surface.is_winding_consistent)
            self.assertGreater(surface.volume, 0.0)
            self.assertGreater(np.min(surface.area_faces), 0.0)
            self.assertLessEqual(len(surface.faces), 3200)
            self.assertEqual(surface.euler_number, 2 if kind == "bolt" else 0)
            if kind != "bolt":
                self.assertGreater(np.linalg.norm(surface.vertices[:, :2], axis=1).min(), 0.003)

    def test_hardware_mass_placement_and_fill(self):
        """Verify a filled movable bin with disjoint initial pieces and shared meshes."""
        from newton.examples.robot._clutter_hardware import build_hardware  # noqa: PLC0415

        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            scene = build_hardware(builder)
            model = builder.finalize()
        self.assertEqual(len(scene["object_bodies"]), 256)
        self.assertEqual(model.body_count, 257)
        self.assertEqual(scene["object_kinds"], ["bolt", "nut", "washer", "gear"] * 64)
        self.assertTrue(np.all(model.body_mass.numpy() > 0.0))
        self.assertTrue(np.all(np.linalg.eigvalsh(model.body_inertia.numpy()) > 0.0))
        np.testing.assert_allclose(builder.shape_gap, 0.0003)
        self.assertEqual(len(scene["sdf_meshes"]), 10)
        self.assertGreater(scene["solid_volume_m3"] / scene["interior_volume_m3"], 0.5)
        self.assertLess(scene["solid_volume_m3"] / scene["interior_volume_m3"], 0.6)
        low, high = np.asarray(scene["bin_bounds"])
        for i, (a, b) in enumerate(scene["initial_aabbs"]):
            self.assertTrue(np.all(np.asarray(a[:2]) > low[:2]))
            self.assertTrue(np.all(np.asarray(b[:2]) < high[:2]))
            self.assertGreater(a[2], low[2])
            self.assertLess(b[2], 0.17)
            for c, d in scene["initial_aabbs"][:i]:
                self.assertTrue(np.any(np.asarray(a) > d) or np.any(np.asarray(c) > b))
            body = scene["object_bodies"][i]
            self.assertEqual(len(builder.body_shapes[body]), 1)
        self.assertEqual(scene["motion"], "container_slide")
        self.assertEqual(scene["handle_target"], scene["target"])

    def test_seed_and_variable_count(self):
        """Keep seed-repeatable poses and support empty or larger bins."""
        from newton.examples.robot._clutter_hardware import build_hardware  # noqa: PLC0415

        scenes, builders = [], []
        with wp.ScopedDevice("cpu"):
            for seed in (7, 7, 8):
                builder = newton.ModelBuilder()
                scenes.append(build_hardware(builder, count=7, seed=seed, center=(0.4, -0.1, 0.2)))
                builders.append(builder)
            empty = build_hardware(newton.ModelBuilder(), count=0)
            large = build_hardware(newton.ModelBuilder(), count=300)
        np.testing.assert_array_equal(builders[0].body_q, builders[1].body_q)
        self.assertFalse(np.array_equal(builders[0].body_q, builders[2].body_q))
        self.assertEqual(empty["object_bodies"], [])
        self.assertEqual(len(large["object_bodies"]), 300)
        np.testing.assert_allclose(scenes[0]["target"], (0.4, -0.245, 0.2496))


if __name__ == "__main__":
    unittest.main()

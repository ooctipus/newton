# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check original dish CAD before GPU SDF cooking or dynamics."""

import unittest
from collections import Counter

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf
from newton.examples.robot._clutter_dishes import build_dishes, dish_cad_geometry
from newton.examples.robot._clutter_mesh import cook_sdf_meshes


@wp.kernel
def _sample_cad_sdf(sdf: TextureSDFData, points: wp.array[wp.vec3], distances: wp.array[float]):
    i = wp.tid()
    distances[i] = texture_sample_sdf(sdf, points[i])


def _winding(vertices, triangles, point):
    vectors = vertices[triangles].astype(np.float64) - point
    vectors /= np.linalg.norm(vectors, axis=2)[..., None]
    a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = 1 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return np.arctan2(numerator, denominator).sum() / (2 * np.pi)


class TestClutterDishesCad(unittest.TestCase):
    def test_plate_guides_reach_inside_dish_profiles(self):
        """Support upright plates with real posts inside their circular profiles."""
        with wp.ScopedDevice("cpu"):
            scene = build_dishes(newton.ModelBuilder(), center=(0, 0, 0))
        mesh = scene["sdf_meshes"][0][0]
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.indices).reshape(-1, 3)
        for i in range(scene["dish_count"] + 1):
            x = (i - scene["dish_count"] / 2) * 0.026
            for y in (0.005, 0.086):
                self.assertAlmostEqual(_winding(vertices, faces, (x, y, 0.080)), 1.0, places=4)
                # Even the smaller bowls overlap this post's YZ projection.
                self.assertLess(np.hypot(y - 0.043, 0.080 - 0.07625), 0.0615)

    @unittest.skipUnless(wp.is_cuda_available(), "CAD SDF cooking requires CUDA")
    def test_sdf_preserves_cavities_and_fork_gaps(self):
        """Keep empty holes positive and thin solid walls negative after SDF cooking."""
        with wp.ScopedDevice("cuda:0"):
            for kind in ("bowl", "fork", "spoon"):
                vertices, faces = dish_cad_geometry(kind)
                with wp.ScopedDevice("cpu"):
                    mesh = newton.Mesh(vertices, faces.ravel())
                pitch = 0.00045 if kind == "bowl" else 0.0002
                cook_sdf_meshes([(mesh, pitch)], device="cuda:0")
                if kind == "bowl":
                    inner_z = vertices[np.linalg.norm(vertices[:, :2], axis=1) < 1.0e-7, 2].max()
                    points = [(0, 0, inner_z + 0.004), (0, 0, inner_z - 0.001)]
                    signs = (1, -1)
                elif kind == "fork":
                    z = 0.0035 * ((0.060 - 0.022) / 0.056) ** 2
                    points = [(0.060, y, z) for y in (-0.0037, 0, 0.0037, -0.00555, -0.00185, 0.00185, 0.00555)]
                    signs = (1, 1, 1, -1, -1, -1, -1)
                else:
                    points = [(0.052, 0, -0.001), (0.052, 0, -0.0044)]
                    signs = (1, -1)
                distances = wp.empty(len(points), dtype=float)
                wp.launch(
                    _sample_cad_sdf, len(points), [mesh.sdf.texture_data, wp.array(points, dtype=wp.vec3), distances]
                )
                actual = distances.numpy()
                with self.subTest(kind=kind, distances=actual.tolist()):
                    self.assertTrue(np.all(actual * signs > 0.00015))

    def test_closed_oriented_low_polygon_surfaces(self):
        """Keep every CAD shell watertight, consistently oriented, and modest."""
        with wp.ScopedDevice("cpu"):
            scene = build_dishes(newton.ModelBuilder())
        for mesh, _ in scene["sdf_meshes"]:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.indices).reshape(-1, 3)
            with self.subTest(triangles=len(triangles)):
                self.assertLess(len(triangles), 5000)
                edges = [(int(a), int(b)) for face in triangles for a, b in zip(face, np.roll(face, -1), strict=True)]
                counts = Counter(tuple(sorted(edge)) for edge in edges)
                self.assertEqual(set(counts.values()), {2})
                self.assertEqual(Counter(edges), Counter((b, a) for a, b in edges))
                p = vertices[triangles]
                normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
                self.assertGreater(float(np.linalg.norm(normal, axis=1).min()), 1.0e-12)
                self.assertGreater(float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()), 0.0)

    def test_real_open_cavities_and_fork_gaps(self):
        """Preserve empty bowl interiors, spoon concavity and all three fork gaps."""
        for kind in ("plate", "bowl"):
            vertices, faces = dish_cad_geometry(kind)
            axis = vertices[np.linalg.norm(vertices[:, :2], axis=1) < 1.0e-7]
            inner_z = axis[:, 2].max()
            self.assertAlmostEqual(_winding(vertices, faces, (0, 0, inner_z - 0.001)), 1.0, places=5)
            self.assertAlmostEqual(_winding(vertices, faces, (0, 0, inner_z + 0.004)), 0.0, places=5)
        vertices, faces = dish_cad_geometry("fork")
        z = 0.0035 * ((0.060 - 0.022) / 0.056) ** 2
        for y in (-0.0037, 0.0, 0.0037):
            self.assertAlmostEqual(_winding(vertices, faces, (0.060, y, z)), 0.0, places=5)
        for y in (-0.00555, -0.00185, 0.00185, 0.00555):
            self.assertAlmostEqual(_winding(vertices, faces, (0.060, y, z)), 1.0, places=5)
        vertices, faces = dish_cad_geometry("spoon")
        self.assertAlmostEqual(_winding(vertices, faces, (0.052, 0, -0.001)), 0.0, places=5)
        self.assertAlmostEqual(_winding(vertices, faces, (0.052, 0, -0.0037)), 1.0, places=5)

    def test_layout_and_sdf_resolution(self):
        """Start objects separated inside the rack and resolve thin SDF walls."""
        center = np.array((0.43, -0.07, 0.12))
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            scene = build_dishes(builder, center=tuple(center))
        self.assertEqual(builder.body_count, 41)
        self.assertEqual(builder.shape_count, 43)
        self.assertNotIn(scene["container_body"], scene["object_bodies"])
        self.assertEqual(scene["motion"], "container_slide")
        for shape in scene["rack_shapes"]:
            self.assertEqual(builder.shape_body[shape], scene["container_body"])
        self.assertTrue(np.all(np.asarray(builder.body_mass) > 0))
        self.assertTrue(np.all(np.linalg.eigvalsh(np.asarray(builder.body_inertia).reshape(-1, 3, 3)) > 0))
        bounds = []
        for body, kind in zip(scene["object_bodies"], scene["object_kinds"], strict=True):
            points = dish_cad_geometry(kind)[0]
            pose = builder.body_q[body]
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(pose))).reshape(3, 3)
            transformed = points @ rotation.T + np.asarray(wp.transform_get_translation(pose))
            low, high = transformed.min(axis=0), transformed.max(axis=0)
            self.assertTrue(np.all(low[:2] > center[:2] - (0.175, 0.140)))
            self.assertTrue(np.all(high[:2] < center[:2] + np.array((0.175, 0.140))))
            for other_low, other_high in bounds:
                self.assertGreater(float(np.maximum(low - other_high, other_low - high).max()), 0)
            bounds.append((low, high))
        for _, pitch in scene["sdf_meshes"]:
            self.assertLessEqual(pitch, 0.0008)
        for _, pitch in scene["sdf_meshes"][-4:]:
            self.assertLessEqual(pitch, 0.0002)


if __name__ == "__main__":
    unittest.main()

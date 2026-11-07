# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check tight-fit cube CAD, mass properties, and contact-compatible face turns."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.examples.robot._clutter_cube import build_cube
from newton.examples.robot._clutter_cube_probe import _load


def _unique_directions(vectors):
    vectors = np.asarray(vectors).reshape(-1, 3)
    lengths = np.linalg.norm(vectors, axis=1)
    vectors = vectors[lengths > 1.0e-10] / lengths[lengths > 1.0e-10, None]
    first = np.argmax(np.abs(vectors) > 1.0e-8, axis=1)
    vectors *= np.sign(vectors[np.arange(len(vectors)), first])[:, None]
    return np.unique(np.round(vectors, 7), axis=0)


def _penetrating_pairs(mesh, rotations, offsets):
    """Use convex separating axes, including edge cross products, on CPU."""
    triangles = mesh.vertices[mesh.indices.reshape(-1, 3)]
    edges = _unique_directions(np.concatenate((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])))
    normals = _unique_directions(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]))
    points = [mesh.vertices @ rotation.T + offset for rotation, offset in zip(rotations, offsets, strict=True)]
    axes_cache, penetrating = {}, []
    for first in range(len(points)):
        for second in range(first):
            a, b = points[first], points[second]
            if np.any(np.maximum(a.min(axis=0) - b.max(axis=0), b.min(axis=0) - a.max(axis=0)) >= -1.0e-7):
                continue
            ra, rb = rotations[first], rotations[second]
            key = tuple(np.round(np.concatenate((ra.flatten(), rb.flatten())), 6))
            if key not in axes_cache:
                cross = np.cross((edges @ ra.T)[:, None, :], (edges @ rb.T)[None, :, :]).reshape(-1, 3)
                axes_cache[key] = _unique_directions(np.concatenate((normals @ ra.T, normals @ rb.T, cross))).T
            pa, pb = a @ axes_cache[key], b @ axes_cache[key]
            separation = np.maximum(pa.min(axis=0) - pb.max(axis=0), pb.min(axis=0) - pa.max(axis=0))
            if np.max(separation) < -1.0e-7:
                penetrating.append((second, first))
    return penetrating


class TestClutterCube(unittest.TestCase):
    def setUp(self):
        self._device_scope = wp.ScopedDevice("cpu")
        self._device_scope.__enter__()
        self.addCleanup(self._device_scope.__exit__, None, None, None)

    def test_structure_and_mass(self):
        """Retain all 26 massive cubies while leaving their collisions enabled."""
        builder = newton.ModelBuilder()
        scene = build_cube(builder)
        data = scene["metadata"]
        bodies = scene["object_bodies"]
        lattice = np.asarray(data["lattice"])
        self.assertEqual(len(bodies), 26)
        self.assertEqual(builder.joint_type.count(newton.JointType.REVOLUTE), 6)
        self.assertEqual(builder.joint_type.count(newton.JointType.BALL), 20)
        self.assertEqual(builder.joint_type.count(newton.JointType.FIXED), 1)
        self.assertEqual(len(builder.articulation_label), 1)
        self.assertEqual(len(builder.body_shapes[data["core"]]), 1)
        self.assertEqual(builder.joint_parent[data["core_joint"]], -1)
        self.assertTrue(all(builder.joint_parent[joint] == data["core"] for joint in data["joints"]))

        mass = data["cubie_mass"]
        mesh = builder.shape_source[data["collision_shapes"][0]]
        inertia = np.asarray(mesh.inertia).reshape(3, 3) * mass / mesh.mass
        np.testing.assert_allclose(np.asarray(builder.body_mass)[bodies], mass, rtol=1.0e-6)
        np.testing.assert_allclose(np.asarray(builder.body_com)[bodies], lattice * data["pitch"], atol=1.0e-8)
        for body in bodies:
            matrix = np.asarray(builder.body_inertia[body]).reshape(3, 3)
            self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0.0))
            np.testing.assert_allclose(matrix, inertia, rtol=1.0e-5, atol=1.0e-12)

        self.assertEqual(len(data["sticker_shapes"]), 54)
        for shape in data["sticker_shapes"]:
            self.assertEqual(builder.shape_flags[shape], newton.ShapeFlags.VISIBLE)
        collision_shapes = set(data["collision_shapes"])
        for first, second in builder.shape_collision_filter_pairs:
            self.assertFalse(first in collision_shapes and second in collision_shapes)
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(len(mesh.indices) // 3, 44)
        self.assertEqual(len(scene["sdf_meshes"]), 1)
        self.assertLess(data["seam"], 0.0001)
        np.testing.assert_allclose(builder.joint_armature, 0.0001)
        np.testing.assert_allclose(builder.joint_damping, 0.0005)

    def test_cad_surface_is_watertight(self):
        """Keep bevelled cubies closed and facelets separate from collision geometry."""
        builder = newton.ModelBuilder()
        scene = build_cube(builder)
        mesh = scene["sdf_meshes"][0][0]
        triangles = mesh.indices.reshape(-1, 3)
        edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        _, count = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        np.testing.assert_array_equal(count, 2)
        vertices = mesh.vertices[triangles]
        normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        self.assertTrue(np.all(np.sum(normals * vertices.mean(axis=1), axis=1) > 0))
        for shape in scene["metadata"]["sticker_shapes"]:
            self.assertFalse(builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)
            self.assertEqual(builder.shape_type[shape], newton.GeoType.MESH)

    def test_convex_control_preserves_geometry_and_mechanism(self):
        """Change only collision representation, not the hull, mass, or joints."""
        source, control = newton.ModelBuilder(), newton.ModelBuilder()
        source_scene, control_scene = build_cube(source), build_cube(control, collision="convex")
        self.assertEqual(control_scene["sdf_meshes"], [])
        self.assertEqual(control_scene["metadata"]["collision"], "convex")
        np.testing.assert_array_equal(source.body_mass, control.body_mass)
        np.testing.assert_array_equal(source.body_inertia, control.body_inertia)
        np.testing.assert_array_equal(source.joint_type, control.joint_type)
        for first, second in zip(
            source_scene["metadata"]["collision_shapes"], control_scene["metadata"]["collision_shapes"], strict=True
        ):
            self.assertEqual(control.shape_type[second], newton.GeoType.CONVEX_MESH)
            np.testing.assert_array_equal(source.shape_source[first].vertices, control.shape_source[second].vertices)
            np.testing.assert_array_equal(source.shape_source[first].indices, control.shape_source[second].indices)
        with self.assertRaisesRegex(ValueError, "collision"):
            build_cube(newton.ModelBuilder(), collision="box")

    def test_top_grasp_clears_middle_row(self):
        """Keep Franka's pads above the middle row with a 3 mm tracking allowance."""
        builder = newton.ModelBuilder()
        scene = build_cube(builder)
        data = scene["metadata"]
        target_z = scene["target"][2] - data["cube_center"][2]
        pad_half_height, tracking_allowance = 0.00925, 0.003
        middle_row_top = data["cubie_half_extent"]
        self.assertGreater(target_z - pad_half_height - tracking_allowance, middle_row_top)
        self.assertGreater(data["half_extent"] - (target_z - pad_half_height), 0.010)

    def test_physical_inertia_probe_parameters(self):
        """Remove numerical armature and damping without changing cubie mass or inertia."""
        reference = newton.ModelBuilder()
        reference_scene = build_cube(reference)
        physical = newton.ModelBuilder()
        scene = build_cube(physical, armature=0.0, damping=0.0)
        np.testing.assert_allclose(physical.joint_armature, 0.0)
        np.testing.assert_allclose(physical.joint_damping, 0.0)
        np.testing.assert_array_equal(physical.body_mass, reference.body_mass)
        np.testing.assert_array_equal(physical.body_inertia, reference.body_inertia)
        self.assertEqual(scene["metadata"]["top_layer_added_izz_kg_m2"], 0.0)
        data = reference_scene["metadata"]
        ratio = 1 + data["top_layer_added_izz_kg_m2"] / data["top_layer_physical_izz_kg_m2"]
        self.assertGreater(ratio, 74.0)
        self.assertLess(ratio, 74.3)

    def test_virtual_force_couple_has_correct_world_wrench(self):
        """Apply equal opposite pad forces with zero net force and the requested torque."""
        builder = newton.ModelBuilder()
        scene = build_cube(builder, armature=0.0, damping=0.0)
        with patch.object(newton, "use_coord_layout_targets", True):
            model = builder.finalize(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        data = scene["metadata"]
        bodies = [scene["object_bodies"][data["lattice"].index(p)] for p in ((0, 1, 1), (0, -1, 1))]
        radius, pitch = data["half_extent"], data["pitch"]
        measured = wp.zeros((1, 4), dtype=float, device="cpu")
        wp.launch(
            _load,
            dim=1,
            inputs=[
                1,
                model.body_count,
                wp.array(bodies, dtype=int),
                wp.array([(0, radius, pitch), (0, -radius, pitch)], dtype=wp.vec3),
                wp.array([0.01], dtype=float),
                wp.array([1.0], dtype=float),
                radius,
                model.body_com,
                state.body_q,
                state.body_f,
                measured,
                wp.zeros(1, dtype=int),
                wp.zeros((1, 4), dtype=float),
            ],
            device="cpu",
        )
        wrenches = state.body_f.numpy()[bodies]
        np.testing.assert_allclose(wrenches[:, :3].sum(axis=0), 0.0, atol=1.0e-8)
        com = np.asarray(builder.body_com)[bodies]
        moment = (wrenches[:, 3:] + np.cross(com, wrenches[:, :3])).sum(axis=0)
        np.testing.assert_allclose(moment, (0.0, 0.0, 0.01), rtol=1.0e-6, atol=1.0e-8)
        np.testing.assert_allclose(measured.numpy()[0], (0.0, 0.0, 0.01, 0.01 / (2 * radius)), rtol=1.0e-6)

    def test_off_axis_force_reports_core_moment(self):
        """A force at a corner COM has no local moment but has a core-referenced moment."""
        builder = newton.ModelBuilder()
        scene = build_cube(builder, armature=0.0, damping=0.0)
        with patch.object(newton, "use_coord_layout_targets", True):
            model = builder.finalize(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        data = scene["metadata"]
        body = scene["object_bodies"][data["lattice"].index((1, 1, 1))]
        measured = wp.zeros((1, 4), dtype=float)
        wp.launch(
            _load,
            dim=1,
            inputs=[
                2,
                model.body_count,
                wp.array([body, body], dtype=int),
                wp.zeros(2, dtype=wp.vec3),
                wp.array([1.0], dtype=float),
                wp.array([1.0], dtype=float),
                data["half_extent"],
                model.body_com,
                state.body_q,
                state.body_f,
                measured,
                wp.zeros(1, dtype=int),
                wp.zeros((1, 4), dtype=float),
            ],
            device="cpu",
        )
        np.testing.assert_allclose(state.body_f.numpy()[body], (1, 0, 0, 0, 0, 0), atol=1.0e-8)
        np.testing.assert_allclose(measured.numpy()[0], (0, data["pitch"], -data["pitch"], 1), atol=1.0e-8)

    def test_six_face_quarter_turns(self):
        """Use actual forward kinematics to permute each face's nine cubies."""
        builder = newton.ModelBuilder()
        scene = build_cube(builder)
        with patch.object(newton, "use_coord_layout_targets", True):
            model = builder.finalize(device="cpu")
        state = model.state()
        data = scene["metadata"]
        lattice = np.asarray(data["lattice"], dtype=np.float32)
        bodies = scene["object_bodies"]
        center = np.asarray(data["cube_center"])
        com = model.body_com.numpy()
        initial_q = model.joint_q.numpy()

        for axis in range(3):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    normal = np.eye(3, dtype=np.float32)[axis] * sign
                    turn = wp.quat_from_axis_angle(wp.vec3(*normal), wp.pi / 2.0)
                    rotation = np.asarray(wp.quat_to_matrix(turn)).reshape(3, 3)
                    q = initial_q.copy()
                    selected = lattice[:, axis] == sign
                    self.assertEqual(np.count_nonzero(selected), 9)
                    for piece in np.flatnonzero(selected):
                        joint = data["joints"][piece]
                        start = builder.joint_q_start[joint]
                        if builder.joint_type[joint] == newton.JointType.REVOLUTE:
                            q[start] = np.pi / 2.0
                        else:
                            q[start : start + 4] = np.asarray(turn)
                    state.joint_q.assign(q)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    poses = state.body_q.numpy()
                    positions = np.asarray(
                        [wp.transform_point(wp.transform(*poses[body]), wp.vec3(*com[body])) for body in bodies]
                    )
                    expected = lattice.copy()
                    expected[selected] = expected[selected] @ rotation.T
                    np.testing.assert_allclose(positions, center + expected * data["pitch"], atol=1.0e-7)
                    np.testing.assert_allclose(
                        np.linalg.norm(positions - center, axis=1),
                        np.linalg.norm(lattice * data["pitch"], axis=1),
                        atol=1.0e-7,
                    )
                    self.assertEqual(
                        {tuple(position) for position in np.rint(expected).astype(int)},
                        {tuple(position) for position in lattice.astype(int)},
                    )
                    for piece, body in enumerate(bodies):
                        orientation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:]))).reshape(3, 3)
                        np.testing.assert_allclose(orientation, rotation if selected[piece] else np.eye(3), atol=1.0e-6)

    def test_successive_face_turns_clear_but_isolated_cubie_motion_does_not(self):
        """Admit U then R face turns while making an isolated edge collide with neighbors."""
        builder = newton.ModelBuilder()
        scene = build_cube(builder)
        data = scene["metadata"]
        mesh = scene["sdf_meshes"][0][0]
        positions = np.asarray(data["lattice"], dtype=float) * data["pitch"]
        rotations = np.tile(np.eye(3), (26, 1, 1))
        for axis in (2, 0):
            selected = np.isclose(positions[:, axis], data["pitch"])
            self.assertEqual(np.count_nonzero(selected), 9)
            old_positions, old_rotations = positions.copy(), rotations.copy()
            for angle in np.linspace(0.0, np.pi / 2.0, 5):
                quaternion = wp.quat_from_axis_angle(wp.vec3(*np.eye(3)[axis]), float(angle))
                turn = np.asarray(wp.quat_to_matrix(quaternion)).reshape(3, 3)
                positions[selected] = old_positions[selected] @ turn.T
                rotations[selected] = turn @ old_rotations[selected]
                self.assertEqual(_penetrating_pairs(mesh, rotations, positions), [])
                np.testing.assert_array_equal(positions[~selected], old_positions[~selected])
                np.testing.assert_array_equal(rotations[~selected], old_rotations[~selected])
            np.testing.assert_allclose(positions / data["pitch"], np.rint(positions / data["pitch"]), atol=1.0e-6)

        # The loose original boxes allowed substantial independent cubie drift.
        # Tight contact faces must resist rotating one edge without its layer.
        positions = np.asarray(data["lattice"], dtype=float) * data["pitch"]
        rotations[:] = np.eye(3)
        edge = data["lattice"].index((1, 0, 1))
        turn = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.05))).reshape(3, 3)
        positions[edge] = turn @ positions[edge]
        rotations[edge] = turn
        self.assertTrue(_penetrating_pairs(mesh, rotations, positions))

    def test_reject_incomplete_shell(self):
        """Reject object counts that would remove part of the mechanism."""
        builder = newton.ModelBuilder()
        with self.assertRaisesRegex(ValueError, "exactly 26"):
            build_cube(builder, count=25)
        self.assertEqual(builder.body_count, 0)
        self.assertEqual(builder.shape_count, 0)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for precompiled MuJoCo Warp mesh variants."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

_CUBE_FACES = np.array(
    [
        0,
        2,
        1,
        0,
        3,
        2,
        4,
        5,
        6,
        4,
        6,
        7,
        0,
        1,
        5,
        0,
        5,
        4,
        1,
        2,
        6,
        1,
        6,
        5,
        2,
        3,
        7,
        2,
        7,
        6,
        3,
        0,
        4,
        3,
        4,
        7,
    ],
    dtype=np.int32,
)


def _cube_mesh(half_extent: float) -> newton.Mesh:
    """Build a closed cube mesh with the requested half extent [m]."""
    vertices = np.array(
        [
            (-half_extent, -half_extent, -half_extent),
            (half_extent, -half_extent, -half_extent),
            (half_extent, half_extent, -half_extent),
            (-half_extent, half_extent, -half_extent),
            (-half_extent, -half_extent, half_extent),
            (half_extent, -half_extent, half_extent),
            (half_extent, half_extent, half_extent),
            (-half_extent, half_extent, half_extent),
        ],
        dtype=np.float32,
    )
    return newton.Mesh(vertices, _CUBE_FACES)


def _tetrahedron_mesh() -> newton.Mesh:
    vertices = np.array(
        [(-0.12, -0.08, -0.04), (0.28, -0.07, 0.02), (-0.03, 0.24, 0.01), (0.04, 0.03, 0.31)],
        dtype=np.float32,
    )
    faces = np.array((0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3), dtype=np.int32)
    return newton.Mesh(vertices, faces)


def _cube_inertia(mass: float, half_extent: float) -> np.ndarray:
    """Return a cube's inertia tensor about its center of mass [kg*m^2]."""
    diagonal = mass * (2.0 * half_extent) ** 2 / 6.0
    return np.diag([diagonal, diagonal, diagonal]).astype(np.float32)


def _cube_builder(
    mesh: newton.Mesh,
    mass: float,
    half_extent: float,
    *,
    com: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inertia: np.ndarray | None = None,
    shape_xform: wp.transformf | None = None,
    sdf: bool = False,
) -> newton.ModelBuilder:
    if sdf:
        mesh.build_sdf(max_resolution=32)
    builder = newton.ModelBuilder()
    if inertia is None:
        inertia = _cube_inertia(mass, half_extent)
    body = builder.add_link(
        mass=mass,
        com=wp.vec3(*com),
        inertia=wp.mat33(*inertia.flatten()),
        xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()),
        label="box",
        lock_inertia=True,
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.0)
    builder.add_shape_mesh(
        body,
        xform=shape_xform,
        mesh=mesh,
        cfg=shape_cfg,
        label="box_mesh",
    )
    joint = builder.add_joint_free(child=body)
    builder.add_articulation([joint])
    return builder


@unittest.skipUnless(wp.is_cuda_available(), "MuJoCo Warp mesh variants require CUDA")
class TestMuJoCoMeshVariants(unittest.TestCase):
    """Verify complete geometry and inertial rows switch together."""

    def test_mesh_variant_switch_wakes_selected_sleeping_tree(self):
        small = _cube_builder(_cube_mesh(0.05), 1.0, 0.05)
        large = _cube_builder(_cube_mesh(0.10), 8.0, 0.10)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(small)
        builder.add_world(large)
        model = builder.finalize()
        variants = SolverMuJoCo.MeshVariantSet(
            name="box",
            shape_indices=((0,), (1,)),
            variant_builders=(small, large),
            initial_variant_ids=(0, 1),
        )
        solver = SolverMuJoCo(
            model,
            mesh_variant_sets=(variants,),
            enable_sleeping=True,
            nvmax=6,
            disable_contacts=True,
            use_mujoco_contacts=False,
        )
        solver.mjw_data.tree_asleep.assign([[0], [0]])

        solver.set_mesh_variant_index(
            "box",
            variant_ids=wp.array([0], dtype=wp.int32, device=model.device),
            world_ids=wp.array([1], dtype=wp.int32, device=model.device),
        )
        self.assertEqual(int(solver.mjw_data.tree_asleep.numpy()[0, 0]), 0)
        self.assertLess(int(solver.mjw_data.tree_asleep.numpy()[1, 0]), 0)

    def test_mesh_variant_sets_use_compiled_source_shapes(self):
        """Compile candidates independently from the initial world assignment."""
        base = _cube_builder(_cube_mesh(0.05), 1.0, 0.05)
        large = _cube_builder(_cube_mesh(0.10), 8.0, 0.10)
        builder = newton.ModelBuilder()
        builder.add_world(base)
        builder.add_world(base)
        source_shapes = []
        for source in (base, large):
            shape = source.body_shapes[0][0]
            source_shapes.append(
                builder.add_shape(
                    body=-1,
                    type=source.shape_type[shape],
                    xform=source.shape_transform[shape],
                    scale=source.shape_scale[shape],
                    src=source.shape_source[shape],
                    is_static=True,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.0, is_visible=False),
                )
            )
        model = builder.finalize()
        flags = model.shape_flags.numpy().copy()
        flags[source_shapes] = 0
        model.shape_flags.assign(flags)
        variants = SolverMuJoCo.MeshVariantSet(
            name="box",
            shape_indices=((0,), (1,)),
            variant_builders=(base, large),
            initial_variant_ids=(0, 0),
            source_shape_indices=np.asarray(source_shapes)[:, None],
        )
        solver = SolverMuJoCo(model, mesh_variant_sets=(variants,), disable_contacts=True)

        solver.set_mesh_variant_index(
            "box",
            variant_ids=wp.array([1], dtype=wp.int32, device=model.device),
            world_ids=wp.array([1], dtype=wp.int32, device=model.device),
        )
        np.testing.assert_allclose(model.body_mass.numpy(), (1.0, 8.0))

    def test_mesh_variant_sets_reject_shared_body_ownership(self):
        """Reject variant sets that both own the same rigid body."""
        base = _cube_builder(_cube_mesh(0.05), 1.0, 0.05)
        large = _cube_builder(_cube_mesh(0.10), 8.0, 0.10)
        builder = newton.ModelBuilder()
        builder.add_world(base)
        builder.add_world(large)
        model = builder.finalize()
        first = SolverMuJoCo.MeshVariantSet(
            name="first", shape_indices=((0,), (1,)), variant_builders=(base, large), initial_variant_ids=(0, 1)
        )
        second = SolverMuJoCo.MeshVariantSet(
            name="second", shape_indices=((0,), (1,)), variant_builders=(base, large), initial_variant_ids=(0, 1)
        )

        with self.assertRaisesRegex(ValueError, "shares body ownership"):
            SolverMuJoCo(model, mesh_variant_sets=(first, second), disable_contacts=True)

    def test_mesh_variant_switch_updates_bounds_contacts_and_inertia(self):
        """Switch one world without leaving stale broadphase bounds or inertia."""
        small_half_extent = 0.05
        large_half_extent = 0.30
        small_mass = 1.0
        large_mass = 216.0
        small_mesh = _cube_mesh(small_half_extent)
        large_mesh = _cube_mesh(large_half_extent)

        small_builder = _cube_builder(small_mesh, small_mass, small_half_extent)
        large_builder = _cube_builder(large_mesh, large_mass, large_half_extent)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_world(small_builder)
        builder.add_world(large_builder)
        model = builder.finalize()

        variants = SolverMuJoCo.MeshVariantSet(
            name="box",
            shape_indices=((1,), (2,)),
            variant_builders=(small_builder, large_builder),
            initial_variant_ids=(0, 1),
        )
        solver = SolverMuJoCo(model, mesh_variant_sets=(variants,), nconmax=32, njmax=128)
        bank = solver._mesh_variant_banks["box"]
        self.assertEqual(bank.shape_indices.shape, (2, 1))
        self.assertEqual(bank.shapes.shape, (2, 1))
        self.assertEqual(solver._mesh_variant_definitions, ())

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = newton.Contacts(rigid_contact_max=32, soft_contact_max=0, device=model.device)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        solver.set_mesh_variant_index(
            "box",
            variant_ids=wp.array([0], dtype=wp.int32, device=model.device),
            world_ids=wp.array([1], dtype=wp.int32, device=model.device),
        )

        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        self.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), 0)

        solver.mjw_data.qacc_warmstart.fill_(7.0)
        solver.mjw_data.nacon.fill_(3)
        solver.set_mesh_variant_index(
            "box",
            variant_ids=wp.array([1], dtype=wp.int32, device=model.device),
            world_ids=wp.array([1], dtype=wp.int32, device=model.device),
        )
        np.testing.assert_allclose(solver.mjw_data.qacc_warmstart.numpy()[0], 7.0)
        np.testing.assert_allclose(solver.mjw_data.qacc_warmstart.numpy()[1], 0.0)
        self.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), 0)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)

        contact_count = int(solver.mjw_data.nacon.numpy()[0])
        self.assertGreater(contact_count, 0)
        np.testing.assert_array_equal(solver.mjw_data.contact.worldid.numpy()[:contact_count], 1)
        np.testing.assert_array_equal(solver.mesh_variant_ids("box").numpy(), [0, 1])

        geom = 1
        mj_body = 1
        expected_inertia = _cube_inertia(large_mass, large_half_extent)
        expected_rotational_invweight = 1.0 / expected_inertia[0, 0]
        mj_model = solver.mjw_model
        np.testing.assert_allclose(mj_model.geom_size.numpy()[1, geom], large_half_extent)
        np.testing.assert_allclose(mj_model.geom_rbound.numpy()[1, geom], np.sqrt(3.0) * large_half_extent)
        np.testing.assert_allclose(mj_model.geom_aabb.numpy()[1, geom], ((0.0, 0.0, 0.0), (0.3, 0.3, 0.3)))
        np.testing.assert_allclose(model.body_mass.numpy(), [small_mass, large_mass])
        np.testing.assert_allclose(model.body_inertia.numpy()[1], expected_inertia, rtol=1.0e-5)
        np.testing.assert_allclose(model.body_inv_inertia.numpy()[1], np.linalg.inv(expected_inertia), rtol=1.0e-5)
        np.testing.assert_allclose(mj_model.body_mass.numpy()[1, mj_body], large_mass)
        np.testing.assert_allclose(mj_model.body_subtreemass.numpy()[1], [large_mass, large_mass])
        np.testing.assert_allclose(
            mj_model.body_invweight0.numpy()[1, mj_body],
            (1.0 / large_mass, expected_rotational_invweight),
        )
        np.testing.assert_allclose(
            mj_model.dof_invweight0.numpy()[1],
            (1.0 / large_mass,) * 3 + (expected_rotational_invweight,) * 3,
        )
        expected_meaninertia = (3.0 * large_mass + np.trace(expected_inertia)) / 6.0
        np.testing.assert_allclose(mj_model.stat.meaninertia.numpy()[1], expected_meaninertia, rtol=1.0e-5)

    def test_mesh_variant_switch_updates_newton_collision_metadata(self):
        """Switch the mesh and SDF data consumed by Newton's collision pipeline."""
        small_builder = _cube_builder(_cube_mesh(0.05), 1.0, 0.05, sdf=True)
        large_builder = _cube_builder(_cube_mesh(0.30), 216.0, 0.30, sdf=True)

        floor_mesh = newton.Mesh.create_box(1.0, 1.0, 0.05, duplicate_vertices=False, compute_inertia=False)
        floor_mesh.build_sdf(max_resolution=32)
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(
            body=-1,
            mesh=floor_mesh,
            xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()),
            cfg=newton.ModelBuilder.ShapeConfig(gap=0.0),
        )
        builder.add_world(small_builder)
        builder.add_world(large_builder)
        model = builder.finalize()
        variants = SolverMuJoCo.MeshVariantSet(
            name="box",
            shape_indices=((1,), (2,)),
            variant_builders=(small_builder, large_builder),
            initial_variant_ids=(0, 1),
        )
        solver = SolverMuJoCo(model, mesh_variant_sets=(variants,), use_mujoco_contacts=False)

        metadata_fields = (
            "shape_source_ptr",
            "_shape_mesh_properties",
            "_shape_sdf_index",
            "shape_edge_range",
            "shape_collision_aabb_lower",
            "shape_collision_aabb_upper",
            "_shape_voxel_resolution",
        )
        small_metadata = {name: getattr(model, name).numpy()[1].copy() for name in metadata_fields}
        large_metadata = {name: getattr(model, name).numpy()[2].copy() for name in metadata_fields}
        self.assertNotEqual(int(small_metadata["shape_source_ptr"]), int(large_metadata["shape_source_ptr"]))
        self.assertNotEqual(int(small_metadata["_shape_sdf_index"]), int(large_metadata["_shape_sdf_index"]))

        world = wp.array([1], dtype=wp.int32, device=model.device)
        solver.set_mesh_variant_index(
            "box", variant_ids=wp.array([0], dtype=wp.int32, device=model.device), world_ids=world
        )
        for name, expected in small_metadata.items():
            np.testing.assert_array_equal(getattr(model, name).numpy()[2], expected)

        pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=64, max_triangle_pairs=100_000)
        contacts = pipeline.contacts()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)

        solver.set_mesh_variant_index(
            "box", variant_ids=wp.array([1], dtype=wp.int32, device=model.device), world_ids=world
        )
        for name, expected in large_metadata.items():
            np.testing.assert_array_equal(getattr(model, name).numpy()[2], expected)
        pipeline.collide(state, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        np.testing.assert_array_equal(solver.mesh_variant_ids("box").numpy(), [0, 1])

    def test_mesh_variant_matches_native_inertial_compilation(self):
        """Match native MuJoCo constants for a translated mesh and full inertia tensor."""
        base = _cube_builder(_cube_mesh(0.08), 1.0, 0.08)
        inertia = np.array(((0.32, 0.04, -0.02), (0.04, 0.41, 0.03), (-0.02, 0.03, 0.50)), dtype=np.float32)
        angle = 0.6
        target = _cube_builder(
            _tetrahedron_mesh(),
            3.5,
            0.23,
            com=(0.07, -0.04, 0.09),
            inertia=inertia,
            shape_xform=wp.transform((0.11, -0.06, 0.03), (0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0))),
        )

        builder = newton.ModelBuilder()
        builder.add_world(base)
        builder.add_world(target)
        model = builder.finalize()
        variants = SolverMuJoCo.MeshVariantSet(
            name="box",
            shape_indices=((0,), (1,)),
            variant_builders=(base, target),
            initial_variant_ids=(0, 1),
        )
        solver = SolverMuJoCo(model, mesh_variant_sets=(variants,), disable_contacts=True)
        solver.set_mesh_variant_index(
            "box",
            variant_ids=wp.array([1], dtype=wp.int32, device=model.device),
            world_ids=wp.array([1], dtype=wp.int32, device=model.device),
        )

        reference_builder = newton.ModelBuilder()
        reference_builder.add_world(target)
        reference_builder.add_world(target)
        reference_model = reference_builder.finalize()
        reference = SolverMuJoCo(reference_model, disable_contacts=True)

        np.testing.assert_allclose(model.body_mass.numpy()[1], reference_model.body_mass.numpy()[1])
        np.testing.assert_allclose(model.body_com.numpy()[1], reference_model.body_com.numpy()[1])
        np.testing.assert_allclose(model.body_inertia.numpy()[1], reference_model.body_inertia.numpy()[1])
        np.testing.assert_allclose(model.body_inv_inertia.numpy()[1], reference_model.body_inv_inertia.numpy()[1])

        actual, expected = solver.mjw_model, reference.mjw_model
        for field in (
            "geom_size",
            "geom_rbound",
            "geom_aabb",
            "geom_pos",
            "body_mass",
            "body_subtreemass",
            "body_ipos",
            "body_inertia",
            "body_invweight0",
            "body_simple",
            "dof_invweight0",
        ):
            expected_values = getattr(expected, field).numpy()
            np.testing.assert_allclose(
                getattr(actual, field).numpy()[1], expected_values[min(1, expected_values.shape[0] - 1)], rtol=1.0e-5
            )
        expected_geom_quat = expected.geom_quat.numpy()
        expected_body_iquat = expected.body_iquat.numpy()
        self.assertAlmostEqual(
            abs(np.dot(actual.geom_quat.numpy()[1, 0], expected_geom_quat[min(1, len(expected_geom_quat) - 1), 0])),
            1.0,
            places=5,
        )
        mujoco, _ = SolverMuJoCo.import_mujoco()
        actual_rotation, expected_rotation = np.empty(9), np.empty(9)
        mujoco.mju_quat2Mat(actual_rotation, actual.body_iquat.numpy()[1, 1])
        mujoco.mju_quat2Mat(expected_rotation, expected_body_iquat[min(1, len(expected_body_iquat) - 1), 1])
        actual_rotation = actual_rotation.reshape(3, 3)
        expected_rotation = expected_rotation.reshape(3, 3)
        np.testing.assert_allclose(
            actual_rotation @ np.diag(actual.body_inertia.numpy()[1, 1]) @ actual_rotation.T,
            expected_rotation
            @ np.diag(expected.body_inertia.numpy()[min(1, expected.body_inertia.shape[0] - 1), 1])
            @ expected_rotation.T,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            actual.stat.meaninertia.numpy()[1], expected.stat.meaninertia.numpy()[1], rtol=1.0e-5
        )

        def step_with_torque(test_model, test_solver):
            state_in = test_model.state()
            state_out = test_model.state()
            newton.eval_fk(test_model, test_model.joint_q, test_model.joint_qd, state_in)
            body_f = np.zeros((test_model.body_count, 6), dtype=np.float32)
            body_f[1, 3] = 0.1
            state_in.body_f.assign(body_f)
            test_solver.step(state_in, state_out, test_model.control(), None, 0.01)
            return state_out.body_qd.numpy()[1]

        np.testing.assert_allclose(
            step_with_torque(model, solver), step_with_torque(reference_model, reference), rtol=1.0e-5, atol=1.0e-6
        )

    def test_mesh_variant_uses_finalized_inertia(self):
        """Apply diagonal offsets after validating source-builder inertia."""
        base = _cube_builder(_cube_mesh(0.05), 1.0, 0.05)
        authored_inertia = np.diag((0.005, 0.06, 0.10)).astype(np.float32)
        target = _cube_builder(_cube_mesh(0.10), 2.0, 0.10, inertia=authored_inertia)
        inertia_offset = 0.05

        builder = newton.ModelBuilder()
        builder.add_world(base)
        builder.add_world(target)
        model = builder.finalize()
        corrected_inertia = model.body_inertia.numpy()[1].copy()
        self.assertFalse(np.allclose(corrected_inertia, authored_inertia))

        variants = SolverMuJoCo.MeshVariantSet(
            name="box",
            shape_indices=((0,), (1,)),
            variant_builders=(base, target),
            initial_variant_ids=(0, 1),
            inertia_diagonal_offset=inertia_offset,
        )
        solver = SolverMuJoCo(model, mesh_variant_sets=(variants,), disable_contacts=True)
        solver.set_mesh_variant_index(
            "box",
            variant_ids=wp.array([1], dtype=wp.int32, device=model.device),
            world_ids=wp.array([0], dtype=wp.int32, device=model.device),
        )

        expected_inertia = corrected_inertia + np.eye(3) * inertia_offset
        np.testing.assert_allclose(model.body_inertia.numpy()[0], expected_inertia)
        np.testing.assert_allclose(model.body_inv_inertia.numpy()[0], np.linalg.inv(expected_inertia))


if __name__ == "__main__":
    unittest.main()

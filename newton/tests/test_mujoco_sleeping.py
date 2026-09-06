# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for optional MuJoCo Warp sleeping support."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverMuJoCo


def _build_sleep_model(world_count: int = 1, *, register_custom_attributes: bool = False) -> newton.Model:
    """Build identical unactuated one-DOF trees that can become inactive."""
    template = newton.ModelBuilder()
    if register_custom_attributes:
        SolverMuJoCo.register_custom_attributes(template)
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))
    template.add_articulation([joint])

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if register_custom_attributes:
        SolverMuJoCo.register_custom_attributes(builder)
    for i in range(world_count):
        builder.add_world(template, xform=wp.transform((float(i), 0.0, 0.0), wp.quat_identity()))
    return builder.finalize()


def _build_contact_wake_model(*, sleeping_policies: bool = False) -> newton.Model:
    """Build two unactuated free spheres in one zero-gravity world."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if sleeping_policies:
        SolverMuJoCo.register_custom_attributes(builder)
    joints = []
    policies = (SolverMuJoCo.SleepPolicy.NEVER, SolverMuJoCo.SleepPolicy.ALLOWED)
    for index, x in enumerate((0.0, -0.35)):
        custom_attributes = {"mujoco:sleep_policy": policies[index]} if sleeping_policies else None
        body = builder.add_link(
            xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes=custom_attributes,
        )
        builder.add_shape_sphere(body=body, radius=0.1)
        joints.append(builder.add_joint_free(child=body))
    for joint in joints:
        builder.add_articulation([joint])
    return builder.finalize()


def _build_mocap_descendant_model() -> newton.Model:
    """Build a fixed child under a fixed root exported as a mocap body."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    root_joint = builder.add_joint_fixed(parent=-1, child=root)
    child_joint = builder.add_joint_fixed(parent=root, child=child)
    builder.add_articulation([root_joint, child_joint])
    return builder.finalize()


def _build_imported_sleep_policy_model() -> newton.Model:
    """Import four trees covering MuJoCo's authored sleep policies."""
    mjcf = """
    <mujoco>
        <option sleep_tolerance="0.123">
            <flag sleep="enable"/>
        </option>
        <worldbody>
            <body name="auto" pos="-1.5 0 0" sleep="auto">
                <joint name="auto_joint" type="hinge"/>
                <geom type="sphere" size="0.1"/>
            </body>
            <body name="never" pos="-0.5 0 0" sleep="never">
                <joint name="never_joint" type="hinge"/>
                <geom type="sphere" size="0.1"/>
            </body>
            <body name="allowed" pos="0.5 0 0" sleep="allowed">
                <joint name="allowed_joint" type="hinge"/>
                <geom type="sphere" size="0.1"/>
            </body>
            <body name="init" pos="1.5 0 0" sleep="init">
                <joint name="init_joint" type="hinge"/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_mjcf(mjcf)
    return builder.finalize()


def _build_free_body_worlds_model(world_count: int, body_count: int = 6) -> newton.Model:
    """Build worlds of free spheres large enough for MuJoCo Warp's fused per-world path (nv > 32)."""
    template = newton.ModelBuilder()
    for index in range(body_count):
        body = template.add_link(
            xform=wp.transform((0.5 * index, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        template.add_shape_sphere(body=body, radius=0.1)
        template.add_articulation([template.add_joint_free(child=body)])

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for i in range(world_count):
        builder.add_world(template, xform=wp.transform((0.0, 4.0 * i, 0.0), wp.quat_identity()))
    return builder.finalize()


def _build_selective_wake_model() -> newton.Model:
    """Build one awake tree and two initially sleeping trees."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMuJoCo.register_custom_attributes(builder)
    for policy in (
        SolverMuJoCo.SleepPolicy.AUTO,
        SolverMuJoCo.SleepPolicy.INIT,
        SolverMuJoCo.SleepPolicy.INIT,
    ):
        body = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:sleep_policy": policy},
        )
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint])
    return builder.finalize()


class TestMuJoCoSleeping(unittest.TestCase):
    def _make_sim(self, *, world_count: int = 1, **solver_kwargs):
        model = _build_sleep_model(world_count)
        solver = SolverMuJoCo(
            model,
            iterations=2,
            ls_iterations=2,
            disable_contacts=True,
            **solver_kwargs,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        self.collision_pipeline = newton.CollisionPipeline(model)
        contacts = self.collision_pipeline.contacts()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        return model, solver, state_0, state_1, control, contacts

    def _sleep_all(self, solver, state_0, state_1, control, contacts):
        for _ in range(20):
            solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        return state_0, state_1

    def test_sleeping_disabled_by_default(self):
        model, solver, *_ = self._make_sim()
        sleep_bit = int(solver._mujoco.mjtEnableBit.mjENBL_SLEEP)

        self.assertFalse(solver.enable_sleeping)
        self.assertEqual(int(solver.mj_model.opt.enableflags) & sleep_bit, 0)
        self.assertEqual(solver.nvmax, model.joint_dof_count)

    def test_sleeping_configuration_reaches_mujoco_warp(self):
        _, solver, *_ = self._make_sim(enable_sleeping=True, nvmax=1, sleep_tolerance=0.025)
        sleep_bit = int(solver._mujoco.mjtEnableBit.mjENBL_SLEEP)

        self.assertNotEqual(int(solver.mj_model.opt.enableflags) & sleep_bit, 0)
        self.assertNotEqual(int(solver.mjw_model.opt.enableflags) & sleep_bit, 0)
        self.assertEqual(solver.nvmax, 1)
        self.assertAlmostEqual(float(solver.mjw_model.opt.sleep_tolerance.numpy()[0]), 0.025)

    def test_collision_sleep_filter_maps_shapes_to_trees(self):
        model = _build_contact_wake_model()
        solver = SolverMuJoCo(model, enable_sleeping=True, nvmax=12, disable_contacts=True)

        sleep_filter = solver.collision_sleep_filter
        assert sleep_filter is not None
        shape_sleep_index, tree_asleep = sleep_filter

        np.testing.assert_array_equal(shape_sleep_index.numpy(), [[0, 0], [0, 1]])
        self.assertIs(tree_asleep, solver.mjw_data.tree_asleep)

    def test_set_body_sleep_state_updates_compact_indices(self):
        model = _build_contact_wake_model()
        solver = SolverMuJoCo(model, enable_sleeping=True, nvmax=12, disable_contacts=True)
        state = model.state()
        body_ids = wp.array([[0, 1]], dtype=wp.int32, device=model.device)
        world_ids = wp.array([0], dtype=wp.int32, device=model.device)

        solver.set_body_sleep_state(
            body_ids,
            wp.array([[False, True]], dtype=wp.bool, device=model.device),
            world_ids,
        )
        solver.reset(state, flags=0)
        np.testing.assert_array_equal(solver.mjw_data.tree_awake.numpy(), [[1, 0]])
        np.testing.assert_array_equal(solver.mjw_data.nv_awake.numpy(), [6])

        solver.set_body_sleep_state(
            body_ids,
            wp.array([[False, False]], dtype=wp.bool, device=model.device),
            world_ids,
        )
        solver.reset(state, flags=0)
        np.testing.assert_array_equal(solver.mjw_data.tree_awake.numpy(), [[1, 1]])
        np.testing.assert_array_equal(solver.mjw_data.nv_awake.numpy(), [12])

    def test_per_world_sleep_tolerance(self):
        model = _build_sleep_model(world_count=2, register_custom_attributes=True)
        model.mujoco.sleep_tolerance.assign(np.array([0.01, 0.02], dtype=np.float32))

        solver = SolverMuJoCo(model, enable_sleeping=True, nvmax=1, iterations=2, disable_contacts=True)

        np.testing.assert_allclose(solver.mjw_model.opt.sleep_tolerance.numpy(), [0.01, 0.02])

    def test_invalid_sleeping_configurations_fail_early(self):
        model = _build_sleep_model()

        with self.assertRaisesRegex(ValueError, "GPU backend"):
            SolverMuJoCo(model, enable_sleeping=True, use_mujoco_cpu=True)
        with self.assertRaisesRegex(ValueError, "solver='newton'"):
            SolverMuJoCo(model, enable_sleeping=True, solver="cg")
        with self.assertRaisesRegex(ValueError, "does not support integrator='rk4'"):
            SolverMuJoCo(model, enable_sleeping=True, integrator="rk4")
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            SolverMuJoCo(model, enable_sleeping=True, nvmax=2)
        with self.assertRaisesRegex(ValueError, "only supported when sleeping is enabled"):
            SolverMuJoCo(model, nvmax=0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SolverMuJoCo(model, enable_sleeping=True, sleep_tolerance=-1.0)

        multi_tree_model = _build_contact_wake_model()
        with self.assertRaisesRegex(ValueError, "initial state has 12 awake"):
            SolverMuJoCo(multi_tree_model, enable_sleeping=True, nvmax=6)

        custom_integrator_model = _build_sleep_model(register_custom_attributes=True)
        custom_integrator_model.mujoco.integrator.fill_(SolverMuJoCo._parse_integrator("rk4"))
        with self.assertRaisesRegex(ValueError, "does not support integrator='rk4'"):
            SolverMuJoCo(custom_integrator_model, enable_sleeping=True)

    def test_invalid_per_world_sleep_tolerance_fails_early(self):
        model = _build_sleep_model(world_count=2, register_custom_attributes=True)
        model.mujoco.sleep_tolerance.assign(np.array([0.01, -1.0], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "got -1.0 at world 1"):
            SolverMuJoCo(model, enable_sleeping=True, nvmax=1)

    def test_parallel_world_sleeping_requires_matching_default_velocities(self):
        model = _build_sleep_model(world_count=2)
        joint_qd = model.joint_qd.numpy()
        joint_qd[1] = 0.25
        model.joint_qd.assign(joint_qd)

        with self.assertRaisesRegex(ValueError, "identical default joint velocities"):
            SolverMuJoCo(model, enable_sleeping=True, nvmax=1)

    def test_mjcf_sleep_configuration_enables_compact_initial_state(self):
        model = _build_imported_sleep_policy_model()
        sleep_policy = SolverMuJoCo.SleepPolicy

        self.assertTrue(bool(model.mujoco.enable_sleeping.numpy()[0]))
        self.assertAlmostEqual(float(model.mujoco.sleep_tolerance.numpy()[0]), 0.123)
        np.testing.assert_array_equal(
            model.mujoco.sleep_policy.numpy(),
            [sleep_policy.AUTO, sleep_policy.NEVER, sleep_policy.ALLOWED, sleep_policy.INIT],
        )

        solver = SolverMuJoCo(model, nvmax=3, iterations=2, disable_contacts=True)
        mujoco_policy = solver._mujoco.mjtSleepPolicy

        self.assertTrue(solver.enable_sleeping)
        self.assertEqual(solver.nvmax, 3)
        np.testing.assert_array_equal(
            solver.mj_model.tree_sleep_policy,
            [
                mujoco_policy.mjSLEEP_AUTO_ALLOWED,
                mujoco_policy.mjSLEEP_NEVER,
                mujoco_policy.mjSLEEP_ALLOWED,
                mujoco_policy.mjSLEEP_INIT,
            ],
        )
        np.testing.assert_array_equal(
            solver.mjw_model.tree_sleep_policy.numpy(),
            [
                mujoco_policy.mjSLEEP_AUTO_ALLOWED,
                mujoco_policy.mjSLEEP_AUTO_NEVER,
                mujoco_policy.mjSLEEP_AUTO_ALLOWED,
                mujoco_policy.mjSLEEP_AUTO_ALLOWED,
            ],
        )
        np.testing.assert_array_equal(solver.mjw_data.tree_asleep.numpy()[0] < 0, [True, True, True, False])
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 3)

    def test_constructor_can_disable_imported_sleep_flag(self):
        model = _build_imported_sleep_policy_model()
        solver = SolverMuJoCo(model, enable_sleeping=False, disable_contacts=True)
        sleep_bit = int(solver._mujoco.mjtEnableBit.mjENBL_SLEEP)

        self.assertFalse(solver.enable_sleeping)
        self.assertEqual(int(solver.mj_model.opt.enableflags) & sleep_bit, 0)

    def test_reset_restores_compact_initial_sleep_state(self):
        model = _build_imported_sleep_policy_model()
        solver = SolverMuJoCo(model, nvmax=3, iterations=2, disable_contacts=True)
        state = model.state()
        solver.mjw_data.overflow.fill_(64)

        solver._wake_sleeping_worlds()
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 4)

        solver.reset(state, flags=0)

        np.testing.assert_array_equal(solver.mjw_data.tree_asleep.numpy()[0] < 0, [True, True, True, False])
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 3)
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), [0])

    def test_reset_rebuilds_initially_sleeping_tree_state(self):
        model = _build_imported_sleep_policy_model()
        solver = SolverMuJoCo(model, nvmax=3, iterations=2, disable_contacts=True)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        state_0, state_1 = self._sleep_all(solver, state_0, state_1, control, contacts)
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 1)

        joint_q = state_0.joint_q.numpy()
        joint_q[3] = 0.5
        state_0.joint_q.assign(joint_q)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 2)

        solver.reset(state_0, flags=newton.StateFlags.JOINT_Q)

        np.testing.assert_allclose(state_0.joint_q.numpy(), model.joint_q.numpy())
        np.testing.assert_allclose(solver.mjw_data.qpos.numpy()[0], model.joint_q.numpy())
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 3)

        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)

        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 3)
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), [0])

    def test_resting_tree_sleeps_and_force_wakes_it(self):
        _, solver, state_0, state_1, control, contacts = self._make_sim(enable_sleeping=True, nvmax=1)
        state_0, state_1 = self._sleep_all(solver, state_0, state_1, control, contacts)

        self.assertEqual(int(solver.mjw_data.ntree_awake.numpy()[0]), 0)
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 0)

        control.joint_f.fill_(1.0)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)

        self.assertEqual(int(solver.mjw_data.ntree_awake.numpy()[0]), 1)
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 1)

    def test_contact_with_awake_tree_wakes_sleeping_tree(self):
        model = _build_contact_wake_model()
        solver = SolverMuJoCo(model, enable_sleeping=True, nvmax=12, iterations=2, ls_iterations=2)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        state_0, state_1 = self._sleep_all(solver, state_0, state_1, control, contacts)
        self.assertEqual(int(solver.mjw_data.ntree_awake.numpy()[0]), 0)

        qd = state_0.joint_qd.numpy()
        second_joint_dof = int(model.joint_qd_start.numpy()[1])
        qd[second_joint_dof] = 1.0
        state_0.joint_qd.assign(qd)
        for _ in range(20):
            solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
            if int(solver.mjw_data.ntree_awake.numpy()[0]) == 2:
                break

        self.assertEqual(int(solver.mjw_data.ntree_awake.numpy()[0]), 2)

    def test_external_contact_with_awake_tree_wakes_sleeping_tree(self):
        model = _build_contact_wake_model(sleeping_policies=True)
        solver = SolverMuJoCo(
            model,
            enable_sleeping=True,
            nvmax=12,
            iterations=2,
            ls_iterations=2,
            use_mujoco_contacts=False,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_0, state_1 = self._sleep_all(solver, state_0, state_1, control, contacts)
        np.testing.assert_array_equal(solver.mjw_data.tree_awake.numpy()[0], [1, 0])

        joint_q = state_0.joint_q.numpy()
        joint_q[0] = -0.16
        state_0.joint_q.assign(joint_q)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        collision_pipeline.collide(state_0, contacts)

        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)

        np.testing.assert_array_equal(solver.mjw_data.tree_awake.numpy()[0], [1, 1])
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), [0])

    def test_external_pose_edit_wakes_contacting_sleeping_tree(self):
        model = _build_contact_wake_model()
        solver = SolverMuJoCo(
            model,
            enable_sleeping=True,
            nvmax=12,
            iterations=2,
            ls_iterations=2,
            use_mujoco_contacts=False,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_0, state_1 = self._sleep_all(solver, state_0, state_1, control, contacts)
        np.testing.assert_array_equal(solver.mjw_data.tree_awake.numpy()[0], [0, 0])

        joint_q = state_0.joint_q.numpy()
        joint_q[0] = -0.16
        state_0.joint_q.assign(joint_q)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        collision_pipeline.collide(state_0, contacts)

        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)

        np.testing.assert_array_equal(solver.mjw_data.tree_awake.numpy()[0], [1, 1])
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), [0])

    def test_dormant_contact_filter_parks_sleeping_contacts_and_injects_on_wake(self):
        """Park asleep-vs-kinematic live rows and restore them in the substep the tree wakes.

        Runs without the pipeline's dormant contact store so every row reaches the adapter.
        """
        if not wp.is_cuda_available():
            self.skipTest("Texture SDF construction requires CUDA")

        device = wp.get_device()
        support_mesh = newton.Mesh.create_box(0.3, 0.3, 0.08, duplicate_vertices=False)
        dynamic_mesh = newton.Mesh.create_box(0.1, 0.1, 0.08, duplicate_vertices=False)
        for mesh in (support_mesh, dynamic_mesh):
            mesh.build_sdf(device=device, max_resolution=16, narrow_band_range=(-0.02, 0.02), margin=0.02)

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.005
        # A kinematic articulated fixed root becomes a MuJoCo mocap body without a tree
        # (sleep index -2), the same classification as fixed-base sockets in Isaac Lab.
        support_body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=True,
        )
        builder.add_articulation([builder.add_joint_fixed(parent=-1, child=support_body)])
        builder.add_shape_mesh(body=support_body, mesh=support_mesh)
        dynamic_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.158), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            lock_inertia=True,
        )
        dynamic_shape = builder.add_shape_mesh(body=dynamic_body, mesh=dynamic_mesh)
        model = builder.finalize(device=device)

        def run_case(*, dormant_contact_filter: bool):
            solver = SolverMuJoCo(
                model,
                enable_sleeping=True,
                nvmax=model.joint_dof_count,
                iterations=10,
                ls_iterations=5,
                njmax=128,
                nconmax=64,
                use_mujoco_contacts=False,
                dormant_contact_filter=dormant_contact_filter,
            )
            self.assertEqual(solver.dormant_contact_filter, dormant_contact_filter)
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="sap",
                rigid_contact_max=64,
                max_triangle_pairs=4096,
                deterministic=True,
                verify_buffers=False,
            )
            shape_sleep_index, tree_asleep = solver.collision_sleep_filter
            dynamic_world, dynamic_tree = (int(value) for value in shape_sleep_index.numpy()[dynamic_shape])
            pipeline.configure_sleep_filter(shape_sleep_index, tree_asleep)

            state_in = model.state()
            state_out = model.state()
            control = model.control()
            contacts = pipeline.contacts()
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

            # Put the resting body to sleep.
            pipeline.collide(state_in, contacts)
            solver.set_body_sleep_state(
                wp.array([[dynamic_body]], dtype=wp.int32, device=device),
                wp.array([[True]], dtype=wp.bool, device=device),
                wp.array([dynamic_world], dtype=wp.int32, device=device),
            )
            solver.reset(state_in, flags=0)
            self.assertGreaterEqual(int(tree_asleep.numpy()[dynamic_world, dynamic_tree]), 0)

            # A quiet substep: the recomputed support rows reach MJWarp only without the filter.
            pipeline.collide(state_in, contacts)
            replayed_count = int(contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(replayed_count, 0)
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            self.assertGreaterEqual(int(tree_asleep.numpy()[dynamic_world, dynamic_tree]), 0)
            quiet_nacon = int(solver.mjw_data.nacon.numpy()[0])
            if dormant_contact_filter:
                self.assertEqual(quiet_nacon, 0)
                self.assertEqual(int(solver._dormant_count.numpy()[0]), replayed_count)
            else:
                self.assertEqual(quiet_nacon, replayed_count)

            # A force wakes the tree: the parked rows must be converted in that same substep.
            body_force = np.zeros((model.body_count, 6), dtype=np.float32)
            body_force[dynamic_body, 3] = 20.0
            state_in.body_f.assign(body_force)
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            self.assertLess(int(tree_asleep.numpy()[dynamic_world, dynamic_tree]), 0)
            woken_nacon = int(solver.mjw_data.nacon.numpy()[0])
            self.assertEqual(woken_nacon, replayed_count)
            return {
                "joint_q": state_out.joint_q.numpy(),
                "joint_qd": state_out.joint_qd.numpy(),
                "body_q": state_out.body_q.numpy(),
                "body_qd": state_out.body_qd.numpy(),
                "qacc": solver.mjw_data.qacc.numpy(),
                "qfrc_constraint": solver.mjw_data.qfrc_constraint.numpy(),
                "nefc": int(solver.mjw_data.nefc.numpy()[0]),
            }

        reference = run_case(dormant_contact_filter=False)
        filtered = run_case(dormant_contact_filter=True)
        self.assertGreater(reference["nefc"], 0)
        self.assertEqual(filtered["nefc"], reference["nefc"])
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "qacc", "qfrc_constraint"):
            np.testing.assert_allclose(filtered[name], reference[name], rtol=1.0e-5, atol=1.0e-6, err_msg=name)

    def test_reset_wakes_only_selected_worlds(self):
        model, solver, state_0, state_1, control, contacts = self._make_sim(
            world_count=2, enable_sleeping=True, nvmax=1
        )
        state_0, _ = self._sleep_all(solver, state_0, state_1, control, contacts)
        np.testing.assert_array_equal(solver.mjw_data.ntree_awake.numpy(), [0, 0])

        # Make the parent state disagree with MuJoCo so an unmasked sync would
        # visibly overwrite the unselected world's native state.
        joint_q = state_0.joint_q.numpy()
        joint_qd = state_0.joint_qd.numpy()
        joint_q[1] += 0.25
        joint_qd[1] += 0.5
        state_0.joint_q.assign(joint_q)
        state_0.joint_qd.assign(joint_qd)
        preserved_names = (
            "qpos",
            "qvel",
            "tree_asleep",
            "tree_awake",
            "body_awake",
            "ntree_awake",
            "nbody_awake",
            "nv_awake",
        )
        before = {name: getattr(solver.mjw_data, name).numpy().copy() for name in preserved_names}

        mask = wp.array([True, False, False], dtype=wp.bool, device=model.device)
        solver.reset(state_0, world_mask=mask, flags=0)

        np.testing.assert_array_equal(solver.mjw_data.ntree_awake.numpy(), [1, 0])
        np.testing.assert_array_equal(solver.mjw_data.nv_awake.numpy(), [1, 0])
        self.assertLess(int(solver.mjw_data.tree_asleep.numpy()[0, 0]), 0)
        self.assertGreaterEqual(int(solver.mjw_data.tree_asleep.numpy()[1, 0]), 0)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(solver.mjw_data, name).numpy()[1], values[1], err_msg=name)

        before = {name: getattr(solver.mjw_data, name).numpy().copy() for name in preserved_names}
        all_false = wp.zeros(model.world_count + 1, dtype=wp.bool, device=model.device)
        solver.reset(state_0, world_mask=all_false, flags=0)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(solver.mjw_data, name).numpy(), values, err_msg=name)

    def test_reset_rebuilds_only_selected_worlds(self):
        """A masked reset rebuilds the selected world's derived data and leaves other worlds untouched."""
        from mujoco_warp._src import fused_world

        model = _build_free_body_worlds_model(world_count=2)
        solver = SolverMuJoCo(
            model,
            enable_sleeping=True,
            use_mujoco_contacts=False,
            disable_contacts=True,
            iterations=2,
            ls_iterations=2,
        )
        m, d = solver.mjw_model, solver.mjw_data
        if not fused_world.fused_world(m, d):
            self.skipTest("model does not take MuJoCo Warp's fused per-world path")
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = newton.CollisionPipeline(model).contacts()
        dofs_per_world = model.joint_dof_count // model.world_count
        coords_per_world = model.joint_coord_count // model.world_count

        # World 1 drifts, so its kinematic data lags the integrated coordinates by one substep:
        # an all-worlds rebuild at reset would visibly rewrite it.
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[dofs_per_world::6] = 0.3
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0

        # Authored reset of world 0: lift the first sphere, zero the velocities and park it asleep
        # through the overrides (a moving tree would be woken again by the next step).
        joint_q = state_0.joint_q.numpy()
        joint_q[2] += 0.5
        state_0.joint_q.assign(joint_q)
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[:dofs_per_world] = 0.0
        state_0.joint_qd.assign(joint_qd)
        first_bodies = np.nonzero(model.body_world.numpy() == 0)[0][:1]
        solver.set_body_sleep_state(
            wp.array(np.tile(first_bodies, (model.world_count, 1)), dtype=wp.int32, device=model.device),
            wp.array(np.ones((model.world_count, 1), dtype=bool), dtype=wp.bool, device=model.device),
            wp.array([0], dtype=wp.int32, device=model.device),
        )
        derived = (
            "xpos",
            "xquat",
            "xipos",
            "subtree_com",
            "cinert",
            "cdof",
            "cvel",
            "cdof_dot",
            "cacc",
            "qfrc_bias",
            "M",
        )
        bookkeeping = ("tree_asleep", "tree_awake", "body_awake", "body_awake_ind", "dof_awake_ind")
        counts = ("ntree_awake", "nbody_awake", "nv_awake")
        before = {name: getattr(d, name).numpy().copy() for name in derived + bookkeeping + counts + ("qpos", "qvel")}

        mask = wp.array([True, False, False], dtype=wp.bool, device=model.device)
        solver.reset(state_0, world_mask=mask, flags=0)

        for name, values in before.items():
            np.testing.assert_array_equal(getattr(d, name).numpy()[1], values[1], err_msg=f"{name} changed in world 1")
        self.assertGreater(np.abs(d.xpos.numpy()[0] - before["xpos"][0]).max(), 0.4)
        # world 0 poses agree with Newton's forward kinematics at the authored coordinates
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        body_q = state_0.body_q.numpy()
        newton_body = solver.mjc_body_to_newton.numpy()[0]
        for mj_body in range(1, m.nbody):
            np.testing.assert_allclose(
                d.xpos.numpy()[0, mj_body], body_q[newton_body[mj_body]][:3], atol=1e-6, err_msg=f"xpos body {mj_body}"
            )
        np.testing.assert_allclose(d.qpos.numpy()[0, :coords_per_world][:3], joint_q[:3], atol=1e-6)
        # sleep bookkeeping of world 0: the parked tree asleep, everything else awake, compact lists in order
        tree_awake = d.tree_awake.numpy()[0]
        np.testing.assert_array_equal(tree_awake, [0] + [1] * (m.ntree - 1))
        self.assertEqual(int(d.ntree_awake.numpy()[0]), m.ntree - 1)
        awake_bodies = [
            body
            for body in range(m.nbody)
            if m.body_treeid.numpy()[body] < 0 or tree_awake[m.body_treeid.numpy()[body]]
        ]
        self.assertEqual(int(d.nbody_awake.numpy()[0]), len(awake_bodies))
        np.testing.assert_array_equal(d.body_awake_ind.numpy()[0, : len(awake_bodies)], awake_bodies)
        awake_dofs = [dof for dof in range(m.nv) if tree_awake[m.body_treeid.numpy()[m.dof_bodyid.numpy()[dof]]]]
        self.assertEqual(int(d.nv_awake.numpy()[0]), len(awake_dofs))
        np.testing.assert_array_equal(d.dof_awake_ind.numpy()[0, : len(awake_dofs)], awake_dofs)
        np.testing.assert_array_equal(d.overflow.numpy(), [0, 0])

        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        self.assertTrue(np.all(np.isfinite(state_1.joint_q.numpy())))
        np.testing.assert_array_equal(d.tree_awake.numpy()[0], tree_awake)

    def test_model_update_wakes_sleeping_trees(self):
        _, solver, state_0, state_1, control, contacts = self._make_sim(enable_sleeping=True, nvmax=1)
        self._sleep_all(solver, state_0, state_1, control, contacts)
        self.assertEqual(int(solver.mjw_data.ntree_awake.numpy()[0]), 0)

        solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)

        self.assertEqual(int(solver.mjw_data.ntree_awake.numpy()[0]), 1)
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 1)

    def test_wake_preserves_overflow_until_reset(self):
        _, solver, state_0, *_ = self._make_sim(enable_sleeping=True, nvmax=1)
        solver.mjw_data.overflow.fill_(64)

        solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)

        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), [64])
        solver.reset(state_0, flags=0)
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), [0])

    def test_wake_keeps_fixed_mocap_descendants_awake(self):
        model = _build_mocap_descendant_model()
        solver = SolverMuJoCo(model, enable_sleeping=True, nvmax=0, disable_contacts=True)

        solver.notify_model_changed(ModelFlags.MODEL_PROPERTIES)

        sleep_state = solver._mujoco.mjtSleepState
        np.testing.assert_array_equal(
            solver.mjw_data.body_awake.numpy()[0],
            [
                int(sleep_state.mjS_STATIC),
                int(sleep_state.mjS_AWAKE),
                int(sleep_state.mjS_AWAKE),
            ],
        )

    def test_joint_position_update_wakes_sleeping_tree(self):
        _, solver, state_0, state_1, control, contacts = self._make_sim(enable_sleeping=True, nvmax=1)
        state_0, state_1 = self._sleep_all(solver, state_0, state_1, control, contacts)
        self.assertEqual(int(solver.mjw_data.ntree_awake.numpy()[0]), 0)

        state_0.joint_q.fill_(0.5)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)

        self.assertEqual(int(solver.mjw_data.ntree_awake.numpy()[0]), 1)
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 1)

    def test_joint_position_update_wakes_only_affected_tree(self):
        model = _build_selective_wake_model()
        solver = SolverMuJoCo(model, enable_sleeping=True, nvmax=2, iterations=2, disable_contacts=True)
        state_0 = model.state()
        state_1 = model.state()
        joint_q = state_0.joint_q.numpy()
        joint_q[1] = 0.5
        state_0.joint_q.assign(joint_q)

        collision_pipeline = newton.CollisionPipeline(model)
        solver.step(state_0, state_1, model.control(), collision_pipeline.contacts(), 1.0 / 60.0)

        np.testing.assert_array_equal(solver.mjw_data.tree_asleep.numpy()[0] < 0, [True, True, False])
        self.assertEqual(int(solver.mjw_data.nv_awake.numpy()[0]), 2)
        np.testing.assert_array_equal(solver.mjw_data.overflow.numpy(), [0])

    def test_intermediate_substeps_share_the_tick_timestep(self):
        """opt.timestep is written at the first substep of a tick and whenever dt changes."""
        _, solver, state_0, state_1, control, contacts = self._make_sim(enable_sleeping=True, nvmax=1)

        solver._step_intermediate(state_0, state_1, control, contacts, 1.0 / 60.0)
        np.testing.assert_allclose(solver.mjw_model.opt.timestep.numpy(), 1.0 / 60.0)
        solver._step_intermediate(state_1, state_0, control, contacts, 1.0 / 120.0)
        np.testing.assert_allclose(solver.mjw_model.opt.timestep.numpy(), 1.0 / 120.0)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        solver._step_intermediate(state_1, state_0, control, contacts, 1.0 / 240.0)
        np.testing.assert_allclose(solver.mjw_model.opt.timestep.numpy(), 1.0 / 240.0)

    def test_world_solver_option_reaches_mjwarp(self):
        """``world_solver=True`` selects MuJoCo Warp's per-world solver through ``Option.world_solver``."""
        if not wp.is_cuda_available():
            self.skipTest("MuJoCo Warp requires a CUDA device")
        kwargs = {"enable_sleeping": True, "nvmax": 12, "use_mujoco_contacts": False, "jacobian": "sparse"}
        solver = SolverMuJoCo(_build_contact_wake_model(), **kwargs)
        if not hasattr(solver.mjw_model.opt, "world_solver"):
            self.skipTest("The installed MuJoCo Warp has no Option.world_solver")
        self.assertFalse(solver.mjw_model.opt.world_solver)
        solver = SolverMuJoCo(_build_contact_wake_model(), world_solver=True, **kwargs)
        self.assertTrue(solver.mjw_model.opt.world_solver)

    def test_contact_pose_hook_matches_pre_step_refresh_and_defers_body_state(self):
        """Contact poses refreshed inside the MJWarp step match the pre-step refresh from body_q."""
        if not wp.is_cuda_available():
            self.skipTest("The fused per-world MJWarp path requires a CUDA device")
        model = _build_contact_wake_model()
        substeps = 4
        dt = 1.0 / 240.0

        def run(*, hook: bool, publish_intermediate: bool, publish_joint_state: bool = True):
            solver = SolverMuJoCo(
                model,
                enable_sleeping=True,
                nvmax=12,
                iterations=10,
                ls_iterations=10,
                use_mujoco_contacts=False,
                jacobian="sparse",
                update_data_interval=2,
            )
            if not hasattr(solver.mjw_model.callback, "post_position"):
                self.skipTest("The installed MuJoCo Warp has no post_position callback")
            if not hook:
                # Exercise the pre-step conversion path the hook replaces.
                solver._contact_pose_hook = False
            self.assertEqual(solver._contact_pose_hook, hook)
            solver.publish_intermediate_body_state = publish_intermediate
            solver.publish_intermediate_joint_state = publish_joint_state
            state = model.state()
            control = model.control()
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            state, _ = self._sleep_all(solver, state, model.state(), control, contacts)
            np.testing.assert_array_equal(solver.mjw_data.tree_awake.numpy()[0], [0, 0])
            # Launch the second sphere at the sleeping first one; contacts wake it mid-run.
            qd = state.joint_qd.numpy()
            qd[int(model.joint_qd_start.numpy()[1])] = 1.0
            state.joint_qd.assign(qd)
            solver.reset(state, flags=0)
            tree_asleep = []
            nacon = []
            for _tick in range(12):
                pipeline.collide(state, contacts)
                for substep in range(substeps):
                    if substep == substeps - 1:
                        solver.step(state, state, control, contacts, dt)
                    else:
                        solver._step_intermediate(state, state, control, contacts, dt)
                    tree_asleep.append(solver.mjw_data.tree_asleep.numpy().copy())
                    nacon.append(int(solver.mjw_data.nacon.numpy()[0]))
            # the hook is unbound outside the step
            self.assertIsNone(solver.mjw_model.callback.post_position)
            fk_state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, fk_state)
            return {
                "tree_asleep": np.stack(tree_asleep),
                "nacon": np.array(nacon),
                "qpos": solver.mjw_data.qpos.numpy().copy(),
                "qvel": solver.mjw_data.qvel.numpy().copy(),
                "body_q": state.body_q.numpy().copy(),
                "body_qd": state.body_qd.numpy().copy(),
                "body_q_fk": fk_state.body_q.numpy().copy(),
                "body_qd_fk": fk_state.body_qd.numpy().copy(),
            }

        reference = run(hook=False, publish_intermediate=True)
        self.assertGreater(reference["nacon"].max(), 0)
        self.assertTrue((reference["tree_asleep"][-1] < 0).all())
        for publish_intermediate, publish_joint_state in ((True, True), (False, True), (False, False)):
            hooked = run(hook=True, publish_intermediate=publish_intermediate, publish_joint_state=publish_joint_state)
            np.testing.assert_array_equal(hooked["tree_asleep"], reference["tree_asleep"])
            np.testing.assert_array_equal(hooked["nacon"], reference["nacon"])
            for name in ("qpos", "qvel", "body_q", "body_qd"):
                np.testing.assert_allclose(hooked[name], reference[name], rtol=1.0e-5, atol=1.0e-5, err_msg=name)
            # The finalizing step always publishes body state consistent with the joint coordinates.
            np.testing.assert_array_equal(hooked["body_q"], hooked["body_q_fk"])
            np.testing.assert_array_equal(hooked["body_qd"], hooked["body_qd_fk"])

    def test_mjwarp_derived_publish_follows_state_requests(self):
        """The fused forward publishes MuJoCo Warp's derived data only when something reads it."""
        model = _build_contact_wake_model()
        solver = SolverMuJoCo(model, enable_sleeping=True, nvmax=12, iterations=2, ls_iterations=2)
        if not hasattr(solver.mjw_model.opt, "fused_world_publish_derived"):
            self.skipTest("The installed MuJoCo Warp has no fused_world_publish_derived option")
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = newton.CollisionPipeline(model).contacts()
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        self.assertFalse(solver.mjw_model.opt.fused_world_publish_derived)

        solver.mjwarp_publish_derived = True
        solver.step(state_1, state_0, control, contacts, 1.0 / 60.0)
        self.assertTrue(solver.mjw_model.opt.fused_world_publish_derived)

        # body_qdd / body_parent_f read cacc / cfrc_int, so the request forces the publish.
        solver.mjwarp_publish_derived = False
        model.request_state_attributes("body_qdd", "body_parent_f")
        derived_state = model.state()
        solver.step(state_0, derived_state, control, contacts, 1.0 / 60.0)
        self.assertTrue(solver.mjw_model.opt.fused_world_publish_derived)
        self.assertTrue(np.all(np.isfinite(derived_state.body_qdd.numpy())))

    def test_sleeping_step_and_reset_support_cuda_graph_capture(self):
        model, solver, state_0, state_1, control, contacts = self._make_sim(enable_sleeping=True, nvmax=1)
        if not model.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")

        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)

        wp.capture_launch(capture.graph)
        self.assertTrue(np.all(np.isfinite(state_1.joint_q.numpy())))

        all_false = wp.zeros(model.world_count + 1, dtype=wp.bool, device=model.device)
        solver.reset(state_0, world_mask=all_false, flags=0)
        qpos_before = solver.mjw_data.qpos.numpy().copy()
        qvel_before = solver.mjw_data.qvel.numpy().copy()
        tree_asleep_before = solver.mjw_data.tree_asleep.numpy().copy()
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state_0, world_mask=all_false, flags=0)

        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(solver.mjw_data.qpos.numpy(), qpos_before)
        np.testing.assert_array_equal(solver.mjw_data.qvel.numpy(), qvel_before)
        np.testing.assert_array_equal(solver.mjw_data.tree_asleep.numpy(), tree_asleep_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)

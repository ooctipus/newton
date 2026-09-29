# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared MuJoCo solver replication without topology reconstruction."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT
from newton.solvers import SolverMuJoCo


def build_prototype() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [builder.add_joint_fixed(-1, root)]
    for i in range(3):
        position = (0.3 * i, 0.0, 0.09)
        body = builder.add_link(xform=wp.transform(position, wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        joints.append(
            builder.add_joint_prismatic(
                root,
                body,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform(position, wp.quat_identity()),
                target_ke=20.0,
                target_kd=2.0,
                limit_lower=-0.02,
                limit_upper=0.02,
            )
        )
    builder.add_articulation(joints)
    ball = builder.add_link(xform=wp.transform((1.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(ball, radius=0.1)
    builder.add_articulation([builder.add_joint_free(ball)])
    builder.add_ground_plane()
    return builder


@unittest.skipUnless(wp.is_cuda_available(), "MuJoCo replication requires CUDA")
class TestMuJoCoReplication(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_cuda_device()
        self.scope = wp.ScopedDevice(self.device)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def make_solver(self, count=1, *, implicit_limits=False, **kwargs):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(build_prototype(), count)
        model = builder.finalize(device=self.device)
        if implicit_limits:
            model.mujoco.solreflimit_mode.fill_(SOLREF_MODE_MJCF_DEFAULT)
        return SolverMuJoCo(model, njmax=64, nconmax=32, **kwargs)

    def test_replication_matches_fresh_construction_and_steps(self):
        for sleeping in (False, True):
            with self.subTest(sleeping=sleeping):
                source = self.make_solver(enable_sleeping=sleeping)
                reference = self.make_solver(3, enable_sleeping=sleeping)
                with ExitStack() as stack:
                    # Architecture gate: population expansion cannot rediscover
                    # topology or derive the already prepared physical model.
                    for owner, name in (
                        (SolverMuJoCo, "_convert_to_mjc"),
                        (SolverMuJoCo, "notify_model_changed"),
                        (SolverMuJoCo, "_set_const_0_with_physical_meaninertia"),
                        (newton.ModelBuilder, "finalize"),
                        (source._mujoco.MjSpec, "compile"),
                        (source._mujoco, "mj_setConst"),
                        (source._mujoco_warp, "put_model"),
                        (source._mujoco_warp, "put_data"),
                        (source._mujoco_warp, "make_data"),
                        (source._mujoco_warp, "set_const"),
                        (wp.array, "numpy"),
                    ):
                        stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
                    actual = source.replicate(3)
                self.assertEqual(actual.model.world_count, 3)
                for name, expected in vars(reference).items():
                    if not isinstance(expected, wp.array):
                        continue
                    received = getattr(actual, name)
                    self.assertEqual(received.shape, expected.shape, name)
                    if name != "_sleep_qpos":  # Deliberately uninitialized scratch.
                        np.testing.assert_array_equal(received.numpy(), expected.numpy(), err_msg=name)
                self.assertIsNot(actual.mj_model, source.mj_model)
                self.assertIsNot(actual.mj_data, source.mj_data)
                self.assertIsNot(actual.mjw_model.opt, source.mjw_model.opt)
                np.testing.assert_array_equal(actual.mj_model.exclude_signature, source.mj_model.exclude_signature)
                for name, value in vars(source).items():
                    if isinstance(value, wp.array) and value.size:
                        self.assertNotEqual(value.ptr, getattr(actual, name).ptr, name)

                actual_states = [actual.model.state(), actual.model.state()]
                reference_states = [reference.model.state(), reference.model.state()]
                actual_control = actual.model.control()
                reference_control = reference.model.control()
                actual_control.joint_target_q.fill_(-0.005)
                reference_control.joint_target_q.fill_(-0.005)
                for _ in range(4):
                    actual.step(*actual_states, actual_control, None, 0.002)
                    reference.step(*reference_states, reference_control, None, 0.002)
                    actual_states.reverse()
                    reference_states.reverse()
                np.testing.assert_allclose(
                    actual_states[0].joint_q.numpy(), reference_states[0].joint_q.numpy(), rtol=2.0e-5, atol=2.0e-6
                )
                np.testing.assert_allclose(
                    actual_states[0].joint_qd.numpy(), reference_states[0].joint_qd.numpy(), rtol=2.0e-5, atol=2.0e-6
                )
                # Changing a replica must not alter the prepared source.
                mass_before = source.mjw_model.body_mass.numpy().copy()
                actual.mjw_model.body_mass.fill_(3.0)
                np.testing.assert_array_equal(source.mjw_model.body_mass.numpy(), mass_before)

    def test_rejects_modified_and_non_template_solvers(self):
        source = self.make_solver()
        with self.assertRaisesRegex(ValueError, "single-world"):
            self.make_solver(2).replicate(3)
        source.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        with self.assertRaisesRegex(ValueError, "unmodified"):
            source.replicate(2)
        source = self.make_solver()
        source.reset(source.model.state())
        with self.assertRaisesRegex(ValueError, "unmodified"):
            source.replicate(2)
        source = self.make_solver(use_mujoco_contacts=False)
        with self.assertRaisesRegex(ValueError, "native-contact"):
            source.replicate(2)
        source = self.make_solver()
        source.step(source.model.state(), source.model.state(), source.model.control(), None, 0.002)
        with self.assertRaisesRegex(ValueError, "unstepped"):
            source.replicate(2)

    def test_world_transfer_preserves_live_episodes_and_captured_graph(self):
        """Move arbitrary same-prototype rows without reset or physical reconstruction."""
        prepared = self.make_solver(enable_sleeping=True, update_data_interval=2, implicit_limits=True)
        source, target = prepared.replicate(3), prepared.replicate(5)
        source_states = [source.model.state(), source.model.state()]
        target_states = [target.model.state(), target.model.state()]
        source_control, target_control = source.model.control(), target.model.control()
        # Runtime root frames, mass and material edits must move with their world.
        mass = source.model.body_mass.numpy()
        mass *= np.repeat([1.0, 1.2, 0.8], prepared.model.body_count)
        source.model.body_mass.assign(mass)
        frames = source.model.joint_X_p.numpy()
        frames[2 * prepared.model.joint_count, 0] += 0.05
        source.model.joint_X_p.assign(frames)
        source.model.shape_material_mu.fill_(0.7)
        source.model.set_gravity((0.0, 0.0, -7.0), world=2)
        gains = source.model.joint_limit_ke.numpy()
        gains[2 * prepared.model.joint_dof_count] += 10.0
        source.model.joint_limit_ke.assign(gains)
        source.notify_model_changed(
            newton.ModelFlags.BODY_INERTIAL_PROPERTIES
            | newton.ModelFlags.JOINT_PROPERTIES
            | newton.ModelFlags.SHAPE_PROPERTIES
            | newton.ModelFlags.MODEL_PROPERTIES
            | newton.ModelFlags.JOINT_DOF_PROPERTIES
        )
        source_control.joint_target_q.fill_(-0.008)
        for solver, states in ((source, source_states), (target, target_states)):
            newton.eval_fk(solver.model, states[0].joint_q, states[0].joint_qd, states[0])
            states[1].assign(states[0])
        for _ in range(6):
            source.step(*source_states, source_control, None, 0.002)
            source_states.reverse()
        with wp.ScopedCapture(device=self.device) as capture:
            for _ in range(2):
                target.step(*target_states, target_control, None, 0.002)
                target_states.reverse()
        source_ids, target_ids = [2, 0], [1, 4]
        untouched = target_states[0].joint_q.numpy().reshape(5, -1).copy()
        body_mass_before = prepared.model.body_mass.numpy().copy()
        pointers = {name: value.ptr for name, value in vars(target_states[0]).items() if isinstance(value, wp.array)}
        with ExitStack() as stack:
            for owner, name in (
                (SolverMuJoCo, "reset"),
                (SolverMuJoCo, "notify_model_changed"),
                (source._mujoco_warp, "forward"),
                (source._mujoco_warp, "put_model"),
                (source._mujoco_warp, "put_data"),
                (wp.array, "numpy"),
            ):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
            status = target.copy_worlds_from(
                source,
                source_ids,
                target_ids,
                states=tuple(zip(source_states, target_states, strict=True)),
                controls=((source_control, target_control),),
            )
        np.testing.assert_array_equal(status.numpy(), 0)
        for name, ptr in pointers.items():
            self.assertEqual(getattr(target_states[0], name).ptr, ptr, name)
        np.testing.assert_array_equal(prepared.model.body_mass.numpy(), body_mass_before)
        np.testing.assert_array_equal(target_states[0].joint_q.numpy().reshape(5, -1)[[0, 2, 3]], untouched[[0, 2, 3]])
        for source_obj, target_obj in (
            *zip(source_states, target_states, strict=True),
            (source_control, target_control),
        ):
            for name, value in vars(source_obj).items():
                if isinstance(value, wp.array) and value.size:
                    actual = getattr(target_obj, name).numpy().reshape(5, -1)[target_ids]
                    expected = value.numpy().reshape(3, -1)[source_ids]
                    np.testing.assert_array_equal(actual, expected, err_msg=name)
        for name in ("body_mass", "body_inertia", "joint_X_p", "shape_material_mu"):
            actual = getattr(target.model, name).numpy().reshape(5, -1)[target_ids]
            expected = getattr(source.model, name).numpy().reshape(3, -1)[source_ids]
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        np.testing.assert_array_equal(
            target.model.gravity.numpy()[target_ids], source.model.gravity.numpy()[source_ids]
        )
        for name in ("_joint_limit_ke_snapshot", "_joint_limit_kd_snapshot", "_solreflimit_mode_snapshot"):
            expected = getattr(source, name).reshape(3, -1)[source_ids]
            actual = getattr(target, name).reshape(5, -1)[target_ids]
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        for tick in range(32):
            wp.capture_launch(capture.graph)
            for _ in range(2):
                source.step(*source_states, source_control, None, 0.002)
                source_states.reverse()
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                actual = getattr(target_states[0], name).numpy().reshape(5, -1)[target_ids]
                expected = getattr(source_states[0], name).numpy().reshape(3, -1)[source_ids]
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5, err_msg=f"{tick}: {name}")
            np.testing.assert_array_equal(
                target.mjw_data.time.numpy()[target_ids], source.mjw_data.time.numpy()[source_ids]
            )
            np.testing.assert_array_equal(
                target.mjw_data.tree_asleep.numpy()[target_ids], source.mjw_data.tree_asleep.numpy()[source_ids]
            )

    def test_world_transfer_rejects_strided_custom_storage_before_native_mutation(self):
        """Keep native state, constraints and custom properties unchanged on an invalid layout."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(build_prototype(), 1)
        model = builder.finalize(self.device)
        model.custom_values = wp.array([[2.0, 3.0]], dtype=float, device=self.device)
        model.attribute_specs["custom_values"] = newton.Model.AttributeSpec(newton.Model.AttributeFrequency.WORLD)
        model.attribute_frequency["custom_values"] = newton.Model.AttributeFrequency.WORLD
        prepared = SolverMuJoCo(model, njmax=64, nconmax=32)
        for source_strided in (False, True):
            with self.subTest(source_strided=source_strided):
                source, target = prepared.replicate(2), prepared.replicate(3)
                source.mjw_data.qpos.fill_(0.25)
                source.mjw_data.efc.id.fill_(21)
                owner = source.model if source_strided else target.model
                owner.custom_values = wp.zeros((2 * owner.world_count, 2), dtype=float, device=self.device)[::2]
                arrays = (target.mjw_data.qpos, target.mjw_data.efc.id, target.model.custom_values)
                snapshots = [array.numpy().copy() for array in arrays]
                states = ((source.model.state(), target.model.state()),)
                controls = ((source.model.control(), target.model.control()),)
                with self.assertRaisesRegex(ValueError, "must be contiguous: custom_values"):
                    target.copy_worlds_from(source, [0], [0], states=states, controls=controls)
                for array, snapshot in zip(arrays, snapshots, strict=True):
                    np.testing.assert_array_equal(array.numpy(), snapshot)

    def test_world_transfer_preserves_uniform_cpu_history_without_materializing(self):
        prepared = self.make_solver(implicit_limits=True)
        names = ("_joint_limit_ke_snapshot", "_joint_limit_kd_snapshot", "_solreflimit_mode_snapshot")
        for case in ("uniform", "unequal", "writable", "materialized", "readonly_materialized", "nan"):
            with self.subTest(case=case):
                source, target = prepared.replicate(2), prepared.replicate(3)
                for name in names:
                    snapshot = getattr(source, name)
                    if case == "unequal":
                        row = snapshot[0].copy()
                        row[0] += 1
                        setattr(source, name, np.broadcast_to(row, snapshot.shape))
                    elif case in ("writable", "materialized", "readonly_materialized"):
                        snapshot = snapshot.copy()
                        if case != "writable":
                            snapshot[0, 0] += 1
                        if case == "materialized":
                            snapshot = snapshot.reshape(-1)
                        elif case == "readonly_materialized":
                            snapshot.flags.writeable = False
                        setattr(source, name, snapshot)
                    elif case == "nan" and name == names[0]:
                        row = snapshot[0].copy()
                        row[0] = np.nan
                        setattr(source, name, np.broadcast_to(row, snapshot.shape))
                        setattr(target, name, np.broadcast_to(row.copy(), getattr(target, name).shape))
                before = {name: getattr(target, name) for name in names}
                expected = {name: value.copy().reshape(3, -1) for name, value in before.items()}
                for name in names:
                    expected[name][[2, 0]] = getattr(source, name).reshape(2, -1)[[0, 1]]
                status = target.copy_worlds_from(
                    source,
                    [0, 1],
                    [2, 0],
                    states=((source.model.state(), target.model.state()),),
                    controls=((source.model.control(), target.model.control()),),
                )
                np.testing.assert_array_equal(status.numpy(), 0)
                for name in names:
                    actual = getattr(target, name)
                    np.testing.assert_array_equal(actual.reshape(3, -1), expected[name], err_msg=name)
                    if case == "uniform" or (case == "nan" and name != names[0]):
                        self.assertIs(actual, before[name])
                    else:
                        self.assertIsNot(actual, before[name])
                        self.assertTrue(actual.flags.writeable)

    def test_world_transfer_rejects_lineage_mapping_and_cadence(self):
        prepared = self.make_solver(update_data_interval=2)
        source, target = prepared.replicate(2), prepared.replicate(3)
        with self.assertRaisesRegex(ValueError, "same prepared solver"):
            target.copy_worlds_from(self.make_solver().replicate(2), [0], [0])
        with self.assertRaisesRegex(ValueError, "distinct"):
            source.copy_worlds_from(source, [0], [1])
        for src, dst in (([0, 0], [0, 1]), ([0, 1], [0, 0]), ([-1], [0]), ([0], [3])):
            with self.assertRaisesRegex(ValueError, "unique and in range"):
                target.copy_worlds_from(source, src, dst)
        source.step(source.model.state(), source.model.state(), source.model.control(), None, 0.002)
        with self.assertRaisesRegex(ValueError, "phase"):
            target.copy_worlds_from(source, [0], [0])

    def test_fresh_replica_captures_without_warmup(self):
        """Capture preserves fresh state and agrees with uncaptured first use."""
        source = self.make_solver(enable_sleeping=True, update_data_interval=2)
        actual = source.replicate(3)
        expected = self.make_solver(3, enable_sleeping=True, update_data_interval=2)
        actual_states = [actual.model.state(), actual.model.state()]
        expected_states = [expected.model.state(), expected.model.state()]
        actual_control, expected_control = actual.model.control(), expected.model.control()
        for solver, states in ((actual, actual_states), (expected, expected_states)):
            newton.eval_fk(solver.model, states[0].joint_q, states[0].joint_qd, states[0])
            states[1].assign(states[0])
        initial_q = actual_states[0].joint_q.numpy().copy()
        initial_time = actual.mjw_data.time.numpy().copy()
        # Capture records operations without stepping either snapshot. Two
        # substeps restore buffer parity and the update_data_interval phase.
        with wp.ScopedCapture(device=self.device) as capture:
            for _ in range(2):
                actual_states[0].clear_forces()
                actual.step(*actual_states, actual_control, None, 0.002)
                actual_states.reverse()
        np.testing.assert_array_equal(actual_states[0].joint_q.numpy(), initial_q)
        np.testing.assert_array_equal(actual.mjw_data.time.numpy(), initial_time)
        for tick in range(128):
            target = -0.005 if tick < 64 else 0.007
            actual_control.joint_target_q.fill_(target)
            expected_control.joint_target_q.fill_(target)
            wp.capture_launch(capture.graph)
            for _ in range(2):
                expected_states[0].clear_forces()
                expected.step(*expected_states, expected_control, None, 0.002)
                expected_states.reverse()
            if tick % 16 == 0 or tick == 127:
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    np.testing.assert_allclose(
                        getattr(actual_states[0], name).numpy(),
                        getattr(expected_states[0], name).numpy(),
                        rtol=2.0e-5,
                        atol=2.0e-6,
                        err_msg=f"{name} at tick {tick}",
                    )
                for name in ("nefc", "body_awake", "tree_asleep", "nacon"):
                    np.testing.assert_array_equal(
                        getattr(actual.mjw_data, name).numpy(), getattr(expected.mjw_data, name).numpy(), err_msg=name
                    )
                contact_rows = []
                for solver in (actual, expected):
                    count = solver.mjw_data.nacon.numpy()[0]
                    contact = solver.mjw_data.contact
                    ids = np.column_stack((contact.worldid.numpy()[:count], contact.geom.numpy()[:count]))
                    order = np.lexsort(ids.T[::-1])
                    contact_rows.append((ids[order], contact.dist.numpy()[:count][order]))
                np.testing.assert_array_equal(contact_rows[0][0], contact_rows[1][0])
                np.testing.assert_allclose(contact_rows[0][1], contact_rows[1][1], atol=2.0e-6)

    def test_replica_updates_preserve_joint_limit_edit_detection(self):
        source = self.make_solver(implicit_limits=True)
        actual = source.replicate(3)
        expected = self.make_solver(3, implicit_limits=True)
        np.testing.assert_array_equal(actual.model.mujoco.solreflimit_mode.numpy(), SOLREF_MODE_MJCF_DEFAULT)
        for solver in (actual, expected):
            solver.model.joint_limit_ke.fill_(200.0)
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_array_equal(actual.model.mujoco.solreflimit_mode.numpy(), SOLREF_MODE_FORCE_SPACE)
        np.testing.assert_allclose(actual.mjw_model.jnt_solref.numpy(), expected.mjw_model.jnt_solref.numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)

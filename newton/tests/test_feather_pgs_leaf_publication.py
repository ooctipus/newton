# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete state publication of the ordinary dynamics route (stage 7) on chains, leaves and branched trees."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


def _build_model(device="cpu", *, locked_d6=False, chain=False, leaves=7):
    """Use real joints, nontrivial anchors/COMs, and unrelated moving bodies."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))

    def body(index):
        return builder.add_link(
            mass=0.5 + 0.1 * index,
            com=wp.vec3(0.07, -0.03, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )

    base = body(0)
    root_kwargs = {
        "parent": -1,
        "child": base,
        "parent_xform": wp.transform(wp.vec3(2.0, -1.0, 0.6), wp.quat_rpy(0.3, -0.2, 0.4)),
        "child_xform": wp.transform(wp.vec3(0.2, -0.1, 0.05), wp.quat_rpy(-0.1, 0.2, 0.0)),
    }
    root = builder.add_joint_d6(**root_kwargs) if locked_d6 else builder.add_joint_fixed(**root_kwargs)
    joints = [root]
    leaf_bodies = []
    for index in range(leaves):
        child = body(index + 1)
        leaf_bodies.append(child)
        joints.append(
            builder.add_joint_prismatic(
                parent=(base if index == 0 else leaf_bodies[index - 1]) if chain else base,
                child=child,
                axis=wp.vec3(0.2 + 0.1 * index, 0.5, 0.9),
                parent_xform=wp.transform(wp.vec3(0.1 * index, -0.05, 0.13), wp.quat_rpy(0.2, -0.1 * index, 0.3)),
                child_xform=wp.transform(wp.vec3(-0.09, 0.04, 0.08), wp.quat_rpy(0.4, 0.2, -0.1)),
                target_ke=3.0,
                target_kd=0.2,
                limit_lower=-0.5,
                limit_upper=0.5,
            )
        )
    builder.add_articulation(joints)
    arm = body(leaves + 1)
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(-0.8, 0.1, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([hinge])
    free = body(leaves + 2)
    builder.add_articulation([builder.add_joint_free(free)])
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    q[: leaves + 1] = np.linspace(-0.31, 0.27, leaves + 1)
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    # The specialized producer must not assume a unit axis.
    axes = model.joint_axis.numpy()
    axes[0] *= 1.4
    model.joint_axis.assign(axes)
    return model, joints, leaf_bodies


def _build_branched_model(device="cpu", *, floating_root=False):
    """Interleave terminal joints and internal parents with nonzero parent motion."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))
    bodies = [
        builder.add_link(
            mass=0.7 + 0.1 * index,
            com=wp.vec3(0.03, -0.04, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )
        for index in range(7)
    ]
    root_args = {"parent": -1, "child": bodies[0]}
    root = builder.add_joint_free(**root_args) if floating_root else builder.add_joint_revolute(**root_args)
    joints = [root]
    # J1 is terminal despite preceding the internal J2. A prefix cannot express
    # the retained traversal [J0, J2]; later leaves use both moving parents.
    for index, kind in enumerate(("revolute", "revolute", "prismatic", "ball", "d6", "fixed"), start=1):
        args = {
            "parent": bodies[2] if index in (4, 5) else bodies[0],
            "child": bodies[index],
            "parent_xform": wp.transform(wp.vec3(0.1 * index, -0.07, 0.13), wp.quat_rpy(0.2, -0.1, 0.3)),
            "child_xform": wp.transform(wp.vec3(-0.03, 0.04, 0.02), wp.quat_rpy(-0.1, 0.2, 0.1)),
        }
        if kind in ("revolute", "prismatic"):
            args["axis"] = wp.vec3(0.2, 0.5, 0.9)
        if kind == "d6":
            args["linear_axes"] = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]
            args["angular_axes"] = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y)]
        joints.append(getattr(builder, f"add_joint_{kind}")(**args))
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    starts = model.joint_q_start.numpy()
    for joint in (1, 2, 3, 5):
        q[starts[joint] : starts[joint + 1]] = 0.07 * joint
    if floating_root:
        q[:7] = (0.4, -0.2, 0.6, *wp.quat_rpy(0.3, -0.2, 0.1))
    else:
        q[0] = 0.31
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.9, 0.8, model.joint_dof_count, dtype=np.float32))
    return model, joints, bodies


def _solver(model, *, mode="split"):
    solver = SolverFeatherPGS(
        model,
        pgs_mode=mode,
        pgs_kernel="auto" if mode == "matrix_free" else "loop",
        pgs_iterations=8,
        update_mass_matrix_interval=2,
        use_parallel_streams=False,
        cached_dynamics="off",
    )
    # Exercise the production cached Stage 7 on CPU, where it is normally off.
    if model.device.is_cpu:
        solver._fk_id_cache_enabled = True
    solver._prepare_augmented_state(model.state(), model.state(), model.control())
    return solver


def _fields(solver, state):
    return {
        "body_q": state.body_q,
        "body_qd": state.body_qd,
        "body_q_com": solver.body_q_com,
        "origin": solver.articulation_origin,
        "S": solver.joint_S_s,
        "v": solver.body_v_s,
        "a": solver.body_a_s,
        "I": solver.body_I_s,
        "terms": solver._body_inertia_terms,
        "f": solver.body_f_s,
        "valid": solver._fk_id_cache_valid,
    }


class TestLeafPublication(unittest.TestCase):
    def test_cuda_actual_graph_publication(self):
        """Match poisoned-buffer graph replays for prismatic and mixed moving trees."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph replay requires CUDA")
        for device in devices:
            self._check_graph_publication(_build_model(device, locked_d6=True, leaves=108)[0])
            self._check_graph_publication(_build_branched_model(device, floating_root=True)[0])

    def _check_graph_publication(self, model):
        device = model.device
        solver = _solver(model, mode="matrix_free")
        inputs = {
            name: getattr(model, name)
            for name in (
                "joint_X_p",
                "joint_X_c",
                "joint_axis",
                "body_com",
                "body_mass",
                "body_inertia",
                "gravity",
                "joint_type",
                "joint_parent",
                "joint_child",
                "joint_q_start",
                "joint_qd_start",
            )
        }
        before = {name: array.numpy().copy() for name, array in inputs.items()}
        for step in (0, 1):
            state = model.state()
            solver._step = step
            fields = _fields(solver, state)
            q, qd = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()

            def poison(outputs=fields):
                for name, array in outputs.items():
                    array.fill_(-123 if name == "valid" else -123.25)

            poison()
            solver._stage7_update_kinematics(state, solver)
            expected = {name: array.numpy().copy() for name, array in fields.items()}
            self.assertTrue(all(np.isfinite(value).all() for value in expected.values()))
            with wp.ScopedCapture(device=device) as capture:
                solver._stage7_update_kinematics(state, solver)
            for _ in range(2):
                poison()
                wp.capture_launch(capture.graph)
                for name, array in fields.items():
                    np.testing.assert_array_equal(array.numpy(), expected[name], err_msg=name)
            np.testing.assert_array_equal(state.joint_q.numpy(), q)
            np.testing.assert_array_equal(state.joint_qd.numpy(), qd)
        for name, array in inputs.items():
            np.testing.assert_array_equal(array.numpy(), before[name], err_msg=name)

    def test_interleaved_mixed_leaves_and_moving_parents(self):
        """Keep internal joints in order and preserve general moving-parent kinematics."""
        for floating in (False, True):
            model, _, bodies = _build_branched_model(floating_root=floating)
            solver = _solver(model)
            leaves = [bodies[index] for index in (1, 3, 4, 5, 6)]
            state = model.state()
            solver._step = 1
            for name, array in _fields(solver, state).items():
                array.fill_(-123 if name == "valid" else -123.25)
            solver._stage7_update_kinematics(state, solver)
            snapshot = {name: array.numpy().copy() for name, array in _fields(solver, state).items()}
            self.assertGreater(float(np.max(np.abs(snapshot["a"][leaves]))), 1e-3)
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            np.testing.assert_allclose(snapshot["body_q"], reference.body_q.numpy(), rtol=3e-6, atol=3e-6)
            # Ordinary FPGS already differs from eval_fk for mixed D6
            # translation/angular lever arms.
            ordinary_velocity = model.joint_type.numpy() != int(newton.JointType.D6)
            checked_bodies = model.joint_child.numpy()[ordinary_velocity]
            np.testing.assert_allclose(
                snapshot["body_qd"][checked_bodies],
                reference.body_qd.numpy()[checked_bodies],
                rtol=3e-6,
                atol=3e-6,
            )

    def test_complete_canonical_publication_and_held_inertia(self):
        """Publish every canonical field on each cadence arm: inertia records are placed only for a refresh
        step, and dense inertia only for free rigid bodies."""
        model, _, leaves = _build_model(locked_d6=True, leaves=108)
        solver = _solver(model)
        for step in (0, 1):
            state = model.state()
            solver._step = step
            for name, array in _fields(solver, state).items():
                array.fill_(-123 if name == "valid" else -123.25)
            solver._stage7_update_kinematics(state, solver)
            snapshot = {name: array.numpy().copy() for name, array in _fields(solver, state).items()}
            np.testing.assert_array_equal(snapshot["valid"], np.ones(model.articulation_count))
            np.testing.assert_array_equal(snapshot["a"][leaves], 0.0)
            np.testing.assert_array_equal(snapshot["I"][leaves], -123.25)
            if step == 0:
                np.testing.assert_array_equal(snapshot["terms"][leaves], -123.25)
            # Independent public FK computes COM velocities from q/qd, not our cache.
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            np.testing.assert_allclose(snapshot["body_q"], reference.body_q.numpy(), rtol=3e-6, atol=3e-6)
            np.testing.assert_allclose(snapshot["body_qd"], reference.body_qd.numpy(), rtol=3e-6, atol=3e-6)

    def test_current_root_frames_and_inertial_notifications(self):
        """Use notified current root/leaf frames, axes, COMs, mass, inertia, and gravity."""
        self._check_current_frames_and_notifications("cpu")

    def test_cuda_current_root_frames_and_inertial_notifications(self):
        """The same notifications on CUDA."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA notifications require CUDA")
        for device in devices:
            self._check_current_frames_and_notifications(device)

    def _check_current_frames_and_notifications(self, device):
        model, _, _ = _build_model(device)
        solver = _solver(model)
        state = model.state()
        solver._step = 1
        solver._stage7_update_kinematics(state, solver)
        before = state.body_q.numpy().copy()
        frames = model.joint_X_p.numpy()
        frames[0, :3] += (0.3, -0.7, 0.5)
        frames[1, :3] += (0.2, 0.4, -0.1)
        model.joint_X_p.assign(frames)
        child_frames = model.joint_X_c.numpy()
        child_frames[1, :3] += (-0.1, 0.07, 0.09)
        model.joint_X_c.assign(child_frames)
        axes = model.joint_axis.numpy()
        axes[0] = (0.4, -0.7, 1.2)
        model.joint_axis.assign(axes)
        com = model.body_com.numpy()
        com += (0.01, 0.02, -0.03)
        model.body_com.assign(com)
        model.body_mass.assign(model.body_mass.numpy() * 1.3)
        model.body_inertia.assign(model.body_inertia.numpy() * 1.2)
        model.gravity.assign(np.array([[1.0, -2.0, -8.0]], dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.ALL)
        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
        state = model.state()
        solver._stage7_update_kinematics(state, solver)
        self.assertIs(solver._fk_id_cache_source_state, state)
        reference = model.state()
        newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
        np.testing.assert_allclose(state.body_q.numpy(), reference.body_q.numpy(), rtol=3e-6, atol=3e-6)
        solver.reset(state)
        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
        self.assertFalse(np.array_equal(before, state.body_q.numpy()))


if __name__ == "__main__":
    unittest.main()

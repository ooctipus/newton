# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused controls for model-compiled coordinate-to-state ownership."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import compiled_coordinate_state, g1_kinetic_state
from tools.fpgs_bench.test_g1_kinetic_state import DT, public_state, rotated_input, set_anchor
from tools.fpgs_bench.test_sparse_factor import fixture
from tools.fpgs_bench.test_world_scan_publication import compose, inverse, rotate


@wp.kernel
def local_algebra(
    plan: compiled_coordinate_state.CoordinatePlan,
    q: wp.array[float],
    body_pose: wp.array[wp.transform],
    poses: wp.array[wp.transform],
    screws: wp.array[wp.spatial_vector],
):
    joint = wp.tid()
    slot = plan.index[joint]
    if slot >= 0:
        poses[joint] = compiled_coordinate_state.local_pose(plan, slot, q[joint])
        screws[joint] = compiled_coordinate_state.current_screw(plan, slot, body_pose[joint], wp.vec3(0.2, -0.1, 0.3))


class TestCompiledCoordinates(unittest.TestCase):
    def test_descriptor_api(self):
        self.assertTrue(callable(compiled_coordinate_state.build_coordinates))

    def test_scalar_algebra_dedup_and_nonrigid_fallback(self):
        builder = newton.ModelBuilder()
        qp = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, -1)), 0.37)
        qc = wp.quat_from_axis_angle(wp.normalize(wp.vec3(-2, 1, 3)), -0.21)
        xp = wp.transform(wp.vec3(0.04, -0.07, 0.09), qp)
        xc = wp.transform(wp.vec3(-0.03, 0.02, 0.01), qc)
        axis = wp.normalize(wp.vec3(1, -3, 2))
        for kind in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC, newton.JointType.REVOLUTE):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(0.1, 0, 0, 0, 0.2, 0, 0, 0, 0.3))
            if kind == newton.JointType.REVOLUTE:
                builder.add_joint_revolute(-1, body, parent_xform=xp, child_xform=xc, axis=axis)
            else:
                builder.add_joint_prismatic(-1, body, parent_xform=xp, child_xform=xc, axis=axis)
        model = builder.finalize(device="cpu")
        plan = compiled_coordinate_state.build_coordinates(model)
        np.testing.assert_array_equal(plan.index.numpy(), [0, 1, 0])
        values = np.asarray([0.8, -0.2, -0.7], np.float32)
        anchor = np.asarray(wp.transform(wp.vec3(0.7, -0.4, 0.2), wp.quat_from_axis_angle(axis, 0.31)), float)
        expected, poses, expected_s = [], [], []
        for joint, value in enumerate(values):
            moving = np.array([0, 0, 0, 0, 0, 0, 1], float)
            if joint == 1:
                moving[:3] = np.asarray(axis) * value
            else:
                moving[3:6] = np.asarray(axis) * np.sin(value / 2)
                moving[6] = np.cos(value / 2)
            relative = compose(compose(np.asarray(xp), moving), inverse(np.asarray(xc)))
            expected.append(relative)
            poses.append(compose(anchor, relative))
            actual_anchor = compose(anchor, np.asarray(xp))
            direction = rotate(actual_anchor[3:], np.asarray(axis))
            if joint == 1:
                expected_s.append(np.r_[direction, [0, 0, 0]])
            else:
                expected_s.append(np.r_[np.cross(actual_anchor[:3] - [0.2, -0.1, 0.3], direction), direction])
        result = wp.zeros(3, dtype=wp.transform, device="cpu")
        screws = wp.zeros(3, dtype=wp.spatial_vector, device="cpu")
        wp.launch(
            local_algebra,
            3,
            inputs=[
                plan,
                wp.array(values, dtype=float, device="cpu"),
                wp.array(np.asarray(poses), dtype=wp.transform, device="cpu"),
                result,
                screws,
            ],
            device="cpu",
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(screws.numpy(), expected_s, rtol=3e-6, atol=3e-6)
        changed = model.joint_axis.numpy()
        changed[0] *= 2
        model.joint_axis.assign(changed)
        self.assertEqual(int(compiled_coordinate_state.build_coordinates(model).index.numpy()[0]), -1)

    def test_complete_current_held_and_kinematic_publication(self):
        results = []
        for enabled in (False, True):
            with patch.dict(
                os.environ,
                {"FEATHER_PGS_G1_CHAIN_SCAN": "1", "FEATHER_PGS_COMPILED_COORDINATE_STATE": str(int(enabled))},
            ):
                case = fixture("cpu")
                model, solver, sparse = case["model"], case["solver"], case["owner"]
                set_anchor(model)
                values = model.joint_X_c.numpy()
                values[1, :3] = [-0.012, 0.017, -0.008]
                values[1, 3:] = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.23))
                model.joint_X_c.assign(values)
                owner = g1_kinetic_state.G1KineticState(sparse)
            self.assertEqual(owner.compiled_coordinate_state, enabled)
            self.assertEqual(owner.finish_kernel.key.endswith("_compiled"), enabled)
            W = np.linalg.solve(np.linalg.cholesky(case["H"][::-1, ::-1]), np.eye(43))
            packed = W[sparse.host["row"], sparse.host["col"]][None].astype(np.float32)
            sparse.data.W.assign(packed)
            sparse.data.valid.fill_(1)
            control = model.control()
            control.joint_f.assign(np.linspace(-0.2, 0.4, 43, dtype=np.float32))
            epochs = []
            for epoch in range(3):
                state, output = model.state(), model.state()
                rotated_input(model, state, epoch=epoch)
                augmented = solver._prepare_augmented_state(state, output, control)
                augmented.joint_tau.assign(np.linspace(0.02, -0.01, 43, dtype=np.float32))
                owner.invalidate()
                owner.begin(state, augmented, state.joint_qd, DT, epoch != 1)
                owner.predict(state, augmented, control, state.joint_qd, DT)
                vhat = solver.v_hat.numpy().copy()
                solver.v_out.assign(state.joint_qd.numpy() + np.linspace(0.1, -0.1, 43, dtype=np.float32))
                if epoch == 2:
                    dofs = solver._kinematic_dof_mask.numpy()
                    joints = solver._kinematic_joint_mask.numpy()
                    dofs[6], joints[1] = 1, 1
                    solver._kinematic_dof_mask.assign(dofs)
                    solver._kinematic_joint_mask.assign(joints)
                owner.finish(state, augmented, output, DT, epoch != 1)
                public_state(self, model, output)
                np.testing.assert_array_equal(owner.status.numpy(), [0])
                np.testing.assert_array_equal(sparse.data.W.numpy(), packed)
                if epoch == 2:
                    self.assertEqual(float(solver.v_out.numpy()[6]), float(state.joint_qd.numpy()[6]))
                    self.assertEqual(float(solver.joint_qdd.numpy()[6]), 0.0)
                    self.assertEqual(float(output.joint_q.numpy()[7]), float(state.joint_q.numpy()[7]))
                epochs.append(
                    (
                        vhat,
                        output.joint_q.numpy(),
                        output.joint_qd.numpy(),
                        solver.joint_qdd.numpy(),
                        owner.bias.numpy(),
                        owner.geometric.numpy(),
                        solver.joint_S_s.numpy(),
                    )
                )
            results.append(epochs)
        for old, new in zip(results[0], results[1], strict=True):
            for expected, actual in zip(old, new, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()

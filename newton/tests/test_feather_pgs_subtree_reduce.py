# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The prefix-scan subtree reduction of the tree finish kernel matches the depth-level loop."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton._src.solvers.feather_pgs.tree_kinetic_owner import dfs_intervals
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT, _build


class TestDepthFirstIntervals(unittest.TestCase):
    def test_breadth_first_input_is_reordered(self):
        """Legs added breadth-first still get one contiguous interval per subtree."""
        order, end = dfs_intervals([-1, 0, 0, 1, 2, 3])
        self.assertEqual(order.tolist(), [0, 1, 3, 5, 2, 4])
        self.assertEqual(end.tolist(), [6, 4, 4, 4, 6, 6])

    def test_forest_roots_are_separate_intervals(self):
        order, end = dfs_intervals([-1, 0, -1, 2, 2])
        self.assertEqual(order.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(end.tolist(), [2, 2, 5, 4, 5])

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            dfs_intervals([-1, 2, 1])


def _interleaved_legs(device, worlds=4):
    """A free trunk whose two three-joint legs are added breadth-first: parent-first but not depth-first."""
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    stance = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
    trunk = scene.add_link(xform=stance, mass=4.0, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
    scene.add_shape_box(trunk, hx=0.2, hy=0.1, hz=0.05, cfg=cfg)
    joints = [scene.add_joint_free(child=trunk, parent_xform=stance)]
    parents = [trunk, trunk]
    for segment in range(3):
        for leg in range(2):
            body = scene.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
            origin = wp.vec3(0.15 * (1 if leg else -1), 0.0, 0.0) if segment == 0 else wp.vec3(0.0, 0.0, -0.15)
            joints.append(
                scene.add_joint_revolute(
                    parent=parents[leg],
                    child=body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(origin, wp.quat_identity()),
                    target_ke=30.0,
                    target_kd=2.0,
                    limit_lower=-1.5,
                    limit_upper=1.5,
                )
            )
            if segment == 2:
                scene.add_shape_sphere(body, radius=0.03, cfg=cfg)
            parents[leg] = body
    scene.add_articulation(joints)
    # A free box keeps the world on the dense route (the mixed owner's class).
    box = scene.add_body(xform=wp.transform(wp.vec3(0.6, 0.0, 0.06), wp.quat_identity()), mass=0.5)
    scene.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 512
    return model


def _rollout(model, *, level_loop, steps=60):
    env = {
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE": "1",
        "FEATHER_PGS_TREE_KINETIC_STATE": "0",
        "FEATHER_PGS_RESOLVED_WORLDS": "0",
        "FEATHER_PGS_TREE_REFERENCE": str(int(level_loop)),
    }
    with patch.dict(os.environ, env):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=2,
            friction_anchor_beta=0.0,
        )
    owner = getattr(solver, "_tree_kinetic_state", None)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_target_q.assign(np.zeros_like(model.joint_target_q.numpy()))
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    types, qd_start = model.joint_type.numpy(), model.joint_qd_start.numpy()
    for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[::2]:
        qd[qd_start[joint]] = 1.0
    states[0].joint_qd.assign(qd)
    poses, biases = [], []
    for _ in range(steps):
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        poses.append(states[0].body_q.numpy().copy())
        biases.append(np.concatenate([group.data.bias.numpy().reshape(-1) for group in owner.groups]))
    wp.synchronize_device(model.device)
    return owner, np.array(poses), np.array(biases)


@unittest.skipUnless(wp.is_cuda_available(), "The held-tree owner is a CUDA path")
class TestPrefixSubtreeReduction(unittest.TestCase):
    def _compare(self, model, *, bias_tolerance=(1e-5, 1e-4)):
        _owner, plain, plain_bias = _rollout(model, level_loop=True)
        owner, poses, bias = _rollout(model, level_loop=False)
        self.assertIsNotNone(owner)
        self.assertTrue(all(group.layout.lanes <= 32 for group in owner.groups))
        self.assertTrue(np.isfinite(poses).all())
        # Both forms add the same body terms in a different grouping; they differ by float rounding,
        # amplified where a subtree's terms cancel (a walker landing on its feet).
        np.testing.assert_allclose(bias, plain_bias, rtol=bias_tolerance[0], atol=bias_tolerance[1])
        np.testing.assert_allclose(poses[:, :, :3], plain[:, :, :3], atol=1e-5, rtol=0.0)
        np.testing.assert_allclose(poses[:, :, 3:], plain[:, :, 3:], atol=1e-5, rtol=0.0)
        return owner

    def test_forest_with_prescribed_root(self):
        """Arm, free box and kinematic tray: one forest, three intervals."""
        self._compare(_build(wp.get_device("cuda:0"), worlds=6, prescribed=True))

    def test_branching_tree_added_breadth_first(self):
        """A trunk with two legs interleaved in joint order: the plan reorders lanes depth-first."""
        owner = self._compare(_interleaved_legs(wp.get_device("cuda:0")))
        order = owner.groups[0].plan.dfs_order.numpy().tolist()
        self.assertNotEqual(order, sorted(order))

    def test_free_root_walker(self):
        """An 18-coordinate quadruped beside the arm: two groups, both on the prefix reduction."""
        self._compare(_build(wp.get_device("cuda:0"), worlds=4, quadruped=True), bias_tolerance=(1e-4, 2e-3))


if __name__ == "__main__":
    unittest.main()

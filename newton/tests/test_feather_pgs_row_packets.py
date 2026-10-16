# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general row-packet producer builds a world's prefix and contact packets in place."""

import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import row_packets as rp
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT, _build


class TestRowPacketPlan(unittest.TestCase):
    def test_prefix_topology_validation(self):
        """Every prefix entry is checked, including disabled mimic entries."""
        groups = np.array([0, 2], dtype=np.int32)  # primaries of two worlds
        worlds = np.array([0, 0, 1, 1], dtype=np.int32)
        starts = np.array([0, 3, 9, 12, 18], dtype=np.int32)  # 3-DOF arms and 6-DOF boxes
        mimic_start = np.array([0, 1, 1, 2, 2], dtype=np.int32)
        mimic_list = np.array([0, 1], dtype=np.int32)
        dof0 = np.array([1, 10], dtype=np.int32)
        dof1 = np.array([2, 11], dtype=np.int32)
        mimic_world = np.array([0, 1], dtype=np.int32)
        ok = rp.validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, dof0, dof1, 3, 1, mimic_world)
        self.assertTrue(ok)
        # A mimic coefficient outside its primary's coordinates is rejected.
        bad = np.array([1, 3], dtype=np.int32)
        self.assertFalse(
            rp.validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, dof0, bad, 3, 1, mimic_world)
        )
        # More mimic entries than the prefix admits are rejected.
        self.assertFalse(
            rp.validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, dof0, dof1, 3, 0, mimic_world)
        )
        # The primary's coordinate count must match the layout.
        self.assertFalse(
            rp.validate_prefix_topology(groups, worlds, starts, mimic_start, mimic_list, dof0, dof1, 4, 1, mimic_world)
        )

    def test_prefix_kernel_sizes_come_from_the_layout(self):
        """Any primary whose candidates fit the packet capacity; nothing names a size."""
        for primary, mimics, packets in ((3, 0, 40), (9, 2, 40), (19, 2, 40), (7, 0, 14)):
            self.assertTrue(rp.get_prefix_kernel(primary, mimics, packets))
        for primary, mimics, packets in ((20, 1, 40), (0, 0, 40), (9, 3, 20)):
            with self.subTest(primary=primary), self.assertRaises(ValueError):
                rp.get_prefix_kernel(primary, mimics, packets)

    def test_module_names_no_robot(self):
        source = inspect.getsource(rp).lower()
        for token in ("franka", "kuka", "allegro", "[6, 9]", "size_groups) != [", "== 9"):
            self.assertNotIn(token, source)


def _rollout(model, *, packets, steps=120, swing_stride=3):
    """Swing the arms over the boxes; return the solver and the body poses."""
    env = {
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE": "1",
        "FEATHER_PGS_TREE_KINETIC_STATE": "0",
        "FEATHER_PGS_RESOLVED_WORLDS": "0",
        "FEATHER_PGS_ROW_PACKETS": str(int(packets)),
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
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_target_q.assign(np.zeros_like(model.joint_target_q.numpy()))
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    types, qd_start = model.joint_type.numpy(), model.joint_qd_start.numpy()
    for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[::swing_stride]:
        qd[qd_start[joint]] = 1.5
    states[0].joint_qd.assign(qd)
    poses = []
    for _ in range(steps):
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        poses.append(states[0].body_q.numpy().copy())
    wp.synchronize_device(model.device)
    return solver, np.array(poses)


@unittest.skipUnless(wp.is_cuda_available(), "The local solve owners are a CUDA path")
class TestRowPacketsCUDA(unittest.TestCase):
    def _compare(self, model, **kwargs):
        plain_solver, plain = _rollout(model, packets=False, **kwargs)
        self.assertIsNone(plain_solver._row_packets)
        solver, poses = _rollout(model, packets=True, **kwargs)
        self.assertIsInstance(solver._row_packets, rp.LocalRowPackets)
        self.assertTrue(np.isfinite(poses).all())
        np.testing.assert_allclose(poses[:, :, :3], plain[:, :, :3], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(poses[:, :, 3:], plain[:, :, 3:], atol=1.0e-4, rtol=0.0)
        return solver

    def test_arm_and_box_match_the_ordinary_producers(self):
        """A fixed-base arm over a free box: the packet owner tracks the ordinary row producers."""
        model = _build(wp.get_device("cuda:0"), worlds=6)
        solver = self._compare(model)
        self.assertEqual((solver._row_packets.primary, solver._row_packets.secondary), (3, 6))
        self.assertEqual(solver._row_packets.mimics, 0)

    def test_mimic_rows_join_the_prefix(self):
        """A following finger joint: the mimic row is built in the prefix with the limits."""
        model = _build(wp.get_device("cuda:0"), worlds=6, mimic=True)
        solver = self._compare(model)
        self.assertEqual(solver._row_packets.mimics, 1)
        self.assertEqual(int(solver._mimic_count), 6)

    def test_longer_arm_takes_the_same_owner(self):
        """Nothing in the owner is sized for one arm: a five-joint arm is admitted the same way."""
        model = _build(wp.get_device("cuda:0"), worlds=4, links=5)
        solver = self._compare(model)
        self.assertEqual(solver._row_packets.primary, 5)


if __name__ == "__main__":
    unittest.main()

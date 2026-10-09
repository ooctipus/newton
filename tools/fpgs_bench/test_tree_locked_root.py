# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify zero-coordinate D6 roots using the existing fixed-root arithmetic."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.tree_kinetic_plan import make_plan
from tools.fpgs_bench import test_tree_prescaled_state as prescaled


@contextmanager
def locked_roots():
    """Build real locked D6 roots while retaining fixed descendant joints."""
    original = newton.ModelBuilder.add_joint_fixed

    def add(builder, parent, child, **kwargs):
        if parent == -1:
            return builder.add_joint_d6(parent, child, **kwargs)
        return original(builder, parent, child, **kwargs)

    with patch.object(newton.ModelBuilder, "add_joint_fixed", add):
        yield


class TestLockedD6Root(unittest.TestCase):
    def test_physical_motion_cpu(self):
        """Match locked-root physical mass, bias, live force and mutable motion."""
        with locked_roots():
            prescaled.TestPrescaledTreeContract.check_current_motion(self, "cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_physical_motion_cuda(self):
        """Preserve locked-root motion and excluded coordinates under graph replay."""
        with locked_roots():
            prescaled.TestPrescaledTreeContract.check_current_motion(self, "cuda:0", exclude_driven=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_loaded_steps_graph_and_reset(self):
        """Retain loaded fixed-root behavior through held factors and masked resets."""
        with locked_roots():
            prescaled.TestPrescaledTreeContract.test_loaded_steps_and_masked_reset(self)
            prescaled.TestPrescaledTreeContract.test_loaded_graph_replay_and_reset(self)

    def test_moving_d6_root_stays_excluded(self):
        """Reject a moving D6 root before allocating a cached-tree owner."""
        scene = newton.ModelBuilder()
        root, child = [scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
        joints = [
            scene.add_joint_d6(-1, root, linear_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]),
            scene.add_joint_revolute(root, child, axis=newton.Axis.Y),
        ]
        scene.add_articulation(joints)
        builder = newton.ModelBuilder()
        builder.replicate(scene, 2)
        model = builder.finalize(device="cpu")
        mapping = wp.array([0, 1], dtype=int, device="cpu")
        operator = SimpleNamespace(plan=SimpleNamespace(group_to_art=mapping, art_to_world=mapping))
        with self.assertRaisesRegex(ValueError, "root"):
            make_plan(model, operator)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The prefix-scan subtree reduction of the tree finish kernel matches the depth-level loop."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.tree_plan import TreePlan


def _excess(actual, desired, rtol=0.0):
    """Largest ``|actual - desired| - rtol * |desired|``: within ``atol`` exactly when assert_allclose passes."""
    actual, desired = np.asarray(actual, np.float64), np.asarray(desired, np.float64)
    return max(float(np.max(np.abs(actual - desired) - rtol * np.abs(desired))), 0.0)


def _assert_switch_within(test, label, switch, again, tolerance):
    """Bound an A/B difference by the tolerance plus the default arm's own run-to-run (A/A) spread.

    The A/A spread must itself stay within the tolerance: a larger one means the rollout's
    nondeterminism, not the switch, decides the comparison.
    """
    test.assertLessEqual(
        again,
        tolerance,
        f"{label}: the default arm differs from itself by {again:.3g} (> {tolerance:.3g}); "
        "the rollout cannot resolve the switch",
    )
    test.assertLessEqual(
        switch,
        tolerance + again,
        f"{label}: the switch changes it by {switch:.3g}, above {tolerance:.3g} plus the A/A spread {again:.3g}",
    )


class TestDepthFirstIntervals(unittest.TestCase):
    def test_breadth_first_input_is_reordered(self):
        """Legs added breadth-first still get one contiguous interval per subtree."""
        plan = TreePlan.build([-1, 0, 0, 1, 2, 3])
        self.assertEqual(plan.dfs_order.tolist(), [0, 1, 3, 5, 2, 4])
        self.assertEqual(plan.dfs_end.tolist(), [6, 4, 4, 4, 6, 6])

    def test_forest_roots_are_separate_intervals(self):
        plan = TreePlan.build([-1, 0, -1, 2, 2])
        self.assertEqual(plan.dfs_order.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(plan.dfs_end.tolist(), [2, 2, 5, 4, 5])

    def test_cycle_is_rejected(self):
        with self.assertRaises(ValueError):
            TreePlan.build([-1, 2, 1])


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


if __name__ == "__main__":
    unittest.main()

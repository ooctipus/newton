# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the FP64 public-state oracle that the held-tree cache tests compare against."""

import unittest

import numpy as np
import warp as wp

import newton
from tools.fpgs_bench.fp64_transforms import rotate
from tools.fpgs_bench.public_state_oracle import public_reference


def anchored_tree(device):
    """Build one free-root tree with revolute, prismatic and welded links and authored root anchors."""
    builder = newton.ModelBuilder()
    parents = [-1, 0, 0, 1, 2, 1, 4]
    bodies, joints = [], []
    for index, parent in enumerate(parents):
        body = builder.add_link(
            mass=0.5 + 0.2 * index,
            com=wp.vec3(0.01 * index, -0.02, 0.015),
            inertia=wp.mat33(np.diag([0.04, 0.05, 0.07])),
        )
        bodies.append(body)
        anchor = wp.transform(wp.vec3(0.07, 0.03 * index, -0.08), wp.quat_identity())
        if index == 0:
            joint = builder.add_joint_free(
                body,
                parent_xform=wp.transform((0.11, -0.07, 0.04), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.23)),
                child_xform=wp.transform((-0.03, 0.02, 0.01), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.17)),
            )
        elif index == 5:
            joint = builder.add_joint_fixed(bodies[parent], body, parent_xform=anchor)
        else:
            add = builder.add_joint_prismatic if index == 3 else builder.add_joint_revolute
            joint = add(bodies[parent], body, axis=newton.Axis.Y, parent_xform=anchor)
        joints.append(joint)
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def moving_state(model):
    """Give the root an arbitrary orientation and every joint a nonzero position and rate."""
    state = model.state()
    q = model.joint_q.numpy().copy()
    q[:3] = (0.4, -0.3, 0.9)
    q[3:7] = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -2.0, 3.0)), 0.31)
    q[7:] += np.linspace(-0.06, 0.07, len(q) - 7, dtype=np.float32)
    state.joint_q.assign(q)
    state.joint_qd.assign(np.linspace(-0.4, 0.6, model.joint_dof_count, dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


class TestPublicStateOracle(unittest.TestCase):
    def test_anchored_root_translation_and_rest(self):
        """Match generic FK, translate rigidly with the anchored root, and rest at zero rates."""
        model = anchored_tree("cpu")
        state = moving_state(model)
        reference = public_reference(model, state)
        np.testing.assert_allclose(state.body_q.numpy(), reference["body_q"], rtol=2e-5, atol=3e-6)
        np.testing.assert_allclose(state.body_qd.numpy(), reference["body_qd"], rtol=2e-5, atol=3e-6)

        # A world translation of the root moves every pose rigidly along the
        # root anchor's axes and leaves the COM twists unchanged.
        before = state.joint_q.numpy().copy()
        shifted = before.copy()
        shifted[:3] += np.array([-36.0, -76.0, 0.0], np.float32)
        state.joint_q.assign(shifted)
        translated = public_reference(model, state)
        displacement = rotate(
            model.joint_X_p.numpy()[0, 3:].astype(np.float64), shifted[:3].astype(np.float64) - before[:3]
        )
        np.testing.assert_allclose(
            translated["body_q"][:, :3], reference["body_q"][:, :3] + displacement, atol=3e-12, rtol=0
        )
        np.testing.assert_allclose(translated["body_q"][:, 3:], reference["body_q"][:, 3:], atol=3e-12, rtol=0)
        np.testing.assert_allclose(translated["body_qd"], reference["body_qd"], atol=3e-12, rtol=0)

        state.joint_qd.zero_()
        np.testing.assert_array_equal(public_reference(model, state)["body_qd"], np.zeros((model.body_count, 6)))


if __name__ == "__main__":
    unittest.main()

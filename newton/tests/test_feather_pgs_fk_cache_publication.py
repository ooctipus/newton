# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Graph replays of a split-mode contact scene: the published state advances and replays deterministically."""

import hashlib
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS


def _build(device):
    """Three articulated links per world with fabricated ground contacts on the last link."""
    builder = newton.ModelBuilder()
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    shape_cfg.mu = 0.6
    shapes = []
    for world in range(6):
        builder.begin_world()
        parent = -1
        joints = []
        for link in range(3):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
            if link == 0:
                joint = builder.add_joint_prismatic(parent=parent, child=body, axis=newton.Axis.Z)
            else:
                joint = builder.add_joint_revolute(
                    parent=parent,
                    child=body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.3), wp.quat_identity()),
                )
            joints.append(joint)
            parent = body
        builder.add_articulation(joints)
        shapes.append(
            builder.add_shape_sphere(
                parent,
                radius=0.02,
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.05 - 0.002 * world), wp.quat_identity()),
                cfg=shape_cfg,
            )
        )
        builder.end_world()
    plane = builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = len(shapes)
    contacts = newton.Contacts(rigid_contact_max=len(shapes), soft_contact_max=0, device=device)
    count = len(shapes)
    contacts.rigid_contact_count.assign(np.array([count], dtype=np.int32))
    contacts.rigid_contact_shape0.assign(np.asarray(shapes, dtype=np.int32))
    contacts.rigid_contact_shape1.assign(np.full(count, plane, dtype=np.int32))
    points = np.zeros((count, 3), dtype=np.float32)
    points[:, 2] = -0.001
    contacts.rigid_contact_point0.assign(points)
    contacts.rigid_contact_point1.assign(np.zeros((count, 3), dtype=np.float32))
    contacts.rigid_contact_normal.assign(np.tile([0.0, 0.0, -1.0], (count, 1)).astype(np.float32))
    return model, contacts


def _digest(state):
    return hashlib.sha256(
        b"".join(getattr(state, name).numpy().tobytes() for name in ("joint_q", "joint_qd", "body_q", "body_qd"))
    ).hexdigest()


@unittest.skipUnless(wp.is_cuda_available(), "Graph replays are a CUDA path")
class TestFeatherPGSFkCachePublication(unittest.TestCase):
    def _run(self, replays, substeps=3, dt=0.005):
        device = wp.get_device("cuda:0")
        model, contacts = _build(device)
        solver = SolverFeatherPGS(
            model,
            dense_max_constraints=16,
            pgs_iterations=6,
            friction_anchor_beta=0.0,
        )
        states = [model.state() for _ in range(2)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()

        def substep_block():
            for _ in range(substeps):
                solver.step(states[0], states[1], control, contacts, dt)
                states.reverse()

        # Step one block eagerly, then capture one block and replay it.
        substep_block()
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            substep_block()
        digests = []
        for _ in range(replays):
            wp.capture_launch(capture.graph)
            wp.synchronize_device(device)
            digests.append(_digest(states[0]))
        self.assertTrue(np.isfinite(states[0].joint_q.numpy()).all())
        return digests

    def test_in_place_publication_advances_across_replays(self):
        """Advance the trajectory at every replay: the published state is not frozen."""
        self.assertGreater(len(set(self._run(replays=40))), 30)

    def test_in_place_publication_is_deterministic(self):
        """Two independent captures replay identically."""
        self.assertEqual(self._run(replays=20), self._run(replays=20))


if __name__ == "__main__":
    unittest.main()

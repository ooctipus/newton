# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The fused general-route publication reproduces the separate launches bit for bit."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.test_feather_pgs_sparse_warmstart import _build as build_legged


def _build_pendulums(device, worlds=5):
    """Fixed-base double pendulums whose tips touch the ground through the collision pipeline."""
    builder = newton.ModelBuilder()
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    shape_cfg.mu = 0.5
    for world in range(worlds):
        builder.begin_world()
        parent = -1
        joints = []
        for link in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.42 if link == 0 else -0.2), wp.quat_identity()),
                    target_ke=5.0 * world,
                    target_kd=1.0,
                )
            )
            parent = body
        builder.add_shape_sphere(
            parent, radius=0.03, xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()), cfg=shape_cfg
        )
        builder.add_articulation(joints)
        builder.end_world()
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _rollout(model, fused, *, steps=48, dt=1.0 / 240.0, graph=False, **solver_kwargs):
    device = model.device
    with patch.object(solver_module, "_FUSED_PUBLISH", fused):
        solver = SolverFeatherPGS(model, pgs_iterations=6, **solver_kwargs)
        admitted = solver._fused_publication_admitted()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    states[0].joint_qd.fill_(0.2)
    control = model.control()

    def substep(state_in, state_out):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, dt)

    captured = None
    for i in range(steps):
        pipeline.collide(states[0], contacts)
        if graph and device.is_cuda:
            if captured is None:
                captured = []
                for order in ((states[0], states[1]), (states[1], states[0])):
                    with wp.ScopedCapture(device=device) as capture:
                        solver.seed_double_buffer_events()
                        substep(*order)
                    captured.append(capture.graph)
            wp.capture_launch(captured[i % 2])
        else:
            substep(states[0], states[1])
        states.reverse()
    wp.synchronize_device(device)
    state = states[0]
    return admitted, {
        "joint_q": state.joint_q.numpy().copy(),
        "joint_qd": state.joint_qd.numpy().copy(),
        "body_q": state.body_q.numpy().copy(),
        "body_qd": state.body_qd.numpy().copy(),
    }


@unittest.skipUnless(wp.is_cuda_available(), "The FK/ID cache publication is a CUDA path")
class TestFeatherPGSFusedPublication(unittest.TestCase):
    def check_identity(self, model, expect_admitted, **solver_kwargs):
        fused_admitted, fused = _rollout(model, True, **solver_kwargs)
        plain_admitted, plain = _rollout(model, False, **solver_kwargs)
        self.assertEqual(fused_admitted, expect_admitted)
        self.assertFalse(plain_admitted)
        for name in fused:
            self.assertTrue(np.isfinite(fused[name]).all(), name)
            np.testing.assert_array_equal(fused[name], plain[name], err_msg=name)
        return fused

    def test_floating_base_identity(self):
        """Free-root legged articulations publish identical states with one launch per size group."""
        model, _ = build_legged(wp.get_device("cuda:0"))
        fused = self.check_identity(model, True, pgs_mode="split")
        torso = fused["body_q"].reshape(model.world_count, -1, 7)[:, 0, 2]
        self.assertTrue(np.all(torso > 0.4), "the stance must still be standing")

    def test_fixed_base_identity_dense_and_split(self):
        """Fixed-base chains publish identical states in dense and split compositions."""
        model = _build_pendulums(wp.get_device("cuda:0"))
        for mode in ("split", "matrix_free"):
            with self.subTest(pgs_mode=mode):
                self.check_identity(model, True, pgs_mode=mode)

    def test_graph_replay_identity(self):
        """Captured substeps replay the fused publication identically to eager execution."""
        model, _ = build_legged(wp.get_device("cuda:0"))
        _, eager = _rollout(model, True, pgs_mode="split")
        _, replayed = _rollout(model, True, pgs_mode="split", graph=True)
        for name in eager:
            np.testing.assert_array_equal(eager[name], replayed[name], err_msg=name)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check isotropic friction against the direction and magnitude of Coulomb sliding."""

import unittest

import numpy as np
import warp as wp

import newton


def _slide(device, beta, mesh, angle, articulated=False, *, locked_tangent=False):
    """Slide a symmetric, explicitly massed box under a force above the friction limit.

    ``articulated`` builds the box as a D6-jointed articulation instead of a free body.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, gap=0.0, restitution=0.0)
    builder.add_shape_box(-1, hx=2.0, hy=2.0, hz=0.05, xform=wp.transform((0, 0, -0.05)), cfg=cfg)
    inertia = np.diag([(0.2**2 + 0.05**2) / 12] * 2 + [2 * 0.2**2 / 12]).astype(np.float32)
    add_body = builder.add_link if articulated else builder.add_body
    body = add_body(xform=wp.transform((0, 0, 0.025)), mass=1.0, inertia=wp.mat33(inertia), lock_inertia=True)
    if articulated:
        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            parent_xform=wp.transform((0, 0, 0.025)),
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(axis=axis)
                for axis in ((newton.Axis.X, newton.Axis.Z) if locked_tangent else newton.Axis)
            ],
        )
        builder.add_articulation([joint])
    if mesh:
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.1, 0.1, 0.025), cfg=cfg)
    else:
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.025, cfg=cfg)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        friction_anchor_beta=beta,
        pgs_iterations=64,
        pgs_cfm=0.0 if locked_tangent else 1.0e-6,
        angular_damping=0.0,
        mf_max_constraints=128,
        dense_max_constraints=32,
        row_watermark=True,
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
    contacts = pipeline.contacts()
    state, next_state = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    direction = np.array([np.cos(angle), np.sin(angle)])
    force = np.zeros((1, 6), dtype=np.float32)
    force[0, :2] = direction * 1.2 * 0.3 * 9.81
    hz = 240
    for step in range(hz // 5 + hz // 2):
        state.clear_forces()
        if step >= hz // 5:
            state.body_f.assign(force)
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / hz)
        state, next_state = next_state, state
    return state.body_qd.numpy()[0, :2], direction, solver.constraint_row_watermarks()


class TestFeatherPGSFrictionDirection(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_articulated_sliding_direction(self):
        """Preserve sliding direction through the dense articulated route."""
        velocity, direction, _ = _slide("cuda:0", 0.2, False, np.pi / 4, articulated=True)
        np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_locked_tangent_preserves_other_friction_direction(self):
        """Retain friction along a slider when its first tangent has zero response."""
        for beta in (0.0, 0.2):
            with self.subTest(beta=beta):
                velocity, direction, watermarks = _slide(
                    "cuda:0", beta, False, 0.0, articulated=True, locked_tangent=True
                )
                np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
                for key, value in watermarks.items():
                    if "overflow" in key or "dropped" in key:
                        self.assertEqual(value, 0, key)

    def test_sliding_direction(self):
        """Oppose sliding in every direction for analytic and mesh boxes, with and without patches."""
        devices = ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",)
        for device in devices:
            for beta in (0.0, 0.2):
                for mesh in (False, True):
                    for angle in (0.0, np.pi / 6, np.pi / 4, 2 * np.pi / 3):
                        with self.subTest(device=device, beta=beta, mesh=mesh, angle=angle):
                            velocity, direction, watermarks = _slide(device, beta, mesh, angle)
                            expected = direction * (0.2 * 0.3 * 9.81) * 0.5
                            np.testing.assert_allclose(velocity, expected, atol=0.015, rtol=0.03)
                            for key, value in watermarks.items():
                                if "overflow" in key or "dropped" in key:
                                    self.assertEqual(value, 0, key)


if __name__ == "__main__":
    unittest.main()

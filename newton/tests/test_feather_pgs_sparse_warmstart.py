# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native sparse sweeps carry identity-matched contact impulses across steps."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

GRAVITY = 9.81


def _build(device, worlds=4, drop=0.01):
    """Floating torsos on two position-held legs whose sphere feet rest on the ground."""
    builder = newton.ModelBuilder()
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    shape_cfg.mu = 0.8
    masses = []
    for _world in range(worlds):
        builder.begin_world()
        torso = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.48 + drop), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3) * 0.1),
            lock_inertia=True,
        )
        joints = [builder.add_joint_free(child=torso)]
        mass = 2.0
        for side in (-1.0, 1.0):
            parent = torso
            for link in range(2):
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05), lock_inertia=True)
                mass += 1.0
                offset = wp.vec3(0.0, 0.15 * side, 0.0) if link == 0 else wp.vec3(0.0, 0.0, -0.25)
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=body,
                        axis=newton.Axis.Y if link else newton.Axis.X,
                        parent_xform=wp.transform(offset, wp.quat_identity()),
                        target_ke=400.0,
                        target_kd=20.0,
                    )
                )
                parent = body
            # Two feet per shin give a support polygon around the center of mass.
            for toe in (-0.1, 0.1):
                builder.add_shape_sphere(
                    parent,
                    radius=0.03,
                    xform=wp.transform(wp.vec3(toe, 0.0, -0.2), wp.quat_identity()),
                    cfg=shape_cfg,
                )
        builder.add_articulation(joints)
        builder.end_world()
        masses.append(mass)
    builder.add_ground_plane()
    return builder.finalize(device=device), masses


@unittest.skipUnless(wp.is_cuda_available(), "Native sparse factors are a CUDA path")
class TestFeatherPGSSparseWarmstart(unittest.TestCase):
    def _rollout(self, iterations, *, warmstart, steps=160, dt=1.0 / 240.0):
        device = wp.get_device("cuda:0")
        model, masses = _build(device)
        solver = SolverFeatherPGS(
            model,
            pgs_iterations=iterations,
            pgs_warmstart=warmstart,
            friction_anchor_beta=0.0,
        )
        self.assertIsNotNone(solver.placement.sparse_mass_matrix_size, solver.routes.row.reasons["K1"])
        self.assertEqual("_warm_start" in solver._pgs_solve_sparse_kernel.key, warmstart)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        states = [model.state() for _ in range(2)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        impulse_sums = np.zeros((steps, model.world_count))
        for i in range(steps):
            pipeline.collide(states[0], contacts)
            states[0].clear_forces()
            solver.step(states[0], states[1], control, contacts, dt)
            states.reverse()
            counts = solver.placement.constraint_count.numpy()
            rows = solver.placement.row_type.numpy()
            lam = solver.placement.impulses.numpy()
            for world in range(model.world_count):
                active = rows[world, : counts[world]] == PGS_CONSTRAINT_TYPE_CONTACT
                impulse_sums[i, world] = lam[world, : counts[world]][active].sum()
        wp.synchronize_device(device)
        body_q = states[0].body_q.numpy()
        body_qd = states[0].body_qd.numpy()
        self.assertTrue(np.isfinite(body_q).all() and np.isfinite(body_qd).all())
        torso = body_q.reshape(model.world_count, -1, 7)[:, 0, 2]
        return impulse_sums, torso, body_qd, np.asarray(masses) * GRAVITY * dt

    def test_warm_start_is_admitted_and_physical(self):
        """A warm-started sparse stance carries the weight impulse without ramping."""
        for warmstart in (False, True):
            with self.subTest(warmstart=warmstart):
                impulses, torso, body_qd, weight = self._rollout(8, warmstart=warmstart)
                # Settled stance: per-step normal impulse equals the weight impulse.
                settled = impulses[-40:]
                np.testing.assert_allclose(settled.mean(axis=0), weight, rtol=0.05)
                self.assertLess(float(np.abs(settled - settled.mean(axis=0)).max() / weight.max()), 0.05)
                np.testing.assert_allclose(torso, 0.48, atol=0.01)
                self.assertLess(float(np.abs(body_qd).max()), 0.05)

    def test_warm_start_improves_a_small_iteration_budget(self):
        """Carried impulses leave less residual stance velocity than a cold two-sweep solve."""
        _, _, cold_qd, _ = self._rollout(2, warmstart=False)
        _, _, warm_qd, _ = self._rollout(2, warmstart=True)
        _, _, reference_qd, _ = self._rollout(32, warmstart=False)
        cold_error = float(np.abs(cold_qd - reference_qd).max())
        warm_error = float(np.abs(warm_qd - reference_qd).max())
        self.assertLess(warm_error, cold_error)

    def test_early_exit_with_warm_start(self):
        """A warm-started sweep that the exit law stops early stays close to a long solve."""
        _, torso_long, qd_long, _ = self._rollout(64, warmstart=True)
        _, torso_short, qd_short, _ = self._rollout(8, warmstart=True)
        np.testing.assert_allclose(torso_short, torso_long, atol=2.0e-3)
        self.assertLess(float(np.abs(qd_short - qd_long).max()), 2.0e-2)


if __name__ == "__main__":
    unittest.main()

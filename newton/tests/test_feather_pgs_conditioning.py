# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep the mass matrices of deep articulations within float32 factorization.

A 100-link chain without armature has pivots ratios down to 3.5e-7 of their diagonal, below what a float32
factorization resolves, and its factor turns non-finite within a few steps. The solver adds the least armature
that lifts every pivot to ``4 n u`` of its diagonal, only where a pivot falls short.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import conditioning
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes


def _solver(scene, device="cpu"):
    model = scene.build(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverFeatherPGS(model, pgs_mode="split", dense_max_constraints=384)
    added = [str(warning.message) for warning in caught if "armature" in str(warning.message)]
    return model, solver, added


class TestFeatherPGSConditioning(unittest.TestCase):
    def test_pivot_floor_lifts_only_short_pivots(self):
        """Lift each pivot below the floor exactly onto it and leave well-conditioned matrices untouched."""
        rng = np.random.default_rng(3)
        factor = rng.standard_normal((2, 12, 12))
        mass = factor @ factor.transpose(0, 2, 1) + 12.0 * np.eye(12)
        np.testing.assert_array_equal(conditioning.pivot_floor(mass), 0.0)
        # A light coordinate nearly redundant with a heavy one: its pivot is 1e-9 of its diagonal. The factor is
        # leaf-first (the coordinates reversed), so coordinate 1 is eliminated first and coordinate 0 is lifted.
        mass = np.array([[[1.0, 1.0 - 1.0e-9], [1.0 - 1.0e-9, 1.0]]])
        added = conditioning.pivot_floor(mass)
        ratio = conditioning.PIVOT_FLOOR_FACTOR * 2 * conditioning.FLOAT32_ROUNDING
        self.assertEqual(added[0, 1], 0.0)
        lifted = mass[0] + np.diag(added[0])
        pivots = np.diag(np.linalg.cholesky(lifted[::-1, ::-1])) ** 2
        np.testing.assert_allclose(pivots[1] / lifted[0, 0], ratio, rtol=1.0e-6)

    def test_depths_follow_the_longest_path(self):
        """Count the joints on the longest root-to-leaf path of each articulation."""

        def depths(scene):
            model = scene.build("cpu")
            return conditioning.articulation_depths(model, TreePlan.build(model.joint_ancestor.numpy()))

        np.testing.assert_array_equal(depths(scenes.long_chain()), [101])
        self.assertLess(int(depths(scenes.sprawl()).max()), 32)

    def test_bare_chain_gains_armature_near_its_head(self):
        """Add armature to the root's yaw and the joints near a bare chain's head, record it and warn once.

        The factor is leaf-first, so the free root's pivots are its articulated inertia: its yaw about the chain's
        plane normal is the least of them, and the coordinates near the head follow."""
        model, solver, added = _solver(scenes.long_chain(armature=0.0))
        floor = solver.get_path_selection()["routes"]["dynamics"]["armature_floor"]
        self.assertEqual((floor["evaluated_articulations"], floor["articulations"]), (1, 1))
        extra = solver._joint_armature_effective - model.joint_armature.numpy()
        self.assertEqual(floor["coordinates"], int(np.count_nonzero(extra)))
        self.assertAlmostEqual(floor["largest"], float(extra.max()), places=6)
        # The free root's translations and tilts keep their armature; its yaw gains the most.
        np.testing.assert_array_equal(extra[:5], 0.0)
        self.assertEqual(int(np.argmax(extra)), 5)
        self.assertLess(float(extra.max()), 1.0e-3)
        self.assertEqual(len(added), 1)

    def test_conditioned_and_shallow_models_keep_their_armature(self):
        """Keep the armature bit for bit where every pivot clears the floor or no path is deep."""
        for scene in (scenes.sprawl(), scenes.legged()):
            with self.subTest(scene=scene.name):
                model, solver, added = _solver(scene)
                floor = solver.get_path_selection()["routes"]["dynamics"]["armature_floor"]
                self.assertEqual(floor["evaluated_articulations"], 0)
                self.assertEqual(floor["coordinates"], 0)
                np.testing.assert_array_equal(solver._joint_armature_effective, model.joint_armature.numpy())
                self.assertEqual(added, [])

    def test_geared_chain_lifts_only_its_root_yaw(self):
        """A geared chain's joints clear the floor; only its free root's yaw, whose leaf-first pivot is the chain's
        articulated inertia, is lifted, by a few millionths of its composite inertia's scale."""
        model, solver, _ = _solver(scenes.long_chain())
        floor = solver.get_path_selection()["routes"]["dynamics"]["armature_floor"]
        self.assertEqual(floor["evaluated_articulations"], 1)
        extra = solver._joint_armature_effective - model.joint_armature.numpy()
        self.assertEqual(np.flatnonzero(extra).tolist(), [5])
        self.assertLess(float(extra[5]), 1.0e-5)

    def test_bare_chain_stays_finite(self):
        """Step a bare 100-link chain on the ground without a non-finite factor or velocity."""
        scene = scenes.long_chain(armature=0.0)
        model, solver, _ = _solver(scene)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=scene.rigid_contact_max)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        for _ in range(scene.steps):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, scene.dt)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.isfinite(state_0.joint_qd.numpy()).all())
        size = solver.size_groups[0]
        self.assertTrue(np.isfinite(solver.L_by_size[size].numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_cuda_adds_the_cpu_armature(self):
        """Add the armature on CUDA that the CPU adds, up to the float32 rounding of the mass matrices.

        A pivot near the floor is the difference of two large terms, so a rounding of the mass matrix moves the
        armature that lifts it. The leaf-first floor lifts about 37 coordinates by nearly equal amounts, so a
        rounding trades up to a tenth of the largest addition between neighbours; the total stays within 2%.
        """
        model, cpu, _ = _solver(scenes.long_chain(armature=0.0), "cpu")
        _, cuda, _ = _solver(scenes.long_chain(armature=0.0), "cuda:0")
        extra = cpu._joint_armature_effective - model.joint_armature.numpy()
        cuda_extra = cuda._joint_armature_effective - model.joint_armature.numpy()
        np.testing.assert_allclose(cuda_extra, extra, rtol=0.0, atol=0.1 * float(extra.max()))
        self.assertAlmostEqual(float(cuda_extra.sum()), float(extra.sum()), delta=0.02 * float(extra.sum()))


if __name__ == "__main__":
    unittest.main()

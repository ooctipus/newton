# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep the mass matrices of deep articulations within float32 factorization.

A 100-link chain without armature has pivots ratios down to 3.5e-7 of their diagonal, below what a float32
factorization resolves, and its factor turns non-finite within a few steps. The solver adds the least armature
that lifts every pivot to ``4 n u`` of its diagonal, only where a pivot falls short.
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import conditioning
from newton._src.solvers.feather_pgs.forest_plan import joint_forest
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes
from newton.tests.test_feather_pgs_connect import _LOOP_FRAMES, _four_bar_bars, _loop_anchor_gap


def _solver(scene, device="cpu"):
    model = scene.build(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverFeatherPGS(model, dense_max_constraints=384)
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
        extra = solver.factor.armature - model.joint_armature.numpy()
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
                np.testing.assert_array_equal(solver.factor.armature, model.joint_armature.numpy())
                self.assertEqual(added, [])

    def test_geared_chain_lifts_only_its_root_yaw(self):
        """A geared chain's joints clear the floor; only its free root's yaw, whose leaf-first pivot is the chain's
        articulated inertia, is lifted, by a few millionths of its composite inertia's scale."""
        model, solver, _ = _solver(scenes.long_chain())
        floor = solver.get_path_selection()["routes"]["dynamics"]["armature_floor"]
        self.assertEqual(floor["evaluated_articulations"], 1)
        extra = solver.factor.armature - model.joint_armature.numpy()
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
        self.assertTrue(np.isfinite(solver.factor.walk_data.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_cuda_adds_the_cpu_armature(self):
        """Add the armature on CUDA that the CPU adds, up to the float32 rounding of the mass matrices.

        A pivot near the floor is the difference of two large terms, so a rounding of the mass matrix moves the
        armature that lifts it. The leaf-first floor lifts about 37 coordinates by nearly equal amounts, so a
        rounding trades up to a tenth of the largest addition between neighbours; the total stays within 2%.
        """
        model, cpu, _ = _solver(scenes.long_chain(armature=0.0), "cpu")
        _, cuda, _ = _solver(scenes.long_chain(armature=0.0), "cuda:0")
        extra = cpu.factor.armature - model.joint_armature.numpy()
        cuda_extra = cuda.factor.armature - model.joint_armature.numpy()
        np.testing.assert_allclose(cuda_extra, extra, rtol=0.0, atol=0.1 * float(extra.max()))
        self.assertAlmostEqual(float(cuda_extra.sum()), float(extra.sum()), delta=0.02 * float(extra.sum()))


class TestAssemblyFloor(unittest.TestCase):
    """A rotation's diagonal is the remainder of terms of order ``m R^2``: float32 can round a thin axis to nothing."""

    def _rocker(self, inertia: float):
        builder = _four_bar_bars()
        bars, coupler, rocker = builder
        bars.body_inertia[rocker] = wp.mat33(inertia * np.eye(3))
        bars.add_joint_ball(parent=coupler, child=rocker, **_LOOP_FRAMES)
        return bars.finalize(device="cpu")

    def test_a_thin_ball_loop_steps(self):
        """The rocker's axial inertia of 1e-9 kg m^2 is below the rounding of its shadow's ball joint's diagonal, about
        3e-9: the factor once went NaN at the first step. The floor lifts the one axis, and the loop closes."""
        model = self._rocker(1.0e-9)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = SolverFeatherPGS(model, pgs_iterations=8, pgs_beta=0.1)
        self.assertTrue([w for w in caught if "armature" in str(w.message)], "the floor added nothing")
        floor = solver.factor.armature_floor
        self.assertGreater(floor.coordinates, 0)
        self.assertLess(floor.largest, 1.0e-7)
        state_0, state_1, control = model.state(), model.state(), model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.3
        control.joint_target_q.assign(targets)
        for _ in range(120):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertLess(_loop_anchor_gap(model, state_0), 1.0e-5)
        self.assertAlmostEqual(float(state_0.joint_q.numpy()[0]), 0.2879, delta=2.0e-3)

    def test_every_inertia_around_the_rounding_steps(self):
        """Rounding lands on either side of zero around the floor, so no inertia near it may fail."""
        for inertia in (1.0e-7, 3.0e-8, 1.0e-8, 3.0e-9, 1.0e-9, 3.0e-10, 1.0e-10):
            with self.subTest(inertia=inertia):
                model = self._rocker(inertia)
                solver = SolverFeatherPGS(model, pgs_iterations=4, pgs_beta=0.1)
                state_0, state_1, control = model.state(), model.state(), model.control()
                for _ in range(8):
                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, None, 1.0 / 240.0)
                    state_0, state_1 = state_1, state_0
                self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())

    def test_a_pure_tree_takes_the_same_rule(self):
        """No loop term: a plain ball joint on a thin body far from its root takes the same armature."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.5, hy=0.05, hz=0.05)
        root = builder.add_joint_fixed(parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
        thin = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.9), wp.quat_identity()), mass=1.0)
        builder.body_inertia[thin] = wp.mat33(1.0e-9 * np.eye(3))
        ball = builder.add_joint_ball(
            parent=base,
            child=thin,
            parent_xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        )
        builder.add_articulation([root, ball])
        model = builder.finalize(device="cpu")
        added = conditioning.assembly_floor(model, model.joint_armature.numpy(), joint_forest(model))
        self.assertGreater(float(added.max()), 0.0)

    @staticmethod
    def _thin_hinges(worlds: int = 1):
        """Two thin bodies on hinges about ``x``, 2 and 2.5 m out along ``x`` from a fixed root, in ``worlds`` worlds."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        base = builder.add_link(xform=wp.transform_identity(), mass=1.0)
        builder.body_inertia[base] = wp.mat33(np.eye(3))
        joints = [builder.add_joint_fixed(parent=-1, child=base)]
        parent = base
        for offset, x, mass in ((2.0, 2.0, 1.0), (0.5, 2.5, 0.5)):
            body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), mass=mass)
            builder.body_inertia[body] = wp.mat33(1.0e-9 * np.eye(3))
            frame = wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity())
            joints.append(
                builder.add_joint_revolute(
                    parent=parent, child=body, parent_xform=frame, axis=(1.0, 0.0, 0.0), armature=0.0
                )
            )
            parent = body
        builder.add_articulation(joints)
        world = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(worlds):
            world.add_world(builder)
        return world.finalize(device="cpu")

    def test_a_hinge_reads_its_subtree_about_its_axis(self):
        """The lift is the floor less the subtree's inertia about the hinge's own axis, the rest of the tree aside."""
        model = self._thin_hinges()
        added = conditioning.assembly_floor(model, model.joint_armature.numpy(), joint_forest(model))
        terms = 1.0 * 2.0**2 + 0.5 * 2.5**2
        floor = conditioning.PIVOT_FLOOR_FACTOR * 2 * conditioning.FLOAT32_ROUNDING * terms
        np.testing.assert_allclose(added, [floor - 2.0e-9, floor - 1.0e-9], rtol=1.0e-6)

    def test_every_world_costs_the_model_one_pass(self):
        """Real assets name thin axes in every world, and a notification repeats the rule, so nothing runs per joint."""

        def reads(worlds: int) -> int:
            model = self._thin_hinges(worlds)
            forest, armature = joint_forest(model), model.joint_armature.numpy()
            with patch.object(wp.array, "numpy", autospec=True, side_effect=wp.array.numpy) as read:
                added = conditioning.assembly_floor(model, armature, forest)
            np.testing.assert_allclose(added.reshape(worlds, 2), np.broadcast_to(added[:2], (worlds, 2)), rtol=1.0e-6)
            return read.call_count

        self.assertEqual(reads(2), reads(16))

    def test_a_model_without_thin_axes_keeps_its_armature(self):
        """The catalogue adds nothing: the floor is the rounding of a term, far below any real body's inertia."""
        for scene in (scenes.chain(), scenes.mimic_pair(), scenes.five_bar_leg(), scenes.stewart_platform()):
            with self.subTest(scene=scene.name):
                model = scene.build("cpu")
                armature = model.joint_armature.numpy()
                added = conditioning.assembly_floor(model, armature, joint_forest(model))
                np.testing.assert_array_equal(added, 0.0)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The lane-parallel held factor solve of the tree predictor matches the serial law."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import response_operator, tree_kinetic_state
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT, _build
from newton.tests.test_feather_pgs_subtree_reduce import _assert_same_layouts, _assert_switch_within, _excess


class TestLaneCholeskyEmitter(unittest.TestCase):
    def test_emitter_is_sized_by_the_family(self):
        """Every loop bound and row stride is the family size; the lane, mask and width are the caller's."""
        text = response_operator.LaneCholeskySolve(23, "L").solve("$SIX_BODIES+0", "$WHITE+0")
        self.assertIn("for(int k=0;k<23;++k)", text)
        self.assertIn("for(int k=23-1;k>=0;--k)", text)
        self.assertIn("row[k]=Lr[ic*23+k];col[k]=Lr[k*23+ic];", text)
        self.assertIn("float row[23],col[23];", text)
        self.assertIn("__shfl_sync(MASK,x,k,WIDTH)", text)
        self.assertIn("diagonal!=0.0f?value/diagonal:0.0f", text)
        custom = response_operator.LaneCholeskySolve(6, "L").solve("a", "b", lane="l", mask="m", width="16")
        self.assertIn("__shfl_sync(m,z,k,16)", custom)
        self.assertIn("const int i=l;", custom)


def _rollout(model, *, serial, steps=90, cached=True):
    pins = {"storage": "dense", "claims": ()} if cached else {"evaluator": "ordinary", "claims": ()}
    with (
        patch.dict(os.environ, {"FEATHER_PGS_TREE_REFERENCE": str(int(serial))}),
        patch.object(SolverFeatherPGS, "_route_pins", pins),
    ):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=2,
            friction_anchor_beta=0.0,
            cached_dynamics="auto" if cached else "off",
        )
    owner = getattr(solver, "_tree_kinetic_state", None)
    # Sorted contacts: the narrow phase's output order no longer depends on thread scheduling.
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_target_q.assign(np.zeros_like(model.joint_target_q.numpy()))
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    types, qd_start = model.joint_type.numpy(), model.joint_qd_start.numpy()
    for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[::3]:
        qd[qd_start[joint]] = 1.5
    states[0].joint_qd.assign(qd)
    poses, predictors = [], []
    for _ in range(steps):
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        poses.append(states[0].body_q.numpy().copy())
        predictors.append(solver.v_hat.numpy().copy())
    wp.synchronize_device(model.device)
    return owner, np.array(poses), np.array(predictors)


# Steps over which the lane-parallel law is compared with the serial law and the ordinary solver.
_EARLY_STEPS = 8


def _predictor(group, reference):
    """The predictor kernel the factory builds for a group's layout under ``reference``."""
    schedules = tree_kinetic_state.held_schedules(
        group.tree, group.layout.lanes, chains=group.chains, reference=reference
    )
    return tree_kinetic_state.get_predictor_kernel(
        schedules, layout=group.layout, response=group._response, reference=reference, clocked=group.clocked
    )


def _lane_parallel(group):
    """Whether a group holds a factor family its lanes solve in parallel (one lane per row)."""
    return group._response == "cholesky" and any(
        0 < family <= min(32, group.layout.lanes) for family in group.layout.root_families
    )


@unittest.skipUnless(wp.is_cuda_available(), "The held-tree owner is a CUDA path")
class TestLaneParallelHeldSolve(unittest.TestCase):
    def _compare(self, model, families):
        owner_serial, plain, plain_predictor = _rollout(model, serial=True)
        owner, poses, predictor = _rollout(model, serial=False)
        # A/A control: the default arm once more, for its run-to-run spread.
        _, poses_again, predictor_again = _rollout(model, serial=False)
        self.assertIsNotNone(owner)
        self.assertEqual(sorted(group.layout.root_families for group in owner.groups), sorted(families))
        self.assertIs(type(owner), type(owner_serial))
        _assert_same_layouts(self, owner_serial, owner)
        # Each arm ran the predictor its own switch selects: the switch is a factory
        # argument, so the second arm cannot reuse the first arm's kernels, and every
        # lane-parallel group differs from the serial law built for its own layout.
        for arm, reference in ((owner_serial, True), (owner, False)):
            self.assertEqual(arm.reference, reference)
            for group in arm.groups:
                self.assertIs(group.predictor_kernel, _predictor(group, reference))
        parallel = [group for group in owner.groups if _lane_parallel(group)]
        self.assertTrue(parallel)
        for group in parallel:
            self.assertIsNot(group.predictor_kernel, _predictor(group, True))
        self.assertTrue(np.isfinite(poses).all())
        # The two laws differ in float order alone: the forward pass is the same arithmetic in
        # the same order, the transposed pass accumulates in the other direction. For a long
        # family the resulting difference is set by the factor's conditioning, not by a fixed
        # tolerance, so it is bounded by how far the ordinary (uncached) solver, a third fp32
        # evaluation of the same predictor, lies from the serial law. Rollouts diverge after a
        # few steps, so the bound is taken over the early steps; later steps must stay finite.
        _, ordinary, ordinary_predictor = _rollout(model, serial=True, steps=_EARLY_STEPS, cached=False)
        early = slice(0, _EARLY_STEPS)
        reference = _excess(ordinary_predictor, plain_predictor[early], 1e-5)
        switch = _excess(predictor[early], plain_predictor[early], 1e-5)
        again = _excess(predictor_again[early], predictor[early], 1e-5)
        _assert_switch_within(self, "predictor", switch, again, max(1e-5, 2.0 * reference))
        for label, part in (("position [m]", slice(0, 3)), ("orientation", slice(3, None))):
            reference = _excess(ordinary[:, :, part], plain[early, :, part])
            switch = _excess(poses[early, :, part], plain[early, :, part])
            again = _excess(poses_again[early, :, part], poses[early, :, part])
            _assert_switch_within(self, label, switch, again, max(1e-5, 2.0 * reference))

    def test_forest_with_prescribed_and_free_roots(self):
        """A three-joint arm, a free box and a kinematic tray: families 3, 6 and a prescribed root."""
        self._compare(_build(wp.get_device("cuda:0"), worlds=6, prescribed=True), [(3, 6, 0)])

    def test_long_arm_family_fills_most_of_the_warp(self):
        """A seventeen-joint arm and the box in per-size groups: 17 and 6 coordinate families."""
        self._compare(_build(wp.get_device("cuda:0"), worlds=4, links=17), [(6,), (17,)])

    def test_per_size_groups_without_a_prescribed_root(self):
        """Without a prescribed root the owner keeps per-size groups; each family solves on its own lanes."""
        self._compare(_build(wp.get_device("cuda:0"), worlds=6), [(3,), (6,)])


if __name__ == "__main__":
    unittest.main()

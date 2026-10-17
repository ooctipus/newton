# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cached tree dynamics treat kinematic bodies as ordinary dynamics does."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.mixed_tree_kinetic import MixedTreeKineticState
from newton._src.solvers.feather_pgs.tree_kinetic_owner import TreeKineticState
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_sparse_split as sparse_split

DT = 1.0 / 240.0
KINEMATIC = int(newton.BodyFlags.KINEMATIC)
# Point friction and Newton's default patch friction admit the owner through different contact paths.
FRICTION_LAWS = {"point": 0.0, "patch": 0.2}
STATE_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")


def _set_flag(model, body, flag):
    flags = model.body_flags.numpy()
    flags[body] = flag
    model.body_flags.assign(flags)


def _solver(model, beta, cached, **options):
    """A default solver under one friction law; ``cached`` is the ``cached_dynamics`` choice."""
    return SolverFeatherPGS(model, angular_damping=0.0, friction_anchor_beta=beta, cached_dynamics=cached, **options)


def _chain(*, floating):
    """A moving three-link revolute chain on a fixed or floating base, in zero gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    links, joints = [], []
    for i in range(3):
        link = builder.add_link(xform=wp.transform(wp.vec3(0.3 * i, 0.0, 1.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(link, hx=0.1, hy=0.05, hz=0.05)
        if not links:
            base = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
            joint = builder.add_joint_free(link) if floating else builder.add_joint_fixed(-1, link, parent_xform=base)
        else:
            offset = wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity())
            joint = builder.add_joint_revolute(links[-1], link, axis=newton.Axis.Y, parent_xform=offset)
        links.append(link)
        joints.append(joint)
    builder.add_articulation(joints)
    builder.joint_qd[:] = [0.3] * len(builder.joint_qd)
    return builder.finalize(device="cuda:0"), links


def _sparse_case(beta, cached, *, kinematic_body=None):
    """The scene of test_feather_pgs_sparse_split on its native sparse matrix-free route."""
    model = sparse_split._model()
    if kinematic_body is not None:
        _set_flag(model, kinematic_body, KINEMATIC)
    solver = sparse_split._solver(
        model, True, pgs_mode="matrix_free", friction_anchor_beta=beta, cached_dynamics=cached
    )
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control.joint_f.assign(np.linspace(-0.08, 0.11, model.joint_dof_count, dtype=np.float32))
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
    return model, solver, state, out, control, pipeline, pipeline.contacts()


def _snapshot(state):
    return {name: getattr(state, name).numpy().copy() for name in STATE_FIELDS}


class _Rollout:
    """Step one solver under a constant wrench on one body."""

    def __init__(self, model, solver, body, wrench):
        self.model, self.solver = model, solver
        self.pipeline = newton.CollisionPipeline(model)
        self.contacts = self.pipeline.contacts()
        self.control = model.control()
        self.state, self.next_state = model.state(), model.state()
        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)
        # Both banks carry the wrench, so the copy that closes each step keeps it.
        wrenches = np.zeros((model.body_count, 6), np.float32)
        wrenches[body] = wrench
        self.state.body_f.assign(wrenches)
        self.next_state.body_f.assign(wrenches)

    def step(self):
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, DT)
        self.state.assign(self.next_state)

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return _snapshot(self.state)


@unittest.skipUnless(wp.is_cuda_available(), "Cached tree owners are a CUDA path")
class TestFeatherPGSTreeBodyFlags(unittest.TestCase):
    def _assert_states_close(self, actual, expected, atol=1.0e-5, what=""):
        for name in STATE_FIELDS:
            np.testing.assert_allclose(actual[name], expected[name], atol=atol, rtol=0.0, err_msg=f"{what} {name}")

    def test_wrench_on_a_kinematic_link_stays_off_its_ancestors(self):
        """An external wrench on a kinematic link moves nothing under the owner, as under ordinary dynamics.

        Model body flags are live data: the leaf of a moving chain is made
        kinematic after finalize, before the solver is built (the builder
        only marks roots kinematic). The held tree's predictor reduced the
        leaf's wrench into its ancestors' joints; ordinary dynamics drops it.
        """
        wrench = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0], np.float32)
        for floating in (False, True):
            for law, beta in FRICTION_LAWS.items():
                with self.subTest(base="floating" if floating else "fixed", law=law):
                    runs = {}
                    for cached, load in (("off", wrench), ("auto", wrench), ("auto", 0.0 * wrench)):
                        model, links = _chain(floating=floating)
                        _set_flag(model, links[2], KINEMATIC)
                        solver = _solver(model, beta, cached, pgs_mode="matrix_free")
                        if cached == "auto":
                            self.assertIsInstance(solver._tree_kinetic_state, MixedTreeKineticState)
                        runs[cached, bool(load.any())] = _Rollout(model, solver, links[2], load).run(40)
                    moving = runs["off", True]
                    self.assertGreater(float(np.abs(moving["joint_qd"]).max()), 0.1, "the chain keeps moving")
                    self._assert_states_close(runs["auto", True], moving, what="owner against ordinary")
                    # The kinematic leaf's wrench changes nothing at all.
                    self._assert_states_close(runs["auto", True], runs["auto", False], atol=1.0e-6, what="wrench")

    def test_wrench_on_a_kinematic_link_stays_off_its_ancestors_on_sparse_factors(self):
        """The sparse tree owner (chain-scan predictor) drops a kinematic leaf's wrench as ordinary dynamics does."""
        for law, beta in FRICTION_LAWS.items():
            with self.subTest(law=law):
                runs = {}
                for cached in ("off", "auto"):
                    # Body 1 is the contacting leaf of the first world's tree; every body carries a wrench.
                    case = _sparse_case(beta, cached, kinematic_body=1)
                    solver = case[1]
                    self.assertIsNotNone(solver._sparse_mass_matrix_size)
                    if cached == "auto":
                        self.assertIsInstance(solver._tree_kinetic_state, TreeKineticState)
                    for _ in range(40):
                        sparse_split._step(case)
                    runs[cached] = _snapshot(case[2])
                self.assertTrue(np.isfinite(runs["auto"]["joint_qd"]).all())
                self._assert_states_close(runs["auto"], runs["off"], what="owner against ordinary")


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose native mimic rows with cached dynamics without changing their law."""

import unittest
import warnings
from contextlib import contextmanager, nullcontext
from itertools import pairwise
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests import feather_pgs_sparse_scene as split
from tools.fpgs_bench import test_tree_prescaled_state as prescaled
from tools.fpgs_bench.test_tree_locked_root import locked_roots


@contextmanager
def mimic_trees(*, prismatic=False):
    """Couple two scalar branches before replication and model construction."""
    original_articulation = newton.ModelBuilder.add_articulation
    original_revolute = newton.ModelBuilder.add_joint_revolute

    def articulation(builder, joints, **kwargs):
        result = original_articulation(builder, joints, **kwargs)
        scalar = [j for j in joints if builder.joint_type[j] in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)]
        if len(scalar) >= 3:
            builder.add_constraint_mimic(joint0=scalar[2], joint1=scalar[0], coef0=0.02, coef1=-0.5)
        return result

    def revolute(builder, parent, child, **kwargs):
        if prismatic and len(builder.joint_type) in (1, 3):
            return builder.add_joint_prismatic(parent, child, **kwargs)
        return original_revolute(builder, parent, child, **kwargs)

    with (
        patch.object(newton.ModelBuilder, "add_articulation", articulation),
        patch.object(newton.ModelBuilder, "add_joint_revolute", revolute),
    ):
        yield


def mimic_model(*, floating=False, prismatic=False, locked=False, device="cuda:0"):
    """Create the loaded branched fixture with one scalar mimic per world."""
    with mimic_trees(prismatic=prismatic), locked_roots() if locked else nullcontext():
        model = split.branched_model(floating=floating, branches=2, depth=2, device=device)
    assert model.constraint_mimic_count == model.world_count == 2
    return model


def solver_for(model, cached, interval=2):
    """Change only cached dynamics while retaining native mimic and velocity rows."""
    return split.make_solver(
        model,
        True,
        pins=None if cached else {"evaluator": "ordinary"},
        enable_joint_velocity_limits=True,
        update_mass_matrix_interval=interval,
    )


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestTreeMimicCache(unittest.TestCase):
    def test_loaded_steps_and_masked_resets(self):
        """Preserve loaded scalar couplings across held factors and notifications."""
        with mimic_trees():
            prescaled.TestPrescaledTreeContract.test_loaded_steps_and_masked_reset(self)

    def test_graph_live_parameters_and_reset(self):
        """Replay live coupling coefficients and enable flags through cached motion."""
        for floating, prismatic, locked in ((False, False, False), (True, True, False), (False, True, True)):
            with self.subTest(floating=floating, prismatic=prismatic, locked=locked):
                model = mimic_model(floating=floating, prismatic=prismatic, locked=locked)
                model.joint_velocity_limit.fill_(0.2)
                model.constraint_mimic_coef1.fill_(0.0)
                model.constraint_mimic_enabled.fill_(False)
                cases, graphs = [], []
                for enabled in (False, True):
                    solver = solver_for(model, enabled)
                    self.assertEqual(solver._tree_kinetic_state is not None, enabled)
                    self.assertTrue(solver._sparse_velocity_limits)
                    self.assertEqual(solver._mimic_count, 2)
                    first, second, control = model.state(), model.state(), model.control()
                    newton.eval_fk(model, first.joint_q, first.joint_qd, first)
                    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
                    cases.append((solver, first, second, control, pipeline, pipeline.contacts()))

                def two_steps(case):
                    solver, first, second, control, pipeline, contacts = case
                    model = solver.model
                    for state, output in ((first, second), (second, first)):
                        state.clear_forces()
                        wp.launch(
                            split._external_forces, dim=model.body_count, inputs=[state.body_f], device=model.device
                        )
                        pipeline.collide(state, contacts)
                        solver.step(state, output, control, contacts, split.DT)

                for case in cases:
                    two_steps(case)
                    with wp.ScopedCapture(device=model.device) as capture:
                        two_steps(case)
                    graphs.append(capture.graph)
                mask = wp.array([True, False], dtype=wp.bool, device=model.device)
                contact_seen = False
                for replay in range(12):
                    if replay in (2, 5):
                        model.constraint_mimic_enabled.fill_(True)
                        model.constraint_mimic_coef1.fill_(-0.5 if replay == 2 else 1.25)
                        model.constraint_mimic_coef0.fill_(0.02 if replay == 2 else -0.01)
                    elif replay == 4:
                        model.constraint_mimic_enabled.fill_(False)
                    if replay == 3:
                        model.joint_velocity_limit.fill_(0.12)
                        for solver, state, *_ in cases:
                            q = state.joint_q.numpy()
                            q[7 if floating else 0] += 0.025
                            state.joint_q.assign(q)
                            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                            solver.reset(state, world_mask=mask)
                    for graph, case in zip(graphs, cases, strict=True):
                        wp.capture_launch(graph)
                        case[0].check_constraint_capacity()
                        contact_seen |= bool(case[-1].rigid_contact_count.numpy()[0])
                        np.testing.assert_array_equal(
                            case[0].mimic_slot.numpy() >= 0, model.constraint_mimic_enabled.numpy()
                        )
                    for name in split.STATE_FIELDS:
                        np.testing.assert_allclose(
                            getattr(cases[1][1], name).numpy(),
                            getattr(cases[0][1], name).numpy(),
                            atol=4e-4,
                            rtol=4e-4,
                            err_msg=name,
                        )
                    np.testing.assert_array_equal(cases[1][0]._tree_kinetic_state.groups[0].status.numpy(), 0)
                self.assertTrue(contact_seen)

    def test_mimic_closure_converges(self):
        """Track a driven leader while converging an offset negative coupling."""
        scene = newton.ModelBuilder(gravity=0.0)
        bodies = [scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(3)]
        joints = [scene.add_joint_fixed(-1, bodies[0])]
        for parent, child in pairwise(bodies):
            joints.append(
                scene.add_joint_revolute(
                    parent,
                    child,
                    axis=newton.Axis.Z,
                    target_ke=50.0 if len(joints) == 1 else 0.0,
                    target_kd=5.0 if len(joints) == 1 else 0.0,
                    armature=0.05,
                )
            )
        scene.add_articulation(joints)
        scene.add_constraint_mimic(joint0=joints[2], joint1=joints[1], coef0=0.1, coef1=-0.5)
        builder = newton.ModelBuilder(gravity=0.0)
        builder.replicate(scene, 2)
        model = builder.finalize(device="cuda:0")
        model.joint_velocity_limit.fill_(20.0)
        states = []
        for enabled in (False, True):
            solver = solver_for(model, enabled, interval=3)
            self.assertEqual(solver._tree_kinetic_state is not None, enabled)
            first, second, control = model.state(), model.state(), model.control()
            target = control.joint_target_q.numpy()
            target[::2] = 0.4
            control.joint_target_q.assign(target)
            newton.eval_fk(model, first.joint_q, first.joint_qd, first)
            for _ in range(600):
                first.clear_forces()
                solver.step(first, second, control, None, 1.0 / 240.0)
                first, second = second, first
            solver.check_constraint_capacity()
            states.append(first.joint_q.numpy().reshape(2, 2))
        for state in states:
            np.testing.assert_allclose(state[:, 0], 0.4, atol=0.05)
            np.testing.assert_allclose(state[:, 1] + 0.5 * state[:, 0], 0.1, atol=0.002)
        np.testing.assert_allclose(states[1], states[0], atol=2e-4, rtol=2e-4)

    def test_ignored_mimic_keeps_cache_disabled(self):
        """Keep invalid scalar couplings outside complete cached ownership."""
        model = mimic_model()
        model.constraint_mimic_joint1.assign(model.constraint_mimic_joint0)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            solver = solver_for(model, True)
        self.assertTrue(any("couples a DOF to itself" in str(m.message) for m in messages))
        self.assertIsNone(solver._tree_kinetic_state)
        self.assertFalse(solver._fk_id_cache_enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)

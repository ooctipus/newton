# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the predictor-resolved world classifier."""

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import resolved_worlds as rw
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton.tests.test_feather_pgs_mixed_tree_free_body import DT, _add_quadruped, _build


def make_case(*, free=False):
    """Build one responsive body and one prescribed endpoint on CPU."""
    device = "cpu"
    width = 6 if free else 1

    def a(value, dtype=wp.float32):
        return wp.array(value, dtype=dtype, device=device)

    data = rw.ResolvedWorldInput()
    data.world_dof_indices = a([list(range(width))], wp.int32)
    data.world_dof_count = a([width], wp.int32)
    data.limit_q_index = a([-1] * width if free else [0], wp.int32)
    data.lower, data.upper = a([-np.inf] * width), a([np.inf] * width)
    data.q, data.v_hat = a([0.0]), a([0.0] * width)
    data.body_to_articulation = a([0, 1], wp.int32)
    data.art_to_world = a([0, 0], wp.int32)
    data.free_bodies = wp.array([0] if free else [], dtype=wp.int32, device=device)
    data.articulation_dof_start = a([0, width], wp.int32)
    data.articulation_response_dof_count = a([width, 0], wp.int32)
    data.prescribed_articulation = a([0, 1], wp.int32)
    data.is_free_rigid = a([int(free), 0], wp.int32)
    data.body_flags = a([1, 2], wp.int32)
    data.body_has_response_dofs = a([1, 0], wp.int32)
    data.body_response_dof_mask = a([(1 << width) - 1, 0], wp.uint32)
    data.body_q = a([[0, 0, 0, 0, 0, 0, 1]] * 2, wp.transform)
    data.body_v_s = a([[999, 999, 999, 0, 0, 0], [0, 0, 0, 0, 0, 0]], wp.spatial_vector)
    data.joint_S_s = a(np.eye(6, dtype=np.float32)[:width], wp.spatial_vector)
    data.articulation_origin = a([[0, 0, 0], [0, 0, 0]], wp.vec3)
    data.max_linear_velocity = a([np.inf, np.inf])
    data.max_angular_velocity = a([np.inf, np.inf])
    data.max_depenetration_velocity = a([np.inf, np.inf])
    data.dt, data.beta, data.speculative_scale = 0.1, 0.2, 1.0
    data.activation_gap, data.restitution_threshold = 0.0, 0.5
    data.shared_anchor = 0
    data.absolute_margin, data.relative_margin = 0.0, 0.0
    raw = rw.RawContacts()
    raw.count = a([1], wp.int32)
    raw.shape0, raw.shape1 = a([0], wp.int32), a([1], wp.int32)
    raw.point0, raw.point1 = a([[0.1, 0, 0]], wp.vec3), a([[0, 0, 0]], wp.vec3)
    raw.normal = a([[-1, 0, 0]], wp.vec3)
    raw.margin0, raw.margin1 = a([0]), a([0])
    raw.shape_body = a([0, 1], wp.int32)
    raw.shape_mu, raw.shape_restitution = a([0.5, 0.5]), a([0, 0])
    scratch = rw.ResolvedWorldScratch()
    for name in ("rejected", "resolved", "checked_limits", "checked_normals", "global_invalid"):
        setattr(scratch, name, a([0], wp.int32))
    scratch.minimum_residual = a([0])
    return data, raw, scratch


def evaluate(data, raw, scratch, *, enabled=1):
    """Execute the same decision launches on CPU, without initializing CUDA."""
    rw.launch(data, raw, scratch, 1, "cpu", enabled=enabled)
    return int(scratch.resolved.numpy()[0])


class TestResolvedWorlds(unittest.TestCase):
    def test_current_predictor_not_incoming_body_twist(self):
        """Reject a closing predictor despite a separating stale body twist."""
        data, raw, scratch = make_case()
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.v_hat.assign([-2.0])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        self.assertAlmostEqual(float(scratch.minimum_residual.numpy()[0]), -1.0, places=6)
        self.assertEqual(int(scratch.checked_normals.numpy()[0]), 1)

    def test_position_limits_both_sides_and_speculative_bias(self):
        """Check activated lower and upper rows without constructing J."""
        data, raw, scratch = make_case()
        data.lower.assign([0.0])
        data.q.assign([-0.01])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.v_hat.assign([0.03])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.q.assign([0.01])
        data.activation_gap = 0.02
        data.v_hat.assign([-0.2])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.lower.assign([-np.inf])
        data.upper.assign([0.0])
        data.q.assign([0.01])
        data.v_hat.assign([-0.03])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(int(scratch.checked_limits.numpy()[0]), 1)
        # The default activation gap is infinite: every limit is a candidate, none is invalid.
        data.activation_gap = float("inf")
        data.lower.assign([-1.0])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(int(scratch.checked_limits.numpy()[0]), 2)
        self.assertEqual(int(scratch.global_invalid.numpy()[0]), 0)

    def test_many_dofs_span_the_lane_stride(self):
        """Lanes stride the world's DOF list; a violation past the first stride still rejects."""
        data, raw, scratch = make_case()
        width = 70
        device = "cpu"
        data.world_dof_indices = wp.array([list(range(width))], dtype=wp.int32, device=device)
        data.world_dof_count = wp.array([width], dtype=wp.int32, device=device)
        data.limit_q_index = wp.array(list(range(width)), dtype=wp.int32, device=device)
        data.lower = wp.array([0.0] * width, dtype=wp.float32, device=device)
        data.upper = wp.array([np.inf] * width, dtype=wp.float32, device=device)
        data.q = wp.array([1.0] * width, dtype=wp.float32, device=device)
        data.v_hat = wp.array([0.0] * width, dtype=wp.float32, device=device)
        data.articulation_response_dof_count = wp.array([width, 0], dtype=wp.int32, device=device)
        data.body_has_response_dofs = wp.array([0, 0], dtype=wp.int32, device=device)
        self.assertEqual(evaluate(data, raw, scratch), 1)
        q = np.ones(width, np.float32)
        q[69] = -0.01
        data.q.assign(q)
        self.assertEqual(evaluate(data, raw, scratch), 0)
        self.assertEqual(int(scratch.checked_limits.numpy()[0]), 1)

    def test_prescribed_endpoint_and_restitution(self):
        """Retain known endpoint motion and frozen incident rebound semantics."""
        data, raw, scratch = make_case()
        raw.point0.assign([[0, 0, 0]])
        raw.shape_restitution.assign([1.0, 1.0])
        data.body_v_s.assign([[999, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        self.assertAlmostEqual(float(scratch.minimum_residual.numpy()[0]), -2.0, places=6)
        data.body_v_s.assign([[999, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_zero_friction_radius_allows_slip(self):
        """Accept zero normal impulse without requiring zero tangential speed."""
        data, raw, scratch = make_case(free=True)
        raw.point0.assign([[0, 0, 0]])
        raw.shape_mu.assign([1000, 1000])
        data.v_hat.assign([0, 100, 0, 0, 0, 0])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_rigid_caps_and_uncapped_mf_depenetration(self):
        """Preserve finite speed caps and positive-infinity uncapped bias."""
        data, raw, scratch = make_case(free=True)
        data.v_hat.assign([0, 0, 0, 3, 0, 0])
        data.max_angular_velocity.assign([2, np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.max_angular_velocity.assign([np.inf, np.inf])
        raw.point0.assign([[-1, 0, 0]])
        data.v_hat.assign([1, 0, 0, 0, 0, 0])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.max_depenetration_velocity.assign([0.5, np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertAlmostEqual(float(scratch.minimum_residual.numpy()[0]), 0.5, places=6)

    def test_shared_anchor_and_large_world_translation(self):
        """Use shared normal anchors and stable endpoint-relative geometry."""
        data, raw, scratch = make_case()
        data.joint_S_s.assign([[0, 0, 0, 0, 0, 1]])
        data.v_hat.assign([1])
        raw.point0.assign([[0, 1, 0]])
        raw.point1.assign([[0, -1, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.shared_anchor = 1
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.body_q.assign([[1e6, 0, 0, 0, 0, 0, 1]] * 2)
        data.articulation_origin.assign([[1e6, 0, 0]] * 2)
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(float(scratch.minimum_residual.numpy()[0]), 0.0)

    def test_empty_disabled_and_transient_decisions(self):
        """Clear prior acceptance on disabled and newly violating calls."""
        data, raw, scratch = make_case()
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(evaluate(data, raw, scratch, enabled=0), 0)
        self.assertEqual(evaluate(data, raw, scratch), 1)
        raw.count.assign([0])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        self.assertEqual(int(scratch.checked_normals.numpy()[0]), 0)

    def test_malformed_prefix_ids_and_nonfinite_input(self):
        """Reject incomplete or invalid raw constraints instead of dropping them."""
        data, raw, scratch = make_case()
        raw.count.assign([2])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.count.assign([1])
        raw.shape0.assign([999])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.shape0.assign([0])
        raw.normal.assign([[0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.normal.assign([[np.nan, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        raw.normal.assign([[-1, 0, 0]])
        data.v_hat.assign([np.inf])
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.v_hat.assign([0.0])
        data.dt = -1.0
        self.assertEqual(evaluate(data, raw, scratch), 0)

    def test_rejection_margin_is_not_violation_allowance(self):
        """Send numerically ambiguous nonnegative residuals to fallback."""
        data, raw, scratch = make_case()
        raw.point0.assign([[0, 0, 0]])
        self.assertEqual(evaluate(data, raw, scratch), 1)
        data.absolute_margin = 1.0e-5
        self.assertEqual(evaluate(data, raw, scratch), 0)
        data.v_hat.assign([2.0e-5])
        self.assertEqual(evaluate(data, raw, scratch), 1)

    def test_unsupported_constructor_has_empty_mask(self):
        """Keep unsupported solvers disabled without raising or launching."""
        solver = SimpleNamespace(model=SimpleNamespace(device=wp.get_device("cpu")))
        classifier = rw.ResolvedWorlds(solver)
        self.assertFalse(classifier.enabled)
        self.assertEqual(classifier.resolved.shape, (0,))
        classifier.classify(None, None, None, 0.0)

    def test_policy_names_no_robot(self):
        """The admission reads row kinds and sizes, never a robot."""
        source = inspect.getsource(rw).lower()
        for token in ("franka", "kuka", "allegro", "size_groups", "[23, 6]"):
            self.assertNotIn(token, source)


def _rollout(model, *, classifier, steps=90):
    """Swing the arms and toss the boxes; return the solver, poses and the per-step resolved masks."""
    env = {
        "FEATHER_PGS_MIXED_TREE_KINETIC_STATE": "1",
        "FEATHER_PGS_TREE_KINETIC_STATE": "0",
        "FEATHER_PGS_RESOLVED_WORLDS": str(int(classifier)),
    }
    with patch.dict(os.environ, env):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=2,
            friction_anchor_beta=0.0,
        )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    states = [model.state() for _ in range(2)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_target_q.assign(np.zeros_like(model.joint_target_q.numpy()))
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    types, qd_start = model.joint_type.numpy(), model.joint_qd_start.numpy()
    for joint in np.flatnonzero(types == int(newton.JointType.REVOLUTE))[::3]:
        qd[qd_start[joint]] = 1.5
    # Every other box starts moving up: its ground contact separates at the predictor.
    for joint in np.flatnonzero(types == int(newton.JointType.FREE))[::2]:
        qd[qd_start[joint] + 2] = 1.0
    states[0].joint_qd.assign(qd)
    poses, masks = [], []
    for _ in range(steps):
        pipeline.collide(states[0], contacts)
        states[0].clear_forces()
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()
        poses.append(states[0].body_q.numpy().copy())
        masks.append(solver._resolved_worlds.numpy().copy())
    wp.synchronize_device(model.device)
    return solver, np.array(poses), np.array(masks)


@unittest.skipUnless(wp.is_cuda_available(), "The classifier is a CUDA path")
class TestResolvedWorldsCUDA(unittest.TestCase):
    def test_resolved_worlds_match_the_complete_path(self):
        """Fixed-base arms over free boxes: resolved worlds step identically with no rows built."""
        model = _build(wp.get_device("cuda:0"), worlds=6)
        plain_solver, plain, plain_masks = _rollout(model, classifier=False)
        self.assertIsNone(plain_solver._resolved_worlds_owner)
        self.assertEqual(plain_masks.shape[1], 0)
        solver, poses, masks = _rollout(model, classifier=True)
        self.assertIsNotNone(solver._resolved_worlds_owner)
        self.assertEqual(masks.shape, (90, 6))
        resolved_steps = int(masks.sum())
        self.assertGreater(resolved_steps, 0)
        self.assertLess(resolved_steps, masks.size)
        # The tossed boxes resolve while they rise; once a box rests its contact closes the world.
        self.assertTrue(masks[:5, ::2].all())
        self.assertFalse(masks[-30:].all(axis=0).any())
        self.assertTrue(np.isfinite(poses).all())
        np.testing.assert_allclose(poses[:, :, :3], plain[:, :, :3], atol=1.0e-5, rtol=0.0)
        np.testing.assert_allclose(poses[:, :, 3:], plain[:, :, 3:], atol=1.0e-5, rtol=0.0)


def _walkers(device, worlds=4):
    """One free-root quadruped per world on the ground: the packed sparse-factor route's class."""
    scene = newton.ModelBuilder()
    cfg = scene.default_shape_cfg.copy()
    cfg.mu = 0.6
    _add_quadruped(scene, cfg)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 2048
    return model


@unittest.skipUnless(wp.is_cuda_available(), "The sparse route is a CUDA path")
class TestResolvedWorldsSparseRoute(unittest.TestCase):
    def test_sparse_route_allocates_contacts_with_an_empty_mask(self):
        """The packed owner declines the classifier but its allocator still takes the mask argument."""
        model = _walkers(wp.get_device("cuda:0"))
        env = {
            "FEATHER_PGS_SPARSE_FACTOR": "1",
            "FEATHER_PGS_SPARSE_LEVEL_UPDATE": "1",
            "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "1",
        }
        with patch.dict(os.environ, env):
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_iterations=8,
                enable_joint_limits=True,
                grouped_dynamics=True,
                dense_max_constraints=56,
                friction_anchor_beta=0.0,
                mf_gs_incremental_rows=0,
            )
        self.assertIsNotNone(solver._sparse_factor)
        self.assertIsNone(solver._resolved_worlds_owner)
        self.assertEqual(solver._resolved_worlds.shape, (0,))
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        states = [model.state() for _ in range(2)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        for _ in range(30):
            pipeline.collide(states[0], contacts)
            states[0].clear_forces()
            solver.step(states[0], states[1], control, contacts, DT)
            states.reverse()
        wp.synchronize_device(model.device)
        poses = states[0].body_q.numpy().reshape(model.world_count, -1, 7)
        self.assertTrue(np.isfinite(poses).all())
        self.assertTrue(np.all(poses[:, 0, 2] > 0.2))
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main()

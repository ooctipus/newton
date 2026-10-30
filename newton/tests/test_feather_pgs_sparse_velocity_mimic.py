# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve bilateral mimic rows with factor-coordinate velocity limits."""

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import dense_rows, field_group, response_builds_rows, sparse_contact
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_MIMIC as MIMIC
from newton._src.solvers.feather_pgs.sparse_mass_matrix import factor_indices
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_sparse_scene as sparse_scene
from newton.tests import test_feather_pgs_sparse_velocity_limits as velocity_tests
from newton.tests.test_feather_pgs_sparse_research_admission import _eligible_solver, _TopologyReached


class TestSparseVelocityMimicAdmission(unittest.TestCase):
    def test_mimic_velocity_owner_reaches_topology(self):
        """Admit a same-articulation mimic under the explicit velocity owner."""
        solver = _eligible_solver()
        solver.options.enable_joint_velocity_limits = True
        solver.laws.mimic_count = 1
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaises(_TopologyReached),
        ):
            SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)


class TestSparseVelocityMimicRows(unittest.TestCase):
    def _check_rows(self, device):
        """Project the union of sibling supports and retain bilateral ERP for either sign."""
        plan = TreePlan.build([-1, 0, 0], [2, 2, 2]).factor
        indices = factor_indices(plan, device)
        dofs = plan.dof_count
        rng = np.random.default_rng(451)
        lower = np.zeros((dofs, dofs))
        lower[plan.entry_rows, plan.columns] = rng.normal(scale=0.06, size=plan.nonzero_count)
        lower[np.diag_indices(dofs)] = 1.8
        inverse = np.linalg.inv(lower)
        packed = np.stack([scale * inverse[plan.entry_rows, plan.columns] for scale in (1.0, 1.3)])
        velocity = np.linspace(-0.7, 0.8, 2 * dofs, dtype=np.float32)
        position = np.zeros(2 * dofs, np.float32)
        position[2], position[4] = 0.12, -0.2
        union = int(plan.ancestor_mask[2]) | int(plan.ancestor_mask[4])
        self.assertGreater(union.bit_count(), max(int(plan.ancestor_mask[i]).bit_count() for i in (2, 4)))

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        for offset, scale in ((0.03, -0.5), (-0.04, 1.7), (0.03, 0.0)):
            with self.subTest(offset=offset, scale=scale):
                kinds, parents = [full((2, 2), -9, int) for _ in range(2)]
                mu, beta, cfm, phi, target, incident, diagonal = [full((2, 2), -9.0) for _ in range(7)]
                row_dof = full((2, 2, dofs), -9, int)
                factors = full((2, 2, dofs), -9.0)
                wp.launch(
                    response_builds_rows.factor_mimic_rows_kernel,
                    dim=64,
                    block_dim=128,
                    device=device,
                    inputs=[
                        array([1, 0], int),
                        array([0, 1], int),
                        array([0, 0], int),
                        array([0, dofs], int),
                        array([0, -1], int),
                        array([0, 1, 2], int),
                        array([0, 1], int),
                        array([2, dofs + 2], int),
                        array([4, dofs + 4], int),
                        array([2, dofs + 2], int),
                        array([4, dofs + 4], int),
                        array([offset, 0.02]),
                        array([scale, -2.0]),
                        array(position),
                        0.07,
                        2e-4,
                        indices.ancestor_mask,
                        indices.inverse_permutation,
                        indices.lookup,
                        array(packed),
                        array(velocity),
                    ],
                    outputs=[
                        field_group.upload(
                            dense_rows.DenseRows,
                            device,
                            row_type=kinds,
                            row_parent=parents,
                            row_mu=mu,
                            row_beta=beta,
                            row_cfm=cfm,
                            phi=phi,
                            target=target,
                            diag=diagonal,
                        ),
                        row_dof,
                        factors,
                        incident,
                    ],
                )
                for output in (kinds, parents, mu, beta, cfm, phi, target, row_dof, factors, incident, diagonal):
                    np.testing.assert_array_equal(output.numpy()[1], -9)
                    np.testing.assert_array_equal(output.numpy()[0, 1], -9)
                columns = row_dof.numpy()[0, 0]
                valid = columns >= 0
                self.assertEqual(int(valid.sum()), union.bit_count())
                expected = 1.3 * (
                    inverse[:, plan.inverse_permutation[2]] - scale * inverse[:, plan.inverse_permutation[4]]
                )
                actual = np.zeros(dofs)
                actual[columns[valid]] = factors.numpy()[0, 0, valid]
                np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
                self.assertEqual(kinds.numpy()[0, 0], MIMIC)
                self.assertEqual(parents.numpy()[0, 0], -1)
                self.assertAlmostEqual(incident.numpy()[0, 0], velocity[2] - scale * velocity[4], places=6)
                expected_phi = float(position[2]) - offset - scale * float(position[4])
                self.assertAlmostEqual(phi.numpy()[0, 0], expected_phi, places=6)
                rows = field_group.upload(
                    dense_rows.DenseRows,
                    device,
                    row_type=kinds,
                    phi=phi,
                    row_beta=beta,
                    row_cfm=cfm,
                    target=target,
                    rhs=full((2, 2), -9.0),
                    diag=diagonal,
                )
                wp.launch(
                    sparse_contact.finalize_sparse_joint_limit_rows,
                    dim=2,
                    inputs=[array([[0, 1], [0, 0]], int), array([1, 0], int), rows, 0.0123],
                    device=device,
                )
                self.assertAlmostEqual(diagonal.numpy()[0, 0], expected @ expected + 2e-4, places=6)
                self.assertAlmostEqual(rows.rhs.numpy()[0, 0], 0.07 * expected_phi / 0.0123, places=6)
                np.testing.assert_array_equal(rows.rhs.numpy()[1], -9)
                self.assertEqual(rows.rhs.numpy()[0, 1], -9)

    def test_union_response_and_bilateral_bias_cpu(self):
        """Match nonunit, negative and zero-scale mimic rows on CPU."""
        self._check_rows("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_union_response_and_bilateral_bias_cuda(self):
        """Match the same row and CFM law across cooperative CUDA lanes."""
        self._check_rows("cuda:0")


def _model(offset, scale, error, device="cuda:0"):
    """Build independently driven sibling joints coupled by one authored mimic."""
    scene = newton.ModelBuilder()
    # The root turns on a fixed base: the held tree, which alone evaluates native sparse factors, takes fixed or free
    # roots.
    base = scene.add_link(mass=2.0, inertia=wp.mat33(np.diag([0.025, 0.03, 0.04])))
    root = scene.add_link(mass=2.0, inertia=wp.mat33(np.diag([0.025, 0.03, 0.04])))
    joints = [
        scene.add_joint_fixed(-1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.16), wp.quat_identity())),
        scene.add_joint_revolute(base, root, axis=newton.Axis.Z, armature=0.05),
    ]
    for branch in range(2):
        body = scene.add_link(mass=0.5, inertia=wp.mat33(np.diag([0.002, 0.003, 0.004])))
        joints.append(
            scene.add_joint_revolute(
                root,
                body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.1, 0.12 * branch, -0.11), wp.quat_identity()),
                target_pos=0.03,
                target_ke=15.0 if branch == 0 else 0.0,
                target_kd=1.0 if branch == 0 else 0.0,
                armature=0.05,
                limit_lower=-0.3,
                limit_upper=0.3,
            )
        )
        if branch == 1:
            cfg = scene.default_shape_cfg.copy()
            cfg.density, cfg.mu, cfg.restitution = 0.0, 0.6, 0.3
            scene.add_shape_sphere(
                body, radius=0.055, xform=wp.transform(wp.vec3(0.02, 0.0, 0.0), wp.quat_identity()), cfg=cfg
            )
    scene.add_articulation(joints)
    scene.add_constraint_mimic(joint0=joints[3], joint1=joints[2], coef0=offset, coef1=scale)
    scene.joint_q[:] = [0.0, 0.1, offset + scale * 0.1 + error]
    scene.joint_qd[:] = [0.08, 0.7, -0.6]
    builder = newton.ModelBuilder()
    builder.replicate(scene, 2)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _case(enabled, offset=0.025, scale=-0.5, error=-0.015, *, friction_anchor_beta=0.0):
    model = _model(offset, scale, error)
    model.joint_velocity_limit.assign(np.tile([0.4, 0.25, 0.2], 2).astype(np.float32))
    # The reference arm runs ordinary dynamics; both keep one buffer set, so their rows compare after a step.
    pins = {"double_buffer": False, **({} if enabled else {"storage": "dense", "evaluator": "ordinary"})}
    with mock.patch.dict(SolverFeatherPGS._route_pins, pins):
        solver = SolverFeatherPGS(
            model,
            friction_anchor_beta=friction_anchor_beta,
            enable_joint_limits=True,
            enable_joint_velocity_limits=True,
            pgs_iterations=8,
            pgs_warmstart=False,
            pgs_beta=0.07,
            pgs_cfm=2e-4,
            dense_max_constraints=32,
            mf_max_constraints=32,
            update_mass_matrix_interval=2,
            restitution_velocity_threshold=0.01,
        )
    state, out, control = model.state(), model.state(), model.control()
    velocity = state.joint_qd.numpy().copy().reshape(2, -1)
    velocity[1] *= -1.0
    state.joint_qd.assign(velocity.reshape(-1))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control.joint_f.assign(np.tile([0.03, -0.08, 0.11], 2).astype(np.float32))
    # Sorted contacts: both arms see the same contact order, which the by-contact patch checks key on.
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, deterministic=True)
    return model, solver, state, out, control, pipeline, pipeline.contacts()


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSparseVelocityMimicPhysics(unittest.TestCase):
    def _assert_state(self, reference, candidate):
        velocity_tests.TestSparseVelocityLimitPhysics._assert_state(self, reference, candidate)

    def _assert_rows(self, reference, candidate):
        velocity_tests.TestSparseVelocityLimitPhysics._assert_rows(self, reference, candidate)

    def test_loaded_mimic_velocity_contact_matches_dense(self):
        """Preserve bilateral response and speed clamps for either mimic error sign and friction law."""
        for beta, (offset, scale, error) in itertools.product((0.0, 0.2), ((0.025, -0.5, -0.015), (-0.03, 1.5, 0.015))):
            with self.subTest(friction_anchor_beta=beta, offset=offset, scale=scale, error=error):
                reference = _case(False, offset, scale, error, friction_anchor_beta=beta)
                candidate = _case(True, offset, scale, error, friction_anchor_beta=beta)
                self.assertTrue(candidate[1]._sparse_velocity_limits)
                self.assertFalse(candidate[1]._sparse_row_support.enabled)
                self.assertEqual(candidate[1].laws.mimic_count, 2)
                for step in range(4):
                    sparse_scene.step(reference)
                    sparse_scene.step(candidate)
                    self._assert_state(reference, candidate)
                    if step == 0:
                        self._assert_rows(reference, candidate)
                        for world, count in enumerate(candidate[1].constraint_count.numpy()):
                            kinds = candidate[1].row_type.numpy()[world, :count]
                            for kind in (MIMIC, VELOCITY_LIMIT, CONTACT):
                                self.assertIn(kind, kinds)
                            row = int(candidate[1].laws.mimic_slot.numpy()[world])
                            expected = 0.07 * error / sparse_scene.DT
                            self.assertAlmostEqual(candidate[1].rhs.numpy()[world, row], expected, places=5)

    def test_live_coefficients_enabled_graph_and_held_reset(self):
        """Read current mimic coefficients and enabled state without stale captured rows."""
        self._check_live_coefficients(0.0)

    def test_patch_live_coefficients_enabled_graph_and_held_reset(self):
        """Keep mimic coefficient updates, held resets and the patch anchor lifecycle through graph replay."""
        self._check_live_coefficients(0.2)

    def _check_live_coefficients(self, friction_anchor_beta):
        reference = _case(False, friction_anchor_beta=friction_anchor_beta)
        candidate = _case(True, friction_anchor_beta=friction_anchor_beta)
        self.assertTrue(candidate[1]._sparse_velocity_limits)
        for case in (reference, candidate):
            sparse_scene.step(case)
        held = candidate[1]._sparse_Linv.numpy().copy()
        for model, solver, state, *_ in (reference, candidate):
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (reference, candidate):
            sparse_scene.step(case)
        np.testing.assert_array_equal(candidate[1].mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(candidate[1]._sparse_Linv.numpy()[1], held[1])
        self._assert_state(reference, candidate)
        with wp.ScopedCapture("cuda:0") as capture:
            sparse_scene.step(candidate)
            sparse_scene.step(candidate)
        for enabled, offset, scale in ((False, -0.02, 1.7), (True, -0.02, 1.7), (True, 0.03, 0.0), (True, 0.01, -0.7)):
            for model, solver, *_ in (reference, candidate):
                model.constraint_mimic_enabled.fill_(enabled)
                model.constraint_mimic_coef0.fill_(offset)
                model.constraint_mimic_coef1.fill_(scale)
                solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
            sparse_scene.step(reference)
            sparse_scene.step(reference)
            wp.capture_launch(capture.graph)
            self._assert_state(reference, candidate)
            self._assert_rows(reference, candidate)
            for case in (reference, candidate):
                for world, count in enumerate(case[1].constraint_count.numpy()):
                    kinds = case[1].row_type.numpy()[world, :count]
                    self.assertEqual(int(np.count_nonzero(kinds == MIMIC)), int(enabled))


if __name__ == "__main__":
    unittest.main()

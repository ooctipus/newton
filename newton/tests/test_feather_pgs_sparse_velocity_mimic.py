# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve bilateral mimic rows with factor-coordinate velocity limits."""

import itertools
import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import class_plan
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_MIMIC as MIMIC
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_sparse_scene as sparse_scene
from newton.tests import test_feather_pgs_sparse_velocity_limits as velocity_tests
from newton.tests.feather_pgs_sparse_scene import TopologyReached, eligible_inputs


class TestSparseVelocityMimicAdmission(unittest.TestCase):
    def test_mimic_velocity_owner_reaches_topology(self):
        """Admit a same-articulation mimic under the explicit velocity owner."""
        inputs = eligible_inputs()
        inputs.options.enable_joint_velocity_limits = True
        inputs.laws.mimic_count = 1
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaises(TopologyReached),
        ):
            class_plan.packed_rule(inputs)


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


def _case(enabled, offset=0.025, scale=-0.5, error=-0.015, *, friction_anchor_beta=0.2):
    model = _model(offset, scale, error)
    model.joint_velocity_limit.assign(np.tile([0.4, 0.25, 0.2], 2).astype(np.float32))
    # The reference arm runs dense storage.
    pins = {} if enabled else {"storage": "dense"}
    solver = SolverFeatherPGS(
        model,
        pins=SolverFeatherPGS.Pins(**pins),
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


def _records(solver, group: int) -> np.ndarray:
    """The walk records of the sparse group's ``group``-th articulation (the factor the walk refreshes)."""
    art = int(solver.forest.group_to_art[solver.spec.classes.packed_size].numpy()[group])
    d0 = int(solver.forest.articulation_dof_start.numpy()[art])
    return solver.factor.walk_data.numpy()[d0 : d0 + int(solver.forest.articulation_dof_count.numpy()[art])].copy()


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSparseVelocityMimicPhysics(unittest.TestCase):
    def _assert_state(self, reference, candidate):
        velocity_tests.TestSparseVelocityLimitPhysics._assert_state(self, reference, candidate)

    def _assert_rows(self, reference, candidate):
        velocity_tests.TestSparseVelocityLimitPhysics._assert_rows(self, reference, candidate)

    def test_loaded_mimic_velocity_contact_matches_dense(self):
        """Preserve bilateral response and speed clamps for either mimic error sign."""
        for beta, (offset, scale, error) in itertools.product((0.2,), ((0.025, -0.5, -0.015), (-0.03, 1.5, 0.015))):
            with self.subTest(friction_anchor_beta=beta, offset=offset, scale=scale, error=error):
                reference = _case(False, offset, scale, error, friction_anchor_beta=beta)
                candidate = _case(True, offset, scale, error, friction_anchor_beta=beta)
                self.assertTrue(candidate[1].sweep.sparse_velocity_limits)
                self.assertFalse(candidate[1].placement.sparse_row_support.enabled)
                self.assertEqual(candidate[1].laws.mimic_count, 2)
                for step in range(4):
                    sparse_scene.step(reference)
                    sparse_scene.step(candidate)
                    self._assert_state(reference, candidate)
                    if step == 0:
                        self._assert_rows(reference, candidate)
                        for world, count in enumerate(candidate[1].placement.constraint_count.numpy()):
                            kinds = candidate[1].placement.row_type.numpy()[world, :count]
                            for kind in (MIMIC, VELOCITY_LIMIT, CONTACT):
                                self.assertIn(kind, kinds)
                            row = int(candidate[1].laws.mimic_slot.numpy()[world])
                            expected = 0.07 * error / sparse_scene.DT
                            self.assertAlmostEqual(candidate[1].placement.rhs.numpy()[world, row], expected, places=5)

    def test_patch_live_coefficients_enabled_graph_and_held_reset(self):
        """Keep mimic coefficient updates, held resets and the patch anchor lifecycle through graph replay."""
        self._check_live_coefficients(0.2)

    def _check_live_coefficients(self, friction_anchor_beta):
        reference = _case(False, friction_anchor_beta=friction_anchor_beta)
        candidate = _case(True, friction_anchor_beta=friction_anchor_beta)
        self.assertTrue(candidate[1].sweep.sparse_velocity_limits)
        for case in (reference, candidate):
            sparse_scene.step(case)
        held = _records(candidate[1], 1)
        for model, solver, state, *_ in (reference, candidate):
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (reference, candidate):
            sparse_scene.step(case)
        np.testing.assert_array_equal(candidate[1].factor.mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(_records(candidate[1], 1), held)
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
                for world, count in enumerate(case[1].placement.constraint_count.numpy()):
                    kinds = case[1].placement.row_type.numpy()[world, :count]
                    self.assertEqual(int(np.count_nonzero(kinds == MIMIC)), int(enabled))


if __name__ == "__main__":
    unittest.main()

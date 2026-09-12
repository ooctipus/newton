# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject DOF-less external contacts without rejecting responsive fixed children."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverMuJoCo


def _fixture(kind="locked_d6"):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    root_pos = (0.0, 0.0, 0.09)
    root = builder.add_link(
        xform=wp.transform(root_pos, wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
        is_kinematic=kind.startswith("kinematic"),
        label="root",
    )
    if kind == "fixed_child":
        joint = builder.add_joint_free(root)
        child = builder.add_link(xform=wp.transform(root_pos, wp.quat_identity()), label="fixed_child")
        child_joint = builder.add_joint_fixed(root, child)
        builder.add_articulation([joint, child_joint])
        case = child
    else:
        if kind == "locked_d6":
            joint = builder.add_joint_d6(-1, root, parent_xform=wp.transform(root_pos, wp.quat_identity()))
        elif kind.startswith("kinematic"):
            joint = builder.add_joint_free(root)
        else:
            # Explicitly retain raw fixed-root/ground pairs to exercise the converter.
            joint = builder.add_joint_fixed(
                -1, root, parent_xform=wp.transform(root_pos, wp.quat_identity()), collision_filter_parent=False
            )
        builder.add_articulation([joint])
        case = root
    case_shape = builder.add_shape_sphere(case, radius=0.1, label="case")
    dynamic_pos = (0.15, 0.0, 0.09) if kind.endswith("mixed") else (2.0, 0.0, 0.09)
    dynamic = builder.add_body(xform=wp.transform(dynamic_pos, wp.quat_identity()), label="dynamic")
    dynamic_shape = builder.add_shape_sphere(dynamic, radius=0.1, label="dynamic_geom")
    model = builder.finalize()
    model.request_contact_attributes("force")
    solver = SolverMuJoCo(model, use_mujoco_contacts=False, nconmax=64, njmax=64, iterations=8, jacobian="sparse")
    solver._create_inverse_shape_mapping()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, solver, state, pipeline, contacts, case_shape, dynamic_shape


def _retained_pairs(solver):
    count = int(solver.mjw_data.nacon.numpy()[0])
    shapes = solver.mjc_geom_to_newton_shape.numpy()
    geoms = solver.mjw_data.contact.geom.numpy()[:count]
    worlds = solver.mjw_data.contact.worldid.numpy()[:count]
    return [tuple(sorted((int(shapes[w, a]), int(shapes[w, b])))) for w, (a, b) in zip(worlds, geoms, strict=True)]


class TestMuJoCoResponseFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mujoco, _ = SolverMuJoCo.import_mujoco()

    def test_fixed_and_locked_roots_match_native_response_filter(self):
        """Reject fixed articulated roots exported as non-world mocap bodies."""
        for kind in ("fixed", "locked_d6"):
            with self.subTest(kind=kind):
                model, solver, state, _, contacts, case, dynamic = _fixture(kind)
                geom = int(solver.newton_shape_to_mjc_geom.numpy()[case])
                body = int(solver.mj_model.geom_bodyid[geom])
                self.assertGreater(int(solver.mj_model.body_weldid[body]), 0)
                self.assertEqual(int(solver.mj_model.body_dofnum[body]), 0)
                self.assertGreaterEqual(int(solver.mj_model.body_mocapid[body]), 0)
                solver._convert_contacts_to_mjwarp(model, state, contacts)
                self.assertEqual(_retained_pairs(solver), [(0, dynamic)])
                reference = self.mujoco.MjData(solver.mj_model)
                self.mujoco.mj_forward(solver.mj_model, reference)
                self.assertEqual(reference.ncon, 1)
                self.assertGreater(reference.nefc, 0)

    def test_fixed_child_of_dynamic_body_keeps_ancestor_response(self):
        """Retain a fixed child whose welded ancestor owns dynamic DOFs."""
        model, solver, state, _, contacts, case, dynamic = _fixture("fixed_child")
        geom = int(solver.newton_shape_to_mjc_geom.numpy()[case])
        body = int(solver.mj_model.geom_bodyid[geom])
        self.assertEqual(int(solver.mj_model.body_dofnum[body]), 0)
        self.assertGreater(int(solver.mj_model.body_dofnum[solver.mj_model.body_weldid[body]]), 0)
        solver._convert_contacts_to_mjwarp(model, state, contacts)
        self.assertCountEqual(_retained_pairs(solver), [(0, case), (0, dynamic)])

    def test_mocap_dynamic_and_existing_kinematic_semantics(self):
        """Preserve responsive mixed contacts and the existing kinematic exclusion."""
        for kind in ("mixed", "kinematic", "kinematic_mixed"):
            with self.subTest(kind=kind):
                model, solver, state, _, contacts, case, dynamic = _fixture(kind)
                solver._convert_contacts_to_mjwarp(model, state, contacts)
                expected = [(0, dynamic), tuple(sorted((case, dynamic)))] if kind.endswith("mixed") else [(0, dynamic)]
                self.assertCountEqual(_retained_pairs(solver), expected)

    def test_cached_generation_and_invalidation_keep_rejected_mapping(self):
        """Keep rejected raw contacts unmapped across cached and invalidated calls."""
        model, solver, state, pipeline, contacts, case, dynamic = _fixture()
        raw_count = int(contacts.rigid_contact_count.numpy()[0])
        for action in (None, "cached", "notify", "empty", "empty_cached", "collision"):
            if action == "notify":
                solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
            elif action == "empty":
                contacts.rigid_contact_count.zero_()
                contacts.contact_generation.assign(contacts.contact_generation.numpy() + 1)
            elif action == "collision":
                pipeline.collide(state, contacts)
            solver._convert_contacts_to_mjwarp(model, state, contacts)
            if action in ("empty", "empty_cached"):
                self.assertEqual(_retained_pairs(solver), [])
                np.testing.assert_array_equal(solver._contact_tid_to_cid.numpy()[:raw_count], -1)
                continue
            raw_count = int(contacts.rigid_contact_count.numpy()[0])
            a = contacts.rigid_contact_shape0.numpy()[:raw_count]
            b = contacts.rigid_contact_shape1.numpy()[:raw_count]
            rejected = (a == case) | (b == case)
            self.assertTrue(rejected.any())
            self.assertEqual(_retained_pairs(solver), [(0, dynamic)])
            np.testing.assert_array_equal(solver._contact_tid_to_cid.numpy()[:raw_count][rejected], -1)
            self.assertEqual(int(solver._last_nacon_count.numpy()[0]), 1)

    def test_graph_replay_observes_generation_and_empty_contact_transition(self):
        """Replay conversion and publication with stable buffers and changing device counts."""
        if not wp.get_device().is_cuda:
            self.skipTest("CUDA graph replay requires a CUDA device")
        model, solver, state, pipeline, contacts, case, dynamic = _fixture()
        published = pipeline.contacts()
        solver._convert_contacts_to_mjwarp(model, state, contacts)
        solver.update_contacts(published, state)
        with wp.ScopedCapture(device=model.device) as capture:
            solver._convert_contacts_to_mjwarp(model, state, contacts)
            solver.update_contacts(published, state)
        for empty in (False, True, False):
            if empty:
                contacts.rigid_contact_count.zero_()
                contacts.contact_generation.assign(contacts.contact_generation.numpy() + 1)
            else:
                pipeline.collide(state, contacts)
            for _ in range(2):
                wp.capture_launch(capture.graph)
                expected = [] if empty else [(0, dynamic)]
                self.assertEqual(_retained_pairs(solver), expected)
                count = int(published.rigid_contact_count.numpy()[0])
                self.assertEqual(count, len(expected))
                self.assertNotIn(case, published.rigid_contact_shape0.numpy()[:count])
                self.assertNotIn(case, published.rigid_contact_shape1.numpy()[:count])

    def test_solve_and_public_force_exclude_nonresponsive_contacts(self):
        """Publish finite responsive forces without DOF-less active EFC rows."""
        model, solver, state, pipeline, contacts, case, dynamic = _fixture()
        state_out = model.state()
        solver.step(state, state_out, model.control(), contacts, 1.0 / 240.0)
        nefc = int(solver.mjw_data.nefc.numpy()[0])
        self.assertGreater(nefc, 0)
        self.assertTrue(np.all(solver.mjw_data.efc.J_rownnz.numpy()[0, :nefc] > 0))
        self.assertTrue(np.isfinite(solver.mjw_data.efc.force.numpy()[0, :nefc]).all())
        published = pipeline.contacts()
        solver.update_contacts(published, state_out)
        count = int(published.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 1)
        self.assertNotIn(case, published.rigid_contact_shape0.numpy()[:count])
        self.assertNotIn(case, published.rigid_contact_shape1.numpy()[:count])
        forces = published.force.numpy()[:count]
        self.assertTrue(np.isfinite(forces).all())
        self.assertGreater(float(np.linalg.norm(forces)), 0.0)
        self.assertLess(float(np.linalg.norm(forces)), 1.0e6)
        self.assertEqual(_retained_pairs(solver), [(0, dynamic)])


if __name__ == "__main__":
    unittest.main()

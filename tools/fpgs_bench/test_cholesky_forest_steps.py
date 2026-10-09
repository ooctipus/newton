# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the original loaded solve around topology-derived mixed cached forests."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT
from tools.fpgs_bench.test_cholesky_forest_cache import DT, build_plan, make_solver, model_fixture
from tools.fpgs_bench.test_g1_kinetic_state import physical_mass, public_state


def make_case(enabled, variant):
    """Use one contact per world with all three primary/free/prescribed pairings."""
    model = model_fixture("cuda:0", variant=variant, reordered=True)
    solver = make_solver(model, enabled=enabled, permuted=True)
    plan = build_plan(solver)
    state, output, control = model.state(), model.state(), model.control()
    state.joint_qd.assign(np.linspace(-0.12, 0.16, model.joint_dof_count, dtype=np.float32))
    state.body_f.assign(np.random.default_rng(101).normal(0, 0.05, (model.body_count, 6)).astype(np.float32))
    output.body_f.assign(state.body_f)
    control.joint_f.assign(np.random.default_rng(102).normal(0, 0.07, model.joint_dof_count).astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    contacts = newton.Contacts(rigid_contact_max=32, soft_contact_max=0, device=model.device)
    first, second = np.full(32, -1, np.int32), np.full(32, -1, np.int32)
    point0, point1, normal = (np.zeros((32, 3), np.float32) for _ in range(3))
    shapes, poses = model.shape_body.numpy(), state.body_q.numpy()
    for world, (a, b) in enumerate(((9, 11), (9, 12), (11, 12))):
        body0, body1 = plan["body_ids"][world, [a, b]]
        first[world], second[world] = (int(np.flatnonzero(shapes == body)[0]) for body in (body0, body1))
        normal[world] = [0, 0, 1]
        point0[world] = [0.01, -0.01, 0]
        world_point = wp.transform_point(wp.transform(*poses[body0]), wp.vec3(*point0[world]))
        point1[world] = np.asarray(
            wp.transform_point(wp.transform_inverse(wp.transform(*poses[body1])), world_point - wp.vec3(0, 0, 0.001))
        )
    for name, values in (
        ("shape0", first),
        ("shape1", second),
        ("point0", point0),
        ("point1", point1),
        ("normal", normal),
    ):
        getattr(contacts, "rigid_contact_" + name).assign(values)
    contacts.rigid_contact_count.fill_(3)
    return model, solver, state, output, control, contacts


def step(case):
    """Advance one original eight-sweep step and exchange the state banks."""
    model, solver, state, output, control, contacts = case
    solver.step(state, output, control, contacts, DT)
    return model, solver, output, state, control, contacts


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestCholeskyForestSteps(unittest.TestCase):
    def assert_pair(self, cases):
        """Retain inherited physical tolerances and observe loaded unilateral rows."""
        for case in cases:
            model, solver, state, *_ = case
            solver.check_constraint_capacity()
            public_state(self, model, state)
            for name in ("constraint_count", "mf_constraint_count"):
                self.assertGreater(int(getattr(solver, name).numpy().sum()), 0)
            for prefix in ("", "mf_"):
                counts = getattr(solver, prefix + "constraint_count").numpy()
                impulses = getattr(solver, prefix + "impulses").numpy()
                kinds = getattr(solver, prefix + "row_type").numpy()
                for world, count in enumerate(counts):
                    active = impulses[world, :count]
                    self.assertTrue(np.isfinite(active).all())
                    # Contact normal and scalar limit laws are unilateral.
                    unilateral = np.isin(
                        kinds[world, :count], [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT]
                    )
                    self.assertTrue(np.all(active[unilateral] >= -1e-7))
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            a, b = (getattr(case[2], name).numpy().reshape(case[0].world_count, -1).astype(float) for case in cases)
            error = np.linalg.norm(a - b, axis=1) / (1 + np.linalg.norm(a, axis=1))
            self.assertLess(float(error.max()), 7e-4, name)
        owner = cases[1][1]._franka_kinetic_state
        owner.check()
        np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)

    def assert_factor(self, case, state):
        """Compare retained factors with public-twist physical mass plus original drive augmentation."""
        model, solver, *_ = case
        plan = solver._franka_kinetic_state.host_plan
        mass = physical_mass(model, state)
        rows, coefficients = solver._augmented_drive_row_by_dof.numpy(), solver.aug_row_K.numpy()
        for size, first, groups in ((9, 0, plan["primary_group"]), (6, 9, plan["secondary_group"])):
            lower = np.tril(solver.L_by_size[size].numpy().astype(float))
            regularizer = solver.R_by_size[size].numpy()
            for world, group in enumerate(groups):
                ids = plan["dof_ids"][world, first : first + size]
                diagonal = regularizer[group].astype(float).copy()
                diagonal += [max(0.0, float(coefficients[rows[dof]])) if rows[dof] >= 0 else 0 for dof in ids]
                expected = mass[np.ix_(ids, ids)] + np.diag(diagonal)
                np.testing.assert_allclose(lower[group] @ lower[group].T, expected, atol=5e-6, rtol=5e-5)

    def test_loaded_steps_and_model_changes(self):
        """Match complete loaded steps, held factors, selective resets and numeric notifications."""
        for variant in range(3):
            with self.subTest(variant=variant):
                cases = [make_case(enabled, variant) for enabled in (False, True)]
                self.assertIsNone(cases[0][1]._franka_kinetic_state)
                self.assertIsNotNone(cases[1][1]._franka_kinetic_state)
                held = None
                for index in range(24):
                    for arm, case in enumerate(cases):
                        model, solver, state, *_ = case
                        if index in (5, 13, 19):
                            mask = wp.array([index == 5, index == 13, index == 19], dtype=wp.bool, device=model.device)
                            solver.reset(state, world_mask=mask)
                        if index == 9:
                            model.body_mass.assign(model.body_mass.numpy() * np.float32(1.01))
                            model.body_inertia.assign(model.body_inertia.numpy() * np.float32(1.02))
                            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                        cases[arm] = step(case)
                        if arm and index == 0:
                            self.assert_factor(cases[arm], state)
                            held = {size: solver.L_by_size[size].numpy().copy() for size in (9, 6)}
                        if arm and index == 1:
                            for size in (9, 6):
                                np.testing.assert_array_equal(solver.L_by_size[size].numpy(), held[size])
                    self.assert_pair(cases)

    def test_complete_graph_and_dynamic_masks(self):
        """Replay a full held-factor cycle under changing world reset masks."""
        cases = [make_case(enabled, 2) for enabled in (False, True)]
        graphs, masks = [], []
        for arm in range(len(cases)):
            case = cases[arm]
            for _ in range(4):
                case = step(case)
            mask = wp.array([False] * 3, dtype=wp.bool, device=case[0].device)
            with wp.ScopedCapture(device=case[0].device) as capture:
                case[1].seed_double_buffer_events()
                case[1].reset(case[2], world_mask=mask)
                for _ in range(4):
                    case = step(case)
            cases[arm] = case
            graphs.append(capture.graph)
            masks.append(mask)
        for selected in ([True, False, False], [False, True, False], [False, False, True], [False] * 3, [True] * 3):
            for mask, graph in zip(masks, graphs, strict=True):
                mask.assign(np.array(selected))
                wp.capture_launch(graph)
            self.assert_pair(cases)


if __name__ == "__main__":
    unittest.main(verbosity=2)

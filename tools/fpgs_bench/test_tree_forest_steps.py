# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise cached forests inside the unchanged complete native contact solve."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import forest_plan
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_sparse_solver as native
from newton.tests.feather_pgs_sparse_scene import _external_forces
from tools.fpgs_bench.forest_row_order import RowOrder
from tools.fpgs_bench.public_state_oracle import public_state


def make_case(enabled, *, permuted=False, motion=False):
    """Build group maps before any operator, publication or row allocation."""
    size_groups = forest_plan._size_groups

    def grouping(plan, model):
        size_groups(plan, model)
        if permuted:
            order = np.array([2, 0, 3, 1], np.int32)
            plan.group_to_art[9].assign(order)
            plan.art_group_idx.assign(np.argsort(order).astype(np.int32))

    with (
        patch.dict(SolverFeatherPGS._route_pins, {} if enabled else {"evaluator": "ordinary"}),
        patch.object(forest_plan, "_size_groups", grouping),
    ):
        model = native._build_model(robot_count=2)
        model.joint_velocity_limit.fill_(0.08)
        solver = native._make_solver(
            model,
            True,
            dense_max_constraints=96,
            mf_max_constraints=64,
            enable_joint_velocity_limits=motion,
        )
    state, output, control = model.state(), model.state(), model.control()
    control.joint_f.assign(np.random.default_rng(517).normal(0.0, 0.04, model.joint_dof_count).astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, solver, state, output, control, pipeline, pipeline.contacts()


def step(case):
    """Use identical current external loads and alternate generalized state banks."""
    model, solver, state, output, control, pipeline, contacts = case
    state.clear_forces()
    wp.launch(_external_forces, dim=model.body_count, inputs=[state.body_f], device=model.device)
    pipeline.collide(state, contacts)
    solver.step(state, output, control, contacts, native.DT)
    return model, solver, output, state, control, pipeline, contacts


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestTreeForestSteps(unittest.TestCase):
    def test_loaded_steps_reset_and_permuted_groups(self):
        """Preserve coupled impulses and all public state across masked resets."""
        inspector = native.TestFeatherPGSSparseSolver()
        for permuted, motion in ((False, False), (True, False), (True, True)):
            with self.subTest(permuted=permuted, motion=motion):
                cases = [make_case(enabled, permuted=permuted, motion=motion) for enabled in (False, True)]
                self.assertIsNone(cases[0][1].factor.held)
                self.assertIsNotNone(cases[1][1].factor.held)
                coupled = False
                for index in range(24):
                    orders = []
                    for arm, case in enumerate(cases):
                        solver = case[1]
                        if index in (5, 13, 19):
                            mask = wp.array([index != 13, index == 13], dtype=wp.bool, device=case[0].device)
                            solver.reset(case[2], world_mask=mask)
                        solve = solver._launch_matrix_free_gs_solve

                        def ordered(*args, _case=case, _solve=solve, _orders=orders, **kwargs):
                            _orders.append(inspector._order_multi_articulation_rows(_case))
                            return _solve(*args, **kwargs)

                        with patch.object(solver, "_launch_matrix_free_gs_solve", side_effect=ordered):
                            cases[arm] = step(case)
                    self.assertEqual(orders[0], orders[1])
                    inspector.assert_states_close(cases[0][2], cases[1][2])
                    model, solver, state, _, _, _, contacts = cases[1]
                    public_state(self, model, state)
                    inspector.assert_sparse_state(cases[1])
                    np.testing.assert_array_equal(solver.factor.held.groups[0].data.current_valid.numpy(), 1)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    bodies, arts = model.shape_body.numpy(), solver.forest.body_to_articulation.numpy()
                    for a, b in zip(
                        contacts.rigid_contact_shape0.numpy()[:count],
                        contacts.rigid_contact_shape1.numpy()[:count],
                        strict=True,
                    ):
                        coupled |= bodies[a] >= 0 and bodies[b] >= 0 and arts[bodies[a]] != arts[bodies[b]]
                self.assertTrue(coupled, "Fixture must contact between independent responsive trees")

    def test_complete_graph_replay_and_dynamic_reset_mask(self):
        """Publish both trees after captured reset masks and held-factor cycles."""
        inspector = native.TestFeatherPGSSparseSolver()
        for motion in (False, True):
            with self.subTest(motion=motion):
                cases = [make_case(enabled, permuted=True, motion=motion) for enabled in (False, True)]
                graphs, masks = [], []
                ordering = [RowOrder(case) for case in cases]
                for arm in range(len(cases)):
                    case = cases[arm]
                    with patch.object(case[1], "_launch_matrix_free_gs_solve", side_effect=ordering[arm]):
                        for _ in range(4):
                            case = step(case)
                        mask = wp.array([False, False], dtype=wp.bool, device=case[0].device)
                        with wp.ScopedCapture(device=case[0].device) as capture:
                            case[1].reset(case[2], world_mask=mask)
                            # Match the retained four-step mass-refresh pattern.
                            for _ in range(4):
                                case = step(case)
                    cases[arm] = case
                    graphs.append(capture.graph)
                    masks.append(mask)
                for selected in ([True, False], [False, True], [False, False], [True, True]):
                    for mask, graph in zip(masks, graphs, strict=True):
                        mask.assign(np.array(selected))
                        wp.capture_launch(graph)
                    for case in cases:
                        inspector.assert_sparse_state(case)
                    public_state(self, cases[1][0], cases[1][2])
                    np.testing.assert_array_equal(cases[1][1].factor.held.groups[0].data.current_valid.numpy(), 1)
                    inspector.assert_states_close(cases[0][2], cases[1][2])


if __name__ == "__main__":
    unittest.main(verbosity=2)

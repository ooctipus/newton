# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check grouped dynamics composed with the native sparse factor owner."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.test_feather_pgs_sparse_solver import STATE_FIELDS, _build_model, _make_solver, _step


def _case(grouped, *, free_count=0, friction_anchor_beta=0.0):
    """Construct the same eight-sweep native problem with either dynamics mapping.

    ``friction_anchor_beta=None`` selects the solver's default patch friction.
    """
    model = _build_model(free_count=free_count)
    with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_GROUPED_DYNAMICS": "1"}):
        solver = _make_solver(
            model,
            True,
            grouped_dynamics=grouped,
            pgs_iterations=8,
            friction_anchor_beta=friction_anchor_beta,
            dense_max_constraints=64,
            mf_max_constraints=64,
        )
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
    return model, solver, state, out, model.control(), pipeline, pipeline.contacts()


@unittest.skipUnless(wp.is_cuda_available(), "Sparse grouped dynamics requires CUDA")
class TestSparseGroupedDynamics(unittest.TestCase):
    def assert_native_grouped(self, case):
        """Require sparse storage and prohibit dense-factor-writing fusions."""
        solver = case[1]
        self.assertEqual(solver._sparse_mass_matrix_size, 9)
        self.assertIsNotNone(solver._grouped_topology)
        self.assertFalse(solver._grouped_tau_mass)
        self.assertFalse(solver._grouped_mass)
        self.assertFalse(solver._fused_k1)
        self.assertEqual(solver.L_by_size[9].shape, (1, 1, 1))

    def assert_physics_close(self, first, second):
        """Compare current poses, velocities, factors and unilateral feasibility."""
        for field in STATE_FIELDS:
            np.testing.assert_allclose(
                getattr(first[2], field).numpy(),
                getattr(second[2], field).numpy(),
                rtol=2e-4,
                atol=2e-3 if field.endswith("qd") else 2e-4,
                err_msg=field,
            )
        for case in (first, second):
            solver = case[1]
            self.assertFalse(solver._sparse_mass_matrix_status.numpy().any())
            self.assertFalse(solver.constraint_overflow.numpy().any())
            for world, count in enumerate(solver.constraint_count.numpy()):
                lam = solver.impulses.numpy()[world, :count]
                types = solver.row_type.numpy()[world, :count]
                self.assertTrue(np.isfinite(lam).all())
                self.assertTrue(np.all(lam[np.isin(types, (0, 3))] >= -1e-7))

    def test_grouped_sparse_contact_response(self):
        """Preserve eight-sweep physics on branched robot worlds.

        Worlds that pair an articulation with free rigid bodies keep the dense route under
        both friction laws, where grouped dynamics runs on ordinary factors.
        """
        plain, grouped = (_case(flag) for flag in (False, True))
        self.assert_native_grouped(grouped)
        contact_seen = limit_seen = False
        for _ in range(8):
            for case in (plain, grouped):
                _step(case)
            contact_seen |= bool(grouped[-1].rigid_contact_count.numpy()[0])
            limit_seen |= bool(np.any(grouped[1].row_type.numpy() == 3))
            self.assert_physics_close(plain, grouped)
        self.assertTrue(contact_seen and limit_seen)

    def test_free_body_worlds_keep_the_dense_route(self):
        """Keep mixed robot/free worlds on the dense route under both friction laws."""
        for beta in (0.0, None):
            with self.subTest(friction_anchor_beta=beta):
                solver = _case(True, free_count=2, friction_anchor_beta=beta)[1]
                self.assertIsNone(solver._sparse_mass_matrix_size)
                reason = solver.get_path_selection()["reasons"]["sparse_mass_matrix"]
                self.assertIn("free rigid bodies keep the dense route", reason)
                self.assertIsNotNone(solver._grouped_topology)

    def test_grouped_sparse_graph_held_factor_and_masked_reset(self):
        """Replay the mass cadence and refresh only the explicitly reset world."""
        eager, graph = _case(False), _case(True)
        self.assert_native_grouped(graph)
        for case in (eager, graph):
            _step(case)
        with wp.ScopedCapture("cuda:0") as capture:
            for _ in range(4):
                _step(graph)
        for _ in range(4):
            wp.capture_launch(capture.graph)
            for _ in range(4):
                _step(eager)
        self.assert_physics_close(eager, graph)
        held = graph[1]._sparse_Linv.numpy().copy()
        for model, solver, state, *_ in (eager, graph):
            q = state.joint_q.numpy()
            q[2] += 0.01
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (eager, graph):
            _step(case)
            np.testing.assert_array_equal(case[1].mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(graph[1]._sparse_Linv.numpy()[1], held[1])
        self.assert_physics_close(eager, graph)

    def test_default_grouped_keeps_existing_owner(self):
        """Retain the established grouped owner unless composition is requested."""
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_GROUPED_DYNAMICS": "0"}):
            solver = _make_solver(_build_model(), True, grouped_dynamics=True)
        self.assertIsNone(solver._sparse_mass_matrix_size)
        self.assertIsNotNone(solver._grouped_topology)


if __name__ == "__main__":
    unittest.main()

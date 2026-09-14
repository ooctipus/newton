# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded full-Solver.step lifecycle checks; not a new benchmark owner."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sparse_factor
from newton._src.solvers.feather_pgs.small_step_dispatch import SmallStep
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from tools.fpgs_bench.test_sparse_factor import fixture

ENV = {
    "FEATHER_PGS_SPARSE_FACTOR": "1",
    "FEATHER_PGS_SINGLE_FACTOR": "0",
    "FEATHER_PGS_SPARSE_PACKETS": "0",
    "FEATHER_PGS_SPARSE_CONTACT_BLOCK": "0",
    "FEATHER_PGS_SPARSE_PARALLEL_LIMITS": "1",
    "FEATHER_PGS_SPARSE_LEVEL_UPDATE": "1",
    "FEATHER_PGS_SPARSE_METRIC_TANGENTS": "1",
}


def original_case(device="cpu"):
    """Reuse the exact G1 USD/model fixture with no small owner on its helper."""
    with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_SMALL_STEP": "0"}):
        return fixture(device)


def construct(case, enabled):
    """Construct actual async production dispatch with the original eight allowance."""
    with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_SMALL_STEP": str(int(enabled))}):
        return SolverFeatherPGS(
            case["model"],
            pgs_mode="matrix_free",
            pgs_iterations=8,
            dense_max_constraints=100,
            enable_joint_limits=True,
            joint_limit_activation_gap=0.01,
            update_mass_matrix_interval=2,
            mf_gs_incremental_rows=0,
            fuse_joint_velocity_limits=False,
            grouped_dynamics=True,
            use_parallel_streams=True,
            double_buffer=False,
        )


def author_state(case, solver, state, active_limits, contact_count):
    """Use original joint bounds and current contact geometry for an exact row boundary."""
    model, contacts = case["model"], case["contacts"]
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    q = model.joint_q.numpy().copy()
    q[7:] = 0.5 * (lower[6:] + upper[6:])
    q[7 : 7 + active_limits] = lower[6 : 6 + active_limits] + 0.5 * solver.joint_limit_activation_gap
    state.joint_q.assign(q)
    state.joint_qd.zero_()
    state.body_f.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver._fk_id_cache_valid.zero_()
    poses = state.body_q.numpy()
    shape_body = model.shape_body.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()
    shape1 = contacts.rigid_contact_shape1.numpy()
    point0 = contacts.rigid_contact_point0.numpy()
    point1 = contacts.rigid_contact_point1.numpy()
    normal = contacts.rigid_contact_normal.numpy()
    for i in range(8):
        source = i % 3
        shape0[i], shape1[i] = shape0[source], shape1[source]
        point0[i] = [0.015 * ((i % 3) - 1), -0.01, -0.05]
        body = int(shape_body[shape0[i]])
        point1[i] = np.asarray(wp.transform_point(wp.transform(*poses[body]), wp.vec3(*point0[i])))
        point1[i, 2] += 0.003
        normal[i] = [0.0, 0.0, -1.0]
    for name, data in (
        ("shape0", shape0),
        ("shape1", shape1),
        ("point0", point0),
        ("point1", point1),
        ("normal", normal),
    ):
        getattr(contacts, "rigid_contact_" + name).assign(data)
    contacts.rigid_contact_count.assign(np.array([contact_count], np.int32))


class TestSmallStepLifecycleCPU(unittest.TestCase):
    def test_unsupported_constructor_keeps_original_arrays(self):
        """Reject unsupported small dispatch before any matrix ownership mutation."""
        case = original_case()
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FACTOR": "0", "FEATHER_PGS_SPARSE_SMALL_STEP": "0"}):
            solver = SolverFeatherPGS(case["model"], pgs_mode="split", dense_max_constraints=100)
        plan, host = sparse_factor.make_plan(case["model"], solver)
        names = ("H_by_size", "L_by_size", "J_by_size", "Y_by_size", "tau_by_size", "qdd_by_size")
        old = {name: getattr(solver, name)[43] for name in names}
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_SMALL_STEP": "1"}):
            with self.assertRaisesRegex(ValueError, "async augmented"):
                sparse_factor.SparseFactor(solver, plan, host)
        for name, previous in old.items():
            self.assertIs(getattr(solver, name)[43], previous)

    def test_live_force_route_guard(self):
        """Reject newly enabled direct or fused force producers before stepping."""
        solver = SimpleNamespace(
            dense_max_constraints=100,
            _async_augmented_drives=True,
            _parallel_augmented_drive_topology=True,
            drive_mode="augmented",
            _grouped_topology=object(),
            _direct_branch_tau_kernel=None,
            _fused_k1=False,
            _grouped_tau_mass=False,
            _grouped_mass=False,
        )
        owner = SimpleNamespace(
            metric_tangents=True, parallel_limit_prefix=True, packet_rows=False, block_contacts=False
        )
        small = SmallStep.__new__(SmallStep)
        small.owner, small.solver = owner, solver
        small.validate()
        for name, value in (("_grouped_topology", None), ("_direct_branch_tau_kernel", object()), ("_fused_k1", True)):
            previous = getattr(solver, name)
            setattr(solver, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "reconstruct"):
                small.validate()
            setattr(solver, name, previous)


class TestSmallStepLifecycleCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use only the parent-owned visible device when explicitly selected."""
        wp.init()
        if not wp.get_cuda_devices():
            raise unittest.SkipTest("Root owns the native GPU lease")
        cls.device = wp.get_cuda_devices()[0]

    def test_actual_step_selection_refresh_reset_and_graph(self):
        """Exercise both complete dispatch routes through actual Solver.step."""
        cases = [original_case(self.device), original_case(self.device)]
        solvers = [construct(case, enabled) for case, enabled in zip(cases, (False, True), strict=True)]
        states = [[case["model"].state(), case["model"].state()] for case in cases]
        controls = [case["model"].control() for case in cases]
        small = solvers[1]._sparse_factor.small_step
        self.assertIsNotNone(small)
        self.assertTrue(solvers[1]._async_augmented_drives)
        dt = 1.0 / 240.0
        seen = set()
        original_launch, original_tiled = wp.launch, wp.launch_tiled

        def watched(original):
            def launch(*args, **kwargs):
                kernel = kwargs.get("kernel", args[0] if args else None)
                seen.add(getattr(kernel, "key", ""))
                return original(*args, **kwargs)

            return launch

        def check_public(case, solver, state):
            solver.check_constraint_capacity()
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
            public = case["model"].state()
            newton.eval_fk(case["model"], state.joint_q, state.joint_qd, public)
            np.testing.assert_allclose(state.body_q.numpy(), public.body_q.numpy(), rtol=3e-5, atol=3e-6)

        held = None
        # All rows come from current authored geometry and unchanged original
        # bounds: 8 contact triplets +8/9 limits gives32/33, then empty/regrowing.
        for step, (limits, contacts, rows, selected) in enumerate(
            ((8, 8, 32, 1), (9, 8, 33, 0), (0, 0, 0, 1), (8, 8, 32, 1))
        ):
            for index, (case, solver) in enumerate(zip(cases, solvers, strict=True)):
                author_state(case, solver, states[index][0], limits, contacts)
                if step == 2:
                    solver.reset(states[index][0])
                    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                context = (
                    patch.object(wp, "launch", watched(original_launch)),
                    patch.object(wp, "launch_tiled", watched(original_tiled)),
                )
                with (
                    context[0] if index else patch.dict(os.environ, {}),
                    context[1] if index else patch.dict(os.environ, {}),
                ):
                    solver.step(states[index][0], states[index][1], controls[index], case["contacts"], dt)
                check_public(case, solver, states[index][1])
                self.assertEqual(int(solver.constraint_count.numpy()[0]), rows)
                self.assertFalse(solver._force_mass_update)
                np.testing.assert_array_equal(solver._mass_update_requested.numpy(), [0])
            self.assertEqual(int(small.selected.numpy()[0]), selected)
            self.assertEqual(int(small.fallback_counts.numpy()[0]), 0 if selected else rows)
            baseline, actual = (solver.v_out.numpy().astype(float) for solver in solvers)
            self.assertLess(np.linalg.norm(actual - baseline) / (1 + np.linalg.norm(baseline)), 7e-4)
            current = solvers[1]._sparse_factor.data.W.numpy()
            if step % 2:
                np.testing.assert_array_equal(solvers[1].mass_update_mask.numpy(), [0])
                np.testing.assert_array_equal(current, held)
            else:
                np.testing.assert_array_equal(solvers[1].mass_update_mask.numpy(), [1])
                held = current.copy()
        self.assertIn("small_step_force_rows_metric43_s18_c32", seen)
        self.assertIn("grouped_tau_fallback", seen)
        self.assertIn("sparse_factor_predict43_fallback", seen)
        self.assertIn("sparse_factor_contact_triplet18_fallback", seen)
        self.assertNotIn("grouped_tau", seen)
        self.assertNotIn("sparse_factor_predict43", seen)

        solver, case = solvers[1], cases[1]
        case["contacts"].rigid_contact_count.zero_()
        with wp.ScopedCapture(device=self.device) as capture:
            solver.step(states[1][1], states[1][0], controls[1], case["contacts"], dt)
            solver.step(states[1][0], states[1][1], controls[1], case["contacts"], dt)
        for _ in range(2):
            solver.reset(states[1][1])
            wp.capture_launch(capture.graph)
            check_public(case, solver, states[1][1])
            self.assertEqual(int(small.selected.numpy()[0]), 1)


if __name__ == "__main__":
    unittest.main()

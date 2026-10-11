# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise public automatic choices and forced implementation contracts."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_sparse_solver import _build_model


def _cpu_model():
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    return builder.finalize(device="cpu")


class TestFeatherPGSPathSelection(unittest.TestCase):
    def test_scalar_overrides_reach_solver(self):
        """Honor public choices independently of private test overrides."""
        choices = {
            "cholesky_kernel": "loop",
            "trisolve_kernel": "loop",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
        }
        with patch.dict(SolverFeatherPGS._kernel_overrides, {"cholesky_kernel": "tiled"}):
            solver = SolverFeatherPGS(_cpu_model(), **choices)
        for name, value in choices.items():
            self.assertEqual(getattr(solver, name), value)
        self.assertEqual(solver.get_path_selection()["requested"]["cholesky_kernel"], "loop")

    def test_auto_cpu_and_invalid_forced_choices(self):
        """Resolve CPU auto to scalar kernels and reject unsupported explicit choices."""
        model = _cpu_model()
        self.assertEqual(SolverFeatherPGS(model).get_path_selection()["pgs_kernel"], "loop")
        for name, value in (
            ("cholesky_kernel", "tiled"),
            ("trisolve_kernel", "tiled"),
            ("hinv_jt_kernel", "tiled"),
            ("delassus_kernel", "tiled"),
            ("pgs_kernel", "tiled_row"),
            ("sparse_mass_matrix", "on"),
            ("cholesky_kernel", "typo"),
        ):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                SolverFeatherPGS(model, **{name: value})

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factors require CUDA")
    def test_public_dense_kernels_override_sparse_and_fused_dispatch(self):
        """Select actual dense kernels while leaving unspecified choices automatic."""
        model = _build_model()
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", trisolve_kernel="loop")
        selected = solver.get_path_selection()
        self.assertEqual(selected["sparse_mass_matrix"], "off")
        self.assertEqual(selected["kernel_plan_by_dof"][9]["trisolve_kernel"], "loop")
        self.assertEqual(selected["requested"]["hinv_jt_kernel"], "auto")
        split = SolverFeatherPGS(model, pgs_mode="split", delassus_kernel="par_row_col")
        self.assertFalse(split._execution_plan.hinv_jt_fused_sizes)
        for kwargs, message in (
            ({"sparse_mass_matrix": "on", "trisolve_kernel": "loop"}, "conflicts"),
            ({"pgs_kernel": "loop"}, "pgs_kernel"),
            ({"delassus_kernel": "tiled"}, "delassus_kernel"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factors require CUDA")
    def test_force_sparse_and_dense_with_graph_replay(self):
        """Force both representations and preserve a contacting articulated rollout."""
        model = _build_model()
        outputs = []
        for choice in ("on", "off"):
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                sparse_mass_matrix=choice,
                enable_joint_limits=True,
                pgs_iterations=32,
                double_buffer=False,
            )
            self.assertEqual(solver.get_path_selection()["sparse_mass_matrix"], choice)
            state, output = model.state(), model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            pipeline = newton.CollisionPipeline(model, convex_collision="ordinary")
            contacts, control = pipeline.contacts(), model.control()

            def step(pipeline=pipeline, state=state, contacts=contacts, solver=solver, output=output, control=control):
                pipeline.collide(state, contacts)
                solver.step(state, output, control, contacts, 1.0 / 400.0)
                state.assign(output)

            step()
            with wp.ScopedCapture(device=model.device) as capture:
                step()
            for _ in range(8):
                wp.capture_launch(capture.graph)
            solver.check_constraint_capacity()
            outputs.append(state.joint_qd.numpy())
        np.testing.assert_allclose(outputs[0], outputs[1], atol=3e-4, rtol=3e-4)


if __name__ == "__main__":
    unittest.main()

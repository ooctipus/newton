# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise public automatic choices and forced implementation contracts."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_sparse_solver import _build_model, _make_solver


def _cpu_model():
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    return builder.finalize(device="cpu")


class TestFeatherPGSPathSelection(unittest.TestCase):
    def test_cpu_selection_reasons_are_detached_and_host_only(self):
        """Explain CPU fallbacks without synchronizing or exposing mutable solver state."""
        solver = SolverFeatherPGS(_cpu_model())
        with patch.object(wp, "synchronize", side_effect=AssertionError("unexpected synchronization")):
            with patch.object(wp.array, "numpy", side_effect=AssertionError("unexpected device read")):
                selected = solver.get_path_selection()
        self.assertIn("CPU", selected["reasons"]["sparse_mass_matrix"])
        self.assertIn("CPU", selected["reasons"]["cached_dynamics"])
        self.assertIn("CPU", selected["reasons"]["kernel_plan_by_dof"][1]["cholesky_kernel"])
        selected["reasons"]["kernel_plan_by_dof"][1]["cholesky_kernel"] = "edited"
        self.assertNotEqual(solver.get_path_selection()["reasons"], selected["reasons"])

    def test_private_override_validation_and_cpu_precedence(self):
        """Validate effective overrides while preserving explicit choices and CPU fallbacks."""
        model = _cpu_model()
        with patch.dict(SolverFeatherPGS._kernel_overrides, {"cholesky_kernel": "invalid"}):
            with self.assertRaisesRegex(ValueError, "cholesky_kernel"):
                SolverFeatherPGS(model)
            explicit = SolverFeatherPGS(model, cholesky_kernel="loop").get_path_selection()
            self.assertIn("Explicit", explicit["reasons"]["kernel_plan_by_dof"][1]["cholesky_kernel"])
        with patch.dict(SolverFeatherPGS._kernel_overrides, {"cholesky_kernel": "tiled"}):
            solver = SolverFeatherPGS(model)
            self.assertEqual(solver.cholesky_kernel, "loop")
            reason = solver.get_path_selection()["reasons"]["kernel_plan_by_dof"][1]["cholesky_kernel"]
            self.assertIn("CPU", reason)
            self.assertIn("override", reason)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_selection_reasons_distinguish_policy_capacity_and_override(self):
        """Explain the actual sparse rejection while retaining the dense fallback."""
        model = _build_model()
        cases = (
            ({"sparse_mass_matrix": "off"}, "Explicit"),
            ({"trisolve_kernel": "loop"}, "explicit dense"),
            ({"dense_max_constraints": 2048}, "shared memory"),
        )
        for options, expected in cases:
            with self.subTest(options=options):
                selected = _make_solver(model, True, **options).get_path_selection()
                self.assertEqual(selected["sparse_mass_matrix"], "off")
                self.assertIn(expected, selected["reasons"]["sparse_mass_matrix"])
        selected = _make_solver(model, True).get_path_selection()
        self.assertEqual(selected["sparse_mass_matrix"], "on")
        self.assertIn("topology", selected["reasons"]["sparse_mass_matrix"])

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cache_selection_reports_environment_and_explicit_precedence(self):
        """Retain construction-time cache decisions after environment overrides change."""
        model = _build_model()
        with patch.dict(os.environ, {"FEATHER_PGS_TREE_KINETIC_STATE": "0"}):
            disabled = _make_solver(model, True)
            forced = _make_solver(model, True, cached_dynamics="on")
        self.assertEqual(disabled.get_path_selection()["cached_dynamics"], "off")
        self.assertIn("FEATHER_PGS_TREE_KINETIC_STATE", disabled.get_path_selection()["reasons"]["cached_dynamics"])
        self.assertEqual(forced.get_path_selection()["cached_dynamics"], "on")
        self.assertIn("Explicit", forced.get_path_selection()["reasons"]["cached_dynamics"])

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_auto_composes_existing_execution_choices(self):
        """Select qualified split, velocity-limit and grouped paths without private switches."""
        model = _build_model()
        flags = (
            "FEATHER_PGS_SPARSE_SPLIT",
            "FEATHER_PGS_SPARSE_VELOCITY_LIMITS",
            "FEATHER_PGS_SPARSE_GROUPED_DYNAMICS",
        )
        cases = (
            {"pgs_mode": "split"},
            {"enable_joint_velocity_limits": True},
            {"grouped_dynamics": True},
        )
        with patch.dict(os.environ):
            for name in flags:
                os.environ.pop(name, None)
            for options in cases:
                with self.subTest(options=options):
                    solver = _make_solver(model, True, friction_anchor_beta=0.0, **options)
                    self.assertEqual(solver.get_path_selection()["sparse_mass_matrix"], "on")
                    disabled = _make_solver(model, True, sparse_mass_matrix="off", friction_anchor_beta=0.0, **options)
                    self.assertEqual(disabled.get_path_selection()["sparse_mass_matrix"], "off")
            for name in flags:
                os.environ[name] = "0"
            for options in cases:
                with self.subTest(legacy_disabled=options):
                    ordinary = _make_solver(model, True, friction_anchor_beta=0.0, **options)
                    self.assertEqual(ordinary.get_path_selection()["sparse_mass_matrix"], "off")
                    forced = _make_solver(model, True, sparse_mass_matrix="on", friction_anchor_beta=0.0, **options)
                    self.assertEqual(forced.get_path_selection()["sparse_mass_matrix"], "on")

    def test_cached_dynamics_cpu_choices(self):
        """Keep CPU auto ordinary and reject a forced unsupported cache."""
        model = _cpu_model()
        for choice in ("auto", "off"):
            selected = SolverFeatherPGS(model, cached_dynamics=choice).get_path_selection()
            self.assertEqual(selected["requested"]["cached_dynamics"], choice)
            self.assertEqual(selected["cached_dynamics"], "off")
            self.assertEqual(selected["publication"], "ordinary")
        for choice in ("on", "typo"):
            with self.subTest(choice=choice), self.assertRaisesRegex(ValueError, "cached_dynamics"):
                SolverFeatherPGS(model, cached_dynamics=choice)

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_overrides_environment_and_dense_conflicts(self):
        """Honor explicit cache choices and retain automatic dense kernel requests."""
        model = _build_model()
        for choice, legacy, expected in (("on", "0", True), ("off", "1", False), ("auto", "0", False)):
            with self.subTest(choice=choice), patch.dict(os.environ, {"FEATHER_PGS_TREE_KINETIC_STATE": legacy}):
                solver = _make_solver(model, True, cached_dynamics=choice)
                selected = solver.get_path_selection()
                self.assertEqual(solver._tree_kinetic_state is not None, expected)
                self.assertEqual(selected["cached_dynamics"], "on" if expected else "off")
                self.assertEqual(selected["publication"], "cached" if expected else "ordinary")
        with self.assertRaisesRegex(ValueError, "cached_dynamics.*conflicts"):
            _make_solver(model, False, cached_dynamics="on", trisolve_kernel="loop")
        solver = _make_solver(model, False, cached_dynamics="auto", trisolve_kernel="loop")
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "off")

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_mixed_layout_is_opt_in_and_unsupported_roots_reject(self):
        """A fixed arm with a free body is cached only on request; a revolute world root is rejected."""
        world = newton.ModelBuilder()
        joints = []
        parent = -1
        for index in range(5):
            body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            if index in (0, 4):
                joint = world.add_joint_fixed(parent, body)
            else:
                joint = world.add_joint_revolute(parent, body, armature=0.01)
            joints.append(joint)
            parent = body
        world.add_articulation(joints)
        body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        world.add_articulation([world.add_joint_free(body)])
        builder = newton.ModelBuilder()
        builder.replicate(world, 2)
        model = builder.finalize(device="cuda:0")
        solver = _make_solver(model, False, cached_dynamics="auto", friction_anchor_beta=0.0)
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "off")
        # The generic mixed-tree owner now covers fixed arms with free rigid bodies on request.
        solver = _make_solver(model, False, cached_dynamics="on", friction_anchor_beta=0.0)
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "on")
        self.assertEqual(solver.get_path_selection()["cached_prediction"], "tree")
        # Cached trees require a fixed or free root; a revolute joint at the world is rejected.
        world = newton.ModelBuilder()
        joints, parent = [], -1
        for _ in range(3):
            body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joints.append(world.add_joint_revolute(parent, body, armature=0.01))
            parent = body
        world.add_articulation(joints)
        body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        world.add_articulation([world.add_joint_free(body)])
        builder = newton.ModelBuilder()
        builder.replicate(world, 2)
        model = builder.finalize(device="cuda:0")
        with self.assertRaisesRegex(ValueError, "cached_dynamics"):
            _make_solver(model, False, cached_dynamics="on", friction_anchor_beta=0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_contact_graph_and_reset(self):
        """Preserve contacting states through graph replay and a partial-world reset."""
        model = _build_model()
        outputs = []
        for choice in ("off", "on"):
            solver = _make_solver(model, True, cached_dynamics=choice)
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
            solver.reset(state, world_mask=wp.array([True, False], dtype=wp.bool, device=model.device))
            for _ in range(8):
                wp.capture_launch(capture.graph)
            solver.check_constraint_capacity()
            self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            outputs.append(
                {name: getattr(state, name).numpy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
            )
        for name in outputs[0]:
            np.testing.assert_allclose(outputs[0][name], outputs[1][name], atol=3e-4, rtol=3e-4, err_msg=name)

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

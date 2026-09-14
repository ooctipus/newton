# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check register retention against the existing current/held physical fixtures."""

# Keep native imports lazy for the regression-first and selected-device gates.
# ruff: noqa: PLC0415

import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton

HELPER = Path(
    "/home/octi/Projects/newton-fpgs-anymal-contiguous-response-20260914/tools/fpgs_bench/test_branch_response.py"
)
HELPER_SHA = "461563b3252bf42ffacaccbb66c479ea19d7504b57ad7e9f97fe1840f2d217a1"


def helpers():
    """Reuse the pinned current-J/held-L and uncancelled momentum evaluators."""
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError("Preserved ANYmal physical helper changed")
    spec = importlib.util.spec_from_file_location("register_gram_physical_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def factory_arguments(tier):
    return {
        "rows": tier,
        "min_rows": 0 if tier == 32 else 32,
        "sweeps": 24,
        "matrix_free": True,
        "inkernel_response": (18, 0, 0, 0),
        "exact_row_sums": True,
        "world_rows": True,
    }


def launch_saved(a, scalar, tier, device, retained):
    """Replay the actual original ABI; no response/factor conversion is inserted."""
    from newton._src.solvers.feather_pgs import register_gram
    from newton._src.solvers.feather_pgs import solver_feather_pgs as original

    factory = original._get_pgs_solve_parallel_kernel
    if retained:
        factory = register_gram.get_parallel_factory(factory)
    kernel = factory(
        a["world_impulses"].shape[1],
        a["mf_impulses"].shape[1],
        18,
        wp.get_device(device).arch,
        **factory_arguments(tier),
    )
    assert bool(getattr(kernel, "_fpgs_register_gram", False)) == retained
    buffers, args = {}, []
    for arg in kernel.adj.args:
        name = arg.label
        if name in a:
            buffers[name] = wp.array(a[name], dtype=arg.type.dtype, device=device)
            args.append(buffers[name])
        else:
            args.append(scalar[name])
    wp.launch_tiled(
        kernel, dim=[len(a["world_constraint_count"])], inputs=args, block_dim=kernel._fpgs_block_dim, device=device
    )
    return {
        name: buffers[name].numpy()
        for name in ("v_out", "world_impulses", "world_row_type", "world_row_parent", "world_row_mu")
    }


def dense_reference(problem):
    """Independent FP64 A*y recurrence with the unchanged full-row24 projection."""
    z = problem["Z"].astype(np.float64)
    a = z @ z.T
    diagonal = np.diag(a)
    eta = np.minimum(diagonal / np.sum(np.abs(a), axis=1), 1) / (diagonal + problem["cfm"])
    normal = problem["kind"] != 2
    x = np.zeros(len(z))
    y = x.copy()
    t = 1.0
    for _ in range(24):
        value = y - eta * (problem["b"] + a @ y)
        value[normal] = np.maximum(value[normal], 0)
        for p in np.flatnonzero(problem["kind"] == 0):
            pair = np.flatnonzero((problem["kind"] == 2) & (problem["parent"] == p))
            if not len(pair):
                continue
            assert len(pair) == 2
            radius = max(0, problem["mu"][pair[0]] * value[p])
            length = np.linalg.norm(value[pair])
            if radius == 0:
                value[pair] = 0
            elif length > radius:
                value[pair] *= radius / length
        if np.dot(y - value, value - x) > 0:
            t = 1.0
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        changed = np.any(np.abs(value - x) > 1e-4 * (np.abs(value) + 1e-4))
        y = value + ((t - 1) / t_next) * (value - x)
        x, t = value, t_next
        if not changed:
            break
    velocity = problem["predictor"] + np.linalg.solve(problem["L"].T, z.T @ x)
    return x, velocity


class TestRegisterGramCPU(unittest.TestCase):
    def test_factory_api(self):
        """Require the opt-in factory without changing the original kernel ABI."""
        from newton._src.solvers.feather_pgs import register_gram
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        self.assertTrue(callable(register_gram.get_parallel_factory(original._get_pgs_solve_parallel_kernel)))

    def test_native_owner_and_unsupported_factory(self):
        """Both ordinary tiers retain Gram registers; non-admitted ABIs stay original."""
        from newton._src.solvers.feather_pgs import register_gram
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING"):
            self.enterContext(patch.object(original, name, False))
        self.enterContext(patch.dict(os.environ, {"FEATHER_PGS_WR_STOP": "0"}))
        factory = register_gram.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
        for arch in (120, 100):
            for tier in (32, 48):
                kernel = factory(72, 32, 18, arch, **factory_arguments(tier))
                self.assertTrue(kernel._fpgs_register_gram)
                self.assertTrue(kernel.key.endswith("_rg1"))
                snippet = kernel._fpgs_register_gram_native
                self.assertIn(f"float rg_a{tier - 1} = 0.0f;", snippet)
                self.assertNotIn("// dv = Y^T y", snippet)
                self.assertIn("const float4* y4", snippet)
        for changed in ({"sweeps": 8}, {"matrix_free": False}, {"exact_row_sums": False}, {"world_rows": False}):
            kw = factory_arguments(32) | changed
            old = original._get_pgs_solve_parallel_kernel(72, 32, 18, 120, **kw)
            new = factory(72, 32, 18, 120, **kw)
            self.assertFalse(getattr(new, "_fpgs_register_gram", False))
            self.assertEqual(new.key, old.key)

    def test_requested_owner_admission(self):
        """Host-only constructor metadata rejects competing and warm/debug owners."""
        from newton._src.solvers.feather_pgs import register_gram
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        for name in ("_INK_ON", "_WR_ON", "_MF_EXACT_ROWSUM"):
            self.enterContext(patch.object(original, name, True))
        for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING", "_SHADOW_LEAN"):
            self.enterContext(patch.object(original, name, False))
        self.enterContext(patch.dict(os.environ, {"FEATHER_PGS_WR_STOP": "0"}))
        solver = SimpleNamespace(
            model=SimpleNamespace(device=SimpleNamespace(is_cuda=True)),
            grouped_dynamics=True,
            max_world_dofs=18,
            mf_gs_parallel_rows=48,
            mf_gs_parallel_sweeps=24,
            mf_gs_parallel_nesterov=True,
            mf_gs_parallel_matrix_free=True,
            mf_gs_response_block_rows=0,
            dense_max_constraints=72,
            friction_mode="current",
            pgs_warmstart=False,
            enable_joint_velocity_limits=False,
            fuse_joint_velocity_limits=False,
            drive_mode="implicit",
            _paired_factor_coordinates=False,
            _local_internal_fast_path=False,
        )
        register_gram.validate_solver(solver, original)
        for name, value in (
            ("max_world_dofs", 22),
            ("mf_gs_parallel_rows", 32),
            ("mf_gs_parallel_sweeps", 8),
            ("mf_gs_response_block_rows", 32),
            ("friction_mode", "delayed"),
            ("pgs_warmstart", True),
            ("enable_joint_velocity_limits", True),
            ("drive_mode", "physx_pgs"),
            ("_paired_factor_coordinates", True),
        ):
            with patch.object(solver, name, value), self.assertRaises(ValueError, msg=name):
                register_gram.validate_solver(solver, original)
        for name in ("_WR_WARM", "_INK_CHECK", "_REGISTER_WHITENING", "_SHADOW_LEAN"):
            with patch.object(original, name, True), self.assertRaises(ValueError, msg=name):
                register_gram.validate_solver(solver, original)

    def test_current_held_physical_gram_reference(self):
        """Current J and held L give the same24 law with dense or factor-space products."""
        helper = helpers()
        rows, method, physical = (helper.reference(name) for name in ("rows", "method", "physical"))
        cases = 0
        for gpu in (0, 1):
            for step in (0, 1):
                for tier in (32, 48):
                    a, saved, scalar, owned, _ = rows.load(gpu, step, tier)
                    for world in owned[:4]:
                        problem = rows.world(a, saved, scalar, int(world))
                        old = method.solve(problem, lazy=False)
                        impulse, velocity = dense_reference(problem)
                        scale = max(1.0, float(np.max(np.abs(old["velocity"]))))
                        self.assertLess(np.max(np.abs(velocity - old["velocity"])) / scale, 3e-5)
                        metrics = physical.residuals(problem, impulse, velocity)
                        self.assertLess(metrics["momentum_scaled"], 3e-5)
                        self.assertLess(metrics["disk_excess"], 3e-5)
                        # Corrupting the final physical velocity must fail this same momentum gate.
                        corrupted = velocity.copy()
                        corrupted[0] += 0.1
                        self.assertGreater(physical.residuals(problem, impulse, corrupted)["momentum_scaled"], 3e-5)
                        cases += 1
        self.assertEqual(cases, 32)

    def test_uncancelled_momentum_scale(self):
        helpers().TestBranchResponse.test_momentum_cancellation_scale(self)


@unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "").startswith("cuda"), "root-owned CUDA gate only")
class TestRegisterGramCUDA(unittest.TestCase):
    def setUp(self):
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING", "_SHADOW_LEAN"):
            self.enterContext(patch.object(original, name, False))
        self.enterContext(patch.dict(os.environ, {"FEATHER_PGS_WR_STOP": "0"}))

    def test_native_saved_current_held_both_tiers(self):
        """All captured worlds retain current-J/held-H momentum and constraint quality."""
        helper = helpers()
        rows, physical = (helper.reference(name) for name in ("rows", "physical"))
        device = os.environ["FPGS_TEST_DEVICE"]
        worst, cases, prefix = 0.0, 0, 0
        maxima = {}
        for gpu in (0, 1):
            for step in (0, 1):
                for tier in (32, 48):
                    a, saved, scalar, owned, _ = rows.load(gpu, step, tier)
                    old = launch_saved(a, scalar, tier, device, False)
                    new = launch_saved(a, scalar, tier, device, True)
                    for name in ("world_row_type", "world_row_parent", "world_row_mu"):
                        np.testing.assert_array_equal(new[name], old[name])
                    for world in owned:
                        problem = rows.world(a, saved, scalar, int(world))
                        n, ids = len(problem["J"]), a["world_dof_indices"][world]
                        v0, v1 = old["v_out"][ids], new["v_out"][ids]
                        error = float(np.max(np.abs(v1 - v0))) / max(1.0, float(np.max(np.abs(v0))))
                        worst = max(worst, error)
                        self.assertLess(error, 3e-5, (gpu, step, tier, int(world), "velocity"))
                        p0 = physical.residuals(problem, old["world_impulses"][world, :n], v0)
                        p1 = physical.residuals(problem, new["world_impulses"][world, :n], v1)
                        self.assertLess(p1["momentum_scaled"], 3e-5)
                        self.assertLess(p1["negative_normal_impulse"], 1e-7)
                        self.assertLess(p1["disk_excess"], 3e-5)
                        # Normal includes contact and joint-limit units; no universal mm/s claim.
                        for key in (
                            "natural_residual_scaled",
                            "normal_negative_velocity",
                            "complementarity",
                            "mdp_gap",
                        ):
                            self.assertLessEqual(p1[key], p0[key] + 3e-5 * max(1.0, p0[key]), (world, key, p0, p1))
                            maxima[key] = max(maxima.get(key, 0.0), p1[key])
                        cases += 1
                        prefix += int(np.count_nonzero(problem["kind"] == 3))
        self.assertGreater(prefix, 0, "Saved gate must include actual joint-limit rows")
        print("REGISTER_GRAM_SAVED", json.dumps(dict(cases=cases, limits=prefix, velocity=worst, **maxima)), flush=True)

    def test_native_warm_self_contact_and_padding(self):
        """Reuse the exact two-leg/warm-null-cycle controls without a new oracle."""
        helper = helpers()
        with patch.object(helper, "launch_saved", launch_saved):
            test = helper.TestBranchResponse.test_cuda_two_leg_self_contact_and_warm
            getattr(test, "__wrapped__", test)(self)

    def test_native_constructor_fallback_current_held_graph(self):
        """Actual two-world Solver steps retain L18, fallback, public state and reset."""
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        helper = helpers()
        device = os.environ["FPGS_TEST_DEVICE"]
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.begin_world()
            builder.add_usd(str(helper.ASSET), load_visual_shapes=False, enable_self_collisions=True)
            builder.end_world()
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        model.rigid_contact_max = 64
        q = model.joint_q.numpy().reshape(2, 19).copy()
        q[:, 2] = 0.6
        q[:, 7:] = [0, -0.4, 0.8, 0, -0.4, 0.8, 0, 0.4, -0.8, 0, 0.4, -0.8]
        model.joint_q.assign(q.reshape(-1))
        for name in ("_INK_ON", "_WR_ON", "_MF_EXACT_ROWSUM"):
            self.enterContext(patch.object(original, name, True))
        solvers = []
        for retained in (False, True):
            with patch.dict(os.environ, {"FEATHER_PGS_REGISTER_GRAM": str(int(retained))}):
                solvers.append(
                    original.SolverFeatherPGS(
                        model,
                        pgs_mode="matrix_free",
                        pgs_iterations=8,
                        mf_gs_parallel_rows=48,
                        mf_gs_parallel_sweeps=24,
                        mf_gs_parallel_matrix_free=True,
                        dense_max_constraints=72,
                        mf_max_constraints=32,
                        mf_gs_incremental_rows=0,
                        grouped_dynamics=True,
                        lazy_kinematics=True,
                        update_mass_matrix_interval=2,
                    )
                )
        for retained, solver in zip((False, True), solvers, strict=True):
            self.assertEqual(solver._register_gram, retained)
            self.assertEqual(solver.L_by_size[18].shape, (2, 18, 18))
            self.assertEqual(solver.H_by_size[18].shape, (2, 18, 18))
            self.assertEqual(len(solver._pgs_solve_mf_gs_incremental_kernels), 2)
            for kernel in solver._pgs_solve_mf_gs_incremental_kernels:
                self.assertEqual(bool(getattr(kernel, "_fpgs_register_gram", False)), retained)
        banks = [(model.state(), model.state()) for _ in solvers]
        for states in banks:
            for state in states:
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts = [newton.Contacts(64, 0, device=device) for _ in solvers]
        shape_bodies, child = model.shape_body.numpy(), model.joint_child.numpy()
        ground = int(np.flatnonzero(shape_bodies < 0)[0])
        poses = banks[0][0].body_q.numpy()
        shape0 = np.full(64, -1, np.int32)
        shape1 = shape0.copy()
        point0 = np.zeros((64, 3), np.float32)
        point1, normal = point0.copy(), point0.copy()
        cursor = 0
        for world, total in enumerate((4, 18)):
            feet = [int(child[17 * world + j]) for j in (4, 8, 12, 16)]
            shapes = [int(np.flatnonzero(shape_bodies == body)[0]) for body in feet]
            for i in range(total):
                shape = shapes[i % 4]
                body = shape_bodies[shape]
                shape0[cursor], shape1[cursor] = shape, ground
                point0[cursor] = [(i // 4) * 0.002, 0, 0]
                current = np.asarray(wp.transform_point(wp.transform(*poses[body]), wp.vec3(*point0[cursor])))
                point1[cursor] = current + np.array([0, 0, 0.003])
                normal[cursor] = [0, 0, -1]
                cursor += 1
        for contact in contacts:
            contact.rigid_contact_count.assign(np.array([cursor], np.int32))
            for field, value in (
                ("shape0", shape0),
                ("shape1", shape1),
                ("point0", point0),
                ("point1", point1),
                ("normal", normal),
            ):
                getattr(contact, "rigid_contact_" + field).assign(value)
            contact.rigid_contact_margin0.zero_()
            contact.rigid_contact_margin1.zero_()
        controls = [model.control() for _ in solvers]

        def one(which, flip):
            source, target = banks[which][flip], banks[which][1 - flip]
            source.clear_forces()
            solvers[which].step(source, target, controls[which], contacts[which], 0.0025)
            solvers[which].publish_kinematics(target)

        def compare(flip):
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                old = getattr(banks[0][flip], name).numpy()
                new = getattr(banks[1][flip], name).numpy()
                self.assertTrue(np.isfinite(new).all(), name)
                error = np.max(np.abs(new - old)) / max(1.0, float(np.max(np.abs(old))))
                self.assertLess(error, 3e-5, (name, float(error)))
            for solver in solvers:
                solver.check_constraint_capacity()

        one(0, 0)
        one(1, 0)
        compare(1)
        count = solvers[1].constraint_count.numpy()
        self.assertTrue(0 < count[0] <= 48)
        self.assertTrue(48 < count[1] <= 72)
        response = solvers[1].Y_world.numpy()[1, : count[1]]
        self.assertTrue(np.any(response != 0) and np.isfinite(response).all())
        self.assertNotEqual(solvers[1]._J_bufs[0][18].ptr, solvers[1]._J_bufs[1][18].ptr)
        before = solvers[1].L_by_size[18].numpy().copy()
        one(0, 1)
        one(1, 1)
        compare(0)
        np.testing.assert_array_equal(solvers[1].L_by_size[18].numpy(), before)
        graphs = []
        wp.synchronize_device(device)
        for which in range(2):
            with wp.ScopedCapture(device=device) as capture:
                solvers[which].seed_double_buffer_events()
                one(which, 0)
                one(which, 1)
            graphs.append(capture.graph)
        # The same captured buffers cross empty, small and >48 fallback boundaries.
        for active in (cursor, 0, 4, cursor):
            for contact in contacts:
                contact.rigid_contact_count.assign(np.array([active], np.int32))
            for graph in graphs:
                wp.capture_launch(graph)
            compare(0)
        for which, solver in enumerate(solvers):
            solver.reset(banks[which][0], world_mask=wp.array([True, False], dtype=wp.bool, device=device))
        one(0, 0)
        one(1, 0)
        compare(1)
        for solver, contact in zip(solvers, contacts, strict=True):
            solver.update_contacts(contact)
            self.assertTrue(np.isfinite(contact.rigid_contact_force.numpy()).all())
        print("REGISTER_GRAM_LIFECYCLE", {"count": count.tolist(), "graphs": 2, "regrown": cursor}, flush=True)


if __name__ == "__main__":
    unittest.main()

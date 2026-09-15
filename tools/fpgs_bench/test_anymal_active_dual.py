# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual active-dual native owner with the preserved physical inputs."""

# Preserve lazy native imports for regression-first and root-owned CUDA gates.
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
    spec = importlib.util.spec_from_file_location("active_dual_physical_helpers", HELPER)
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


def launch_saved(a, scalar, tier, device, active, *, sweeps=24):
    """Bind the unchanged real native ABI, including an original remaining-budget control."""
    from newton._src.solvers.feather_pgs import active_dual
    from newton._src.solvers.feather_pgs import solver_feather_pgs as original

    factory = original._get_pgs_solve_parallel_kernel
    if active:
        factory = active_dual.get_parallel_factory(factory)
    kernel = factory(
        a["world_impulses"].shape[1],
        a["mf_impulses"].shape[1],
        18,
        wp.get_device(device).arch,
        **(factory_arguments(tier) | {"sweeps": sweeps}),
    )
    assert bool(getattr(kernel, "_fpgs_active_dual", False)) == active
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
    written = ("v_out", "world_impulses", "world_row_type", "world_row_parent", "world_row_mu")
    # The new solve must not modify retained L, current geometry, counts, or input RHS.
    for name, buffer in buffers.items():
        if name not in written:
            np.testing.assert_array_equal(buffer.numpy(), a[name], err_msg=name)
    return {name: buffers[name].numpy() for name in written}


def limits_case(kind):
    """Mutate one preserved raw record into a small, independently soluble safeguard case."""
    rows = helpers().reference("rows")
    arrays, _, scalar, _, _ = rows.load(0, 0, 32)
    a = {name: value.copy() for name, value in arrays.items()}
    a["world_constraint_count"][:] = 0
    a["mf_constraint_count"][:] = 0
    a["world_contact_counts"][:] = 0
    a["contact_count"][:] = 0
    a["ink_meta"].reshape(-1, 4)[0] = [0, 0, -1, 0]
    a["observer_group_to_art18"][0] = 0
    a["ink_L_a"][0] = np.eye(18, dtype=np.float32)
    a["ink_J_a"][0] = 0
    a["world_dof_indices"][0] = np.arange(18)
    a["v_out"][:18] = 0
    a["world_impulses"][0] = 0
    count = 2 if kind == "singular" else 20
    a["world_constraint_count"][0] = count
    a["world_row_type"][0, :count] = 3
    a["world_row_parent"][0, :count] = -1
    a["world_row_mu"][0, :count] = 0
    a["world_row_cfm"][0, :count] = 1e-6
    a["rhs_bias"][0, :count] = -1
    if kind == "singular":
        a["ink_J_a"][0, :2, 0] = 1
    else:
        a["ink_J_a"][0, :18, :18] = np.eye(18, dtype=np.float32)
        a["ink_J_a"][0, 18, 0] = -1
        a["ink_J_a"][0, 19, 1] = -1
        if kind == "late":
            a["rhs_bias"][0, 18:20] = 0.25
        elif kind != "initial_overflow":
            raise ValueError(kind)
    expected = {
        name: a[name].copy()
        for name in ("v_out", "world_impulses", "world_row_type", "world_row_parent", "world_row_mu")
    }
    problem = rows.world(a, expected, scalar, 0)
    return a, dict(scalar), problem


def assert_untouched_worlds(test, arrays, output, tier):
    """Tier and inactive-row ownership must hold even when the numerical algorithm changes."""
    count = arrays["world_constraint_count"]
    owned = (count > (0 if tier == 32 else 32)) & (count <= tier) & (arrays["mf_constraint_count"] == 0)
    inactive = (~owned[:, None]) | (np.arange(arrays["world_impulses"].shape[1])[None, :] >= count[:, None])
    np.testing.assert_array_equal(output["world_impulses"][inactive], arrays["world_impulses"][inactive])
    unowned_dofs = arrays["world_dof_indices"][~owned].reshape(-1)
    unowned_dofs = unowned_dofs[unowned_dofs >= 0]
    np.testing.assert_array_equal(output["v_out"][unowned_dofs], arrays["v_out"][unowned_dofs])
    test.assertTrue(np.isfinite(output["v_out"]).all())


class TestActiveDualCPU(unittest.TestCase):
    def test_factory_api(self):
        """Require the new owner without changing the original factory ABI."""
        from newton._src.solvers.feather_pgs import active_dual
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        self.assertTrue(callable(active_dual.get_parallel_factory(original._get_pgs_solve_parallel_kernel)))

    def test_native_owner_abi_and_remaining_budget(self):
        """Both real native tiers retain the ABI; late fallback cannot receive a fresh24."""
        from newton._src.solvers.feather_pgs import active_dual
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING", "_SHADOW_LEAN"):
            self.enterContext(patch.object(original, name, False))
        self.enterContext(patch.dict(os.environ, {"FEATHER_PGS_WR_STOP": "0"}))
        factory = active_dual.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
        for arch in (120, 100):
            for tier in (32, 48):
                old = original._get_pgs_solve_parallel_kernel(72, 32, 18, arch, **factory_arguments(tier))
                new = factory(72, 32, 18, arch, **factory_arguments(tier))
                self.assertTrue(new._fpgs_active_dual)
                self.assertTrue(new.key.endswith("_ad1"))
                self.assertEqual(
                    [(x.label, str(x.type)) for x in new.adj.args], [(x.label, str(x.type)) for x in old.adj.args]
                )
                native = new._fpgs_active_dual_native
                self.assertIn("ad_consumed", native)
                self.assertIn("24 - ad_consumed", native)
                self.assertIn("ad_lu_ok", native)
                self.assertIn("ad_nactive", native)
                self.assertIn("direction < 2 && !accepted", native)
                self.assertIn("trial < 8", native)
                self.assertEqual(native.count("++ad_consumed"), 1)
                self.assertLess(native.index("const float projected_direction"), native.index("for (int direction"))
                self.assertLess(native.index("if (!accepted) break;"), native.index("++ad_consumed"))
        for changed in ({"sweeps": 8}, {"matrix_free": False}, {"exact_row_sums": False}, {"world_rows": False}):
            kw = factory_arguments(32) | changed
            old = original._get_pgs_solve_parallel_kernel(72, 32, 18, 120, **kw)
            new = factory(72, 32, 18, 120, **kw)
            self.assertFalse(getattr(new, "_fpgs_active_dual", False))
            self.assertEqual(new.key, old.key)

    def test_requested_owner_admission(self):
        """Require the original ANYmal topology/budget and reject competing or warm owners."""
        from newton._src.solvers.feather_pgs import active_dual
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
            pgs_debug=False,
            enable_joint_velocity_limits=False,
            fuse_joint_velocity_limits=False,
            drive_mode="implicit",
            _paired_factor_coordinates=False,
            _local_internal_fast_path=False,
        )
        active_dual.validate_solver(solver, original)
        for name, value in (
            ("max_world_dofs", 22),
            ("mf_gs_parallel_rows", 32),
            ("mf_gs_parallel_sweeps", 8),
            ("mf_gs_response_block_rows", 32),
            ("friction_mode", "delayed"),
            ("pgs_warmstart", True),
            ("enable_joint_velocity_limits", True),
            ("pgs_debug", True),
            ("drive_mode", "physx_pgs"),
            ("_paired_factor_coordinates", True),
        ):
            with patch.object(solver, name, value), self.assertRaises(ValueError, msg=name):
                active_dual.validate_solver(solver, original)
        for name in ("_WR_WARM", "_INK_CHECK", "_REGISTER_WHITENING", "_SHADOW_LEAN"):
            with patch.object(original, name, True), self.assertRaises(ValueError, msg=name):
                active_dual.validate_solver(solver, original)

    def test_safeguard_raw_record_algebra(self):
        """Prove singularity and a strictly late18→20 crossing before checking native behavior."""
        _, _, singular = limits_case("singular")
        self.assertEqual(np.linalg.matrix_rank(singular["A"]), 1)
        self.assertEqual(len(singular["J"]), 2)
        # The full diagonal projected action nearly oscillates and fails Armijo;
        # its half step decreases the same merit and satisfies physical quality.
        eta = 1 / (np.diag(singular["A"]) + singular["cfm"])
        direction = np.maximum(0, -eta * singular["b"])
        merit0 = np.linalg.norm(direction / eta)
        for alpha, accepted in ((1.0, False), (0.5, True)):
            value = alpha * direction
            residual = singular["b"] + singular["A"] @ value
            projected = np.maximum(0, value - eta * residual)
            merit = np.linalg.norm((value - projected) / eta)
            self.assertEqual(merit < (1 - 1e-4 * alpha) * merit0, accepted)
            if accepted:
                self.assertLess(np.max(np.abs(residual)), 3e-5)
                self.assertLess(np.max(np.abs(value * residual)), 3e-5)
        _, _, late = limits_case("late")
        active0 = late["b"] < 0
        self.assertEqual(int(active0.sum()), 18)
        np.testing.assert_array_equal(late["A"][:18, :18], np.eye(18))
        step = np.r_[np.ones(18), np.zeros(2)]
        residual = late["b"] + late["A"] @ step
        projected = np.maximum(0, step - residual / (np.diag(late["A"]) + late["cfm"]))
        self.assertEqual(int(np.count_nonzero(projected > 0)), 20)
        self.assertLess(np.linalg.norm(step - projected), np.linalg.norm(np.maximum(0, -late["b"])))
        self.assertLess(residual[18], 0)
        # Opposing bounds x>=1 and x<=.25 are genuinely inconsistent, not a convergence fixture.
        self.assertEqual(float(residual[18]), -0.75)

    def test_uncancelled_momentum_scale(self):
        helpers().TestBranchResponse.test_momentum_cancellation_scale(self)


@unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "").startswith("cuda"), "root-owned CUDA gate only")
class TestActiveDualCUDA(unittest.TestCase):
    def setUp(self):
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING", "_SHADOW_LEAN"):
            self.enterContext(patch.object(original, name, False))
        self.enterContext(patch.dict(os.environ, {"FEATHER_PGS_WR_STOP": "0"}))

    def test_native_saved_current_held_both_tiers(self):
        """All2048 actual inputs retain FP64 physical quality, not the original trajectory."""
        helper = helpers()
        rows, physical = (helper.reference(name) for name in ("rows", "physical"))
        device = os.environ["FPGS_TEST_DEVICE"]
        cases, limits, maxima = 0, 0, {}
        for gpu in (0, 1):
            for step in (0, 1):
                for tier in (32, 48):
                    a, saved, scalar, owned, _ = rows.load(gpu, step, tier)
                    old = launch_saved(a, scalar, tier, device, False)
                    new = launch_saved(a, scalar, tier, device, True)
                    assert_untouched_worlds(self, a, new, tier)
                    for name in ("world_row_type", "world_row_parent", "world_row_mu"):
                        np.testing.assert_array_equal(new[name], old[name])
                    for world in owned:
                        problem = rows.world(a, saved, scalar, int(world))
                        n, ids = len(problem["J"]), a["world_dof_indices"][world]
                        impulse = new["world_impulses"][world, :n]
                        p0 = physical.residuals(problem, old["world_impulses"][world, :n], old["v_out"][ids])
                        p1 = physical.residuals(problem, impulse, new["v_out"][ids])
                        self.assertTrue(all(np.isfinite(v) for v in p1.values()), (gpu, step, tier, int(world)))
                        self.assertLess(p1["momentum_scaled"], 3e-5)
                        self.assertLess(
                            helper.momentum_backward(problem, impulse, new["v_out"][ids])["backward_scaled"], 3e-5
                        )
                        self.assertLess(p1["negative_normal_impulse"], 1e-7)
                        self.assertLess(p1["disk_excess"], 3e-5)
                        for key in (
                            "natural_residual_scaled",
                            "normal_negative_velocity",
                            "complementarity",
                            "mdp_gap",
                        ):
                            self.assertLessEqual(
                                p1[key], p0[key] + 3e-5 * max(1.0, p0[key]), (gpu, step, tier, int(world), key, p0, p1)
                            )
                            maxima[key] = max(maxima.get(key, 0.0), p1[key])
                        cases += 1
                        limits += int(np.count_nonzero(problem["kind"] == 3))
        self.assertEqual(cases, 2048)
        self.assertEqual(limits, 29)
        print("ACTIVE_DUAL_SAVED", json.dumps(dict(cases=cases, limits=limits, **maxima)), flush=True)

    def test_native_warm_self_contact_and_padding(self):
        """Nonzero incoming impulses preserve exact original fallback, including null cycles."""
        helper = helpers()
        original_result = None

        def fallback_launch(a, scalar, tier, device, active):
            nonlocal original_result
            output = launch_saved(a, scalar, tier, device, active)
            if active:
                for name, value in output.items():
                    np.testing.assert_array_equal(value, original_result[name], err_msg=name)
            else:
                original_result = output
            return output

        with patch.object(helper, "launch_saved", fallback_launch):
            test = helper.TestBranchResponse.test_cuda_two_leg_self_contact_and_warm
            getattr(test, "__wrapped__", test)(self)

    def test_native_singular_and_late_remaining_budget(self):
        """Singular LU stays physical; initial/late overflow preserve original24/remaining23."""
        helper = helpers()
        physical = helper.reference("physical")
        device = os.environ["FPGS_TEST_DEVICE"]
        for kind in ("singular", "initial_overflow", "late"):
            a, scalar, problem = limits_case(kind)
            new = launch_saved(a, scalar, 32, device, True)
            assert_untouched_worlds(self, a, new, 32)
            count = len(problem["J"])
            impulse, velocity = new["world_impulses"][0, :count], new["v_out"][:18]
            metric = physical.residuals(problem, impulse, velocity)
            self.assertLess(helper.momentum_backward(problem, impulse, velocity)["backward_scaled"], 3e-5)
            self.assertLess(metric["negative_normal_impulse"], 1e-7)
            self.assertTrue(np.isfinite(impulse).all())
            if kind == "singular":
                # A merit-tested projected retry can converge without terminal
                # fallback. Its nonunique impulses need physical quality, not
                # exact agreement with the original numerical trajectory.
                self.assertLess(metric["natural_residual_scaled"], 3e-5)
                self.assertLess(metric["normal_negative_velocity"], 3e-5)
                self.assertLess(metric["complementarity"], 3e-5)
            elif kind == "initial_overflow":
                old = launch_saved(a, scalar, 32, device, False)
                for name, value in new.items():
                    np.testing.assert_array_equal(value, old[name], err_msg=name)
            else:
                # First Newton action solves the initial18 identity rows, then the
                # two opposing limits enter. The independent old solve starts at
                # that same lambda/physical velocity and receives ONLY23 sweeps.
                seeded = {name: value.copy() for name, value in a.items()}
                seeded["world_impulses"][0, :18] = 1
                seeded["v_out"][:18] = 1
                expected = launch_saved(seeded, scalar, 32, device, False, sweeps=23)
                np.testing.assert_allclose(impulse, expected["world_impulses"][0, :count], rtol=3e-5, atol=3e-5)
                np.testing.assert_allclose(velocity, expected["v_out"][:18], rtol=3e-5, atol=3e-5)
            print("ACTIVE_DUAL_SAFEGUARD", json.dumps(dict(kind=kind, **metric)), flush=True)

    def test_native_unsupported_flags_keep_original(self):
        """Each scalar admission rejection keeps the original current-input solve."""
        rows = helpers().reference("rows")
        device = os.environ["FPGS_TEST_DEVICE"]
        arrays, _, scalar, _, _ = rows.load(0, 0, 32)
        for changed in (
            {"row_phase": 1},
            {"freeze_drive_rows": 1},
            {"friction_start_iteration": 2},
            {"iteration_offset": 1},
            {"regularize": 1},
            {"omega": 0.9},
        ):
            inputs = dict(scalar) | changed
            old = launch_saved(arrays, inputs, 32, device, False)
            new = launch_saved(arrays, inputs, 32, device, True)
            for name, value in new.items():
                np.testing.assert_array_equal(value, old[name], err_msg=f"{changed}: {name}")

    def test_native_constructor_fallback_current_held_graph(self):
        """Reuse actual two-world current/held, >48 fallback, public state, reset and graph inputs."""
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
        for active in (False, True):
            with patch.dict(os.environ, {"FEATHER_PGS_ACTIVE_DUAL": str(int(active))}):
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
        for active, solver in zip((False, True), solvers, strict=True):
            self.assertEqual(solver._active_dual, active)
            self.assertEqual(solver.L_by_size[18].shape, (2, 18, 18))
            self.assertEqual(solver.H_by_size[18].shape, (2, 18, 18))
            self.assertEqual(solver.pgs_iterations, 8)
            self.assertEqual(solver.mf_gs_parallel_sweeps, 24)
            self.assertEqual(len(solver._pgs_solve_mf_gs_incremental_kernels), 2)
            for kernel in solver._pgs_solve_mf_gs_incremental_kernels:
                self.assertEqual(bool(getattr(kernel, "_fpgs_active_dual", False)), active)
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
        public_reference = model.state()

        def one(which, flip):
            source, target = banks[which][flip], banks[which][1 - flip]
            source.clear_forces()
            solvers[which].step(source, target, controls[which], contacts[which], 0.0025)
            solvers[which].publish_kinematics(target)

        def compare(flip):
            # World0 deliberately uses a different converged numerical algorithm.
            # Require its public FK to agree with its own current joint state;
            # the independent saved-input test checks contact/momentum quality.
            for which in range(2):
                state = banks[which][flip]
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), (which, name))
                newton.eval_fk(model, state.joint_q, state.joint_qd, public_reference)
                for name in ("body_q", "body_qd"):
                    np.testing.assert_allclose(
                        getattr(state, name).numpy(),
                        getattr(public_reference, name).numpy(),
                        rtol=3e-5,
                        atol=3e-5,
                        err_msg=f"public {name}",
                    )
            # World1 stays the untouched ordinary >48 fallback, even as world0 diverges.
            for name, start in (("joint_q", 19), ("joint_qd", 18), ("body_q", 17), ("body_qd", 17)):
                old = getattr(banks[0][flip], name).numpy()[start:]
                new = getattr(banks[1][flip], name).numpy()[start:]
                np.testing.assert_allclose(new, old, rtol=3e-5, atol=3e-5, err_msg=f"fallback {name}")
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
        print("ACTIVE_DUAL_LIFECYCLE", {"count": count.tolist(), "graphs": 2, "regrown": cursor}, flush=True)


if __name__ == "__main__":
    unittest.main()

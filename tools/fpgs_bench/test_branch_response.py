# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent CPU controls for the complete opt-in branch factor owner."""

# Preserve lazy native imports for the regression-first and selected-device gates.
# ruff: noqa: PLC0415

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton

ASSET = Path(
    "/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-D/anymal_d.usd"
)
REFERENCE = Path("/tmp/fpgs-anymal-lazy-reference-zE5yUe")


def reference(name):
    """Reuse the preserved actual-input physical evaluator, test-only."""
    pins = {
        "rows": "2e0e10405acb82a9bba917b41f0afe50f7c7060a22c19845f7f04c7e029b3e06",
        "physical": "c69840664b048293c82fb6ff730cec72bacb896585880150244e95481be71d08",
        "method": "553c12619971e9cbeb950e1940797d91057950d70d789c1202861f1e6fcf485f",
    }
    path = REFERENCE / (name + ".py")
    if name in pins:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins[name]
    spec = importlib.util.spec_from_file_location("branch_reference_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packed_from_original(a):
    """Convert only in this diagnostic; live refresh never reads original L."""
    from newton._src.solvers.feather_pgs import branch_response as br

    index, row, col = br.factor_pattern()
    old = np.tril(a["ink_L_a"].astype(np.float64))
    H = old @ old.transpose(0, 2, 1)
    H = H[:, ::-1, ::-1].copy()
    pattern = (index >= 0) | (index.T >= 0)
    H[:, ~pattern] = 0.0
    factor = np.linalg.cholesky(H)
    return factor[:, row, col][:, None, :].astype(np.float32), factor, H


def launch_saved(a, scalar, tier, device, branch):
    """Bind the unchanged captured argument ABI to the selected current native factory."""
    from newton._src.solvers.feather_pgs import solver_feather_pgs as original

    capacity = a["world_impulses"].shape[1]
    kernel = original._get_pgs_solve_parallel_kernel(
        capacity,
        a["mf_impulses"].shape[1],
        18,
        wp.get_device(device).arch,
        rows=tier,
        min_rows=0 if tier == 32 else 32,
        sweeps=24,
        matrix_free=True,
        inkernel_response=(18, 0, 0, 0),
        exact_row_sums=True,
        world_rows=True,
        branch_response=branch,
    )
    buffers = {}
    args = []
    for arg in kernel.adj.args:
        name = arg.label
        if name in a:
            value = a[name]
            if branch and name == "ink_L_a":
                value = packed_from_original(a)[0]
            buffers[name] = wp.array(value, dtype=arg.type.dtype, device=device)
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


def fixture(device="cpu"):
    """Use the actual fixed-foot topology and an independent COM kinetic operator."""
    from newton._src.solvers.feather_pgs import branch_response as br
    from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS

    b = newton.ModelBuilder()
    b.begin_world()
    b.add_usd(str(ASSET), load_visual_shapes=False, enable_self_collisions=True)
    b.end_world()
    b.add_ground_plane()
    m = b.finalize(device="cpu")
    q = m.joint_q.numpy().copy()
    q[2] = 0.65
    q[7:] += np.random.default_rng(5).uniform(-0.2, 0.2, 12).astype(np.float32)
    state = m.state()
    twists = []
    for dof in range(18):
        v = np.zeros(18, np.float32)
        v[dof] = 1
        newton.eval_fk(m, wp.array(q, dtype=float, device="cpu"), wp.array(v, dtype=float, device="cpu"), state)
        twists.append(state.body_qd.numpy().copy())
    twists = np.asarray(twists, np.float64).transpose(1, 2, 0)
    poses = state.body_q.numpy().copy()
    host = br.make_plan(m)
    mass, inertia, com = m.body_mass.numpy(), m.body_inertia.numpy(), m.body_com.numpy()
    centers = np.array(
        [np.asarray(wp.transform_point(wp.transform(*p), wp.vec3(*c))) for p, c in zip(poses, com, strict=True)]
    )
    origin = centers[0].astype(float)
    S = np.zeros((18, 6), np.float64)
    Is = np.zeros((m.body_count, 6, 6), np.float64)
    H = np.zeros((18, 18), np.float64)
    for body in range(m.body_count):
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:])), float).reshape(3, 3)
        ic = rotation @ inertia[body].astype(float) @ rotation.T
        r = centers[body].astype(float) - origin
        cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        Is[body, :3, :3] = mass[body] * np.eye(3)
        Is[body, :3, 3:] = -mass[body] * cross
        Is[body, 3:, :3] = mass[body] * cross
        Is[body, 3:, 3:] = ic + mass[body] * ((r @ r) * np.eye(3) - np.outer(r, r))
        H += mass[body] * twists[body, :3].T @ twists[body, :3] + twists[body, 3:].T @ ic @ twists[body, 3:]
    child = m.joint_child.numpy()
    for dof, joint in enumerate(host["dof_joint"]):
        body = child[joint]
        angular = twists[body, 3:, dof]
        S[dof, 3:] = angular
        S[dof, :3] = twists[body, :3, dof] - np.cross(angular, centers[body] - origin)
    composite = Is.copy()
    parent = m.joint_parent.numpy()
    for joint in range(m.joint_count - 1, 0, -1):
        composite[parent[joint]] += composite[child[joint]]
    R = np.r_[np.full(6, 0.001), np.linspace(0.01, 0.04, 12)].astype(np.float32)
    K = np.linspace(0.001, 0.04, 18).astype(np.float32)
    H += np.diag(R + K)
    if device != "cpu":
        m = b.finalize(device=device)
    m.rigid_contact_max = 16
    with patch.dict(os.environ, {"FEATHER_PGS_BRANCH_RESPONSE": "0", "FEATHER_PGS_SPARSE_FACTOR": "0"}):
        solver = SolverFeatherPGS(
            m,
            pgs_mode="split",
            dense_max_constraints=72,
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=2,
            use_parallel_streams=False,
            double_buffer=False,
        )
    plan, host = br.make_plan(m, solver)
    data = br.BranchData()
    data.L = wp.empty((1, 1, 117), dtype=float, device=device)
    data.valid = wp.zeros(1, dtype=int, device=device)
    data.status = wp.zeros(1, dtype=int, device=device)
    args = [
        plan,
        data,
        wp.ones(1, dtype=int, device=device),
        wp.array(S.astype(np.float32), dtype=wp.spatial_vector, device=device),
        wp.array(composite.astype(np.float32), dtype=wp.spatial_matrix, device=device),
        wp.array(R[None], dtype=float, device=device),
        wp.array(np.arange(18, dtype=np.int32), dtype=int, device=device),
        wp.array(K, dtype=float, device=device),
        wp.array([18], dtype=int, device=device),
        wp.array(np.arange(18, dtype=np.int32), dtype=int, device=device),
        18,
        1,
    ]
    return {
        "model": m,
        "solver": solver,
        "plan": plan,
        "host": host,
        "data": data,
        "args": args,
        "H": H,
        "R": R,
        "K": K,
    }


class TestBranchResponse(unittest.TestCase):
    def test_packed_leaf_first_pattern(self):
        """The admitted root-six/four-leg-three tree has precisely 117 entries."""
        from newton._src.solvers.feather_pgs import branch_response as br

        index, row, col = br.factor_pattern()
        self.assertEqual(len(row), 117)
        self.assertEqual(index.shape, (18, 18))
        rng = np.random.default_rng(31)
        lower = np.zeros((18, 18))
        lower[row, col] = rng.normal(size=117) * 0.05
        np.fill_diagonal(lower, np.linspace(0.5, 2.0, 18))
        mass = lower @ lower.T
        np.testing.assert_allclose(np.linalg.cholesky(mass), lower, atol=1e-14)

    def test_actual_factor_predictor_and_fallback(self):
        """Direct native L117 and both actions match physical body energy, including reuse."""
        from newton._src.solvers.feather_pgs import branch_response as br

        f = fixture()
        wp.launch_tiled(br.get_refresh_kernel(), dim=[1], inputs=f["args"], block_dim=32, device="cpu")
        L = np.zeros((18, 18))
        L[f["host"]["row"], f["host"]["col"]] = f["data"].L.numpy()[0, 0]
        self.assertEqual(f["data"].valid.numpy().tolist(), [1])
        np.testing.assert_allclose(L @ L.T, f["H"][::-1, ::-1], rtol=2e-5, atol=5e-5)
        tau = np.random.default_rng(16).normal(size=18).astype(np.float32)
        out = wp.zeros(18, dtype=float, device="cpu")
        wp.launch_tiled(
            br.get_predictor_kernel(),
            dim=[1],
            inputs=[f["plan"], f["data"], wp.array(tau, dtype=float, device="cpu"), out],
            device="cpu",
            block_dim=32,
        )
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(f["H"], tau), rtol=3e-5, atol=5e-5)
        held = f["data"].L.numpy().copy()
        f["args"][2].zero_()
        f["args"][5].fill_(99.0)
        wp.launch_tiled(br.get_refresh_kernel(), dim=[1], inputs=f["args"], block_dim=32, device="cpu")
        np.testing.assert_array_equal(f["data"].L.numpy(), held)
        capacity = 72
        J = np.random.default_rng(17).normal(size=(1, capacity, 18)).astype(np.float32)
        arrays = [
            wp.array(J, dtype=float, device="cpu"),
            wp.array([60], dtype=int, device="cpu"),
            wp.zeros(1, dtype=int, device="cpu"),
            48,
            wp.zeros_like(wp.array(J, dtype=float, device="cpu")),
            wp.zeros_like(wp.array(J, dtype=float, device="cpu")),
            wp.zeros_like(wp.array(J, dtype=float, device="cpu")),
            wp.zeros((1, capacity), dtype=float, device="cpu"),
        ]
        wp.launch_tiled(
            br.get_response_kernel(capacity),
            dim=[1],
            inputs=[f["plan"], f["data"], *arrays],
            block_dim=32,
            device="cpu",
        )
        expected = np.linalg.solve(f["H"], J[0, :60].T).T
        np.testing.assert_allclose(arrays[4].numpy()[0, :60], expected, rtol=4e-5, atol=6e-5)
        np.testing.assert_array_equal(arrays[5].numpy()[0, :60], J[0, :60])
        np.testing.assert_allclose(arrays[7].numpy()[0, :60], np.sum(J[0, :60] * expected, axis=1), rtol=3e-5)
        f["args"][2].fill_(1)
        wp.launch_tiled(br.get_refresh_kernel(), dim=[1], inputs=f["args"], block_dim=32, device="cpu")
        L[f["host"]["row"], f["host"]["col"]] = f["data"].L.numpy()[0, 0]
        changed = f["H"] + np.diag(99.0 - f["R"])
        np.testing.assert_allclose(L @ L.T, changed[::-1, ::-1], rtol=2e-5, atol=5e-5)

    def test_topology_notification_guard(self):
        """Numeric changes retain admission, but a cross-leg parent change is rejected."""
        from newton._src.solvers.feather_pgs import branch_response as br

        f = fixture()
        m = f["model"]
        before = br.make_plan(m)
        m.body_mass.assign(m.body_mass.numpy() * 1.1)
        np.testing.assert_array_equal(br.make_plan(m)["dof_joint"], before["dof_joint"])
        parent = m.joint_parent.numpy().copy()
        parent[5] = m.joint_child.numpy()[3]
        m.joint_parent.assign(parent)
        with self.assertRaisesRegex(ValueError, "four independent"):
            br.make_plan(m)

    def test_complete_factory_seams(self):
        """Both cooperative tiers compile their compact source without altering projection law."""
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        for rows in (32, 48):
            kernel = original._get_pgs_solve_parallel_kernel(
                72,
                64,
                18,
                120,
                rows=rows,
                sweeps=24,
                matrix_free=True,
                inkernel_response=(18, 0, 0, 0),
                exact_row_sums=True,
                world_rows=True,
                branch_response=True,
            )
            self.assertIn("branch12_L117", kernel.key)

    def test_actual_current_held_gram(self):
        """Both recorded epochs retain physical Gram and finite24 output under reordering."""
        rows = reference("rows")
        method = reference("method")
        worst = 0.0
        for gpu in (0, 1):
            for step in (0, 1):
                for tier in (32, 48):
                    a, output, s, owned, _ = rows.load(gpu, step, tier)
                    _, factor, _ = packed_from_original(a)
                    for world in owned[:4]:
                        problem = rows.world(a, output, s, int(world))
                        group = a["ink_meta"].reshape(-1, 4)[world, 0]
                        Z = np.linalg.solve(factor[group], problem["J"][:, ::-1].T).T
                        gram = Z @ Z.T
                        error = np.max(np.abs(gram - problem["A"])) / max(1.0, np.max(np.abs(problem["A"])))
                        worst = max(worst, float(error))
                        self.assertLess(error, 3e-6)
                        alternate = dict(problem, Z=Z, L=factor[group], predictor=problem["predictor"][::-1])
                        old = method.solve(problem, lazy=False)
                        new = method.solve(alternate, lazy=False)
                        # This is the unchanged full-row24 reference, not a lazy method.
                        np.testing.assert_allclose(new["impulse"], old["impulse"], rtol=5e-5, atol=3e-6)
        print("BRANCH_HISTORICAL_GRAM", worst, flush=True)

    def test_offline_cuda_compilation(self):
        """Compile both real architecture targets without creating a CUDA context."""
        from newton._src.solvers.feather_pgs import branch_response as br
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        wp.init()
        with tempfile.TemporaryDirectory(prefix="fpgs-branch-compile-") as directory:
            for arch in (120, 103):
                kernels = [
                    (br.get_refresh_kernel(), 32),
                    (br.get_predictor_kernel(), 32),
                    (br.get_response_kernel(72), 32),
                ]
                for rows in (32, 48):
                    kernel = original._get_pgs_solve_parallel_kernel(
                        72,
                        64,
                        18,
                        arch,
                        rows=rows,
                        sweeps=24,
                        matrix_free=True,
                        inkernel_response=(18, 0, 0, 0),
                        exact_row_sums=True,
                        world_rows=True,
                        branch_response=True,
                    )
                    kernels.append((kernel, kernel._fpgs_block_dim))
                for number, (kernel, block) in enumerate(kernels):
                    folder = Path(directory) / f"sm{arch}_{number}"
                    folder.mkdir()
                    options = kernel.module.resolve_options(wp.config, block_dim=block)
                    kernel.module._compile(
                        device=None, output_dir=str(folder), output_arch=arch, use_ptx=True, options=options
                    )

    @unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "").startswith("cuda"), "root-owned CUDA gate only")
    def test_cuda_actual_current_held(self):
        """Actual native original and compact24 retain velocity, momentum, cones and residuals."""
        rows = reference("rows")
        physical = reference("physical")
        device = os.environ["FPGS_TEST_DEVICE"]
        worst_velocity = 0.0
        for gpu in (0, 1):
            for step in (0, 1):
                for tier in (32, 48):
                    a, saved, s, owned, _ = rows.load(gpu, step, tier)
                    old = launch_saved(a, s, tier, device, False)
                    new = launch_saved(a, s, tier, device, True)
                    for name in ("world_row_type", "world_row_parent", "world_row_mu"):
                        np.testing.assert_array_equal(new[name], old[name])
                    for world in owned:
                        problem = rows.world(a, saved, s, int(world))
                        n = len(problem["J"])
                        ids = a["world_dof_indices"][world]
                        v0, v1 = old["v_out"][ids], new["v_out"][ids]
                        scale = max(1.0, float(np.max(np.abs(v0))))
                        delta = float(np.max(np.abs(v1 - v0))) / scale
                        worst_velocity = max(worst_velocity, delta)
                        self.assertLess(delta, 3e-5, (gpu, step, tier, int(world), "velocity", delta))
                        p0 = physical.residuals(problem, old["world_impulses"][world, :n], v0)
                        p1 = physical.residuals(problem, new["world_impulses"][world, :n], v1)
                        self.assertLess(p1["momentum_scaled"], 3e-5)
                        self.assertLess(p1["negative_normal_impulse"], 1e-7)
                        self.assertLess(p1["disk_excess"], 3e-5)
                        for key in ("natural_residual_scaled", "normal_negative_velocity"):
                            self.assertLessEqual(p1[key], p0[key] + 3e-5 * max(1.0, p0[key]))
        print("BRANCH_CUDA_VELOCITY_MAX", worst_velocity, flush=True)

    @unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "").startswith("cuda"), "root-owned CUDA gate only")
    def test_cuda_two_leg_self_contact_and_warm(self):
        """Loaded two-leg normal/tangents and nonzero warm impulses use all twelve coordinates."""
        rows = reference("rows")
        physical = reference("physical")
        device = os.environ["FPGS_TEST_DEVICE"]
        a, _, s, _, _ = rows.load(0, 0, 32)
        a = {key: value.copy() for key, value in a.items()}
        a["world_constraint_count"][:] = 0
        a["world_constraint_count"][0] = 3
        a["mf_constraint_count"][:] = 0
        a["world_contact_counts"][:] = 0
        a["world_contact_counts"][0] = 1
        a["world_contacts"][0, 0] = 0
        a["contact_count"][0] = 1
        a["contact_slot"][0] = 0
        a["contact_slots_needed"][0] = 3
        art = int(a["observer_group_to_art18"][a["ink_meta"].reshape(-1, 4)[0, 0]])
        a["contact_art_a"][0] = a["contact_art_b"][0] = art
        bodies = a["shape_body"]
        shapes = []
        for leg in (0, 1):
            found = [
                i
                for i, b in enumerate(bodies)
                if 0 <= b < 17 and (int(a["body_response_dof_mask"][b]) & (7 << (6 + 3 * leg))) == (7 << (6 + 3 * leg))
            ]
            self.assertTrue(found)
            shapes.append(found[0])
        a["contact_shape0"][0], a["contact_shape1"][0] = shapes
        a["contact_point0"][0] = a["contact_point1"][0] = 0
        a["contact_thickness0"][0] = a["contact_thickness1"][0] = 0
        a["contact_normal"][0] = [0, 0, -1]
        a["shape_material_mu"][shapes] = 0.7
        a["shape_material_restitution"][shapes] = 0
        a["world_row_type"][0, :3] = [0, 2, 2]
        a["world_row_parent"][0, :3] = [-1, 0, 0]
        a["world_impulses"][0, :3] = [0.2, 0.05, -0.06]
        a["v_out"][:18] += np.linspace(-0.3, 0.3, 18).astype(np.float32)
        old = launch_saved(a, s, 32, device, False)
        new = launch_saved(a, s, 32, device, True)
        problem = rows.world(a, old, s, 0)
        velocity_error = float(np.max(np.abs(old["v_out"][:18] - new["v_out"][:18])))
        self.assertLess(velocity_error, 3e-5 * max(1, float(np.max(np.abs(old["v_out"][:18])))))
        # Momentum uses the applied impulse DELTA; the original warm reference is retained.
        delta = new["world_impulses"][0, :3] - a["world_impulses"][0, :3]
        metric = physical.residuals(problem, delta, new["v_out"][:18])
        self.assertLess(metric["momentum_scaled"], 3e-5)
        self.assertTrue(np.isfinite(new["v_out"]).all())
        lam = new["world_impulses"][0, :3]
        self.assertGreaterEqual(lam[0], 0)
        self.assertLessEqual(np.linalg.norm(lam[1:]), 0.7 * lam[0] + 3e-5)

    @unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "").startswith("cuda"), "root-owned CUDA gate only")
    def test_cuda_live_constructor_fallback_and_graph(self):
        """Actual Newton steps own L117, current fallback rows, held reuse and public state."""
        from newton._src.solvers.feather_pgs import solver_feather_pgs as original

        device = os.environ["FPGS_TEST_DEVICE"]
        b = newton.ModelBuilder()
        for _ in range(2):
            b.begin_world()
            b.add_usd(str(ASSET), load_visual_shapes=False, enable_self_collisions=True)
            b.end_world()
        b.add_ground_plane()
        m = b.finalize(device=device)
        m.rigid_contact_max = 64
        q = m.joint_q.numpy().reshape(2, 19).copy()
        q[:, 2] = 0.6
        q[:, 7:] = [0.0, -0.4, 0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, 0.4, -0.8]
        m.joint_q.assign(q.reshape(-1))
        solvers = []
        self.enterContext(patch.object(original, "_INK_ON", True))
        self.enterContext(patch.object(original, "_WR_ON", True))
        self.enterContext(patch.object(original, "_MF_EXACT_ROWSUM", True))
        for branch in (False, True):
            with patch.dict(
                os.environ, {"FEATHER_PGS_BRANCH_RESPONSE": str(int(branch)), "FEATHER_PGS_SPARSE_FACTOR": "0"}
            ):
                solvers.append(
                    original.SolverFeatherPGS(
                        m,
                        pgs_mode="matrix_free",
                        pgs_iterations=8,
                        mf_gs_parallel_rows=48,
                        mf_gs_parallel_matrix_free=True,
                        dense_max_constraints=72,
                        mf_max_constraints=32,
                        mf_gs_incremental_rows=0,
                        grouped_dynamics=True,
                        lazy_kinematics=True,
                        update_mass_matrix_interval=2,
                    )
                )
        owner = solvers[1]._branch_response
        self.assertIsNotNone(owner, "Actual constructor unexpectedly selected original fallback")
        self.assertIsNone(solvers[0]._branch_response)
        self.assertEqual(solvers[1].L_by_size[18].shape, (1, 1, 1))
        self.assertEqual(solvers[1].H_by_size[18].shape, (1, 1, 1))
        self.assertEqual(owner.data.L.shape, (2, 1, 117))
        banks = [(m.state(), m.state()) for _ in solvers]
        for states in banks:
            newton.eval_fk(m, states[0].joint_q, states[0].joint_qd, states[0])
            newton.eval_fk(m, states[1].joint_q, states[1].joint_qd, states[1])
        contacts = [newton.Contacts(rigid_contact_max=64, soft_contact_max=0, device=device) for _ in solvers]
        shape_bodies = m.shape_body.numpy()
        child = m.joint_child.numpy()
        ground = int(np.flatnonzero(shape_bodies < 0)[0])
        poses = banks[0][0].body_q.numpy()
        sh0 = np.full(64, -1, np.int32)
        sh1 = sh0.copy()
        p0 = np.zeros((64, 3), np.float32)
        p1 = p0.copy()
        normal = p0.copy()
        cursor = 0
        for world, total in enumerate((4, 18)):
            feet = [int(child[17 * world + j]) for j in (4, 8, 12, 16)]
            shapes = [int(np.flatnonzero(shape_bodies == body)[0]) for body in feet]
            for i in range(total):
                shape = shapes[i % 4]
                body = shape_bodies[shape]
                sh0[cursor] = shape
                sh1[cursor] = ground
                p0[cursor] = [(i // 4) * 0.002, 0, 0]
                point = np.asarray(wp.transform_point(wp.transform(*poses[body]), wp.vec3(*p0[cursor])))
                p1[cursor] = point + np.array([0, 0, 0.003])
                normal[cursor] = [0, 0, -1]
                cursor += 1
        for c in contacts:
            c.rigid_contact_count.assign(np.array([cursor], np.int32))
            for field, value in (("shape0", sh0), ("shape1", sh1), ("point0", p0), ("point1", p1), ("normal", normal)):
                getattr(c, "rigid_contact_" + field).assign(value)
            c.rigid_contact_margin0.zero_()
            c.rigid_contact_margin1.zero_()
        controls = [m.control() for _ in solvers]

        def one(which, flip):
            solver = solvers[which]
            source, target = banks[which][flip], banks[which][1 - flip]
            source.clear_forces()
            solver.step(source, target, controls[which], contacts[which], 0.0025)
            solver.publish_kinematics(target)

        def compare(flip):
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                old = getattr(banks[0][flip], name).numpy()
                new = getattr(banks[1][flip], name).numpy()
                self.assertTrue(np.isfinite(new).all(), name)
                error = np.max(np.abs(new - old)) / max(1.0, float(np.max(np.abs(old))))
                self.assertLess(error, 3e-5, (name, float(error)))
            for solver in solvers:
                solver.check_constraint_capacity()

        # First actual cold construction produces compact contact rows AND
        # >48 fallback J/Y through the original row builder, not supplied J.
        one(0, 0)
        one(1, 0)
        compare(1)
        count = solvers[1].constraint_count.numpy()
        self.assertTrue(0 < count[0] <= 48)
        self.assertTrue(48 < count[1] <= 72)
        fallback_y = solvers[1].Y_world.numpy()[1, : count[1]]
        self.assertTrue(np.any(fallback_y != 0) and np.isfinite(fallback_y).all())
        before = owner.data.L.numpy().copy()
        one(0, 1)
        one(1, 1)
        compare(0)
        np.testing.assert_array_equal(owner.data.L.numpy(), before)
        # Capture each two-call boundary from its own evolved q/qd. No cache
        # restoration or extra simulation call is inserted during replay.
        graphs = []
        for which in range(2):
            with wp.ScopedCapture(device=device) as capture:
                one(which, 0)
                one(which, 1)
            graphs.append(capture.graph)
        for _ in range(2):
            for graph in graphs:
                wp.capture_launch(graph)
            compare(0)
        mask = wp.array([1, 0], dtype=int, device=device)
        for which, solver in enumerate(solvers):
            solver.reset(banks[which][0], world_mask=mask)
        one(0, 0)
        one(1, 0)
        compare(1)
        solvers[1].update_contacts(contacts[1])
        solvers[0].update_contacts(contacts[0])
        self.assertTrue(np.isfinite(contacts[1].rigid_contact_force.numpy()).all())
        print(
            "BRANCH_LIVE_OWNER",
            {"count": count.tolist(), "valid": owner.data.valid.numpy().tolist(), "graphs": 2},
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete mixed forests without captured robot assets."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.franka_kinetic_state import FrankaKineticState, build_plan
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS, _FeatherPGSModelPlan
from tools.fpgs_bench.test_g1_kinetic_state import physical_mass, public_state

DT = 1.0 / 240.0


def model_fixture(device="cpu", *, variant=0, worlds=3, reordered=False):
    """Build one scalar tree, one dynamic root and one prescribed root per world."""
    parents = (
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
        [-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [-1, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7],
    )[variant]
    weld = 8 if variant == 0 else 4
    scene = newton.ModelBuilder()
    for kind in (2, 0, 1) if reordered else (0, 1, 2):
        bodies, joints = [], []
        for local in range(11 if kind == 0 else 1):
            inertia = np.array([[0.09, 0.003, -0.002], [0.003, 0.11, 0.001], [-0.002, 0.001, 0.14]])
            body = scene.add_link(
                mass=0.7 + 0.11 * local,
                inertia=wp.mat33(inertia),
                com=wp.vec3(0.013, -0.009, 0.017),
                lock_inertia=True,
                is_kinematic=kind == 2,
            )
            bodies.append(body)
            anchor = wp.transform(
                wp.vec3(0.08, -0.02 + kind * 0.4, 0.12),
                wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.07 * (local + 1)),
            )
            child = wp.transform(wp.vec3(0.01, -0.015, 0.02), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -0.04))
            if kind:
                joint = scene.add_joint_free(body, parent_xform=anchor, child_xform=child)
            else:
                parent = -1 if local == 0 else bodies[parents[local]]
                kwargs = {"parent": parent, "child": body, "parent_xform": anchor, "child_xform": child}
                if local in (0, weld):
                    joint = scene.add_joint_fixed(**kwargs)
                else:
                    linear = local in ((9, 10) if variant == 0 else (2, 6, 9))
                    add = scene.add_joint_prismatic if linear else scene.add_joint_revolute
                    joint = add(
                        **kwargs,
                        axis=newton.Axis.Y,
                        armature=0.04,
                        target_ke=1.2,
                        target_kd=0.15,
                        limit_lower=-0.8,
                        limit_upper=0.8,
                    )
            joints.append(joint)
        scene.add_articulation(joints)
        scene.add_shape_box(bodies[-2] if kind == 0 else bodies[0], hx=0.02, hy=0.02, hz=0.02)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(2.0, 2.0, 0.0))
    model = builder.finalize(device=device)
    model.rigid_contact_max = 32
    return model


def make_solver(model, *, enabled=False, permuted=False):
    """Keep the retained Cholesky recipe; use a response-ABI shim only on CPU."""
    original_plan = _FeatherPGSModelPlan.build
    original_groups = SolverFeatherPGS._setup_size_grouping

    def response_plan(model, kinematic_dof_mask, **kwargs):
        return original_plan(model, kinematic_dof_mask, enable_prescribed_response=True)

    def setup_groups(solver, source_model):
        original_groups(solver, source_model)
        if permuted:
            inverse = solver.art_group_idx.numpy().copy()
            for size in (9, 6):
                values = solver.group_to_art[size].numpy().copy()
                order = np.roll(values[::-1], size % len(values))
                solver.group_to_art[size].assign(order)
                inverse[order] = np.arange(len(order))
            solver.art_group_idx.assign(inverse)

    with (
        patch.dict(
            os.environ,
            {
                "FEATHER_PGS_FRANKA_KINETIC_STATE": str(int(enabled)),
                "FEATHER_PGS_LOCAL_ROW_PACKETS": "0",
                "FEATHER_PGS_SPARSE_FACTOR": "0",
                "FEATHER_PGS_WORLD_SCAN_PUBLICATION": "0",
            },
        ),
        patch.object(SolverFeatherPGS, "_setup_size_grouping", setup_groups),
        patch.dict(SolverFeatherPGS._kernel_overrides, {"sparse_mass_matrix": False}),
        patch.object(_FeatherPGSModelPlan, "build", side_effect=response_plan),
    ):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free" if model.device.is_cuda else "split",
            pgs_iterations=8,
            pgs_beta=0.05,
            pgs_cfm=1e-6,
            dense_max_constraints=192,
            mf_max_constraints=64,
            enable_joint_limits=True,
            joint_limit_activation_gap=0.1,
            enable_joint_velocity_limits=False,
            update_mass_matrix_interval=2,
            grouped_dynamics=False,
            double_buffer=True,
            use_parallel_streams=model.device.is_cuda,
            angular_damping=0.05,
            friction_anchor_beta=0.0,
        )
    if not model.device.is_cuda:
        solver._compute_world_response_dof_mapping(model)
        solver.pgs_mode = "matrix_free"
        solver._allocate_world_buffers(model)
    return solver


class TestCholeskyForestCache(unittest.TestCase):
    def test_derive_complete_scalar_topology(self):
        """Admit distinct scalar trees and coordinate laws under original factor sizes."""
        for variant in range(3):
            for reordered in (False, True):
                with self.subTest(variant=variant, reordered=reordered):
                    model = model_fixture(variant=variant, reordered=reordered)
                    solver = make_solver(model, permuted=True)
                    plan = build_plan(solver)
                    self.assertEqual(plan["dof_ids"].shape, (3, 21))
                    np.testing.assert_array_equal(np.sort(plan["body_ids"][:, :13].ravel()), np.arange(39))
                    np.testing.assert_array_equal(np.sort(plan["dof_ids"].ravel()), np.arange(63))
                    for world, arts in enumerate(plan["arts"]):
                        np.testing.assert_array_equal(solver.art_to_world.numpy()[arts], world)
                        self.assertEqual(solver.group_to_art[9].numpy()[plan["primary_group"][world]], arts[0])
                        self.assertEqual(solver.group_to_art[6].numpy()[plan["secondary_group"][world]], arts[1])
                    owner = FrankaKineticState(solver)
                    self.assertEqual(owner.data.current_valid.shape, (9,))
                    self.assertEqual(owner.data.geometry_valid.shape, (3,))

    def check_physical_forest(self, device):
        """Compare independent physical mass and ordinary inverse dynamics on every tree."""
        for variant in range(3):
            with self.subTest(variant=variant, device=device):
                model = model_fixture(device, variant=variant, reordered=variant != 0)
                model.joint_target_ke.zero_()
                model.joint_target_kd.zero_()
                solver, reference = (make_solver(model, permuted=True) for _ in range(2))
                owner = FrankaKineticState(solver)
                plan = owner.host_plan
                state, output, control = model.state(), model.state(), model.control()
                rng = np.random.default_rng(945)
                q = state.joint_q.numpy().copy()
                q[plan["q_index"]] = rng.uniform(-0.15, 0.15, (model.world_count, 9))
                state.joint_q.assign(q)
                raw = rng.uniform(-0.6, 0.6, model.joint_dof_count).astype(np.float32)
                state.joint_qd.assign(raw)
                state.body_f.assign(rng.normal(0, 0.2, (model.body_count, 6)).astype(np.float32))
                control.joint_f.assign(rng.uniform(-0.2, 0.2, model.joint_dof_count).astype(np.float32))
                for item in (solver, reference):
                    item._last_step_dt = DT
                    item._passive_spring_stiffness.fill_(0.31)
                    item._passive_spring_ref.fill_(-0.07)
                    item._passive_joint_damping.fill_(0.12)
                reference._fk_id_cache_enabled = False
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                augmented = solver._prepare_augmented_state(state, output, control)
                ref_aug = reference._prepare_augmented_state(state, model.state(), control)
                reference._stage1_fk_id(state, ref_aug, model.state())
                reference._stage1_drives(state, ref_aug, control, DT)
                rhs = ref_aug.joint_tau.numpy().astype(float)
                physical = physical_mass(model, state)
                owner.begin(state, augmented, DT, True)
                # Primary twists remain private scratch; only the free-root
                # finalizer owns body_v_s on this retained compact path.
                free_bodies = plan["body_ids"][:, 11:13]
                np.testing.assert_allclose(
                    solver.body_v_s.numpy()[free_bodies], reference.body_v_s.numpy()[free_bodies], atol=4e-6, rtol=3e-5
                )
                np.testing.assert_allclose(solver.joint_S_s.numpy(), reference.joint_S_s.numpy(), atol=4e-6, rtol=3e-5)
                public_state(self, model, state)
                geometry = owner.geometric.numpy().reshape(-1, 9, 9)
                lower = {size: np.zeros_like(solver.L_by_size[size].numpy()) for size in (9, 6)}
                expected_qdd = np.zeros(model.joint_dof_count)
                u0 = rng.uniform(-0.1, 0.1, model.joint_dof_count).astype(np.float32)
                for world, dofs in enumerate(plan["dof_ids"]):
                    np.testing.assert_allclose(
                        geometry[plan["primary_group"][world]],
                        physical[np.ix_(dofs[:9], dofs[:9])],
                        atol=4e-6,
                        rtol=4e-5,
                    )
                    for size, first, groups in ((9, 0, plan["primary_group"]), (6, 9, plan["secondary_group"])):
                        ids, group = dofs[first : first + size], groups[world]
                        block = physical[np.ix_(ids, ids)] + np.diag(solver.R_by_size[size].numpy()[group])
                        lower[size][group] = np.linalg.cholesky(block)
                        expected_qdd[ids] = np.linalg.solve(block, rhs[ids] + u0[ids])
                for size in (9, 6):
                    solver.L_by_size[size].assign(lower[size])
                solver.joint_tau.assign(u0)
                owner.predict(state, augmented, control, state.joint_qd, DT)
                np.testing.assert_allclose(solver.joint_qdd.numpy(), expected_qdd, atol=5e-5, rtol=5e-4)
                expected_velocity = raw.astype(float) + DT * expected_qdd
                for dofs in plan["dof_ids"]:
                    expected_velocity[dofs[9:12]] += DT * np.cross(raw[dofs[12:15]], raw[dofs[9:12]])
                np.testing.assert_allclose(solver.v_hat.numpy(), expected_velocity, atol=4e-6, rtol=3e-5)
                np.testing.assert_array_equal(solver.joint_qdd.numpy()[plan["dof_ids"][:, 15:]], 0)
                solver.v_out.assign(solver.v_hat)
                output.body_q.fill_(wp.transform(wp.vec3(123), wp.quat_identity()))
                output.body_qd.fill_(wp.spatial_vector(123))
                owner.finish(state, augmented, output, DT, False)
                public_state(self, model, output)
                np.testing.assert_array_equal(
                    output.joint_qd.numpy()[plan["dof_ids"][:, 15:]], raw[plan["dof_ids"][:, 15:]]
                )
                generation = owner.data.generation.numpy().copy()
                owner.begin(output, augmented, DT, False)
                np.testing.assert_array_equal(owner.data.generation.numpy(), generation)
                mask = wp.array([False, True, False], dtype=wp.bool, device=device)
                owner.invalidate(mask)
                np.testing.assert_array_equal(owner.data.current_valid.numpy(), solver.art_to_world.numpy() != 1)
                owner.begin(output, augmented, DT, False)
                np.testing.assert_array_equal(owner.data.generation.numpy() - generation, [0, 1, 0])
                generation = owner.data.generation.numpy().copy()
                requests = np.zeros(model.articulation_count, np.int32)
                requests[plan["arts"][2, 2]] = 1
                solver._mass_update_requested.assign(requests)
                owner.begin(output, augmented, DT, False)
                np.testing.assert_array_equal(owner.data.generation.numpy() - generation, [0, 0, 1])
                np.testing.assert_array_equal(owner.geometry_valid.numpy(), [0, 0, 1])
                solver._mass_update_requested.zero_()
                generation = owner.data.generation.numpy().copy()
                valid = owner.data.current_valid.numpy().copy()
                valid[plan["arts"][0, 1]] = 0
                owner.data.current_valid.assign(valid)
                owner.begin(output, augmented, DT, False)
                np.testing.assert_array_equal(owner.data.generation.numpy() - generation, [1, 0, 0])
                np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)
                bank = model.state()
                for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
                    wp.copy(getattr(bank, name), getattr(output, name))
                generation = owner.data.generation.numpy().copy()
                owner.begin(bank, augmented, DT, False)
                np.testing.assert_array_equal(owner.data.generation.numpy(), generation + 1)
                if model.device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        owner.invalidate(mask)
                        owner.begin(bank, augmented, DT, False)
                        owner.predict(bank, augmented, control, bank.joint_qd, DT)
                    generation = owner.data.generation.numpy().copy()
                    for selected in ([True, False, False], [False, True, True], [False] * 3):
                        mask.assign(np.array(selected))
                        wp.capture_launch(capture.graph)
                        generation += np.array(selected, np.int64)
                        np.testing.assert_array_equal(owner.data.generation.numpy(), generation)
                        np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)
                for size in (9, 6):
                    np.testing.assert_array_equal(solver.L_by_size[size].numpy(), lower[size])
                np.testing.assert_array_equal(state.joint_qd.numpy(), raw)
                owner.check()

    def test_physical_forest_cpu(self):
        """Verify physical operators, forces and complete mixed publication on CPU."""
        self.check_physical_forest("cpu")

    def test_compact_workspace_cpu(self):
        """Retain the compact scratch adapter for alternate scalar parent trees."""
        with patch.dict(os.environ, {"FEATHER_PGS_FRANKA_COMPACT_WORKSPACE": "1"}):
            self.check_physical_forest("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_physical_forest_cuda(self):
        """Verify both half-warps and the odd-world tail with distinct tree shapes."""
        self.check_physical_forest("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_compact_workspace_cuda(self):
        """Preserve compact half-warp storage and captured mixed-forest resets."""
        with patch.dict(os.environ, {"FEATHER_PGS_FRANKA_COMPACT_WORKSPACE": "1"}):
            self.check_physical_forest("cuda:0")

    def check_reset_ownership_and_rejection(self, device):
        """Cover every world mask and reject changed ownership before cache mutation."""
        model = model_fixture(device, variant=1, reordered=True)
        solver = make_solver(model, permuted=True)
        owner = FrankaKineticState(solver)
        for values in (
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, False, True],
            [True] * 3,
            [False] * 3,
            None,
        ):
            owner.data.current_valid.fill_(1)
            owner.geometry_valid.fill_(1)
            mask = None if values is None else wp.array(values, dtype=wp.bool, device=model.device)
            owner.invalidate(mask)
            selected = np.ones(3, bool) if values is None else np.array(values)
            np.testing.assert_array_equal(owner.data.current_valid.numpy(), ~selected[solver.art_to_world.numpy()])
            np.testing.assert_array_equal(owner.geometry_valid.numpy(), ~selected)
        owner.data.current_valid.fill_(1)
        owner.geometry_valid.fill_(1)
        for field in (solver.group_to_art[9], solver.art_to_world, model.joint_parent):
            old = field.numpy().copy()
            changed = old.copy()
            changed.flat[0] += 1
            field.assign(changed)
            with self.assertRaises((RuntimeError, ValueError)):
                owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)
            np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)
            np.testing.assert_array_equal(owner.geometry_valid.numpy(), 1)
            field.assign(old)
        state, control = model.state(), model.control()
        augmented = solver._prepare_augmented_state(state, model.state(), control)
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            owner.finish(state, augmented, state, DT, False)
        np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)
        np.testing.assert_array_equal(owner.geometry_valid.numpy(), 1)

    def test_reset_ownership_and_rejection(self):
        """Reject structural edits and overlapping state banks on the host path."""
        self.check_reset_ownership_and_rejection("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_device_proof_rejects_before_mutation(self):
        """Exercise the exact device proof before any mixed-cache invalidation."""
        self.check_reset_ownership_and_rejection("cuda:0")

    def test_locked_root_and_live_numeric_bindings(self):
        """Preserve fixed-root arithmetic and rebind replaced numeric model arrays."""
        from tools.fpgs_bench.test_tree_locked_root import locked_roots  # noqa: PLC0415

        with locked_roots():
            model = model_fixture(variant=2, reordered=True)
        solver = make_solver(model, permuted=True)
        owner = FrankaKineticState(solver)
        state, output, control = model.state(), model.state(), model.control()
        state.joint_qd.assign(np.linspace(-0.3, 0.4, model.joint_dof_count, dtype=np.float32))
        augmented = solver._prepare_augmented_state(state, output, control)
        owner.begin(state, augmented, DT, True)
        expected = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, expected)
        np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(state.body_qd.numpy(), expected.body_qd.numpy(), atol=4e-6, rtol=3e-5)
        original = owner.geometric.numpy().copy()
        model.body_mass = wp.array(model.body_mass.numpy() * 1.1, dtype=float, device=model.device)
        model.body_inertia = wp.array(model.body_inertia.numpy() * 1.1, dtype=wp.mat33, device=model.device)
        owner.validate_model(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        owner.invalidate()
        owner.begin(state, augmented, DT, True)
        np.testing.assert_allclose(owner.geometric.numpy(), original * 1.1, atol=3e-6, rtol=3e-5)
        data = owner._publication(state, augmented, output, DT)
        self.assertEqual(data.body_mass.ptr, model.body_mass.ptr)
        self.assertEqual(data.body_inertia.ptr, model.body_inertia.ptr)
        owner.check()


if __name__ == "__main__":
    unittest.main()

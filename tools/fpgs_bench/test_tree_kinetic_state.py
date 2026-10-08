# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check generated state caches against physical trees outside the G1 layout."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseMassMatrixPlan
from newton._src.solvers.feather_pgs.tree_chain_scan import make_plan
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton._src.solvers.feather_pgs.tree_kinetic_owner import TreeKineticState
from newton._src.solvers.feather_pgs.tree_kinetic_plan import cooperative_block_dim
from newton._src.solvers.feather_pgs.tree_kinetic_sparse import TreeSparseOperator
from newton.tests import test_feather_pgs_sparse_split as split
from tools.fpgs_bench.test_g1_kinetic_state import physical_mass, public_state


class TestTreeLayout(unittest.TestCase):
    def test_complete_cooperative_lane_coverage(self):
        """Keep every body and padded chain lane inside the selected block."""

        small = TreeLayout(7, 11, 60, 11, 3, 6, tuple(range(11)))
        self.assertEqual(cooperative_block_dim(small), 32)
        self.assertEqual(cooperative_block_dim(small, make_plan([-1, 0, 0, 1, 2, 1, 4])), 32)
        parents = [-1]
        for _ in range(10):
            parent = 0
            for _ in range(3):
                parents.append(parent)
                parent = len(parents) - 1
        padded = TreeLayout(31, 36, 666, 36, 3, 6, tuple(range(36)))
        self.assertEqual(cooperative_block_dim(padded), 32)
        self.assertEqual(cooperative_block_dim(padded, make_plan(parents)), 64)
        large = TreeLayout(44, 43, 434, 18, 10, 6, tuple(range(43)))
        self.assertEqual(cooperative_block_dim(large), 64)

    def test_root_only_and_large_tree_scratch_do_not_alias_live_axes(self):
        """Keep the temporary projected inertia clear of live joint screws."""
        for bodies, dofs in ((1, 6), (7, 12), (44, 43), (59, 64)):
            layout = TreeLayout(bodies, dofs, dofs * (dofs + 1) // 2, dofs, min(5, bodies - 1), 6, tuple(range(dofs)))
            self.assertGreaterEqual(layout.axis_offset, 6 * dofs)
            self.assertGreaterEqual(layout.wrench_offset, layout.axis_offset + 6 * dofs)
            self.assertGreaterEqual(layout.moment_offset, layout.wrench_offset + 6 * bodies)
            self.assertGreaterEqual(layout.origin_offset, layout.moment_offset + 13 * bodies)

    def test_reject_unsupported_coordinate_maps(self):
        """Reject incomplete permutations before constructing generated kernels."""
        with self.assertRaises(ValueError):
            TreeLayout(3, 2, 3, 2, 2, 0, (0, 0))


def fixture(floating, device):
    """Use interleaved branches, a prismatic joint and a welded inertial body."""
    scene = newton.ModelBuilder()
    parents = [-1, 0, 0, 1, 2, 1, 4]
    bodies, joints = [], []
    for index, parent in enumerate(parents):
        body = scene.add_link(mass=0.5 + index * 0.2, inertia=wp.mat33(np.diag([0.04, 0.05, 0.07])))
        bodies.append(body)
        anchor = wp.transform(wp.vec3(0.07, 0.03 * index, -0.08), wp.quat_identity())
        if index == 0:
            joint = scene.add_joint_free(body) if floating else scene.add_joint_fixed(-1, body, parent_xform=anchor)
        elif index == 5:
            joint = scene.add_joint_fixed(bodies[parent], body, parent_xform=anchor)
        else:
            add = scene.add_joint_prismatic if index == 3 else scene.add_joint_revolute
            joint = add(bodies[parent], body, axis=newton.Axis.Y, parent_xform=anchor, armature=0.05)
        joints.append(joint)
    scene.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(scene, 2)
    model = builder.finalize(device=device)
    model.rigid_contact_max = 8
    options = {"pgs_mode": "split", "friction_anchor_beta": 0.0, "use_parallel_streams": False, "double_buffer": False}
    with patch.dict(os.environ, {"FEATHER_PGS_TREE_KINETIC_STATE": "0"}):
        solver, reference = (SolverFeatherPGS(model, **options) for _ in range(2))
    count = model.joint_dof_count // 2
    mass = _SparseMassMatrixPlan.build(parents, np.diff(model.joint_qd_start.numpy())[: len(parents)])
    solver._sparse_mass_matrix_size = count
    solver._sparse_mass_matrix_plan = mass
    solver._sparse_mass_matrix_indices = mass.to_device(device)
    solver._sparse_Linv = wp.zeros((2, mass.nonzero_count), dtype=float, device=device)
    solver._sparse_mass_matrix_status = wp.zeros(2, dtype=int, device=device)
    solver._sparse_factor_warps_per_block = 4
    operator = TreeSparseOperator(solver)
    owner = TreeKineticState(operator)
    operator.kinetic_state = owner
    return model, solver, reference, operator, owner, mass


class TestTreePhysicalCPU(unittest.TestCase):
    def check_physical_cache(self, device):
        """Compare public motion, physical mass and both held-operator actions."""
        for floating in (False, True):
            for chains in (False, True):
                with (
                    self.subTest(floating=floating, chains=chains),
                    patch.dict(os.environ, {"FEATHER_PGS_TREE_CHAIN_SCAN": str(int(chains))}),
                ):
                    model, solver, reference, operator, owner, mass = fixture(floating, device)
                    self.assertNotEqual(tuple(mass.permutation), tuple(range(mass.dof_count - 1, -1, -1)))
                    state, output, control = model.state(), model.state(), model.control()
                    rng = np.random.default_rng(81)
                    state.joint_qd.assign(rng.uniform(-0.4, 0.4, model.joint_dof_count).astype(np.float32))
                    state.body_f.assign(rng.normal(0, 0.1, (model.body_count, 6)).astype(np.float32))
                    control.joint_f.assign(rng.uniform(-0.1, 0.1, model.joint_dof_count).astype(np.float32))
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    dt, count = 0.004, mass.dof_count
                    reference._last_step_dt = dt
                    ref_aug = reference._prepare_augmented_state(state, model.state(), control)
                    reference._stage1_fk_id(state, ref_aug, model.state())
                    reference._stage1_drives(state, ref_aug, control, dt)
                    rhs = ref_aug.joint_tau.numpy().reshape(2, count).astype(float)
                    augmented = solver._prepare_augmented_state(state, output, control)
                    augmented.joint_tau.zero_()
                    owner.begin(state, augmented, state.joint_qd, dt, True)
                    public_state(self, model, state)
                    np.testing.assert_allclose(
                        augmented.body_v_s.numpy(), ref_aug.body_v_s.numpy(), atol=2e-6, rtol=2e-5
                    )
                    physical = physical_mass(model, state)
                    matrices = [physical[i * count : (i + 1) * count, i * count : (i + 1) * count] for i in range(2)]
                    geometry = owner.geometric.numpy()
                    for group, matrix in enumerate(matrices):
                        expected = matrix[np.ix_(mass.permutation, mass.permutation)][mass.entry_rows, mass.columns]
                        np.testing.assert_allclose(geometry[group], expected, atol=3e-6, rtol=3e-5)
                    solver.mass_update_mask.fill_(1)
                    operator.refresh(augmented)
                    owner.predict(state, augmented, control, state.joint_qd, dt)
                    for group, matrix in enumerate(matrices):
                        held_matrix = matrix + np.diag(solver.R_by_size[count].numpy()[group])
                        expected = np.linalg.solve(held_matrix, rhs[group])
                        actual = augmented.joint_qdd.numpy().reshape(2, count)[group]
                        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
                    solver.v_out.assign(solver.v_hat)
                    owner.finish(state, augmented, output, dt, False)
                    public_state(self, model, output)
                    before = owner.data.generation.numpy().copy()
                    owner.begin(output, augmented, output.joint_qd, dt, False)
                    np.testing.assert_array_equal(owner.data.generation.numpy(), before)
                    owner.invalidate(wp.array([True, False], dtype=wp.bool, device=device))
                    owner.begin(output, augmented, output.joint_qd, dt, False)
                    np.testing.assert_array_equal(owner.data.generation.numpy(), np.add(before, [1, 0]))
                    held = operator.data.W.numpy().copy()
                    solver.mass_update_mask.zero_()
                    operator.refresh(augmented)
                    np.testing.assert_array_equal(operator.data.W.numpy(), held)

    def test_physical_cache(self):
        """Exercise both free/fixed roots and traversal schedules on CPU."""
        self.check_physical_cache("cpu")


class TestTreePhysicalCUDA(TestTreePhysicalCPU):
    def test_default_admission_and_velocity_fallback(self):
        """Select complete native trees without requiring an environment opt-in."""
        if not wp.get_cuda_devices():
            self.skipTest("No leased CUDA device")
        with patch.dict(os.environ):
            os.environ.pop("FEATHER_PGS_TREE_KINETIC_STATE", None)
            os.environ.pop("FEATHER_PGS_TREE_CHAIN_SCAN", None)
            os.environ["FEATHER_PGS_G1_CHAIN_SCAN"] = "0"
            case = split._case(True, floating=True, branches=3, depth=2, device="cuda:0")
            self.assertIsNotNone(case[1]._tree_kinetic_state)
            self.assertTrue(case[1]._tree_kinetic_state.chain_scan)
            self.assertEqual(case[1]._tree_kinetic_state.block_dim, 32)
            ordinary = split._solver(case[0], True, pgs_mode="matrix_free", enable_joint_velocity_limits=True)
            self.assertIsNone(ordinary._tree_kinetic_state)
            os.environ["FEATHER_PGS_TREE_KINETIC_STATE"] = "0"
            explicit = split._solver(case[0], True)
            self.assertIsNone(explicit._tree_kinetic_state)

    def test_physical_cache(self):
        """Check non-G1 geometry and force actions on the explicitly leased GPU."""
        if not wp.get_cuda_devices():
            self.skipTest("No leased CUDA device")
        self.check_physical_cache("cuda:0")

    def test_complete_steps_reset_and_graph(self):
        """Preserve native contact/drive behavior through held factors and resets."""
        if not wp.get_cuda_devices():
            self.skipTest("No leased CUDA device")

        for floating in (False, True):
            cases = []
            for enabled in (False, True):
                with patch.dict(
                    os.environ,
                    {"FEATHER_PGS_TREE_KINETIC_STATE": str(int(enabled)), "FEATHER_PGS_TREE_CHAIN_SCAN": "1"},
                ):
                    # This helper preserves the existing split contact law.

                    original = split._solver

                    def make_solver(model, sparse, _original=original, **overrides):
                        return _original(model, sparse, use_parallel_streams=True, **overrides)

                    with patch.object(split, "_solver", make_solver):
                        case = split._case(True, floating=floating, branches=3, depth=2, device="cuda:0")
                self.assertIsNotNone(case[1]._sparse_mass_matrix_size)
                self.assertEqual(case[1]._tree_kinetic_state is not None, enabled)
                cases.append(case)
            for step in range(12):
                for case in cases:
                    if step in (4, 9):
                        mask = wp.array([True, False], dtype=wp.bool, device="cuda:0")
                        case[1].reset(case[2], world_mask=mask)
                    split._step(case)
                    public_state(self, case[0], case[3])
                for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    np.testing.assert_allclose(
                        getattr(cases[1][3], field).numpy(), getattr(cases[0][3], field).numpy(), atol=3e-4, rtol=3e-4
                    )
            graphs = []
            for case in cases:
                with wp.ScopedCapture(device="cuda:0") as capture:
                    split._step(case)
                    split._step(case)
                graphs.append(capture.graph)
            for _ in range(4):
                for graph in graphs:
                    wp.capture_launch(graph)
            for case in cases:
                public_state(self, case[0], case[3])
                np.testing.assert_array_equal(case[1]._sparse_mass_matrix_status.numpy(), 0)
            for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(cases[1][3], field).numpy(), getattr(cases[0][3], field).numpy(), atol=4e-4, rtol=4e-4
                )


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check generated state caches against physical trees outside the G1 layout."""

import contextlib
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import tree_kinetic_state
from newton._src.solvers.feather_pgs import tree_scan as ts
from newton._src.solvers.feather_pgs.held_tree import TreeSparseOperator
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton._src.solvers.feather_pgs.tree_kinetic_owner import TreeKineticState
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.tests import feather_pgs_sparse_scene as split
from newton.tests.feather_pgs_law_harness import recomputing
from tools.fpgs_bench.public_state_oracle import force_response, physical_mass, public_state


@contextlib.contextmanager
def _heavy_chains():
    """Run every scan of the held trees built inside on the heavy chains (tools.fpgs_bench.held_tree_ab)."""
    from tools.fpgs_bench.held_tree_ab import _heavy_chains as chains  # noqa: PLC0415

    with chains():
        yield


class TestTreeLayout(unittest.TestCase):
    def test_complete_cooperative_lane_coverage(self):
        """Keep every body and padded chain lane inside a tree's lanes: a tree whose heavy chains pack into two
        warps takes them, also at 32 bodies or fewer."""
        small = TreePlan.build([-1, 0, 0, 1, 2, 1, 4], [6, 1, 1, 1, 1, 1, 1])
        self.assertLessEqual(small.chain_packing(2).lanes, 32)
        self.assertEqual(tree_kinetic_state.held_lanes(small, 64, cuda=True), 8)
        parents = [-1]
        for _ in range(10):
            parent = 0
            for _ in range(3):
                parents.append(parent)
                parent = len(parents) - 1
        padded = TreePlan.build(parents, [6] + [1] * 30)
        self.assertEqual(padded.chain_packing(2).lanes, 64)
        layout = TreeLayout(31, 36, 666, 36, 3, 6, tuple(range(36)), 64)
        self.assertEqual((layout.block, layout.pack), (64, 1))
        # A warp's lanes leave such a tree no chain schedule; its two warps admit them.
        self.assertFalse(tree_kinetic_state.held_schedules(padded, 32, chains=True).chains)
        self.assertIn(
            ts.Schedule.HEAVY_CHAIN,
            ts.schedule_costs(padded, ts.Direction.UP, ts.add(ts.Group("wrench", 6)), lanes=64, chains=True),
        )


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
    # The test builds its own owner on the solver.
    solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
    count = model.joint_dof_count // 2
    mass = TreePlan.build(parents, np.diff(model.joint_qd_start.numpy())[: len(parents)]).factor
    solver.placement.sparse_mass_matrix_size = count
    solver.factor.allocate_sparse(mass)
    solver.factor.sparse_factor_warps_per_block = 4
    operator = TreeSparseOperator(solver)
    owner = TreeKineticState(operator)
    operator.kinetic_state = owner
    return model, solver, operator, owner, mass


class TestTreePhysicalCPU(unittest.TestCase):
    def check_physical_cache(self, device):
        """Compare public motion, physical mass and both held-operator actions."""
        for floating in (False, True):
            for chains in (False, True):
                with (
                    self.subTest(floating=floating, chains=chains),
                    _heavy_chains() if chains else contextlib.nullcontext(),
                ):
                    model, solver, operator, owner, mass = fixture(floating, device)
                    self.assertNotEqual(tuple(mass.permutation), tuple(range(mass.dof_count - 1, -1, -1)))
                    state, output, control = model.state(), model.state(), model.control()
                    rng = np.random.default_rng(81)
                    state.joint_qd.assign(rng.uniform(-0.4, 0.4, model.joint_dof_count).astype(np.float32))
                    state.body_f.assign(rng.normal(0, 0.1, (model.body_count, 6)).astype(np.float32))
                    control.joint_f.assign(rng.uniform(-0.1, 0.1, model.joint_dof_count).astype(np.float32))
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    dt, count = 0.004, mass.dof_count
                    solver.factor.joint_tau.zero_()
                    owner.begin(state, solver.factor, state.joint_qd, dt, True)
                    public_state(self, model, state)
                    physical = physical_mass(model, state)
                    matrices = [physical[i * count : (i + 1) * count, i * count : (i + 1) * count] for i in range(2)]
                    solver.factor.mass_update_mask.fill_(1)
                    operator.refresh(solver.factor)
                    response = force_response(owner, solver, state, control, state.joint_qd, dt).reshape(2, count)
                    forces = control.joint_f.numpy().reshape(2, count).astype(float)
                    for group, matrix in enumerate(matrices):
                        armature = solver.factor.dof_armature.numpy()[group * count : (group + 1) * count]
                        held_matrix = matrix + np.diag(armature)
                        expected = np.linalg.solve(held_matrix, forces[group])
                        np.testing.assert_allclose(response[group], expected, atol=3e-5, rtol=3e-4)
                    solver.v_out.assign(solver.v_hat)
                    owner.finish(state, solver.factor, output, dt, False)
                    public_state(self, model, output)
                    before = owner.data.generation.numpy().copy()
                    owner.begin(output, solver.factor, output.joint_qd, dt, False)
                    np.testing.assert_array_equal(owner.data.generation.numpy(), before)
                    owner.invalidate(wp.array([True, False], dtype=wp.bool, device=device))
                    owner.begin(output, solver.factor, output.joint_qd, dt, False)
                    np.testing.assert_array_equal(owner.data.generation.numpy(), np.add(before, [1, 0]))
                    held = operator.data.W.numpy().copy()
                    solver.factor.mass_update_mask.zero_()
                    operator.refresh(solver.factor)
                    np.testing.assert_array_equal(operator.data.W.numpy(), held)

    def test_physical_cache(self):
        """Exercise both free/fixed roots and traversal schedules on CPU."""
        self.check_physical_cache("cpu")


class TestTreePhysicalCUDA(TestTreePhysicalCPU):
    def test_default_admission_and_motion_velocity(self):
        """Select native trees, their launch from the tree's data and the prescaled-velocity contract."""
        if not wp.get_cuda_devices():
            self.skipTest("No leased CUDA device")
        case = split.make_case(True, floating=True, branches=3, depth=2, device="cuda:0")
        self.assertIsNotNone(case[1].factor.held)
        (owner,) = case[1].factor.held.groups
        self.assertEqual(owner.layout.lanes, tree_kinetic_state.held_lanes(owner.tree, owner.group_count))
        self.assertEqual(owner.block_dim, owner.layout.block)
        limited = split.make_solver(case[0], True, enable_joint_velocity_limits=True)
        self.assertIsNotNone(limited.factor.held)
        self.assertTrue(limited.factor.held.groups[0].motion_velocity)

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
                with _heavy_chains():
                    case = split.make_case(True, floating=floating, branches=3, depth=2, device="cuda:0")
                self.assertIsNotNone(case[1].placement.sparse_mass_matrix_size)
                if not enabled:
                    recomputing(case[1])
                cases.append(case)
            for step in range(12):
                for case in cases:
                    if step in (4, 9):
                        mask = wp.array([True, False], dtype=wp.bool, device="cuda:0")
                        case[1].reset(case[2], world_mask=mask)
                    split.step(case)
                    public_state(self, case[0], case[3])
                for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    np.testing.assert_allclose(
                        getattr(cases[1][3], field).numpy(), getattr(cases[0][3], field).numpy(), atol=3e-4, rtol=3e-4
                    )
            graphs = []
            for case in cases:
                with wp.ScopedCapture(device="cuda:0") as capture:
                    split.step(case)
                    split.step(case)
                graphs.append(capture.graph)
            for _ in range(4):
                for graph in graphs:
                    wp.capture_launch(graph)
            for case in cases:
                public_state(self, case[0], case[3])
            for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(cases[1][3], field).numpy(), getattr(cases[0][3], field).numpy(), atol=4e-4, rtol=4e-4
                )


if __name__ == "__main__":
    unittest.main()

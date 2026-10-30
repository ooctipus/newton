# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check heterogeneous scalar trees against independent physical operators."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.held_tree import HeldTree
from newton.solvers import SolverFeatherPGS
from tools.fpgs_bench.public_state_oracle import physical_mass, public_state

DT = 1.0 / 240.0


def model_fixture(device="cpu", *, worlds=3, reordered=False, primary_dofs=9):
    """Build a nine-coordinate tree and four independent scalar branches."""
    scene = newton.ModelBuilder()
    for kind in (1, 0) if reordered else (0, 1):
        parents = [-1, *range(primary_dofs + 1)] if kind == 0 else [-1, 0, 1, 0, 3, 0, 5, 0, 7]
        if kind == 0:
            parents[-1] = len(parents) - 3
        bodies, joints = [], []
        for local, parent in enumerate(parents):
            body = scene.add_link(
                mass=0.7 + 0.11 * local,
                inertia=wp.mat33(np.array([[0.09, 0.003, -0.002], [0.003, 0.11, 0.001], [-0.002, 0.001, 0.14]])),
                com=wp.vec3(0.013, -0.009, 0.017),
                lock_inertia=True,
            )
            bodies.append(body)
            anchor = wp.transform(
                wp.vec3(0.07, -0.02 + kind * 0.4, 0.10), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.07 * (local + 1))
            )
            child = wp.transform(wp.vec3(0.01, -0.015, 0.02), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -0.04))
            values = {
                "parent": -1 if parent < 0 else bodies[parent],
                "child": body,
                "parent_xform": anchor,
                "child_xform": child,
            }
            fixed = local in (0, min(8, primary_dofs)) if kind == 0 else local % 2 == 0
            if fixed:
                joint = scene.add_joint_fixed(**values)
            else:
                linear = (kind == 1 and local >= 5) or (kind == 0 and local in (9, 10))
                add = scene.add_joint_prismatic if linear else scene.add_joint_revolute
                joint = add(
                    **values,
                    axis=newton.Axis.Y,
                    armature=0.04,
                    target_ke=1.2,
                    target_kd=0.15,
                    limit_lower=-0.8,
                    limit_upper=0.8,
                )
            joints.append(joint)
        scene.add_articulation(joints)
        scene.add_shape_box(bodies[-1], hx=0.02, hy=0.02, hz=0.02)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(2.0, 2.0, 0.0))
    model = builder.finalize(device=device)
    model.rigid_contact_max = 32
    return model


def make_solver(model, *, enabled=False, permuted=False, friction_anchor_beta=0.0):
    """Retain original Cholesky and independent-branch diagonal representations."""
    setup = SolverFeatherPGS._setup_size_grouping

    def grouping(solver, source):
        setup(solver, source)
        if permuted:
            inverse = solver.art_group_idx.numpy().copy()
            for size, group in solver.group_to_art.items():
                values = np.roll(group.numpy()[::-1], size % group.size)
                group.assign(values)
                inverse[values] = np.arange(group.size)
            solver.art_group_idx.assign(inverse)

    # The held owner on dense storage, or ordinary dynamics; one buffer set either way.
    pins = {"storage": "dense", "double_buffer": False, **({} if enabled else {"evaluator": "ordinary"})}
    with (
        patch.object(SolverFeatherPGS, "_route_pins", pins),
        patch.object(SolverFeatherPGS, "_setup_size_grouping", grouping),
    ):
        solver = SolverFeatherPGS(
            model,
            pgs_iterations=8,
            pgs_beta=0.05,
            pgs_cfm=1e-6,
            dense_max_constraints=192,
            mf_max_constraints=64,
            enable_joint_limits=True,
            enable_joint_velocity_limits=False,
            update_mass_matrix_interval=2,
            angular_damping=0.05,
            friction_anchor_beta=friction_anchor_beta,
        )
    if not model.device.is_cuda:
        # Direct CPU physical tests supply the original compact descriptor;
        # production admission and complete stepping remain CUDA-only.
        solver._compact_diagonal_mass_size = 4
        solver._diagonal_inverse_mass = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
    return solver


class TestMixedTreeCache(unittest.TestCase):
    def test_complete_heterogeneous_owner(self):
        """Cover every body, coordinate and articulation with independent group IDs."""
        for reordered in (False, True):
            model = model_fixture(reordered=reordered)
            solver = make_solver(model, permuted=True)
            owner = HeldTree(solver)
            self.assertEqual(sorted(group.layout.dofs for group in owner.groups), [4, 9])
            covered = np.concatenate([group.sparse.plan.group_to_art.numpy() for group in owner.groups])
            np.testing.assert_array_equal(np.sort(covered), np.arange(6))

    def check_physical(self, device):
        """Compare physical mass, bias, held-factor actions and public state."""
        for primary_dofs, reordered in ((9, False), (9, True), (7, True), (12, False)):
            model = model_fixture(device, reordered=reordered, primary_dofs=primary_dofs)
            model.joint_target_ke.zero_()
            model.joint_target_kd.zero_()
            solver, reference = (make_solver(model, permuted=True) for _ in range(2))
            owner = HeldTree(solver)
            state, output, control = model.state(), model.state(), model.control()
            rng = np.random.default_rng(3401)
            state.joint_q.assign(rng.uniform(-0.15, 0.15, model.joint_coord_count).astype(np.float32))
            raw = rng.uniform(-0.6, 0.6, model.joint_dof_count).astype(np.float32)
            state.joint_qd.assign(raw)
            state.body_f.assign(rng.normal(0, 0.2, (model.body_count, 6)).astype(np.float32))
            control.joint_f.assign(rng.uniform(-0.2, 0.2, model.joint_dof_count).astype(np.float32))
            for item in (solver, reference):
                item._last_step_dt = DT
                item._fk_id_cache_enabled = False
                item._passive_spring_stiffness.fill_(0.31)
                item._passive_spring_ref.fill_(-0.07)
                item._passive_joint_damping.fill_(0.12)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            aug = solver._prepare_augmented_state(state, output, control)
            ref = reference._prepare_augmented_state(state, model.state(), control)
            reference._stage1_fk_id(state, ref, model.state())
            reference._stage1_drives(state, ref, control, DT)
            rhs = ref.joint_tau.numpy().astype(float)
            physical = physical_mass(model, state)
            aug.joint_tau.zero_()
            owner.begin(state, aug, state.joint_qd, DT, True)
            np.testing.assert_allclose(solver.joint_S_s.numpy(), reference.joint_S_s.numpy(), atol=4e-6, rtol=3e-5)
            np.testing.assert_allclose(solver.body_v_s.numpy(), reference.body_v_s.numpy(), atol=4e-6, rtol=3e-5)
            expected = np.zeros(model.joint_dof_count)
            starts = solver.articulation_dof_start.numpy()
            for group in owner.groups:
                operator, n = group.sparse, group.layout.dofs
                mass = operator.mass
                for index, art in enumerate(operator.plan.group_to_art.numpy()):
                    ids = np.arange(starts[art], starts[art] + n)
                    block = physical[np.ix_(ids, ids)]
                    packed = block[np.ix_(mass.permutation, mass.permutation)][mass.entry_rows, mass.columns]
                    np.testing.assert_allclose(group.geometric.numpy()[index], packed, atol=4e-6, rtol=4e-5)
                    expected[ids] = np.linalg.solve(block + np.diag(solver.R_by_size[n].numpy()[index]), rhs[ids])
                    if not model.device.is_cuda and not operator.diagonal:
                        lower = solver.L_by_size[n].numpy()
                        lower[index] = np.linalg.cholesky(block + np.diag(solver.R_by_size[n].numpy()[index]))
                        solver.L_by_size[n].assign(lower)
            solver.mass_update_mask.fill_(1)
            owner.refresh(aug)
            held = {
                group.layout.dofs: (
                    solver._diagonal_inverse_mass.numpy().copy()
                    if group.sparse.diagonal
                    else solver.L_by_size[group.layout.dofs].numpy().copy()
                )
                for group in owner.groups
            }
            owner.predict(state, aug, control, state.joint_qd, DT)
            np.testing.assert_allclose(solver.joint_qdd.numpy(), expected, atol=5e-5, rtol=5e-4)
            np.testing.assert_allclose(solver.v_hat.numpy(), raw + DT * expected, atol=4e-6, rtol=3e-5)
            solver.v_out.assign(solver.v_hat)
            owner.finish(state, aug, output, DT, False)
            public_state(self, model, output)
            generation = [group.data.generation.numpy().copy() for group in owner.groups]
            owner.begin(output, aug, output.joint_qd, DT, False)
            for group, before in zip(owner.groups, generation, strict=False):
                np.testing.assert_array_equal(group.data.generation.numpy(), before)
            mask = wp.array([False, True, False], dtype=wp.bool, device=device)
            owner.invalidate(mask)
            np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), [1, 1, 0, 0, 1, 1])
            owner.begin(output, aug, output.joint_qd, DT, False)
            for group, before in zip(owner.groups, generation, strict=False):
                arts = group.sparse.plan.group_to_art.numpy()
                np.testing.assert_array_equal(
                    group.data.generation.numpy() - before, solver.art_to_world.numpy()[arts] == 1
                )
            requests = np.zeros(model.articulation_count, np.int32)
            requests[-1] = 1
            solver._mass_update_requested.assign(requests)
            generation = [group.data.generation.numpy().copy() for group in owner.groups]
            owner.begin(output, aug, output.joint_qd, DT, False)
            for group, before in zip(owner.groups, generation, strict=False):
                arts = group.sparse.plan.group_to_art.numpy()
                np.testing.assert_array_equal(
                    group.data.generation.numpy() - before, arts == model.articulation_count - 1
                )
            solver._mass_update_requested.zero_()
            bank = model.state()
            for key in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
                wp.copy(getattr(bank, key), getattr(output, key))
            generation = [group.data.generation.numpy().copy() for group in owner.groups]
            # The copy holds the words of the bank the caches were stamped with: no group repairs.
            owner.begin(bank, aug, bank.joint_qd, DT, False, state_out=output)
            for group, before in zip(owner.groups, generation, strict=False):
                np.testing.assert_array_equal(group.data.generation.numpy(), before)
            solver.mass_update_mask.zero_()
            owner.refresh(aug)
            for group in owner.groups:
                value = solver._diagonal_inverse_mass if group.sparse.diagonal else solver.L_by_size[group.layout.dofs]
                np.testing.assert_array_equal(value.numpy(), held[group.layout.dofs])
            if model.device.is_cuda:
                wp.synchronize()  # Complete fixture allocations before graph capture.
                with wp.ScopedCapture(device=device) as capture:
                    owner.invalidate(mask)
                    owner.begin(bank, aug, bank.joint_qd, DT, False)
                    owner.predict(bank, aug, control, bank.joint_qd, DT)
                generation = [group.data.generation.numpy().copy() for group in owner.groups]
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                for group, before in zip(owner.groups, generation, strict=False):
                    arts = group.sparse.plan.group_to_art.numpy()
                    np.testing.assert_array_equal(
                        group.data.generation.numpy() - before, 3 * (solver.art_to_world.numpy()[arts] == 1)
                    )
            np.testing.assert_array_equal(state.joint_qd.numpy(), raw)
            if model.device.is_cuda:
                wp.synchronize()  # Finish auxiliary streams before replacing this fixture.

    def test_ownership_change_rejected_before_invalidation(self):
        """Reject omitted, duplicated and rebound factor ownership without writes."""
        model = model_fixture()
        solver = make_solver(model)
        owner = HeldTree(solver)
        solver._fk_id_cache_valid.fill_(1)
        original = solver.group_to_art[9]
        values = original.numpy()
        for changed in (values[::-1].copy(), np.full_like(values, values[0])):
            solver.group_to_art[9] = wp.array(changed, dtype=int, device=model.device)
            with self.assertRaises((ValueError, RuntimeError)):
                owner.validate_model()
            np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 1)
        solver.group_to_art[9] = original
        owner.validate_model()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_physical_cuda(self):
        """Check cooperative caches and held response bindings on CUDA."""
        self.check_physical("cuda:0")


if __name__ == "__main__":
    unittest.main()

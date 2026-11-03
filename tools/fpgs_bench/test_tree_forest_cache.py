# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete cached publication with distinct group/articulation/world IDs."""

import contextlib
import dataclasses
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.held_tree import TreeSparseOperator
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton._src.solvers.feather_pgs.tree_kinetic_owner import TreeKineticState
from newton._src.solvers.feather_pgs.tree_kinetic_plan import make_plan
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.tests import test_feather_pgs_sparse_solver as native
from tools.fpgs_bench.public_state_oracle import force_response, physical_mass, public_state
from tools.fpgs_bench.test_tree_locked_root import locked_roots


def fixture(floating, device, *, permuted=True, motion=False):
    """Place two identical responsive trees in each of three independent worlds."""
    scene = newton.ModelBuilder()
    parents = [-1, 0, 0, 1, 2, 1, 4]
    for tree in range(2):
        bodies, joints = [], []
        for index, parent in enumerate(parents):
            body = scene.add_link(mass=0.5 + index * 0.2, inertia=wp.mat33(np.diag([0.04, 0.05, 0.07])))
            bodies.append(body)
            anchor = wp.transform(
                wp.vec3(0.07 + (0.6 * tree if index == 0 else 0.0), 0.03 * index, -0.08), wp.quat_identity()
            )
            if index == 0:
                joint = (
                    scene.add_joint_free(body, parent_xform=anchor)
                    if floating
                    else scene.add_joint_fixed(-1, body, parent_xform=anchor)
                )
            elif index == 5:
                joint = scene.add_joint_fixed(bodies[parent], body, parent_xform=anchor)
            else:
                add = scene.add_joint_prismatic if index == 3 else scene.add_joint_revolute
                joint = add(bodies[parent], body, axis=newton.Axis.Y, parent_xform=anchor, armature=0.05)
            joints.append(joint)
        scene.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(scene, 3)
    model = builder.finalize(device=device)
    model.rigid_contact_max = 8
    # The test builds its own owner on the solver.
    solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
    arts = model.articulation_count
    count = model.joint_dof_count // arts
    mass = TreePlan.build(parents, np.diff(model.joint_qd_start.numpy())[: len(parents)]).factor
    solver.placement.sparse_mass_matrix_size = count
    solver.factor.allocate_sparse(mass)
    solver.factor.sparse_factor_warps_per_block = 4
    groups = np.array([3, 0, 5, 1, 4, 2] if permuted else list(range(arts)), np.int32)
    solver.forest.group_to_art[count].assign(groups)
    solver.options = solver.laws.options = solver.factor.options = solver.placement.options = solver.sweep.options = (
        dataclasses.replace(solver.options, enable_joint_velocity_limits=motion)
    )
    solver.factor.passive_spring_stiffness.fill_(0.27)
    solver.factor.passive_spring_ref.fill_(-0.1)
    solver.factor.passive_joint_damping.fill_(0.19)
    model.joint_velocity_limit.fill_(0.25)
    operator = TreeSparseOperator(solver)
    owner = TreeKineticState(operator)
    operator.kinetic_state = owner
    return model, solver, operator, owner, mass


def _heavy_chains():
    """Run every scan of the held trees built inside on the heavy chains (tools.fpgs_bench.held_tree_ab)."""
    from tools.fpgs_bench.held_tree_ab import _heavy_chains as chains  # noqa: PLC0415

    return chains()


class TestTreeForestCache(unittest.TestCase):
    def check_physical_forest(self, device):
        """Compare every articulation's physical operator and complete publication."""
        for floating, chains, motion in (
            (False, False, False),
            (True, True, False),
            (False, True, True),
            (True, False, True),
        ):
            with (
                self.subTest(floating=floating, chains=chains, motion=motion),
                _heavy_chains() if chains else contextlib.nullcontext(),
            ):
                model, solver, operator, owner, mass = fixture(floating, device, motion=motion)
                count, arts, dt = mass.dof_count, model.articulation_count, native.DT
                groups = operator.plan.group_to_art.numpy()
                self.assertEqual(model.world_count, 3)
                self.assertEqual(arts, 6)
                self.assertEqual(owner.data.bias.shape, (arts, count))
                self.assertEqual(operator.data.valid.shape, (arts,))
                state, output, control = model.state(), model.state(), model.control()
                rng = np.random.default_rng(871)
                raw = rng.uniform(-0.8, 0.8, model.joint_dof_count).astype(np.float32)
                raw[2 * count :] *= 0.03
                state.joint_qd.assign(raw)
                state.body_f.assign(rng.normal(0, 0.1, (model.body_count, 6)).astype(np.float32))
                control.joint_f.assign(rng.uniform(-0.2, 0.2, model.joint_dof_count).astype(np.float32))
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                solver.factor.last_step_dt = dt
                velocity = state.joint_qd
                if motion:
                    solver.factor.prescale_velocity(state.joint_qd, solver.qd_work)
                    velocity = solver.qd_work
                forces = control.joint_f.numpy().reshape(arts, count).astype(float)
                physical = physical_mass(model, state)
                solver.factor.joint_tau.zero_()
                owner.begin(state, solver.factor, velocity, dt, True)
                expected = np.empty_like(forces)
                for art in groups:
                    block = physical[art * count : (art + 1) * count, art * count : (art + 1) * count]
                    expected[art] = np.linalg.solve(
                        block + np.diag(solver.factor.dof_armature.numpy()[art * count : (art + 1) * count]),
                        forces[art],
                    )
                solver.factor.mass_update_mask.fill_(1)
                operator.refresh(solver.factor)
                response = force_response(owner, solver, state, control, velocity, dt)
                np.testing.assert_allclose(response.reshape(arts, count), expected, atol=4e-5, rtol=4e-4)
                held = operator.data.W.numpy().copy()
                # A failed last factor must poison only its mapped articulation,
                # even when the factor-group index exceeds the number of worlds.
                status = np.zeros(arts, np.int32)
                status[-1] = 1
                operator.data.status.assign(status)
                owner.predict(state, solver.factor, control, velocity, dt)
                bad = ~np.isfinite(solver.factor.joint_qdd.numpy().reshape(arts, count)).all(axis=1)
                np.testing.assert_array_equal(np.flatnonzero(bad), groups[-1:])
                operator.data.status.zero_()
                owner.status.zero_()
                owner.predict(state, solver.factor, control, velocity, dt)
                solver.v_out.assign(solver.v_hat)
                output.body_q.fill_(wp.transform(wp.vec3(123.0), wp.quat_identity()))
                output.body_qd.fill_(wp.spatial_vector(123.0))
                owner.finish(state, solver.factor, output, dt, False)
                public_state(self, model, output)
                generation = owner.data.generation.numpy().copy()
                owner.begin(output, solver.factor, output.joint_qd, dt, False)
                np.testing.assert_array_equal(owner.data.generation.numpy(), generation)
                mask = wp.array([False, True, False], dtype=wp.bool, device=device)
                owner.invalidate(mask)
                np.testing.assert_array_equal(owner.data.current_valid.numpy(), [1, 1, 0, 0, 1, 1])
                owner.begin(output, solver.factor, output.joint_qd, dt, False)
                np.testing.assert_array_equal(owner.data.generation.numpy() - generation, np.isin(groups, [2, 3]))
                np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)
                # A late request belongs to one articulation, independently of
                # the other tree in that same world and of factor ordering.
                generation = owner.data.generation.numpy().copy()
                solver.factor.mass_update_requested.zero_()
                requests = np.zeros(arts, np.int32)
                requests[-1] = 1
                solver.factor.mass_update_requested.assign(requests)
                owner.begin(output, solver.factor, output.joint_qd, dt, False)
                np.testing.assert_array_equal(owner.data.generation.numpy() - generation, groups == arts - 1)
                np.testing.assert_array_equal(owner.geometry_valid.numpy(), groups == arts - 1)
                solver.factor.mass_update_requested.zero_()
                bank = model.state()
                for name in native.STATE_FIELDS:
                    wp.copy(getattr(bank, name), getattr(output, name))
                # Identity alone does not cover a copy: every group repairs.
                generation = owner.data.generation.numpy().copy()
                owner.begin(bank, solver.factor, bank.joint_qd, dt, False)
                np.testing.assert_array_equal(owner.data.generation.numpy(), generation + 1)
                # Comparing the copy with the bank the finish published keeps every cache,
                # except under prescaled motion, which validates by identity.
                solver.v_out.assign(solver.v_hat)
                owner.finish(bank, solver.factor, output, dt, False)
                copy = model.state()
                for name in native.STATE_FIELDS:
                    wp.copy(getattr(copy, name), getattr(output, name))
                generation = owner.data.generation.numpy().copy()
                owner.begin(copy, solver.factor, copy.joint_qd, dt, False, state_out=output)
                np.testing.assert_array_equal(owner.data.generation.numpy(), generation + int(motion))
                bank = copy
                if device.startswith("cuda"):
                    with wp.ScopedCapture(device=device) as capture:
                        owner.invalidate(mask)
                        owner.begin(bank, solver.factor, bank.joint_qd, dt, False)
                        owner.predict(bank, solver.factor, control, bank.joint_qd, dt)
                    generation = owner.data.generation.numpy().copy()
                    for _ in range(3):
                        wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(
                        owner.data.generation.numpy() - generation, 3 * np.isin(groups, [2, 3])
                    )
                    np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)
                solver.factor.mass_update_mask.zero_()
                operator.refresh(solver.factor)
                np.testing.assert_array_equal(operator.data.W.numpy(), held)
                np.testing.assert_array_equal(state.joint_qd.numpy(), raw)
                np.testing.assert_array_equal(owner.status.numpy(), 0)

    def test_physical_forest_cpu(self):
        """Check permuted factor maps, physical forces, held mass and world resets."""
        self.check_physical_forest("cpu")

    def test_world_reset_covers_every_mapped_group(self):
        """Invalidate all requested articulations under identity and permuted maps."""
        for permuted in (False, True):
            model, solver, operator, owner, _ = fixture(False, "cpu", permuted=permuted)
            groups = operator.plan.group_to_art.numpy()
            worlds = solver.forest.art_to_world.numpy()
            for values in (
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, False, True],
                [True] * 3,
                [False] * 3,
                None,
            ):
                with self.subTest(permuted=permuted, mask=values):
                    owner.data.current_valid.fill_(1)
                    owner.geometry_valid.fill_(1)
                    mask = None if values is None else wp.array(values, dtype=wp.bool, device=model.device)
                    owner.invalidate(mask)
                    selected = np.ones(3, bool) if values is None else np.array(values)
                    expected = ~selected[worlds]
                    np.testing.assert_array_equal(owner.data.current_valid.numpy(), expected)
                    np.testing.assert_array_equal(owner.geometry_valid.numpy(), expected[groups])

    def test_locked_roots_publish_complete_forests(self):
        """Preserve zero-coordinate root laws across multiple mapped articulations."""
        with locked_roots():
            model, solver, operator, owner, mass = fixture(False, "cpu")
        state, output, control = model.state(), model.state(), model.control()
        state.joint_qd.assign(np.linspace(-0.3, 0.4, model.joint_dof_count, dtype=np.float32))
        solver.factor.last_step_dt = native.DT
        control.joint_f.assign(np.linspace(-0.2, 0.3, model.joint_dof_count, dtype=np.float32))
        physical = physical_mass(model, state)
        solver.factor.joint_tau.zero_()
        owner.begin(state, solver.factor, state.joint_qd, native.DT, True)
        solver.factor.mass_update_mask.fill_(1)
        operator.refresh(solver.factor)
        response = force_response(owner, solver, state, control, state.joint_qd, native.DT)
        regularization = np.empty(model.joint_dof_count)
        armature = solver.factor.dof_armature.numpy()
        for art in operator.plan.group_to_art.numpy():
            span = slice(art * mass.dof_count, (art + 1) * mass.dof_count)
            regularization[span] = armature[span]
        expected = np.linalg.solve(physical + np.diag(regularization), control.joint_f.numpy().astype(float))
        np.testing.assert_allclose(response, expected, atol=4e-5, rtol=4e-4)
        solver.v_out.assign(solver.v_hat)
        owner.finish(state, solver.factor, output, native.DT, False)
        probe = model.state()
        newton.eval_fk(model, output.joint_q, output.joint_qd, probe)
        for name in ("body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(output, name).numpy(), getattr(probe, name).numpy(), atol=3e-6, rtol=3e-5
            )
        np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)

    def test_incomplete_ownership_is_rejected_before_mutation(self):
        """Reject duplicate/missing groups and invalid world IDs before cache writes."""
        model, _, operator, owner, _ = fixture(False, "cpu")
        groups = operator.plan.group_to_art.numpy().copy()
        worlds = operator.plan.art_to_world.numpy().copy()
        owner.data.current_valid.fill_(1)
        owner.geometry_valid.fill_(1)
        broken = groups.copy()
        broken[-1] = broken[0]
        operator.plan.group_to_art.assign(broken)
        with self.assertRaises(ValueError):
            make_plan(model, operator)
        operator.plan.group_to_art.assign(groups)
        broken = worlds.copy()
        broken[-1] = model.world_count
        operator.plan.art_to_world.assign(broken)
        with self.assertRaises(ValueError):
            make_plan(model, operator)
        operator.plan.art_to_world.assign(worlds)
        operator.plan.group_to_art.assign(groups[::-1].copy())
        with self.assertRaisesRegex(RuntimeError, "group_to_art"):
            owner.validate_model()
        operator.plan.group_to_art.assign(groups)
        with patch.object(model, "world_count", model.world_count + 1):
            with self.assertRaisesRegex(RuntimeError, "dimensions"):
                owner.validate_model()
        np.testing.assert_array_equal(owner.data.current_valid.numpy(), 1)
        np.testing.assert_array_equal(owner.geometry_valid.numpy(), 1)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_physical_forest_cuda(self):
        """Exercise both cooperative schedules with independent physical oracles."""
        self.check_physical_forest("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_coupled_trees_select_cache(self):
        """Admit complete multi-articulation worlds without changing their row law."""
        case = native._case(True, robot_count=2)
        self.assertIsNotNone(case[1].factor.held)
        self.assertEqual(case[1].factor.held.groups[0].data.bias.shape, (4, 9))
        native._step(case)
        public_state(self, case[0], case[3])
        np.testing.assert_array_equal(case[1].factor.held.groups[0].data.current_valid.numpy(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

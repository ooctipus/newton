# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve articulated velocity clamps in factor-coordinate response."""

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group, response_builds_rows
from newton._src.solvers.feather_pgs.dense_rows import DenseRows
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_FRICTION as FRICTION
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport, factor_indices, row_supports
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_sparse_scene as sparse_scene
from newton.tests import test_feather_pgs_sparse_contact_triplets as triplet_tests
from newton.tests import test_feather_pgs_sparse_solver as sparse_tests
from newton.tests.feather_pgs_law_harness import unpacked_rows
from newton.tests.feather_pgs_patch_parity import (
    assert_contact_impulses_close,
    assert_history_close,
    assert_ring_coulomb,
    assert_rows_close,
    contact_rows,
)
from newton.tests.test_feather_pgs_sparse_pgs import _sweep
from newton.tests.test_feather_pgs_sparse_research_admission import (
    _change,
    _eligible_solver,
    _ForestProbe,
    _TopologyReached,
)


class TestSparseVelocityLimitAdmission(unittest.TestCase):
    def test_auto_and_enabled_velocity_owner_reach_topology(self):
        """Admit the unchanged augmented matrix-free recipe under either friction law before planning its factor.

        Warm starting is admitted without free rigid bodies: the sparse sweep
        seeds its factor delta with the carried impulses' response. Material
        contacts are admitted: the sparse sweeps step them by their weights.
        """
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        recipes = ({}, {"pgs_warmstart": True}, {"contact_compliance": True})
        for patches, recipe in itertools.product((False, True), recipes):
            solver = _eligible_solver()
            solver.options.enable_joint_velocity_limits = True
            solver.laws.friction_anchors_enabled = patches
            for key, value in recipe.items():
                _change(solver, key, value)
            with (
                self.subTest(patches=patches, recipe=recipe),
                mock.patch.dict(os.environ, {}, clear=True),
                self.assertRaises(_TopologyReached),
            ):
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)

    def test_default_and_unsupported_recipes_keep_existing_owner(self):
        """Decline unsupported laws before removing ordinary response storage."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        for changes in (
            {"_route_pins": {"storage": "dense"}},
            {"pgs_warmstart": True, "forest": _ForestProbe(has_loop_joints=False, has_free_rigid_bodies=True)},
            {"forest": _ForestProbe(has_loop_joints=False, has_free_rigid_bodies=True)},
        ):
            with self.subTest(changes=changes), mock.patch.dict(os.environ, {}, clear=True):
                solver = _eligible_solver()
                solver.options.enable_joint_velocity_limits = True
                for key, value in changes.items():
                    _change(solver, key, value)
                SolverFeatherPGS._setup_sparse_mass_matrix(solver, model)
                self.assertFalse(solver._sparse_velocity_limits)
                self.assertIsNone(solver.placement.sparse_mass_matrix_size)


class TestSparseVelocityLimitRows(unittest.TestCase):
    def _check_rows(self, device):
        """Project signed columns with permuted groups, empty worlds and bounded slots."""
        for counts in ((2, 2, 2), (2, 19, 19)):
            for capacity, table in ((4, False), (5, False), (9, False), (4, True), (5, True), (9, True)):
                with self.subTest(counts=counts, capacity=capacity, table=table):
                    tree = TreePlan.build([-1, 0, 0], counts)
                    plan = tree.factor
                    dofs = plan.dof_count
                    width = max(int(mask).bit_count() for mask in plan.ancestor_mask)
                    rng = np.random.default_rng(932)
                    lower = np.zeros((dofs, dofs))
                    lower[plan.entry_rows, plan.columns] = rng.normal(scale=0.04, size=plan.nonzero_count)
                    lower[np.diag_indices(dofs)] = 1.7
                    inverse = np.linalg.inv(lower)
                    packed = np.stack([scale * inverse[plan.entry_rows, plan.columns] for scale in (1.0, 1.3)])
                    indices = factor_indices(plan, device)

                    def array(values, dtype=float):
                        return wp.array(values, dtype=dtype, device=device)

                    def full(shape, value, dtype=float):
                        return wp.full(shape, value, dtype=dtype, device=device)

                    slots = np.full(4 * dofs, -1, np.int32)
                    signs = np.zeros(4 * dofs, np.float32)
                    limits = np.full(2 * dofs, 0.25, np.float32)
                    velocity = rng.normal(size=2 * dofs).astype(np.float32)
                    live = []
                    # Articulation zero belongs to factor group one; articulation one stays empty.
                    # Capacity four retains only the lower side of the middle DOF's pair.
                    for dof in (0, dofs // 2, dofs - 1):
                        for side, sign in enumerate((1.0, -1.0)):
                            row = 1 + len(live)
                            live.append((row, dof, sign))
                            if row < capacity:
                                slots[2 * dof + side] = row
                                signs[2 * dof + side] = sign
                    limits[dofs // 2] = np.inf  # Preserve the existing fraction-zero allocation law.
                    support = _SparseRowSupport()
                    support_plan = None
                    if table:
                        support_plan = tree.support_classes(np.zeros(0, np.uint64), width)
                        self.assertIsNotNone(support_plan)
                        support = row_supports(support_plan, 2, capacity, device)
                        support.row_class.fill_(-9)
                    kinds, parents = [full((2, capacity), -9, int) for _ in range(2)]
                    mu, beta, cfm, phi, target, incident, diagonal = [full((2, capacity), -9.0) for _ in range(7)]
                    row_dof = full((1, 1, 1) if table else (2, capacity, width), -9, int)
                    factors = full((2, capacity, width), -9.0)
                    wp.launch(
                        response_builds_rows.factor_velocity_limit_rows_kernel,
                        dim=64,
                        block_dim=128,
                        device=device,
                        inputs=[
                            array([1, 0], int),
                            array([0, 1], int),
                            array([0, 0], int),
                            array([0, dofs], int),
                            array(limits),
                            array(slots, int),
                            array(signs),
                            2e-4,
                            indices.ancestor_mask,
                            indices.inverse_permutation,
                            indices.lookup,
                            array(packed),
                            array(velocity),
                        ],
                        outputs=[
                            field_group.upload(
                                DenseRows,
                                device,
                                row_type=kinds,
                                row_parent=parents,
                                row_mu=mu,
                                row_beta=beta,
                                row_cfm=cfm,
                                phi=phi,
                                target=target,
                                diag=diagonal,
                            ),
                            row_dof,
                            factors,
                            incident,
                            support,
                        ],
                    )
                    outputs = [
                        a.numpy() for a in (kinds, parents, mu, beta, cfm, phi, target, factors, incident, diagonal)
                    ]
                    for values in outputs:
                        np.testing.assert_array_equal(values[1], -9)
                        np.testing.assert_array_equal(values[0, 0], -9)
                        np.testing.assert_array_equal(values[0, 7:], -9)
                    dof_values = row_dof.numpy()
                    if table:
                        np.testing.assert_array_equal(dof_values, -9)
                        classes = support.row_class.numpy()
                        np.testing.assert_array_equal(classes[1], -9)
                    for row, dof, sign in live:
                        if row >= capacity:
                            continue
                        columns = support_plan.class_dof[classes[0, row]] if table else dof_values[0, row]
                        actual = np.zeros(dofs)
                        valid = columns >= 0
                        actual[columns[valid]] = outputs[7][0, row, valid]
                        expected = sign * 1.3 * inverse[:, plan.inverse_permutation[dof]]
                        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
                        self.assertEqual(outputs[0][0, row], VELOCITY_LIMIT)
                        self.assertEqual(outputs[1][0, row], -1)
                        for index in (2, 3, 5):
                            self.assertEqual(outputs[index][0, row], 0.0)
                        self.assertAlmostEqual(outputs[4][0, row], 2e-4, places=9)
                        self.assertEqual(outputs[6][0, row], -limits[dof])
                        self.assertAlmostEqual(outputs[8][0, row], sign * velocity[dof], places=6)
                        self.assertAlmostEqual(outputs[9][0, row], expected @ expected, places=6)

    def test_response_rows_cpu(self):
        """Match an independent inverse-factor column oracle on CPU."""
        self._check_rows("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_response_rows_cuda(self):
        """Match the same producer law across cooperative CUDA lanes."""
        self._check_rows("cuda:0")


def _physical_sweep(jacobian, response, initial, bias, diagonal, kinds, parents, mu, impulses, *, iterations, omega):
    """Apply ordinary physical rows, then unrelaxed velocity-limit rows, which keep a non-negative total."""
    velocity, impulses = initial.copy(), impulses.copy()
    ordinary = np.flatnonzero(kinds != VELOCITY_LIMIT)
    mapping = {int(row): i for i, row in enumerate(ordinary)}
    ordinary_parents = np.array([mapping.get(int(parents[row]), -1) for row in ordinary], np.int32)
    for _ in range(iterations):
        velocity, ordinary_impulses = _sweep(
            jacobian[ordinary],
            response[ordinary],
            velocity,
            bias[ordinary],
            diagonal[ordinary],
            kinds[ordinary],
            ordinary_parents,
            mu[ordinary],
            impulses[ordinary],
            iterations=1,
            omega=omega,
        )
        impulses[ordinary] = ordinary_impulses
        for row in np.flatnonzero(kinds == VELOCITY_LIMIT):
            residual = jacobian[row] @ velocity + bias[row]
            total = max(impulses[row] - residual / diagonal[row], 0.0) if diagonal[row] > 0.0 else impulses[row]
            correction = total - impulses[row]
            impulses[row] = total
            velocity += response[row] * correction
    return velocity, impulses


def _solve_data(sign, *, contact, iterations, omega):
    """Put velocity slots before a coupled contact that must be visited first."""
    mass = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 0.8]])
    factor = np.linalg.cholesky(mass)
    jacobian = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    kinds = np.array([VELOCITY_LIMIT, VELOCITY_LIMIT], np.int32)
    parents = np.array([-1, -1], np.int32)
    bias = np.array([0.25, 0.25])
    mu = np.zeros(2)
    initial = np.array([sign * 2.0, 0.4, -0.2])
    seeds = np.array([3.0, 3.0])
    if contact:
        jacobian = np.vstack((jacobian, [[sign, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        kinds = np.concatenate((kinds, [CONTACT, FRICTION, FRICTION])).astype(np.int32)
        parents = np.array([-1, -1, -1, 2, 2], np.int32)
        bias = np.concatenate((bias, [-1.0, 0.0, 0.0]))
        mu = np.array([0.0, 0.0, 0.0, 0.3, 0.3])
        initial[0] = sign * 0.1
        seeds = np.zeros(5)
    response = np.linalg.solve(mass, jacobian.T).T
    z = np.linalg.solve(factor, jacobian.T).T
    diagonal = np.einsum("ij,ij->i", jacobian, response)
    count, worlds, capacity = len(kinds), 3, 8

    def padded(value, dtype=np.float32):
        result = np.zeros((worlds, capacity, *value.shape[1:]), dtype=dtype)
        result[:, : len(value)] = value
        return result

    data = {
        "world_constraint_count": np.array([count, 0, count], np.int32),
        "rhs_bias": padded(bias),
        "diagonal": padded(diagonal),
        "initial_impulses": padded(seeds),
        "row_factor": padded(z),
        "row_incident": padded(jacobian @ initial),
        "row_type": padded(kinds, np.int32),
        "row_parent": padded(parents, np.int32),
        "row_mu": padded(mu),
        "support_row_class": np.zeros((worlds, capacity), np.int32),
        "support_class_dof": np.arange(3, dtype=np.int32)[None],
        "iterations": np.array(iterations),
        "omega": np.array(omega),
        "friction_start_iteration": np.array(0),
        "iteration_offset": np.array(0),
        "reference_factor_delta": np.zeros((worlds, 3), np.float32),
    }
    physical = (jacobian, response, initial, bias, diagonal, kinds, parents, mu, seeds)
    return data, physical, factor


def _assert_ordinary_route(test, case):
    """Require the reference arm to run ordinary factors, dynamics and matrix-free sweeps."""
    selected = case[1].get_path_selection()
    test.assertEqual(selected["sparse_mass_matrix"], "off")
    test.assertEqual(selected["cached_dynamics"], "off")
    test.assertIsNone(selected["routes"]["sweep"]["tiers"])


def _case(
    enabled,
    *,
    floating=True,
    branches=3,
    depth=1,
    activation_fraction=0.0,
    friction_anchor_beta=0.0,
):
    model = sparse_scene.branched_model(floating=floating, branches=branches, depth=depth)
    limits = np.full(model.joint_dof_count, 0.25, np.float32)
    limits.reshape(2, -1)[:, -1] = np.inf
    model.joint_velocity_limit.assign(limits)
    # Both arms keep one buffer set, so the rows compared after a step are still there.
    pins = {"double_buffer": False, **({} if enabled else {"storage": "dense", "evaluator": "ordinary"})}
    with mock.patch.dict(SolverFeatherPGS._route_pins, pins):
        solver = SolverFeatherPGS(
            model,
            friction_anchor_beta=friction_anchor_beta,
            enable_joint_limits=True,
            enable_joint_velocity_limits=True,
            velocity_limit_activation_fraction=activation_fraction,
            pgs_iterations=8,
            pgs_warmstart=False,
            dense_max_constraints=32,
            mf_max_constraints=32,
            update_mass_matrix_interval=2,
            restitution_velocity_threshold=0.01,
        )
    state, out, control = model.state(), model.state(), model.control()
    qd = state.joint_qd.numpy().copy().reshape(2, -1)
    qd[1] *= -1.0
    state.joint_qd.assign(qd.reshape(-1))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control.joint_f.assign(np.linspace(-0.08, 0.11, model.joint_dof_count, dtype=np.float32))
    # Sorted contacts: both arms see the same contact order, which the by-contact patch checks key on.
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, deterministic=True)
    return model, solver, state, out, control, pipeline, pipeline.contacts()


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSparseVelocityLimitPhysics(unittest.TestCase):
    def _assert_state(self, reference, candidate):
        _assert_ordinary_route(self, reference)
        sparse_tests.TestFeatherPGSSparseSolver.assert_states_close(self, reference[2], candidate[2])
        for case in (reference, candidate):
            self.assertFalse(case[1].placement.constraint_overflow.numpy().any())
            for name in sparse_scene.STATE_FIELDS:
                self.assertTrue(np.isfinite(getattr(case[2], name).numpy()).all(), name)
        np.testing.assert_array_equal(candidate[1]._sparse_mass_matrix_status.numpy(), 0)
        if candidate[1].laws.friction_anchors_enabled:
            # Called unbound so that suites reusing this check (mimic rows) inherit it.
            TestSparseVelocityLimitPhysics._assert_patch(self, reference, candidate)

    def _assert_patch(self, reference, candidate):
        """Match patch rows and contact impulses, the pooled Coulomb bound and the anchor lifecycle."""
        self.assertTrue(reference[1].laws.friction_anchors_enabled)
        # Keyed by contact: ring parents and patch rows are slots of per-world atomic reservations.
        rows = contact_rows(reference[1], reference[-1])
        candidate_rows = contact_rows(candidate[1], candidate[-1])
        assert_rows_close(rows, candidate_rows)
        assert_contact_impulses_close(rows, candidate_rows)
        for case in (reference, candidate):
            assert_ring_coulomb(case[1])
        assert_history_close(reference[1], candidate[1])

    def _assert_rows(self, reference, candidate):
        """Compare signed physical response, final residual and live speed-limit targets."""
        old, new = reference[1], candidate[1]
        counts = new.placement.constraint_count.numpy()
        np.testing.assert_array_equal(counts, old.placement.constraint_count.numpy())
        old_j, old_y = unpacked_rows(old)
        for world, row_count in enumerate(counts):
            count = int(row_count)
            z, jacobian = sparse_tests.TestFeatherPGSSparseSolver._sparse_rows_physical(self, new, world, count)
            np.testing.assert_allclose(jacobian, old_j[world, :count], atol=3e-5, rtol=3e-4)
            np.testing.assert_allclose(z @ z.T, old_j[world, :count] @ old_y[world, :count].T, atol=3e-5, rtol=3e-4)
            # Patch ring parents name other contacts' slots; _assert_patch compares them by contact.
            for name in ("row_type",) if old.laws.friction_anchors_enabled else ("row_type", "row_parent"):
                np.testing.assert_array_equal(
                    getattr(new.placement, name).numpy()[world, :count],
                    getattr(old.placement, name).numpy()[world, :count],
                )
            for name in ("diag", "rhs"):
                np.testing.assert_allclose(
                    getattr(new.placement, name).numpy()[world, :count],
                    getattr(old.placement, name).numpy()[world, :count],
                    atol=3e-5,
                    rtol=3e-4,
                )
            finite = np.isfinite(new.placement.rhs.numpy()[world, :count])
            dofs = new.forest.world_dof_indices.numpy()[world]
            residual = jacobian @ new.v_out.numpy()[dofs] + new.placement.rhs.numpy()[world, :count]
            old_residual = old_j[world, :count] @ old.v_out.numpy()[dofs] + old.placement.rhs.numpy()[world, :count]
            np.testing.assert_allclose(residual[finite], old_residual[finite], atol=5e-5, rtol=5e-4)

    def test_last_phase_and_physical_response(self):
        """Accumulate unrelaxed velocity-limit totals in the last phase, after contact-first ordering.

        Velocity-limit rows enter every solve with zero impulse, so the rows start
        cold here. A second iteration keeps the first one's total.
        """
        for sign in (-1.0, 1.0):
            for contact, iterations in ((False, 1), (False, 2), (True, 1), (True, 8)):
                with self.subTest(sign=sign, contact=contact, iterations=iterations):
                    data, physical, factor = _solve_data(sign, contact=contact, iterations=iterations, omega=0.4)
                    data["initial_impulses"][:] = 0.0
                    physical = (*physical[:-1], np.zeros_like(physical[-1]))

                    def factory(capacity, dofs, width, **_kwargs):
                        return sparse_kernel(
                            capacity,
                            dofs,
                            width,
                            active_rows=False,
                            use_support_table=True,
                            velocity_limits=True,
                        )

                    with mock.patch.object(triplet_tests, "factory", factory):
                        solve = triplet_tests.DeviceSolve(data, candidate=True)
                    solve.launch()
                    delta, impulses = solve.result()
                    expected_velocity, expected_impulses = _physical_sweep(*physical, iterations=iterations, omega=0.4)
                    for world in (0, 2):
                        velocity = physical[2] + np.linalg.solve(factor.T, delta[world])
                        np.testing.assert_allclose(velocity, expected_velocity, atol=3e-5, rtol=3e-4)
                        np.testing.assert_allclose(
                            impulses[world, : len(expected_impulses)], expected_impulses, atol=3e-5, rtol=3e-4
                        )
                        self.assertLessEqual(abs(velocity[0]), 0.25 + 3e-5)
                        if not contact:
                            # The violated side keeps its total: 1.75 speed units over the response.
                            self.assertGreater(max(impulses[world, :2]), 0.1)
                    np.testing.assert_array_equal(delta[1], 0.0)

    def test_serial_and_branched_dense_reference(self):
        """Keep signed clamps, contacts and momentum response on independent tree shapes under either friction law."""
        for beta, shape in itertools.product(
            (0.0, 0.2), ({"floating": False, "branches": 1, "depth": 3}, {"floating": True, "branches": 3, "depth": 1})
        ):
            with self.subTest(friction_anchor_beta=beta, shape=shape):
                reference = _case(False, friction_anchor_beta=beta, **shape)
                candidate = _case(True, friction_anchor_beta=beta, **shape)
                self.assertFalse(reference[1]._sparse_velocity_limits)
                self.assertTrue(candidate[1]._sparse_velocity_limits)
                self.assertEqual(candidate[1].options.pgs_iterations, 8)
                for step in range(4):
                    sparse_scene.step(reference)
                    sparse_scene.step(candidate)
                    self._assert_state(reference, candidate)
                    if step == 0:
                        self._assert_rows(reference, candidate)
                        live = candidate[1].placement.row_type.numpy()[
                            0, : candidate[1].placement.constraint_count.numpy()[0]
                        ]
                        self.assertIn(VELOCITY_LIMIT, live)
                        self.assertIn(CONTACT, live)

    def test_held_factor_reset_and_limit_mutation_graph(self):
        """Retain factor cadence while finite, unlimited and disabled speed rows regrow."""
        self._check_held_factor_graph(0.0)

    def test_patch_held_factor_reset_and_limit_mutation_graph(self):
        """Retain the factor cadence and the patch anchor lifecycle through resets and captured limit changes."""
        self._check_held_factor_graph(0.2)

    def _check_held_factor_graph(self, friction_anchor_beta):
        reference = _case(False, friction_anchor_beta=friction_anchor_beta)
        candidate = _case(True, friction_anchor_beta=friction_anchor_beta)
        self.assertTrue(candidate[1]._sparse_velocity_limits)
        for case in (reference, candidate):
            sparse_scene.step(case)
        held = candidate[1]._sparse_Linv.numpy().copy()
        for model, solver, state, *_ in (reference, candidate):
            q = state.joint_q.numpy().copy()
            q[2] += 0.002
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (reference, candidate):
            sparse_scene.step(case)
        np.testing.assert_array_equal(candidate[1].factor.mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(candidate[1]._sparse_Linv.numpy()[1], held[1])
        self._assert_state(reference, candidate)
        with wp.ScopedCapture("cuda:0") as capture:
            sparse_scene.step(candidate)
            sparse_scene.step(candidate)
        for limit in (np.inf, 0.0, -1.0, 0.2):
            for model, solver, *_ in (reference, candidate):
                model.joint_velocity_limit.fill_(float(limit))
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            sparse_scene.step(reference)
            sparse_scene.step(reference)
            wp.capture_launch(capture.graph)
            self._assert_state(reference, candidate)
            for case in (reference, candidate):
                for world, count in enumerate(case[1].placement.constraint_count.numpy()):
                    kinds = case[1].placement.row_type.numpy()[world, :count]
                    if limit <= 0:
                        self.assertFalse(np.any(kinds == VELOCITY_LIMIT))
                    else:
                        self.assertTrue(np.any(kinds == VELOCITY_LIMIT))

    def test_activation_fraction_preserves_existing_allocation(self):
        """Keep threshold gating and the infinite-fraction disable behavior unchanged."""
        for fraction in (0.9, np.inf):
            with self.subTest(fraction=fraction):
                reference, candidate = (
                    _case(False, activation_fraction=fraction),
                    _case(True, activation_fraction=fraction),
                )
                self.assertTrue(candidate[1]._sparse_velocity_limits)
                for _ in range(2):
                    sparse_scene.step(reference)
                    sparse_scene.step(candidate)
                    self._assert_state(reference, candidate)
                    np.testing.assert_array_equal(
                        candidate[1].laws.velocity_limit_slot.numpy(), reference[1].laws.velocity_limit_slot.numpy()
                    )
                if np.isinf(fraction):
                    np.testing.assert_array_equal(candidate[1].laws.velocity_limit_slot.numpy(), -1)


if __name__ == "__main__":
    unittest.main()

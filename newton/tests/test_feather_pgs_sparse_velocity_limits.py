# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve articulated velocity clamps in factor-coordinate response."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_CONTACT as CONTACT
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_FRICTION as FRICTION
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT as VELOCITY_LIMIT
from newton._src.solvers.feather_pgs.sweep_plan import Sweep
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel
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
from newton.tests.feather_pgs_sparse_scene import (
    ForestProbe,
    TopologyReached,
    change_probe,
    eligible_solver,
)
from newton.tests.test_feather_pgs_sparse_pgs import _sweep


class TestSparseVelocityLimitAdmission(unittest.TestCase):
    def test_auto_and_enabled_velocity_owner_reach_topology(self):
        """Admit the unchanged augmented matrix-free recipe before planning its factor.

        Warm starting is admitted without free rigid bodies: the sparse sweep
        seeds its factor delta with the carried impulses' response. Material
        contacts are admitted: the sparse sweeps step them by their weights.
        """
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        recipes = ({}, {"pgs_warmstart": True}, {"contact_compliance": True})
        for recipe in recipes:
            solver = eligible_solver()
            solver.options.enable_joint_velocity_limits = True
            for key, value in recipe.items():
                change_probe(solver, key, value)
            with (
                self.subTest(recipe=recipe),
                mock.patch.dict(os.environ, {}, clear=True),
                self.assertRaises(TopologyReached),
            ):
                Sweep.admit_packed(solver, model, None)

    def test_default_and_unsupported_recipes_keep_existing_owner(self):
        """Decline unsupported laws before removing ordinary response storage."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        for changes in (
            {"pins": {"storage": "dense"}},
            {"pgs_warmstart": True, "forest": ForestProbe(has_free_rigid_bodies=True)},
            {"forest": ForestProbe(has_free_rigid_bodies=True)},
        ):
            with self.subTest(changes=changes), mock.patch.dict(os.environ, {}, clear=True):
                solver = eligible_solver()
                solver.options.enable_joint_velocity_limits = True
                for key, value in changes.items():
                    change_probe(solver, key, value)
                Sweep.admit_packed(solver, model, None)
                self.assertFalse(solver.sparse_velocity_limits)
                self.assertIsNone(solver.placement.sparse_mass_matrix_size)


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
        "reference_factor_delta": np.zeros((worlds, 3), np.float32),
    }
    physical = (jacobian, response, initial, bias, diagonal, kinds, parents, mu, seeds)
    return data, physical, factor


def _assert_ordinary_route(test, case):
    """Require the reference arm to run dense factors and matrix-free sweeps."""
    selected = case[1].get_path_selection()
    test.assertEqual(selected["sparse_mass_matrix"], "off")
    test.assertIsNone(selected["routes"]["sweep"]["tiers"])


def _case(
    enabled,
    *,
    floating=True,
    branches=3,
    depth=1,
    activation_fraction=0.0,
):
    model = sparse_scene.branched_model(floating=floating, branches=branches, depth=depth)
    limits = np.full(model.joint_dof_count, 0.25, np.float32)
    limits.reshape(2, -1)[:, -1] = np.inf
    model.joint_velocity_limit.assign(limits)
    pins = {} if enabled else {"storage": "dense"}
    with mock.patch.dict(SolverFeatherPGS._route_pins, pins):
        solver = SolverFeatherPGS(
            model,
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


def _records(solver, group: int) -> np.ndarray:
    """The walk records of the sparse group's ``group``-th articulation (the factor the walk refreshes)."""
    art = int(solver.forest.group_to_art[solver.placement.sparse_mass_matrix_size].numpy()[group])
    d0 = int(solver.forest.articulation_dof_start.numpy()[art])
    return solver.factor.walk_data.numpy()[d0 : d0 + int(solver.forest.articulation_dof_count.numpy()[art])].copy()


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSparseVelocityLimitPhysics(unittest.TestCase):
    def _assert_state(self, reference, candidate):
        _assert_ordinary_route(self, reference)
        sparse_tests.TestFeatherPGSSparseSolver.assert_states_close(self, reference[2], candidate[2])
        for case in (reference, candidate):
            self.assertFalse(case[1].placement.constraint_overflow.numpy().any())
            for name in sparse_scene.STATE_FIELDS:
                self.assertTrue(np.isfinite(getattr(case[2], name).numpy()).all(), name)
        # Called unbound so that suites reusing this check (mimic rows) inherit it.
        TestSparseVelocityLimitPhysics._assert_patch(self, reference, candidate)

    def _assert_patch(self, reference, candidate):
        """Match patch rows and contact impulses, the pooled Coulomb bound and the anchor lifecycle."""
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
            for name in ("row_type",):
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
        """Keep signed clamps, contacts and momentum response on independent tree shapes."""
        for shape in ({"floating": False, "branches": 1, "depth": 3}, {"floating": True, "branches": 3, "depth": 1}):
            with self.subTest(shape=shape):
                reference = _case(False, **shape)
                candidate = _case(True, **shape)
                self.assertFalse(reference[1].sweep.sparse_velocity_limits)
                self.assertTrue(candidate[1].sweep.sparse_velocity_limits)
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
        """Retain the factor cadence and the anchor lifecycle while finite, unlimited and disabled speed rows regrow
        through resets and captured limit changes."""
        reference = _case(False)
        candidate = _case(True)
        self.assertTrue(candidate[1].sweep.sparse_velocity_limits)
        for case in (reference, candidate):
            sparse_scene.step(case)
        held = _records(candidate[1], 1)
        for model, solver, state, *_ in (reference, candidate):
            q = state.joint_q.numpy().copy()
            q[2] += 0.002
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (reference, candidate):
            sparse_scene.step(case)
        np.testing.assert_array_equal(candidate[1].factor.mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(_records(candidate[1], 1), held)
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
                self.assertTrue(candidate[1].sweep.sparse_velocity_limits)
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

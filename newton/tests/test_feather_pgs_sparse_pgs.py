# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare sparse factor-coordinate sweeps with physical-coordinate PGS."""

import unittest
from itertools import product

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
)
from newton._src.solvers.feather_pgs.projection_laws import friction_pair
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport
from newton._src.solvers.feather_pgs.sweep_routes_sparse import sparse_kernel

_SWEEP_OMEGAS = (1.0, 0.7, 1.2)


def _ring(parents, first):
    """Return the patch ring of normal rows that starts at ``first``, in the order the pair update pools it."""
    rows, row = [int(first)], int(parents[first])
    while row >= 0 and row != first:
        rows.append(row)
        row = int(parents[row])
    return rows


def _sweep(
    jacobian,
    response,
    velocity,
    bias,
    diagonal,
    types,
    parents,
    mu,
    impulses,
    *,
    omega=1.0,
    active_rows=False,
    linked_rows=False,
    select_friction=True,
    iterations=8,
):
    """Run ordered PGS sweeps in physical coordinates.

    ``active_rows`` applies the deferral law: a sweep visits only the normal and
    limit rows that hold impulse or are violated at its entry. Without
    ``linked_rows`` a world with a ring-linked normal keeps the full sweep; with
    it a friction pair is visited while any normal of its ring is selected or the
    pair holds impulse. ``select_friction=False`` visits every friction pair
    instead, the deferral law the selection must reproduce exactly.
    """
    velocity, impulses = velocity.copy(), impulses.copy()
    active_rows = (
        active_rows
        and np.isfinite(omega)
        and omega >= 0
        and (linked_rows or not np.any((types == PGS_CONSTRAINT_TYPE_CONTACT) & (parents >= 0)))
    )
    for _ in range(iterations):
        selected = np.ones(len(types), dtype=bool)
        if active_rows:
            unilateral = np.isin(types, (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
            selected[unilateral] = (impulses[unilateral] != 0) | ((jacobian @ velocity + bias)[unilateral] < 0)
            for row in np.flatnonzero((types == PGS_CONSTRAINT_TYPE_FRICTION) & select_friction):
                parent = parents[row]
                ring = _ring(parents, parent) if linked_rows else [parent]
                selected[row] = np.any(selected[ring]) or np.any(impulses[parent + 1 : parent + 3] != 0)
        for row, kind in enumerate(types):
            if not selected[row]:
                continue
            if kind == PGS_CONSTRAINT_TYPE_FRICTION and row != parents[row] + 1:
                continue
            residual = jacobian[row] @ velocity + bias[row]
            old = impulses[row]
            if kind == PGS_CONSTRAINT_TYPE_FRICTION:
                parent, sibling = parents[row], row + 1
                load = impulses[parent]
                next_row = parents[parent]
                while next_row >= 0 and next_row != parent:
                    load += impulses[next_row]
                    next_row = parents[next_row]
                radius = max(mu[row] * load, 0.0)
                pair = friction_pair(
                    float(diagonal[row]),
                    float(jacobian[row] @ response[sibling]),
                    float(diagonal[sibling]),
                    wp.vec2(float(residual), float(jacobian[sibling] @ velocity + bias[sibling])),
                    wp.vec2(float(old), float(impulses[sibling])),
                    float(radius),
                    omega,
                )
                pair = np.asarray(pair, dtype=np.float64)
                magnitude = np.linalg.norm(pair)
                if magnitude > radius:
                    pair *= radius / magnitude
                velocity += response[sibling] * (pair[1] - impulses[sibling])
                impulses[sibling] = pair[1]
                value = pair[0]
            else:
                value = old - omega * residual / diagonal[row]
                if kind in (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT):
                    value = max(value, 0.0)
            velocity += response[row] * (value - old)
            impulses[row] = value
    return velocity, impulses


def _problem(dofs=43, support=18, seeded=False, *, contact_activity=None):
    rng = np.random.default_rng(471)
    factor = np.tril(rng.normal(scale=0.04, size=(dofs, dofs))) + np.eye(dofs)
    indices = np.sort(rng.choice(dofs, support, replace=False))
    rows = np.zeros((8, dofs))
    rows[:, indices] = rng.normal(scale=0.3, size=(8, support))
    rows[0] = 0.0
    rows[0, indices[0]] = 0.8
    initial = rng.normal(scale=0.2, size=dofs)
    bias = np.array((-0.2, -0.1, 0.04, -0.03, -0.12, -0.15, 0.02, 0.04))
    types = np.array(
        (
            PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
        ),
        dtype=np.int32,
    )
    parents = np.array((-1, 4, 1, 1, 1, -1, 5, 5), dtype=np.int32)
    mu = np.array((0, 0, 0.6, 0.6, 0, 0, 0.4, 0.4), dtype=np.float64)
    impulses = np.array((0, 0.4, 0.02, -0.01, 0.3, 0.2, -0.03, 0.01)) if seeded else np.zeros(8)
    if contact_activity is not None:
        factor[:] = np.eye(dofs)
        rows.fill(0.0)
        rows[:3, :3] = np.eye(3)
        initial.fill(0.0)
        bias.fill(0.0)
        bias[:3] = (0.25, 1.0, -0.25)
        types[:] = PGS_CONSTRAINT_TYPE_CONTACT
        types[1:3] = PGS_CONSTRAINT_TYPE_FRICTION
        parents.fill(-1)
        parents[1:3] = 0
        mu.fill(0.0)
        mu[1:3] = 0.6
        impulses.fill(0.0)
        if contact_activity == "late_activation":
            # A later normal row activates the initially unloaded pair next sweep.
            rows[3, :3] = (-1.0, 0.0, 1.0)
            bias[3] = -1.0
        elif contact_activity in ("deferred_activation", "linked_deferral"):
            # An earlier normal makes a later zero normal eligible only next sweep.
            rows[0, 0], rows[3, 0] = -1.0, 1.0
            bias[0], bias[3] = -1.0, 0.25
            if contact_activity == "linked_deferral":
                # Both normals share one patch ring, so the late normal also loads the pair.
                parents[0], parents[3] = 3, 0
        elif contact_activity == "near_singular":
            rows[2, 1:3] = (1.0, 0.001)
            bias[:3] = (-1.0, -1.0, 0.25)
        elif contact_activity == "negative_omega":
            # Negative relaxation changes zero normals with positive residuals.
            mu.fill(0.0)
        elif contact_activity == "seeded_retraction":
            # Zero load must retract both signs of old tangent impulses.
            impulses[1:3] = (-0.25, 0.125)
        elif contact_activity == "linked_load":
            # The anchor stays unloaded, but a linked normal supplies friction.
            rows[3, 2] = 1.0
            bias[3] = -1.0
            parents[0], parents[3] = 3, 0
            impulses[3] = 0.5
        elif contact_activity == "linked_far_ring":
            # The anchor and the next normal of its ring separate; only the third ring
            # member is violated at sweep entry, so only a whole ring walk visits the pair.
            rows[4, 2] = 1.0
            bias[3], bias[4] = 0.25, -1.0
            parents[0], parents[3], parents[4] = 3, 4, 0
        elif contact_activity == "linked_long_ring":
            # Patch regions have no member limit: all six normals form one ring, longer than a
            # box foot's four corners. Only the last is violated at sweep entry, so a walk cut
            # before it, in the selection or in the pooled load, leaves the pair's friction at zero.
            rows[7, 2] = 1.0
            bias[3:7], bias[7] = 0.25, -1.0
            parents[[0, 3, 4, 5, 6, 7]] = (3, 4, 5, 6, 7, 0)
        else:
            raise ValueError(f"Unknown contact activity: {contact_activity}")
    jacobian = rows @ factor.T
    response = np.linalg.solve(factor.T, rows.T).T
    incident = jacobian @ initial
    diagonal = np.sum(rows * rows, axis=1) + 1.0e-6
    return factor, indices, rows, initial, bias, types, parents, mu, impulses, jacobian, response, incident, diagonal


def _patch_problem(rng, *, dofs=10, regions=4, limits=1):
    """Build random patch rows: limit rows, then interleaved one-row normal and three-row anchor units.

    Each region links its normals in one ring (a lone normal links to itself or
    to nothing). Responses equal the rows, as in factor coordinates. Normals and
    tangents start at zero or seeded; biases leave some normals separating, some
    violated and some close enough for an earlier update to violate them.
    """
    units = []
    for region in range(regions):
        contacts = int(rng.integers(1, 5))
        anchors = rng.choice(contacts, size=int(rng.integers(0, min(contacts, 2) + 1)), replace=False)
        units += [(region, contact in anchors) for contact in range(contacts)]
    rng.shuffle(units)
    types = [PGS_CONSTRAINT_TYPE_JOINT_LIMIT] * limits
    parents, mu, members = [-1] * limits, [0.0] * limits, [[] for _ in range(regions)]
    for region, anchor in units:
        normal = len(types)
        members[region].append(normal)
        types.append(PGS_CONSTRAINT_TYPE_CONTACT)
        parents.append(-1)
        mu.append(0.0)
        if anchor:
            types += [PGS_CONSTRAINT_TYPE_FRICTION] * 2
            parents += [normal] * 2
            mu += [rng.uniform(0.2, 0.8)] * 2
    types, parents, mu = np.array(types, np.int32), np.array(parents, np.int32), np.array(mu)
    for ring in members:
        if len(ring) > 1 or rng.random() < 0.5:
            parents[ring] = np.roll(ring, -1)
    rows = rng.normal(scale=0.5, size=(len(types), dofs))
    bias = rng.uniform(-0.3, 0.3, size=len(types))
    impulses = np.where(rng.random(len(types)) < 0.3, rng.uniform(0.0, 0.2, size=len(types)), 0.0)
    friction = types == PGS_CONSTRAINT_TYPE_FRICTION
    impulses[friction] = np.where(rng.random(friction.sum()) < 0.2, rng.normal(scale=0.02, size=friction.sum()), 0.0)
    diagonal = np.sum(rows * rows, axis=1) + 1.0e-6
    return rows, bias, diagonal, types, parents, mu, impulses


class TestFeatherPGSSparsePGS(unittest.TestCase):
    def test_support_table_factory(self):
        """Build the same ordered solver with shared topology indices."""
        kernel = sparse_kernel(16, 6, 4, active_rows=True, use_support_table=True)
        self.assertIn("support_table", kernel.key)

    def test_factory_rejects_free_body_rows(self):
        """Reject physical free-body rows, which native sparse factors never hold."""
        for options in ({"mf_max_constraints": 1}, {"free_row_dofs": 1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "free-body rows"):
                sparse_kernel(16, 6, 4, **options)

    def test_factor_coordinate_residual_and_cross_term(self):
        """Preserve physical residuals and paired-tangent effective mass."""
        for dofs, support in ((7, 6), (43, 18), (75, 39)):
            with self.subTest(dofs=dofs, support=support):
                factor, _, rows, initial, _, _, _, _, _, jacobian, response, incident, _ = _problem(dofs, support)
                delta = np.linspace(-0.3, 0.2, dofs)
                physical = initial + np.linalg.solve(factor.T, delta)
                np.testing.assert_allclose(jacobian @ physical, incident + rows @ delta, atol=1.0e-12)
                np.testing.assert_allclose(jacobian @ response.T, rows @ rows.T, atol=1.0e-12)

    def test_factor_sweeps_match_physical_sweeps(self):
        """Preserve joint limits, pooled friction loads and seeded-impulse semantics."""
        for seeded, omega in product((False, True), _SWEEP_OMEGAS):
            with self.subTest(seeded=seeded, omega=omega):
                factor, _, rows, initial, bias, types, parents, mu, impulses, jacobian, response, incident, diag = (
                    _problem(seeded=seeded)
                )
                settings = {"omega": omega}
                expected, expected_impulses = _sweep(
                    jacobian, response, initial, bias, diag, types, parents, mu, impulses, **settings
                )
                delta, actual_impulses = _sweep(
                    rows, rows, np.zeros_like(initial), incident + bias, diag, types, parents, mu, impulses, **settings
                )
                actual = initial + np.linalg.solve(factor.T, delta)
                np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=2.0e-5)
                np.testing.assert_allclose(actual_impulses, expected_impulses, atol=2.0e-6, rtol=2.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_sparse_sweeps_match_reference(self):
        """Preserve sweeps, zero-load retraction, linked loads and late contact activation."""
        self._check_cuda_sparse_sweeps(active_rows=False)

    def test_active_rows_defer_new_violation_to_next_sweep(self):
        """Distinguish sweep-start eligibility from immediate ordered activation."""
        _, _, rows, initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(
            3, 3, contact_activity="deferred_activation"
        )
        args = (rows, rows, np.zeros_like(initial), incident + bias, diag, types, parents, mu, impulses)
        _, exact = _sweep(*args, iterations=1)
        _, deferred = _sweep(*args, active_rows=True, iterations=1)
        _, next_sweep = _sweep(*args, active_rows=True, iterations=2)
        self.assertGreater(exact[3], 0.0)
        self.assertEqual(deferred[3], 0.0)
        self.assertGreater(next_sweep[3], 0.0)
        exact = _sweep(*args, omega=-0.5)
        fallback = _sweep(*args, omega=-0.5, active_rows=True)
        np.testing.assert_array_equal(fallback[0], exact[0])
        np.testing.assert_array_equal(fallback[1], exact[1])

    def test_linked_active_rows_defer_patch_rings(self):
        """Defer a newly violated ring normal under linked selection, where point selection keeps the full sweep."""
        _, _, rows, initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(
            3, 3, contact_activity="linked_deferral"
        )
        self.assertEqual(_ring(parents, 0), [0, 3])
        args = (rows, rows, np.zeros_like(initial), incident + bias, diag, types, parents, mu, impulses)
        _, exact = _sweep(*args, iterations=1)
        _, point = _sweep(*args, active_rows=True, iterations=1)
        _, linked = _sweep(*args, active_rows=True, linked_rows=True, iterations=1)
        _, next_sweep = _sweep(*args, active_rows=True, linked_rows=True, iterations=2)
        # A ring-linked world falls back to the full sweep without linked selection.
        np.testing.assert_array_equal(point, exact)
        self.assertGreater(exact[3], 0.0)
        self.assertEqual(linked[3], 0.0)
        self.assertGreater(next_sweep[3], 0.0)
        np.testing.assert_array_equal(linked[:3], exact[:3])
        exact = _sweep(*args, omega=-0.5)
        fallback = _sweep(*args, omega=-0.5, active_rows=True, linked_rows=True)
        np.testing.assert_array_equal(fallback[0], exact[0])
        np.testing.assert_array_equal(fallback[1], exact[1])

    def test_linked_selection_matches_the_deferral_law(self):
        """Skip only friction pairs whose deferral-law update is a no-op, bitwise.

        Random patch rows interleave one-row normals and three-row anchor units in
        rings of one to four normals (self-loops and unlinked normals included),
        with zero and seeded impulses and relaxation. Selecting a
        pair while any normal of its ring is selected, or while it holds impulse,
        equals visiting every pair. The rows include pairs whose own normal is
        unselected at sweep entry while another normal of its ring is selected, so
        a selection by the pair's own normal alone would drop their load.
        """
        rng = np.random.default_rng(13)
        ring_loaded = 0
        for case in range(120):
            rows, bias, diagonal, types, parents, mu, impulses = _patch_problem(rng)
            settings = {"omega": _SWEEP_OMEGAS[case % len(_SWEEP_OMEGAS)]}
            args = (rows, rows, np.zeros(rows.shape[1]), bias, diagonal, types, parents, mu, impulses)
            with self.subTest(case=case):
                selected = _sweep(*args, active_rows=True, linked_rows=True, **settings)
                every_pair = _sweep(*args, active_rows=True, linked_rows=True, select_friction=False, **settings)
                np.testing.assert_array_equal(selected[0], every_pair[0])
                np.testing.assert_array_equal(selected[1], every_pair[1])
            entry = (impulses != 0) | (bias < 0)
            for row in np.flatnonzero(types == PGS_CONSTRAINT_TYPE_FRICTION):
                parent = parents[row]
                ring_loaded += row == parent + 1 and not entry[parent] and bool(np.any(entry[_ring(parents, parent)]))
        self.assertGreater(ring_loaded, 20)

    def test_linked_active_rows_differ_from_the_full_sweep_by_deferral(self):
        """Match the full patch sweep bitwise unless a sweep's earlier updates violate an unselected row.

        The deferral law is not float reordering: such a row, and the pairs its ring
        loads, wait for the next sweep. Stepping the full sweep one sweep at a time, a
        sweep that the deferral law changes is a deferral. Problems without one end
        bitwise equal after eight sweeps; every problem with one differs.
        """
        rng = np.random.default_rng(29)
        deferred = 0
        for case in range(60):
            rows, bias, diagonal, types, parents, mu, impulses = _patch_problem(rng)
            settings = {"omega": _SWEEP_OMEGAS[case % len(_SWEEP_OMEGAS)]}
            velocity, steps, deferrals = np.zeros(rows.shape[1]), impulses, 0
            for _ in range(8):
                args = (rows, rows, velocity, bias, diagonal, types, parents, mu, steps)
                once = {**settings, "iterations": 1}
                full = _sweep(*args, **once)
                lazy = _sweep(*args, active_rows=True, linked_rows=True, **once)
                deferrals += not (np.array_equal(full[0], lazy[0]) and np.array_equal(full[1], lazy[1]))
                velocity, steps = full
            args = (rows, rows, np.zeros(rows.shape[1]), bias, diagonal, types, parents, mu, impulses)
            full = _sweep(*args, **settings)
            lazy = _sweep(*args, active_rows=True, linked_rows=True, **settings)
            same = np.array_equal(full[0], lazy[0]) and np.array_equal(full[1], lazy[1])
            with self.subTest(case=case, deferrals=deferrals):
                self.assertEqual(same, deferrals == 0)
            deferred += deferrals > 0
        self.assertGreater(deferred, 10)
        self.assertLess(deferred, 50)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_active_sweeps_match_reference(self):
        """Match independent eligibility, seeded retraction and linked fallback."""
        self._check_cuda_sparse_sweeps(active_rows=True)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_support_table_sweeps(self):
        """Match packed-index sweeps and graph replay for limits, seeded impulses and contact fallback."""
        for active_rows in (False, True):
            self._check_cuda_sparse_sweeps(active_rows=active_rows, use_support_table=True)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_subwarp_sweeps_match_reference(self):
        """Preserve linked friction and seeded retraction."""
        for lanes in (8, 16):
            self._check_cuda_sparse_sweeps(active_rows=False, use_support_table=True, lanes_per_world=lanes)
            self._check_cuda_sparse_sweeps(
                active_rows=False, use_support_table=True, lanes_per_world=lanes, local_factors=True
            )

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_linked_active_sweeps_match_reference(self):
        """Select patch rings at sweep entry with seeded impulses, word-crossing rings and replay.

        Rings loaded only through their third or their sixth member catch a ring walk, in
        the pair selection or in its pooled load, that stops after up to five members.
        Worlds without a linked normal also match the point active kernel bitwise.
        """
        for use_support_table in (False, True):
            with self.subTest(use_support_table=use_support_table):
                self._check_cuda_sparse_sweeps(active_rows=True, use_support_table=use_support_table, linked_rows=True)

    def _check_cuda_sparse_sweeps(
        self, *, active_rows, use_support_table=False, lanes_per_world=32, local_factors=False, linked_rows=False
    ):
        """Compare each schedule with its own physical recurrence without weakening the exact control."""
        device = wp.get_device("cuda:0")
        cases = [(43, 18, None), (75, 39, None)]
        cases.extend((3, 3, activity) for activity in ("late_activation", "seeded_retraction", "linked_load"))
        if active_rows:
            cases.extend((3, 3, activity) for activity in ("deferred_activation", "near_singular", "negative_omega"))
        if linked_rows:
            cases.extend((3, 3, activity) for activity in ("linked_deferral", "linked_far_ring", "linked_long_ring"))
        for dofs, support, contact_activity in cases:
            if lanes_per_world < 32 and dofs > lanes_per_world:
                continue
            if use_support_table and dofs > 64:
                continue
            settings = product((False, True), _SWEEP_OMEGAS) if contact_activity is None else [(False, 1.0)]
            if contact_activity == "negative_omega":
                settings = [(False, -0.5)]
            for seeded, omega in settings:
                with self.subTest(
                    dofs=dofs,
                    support=support,
                    contact_activity=contact_activity,
                    seeded=seeded,
                    omega=omega,
                ):
                    _, indices, rows, initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(
                        dofs, support, seeded, contact_activity=contact_activity
                    )
                    if active_rows and contact_activity is None and not linked_rows:
                        parents[types == PGS_CONSTRAINT_TYPE_CONTACT] = -1
                    offset = 30 if active_rows and (dofs == 75 or (use_support_table and dofs == 43)) else 0
                    if offset:
                        # Cross a bitset word boundary and retain an incomplete final word.
                        rows = np.pad(rows, ((offset, 0), (0, 0)))
                        bias, incident = np.pad(bias, (offset, 0), constant_values=1.0), np.pad(incident, (offset, 0))
                        diag, impulses = np.pad(diag, (offset, 0), constant_values=1.0), np.pad(impulses, (offset, 0))
                        types = np.pad(types, (offset, 0), constant_values=PGS_CONSTRAINT_TYPE_CONTACT)
                        parents = np.pad(np.where(parents >= 0, parents + offset, -1), (offset, 0), constant_values=-1)
                        mu = np.pad(mu, (offset, 0))
                    count = len(types)
                    expected, expected_impulses = _sweep(
                        rows,
                        rows,
                        np.zeros_like(initial),
                        incident + bias,
                        diag,
                        types,
                        parents,
                        mu,
                        impulses,
                        omega=omega,
                        active_rows=active_rows,
                        linked_rows=linked_rows,
                    )
                    if contact_activity == "seeded_retraction":
                        np.testing.assert_array_equal(expected_impulses[1:3], 0.0)
                        np.testing.assert_allclose(expected[1:3], -impulses[1:3], atol=1.0e-12)
                    elif contact_activity == "negative_omega":
                        self.assertGreater(expected_impulses[0], 0.0)
                    elif contact_activity is not None:
                        self.assertGreater(np.linalg.norm(expected_impulses[1:3]), 0.0)
                        if contact_activity == "linked_load":
                            self.assertEqual(expected_impulses[0], 0.0)
                        elif contact_activity == "linked_far_ring":
                            # The pair's load comes from the ring's third member alone.
                            np.testing.assert_array_equal(expected_impulses[[0, 3]], 0.0)
                            self.assertGreater(expected_impulses[4], 0.0)
                        elif contact_activity == "linked_long_ring":
                            # The pair's load comes from the ring's sixth member alone.
                            self.assertEqual(_ring(parents, 0), [0, 3, 4, 5, 6, 7])
                            np.testing.assert_array_equal(expected_impulses[[0, 3, 4, 5, 6]], 0.0)
                            self.assertGreater(expected_impulses[7], 0.0)
                    worlds, capacity = 3, count + 2
                    row_dof = np.full((worlds, capacity, support), -1, dtype=np.int32)
                    row_dof[:, :count] = indices
                    row_dof[:, offset, 1:] = -1
                    row_factor = np.zeros((worlds, capacity, support), dtype=np.float32)
                    row_factor[:, :count] = rows[:, indices]

                    def padded(array, dtype=np.float32, shape=(worlds, capacity)):
                        result = np.zeros(shape, dtype=dtype)
                        result[:, : len(array)] = array
                        return wp.array(result, device=device)

                    initial_impulses = padded(impulses)
                    actual_impulses = wp.clone(initial_impulses)
                    delta = wp.full((worlds, dofs), float("nan"), device=device)
                    row_support = _SparseRowSupport()
                    packed_indices = wp.array(row_dof, device=device)
                    if use_support_table:
                        unique, row_class = np.unique(row_dof.reshape(-1, support), axis=0, return_inverse=True)
                        row_support.enabled = 1
                        row_support.class_dof = wp.array(unique, dtype=int, device=device)
                        row_support.row_class = wp.array(row_class.reshape(worlds, capacity), dtype=int, device=device)
                        packed_indices = wp.full((1, 1, 1), -9, dtype=int, device=device)
                    kernel = sparse_kernel(
                        capacity,
                        dofs,
                        support,
                        active_rows=active_rows,
                        use_support_table=use_support_table,
                        lanes_per_world=lanes_per_world,
                        local_factors=local_factors,
                        linked_rows=linked_rows,
                    )
                    self.assertEqual("_linked" in kernel.key, linked_rows and active_rows)
                    launch_args = {
                        "dim": [2],
                        "inputs": [
                            worlds,
                            wp.array([count, 0, count], dtype=int, device=device),
                            padded(bias),
                            padded(diag),
                            actual_impulses,
                            packed_indices,
                            wp.array(row_factor, device=device),
                            padded(incident),
                            padded(types, np.int32),
                            padded(parents, np.int32),
                            padded(mu),
                            wp.array(np.arange(worlds * dofs).reshape(worlds, dofs), dtype=int, device=device),
                            wp.zeros(worlds * dofs, device=device),
                            wp.zeros(worlds, dtype=int, device=device),
                            wp.zeros((worlds, 4), dtype=int, device=device),
                            wp.zeros((worlds, 1), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1), device=device),
                            8,
                            omega,
                        ],
                        "outputs": [delta, row_support],
                        "block_dim": 64,
                        "device": device,
                    }
                    wp.launch_tiled(kernel, **launch_args)
                    actual = delta.numpy()
                    if use_support_table:
                        np.testing.assert_array_equal(packed_indices.numpy(), -9)
                    result_impulses = actual_impulses.numpy()
                    np.testing.assert_array_equal(actual[1], 0.0)
                    for world in (0, 2):
                        np.testing.assert_allclose(actual[world], expected, atol=2.0e-5, rtol=2.0e-4)
                        np.testing.assert_allclose(
                            result_impulses[world, :count], expected_impulses, atol=2.0e-5, rtol=2.0e-4
                        )
                    if linked_rows and not np.any((types == PGS_CONSTRAINT_TYPE_CONTACT) & (parents >= 0)):
                        # Without a linked normal the ring walk selects as the point scan does.
                        point_kernel = sparse_kernel(
                            capacity,
                            dofs,
                            support,
                            active_rows=active_rows,
                            use_support_table=use_support_table,
                            lanes_per_world=lanes_per_world,
                            local_factors=local_factors,
                        )
                        self.assertNotIn("_linked", point_kernel.key)
                        wp.copy(actual_impulses, initial_impulses)
                        wp.launch_tiled(point_kernel, **launch_args)
                        np.testing.assert_array_equal(delta.numpy(), actual)
                        np.testing.assert_array_equal(actual_impulses.numpy(), result_impulses)
                    if use_support_table:
                        packed_kernel = sparse_kernel(
                            capacity,
                            dofs,
                            support,
                            active_rows=active_rows,
                            lanes_per_world=lanes_per_world,
                            local_factors=local_factors,
                            linked_rows=linked_rows,
                        )
                        wp.copy(actual_impulses, initial_impulses)
                        launch_args["inputs"][5] = wp.array(row_dof, device=device)
                        launch_args["outputs"][1] = _SparseRowSupport()
                        wp.launch_tiled(packed_kernel, **launch_args)
                        np.testing.assert_array_equal(delta.numpy(), actual)
                        np.testing.assert_array_equal(actual_impulses.numpy(), result_impulses)
                        launch_args["inputs"][5] = packed_indices
                        launch_args["outputs"][1] = row_support
                        with wp.ScopedCapture(device=device) as capture:
                            wp.copy(actual_impulses, initial_impulses)
                            wp.launch_tiled(kernel, **launch_args)
                        for _ in range(2):
                            wp.capture_launch(capture.graph)
                            np.testing.assert_array_equal(delta.numpy(), actual)
                            np.testing.assert_array_equal(actual_impulses.numpy(), result_impulses)


if __name__ == "__main__":
    unittest.main()

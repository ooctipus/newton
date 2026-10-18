# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve persistent patch friction in the parallel matrix-free tiers.

The parallel tiers update all rows of a world together (projected Jacobi with
per-row step scales and Nesterov momentum). Under patch friction a contact's
normal row links the next normal of its patch ring, and each tangent pair is
projected into the disk of radius ``mu * (sum of its ring's normal impulses)``.
The reference below repeats the kernel's sweep in float64 so the CUDA kernels
can be checked row by row. Keyed rows write the patch row law themselves; the
CPU fallback checks it against the general producer and the patch link.
"""

import inspect
import itertools
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import keyed_rows, path_selection
from newton._src.solvers.feather_pgs import solver_feather_pgs as source
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    compute_world_contact_bias,
)
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import assert_loaded_anchors, assert_ring_coulomb, standing_model

_DOFS = 6
_CAPACITY = 64
# spread_rows seed that puts all three rings of the two-warp case across row 32.
_SPREAD_SEED = 0


def ring_load(parents, impulses, normal, *, linked=True):
    """Sum the impulses of the patch ring that starts at ``normal``, in the kernels' ring order."""
    load = impulses[normal]
    if not linked:
        return load
    row = parents[normal]
    while 0 <= row < len(parents) and row != normal:
        load += impulses[row]
        row = parents[row]
    return load


def _sibling(row, parent):
    return parent + 2 if row == parent + 1 else parent + 1


def parallel_sweep_reference(
    case, *, matrix_free, inkernel, exact, sweeps, nesterov, tol, linked, omega=1.0, friction_start=0
):
    """Repeat one world of the parallel tier's sweep in float64.

    Returns the impulses and the generalized velocity the kernel publishes. The
    response block is ``A = J H^-1 J^T``. The step scale of row ``i`` is
    ``min(A_ii / bound_i, 1) * omega / diag_i``: the bound is the row sum of
    ``|A|`` over admitted rows (A-based tiers and exact row sums) or the
    row-sum bound ``sum_d |P_id| sum_j |Q_jd|`` of the matrix-free tiers, with
    ``(P, Q) = (J, Y)`` or the whitened rows ``(Z, Z)`` of the in-kernel
    response. Every sweep steps all rows from the extrapolated impulses, clamps
    normal and limit rows, scales each tangent pair into its disk (the ring's
    pooled normal load when ``linked``), restarts momentum against the last step
    and stops once no row changes.
    """
    J, Y, Z = case["J"], case["Y"], case["Z"]
    kinds, parents, mu = case["kinds"], case["parents"], case["mu"]
    lam0, v = case["impulses"].astype(np.float64), case["velocity"].astype(np.float64)
    A = J @ Y.T
    if inkernel:
        diag = np.diag(A) + case["cfm"]
        P = Q = Z
    else:
        diag = case["diag"].astype(np.float64)
        P, Q = J, Y
    kinds = np.where((kinds >= 0) & (diag > 0.0), kinds, -1)
    admitted = kinds >= 0
    if matrix_free and not exact:
        bound = np.abs(P) @ np.abs(Q[admitted]).sum(axis=0)
    else:
        bound = np.abs(A[:, admitted]).sum(axis=1)
    weight = np.where(bound > 0.0, np.minimum(np.diag(A) / np.where(bound > 0.0, bound, 1.0), 1.0), 0.0)
    step = np.where(admitted, weight * omega / np.where(diag > 0.0, diag, 1.0), 0.0)
    b = case["rhs"] + J @ v - A @ lam0
    x, y, t = lam0.copy(), lam0.copy(), 1.0
    for sweep in range(sweeps):
        residual = b + A @ y
        stepped = x.copy()
        for row in np.flatnonzero(admitted):
            value = y[row] - step[row] * residual[row]
            if kinds[row] == 0:
                value = max(value, 0.0)
            elif kinds[row] == 1 and sweep < friction_start:
                value = 0.0
            stepped[row] = value
        projected = stepped.copy()
        for row in np.flatnonzero(kinds == 1):
            parent = parents[row]
            radius = max(mu[row] * ring_load(parents, stepped, parent, linked=linked), 0.0)
            magnitude = math.hypot(stepped[row], stepped[_sibling(row, parent)])
            if radius <= 0.0:
                projected[row] = 0.0
            elif magnitude > radius:
                projected[row] = stepped[row] * radius / magnitude
        if np.dot(y - projected, projected - x) > 0.0:
            t = 1.0
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_next if nesterov else 0.0
        t = t_next
        changed = np.any(np.abs(projected - x) > tol * (np.abs(projected) + tol))
        x, y = projected, projected + beta * (projected - x)
        if sweep >= friction_start and not changed:
            break
    return x, v + Y.T @ (x - lam0)


def _box_contact_rows(rng, ring_base, *, linked):
    """Rows of one box face: a limit row, a four-normal patch with two anchors and three loose contacts.

    Returns ``(directions, points, kinds, types, parents, mu, rhs)``. The patch
    normals link circularly (``ring_base`` offsets row indices); its anchors
    hold half the coefficient, as the patch link divides it. Without ``linked``
    every normal parent is -1, the point friction law.
    """
    up, x_axis, y_axis = np.array((0.0, 0.0, 1.0)), np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))
    rows = []

    def contact(point, tangents, mu):
        first = ring_base + len(rows)
        rows.append((up, point, 0, PGS_CONSTRAINT_TYPE_CONTACT, -1, 0.9))
        if tangents:
            for direction in (x_axis, y_axis):
                rows.append((direction, point, 1, PGS_CONSTRAINT_TYPE_FRICTION, first, mu))
        return first

    rows.append((None, None, 0, PGS_CONSTRAINT_TYPE_JOINT_LIMIT, -1, 0.0))
    corners = [
        np.array((sx * 0.1, sy * 0.08, -0.05)) + rng.normal(0.0, 0.004, 3)
        for sx, sy in itertools.product((-1, 1), repeat=2)
    ]
    patch = [contact(corners[0], True, 0.4), contact(corners[1], False, 0.0)]
    patch += [contact(corners[2], True, 0.4), contact(corners[3], False, 0.0)]
    contact(np.array((0.0, 0.02, -0.05)), True, 0.8)  # a contact without a patch keeps parent -1
    single = contact(np.array((0.05, -0.02, -0.05)), True, 0.8)  # a patch with one live member links itself
    contact(np.array((-0.03, 0.0, -0.05)), False, 0.0)
    directions = [row[0] for row in rows]
    points = [row[1] for row in rows]
    kinds = np.array([row[2] for row in rows], np.int32)
    types = np.array([row[3] for row in rows], np.int32)
    parents = np.array([row[4] for row in rows], np.int32)
    mu = np.array([row[5] for row in rows])
    if linked:
        for k, normal in enumerate(patch):
            parents[normal - ring_base] = patch[(k + 1) % len(patch)]
        parents[single - ring_base] = single
    rhs = np.where(types == PGS_CONSTRAINT_TYPE_CONTACT, rng.uniform(-0.02, 0.0, len(rows)), 0.0)
    rhs[0] = -0.05
    return directions, points, kinds, types, parents, mu, rhs


def parallel_case(seed, *, faces=1, linked=True, warm=False):
    """One world of ``faces`` box faces against the ground on a six-coordinate body, sliding and falling.

    Holds the physical rows ``J``, the response rows ``Y = H^-1 J^T``, the
    whitened rows ``Z = L^-1 J^T`` (``H = L L^T``) and the row metadata of the
    kernel's inputs.
    """
    rng = np.random.default_rng(seed)
    angular = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    H = np.zeros((_DOFS, _DOFS))
    H[:3, :3] = 2.0 * np.eye(3)
    H[3:, 3:] = angular @ np.diag(rng.uniform(0.02, 0.04, 3)) @ angular.T
    L = np.linalg.cholesky(H)
    J, kinds, types, parents, mu, rhs = [], [], [], [], [], []
    for face in range(faces):
        directions, points, k, t, p, m, r = _box_contact_rows(rng, len(kinds), linked=linked)
        for direction, point in zip(directions, points, strict=True):
            if direction is None:
                row = np.zeros(_DOFS)
                row[3 + face % 3] = 1.0
            else:
                row = np.concatenate((direction, np.cross(point, direction)))
            J.append(row)
        kinds += list(k)
        types += list(t)
        parents += list(p)
        mu += list(m)
        rhs += list(r)
    J = np.array(J)
    count = len(J)
    impulses = np.zeros(count)
    if warm:
        impulses = np.where(np.array(types) == PGS_CONSTRAINT_TYPE_FRICTION, rng.uniform(-0.01, 0.01, count), 0.02)
    Y = np.linalg.solve(H, J.T).T
    cfm = 1.0e-5
    return {
        "J": J,
        "Y": Y,
        "Z": np.linalg.solve(L, J.T).T,
        "L": L,
        "cfm": cfm,
        "diag": np.einsum("ij,ij->i", J, Y) + cfm,
        "kinds": np.array(kinds, np.int32),
        "types": np.array(types, np.int32),
        "parents": np.array(parents, np.int32),
        "mu": np.array(mu),
        "rhs": np.array(rhs),
        "impulses": impulses,
        "velocity": np.array((0.3, -0.12, -0.25, 0.4, -0.3, 0.2)),
    }


def spread_rows(case, seed):
    """Shuffle a case's row slots by contact, as atomic slot allocation spreads a patch over a world's rows.

    A contact's normal keeps its tangent rows right behind it; a limit row moves
    alone. Parents follow their rows. Returns the shuffled case and ``order``
    (shuffled row -> original row): a parallel sweep updates every row from the
    same impulses, so its solution moves with the rows.
    """
    types, parents = case["types"], case["parents"]
    blocks, row = [], 0
    while row < len(types):
        size = 1
        while row + size < len(types) and types[row + size] == PGS_CONSTRAINT_TYPE_FRICTION:
            size += 1
        blocks.append(np.arange(row, row + size))
        row += size
    order = np.concatenate([blocks[b] for b in np.random.default_rng(seed).permutation(len(blocks))])
    position = np.empty_like(order)
    position[order] = np.arange(len(order))
    spread = dict(case)
    for name in ("J", "Y", "Z", "diag", "kinds", "types", "mu", "rhs", "impulses"):
        spread[name] = case[name][order]
    moved = parents[order]
    spread["parents"] = np.where(moved >= 0, position[np.maximum(moved, 0)], -1).astype(np.int32)
    return spread, order


def ring_rows(parents, normal):
    """The normal rows of the patch ring that starts at ``normal``, in ring order."""
    ring, row = [normal], parents[normal]
    while 0 <= row < len(parents) and row != normal:
        ring.append(int(row))
        row = parents[row]
    return ring


def assert_pooled_coulomb(test, case, impulses, *, linked, atol=1.0e-6):
    """Bound every tangent pair by its coefficient times its ring's (or its own) normal impulse."""
    pairs = 0
    for row in np.flatnonzero(case["types"] == PGS_CONSTRAINT_TYPE_FRICTION):
        parent = int(case["parents"][row])
        if row != parent + 1:
            continue
        pairs += 1
        budget = case["mu"][row] * ring_load(case["parents"], impulses, parent, linked=linked)
        test.assertLessEqual(math.hypot(impulses[row], impulses[row + 1]), budget + atol, f"pair at row {row}")
    test.assertGreater(pairs, 0)


# Golden text of the parallel sweep's friction projection, one warp (NT == 32) and several warps. The point
# pair scales into the disk of its own freshly clamped normal. The linked pair stages every row's fresh value
# in shared memory (one warp: store, then __syncwarp; several warps stage before projecting) and walks its
# ring from its own normal, summing in ring order.
_RING_WALK = (
    "                float load = s_x[p];\n"
    "                for (int ring = s_parent[p], hop = 0; ring >= 0 && ring < n_rows && ring != p && hop < n_rows;"
    " ring = s_parent[ring], ++hop)\n"
    "                    load += s_x[ring];\n"
    "                const float radius = fmaxf(s_mu[i] * load, 0.0f);\n"
)
_POINT_WARP_PAIR = (
    "            const float xp = __shfl_sync(MASK, x_new[0], p & 31);\n"
    "            const float xs = __shfl_sync(MASK, x_new[0], sib & 31);\n"
    "            if (fr) {\n"
    "                const float radius = fmaxf(s_mu[i] * xp, 0.0f);\n"
    "                const float a_val = x_new[0];\n"
)
_LINKED_WARP_PAIR = (
    "            const float xs = __shfl_sync(MASK, x_new[0], sib & 31);\n"
    "            if (i < n_rows) s_x[i] = x_new[0];\n"
    "            __syncwarp();\n"
    "            if (fr) {\n" + _RING_WALK + "                const float a_val = x_new[0];\n"
)
_POINT_BLOCK_PAIR = (
    "                const int sib = (i == p + 1) ? p + 2 : p + 1;\n"
    "                const float radius = fmaxf(s_mu[i] * s_x[p], 0.0f);\n"
    "                const float a_val = s_x[i];\n"
)
_LINKED_BLOCK_PAIR = (
    "                const int sib = (i == p + 1) ? p + 2 : p + 1;\n"
    + _RING_WALK
    + "                const float a_val = s_x[i];\n"
)


class TestParallelPatchReference(unittest.TestCase):
    def test_linked_factory_pools_only_the_friction_disk(self):
        """Build linked tiers as the point tiers with the golden ring walk, and refuse in-sweep world rows."""
        for rows, min_rows, matrix_free, inkernel in itertools.product(
            (32, 64), (0,), (False, True), (None, (6, 0, 0, 0))
        ):
            options = {
                "rows": rows,
                "min_rows": min_rows,
                "sweeps": 24,
                "matrix_free": matrix_free,
                "inkernel_response": inkernel,
            }
            with self.subTest(**options):
                point = source._get_pgs_solve_parallel_kernel(_CAPACITY, 32, _DOFS, "120", **options)
                linked = source._get_pgs_solve_parallel_kernel(_CAPACITY, 32, _DOFS, "120", linked_rows=True, **options)
                self.assertEqual(linked.key, point.key + "_linked")
                point_source = point._fpgs_row_native_source
                linked_source = linked._fpgs_row_native_source
                for text, golden in (
                    (point_source, _POINT_WARP_PAIR),
                    (point_source, _POINT_BLOCK_PAIR),
                    (linked_source, _LINKED_WARP_PAIR),
                    (linked_source, _LINKED_BLOCK_PAIR),
                ):
                    self.assertEqual(text.count(golden), 1, golden)
                # Nothing but the two projections differs from the point tier.
                restored = linked_source.replace(_LINKED_WARP_PAIR, _POINT_WARP_PAIR)
                self.assertEqual(restored.replace(_LINKED_BLOCK_PAIR, _POINT_BLOCK_PAIR), point_source)
                self.assertEqual(linked._fpgs_block_dim, point._fpgs_block_dim)
        with self.assertRaisesRegex(ValueError, "linked_rows"):
            source._get_pgs_solve_parallel_kernel(
                _CAPACITY,
                32,
                _DOFS,
                "120",
                rows=32,
                sweeps=4,
                matrix_free=True,
                inkernel_response=(6, 0, 0, 0),
                world_rows=True,
                linked_rows=True,
            )

    def test_reference_holds_pooled_cones_and_restores_point_rings(self):
        """Keep each pair inside its pooled ring disk, and reduce to point friction without ring links."""
        for seed, faces in ((0, 1), (1, 3)):
            linked_case = parallel_case(seed, faces=faces)
            point_case = parallel_case(seed, faces=faces, linked=False)
            for matrix_free, inkernel in itertools.product((False, True), repeat=2):
                with self.subTest(seed=seed, faces=faces, matrix_free=matrix_free, inkernel=inkernel):
                    options = {
                        "matrix_free": matrix_free,
                        "inkernel": inkernel,
                        "exact": False,
                        "sweeps": 24,
                        "nesterov": True,
                        "tol": 1.0e-4,
                    }
                    linked, _ = parallel_sweep_reference(linked_case, linked=True, **options)
                    point, _ = parallel_sweep_reference(point_case, linked=False, **options)
                    assert_pooled_coulomb(self, linked_case, linked, linked=True)
                    assert_pooled_coulomb(self, point_case, point, linked=False)
                    # Without links the ring walk stops at the normal: the point law exactly.
                    same, _ = parallel_sweep_reference(point_case, linked=True, **options)
                    np.testing.assert_array_equal(same, point)
                    # The patch anchors pool four normals at half the coefficient, which moves the solution.
                    self.assertGreater(np.max(np.abs(linked - point)), 1.0e-3)

    def test_spread_rings_cross_the_warp_boundary(self):
        """Spread every patch ring of the two-warp case across row 32 and keep the sweep's solution.

        The two-warp tiers read a ring normal staged by the other warp; the spread
        case puts pairs and ring members on both sides of row 32. The reference
        solution moves with the rows, which also checks the parent relinking.
        """
        case = parallel_case(3, faces=3)
        spread, order = spread_rows(case, _SPREAD_SEED)
        parents = spread["parents"]
        rings = {
            tuple(sorted(ring_rows(parents, int(n))))
            for n in np.flatnonzero(spread["types"] == PGS_CONSTRAINT_TYPE_CONTACT)
            if parents[n] >= 0 and parents[n] != n
        }
        self.assertEqual(len(rings), 3)
        self.assertTrue(all(min(ring) < 32 <= max(ring) for ring in rings), rings)
        crossing = [
            row
            for row in np.flatnonzero(spread["types"] == PGS_CONSTRAINT_TYPE_FRICTION)
            if row == parents[row] + 1
            and any((member >= 32) != (row >= 32) for member in ring_rows(parents, int(parents[row])))
        ]
        self.assertGreaterEqual(len(crossing), 2)
        for matrix_free, inkernel in itertools.product((False, True), repeat=2):
            with self.subTest(matrix_free=matrix_free, inkernel=inkernel):
                options = {
                    "matrix_free": matrix_free,
                    "inkernel": inkernel,
                    "exact": False,
                    "sweeps": 24,
                    "nesterov": True,
                    "tol": 0.0,
                    "linked": True,
                }
                impulses, velocity = parallel_sweep_reference(case, **options)
                moved, moved_velocity = parallel_sweep_reference(spread, **options)
                np.testing.assert_allclose(moved, impulses[order], rtol=1.0e-12, atol=1.0e-14)
                np.testing.assert_allclose(moved_velocity, velocity, rtol=1.0e-12, atol=1.0e-14)
                assert_pooled_coulomb(self, spread, moved, linked=True)


class _RowsBuilt(Exception):
    """Stops a step once its contact rows are built and linked, before any sweep."""


_ROW_ARRAYS = (
    "constraint_count",
    "row_type",
    "row_parent",
    "row_mu",
    "row_beta",
    "row_cfm",
    "phi",
    "target_velocity",
    "row_restitution",
    "rhs",
    "contact_slot",
    "contact_world",
    "contact_slots_needed",
    "contact_path",
)


def _linked_rows(solver, state, contacts, dt):
    """Step until the contact rows are built and linked, and return host copies of the rows."""
    rows = {}
    finalize = SolverFeatherPGS._finalize_constraint_status

    def capture(self, step_contacts):
        finalize(self, step_contacts)
        if self is solver:
            rows.update({name: getattr(self, name).numpy().copy() for name in _ROW_ARRAYS})
            rows["J"] = {size: self.J_by_size[size].numpy().copy() for size in self.size_groups}
            raise _RowsBuilt

    with patch.object(SolverFeatherPGS, "_finalize_constraint_status", capture):
        try:
            solver.step(state, solver.model.state(), solver.model.control(), contacts, dt)
        except _RowsBuilt:
            pass
    if not rows:
        raise AssertionError("The step did not build contact rows")
    return rows


def _general_bias(solver, rows, dt):
    """The general route's contact and friction bias for captured rows (without ``J v``)."""
    device = solver.model.device
    count = wp.array(rows["constraint_count"], dtype=wp.int32, device=device)
    rhs = wp.zeros(rows["rhs"].shape, dtype=wp.float32, device=device)
    wp.launch(
        compute_world_contact_bias,
        dim=rhs.shape[0] * rhs.shape[1],
        inputs=[
            count,
            rhs.shape[1],
            wp.array(rows["phi"], dtype=wp.float32, device=device),
            wp.array(rows["row_beta"], dtype=wp.float32, device=device),
            wp.array(rows["row_type"], dtype=wp.int32, device=device),
            wp.array(rows["target_velocity"], dtype=wp.float32, device=device),
            dt,
            1.0,
            solver.contact_speculative_scale,
            1.0,
            1.0,
            wp.zeros(rhs.shape[0], dtype=wp.int32, device=device),
            0,
        ],
        outputs=[rhs, wp.zeros_like(rhs)],
        device=device,
    )
    return rhs.numpy()


def _contact_rows_by_id(rows, count):
    """Map each live dense-routed contact to its rows: ``{contact: (world, first row, row count)}``."""
    owner = {}
    for c in range(count):
        slot, needed = int(rows["contact_slot"][c]), int(rows["contact_slots_needed"][c])
        if rows["contact_path"][c] == 0 and slot >= 0 and needed > 0:
            owner[c] = (int(rows["contact_world"][c]), slot, needed)
    return owner


class TestKeyedPatchRows(unittest.TestCase):
    def test_keyed_rows_write_the_patch_row_law(self):
        """Write keyed patch rows as the general producer and the patch link write them.

        Every world above the keyed row budget takes the fallback, which runs on
        CPU and shares its row source with the parallel tiers' in-kernel reader:
        tangent rows act at the patch friction location with the anchor's phi and
        friction_anchor_beta, normals link their ring and anchors divide the
        coefficient, and the bias matches the general route's. Both solvers build
        the frame from the same anchor history, which a few general steps load.
        Both also read a friction_anchor_beta assigned after construction, as the
        general producer reads it every step.
        """
        for late_beta in (None, 0.35):
            with self.subTest(late_beta=late_beta):
                self._check_keyed_patch_rows(late_beta)

    def _check_keyed_patch_rows(self, late_beta):
        model = standing_model(device="cpu", worlds=2)
        dt = 1.0 / 240.0
        options = {
            "friction_anchor_beta": 0.2,
            "pgs_iterations": 4,
            "dense_max_constraints": 64,
            "joint_limit_activation_gap": 0.1,
        }
        general = SolverFeatherPGS(model, **options)
        keyed = SolverFeatherPGS(model, **options)
        # Stand-ins for the matrix-free world layout keyed rows read: one articulation per world.
        keyed.max_world_dofs = int(model.joint_dof_count) // keyed.world_count
        keyed.articulation_world_dof_offset = wp.zeros(model.articulation_count, dtype=int, device=model.device)
        keyed.mf_constraint_count = wp.zeros(keyed.world_count, dtype=int, device=model.device)
        keyed.mf_gs_parallel_rows = 4
        keyed._friction_patches = general._friction_patches
        keyed._keyed_rows = keyed_rows.KeyedRows(keyed)
        if late_beta is not None:
            general.friction_anchor_beta = keyed.friction_anchor_beta = late_beta
        state, out = model.state(), model.state()
        velocity = state.joint_qd.numpy()
        velocity[0], velocity[11] = 0.05, -0.04
        state.joint_qd.assign(velocity)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, deterministic=True)
        contacts = pipeline.contacts()
        for _ in range(6):
            pipeline.collide(state, contacts)
            general.step(state, out, model.control(), contacts, dt)
            state, out = out, state
        pipeline.collide(state, contacts)
        expected = _linked_rows(general, state, contacts, dt)
        actual = _linked_rows(keyed, state, contacts, dt)
        live = int(contacts.rigid_contact_count.numpy()[0])
        owner = _contact_rows_by_id(expected, live)
        self.assertEqual(owner, _contact_rows_by_id(actual, live))
        slot_contact = {(world, first): c for c, (world, first, _) in owner.items()}
        bias = _general_bias(general, expected, dt)
        size = keyed._keyed_rows.layout.primary
        linked = tangent_phi = 0
        for c, (world, first, needed) in owner.items():
            rows = slice(first, first + needed)
            with self.subTest(contact=c):
                for name in ("row_type", "row_beta", "row_cfm", "row_restitution"):
                    np.testing.assert_array_equal(actual[name][world, rows], expected[name][world, rows], name)
                # Ring links by contact: the next normal of the patch, the owning normal of a tangent row.
                for rows_of in (expected, actual):
                    parents = [
                        slot_contact.get((world, int(p)), -2) if p >= 0 else -1
                        for p in rows_of["row_parent"][world, rows]
                    ]
                    rows_of.setdefault("linked", {})[c] = parents
                self.assertEqual(actual["linked"][c], expected["linked"][c])
                for name in ("row_mu", "phi", "target_velocity"):
                    np.testing.assert_allclose(
                        actual[name][world, rows], expected[name][world, rows], rtol=1.0e-6, atol=1.0e-9, err_msg=name
                    )
                np.testing.assert_allclose(
                    actual["J"][size][world, rows], expected["J"][size][world, rows], rtol=1.0e-6, atol=1.0e-7
                )
                np.testing.assert_allclose(actual["rhs"][world, rows], bias[world, rows], rtol=1.0e-5, atol=1.0e-7)
                linked += expected["linked"][c][0] not in (-1, c)
                tangent_phi += needed == 3 and bool(np.any(expected["phi"][world, first + 1 : first + 3] != 0.0))
                if needed == 3:
                    # The anchors carry the coefficient in force at this step.
                    beta = options["friction_anchor_beta"] if late_beta is None else late_beta
                    np.testing.assert_array_equal(expected["row_beta"][world, first + 1 : first + 3], np.float32(beta))
        self.assertGreater(linked, 0, "A box foot must link several normals in one ring")
        self.assertGreater(tangent_phi, 0, "Anchors must carry tangential phi")

    def test_keyed_factory_reads_linked_parents(self):
        """Read every row's parent and coefficient in the keyed patch tiers instead of building them."""
        layout = KeyedLayout(16, 6, 0, 16, 128, False, 0)
        factory = keyed_rows.get_parallel_factory(source._get_pgs_solve_parallel_kernel, layout)
        options = {"rows": 32, "sweeps": 24, "matrix_free": True, "inkernel_response": (16, 6, 0, 16)}
        point = factory(128, 32, 22, "120", **options)
        linked = factory(128, 32, 22, "120", linked_rows=True, **options)
        self.assertEqual(linked.key, point.key + "_linked")
        self.assertTrue(linked.key.startswith("pgs_solve_kinetic_keyed_parallel_"))
        self.assertIn("patch_law", inspect.signature(linked.func).parameters)
        self.assertNotIn("patch_law", inspect.signature(point.func).parameters)
        text = linked._fpgs_row_native_source
        self.assertIn("s_parent[i] = world_row_parent.data[off_dense + i];", text)
        self.assertIn("s_mu[i] = world_row_mu.data[off_dense + i];", text)
        for written in ("c.row_parent.data", "c.row_mu.data", "c.row_type.data", "built_parent", "built_mu"):
            self.assertNotIn(written, text)
        self.assertIn("patch_law.patches.point_a.data[contact]", text)
        self.assertIn("patch_law.patches.phi.data[contact][r - 1]", text)
        for golden in (_LINKED_WARP_PAIR, _LINKED_BLOCK_PAIR):
            self.assertEqual(text.count(golden), 1, golden)


_SWITCH = "FEATHER_PGS_PATCH_PARALLEL_ROWS"


def _cpu_arm():
    """A one-joint arm: enough for construction-time admission on CPU."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    return builder.finalize(device="cpu")


class TestPatchParallelAdmission(unittest.TestCase):
    def test_opt_in_switch_admits_patch_friction_and_records_why(self):
        """Keep patch friction on its serial sweeps by default, admit it under the switch, and record why.

        Only ``FEATHER_PGS_PATCH_PARALLEL_ROWS=1`` admits patch friction to the parallel
        tiers and keyed rows; unset or ``0`` it keeps its serial sweeps and ordinary
        rows. Point friction is admitted whatever the switch says. Tests that need
        patch admission set the switch rather than rely on its default.
        """
        model = _cpu_arm()
        for beta, switch, admitted in (
            (0.0, None, True),
            (0.0, "0", True),
            (0.0, "1", True),
            (0.2, None, False),  # the default: off
            (0.2, "0", False),
            (0.2, "1", True),
        ):
            with self.subTest(beta=beta, switch=switch), patch.dict(os.environ):
                os.environ.pop(_SWITCH, None)
                if switch is not None:
                    os.environ[_SWITCH] = switch
                solver = SolverFeatherPGS(model, friction_anchor_beta=beta)
                selected = solver.get_path_selection()
                friction = selected["routes"]["friction"]
                keyed = selected["routes"]["contact_rows"]["keyed_rows"]
                self.assertEqual(solver._parallel_contact_paths, admitted)
                self.assertEqual(friction["parallel_rows"]["admitted"], admitted)
                self.assertEqual(selected["environment"]["applied"].get(_SWITCH), switch)
                if beta == 0.0:
                    self.assertIn("Point friction", friction["parallel_rows"]["reason"])
                    continue
                # Under the switch the only parallel-row owners left to point friction are the response blocks.
                self.assertEqual("keyed rows" in friction["reason"], not admitted)
                self.assertIn("need point friction", friction["reason"])
                if admitted:
                    self.assertIn("patch ring", friction["parallel_rows"]["reason"])
                    self.assertNotIn("point contacts", keyed["reason"])
                else:
                    # The serial patch route; both reasons name the opt-in.
                    self.assertIn("serial sweeps", friction["parallel_rows"]["reason"])
                    self.assertNotIn("patch ring", friction["parallel_rows"]["reason"])
                    self.assertIn(f"{_SWITCH}=1", friction["parallel_rows"]["reason"])
                    self.assertIn(f"{_SWITCH}=1", keyed["reason"])
        # Torsion and compliance (CUDA-only laws here) keep ordinary rows whatever the switch says.
        other_law = SimpleNamespace(_point_contact_paths=False, _law_agnostic_contact_paths=False)
        for switch in (None, "0", "1"):
            with self.subTest(law="torsion or compliance", switch=switch), patch.dict(os.environ):
                os.environ.pop(_SWITCH, None)
                if switch is not None:
                    os.environ[_SWITCH] = switch
                admitted, reason = SolverFeatherPGS._admit_parallel_contact_paths(other_law)
                self.assertFalse(admitted)
                self.assertIn("torsion", reason)

    def test_route_record_reads_the_built_tiers(self):
        """Record the tiers' ring-pooled projection from the kernels built, not from the friction law."""
        options = {"rows": 32, "sweeps": 24, "matrix_free": True}
        for anchors, linked in itertools.product((False, True), repeat=2):
            kernel = source._get_pgs_solve_parallel_kernel(_CAPACITY, 32, _DOFS, "120", linked_rows=linked, **options)
            probe = SimpleNamespace(
                _pgs_solve_mf_gs_incremental_kernels=[kernel],
                _par_tiers=[(32, 0)],
                _ink_rows=32,
                _ink_sizes=None,
                _keyed_rows=None,
                _friction_anchors_enabled=anchors,
                mf_gs_parallel_nesterov=True,
                mf_gs_parallel_matrix_free=True,
                mf_gs_parallel_sweeps=24,
                mf_gs_parallel_tol=0.0,
            )
            with self.subTest(anchors=anchors, linked=linked):
                tiers = path_selection._tiers(probe)
                self.assertEqual(tiers["linked_rows"], linked)
                self.assertEqual(tiers["tiers"][0]["kernel"].endswith("_linked"), linked)

    def test_in_sweep_world_rows_serve_point_friction_only(self):
        """Build contact rows inside the sweep only under point friction, the law that builder writes.

        The builder runs after the patch link and writes parents -1 and undivided
        coefficients, so patch friction keeps the rows its producers linked.
        """
        for point, keyed, inkernel, switch in itertools.product((False, True), repeat=4):
            probe = SimpleNamespace(
                _point_contact_paths=point,
                _keyed_rows=object() if keyed else None,
                _ink_sizes=(18, 0, 0, 0) if inkernel else None,
            )
            with self.subTest(point=point, keyed=keyed, inkernel=inkernel, switch=switch):
                with patch.object(source, "_WR_ON", switch):
                    admitted = SolverFeatherPGS._admit_world_rows(probe)
                self.assertEqual(admitted, point and not keyed and inkernel and switch)


def _dummy(arg):
    """A one-element stand-in for an input the kernel variant never reads."""
    if wp.types.is_array(arg.type):
        return wp.zeros((1,) * arg.type.ndim, dtype=arg.type.dtype)
    return arg.type(0)


def _launch_parallel(kernel, case, *, inkernel, device):
    """Launch one world through a parallel tier and return its impulses and velocity."""
    count = len(case["J"])
    padded = np.zeros((1, _CAPACITY))

    def rows(values, dtype=wp.float32):
        out = padded.copy().astype(np.float32 if dtype == wp.float32 else np.int32)
        out[0, :count] = values
        return wp.array(out, dtype=dtype, device=device)

    J = np.zeros((1, _CAPACITY, _DOFS), np.float32)
    J[0, :count] = case["J"]
    Y = np.zeros((1, _CAPACITY, _DOFS), np.float32)
    Y[0, :count] = case["Y"]
    values = {
        "general_world_count": wp.array([1], dtype=wp.int32, device=device),
        "general_worlds": wp.zeros(1, dtype=wp.int32, device=device),
        "general_world_grid_stride": 1,
        "use_general_world_queue": 0,
        "world_constraint_count": wp.array([count], dtype=wp.int32, device=device),
        "mf_constraint_count": wp.zeros(1, dtype=wp.int32, device=device),
        "dense_phase_bounds": wp.zeros((1, 2), dtype=wp.int32, device=device),
        "world_dof_indices": wp.array(np.arange(_DOFS, dtype=np.int32)[None], dtype=wp.int32, device=device),
        "rhs_bias": rows(case["rhs"]),
        "world_diag": rows(case["diag"]),
        "world_impulses": rows(case["impulses"]),
        "J_world": wp.array(J, dtype=wp.float32, device=device),
        "Y_world": wp.array(Y, dtype=wp.float32, device=device),
        "world_row_type": rows(case["types"], wp.int32),
        "world_row_parent": rows(case["parents"], wp.int32),
        "world_row_mu": rows(case["mu"]),
        "world_row_cfm": rows(np.full(count, case["cfm"])),
        "iterations": 1,
        "omega": 1.0,
        "v_out": wp.array(case["velocity"].astype(np.float32), dtype=wp.float32, device=device),
    }
    if inkernel:
        L = case["L"].astype(np.float32)[None]
        values["ink_meta"] = wp.array([0, 0, -1, 0], dtype=wp.int32, device=device)
        values["ink_L_a"] = wp.array(L, dtype=wp.float32, device=device)
        values["ink_J_a"] = wp.array(J, dtype=wp.float32, device=device)
    arguments = {arg.label: arg for arg in kernel.adj.args}
    inputs = [
        values[name] if name in values else _dummy(arguments[name])
        for name in inspect.signature(kernel.func).parameters
    ]
    wp.launch_tiled(kernel, dim=[1], inputs=inputs, block_dim=kernel._fpgs_block_dim, device=device)
    return values["world_impulses"].numpy()[0, :count], values["v_out"].numpy()


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestParallelPatchKernel(unittest.TestCase):
    def test_linked_tiers_match_the_reference(self):
        """Match the linked tiers (one and two warps, A-based and matrix-free, in-kernel response) to the reference.

        The point tiers on the same rows without ring links pin the reference to the
        kernel's existing law first, so a mismatch of the linked tiers isolates the ring walk.
        Each case also runs with its contacts' slots spread (spread_rows), which puts the
        two-warp case's rings and pairs on both sides of row 32.
        """
        device = wp.get_device("cuda:0")
        arch = str(device.arch)
        for (faces, rows, min_rows), matrix_free, inkernel, nesterov, warm, spread in itertools.product(
            ((1, 32, 0), (3, 64, 32)), (False, True), (False, True), (False, True), (False, True), (False, True)
        ):
            options = {"matrix_free": matrix_free, "inkernel": inkernel, "exact": False, "nesterov": nesterov}
            with self.subTest(faces=faces, warm=warm, spread=spread, **options):
                for linked in (False, True):
                    case = parallel_case(faces, faces=faces, linked=linked, warm=warm)
                    if spread:
                        case, _ = spread_rows(case, _SPREAD_SEED)
                    self.assertGreater(len(case["J"]), min_rows)
                    kernel = source._get_pgs_solve_parallel_kernel(
                        _CAPACITY,
                        32,
                        _DOFS,
                        arch,
                        rows=rows,
                        min_rows=min_rows,
                        sweeps=24,
                        nesterov=nesterov,
                        tol=0.0,
                        matrix_free=matrix_free,
                        inkernel_response=(_DOFS, 0, 0, 0) if inkernel else None,
                        linked_rows=linked,
                    )
                    impulses, velocity = _launch_parallel(kernel, case, inkernel=inkernel, device=device)
                    expected, expected_velocity = parallel_sweep_reference(
                        case, sweeps=24, tol=0.0, linked=linked, **options
                    )
                    scale = max(1.0, float(np.max(np.abs(expected))))
                    np.testing.assert_allclose(
                        impulses, expected, rtol=1.0e-3, atol=2.0e-5 * scale, err_msg=f"linked={linked}"
                    )
                    np.testing.assert_allclose(velocity, expected_velocity, rtol=1.0e-3, atol=2.0e-5 * scale)
                    assert_pooled_coulomb(self, case, impulses.astype(np.float64), linked=linked, atol=2.0e-5 * scale)


_STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")
# Float-order tolerances of an exact route change (assert_states_close of the sparse solver tests).
_STATE_FLOOR = {"joint_q": 2.0e-4, "body_q": 2.0e-4, "joint_qd": 2.0e-3, "body_qd": 2.0e-3}


def _matrix_free_solver(model, *, beta, parallel_rows, **options):
    """A dense matrix-free solver whose sweep is serial (``parallel_rows=0``) or the parallel tiers."""
    settings = {
        "pgs_mode": "matrix_free",
        "friction_anchor_beta": beta,
        "pgs_iterations": 8,
        "dense_max_constraints": 32,
        "sparse_mass_matrix": "off",
        "mf_gs_incremental_rows": 0,
        "mf_gs_parallel_rows": parallel_rows,
        "mf_gs_parallel_matrix_free": True,
        "mf_gs_parallel_sweeps": 24,
        "double_buffer": False,
    }
    settings.update(options)
    return SolverFeatherPGS(model, **settings)


def _rollout(model, solver, steps, *, dt=1.0 / 240.0, after_step=None):
    """Step through the collision pipeline; return the final state and the contacts of the last step."""
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, deterministic=True)
    contacts = pipeline.contacts()
    for step in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, out, control, contacts, dt)
        state, out = out, state
        if after_step is not None:
            after_step(step, solver, contacts, state)
    return state, contacts


def _state_differences(first, second):
    return {
        name: float(np.max(np.abs(getattr(first, name).numpy() - getattr(second, name).numpy())))
        for name in _STATE_FIELDS
    }


def _sliding_boxes(device, speeds, *, mu=0.5):
    """One box per world on prismatic x, y and z joints over the ground, starting at ``speeds`` along x.

    The box rests on four corner contacts, one patch ring with two anchors. It
    cannot rotate, so its friction acts along a fixed tangent and a slide stops
    after ``v^2 / (2 mu g)``.
    """
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.density, cfg.mu = 0.0, mu
    for _ in speeds:
        builder.begin_world()
        parent, joints = -1, []
        for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z):
            box = axis == newton.Axis.Z
            body = builder.add_link(
                mass=1.0 if box else 0.01, inertia=wp.mat33(np.eye(3) * (0.005 if box else 1.0e-5)), lock_inertia=True
            )
            joints.append(builder.add_joint_prismatic(parent=parent, child=body, axis=axis))
            parent = body
        builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.05, cfg=cfg)
        builder.add_articulation(joints)
        builder.end_world()
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    q, qd = model.joint_q.numpy(), model.joint_qd.numpy()
    for world, speed in enumerate(speeds):
        q[3 * world + 2] = 0.05
        qd[3 * world] = speed
    model.joint_q.assign(q)
    model.joint_qd.assign(qd)
    return model


def _assert_tier_owned(test, solver, budget):
    """Require the parallel tiers to have solved every world with rows in the last step.

    A tier takes a world with at most ``budget`` dense rows and no matrix-free rows,
    unless a local owner holds it; the serial sweep returns early for exactly those
    worlds, so a world in that domain is the tier's (``skip_incremental_rows``).
    """
    test.assertEqual(solver._ink_rows, budget)
    test.assertTrue(solver._pgs_solve_mf_gs_incremental_kernels)
    test.assertFalse(solver._local_internal_fast_path)
    counts, free_rows = solver.constraint_count.numpy(), solver.mf_constraint_count.numpy()
    test.assertGreater(int(counts.max()), 0)
    for world, (rows, mf_rows) in enumerate(zip(counts, free_rows, strict=True)):
        if rows:
            test.assertLessEqual(int(rows), budget, f"world {world}")
            test.assertEqual(int(mf_rows), 0, f"world {world}")


def _friction_pairs(solver, world):
    """Each friction pair's impulse magnitude and pooled budget ``mu * (ring's normal impulses)``."""
    count = int(solver.constraint_count.numpy()[world])
    kinds = solver.row_type.numpy()[world, :count]
    parents = solver.row_parent.numpy()[world, :count]
    mu, impulses = solver.row_mu.numpy()[world, :count], solver.impulses.numpy()[world, :count]
    pairs = []
    for row in np.flatnonzero(kinds == PGS_CONSTRAINT_TYPE_FRICTION):
        parent = int(parents[row])
        if row == parent + 1:
            budget = float(mu[row]) * ring_load(parents, impulses.astype(np.float64), parent)
            pairs.append((math.hypot(impulses[row], impulses[row + 1]), budget))
    return pairs


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestPatchParallelRoutes(unittest.TestCase):
    def test_patch_runs_linked_tiers_only_when_switched_on(self):
        """Build linked tiers for patch friction only under the switch, and point tiers for point friction.

        By default patch friction keeps the route it had before the tiers learned the
        ring: the serial sweep, or the incremental tier that the constructor's default
        ``mf_gs_incremental_rows=32`` requests and an admitted parallel tier replaces.
        Point friction builds its tiers with the switch unset, as it runs by default.
        """
        model = standing_model()
        options = {"sparse_mass_matrix": "auto", "cached_dynamics": "auto"}
        with patch.dict(os.environ, {_SWITCH: "1"}):
            patch_tiers = _matrix_free_solver(model, beta=0.2, parallel_rows=32, **options)
            serial = _matrix_free_solver(model, beta=0.2, parallel_rows=0, **options)
            # Both tiers requested on the dense route (sparse factors off), so only the switch picks one.
            over_incremental = _matrix_free_solver(model, beta=0.2, parallel_rows=32, mf_gs_incremental_rows=32)
            del os.environ[_SWITCH]  # the default
            default = _matrix_free_solver(model, beta=0.2, parallel_rows=32, **options)
            incremental = _matrix_free_solver(model, beta=0.2, parallel_rows=32, mf_gs_incremental_rows=32)
            point_tiers = _matrix_free_solver(model, beta=0.0, parallel_rows=32, **options)
        for solver, linked in ((patch_tiers, True), (point_tiers, False)):
            self.assertEqual(solver._ink_rows, 32)
            # The tier owns the ordinary buffers for either law.
            self.assertIsNone(solver._sparse_mass_matrix_size)
            keys = [kernel.key for kernel in solver._pgs_solve_mf_gs_incremental_kernels]
            self.assertTrue(keys)
            self.assertTrue(all(key.endswith("_linked") == linked for key in keys), keys)
            tiers = solver.get_path_selection()["routes"]["sweep"]["tiers"]
            self.assertEqual((tiers["family"], tiers["linked_rows"]), ("parallel", linked))
        # By default, patch friction keeps the serial route it had before the tiers learned the ring.
        self.assertEqual(default._ink_rows, 0)
        self.assertEqual(default._pgs_solve_mf_gs_incremental_kernels, [])
        expected, actual = serial.get_path_selection(), default.get_path_selection()
        for record in (expected, actual):
            record["routes"]["contact_rows"]["keyed_rows"].pop("reason")
        for name in ("sweep", "sparse_mass_matrix", "dynamics", "publication", "contact_rows"):
            self.assertEqual(actual["routes"][name], expected["routes"][name], name)
        # Held-tree lane widths are timed at construction; the owner is the route.
        self.assertEqual(actual["routes"]["held_tree"]["owner"], expected["routes"]["held_tree"]["owner"])
        self.assertEqual(actual["kernel_plan_by_dof"], expected["kernel_plan_by_dof"])
        self.assertFalse(actual["routes"]["friction"]["parallel_rows"]["admitted"])
        self.assertIn(f"{_SWITCH}=1", actual["routes"]["friction"]["parallel_rows"]["reason"])
        # With the incremental tier requested too, the default keeps it; opted in, the linked tiers replace it.
        self.assertEqual((incremental._ink_rows, incremental._incremental_rows), (0, 32))
        self.assertEqual(incremental.get_path_selection()["routes"]["sweep"]["tiers"]["family"], "incremental")
        self.assertEqual((over_incremental._ink_rows, over_incremental._incremental_rows), (32, 0))
        tiers = over_incremental.get_path_selection()["routes"]["sweep"]["tiers"]
        self.assertEqual((tiers["family"], tiers["linked_rows"]), ("parallel", True))

    def test_in_sweep_world_rows_stay_point_only(self):
        """Own no in-sweep world rows under patch friction with INK and WORLD_ROWS and keyed rows declined."""
        model = standing_model()
        built = {}
        with (
            patch.object(source, "_INK_ON", True),
            patch.object(source, "_WR_ON", True),
            patch.dict(os.environ, {"FEATHER_PGS_KEYED_ROWS": "0", _SWITCH: "1"}),
        ):
            for beta in (0.0, 0.2):
                built[beta] = _matrix_free_solver(model, beta=beta, parallel_rows=32)
        point, patch_friction = built[0.0], built[0.2]
        for solver in (point, patch_friction):
            self.assertIsNone(solver._keyed_rows)
            self.assertIsNotNone(solver._ink_sizes)
            self.assertEqual(solver._ink_rows, 32)
        self.assertTrue(point._world_rows)
        self.assertEqual(point._world_rows_owned_le(), 32)
        self.assertFalse(patch_friction._world_rows)
        self.assertEqual(patch_friction._world_rows_owned_le(), 0)
        for kernel in patch_friction._pgs_solve_mf_gs_incremental_kernels:
            self.assertFalse(kernel._fpgs_world_rows)
            self.assertTrue(kernel.key.endswith("_linked"))


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestPatchParallelPhysics(unittest.TestCase):
    """Compare each law's parallel tiers with its serial sweep, the patch gap bounded by the point gap.

    Parallel tiers are Jacobi sweeps, not float reorderings of the serial sweep.
    While every pair sticks the two schemes share their fixed point, and the
    rollouts differ by the convergence gap. Once a pair slides they do not: the
    tiers scale the pair radially after per-row step scales, which turns its
    sliding direction and, through the coupling, moves the normals and the body
    velocity, for point and patch friction alike (on this module's reference case
    the converged velocities of the two schemes differ by 4-15 % for either law).
    So the point tiers' measured gap on the same fixture, with a floor at the
    float-order tolerance of an exact route change, bounds the patch tiers' gap.
    Every parallel step must leave each world in the tiers' domain, so the
    comparison is never serial against serial.
    """

    def test_standing_feet_match_the_serial_patch_sweep(self):
        """Keep loaded, ring-linked anchors and the serial sweep's motion on two box feet."""
        model = standing_model()
        velocity = model.joint_qd.numpy()
        velocity[0::10] = 0.05  # the torsos drift sideways, loading the feet's friction without slipping
        model.joint_qd.assign(velocity)
        with patch.dict(os.environ, {_SWITCH: "1"}):
            states = {}
            for beta, parallel_rows in itertools.product((0.0, 0.2), (0, 32)):

                def check(step, solver, contacts, state, beta=beta, parallel_rows=parallel_rows):
                    for name in _STATE_FIELDS:
                        self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
                    self.assertFalse(solver.constraint_overflow.numpy().any())
                    if parallel_rows:
                        _assert_tier_owned(self, solver, parallel_rows)
                    if beta and parallel_rows:
                        # Every stored parallel iterate lies in its pooled cone; a serial sweep's last
                        # pair update can precede a ring normal's, so only the tiers hold it exactly.
                        assert_ring_coulomb(solver)
                        assert_loaded_anchors(solver, contacts, carried=step > 0)

                solver = _matrix_free_solver(model, beta=beta, parallel_rows=parallel_rows)
                states[beta, parallel_rows] = _rollout(model, solver, 24, after_step=check)[0]
        point_gap = _state_differences(states[0.0, 32], states[0.0, 0])
        patch_gap = _state_differences(states[0.2, 32], states[0.2, 0])
        measured = {"point_gap": point_gap, "patch_gap": patch_gap}
        for name in _STATE_FIELDS:
            self.assertLessEqual(patch_gap[name], 3.0 * point_gap[name] + _STATE_FLOOR[name], (name, measured))

    def test_box_rests_and_slides_like_the_serial_patch_sweep(self):
        """Hold a resting box, and stop a sliding one where the serial patch sweep stops it.

        A sliding box saturates its patch's pooled cone: each anchor pair carries
        mu / 2 of the four normals' load. Pooling each pair on its own normal would
        halve the friction and double the stopping distance.
        """
        speed, mu, gravity = 1.0, 0.5, 9.81
        expected = speed * speed / (2.0 * mu * gravity)
        model = _sliding_boxes("cuda:0", (0.0, speed), mu=mu)
        distances = {}
        with patch.dict(os.environ, {_SWITCH: "1"}):
            for beta, parallel_rows in itertools.product((0.0, 0.2), (0, 32)):
                ratios = []

                def check(step, solver, contacts, state, ratios=ratios, parallel_rows=parallel_rows):
                    if parallel_rows:
                        _assert_tier_owned(self, solver, parallel_rows)
                    if 2 <= step <= 30:  # sliding: every friction pair sits on its cone
                        pairs = _friction_pairs(solver, world=1)
                        ratios.extend(magnitude / budget for magnitude, budget in pairs if budget > 0.0)

                solver = _matrix_free_solver(model, beta=beta, parallel_rows=parallel_rows)
                state, _ = _rollout(model, solver, 96, after_step=check)
                q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
                with self.subTest(beta=beta, parallel_rows=parallel_rows):
                    self.assertLess(np.max(np.abs(q[0:2])), 2.0e-4, "the resting box must not creep")
                    self.assertLess(abs(qd[3]), 1.0e-3, "the sliding box must stop")
                    self.assertLess(abs(q[3] - expected), 0.2 * expected, "friction must act at the full coefficient")
                    self.assertTrue(ratios)
                    self.assertGreater(min(ratios), 0.95)
                    # A parallel iterate is projected onto its pooled cone; a serial sweep may update
                    # a ring normal after the pair, leaving the pair marginally outside.
                    self.assertLess(max(ratios), 1.0 + (1.0e-4 if parallel_rows else 5.0e-3))
                distances[beta, parallel_rows] = float(q[3])
        point_gap = abs(distances[0.0, 32] - distances[0.0, 0])
        patch_gap = abs(distances[0.2, 32] - distances[0.2, 0])
        self.assertLessEqual(patch_gap, max(3.0 * point_gap, 0.01 * expected), distances)


if __name__ == "__main__":
    unittest.main()

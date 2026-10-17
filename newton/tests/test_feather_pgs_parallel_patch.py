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
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import keyed_rows
from newton._src.solvers.feather_pgs import solver_feather_pgs as source
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    compute_world_contact_bias,
)
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import standing_model

_DOFS = 6
_CAPACITY = 64


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


class TestParallelPatchReference(unittest.TestCase):
    def test_linked_factory_pools_only_the_friction_disk(self):
        """Build linked tiers as the point tiers plus the ring walk, and refuse in-sweep world rows."""
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
                self.assertEqual(source._link_parallel_friction_pairs(point_source), linked_source)
                self.assertNotIn("xp = __shfl_sync", linked_source)
                self.assertEqual(linked_source.count("load += s_x[ring];"), 2)
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
        """
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
        self.assertEqual(text.count("load += s_x[ring];"), 2)


class TestPatchParallelAdmission(unittest.TestCase):
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
        """
        device = wp.get_device("cuda:0")
        arch = str(device.arch)
        for (faces, rows, min_rows), matrix_free, inkernel, nesterov, warm in itertools.product(
            ((1, 32, 0), (3, 64, 32)), (False, True), (False, True), (False, True), (False, True)
        ):
            options = {"matrix_free": matrix_free, "inkernel": inkernel, "exact": False, "nesterov": nesterov}
            with self.subTest(faces=faces, warm=warm, **options):
                for linked in (False, True):
                    case = parallel_case(faces, faces=faces, linked=linked, warm=warm)
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


if __name__ == "__main__":
    unittest.main()

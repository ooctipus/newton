# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve persistent patch friction in the parallel matrix-free tiers.

The parallel tiers update all rows of a world together (projected Jacobi with
per-row step scales and Nesterov momentum). Under patch friction a contact's
normal row links the next normal of its patch ring, and each tangent pair is
projected into the disk of radius ``mu * (sum of its ring's normal impulses)``
with one isotropic step, so a sliding pair converges to the serial pair solve's
maximal dissipation. The reference below repeats the kernel's sweep in float64
so the CUDA kernels can be checked row by row; on CPU it also sweeps the rows a
split solver builds, against that solver's serial solution. Keyed rows write the
patch row law themselves; the CPU fallback checks it against the general
producer and the patch link.
"""

import inspect
import itertools
import math
import os
import unittest
from types import MethodType, SimpleNamespace
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
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    compute_world_contact_bias,
    prescale_joint_velocity_limits,
)
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import (
    SLIDING_FEET,
    assert_loaded_anchors,
    assert_ring_coulomb,
    standing_model,
)
from newton.tests.feather_pgs_row_builders import (
    ROW_SWITCHES_OFF,
    assert_row_builders_named,
    install_cpu_keyed_rows,
    launched_kernels,
)

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
    case,
    *,
    matrix_free,
    inkernel,
    exact,
    sweeps,
    nesterov,
    tol,
    linked,
    omega=1.0,
    friction_start=0,
    pair_step=None,
    velocity_limit_pass=False,
):
    """Repeat one world of the parallel tier's sweep in float64.

    Returns the impulses and the generalized velocity the kernel publishes. The
    response block is ``A = J H^-1 J^T``. The step scale of row ``i`` is
    ``min(A_ii / bound_i, 1) * omega / diag_i``: the bound is the row sum of
    ``|A|`` over admitted rows (A-based tiers and exact row sums) or the
    row-sum bound ``sum_d |P_id| sum_j |Q_jd|`` of the matrix-free tiers, with
    ``(P, Q) = (J, Y)`` or the whitened rows ``(Z, Z)`` of the in-kernel
    response. With the isotropic ``pair_step`` (the default of ``linked``) both
    rows of a friction pair take the smaller of their two steps, unless the
    sibling is not an admitted friction row of the same normal or either step is
    zero. Every sweep steps all rows from the extrapolated impulses, clamps
    normal and limit rows, scales each tangent pair into its disk (the ring's
    pooled normal load when ``linked``), restarts momentum against the last step
    and stops once no row changes. ``velocity_limit_pass`` then pushes every
    violated velocity-limit row onto its limit, in row order.
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
    if (pair_step or ("isotropic" if linked else "per_row")) == "isotropic":
        for row in np.flatnonzero(kinds == 1):
            parent, sibling = parents[row], row + 1
            if row != parent + 1 or sibling >= len(kinds) or kinds[sibling] != 1 or parents[sibling] != parent:
                continue
            if step[row] > 0.0 and step[sibling] > 0.0:
                step[row] = step[sibling] = min(step[row], step[sibling])
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
    if velocity_limit_pass:
        x = x.copy()
        for row in np.flatnonzero((kinds == 0) & (case["types"] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)):
            residual = b[row] + A[row] @ x
            if residual < 0.0 and diag[row] > 0.0:
                x[row] -= residual / diag[row]
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


def sliding_pair_case(*, masses=(4.0, 1.0), sliding=(1.0, 1.0), live_tangents=2):
    """One contact on a six-coordinate body whose two tangents have unequal effective masses, sliding.

    The normal acts along coordinate 2 and closes at 1 m/s; the tangents act along
    coordinates 0 and 1 with effective masses ``masses``, at the tangential velocity
    ``sliding``. ``live_tangents=1`` zeroes the second tangent's Jacobian, as a pair
    whose second direction no joint can move. The normal links itself, a ring of
    one, and the pair's coefficient 0.5 makes it slide.
    """
    H = np.diag((masses[0], masses[1], 1.0, 1.0, 1.0, 1.0))
    L = np.linalg.cholesky(H)
    J = np.zeros((3, _DOFS))
    J[0, 2] = J[1, 0] = 1.0
    if live_tangents == 2:
        J[2, 1] = 1.0
    Y = np.linalg.solve(H, J.T).T
    cfm = 1.0e-5
    return {
        "J": J,
        "Y": Y,
        "Z": np.linalg.solve(L, J.T).T,
        "L": L,
        "cfm": cfm,
        "diag": np.einsum("ij,ij->i", J, Y) + cfm,
        "kinds": np.array((0, 1, 1), np.int32),
        "types": np.array((PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION)),
        "parents": np.array((0, 0, 0), np.int32),
        "mu": np.array((0.0, 0.5, 0.5)),
        "rhs": np.zeros(3),
        "impulses": np.zeros(3),
        "velocity": np.array((sliding[0], sliding[1], -1.0, 0.0, 0.0, 0.0)),
    }


def limited_contact_case(limit=0.5):
    """A contact whose impulse pushes a velocity-limited coordinate past its limit.

    The normal closes at 2 m/s along coordinate 2 with a lever onto coordinate 3,
    which moves at 0.3 against an upper limit of ``limit``. Two Jacobi sweeps leave
    the limit violated by about 0.76; converged, the limit holds with an impulse.
    """
    H = np.eye(_DOFS)
    H[3, 3] = 0.2
    L = np.linalg.cholesky(H)
    J = np.zeros((2, _DOFS))
    J[0, 2], J[0, 3] = 1.0, 0.6
    J[1, 3] = -1.0  # upper limit: limit - qd_3 >= 0
    Y = np.linalg.solve(H, J.T).T
    cfm = 1.0e-5
    return {
        "J": J,
        "Y": Y,
        "Z": np.linalg.solve(L, J.T).T,
        "L": L,
        "cfm": cfm,
        "diag": np.einsum("ij,ij->i", J, Y) + cfm,
        "kinds": np.array((0, 0), np.int32),
        "types": np.array((PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT), np.int32),
        "parents": np.array((-1, -1), np.int32),
        "mu": np.zeros(2),
        "rhs": np.array((0.0, limit)),
        "impulses": np.zeros(2),
        "velocity": np.array((0.0, 0.0, -2.0, 0.3, 0.0, 0.0)),
    }


def sliding_stance(device):
    """Two box feet sliding forward at 0.5 m/s on low friction (:data:`SLIDING_FEET`), one world.

    Each foot is a patch: a ring of four normals with two anchors. The floating,
    legged body couples all sixteen rows (the response block of the first step
    has a condition number of about 500 on its range), so a parallel tier needs
    many sweeps where a single contact needs one or two.
    """
    return standing_model(device=device, worlds=1, **SLIDING_FEET)


def split_world_case(solver, world=0):
    """One world's rows of a CPU split solver's last step as a reference case, and the velocity of its own solve.

    Split mode keeps each articulation's columns in its size group (the world's
    articulations in index order) and adds the predicted velocity's ``J v_hat``
    into the right-hand side; the case keeps the bias apart, ``v_hat`` as its
    velocity and cold impulses. Contact and limit rows clamp at zero and
    friction rows pair; a world with any other row, such as a drive, mimic or
    connect row, raises ``ValueError``. Returns the case and ``v_hat + Y^T lambda``
    of the impulses the split sweep solved.
    """
    count = int(solver.constraint_count.numpy()[world])
    types = solver.row_type.numpy()[world, :count]
    unmodelled = sorted(
        set(types.tolist())
        - {
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
            PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
        }
    )
    if unmodelled:
        raise ValueError(f"split_world_case models contact, friction and limit rows only, not row types {unmodelled}")
    worlds, starts = solver.art_to_world.numpy(), solver.articulation_dof_start.numpy()
    predicted = solver.v_hat.numpy().astype(np.float64)
    blocks = []
    for size in solver.size_groups:
        jacobian, response = solver.J_by_size[size].numpy(), solver.Y_by_size[size].numpy()
        for group, art in enumerate(solver.group_to_art[size].numpy()):
            if worlds[art] == world:
                start = int(starts[art])
                blocks.append(
                    (int(art), jacobian[group, :count], response[group, :count], predicted[start : start + size])
                )
    blocks.sort(key=lambda block: block[0])
    J = np.concatenate([block[1] for block in blocks], axis=1).astype(np.float64)
    Y = np.concatenate([block[2] for block in blocks], axis=1).astype(np.float64)
    velocity = np.concatenate([block[3] for block in blocks])
    cfm = solver.row_cfm.numpy()[world, :count].astype(np.float64)
    case = {
        "J": J,
        "Y": Y,
        "Z": None,
        "L": None,
        "cfm": cfm,
        "diag": np.einsum("ij,ij->i", J, Y) + cfm,
        "kinds": np.where(types == PGS_CONSTRAINT_TYPE_FRICTION, 1, 0).astype(np.int32),
        "types": types,
        "parents": solver.row_parent.numpy()[world, :count].astype(np.int32),
        "mu": solver.row_mu.numpy()[world, :count].astype(np.float64),
        "rhs": solver.rhs.numpy()[world, :count].astype(np.float64) - J @ velocity,
        "impulses": np.zeros(count),
        "velocity": velocity,
    }
    return case, velocity + Y.T @ solver.impulses.numpy()[world, :count].astype(np.float64)


def prescaled_joint_velocity(model):
    """The model's joint velocity after the pre-solve prescale of a step with joint velocity limits.

    With ``enable_joint_velocity_limits`` the solver scales each articulation's
    velocity by one ratio, the smallest ``limit / |qd|`` over its DOFs (PhysX's
    ``maxJointVelocity`` ratio), and predicts the step's velocity from the
    result. Split mode, which builds this module's CPU rows, has no velocity
    limits and predicts from the unscaled velocity. The launch takes the
    solver's lanes and thread count (``_velocity_prescale_launch``).
    """
    lanes, threads = source._velocity_prescale_launch(
        model.articulation_start.numpy(), joint_count=model.joint_count, cuda=model.device.is_cuda
    )
    qd = wp.clone(model.joint_qd)
    wp.launch(
        prescale_joint_velocity_limits,
        dim=threads,
        inputs=[
            model.articulation_count,
            lanes,
            model.articulation_start,
            model.joint_type,
            model.joint_child,
            model.joint_qd_start,
            model.joint_dof_dim,
            model.joint_velocity_limit,
            model.body_flags,
            wp.full(1, -1, dtype=wp.int32, device=model.device),  # drive slots: unread without the fused clamp
            0,
            model.joint_qd,
        ],
        outputs=[qd],
        device=model.device,
    )
    return qd.numpy()


def with_velocity_limit_rows(case, mass_matrix, limits, cfm):
    """``case`` with every DOF's lower and upper velocity-limit rows appended, as the matrix-free builder writes them.

    The rows ``J = +e_d`` and ``J = -e_d`` hold ``limit + J v >= 0`` without
    bias and clamp at zero like contact rows; ``mass_matrix`` is the joint-space
    mass matrix of the case's columns.
    """
    J = np.repeat(np.eye(len(limits)), 2, axis=0) * np.tile((1.0, -1.0), len(limits))[:, None]
    count = len(J)
    limited = dict(case)
    limited["J"] = np.vstack((case["J"], J))
    limited["Y"] = np.vstack((case["Y"], np.linalg.solve(mass_matrix, J.T).T))
    limited["cfm"] = np.concatenate((np.broadcast_to(case["cfm"], len(case["J"])), np.full(count, cfm)))
    limited["diag"] = np.einsum("ij,ij->i", limited["J"], limited["Y"]) + limited["cfm"]
    limited["kinds"] = np.concatenate((case["kinds"], np.zeros(count, np.int32)))
    limited["types"] = np.concatenate(
        (case["types"], np.full(count, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT, np.int32))
    )
    limited["parents"] = np.concatenate((case["parents"], np.full(count, -1, np.int32)))
    limited["mu"] = np.concatenate((case["mu"], np.zeros(count)))
    limited["rhs"] = np.concatenate((case["rhs"], np.repeat(np.asarray(limits, np.float64), 2)))
    limited["impulses"] = np.concatenate((case["impulses"], np.zeros(count)))
    return limited


def friction_alignment(impulses, velocity):
    """Cosine between a pair's tangent impulse and the negated tangential velocity it leaves (1: maximal dissipation)."""
    friction, slip = impulses[1:3], velocity[:2]
    return float(-friction @ slip / (np.linalg.norm(friction) * np.linalg.norm(slip)))


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


# Sweep budgets compare a tier's velocity with the serial fixed point's, relative to its largest component. Two
# sweeps must leave at least _TRUNCATED_GAP of it, or the fixture converges too fast to show what a budget carries
# (the sliding stance: 3.4e-2); the budget must close that gap to _BUDGET_SHRINK of itself and land within
# _BUDGET_GAP of the scale (float32 solves of the stance agree to about 1e-7 of it).
_TRUNCATED_GAP = 1.0e-3
_BUDGET_SHRINK = 1.0e-3
_BUDGET_GAP = 1.0e-5


def assert_budget_reaches_the_fixed_point(test, truncated, budget, scale):
    """Require a two-sweep tier short of the serial fixed point and its sweep budget carrying it there.

    ``truncated`` and ``budget`` are the largest velocity gaps to the serial
    solution after two sweeps and after the budget, ``scale`` the solution's
    largest velocity. A fixture that two sweeps converge fails here instead of
    passing vacuously.
    """
    gaps = {"two sweeps": truncated, "budget": budget, "scale": scale}
    test.assertGreater(truncated, _TRUNCATED_GAP * scale, f"two sweeps already converge the fixture: {gaps}")
    test.assertLess(budget, _BUDGET_SHRINK * truncated, f"the budget leaves the truncated gap open: {gaps}")
    test.assertLess(budget, _BUDGET_GAP * scale, f"the budget stops short of the serial fixed point: {gaps}")


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


# Golden text of the isotropic pair step: the pair's first tangent row's lane gives both rows the smaller step,
# unless the sibling is not an admitted friction row of the same normal or either step is zero.
_ISOTROPIC_STEP = (
    "    for (int i = lane; i < n_rows; i += NT) {\n"
    "        if (s_kind[i] == 1 && i == s_parent[i] + 1) {\n"
    "            const int j = i + 1;\n"
    "            if (j < n_rows && s_kind[j] == 1 && s_parent[j] == s_parent[i]) {\n"
    "                const float step_a = s_step[i];\n"
    "                const float step_b = s_step[j];\n"
    "                if (step_a > 0.0f && step_b > 0.0f) {\n"
    "                    const float step_pair = fminf(step_a, step_b);\n"
    "                    s_step[i] = step_pair;\n"
    "                    s_step[j] = step_pair;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    SYNC();\n"
    "    if (0 == 3) return;\n"
    "    float t_k = 1.0f;\n"
)


def _without_isotropic_step(text):
    """Remove the isotropic pair step, comment included, from a tier's native source."""
    return text.replace(source._ISOTROPIC_PAIR_STEP, "")


class TestParallelPatchReference(unittest.TestCase):
    def test_linked_factory_pools_only_the_friction_disk(self):
        """Build linked tiers as the point tiers with the golden ring walk and isotropic step; refuse world rows."""
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
                self.assertEqual(linked.key, point.key + "_linked_iso")
                self.assertEqual((point._fpgs_pair_step, linked._fpgs_pair_step), ("per_row", "isotropic"))
                point_source = point._fpgs_row_native_source
                linked_source = linked._fpgs_row_native_source
                for text, golden in (
                    (point_source, _POINT_WARP_PAIR),
                    (point_source, _POINT_BLOCK_PAIR),
                    (linked_source, _LINKED_WARP_PAIR),
                    (linked_source, _LINKED_BLOCK_PAIR),
                    (linked_source, _ISOTROPIC_STEP),
                ):
                    self.assertEqual(text.count(golden), 1, golden)
                self.assertNotIn("step_pair", point_source)
                # Nothing but the two projections and the pair step differs from the point tier.
                restored = _without_isotropic_step(linked_source).replace(_LINKED_WARP_PAIR, _POINT_WARP_PAIR)
                self.assertEqual(restored.replace(_LINKED_BLOCK_PAIR, _POINT_BLOCK_PAIR), point_source)
                self.assertEqual(linked._fpgs_block_dim, point._fpgs_block_dim)
                # The per-row law of the linked tiers remains available to compare against.
                per_row = source._get_pgs_solve_parallel_kernel(
                    _CAPACITY, 32, _DOFS, "120", linked_rows=True, pair_step="per_row", **options
                )
                self.assertEqual(per_row.key, point.key + "_linked")
                self.assertEqual(per_row._fpgs_row_native_source, _without_isotropic_step(linked_source))
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
                    # Without links the ring walk stops at the normal: the point law exactly, with either pair step.
                    for pair_step in ("per_row", "isotropic"):
                        same, _ = parallel_sweep_reference(point_case, linked=True, pair_step=pair_step, **options)
                        expected, _ = parallel_sweep_reference(point_case, linked=False, pair_step=pair_step, **options)
                        np.testing.assert_array_equal(same, expected)
                    np.testing.assert_array_equal(
                        expected, parallel_sweep_reference(point_case, linked=True, **options)[0]
                    )
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

    def test_point_tiers_opt_into_the_isotropic_step(self):
        """Give point tiers the isotropic pair step only on request, and reject unknown pair steps."""
        options = {"rows": 32, "sweeps": 24, "matrix_free": True, "inkernel_response": (6, 0, 0, 0)}
        point = source._get_pgs_solve_parallel_kernel(_CAPACITY, 32, _DOFS, "120", **options)
        isotropic = source._get_pgs_solve_parallel_kernel(_CAPACITY, 32, _DOFS, "120", pair_step="isotropic", **options)
        self.assertEqual(isotropic.key, point.key + "_iso")
        self.assertEqual(isotropic._fpgs_row_native_source.count(_ISOTROPIC_STEP), 1)
        self.assertEqual(_without_isotropic_step(isotropic._fpgs_row_native_source), point._fpgs_row_native_source)
        with self.assertRaisesRegex(ValueError, "pair_step"):
            source._get_pgs_solve_parallel_kernel(_CAPACITY, 32, _DOFS, "120", pair_step="radial", **options)

    def test_solver_steps_linked_pairs_isotropically(self):
        """Step every linked pair isotropically, and point pairs per row unless the private experiment opts in."""
        decide = SolverFeatherPGS._parallel_pair_step
        for anchors, flag, expected in (
            (True, None, "isotropic"),
            (True, "per_row", "isotropic"),
            (False, None, "per_row"),
            (False, "per_row", "per_row"),
            (False, "isotropic", "isotropic"),
        ):
            with self.subTest(anchors=anchors, flag=flag), patch.dict(os.environ):
                os.environ.pop(_PAIR_STEP_FLAG, None)
                if flag is not None:
                    os.environ[_PAIR_STEP_FLAG] = flag
                self.assertEqual(decide(SimpleNamespace(_friction_anchors_enabled=anchors)), expected)
        with patch.dict(os.environ, {_PAIR_STEP_FLAG: "radial"}), self.assertRaisesRegex(ValueError, "radial"):
            decide(SimpleNamespace(_friction_anchors_enabled=False))

    def test_isotropic_pairs_slide_with_maximal_dissipation(self):
        """Slide a pair with fourfold unequal tangent masses against its velocity, as the serial pair solve does.

        Per-row steps turn the converged friction towards the heavier tangent (cosine 0.86
        against the slip); the isotropic step aligns it with the slip.
        """
        case = sliding_pair_case()
        for matrix_free, inkernel in itertools.product((False, True), repeat=2):
            options = {"matrix_free": matrix_free, "inkernel": inkernel, "exact": False, "nesterov": True, "tol": 0.0}
            with self.subTest(matrix_free=matrix_free, inkernel=inkernel):
                impulses, velocity = parallel_sweep_reference(case, sweeps=2000, linked=True, **options)
                per_row, per_row_velocity = parallel_sweep_reference(
                    case, sweeps=2000, linked=True, pair_step="per_row", **options
                )
                # The pair slides on its disk, radius 0.5 times the normal impulse.
                self.assertAlmostEqual(np.linalg.norm(impulses[1:3]), 0.5 * impulses[0], delta=1.0e-9)
                self.assertGreater(np.linalg.norm(velocity[:2]), 0.5)
                self.assertGreaterEqual(friction_alignment(impulses, velocity), 1.0 - 1.0e-9)
                self.assertLess(friction_alignment(per_row, per_row_velocity), 0.9)

    def test_isotropic_step_keeps_a_lone_live_tangent(self):
        """Drive the live tangent of a pair whose sibling has no Jacobian to the disk, as without the pair step."""
        case = sliding_pair_case(sliding=(1.0, 0.0), live_tangents=1)
        for matrix_free, inkernel in itertools.product((False, True), repeat=2):
            options = {"matrix_free": matrix_free, "inkernel": inkernel, "exact": False, "nesterov": True, "tol": 0.0}
            with self.subTest(matrix_free=matrix_free, inkernel=inkernel):
                impulses, _ = parallel_sweep_reference(case, sweeps=2000, linked=True, **options)
                per_row, _ = parallel_sweep_reference(case, sweeps=2000, linked=True, pair_step="per_row", **options)
                np.testing.assert_allclose(impulses, per_row, rtol=1.0e-12, atol=1.0e-15)
                self.assertAlmostEqual(-impulses[1], 0.5 * impulses[0], delta=1.0e-9)
                self.assertEqual(impulses[2], 0.0)

    def test_velocity_limit_pass_reaches_every_tier_variant(self):
        """Stage the velocity-limit flag and append the last word to each tier variant only on request.

        The keyed-row tiers read their rows from a layout that plans joint velocity
        limits, as a solver with ``enable_joint_velocity_limits`` builds them.
        """
        plain_tiers = (source._get_pgs_solve_parallel_kernel, (_CAPACITY, 32, _DOFS, "120"))
        keyed_tiers = (
            keyed_rows.get_parallel_factory(
                source._get_pgs_solve_parallel_kernel, KeyedLayout(16, 6, 0, 16, 128, True, (1 << 16) - 1)
            ),
            (128, 32, 22, "120"),
        )
        variants = {
            "response block": (plain_tiers, {"matrix_free": False}, "A_AT(i, j) * s_x[j]"),
            "matrix-free": (plain_tiers, {"matrix_free": True}, "J_world.data[jy_world_base + i * 6 + d]"),
            "in-kernel matrix-free": (
                plain_tiers,
                {"matrix_free": True, "inkernel_response": (6, 0, 0, 0)},
                "ZAT(d, i)",
            ),
            "in-kernel response block": (
                plain_tiers,
                {"matrix_free": False, "inkernel_response": (6, 0, 0, 0)},
                "ZAT(d, i)",
            ),
            "keyed rows": (keyed_tiers, {"matrix_free": True, "inkernel_response": (16, 6, 0, 16)}, "ZAT(d, i)"),
        }
        for name, ((factory, args), options, row_read) in variants.items():
            with self.subTest(variant=name):
                settings = {"rows": 32, "sweeps": 24, "linked_rows": True, **options}
                plain = factory(*args, **settings)
                last_word = factory(*args, velocity_limit_pass=True, **settings)
                self.assertEqual(last_word.key, plain.key + "_lw")
                self.assertEqual((plain._fpgs_velocity_limit_pass, last_word._fpgs_velocity_limit_pass), (False, True))
                text = last_word._fpgs_row_native_source
                self.assertNotIn("s_vlim", plain._fpgs_row_native_source)
                self.assertEqual(text.count("s_vlim[i] = row_type == 4 ? 1 : 0;"), 1)
                self.assertEqual(text.count("Last word: ordered push-only velocity-limit pass"), 1)
                self.assertIn(row_read, text)
                # The pass runs between the final sweep and the velocity reconstruction.
                self.assertLess(text.index("Last word"), text.index("v_out.data[global_dof] ="))
                self.assertGreater(text.index("Last word"), text.index("for (int sweep = 0;"))

    def test_solver_builds_the_pass_for_limited_linked_tiers(self):
        """Build the last-word pass only for linked tiers with joint velocity limits, and only on request."""
        decide = SolverFeatherPGS._parallel_velocity_limit_pass
        for requested, anchors, limits in itertools.product((False, True), repeat=3):
            probe = SimpleNamespace(
                mf_gs_parallel_linked_velocity_limit_pass=requested,
                _friction_anchors_enabled=anchors,
                enable_joint_velocity_limits=limits,
            )
            with self.subTest(requested=requested, anchors=anchors, limits=limits):
                self.assertEqual(decide(probe), requested and anchors and limits)

    def test_last_word_holds_a_truncated_limit(self):
        """Hold the limit after two sweeps with the last word, and leave a converged tier unchanged."""
        case = limited_contact_case()
        for matrix_free, inkernel in itertools.product((False, True), repeat=2):
            options = {"matrix_free": matrix_free, "inkernel": inkernel, "exact": False, "nesterov": True, "tol": 0.0}
            with self.subTest(matrix_free=matrix_free, inkernel=inkernel):
                _, truncated = parallel_sweep_reference(case, sweeps=2, linked=True, **options)
                _, held = parallel_sweep_reference(case, sweeps=2, linked=True, velocity_limit_pass=True, **options)
                self.assertLess(case["J"][1] @ truncated + case["rhs"][1], -0.5)
                # The push divides by the regularized diagonal: cfm leaves 3e-6 of the limit.
                self.assertGreaterEqual(case["J"][1] @ held + case["rhs"][1], -1.0e-5 * case["rhs"][1])
                converged, _ = parallel_sweep_reference(case, sweeps=2000, linked=True, **options)
                unchanged, _ = parallel_sweep_reference(
                    case, sweeps=2000, linked=True, velocity_limit_pass=True, **options
                )
                np.testing.assert_allclose(unchanged, converged, atol=1.0e-12)

    def test_last_word_holds_a_pushed_sphere_at_its_limit(self):
        """Hold a sphere the step pushes past its limit: two sweeps hold it only with the last word, 2000 either way.

        The rows are the first step of :func:`_pushed_sphere` as a CPU split solver
        builds them, started from the prescaled velocity the matrix-free route
        predicts from, with every DOF's velocity-limit rows added as the matrix-free
        builder writes them (split mode has neither). Two sweeps of the matrix-free
        linked tier (the float64 reference) leave the sphere 43 % past its limit;
        the pass holds it to 1e-5 (1.7e-6 past: the regularized diagonal's share).
        A sphere that starts past its limit cannot show this: the prescale puts it
        on the limit before the solve, and two sweeps leave it 0.9 % under.
        TestPatchParallelRoutes runs the solver's tiers on this sphere.
        """
        model = _pushed_sphere("cpu")
        limits = model.joint_velocity_limit.numpy().astype(np.float64)
        model.joint_qd.assign(prescaled_joint_velocity(model))  # what the matrix-free route predicts from
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.2, sparse_mass_matrix="off", cached_dynamics="off")
        _rollout(model, solver, 1)
        (size,) = solver.size_groups
        mass_matrix = solver.H_by_size[size].numpy()[0].astype(np.float64)
        case = with_velocity_limit_rows(split_world_case(solver)[0], mass_matrix, limits, solver.pgs_cfm)
        options = {"matrix_free": True, "inkernel": False, "exact": False, "nesterov": True, "tol": 0.0, "linked": True}
        speeds = {}
        for sweeps, last_word in itertools.product((2, 2000), (False, True)):
            _, velocity = parallel_sweep_reference(case, sweeps=sweeps, velocity_limit_pass=last_word, **options)
            speeds[sweeps, last_word] = float(np.max(np.abs(velocity[:2])))
        limit = float(limits[0])
        self.assertGreater(speeds[2, False], 1.05 * limit, f"two sweeps already hold the limit: {speeds}")
        for key in ((2, True), (2000, False), (2000, True)):
            self.assertLessEqual(speeds[key], limit * (1.0 + 1.0e-5), (key, speeds))


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
        keyed._friction_patches = general._friction_patches
        install_cpu_keyed_rows(keyed)
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
        self.assertEqual(linked.key, point.key + "_linked_iso")
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
        for golden in (_LINKED_WARP_PAIR, _LINKED_BLOCK_PAIR, _ISOTROPIC_STEP):
            self.assertEqual(text.count(golden), 1, golden)


_SWITCH = "FEATHER_PGS_PATCH_PARALLEL_ROWS"
_PAIR_STEP_FLAG = "FEATHER_PGS_POINT_PAIR_STEP"


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

    def test_admission_says_where_the_tiers_share_the_serial_fixed_point(self):
        """Claim the serial sweep's fixed point for patch tiers only where both accumulate velocity-limit rows.

        The tiers project joint velocity-limit rows as accumulated unilateral rows. The
        serial sweeps do so only under ``velocity_limit_law="accumulated"`` and outside
        the PhysX grasp schedule; under the stateless clamp the two converge to
        different velocities wherever a limit binds, so the reason must not claim a
        shared fixed point there.
        """
        for limits, fraction, law, schedule, shared in (
            (False, 0.0, "stateless", "interleaved", True),
            (True, float("inf"), "stateless", "interleaved", True),  # the limits never activate
            (True, 0.0, "accumulated", "interleaved", True),
            (True, 0.0, "stateless", "interleaved", False),
            (True, 0.0, "accumulated", "physx_grasp", False),  # the grasp schedule keeps PhysX's clamp
        ):
            probe = SimpleNamespace(
                _point_contact_paths=False,
                _law_agnostic_contact_paths=True,
                pgs_mode="matrix_free",
                pgs_schedule=schedule,
                enable_joint_velocity_limits=limits,
                velocity_limit_activation_fraction=fraction,
                velocity_limit_law=law,
                _environment_options=frozenset(),
            )
            for name in ("_velocity_limit_sweep_law", "_parallel_velocity_limit_agreement"):
                setattr(probe, name, MethodType(getattr(SolverFeatherPGS, name), probe))
            with self.subTest(limits=limits, fraction=fraction, law=law, schedule=schedule):
                with patch.dict(os.environ, {_SWITCH: "1"}):
                    admitted, reason = SolverFeatherPGS._admit_parallel_contact_paths(probe)
                self.assertTrue(admitted)
                self.assertIn("one isotropic step per pair", reason)
                self.assertEqual("share the serial sweep's fixed point" in reason, shared, reason)
                self.assertEqual("fixed points differ wherever a limit binds" in reason, not shared, reason)

    def test_route_record_reads_the_built_tiers(self):
        """Record the tiers' ring-pooled projection and sweep budgets from the kernels built, not from the law."""
        for anchors, linked in itertools.product((False, True), repeat=2):
            options = {"matrix_free": True, "linked_rows": linked}
            kernels = [
                source._get_pgs_solve_parallel_kernel(
                    _CAPACITY, 32, _DOFS, "120", rows=rows, min_rows=rows - 32, sweeps=sweeps, **options
                )
                for rows, sweeps in ((32, 24), (64, 34))
            ]
            probe = SimpleNamespace(
                _pgs_solve_mf_gs_incremental_kernels=kernels,
                _par_tiers=[(32, 0), (64, 32)],
                _par_sweeps_rule="auto",
                _par_sweeps_reason="mf_gs_parallel_linked_sweeps='auto': ...",
                _ink_rows=64,
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
                self.assertEqual(tiers["tiers"][0]["kernel"].endswith("_linked_iso"), linked)
                self.assertEqual(tiers["pair_step"], "isotropic" if linked else "per_row")
                self.assertEqual([tier["sweeps"] for tier in tiers["tiers"]], [24, 34])
                self.assertEqual((tiers["sweeps"], tiers["sweeps_rule"]), (34, "auto"))
                # The applied linked budget is the largest built linked tier's; point tiers apply none.
                self.assertEqual(path_selection._EFFECTIVE["mf_gs_parallel_linked_sweeps"](probe), 34 if linked else 0)

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


_SWEEPS_FLAG = "FEATHER_PGS_PATCH_PARALLEL_SWEEPS"


class TestLinkedTierSweeps(unittest.TestCase):
    """Give the linked (patch) tiers their own sweep budget, by count or by the ``"auto"`` rule."""

    def test_requests_are_validated(self):
        """Accept None, "auto" and positive counts (decimal text too); reject the rest and name their source."""
        for value, expected in ((None, None), ("auto", "auto"), (64, 64), (np.int64(40), 40), ("48", 48)):
            with self.subTest(value=value):
                self.assertEqual(source._linked_sweeps_request(value, "option"), expected)
        model = _cpu_arm()
        for value in (0, -3, True, 2.5, "AUTO", "fast"):
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop(_SWEEPS_FLAG, None)
                with self.assertRaisesRegex(ValueError, "mf_gs_parallel_linked_sweeps must be"):
                    SolverFeatherPGS(model, mf_gs_parallel_linked_sweeps=value)
                if isinstance(value, str):
                    os.environ[_SWEEPS_FLAG] = value
                    with self.assertRaisesRegex(ValueError, f"{_SWEEPS_FLAG} must be"):
                        SolverFeatherPGS(model)

    def test_auto_budget_grows_with_iterations_and_tier_rows(self):
        """Sweep ceil(3 * pgs_iterations * sqrt(rows / 32)) times within the linked tiers' bounds.

        The budget is at least mf_gs_parallel_sweeps and at most 128, unless
        mf_gs_parallel_sweeps is larger.
        """
        for rows, iterations, sweeps, expected in (
            (32, 12, 24, 36),
            (64, 12, 24, 51),
            (96, 12, 24, 63),
            (128, 12, 24, 72),  # 3 * 12 * 2 exactly: no rounding up past a whole budget
            (48, 8, 24, 30),
            (64, 8, 24, 34),
            (32, 8, 24, 24),  # never fewer sweeps than mf_gs_parallel_sweeps
            (128, 40, 24, 128),  # the cap
            (32, 4, 160, 160),  # mf_gs_parallel_sweeps above the cap keeps its own budget
            (64, 0, 24, 24),
        ):
            with self.subTest(rows=rows, iterations=iterations, sweeps=sweeps):
                self.assertEqual(source._auto_linked_sweeps(rows, iterations, sweeps), expected)

    def test_solver_budgets_linked_tiers_only(self):
        """Budget linked tiers by the request and point tiers by mf_gs_parallel_sweeps; record the rule and source."""
        tiers = [(32, 0), (64, 32), (96, 64), (128, 96)]
        for anchors, linked, environment in itertools.product((False, True), (None, 40, "auto"), (False, True)):
            probe = SimpleNamespace(
                _friction_anchors_enabled=anchors,
                _environment_options=frozenset({"mf_gs_parallel_linked_sweeps"} if environment else ()),
                mf_gs_parallel_linked_sweeps=linked,
                mf_gs_parallel_sweeps=24,
                pgs_iterations=12,
            )
            with self.subTest(anchors=anchors, linked=linked, environment=environment):
                budgets, rule, reason = SolverFeatherPGS._parallel_tier_sweeps(probe, tiers)
                if not anchors or linked is None:
                    self.assertEqual((budgets, rule), ([24] * 4, "mf_gs_parallel_sweeps"))
                    self.assertIn("mf_gs_parallel_sweeps", reason)
                    continue
                if linked == "auto":
                    self.assertEqual((budgets, rule), ([36, 51, 63, 72], "auto"))
                    self.assertIn("pgs_iterations=12", reason)
                    self.assertIn("at least mf_gs_parallel_sweeps=24 and at most 128", reason)
                else:
                    self.assertEqual((budgets, rule), ([40] * 4, "mf_gs_parallel_linked_sweeps"))
                self.assertTrue(reason.startswith(_SWEEPS_FLAG if environment else "mf_gs_parallel_linked_sweeps"))
        # mf_gs_parallel_sweeps above the cap keeps its own budget; the reason states the bounds that way round.
        probe = SimpleNamespace(
            _friction_anchors_enabled=True,
            _environment_options=frozenset(),
            mf_gs_parallel_linked_sweeps="auto",
            mf_gs_parallel_sweeps=160,
            pgs_iterations=12,
        )
        budgets, rule, reason = SolverFeatherPGS._parallel_tier_sweeps(probe, tiers)
        self.assertEqual((budgets, rule), ([160] * 4, "auto"))
        self.assertIn("at most 128 unless mf_gs_parallel_sweeps is larger", reason)
        self.assertNotIn("[mf_gs_parallel_sweeps", reason)

    def test_option_records_its_source(self):
        """Report the override as the option's source; a solver without linked tiers applies no linked budget."""
        model = _cpu_arm()
        for environment, argument, expected, source_name in (
            (None, None, None, "default"),
            (None, "auto", "auto", "argument"),
            ("64", None, 64, "environment"),
            ("auto", 32, "auto", "environment"),
            ("48", 48, 48, "argument"),
        ):
            with self.subTest(environment=environment, argument=argument), patch.dict(os.environ):
                os.environ.pop(_SWEEPS_FLAG, None)
                if environment is not None:
                    os.environ[_SWEEPS_FLAG] = environment
                solver = SolverFeatherPGS(model, friction_anchor_beta=0.2, mf_gs_parallel_linked_sweeps=argument)
                self.assertEqual(solver.mf_gs_parallel_linked_sweeps, expected)
                selected = solver.get_path_selection()
                option = selected["options"]["mf_gs_parallel_linked_sweeps"]
                self.assertEqual((option["requested"], option["source"]), (argument, source_name))
                self.assertEqual(option["applied"], 0)  # no tiers on CPU
                self.assertEqual(selected["environment"]["applied"].get(_SWEEPS_FLAG), environment)

    def test_budget_carries_a_truncated_tier_to_the_serial_fixed_point(self):
        """Leave a two-sweep linked tier short of a sliding stance's serial fixed point, and reach it within budget.

        The rows are the first step of :func:`sliding_stance` as a CPU split solver
        builds them, and that solver's serial sweep (2000 iterations) gives the
        fixed point. The matrix-free linked tier (the float64 reference) stops
        3.4e-2 of the velocity scale short of it after two sweeps and lands within
        1.1e-7 of the scale after 2000 (:func:`assert_budget_reaches_the_fixed_point`).
        TestPatchParallelPhysics runs the solver's tiers on the same stance.
        """
        model = sliding_stance("cpu")
        solver = SolverFeatherPGS(
            model, friction_anchor_beta=0.2, pgs_iterations=2000, sparse_mass_matrix="off", cached_dynamics="off"
        )
        _rollout(model, solver, 1)
        case, serial = split_world_case(solver)
        # Sixteen contact rows: two patch rings of four normals, each ring with two anchors.
        rings = {
            tuple(sorted(ring_rows(case["parents"], int(normal))))
            for normal in np.flatnonzero(case["types"] == PGS_CONSTRAINT_TYPE_CONTACT)
        }
        self.assertEqual(sorted(len(ring) for ring in rings), [4, 4])
        self.assertEqual(int(np.count_nonzero(case["types"] == PGS_CONSTRAINT_TYPE_FRICTION)), 8)
        self.assertEqual(len(case["types"]), 16)
        options = {"matrix_free": True, "inkernel": False, "exact": False, "nesterov": True, "tol": 0.0, "linked": True}
        gaps = {}
        for sweeps in (2, 2000):
            _, velocity = parallel_sweep_reference(case, sweeps=sweeps, **options)
            gaps[sweeps] = float(np.max(np.abs(velocity - serial)))
        assert_budget_reaches_the_fixed_point(self, gaps[2], gaps[2000], float(np.max(np.abs(serial))))


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

    def test_tiers_slide_pairs_with_maximal_dissipation(self):
        """Converge a sliding pair of unequal tangent masses to maximal dissipation, and keep a lone live tangent.

        With tolerance 0 and 2000 sweeps every tier reaches the reference's fixed
        point: the friction opposes the slip it leaves (cosine at least 1 - 1e-5),
        and the live tangent of a pair whose sibling has no Jacobian saturates its disk.
        """
        device = wp.get_device("cuda:0")
        arch = str(device.arch)
        cases = {
            "unequal masses": sliding_pair_case(),
            "one live tangent": sliding_pair_case(sliding=(1.0, 0.0), live_tangents=1),
        }
        for (name, case), matrix_free, inkernel in itertools.product(cases.items(), (False, True), (False, True)):
            options = {"matrix_free": matrix_free, "inkernel": inkernel, "exact": False, "nesterov": True}
            with self.subTest(case=name, **options):
                kernel = source._get_pgs_solve_parallel_kernel(
                    _CAPACITY,
                    32,
                    _DOFS,
                    arch,
                    rows=32,
                    sweeps=2000,
                    tol=0.0,
                    matrix_free=matrix_free,
                    inkernel_response=(_DOFS, 0, 0, 0) if inkernel else None,
                    linked_rows=True,
                )
                impulses, velocity = _launch_parallel(kernel, case, inkernel=inkernel, device=device)
                impulses, velocity = impulses.astype(np.float64), velocity.astype(np.float64)
                expected, expected_velocity = parallel_sweep_reference(
                    case, sweeps=2000, tol=0.0, linked=True, **options
                )
                np.testing.assert_allclose(impulses, expected, rtol=1.0e-4, atol=1.0e-6)
                np.testing.assert_allclose(velocity, expected_velocity, rtol=1.0e-4, atol=1.0e-6)
                if name == "unequal masses":
                    self.assertGreaterEqual(friction_alignment(impulses, velocity), 1.0 - 1.0e-5)
                else:
                    self.assertAlmostEqual(-impulses[1], 0.5 * impulses[0], delta=1.0e-5)

    def test_last_word_holds_a_truncated_tier_at_the_limit(self):
        """Exit a tier stopped after two sweeps within its velocity limit, and change a converged one by 1e-6 at most.

        The last word pushes the limit row that the contact's impulse drove past its limit
        back onto it (relative 1e-5); with 2000 sweeps the limit holds already, and the
        pass leaves the impulses within 1e-6.
        """
        device = wp.get_device("cuda:0")
        arch = str(device.arch)
        case = limited_contact_case()
        limit = case["rhs"][1]
        for matrix_free, inkernel, sweeps in itertools.product((False, True), (False, True), (2, 2000)):
            options = {"matrix_free": matrix_free, "inkernel": inkernel, "exact": False, "nesterov": True}
            with self.subTest(sweeps=sweeps, **options):
                results = {}
                for last_word in (False, True):
                    kernel = source._get_pgs_solve_parallel_kernel(
                        _CAPACITY,
                        32,
                        _DOFS,
                        arch,
                        rows=32,
                        sweeps=sweeps,
                        tol=0.0,
                        matrix_free=matrix_free,
                        inkernel_response=(_DOFS, 0, 0, 0) if inkernel else None,
                        linked_rows=True,
                        velocity_limit_pass=last_word,
                    )
                    impulses, velocity = _launch_parallel(kernel, case, inkernel=inkernel, device=device)
                    expected, expected_velocity = parallel_sweep_reference(
                        case, sweeps=sweeps, tol=0.0, linked=True, velocity_limit_pass=last_word, **options
                    )
                    np.testing.assert_allclose(impulses, expected, rtol=1.0e-4, atol=1.0e-5)
                    np.testing.assert_allclose(velocity, expected_velocity, rtol=1.0e-4, atol=1.0e-5)
                    results[last_word] = impulses.astype(np.float64), velocity.astype(np.float64)
                speed = -(case["J"][1] @ results[True][1])  # qd_3
                self.assertLessEqual(speed, limit * (1.0 + 1.0e-5))
                if sweeps == 2:
                    self.assertGreater(-(case["J"][1] @ results[False][1]), 1.5 * limit)
                else:
                    np.testing.assert_allclose(results[True][0], results[False][0], atol=1.0e-6)


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


def _sliding_sphere(device, axes, masses, speed, *, mu=0.5):
    """A sphere on a chain of prismatic joints along ``axes`` (link masses ``masses``, the sphere last) on the ground.

    The sphere starts sliding at ``speed`` along the first two axes. Each tangent of
    its single contact moves the links downstream of its joint, so unequal link masses
    give unequal tangent effective masses; an axis left out leaves its tangent row
    without a Jacobian.
    """
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.density, cfg.mu = 0.0, mu
    parent, joints = -1, []
    for axis, mass in zip(axes, masses, strict=True):
        body = builder.add_link(mass=mass, inertia=wp.mat33(np.eye(3) * 1.0e-3), lock_inertia=True)
        joints.append(builder.add_joint_prismatic(parent=parent, child=body, axis=axis))
        parent = body
    builder.add_shape_sphere(parent, radius=0.05, cfg=cfg)
    builder.add_articulation(joints)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    q, qd = model.joint_q.numpy(), model.joint_qd.numpy()
    q[len(axes) - 1] = 0.049  # the vertical joint, last: the sphere rests 1 mm deep
    qd[: len(speed)] = speed
    model.joint_q.assign(q)
    model.joint_qd.assign(qd)
    return model


def _pushed_sphere(device, *, limit=0.5, dt=1.0 / 240.0):
    """A resting sphere on x, y and z joints limited to ``limit``; a constant force on x adds ``2 * limit`` each step.

    A step with joint velocity limits first scales a starting velocity past the
    limits back onto them (:func:`prescaled_joint_velocity`), so a sphere that
    starts past its limit reaches the solve within it. From rest, or from the
    limit, the ratio is 1: the force on the first joint, which carries the whole
    chain, takes the x velocity past the limit within each step of ``dt``, and
    only the solve's limit rows can take it out.
    """
    masses = (3.0, 0.01, 1.0)
    model = _sliding_sphere(device, (newton.Axis.X, newton.Axis.Y, newton.Axis.Z), masses, (0.0, 0.0))
    model.joint_velocity_limit.assign(np.full(model.joint_dof_count, limit, np.float32))
    force = np.zeros(model.joint_dof_count, np.float32)
    force[0] = sum(masses) * 2.0 * limit / dt
    model.joint_f.assign(force)
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
            self.assertTrue(all(key.endswith("_linked_iso") == linked for key in keys), keys)
            tiers = solver.get_path_selection()["routes"]["sweep"]["tiers"]
            self.assertEqual((tiers["family"], tiers["linked_rows"]), ("parallel", linked))
            self.assertEqual(tiers["pair_step"], "isotropic" if linked else "per_row")
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

    def test_keyed_rows_record_names_the_launched_builders(self):
        """Name in the route record the keyed-row builders one step launches, under both laws.

        The in-kernel parallel sweep (FEATHER_PGS_INK=1) admits keyed rows on CUDA, for patch friction
        under FEATHER_PGS_PATCH_PARALLEL_ROWS=1. The keyed parallel tiers build the direct rows in the
        sweep, which every entry points to; the scatter and the fallback are the row builders a step
        launches outside it.
        """
        model = standing_model()
        with patch.object(source, "_INK_ON", True), patch.dict(os.environ, {_SWITCH: "1"}):
            os.environ.pop("FEATHER_PGS_KEYED_ROWS", None)
            for law, beta in (("point", 0.0), ("patch", 0.2)):
                with self.subTest(law=law), patch.multiple(source, **ROW_SWITCHES_OFF):
                    solver = _matrix_free_solver(model, beta=beta, parallel_rows=32, enable_joint_limits=True)
                    routes = solver.get_path_selection()["routes"]
                    record = routes["contact_rows"]
                    self.assertTrue(record["keyed_rows"]["enabled"], record["keyed_rows"]["reason"])
                    self.assertEqual(routes["sweep"]["tiers"]["family"], "keyed_parallel")
                    self.assertEqual((record["producer"], record["law"]), ("keyed_rows", law))
                    layout = solver._keyed_rows.layout
                    scatter, fallback = (
                        ("scatter_contact_keys", keyed_rows.get_fallback_kernel(layout))
                        if law == "point"
                        else ("scatter_patch_contact_keys", keyed_rows.get_patch_fallback_kernel(layout))
                    )
                    builder = {
                        "method": "keyed_rows",
                        "kernels": [scatter, fallback.key],
                        "law": law,
                        "direct_rows": "routes.sweep.tiers",
                    }
                    self.assertEqual(record["builder_by_dof"], dict.fromkeys(record["builder_by_dof"], builder))
                    self.assertIsNone(record["row_check_by_dof"])
                    with launched_kernels() as keys:
                        _rollout(model, solver, 1)
                    assert_row_builders_named(self, record, keys)

    def test_limited_linked_tiers_take_the_last_word_on_request(self):
        """Build the last-word pass into velocity-limited linked tiers only on request, record it, and hold the limit.

        A constant joint force pushes a resting sphere, whose joints are limited to
        0.5, by twice its limit within every step (:func:`_pushed_sphere`): without
        limit rows two steps carry it past twice the limit. A sphere that starts past
        its limit cannot show the pass, because the pre-solve prescale puts it on the
        limit and its limit rows never bind. Only the pass promises the limit: a tier
        without it stops where its budget or tolerance leaves it (24 sweeps to 1e-4:
        1.9e-5 past after the first step in the float64 reference), while the pass
        holds it to 1e-5. Truncated to two sweeps (tolerance 0), the tier without the
        pass must leave the sphere at least 5 % past its limit, or the fixture cannot
        show what the pass does, and the pass must hold it to 1e-5 (the reference on
        the CPU rows of test_last_word_holds_a_pushed_sphere_at_its_limit: 43.5 % and
        1.7e-6 past). Converged (2000 sweeps, tolerance 0), the tier's limit rows hold
        the limit to 1e-5 without the pass, and the pass moves no velocity by more
        than 1e-6.
        """
        limit = 0.5
        model = _pushed_sphere("cuda:0", limit=limit)
        built = {}
        with patch.dict(os.environ, {_SWITCH: "1"}):
            for limits, requested in itertools.product((False, True), repeat=2):
                built[limits, requested] = _matrix_free_solver(
                    model,
                    beta=0.2,
                    parallel_rows=32,
                    enable_joint_velocity_limits=limits,
                    mf_gs_parallel_linked_velocity_limit_pass=requested,
                )
            budgets = {
                (sweeps, requested): _matrix_free_solver(
                    model,
                    beta=0.2,
                    parallel_rows=32,
                    enable_joint_velocity_limits=True,
                    mf_gs_parallel_linked_velocity_limit_pass=requested,
                    mf_gs_parallel_sweeps=sweeps,
                    mf_gs_parallel_tol=0.0,
                )
                for sweeps, requested in itertools.product((2, 2000), (False, True))
            }
        for (limits, requested), solver in built.items():
            with self.subTest(limits=limits, requested=requested):
                last_word = limits and requested
                keys = [kernel.key for kernel in solver._pgs_solve_mf_gs_incremental_kernels]
                self.assertTrue(keys)
                self.assertTrue(all(key.endswith("_linked_iso_lw" if last_word else "_linked_iso") for key in keys))
                selected = solver.get_path_selection()
                self.assertEqual(selected["routes"]["sweep"]["tiers"]["velocity_limit_pass"], last_word)
                self.assertEqual(selected["options"]["mf_gs_parallel_linked_velocity_limit_pass"]["applied"], last_word)
                state, _ = _rollout(model, solver, 2)
                _assert_tier_owned(self, solver, 32)
                for name in _STATE_FIELDS:
                    self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
                speed = float(np.max(np.abs(state.joint_qd.numpy()[:2])))
                if not limits:
                    self.assertGreater(speed, 2.0 * limit, "the push must carry an unlimited sphere past its limit")
                elif requested:
                    self.assertLessEqual(speed, limit * (1.0 + 1.0e-5))
        velocities = {}
        for (sweeps, requested), solver in budgets.items():
            keys = [kernel.key for kernel in solver._pgs_solve_mf_gs_incremental_kernels]
            self.assertTrue(
                keys and all(f"_sweeps{sweeps}_" in key and key.endswith("_lw") == requested for key in keys), keys
            )
            state, _ = _rollout(model, solver, 1)
            _assert_tier_owned(self, solver, 32)
            velocities[sweeps, requested] = state.joint_qd.numpy().astype(np.float64)
        speeds = {budget: float(np.max(np.abs(velocity[:2]))) for budget, velocity in velocities.items()}
        self.assertGreater(speeds[2, False], 1.05 * limit, f"two sweeps already hold the limit: {speeds}")
        for budget in ((2, True), (2000, False), (2000, True)):
            self.assertLessEqual(speeds[budget], limit * (1.0 + 1.0e-5), (budget, speeds))
        # A converged tier leaves the pass nothing to push.
        np.testing.assert_allclose(velocities[2000, True], velocities[2000, False], rtol=0.0, atol=1.0e-6)

    def test_linked_tiers_take_their_own_sweep_budget(self):
        """Build each linked tier with its budget, record it, and leave point tiers on mf_gs_parallel_sweeps."""
        model = standing_model()
        options = {"dense_max_constraints": 64, "pgs_iterations": 8}
        with patch.dict(os.environ, {_SWITCH: "1"}):
            built = {
                (beta, linked): _matrix_free_solver(
                    model, beta=beta, parallel_rows=64, mf_gs_parallel_linked_sweeps=linked, **options
                )
                for beta, linked in itertools.product((0.0, 0.2), (None, 40, "auto"))
            }
        for (beta, linked), solver in built.items():
            with self.subTest(beta=beta, linked=linked):
                if not beta or linked is None:
                    expected, rule = [24, 24], "mf_gs_parallel_sweeps"
                elif linked == "auto":
                    expected, rule = [24, 34], "auto"  # ceil(3 * 8 * sqrt(rows / 32)) over 24 for 64 rows
                else:
                    expected, rule = [40, 40], "mf_gs_parallel_linked_sweeps"
                keys = [kernel.key for kernel in solver._pgs_solve_mf_gs_incremental_kernels]
                self.assertEqual(len(keys), 2, keys)
                for key, sweeps in zip(keys, expected, strict=True):
                    self.assertIn(f"_sweeps{sweeps}_", key)
                    self.assertEqual(key.endswith("_linked_iso"), bool(beta))
                selected = solver.get_path_selection()
                tiers = selected["routes"]["sweep"]["tiers"]
                self.assertEqual([tier["sweeps"] for tier in tiers["tiers"]], expected)
                self.assertEqual((tiers["sweeps"], tiers["sweeps_rule"]), (max(expected), rule))
                applied = selected["options"]["mf_gs_parallel_linked_sweeps"]["applied"]
                self.assertEqual(applied, max(expected) if beta else 0)
                state, _ = _rollout(model, solver, 2)
                for name in _STATE_FIELDS:
                    self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)

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
            self.assertTrue(kernel.key.endswith("_linked_iso"))


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestPatchParallelPhysics(unittest.TestCase):
    """Compare each law's parallel tiers with its serial sweep, the patch gap bounded by the point gap.

    Parallel tiers are Jacobi sweeps, not float reorderings of the serial sweep.
    While every pair sticks the two schemes share their fixed point, and the
    rollouts differ by the convergence gap. Once a pair slides the point tiers do
    not: they scale the pair radially after per-row step scales, which turns its
    sliding direction and, through the coupling, moves the normals and the body
    velocity (on this module's reference case the converged velocities of the two
    schemes differ by 4-15 %). The linked tiers step each pair isotropically and
    keep the serial fixed point, so the point tiers' measured gap on the same
    fixture, with a floor at the float-order tolerance of an exact route change,
    bounds the patch tiers' gap. Every parallel step must leave each world in the
    tiers' domain, so the comparison is never serial against serial.
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

    def test_sliding_contact_reaches_the_serial_fixed_point(self):
        """Slide one contact of fourfold unequal tangent masses to the serial patch sweep's converged velocity.

        The linked tiers (tolerance 0, 2000 sweeps) and the serial sweep (2000
        iterations) converge from the same state. Each step's friction impulse,
        recovered from the velocity change through the diagonal joint-space mass,
        opposes the slip it leaves (cosine at least 1 - 1e-5), and the two routes'
        velocities agree to 1e-5 relative. Per-row steps turned the tiers' friction
        towards the heavier tangent (cosine 0.86).
        """
        axes, masses = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z), (3.0, 0.01, 1.0)
        tangent_mass = np.array((sum(masses), sum(masses[1:])))
        model = _sliding_sphere("cuda:0", axes, masses, (1.0, 1.0))
        converged = {"mf_gs_parallel_sweeps": 2000, "mf_gs_parallel_tol": 0.0}
        with patch.dict(os.environ, {_SWITCH: "1"}):
            routes = {
                "serial": _matrix_free_solver(model, beta=0.2, parallel_rows=0, pgs_iterations=2000),
                "tier": _matrix_free_solver(model, beta=0.2, parallel_rows=32, **converged),
                "tier_response_block": _matrix_free_solver(
                    model, beta=0.2, parallel_rows=32, mf_gs_parallel_matrix_free=False, **converged
                ),
            }
        velocities = {}
        for name, solver in routes.items():
            alignments, previous = [], model.joint_qd.numpy()[:2].astype(np.float64)

            def check(step, solver, contacts, state, name=name, alignments=alignments):
                nonlocal previous
                if name != "serial":
                    _assert_tier_owned(self, solver, 32)
                velocity = state.joint_qd.numpy()[:2].astype(np.float64)
                friction = tangent_mass * (velocity - previous)
                previous = velocity
                if np.linalg.norm(friction) > 1.0e-6:
                    alignments.append(
                        float(-friction @ velocity / (np.linalg.norm(friction) * np.linalg.norm(velocity)))
                    )

            state, _ = _rollout(model, solver, 4, after_step=check)
            with self.subTest(route=name):
                self.assertTrue(alignments, "the contact never carried friction")
                self.assertGreaterEqual(min(alignments), 1.0 - 1.0e-5, alignments)
            velocities[name] = state.joint_qd.numpy().astype(np.float64)
        scale = float(np.max(np.abs(velocities["serial"])))
        for name in ("tier", "tier_response_block"):
            np.testing.assert_allclose(velocities[name], velocities["serial"], atol=1.0e-5 * scale, err_msg=name)

    def test_linked_budget_carries_the_tiers_to_the_serial_fixed_point(self):
        """Converge the linked tiers on their own budget while the shared budget stays truncated.

        On the first step of :func:`sliding_stance` a linked tier that
        ``mf_gs_parallel_sweeps=2`` (tolerance 0) stops must stay short of the
        serial sweep's velocity (2000 iterations), and the linked budget of 2000
        sweeps alone must carry it there (:func:`assert_budget_reaches_the_fixed_point`;
        the float64 reference on the CPU split rows: 3.4e-2 of the scale after two
        sweeps, 1.1e-7 after 2000). A single sliding contact cannot show what a budget
        carries: two sweeps converge it (4.2e-6 of the scale in the reference). One
        step keeps the comparison on the solve, as later steps add the float32 drift
        of the feet's poses that the position correction amplifies. Point tiers built
        with the same request keep two sweeps.
        """
        model = sliding_stance("cuda:0")
        truncated = {"mf_gs_parallel_sweeps": 2, "mf_gs_parallel_tol": 0.0}
        with patch.dict(os.environ, {_SWITCH: "1"}):
            routes = {
                "serial": _matrix_free_solver(model, beta=0.2, parallel_rows=0, pgs_iterations=2000),
                "truncated": _matrix_free_solver(model, beta=0.2, parallel_rows=32, **truncated),
                "linked_budget": _matrix_free_solver(
                    model, beta=0.2, parallel_rows=32, mf_gs_parallel_linked_sweeps=2000, **truncated
                ),
            }
            point = _matrix_free_solver(
                model, beta=0.0, parallel_rows=32, mf_gs_parallel_linked_sweeps=2000, **truncated
            )
        point_keys = [kernel.key for kernel in point._pgs_solve_mf_gs_incremental_kernels]
        self.assertTrue(
            point_keys and all("_sweeps2_" in key and "_linked" not in key for key in point_keys), point_keys
        )
        for name in ("truncated", "linked_budget"):
            sweeps = 2000 if name == "linked_budget" else 2
            kernels = routes[name]._pgs_solve_mf_gs_incremental_kernels
            self.assertTrue(kernels and all(f"_sweeps{sweeps}_" in kernel.key for kernel in kernels), name)
        velocities = {}
        for name, solver in routes.items():
            state, _ = _rollout(model, solver, 1)
            if name != "serial":
                _assert_tier_owned(self, solver, 32)
            velocities[name] = state.joint_qd.numpy().astype(np.float64)
        gaps = {
            name: float(np.max(np.abs(velocities[name] - velocities["serial"])))
            for name in ("truncated", "linked_budget")
        }
        scale = float(np.max(np.abs(velocities["serial"])))
        assert_budget_reaches_the_fixed_point(self, gaps["truncated"], gaps["linked_budget"], scale)

    def test_lone_live_tangent_keeps_its_friction(self):
        """Brake a sphere whose second tangent no joint moves as the serial sweep brakes it.

        The isotropic step leaves a pair alone when either step is zero: the zero
        Jacobian of the frozen tangent must not stop the live one.
        """
        model = _sliding_sphere("cuda:0", (newton.Axis.X, newton.Axis.Z), (3.0, 1.0), (1.0,))
        converged = {"mf_gs_parallel_sweeps": 2000, "mf_gs_parallel_tol": 0.0}
        with patch.dict(os.environ, {_SWITCH: "1"}):
            serial = _matrix_free_solver(model, beta=0.2, parallel_rows=0, pgs_iterations=2000)
            tier = _matrix_free_solver(model, beta=0.2, parallel_rows=32, **converged)
        speeds = {}
        for name, solver in (("serial", serial), ("tier", tier)):
            state, _ = _rollout(model, solver, 6)
            speeds[name] = float(state.joint_qd.numpy()[0])
        _assert_tier_owned(self, tier, 32)
        # Six steps of mu * g at the sphere's weight over the four-fold chain mass brake it by about 0.03 m/s.
        self.assertLess(speeds["serial"], 1.0 - 0.01)
        self.assertAlmostEqual(speeds["tier"], speeds["serial"], delta=1.0e-5)


if __name__ == "__main__":
    unittest.main()

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
CPU solver builds, against that solver's serial solution. Keyed rows write the
patch row law themselves; the CPU fallback checks it against the general
producer and the patch link.
"""

import inspect
import itertools
import math
import os
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group, keyed_rows, path_selection, sweep_routes_tiers
from newton._src.solvers.feather_pgs import sweep as emitter
from newton._src.solvers.feather_pgs.device_preludes import RING
from newton._src.solvers.feather_pgs.factor.kernels import prescale_joint_velocity_limits
from newton._src.solvers.feather_pgs.factor.walk import WALK_LINK, WalkTrees, pack_walk_records, release_view
from newton._src.solvers.feather_pgs.joint_table import JointTable
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout
from newton._src.solvers.feather_pgs.placement import Placement
from newton._src.solvers.feather_pgs.route_table import RouteTable
from newton._src.solvers.feather_pgs.row_kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton._src.solvers.feather_pgs.row_records import DenseRows
from newton._src.solvers.feather_pgs.rows import RowLaws, velocity_prescale_launch
from newton._src.solvers.feather_pgs.sweep_plan import Sweep
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import (
    bias_rows,
    keyed_classes,
    leave_walk,
    plain_laws,
    row_bias,
    row_lines,
    unpacked_rows,
)
from newton.tests.feather_pgs_mass_reference import MassReference
from newton.tests.feather_pgs_patch_parity import (
    SLIDING_FEET,
    assert_loaded_anchors,
    assert_ring_coulomb,
    standing_model,
)
from newton.tests.feather_pgs_row_builders import (
    assert_row_builders_named,
    install_cpu_keyed_rows,
    launched_kernels,
)
from newton.tests.test_feather_pgs_native_prelude import _native_snippets
from newton.tests.test_feather_pgs_velocity_limit_law import _HAND_LIMIT, _limited_hand


def _source(tier):
    """A parallel tier's native sweep source (its kernel's longest native snippet)."""
    return max(_native_snippets(tier.kernel), key=len)


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


def _law_key(key):
    """A tier kernel's name without its spin-row suffix: patch friction's tiers read spin rows (``_torsion``)."""
    return key.removesuffix("_torsion")


def _spin(kinds, parents, parent):
    """The spin row of the contact whose normal is ``parent``: the friction-kind row of the same normal after its pair,
    or -1 (a contact's rows read from the data, as the sweeps read them)."""
    row = parent + 3
    return row if row < len(kinds) and kinds[row] == 1 and parents[row] == parent else -1


def _ball(triple, mu, mu_t, load):
    """Project a friction triple onto its cone ``|(t1 / mu, t2 / mu, s / mu_t)| <= load``; without spin (``mu_t = 0``)
    the pair onto its disk (``ConeLaw::ball``)."""
    triple = np.asarray(triple, dtype=np.float64)
    if mu_t <= 0.0:
        radius = mu * load
        magnitude = math.hypot(triple[0], triple[1])
        pair = triple[:2] * (radius / magnitude) if magnitude > radius else triple[:2]
        return np.array([pair[0], pair[1], 0.0])
    scale = np.array([1.0 / mu if mu > 0.0 else 0.0, 1.0 / mu if mu > 0.0 else 0.0, 1.0 / mu_t])
    magnitude = float(np.linalg.norm(triple * scale))
    pinned = np.array([1.0 if mu > 0.0 else 0.0, 1.0 if mu > 0.0 else 0.0, 1.0])
    return triple * pinned * (load / magnitude if magnitude > load else 1.0)


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
    pair_step="isotropic",
    velocity_limit_pass=False,
):
    """Repeat one world of the parallel tier's sweep in float64.

    Returns the impulses and the generalized velocity the kernel publishes. The
    response block is ``A = J H^-1 J^T``. The step scale of row ``i`` is
    ``min(A_ii / bound_i, 1) * omega / diag_i``: the bound is the row sum of
    ``|A|`` over admitted rows (A-based tiers and exact row sums) or the
    row-sum bound ``sum_d |P_id| sum_j |Q_jd|`` of the matrix-free tiers, with
    ``(P, Q) = (J, Y)`` or the whitened rows ``(Z, Z)`` of the in-kernel
    response. A friction row is admitted whatever its diagonal. With the isotropic
    ``pair_step``, every tier's law, both rows of a friction pair take the smallest
    of their positive steps (a tangent that moves nothing takes its sibling's),
    unless the sibling is not an admitted friction row of the same normal;
    ``"per_row"`` keeps each row's own step, the retired law that turns a sliding
    pair's friction. The bound counts the rows that can move: a unilateral row at zero
    impulse whose residual at zero impulse is non-negative is idle and out of it, until
    its residual at the extrapolated impulses turns negative; it then keeps its impulse for
    that sweep and joins after it: every step is set again, the extrapolation dropped and
    the momentum restarted, and it moves from the next sweep. Every sweep steps all rows from the extrapolated
    impulses, clamps normal and limit rows, scales each tangent pair, and its contact's spin
    row where it has one, into their cone (the ring's pooled normal load when ``linked``;
    a spin row steps isotropically with its pair in the cone's scaled coordinates, its step
    the pair's times ``(mu_t / mu)^2``), restarts momentum against the last step
    and stops once no row changes. ``velocity_limit_pass`` then pushes every violated
    velocity-limit row onto its limit, in row order.
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
    kinds = np.where((kinds >= 0) & ((diag > 0.0) | (kinds == 1)), kinds, -1)
    admitted = kinds >= 0
    b = case["rhs"] + J @ v - A @ lam0
    idle = (kinds == 0) & (lam0 == 0.0) & (b >= 0.0)

    def steps(counted):
        if matrix_free and not exact:
            bound = np.abs(P) @ np.abs(Q[counted]).sum(axis=0)
        else:
            bound = np.abs(A[:, counted]).sum(axis=1)
        weight = np.where(bound > 0.0, np.minimum(np.diag(A) / np.where(bound > 0.0, bound, 1.0), 1.0), 0.0)
        step = np.where(admitted, weight * omega / np.where(diag > 0.0, diag, 1.0), 0.0)
        if pair_step == "isotropic":
            for row in np.flatnonzero(kinds == 1):
                parent, sibling = parents[row], row + 1
                if row != parent + 1 or sibling >= len(kinds) or kinds[sibling] != 1 or parents[sibling] != parent:
                    continue
                positive = [value for value in (step[row], step[sibling]) if value > 0.0]
                block = min(positive) if positive else 0.0
                spin = _spin(kinds, parents, parent)
                if spin >= 0 and mu[row] > 0.0:
                    spin_scale = (mu[spin] / mu[row]) ** 2
                    if spin_scale > 0.0 and step[spin] > 0.0:
                        spin_block = step[spin] / spin_scale
                        block = min(block, spin_block) if block > 0.0 else spin_block
                    if block > 0.0 and spin_scale > 0.0:
                        step[spin] = block * spin_scale
                if block > 0.0:
                    step[row] = step[sibling] = block
        return step

    step = steps(admitted & ~idle)
    x, y, t = lam0.copy(), lam0.copy(), 1.0
    for _ in range(sweeps):
        residual = b + A @ y
        joins = idle & (residual < 0.0)
        stepped = x.copy()
        for row in np.flatnonzero(admitted & ~joins):
            value = y[row] - step[row] * residual[row]
            if kinds[row] == 0:
                value = max(value, 0.0)
            stepped[row] = value
        projected = stepped.copy()
        for row in np.flatnonzero(kinds == 1):
            parent = parents[row]
            spin = _spin(kinds, parents, parent)
            load = max(ring_load(parents, stepped, parent, linked=linked), 0.0)
            triple = (stepped[parent + 1], stepped[parent + 2], stepped[spin] if spin >= 0 else 0.0)
            projected[row] = _ball(triple, mu[parent + 1], mu[spin] if spin >= 0 else 0.0, load)[row - parent - 1]
        changed = joins.any() or np.any(np.abs(projected - x) > tol * (np.abs(projected) + tol))
        if np.dot(y - projected, projected - x) > 0.0:
            t = 1.0
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_next if nesterov else 0.0
        t = t_next
        x, y = projected, projected + beta * (projected - x)
        if not changed:
            break
        if joins.any():
            idle &= ~joins
            step = steps(admitted & ~idle)
            t, y = 1.0, x.copy()
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
    L = np.linalg.cholesky(H[::-1, ::-1])  # leaf-first: P H P^T = L L^T
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
        "Z": np.linalg.solve(L, J[:, ::-1].T).T[:, ::-1],
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
    L = np.linalg.cholesky(H[::-1, ::-1])  # leaf-first: P H P^T = L L^T
    J = np.zeros((3, _DOFS))
    J[0, 2] = J[1, 0] = 1.0
    if live_tangents == 2:
        J[2, 1] = 1.0
    Y = np.linalg.solve(H, J.T).T
    cfm = 1.0e-5
    return {
        "J": J,
        "Y": Y,
        "Z": np.linalg.solve(L, J[:, ::-1].T).T[:, ::-1],
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
    L = np.linalg.cholesky(H[::-1, ::-1])  # leaf-first: P H P^T = L L^T
    J = np.zeros((2, _DOFS))
    J[0, 2], J[0, 3] = 1.0, 0.6
    J[1, 3] = -1.0  # upper limit: limit - qd_3 >= 0
    Y = np.linalg.solve(H, J.T).T
    cfm = 1.0e-5
    return {
        "J": J,
        "Y": Y,
        "Z": np.linalg.solve(L, J[:, ::-1].T).T[:, ::-1],
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


def world_case(solver, world=0):
    """One world's rows of a CPU solver's last step as a reference case, and the velocity of its own solve.

    The rows are the matrix-free sweep's: each row's Jacobian ``J`` and response
    ``Y = H^-1 J^T`` over the world's coordinates, from the packed rows
    (:func:`.feather_pgs_law_harness.unpacked_rows`), and the bias in the
    right-hand side; the case takes ``v_hat`` as its velocity and cold impulses.
    Contact and limit rows clamp at zero and friction rows pair; a world with any
    other row, such as a drive, mimic or connect row, raises ``ValueError``.
    Returns the case and ``v_hat + Y^T lambda`` of the impulses the sweep solved.
    """
    count = int(solver.placement.constraint_count.numpy()[world])
    types = solver.placement.row_type.numpy()[world, :count]
    unmodelled = sorted(
        set(types.tolist())
        - {
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
            PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
            PGS_CONSTRAINT_TYPE_TORSION,
        }
    )
    if unmodelled:
        raise ValueError(f"world_case models contact, friction, spin and limit rows only, not row types {unmodelled}")
    jacobian, response = unpacked_rows(solver)
    J, Y = jacobian[world, :count], response[world, :count]
    forest = solver.forest
    starts, offsets = forest.articulation_dof_start.numpy(), forest.articulation_world_dof_offset.numpy()
    predicted, velocity = solver.v_hat.numpy().astype(np.float64), np.zeros(J.shape[1])
    for art, n in enumerate(forest.articulation_response_dof_count.numpy()):
        if forest.art_to_world.numpy()[art] == world:
            velocity[offsets[art] : offsets[art] + n] = predicted[starts[art] : starts[art] + n]
    # The rows' law's regularization, one value (pgs_cfm).
    cfm = np.full(count, solver.options.pgs_cfm, dtype=np.float32).astype(np.float64)
    case = {
        "J": J,
        "Y": Y,
        "Z": None,
        "L": None,
        "cfm": cfm,
        "diag": np.einsum("ij,ij->i", J, Y) + cfm,
        "kinds": np.where(np.isin(types, (PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_TORSION)), 1, 0).astype(
            np.int32
        ),
        "types": types,
        "parents": solver.placement.row_parent.numpy()[world, :count].astype(np.int32),
        "mu": solver.placement.row_mu.numpy()[world, :count].astype(np.float64),
        "rhs": solver.placement.rhs.numpy()[world, :count].astype(np.float64),
        "impulses": np.zeros(count),
        "velocity": velocity,
    }
    return case, velocity + Y.T @ solver.placement.impulses.numpy()[world, :count].astype(np.float64)


def prescaled_joint_velocity(model):
    """The model's joint velocity after the pre-solve prescale of a step with joint velocity limits.

    With ``enable_joint_velocity_limits`` the solver scales each articulation's
    velocity by one ratio, the smallest ``limit / |qd|`` over its DOFs (PhysX's
    ``maxJointVelocity`` ratio), and predicts the step's velocity from the
    result. A solver without them, which builds this module's CPU rows, predicts
    from the unscaled velocity. The launch takes the
    solver.s lanes and thread count (``rows.velocity_prescale_launch``).
    """
    lanes, threads = velocity_prescale_launch(
        model.articulation_start.numpy(), joint_count=model.joint_count, cuda=model.device.is_cuda
    )
    qd = wp.clone(model.joint_qd)
    wp.launch(
        prescale_joint_velocity_limits,
        dim=threads,
        inputs=[
            field_group.upload(JointTable, model.device, model),
            model.articulation_count,
            lanes,
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


def _tier_texts(rows: int):
    """The emitter's text of a ``rows``-row tier's pair projections, point and pooled, and of its isotropic step."""
    store = emitter.JacobiRows(rows, 32 * ((rows + 31) // 32))
    point = emitter._jacobi_projection(store, pooled=False, pad="            ")
    pooled = emitter._jacobi_projection(store, pooled=True, pad="            ")
    return point, pooled, emitter.isotropic_friction_step(store, "    ")


class TestParallelPatchReference(unittest.TestCase):
    def test_linked_factory_pools_only_the_friction_disk(self):
        """Build linked tiers as the point tiers with the golden ring walk; both take the isotropic pair step."""
        for rows, inkernel in itertools.product((32, 64), ((6, 0, 0, 0), (_DOFS, 0, 0, 0))):
            options = {"rows": rows, "min_rows": 0, "sweeps": 24, "inkernel_response": inkernel}
            with self.subTest(**options):
                point = sweep_routes_tiers.parallel_kernel(_CAPACITY, 32, _DOFS, "120", **options)
                linked = sweep_routes_tiers.parallel_kernel(_CAPACITY, 32, _DOFS, "120", linked_rows=True, **options)
                self.assertFalse(point.kernel.key.endswith("_linked"), point.kernel.key)
                self.assertEqual(linked.kernel.key, point.kernel.key + "_linked")
                point_source = _source(point)
                linked_source = _source(linked)
                projection, pooled, isotropic = _tier_texts(rows)
                for text, golden in (
                    (point_source, projection),
                    (point_source, isotropic),
                    (linked_source, pooled),
                    (linked_source, isotropic),
                ):
                    self.assertEqual(text.count(golden), 1, golden)
                # Nothing but the ring walk's prelude and the projection differs from the point tier.
                self.assertEqual(linked_source.count(RING.source), 1)
                restored = linked_source.replace(RING.source + "\n", "").replace(pooled, projection)
                self.assertEqual(restored, point_source)
                self.assertEqual(linked.work, point.work)

    def test_reference_holds_pooled_cones_and_restores_point_rings(self):
        """Keep each pair inside its pooled ring disk, and reduce to point friction without ring links."""
        for seed, faces in ((0, 1), (1, 3)):
            linked_case = parallel_case(seed, faces=faces)
            point_case = parallel_case(seed, faces=faces, linked=False)
            for inkernel in (False, True):
                with self.subTest(seed=seed, faces=faces, inkernel=inkernel):
                    options = {
                        "matrix_free": True,
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
        for inkernel in (False, True):
            with self.subTest(inkernel=inkernel):
                options = {
                    "matrix_free": True,
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

    def test_every_tier_steps_pairs_isotropically(self):
        """Give every tier variant, point and linked, the isotropic pair step once; no per-row variant remains."""
        layout = KeyedLayout(16, 6, 0, 16, 128, False, 0)
        keyed = keyed_rows.get_parallel_factory(sweep_routes_tiers.parallel_kernel, layout)
        for factory, shape, extra in (
            (
                sweep_routes_tiers.parallel_kernel,
                (_CAPACITY, 32, _DOFS, "120"),
                {"inkernel_response": (_DOFS, 0, 0, 0)},
            ),
            (keyed, (128, 32, 22, "120"), {"inkernel_response": (16, 6, 0, 16), "classes": keyed_classes(layout)}),
        ):
            for linked in (False, True):
                with self.subTest(shape=shape, linked=linked, **extra):
                    tier = factory(*shape, rows=32, sweeps=24, linked_rows=linked, **extra)
                    self.assertEqual(tier.kernel.key.endswith("_linked"), linked, tier.kernel.key)
                    self.assertEqual(_source(tier).count(_tier_texts(32)[2]), 1)
        self.assertNotIn("pair_step", inspect.signature(sweep_routes_tiers.parallel_kernel).parameters)

    def test_isotropic_pairs_slide_with_maximal_dissipation(self):
        """Slide a pair with fourfold unequal tangent masses against its velocity, as the serial pair solve does.

        Per-row steps turn the converged friction towards the heavier tangent (cosine 0.86
        against the slip); the isotropic step aligns it with the slip.
        """
        case = sliding_pair_case()
        for inkernel in (False, True):
            options = {"matrix_free": True, "inkernel": inkernel, "exact": False, "nesterov": True, "tol": 0.0}
            with self.subTest(inkernel=inkernel):
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
        for inkernel in (False, True):
            options = {"matrix_free": True, "inkernel": inkernel, "exact": False, "nesterov": True, "tol": 0.0}
            with self.subTest(inkernel=inkernel):
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
        plain_tiers = (sweep_routes_tiers.parallel_kernel, (_CAPACITY, 32, _DOFS, "120"))
        layout = KeyedLayout(16, 6, 0, 16, 128, True, (1 << 16) - 1)
        keyed_tiers = (
            keyed_rows.get_parallel_factory(sweep_routes_tiers.parallel_kernel, layout),
            (128, 32, 22, "120"),
        )
        variants = {
            "group rows": (plain_tiers, {"inkernel_response": (6, 0, 0, 0)}),
            "keyed rows": (keyed_tiers, {"inkernel_response": (16, 6, 0, 16), "classes": keyed_classes(layout)}),
        }
        for name, ((factory, args), options) in variants.items():
            with self.subTest(variant=name):
                settings = {"rows": 32, "sweeps": 24, "linked_rows": True, **options}
                plain = factory(*args, **settings)
                last_word = factory(*args, velocity_limit_pass=True, **settings)
                self.assertEqual(last_word.kernel.key, plain.kernel.key + "_lw")
                self.assertEqual((plain.velocity_limit_pass, last_word.velocity_limit_pass), (False, True))
                text = _source(last_word)
                self.assertNotIn("s_vlim", _source(plain))
                self.assertEqual(text.count("s_vlim[i] = row_type == 4 ? 1 : 0;"), 1)
                self.assertEqual(text.count("const float r_vlim = s_rhs[i] + part;"), 1)
                # The pass runs between the final sweep and the velocity change's publish to the step's leave.
                self.assertLess(text.index("const float r_vlim"), text.index("walk.du.data[global_dof] +="))
                self.assertGreater(text.index("const float r_vlim"), text.index("for (; sweep <"))

    def test_solver_builds_the_pass_for_limited_linked_tiers(self):
        """Build the last-word pass exactly for linked tiers with joint velocity limits."""
        decide = Sweep._parallel_velocity_limit_pass
        for limits in (False, True):
            probe = SimpleNamespace(options=SimpleNamespace(enable_joint_velocity_limits=limits))
            with self.subTest(limits=limits):
                self.assertEqual(decide(probe), limits)

    def test_last_word_holds_a_truncated_limit(self):
        """Hold the limit after two sweeps with the last word, and leave a converged tier unchanged."""
        case = limited_contact_case()
        for inkernel in (False, True):
            options = {"matrix_free": True, "inkernel": inkernel, "exact": False, "nesterov": True, "tol": 0.0}
            with self.subTest(inkernel=inkernel):
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

        The rows are the first step of :func:`_pushed_sphere` as a CPU solver
        without velocity limits builds them, started from the prescaled velocity
        the route with them predicts from, with every DOF's velocity-limit rows added
        as the matrix-free builder writes them. Two sweeps of the matrix-free
        linked tier (the float64 reference) leave the sphere 43 % past its limit;
        the pass holds it to 1e-5 (1.7e-6 past: the regularized diagonal's share).
        A sphere that starts past its limit cannot show this: the prescale puts it
        on the limit before the solve, and two sweeps leave it 0.9 % under.
        TestPatchParallelRoutes runs the solver's tiers on this sphere.
        """
        model = _pushed_sphere("cpu")
        limits = model.joint_velocity_limit.numpy().astype(np.float64)
        model.joint_qd.assign(prescaled_joint_velocity(model))  # what the matrix-free route predicts from
        with patch.dict(SolverFeatherPGS._route_pins, _DENSE):
            solver = SolverFeatherPGS(model, friction_anchor_beta=0.2)
        _rollout(model, solver, 1)
        mass_matrix = MassReference(model).mass_matrix(model.joint_q.numpy(), 0)
        case = with_velocity_limit_rows(world_case(solver)[0], mass_matrix, limits, solver.options.pgs_cfm)
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
    """Step until the contact rows are built and linked, and return host copies of the rows: the row pass's records,
    with those the one writer writes after it where the placement walks, once the patches link them."""
    rows = {}
    finalize = Placement.latch_status
    link_patches = RowLaws.link_patches

    def capture(self, step_contacts):
        finalize(self, step_contacts)
        if self is solver.placement:
            rows.update({name: getattr(self, name).numpy().copy() for name in _ROW_ARRAYS})
            if not self.walks:
                raise _RowsBuilt

    def linked(self, step_solver, *args):
        link_patches(self, step_solver, *args)
        if step_solver is solver and step_solver.placement.walks:
            rows.update({name: getattr(solver.placement, name).numpy().copy() for name in _ROW_ARRAYS})
            raise _RowsBuilt

    with patch.object(Placement, "latch_status", capture), patch.object(RowLaws, "link_patches", linked):
        try:
            solver.step(state, solver.model.state(), solver.model.control(), contacts, dt)
        except _RowsBuilt:
            pass
    if not rows:
        raise AssertionError("The step did not build contact rows")
    return rows


def _walked_jacobian(solver, joint_q, rows):
    """The Jacobian rows of the general placement's walked rows (the one writer forms no ``J``), at their packed
    entries: ``J^T = P^T L P z`` of each side by the float64 walk at the step's coordinates (FL3.3)."""
    placement, forest = solver.placement, solver.forest
    Z = placement.packed_Z.numpy().astype(np.float64)
    J = np.zeros_like(Z)
    sides = placement.row_sides.numpy()
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    factors = {}
    for world, count in enumerate(rows["constraint_count"]):
        line = int(placement.dense_first.numpy()[world])
        for row in range(int(count)):
            first = 0
            for art in sides[line, row]:
                if art < 0:
                    continue
                n = int(sizes[art])
                at = int(offsets[art]) if placement.packed_dense else first
                first += n
                if art not in factors:
                    inverse, _ = solver.factor.walk(joint_q, int(art)).up(covector=np.eye(n))
                    factors[art] = np.linalg.inv(inverse)
                J[line, row, at : at + n] = factors[art] @ Z[line, row, at : at + n]
    return J


def _general_bias(solver, rows, dt):
    """The general route's contact and friction bias for captured rows (without ``J v``)."""
    device = solver.model.device
    count = wp.array(rows["constraint_count"], dtype=wp.int32, device=device)
    rhs = wp.zeros(rows["rhs"].shape, dtype=wp.float32, device=device)
    records = field_group.upload(
        DenseRows,
        device,
        phi=rows["phi"],
        row_beta=rows["row_beta"],
        row_type=rows["row_type"],
        target=rows["target_velocity"],
        restitution=wp.zeros_like(rhs),
        rhs=rhs,
        row_w=wp.zeros_like(rhs),
        first=row_lines(device),
    )
    wp.launch(
        bias_rows,
        dim=(rhs.shape[0], rhs.shape[1]),
        inputs=[
            records,
            wp.zeros_like(rhs),
            count,
            rhs.shape[1],
            rhs.shape[1],
            row_bias(dt, solver.options.contact_speculative_scale, 1.0, 0.0, 0, 0),
        ],
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
        keyed.laws.friction_patches = general.laws.friction_patches
        install_cpu_keyed_rows(keyed)
        if late_beta is not None:
            general.laws.friction_anchor_beta = keyed.laws.friction_anchor_beta = late_beta
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
        joint_q = state.joint_q.numpy().astype(np.float64)
        size = keyed.placement.keyed_rows.layout.primary
        # The keyed rows first: a whole step's finish moves the shared history (each region's anchor is the centre of
        # pressure its solve leaves), and both writers must build from the same history.
        actual = _linked_rows(keyed, state, contacts, dt)
        # The keyed class's general worlds are the one writer's walked rows too (keyed_rows.keyed_general_units).
        actual["J"] = {size: _walked_jacobian(keyed, joint_q, actual)}
        expected = _linked_rows(general, state, contacts, dt)
        # The general placement's one writer walks its rows after the row pass (FL3.3): its J, from the step's
        # walked rows, is the reference of the keyed rows' J.
        general.step(state, out, model.control(), contacts, dt)
        expected["J"] = {size: _walked_jacobian(general, joint_q, expected)}
        live = int(contacts.rigid_contact_count.numpy()[0])
        owner = _contact_rows_by_id(expected, live)
        self.assertEqual(owner, _contact_rows_by_id(actual, live))
        slot_contact = {(world, first): c for c, (world, first, _) in owner.items()}
        bias = _general_bias(general, expected, dt)
        linked = tangent_phi = 0
        for c, (world, first, needed) in owner.items():
            rows = slice(first, first + needed)
            with self.subTest(contact=c):
                for name in ("row_type", "row_beta", "row_restitution"):
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
                    actual["J"][size][world, rows], expected["J"][size][world, rows], rtol=1.0e-4, atol=1.0e-5
                )
                np.testing.assert_allclose(actual["rhs"][world, rows], bias[world, rows], rtol=1.0e-5, atol=1.0e-7)
                linked += expected["linked"][c][0] not in (-1, c)
                tangent_phi += needed >= 3 and bool(np.any(expected["phi"][world, first + 1 : first + 3] != 0.0))
                if needed >= 3:
                    # The anchors carry the coefficient in force at this step, on their spin rows too.
                    beta = options["friction_anchor_beta"] if late_beta is None else late_beta
                    np.testing.assert_array_equal(
                        expected["row_beta"][world, first + 1 : first + needed], np.float32(beta)
                    )
        self.assertGreater(linked, 0, "A box foot must link several normals in one ring")
        self.assertGreater(tangent_phi, 0, "Anchors must carry tangential phi")

    def test_keyed_factory_reads_linked_parents(self):
        """Read every row's parent and coefficient in the keyed patch tiers instead of building them."""
        layout = KeyedLayout(16, 6, 0, 16, 128, False, 0)
        factory = keyed_rows.get_parallel_factory(sweep_routes_tiers.parallel_kernel, layout)
        options = {"rows": 32, "sweeps": 24, "inkernel_response": (16, 6, 0, 16), "classes": keyed_classes(layout)}
        linked = factory(128, 32, 22, "120", linked_rows=True, **options)
        self.assertTrue(linked.kernel.key.startswith("sweep_keyed_parallel_"))
        self.assertTrue(linked.kernel.key.endswith("_linked"))
        self.assertIn("patch_law", inspect.signature(linked.kernel.func).parameters)
        text = _source(linked)
        self.assertIn("s_parent[i] = world_row_parent.data[off_dense + i];", text)
        self.assertIn("s_mu[i] = world_row_mu.data[off_dense + i];", text)
        for written in ("c.row_parent.data", "c.row_mu.data", "c.row_type.data", "built_parent", "built_mu"):
            self.assertNotIn(written, text)
        self.assertIn("patch_law.patches.point.data[contact]", text)
        self.assertIn("patch_law.patches.phi.data[contact][r - 1]", text)
        for golden in _tier_texts(32)[1:]:
            self.assertEqual(text.count(golden), 1, golden)


def _cpu_arm():
    """A one-joint arm: enough for construction-time admission on CPU."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    return builder.finalize(device="cpu")


class TestPatchParallelAdmission(unittest.TestCase):
    def test_both_laws_are_admitted_and_the_record_says_why(self):
        """Admit patch friction to the parallel rows, record why, and admit contact torsion (the tiers solve
        the friction triple) and weighted contact rows."""
        model = _cpu_arm()
        for beta in (0.2,):
            with self.subTest(beta=beta):
                solver = SolverFeatherPGS(model, friction_anchor_beta=beta)
                selected = solver.get_path_selection()
                friction = selected["routes"]["friction"]
                keyed = selected["routes"]["contact_rows"]["keyed_rows"]
                self.assertNotIn("keyed rows", friction["reason"])
                self.assertNotIn("need point friction", friction["reason"])
                self.assertNotIn("point contacts", keyed["reason"])

    def test_route_record_reads_the_built_tiers(self):
        """Record the tiers' ring-pooled projection and sweep budgets from the kernels built, not the law."""
        for linked in (False, True):
            options = {"inkernel_response": (_DOFS, 0, 0, 0), "linked_rows": linked}
            kernels = [
                sweep_routes_tiers.parallel_kernel(
                    _CAPACITY, 32, _DOFS, "120", rows=rows, min_rows=rows - 32, sweeps=sweeps, **options
                )
                for rows, sweeps in ((32, 24), (64, 34))
            ]
            probe = SimpleNamespace(
                sweep=SimpleNamespace(
                    tiers=kernels,
                    par_tiers=[(32, 0), (64, 32)],
                    par_sweeps_rule="auto",
                    par_sweeps_reason="auto: ...",
                ),
                placement=SimpleNamespace(ink_rows=64, ink_sizes=(_DOFS, 0, 0, 0)),
                spec=SimpleNamespace(
                    route=SimpleNamespace(
                        producer="walk",
                        reasons={"K4": "The parallel tiers form the response of sizes [12] in-kernel."},
                    )
                ),
            )
            with self.subTest(linked=linked):
                tiers = path_selection._tiers(probe)
                self.assertEqual(tiers["linked_rows"], linked)
                self.assertEqual(_law_key(tiers["tiers"][0]["kernel"]).endswith("_linked"), linked)
                self.assertEqual("_linked" in tiers["tiers"][0]["kernel"], linked)
                self.assertNotIn("pair_step", tiers)
                self.assertEqual([tier["sweeps"] for tier in tiers["tiers"]], [24, 34])
                self.assertEqual((tiers["sweeps"], tiers["sweeps_rule"]), (34, "auto"))


class TestTierSweeps(unittest.TestCase):
    """Budget every parallel tier by one rule, the equal-accuracy budget, under either law."""

    def test_auto_budget_grows_with_iterations_and_tier_rows(self):
        """Sweep ceil(7/2 * pgs_iterations * sqrt(rows / 32)) times within every tier's bounds.

        The budget is at least one sweep and at most 128. The factor is fit per row bucket (see
        _TIER_SWEEPS_PER_ITERATION): at 12 iterations the four tiers sweep 42, 60, 73 and 84
        times, at 8 iterations the 32- and 48-row tiers 28 and 35 times, under either law.
        """
        self.assertEqual(sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION, Fraction(7, 2))
        self.assertEqual(sweep_routes_tiers.TIER_SWEEPS_CAP, 128)
        for rows, iterations, expected in (
            (32, 12, 42),  # 7/2 * 12 exactly: no rounding up past a whole budget
            (64, 12, 60),
            (96, 12, 73),
            (128, 12, 84),  # 7/2 * 12 * 2 exactly
            (32, 8, 28),  # 7/2 * 8 exactly
            (48, 8, 35),
            (64, 8, 40),
            (32, 6, 21),
            (32, 1, 4),  # 3.5 rounds up
            (32, 32, 112),
            (32, 64, 128),  # the cap
            (64, 0, 1),  # at least one sweep
        ):
            with self.subTest(rows=rows, iterations=iterations):
                self.assertEqual(sweep_routes_tiers.tier_sweeps(rows, iterations), expected)
        # The integer arithmetic is the exact ceiling: the least n with 32 (q n)^2 >= (p iterations)^2 rows.
        p, q = (
            sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION.numerator,
            sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION.denominator,
        )
        wrong = []
        for rows, iterations in itertools.product(range(1, 129), range(41)):
            exact = 0
            while 32 * (q * exact) ** 2 < (p * iterations) ** 2 * rows:
                exact += 1
            if sweep_routes_tiers.tier_sweeps(rows, iterations) != min(
                max(exact, 1), sweep_routes_tiers.TIER_SWEEPS_CAP
            ):
                wrong.append((rows, iterations))
        self.assertEqual(wrong, [])

    def test_solver_budgets_every_tier(self):
        """Budget point and linked tiers by one rule."""
        tiers = [(32, 0), (64, 32), (96, 64), (128, 96)]
        for _ in range(1):
            probe = SimpleNamespace(pins={}, options=SimpleNamespace(pgs_iterations=12))
            with self.subTest():
                budgets, rule, reason = Sweep._parallel_tier_sweeps(probe, tiers)
                self.assertEqual((budgets, rule), ([42, 60, 73, 84], "auto"))
                self.assertIn("ceil(7/2 * pgs_iterations * sqrt(R / 32))", reason)
                self.assertIn("pgs_iterations=12", reason)
                self.assertIn("at most 128", reason)
        # A test pin sets every tier's budget.
        probe = SimpleNamespace(
            pins={"tier_sweeps": 160},
            options=SimpleNamespace(pgs_iterations=12),
        )
        budgets, rule, reason = Sweep._parallel_tier_sweeps(probe, tiers)
        self.assertEqual((budgets, rule), ([160] * 4, "pinned"))
        self.assertIn("tier_sweeps=160", reason)

    def test_budget_carries_a_truncated_tier_to_the_serial_fixed_point(self):
        """Leave a two-sweep linked tier short of a sliding stance's serial fixed point, and reach it within budget.

        The rows are the first step of :func:`sliding_stance` as a CPU solver
        builds them, and that solver's serial sweep (2000 iterations) gives the
        fixed point. The matrix-free linked tier (the float64 reference) stops
        3.4e-2 of the velocity scale short of it after two sweeps and lands within
        1.1e-7 of the scale after 2000 (:func:`assert_budget_reaches_the_fixed_point`).
        TestPatchParallelPhysics runs the solver's tiers on the same stance.
        """
        model = sliding_stance("cpu")
        with patch.dict(SolverFeatherPGS._route_pins, _DENSE):
            solver = SolverFeatherPGS(model, friction_anchor_beta=0.2, pgs_iterations=2000)
        _rollout(model, solver, 1)
        case, serial = world_case(solver)
        # Fourteen contact rows: two patch rings of four normals, each ring with one anchor (a pair and a spin row).
        rings = {
            tuple(sorted(ring_rows(case["parents"], int(normal))))
            for normal in np.flatnonzero(case["types"] == PGS_CONSTRAINT_TYPE_CONTACT)
        }
        self.assertEqual(sorted(len(ring) for ring in rings), [4, 4])
        self.assertEqual(int(np.count_nonzero(case["types"] == PGS_CONSTRAINT_TYPE_FRICTION)), 4)
        self.assertEqual(int(np.count_nonzero(case["types"] == PGS_CONSTRAINT_TYPE_TORSION)), 2)
        self.assertEqual(len(case["types"]), 14)
        options = {"matrix_free": True, "inkernel": False, "exact": False, "nesterov": True, "tol": 0.0, "linked": True}
        gaps = {}
        for sweeps in (2, 2000):
            _, velocity = parallel_sweep_reference(case, sweeps=sweeps, **options)
            gaps[sweeps] = float(np.max(np.abs(velocity - serial)))
        assert_budget_reaches_the_fixed_point(self, gaps[2], gaps[2000], float(np.max(np.abs(serial))))


def _routes(case=None, **pins):
    """A route table whose class ``case`` holds (``None``: no earlier case claims it)."""
    routes = RouteTable(pins)
    if case is not None:
        routes.taken(case)
        routes.claim(case)
    return routes


class TestTierRule(unittest.TestCase):
    """Build the Jacobi tiers for case K4, a tree beside a free body, where no earlier case claims the class."""

    @staticmethod
    def _probe(**changes):
        """A solver's tier inputs, allegro's by default (a 16-coordinate hand and a cube, 192 dense rows)."""
        probe = SimpleNamespace(
            factor=SimpleNamespace(eliminates=False),
            placement=SimpleNamespace(
                mf_sweeps=changes.pop("mf_sweeps", True),
                dense_max_constraints=changes.pop("dense_max_constraints", 192),
                inkernel_response_meta=changes.pop(
                    "inkernel_response_meta",
                    lambda: ((16, 6, 0, 16), "The parallel tiers form the response in-kernel."),
                ),
            ),
            pins={},
            routes=_routes(),
            laws=SimpleNamespace(set=plain_laws()),
            forest=SimpleNamespace(has_free_rigid_bodies=True, has_non_free_articulations=True),
        )
        for name, value in changes.items():
            setattr(probe, name, value)
        Sweep.plan_tier_rows(probe)
        return probe

    def test_a_tree_beside_a_body_takes_the_tiers(self):
        """Bound the tiers by the dense row capacity, at most 128 rows, in whole quads."""
        probe = self._probe()
        self.assertEqual((probe.placement.ink_rows, probe.placement.ink_sizes), (128, (16, 6, 0, 16)))
        self.assertEqual(self._probe(dense_max_constraints=50).placement.ink_rows, 48)
        self.assertIsNone(self._probe(dense_max_constraints=3).placement.ink_sizes)

    def test_other_classes_keep_their_placement(self):
        """Leave every other class to its own placement and say why."""
        for changes, reason in (
            # Packed factors (a single tree) or the sparse diagonal (a star) leave no matrix-free sweep.
            ({"mf_sweeps": False}, "No matrix-free sweep"),
            ({"factor": SimpleNamespace(eliminates=True)}, "pre-elimination"),
            ({"routes": _routes("K5")}, "case K5"),
            ({"forest": SimpleNamespace(has_free_rigid_bodies=False, has_non_free_articulations=True)}, "case K4"),
            ({"forest": SimpleNamespace(has_free_rigid_bodies=True, has_non_free_articulations=False)}, "case K4"),
            ({"routes": _routes(placement="general")}, "general placement"),
            # The tiers' bodies are written for CUDA only (the CPU takes the general row).
            ({"laws": SimpleNamespace(set=plain_laws(cuda=False))}, "CUDA device"),
            ({"pins": {"tiers": 0}}, "tiers=0"),
            (
                {"inkernel_response_meta": lambda: (None, "A world has several articulations of size 6.")},
                "several",
            ),
        ):
            with self.subTest(changes=sorted(changes)):
                probe = self._probe(**changes)
                self.assertEqual((probe.placement.ink_rows, probe.placement.ink_sizes), (0, None))
                self.assertIn(reason, probe.routes.reasons["K4"])

    def test_a_pin_builds_the_tiers_on_a_single_tree(self):
        """Build the tiers a test pin asks for on a class whose response they form, at the pinned bound."""
        probe = self._probe(_has_free_rigid_bodies=False, pins={"tiers": 32})
        self.assertEqual((probe.placement.ink_rows, probe.placement.ink_sizes), (32, (16, 6, 0, 16)))


def _dummy(arg):
    """A stand-in for an input the kernel variant never reads: a one-element array, an empty struct or zero."""
    if wp.types.is_array(arg.type):
        return wp.zeros((1,) * arg.type.ndim, dtype=arg.type.dtype)
    if wp.types.type_is_struct(arg.type):
        return arg.type()
    return arg.type(0)


def identity_walk(dofs: int, device) -> WalkTrees:
    """The walk of one body of ``dofs`` coordinates whose factor is the identity (``s = ut = 0``, ``ell = 1``): its
    rows are the coordinates' own, and its leave is ``u`` itself."""
    walk = WalkTrees()
    zero = np.zeros((dofs, 6), np.float32)
    walk.data = wp.array(pack_walk_records(zero, zero, np.ones(dofs), np.zeros(dofs)), dtype=wp.float32, device=device)
    walk.link = wp.zeros((1, WALK_LINK), dtype=wp.float32, device=device)
    walk.now = wp.zeros(dofs, dtype=wp.spatial_vector, device=device)
    walk.now_link = wp.zeros((1, WALK_LINK), dtype=wp.float32, device=device)
    walk.now_com = wp.zeros(1, dtype=wp.vec3, device=device)
    walk.origin = walk.now_origin = wp.zeros(1, dtype=wp.vec3, device=device)
    for name, values in (
        ("joint_start", [0]),
        ("joint_end", [1]),
        ("dof_start", [0]),
        ("dofs", [dofs]),
        ("dof_body", [0] * dofs),
        ("body_joint", [0]),
        ("joint_art", [0]),
    ):
        setattr(walk, name, wp.array(values, dtype=wp.int32, device=device))
    walk.release = release_view(1, 0, 1, device)
    walk.du = wp.zeros(dofs, dtype=wp.float32, device=device)
    return walk


def _launch_parallel(tier, case, *, device):
    """Launch one world through a parallel tier and return its impulses and velocity.

    The tier reads the one writer's stored rows and leaves by the walk. The case is posed in its factor's coordinates
    so the walk is the identity: the rows are ``Z = L^-1 P J^T``, the velocity ``u = M^-1 v`` with ``M = P^T L^-T P``
    (``dv = M Z^T lambda``), the incident velocity ``J v_hat``; the published velocity maps back by ``M``."""
    kernel = tier.kernel
    count = len(case["J"])
    padded = np.zeros((1, _CAPACITY))
    L = case["L"]

    def rows(values, dtype=wp.float32):
        out = padded.copy().astype(np.float32 if dtype == wp.float32 else np.int32)
        out[0, :count] = values
        return wp.array(out, dtype=dtype, device=device)

    v_hat = case["velocity"] - case["impulses"] @ case["Y"]
    # The rows' law's regularization: one value for every row.
    (cfm,) = np.unique(np.asarray(case["cfm"], np.float32))
    Z = np.zeros((1, _CAPACITY, _DOFS), np.float32)
    Z[0, :count] = case["Z"]
    values = {
        "general_world_count": wp.array([1], dtype=wp.int32, device=device),
        "general_worlds": wp.zeros(1, dtype=wp.int32, device=device),
        "general_world_grid_stride": 1,
        "world_constraint_count": wp.array([count], dtype=wp.int32, device=device),
        "mf_constraint_count": wp.zeros(1, dtype=wp.int32, device=device),
        "dense_phase_bounds": wp.zeros((1, 2), dtype=wp.int32, device=device),
        "world_dof_indices": wp.array(np.arange(_DOFS, dtype=np.int32)[None], dtype=wp.int32, device=device),
        "rhs_bias": rows(case["rhs"]),
        "row_incident": rows(case["J"] @ v_hat),
        "world_diag": rows(case["diag"]),
        "world_impulses": rows(case["impulses"]),
        # The world's stored rows (dense): every row on articulation 0, its coordinates from 0.
        "packed_Z": wp.array(Z, dtype=wp.float32, device=device),
        "world_row_type": rows(case["types"], wp.int32),
        "world_row_parent": rows(case["parents"], wp.int32),
        "world_row_mu": rows(case["mu"]),
        "cfm": float(cfm),
        "iterations": 1,
        "omega": 1.0,
        # The predictor: the case's velocity less the response of its starting impulses, which the tier applies.
        "v_out": wp.array((L.T @ v_hat[::-1])[::-1].astype(np.float32), dtype=wp.float32, device=device),
        "walk": identity_walk(_DOFS, device),
    }
    values["ink_meta"] = wp.array([0, 0, -1, 0], dtype=wp.int32, device=device)
    values["row_sides"] = wp.array(np.tile([0, -1], (1, _CAPACITY, 1)).astype(np.int32), dtype=wp.vec2i, device=device)
    values["response_dofs"] = wp.array([_DOFS], dtype=wp.int32, device=device)
    values["articulation_world_dof_offset"] = wp.zeros(1, dtype=wp.int32, device=device)
    values["packed_dense"] = 1
    arguments = {arg.label: arg for arg in kernel.adj.args}
    inputs = [
        values[name] if name in values else _dummy(arguments[name])
        for name in inspect.signature(kernel.func).parameters
    ]
    wp.launch_tiled(kernel, dim=[1], inputs=inputs, block_dim=tier.work.block, device=device)
    leave_walk(values["walk"], values["v_out"], device)
    u = values["v_out"].numpy().astype(np.float64)
    return values["world_impulses"].numpy()[0, :count], np.linalg.solve(L.T, u[::-1])[::-1]


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestParallelPatchKernel(unittest.TestCase):
    def test_linked_tiers_match_the_reference(self):
        """Match the linked tiers (one and two warps, in-kernel response) to the reference.

        The point tiers on the same rows without ring links pin the reference to the
        kernel's existing law first, so a mismatch of the linked tiers isolates the ring walk.
        Each case also runs with its contacts' slots spread (spread_rows), which puts the
        two-warp case's rings and pairs on both sides of row 32.
        """
        device = wp.get_device("cuda:0")
        arch = str(device.arch)
        for (faces, rows, min_rows), warm, spread in itertools.product(
            ((1, 32, 0), (3, 64, 32)), (False, True), (False, True)
        ):
            options = {"matrix_free": True, "inkernel": True, "exact": False, "nesterov": True}
            with self.subTest(faces=faces, warm=warm, spread=spread, **options):
                for linked in (False, True):
                    case = parallel_case(faces, faces=faces, linked=linked, warm=warm)
                    if spread:
                        case, _ = spread_rows(case, _SPREAD_SEED)
                    self.assertGreater(len(case["J"]), min_rows)
                    kernel = sweep_routes_tiers.parallel_kernel(
                        _CAPACITY,
                        32,
                        _DOFS,
                        arch,
                        rows=rows,
                        min_rows=min_rows,
                        sweeps=24,
                        inkernel_response=(_DOFS, 0, 0, 0),
                        linked_rows=linked,
                    )
                    impulses, velocity = _launch_parallel(kernel, case, device=device)
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
        for name, case in cases.items():
            options = {"matrix_free": True, "inkernel": True, "exact": False, "nesterov": True}
            with self.subTest(case=name):
                kernel = sweep_routes_tiers.parallel_kernel(
                    _CAPACITY,
                    32,
                    _DOFS,
                    arch,
                    rows=32,
                    sweeps=2000,
                    inkernel_response=(_DOFS, 0, 0, 0),
                    linked_rows=True,
                )
                impulses, velocity = _launch_parallel(kernel, case, device=device)
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
        for sweeps in (2, 2000):
            options = {"matrix_free": True, "inkernel": True, "exact": False, "nesterov": True}
            with self.subTest(sweeps=sweeps):
                results = {}
                for last_word in (False, True):
                    kernel = sweep_routes_tiers.parallel_kernel(
                        _CAPACITY,
                        32,
                        _DOFS,
                        arch,
                        rows=32,
                        sweeps=sweeps,
                        inkernel_response=(_DOFS, 0, 0, 0),
                        linked_rows=True,
                        velocity_limit_pass=last_word,
                    )
                    impulses, velocity = _launch_parallel(kernel, case, device=device)
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
# Dense storage (a test-only route pin).
_DENSE = {"storage": "dense"}
# Float-order tolerances of an exact route change (assert_states_close of the sparse solver tests).
_STATE_FLOOR = {"joint_q": 2.0e-4, "body_q": 2.0e-4, "joint_qd": 2.0e-3, "body_qd": 2.0e-3}


def _matrix_free_solver(model, *, beta, parallel_rows, pins=_DENSE, sweeps=None, **options):
    """A matrix-free solver whose sweep is serial (``parallel_rows=0``) or the Jacobi tiers of ``parallel_rows`` rows
    (test-only route pins: ``tiers``, and ``tier_sweeps`` when ``sweeps`` replaces the equal-accuracy budget), on
    dense storage unless ``pins`` say otherwise."""
    settings = {"friction_anchor_beta": beta, "pgs_iterations": 8, "dense_max_constraints": 32}
    settings.update(options)
    tier_pins = {"tiers": parallel_rows, **({"tier_sweeps": sweeps} if sweeps is not None else {})}
    with patch.dict(SolverFeatherPGS._route_pins, {**pins, **tier_pins}):
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


def _spinning_sphere(device, *, speed, spin, mu=0.5):
    """A sphere on prismatic x, y and z joints and a yaw joint over the ground, sliding at ``speed`` along x and
    spinning at ``spin`` about the vertical: one contact, whose friction pair and, under contact torsion, spin row
    (the shapes' default torsional friction, ``mu_t / mu = 0.01`` m) brake both."""
    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    cfg.density, cfg.mu = 0.0, mu
    parent, joints = -1, []
    for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z):
        body = builder.add_link(mass=0.01, inertia=wp.mat33(np.eye(3) * 1.0e-5), lock_inertia=True)
        joints.append(builder.add_joint_prismatic(parent=parent, child=body, axis=axis))
        parent = body
    sphere = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 1.0e-3), lock_inertia=True)
    joints.append(builder.add_joint_revolute(parent=parent, child=sphere, axis=newton.Axis.Z))
    builder.add_shape_sphere(sphere, radius=0.05, cfg=cfg)
    builder.add_articulation(joints)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    q, qd = model.joint_q.numpy(), model.joint_qd.numpy()
    q[2] = 0.049  # the vertical joint: the sphere rests 1 mm deep
    qd[0], qd[3] = speed, spin
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

    A tier takes a world with at most ``budget`` dense rows and no matrix-free rows; the serial sweep returns early
    for exactly those worlds, so a world in that domain is the tier's (``skip_tier_rows``).
    """
    test.assertEqual(solver.placement.ink_rows, budget)
    test.assertTrue(solver.sweep.tiers)
    counts, free_rows = solver.placement.constraint_count.numpy(), solver.placement.mf_constraint_count.numpy()
    test.assertGreater(int(counts.max()), 0)
    for world, (rows, mf_rows) in enumerate(zip(counts, free_rows, strict=True)):
        if rows:
            test.assertLessEqual(int(rows), budget, f"world {world}")
            test.assertEqual(int(mf_rows), 0, f"world {world}")


def _friction_pairs(solver, world):
    """Each friction pair's impulse magnitude and pooled budget ``mu * (ring's normal impulses)``."""
    count = int(solver.placement.constraint_count.numpy()[world])
    kinds = solver.placement.row_type.numpy()[world, :count]
    parents = solver.placement.row_parent.numpy()[world, :count]
    mu, impulses = solver.placement.row_mu.numpy()[world, :count], solver.placement.impulses.numpy()[world, :count]
    pairs = []
    for row in np.flatnonzero(kinds == PGS_CONSTRAINT_TYPE_FRICTION):
        parent = int(parents[row])
        if row == parent + 1:
            budget = float(mu[row]) * ring_load(parents, impulses.astype(np.float64), parent)
            pairs.append((math.hypot(impulses[row], impulses[row + 1]), budget))
    return pairs


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestPatchParallelRoutes(unittest.TestCase):
    def test_patch_runs_linked_tiers(self):
        """Build linked tiers for patch friction.

        Every tier sweeps the equal-accuracy budget of 32 rows at eight iterations.
        """
        model = standing_model()
        options = {"pins": {}}
        patch_tiers = _matrix_free_solver(model, beta=0.2, parallel_rows=32, **options)
        for solver, linked in ((patch_tiers, True),):
            self.assertEqual(solver.placement.ink_rows, 32)
            # The tier owns the ordinary buffers for either law.
            self.assertIsNone(solver.placement.sparse_mass_matrix_size)
            keys = [tier.kernel.key for tier in solver.sweep.tiers]
            self.assertTrue(keys)
            self.assertTrue(all(_law_key(key).endswith("_linked") == linked for key in keys), keys)
            self.assertTrue(all(("_linked" in key) == linked for key in keys), keys)
            sweeps = sweep_routes_tiers.tier_sweeps(32, 8)
            self.assertTrue(all(f"_sweeps{sweeps}_" in key for key in keys), keys)
            tiers = solver.get_path_selection()["routes"]["sweep"]["tiers"]
            self.assertIn(tiers["family"], ("parallel", "keyed_parallel"))
            self.assertEqual(tiers["linked_rows"], linked)
            self.assertEqual((tiers["sweeps"], tiers["sweeps_rule"]), (sweeps, "auto"))

    def test_keyed_rows_record_names_the_launched_builders(self):
        """Name in the route record the keyed-row builders one step launches.

        Matrix-free parallel tiers that form the response in-kernel admit keyed rows on CUDA. The keyed parallel tiers build the direct rows in the sweep, which every entry
        points to; the scatter and the fallback are the row builders a step launches outside it.
        """
        model = standing_model()
        with patch.dict(os.environ):
            for law, beta in (("patch", 0.2),):
                with self.subTest(law=law):
                    solver = _matrix_free_solver(model, beta=beta, parallel_rows=32, enable_joint_limits=True)
                    routes = solver.get_path_selection()["routes"]
                    record = routes["contact_rows"]
                    self.assertTrue(record["keyed_rows"]["enabled"], record["keyed_rows"]["reason"])
                    self.assertEqual(routes["sweep"]["tiers"]["family"], "keyed_parallel")
                    self.assertEqual(record["producer"], "keyed_rows")
                    builder = {
                        "method": "keyed_rows",
                        "kernels": ["scatter_contact_keys", "keyed_general_units", "write_walked_rows"],
                        "direct_rows": "routes.sweep.tiers",
                    }
                    self.assertEqual(record["builder_by_dof"], dict.fromkeys(record["builder_by_dof"], builder))
                    with launched_kernels() as keys:
                        _rollout(model, solver, 1)
                    assert_row_builders_named(self, record, keys)

    def test_keyed_linked_tiers_replay_alike_with_and_without_frame_reuse(self):
        """Replay captured keyed linked-tier steps bit for bit whether a shared contact frame is kept or sorted again.

        Patch friction runs keyed rows into the linked tiers by default, and a captured build that shares the
        previous captured build's contact frame keeps its sorted pairs (``FEATHER_PGS_PATCH_FRAME_REUSE``, default
        on). Neither the source A/B nor the tier probes ran the two together. A graph of one collision pass and two
        substeps sorts its first build and keeps the second; under ``=0`` it sorts both. Frame reuse keeps the
        patch state bitwise, so every replay must leave the same joint coordinates, velocities and impulses
        under both settings.
        """
        from newton.tests.test_feather_pgs_patch_mechanism import _admitted_modes  # noqa: PLC0415
        from newton.tests.unittest_utils import scoped_capture_after_gc  # noqa: PLC0415

        model = standing_model()
        dt = 1.0 / 240.0
        replays = {}
        for reuse in (None, "0"):
            with patch.dict(os.environ):
                os.environ.pop("FEATHER_PGS_PATCH_FRAME_REUSE", None)
                if reuse is not None:
                    os.environ["FEATHER_PGS_PATCH_FRAME_REUSE"] = reuse
                solver = _matrix_free_solver(model, beta=0.2, parallel_rows=32, enable_joint_limits=True)
                self.assertIsNotNone(solver.placement.keyed_rows, solver.get_path_selection()["routes"]["contact_rows"])
                tiers = solver.get_path_selection()["routes"]["sweep"]["tiers"]
                self.assertEqual((tiers["family"], tiers["linked_rows"]), ("keyed_parallel", True))
                self.assertEqual(solver.laws.friction_patches.reuse_frames, reuse is None)
                states, control = [model.state(), model.state()], model.control()
                newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, deterministic=True)
                contacts = pipeline.contacts()

                def pair(states=states, solver=solver, pipeline=pipeline, contacts=contacts, control=control):
                    """One collision pass, then two substeps that share its contact frame and return to states[0]."""
                    pipeline.collide(states[0], contacts)
                    for state, following in ((states[0], states[1]), (states[1], states[0])):
                        state.clear_forces()
                        solver.step(state, following, control, contacts, dt)

                # Eager pairs load every kernel and buffer the graph launches; eager builds always sort.
                for _ in range(2):
                    pair()
                outcomes = _admitted_modes(solver.laws.friction_patches)
                wp.synchronize_device(model.device)
                with scoped_capture_after_gc(device=model.device) as capture:
                    pair()
                self.assertEqual(outcomes, ["sort", "keep"] if reuse is None else ["sort", "sort"], reuse)
                replays[reuse] = []
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                    replays[reuse].append(
                        {
                            "joint_q": states[0].joint_q.numpy(),
                            "joint_qd": states[0].joint_qd.numpy(),
                            "impulses": solver.placement.impulses.numpy(),
                        }
                    )
                self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0, "the feet must touch the ground")
        self.assertTrue(np.any(replays[None][-1]["impulses"] != 0.0), "the feet must carry impulses")
        for replay, (kept, sorted_every_build) in enumerate(zip(replays[None], replays["0"], strict=True)):
            for name, value in kept.items():
                np.testing.assert_array_equal(value, sorted_every_build[name], f"replay {replay}: {name}")

    def test_limited_linked_tiers_take_the_last_word(self):
        """Build the last-word pass into velocity-limited linked tiers, record it, and hold the limit.

        A constant joint force pushes a resting sphere, whose joints are limited to
        0.5, by twice its limit within every step (:func:`_pushed_sphere`): without
        limit rows two steps carry it past twice the limit. A sphere that starts past
        its limit cannot show the pass, because the pre-solve prescale puts it on the
        limit and its limit rows never bind. Only the pass promises the limit: a tier
        without it stops where its budget leaves it, while the pass holds it to 1e-5.
        Truncated to two sweeps, the tier without the
        pass must leave the sphere at least 5 % past its limit, or the fixture cannot
        show what the pass does, and the pass must hold it to 1e-5 (the reference on
        the CPU rows of test_last_word_holds_a_pushed_sphere_at_its_limit: 43.5 % and
        1.7e-6 past). Converged (2000 sweeps, tolerance 0), the tier's limit rows hold
        the limit to 1e-5 without the pass, and the pass moves no velocity by more
        than 1e-6.
        """
        limit = 0.5
        model = _pushed_sphere("cuda:0", limit=limit)
        with patch.dict(os.environ):
            built = {
                limits: _matrix_free_solver(model, beta=0.2, parallel_rows=32, enable_joint_velocity_limits=limits)
                for limits in (False, True)
            }
            budgets = {
                (sweeps, True): _matrix_free_solver(
                    model, beta=0.2, parallel_rows=32, sweeps=sweeps, enable_joint_velocity_limits=True
                )
                for sweeps in (2, 2000)
            }
        for limits, solver in built.items():
            with self.subTest(limits=limits):
                keys = [tier.kernel.key for tier in solver.sweep.tiers]
                self.assertTrue(keys)
                self.assertTrue(all(_law_key(key).endswith("_linked_lw" if limits else "_linked") for key in keys))
                selected = solver.get_path_selection()
                self.assertEqual(selected["routes"]["sweep"]["tiers"]["velocity_limit_pass"], limits)
                state, _ = _rollout(model, solver, 2)
                _assert_tier_owned(self, solver, 32)
                for name in _STATE_FIELDS:
                    self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
                speed = float(np.max(np.abs(state.joint_qd.numpy()[:2])))
                if not limits:
                    self.assertGreater(speed, 2.0 * limit, "the push must carry an unlimited sphere past its limit")
                else:
                    self.assertLessEqual(speed, limit * (1.0 + 1.0e-5))
        velocities = {}
        for (sweeps, last_word), solver in budgets.items():
            keys = [tier.kernel.key for tier in solver.sweep.tiers]
            self.assertTrue(
                keys and all(f"_sweeps{sweeps}_" in key and _law_key(key).endswith("_lw") == last_word for key in keys),
                keys,
            )
            state, _ = _rollout(model, solver, 1)
            _assert_tier_owned(self, solver, 32)
            velocities[sweeps, last_word] = state.joint_qd.numpy().astype(np.float64)
        speeds = {budget: float(np.max(np.abs(velocity[:2]))) for budget, velocity in velocities.items()}
        for budget in ((2, True), (2000, True)):
            self.assertLessEqual(speeds[budget], limit * (1.0 + 1.0e-5), (budget, speeds))

    def test_every_tier_takes_the_automatic_budget(self):
        """Build linked tiers with the equal-accuracy budget, or a pinned one, and record it."""
        model = standing_model()
        options = {"dense_max_constraints": 64, "pgs_iterations": 8}
        with patch.dict(os.environ):
            built = {}
            for beta, sweeps in itertools.product((0.2,), (None, 64)):
                built[beta, sweeps] = _matrix_free_solver(model, beta=beta, parallel_rows=64, sweeps=sweeps, **options)
        for (beta, sweeps), solver in built.items():
            with self.subTest(beta=beta, sweeps=sweeps):
                # 7/2 * 8 * sqrt(rows / 32): 28 and 40 sweeps for 32 and 64 rows, unless a test pin sets them.
                expected, rule = ([28, 40], "auto") if sweeps is None else ([sweeps, sweeps], "pinned")
                keys = [tier.kernel.key for tier in solver.sweep.tiers]
                self.assertEqual(len(keys), 2, keys)
                for key, budget in zip(keys, expected, strict=True):
                    self.assertIn(f"_sweeps{budget}_", key)
                    self.assertEqual(_law_key(key).endswith("_linked"), bool(beta), key)
                selected = solver.get_path_selection()
                tiers = selected["routes"]["sweep"]["tiers"]
                self.assertEqual([tier["sweeps"] for tier in tiers["tiers"]], expected)
                self.assertEqual((tiers["sweeps"], tiers["sweeps_rule"]), (max(expected), rule))
                state, _ = _rollout(model, solver, 2)
                for name in _STATE_FIELDS:
                    self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)


# 3 x the point tiers' gap plus the floor (_STATE_FLOOR), measured on cd41551f2 (2026-11-05): the point gaps were
# 1.23e-4, 1.48e-3, 6.13e-5 and 1.48e-3; the law's tiers measure 1.27e-4, 1.14e-3, 6.44e-5 and 1.15e-3.
_FEET_BOUND = {"joint_q": 5.7e-4, "joint_qd": 6.4e-3, "body_q": 3.8e-4, "body_qd": 6.4e-3}
# The point tiers stopped the box where their serial sweep did (gap 0), so the bound was its floor, 1% of the analytic
# stopping distance; the law's tiers measure a gap of 0 too.
_SLIDE_BOUND = 0.01


@unittest.skipUnless(wp.is_cuda_available(), "Parallel matrix-free tiers require CUDA")
class TestPatchParallelPhysics(unittest.TestCase):
    """Compare the parallel tiers with the serial sweep, their gap within a stated bound.

    Parallel tiers are Jacobi sweeps, not float reorderings of the serial sweep.
    Every tier steps each friction pair isotropically, so the tiers share the serial
    sweep's fixed point (a sliding pair converges to maximal dissipation) and the
    rollouts differ by the convergence gap. The bounds are the ones the deleted
    no-stick point friction's tiers set on the same fixtures at the same default
    budget (three times their measured gap, with a floor at the float-order
    tolerance of an exact route change), held as numbers. Every parallel step must
    leave each world in the tiers' domain, so the comparison is never serial against
    serial.
    """

    # Tier rows: the serial sweep and the tiers at their default budget.
    _ARMS = (0, 32)

    def _assert_budget(self, solver, parallel_rows):
        """Require the tiers to sweep the automatic budget of 32 rows at eight iterations."""
        keys = [tier.kernel.key for tier in solver.sweep.tiers]
        self.assertEqual(bool(keys), bool(parallel_rows), keys)
        budget = sweep_routes_tiers.tier_sweeps(32, 8)
        self.assertTrue(all(f"_sweeps{budget}_" in key for key in keys), keys)

    def test_standing_feet_match_the_serial_patch_sweep(self):
        """Keep loaded, ring-linked anchors and the serial sweep's motion on two box feet, the tiers' gap to the
        serial sweep within :data:`_FEET_BOUND`."""
        model = standing_model()
        velocity = model.joint_qd.numpy()
        velocity[0::10] = 0.05  # the torsos drift sideways, loading the feet's friction without slipping
        model.joint_qd.assign(velocity)
        with patch.dict(os.environ):
            states = {}
            for parallel_rows in self._ARMS:

                def check(step, solver, contacts, state, parallel_rows=parallel_rows):
                    for name in _STATE_FIELDS:
                        self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
                    self.assertFalse(solver.placement.constraint_overflow.numpy().any())
                    if parallel_rows:
                        _assert_tier_owned(self, solver, parallel_rows)
                    if parallel_rows:
                        # Every stored parallel iterate lies in its pooled cone; a serial sweep's last
                        # pair update can precede a ring normal's, so only the tiers hold it exactly.
                        assert_ring_coulomb(solver)
                        assert_loaded_anchors(solver, contacts, carried=step > 0)

                solver = _matrix_free_solver(model, beta=0.2, parallel_rows=parallel_rows)
                self._assert_budget(solver, parallel_rows)
                states[parallel_rows] = _rollout(model, solver, 24, after_step=check)[0]
        gap = _state_differences(states[32], states[0])
        for name in _STATE_FIELDS:
            self.assertLessEqual(gap[name], _FEET_BOUND[name], (name, gap))

    def test_box_rests_and_slides_like_the_serial_patch_sweep(self):
        """Hold a resting box, and stop a sliding one where the serial patch sweep stops it.

        A sliding box saturates its patch's pooled cone: each anchor pair carries
        mu / 2 of the four normals' load. Pooling each pair on its own normal would
        halve the friction and double the stopping distance. The tiers stop the box
        within :data:`_SLIDE_BOUND` of the analytic distance of the serial sweep.
        """
        speed, mu, gravity = 1.0, 0.5, 9.81
        expected = speed * speed / (2.0 * mu * gravity)
        model = _sliding_boxes("cuda:0", (0.0, speed), mu=mu)
        distances = {}
        with patch.dict(os.environ):
            for parallel_rows in self._ARMS:
                ratios = []

                def check(step, solver, contacts, state, ratios=ratios, parallel_rows=parallel_rows):
                    if parallel_rows:
                        _assert_tier_owned(self, solver, parallel_rows)
                    if 2 <= step <= 30:  # sliding: every friction pair sits on its cone
                        pairs = _friction_pairs(solver, world=1)
                        ratios.extend(magnitude / budget for magnitude, budget in pairs if budget > 0.0)

                solver = _matrix_free_solver(model, beta=0.2, parallel_rows=parallel_rows)
                self._assert_budget(solver, parallel_rows)
                state, _ = _rollout(model, solver, 96, after_step=check)
                q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
                with self.subTest(parallel_rows=parallel_rows):
                    self.assertLess(np.max(np.abs(q[0:2])), 2.0e-4, "the resting box must not creep")
                    self.assertLess(abs(qd[3]), 1.0e-3, "the sliding box must stop")
                    self.assertLess(abs(q[3] - expected), 0.2 * expected, "friction must act at the full coefficient")
                    self.assertTrue(ratios)
                    self.assertGreater(min(ratios), 0.95)
                    # A parallel iterate is projected onto its pooled cone; a serial sweep may update
                    # a ring normal after the pair, leaving the pair marginally outside.
                    self.assertLess(max(ratios), 1.0 + (1.0e-4 if parallel_rows else 5.0e-3))
                distances[parallel_rows] = float(q[3])
        self.assertLessEqual(abs(distances[32] - distances[0]), _SLIDE_BOUND * expected, distances)

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
        with patch.dict(os.environ):
            routes = {
                "serial": _matrix_free_solver(model, beta=0.2, parallel_rows=0, pgs_iterations=2000),
                "tier": _matrix_free_solver(model, beta=0.2, parallel_rows=32, sweeps=2000),
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
        np.testing.assert_allclose(velocities["tier"], velocities["serial"], atol=1.0e-5 * scale)

    def test_budget_carries_the_tiers_to_the_serial_fixed_point(self):
        """Converge the linked tiers on their budget, where a truncated budget stays short.

        On the first step of :func:`sliding_stance` a linked tier pinned to two
        sweeps (a test pin, ``tier_sweeps``) must stay short of the serial
        sweep's velocity (2000 iterations), and a budget of 2000 sweeps must
        carry it there (:func:`assert_budget_reaches_the_fixed_point`; the float64
        reference on the CPU solver's rows: 3.4e-2 of the scale after two sweeps, 1.1e-7
        after 2000). A single sliding contact cannot show what a budget carries: two
        sweeps converge it (4.2e-6 of the scale in the reference). One step keeps the
        comparison on the solve, as later steps add the float32 drift of the feet's
        poses that the position correction amplifies.
        """
        model = sliding_stance("cuda:0")
        with patch.dict(os.environ):
            routes = {"serial": _matrix_free_solver(model, beta=0.2, parallel_rows=0, pgs_iterations=2000)}
            routes["truncated"] = _matrix_free_solver(model, beta=0.2, parallel_rows=32, sweeps=2)
            routes["budgeted"] = _matrix_free_solver(model, beta=0.2, parallel_rows=32, sweeps=2000)
        for name in ("truncated", "budgeted"):
            sweeps = 2000 if name == "budgeted" else 2
            tiers = routes[name].sweep.tiers
            self.assertTrue(tiers and all(f"_sweeps{sweeps}_" in tier.kernel.key for tier in tiers), name)
        velocities = {}
        for name, solver in routes.items():
            state, _ = _rollout(model, solver, 1)
            if name != "serial":
                _assert_tier_owned(self, solver, 32)
            velocities[name] = state.joint_qd.numpy().astype(np.float64)
        gaps = {
            name: float(np.max(np.abs(velocities[name] - velocities["serial"]))) for name in ("truncated", "budgeted")
        }
        scale = float(np.max(np.abs(velocities["serial"])))
        assert_budget_reaches_the_fixed_point(self, gaps["truncated"], gaps["budgeted"], scale)

    def test_lone_live_tangent_keeps_its_friction(self):
        """Brake a sphere whose second tangent no joint moves as the serial sweep brakes it.

        The tangent without a Jacobian takes the live one's step: its zero step must
        not stop the live one.
        """
        model = _sliding_sphere("cuda:0", (newton.Axis.X, newton.Axis.Z), (3.0, 1.0), (1.0,))
        with patch.dict(os.environ):
            serial = _matrix_free_solver(model, beta=0.2, parallel_rows=0, pgs_iterations=2000)
            tier = _matrix_free_solver(model, beta=0.2, parallel_rows=32, sweeps=2000)
        speeds = {}
        for name, solver in (("serial", serial), ("tier", tier)):
            state, _ = _rollout(model, solver, 6)
            speeds[name] = float(state.joint_qd.numpy()[0])
        _assert_tier_owned(self, tier, 32)
        # Six steps of mu * g at the sphere's weight over the four-fold chain mass brake it by about 0.03 m/s.
        self.assertLess(speeds["serial"], 1.0 - 0.01)
        self.assertAlmostEqual(speeds["tier"], speeds["serial"], delta=1.0e-5)

    def test_sliding_spinning_triple_keeps_the_serial_friction(self):
        """Brake a sphere that slides and spins under contact torsion as the serial sweep brakes it.

        The friction triple steps isotropically in its ball's scaled coordinates, so its
        fixed point opposes the slip and the spin in the law's metric (maximal
        dissipation), the serial sweep's: both decay as the serial sweep's. A spin row
        stepped by its own diagonal, ten times its isotropic step here, turns the
        friction towards the spin: the spin stops early and the slide late.
        """
        model = _spinning_sphere("cuda:0", speed=1.0, spin=20.0)
        with patch.dict(os.environ):
            serial = _matrix_free_solver(model, beta=0.2, parallel_rows=0, pgs_iterations=2000, contact_torsion=True)
            tier = _matrix_free_solver(model, beta=0.2, parallel_rows=32, sweeps=2000, contact_torsion=True)
        velocities = {}
        for name, solver in (("serial", serial), ("tier", tier)):
            state, _ = _rollout(model, solver, 24)
            velocities[name] = state.joint_qd.numpy()[[0, 3]].astype(np.float64)
        _assert_tier_owned(self, tier, 32)
        # Twenty-four steps of the elliptic law brake the slide by about 0.5 m/s and the spin by about 1.3 rad/s; the
        # spin row's own step would turn the friction towards the spin (about 0.2 m/s and 4 rad/s).
        self.assertLess(velocities["serial"][0], 0.6)
        self.assertLess(velocities["serial"][1], 19.5)
        np.testing.assert_allclose(velocities["tier"], velocities["serial"], rtol=1.0e-4)

    def test_declined_world_keeps_the_serial_sweep(self):
        """Sweep a world the linked tiers decline serially, at the tiers' fixed point.

        Patch friction runs the linked tiers by default. A palm and three two-link
        fingers, every joint velocity-limited (fourteen limit rows, no contacts), is
        too large for an 8-row tier, so its world keeps the serial sweep: it lands
        where the serial route (no tier requested) lands, to float order. Both sweeps
        accumulate velocity-limit rows, so a 32-row tier that owns the world lands
        there too. Converged settings (2000 iterations and sweeps) make the law, not
        the budget, decide.
        """
        model = _limited_hand("cuda:0")
        torques = np.array([0.03, 0.006, -0.005, 0.0002, 0.008, -0.007, 0.001], np.float32)
        # As test_serial_sweep_converges_to_the_parallel_tier builds its routes on this hand.
        converged = {"enable_joint_velocity_limits": True, "pins": _DENSE, "pgs_iterations": 2000, "sweeps": 2000}
        arms = {
            "declined": _matrix_free_solver(model, beta=0.2, parallel_rows=8, **converged),
            "owned": _matrix_free_solver(model, beta=0.2, parallel_rows=32, **converged),
            "serial": _matrix_free_solver(model, beta=0.2, parallel_rows=0, **converged),
        }
        for name, rows in (("declined", 8), ("owned", 32), ("serial", 0)):
            self.assertEqual(arms[name].placement.ink_rows, rows, name)
            keys = [tier.kernel.key for tier in arms[name].sweep.tiers]
            self.assertEqual(bool(keys), bool(rows), name)
            # Linked tiers with joint velocity limits take the last word.
            self.assertTrue(all(_law_key(key).endswith("_linked_lw") for key in keys), keys)
        velocities = {}
        for name, solver in arms.items():
            state, out, control = model.state(), model.state(), model.control()
            control.joint_f.assign(torques)
            solver.step(state, out, control, None, 1.0 / 240.0)
            velocities[name] = out.joint_qd.numpy().astype(np.float64)
        # The serial sweep returns early only for worlds of at most 8 dense rows and no matrix-free rows.
        self.assertGreater(int(arms["declined"].placement.constraint_count.numpy()[0]), 8)
        self.assertEqual(int(arms["declined"].placement.mf_constraint_count.numpy()[0]), 0)
        _assert_tier_owned(self, arms["owned"], 32)
        limited = np.abs(velocities["serial"])
        self.assertTrue(np.any(limited >= _HAND_LIMIT * (1.0 - 1.0e-4)), "no limit binds")
        np.testing.assert_allclose(velocities["declined"], velocities["serial"], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(velocities["owned"], velocities["serial"], rtol=0.0, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()

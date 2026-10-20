# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Loop sweeps that update contact units exactly (``contact_update="unit"``).

These are the ``"unit"`` variants of the loop sweeps :func:`~.kernels.pgs_solve_loop`
(assembled Delassus rows) and :func:`~.kernels.pgs_solve_mf_loop` (free-body
rows): one thread sweeps one world in the same forward row order. A contact
unit (:mod:`.contact_unit`) is visited at its first row, where all of its rows
take the unit's exact local solution against the residuals of the same
impulses; its other rows are not visited again in that sweep. The local solve
sees the unit's current impulses and returns them whenever they solve its
problem, so every solution of the whole problem is a fixed point of the
sweeps. Rows outside
units, frictionless point normals, and every row while friction has not
started (``friction_start_iteration``), keep the ``"sequential"`` update of the
original sweep. A unit's exact solution ignores ``pgs_omega``; only its
alternation latch relaxes it.

Units of a patch region:

* ring units (``ring_units=1``): the ring's normals and its anchors' pairs as
  one unit;
* anchor units (``ring_units=0``): each anchor as a 3x3 unit whose ring keeps
  the other members' load, the other members with the scalar update, all at the
  ring's first row in row order.

A ring of more than :data:`~.contact_unit.CONTACT_UNIT_RING_NORMALS` normals or
:data:`~.contact_unit.CONTACT_UNIT_RING_ANCHORS` anchors always takes anchor
units: it never falls back to the sequential update, whose normal steps against
stale friction can end a solve with no normal impulse on an approaching contact.

A unit that holds no impulse, keeps no ring load and does not approach at zero
impulse stays at zero without its tangent residuals (separated early exit),
exactly as the full visit would leave it.

The slide roots and latches are per solve: a launch with ``iteration_offset``
0 starts a solve and clears them, and the one-iteration launches of the
interleaved split driver continue it. ``unit_state[world, row]`` holds the
slide root of the unit or anchor whose normal is ``row`` and the latch (last
load change, factor) of the unit whose first row is ``row``.
"""

import warp as wp

from .contact_unit import (
    CONTACT_UNIT_NEWTON_CAP,
    CONTACT_UNIT_PROXIMAL,
    CONTACT_UNIT_RING_ANCHORS,
    CONTACT_UNIT_RING_NORMALS,
    CONTACT_UNIT_RING_PASSES,
    RING_ALPHA,
    RING_BLOCK,
    RING_BLOCK_SIZE,
    RING_IMPULSE,
    RING_RESIDUAL,
    RING_SOLUTION,
    contact_unit_latch,
    contact_unit_ring,
    contact_unit_solve,
)
from .friction import friction_pair_candidate
from .friction_patches import patch_normal_load
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    friction_step_current,
)

CONTACT_UPDATES = ("sequential", "unit")
"""The contact updates a sweep can apply (``SolverFeatherPGS(contact_update=...)``)."""

CONTACT_UNIT_STATE_FIELDS = 3
"""Per-row unit state: slide root, last applied load change, latch factor."""


def check_contact_update(update: str, *, source: str = "contact_update") -> str:
    """Return ``update`` when it names a contact update.

    Args:
        update: The requested contact update.
        source: The argument or environment variable that supplied ``update``, named in the error.

    Raises:
        ValueError: If ``update`` is not one of :data:`CONTACT_UPDATES`.
    """
    if update not in CONTACT_UPDATES:
        raise ValueError(f"{source} must be one of {CONTACT_UPDATES}, got {update!r}")
    return update


_STATE_ALPHA = wp.constant(0)
_STATE_CHANGE = wp.constant(1)
_STATE_FACTOR = wp.constant(2)
_CAP = wp.constant(CONTACT_UNIT_NEWTON_CAP)
_PASSES = wp.constant(CONTACT_UNIT_RING_PASSES)
_PROXIMAL = wp.constant(CONTACT_UNIT_PROXIMAL)
_RING_NORMALS = wp.constant(CONTACT_UNIT_RING_NORMALS)
_RING_ANCHORS = wp.constant(CONTACT_UNIT_RING_ANCHORS)
_BLOCK = wp.constant(RING_BLOCK)
_RESIDUAL = wp.constant(RING_RESIDUAL)
_IMPULSE = wp.constant(RING_IMPULSE)
_SOLUTION = wp.constant(RING_SOLUTION)
_ALPHA = wp.constant(RING_ALPHA)
_BLOCK_SIZE = wp.constant(RING_BLOCK_SIZE)


@wp.func
def _has_pair(row_type: wp.array2d[int], row_parent: wp.array2d[int], world: int, m: int, n: int) -> bool:
    """Whether normal row ``n`` is an anchor: its tangent pair follows it."""
    return n + 2 < m and row_type[world, n + 1] == PGS_CONSTRAINT_TYPE_FRICTION and row_parent[world, n + 1] == n


@wp.func
def _unit_shape(row_type: wp.array2d[int], row_parent: wp.array2d[int], world: int, m: int, n: int) -> wp.vec3i:
    """Return the normals, the anchors and the first row of the patch ring (or point contact) of normal ``n``."""
    normals = int(1)
    anchors = int(0)
    if _has_pair(row_type, row_parent, world, m, n):
        anchors = 1
    first = n
    row = row_parent[world, n]
    count = int(0)
    while row >= 0 and row != n and count < m:
        normals += 1
        if _has_pair(row_type, row_parent, world, m, row):
            anchors += 1
        if row < first:
            first = row
        row = row_parent[world, row]
        count += 1
    return wp.vec3i(normals, anchors, first)


@wp.func
def _sort_ring(row_parent: wp.array2d[int], unit_rows: wp.array2d[int], world: int, m: int, first: int) -> int:
    """Write the ring of ``first``, its smallest row, in increasing row order to ``unit_rows[world]``."""
    unit_rows[world, 0] = first
    count = int(1)
    row = row_parent[world, first]
    guard = int(0)
    while row >= 0 and row != first and guard < m:
        position = count
        while position > 0 and unit_rows[world, position - 1] > row:
            unit_rows[world, position] = unit_rows[world, position - 1]
            position -= 1
        unit_rows[world, position] = row
        count += 1
        row = row_parent[world, row]
        guard += 1
    return count


@wp.func
def _skip_friction(row_type: wp.array2d[int], row_parent: wp.array2d[int], world: int, m: int, i: int) -> bool:
    """Whether tangent row ``i`` belongs to a unit, which its anchor's visit updates."""
    parent = row_parent[world, i]
    if parent < 0 or parent >= m:
        return False
    if row_type[world, parent] != PGS_CONSTRAINT_TYPE_CONTACT:
        return False
    return _has_pair(row_type, row_parent, world, m, parent)


# -- assembled Delassus rows ------------------------------------------------------------------------------------


@wp.func
def _dense_residual(
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_impulses: wp.array2d[float],
    world: int,
    m: int,
    row: int,
) -> float:
    w = world_rhs[world, row]
    for j in range(m):
        w += world_C[world, row, j] * world_impulses[world, j]
    return w


@wp.func
def _dense_point_unit(
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_impulses: wp.array2d[float],
    world_row_mu: wp.array2d[float],
    unit_state: wp.array3d[float],
    world: int,
    m: int,
    n: int,
    ring_load: float,
    anchor: int,
):
    """Visit the 3x3 unit of normal ``n`` and its pair, a point contact or a patch anchor with ``ring_load`` held.

    A point unit reads its normal-tangent coupling from the normal row, an anchor (``anchor`` 1) from its tangent
    rows; the two agree up to the assembly's rounding.
    """
    t0 = n + 1
    t1 = n + 2
    r_n = _dense_residual(world_C, world_rhs, world_impulses, world, m, n)
    o_n = world_impulses[world, n]
    o_t0 = world_impulses[world, t0]
    o_t1 = world_impulses[world, t1]
    if o_n == 0.0 and o_t0 == 0.0 and o_t1 == 0.0 and ring_load == 0.0 and r_n >= 0.0:
        # Separated: the exact solution is zero, and so is the latch's load change.
        world_impulses[world, n] = 0.0
        world_impulses[world, t0] = 0.0
        world_impulses[world, t1] = 0.0
        unit_state[world, n, _STATE_CHANGE] = 0.0
        return
    w_nn = wp.max(world_diag[world, n], world_C[world, n, n])
    if w_nn <= 0.0:
        return  # A normal without response cannot move, as the sequential update skips it.
    r_t0 = _dense_residual(world_C, world_rhs, world_impulses, world, m, t0)
    r_t1 = _dense_residual(world_C, world_rhs, world_impulses, world, m, t1)
    w_t0 = wp.max(world_diag[world, t0], world_C[world, t0, t0])
    w_t1 = wp.max(world_diag[world, t1], world_C[world, t1, t1])
    # The unit's residuals with its own impulses removed: b = r - W_U lambda_U, row by row.
    b_n = r_n - w_nn * o_n - world_C[world, n, t0] * o_t0 - world_C[world, n, t1] * o_t1
    b_t0 = r_t0 - world_C[world, t0, n] * o_n - w_t0 * o_t0 - world_C[world, t0, t1] * o_t1
    b_t1 = r_t1 - world_C[world, t1, n] * o_n - world_C[world, t1, t0] * o_t0 - w_t1 * o_t1
    w_nt0 = world_C[world, n, t0]
    w_nt1 = world_C[world, n, t1]
    if anchor != 0:
        w_nt0 = world_C[world, t0, n]
        w_nt1 = world_C[world, t1, n]
    solution = contact_unit_solve(
        w_nn,
        w_t0,
        world_C[world, t0, t1],
        w_t1,
        w_nt0,
        w_nt1,
        b_n,
        b_t0,
        b_t1,
        world_row_mu[world, t0],
        ring_load,
        unit_state[world, n, _STATE_ALPHA],
        _CAP,
        o_n,
        o_t0,
        o_t1,
    )
    if solution[3] > 0.0:
        unit_state[world, n, _STATE_ALPHA] = solution[3]
    latch = contact_unit_latch(
        unit_state[world, n, _STATE_FACTOR], unit_state[world, n, _STATE_CHANGE], o_n, solution[0]
    )
    unit_state[world, n, _STATE_FACTOR] = latch[1]
    unit_state[world, n, _STATE_CHANGE] = latch[2]
    weight = latch[0]
    world_impulses[world, n] = o_n + weight * (solution[0] - o_n)
    world_impulses[world, t0] = o_t0 + weight * (solution[1] - o_t0)
    world_impulses[world, t1] = o_t1 + weight * (solution[2] - o_t1)


@wp.func
def _dense_anchor_units(
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_impulses: wp.array2d[float],
    omega: float,
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    unit_state: wp.array3d[float],
    unit_rows: wp.array2d[int],
    world: int,
    m: int,
    normals: int,
):
    """Visit a ring's members in row order: anchors as 3x3 units with the others' load held, the rest scalar."""
    for a in range(normals):
        row = unit_rows[world, a]
        if _has_pair(world_row_type, world_row_parent, world, m, row):
            ring_load = float(0.0)
            for q in range(normals):
                if q != a:
                    ring_load += world_impulses[world, unit_rows[world, q]]
            _dense_point_unit(
                world_diag,
                world_C,
                world_rhs,
                world_impulses,
                world_row_mu,
                unit_state,
                world,
                m,
                row,
                ring_load,
                1,
            )
        else:
            w = _dense_residual(world_C, world_rhs, world_impulses, world, m, row)
            denom = world_diag[world, row]
            if denom > 0.0:
                delta = -w / denom
                new_impulse = world_impulses[world, row] + omega * delta
                if new_impulse < 0.0:
                    new_impulse = 0.0
                world_impulses[world, row] = new_impulse


@wp.func
def _dense_ring_unit(
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_impulses: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    unit_state: wp.array3d[float],
    unit_rows: wp.array2d[int],
    unit_block: wp.array[float],
    world: int,
    m: int,
    normals: int,
):
    """Visit a ring unit: its normals and its anchors' pairs solved together."""
    base = world * _BLOCK_SIZE
    first = unit_rows[world, 0]
    anchors = int(0)
    held = int(0)
    approaching = int(0)
    for a in range(normals):
        row = unit_rows[world, a]
        if _has_pair(world_row_type, world_row_parent, world, m, row):
            unit_rows[world, normals + 2 * anchors] = row + 1
            unit_rows[world, normals + 2 * anchors + 1] = row + 2
            anchors += 1
        r = _dense_residual(world_C, world_rhs, world_impulses, world, m, row)
        unit_block[base + _RESIDUAL + a] = r
        if world_impulses[world, row] != 0.0:
            held = 1
        if r < 0.0:
            approaching = 1
    rows = normals + 2 * anchors
    for a in range(normals, rows):
        if world_impulses[world, unit_rows[world, a]] != 0.0:
            held = 1
    if held == 0 and approaching == 0:
        # Separated: the exact solution is zero, and so is the latch's load change.
        for a in range(rows):
            world_impulses[world, unit_rows[world, a]] = 0.0
        unit_state[world, first, _STATE_CHANGE] = 0.0
        return
    for a in range(normals, rows):
        unit_block[base + _RESIDUAL + a] = _dense_residual(
            world_C, world_rhs, world_impulses, world, m, unit_rows[world, a]
        )
    for a in range(rows):
        row = unit_rows[world, a]
        unit_block[base + _IMPULSE + a] = world_impulses[world, row]
        for c in range(rows):
            value = world_C[world, row, unit_rows[world, c]]
            if c == a:
                value = wp.max(world_diag[world, row], value)
            unit_block[base + _BLOCK + a * rows + c] = value
    for a in range(rows):
        v = unit_block[base + _RESIDUAL + a]
        for c in range(rows):
            v -= unit_block[base + _BLOCK + a * rows + c] * unit_block[base + _IMPULSE + c]
        unit_block[base + _RESIDUAL + a] = v
    mu0 = float(0.0)
    mu1 = float(0.0)
    if anchors >= 1:
        mu0 = world_row_mu[world, unit_rows[world, normals]]
        unit_block[base + _ALPHA] = unit_state[world, unit_rows[world, normals] - 1, _STATE_ALPHA]
    if anchors >= 2:
        mu1 = world_row_mu[world, unit_rows[world, normals + 2]]
        unit_block[base + _ALPHA + 1] = unit_state[world, unit_rows[world, normals + 2] - 1, _STATE_ALPHA]
    contact_unit_ring(unit_block, base, normals, anchors, mu0, mu1, _PASSES, _CAP, _PROXIMAL)
    if anchors >= 1:
        unit_state[world, unit_rows[world, normals] - 1, _STATE_ALPHA] = unit_block[base + _ALPHA]
    if anchors >= 2:
        unit_state[world, unit_rows[world, normals + 2] - 1, _STATE_ALPHA] = unit_block[base + _ALPHA + 1]
    old_load = float(0.0)
    new_load = float(0.0)
    for a in range(normals):
        old_load += unit_block[base + _IMPULSE + a]
        new_load += unit_block[base + _SOLUTION + a]
    latch = contact_unit_latch(
        unit_state[world, first, _STATE_FACTOR], unit_state[world, first, _STATE_CHANGE], old_load, new_load
    )
    unit_state[world, first, _STATE_FACTOR] = latch[1]
    unit_state[world, first, _STATE_CHANGE] = latch[2]
    weight = latch[0]
    for a in range(rows):
        old = unit_block[base + _IMPULSE + a]
        world_impulses[world, unit_rows[world, a]] = old + weight * (unit_block[base + _SOLUTION + a] - old)


@wp.kernel(enable_backward=False)
def pgs_solve_loop_unit(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_impulses: wp.array2d[float],
    iterations: int,
    omega: float,
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    friction_start_iteration: int,
    iteration_offset: int,
    ring_units: int,
    unit_state: wp.array3d[float],
    unit_rows: wp.array2d[int],
    unit_block: wp.array[float],
):
    """World-level projected Gauss-Seidel over assembled Delassus rows with exact contact units.

    The ``"unit"`` variant of :func:`~.kernels.pgs_solve_loop`, with the same
    arguments followed by the unit switches and per-world unit storage:
    ``ring_units`` (1 ring units, 0 anchor units for patch regions),
    ``unit_state`` (``[world, row, 3]``), ``unit_rows`` (``[world, row]``) and
    ``unit_block`` (:data:`~.contact_unit.RING_BLOCK_SIZE` floats per world).
    """
    world = wp.tid()
    m = world_constraint_count[world]

    if m == 0:
        return

    if iteration_offset == 0:
        for i in range(m):
            unit_state[world, i, _STATE_ALPHA] = 0.0
            unit_state[world, i, _STATE_CHANGE] = 0.0
            unit_state[world, i, _STATE_FACTOR] = 0.0

    for it in range(iterations):
        friction = iteration_offset + it >= friction_start_iteration
        for i in range(m):
            row_type = world_row_type[world, i]
            if friction and row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                if _skip_friction(world_row_type, world_row_parent, world, m, i):
                    continue
            if friction and row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                shape = _unit_shape(world_row_type, world_row_parent, world, m, i)
                normals = shape[0]
                anchors = shape[1]
                if normals > 1 or anchors > 0:
                    if i != shape[2]:
                        continue
                    if normals == 1:
                        _dense_point_unit(
                            world_diag,
                            world_C,
                            world_rhs,
                            world_impulses,
                            world_row_mu,
                            unit_state,
                            world,
                            m,
                            i,
                            0.0,
                            0,
                        )
                    else:
                        _sort_ring(world_row_parent, unit_rows, world, m, i)
                        if ring_units != 0 and normals <= _RING_NORMALS and anchors <= _RING_ANCHORS:
                            _dense_ring_unit(
                                world_diag,
                                world_C,
                                world_rhs,
                                world_impulses,
                                world_row_type,
                                world_row_parent,
                                world_row_mu,
                                unit_state,
                                unit_rows,
                                unit_block,
                                world,
                                m,
                                normals,
                            )
                        else:
                            _dense_anchor_units(
                                world_diag,
                                world_C,
                                world_rhs,
                                world_impulses,
                                omega,
                                world_row_type,
                                world_row_parent,
                                world_row_mu,
                                unit_state,
                                unit_rows,
                                world,
                                m,
                                normals,
                            )
                    continue

            # Rows outside units: the sequential update of pgs_solve_loop.
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and iteration_offset + it < friction_start_iteration:
                world_impulses[world, i] = 0.0
                continue
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and i != world_row_parent[world, i] + 1:
                continue

            w = world_rhs[world, i]
            for j in range(m):
                w += world_C[world, i, j] * world_impulses[world, j]

            denom = world_diag[world, i]
            if denom <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
                continue

            delta = float(0.0)
            if denom > 0.0:
                delta = -w / denom
            new_impulse = world_impulses[world, i] + omega * delta

            if (
                row_type == PGS_CONSTRAINT_TYPE_CONTACT
                or row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                or row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
            ):
                if new_impulse < 0.0:
                    new_impulse = 0.0
                world_impulses[world, i] = new_impulse

            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = world_row_parent[world, i]
                lambda_n = patch_normal_load(world_row_parent, world_impulses, world, parent_idx)
                mu = world_row_mu[world, i]
                radius = wp.max(mu * lambda_n, 0.0)

                if radius <= 0.0:
                    world_impulses[world, i] = 0.0
                    world_impulses[world, i + 1] = 0.0
                    continue

                if i != parent_idx + 1:
                    continue
                sib = parent_idx + 2
                sibling_residual = world_rhs[world, sib]
                for j in range(m):
                    sibling_residual += world_C[world, sib, j] * world_impulses[world, j]

                trial = friction_pair_candidate(
                    denom,
                    world_C[world, i, sib],
                    world_diag[world, sib],
                    wp.vec2(w, sibling_residual),
                    wp.vec2(world_impulses[world, i], world_impulses[world, sib]),
                    radius,
                    omega,
                )
                a = trial[0]
                b = trial[1]

                mag = wp.sqrt(a * a + b * b)
                if mag > radius:
                    scale = radius / mag
                    a *= scale
                    b *= scale
                world_impulses[world, i] = a
                world_impulses[world, sib] = b

            else:
                world_impulses[world, i] = new_impulse


# -- free-body (matrix-free) rows --------------------------------------------------------------------------------


@wp.func
def _mf_residual(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_rhs: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    v_out: wp.array[float],
    world: int,
    row: int,
) -> float:
    jv = float(0.0)
    ba = mf_body_a[world, row]
    bb = mf_body_b[world, row]
    if ba >= 0:
        ds_a = art_dof_start[body_to_articulation[ba]]
        for k in range(6):
            jv += mf_J_a[world, row, k] * v_out[ds_a + k]
    if bb >= 0:
        ds_b = art_dof_start[body_to_articulation[bb]]
        for k in range(6):
            jv += mf_J_b[world, row, k] * v_out[ds_b + k]
    return jv + mf_rhs[world, row]


@wp.func
def _mf_coupling(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    world: int,
    i: int,
    j: int,
) -> float:
    """Delassus entry ``J_i M^-1 J_j^T`` over the bodies rows ``i`` and ``j`` share."""
    c = float(0.0)
    ai = mf_body_a[world, i]
    bi = mf_body_b[world, i]
    aj = mf_body_a[world, j]
    bj = mf_body_b[world, j]
    if ai >= 0:
        if ai == aj:
            for k in range(6):
                c += mf_J_a[world, i, k] * mf_MiJt_a[world, j, k]
        if ai == bj:
            for k in range(6):
                c += mf_J_a[world, i, k] * mf_MiJt_b[world, j, k]
    if bi >= 0:
        if bi == aj:
            for k in range(6):
                c += mf_J_b[world, i, k] * mf_MiJt_a[world, j, k]
        if bi == bj:
            for k in range(6):
                c += mf_J_b[world, i, k] * mf_MiJt_b[world, j, k]
    return c


@wp.func
def _mf_diagonal(mf_eff_mass_inv: wp.array2d[float], world: int, row: int) -> float:
    """The row's sweep diagonal, CFM included; 0 for a row without response."""
    eff_inv = mf_eff_mass_inv[world, row]
    if eff_inv > 0.0:
        return 1.0 / eff_inv
    return 0.0


@wp.func
def _mf_apply(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    world: int,
    row: int,
    delta: float,
    v_out: wp.array[float],
):
    ba = mf_body_a[world, row]
    bb = mf_body_b[world, row]
    if ba >= 0:
        ds_a = art_dof_start[body_to_articulation[ba]]
        for k in range(6):
            v_out[ds_a + k] = v_out[ds_a + k] + mf_MiJt_a[world, row, k] * delta
    if bb >= 0:
        ds_b = art_dof_start[body_to_articulation[bb]]
        for k in range(6):
            v_out[ds_b + k] = v_out[ds_b + k] + mf_MiJt_b[world, row, k] * delta


@wp.func
def _mf_normal_weight(mf_row_w: wp.array2d[float], regularize: int, world: int, row: int) -> float:
    """The contact regularization weight: the unit solves its normal against the diagonal over this weight."""
    if regularize != 0:
        return mf_row_w[world, row]
    return 1.0


@wp.func
def _mf_point_unit(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_mu: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    regularize: int,
    unit_state: wp.array3d[float],
    world: int,
    n: int,
    ring_load: float,
    anchor: int,
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Visit the 3x3 unit of free-body normal ``n`` and its pair (see :func:`_dense_point_unit`)."""
    t0 = n + 1
    t1 = n + 2
    r_n = _mf_residual(
        mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_rhs, body_to_articulation, art_dof_start, v_out, world, n
    )
    o_n = mf_impulses[world, n]
    o_t0 = mf_impulses[world, t0]
    o_t1 = mf_impulses[world, t1]
    if o_n == 0.0 and o_t0 == 0.0 and o_t1 == 0.0 and ring_load == 0.0 and r_n >= 0.0:
        mf_impulses[world, n] = 0.0
        mf_impulses[world, t0] = 0.0
        mf_impulses[world, t1] = 0.0
        unit_state[world, n, _STATE_CHANGE] = 0.0
        return
    d_n = _mf_diagonal(mf_eff_mass_inv, world, n)
    if d_n <= 0.0:
        return
    r_t0 = _mf_residual(
        mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_rhs, body_to_articulation, art_dof_start, v_out, world, t0
    )
    r_t1 = _mf_residual(
        mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_rhs, body_to_articulation, art_dof_start, v_out, world, t1
    )
    d_t0 = _mf_diagonal(mf_eff_mass_inv, world, t0)
    d_t1 = _mf_diagonal(mf_eff_mass_inv, world, t1)
    c_nt0 = _mf_coupling(mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, world, n, t0)
    c_nt1 = _mf_coupling(mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, world, n, t1)
    c_t0n = _mf_coupling(mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, world, t0, n)
    c_t0t1 = _mf_coupling(mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, world, t0, t1)
    c_t1n = _mf_coupling(mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, world, t1, n)
    c_t1t0 = _mf_coupling(mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, world, t1, t0)
    b_n = r_n - d_n * o_n - c_nt0 * o_t0 - c_nt1 * o_t1
    b_t0 = r_t0 - c_t0n * o_n - d_t0 * o_t0 - c_t0t1 * o_t1
    b_t1 = r_t1 - c_t1n * o_n - c_t1t0 * o_t0 - d_t1 * o_t1
    w_nt0 = c_nt0
    w_nt1 = c_nt1
    if anchor != 0:
        w_nt0 = c_t0n
        w_nt1 = c_t1n
    solution = contact_unit_solve(
        d_n / _mf_normal_weight(mf_row_w, regularize, world, n),
        d_t0,
        c_t0t1,
        d_t1,
        w_nt0,
        w_nt1,
        b_n,
        b_t0,
        b_t1,
        mf_row_mu[world, t0],
        ring_load,
        unit_state[world, n, _STATE_ALPHA],
        _CAP,
        o_n,
        o_t0,
        o_t1,
    )
    if solution[3] > 0.0:
        unit_state[world, n, _STATE_ALPHA] = solution[3]
    latch = contact_unit_latch(
        unit_state[world, n, _STATE_FACTOR], unit_state[world, n, _STATE_CHANGE], o_n, solution[0]
    )
    unit_state[world, n, _STATE_FACTOR] = latch[1]
    unit_state[world, n, _STATE_CHANGE] = latch[2]
    weight = latch[0]
    new_n = o_n + weight * (solution[0] - o_n)
    new_t0 = o_t0 + weight * (solution[1] - o_t0)
    new_t1 = o_t1 + weight * (solution[2] - o_t1)
    mf_impulses[world, n] = new_n
    mf_impulses[world, t0] = new_t0
    mf_impulses[world, t1] = new_t1
    _mf_apply(
        mf_body_a, mf_body_b, mf_MiJt_a, mf_MiJt_b, body_to_articulation, art_dof_start, world, n, new_n - o_n, v_out
    )
    _mf_apply(
        mf_body_a, mf_body_b, mf_MiJt_a, mf_MiJt_b, body_to_articulation, art_dof_start, world, t0, new_t0 - o_t0, v_out
    )
    _mf_apply(
        mf_body_a, mf_body_b, mf_MiJt_a, mf_MiJt_b, body_to_articulation, art_dof_start, world, t1, new_t1 - o_t1, v_out
    )


@wp.func
def _mf_scalar_normal(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    omega: float,
    regularize: int,
    world: int,
    i: int,
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """The sequential update of one free-body contact normal row, as :func:`~.kernels.pgs_solve_mf_loop` makes it."""
    eff_inv = mf_eff_mass_inv[world, i]
    if eff_inv <= 0.0:
        return
    residual = _mf_residual(
        mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_rhs, body_to_articulation, art_dof_start, v_out, world, i
    )
    old_impulse = mf_impulses[world, i]
    w = float(1.0)
    if regularize != 0:
        w = mf_row_w[world, i]
    delta = -residual * eff_inv * w - (1.0 - w) * old_impulse
    new_impulse = old_impulse + omega * delta
    if new_impulse < 0.0:
        new_impulse = 0.0
    mf_impulses[world, i] = new_impulse
    _mf_apply(
        mf_body_a,
        mf_body_b,
        mf_MiJt_a,
        mf_MiJt_b,
        body_to_articulation,
        art_dof_start,
        world,
        i,
        new_impulse - old_impulse,
        v_out,
    )


@wp.func
def _mf_anchor_units(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    omega: float,
    regularize: int,
    unit_state: wp.array3d[float],
    unit_rows: wp.array2d[int],
    world: int,
    m: int,
    normals: int,
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Free-body anchor units of a ring (see :func:`_dense_anchor_units`)."""
    for a in range(normals):
        row = unit_rows[world, a]
        if _has_pair(mf_row_type, mf_row_parent, world, m, row):
            ring_load = float(0.0)
            for q in range(normals):
                if q != a:
                    ring_load += mf_impulses[world, unit_rows[world, q]]
            _mf_point_unit(
                mf_body_a,
                mf_body_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_J_a,
                mf_J_b,
                mf_eff_mass_inv,
                mf_rhs,
                mf_row_mu,
                mf_row_w,
                body_to_articulation,
                art_dof_start,
                regularize,
                unit_state,
                world,
                row,
                ring_load,
                1,
                mf_impulses,
                v_out,
            )
        else:
            _mf_scalar_normal(
                mf_body_a,
                mf_body_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_J_a,
                mf_J_b,
                mf_eff_mass_inv,
                mf_rhs,
                mf_row_w,
                body_to_articulation,
                art_dof_start,
                omega,
                regularize,
                world,
                row,
                mf_impulses,
                v_out,
            )


@wp.func
def _mf_ring_unit(
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    regularize: int,
    unit_state: wp.array3d[float],
    unit_rows: wp.array2d[int],
    unit_block: wp.array[float],
    world: int,
    m: int,
    normals: int,
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """A free-body ring unit (see :func:`_dense_ring_unit`)."""
    base = world * _BLOCK_SIZE
    first = unit_rows[world, 0]
    anchors = int(0)
    held = int(0)
    approaching = int(0)
    for a in range(normals):
        row = unit_rows[world, a]
        if _has_pair(mf_row_type, mf_row_parent, world, m, row):
            unit_rows[world, normals + 2 * anchors] = row + 1
            unit_rows[world, normals + 2 * anchors + 1] = row + 2
            anchors += 1
        r = _mf_residual(
            mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_rhs, body_to_articulation, art_dof_start, v_out, world, row
        )
        unit_block[base + _RESIDUAL + a] = r
        if mf_impulses[world, row] != 0.0:
            held = 1
        if r < 0.0:
            approaching = 1
    rows = normals + 2 * anchors
    for a in range(normals, rows):
        if mf_impulses[world, unit_rows[world, a]] != 0.0:
            held = 1
    if held == 0 and approaching == 0:
        for a in range(rows):
            mf_impulses[world, unit_rows[world, a]] = 0.0
        unit_state[world, first, _STATE_CHANGE] = 0.0
        return
    for a in range(normals, rows):
        unit_block[base + _RESIDUAL + a] = _mf_residual(
            mf_body_a,
            mf_body_b,
            mf_J_a,
            mf_J_b,
            mf_rhs,
            body_to_articulation,
            art_dof_start,
            v_out,
            world,
            unit_rows[world, a],
        )
    # b = r - W_U lambda_U with the unregularized block, which the regularized one leaves unchanged.
    for a in range(rows):
        row = unit_rows[world, a]
        unit_block[base + _IMPULSE + a] = mf_impulses[world, row]
        for c in range(rows):
            value = float(0.0)
            if c == a:
                value = _mf_diagonal(mf_eff_mass_inv, world, row)
            else:
                value = _mf_coupling(
                    mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, world, row, unit_rows[world, c]
                )
            unit_block[base + _BLOCK + a * rows + c] = value
    for a in range(rows):
        v = unit_block[base + _RESIDUAL + a]
        for c in range(rows):
            v -= unit_block[base + _BLOCK + a * rows + c] * unit_block[base + _IMPULSE + c]
        unit_block[base + _RESIDUAL + a] = v
    for a in range(normals):
        diagonal = base + _BLOCK + a * rows + a
        unit_block[diagonal] = unit_block[diagonal] / _mf_normal_weight(
            mf_row_w, regularize, world, unit_rows[world, a]
        )
    mu0 = float(0.0)
    mu1 = float(0.0)
    if anchors >= 1:
        mu0 = mf_row_mu[world, unit_rows[world, normals]]
        unit_block[base + _ALPHA] = unit_state[world, unit_rows[world, normals] - 1, _STATE_ALPHA]
    if anchors >= 2:
        mu1 = mf_row_mu[world, unit_rows[world, normals + 2]]
        unit_block[base + _ALPHA + 1] = unit_state[world, unit_rows[world, normals + 2] - 1, _STATE_ALPHA]
    contact_unit_ring(unit_block, base, normals, anchors, mu0, mu1, _PASSES, _CAP, _PROXIMAL)
    if anchors >= 1:
        unit_state[world, unit_rows[world, normals] - 1, _STATE_ALPHA] = unit_block[base + _ALPHA]
    if anchors >= 2:
        unit_state[world, unit_rows[world, normals + 2] - 1, _STATE_ALPHA] = unit_block[base + _ALPHA + 1]
    old_load = float(0.0)
    new_load = float(0.0)
    for a in range(normals):
        old_load += unit_block[base + _IMPULSE + a]
        new_load += unit_block[base + _SOLUTION + a]
    latch = contact_unit_latch(
        unit_state[world, first, _STATE_FACTOR], unit_state[world, first, _STATE_CHANGE], old_load, new_load
    )
    unit_state[world, first, _STATE_FACTOR] = latch[1]
    unit_state[world, first, _STATE_CHANGE] = latch[2]
    weight = latch[0]
    for a in range(rows):
        row = unit_rows[world, a]
        old = unit_block[base + _IMPULSE + a]
        new = old + weight * (unit_block[base + _SOLUTION + a] - old)
        mf_impulses[world, row] = new
        _mf_apply(
            mf_body_a,
            mf_body_b,
            mf_MiJt_a,
            mf_MiJt_b,
            body_to_articulation,
            art_dof_start,
            world,
            row,
            new - old,
            v_out,
        )


@wp.kernel(enable_backward=False)
def pgs_solve_mf_loop_unit(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    iterations: int,
    omega: float,
    regularize: int,
    friction_start_iteration: int,
    iteration_offset: int,
    ring_units: int,
    unit_state: wp.array3d[float],
    unit_rows: wp.array2d[int],
    unit_block: wp.array[float],
    # in/out
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Matrix-free projected Gauss-Seidel over free-body rows with exact contact units.

    The ``"unit"`` variant of :func:`~.kernels.pgs_solve_mf_loop` under the
    ``"current"`` friction mode, its only mode on the CPU split route, with
    the unit switches and storage of :func:`pgs_solve_loop_unit`. A unit reads
    its residuals from ``J v``, its block from ``J M^-1 J^T`` and its diagonal
    from ``mf_eff_mass_inv``, and applies each changed impulse to ``v_out``.
    Regularized contact normals solve against their diagonal over the row
    weight, the same law as the w-form update.
    """
    world = wp.tid()
    m_count = mf_constraint_count[world]
    if m_count == 0:
        return

    if iteration_offset == 0:
        for i in range(m_count):
            unit_state[world, i, _STATE_ALPHA] = 0.0
            unit_state[world, i, _STATE_CHANGE] = 0.0
            unit_state[world, i, _STATE_FACTOR] = 0.0

    for it in range(iterations):
        friction = iteration_offset + it >= friction_start_iteration
        for i in range(m_count):
            row_type = mf_row_type[world, i]
            if friction and row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                if _skip_friction(mf_row_type, mf_row_parent, world, m_count, i):
                    continue
            if friction and row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                shape = _unit_shape(mf_row_type, mf_row_parent, world, m_count, i)
                normals = shape[0]
                anchors = shape[1]
                if normals > 1 or anchors > 0:
                    if i != shape[2]:
                        continue
                    if normals == 1:
                        _mf_point_unit(
                            mf_body_a,
                            mf_body_b,
                            mf_MiJt_a,
                            mf_MiJt_b,
                            mf_J_a,
                            mf_J_b,
                            mf_eff_mass_inv,
                            mf_rhs,
                            mf_row_mu,
                            mf_row_w,
                            body_to_articulation,
                            art_dof_start,
                            regularize,
                            unit_state,
                            world,
                            i,
                            0.0,
                            0,
                            mf_impulses,
                            v_out,
                        )
                    else:
                        _sort_ring(mf_row_parent, unit_rows, world, m_count, i)
                        if ring_units != 0 and normals <= _RING_NORMALS and anchors <= _RING_ANCHORS:
                            _mf_ring_unit(
                                mf_body_a,
                                mf_body_b,
                                mf_MiJt_a,
                                mf_MiJt_b,
                                mf_J_a,
                                mf_J_b,
                                mf_eff_mass_inv,
                                mf_rhs,
                                mf_row_type,
                                mf_row_parent,
                                mf_row_mu,
                                mf_row_w,
                                body_to_articulation,
                                art_dof_start,
                                regularize,
                                unit_state,
                                unit_rows,
                                unit_block,
                                world,
                                m_count,
                                normals,
                                mf_impulses,
                                v_out,
                            )
                        else:
                            _mf_anchor_units(
                                mf_body_a,
                                mf_body_b,
                                mf_MiJt_a,
                                mf_MiJt_b,
                                mf_J_a,
                                mf_J_b,
                                mf_eff_mass_inv,
                                mf_rhs,
                                mf_row_type,
                                mf_row_parent,
                                mf_row_mu,
                                mf_row_w,
                                body_to_articulation,
                                art_dof_start,
                                omega,
                                regularize,
                                unit_state,
                                unit_rows,
                                world,
                                m_count,
                                normals,
                                mf_impulses,
                                v_out,
                            )
                    continue

            # Rows outside units: the sequential update of pgs_solve_mf_loop (friction mode "current").
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and iteration_offset + it < friction_start_iteration:
                mf_impulses[world, i] = 0.0
                continue

            eff_inv = mf_eff_mass_inv[world, i]
            if eff_inv <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
                continue

            ba = mf_body_a[world, i]
            bb = mf_body_b[world, i]

            jv = float(0.0)
            if ba >= 0:
                art_a = body_to_articulation[ba]
                ds_a = art_dof_start[art_a]
                for k in range(6):
                    jv += mf_J_a[world, i, k] * v_out[ds_a + k]
            if bb >= 0:
                art_b = body_to_articulation[bb]
                ds_b = art_dof_start[art_b]
                for k in range(6):
                    jv += mf_J_b[world, i, k] * v_out[ds_b + k]

            residual = jv + mf_rhs[world, i]
            old_impulse = mf_impulses[world, i]
            delta = -residual * eff_inv
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                w = float(1.0)
                if regularize != 0:
                    w = mf_row_w[world, i]
                delta = -residual * eff_inv * w - (1.0 - w) * old_impulse
            new_impulse = old_impulse + omega * delta
            delta_impulse = float(0.0)

            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
            elif row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                if residual < 0.0:
                    delta_impulse = delta
                    new_impulse = delta
                else:
                    new_impulse = 0.0
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                new_impulse = friction_step_current(
                    world,
                    i,
                    omega,
                    mf_eff_mass_inv,
                    mf_rhs,
                    mf_J_a,
                    mf_J_b,
                    mf_body_a,
                    mf_body_b,
                    mf_MiJt_a,
                    mf_MiJt_b,
                    mf_row_parent,
                    mf_row_mu,
                    body_to_articulation,
                    art_dof_start,
                    mf_impulses,
                    v_out,
                )

            if row_type != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                delta_impulse = new_impulse - old_impulse
            mf_impulses[world, i] = new_impulse

            if ba >= 0:
                art_a2 = body_to_articulation[ba]
                ds_a2 = art_dof_start[art_a2]
                for k in range(6):
                    v_out[ds_a2 + k] = v_out[ds_a2 + k] + mf_MiJt_a[world, i, k] * delta_impulse
            if bb >= 0:
                art_b2 = body_to_articulation[bb]
                ds_b2 = art_dof_start[art_b2]
                for k in range(6):
                    v_out[ds_b2 + k] = v_out[ds_b2 + k] + mf_MiJt_b[world, i, k] * delta_impulse

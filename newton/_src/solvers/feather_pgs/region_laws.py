# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The region law: a contact region solved as one block in its frame's six coordinates.

A *region* is one friction patch of one body pair. It has one frame at its
anchor: a normal ``n`` and a tangent pair ``t0``, ``t1``. The pair's relative
twist at the anchor, in that frame, splits three and three:

* ``N = (v_n, w_t0, w_t1)``: the closing speed at the anchor and the two
  tilts. Every contact of the region, a *candidate*, is a row over ``N``:
  ``r_i = v_n + a_i1 w_t0 + a_i2 w_t1 + b_i (+ eps_i lambda_i)`` with
  ``(a_i1, a_i2) = ((d_i x n) . t0, (d_i x n) . t1)`` for its point ``d_i``
  from the anchor. A candidate's record is ``(a_i1, a_i2, b_i, eps_i)``, its
  compliance ``eps_i`` [1/kg] as the listing forms it (0: rigid).
* ``F = (v_t0, v_t1, w_n)``: the two slips at the anchor and the spin, the
  region's friction unit, on the cone ``|(f_0 / mu, f_1 / mu, f_2 / mu_t)| <=
  load`` at the region's load, the sum of its candidates' impulses.

The pair's response in those coordinates is one symmetric 6 x 6 matrix
(:func:`frame_response`, summed over the bodies that move): ``Wnn`` over
``N``, ``G`` over ``F`` and their coupling ``C``. The visit never sees the
support the bodies' coordinates live on: a caller gathers the pair's twist
into the frame (:func:`frame_twist`), calls :func:`region_visit`, and
scatters the change of the region's wrench (:func:`frame_wrench`). So one
visit serves a free body against an immovable partner, two movable bodies,
and a body at a tree's endpoint.

The visit (``newton/tests/feather_pgs_region_reference.py`` is its float64
reference): the region's own wrench leaves the velocity; if no candidate
closes there the region holds nothing (*open*); else *stick first*: the
candidates' block with the friction unit held at zero residual, which stands
when the friction that holds the unit lies in its cone at the block's load;
else the region *slides*: the candidates' block with the friction as it was,
then the unit's cone law at the block's load, once.

The candidates' block (:func:`region_block`) is exact for any mix of rigid
and compliant candidates. Rigid candidates (``eps = 0``) are a projection
onto their polyhedron in the block's metric: a dual active set of at most
three rows, three coordinates holding no more independent ones. A compliant
candidate (``eps > 0``) holds ``-(r_i) / eps_i`` wherever it closes: the
closing ones change the block's metric and start (Newton on the three
coordinates, one sum over the candidates a step), and the rigid set is
solved in that metric.
"""

import warp as wp

from .law_source import LawFamily
from .projection_laws import BILATERAL_PIVOT, CONE_LAW, friction_triple, friction_triple_from_metric

REGION_LAW = LawFamily("RegionLaw", requires=(CONE_LAW,))
"""The region law's family; native code calls ``RegionLaw::region_visit(...)``."""

REGION_OPEN = wp.constant(0)
"""A visit's outcome: the region holds nothing."""
REGION_STICK = wp.constant(1)
"""A visit's outcome: the friction unit is held inside its cone (settled)."""
REGION_SLIDE = wp.constant(2)
"""A visit's outcome: the friction unit is on its cone (visited again next sweep)."""

REGION_STEPS = wp.constant(16)
"""Steps a block takes at most: its active-set steps and its Newton steps together."""

_NONE = wp.constant(1.0e30)


@REGION_LAW.law
def metric_factor(K: wp.mat33) -> wp.mat33:
    """The block's metric ``K = L L^T`` (Cholesky, lower ``L``): a row ``a`` whitens to ``z = L^T a``, and the metric's
    products are Euclidean ones of the whitened rows (``a . K b = z_a . z_b``)."""
    l00 = wp.sqrt(wp.max(K[0, 0], 0.0))
    i00 = 1.0 / wp.max(l00, 1.0e-30)
    l10 = K[1, 0] * i00
    l20 = K[2, 0] * i00
    l11 = wp.sqrt(wp.max(K[1, 1] - l10 * l10, 0.0))
    i11 = 1.0 / wp.max(l11, 1.0e-30)
    l21 = (K[2, 1] - l20 * l10) * i11
    l22 = wp.sqrt(wp.max(K[2, 2] - l20 * l20 - l21 * l21, 0.0))
    return wp.mat33(l00, 0.0, 0.0, l10, l11, 0.0, l20, l21, l22)


@REGION_LAW.law
def set_residual(x: wp.vec3, e0: wp.vec3, e1: wp.vec3) -> tuple[wp.vec3, float, float]:
    """What the orthonormal ``e0``, ``e1`` (zero where unused) leave of the whitened ``x``, and ``x``'s coordinates
    along them: Gram-Schmidt taken twice, so the residual is ``x``'s to rounding however close ``x`` lies to their
    span."""
    c0 = wp.dot(x, e0)
    c1 = wp.dot(x, e1)
    v = x - e0 * c0 - e1 * c1
    d0 = wp.dot(v, e0)
    d1 = wp.dot(v, e1)
    return v - e0 * d0 - e1 * d1, c0 + d0, c1 + d1


@REGION_LAW.law
def set_basis(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, L: wp.mat33, n: int) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.mat33]:
    """The set's ``n`` rows whitened by the metric's factor ``L`` (:func:`metric_factor`) and made orthonormal:
    ``L^T a_k = sum_l R[k, l] e_l`` with ``R`` lower triangular (an unused row: ``e`` zero, ``R``'s row the
    identity's), the factor holding each diagonal entry as its reciprocal (the solves multiply, the one division a
    row is its normalization's). So ``A K A^T = R R^T`` with ``R`` formed from residuals, never from differences of
    the rows' Gram entries: rows that lie close together keep their geometry in float32, which the normal equations lose (a part's
    tilt response is 1e4 times its linear one)."""
    zero = wp.vec3(0.0, 0.0, 0.0)
    Lt = wp.transpose(L)
    z0 = Lt * a0
    i00 = 1.0 / wp.max(wp.length(z0), 1.0e-30)
    e0 = wp.where(n >= 1, z0 * i00, zero)
    v1, c10, _c1z = set_residual(Lt * a1, e0, zero)
    i11 = 1.0 / wp.max(wp.length(v1), 1.0e-30)
    e1 = wp.where(n >= 2, v1 * i11, zero)
    v2, c20, c21 = set_residual(Lt * a2, e0, e1)
    i22 = 1.0 / wp.max(wp.length(v2), 1.0e-30)
    e2 = wp.where(n >= 3, v2 * i22, zero)
    R = wp.mat33(
        wp.where(n >= 1, i00, 1.0),
        0.0,
        0.0,
        wp.where(n >= 2, c10, 0.0),
        wp.where(n >= 2, i11, 1.0),
        0.0,
        wp.where(n >= 3, c20, 0.0),
        wp.where(n >= 3, c21, 0.0),
        wp.where(n >= 3, i22, 1.0),
    )
    return e0, e1, e2, R


@REGION_LAW.law
def set_lower(R: wp.mat33, rhs: wp.vec3) -> wp.vec3:
    """``R^-1 rhs`` for the lower triangular ``R`` of :func:`set_basis` (its diagonal held as reciprocals)."""
    y0 = rhs[0] * R[0, 0]
    y1 = (rhs[1] - R[1, 0] * y0) * R[1, 1]
    y2 = (rhs[2] - R[2, 0] * y0 - R[2, 1] * y1) * R[2, 2]
    return wp.vec3(y0, y1, y2)


@REGION_LAW.law
def set_upper(R: wp.mat33, y: wp.vec3) -> wp.vec3:
    """``R^-T y`` for the lower triangular ``R`` of :func:`set_basis` (its diagonal held as reciprocals)."""
    x2 = y[2] * R[2, 2]
    x1 = (y[1] - R[2, 1] * x2) * R[1, 1]
    x0 = (y[0] - R[1, 0] * x1 - R[2, 0] * x2) * R[0, 0]
    return wp.vec3(x0, x1, x2)


@REGION_LAW.law
def set_solve(R: wp.mat33, n: int, rhs: wp.vec3) -> wp.vec3:
    """``(A K A^T)^-1 rhs = R^-T R^-1 rhs`` over the set's ``n`` rows, ``R`` the set's factor (:func:`set_basis`; the
    unused rows' multipliers zero)."""
    kept = wp.vec3(wp.where(n >= 1, rhs[0], 0.0), wp.where(n >= 2, rhs[1], 0.0), wp.where(n >= 3, rhs[2], 0.0))
    return set_upper(R, set_lower(R, kept))


@REGION_LAW.law
def set_start(
    start: wp.vec3,
    L: wp.mat33,
    a0_in: wp.vec3,
    a1_in: wp.vec3,
    a2_in: wp.vec3,
    ab_in: wp.vec3,
    i0_in: int,
    i1_in: int,
    i2_in: int,
    n_in: int,
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3, int, int, int, wp.vec3, int, wp.vec3, wp.vec3, wp.vec3, wp.mat33]:
    """The given set's multipliers at ``start`` in the metric ``K = L L^T``; a row whose multiplier is not positive
    leaves the set. Returns the set (its rows, their right-hand sides, their candidates), its multipliers, its size
    and its factor (:func:`set_basis`, made once and again only after a row leaves)."""
    a0 = a0_in
    a1 = a1_in
    a2 = a2_in
    ab = ab_in
    i0 = i0_in
    i1 = i1_in
    i2 = i2_in
    n = n_in
    zero = wp.vec3(0.0, 0.0, 0.0)
    mu = zero
    e0, e1, e2, R = set_basis(a0, a1, a2, L, n)
    turn = int(0)
    while n > 0 and turn < 4:
        turn += 1
        residual = wp.vec3(wp.dot(a0, start) + ab[0], wp.dot(a1, start) + ab[1], wp.dot(a2, start) + ab[2])
        mu = set_solve(R, n, zero - residual)
        worst = float(0.0)
        leaving = int(-1)
        for k in range(3):
            if k < n and mu[k] <= worst:
                worst = mu[k]
                leaving = k
        if leaving < 0:
            break
        a0 = wp.where(leaving == 0, a1, a0)
        a1 = wp.where(leaving <= 1, a2, a1)
        a2 = wp.vec3(0.0, 0.0, 0.0)
        ab = wp.vec3(wp.where(leaving == 0, ab[1], ab[0]), wp.where(leaving <= 1, ab[2], ab[1]), 0.0)
        i0 = wp.where(leaving == 0, i1, i0)
        i1 = wp.where(leaving <= 1, i2, i1)
        i2 = -1
        n -= 1
        e0, e1, e2, R = set_basis(a0, a1, a2, L, n)
    if n == 0:
        mu = wp.vec3(0.0, 0.0, 0.0)
    return a0, a1, a2, ab, i0, i1, i2, mu, n, e0, e1, e2, R


@REGION_LAW.law
def set_enter(
    s_in: wp.vec3,
    L: wp.mat33,
    aj: wp.vec3,
    bj: float,
    j: int,
    a0_in: wp.vec3,
    a1_in: wp.vec3,
    a2_in: wp.vec3,
    ab_in: wp.vec3,
    i0_in: int,
    i1_in: int,
    i2_in: int,
    mu_in: wp.vec3,
    n_in: int,
    e0_in: wp.vec3,
    e1_in: wp.vec3,
    e2_in: wp.vec3,
    R_in: wp.mat33,
) -> tuple[
    wp.vec3, wp.vec3, wp.vec3, wp.vec3, int, int, int, wp.vec3, int, wp.vec3, int, wp.vec3, wp.vec3, wp.vec3, wp.mat33
]:
    """One step of the dual active set (Goldfarb and Idnani) in the metric ``K = L L^T``: candidate ``j`` (its row ``aj``, its
    right-hand side ``bj``), which closes at ``s_in``, enters the set; rows whose multiplier reaches zero on the way
    leave it. A candidate depends on the set when what the set leaves of its row falls to
    :data:`.projection_laws.BILATERAL_PIVOT` of the row's own squared norm (the bilateral block's rank rule): it
    then enters only as a row leaves. The set's factor (:func:`set_basis`) comes in with it and goes out with it: an
    entering row's remainder is its new column, and the factor is made again only after a row leaves. Returns the
    set, its multipliers, its size, the velocity, whether the step ended (1; it does not where the rows admit no
    velocity) and the set's factor."""
    s = s_in
    a0 = a0_in
    a1 = a1_in
    a2 = a2_in
    ab = ab_in
    i0 = i0_in
    i1 = i1_in
    i2 = i2_in
    mu = mu_in
    n = n_in
    e0 = e0_in
    e1 = e1_in
    e2 = e2_in
    R = R_in
    ended = int(0)
    zj = wp.transpose(L) * aj
    own = wp.dot(zj, zj)
    entering = float(0.0)
    turn = int(0)
    while turn < 5:
        turn += 1
        # What the set leaves of the candidate's whitened row, by the set's whitened rows made orthonormal
        # (set_basis): its square is the rank rule's measure and L times it the velocity's direction (K times the
        # row's remainder); its coordinates along them give the multipliers' change d = (A K A^T)^-1 A K aj = R^-T c.
        part, c0, c1 = set_residual(zj, e0, e1)
        rest, c2, _cz = set_residual(part, e2, wp.vec3(0.0, 0.0, 0.0))
        d = set_upper(R, wp.vec3(wp.where(n >= 1, c0, 0.0), wp.where(n >= 2, c1, 0.0), wp.where(n >= 3, c2, 0.0)))
        direction = L * rest
        along = wp.dot(rest, rest)
        primal = float(_NONE)
        if n < 3 and along > BILATERAL_PIVOT * own:
            primal = -(wp.dot(aj, s) + bj) / along
        dual = float(_NONE)
        leaving = int(-1)
        for k in range(3):
            if k < n and d[k] > 0.0:
                ratio = mu[k] / d[k]
                if ratio < dual:
                    dual = ratio
                    leaving = k
        step = wp.min(primal, dual)
        if step >= _NONE:
            break
        mu = mu - step * d
        entering += step
        if primal < _NONE:
            s = s + step * direction
        if primal <= dual:
            a0 = wp.where(n == 0, aj, a0)
            a1 = wp.where(n == 1, aj, a1)
            a2 = wp.where(n == 2, aj, a2)
            ab = wp.vec3(wp.where(n == 0, bj, ab[0]), wp.where(n == 1, bj, ab[1]), wp.where(n == 2, bj, ab[2]))
            i0 = wp.where(n == 0, j, i0)
            i1 = wp.where(n == 1, j, i1)
            i2 = wp.where(n == 2, j, i2)
            mu = wp.vec3(
                wp.where(n == 0, entering, mu[0]), wp.where(n == 1, entering, mu[1]), wp.where(n == 2, entering, mu[2])
            )
            # The entering row's remainder is the factor's new column: set_basis's own arithmetic for the grown set.
            inv = 1.0 / wp.max(wp.sqrt(along), 1.0e-30)
            column = rest * inv
            e0 = wp.where(n == 0, column, e0)
            e1 = wp.where(n == 1, column, e1)
            e2 = wp.where(n == 2, column, e2)
            R = wp.mat33(
                wp.where(n == 0, inv, R[0, 0]),
                0.0,
                0.0,
                wp.where(n == 1, c0, R[1, 0]),
                wp.where(n == 1, inv, R[1, 1]),
                0.0,
                wp.where(n == 2, c0, R[2, 0]),
                wp.where(n == 2, c1, R[2, 1]),
                wp.where(n == 2, inv, R[2, 2]),
            )
            n += 1
            ended = 1
            break
        a0 = wp.where(leaving == 0, a1, a0)
        a1 = wp.where(leaving <= 1, a2, a1)
        a2 = wp.vec3(0.0, 0.0, 0.0)
        ab = wp.vec3(wp.where(leaving == 0, ab[1], ab[0]), wp.where(leaving <= 1, ab[2], ab[1]), 0.0)
        i0 = wp.where(leaving == 0, i1, i0)
        i1 = wp.where(leaving <= 1, i2, i1)
        i2 = -1
        mu = wp.vec3(wp.where(leaving == 0, mu[1], mu[0]), wp.where(leaving <= 1, mu[2], mu[1]), 0.0)
        n -= 1
        e0, e1, e2, R = set_basis(a0, a1, a2, L, n)
    return a0, a1, a2, ab, i0, i1, i2, mu, n, s, ended, e0, e1, e2, R


@REGION_LAW.law
def candidate_row(record: wp.vec4) -> wp.vec3:
    """A candidate's row over the region's normal coordinates from its record ``(a1, a2, b, eps)``."""
    return wp.vec3(1.0, record[0], record[1])


@REGION_LAW.law
def region_block(
    start: wp.vec3,
    K: wp.mat33,
    L: wp.mat33,
    without: wp.vec3,
    cand: wp.array[wp.vec4],
    g0: int,
    g1: int,
    i0_in: int,
    i1_in: int,
    i2_in: int,
    spring_in: wp.vec3,
    tol: float,
) -> tuple[wp.vec3, int, int, int, wp.vec3, wp.vec3, wp.vec3, int, int, int]:
    """The candidates' block of the region whose candidates are ``cand[g0:g1]``, from the velocity ``start`` (the
    region's normal coordinates) in the metric ``K``: the impulses ``lambda >= 0`` with
    ``r = A (start + K A^T lambda) + b + eps lambda >= 0`` and ``lambda . r = 0``, solved exactly (the module's
    statement), from the set and the springs' wrench its last solve left.

    ``without`` is the velocity without the region: the block is *open* when no candidate closes at ``without`` faster
    than ``tol``.

    Returns ``(wrench, i0, i1, i2, multipliers, springs, velocity, open, bounded, steps)``: the block's wrench
    ``A^T lambda`` (its first coordinate the load), the rigid set and its multipliers, the compliant candidates'
    part of the wrench, the velocity it leaves, whether it is open (1) and whether the rigid rows admitted a
    velocity (1)."""
    zero = wp.vec3(0.0, 0.0, 0.0)
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    v0 = cand[wp.max(i0_in, g0)]
    v1 = cand[wp.max(i1_in, g0)]
    v2 = cand[wp.max(i2_in, g0)]
    a0 = wp.where(i0_in >= 0, candidate_row(v0), zero)
    a1 = wp.where(i1_in >= 0, candidate_row(v1), zero)
    a2 = wp.where(i2_in >= 0, candidate_row(v2), zero)
    ab = wp.vec3(wp.where(i0_in >= 0, v0[2], 0.0), wp.where(i1_in >= 0, v1[2], 0.0), wp.where(i2_in >= 0, v2[2], 0.0))
    n = wp.where(i0_in >= 0, 1, 0) + wp.where(i1_in >= 0, 1, 0) + wp.where(i2_in >= 0, 1, 0)
    Lk = L
    a0, a1, a2, ab, i0, i1, i2, mu, n, e0, e1, e2, R = set_start(start, Lk, a0, a1, a2, ab, i0_in, i1_in, i2_in, n)
    # The metric and the start with the closing springs in them: none yet.
    Ks = K
    Ls = Lk
    ss = start
    S = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    q = zero
    s = start + K * (a0 * mu[0] + a1 * mu[1] + a2 * mu[2] + spring_in)
    # The closing springs the velocity was solved with: none, or (-1) those of the last solve, which are not known.
    last_count = int(0)
    if spring_in[0] != 0.0:
        last_count = -1
    last_sum = int(0)
    last_square = int(0)
    opened = int(1)
    bounded = int(1)
    steps = int(0)
    turn = int(0)
    while turn < REGION_STEPS:
        turn += 1
        # One pass over the candidates, four loads a step: who closes without the region, the springs that close at
        # the velocity (their sums), and the rigid candidate outside the set that closes fastest there.
        s00 = float(0.0)
        s01 = float(0.0)
        s02 = float(0.0)
        s11 = float(0.0)
        s12 = float(0.0)
        s22 = float(0.0)
        qs = zero
        count = int(0)
        index_sum = int(0)
        index_square = int(0)
        j = int(-1)
        worst = -tol
        i = g0
        while i < g1:
            for k in range(4):
                ii = wp.min(i + k, g1 - 1)
                record = cand[ii]
                a = candidate_row(record)
                if i + k < g1:
                    if wp.dot(a, without) + record[2] < -tol:
                        opened = 0
                    r = wp.dot(a, s) + record[2]
                    if record[3] != 0.0:
                        if r < 0.0:
                            ai = a * (1.0 / record[3])
                            s00 += ai[0] * a[0]
                            s01 += ai[0] * a[1]
                            s02 += ai[0] * a[2]
                            s11 += ai[1] * a[1]
                            s12 += ai[1] * a[2]
                            s22 += ai[2] * a[2]
                            qs += ai * record[2]
                            count += 1
                            index_sum += ii - g0
                            index_square += (ii - g0) * (ii - g0)
                    elif ii != i0 and ii != i1 and ii != i2 and r < worst:
                        worst = r
                        j = ii
            i += 4
        if opened == 1:
            break
        if count != last_count or index_sum != last_sum or index_square != last_square:
            # The closing springs changed: Newton's step on the three coordinates, the rigid set solved in it.
            last_count = count
            last_sum = index_sum
            last_square = index_square
            S = wp.mat33(s00, s01, s02, s01, s11, s12, s02, s12, s22)
            q = qs
            shape = wp.inverse(identity + K * S)
            Ks = shape * K
            Ls = metric_factor(Ks)
            ss = shape * (start - K * q)
            a0, a1, a2, ab, i0, i1, i2, mu, n, e0, e1, e2, R = set_start(ss, Ls, a0, a1, a2, ab, i0, i1, i2, n)
            s = ss + Ks * (a0 * mu[0] + a1 * mu[1] + a2 * mu[2])
            steps += 1
            continue
        if j < 0:
            break
        record = cand[j]
        a0, a1, a2, ab, i0, i1, i2, mu, n, s, ended, e0, e1, e2, R = set_enter(
            s, Ls, candidate_row(record), record[2], j, a0, a1, a2, ab, i0, i1, i2, mu, n, e0, e1, e2, R
        )
        steps += 1
        if ended == 0:
            bounded = 0
            break
    if opened == 1:
        return zero, -1, -1, -1, zero, zero, without, 1, 1, steps
    spring = zero
    if last_count > 0:
        spring = zero - (S * s + q)
    return a0 * mu[0] + a1 * mu[1] + a2 * mu[2] + spring, i0, i1, i2, mu, spring, s, 0, bounded, steps


@REGION_LAW.law
def friction_unit(C: wp.mat33, G: wp.mat33, friction: int) -> tuple[wp.mat33, wp.mat33]:
    """A friction unit's coupling and block in a region's coordinates, ``(Cu, Gu)``: a unit without a spin row holds two
    coordinates, its third free (no coupling, a unit diagonal)."""
    Cu = C
    Gu = G
    if friction < 3:
        Cu = wp.mat33(C[0, 0], C[0, 1], 0.0, C[1, 0], C[1, 1], 0.0, C[2, 0], C[2, 1], 0.0)
        Gu = wp.mat33(G[0, 0], G[0, 1], 0.0, G[1, 0], G[1, 1], 0.0, 0.0, 0.0, 1.0)
    return Cu, Gu


@REGION_LAW.law
def stick_terms(
    Wnn: wp.mat33, C: wp.mat33, G: wp.mat33, cfm: float, friction: int
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
    """A region's terms that stand for a step (its response does): the regularized friction block's inverse ``Ginv``,
    the stick metric ``K = Wnn - C Ginv C^T`` (the unit held), and the factors of ``K`` and of ``Wnn``
    (:func:`metric_factor`), ``(Ginv, K, Lk, Lw)``; a region without friction rows holds ``Wnn``'s alone."""
    Cu, Gu = friction_unit(C, G, friction)
    Gs = Gu + wp.mat33(cfm, 0.0, 0.0, 0.0, cfm, 0.0, 0.0, 0.0, cfm)
    Ginv = wp.inverse(Gs)
    CG = Cu * Ginv
    K = Wnn - CG * wp.transpose(Cu)
    Lw = metric_factor(Wnn)
    Lk = Lw
    if friction > 0:
        Lk = metric_factor(K)
    return Ginv, K, Lk, Lw


@REGION_LAW.law
def region_visit(
    sn: wp.vec3,
    sf: wp.vec3,
    Wnn: wp.mat33,
    C: wp.mat33,
    G: wp.mat33,
    cfm: float,
    bf: wp.vec3,
    mu: float,
    mu_t: float,
    friction: int,
    ball_values: wp.vec3,
    ball_vectors: wp.mat33,
    Ginv: wp.mat33,
    K: wp.mat33,
    Lk: wp.mat33,
    Lw: wp.mat33,
    cand: wp.array[wp.vec4],
    g0: int,
    g1: int,
    wrench_in: wp.vec3,
    phi_in: wp.vec3,
    i0_in: int,
    i1_in: int,
    i2_in: int,
    spring_in: wp.vec3,
    tol: float,
) -> tuple[wp.vec3, wp.vec3, int, int, int, wp.vec3, wp.vec3, wp.vec3, int, int]:
    """One visit of a region (the module's statement), every other impulse held.

    Args:
        sn: The pair's relative twist in the region's normal coordinates ``N``, with the region's wrench in it.
        sf: The same in its friction coordinates ``F``.
        Wnn: The pair's response over ``N``.
        C: The response of ``N`` to a wrench over ``F`` (``Wnf``).
        G: The response over ``F``.
        cfm: The friction unit's regularization, added to ``G``'s diagonal in its solves.
        bf: The friction rows' right-hand sides.
        mu: The friction coefficient.
        mu_t: The torsional coefficient [m], 0 for a unit without a spin row.
        friction: The friction unit's rows: 0 (none), 2 (a tangent pair) or 3 (with the spin row).
        ball_values: The cone law's metric of the regularized friction block (:func:`.projection_laws.friction_ball_metric`
            of ``G`` plus ``cfm``), its eigenvalues; read where ``mu_t`` is positive.
        ball_vectors: Its eigenvectors.
        Ginv: The regularized friction block's inverse (:func:`stick_terms`).
        K: The stick metric.
        Lk: Its factor.
        Lw: ``Wnn``'s factor.
        cand: The candidates' records.
        g0: The region's first candidate.
        g1: One past its last.
        wrench_in: The normal wrench the region holds (over ``N``).
        phi_in: The friction it holds (over ``F``).
        i0_in: The rigid set its last visit left (-1: none), as indices of ``cand``.
        i1_in: Its second row.
        i2_in: Its third row.
        spring_in: The compliant candidates' part of ``wrench_in``.
        tol: The closing threshold [m/s].

    Returns:
        ``(wrench, phi, i0, i1, i2, multipliers, springs, velocity, outcome, steps)``: the wrench and friction the
        region holds now, its rigid set and that set's multipliers, its springs' wrench, its normal coordinates'
        velocity after the visit (a compliant candidate's impulse is ``max(0, -(a . velocity + b)) / eps``), and
        :data:`REGION_OPEN`, :data:`REGION_STICK` or :data:`REGION_SLIDE`.
    """
    zero = wp.vec3(0.0, 0.0, 0.0)
    # A unit without a spin row holds two coordinates: the third is free (no coupling, no residual).
    Cu, Gu = friction_unit(C, G, friction)
    rf0 = sf + bf
    if friction < 3:
        rf0 = wp.vec3(rf0[0], rf0[1], 0.0)
    # The region's own wrench leaves the velocity.
    without = sn - Wnn * wrench_in
    if friction > 0:
        without = without - Cu * phi_in
        rf0 = rf0 - wp.transpose(Cu) * wrench_in - Gu * phi_in
    if friction == 0:
        wrench, i0, i1, i2, multipliers, spring, velocity, opened, _bounded, steps = region_block(
            without, Wnn, Lw, without, cand, g0, g1, i0_in, i1_in, i2_in, spring_in, tol
        )
        if opened == 1:
            return zero, zero, -1, -1, -1, zero, zero, without, REGION_OPEN, steps
        return wrench, zero, i0, i1, i2, multipliers, spring, velocity, REGION_STICK, steps
    # Stick first: the friction unit held at zero residual with the candidates' block.
    Gs = Gu + wp.mat33(cfm, 0.0, 0.0, 0.0, cfm, 0.0, 0.0, 0.0, cfm)
    CG = Cu * Ginv
    wrench, i0, i1, i2, multipliers, spring, velocity, opened, bounded, steps = region_block(
        without - CG * rf0, K, Lk, without, cand, g0, g1, i0_in, i1_in, i2_in, spring_in, tol
    )
    if opened == 1:
        return zero, zero, -1, -1, -1, zero, zero, without, REGION_OPEN, steps
    load = wrench[0]
    phi = zero - Ginv * (rf0 + wp.transpose(Cu) * wrench)
    scaled = wp.vec3(phi[0] / mu, phi[1] / mu, 0.0)
    if mu_t > 0.0:
        scaled = wp.vec3(phi[0] / mu, phi[1] / mu, phi[2] / mu_t)
    if bounded == 1 and load > 0.0 and mu > 0.0 and wp.dot(scaled, scaled) <= load * load:
        return wrench, phi, i0, i1, i2, multipliers, spring, velocity, REGION_STICK, steps
    # Slide: the candidates' block with the friction as it was, then the unit's cone law at the block's load, once.
    start = without + Cu * phi_in
    wrench, i0, i1, i2, multipliers, spring, velocity, _opened, _bounded, more = region_block(
        start, Wnn, Lw, without, cand, g0, g1, i0, i1, i2, spring, tol
    )
    residual = rf0 + wp.transpose(Cu) * wrench + Gu * phi_in
    phi = wp.vec3(0.0, 0.0, 0.0)
    if mu_t > 0.0:
        phi = friction_triple_from_metric(ball_values, ball_vectors, residual, phi_in, mu, mu_t, wrench[0], 1.0)
    else:
        phi = friction_triple(Gs, residual, phi_in, mu, mu_t, wrench[0], 1.0)
    return wrench, phi, i0, i1, i2, multipliers, spring, velocity, REGION_SLIDE, steps + more


@REGION_LAW.law
def frame_twist(v: wp.vec3, w: wp.vec3, r: wp.vec3, n: wp.vec3, t0: wp.vec3, t1: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """A body's twist ``(v, w)`` about its origin in a region's coordinates ``(N, F)``: the frame ``n``, ``t0``,
    ``t1`` at the anchor ``r`` from the origin."""
    at = v + wp.cross(w, r)
    return wp.vec3(wp.dot(n, at), wp.dot(t0, w), wp.dot(t1, w)), wp.vec3(wp.dot(t0, at), wp.dot(t1, at), wp.dot(n, w))


@REGION_LAW.law
def frame_wrench(
    wrench: wp.vec3, phi: wp.vec3, r: wp.vec3, n: wp.vec3, t0: wp.vec3, t1: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    """A region's wrench ``(N, F)`` as a force and a torque about a body's origin, :func:`frame_twist`'s transpose."""
    force = n * wrench[0] + t0 * phi[0] + t1 * phi[1]
    return force, wp.cross(r, force) + t0 * wrench[1] + t1 * wrench[2] + n * phi[2]


@REGION_LAW.law
def frame_response(
    Avv: wp.mat33, Avw: wp.mat33, Aww: wp.mat33, r: wp.vec3, n: wp.vec3, t0: wp.vec3, t1: wp.vec3
) -> tuple[wp.mat33, wp.mat33, wp.mat33]:
    """A body's response in a region's coordinates, ``(Wnn, C, G)``: the twist :func:`frame_twist` reads per unit of
    the wrench :func:`frame_wrench` applies, from the body's inverse inertia about its origin
    ``[[Avv, Avw], [Avw^T, Aww]]`` (its twist per wrench, linear part first)."""
    R = wp.mat33(n[0], n[1], n[2], t0[0], t0[1], t0[2], t1[0], t1[1], t1[2])
    Rt = wp.transpose(R)
    X = wp.skew(r)
    # At the anchor: v_p = v - r x w.
    Pvw = Avw - X * Aww
    Pvv = Avv - X * wp.transpose(Avw) + Pvw * X
    V = R * Pvv * Rt
    B = R * Pvw * Rt
    O = R * Aww * Rt
    Wnn = wp.mat33(V[0, 0], B[0, 1], B[0, 2], B[0, 1], O[1, 1], O[1, 2], B[0, 2], O[1, 2], O[2, 2])
    C = wp.mat33(V[0, 1], V[0, 2], B[0, 0], B[1, 1], B[2, 1], O[1, 0], B[1, 2], B[2, 2], O[2, 0])
    G = wp.mat33(V[1, 1], V[1, 2], B[1, 0], V[1, 2], V[2, 2], B[2, 0], B[1, 0], B[2, 0], O[0, 0])
    return Wnn, C, G

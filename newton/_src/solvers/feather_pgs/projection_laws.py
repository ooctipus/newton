# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The projection laws: how a sweep's row update meets the row's admissible set.

One law each, written once for Warp kernels and native CUDA (:mod:`.law_source`):

* the update: :func:`relaxed_step` (``old + omega delta``), the proximal
  regularization of a row's step, :func:`regularized_delta` (the w-form), its
  diagonal for block solves, :func:`regularized_diagonal`, and
  :func:`friction_gate` (friction rows wait for ``friction_start_iteration``);
* sets: :func:`unilateral` (``lambda >= 0``), :func:`box` (``|lambda| <= b``) and
  :func:`disk` (``|lambda_t| <= r``);
* friction at a fixed normal load: :func:`friction_pair`, the bounded solve of a
  tangent pair on its disk, composed of :func:`friction_pair_metric` and
  :func:`friction_pair_from_metric` for sweeps that keep a pair's block across
  iterations; and :func:`friction_ball`, the joint solve of the pair and the
  spin row on their elliptic cone (the pair law's scheme in three dimensions,
  :func:`friction_ball_metric` and :func:`friction_ball_from_metric`);
* velocity limits: :func:`velocity_limit_update`, the unrelaxed unilateral law
  of a velocity-limit row, and its free-body speed form :func:`speed_row`.
"""

import warp as wp

from .law_source import LawFamily
from .row_laws import ROW_FRICTION

PAIR_FLOOR = wp.constant(2.0e-7)
"""The pair's rounding floor, about two float32 ulps of its impulse: the root solve stops within it of the disk's
radius, and a sliding impulse that already meets the disk problem's KKT conditions within it is kept."""

PROJECTION_LAW = LawFamily("ProjectionLaw")
"""The projection laws' family; native code calls ``ProjectionLaw::friction_pair(...)``."""


@PROJECTION_LAW.law
def relaxed_step(old: float, delta: float, omega: float) -> float:
    """Return a row's relaxed update, ``old + omega delta``, for its step ``delta`` (``-r / d``, or regularized)."""
    return old + omega * delta


@PROJECTION_LAW.law
def regularized_delta(delta: float, w: float, old: float) -> float:
    """Return a row's step in the proximal w-form, ``w delta - (1 - w) old``, for its hard step ``delta``.

    ``w = 1 / (1 + g)`` is the row's regularization weight. The form has no
    ``g lambda`` product that could overflow for an extreme finite ``g``, and
    ``w = 1`` leaves the hard step: sweeps apply it only to regularized rows.
    """
    return delta * w - (1.0 - w) * old


@PROJECTION_LAW.law
def regularized_diagonal(diagonal: float, w: float) -> float:
    """Return the diagonal a block solve gives a row of weight ``w``, ``d / w``.

    A block solved exactly with it reaches the fixed point of the regularized
    step (:func:`regularized_delta`): the row's residual settles at
    ``-(1 / w - 1) d lambda``.
    """
    return diagonal / w


@PROJECTION_LAW.law
def friction_gate(kind: int, iteration: int, start_iteration: int) -> bool:
    """Return whether a row is a friction row still held at zero: friction starts at ``start_iteration``."""
    return kind == ROW_FRICTION and iteration < start_iteration


@PROJECTION_LAW.law
def unilateral(impulse: float) -> float:
    """Project an impulse onto ``impulse >= 0`` (contact normals, position and velocity limits).

    Selected, not branched, as the sweeps' own clamps compiled.
    """
    return wp.where(impulse < 0.0, 0.0, impulse)


@PROJECTION_LAW.law
def box(impulse: float, lower: float, upper: float) -> float:
    """Project an impulse onto ``[lower, upper]``, testing the upper bound first."""
    result = wp.where(impulse > upper, upper, impulse)
    return wp.where(result < lower, lower, result)


@PROJECTION_LAW.law
def disk(pair: wp.vec2, radius: float) -> wp.vec2:
    """Scale a tangent impulse pair back onto the disk of ``radius`` when it lies outside.

    The scale is selected, not branched, as the sweeps' own projections selected it.
    """
    magnitude = wp.sqrt(pair[0] * pair[0] + pair[1] * pair[1])
    return pair * wp.where(magnitude > radius, radius / magnitude, 1.0)


@PROJECTION_LAW.law
def friction_pair_metric(a: float, c: float, d: float) -> tuple[float, float, float, wp.vec2]:
    """Return the eigen-decomposition :func:`friction_pair` solves in: ``(scale, largest, smallest, axis)``.

    ``[[a, c], [c, d]]`` is the pair's Delassus block, scaled by its largest
    diagonal; ``axis`` is the unit eigenvector of the larger eigenvalue. A sweep
    whose block stays fixed computes it once and calls :func:`friction_pair_from_metric`.
    """
    scale = wp.max(wp.max(a, d), 1.0e-20)
    a /= scale
    c /= scale
    d /= scale
    largest = 0.5 * (a + d + wp.sqrt((a - d) * (a - d) + 4.0 * c * c))
    smallest = float(0.0)
    if largest > 0.0:
        smallest = wp.max((a * d - c * c) / largest, 0.0)
    axis = wp.vec2(c, largest - a)
    if a >= d:
        axis = wp.vec2(largest - d, c)
    if wp.length(axis) > 0.0:
        axis = wp.normalize(axis)
    else:
        axis = wp.vec2(1.0, 0.0)
    return scale, largest, smallest, axis


@PROJECTION_LAW.law(inline="default")
def friction_pair_from_metric(
    scale: float,
    largest: float,
    smallest: float,
    axis: wp.vec2,
    residual: wp.vec2,
    old: wp.vec2,
    radius: float,
    omega: float,
) -> wp.vec2:
    """Solve the tangent pair of :func:`friction_pair` from its block's :func:`friction_pair_metric`."""
    if radius <= 0.0:
        return wp.vec2(0.0)
    perpendicular = wp.vec2(-axis[1], axis[0])
    residual_rotated = wp.vec2(wp.dot(axis, residual), wp.dot(perpendicular, residual)) / scale
    # The hold test below reads |old|. Squared norms screen it without a root, beside the sticking test: a pair the
    # test keeps has |old|^2 within three floors of radius^2 (squaring doubles the floor; rounding adds under one), so
    # ten floors pass every pair it keeps. A radius under 1e-15 skips the screen, which keeps its operands normal.
    old_norm_sq = wp.dot(old, old)
    radius_sq = radius * radius
    near_circle = radius < 1.0e-15 or wp.abs(old_norm_sq - radius_sq) <= 10.0 * PAIR_FLOOR * radius_sq
    result = old
    sticking = bool(False)
    if largest > 0.0 and (smallest > 0.0 or residual_rotated[1] == 0.0):
        correction = (residual_rotated[0] / largest) * axis
        if smallest > 0.0:
            correction += (residual_rotated[1] / smallest) * perpendicular
        # Incremental sticking preserves a feasible zero-residual impulse exactly.
        result = old - correction
        sticking = wp.length(result) <= radius
    if not sticking:
        # A sliding pair already at its disk problem's solution to the rounding floor keeps its impulse: on its circle
        # within the floor of the radius, and its residual anti-parallel to it within the floor (the KKT conditions, at
        # the floor the root solve stops at). Its re-solve could move it only within that floor.
        if near_circle:
            old_norm = wp.length(old)
            cross_residual = residual[0] * old[1] - residual[1] * old[0]
            if (
                old_norm > 0.0
                and old_norm <= radius
                and radius - old_norm <= PAIR_FLOOR * radius
                and wp.dot(residual, old) <= 0.0
                and wp.abs(cross_residual) <= PAIR_FLOOR * old_norm * wp.length(residual)
            ):
                return old
        old_rotated = wp.vec2(wp.dot(axis, old), wp.dot(perpendicular, old))
        b = wp.vec2(largest * old_rotated[0], smallest * old_rotated[1]) - residual_rotated
        solution = wp.vec2(0.0)
        lo = float(0.0)
        hi = wp.length(b) / radius
        if hi > 0.0:
            # Both bounds are <= the multiplier root; together they also bound
            # the starting norm by sqrt(2) * radius. The reciprocal norm is
            # increasing and concave, giving Newton a useful step near singularity.
            lo = wp.max(wp.max(hi - largest, wp.abs(b[1]) / radius - smallest), 0.0)
            alpha = lo
            converged = bool(False)
            for _ in range(8):
                inv0 = 1.0 / (largest + alpha)
                inv1 = float(0.0)
                if smallest + alpha > 0.0:
                    inv1 = 1.0 / (smallest + alpha)
                trial = old_rotated - wp.vec2(
                    (residual_rotated[0] + alpha * old_rotated[0]) * inv0,
                    (residual_rotated[1] + alpha * old_rotated[1]) * inv1,
                )
                norm = wp.length(trial)
                # The stopping error is relative to the impulse budget. The
                # caller projects after relaxation; keep bisection for hard cases.
                if wp.abs(norm - radius) <= PAIR_FLOOR * radius or (alpha == 0.0 and norm <= radius):
                    solution = trial
                    converged = True
                    break
                if norm > radius:
                    lo = alpha
                else:
                    hi = alpha
                slope = trial[0] * trial[0] * inv0 + trial[1] * trial[1] * inv1
                candidate = alpha
                if slope > 0.0:
                    candidate = alpha + (norm / radius - 1.0) * norm * norm / slope
                alpha = 0.5 * (lo + hi)
                if candidate > lo and candidate < hi:
                    alpha = candidate
            if not converged:
                for _ in range(24):
                    alpha = 0.5 * (lo + hi)
                    trial = wp.vec2(b[0] / (largest + alpha), b[1] / (smallest + alpha))
                    if wp.length(trial) > radius:
                        lo = alpha
                    else:
                        hi = alpha
                solution = wp.vec2(b[0] / (largest + hi), b[1] / (smallest + hi))
        result = old + (solution[0] - old_rotated[0]) * axis + (solution[1] - old_rotated[1]) * perpendicular
    return old + omega * (result - old)


@PROJECTION_LAW.law(inline="default")
def friction_pair(
    a: float, c: float, d: float, residual: wp.vec2, old: wp.vec2, radius: float, omega: float
) -> wp.vec2:
    """Minimize the tangent quadratic on a disk at a fixed normal load, then apply relaxation.

    ``[[a, c], [c, d]]`` is the pair's Delassus block and ``residual`` its rows'
    residuals at ``old``. The caller projects the relaxed result back onto the
    disk. Sticking uses the block inverse; sliding solves ``(A + alpha I) x = b``
    with ``|x| = radius``. Incremental sticking avoids perturbing fixed points. An
    orthonormal eigenbasis avoids cancellation in nearly singular blocks.
    Safeguarded Newton accelerates sliding; bisection retains a feasible
    fallback. A sliding impulse that already meets the disk problem's KKT
    conditions at the rounding floor (:data:`PAIR_FLOOR`) is kept.
    """
    scale, largest, smallest, axis = friction_pair_metric(a, c, d)
    return friction_pair_from_metric(scale, largest, smallest, axis, residual, old, radius, omega)


BALL_POLISH = 1
"""Exact Jacobi sweeps that polish ``wp.eig3``'s eigenbasis in :func:`friction_ball_metric`. Its four approximate
sweeps leave off-diagonal terms up to 5e-3 of the scaled block on some float32 blocks, and the triple's solution
errors to 2e-4 of its load; one exact sweep brings both to the rounding floor (2e-7 and 2e-6)."""


@PROJECTION_LAW.law
def jacobi_rotation(matrix: wp.mat33, vectors: wp.mat33, p: int, q: int) -> tuple[wp.mat33, wp.mat33]:
    """Zero a symmetric matrix's ``(p, q)`` entry by a rotation ``R`` in that plane: ``(R^T matrix R, vectors R)``.

    Golub and Van Loan's ``schur2``; ``vectors`` holds eigenvectors in its columns, rotated with the matrix.
    """
    off = matrix[p, q]
    if off == 0.0:
        return matrix, vectors
    theta = (matrix[q, q] - matrix[p, p]) / (2.0 * off)
    t = 1.0 / (wp.abs(theta) + wp.sqrt(theta * theta + 1.0))
    if theta < 0.0:
        t = -t
    c = 1.0 / wp.sqrt(t * t + 1.0)
    s = t * c
    first = wp.vec3(wp.where(p == 0, 1.0, 0.0), wp.where(p == 1, 1.0, 0.0), wp.where(p == 2, 1.0, 0.0))
    second = wp.vec3(wp.where(q == 0, 1.0, 0.0), wp.where(q == 1, 1.0, 0.0), wp.where(q == 2, 1.0, 0.0))
    rotation = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    rotation = rotation + (c - 1.0) * (wp.outer(first, first) + wp.outer(second, second))
    rotation = rotation + s * (wp.outer(first, second) - wp.outer(second, first))
    return wp.transpose(rotation) * matrix * rotation, vectors * rotation


@PROJECTION_LAW.law
def friction_ball_metric(block: wp.mat33, mu: float, mu_t: float) -> tuple[float, wp.vec3, wp.mat33]:
    """Return the eigen-decomposition :func:`friction_ball` solves in: ``(scale, values, vectors)``.

    ``block`` is the Delassus block of the rows ``(t1, t2, s)``. In the scaled impulse
    ``y = (t1 / mu, t2 / mu, s / mu_t)`` it is ``C block C``, ``C = diag(mu, mu, mu_t)``, here divided by its largest
    diagonal ``scale``. ``vectors`` holds its unit eigenvectors in columns and ``values`` their eigenvalues, clamped
    at zero: ``wp.eig3``'s, polished by :data:`BALL_POLISH` exact Jacobi sweeps. A sweep whose block stays fixed
    computes it once and calls :func:`friction_ball_from_metric`.
    """
    coefficients = wp.vec3(mu, mu, mu_t)
    scaled = wp.cw_mul(block, wp.outer(coefficients, coefficients))
    scale = wp.max(wp.max(wp.max(scaled[0, 0], scaled[1, 1]), scaled[2, 2]), 1.0e-20)
    scaled = scaled / scale
    vectors, values = wp.eig3(scaled)
    diagonal = wp.transpose(vectors) * scaled * vectors
    for _ in range(wp.static(BALL_POLISH)):
        diagonal, vectors = jacobi_rotation(diagonal, vectors, 0, 1)
        diagonal, vectors = jacobi_rotation(diagonal, vectors, 0, 2)
        diagonal, vectors = jacobi_rotation(diagonal, vectors, 1, 2)
    values = wp.vec3(wp.max(diagonal[0, 0], 0.0), wp.max(diagonal[1, 1], 0.0), wp.max(diagonal[2, 2], 0.0))
    return scale, values, vectors


@PROJECTION_LAW.law(inline="default")
def friction_ball_from_metric(
    scale: float,
    values: wp.vec3,
    vectors: wp.mat33,
    residual: wp.vec3,
    old: wp.vec3,
    mu: float,
    mu_t: float,
    load: float,
    omega: float,
) -> wp.vec3:
    """Solve the friction triple of :func:`friction_ball` from its block's :func:`friction_ball_metric`.

    The pair law's scheme in the scaled impulse ``y``, on the ball ``|y| <= load``: incremental sticking by the
    block inverse, the hold of a sliding impulse at the KKT conditions' rounding floor (:data:`PAIR_FLOOR`), and
    Newton on the reciprocal norm of ``y(alpha) = (Y + alpha I)^-1 b`` in the eigenbasis, bisection its guard.
    """
    if load <= 0.0:
        return wp.vec3(0.0)
    # A zero coefficient pins its component: its scaled impulse, gradient and change are zero.
    reciprocal = wp.where(mu > 0.0, 1.0 / mu, 0.0)
    old_scaled = wp.cw_mul(old, wp.vec3(reciprocal, reciprocal, wp.where(mu_t > 0.0, 1.0 / mu_t, 0.0)))
    coefficients = wp.vec3(mu, mu, mu_t)
    gradient = wp.cw_mul(residual, coefficients)
    old_rotated = wp.transpose(vectors) * old_scaled
    residual_rotated = (wp.transpose(vectors) * gradient) / scale
    largest = wp.max(wp.max(values[0], values[1]), values[2])
    change = wp.vec3(0.0)
    sticking = bool(False)
    invertible = (values[0] > 0.0 or residual_rotated[0] == 0.0) and (values[1] > 0.0 or residual_rotated[1] == 0.0)
    if largest > 0.0 and invertible and (values[2] > 0.0 or residual_rotated[2] == 0.0):
        step = wp.vec3(
            wp.where(values[0] > 0.0, residual_rotated[0] / values[0], 0.0),
            wp.where(values[1] > 0.0, residual_rotated[1] / values[1], 0.0),
            wp.where(values[2] > 0.0, residual_rotated[2] / values[2], 0.0),
        )
        # Incremental sticking preserves a feasible zero-residual impulse exactly.
        change = -(vectors * step)
        sticking = wp.length(old_scaled + change) <= load
    if not sticking:
        # A sliding impulse already at its ball problem's solution to the rounding floor keeps its impulse: on the
        # sphere within the floor of the load, its gradient anti-parallel to it within the floor.
        load_sq = load * load
        if load < 1.0e-15 or wp.abs(wp.dot(old_scaled, old_scaled) - load_sq) <= 10.0 * PAIR_FLOOR * load_sq:
            old_norm = wp.length(old_scaled)
            if (
                old_norm > 0.0
                and old_norm <= load
                and load - old_norm <= PAIR_FLOOR * load
                and wp.dot(gradient, old_scaled) <= 0.0
                and wp.length(wp.cross(gradient, old_scaled)) <= PAIR_FLOOR * old_norm * wp.length(gradient)
            ):
                return old
        b = wp.cw_mul(values, old_rotated) - residual_rotated
        solution = wp.vec3(0.0)
        hi = wp.length(b) / load
        if hi > 0.0:
            # Both bounds are <= the multiplier root (|y| >= |b_k| / (values_k + alpha), |b| / (largest + alpha)).
            # The reciprocal norm is increasing and concave, giving Newton a useful step near singularity.
            lo = wp.max(hi - largest, 0.0)
            lo = wp.max(wp.max(lo, wp.abs(b[0]) / load - values[0]), wp.abs(b[1]) / load - values[1])
            lo = wp.max(lo, wp.abs(b[2]) / load - values[2])
            alpha = lo
            converged = bool(False)
            for _ in range(8):
                inv = wp.vec3(
                    wp.where(values[0] + alpha > 0.0, 1.0 / (values[0] + alpha), 0.0),
                    wp.where(values[1] + alpha > 0.0, 1.0 / (values[1] + alpha), 0.0),
                    wp.where(values[2] + alpha > 0.0, 1.0 / (values[2] + alpha), 0.0),
                )
                trial = old_rotated - wp.cw_mul(residual_rotated + alpha * old_rotated, inv)
                norm = wp.length(trial)
                if wp.abs(norm - load) <= PAIR_FLOOR * load or (alpha == 0.0 and norm <= load):
                    solution = trial
                    converged = True
                    break
                if norm > load:
                    lo = alpha
                else:
                    hi = alpha
                slope = wp.dot(wp.cw_mul(trial, trial), inv)
                candidate = alpha
                if slope > 0.0:
                    candidate = alpha + (norm / load - 1.0) * norm * norm / slope
                alpha = 0.5 * (lo + hi)
                if candidate > lo and candidate < hi:
                    alpha = candidate
            if not converged:
                for _ in range(24):
                    alpha = 0.5 * (lo + hi)
                    if wp.length(wp.cw_div(b, values + wp.vec3(alpha))) > load:
                        lo = alpha
                    else:
                        hi = alpha
                solution = wp.cw_div(b, values + wp.vec3(hi))
        change = vectors * (solution - old_rotated)
    return old + omega * wp.cw_mul(change, coefficients)


@PROJECTION_LAW.law(inline="default")
def friction_ball(
    block: wp.mat33, residual: wp.vec3, old: wp.vec3, mu: float, mu_t: float, load: float, omega: float
) -> wp.vec3:
    """Minimize the friction triple's quadratic on its elliptic cone at a fixed normal load, then apply relaxation.

    The triple is the friction pair and the spin row of a contact, ``block`` their Delassus block and ``residual``
    their residuals at ``old``. The cone is ``|(t1 / mu, t2 / mu, s / mu_t)| <= load``: the friction coefficient
    ``mu`` and the torsional one ``mu_t`` (a length) bound the impulses jointly, as maximal dissipation on a patch
    does. A zero coefficient pins its component; without spin (``mu_t = 0``) the triple is the pair, solved by
    :func:`friction_pair` bit for bit. The caller projects the relaxed result back onto the cone.
    """
    if mu_t <= 0.0:
        tangents = wp.vec2(residual[0], residual[1])
        pair = friction_pair(block[0, 0], block[0, 1], block[1, 1], tangents, wp.vec2(old[0], old[1]), mu * load, omega)
        return wp.vec3(pair[0], pair[1], 0.0)
    scale, values, vectors = friction_ball_metric(block, mu, mu_t)
    return friction_ball_from_metric(scale, values, vectors, residual, old, mu, mu_t, load, omega)


@PROJECTION_LAW.law
def velocity_limit_update(old: float, step: float) -> tuple[float, float]:
    """Return a velocity-limit row's new impulse and the impulse change to apply: ``(new, delta)``.

    The row is unilateral, ``r = J v + b >= 0`` with ``J`` signed by the side of the
    limit it bounds, and ``step = r / d`` its unrelaxed step. The row keeps a total,
    ``max(old - step, 0)``, like a contact normal or a position limit, so sweeps
    converge to the complementarity solution, which is independent of the row order
    and is the fixed point of the parallel tiers. From a zero impulse the update is
    the push of a violated row onto its limit, ``-step`` while ``r < 0``.
    """
    new = wp.max(old - step, 0.0)
    return new, new - old


@PROJECTION_LAW.law
def speed_row(speed: float, limit: float, effective_mass: float, old: float) -> wp.vec3:
    """Project a free body's speed row, ``|v| <= limit``, along its current velocity.

    The row's residual is ``limit - |v|`` and its step ``residual * effective_mass``
    (:func:`velocity_limit_update`). A correction acts along ``v``, so a body over
    its limit keeps its direction and ends on the limit.

    Returns:
        ``(new, delta, scale)``: the row's new impulse, the impulse change and the
        velocity change per unit velocity, ``v += scale v`` (0 without a change).
    """
    residual = limit - speed
    new, delta = velocity_limit_update(old, residual * effective_mass)
    scale = float(0.0)
    if delta != 0.0 and speed > 0.0:
        scale = -(delta / effective_mass) / speed
    return wp.vec3(new, delta, scale)

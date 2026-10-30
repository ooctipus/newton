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
  iterations;
* velocity limits: :func:`velocity_limit_update` (the stateless PhysX clamp or
  the accumulated unilateral law, :mod:`.velocity_limit_law`) and its free-body
  speed form :func:`speed_row`.
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
    """Project an impulse onto ``impulse >= 0`` (contact normals, position and accumulated velocity limits).

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


@PROJECTION_LAW.law
def velocity_limit_update(old: float, residual: float, step: float, accumulated: int) -> tuple[float, float]:
    """Return a velocity-limit row's new impulse and the impulse change to apply: ``(new, delta)``.

    The row is unilateral, ``r = J v + b >= 0``, and ``step = r / d`` its unrelaxed
    step. The accumulated law keeps a total, ``max(old - step, 0)``, and converges
    to the complementarity solution. The stateless law (PhysX's clamp) applies only
    the correction of the current overshoot, ``-step`` while ``r < 0``, and stores
    that correction instead of a total.
    """
    if accumulated != 0:
        new = wp.max(old - step, 0.0)
        return new, new - old
    if residual < 0.0:
        return -step, -step
    return 0.0, 0.0


@PROJECTION_LAW.law
def speed_row(speed: float, limit: float, effective_mass: float, old: float, accumulated: int) -> wp.vec3:
    """Project a free body's speed row, ``|v| <= limit``, along its current velocity.

    The row's residual is ``limit - |v|`` and its step ``residual * effective_mass``
    (:func:`velocity_limit_update`). A correction acts along ``v``, so a body over
    its limit keeps its direction and ends on the limit.

    Returns:
        ``(new, delta, scale)``: the row's new impulse, the impulse change and the
        velocity change per unit velocity, ``v += scale v`` (0 without a change).
    """
    residual = limit - speed
    new, delta = velocity_limit_update(old, residual, residual * effective_mass, accumulated)
    scale = float(0.0)
    if delta != 0.0 and speed > 0.0:
        scale = -(delta / effective_mass) / speed
    return wp.vec3(new, delta, scale)

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
* the contact's Coulomb problem of the research friction modes (the
  :data:`COULOMB_LAW` family): :func:`coulomb_bisection` (RAISim's bisection on
  the normal impulse, with :func:`de_saxce_target` for its de Saxce variant) and
  :func:`coulomb_newton` (Daviet's scalar Newton, :func:`coulomb_row`);
* velocity limits: :func:`velocity_limit_update` (the stateless PhysX clamp or
  the accumulated unilateral law, :mod:`.velocity_limit_law`), its free-body
  speed form :func:`speed_row`, and PhysX's velocity clamp of a driven
  coordinate, :func:`velocity_clamp_impulse`;
* :func:`physx_drive`, the PhysX drive row's impulse update.
"""

import warp as wp

from .law_source import LawFamily
from .row_laws import ROW_FRICTION

PROJECTION_LAW = LawFamily("ProjectionLaw")
"""The projection laws' family; native code calls ``ProjectionLaw::friction_pair(...)``."""
COULOMB_LAW = LawFamily("CoulombLaw")
"""The research friction modes' contact solvers (``friction_mode`` bisection and coulomb_newton), a family of
their own so the sweeps that do not use them do not compile them; native code calls ``CoulombLaw::...``."""

COULOMB_BISECTION_STEPS = wp.constant(20)
"""Bisection steps on the normal impulse of :func:`coulomb_bisection`."""
COULOMB_NEWTON_EXPANSIONS = wp.constant(30)
"""Doublings of the upper bracket of :func:`coulomb_row`."""
COULOMB_NEWTON_STEPS = wp.constant(50)
"""Safeguarded Newton steps of :func:`coulomb_row`."""
CoulombRowResult = wp.types.vector(length=6, dtype=float)
"""``[alpha, r_N, r_T0, r_T1, iterations, status]`` of :func:`coulomb_row` (status 0 sticks, 1 slides)."""


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
        # Retain an already feasible sliding solution within the same relative
        # accuracy as the root solve. Re-solving it can otherwise introduce
        # roundoff cycles that prevent a stationary PGS sweep.
        old_norm = wp.length(old)
        residual_norm = wp.length(residual)
        cross_residual = residual[0] * old[1] - residual[1] * old[0]
        if (
            old_norm > 0.0
            and old_norm <= radius
            and radius - old_norm <= 2.0e-7 * radius
            and wp.dot(residual, old) <= 0.0
            and wp.abs(cross_residual) <= 2.0e-7 * old_norm * residual_norm
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
                if wp.abs(norm - radius) <= 2.0e-7 * radius or (alpha == 0.0 and norm <= radius):
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
    fallback. Feasible sliding impulses already satisfying the KKT conditions to
    the same relative accuracy (2e-7) are retained to avoid roundoff cycles.
    """
    scale, largest, smallest, axis = friction_pair_metric(a, c, d)
    return friction_pair_from_metric(scale, largest, smallest, axis, residual, old, radius, omega)


@COULOMB_LAW.law
def de_saxce_target(target_n: float, u: wp.vec3, mu: float) -> float:
    """Raise a contact's normal target by ``mu |u_T|``, de Saxce's maximum-dissipation bias (Le Lidec 2024)."""
    return target_n + mu * wp.sqrt(u[1] * u[1] + u[2] * u[2])


@COULOMB_LAW.law(inline="default")
def coulomb_bisection(W: wp.mat33, u: wp.vec3, old: wp.vec3, mu: float, target_n: float) -> wp.vec3:
    """Solve a contact's Coulomb problem by bisection on its normal impulse (RAISim's law).

    ``W`` is the contact's symmetric 3x3 Delassus block (normal row first), ``u``
    its rows' current velocities at the impulses ``old`` and ``target_n`` the
    normal velocity the contact must reach (``-rhs`` of the normal row). Each
    probe of the normal impulse solves the tangent pair's 2x2 system and clamps
    it to the friction disk; the normal impulse is the smallest that reaches the
    target. A separating contact releases all three impulses; a degenerate block
    (``W_nn < 1e-20``) keeps them.

    Returns:
        The new normal and tangent impulses.
    """
    G_nn = W[0, 0]
    G_nt1 = W[0, 1]
    G_nt2 = W[0, 2]
    G_t1t1 = W[1, 1]
    G_t1t2 = W[1, 2]
    G_t2t2 = W[2, 2]
    u_n = u[0]
    u_t1 = u[1]
    u_t2 = u[2]
    old_lambda_n = old[0]
    old_lambda_t1 = old[1]
    old_lambda_t2 = old[2]
    new_lambda_n = old_lambda_n
    new_lambda_t1 = old_lambda_t1
    new_lambda_t2 = old_lambda_t2
    # Nested as the native sweep nests it: early returns change how the CUDA compiler unrolls the probes.
    if G_nn >= 1.0e-20:
        # The normal velocity with the normal impulse removed: lambda_n = 0 solves a separating contact.
        u_n_at_zero = u_n + G_nn * (0.0 - old_lambda_n)
        if u_n_at_zero >= target_n:
            new_lambda_n = 0.0
            new_lambda_t1 = 0.0
            new_lambda_t2 = 0.0
        else:
            lo = float(0.0)
            hi = wp.max(old_lambda_n * 2.0, (target_n - u_n) / G_nn + old_lambda_n)
            hi = wp.max(hi, 1.0)
            for _ in range(COULOMB_BISECTION_STEPS):
                mid = 0.5 * (lo + hi)
                d_n = mid - old_lambda_n
                ut1_eff = u_t1 + G_nt1 * d_n
                ut2_eff = u_t2 + G_nt2 * d_n
                det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2
                d_t1 = float(0.0)
                d_t2 = float(0.0)
                if wp.abs(det) > 1.0e-20:
                    d_t1 = (-ut1_eff * G_t2t2 + ut2_eff * G_t1t2) / det
                    d_t2 = (ut1_eff * G_t1t2 - ut2_eff * G_t1t1) / det
                trial_t1 = old_lambda_t1 + d_t1
                trial_t2 = old_lambda_t2 + d_t2
                flimit = mu * mid
                tmag = wp.sqrt(trial_t1 * trial_t1 + trial_t2 * trial_t2)
                if tmag > flimit and tmag > 1.0e-20:
                    sc = flimit / tmag
                    trial_t1 = trial_t1 * sc
                    trial_t2 = trial_t2 * sc
                d_t1_actual = trial_t1 - old_lambda_t1
                d_t2_actual = trial_t2 - old_lambda_t2
                u_n_trial = u_n + G_nn * d_n + G_nt1 * d_t1_actual + G_nt2 * d_t2_actual
                if u_n_trial < target_n:
                    lo = mid
                else:
                    hi = mid
            new_lambda_n = 0.5 * (lo + hi)
            # The tangent pair at the converged normal impulse.
            d_n_final = new_lambda_n - old_lambda_n
            ut1_f = u_t1 + G_nt1 * d_n_final
            ut2_f = u_t2 + G_nt2 * d_n_final
            det_f = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2
            d_t1_f = float(0.0)
            d_t2_f = float(0.0)
            if wp.abs(det_f) > 1.0e-20:
                d_t1_f = (-ut1_f * G_t2t2 + ut2_f * G_t1t2) / det_f
                d_t2_f = (ut1_f * G_t1t2 - ut2_f * G_t1t1) / det_f
            new_lambda_t1 = old_lambda_t1 + d_t1_f
            new_lambda_t2 = old_lambda_t2 + d_t2_f
            flimit_f = mu * new_lambda_n
            tmag_f = wp.sqrt(new_lambda_t1 * new_lambda_t1 + new_lambda_t2 * new_lambda_t2)
            if tmag_f > flimit_f and tmag_f > 1.0e-20:
                sc_f = flimit_f / tmag_f
                new_lambda_t1 = new_lambda_t1 * sc_f
                new_lambda_t2 = new_lambda_t2 * sc_f
    return wp.vec3(new_lambda_n, new_lambda_t1, new_lambda_t2)


@COULOMB_LAW.law
def mat22_solve(M: wp.mat22, rhs: wp.vec2) -> wp.vec2:
    """Solve ``M x = rhs`` by Cramer's rule."""
    inv_det = 1.0 / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    return wp.vec2(
        (M[1, 1] * rhs[0] - M[0, 1] * rhs[1]) * inv_det,
        (M[0, 0] * rhs[1] - M[1, 0] * rhs[0]) * inv_det,
    )


@COULOMB_LAW.law
def coulomb_root(AT: wp.mat22, cT: wp.vec2, wNT: wp.vec2, bN: float, WN: float, mu: float, alpha: float) -> wp.vec4:
    """Return ``(phi, phi', s[0], s[1])`` of Daviet's root function at ``alpha``, ``s = (A_T + alpha I)^-1 c_T``.

    ``phi(alpha) = |s| - mu (w_NT . s - b_N) / W_N``; the two solves share one determinant.
    """
    a = AT[0, 0] + alpha
    b_ = AT[0, 1]
    c = AT[1, 0]
    d = AT[1, 1] + alpha
    inv_det = 1.0 / (a * d - b_ * c)
    s0 = (d * cT[0] - b_ * cT[1]) * inv_det
    s1 = (a * cT[1] - c * cT[0]) * inv_det
    t0 = (d * s0 - b_ * s1) * inv_det
    t1 = (a * s1 - c * s0) * inv_det
    norm_s = wp.sqrt(s0 * s0 + s1 * s1)
    wNT_s = wNT[0] * s0 + wNT[1] * s1
    wNT_t = wNT[0] * t0 + wNT[1] * t1
    val = norm_s - mu * (wNT_s - bN) / WN
    dval = -(s0 * t0 + s1 * t1) / norm_s + mu * wNT_t / WN
    return wp.vec4(val, dval, s0, s1)


@COULOMB_LAW.law(inline="default")
def coulomb_row(W: wp.mat33, b: wp.vec3, mu: float) -> CoulombRowResult:
    """Solve one contact's Coulomb problem exactly: Daviet's scalar bracketed Newton (Daviet et al. 2011).

    ``W`` is the contact's SPD Delassus block (normal row first) and ``b`` its rows'
    velocities at zero impulse, ``b_N < 0`` for an approaching contact. With
    ``A_T = W_T - w_NT w_NT^T / W_N`` and ``c_T = b_T - (b_N / W_N) w_NT``, the
    tangent impulse is ``-s(alpha)``, ``s = (A_T + alpha I)^-1 c_T``, where
    ``alpha = 0`` sticks and a sliding ``alpha > 0`` is the root of
    ``|s| - mu (w_NT . s - b_N) / W_N``.

    Returns:
        ``[alpha, r_N, r_T0, r_T1, iterations, status]`` (:data:`CoulombRowResult`).
    """
    WN = W[0, 0]
    wNT = wp.vec2(W[1, 0], W[2, 0])
    WT = wp.mat22(W[1, 1], W[1, 2], W[2, 1], W[2, 2])
    bN = b[0]
    bT = wp.vec2(b[1], b[2])
    AT = WT - wp.outer(wNT, wNT) / WN
    cT = bT - (bN / WN) * wNT
    # Sticking: alpha = 0.
    s0 = mat22_solve(AT, cT)
    phi0 = wp.length(s0) - mu * (wp.dot(wNT, s0) - bN) / WN
    if phi0 <= 0.0:
        rT_stick = -s0
        rN_stick = -(wp.dot(wNT, rT_stick) + bN) / WN
        return CoulombRowResult(0.0, rN_stick, rT_stick[0], rT_stick[1], 0.0, 0.0)
    # Upper bracket by doubling.
    hi = float(1.0)
    for _ in range(COULOMB_NEWTON_EXPANSIONS):
        vhi = coulomb_root(AT, cT, wNT, bN, WN, mu, hi)
        if vhi[0] < 0.0:
            break
        hi = hi * 2.0
    lo = float(0.0)
    x = 0.5 * (lo + hi)
    iterations = int(0)
    tol = 1.0e-6 + 1.0e-6 * phi0
    last_s = wp.vec2(0.0, 0.0)
    for _ in range(COULOMB_NEWTON_STEPS):
        vd = coulomb_root(AT, cT, wNT, bN, WN, mu, x)
        fx = vd[0]
        dfx = vd[1]
        last_s = wp.vec2(vd[2], vd[3])
        iterations = iterations + 1
        if wp.abs(fx) < tol or wp.abs(hi - lo) < 1.0e-6 * (1.0 + hi):
            break
        if fx > 0.0:
            lo = x
        else:
            hi = x
        x_new = float(0.5) * (lo + hi)
        if dfx != 0.0:
            x_newton = x - fx / dfx
            if x_newton > lo and x_newton < hi:
                x_new = x_newton
        x = x_new
    # Recover r_T, r_N from the last evaluated s.
    rT_final = -last_s
    rN_final = -(wp.dot(wNT, rT_final) + bN) / WN
    return CoulombRowResult(x, rN_final, rT_final[0], rT_final[1], float(iterations), 1.0)


@COULOMB_LAW.law
def coulomb_newton(W: wp.mat33, u: wp.vec3, old: wp.vec3, mu: float, target_n: float) -> wp.vec3:
    """Solve a contact's Coulomb problem with Daviet's Newton, from its rows' current velocities.

    ``u`` are the rows' velocities at the impulses ``old`` and ``target_n`` the
    normal velocity to reach (``-rhs`` of the normal row). The problem's velocities
    at zero impulse are ``u - W old``; a contact that separates there releases its
    impulses, a degenerate block (``W_nn < 1e-20``) keeps them.

    Returns:
        The new normal and tangent impulses.
    """
    G_nn = W[0, 0]
    G_nt1 = W[0, 1]
    G_nt2 = W[0, 2]
    G_t1t1 = W[1, 1]
    G_t1t2 = W[1, 2]
    G_t2t2 = W[2, 2]
    if G_nn < 1.0e-20:
        return old
    u_free_n = u[0] - (G_nn * old[0] + G_nt1 * old[1] + G_nt2 * old[2])
    u_free_t1 = u[1] - (G_nt1 * old[0] + G_t1t1 * old[1] + G_t1t2 * old[2])
    u_free_t2 = u[2] - (G_nt2 * old[0] + G_t1t2 * old[1] + G_t2t2 * old[2])
    if u_free_n < target_n:
        result = coulomb_row(W, wp.vec3(u_free_n - target_n, u_free_t1, u_free_t2), mu)
        return wp.vec3(result[1], result[2], result[3])
    return wp.vec3(0.0)


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


@PROJECTION_LAW.law
def velocity_clamp_impulse(jv: float, limit: float, diagonal: float) -> float:
    """Return the impulse that clamps a coordinate's velocity ``jv`` into ``[-limit, limit]`` (PhysX's PxClamp)."""
    return (wp.min(wp.max(jv, -limit), limit) - jv) / diagonal


@PROJECTION_LAW.law
def physx_drive(
    old: float, jv: float, impulse_multiplier: float, velocity_multiplier: float, target: float, max_impulse: float
) -> float:
    """Return a PhysX drive row's new impulse: its descriptor's affine update, clamped to ``+-max_impulse``.

    The descriptor (``compute_physx_pgs_drive_desc``) folds stiffness, damping and the
    drive target into the multipliers and ``target``; ``jv`` is the row's velocity.
    """
    return box(old * impulse_multiplier + jv * velocity_multiplier + target, -max_impulse, max_impulse)

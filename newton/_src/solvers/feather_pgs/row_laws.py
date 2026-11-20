# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row laws: each row's right-hand side, and the candidates of the prefix row families.

A row solves ``J v + rhs = 0`` (or projects it), so ``rhs`` carries the row's
target and its stabilization. Every row builder and sweep that forms a row's
``rhs`` takes it from these laws, in Warp or in native CUDA (:mod:`.law_source`):

* :func:`row_rhs` is the position-level law of every row type, the canonical
  form of :func:`.row_kernels.row_bias`: ``x * inv_dt`` with
  ``inv_dt = 1 / dt``, the bias scaled by ``bias_scale``, a contact row's
  regularization weight ``contact_w`` on Baumgarte rows only;
* :func:`contact_rhs` is a contact normal row's law: Baumgarte correction of a
  penetrating row and the speculative bias of a separated row, at a ``beta``
  :func:`contact_capped_beta` holds to its bodies' depenetration speed;
* :func:`contact_material_rhs` is a material (compliant) contact normal's law:
  its spring-dashpot bias and its compliance ``eps``; :func:`authored_contact_law`
  gives an authored material's spring and dashpot per unit mass (MuJoCo's rule);
* :func:`contact_restitution_rhs` replaces a closing row's bias by the
  restitution target ``-target + e u`` where :func:`contact_restitution_fires`;
* :func:`joint_limit_side` and :func:`mimic_phi` are the candidates of the prefix families.

The row kinds are the solver's constraint types (:data:`ROW_CONTACT` ...).
"""

import warp as wp

from .law_source import LawFamily

ROW_LAW = LawFamily("RowLaw")
"""The row laws' family; native code calls ``RowLaw::row_rhs(...)``."""

# Row kinds (the solver's constraint types).
ROW_CONTACT = wp.constant(0)
# Kinds 1 and 4 are retired (a free body's speed row and a joint velocity-limit row: a speed limit is the drive's
# motor law, not a constraint).
ROW_FRICTION = wp.constant(2)
ROW_JOINT_LIMIT = wp.constant(3)
# Mimic (joint coupling) row: the bilateral equality ``q_follower - coef1 * q_leader - coef0 = 0`` between two
# coordinates of one articulation, from either of the model's mimic representations
# (:class:`.forest_plan.MimicRelations`); lambda unbounded, Baumgarte drift correction.
ROW_MIMIC = wp.constant(5)
# Link row: one of the six bilateral rows that tie a body's shadow to its primary piece (:mod:`.body_cut`), unbounded
# lambda, Baumgarte drift correction.
ROW_LINK = wp.constant(6)
ROW_TORSION = wp.constant(7)
ROW_KINDS = 8

LINK_ROWS = wp.constant(6)
"""A link's rows: three linear and three angular, whatever joint made it."""

CONTACT_END_GAP_SLOP = wp.constant(1.0e-6)
"""Gap within which a contact counts as reached: absorbs float32 residuals of rows that land exactly at contact."""

SEPARATION_MARGIN = wp.constant(1.0e-5)
SEPARATION_SCALE = wp.constant(2.0**-20)


@ROW_LAW.law
def separates(residual: float, magnitude: float) -> bool:
    """Whether a unilateral row keeps its zero impulse at a motion: its residual there (at the predictor,
    ``J v_hat + b``) stays above ``SEPARATION_MARGIN`` [m/s] plus ``SEPARATION_SCALE`` of its terms' ``magnitude``, so
    the complete path's own rounding of it is positive too. The one verdict of every unilateral row: a contact normal
    (:func:`.row_slots._admit_contact`) and a joint position-limit side (:func:`.prefix_rows.limits`); a row that
    does not separate is marked (it may act), and a world none of whose rows is marked keeps zero impulse (P4)."""
    margin = SEPARATION_MARGIN + SEPARATION_SCALE * magnitude
    return wp.isfinite(residual) and wp.isfinite(margin) and residual >= margin


@ROW_LAW.law
def contact_restitution_fires(phi: float, relative_incident: float, dt: float, threshold: float) -> bool:
    """Return whether a frozen incident impact qualifies for a rebound target.

    Fires only for a closing contact faster than ``threshold`` that is already at
    the surface or reaches it during the step; the end-gap slop absorbs float32
    residuals on rows that land exactly at contact.
    """
    if relative_incident >= -threshold:
        return False
    if phi <= CONTACT_END_GAP_SLOP:
        return True
    return phi + dt * relative_incident <= CONTACT_END_GAP_SLOP


@ROW_LAW.law
def contact_restitution_rhs(
    phi: float, relative_incident: float, target: float, restitution: float, dt: float, threshold: float
) -> tuple[bool, float]:
    """Return whether restitution replaces a contact normal row's bias, and the rebound ``rhs``.

    ``relative_incident`` is the row's frozen incident velocity less its target. A
    firing row solves ``J v = -e u``: its ``rhs`` is ``-target + e u`` and its
    regularization weight 1 (an impact is impulsive, not a spring).
    """
    fires = restitution > 0.0 and contact_restitution_fires(phi, relative_incident, dt, threshold)
    return fires, -target + restitution * relative_incident


@ROW_LAW.law
def contact_rhs(
    phi: float,
    beta: float,
    target: float,
    inv_dt: float,
    bias_scale: float,
    speculative_scale: float,
    contact_w: float,
) -> tuple[float, float]:
    """Return a contact normal row's ``rhs`` and its regularization weight.

    A penetrating row (``phi <= 0``) takes the Baumgarte bias
    ``bias_scale * beta * phi / dt`` and the weight ``contact_w``.
    A separated row may close its gap during the step: it takes
    ``speculative_scale * phi / dt``. Separated rows stay rigid.
    """
    rhs = -target
    row_w = float(1.0)
    if phi <= 0.0:
        rhs += bias_scale * beta * phi * inv_dt
        row_w = contact_w
    else:
        rhs += speculative_scale * phi * inv_dt
    return rhs, row_w


@ROW_LAW.law
def contact_rhs_rate(phi: float, beta: float, inv_dt: float, bias_scale: float, speculative_scale: float) -> float:
    """Return ``|d rhs / d phi|`` [1/s] of :func:`contact_rhs`: the rate at which the row's law turns an error of its gap
    into one of its right-hand side, ``bias_scale beta / dt`` in penetration and ``speculative_scale / dt`` across an
    open gap."""
    rate = speculative_scale * inv_dt
    if phi <= 0.0:
        rate = bias_scale * beta * inv_dt
    return wp.abs(rate)


@ROW_LAW.law
def contact_material_rhs(phi: float, target: float, stiffness: float, damping: float, dt: float) -> tuple[float, float]:
    """Return a material contact normal row's ``rhs`` and its compliance ``eps`` [1/kg].

    The contact pushes with ``F = max(0, -k phi' - c u')`` at the step's end, backward Euler in the force: the row
    ``J v + rhs + eps lambda`` with ``eps = 1 / (dt (dt k + c'))``, ``rhs = k phi / (dt k + c') - target`` and
    ``c' = c`` while ``phi <= 0``; a dashpot does not act across an open gap (``c' = 0``). The row takes neither
    restitution nor the relative regularization: its dashpot is its rebound law, its ``eps`` physical.
    """
    denominator = dt * stiffness + wp.where(phi <= 0.0, damping, 0.0)
    return stiffness * phi / denominator - target, 1.0 / (dt * denominator)


@ROW_LAW.law
def contact_material_rate(phi: float, stiffness: float, damping: float, dt: float) -> float:
    """Return ``|d rhs / d phi|`` [1/s] of :func:`contact_material_rhs`: ``k / (dt k + c')``."""
    denominator = dt * stiffness + wp.where(phi <= 0.0, damping, 0.0)
    return wp.abs(stiffness / denominator)


MATERIAL_TIMECONST = wp.constant(0.02)
MATERIAL_DAMPRATIO = wp.constant(1.0)
"""MuJoCo's default solref ``(timeconst [s], dampratio)``: a shape's whose stiffness or damping is not positive."""
MATERIAL_IMP_MIN = wp.constant(0.9)
MATERIAL_IMP_MAX = wp.constant(0.95)
MATERIAL_IMP_WIDTH = wp.constant(0.001)
MATERIAL_IMP_MID = wp.constant(0.5)
"""MuJoCo's default solimp ``(dmin, dmax, width [m], midpoint, power 2)``: the impedance over a contact's depth."""


@ROW_LAW.law(cuda=False)
def material_solref(stiffness: float, damping: float) -> wp.vec2:
    """Return a shape's MuJoCo ``solref`` from its material: ``(2 / kd, kd / (2 sqrt(ke)))``, Newton's conversion for
    MJWarp (``convert_solref(ke, kd, 1, 1)``), and MuJoCo's default where either value is not positive."""
    if stiffness > 0.0 and damping > 0.0:
        return wp.vec2(2.0 / damping, 0.5 * damping / wp.sqrt(stiffness))
    return wp.vec2(MATERIAL_TIMECONST, MATERIAL_DAMPRATIO)


@ROW_LAW.law(cuda=False)
def material_impedance(phi: float) -> float:
    """Return MuJoCo's impedance at gap ``phi`` [m] on the default solimp curve: ``dmin`` at contact rising to
    ``dmax`` at a depth of ``width``, a power-2 sigmoid through the midpoint (``_contact_kbimp``)."""
    x = wp.min(wp.abs(phi) / MATERIAL_IMP_WIDTH, 1.0)
    y = 2.0 * x * x
    if x > MATERIAL_IMP_MID:
        y = 1.0 - 2.0 * (1.0 - x) * (1.0 - x)
    return MATERIAL_IMP_MIN + y * (MATERIAL_IMP_MAX - MATERIAL_IMP_MIN)


@ROW_LAW.law(cuda=False)
def authored_contact_law(
    stiffness_a: float, damping_a: float, stiffness_b: float, damping_b: float, phi: float, dt: float
) -> tuple[float, float]:
    """Return an authored material contact's spring ``kappa`` [1/s^2] and dashpot ``beta`` [1/s] per unit effective
    mass, MuJoCo's soft contact at rest (DECISIONS 11-16, "Authored materials only").

    MuJoCo's pair rule: the mean of the two shapes' :func:`material_solref` (equal solmix, no priority), its time
    constant at least ``2 dt`` (refsafe). Its row ``k = 1 / (dmax^2 tc^2 dr^2)``, ``b = 2 / (dmax tc)`` with the
    impedance ``imp`` (:func:`material_impedance`) and regularization ``R = (1 - imp) / imp w`` holds a resting
    contact at ``f = -(k imp pos + b v) / R``: the spring ``kappa = k imp^2 / (1 - imp)`` and dashpot
    ``beta = b imp / (1 - imp)`` times the contact's effective mass. The law is the material's normal stiffness,
    independent of the cone (no pyramid factor). :func:`contact_material_rhs` given ``kappa`` and ``beta`` returns
    the row's rhs and its relative compliance ``rho``: ``eps = rho d`` on the row's diagonal ``d``.
    """
    solref = 0.5 * (material_solref(stiffness_a, damping_a) + material_solref(stiffness_b, damping_b))
    timeconst = wp.max(solref[0], 2.0 * dt)
    dampratio = solref[1]
    k = 1.0 / (MATERIAL_IMP_MAX * MATERIAL_IMP_MAX * timeconst * timeconst * dampratio * dampratio)
    b = 2.0 / (MATERIAL_IMP_MAX * timeconst)
    imp = material_impedance(phi)
    share = imp / (1.0 - imp)
    return k * imp * share, b * share


@ROW_LAW.law
def contact_capped_beta(phi: float, beta: float, cap: float, dt: float) -> float:
    """Return a contact normal row's ``beta`` under the depenetration speed ``cap`` [m/s]: the largest value up to
    ``beta`` whose Baumgarte speed ``beta |phi| / dt`` stays within ``cap``. The cap lives in the row's ``beta``, so
    every row store's rhs reads it. A separated row, and a cap that is not positive and finite, keep ``beta``.
    """
    if phi < 0.0 and cap > 0.0 and wp.isfinite(cap):
        return wp.min(beta, -cap * dt / phi)
    return beta


@ROW_LAW.law
def contact_depenetration_cap(body_a: int, body_b: int, max_depenetration_velocity: wp.array[float]) -> float:
    """Return the depenetration speed cap of a contact from its responding bodies (-1: none on that side).

    Body A's cap stands unless body B's is positive, finite and smaller; a cap that
    is not positive and finite applies none (:func:`contact_capped_beta`). Without a
    responding body the cap is ``1e20``.
    """
    cap = float(1.0e20)
    if body_a >= 0:
        cap = max_depenetration_velocity[body_a]
    if body_b >= 0:
        cap_b = max_depenetration_velocity[body_b]
        if cap_b > 0.0 and wp.isfinite(cap_b):
            if cap_b < cap:
                cap = cap_b
    return cap


@ROW_LAW.law
def limit_rhs(
    phi: float, beta: float, target: float, inv_dt: float, bias_scale: float, speculative_scale: float
) -> float:
    """Return a joint-limit row's ``rhs``: Baumgarte on a violated limit, else motion up to the bound.

    A row inside its bound (``phi >= 0``) allows closing it during the step, PhysX's
    ``nextErr`` limit branch, scaled by ``speculative_scale``.
    """
    rhs = -target
    if phi < 0.0:
        rhs += bias_scale * beta * phi * inv_dt
    else:
        rhs += speculative_scale * phi * inv_dt
    return rhs


@ROW_LAW.law
def row_rhs(
    kind: int,
    phi: float,
    beta: float,
    target: float,
    inv_dt: float,
    bias_scale: float,
    contact_speculative_scale: float,
    limit_speculative_scale: float,
    contact_w: float,
) -> tuple[float, float]:
    """Return a row's position-level ``rhs`` and regularization weight, by row kind.

    * contact: :func:`contact_rhs`, its ``beta`` already capped (:func:`contact_capped_beta`);
    * friction (the anchor's tangential displacement), a contact's spin row (its twist displacement), mimic and link
      (bilateral equalities): Baumgarte from both sides, ``bias_scale * beta * phi / dt``;
    * joint limit: :func:`limit_rhs`;
    * other rows: ``-target`` only.
    """
    rhs = -target
    row_w = float(1.0)
    if kind == ROW_CONTACT:
        rhs, row_w = contact_rhs(phi, beta, target, inv_dt, bias_scale, contact_speculative_scale, contact_w)
    elif kind == ROW_FRICTION or kind == ROW_TORSION or kind == ROW_MIMIC or kind == ROW_LINK:
        rhs += bias_scale * beta * phi * inv_dt
    elif kind == ROW_JOINT_LIMIT:
        rhs = limit_rhs(phi, beta, target, inv_dt, bias_scale, limit_speculative_scale)
    return rhs, row_w


@ROW_LAW.law
def joint_limit_side(
    q: float, lower: float, upper: float, side: int, activation_gap: float
) -> tuple[bool, float, float]:
    """Return one side of a coordinate's position limit as a row candidate: ``(active, phi, sign)``.

    Side 0 bounds from below (``phi = q - lower``, ``J = +1``), side 1 from above
    (``phi = upper - q``, ``J = -1``). A side is active when its bound is finite and
    ``q`` lies within ``activation_gap`` of it, violated or not.
    """
    if side == 0:
        return wp.isfinite(lower) and q <= lower + activation_gap, q - lower, 1.0
    return wp.isfinite(upper) and q >= upper - activation_gap, upper - q, -1.0


@ROW_LAW.law
def mimic_phi(q_follower: float, q_leader: float, coef0: float, coef1: float) -> float:
    """Return a mimic row's violation ``q_follower - coef1 q_leader - coef0`` (row ``J = e_f - coef1 e_l``)."""
    return q_follower - coef1 * q_leader - coef0

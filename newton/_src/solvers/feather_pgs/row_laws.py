# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row laws: each row's right-hand side, and the candidates of the prefix row families.

A row solves ``J v + rhs = 0`` (or projects it), so ``rhs`` carries the row's
target and its stabilization. Every row builder and sweep that forms a row's
``rhs`` takes it from these laws, in Warp or in native CUDA (:mod:`.law_source`):

* :func:`row_rhs` is the position-level law of every row type, the canonical
  form of :func:`.row_kernels.compute_world_contact_bias`: ``x * inv_dt`` with
  ``inv_dt = 1 / dt``, the bias scaled by ``bias_scale``, a contact row's
  regularization weight ``contact_w`` on Baumgarte rows only;
* :func:`contact_rhs` is a contact normal row's law: Baumgarte correction of a
  penetrating row and the speculative bias of a separated row, at a ``beta``
  :func:`contact_capped_beta` holds to its bodies' depenetration speed;
* :func:`contact_material_rhs` is a material (compliant) contact normal's law:
  its spring-dashpot bias and its compliance ``eps``;
* :func:`contact_restitution_rhs` replaces a closing row's bias by the
  restitution target ``-target + e u`` where :func:`contact_restitution_fires`;
* :func:`joint_limit_side`, :func:`velocity_limit_active` and :func:`mimic_phi`
  are the candidates of the prefix families.

The row kinds are the solver's constraint types (:data:`ROW_CONTACT` ...,
equal to ``row_kernels.PGS_CONSTRAINT_TYPE_*``).
"""

import warp as wp

from .law_source import LawFamily

ROW_LAW = LawFamily("RowLaw")
"""The row laws' family; native code calls ``RowLaw::row_rhs(...)``."""

# Row kinds (the solver's constraint types).
ROW_CONTACT = wp.constant(0)
ROW_SPEED = wp.constant(1)
ROW_FRICTION = wp.constant(2)
ROW_JOINT_LIMIT = wp.constant(3)
ROW_JOINT_VELOCITY_LIMIT = wp.constant(4)
ROW_MIMIC = wp.constant(5)
ROW_LINK = wp.constant(6)
ROW_TORSION = wp.constant(7)

LINK_ROWS = wp.constant(6)
"""A link's rows: three linear and three angular, whatever joint made it."""

CONTACT_END_GAP_SLOP = wp.constant(1.0e-6)
"""Gap within which a contact counts as reached: absorbs float32 residuals of rows that land exactly at contact."""


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
    * velocity limit and other rows: ``-target`` only.
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
def velocity_limit_active(limit: float, velocity: float, activation_fraction: float) -> bool:
    """Return whether a coordinate's velocity limit takes its row pair (target ``-limit`` on each side).

    A non-positive limit means unlimited. A positive ``activation_fraction`` admits
    only coordinates whose speed reaches that fraction of the limit, PhysX's
    proximity gate.
    """
    if limit <= 0.0:
        return False
    if activation_fraction > 0.0:
        if wp.abs(velocity) < activation_fraction * limit:
            return False
    return True


@ROW_LAW.law
def mimic_phi(q_follower: float, q_leader: float, coef0: float, coef1: float) -> float:
    """Return a mimic row's violation ``q_follower - coef1 q_leader - coef0`` (row ``J = e_f - coef1 e_l``)."""
    return q_follower - coef1 * q_leader - coef0

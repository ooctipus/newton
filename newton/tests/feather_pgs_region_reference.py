# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Float64 reference of the region block law: a contact region solved as one block.

A *region* is one friction patch of one body pair: its contacts, here
*candidates* (a covector over the pair's whitened coordinates, a right-hand
side and a compliance each), and one friction unit (a tangent pair at the
region's anchor, with a spin row where the region has one). The rows are
whitened, so the kinetic metric is the Euclidean one and ``velocity`` below
is the change of the whitened velocity the impulses have made,
``Z^T lambda``.

A *visit* of a region, every other impulse held:

1. The region's own impulses leave the velocity.
2. If no candidate closes faster than its right-hand side allows at that
   velocity, the region holds nothing (*open*).
3. Otherwise *stick first*: the candidates' block with the friction unit held
   at zero residual, one solve (:func:`candidates_block` in the metric
   :func:`held_metric`). It stands when the friction that holds the unit lies
   in the unit's cone at the block's load, the sum of the candidates'
   impulses.
4. Otherwise the region *slides*: the candidates' block with the friction as
   it was, then the unit's cone law at the block's load, once. A sliding
   region is visited again in the next sweep, as a row is today.

The candidates' block is the complementarity problem of the region's normal
rows, solved exactly: ``r = Z v + b + eps lambda >= 0``, ``lambda >= 0``,
``lambda . r = 0``. For rigid candidates it is the projection of the velocity
onto the region's polyhedron in the block's metric: the velocity is unique,
the pooled load and the wrench are, a single candidate's impulse is not. A
compliant candidate (``eps > 0``) carries its own impulse
``-(Z v + b) / eps`` wherever it closes, so every closing candidate is
active and the impulses are unique.

The *pass rule*: a region whose last visit settled it (open or stick) is
passed while the velocity over its support has the bits that visit left. In
exact arithmetic the visit of such a region changes nothing (steps 1 to 3
return the state they started from), so passing it is the visit.

This module holds the reference and its helpers; it defines no tests.
"""

import dataclasses

import numpy as np

CLOSING = 1.0e-12
"""A candidate closes when its residual is below this share of the visit's scale (its largest right-hand side or row
speed): the rounding floor of the block's solve in float64."""


@dataclasses.dataclass
class FrictionUnit:
    """A region's friction unit: ``rows`` (k, D) the tangent pair and, with ``k = 3``, the spin row; their
    right-hand sides; the cone's coefficients ``(mu, mu[, mu_t])``."""

    rows: np.ndarray
    rhs: np.ndarray
    coef: np.ndarray


@dataclasses.dataclass
class Region:
    """A region: its candidates' covectors ``rows`` (n, D), right-hand sides, compliances (0: rigid) and unit."""

    rows: np.ndarray
    rhs: np.ndarray
    eps: np.ndarray
    unit: FrictionUnit | None = None

    @property
    def support(self) -> np.ndarray:
        """The coordinates the region's rows touch: its bodies'."""
        touched = np.abs(self.rows).sum(axis=0)
        if self.unit is not None:
            touched = touched + np.abs(self.unit.rows).sum(axis=0)
        return np.flatnonzero(touched)


@dataclasses.dataclass
class RegionState:
    """What a region keeps across its visits: the candidates' impulses, the unit's, and the velocity over its support
    as its last settled visit left it (None: not settled)."""

    lam: np.ndarray
    phi: np.ndarray
    settled: np.ndarray | None = None


def new_states(regions: list[Region]) -> list[RegionState]:
    return [RegionState(np.zeros(len(g.rhs)), np.zeros(0 if g.unit is None else len(g.unit.rhs))) for g in regions]


def project(B: np.ndarray, c: np.ndarray, warm: np.ndarray | None = None, max_steps: int = 1000):
    """The point ``x = B^T lam`` nearest the origin with ``B x + c >= 0``, ``lam >= 0`` its multipliers: the dual
    active set of Goldfarb and Idnani, from the set ``warm``. The set's rows stay linearly independent: a row that
    depends on the set enters only as a row of the set leaves, so rank-deficient candidates (four coplanar contacts)
    are solved exactly. Returns ``(lam, steps, bounded)``; ``bounded`` is False where no point satisfies the rows."""
    n = len(c)
    lam = np.zeros(n)
    active: list[int] = []
    steps = 0
    scale = max(float(np.abs(c).max()) if n else 0.0, 1.0e-300)
    if warm is not None and warm.any():
        active = [int(i) for i in np.flatnonzero(warm)]
        while active:
            rows = B[active]
            mu = np.linalg.lstsq(rows @ rows.T, -c[active], rcond=1.0e-13)[0]
            if (mu > 0.0).all():
                lam[active] = mu
                break
            active.pop(int(np.argmin(mu)))
            steps += 1
    x = B.T @ lam
    while steps < max_steps:
        r = B @ x + c
        r[active] = 0.0
        j = int(np.argmin(r))
        if r[j] >= -CLOSING * scale:
            return lam, steps, True
        row = B[j]
        entering = 0.0
        while True:
            rows = B[active]
            d = np.linalg.solve(rows @ rows.T, rows @ row) if active else np.zeros(0)
            direction = row - rows.T @ d
            along = float(row @ direction)
            primal = -(float(row @ x) + c[j]) / along if along > 1.0e-12 * float(row @ row) else np.inf
            leaving = np.flatnonzero(d > 0.0)
            ratios = lam[active][leaving] / d[leaving]
            dual = float(ratios.min()) if len(leaving) else np.inf
            step = min(primal, dual)
            if not np.isfinite(step):
                return lam, steps, False
            lam[active] -= step * d
            entering += step
            steps += 1
            if primal < np.inf:
                x = x + step * direction
            if primal <= dual:
                active.append(j)
                lam[j] = entering
                break
            out = active.pop(int(leaving[int(np.argmin(ratios))]))
            lam[out] = 0.0
    return lam, steps, False


def held_metric(unit: FrictionUnit | None, dim: int) -> np.ndarray:
    """``K = I - F^T (F F^T)^-1 F``: the response to an impulse with the friction unit held at zero residual."""
    if unit is None:
        return np.eye(dim)
    F = unit.rows
    return np.eye(dim) - F.T @ np.linalg.solve(F @ F.T, F)


def candidates_block(region: Region, start: np.ndarray, metric: np.ndarray, warm: np.ndarray | None = None):
    """The candidates' block from the velocity ``start`` in ``metric`` (a projector: the identity or
    :func:`held_metric`): the impulses ``lam`` with ``r = Z (start + K Z^T lam) + b + eps lam >= 0``, ``lam >= 0``,
    ``lam . r = 0``. It is a projection (:func:`project`) over the velocity and, for the compliant candidates, their
    own impulses scaled by ``sqrt(eps)``. Returns ``(lam, velocity, steps, bounded)``."""
    Z = region.rows
    rows = np.hstack([Z @ metric, np.diag(np.sqrt(region.eps))])
    lam, steps, bounded = project(rows, Z @ start + region.rhs, warm)
    return lam, start + metric @ (Z.T @ lam), steps, bounded


def compliant_block(region: Region, start: np.ndarray, metric: np.ndarray, max_steps: int = 60):
    """:func:`candidates_block` for a region whose candidates are all compliant, in the form a kernel takes: Newton on
    the velocity, a sum over the candidates a step. A compliant candidate that closes at ``v`` holds
    ``lam = -(z . v + b) / eps``, so with ``C`` the closing candidates the block is
    ``(I + K S) v = start - K q``, ``S = sum_C z z^T / eps``, ``q = sum_C z b / eps``: one pass over the candidates
    and one solve in the velocity's coordinates, however many candidates close. The step is exact once the closing
    set stands at its own solution; before that it is shortened to lower the block's energy
    ``1/2 |v - start|^2 + 1/2 sum max(0, -(z . v + b))^2 / eps``, which the solution minimizes.
    Returns ``(lam, velocity, steps)``."""
    Z, rhs, eps = region.rows, region.rhs, region.eps
    identity = np.eye(len(start))

    def energy(v):
        closing_speed = np.minimum(Z @ v + rhs, 0.0)
        return 0.5 * float((v - start) @ (v - start)) + 0.5 * float((closing_speed * closing_speed / eps).sum())

    velocity = start.copy()
    steps = 0
    while steps < max_steps:
        steps += 1
        closing = Z @ velocity + rhs < 0.0
        rows = Z[closing]
        S = (rows.T / eps[closing]) @ rows
        q = rows.T @ (rhs[closing] / eps[closing])
        target = np.linalg.solve(identity + metric @ S, start - metric @ q)
        if np.array_equal(Z @ target + rhs < 0.0, closing):
            velocity = target
            break
        share, before = 1.0, energy(velocity)
        while share > 1.0e-12 and energy(velocity + share * (target - velocity)) > before:
            share *= 0.5
        velocity = velocity + share * (target - velocity)
    speed = Z @ velocity + rhs
    return np.where(speed < 0.0, -speed / eps, 0.0), velocity, steps


def cone_law(unit: FrictionUnit, residual: np.ndarray, old: np.ndarray, load: float) -> np.ndarray:
    """The unit's block at a fixed load: the minimizer of ``1/2 d^T G d + residual . d`` over the impulses
    ``old + d`` in the elliptic cone ``|impulse / coef| <= load`` (``projection_laws.friction_triple``, exactly)."""
    if load <= 0.0:
        return np.zeros_like(old)
    coef = unit.coef
    A = (unit.rows @ unit.rows.T) * np.outer(coef, coef)
    y_old = old / coef
    g = residual * coef
    y = y_old - np.linalg.solve(A, g)
    if y @ y <= load * load:
        return y * coef
    values, vectors = np.linalg.eigh(A)
    rhs = vectors.T @ (A @ y_old - g)
    lo, hi = 0.0, float(np.linalg.norm(rhs)) / load
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(rhs / (values + mid)) > load:
            lo = mid
        else:
            hi = mid
    return (vectors @ (rhs / (values + hi))) * coef


def visit(region: Region, state: RegionState, velocity: np.ndarray) -> str:
    """Visit ``region`` in place (``velocity`` and ``state``); returns ``open``, ``stick`` or ``slide``."""
    Z, unit = region.rows, region.unit
    warm = state.lam > 0.0
    own = Z.T @ state.lam
    friction_old = np.zeros_like(velocity) if unit is None else unit.rows.T @ state.phi
    u0 = velocity - own - friction_old
    speed = Z @ u0
    scale = max(float(np.abs(region.rhs).max()), float(np.abs(speed).max()), 1.0e-300)
    if not (speed + region.rhs < -CLOSING * scale).any():
        state.lam[:] = 0.0
        state.phi[:] = 0.0
        velocity[:] = u0
        return "open"
    if unit is None:
        lam, u, _, _ = candidates_block(region, u0, np.eye(len(velocity)), warm)
        state.lam[:] = lam
        velocity[:] = u
        return "stick"
    F = unit.rows
    G = F @ F.T
    K = held_metric(unit, len(velocity))
    rf0 = F @ u0 + unit.rhs
    lam, _, _, bounded = candidates_block(region, u0 - F.T @ np.linalg.solve(G, rf0), K, warm)
    wrench = Z.T @ lam
    load = float(lam.sum())
    phi = -np.linalg.solve(G, rf0 + F @ wrench)
    scaled = phi / unit.coef
    if bounded and load > 0.0 and scaled @ scaled <= load * load:
        state.lam[:] = lam
        state.phi[:] = phi
        velocity[:] = u0 + wrench + F.T @ phi
        return "stick"
    lam, u, _, _ = candidates_block(region, u0 + friction_old, np.eye(len(velocity)), lam > 0.0)
    phi = cone_law(unit, F @ u + unit.rhs, state.phi, float(lam.sum()))
    velocity[:] = u + F.T @ (phi - state.phi)
    state.lam[:] = lam
    state.phi[:] = phi
    return "slide"


def sweep(regions: list[Region], states: list[RegionState], velocity: np.ndarray, passing: bool = True) -> list[str]:
    """One sweep: the regions in their order; returns each region's outcome (``passed`` under the pass rule)."""
    outcomes = []
    for region, state in zip(regions, states, strict=True):
        support = region.support
        if passing and state.settled is not None and np.array_equal(velocity[support], state.settled):
            outcomes.append("passed")
            continue
        outcome = visit(region, state, velocity)
        state.settled = None if outcome == "slide" else velocity[support].copy()
        outcomes.append(outcome)
    return outcomes


def solve(regions: list[Region], dim: int, sweeps: int, passing: bool = True):
    """``sweeps`` sweeps from rest; returns ``(velocity, states, outcomes a sweep)``."""
    velocity = np.zeros(dim)
    states = new_states(regions)
    log = [sweep(regions, states, velocity, passing) for _ in range(sweeps)]
    return velocity, states, log


def scalar_sweep(regions: list[Region], states: list[RegionState], velocity: np.ndarray) -> None:
    """Today's unit law in this module's order, for comparison: each candidate a scalar row
    (``lambda <- max(0, lambda - r / (|z|^2 + eps))``), then the region's friction unit at the pooled load."""
    for region, state in zip(regions, states, strict=True):
        for i in range(len(region.rhs)):
            z = region.rows[i]
            r = z @ velocity + region.rhs[i] + region.eps[i] * state.lam[i]
            new = max(state.lam[i] - r / (z @ z + region.eps[i]), 0.0)
            velocity += z * (new - state.lam[i])
            state.lam[i] = new
        unit = region.unit
        if unit is not None:
            phi = cone_law(unit, unit.rows @ velocity + unit.rhs, state.phi, float(state.lam.sum()))
            velocity += unit.rows.T @ (phi - state.phi)
            state.phi[:] = phi


def kkt(regions: list[Region], states: list[RegionState], velocity: np.ndarray) -> tuple[float, float]:
    """``(D, G)``: the largest closing speed left on a candidate past its right-hand side, and the complementarity
    gap ``sum |lambda r|`` of the normals with the friction units' ``|phi . r + load |coef r||``."""
    violation, gap = 0.0, 0.0
    for region, state in zip(regions, states, strict=True):
        r = region.rows @ velocity + region.rhs + region.eps * state.lam
        violation = max(violation, float(np.maximum(-r, 0.0).max()))
        gap += float(np.abs(state.lam * r).sum())
        unit = region.unit
        if unit is not None:
            rf = unit.rows @ velocity + unit.rhs
            gap += abs(float(state.phi @ rf) + float(state.lam.sum()) * float(np.linalg.norm(unit.coef * rf)))
    return violation, gap


def body_rows(mass: float, inertia: np.ndarray, points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Whitened covectors of unit ``directions`` at ``points`` (relative to the mass centre) of a free body with a
    diagonal ``inertia``: ``[d / sqrt(m), (p x d) / sqrt(I)]``; the row's speed is ``d . (v + w x p)``."""
    points, directions = np.atleast_2d(points), np.atleast_2d(directions)
    return np.hstack([directions / np.sqrt(mass), np.cross(points, directions) / np.sqrt(inertia)])


def spin_row(inertia: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The whitened covector of the angular speed about ``axis``."""
    return np.concatenate([np.zeros(3), axis / np.sqrt(inertia)])


def unwhiten(mass: float, inertia: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A free body's linear and angular velocity change from its whitened one."""
    return velocity[:3] / np.sqrt(mass), velocity[3:6] / np.sqrt(inertia)

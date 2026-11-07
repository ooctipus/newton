# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""References of the contact torsion and contact compliance scenes of :mod:`newton.tests.feather_pgs_scenes`.

Plain numpy in SI units. Each reference is one of:

- analytic: a closed form of the continuum law the scene obeys;
- converged: the continuum law integrated with a step far below the scene's, or by quadrature;
- discrete: the scene's own time discretization of the law, solved exactly: what a converged solver reproduces at the
  scene's step.

Torsion laws of a contact of normal load ``N``, friction coefficient ``mu`` and effective spin radius ``r``, with
tangential impulse ``t`` and spin impulse ``s``:

- ``elliptic``: ``|t|^2 + (s / r)^2 <= (mu N)^2``, solved jointly with maximal dissipation (the soft-finger contact
  of grasp analysis, MuJoCo's elliptic cone with ``mu_torsion = mu r``);
- ``linear``: ``|t| + |s| / r <= mu N`` with the friction pair solved first and the spin taking its leftover
  ``r (mu N - |t|)``, the fixed point of a pair-first block Gauss-Seidel;
- ``patch``: the exact limit surface of a flat circular patch of radius ``a`` under a pressure law, by quadrature.
  A uniform patch's pure-spin capacity is ``(2/3) mu N a``, so ``a = 1.5 r`` matches the laws' ``mu N r``.
"""

from __future__ import annotations

import math

import numpy as np

GRAVITY = 9.81


def pinch_twist(mass, offset, mu, squeeze, radius, inertia_cm, *, law="elliptic", g=GRAVITY):
    """Analytic soft-finger pinch: two point contacts on the pinch axis, each pressed by ``squeeze`` [N], hold a body
    whose centre lies ``offset`` [m] off the axis. Each carries ``m g / 2`` tangentially, and gravity twists the body
    about the normals with ``m g d``.

    The two contacts' twist capacity: ``elliptic`` ``2 r sqrt((mu F)^2 - (m g / 2)^2)``, ``linear``
    ``2 r (mu F - m g / 2)``, ``pure`` (no carried load) ``2 mu F r``; zero without torsion.
    Returns (capacity [N m], gravity torque [N m], held, angular acceleration about the axis [rad/s^2]); the body
    pivots about the axis, ``I = I_cm + m d^2``.
    """
    carried, cone = 0.5 * mass * g, mu * squeeze
    capacity = {
        "elliptic": 2.0 * radius * math.sqrt(max(cone * cone - carried * carried, 0.0)),
        "linear": 2.0 * radius * max(cone - carried, 0.0),
        "pure": 2.0 * radius * cone,
    }[law]
    torque = mass * g * offset
    held = torque <= capacity
    alpha = 0.0 if held else (torque - capacity) / (inertia_cm + mass * offset * offset)
    return capacity, torque, held, alpha


def patch_points(a, pressure="uniform", rings=48, sectors=96):
    """Quadrature points of a circular patch of radius ``a`` and their load weights (summing to 1): the midpoint
    rule in radius and angle under ``uniform`` or ``hertz`` (``p ~ sqrt(1 - rho^2 / a^2)``) pressure."""
    rho = (np.arange(rings) + 0.5) / rings * a
    theta = (np.arange(sectors) + 0.5) / sectors * 2.0 * np.pi
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    weight = rr if pressure == "uniform" else rr * np.sqrt(np.maximum(1.0 - (rr / a) ** 2, 0.0))
    weight = weight / weight.sum()
    return np.stack([rr * np.cos(tt), rr * np.sin(tt)], axis=-1).reshape(-1, 2), weight.reshape(-1)


def patch_capacity(a, pressure="uniform"):
    """Pure-spin torque capacity of a patch per unit ``mu N``: ``(2/3) a`` uniform, ``(3 pi / 16) a`` Hertz."""
    points, weight = patch_points(a, pressure)
    return float((weight * np.hypot(points[:, 0], points[:, 1])).sum())


def _friction(v, w, law, mu_n, radius, patch):
    """Friction force (against ``v``) and torque (against ``w``) of one contact sliding at ``v`` and spinning at
    ``w`` under ``law``."""
    if v <= 0.0 and w <= 0.0:
        return 0.0, 0.0
    if law == "elliptic":
        # Maximal dissipation on the ellipse: (t, s / r) along (v, r w).
        s = math.hypot(v, radius * w)
        return mu_n * v / s, mu_n * radius * radius * w / s
    if law == "linear":
        return (mu_n, 0.0) if v > 0.0 else (0.0, mu_n * radius)
    points, weight = patch
    # Each patch point slides at (v, 0) + w z x rho = (v - w y, w x).
    vx, vy = v - w * points[:, 1], w * points[:, 0]
    speed = np.hypot(vx, vy)
    speed = np.where(speed > 0.0, speed, 1.0)
    ux, uy = vx / speed, vy / speed
    return mu_n * float((weight * ux).sum()), mu_n * float((weight * (points[:, 0] * uy - points[:, 1] * ux)).sum())


def slide_spin(v0, w0, mu, mass, inertia, radius, law, *, pressure="uniform", dt=5.0e-5, t_end=1.0, g=GRAVITY):
    """Converged trajectory of a body sliding at ``v0`` and spinning at ``w0`` on one flat contact of load ``m g``
    under ``law`` (``linear``, ``elliptic`` or ``patch``: a patch of equal spin capacity, ``a = 1.5 r`` uniform or
    ``16 r / (3 pi)`` Hertz), by the midpoint rule; a speed whose rate would carry it past zero stops there.
    Returns dict(t, v, w, t_v (sliding stop), t_w (spin stop))."""
    patch = None
    if law == "patch":
        patch = patch_points(radius * (1.5 if pressure == "uniform" else 16.0 / (3.0 * math.pi)), pressure)
    mu_n = mu * mass * g
    steps = int(round(t_end / dt))
    t = np.arange(steps + 1) * dt
    v, w = np.zeros(steps + 1), np.zeros(steps + 1)
    v[0], w[0] = v0, w0
    stops = {}

    def rate(vv, ww):
        force, torque = _friction(max(vv, 0.0), max(ww, 0.0), law, mu_n, radius, patch)
        return -force / mass, -torque / inertia

    for i in range(steps):
        a1, b1 = rate(v[i], w[i])
        a2, b2 = rate(v[i] + 0.5 * dt * a1, w[i] + 0.5 * dt * b1)
        v[i + 1] = 0.0 if v[i] + dt * a1 <= 0.0 else max(v[i] + dt * a2, 0.0)
        w[i + 1] = 0.0 if w[i] + dt * b1 <= 0.0 else max(w[i] + dt * b2, 0.0)
        for name, value in (("t_v", v[i + 1]), ("t_w", w[i + 1])):
            if value == 0.0:
                stops.setdefault(name, t[i + 1])
    return {"t": t, "v": v, "w": w, "t_v": stops.get("t_v"), "t_w": stops.get("t_w")}


def compliant_drop(mass, stiffness, damping, height, *, dt, steps, g=GRAVITY):
    """Discrete reference of a mass dropped from ``height`` [m] above first touch onto a unilateral spring-dashpot
    ground: the material law ``F = max(0, -k phi - c u)`` (no dashpot across an open gap) in backward Euler, the
    position following the new velocity.

    Each step solves ``u+ + b + gamma lambda = 0``, ``u+ = u - g dt + lambda / m``, ``lambda >= 0``, with
    ``gamma = 1 / (dt (dt k + c'))``, ``b = k phi / (dt k + c')`` and ``c' = c`` while ``phi <= 0``, else 0.
    Returns the gap ``phi`` and velocity ``u`` at the ``steps + 1`` step boundaries.
    """
    phi, u = np.zeros(steps + 1), np.zeros(steps + 1)
    phi[0] = height
    for i in range(steps):
        denominator = dt * stiffness + (damping if phi[i] <= 0.0 else 0.0)
        gamma, bias = 1.0 / (dt * denominator), stiffness * phi[i] / denominator
        free = u[i] - g * dt
        impulse = max(0.0, -(free + bias) / (1.0 / mass + gamma))
        u[i + 1] = free + impulse / mass
        phi[i + 1] = phi[i] + dt * u[i + 1]
    return phi, u


def drop_metrics(t, phi, u, height, g=GRAVITY):
    """First-impact metrics of a drop: ``e`` (the speed leaving first contact over the analytic impact speed
    ``sqrt(2 g h)``; 0 if it never leaves), the peak penetration of the first contact and the resting gap (the mean of
    the last tenth)."""
    impact = math.sqrt(2.0 * g * height)
    inside = np.flatnonzero(phi <= 0.0)
    if inside.size == 0:
        return {"e": math.nan, "peak": 0.0, "rest": float(phi[-1])}
    first = inside[0]
    after = np.flatnonzero(phi[first:] > 0.0)
    leave = first + after[0] if after.size else len(phi)
    e = float(u[leave]) / impact if after.size else 0.0
    return {"e": e, "peak": float(-phi[first:leave].min()), "rest": float(np.mean(phi[-max(1, len(phi) // 10) :]))}


def plate_on_springs(mass, half_x, half_y, com_x, com_y, stiffness, g=GRAVITY):
    """Analytic split of a rigid plate on four equal springs at its corners ``(+-a, +-b)`` loaded at
    ``(e_x, e_y)``: ``F_i = m g [1/4 + e_x x_i / (4 a^2) + e_y y_i / (4 b^2)]``.

    Returns the corners [4, 2] (``-x-y``, ``+x-y``, ``-x+y``, ``+x+y``), their forces [4] and the mean indentation
    ``m g / 4k``.
    """
    corners = np.array([[-half_x, -half_y], [half_x, -half_y], [-half_x, half_y], [half_x, half_y]])
    weight = mass * g
    forces = weight * (0.25 + com_x * corners[:, 0] / (4 * half_x**2) + com_y * corners[:, 1] / (4 * half_y**2))
    return corners, forces, weight / (4.0 * stiffness)


def stack_compression(count, mass, stiffness, corners=4, g=GRAVITY):
    """Analytic compression of each interface of a column of ``count`` equal boxes on compliant ground with
    ``corners`` equal springs per interface: interface ``j`` (0 the ground) compresses ``(count - j) m g / (n k)``."""
    return np.array([(count - j) * mass * g / (corners * stiffness) for j in range(count)])


def drive_rollout(inertia, ke, kd, effort, target, *, dt, steps):
    """The exact discrete step of one hinge on an effort-limited implicit drive, from rest at zero (float64).

    The drive energy is Huber: ``I dv = h clamp(u0 - (K / h) dv, +-U)`` with ``K = h (h ke + kd)`` and ``u0 = -(ke (q - q* +
    h v) + kd v)``, whose unique solution is the unsaturated step when its force lies inside the limit, else
    ``dv = h sign(u0) U / I``. ``effort`` 0 is no limit. Returns ``(q, v)`` after each step.
    """
    q, v = 0.0, 0.0
    stiffness = dt * (dt * ke + kd)
    history = []
    for _ in range(steps):
        u0 = -(ke * (q - target + dt * v) + kd * v)
        dv = dt * u0 / (inertia + stiffness)
        force = u0 - stiffness / dt * dv
        if effort > 0.0 and abs(force) > effort:
            dv = dt * math.copysign(effort, force) / inertia
        v += dv
        q += dt * v
        history.append((q, v))
    return np.asarray(history)


def spring_rollout(omega, q0, *, dt, steps):
    """Symplectic Euler of an undamped oscillator ``q'' = -omega^2 q`` (float64): ``(q, v)`` after each step."""
    q, v = float(q0), 0.0
    history = []
    for _ in range(steps):
        v -= dt * omega * omega * q
        q += dt * v
        history.append((q, v))
    return np.asarray(history)


def rolling_speed(v0, gyration):
    """The speed a body sliding at ``v0`` without spin ends rolling at, ``v0 / (1 + k)`` with ``I = k m r^2``: 5/7 for a
    solid sphere, 2/3 for a disc (Coulomb's work is independent of ``mu``)."""
    return v0 / (1.0 + gyration)


def tipping_force(mass, width, height, g=GRAVITY):
    """The force at ``height`` above the support that lifts a body of base ``width`` about its far edge."""
    return mass * g * width / (2.0 * height)


def patch_footprint(half_x, half_y, cells=64):
    """Quadrature points of a uniform rectangular patch, with equal load weights."""
    x = ((np.arange(cells) + 0.5) / cells - 0.5) * 2.0 * half_x
    y = ((np.arange(cells) + 0.5) / cells - 0.5) * 2.0 * half_y
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.stack([xx, yy], -1).reshape(-1, 2)
    return points, np.full(points.shape[0], 1.0 / points.shape[0])


def patch_slide_spin(half_x, half_y, speed, spin, mu, *, g=GRAVITY, dt=2.0e-4, t_end=1.2):
    """A flat uniform box sliding at ``speed`` along x and spinning at ``spin`` about z under Coulomb friction at every
    point of its footprint (quadrature, a body of unit mass; the motion is independent of the mass).

    Returns ``(x [m], yaw [rad], stop [s])``, the distance at the end, the yaw swept and the time the spin stops.
    """
    points, weights = patch_footprint(half_x, half_y)
    inertia = ((2.0 * half_x) ** 2 + (2.0 * half_y) ** 2) / 12.0
    v, w, x, theta, t, stop = np.array([speed, 0.0]), float(spin), np.zeros(2), 0.0, 0.0, None
    while t < t_end and (np.linalg.norm(v) > 1.0e-3 or abs(w) > 1.0e-2):
        c, s = math.cos(theta), math.sin(theta)
        r = points @ np.array([[c, s], [-s, c]])
        u = np.stack([v[0] - w * r[:, 1], v[1] + w * r[:, 0]], -1)
        force = -(weights[:, None] * u / (np.linalg.norm(u, axis=1, keepdims=True) + 1.0e-9)) * mu * g
        v = v + dt * force.sum(0)
        w += dt * float(np.sum(r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0])) / inertia
        x += dt * v
        theta += dt * w
        t += dt
        if stop is None and abs(w) < 1.0e-1:
            stop = t
    return float(np.linalg.norm(x)), theta, t if stop is None else stop


def pendulum_rollout(lengths, masses, angles, *, t_end, g=GRAVITY, dt=1.0e-4):
    """A planar chain of point masses on rigid massless links in gravity, by RK4 in float64 (one or two links).

    ``angles`` are each link's angle from the downward vertical at rest in time. Returns the angles at ``t_end``.
    """
    l1 = lengths[0]
    m1 = masses[0]
    two = len(lengths) == 2
    l2, m2 = (lengths[1], masses[1]) if two else (0.0, 0.0)

    def deriv(y):
        a1, a2, w1, w2 = y
        delta = a1 - a2
        total = m1 + m2
        h = np.array(
            [[total * l1 * l1, m2 * l1 * l2 * math.cos(delta)], [m2 * l1 * l2 * math.cos(delta), m2 * l2 * l2]]
        )
        f = np.array(
            [
                -m2 * l1 * l2 * w2 * w2 * math.sin(delta) - total * g * l1 * math.sin(a1),
                m2 * l1 * l2 * w1 * w1 * math.sin(delta) - m2 * g * l2 * math.sin(a2),
            ]
        )
        if not two:
            accel = np.array([f[0] / h[0, 0], 0.0])
        else:
            accel = np.linalg.solve(h, f)
        return np.array([w1, w2, accel[0], accel[1]])

    y = np.array([angles[0], angles[1] if two else 0.0, 0.0, 0.0])
    for _ in range(round(t_end / dt)):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y[:2] if two else y[:1]

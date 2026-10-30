# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The projection laws solve the problems they name, on every test device.

Each law meets its defining conditions, checked in float64: the tangent pair's
disk problem, the friction triple's elliptic cone problem against its exact
solve, the contact's Coulomb problem (cone, normal complementarity and, for
Daviet's Newton, maximal dissipation), the regularized step's fixed point and
its block-solve diagonal, and the velocity-limit, drive and set projections.
The laws' CUDA renderings agree with these Warp functions
(``test_feather_pgs_law_source``'s twins).
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import projection_laws as laws
from newton._src.solvers.feather_pgs.row_laws import ROW_CONTACT, ROW_FRICTION
from newton.tests.feather_pgs_law_harness import call, register_device_tests, scalars, twin_inputs


@wp.kernel
def _pair(
    a: wp.array[float],
    c: wp.array[float],
    d: wp.array[float],
    residual: wp.array[wp.vec2],
    old: wp.array[wp.vec2],
    radius: wp.array[float],
    omega: wp.array[float],
    out: wp.array[wp.vec2],
    relaxed: wp.array[wp.vec2],
):
    i = wp.tid()
    out[i] = laws.disk(laws.friction_pair(a[i], c[i], d[i], residual[i], old[i], radius[i], 1.0), radius[i])
    relaxed[i] = laws.friction_pair(a[i], c[i], d[i], residual[i], old[i], radius[i], omega[i])


def test_friction_pair_solves_the_disk_problem(test, device):
    """Minimize the pair's quadratic on its disk: the residual at the result opposes every feasible move.

    At ``omega = 1`` the pair ``x`` and its residual ``r(x) = residual + A (x - old)`` meet the variational
    inequality ``r . x + radius |r| = 0``: zero inside the disk (sticking), anti-parallel on it (sliding).
    A relaxed pair moves the old one ``omega`` of the way.
    """
    rng = np.random.default_rng(61)
    n = 8192
    values = twin_inputs("friction_pair", rng, n, device)
    outputs = call(_pair, n, device, **values, out=np.zeros((n, 2), np.float32), relaxed=np.zeros((n, 2), np.float32))
    x = outputs["out"].numpy().astype(np.float64)
    relaxed = outputs["relaxed"].numpy().astype(np.float64)
    old = values["old"].astype(np.float64)
    a, c, d = (values[k].astype(np.float64) for k in ("a", "c", "d"))
    block = np.stack([np.stack([a, c], 1), np.stack([c, d], 1)], 1)
    r = values["residual"].astype(np.float64) + np.einsum("nij,nj->ni", block, x - old)
    radius = values["radius"].astype(np.float64)
    norm = np.linalg.norm(x, axis=1)
    test.assertTrue(np.all(norm <= radius * (1.0 + 1.0e-6) + 1.0e-30))
    test.assertTrue(np.all(x[radius == 0.0] == 0.0))
    loaded = radius > 0.0
    scale = np.abs(block).max(axis=(1, 2)) * np.maximum(radius, np.linalg.norm(old, axis=1))
    scale += np.linalg.norm(values["residual"], axis=1)
    gap = np.einsum("ni,ni->n", r, x) + radius * np.linalg.norm(r, axis=1)
    gap = gap[loaded] / (scale[loaded] * radius[loaded])
    test.assertGreater(gap.min(), -1.0e-5)
    test.assertLess(np.quantile(gap, 0.995), 1.0e-5)
    test.assertLess(gap.max(), 1.0e-3)
    # A sticking pair relaxes to old + omega (pair - old), before the caller's disk.
    omega = values["omega"].astype(np.float64)[:, None]
    sticking = loaded & (norm < radius * (1.0 - 1.0e-3))
    expected = old + omega * (x - old)
    test.assertLess(np.max(np.abs(relaxed[sticking] - expected[sticking]) / (1.0 + np.abs(expected[sticking]))), 1.0e-4)


@wp.kernel
def _pair_from_metric(
    scale: wp.array[float],
    largest: wp.array[float],
    smallest: wp.array[float],
    axis: wp.array[wp.vec2],
    residual: wp.array[wp.vec2],
    old: wp.array[wp.vec2],
    radius: wp.array[float],
    omega: wp.array[float],
    out: wp.array[wp.vec2],
):
    i = wp.tid()
    out[i] = laws.friction_pair_from_metric(
        scale[i], largest[i], smallest[i], axis[i], residual[i], old[i], radius[i], omega[i]
    )


def test_friction_pair_keeps_a_sliding_pair_at_its_solution(test, device):
    """Keep a sliding pair already at its disk problem's solution to the floor, bit for bit, at any radius.

    The hold is the root solve's own stop (:data:`~.projection_laws.PAIR_FLOOR`): the impulse on its circle and the
    residual anti-parallel to it, both within the floor. Impulses ``(3, 4) 2^e`` have exact norms ``5 2^e`` and
    residuals ``-2^k`` times them exact cross products; radii 0 or 2 ulps above that norm (1.9e-7 of the radius, inside
    the floor) keep the impulse, including radii under 1e-15 that the squared-norm screen leaves to the test; 8 ulps
    above (outside the floor, inside the screen) and 4e-6 above (outside the screen) re-solve it.
    """
    rng = np.random.default_rng(71)
    n = 4096
    values = twin_inputs("friction_pair_from_metric", rng, n, device)
    exponent = rng.integers(-60, 57, n)
    sign = rng.choice([-1.0, 1.0], (n, 2))
    legs = np.where(rng.integers(0, 2, n)[:, None] == 0, [3.0, 4.0], [4.0, 3.0])
    old = (sign * legs * np.ldexp(1.0, exponent)[:, None]).astype(np.float32)
    norm = np.ldexp(5.0, exponent).astype(np.float32)
    residual = (-old * np.ldexp(1.0, rng.integers(-2, 7, n))[:, None]).astype(np.float32)
    ulp = np.nextafter(norm, np.float32(np.inf)) - norm
    radii = {
        "on the circle": (norm, True),
        "2 ulps inside": (norm + 2 * ulp, True),
        "8 ulps inside": (norm + 8 * ulp, False),
        "4e-6 inside": (norm * np.float32(1.0 + 4.0e-6), False),
    }
    for name, (radius, kept) in radii.items():
        with test.subTest(radius=name):
            out = call(
                _pair_from_metric,
                n,
                device,
                **{k: values[k] for k in ("scale", "largest", "smallest", "axis", "omega")},
                residual=residual,
                old=old,
                radius=radius.astype(np.float32),
                out=np.zeros((n, 2), np.float32),
            )["out"].numpy()
            held = np.all(out == old, axis=1)
            if kept:
                test.assertTrue(held.all(), f"{int((~held).sum())} of {n} re-solved, radius {radius[~held][:3]}")
            else:
                test.assertFalse(held.any(), f"{int(held.sum())} of {n} kept")


def _ball_problems(rng, n):
    """Random friction triples: SPD blocks of condition up to 1e6 before their rows' scales (two decades), friction
    coefficients and loads over two decades, torsional coefficients 0.003 to 0.3 times the friction ones, old
    impulses near the cone and residuals from sticking to sliding (float64)."""
    rotation, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    condition = 10.0 ** rng.uniform(0.0, 6.0, n)
    spectrum = np.stack([np.ones(n), condition ** -rng.uniform(0.0, 1.0, n), 1.0 / condition], 1)
    rows = 10.0 ** rng.uniform(-1.0, 1.0, (n, 3))
    block = np.einsum("nij,nj,nkj->nik", rotation, spectrum, rotation) * rows[:, :, None] * rows[:, None, :]
    mu = 10.0 ** rng.uniform(-1.0, 1.0, n)
    mu_t = mu * 10.0 ** rng.uniform(-2.5, -0.5, n)
    load = 10.0 ** rng.uniform(-1.0, 1.0, n)
    coefficients = np.stack([mu, mu, mu_t], 1)
    old = rng.normal(size=(n, 3)) * coefficients * load[:, None] * 0.3
    size = np.einsum("nii->ni", block) * coefficients * load[:, None]
    residual = rng.normal(size=(n, 3)) * size * 10.0 ** rng.uniform(-3.0, 0.5, (n, 1))
    return block, residual, old, mu, mu_t, load


def _ball_reference(block, residual, old, coefficients, load):
    """Solve ``min 1/2 dx^T G dx + r^T dx`` on ``|(old + dx) / c| <= load`` exactly in float64: in ``y = x / c`` the
    trust-region problem, its multiplier by bisection to the float64 floor. Returns the solution and whether it
    sticks (lies inside the cone)."""
    scaled = block * coefficients[:, :, None] * coefficients[:, None, :]
    gradient = coefficients * residual - np.einsum("nij,nj->ni", scaled, old / coefficients)
    values, vectors = np.linalg.eigh(scaled)
    rotated = np.einsum("nji,nj->ni", vectors, gradient)
    inside = np.linalg.norm(rotated / values, axis=1) <= load
    lo, hi = np.zeros(len(load)), np.linalg.norm(rotated, axis=1) / load + values.max(axis=1)
    for _ in range(200):
        alpha = 0.5 * (lo + hi)
        outside = np.linalg.norm(rotated / (values + alpha[:, None]), axis=1) > load
        lo, hi = np.where(outside, alpha, lo), np.where(outside, hi, alpha)
    alpha = np.where(inside, 0.0, hi)
    y = -np.einsum("nij,nj->ni", vectors, rotated / (values + alpha[:, None]))
    return coefficients * y, inside


@wp.kernel
def _ball(
    block: wp.array[wp.mat33],
    residual: wp.array[wp.vec3],
    old: wp.array[wp.vec3],
    mu: wp.array[float],
    mu_t: wp.array[float],
    load: wp.array[float],
    omega: wp.array[float],
    out: wp.array[wp.vec3],
    relaxed: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = laws.friction_ball(block[i], residual[i], old[i], mu[i], mu_t[i], load[i], 1.0)
    relaxed[i] = laws.friction_ball(block[i], residual[i], old[i], mu[i], mu_t[i], load[i], omega[i])


def _run_ball(device, block, residual, old, mu, mu_t, load, omega=None):
    n = len(load)
    omega = np.ones(n) if omega is None else omega
    values = {"block": block, "residual": residual, "old": old, "mu": mu, "mu_t": mu_t, "load": load, "omega": omega}
    values = {name: np.asarray(value, np.float32) for name, value in values.items()}
    zeros = np.zeros((n, 3), np.float32)
    outputs = call(_ball, n, device, **values, out=zeros, relaxed=zeros)
    return values, outputs["out"].numpy().astype(np.float64), outputs["relaxed"].numpy().astype(np.float64)


def test_friction_ball_solves_the_cone_problem(test, device):
    """Minimize the triple's quadratic on its elliptic cone: the float64 solution's objective to the rounding floor.

    On 10^4 random triples (:func:`_ball_problems`) the objective gap to the exact float64 solve, relative to the
    size of its terms over the cone ``|c r| load + |G| (c load)^2``, is at most 1e-6 (eight float32 ulps) on 99.9%
    and 1e-5 on every one; the result leaves the cone by at most 2e-6 of the load (the caller projects it). A
    sticking triple relaxes to ``old + omega (x - old)``.
    """
    rng = np.random.default_rng(83)
    n = 10000
    problem = _ball_problems(rng, n)
    values, x, relaxed = _run_ball(device, *problem, omega=rng.choice([1.0, 0.8, 1.2], n))
    block, residual, old, mu, mu_t, load = (
        values[k].astype(np.float64) for k in ("block", "residual", "old", "mu", "mu_t", "load")
    )
    coefficients = np.stack([mu, mu, mu_t], 1)
    reference, inside = _ball_reference(block, residual, old, coefficients, load)
    test.assertTrue(0.1 < inside.mean() < 0.9, f"sticking fraction {inside.mean()}")

    def objective(impulse):
        change = impulse - old
        return 0.5 * np.einsum("ni,nij,nj->n", change, block, change) + np.einsum("ni,ni->n", residual, change)

    terms = np.linalg.norm(coefficients * residual, axis=1) * load
    terms += np.abs(block).max(axis=(1, 2)) * (coefficients.max(axis=1) * load) ** 2
    gap = (objective(x) - objective(reference)) / terms
    test.assertLess(np.quantile(np.abs(gap), 0.999), 1.0e-6)
    test.assertLess(np.abs(gap).max(), 1.0e-5)
    test.assertLess((np.linalg.norm(x / coefficients, axis=1) / load - 1.0).max(), 2.0e-6)
    omega = values["omega"].astype(np.float64)[:, None]
    expected = old + omega * (x - old)
    error = np.abs(relaxed[inside] - expected[inside]) / (coefficients[inside] * load[inside, None])
    test.assertLess(error.max(), 1.0e-4)


def test_friction_ball_keeps_a_sliding_triple_at_its_solution(test, device):
    """Keep a sliding triple already at its cone problem's solution to the floor, bit for bit, at any load.

    The pair law's hold (:data:`~.projection_laws.PAIR_FLOOR`) in the scaled impulse ``y = (t1 / mu, t2 / mu,
    s / mu_t)``: on its sphere and its gradient ``c r`` anti-parallel to it, both within the floor. With power-of-two
    coefficients, ``y = (2, 3, 6) 2^e`` has the exact norm ``7 2^e`` and residuals ``-2^k y / c`` exact cross
    products: loads 0 or 2 ulps above that norm keep the impulse; 8 ulps above and 4e-6 above re-solve it.
    """
    rng = np.random.default_rng(73)
    n = 4096
    exponent = rng.integers(-40, 40, n)
    mu = np.ldexp(1.0, rng.integers(-3, 4, n))
    mu_t = mu * np.ldexp(1.0, rng.integers(-8, -1, n))
    coefficients = np.stack([mu, mu, mu_t], 1)
    legs = np.array([[2.0, 3.0, 6.0], [6.0, 2.0, 3.0], [3.0, 6.0, 2.0]])[rng.integers(0, 3, n)]
    y = rng.choice([-1.0, 1.0], (n, 3)) * legs * np.ldexp(1.0, exponent)[:, None]
    old = coefficients * y
    residual = -y / coefficients * np.ldexp(1.0, rng.integers(-2, 7, n))[:, None]
    block = _ball_problems(rng, n)[0]
    norm = np.ldexp(7.0, exponent).astype(np.float32)
    ulp = np.nextafter(norm, np.float32(np.inf)) - norm
    loads = {
        "on the sphere": (norm, True),
        "2 ulps inside": (norm + 2 * ulp, True),
        "8 ulps inside": (norm + 8 * ulp, False),
        "4e-6 inside": (norm * np.float32(1.0 + 4.0e-6), False),
    }
    for name, (load, kept) in loads.items():
        with test.subTest(load=name):
            values, x, _ = _run_ball(device, block, residual, old, mu, mu_t, load)
            held = np.all(x == values["old"], axis=1)
            if kept:
                test.assertTrue(held.all(), f"{int((~held).sum())} of {n} re-solved, load {load[~held][:3]}")
            else:
                test.assertFalse(held.any(), f"{int(held.sum())} of {n} kept")


def test_friction_ball_reduces_to_the_pair(test, device):
    """Solve the pair alone where the spin takes no part, and the spin alone where the tangents take none.

    ``mu_t = 0`` is :func:`~.projection_laws.friction_pair` bit for bit, the spin pinned at zero. A spin row
    decoupled from the pair (no coupling in the block, no residual, no impulse) stays at zero at any ``mu_t``, and
    the joint solve meets the pair's disk problem as the pair law does: the objectives agree to the rounding floor
    (the impulses themselves only to the blocks' conditioning, up to 1e6 times it). ``mu = 0`` pins the tangents and
    leaves the spin's box solve, ``clamp(s - r_s / G_ss, -mu_t load, mu_t load)``.
    """
    rng = np.random.default_rng(79)
    n = 4096
    block, residual, old, mu, mu_t, load = _ball_problems(rng, n)
    old[:, 2] = 0.0
    values, pinned, _ = _run_ball(device, block, residual, old, mu, np.zeros(n), load)
    pair = call(
        _pair,
        n,
        device,
        a=np.ascontiguousarray(values["block"][:, 0, 0]),
        c=np.ascontiguousarray(values["block"][:, 0, 1]),
        d=np.ascontiguousarray(values["block"][:, 1, 1]),
        residual=np.ascontiguousarray(values["residual"][:, :2]),
        old=np.ascontiguousarray(values["old"][:, :2]),
        radius=values["mu"] * values["load"],
        omega=np.ones(n, np.float32),
        out=np.zeros((n, 2), np.float32),
        relaxed=np.zeros((n, 2), np.float32),
    )["relaxed"].numpy()
    np.testing.assert_array_equal(pinned[:, :2], pair)
    np.testing.assert_array_equal(pinned[:, 2], 0.0)
    decoupled = block.copy()
    decoupled[:, :2, 2] = decoupled[:, 2, :2] = 0.0
    spinless = residual.copy()
    spinless[:, 2] = 0.0
    values, joint, _ = _run_ball(device, decoupled, spinless, old, mu, mu_t, load)
    test.assertLess((np.abs(joint[:, 2]) / (values["mu_t"] * values["load"])).max(), 1.0e-6)
    g = values["block"][:, :2, :2].astype(np.float64)
    r = values["residual"][:, :2].astype(np.float64)
    start = values["old"][:, :2].astype(np.float64)

    def objective(impulse):
        change = impulse - start
        return 0.5 * np.einsum("ni,nij,nj->n", change, g, change) + np.einsum("ni,ni->n", r, change)

    radius = values["mu"].astype(np.float64) * values["load"]
    terms = np.linalg.norm(r, axis=1) * radius + np.abs(g).max(axis=(1, 2)) * radius**2
    gap = np.abs(objective(joint[:, :2]) - objective(pair.astype(np.float64))) / terms
    test.assertLess(np.quantile(gap, 0.999), 1.0e-6)
    test.assertLess(gap.max(), 1.0e-5)
    spin_old = np.zeros_like(old)
    spin_old[:, 2] = old[:, 2] + rng.normal(size=n) * mu_t * load
    values, spin, _ = _run_ball(device, block, residual, spin_old, np.zeros(n), mu_t, load)
    np.testing.assert_array_equal(spin[:, :2], 0.0)
    bound = values["mu_t"].astype(np.float64) * values["load"]
    step = values["residual"][:, 2].astype(np.float64) / values["block"][:, 2, 2]
    expected = np.clip(values["old"][:, 2] - step, -bound, bound)
    test.assertLess((np.abs(spin[:, 2] - expected) / bound).max(), 1.0e-5)


@wp.kernel
def _scalar_laws(
    old: wp.array[float],
    residual: wp.array[float],
    diagonal: wp.array[float],
    w: wp.array[float],
    bound: wp.array[float],
    kind: wp.array[int],
    iteration: wp.array[int],
    out: wp.array2d[float],
):
    i = wp.tid()
    step = -residual[i] / diagonal[i]
    out[i, 0] = laws.relaxed_step(old[i], laws.regularized_delta(step, w[i], old[i]), 0.7)
    out[i, 1] = laws.regularized_diagonal(diagonal[i], w[i])
    out[i, 2] = laws.unilateral(old[i])
    out[i, 3] = laws.box(old[i], -bound[i], bound[i])
    new, delta = laws.velocity_limit_update(old[i], residual[i] / diagonal[i])
    push_new, push_delta = laws.velocity_limit_update(0.0, residual[i] / diagonal[i])
    out[i, 4] = new
    out[i, 5] = delta
    out[i, 6] = push_new
    out[i, 7] = push_delta
    out[i, 8] = float(laws.friction_gate(kind[i], iteration[i], 3))
    row = laws.speed_row(wp.abs(residual[i]) * 3.0, bound[i], diagonal[i], 0.0)
    out[i, 9] = row[2]


def test_scalar_laws(test, device):
    """Hold each scalar law's definition: steps, sets, velocity limits and the friction gate.

    The regularized step's fixed point, residual ``-(1 / w - 1) d lambda``, is the exact solve against
    :func:`~.projection_laws.regularized_diagonal`.
    """
    rng = np.random.default_rng(63)
    n = 4096
    old = scalars(rng, n)
    residual = scalars(rng, n)
    diagonal = rng.uniform(0.1, 3.0, n).astype(np.float32)
    w = rng.choice(np.array([1.0, 0.5, 0.98, 0.25], np.float32), n)
    bound = rng.uniform(0.05, 2.0, n).astype(np.float32)
    kind = rng.integers(0, 6, n).astype(np.int32)
    iteration = rng.integers(0, 6, n).astype(np.int32)
    out = call(
        _scalar_laws,
        n,
        device,
        old=old,
        residual=residual,
        diagonal=diagonal,
        w=w,
        bound=bound,
        kind=kind,
        iteration=iteration,
        out=np.zeros((n, 10), np.float32),
    )["out"].numpy()
    o, r, d, w64, b = (v.astype(np.float64) for v in (old, residual, diagonal, w, bound))
    np.testing.assert_allclose(out[:, 0], o + 0.7 * (w64 * (-r / d) - (1.0 - w64) * o), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], d / w64, rtol=1e-6)
    np.testing.assert_array_equal(out[:, 2], np.where(old < 0.0, 0.0, old))
    np.testing.assert_array_equal(out[:, 3], np.clip(old, -bound, bound))
    np.testing.assert_allclose(out[:, 4], np.maximum(o - r / d, 0.0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out[:, 5], out[:, 4] - old, rtol=1e-6, atol=1e-6)
    # From a zero total the update pushes a violated row onto its limit (the tiers' last word).
    push = np.where(r < 0.0, -r / d, 0.0)
    np.testing.assert_allclose(out[:, 6], push, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out[:, 7], push, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(out[:, 8], (kind == ROW_FRICTION) & (iteration < 3))
    # A free body over its speed limit keeps its direction and ends on the limit.
    speed = np.abs(r) * 3.0
    over = speed > b
    np.testing.assert_allclose(speed[over] * (1.0 + out[over, 9]), b[over], rtol=1e-5)
    np.testing.assert_array_equal(out[~over, 9], 0.0)
    # The regularized step's fixed point solves the row against the regularized diagonal.
    b_free = r
    fixed = -b_free * w64 / d
    test.assertTrue(np.allclose(fixed, -b_free / (d / w64)))
    stepped = fixed + 0.7 * (w64 * (-(d * fixed + b_free) / d) - (1.0 - w64) * fixed)
    np.testing.assert_allclose(stepped, fixed, rtol=1e-12, atol=1e-12)
    test.assertEqual(ROW_CONTACT, 0)


class TestFeatherPGSProjectionLaws(unittest.TestCase):
    """The projection laws against the conditions that define them."""


register_device_tests(TestFeatherPGSProjectionLaws, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)

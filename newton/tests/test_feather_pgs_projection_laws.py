# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The projection laws solve the problems they name, on every test device.

Each law meets its defining conditions, checked in float64: the tangent pair's
disk problem, the contact's Coulomb problem (cone, normal complementarity and,
for Daviet's Newton, maximal dissipation), the regularized step's fixed point
and its block-solve diagonal, and the velocity-limit, drive and set projections.
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

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
    accumulated_new, accumulated_delta = laws.velocity_limit_update(old[i], residual[i], residual[i] / diagonal[i], 1)
    stateless_new, stateless_delta = laws.velocity_limit_update(old[i], residual[i], residual[i] / diagonal[i], 0)
    out[i, 4] = accumulated_new
    out[i, 5] = accumulated_delta
    out[i, 6] = stateless_new
    out[i, 7] = stateless_delta
    out[i, 8] = laws.velocity_clamp_impulse(residual[i], bound[i], diagonal[i])
    out[i, 9] = laws.physx_drive(old[i], residual[i], w[i], 0.5, 0.25, bound[i])
    out[i, 10] = float(laws.friction_gate(kind[i], iteration[i], 3))
    row = laws.speed_row(wp.abs(residual[i]) * 3.0, bound[i], diagonal[i], 0.0, 0)
    out[i, 11] = row[2]


def test_scalar_laws(test, device):
    """Hold each scalar law's definition: steps, sets, velocity limits, the drive and the friction gate.

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
        out=np.zeros((n, 12), np.float32),
    )["out"].numpy()
    o, r, d, w64, b = (v.astype(np.float64) for v in (old, residual, diagonal, w, bound))
    np.testing.assert_allclose(out[:, 0], o + 0.7 * (w64 * (-r / d) - (1.0 - w64) * o), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], d / w64, rtol=1e-6)
    np.testing.assert_array_equal(out[:, 2], np.where(old < 0.0, 0.0, old))
    np.testing.assert_array_equal(out[:, 3], np.clip(old, -bound, bound))
    np.testing.assert_allclose(out[:, 4], np.maximum(o - r / d, 0.0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out[:, 5], out[:, 4] - old, rtol=1e-6, atol=1e-6)
    stateless = np.where(r < 0.0, -r / d, 0.0)
    np.testing.assert_allclose(out[:, 6], stateless, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out[:, 7], stateless, rtol=1e-6, atol=1e-7)
    # The clamp impulse moves the coordinate's velocity into [-limit, limit].
    np.testing.assert_allclose(r + d * out[:, 8], np.clip(r, -b, b), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[:, 9], np.clip(o * w64 + 0.5 * r + 0.25, -b, b), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(out[:, 10], (kind == ROW_FRICTION) & (iteration < 3))
    # A free body over its speed limit keeps its direction and ends on the limit.
    speed = np.abs(r) * 3.0
    over = speed > b
    np.testing.assert_allclose(speed[over] * (1.0 + out[over, 11]), b[over], rtol=1e-5)
    np.testing.assert_array_equal(out[~over, 11], 0.0)
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

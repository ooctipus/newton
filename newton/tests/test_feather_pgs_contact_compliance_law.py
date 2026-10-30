# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The material contact law: a compliant normal row of the one row law, on every test device.

A material contact (``contact_compliance``) is the row ``J v + rhs + eps lambda`` of
:func:`~newton._src.solvers.feather_pgs.row_laws.contact_material_rhs`, and the sweeps step it by the regularized
step (:func:`~newton._src.solvers.feather_pgs.projection_laws.regularized_delta`) with the weight ``d / (d + eps)``.
The kernels below run those laws on frozen problems: a supported mass, and coupled normal rows.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import projection_laws as laws
from newton._src.solvers.feather_pgs.row_laws import contact_material_rhs, contact_rhs
from newton.tests.feather_pgs_law_harness import call, register_device_tests

MASS, GRAVITY, STIFFNESS = 0.3, 9.81, 3000.0


@wp.kernel
def _material_laws(
    phi: wp.array[float],
    target: wp.array[float],
    stiffness: wp.array[float],
    damping: wp.array[float],
    dt: wp.array[float],
    rhs: wp.array[float],
    eps: wp.array[float],
):
    i = wp.tid()
    value, compliance = contact_material_rhs(phi[i], target[i], stiffness[i], damping[i], dt[i])
    rhs[i] = value
    eps[i] = compliance


@wp.kernel
def _support(
    damping: float,
    dt: float,
    position: float,
    velocity: float,
    iterations: int,
    compliant: int,
    trajectory: wp.array2d[float],
):
    """Step a 0.3 kg mass on a 3000 N/m plane contact through the row law: its material law, or the rigid one (beta
    0.05)."""
    x = position
    v = velocity
    diagonal = 1.0 / MASS
    for step in range(trajectory.shape[0]):
        free = v - GRAVITY * dt
        rigid, _w = contact_rhs(x, 0.05, 0.0, 1.0 / dt, 1.0, 1.0, 1.0, 0.0)
        material, eps = contact_material_rhs(x, 0.0, STIFFNESS, damping, dt)
        rhs = rigid
        weight = float(1.0)
        if compliant != 0:
            rhs = material
            weight = diagonal / (diagonal + eps)
        impulse = float(0.0)
        for _ in range(iterations):
            delta = laws.regularized_delta(-(free + diagonal * impulse + rhs) / diagonal, weight, impulse)
            impulse = laws.unilateral(laws.relaxed_step(impulse, delta, 1.0))
        v = free + diagonal * impulse
        x += dt * v
        trajectory[step, 0] = x
        trajectory[step, 1] = v
        trajectory[step, 2] = impulse / dt


@wp.kernel
def _coupled(
    delassus: wp.array2d[float],
    free: wp.array[float],
    rhs: wp.array[float],
    weight: wp.array[float],
    iterations: int,
    impulses: wp.array[float],
):
    """Gauss-Seidel on coupled material normal rows, each by the weighted step."""
    count = impulses.shape[0]
    for _ in range(iterations):
        for row in range(count):
            residual = free[row] + rhs[row]
            for other in range(count):
                residual += delassus[row, other] * impulses[other]
            old = impulses[row]
            delta = laws.regularized_delta(-residual / delassus[row, row], weight[row], old)
            impulses[row] = laws.unilateral(laws.relaxed_step(old, delta, 1.0))


def _material(phi, stiffness, damping, dt):
    """The law in float64: ``(rhs, eps)`` of a row without target."""
    denominator = dt * stiffness + np.where(np.asarray(phi) <= 0.0, damping, 0.0)
    return stiffness * np.asarray(phi) / denominator, 1.0 / (dt * denominator)


def _support_trace(device, *, dt=0.0025, iterations=16, damping=20.0, position=0.0, velocity=0.0, compliant=True):
    """Eight seconds of the supported mass: ``(x, v, force)`` per step."""
    trajectory = np.zeros((round(8.0 / dt), 3), np.float32)
    args = {"damping": damping, "dt": dt, "position": position, "velocity": velocity}
    out = call(_support, 1, device, **args, iterations=iterations, compliant=int(compliant), trajectory=trajectory)
    return out["trajectory"].numpy()


def _coupled_solve(device, delassus, free, phi, stiffness, damping, *, dt=0.0025, iterations=128):
    """The coupled rows' impulses by the weighted Gauss-Seidel step, and their material residuals."""
    rhs, eps = _material(phi, np.asarray(stiffness, float), np.asarray(damping, float), dt)
    delassus = np.asarray(delassus, float)
    weight = np.diag(delassus) / (np.diag(delassus) + eps)
    values = {"delassus": delassus, "free": free, "rhs": rhs, "weight": weight}
    values = {name: np.asarray(value, np.float32) for name, value in values.items()}
    impulses = call(_coupled, 1, device, **values, iterations=iterations, impulses=np.zeros(len(free), np.float32))
    result = impulses["impulses"].numpy().astype(np.float64)
    return result, np.asarray(free) + rhs + delassus @ result + eps * result


def test_law_is_the_backward_euler_spring_dashpot(test, device):
    """Return ``rhs = k phi / (dt k + c') - target`` and ``eps = 1 / (dt (dt k + c'))``, ``c'`` the damping while the
    gap is closed and zero across an open one."""
    rng = np.random.default_rng(91)
    n = 4096
    phi = rng.normal(scale=0.01, size=n)
    phi[::7] = 0.0
    values = {
        "phi": phi,
        "target": rng.normal(size=n),
        "stiffness": 10.0 ** rng.uniform(1.0, 6.0, n),
        "damping": rng.uniform(0.0, 50.0, n),
        "dt": rng.choice([1.0 / 60.0, 1.0 / 240.0, 1.0 / 480.0], n),
    }
    values = {name: np.asarray(value, np.float32) for name, value in values.items()}
    out = call(_material_laws, n, device, **values, rhs=np.zeros(n, np.float32), eps=np.zeros(n, np.float32))
    f64 = {name: value.astype(np.float64) for name, value in values.items()}
    rhs, eps = _material(f64["phi"], f64["stiffness"], f64["damping"], f64["dt"])
    np.testing.assert_allclose(out["rhs"].numpy(), rhs - f64["target"], rtol=2.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(out["eps"].numpy(), eps, rtol=2.0e-6)


def test_weighted_step_is_the_material_gauss_seidel_step(test, device):
    """The regularized step with ``w = d / (d + eps)`` is the material row's exact Gauss-Seidel step,
    ``lambda - (r + eps lambda) / (d + eps)``, and holds the row's fixed point ``r = -eps lambda``."""
    del device
    rng = np.random.default_rng(93)
    d, eps, old = (10.0 ** rng.uniform(-3.0, 3.0, 4096) for _ in range(3))
    residual = rng.normal(size=4096) * d
    w = d / (d + eps)
    np.testing.assert_allclose(w * (-residual / d) - (1.0 - w) * old, -(residual + eps * old) / (d + eps), rtol=1e-9)
    test.assertTrue(np.allclose(w * (eps * old / d) - (1.0 - w) * old, 0.0, atol=1e-9 * np.abs(old).max()))


def test_static_indentation_and_rigid_baseline(test, device):
    """Rest at ``-m g / k`` at every step size and budget, which the rigid law cannot; carry the weight."""
    target = -MASS * GRAVITY / STIFFNESS
    for dt in (0.005, 0.0025, 0.00125):
        for iterations in (1, 8, 128):
            trace = _support_trace(device, dt=dt, iterations=iterations)
            test.assertLess(abs(trace[-1, 0] - target), 2.0e-7)
    trace = _support_trace(device)
    test.assertLess(abs(trace[-1, 2] - MASS * GRAVITY), 2.0e-4)
    test.assertGreater(abs(_support_trace(device, compliant=False)[-1, 0] - target), 9.0e-4)


def test_damping_reduces_oscillation(test, device):
    """Damp the early oscillation of a released indentation without moving the rest."""
    undamped = _support_trace(device, damping=0.0, position=-0.002)
    damped = _support_trace(device, damping=20.0, position=-0.002)
    test.assertLess(np.sqrt(np.mean(damped[:80, 1] ** 2)), np.sqrt(np.mean(undamped[:80, 1] ** 2)))
    test.assertLess(abs(damped[-1, 0] - undamped[-1, 0]), 2.0e-7)


def test_no_attraction_and_no_dashpot_across_a_gap(test, device):
    """Push only: no tensile impulse for a separating or open contact, and no dashpot before the gap closes."""
    for phi, velocity, damping in ((0.01, 0.0, 20.0), (0.0, 1.0, 20.0), (-0.001, 10.0, 20.0), (0.01, -0.1, 1.0e4)):
        impulse, residual = _coupled_solve(device, [[1.0 / MASS]], [velocity], [phi], [STIFFNESS], [damping])
        test.assertEqual(float(impulse[0]), 0.0)
        test.assertGreaterEqual(float(residual[0]), 0.0)


def test_coupled_rows_match_the_linear_solve(test, device):
    """Reach ``(G + diag(eps)) lambda = -(v + rhs)`` on coupled rows that all push, and keep a split load whole."""
    delassus = np.array([[4.0, 1.0], [1.0, 3.0]])
    phi, stiffness, damping, free = [-0.002, -0.001], [3000.0, 5000.0], [20.0, 10.0], np.array([-0.1, -0.2])
    rhs, eps = _material(phi, np.array(stiffness), np.array(damping), 0.0025)
    expected = np.linalg.solve(delassus + np.diag(eps), -(free + rhs))
    test.assertTrue((expected > 0.0).all())
    result, residual = _coupled_solve(device, delassus, free, phi, stiffness, damping)
    np.testing.assert_allclose(result, expected, rtol=2.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(residual, 0.0, atol=1.0e-6)
    # One patch split into n equal quadrature rows (k / n, c / n each) carries the single row's load.
    rhs, eps = _material(-0.001, STIFFNESS, 20.0, 0.0025)
    whole = (0.1 - rhs) / (1.0 / MASS + eps)
    for count in (1, 4, 16):
        result, residual = _coupled_solve(
            device,
            np.full((count, count), 1.0 / MASS),
            np.full(count, -0.1),
            np.full(count, -0.001),
            np.full(count, STIFFNESS / count),
            np.full(count, 20.0 / count),
        )
        test.assertAlmostEqual(float(result.sum()), whole, places=6)
        np.testing.assert_allclose(residual, 0.0, atol=1.0e-6)


class TestFeatherPGSContactComplianceLaw(unittest.TestCase):
    """The material contact law against its definition, on frozen problems."""


register_device_tests(TestFeatherPGSContactComplianceLaw, globals())


if __name__ == "__main__":
    unittest.main(verbosity=2)

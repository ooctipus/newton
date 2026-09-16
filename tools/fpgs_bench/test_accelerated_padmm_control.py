# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent Kamino accelerated PADMM update-order controls."""

import unittest

import numpy as np

from tools.fpgs_bench import accelerated_padmm_control as control


class TestAcceleratedPADMMControl(unittest.TestCase):
    def setUp(self):
        self.data = control.fixed.prepare(
            np.diag([1.0, 1.5, 2.0]),
            np.array([-0.7, 0.2, 0.3]),
            np.array([0, 2, 2]),
            np.array([-1, 0, 0]),
            np.array([0.0, 0.5, 0.5]),
        )

    def _oracle(self, fprev, yprev, zprev, y_hat, zhat, a, energy):
        data = self.data
        correction = np.array([0.5 * np.linalg.norm(zhat[1:]), 0.0, 0.0])
        right = -data["b"] - correction + control.fixed.ETA * fprev + control.fixed.RHO * y_hat + zhat
        f = np.linalg.solve(data["U"] @ data["U"].T + data["alpha"] * np.eye(3), right)
        argument = f - zhat / control.fixed.RHO
        normal, length = argument[0], np.linalg.norm(argument[1:])
        if normal + 0.5 * length <= 0:
            y = np.zeros(3)
        elif length <= 0.5 * normal:
            y = argument.copy()
        else:
            yn = (normal + 0.5 * length) / 1.25
            y = np.r_[yn, argument[1:] * (0.5 * yn / length)]
        z = zhat + control.fixed.RHO * (y - f)
        e = (
            control.fixed.RHO * np.sum((data["scale"] * (y - y_hat)) ** 2)
            + np.sum(((z - zhat) / data["scale"]) ** 2) / control.fixed.RHO
        )
        if e < 0.999 * energy:
            an = (1 + np.sqrt(1 + 4 * a * a)) / 2
            beta = (a - 1) / an
            return f, y, z, y + beta * (y - yprev), z + beta * (z - zprev), an, e
        return f, y, z, yprev, zprev, 1.0, energy / 0.999

    def test_successful_momentum_matches_dense_source_order(self):
        """Extrapolate current y/z relative to previous unaccelerated iterates."""
        args = (
            np.array([0.3, 0.2, -0.1]),
            np.array([0.2, -0.03, 0.04]),
            np.array([0.1, 0.6, -0.8]),
            np.array([0.4, 0.02, 0.03]),
            np.array([0.2, 0.7, -0.6]),
            2.0,
            float(np.finfo(np.float32).max),
        )
        result = control.accelerated_update(self.data, *args)
        expected = self._oracle(*args)
        for actual, target in zip(result[:7], expected, strict=True):
            np.testing.assert_allclose(actual, target, atol=3e-15)
        self.assertFalse(result[-1]["restart"])
        self.assertGreater(result[-1]["beta"], 0)

    def test_forced_restart_rewinds_hats_not_published_state(self):
        """Rewind hats to previous y/z while retaining the just-computed state."""
        args = (
            np.array([0.3, 0.2, -0.1]),
            np.array([0.2, -0.03, 0.04]),
            np.array([0.1, 0.6, -0.8]),
            np.array([0.4, 0.02, 0.03]),
            np.array([0.2, 0.7, -0.6]),
            2.0,
            0.0,
        )
        result = control.accelerated_update(self.data, *args)
        expected = self._oracle(*args)
        for actual, target in zip(result[:7], expected, strict=True):
            np.testing.assert_allclose(actual, target, atol=3e-15)
        self.assertTrue(result[-1]["restart"])
        np.testing.assert_array_equal(result[3], args[1])
        np.testing.assert_array_equal(result[4], args[2])
        self.assertFalse(np.array_equal(result[1], result[3]))

    def test_physical_sliding_fixed_point(self):
        """Preserve a physical Coulomb fixed point with consistent dual hats."""
        Z = np.diag([1.0, 1.0, np.sqrt(2.0)])
        impulse, residual = np.array([1.0, -0.3, -0.4]), np.array([0.0, 0.6, 0.8])
        data = control.fixed.prepare(
            Z, residual - Z @ Z.T @ impulse, np.array([0, 2, 2]), np.array([-1, 0, 0]), np.array([0.0, 0.5, 0.5])
        )
        y = impulse / data["scale"]
        z = data["scale"] * (residual + np.array([0.5, 0.0, 0.0]))
        result = control.accelerated_update(data, y, y, z, y, z, 1.0, float(np.finfo(np.float32).max))
        for actual, target in ((result[0], y), (result[1], y), (result[2], z), (result[3], y), (result[4], z)):
            np.testing.assert_allclose(actual, target, atol=4e-15)


if __name__ == "__main__":
    unittest.main()

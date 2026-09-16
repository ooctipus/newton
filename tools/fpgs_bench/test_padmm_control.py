# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-operator PADMM algebra and cone-law controls."""

import unittest

import numpy as np

from tools.fpgs_bench import padmm_control as control


class TestPADMMControl(unittest.TestCase):
    def test_woodbury_matches_dense(self):
        """Solve the fixed augmented operator without a row-space factor."""
        z = np.array([[1.0, 0.2], [0.3, 1.1], [-0.4, 0.6]])
        data = control.prepare(
            z, np.array([-1.0, 0.4, 0.7]), np.array([0, 2, 2]), np.array([-1, 0, 0]), np.array([0.0, 0.5, 0.5])
        )
        rhs = np.array([0.7, -0.5, 1.3])
        expected = np.linalg.solve(data["U"] @ data["U"].T + data["alpha"] * np.eye(3), rhs)
        np.testing.assert_allclose(control.woodbury(data, rhs), expected, atol=2e-15)
        self.assertEqual(np.unique(data["scale"]).size, 1)

    def test_cone_boundary_polar_and_zero_friction(self):
        """Project normal-first coordinates with the correct polar sign."""
        np.testing.assert_array_equal(control.project_cone(np.array([-2.0, 1.0, 0.0]), 0.5), np.zeros(3))
        np.testing.assert_array_equal(control.project_cone(np.array([-2.0, 0.0, 0.0]), 0.0), np.zeros(3))
        np.testing.assert_array_equal(control.project_cone(np.array([2.0, 3.0, 4.0]), 0.0), [2.0, 0.0, 0.0])
        result = control.project_cone(np.array([0.2, 0.6, 0.8]), 0.5)
        self.assertAlmostEqual(np.linalg.norm(result[1:]), 0.5 * result[0])

    def test_isolated_anisotropic_sliding_fixed_point(self):
        """Preserve an exact physical sliding solution under one PADMM update."""
        Z = np.diag([1.0, 1.0, np.sqrt(2.0)])
        impulse, residual = np.array([1.0, -0.3, -0.4]), np.array([0.0, 0.6, 0.8])
        data = control.prepare(
            Z, residual - Z @ Z.T @ impulse, np.array([0, 2, 2]), np.array([-1, 0, 0]), np.array([0.0, 0.5, 0.5])
        )
        y = impulse / data["scale"]
        dual = data["scale"] * (residual + np.array([0.5 * np.linalg.norm(residual[1:]), 0.0, 0.0]))
        f1, y1, z1, _ = control.update(data, y, y, dual)
        for actual, expected in ((f1, y), (y1, y), (z1, dual)):
            np.testing.assert_allclose(actual, expected, atol=3e-15)

    def test_dual_correction_uses_previous_state(self):
        """Match Algorithm 1 ordering with nonzero prior primal and dual state."""
        data = control.prepare(
            np.diag([1.0, 1.5, 2.0]),
            np.array([-0.7, 0.2, 0.3]),
            np.array([0, 2, 2]),
            np.array([-1, 0, 0]),
            np.array([0.0, 0.5, 0.5]),
        )
        f, y, z = np.array([0.3, 0.2, -0.1]), np.array([0.2, -0.03, 0.04]), np.array([0.1, 0.6, -0.8])
        h = -data["b"] - np.array([0.5, 0.0, 0.0]) + control.ETA * f + control.RHO * y + z
        expected_f = np.linalg.solve(data["U"] @ data["U"].T + data["alpha"] * np.eye(3), h)
        expected_y = control.project_cone(expected_f - z / control.RHO, 0.5)
        f1, y1, z1, _ = control.update(data, f, y, z)
        np.testing.assert_allclose(f1, expected_f, atol=2e-15)
        np.testing.assert_allclose(y1, expected_y, atol=2e-15)
        np.testing.assert_allclose(z1, z + control.RHO * (expected_y - expected_f), atol=2e-15)


if __name__ == "__main__":
    unittest.main()

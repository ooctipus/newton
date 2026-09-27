# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ordered root-free contact-law regression controls."""

import unittest

import numpy as np

from tools.fpgs_bench import spectral_gs_control as control


class TestSpectralGSControl(unittest.TestCase):
    def test_duplicate_normals_see_preceding_response(self):
        """Resolve duplicate normals in one ordered pass without amplification."""
        result = control.solve(
            np.ones((3, 1)),
            np.eye(1),
            np.ones(3),
            -np.ones(3),
            np.zeros(3, int),
            np.full(3, -1),
            np.zeros(3),
            np.zeros(1),
            iterations=24,
        )
        np.testing.assert_array_equal(result.impulses, [1.0, 0.0, 0.0])
        self.assertEqual(result.work["sweeps"], 1)
        self.assertTrue(result.work["physical_stop"])

    def test_isolated_anisotropic_sliding(self):
        """Converge to physical antiparallel slip using one common tangent step."""
        jacobian = np.diag([1.0, 1.0, np.sqrt(2.0)])
        expected = np.array([1.0, -0.3, -0.4])
        residual = np.array([0.0, 0.6, 0.8])
        result = control.solve(
            jacobian,
            np.eye(3),
            np.sum(jacobian**2, axis=1),
            residual - jacobian @ jacobian.T @ expected,
            np.array([0, 2, 2]),
            np.array([-1, 0, 0]),
            np.array([0.0, 0.5, 0.5]),
            np.zeros(3),
            iterations=24,
        )
        np.testing.assert_allclose(result.impulses, expected, atol=3e-5, rtol=0)
        self.assertLess(result.work["final_physical"]["mdp"], 3e-5)
        self.assertEqual(result.work["root_probes"], 0)
        self.assertEqual(result.work["majorizer_products"], 0)

    def test_incoming_state_applies_only_delta(self):
        """Retain current velocity and use only impulse changes within budget."""
        jacobian = np.array([[1.0, 0.2], [0.3, 0.8]])
        factor = np.array([[2.0, 0.0], [0.4, 1.0]])
        z = np.linalg.solve(factor, jacobian.T).T
        response = np.linalg.solve(factor.T, z.T).T
        diagonal = np.sum(z * z, axis=1) + 0.1
        current, incoming = np.array([-0.3, 0.1]), np.array([0.2, 0.4])
        result = control.solve(
            jacobian,
            factor,
            diagonal,
            -np.ones(2),
            np.array([0, 3]),
            np.full(2, -1),
            np.zeros(2),
            current,
            iterations=1,
            incoming=incoming,
        )
        np.testing.assert_allclose(result.velocity, current + response.T @ (result.impulses - incoming), atol=2e-15)
        self.assertEqual(result.work["sweeps"], 1)
        self.assertEqual(result.work["unused_allowance"], 0)


if __name__ == "__main__":
    unittest.main()

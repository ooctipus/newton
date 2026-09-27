# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical-map and residual-carry controls; no native performance claims."""

import unittest

import numpy as np

from tools.fpgs_bench import natural_nesterov_control as control


class TestNaturalNesterovControl(unittest.TestCase):
    def test_isolated_anisotropic_sliding(self):
        """Converge to antiparallel physical friction with a common pair step."""
        z = np.diag([1.0, 1.0, np.sqrt(2.0)])
        expected = np.array([1.0, -0.3, -0.4])
        residual = np.array([0.0, 0.6, 0.8])
        result = control.natural_continue(
            z,
            np.sum(z * z, axis=1),
            residual - z @ z.T @ expected,
            np.array([0, 2, 2]),
            np.array([-1, 0, 0]),
            np.array([0.0, 0.5, 0.5]),
            np.zeros(3),
            iterations=24,
        )
        np.testing.assert_allclose(result.impulses, expected, atol=3e-5, rtol=0)
        self.assertLess(result.work["final_physical"]["mdp"], 3e-5)
        self.assertEqual(result.work["majorizer_products"], 0)

    def test_residual_history_matches_direct_operator(self):
        """Recover the feasible-state residual before restart changes momentum."""
        z = np.array([[1.0, 0.2], [0.3, 0.8], [-0.1, 0.4]])
        rhs, incoming = np.array([-1.0, -0.4, -0.2]), np.array([0.2, 0.1, 0.05])
        result = control.natural_continue(
            z,
            np.sum(z * z, axis=1) + 2.0,
            rhs,
            np.array([0, 3, 3]),
            np.full(3, -1),
            np.zeros(3),
            incoming,
            iterations=24,
            trace=True,
        )
        self.assertTrue(any(entry["beta"] > 0 for entry in result.work["trace"]))
        for entry in result.work["trace"]:
            np.testing.assert_allclose(entry["residual"], rhs + z @ (z.T @ entry["impulse"]), atol=3e-15)
        np.testing.assert_allclose(result.velocity, z.T @ (result.impulses - incoming), atol=3e-15)
        self.assertEqual(result.work["operator_products"], 2 * z.size * result.work["operator_passes"])
        self.assertEqual(result.work["stop_operator_products"], 0)

    def test_duplicate_normals_expose_unstable_map(self):
        """Preserve the physical PSD counterexample rather than hiding a cycle."""
        result = control.natural_continue(
            np.ones((3, 1)),
            np.ones(3),
            -np.ones(3),
            np.zeros(3, int),
            np.full(3, -1),
            np.zeros(3),
            np.zeros(3),
            iterations=24,
            trace=True,
        )
        self.assertFalse(result.work["physical_stop"])
        self.assertFalse(result.work["final_physical"]["stop"])
        self.assertEqual(result.work["sweeps"], 24)
        np.testing.assert_array_equal(result.work["trace"][0]["impulse"], np.zeros(3))
        np.testing.assert_array_equal(result.work["trace"][1]["impulse"], np.ones(3))
        np.testing.assert_array_equal(result.work["trace"][2]["impulse"], np.zeros(3))


if __name__ == "__main__":
    unittest.main()

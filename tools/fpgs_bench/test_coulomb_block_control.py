# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical controls for simultaneous normal and circular-friction blocks."""

import unittest

import numpy as np

from tools.fpgs_bench import coulomb_block_control as control


class TestCoulombBlockControl(unittest.TestCase):
    def check_law(self, matrix, bias, friction):
        """Check every physical local equation, not a selected root branch."""
        value, info = control.local_block(matrix, bias, friction)
        self.assertIsNotNone(value, info)
        residual = matrix @ value + bias
        self.assertGreaterEqual(value[0], 0)
        self.assertLessEqual(np.linalg.norm(value[1:]), friction * value[0] + 1e-10)
        self.assertGreaterEqual(residual[0], -1e-8)
        self.assertLess(abs(value[0] * residual[0]), 1e-8)
        if value[0] > 0:
            # Coupled inward roundoff repair retains the declared KKT bound.
            scale = 1 + np.linalg.norm(bias) + (np.linalg.norm(matrix, 2) + info["gamma"]) * np.linalg.norm(value)
            self.assertLessEqual(np.linalg.norm(residual[1:] + info["gamma"] * value[1:]), 8e-6 * scale)
        return value, info

    def test_coupled_slip_and_multiple_valid_roots(self):
        """The bracket remains valid when the Coulomb root is not monotone."""
        self.check_law(
            np.array([[2.0, 0.7, -0.2], [0.7, 1.0, 0.25], [-0.2, 0.25, 1.3]]), np.array([-1.0, 2.0, -1.0]), 0.6
        )
        h = np.array([0.9977495887413199, -0.2281392022580242])
        bn = -0.02731990326961045
        matrix = np.block([[np.ones((1, 1)), h[None, :]], [h[:, None], np.diag([0.1, 10.0]) + np.outer(h, h)]])
        self.check_law(matrix, np.r_[bn, np.ones(2) + h * bn], 1.0)

    def test_open_frictionless_and_unsafe(self):
        """Keep zero-radius/open modes explicit and reject an unsafe Schur block."""
        value, _ = self.check_law(np.eye(3), np.array([1.0, 2.0, 3.0]), 0.5)
        np.testing.assert_array_equal(value, 0)
        value, _ = control.local_block(np.eye(3), np.array([-1.0, 2.0, 3.0]), 0)
        np.testing.assert_array_equal(value, [1, 0, 0])
        value, _ = control.local_block(np.ones((3, 3)), np.array([-1.0, 2.0, 3.0]), 0.5)
        self.assertIsNone(value)

    def test_cfm_is_proximal_delta_and_sweep_budget(self):
        """Do not turn original update denominators into physical compliance."""
        J = np.array([[1.0, 0, 0], [0.3, 1.0, 0], [-0.2, 0.15, 0.8]])
        physical = J @ J.T
        matrix = physical + 0.2 * np.eye(3)
        old = np.array([0.5, -0.1, 0.1])
        residual = physical @ old + np.array([-1.0, 2.0, -1.0])
        bias = residual - matrix @ old
        value, info = control.local_block(matrix, bias, 0.6)
        self.assertIsNotNone(value, info)
        actual = residual + physical @ (value - old)
        proximal = actual + 0.2 * (value - old)
        scale = 1 + np.linalg.norm(bias) + (np.linalg.norm(matrix, 2) + info["gamma"]) * np.linalg.norm(value)
        self.assertLessEqual(np.linalg.norm(proximal + np.r_[0, info["gamma"] * value[1:]]), 8e-6 * scale)
        self.assertGreater(np.linalg.norm(actual + 0.2 * value + np.r_[0, info["gamma"] * value[1:]]), 0.05)


if __name__ == "__main__":
    unittest.main()

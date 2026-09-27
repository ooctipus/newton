# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-evaluation joint-state law and matched damping controls."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import coulomb_block_joint as joint


class TestCoulombBlockJoint(unittest.TestCase):
    def problem(self):
        """Construct an anisotropic contact with a known sliding fixed point."""
        matrix = np.array([[2.0, 0.8, 0.3], [0.8, 1.5, 0.2], [0.3, 0.2, 0.9]])
        impulse = np.array([1.0, -0.3, 0.4])
        residual = np.array([0.0, 0.6, -0.8])
        return matrix, residual - matrix @ impulse, impulse

    def test_one_evaluation_publishes_feasible_unfinished_trial(self):
        """Keep an unfinished root feasible without invoking metric fallback."""
        matrix, bias, _target = self.problem()
        value, next_gamma, info = joint.joint_block(matrix, bias, 0.5, 0.0)
        self.assertIsNotNone(value, info)
        self.assertEqual(info["kind"], "progress")
        self.assertEqual(info["probes"], 1)
        self.assertEqual(info["derivative_evaluations"], 1)
        self.assertGreater(next_gamma, 0.0)
        self.assertGreaterEqual(value[0], 0.0)
        self.assertAlmostEqual(np.linalg.norm(value[1:]), 0.5 * value[0], places=12)
        self.assertAlmostEqual((matrix @ value + bias)[0], 0.0, places=12)

    def test_exact_sliding_and_sticking_fixed_points(self):
        """Recover the original Coulomb law at a joint fixed point."""
        matrix, bias, target = self.problem()
        value, gamma, info = joint.joint_block(matrix, bias, 0.5, 2.0)
        np.testing.assert_allclose(value, target, atol=1e-12)
        self.assertAlmostEqual(gamma, 2.0, places=12)
        self.assertEqual(info["probes"], 1)
        sticking = np.array([2.0, 0.1, -0.2])
        value, gamma, info = joint.joint_block(matrix, -matrix @ sticking, 0.5, 0.0)
        np.testing.assert_allclose(value, sticking, atol=1e-12)
        self.assertEqual(info["kind"], "stick")
        self.assertEqual(gamma, 0.0)
        self.assertEqual(info["probes"], 1)

    def test_nonnegative_derivative_is_unsafe_not_an_extra_root_loop(self):
        """Reject the positive-slope middle root of a nonmonotone contact."""
        h = np.array([0.9977495887413199, -0.2281392022580242])
        matrix = np.empty((3, 3))
        matrix[0, 0], matrix[0, 1:], matrix[1:, 0] = 1.0, h, h
        matrix[1:, 1:] = np.diag([0.1, 10.0]) + np.outer(h, h)
        bn = -0.02731990326961045
        bias = np.r_[bn, np.ones(2) + h * bn]
        value, gamma, info = joint.joint_block(matrix, bias, 1.0, 2.0)
        self.assertIsNone(value)
        self.assertEqual(gamma, 0.0)
        self.assertEqual(info["reason"], "nonnegative_derivative")
        self.assertEqual(info["probes"], 1)

    def test_multiplier_uses_the_same_permanent_half_damping(self):
        """Blend lambda and gamma once at the latch and on subsequent passes."""
        energies = iter((10.0, 20.0, 12.0, 13.0))

        def score(*_args):
            return {
                "natural": next(energies) * 1e-5,
                "normal": 0.0,
                "complementarity": 0.0,
                "mdp": 0.0,
                "cone": 0.0,
                "stop": False,
            }

        proposed = (np.array([1.0, 0.0, 0.0]), 4.0, {"kind": "progress", "probes": 1, "derivative_evaluations": 1})
        with patch.dict(
            joint.solve.__globals__, joint_block=lambda *_args, **_kwargs: proposed, physical_from_residual=score
        ):
            result = joint.solve(
                np.eye(3),
                np.eye(3),
                np.ones(3),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0, 2, 2]),
                np.array([-1, 0, 0]),
                np.ones(3),
                np.zeros(3),
                iterations=2,
            )
        np.testing.assert_array_equal(result.impulses, [0.75, 0.0, 0.0])
        np.testing.assert_array_equal(result.velocity, [0.75, 0.0, 0.0])
        np.testing.assert_array_equal(result.work["gamma"], [3.0, 0.0, 0.0])
        self.assertEqual(result.work["sweeps"], 2)
        self.assertEqual(result.work["operator_products"], 36)
        self.assertEqual(result.work["root_probes"], 2)
        self.assertEqual(result.work["midpoint_row_projections"], 3)

    def test_open_and_frictionless_clear_proposed_multiplier(self):
        """Discard an old sliding multiplier for exact open/frictionless laws."""
        for bias, mu, target in (
            (np.array([1.0, 2.0, 3.0]), 0.5, np.zeros(3)),
            (np.array([-1.0, 2.0, 3.0]), 0.0, np.array([1.0, 0.0, 0.0])),
        ):
            value, gamma, info = joint.joint_block(np.eye(3), bias, mu, 7.0)
            np.testing.assert_array_equal(value, target)
            self.assertEqual(gamma, 0.0)
            self.assertEqual(info["probes"], 0)


if __name__ == "__main__":
    unittest.main()

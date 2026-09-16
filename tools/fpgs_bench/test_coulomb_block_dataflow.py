# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent whole-sweep and impulse-lifecycle contact-block controls."""

import unittest

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench.test_sparse_metric_tangents import physical_metrics


class TestCoulombBlockDataflow(unittest.TestCase):
    def problem(self):
        local = np.array([[2.0, 0.8, 0.3], [0.8, 1.5, 0.2], [0.3, 0.2, 0.9]])
        response = np.zeros((6, 6))
        response[:3, :3] = np.linalg.cholesky(local)
        response[3:, 3:] = np.linalg.cholesky(local)
        target = np.tile([1.0, -0.3, 0.4], 2)
        residual = np.tile([0.0, 0.6, -0.8], 2)
        bias = residual - response @ response.T @ target
        types = np.tile([0, 2, 2], 2)
        parents = np.array([-1, 0, 0, -1, 3, 3])
        friction = np.full(6, 0.5)
        return response, np.sum(response**2, axis=1), bias, types, parents, friction

    def test_independent_contacts_finish_in_one_complete_sweep(self):
        """Satisfy independent full contact laws within one original sweep."""
        z, diagonal, bias, types, parents, friction = self.problem()
        result = block.solve(z, np.eye(6), diagonal, bias, types, parents, friction, np.zeros(6), iterations=1)
        score = physical_metrics(
            z, diagonal, bias, types, parents, friction, np.zeros(6), result.velocity, result.impulses
        )
        for name in ("normal", "complementarity", "mdp", "natural", "cone"):
            self.assertLessEqual(score[name], 3e-5, (name, score))
        np.testing.assert_allclose(result.velocity, z.T @ result.impulses, atol=2e-12)

    def test_incoming_impulse_is_not_silently_applied_twice(self):
        """Preserve the incoming velocity and apply only impulse differences."""
        z, diagonal, bias, types, parents, friction = self.problem()
        incoming = np.tile([0.3, -0.06, 0.08], 2)
        predictor = np.array([0.1, -0.2, 0.05, -0.05, 0.1, -0.1])
        for value in (z, diagonal, bias, types, parents, friction, incoming, predictor):
            value.setflags(write=False)
        result = block.solve(
            z, np.eye(6), diagonal, bias, types, parents, friction, predictor, iterations=1, incoming=incoming
        )
        np.testing.assert_allclose(result.velocity, predictor + z.T @ (result.impulses - incoming), atol=2e-12)
        score = physical_metrics(
            z, diagonal, bias, types, parents, friction, predictor, result.velocity, result.impulses
        )
        for name in ("normal", "complementarity", "mdp", "natural", "cone"):
            self.assertLessEqual(score[name], 3e-5, (name, score))

    def test_circular_friction_is_independent_of_tangent_basis(self):
        """Preserve the physical contact under a rotated tangent basis."""
        z, _diagonal, bias, _types, _parents, _friction = self.problem()
        matrix = z[:3] @ z[:3].T
        rotation = np.eye(3)
        c, s = np.cos(0.37), np.sin(0.37)
        rotation[1:, 1:] = [[c, -s], [s, c]]
        value, info = block.local_block(matrix, bias[:3], 0.5)
        rotated, rotated_info = block.local_block(rotation @ matrix @ rotation.T, rotation @ bias[:3], 0.5)
        self.assertIsNotNone(value, info)
        self.assertIsNotNone(rotated, rotated_info)
        np.testing.assert_allclose(rotated, rotation @ value, rtol=1e-8, atol=1e-9)


if __name__ == "__main__":
    unittest.main()

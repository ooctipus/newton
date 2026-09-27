# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Original FP32 parallel recurrence controls for both ANYmal row tiers."""

import unittest

import numpy as np

from tools.fpgs_bench import coulomb_block_hybrid as hybrid


class TestCoulombBlockHybrid(unittest.TestCase):
    def test_fresh_parallel_reproduces_pinned_reference_both_tiers(self):
        """Keep anisotropic steps, fresh-normal disk projection and warp association."""
        for count in (6, 36):
            with self.subTest(rows=count):
                rng = np.random.default_rng(42)
                z = rng.normal(size=(count, 18)).astype(np.float32)
                parents = np.repeat(np.arange(0, count, 3), 3)
                parents[::3] = -1
                problem = {
                    "Z": z,
                    "b": rng.normal(size=count).astype(np.float32),
                    "kind": np.tile([0, 2, 2], count // 3),
                    "parent": parents,
                    "mu": np.full(count, 0.6, np.float32),
                    "cfm": np.full(count, 0.02, np.float32),
                    "L": np.eye(18, dtype=np.float32),
                    "predictor": np.zeros(18, np.float32),
                }
                diagonal = np.diag(hybrid.reference.gram(z)) + problem["cfm"]
                expected = hybrid.reference.solve(problem, lazy=False, sweeps=24)
                result = hybrid.parallel_continue(
                    z,
                    diagonal,
                    problem["b"],
                    problem["kind"],
                    parents,
                    problem["mu"],
                    np.zeros(count, np.float32),
                    iterations=24,
                )
                np.testing.assert_array_equal(result.impulses, expected["impulse"])
                np.testing.assert_array_equal(result.velocity, expected["velocity"])
                self.assertEqual(result.work["sweeps"], expected["sweeps"])
                self.assertEqual(result.work["majorizer_products"], count * count * 18)
                self.assertLessEqual(result.work["sweeps"], 24)


if __name__ == "__main__":
    unittest.main()

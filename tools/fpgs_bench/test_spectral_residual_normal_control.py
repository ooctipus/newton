# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cause-specific independent absolute-normal stream controls."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from tools.fpgs_bench import spectral_jacobi_control as original
from tools.fpgs_bench import spectral_residual_normal_control as candidate


class ResidualNormalTests(unittest.TestCase):
    def test_normal_score_matches_full_physics(self):
        rng = np.random.default_rng(12)
        kinds = np.array([0, 2, 2, 3])
        parents = np.array([-1, 0, 0, -1])
        normals = np.array([0, 3])
        for _ in range(128):
            impulse = rng.uniform(0, 2, 4)
            residual = rng.normal(size=4)
            score = original.physical_score(residual, impulse, np.ones(4), kinds, parents, np.ones(4), 1)
            self.assertEqual(
                candidate.normal_merit(impulse, residual, normals),
                max(score["normal"], score["complementarity"]) / 3e-5,
            )

    def test_same_operators_extra_stream_accounted(self):
        Z = np.ones((3, 1))
        result = candidate.solve(
            Z,
            np.eye(1),
            np.ones(3),
            -np.ones(3),
            np.zeros(3, int),
            np.full(3, -1),
            np.zeros(3),
            np.zeros(1),
            incoming=np.ones(3),
            iterations=4,
            early_stop=False,
        )
        w = result.work
        self.assertEqual(w["latch_step"], 1)
        self.assertEqual(w["operator_products"], 24)
        self.assertEqual(
            w["normal_merit_reductions"],
            w["initial_proposals"] + w["lookahead_proposals"] + w["midpoint_replacement_proposals"],
        )
        self.assertEqual(w["normal_merit_row_visits"], 3 * w["normal_merit_reductions"])
        np.testing.assert_allclose(result.velocity, Z.T @ (result.impulses - 1), atol=1e-15)

    def test_saved_masked_normal_instability_latches_at_three(self):
        path = Path(
            "/home/octi/Projects/newton-fpgs-anymal-contiguous-response-20260914/tools/fpgs_bench/test_branch_response.py"
        )
        spec = importlib.util.spec_from_file_location("residual_test_reference", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        rows = helper.reference("rows")
        a, saved, scalar, owned, _pins = rows.load(0, 1, 48)
        self.assertIn(42, owned)
        p = rows.world(a, saved, scalar, 42)
        Z = p["Z"]
        args = (Z, np.eye(18), np.sum(Z * Z, axis=1) + p["cfm"], p["b"], p["kind"], p["parent"], p["mu"], np.zeros(18))
        result = candidate.solve(*args)
        cached = original.solve(*args)
        self.assertEqual(result.work["latch_step"], 3)
        curve = result.work["curve"][2]
        self.assertLess(curve["trial_q"], curve["current_q"])
        self.assertGreater(curve["trial_normal_merit"], curve["current_normal_merit"])
        self.assertTrue(curve["normal_increased"])
        np.testing.assert_array_equal(result.impulses, cached.impulses)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU/source controls for the fixed Q-or-N native workflow, no GPU launches."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs as original
from newton._src.solvers.feather_pgs import spectral_jacobi, spectral_residual
from tools.fpgs_bench import spectral_residual_normal_control as control


@wp.kernel
def _metrics(
    old: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    coefficients: wp.array[wp.vec4],
    weights: wp.array[wp.vec3],
    friction: wp.array[float],
    output: wp.array[wp.vec2],
):
    i = wp.tid()
    trial = spectral_jacobi.project_contact_cached(old[i], residual[i], coefficients[i], friction[i])
    output[i] = spectral_residual.proposal_metrics(old[i], residual[i], trial, weights[i])


class SpectralResidualNativeTests(unittest.TestCase):
    def test_actual_native_metrics_match_frozen_cpu_definition(self):
        rng = np.random.default_rng(56)
        size = 256
        old, residual, coefficients, weights = (np.zeros((size, k), np.float32) for k in (3, 3, 4, 3))
        friction = rng.uniform(0, 1, size).astype(np.float32)
        expected = np.zeros((size, 2))
        for i in range(size):
            Z = rng.normal(size=(3, 4))
            diagonal = np.sum(Z * Z, axis=1) + np.exp(rng.uniform(-5, 2, 3))
            old[i] = rng.normal(size=3)
            old[i, 0] = abs(old[i, 0])
            old[i, 1:] *= min(1, 0.99 * friction[i] * old[i, 0] / np.linalg.norm(old[i, 1:]))
            residual[i] = rng.normal(size=3)
            prep = control.prepare(
                Z, np.eye(4), diagonal, np.array([0, 2, 2]), np.array([-1, 0, 0]), np.full(3, friction[i]), old[i]
            )
            _, _, _, cross, spectral, _mu, wt, wn = prep["cache"][0]
            cold_scale = float(rng.uniform(1, 10))
            coefficients[i] = (1 / diagonal[0], 1 / spectral, *cross)
            weights[i] = (np.sqrt(diagonal[0]), wt, wn)
            weights[i] /= 1e-5 * cold_scale
            _, q = control.propose(prep, old[i].astype(float), residual[i].astype(float), cold_scale)
            expected[i] = (
                q / 1e-5,
                control.normal_merit(old[i].astype(float), residual[i].astype(float), np.array([0])),
            )
        output = wp.empty(size, dtype=wp.vec2, device="cpu")
        inputs = [
            wp.array(old, dtype=wp.vec3, device="cpu"),
            wp.array(residual, dtype=wp.vec3, device="cpu"),
            wp.array(coefficients, dtype=wp.vec4, device="cpu"),
            wp.array(weights, dtype=wp.vec3, device="cpu"),
            wp.array(friction, device="cpu"),
        ]
        wp.launch(_metrics, dim=size, inputs=inputs, outputs=[output], device="cpu")
        np.testing.assert_allclose(output.numpy(), expected, rtol=8e-6, atol=3e-3)

    def test_live_owner_abi_and_complete_lookahead_workflow(self):
        with patch.object(original, "_REGISTER_WHITENING", True):
            factory = spectral_residual.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
            for arch in (120, 103):
                for tier in (32, 48):
                    args = {
                        "rows": tier,
                        "min_rows": 0 if tier == 32 else 32,
                        "sweeps": 24,
                        "matrix_free": True,
                        "inkernel_response": (18, 0, 0, 0),
                        "exact_row_sums": True,
                        "world_rows": True,
                    }
                    baseline = original._get_pgs_solve_parallel_kernel(72, 32, 18, arch, **args)
                    kernel = factory(72, 32, 18, arch, **args)
                    self.assertTrue(kernel._fpgs_spectral_residual)
                    self.assertIn("_sr24", kernel.key)
                    self.assertEqual([a.label for a in baseline.adj.args], [a.label for a in kernel.adj.args])
                    self.assertEqual(kernel._fpgs_block_dim, baseline._fpgs_block_dim)
                    src = kernel._fpgs_spectral_residual_native
                    self.assertIn("pass<24", src)
                    self.assertIn("sr_trial[0]>sr_previous[0]||sr_trial[1]>sr_previous[1]", src)
                    self.assertIn("sr_previous[0]<=1.0f||pass==23", src)
                    self.assertEqual(src.count("sr_lookahead(s_y,s_step)"), 2)
                    self.assertIn("sr_lookahead(s_x,s_rhs)", src)
                    self.assertIn("0.5f*(s_rhs[lane]+s_step[lane])", src)
                    self.assertIn("s_x[i] - s_lam0[i]", src)
                    self.assertNotIn("cc_block", src)
                    self.assertNotIn("cc_schur", src)
                    self.assertNotIn("candidate_energy>sj_energy", src)

    def test_unsupported_configuration_retains_original(self):
        factory = spectral_residual.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
        args = {
            "rows": 32,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "exact_row_sums": True,
            "world_rows": True,
        }
        for changed in ({"sweeps": 8}, {"has_drive_rows": True}, {"nesterov": False}, {"exact_row_sums": False}):
            kernel = factory(72, 32, 18, 120, **(args | changed))
            self.assertFalse(getattr(kernel, "_fpgs_spectral_residual", False))


if __name__ == "__main__":
    unittest.main()

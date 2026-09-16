# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Factory/dataflow controls for concurrent spectral proposals, without CUDA."""

import inspect
import unittest
from unittest.mock import patch

from newton._src.solvers.feather_pgs import solver_feather_pgs as original
from newton._src.solvers.feather_pgs import spectral_jacobi


class TestSpectralJacobiNative(unittest.TestCase):
    def test_actual_owner_abis_and_concurrent_dataflow(self):
        with patch.object(original, "_REGISTER_WHITENING", True):
            factory = spectral_jacobi.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
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
                    self.assertTrue(kernel._fpgs_spectral_jacobi)
                    self.assertIn("_sj24", kernel.key)
                    self.assertEqual([a.label for a in baseline.adj.args], [a.label for a in kernel.adj.args])
                    self.assertEqual(baseline._fpgs_block_dim, kernel._fpgs_block_dim)
                    source = kernel._fpgs_spectral_jacobi_native
                    self.assertIn("pass<24", source)
                    self.assertIn("if(lane<n_rows&&s_kind[lane]==0)", source)
                    self.assertIn("sg_project(old,residual,coefficients,s_mu[row+1])", source)
                    self.assertIn("sj_alpha==1.0f&&candidate_energy>sj_energy", source)
                    self.assertIn("0.5f*(s_rhs[lane]+s_step[lane])", source)
                    self.assertIn("1.0e-5f*sj_scale", source)
                    self.assertIn("cone_tolerance=3.0e-5f", source)
                    self.assertIn("s_x[i] - s_lam0[i]", source)
                    self.assertNotIn("while(row<n_rows)", source)
                    self.assertNotIn("cc_block", source)
                    self.assertNotIn("cc_metric", source)
                    self.assertNotIn("cc_schur", source)
                    self.assertNotIn("probe<", source)
            self.assertEqual(inspect.signature(factory), inspect.signature(original._get_pgs_solve_parallel_kernel))

    def test_unsupported_policies_keep_original_factory(self):
        factory = spectral_jacobi.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
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
            self.assertFalse(getattr(kernel, "_fpgs_spectral_jacobi", False))
            self.assertNotIn("_sj24", kernel.key)


if __name__ == "__main__":
    unittest.main()

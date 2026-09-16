# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the simultaneous native owner's unchanged ABI and bounded policy."""

import inspect
import unittest
from unittest.mock import patch

from newton._src.solvers.feather_pgs import coupled_jacobi
from newton._src.solvers.feather_pgs import solver_feather_pgs as original


class TestCoupledJacobiCPU(unittest.TestCase):
    def test_factory_abi_and_concurrent_boundary(self):
        """Retain both tier ABIs, one concurrent proposal owner and native cone law."""
        with patch.object(original, "_REGISTER_WHITENING", True):
            factory = coupled_jacobi.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
            for arch in (120, 103):
                for tier in (32, 48):
                    with self.subTest(arch=arch, tier=tier):
                        arguments = {
                            "rows": tier,
                            "min_rows": 0 if tier == 32 else 32,
                            "sweeps": 24,
                            "matrix_free": True,
                            "inkernel_response": (18, 0, 0, 0),
                            "exact_row_sums": True,
                            "world_rows": True,
                        }
                        old = original._get_pgs_solve_parallel_kernel(192, 1, 18, arch, **arguments)
                        new = factory(192, 1, 18, arch, **arguments)
                        self.assertTrue(new._fpgs_coupled_jacobi)
                        self.assertIn("_ccj24", new.key)
                        self.assertEqual([a.label for a in old.adj.args], [a.label for a in new.adj.args])
                        self.assertEqual(new._fpgs_block_dim, old._fpgs_block_dim)
                        source = new._fpgs_coupled_jacobi_native
                        self.assertIn("for(int pass=0;pass<24;++pass)", source)
                        self.assertIn("if(lane<n_rows&&s_kind[lane]==0)", source)
                        self.assertNotIn("while(row<n_rows)", source)
                        self.assertIn("cj_alpha==1.0f&&candidate_energy>cj_energy", source)
                        self.assertEqual(source.count("cj_alpha=0.5f;"), 1)
                        self.assertIn("const float cone_tolerance=3.0e-5f;", source)
                        self.assertIn("s_x[i] - s_lam0[i]", source)
                        if arch == 103 and tier == 32:
                            self.assertIn("float* s_L = s_Yt", source)
                            self.assertIn("s_L[e] = ink_L_a.data", source)
            self.assertEqual(inspect.signature(factory), inspect.signature(original._get_pgs_solve_parallel_kernel))

    def test_unsupported_factory_retains_original(self):
        """Keep unsupported dimensions and phases on the unchanged original owner."""
        factory = coupled_jacobi.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
        arguments = {
            "rows": 32,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "exact_row_sums": True,
            "world_rows": True,
        }
        for changed in ({"sweeps": 8}, {"has_drive_rows": True}, {"nesterov": False}):
            kernel = factory(192, 1, 18, 120, **(arguments | changed))
            self.assertFalse(getattr(kernel, "_fpgs_coupled_jacobi", False))


if __name__ == "__main__":
    unittest.main()

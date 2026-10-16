# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable source/ABI checks for the experimental sparse solve factories.

The native-body digests were recorded from the measured checkpoint
301b6688ea74a7372a007f03989664890f0d2540 before replacing Python-source cloning,
and re-recorded when the bodies took their coordinate count, support width and
physical-coordinate permutation from the plan instead of the checked tree's
constants. They protect those refactors, not numerical correctness.
No robot asset, local capture, CUDA device, or external checkout is required.
"""

import hashlib
import unittest

from newton._src.solvers.feather_pgs import sparse_factor_rows as rows
from newton._src.solvers.feather_pgs import sparse_limit_jacobi as limits
from newton._src.solvers.feather_pgs import sparse_spectral_tangents as spectral

# The checked 44-body, 43-coordinate tree the digests were recorded on.
DIMS = {"dof_count": 43, "support_width": 18, "inverse_width": 18, "nonzero_count": 434}


class TestSparseSolveFactory(unittest.TestCase):
    def test_frozen_native_body_and_abi(self):
        """Preserve every native body, kernel key, and argument order exactly."""
        cases = (
            (
                {"capacity": 64},
                "sparse_factor_gs43_s18_c64",
                "645b782269a17d487f0b2e2bd6df5b69b2310f76e54f99c659162a3d2a8aab16",
            ),
            (
                {"capacity": 100},
                "sparse_factor_gs43_s18_c100",
                "17e7c055a41eb989dd04f79b0d48b7d5c9e6dc89bfdd1ac8238cad155cbc7332",
            ),
            (
                {"capacity": 100, "block_contacts": True},
                "sparse_contact_block43_s18_c100",
                "acbfb8d619d246b9e5272c8eedd2a6a3c5924e0e97ae944058b413f5dba44b7b",
            ),
            (
                {"capacity": 100, "metric_tangents": True},
                "sparse_metric_tangent43_s18_c100",
                "a72bf0587e4dd63eb3ae70a9026cb5d3e96598aa0827efb2d568f451625176f9",
            ),
            (
                {"capacity": 100, "metric_tangents": True, "spectral_tangents": True},
                "sparse_spectral_tangent43_s18_c100",
                "ba1b84b3262acfd978b666edc2a33d65633c49cd749794d774c156345bf64d03",
            ),
            (
                {"capacity": 100, "metric_tangents": True, "spectral_tangents": True, "limit_jacobi": True},
                "sparse_spectral_limit_jacobi43_s18_c100",
                "d3fdb24a86b79e532de12f6fe540a2478f5159981a4ca79823b52bd3ffc83b96",
            ),
        )
        for options, key, digest in cases:
            with self.subTest(key=key):
                kernel = rows.get_solve_kernel(**options, **DIMS)
                native = next(
                    cell.cell_contents
                    for cell in kernel.func.__closure__
                    if hasattr(cell.cell_contents, "native_snippet")
                )
                expected = ["p", "d", "counts", "rhs", "diagonal"]
                if options.get("spectral_tangents"):
                    expected.append("cfm")
                expected.extend(
                    ("impulses", "row_type", "parent", "mu", "iterations", "omega", "friction_start", "vhat", "vout")
                )
                self.assertEqual(kernel.key, key)
                self.assertEqual(hashlib.sha256(native.native_snippet.encode()).hexdigest(), digest)
                self.assertEqual(list(native.input_types), ["group", *expected])
                self.assertEqual([arg.label for arg in kernel.adj.args], expected)

    def test_wrappers_share_explicit_factory(self):
        """Make retained entry points use a single cached, explicit factory."""
        self.assertIs(
            spectral.get_solve_kernel(**DIMS),
            rows.get_solve_kernel(100, metric_tangents=True, spectral_tangents=True, **DIMS),
        )
        self.assertIs(
            limits.get_solve_kernel(**DIMS),
            rows.get_solve_kernel(100, metric_tangents=True, spectral_tangents=True, limit_jacobi=True, **DIMS),
        )

    def test_reject_inconsistent_modes(self):
        """Reject unsupported combinations instead of silently changing laws."""
        cases = (
            {"spectral_tangents": True},
            {"limit_jacobi": True},
            {"metric_tangents": True, "limit_jacobi": True},
            {"metric_tangents": True, "spectral_tangents": True, "block_contacts": True},
        )
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                rows.get_solve_kernel(100, **options, **DIMS)
        with self.assertRaises(ValueError):
            rows.get_solve_kernel(64, metric_tangents=True, spectral_tangents=True, **DIMS)


if __name__ == "__main__":
    unittest.main()

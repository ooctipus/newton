# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable source/ABI checks for the experimental sparse solve factories.

The native-body digests were recorded from the measured checkpoint
301b6688ea74a7372a007f03989664890f0d2540 before replacing Python-source cloning.
They protect this refactor, not numerical correctness or future solver changes.
No robot asset, local capture, CUDA device, or external checkout is required.
"""

import hashlib
import unittest

from newton._src.solvers.feather_pgs import sparse_factor_rows as rows
from newton._src.solvers.feather_pgs import sparse_limit_jacobi as limits
from newton._src.solvers.feather_pgs import sparse_spectral_tangents as spectral


class TestSparseSolveFactory(unittest.TestCase):
    def test_frozen_native_body_and_abi(self):
        """Preserve every native body, kernel key, and argument order exactly."""
        cases = (
            (
                {"capacity": 64},
                "sparse_factor_gs43_s18_c64",
                "e9c44b0a52c30e6348df6da65a0a9306952df5ad13bfdde5cab4524f9b44b62b",
            ),
            (
                {"capacity": 100},
                "sparse_factor_gs43_s18_c100",
                "d376d30ba6ba55702e9a525c913a92fed719d94caa569984a1dccb387d8d3f6e",
            ),
            (
                {"capacity": 100, "block_contacts": True},
                "sparse_contact_block43_s18_c100",
                "e3a2506911816209214141fd10c9397587363982bbd47c35945ba69a9b010e73",
            ),
            (
                {"capacity": 100, "metric_tangents": True},
                "sparse_metric_tangent43_s18_c100",
                "eda6abf58c026c3abf9ebe6196c7790ed90444a0e5ce99f061d09fbd8a364352",
            ),
            (
                {"capacity": 100, "metric_tangents": True, "spectral_tangents": True},
                "sparse_spectral_tangent43_s18_c100",
                "552a94ec451f352c3e1834fda6fd5cde05740c3c6901ea0dd696d1850fcd3eac",
            ),
            (
                {"capacity": 100, "metric_tangents": True, "spectral_tangents": True, "limit_jacobi": True},
                "sparse_spectral_limit_jacobi43_s18_c100",
                "1df79d9479ba37d562f3367f232de272ed3e7a8b05bb6e18b8f230369936dc6f",
            ),
        )
        for options, key, digest in cases:
            with self.subTest(key=key):
                kernel = rows.get_solve_kernel(**options)
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
            spectral.get_solve_kernel(),
            rows.get_solve_kernel(100, metric_tangents=True, spectral_tangents=True),
        )
        self.assertIs(
            limits.get_solve_kernel(),
            rows.get_solve_kernel(100, metric_tangents=True, spectral_tangents=True, limit_jacobi=True),
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
                rows.get_solve_kernel(100, **options)
        with self.assertRaises(ValueError):
            rows.get_solve_kernel(64, metric_tangents=True, spectral_tangents=True)


if __name__ == "__main__":
    unittest.main()

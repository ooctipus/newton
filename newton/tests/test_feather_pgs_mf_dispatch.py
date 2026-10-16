# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve the matrix-free world-queue dispatch and its native ABI."""

import ast
import hashlib
import inspect
import textwrap
import unittest

from newton._src.solvers.feather_pgs import mf_dispatch
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_mf_gs_kernel


class TestFeatherPGSMFDispatch(unittest.TestCase):
    """Check the shared ABI, the retained dispatch body and negative admission."""

    def test_dispatch_preserves_complete_retained_ast(self):
        """Match the dispatch captured from 4534ffb4 before removing extraction."""
        kernel = _get_pgs_solve_mf_gs_kernel(
            192,
            64,
            29,
            "cpu",
            has_drive_rows=False,
            has_dense_velocity_limit_rows=False,
            factor_coordinates=True,
        )
        self.assertEqual(kernel.func.mf_dispatch_policy, "plain")
        node = ast.parse(textwrap.dedent(inspect.getsource(kernel.func))).body[0]
        node.decorator_list = []
        actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        self.assertEqual(actual, "4073c76b80171645d46ad8b123ef61ae690a469b2fdd9a449735aaa3f8e0d553")

    def test_private_operands_cannot_shadow_original_abi(self):
        """Reject duplicated operands before producing a native declaration."""
        with self.assertRaisesRegex(ValueError, "distinct"):
            mf_dispatch.make_native("", extra={"world": int})
        with self.assertRaisesRegex(ValueError, "valid name"):
            mf_dispatch.make_native("", name="bad name")


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve the native ABI of the matrix-free world-queue dispatch."""

import unittest

from newton._src.solvers.feather_pgs import mf_dispatch


class TestFeatherPGSMFDispatch(unittest.TestCase):
    """Check negative admission of operands that would change the shared ABI.

    The point-stream pins hold the dispatch body: every matrix-free
    Gauss-Seidel kernel's content hash covers it.
    """

    def test_private_operands_cannot_shadow_original_abi(self):
        """Reject duplicated operands before producing a native declaration."""
        with self.assertRaisesRegex(ValueError, "distinct"):
            mf_dispatch.make_native("", extra={"world": int})
        with self.assertRaisesRegex(ValueError, "valid name"):
            mf_dispatch.make_native("", name="bad name")


if __name__ == "__main__":
    unittest.main()

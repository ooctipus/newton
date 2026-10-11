# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check retained entry-point contracts for shared publication."""

import unittest

from newton._src.solvers.feather_pgs import world_scan_publication as publication


class TestWorldScanResolvedSource(unittest.TestCase):
    def test_current_resolved_api(self):
        """Require the new three-operand variant while preserving standalone ABI."""
        original = publication.get_kernel("cpu")
        resolved = publication.get_resolved_kernel("cpu")
        self.assertEqual([arg.label for arg in original.adj.args], ["plan", "data"])
        self.assertEqual([arg.label for arg in resolved.adj.args], ["plan", "data", "resolved"])
        self.assertIs(publication.get_resolved_kernel("cpu"), resolved)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The static shared-memory budget of a device architecture."""

import unittest

from newton._src.solvers.feather_pgs.staging import STATIC_SHARED_BUDGETS, STATIC_SHARED_BYTES, static_budget


class TestStaticBudget(unittest.TestCase):
    """The static shared-memory budget of a device architecture."""

    def test_budget(self):
        self.assertEqual(static_budget(12345), STATIC_SHARED_BYTES)  # no device of the architecture: offline
        STATIC_SHARED_BUDGETS[12345] = 81724
        try:
            self.assertEqual(static_budget(12345), 81724)
        finally:
            del STATIC_SHARED_BUDGETS[12345]


if __name__ == "__main__":
    unittest.main(verbosity=2)

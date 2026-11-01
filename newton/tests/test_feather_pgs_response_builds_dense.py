# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The dense kit on its own: its triangular solves' text and the static shared-memory budget."""

import unittest

from newton._src.solvers.feather_pgs.response_operator import CholeskySolve
from newton._src.solvers.feather_pgs.staging import STATIC_SHARED_BUDGETS, STATIC_SHARED_BYTES, static_budget

CPU = "cpu"


class TestKit(unittest.TestCase):
    """The small dense kit's text."""

    def test_solve_text_keeps_its_default(self):
        """A matrix operand reads the factor as the storage form does, and the defaults add no unroll directive."""

        def x(i):
            return f"x[{i}]"

        def y(i):
            return f"y[{i}]"

        storage = CholeskySolve(6, "L")
        accessor = CholeskySolve(6, lambda r, c: f"(L)[{r}*6+{c}]")
        self.assertEqual(storage.whiten(x, y), accessor.whiten(x, y))
        self.assertEqual(storage.leave(y, y), accessor.leave(y, y))
        self.assertNotIn("#pragma", storage.whiten(x, y) + storage.leave(y, y))
        unrolled = CholeskySolve(6, "L", unroll=True)
        self.assertEqual(unrolled.whiten(x, y).count("#pragma unroll"), 2)
        self.assertEqual(unrolled.leave(y, y).count("#pragma unroll"), 2)


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

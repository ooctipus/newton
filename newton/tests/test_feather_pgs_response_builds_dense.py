# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The dense response builds on their own: the kit's factor against NumPy, its text, and the operators' registry of
builds by stage and mapping."""

import unittest

from newton._src.solvers.feather_pgs import response_builds_dense as builds
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


class TestRegistry(unittest.TestCase):
    """The operators' builds by stage and mapping."""

    def test_dense(self):
        tiles = builds.DENSE_JY_BUILDS["response", "tiles"](
            n_dofs=18, max_constraints=32, tile_threads=64, compute_diag=True
        )
        self.assertEqual((tiles.block_dim, tiles.tiled, tiles.grid), (64, True, ("articulations", "row chunks")))
        levels = builds.DENSE_JY_BUILDS["factor", "levels"](size=9)
        self.assertEqual((levels.kernel, levels.tiled), (builds.dense_factor_kernel(9), False))
        for stage, _mapping in builds.DENSE_JY_BUILDS:
            self.assertIn(stage, builds.STAGES)


if __name__ == "__main__":
    unittest.main(verbosity=2)

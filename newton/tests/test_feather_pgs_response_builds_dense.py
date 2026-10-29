# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The dense response builds on their own: the kit's factor against NumPy, its text, the streaming Delassus's
staging, and the operators' registry of builds by stage and mapping."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import response_builds_dense as builds
from newton._src.solvers.feather_pgs.response_operator import Assembled, CholeskyFactor, CholeskySolve
from newton._src.solvers.feather_pgs.staging import STATIC_SHARED_BUDGETS, STATIC_SHARED_BYTES, static_budget

CPU = "cpu"


class TestKit(unittest.TestCase):
    """The small dense kit's text."""

    def test_factor_matches_numpy(self):
        """The kit's column Cholesky factors a positive definite matrix leaf-first, ``P H P^T`` with the coordinates
        reversed (float32, against float64)."""
        rng = np.random.default_rng(0)
        n, arts = 9, 5
        A = rng.standard_normal((arts, n, n)).astype(np.float32)
        H = (A @ A.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)).astype(np.float32)
        R = rng.uniform(0.0, 0.1, (arts, n)).astype(np.float32)
        group_to_art = rng.permutation(arts).astype(np.int32)
        L = wp.zeros(H.shape, dtype=float, device=CPU)
        inputs = [wp.array(H, device=CPU), wp.array(R, device=CPU), wp.array(group_to_art, device=CPU)]
        inputs += [wp.array(np.ones(arts, dtype=np.int32), device=CPU), n]
        wp.launch(builds.factor_articulations_kernel(), arts, inputs=inputs, outputs=[L], device=CPU)
        expected = H.astype(np.float64) + np.einsum("gi,ij->gij", R.astype(np.float64), np.eye(n))
        np.testing.assert_allclose(
            np.tril(L.numpy()), np.linalg.cholesky(expected[:, ::-1, ::-1]), rtol=1e-4, atol=1e-5
        )

    def test_factor_text(self):
        factor = CholeskyFactor(3, lambda r, c: f"f[{r} * 3 + {c}]", unroll=True)
        text = factor.factor()
        self.assertEqual(text.count("#pragma unroll"), 4)
        self.assertIn("diagonal = sqrtf(diagonal);", text)
        self.assertIn("f[row * 3 + col] = value * inverse_diagonal;", text)
        self.assertEqual(factor.pivot("d"), "f[d * 3 + d] = sqrtf(f[d * 3 + d]);")

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
        build = builds.DENSE_JY_BUILDS["factor", "articulations"]()
        self.assertEqual((build.stage, build.mapping, build.tiled), ("factor", "articulations", False))
        self.assertIs(build.kernel, builds.factor_articulations_kernel())
        tiles = builds.DENSE_JY_BUILDS["response", "tiles"](
            n_dofs=18, max_constraints=32, tile_threads=64, compute_diag=True
        )
        self.assertEqual((tiles.block_dim, tiles.tiled, tiles.grid), (64, True, ("articulations", "row chunks")))
        warps = builds.DENSE_JY_BUILDS["factor", "warps"](size=9, geometric_index=tuple(range(81)), warps_per_block=2)
        self.assertEqual((warps.block_dim, warps.tiled), (64, False))
        for stage, _mapping in (*builds.DENSE_JY_BUILDS, *builds.ASSEMBLED_BUILDS):
            self.assertIn(stage, builds.STAGES)

    def test_assembled(self):
        build = Assembled.build("delassus", "entries")
        self.assertIs(build.kernel, builds.delassus_entries_kernel())
        with self.assertRaises(KeyError):
            Assembled.build("factor", "articulations")


if __name__ == "__main__":
    unittest.main(verbosity=2)

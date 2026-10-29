# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the held tree's warp-mapped factor build (``factor_warps_kernel``) against independent dense matrices."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.response_builds_dense import factor_warps_kernel, leaf_index


class TestSmallCholesky(unittest.TestCase):
    def test_reject_unaddressable_inputs(self):
        """Reject sizes, launch shapes and geometry indices outside the factor contract."""
        for size, index, options in (
            (0, (), {}),
            (33, tuple(range(33 * 33)), {}),
            (9, tuple(range(81)), {"warps_per_block": 0}),
            (9, tuple(range(81)), {"warps_per_block": 33}),
            (2, (0, 1), {}),
            (2, (0, 1, -2, 3), {}),
            (2, (0, 1, 2, 3), {"geometric_stride": 3}),
        ):
            with self.subTest(size=size, options=options), self.assertRaises(ValueError):
                factor_warps_kernel(size, index, **options)

    def _compare(self, size, warps, mode, augmented):
        rng = np.random.default_rng(19 + size)
        groups, articulations = 3, 7
        group_to_art = np.array([5, 2, 0], dtype=np.int32)
        masks = np.ones(articulations, dtype=np.int32)
        masks[group_to_art[1]] = 0
        factor = rng.normal(size=(groups, size, size))
        mass = factor @ np.swapaxes(factor, 1, 2) + size * np.eye(size)
        present = np.ones((size, size), dtype=bool)
        if mode == "indexed":
            block = np.arange(size) < size // 2
            present = block[:, None] == block[None, :]
            mass[:, ~present] = 0
        index = np.full((size, size), -1, dtype=np.int32)
        index[present] = np.arange(np.count_nonzero(present))
        # The kernel factors leaf-first: factor (row, col) reads physical (n - 1 - row, n - 1 - col).
        lookup = leaf_index(index)
        geometric = mass[:, present].astype(np.float32)
        regularizer = rng.uniform(1, 2, size=(groups, size)).astype(np.float32)
        drive = np.arange(articulations * size, dtype=np.int32)
        drive[::3] = -1
        if not augmented:
            drive[:] = -1
        drive_k = rng.uniform(-1, 1, size=articulations * size).astype(np.float32)
        expected = mass.astype(np.float32).astype(np.float64)
        for group, articulation in enumerate(group_to_art):
            rows = drive[articulation * size : (articulation + 1) * size]
            diagonal = regularizer[group] + np.where(rows >= 0, np.maximum(drive_k[np.maximum(rows, 0)], 0), 0)
            expected[group] += np.diag(diagonal)
        kernel = factor_warps_kernel(size, lookup, warps_per_block=warps)

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device="cuda:0")

        inputs = [
            groups,
            array(np.arange(articulations) * size, int),
            array(masks, int),
            array(group_to_art, int),
            array(regularizer),
            array([row * size + col for row in range(size) for col in range(row + 1)], int),
            array(drive, int),
            array(drive_k),
            array(geometric),
            array(lookup, int),
        ]
        output = wp.full((groups, size, size), -17.0, device="cuda:0")
        wp.launch(kernel, dim=groups * 32, block_dim=warps * 32, inputs=inputs, outputs=[output], device="cuda:0")
        observed = output.numpy()
        for group, articulation in enumerate(group_to_art):
            if not masks[articulation]:
                np.testing.assert_array_equal(observed[group], -17.0)
                continue
            lower = np.tril(observed[group])
            leaf_first = expected[group][::-1, ::-1]
            np.testing.assert_allclose(lower, np.linalg.cholesky(leaf_first), atol=2e-4, rtol=2e-5)
            np.testing.assert_allclose(lower @ lower.T, leaf_first, atol=3e-4, rtol=3e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "Small factor kernels require CUDA")
    def test_geometric_factor_cuda(self):
        """Preserve physical matrices, drive augmentation and inactive articulation factors."""
        for size in (1, 4, 9, 12, 16, 24, 32):
            for mode in ("full", "indexed"):
                for augmented in (False, True):
                    with self.subTest(size=size, mode=mode, augmented=augmented):
                        self._compare(size, 4, mode, augmented)

    @unittest.skipUnless(wp.is_cuda_available(), "Small factor kernels require CUDA")
    def test_single_warp_groups_cuda(self):
        """Use the same geometry with one warp per block."""
        for mode in ("full", "indexed"):
            with self.subTest(mode=mode):
                self._compare(9, 1, mode, True)


if __name__ == "__main__":
    unittest.main()

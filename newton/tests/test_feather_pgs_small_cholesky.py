# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the held tree's dense factor (``dense_factor_kernel``) against independent dense matrices."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.response_builds_dense import dense_factor_kernel, dense_factor_plan


def _devices() -> tuple:
    return ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",)


class TestDenseFactor(unittest.TestCase):
    def _compare(self, size, mode, augmented, device):
        rng = np.random.default_rng(19 + size)
        groups, articulations = 3, 7
        group_to_art = np.array([5, 2, 0], dtype=np.int32)
        masks = np.ones(articulations, dtype=np.int32)
        masks[group_to_art[1]] = 0
        factor = rng.normal(size=(groups, size, size))
        mass = factor @ np.swapaxes(factor, 1, 2) + size * np.eye(size)
        present = np.ones((size, size), dtype=bool)
        if mode == "blocks":
            block = np.arange(size) < size // 2
            present = block[:, None] == block[None, :]
        elif mode == "fill":
            # A star whose hub is the first coordinate reversal eliminates: its column fills every leaf pair.
            present = np.eye(size, dtype=bool)
            present[-1, :] = present[:, -1] = True
        mass[:, ~present] = 0
        index = np.full((size, size), -1, dtype=np.int32)
        index[present] = np.arange(np.count_nonzero(present))
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
            expected[group] += np.diag(
                regularizer[group] + np.where(rows >= 0, np.maximum(drive_k[np.maximum(rows, 0)], 0), 0)
            )

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        output = wp.zeros((groups, size, size), device=device)
        pivots = wp.zeros(drive_k.size, device=device)
        wp.launch(
            dense_factor_kernel(size),
            dim=groups * 32,
            inputs=[
                array(np.arange(articulations) * size, int),
                array(masks, int),
                array(group_to_art, int),
                array(np.arange(groups, dtype=np.int32), int),
                dense_factor_plan(index, device),
                array(regularizer),
                array(drive, int),
                array(drive_k),
                array(np.zeros_like(drive_k)),
                pivots,
                array(geometric),
            ],
            outputs=[output],
            block_dim=128,
            device=device,
        )
        observed = output.numpy()
        for group, articulation in enumerate(group_to_art):
            if not masks[articulation]:
                np.testing.assert_array_equal(observed[group], 0.0)
                continue
            lower = np.tril(observed[group])
            leaf_first = expected[group][::-1, ::-1]
            np.testing.assert_allclose(lower, np.linalg.cholesky(leaf_first), atol=2e-3, rtol=2e-4)
            np.testing.assert_allclose(lower @ lower.T, leaf_first, atol=2e-3, rtol=5e-5)
            # A drive's pivot without its K: its own pivot less the K the diagonal took (a release's pivot).
            rows = drive[articulation * size : (articulation + 1) * size][::-1]
            held = rows >= 0
            pivot = np.diag(lower).astype(np.float64) ** 2
            added = np.maximum(drive_k[rows[held]], 0.0)
            np.testing.assert_allclose(
                pivots.numpy()[rows[held]], pivot[held] - added, rtol=0.0, atol=2e-5 * float(pivot.max(initial=1.0))
            )

    def test_any_size_and_pattern(self):
        """Factor every pattern, fill included, with the drives on the diagonal, and leave inactive articulations."""
        for device in _devices():
            for size in (1, 4, 9, 12, 16, 24, 32, 33, 66, 106):
                for mode in ("full", "blocks", "fill"):
                    for augmented in (False, True):
                        with self.subTest(device=device, size=size, mode=mode, augmented=augmented):
                            self._compare(size, mode, augmented, device)


if __name__ == "__main__":
    unittest.main()

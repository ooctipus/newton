# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the warp-mapped factor build (``factor_warps_kernel``) against independent dense matrix construction."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.response_builds_dense import factor_warps_kernel
from newton._src.solvers.feather_pgs.spatial_laws import inertia_record
from newton.tests.test_feather_pgs_sparse_mass_matrix import _random_inertia


class TestSmallCholesky(unittest.TestCase):
    def test_reject_unaddressable_inputs(self):
        """Reject schedules and launch shapes outside the packed factor contract."""
        for size, options in (
            (0, {}),
            (17, {}),
            (9, {"warps_per_block": 0}),
            (9, {"warps_per_block": 33}),
            (2, {"geometric_index": (0, 1, 2, 3)}),
            (2, {"geometric": True, "geometric_index": (0, 1)}),
            (2, {"geometric": True, "geometric_index": (0, 1, -2, 3)}),
        ):
            with self.subTest(size=size, options=options), self.assertRaises(ValueError):
                factor_warps_kernel(size, **options)

    def _compare(self, size, warps, mode, augmented):
        rng = np.random.default_rng(19 + size)
        groups, articulations = 3, 7
        group_to_art = np.array([5, 2, 0], dtype=np.int32)
        masks = np.ones(articulations, dtype=np.int32)
        masks[group_to_art[1]] = 0
        motion = rng.normal(size=(articulations, size, 6)).astype(np.float32)
        pairs = [_random_inertia(rng) for _ in range(articulations)]
        records = np.asarray([record for record, _ in pairs], dtype=np.float32)
        inertia = np.asarray([dense for _, dense in pairs])
        regularizer = rng.uniform(1, 2, size=(groups, size)).astype(np.float32)
        drive = np.arange(articulations * size, dtype=np.int32)
        drive[::3] = -1
        drive_k = rng.uniform(-1, 1, size=articulations * size).astype(np.float32)
        mass = np.einsum("gnc,gcd,gmd->gnm", motion[group_to_art], inertia[group_to_art], motion[group_to_art])
        lookup = ()
        if mode == "indexed":
            block = np.arange(size) < size // 2
            present = block[:, None] == block[None, :]
            mass[:, ~present] = 0
            index = np.full((size, size), -1, dtype=np.int32)
            index[present] = np.arange(np.count_nonzero(present))
            lookup = tuple(map(int, index.ravel()))
            geometric = mass[:, present].copy()
        else:
            geometric = mass.reshape(groups, -1).copy()
        expected = mass.astype(np.float64)
        for group, articulation in enumerate(group_to_art):
            diagonal = regularizer[group].copy()
            if augmented:
                rows = drive[articulation * size : (articulation + 1) * size]
                diagonal += np.where(rows >= 0, np.maximum(drive_k[np.maximum(rows, 0)], 0), 0)
            expected[group] += np.diag(diagonal)
        kernel = factor_warps_kernel(size, warps_per_block=warps, geometric=mode != "ordinary", geometric_index=lookup)

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device="cuda:0")

        schedule = [row * size + col + ((col + 1) << 8) + (row << 16) for row in range(size) for col in range(row + 1)]
        inputs = [
            groups,
            array(np.arange(articulations) * size, int),
            array(np.arange(articulations) * size, int),
            array(masks, int),
            array(np.arange(articulations * size), int),
            array(motion.reshape(-1, 6), wp.spatial_vector),
            array(np.repeat(records, size, axis=0), inertia_record),
            array(group_to_art, int),
            array(regularizer),
            array(np.arange(size), int),
            array(schedule, int),
            int(augmented),
            array(drive, int),
            array(drive_k),
        ]
        if mode != "ordinary":
            inputs.append(array(geometric))
            if lookup:
                inputs.append(array(lookup, int))
        output = wp.full((groups, size, size), -17.0, device="cuda:0")
        wp.launch(kernel, dim=groups * 32, block_dim=warps * 32, inputs=inputs, outputs=[output], device="cuda:0")
        observed = output.numpy()
        for group, articulation in enumerate(group_to_art):
            if not masks[articulation]:
                np.testing.assert_array_equal(observed[group], -17.0)
                continue
            lower = np.tril(observed[group])
            np.testing.assert_allclose(lower, np.linalg.cholesky(expected[group]), atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(lower @ lower.T, expected[group], atol=3e-5, rtol=3e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "Small factor kernels require CUDA")
    def test_composite_and_geometric_factor_cuda(self):
        """Preserve physical matrices, drive augmentation and inactive articulation factors."""
        for size in (4, 9, 12, 16):
            for mode in ("ordinary", "geometric", "indexed"):
                for augmented in (False, True):
                    with self.subTest(size=size, mode=mode, augmented=augmented):
                        self._compare(size, 4, mode, augmented)

    @unittest.skipUnless(wp.is_cuda_available(), "Small factor kernels require CUDA")
    def test_single_warp_groups_cuda(self):
        """Use the same geometric and current-state inputs with one warp per block."""
        for mode in ("ordinary", "geometric", "indexed"):
            with self.subTest(mode=mode):
                self._compare(9, 1, mode, True)


if __name__ == "__main__":
    unittest.main()

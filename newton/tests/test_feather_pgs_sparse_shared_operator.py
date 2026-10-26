# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check shared sparse operators on structurally different articulations."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sparse_factorization import _sparse_inverse_source
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseMassMatrixIndices, factor_indices
from newton._src.solvers.feather_pgs.tree_plan import TreePlan


def _inverse_kernel(dof_count, local_column_width):
    """Exercise the storage mappings without a task-specific producer."""
    source = (
        """
#if defined(__CUDA_ARCH__)
    const int lane = tid, stride = 32;
    const float* factor = packed.data;
    float* inverse = result.data;
    __shared__ int bad;
    if (lane == 0) bad = 0;
    __syncwarp();
    """
        + _sparse_inverse_source(dof_count, local_column_width=local_column_width)
        + """
    __syncwarp();
    if (lane == 0) status.data[0] = bad;
#endif
    """
    )

    @wp.func_native(source)
    def native(
        tid: int,
        indices: _SparseMassMatrixIndices,
        packed: wp.array[float],
        result: wp.array[float],
        status: wp.array[int],
    ): ...

    def kernel(
        indices: _SparseMassMatrixIndices,
        packed: wp.array[float],
        result: wp.array[float],
        status: wp.array[int],
    ):
        native(wp.tid(), indices, packed, result, status)

    return wp.kernel(module="unique", enable_backward=False)(kernel)


class TestSparseSharedOperator(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Require CUDA for local inverse storage")
    def test_inverse_storage_mappings_match_dense_solve(self):
        """Check the shared and local inverse mappings on branched, reordered and scalar topologies."""
        for parents, counts in [([-1], [1]), ([-1, 0, 0, 1], [6, 3, 2, 0]), ([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2])]:
            plan = TreePlan.build(parents, counts).factor
            rng = np.random.default_rng(1067)
            factor = np.zeros_like(plan.lookup, dtype=np.float32)
            factor[plan.entry_rows, plan.columns] = rng.normal(0, 0.1, plan.nonzero_count)
            factor[np.diag_indices(plan.dof_count)] += 2
            indices = factor_indices(plan, "cuda:0")
            local_width = int(np.max(np.diff(plan.column_offsets)))
            packed = wp.array(factor[plan.entry_rows, plan.columns], dtype=float, device="cuda:0")
            expected = np.linalg.inv(factor)[plan.entry_rows, plan.columns]
            for width in (0, local_width):
                with self.subTest(parents=parents, counts=counts, local_column_width=width):
                    result = wp.full(plan.nonzero_count, np.nan, dtype=float, device="cuda:0")
                    status = wp.full(1, -1, dtype=int, device="cuda:0")
                    wp.launch(
                        _inverse_kernel(plan.dof_count, width),
                        dim=32,
                        block_dim=32,
                        inputs=[indices, packed, result, status],
                        device="cuda:0",
                    )
                    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6, atol=1e-7)
                    self.assertEqual(status.numpy()[0], 0)


if __name__ == "__main__":
    unittest.main()

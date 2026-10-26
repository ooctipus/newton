# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate generated sparse factorization against independent dense matrices."""

import unittest
from functools import cache

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_factorization
from newton._src.solvers.feather_pgs.device_preludes import cuda_native, lane_group
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseMassMatrixIndices, factor_indices
from newton._src.solvers.feather_pgs.tree_plan import TreePlan


@cache
def _factor_kernel(dofs, nonzeros):
    """Bind the factorization without a robot-specific producer or cache."""
    source = cuda_native(
        f"""    const int lane=LANE, stride=STRIDE, group=0;
    __shared__ float factor[{nonzeros}];
    if(lane==0) status.data[0]=0;
    for(int entry=lane;entry<{nonzeros};entry+=stride)factor[entry]=matrix.data[entry];
    SYNC();
    {sparse_factorization._sparse_cholesky_source(dofs, nonzeros)}
    SYNC();
    for(int entry=lane;entry<{nonzeros};entry+=stride)result.data[entry]=factor[entry];
#endif
""",
        preludes=(lane_group(32),),
    )

    @wp.func_native(source)
    def native(
        indices: _SparseMassMatrixIndices, matrix: wp.array[float], result: wp.array[float], status: wp.array[int]
    ): ...

    def factor(
        indices: _SparseMassMatrixIndices, matrix: wp.array[float], result: wp.array[float], status: wp.array[int]
    ):
        native(indices, matrix, result, status)

    return wp.kernel(enable_backward=False, module="unique")(factor)


class TestSparseFactorization(unittest.TestCase):
    def test_reject_invalid_mapping_contract(self):
        """Reject dimensions that cannot describe the emitted work."""
        emit = sparse_factorization._sparse_cholesky_source
        for n, nz in [(0, 0), (65, 65), (4, 3), (4, 11)]:
            with self.subTest(n=n, nz=nz), self.assertRaises(ValueError):
                emit(n, nz)

    @unittest.skipUnless(wp.is_cuda_available(), "Require CUDA for block factor mappings")
    def test_topology_variants_and_bad_pivots(self):
        """Factor diverse forests and report nonpositive pivots without a floor."""
        cases = [
            ([-1], [1]),
            ([-1, 0, 0, 1], [6, 3, 2, 0]),
            ([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2]),
            ([-1, 0, 0, 0, 0], [6, 3, 3, 3, 3]),
            ([-1, 0, 0, 1, 2], [6, 5, 5, 5, 4]),
            ([-1, 0, 0, 1, 2], [6, 10, 10, 9, 8]),
            ([-1, 0, 0, 1, 2], [6, 15, 15, 14, 14]),
        ]
        for parents, counts in cases:
            plan = TreePlan.build(parents, counts).factor
            rng = np.random.default_rng(5208)
            expected = np.zeros_like(plan.lookup, dtype=np.float64)
            expected[plan.entry_rows, plan.columns] = rng.normal(0, 0.05, plan.nonzero_count)
            expected[np.diag_indices(plan.dof_count)] += 2
            matrix = expected @ expected.T
            # The independently computed dense Cholesky defines the reference.
            reference = np.linalg.cholesky(matrix)[plan.entry_rows, plan.columns]
            packed = matrix[plan.entry_rows, plan.columns].astype(np.float32)
            kernel = _factor_kernel(plan.dof_count, plan.nonzero_count)
            bound = factor_indices(plan, "cuda:0")
            for invalid in (False, True):
                with self.subTest(parents=parents, counts=counts, invalid=invalid):
                    values = packed.copy()
                    if invalid:
                        values[plan.lookup[0, 0]] = -1
                    a = wp.array(values, dtype=float, device="cuda:0")
                    result = wp.full(plan.nonzero_count, np.nan, dtype=float, device="cuda:0")
                    status = wp.full(1, -1, dtype=int, device="cuda:0")
                    wp.launch_tiled(kernel, dim=(1,), inputs=[bound, a, result, status], block_dim=32, device="cuda:0")
                    self.assertEqual(status.numpy()[0], int(invalid))
                    if not invalid:
                        np.testing.assert_allclose(result.numpy(), reference, rtol=3e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate generated sparse factorization against independent dense matrices."""

import unittest
from functools import cache

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_factorization
from newton._src.solvers.feather_pgs.sparse_mass_matrix import (
    _factor_update_schedule,
    _SparseMassMatrixIndices,
    _SparseMassMatrixPlan,
)


@wp.struct
class _UpdateIndices:
    index: wp.array2d[int]
    row: wp.array[int]
    col: wp.array[int]
    level_offsets: wp.array2d[int]
    level_pivots: wp.array[int]
    level_scales: wp.array[int]
    level_entries: wp.array[int]
    level_term_offsets: wp.array[int]
    level_terms: wp.array2d[int]


@cache
def _factor_kernel(dofs, nonzeros, levels, algorithm):
    """Bind each mapping without a robot-specific producer or cache."""
    index_type = _SparseMassMatrixIndices if algorithm == "left_looking" else _UpdateIndices
    source = f"""
#if defined(__CUDA_ARCH__)
    const int lane=threadIdx.x, stride=blockDim.x, group=0;
    __shared__ float factor[{nonzeros}];
    __shared__ int bad;
    if(lane==0) {{bad=0;status.data[0]=0;}}
    for(int entry=lane;entry<{nonzeros};entry+=stride)factor[entry]=matrix.data[entry];
    __syncthreads();
    {sparse_factorization._sparse_cholesky_source(dofs, nonzeros, algorithm=algorithm, level_count=levels)}
    __syncthreads();
    if(lane==0 && bad)status.data[0]=1;
    for(int entry=lane;entry<{nonzeros};entry+=stride)result.data[entry]=factor[entry];
#endif
"""

    @wp.func_native(source)
    def native(indices: index_type, matrix: wp.array[float], result: wp.array[float], status: wp.array[int]): ...

    def factor(indices: index_type, matrix: wp.array[float], result: wp.array[float], status: wp.array[int]):
        native(indices, matrix, result, status)

    return wp.kernel(enable_backward=False, module="unique")(factor)


class TestSparseFactorization(unittest.TestCase):
    def test_reject_invalid_mapping_contract(self):
        """Reject dimensions and schedules that cannot describe the emitted work."""
        emit = sparse_factorization._sparse_cholesky_source
        for n, nz, algorithm, levels in [
            (0, 0, "left_looking", 0),
            (65, 65, "right_looking", 0),
            (4, 3, "left_looking", 0),
            (4, 11, "right_looking", 0),
            (4, 6, "unknown", 0),
            (4, 6, "right_looking_levels", 0),
            (4, 6, "right_looking_levels", 5),
        ]:
            with self.subTest(n=n, nz=nz, algorithm=algorithm, levels=levels):
                with self.assertRaises(ValueError):
                    emit(n, nz, algorithm=algorithm, level_count=levels)

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
            plan = _SparseMassMatrixPlan.build(parents, counts)
            schedule = _factor_update_schedule(plan.lookup)
            indices = _UpdateIndices()
            for name, values in dict(index=plan.lookup, row=plan.entry_rows, col=plan.columns, **schedule).items():
                setattr(indices, name, wp.array(values, dtype=int, device="cuda:0"))
            rng = np.random.default_rng(5208)
            expected = np.zeros_like(plan.lookup, dtype=np.float64)
            expected[plan.entry_rows, plan.columns] = rng.normal(0, 0.05, plan.nonzero_count)
            expected[np.diag_indices(plan.dof_count)] += 2
            matrix = expected @ expected.T
            # The independently computed dense Cholesky defines the reference.
            reference = np.linalg.cholesky(matrix)[plan.entry_rows, plan.columns]
            packed = matrix[plan.entry_rows, plan.columns].astype(np.float32)
            for algorithm in ("left_looking", "right_looking", "right_looking_levels"):
                kernel = _factor_kernel(plan.dof_count, plan.nonzero_count, plan.factor_level_count, algorithm)
                bound = plan.to_device("cuda:0") if algorithm == "left_looking" else indices
                for invalid in (False, True):
                    with self.subTest(parents=parents, counts=counts, algorithm=algorithm, invalid=invalid):
                        values = packed.copy()
                        if invalid:
                            values[plan.lookup[0, 0]] = -1
                        a = wp.array(values, dtype=float, device="cuda:0")
                        result = wp.full(plan.nonzero_count, np.nan, dtype=float, device="cuda:0")
                        status = wp.full(1, -1, dtype=int, device="cuda:0")
                        block = 32 if algorithm == "left_looking" else 128
                        wp.launch_tiled(
                            kernel, dim=(1,), inputs=[bound, a, result, status], block_dim=block, device="cuda:0"
                        )
                        self.assertEqual(status.numpy()[0], int(invalid))
                        if not invalid:
                            np.testing.assert_allclose(result.numpy(), reference, rtol=3e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

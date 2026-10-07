# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check shared sparse operators on structurally different articulations."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sparse_factorization import _sparse_inverse_source
from newton._src.solvers.feather_pgs.sparse_mass_matrix import (
    _factor_update_schedule,
    _SparseMassMatrixIndices,
    _SparseMassMatrixPlan,
    _SparseRowSupportPlan,
)


@wp.struct
class _PaddedColumns:
    index: wp.array2d[int]
    inverse_nodes: wp.array2d[int]
    inverse_count: wp.array[int]


def _inverse_kernel(dof_count, local_column_width, column_layout):
    """Exercise storage mappings without a task-specific producer."""
    index_type = _SparseMassMatrixIndices if column_layout == "csc" else _PaddedColumns
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
        + _sparse_inverse_source(dof_count, local_column_width=local_column_width, column_layout=column_layout)
        + """
    __syncwarp();
    if (lane == 0) status.data[0] = bad;
#endif
    """
    )

    @wp.func_native(source)
    def native(
        tid: int,
        indices: index_type,
        packed: wp.array[float],
        result: wp.array[float],
        status: wp.array[int],
    ): ...

    def kernel(
        indices: index_type,
        packed: wp.array[float],
        result: wp.array[float],
        status: wp.array[int],
    ):
        native(wp.tid(), indices, packed, result, status)

    return wp.kernel(module="unique", enable_backward=False)(kernel)


class TestSparseSharedOperator(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Require CUDA for local inverse storage")
    def test_inverse_storage_mappings_match_dense_solve(self):
        """Check all inverse mappings on branched, reordered and scalar topologies."""
        for parents, counts in [([-1], [1]), ([-1, 0, 0, 1], [6, 3, 2, 0]), ([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2])]:
            plan = _SparseMassMatrixPlan.build(parents, counts)
            rng = np.random.default_rng(1067)
            factor = np.zeros_like(plan.lookup, dtype=np.float32)
            factor[plan.entry_rows, plan.columns] = rng.normal(0, 0.1, plan.nonzero_count)
            factor[np.diag_indices(plan.dof_count)] += 2
            csc = plan.to_device("cuda:0")
            local_width = int(np.max(np.diff(plan.column_offsets)))
            nodes = np.full((plan.dof_count, local_width), -1, np.int32)
            for col in range(plan.dof_count):
                begin, end = plan.column_offsets[col : col + 2]
                nodes[col, : end - begin] = plan.entry_rows[plan.column_entries[begin:end]]
            padded = _PaddedColumns()
            padded.index = csc.lookup
            padded.inverse_nodes = wp.array(nodes, dtype=int, device="cuda:0")
            padded.inverse_count = wp.array(np.diff(plan.column_offsets), dtype=int, device="cuda:0")
            packed = wp.array(factor[plan.entry_rows, plan.columns], dtype=float, device="cuda:0")
            expected = np.linalg.inv(factor)[plan.entry_rows, plan.columns]
            for width, layout, indices in [(0, "csc", csc), (local_width, "csc", csc), (local_width, "padded", padded)]:
                with self.subTest(parents=parents, counts=counts, local_column_width=width, layout=layout):
                    result = wp.full(plan.nonzero_count, np.nan, dtype=float, device="cuda:0")
                    status = wp.full(1, -1, dtype=int, device="cuda:0")
                    wp.launch(
                        _inverse_kernel(plan.dof_count, width, layout),
                        dim=32,
                        block_dim=32,
                        inputs=[indices, packed, result, status],
                        device="cuda:0",
                    )
                    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-6, atol=1e-7)
                    self.assertEqual(status.numpy()[0], 0)

    def test_level_updates_reconstruct_dense_operator(self):
        """Factor positive definite forest matrices with generic Schur schedules."""
        for parents, counts in [([-1], [1]), ([-1, 0, 0, 1], [6, 3, 2, 0]), ([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2])]:
            with self.subTest(parents=parents, counts=counts):
                plan = _SparseMassMatrixPlan.build(parents, counts)
                schedule = _factor_update_schedule(plan.lookup)
                rng = np.random.default_rng(1027)
                factor = np.zeros_like(plan.lookup, dtype=float)
                factor[plan.entry_rows, plan.columns] = rng.normal(0, 0.1, plan.nonzero_count)
                factor[np.diag_indices(plan.dof_count)] += 2
                matrix = factor @ factor.T
                packed = matrix[plan.entry_rows, plan.columns].copy()
                offsets = schedule["level_offsets"]
                np.testing.assert_array_equal(offsets[:, 0], plan.factor_level_offsets)
                np.testing.assert_array_equal(schedule["level_pivots"], plan.factor_level_columns)
                for level in range(len(offsets) - 1):
                    for col in schedule["level_pivots"][offsets[level, 0] : offsets[level + 1, 0]]:
                        packed[plan.lookup[col, col]] = np.sqrt(packed[plan.lookup[col, col]])
                    for entry in schedule["level_scales"][offsets[level, 1] : offsets[level + 1, 1]]:
                        col = plan.columns[entry]
                        packed[entry] /= packed[plan.lookup[col, col]]
                    for work in range(offsets[level, 2], offsets[level + 1, 2]):
                        start, end = schedule["level_term_offsets"][work : work + 2]
                        for first, second in schedule["level_terms"][start:end]:
                            packed[schedule["level_entries"][work]] -= packed[first] * packed[second]
                np.testing.assert_allclose(packed, factor[plan.entry_rows, plan.columns], atol=1e-12)

    def test_contact_catalogue_keeps_all_limit_supports(self):
        """Restrict contacts without truncating other joints' limit responses."""
        plan = _SparseMassMatrixPlan.build([-1, 0, 0], [6, 3, 3])
        masks = plan.joint_ancestor_mask
        self.assertIsNone(_SparseRowSupportPlan.build(plan, masks, max_row_dofs=9))
        support = _SparseRowSupportPlan.build(plan, masks, max_row_dofs=9, contact_bodies=[1])
        self.assertIsNotNone(support)
        self.assertEqual(support.body_class[2], -1)
        for dof, mask in enumerate(plan.ancestor_mask):
            nodes = support.class_dof[support.limit_class[dof]]
            expected = [index for index in range(plan.dof_count) if int(mask) & (1 << index)]
            np.testing.assert_array_equal(nodes[nodes >= 0], expected)


if __name__ == "__main__":
    unittest.main()

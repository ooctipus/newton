# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the generated shared-operand projection with dense factor action."""

import unittest
from functools import cache

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.device_preludes import dual_native, lane_group
from newton._src.solvers.feather_pgs.sparse_row_projection import _sparse_contact_projection_source
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.tests.test_feather_pgs_sparse_mass_matrix import _endpoint_support


@cache
def projection_kernel(n, nnz):
    """Execute production projection source with independently supplied operands."""
    fragment = _sparse_contact_projection_source("shared_mask", dof_count=n, scratch_stride=n)
    body = f"""
    #if defined(__CUDA_ARCH__)
    __shared__ float jacobian[{3 * n}];
    #else
    float jacobian[{3 * n}];
    #endif
    for(int i=LANE;i<{3 * n};i+=STRIDE)jacobian[i]=input.data[i];
    SYNC();
    const int factor_base=0;
    for(int node=LANE;node<{n};node+=STRIDE){{
        if((mask&(1ull<<node))==0)continue;
        {fragment}
        for(int r=0;r<3;++r)output.data[node*3+r]=value.c[r];
    }}
    """

    @wp.func_native(dual_native(body, preludes=(lane_group(32),)))
    def native(
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        input: wp.array[float],
        mask: wp.uint64,
        row_count: int,
        output: wp.array[float],
    ): ...

    def kernel(
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        input: wp.array[float],
        mask: wp.uint64,
        row_count: int,
        output: wp.array[float],
    ):
        native(factor_row_offsets, inverse_factor, input, mask, row_count, output)

    kernel.__name__ = kernel.__qualname__ = f"row_projection_shared_mask_{n}_{nnz}"
    return wp.kernel(enable_backward=False, module="unique")(kernel)


class TestSparseRowProjection(unittest.TestCase):
    def test_invalid_mapping_dimensions(self):
        """Reject layouts that would access invalid storage and unknown mappings."""
        for mapping, dimensions in [
            ("shared_mask", {"dof_count": 0, "scratch_stride": 4}),
            ("shared_mask", {"dof_count": 8, "scratch_stride": 7}),
            ("task_name", {"dof_count": 18}),
        ]:
            with self.subTest(mapping=mapping, dimensions=dimensions), self.assertRaises(ValueError):
                _sparse_contact_projection_source(mapping, **dimensions)

    def _project(self, device):
        """Preserve dense factor action across dimensions, supports and contact directions."""
        rng = np.random.default_rng(749)
        for n in (7, 18, 43, 64):
            for branched in (False, True):
                parents = [-1] + [(i - 1) // 2 if branched else i - 1 for i in range(1, n)]
                plan = TreePlan.build(parents, [1] * n).factor
                # Single endpoint for deep trees; union for branches. The final
                # coordinate/root exercises bit 63 without shifting by 64.
                endpoint = n - 1 if branched else min(n - 1, 19)
                indices = _endpoint_support(plan, endpoint, n - 2 if branched else -1)
                self.assertLessEqual(len(indices), 32)
                packed = rng.normal(size=(1, plan.nonzero_count)).astype(np.float32)
                matrix = np.zeros((n, n), dtype=np.float64)
                matrix[plan.entry_rows, plan.columns] = packed[0]
                jacobian = np.zeros((3, n), dtype=np.float32)
                jacobian[:, indices] = rng.normal(size=(3, len(indices)))

                def array(value, dtype=float):
                    return wp.array(value, dtype=dtype, device=device)

                mask = sum(1 << int(i) for i in indices)
                for directions in (1, 2, 3):
                    operands = jacobian.copy()
                    expected = matrix @ operands.T
                    expected[:, directions:] = 0
                    output = wp.zeros(n * 3, dtype=float, device=device)
                    kernel = projection_kernel(n, plan.nonzero_count)
                    with self.subTest(n=n, branched=branched, directions=directions, device=device):
                        wp.launch_tiled(
                            kernel,
                            dim=[1],
                            inputs=[
                                array(plan.row_offsets, int),
                                array(packed),
                                array(operands.ravel()),
                                wp.uint64(mask),
                                directions,
                            ],
                            outputs=[output],
                            block_dim=32,
                            device=device,
                        )
                        np.testing.assert_allclose(output.numpy().reshape(n, 3), expected, atol=3e-6, rtol=2e-5)

    def test_shared_cpu(self):
        """Check shared-operand projection with the dense CPU reference."""
        self._project("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_shared_cuda(self):
        """Check shared-operand projection on CUDA."""
        self._project("cuda:0")


if __name__ == "__main__":
    unittest.main()

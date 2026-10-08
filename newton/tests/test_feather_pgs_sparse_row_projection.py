# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare both generated operand mappings with dense factor action."""

import unittest
from functools import cache

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sparse_factor import SparseData, SparsePlan
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseMassMatrixPlan
from newton._src.solvers.feather_pgs.sparse_row_projection import _sparse_contact_projection_source


@cache
def projection_kernel(mapping, n, width, nnz):
    """Execute production projection source with independently supplied operands."""
    fragment = _sparse_contact_projection_source(
        mapping, dof_count=n, support_width=width, factor_stride=nnz, scratch_stride=n
    )
    if mapping == "shared_mask":
        body = f"""
        #if defined(__CUDA_ARCH__)
        const int lane=threadIdx.x&31, stride=32;
        __shared__ float jacobian[{3 * n}];
        #else
        const int lane=0, stride=1;
        float jacobian[{3 * n}];
        #endif
        for(int i=lane;i<{3 * n};i+=stride)jacobian[i]=input.data[i];
        #if defined(__CUDA_ARCH__)
        __syncwarp();
        #endif
        const auto first_bit=[](unsigned long long m){{int i=0;while((m&1ull)==0){{m>>=1;++i;}}return i;}};
        const int factor_base=0;
        for(int node=lane;node<{n};node+=stride){{
            if((mask&(1ull<<node))==0)continue;
            {fragment}
            for(int r=0;r<3;++r)output.data[node*3+r]=value.c[r];
        }}
        """
    else:
        body = f"""
        #if defined(__CUDA_ARCH__)
        const int lane=threadIdx.x&31, tpl=0, group=0;
        const int node=lane<length?p.support_nodes.data[lane]:-1;
        const float jn=lane<length?input.data[node]:0.0f;
        const float jt0=lane<length?input.data[{n}+node]:0.0f;
        const float jt1=lane<length?input.data[{2 * n}+node]:0.0f;
        {fragment}
        if(lane<length){{output.data[node*3]=zn;output.data[node*3+1]=zt0;output.data[node*3+2]=zt1;}}
        #endif
        """

    @wp.func_native(body)
    def native(
        p: SparsePlan,
        d: SparseData,
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        input: wp.array[float],
        mask: wp.uint64,
        length: int,
        row_count: int,
        output: wp.array[float],
    ): ...

    def kernel(
        p: SparsePlan,
        d: SparseData,
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        input: wp.array[float],
        mask: wp.uint64,
        length: int,
        row_count: int,
        output: wp.array[float],
    ):
        native(p, d, factor_row_offsets, inverse_factor, input, mask, length, row_count, output)

    kernel.__name__ = kernel.__qualname__ = f"row_projection_{mapping}_{n}_{width}_{nnz}"
    return wp.kernel(enable_backward=False, module="unique")(kernel)


class TestSparseRowProjection(unittest.TestCase):
    def test_invalid_mapping_dimensions(self):
        """Reject layouts that would truncate supports or access invalid storage."""
        for mapping, dimensions in [
            ("shared_mask", {"dof_count": 0, "scratch_stride": 4}),
            ("shared_mask", {"dof_count": 8, "scratch_stride": 7}),
            ("warp_catalogue", {"dof_count": 43, "support_width": 33, "factor_stride": 434}),
            ("warp_catalogue", {"dof_count": 7, "support_width": 8, "factor_stride": 28}),
            ("warp_catalogue", {"dof_count": 18, "support_width": 12, "factor_stride": 0}),
            ("task_name", {"dof_count": 18}),
        ]:
            with self.subTest(mapping=mapping, dimensions=dimensions), self.assertRaises(ValueError):
                _sparse_contact_projection_source(mapping, **dimensions)

    def _project(self, device, mapping):
        """Preserve dense factor action across dimensions, supports and contact directions."""
        rng = np.random.default_rng(749)
        for n in (7, 18, 43, 64):
            for branched in (False, True):
                parents = [-1] + [(i - 1) // 2 if branched else i - 1 for i in range(1, n)]
                plan = _SparseMassMatrixPlan.build(parents, [1] * n)
                # Single endpoint for deep trees; union for branches. The final
                # coordinate/root exercises bit 63 without shifting by 64.
                endpoint = n - 1 if branched else min(n - 1, 19)
                indices = plan.endpoint_support(endpoint, n - 2 if branched else -1)
                width = len(indices)
                self.assertLessEqual(width, 32)
                packed = rng.normal(size=(1, plan.nonzero_count)).astype(np.float32)
                matrix = np.zeros((n, n), dtype=np.float64)
                matrix[plan.entry_rows, plan.columns] = packed[0]
                jacobian = np.zeros((3, n), dtype=np.float32)
                jacobian[:, indices] = rng.normal(size=(3, width))

                def array(value, dtype=float):
                    return wp.array(value, dtype=dtype, device=device)

                p, d = SparsePlan(), SparseData()
                p.index = array(plan.lookup, int)
                p.support_nodes = array(indices[None], int)
                d.W = array(packed)
                mask = sum(1 << int(i) for i in indices)
                for directions in (1, 2, 3):
                    operands = jacobian.copy()
                    if mapping == "warp_catalogue":
                        # Production broadcasts all three operands even for a
                        # normal-only contact; publication selects live rows.
                        expected = matrix @ operands.T
                    else:
                        expected = matrix @ operands.T
                        expected[:, directions:] = 0
                    output = wp.zeros(n * 3, dtype=float, device=device)
                    kernel = projection_kernel(mapping, n, width, plan.nonzero_count)
                    with self.subTest(n=n, branched=branched, directions=directions, device=device):
                        wp.launch_tiled(
                            kernel,
                            dim=[1],
                            inputs=[
                                p,
                                d,
                                array(plan.row_offsets, int),
                                d.W,
                                array(operands.ravel()),
                                wp.uint64(mask),
                                width,
                                directions,
                            ],
                            outputs=[output],
                            block_dim=32,
                            device=device,
                        )
                        np.testing.assert_allclose(output.numpy().reshape(n, 3), expected, atol=3e-6, rtol=2e-5)

    def test_shared_cpu(self):
        """Check shared-operand projection with the dense CPU reference."""
        self._project("cpu", "shared_mask")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_shared_cuda(self):
        """Check shared-operand projection on CUDA."""
        self._project("cuda:0", "shared_mask")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_catalogue_cuda(self):
        """Check full-warp broadcasts on multiple topologies and widths."""
        self._project("cuda:0", "warp_catalogue")


if __name__ == "__main__":
    unittest.main()

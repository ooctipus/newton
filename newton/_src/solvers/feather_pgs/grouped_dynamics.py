# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The grouped contact response of ``grouped_dynamics``: one warp per articulation, one lane per row."""

from __future__ import annotations

import warp as wp


def get_grouped_hinv_jt_kernel(
    n_dofs: int,
    max_constraints: int,
    *,
    write_world: bool,
    write_group: bool,
    compute_diag: bool,
) -> wp.Kernel:
    """Response rows ``Y = H^-1 J^T`` with one warp per articulation and one lane per row.

    The Cholesky factor lives in shared memory; each lane solves its own row (forward then
    backward substitution) with the row's Jacobian and response in registers. Rows past the
    world's constraint count are not touched. Optionally the diagonal ``J_i . Y_i`` and the
    world-layout copies of ``J`` and ``Y`` are written, matching the tiled kernel's outputs.
    """
    N = int(n_dofs)
    M = int(max_constraints)
    world_store = (
        """
            const int dof_offset = articulation_world_dof_offset.data[art];
            const size_t wbase = (size_t)world * world_rows * world_dofs + (size_t)row * world_dofs + dof_offset;
            #pragma unroll
            for (int d = 0; d < N_; ++d) {
                J_world.data[wbase + d] = j[d];
                Y_world.data[wbase + d] = y[d];
            }"""
        if write_world
        else ""
    )
    group_store = (
        """
            #pragma unroll
            for (int d = 0; d < N_; ++d) Y_group.data[gbase + d] = y[d];"""
        if write_group
        else ""
    )
    diag_store = (
        """
            float acc = 0.0f;
            #pragma unroll
            for (int d = 0; d < N_; ++d) acc += j[d] * y[d];
            diag_group.data[(size_t)idx * M_ + row] = acc;"""
        if compute_diag
        else ""
    )
    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int N_ = {N};
    constexpr int M_ = {M};
    constexpr int WPB = 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int idx = block * warps_per_block + warp;
    __shared__ float Ls[WPB * N_ * N_];
    __shared__ float Ds[WPB * N_];
    float* L = Ls + warp * N_ * N_;
    float* Dinv = Ds + warp * N_;
    if (idx >= n_arts) return;
    const int art = group_to_art.data[idx];
    const int world = art_to_world.data[art];
    const int n_rows = world_constraint_count.data[world];
    if (n_rows <= 0) return;
    // Worlds the parallel sweep kernel owns (dense rows within its budget, no matrix-free rows) form the
    // response in-kernel; their world-layout J/Y are never read.
    if (skip_rows_le > 0 && n_rows <= skip_rows_le && mf_constraint_count.data[world] == 0) return;
    for (int e = lane; e < N_ * N_; e += 32) L[e] = L_group.data[(size_t)idx * N_ * N_ + e];
    __syncwarp();
    if (lane < N_) Dinv[lane] = 1.0f / L[lane * N_ + lane];
    __syncwarp();
    const int world_rows = Y_world.shape[1];
    const int world_dofs = Y_world.shape[2];
    // Match the tiled kernel: every row of a touched 32-row chunk is written (inactive rows
    // carry zero Jacobians and get zero responses).
    const int n_pad = ((n_rows + 31) / 32) * 32;
    for (int row = lane; row < n_pad && row < M_; row += 32) {{
        const size_t gbase = ((size_t)idx * M_ + row) * N_;
        float j[N_], y[N_];
        #pragma unroll
        for (int d = 0; d < N_; ++d) {{ j[d] = J_group.data[gbase + d]; y[d] = 0.0f; }}
        // Forward: L z = j, then backward: L^T y = z. Both loops fully unrolled at compile time so every
        // y[k] index is a constant (registers) and only the N(N-1)/2 real multiply-adds are emitted per pass;
        // the previous predicated form emitted N^2 per step (N^3 per row).
        #pragma unroll
        for (int i = 0; i < N_; ++i) {{
            float v = j[i];
            #pragma unroll
            for (int k = 0; k < i; ++k) v -= L[i * N_ + k] * y[k];
            y[i] = v * Dinv[i];
        }}
        #pragma unroll
        for (int i = N_ - 1; i >= 0; --i) {{
            float v = y[i];
            #pragma unroll
            for (int k = i + 1; k < N_; ++k) v -= L[k * N_ + i] * y[k];
            y[i] = v * Dinv[i];
        }}{group_store}{diag_store}{world_store}
    }}
#endif
"""

    @wp.func_native(snippet)
    def grouped_hinv_jt_native(
        block: int,
        warps_per_block: int,
        n_arts: int,
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        diag_group: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        skip_rows_le: int,
    ): ...

    def grouped_hinv_jt_template(
        warps_per_block: int,
        n_arts: int,
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        diag_group: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        skip_rows_le: int,
    ):
        block, _lane = wp.tid()
        grouped_hinv_jt_native(
            block,
            warps_per_block,
            n_arts,
            L_group,
            J_group,
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            world_constraint_count,
            Y_group,
            J_world,
            Y_world,
            diag_group,
            mf_constraint_count,
            skip_rows_le,
        )

    suffix = (
        ("_world" if write_world else "") + ("_nogroup" if not write_group else "") + ("_diag" if compute_diag else "")
    )
    name = f"grouped_hinv_jt_{N}_{M}{suffix}"
    grouped_hinv_jt_template.__name__ = name
    grouped_hinv_jt_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(grouped_hinv_jt_template)

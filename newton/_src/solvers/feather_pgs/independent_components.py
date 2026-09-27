# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental exact operator partition of held dense hand and current MF body."""

from functools import cache

import warp as wp


@wp.struct
class IndependentComponentData:
    world_count: int
    counts: wp.array[int]
    mf_counts: wp.array[int]
    primary_offset: wp.array[int]
    secondary_offset: wp.array[int]
    primary_group: wp.array[int]
    selector: wp.array[int]
    J: wp.array3d[float]
    Y: wp.array3d[float]
    L: wp.array3d[float]
    rhs: wp.array2d[float]
    diag: wp.array2d[float]
    mu: wp.array2d[float]
    row_type: wp.array2d[int]
    row_parent: wp.array2d[int]
    mf_meta: wp.array2d[int]
    mf_J_a: wp.array3d[float]
    mf_J_b: wp.array3d[float]
    mf_MiJt_a: wp.array3d[float]
    mf_MiJt_b: wp.array3d[float]
    mf_mu: wp.array2d[float]


@cache
def get_prepare_kernel(primary_size=23):
    """Qualify complete current rows before replacing physical Y with held-L Z.

    The signed selector is private dispatch metadata, never a canonical count:
    zero=no MF, positive=original mixed fallback, negative=independent components.
    A generous finite-product bound rejects extreme conditioning before any Y
    write; it never clips an impulse or changes the original fallback law.
    """
    if primary_size != 23:
        raise ValueError("The experimental component owner admits 23+6 only")
    snippet = """
#if defined(__CUDA_ARCH__)
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw = nt / 32;
    __shared__ int accepted;
#else
    if (logical_lane != 0) return;
    const int tid = 0, nt = 1, lane = 0, warp = 0, nw = 1;
    int accepted;
#endif
    const int M = data.J.shape[1];
    const int F = data.mf_meta.shape[1] / 4;
    for (int world = worker; world < data.world_count; world += workers) {
        const int m = data.counts.data[world];
        const int mf = data.mf_counts.data[world];
        if (tid == 0) data.selector.data[world] = mf == 0 ? 0 : wp::max(mf, 1);
        if (mf == 0) continue;
        if (m < 0 || m > M || mf < 1 || mf > F) continue;
        const int po = data.primary_offset.data[world];
        const int so = data.secondary_offset.data[world];
        const int group = data.primary_group.data[world];
        if (!((po == 0 && so == 23) || (po == 6 && so == 0))
            || group < 0 || group >= data.L.shape[0]) continue;
        if (tid == 0) accepted = 1;
#if defined(__CUDA_ARCH__)
        __syncthreads();
#endif
        int valid = 1;
        for (int k = tid; k < 23 * 23; k += nt) {
            const float x = data.L.data[group * 23 * 23 + k];
            valid &= wp::isfinite(x) && wp::abs(x) <= 1.0e15f;
        }
        for (int row = tid; row < m; row += nt) {
            const int r = world * M + row;
            const int type = data.row_type.data[r];
            const float diag = data.diag.data[r];
            const float mu = data.mu.data[r];
            valid &= (type == 0 || type == 2 || type == 3)
                && wp::isfinite(diag) && diag > 0.0f
                && wp::isfinite(mu) && mu >= 0.0f && wp::isfinite(data.rhs.data[r]);
            if (type == 2) {
                const int p = data.row_parent.data[r];
                if (p < 0 || p + 2 >= m || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = world * M + p;
                    valid &= data.row_type.data[base] == 0
                        && data.row_type.data[base + 1] == 2 && data.row_type.data[base + 2] == 2
                        && data.row_parent.data[base + 1] == p && data.row_parent.data[base + 2] == p;
                }
            }
            for (int d = 0; d < 29; ++d) {
                const float j = data.J.data[r * 29 + d], y = data.Y.data[r * 29 + d];
                valid &= wp::isfinite(j) && wp::isfinite(y) && wp::abs(y) <= 1.0e15f;
                if (d >= so && d < so + 6) valid &= j == 0.0f && y == 0.0f;
            }
        }
        for (int row = tid; row < mf; row += nt) {
            const int r = world * F + row;
            const int packed = data.mf_meta.data[r * 4];
            const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535));
            const int tp = data.mf_meta.data[r * 4 + 3], type = tp & 65535;
            union { int bits; float value; } diag, rhs;
            diag.bits = data.mf_meta.data[r * 4 + 1];
            rhs.bits = data.mf_meta.data[r * 4 + 2];
            const float mu = data.mf_mu.data[r];
            valid &= (a == -1 || a == so) && (b == -1 || b == so) && (a == so || b == so)
                && (type == 0 || type == 2 || type == 4)
                && wp::isfinite(diag.value) && diag.value > 0.0f && wp::isfinite(rhs.value)
                && wp::isfinite(mu) && mu >= 0.0f;
            if (type == 2) {
                const int p = tp >> 16;
                if (p < 0 || p + 2 >= mf || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = (world * F + p) * 4 + 3;
                    const int t1 = data.mf_meta.data[base + 4], t2 = data.mf_meta.data[base + 8];
                    valid &= (data.mf_meta.data[base] & 65535) == 0
                        && (t1 & 65535) == 2 && (t2 & 65535) == 2 && (t1 >> 16) == p && (t2 >> 16) == p;
                }
            }
            for (int d = 0; d < 6; ++d) {
                valid &= wp::isfinite(data.mf_J_a.data[r * 6 + d])
                    && wp::isfinite(data.mf_J_b.data[r * 6 + d])
                    && wp::isfinite(data.mf_MiJt_a.data[r * 6 + d])
                    && wp::isfinite(data.mf_MiJt_b.data[r * 6 + d]);
            }
        }
#if defined(__CUDA_ARCH__)
        if (!valid) atomicExch(&accepted, 0);
        __syncthreads();
#else
        accepted = valid;
#endif
        if (accepted) {
            for (int row = warp; row < m; row += nw) {
                const int base = (world * M + row) * 29 + po;
#if defined(__CUDA_ARCH__)
                const float y = lane < 23 ? data.Y.data[base + lane] : 0.0f;
                float z = 0.0f;
                for (int k = 0; k < 23; ++k) {
                    const float yk = __shfl_sync(0xffffffffu, y, k);
                    if (lane < 23 && lane <= k)
                        z += data.L.data[group * 23 * 23 + k * 23 + lane] * yk;
                }
                if (lane < 23) data.Y.data[base + lane] = z;
#else
                float y[23];
                for (int k = 0; k < 23; ++k) y[k] = data.Y.data[base + k];
                for (int d = 0; d < 23; ++d) {
                    float z = 0.0f;
                    for (int k = d; k < 23; ++k)
                        z += data.L.data[group * 23 * 23 + k * 23 + d] * y[k];
                    data.Y.data[base + d] = z;
                }
#endif
            }
            if (tid == 0) data.selector.data[world] = -1;
        }
#if defined(__CUDA_ARCH__)
        // Do not reuse the shared admission word while another warp consumes it.
        __syncthreads();
#endif
    }
"""

    @wp.func_native(snippet)
    def prepare_native(worker: int, logical_lane: int, workers: int, data: IndependentComponentData): ...

    @wp.kernel(enable_backward=False)
    def prepare(workers: int, data: IndependentComponentData):
        worker, lane = wp.tid()
        prepare_native(worker, lane, workers, data)

    return prepare


class IndependentComponents:
    """One private selector allocation; all numerical arrays retain their owners."""

    def __init__(self, solver):
        self.data = IndependentComponentData()
        self.data.world_count = solver.world_count
        self.data.selector = wp.zeros(solver.world_count, dtype=wp.int32, device=solver.model.device)
        self.workers = min(solver.world_count, 512)
        self.kernel = get_prepare_kernel()

    @property
    def selector(self):
        return self.data.selector

    def prepare(self, solver, rhs, mf_meta):
        data = self.data
        for target, source in (
            ("counts", "constraint_count"),
            ("mf_counts", "mf_constraint_count"),
            ("primary_offset", "_paired_factor_primary_offsets_by_world"),
            ("secondary_offset", "_paired_factor_secondary_offsets_by_world"),
            ("primary_group", "_paired_factor_primary_groups_by_world"),
            ("J", "J_world"),
            ("Y", "Y_world"),
            ("diag", "diag"),
            ("mu", "row_mu"),
            ("row_type", "row_type"),
            ("row_parent", "row_parent"),
            ("mf_J_a", "mf_J_a"),
            ("mf_J_b", "mf_J_b"),
            ("mf_MiJt_a", "mf_MiJt_a"),
            ("mf_MiJt_b", "mf_MiJt_b"),
            ("mf_mu", "mf_row_mu"),
        ):
            setattr(data, target, getattr(solver, source))
        data.rhs, data.mf_meta, data.L = rhs, mf_meta, solver.L_by_size[23]
        wp.launch_tiled(
            self.kernel, dim=[self.workers], inputs=[self.workers, data], block_dim=128, device=solver.model.device
        )

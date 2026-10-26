# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split a paired world's rows into independent articulation and free-body components.

A world that pairs one held articulation with one free rigid body solves its
dense rows (the articulation's) in factor coordinates and its matrix-free
rows (the body's) in physical coordinates. When no row couples the two, the
world is partitioned: the paired factor kernel sweeps the dense component and
the matrix-free kernel sweeps the body's component, each reading and writing
only its own coordinates, so the two launches run concurrently instead of the
whole world falling to the general mixed sweep. The qualification kernel
checks every current row of a world before replacing its physical response
rows with the held-factor form, and publishes a signed selector per world:
zero for no matrix-free rows, positive for the mixed fallback, negative for
independent components.
"""

from functools import cache

import warp as wp

from .device_preludes import dual_native, lane_group


@wp.struct
class PairedComponentData:
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


FREE_BODY_DOFS = 6
# Threads of one block, which qualifies and publishes one world at a time.
PAIRED_COMPONENT_THREADS = 128

_SOURCE = r"""
#if !defined(__CUDA_ARCH__)
    if (logical_lane != 0) return;
#endif
    const int M = data.J.shape[1];
    const int F = data.mf_meta.shape[1] / 4;
    for (int world = worker; world < data.world_count; world += workers) {
        const int m = data.counts.data[world];
        const int mf = data.mf_counts.data[world];
        if (LANE == 0) data.selector.data[world] = mf == 0 ? 0 : wp::max(mf, 1);
        if (mf == 0) continue;
        if (m < 0 || m > M || mf < 1 || mf > F) continue;
        const int po = data.primary_offset.data[world];
        const int so = data.secondary_offset.data[world];
        const int group = data.primary_group.data[world];
        if (!((po == 0 && so == PRIMARY) || (po == SECONDARY && so == 0))
            || group < 0 || group >= data.L.shape[0]) continue;
        int valid = 1;
        for (int k = LANE; k < PRIMARY * PRIMARY; k += STRIDE) {
            const float x = data.L.data[group * PRIMARY * PRIMARY + k];
            valid &= wp::isfinite(x) && wp::abs(x) <= 1.0e15f;
        }
        for (int row = LANE; row < m; row += STRIDE) {
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
            for (int d = 0; d < WORLD; ++d) {
                const float j = data.J.data[r * WORLD + d], y = data.Y.data[r * WORLD + d];
                valid &= wp::isfinite(j) && wp::isfinite(y) && wp::abs(y) <= 1.0e15f;
                if (d >= so && d < so + SECONDARY) valid &= j == 0.0f && y == 0.0f;
            }
        }
        for (int row = LANE; row < mf; row += STRIDE) {
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
                const int encoded_parent = static_cast<unsigned int>(tp) >> 16;
                const int p = encoded_parent == 65535 ? -1 : encoded_parent;
                if (p < 0 || p + 2 >= mf || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = (world * F + p) * 4 + 3;
                    const int t1 = data.mf_meta.data[base + 4], t2 = data.mf_meta.data[base + 8];
                    valid &= (data.mf_meta.data[base] & 65535) == 0
                        && (t1 & 65535) == 2 && (t2 & 65535) == 2
                        && static_cast<int>(static_cast<unsigned int>(t1) >> 16) == p
                        && static_cast<int>(static_cast<unsigned int>(t2) >> 16) == p;
                }
            }
            for (int d = 0; d < SECONDARY; ++d) {
                valid &= wp::isfinite(data.mf_J_a.data[r * SECONDARY + d])
                    && wp::isfinite(data.mf_J_b.data[r * SECONDARY + d])
                    && wp::isfinite(data.mf_MiJt_a.data[r * SECONDARY + d])
                    && wp::isfinite(data.mf_MiJt_b.data[r * SECONDARY + d]);
            }
        }
        if (ALL(valid != 0)) {
            for (int row = WARP; row < m; row += WARPS) {
                const int base = (world * M + row) * WORLD + po;
#if defined(__CUDA_ARCH__)
                const int lane = LANE & 31;
                const float y = lane < PRIMARY ? data.Y.data[base + lane] : 0.0f;
                float z = 0.0f;
                for (int k = 0; k < PRIMARY; ++k) {
                    const float yk = READ(y, k);
                    if (lane < PRIMARY && lane <= k)
                        z += data.L.data[group * PRIMARY * PRIMARY + k * PRIMARY + lane] * yk;
                }
                if (lane < PRIMARY) data.Y.data[base + lane] = z;
#else
                float y[PRIMARY];
                for (int k = 0; k < PRIMARY; ++k) y[k] = data.Y.data[base + k];
                for (int d = 0; d < PRIMARY; ++d) {
                    float z = 0.0f;
                    for (int k = d; k < PRIMARY; ++k)
                        z += data.L.data[group * PRIMARY * PRIMARY + k * PRIMARY + d] * y[k];
                    data.Y.data[base + d] = z;
                }
#endif
            }
            if (LANE == 0) data.selector.data[world] = -1;
        }
    }
"""


def get_source(primary_size: int, secondary_size: int = FREE_BODY_DOFS) -> str:
    """The qualification and held-form publication for one articulation/free-body pair."""
    if not 1 <= primary_size <= 32 or secondary_size != FREE_BODY_DOFS:
        raise ValueError("Paired components take an articulation of at most 32 coordinates and one free rigid body")
    source = _SOURCE
    for token, value in (
        ("PRIMARY", primary_size),
        ("SECONDARY", secondary_size),
        ("WORLD", primary_size + secondary_size),
    ):
        source = source.replace(token, str(value))
    return source


@cache
def get_prepare_kernel(primary_size: int, secondary_size: int = FREE_BODY_DOFS):
    """Qualify a world's current rows, then replace physical Y with the held-L form.

    The signed selector is private dispatch metadata, never a canonical count.
    A generous finite-product bound rejects extreme conditioning before any Y
    write; it never clips an impulse or changes the original fallback law.
    """
    snippet = dual_native(get_source(primary_size, secondary_size), preludes=(lane_group(PAIRED_COMPONENT_THREADS),))

    @wp.func_native(snippet)
    def prepare_native(worker: int, logical_lane: int, workers: int, data: PairedComponentData): ...

    def prepare(workers: int, data: PairedComponentData):
        worker, lane = wp.tid()
        prepare_native(worker, lane, workers, data)

    prepare.__name__ = prepare.__qualname__ = f"paired_components_prepare{primary_size}_{secondary_size}"
    return wp.kernel(enable_backward=False, module="unique")(prepare)


def supported(solver) -> bool:
    """Whether the solver's paired worlds can be split: factor coordinates, one free body, plain sweeps."""
    return bool(
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and solver._paired_factor_coordinates
        and solver._paired_response_secondary_size == FREE_BODY_DOFS
        and 1 <= (solver._paired_response_primary_size or 0) <= 32
        and solver.pgs_velocity_iterations == 0
        and not solver.pgs_debug
        and solver.pgs_schedule == "interleaved"
        and not solver.pgs_warmstart
        and not solver._mf_warmstart_enabled
        and not solver._regularization_enabled
        and not solver._preelim_active
        and not solver.contact_compliance
        and solver._mimic_count == 0
        and solver._connect_count == 0
    )


class PairedComponents:
    """One private selector allocation; every numerical array keeps its owner."""

    def __init__(self, solver):
        self.primary_size = int(solver._paired_response_primary_size)
        self.secondary_size = int(solver._paired_response_secondary_size)
        self.data = PairedComponentData()
        self.data.world_count = solver.world_count
        self.data.selector = wp.zeros(solver.world_count, dtype=wp.int32, device=solver.model.device)
        self.workers = min(solver.world_count, 512)
        self.kernel = get_prepare_kernel(self.primary_size, self.secondary_size)

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
        data.rhs, data.mf_meta, data.L = rhs, mf_meta, solver.L_by_size[self.primary_size]
        wp.launch_tiled(
            self.kernel,
            dim=[self.workers],
            inputs=[self.workers, data],
            block_dim=PAIRED_COMPONENT_THREADS,
            device=solver.model.device,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split a paired world's rows into independent articulation and free-body components.

A world that pairs one held articulation with one free rigid body solves its
dense rows (the articulation's) in the articulation's factor coordinates and
its matrix-free rows (the body's) in the body's. When no row couples the two, the
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
from types import SimpleNamespace

import warp as wp

from . import field_group, kernel_template
from .device_preludes import lane_group
from .field_group import Field
from .kernel_template import KernelTemplate, items
from .route_table import capable
from .row_kernels import PGS_CONSTRAINT_TYPE_SPEED
from .step_plan import first_failed

PAIRED_COMPONENTS = (
    Field("world_count", wp.int32, owner="components"),
    Field("counts", wp.int32, ("world",), attr="constraint_count", owner="placement"),
    Field("mf_counts", wp.int32, ("world",), attr="mf_constraint_count", owner="placement"),
    Field("first", wp.int32, ("world",), attr="dense_first", owner="placement"),
    Field("primary_offset", wp.int32, ("world",), owner="sweep", attr="paired_factor_primary_offsets_by_world"),
    Field("secondary_offset", wp.int32, ("world",), owner="sweep", attr="paired_factor_secondary_offsets_by_world"),
    Field("selector", wp.int32, ("world",), owner="components"),
    Field("Y", wp.float32, ("world", "row", "world_dof"), attr="packed_Z", owner="placement"),
    Field("primary_art", wp.int32, ("world",), owner="sweep", attr="paired_factor_primary_arts_by_world"),
    Field("dof_start", wp.int32, ("articulation",), owner="step"),
    Field("walk", wp.float32, ("dof", "record"), owner="step"),
    Field("rhs", wp.float32, ("world", "row"), owner="step"),
    Field("diag", wp.float32, ("world", "row"), owner="placement"),
    Field("mu", wp.float32, ("world", "row"), attr="row_mu", owner="placement"),
    Field("row_type", wp.int32, ("world", "row"), owner="placement"),
    Field("row_parent", wp.int32, ("world", "row"), owner="placement"),
    Field("mf_meta", wp.int32, ("world", "mf_meta"), owner="step"),
    Field("body_base", wp.int32, owner="placement"),
)
"""The component classifier's inputs: the row store's records and packed rows (the dense rows from row 0, whitened at
their world coordinates, :mod:`.packed_rows`; the free bodies' from ``body_base``, in the sides layout), the free
bodies' headers, the pair layout and its own selector."""

PairedComponentData = field_group.struct("PairedComponentData", PAIRED_COMPONENTS, __name__)


FREE_BODY_DOFS = 6
# Threads of one block, which qualifies and publishes one world at a time.
PAIRED_COMPONENT_THREADS = 128


def decline(sweep) -> str | None:
    """Why the sweep's paired worlds cannot be split into components, or ``None`` when they can: the laws
    components implement (``route_table.CAPABILITIES``), then their fit."""
    return capable("components", sweep.laws.set) or first_failed(
        (
            (sweep.routes.case == "K5", "Components need the paired factor-coordinate sweep (case K5)."),
            (
                sweep.paired_response_secondary_size == FREE_BODY_DOFS
                and 1 <= (sweep.paired_response_primary_size or 0) <= 32,
                "Components pair an articulation of at most 32 DOFs with one free body per world.",
            ),
        )
    )


class PairedComponents:
    """One private selector allocation; every numerical array keeps its owner."""

    def __init__(self, sweep):
        self.primary_size = int(sweep.paired_response_primary_size)
        self.secondary_size = int(sweep.paired_response_secondary_size)
        self.world_count = sweep.forest.world_count
        self.data = PairedComponentData()
        self.data.selector = wp.zeros(sweep.forest.world_count, dtype=wp.int32, device=sweep.model.device)
        # Workers stride over the worlds. One wave of resident blocks (2,256 on the PRO) finishes the classifier
        # sooner (165 -> 105 us a step on kuka) and lets the free-body sweep overlap the paired one: the step 4% faster,
        # the two sweeps' summed kernel time 2-5% longer (three interleaved pairs, TS 11-04). Kept until that is weighed.
        self.workers = min(sweep.forest.world_count, 512)
        self.kernel = paired_components_kernel(self.primary_size, self.secondary_size)

    @property
    def selector(self):
        return self.data.selector

    def prepare(self, sweep, rhs, mf_meta):
        data = field_group.bind(
            PairedComponentData,
            self.data,
            sweep=sweep,
            placement=sweep.placement,
            components=SimpleNamespace(world_count=self.world_count, selector=self.data.selector),
            dof_start=sweep.forest.articulation_dof_start,
            walk=sweep.factor.walk_data,
            rhs=rhs,
            mf_meta=mf_meta,
        )
        wp.launch_tiled(
            self.kernel,
            dim=[self.workers],
            inputs=[self.workers, data],
            block_dim=PAIRED_COMPONENT_THREADS,
            device=sweep.model.device,
        )


# ---------------------------------------------------------------------------------------------------------------------
# Paired factor rows (the paired route's dense rows in registers, sweep_routes_local)
# ---------------------------------------------------------------------------------------------------------------------
#
# A paired world holds one articulation (``P`` coordinates, its primary group) and one free body (``S = 6``, its
# secondary group), each a diagonal block of the leaf-first ``P H P^T = L L^T``. Its rows' factor rows
# ``z = L^-1 P j`` are its packed rows at their world coordinates (:mod:`.packed_rows`, ``A_ii = |z|^2 + cfm``).
#
# A paired world's qualification as independent components: every current row of the world is valid and touches
# only its own component (dense rows the articulation's coordinates, free-body rows the body's), its pair structure
# intact; a qualified world's selector turns negative (zero: no free-body rows, positive: the mixed fallback).
_PAIRED_COMPONENTS = r"""
#if !defined(__CUDA_ARCH__)
    if (lane != 0) return;
#endif
    const int M = data.Y.shape[1];
    const int F = data.mf_meta.shape[1] / 4;
    const int pitch = data.row_type.shape[1];
    for (int world = worker; world < data.world_count; world += workers) {
        const int m = data.counts.data[world];
        const int mf = data.mf_counts.data[world];
        if (LANE == 0) data.selector.data[world] = mf == 0 ? 0 : wp::max(mf, 1);
        if (mf == 0) continue;
        if (m < 0 || m > M || mf < 1 || mf > F) continue;
        const int po = data.primary_offset.data[world];
        const int so = data.secondary_offset.data[world];
        const int art = data.primary_art.data[world];
        if (!((po == 0 && so == $PRIMARY) || (po == $SECONDARY && so == 0)) || art < 0
            || art >= data.dof_start.shape[0]) continue;
        // The articulation's factor is its walk: every record of its coordinates finite and bounded.
        int valid = 1;
        const int records = data.walk.shape[1];
        const float* walk = data.walk.data + (size_t)data.dof_start.data[art] * records;
        for (int k = LANE; k < $PRIMARY * records; k += STRIDE) {
            const float x = walk[k];
            valid &= wp::isfinite(x) && wp::abs(x) <= 1.0e15f;
        }
        for (int row = LANE; row < m; row += STRIDE) {
            const int r = data.first.data[world] * M + row;
            const int at = data.first.data[world] * pitch + row;
            const int type = data.row_type.data[at];
            const float diag = data.diag.data[at];
            const float mu = data.mu.data[at];
            valid &= (type == 0 || type == 2 || type == 3)
                && wp::isfinite(diag) && diag > 0.0f
                && wp::isfinite(mu) && mu >= 0.0f && wp::isfinite(data.rhs.data[at]);
            if (type == 2) {
                const int p = data.row_parent.data[at];
                if (p < 0 || p + 2 >= m || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = data.first.data[world] * pitch + p;
                    valid &= data.row_type.data[base] == 0
                        && data.row_type.data[base + 1] == 2 && data.row_type.data[base + 2] == 2
                        && data.row_parent.data[base + 1] == p && data.row_parent.data[base + 2] == p;
                }
            }
            for (int d = 0; d < $WORLD; ++d) {
                const float y = data.Y.data[r * $WORLD + d];
                valid &= wp::isfinite(y) && wp::abs(y) <= 1.0e15f;
                if (d >= so && d < so + $SECONDARY) valid &= y == 0.0f;
            }
        }
        for (int row = LANE; row < mf; row += STRIDE) {
            const int r = data.first.data[world] * F + row;
            const int packed = data.mf_meta.data[r * 4];
            const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535));
            const int tp = data.mf_meta.data[r * 4 + 3], type = tp & 65535;
            union { int bits; float value; } diag, rhs;
            diag.bits = data.mf_meta.data[r * 4 + 1];
            rhs.bits = data.mf_meta.data[r * 4 + 2];
            const float mu = data.mu.data[data.first.data[world] * pitch + data.body_base + row];
            valid &= (a == -1 || a == so) && (b == -1 || b == so) && (a == so || b == so)
                && (type == 0 || type == 2 || type == $SPEED)
                && wp::isfinite(diag.value) && diag.value > 0.0f && wp::isfinite(rhs.value)
                && wp::isfinite(mu) && mu >= 0.0f;
            if (type == 2) {
                const int encoded_parent = static_cast<unsigned int>(tp) >> 16;
                const int p = encoded_parent == 65535 ? -1 : encoded_parent;
                if (p < 0 || p + 2 >= mf || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = (data.first.data[world] * F + p) * 4 + 3;
                    const int t1 = data.mf_meta.data[base + 4], t2 = data.mf_meta.data[base + 8];
                    valid &= (data.mf_meta.data[base] & 65535) == 0
                        && (t1 & 65535) == 2 && (t2 & 65535) == 2
                        && static_cast<int>(static_cast<unsigned int>(t1) >> 16) == p
                        && static_cast<int>(static_cast<unsigned int>(t2) >> 16) == p;
                }
            }
            // The row's packed entries, the two sides' six each where the line holds them (free_body_rows).
            const float* z = data.Y.data + ((size_t)data.first.data[world] * M + data.body_base + row) * $WORLD;
            for (int d = 0; d < 2 * $SECONDARY && d < $WORLD; ++d) valid &= wp::isfinite(z[d]);
        }
        if (ALL(valid != 0) && LANE == 0) data.selector.data[world] = -1;
    }
"""


@cache
def paired_components_kernel(primary_size: int, secondary_size: int = FREE_BODY_DOFS) -> wp.Kernel:
    """Qualify each paired world's rows as independent components. A block of
    :data:`.paired_components.PAIRED_COMPONENT_THREADS` threads per worker strides over the worlds; the signed selector
    is the dispatch's private metadata."""
    if not 1 <= primary_size <= 32 or secondary_size != FREE_BODY_DOFS:
        raise ValueError("Paired components take an articulation of at most 32 coordinates and one free rigid body")
    P = int(primary_size)
    source = kernel_template.substitute(
        _PAIRED_COMPONENTS,
        PRIMARY=P,
        SECONDARY=secondary_size,
        WORLD=P + secondary_size,
        SPEED=int(PGS_CONSTRAINT_TYPE_SPEED),
    )
    return KernelTemplate(
        f"paired_components_{P}_{secondary_size}",
        (("workers", int), ("data", PairedComponentData)),
        source,
        # A lane group per worker: its worker index and the thread's lane in it.
        items("worker", lane=True),
        preludes=(lane_group(PAIRED_COMPONENT_THREADS),),
        dual=True,
        native_name="paired_components_native",
    ).kernel()

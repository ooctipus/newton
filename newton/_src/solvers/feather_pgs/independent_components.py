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
    from .component_qualification import get_source  # noqa: PLC0415

    snippet = get_source()

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

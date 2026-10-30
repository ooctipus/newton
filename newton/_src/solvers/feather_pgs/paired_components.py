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

from types import SimpleNamespace

import warp as wp

from . import field_group
from .field_group import Field
from .route_table import capable
from .step_plan import first_failed

PAIRED_COMPONENTS = (
    Field("world_count", wp.int32, owner="components"),
    Field("counts", wp.int32, ("world",), attr="constraint_count"),
    Field("mf_counts", wp.int32, ("world",), attr="mf_constraint_count"),
    Field("primary_offset", wp.int32, ("world",), attr="_paired_factor_primary_offsets_by_world"),
    Field("secondary_offset", wp.int32, ("world",), attr="_paired_factor_secondary_offsets_by_world"),
    Field("primary_group", wp.int32, ("world",), attr="_paired_factor_primary_groups_by_world"),
    Field("selector", wp.int32, ("world",), owner="components"),
    Field("J", wp.float32, ("world", "row", "world_dof"), attr="_packed_J"),
    Field("Y", wp.float32, ("world", "row", "world_dof"), attr="_packed_Z"),
    Field("L", wp.float32, ("art", "size", "size"), owner="step"),
    Field("rhs", wp.float32, ("world", "row"), owner="step"),
    Field("diag", wp.float32, ("world", "row")),
    Field("mu", wp.float32, ("world", "row"), attr="row_mu"),
    Field("row_type", wp.int32, ("world", "row")),
    Field("row_parent", wp.int32, ("world", "row")),
    Field("mf_meta", wp.int32, ("world", "mf_meta"), owner="step"),
    Field("mf_Z_a", wp.float32, ("world", "mf_row", 6)),
    Field("mf_Z_b", wp.float32, ("world", "mf_row", 6)),
    Field("mf_mu", wp.float32, ("world", "mf_row"), attr="mf_row_mu"),
)
"""The component classifier's inputs: the dense rows (packed at their world coordinates: their Jacobian and whitened
rows, :mod:`.packed_rows`) and the matrix-free rows, the pair layout and its own selector."""

PairedComponentData = field_group.struct("PairedComponentData", PAIRED_COMPONENTS, __name__)


FREE_BODY_DOFS = 6
# Threads of one block, which qualifies and publishes one world at a time.
PAIRED_COMPONENT_THREADS = 128


def decline(solver) -> str | None:
    """Why the solver's paired worlds cannot be split into components, or ``None`` when they can: the laws
    components implement (``route_table.CAPABILITIES``), then their fit."""
    return capable("components", solver.laws.set) or first_failed(
        (
            (solver.routes.case == "K5", "Components need the paired factor-coordinate sweep (case K5)."),
            (
                solver._paired_response_secondary_size == FREE_BODY_DOFS
                and 1 <= (solver._paired_response_primary_size or 0) <= 32,
                "Components pair an articulation of at most 32 DOFs with one free body per world.",
            ),
        )
    )


class PairedComponents:
    """One private selector allocation; every numerical array keeps its owner."""

    def __init__(self, solver):
        self.primary_size = int(solver._paired_response_primary_size)
        self.secondary_size = int(solver._paired_response_secondary_size)
        self.world_count = solver.forest.world_count
        self.data = PairedComponentData()
        self.data.selector = wp.zeros(solver.forest.world_count, dtype=wp.int32, device=solver.model.device)
        self.workers = min(solver.forest.world_count, 512)
        from .response_builds_rows import paired_components_kernel  # noqa: PLC0415 - the build reads this module

        self.kernel = paired_components_kernel(self.primary_size, self.secondary_size)

    @property
    def selector(self):
        return self.data.selector

    def prepare(self, solver, rhs, mf_meta):
        data = field_group.bind(
            PairedComponentData,
            self.data,
            solver=solver,
            components=SimpleNamespace(world_count=self.world_count, selector=self.data.selector),
            L=solver.factor.L_by_size[self.primary_size],
            rhs=rhs,
            mf_meta=mf_meta,
        )
        wp.launch_tiled(
            self.kernel,
            dim=[self.workers],
            inputs=[self.workers, data],
            block_dim=PAIRED_COMPONENT_THREADS,
            device=solver.model.device,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compact active dispatch for the paired factor-coordinate response law."""

from functools import cache

import warp as wp


@cache
def get_response_kernel(device_arch, warps_per_block=4, primary_dofs=23, secondary_dofs=6, max_constraints=192):
    """Return the paired response kernel launched over a compact active-world list.

    Arguments are the original paired-response arguments followed by the current
    compact active-world list, its count, and the static world-to-primary group
    map. The light owner must validate the complete list before dispatch.
    """
    # The solver module imports this owner; resolve its factory after construction.
    from .solver_feather_pgs import _get_paired_hinv_jt_kernel  # noqa: PLC0415

    return _get_paired_hinv_jt_kernel(
        primary_dofs,
        secondary_dofs,
        max_constraints,
        primary_dofs + secondary_dofs,
        device_arch,
        warps_per_block,
        factor_coordinates=True,
        active_list=True,
    )


class ActiveKukaResponse:
    """Bind canonical buffers to the light owner's already-validated active list."""

    def __init__(self, solver, active_worlds, active_count, *, warps_per_block=4):
        primary, secondary = solver._paired_response_primary_size, solver._paired_response_secondary_size
        if (
            primary is None
            or secondary is None
            or not solver._paired_factor_coordinates
            or solver.max_world_dofs != primary + secondary
            or active_worlds.shape != (solver.world_count,)
            or active_count.shape != (1,)
        ):
            raise ValueError("Active response requires a paired factor-coordinate layout owning every world DOF")
        self.solver = solver
        self.primary, self.secondary = primary, secondary
        self.active_worlds = active_worlds
        self.active_count = active_count
        self.block_dim = 32 * warps_per_block
        arch = str(solver.model.device.arch) if solver.model.device.is_cuda else "cpu"
        self.kernel = get_response_kernel(arch, warps_per_block, primary, secondary, solver.dense_max_constraints)

    def inputs(self):
        """Read current held factors and canonical owners without allocating or copying."""
        solver, primary, secondary = self.solver, self.primary, self.secondary
        return [
            solver.Hinv_by_size[primary],
            solver.Linv_by_size[primary],
            solver.J_by_size[primary],
            solver.group_to_art[primary],
            solver.Hinv_by_size[secondary],
            solver.Linv_by_size[secondary],
            solver.J_by_size[secondary],
            solver.group_to_art[secondary],
            solver._paired_response_secondary_groups,
            solver.art_to_world,
            solver.articulation_world_dof_offset,
            solver.constraint_count,
            solver.mf_constraint_count,
            solver.row_cfm,
            solver.J_world,
            solver.Y_world,
            solver.diag,
            self.active_worlds,
            self.active_count,
            solver._paired_factor_primary_groups_by_world,
        ]

    def launch(self):
        """Launch before original bias/restitution and mixed-world qualification."""
        wp.launch_tiled(
            self.kernel,
            dim=[self.solver.world_count],
            inputs=self.inputs(),
            block_dim=self.block_dim,
            device=self.solver.model.device,
        )

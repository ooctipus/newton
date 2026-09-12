# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental packed geometry and contact-local sequential GS ownership."""

import warp as wp

from . import compact_contact as compact
from . import contact_block_data as packed
from .contact_block_residual import get_kernel
from .fused_contact_solve import BiasData, mark_fallback, produce_fallback
from .private_contact_islands import KEYS, PrivateContactIslands


class ContactBlockIslands(PrivateContactIslands):
    """Retain scalar islands and fallback while replacing the coupled boundary."""

    def __init__(self, solver):
        super().__init__(solver)
        device = solver.model.device
        self.block = packed.allocate_block_data(solver._max_contacts_alloc, device)
        self.partition_kernel = packed.get_partition_kernel(str(device.arch))
        self.residual_kernel = get_kernel(str(device.arch))

    def prepare(self, solver, state_in, state_aug, contacts, dt):
        """Complete current admission before any raw tag confers private ownership."""
        data = compact.bind_contact_data(solver, state_in, state_aug, contacts)
        self.data = data
        bias = BiasData()
        bias.dt = dt
        bias.contact_speculative_scale = solver.contact_speculative_scale
        bias.joint_limit_speculative_scale = 1.0
        bias.restitution_threshold = solver._effective_restitution_velocity_threshold
        bias.incident = solver.v_hat
        self.bias = bias
        device = solver.model.device
        wp.launch(
            packed.route_contacts,
            dim=self.block.raw_tag.shape[0],
            inputs=[data, self.row_contact, self.block],
            device=device,
        )
        wp.launch_tiled(
            self.partition_kernel,
            dim=[solver.world_count],
            inputs=[data, solver.slot_counter, solver.dense_phase_bounds, self.routing, self.block],
            block_dim=32,
            device=device,
        )
        wp.launch(
            compact.clear_contact_response,
            dim=(solver.world_count, 256),
            inputs=[self.fallback_counts, self.fallback_bounds, data.dense_group, data.J, data.Y],
            block_dim=256,
            device=device,
        )
        wp.launch(produce_fallback, dim=contacts.rigid_contact_max, inputs=[data, self.owner], device=device)
        wp.launch(mark_fallback, dim=contacts.rigid_contact_max, inputs=[data, self.owner], device=device)

    def launch_solve(self, solver, dense_rhs, iterations, omega, friction_start_iteration, iteration_offset):
        """Order packed preparation after predictor joins and before its consumer."""
        solve = self._bind_solve(solver, dense_rhs, iterations, omega, friction_start_iteration, iteration_offset)
        device = solver.model.device
        wp.launch(
            packed.produce_contacts,
            dim=self.block.raw_tag.shape[0],
            inputs=[self.data, solve, self.bias, self.routing, self.block],
            device=device,
        )
        wp.launch(
            self.scalar_kernel,
            dim=(solver.world_count, KEYS),
            inputs=[self.data, solve, self.bias, self.routing],
            block_dim=32,
            device=device,
        )
        wp.launch_tiled(
            self.residual_kernel,
            dim=[solver.world_count],
            inputs=[self.data, solve, self.bias, self.routing, self.block],
            block_dim=32,
            device=device,
        )
        wp.launch_tiled(
            self.fallback_kernel,
            dim=[solver.world_count],
            inputs=[solve, self.owner],
            block_dim=32,
            device=device,
        )

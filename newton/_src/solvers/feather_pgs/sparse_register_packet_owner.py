# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind current key-only row production without altering older owner factories."""

import warp as wp


def build_rows(self, state_in, state_aug, contacts, dt):
    """Retain original allocation/metadata while replacing all dense J production."""
    from . import kernels as k  # noqa: PLC0415

    s, model, device = self.solver, self.solver.model, self.solver.model.device
    c = s.dense_max_constraints
    if self.packet_rows or self.register_packets:
        from .sparse_packet_rows import bind_current  # noqa: PLC0415

        bind_current(self, state_in, state_aug, contacts, dt)
    self._register_packet_contacts = None
    s.dense_contact_world_flag.zero_()
    if s._row_watermark:
        s._row_dropped_dense.zero_()
        s._row_dropped_mf.zero_()
        s._row_dropped_propagation.zero_()
    tiled_prefix = self.packet_rows or self.register_packets or self.parallel_limit_prefix
    prefix_launch = wp.launch_tiled if tiled_prefix else wp.launch
    prefix_launch(
        self.kernels.prefix,
        dim=[s.world_count] if tiled_prefix else s.world_count,
        inputs=[
            self.plan,
            self.data,
            s._joint_limit_q_index,
            model.joint_limit_lower,
            model.joint_limit_upper,
            state_in.joint_q,
            s.v_hat,
            int(s.enable_joint_limits),
            s.joint_limit_activation_gap,
            s.pgs_beta,
            s.pgs_cfm,
            s.slot_counter,
            s.row_type,
            s.row_parent,
            s.row_mu,
            s.row_beta,
            s.row_cfm,
            s.phi,
            s.target_velocity,
            s.diag,
            s.dense_phase_bounds,
        ],
        block_dim=32 if tiled_prefix else 256,
        device=device,
    )
    if contacts is not None and contacts.rigid_contact_max > 0:
        threads = min(contacts.rigid_contact_max, 65536)
        is_free = s.is_free_rigid if s.is_free_rigid is not None else s._dummy_is_free_rigid
        dummy = s._dummy_mf_slot_counter
        drops = (
            [s._row_dropped_dense, s._row_dropped_mf, s._row_dropped_propagation]
            if s._row_watermark
            else [dummy, dummy, dummy]
        )
        wp.launch(
            k.allocate_world_contact_slots,
            dim=threads,
            inputs=[
                contacts.rigid_contact_count,
                threads,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                state_in.body_q,
                model.shape_transform,
                model.shape_body,
                s.body_to_articulation,
                s.art_to_world,
                s.articulation_response_dof_count,
                model.body_flags,
                s.body_has_response_dofs,
                is_free,
                0,
                0,
                int(s.propagation_same_articulation_rows),
                0,
                s.contact_gap_gate,
                s.same_articulation_contact_gap_gate,
                s.articulation_pair_contact_gap_gate,
                c,
                s.mf_max_constraints,
                s.propagation_max_constraints,
                int(s.enable_contact_friction),
                s.contact_friction_gap_threshold,
                s.contact_friction_anchor_limit,
                int(s.contact_friction_articulation_pairs_only),
                int(s._row_watermark),
                s._resolved_simple_worlds,
            ],
            outputs=[
                s.contact_world,
                s.contact_slot,
                s.contact_art_a,
                s.contact_art_b,
                s.slot_counter,
                s.contact_path,
                dummy,
                dummy,
                s.dense_contact_world_flag,
                s.contact_slots_needed,
                *drops,
                s._constraint_capacity_status,
            ],
            device=device,
        )
        wp.launch(
            k.prepare_world_contact_rows,
            dim=threads,
            inputs=[
                contacts.rigid_contact_count,
                threads,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                s.contact_world,
                s.contact_slot,
                s.contact_art_a,
                s.contact_art_b,
                s.contact_path,
                s.contact_slots_needed,
                model.shape_body,
                state_in.body_q,
                state_aug.body_v_s,
                s._prescribed_articulation,
                s.articulation_origin,
                s.shape_material_mu,
                s.shape_material_restitution,
                int(s.enable_contact_friction),
                s.contact_friction_gap_threshold,
                int(s.contact_friction_shared_anchor),
                s.contact_friction_anchor_limit,
                int(s.contact_friction_articulation_pairs_only),
                is_free,
                s.contact_friction_scale,
                int(s.contact_shared_anchor),
                s.pgs_beta,
                s.pgs_cfm,
            ],
            outputs=[
                s.row_type,
                s.row_parent,
                s.row_mu,
                s.row_beta,
                s.row_cfm,
                s.phi,
                s.target_velocity,
                s.row_restitution,
            ],
            device=device,
        )
        if self.packet_rows or self.register_packets:
            wp.launch(
                self.kernels.contacts,
                dim=threads,
                inputs=[
                    contacts.rigid_contact_count,
                    threads,
                    s.contact_path,
                    s.contact_slot,
                    s.contact_world,
                    s.contact_slots_needed,
                    self.data.support,
                ],
                device=device,
            )
        if not self.packet_rows:
            workers = min(contacts.rigid_contact_max, 16384)
            contact_inputs = [
                workers,
                self.plan,
                self.data,
                contacts.rigid_contact_count,
                s.contact_path,
                s.contact_slot,
                s.contact_world,
                s.contact_art_a,
                s.contact_art_b,
                s.contact_slots_needed,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                model.shape_body,
                state_in.body_q,
                state_aug.joint_S_s,
                s.articulation_origin,
                s.art_group_idx,
                s.v_hat,
                int(s.contact_shared_anchor),
                int(s.contact_friction_shared_anchor),
                s.row_cfm,
                s.diag,
            ]
            if self.register_packets:
                # Current pointers remain capture-owned; route decisions are
                # produced by the small solve before this original producer.
                self._register_packet_contacts = (workers, contact_inputs)
            else:
                wp.launch_tiled(
                    self.kernels.contacts,
                    dim=[workers],
                    inputs=contact_inputs,
                    block_dim=32,
                    device=device,
                )
    wp.launch(
        k.finalize_constraint_counts_with_status,
        dim=s.world_count,
        inputs=[s.slot_counter, c, 0],
        outputs=[s.constraint_count, s._constraint_capacity_status],
        device=device,
    )

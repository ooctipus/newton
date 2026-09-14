# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current-layout dispatch for the bounded force-to-velocity G1 owner."""

import numpy as np
import warp as wp

from .grouped_dynamics import warp_sync
from .kernels import BodyFlags, jcalc_tau
from .small_step_force import ForceInput, build_force_plan, make_force_input
from .small_step_rows import get_solve_kernel as get_small_solve_kernel
from .sparse_factor import SparseData, SparsePlan, get_predictor_kernel
from .sparse_factor_rows import get_contact_kernel, get_parallel_limit_kernel, get_solve_kernel
from .sparse_packet_rows import PacketInput, bind_current, get_prefix_kernel, packet_contacts


@wp.kernel(enable_backward=False)
def classify(
    counts: wp.array[int],
    row_type: wp.array2d[int],
    selected: wp.array[int],
    fallback_counts: wp.array[int],
):
    """Classify current reservations, never final impulse values or old membership."""
    world = wp.tid()
    count = counts[world]
    small = count >= 0 and count <= 32
    normals = int(0)
    if small:
        for row in range(count):
            if row_type[world, row] == 0:
                normals += 1
        small = normals <= 8
    selected[world] = int(small)
    fallback_counts[world] = 0 if small else count


@wp.kernel(enable_backward=False)
def clear_fallback_force(p: SparsePlan, selected: wp.array[int], body_ft: wp.array[wp.spatial_vector]):
    """Clear original recurrence input only for the fallback body's owner."""
    group, local = wp.tid()
    art = p.group_to_art[group]
    world = p.art_to_world[art]
    if selected[world] == 0:
        body_ft[p.joint_child[p.art_joint_start[art] + local]] = wp.spatial_vector()


@wp.kernel(enable_backward=False)
def predict_fallback_velocity(
    p: SparsePlan,
    selected: wp.array[int],
    qd: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    dt: float,
    qdd: wp.array[float],
    vhat: wp.array[float],
):
    """Retain the original predictor followed by live-mask free-root transport."""
    group, local = wp.tid()
    art = p.group_to_art[group]
    world = p.art_to_world[art]
    if selected[world] != 0:
        return
    start = p.art_dof_start[art]
    dof = start + local
    if kinematic_dof_mask[dof] != 0:
        qdd[dof] = 0.0
    value = qd[dof] + qdd[dof] * dt
    if local < 3 and kinematic_joint_mask[p.art_joint_start[art]] == 0:
        v = wp.vec3(qd[start], qd[start + 1], qd[start + 2])
        w = wp.vec3(qd[start + 3], qd[start + 4], qd[start + 5])
        value = value + wp.cross(w, v)[local] * dt
    vhat[dof] = value


@wp.kernel(enable_backward=False)
def prepare_fallback_velocity(p: SparsePlan, selected: wp.array[int], vhat: wp.array[float], vout: wp.array[float]):
    """Initialize only the old solver's fallback output from its own predictor."""
    group, local = wp.tid()
    art = p.group_to_art[group]
    if selected[p.art_to_world[art]] == 0:
        dof = p.art_dof_start[art] + local
        vout[dof] = vhat[dof]


@wp.kernel(enable_backward=False)
def grouped_tau_fallback(
    selected: wp.array[int],
    art_to_world: wp.array[int],
    lanes_per_articulation: int,
    articulation_count: int,
    max_levels: int,
    art_level_offsets: wp.array2d[int],
    level_joint_local: wp.array[int],
    children_offsets: wp.array[int],
    children: wp.array[int],
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    add_existing_tau: int,
    body_ft_s: wp.array[wp.spatial_vector],
    body_fs_scratch: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    """Run the original backward recurrence with all-lane synchronization intact."""
    block, lane = wp.tid()
    group = lane // lanes_per_articulation
    local = lane - group * lanes_per_articulation
    art = block * (32 // lanes_per_articulation) + group
    valid = art < articulation_count
    if valid:
        valid = selected[art_to_world[art]] == 0
    start = int(0)
    origin = wp.vec3()
    if valid:
        start = articulation_start[art]
        origin = articulation_origin[art]
    for step in range(max_levels):
        level = max_levels - 1 - step
        if valid:
            lo = art_level_offsets[art, level]
            hi = art_level_offsets[art, level + 1]
            k = lo + local
            while k < hi:
                i = start + level_joint_local[k]
                child = joint_child[i]
                f_t_s = body_ft_s[child]
                c_lo = children_offsets[i]
                c_hi = children_offsets[i + 1]
                for c in range(c_lo, c_hi):
                    f_t_s = f_t_s + body_fs_scratch[joint_child[children[c]]]
                body_ft_s[child] = f_t_s
                f_ext_com = wp.spatial_vector()
                if (body_flags[child] & BodyFlags.KINEMATIC) == 0:
                    f_ext_com = body_f_ext[child]
                f_ext_f = wp.spatial_bottom(f_ext_com)
                f_ext_t = wp.spatial_top(f_ext_com)
                com_world = wp.transform_point(body_q[child], body_com[child])
                com_rel = com_world - origin
                tau_origin = f_ext_f + wp.cross(com_rel, f_ext_t)
                f_ext_origin = wp.spatial_vector(f_ext_t, tau_origin)
                f_s = body_fb_s[child] + f_t_s - f_ext_origin
                jcalc_tau(
                    joint_type[i],
                    joint_S_s,
                    joint_f,
                    joint_q,
                    joint_qd,
                    joint_spring_stiffness,
                    joint_spring_ref,
                    joint_damping,
                    joint_q_start[i],
                    joint_qd_start[i],
                    joint_dof_dim[i, 0],
                    joint_dof_dim[i, 1],
                    f_s,
                    add_existing_tau,
                    tau,
                )
                body_fs_scratch[child] = f_s
                k += lanes_per_articulation
        warp_sync()


class SmallStep:
    """Own current row admission and disjoint complete/fallback work."""

    def __init__(self, owner):
        self.owner, self.solver = owner, owner.solver
        s, device = self.solver, self.solver.model.device
        error = self._configuration_error()
        if error is not None:
            raise ValueError(error)
        self.force_plan = build_force_plan(s.model)
        self._joint_address_signature = {
            name: getattr(s.model, name).numpy().copy() for name in ("joint_q_start", "joint_qd_start")
        }
        self.force_input = ForceInput()
        self.packet_input = PacketInput()
        self.selected = owner.data.selected
        self.packets = wp.empty((s.world_count, 100), dtype=int, device=device)
        self.fallback_counts = wp.zeros(s.world_count, dtype=int, device=device)
        self.dummy_counter = wp.empty(s.world_count, dtype=int, device=device)
        self.layout_data = SparseData()
        for name in ("W", "valid", "status", "selected", "Z", "incident"):
            setattr(self.layout_data, name, getattr(owner.data, name))
        self.layout_data.support = self.packets
        self.layout_prefix = get_prefix_kernel()
        self.packet_contacts = packet_contacts
        self.solve_kernel = get_small_solve_kernel()
        self.tau_kernel = grouped_tau_fallback
        owner.kernels.prefix = get_parallel_limit_kernel(skip_selected=True)
        owner.kernels.predictor = get_predictor_kernel(skip_selected=True)
        owner.kernels.contacts = get_contact_kernel(skip_selected=True)
        owner.kernels.solve = get_solve_kernel(100, metric_tangents=True, skip_selected=True)

    def _configuration_error(self):
        """Require all replaced force and response producers to use masked routes."""
        owner, s = self.owner, self.solver
        if (
            s.dense_max_constraints != 100
            or not owner.metric_tangents
            or not owner.parallel_limit_prefix
            or owner.packet_rows
            or owner.block_contacts
            or not s._async_augmented_drives
            or not s._parallel_augmented_drive_topology
            or s.drive_mode != "augmented"
        ):
            return "Small-step requires the current c100 metric/parallel sparse owner with async augmented drives"
        if (
            getattr(s, "_grouped_topology", None) is None
            or getattr(s, "_direct_branch_tau_kernel", None) is not None
            or getattr(s, "_fused_k1", False)
            or getattr(s, "_grouped_tau_mass", False)
            or getattr(s, "_grouped_mass", False)
        ):
            return "Small-step requires the original grouped current-force producer without fused or direct overrides"
        return None

    def validate(self):
        """Reject changed dispatch before a step reads any retired intermediate."""
        error = self._configuration_error()
        if error is not None:
            raise RuntimeError(error + "; reconstruct the solver and recapture")

    def validate_notification(self):
        """Keep static force addresses fixed after the outer topology check."""
        self.validate()
        for name, previous in self._joint_address_signature.items():
            if not np.array_equal(getattr(self.solver.model, name).numpy(), previous):
                raise RuntimeError(
                    "Small-step joint coordinate ownership changed; reconstruct the solver and recapture"
                )

    def begin_step(self, state_in, state_aug, control, contacts, stage3_qd, dt, collide_done_event):
        """Build original identities once, then classify the complete current layout."""
        if collide_done_event is not None:
            wp.get_stream(self.solver.model.device).wait_event(collide_done_event)
        self.force_input = make_force_input(
            self.solver, state_in, state_aug, control, stage3_qd, dt, plan=self.force_plan
        )
        bind_current(self, state_in, state_aug, contacts, dt)
        self.owner.build_rows(state_in, state_aug, contacts, dt, layout_only=True)
        wp.launch(
            classify,
            dim=self.solver.world_count,
            inputs=[self.solver.constraint_count, self.solver.row_type, self.selected, self.fallback_counts],
            device=self.solver.model.device,
        )

    def fallback_rows(self, state_in, state_aug, contacts, dt):
        """Produce only fallback responses without reallocating canonical row identities."""
        owner, s = self.owner, self.solver
        model, device = s.model, s.model.device
        wp.launch_tiled(
            owner.kernels.prefix,
            dim=[s.world_count],
            inputs=[
                owner.plan,
                owner.data,
                s._joint_limit_q_index,
                model.joint_limit_lower,
                model.joint_limit_upper,
                state_in.joint_q,
                s.v_hat,
                int(s.enable_joint_limits),
                s.joint_limit_activation_gap,
                s.pgs_beta,
                s.pgs_cfm,
                self.dummy_counter,
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
            block_dim=32,
            device=device,
        )
        if contacts is not None and contacts.rigid_contact_max > 0:
            workers = min(contacts.rigid_contact_max, 16384)
            wp.launch_tiled(
                owner.kernels.contacts,
                dim=[workers],
                inputs=[
                    workers,
                    owner.plan,
                    owner.data,
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
                ],
                block_dim=32,
                device=device,
            )

    def predict_velocity(self, state_in, state_aug, dt, stage3_qd):
        """Publish the original predictor only on fallback worlds."""
        s = self.solver
        wp.launch(
            predict_fallback_velocity,
            dim=(s.world_count, 43),
            inputs=[
                self.owner.plan,
                self.selected,
                stage3_qd,
                s._kinematic_dof_mask,
                s._kinematic_joint_mask,
                dt,
                state_aug.joint_qdd,
                s.v_hat,
            ],
            device=s.model.device,
        )

    def prepare_velocity(self):
        """Preserve selected output ownership during old-solver preparation."""
        s = self.solver
        wp.launch(
            prepare_fallback_velocity,
            dim=(s.world_count, 43),
            inputs=[self.owner.plan, self.selected, s.v_hat, s.v_out],
            device=s.model.device,
        )

    def solve(self, rhs, iterations, omega, friction_start):
        """Run the selected complete owner after the masked original solve."""
        s = self.solver
        wp.launch_tiled(
            self.solve_kernel,
            dim=[s.world_count],
            inputs=[
                self.owner.plan,
                self.owner.data,
                self.force_input,
                self.packet_input,
                self.selected,
                self.packets,
                s.constraint_count,
                s.row_beta,
                s.row_type,
                s.row_parent,
                s.row_mu,
                iterations,
                omega,
                friction_start,
                s.contact_speculative_scale,
                s.impulses,
                s.v_out,
            ],
            block_dim=32,
            device=s.model.device,
        )

    def clear_force(self, state_aug):
        """Leave selected drive input untouched while clearing fallback recurrence."""
        wp.launch(
            clear_fallback_force,
            dim=(self.solver.world_count, 44),
            inputs=[self.owner.plan, self.selected, state_aug.body_ft_s],
            device=self.solver.model.device,
        )

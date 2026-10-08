# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind shared cached tree dynamics to the existing native sparse operator."""

import numpy as np
import warp as wp

from .sparse_factor import SparseData, SparsePlan
from .sparse_mass_matrix import _get_crba_sparse_factor_kernel
from .tree_kinetic_owner import TreeKineticState


def supported(solver):
    """Require complete cached-state ownership without changing row policy."""
    return bool(
        solver.model.device.is_cuda
        and solver._sparse_mass_matrix_size is not None
        and not solver.model.requires_grad
        and not solver.model.particle_count
        and not solver._has_free_rigid_bodies
        and not solver._mimic_count
        and not solver._connect_count
        and not solver._has_loop_joints
        and not solver._has_prescribed_response
        and solver._fk_id_cache_enabled
        and (solver.pgs_mode == "matrix_free" or solver._sparse_split)
        and solver._async_augmented_drives
        and solver._parallel_augmented_drive_topology
        and solver._articulation_dynamics_stream is not None
        and not solver.lazy_kinematics
        and not solver._fused_k1
        and not solver.enable_joint_velocity_limits
        and not solver._debug_buffers_enabled
        and not solver._double_buffer
        and getattr(solver, "_world_scan_publication", None) is None
        and getattr(solver, "_prismatic_publication", None) is None
        and getattr(solver, "_kinetic_world", None) is None
    )


class TreeSparseOperator:
    """Reuse the native factor representation, algorithm and refresh cadence."""

    def __init__(self, solver):
        self.solver = solver
        self.requires_body_velocity = True
        mass = solver._sparse_mass_matrix_plan
        n, device = mass.dof_count, solver.model.device
        counts = np.diff(mass.column_offsets).astype(np.int32)
        nodes = np.full((n, int(counts.max())), -1, np.int32)
        for col in range(n):
            begin, end = mass.column_offsets[col : col + 2]
            nodes[col, : end - begin] = mass.entry_rows[mass.column_entries[begin:end]]
        self.host = {
            "index": mass.lookup,
            "row": mass.entry_rows,
            "col": mass.columns,
            "source": mass.permutation[mass.columns],
            "inverse_nodes": nodes,
            "inverse_count": counts,
            "dof_joint": mass.dof_joint,
        }
        self.plan = SparsePlan()
        for name, values in self.host.items():
            setattr(self.plan, name, wp.array(values, dtype=int, device=device))
        self.plan.group_to_art = solver.group_to_art[n]
        self.plan.art_to_world = solver.art_to_world
        self.plan.art_dof_start = solver.articulation_dof_start
        self.plan.art_joint_start = solver.model.articulation_start
        self.plan.joint_child = solver.model.joint_child
        self.data = SparseData()
        self.data.W = solver._sparse_Linv
        # The native factor publishes status instead of a second validity bit.
        # Initial and reset mass requests are consumed before any prediction.
        self.data.valid = wp.ones(solver.world_count, dtype=int, device=device)
        self.data.status = solver._sparse_mass_matrix_status
        self.refresh_kernel = _get_crba_sparse_factor_kernel(
            n, mass.nonzero_count, warps_per_block=solver._sparse_factor_warps_per_block, geometric=True
        )
        self.kinetic_state = None

    def refresh(self, state_aug):
        """Read cached physical mass terms through the unchanged sparse factor."""
        s, n = self.solver, self.solver._sparse_mass_matrix_size
        wp.launch(
            self.refresh_kernel,
            dim=s.n_arts_by_size[n] * 32,
            inputs=[
                s.group_to_art[n],
                s.mass_update_mask,
                s.model.articulation_start,
                s.articulation_dof_start,
                s.model.joint_child,
                state_aug.joint_S_s,
                self.kinetic_state.geometric,
                s.R_by_size[n],
                1,
                s._augmented_drive_row_by_dof,
                s.aug_row_K,
                s._sparse_mass_matrix_indices,
            ],
            outputs=[s._sparse_Linv, s._sparse_mass_matrix_status],
            block_dim=32 * s._sparse_factor_warps_per_block,
            device=s.model.device,
        )


def create_owner(solver):
    """Keep an unsupported constructor on its complete existing native path."""
    if not supported(solver):
        return None
    operator = TreeSparseOperator(solver)
    try:
        owner = TreeKineticState(operator, admission=supported, chain_scan_default=True)
    except ValueError:
        return None
    operator.kinetic_state = owner
    # The complete sparse solve concludes before finish publishes the next
    # S/velocity cache. It has no late dense/MF scratch readers, which require
    # the ordinary split path's separate kinematics snapshot.
    solver._fk_id_cache_uses_snapshot = False
    solver._fk_id_cache = None
    return owner

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Admission for the existing sparse representation using shared tree state."""

from . import sparse_factor
from .tree_kinetic_owner import TreeKineticState


def supported(solver):
    """Keep unsupported constructors on the original complete sparse path."""
    owner = getattr(solver, "_sparse_factor", None)
    return bool(
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and not solver.model.particle_count
        and owner is not None
        and sparse_factor.supported(solver)
        and owner.level_update
        # State and held-operator generations do not depend on the row projector
        # or the mapping used to prepare the joint-limit prefix.
        and not owner.packet_rows
        and not owner.block_contacts
        and solver._fk_id_cache_enabled
        and not solver._fk_id_cache_uses_snapshot
        and solver._fk_id_cache is None
        and solver._async_augmented_drives
        and solver._parallel_augmented_drive_topology
        and solver._articulation_dynamics_stream is not None
        and not solver.lazy_kinematics
        and not solver._fused_k1
        and getattr(solver, "_world_scan_publication", None) is None
        and getattr(solver, "_prismatic_publication", None) is None
        and getattr(solver, "_kinetic_world", None) is None
        and getattr(owner, "spatial_dynamics", None) is None
        and getattr(owner, "lazy_compliance", None) is None
        and not getattr(owner, "small_step", False)
    )


class G1KineticState(TreeKineticState):
    """Keep the existing sparse representation's production admission."""

    def __init__(self, sparse_owner):
        super().__init__(sparse_owner, admission=supported if sparse_owner.solver.model.device.is_cuda else None)

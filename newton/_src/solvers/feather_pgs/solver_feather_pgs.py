# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import dataclasses
import functools
import math
import os
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from typing import ClassVar, Literal

import numpy as np
import warp as wp

from ...core.contact_pool import ContactField, model_contact_pool
from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags
from ...sim.articulation import eval_fk
from ...sim.contacts import contact_capacity_blocks
from ...sim.enums import BodyFlags, JointType
from ...utils.deprecation import deprecate_nonkeyword_arguments
from ..semi_implicit.kernels_contact import (
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
)
from ..semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from ..solver import SolverBase
from . import contact_compliance as _contact_compliance
from . import environment_flags as _environment_flags
from . import execution_plan as _execution_plan
from . import (
    field_group,
    host_index,
    kernel_template,
    packed_rows,
    prefix_rows,
    response_builds_dense,
    response_builds_rows,
    sweep_routes,
    sweep_routes_diag,
    sweep_routes_local,
    sweep_routes_sparse,
    sweep_routes_split,
    sweep_routes_tiers,
    sweep_units,
)
from . import row_metadata as _row_metadata
from .body_mapping import build_body_maps
from .conditioning import ArmatureFloor, conditioning_armature
from .configuration import check_options, resolve_kernel_choices
from .contact_input import ContactInput
from .contact_input import bind as bind_contact_input
from .contact_input import bind_routes as bind_contact_routes
from .contact_row_plan import SHARED_METADATA_PRODUCERS, contact_row_plan
from .contact_torsion import (
    configure_contact_torsion,
    prepare_torsion_rows,
    prepare_torsion_velocity_pass,
    validate_torsion_step,
)
from .contact_unit import CONTACT_UNIT_STATE_FIELDS, RING_BLOCK_SIZE, check_contact_update
from .dense_rows import RowSupport, bind_mf_body_map
from .dense_rows import bind as bind_dense_rows
from .dense_rows import bind_mf as bind_mf_rows
from .device_preludes import lane_group
from .field_group import SolverArrays
from .friction_patches import (
    _FrictionPatchState,
    patch_routes,
    patch_row_law,
)
from .joint_table import bind as bind_joint_table
from .joint_table import bind_free_bodies
from .kernel_template import Dispatch, KernelTemplate, SweepTier, python_kernel
from .kernels import (
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
    PREELIM_MAX_ROWS,
    RIGID_SPEED_ROWS,
    BodyFactors,
    allocate_connect_slots,
    allocate_joint_velocity_limit_slots,
    allocate_mimic_slots,
    allocate_physx_drive_slots,
    allocate_rigid_velocity_limit_slots,
    allocate_world_contact_slots,
    apply_augmented_mass_diagonal_grouped,
    apply_free_root_transport_to_predictor,
    apply_free_root_velocity_corrections,
    apply_world_contact_restitution_accumulated,
    apply_world_contact_restitution_matrix_free,
    build_joint_limit_rows_for_size,
    build_mass_update_mask,
    build_mf_contact_rows,
    clamp_free_root_velocity_limits,
    classify_local_solve_worlds,
    classify_parallel_sweep_tiers,
    clear_grouped_jacobian_active_rows,
    clear_local_solve_diag,
    clear_sparse_diagonal_response_prefix,
    compact_local_pair_candidates,
    compute_com_transforms,
    compute_contact_linear_force_from_impulses,
    compute_delta_and_accumulate,
    compute_mf_rhs_bias,
    compute_physx_pgs_drive_desc,
    compute_velocity_predictor,
    compute_world_contact_bias,
    eval_rigid_tau_and_augmented_drives,
    finalize_constraint_counts_with_status,
    gather_dense_warmstart,
    gather_mf_warmstart,
    gather_tau_to_groups,
    integrate_generalized_joints,
    invalidate_articulation_fk_id_cache,
    invalidate_fk_id_cache_for_articulations,
    local_solve_launch_gate,
    pack_contact_linear_force_as_spatial,
    pgs_convergence_diagnostic_velocity,
    pgs_ncp_residuals_diagnostic_velocity,
    place_world_contact_runs,
    populate_connect_J_for_size,
    populate_joint_velocity_limit_J_for_size,
    populate_mimic_J_for_size,
    populate_physx_drive_J_for_size,
    populate_rigid_velocity_limit_rows,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_fused_diagonal_joint_limits,
    prepare_world_contact_rows,
    prepare_world_impulses,
    prescale_joint_velocity_limits,
    record_world_contact_runs,
    remove_free_root_transport_from_qdd,
    reset_world_warmstart_buffers,
    rhs_accum_world_par_art,
    rhs_accum_world_par_row,
    scatter_qdd_from_groups,
    snapshot_contact_warmstart,
    snapshot_dense_phase_bound,
    solve_diagonal_mass,
    trisolve_loop,
    update_body_qd_from_featherstone,
    update_qdd_from_velocity,
    vector_add_inplace,
)
from .path_selection import (
    applied_options,
    contact_update_warnings,
    measurement_selection,
    options_selection,
    route_selection,
)
from .row_build import contact_row_prelude, populate_world_J_masked
from .row_laws import ROW_LAW
from .solve_policy import SolvePolicy, SolveReadoutRecord
from .solver_fields import (
    ARTICULATION_ORIGIN,
    ARTICULATION_WORK,
    AUGMENTED_DRIVE_ROWS,
    BODY_INERTIA,
    BODY_MAPS,
    CAPACITY_STATUS,
    CONNECT_PLAN,
    CONNECT_SLOTS,
    CONTACT_RUNS,
    DENSE_CONTACT_UNITS,
    DENSE_DELASSUS,
    DENSE_DRIVE_LIMITS,
    DENSE_ROW_WEIGHTS,
    DENSE_ROWS,
    DENSE_WARMSTART,
    DOF_WORK,
    DRIVE_SLOTS,
    FUSED_DIAGONAL_LIMITS,
    JOINT_FOREST,
    MF_BODY_MAP,
    MF_CONTACT_UNITS,
    MF_COUNTERS,
    MF_ROW_WEIGHTS,
    MF_ROWS,
    MF_WARMSTART,
    MIMIC_PLAN,
    MIMIC_SLOTS,
    OVERFLOW_WARNING,
    PGS_DIAGNOSTICS,
    PGS_MF_DIAGNOSTICS,
    PREELIMINATION,
    ROW_SLOT_UNBOUNDED,
    ROW_WATERMARKS,
    SIZE_FACTORS,
    SIZE_GROUPED_SOLVE,
    SIZE_SYSTEM,
    SPARSE_CONTACT_GROUPS,
    SPARSE_CONTACT_RECORDS,
    SPARSE_CONTACT_SERIAL,
    SPARSE_DIAGONAL_ROWS,
    SPARSE_FACTOR,
    SPARSE_ROWS,
    STAGE3_SNAPSHOTS,
    STATE_AUX,
    STATE_GRAD_SCRATCH,
    VELOCITY_LIMIT_SLOTS,
    WORLD_RESPONSE,
    WORLD_ROW_STATE,
    Connects,
    LocalOwners,
    Mimics,
    Preelim,
)
from .sparse_contact import (
    CONTACT_RECORD_FLOATS,
    CONTACT_RING_FLOATS,
    finalize_sparse_joint_limit_rows,
)
from .sparse_mass_matrix import (
    _SparseRowSupport,
    factor_indices,
    row_supports,
)
from .staging import STATIC_SHARED_BYTES, arch_budget
from .step_plan import AdmissionFacts, StepPlan, first_failed
from .sweep_routes_sparse import _SPARSE_LOCAL_WORKSPACE_BYTES, CONTACT_TRANSACTIONS
from .tree_passes import (
    SERIAL_BLOCK_DIM,
    DynamicsData,
    composite_inertia,
    crba_fill,
    fk_id,
    inverse_dynamics_tau,
    refresh_body_inertia,
)
from .tree_plan import TreePlan, local_parents
from .velocity_limit_law import check_velocity_limit_law

_SPARSE_CONTACT_DIRECT = os.environ.get("FEATHER_PGS_SPARSE_CONTACT_DIRECT") == "1"
_MF_EXACT_ROWSUM = (
    os.environ.get("FEATHER_PGS_MF_EXACT_ROWSUM") == "1"
)  # experiment: matrix-free sweep with the exact A-based step scale
_TIER_BLOCKS = int(
    os.environ.get("FEATHER_PGS_TIER_BLOCKS", "0")
)  # experiment: parallel tiers grid-stride over per-tier world lists (0 = off)
_FPGS_SYNC_TIMINGS = os.environ.get("FEATHER_PGS_SYNC_TIMINGS", "0").lower() in {"1", "true", "yes", "on"}
_FPGS_SYNC_TIMINGS_START = int(os.environ.get("FEATHER_PGS_SYNC_TIMINGS_START", "20"))
_FPGS_SYNC_TIMINGS_COUNT = max(int(os.environ.get("FEATHER_PGS_SYNC_TIMINGS_COUNT", "1")), 0)
_SMALL_DOF_THRESHOLD_DEFAULT = 12
# Beyond this guard, the relaxation weight is too close to zero to be a
# useful float32 PGS update and the assembled split path risks overflow when
# it converts the weight back to a diagonal regularizer.
_MAX_CONTACT_REGULARIZATION = 1.0e6


# Per-world preset of the first rejected row slot: no reservation was rejected.
_ROW_SLOT_UNBOUNDED = ROW_SLOT_UNBOUNDED
_CONTACT_BUILD_THREAD_CAP = 65536
_CONTACT_THREADS_X = max(1, int(os.environ.get("FEATHER_PGS_CONTACT_THREADS_X", "1")))
_ROWS_THREADS_X = max(1, int(os.environ.get("FEATHER_PGS_ROWS_THREADS_X", "4")))
_CONTACT_JACOBIAN_WORKER_CAP = 4096
_CONTACT_JACOBIAN_MAX_DOF = 10
_JOINT_LIMIT_WARPS_PER_BLOCK = 4
_CRBA_CHOLESKY_WARPS_PER_BLOCK = 4
# A persistent queue avoids launching one expensive general-solver block for
# every mostly local world. Eight blocks per SM preserves general-only latency.
_LOCAL_GENERAL_BLOCKS_PER_SM = 8
_LOCAL_PAIR_BLOCKS_PER_SM = 8
# The local solver performs a serial O(dof_count**2) triangular solve per row.
_LOCAL_INTERNAL_MAX_DOF = 16
_LOCAL_SOLVE_MAX_ROWS = 20
_LOCAL_RESIDUAL_MAX_ROWS = 40
_LOCAL_RESIDUAL_MF_MAX_ROWS = 12
_PERSISTENT_HINV_JT_MIN_DOF = 17


@wp.kernel
def _prepare_augmented_joint_drives_by_dof(
    drive_row_by_dof: wp.array[int],
    drive_q_index_by_dof: wp.array[int],
    drive_target_index_by_dof: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    dt: float,
    row_K: wp.array[float],
    drive_tau: wp.array[float],
):
    """Evaluate dynamic drive terms from an immutable DOF-to-row topology."""
    dof = wp.tid()
    drive_tau[dof] = 0.0

    row = drive_row_by_dof[dof]
    if row < 0:
        return

    ke = joint_target_ke[dof]
    kd = joint_target_kd[dof]
    K = ke * dt * dt + kd * dt
    if K <= 0.0:
        row_K[row] = 0.0
        return

    q = joint_q[drive_q_index_by_dof[dof]]
    qd = joint_qd[dof]
    u0 = -(ke * (q - joint_target_pos[drive_target_index_by_dof[dof]] + dt * qd) + kd * (qd - joint_target_vel[dof]))
    effort_limit = joint_effort_limit[dof]
    if effort_limit > 0.0:
        u0 = wp.clamp(u0, -effort_limit, effort_limit)
    row_K[row] = K
    drive_tau[dof] = u0


@wp.kernel
def _accumulate_row_watermark(
    counts: wp.array[wp.int32],
    watermark: wp.array[wp.int32],
):
    # Per-world reduce-max into a single scalar slot. Read-only on ``counts``;
    # writes only the separate ``watermark`` buffer, so this is numerics-neutral.
    # Matches the perf-harness pattern in newton_solver_core_perf.py.
    tid = wp.tid()
    wp.atomic_max(watermark, 0, counts[tid])


@wp.kernel
def _accumulate_row_capacity_telemetry(
    raw_counts: wp.array[wp.int32],
    dropped_contact_rows: wp.array[wp.int32],
    capacity: int,
    raw_watermark: wp.array[wp.int32],
    dropped_contact_rows_watermark: wp.array[wp.int32],
    overflow_excess_watermark: wp.array[wp.int32],
    overflow_world_steps: wp.array[wp.int32],
):
    """Accumulate raw row demand and capacity overflow without changing solver state."""
    tid = wp.tid()
    dropped = dropped_contact_rows[tid]
    count = raw_counts[tid]  # monotone counter: accepted rows plus rejected reservations
    wp.atomic_max(raw_watermark, 0, count)
    wp.atomic_max(dropped_contact_rows_watermark, 0, dropped)
    excess = count - capacity
    if excess > 0:
        wp.atomic_max(overflow_excess_watermark, 0, excess)
        wp.atomic_add(overflow_world_steps, 0, 1)


@wp.kernel
def _clear_dense_row_state(
    slot_counter: wp.array[int],
    dropped: wp.array2d[int],
    dense_first_rejected_slot: wp.array[int],
    has_free_rigid: int,
    mf_slot_counter: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    mf_constraint_count: wp.array[int],
    mf_impulses: wp.array2d[float],
):
    """Clear allocation state, row-loss counters and free-body row state in one launch."""
    world, column = wp.tid()
    if column == 0:
        slot_counter[world] = 0
        for family in range(2):
            dropped[family, world] = 0
        dense_first_rejected_slot[world] = _ROW_SLOT_UNBOUNDED
        if has_free_rigid != 0:
            mf_slot_counter[world] = 0
            mf_first_rejected_slot[world] = _ROW_SLOT_UNBOUNDED
            mf_constraint_count[world] = 0
    if has_free_rigid != 0:
        mf_impulses[world, column] = 0.0


@wp.kernel
def _finalize_constraint_status(
    dense_count: wp.array[int],
    mf_count: wp.array[int],
    dropped: wp.array2d[int],
    contact_count: wp.array[int],
    reduction_overflow: wp.array[int],
    dense_capacity: int,
    mf_capacity: int,
    contact_capacity: int,
    overflow: wp.array[wp.bool],
):
    """Latch invalid-world status until reset; counts come from the per-family finalize kernels."""
    world = wp.tid()
    if (
        dense_count[world] > dense_capacity
        or mf_count[world] > mf_capacity
        or dropped[0, world] > 0
        or dropped[1, world] > 0
        or contact_count[0] > contact_capacity
        or reduction_overflow[0] != 0
    ):
        overflow[world] = True


@wp.kernel
def _reset_solver_status(
    world_mask: wp.array[wp.bool],
    articulation_world: wp.array[int],
    mass_update_requested: wp.array[int],
    overflow: wp.array[wp.bool],
):
    """Clear selected world status and request only its articulation factors."""
    tid = wp.tid()
    if tid < overflow.shape[0]:
        if not world_mask or world_mask[tid] != 0:
            overflow[tid] = False
    if tid < mass_update_requested.shape[0]:
        world = articulation_world[tid]
        if not world_mask or world_mask[world] != 0:
            mass_update_requested[tid] = 1


@wp.kernel
def _warn_constraint_row_overflow(
    dense_raw_counts: wp.array[wp.int32],
    dense_dropped_contact_rows: wp.array[wp.int32],
    dense_capacity: int,
    mf_raw_counts: wp.array[wp.int32],
    mf_dropped_contact_rows: wp.array[wp.int32],
    mf_capacity: int,
    mf_active: int,
    warning_emitted: wp.array[wp.int32],
):
    """Emit one device-side warning per overflowing FeatherPGS row family."""
    world = wp.tid()

    dense_dropped = dense_dropped_contact_rows[world]
    dense_requested = dense_raw_counts[world]
    if dense_requested > dense_capacity and wp.atomic_exch(warning_emitted, 0, 1) == 0:
        wp.printf(
            "Warning: FeatherPGS dense constraint-row overflow in world %d: requested %d rows, limit %d; "
            "dropped %d contact/friction rows. Increase dense_max_constraints.\n",
            world,
            dense_requested,
            dense_capacity,
            dense_dropped,
        )

    if mf_active != 0:
        mf_dropped = mf_dropped_contact_rows[world]
        mf_requested = mf_raw_counts[world]
        if mf_requested > mf_capacity and wp.atomic_exch(warning_emitted, 1, 1) == 0:
            wp.printf(
                "Warning: FeatherPGS matrix-free constraint-row overflow in world %d: requested %d rows, "
                "limit %d; dropped %d contact/friction rows. Increase mf_max_constraints.\n",
                world,
                mf_requested,
                mf_capacity,
                mf_dropped,
            )


def _world_diagonal_pairs(
    articulation_world: np.ndarray, response_dof_count: np.ndarray, world_count: int, diagonal_sizes: frozenset[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split each world's two response articulations into the one whose size is in ``diagonal_sizes`` and the other.

    Returns:
        ``(diagonal, dense)`` articulation indices per world, or ``None`` unless every
        world has exactly two response articulations, exactly one of them diagonal.
    """
    articulations = np.flatnonzero(response_dof_count > 0)
    bounds, order = host_index.group_by(articulation_world[articulations], world_count)
    if np.any(np.diff(bounds) != 2):
        return None
    # Each world's pair, in ascending articulation order.
    pairs = articulations[order].reshape(world_count, 2)
    diagonal = np.isin(response_dof_count[pairs], list(diagonal_sizes))
    if np.any(diagonal.sum(axis=1) != 1):
        return None
    first_diagonal = diagonal[:, 0]
    return np.where(first_diagonal, pairs[:, 0], pairs[:, 1]), np.where(first_diagonal, pairs[:, 1], pairs[:, 0])


def _velocity_prescale_launch(articulation_start: np.ndarray, *, joint_count: int, cuda: bool) -> tuple[int, int]:
    """Return the lanes per articulation and the thread count of the joint velocity prescale.

    A lane group as wide as the mean articulation's joint count, rounded up to a
    power of two of at most one warp, covers a typical articulation in one
    strided pass while many articulations share the device; larger ones take a
    few passes. CUDA launches cover complete warps. CPU threads cannot exchange
    values, so a CPU launch runs one lane per articulation.

    Raises:
        ValueError: If the last articulation does not end at ``joint_count``. The
            prescale writes no DOF of a joint past it.
    """
    count = len(articulation_start) - 1
    if count <= 0:
        return 1, 0
    if int(articulation_start[-1]) != joint_count:
        raise ValueError(
            "SolverFeatherPGS requires Model.articulation_start to end with the joint count "
            f"{joint_count}, got {int(articulation_start[-1])}; ModelBuilder.finalize() writes this sentinel."
        )
    if not cuda:
        return 1, count
    mean_joints = (int(articulation_start[-1]) - int(articulation_start[0])) / count
    lanes = 1
    while lanes < 32 and lanes < mean_joints:
        lanes *= 2
    return lanes, (count * lanes + 31) // 32 * 32


@dataclass(frozen=True)
class _FeatherPGSModelPlan:
    """Immutable articulation and generalized-response plan."""

    articulation_dof_count: np.ndarray
    articulation_dof_start: np.ndarray
    articulation_world: np.ndarray
    is_free_rigid: np.ndarray
    response_free_rigid_body_indices: np.ndarray
    prescribed_articulation: np.ndarray
    response_dof_count: np.ndarray
    articulation_joint_end: np.ndarray
    loop_joint_articulation: np.ndarray
    world_count: int

    @classmethod
    def build(
        cls, model: Model, kinematic_dof_mask: np.ndarray, *, enable_prescribed_response: bool
    ) -> "_FeatherPGSModelPlan":
        """Build the immutable response layout from physical model topology."""
        articulation_count = model.articulation_count
        articulation_dof_count = np.zeros(articulation_count, dtype=np.int32)
        articulation_dof_start = np.zeros(articulation_count, dtype=np.int32)
        articulation_world = np.zeros(articulation_count, dtype=np.int32)
        is_free_rigid = np.zeros(articulation_count, dtype=np.int32)
        free_rigid_body_indices = np.zeros(0, dtype=np.int32)

        if articulation_count:
            if model.articulation_world is None:
                articulation_world = np.arange(articulation_count, dtype=np.int32)
            else:
                articulation_world = model.articulation_world.numpy().astype(np.int32, copy=True)
                articulation_world[articulation_world < 0] = 0

        articulation_joint_end = np.zeros(articulation_count, dtype=np.int32)
        loop_joint_articulation = np.full(model.joint_count, -1, dtype=np.int32)
        if articulation_count and model.joint_count:
            articulation_start = model.articulation_start.numpy()
            joint_parent = model.joint_parent.numpy()
            joint_qd_start = model.joint_qd_start.numpy()
            joint_type = model.joint_type.numpy()
            joint_child = model.joint_child.numpy()

            joint_articulation_arr = getattr(model, "joint_articulation", None)
            joint_articulation = joint_articulation_arr.numpy() if joint_articulation_arr is not None else None

            first_joint = articulation_start[:articulation_count]
            last_joint = articulation_start[1 : articulation_count + 1]

            # Loop-closing joints (how the MJCF/USD importers close `connect`/`weld`
            # equalities) are not part of the kinematic tree. Excluding them here
            # keeps their DOFs out of the mass matrix (garbage motion subspaces made
            # H singular -> instant NaN) and out of the FK walk (they would overwrite
            # their child's tree pose). They are consumed as CONNECT rows instead.
            if joint_articulation is not None:
                # An unowned joint is a closure only when its child body already
                # belongs to an articulation through a different, tree-owned joint.
                # Standalone world-root joints are also unowned, but their child has
                # no articulation owner and must remain outside the solver plan.
                # A body's last owned joint names its articulation.
                joint_index = np.arange(model.joint_count)
                owned = (joint_articulation >= 0) & (joint_child >= 0)
                body_joint = np.full(model.body_count, -1, dtype=np.int64)
                np.maximum.at(body_joint, joint_child[owned], joint_index[owned])
                body_articulation = np.full(model.body_count, -1, dtype=np.int32)
                has_owner = body_joint >= 0
                body_articulation[has_owner] = joint_articulation[body_joint[has_owner]]
                closing = (joint_articulation < 0) & (joint_child >= 0)
                loop_joint_articulation[closing] = body_articulation[joint_child[closing]]

                # Model.joint_articulation identifies the articulation's own contiguous
                # tree joints; the body-ownership pass above separately distinguishes
                # closures from other unowned joints. The prefix range derived from
                # articulation_start may contain foreign entries after the tree — e.g.
                # multi-source add_builder defers loop joints past a later articulation's
                # free joint — so never adopt range members by position.
                range_art, position = host_index.expand(np.maximum(last_joint - first_joint, 0))
                member = joint_articulation[first_joint[range_art] + position] == range_art
                member_count = np.bincount(range_art[member], minlength=articulation_count)
                # Members must open their range: as many members precede each one as joints do.
                members_before = host_index.segmented_rank(range_art, member)
                stray = np.flatnonzero(member & (members_before != position))
                if stray.size:
                    art = int(range_art[stray[0]])
                    j = int(first_joint[art] + position[stray[0]])
                    tree_end = int(first_joint[art] + members_before[stray[0]])
                    raise ValueError(
                        f"SolverFeatherPGS: articulation {art}'s joints are not "
                        f"contiguous (joint {j} follows a foreign joint at "
                        f"{tree_end}). The ModelBuilder guarantees contiguity; "
                        "this model is malformed."
                    )
                articulation_joint_end[:] = first_joint + member_count
            else:
                for art in range(articulation_count):
                    # Legacy models without joint_articulation: fall back to the
                    # second-inbound-joint heuristic (loop joints must be trailing).
                    seen_children: set[int] = set()
                    tree_end = int(first_joint[art])
                    for j in range(int(first_joint[art]), int(last_joint[art])):
                        child = int(joint_child[j])
                        if child in seen_children:
                            break  # first loop joint; the rest of the range must be loops too
                        seen_children.add(child)
                        tree_end = j + 1
                    for j in range(tree_end, int(last_joint[art])):
                        if int(joint_child[j]) not in seen_children:
                            raise ValueError(
                                "SolverFeatherPGS: articulation joint range mixes tree joints "
                                f"after loop joints (articulation {art}, joint {j}). "
                                "Loop-closing joints must be trailing."
                            )
                        loop_joint_articulation[j] = art
                    articulation_joint_end[art] = tree_end

            articulation_dof_start[:] = joint_qd_start[first_joint]
            articulation_dof_count[:] = joint_qd_start[articulation_joint_end] - articulation_dof_start
            single = np.flatnonzero(articulation_joint_end - first_joint == 1)
            root = first_joint[single]
            free = single[(joint_type[root] == int(JointType.FREE)) & (joint_parent[root] == -1)]
            is_free_rigid[free] = 1
            free_rigid_body_indices = joint_child[first_joint[free]]

        prescribed = np.zeros(articulation_count, dtype=np.int32)
        if enable_prescribed_response and articulation_count and model.joint_count:
            candidates = np.zeros(articulation_count, dtype=np.int32)
            free_six = np.flatnonzero((is_free_rigid != 0) & (articulation_dof_count == 6))
            free_dofs = articulation_dof_start[free_six, None] + np.arange(6)
            candidates[free_six[np.all(kinematic_dof_mask[free_dofs] != 0, axis=1)]] = 1

            # Never elide the only response articulation in a world. Such
            # worlds retain the exact large-armature kinematic fallback.
            retained_response = np.zeros(int(np.max(articulation_world)) + 1, dtype=bool)
            retained_response[articulation_world[(articulation_dof_count > 0) & (candidates == 0)]] = True
            prescribed[(candidates != 0) & retained_response[articulation_world]] = 1

        response_dof_count = articulation_dof_count.copy()
        response_dof_count[prescribed != 0] = 0
        free_articulations = np.nonzero(is_free_rigid != 0)[0]
        free_bodies = np.asarray(free_rigid_body_indices, dtype=np.int32)
        response_free_rigid_body_indices = free_bodies[prescribed[free_articulations] == 0]
        world_count = int(np.max(articulation_world)) + 1 if articulation_count else 0

        arrays = (
            articulation_dof_count,
            articulation_dof_start,
            articulation_world,
            is_free_rigid,
            response_free_rigid_body_indices,
            prescribed,
            response_dof_count,
            articulation_joint_end,
            loop_joint_articulation,
        )
        for array in arrays:
            array.setflags(write=False)
        return cls(*arrays, world_count)


def _joint_forest(model: Model) -> TreePlan:
    """The forest of the model's articulation joints by their ancestry (``Model.joint_ancestor``); a joint
    outside every articulation, a loop closure, stands alone."""
    if not model.joint_count:
        return TreePlan.build(np.zeros(0, dtype=np.int32))
    member = model.joint_articulation.numpy() >= 0
    try:
        return TreePlan.build(np.where(member, model.joint_ancestor.numpy(), -1))
    except ValueError as error:
        raise ValueError(f"SolverFeatherPGS: the model's joint ancestry is not a forest ({error})") from error


# Retain the internal metadata API while sharing its implementation.
_DENSE_META_ROW_TYPE_BITS = _row_metadata._DENSE_META_ROW_TYPE_BITS
_DENSE_META_ROW_TYPE_MASK = _row_metadata._DENSE_META_ROW_TYPE_MASK
_DENSE_META_MAX_PARENT = _row_metadata._DENSE_META_MAX_PARENT
_MF_META_MAX_CAPACITY = _row_metadata._MF_META_MAX_CAPACITY
_validate_dense_metadata_encoding = _row_metadata._validate_dense_metadata_encoding
_validate_mf_metadata_encoding = _row_metadata._validate_mf_metadata_encoding
_mf_parent_expression = _row_metadata._mf_parent_expression


# Retain internal imports while launch-resource policy has one implementation.
_MFGS_RESIDENT_METADATA_MAX_BYTES = _execution_plan._MFGS_RESIDENT_METADATA_MAX_BYTES
_HINV_JT_MAX_CHUNK_SIZE = _execution_plan._HINV_JT_MAX_CHUNK_SIZE
_MFGSStorage = _execution_plan._MFGSStorage
_align_shared_memory = _execution_plan._align_shared_memory
_estimate_cholesky_shared_memory = _execution_plan._estimate_cholesky_shared_memory
_estimate_hinv_jt_shared_memory = _execution_plan._estimate_hinv_jt_shared_memory
_select_hinv_jt_chunk_size = _execution_plan._select_hinv_jt_chunk_size
_FeatherPGSExecutionPlan = _execution_plan._FeatherPGSExecutionPlan

# Warnings raised in the constructor body name the constructor's caller, two frames up: the
# body runs inside the keyword-only shim (deprecate_nonkeyword_arguments).
_CONSTRUCTOR_CALLER = 3


class SolverFeatherPGS(SolverBase):
    """A semi-implicit integrator using symplectic Euler that operates
    on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Instead of maximal coordinates :attr:`~newton.State.body_q` (rigid body positions) and :attr:`~newton.State.body_qd`
    (rigid body velocities) as is the case in :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`,
    :class:`~newton.solvers.SolverFeatherPGS` uses :attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd` to represent
    the positions and velocities of joints without allowing any redundant degrees of freedom.

    After constructing :class:`~newton.Model` and :class:`~newton.State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Note:
        Unlike :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverFeatherPGS`
        does not simulate rigid bodies with nonzero mass as floating bodies if they are not connected through any joints.
        Floating-base systems require an explicit free joint with which the body is connected to the world,
        see :meth:`newton.ModelBuilder.add_joint_free`.

    Semi-implicit integration is not unconditionally stable. Accuracy and
    stability depend on the timestep, inertia conditioning and constraint
    convergence; contact and friction also exchange and dissipate energy.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    This solver uses the routines from :class:`~newton.solvers.SolverSemiImplicit` to simulate particles, cloth, and soft bodies.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverFeatherPGS(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    Rigid-contact restitution is part of the ordinary normal contact row in
    ``"split"`` and ``"matrix_free"`` modes. It uses the
    arithmetic mean of the two shape coefficients and freezes the incident
    relative normal velocity before contact impulses are applied. A sufficiently
    fast contact predicted to reach the surface during the step replaces its
    geometric/speculative bias with Newton's rebound target. Optional
    ``pgs_velocity_iterations`` refine the result but are never enabled
    implicitly. As in a conventional discrete PGS solver, the qualifying
    rebound velocity is used for the whole step; this gives the intended
    post-impact velocity but a first-order, impact-phase-dependent position
    offset. Reduce the timestep when substep impact position matters.

    ``constraint_overflow`` is a device boolean array with one entry per solver
    world. It records contact-buffer overflow or dropped constraint rows even
    when detailed row watermarks are disabled, and persists until :meth:`reset`.
    Global mesh-reducer table/buffer losses conservatively mark every world
    invalid because that producer does not attribute losses to individual worlds.
    Other broad/narrow-phase pair-buffer failures are outside this status contract.
    Read it in an RL observation/termination kernel, or call
    :meth:`check_constraint_capacity` outside capture at an observation boundary
    to reject incomplete physics. Ordinary stepping does not synchronize it.

    The solver factors each articulation's mass matrix in float32. Along a
    serial chain its pivots fall roughly as the cube of the chain's length, so
    a long chain without joint armature leaves float32's resolution: where an
    articulation with a root-to-leaf path of at least 32 joints has a pivot
    below ``4 n u`` of its diagonal (``n`` coordinates, ``u = 2^-24``) at the
    model's configuration, the solver adds the least joint armature that lifts
    it there, warns, and reports it under ``routes.dynamics.armature_floor`` of
    :meth:`get_path_selection`. Give such joints armature to choose it yourself.

    Single-body FREE-joint articulations use an energy-preserving local
    gyroscopic update to prevent explicit angular-bias runaway. Fast rotation
    can trigger bounded gyro microsteps inside the velocity predictor, without
    additional collision or contact-solver passes. Pose integration remains
    first-order; accurate rotational trajectories still require a suitable
    simulation timestep.

    Construction chooses each world's owners, factors, rows and sweeps from the
    model's topology, the options and the device's resources (never by timing);
    :meth:`get_path_selection` records every choice with its reason. The
    ``FEATHER_PGS_*`` environment switches it reads are declared, each with its
    meaning, in ``environment_flags.FLAGS`` of this package, and the record lists
    those present at construction (``environment``).

    """

    # The default configuration is conformance-tested against body-pair-reduced
    # contact buffers (test_contact_reduction_body_pairs). Warm-start modes have
    # stricter contact-identity requirements and reject those buffers in step().
    supports_body_pair_reduced_contacts: bool = True

    joint_qd_public_convention: bool = True
    """Whether free-joint ``joint_qd`` uses Newton's public twist convention.

    FeatherPGS stores free-joint ``joint_qd`` as ``(v_com_world, omega_world)`` [m/s, rad/s],
    matching :func:`newton.eval_fk`. The vanilla Featherstone solver instead uses an internal
    world-origin-referenced spatial twist, which requires
    ``eval_fk_with_velocity_conversion``.

    Integration layers that refresh maximal body state from joint state (for example after
    reset writes) must dispatch on this attribute. Applying the Featherstone-internal helper
    to public-convention ``joint_qd`` re-references the free-joint twist to the world origin,
    adding ``omega x x_com_world`` of phantom linear velocity to :attr:`newton.State.body_qd`
    that grows with the body's distance from the world origin.
    """

    # Test hook: pin a kernel implementation regardless of the size heuristic
    # (keys: cholesky_kernel, trisolve_kernel, hinv_jt_kernel, delassus_kernel).
    # sparse_mass_matrix=False retains the existing representation for comparisons.
    _kernel_overrides: ClassVar[dict[str, str | bool]] = {}

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register PhysX rigid-body attributes consumed by FeatherPGS.

        ``rigid_body_max_linear_velocity`` [m/s] and ``rigid_body_max_angular_velocity``
        [rad/s] bound the speeds of a free body, the magnitudes of its center of mass
        velocity and of its angular velocity (``physxRigidBody:maxLinearVelocity`` and
        ``physxRigidBody:maxAngularVelocity``, default unbounded).
        ``rigid_body_max_depenetration_velocity`` [m/s] bounds its contact correction velocity.
        """

        def _angular_velocity_deg_to_rad(value, _context):
            if value is None:
                return float("inf")
            value = float(value)
            if not np.isfinite(value):
                return float("inf")
            return value * np.pi / 180.0

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rigid_body_max_linear_velocity",
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=float("inf"),
                usd_attribute_name="physxRigidBody:maxLinearVelocity",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rigid_body_max_angular_velocity",
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=float("inf"),
                usd_attribute_name="physxRigidBody:maxAngularVelocity",
                usd_value_transformer=_angular_velocity_deg_to_rad,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rigid_body_max_depenetration_velocity",
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=float("inf"),
                usd_attribute_name="physxRigidBody:maxDepenetrationVelocity",
            )
        )

    @property
    def contact_torsion_radius(self) -> float:
        """Experimental construction-only effective spin radius [m]."""
        return self._contact_torsion_radius

    @property
    def contact_torsion_shape_indices(self) -> tuple[int, ...] | None:
        """Experimental construction-only global shape selection."""
        return self._contact_torsion_shape_indices

    @property
    def contact_torsion_shape_patterns(self) -> tuple[str, ...] | None:
        """Experimental construction-only shape-label regular expressions."""
        return self._contact_torsion_shape_patterns

    @deprecate_nonkeyword_arguments(last_positional="friction_anchor_beta")
    def __init__(
        self,
        model: Model,
        *,
        angular_damping: float = 0.05,
        update_mass_matrix_interval: int = 1,
        enable_contact_friction: bool = True,
        contact_friction_gap_threshold: float = float("inf"),
        contact_friction_position_iterations: int = -1,
        contact_friction_shared_anchor: bool = False,
        contact_friction_anchor_limit: int = 0,
        contact_friction_scale: float = 1.0,
        contact_shared_anchor: bool = False,
        enable_joint_limits: bool = False,
        enable_bilateral_preelimination: bool = False,
        joint_limit_activation_gap: float = float("inf"),
        enable_joint_velocity_limits: bool = False,
        velocity_limit_activation_fraction: float = 0.0,
        fuse_joint_velocity_limits: bool = True,
        pgs_iterations: int = 12,
        pgs_velocity_iterations: int = 0,
        pgs_beta: float = 0.2,
        pgs_cfm: float = 1.0e-6,
        pgs_omega: float = 1.0,
        pgs_contact_regularization: float = 0.0,
        pgs_velocity_drive_mode: Literal["active", "freeze"] = "freeze",
        dense_max_constraints: int = 32,
        pgs_warmstart: bool = False,
        mf_warmstart: bool = False,
        mf_warmstart_decay: float = 1.0,
        pgs_mode: str = "split",
        articulated_contact_response: Literal[
            "immediate", "propagation", "propagation-fused", "propagation-colored"
        ] = "immediate",
        propagation_same_articulation_rows: bool = False,
        propagation_cached_response: bool = True,
        propagation_cached_response_max_bodies: int = 8,
        pgs_schedule: Literal["interleaved", "contact_then_internal", "physx_grasp"] = "interleaved",
        mf_max_constraints: int = 512,
        # Dense-path PGS kernel (matrix_free solves through the fused MF-GS kernel)
        pgs_kernel: str = "auto",
        pgs_chunk_size: int | None = None,
        # Parallelism options
        use_parallel_streams: bool = True,
        double_buffer: bool = True,
        nvtx: bool = False,
        pgs_debug: bool = False,
        drive_mode: Literal["augmented", "physx_pgs"] = "augmented",
        tile_threads: int = 64,
        row_watermark: bool = False,
        restitution_velocity_threshold: float = 0.5,
        contact_speculative_scale: float = 1.0,
        contact_gap_gate: float = 0.0,
        contact_friction_articulation_pairs_only: bool = False,
        enable_restitution: bool = True,
        same_articulation_contact_gap_gate: float = 0.0,
        articulation_pair_contact_gap_gate: float = 0.0,
        pgs_warmstart_decay: float = 1.0,
        warn_constraint_overflow: bool = True,
        friction_anchor_beta: float | None = None,
        pgs_solve_readout: Literal["off", "sweeps", "trace"] = "off",
        mf_gs_incremental_rows: int = 32,
        mf_gs_parallel_rows: int = 0,
        mf_gs_parallel_sweeps: int = 24,
        mf_gs_parallel_linked_sweeps: int | Literal["auto"] | None = "auto",
        mf_gs_parallel_nesterov: bool = True,
        mf_gs_parallel_matrix_free: bool = False,
        mf_gs_parallel_linked_velocity_limit_pass: bool = False,
        velocity_limit_law: Literal["stateless", "accumulated"] = "stateless",
        contact_update: Literal["sequential", "unit"] = "sequential",
        bilateral_preelimination_include_mimics: bool = True,
        contact_torsion_radius: float = 0.0,
        contact_torsion_shape_indices: tuple[int, ...] | None = None,
        contact_torsion_shape_patterns: tuple[str, ...] | None = None,
        contact_torsion_device: bool = False,
        contact_compliance: bool = False,
        cholesky_kernel: Literal["auto", "tiled", "loop"] = "auto",
        trisolve_kernel: Literal["auto", "tiled", "loop"] = "auto",
        hinv_jt_kernel: Literal["auto", "tiled", "par_row"] = "auto",
        delassus_kernel: Literal["auto", "tiled", "par_row_col"] = "auto",
        sparse_mass_matrix: Literal["auto", "on", "off"] = "auto",
        cached_dynamics: Literal["auto", "on", "off"] = "auto",
    ):
        """Create the solver for ``model``.

        Every option after ``model`` is keyword-only. Passing the options from
        ``angular_damping`` through ``friction_anchor_beta`` positionally, as earlier
        releases allowed, is deprecated: they still bind in declaration order, with a
        :class:`DeprecationWarning` that names the keywords to use. The later options
        were always keyword-only and still raise :class:`TypeError` when passed
        positionally.

        Args:
            model (Model): the model to be simulated.
            cholesky_kernel: Factor kernel. ``auto`` uses topology, size and device resources.
                Explicit choices override private test hooks and disable competing sparse/paired owners.
            trisolve_kernel: Triangular solve kernel; ``auto`` selects by articulation size.
            hinv_jt_kernel: Dense contact response kernel; ``auto`` selects by size and resources.
            delassus_kernel: Dense Delassus assembly kernel; explicit choices require split mode.
            sparse_mass_matrix: Experimental native sparse factors: ``auto`` retains topology/resource
                admission, ``on`` requires admission, and ``off`` retains the ordinary representation.
                Only the held-tree owner evaluates them, so ``cached_dynamics='off'`` keeps the ordinary one.
                Compatible split sweeps and velocity limits compose automatically.
                Legacy sparse-composition opt-outs apply to ``auto``; ``on`` overrides those gates.
                This option does not enable research packed-factor owners. Incompatible forced choices
                raise ValueError; this experimental selector may change without notice.
            cached_dynamics: Experimental paired cached prediction and next-state publication.
                ``auto`` selects a qualified cache for the current topology, factor and row policy;
                unsupported cases use ordinary prediction/publication. ``on`` requires admission;
                ``off`` disables these paired caches. Explicit choices override legacy cache
                environment settings; ``auto`` retains those settings when present. Under ``auto``
                the held-tree owner is admitted wherever its topology, representation and row policy
                allow (``FEATHER_PGS_MIXED_TREE_KINETIC_STATE=0`` disables it). Explicit dense kernels
                disable auto and conflict with ``on``. Factor refresh intervals and numerical budgets
                are unchanged. Where the held-tree owner is admitted, a diagonal-mass group of
                world-fixed roots over one-coordinate prismatic or revolute leaves runs as its scalar
                branches by default (``FEATHER_PGS_SCALAR_BRANCHES=0`` opts out);
                :meth:`get_path_selection` reports it under ``routes.held_tree.scalar_branches``.
            contact_torsion_radius: Experimental effective spin radius [m], zero disables.
                Explicit material/footprint assumption: for uniform pressure on a disk
                of radius R, the effective radius is 2*R/3. Does not consume the generic
                material torsion default. Sliding and spin share one Coulomb budget.
                Currently supports dense articulated contacts in CUDA matrix-free,
                immediate/current/interleaved mode without warmstarting,
                hydroelastic contact, contact compliance,
                or debug mode. Velocity post-passes require bilateral preelimination
                disabled; they preserve accumulated impulses and use the unbiased
                angular target. Initially touching groups remain eligible through
                rebound, using the end-gap slop as the initial touching tolerance;
                groups outside this tolerance use the position solve's end-gap
                admission test. Final normal load always bounds the friction budget.
                Radius and selectors are
                construction-only; recreate the solver to change them. With point
                friction, one spin row bounds each coplanar exact-shape-pair group. With
                persistent friction patches (the default ``friction_anchor_beta``), one
                spin row bounds each patch region by the region's pooled normal load
                times the undivided friction coefficient, less the tangent impulses of
                its anchor rows; normal rows stay independent. Contact regularization
                softens the position solve's normal rows; velocity cleanup is
                unregularized. Spin uses the final solved normal load without
                applying a second regularization factor. Capacity
                exhaustion raises even when optional overflow diagnostics are off. Host
                grouping is diagnostic, not optimized for throughput. This experimental
                parameter may change without the normal deprecation policy.
            contact_torsion_shape_indices: Optional global shape indices selecting
                contacts with at least one selected collider. Supported geometry is
                sphere, box, capsule, cylinder, cone, ellipsoid, plane, and convex mesh.
                Explicitly selecting unsupported geometry raises; without selectors,
                contacts involving unsupported geometry are skipped. None selects all shapes;
                an empty tuple selects none. Mutually exclusive with shape patterns.
                Experimental, with the same compatibility limitations as the radius.
            contact_torsion_shape_patterns: Optional regex patterns, full-matched against
                finalized model.shape_label at construction. For substring matching use
                an explicit pattern such as ".*fingertip.*". Empty tuple selects none;
                unmatched nonempty patterns raise ValueError. Experimental, with the
                same compatibility limitations as the radius.
            contact_torsion_device: Experimental opt-in device-resident torsion preparation.
                False retains the host reference implementation. True preserves the
                grouping and friction law while using preallocated GPU buffers.
                Eager stepping checks errors synchronously. For CUDA graph capture,
                first call :meth:`prepare_contact_torsion_capture` outside capture,
                then call :meth:`validate_contact_torsion` after every replay batch
                before consuming results. Errors latch permanently; recreate the
                solver rather than continuing an invalid graph. No effect when the
                torsion radius is zero. The host reference cannot be captured.
                Grouping currently runs serially within each world with worst-case
                quadratic contact-count cost. Many small worlds can benefit, but
                a single contact-dense world may be slower than host preparation;
                CUDA graphs do not remove that device-side work. Benchmark the
                intended per-world contact distribution before enabling.
            contact_compliance: Experimental opt-in implicit unilateral contact material response.
                Positive ``Contacts.rigid_contact_stiffness`` [N/m] replaces the hard normal law;
                zero stiffness remains hard. Uses exported damping [N s/m] (zero stays zero) and
                applies exported friction quadrature weighting once to the existing pair friction.
                Requires CUDA matrix-free/immediate/interleaved solving, full position friction,
                no warm start, restitution, global contact regularization, debug, shared normal
                anchors, persistent-patch friction, or friction-anchor reduction. Intentional
                allocator exclusions are counted in ``compliance_skipped_contact_count``;
                contact/row capacity loss remains an error even with warnings disabled.
                With omitted ``friction_anchor_beta``, enabling compliance selects point friction
                and warns; an explicit positive patch gain is rejected at construction.
                CUDA graph capture is rejected. This
                host-synchronizing experimental implementation is not a performance path and may
                change without the normal deprecation period. Defaults to False.
                Enabling this option with zero stiffness preserves the hard-contact law, not
                the runtime cost of the default-OFF path.

                .. experimental::

                    The ``contact_compliance=True`` material response and its supported combinations
                    may change without prior notice.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_interval (int, optional): How often to update the mass matrix (every n-th time the
                :meth:`step` function gets called). The cadence flag is evaluated on the host each step, so under
                CUDA graph capture the 0/1 pattern is baked at capture time: with K steps captured per graph,
                replays repeat the captured K-step pattern, so only intervals whose pattern fits the capture are
                honored (a 2-step graph captured at step 0 bakes [1, 0]; interval=2 is exact while interval>=3
                silently degrades to an effective interval of 2). Intervals must divide or equal the captured
                step pattern to behave as requested. Per-articulation limit-count changes still trigger a
                device-side mass refresh on skipped steps. Defaults to 1.
            enable_contact_friction (bool, optional): Enables Coulomb friction contacts inside the PGS solve. Defaults to True.
            contact_friction_gap_threshold (float, optional): Only build friction rows for contacts with
                ``phi <= contact_friction_gap_threshold``. This mirrors PhysX's separate friction-anchor
                distance threshold while leaving speculative normal rows intact. Defaults to ``inf`` to
                preserve legacy FeatherPGS behavior.
            contact_friction_position_iterations (int, optional): Number of biased PGS position
                iterations that solve contact-friction rows. ``-1`` preserves legacy FeatherPGS
                behavior by solving friction in every position iteration; ``0`` skips position-pass
                friction entirely; ``N > 0`` solves friction only in the final ``N`` position
                iterations. Velocity-only post iterations always solve friction every iteration,
                matching PhysX PGS's documented default of friction in the last three position
                iterations and all velocity iterations. Defaults to ``-1``.
            contact_friction_shared_anchor (bool, optional): If true, friction rows use the midpoint
                between the two contact witness points as the Jacobian point on both bodies. Normal
                rows keep their original witness points. This avoids a tangential force couple when
                the witnesses are separated along the contact normal. Applies to velocity-only point
                friction (``friction_anchor_beta=0``); patch friction warns and uses its persistent
                material anchors instead. Defaults to False.
            contact_friction_anchor_limit (int, optional): Deprecated compatibility argument.
                The old contact-index approximation has been removed. A positive value warns
                and has no effect. Patch friction is enabled by default; use
                ``friction_anchor_beta=0`` to explicitly select velocity-only point friction.
            contact_friction_articulation_pairs_only (bool, optional): Apply
                ``contact_friction_gap_threshold`` only when
                both contact bodies belong to non-free articulations. Contacts involving ground or a
                free rigid body retain the legacy unbounded friction-row behavior. Defaults to False.
            contact_friction_scale (float, optional): Multiplies the effective Coulomb coefficient
                used by generated friction rows. This is a diagnostic hook for matching solver-prep
                semantics such as PhysX's per-friction-anchor scaling; it does not affect normal
                contact rows. Defaults to 1.0.
            friction_anchor_beta (float | None, optional): Set the positional correction strength
                for persistent patch friction. ``None`` enables it at 0.2 for the default solver;
                the experimental contact material law instead retains velocity-only point friction and warns.
                Zero explicitly selects velocity-only point friction. Positive values
                group compatible contacts on a body pair into regions,
                select up to two friction locations per region, and share the total
                normal impulse equally between those anchors. Normal contacts are preserved.
                Twisting resistance comes only from the separation between the anchors;
                a single-anchor region has no independent torsional stiction constraint
                unless ``contact_torsion_radius`` adds one bounded spin row per region.
                For uniform pad-friction randomization, sample one coefficient per pad
                and assign it to all constituent convex shapes. Different coefficients
                define separate regions and are not pooled across material boundaries.
                Anchor history is independent of collision contact matching, including across
                convex shapes on the same body. Geometry-scaled correlation and detected
                sliding determine when anchors are replaced; they do not require user tuning.
                Friction locations follow the current contact region while tangential displacement
                and impulses persist. The displacement follows pose increments at the current row points,
                including externally imposed motion and position-only solver passes; it does not tether rolling objects to
                an old footprint. This applies equally to analytic shapes, convex hulls, and meshes.
                After teleporting a body, call ``reset(state)`` (optionally with ``world_mask``)
                to discard history, since jumps within the correlation limits contribute to
                displacement correction instead of automatically re-anchoring.
                The tangent RHS includes ``friction_anchor_beta * displacement / dt``. Unloaded
                regions discard history. Regions with no friction rows for a step (gap filters or
                row capacity) retain supported history. Patch reduction approximates the friction
                wrench with two locations; faceted wheels still experience facet-impact losses.
                Patch locations define the friction row points, so ``contact_shared_anchor`` and
                ``contact_friction_shared_anchor`` apply only to normal rows and velocity-only friction.
                Explicit positive gains with an incompatible point algorithm raise an error.
                Each step sorts its patch contact pairs, except a step captured in a CUDA graph
                right after another captured step of the same graph over the same contact stream
                (the same ``contact_generation`` array, which every object wrapping the arrays of
                one :class:`~newton.Contacts` shares), with no write of that stream by Newton in
                between (a collision pass, :meth:`~newton.Contacts.clear`, a coupled-solver
                contact filter or :meth:`~newton.solvers.SolverMuJoCo.update_contacts`): it
                reuses that step's sorted pairs, which saves the sort on every substep that
                shares a collision pass. Code that rewrites contact arrays between two such
                captured steps, including passing other contact arrays with the same
                ``contact_generation`` array, must call :meth:`~newton.Contacts.clear` first,
                which makes the next step sort, or set ``FEATHER_PGS_PATCH_FRAME_REUSE=0``.
                Otherwise only a change of ``contact_generation`` or of the contact count is
                detected: that step is solved without patch friction rows and its anchors are
                released; :meth:`constraint_capacity_status` reports it as ``patch_frames`` and
                :meth:`check_constraint_capacity` raises. A rewrite that keeps both but changes
                which shapes a contact slot pairs, or reorders contacts, is not detected: that
                step groups its contacts by the previous step's sorted pairs. Rewriting only
                contact points, normals or margins is exact, since pairs depend on the shapes
                alone and every step recomputes the rest. ``FEATHER_PGS_PATCH_FRAME_REUSE=0``
                sorts every step.
                Defaults to None.
            contact_speculative_scale (float, optional): Multiplies the positive-gap position RHS
                for normal contact rows on the dense and matrix-free free-rigid paths. A value of
                0.0 removes speculative closing allowance without changing
                penetrating-contact stabilization or the unbiased velocity pass. Must be finite
                and non-negative. Defaults to 1.0.
            contact_gap_gate (float, optional): If positive, skip normal contacts whose
                world-space gap exceeds this distance before allocating any dense or matrix-free
                rows. A value of 0.0 disables the gate and preserves all
                collision-generated contacts. Must be finite and non-negative. Defaults to 0.0.
            same_articulation_contact_gap_gate (float, optional): If positive, skip only
                contacts between two links of the same non-free articulation whose world-space
                gap exceeds this distance. This can retain a long predictive horizon for fast
                free bodies while bounding speculative self-contact row demand. A value of 0.0
                disables the scoped gate. Must be finite and non-negative. Defaults to 0.0.
            articulation_pair_contact_gap_gate (float, optional): If positive, skip contacts
                between two non-free articulated bodies whose world-space gap exceeds this
                distance. Contacts involving ground or a free rigid body retain the full
                predictive horizon. A value of 0.0 disables the scoped gate. Must be finite
                and non-negative. Defaults to 0.0.
            contact_shared_anchor (bool, optional): If true, all contact rows use the midpoint between
                the two witness points as the Jacobian point on both bodies, matching PhysX contact
                prep's single ``contact.point`` lever arm. ``phi`` is still computed from the original
                witness points. With patch friction, only normal rows use this midpoint; a warning
                explains how to select point friction for the original all-row behavior. Defaults to False.
            enable_joint_limits (bool, optional): Enforce joint position limits as unilateral PGS
                constraints. Each active limit side adds one constraint row. Defaults to False.
            enable_bilateral_preelimination (bool, optional): Pre-eliminate the mimic and
                connect rows of each articulation through a small Schur complement so every
                other row (contacts especially) sees the closed-loop effective mass exactly:
                responses become ``Y' = Y - Y_B S^-1 (J_B Y)`` and the predictor velocity is
                projected onto the bilateral constraint manifold once per step, replacing the
                iterative enforcement. Requires ``pgs_mode='matrix_free'`` with
                ``pgs_velocity_iterations == 0`` and at most 8 bilateral rows per
                articulation; unsupported configurations warn and fall back to iterative
                rows. Defaults to False.
            bilateral_preelimination_include_mimics (bool, optional): Include mimic rows
                in the pre-eliminated bilateral block. When False, only connect rows are
                pre-eliminated and mimic rows remain in the iterative sweep. This avoids
                making the Schur block singular when a mimic row is nearly dependent on
                the loop-closure rows. Defaults to True.
            joint_limit_activation_gap (float, optional): Distance from a finite lower or upper
                position limit at which a joint-limit PGS row becomes active. A lower row is
                allocated when ``q <= lower + gap``; an upper row is allocated when
                ``q >= upper - gap``. ``inf`` preserves the historical behavior of allocating all
                finite limit rows every step. Must be non-negative and not NaN. Defaults to ``inf``.
            enable_joint_velocity_limits (bool, optional): Enforce joint velocity limits
                (``model.joint_velocity_limit``) as per-DOF PGS constraints. Mirrors the
                PhysX velocity-limit row: when ``|qdot_i| > qdot_max_i``, a single
                signed-Jacobian row is added that projects ``qdot_i`` back onto the
                bilateral box ``[-qdot_max_i, +qdot_max_i]`` through the articulated-body
                response ``J M^-1 J^T + cfm``, so momentum redistributes correctly across
                the articulation (rather than a naive joint-space clip which breaks
                multi-body Newton's third law). No Baumgarte bias.

                Only supported with ``pgs_mode="matrix_free"``; passing any other
                ``pgs_mode`` together with ``enable_joint_velocity_limits=True`` raises
                :class:`NotImplementedError` at construction. Defaults to False.
            velocity_limit_activation_fraction (float, optional): Proximity gate for velocity-limit
                row allocation. ``0.0`` allocates the lower/upper row pair of every finitely
                limited joint DOF, and the speed row of every finitely limited free-rigid-body
                linear or angular speed, each step. A positive fraction allocates them only when
                the current speed satisfies ``|qd| >= fraction * limit`` (the magnitude of the
                linear or angular velocity for a free body's speed row), skipping rows that would
                be exact no-op clamps. Because the gate samples the pre-solve velocity, a DOF
                crossing the threshold between steps is clamped one step late. Must be non-negative
                and not NaN. Defaults to ``0.0``.
            fuse_joint_velocity_limits (bool, optional): Enforce joint velocity limits of
                *driven* DOFs (``drive_slot[dof] >= 0``, i.e. ``joint_target_ke/kd > 0``) as a
                stateless clamp instead of two dedicated dense rows per DOF. This mirrors
                PhysX's ``PxClamp`` on ``jointV`` (``DyFeatherstoneArticulation.cpp``): if
                ``|jointV| > maxJointVel`` an extra impulse
                ``(clamp(jointV) - jointV) * recipResponse`` is applied — no accumulated
                lambda, no relaxation, no Baumgarte. DOFs with a velocity limit
                but no drive row keep their dedicated fallback rows. The clamp runs at the
                end of every PGS iteration, after the contact rows — the position the
                dedicated velocity-limit rows it replaces own — so the limit keeps the
                last word of each iteration even when the sweep is under-converged. That
                is the ordering PhysX documents for gripping scenarios
                (``PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST`` /
                ``VelLimit::eAFTER_STATIC_CONSTRAINTS``); clamping inside the drive-row
                visit instead would let contact impulses landing later in the iteration
                push driven DOFs past their limits with nothing left to re-clamp them.
                The fused clamp also skips driven DOFs in the PhysX-style pre-solve
                velocity prescale ratio. Requires ``model.joint_velocity_limit`` to be
                allocated; when it is absent the flag is inert (the dedicated-row path
                skips allocation without that array too, so behavior matches). The flag is
                also inert when ``velocity_limit_activation_fraction`` is ``inf``: that value
                is the explicit never-activate kill-switch for velocity limits, and the fused
                clamp rides the drive-row visit rather than the gated velocity-limit phase,
                so it must not re-enforce limits the fraction disabled. The fused clamp only applies to the
                ``pgs_mode="matrix_free"`` + ``enable_joint_velocity_limits=True`` +
                ``drive_mode="physx_pgs"`` formulation; in any other configuration the flag
                is inert and velocity limits keep their dedicated rows (no error — the flag
                is an engage-where-applicable request, not a demand, which lets it default
                to True without constraining unrelated solver modes). The clamp pass is
                independent of the drive-row visit, so it stays active during velocity-only
                iterations under ``pgs_velocity_drive_mode="freeze"``, exactly like the
                dedicated rows it replaces. Defaults to True.
            velocity_limit_law (str, optional): How the matrix-free sweeps project the dedicated
                joint and free-body velocity-limit rows, which keep the last word at the end of
                every iteration either way. A free body's row bounds the magnitude of its linear or
                angular velocity (a speed row: residual ``limit - |v|``, correction along ``v``).
                ``"stateless"`` is PhysX's clamp: each visit applies
                only the impulse that removes the current overshoot and stores that correction,
                so an impulse that later rows made unnecessary is never released and the
                converged velocity depends on the row order. ``"accumulated"`` treats the row like
                a contact normal or a position limit, with a non-negative accumulated impulse
                ``max(old - r / d, 0)``; the sweeps then converge to the complementarity solution,
                independent of the row order, which the parallel matrix-free tiers also solve. The
                ``"physx_grasp"`` schedule always keeps PhysX's stateless clamp, and the fused
                drive clamp of ``fuse_joint_velocity_limits`` is unaffected. Split mode, which has
                no joint velocity-limit rows, applies neither law: where its free-body sweep has
                velocity-limit rows it keeps the stateless clamp that ``get_path_selection()``
                reports under ``routes.velocity_limits.free_body_sweep_law``. The environment variable
                ``FEATHER_PGS_VELOCITY_LIMIT_LAW`` overrides the argument.
                ``get_path_selection()`` reports the law the serial sweeps apply under
                ``routes.velocity_limits.law``, and the parallel tiers' law, accumulated under either
                argument, under ``routes.sweep.tiers.velocity_limit_law``. Experimental. Defaults to
                ``"stateless"``.
            contact_update (str, optional): How a Gauss-Seidel sweep updates a contact. ``"sequential"``
                steps a contact's normal row against the friction of its previous update, then solves its
                friction pair at the held normal load. ``"unit"`` solves each contact unit exactly against its
                own Delassus block, with every other impulse held: a point contact's normal and friction
                together (take-off, else stick, else Daviet's single-unknown slide root), a patch region's ring
                normals and anchor pairs as one unit, a ring of more than 8 normals or 2 anchors as anchor units
                with the other members' load held. A unit whose load change keeps reversing is relaxed toward
                its exact solution, which keeps the fixed points of the exact sweep. The contact law is that of
                ``"sequential"``; ``"unit"`` solves a contact's normal and friction together instead of stepping
                its normal row against stale friction, which cycles with period two for a sliding contact whose
                friction lifts its normal (zero normal impulse on an approaching contact at the end of a solve).
                Every solution of the whole problem is a fixed point of the unit sweeps: where a unit's own
                problem has several solutions (a sliding contact whose friction presses its normal down), the
                unit keeps the one it holds, and an anchor whose take-off fails where only friction that presses
                its normal carries load takes the press slide of the larger load. A slide the unit holds whose
                load answers the contact's approach with the wrong sign, or many times its normal row's own
                response, takes its normal with that friction held, then the friction, which keeps it and
                stabilizes sweeps that the coupling of units would drive away from it; that step never unloads
                its contact. A unit's exact solution does not use ``pgs_omega``; rows outside units keep the
                sequential update. Only the loop sweeps implement ``"unit"``: the dense ``pgs_kernel="loop"``
                sweep of split mode on any device and the split free-body sweep on CPU. Every other sweep keeps
                ``"sequential"``; ``get_path_selection()`` records which sweep runs which law under
                ``routes.sweep.contact_update``, and construction warns when a requested ``"unit"`` reaches only
                some sweeps. ``FEATHER_PGS_CONTACT_UPDATE`` overrides the argument. Experimental. Defaults to
                ``"sequential"``.
            mf_gs_incremental_rows (int, optional): Row budget of the incremental-residual
                matrix-free Gauss-Seidel kernel. Worlds with only dense articulated rows and at
                most this many rows solve through a shared-memory response block instead of
                re-evaluating ``J v`` per row. Capped at 64. ``0`` disables the kernel. Defaults to 32.
            mf_gs_parallel_rows (int, optional): Row budget of the scaled parallel projection sweep. Worlds with
                only dense articulated rows and at most this many rows are solved by the parallel kernel (all rows
                updated at once with per-row step scaling and Nesterov momentum) instead of the row-serial kernels.
                ``0`` disables it. Requires the same conditions as ``mf_gs_incremental_rows`` and takes precedence.
                Consider small contact sets, such as an arm against fixed geometry; worlds with mixed/free-body
                rows or excess rows retain serial solving. The effective limit is capped at 96 for response-block
                sweeps or 128 for matrix-free sweeps, and rounded down to a multiple of four. Each tangent pair
                takes one isotropic step, the serial pair solve's friction law, so a sliding pair converges to
                maximal dissipation; under persistent patch friction it projects on its patch's pooled normal
                load, and ``FEATHER_PGS_PATCH_PARALLEL_ROWS=0`` keeps patch friction on the serial kernels. Each
                tier's sweep budget follows ``mf_gs_parallel_sweeps``. The tiers accumulate joint velocity-limit rows
                (``routes.sweep.tiers.velocity_limit_law``). The worlds they decline keep the serial sweep and its
                ``velocity_limit_law``, ``"stateless"`` by default, so tier and serial worlds share one fixed point
                under ``velocity_limit_law="accumulated"`` or without joint velocity limits. The tiers solve the
                unregularized contact law, so regularized contacts (``pgs_contact_regularization``) keep the serial
                kernels.
            mf_gs_parallel_sweeps (int, optional): Minimum sweep budget of each parallel tier
                (``mf_gs_parallel_rows``). Jacobi sweeps need more passes than the serial sweep's Gauss-Seidel
                iterations to reach its accuracy, the more so the more rows couple through one body, so every tier,
                under either friction law, sweeps ``ceil(7/2 * pgs_iterations * sqrt(R / 32))`` times for its row
                bound ``R``, at least ``mf_gs_parallel_sweeps`` and at most 128 unless ``mf_gs_parallel_sweeps`` is
                larger, from ``pgs_iterations`` at construction: the serial sweep's iteration count sets the
                accuracy to match. The factor is fit, by interpolating each world's error between measured sweep
                caps, so that each evaluated row bucket of contact-rich point- and patch-friction workloads reaches
                the serial sweep's accuracy. A world stops earlier under the sweeps' exit law, once its rows'
                changes are at the rounding floor.
                ``get_path_selection()`` reports each tier's budget and its rule under ``routes.sweep.tiers``.
                Defaults to 24.
            mf_gs_parallel_linked_sweeps (int | Literal["auto"] | None, optional): Deprecated: the tiers that serve
                persistent patch friction (linked tiers) follow the budget rule of every tier, the default
                ``"auto"``. A count, or ``None`` for exactly ``mf_gs_parallel_sweeps``, still sets the linked tiers'
                budget, with a :class:`DeprecationWarning`, and so does the environment variable
                ``FEATHER_PGS_PATCH_PARALLEL_SWEEPS``; omit both. Defaults to ``"auto"``.
            mf_gs_parallel_nesterov (bool, optional): Apply Nesterov momentum to the parallel sweeps.
            mf_gs_parallel_matrix_free (bool, optional): Parallel sweep without the response block (per sweep
                ``Y^T y`` and ``J . dv`` from staged rows; row-sum-bound step scale). With ``m`` rows and ``D`` DOFs,
                the block costs ``O(m^2 D)`` to build plus ``O(m^2)`` per sweep; matrix-free costs ``O(m D)`` per
                sweep. Consider matrix-free when block construction dominates, such as contact-rich hands;
                small row sets or many sweeps may favor the block. Compare total cost at the required convergence.
            mf_gs_parallel_linked_velocity_limit_pass (bool, optional): Give the joint velocity limits the last word
                in the parallel tiers that serve persistent patch friction, as the serial sweeps' end-of-iteration
                pass gives it to them: after the final sweep one warp pushes every violated velocity-limit row back
                onto its limit, in row order. It changes nothing once a tier converges and bounds the limit overshoot
                of a tier its sweep budget stops early. Applies only with ``enable_joint_velocity_limits`` and
                linked tiers. Experimental. Defaults to False.
            pgs_iterations (int, optional): Number of Gauss-Seidel iterations to apply per frame. Defaults to 12.
            pgs_velocity_iterations (int, optional): Number of matrix-free iterations using a velocity-only RHS
                after integrating positions with the biased PGS result. This mirrors the PhysX-style split where
                geometric bias corrects positions first, then final velocity iterations run without Baumgarte bias.
                Positive-gap rows whose linearized position trajectory remains separated by more than a 1e-6 m
                numerical slop keep their speculative closing allowance; rows within the slop use the unbiased
                contact law. Restitution does not change this value: zero means no velocity-only iteration, while a
                positive value explicitly requests that many refinement iterations for coupled contacts. Only
                supported with ``pgs_mode="matrix_free"``. Defaults to 0.
            pgs_beta (float, optional): ERP style position correction factor for contact, joint-limit,
                mimic and connect rows. Defaults to 0.2.
            pgs_cfm (float, optional): Compliance/regularization added to the Delassus diagonal. Defaults to 1.0e-6.
            pgs_omega (float, optional): Successive over-relaxation factor for the PGS sweep. Defaults to 1.0.
            pgs_solve_readout (str, optional): Opt-in counters of the native Gauss-Seidel sweeps.
                ``"sweeps"`` counts, per sweep kernel and world, how many sweeps each solve executed
                and whether it stopped before its budget, the velocity pass of
                ``pgs_velocity_iterations`` apart from the position solve; ``"trace"`` adds, per
                solve, the sweep at which each absolute velocity tolerance from 1e-7 to 1e-1 [m/s or
                rad/s] would have stopped it, its exit reason (no change, changes within the exit law's
                rounding floor, or the budget), and the largest velocity-scaled change of its final sweep
                with that row's kind. ``get_path_selection()`` summarizes them under
                ``measurements["solve_readout"]``, which synchronizes the device once. Neither
                readout changes a computed value. Every sweep counts (``routes.sweep.readout``). The
                matrix-free sweeps enter no world without rows (such as one the resolved-world
                classifier left without rows); the readout counts those solves as ``skipped``.
                Defaults to ``"off"``.
            pgs_contact_regularization (float, optional): Dimensionless regularizer ``g`` of contact
                rows on every route (matrix-free and dense). Each position iteration moves a
                row's impulse toward the hard solution with weight ``1/(1+g)`` and toward zero with
                weight ``g/(1+g)``, which is the same update as a damped contact spring integrated
                implicitly. It is a numerical stabilizer, not a material model: it makes statically
                indeterminate normal-force splits unique, damps the Gauss-Seidel sweep enough to hold
                stacks that the exact rigid law drops at the same iteration count, and costs a resting
                sag of ``g * a * dt^2 / pgs_beta`` per loaded row (0.3 mm at 60 Hz for a body under
                gravity at ``g = 0.02``). Positive-gap speculative rows and rows whose rebound
                target fires are solved rigid, and the velocity-only pass ignores ``g``. ``0`` is the
                exact rigid law. Values above ``1e6`` are rejected because they are not
                numerically useful in the float32 solve. Defaults to 0.0.
            pgs_velocity_drive_mode (str, optional): Drive-row treatment during velocity-only post-pass
                iterations. ``"freeze"`` keeps PhysX-style drive impulses from the biased position
                solve and lets only contacts, friction, and limits clean up velocity residuals;
                ``"active"`` continues updating drive rows during the velocity-only pass.
                Defaults to ``"freeze"``.
            dense_max_constraints (int, optional): Maximum number of dense (articulation) contact constraint
                rows stored per world. Free rigid body contacts are stored separately, bounded by
                mf_max_constraints. Defaults to 32.
            pgs_warmstart (bool, optional): Re-use dense and matrix-free contact
                impulses from the previous frame. Contact and friction rows always carry by
                contact identity through the collision pipeline's ``rigid_contact_match_index``;
                non-contact dense rows cold-start because their runtime allocation does not
                provide an identity contract. A non-``None`` Contacts buffer therefore requires
                contact matching. Body-pair reduction is supported with
                ``contact_matching="latest"``. Carried friction is transported into
                the current tangent frame and clamped to the current friction cone.
                Defaults to False.
            pgs_warmstart_decay (float, optional): Finite non-negative scale applied to
                contact impulses carried from the previous frame. Defaults to 1.0.
            warn_constraint_overflow (bool, optional): Emit a device-side warning the first time each dense
                or matrix-free row family exceeds its configured per-world capacity. The warning
                reports the world, requested rows, row limit, and dropped contact/friction rows without a host
                synchronization, so it remains compatible with CUDA graph capture. Defaults to True.
            mf_warmstart (bool, optional): Legacy compatibility alias for ``pgs_warmstart``
                (this option was historically matrix-free-only). New callers should use
                ``pgs_warmstart``. Defaults to False.
            mf_warmstart_decay (float, optional): Legacy decay alias used when
                ``mf_warmstart`` enables the unified mode. Defaults to 1.0.
            pgs_mode (str, optional): ``"split"`` forms a dense Delassus matrix for articulated rows and
                uses matrix-free PGS for free-body contacts; it is the only CPU mode. ``"matrix_free"``
                avoids that global matrix; CPU construction raises :class:`NotImplementedError`.
                For ordinary dense dynamics/immediate response, with ``m`` rows and ``D`` response DOFs,
                split adds ``O(m²D)`` assembly, ``O(m²)`` storage and ``O(m²)`` work per sweep;
                unblocked matrix-free sweeps cost ``O(mD)``. Both retain ``O(mD)`` J/Y and ``O(D²)`` dynamics
                storage; optional matrix-free response-block kernels still build small local blocks.
                Try matrix-free when assembly dominates (e.g. contact-rich hands); compact row sets with
                many sweeps may favor split. Compare total cost and convergence at equal iteration
                allowances. Defaults to ``"split"``.
            articulated_contact_response (str, optional): Contact response for articulated rows.
                ``"immediate"`` solves each contact row against the articulation's ``H^-1 J^T``
                response. Defaults to ``"immediate"``.

                .. deprecated:: 1.6
                    ``"propagation"``, ``"propagation-fused"`` and ``"propagation-colored"``
                    have no effect: they emit a :class:`DeprecationWarning`, run the
                    ``"immediate"`` response and will be removed with this parameter.
                    Omit the parameter.
            propagation_same_articulation_rows (bool, optional): No effect.

                .. deprecated:: 1.6
                    It configured the deprecated propagation responses. ``True`` emits a
                    :class:`DeprecationWarning`; omit the parameter.
            propagation_cached_response (bool, optional): No effect.

                .. deprecated:: 1.6
                    It configured the deprecated propagation responses. ``False`` emits a
                    :class:`DeprecationWarning`; omit the parameter.
            propagation_cached_response_max_bodies (int, optional): No effect.

                .. deprecated:: 1.6
                    It configured the deprecated propagation responses. A value other than
                    8 emits a :class:`DeprecationWarning`; omit the parameter.
            pgs_schedule (str, optional): Matrix-free row ordering. ``"interleaved"`` preserves the
                legacy per-iteration dense+matrix-free sweep. ``"contact_then_internal"`` runs all
                contact/friction sweeps first, then all internal articulation sweeps (drive, joint
                limits, joint velocity limits). ``"physx_grasp"`` approximates PhysX's two-pass
                grabbing order by solving drive/position-limit rows first, then contacts/friction,
                then joint velocity limits. Defaults to ``"interleaved"``.
            mf_max_constraints (int, optional): Maximum number of matrix-free constraints per world.
                Packed metadata in matrix-free mode supports at most 65535 rows. Defaults to 512.
            pgs_kernel: Dense PGS implementation. ``auto`` uses ``loop`` on CPU and ``tiled_row`` on CUDA.
                Explicit ``loop`` and ``tiled_row`` select the split-mode kernel and require a supported
                device. ``tiled_row`` stages the packed Delassus triangle in shared memory while it fits the
                device's shared memory per block and reads the Delassus rows from global memory past that
                (``routes.sweep.dense.staging``). Matrix-free mode owns its sweep dispatcher: an explicit
                choice there warns and is ignored. ``tiled_contact`` and ``streaming`` are deprecated: they
                warn, have no effect and select the ``auto`` kernel.
            pgs_chunk_size (int, optional): Deprecated compatibility argument. It tuned the removed
                streaming kernel; a value other than ``None`` warns and has no effect. Defaults to None.
            use_parallel_streams (bool, optional): Overlap independent work on separate CUDA streams: the
                size groups, and the drive preparation beside inverse dynamics. It never selects the
                dynamics evaluator. Defaults to True.
            drive_mode (str, optional): Joint target drive implementation.
                ``"augmented"`` keeps the legacy FeatherPGS implicit-PD path
                that folds drive stiffness into the articulated mass matrix.
                ``"physx_pgs"`` creates one dense PGS constraint row per
                driven scalar DOF and updates its drive impulse with PhysX's
                force-drive PGS formula each Gauss-Seidel iteration. The
                PhysX row mode is currently supported only with
                ``pgs_mode="matrix_free"``. Defaults to ``"augmented"``.
            tile_threads (int, optional): Threads per block for the tiled
                Cholesky / triangular-solve / H^-1 J^T kernels. Tile primitives may
                change floating-point reduction order with the block size, so non-default
                values need numerical validation, not bit-identical comparison.
                Must be one of {32, 64, 128, 256}. Defaults to 64.
            enable_restitution: Whether rigid contacts apply their authored restitution coefficients. Defaults to
                ``True`` to preserve FeatherPGS behavior before this option was exposed.
            restitution_velocity_threshold (float, optional): Minimum magnitude of the frozen pre-contact
                relative normal velocity required to apply restitution. The contact must also be at the surface or
                be predicted to reach it during the timestep. Slower contacts retain the existing speculative or
                Baumgarte contact law, preventing resting contacts from repeatedly bouncing under small
                accelerations.
                Restitution coefficients are averaged across the two shapes after clamping finite values to
                ``[0, 1]``; non-finite values are treated as zero. Set the threshold to zero to apply restitution
                to every closing impact. Defaults to 0.5 m/s.
        """
        # Every option as requested, before any default resolution; get_path_selection() reports it.
        requested_options = dict(locals())
        if contact_compliance:
            _contact_compliance.validate_configuration(
                model,
                {
                    "pgs_mode": pgs_mode,
                    "pgs_schedule": pgs_schedule,
                    "pgs_iterations": pgs_iterations,
                    "pgs_velocity_iterations": pgs_velocity_iterations,
                    "pgs_warmstart": pgs_warmstart,
                    "mf_warmstart": mf_warmstart,
                    "enable_restitution": enable_restitution,
                    "pgs_contact_regularization": pgs_contact_regularization,
                    "pgs_debug": pgs_debug,
                    "contact_friction_position_iterations": contact_friction_position_iterations,
                    "contact_friction_anchor_limit": contact_friction_anchor_limit,
                    "contact_friction_shared_anchor": contact_friction_shared_anchor,
                    "contact_shared_anchor": contact_shared_anchor,
                },
            )
        self.contact_compliance = bool(contact_compliance)
        self.compliance_contact_count = 0
        self.compliance_skipped_contact_count = 0
        self._compliant_contacts = None
        self._compliant_prepared = False
        # Read the FEATHER_PGS_* environment once; undeclared names are recorded, not rejected.
        self._environment_record = _environment_flags.record()
        super().__init__(model)
        # Every scalar option coerced and checked once (configuration.OPTION_RULES).
        options = check_options(requested_options)

        if pgs_kernel in ("tiled_contact", "streaming"):
            warnings.warn(
                f"pgs_kernel={pgs_kernel!r} is deprecated and has no effect, and will be removed in a future "
                "release; the solver selects the 'auto' kernel. Omit pgs_kernel, and pass friction_anchor_beta=0 "
                "for velocity-only point friction.",
                DeprecationWarning,
                stacklevel=_CONSTRUCTOR_CALLER,
            )
            pgs_kernel = "auto"
        if pgs_chunk_size is not None:
            warnings.warn(
                "pgs_chunk_size is deprecated and has no effect, and will be removed in a future release; "
                "it tuned the removed streaming kernel. Omit it.",
                DeprecationWarning,
                stacklevel=_CONSTRUCTOR_CALLER,
            )
        self._requested_kernel_choices = {
            "cholesky_kernel": cholesky_kernel,
            "trisolve_kernel": trisolve_kernel,
            "hinv_jt_kernel": hinv_jt_kernel,
            "delassus_kernel": delassus_kernel,
            "pgs_kernel": pgs_kernel,
            "sparse_mass_matrix": sparse_mass_matrix,
            "cached_dynamics": cached_dynamics,
        }
        kernel_choices, self._explicit_dense_kernels, self._kernel_choice_reasons, ignored_choices = (
            resolve_kernel_choices(
                self._requested_kernel_choices,
                self._kernel_overrides,
                is_cpu=model.device.is_cpu,
                pgs_mode=pgs_mode,
            )
        )
        cholesky_kernel = kernel_choices["cholesky_kernel"]
        trisolve_kernel = kernel_choices["trisolve_kernel"]
        hinv_jt_kernel = kernel_choices["hinv_jt_kernel"]
        delassus_kernel = kernel_choices["delassus_kernel"]
        pgs_kernel = kernel_choices["pgs_kernel"]
        self.sparse_mass_matrix = sparse_mass_matrix
        self.cached_dynamics = cached_dynamics

        self.angular_damping = angular_damping
        self.update_mass_matrix_interval = update_mass_matrix_interval
        self.enable_contact_friction = enable_contact_friction
        self.contact_friction_gap_threshold = contact_friction_gap_threshold
        self.contact_friction_position_iterations = options["contact_friction_position_iterations"]
        self.contact_friction_shared_anchor = bool(contact_friction_shared_anchor)
        self.contact_friction_articulation_pairs_only = bool(contact_friction_articulation_pairs_only)
        self.contact_friction_scale = options["contact_friction_scale"]
        # How friction_anchor_beta was resolved, for get_path_selection().
        self._friction_anchor_beta_source = "argument"
        if friction_anchor_beta is None:
            # An explicit point algorithm remains a valid way to select point
            # friction. The ordinary constructor enables persistent patches.
            if contact_compliance:
                friction_anchor_beta = 0.0
                self._friction_anchor_beta_source = "default_compliance"
                warnings.warn(
                    "The selected contact material law uses velocity-only point friction; "
                    "contact_compliance does not support persistent friction patches. "
                    "Set friction_anchor_beta=0 explicitly to retain this law without the warning.",
                    UserWarning,
                    stacklevel=_CONSTRUCTOR_CALLER,
                )
            else:
                friction_anchor_beta = 0.2
                self._friction_anchor_beta_source = "default"
        for message in ignored_choices.values():
            warnings.warn(message, UserWarning, stacklevel=_CONSTRUCTOR_CALLER)
        self.friction_anchor_beta = float(friction_anchor_beta)
        if not np.isfinite(self.friction_anchor_beta) or self.friction_anchor_beta < 0.0:
            raise ValueError("friction_anchor_beta must be finite and non-negative")
        if options["contact_friction_anchor_limit"] > 0:
            warnings.warn(
                "contact_friction_anchor_limit is deprecated and ignored. Patch friction is enabled by default; "
                "use friction_anchor_beta to adjust it or explicitly set zero to disable it.",
                DeprecationWarning,
                stacklevel=_CONSTRUCTOR_CALLER,
            )
        self._friction_anchors_enabled = self.friction_anchor_beta > 0.0
        if self.contact_compliance and self._friction_anchors_enabled:
            raise ValueError("contact_compliance is not validated with friction_anchor_beta > 0")
        if self._friction_anchors_enabled and (contact_shared_anchor or contact_friction_shared_anchor):
            warnings.warn(
                "Patch friction selects its own friction locations and carries tangential displacement. "
                "contact_shared_anchor still applies to normal rows; set friction_anchor_beta=0 "
                "to apply shared-anchor flags to point friction rows as well.",
                UserWarning,
                stacklevel=_CONSTRUCTOR_CALLER,
            )
        self.contact_speculative_scale = options["contact_speculative_scale"]
        self.contact_gap_gate = options["contact_gap_gate"]
        self.same_articulation_contact_gap_gate = options["same_articulation_contact_gap_gate"]
        self.articulation_pair_contact_gap_gate = options["articulation_pair_contact_gap_gate"]
        self.contact_shared_anchor = bool(contact_shared_anchor)
        # The deprecated input must not reach legacy point-row writers or scale their friction.
        self.contact_friction_anchor_limit = 0
        self.enable_joint_limits = enable_joint_limits
        self.enable_bilateral_preelimination = bool(enable_bilateral_preelimination)
        self.bilateral_preelimination_include_mimics = bool(bilateral_preelimination_include_mimics)
        self.joint_limit_activation_gap = options["joint_limit_activation_gap"]
        self.enable_joint_velocity_limits = enable_joint_velocity_limits
        self.velocity_limit_activation_fraction = options["velocity_limit_activation_fraction"]
        self.pgs_iterations = pgs_iterations
        self.pgs_beta = pgs_beta
        self.pgs_contact_regularization = options["pgs_contact_regularization"]
        if self.pgs_contact_regularization > _MAX_CONTACT_REGULARIZATION:
            raise ValueError(
                f"pgs_contact_regularization must be at most {_MAX_CONTACT_REGULARIZATION:g}; "
                "larger values are not numerically useful in the float32 solve"
            )
        # Kernels consume the float32 weight w = 1/(1+g). Values too small to
        # change w are the exact hard update and stay on the zero-cost path.
        self._contact_w = float(np.float32(1.0 / (1.0 + self.pgs_contact_regularization)))
        self._regularization_enabled = self._contact_w < 1.0
        self.pgs_cfm = pgs_cfm
        self.pgs_omega = pgs_omega
        self.pgs_solve_readout = pgs_solve_readout
        # Every sweep compiles the exit law and the policy's readout.
        self._solve_policy = SolvePolicy(readout=pgs_solve_readout)
        self._solve_readout = SolveReadoutRecord(self._solve_policy)
        self.pgs_velocity_iterations = max(int(pgs_velocity_iterations), 0)
        self.enable_restitution = bool(enable_restitution)
        self.restitution_velocity_threshold = options["restitution_velocity_threshold"]
        self._effective_restitution_velocity_threshold = (
            self.restitution_velocity_threshold if self.enable_restitution else np.finfo(np.float32).max
        )
        self.pgs_velocity_drive_mode = pgs_velocity_drive_mode
        dense_max_constraints = options["dense_max_constraints"]
        self.warn_constraint_overflow = bool(warn_constraint_overflow)
        if articulated_contact_response in ("propagation", "propagation-fused", "propagation-colored"):
            warnings.warn(
                f"articulated_contact_response={articulated_contact_response!r} is deprecated and has no effect: "
                "the 'immediate' response runs, and the propagation responses will be removed. "
                "Omit articulated_contact_response.",
                DeprecationWarning,
                stacklevel=_CONSTRUCTOR_CALLER,
            )
            articulated_contact_response = "immediate"
        elif articulated_contact_response != "immediate":
            raise ValueError(f"articulated_contact_response must be 'immediate', got {articulated_contact_response!r}")
        self.articulated_contact_response = articulated_contact_response
        for name, ignored in (
            ("propagation_same_articulation_rows", bool(propagation_same_articulation_rows)),
            ("propagation_cached_response", not propagation_cached_response),
            ("propagation_cached_response_max_bodies", propagation_cached_response_max_bodies != 8),
        ):
            if ignored:
                warnings.warn(
                    f"{name} is deprecated and has no effect: it configured the propagation contact responses, "
                    "which will be removed. Omit it.",
                    DeprecationWarning,
                    stacklevel=_CONSTRUCTOR_CALLER,
                )
        # ``mf_warmstart`` and its environment switch predate the unified
        # contact-warm-start contract. Keep them as compatibility aliases, but
        # map them onto the one all-route mode rather than retaining a second
        # MF-only behavior.
        _env_ws = os.getenv("IL_NEWTON_FPGS_MF_WARMSTART", "0").lower() in {"1", "true", "yes", "on"}
        legacy_mf_warmstart = bool(mf_warmstart) or _env_ws
        try:
            legacy_mf_decay = float(os.getenv("IL_NEWTON_FPGS_MF_WARMSTART_DECAY", str(mf_warmstart_decay)))
        except (TypeError, ValueError):
            legacy_mf_decay = float(mf_warmstart_decay)
        if not math.isfinite(legacy_mf_decay) or legacy_mf_decay < 0.0:
            raise ValueError(f"mf_warmstart_decay must be finite and non-negative, got {legacy_mf_decay!r}")
        # Effective legacy alias values (argument or IL_NEWTON_FPGS_* environment), for get_path_selection().
        self._legacy_mf_warmstart = legacy_mf_warmstart
        self._legacy_mf_warmstart_decay = legacy_mf_decay

        requested_pgs_decay = options["pgs_warmstart_decay"]
        self.pgs_warmstart = bool(pgs_warmstart) or legacy_mf_warmstart
        self.pgs_warmstart_decay = requested_pgs_decay if pgs_warmstart or not legacy_mf_warmstart else legacy_mf_decay
        # FEATHER_PGS_VELOCITY_LIMIT_LAW overrides the argument, so recipes select the law without code changes.
        check_velocity_limit_law(velocity_limit_law)
        environment_law = os.environ.get("FEATHER_PGS_VELOCITY_LIMIT_LAW")
        self.velocity_limit_law = (
            check_velocity_limit_law(environment_law, source="FEATHER_PGS_VELOCITY_LIMIT_LAW")
            if environment_law
            else velocity_limit_law
        )
        # FEATHER_PGS_CONTACT_UPDATE overrides the argument, so recipes select the contact update without code changes.
        check_contact_update(contact_update)
        environment_update = os.environ.get("FEATHER_PGS_CONTACT_UPDATE")
        self.contact_update = (
            check_contact_update(environment_update, source="FEATHER_PGS_CONTACT_UPDATE")
            if environment_update
            else contact_update
        )
        # One budget rule for every parallel tier. The linked tiers' own budget and its
        # FEATHER_PGS_PATCH_PARALLEL_SWEEPS override are deprecated: they still set the linked tiers' budget.
        requested_linked_sweeps = _linked_sweeps_request(mf_gs_parallel_linked_sweeps, "mf_gs_parallel_linked_sweeps")
        environment_linked_sweeps = os.environ.get("FEATHER_PGS_PATCH_PARALLEL_SWEEPS") or None
        self.mf_gs_parallel_linked_sweeps = (
            _linked_sweeps_request(environment_linked_sweeps, "FEATHER_PGS_PATCH_PARALLEL_SWEEPS")
            if environment_linked_sweeps
            else requested_linked_sweeps
        )
        for name, given in (
            ("mf_gs_parallel_linked_sweeps", requested_linked_sweeps != "auto"),
            ("FEATHER_PGS_PATCH_PARALLEL_SWEEPS", environment_linked_sweeps is not None),
        ):
            if given:
                warnings.warn(
                    f"{name} is deprecated and will be removed in a future release: every parallel tier now sweeps "
                    "the automatic budget, at least mf_gs_parallel_sweeps, under either friction law. Until then a "
                    "count, or None for exactly mf_gs_parallel_sweeps, still sets the budget of the tiers that serve "
                    "persistent patch friction; omit it.",
                    DeprecationWarning,
                    stacklevel=_CONSTRUCTOR_CALLER,
                )
        # The sparse triplet transaction's flag, checked on every route; the sparse route applies it.
        self._sparse_transaction_request = self._sparse_transaction_flags()
        # Options whose effective value an IL_NEWTON_FPGS_* alias or a FEATHER_PGS_* override changed;
        # get_path_selection() reports their source as "environment".
        argument_decay = requested_pgs_decay if pgs_warmstart or not mf_warmstart else float(mf_warmstart_decay)
        self._environment_options = frozenset(
            name
            for name, changed in (
                ("mf_warmstart", legacy_mf_warmstart != bool(mf_warmstart)),
                ("mf_warmstart_decay", legacy_mf_decay != float(mf_warmstart_decay)),
                ("pgs_warmstart", self.pgs_warmstart != (bool(pgs_warmstart) or bool(mf_warmstart))),
                ("pgs_warmstart_decay", self.pgs_warmstart_decay != argument_decay),
                ("velocity_limit_law", self.velocity_limit_law != velocity_limit_law),
                ("contact_update", self.contact_update != contact_update),
                ("mf_gs_parallel_linked_sweeps", self.mf_gs_parallel_linked_sweeps != requested_linked_sweeps),
            )
            if changed
        )
        self._ws_prev_dense_impulses = None
        self._ws_prev_dense_row_type = None
        self._ws_prev_dense_row_parent = None
        self._ws_prev_dense_slot_sorted = None
        if self.pgs_warmstart and self.contact_friction_position_iterations >= 0:
            raise NotImplementedError(
                "contact_friction_position_iterations with pgs_warmstart=True needs an explicit "
                "pre-solve zeroing pass for skipped friction rows"
            )

        # MF contacts are one route owned by the unified mode. The internal
        # names remain because the MF buffers and kernels predate that API.
        self._mf_warmstart_enabled = self.pgs_warmstart
        self._mf_warmstart_decay = self.pgs_warmstart_decay
        # Allocated lazily in :meth:`_allocate_mf_buffers` only when enabled.
        self._ws_prev_mf_impulses = None
        self._ws_prev_dt = 0.0
        self._ws_prev_mf_row_type = None
        self._ws_prev_mf_row_parent = None
        self._ws_prev_slot_sorted = None

        self.pgs_mode = pgs_mode
        self.mf_max_constraints = int(mf_max_constraints)
        if self.pgs_mode == "matrix_free":
            _validate_mf_metadata_encoding(self.mf_max_constraints)
        # What only matrix-free mode (CUDA only) implements.
        if self.enable_joint_velocity_limits and pgs_mode != "matrix_free":
            raise NotImplementedError("enable_joint_velocity_limits=True currently requires pgs_mode='matrix_free'")
        if model.device.is_cpu and pgs_mode == "matrix_free":
            raise NotImplementedError(
                "SolverFeatherPGS pgs_mode='matrix_free' requires CUDA; use pgs_mode='split' on CPU."
            )
        if pgs_schedule != "interleaved" and pgs_mode != "matrix_free":
            raise ValueError("pgs_schedule is currently only supported with pgs_mode='matrix_free'")
        if self.pgs_velocity_iterations > 0 and pgs_mode != "matrix_free":
            raise ValueError("pgs_velocity_iterations is currently only supported with pgs_mode='matrix_free'")
        if drive_mode == "physx_pgs" and pgs_mode != "matrix_free":
            raise NotImplementedError("drive_mode='physx_pgs' currently requires pgs_mode='matrix_free'")
        self.pgs_schedule = pgs_schedule
        self.drive_mode = drive_mode
        # fuse_joint_velocity_limits is an engage-where-applicable request,
        # not a demand: the fused clamp only exists in the matrix_free +
        # physx_pgs + velocity-limits formulation (and needs
        # model.joint_velocity_limit), so any other configuration keeps its
        # dedicated rows silently instead of erroring. An inf activation
        # fraction is the explicit never-activate kill-switch for velocity
        # limits; the fused clamp rides the drive-row visit rather than the
        # gated velocity-limit phase, so it must stay disengaged there too.
        _fuse_applicable = (
            self.pgs_mode == "matrix_free"
            and self.enable_joint_velocity_limits
            and self.drive_mode == "physx_pgs"
            and not np.isinf(self.velocity_limit_activation_fraction)
            and getattr(model, "joint_velocity_limit", None) is not None
        )
        self.fuse_joint_velocity_limits = bool(fuse_joint_velocity_limits) and _fuse_applicable
        # Row budget of the incremental-residual matrix-free GS kernel; 0 disables it.
        self.mf_gs_incremental_rows = int(mf_gs_incremental_rows)
        self.mf_gs_parallel_rows = int(mf_gs_parallel_rows)
        self.mf_gs_parallel_sweeps = int(mf_gs_parallel_sweeps)
        self.mf_gs_parallel_nesterov = bool(mf_gs_parallel_nesterov)
        self.mf_gs_parallel_matrix_free = bool(mf_gs_parallel_matrix_free)
        self.mf_gs_parallel_linked_velocity_limit_pass = bool(mf_gs_parallel_linked_velocity_limit_pass)
        self._grouped_response = False
        self.rigid_body_max_linear_velocity = getattr(model, "rigid_body_max_linear_velocity", None)
        self.rigid_body_max_angular_velocity = getattr(model, "rigid_body_max_angular_velocity", None)
        self.rigid_body_max_depenetration_velocity = getattr(model, "rigid_body_max_depenetration_velocity", None)
        if self.rigid_body_max_depenetration_velocity is None:
            self.rigid_body_max_depenetration_velocity = wp.full(
                model.body_count,
                float("inf"),
                dtype=wp.float32,
                device=model.device,
                requires_grad=model.requires_grad,
            )
        self._has_rigid_body_velocity_limits = (
            self.rigid_body_max_linear_velocity is not None and self.rigid_body_max_angular_velocity is not None
        )

        self._double_buffer = double_buffer
        self._nvtx = nvtx
        self.pgs_debug = pgs_debug
        # The position/velocity split needs only v_out_snap. Full row snapshots
        # are diagnostics and can consume several GB in large batched models.
        self._debug_buffers_enabled = bool(pgs_debug)
        # External NaN dumpers read the stage-3 snapshots; nothing in the solver
        # does. They cost three copies per substep, so they are opt-in, and
        # pgs_debug, which keeps the solver's other diagnostics, keeps them too.
        self._debug_stage3_snapshot_flag = os.environ.get("FEATHER_PGS_DEBUG_SNAPSHOTS") == "1"
        self._debug_stage3_snapshots = self._debug_stage3_snapshot_flag or self._debug_buffers_enabled
        # The solver's device fields (solver_fields): one device, the model's gradient policy, debug mirrors.
        self._arrays = SolverArrays(
            self, model.device, requires_grad=model.requires_grad, mirrors=self._debug_buffers_enabled
        )
        self._pgs_convergence_log: list[np.ndarray] = []
        # Per-step, per-iteration NCP / MDP residual log for the matrix_free
        # debug path. Each entry is an ndarray of shape
        # ``[pgs_iterations, world_count, 6]`` holding
        # ``(r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir)``
        # per world per iteration. Populated only when ``pgs_debug`` is
        # ``True`` and ``pgs_mode == "matrix_free"``.
        self._pgs_ncp_residual_log: list[np.ndarray] = []

        small_dof_threshold = _SMALL_DOF_THRESHOLD_DEFAULT

        self.tile_threads = options["tile_threads"]
        # Launches over the contact capacity (the force export, the warm-start snapshot
        # and gathers, the keyed-row key scatters) grid-stride over about one wave of blocks.
        self._contact_capacity_blocks = contact_capacity_blocks(model.device)

        self.cholesky_kernel = cholesky_kernel
        self.trisolve_kernel = trisolve_kernel
        self.hinv_jt_kernel = hinv_jt_kernel
        self.delassus_kernel = delassus_kernel
        self.pgs_kernel = pgs_kernel
        self.small_dof_threshold = small_dof_threshold
        self.use_parallel_streams = use_parallel_streams

        self._step = 0
        self._force_mass_update = False
        self._mass_update_requested = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
        # Host mirror of the global mass-update flag evaluated by the last
        # _stage1_crba call; gates the per-step H memsets (see _stage1_crba).
        self._mass_update_global_flag = True
        self._last_step_dt = None
        self._fk_id_cache_available = bool(
            model.device.is_cuda and model.articulation_count and model.joint_count and not model.requires_grad
        )
        # The held tree's cache of next-step dynamics (held_tree.install enables it); ordinary dynamics derive
        # them from the state at every step, on CUDA as on the CPU.
        self._fk_id_cache_enabled = False
        self._fk_id_cache_source_state = None
        self._fk_id_cache_valid = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
        if model.body_count and not model.requires_grad:
            # Allocate the solver-owned augmented state here rather than on the first step(): a
            # wp.zeros() issued inside a CUDA graph capture becomes a memset node that replays every
            # launch and wiped the cached bias forces (body_f_s) at the first substep of each replay.
            self._allocate_state_aux_vars(model, self, False)

        self._model_plan: _FeatherPGSModelPlan | None = None
        # The model's joint forest (Model.joint_ancestor): conditioning depths, body maps, CRBA's ancestor test.
        self._joint_forest = _joint_forest(model)
        self._kinematic_joint_mask = wp.zeros(model.joint_count, dtype=wp.int32, device=model.device)
        self._kinematic_dof_mask = wp.zeros(model.joint_dof_count, dtype=wp.int32, device=model.device)
        self._apply_kinematic_state(self._kinematic_state())
        # Prescribed-response elision removes fully-kinematic free bases from
        # the compact response mapping of the matrix-free rows.
        self._model_plan = _FeatherPGSModelPlan.build(
            model,
            self._kinematic_dof_mask_host,
            enable_prescribed_response=self.pgs_mode == "matrix_free",
        )
        self._prescribed_articulation = wp.array(
            self._model_plan.prescribed_articulation, dtype=wp.int32, device=model.device
        )
        self._has_prescribed_response = bool(np.any(self._model_plan.prescribed_articulation != 0))
        self._compute_articulation_metadata(model)
        # Loop-closing joints are excluded from the tree (see _FeatherPGSModelPlan.build)
        # and enforced as CONNECT rows.
        self._has_loop_joints = bool(np.any(self._model_plan.loop_joint_articulation >= 0))
        # Mimic/connect plans must exist before capacity selection so the row census
        # counts their rows (see _estimate_dense_internal_rows_per_world).
        self._build_mimic_plan(model)
        self._build_connect_plan(model)
        self._build_preelimination_plan(model)
        # Contact torsion is decided with the other options, before the owners that exclude its appended row.
        configure_contact_torsion(
            self, contact_torsion_radius, contact_torsion_shape_indices, contact_torsion_shape_patterns
        )
        # The facts the owners' admissions test, computed once (step_plan.AdmissionFacts).
        self._facts = facts = AdmissionFacts.build(self)
        self._setup_passive_joint_forces(model)
        if self.pgs_mode == "matrix_free":
            self._compute_world_response_dof_mapping(model)
        self._setup_fused_diagonal_joint_limits(model)
        # Owners and consumers that neither write friction rows nor project
        # friction without the linked normal ring serve point and patch
        # friction. Producers that write their own friction rows stay
        # point-only until they write the patch row law. Defined before route
        # selection so admission can read both.
        self._law_agnostic_contact_paths = not (self._contact_torsion_enabled or self.contact_compliance)
        # The held tree's per-coordinate drive rows (each coordinate owns one, prepared apart from inverse dynamics)
        # on CUDA without gradients wherever each coordinate has one owning drive row. The held tree's admission,
        # which the representations only it evaluates (the compact diagonal, the native sparse factor) read before
        # they are chosen, needs them; ordinary dynamics drop them once the held tree declines (below) and evaluate
        # drives with the joint forces, as on the CPU.
        self._setup_parallel_augmented_drive_topology(
            model,
            model.device.is_cuda
            and not model.requires_grad
            and self.drive_mode == "augmented"
            and self.articulation_max_dofs > 0,
        )
        # The switch is read here, at construction, like the environment record.
        self._scalar_branches_requested = os.environ.get("FEATHER_PGS_SCALAR_BRANCHES", "1") != "0"
        from . import held_tree  # noqa: PLC0415
        from .scalar_branches import plan_scalar_branches  # noqa: PLC0415

        # The sparse owner does not solve the appended torsion row.
        sparse_diagonal_pair = (
            None
            if self._contact_torsion_enabled or self._explicit_dense_kernels
            else self._select_sparse_diagonal_response_pair(factor_dense_contract=self._fused_diagonal_joint_limits)
        )
        held = None
        if sparse_diagonal_pair is not None:
            # Its compact diagonal is the held tree's representation: a candidate only where the held tree takes
            # the dense groups with it, the diagonal group as scalar branches or a held family.
            scalar, _ = plan_scalar_branches(
                self, requested=self._scalar_branches_requested, size=sparse_diagonal_pair[0]
            )
            held = held_tree.compact_decline(self, sparse_diagonal_pair[0], scalar=scalar is not None)
            if held is not None:
                sparse_diagonal_pair = None
        self._sparse_diagonal_contact_solve = sparse_diagonal_pair is not None
        if self._contact_torsion_enabled or self._explicit_dense_kernels:
            self._sparse_diagonal_reason = (
                "Contact torsion or an explicit dense kernel choice keeps the general response."
            )
        elif held is not None:
            self._sparse_diagonal_reason = f"Only the held tree evaluates the compact diagonal. {held}"
        elif sparse_diagonal_pair is None:
            self._sparse_diagonal_reason = (
                "No world pairs a diagonal-mass articulation with fused limits and one compact dense articulation."
            )
        else:
            self._sparse_diagonal_reason = (
                "One diagonal-mass articulation and one compact dense articulation per world share one owner."
            )
        if sparse_diagonal_pair is None:
            # Limit rows belong to this specialized owner. If the complete
            # response composition is unavailable, retain the ordinary rows.
            self._fused_diagonal_joint_limits = False
            self._fused_diagonal_joint_limit_sizes = frozenset()
            self._fused_diagonal_limit_dof_mask_host.fill(0)
        self._sparse_diagonal_response_size = sparse_diagonal_pair[0] if sparse_diagonal_pair is not None else None
        self._sparse_diagonal_dense_size = sparse_diagonal_pair[1] if sparse_diagonal_pair is not None else 0
        # Persistent patches allocate one tangent pair per surviving anchor, so their contact rows are not
        # uniform normal/tangent/tangent triples; keep the triple schedule for point friction only.
        self._sparse_diagonal_contact_triples = bool(
            self._sparse_diagonal_contact_solve
            and self.enable_contact_friction
            and self.contact_friction_gap_threshold == math.inf
            and not self._friction_anchors_enabled
        )
        # Patch contacts are units instead: a normal row, followed by its two tangent rows only when the contact
        # owns an anchor. The unit schedule replaces every stride-3 assumption of the triple schedule.
        self._sparse_diagonal_contact_units = bool(
            self._sparse_diagonal_contact_solve
            and self.enable_contact_friction
            and self.contact_friction_gap_threshold == math.inf
            and self._friction_anchors_enabled
        )
        self._sparse_diagonal_speculative_contact_batches = bool(
            (self._sparse_diagonal_contact_triples or self._sparse_diagonal_contact_units)
            and self._sparse_diagonal_dense_size + 2 <= 16
        )
        self._sparse_diagonal_dense_offsets = wp.array(
            sparse_diagonal_pair[2] if sparse_diagonal_pair is not None else np.zeros(1, dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        self._sparse_diagonal_dense_groups = wp.array(
            sparse_diagonal_pair[3] if sparse_diagonal_pair is not None else np.zeros(1, dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        self._dense_internal_max_rows = self._estimate_dense_internal_rows_per_world(model)
        self.dense_max_constraints = dense_max_constraints
        self._hinv_jt_computes_diag = self.pgs_mode == "matrix_free" and not self._preelim_active
        self._execution_plan = _FeatherPGSExecutionPlan.build(
            self.size_groups,
            max_constraints=self.dense_max_constraints,
            max_shared_memory=int(getattr(model.device, "max_shared_memory_per_block", 0)),
            cholesky_kernel=self.cholesky_kernel,
            hinv_jt_kernel=self.hinv_jt_kernel,
            small_dof_threshold=self.small_dof_threshold,
            tile_threads=self.tile_threads,
            fuse_delassus=self.delassus_kernel == "auto",
            diagonal_mass_sizes=(
                self._diagonal_mass_sizes
                if self.cholesky_kernel == "auto" and self.trisolve_kernel == "auto"
                else frozenset()
            ),
        )
        self._hinv_jt_diag_sizes = frozenset(
            size
            for size in self.size_groups
            if self._hinv_jt_computes_diag
            and (self._execution_plan.use_diagonal_mass(size) or self._execution_plan.use_tiled_hinv_jt(size))
        )
        self._paired_response_candidate_sizes = frozenset(
            size
            for size in self.size_groups
            if model.device.is_cuda
            and not model.requires_grad
            and self.pgs_mode == "matrix_free"
            and not self._preelim_active
            and size >= _PERSISTENT_HINV_JT_MIN_DOF
            and size + 6 <= 32
            and self._execution_plan.use_tiled_hinv_jt(size)
            and self._execution_plan.hinv_jt_chunk_size(size) < self.dense_max_constraints
        )
        self._local_solve_max_rows = min(_LOCAL_SOLVE_MAX_ROWS, self.dense_max_constraints)
        self._local_residual_max_rows = min(_LOCAL_RESIDUAL_MAX_ROWS, self.dense_max_constraints)
        self._local_residual_mf_max_rows = min(_LOCAL_RESIDUAL_MF_MAX_ROWS, self.mf_max_constraints)
        local_budget = arch_budget(model.device.arch) if model.device.is_cuda else 0
        local_decline = first_failed(
            (
                (facts.forward_cuda, "Local owners need CUDA without gradient recording."),
                (self._has_free_rigid_bodies, "Local owners serve worlds with free rigid bodies; there are none."),
                (facts.matrix_free, "Local owners solve in matrix-free mode only."),
                (facts.plain_sweep, f"Local owners need {AdmissionFacts.PLAIN_SWEEP}."),
                (not facts.velocity_limits, "Local owners do not solve joint velocity-limit rows."),
                (self._local_solve_max_rows > 0, "dense_max_constraints leaves the local owners no row."),
                (not self._contact_torsion_enabled, "Local owners do not solve the appended torsion row."),
            )
        )
        local_base_supported = local_decline is None
        local_primary_articulation = np.full(self.world_count, -1, dtype=np.int32)
        local_pair_articulation = np.full(self.world_count, -1, dtype=np.int32)
        local_residual_pair_articulation = np.full(self.world_count, -1, dtype=np.int32)
        local_candidates: dict[int, list[int]] = {size: [] for size in self.size_groups}
        local_pair_candidates: dict[int, list[int]] = {size: [] for size in self.size_groups}
        local_pair_secondaries: dict[int, list[int]] = {size: [] for size in self.size_groups}
        local_residual_candidates: dict[int, list[int]] = {size: [] for size in self.size_groups}
        local_residual_secondaries: dict[int, list[int]] = {size: [] for size in self.size_groups}
        response_by_world: list[list[int]] = [[] for _ in range(self.world_count)]
        for art in np.flatnonzero(self._model_plan.response_dof_count > 0):
            response_by_world[int(self._model_plan.articulation_world[art])].append(int(art))

        def local_fits(
            primary_dofs: int,
            secondary_dofs: int = 0,
            max_rows: int | None = None,
            mf_rows: int = 0,
            *,
            warps_per_block: int = 1,
        ) -> bool:
            """Whether a local owner's block staging (its kernel's own declaration) fits the device's budget."""
            staging = sweep_routes_local.local_staging(
                self._local_solve_max_rows if max_rows is None else max_rows,
                primary_dofs,
                secondary_dofs,
                warps_per_block=warps_per_block,
                contact_capable=secondary_dofs > 0,
                local_mf_max_constraints=mf_rows,
            )
            return staging.fits(local_budget)

        if local_base_supported:
            articulation_dof_start = self._model_plan.articulation_dof_start
            for world, response_arts in enumerate(response_by_world):
                nonfree_arts = [art for art in response_arts if self._model_plan.is_free_rigid[art] == 0]
                if len(nonfree_arts) != 1:
                    continue
                primary_art = nonfree_arts[0]
                primary_dofs = int(self._model_plan.response_dof_count[primary_art])
                if (
                    primary_dofs <= 0
                    or primary_dofs > _LOCAL_INTERNAL_MAX_DOF
                    or self._execution_plan.use_tiled_hinv_jt(primary_dofs)
                    or not local_fits(primary_dofs, max_rows=min(self._local_solve_max_rows, primary_dofs))
                ):
                    continue
                local_primary_articulation[world] = primary_art
                local_candidates[primary_dofs].append(primary_art)

                free_arts = [art for art in response_arts if self._model_plan.is_free_rigid[art] != 0]
                if len(response_arts) != 2 or len(free_arts) != 1:
                    continue
                secondary_art = free_arts[0]
                secondary_dofs = int(self._model_plan.response_dof_count[secondary_art])
                if (
                    secondary_dofs != 6
                    or self._execution_plan.use_tiled_hinv_jt(secondary_dofs)
                    or not local_fits(primary_dofs, secondary_dofs)
                ):
                    continue
                local_pair_articulation[world] = secondary_art
                local_pair_candidates[primary_dofs].append(primary_art)
                local_pair_secondaries[primary_dofs].append(secondary_art)
                # The residual loop reads matrix-free metadata in world-packed DOF offsets while its scratch is
                # laid out primary-first; the world packing sorts articulations by DOF start, so only worlds where
                # the primary articulation precedes the free body qualify.
                primary_first_layout = int(articulation_dof_start[primary_art]) < int(
                    articulation_dof_start[secondary_art]
                )
                if primary_first_layout and local_fits(
                    primary_dofs, secondary_dofs, self._local_residual_max_rows, self._local_residual_mf_max_rows
                ):
                    local_residual_pair_articulation[world] = secondary_art
                    local_residual_candidates[primary_dofs].append(primary_art)
                    local_residual_secondaries[primary_dofs].append(secondary_art)

        self._local_primary_articulation = wp.array(local_primary_articulation, dtype=wp.int32, device=model.device)
        self._local_pair_articulation = wp.array(local_pair_articulation, dtype=wp.int32, device=model.device)
        self._local_residual_pair_articulation = wp.array(
            local_residual_pair_articulation, dtype=wp.int32, device=model.device
        )
        self._local_solve_owner = wp.zeros(self.world_count, dtype=wp.int32, device=model.device)
        self._local_general_world_count = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._local_general_worlds = wp.empty(self.world_count, dtype=wp.int32, device=model.device)
        self._local_general_solver_blocks = min(
            self.world_count, max(1, int(model.device.sm_count) * _LOCAL_GENERAL_BLOCKS_PER_SM)
        )

        def arrays(lists):
            """Each size's non-empty articulation list as a device array."""
            return {size: wp.array(arts, dtype=wp.int32, device=model.device) for size, arts in lists.items() if arts}

        def per_size(counts, make):
            """One buffer per size with candidates, made from its candidate count."""
            return {size: make(count) for size, count in counts.items() if count > 0}

        def counter(count):
            return wp.zeros(1, dtype=wp.int32, device=model.device)

        def slots(count):
            return wp.empty(count, dtype=wp.int32, device=model.device)

        self._local_internal_candidates = arrays(local_candidates)
        self._local_pair_candidates = arrays(local_pair_candidates)
        self._local_pair_secondaries = arrays(local_pair_secondaries)
        self._local_residual_candidates = arrays(local_residual_candidates)
        self._local_residual_secondaries = arrays(local_residual_secondaries)
        self._local_internal_art_counts = {size: len(arts) for size, arts in local_candidates.items()}
        self._local_pair_art_counts = {size: len(arts) for size, arts in local_pair_candidates.items()}
        self._local_residual_art_counts = {size: len(arts) for size, arts in local_residual_candidates.items()}
        self._local_pair_active_counts = per_size(self._local_pair_art_counts, counter)
        self._local_pair_active_candidates = per_size(self._local_pair_art_counts, slots)
        self._local_pair_active_secondaries = per_size(self._local_pair_art_counts, slots)
        self._local_residual_active_counts = per_size(self._local_residual_art_counts, counter)
        self._local_residual_active_candidates = per_size(self._local_residual_art_counts, slots)
        self._local_residual_active_secondaries = per_size(self._local_residual_art_counts, slots)
        sm_count = int(model.device.sm_count)
        self._local_pair_solver_blocks = per_size(
            self._local_pair_art_counts, lambda count: min(count, max(1, sm_count * _LOCAL_PAIR_BLOCKS_PER_SM))
        )
        self._local_residual_solver_warps = per_size(
            self._local_residual_art_counts, lambda count: min(count, max(1, sm_count))
        )
        self._local_residual_warps_per_block = {
            size: (
                2
                if solver_warps % 2 == 0
                and local_fits(
                    size, 6, self._local_residual_max_rows, self._local_residual_mf_max_rows, warps_per_block=2
                )
                else 1
            )
            for size, solver_warps in self._local_residual_solver_warps.items()
        }
        self._local_internal_fast_path = bool(
            local_base_supported and any(count > 0 for count in self._local_internal_art_counts.values())
        )
        self._local_solve_reason = local_decline or (
            "Worlds with one small non-free articulation (and at most one free body) solve in local shared memory."
            if self._local_internal_fast_path
            else "No world holds one non-free articulation of at most "
            f"{_LOCAL_INTERNAL_MAX_DOF} DOFs whose local staging fits the device."
        )
        # All inactive weight arguments alias this one element. Their producer
        # kernels skip stores and their consumer kernels skip reads when
        # regularization is disabled.
        self._contact_row_w_dummy = self._arrays.placeholder(wp.float32, (1, 1), 1.0)
        self._point_contact_paths = self._law_agnostic_contact_paths and not self._friction_anchors_enabled
        # The parallel matrix-free tiers pool each friction pair's cone over its patch
        # ring and keyed rows write the patch row law, so both can serve point and patch
        # friction (Jacobi sweeps, not the serial patch sweep). Patch friction takes them
        # by default; FEATHER_PGS_PATCH_PARALLEL_ROWS=0 keeps its serial sweeps and
        # ordinary rows.
        self._parallel_contact_paths, self._parallel_contact_paths_reason = self._admit_parallel_contact_paths()
        # Patch friction on the dense route: the paired factor sweep walks patch
        # contact units, the resolved-world classifier skips worlds whose rows
        # all separate, and worlds with free rigid bodies keep the dense route.
        self._patch_dense_paths = bool(self._law_agnostic_contact_paths and self._friction_anchors_enabled)
        if self.pgs_mode == "matrix_free":
            self._jy_world_aliased = self._detect_jy_world_identity()
        else:
            self._jy_world_aliased = False
        self._allocate_common_buffers(model)
        self._compact_diagonal_mass_size = self._sparse_diagonal_response_size
        self._row_watermark = bool(row_watermark)
        self._setup_sparse_mass_matrix(model)
        if sparse_mass_matrix == "on" and self._sparse_mass_matrix_size is None:
            raise ValueError(
                "sparse_mass_matrix='on' is unsupported by this topology, device, solver policy or research mode; "
                "use 'auto' to allow the ordinary representation"
            )
        # The matrix-free sweeps read the dense rows packed (packed_rows): decided before the buffers, since that
        # placement allocates neither a size group's response nor the world layout.
        self._mf_sweeps = bool(
            self.pgs_mode == "matrix_free"
            and not self._sparse_diagonal_contact_solve
            and self._sparse_mass_matrix_size is None
            and self.world_count > 0
            and getattr(self, "max_world_dofs", 0) > 0
            and self.mf_max_constraints > 0
        )
        if self._mf_sweeps:
            self._jy_world_aliased = False
            self._hinv_jt_diag_sizes = frozenset()
        self._allocate_buffers(model)
        self._allocate_world_buffers(model)
        # CONTACT row_parent belongs to ordinary friction/patch load linkage.
        # Torsion must never reinterpret it, including when patch PRs are merged.
        self._contact_torsion_group = wp.full(
            self.row_parent.shape if self._contact_torsion_enabled else (1, 1),
            -1,
            dtype=wp.int32,
            device=model.device,
        )
        # Bilateral pre-elimination corrects the grouped response after H^-1 J^T,
        # so that path must retain the existing post-correction world gather.
        self._hinv_jt_writes_world = (
            self.pgs_mode == "matrix_free"
            and not self._jy_world_aliased
            and not self._preelim_active
            and not self._sparse_diagonal_contact_solve
            and self._sparse_mass_matrix_size is None
        )
        self._hinv_jt_tiled_writes_group = not self._hinv_jt_writes_world or self.pgs_warmstart
        self._paired_response_reason = first_failed(
            (
                (
                    not (self._sparse_diagonal_contact_solve or self._sparse_mass_matrix_size is not None),
                    "A sparse response owner owns response.",
                ),
                (not self._explicit_dense_kernels, "An explicit dense kernel choice owns response."),
            )
        )
        response_pair = None if self._paired_response_reason else self._select_paired_response()
        if response_pair is None and self._paired_response_reason is None:
            self._paired_response_reason = "No world pairs one tiled-response articulation with exactly one free body."
        self._paired_response_primary_size = response_pair[0] if response_pair is not None else None
        self._paired_response_secondary_size = response_pair[1] if response_pair is not None else None
        self._paired_response_secondary_groups = (
            wp.array(response_pair[2], dtype=wp.int32, device=model.device) if response_pair is not None else None
        )
        self._factor_coordinates_reason = first_failed(
            (
                (response_pair is not None, "No paired response."),
                (not self._local_internal_fast_path, "Local owners claim the paired worlds."),
                (facts.plain_sweep, f"The factor-coordinate sweep needs {AdmissionFacts.PLAIN_SWEEP}."),
                (not facts.regularized, "The factor-coordinate sweep solves the unregularized law."),
                (self.pgs_iterations > 0, "The factor-coordinate sweep needs a positive iteration budget."),
                (not facts.velocity_limits, "The factor-coordinate sweep solves no joint velocity-limit rows."),
                (not self._contact_torsion_enabled, "The factor-coordinate owner does not solve the torsion row."),
            )
        )
        self._paired_factor_coordinates = self._factor_coordinates_reason is None
        # Persistent patches allocate one tangent pair per surviving anchor, so their contact rows are not
        # uniform normal/tangent/tangent triples; the generic row path pools the patch load through the linked
        # normal rows instead.
        self._factor_coordinate_contact_triples = bool(
            self._paired_factor_coordinates
            and self.enable_contact_friction
            and self.contact_friction_gap_threshold == math.inf
            and not self._friction_anchors_enabled
        )
        # Patch contacts are units of one row, or three when they own an anchor;
        # the factor sweep walks them with the generic path's row order and
        # arithmetic, solving each tangent pair once.
        self._factor_coordinate_contact_units = bool(self._paired_factor_coordinates and self._patch_dense_paths)
        self._paired_factor_primary_groups_by_world = None
        self._paired_factor_secondary_groups_by_world = None
        self._paired_factor_primary_offsets_by_world = None
        self._paired_factor_secondary_offsets_by_world = None
        if response_pair is not None:
            if self._paired_factor_coordinates:
                response_dof_count = self._model_plan.response_dof_count
                primary_arts = np.flatnonzero(response_dof_count == response_pair[0])
                secondary_arts = np.flatnonzero(response_dof_count == response_pair[1])
                primary_groups = np.empty(self.world_count, dtype=np.int32)
                secondary_groups = np.empty(self.world_count, dtype=np.int32)
                primary_offsets = np.empty(self.world_count, dtype=np.int32)
                secondary_offsets = np.empty(self.world_count, dtype=np.int32)
                articulation_world = self._model_plan.articulation_world
                articulation_dof_start = self._model_plan.articulation_dof_start
                for primary_group, primary_art in enumerate(primary_arts):
                    world = int(articulation_world[primary_art])
                    secondary_group = int(response_pair[2][primary_group])
                    secondary_art = int(secondary_arts[secondary_group])
                    primary_groups[world] = primary_group
                    secondary_groups[world] = secondary_group
                    primary_first = articulation_dof_start[primary_art] < articulation_dof_start[secondary_art]
                    primary_offsets[world] = 0 if primary_first else response_pair[1]
                    secondary_offsets[world] = response_pair[0] if primary_first else 0
                self._paired_factor_primary_groups_by_world = wp.array(
                    primary_groups, dtype=wp.int32, device=model.device
                )
                self._paired_factor_secondary_groups_by_world = wp.array(
                    secondary_groups, dtype=wp.int32, device=model.device
                )
                self._paired_factor_primary_offsets_by_world = wp.array(
                    primary_offsets, dtype=wp.int32, device=model.device
                )
                self._paired_factor_secondary_offsets_by_world = wp.array(
                    secondary_offsets, dtype=wp.int32, device=model.device
                )
        if not self._hinv_jt_writes_world and response_pair is None:
            self._hinv_jt_diag_sizes = frozenset()
        # The sparse-diagonal schedule's buffers, one row per world when it is off.
        if self._sparse_diagonal_contact_solve:
            self._arrays.allocate(SPARSE_DIAGONAL_ROWS, world=self.world_count, row=self.dense_max_constraints)
        else:
            self._arrays.allocate(SPARSE_DIAGONAL_ROWS, world=1, row=1)
        if self._sparse_diagonal_contact_triples or self._sparse_diagonal_contact_units:
            self._arrays.allocate(SPARSE_CONTACT_GROUPS, world=self.world_count, world_dof=self.max_world_dofs)
        else:
            self._arrays.allocate(SPARSE_CONTACT_GROUPS, world=1, world_dof=1)
        if self._sparse_diagonal_speculative_contact_batches:
            # A triple spans three rows, while a unit may be a single normal row.
            rows = self.dense_max_constraints
            serial = rows if self._sparse_diagonal_contact_units else (rows + 2) // 3
            self._arrays.allocate(SPARSE_CONTACT_SERIAL, world=self.world_count, serial=serial)
        else:
            self._arrays.allocate(SPARSE_CONTACT_SERIAL, world=1, serial=1)
        self._allocate_mf_buffers(model)
        self._body_factors_data = self._build_body_factors()
        self.mf_target_velocity = (
            wp.zeros_like(self.mf_rhs, requires_grad=model.requires_grad)
            if self._has_prescribed_response
            else self._arrays.placeholder(wp.float32, (1, 1))
        )
        self._allocate_debug_buffers(model)
        self._scatter_armature_to_groups()
        self._plan_tier_rows()
        self._allocate_packed_rows()
        # Worlds of one limited articulation and at most one other get their
        # limit prefix and contact packets built in place for the local solve
        # owners (row_packets.py), with the point or the patch row law;
        # FEATHER_PGS_ROW_PACKETS=0 opts out. The keyed rows below take patch
        # friction unless FEATHER_PGS_PATCH_PARALLEL_ROWS=0.
        from . import row_packets  # noqa: PLC0415

        self._row_packets = None
        self._row_packets_reason = first_failed(
            (
                (self._sparse_mass_matrix_size is None, "Native sparse factors own the contact rows."),
                (
                    self._law_agnostic_contact_paths,
                    "Local packets implement the point and patch row laws; torsion and compliance keep ordinary rows.",
                ),
                (
                    os.environ.get("FEATHER_PGS_ROW_PACKETS", "1") == "1",
                    "FEATHER_PGS_ROW_PACKETS disables local packets.",
                ),
            )
        ) or row_packets.decline(self)
        if self._row_packets_reason is None:
            self._row_packets = row_packets.create_owner(self)
            self._row_packets_reason = (
                "Admitted: worlds of one limited articulation and at most one other build packets in place."
                if self._row_packets is not None
                else "The local packet law does not admit this topology."
            )
        # Keyed rows are on by policy wherever their producer admits the solver
        # (keyed_rows.create_owner); FEATHER_PGS_KEYED_ROWS=0 opts out.
        self._keyed_rows = None
        if self._sparse_mass_matrix_size is not None:
            self._keyed_rows_reason = "Native sparse factors own the contact rows."
        elif self._regularization_enabled:
            self._keyed_rows_reason = self._parallel_contact_paths_reason
        elif not self._parallel_contact_paths:
            self._keyed_rows_reason = (
                "Keyed rows write the point and patch row laws; torsion and compliance keep ordinary rows."
                if not self._law_agnostic_contact_paths
                else "Patch friction keeps ordinary rows: FEATHER_PGS_PATCH_PARALLEL_ROWS=0 opts it out."
            )
        elif os.environ.get("FEATHER_PGS_KEYED_ROWS", "1") == "0":
            self._keyed_rows_reason = "FEATHER_PGS_KEYED_ROWS=0 disables keyed rows."
        else:
            from .keyed_rows import create_owner as create_kinetic_rows_owner  # noqa: PLC0415

            self._keyed_rows = create_kinetic_rows_owner(self)
            self._keyed_rows_reason = (
                "Admitted: the in-kernel parallel sweep builds keyed rows for its bounded worlds."
                if self._keyed_rows is not None
                else "Keyed rows need matrix-free parallel tiers (mf_gs_parallel_rows and mf_gs_parallel_matrix_free) "
                "that build the response in-kernel (routes.sweep.tiers) and a plain matrix-free policy."
            )
        self._init_tiled_kernels(model)
        self._init_size_group_streams(model)
        # Operands of the features that are off (SolverArrays.placeholder). reset_world_warmstart_buffers
        # writes -1 into its row-type and parent arguments, so their sink holds -1; the matrix-free counters
        # stand in at world width, and allocate_world_contact_slots writes none of them when has_free_rigid == 0.
        placeholder = self._arrays.placeholder
        self._dummy_contact_impulses = placeholder(wp.float32, (1, 1))
        self._dummy_world_J = placeholder(wp.float32, (1, 1, 1))
        self._dummy_is_free_rigid = placeholder(wp.int32, (1,))
        self._dummy_mf_slot_counter = placeholder(wp.int32, (self.world_count,))
        self._dummy_mf_impulses = placeholder(wp.float32, (1, 1))
        self._dummy_contact_count = placeholder(wp.int32, (1,))
        self._dummy_contact_row_type = placeholder(wp.int32, (1, 1))
        self._reset_dummy_mf_row_type = placeholder(wp.int32, (1, 1), -1)
        self._reset_dummy_mf_row_parent = placeholder(wp.int32, (1, 1), -1)
        self._dummy_contact_row_parent = placeholder(wp.int32, (1, 1), -1)

        # Sticky flags of constraint_capacity_status() (dense and matrix-free rows), the per-world overflow
        # latch and the dropped rows of each family.
        self._arrays.allocate(CAPACITY_STATUS, world=self.world_count, status_world=max(self.world_count, 1))
        self._row_dropped_dense = self._row_dropped_all[0]
        self._row_dropped_mf = self._row_dropped_all[1]
        # Constraint-capacity diagnostics are allocated once before CUDA graph
        # capture. The warning path tracks only current-step dropped rows and a
        # one-shot flag per row family; row_watermark additionally accumulates
        # whole-run high-water telemetry for explicit readback.
        self._track_row_capacity = (
            self._row_watermark
            or self.warn_constraint_overflow
            or self._contact_torsion_enabled
            or self.contact_compliance
        )
        # Dropped-row accounting is unconditional (see _row_dropped_all above);
        # _track_row_capacity now gates only the optional warning/watermark readbacks.
        if self.warn_constraint_overflow:
            self._arrays.allocate(OVERFLOW_WARNING)
        else:
            self._arrays.release(OVERFLOW_WARNING)
        # The telemetry words share one array, which constraint_row_watermarks() reads back once.
        self._row_watermark_words = None
        if self._row_watermark:
            self._row_watermark_words = self._arrays.allocate_words(ROW_WATERMARKS)
        else:
            self._arrays.release(ROW_WATERMARKS)

        if model.shape_material_mu is not None:
            self.shape_material_mu = model.shape_material_mu
        else:
            self.shape_material_mu = wp.zeros(
                (1,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )
        if model.shape_material_restitution is not None:
            self.shape_material_restitution = model.shape_material_restitution
        else:
            self.shape_material_restitution = wp.zeros(
                (1,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )

        self._init_double_buffer_stream()

        self.contact_torsion_device = bool(contact_torsion_device)
        if self._contact_torsion_enabled and self.contact_torsion_device:
            from .contact_torsion_device import enable_device_torsion  # noqa: PLC0415

            enable_device_torsion(self)

        # Scalar branches take a diagonal-mass group of world-fixed roots over
        # one-coordinate leaves (scalar_branches.py), which the held-tree owner
        # then publishes: the compact diagonal's admission planned them.
        self._scalar_branches, self._scalar_branches_reason = plan_scalar_branches(
            self, requested=self._scalar_branches_requested
        )

        # The compact boundary owns the rows and response of every world the sparse-diagonal owner pairs with a
        # six-coordinate dense articulation. It writes the point and the patch row law; torsion and compliance keep
        # ordinary rows.
        from .compact_contact import supported  # noqa: PLC0415

        if not self._law_agnostic_contact_paths:
            declined = (
                "Compact contacts implement the point and patch row laws; torsion and compliance keep ordinary rows."
            )
        elif not supported(self):
            declined = (
                "Declined: requires the six-coordinate sparse-diagonal triple or unit contract and no row checks."
            )
        else:
            declined = None
        self._compact_contact_boundary = declined is None
        self._compact_contact_reason = declined or "Admitted: the six-coordinate sparse-diagonal triples or units."

        held_tree.install(self)
        # Ordinary dynamics take the CPU's drive pass and layout (_stage1_drives: the joint forces and the drives in
        # one launch, K added after the assembly); the per-coordinate drive rows are the held tree's.
        # use_parallel_streams overlaps their preparation with FK/ID on a stream of its own.
        if self._tree_kinetic_state is None:
            self._parallel_augmented_drive_topology = False
        self._articulation_dynamics_stream = (
            wp.Stream(model.device) if self._parallel_augmented_drive_topology and self.use_parallel_streams else None
        )

        # Worlds whose every candidate row separates at the predictor skip row
        # allocation entirely (resolved_worlds.py); the mask is empty when the
        # solver's row law is outside what the classifier evaluates.
        from .resolved_worlds import ResolvedWorlds  # noqa: PLC0415

        classifier = ResolvedWorlds(self)
        self._resolved_worlds_owner = classifier if classifier.enabled else None
        self._resolved_worlds = classifier.resolved
        self._resolved_worlds_reason = classifier.reason

        self._free_body_gyro_kernel = None
        if os.environ.get("FEATHER_PGS_FREE_BODY_IMPLICIT_GYRO") == "1" and self._has_free_rigid_bodies:
            # A predictor experiment independent of contact scheduling. The
            # effective six-DOF factor retains drive/armature contributions;
            # physical COM inertia supplies only the gyroscopic correction.
            if model.requires_grad or self.L_by_size.get(6) is None:
                raise ValueError("Experimental implicit free-body gyro requires ordinary six-DOF mass factors")
            from .free_body_gyro import correct_free_body_gyroscopic_predictor  # noqa: PLC0415

            self._free_body_gyro_kernel = correct_free_body_gyroscopic_predictor

        self._place_dense_response()
        # The step's per-stage dispatch tables, which step() and the route record read, and the kernels they launch.
        self._plan = StepPlan.build(self)
        self._build_stage_kernels()
        self._setup_contact_update(model)
        # Record the applied options once; get_path_selection() returns detached copies.
        self._applied_options = applied_options(self, SolverFeatherPGS.__init__, requested_options)
        if self.contact_update != "sequential":
            for message in contact_update_warnings(route_selection(self)["sweep"]):
                warnings.warn(message, UserWarning, stacklevel=_CONSTRUCTOR_CALLER)

    def _setup_contact_update(self, model):
        """Select the sweeps that solve contact units exactly, allocate their unit storage and build the loop sweeps.

        ``contact_update="unit"`` reaches the loop sweeps only: the dense loop sweep of split mode on
        any device and, on CPU, the split free-body loop sweep (CUDA builds a native free-body sweep).
        Every other sweep keeps the sequential law (``routes.sweep.contact_update``). ``_contact_unit_patch`` is an
        internal switch for A/B runs: ``"ring"`` solves a patch region as one ring unit, ``"anchor"``
        each anchor as a 3x3 unit with the ring's other load held; launches read it.
        """
        dense_split = self.pgs_mode != "matrix_free" and not self._sparse_split
        unit = self.contact_update == "unit"
        self._dense_contact_unit = bool(unit and dense_split and self.pgs_kernel == "loop")
        self._free_body_contact_unit = bool(
            unit and dense_split and self._has_free_rigid_bodies and model.device.is_cpu
        )
        self._contact_unit_patch = "ring"
        units = {"world": self.world_count, "unit_fields": CONTACT_UNIT_STATE_FIELDS}
        units["unit_block"] = self.world_count * RING_BLOCK_SIZE
        if self._dense_contact_unit:
            self._arrays.allocate(DENSE_CONTACT_UNITS, row=self.dense_max_constraints, **units)
        else:
            self._arrays.release(DENSE_CONTACT_UNITS)
        if self._free_body_contact_unit:
            self._arrays.allocate(MF_CONTACT_UNITS, mf_row=self.mf_max_constraints, **units)
        else:
            self._arrays.release(MF_CONTACT_UNITS)
        # Split mode's loop sweeps, one thread per world, under the law chosen above: the dense rows' and, on the
        # CPU, the free-body rows'.
        self._pgs_solve_loop_kernel = self._pgs_solve_mf_loop_kernel = None
        if self.pgs_mode == "split" and self.pgs_kernel != "tiled_row":
            self._pgs_solve_loop_kernel = self._sweep_kernel(
                "dense_loop",
                sweep_routes_split.loop_unit_kernel if self._dense_contact_unit else sweep_routes_split.loop_kernel,
            )
        if self._has_free_rigid_bodies and model.device.is_cpu:
            self._pgs_solve_mf_loop_kernel = self._sweep_kernel(
                "free_body_loop",
                sweep_routes_split.mf_loop_unit_kernel
                if self._free_body_contact_unit
                else sweep_routes_split.mf_loop_kernel,
            )

    def get_path_selection(self) -> dict:
        """Return requested choices, active plans and selection reasons.

        The record is read on the host without device synchronization, except
        under ``pgs_solve_readout``: reading its counters synchronizes once.

        Automatic choices use topology and resource rules, never timing.
        Recreate the solver to change them. The ``reasons`` mapping explains construction-time admission and
        fallbacks; it does not predict dynamic per-world branches.

        ``routes`` records every construction-time owner and variant that decides
        which kernels run: ``routes.stages`` the step plan's per-size stage routes
        (factor, predictor solve, response, Delassus) that ``step()`` launches,
        each with its reason (the kernel choice, the size's resource rule), the
        held-tree owner and its lane layouts, the contact-row
        producer with each DOF size's row-builder kernels and friction law (from
        the plan the row build dispatches on), local, paired and sparse responses,
        the sweep kernels per route and tier, what drives them and the matrix-free
        sweep's shared-memory storage (``sweep.mf_gs_storage``), resolved worlds,
        publication, velocity limits and the lane width of their prescale, free
        bodies and the friction law, each with a reason; ``routes.sweep.readout`` names the
        sweep kernels that count their sweeps, the passes that count the solves
        of worlds without rows as skipped and the owners of the worlds no counter
        counts, and ``routes.sweep.exit_law`` the law every sweep stops under. The
        row builders (``contact_rows.builder_by_dof``) are those of the dense
        contact rows built outside the sweep: the parallel tiers build keyed rows'
        direct rows in the sweep (``direct_rows``) and the free-body matrix-free
        rows follow ``free_rigid_bodies``.
        ``dynamics.double_buffer_mass_matrix_clear_dofs`` lists the DOF sizes
        whose double-buffered mass matrix the end of a refresh step clears (empty
        without a double buffer, where the ordinary route's stage 1 clears it), and
        ``contact_consumers.max_blocks`` the block cap of the launches over the
        contact capacity (0: uncapped). ``measurements`` holds, under
        ``pgs_solve_readout``, the cumulative sweep counts
        (``solve_readout``), kept out of ``routes`` so that two runs of one
        configuration compare equal. ``options`` records every constructor
        option's requested, declared-default and applied value, captured once at
        construction; the applied value is the one in effect after construction's
        gates and clamps, off (``False``, ``0`` or ``"off"``) when the owner, tier
        or stream an option requests was not built.
        ``routes.held_tree.scalar_branches`` states whether a diagonal-mass group
        runs as scalar branches, and why.
        ``environment`` records the ``FEATHER_PGS_*`` and
        ``IL_NEWTON_FPGS_*`` variables present at construction: ``applied`` maps
        declared names to their values, ``unknown`` lists undeclared names
        (recorded, not rejected; removed switches among them), and ``retired`` and
        ``isaac_lab`` are empty for this family.
        ``routes.diagnostics.stage3_snapshots`` records whether
        ``FEATHER_PGS_DEBUG_SNAPSHOTS=1`` or ``pgs_debug`` allocated the stage-3
        snapshots, which every substep then copies.
        Switches read at import keep the values present at import. The record is
        host-only, detached from solver state and JSON-compliant (non-finite
        floats are strings).
        """
        cached = self._tree_kinetic_state is not None
        selection = {
            "requested": dict(self._requested_kernel_choices),
            "sparse_mass_matrix": "on" if self._sparse_mass_matrix_size is not None else "off",
            "cached_dynamics": "on" if cached else "off",
            "publication": "cached" if cached else "ordinary",
            "pgs_kernel": "matrix_free" if self.pgs_mode == "matrix_free" else self.pgs_kernel,
            "cached_prediction": "tree" if self._tree_kinetic_state is not None else None,
            "reasons": {
                "sparse_mass_matrix": self._sparse_selection_reason,
                "cached_dynamics": self._cached_dynamics_selection_reason,
                "publication": "Paired cache owns next-state publication."
                if cached
                else "Ordinary state publication remains active.",
                "pgs_kernel": "Matrix-free mode owns the iteration kernel; split PGS selection is unused."
                + (
                    ""
                    if self._requested_kernel_choices["pgs_kernel"] == "auto"
                    else f" {self._kernel_choice_reasons['pgs_kernel']}"
                )
                if self.pgs_mode == "matrix_free"
                else self._kernel_choice_reasons["pgs_kernel"],
            },
        }
        selection["routes"] = route_selection(self)
        selection["measurements"] = measurement_selection(self)
        selection["options"] = options_selection(self)
        selection["environment"] = copy.deepcopy(self._environment_record)
        return selection

    def prepare_contact_torsion_capture(self, state_in: State, state_out: State) -> None:
        """Prepare experimental torsion rollback buffers before CUDA graph capture.

        Requires ``contact_torsion_device=True`` when torsion is active. Call
        outside capture with the state layouts used by the graph. Subsequent
        graph replay requires :meth:`validate_contact_torsion` at every batch
        boundary before results are consumed. Eager steps continue to validate
        automatically. This method does not advance physics.
        """
        if not self._contact_torsion_enabled:
            return
        preparation = getattr(self, "_device_torsion", None)
        if preparation is None:
            raise RuntimeError("Contact torsion graph capture requires contact_torsion_device=True")
        if wp.get_stream(self.model.device).is_capturing:
            raise RuntimeError("Prepare contact torsion buffers outside CUDA graph capture")
        preparation.validate()
        preparation.deferred_errors = True
        preparation.begin_step(state_in, state_out)

    def validate_contact_torsion(self) -> None:
        """Raise latched torsion errors after every experimental graph replay batch.

        This synchronous status check must run before consuming captured results.
        Invalid input suppresses incomplete contact solving and restores published
        dynamic state, but internal caches are not rolled back: recreate the solver
        after correcting invalid input. No-op without device torsion preparation.
        """
        preparation = getattr(self, "_device_torsion", None)
        if preparation is not None:
            if wp.get_stream(self.model.device).is_capturing:
                raise RuntimeError("Validate contact torsion outside CUDA graph capture")
            preparation.validate()

    def _kinematic_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, ArmatureFloor]:
        """Derive the kinematic masks and effective joint armature from the body flags, changing nothing.

        The effective armature also holds the least armature that keeps the mass matrix of a deep
        articulation within float32 factorization (:func:`conditioning_armature`).

        Returns:
            ``(armature, joint_mask, dof_mask, floor)`` for :meth:`_apply_kinematic_state`.

        Raises:
            RuntimeError: If the flags make a DOF of a sparse split or
                velocity-limit response kinematic, or a DOF of a
                prescribed-response articulation dynamic. Both need a new solver.
        """
        model = self.model
        armature = model.joint_armature.numpy().copy()
        joint_mask = np.zeros(model.joint_count, dtype=np.int32)
        dof_mask = np.zeros(model.joint_dof_count, dtype=np.int32)

        if model.body_count and model.joint_count:
            kinematic_bodies = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
            joint_child = model.joint_child.numpy()
            joint_qd_start = model.joint_qd_start.numpy()
            kinematic_joints = np.flatnonzero(kinematic_bodies[joint_child])
            joint_mask[kinematic_joints] = 1
            dof_start = joint_qd_start[kinematic_joints]
            joint, offset = host_index.expand(np.maximum(joint_qd_start[kinematic_joints + 1] - dof_start, 0))
            dof_mask[dof_start[joint] + offset] = 1
            armature[dof_start[joint] + offset] = 1.0e10

        if (getattr(self, "_sparse_split", False) or getattr(self, "_sparse_velocity_limits", False)) and np.any(
            dof_mask
        ):
            raise RuntimeError(
                "Sparse response gained prescribed DOFs; reconstruct the solver and recapture its CUDA graph."
            )

        if self._model_plan is not None:
            selected = np.nonzero(self._model_plan.prescribed_articulation != 0)[0]
            for articulation in selected:
                dof_start = int(self._model_plan.articulation_dof_start[articulation])
                dof_count = int(self._model_plan.articulation_dof_count[articulation])
                if not np.all(dof_mask[dof_start : dof_start + dof_count] != 0):
                    raise RuntimeError(
                        "A prescribed-response articulation changed kinematic membership; "
                        "reconstruct the solver and recapture its CUDA graph."
                    )
        armature, floor = conditioning_armature(model, armature, self._joint_forest)
        return armature, joint_mask, dof_mask, floor

    def _apply_kinematic_state(self, kinematic: tuple[np.ndarray, np.ndarray, np.ndarray, ArmatureFloor]) -> None:
        """Publish derived kinematic masks and armature in place, where captured graphs read them."""
        armature, joint_mask, dof_mask, floor = kinematic
        if floor.coordinates and floor != getattr(self, "_armature_floor", None):
            warnings.warn(
                f"SolverFeatherPGS adds joint armature, up to {floor.largest:.3g}, to {floor.coordinates} "
                f"coordinate(s) of {floor.articulations} articulation(s) whose mass matrix float32 cannot factor "
                "without it (routes.dynamics.armature_floor). Give these joints armature to choose it yourself.",
                UserWarning,
                stacklevel=3,
            )
        self._armature_floor = floor
        self._joint_armature_effective = armature
        self._kinematic_dof_mask_host = dof_mask.copy()
        self._kinematic_joint_mask.assign(joint_mask)
        self._kinematic_dof_mask.assign(dof_mask)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh cached solver data after supported model changes and invalidate affected impulse history.

        With patch friction enabled, shape-property notifications copy geometry
        to the host and synchronize with the device; issue them outside CUDA graph
        capture. Ordinary simulation steps do not perform those host copies.
        Material-equivalence changes also require
        :meth:`newton.CollisionPipeline.refresh_body_pair_reduction_groups` when
        body-pair reduction is enabled. Capacity failures remain latched until
        an explicit episode reset. Body flags that make a DOF of a
        prescribed-response articulation dynamic, or a DOF of a sparse split or
        velocity-limit response kinematic, raise before any solver state
        changes; reconstruct the solver for them.

        With cached dynamics active, a notification whose only flag among the
        cached inputs is :attr:`~newton.ModelFlags.JOINT_PROPERTIES` keeps the
        cache once the cache's device checksums of every joint frame and axis
        match their stored values. A notification whose flags are not all
        numeric runs that check, and one that finds a moved joint frame, as
        when Isaac Lab re-poses a fixed root through
        :attr:`~newton.Model.joint_X_p`, invalidates every cache instead, while
        a changed joint axis raises. Cached mass factors take a moved frame at
        their next mass refresh, as ordinary dynamics' mass matrix does: a
        rigid re-pose of a fixed root leaves them unchanged up to rounding, and
        :meth:`reset` requests that refresh for the worlds it selects. That
        flag covers the :class:`~newton.Model` joint arrays only: after
        writing :class:`~newton.State` joint coordinates outside :meth:`step`,
        call :meth:`reset` with a world mask or :meth:`notify_state_changed`.
        """
        # Derive the kinematic state first: a body-flag change the solver
        # cannot absorb raises here, before any mask, cache or owner changes.
        kinematic = (
            self._kinematic_state() if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES) else None
        )
        if self._friction_anchors_enabled:
            # The next captured build sorts its pairs again rather than reusing the
            # sorted contact frame across a model change.
            self._friction_patches.forget_frame()
        if self._friction_anchors_enabled and flags & ModelFlags.SHAPE_PROPERTIES:
            # Geometry edits retire affected material points; unrelated shape
            # properties keep their history and live materials are checked per step.
            self._friction_patches.update_geometry(self.model)
        if self._row_packets is not None:
            self._row_packets.validate_notification(flags)
        if self._keyed_rows is not None:
            self._keyed_rows.validate_notification(flags)
        # The owner's proof stores a moved joint frame as its new reference, so it runs
        # after every check that can still refuse the notification.
        tree_kinetic = getattr(self, "_tree_kinetic_state", None)
        frames_proven = tree_kinetic is not None and tree_kinetic.validate_model(flags)
        self._forget_repair_continuity()
        cache_inputs = int(flags) & int(
            ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        if frames_proven and cache_inputs == int(ModelFlags.JOINT_PROPERTIES):
            # The owner's device checksums of every joint frame and axis just
            # matched (an exact host fingerprint on CPU), and stepping never reads
            # Model.joint_q: every cache input is as it was. Isaac Lab notifies so
            # on each fixed-root pose write. A moved frame leaves the flag in place
            # and invalidates every cache below.
            cache_inputs = 0
        elif tree_kinetic is not None and not frames_proven:
            from .tree_kinetic_owner import proof_runs  # noqa: PLC0415

            if proof_runs(flags):
                # The owner's proof ran and found a moved joint frame (a changed axis
                # raises in every group's proof): invalidate every cache, whichever flag
                # announced the move.
                cache_inputs |= int(ModelFlags.JOINT_PROPERTIES)
        if tree_kinetic is not None and cache_inputs:
            tree_kinetic.invalidate()
        if self._fk_id_cache_enabled and cache_inputs:
            self._fk_id_cache_valid.zero_()
        if kinematic is not None:
            self._apply_kinematic_state(kinematic)
            self._scatter_armature_to_groups()
            self._mass_update_requested.fill_(1)
        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES and self.model.body_count:
            # Re-derive the COM frames baked from body_com in _allocate_common_buffers so runtime
            # CoM randomization reaches FK; mass and inertia are read live.
            wp.launch(
                compute_com_transforms,
                self.model.body_count,
                inputs=[self.model.body_com],
                outputs=[self.body_X_com],
                device=self.model.device,
            )
            self._mass_update_requested.fill_(1)
        if flags & (
            ModelFlags.BODY_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
        ):
            self._clear_warmstart_history(None)

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Invalidate cached dynamics and clear warm-start state and friction-anchor history for reset worlds.

        The authored simulation state is preserved. Solver-owned state derived
        from it, along with dense and matrix-free impulse history, the carried
        friction anchors, and overflow status, is cleared. Mass
        factors for selected worlds refresh on the next step after a teleport;
        other worlds retain their normal refresh cadence.

        Args:
            state: Simulation state, which is left unchanged.
            world_mask: Optional per-world reset mask. ``None`` resets every world.
            flags: State flags, which do not affect solver-owned impulse history.
        """
        del flags
        if self.contact_compliance:
            self._compliant_contacts = None
            self._compliant_prepared = False
            self.compliance_contact_count = 0
            self.compliance_skipped_contact_count = 0
        if world_mask is not None and world_mask.shape[0] != self.world_count:
            raise ValueError(
                f"world_mask has length {world_mask.shape[0]}, expected {self.world_count} (one entry per world)."
            )
        if self.world_count == 0:
            return

        if getattr(self, "_tree_kinetic_state", None) is not None:
            self._tree_kinetic_state.invalidate(world_mask)

        wp.launch(
            _reset_solver_status,
            dim=max(self.world_count, self.model.articulation_count),
            inputs=[world_mask, self.art_to_world, self._mass_update_requested, self.constraint_overflow],
            device=self.model.device,
        )
        if self._friction_anchors_enabled:
            # Use the world associated with each stored patch anchor.
            self._friction_patches.reset(world_mask)
        if self._fk_id_cache_enabled:
            wp.launch(
                invalidate_articulation_fk_id_cache,
                dim=self.model.articulation_count,
                inputs=[world_mask, self.art_to_world],
                outputs=[self._fk_id_cache_valid],
                device=self.model.device,
            )
        if (
            self._fk_id_cache_enabled
            and self._fk_id_cache_source_state is not None
            and state is not self._fk_id_cache_source_state
            and not getattr(getattr(self, "_tree_kinetic_state", None), "bank_compare", False)
        ):
            self._fk_id_cache_valid.zero_()
            self._fk_id_cache_source_state = state
        self._clear_warmstart_history(world_mask)

    def _clear_warmstart_history(self, world_mask: wp.array | None) -> None:
        """Discard solver impulses independently of latched capacity status."""
        if self.world_count == 0:
            return
        prev_mf_impulses = self._ws_prev_mf_impulses
        if not self.pgs_warmstart and prev_mf_impulses is None:
            return

        dense_impulses = self.impulses if self.pgs_warmstart else self._dummy_contact_impulses
        if prev_mf_impulses is None:
            prev_mf_impulses = self._dummy_contact_impulses
        prev_mf_row_type = self._ws_prev_mf_row_type
        prev_mf_row_parent = self._ws_prev_mf_row_parent
        if prev_mf_row_type is None:
            # Use the dedicated reset sink, not _dummy_contact_row_type: the
            # kernel writes -1 into this buffer and the shared dummy must stay
            # all zeros for its other consumers.
            prev_mf_row_type = self._reset_dummy_mf_row_type
        if prev_mf_row_parent is None:
            prev_mf_row_parent = self._reset_dummy_mf_row_parent
        # _ws_prev_slot_sorted is intentionally NOT cleared here:
        # gather_mf_warmstart only seeds impulses from prev slots whose
        # prev_mf_row_type marks a contact row, so the row_type = -1 sentinel
        # written below (together with the zeroed impulses) neutralizes any
        # stale prev-slot mapping for reset worlds.
        wp.launch(
            reset_world_warmstart_buffers,
            dim=self.world_count,
            inputs=[world_mask, dense_impulses, prev_mf_impulses, prev_mf_row_type, prev_mf_row_parent],
            device=self.model.device,
        )
        if self.pgs_warmstart and self._ws_prev_dense_impulses is not None:
            # Neutralize the dense identity-matched carry for reset worlds
            # (the kernel is width-generic: it zeroes impulses and writes -1
            # into the row-type table).
            wp.launch(
                reset_world_warmstart_buffers,
                dim=self.world_count,
                inputs=[
                    world_mask,
                    self._dummy_contact_impulses,
                    self._ws_prev_dense_impulses,
                    self._ws_prev_dense_row_type,
                    self._ws_prev_dense_row_parent,
                ],
                device=self.model.device,
            )

    def _sync_timing_active(self) -> bool:
        if not _FPGS_SYNC_TIMINGS or _FPGS_SYNC_TIMINGS_COUNT <= 0:
            return False
        step = getattr(self, "_step", -1)
        return _FPGS_SYNC_TIMINGS_START <= step < _FPGS_SYNC_TIMINGS_START + _FPGS_SYNC_TIMINGS_COUNT

    @contextmanager
    def _sync_timed(self, label: str):
        if not self._sync_timing_active():
            yield
            return

        wp.synchronize()
        t0 = time.perf_counter()
        yield
        wp.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[fpgs-sync] step={getattr(self, '_step', -1)} {label}={dt_ms:.3f} ms",
            flush=True,
        )

    def _compute_articulation_metadata(self, model):
        self._compute_articulation_indices(model)
        self._compute_root_free_metadata(model)
        self._setup_size_grouping(model)
        self._setup_world_mapping(model)
        self._build_body_maps(model)
        self._classify_free_rigid_bodies(model)
        self._compact_contact_jacobian = bool(
            model.device.is_cuda
            and not model.requires_grad
            and self.body_response_dof_mask is not None
            and self.size_groups
            and max(self.size_groups) <= _CONTACT_JACOBIAN_MAX_DOF
        )
        self._setup_crba_topology_schedules(model)
        self._setup_world_size_grouping(model)

    def _build_mimic_plan(self, model) -> None:
        """Precompute the static per-mimic lookup tables for the MIMIC row family.

        Resolves ``Model.constraint_mimic_*`` joint indices into articulation / DOF /
        coordinate indices and validates the supported subset: both joints 1-DoF
        (REVOLUTE or PRISMATIC), both in the same articulation, distinct DOFs — matching
        SolverMuJoCo's own restriction. Invalid entries are warned about and masked off;
        ``coef0``/``coef1``/``enabled`` stay live reads from the model arrays so runtime
        mutation needs no re-initialization. All buffers are allocated here, once, so the
        per-step launches are CUDA-graph-safe.
        """
        self._mimic_count = 0
        self._mimic_valid_np = None
        self._mimic_world_np = None
        self._mimic_art_start_np = None
        self._mimic_art_start = None
        self._mimic_art_list = None
        self._mimic_sizes = frozenset()
        self.mimic_slot = None
        n = int(getattr(model, "constraint_mimic_count", 0) or 0)
        if n == 0 or not model.articulation_count or self.art_to_world is None:
            return

        j0 = model.constraint_mimic_joint0.numpy().astype(np.int32, copy=False)
        j1 = model.constraint_mimic_joint1.numpy().astype(np.int32, copy=False)
        articulation_start = model.articulation_start.numpy().astype(np.int64, copy=False)
        joint_type = model.joint_type.numpy().astype(np.int32, copy=False)
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_q_start = model.joint_q_start.numpy().astype(np.int32, copy=False)
        labels = list(getattr(model, "constraint_mimic_label", []) or [])

        one_dof = (int(JointType.REVOLUTE), int(JointType.PRISMATIC))
        articulation_bounds = articulation_start[: model.articulation_count + 1]
        a0, a1 = host_index.lookup(articulation_bounds, j0), host_index.lookup(articulation_bounds, j1)
        spans = (a0 < 0) | (a1 < 0) | (a0 != a1)
        # Joint checks apply to the constraints within one articulation, in order.
        within = np.flatnonzero(~spans)
        one_dof_pair = np.zeros(n, dtype=bool)
        one_dof_pair[within] = np.isin(joint_type[j0[within]], one_dof) & np.isin(joint_type[j1[within]], one_dof)
        self_coupled = np.zeros(n, dtype=bool)
        self_coupled[within] = joint_qd_start[j0[within]] == joint_qd_start[j1[within]]
        valid = (~spans & one_dof_pair & ~self_coupled).astype(np.int32)
        art = np.where(valid != 0, a0, -1).astype(np.int32)
        for k in np.flatnonzero(valid == 0):
            name = labels[k] if k < len(labels) else f"mimic_{k}"
            if spans[k]:
                warnings.warn(
                    f"SolverFeatherPGS: mimic constraint '{name}' spans articulations "
                    f"({int(a0[k])} vs {int(a1[k])}); cross-articulation mimic is unsupported and the "
                    "constraint is ignored.",
                    stacklevel=2,
                )
            elif not one_dof_pair[k]:
                warnings.warn(
                    f"SolverFeatherPGS: mimic constraint '{name}' references a non-1-DoF "
                    "joint (only REVOLUTE/PRISMATIC are supported); the constraint is ignored.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"SolverFeatherPGS: mimic constraint '{name}' couples a DOF to itself; ignored.", stacklevel=2
                )

        valid_indices = np.flatnonzero(valid).astype(np.int32, copy=False)
        art_start, order = host_index.group_by(art[valid_indices], model.articulation_count)
        art_start = art_start.astype(np.int32)
        order = valid_indices[order]
        # A mimic row is reserved and solved in its articulation's solve world: a global articulation (and its
        # mimic, world -1 in the model) solves in world 0.
        mimic_world = np.zeros(n, dtype=np.int32)
        mimic_world[valid_indices] = self._model_plan.articulation_world[art[valid_indices]]

        self._mimic_valid_np = valid
        self._mimic_world_np = mimic_world
        self._mimic_art_start_np = art_start
        self._mimic_sizes = frozenset(
            int(self._model_plan.response_dof_count[a])
            for a in np.unique(art[valid_indices])
            if self._model_plan.response_dof_count[a] > 0
        )
        self._arrays.upload(
            MIMIC_PLAN,
            _mimic_art_start=art_start,
            _mimic_art_list=order,
            _mimic_valid=valid,
            _mimic_world=mimic_world,
            _mimic_dof0=joint_qd_start[j0],
            _mimic_dof1=joint_qd_start[j1],
            _mimic_q0=joint_q_start[j0],
            _mimic_q1=joint_q_start[j1],
        )
        self._arrays.allocate(MIMIC_SLOTS, mimic=n)
        self._mimic_count = n

    def _build_connect_plan(self, model) -> None:
        """Precompute the static per-closure lookup tables for the CONNECT row family.

        Loop-closing BALL joints are excluded from FK/ID by
        :meth:`_FeatherPGSModelPlan.build` and enforced here as three bilateral rows
        pinning the joint's parent/child anchors together. Their stored owner comes from
        the child body's tree articulation, independent of joint ordering. FIXED (weld)
        loop joints warn and are ignored for now. All buffers are allocated once
        (CUDA-graph-safe); ``joint_enabled`` is snapshotted at init and may be changed
        later through :meth:`set_loop_joint_enabled`.

        A closure's parent may live outside the child's articulation when it is
        prescribed: a kinematic body (:attr:`~newton.BodyFlags.KINEMATIC`) or the world
        (``parent == -1``). Such closures contribute no parent DOFs; the parent anchor
        velocity enters the row target instead (a compliant attachment to a driven hand,
        for example). Dynamic cross-articulation closures still warn and are ignored.
        """
        self._connect_count = 0
        self.connect_slot = None
        self._connect_joint_to_index: dict[int, int] = {}
        if not self._has_loop_joints:
            self._connect_world_np = None
            return
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_X_p = model.joint_X_p.numpy()
        joint_X_c = model.joint_X_c.numpy()
        joint_enabled_arr = getattr(model, "joint_enabled", None)
        joint_enabled = joint_enabled_arr.numpy() if joint_enabled_arr is not None else None
        articulation_world = self._model_plan.articulation_world
        loop_joint_articulation = self._model_plan.loop_joint_articulation
        body_articulation = self.body_to_articulation.numpy() if self.body_to_articulation is not None else None
        kinematic_bodies = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0

        loops = np.flatnonzero(loop_joint_articulation >= 0)
        art = loop_joint_articulation[loops]
        ball = joint_type[loops] == int(JointType.BALL)
        parent = joint_parent[loops]
        has_parent = parent >= 0
        parent_art = np.full(loops.size, -1, dtype=np.int32)
        if body_articulation is not None:
            parent_art[has_parent] = body_articulation[parent[has_parent]]
        # A closure to another articulation is enforced only from a kinematic parent.
        cross = has_parent & (parent_art != art)
        dynamic_cross = np.zeros(loops.size, dtype=bool)
        dynamic_cross[cross] = ~kinematic_bodies[parent[cross]]
        for i in np.flatnonzero(~ball | dynamic_cross):
            j = loops[i]
            if not ball[i]:
                warnings.warn(
                    f"SolverFeatherPGS: loop joint {j} has unsupported type "
                    f"{int(joint_type[j])} (only BALL/connect closures are enforced); ignored.",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"SolverFeatherPGS: loop joint {j} closes across articulations ({int(parent_art[i])} vs "
                    f"{int(art[i])}) and its parent is not kinematic; ignored.",
                    stacklevel=2,
                )
        enforced = ball & ~dynamic_cross
        joints = loops[enforced]

        n = int(joints.size)
        if n == 0:
            self._connect_world_np = None
            self._connect_joint_to_index = {}
            return
        self._connect_joint_to_index = {int(j): index for index, j in enumerate(joints)}
        art = art[enforced]
        self._connect_world_np = articulation_world[art].astype(np.int32)
        self._connect_valid_np = (
            np.ones(n, dtype=np.int32) if joint_enabled is None else (joint_enabled[joints] != 0).astype(np.int32)
        )
        self._connect_parent_prescribed_np = (~has_parent | cross)[enforced].astype(np.int32)
        self._connect_enabled_np = self._connect_valid_np.copy()
        self._connect_anchor_p_np = joint_X_p[joints, :3].astype(np.float32)
        self._connect_anchor_c_np = joint_X_c[joints, :3].astype(np.float32)
        self._arrays.upload(
            CONNECT_PLAN,
            _connect_art=art.astype(np.int32),
            _connect_body_p=parent[enforced].astype(np.int32),
            _connect_body_c=joint_child[joints].astype(np.int32),
            _connect_anchor_p=self._connect_anchor_p_np,
            _connect_anchor_c=self._connect_anchor_c_np,
            _connect_parent_prescribed=self._connect_parent_prescribed_np,
            _connect_world=self._connect_world_np,
            _connect_valid=np.ones(n, dtype=np.int32),
            _connect_enabled=self._connect_enabled_np,
        )
        self._arrays.allocate(CONNECT_SLOTS, closure=n)
        self._connect_count = n

    def _connect_index(self, joint: int) -> int:
        index = getattr(self, "_connect_joint_to_index", {}).get(int(joint))
        if index is None:
            raise ValueError(
                f"SolverFeatherPGS: joint {joint} is not an enforced loop-closing BALL joint of this solver."
            )
        return index

    def set_loop_joint_enabled(self, joint: int, enabled: bool) -> None:
        """Enable or disable the connect rows of a loop-closing BALL joint at runtime.

        The closure keeps its allocated capacity, so toggling is CUDA-graph-safe and
        takes effect on the next :meth:`step`. Closures disabled through
        :attr:`~newton.Model.joint_enabled` at construction still count as capacity.

        Args:
            joint: Model joint index of a loop-closing BALL joint.
            enabled: ``True`` to enforce the closure, ``False`` to release it.
        """
        index = self._connect_index(joint)
        self._connect_enabled_np[index] = 1 if enabled else 0
        self._connect_enabled.assign(self._connect_enabled_np)

    def set_loop_joint_anchors(self, joint: int, parent_anchor, child_anchor) -> None:
        """Move the anchors of a loop-closing BALL joint at runtime.

        Both anchors are body-local points [m] (world-frame for a world parent). Use it
        to re-target a closure before enabling it, for example to attach a body to a
        prescribed hand at the relative pose measured at acquisition.

        Args:
            joint: Model joint index of a loop-closing BALL joint.
            parent_anchor: Anchor in the parent body frame [m].
            child_anchor: Anchor in the child body frame [m].
        """
        index = self._connect_index(joint)
        self._connect_anchor_p_np[index] = np.asarray(parent_anchor, dtype=np.float32)
        self._connect_anchor_c_np[index] = np.asarray(child_anchor, dtype=np.float32)
        self._connect_anchor_p.assign(self._connect_anchor_p_np)
        self._connect_anchor_c.assign(self._connect_anchor_c_np)

    def loop_joint_anchors(self, joint: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the current ``(parent_anchor, child_anchor)`` of a loop-closing BALL joint [m]."""
        index = self._connect_index(joint)
        return self._connect_anchor_p_np[index].copy(), self._connect_anchor_c_np[index].copy()

    def _forget_repair_continuity(self) -> None:
        """Launch the held-tree owner's next repair after a host call that changes its validity or inputs."""
        owner = getattr(self, "_tree_kinetic_state", None)
        if owner is not None:
            owner.forget_continuity()

    def _track_step_dt(self, dt: float) -> None:
        """Record the step duration; a change refreshes mass and invalidates the FK/ID cache of every articulation."""
        if self._last_step_dt is not None and abs(self._last_step_dt - dt) > 1.0e-8:
            self._force_mass_update = True
            if self._fk_id_cache_enabled:
                self._fk_id_cache_valid.zero_()
            self._forget_repair_continuity()
        self._last_step_dt = dt

    def notify_state_changed(self, articulations: wp.array | None = None) -> None:
        """Invalidate cached kinematics after writing joint state outside :meth:`step`.

        The held tree caches forward kinematics and inverse dynamics of the state it
        produced (ordinary dynamics derive them from the state at every step). Writes to
        ``joint_q``/``joint_qd`` of the next input state between steps (for example to
        prescribe kinematic bodies) must be notified here: the held tree sees an
        un-notified write one step late.

        Args:
            articulations: Optional device array of articulation indices whose joint
                state changed. ``None`` invalidates every articulation.
        """
        if not self._fk_id_cache_enabled:
            return
        self._forget_repair_continuity()
        if articulations is None:
            self._fk_id_cache_valid.zero_()
            return
        wp.launch(
            invalidate_fk_id_cache_for_articulations,
            dim=articulations.shape[0],
            inputs=[articulations],
            outputs=[self._fk_id_cache_valid],
            device=self.model.device,
        )

    def _build_preelimination_plan(self, model) -> None:
        """Precompute the static tables for bilateral (mimic + connect) pre-elimination.

        Builds, per articulation that owns bilateral rows: an index into the packed
        per-elimination buffers, plus the S-factor scratch. All buffers are allocated
        once here (CUDA-graph-safe); the slot lists and the S factor are rebuilt every
        step on device because both the row slots and J_B depend on the current state.
        Unsupported configurations warn and fall back to iterative bilateral rows.
        """
        self._preelim_count = 0
        self._preelim_active = False
        n_bilateral = int(self._connect_count or 0)
        if self.bilateral_preelimination_include_mimics:
            n_bilateral += int(self._mimic_count or 0)
        if not self.enable_bilateral_preelimination or n_bilateral == 0:
            return
        if self.pgs_mode != "matrix_free":
            warnings.warn(
                "SolverFeatherPGS: enable_bilateral_preelimination requires "
                f"pgs_mode='matrix_free' (got {self.pgs_mode!r}); falling back to "
                "iterative mimic/connect rows.",
                stacklevel=2,
            )
            return
        if self.pgs_velocity_iterations > 0:
            warnings.warn(
                "SolverFeatherPGS: enable_bilateral_preelimination does not support "
                "pgs_velocity_iterations > 0 yet (the velocity pass rebuilds v_out "
                "without the bilateral projection); falling back to iterative rows.",
                stacklevel=2,
            )
            return

        if self._connect_count and np.any(self._connect_parent_prescribed_np != 0):
            warnings.warn(
                "SolverFeatherPGS: enable_bilateral_preelimination does not support closures "
                "with a prescribed (kinematic or world) parent yet; falling back to iterative "
                "mimic/connect rows.",
                stacklevel=2,
            )
            return

        # Per-articulation bilateral row capacity (upper bound: every valid constraint).
        counts = np.zeros(model.articulation_count, dtype=np.int64)
        if self._mimic_count and self.bilateral_preelimination_include_mimics:
            counts += np.diff(self._mimic_art_start_np)
        if self._connect_count:
            counts += 3 * np.bincount(self._connect_art.numpy(), minlength=model.articulation_count)
        arts = np.flatnonzero(counts)
        if not arts.size:
            return
        n_b_max = int(counts.max())
        if n_b_max > PREELIM_MAX_ROWS:
            warnings.warn(
                f"SolverFeatherPGS: an articulation owns {n_b_max} bilateral rows, more "
                f"than the pre-elimination capacity of {PREELIM_MAX_ROWS}; falling back "
                "to iterative mimic/connect rows.",
                stacklevel=2,
            )
            return

        device = model.device
        n_pe = len(arts)
        art_to_idx = np.full(model.articulation_count, -1, dtype=np.int32)
        art_to_idx[arts] = np.arange(n_pe)
        self._preelim_art_to_idx = wp.array(art_to_idx, dtype=wp.int32, device=device)
        self._arrays.allocate(
            PREELIMINATION,
            preelim=n_pe,
            preelim_row=n_pe * PREELIM_MAX_ROWS,
            preelim_block=n_pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS,
        )
        self._preelim_reg_rel = 1.0e-3
        self._preelim_reg_floor = max(self.pgs_cfm, 1.0e-7)
        self._preelim_count = n_pe
        self._preelim_active = True

    def _stage4_bilateral_preelim(self) -> None:
        """Pre-eliminate the bilateral blocks on the packed rows (:data:`.response_builds_rows.ELIMINATION`).

        Form and factor each articulation's block ``S = Z_B^T Z_B + R``, then correct every other row, its
        right-hand side and its diagonal. Runs once the right-hand sides are final (bias and restitution) and
        before the warm start, whose impulses the corrected rows then carry.
        """
        dense = int(self._packed_dense)
        for size in self.size_groups:
            wp.launch(
                response_builds_rows.bilateral_setup_kernel,
                dim=self.n_arts_by_size[size],
                inputs=[
                    field_group.bind(Preelim, solver=self),
                    self.group_to_art[size],
                    self.art_to_world,
                    self.articulation_world_dof_offset,
                    dense,
                    self.mimic_slot if self.mimic_slot is not None else self._dummy_is_free_rigid,
                    self._mimic_art_start if self._mimic_count else self._dummy_is_free_rigid,
                    self._mimic_art_list if self._mimic_count else self._dummy_is_free_rigid,
                    int(self._mimic_count or 0) if self.bilateral_preelimination_include_mimics else 0,
                    self.connect_slot if self.connect_slot is not None else self._dummy_is_free_rigid,
                    self._connect_art if self._connect_count else self._dummy_is_free_rigid,
                    int(self._connect_count or 0),
                    self._packed_J,
                    self._packed_Z,
                    size,
                ],
                device=self.model.device,
            )
        wp.launch(
            response_builds_rows.bilateral_correct_kernel,
            dim=self.world_count * self.dense_max_constraints,
            inputs=[
                field_group.bind(Preelim, solver=self),
                self._row_sides,
                self.articulation_response_dof_count,
                self.articulation_world_dof_offset,
                dense,
                self.constraint_count,
                self._local_solve_owner,
                self.row_cfm,
            ],
            outputs=[self._packed_Z, self.rhs, self.diag],
            device=self.model.device,
        )

    def _stage6_project_bilateral_velocity(self) -> None:
        """Enforce J_B v_out = -b_B once, right after v_out is seeded with v_hat: the block weights, then ``v -= P^T
        L^-T Z_B y`` (:func:`.packed_rows.project_kernel`)."""
        dense = int(self._packed_dense)
        for size in self.size_groups:
            wp.launch(
                response_builds_rows.bilateral_weights_kernel,
                dim=self.n_arts_by_size[size],
                inputs=[
                    field_group.bind(Preelim, solver=self),
                    self.group_to_art[size],
                    self.art_to_world,
                    self.articulation_dof_start,
                    self.articulation_world_dof_offset,
                    dense,
                    self._packed_J,
                    self.rhs,
                    self.v_out,
                    size,
                ],
                device=self.model.device,
            )
            wp.launch(
                packed_rows.project_kernel(),
                dim=self.n_arts_by_size[size],
                inputs=[
                    self.L_by_size[size],
                    self.group_to_art[size],
                    self.art_to_world,
                    self.articulation_dof_start,
                    self.articulation_world_dof_offset,
                    size,
                    dense,
                    self._preelim_art_to_idx,
                    self._preelim_slots,
                    self._preelim_nB,
                    self._preelim_weight,
                    self._packed_Z,
                ],
                outputs=[self.v_out],
                device=self.model.device,
            )

    def _setup_passive_joint_forces(self, model) -> None:
        """Select the passive spring/damping arrays applied in the inverse-dynamics pass.

        The toggles resolve here, once, to fixed device arrays (zeros when disabled or
        absent on the model), so inverse-dynamics launches stay CUDA-graph-safe. Springs
        on unsupported DOF types (BALL / FREE / DISTANCE coordinates) warn and are masked
        off — only REVOLUTE, PRISMATIC, and D6 DOFs receive the explicit spring torque.
        """
        device = model.device
        n_dofs = int(model.joint_dof_count)

        def _zeros():
            return self._arrays.placeholder(wp.float32, (n_dofs,))

        spring_k = getattr(model, "joint_spring_stiffness", None)
        spring_ref = getattr(model, "joint_spring_ref", None)
        if spring_k is None or spring_ref is None:
            self._passive_spring_stiffness = _zeros()
            self._passive_spring_ref = _zeros()
        else:
            k_np = spring_k.numpy()
            if np.any(k_np != 0.0) and model.joint_count:
                # Mask springs on unsupported joint types (kernel applies them per 1-DoF axis).
                supported = (int(JointType.REVOLUTE), int(JointType.PRISMATIC), int(JointType.D6))
                joint_type = model.joint_type.numpy()
                # joint_qd_start is sentinel-closed (length joint_count + 1).
                dof_joint, _ = host_index.expand(np.diff(model.joint_qd_start.numpy()))
                bad = (k_np != 0.0) & ~np.isin(joint_type[dof_joint], supported)
                if np.any(bad):
                    warnings.warn(
                        f"SolverFeatherPGS: passive joint springs on {int(bad.sum())} DOF(s) of "
                        "unsupported joint types (only REVOLUTE/PRISMATIC/D6 are supported); "
                        "those springs are ignored.",
                        stacklevel=2,
                    )
                    k_np = k_np.copy()
                    k_np[bad] = 0.0
                    spring_k = wp.array(k_np, dtype=wp.float32, device=device)
            self._passive_spring_stiffness = spring_k
            self._passive_spring_ref = spring_ref

        damping = getattr(model, "joint_damping", None)
        if damping is None:
            self._passive_joint_damping = _zeros()
        else:
            self._passive_joint_damping = damping

    def _setup_fused_diagonal_joint_limits(self, model) -> None:
        """Select independent position limits that can remain outside the dense row system."""
        self._fused_diagonal_joint_limits = False
        self._fused_diagonal_joint_limit_sizes = frozenset()
        self._fused_diagonal_limit_dof_mask_host = np.zeros(model.joint_dof_count, dtype=np.int32)
        supported = bool(
            model.device.is_cuda
            and not model.requires_grad
            and model.joint_dof_count > 0
            and self.enable_joint_limits
            and self.pgs_mode == "matrix_free"
            and self.pgs_schedule == "interleaved"
            and self.drive_mode == "augmented"
            and self.use_parallel_streams
            and self.pgs_velocity_iterations == 0
            and self.enable_contact_friction
            and not self.enable_joint_velocity_limits
            and not self.pgs_warmstart
            and not self._mf_warmstart_enabled
            and not self._preelim_active
            and not self._has_free_rigid_bodies
            and not self._debug_buffers_enabled
            and not self._regularization_enabled
            and self._mimic_count == 0
            and self._connect_count == 0
        )
        if not supported:
            return

        candidate_sizes = self._diagonal_mass_sizes.intersection(self._joint_limit_sizes)
        if not candidate_sizes:
            return
        response_dof_count = self._model_plan.response_dof_count
        articulation_world = self._model_plan.articulation_world
        articulation_dof_start = self._model_plan.articulation_dof_start
        pairs = _world_diagonal_pairs(articulation_world, response_dof_count, self.world_count, candidate_sizes)
        if pairs is None:
            return
        diagonal_articulations, dense_articulations = pairs
        dense_size = response_dof_count[dense_articulations]
        if np.any((dense_size <= 0) | (dense_size > _CONTACT_JACOBIAN_MAX_DOF)):
            return

        # The diagonal articulations with a limited DOF fuse all their DOFs' limits.
        diagonal_size = response_dof_count[diagonal_articulations]
        art, local_dof = host_index.expand(diagonal_size)
        dof = articulation_dof_start[diagonal_articulations][art] + local_dof
        valid = self._joint_limit_q_index_host[dof] >= 0
        limited = np.bincount(art[valid], minlength=diagonal_articulations.size) > 0
        self._fused_diagonal_limit_dof_mask_host[dof[limited[art]]] = valid[limited[art]]
        # Add each size where a world first has it, as the per-world set did.
        selected_sizes: set[int] = {int(size) for size in host_index.ordered_unique(diagonal_size[limited])}
        self._fused_diagonal_joint_limit_sizes = frozenset(selected_sizes)
        self._fused_diagonal_joint_limits = bool(selected_sizes)

    def _estimate_dense_internal_rows_per_world(self, model) -> int:
        if not model.articulation_count or not model.joint_count or self.art_to_world is None:
            return 0

        has_drive_rows = self.drive_mode == "physx_pgs" and model.joint_dof_count > 0
        has_position_limit_rows = self.enable_joint_limits and model.joint_dof_count > 0
        has_velocity_limit_rows = (
            self.enable_joint_velocity_limits
            and model.joint_dof_count > 0
            and not np.isinf(self.velocity_limit_activation_fraction)
        )
        has_mimic_rows = getattr(self, "_mimic_count", 0) > 0
        has_connect_rows = getattr(self, "_connect_count", 0) > 0
        if not (
            has_drive_rows or has_position_limit_rows or has_velocity_limit_rows or has_mimic_rows or has_connect_rows
        ):
            return 0

        art_to_world = self.art_to_world.numpy().astype(np.int32, copy=False)
        per_world = np.zeros(max(int(self.world_count), 1), dtype=np.int64)

        if has_mimic_rows and self._mimic_world_np is not None:
            valid_worlds = self._mimic_world_np[self._mimic_valid_np != 0]
            in_range = valid_worlds[(valid_worlds >= 0) & (valid_worlds < per_world.shape[0])]
            np.add.at(per_world, in_range, 1)
        if has_connect_rows and self._connect_world_np is not None:
            valid_worlds = self._connect_world_np[self._connect_valid_np != 0]
            in_range = valid_worlds[(valid_worlds >= 0) & (valid_worlds < per_world.shape[0])]
            np.add.at(per_world, in_range, 3)

        articulation_start = model.articulation_start.numpy().astype(np.int32, copy=False)
        joint_type = model.joint_type.numpy().astype(np.int32, copy=False)
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_dof_dim = model.joint_dof_dim.numpy().astype(np.int32, copy=False)

        joint_target_ke_arr = getattr(model, "joint_target_ke", None)
        joint_target_kd_arr = getattr(model, "joint_target_kd", None)
        if has_drive_rows and (joint_target_ke_arr is None or joint_target_kd_arr is None):
            has_drive_rows = False
        joint_limit_lower_arr = getattr(model, "joint_limit_lower", None)
        joint_limit_upper_arr = getattr(model, "joint_limit_upper", None)
        if has_position_limit_rows and (joint_limit_lower_arr is None or joint_limit_upper_arr is None):
            has_position_limit_rows = False
        joint_velocity_limit_arr = getattr(model, "joint_velocity_limit", None)
        if has_velocity_limit_rows and joint_velocity_limit_arr is None:
            has_velocity_limit_rows = False
        if not (has_drive_rows or has_position_limit_rows or has_velocity_limit_rows):
            return 0

        joint_target_ke = joint_target_ke_arr.numpy() if has_drive_rows else None
        joint_target_kd = joint_target_kd_arr.numpy() if has_drive_rows else None
        joint_limit_lower = joint_limit_lower_arr.numpy() if has_position_limit_rows else None
        joint_limit_upper = joint_limit_upper_arr.numpy() if has_position_limit_rows else None
        joint_velocity_limit = joint_velocity_limit_arr.numpy() if has_velocity_limit_rows else None

        dense_joint_types = (
            int(JointType.PRISMATIC),
            int(JointType.REVOLUTE),
            int(JointType.D6),
        )

        # Every axis of every dense joint in each articulation's joint range, by articulation.
        articulation_count = int(model.articulation_count)
        first_joint = articulation_start[:articulation_count]
        art, position = host_index.expand(np.maximum(articulation_start[1 : articulation_count + 1] - first_joint, 0))
        joint = first_joint[art] + position
        dense = np.isin(joint_type[joint], dense_joint_types)
        art, joint = art[dense], joint[dense]
        axis_joint, axis = host_index.expand(joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1])
        art = art[axis_joint]
        dof = joint_qd_start[joint][axis_joint] + axis

        rows = np.zeros(dof.size, dtype=np.int64)
        if has_drive_rows:
            rows += (joint_target_ke[dof] > 0.0) | (joint_target_kd[dof] > 0.0)
        if has_position_limit_rows:
            fuse_position_limits = np.isin(
                self._model_plan.response_dof_count[art], list(self._fused_diagonal_joint_limit_sizes)
            )
            rows += ~fuse_position_limits & np.isfinite(joint_limit_lower[dof])
            rows += ~fuse_position_limits & np.isfinite(joint_limit_upper[dof])
        if has_velocity_limit_rows:
            rows += 2 * (joint_velocity_limit[dof] > 0.0)
        world = art_to_world[art]
        in_range = (world >= 0) & (world < per_world.size)
        np.add.at(per_world, world[in_range], rows[in_range])

        return int(np.max(per_world)) if per_world.size else 0

    def _compute_articulation_indices(self, model):
        # calculate total size and offsets of Jacobian and mass matrices for entire system
        if model.joint_count:
            articulation_count = model.articulation_count
            articulation_start = model.articulation_start.numpy()
            joint_q_start = model.joint_q_start.numpy()
            if self._model_plan is None:
                raise RuntimeError("FeatherPGS model plan must be built before articulation metadata")
            first_joint = articulation_start[:articulation_count]
            joint_count = articulation_start[1 : articulation_count + 1] - first_joint
            dof_count = self._model_plan.articulation_dof_count

            # Jacobian sizes accumulate in int32, as they always have. Mass matrix
            # offsets reach wp.array as Python ints, so one past int32 still raises.
            J_bounds = host_index.offsets(6 * joint_count * dof_count, dtype=np.int32)
            M_bounds = host_index.offsets(joint_count * 36)
            H_bounds = host_index.offsets(dof_count.astype(np.int64) ** 2)
            self.J_size = J_bounds[-1] if articulation_count else 0
            self.M_size = int(M_bounds[-1])
            self.H_size = int(H_bounds[-1])

            # matrix offsets for grouped gemm
            self.articulation_J_start = wp.array(J_bounds[:-1], dtype=wp.int32, device=model.device)
            self.articulation_M_start = wp.array(M_bounds[:-1].tolist(), dtype=wp.int32, device=model.device)
            self.articulation_H_start = wp.array(H_bounds[:-1].tolist(), dtype=wp.int32, device=model.device)

            self.articulation_M_rows = wp.array(joint_count * 6, dtype=wp.int32, device=model.device)
            self.articulation_H_rows = wp.array(dof_count, dtype=wp.int32, device=model.device)
            self.articulation_J_rows = wp.array(joint_count * 6, dtype=wp.int32, device=model.device)
            self.articulation_J_cols = wp.array(dof_count, dtype=wp.int32, device=model.device)

            self.articulation_dof_start = wp.array(
                self._model_plan.articulation_dof_start, dtype=wp.int32, device=model.device
            )
            self.articulation_coord_start = wp.array(joint_q_start[first_joint], dtype=wp.int32, device=model.device)

            self.articulation_max_dofs = int(np.max(dof_count)) if articulation_count else 0
        else:
            self.M_size = 0
            self.articulation_max_dofs = 0

    def _compute_root_free_metadata(self, model):
        if model.joint_count:
            joint_type = model.joint_type.numpy()
            joint_parent = model.joint_parent.numpy()
            root_mask = ((joint_type == JointType.FREE) | (joint_type == JointType.DISTANCE)) & (joint_parent < 0)
            free_root_joint_indices = np.flatnonzero(root_mask).astype(np.int32, copy=False)
            self._free_root_joint_count = int(free_root_joint_indices.size)
            self._free_root_joint_indices = (
                wp.array(free_root_joint_indices, dtype=wp.int32, device=model.device)
                if self._free_root_joint_count
                else None
            )
        else:
            self._free_root_joint_count = 0
            self._free_root_joint_indices = None

        if not model.articulation_count or not model.joint_count:
            self.articulation_root_is_free = None
            self.articulation_root_dof_start = None
            self._has_root_free = False
            return

        root_joint = model.articulation_start.numpy()[: model.articulation_count]
        root_dof_start = model.joint_qd_start.numpy()[root_joint].astype(np.int32, copy=False)
        root_type = joint_type[root_joint]
        root_is_free = (
            ((root_type == int(JointType.FREE)) | (root_type == int(JointType.DISTANCE)))
            & (joint_parent[root_joint] == -1)
        ).astype(np.int32)

        self.articulation_root_is_free = wp.array(root_is_free, dtype=wp.int32, device=model.device)
        self.articulation_root_dof_start = wp.array(root_dof_start, dtype=wp.int32, device=model.device)
        self._has_root_free = bool(np.any(root_is_free != 0))

    def _setup_size_grouping(self, model):
        """Build response-size groups without modifying physical topology."""
        if not model.articulation_count or not model.joint_count:
            self.size_groups = []
            self.n_arts_by_size = {}
            self._free_body_inertia_sizes = frozenset()
            self.articulation_response_dof_count = wp.zeros(
                model.articulation_count, dtype=wp.int32, device=model.device
            )
            self._joint_limit_sizes = frozenset()
            self._joint_limit_q_index = None
            self._joint_limit_q_index_host = np.zeros(model.joint_dof_count, dtype=np.int32)
            self._joint_limit_warp_kernels = {}
            return

        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before size grouping")
        response_dof_count = self._model_plan.response_dof_count
        # Each size group holds its articulations in ascending order, the groups by descending size.
        negated_sizes, group = np.unique(-response_dof_count, return_inverse=True)
        bounds, order = host_index.group_by(group, negated_sizes.size)
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = host_index.segmented_rank(group[order])
        art_group_idx = np.where(response_dof_count > 0, rank, -1).astype(np.int32)
        group_to_art = {
            int(-size): order[bounds[g] : bounds[g + 1]].astype(np.int32)
            for g, size in enumerate(negated_sizes)
            if size < 0
        }
        solve_sizes = list(group_to_art)
        self.size_groups = solve_sizes
        self.n_arts_by_size = {size: int(group_to_art[size].size) for size in solve_sizes}

        self.articulation_response_dof_count = wp.array(response_dof_count, dtype=wp.int32, device=model.device)
        self.art_group_idx = wp.array(art_group_idx, dtype=wp.int32, device=model.device)
        self.group_to_art = {
            size: wp.array(group_to_art[size], dtype=wp.int32, device=model.device) for size in solve_sizes
        }
        # A one-body free articulation has I_composite == I_body. Select the direct
        # source only for whole size groups so the CRBA kernel remains branch-free.
        self._free_body_inertia_sizes = frozenset(
            size for size in solve_sizes if np.all(self._model_plan.is_free_rigid[group_to_art[size]] != 0)
        )

        articulation_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
        limit_joint = np.isin(
            joint_type,
            (int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6)),
        )
        # Limit joints before each articulation's range ends, minus those before it starts.
        limit_joints_before = host_index.offsets(limit_joint)
        art_start = articulation_start[: model.articulation_count]
        art_end = articulation_start[1 : model.articulation_count + 1]
        limited_sizes = response_dof_count[
            (response_dof_count > 0) & (limit_joints_before[art_end] > limit_joints_before[art_start])
        ]
        # Insert each size where an articulation first has it, as the per-articulation set did.
        self._joint_limit_sizes = frozenset(int(size) for size in host_index.ordered_unique(limited_sizes))
        joint_q_start = model.joint_q_start.numpy().astype(np.int32, copy=False)
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_dof_dim = model.joint_dof_dim.numpy().astype(np.int32, copy=False)
        limit_q_index = np.full(model.joint_dof_count, -1, dtype=np.int32)
        limit_joints = np.flatnonzero(limit_joint)
        limit_joint_of, axis = host_index.expand(joint_dof_dim[limit_joints, 0] + joint_dof_dim[limit_joints, 1])
        limit_q_index[joint_qd_start[limit_joints][limit_joint_of] + axis] = (
            joint_q_start[limit_joints][limit_joint_of] + axis
        )
        self._joint_limit_q_index = wp.array(limit_q_index, dtype=wp.int32, device=model.device)
        self._joint_limit_q_index_host = limit_q_index
        self._joint_limit_warp_kernels = (
            {
                size: _get_joint_limit_warp_kernel(size, warps_per_block=_JOINT_LIMIT_WARPS_PER_BLOCK)
                for size in self._joint_limit_sizes
            }
            if model.device.is_cuda and not model.requires_grad
            else {}
        )

    def _setup_world_mapping(self, model):
        """Materialize articulation-to-world mappings from the model plan."""
        if not model.articulation_count:
            self.world_count = 0
            self.art_to_world = None
            self.world_art_start = None
            self._is_one_solve_art_per_world = False
            return
        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before world mapping")

        articulation_world = self._model_plan.articulation_world
        self.world_count = self._model_plan.world_count
        self.art_to_world = wp.array(articulation_world, dtype=wp.int32, device=model.device)
        # Per-articulation end of the kinematic-tree joint prefix: trailing loop-closing
        # joints (imported `connect`/`weld` equalities) are excluded from FK/ID iteration.
        self.articulation_joint_end = wp.array(
            self._model_plan.articulation_joint_end, dtype=wp.int32, device=model.device
        )

        world_art_counts = np.bincount(articulation_world, minlength=self.world_count).astype(np.int32)
        world_art_start = host_index.offsets(world_art_counts, dtype=np.int32)
        self.world_art_start = wp.array(world_art_start, dtype=wp.int32, device=model.device)

        response_world = articulation_world[self._model_plan.response_dof_count > 0]
        world_response_art_counts = np.bincount(response_world, minlength=self.world_count)
        self._is_one_solve_art_per_world = bool(np.all(world_response_art_counts == 1))

    def _setup_world_size_grouping(self, model):
        self.world_response_group_art_start = {}
        self.world_response_group_to_art = {}
        if (
            not model.articulation_count
            or self.art_to_world is None
            or self._model_plan is None
            or self.is_free_rigid is None
        ):
            return

        device = model.device
        art_to_world_np = self.art_to_world.numpy().astype(np.int32, copy=False)
        art_size_np = self._model_plan.response_dof_count.astype(np.int32, copy=False)
        world_count = max(int(self.world_count), 0)
        for size in self.size_groups:
            # The fused diagonal accumulator must retain every solved articulation:
            # each world's, in ascending order, one world after another.
            arts = np.flatnonzero(art_size_np == int(size))
            worlds = art_to_world_np[arts]
            in_range = (worlds >= 0) & (worlds < world_count)
            arts, worlds = arts[in_range], worlds[in_range]
            response_starts, order = host_index.group_by(worlds, world_count)
            response_flat_np = arts[order].astype(np.int32)
            self.world_response_group_art_start[int(size)] = wp.array(response_starts, dtype=wp.int32, device=device)
            self.world_response_group_to_art[int(size)] = wp.array(response_flat_np, dtype=wp.int32, device=device)

    def _setup_crba_topology_schedules(self, model) -> None:
        """Build immutable mass-matrix schedules for topology-homogeneous size groups."""
        self._crba_dof_joint_offset_by_size = {}
        diagonal_mass_sizes: set[int] = set()
        direct_diagonal_inertia_sizes: set[int] = set()
        self._diagonal_mass_sizes = frozenset()
        self._direct_diagonal_inertia_sizes = frozenset()
        if not model.articulation_count or not model.joint_count:
            return

        articulation_start = model.articulation_start.numpy()
        articulation_dof_start = self._model_plan.articulation_dof_start
        response_dof_count = self._model_plan.response_dof_count
        joint_qd_start = model.joint_qd_start.numpy()
        joint_ancestor = model.joint_ancestor.numpy()
        joint_child = model.joint_child.numpy()
        body_single_response_dof = self._body_single_response_dof_host
        response_body_counts = np.bincount(
            body_single_response_dof[body_single_response_dof >= 0], minlength=model.joint_dof_count
        )

        for size in self.size_groups:
            arts = np.flatnonzero(response_dof_count == size)
            joint_start = articulation_start[arts]
            joint_count = np.maximum(articulation_start[arts + 1] - joint_start, 0)
            dof_start = articulation_dof_start[arts]
            # Each DOF's joint, as an offset from the articulation's first joint;
            # a later joint wins a DOF that an earlier one also spans.
            art, position = host_index.expand(joint_count)
            joint = joint_start[art] + position
            local_start = np.maximum(0, joint_qd_start[joint] - dof_start[art])
            local_end = np.minimum(size, joint_qd_start[joint + 1] - dof_start[art])
            entry, local_dof = host_index.expand(np.maximum(local_end - local_start, 0))
            offsets = np.full((arts.size, size), -1, dtype=np.int32)
            np.maximum.at(offsets, (art[entry], local_start[entry] + local_dof), position[entry].astype(np.int32))
            # One schedule serves the group only if every articulation has the first one's topology.
            compatible = bool(
                not np.any(offsets < 0) and np.all(offsets == offsets[0]) and np.all(joint_count == joint_count[0])
            )
            if compatible:
                parents = joint_ancestor[joint_start[:, None] + np.arange(joint_count[0])]
                parents = np.where(parents >= joint_start[:, None], parents - joint_start[:, None], -1).astype(np.int32)
                compatible = bool(np.all(parents == parents[0]))

            if compatible:
                reference_offsets = offsets[0]
                # Entry (row, col) is the inertia of the deeper of two related coordinates' joints, read at its
                # column: col's when row's joint is col's or an ancestor of it, else row's when the reverse holds.
                related = TreePlan.build(parents[0]).ancestor_matrix()[np.ix_(reference_offsets, reference_offsets)]
                coordinates = np.arange(int(size), dtype=np.int32)
                sources = np.where(related.T, coordinates[None, :], np.where(related, coordinates[:, None], -1)).astype(
                    np.int32
                )
                if np.count_nonzero(sources >= 0) == int(size) and np.all(np.diag(sources) >= 0):
                    diagonal_mass_sizes.add(int(size))
                    # Direct when each DOF's child body has that DOF as its single
                    # response DOF, and no other body does.
                    child = joint_child[joint_start[:, None] + reference_offsets]
                    global_dof = dof_start[:, None] + np.arange(size)
                    single_dof = np.where(child >= 0, body_single_response_dof[np.maximum(child, 0)], -1)
                    owned = single_dof == global_dof
                    owned &= response_body_counts[np.where(owned, global_dof, 0)] == 1
                    if np.all(owned):
                        direct_diagonal_inertia_sizes.add(int(size))
                self._crba_dof_joint_offset_by_size[int(size)] = wp.array(
                    reference_offsets, dtype=wp.int32, device=model.device
                )

        self._diagonal_mass_sizes = frozenset(diagonal_mass_sizes)
        self._direct_diagonal_inertia_sizes = frozenset(direct_diagonal_inertia_sizes)

    def _setup_sparse_mass_matrix(self, model: Model) -> None:
        """Select sparse articulation factors with independent rigid-body response blocks."""
        self._sparse_mass_matrix_size = None
        self._sparse_split = False
        self._sparse_velocity_limits = False
        self._sparse_prepare_contacts = False
        self._sparse_row_support = _SparseRowSupport()
        self._sparse_contact_triplets = False
        # Policy: patch friction may adopt the active-row scan's deferral law.
        self._patch_active_rows = bool(
            self._friction_anchors_enabled and os.environ.get("FEATHER_PGS_PATCH_ACTIVE_ROWS", "1") != "0"
        )
        # Route details for get_path_selection(), set where the sparse owner is admitted.
        self._sparse_active_rows = False
        self._sparse_active_rows_reason = "Native sparse factors are not selected."
        self._sparse_contact_triplets_reason = "Native sparse factors are not selected."
        # The admitted triplets' transaction ("single" or "staged") and its records.
        self._sparse_contact_transaction = None
        self._sparse_contact_transaction_reason = "Native sparse factors are not selected."
        self._sparse_contact_records = None
        self._sparse_max_row_support = None
        sparse_choice = getattr(self, "sparse_mass_matrix", "auto")
        if sparse_choice == "off":
            self._sparse_selection_reason = "Explicit sparse_mass_matrix='off' retains ordinary factors."
            return
        if getattr(self, "_explicit_dense_kernels", False):
            self._sparse_selection_reason = "An explicit dense kernel choice owns factor and response construction."
            return
        # Reuse the factor representation without changing split's row law or
        # state-cache lifecycle. The full ordered sweep pools linked patch normal
        # loads; carried impulses seed the factor delta (warm_start).
        split_candidate = (
            self.pgs_mode == "split"
            and (sparse_choice == "on" or os.environ.get("FEATHER_PGS_SPARSE_SPLIT", "1") == "1")
            and self.pgs_kernel == "tiled_row"
            and not self._has_free_rigid_bodies
        )
        velocity_candidate = (
            self.pgs_mode == "matrix_free"
            and self.enable_joint_velocity_limits
            and (sparse_choice == "on" or os.environ.get("FEATHER_PGS_SPARSE_VELOCITY_LIMITS", "1") == "1")
            and not self._has_free_rigid_bodies
        )
        for supported, reason in (
            (
                sparse_choice == "on" or self._kernel_overrides.get("sparse_mass_matrix", True),
                "Private sparse_mass_matrix override disables native sparse factors.",
            ),
            (model.device.is_cuda, "CPU uses ordinary factors; native sparse factors require CUDA."),
            (not model.requires_grad, "Gradient recording requires ordinary factors."),
            (
                self.pgs_mode == "matrix_free" or split_candidate,
                "Solver policy or legacy options do not admit the sparse split-row composition.",
            ),
            (self.pgs_schedule == "interleaved", "Sparse factors require the interleaved row schedule."),
            (self.drive_mode == "augmented", "Sparse factors require augmented drives."),
            (self._parallel_augmented_drive_topology, "Drive topology is not supported by native sparse factors."),
            (self.pgs_iterations > 0, "Native sparse solving requires a positive iteration budget."),
            (self.pgs_velocity_iterations == 0, "Separate velocity iterations retain ordinary factors."),
            (
                not self.enable_joint_velocity_limits or velocity_candidate,
                "Velocity-limit policy or legacy options do not admit the sparse velocity-row composition.",
            ),
            (
                not self.pgs_warmstart or not self._has_free_rigid_bodies,
                "PGS warm starting with free rigid bodies requires the ordinary factor/row composition.",
            ),
            (
                not self._mf_warmstart_enabled or not self._has_free_rigid_bodies,
                "Matrix-free warm starting requires ordinary factors.",
            ),
            (not self._debug_buffers_enabled, "Debug row snapshots require ordinary factors."),
            (not self._regularization_enabled, "Contact regularization requires ordinary factors."),
            (not self._preelim_active, "Bilateral pre-elimination requires ordinary factors."),
            (not self._contact_torsion_enabled, "Contact torsion requires ordinary factors."),
            (not self.contact_compliance, "Contact compliance requires ordinary factors."),
            (not self._mimic_count or velocity_candidate, "Mimic rows require the sparse velocity-row composition."),
            (not self._connect_count, "Connect rows require ordinary factors."),
            (not self._has_loop_joints, "Loop-joint topology requires ordinary factors."),
            (
                not self._sparse_diagonal_contact_solve,
                "The independent-coordinate contact owner already owns response.",
            ),
        ):
            if not supported:
                self._sparse_selection_reason = reason
                return
        # Decide ownership before omitting dense buffers. Explicit research
        # recipes retain their own admission and fallback, including when their
        # requested owner later declines the topology. They cannot acquire
        # dense storage after this sparse allocation decision.
        if self._has_free_rigid_bodies and os.environ.get("FEATHER_PGS_FREE_BODY_IMPLICIT_GYRO") == "1":
            self._sparse_selection_reason = "The free-body gyro experiment retains ordinary factors."
            return
        # A parallel tier that serves the active friction law owns the ordinary buffers.
        # Patch friction yields only to a parallel tier, which split mode never builds;
        # point friction keeps its unconditional yield. The yield reserves the dense route,
        # not the tier: a paired factor sweep admitted there later excludes the tiers and
        # keyed rows under both laws (_init_tiled_kernels, keyed_rows).
        parallel_tier = self.mf_gs_parallel_rows > 0 and (
            self._point_contact_paths or (self._parallel_contact_paths and self.pgs_mode == "matrix_free")
        )
        if parallel_tier:
            self._sparse_selection_reason = "A competing parallel tier reserves the ordinary buffers."
            return
        if self._has_free_rigid_bodies:
            # A world that pairs an articulation with free rigid bodies is served
            # by the held-tree owner and the paired response on the dense route
            # (measured: Kuka lift 12.7 ms there against 15.5 on sparse factors).
            # Both friction laws: patch friction walks contact units in the paired
            # sweep and keeps the resolved-world classifier. A parallel tier that
            # mf_gs_parallel_rows requests (unless FEATHER_PGS_PATCH_PARALLEL_ROWS=0)
            # declines sparse factors first (above), so native sparse factors never
            # serve free rigid bodies.
            self._sparse_selection_reason = (
                "Worlds with free rigid bodies keep the dense route for the held-tree owner and paired response."
            )
            return
        response_dofs = self._model_plan.response_dof_count
        if (split_candidate or velocity_candidate) and (
            self._has_prescribed_response or np.any(self._kinematic_dof_mask_host)
        ):
            self._sparse_selection_reason = "Sparse split/velocity rows do not support prescribed-coordinate topology."
            return
        free_rigid = self._model_plan.is_free_rigid != 0
        articulated = np.flatnonzero((response_dofs > 0) & ~free_rigid)
        sizes = np.unique(response_dofs[articulated])
        if len(sizes) != 1:
            self._sparse_selection_reason = "Native sparse factors require one articulated response DOF size."
            return
        size = int(sizes[0])
        # The endpoint support uses one 64-bit ancestry mask, not a robot-specific DOF limit.
        if size > 64 or size not in self._crba_dof_joint_offset_by_size or size in self._diagonal_mass_sizes:
            self._sparse_selection_reason = (
                "Articulated topology needs at most 64 response DOFs, a CRBA mapping and non-diagonal inertia."
            )
            return
        articulated_counts = np.bincount(self._model_plan.articulation_world[articulated], minlength=self.world_count)
        if split_candidate and not np.all(articulated_counts == 1):
            self._sparse_selection_reason = "Sparse split rows require exactly one articulation per world."
            return
        articulation_start = model.articulation_start.numpy()
        joint_child = model.joint_child.numpy()
        joint_ancestor = model.joint_ancestor.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        art = int(self.group_to_art[size].numpy()[0])
        start = int(articulation_start[art])
        end = int(self._model_plan.articulation_joint_end[art])
        tree = TreePlan.build(
            local_parents(joint_ancestor, [start], end - start)[0], np.diff(joint_qd_start[start : end + 1])
        )
        plan = tree.factor
        # Velocity rows can benefit from factor-coordinate response even for a
        # serial chain; this owner does not require zeros in the mass factor.
        if plan.dof_count != size or (not velocity_candidate and plan.nonzero_count >= size * (size + 1) // 2):
            self._sparse_selection_reason = "Topology has no usable sparse factor layout for this row policy."
            return
        # Native sparse factors are the held tree's representation: a candidate only where it takes their group.
        from . import held_tree  # noqa: PLC0415

        held = held_tree.sparse_decline(self, size, plan, velocity_limits=velocity_candidate, split=split_candidate)
        if held is not None:
            self._sparse_selection_reason = f"Only the held tree evaluates native sparse factors. {held}"
            return

        if split_candidate:
            self._compute_world_response_dof_mapping(model)

        joint_masks = plan.joint_ancestor_mask
        body_masks = np.zeros(model.body_count, dtype=np.uint64)
        first_joint = articulation_start[articulated]
        art, local_joint = host_index.expand(self._model_plan.articulation_joint_end[articulated] - first_joint)
        children = joint_child[first_joint[art] + local_joint]
        valid = children >= 0
        body_masks[children[valid]] = joint_masks[local_joint[valid]]
        shape_body = model.shape_body.numpy()
        contact_masks = np.unique(body_masks[shape_body[shape_body >= 0]])
        max_support = max((int(mask).bit_count() for mask in joint_masks), default=0)
        for first in contact_masks:
            for second in contact_masks:
                max_support = max(max_support, int(first | second).bit_count())
        if self._mimic_count:
            # Coefficients and enable bits stay live, so retain both endpoint
            # supports even when a coupling is initially disabled or zero.
            mimic_indices = self._mimic_art_list.numpy()
            mimic_arts, _ = host_index.expand(np.diff(self._mimic_art_start_np))
            starts = self._model_plan.articulation_dof_start[mimic_arts]
            first_dofs = self._mimic_dof0.numpy()[mimic_indices] - starts
            second_dofs = self._mimic_dof1.numpy()[mimic_indices] - starts
            masks = plan.ancestor_mask[first_dofs] | plan.ancestor_mask[second_dofs]
            max_support = max(max_support, max((int(mask).bit_count() for mask in masks), default=0))
        if np.any(articulated_counts > 1):
            # Different articulations have disjoint coordinate blocks, even
            # when they share the same topology and local ancestor indices.
            max_support = max(max_support, 2 * max(int(mask).bit_count() for mask in joint_masks))
        # Response groups need not coexist in every world. No row can contain
        # more distinct coordinates than the largest world's response vector.
        max_support = min(max_support, self.max_world_dofs)
        if not max_support:
            self._sparse_selection_reason = "Topology has no responsive row support."
            return
        # The sweep's shared working set (sweep_routes_sparse.sparse_staging) against the device's static budget:
        # every shared-memory choice below reads both, and the sweep's factory checks the same.
        budget = arch_budget(model.device.arch)
        staging = functools.partial(
            sweep_routes_sparse.sparse_staging,
            self.dense_max_constraints,
            self.max_world_dofs,
            int(max_support),
            velocity_limits=velocity_candidate,
            linked_rows=self._friction_anchors_enabled,
        )
        # Patch friction keeps the packed lanes of small worlds, so size them from the
        # full sweep before the active-row scan claims whole warps.
        small_world_lanes = self._sparse_small_world_lanes(staging, budget)
        active_rows, active_rows_reason = self._admit_sparse_active_rows(
            split_candidate=split_candidate,
            velocity_candidate=velocity_candidate,
            small_world_lanes=small_world_lanes,
        )
        solve = staging()
        if active_rows:
            active = staging(active_rows=True)
            if self._friction_anchors_enabled and not active.fits(budget):
                # Patch friction admitted this route before it adopted the scan; keep it.
                active_rows = False
                active_rows_reason = (
                    "Patch friction keeps the full ordered sweep: the row bitsets exceed shared memory."
                )
            else:
                solve = active
        if not solve.fits(budget):
            self._sparse_selection_reason = (
                f"Row capacity needs {solve.bytes} bytes of shared memory for sparse solving; limit is {budget}."
            )
            return
        # The factor build's warps share a block within the offline static budget.
        shared_limit = min(STATIC_SHARED_BYTES, int(model.device.max_shared_memory_per_block))
        shared_bytes_per_warp = 4 * (2 * plan.nonzero_count + 6 * size)
        self._sparse_factor_warps_per_block = min(
            _CRBA_CHOLESKY_WARPS_PER_BLOCK,
            shared_limit // shared_bytes_per_warp,
        )
        if self._sparse_factor_warps_per_block < 1:
            self._sparse_selection_reason = (
                f"One sparse factor warp needs {shared_bytes_per_warp} bytes of shared memory; limit is {shared_limit}."
            )
            return

        source = "Explicit sparse_mass_matrix='on'" if sparse_choice == "on" else "Automatic sparse selection"
        self._sparse_selection_reason = (
            f"{source}: {size}-DOF topology and row policy admitted; "
            f"solve uses {solve.bytes}/{budget} bytes of shared memory."
        )
        self._sparse_mass_matrix_size = size
        self._sparse_active_rows = active_rows
        self._sparse_active_rows_reason = active_rows_reason
        self._sparse_max_row_support = int(max_support)
        self._sparse_split = split_candidate
        self._sparse_velocity_limits = velocity_candidate
        # The sparse response kernel prepares the contact rows' metadata, CFM, bias and restitution.
        self._sparse_prepare_contacts = True
        # The sparse producers overwrite every live value and retain factors by the mass mask.
        self._double_buffer = False
        self._local_internal_fast_path = False
        self._local_solve_reason = "Native sparse factors replace the local owners."
        self._sparse_mass_matrix_plan = plan
        self._sparse_mass_matrix_indices = factor_indices(plan, model.device)
        self._sparse_body_dof_mask = wp.array(body_masks, dtype=wp.uint64, device=model.device)
        self._arrays.allocate(SPARSE_FACTOR, group=self.n_arts_by_size[size], nonzero=plan.nonzero_count)
        shape = (self.world_count, self.dense_max_constraints, max_support)
        offsets = self.articulation_world_dof_offset.numpy()
        if (
            self._kernel_overrides.get("sparse_row_support", True)
            and not self._mimic_count
            and np.all(articulated_counts == 1)
            and size == self.max_world_dofs
            and np.all(offsets[articulated] == 0)
        ):
            classes = tree.support_classes(body_masks, max_support)
            if classes is not None:
                table_bytes = sum(
                    getattr(classes, name).nbytes for name in ("body_class", "pair_class", "class_dof", "limit_class")
                )
                table_bytes += 4 * self.world_count * self.dense_max_constraints
                # Small batches with many support patterns can favor direct indices.
                if table_bytes < 4 * int(np.prod(shape)):
                    self._sparse_row_support = row_supports(
                        classes, self.world_count, self.dense_max_constraints, model.device
                    )
        # Unsupported mixed worlds retain explicit row indices and their existing kernels.
        use_support_table = bool(self._sparse_row_support.enabled)
        # Reuse the local solver's small-world packing without changing this
        # owner's row representation, sweep policy or velocity-limit phase.
        self._sparse_solve_lanes = small_world_lanes if small_world_lanes and not active_rows else 32
        triplet_capacity = self.dense_max_constraints // 3
        transaction, transaction_reason = self._request_sparse_contact_transaction()
        self._sparse_contact_triplets, self._sparse_contact_triplets_reason = self._admit_sparse_contact_triplets(
            active_rows=active_rows,
            split_candidate=split_candidate,
            velocity_candidate=velocity_candidate,
            use_support_table=use_support_table,
            # The selected transaction's working set: the staged one's metric cache, the single-stage one's none.
            fits=lambda: (
                triplet_capacity > 0
                and staging(
                    active_rows=active_rows,
                    use_support_table=use_support_table,
                    contact_triplets=True,
                    contact_transaction=transaction,
                ).fits(budget)
            ),
        )
        transaction_options = {}
        if self._sparse_contact_triplets:
            self._sparse_contact_transaction = transaction
            self._sparse_contact_transaction_reason = transaction_reason
            transaction_options["contact_transaction"] = transaction
            if transaction == "single":
                # Static storage for graph capture: the builder writes the records of each step's
                # triplets; linked sweeps add each anchor's ring members once per call.
                width = CONTACT_RECORD_FLOATS + (CONTACT_RING_FLOATS if self._friction_anchors_enabled else 0)
                self._arrays.allocate(
                    SPARSE_CONTACT_RECORDS, world=self.world_count, triplet=triplet_capacity, record=width
                )
        else:
            self._sparse_contact_transaction_reason = "No contact triplets are admitted."
        index_shape = (1, 1, 1) if use_support_table else shape
        self._arrays.allocate(
            SPARSE_ROWS,
            index_world=index_shape[0],
            index_row=index_shape[1],
            index_support=index_shape[2],
            world=self.world_count,
            row=self.dense_max_constraints,
            support=max_support,
            world_dof=self.max_world_dofs,
        )
        # The response kernel's free-body response output, where no contact records take
        # it, and its free-body factor: placeholders, since native sparse factors decline
        # free rigid bodies.
        self._sparse_row_free_response = self._arrays.placeholder(wp.float32, (1, 1, 1))
        self._sparse_free_factor_dummy = self._arrays.placeholder(wp.float32, (1, 6, 6))
        self._sparse_contact_lanes = 8
        self._sparse_contact_response_kernel = response_builds_rows.factor_contact_rows_kernel(
            size,
            lanes_per_contact=self._sparse_contact_lanes,
            prepare_rows=True,
            use_support_table=use_support_table,
            **({"contact_records": True} if self._sparse_contact_records is not None else {}),
        )
        # Large resident workspaces regressed whole-step timing in measured
        # small systems. Keep the global-row consumer when the cache is large.
        self._sparse_local_factors = bool(
            self._sparse_solve_lanes < 32
            and os.environ.get("FEATHER_PGS_SPARSE_LOCAL_FACTORS", "1") != "0"
            and staging(lanes_per_world=self._sparse_solve_lanes, local_factors=True).bytes
            <= min(budget, _SPARSE_LOCAL_WORKSPACE_BYTES)
        )
        self._pgs_solve_sparse_kernel = self._sweep_kernel(
            "sparse_pgs",
            sweep_routes_sparse.sparse_kernel,
            self.dense_max_constraints,
            self.max_world_dofs,
            max_support,
            active_rows=active_rows,
            use_support_table=use_support_table,
            contact_triplets=self._sparse_contact_triplets,
            velocity_limits=self._sparse_velocity_limits,
            lanes_per_world=self._sparse_solve_lanes,
            local_factors=self._sparse_local_factors,
            warm_start=self.pgs_warmstart,
            linked_rows=self._friction_anchors_enabled,
            velocity_limit_law=self._velocity_limit_kernel_law(),
            device_arch=model.device.arch,
            **transaction_options,
        )

    def _sweep_kernel(self, route: str, factory, *args, readout_budget: int | None = None, **kwargs):
        """Build a sweep under the solve policy and allocate its readout counters, outside any graph capture.

        Args:
            route: Owner of the sweep in the records, such as ``"sparse_pgs"``.
            factory: The sweep factory, which takes the policy as ``solve_policy`` and returns a kernel or a
                :class:`SweepTier`.
            *args: Positional arguments of the factory.
            readout_budget: The sweep budget the kernel compiles in (a parallel tier's), or ``None`` for the
                solver's iteration counts.
            **kwargs: Keyword arguments of the factory.
        """
        policy = self._solve_policy
        built = factory(*args, solve_policy=policy, **kwargs)
        kernel = built.kernel if isinstance(built, SweepTier) else built
        budget, velocity_budget = int(self.pgs_iterations), int(self.pgs_velocity_iterations)
        if readout_budget is not None:
            budget, velocity_budget = int(readout_budget), int(readout_budget) if velocity_budget > 0 else 0
        self._solve_readout.register(
            kernel,
            route,
            readout=policy.readout if policy.counts else None,
            world_count=int(self.world_count),
            budget=max(budget, 1),
            device=self.model.device,
            velocity_budget=velocity_budget,
        )
        return built

    def _sparse_small_world_lanes(self, staging, budget: int) -> int:
        """Return the lanes per world (8 or 16) that pack a small complete sparse sweep, or 0 for whole warps.

        Args:
            staging: The sweep's working set for its options (:func:`.sweep_routes_sparse.sparse_staging`).
            budget: Static shared memory available to one block [B].
        """
        if (
            os.environ.get("FEATHER_PGS_SPARSE_SMALL_WORLDS", "1") == "0"
            or self.max_world_dofs > 16
            or self.dense_max_constraints > 64
        ):
            return 0
        lanes = 8 if self.max_world_dofs <= 8 else 16
        return lanes if staging(lanes_per_world=lanes).fits(budget) else 0

    def _admit_sparse_active_rows(
        self, *, split_candidate: bool, velocity_candidate: bool, small_world_lanes: int
    ) -> tuple[bool, str]:
        """Decide whether the native sparse sweep selects its normal and limit rows at sweep entry, and why.

        Under this law a zero-impulse row that a sweep's earlier updates violate waits for
        the next sweep. Patch friction (``friction_anchor_beta > 0``) adopts it: a friction
        pair is visited while any normal of its patch ring is selected or the pair holds
        impulse, which skips only pairs whose update would be a no-op. Small worlds keep
        their packed lanes, and ``FEATHER_PGS_PATCH_ACTIVE_ROWS=0`` keeps patch friction on
        the full ordered sweep.

        Args:
            split_candidate: Whether split mode sweeps the sparse rows.
            velocity_candidate: Whether the sweep carries joint velocity-limit rows.
            small_world_lanes: Lanes per world of a packed small-world sweep, 0 for whole warps.
        """
        if not self._kernel_overrides.get("sparse_active_contacts", True):
            return False, "A private sparse_active_contacts override disables the active-row scan."
        if split_candidate or velocity_candidate:
            return False, "Split and velocity-limit rows keep full ordered sweeps."
        if not self._friction_anchors_enabled:
            return True, "Point-contact normal and limit rows are selected at sweep entry."
        if not self._patch_active_rows:
            return False, (
                "FEATHER_PGS_PATCH_ACTIVE_ROWS=0 keeps patch friction (friction_anchor_beta > 0) on the full "
                "ordered sweep."
            )
        if small_world_lanes:
            return False, (
                f"Patch friction (friction_anchor_beta > 0) keeps the packed {small_world_lanes}-lane full sweep "
                "of small worlds."
            )
        return True, (
            "Normal and limit rows are selected at sweep entry; a patch friction pair (friction_anchor_beta > 0) "
            "while any normal of its ring is selected or the pair holds impulse."
        )

    def _admit_sparse_contact_triplets(
        self,
        *,
        active_rows: bool,
        split_candidate: bool,
        velocity_candidate: bool,
        use_support_table: bool,
        fits: bool,
    ) -> tuple[bool, str]:
        """Decide whether the sparse sweep fuses contact triplets (``FEATHER_PGS_SPARSE_CONTACT_TRIPLETS=1``), and why.

        Point friction fuses them in the active-row scan. Patch friction fuses its anchor
        units, pooling their ring loads as the ordinary pair update does, in the active
        scan or else in the full ordered sweep; split, velocity-limit, free-body and
        packed small-world sweeps keep ordinary updates. The staged patch transaction is
        bitwise equal to the same sweep without it. The staged point transaction is too. The
        single-stage transaction (:meth:`_request_sparse_contact_transaction`) differs
        from the staged one by rounding only.

        Args:
            active_rows: Whether the sweep selects its rows at sweep entry.
            split_candidate: Whether split mode sweeps the sparse rows.
            velocity_candidate: Whether the sweep carries joint velocity-limit rows.
            use_support_table: Whether rows read the shared support table.
            fits: Whether the selected triplet transaction fits the row capacity and shared memory, evaluated last.
        """
        if os.environ.get("FEATHER_PGS_SPARSE_CONTACT_TRIPLETS", "0") != "1":
            return False, "Off: FEATHER_PGS_SPARSE_CONTACT_TRIPLETS is an opt-in."
        if not active_rows and not self._friction_anchors_enabled:
            return False, "Requested, but triplets need the active-row scan."
        if not active_rows and (split_candidate or velocity_candidate or self._has_free_rigid_bodies):
            return False, "Requested, but split, velocity-limit and free-body sweeps keep ordinary updates."
        if self._sparse_solve_lanes < 32:
            return False, f"Requested, but small worlds keep their packed {self._sparse_solve_lanes}-lane full sweep."
        if not use_support_table:
            return False, "Requested, but triplets need the row support table."
        if not fits():
            return False, "Requested, but row capacity or shared memory leaves no room for the triplet transaction."
        # contact_transaction_reason records how the admitted triplets update.
        if self._friction_anchors_enabled:
            sweep = "active-row scan" if active_rows else "full ordered sweep"
            where = f"patch anchor units pool their ring loads in the {sweep}"
        else:
            where = "point contacts fuse in the active-row scan"
        return True, f"FEATHER_PGS_SPARSE_CONTACT_TRIPLETS=1: {where}."

    def _request_sparse_contact_transaction(self) -> tuple[str, str]:
        """Return the transaction admitted contact triplets take (``"single"`` or ``"staged"``), and why.

        The single-stage transaction loads a triplet's rows with its block record, which
        the sparse contact builder writes, and reduces its three residuals together; the
        staged one reduces the normal, then the tangents, and caches tangent metrics in
        shared memory. Both apply the paired friction law, so iterates differ by rounding
        only. Patch friction (``friction_anchor_beta > 0``) takes the single-stage one by
        default and point friction the staged one; ``FEATHER_PGS_SPARSE_TRANSACTION``
        (``auto``, ``single`` or ``staged``, see :meth:`_sparse_transaction_flags`) overrides both.
        """
        request = self._sparse_transaction_request
        if request == "single":
            return "single", "FEATHER_PGS_SPARSE_TRANSACTION=single: one load group and one reduction per triplet."
        if request == "staged":
            return "staged", (
                "FEATHER_PGS_SPARSE_TRANSACTION=staged: the normal's reduction, then the tangents', with the metric "
                "cache."
            )
        if self._friction_anchors_enabled:
            return "single", (
                "Patch friction (friction_anchor_beta > 0) takes the single-stage transaction by default "
                "(FEATHER_PGS_SPARSE_TRANSACTION=staged keeps the staged one)."
            )
        return "staged", (
            "Point friction keeps the staged transaction by default (FEATHER_PGS_SPARSE_TRANSACTION=single opts in)."
        )

    @staticmethod
    def _sparse_transaction_flags() -> str:
        """Return the triplet transaction that the environment requests.

        ``FEATHER_PGS_SPARSE_TRANSACTION`` is one of :data:`~.sweep_routes_sparse.CONTACT_TRANSACTIONS`
        (default ``auto``). Every construction checks it, so an invalid value raises on every route.
        """
        transaction = os.environ.get("FEATHER_PGS_SPARSE_TRANSACTION", "auto") or "auto"
        if transaction not in CONTACT_TRANSACTIONS:
            raise ValueError(
                f"FEATHER_PGS_SPARSE_TRANSACTION must be one of {', '.join(CONTACT_TRANSACTIONS)}, got {transaction!r}"
            )
        return transaction

    def _build_body_maps(self, model):
        if not model.body_count or not model.articulation_count:
            self.body_to_joint = None
            self.body_to_articulation = None
            self.body_has_response_dofs = None
            self.body_response_dof_mask = None
            self.body_single_response_dof = None
            self._body_single_response_dof_host = None
            return

        (
            body_to_joint,
            body_to_articulation,
            body_has_response_dofs,
            body_response_dof_mask,
            body_single_response_dof,
        ) = build_body_maps(
            model.body_count,
            model.joint_child.numpy(),
            self._joint_forest,
            model.joint_qd_start.numpy(),
            model.articulation_start.numpy(),
            self._model_plan.articulation_joint_end,
            self._model_plan.articulation_dof_start,
            self._model_plan.response_dof_count,
        )

        self._arrays.upload(
            BODY_MAPS,
            body_to_joint=body_to_joint,
            body_to_articulation=body_to_articulation,
            body_has_response_dofs=body_has_response_dofs,
            body_response_dof_mask=body_response_dof_mask,
            body_single_response_dof=body_single_response_dof,
        )
        self._body_single_response_dof_host = body_single_response_dof

    def _classify_free_rigid_bodies(self, model):
        """Materialize free-rigid execution metadata from the model plan."""
        # Free rigid bodies per world: they bound a world's contact units per color (sweep_units.capable_worlds).
        self._free_rigid_per_world = np.zeros(max(int(model.world_count), 1), dtype=np.int64)
        if not model.articulation_count or not model.joint_count:
            self._has_free_rigid_bodies = False
            self._has_mixed_contacts = False
            self._has_non_free_articulations = False
            self.is_free_rigid = None
            self.free_rigid_body_indices = None
            self._free_rigid_body_count = 0
            return
        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before free-body classification")

        is_free_rigid = self._model_plan.is_free_rigid
        solve_mask = self._model_plan.response_dof_count > 0
        free_mask = is_free_rigid != 0
        self._free_rigid_body_count = len(self._model_plan.response_free_rigid_body_indices)
        self._has_free_rigid_bodies = self._free_rigid_body_count > 0

        articulation_world = self._model_plan.articulation_world
        free_worlds = set(articulation_world[solve_mask & free_mask])
        nonfree_worlds = set(articulation_world[solve_mask & ~free_mask])
        self._has_mixed_contacts = bool(free_worlds & nonfree_worlds)
        # Solve-mask-aware: zero-DOF (kinematic-only) articulations are excluded
        # from the solve size groups, so they must not count as non-free
        # articulations when selecting the patch routes.
        self._has_non_free_articulations = bool(np.any(solve_mask & ~free_mask))
        free_world = articulation_world[solve_mask & free_mask]
        free_world = free_world[free_world >= 0]
        self._free_rigid_per_world = np.bincount(free_world, minlength=self._free_rigid_per_world.size)
        self.is_free_rigid = wp.array(is_free_rigid, dtype=wp.int32, device=model.device)
        self.free_rigid_body_indices = wp.array(
            self._model_plan.response_free_rigid_body_indices, dtype=wp.int32, device=model.device
        )

    def _compute_world_response_dof_mapping(self, model):
        """Build compact per-world response offsets and global-DOF indices."""
        if self._model_plan is None:
            raise RuntimeError("FeatherPGS model plan must be built before response mapping")

        articulation_world = self._model_plan.articulation_world
        response_dof_count = self._model_plan.response_dof_count
        articulation_dof_start = self._model_plan.articulation_dof_start
        articulation_world_dof_offset = np.full(model.articulation_count, -1, dtype=np.int32)

        # Each world lists its response articulations' DOFs by ascending DOF start, ties by articulation.
        articulations = np.flatnonzero(response_dof_count > 0)
        articulations = articulations[
            np.lexsort((articulation_dof_start[articulations], articulation_world[articulations]))
        ]
        worlds = articulation_world[articulations]
        counts = response_dof_count[articulations].astype(np.int64)
        world_dof_count = np.zeros(self.world_count, dtype=np.int64)
        np.add.at(world_dof_count, worlds, counts)
        world_offset = host_index.segmented_rank(worlds, counts)
        articulation_world_dof_offset[articulations] = world_offset

        world_dof_count = world_dof_count.astype(np.int32)
        self.max_world_dofs = int(np.max(world_dof_count)) if len(world_dof_count) else 0
        padded_indices = np.full((self.world_count, self.max_world_dofs), -1, dtype=np.int32)
        entry, local_dof = host_index.expand(counts)
        padded_indices[worlds[entry], world_offset[entry] + local_dof] = (
            articulation_dof_start[articulations][entry] + local_dof
        )

        self.articulation_world_dof_offset = wp.array(
            articulation_world_dof_offset, dtype=wp.int32, device=model.device
        )
        self.world_dof_count = wp.array(world_dof_count, dtype=wp.int32, device=model.device)
        self.world_dof_indices = wp.array(padded_indices, dtype=wp.int32, device=model.device)

    def _detect_jy_world_identity(self) -> bool:
        """Return whether host-side group and compact world J/Y layouts are identical."""
        if len(self.size_groups) != 1 or self.art_to_world is None:
            return False
        size = self.size_groups[0]
        if self.n_arts_by_size[size] != self.world_count:
            return False
        articulation_world = self._model_plan.articulation_world
        group_to_art = np.flatnonzero(self._model_plan.response_dof_count == size)
        expected_world = np.arange(len(group_to_art), dtype=articulation_world.dtype)
        return np.array_equal(articulation_world[group_to_art], expected_world)

    def _select_paired_response(self) -> tuple[int, int, np.ndarray] | None:
        """Select one complete robot/free-body response pair per world."""
        if not self._hinv_jt_writes_world or self._hinv_jt_tiled_writes_group or 6 not in self.size_groups:
            return None
        if self._execution_plan.use_tiled_hinv_jt(6):
            return None

        response_dofs = self._model_plan.response_dof_count
        articulation_world = self._model_plan.articulation_world
        is_free_rigid = self._model_plan.is_free_rigid
        # Each world's response articulations, in ascending order.
        responding = np.flatnonzero(response_dofs > 0)
        world_bounds, order = host_index.group_by(articulation_world[responding], self.world_count)
        responding = responding[order].tolist()
        response_by_world = [responding[world_bounds[w] : world_bounds[w + 1]] for w in range(self.world_count)]

        secondary_group_arts = np.flatnonzero(response_dofs == 6)
        secondary_group_index = {int(art): group for group, art in enumerate(secondary_group_arts)}
        for primary_size in sorted(self._paired_response_candidate_sizes, reverse=True):
            primary_arts = np.flatnonzero(response_dofs == primary_size)
            if len(primary_arts) != self.world_count:
                continue
            paired_groups: list[int] = []
            paired_arts: list[int] = []
            for primary_art in primary_arts:
                world_arts = response_by_world[int(articulation_world[primary_art])]
                secondaries = [
                    art
                    for art in world_arts
                    if art != primary_art and response_dofs[art] == 6 and is_free_rigid[art] != 0
                ]
                if len(world_arts) != 2 or len(secondaries) != 1:
                    break
                secondary_art = secondaries[0]
                paired_arts.append(secondary_art)
                paired_groups.append(secondary_group_index[secondary_art])
            else:
                if len(set(paired_arts)) == len(secondary_group_arts) and set(paired_arts) == set(secondary_group_arts):
                    return primary_size, 6, np.asarray(paired_groups, dtype=np.int32)
        return None

    def _select_sparse_diagonal_response_pair(
        self, *, factor_dense_contract: bool
    ) -> tuple[int, int, np.ndarray, np.ndarray] | None:
        """Select one independent wide component plus one compact dense component per world."""
        if not factor_dense_contract or not self._fused_diagonal_joint_limits:
            return None

        response_dofs = self._model_plan.response_dof_count
        articulation_world = self._model_plan.articulation_world
        is_free_rigid = self._model_plan.is_free_rigid
        offsets = self.articulation_world_dof_offset.numpy()
        group_indices = self.art_group_idx.numpy()
        pairs = _world_diagonal_pairs(
            articulation_world, response_dofs, self.world_count, self._fused_diagonal_joint_limit_sizes
        )
        if pairs is None or self.world_count == 0:
            return None
        diagonal_arts, dense_arts = pairs
        if np.any(is_free_rigid[diagonal_arts] != 0) or np.any(is_free_rigid[dense_arts] != 0):
            return None
        # Every world pairs the same two sizes.
        diagonal_size = response_dofs[diagonal_arts]
        dense_size = response_dofs[dense_arts]
        if np.any((dense_size <= 0) | (dense_size > _CONTACT_JACOBIAN_MAX_DOF)):
            return None
        if np.any(diagonal_size != diagonal_size[0]) or np.any(dense_size != dense_size[0]):
            return None
        dense_offsets = offsets[dense_arts].astype(np.int32)
        dense_groups = group_indices[dense_arts].astype(np.int32)
        if np.any(dense_offsets < 0) or np.any(dense_groups < 0):
            return None
        return int(diagonal_size[0]), int(dense_size[0]), dense_offsets, dense_groups

    def _upload_joint_forest(self, model: Model) -> None:
        """Upload the joint forest's depth-first order, positions and subtree ends (``tree_passes.crba_fill``'s
        ancestor test) and its up flags (the up scans', :attr:`.tree_plan.TreePlan.up_flags`)."""
        if not model.joint_count:
            self._arrays.release(JOINT_FOREST)
            return
        forest = self._joint_forest
        self._arrays.upload(
            JOINT_FOREST,
            _joint_dfs_order=forest.dfs_order,
            _joint_dfs_position=forest.dfs_position,
            _joint_dfs_end=forest.dfs_end,
            _joint_up_flags=forest.up_flags,
        )

    def _allocate_common_buffers(self, model):
        arrays = self._arrays
        if model.joint_count:
            arrays.allocate(ARTICULATION_WORK, articulation=model.articulation_count)
            arrays.allocate(DOF_WORK, dof=model.joint_dof_count)
            # The joint velocity prescale writes every DOF of qd_work (see _launch_velocity_prescale).
            self._velocity_prescale_lanes, self._velocity_prescale_threads = _velocity_prescale_launch(
                model.articulation_start.numpy(), joint_count=model.joint_count, cuda=model.device.is_cuda
            )
            # Stage-3 snapshots for external NaN dumpers, which read their values
            # by these names: None unless opted in.
            if self._debug_stage3_snapshots:
                arrays.allocate(STAGE3_SNAPSHOTS, dof=model.joint_dof_count)
            else:
                arrays.release(STAGE3_SNAPSHOTS)
        else:
            arrays.release(ARTICULATION_WORK)
            arrays.release(DOF_WORK)
            arrays.release(STAGE3_SNAPSHOTS)
            self._velocity_prescale_lanes, self._velocity_prescale_threads = 1, 0

        self._diagonal_inverse_mass = (
            wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
            if self._diagonal_mass_sizes
            else arrays.placeholder(wp.float32)
        )
        self._fused_diagonal_limit_dof_mask = (
            wp.array(self._fused_diagonal_limit_dof_mask_host, dtype=wp.int32, device=model.device)
            if self._fused_diagonal_joint_limits
            else arrays.placeholder(wp.int32)
        )

        if model.body_count:
            arrays.allocate(BODY_INERTIA, body=model.body_count)
            self._upload_joint_forest(model)
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )
        else:
            arrays.release(BODY_INERTIA)

        if not model.articulation_count or not model.joint_count:
            self.articulation_origin = None
            return

        arrays.allocate(ARTICULATION_ORIGIN, articulation=model.articulation_count)

    def _setup_parallel_augmented_drive_topology(self, model, eligible: bool) -> None:
        """Allocate the augmented drive rows and separate immutable per-coordinate row ownership from per-step
        coefficients, where ``eligible``."""
        if model.articulation_count and model.joint_count and self.articulation_max_dofs:
            self._arrays.allocate(
                AUGMENTED_DRIVE_ROWS,
                articulation=model.articulation_count,
                drive_row=model.articulation_count * self.articulation_max_dofs,
            )
        self._parallel_augmented_drive_topology = False
        self._augmented_drive_row_by_dof = wp.full(
            (max(1, int(model.joint_dof_count)),), -1, dtype=wp.int32, device=model.device
        )
        self._augmented_drive_q_index_by_dof = None
        self._augmented_drive_target_index_by_dof = None
        if not eligible or self._has_loop_joints or not model.joint_dof_count:
            return

        dof_count = int(model.joint_dof_count)
        # Every DOF must lie in exactly one articulation's DOF range.
        range_start = self._model_plan.articulation_dof_start.astype(np.int64)
        owner, local_dof = host_index.expand(self._model_plan.articulation_dof_count)
        owned_dof = range_start[owner] + local_dof
        if np.any(np.bincount(owned_dof[owned_dof < dof_count], minlength=dof_count)[:dof_count] != 1):
            return

        joint_type = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_target_q_start = model.joint_target_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        articulation_start = model.articulation_start.numpy()
        max_dofs = self.articulation_max_dofs

        drive_row_by_dof = np.full(dof_count, -1, dtype=np.int32)
        drive_q_index_by_dof = np.full(dof_count, -1, dtype=np.int32)
        drive_target_index_by_dof = np.full(dof_count, -1, dtype=np.int32)
        row_dof_index = np.zeros(model.articulation_count * max_dofs, dtype=np.int32)
        supported = (int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6))
        # Each articulation's supported joint axes, in joint order, take its first max_dofs row slots.
        first_joint = articulation_start[: model.articulation_count]
        articulation, position = host_index.expand(
            np.maximum(articulation_start[1 : model.articulation_count + 1] - first_joint, 0)
        )
        joint = first_joint[articulation] + position
        driven = np.isin(joint_type[joint], supported)
        articulation, joint = articulation[driven], joint[driven]
        axis_joint, axis = host_index.expand(joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1])
        articulation, joint = articulation[axis_joint], joint[axis_joint]
        axis_count = np.bincount(articulation, minlength=model.articulation_count)
        slot = host_index.segmented_rank(articulation)
        slotted = slot < max_dofs
        articulation, joint, axis, slot = articulation[slotted], joint[slotted], axis[slotted], slot[slotted]
        dof = joint_qd_start[joint] + axis
        row = articulation * max_dofs + slot
        drive_row_by_dof[dof] = row
        drive_q_index_by_dof[dof] = joint_q_start[joint] + axis
        drive_target_index_by_dof[dof] = joint_target_q_start[joint] + axis
        row_dof_index[row] = dof
        row_counts = np.minimum(axis_count, max_dofs).astype(np.int32)

        self._augmented_drive_row_by_dof = wp.array(drive_row_by_dof, dtype=wp.int32, device=model.device)
        self._augmented_drive_q_index_by_dof = wp.array(drive_q_index_by_dof, dtype=wp.int32, device=model.device)
        self._augmented_drive_target_index_by_dof = wp.array(
            drive_target_index_by_dof, dtype=wp.int32, device=model.device
        )
        self.aug_row_counts.assign(row_counts)
        self.aug_row_dof_index.assign(row_dof_index)
        self._parallel_augmented_drive_topology = True

    def _allocate_buffers(self, model):
        self.H_by_size = {}
        self.L_by_size = {}
        self.J_by_size = {}
        self.Y_by_size = {}
        self.diag_by_size = {}
        self.R_by_size = {}
        self.tau_by_size = {}
        self.qdd_by_size = {}
        if not self.size_groups:
            self._dummy_hinv_diag = None
            self._H_bufs = None
            self._J_bufs = None
            return

        device = model.device
        requires_grad = model.requires_grad
        arrays = self._arrays
        self._dummy_hinv_diag = wp.zeros((1, 1), dtype=wp.float32, device=device, requires_grad=requires_grad)
        double_buffered = self._double_buffer and device.is_cuda
        self._H_bufs = [{}, {}] if double_buffered else None
        self._J_bufs = [{}, {}] if double_buffered else None

        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            # Compact and sparse sizes keep (1, 1, 1) placeholders of the factor and the response.
            compact_diagonal_mass = size in (self._compact_diagonal_mass_size, self._sparse_mass_matrix_size)
            sparse_response = self._sparse_mass_matrix_size is not None or (
                self._sparse_diagonal_contact_solve and size == self._sparse_diagonal_response_size
            )
            factor = 1 if compact_diagonal_mass else 0
            # The matrix-free sweeps' producers write the packed rows (_jacobian_rows): no size group's Jacobian.
            response = 1 if sparse_response or self._mf_sweeps else 0
            # The matrix-free sweeps read the packed rows: no route reads a size group's response.
            y = 1 if sparse_response or self._mf_sweeps else 0
            sizes = {
                "art": n_arts,
                "size": size,
                "factor_art": factor or n_arts,
                "factor_size": factor or size,
                "response_art": response or n_arts,
                "response_row": response or self.dense_max_constraints,
                "response_size": response or size,
                "y_art": y or n_arts,
                "y_row": y or self.dense_max_constraints,
                "y_size": y or size,
            }
            buffers = arrays.allocate(SIZE_FACTORS, owner={}, **sizes)
            if not compact_diagonal_mass:
                # Grouped scratch belongs to the dense triangular solve, not the compact paths.
                buffers.update(arrays.allocate(SIZE_GROUPED_SOLVE, owner={}, **sizes))
            # Double buffering gives the mass matrix and Jacobian of a size group two sets.
            systems = (
                zip(self._H_bufs, self._J_bufs, strict=True) if double_buffered else ((self.H_by_size, self.J_by_size),)
            )
            for H_by_size, J_by_size in systems:
                system = arrays.allocate(SIZE_SYSTEM, owner={}, **sizes)
                H_by_size[size], J_by_size[size] = system["H"], system["J"]
            for name, array in buffers.items():
                getattr(self, f"{name}_by_size")[size] = array
            self.diag_by_size[size] = (
                wp.zeros(
                    (n_arts, self.dense_max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
                )
                if size in self._hinv_jt_diag_sizes and not sparse_response
                else self._dummy_hinv_diag
            )

        if self._H_bufs is not None:
            self.H_by_size = self._H_bufs[0]
            self.J_by_size = self._J_bufs[0]
            self._buf_idx = 0

        # Every per-contact buffer holds the ready rows of the model's contact pool, which its
        # collision pipelines share: they are sized with the pipeline's measured demand before a graph
        # is captured, and grow with any other contact stream the solver consumes (see _follow_contacts).
        self._contact_pool = model_contact_pool(model, device)
        contact_fields = [
            ContactField(name, wp.int32, requires_grad=requires_grad)
            for name in (
                "contact_world",
                "contact_slot",
                "contact_slots_needed",
                "contact_art_a",
                "contact_art_b",
                "contact_path",
            )
        ]
        # The masked row builders' per-contact flags; every entry below the live
        # contact count is written by the prelude before it is read.
        contact_fields.append(ContactField("_rows_prelude_flags", wp.int32))
        # Row builders always receive a zero phi view when patch friction is off.
        contact_fields.append(ContactField("_friction_patch_phi", wp.vec2))
        if self.pgs_warmstart:
            contact_fields.append(ContactField("_ws_prev_contact_normal", wp.vec3))
        else:
            self._ws_prev_contact_normal = None
        debug_fields = (
            ("contact_world", wp.int32),
            ("contact_slot", wp.int32),
            ("contact_art_a", wp.int32),
            ("contact_art_b", wp.int32),
            ("contact_path", wp.int32),
            ("rigid_contact_shape0", wp.int32),
            ("rigid_contact_shape1", wp.int32),
            ("rigid_contact_point0", wp.vec3),
            ("rigid_contact_point1", wp.vec3),
            ("rigid_contact_normal", wp.vec3),
            ("rigid_contact_margin0", wp.float32),
            ("rigid_contact_margin1", wp.float32),
        )
        if self._debug_buffers_enabled:
            contact_fields += [
                ContactField(f"_debug_position_{name}", dtype, requires_grad=requires_grad)
                for name, dtype in debug_fields
            ]
        else:
            for name, _ in debug_fields:
                setattr(self, f"_debug_position_{name}", None)
        self._contact_rows = self._contact_pool.rows(contact_fields, on_rows=self._on_contact_rows)
        # Persistent state is owned by the body-pair patch builder, independently
        # of collision matching and the contact warm-start tables.
        frame_reuse, self._patch_frame_reuse_reason = self._patch_frame_reuse()
        self._friction_patches = _FrictionPatchState(
            model,
            self._contact_pool.ready,
            self._friction_anchors_enabled,
            self._friction_patch_phi,
            reuse_frames=frame_reuse,
            pool=self._contact_pool,
        )
        # Allocated once here: per-step allocation is forbidden under CUDA graph capture.
        arrays.allocate(WORLD_ROW_STATE, world=self.world_count)
        # Contact runs of each world and block under sorted contacts (record_world_contact_runs):
        # blocks are the global shapes before every world's, the worlds' own shapes, and the global
        # shapes after them; 16 bytes per world and block.
        shape_world_start = model.shape_world_start.numpy() if model.shape_world_start is not None else None
        if shape_world_start is None or model.shape_count == 0:
            self._contact_run_lead, self._contact_run_trail_start = 0, int(model.shape_count)
        else:
            self._contact_run_lead = int(shape_world_start[0])
            self._contact_run_trail_start = int(shape_world_start[-2])
        run_blocks = self._contact_run_lead + 1 + int(model.shape_count) - self._contact_run_trail_start
        arrays.allocate(CONTACT_RUNS, world=self.world_count, run_block=run_blocks)
        self._debug_position_rigid_contact_count = (
            wp.zeros((1,), dtype=wp.int32, device=device) if self._debug_buffers_enabled else None
        )

        # Joint velocity-limit buffers (per-DOF tracking). Independent from the
        # joint-position-limit buffers so the two flags can be used separately.
        # Both rows of a finite limit are allocated every step, so the velocity box
        # is present before the predicted velocity escapes.
        if self.enable_joint_velocity_limits and model.joint_dof_count > 0:
            arrays.allocate(VELOCITY_LIMIT_SLOTS, limit_row=2 * model.joint_dof_count)
        else:
            arrays.release(VELOCITY_LIMIT_SLOTS)

        if self.drive_mode == "physx_pgs" and model.joint_dof_count > 0:
            arrays.allocate(DRIVE_SLOTS, dof=model.joint_dof_count)
        else:
            arrays.release(DRIVE_SLOTS)

        # Velocity-limit kernels take a drive_slot input for the fused-clamp
        # skip; when fusion is off (or there are no drive slots) they receive
        # this persistent 1-element dummy and never read it (skip_driven == 0).
        self._vel_limit_skip_driven = 1 if (self.fuse_joint_velocity_limits and self.drive_slot is not None) else 0
        if self._vel_limit_skip_driven:
            self._vel_limit_drive_slot_arg = self.drive_slot
        else:
            self._vel_limit_drive_slot_arg = self._arrays.placeholder(wp.int32, (1,), -1)
        # The drive-row populate kernel reads joint_velocity_limit only when the fused clamp is enabled.
        if self.fuse_joint_velocity_limits and model.joint_velocity_limit is None:
            raise ValueError("fuse_joint_velocity_limits=True requires model.joint_velocity_limit to be allocated")

    def _allocate_world_buffers(self, model):
        """Allocate world-level constraint system buffers for multi-articulation support."""
        if self.world_count == 0:
            return

        arrays = self._arrays
        # The widest world's DOFs exist where a world response mapping was built (the routes that read them).
        world_dof = getattr(self, "max_world_dofs", None)
        sizes = {"world": self.world_count, "row": self.dense_max_constraints, "world_dof": world_dof}
        # Per-world constraint matrices and vectors. Matrix-free never assembles C.
        if self.pgs_mode != "matrix_free" and not self._sparse_split:
            arrays.allocate(DENSE_DELASSUS, **sizes)
        else:
            self.C = None

        # Matrix-free uses world-indexed J/Y for both dense and rigid phases.
        if self.pgs_mode == "matrix_free" or self._sparse_split:
            if self._sparse_diagonal_contact_solve or self._sparse_mass_matrix_size is not None:
                arrays.allocate(WORLD_RESPONSE, requires_grad=False, world=1, row=1, world_dof=1)
            elif self._mf_sweeps:
                # The matrix-free sweeps read the packed rows (_place_dense_response): no world layout.
                arrays.release(WORLD_RESPONSE)
            elif self._jy_world_aliased:
                # The world gather (gather_coordinates_kernel) is an element-for-element identity copy for
                # this model layout (one articulation per world, single size
                # group, matching dof offsets): alias the world-indexed views
                # directly onto the group buffers and skip the gather launch
                # plus the duplicate (world, max_constraints, dofs) storage.
                size = self.size_groups[0]
                self.J_world = self.J_by_size[size]
                self.Y_world = self.Y_by_size[size]
                arrays.mirror(WORLD_RESPONSE)
            else:
                arrays.allocate(WORLD_RESPONSE, **sizes)
        else:
            self._jy_world_aliased = False
            arrays.release(WORLD_RESPONSE)

        arrays.allocate(DENSE_ROWS, **sizes)
        # Dense identity-matched warm-start carry buffers. Allocated here (not in the MF
        # buffer path) so scenes WITHOUT free rigid bodies still get them —
        # _allocate_mf_buffers early-returns for such scenes and the matched
        # carry must never silently degrade to index reuse.
        if self.pgs_warmstart:
            arrays.allocate(DENSE_WARMSTART, **sizes)
            self._ws_prev_dense_slot_rows = self._contact_pool.rows(
                (ContactField("_ws_prev_dense_slot_sorted", wp.int32, -1),), on_rows=self._publish_contact_rows
            )
        fused = self._fused_diagonal_joint_limits
        arrays.allocate(FUSED_DIAGONAL_LIMITS, **(sizes if fused else {"world": 1, "world_dof": 1}))
        # Per-row weight is needed only when regularization is active because
        # speculative and restitution rows selectively use the rigid law.
        if self._regularization_enabled:
            arrays.allocate(DENSE_ROW_WEIGHTS, **sizes)
        else:
            self.row_w = self._contact_row_w_dummy
        # Per-dense-row joint velocity limit for the fused drive-row clamp
        # (fuse_joint_velocity_limits). Only allocated at full size when the
        # feature is on; otherwise a (1, 1) dummy keeps the drive populate
        # kernel signature satisfied without the memory cost (the kernel
        # never writes it when fuse_vel_limits == 0).
        if self.fuse_joint_velocity_limits:
            arrays.allocate(DENSE_DRIVE_LIMITS, **sizes)
        else:
            arrays.allocate(DENSE_DRIVE_LIMITS, requires_grad=False, world=1, row=1)

    def _allocate_mf_buffers(self, model):
        """Allocate buffers for matrix-free PGS path for free rigid body contacts.

        ``pgs_mode == "matrix_free"`` without free rigid bodies still passes the
        ``mf_*`` arrays to kernels (the fused GS solve, the PGS diagnostics),
        but ``mf_constraint_count`` stays zero (``_clear_dense_row_state`` and
        ``finalize_constraint_counts_with_status`` write it only when MF is
        active), so every access sits inside a loop bounded by
        ``m_mf == 0`` and no element is ever read or written; :meth:`_pack_mf_meta`
        skips its launch. The rows then hold one slot per world, the per-world
        counts keep their real width, and the split-only body maps
        (``mf_body_list`` et al.) are not allocated, so the
        tiled MF PGS kernel is not built.
        """
        if not self._has_free_rigid_bodies and self.pgs_mode != "matrix_free" and not self._sparse_split:
            return

        arrays = self._arrays
        worlds = self.world_count
        rows = self.mf_max_constraints if self._has_free_rigid_bodies else 1
        arrays.allocate(MF_COUNTERS, world=worlds)
        arrays.allocate(MF_ROWS, world=worlds, mf_row=rows, mf_meta=4 * rows)
        if self._regularization_enabled and self._has_free_rigid_bodies:
            arrays.allocate(MF_ROW_WEIGHTS, world=worlds, mf_row=rows)
        else:
            self.mf_row_w = self._contact_row_w_dummy

        if not self._has_free_rigid_bodies:
            # Only the ``is not None`` checks observe this on the inactive
            # path, where no free rigid body owns a speed row.
            self.rigid_velocity_limit_slot = (
                wp.full((1,), -1, dtype=wp.int32, device=model.device, requires_grad=model.requires_grad)
                if self._has_rigid_body_velocity_limits and model.body_count > 0
                else None
            )
            return

        # MF warm-start carry buffers. Allocated ONLY when the feature is on so
        # the default-off path has zero memory / CUDA-graph delta. ``prev_mf_*``
        # snapshot last step's converged impulses + their row-type table;
        # ``_ws_prev_slot_sorted`` records, per *current sorted contact index*,
        # the base MF slot that contact occupied last step. Because FeatherPGS
        # consumes the already-sorted contact arrays, the contact array index is
        # the sorted index, so next step's ``rigid_contact_match_index`` (which
        # references the previous frame's *sorted* index) indexes directly into
        # this table — no matcher permutation access required.
        if self._mf_warmstart_enabled:
            arrays.allocate(MF_WARMSTART, world=worlds, mf_row=rows)
            pool = getattr(self, "_contact_pool", None)
            if pool is None:
                self._ws_prev_slot_sorted = wp.full((1,), -1, dtype=wp.int32, device=model.device)
            else:
                self._ws_prev_slot_rows = pool.rows(
                    (ContactField("_ws_prev_slot_sorted", wp.int32, -1),), on_rows=self._publish_contact_rows
                )

        if self._has_rigid_body_velocity_limits and self._free_rigid_body_count > 0:
            # Two speed rows per free rigid body: its linear and its angular speed.
            self.rigid_velocity_limit_slot = wp.full(
                (RIGID_SPEED_ROWS * self._free_rigid_body_count,),
                -1,
                dtype=wp.int32,
                device=model.device,
                requires_grad=model.requires_grad,
            )
        else:
            self.rigid_velocity_limit_slot = None

        # Body map buffers for tiled MF PGS kernel
        self.max_mf_bodies = 64
        arrays.allocate(MF_BODY_MAP, world=worlds, mf_row=rows, body=model.body_count, mf_body=self.max_mf_bodies)

    def _allocate_debug_buffers(self, model):
        """Allocate buffers for PGS convergence diagnostics."""
        if not self.pgs_debug:
            return
        sizes = {"world": self.world_count, "row": self.dense_max_constraints, "mf_row": self.mf_max_constraints}
        self._arrays.allocate(PGS_DIAGNOSTICS, **sizes)
        if hasattr(self, "mf_impulses"):
            self._arrays.allocate(PGS_MF_DIAGNOSTICS, **sizes)
        else:
            self._diag_prev_mf_impulses = None

    def _scatter_armature_to_groups(self):
        """Copy armature from model (DOF-ordered) to size-grouped storage."""
        if not self.size_groups:
            return

        armature_np = self._joint_armature_effective
        art_dof_start_np = self._model_plan.articulation_dof_start
        art_H_rows_np = self._model_plan.articulation_dof_count

        # R_by_size is sized to actual DOF count (matches H_by_size allocation)
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            R_np = np.zeros((n_arts, size), dtype=np.float32)

            group_to_art_np = np.flatnonzero(self._model_plan.response_dof_count == size)
            for group_idx in range(n_arts):
                art_idx = group_to_art_np[group_idx]
                dof_start = art_dof_start_np[art_idx]
                dof_count = art_H_rows_np[art_idx]
                R_np[group_idx, :dof_count] = armature_np[dof_start : dof_start + dof_count]

            self.R_by_size[size].assign(R_np)

    def _init_tiled_kernels(self, model):
        """Build the paired response and the sweeps.

        The factor, stage-3 and stage-4 kernels follow the step plan (:meth:`_build_stage_kernels`).
        """
        device_arch = model.device.arch
        from . import paired_components  # noqa: PLC0415

        self._paired_components = None
        self._paired_components_reason = (
            "FEATHER_PGS_PAIRED_COMPONENTS disables independent components."
            if os.environ.get("FEATHER_PGS_PAIRED_COMPONENTS", "1") != "1"
            else paired_components.decline(self)
        )
        if self._paired_components_reason is None:
            self._paired_components = paired_components.PairedComponents(self)
            self._paired_components_reason = (
                "Admitted: each paired world splits into independent articulation and free-body components."
            )
        if self._paired_response_primary_size is not None:
            primary_size = self._paired_response_primary_size
            secondary_size = self._paired_response_secondary_size
            self._paired_factor_solve_kernel = (
                self._sweep_kernel(
                    "paired_factor",
                    sweep_routes_local.paired_factor_kernel,
                    self.dense_max_constraints,
                    self.max_world_dofs,
                    primary_size,
                    secondary_size,
                    device_arch,
                    contact_triples=self._factor_coordinate_contact_triples,
                    independent_components=self._paired_components is not None,
                    contact_units=self._factor_coordinate_contact_units,
                )
                if self._paired_factor_coordinates
                else None
            )
        else:
            self._paired_factor_solve_kernel = None

        self._pgs_solve_tiled_row_kernel = None
        self._pgs_solve_tiled_row_budgets = ()
        self._tiled_row_staging = None
        self._tiled_row_staging_reason = None
        if self.dense_max_constraints > 0:
            # Only split mode without native sparse split rows sweeps the assembled Delassus rows.
            if self.pgs_kernel == "tiled_row" and self.pgs_mode == "split" and not self._sparse_split:
                self._tiled_row_staging, self._tiled_row_staging_reason = sweep_routes_diag.tiled_row_staging(
                    self.dense_max_constraints, arch_budget(device_arch)
                )
                self._pgs_solve_tiled_row_kernel = self._sweep_kernel(
                    "tiled_row",
                    sweep_routes_diag.tiled_row_kernel,
                    self.dense_max_constraints,
                    device_arch,
                    staging=self._tiled_row_staging,
                )
                # Large C128 batches benefit from small-world storage without
                # reducing capacity. Avoid extra launches for small batches.
                row_budgets = os.environ.get("FEATHER_PGS_DENSE_ROW_BUDGETS")
                if (
                    model.device.is_cuda
                    and self.dense_max_constraints in (64, 128)
                    and (
                        row_budgets == "1"
                        or (row_budgets is None and self.dense_max_constraints == 128 and self.world_count >= 1024)
                    )
                ):
                    self._pgs_solve_tiled_row_budgets = tuple(
                        self._sweep_kernel(
                            "tiled_row",
                            sweep_routes_diag.tiled_row_kernel,
                            self.dense_max_constraints,
                            device_arch,
                            row_budget=budget,
                            min_rows=minimum,
                        )
                        for budget, minimum in ((32, 0), (self.dense_max_constraints, 32))
                    )

        # Only free rigid bodies produce matrix-free rows, which matrix-free mode packs each step.
        self._pack_mf_meta_kernel = (
            response_builds_rows.free_body_meta_kernel(self.mf_max_constraints, device_arch)
            if self.pgs_mode == "matrix_free" and self._has_free_rigid_bodies and self.mf_max_constraints > 0
            else None
        )

        self._pgs_solve_local_internal_kernels = {}
        self._pgs_solve_local_pair_kernels = {}
        self._pgs_solve_local_residual_kernels = {}
        self._pgs_solve_local_internal_warps_per_block = {}
        self._pgs_solve_local_internal_worlds_per_block = {}
        self._pgs_solve_local_pair_warps_per_block = {}
        local_factory = sweep_routes_local.local_owned_kernel
        if self._row_packets is not None:
            local_factory = self._row_packets.local_factory()
        if self._local_internal_fast_path:
            for size, count in self._local_internal_art_counts.items():
                if count > 0:
                    lanes_per_world = 32
                    if size <= 16:
                        for candidate_lanes in (8, 16):
                            if count % (32 // candidate_lanes) == 0:
                                lanes_per_world = candidate_lanes
                                break
                    worlds_per_warp = 32 // lanes_per_world
                    warps_per_block = 2 if count % (2 * worlds_per_warp) == 0 else 1
                    self._pgs_solve_local_internal_warps_per_block[size] = warps_per_block
                    self._pgs_solve_local_internal_worlds_per_block[size] = warps_per_block * 32 // lanes_per_world
                    self._pgs_solve_local_internal_kernels[size] = self._sweep_kernel(
                        "local_internal",
                        local_factory,
                        self.dense_max_constraints,
                        min(self._local_solve_max_rows, size),
                        size,
                        device_arch,
                        warps_per_block=warps_per_block,
                        lanes_per_world=lanes_per_world,
                        contact_capable=False,
                    )
            for size, count in self._local_pair_art_counts.items():
                if count > 0:
                    solver_blocks = self._local_pair_solver_blocks[size]
                    warps_per_block = 2 if solver_blocks % 2 == 0 else 1
                    self._pgs_solve_local_pair_warps_per_block[size] = warps_per_block
                    self._pgs_solve_local_pair_kernels[size] = self._sweep_kernel(
                        "local_pair",
                        local_factory,
                        self.dense_max_constraints,
                        self._local_solve_max_rows,
                        size,
                        device_arch,
                        paired_dof_count=6,
                        persistent_queue=True,
                        warps_per_block=warps_per_block,
                    )
            for size, count in self._local_residual_art_counts.items():
                if count > 0:
                    warps_per_block = self._local_residual_warps_per_block[size]
                    self._pgs_solve_local_residual_kernels[size] = self._sweep_kernel(
                        "local_residual",
                        local_factory,
                        self.dense_max_constraints,
                        self._local_residual_max_rows,
                        size,
                        device_arch,
                        paired_dof_count=6,
                        persistent_queue=True,
                        warps_per_block=warps_per_block,
                        mf_max_constraints=self.mf_max_constraints,
                        local_mf_max_constraints=self._local_residual_mf_max_rows,
                    )

        sparse_contact_schedule = self._sparse_diagonal_contact_triples or self._sparse_diagonal_contact_units
        self._mark_independent_sparse_contact_candidates_kernel = (
            _get_mark_independent_sparse_contact_candidates_kernel(
                self._sparse_diagonal_response_size,
                device_arch,
                units=self._sparse_diagonal_contact_units,
            )
            if sparse_contact_schedule
            else None
        )
        self._build_independent_sparse_contact_groups_kernel = (
            _get_build_independent_sparse_contact_groups_kernel(
                self.dense_max_constraints,
                self.max_world_dofs,
                device_arch,
                build_serial_contacts=self._sparse_diagonal_speculative_contact_batches,
                units=self._sparse_diagonal_contact_units,
            )
            if sparse_contact_schedule
            else None
        )
        self._pgs_solve_sparse_diagonal_kernel = (
            self._sweep_kernel(
                "sparse_diagonal",
                sweep_routes_diag.sparse_diagonal_kernel,
                self.dense_max_constraints,
                self.max_world_dofs,
                self._sparse_diagonal_dense_size,
                device_arch,
                contact_triples=self._sparse_diagonal_contact_triples,
                speculative_contact_batches=self._sparse_diagonal_speculative_contact_batches,
                contact_units=self._sparse_diagonal_contact_units,
            )
            if self._sparse_diagonal_contact_solve
            else None
        )
        self._pgs_solve_mf_gs_kernel = None
        # The colored mapping of the matrix-free sweep and its units, when a world can take it.
        self._pgs_solve_mf_colored_kernel = None
        self._colored_units = None
        # Storage classes of the MF-GS kernel (_MFGSStorage); recorded by get_path_selection().
        self._mfgs_storage = None
        # Blocks of every MF-GS sweep launch: one per world, or the general-world queue's grid. The
        # storage selection keeps row metadata resident for this grid, so launches must use it too.
        self._mf_gs_block_count = int(
            self._local_general_solver_blocks if self._local_internal_fast_path else self.world_count
        )
        # The tier sweeps ahead of the matrix-free sweep (SweepTier): parallel tiers or incremental tiers.
        self._mf_gs_tiers = []
        if model.device.is_cuda and int(model.articulation_count) > 0:
            # The grouped response solve (_stage4_hinv_jt_grouped), for the sizes StepPlan gives it from the data.
            self._grouped_response = True
            self._grouped_dummy_diag = wp.zeros((1, 1), dtype=wp.float32, device=model.device)
        if self._mf_sweeps:
            has_drive_rows = self.drive_mode == "physx_pgs"
            # Preserve the explicit diagnostic override without hiding it in a
            # cached factory: storage is part of the kernel specialization key.
            # The selection sizes the resident row metadata with its choice.
            force_stream_mf_impulses = os.environ.get("FEATHER_PGS_MFGS_STREAM_LAMBDA", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            storage = _MFGSStorage(
                False, force_stream_mf_impulses, False, "Without CUDA no shared-memory storage applies."
            )
            if model.device.is_cuda:
                storage = _execution_plan._select_mfgs_storage(
                    functools.partial(
                        sweep_routes.mf_gs_staging,
                        self.dense_max_constraints,
                        self.mf_max_constraints,
                        self.max_world_dofs,
                        has_drive_rows=has_drive_rows,
                        fuse_vel_limits=self.fuse_joint_velocity_limits,
                    ),
                    arch_budget(device_arch),
                    block_count=self._mf_gs_block_count,
                    sm_count=int(model.device.sm_count),
                    force_stream_mf_impulses=force_stream_mf_impulses,
                )
            self._mfgs_storage = storage
            parallel_rows, incremental_rows = self._ink_rows, self._incremental_rows
            # The colored mapping takes the worlds the unit builder's rule picks at every pack
            # (sweep_units.COLORED_RULE), among those whose free bodies and row capacity admit it: built, with a
            # storage slot each, for those worlds only, it runs beside the one-warp mapping. More of them than the
            # device holds colored blocks run in waves, which scales the rule (ColoredRule.at_scale).
            rule = sweep_units.COLORED_RULE
            capable = sweep_units.capable_worlds(self._free_rigid_per_world, self.mf_max_constraints, rule)
            if model.device.is_cuda and capable.size > 0:
                rule = rule.at_scale(capable.size, model.device.sm_count)
                capable = sweep_units.capable_worlds(self._free_rigid_per_world, self.mf_max_constraints, rule)
            colored = model.device.is_cuda and capable.size > 0
            mf_gs_options = {
                "has_drive_rows": has_drive_rows,
                "has_dense_velocity_limit_rows": self.enable_joint_velocity_limits,
                "fuse_vel_limits": self.fuse_joint_velocity_limits,
                "shared_metadata": storage.shared_metadata,
                "resident_row_metadata": storage.resident_row_metadata,
                "stream_mf_impulses": storage.stream_mf_impulses,
                "skip_local_internal_worlds": self._local_internal_fast_path,
                "local_internal_max_constraints": self._dense_internal_max_rows,
                "contact_torsion": self._contact_torsion_enabled,
                "factor_coordinates": self._paired_factor_coordinates,
                "skip_incremental_rows": max(incremental_rows, parallel_rows),
                "independent_components": self._paired_components is not None,
                "colored_worlds": colored,
                "max_row_dofs": self._packed_support,
            }
            mf_gs_shape = (self.dense_max_constraints, self.mf_max_constraints, self.max_world_dofs, device_arch)
            self._pgs_solve_mf_gs_kernel = self._sweep_kernel(
                "mf_gs",
                sweep_routes.mf_gs_kernel,
                *mf_gs_shape,
                velocity_limit_law=self._velocity_limit_kernel_law(),
                **mf_gs_options,
            )
            if colored:
                self._colored_units = sweep_units.ColoredUnits(
                    self.world_count,
                    self.mf_max_constraints,
                    self.max_world_dofs,
                    model.device,
                    worlds=capable,
                    rule=rule,
                )
                self._pgs_solve_mf_colored_kernel = self._sweep_kernel(
                    "mf_colored",
                    sweep_routes.mf_gs_kernel,
                    *mf_gs_shape,
                    mapping="colored",
                    velocity_limit_law=self._velocity_limit_kernel_law(),
                    **mf_gs_options,
                )
            if parallel_rows > 0:
                parallel_factory = sweep_routes_tiers.parallel_kernel
                if self._keyed_rows is not None:
                    from .keyed_rows import get_parallel_factory  # noqa: PLC0415

                    parallel_factory = get_parallel_factory(parallel_factory, self._keyed_rows.layout)
                tiers = [(min(32, parallel_rows), 0)]
                if parallel_rows > 32:
                    tiers.append((min(64, parallel_rows), 32))
                if parallel_rows > 64:
                    tiers.append((min(96, parallel_rows), 64))
                if parallel_rows > 96:
                    tiers.append((parallel_rows, 96))
                self._par_tiers = list(tiers)
                if _TIER_BLOCKS > 0:
                    dev = model.device
                    self._tier_lo = wp.array([lo for _, lo in tiers], dtype=wp.int32, device=dev)
                    self._tier_hi = wp.array([hi for hi, _ in tiers], dtype=wp.int32, device=dev)
                    self._tier_counts = wp.zeros(len(tiers), dtype=wp.int32, device=dev)
                    self._tier_worlds = wp.empty((len(tiers), self.world_count), dtype=wp.int32, device=dev)
                    self._tier_count_views = [self._tier_counts[t : t + 1] for t in range(len(tiers))]
                    self._tier_world_views = [self._tier_worlds[t] for t in range(len(tiers))]
                    self._tier_blocks = int(min(self.world_count, _TIER_BLOCKS))
                tier_sweeps, self._par_sweeps_rule, self._par_sweeps_reason = self._parallel_tier_sweeps(tiers)
                for tier_index, ((rows_, min_rows_), sweeps_) in enumerate(zip(tiers, tier_sweeps, strict=True)):
                    tier = self._sweep_kernel(
                        "parallel_tier",
                        parallel_factory,
                        self.dense_max_constraints,
                        self.mf_max_constraints,
                        self.max_world_dofs,
                        device_arch,
                        readout_budget=sweeps_,
                        rows=rows_,
                        sweeps=sweeps_,
                        nesterov=self.mf_gs_parallel_nesterov,
                        min_rows=min_rows_,
                        matrix_free=self.mf_gs_parallel_matrix_free,
                        skip_local_internal_worlds=self._local_internal_fast_path,
                        inkernel_response=self._ink_sizes,
                        has_drive_rows=has_drive_rows,
                        exact_row_sums=_MF_EXACT_ROWSUM,
                        linked_rows=self._friction_anchors_enabled,
                        velocity_limit_pass=self._parallel_velocity_limit_pass(),
                    )
                    self._mf_gs_tiers.append(dataclasses.replace(tier, tier=tier_index))
            if incremental_rows > 0:
                # Two tiers above 32 rows: the 32-row instance keeps its 4 KB response block and
                # occupancy; the wider instance (two rows per lane) takes worlds with 33..N rows.
                tiers = [(min(32, incremental_rows), 0)]
                if incremental_rows > 32:
                    tiers.append((incremental_rows, 32))
                for rows, min_rows in tiers:
                    kernel = self._sweep_kernel(
                        "mf_gs_incremental",
                        sweep_routes.incremental_kernel,
                        self.dense_max_constraints,
                        self.mf_max_constraints,
                        self.max_world_dofs,
                        device_arch,
                        incremental_rows=rows,
                        max_row_dofs=self._packed_support,
                        has_drive_rows=has_drive_rows,
                        has_dense_velocity_limit_rows=self.enable_joint_velocity_limits,
                        skip_local_internal_worlds=self._local_internal_fast_path,
                        min_rows=min_rows,
                        velocity_limit_law=self._velocity_limit_kernel_law(),
                    )
                    self._mf_gs_tiers.append(SweepTier(kernel, velocity_limit_law=self._velocity_limit_kernel_law()))
            # No matrix-free sweep enters a world without rows; the readout counts those solves as skipped.
            self._solve_readout.register_skipped(
                world_count=int(self.world_count),
                device=model.device,
                velocity_budget=int(self.pgs_velocity_iterations),
            )

        # The gate of the row builder (_stage4_build_rows): it allocates free-body velocity-limit rows into
        # rigid_velocity_limit_slot when free rigid bodies exist, and an infinite activation fraction reserves
        # none. The split free-body sweep compiles their law wherever the builder can allocate them, even when
        # every registered limit is infinite and no row is built.
        self._has_free_body_velocity_limit_rows = bool(
            self._has_free_rigid_bodies
            and getattr(self, "rigid_velocity_limit_slot", None) is not None
            and not np.isinf(self.velocity_limit_activation_fraction)
        )
        # Split mode's free-body sweep (the CPU runs its loop sweep, _setup_contact_update).
        self._pgs_solve_mf_kernel = None
        if (
            model.device.is_cuda
            and self.pgs_mode == "split"
            and not self._sparse_split
            and hasattr(self, "max_mf_bodies")
            and self.mf_max_constraints > 0
        ):
            self._pgs_solve_mf_kernel = self._sweep_kernel(
                "free_body_split",
                sweep_routes_diag.mf_split_kernel,
                self.mf_max_constraints,
                self.max_mf_bodies,
                device_arch,
                has_velocity_limit_rows=self._has_free_body_velocity_limit_rows,
            )

    def _init_size_group_streams(self, model):
        """Initialize CUDA streams for parallel kernel launches across size groups.

        When multiple DOF sizes exist (heterogeneous articulations), we can launch
        tiled kernels for different sizes in parallel using separate CUDA streams.

        None of this solver's side streams (size groups, local-owner forks, the
        paired sweep, inertia, dynamics, memset) reads contact rows or the contact
        pool's ready count: they run factorizations, response and row sweeps over
        dense and matrix-free rows, and articulation data, while contact rows are
        read on the launch stream, which the pool registers. A trim therefore
        waits for none of them (:meth:`ContactPool.add_reader_stream`).
        """
        self._size_streams: dict[int, wp.Stream | None] = {}
        self._size_events: dict[int, wp.Event | None] = {}

        if self.use_parallel_streams and model.device.is_cuda and len(self.size_groups) > 1:
            for size in self.size_groups:
                self._size_streams[size] = wp.Stream(model.device)
                self._size_events[size] = wp.Event(model.device)
        else:
            # No streams needed for CPU or single size group
            for size in self.size_groups:
                self._size_streams[size] = None
                self._size_events[size] = None
        self._local_internal_stream = (
            wp.Stream(model.device)
            if self.use_parallel_streams
            and model.device.is_cuda
            and (
                self._pgs_solve_local_internal_kernels
                or self._pgs_solve_local_pair_kernels
                or self._pgs_solve_local_residual_kernels
            )
            else None
        )
        self._local_residual_stream = (
            wp.Stream(model.device, priority=-1)
            if self._local_internal_stream is not None and self._pgs_solve_local_residual_kernels
            else None
        )
        self._local_pair_stream = (
            wp.Stream(model.device)
            if self._local_internal_stream is not None
            and self._pgs_solve_local_internal_kernels
            and self._pgs_solve_local_pair_kernels
            else None
        )
        # With independent components the paired factor sweep and the general
        # matrix-free sweep touch disjoint coordinates of a split world, so
        # they fork after response preparation and join before publication.
        # Components are admitted on CUDA without gradients, under the factor-coordinate sweep's plain policy.
        paired_overlap = bool(
            self._paired_components is not None
            and self.use_parallel_streams
            and self._paired_factor_solve_kernel is not None
            and not getattr(self, "_mf_gs_tiers", ())
            and not _FPGS_SYNC_TIMINGS
        )
        self._paired_general_solve_stream = wp.Stream(model.device) if paired_overlap else None
        self._paired_general_ready = wp.Event(model.device) if paired_overlap else None
        self._paired_general_done = wp.Event(model.device) if paired_overlap else None

    def _init_double_buffer_stream(self):
        """Create a CUDA stream for mass-matrix and active-Jacobian maintenance."""
        if self._H_bufs is None or not self.model.device.is_cuda:
            self._memset_stream = None
            self._j_active_counts = None
            return
        self._memset_stream = wp.Stream(self.model.device)
        self._j_active_counts = [wp.zeros_like(self.constraint_count), wp.zeros_like(self.constraint_count)]
        # Track the last memset-done event per buffer slot so the main stream
        # can wait only for the specific buffer it needs, and whether it is a
        # seeded event (see _wait_memset_done).
        self._memset_done_event: list[wp.Event | None] = [None, None]
        self._memset_done_seeded = [False, False]

    def seed_double_buffer_events(self):
        """Record initial memset_done events on the main stream.

        Must be called inside CUDA graph capture, before the first ``step()`` call.
        Seeded events are not waited on: the first two captured substeps rely on
        their buffers being clear already. That holds for the ``wp.zeros()``
        allocation, but each :meth:`step` that ran before the capture queued
        clears on the memset stream that nothing in the graph waits for. After
        such steps, synchronize the device (for example with
        ``wp.synchronize_device()``) before capturing.
        """
        if self._memset_stream is None:
            return
        main_stream = wp.get_stream(self.model.device)
        self._record_memset_done(0, main_stream.record_event(), seeded=True)
        self._record_memset_done(1, main_stream.record_event(), seeded=True)

    def _record_memset_done(self, slot, event, *, seeded=False):
        """Record the event a buffer slot's consumer waits for."""
        self._memset_done_event[slot] = event
        self._memset_done_seeded[slot] = seeded

    def _wait_memset_done(self, stream, slot):
        """Make ``stream``, the main stream, wait for a buffer slot's memset.

        A seeded event, recorded on the main stream itself, orders nothing, so it
        is not waited on: a capture paused since it was recorded, for example by
        another solver's conditional node, would reject the wait.
        """
        if self._memset_done_event[slot] is not None and not self._memset_done_seeded[slot]:
            stream.wait_event(self._memset_done_event[slot])

    def reset_diagnostic_logs(self) -> None:
        """Clear per-step PGS diagnostic logs populated when ``pgs_debug=True``.

        Deprecated: no Newton code calls it, and it will be removed in a future release.

        :attr:`_pgs_convergence_log` and :attr:`_pgs_ncp_residual_log` are
        append-only across :meth:`step` calls so one solver instance can
        accumulate a trace across many frames.  A replay / sweep harness
        that reuses a single solver to scan ``pgs_iterations`` over a
        snapshot needs the logs reset between sweeps so each entry
        corresponds to exactly one replayed step.  ``pgs_warmstart``
        impulses and all device-side solver buffers are left untouched.
        """
        warnings.warn(
            "SolverFeatherPGS.reset_diagnostic_logs() is deprecated and will be removed in a future release; "
            "the pgs_debug logs it clears are private.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._pgs_convergence_log = []
        self._pgs_ncp_residual_log = []

    def _contact_friction_start_iteration(self, iterations: int, *, velocity_pass: bool = False) -> int:
        """Return the first local PGS iteration that should solve friction rows.

        ``contact_friction_position_iterations=-1`` preserves the legacy all-iteration
        position-pass behavior. Velocity-only post passes intentionally ignore the
        position-pass limit and solve friction every iteration, matching PhysX PGS.
        """
        if velocity_pass or self.contact_friction_position_iterations < 0:
            return 0
        if iterations <= 0:
            return 0
        active = min(self.contact_friction_position_iterations, iterations)
        return iterations - active

    def _pack_mf_meta(self, mf_rhs: wp.array) -> None:
        """Pack matrix-free row metadata for the GS kernels, or skip when no such row can exist."""
        if not self._has_free_rigid_bodies:
            # Only free rigid bodies produce matrix-free rows, and the model plan fixes
            # them at construction: mf_constraint_count stays zero on the one-slot
            # dummy buffers, so the pack would write nothing, and no pack kernel is built.
            return
        wp.launch_tiled(
            self._pack_mf_meta_kernel,
            dim=[self.world_count],
            inputs=[
                self.mf_constraint_count,
                self.mf_dof_a,
                self.mf_dof_b,
                self.mf_eff_mass_inv,
                mf_rhs,
                self.mf_row_type,
                self.mf_row_parent,
            ],
            outputs=[self.mf_meta_packed],
            block_dim=32,
            device=self.model.device,
        )
        if self._colored_units is not None:
            # The colored worlds' units and colors follow the rows the pack just published.
            self._colored_units.build(
                self.mf_constraint_count,
                self.mf_contact_rows_end,
                self.mf_meta_packed,
                self.mf_Z_a,
                self.mf_Z_b,
                self.mf_row_mu,
                self.mf_row_w,
            )

    def _launch_local_internal_solve(
        self,
        dense_rhs: wp.array,
        iterations: int,
        omega: float,
        friction_start_iteration: int,
        iteration_offset: int,
        *,
        solve_single: bool = True,
        solve_pair: bool = True,
        solve_residual: bool = True,
    ) -> None:
        """Solve articulation-local dense rows without world response buffers."""

        def launch(
            kernel,
            primary_candidates: wp.array,
            secondary_candidates: wp.array,
            primary_size: int,
            secondary_size: int,
            owner: int,
            count: int,
            active_count: wp.array,
            use_active_queue: bool,
            warps_per_block: int = 1,
            worlds_per_block: int | None = None,
        ) -> None:
            if worlds_per_block is None:
                worlds_per_block = warps_per_block
            secondary_group_size = secondary_size if secondary_size > 0 else primary_size
            inputs = [primary_candidates, secondary_candidates]
            if use_active_queue:
                inputs.extend([active_count, count])
            inputs.extend(
                [
                    owner,
                    self.art_group_idx,
                    self.art_to_world,
                    self.articulation_dof_start,
                    self._local_solve_owner,
                    self.constraint_count,
                    self.L_by_size[primary_size],
                    self.L_by_size[secondary_group_size],
                    self._packed_J,
                    self._row_sides,
                    self.articulation_response_dof_count,
                    self.articulation_world_dof_offset,
                    int(self._packed_dense),
                    dense_rhs,
                    self.row_type,
                    self.row_parent,
                    self.row_mu,
                    self.mf_constraint_count,
                    self.mf_meta_packed,
                    self.mf_impulses,
                    self.mf_Z_a,
                    self.mf_Z_b,
                    self.mf_row_mu,
                    int(self._regularization_enabled),
                    self.row_w,
                    self.mf_row_w,
                    iterations,
                    omega,
                    friction_start_iteration,
                    iteration_offset,
                ]
            )
            outputs = [self.diag, self.impulses, self.v_out]
            if self._row_packets is not None:
                outputs.extend([self._row_packets.prefix, self._row_packets.packet])
            outputs.extend(self._solve_readout.operands(kernel))
            wp.launch_tiled(
                kernel,
                dim=[count // worlds_per_block],
                inputs=inputs,
                outputs=outputs,
                block_dim=32 * warps_per_block,
                device=self.model.device,
            )

        if solve_single:
            for size, kernel in self._pgs_solve_local_internal_kernels.items():
                launch(
                    kernel,
                    self._local_internal_candidates[size],
                    self._local_internal_candidates[size],
                    size,
                    0,
                    PGS_LOCAL_SOLVE_OWNER_SINGLE,
                    self._local_internal_art_counts[size],
                    self._local_general_world_count,
                    False,
                    self._pgs_solve_local_internal_warps_per_block[size],
                    self._pgs_solve_local_internal_worlds_per_block[size],
                )
        if solve_pair:
            for size, kernel in self._pgs_solve_local_pair_kernels.items():
                launch(
                    kernel,
                    self._local_pair_active_candidates[size],
                    self._local_pair_active_secondaries[size],
                    size,
                    6,
                    PGS_LOCAL_SOLVE_OWNER_PAIR,
                    self._local_pair_solver_blocks[size],
                    self._local_pair_active_counts[size],
                    True,
                    self._pgs_solve_local_pair_warps_per_block[size],
                )
        if solve_residual:
            for size, kernel in self._pgs_solve_local_residual_kernels.items():
                launch(
                    kernel,
                    self._local_residual_active_candidates[size],
                    self._local_residual_active_secondaries[size],
                    size,
                    6,
                    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
                    self._local_residual_solver_warps[size],
                    self._local_residual_active_counts[size],
                    True,
                    self._local_residual_warps_per_block[size],
                )

    def _launch_matrix_free_gs_solve(
        self,
        *,
        dense_rhs: wp.array,
        mf_meta: wp.array,
        iterations: int,
        omega: float,
        friction_start_iteration: int | None = None,
        iteration_offset: int = 0,
        freeze_drive_rows: bool = False,
        soft_relax: bool = False,
    ) -> None:
        if iterations <= 0:
            return
        if friction_start_iteration is None:
            friction_start_iteration = self._contact_friction_start_iteration(iterations)
        route = self._plan.solve.route.name
        if route == "sparse_pgs":
            if soft_relax:
                raise RuntimeError("sparse factor response requires the interleaved augmented-drive solve")
            worlds_per_block = 64 // self._sparse_solve_lanes
            wp.launch_tiled(
                self._pgs_solve_sparse_kernel,
                dim=[(self.world_count + worlds_per_block - 1) // worlds_per_block],
                inputs=[
                    self.world_count,
                    self.constraint_count,
                    dense_rhs,
                    self.diag,
                    self.impulses,
                    self._sparse_row_dof,
                    self._sparse_row_factor,
                    self._sparse_row_incident,
                    self.row_type,
                    self.row_parent,
                    self.row_mu,
                    self.world_dof_indices,
                    self.v_hat,
                    self.mf_constraint_count,
                    mf_meta,
                    self.mf_impulses,
                    self.mf_Z_a,
                    self.mf_Z_b,
                    self.mf_row_mu,
                    iterations,
                    omega,
                    int(friction_start_iteration),
                    int(iteration_offset),
                    # The single-stage transaction reads the triplets' block records.
                    *(() if self._sparse_contact_records is None else (self._sparse_contact_records,)),
                ],
                outputs=[
                    self._sparse_factor_velocity_delta,
                    self._sparse_row_support,
                    *self._solve_readout.operands(self._pgs_solve_sparse_kernel),
                ],
                block_dim=64,
                device=self.model.device,
            )
            indices = self._sparse_mass_matrix_indices
            size = self._sparse_mass_matrix_size
            wp.launch(
                response_builds_rows.factor_publish_kernel,
                dim=(self.n_arts_by_size[size], size),
                inputs=[
                    self.group_to_art[size],
                    self.art_to_world,
                    self.articulation_world_dof_offset,
                    self.articulation_dof_start,
                    indices.permutation,
                    indices.lookup,
                    self._sparse_Linv,
                    self._sparse_factor_velocity_delta,
                    self.v_hat,
                ],
                outputs=[self.v_out],
                device=self.model.device,
            )
            return
        if route == "sparse_diagonal":
            kernel = self._pgs_solve_sparse_diagonal_kernel
            if kernel is None:
                raise RuntimeError("Sparse diagonal GS kernel is unavailable for this solver shape")
            dense_size = self._sparse_diagonal_dense_size
            wp.launch_tiled(
                kernel,
                dim=[self.world_count],
                inputs=[
                    self.constraint_count,
                    self.world_dof_indices,
                    dense_rhs,
                    self.diag,
                    self.impulses,
                    self.row_type,
                    self.row_parent,
                    self.row_mu,
                    self.dense_phase_bounds,
                    self._sparse_contact_group_count,
                    self._sparse_contact_group_heads,
                    self._sparse_contact_serial_count,
                    self._sparse_contact_serial_normals,
                    self._sparse_diagonal_dense_offsets,
                    self._sparse_diagonal_dense_groups,
                    self.J_by_size[dense_size],
                    self.Y_by_size[dense_size],
                    self._sparse_diagonal_row_dof,
                    self._sparse_diagonal_row_jy,
                    self._fused_diagonal_limit_active,
                    self._fused_diagonal_limit_lower_rhs,
                    self._fused_diagonal_limit_upper_rhs,
                    self._diagonal_inverse_mass,
                    self.pgs_cfm,
                    iterations,
                    omega,
                    int(friction_start_iteration),
                    int(iteration_offset),
                ],
                outputs=[
                    self._fused_diagonal_limit_lower_lambda,
                    self._fused_diagonal_limit_upper_lambda,
                    self.v_out,
                    *self._solve_readout.operands(kernel),
                ],
                block_dim=32,
                device=self.model.device,
            )
            return
        mf_gs_kernel = self._pgs_solve_mf_gs_kernel
        if mf_gs_kernel is None:
            raise RuntimeError("Matrix-free GS kernel is unavailable for this solver shape")

        def launch_row_phase(row_phase: int, phase_iterations: int, phase_iteration_offset: int) -> None:
            # Under a readout, count the worlds without rows of any family, which no sweep enters.
            self._solve_readout.count_skipped(self.constraint_count, self.mf_constraint_count)
            launch_local = bool(
                (
                    self._pgs_solve_local_internal_kernels
                    or self._pgs_solve_local_pair_kernels
                    or self._pgs_solve_local_residual_kernels
                )
                and row_phase == 0
            )
            local_stream = self._local_internal_stream if launch_local else None
            pair_stream = self._local_pair_stream if launch_local else None
            residual_stream = self._local_residual_stream if launch_local else None
            if local_stream is not None:
                local_ready_event = wp.get_stream(self.model.device).record_event()
                local_stream.wait_event(local_ready_event)
                if pair_stream is not None:
                    pair_stream.wait_event(local_ready_event)
                if residual_stream is not None:
                    residual_stream.wait_event(local_ready_event)
            use_general_queue = bool(self._local_internal_fast_path)
            general_block_count = self._mf_gs_block_count
            if local_stream is not None:
                # Admit scarce residual and paired worlds before the bulk single-world grid can occupy the GPU.
                if residual_stream is not None:
                    wp.launch(local_solve_launch_gate, dim=1, device=self.model.device)
                    pair_ready_event = wp.get_stream(self.model.device).record_event()
                    if pair_stream is not None:
                        pair_stream.wait_event(pair_ready_event)
                    else:
                        local_stream.wait_event(pair_ready_event)
                if pair_stream is not None:
                    wp.launch(local_solve_launch_gate, dim=1, device=self.model.device)
                    single_ready_event = wp.get_stream(self.model.device).record_event()
                    local_stream.wait_event(single_ready_event)
            factor_kernel = self._paired_factor_solve_kernel
            components = self._paired_components
            if components is not None:
                if row_phase != 0 or soft_relax:
                    raise RuntimeError("Paired components require the physical interleaved solve")
                components.prepare(self, dense_rhs, mf_meta)
            factor_stream = self._paired_general_solve_stream
            if factor_stream is not None:
                if row_phase != 0 or soft_relax:
                    raise RuntimeError("Paired/general overlap requires the physical interleaved solve")
                wp.get_stream(self.model.device).record_event(self._paired_general_ready)
                factor_stream.wait_event(self._paired_general_ready)
            if factor_kernel is not None:
                if row_phase != 0:
                    raise RuntimeError("paired factor-coordinate solve only supports the interleaved row phase")
                primary_size = self._paired_response_primary_size
                secondary_size = self._paired_response_secondary_size
                if (
                    primary_size is None
                    or secondary_size is None
                    or self._paired_factor_primary_groups_by_world is None
                    or self._paired_factor_secondary_groups_by_world is None
                    or self._paired_factor_primary_offsets_by_world is None
                    or self._paired_factor_secondary_offsets_by_world is None
                ):
                    raise RuntimeError("paired factor-coordinate solve metadata is incomplete")

                def launch_factor_solve():
                    with self._sync_timed(f"factor_pgs_iters{phase_iterations}"):
                        wp.launch_tiled(
                            factor_kernel,
                            dim=[(self.world_count + 1) // 2],
                            inputs=[
                                self.world_count,
                                self.constraint_count,
                                self.dense_phase_bounds,
                                self.world_dof_indices,
                                dense_rhs,
                                self.diag,
                                self.impulses,
                                # The factor rows: the packed rows at their world coordinates.
                                self._packed_Z,
                                self.row_type,
                                self.row_parent,
                                self.row_mu,
                                self.mf_constraint_count if components is None else components.selector,
                                self.L_by_size[primary_size],
                                self.L_by_size[secondary_size],
                                self._paired_factor_primary_groups_by_world,
                                self._paired_factor_secondary_groups_by_world,
                                self._paired_factor_primary_offsets_by_world,
                                self._paired_factor_secondary_offsets_by_world,
                                phase_iterations,
                                omega,
                                int(friction_start_iteration),
                                int(phase_iteration_offset),
                            ],
                            outputs=[self.v_out, *self._solve_readout.operands(factor_kernel)],
                            block_dim=64,
                            device=self.model.device,
                            stream=factor_stream,
                        )

                if factor_stream is None:
                    launch_factor_solve()
            with self._sync_timed(f"mfgs_phase{row_phase}_iters{phase_iterations}"):
                mf_gs_inputs = [
                    self._local_general_world_count,
                    self._local_general_worlds,
                    general_block_count,
                    int(use_general_queue),
                    self.constraint_count,
                    self.dense_phase_bounds,
                    self._local_solve_owner if components is None else components.selector,
                    self.world_dof_indices,
                    dense_rhs,
                    self.diag,
                    self.row_w,
                    self.impulses,
                    self.articulation_response_dof_count,
                    self.articulation_world_dof_offset,
                    int(self._packed_dense),
                    self._packed_Z,
                    self._row_sides,
                    self.row_type,
                    self.row_parent,
                    self.row_mu,
                    self._contact_torsion_group,
                    self._contact_torsion_radius,
                    self.drive_target_vel_bias,
                    self.drive_vel_multiplier,
                    self.drive_impulse_multiplier,
                    self.drive_max_impulse,
                    # (1, 1) dummy when the fused clamp is off; the kernel
                    # only reads it when built with fuse_vel_limits.
                    self.drive_vel_limit,
                    self.mf_constraint_count,
                    self.mf_contact_rows_end,
                    mf_meta,
                    self.mf_impulses,
                    self.mf_Z_a,
                    self.mf_Z_b,
                    self._body_factors(),
                    self.mf_row_mu,
                    self.mf_row_w,
                    phase_iterations,
                    omega,
                    int(self._regularization_enabled and not soft_relax),
                    row_phase,
                    int(friction_start_iteration),
                    int(phase_iteration_offset),
                    int(freeze_drive_rows),
                ]
                # The incremental kernel owns worlds with only dense rows inside its row
                # budget; the legacy kernel returns early for those and solves the rest.
                tier_lists = getattr(self, "_tier_worlds", None) is not None
                if tier_lists:
                    self._tier_counts.zero_()
                    wp.launch(
                        classify_parallel_sweep_tiers,
                        dim=self.world_count,
                        inputs=[
                            self.constraint_count,
                            self.mf_constraint_count,
                            self._tier_lo,
                            self._tier_hi,
                            len(self._par_tiers),
                        ],
                        outputs=[self._tier_worlds, self._tier_counts],
                        device=self.model.device,
                    )
                sweeps = list(getattr(self, "_mf_gs_tiers", ()))
                colored_kernel = self._pgs_solve_mf_colored_kernel
                if mf_gs_kernel is not None:
                    sweeps.append(SweepTier(mf_gs_kernel, colored_worlds=colored_kernel is not None))
                for sweep in sweeps:
                    inputs = mf_gs_inputs
                    dim = general_block_count
                    if sweep.tier is not None and tier_lists:
                        # This tier grid-strides over its own compacted world list.
                        inputs = [
                            self._tier_count_views[sweep.tier],
                            self._tier_world_views[sweep.tier],
                            self._tier_blocks,
                            1,
                            *mf_gs_inputs[4:],
                        ]
                        dim = self._tier_blocks
                    if sweep.response_inputs:
                        inputs = inputs + self._inkernel_response_inputs()
                        if sweep.kinetic_rows:
                            inputs = [*inputs, *self._keyed_rows.parallel_arguments()]
                    if sweep.colored_worlds:
                        inputs = [*inputs, self._colored_units.world_colored]
                    wp.launch_tiled(
                        sweep.kernel,
                        dim=[dim],
                        inputs=inputs,
                        outputs=[self.v_out, *self._solve_readout.operands(sweep.kernel)],
                        block_dim=sweep.block_dim,
                        device=self.model.device,
                    )
                if colored_kernel is not None:
                    # The worlds the unit builder's rule picked: a block each, a color's units at once.
                    wp.launch_tiled(
                        colored_kernel,
                        dim=[general_block_count],
                        inputs=[*mf_gs_inputs, *self._colored_units.operands()],
                        outputs=[self.v_out, *self._solve_readout.operands(colored_kernel)],
                        block_dim=sweep_routes.COLORED_THREADS,
                        device=self.model.device,
                    )
            if factor_stream is not None:
                # Queue the scarce general tail first; the independent paired
                # grid shares only the pre-solve ready dependency with it.
                launch_factor_solve()
                factor_stream.record_event(self._paired_general_done)
                wp.get_stream(self.model.device).wait_event(self._paired_general_done)
            local_done_events = []
            if launch_local:
                if local_stream is not None:
                    if residual_stream is not None:
                        with wp.ScopedStream(residual_stream, sync_enter=False):
                            self._launch_local_internal_solve(
                                dense_rhs,
                                phase_iterations,
                                omega,
                                friction_start_iteration,
                                phase_iteration_offset,
                                solve_single=False,
                                solve_pair=False,
                            )
                        local_done_events.append(residual_stream.record_event())
                    if pair_stream is not None:
                        with wp.ScopedStream(pair_stream, sync_enter=False):
                            self._launch_local_internal_solve(
                                dense_rhs,
                                phase_iterations,
                                omega,
                                friction_start_iteration,
                                phase_iteration_offset,
                                solve_single=False,
                                solve_residual=False,
                            )
                        local_done_events.append(pair_stream.record_event())
                    with wp.ScopedStream(local_stream, sync_enter=False):
                        self._launch_local_internal_solve(
                            dense_rhs,
                            phase_iterations,
                            omega,
                            friction_start_iteration,
                            phase_iteration_offset,
                            solve_pair=pair_stream is None,
                            solve_residual=residual_stream is None,
                        )
                    local_done_events.append(local_stream.record_event())
                else:
                    with self._sync_timed(f"mfgs_local_internal_iters{phase_iterations}"):
                        self._launch_local_internal_solve(
                            dense_rhs,
                            phase_iterations,
                            omega,
                            friction_start_iteration,
                            phase_iteration_offset,
                        )
            for local_done_event in local_done_events:
                wp.get_stream(self.model.device).wait_event(local_done_event)

        if self.pgs_schedule == "contact_then_internal":
            row_phases = (1, 2)
            for row_phase in row_phases:
                launch_row_phase(row_phase, iterations, iteration_offset)
        elif self.pgs_schedule == "physx_grasp":
            for schedule_iter in range(iterations):
                phase_iteration_offset = iteration_offset + schedule_iter
                for row_phase in (3, 4, 5):
                    launch_row_phase(row_phase, 1, phase_iteration_offset)
        else:
            launch_row_phase(0, iterations, iteration_offset)

    def _launch_matrix_free_position_solve(
        self,
        *,
        iterations: int,
        friction_start_iteration: int | None = None,
        iteration_offset: int = 0,
    ) -> None:
        if self.contact_compliance:
            _contact_compliance.solve(
                self,
                iterations=iterations,
                friction_start_iteration=friction_start_iteration,
                iteration_offset=iteration_offset,
            )
            return
        self._launch_matrix_free_gs_solve(
            dense_rhs=self.rhs,
            mf_meta=self.mf_meta_packed,
            iterations=iterations,
            omega=self.pgs_omega,
            friction_start_iteration=friction_start_iteration,
            iteration_offset=iteration_offset,
        )

    def _run_matrix_free_velocity_post_solve(self) -> None:
        if self.pgs_velocity_iterations <= 0:
            return

        self._pack_mf_meta(self.mf_rhs_unbiased)
        # The solve readout counts this pass's short sweeps apart from the position solve.
        with self._solve_readout.velocity_pass():
            self._launch_matrix_free_gs_solve(
                soft_relax=True,
                dense_rhs=self.rhs_unbiased,
                mf_meta=self.mf_meta_packed,
                iterations=self.pgs_velocity_iterations,
                omega=self.pgs_omega,
                friction_start_iteration=self._contact_friction_start_iteration(
                    self.pgs_velocity_iterations, velocity_pass=True
                ),
                freeze_drive_rows=self.pgs_velocity_drive_mode == "freeze",
            )

    def _conclude_matrix_free_position_problem(self, dt: float) -> None:
        """Switch the current q0 constraint problem to its velocity-iteration RHS."""
        # PhysX PGS concludes generic 1D/contact rows with zero geometric bias.
        # PhysX PGS keeps articulation drive rows active in velocity
        # iterations, but computeDriveImpulse subtracts the accumulated
        # joint-position delta from the original drive position bias. The
        # velocity-pass descriptor below approximates that delta as
        # ``dt * J * v_position_solve``.
        if self.pgs_velocity_drive_mode == "active":
            self._stage4_compute_physx_drive_desc(
                dt,
                position_bias_scale=1.0,
                position_delta_velocity=self.v_out_snap,
                position_delta_scale=1.0,
            )
        self._stage4_compute_rhs_world(
            dt,
            bias_scale=0.0,
            contact_speculative_scale=0.0,
            preserve_unreached_speculative=True,
            joint_limit_speculative_scale=1.0,
            output=self.rhs_unbiased,
        )
        if self._contact_torsion_enabled:
            prepare_torsion_velocity_pass(self, dt)
        if self._has_free_rigid_bodies:
            self._compute_mf_rhs_bias(
                dt,
                bias_scale=0.0,
                speculative_scale=0.0,
                preserve_unreached_speculative=True,
                apply_restitution=True,
                output=self.mf_rhs_unbiased,
            )

    def _snapshot_matrix_free_position_problem(self, contacts: Contacts | None) -> None:
        """Copy the biased q0 position-solve problem for diagnostics."""
        assert self._debug_buffers_enabled, "debug capture buffers were not allocated"
        self._arrays.snapshot(DENSE_ROWS)
        if self.J_world is not None:
            self._arrays.snapshot(WORLD_RESPONSE)
        if self._has_free_rigid_bodies and hasattr(self, "mf_constraint_count"):
            self._arrays.snapshot(MF_COUNTERS)
            self._arrays.snapshot(MF_ROWS)

        if contacts is None or getattr(contacts, "rigid_contact_count", None) is None:
            return
        wp.copy(self._debug_position_rigid_contact_count, contacts.rigid_contact_count)
        wp.copy(self._debug_position_contact_world, self.contact_world)
        wp.copy(self._debug_position_contact_slot, self.contact_slot)
        wp.copy(self._debug_position_contact_art_a, self.contact_art_a)
        wp.copy(self._debug_position_contact_art_b, self.contact_art_b)
        wp.copy(self._debug_position_contact_path, self.contact_path)
        wp.copy(self._debug_position_rigid_contact_shape0, contacts.rigid_contact_shape0)
        wp.copy(self._debug_position_rigid_contact_shape1, contacts.rigid_contact_shape1)
        wp.copy(self._debug_position_rigid_contact_point0, contacts.rigid_contact_point0)
        wp.copy(self._debug_position_rigid_contact_point1, contacts.rigid_contact_point1)
        wp.copy(self._debug_position_rigid_contact_normal, contacts.rigid_contact_normal)
        wp.copy(self._debug_position_rigid_contact_margin0, contacts.rigid_contact_margin0)
        wp.copy(self._debug_position_rigid_contact_margin1, contacts.rigid_contact_margin1)

    def _on_contact_rows(self, rows) -> None:
        """Assign the per-contact buffers at the contact pool's ready rows, on creation and after each growth."""
        rows.publish(self)
        self._max_contacts_alloc = rows.rows
        patches = getattr(self, "_friction_patches", None)
        if patches is not None:
            patches.view.phi = self._friction_patch_phi

    def _publish_contact_rows(self, rows) -> None:
        rows.publish(self)

    def _follow_contacts(self, contacts: Contacts) -> None:
        """Hold at least ``contacts``' rigid rows, and grow with the pool of a pipeline's buffer from now on."""
        pool = getattr(self, "_contact_pool", None)
        if pool is None:
            return
        rows = contacts._rigid_contact_rows
        try:
            source = getattr(contacts, "_rigid_contact_pool", None)
            if source is not None:
                source.follow(pool)
            if rows > pool.ready:
                pool.grow(rows)
        except RuntimeError as error:
            raise RuntimeError(
                f"FeatherPGS received {rows} contact rows during CUDA graph capture, but its per-contact scratch "
                f"holds {pool.ready}: step once outside capture first."
            ) from error

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
        collide_done_event=None,
    ):
        if self._contact_torsion_enabled:
            validate_torsion_step(self)
            if getattr(self, "_device_torsion", None) is not None:
                self._device_torsion.begin_step(state_in, state_out)
        if self.contact_compliance:
            # Reject incompatible contact preprocessing before it can mutate the stream.
            _contact_compliance.validate_step(self)
        if contacts is not None:
            self._follow_contacts(contacts)
        if self.pgs_warmstart:
            # A reduced stream is valid when it carries retained identities.
            # Preserve the reader lease for an unreduced captured stream: a
            # later producer must not change its layout behind graph replay.
            self._require_unreduced_contacts(
                contacts,
                supports_body_pair_reduced_contacts=bool(
                    contacts is not None and contacts.rigid_contacts_body_pair_reduced
                ),
                configuration="pgs_warmstart=True",
            )
            if contacts is not None and getattr(contacts, "rigid_contact_match_index", None) is None:
                raise NotImplementedError(
                    "FeatherPGS contact warm start requires a Contacts buffer created with "
                    "contact_matching enabled (rigid_contact_match_index is None). Build the "
                    'CollisionPipeline / Contacts with contact_matching="latest" (or "sticky").'
                )
        self._track_step_dt(dt)

        model = self.model

        if control is None:
            control = model.control(clone_variables=False)
        state_aug = self._prepare_augmented_state(state_in, state_out, control)
        self._last_debug_state_aug = state_aug

        if collide_done_event is not None and state_in.particle_count > 0:
            wp.get_stream(self.model.device).wait_event(collide_done_event)
            collide_done_event = None  # consumed

        self._eval_particle_forces(state_in, control, contacts)

        if not model.joint_count:
            self.integrate_particles(model, state_in, state_out, dt)
            self._step += 1
            return state_out

        # Double-buffer: select buffer set and wait for its memset to finish
        if self._memset_stream is not None:
            self.H_by_size = self._H_bufs[self._buf_idx]
            self.J_by_size = self._J_bufs[self._buf_idx]
            if self._jy_world_aliased:
                # J_world aliases the active J buffer; track the ping-pong.
                self.J_world = self.J_by_size[self.size_groups[0]]
            self._wait_memset_done(wp.get_stream(model.device), self._buf_idx)

        # ══════════════════════════════════════════════════════════════
        # STAGE 1: FK/ID + drives + CRBA
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S1_FK_ID_CRBA", print=False, use_nvtx=self._nvtx, synchronize=False):
            drive_rows_ready = self._stage1_prepare_augmented_drives(state_in, state_aug, control, dt)
            stage3_qd = self._stage1_fk_id(state_in, state_aug, state_out)

            if model.articulation_count:
                inverse_dynamics_ready = self._stage1_complete_joint_tau(
                    state_in, state_aug, control, dt, drive_rows_ready
                )
            else:
                inverse_dynamics_ready = None

            self._stage1_crba(state_aug, drive_rows_ready)
        # ══════════════════════════════════════════════════════════════
        # STAGE 2: Cholesky
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S2_Cholesky", print=False, use_nvtx=self._nvtx, synchronize=False):
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    self._stage2_factor(size)
        # ══════════════════════════════════════════════════════════════
        # STAGE 3: Triangular solve + v_hat
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S3_Trisolve_Vhat", print=False, use_nvtx=self._nvtx, synchronize=False):
            if inverse_dynamics_ready is not None:
                wp.get_stream(model.device).wait_event(inverse_dynamics_ready)
            if self._tree_kinetic_state is not None:
                self._tree_kinetic_state.predict(state_in, state_aug, control, stage3_qd, dt)
                self._stage3_correct_free_root_velocity(state_in, stage3_qd, dt, apply_transport=False)
            else:
                self._stage3_zero_qdd(state_aug)
                for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                    with ctx:
                        self._stage3_trisolve(size, state_aug)
                self._stage3_compute_v_hat(state_in, state_aug, dt, stage3_qd)
            if self._debug_stage3_snapshots:
                wp.copy(self._debug_stage3_qd_work, stage3_qd)
                wp.copy(self._debug_stage3_joint_qdd, state_aug.joint_qdd)
                wp.copy(self._debug_stage3_v_hat, self.v_hat)

        # Wait for pipelined collide (if running on separate stream)
        if collide_done_event is not None:
            wp.get_stream(model.device).wait_event(collide_done_event)

        # ══════════════════════════════════════════════════════════════
        # STAGE 4: Build contact problem
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S4_ContactBuild", print=False, use_nvtx=self._nvtx, synchronize=False):
            if self._resolved_worlds_owner is not None:
                self._resolved_worlds_owner.classify(state_in, state_aug, contacts, dt)
            self._stage4_build_rows(state_in, state_aug, control, contacts, dt)
            if self._contact_torsion_enabled:
                prepare_torsion_rows(self, state_in, state_aug, contacts)

        solve = self._plan.solve
        if not solve.assembles_delassus:
            # Compute Y = H^-1 * J^T only (no Delassus C)
            with wp.ScopedTimer("S4_HinvJt_Diag_RHS", print=False, use_nvtx=self._nvtx, synchronize=False):
                for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                    with ctx:
                        self._stage4_response(size)

                # Diagonal from J*Y (no full Delassus), or from the packed rows' |Z|^2
                packed_diag = self._stage4_compute_matrix_free_diag()
                self._stage4_pack_rows(write_diag=packed_diag)
                self._stage4_finalize_world_diag_cfm()
                # Reads J_world only when position_delta_scale != 0; under the
                # J/Y alias (see _detect_jy_world_identity) J_world here holds
                # the CURRENT step's rows rather than last step's gathered
                # copy, so a nonzero position_delta_scale at this call site
                # would change behavior between the two layouts.
                self._stage4_compute_physx_drive_desc(dt)

                # RHS = bias only (J*v recomputed per iteration)
                self._stage4_compute_rhs_world(dt, contact_speculative_scale=self.contact_speculative_scale)
                # NOTE: skip _stage4_accumulate_rhs_world — J*v_hat not baked into rhs

            # MF: whiten the rows (Z = L^-1 J^T), their mf_rhs, mf_eff_mass_inv and incident velocities
            if self._has_free_rigid_bodies:
                with wp.ScopedTimer("S4_MF_Setup", print=False, use_nvtx=self._nvtx, synchronize=False):
                    with self._sync_timed("s4_mf_setup"):
                        self._mf_pgs_setup(state_aug, dt)

                    # MF: compute world-relative DOF offsets for two-phase GS kernel
                    with self._sync_timed("s4_mf_dof_offsets"):
                        wp.launch(
                            response_builds_rows.free_body_coordinates_kernel,
                            dim=self.world_count * self.mf_max_constraints,
                            inputs=[
                                self.mf_constraint_count,
                                self.mf_body_a,
                                self.mf_body_b,
                                self.body_to_articulation,
                                self.articulation_world_dof_offset,
                                self.mf_max_constraints,
                            ],
                            outputs=[self.mf_dof_a, self.mf_dof_b],
                            device=self.model.device,
                        )

        else:
            # Split mode: the response and the Delassus operator C (fused into one tile per articulation, or
            # assembled per size after the response).
            assemble = self._plan.assembles_world_delassus
            if assemble:
                self._stage4_zero_world_C()
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    self._stage4_response(size)
            if assemble:
                for size in self.size_groups:
                    self._stage4_delassus(size)
                self._stage4_finalize_world_diag_cfm()

            self._stage4_compute_rhs_world(dt, contact_speculative_scale=self.contact_speculative_scale)

            for size in self.size_groups:
                self._stage4_accumulate_rhs_world(size)
            self._stage4_apply_world_contact_restitution(dt, matrix_free=False)

        # ══════════════════════════════════════════════════════════════
        # STAGE 5+6: PGS solve
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S5_PGS_Prep", print=False, use_nvtx=self._nvtx, synchronize=False):
            self._stage5_prepare_impulses_world()
            if self.pgs_warmstart and self._ws_prev_dense_impulses is not None and contacts is not None:
                match_index = contacts.rigid_contact_match_index
                pool = self._contact_pool
                pool.launch(
                    gather_dense_warmstart,
                    pool.launch_rows(),
                    inputs=[
                        bind_contact_routes(self, contacts),
                        bind_dense_rows(self),
                        match_index,
                        self._ws_prev_dense_slot_sorted,
                        self._ws_prev_dense_impulses,
                        self._ws_prev_dense_row_type,
                        self._ws_prev_dense_row_parent,
                        self.constraint_count,
                        self._ws_prev_contact_normal,
                        self.pgs_warmstart_decay,
                        dt / self._ws_prev_dt if self._ws_prev_dt > 0.0 else 1.0,
                        self.dense_max_constraints,
                    ],
                    outputs=[self.impulses],
                    extent=contacts.rigid_contact_count,
                    max_blocks=self._contact_capacity_blocks,
                )

        if self._friction_anchors_enabled and self._patch_frame_built(contacts):
            ratio = dt / self._ws_prev_dt if self._ws_prev_dt > 0.0 else 1.0
            seeded = [
                (route, parents, mu, impulses, decay * ratio)
                for route, parents, mu, impulses, decay in self._patch_row_arrays()
                if decay > 0.0
            ]
            if seeded:
                self._friction_patches.seed(
                    state_in,
                    self.contact_world,
                    self.contact_slot,
                    self.contact_path,
                    self.contact_slots_needed,
                    patch_routes(seeded),
                )

        if solve.family == "sparse_split":
            # The sparse producers already published bias, incident velocity,
            # CFM and restitution. Decode the accumulated factor delta once;
            # the dense RHS accumulation and Y*lambda publication do not apply.
            self._stage6_prepare_world_velocity()
            self._launch_matrix_free_gs_solve(
                dense_rhs=self.rhs,
                mf_meta=self.mf_meta_packed,
                iterations=self.pgs_iterations,
                omega=self.pgs_omega,
            )
        elif solve.family == "matrix_free":
            with wp.ScopedTimer("S5_GatherJY", print=False, use_nvtx=self._nvtx, synchronize=False):
                self._stage4_apply_world_contact_restitution(dt, matrix_free=True)
                if self._preelim_active:
                    # Bilateral pre-elimination: fold mimic/connect coupling into the packed rows, their
                    # right-hand sides and diagonals (the rows' closed-loop inertia).
                    self._stage4_bilateral_preelim()

                # Initialize v_out = v_hat before GS loop
                self._stage6_prepare_world_velocity()
                if self.pgs_warmstart and self._sparse_mass_matrix_size is None:
                    # Matrix-free GS updates velocity from impulse deltas, so
                    # cached dense lambdas must be represented in v_out before
                    # the first delta is computed: v += P^T L^-T Z lambda from the
                    # packed rows. Native sparse factors seed their sweep's factor
                    # delta from the carried impulses instead.
                    self._stage6_apply_packed_impulses()
                self._apply_mf_warmstart_velocity(self.v_out, set_output_to_delta=False)
                if self._preelim_active:
                    # One-shot bilateral projection: J_B v_out = -b_B, preserved by
                    # every sweep impulse through the corrected columns. Keep
                    # this after every warm-start velocity install so a carried
                    # contact cannot reopen the projected closure.
                    self._stage6_project_bilateral_velocity()

                # Pack MF metadata into int4 structs for coalesced 128-bit loads
                self._pack_mf_meta(self.mf_rhs)

            with wp.ScopedTimer("S6_PGS_Solve", print=False, use_nvtx=self._nvtx, synchronize=False):
                if self.pgs_debug:
                    self._pgs_convergence_log.append([])
                    self._pgs_ncp_residual_log.append([])
                    friction_start_iteration = self._contact_friction_start_iteration(self.pgs_iterations)
                    for _pgs_dbg_iter in range(self.pgs_iterations):
                        # Snapshot impulses before this iteration
                        wp.copy(self._diag_prev_impulses, self.impulses)
                        if self._diag_prev_mf_impulses is not None:
                            wp.copy(self._diag_prev_mf_impulses, self.mf_impulses)

                        # Run 1 diagnostic iteration. For the experimental split
                        # schedule this is one contact phase followed by one
                        # internal phase per diagnostic sample.
                        self._launch_matrix_free_position_solve(
                            iterations=1,
                            friction_start_iteration=friction_start_iteration,
                            iteration_offset=_pgs_dbg_iter,
                        )

                        # Diagnostic kernel
                        wp.launch(
                            pgs_convergence_diagnostic_velocity,
                            dim=self.world_count,
                            inputs=[
                                bind_dense_rows(self),
                                bind_mf_rows(self),
                                self.constraint_count,
                                self.world_dof_indices,
                                self.impulses,
                                self._diag_prev_impulses,
                                self._row_support(),
                                self._packed_J,
                                self.dense_max_constraints,
                                self.mf_constraint_count,
                                self.mf_impulses,
                                self._diag_prev_mf_impulses,
                                self.mf_max_constraints,
                                self._body_factors(),
                                self.v_out,
                            ],
                            outputs=[self._diag_metrics],
                            device=self.model.device,
                        )

                        # Sync and reduce across worlds
                        metrics_np = self._diag_metrics.numpy()
                        row = np.array(
                            [
                                np.max(metrics_np[:, 0]),  # max|delta_lambda|
                                np.sum(metrics_np[:, 1]),  # complementarity gap
                                np.sum(metrics_np[:, 2]),  # tangent residual
                                np.sum(metrics_np[:, 3]),  # FB merit
                            ]
                        )
                        self._pgs_convergence_log[-1].append(row)

                        # NCP / MDP residual diagnostic ([world, 6]):
                        # (r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir)
                        wp.launch(
                            pgs_ncp_residuals_diagnostic_velocity,
                            dim=self.world_count,
                            inputs=[
                                bind_dense_rows(self),
                                bind_mf_rows(self),
                                self.constraint_count,
                                self.world_dof_indices,
                                self.impulses,
                                self._row_support(),
                                self._packed_J,
                                self.dense_max_constraints,
                                self.mf_constraint_count,
                                self.mf_impulses,
                                self.mf_max_constraints,
                                self._body_factors(),
                                self.v_out,
                            ],
                            outputs=[self._diag_ncp_metrics],
                            device=self.model.device,
                        )
                        ncp_np = self._diag_ncp_metrics.numpy().copy()
                        self._pgs_ncp_residual_log[-1].append(ncp_np)

                    self._pgs_convergence_log[-1] = np.array(self._pgs_convergence_log[-1])
                    # Stack per-iter [world, 6] arrays into [iters, world, 6].
                    self._pgs_ncp_residual_log[-1] = np.stack(self._pgs_ncp_residual_log[-1], axis=0)

                else:
                    self._launch_matrix_free_position_solve(
                        iterations=self.pgs_iterations,
                        friction_start_iteration=self._contact_friction_start_iteration(self.pgs_iterations),
                    )

            if self.pgs_velocity_iterations > 0:
                if self.pgs_debug:
                    wp.copy(self._debug_position_v_out, self.v_out)
                    self._snapshot_matrix_free_position_problem(contacts)
                wp.copy(self.v_out_snap, self.v_out)
                self._conclude_matrix_free_position_problem(dt)
                wp.copy(self.v_hat, self.v_out_snap)
                self._run_matrix_free_velocity_post_solve()
                self._stage6_integrate_position_from_velocity(state_in, state_aug, state_out, dt, self.v_out_snap)
        elif solve.family == "split_interleaved":
            # Split mode with mixed contacts: interleaved dense and MF, 1 iteration each.
            self._mf_pgs_setup(state_aug, dt)
            self._build_mf_body_map()
            self.v_mf_accum.zero_()
            self._apply_mf_warmstart_velocity(
                self.v_mf_accum,
                set_output_to_delta=True,
                body_map_ready=True,
            )
            if self._mf_warmstart_enabled:
                # Split stores dense constraints in impulse space, so its RHS
                # must include the velocity represented by the cached MF
                # lambdas before the first interleaved dense iteration.
                for size in self.size_groups:
                    self._stage4_accumulate_rhs_world(size, velocity=self.v_mf_accum)

            for _pgs_iter in range(self.pgs_iterations):
                # Dense PGS (1 iteration, impulse space)
                self._dispatch_dense_pgs_solve(
                    iterations=1,
                    friction_start_iteration=self._contact_friction_start_iteration(self.pgs_iterations),
                    iteration_offset=_pgs_iter,
                )

                # Rebuild v_out = v_hat + Y*impulses + MF_corrections
                self._stage6_prepare_world_velocity()
                for size in self.size_groups:
                    self._stage6_apply_impulses_world(size)
                wp.launch(
                    vector_add_inplace,
                    dim=self.v_out.size,
                    inputs=[self.v_out, self.v_mf_accum],
                    device=self.model.device,
                )

                # Snapshot v_out, run MF, compute delta
                wp.copy(self.v_out_snap, self.v_out)
                self._mf_pgs_solve(
                    iterations=1,
                    friction_start_iteration=self._contact_friction_start_iteration(self.pgs_iterations),
                    iteration_offset=_pgs_iter,
                    body_map_ready=True,
                )

                # v_mf_accum += (v_out - v_out_snap); v_out_snap = delta
                wp.launch(
                    compute_delta_and_accumulate,
                    dim=self.v_out.size,
                    inputs=[self.v_out, self.v_out_snap, self.v_mf_accum],
                    device=self.model.device,
                )

                # Update dense rhs: world_rhs += J * delta_v_mf
                for size in self.size_groups:
                    self._stage4_accumulate_rhs_world(size, velocity=self.v_out_snap)

            # v_out is already final (includes both dense and MF corrections)

        else:
            # Split without mixed contacts (dense_split): dense PGS, then optional MF
            if self.pgs_debug:
                self._pgs_convergence_log.append([])
                friction_start_iteration = self._contact_friction_start_iteration(self.pgs_iterations)
                for _pgs_dbg_iter in range(self.pgs_iterations):
                    prev_np = self.impulses.numpy().copy()
                    self._dispatch_dense_pgs_solve(
                        iterations=1,
                        friction_start_iteration=friction_start_iteration,
                        iteration_offset=_pgs_dbg_iter,
                    )
                    cur_np = self.impulses.numpy()
                    max_delta = float(np.max(np.abs(cur_np - prev_np)))
                    self._pgs_convergence_log[-1].append(np.array([max_delta, 0.0, 0.0, 0.0]))
                self._pgs_convergence_log[-1] = np.array(self._pgs_convergence_log[-1])
            else:
                self._dispatch_dense_pgs_solve(
                    iterations=self.pgs_iterations,
                    friction_start_iteration=self._contact_friction_start_iteration(self.pgs_iterations),
                )

            self._stage6_prepare_world_velocity()
            for size in self.size_groups:
                self._stage6_apply_impulses_world(size)

            if self.pgs_mode == "split" and self._has_free_rigid_bodies:
                self._stage6b_mf_pgs(state_aug, dt)

        self._stage6_clamp_rigid_velocity_limits()

        # ══════════════════════════════════════════════════════════════
        # STAGE 7: Update qdd + integrate
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S7_Integrate", print=False, use_nvtx=self._nvtx, synchronize=False):
            if self.pgs_mode == "matrix_free" and self.pgs_velocity_iterations > 0:
                self._stage6_write_final_velocity(state_in, state_aug, state_out, dt)
            else:
                self._stage6_update_qdd(state_in, state_aug, dt)
                self._stage6_integrate(state_in, state_aug, state_out, dt)

        # ── MF warm-start carry: snapshot this step's converged impulses +
        # row-type table + per-(sorted)-contact slot map so step N+1 can seed
        # from them by contact identity. Only runs when the feature is on, so
        # the default-off launch sequence is unchanged.
        # Dense identity-matched carry: snapshot this step's converged dense
        # impulses + row types + per-(sorted)-contact dense-slot map (mirror of
        # the MF carry below).
        if self._friction_anchors_enabled:
            with wp.ScopedTimer("S7_Friction_Patch_Carry", print=False, use_nvtx=self._nvtx, synchronize=False):
                self._finish_friction_patches(state_in, state_out, contacts, dt)

        if self.pgs_warmstart and self._ws_prev_dense_impulses is not None:
            with wp.ScopedTimer("S7_Dense_Warmstart_Carry", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                wp.copy(self._ws_prev_dense_impulses, self.impulses)
                wp.copy(self._ws_prev_dense_row_type, self.row_type)
                wp.copy(self._ws_prev_dense_row_parent, self.row_parent)
                if contacts is None:
                    self._ws_prev_dense_row_type.fill_(-1)
                    self._ws_prev_dense_row_parent.fill_(-1)

        if self._mf_warmstart_enabled and self._has_free_rigid_bodies:
            with wp.ScopedTimer("S7_MF_Warmstart_Carry", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                wp.copy(self._ws_prev_mf_impulses, self.mf_impulses)
                wp.copy(self._ws_prev_mf_row_type, self.mf_row_type)
                wp.copy(self._ws_prev_mf_row_parent, self.mf_row_parent)
                if contacts is None:
                    self._ws_prev_mf_row_type.fill_(-1)
                    self._ws_prev_mf_row_parent.fill_(-1)

        if self.pgs_warmstart:
            if contacts is not None:
                # One thread per row the contact stream ever used: the slot tables of every such row are rewritten.
                pool = self._contact_pool
                used = pool.rows_used if contacts._rigid_contact_pool is pool else None
                pool.launch(
                    snapshot_contact_warmstart,
                    pool.launch_rows(),
                    inputs=[bind_contact_routes(self, contacts)],
                    outputs=[
                        self._ws_prev_dense_slot_sorted,
                        self._ws_prev_slot_sorted,
                        self._ws_prev_contact_normal,
                    ],
                    extent=pool.ready_rows if used is None else used,
                    max_blocks=self._contact_capacity_blocks,
                )
            self._ws_prev_dt = float(dt)

        # Double-buffer: fork the maintenance stream to clear the current
        # buffer for reuse. The compact path snapshots this solve's row counts
        # on the main stream, then clears only those active J prefixes; every
        # inactive row is already clean by induction because each previously
        # active prefix was cleared immediately after use. H keeps its existing
        # conditional full-buffer clear.
        # ScopedStream(sync_enter=True) records an event on the main stream and
        # makes the memset stream wait — this is what forks it into graph capture.
        # J must be zeroed every step (constraint rows are rebuilt per step),
        # but H is only zeroed on steps whose host-evaluated global mass-update
        # flag was 1 (see _stage1_crba): with update_mass_matrix_interval=N and
        # graphs capturing a multiple of N steps, each buffer ping-pongs with
        # period 2 and is consumed by steps of a fixed flag parity, so the
        # baked memset cadence matches the baked rebuild cadence. Skipping the
        # H memset is safe in every configuration because crba_fill
        # stores all structural nonzeros of a masked articulation before
        # apply_augmented_mass_diagonal_grouped adds onto the freshly stored
        # diagonal, structural zeros are never written after allocation, and
        # the Cholesky kernels never read H when mass_update_mask is 0.
        # Sizes a held-tree owner factors from its own geometry have no H
        # reader at all (StepPlan.unread_mass_sizes) and are never cleared.
        if self._memset_stream is not None:
            with wp.ScopedTimer("DB_Memset", print=False, use_nvtx=self._nvtx, synchronize=False):
                clear = self._plan.jacobian_clear_sizes
                if clear:
                    wp.copy(self._j_active_counts[self._buf_idx], self.constraint_count)
                with wp.ScopedStream(self._memset_stream):
                    for size in self.size_groups:
                        if self._mass_update_global_flag and size not in self._plan.unread_mass_sizes:
                            self._H_bufs[self._buf_idx][size].zero_()
                        if size not in clear:
                            continue
                        wp.launch(
                            clear_grouped_jacobian_active_rows,
                            dim=self.n_arts_by_size[size] * 32,
                            inputs=[
                                self.group_to_art[size],
                                self.art_to_world,
                                self._j_active_counts[self._buf_idx],
                                size,
                                self.dense_max_constraints,
                            ],
                            outputs=[self._J_bufs[self._buf_idx][size]],
                            block_dim=256,
                            device=self.model.device,
                        )
                self._record_memset_done(self._buf_idx, self._memset_stream.record_event())
                self._buf_idx = 1 - self._buf_idx

        # Opt-in row high-water telemetry (behavior-neutral). Launched here, at
        # the end of step() after _stage4_build_rows has finalized the per-world
        # counts, so the atomic-max runs inside the captured graph every step and
        # catches mid-run peaks. Read-only on the counts; writes only the
        # separate watermark buffers. No-op when the flag is off (nothing
        # allocated, nothing launched), keeping the captured graph identical.
        if self._row_watermark:
            wp.launch(
                _accumulate_row_watermark,
                dim=self.constraint_count.shape[0],
                inputs=[self.constraint_count, self._row_watermark_dense],
                device=self.constraint_count.device,
            )
            wp.launch(
                _accumulate_row_capacity_telemetry,
                dim=self.slot_counter.shape[0],
                inputs=[
                    self.slot_counter,
                    self._row_dropped_dense,
                    self.dense_max_constraints,
                    self._row_watermark_dense_raw,
                    self._row_dropped_dense_high_water,
                    self._row_overflow_dense_excess,
                    self._row_overflow_dense_world_steps,
                ],
                device=self.slot_counter.device,
            )
            mf_counts = getattr(self, "mf_constraint_count", None)
            if mf_counts is not None and mf_counts.shape[0] > 0:
                wp.launch(
                    _accumulate_row_watermark,
                    dim=mf_counts.shape[0],
                    inputs=[mf_counts, self._row_watermark_mf],
                    device=mf_counts.device,
                )
                wp.launch(
                    _accumulate_row_capacity_telemetry,
                    dim=self.mf_slot_counter.shape[0],
                    inputs=[
                        self.mf_slot_counter,
                        self._row_dropped_mf,
                        self.mf_max_constraints,
                        self._row_watermark_mf_raw,
                        self._row_dropped_mf_high_water,
                        self._row_overflow_mf_excess,
                        self._row_overflow_mf_world_steps,
                    ],
                    device=self.mf_slot_counter.device,
                )
            if contacts is not None and getattr(contacts, "rigid_contact_count", None) is not None:
                contact_counts = contacts.rigid_contact_count
                wp.launch(
                    _accumulate_row_watermark,
                    dim=contact_counts.shape[0],
                    inputs=[contact_counts, self._row_watermark_contact],
                    device=contact_counts.device,
                )

        if self._contact_torsion_enabled and getattr(self, "_device_torsion", None) is not None:
            self._device_torsion.end_step(state_out)
            if self._device_torsion.deferred_errors and not wp.get_stream(self.model.device).is_capturing:
                self.validate_contact_torsion()
        self._step += 1
        return state_out

    def constraint_capacity_status(self, *, clear: bool = False) -> dict[str, bool]:
        """Return sticky capacity-overflow flags without enabling row telemetry.

        The keys ``dense`` and ``matrix_free`` identify row buffers whose
        capacity omitted constraints; ``propagation`` is deprecated and always
        ``False``: no route builds propagation rows. ``contacts`` remains for
        compatibility and is always ``False`` too: a contact stream's count
        never exceeds its rows (a collision pass clamps it and flags the drop in
        its own capacity status).
        ``patch_frames`` identifies a captured step of patch friction whose
        ``contact_generation`` or contact count changed since the previous
        captured step of the same graph without a write Newton noted (such as
        a collision pass or :meth:`~newton.Contacts.clear`), and which was
        solved without patch friction rows (see ``friction_anchor_beta``); only
        patch frame reuse sets it. ``contact_launches`` identifies a contact-sized
        launch of a captured graph, the solver's or the collision pipeline's
        sharing its contact rows, that the graph's last replay skipped: its
        device graph updater failed closed after the capture emitted it, or its
        count passed the captured bound. Flags remain set across later successful
        steps and :meth:`reset` calls; ``contact_launches`` until a replay runs
        the launch again.

        Call outside CUDA graph capture, on the solver's execution stream or
        after explicitly joining that stream. This copies two integers to the
        host, and a third with patch frame reuse. Graph replays update the same
        device flags without host reads.

        Args:
            clear: Clear the flags after reading, returning their previous
                values. Clearing acknowledges past loss; it does not repair
                affected states or resize storage. Reconstruct the solver and
                recapture graphs to change capacities safely.

        Raises:
            RuntimeError: If called during CUDA graph capture.
        """
        if wp.get_device(self.model.device).is_capturing:
            raise RuntimeError("Constraint capacity status must be read outside CUDA graph capture")
        dense, matrix_free = map(bool, self._constraint_capacity_status.numpy())
        result = {"dense": dense, "matrix_free": matrix_free, "propagation": False, "contacts": False}
        frame_status = self._patch_frame_status()
        result["patch_frames"] = frame_status is not None and bool(frame_status.numpy()[0])
        pool = getattr(self, "_contact_pool", None)
        result["contact_launches"] = pool is not None and pool.launch_failed(clear=clear)
        if clear:
            self._constraint_capacity_status.zero_()
            if frame_status is not None:
                frame_status.zero_()
        return result

    def check_constraint_capacity(self) -> None:
        """Raise for worlds invalidated by contact or constraint capacity exhaustion.

        Call at a host observation boundary, outside CUDA graph capture. The
        device-resident :attr:`constraint_overflow` boolean array is always
        maintained, including when ``row_watermark=False``; GPU consumers can
        inspect it without host synchronization. Status remains set until
        :meth:`reset` clears the affected worlds. Increasing capacity requires
        constructing a solver and recapturing graphs before restarting the
        invalidated transition. Also raises while the ``patch_frames`` flag of
        :meth:`constraint_capacity_status` is set, until that method clears it.
        """
        if self.model.device.is_cuda and self.model.device.is_capturing:
            raise RuntimeError("check_constraint_capacity() must run outside CUDA graph capture")
        # Skipped contact launches come first: they leave the rows the other checks read unbuilt.
        pool = getattr(self, "_contact_pool", None)
        if pool is not None and pool.launch_failed():
            raise RuntimeError(
                f"Captured contact launches did not run in their graph's last replay ({pool.launch_status()}): a "
                "device graph updater failed closed after its capture emitted a launch, or a count passed what a "
                "launch was captured for. Discard the graph's results and recapture."
            )
        worlds = np.flatnonzero(self.constraint_overflow.numpy())
        if worlds.size:
            raise RuntimeError(
                f"FeatherPGS constraint/contact capacity exceeded in worlds {worlds[:16].tolist()}"
                f" ({worlds.size} invalid worlds); increase capacities and reset before accepting transitions."
            )
        frame_status = self._patch_frame_status()
        if frame_status is not None and frame_status.numpy()[0]:
            raise RuntimeError(
                "FeatherPGS solved captured steps without patch friction: their contact generation or count changed "
                "since the previous captured step of the same graph without a collision pass or Contacts.clear(). "
                "Call Contacts.clear() before rewriting contacts between captured steps, or set "
                "FEATHER_PGS_PATCH_FRAME_REUSE=0; constraint_capacity_status(clear=True) clears the flag."
            )

    def _patch_frame_status(self):
        """Return the device flag of patch frames left empty by a stale reuse, or ``None`` when none can be.

        Only a build that keeps its pairs through frame reuse sets the flag, so
        without frame reuse the host reads it never. A model without
        articulations allocates no patch state at all.
        """
        patches = getattr(self, "_friction_patches", None)
        if patches is None or not patches.reuse_frames:
            return None
        return patches.frame_status

    def constraint_row_watermarks(self) -> dict:
        """Return the opt-in constraint/contact row high-water marks.

        These are whole-run running maxima (warmup included) of retained and
        raw per-world dense / matrix-free / rigid-contact row counts, rejected
        contact rows that allocators rolled back, peak over-capacity demand,
        and the number of overflowing world-steps. They are accumulated inside
        the captured :meth:`step` when
        ``row_watermark`` is enabled. Reads the telemetry words back in one
        copy and so must be called OUTSIDE any captured/timed region (it
        forces a device sync).

        Returns ``0`` for every field when the telemetry was not enabled; these
        zeros do not establish absence of overflow. Use
        :meth:`check_constraint_capacity` for always-available overflow visibility.
        The ``propagation_*`` fields are deprecated and always ``0``: no route
        builds propagation rows.
        """
        words = {}
        if self._row_watermark_words is not None:
            words = dict(
                zip((field.name for field in ROW_WATERMARKS), self._row_watermark_words.numpy().tolist(), strict=True)
            )
        fields = {
            "high_water": "_row_watermark_{}",
            "raw_high_water": "_row_watermark_{}_raw",
            "dropped_contact_rows_high_water": "_row_dropped_{}_high_water",
            "overflow_excess_high_water": "_row_overflow_{}_excess",
            "overflow_world_steps": "_row_overflow_{}_world_steps",
        }
        result = {
            f"{family}_{field}": int(words.get(name.format(family), 0))
            for family in ("dense", "mf", "propagation")
            for field, name in fields.items()
        }
        result["contact_high_water"] = int(words.get("_row_watermark_contact", 0))
        return result

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        """Populate linear contact forces from the last FeatherPGS solve.

        This path reports linear force only: the torque components of
        ``contacts.force`` are zero placeholders, not a solved wrench about
        body0's center of mass. With persistent patches, normal and friction
        rows can act at different points, so crossing the current contact point
        with the combined force does not recover the solved torque. This export
        is not suitable for contact-wrench sensing.
        """
        if contacts is None or contacts.rigid_contact_count is None:
            return

        dt = self._last_step_dt
        inv_dt = 0.0 if dt is None or dt <= 0.0 else 1.0 / dt
        enable_friction_flag = 1 if self.enable_contact_friction else 0
        mf_impulses = getattr(self, "mf_impulses", None)
        if mf_impulses is None:
            mf_impulses = self._dummy_contact_impulses
        mf_constraint_count = getattr(self, "mf_constraint_count", None)
        if mf_constraint_count is None:
            mf_constraint_count = self._dummy_contact_count
        mf_row_type = getattr(self, "mf_row_type", None)
        if mf_row_type is None:
            mf_row_type = self._dummy_contact_row_type
        mf_row_parent = getattr(self, "mf_row_parent", None)
        if mf_row_parent is None:
            mf_row_parent = self._dummy_contact_row_parent

        pool = self._contact_pool
        rows = pool.launch_rows()
        pool.launch(
            compute_contact_linear_force_from_impulses,
            rows,
            inputs=[
                bind_contact_routes(self, contacts),
                self.impulses,
                mf_impulses,
                self.constraint_count,
                mf_constraint_count,
                self.row_type,
                self.row_parent,
                mf_row_type,
                mf_row_parent,
                enable_friction_flag,
                inv_dt,
            ],
            outputs=[contacts.rigid_contact_force],
            extent=contacts.rigid_contact_count,
            max_blocks=self._contact_capacity_blocks,
        )

        if contacts.force is not None:
            pool.launch(
                pack_contact_linear_force_as_spatial,
                rows,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_force,
                ],
                outputs=[contacts.force],
                extent=contacts.rigid_contact_count,
                max_blocks=self._contact_capacity_blocks,
            )

    def _prepare_augmented_state(
        self,
        state_in: State,
        state_out: State,
        control: Control,
    ) -> State:
        requires_grad = state_in.requires_grad
        state_aug = state_out if requires_grad else self
        model = self.model

        if not getattr(state_aug, "_featherstone_augmented", False):
            self._allocate_state_aux_vars(model, state_aug, requires_grad)

        return state_aug

    def _allocate_state_aux_vars(self, model, target, requires_grad):
        """Allocate the state-derived dynamics (maximal coordinates) on ``target``."""
        if not model.body_count:
            return
        arrays = SolverArrays(target, model.device, requires_grad=requires_grad)
        arrays.allocate(STATE_AUX, dof=model.joint_dof_count, body=model.body_count)
        if requires_grad:
            # Scratch of trisolve_loop's custom gradient.
            arrays.allocate(STATE_GRAD_SCRATCH, dof=model.joint_dof_count)
        else:
            target.joint_solve_tmp = None
        target._featherstone_augmented = True

    def _eval_particle_forces(self, state_in: State, control: Control, contacts: Contacts):
        model = self.model

        particle_f = state_in.particle_f if state_in.particle_count else None
        body_f = state_in.body_f if state_in.body_count else None

        # damped springs
        eval_spring_forces(model, state_in, particle_f)

        # triangle elastic and lift/drag forces
        eval_triangle_forces(model, state_in, control, particle_f)

        # triangle bending
        eval_bending_forces(model, state_in, particle_f)

        # tetrahedral FEM
        eval_tetrahedra_forces(model, state_in, control, particle_f)

        # particle-particle interactions
        eval_particle_contact_forces(model, state_in, particle_f)

        # particle shape contact
        eval_particle_body_contact_forces(model, state_in, contacts, particle_f, body_f, body_f_in_world_frame=True)

    @contextmanager
    def _parallel_size_region(self, enabled: bool = True):
        """Context for parallel dispatch across size groups."""
        if not enabled or not self.use_parallel_streams or not self.model.device.is_cuda or len(self.size_groups) <= 1:
            yield
            return

        main_stream = wp.get_stream(self.model.device)
        self._init_event = main_stream.record_event()
        try:
            yield
        finally:
            for size in self.size_groups:
                stream = self._size_streams.get(size)
                if stream is not None:
                    main_stream.wait_event(stream.record_event())
            self._init_event = None

    @contextmanager
    def _on_size_stream(self, size: int):
        """Execute block on this size's CUDA stream."""
        stream = self._size_streams.get(size)
        init_event = getattr(self, "_init_event", None)
        if stream is not None and init_event is not None:
            stream.wait_event(init_event)
            with wp.ScopedStream(stream):
                yield
        else:
            yield

    @contextmanager
    def _size_dispatch(self, enabled: bool):
        with self._parallel_size_region(enabled=enabled):
            yield

    @contextmanager
    def _size_ctx(self, size: int):
        with self._on_size_stream(size):
            yield

    def _for_sizes(self, enabled: bool):
        # convenience generator; keeps step code tight
        with self._size_dispatch(enabled):
            for size in self.size_groups:
                yield size, self._size_ctx(size)

    def _dynamics(
        self, state: State | None, state_aug: State, control: Control | None = None, **values
    ) -> DynamicsData:
        """Bind the dynamics passes' struct (:data:`.tree_passes.DynamicsData`).

        ``state`` holds the coordinates the passes read and receives the poses and COM twists they publish
        (``None`` for passes that read neither); ``state_aug`` the augmented state. ``values`` are the launch's
        switches (zero unless given) and field overrides, such as the prescaled velocities of ``joint_qd``.
        """
        arrays = self._arrays
        values.setdefault("write_inertia_records", 0)
        if state is None:
            for name in ("joint_q", "joint_qd"):
                values.setdefault(name, arrays.placeholder(wp.float32))
            values.setdefault("body_q", arrays.placeholder(wp.transform))
            values.setdefault("body_qd", arrays.placeholder(wp.spatial_vector))
            values.setdefault("body_f", arrays.placeholder(wp.spatial_vector))
        elif "body_f" not in values:
            values["body_f"] = state.body_f if state.body_count else arrays.placeholder(wp.spatial_vector)
        if control is None:
            values.setdefault("joint_f", arrays.placeholder(wp.float32))
        return field_group.bind(
            DynamicsData, model=self.model, solver=self, state=state, aug=state_aug, control=control, **values
        )

    def _launch_velocity_prescale(self, joint_qd: wp.array, prescaled_joint_qd: wp.array) -> None:
        """Write every DOF's PhysX-prescaled joint velocity in one launch, without a separate copy."""
        model = self.model
        if not model.articulation_count:
            wp.copy(prescaled_joint_qd, joint_qd)
            return
        wp.launch(
            prescale_joint_velocity_limits,
            dim=self._velocity_prescale_threads,
            inputs=[
                bind_joint_table(self),
                model.articulation_count,
                self._velocity_prescale_lanes,
                self._vel_limit_drive_slot_arg,
                self._vel_limit_skip_driven,
                joint_qd,
            ],
            outputs=[prescaled_joint_qd],
            device=model.device,
        )

    def _stage1_fk_id(self, state_in: State, state_aug: State, state_out: State) -> wp.array:
        """Forward kinematics and inverse-dynamics terms; returns the predictor's velocity (prescaled when limited)."""
        model = self.model

        # evaluate joint inertias, motion vectors, and forces
        if self._fk_id_cache_enabled:
            if not getattr(getattr(self, "_tree_kinetic_state", None), "bank_compare", False):
                # Owners that compare input banks validate a changed bank on the device.
                if self._fk_id_cache_source_state is not None and state_in is not self._fk_id_cache_source_state:
                    self._fk_id_cache_valid.zero_()
                    self._forget_repair_continuity()
            self._fk_id_cache_source_state = state_in
        else:
            state_aug.body_f_s.zero_()
        if self.enable_joint_velocity_limits:
            self._launch_velocity_prescale(state_in.joint_qd, self.qd_work)
            stage3_qd = self.qd_work
        else:
            stage3_qd = state_in.joint_qd

        refresh_composite = (self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update
        if getattr(self, "_tree_kinetic_state", None) is not None:
            self._tree_kinetic_state.begin(
                state_in, state_aug, stage3_qd, self._last_step_dt, refresh_composite, state_out=state_out
            )
            return stage3_qd
        # A refresh step places every body's inertia record for the composite inertia.
        dynamics = self._dynamics(state_in, state_aug, joint_qd=stage3_qd, write_inertia_records=int(refresh_composite))
        wp.launch(
            fk_id, dim=model.articulation_count, inputs=[dynamics], block_dim=SERIAL_BLOCK_DIM, device=model.device
        )
        if model.body_count:
            wp.launch(
                update_body_qd_from_featherstone,
                dim=model.body_count,
                inputs=[
                    state_aug.body_v_s,
                    state_in.body_q,
                    model.body_com,
                    self.body_to_articulation,
                    self.articulation_origin,
                ],
                outputs=[state_out.body_qd],
                device=model.device,
            )
        return stage3_qd

    def _stage1_prepare_augmented_drives(
        self, state_in: State, state_aug: State, control: Control, dt: float
    ) -> wp.Event | None:
        """Launch the per-coordinate augmented-drive preparation independently of FK/ID."""
        if not self._parallel_augmented_drive_topology:
            return None

        model = self.model
        dynamics_stream = self._articulation_dynamics_stream
        if dynamics_stream is not None:
            dynamics_stream.wait_event(wp.get_stream(model.device).record_event())
        with wp.ScopedStream(dynamics_stream or wp.get_stream(model.device), sync_enter=False):
            scalar = getattr(self._tree_kinetic_state, "scalar", None)
            if scalar is not None:
                # Scalar branches evaluate their drives in their predict.
                scalar.launch_other_drives(state_in, state_aug, control, dt)
            else:
                wp.launch(
                    _prepare_augmented_joint_drives_by_dof,
                    dim=model.joint_dof_count,
                    inputs=[
                        self._augmented_drive_row_by_dof,
                        self._augmented_drive_q_index_by_dof,
                        self._augmented_drive_target_index_by_dof,
                        state_in.joint_q,
                        state_in.joint_qd,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        control.joint_target_q,
                        control.joint_target_qd,
                        model.joint_effort_limit,
                        dt,
                    ],
                    outputs=[self.aug_row_K, state_aug.joint_tau],
                    block_dim=256,
                    device=model.device,
                )
        return None if dynamics_stream is None else dynamics_stream.record_event()

    def _launch_rigid_tau(self, state_in: State, state_aug: State, control: Control) -> None:
        """Launch inverse dynamics for the non-actuator torque bucket."""
        model = self.model
        wp.launch(
            inverse_dynamics_tau,
            dim=model.articulation_count,
            inputs=[self._dynamics(state_in, state_aug, control)],
            block_dim=SERIAL_BLOCK_DIM,
            device=model.device,
        )

    def _stage1_complete_joint_tau(
        self,
        state_in: State,
        state_aug: State,
        control: Control,
        dt: float,
        drive_rows_ready: wp.Event | None,
    ) -> wp.Event | None:
        """Complete ``joint_tau``: the CPU's one pass of joint forces and drives, or the held tree's drives."""
        if getattr(self, "_tree_kinetic_state", None) is not None:
            # joint_tau currently holds only u0. Current external/control/bias
            # contributions are added by the retained-W predictor after waiting.
            return drive_rows_ready
        self._stage1_drives(state_in, state_aug, control, dt)
        return None

    def _stage1_drives(self, state_in: State, state_aug: State, control: Control, dt: float):
        """Populate ``state_aug.joint_tau`` and apply the effort-limit clamp.

        Torque-bucket ownership after this routine returns:

        - ``state_aug.joint_tau`` holds the rigid / passive / Coriolis /
          gravity / external / :attr:`~newton.Control.joint_f` contribution
          summed with the clamped explicit-PD drive contribution ``u0``.
        - The implicit-PD drive response is carried by ``self.aug_row_K``
          and realized through ``H_tilde^{-1}`` during the linear solve.

        :attr:`~newton.Model.joint_effort_limit` is always applied as
        an **actuator-only** clamp on ``u0`` *before* it is added to
        ``joint_tau``; the rigid / passive / external bucket is left uncapped
        (MuJoCo ``actuatorfrcrange`` / PhysX drive ``maxForce`` convention).

        """
        model = self.model

        if model.articulation_count:
            if self.drive_mode == "augmented" and self.articulation_max_dofs > 0:
                wp.launch(
                    eval_rigid_tau_and_augmented_drives,
                    dim=model.articulation_count,
                    inputs=[
                        self._dynamics(state_in, state_aug, control),
                        self.articulation_H_rows,
                        model.joint_target_q_start,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        control.joint_target_q,
                        control.joint_target_qd,
                        model.joint_effort_limit,
                        self.articulation_max_dofs,
                        dt,
                    ],
                    outputs=[self.aug_row_counts, self.aug_row_dof_index],
                    block_dim=SERIAL_BLOCK_DIM,
                    device=model.device,
                )
                return
            # Evaluate joint torques. After this launch `joint_tau` owns
            # the rigid / passive / Coriolis / gravity / external /
            # `control.joint_f` bucket only; the actuator-drive bucket has
            # not been added yet.
            self._launch_rigid_tau(state_in, state_aug, control)

            if self.drive_mode == "physx_pgs":
                if self.articulation_max_dofs > 0:
                    self.aug_row_counts.zero_()
                return

    def _stage1_crba(
        self,
        state_aug: State,
        drive_rows_ready: wp.Event | None,
    ):
        model = self.model
        global_flag = 1 if ((self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update) else 0
        # NOTE: this host-evaluated flag is BAKED into CUDA graphs at capture
        # time. With K physics steps captured per graph, replays repeat the
        # K-step flag pattern captured starting at the capture-time _step, so
        # only intervals that divide the captured pattern behave as requested
        # (e.g. a 2-step graph captured at _step=0 bakes [1, 0]: interval=2 is
        # exact, interval>=3 silently degrades to an effective interval of 2).
        # The same flag gates the H memsets below and at end of step so the
        # baked memset cadence always matches the baked update cadence.
        self._mass_update_global_flag = bool(global_flag)

        wp.launch(
            build_mass_update_mask,
            dim=model.articulation_count,
            inputs=[
                global_flag,
                self._mass_update_requested,
            ],
            outputs=[self.mass_update_mask],
            device=model.device,
        )

        if getattr(self, "_tree_kinetic_state", None) is not None:
            if drive_rows_ready is not None:
                wp.get_stream(model.device).wait_event(drive_rows_ready)
            self._tree_kinetic_state.refresh(state_aug)
            # Consume requests just like the common tail. Leaving either set
            # changes the held-factor cadence on subsequent captured steps.
            self._mass_update_requested.zero_()
            self._force_mass_update = False
            return

        # The composite and every mass consumer read inertia records about the origin: the records of
        # articulations a reuse step's mask selects are placed at their current poses first.
        dynamics = self._dynamics(None, state_aug)
        if not global_flag:
            wp.launch(
                refresh_body_inertia, dim=model.joint_count, inputs=[dynamics], block_dim=128, device=model.device
            )

        # The composite inertia of the articulations the mass mask selects.
        wp.launch(
            composite_inertia,
            dim=model.articulation_count,
            inputs=[dynamics],
            device=model.device,
            block_dim=SERIAL_BLOCK_DIM,
        )

        # Zeroing H is hygiene, not a correctness requirement: H is only read
        # by the Cholesky kernels, which early-exit when mass_update_mask is 0,
        # and a masked rebuild never relies on a zeroed buffer because
        # crba_fill STORES every structural nonzero (full ancestor
        # chain incl. the whole diagonal) before apply_augmented_mass_diagonal
        # _grouped ADDS K onto that freshly stored diagonal under the same
        # mask. Structural zeros are written by no kernel and stay zero from
        # allocation. Skipping the memset on global_flag==0 steps therefore
        # leaves limit-change refreshes (device-side mask) exact while saving
        # the per-step memset.
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            if self._H_bufs is None and global_flag:  # not double-buffered
                self.H_by_size[size].zero_()
            wp.launch(
                crba_fill,
                dim=int(n_arts * size),
                inputs=[
                    dynamics,
                    self._body_inertia_terms if size in self._free_body_inertia_sizes else self.body_I_c,
                    self.group_to_art[size],
                    size,
                ],
                outputs=[self.H_by_size[size]],
                device=model.device,
                block_dim=128,
            )
            # K after the assembly.
            wp.launch(
                apply_augmented_mass_diagonal_grouped,
                dim=n_arts,
                inputs=[
                    self.group_to_art[size],
                    self.articulation_dof_start,
                    size,
                    self.articulation_max_dofs,
                    self.mass_update_mask,
                    self.aug_row_counts,
                    self.aug_row_dof_index,
                    self.aug_row_K,
                ],
                outputs=[self.H_by_size[size]],
                device=model.device,
            )

        self._mass_update_requested.zero_()

        self._force_mass_update = False

    def _build_stage_kernels(self) -> None:
        """Build the stage-3 and stage-4 kernels the plan's tables launch, and only those (one entry per size)."""
        plan, arch, threads, rows = self._plan, self.model.device.arch, self.tile_threads, self.dense_max_constraints
        sizes = self.size_groups

        def routed(stage, size, route):
            entry = getattr(plan, stage).get(size)
            return entry is not None and entry.name == route

        self._cholesky_kernels_by_size = {
            size: response_builds_dense.factor_tiles_kernel(size, threads) if routed("factor", size, "tiles") else None
            for size in sizes
        }
        self._triangular_solve_kernels_by_size = {
            size: _get_triangular_solve_kernel(size, arch, threads) if routed("trisolve", size, "tiles") else None
            for size in sizes
        }
        self._hinv_jt_kernels_by_size = {}
        self._hinv_jt_chunk_count_by_size = {}
        for size in sizes:
            chunk = self._execution_plan.hinv_jt_chunk_size(size) if routed("response", size, "tiles") else None
            self._hinv_jt_chunk_count_by_size[size] = 0 if chunk is None else (rows + chunk - 1) // chunk
            self._hinv_jt_kernels_by_size[size] = (
                None
                if chunk is None
                else response_builds_dense.response_tiles_kernel(
                    size,
                    rows,
                    threads,
                    constraint_chunk_size=chunk,
                    write_world=self._hinv_jt_writes_world,
                    write_group=self._hinv_jt_tiled_writes_group,
                    compute_diag=size in self._hinv_jt_diag_sizes,
                )
            )
        # Sizes the plan routes to the per-row kernel (e.g. free bodies) consume the group-layout response (diag
        # from J.Y, world impulse apply), so the grouped kernel writes the group copy for them.
        self._grouped_kernels_by_size = {
            size: response_builds_dense.response_warps_kernel(
                size,
                int(rows),
                write_world=bool(self._hinv_jt_writes_world),
                write_group=bool(self._hinv_jt_tiled_writes_group) or not self._execution_plan.use_tiled_hinv_jt(size),
                compute_diag=size in self._hinv_jt_diag_sizes,
            )
            for size in plan.sizes("response", "grouped")
        }
        self._hinv_jt_fused_kernels_by_size = {
            size: response_builds_dense.delassus_tiles_kernel(size, rows, threads)
            if routed("response", size, "fused")
            else None
            for size in sizes
        }
        self._delassus_kernels_by_size = {
            size: response_builds_dense.delassus_stream_kernel(size, rows, arch)
            if routed("delassus", size, "tiles")
            else None
            for size in sizes
        }

    def _stage2_factor(self, size: int) -> None:
        """Factor one size's mass matrix along the plan's stage-2 route: none for compact, sparse and fused sizes."""
        route = self._plan.factor[size].name
        if route == "diagonal":
            self._stage2_cholesky_diagonal(size)
        elif route == "tiles":
            self._stage2_cholesky_tiled(size)
        elif route == "loop":
            self._stage2_cholesky_loop(size)

    def _stage2_cholesky_diagonal(self, size: int) -> None:
        """Factor an articulation group known to have diagonal generalized inertia."""
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.factor_coordinates_kernel(),
            dim=n_arts * size,
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.group_to_art[size],
                self.mass_update_mask,
                size,
            ],
            outputs=[self.L_by_size[size]],
            device=self.model.device,
        )

    def _stage2_cholesky_tiled(self, size: int):
        """Factor a tiled size's H (factor route ``tiles``)."""
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch_tiled(
            self._cholesky_kernels_by_size[size],
            dim=[n_arts],
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.group_to_art[size],
                self.mass_update_mask,
            ],
            outputs=[self.L_by_size[size]],
            block_dim=self.tile_threads,
            device=model.device,
        )

    def _stage2_cholesky_loop(self, size: int):
        """Factor a loop size's H (factor route ``loop``)."""
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.factor_articulations_kernel(),
            dim=n_arts,
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.group_to_art[size],
                self.mass_update_mask,
                size,
            ],
            outputs=[self.L_by_size[size]],
            device=model.device,
        )

    def _stage3_zero_qdd(self, state_aug: State):
        state_aug.joint_qdd.zero_()

    def _stage3_trisolve(self, size: int, state_aug: State) -> None:
        """Solve one size's unconstrained acceleration along the plan's stage-3 route."""
        route = self._plan.trisolve[size].name
        if route == "diagonal":
            self._stage3_trisolve_diagonal(size, state_aug)
        elif route == "tiles":
            self._stage3_trisolve_tiled(size, state_aug)
        else:
            self._stage3_trisolve_loop(size, state_aug)

    def _stage3_trisolve_diagonal(self, size: int, state_aug: State) -> None:
        """Apply a diagonal generalized inertia inverse to joint forces."""
        wp.launch(
            solve_diagonal_mass,
            dim=self.n_arts_by_size[size] * size,
            inputs=[
                self.L_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
                state_aug.joint_tau,
            ],
            outputs=[state_aug.joint_qdd],
            device=self.model.device,
        )

    def _stage3_trisolve_tiled(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.n_arts_by_size[size]

        wp.launch(
            gather_tau_to_groups,
            dim=n_arts,
            inputs=[
                state_aug.joint_tau,
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
            ],
            outputs=[self.tau_by_size[size]],
            device=model.device,
        )
        wp.launch_tiled(
            self._triangular_solve_kernels_by_size[size],
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.tau_by_size[size],
            ],
            outputs=[self.qdd_by_size[size]],
            block_dim=self.tile_threads,
            device=model.device,
        )
        wp.launch(
            scatter_qdd_from_groups,
            dim=n_arts,
            inputs=[
                self.qdd_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_trisolve_loop(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            trisolve_loop,
            dim=n_arts,
            inputs=[
                self.L_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
                state_aug.joint_tau,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_compute_v_hat(self, state_in: State, state_aug: State, dt: float, stage3_qd: wp.array):
        model = self.model
        if not model.joint_count:
            return
        wp.launch(
            compute_velocity_predictor,
            dim=model.joint_dof_count,
            inputs=[
                stage3_qd,
                self._kinematic_dof_mask,
                dt,
            ],
            outputs=[state_aug.joint_qdd, self.v_hat],
            device=model.device,
        )
        if self._free_body_gyro_kernel is not None:
            wp.launch(
                self._free_body_gyro_kernel,
                dim=self.n_arts_by_size[6],
                inputs=[
                    self.group_to_art[6],
                    self.is_free_rigid,
                    model.articulation_start,
                    model.joint_child,
                    model.joint_qd_start,
                    self._kinematic_joint_mask,
                    state_aug.body_q_com,
                    model.joint_X_p,
                    model.body_inertia,
                    self.L_by_size[6],
                    stage3_qd,
                    dt,
                ],
                outputs=[self.v_hat],
                device=model.device,
            )
        self._stage3_correct_free_root_velocity(state_in, stage3_qd, dt)

    def _stage3_correct_free_root_velocity(
        self, state_in: State, stage3_qd: wp.array, dt: float, *, apply_transport: bool = True
    ) -> None:
        """Apply canonical gyro physics after either ordinary or private prediction."""
        model = self.model
        # Keep v_hat on jcalc_integrate's free-root convention. The compact
        # launch excludes every non-root joint; a live device mask handles
        # runtime kinematic changes without changing graph topology.
        if not self._free_root_joint_count:
            return
        # The retained clutter experiment replaces, never compounds, main's
        # angular correction. Both paths still need linear transport once.
        if self._free_rigid_body_count and self._free_body_gyro_kernel is None:
            wp.launch(
                apply_free_root_velocity_corrections,
                dim=self._free_root_joint_count,
                inputs=[
                    self._free_root_joint_indices,
                    model.joint_qd_start,
                    model.joint_child,
                    self.body_to_articulation,
                    self.is_free_rigid,
                    self.art_group_idx,
                    self._kinematic_joint_mask,
                    state_in.body_q,
                    model.body_inertia,
                    self.L_by_size[6],
                    stage3_qd,
                    dt,
                    model.requires_grad,
                    apply_transport,
                ],
                outputs=[self.v_hat],
                device=model.device,
            )
        elif apply_transport:
            wp.launch(
                apply_free_root_transport_to_predictor,
                dim=self._free_root_joint_count,
                inputs=[
                    self._free_root_joint_indices,
                    model.joint_qd_start,
                    self._kinematic_joint_mask,
                    stage3_qd,
                    dt,
                ],
                outputs=[self.v_hat],
                device=model.device,
            )

    def _patch_frame_reuse(self) -> tuple[bool, str]:
        """Return whether captured patch builds keep the sorted pairs of a reused contact frame, with the reason.

        A captured build keeps them when the previous build was captured in the same
        graph over the same contact stream, its ``contact_generation`` array, and the
        host issued no write of that stream in between (see :class:`_FrictionPatchState`).
        On by default;
        ``FEATHER_PGS_PATCH_FRAME_REUSE=0`` sorts every build.
        """
        if not self._friction_anchors_enabled:
            return False, "Point friction sorts no patch pairs."
        if os.environ.get("FEATHER_PGS_PATCH_FRAME_REUSE", "1") == "0":
            return False, "FEATHER_PGS_PATCH_FRAME_REUSE=0 sorts the patch pairs of every build."
        return True, (
            "A captured build keeps the sorted patch pairs of the previous build captured in the same graph over the "
            "same contact stream (its contact_generation array, which wrappers of one Contacts share) when the host "
            "noted no write of that stream in between (such as a collision pass, Contacts.clear() or a "
            "coupled-solver filter); other builds sort."
        )

    def _build_friction_patches(self, state_in, contacts):
        """Construct the shared contact law independently of the dynamics owner."""
        if self._friction_anchors_enabled and contacts is not None:
            self._friction_patches.build(
                self.model,
                state_in,
                contacts,
                body_to_articulation=self.body_to_articulation,
                is_free_rigid=self.is_free_rigid if self.is_free_rigid is not None else self._dummy_is_free_rigid,
                contact_gap_gate=self.contact_gap_gate,
                same_articulation_gap_gate=self.same_articulation_contact_gap_gate,
                articulation_pair_gap_gate=self.articulation_pair_contact_gap_gate,
                friction_gap=self.contact_friction_gap_threshold,
                friction_articulation_pairs_only=self.contact_friction_articulation_pairs_only,
            )

    @staticmethod
    def _patch_frame_built(contacts):
        """Whether this step builds a patch frame, under the contact row stage's own condition."""
        return (
            contacts is not None
            and getattr(contacts, "rigid_contact_count", None) is not None
            and contacts._rigid_contact_rows > 0
        )

    def _link_friction_patches(self, contacts):
        """Link shared normal loads before any consumer packs row metadata."""
        if self._friction_anchors_enabled and self._patch_frame_built(contacts):
            self._friction_patches.link(
                self.contact_world,
                self.contact_slot,
                self.contact_path,
                self.contact_slots_needed,
                self._patch_routes(),
            )

    def _finish_friction_patches(self, state_in, state_out, contacts, dt):
        """Publish persistent patch history from the integrated public state."""
        if not self._friction_anchors_enabled:
            return
        if contacts is None:
            # A memset of today's rows: a capture of it is recaptured after growth.
            self._contact_pool.note_unbound()
            self._friction_patches.previous.valid.zero_()
            return
        # A step without a frame holds no patch rows; it stores the last frame again.
        routes = self._patch_routes() if self._patch_frame_built(contacts) else patch_routes(())
        self._friction_patches.finish(
            state_in,
            state_out,
            self.model.body_com,
            self.contact_world,
            self.contact_slot,
            self.contact_path,
            self.contact_slots_needed,
            routes,
            dt,
        )

    def _patch_routes(self):
        """Pack the routes that can hold patch rows into one launch argument."""
        return patch_routes(row for row in self._patch_row_arrays() if row[1] is not None)

    def _patch_row_arrays(self):
        """Yield possible contact routes and their configured warm-start decay."""
        # With only free response bodies, the allocator routes every responsive
        # contact through matrix-free rows. Dense bilateral rows can still exist,
        # but have no patch links or impulses.
        if self._has_non_free_articulations or not self._has_free_rigid_bodies:
            yield (
                0,
                self.row_parent,
                self.row_mu,
                self.impulses,
                self.pgs_warmstart_decay if self.pgs_warmstart else 0.0,
            )
        if self._has_free_rigid_bodies:
            yield (
                1,
                self.mf_row_parent,
                self.mf_row_mu,
                self.mf_impulses,
                self._mf_warmstart_decay if self._mf_warmstart_enabled else 0.0,
            )

    def _finalize_constraint_status(self, contacts: Contacts | None) -> None:
        """Latch current per-world capacity failures for every row owner."""
        # The contact capacity (argument 8) follows the ready rows in a capture whose launches bind.
        self._contact_pool.launch(
            _finalize_constraint_status,
            self.world_count,
            inputs=[
                self.slot_counter,
                self.mf_slot_counter if self._has_free_rigid_bodies else self._dummy_mf_slot_counter,
                self._row_dropped_all,
                contacts.rigid_contact_count if contacts is not None else self._dummy_contact_count,
                contacts._reduction_overflow if contacts is not None else self._dummy_contact_count,
                self.dense_max_constraints,
                self.mf_max_constraints,
                contacts._rigid_contact_rows if contacts is not None else 0,
                self.constraint_overflow,
            ],
            row_args=(8,) if contacts is not None else (),
        )

    def _place_sorted_contact_rows(self, contacts, contact_build_threads, mf_slot_counter, mf_first_rejected_slot):
        """Give each world's contact rows the order of sorted contacts, whatever order the allocator reserved them in.

        Runs only when the contacts come from a deterministic collision pipeline
        (``Contacts.rigid_contacts_pair_sorted``), whose determinism Newton promises when the
        solver is deterministic as well; the default pipeline keeps the allocator's atomic
        reservation, whose order on CUDA follows the warps. A captured step keeps the order of
        its capture. Dense rows are placed unless the local row packets or keyed rows place
        their own; matrix-free rows are placed with free rigid bodies.
        """
        order_dense = int(self._row_packets is None and self._keyed_rows is None)
        order_mf = int(self._has_free_rigid_bodies)
        if not (order_dense or order_mf):
            return
        device = self.model.device
        runs = (self._contact_run_first, self._contact_run_last, self._contact_run_rows)
        blocks = (self._contact_run_lead, self._contact_run_trail_start)
        # Complete warps on CUDA, whose lanes merge each run's contacts before recording it.
        lanes = 32 if device.is_cuda else 1
        record_threads = -(-contact_build_threads // lanes) * lanes
        # Its warps merge runs, so the grid stays whole: no extent follows the count.
        self._contact_pool.launch(
            record_world_contact_runs,
            record_threads,
            inputs=[
                contacts.rigid_contact_count,
                record_threads,
                lanes,
                contacts.rigid_contact_shape0,
                self.contact_world,
                self.contact_path,
                self.contact_slots_needed,
                order_dense,
                order_mf,
                *blocks,
            ],
            outputs=list(runs),
        )
        # One complete warp per world on CUDA, whose lanes sum each other's rows.
        wp.launch(
            place_world_contact_runs,
            dim=(self.world_count, lanes),
            inputs=[
                contacts.rigid_contact_shape0,
                self.contact_world,
                self.contact_path,
                self.contact_slots_needed,
                self.slot_counter,
                self._dense_first_rejected_slot,
                mf_slot_counter,
                mf_first_rejected_slot,
                *blocks,
                lanes,
            ],
            outputs=[*runs, self.contact_slot],
            device=device,
        )

    def _stage4_build_rows(self, state_in: State, state_aug: State, control: Control, contacts: Contacts, dt: float):
        if self.contact_compliance:
            _contact_compliance.start_step(self, contacts, dt)
            # The allocator writes slots_needed only after intentional skip gates.
            # Clear prior-step requests so an excluded contact cannot look overflowed.
            self.contact_slots_needed.zero_()
        model = self.model
        max_constraints = self.dense_max_constraints
        # Every row producer writes the dense row records through one struct, and the joint row producers read
        # the joint table through another, both bound for this step.
        dense_rows = bind_dense_rows(self)
        joints = bind_joint_table(self)
        mf_active = self._has_free_rigid_bodies
        if self._has_prescribed_response:
            self.mf_target_velocity.zero_()

        # One launch clears the world-level allocation state and, with free rigid bodies, the
        # matrix-free counters and impulses (PGS reads mf_impulses before its first write; the
        # warm-start seed pass re-gathers after build_mf_contact_rows). Other row buffers are
        # fully overwritten by their producers.
        wp.launch(
            _clear_dense_row_state,
            dim=(self.world_count, self.mf_impulses.shape[1] if mf_active else 1),
            inputs=[
                self.slot_counter,
                self._row_dropped_all,
                self._dense_first_rejected_slot,
                1 if mf_active else 0,
                self.mf_slot_counter if mf_active else self._dummy_mf_slot_counter,
                self._mf_first_rejected_slot if mf_active else self._dummy_mf_slot_counter,
                self.mf_constraint_count if mf_active else self._dummy_mf_slot_counter,
                self.mf_impulses if mf_active else self._dummy_mf_impulses,
            ],
            device=model.device,
        )

        has_free_rigid_flag = 1 if mf_active else 0
        # Dummy arrays when MF is not active (kernel still needs valid pointers).
        # Preallocated in __init__: a per-step wp.zeros here would add an
        # allocation + memset every step and, under CUDA graph capture, leave
        # the graph holding a pointer to a freed temporary.
        is_free_rigid = self.is_free_rigid if self.is_free_rigid is not None else self._dummy_is_free_rigid
        mf_slot_counter = self.mf_slot_counter if mf_active else self._dummy_mf_slot_counter
        dense_dropped_rows = self._row_dropped_dense
        mf_dropped_rows = self._row_dropped_mf
        mf_first_rejected_slot = self._mf_first_rejected_slot if mf_active else self._dummy_mf_slot_counter
        # The packed rows are never cleared: every producer writes its rows' sides whole (dense_rows.clear_side).
        j_buffers_zeroed = (
            self._mf_sweeps
            or self._sparse_mass_matrix_size is not None
            or self._row_packets is not None
            or self._keyed_rows is not None
        )
        support = self._row_support()
        if self._compact_contact_boundary:
            from .compact_contact import clear_limit_prefix  # noqa: PLC0415

            wp.launch(
                clear_limit_prefix,
                dim=(self.world_count, 12),
                inputs=[self._sparse_diagonal_dense_groups, self.J_by_size[6], self._sparse_diagonal_row_dof],
                device=model.device,
            )
            j_buffers_zeroed = True
        if self._row_packets is not None:
            self._row_packets.prepare_prefix(state_in, dt)
        if self._keyed_rows is not None:
            self._keyed_rows.begin_rows(state_in, state_aug, dt)

        drive_active = self.drive_mode == "physx_pgs" and self.drive_slot is not None
        if drive_active:
            wp.launch(
                allocate_physx_drive_slots,
                dim=model.articulation_count,
                inputs=[
                    joints,
                    max_constraints,
                ],
                outputs=[
                    self.drive_slot,
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                ],
                device=model.device,
            )

            if self._H_bufs is None and not self._mf_sweeps:
                for size in self.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True

            for size in self.size_groups:
                n_arts = self.n_arts_by_size[size]
                wp.launch(
                    populate_physx_drive_J_for_size,
                    dim=n_arts,
                    inputs=[
                        joints,
                        state_in.joint_q,
                        control.joint_target_q,
                        control.joint_target_qd,
                        self._vel_limit_skip_driven,
                        self.drive_slot,
                        self.group_to_art[size],
                        support,
                    ],
                    outputs=[
                        self._jacobian_rows(size),
                        dense_rows,
                        self.drive_stiffness,
                        self.drive_damping,
                        self.drive_geom_error,
                        self.drive_max_force,
                        self.drive_vel_limit,
                    ],
                    device=model.device,
                )

        # ── Dense row-family layout ──────────────────────────────────────
        # Slots are handed out in the order [drive][mimic][joint-limit]
        # [joint-velocity-limit][contact/friction] so each PhysX-grasp GS
        # row phase owns one contiguous dense row range:
        #   row_phase 3 (drive/mimic/position limits): [0, bounds[0])
        #   row_phase 5 (velocity limits):             [bounds[0], bounds[1])
        #   row_phase 4 (contacts + friction):         [bounds[1], m_dense)
        # The two snapshot_dense_phase_bound launches below record the family
        # boundaries and must stay glued to this allocation sequence
        # (slot_counter is zeroed at the top of this method). Mimic rows sit
        # between the (static-count) drive family and the (activation-gated,
        # flickering) limit family so warm-started mimic lambdas keep a stable
        # slot index across steps.

        # Mimic (joint coupling) equality rows — static count per model.
        if self._mimic_count > 0 and self._row_packets is None:
            wp.launch(
                allocate_mimic_slots,
                dim=self._mimic_count,
                inputs=[field_group.bind(Mimics, solver=self, model=model), max_constraints],
                outputs=[
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                ],
                device=model.device,
            )
            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True
            for size in self.size_groups:
                if size not in self._mimic_sizes:
                    continue
                n_arts = self.n_arts_by_size[size]
                if self._sparse_velocity_limits and size == self._sparse_mass_matrix_size:
                    indices = self._sparse_mass_matrix_indices
                    wp.launch(
                        response_builds_rows.factor_mimic_rows_kernel,
                        dim=n_arts * 32,
                        inputs=[
                            self.group_to_art[size],
                            self.art_to_world,
                            self.articulation_world_dof_offset,
                            self.articulation_dof_start,
                            self.mimic_slot,
                            self._mimic_art_start,
                            self._mimic_art_list,
                            self._mimic_dof0,
                            self._mimic_dof1,
                            self._mimic_q0,
                            self._mimic_q1,
                            model.constraint_mimic_coef0,
                            model.constraint_mimic_coef1,
                            state_in.joint_q,
                            self.pgs_beta,
                            self.pgs_cfm,
                            indices.ancestor_mask,
                            indices.inverse_permutation,
                            indices.lookup,
                            self._sparse_Linv,
                            self.v_hat,
                        ],
                        outputs=[dense_rows, self._sparse_row_dof, self._sparse_row_factor, self._sparse_row_incident],
                        block_dim=128,
                        device=model.device,
                    )
                    continue
                wp.launch(
                    populate_mimic_J_for_size,
                    dim=n_arts,
                    inputs=[
                        field_group.bind(Mimics, solver=self, model=model),
                        joints,
                        self.group_to_art[size],
                        state_in.joint_q,
                        self.pgs_beta,
                        self.pgs_cfm,
                        support,
                    ],
                    outputs=[self._jacobian_rows(size), dense_rows],
                    device=model.device,
                )

        # Connect (loop-closure) equality rows — static count per model.
        if self._connect_count > 0:
            wp.launch(
                allocate_connect_slots,
                dim=self._connect_count,
                inputs=[field_group.bind(Connects, solver=self), max_constraints],
                outputs=[
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                ],
                device=model.device,
            )
            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True
            for size in self.size_groups:
                n_arts = self.n_arts_by_size[size]
                wp.launch(
                    populate_connect_J_for_size,
                    dim=n_arts,
                    inputs=[
                        joints,
                        field_group.bind(Connects, solver=self),
                        self.group_to_art[size],
                        size,
                        state_in.body_q,
                        state_in.body_qd,
                        model.body_com,
                        self.body_to_joint,
                        self.body_to_articulation,
                        model.joint_ancestor,
                        state_aug.joint_S_s,
                        self.articulation_origin,
                        self.pgs_beta,
                        self.pgs_cfm,
                        support,
                    ],
                    outputs=[self._jacobian_rows(size), dense_rows],
                    device=model.device,
                )

        if self._fused_diagonal_joint_limits:
            wp.launch(
                prepare_fused_diagonal_joint_limits,
                dim=self.world_count * self.max_world_dofs,
                inputs=[
                    joints,
                    self.world_dof_indices,
                    self.max_world_dofs,
                    self._fused_diagonal_limit_dof_mask,
                    self._joint_limit_q_index,
                    state_in.joint_q,
                    self.joint_limit_activation_gap,
                    self.pgs_beta,
                    dt,
                ],
                outputs=[
                    self._fused_diagonal_limit_active,
                    self._fused_diagonal_limit_lower_rhs,
                    self._fused_diagonal_limit_upper_rhs,
                ],
                device=model.device,
            )

        # Allocate and populate joint-limit rows per response-size group.
        if (
            self.enable_joint_limits
            and self._joint_limit_sizes
            and self._row_packets is None
            and self._keyed_rows is None
        ):
            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True
            # The native limit rows (prefix_rows) read the law and write the records through one struct.
            law = field_group.bind(prefix_rows.PrefixLaw, solver=self, model=model, state_in=state_in, dt=dt)
            for size in self.size_groups:
                if size not in self._joint_limit_sizes or size in self._fused_diagonal_joint_limit_sizes:
                    continue
                n_arts = self.n_arts_by_size[size]
                if size == self._sparse_mass_matrix_size:
                    indices = self._sparse_mass_matrix_indices
                    wp.launch(
                        response_builds_rows.factor_limit_rows_kernel,
                        dim=n_arts * 32,
                        inputs=[
                            self.group_to_art[size],
                            self.art_to_world,
                            self.articulation_world_dof_offset,
                            self.articulation_dof_start,
                            law,
                            indices.ancestor_mask,
                            indices.inverse_permutation,
                            indices.lookup,
                            self._sparse_Linv,
                            self.v_hat,
                        ],
                        outputs=[
                            self._sparse_row_dof,
                            self._sparse_row_factor,
                            self._sparse_row_incident,
                            self._sparse_row_support,
                        ],
                        block_dim=128,
                        device=model.device,
                    )
                    continue
                warp_kernel = self._joint_limit_warp_kernels.get(size)
                if warp_kernel is not None:
                    wp.launch_tiled(
                        warp_kernel,
                        dim=[(n_arts + _JOINT_LIMIT_WARPS_PER_BLOCK - 1) // _JOINT_LIMIT_WARPS_PER_BLOCK],
                        inputs=[
                            n_arts,
                            self.articulation_dof_start,
                            self.art_to_world,
                            self.group_to_art[size],
                            self._resolved_worlds,
                            law,
                            support,
                        ],
                        outputs=[self._jacobian_rows(size)],
                        block_dim=32 * _JOINT_LIMIT_WARPS_PER_BLOCK,
                        device=model.device,
                    )
                else:
                    wp.launch(
                        build_joint_limit_rows_for_size,
                        dim=n_arts,
                        inputs=[
                            joints,
                            state_in.joint_q,
                            self.joint_limit_activation_gap,
                            self.group_to_art[size],
                            max_constraints,
                            self.pgs_beta,
                            self.pgs_cfm,
                            self._resolved_worlds,
                            support,
                        ],
                        outputs=[
                            self.slot_counter,
                            self._dense_first_rejected_slot,
                            self._jacobian_rows(size),
                            dense_rows,
                        ],
                        device=model.device,
                    )

        # Unconditional: the phase-3 boundary must include position-limit
        # rows and remain valid when that family is disabled (when it is just
        # the current watermark at the start of the velocity-limit segment).
        if self._row_packets is None and self._keyed_rows is None:
            wp.launch(
                snapshot_dense_phase_bound,
                dim=self.world_count,
                inputs=[self.slot_counter, 0],
                outputs=[self.dense_phase_bounds],
                device=model.device,
            )

        # Allocate + populate joint velocity-limit rows (per-DOF clamp on
        # |qdot_i| against model.joint_velocity_limit). Dense contact rows
        # are allocated after this block so contact/friction rows form the
        # trailing per-world slot segment; PhysX's documented row-visit
        # ordering (vel-limit rows fire after contact, drive, friction, and
        # positional-limit rows — physx-deep-dive §7) is enforced by the GS
        # phase schedule, which visits velocity-limit rows in a dedicated
        # final pass regardless of slot order.
        if self.enable_joint_velocity_limits and self.velocity_limit_slot is not None and self._keyed_rows is None:
            wp.launch(
                allocate_joint_velocity_limit_slots,
                dim=model.articulation_count,
                inputs=[
                    joints,
                    self.v_hat,
                    self.velocity_limit_activation_fraction,
                    self._vel_limit_drive_slot_arg,
                    self._vel_limit_skip_driven,
                    max_constraints,
                ],
                outputs=[
                    self.velocity_limit_slot,
                    self.velocity_limit_sign,
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                ],
                device=model.device,
            )

        # Unconditional phase-5 boundary snapshot; dense contact rows start here.
        if self._row_packets is None and self._keyed_rows is None:
            wp.launch(
                snapshot_dense_phase_bound,
                dim=self.world_count,
                inputs=[self.slot_counter, 1],
                outputs=[self.dense_phase_bounds],
                device=model.device,
            )

        if self._sparse_diagonal_contact_solve:
            wp.launch(
                clear_sparse_diagonal_response_prefix,
                dim=self.world_count,
                inputs=[self.dense_phase_bounds],
                outputs=[self._sparse_diagonal_row_dof],
                device=model.device,
            )

        if self.enable_joint_velocity_limits and self.velocity_limit_slot is not None and self._keyed_rows is None:
            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True
            for size in self.size_groups:
                n_arts = self.n_arts_by_size[size]
                if self._sparse_velocity_limits and size == self._sparse_mass_matrix_size:
                    indices = self._sparse_mass_matrix_indices
                    wp.launch(
                        response_builds_rows.factor_velocity_limit_rows_kernel,
                        dim=n_arts * 32,
                        inputs=[
                            self.group_to_art[size],
                            self.art_to_world,
                            self.articulation_world_dof_offset,
                            self.articulation_dof_start,
                            model.joint_velocity_limit,
                            self.velocity_limit_slot,
                            self.velocity_limit_sign,
                            self.pgs_cfm,
                            indices.ancestor_mask,
                            indices.inverse_permutation,
                            indices.lookup,
                            self._sparse_Linv,
                            self.v_hat,
                        ],
                        outputs=[
                            dense_rows,
                            self._sparse_row_dof,
                            self._sparse_row_factor,
                            self._sparse_row_incident,
                            self._sparse_row_support,
                        ],
                        block_dim=128,
                        device=model.device,
                    )
                    continue
                wp.launch(
                    populate_joint_velocity_limit_J_for_size,
                    dim=n_arts * int(size) * 2,
                    inputs=[
                        joints,
                        self.velocity_limit_slot,
                        self.velocity_limit_sign,
                        self.group_to_art[size],
                        self.pgs_cfm,
                        int(size),
                        support,
                    ],
                    outputs=[self._jacobian_rows(size), dense_rows],
                    device=model.device,
                )

        if (
            contacts is not None
            and getattr(contacts, "rigid_contact_count", None) is not None
            and contacts._rigid_contact_rows > 0
        ):
            enable_friction_flag = 1 if self.enable_contact_friction else 0
            pool = self._contact_pool
            count = contacts.rigid_contact_count
            # Sized for the bound in a capture whose launches bind: every replay runs them at the live
            # count and strides with the captured threads (ContactPool.launch); otherwise for the ready rows.
            rows = pool.launch_rows()
            contact_build_threads = min(rows, _CONTACT_BUILD_THREAD_CAP * _CONTACT_THREADS_X)

            self._build_friction_patches(state_in, contacts)

            # The allocator and the producers read the contacts, the bodies and the row law, and write the
            # contacts' allocation records and their row records, through one struct
            # (contact_input.ContactInput), bound for this step.
            contact_input = bind_contact_input(self, state_in, state_aug, contacts, dt, 1.0)
            # The friction row law every producer writes: point friction is the patch law with beta 0.
            law = patch_row_law(self)

            # Under persistent patch friction, contacts of resolved worlds that the complete path would
            # give rows are marked so the patch finish releases their anchors (kernels.py).
            pool.launch(
                allocate_world_contact_slots,
                contact_build_threads,
                inputs=[
                    contact_input,
                    contact_build_threads,
                    self.body_to_articulation,
                    self.art_to_world,
                    model.body_flags,
                    self.body_has_response_dofs,
                    is_free_rigid,
                    has_free_rigid_flag,
                    self.contact_gap_gate,
                    self.same_articulation_contact_gap_gate,
                    self.articulation_pair_contact_gap_gate,
                    max_constraints,
                    self.mf_max_constraints,
                    enable_friction_flag,
                    self.contact_friction_gap_threshold,
                    1 if self.contact_friction_articulation_pairs_only else 0,
                    1,
                    self._friction_patches.view,
                    self._resolved_worlds,
                ],
                outputs=[
                    self.slot_counter,
                    mf_slot_counter,
                    dense_dropped_rows,
                    mf_dropped_rows,
                    self._dense_first_rejected_slot,
                    mf_first_rejected_slot,
                ],
                extent=count,
                clamp=True,
            )
            if contacts.rigid_contacts_pair_sorted:
                self._place_sorted_contact_rows(
                    contacts, contact_build_threads, mf_slot_counter, mf_first_rejected_slot
                )

            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True

            # The first admitted producer writes every contact row: local row packets, keyed rows,
            # the compact and sparse producers, then the general builder. Under patch friction this
            # order is the row precedence: the local row packets and keyed rows (unless
            # FEATHER_PGS_PATCH_PARALLEL_ROWS=0) take their worlds' rows before the general builder's
            # masked rows, and native sparse factors admit neither.
            # get_path_selection() records the same plan (routes.contact_rows).
            plan = contact_row_plan(self)
            if plan.producer == "local_packets":
                self._row_packets.produce_contacts(contact_input)
            elif plan.producer == "keyed_rows":
                self._keyed_rows.produce_contacts(contact_input)
            elif plan.producer == "compact_contact_boundary":
                from .compact_contact import launch_contacts  # noqa: PLC0415

                launch_contacts(self, contact_input, contact_build_threads)
            elif plan.producer in SHARED_METADATA_PRODUCERS:
                if plan.metadata_launch:
                    pool.launch(
                        prepare_world_contact_rows,
                        contact_build_threads,
                        inputs=[contact_input, contact_build_threads, law],
                        extent=count,
                        clamp=True,
                    )
                contact_jacobian_workers = min(rows, _CONTACT_JACOBIAN_WORKER_CAP)
                for size in self.size_groups:
                    method = plan.methods[size]
                    if method == "sparse_response":
                        indices = self._sparse_mass_matrix_indices
                        sparse_contact_workers = min(
                            contact_build_threads, contact_jacobian_workers * 32 // self._sparse_contact_lanes
                        )
                        pool.launch(
                            self._sparse_contact_response_kernel,
                            sparse_contact_workers * self._sparse_contact_lanes,
                            inputs=[
                                contact_input,
                                sparse_contact_workers,
                                self.art_group_idx,
                                self.is_free_rigid,
                                self.articulation_world_dof_offset,
                                self._sparse_body_dof_mask,
                                law,
                                indices.permutation,
                                indices.row_offsets,
                                self._sparse_Linv,
                                self._sparse_free_factor_dummy,
                            ],
                            outputs=[
                                self._sparse_row_dof,
                                self._sparse_row_factor,
                                # The record builder writes the triplet records in the free-response
                                # slot, which the support table's routes leave unused.
                                self._sparse_row_free_response
                                if self._sparse_contact_records is None
                                else self._sparse_contact_records,
                                self._sparse_row_incident,
                                self._sparse_row_support,
                            ],
                            extent=count,
                            scale=self._sparse_contact_lanes,
                            clamp=True,
                            block_dim=128,
                        )
                        continue
                    if method == "sparse_diagonal":
                        continue  # sparse_pair_contact_rows_kernel below writes these rows.
                    pool.launch(
                        populate_world_J_for_compact_size,
                        (contact_jacobian_workers, 32),
                        inputs=[contact_input, contact_jacobian_workers, size, self.art_group_idx, law, support],
                        outputs=[self._jacobian_rows(size)],
                        extent=count,
                        clamp=True,
                    )
                if plan.producer == "sparse_diagonal":
                    # The dense builder has 32 lanes per worker. Give the scalar
                    # builder the contact-build thread budget; its live-prefix
                    # stride covers larger counts without visiting unused slots.
                    # Retain the explicit full-capacity experiment for replay.
                    sparse_contact_response_workers = rows if _SPARSE_CONTACT_DIRECT else contact_build_threads
                    pool.launch(
                        response_builds_rows.sparse_pair_contact_rows_kernel,
                        sparse_contact_response_workers,
                        inputs=[
                            contact_input,
                            sparse_contact_response_workers,
                            self._sparse_diagonal_response_size,
                            self.articulation_world_dof_offset,
                            self.body_single_response_dof,
                            self._diagonal_inverse_mass,
                            law,
                        ],
                        outputs=[self._sparse_diagonal_row_dof, self._sparse_diagonal_row_jy],
                        extent=count,
                        clamp=True,
                    )
                    if self._sparse_diagonal_contact_triples or self._sparse_diagonal_contact_units:
                        marker = self._mark_independent_sparse_contact_candidates_kernel
                        if marker is None:
                            raise RuntimeError("Independent sparse-contact marker kernel is unavailable")
                        pool.launch(
                            marker,
                            contact_build_threads,
                            inputs=[contact_input, contact_build_threads],
                            outputs=[self._sparse_diagonal_row_dof],
                            extent=count,
                            clamp=True,
                        )
            else:
                # The masked builder: a prelude writes the row records and per-contact flags, then one
                # thread per (contact, DOF) writes the Jacobian; one launch pair serves the combined sizes.
                flags = self._rows_prelude_flags
                rows_threads = min(rows, _ROWS_THREADS_X * _CONTACT_BUILD_THREAD_CAP)
                combined_done = False
                for size in self.size_groups:
                    method = plan.methods[size]
                    if method in ("masked_combined", "masked"):
                        if method == "masked_combined":
                            if combined_done:
                                continue
                            combined_done = True
                            size_a = plan.combined[0]
                            size_b = plan.combined[1] if len(plan.combined) > 1 else -1
                            lanes = max(size_a, size_b)
                            jacobian_threads = rows_threads * lanes
                        else:
                            size_a, size_b, lanes, jacobian_threads = size, -1, size, rows_threads * 32
                        J_a = self._jacobian_rows(size_a)
                        J_b = self._jacobian_rows(size_b) if size_b > 0 else J_a
                        pool.launch(
                            contact_row_prelude,
                            rows_threads * 3,
                            inputs=[contact_input, rows_threads * 3, size_a, size_b, law],
                            outputs=[flags],
                            extent=count,
                            scale=3,
                            clamp=True,
                        )
                        pool.launch(
                            populate_world_J_masked,
                            jacobian_threads,
                            inputs=[
                                contact_input,
                                jacobian_threads,
                                size_a,
                                size_b,
                                self.art_group_idx,
                                law,
                                flags,
                                support,
                            ],
                            outputs=[J_a, J_b],
                            extent=count,
                            scale=lanes,
                            clamp=True,
                        )
                        continue
                    pool.launch(
                        populate_world_J_for_size,
                        contact_build_threads,
                        inputs=[
                            contact_input,
                            contact_build_threads,
                            size,
                            self.art_group_idx,
                            self.body_to_joint,
                            model.joint_ancestor,
                            model.joint_qd_start,
                            law,
                            support,
                        ],
                        outputs=[self._jacobian_rows(size)],
                        extent=count,
                        clamp=True,
                    )

            # Build MF contact rows
            if mf_active:
                pool.launch(
                    build_mf_contact_rows,
                    contact_build_threads,
                    inputs=[
                        bind_mf_rows(self),
                        contact_input,
                        contact_build_threads,
                        int(self._has_prescribed_response),
                        law,
                    ],
                    extent=count,
                    clamp=True,
                )

            if mf_active:
                wp.launch(
                    finalize_constraint_counts_with_status,
                    dim=self.world_count,
                    inputs=[self.mf_slot_counter, self.mf_max_constraints, 1, self._mf_first_rejected_slot],
                    outputs=[self.mf_constraint_count, self._constraint_capacity_status],
                    device=model.device,
                )

                # ── MF warm-start seed ────────────────────────────────────
                # contact_slot + mf_row_type are now populated for THIS step.
                # mf_impulses was already memset at the top of this method
                # (covers vel-limit rows, padding tail, and any contact that
                # returns early in the gather); here we gather matched contacts'
                # previous impulses into their current slots by identity. The
                # carry itself lives in _ws_prev_mf_impulses.
                if self._mf_warmstart_enabled:
                    match_index = getattr(contacts, "rigid_contact_match_index", None)
                    if match_index is None:
                        raise NotImplementedError(
                            "FeatherPGS mf_warmstart=True requires the Contacts buffer to be "
                            "created with contact_matching enabled (rigid_contact_match_index "
                            "is None). Build the CollisionPipeline / Contacts with "
                            'contact_matching="latest" (or "sticky").'
                        )
                    pool.launch(
                        gather_mf_warmstart,
                        rows,
                        inputs=[
                            bind_contact_routes(self, contacts),
                            bind_mf_rows(self),
                            match_index,
                            self._ws_prev_slot_sorted,
                            self._ws_prev_mf_impulses,
                            self._ws_prev_mf_row_type,
                            self._ws_prev_mf_row_parent,
                            self.mf_constraint_count,
                            self._ws_prev_contact_normal,
                            self._mf_warmstart_decay,
                            # Carried support impulses are proportional to dt, so
                            # rescale by the exact step-size ratio (1 at fixed dt).
                            dt / self._ws_prev_dt if self._ws_prev_dt > 0.0 else 1.0,
                            self.mf_max_constraints,
                        ],
                        outputs=[self.mf_impulses],
                        extent=count,
                        max_blocks=self._contact_capacity_blocks,
                    )

        # Rigid-body velocity limits are matrix-free rows, not post-solve
        # clamps, in the MF path. Allocate after contacts so they occupy the
        # last MF slots; the fused PGS kernel also solves row_type=4 in a
        # final phase so articulated and rigid velocity-limit rows are both
        # visited after drive/contact/friction/position-limit rows.
        if mf_active:
            # Contact/friction rows all precede velocity-limit rows in the MF
            # slot space (the classifier completes before the vlim allocation
            # below). Record the partition boundary so the GS kernels can loop
            # only the contact prefix in contact phases and only the vlim tail
            # in velocity-limit phases instead of scanning all rows.
            wp.copy(self.mf_contact_rows_end, self.mf_slot_counter)
        if mf_active and self.rigid_velocity_limit_slot is not None:
            # Dummy when no articulation metadata exists; the kernel only
            # reads it for free-rigid roots, which then cannot occur.
            root_dof_start = (
                self.articulation_root_dof_start
                if self.articulation_root_dof_start is not None
                else wp.zeros((1,), dtype=wp.int32, device=model.device)
            )
            wp.launch(
                allocate_rigid_velocity_limit_slots,
                dim=self._free_rigid_body_count,
                inputs=[
                    bind_free_bodies(self),
                    model.body_flags,
                    root_dof_start,
                    self.v_hat,
                    self.velocity_limit_activation_fraction,
                    self.mf_max_constraints,
                    self._resolved_worlds,
                ],
                outputs=[
                    self.rigid_velocity_limit_slot,
                    self.mf_slot_counter,
                    self._mf_first_rejected_slot,
                ],
                device=model.device,
            )
            wp.launch(
                populate_rigid_velocity_limit_rows,
                dim=self._free_rigid_body_count,
                inputs=[bind_free_bodies(self), bind_mf_rows(self), self.rigid_velocity_limit_slot],
                device=model.device,
            )
            wp.launch(
                finalize_constraint_counts_with_status,
                dim=self.world_count,
                inputs=[self.mf_slot_counter, self.mf_max_constraints, 1, self._mf_first_rejected_slot],
                outputs=[self.mf_constraint_count, self._constraint_capacity_status],
                device=model.device,
            )

        if self._row_packets is not None:
            self._row_packets.finish_rows()
        wp.launch(
            finalize_constraint_counts_with_status,
            dim=self.world_count,
            inputs=[self.slot_counter, max_constraints, 0, self._dense_first_rejected_slot],
            outputs=[self.constraint_count, self._constraint_capacity_status],
            device=model.device,
        )
        if self._sparse_prepare_contacts:
            wp.launch(
                finalize_sparse_joint_limit_rows,
                dim=self.world_count,
                inputs=[self.dense_phase_bounds, self.constraint_count, dense_rows, dt],
                device=model.device,
            )
        if self._keyed_rows is not None:
            self._keyed_rows.finish_rows()
        # The patch link runs after every row producer: the keyed rows' general worlds take their records last.
        self._link_friction_patches(contacts)
        self._finalize_constraint_status(contacts)
        if self._sparse_diagonal_contact_triples or self._sparse_diagonal_contact_units:
            schedule = self._build_independent_sparse_contact_groups_kernel
            if schedule is None:
                raise RuntimeError("Independent sparse-contact schedule kernel is unavailable")
            # Unit heads come from the row types and the tangent rows' parents.
            unit_rows = [self.row_type, self.row_parent] if self._sparse_diagonal_contact_units else []
            wp.launch(
                schedule,
                dim=self.world_count * 32,
                inputs=[
                    self.constraint_count,
                    self.dense_phase_bounds,
                    *unit_rows,
                ],
                outputs=[
                    self._sparse_diagonal_row_dof,
                    self._sparse_contact_group_count,
                    self._sparse_contact_group_heads,
                    self._sparse_contact_serial_count,
                    self._sparse_contact_serial_normals,
                ],
                device=model.device,
            )
        if self._local_internal_fast_path and self._row_packets is None:
            self._local_general_world_count.zero_()
            wp.launch(
                classify_local_solve_worlds,
                dim=self.world_count,
                inputs=[
                    bind_mf_rows(self),
                    field_group.bind(LocalOwners, solver=self),
                    self.constraint_count,
                    self.dense_phase_bounds,
                    self.mf_constraint_count,
                    self.body_to_articulation,
                    self.articulation_H_rows,
                ],
                device=model.device,
            )
            pair_tiers = (
                (
                    PGS_LOCAL_SOLVE_OWNER_PAIR,
                    self._local_pair_art_counts,
                    self._local_pair_candidates,
                    self._local_pair_secondaries,
                    self._local_pair_active_counts,
                    self._local_pair_active_candidates,
                    self._local_pair_active_secondaries,
                ),
                (
                    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
                    self._local_residual_art_counts,
                    self._local_residual_candidates,
                    self._local_residual_secondaries,
                    self._local_residual_active_counts,
                    self._local_residual_active_candidates,
                    self._local_residual_active_secondaries,
                ),
            )
            for (
                owner,
                counts,
                candidates,
                secondaries,
                active_counts,
                active_candidates,
                active_secondaries,
            ) in pair_tiers:
                for size, active_count in active_counts.items():
                    active_count.zero_()
                    wp.launch(
                        compact_local_pair_candidates,
                        dim=counts[size],
                        inputs=[
                            candidates[size],
                            secondaries[size],
                            self.art_to_world,
                            self._local_solve_owner,
                            owner,
                        ],
                        outputs=[active_count, active_candidates[size], active_secondaries[size]],
                        device=model.device,
                    )

        if self.warn_constraint_overflow:
            mf_raw_counts = self.mf_slot_counter if mf_active else self.slot_counter
            wp.launch(
                _warn_constraint_row_overflow,
                dim=self.world_count,
                inputs=[
                    self.slot_counter,
                    self._row_dropped_dense,
                    self.dense_max_constraints,
                    mf_raw_counts,
                    self._row_dropped_mf,
                    self.mf_max_constraints,
                    1 if mf_active else 0,
                    self._row_overflow_warning_emitted,
                ],
                device=model.device,
            )

    def _stage4_zero_world_C(self):
        self.C.zero_()
        self.diag.zero_()

    def _response_targets(self, size: int) -> tuple:
        """The world-layout J and Y a size's response writes (its group buffers where none is allocated) and the
        DOF offsets it writes them at."""
        J_world = self.J_world if self.J_world is not None else self.J_by_size[size]
        Y_world = self.Y_world if self.Y_world is not None else self.Y_by_size[size]
        offsets = self.articulation_world_dof_offset if self._hinv_jt_writes_world else self.group_to_art[size]
        return J_world, Y_world, offsets

    def _stage4_response(self, size: int) -> None:
        """Build one size's contact response along the plan's stage-4 route (none where an owner publishes it)."""
        route = self._plan.response[size].name
        if route == "fused":
            self._stage4_hinv_jt_tiled_fused(size)
        elif route == "diagonal":
            self._stage4_hinv_jt_diagonal(size)
        elif route == "grouped":
            self._stage4_hinv_jt_grouped(size)
        elif route == "tiles":
            self._stage4_hinv_jt_tiled(size)
        elif route == "lanes":
            self._stage4_hinv_jt_lanes(size)
        elif route == "rows":
            self._stage4_hinv_jt_rows(size)

    def _stage4_delassus(self, size: int) -> None:
        """Assemble one size's Delassus block along the plan's stage-4 route (split mode)."""
        if self._plan.delassus[size].name == "entries":
            self._stage4_delassus_par_row_col(size)
        else:
            self._stage4_delassus_tiled(size)

    def _stage4_hinv_jt_grouped(self, size: int) -> None:
        """The grouped response solve (grouped dynamics), a warp per articulation; verified up to 32 DOFs."""
        n_arts = self.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        diag_out = self.diag_by_size[size] if size in self._hinv_jt_diag_sizes else self._grouped_dummy_diag
        wpb = 4
        wp.launch_tiled(
            self._grouped_kernels_by_size[size],
            dim=[(n_arts + wpb - 1) // wpb],
            inputs=[
                wpb,
                n_arts,
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                world_dof_offset,
                self.constraint_count,
                self.Y_by_size[size],
                J_world,
                Y_world,
                diag_out,
                self._mf_count_or_dummy(),
                int(self._ink_skip_rows()),
            ],
            outputs=[],
            block_dim=32 * wpb,
            device=self.model.device,
        )

    def _stage4_hinv_jt_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        outputs = [self.Y_by_size[size], J_world, Y_world]
        if size in self._hinv_jt_diag_sizes:
            outputs.append(self.diag_by_size[size])
        wp.launch_tiled(
            self._hinv_jt_kernels_by_size[size],
            dim=[n_arts, self._hinv_jt_chunk_count_by_size[size]],
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                world_dof_offset,
                self.constraint_count,
            ],
            outputs=outputs,
            block_dim=self.tile_threads,
            device=model.device,
        )

    def _stage4_hinv_jt_diagonal(self, size: int) -> None:
        """Apply diagonal generalized inertia to the active response rows."""
        n_arts = self.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_diagonal_kernel(),
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                world_dof_offset,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
                int(self._hinv_jt_tiled_writes_group),
                int(self._hinv_jt_writes_world),
                int(size in self._hinv_jt_diag_sizes),
            ],
            outputs=[self.Y_by_size[size], J_world, Y_world, self.diag_by_size[size]],
            device=self.model.device,
        )

    def _stage4_hinv_jt_tiled_fused(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch_tiled(
            self._hinv_jt_fused_kernels_by_size[size],
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                self.row_cfm,
            ],
            outputs=[self.C, self.diag, self.Y_by_size[size]],
            block_dim=self.tile_threads,
            device=model.device,
        )

    def _allocate_packed_rows(self) -> None:
        """The matrix-free sweeps' dense rows, packed by support (:mod:`.packed_rows`): every world's row capacity,
        each row as wide as two of its world's articulations hold; a placeholder without those sweeps."""
        if self._mf_sweeps:
            sizes, worlds = self.articulation_response_dof_count.numpy(), self.art_to_world.numpy()
            offsets = self.articulation_world_dof_offset.numpy()
            # A row's sides follow articulation indices (dense_rows.row_sides); they follow the world's coordinates
            # too, so packed entries ascend with them.
            order = np.lexsort((np.arange(len(worlds)), worlds))
            responding = order[sizes[order] > 0]
            same_world = worlds[responding][1:] == worlds[responding][:-1]
            if np.any(same_world & (offsets[responding][1:] < offsets[responding][:-1])):
                raise RuntimeError("Packed rows need each world's coordinates in articulation order")
            self._packed_support = packed_rows.support_width(sizes, worlds, self.world_count)
            # Two articulations hold every coordinate of the widest world: the rows are laid out at their coordinates.
            self._packed_dense = self._packed_support == self.max_world_dofs
            if self._ink_rows and not self._packed_dense:
                raise RuntimeError("The parallel tiers read their worlds' packed rows at their coordinates")
            sizes = {"world": self.world_count, "row": self.dense_max_constraints, "support": self._packed_support}
        else:
            self._packed_support, self._packed_dense = 1, False
            sizes = {"world": 1, "row": 1, "support": 1}
        self._arrays.allocate(packed_rows.PACKED_ROWS, requires_grad=False, **sizes)

    def _place_dense_response(self) -> None:
        """Where the dense rows' response lives (``_packed_response`` and ``_packed_response_reason``).

        The matrix-free sweeps, and every route beside them, read the dense rows packed by support
        (:mod:`.packed_rows`): the packed build alone serves the step, the response stage builds nothing and the world
        layout and the size groups' responses are not kept.
        """
        self._packed_response_reason = None if self._mf_sweeps else "No matrix-free sweep reads packed rows."
        self._packed_response = self._packed_response_reason is None
        if not self._packed_response:
            if self._paired_factor_coordinates:
                raise RuntimeError(
                    f"The paired factor-coordinate sweep reads packed rows: {self._packed_response_reason}"
                )
            return
        if self._paired_factor_coordinates and not self._packed_dense:
            raise RuntimeError("The paired factor-coordinate sweep reads packed rows at their world coordinates")
        self._hinv_jt_writes_world = False
        self._jy_world_aliased = False
        self._arrays.release(WORLD_RESPONSE)
        placeholder = self._arrays.placeholder(wp.float32, (1, 1, 1))
        for size in self.size_groups:
            self.Y_by_size[size] = placeholder
            self.diag_by_size[size] = self._dummy_hinv_diag
        self._hinv_jt_diag_sizes = frozenset()

    def _row_support(self) -> RowSupport:
        """Where the dense rows' Jacobian entries sit (:class:`.dense_rows.RowSupport`): the matrix-free sweeps' packed
        rows (their sides, the articulations' coordinates, first world coordinates and worlds, and the layout), or
        each size group's rows. Without the matrix-free sweeps there are no packed rows: the articulations' offsets and
        worlds are one-entry placeholders."""
        support = RowSupport()
        support.sides = self._row_sides
        support.dofs = self.articulation_response_dof_count
        support.offset = self.articulation_world_dof_offset if self._mf_sweeps else self._dummy_is_free_rigid
        support.world = self.art_to_world if self._mf_sweeps else self._dummy_is_free_rigid
        support.dense = int(self._packed_dense)
        support.packed = int(self._mf_sweeps)
        return support

    def _jacobian_rows(self, size: int) -> wp.array:
        """The Jacobian rows the producers of a size's rows write (:func:`.dense_rows.jacobian_row`): the packed rows
        (:mod:`.packed_rows`) under the matrix-free sweeps, else the size group's rows."""
        return self._packed_J if self._mf_sweeps else self.J_by_size[size]

    def _stage4_pack_rows(self, *, write_diag: int) -> None:
        """Pack and whiten every world's dense rows and store the general worlds' diagonals as ``write_diag`` asks
        (:func:`.packed_rows.pack_kernel`, one launch; :meth:`_stage4_compute_matrix_free_diag`)."""
        if not self._mf_sweeps:
            return
        # The parallel tiers whiten their worlds' rows in the kernel, and under the row packets the local owners
        # solve their own and the packet producer applies restitution: the build leaves both, as main's response did,
        # unless the warm start applies their impulses through the packed rows. Elsewhere the local owners' worlds
        # are packed: restitution and the drive descriptor read their J v there.
        skip = 0 if self.pgs_warmstart else int(self._ink_skip_rows())
        skip_local = int(self._row_packets is not None and not self.pgs_warmstart)
        # A compact or sparse size keeps a placeholder factor: its rows have no dense factor.
        sizes = tuple(sorted(size for size in self.size_groups if self.L_by_size[size].shape[1] == size))
        wp.launch(
            packed_rows.pack_kernel(sizes),
            dim=self.world_count * self.dense_max_constraints,
            inputs=[
                *(self.L_by_size[size] for size in sizes),
                self._row_sides,
                self.articulation_response_dof_count,
                self.art_group_idx,
                self.articulation_world_dof_offset,
                self.constraint_count,
                self._mf_count_or_dummy(),
                skip,
                self._local_solve_owner,
                skip_local,
                self.row_cfm,
                int(self._packed_dense),
                write_diag,
            ],
            outputs=[self._packed_J, self._packed_Z, self.diag],
            device=self.model.device,
        )

    def _plan_tier_rows(self) -> None:
        """Set the tier sweeps' row budgets and whether the parallel tiers form the response in-kernel.

        ``_ink_rows`` and ``_incremental_rows`` are the parallel and incremental tiers' row budgets (0 without them;
        an admitted parallel tier replaces the incremental one). A parallel tier whitens each world's rows and forms
        their diagonals in the kernel: it is built wherever the model and the tiers fit
        (:meth:`_build_inkernel_response_meta`, :func:`.sweep_routes_tiers.inkernel_response_fits` of the smallest
        tier), and the incremental tier serves elsewhere; ``_ink_reason`` says why or why not
        (``routes.sweep.tiers``). The local packets and the keyed rows read this before the tiers are built.
        """
        tiers = self._mf_sweeps and not self.fuse_joint_velocity_limits and not self._paired_factor_coordinates
        self._ink_rows = self._incremental_rows = 0
        self._ink_sizes, self._ink_reason = None, "No parallel tier is built."
        # The incremental kernel loads the whole row prefix and pools each friction pair's ring of normal loads, so
        # it serves patch friction; the linked parallel tiers pool each friction pair's cone over its patch ring.
        # A parallel tier whitens its worlds' rows in the kernel, where they fit; elsewhere the incremental tier
        # serves as it does without one.
        if tiers and self.mf_gs_parallel_rows > 0 and self._parallel_contact_paths:
            rows = min(
                int(self.dense_max_constraints),
                int(self.mf_gs_parallel_rows),
                128 if self.mf_gs_parallel_matrix_free else 96,
            )
            rows -= rows % 4
            if self._preelim_active:
                self._ink_reason = "Bilateral pre-elimination corrects the packed rows, which the tiers whiten anew."
            elif not sweep_routes_tiers.inkernel_response_fits(
                self.max_world_dofs, min(32, rows), self.mf_gs_parallel_matrix_free
            ):
                self._ink_reason = (
                    f"The response block of the A-based tiers cannot whiten {self.max_world_dofs} world coordinates "
                    "in place; matrix-free tiers (mf_gs_parallel_matrix_free) need no such scratch."
                )
            else:
                self._ink_sizes, self._ink_reason = self._build_inkernel_response_meta()
            if self._ink_sizes is not None:
                self._ink_rows = rows
        if not self._ink_rows and tiers and self.mf_gs_incremental_rows > 0 and self._law_agnostic_contact_paths:
            self._incremental_rows = min(int(self.dense_max_constraints), int(self.mf_gs_incremental_rows), 64)

    def _build_inkernel_response_meta(self) -> tuple[tuple[int, int, int, int] | None, str]:
        """Return the in-kernel response's group sizes and offsets ``(NA, NB, OA, OB)``, or ``None``, and why.

        The parallel tiers form the response of up to two size groups, one articulation of each size per world with
        a primary in every world, at one world DOF offset per size when the response is written in world layout.
        """
        if self.art_to_world is None:
            return None, "No articulation->world map."
        if not (1 <= len(self.size_groups) <= 2):
            return None, f"{len(self.size_groups)} size groups; the in-kernel response takes one or two."
        sizes = sorted((int(s) for s in self.size_groups), reverse=True)
        art_to_world = self.art_to_world.numpy()
        dof_off = self.articulation_world_dof_offset.numpy()
        meta = -np.ones((self.world_count, 4), dtype=np.int32)
        offsets = [0, 0]
        for col, size in enumerate(sizes):
            g2a = self.group_to_art[size].numpy()
            worlds = art_to_world[g2a]
            if np.unique(worlds).size != worlds.size:
                return None, f"A world has several articulations of size {size}."
            # The rows lie at their world coordinates (the packed rows' dense layout).
            off = dof_off[g2a]
            if off.size and np.any(off != off[0]):
                # The register rows use compile-time offsets.
                return None, f"The world DOF offsets of size {size} differ."
            offsets[col] = int(off[0]) if off.size else 0
            meta[worlds, 2 * col] = np.arange(g2a.size, dtype=np.int32)
            meta[worlds, 2 * col + 1] = off
        if np.any(meta[:, 0] < 0):
            return None, "A world has no primary articulation."
        nb = sizes[1] if len(sizes) > 1 else 0
        if offsets[0] + sizes[0] > self.max_world_dofs or (nb and offsets[1] + nb > self.max_world_dofs):
            return None, "The group DOFs exceed the world DOF count."
        self._ink_meta = wp.array(meta.reshape(-1), dtype=wp.int32, device=self.model.device)
        return (sizes[0], nb, offsets[0], offsets[1]), (
            f"The parallel tiers whiten the rows of sizes {sizes} in the kernel."
        )

    def _inkernel_response_inputs(self) -> list:
        """The parallel tiers' in-kernel response operands: the worlds' group indices, the primary and secondary
        groups' factors (the primary's for a model without a secondary), the packed rows' Jacobian and the rows'
        cfm."""
        na, nb = self._ink_sizes[0], self._ink_sizes[1]
        L_b = self.L_by_size[nb] if nb else self.L_by_size[na]
        return [self._ink_meta, self.L_by_size[na], L_b, self._packed_J, self.row_cfm]

    def _admit_parallel_contact_paths(self) -> tuple[bool, str]:
        """Decide whether the parallel tiers and keyed rows may serve this contact law, and why.

        They pool each friction pair's cone over its patch ring and write the patch
        row law, so patch friction honors ``mf_gs_parallel_rows`` and keyed rows as
        point friction does, unless ``FEATHER_PGS_PATCH_PARALLEL_ROWS=0`` keeps it on
        its serial sweeps and ordinary rows. Every tier steps each friction pair
        isotropically, so a sliding pair converges to the serial pair solve's maximal
        dissipation under either law; with per-row steps it did not, which moved
        body-velocity RMS by 0.4-0.9% on two contact-rich patch-friction workloads, and
        converged point tiers sat 3.9 times the serial sweep's p90 error away from its
        fixed point on anymal-D's 33-48-row worlds. Whether the tiers then
        share the serial sweep's whole fixed point depends on the joint velocity
        limits, and the worlds the tiers decline keep the serial sweep and its law
        (:meth:`_parallel_velocity_limit_agreement`). Torsion and compliance keep
        ordinary rows. So does contact regularization, under either law: the tiers
        solve the unregularized law, so the worlds they own would drop
        ``pgs_contact_regularization`` while the serial sweeps apply it.
        """
        if self._regularization_enabled:
            return False, (
                "Contact regularization keeps serial sweeps and ordinary rows: the parallel tiers, which keyed rows "
                "feed, solve the unregularized law, and the serial sweeps apply pgs_contact_regularization."
            )
        if self._point_contact_paths:
            return True, "Point friction: parallel tiers and keyed rows serve it."
        if not self._law_agnostic_contact_paths:
            return False, "Contact torsion and compliance keep serial sweeps and ordinary rows."
        if os.environ.get("FEATHER_PGS_PATCH_PARALLEL_ROWS") == "0":
            return False, (
                "Patch friction ignores mf_gs_parallel_rows and keyed rows and keeps its serial sweeps: "
                "FEATHER_PGS_PATCH_PARALLEL_ROWS=0 opts it out (default on)."
            )
        return True, (
            "Patch friction honors mf_gs_parallel_rows and keyed rows: in the parallel (Jacobi) sweeps each tangent "
            "pair projects on its patch ring's pooled normal load, as point pairs project on their own normal, with "
            "one isotropic step per pair, the serial pair solve's friction law. "
            f"{self._parallel_velocity_limit_agreement()} "
            "FEATHER_PGS_PATCH_PARALLEL_ROWS=0 keeps patch friction on its serial sweeps."
        )

    def _parallel_velocity_limit_agreement(self) -> str:
        """Say whether the parallel tiers share the serial sweep's fixed point at the joint velocity limits.

        Recorded in ``routes.friction.parallel_rows``. The tiers project joint
        velocity-limit rows as accumulated unilateral rows. The worlds they decline
        (more dense rows than the largest tier's, or free-body rows) keep the serial
        sweep, which does so only under the accumulated law
        (``routes.velocity_limits.law``). Under PhysX's stateless clamp, the default,
        tier and serial worlds converge to different velocities wherever a limit
        binds. The tiers serve dense rows only, so free-body limits do not enter.
        """
        if not self.enable_joint_velocity_limits or np.isinf(self.velocity_limit_activation_fraction):
            return "Without joint velocity-limit rows the tiers share the serial sweep's fixed point."
        if self._velocity_limit_sweep_law()[0] == "accumulated":
            return (
                "Both accumulate joint velocity-limit rows, so the tiers share the serial sweep's fixed point; the "
                "worlds they decline keep that sweep."
            )
        return (
            "The tiers accumulate joint velocity-limit rows; the worlds they decline keep the serial sweep, which "
            "clamps them statelessly (routes.velocity_limits.law), so the two fixed points differ wherever a limit "
            "binds."
        )

    def _parallel_velocity_limit_pass(self) -> bool:
        """Whether the parallel tiers end with the last-word velocity-limit pass (``routes.sweep.tiers``).

        ``mf_gs_parallel_linked_velocity_limit_pass`` requests it for the linked (patch)
        tiers; it has rows to push only with joint velocity limits.
        """
        return bool(
            self.mf_gs_parallel_linked_velocity_limit_pass
            and self._friction_anchors_enabled
            and self.enable_joint_velocity_limits
        )

    def _parallel_tier_sweeps(self, tiers) -> tuple[list[int], str, str]:
        """Return each parallel tier's sweep budget, the rule that set it, and why (``routes.sweep.tiers``).

        ``tiers`` lists ``(rows, min_rows)`` per tier. Every tier, under either friction
        law, sweeps :func:`.sweep_routes_tiers.tier_sweeps` of its row bound, at least
        ``mf_gs_parallel_sweeps``. The deprecated ``mf_gs_parallel_linked_sweeps`` (or
        ``FEATHER_PGS_PATCH_PARALLEL_SWEEPS``) still sets the linked (patch) tiers' budget
        where given: a count, or ``mf_gs_parallel_sweeps`` for ``None``. A world stops
        earlier under the sweeps' exit law.
        """
        sweeps = int(self.mf_gs_parallel_sweeps)
        linked = self.mf_gs_parallel_linked_sweeps
        if self._friction_anchors_enabled and linked != "auto":
            source = (
                "FEATHER_PGS_PATCH_PARALLEL_SWEEPS"
                if "mf_gs_parallel_linked_sweeps" in self._environment_options
                else "mf_gs_parallel_linked_sweeps"
            )
            budget = sweeps if linked is None else int(linked)
            reason = (
                f"{source}={linked} (deprecated): every linked tier sweeps at most {budget} times; a world stops "
                "earlier under the sweeps' exit law."
            )
            return [budget] * len(tiers), "mf_gs_parallel_linked_sweeps", reason
        reason = (
            f"auto: a tier serving up to R rows sweeps ceil({sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION} * pgs_iterations * "
            f"sqrt(R / 32)) times with pgs_iterations={int(self.pgs_iterations)}, at least "
            f"mf_gs_parallel_sweeps={sweeps} and at most {sweep_routes_tiers.TIER_SWEEPS_CAP} unless mf_gs_parallel_sweeps is larger; "
            "a world stops earlier under the sweeps' exit law."
        )
        return [sweep_routes_tiers.tier_sweeps(rows, self.pgs_iterations, sweeps) for rows, _ in tiers], "auto", reason

    def _velocity_limit_sweep_law(self) -> tuple[str | None, str]:
        """Return the law the serial matrix-free sweep kernels give velocity-limit rows, and why.

        Recorded as ``routes.velocity_limits.law``; the parallel tiers accumulate joint
        velocity-limit rows whatever this law (``routes.sweep.tiers.velocity_limit_law``).
        Split mode has no joint velocity-limit rows and no matrix-free velocity-limit pass,
        so it applies neither law (``None``); ``routes.velocity_limits.free_body_sweep_law``
        records the law of its free-body sweep, stateless where free-body velocity-limit
        rows can exist and ``None`` where none can (no free rigid bodies, native sparse
        split rows, unregistered speed attributes or an infinite activation fraction). The
        ``physx_grasp`` schedule keeps PhysX's stateless clamp. Without velocity-limit rows
        the kernels keep their stateless build, so a model without limits runs the same
        kernels under either law.
        """
        if self.pgs_mode != "matrix_free":
            # The rule, not this solver's outcome: free_body_sweep_law records whether its sweep has such rows.
            return None, (
                "Split mode has no joint velocity-limit rows and no matrix-free velocity-limit pass, so it applies "
                "neither law. routes.velocity_limits.free_body_sweep_law records the law of its free-body sweep, "
                "stateless where free-body velocity-limit rows can exist, and the post-solve root clamp, where it "
                "runs (routes.velocity_limits.root_clamp), keeps its own."
            )
        if self.pgs_schedule == "physx_grasp":
            return "stateless", "pgs_schedule='physx_grasp' keeps PhysX's stateless velocity clamp."
        rows = self.enable_joint_velocity_limits or (
            self._has_rigid_body_velocity_limits and self._free_rigid_body_count > 0
        )
        if not rows or np.isinf(self.velocity_limit_activation_fraction):
            return "stateless", (
                "No velocity-limit rows can exist: the sweep kernels keep their stateless build under "
                f"velocity_limit_law={self.velocity_limit_law!r}."
            )
        source = (
            "FEATHER_PGS_VELOCITY_LIMIT_LAW"
            if "velocity_limit_law" in self._environment_options
            else "velocity_limit_law"
        )
        return self.velocity_limit_law, f"{source}={self.velocity_limit_law!r}."

    def _velocity_limit_kernel_law(self) -> str:
        """The ``velocity_limit_law`` argument of every matrix-free sweep kernel this solver builds."""
        return self._velocity_limit_sweep_law()[0] or "stateless"

    def _mf_count_or_dummy(self):
        """Matrix-free row counts, or a one-element dummy when the model has no contact rows (never indexed then)."""
        arr = getattr(self, "mf_constraint_count", None)
        return arr if arr is not None else self._dummy_is_free_rigid

    def _ink_skip_rows(self) -> int:
        """Dense-row budget of the in-kernel-response sweep kernels (0 when they are not active)."""
        if getattr(self, "_ink_sizes", None) is None:
            return 0
        return int(getattr(self, "_ink_rows", 0))

    def _stage4_hinv_jt_lanes(self, size: int) -> None:
        """The response a warp per articulation, skipping the worlds the local owners solve (route ``lanes``)."""
        n_arts = self.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_lanes_kernel(),
            dim=n_arts * 32,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                world_dof_offset,
                self.constraint_count,
                self._local_solve_owner,
                self.row_restitution,
                size,
                self.dense_max_constraints,
                n_arts,
                int(self._hinv_jt_writes_world),
            ],
            outputs=[self.Y_by_size[size], J_world, Y_world],
            device=self.model.device,
            block_dim=256,
        )

    def _stage4_hinv_jt_rows(self, size: int) -> None:
        """The response a thread per row (route ``rows``)."""
        n_arts = self.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_rows_kernel(),
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                world_dof_offset,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
                int(self._hinv_jt_writes_world),
            ],
            outputs=[self.Y_by_size[size], J_world, Y_world],
            device=self.model.device,
        )

    def _stage4_delassus_par_row_col(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.delassus_entries_kernel(),
            dim=n_arts * self.dense_max_constraints * self.dense_max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
            ],
            outputs=[self.C, self.diag],
            device=model.device,
        )

    def _stage4_delassus_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch_tiled(
            self._delassus_kernels_by_size[size],
            dim=[n_arts],
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                n_arts,
            ],
            outputs=[self.C, self.diag],
            block_dim=response_builds_dense.DELASSUS_THREADS,
            device=model.device,
        )

    def _stage4_finalize_world_diag_cfm(self):
        if self._packed_response:
            return  # The packed rows' build adds CFM with the diagonal.
        if self._sparse_prepare_contacts:
            return  # Contact and limit producers add CFM exactly once.
        if self._compact_contact_boundary:
            return  # Both complete owners already add the original row CFM exactly once.
        model = self.model
        wp.launch(
            response_builds_dense.cfm_worlds_kernel(),
            dim=self.world_count,
            inputs=[self.constraint_count, self.row_cfm, self._mf_count_or_dummy(), int(self._ink_skip_rows())],
            outputs=[self.diag],
            device=model.device,
        )

    def _stage4_compute_physx_drive_desc(
        self,
        dt: float,
        *,
        position_bias_scale: float = 1.0,
        position_delta_velocity=None,
        position_delta_scale: float = 0.0,
    ):
        if self.drive_mode != "physx_pgs":
            return

        if position_delta_velocity is None:
            position_delta_velocity = self.v_hat

        wp.launch(
            compute_physx_pgs_drive_desc,
            dim=self.world_count,
            inputs=[
                bind_dense_rows(self),
                self.constraint_count,
                self.dense_max_constraints,
                self.world_dof_indices,
                self._row_support(),
                self._packed_J,
                position_delta_velocity,
                self.drive_stiffness,
                self.drive_damping,
                self.drive_geom_error,
                self.drive_max_force,
                dt,
                position_bias_scale,
                position_delta_scale,
            ],
            outputs=[
                self.drive_target_vel_bias,
                self.drive_vel_multiplier,
                self.drive_impulse_multiplier,
                self.drive_max_impulse,
            ],
            device=self.model.device,
        )

    def _stage4_diag_from_JY(self, size: int):
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.diagonal_rows_kernel(),
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
                self._mf_count_or_dummy(),
                int(self._ink_skip_rows()),
            ],
            outputs=[self.diag],
            device=self.model.device,
        )

    def _stage4_accumulate_hinv_diag(self, size: int):
        wp.launch(
            response_builds_dense.diagonal_groups_kernel(),
            dim=self.world_count * self.dense_max_constraints,
            inputs=[
                self.diag_by_size[size],
                self.world_response_group_art_start[size],
                self.world_response_group_to_art[size],
                self.art_group_idx,
                self.constraint_count,
                self.dense_max_constraints,
                self._mf_count_or_dummy(),
                int(self._ink_skip_rows()),
            ],
            outputs=[self.diag],
            device=self.model.device,
        )

    def _stage4_compute_matrix_free_diag(self) -> int:
        """The dense rows' diagonals of the matrix-free route, by their owner; what the packed rows' build owes
        (:meth:`_stage4_pack_rows`, :func:`.packed_rows.pack_kernel`'s ``write_diag``): 0 nothing, 1 ``|Z|^2``, 2
        ``|Z|^2 + cfm``."""
        if self._packed_response:
            # The packed rows' build stores |Z|^2 + cfm in the general worlds; local row packets keep their worlds'
            # private seeds.
            if self._row_packets is None:
                self.diag.zero_()
            return 2
        if self._compact_contact_boundary:
            wp.launch(
                response_builds_rows.sparse_pair_limit_kernel,
                dim=(self.world_count, 12),
                inputs=[
                    self.dense_phase_bounds,
                    self.constraint_count,
                    self._sparse_diagonal_dense_groups,
                    self.L_by_size[6],
                    self.J_by_size[6],
                    self.Y_by_size[6],
                    self.row_cfm,
                    self.diag,
                ],
                device=self.model.device,
            )
            return 0
        if self._sparse_mass_matrix_size is not None:
            # Sparse producers write the diagonal; unfused controls add CFM separately.
            return 0
        self.diag.zero_()
        if self._sparse_diagonal_contact_solve:
            for size in self.size_groups:
                if size != self._sparse_diagonal_response_size:
                    self._stage4_diag_from_JY(size)
            wp.launch(
                response_builds_rows.sparse_pair_diagonal_kernel,
                dim=self.world_count * self.dense_max_constraints,
                inputs=[
                    self.constraint_count,
                    self.dense_max_constraints,
                    self._sparse_diagonal_row_dof,
                    self._sparse_diagonal_row_jy,
                ],
                outputs=[self.diag],
                device=self.model.device,
            )
            return 0
        if self._hinv_jt_diag_sizes:
            for size in self.size_groups:
                if size in self._hinv_jt_diag_sizes:
                    self._stage4_accumulate_hinv_diag(size)
                else:
                    self._stage4_diag_from_JY(size)
        elif self._hinv_jt_writes_world:
            # The packed rows' build stores the diagonal (_stage4_pack_rows); the CFM pass adds CFM after.
            return 1
        else:
            for size in self.size_groups:
                self._stage4_diag_from_JY(size)
        if self._local_internal_fast_path and not self._hinv_jt_writes_world:
            wp.launch(
                clear_local_solve_diag,
                dim=self.world_count * self.dense_max_constraints,
                inputs=[self.constraint_count, self._local_solve_owner, self.dense_max_constraints],
                outputs=[self.diag],
                device=self.model.device,
            )
        return 0

    def _stage4_compute_rhs_world(
        self,
        dt: float,
        *,
        bias_scale: float = 1.0,
        contact_speculative_scale: float = 1.0,
        preserve_unreached_speculative: bool = False,
        joint_limit_speculative_scale: float = 1.0,
        output=None,
    ):
        """Each dense row's rhs bias (``compute_world_contact_bias``); ``preserve_unreached_speculative`` is the velocity
        pass, which also applies restitution."""
        if self._sparse_prepare_contacts:
            return  # The admitted sparse path has a single position pass.
        if self._row_packets is not None or self._keyed_rows is not None:
            if (
                output is not None
                or preserve_unreached_speculative
                or bias_scale != 1.0
                or joint_limit_speculative_scale != 1.0
                or contact_speculative_scale != self.contact_speculative_scale
            ):
                raise RuntimeError("Private rows require the original single position pass")
            return
        velocity = bool(preserve_unreached_speculative)
        wp.launch(
            compute_world_contact_bias,
            dim=self.world_count * self.dense_max_constraints,
            inputs=[
                bind_dense_rows(self) if output is None else bind_dense_rows(self, rhs=output),
                self.constraint_count,
                self.dense_max_constraints,
                dt,
                bias_scale,
                contact_speculative_scale,
                joint_limit_speculative_scale,
                1.0 if velocity else self._contact_w,
                int(velocity),
                # The velocity pass's packed rows (matrix-free); the position pass reads none of them.
                self.world_dof_indices if velocity else self._dummy_contact_row_type,
                self._row_support(),
                self._packed_J,
                self.v_out_snap if velocity else self.v_hat,
                self.v_hat,
                self._effective_restitution_velocity_threshold,
            ],
            device=self.model.device,
        )

    def _stage4_accumulate_rhs_world(self, size: int, velocity=None):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        if velocity is None:
            velocity = self.v_hat
        parallel_rows = model.device.is_cuda and not model.requires_grad and self._is_one_solve_art_per_world
        wp.launch(
            rhs_accum_world_par_row if parallel_rows else rhs_accum_world_par_art,
            dim=n_arts * self.dense_max_constraints if parallel_rows else n_arts,
            inputs=[
                self.constraint_count,
                self.dense_max_constraints,
                self.art_to_world,
                self.articulation_dof_start,
                velocity,
                self.group_to_art[size],
                self.J_by_size[size],
                size,
            ],
            outputs=[self.rhs],
            device=model.device,
        )

    def _stage4_apply_world_contact_restitution(self, dt: float, *, matrix_free: bool) -> None:
        """Replace geometric contact bias for impacts selected from ``v_hat``."""
        if self._sparse_prepare_contacts:
            return  # The sparse contact producer uses its projected incident velocity.
        if self._row_packets is not None or self._keyed_rows is not None:
            return  # The current packet producer applies the same incident-velocity law.
        if matrix_free:
            sparse = self._sparse_diagonal_contact_solve
            dense_size = self._sparse_diagonal_dense_size if sparse else 0
            wp.launch(
                apply_world_contact_restitution_matrix_free,
                dim=self.world_count * self.dense_max_constraints,
                inputs=[
                    bind_dense_rows(self),
                    self.constraint_count,
                    self.dense_max_constraints,
                    self.v_hat,
                    self.world_dof_indices,
                    self._row_support(),
                    self._packed_J,
                    int(sparse),
                    self._sparse_diagonal_dense_offsets,
                    self._sparse_diagonal_dense_groups,
                    dense_size,
                    self.J_by_size[dense_size] if sparse else self._dummy_world_J,
                    self._sparse_diagonal_row_dof,
                    self._sparse_diagonal_row_jy,
                    dt,
                    self._effective_restitution_velocity_threshold,
                    int(self._regularization_enabled),
                ],
                device=self.model.device,
            )
            return

        wp.launch(
            apply_world_contact_restitution_accumulated,
            dim=self.world_count * self.dense_max_constraints,
            inputs=[
                bind_dense_rows(self),
                self.constraint_count,
                self.dense_max_constraints,
                dt,
                self.contact_speculative_scale,
                self._effective_restitution_velocity_threshold,
                int(self._regularization_enabled),
            ],
            device=self.model.device,
        )

    def _stage5_prepare_impulses_world(self):
        warmstart_flag = 1 if self.pgs_warmstart else 0
        wp.launch(
            prepare_world_impulses,
            dim=self.world_count,
            inputs=[
                self.constraint_count,
                self.dense_max_constraints,
                warmstart_flag,
            ],
            outputs=[self.impulses],
            device=self.model.device,
        )

    def _dispatch_dense_pgs_solve(
        self,
        iterations: int,
        friction_start_iteration: int | None = None,
        iteration_offset: int = 0,
    ):
        """Dispatch the dense PGS kernel with a given iteration count."""
        if friction_start_iteration is None:
            friction_start_iteration = self._contact_friction_start_iteration(iterations)
        sweep = (
            self._stage5_pgs_solve_world_tiled_row
            if self.pgs_kernel == "tiled_row"
            else self._stage5_pgs_solve_world_loop
        )
        sweep(iterations, int(friction_start_iteration), int(iteration_offset))

    def _stage5_pgs_solve_world_tiled_row(self, iterations: int, friction_start_iteration: int, iteration_offset: int):
        pgs_kernel = self._pgs_solve_tiled_row_kernel
        if pgs_kernel is None:
            raise RuntimeError("Tiled row PGS kernel is unavailable for this solver shape")
        for kernel in self._pgs_solve_tiled_row_budgets or (pgs_kernel,):
            wp.launch_tiled(
                kernel,
                dim=[self.world_count],
                inputs=[
                    self.constraint_count,
                    self.diag,
                    self.C,
                    self.rhs,
                    self.impulses,
                    iterations,
                    self.pgs_omega,
                    self.row_type,
                    self.row_parent,
                    self.row_mu,
                    self.row_w,
                    int(self._regularization_enabled),
                    friction_start_iteration,
                    iteration_offset,
                    *self._solve_readout.operands(kernel),
                ],
                block_dim=32,
                device=self.model.device,
            )

    def _stage5_pgs_solve_world_loop(self, iterations: int, friction_start_iteration: int, iteration_offset: int):
        # The unit loop sweep (contact_update="unit") also reads the patch form and the per-world unit storage.
        units = []
        if self._dense_contact_unit:
            units = [
                int(self._contact_unit_patch == "ring"),
                self._dense_unit_state,
                self._dense_unit_rows,
                self._dense_unit_block,
            ]
        wp.launch(
            self._pgs_solve_loop_kernel,
            dim=self.world_count,
            inputs=[
                self.constraint_count,
                self.diag,
                self.C,
                self.rhs,
                self.impulses,
                iterations,
                self.pgs_omega,
                self.row_type,
                self.row_parent,
                self.row_mu,
                self.row_w,
                int(self._regularization_enabled),
                friction_start_iteration,
                iteration_offset,
                *units,
                *self._solve_readout.operands(self._pgs_solve_loop_kernel),
            ],
            device=self.model.device,
        )

    def _stage6_prepare_world_velocity(self):
        wp.copy(self.v_out, self.v_hat)

    def _build_mf_body_map(self) -> None:
        """Build the per-world free-body map shared by MF warm start and PGS."""
        wp.launch(
            response_builds_rows.free_body_map_kernel,
            dim=self.world_count,
            inputs=[
                bind_mf_rows(self),
                bind_mf_body_map(self),
                self.mf_constraint_count,
                self.body_to_articulation,
                self.articulation_dof_start,
                self.max_mf_bodies,
            ],
            device=self.model.device,
        )

    def _apply_mf_warmstart_velocity(
        self,
        velocity: wp.array,
        *,
        set_output_to_delta: bool,
        body_map_ready: bool = False,
    ) -> None:
        """Install cached MF impulse response into a live velocity state."""
        if not self._mf_warmstart_enabled or not self._has_free_rigid_bodies:
            return
        if not body_map_ready:
            self._build_mf_body_map()
        wp.launch(
            response_builds_rows.free_body_publish_kernel,
            dim=self.world_count * self.max_mf_bodies,
            inputs=[
                bind_mf_body_map(self),
                self.mf_constraint_count,
                self.mf_Z_a,
                self.mf_Z_b,
                self.mf_row_type,
                self.mf_impulses,
                self._body_factors(),
                self.max_mf_bodies,
                int(set_output_to_delta),
            ],
            outputs=[velocity],
            device=self.model.device,
        )

    def _stage6_clamp_rigid_velocity_limits(self):
        """Bound the solved speeds of every free root in split mode, where its step ends (the root clamp).

        Split mode bounds free rigid bodies inside its free-body sweep with speed rows where the activation
        gate reserved them; this clamp of the solved velocity bounds every free root, floating bases included,
        without changing a predicted velocity the contacts still act on. Matrix-free mode bounds free rigid
        bodies with speed rows alone.
        """
        if not self._has_root_free or not self._has_rigid_body_velocity_limits:
            return
        if self.pgs_mode == "matrix_free":
            return

        wp.launch(
            clamp_free_root_velocity_limits,
            dim=self.model.articulation_count,
            inputs=[
                self.model.articulation_start,
                self.model.joint_child,
                self.model.body_flags,
                self.articulation_root_is_free,
                self.articulation_root_dof_start,
                self.rigid_body_max_linear_velocity,
                self.rigid_body_max_angular_velocity,
            ],
            outputs=[self.v_out],
            device=self.model.device,
        )

    def _stage6_apply_packed_impulses(self) -> None:
        """``v_out = v_hat + P^T L^-T sum_r Z_r lambda_r`` of every articulation with a dense factor, from the packed
        rows (:func:`.packed_rows.apply_kernel`), a launch per size group."""
        for size in self.size_groups:
            if self.L_by_size[size].shape[1] != size:
                continue  # A compact or sparse size keeps a placeholder: its rows have no dense factor.
            wp.launch(
                packed_rows.apply_kernel(),
                dim=self.n_arts_by_size[size],
                inputs=[
                    self.L_by_size[size],
                    self.group_to_art[size],
                    self.art_to_world,
                    self.articulation_dof_start,
                    self.articulation_response_dof_count,
                    size,
                    self._row_sides,
                    self.constraint_count,
                    self.articulation_world_dof_offset,
                    int(self._packed_dense),
                    self._packed_Z,
                    self.impulses,
                    self.v_hat,
                ],
                outputs=[self.v_out],
                device=self.model.device,
            )

    def _stage6_apply_impulses_world(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.apply_coordinates_kernel(),
            dim=int(n_arts * size),
            inputs=[
                self.group_to_art[size],
                self.art_to_world,
                self.articulation_dof_start,
                size,
                n_arts,
                self.constraint_count,
                self.dense_max_constraints,
                self.Y_by_size[size],
                self.impulses,
                self.v_hat,
            ],
            outputs=[self.v_out],
            device=model.device,
        )

    def _stage6b_mf_pgs(self, state_aug: State, dt: float):
        """Run matrix-free PGS for free rigid body contacts."""
        self._mf_pgs_setup(state_aug, dt)
        self._build_mf_body_map()
        self._apply_mf_warmstart_velocity(
            self.v_out,
            set_output_to_delta=False,
            body_map_ready=True,
        )
        self._mf_pgs_solve(
            self.pgs_iterations,
            friction_start_iteration=self._contact_friction_start_iteration(self.pgs_iterations),
            body_map_ready=True,
        )

    def _mf_pgs_setup(self, state_aug: State, dt: float):
        """MF PGS setup: whiten the free-body rows through each body's held factor, their diagonals and RHS."""
        model = self.model
        self.mf_rhs.zero_()
        self.mf_eff_mass_inv.zero_()
        wp.launch(
            response_builds_rows.free_body_rows_kernel,
            dim=self.world_count * self.mf_max_constraints,
            inputs=[
                bind_mf_rows(self),
                self.mf_constraint_count,
                self._body_factors(),
                int(self._has_prescribed_response),
                self.v_hat,
                self.rigid_body_max_depenetration_velocity,
                self.pgs_cfm,
                self.pgs_beta,
                self._contact_w,
                dt,
                self.contact_speculative_scale,
                self._effective_restitution_velocity_threshold,
                self.mf_max_constraints,
            ],
            outputs=[self.mf_eff_mass_inv, self.mf_row_w],
            device=model.device,
        )

    def _build_body_factors(self) -> BodyFactors:
        """Each free body's held factor and coordinates (:class:`.kernels.BodyFactors`): the dynamics' size-6 factors
        ``L_by_size[6]`` (armature included), which whiten its rows and map its velocity into factor coordinates, and
        the lookup of a body's factor by its first global coordinate. Built once with the buffers: a captured step
        allocates and copies nothing."""
        device = self.model.device
        dof_factor = np.full(max(int(self.model.joint_dof_count), 1), -1, dtype=np.int32)
        if self._has_free_rigid_bodies:
            free = np.flatnonzero(self._model_plan.is_free_rigid != 0)
            factor_index = self.art_group_idx.numpy()[free]
            starts = self.articulation_dof_start.numpy()[free]
            held = factor_index >= 0
            dof_factor[starts[held]] = factor_index[held]
        factors = BodyFactors()
        factors.body_to_articulation = getattr(self, "body_to_articulation", None)
        factors.factor_index = getattr(self, "art_group_idx", None)
        factors.dof_start = getattr(self, "articulation_dof_start", None)
        factors.dof_factor = wp.array(dof_factor, dtype=wp.int32, device=device)
        lower = self.L_by_size.get(6) if self._has_free_rigid_bodies else None
        factors.L = lower if lower is not None else wp.zeros((1, 6, 6), dtype=wp.float32, device=device)
        # Every articulation's dense factor, by address: the size groups' factors are allocated once, with the buffers.
        art_count = max(int(self.model.articulation_count), 1)
        first_art = np.full(max(int(self.model.joint_dof_count), 1), -1, dtype=np.int32)
        sizes = np.zeros(art_count, dtype=np.int32)
        offsets = np.zeros(art_count, dtype=np.int32)
        addresses = np.zeros(art_count, dtype=np.uint64)
        if int(self.model.articulation_count) > 0:
            starts = self.articulation_dof_start.numpy()
            counts = self.articulation_response_dof_count.numpy()
            world_offsets = getattr(self, "articulation_world_dof_offset", None)
            offsets[:] = world_offsets.numpy() if world_offsets is not None else 0
            sizes[:] = counts
            responding = np.flatnonzero(counts > 0)
            first_art[starts[responding]] = responding
            for size, lower_group in self.L_by_size.items():
                if lower_group.shape[1] != size or size not in self.group_to_art:
                    continue  # A compact or sparse size keeps a placeholder: no dense factor.
                arts = self.group_to_art[size].numpy()
                addresses[arts] = lower_group.ptr + np.arange(len(arts), dtype=np.uint64) * np.uint64(size * size * 4)
        factors.first_art = wp.array(first_art, dtype=wp.int32, device=device)
        factors.size = wp.array(sizes, dtype=wp.int32, device=device)
        factors.world_offset = wp.array(offsets, dtype=wp.int32, device=device)
        factors.address = wp.array(addresses, dtype=wp.uint64, device=device)
        return factors

    def _body_factors(self) -> BodyFactors:
        """The free bodies' factors and coordinates (:meth:`_build_body_factors`)."""
        return self._body_factors_data

    def _compute_mf_rhs_bias(
        self,
        dt: float,
        *,
        bias_scale: float,
        speculative_scale: float = 1.0,
        preserve_unreached_speculative: bool = False,
        apply_restitution: bool = False,
        output: wp.array,
    ):
        """Recompute MF contact RHS for the current rows without touching effective mass."""
        wp.launch(
            compute_mf_rhs_bias,
            dim=self.world_count * self.mf_max_constraints,
            inputs=[
                bind_mf_rows(self, rhs=output),
                self.mf_constraint_count,
                self._body_factors(),
                int(self._has_prescribed_response),
                self.rigid_body_max_depenetration_velocity,
                self.pgs_beta,
                dt,
                bias_scale,
                speculative_scale,
                self.v_out_snap,
                int(preserve_unreached_speculative),
                int(apply_restitution),
                self._effective_restitution_velocity_threshold,
                self.mf_max_constraints,
            ],
            device=self.model.device,
        )

    def _mf_pgs_solve(
        self,
        iterations: int,
        friction_start_iteration: int | None = None,
        iteration_offset: int = 0,
        body_map_ready: bool = False,
    ):
        """MF PGS solve with given iteration count."""
        model = self.model
        if friction_start_iteration is None:
            friction_start_iteration = self._contact_friction_start_iteration(iterations)

        if not body_map_ready:
            self._build_mf_body_map()

        if model.device.is_cuda:
            mf_pgs_kernel = self._pgs_solve_mf_kernel
            if mf_pgs_kernel is None:
                raise RuntimeError("Standalone matrix-free PGS kernel is unavailable for this solver shape")
            wp.launch_tiled(
                mf_pgs_kernel,
                dim=[self.world_count],
                inputs=[
                    self.mf_constraint_count,
                    self.mf_body_count,
                    self.mf_body_dof_start,
                    self.mf_local_body_a,
                    self.mf_local_body_b,
                    self.mf_Z_a,
                    self.mf_Z_b,
                    self._body_factors(),
                    self.mf_eff_mass_inv,
                    self.mf_rhs,
                    self.mf_row_type,
                    self.mf_row_parent,
                    self.mf_row_mu,
                    self.mf_row_w,
                    self.mf_impulses,
                    self.v_out,
                    iterations,
                    self.pgs_omega,
                    int(self._regularization_enabled),
                    int(friction_start_iteration),
                    int(iteration_offset),
                    *self._solve_readout.operands(mf_pgs_kernel),
                ],
                block_dim=32,
                device=model.device,
            )
        else:
            # The CPU's free-body rows: the loop sweep, whose unit variant also reads the patch form and unit storage.
            units = []
            if self._free_body_contact_unit:
                units = [
                    int(self._contact_unit_patch == "ring"),
                    self._mf_unit_state,
                    self._mf_unit_rows,
                    self._mf_unit_block,
                ]
            wp.launch(
                self._pgs_solve_mf_loop_kernel,
                dim=self.world_count,
                inputs=[
                    self.mf_constraint_count,
                    self.mf_body_a,
                    self.mf_body_b,
                    self.mf_Z_a,
                    self.mf_Z_b,
                    self._body_factors(),
                    self.mf_eff_mass_inv,
                    self.mf_rhs,
                    self.mf_row_type,
                    self.mf_row_parent,
                    self.mf_row_mu,
                    self.mf_row_w,
                    self.body_to_articulation,
                    self.articulation_dof_start,
                    iterations,
                    self.pgs_omega,
                    int(self._regularization_enabled),
                    int(friction_start_iteration),
                    int(iteration_offset),
                    *units,
                ],
                outputs=[
                    self.mf_impulses,
                    self.v_out,
                    *self._solve_readout.operands(self._pgs_solve_mf_loop_kernel),
                ],
                device=model.device,
            )

    def _remove_free_root_transport(self, state_in: State, state_aug: State):
        """Invert jcalc_integrate's free-root transport so it realizes the solved velocity exactly.

        Runs after every velocity-to-acceleration conversion; no-op (no
        launch) for fixed-base models.
        """
        if not self._free_root_joint_count:
            return
        model = self.model
        wp.launch(
            remove_free_root_transport_from_qdd,
            dim=self._free_root_joint_count,
            inputs=[
                self._free_root_joint_indices,
                model.joint_qd_start,
                self._kinematic_joint_mask,
                state_in.joint_qd,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage6_update_qdd(self, state_in: State, state_aug: State, dt: float):
        if getattr(self, "_tree_kinetic_state", None) is not None:
            return  # finish owns conversion, free-root transport and integration
        model = self.model
        wp.launch(
            update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, self._kinematic_dof_mask, 1.0 / dt],
            outputs=[self.v_out, state_aug.joint_qdd],
            device=model.device,
        )
        self._remove_free_root_transport(state_in, state_aug)

    def _stage6_integrate(self, state_in: State, state_aug: State, state_out: State, dt: float):
        model = self.model
        if getattr(self, "_tree_kinetic_state", None) is not None:
            next_refresh = ((self._step + 1) % self.update_mass_matrix_interval) == 0
            self._tree_kinetic_state.finish(state_in, state_aug, state_out, dt, next_refresh)
            self._fk_id_cache_source_state = state_out
            return

        if model.joint_count:
            wp.launch(
                kernel=integrate_generalized_joints,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    self._kinematic_joint_mask,
                    model.joint_dof_dim,
                    model.body_com,
                    model.joint_X_c,
                    state_in.joint_q,
                    state_in.joint_qd,
                    state_aug.joint_qdd,
                    dt,
                    self.angular_damping,
                ],
                outputs=[state_out.joint_q, state_out.joint_qd],
                device=model.device,
            )

            self._stage7_update_kinematics(state_out, state_aug)

        self.integrate_particles(model, state_in, state_out, dt)

    def _stage6_integrate_position_from_velocity(
        self,
        state_in: State,
        state_aug: State,
        state_out: State,
        dt: float,
        velocity_internal: wp.array,
    ):
        """Integrate positions once from an internal-coordinate velocity without consuming it."""
        model = self.model

        if model.joint_count:
            wp.copy(self.qd_work, velocity_internal)

            wp.launch(
                update_qdd_from_velocity,
                dim=model.joint_dof_count,
                inputs=[state_in.joint_qd, self._kinematic_dof_mask, 1.0 / dt],
                outputs=[self.qd_work, state_aug.joint_qdd],
                device=model.device,
            )
            self._remove_free_root_transport(state_in, state_aug)
            wp.launch(
                kernel=integrate_generalized_joints,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    self._kinematic_joint_mask,
                    model.joint_dof_dim,
                    model.body_com,
                    model.joint_X_c,
                    state_in.joint_q,
                    state_in.joint_qd,
                    state_aug.joint_qdd,
                    dt,
                    self.angular_damping,
                ],
                outputs=[state_out.joint_q, state_out.joint_qd],
                device=model.device,
            )
            # _stage6_write_final_velocity replaces joint_qd and evaluates FK
            # once, so evaluating the temporary position velocity here is wasteful.

        self.integrate_particles(model, state_in, state_out, dt)

    def _stage6_write_final_velocity(self, state_in: State, state_aug: State, state_out: State, dt: float):
        """Write the post-position velocity solve result without integrating positions again."""
        model = self.model

        if model.joint_count:
            wp.launch(
                update_qdd_from_velocity,
                dim=model.joint_dof_count,
                inputs=[state_in.joint_qd, self._kinematic_dof_mask, 1.0 / dt],
                outputs=[self.v_out, state_aug.joint_qdd],
                device=model.device,
            )
            self._remove_free_root_transport(state_in, state_aug)
            wp.copy(state_out.joint_qd, self.v_out)
            self._stage7_update_kinematics(state_out, state_aug)

    def _stage7_update_kinematics(self, state_out: State, state_aug: State) -> None:
        """Publish the output state's kinematics (ordinary dynamics derive the next step's from the state)."""
        eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)

    def __del__(self):
        """Drain queued work without accessing cyclically finalized streams."""
        if sys.is_finalizing():
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            # Cyclic GC may finalize a stream before its owning solver. Its
            # Python wrapper can still contain the destroyed native handle.
            # Synchronizing the context avoids dereferencing that handle and
            # drains all solver work before the remaining buffers are released.
            wp.synchronize_device(model.device)
        except (AttributeError, RuntimeError, TypeError):
            # CUDA/Python callables may already be unavailable at shutdown.
            pass


_JOINT_LIMIT_WARP_PARAMS = kernel_template.params(
    ("articulation_count", int),
    ("articulation_dof_start", wp.array[int]),
    ("art_to_world", wp.array[int]),
    ("group_to_art", wp.array[int]),
    ("resolved_worlds", wp.array[int]),
    ("law", prefix_rows.PrefixLaw),
    ("support", RowSupport),
    ("J_group", wp.array3d[float]),
)
"""The warp joint-limit row builder's operands: the rows' placement and Jacobian rows (:func:`.dense_rows.jacobian_row`:
the packed rows, by world, or the size group's)."""


@cache
def _get_joint_limit_warp_kernel(size: int, warps_per_block: int) -> "wp.Kernel":
    """Build the general route's joint-limit rows, one warp per articulation (:func:`.prefix_rows.prefix_rows`)."""
    # A row's side: its articulation's coordinates in the packed rows (cleared, then the coefficient), or the size
    # group's row (cleared each step).
    sink = f"""            float* J_side = &J_group.data[((size_t)row_index * J_group.shape[1] + slot) * J_group.shape[2] + first];
            if (support.packed != 0) {{
                for (int e = 0; e < {size}; ++e) J_side[e] = 0.0f;
            }}
            J_side[d0] = w0;"""
    rows = prefix_rows.prefix_rows(
        ((prefix_rows.limits(tuple(range(size)), "first_dof + d", "law."),),),
        law="law.",
        sink=sink,
        ordered=False,
        rhs=False,
        side="art",
    )
    snippet = f"""    const int group = block * {warps_per_block} + SUB;
    if (group >= articulation_count) return;
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    if (world >= 0 && world < resolved_worlds.shape[0] && resolved_worlds.data[world] != 0) return;
    const int first_dof = articulation_dof_start.data[art];
    // The rows' first index and the side's first entry (dense_rows.jacobian_row; one side).
    const int row_index = support.packed != 0 ? world : group;
    const int first = support.packed != 0 && support.dense != 0 ? support.offset.data[art] : 0;
{rows}#endif
"""
    return KernelTemplate(
        f"build_joint_limit_rows_warp_{size}_{warps_per_block}",
        _JOINT_LIMIT_WARP_PARAMS,
        snippet,
        preludes=(lane_group(32), *ROW_LAW.preludes()),
        native_name="joint_limit_warp_native",
    ).kernel()


@cache
def _get_triangular_solve_kernel(n_dofs: int, device_arch: str, tile_threads: int = 64) -> "wp.Kernel":
    """Build specialized triangular solve kernel for given DOF count.

    Solves H x = b with the leaf-first factor ``P H P^T = L L^T`` by tiled forward and backward substitution, the
    coordinates reversed by indexed loads and stores.
    """
    TILE_DOF_LOCAL = wp.constant(int(n_dofs))
    TILE_DOF_LAST = wp.constant(int(n_dofs) - 1)

    def trisolve_tiled_template(
        L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
        tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
        qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
    ):
        idx = wp.tid()
        reverse = wp.tile_arange(TILE_DOF_LAST, -1, -1, dtype=int)
        L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
        tau_tile = wp.tile_load_indexed(tau_group[idx], indices=reverse, shape=(TILE_DOF_LOCAL, 1), axis=0)

        # Forward substitution: L * z = P tau
        z_tile = wp.tile_lower_solve(L_tile, tau_tile)

        # Backward substitution: L^T * (P qdd) = z
        Lt_tile = wp.tile_transpose(L_tile)
        qdd_tile = wp.tile_upper_solve(Lt_tile, z_tile)

        wp.tile_store_indexed(qdd_group[idx], indices=reverse, t=qdd_tile, axis=0)

    return python_kernel(trisolve_tiled_template, f"trisolve_tiled_{n_dofs}_bd{tile_threads}")


@cache
def _get_mark_independent_sparse_contact_candidates_kernel(
    sparse_response_dofs: int,
    device_arch: str,
    *,
    units: bool = False,
) -> "wp.Kernel":
    """Mark contact triples whose response is confined to one sparse coordinate.

    ``units`` marks patch contact units instead: a normal row, followed by its two
    tangent rows only when the contact owns a friction anchor.
    """
    del device_arch
    sparse_size = int(sparse_response_dofs)
    if sparse_size <= 0:
        raise ValueError("sparse response DOFs must be positive")

    def mark_independent_sparse_contact_units_template(
        data: ContactInput,
        total_num_workers: int,
        sparse_row_dof: wp.array3d[int],
    ):
        worker = wp.tid()
        total_contacts = data.count[0]
        for contact in range(worker, total_contacts, total_num_workers):
            slot = data.slot[contact]
            rows = data.slots_needed[contact]
            if data.path[contact] != 0 or slot < 0 or (rows != 1 and rows != 3):
                continue

            art_a = data.art0[contact]
            art_b = data.art1[contact]
            has_other_response = False
            if art_a >= 0:
                response_dofs = data.response_dofs[art_a]
                has_other_response = response_dofs > 0 and response_dofs != sparse_size
            if art_b >= 0:
                response_dofs = data.response_dofs[art_b]
                has_other_response = has_other_response or (response_dofs > 0 and response_dofs != sparse_size)
            if has_other_response:
                continue

            world = data.world[contact]
            coord_0 = sparse_row_dof[world, slot, 0]
            coord_1 = sparse_row_dof[world, slot, 1]
            if coord_0 >= 0 and coord_1 < 0:
                sparse_row_dof[world, slot, 1] = -2
            elif coord_1 >= 0 and coord_0 < 0:
                sparse_row_dof[world, slot, 0] = -2

    if units:
        name = f"mark_independent_sparse_contact_candidates_{sparse_size}_units"
        return python_kernel(mark_independent_sparse_contact_units_template, name)

    def mark_independent_sparse_contact_candidates_template(
        data: ContactInput,
        total_num_workers: int,
        sparse_row_dof: wp.array3d[int],
    ):
        worker = wp.tid()
        total_contacts = data.count[0]
        for contact in range(worker, total_contacts, total_num_workers):
            slot = data.slot[contact]
            if data.path[contact] != 0 or slot < 0 or data.slots_needed[contact] != 3:
                continue

            art_a = data.art0[contact]
            art_b = data.art1[contact]
            has_other_response = False
            if art_a >= 0:
                response_dofs = data.response_dofs[art_a]
                has_other_response = response_dofs > 0 and response_dofs != sparse_size
            if art_b >= 0:
                response_dofs = data.response_dofs[art_b]
                has_other_response = has_other_response or (response_dofs > 0 and response_dofs != sparse_size)
            if has_other_response:
                continue

            world = data.world[contact]
            coord_0 = sparse_row_dof[world, slot, 0]
            coord_1 = sparse_row_dof[world, slot, 1]
            if coord_0 >= 0 and coord_1 < 0:
                sparse_row_dof[world, slot, 1] = -2
            elif coord_1 >= 0 and coord_0 < 0:
                sparse_row_dof[world, slot, 0] = -2

    name = f"mark_independent_sparse_contact_candidates_{sparse_size}"
    return python_kernel(mark_independent_sparse_contact_candidates_template, name)


@cache
def _get_build_independent_sparse_contact_groups_kernel(
    max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    build_serial_contacts: bool = False,
    units: bool = False,
) -> "wp.Kernel":
    """Build exact independent and serial contact schedules for sparse diagonal response.

    ``units`` schedules patch contact units instead of contact triples (see
    :func:`_build_independent_sparse_contact_units_kernel`). Each world's warp
    links the scalar triples and lists the remaining ones in parallel; the
    schedules equal those of serial walks over the segment's whole triples.
    """
    del device_arch
    M = int(max_constraints)
    D = int(max_world_dofs)
    if units:
        return _build_independent_sparse_contact_units_kernel(M, D, build_serial_contacts)
    S = (M + 2) // 3
    serial_schedule = (
        f"""
    // Compact the remaining contacts in original row order. The solve may
    // inspect up to four normals concurrently, but any active batch is still
    // applied through this serial list in exact GS order.
{_list_serial_contacts_source(count="triple_count", stride=3, head="true", serial_capacity=S)}
    SYNC();
"""
        if build_serial_contacts
        else ""
    )

    snippet = f"""    const int lane = LANE;
    const int contact_start = dense_phase_bounds.data[world * 2 + 1];
    const int contact_end = world_constraint_count.data[world];
    const int head_base = world * {D};
    const int sparse_world_base = world * {M} * 2;
    const unsigned earlier_lanes = (1u << lane) - 1u;
    // The contact segment's whole triples.
    const int triple_count = contact_end > contact_start ? (contact_end - contact_start) / 3 : 0;
    for (int coordinate = lane; coordinate < {D}; coordinate += 32)
        group_heads.data[head_base + coordinate] = -1;
    SYNC();

    // Reserve every diagonal coordinate used by a coupled contact. A scalar
    // row sharing one must remain in the serial chain to preserve GS order.
    for (int normal = contact_start + lane * 3; normal + 2 < contact_end; normal += 96) {{
        const int sparse_row = sparse_world_base + normal * 2;
        const int coord_0 = sparse_row_dof.data[sparse_row];
        const int coord_1 = sparse_row_dof.data[sparse_row + 1];
        const bool scalar_candidate = coord_0 < -1 || coord_1 < -1;
        if (!scalar_candidate) {{
            if (coord_0 >= 0) atomicExch(&group_heads.data[head_base + coord_0], -2);
            if (coord_1 >= 0) atomicExch(&group_heads.data[head_base + coord_1], -2);
        }}
    }}
    SYNC();

    // Link each coordinate's scalar triples in ascending row order, so that
    // every negative link reproduces the original GS order.
{_link_scalar_contacts_source(count="triple_count", stride=3, head="true", dofs=D)}
{serial_schedule}

    int count = 0;
    for (int coordinate_base = 0; coordinate_base < {D}; coordinate_base += 32) {{
        const int coordinate = coordinate_base + lane;
        const int head = coordinate < {D} ? group_heads.data[head_base + coordinate] : -1;
        const int output = ORDERED(count, head >= 0);
        if (head >= 0) group_heads.data[head_base + output] = head;
        SYNC();
    }}
    if (lane == 0) group_count.data[world] = count;
#endif
"""

    name = f"build_independent_sparse_contact_groups_{M}_{D}"
    if build_serial_contacts:
        name += "_serial_contacts"
    return KernelTemplate(
        name,
        kernel_template.params(
            ("world_constraint_count", wp.array[int]),
            ("dense_phase_bounds", wp.array2d[int]),
            ("sparse_row_dof", wp.array3d[int]),
            ("group_count", wp.array[int]),
            ("group_heads", wp.array2d[int]),
            ("serial_count_out", wp.array[int]),
            ("serial_normals", wp.array2d[int]),
        ),
        snippet,
        dispatch=Dispatch(
            "world = wp.tid() // 32\n$CALL",
            ("world",),
            (("world", int),),
        ),
        preludes=(lane_group(32),),
        native_name="build_independent_sparse_contact_groups_native",
    ).kernel()


def _link_scalar_contacts_source(*, count: str, stride: int, head: str, dofs: int) -> str:
    """Return CUDA source that links each coordinate's scalar contacts across a world's warp.

    Contacts lie ``stride`` rows apart from ``contact_start``: position ``p`` is row
    ``contact_start + p * stride`` for ``p`` below the C expression ``count``, and the C
    condition ``head`` on ``normal`` tells whether that row heads a contact. A scalar contact
    (one marked coordinate) links to the next scalar contact of its coordinate in row order
    (``-(next + 3)``, ``-2`` at the chain end) unless a coupled contact reserved the coordinate
    (``-1``), and ``group_heads`` ends with each chain's first contact. These are the outputs of
    prepending the contacts to their chains in reverse row order. The warp visits chunks of 32
    positions from the last: a contact links to the next lane of its coordinate, found by one
    ballot per coordinate bit, else to the coordinate's running head, and the chunk's first
    contact of the coordinate becomes that head.

    The fragment runs on every lane of one world's warp and uses the builder's lane group, ``lane``,
    ``contact_start``, ``sparse_world_base``, ``head_base``, ``sparse_row_dof`` and ``group_heads``,
    the names ``head`` reads, and ``earlier_lanes``, which must equal ``(1u << lane) - 1u``. A scalar
    contact's unmarked slot must hold a coordinate in ``[0, dofs)``, as the marker leaves it: the
    fragment reads ``group_heads`` at that coordinate and matches peers on its low
    ``(dofs - 1).bit_length()`` bits only.
    """
    coordinate_bits = max(dofs - 1, 0).bit_length()
    return f"""
    for (int chunk = ({count} + 31) / 32 - 1; chunk >= 0; --chunk) {{
        const int normal = contact_start + (chunk * 32 + lane) * {stride};
        int coordinate = -1;
        int link_row = 0;
        int running = -1;
        if (chunk * 32 + lane < {count}) {{
            const int sparse_row = sparse_world_base + normal * 2;
            const int coord_0 = sparse_row_dof.data[sparse_row];
            const int coord_1 = sparse_row_dof.data[sparse_row + 1];
            if (({head}) && (coord_0 < -1 || coord_1 < -1)) {{
                const int scalar_coordinate = coord_0 >= 0 ? coord_0 : coord_1;
                running = group_heads.data[head_base + scalar_coordinate];
                if (running != -2) {{
                    coordinate = scalar_coordinate;
                    link_row = coord_0 < 0 ? sparse_row : sparse_row + 1;
                }} else {{
                    sparse_row_dof.data[coord_0 < -1 ? sparse_row : sparse_row + 1] = -1;
                }}
            }}
        }}
        unsigned peers = BALLOT(coordinate >= 0);
        for (int bit = 0; bit < {coordinate_bits}; ++bit) {{
            const unsigned ones = BALLOT((coordinate >> bit) & 1);
            peers &= ((coordinate >> bit) & 1) ? ones : ~ones;
        }}
        if (coordinate >= 0) {{
            const unsigned later = peers & ~earlier_lanes & ~(1u << lane);
            const int next_normal = later != 0u
                ? contact_start + (chunk * 32 + FIRST(later)) * {stride} : running;
            sparse_row_dof.data[link_row] = next_normal >= 0 ? -(next_normal + 3) : -2;
        }}
        SYNC();
        if (coordinate >= 0 && (peers & earlier_lanes) == 0u)
            group_heads.data[head_base + coordinate] = normal;
        SYNC();
    }}
"""


def _list_serial_contacts_source(*, count: str, stride: int, head: str, serial_capacity: int) -> str:
    """Return CUDA source that lists the contacts outside every chain, in row order, across the warp.

    Positions and ``head`` are those of :func:`_link_scalar_contacts_source`. A contact head
    without a link (no marked coordinate left below ``-1``) joins ``serial_normals``, compacted
    by one ballot per chunk of 32 positions, and lane 0 writes their count.

    The fragment runs on every lane of one world's warp and uses the builder's lane group, ``lane``,
    ``world``, ``contact_start``, ``sparse_world_base``, ``sparse_row_dof``, ``serial_normals`` and
    ``serial_count_out``, and the names ``head`` reads. It declares ``serial_count`` in the scope it
    is placed in.
    """
    return f"""
    int serial_count = 0;
    for (int base = 0; base < {count}; base += 32) {{
        const int normal = contact_start + (base + lane) * {stride};
        bool serial = false;
        if (base + lane < {count}) {{
            const int sparse_row = sparse_world_base + normal * 2;
            const int coord_0 = sparse_row_dof.data[sparse_row];
            const int coord_1 = sparse_row_dof.data[sparse_row + 1];
            serial = ({head}) && !(coord_0 < -1 || coord_1 < -1);
        }}
        const int serial_slot = ORDERED(serial_count, serial);
        if (serial) serial_normals.data[world * {serial_capacity} + serial_slot] = normal;
    }}
    if (lane == 0) serial_count_out.data[world] = serial_count;
"""


def _build_independent_sparse_contact_units_kernel(M: int, D: int, build_serial_contacts: bool) -> "wp.Kernel":
    """Build the independent and serial schedules over patch contact units.

    Under patch friction a contact's rows form a unit: its normal row, followed by
    its two tangent rows only when it owns an anchor. Every row of the contact
    segment other than a tangent row heads a unit, and a tangent row names its
    unit's head as parent. The schedules are those of the triple builder, with
    links and the serial list holding unit heads. Members of a patch ring share
    their body pair, hence their coordinate and independence class.

    Two serial walks over the rows define the schedules. When the rows form units
    as the producers write them, both walks visit exactly the non-friction rows,
    and the warp builds the same links, heads and serial list in parallel; lane 0
    runs the walks for any other row layout.
    """
    friction = int(PGS_CONSTRAINT_TYPE_FRICTION)
    unit_head = f"world_row_type.data[row_base + normal] != {friction}"
    # The serial walks that define the schedules, which lane 0 runs for any other row layout.
    serial_walks = f"""
    // Prepending candidates in reverse row order makes each negative link
    // reproduce the original ascending GS order. Each step lands on the last
    // row of the previous unit, which is its head or one of its tangent rows.
    if (lane == 0) {{
        int row = contact_end - 1;
        while (row >= contact_start) {{
            int normal = row;
            if (world_row_type.data[row_base + row] == {friction}) {{
                const int head = world_row_parent.data[row_base + row];
                if (head >= contact_start && head < row) normal = head;
            }}
            const int sparse_row = sparse_world_base + normal * 2;
            const int coord_0 = sparse_row_dof.data[sparse_row];
            const int coord_1 = sparse_row_dof.data[sparse_row + 1];
            const bool scalar_candidate = coord_0 < -1 || coord_1 < -1;
            const int scalar_coordinate = coord_0 >= 0 ? coord_0 : coord_1;
            if (scalar_candidate) {{
                if (group_heads.data[head_base + scalar_coordinate] != -2) {{
                    const int next_normal = group_heads.data[head_base + scalar_coordinate];
                    const int link = next_normal >= 0 ? -(next_normal + 3) : -2;
                    if (sparse_row_dof.data[sparse_row] < 0)
                        sparse_row_dof.data[sparse_row] = link;
                    else
                        sparse_row_dof.data[sparse_row + 1] = link;
                    group_heads.data[head_base + scalar_coordinate] = normal;
                }} else if (coord_0 < -1) {{
                    sparse_row_dof.data[sparse_row] = -1;
                }} else {{
                    sparse_row_dof.data[sparse_row + 1] = -1;
                }}
            }}
            row = normal - 1;
        }}
    }}
    SYNC();
"""
    serial_schedule = (
        f"""
    // Compact the remaining units in original row order. The solve may
    // inspect up to four units concurrently, but any active batch is still
    // applied through this serial list in exact GS order.
    if (lane == 0) {{
        int serial_count = 0;
        int normal = contact_start;
        while (normal < contact_end) {{
            const int sparse_row = sparse_world_base + normal * 2;
            const bool scheduled_independently = sparse_row_dof.data[sparse_row] < -1
                || sparse_row_dof.data[sparse_row + 1] < -1;
            if (!scheduled_independently)
                serial_normals.data[world * {M} + serial_count++] = normal;
            const bool owns_tangents = normal + 2 < contact_end
                && world_row_type.data[row_base + normal + 1] == {friction}
                && world_row_parent.data[row_base + normal + 1] == normal;
            normal += owns_tangents ? 3 : 1;
        }}
        serial_count_out.data[world] = serial_count;
    }}
    SYNC();
"""
        if build_serial_contacts
        else ""
    )
    parallel_serial_schedule = (
        _list_serial_contacts_source(count="contact_end - contact_start", stride=1, head=unit_head, serial_capacity=M)
        if build_serial_contacts
        else ""
    )
    parallel_links = _link_scalar_contacts_source(count="contact_end - contact_start", stride=1, head=unit_head, dofs=D)

    snippet = f"""    const int lane = LANE;
    const int contact_start = dense_phase_bounds.data[world * 2 + 1];
    const int contact_end = world_constraint_count.data[world];
    const int head_base = world * {D};
    const int row_base = world * {M};
    const int sparse_world_base = world * {M} * 2;
    const unsigned earlier_lanes = (1u << lane) - 1u;
    for (int coordinate = lane; coordinate < {D}; coordinate += 32)
        group_heads.data[head_base + coordinate] = -1;
    SYNC();

    // Reserve every diagonal coordinate used by a coupled unit. A scalar
    // unit sharing one must remain in the serial chain to preserve GS order.
    // Also check the layout: the segment starts at a head, and each head is
    // followed by the next head or the segment end, directly or after its two
    // tangent rows naming it. Both serial walks then visit exactly the heads.
    bool rows_form_units = contact_start >= contact_end
        || world_row_type.data[row_base + contact_start] != {friction};
    for (int normal = contact_start + lane; normal < contact_end; normal += 32) {{
        const int sparse_row = sparse_world_base + normal * 2;
        const int coord_0 = sparse_row_dof.data[sparse_row];
        const int coord_1 = sparse_row_dof.data[sparse_row + 1];
        const bool unit_head = world_row_type.data[row_base + normal] != {friction};
        const bool tangent_1 = normal + 1 < contact_end && world_row_type.data[row_base + normal + 1] == {friction};
        const bool tangent_2 = normal + 2 < contact_end && world_row_type.data[row_base + normal + 2] == {friction};
        const bool tangent_3 = normal + 3 < contact_end && world_row_type.data[row_base + normal + 3] == {friction};
        const bool named_1 = normal + 1 < contact_end && world_row_parent.data[row_base + normal + 1] == normal;
        const bool named_2 = normal + 2 < contact_end && world_row_parent.data[row_base + normal + 2] == normal;
        if (!unit_head) continue;
        if (tangent_1 && !(named_1 && tangent_2 && named_2 && !tangent_3)) rows_form_units = false;
        const bool scalar_candidate = coord_0 < -1 || coord_1 < -1;
        if (!scalar_candidate) {{
            if (coord_0 >= 0) atomicExch(&group_heads.data[head_base + coord_0], -2);
            if (coord_1 >= 0) atomicExch(&group_heads.data[head_base + coord_1], -2);
        }}
    }}
    rows_form_units = ALL(rows_form_units);
    SYNC();

    if (rows_form_units) {{
{parallel_links}
{parallel_serial_schedule}
    }} else {{
{serial_walks}
{serial_schedule}
    }}
    SYNC();

    int count = 0;
    for (int coordinate_base = 0; coordinate_base < {D}; coordinate_base += 32) {{
        const int coordinate = coordinate_base + lane;
        const int head = coordinate < {D} ? group_heads.data[head_base + coordinate] : -1;
        const int output = ORDERED(count, head >= 0);
        if (head >= 0) group_heads.data[head_base + output] = head;
        SYNC();
    }}
    if (lane == 0) group_count.data[world] = count;
#endif
"""

    name = f"build_independent_sparse_contact_groups_{M}_{D}"
    if build_serial_contacts:
        name += "_serial_contacts"
    name += "_units"
    return KernelTemplate(
        name,
        kernel_template.params(
            ("world_constraint_count", wp.array[int]),
            ("dense_phase_bounds", wp.array2d[int]),
            ("world_row_type", wp.array2d[int]),
            ("world_row_parent", wp.array2d[int]),
            ("sparse_row_dof", wp.array3d[int]),
            ("group_count", wp.array[int]),
            ("group_heads", wp.array2d[int]),
            ("serial_count_out", wp.array[int]),
            ("serial_normals", wp.array2d[int]),
        ),
        snippet,
        dispatch=Dispatch(
            "world = wp.tid() // 32\n$CALL",
            ("world",),
            (("world", int),),
        ),
        preludes=(lane_group(32),),
        native_name="build_independent_sparse_contact_units_native",
    ).kernel()


def _linked_sweeps_request(value, name: str) -> int | str | None:
    """Validate the deprecated linked-tier budget: ``None``, ``"auto"`` or a positive count (decimal text included)."""
    if value is None or value == "auto":
        return value
    count = value
    if isinstance(value, str):
        try:
            count = int(value)
        except ValueError:
            count = None
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError(f"{name} must be None, 'auto' or a positive sweep count, got {value!r}")
    return int(count)

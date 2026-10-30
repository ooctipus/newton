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
from collections.abc import Callable
from contextlib import contextmanager
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
from . import route_table as _route_table
from . import row_metadata as _row_metadata
from .capture_joins import capturing
from .conditioning import ArmatureFloor, conditioning_armature
from .configuration import check_options, kernel_choices
from .contact_input import ContactInput
from .contact_input import bind as bind_contact_input
from .contact_input import bind_routes as bind_contact_routes
from .contact_row_plan import SHARED_METADATA_PRODUCERS, contact_row_plan
from .contact_torsion import (
    prepare_torsion_rows,
    validate_torsion_step,
)
from .contact_unit import CONTACT_UNIT_STATE_FIELDS, RING_BLOCK_SIZE
from .dense_rows import RowSupport
from .dense_rows import bind as bind_dense_rows
from .dense_rows import bind_mf as bind_mf_rows
from .device_preludes import lane_group
from .field_group import SolverArrays
from .forest_plan import ForestPlan, joint_forest, row_support
from .friction_patches import (
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
    RIGID_SPEED_ROWS,
    BodyFactors,
    allocate_connect_slots,
    allocate_joint_velocity_limit_slots,
    allocate_mimic_slots,
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
from .route_table import RouteTable, capable
from .row_build import contact_row_prelude, populate_world_J_masked
from .rows import JOINT_LIMIT_WARPS_PER_BLOCK, RowLaws
from .solve_policy import SolvePolicy, SolveReadoutRecord
from .solver_fields import (
    ARTICULATION_ORIGIN,
    ARTICULATION_WORK,
    AUGMENTED_DRIVE_ROWS,
    BODY_INERTIA,
    CAPACITY_STATUS,
    CONTACT_RUNS,
    DENSE_CONTACT_UNITS,
    DENSE_DELASSUS,
    DENSE_ROW_WEIGHTS,
    DENSE_ROWS,
    DENSE_WARMSTART,
    DOF_WORK,
    FUSED_DIAGONAL_LIMITS,
    JOINT_FOREST,
    MF_CONTACT_UNITS,
    MF_COUNTERS,
    MF_ROW_WEIGHTS,
    MF_ROWS,
    MF_WARMSTART,
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
from .staging import arch_budget, factor_budget
from .step_plan import StepPlan, first_failed
from .sweep_routes_sparse import _SPARSE_LOCAL_WORKSPACE_BYTES
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

# Parallel tiers grid-stride over per-tier world lists from this many tiers on: one classify launch against
# (tiers - 1) x worlds idle block entries.
_TIER_LIST_MIN_TIERS = 3
# The largest articulation the tiers whiten in-kernel: its factor is staged in the block (case K4).
_INKERNEL_MAX_DOF = 32
_FPGS_SYNC_TIMINGS = os.environ.get("FEATHER_PGS_SYNC_TIMINGS", "0").lower() in {"1", "true", "yes", "on"}
_FPGS_SYNC_TIMINGS_START = int(os.environ.get("FEATHER_PGS_SYNC_TIMINGS_START", "20"))
_FPGS_SYNC_TIMINGS_COUNT = max(int(os.environ.get("FEATHER_PGS_SYNC_TIMINGS_COUNT", "1")), 0)
_SMALL_DOF_THRESHOLD_DEFAULT = 12
# Threads per block of the tiled factor, triangular-solve and response kernels.
_TILE_THREADS = 64
# Beyond this guard, the relaxation weight is too close to zero to be a
# useful float32 PGS update and the assembled split path risks overflow when
# it converts the weight back to a diagonal regularizer.


# Per-world preset of the first rejected row slot: no reservation was rejected.
_ROW_SLOT_UNBOUNDED = ROW_SLOT_UNBOUNDED
_CONTACT_BUILD_THREAD_CAP = 65536
# The masked row builder's thread budget, in multiples of the contact-build cap.
_ROWS_THREADS_X = 4
_CONTACT_JACOBIAN_WORKER_CAP = 4096
_CONTACT_JACOBIAN_MAX_DOF = 10
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
    geometric/speculative bias with Newton's rebound target. As in a conventional discrete PGS solver, the qualifying
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

    # Test-only route pins: an axis value a test fixes where the rules would choose another, so that every route the
    # data can select runs on the test scenes. No option or environment variable sets one; the route record lists them
    # (routes.pins). Keys:
    #   storage "dense": no packed (native sparse) factor.     evaluator "ordinary": no held tree.
    #   grouping "forest" or "per_size": the held families.   star "held": a star's diagonal family in the held tree.
    #   producer "compact_jacobian" or "general": no local packets or keyed rows.
    #   claims: the runtime claims kept, of ("resolved_worlds",).
    #   active_rows False: full-order sparse sweeps.           contact_units None: unfused sparse contact triplets.
    #   transaction "single" or "staged": fused triplets'.     mf_storage "streamed": streamed matrix-free impulses.
    #   repair_skip, repair_classify, predicted_twists False: the held tree's launch mapping without that step.
    #   row_support False: sparse rows on explicit indices.     placement "general": case K6 (route_table.RouteTable).
    #   torsion_preparation "host": the host reference of the torsion rows.
    #   double_buffer False: one set of mass matrices and Jacobians, whose rows stay readable after a step.
    #   tiers: the Jacobi tiers' row bound, on any class whose response they form (0: no tiers).
    #   tier_sweeps: every Jacobi tier's sweep budget, in place of the equal-accuracy rule.
    #   cholesky_kernel, trisolve_kernel, hinv_jt_kernel: a dense kernel (CUDA).
    #   formulation "matrix_free": the CPU solves the rows matrix-free, with the held tree, as CUDA does.
    _route_pins: ClassVar[dict[str, object]] = {}

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
        return self.laws.contact_torsion_radius

    @property
    def contact_torsion_shape_indices(self) -> tuple[int, ...] | None:
        """Experimental construction-only global shape selection."""
        return self.laws.contact_torsion_shape_indices

    @property
    def contact_torsion_shape_patterns(self) -> tuple[str, ...] | None:
        """Experimental construction-only shape-label regular expressions."""
        return self.laws.contact_torsion_shape_patterns

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
        contact_friction_scale: float = 1.0,
        contact_shared_anchor: bool = False,
        enable_joint_limits: bool = False,
        enable_bilateral_preelimination: bool = False,
        joint_limit_activation_gap: float = float("inf"),
        enable_joint_velocity_limits: bool = False,
        velocity_limit_activation_fraction: float = 0.0,
        pgs_iterations: int = 12,
        pgs_beta: float = 0.2,
        pgs_cfm: float = 1.0e-6,
        pgs_omega: float = 1.0,
        pgs_contact_regularization: float = 0.0,
        dense_max_constraints: int = 32,
        pgs_warmstart: bool = False,
        mf_max_constraints: int = 512,
        nvtx: bool = False,
        pgs_debug: bool = False,
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
        contact_update: Literal["sequential", "unit"] = "sequential",
        contact_torsion_radius: float = 0.0,
        contact_torsion_shape_indices: tuple[int, ...] | None = None,
        contact_torsion_shape_patterns: tuple[str, ...] | None = None,
        contact_compliance: bool = False,
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
            contact_torsion_radius: Experimental effective spin radius [m], zero disables.
                Explicit material/footprint assumption: for uniform pressure on a disk
                of radius R, the effective radius is 2*R/3. Does not consume the generic
                material torsion default. Sliding and spin share one Coulomb budget.
                Currently supports dense articulated contacts in CUDA matrix-free,
                immediate/current/interleaved mode without warmstarting,
                hydroelastic contact, contact compliance,
                or debug mode. Final normal load always bounds the friction budget.
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
                iterations. Defaults to ``-1``.
            contact_friction_shared_anchor (bool, optional): If true, friction rows use the midpoint
                between the two contact witness points as the Jacobian point on both bodies. Normal
                rows keep their original witness points. This avoids a tangential force couple when
                the witnesses are separated along the contact normal. Applies to velocity-only point
                friction (``friction_anchor_beta=0``); patch friction warns and uses its persistent
                material anchors instead. Defaults to False.
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
                penetrating-contact stabilization. Must be finite
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
            enable_bilateral_preelimination (bool, optional): Eliminate each articulation's mimic and
                connect rows exactly (:func:`.response_builds_rows.elimination`): every other row
                (contacts especially) takes the closed-loop response, ``Z' = (I - Z_B G^-1 Z_B^T) Z``,
                and the predictor velocity is projected onto the closures once per step. Blocks are
                sized from the model (an articulation's mimic rows plus three per closure); a row
                that depends on the others leaves its block. Requires the packed-row (CUDA) routes;
                the CPU warns and keeps the iterative rows. Defaults to False.
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
                multi-body Newton's third law). No Baumgarte bias. Each row keeps a
                non-negative impulse total, like a contact normal, so the sweeps converge
                to the complementarity solution, independent of the row order.

                Only supported on CUDA; CPU construction with
                ``enable_joint_velocity_limits=True`` raises :class:`NotImplementedError`.
                Defaults to False.
            velocity_limit_activation_fraction (float, optional): Proximity gate for velocity-limit
                row allocation. ``0.0`` allocates the lower/upper row pair of every finitely
                limited joint DOF, and the speed row of every finitely limited free-rigid-body
                linear or angular speed, each step. A positive fraction allocates them only when
                the current speed satisfies ``|qd| >= fraction * limit`` (the magnitude of the
                linear or angular velocity for a free body's speed row), skipping rows that would
                be exact no-op clamps. Because the gate samples the pre-solve velocity, a DOF
                crossing the threshold between steps is clamped one step late. Must be non-negative
                and not NaN. Defaults to ``0.0``.
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
                sequential update. Only the CPU's loop sweeps implement ``"unit"``: its assembled dense rows'
                sweep and its free-body sweep. Every CUDA sweep keeps
                ``"sequential"``; ``get_path_selection()`` records which sweep runs which law under
                ``routes.sweep.contact_update``, and construction warns when a requested ``"unit"`` reaches only
                some sweeps. Experimental. Defaults to ``"sequential"``.
            pgs_iterations (int, optional): Number of Gauss-Seidel iterations to apply per frame. Defaults to 12.
            pgs_beta (float, optional): ERP style position correction factor for contact, joint-limit,
                mimic and connect rows. Defaults to 0.2.
            pgs_cfm (float, optional): Compliance/regularization added to the Delassus diagonal. Defaults to 1.0e-6.
            pgs_omega (float, optional): Successive over-relaxation factor for the PGS sweep. Defaults to 1.0.
            pgs_solve_readout (str, optional): Opt-in counters of the native Gauss-Seidel sweeps.
                ``"sweeps"`` counts, per sweep kernel and world, how many sweeps each solve executed
                and whether it stopped before its budget; ``"trace"`` adds, per
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
                target fires are solved rigid. ``0`` is the
                exact rigid law. Values above ``1e6`` are rejected because they are not
                numerically useful in the float32 solve. Defaults to 0.0.
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
            mf_max_constraints (int, optional): Maximum number of matrix-free constraints per world.
                Packed metadata on CUDA supports at most 65535 rows. Defaults to 512.
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
                    "pgs_iterations": pgs_iterations,
                    "pgs_warmstart": pgs_warmstart,
                    "enable_restitution": enable_restitution,
                    "pgs_contact_regularization": pgs_contact_regularization,
                    "pgs_debug": pgs_debug,
                    "contact_friction_position_iterations": contact_friction_position_iterations,
                    "contact_friction_shared_anchor": contact_friction_shared_anchor,
                    "contact_shared_anchor": contact_shared_anchor,
                },
            )
        self.compliance_contact_count = 0
        self.compliance_skipped_contact_count = 0
        self._compliant_contacts = None
        self._compliant_prepared = False
        # Read the FEATHER_PGS_* environment once; undeclared names are recorded, not rejected.
        self._environment_record = _environment_flags.record()
        # The test pins in force at construction, which every admission reads and the route record lists.
        self._route_pins = dict(type(self)._route_pins)
        super().__init__(model)
        # Every scalar option coerced and checked once (configuration.OPTION_RULES).
        self.options = options = check_options(requested_options)

        # The dense kernels from the device (configuration.kernel_choices).
        choices, self._kernel_choice_reasons = kernel_choices(self._route_pins, is_cpu=model.device.is_cpu)

        # Every sweep compiles the exit law and the policy's readout.
        self._solve_policy = SolvePolicy(readout=options.pgs_solve_readout)
        self._solve_readout = SolveReadoutRecord(self._solve_policy)
        dense_max_constraints = options.dense_max_constraints
        self._ws_prev_dense_impulses = None
        self._ws_prev_dense_row_type = None
        self._ws_prev_dense_row_parent = None
        self._ws_prev_dense_slot_sorted = None
        if options.pgs_warmstart and options.contact_friction_position_iterations >= 0:
            raise NotImplementedError(
                "contact_friction_position_iterations with pgs_warmstart=True needs an explicit "
                "pre-solve zeroing pass for skipped friction rows"
            )

        # Allocated lazily in :meth:`_allocate_mf_buffers` only when enabled.
        self._ws_prev_mf_impulses = None
        self._ws_prev_dt = 0.0
        self._ws_prev_mf_row_type = None
        self._ws_prev_mf_row_parent = None
        self._ws_prev_slot_sorted = None

        # CUDA solves the rows matrix-free, from their whitened placements; the CPU assembles the Delassus rows
        # and sweeps them with its loop kernels, unless the test pin formulation="matrix_free" runs CUDA's there.
        self._matrix_free = bool(model.device.is_cuda or self._route_pins.get("formulation") == "matrix_free")
        self.mf_max_constraints = options.mf_max_constraints
        if self._matrix_free:
            _validate_mf_metadata_encoding(self.mf_max_constraints)
        elif self.options.enable_joint_velocity_limits:
            # Only the CUDA sweeps implement them.
            raise NotImplementedError("enable_joint_velocity_limits=True requires CUDA")
        self._grouped_response = False

        # Double-buffered mass matrices on CUDA (the sparse factor's producers overwrite every value and take none).
        self._double_buffer = bool(self._route_pins.get("double_buffer", True))
        self._nvtx = nvtx
        self._debug_buffers_enabled = bool(pgs_debug)
        # External NaN dumpers read the stage-3 snapshots; nothing in the solver
        # does. They cost three copies per substep, so they are opt-in, and
        # pgs_debug, which keeps the solver's other diagnostics, keeps them too.
        self._debug_stage3_snapshot_flag = os.environ.get("FEATHER_PGS_DEBUG_SNAPSHOTS") == "1"
        self._debug_stage3_snapshots = self._debug_stage3_snapshot_flag or self._debug_buffers_enabled
        # The solver's device fields (solver_fields): one device and the model's gradient policy.
        self._arrays = SolverArrays(self, model.device, requires_grad=model.requires_grad)
        self._pgs_convergence_log: list[np.ndarray] = []
        # Per-step, per-iteration NCP / MDP residual log for the matrix_free
        # debug path. Each entry is an ndarray of shape
        # ``[pgs_iterations, world_count, 6]`` holding
        # ``(r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir)``
        # per world per iteration. Populated only when ``pgs_debug`` is
        # ``True`` on CUDA.
        self._pgs_ncp_residual_log: list[np.ndarray] = []

        small_dof_threshold = _SMALL_DOF_THRESHOLD_DEFAULT

        self.tile_threads = _TILE_THREADS
        # Launches over the contact capacity (the force export, the warm-start snapshot
        # and gathers, the keyed-row key scatters) grid-stride over about one wave of blocks.
        self._contact_capacity_blocks = contact_capacity_blocks(model.device)

        self.cholesky_kernel = choices["cholesky_kernel"]
        self.trisolve_kernel = choices["trisolve_kernel"]
        self.hinv_jt_kernel = choices["hinv_jt_kernel"]
        self.small_dof_threshold = small_dof_threshold

        self._step = 0
        self._force_mass_update = False
        self._mass_update_requested = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
        # Host mirror of the global mass-update flag evaluated by the last
        # _stage1_crba call; gates the per-step H memsets (see _stage1_crba).
        self._mass_update_global_flag = True
        self._last_step_dt = None
        self._fk_id_cache_available = bool(model.articulation_count and model.joint_count and not model.requires_grad)
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

        self.forest: ForestPlan | None = None
        # The model's joint forest (Model.joint_ancestor): conditioning depths, body maps, CRBA's ancestor test.
        forest = joint_forest(model)
        self._kinematic_joint_mask = wp.zeros(model.joint_count, dtype=wp.int32, device=model.device)
        self._kinematic_dof_mask = wp.zeros(model.joint_dof_count, dtype=wp.int32, device=model.device)
        self._apply_kinematic_state(self._kinematic_state(forest))
        # The forest plan: topology, groups, root kinds, world mapping, body maps. The matrix-free formulation
        # elides fully-kinematic free bases from the compact response mapping of its rows.
        self.forest = ForestPlan.build(model, forest, self._kinematic_dof_mask_host, matrix_free=self._matrix_free)
        self._compact_contact_jacobian = bool(
            model.device.is_cuda
            and not model.requires_grad
            and self.forest.body_response_dof_mask is not None
            and self.forest.size_groups
            and max(self.forest.size_groups) <= _CONTACT_JACOBIAN_MAX_DOF
        )
        # The row laws' tables: the contact law, speed limits, the joint-limit tables, the mimic and connect plans,
        # contact torsion and the materials.
        self.laws = RowLaws.build(model, self.forest, options, matrix_free=self._matrix_free)
        self._build_preelimination_plan(model)
        # Axis 2 of the route table, the row laws, which every owner's admission reads (route_table.capable).
        self.laws.decide(preelimination=self._preelim_active)
        laws = self.laws.set
        self._setup_passive_joint_forces(model)
        self._setup_fused_diagonal_joint_limits(model)
        # The held tree's per-coordinate drive rows (each coordinate owns one, prepared apart from inverse dynamics)
        # without gradients wherever each coordinate has one owning drive row. The held tree's admission, which the
        # representations only it evaluates (the compact diagonal, the native sparse factor) read before they are
        # chosen, needs them; ordinary dynamics drop them once the held tree declines (below) and evaluate drives
        # with the joint forces.
        self._setup_parallel_augmented_drive_topology(
            model, not model.requires_grad and self.forest.articulation_max_dofs > 0
        )
        from . import held_tree  # noqa: PLC0415
        from .scalar_branches import plan_scalar_branches  # noqa: PLC0415

        # Axis 3 of the route table, the placement case: the owners below, in precedence order (route_table.CASES),
        # each test their own fit once no earlier case holds the class, and the first that fits claims it.
        self.routes = RouteTable(self._route_pins)
        # Case K2, a tree beside a star: the sparse diagonal, which does not solve the appended torsion row.
        declined = self.routes.taken("K2") or (
            "Contact torsion keeps the general response." if self.laws.contact_torsion_enabled else None
        )
        sparse_diagonal_pair = (
            None
            if declined
            else self._select_sparse_diagonal_response_pair(factor_dense_contract=self._fused_diagonal_joint_limits)
        )
        held = None
        if sparse_diagonal_pair is not None:
            # Its compact diagonal is the held tree's representation: a candidate only where the held tree takes
            # the dense groups with it, the diagonal group as scalar branches or a held family.
            scalar, _ = plan_scalar_branches(self, size=sparse_diagonal_pair[0])
            held = held_tree.compact_decline(self, sparse_diagonal_pair[0], scalar=scalar is not None)
            if held is not None:
                sparse_diagonal_pair = None
        self._sparse_diagonal_contact_solve = self.routes.claim("K2", sparse_diagonal_pair is not None)
        if declined:
            self.routes.reasons["K2"] = declined
        elif held is not None:
            self.routes.reasons["K2"] = f"Only the held tree evaluates the compact diagonal. {held}"
        elif sparse_diagonal_pair is None:
            self.routes.reasons["K2"] = (
                "No world pairs a diagonal-mass articulation with fused limits and one compact dense articulation."
            )
        else:
            self.routes.reasons["K2"] = (
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
            and self.options.enable_contact_friction
            and self.options.contact_friction_gap_threshold == math.inf
            and not self.laws.friction_anchors_enabled
        )
        # Patch contacts are units instead: a normal row, followed by its two tangent rows only when the contact
        # owns an anchor. The unit schedule replaces every stride-3 assumption of the triple schedule.
        self._sparse_diagonal_contact_units = bool(
            self._sparse_diagonal_contact_solve
            and self.options.enable_contact_friction
            and self.options.contact_friction_gap_threshold == math.inf
            and self.laws.friction_anchors_enabled
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
        self._hinv_jt_computes_diag = self._matrix_free and not self._preelim_active
        self._execution_plan = _FeatherPGSExecutionPlan.build(
            self.forest.size_groups,
            max_constraints=self.dense_max_constraints,
            max_shared_memory=int(getattr(model.device, "max_shared_memory_per_block", 0)),
            cholesky_kernel=self.cholesky_kernel,
            hinv_jt_kernel=self.hinv_jt_kernel,
            small_dof_threshold=self.small_dof_threshold,
            tile_threads=self.tile_threads,
            diagonal_mass_sizes=(
                self.forest.diagonal_mass_sizes
                if self.cholesky_kernel == "auto" and self.trisolve_kernel == "auto"
                else frozenset()
            ),
        )
        self._hinv_jt_diag_sizes = frozenset(
            size
            for size in self.forest.size_groups
            if self._hinv_jt_computes_diag
            and (self._execution_plan.use_diagonal_mass(size) or self._execution_plan.use_tiled_hinv_jt(size))
        )
        self._paired_response_candidate_sizes = frozenset(
            size
            for size in self.forest.size_groups
            if model.device.is_cuda
            and not model.requires_grad
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
        # Case K3, a small tree beside a body: the local owners, claimed per world at run time.
        local_decline = (
            self.routes.taken("K3")
            or capable("local", laws)
            or first_failed(
                (
                    (
                        self.forest.has_free_rigid_bodies,
                        "Local owners serve worlds with free rigid bodies; there are none.",
                    ),
                    (self._local_solve_max_rows > 0, "dense_max_constraints leaves the local owners no row."),
                )
            )
        )
        local_base_supported = local_decline is None
        local_primary_articulation = np.full(self.forest.world_count, -1, dtype=np.int32)
        local_pair_articulation = np.full(self.forest.world_count, -1, dtype=np.int32)
        local_residual_pair_articulation = np.full(self.forest.world_count, -1, dtype=np.int32)
        local_candidates: dict[int, list[int]] = {size: [] for size in self.forest.size_groups}
        local_pair_candidates: dict[int, list[int]] = {size: [] for size in self.forest.size_groups}
        local_pair_secondaries: dict[int, list[int]] = {size: [] for size in self.forest.size_groups}
        local_residual_candidates: dict[int, list[int]] = {size: [] for size in self.forest.size_groups}
        local_residual_secondaries: dict[int, list[int]] = {size: [] for size in self.forest.size_groups}
        response_by_world: list[list[int]] = [[] for _ in range(self.forest.world_count)]
        for art in np.flatnonzero(self.forest.response_dof_count > 0):
            response_by_world[int(self.forest.articulation_world[art])].append(int(art))

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
            articulation_dof_start = self.forest.articulation_dof_start_host
            for world, response_arts in enumerate(response_by_world):
                nonfree_arts = [art for art in response_arts if self.forest.is_free_rigid_host[art] == 0]
                if len(nonfree_arts) != 1:
                    continue
                primary_art = nonfree_arts[0]
                primary_dofs = int(self.forest.response_dof_count[primary_art])
                if (
                    primary_dofs <= 0
                    or primary_dofs > _LOCAL_INTERNAL_MAX_DOF
                    or self._execution_plan.use_tiled_hinv_jt(primary_dofs)
                    or not local_fits(primary_dofs, max_rows=min(self._local_solve_max_rows, primary_dofs))
                ):
                    continue
                local_primary_articulation[world] = primary_art
                local_candidates[primary_dofs].append(primary_art)

                free_arts = [art for art in response_arts if self.forest.is_free_rigid_host[art] != 0]
                if len(response_arts) != 2 or len(free_arts) != 1:
                    continue
                secondary_art = free_arts[0]
                secondary_dofs = int(self.forest.response_dof_count[secondary_art])
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
        self._local_solve_owner = wp.zeros(self.forest.world_count, dtype=wp.int32, device=model.device)
        self._local_general_world_count = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._local_general_worlds = wp.empty(self.forest.world_count, dtype=wp.int32, device=model.device)
        self._local_general_solver_blocks = min(
            self.forest.world_count, max(1, int(model.device.sm_count) * _LOCAL_GENERAL_BLOCKS_PER_SM)
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
        self._local_internal_fast_path = self.routes.claim(
            "K3", local_base_supported and any(count > 0 for count in self._local_internal_art_counts.values())
        )
        self.routes.reasons["K3"] = local_decline or (
            "Worlds with one small non-free articulation (and at most one free body) solve in local shared memory."
            if self._local_internal_fast_path
            else "No world holds one non-free articulation of at most "
            f"{_LOCAL_INTERNAL_MAX_DOF} DOFs whose local staging fits the device."
        )
        # All inactive weight arguments alias this one element. Their producer
        # kernels skip stores and their consumer kernels skip reads when
        # regularization is disabled.
        self._contact_row_w_dummy = self._arrays.placeholder(wp.float32, (1, 1), 1.0)
        # The parallel matrix-free tiers pool each friction pair's cone over its patch
        # ring and keyed rows write the patch row law, so both can serve point and patch
        # friction (Jacobi sweeps, not the serial patch sweep).
        self._parallel_contact_paths, self._parallel_contact_paths_reason = self._admit_parallel_contact_paths()
        if self._matrix_free:
            self._jy_world_aliased = self._detect_jy_world_identity()
        else:
            self._jy_world_aliased = False
        self._allocate_common_buffers(model)
        self._compact_diagonal_mass_size = self._sparse_diagonal_response_size
        self._row_watermark = bool(row_watermark)
        self._setup_sparse_mass_matrix(model)
        # The matrix-free sweeps read the dense rows packed (packed_rows): decided before the buffers, since that
        # placement allocates neither a size group's response nor the world layout.
        self._mf_sweeps = bool(
            self._matrix_free
            and self.routes.case not in ("K1", "K2")
            and self.forest.world_count > 0
            and self.forest.max_world_dofs > 0
            and self.mf_max_constraints > 0
        )
        if self._mf_sweeps:
            self._jy_world_aliased = False
            self._hinv_jt_diag_sizes = frozenset()
        # The compact boundary owns the rows and response of every world the sparse-diagonal owner pairs with a
        # six-coordinate dense articulation (decided before the buffers: it clears its own rows, so their Jacobian
        # needs no second buffer). It writes the point and the patch row law; torsion and compliance keep ordinary
        # rows.
        from .compact_contact import supported  # noqa: PLC0415

        declined = capable("compact", self.laws.set)
        if declined is None and not supported(self):
            declined = (
                "Declined: requires the six-coordinate sparse-diagonal triple or unit contract and no row checks."
            )
        self._compact_contact_boundary = declined is None
        self._compact_contact_reason = declined or "Admitted: the six-coordinate sparse-diagonal triples or units."
        self._allocate_buffers(model)
        self._allocate_world_buffers(model)
        # CONTACT row_parent belongs to ordinary friction/patch load linkage.
        # Torsion must never reinterpret it, including when patch PRs are merged.
        self._contact_torsion_group = wp.full(
            self.row_parent.shape if self.laws.contact_torsion_enabled else (1, 1),
            -1,
            dtype=wp.int32,
            device=model.device,
        )
        # Bilateral pre-elimination corrects the grouped response after H^-1 J^T,
        # so that path must retain the existing post-correction world gather.
        self._hinv_jt_writes_world = (
            self._matrix_free
            and not self._jy_world_aliased
            and not self._preelim_active
            and self.routes.case not in ("K1", "K2")
        )
        self._hinv_jt_tiled_writes_group = not self._hinv_jt_writes_world or self.options.pgs_warmstart
        # Case K5, a tree beside one free body: factor coordinates, over the paired response.
        self._paired_response_reason = self.routes.taken("K5")
        response_pair = None if self._paired_response_reason else self._select_paired_response()
        if response_pair is None and self._paired_response_reason is None:
            self._paired_response_reason = "No world pairs one tiled-response articulation with exactly one free body."
        self._paired_response_primary_size = response_pair[0] if response_pair is not None else None
        self._paired_response_secondary_size = response_pair[1] if response_pair is not None else None
        self._paired_response_secondary_groups = (
            wp.array(response_pair[2], dtype=wp.int32, device=model.device) if response_pair is not None else None
        )
        self.routes.reasons["K5"] = ("No paired response." if response_pair is None else None) or capable(
            "paired_factor", laws
        )
        self._paired_factor_coordinates = self.routes.claim("K5", self.routes.reasons["K5"] is None)
        # Persistent patches allocate one tangent pair per surviving anchor, so their contact rows are not
        # uniform normal/tangent/tangent triples; the generic row path pools the patch load through the linked
        # normal rows instead.
        self._factor_coordinate_contact_triples = bool(
            self._paired_factor_coordinates
            and self.options.enable_contact_friction
            and self.options.contact_friction_gap_threshold == math.inf
            and not self.laws.friction_anchors_enabled
        )
        # Patch contacts are units of one row, or three when they own an anchor;
        # the factor sweep walks them with the generic path's row order and
        # arithmetic, solving each tangent pair once.
        self._factor_coordinate_contact_units = bool(self._paired_factor_coordinates and laws.patch)
        self._paired_factor_primary_groups_by_world = None
        self._paired_factor_secondary_groups_by_world = None
        self._paired_factor_primary_offsets_by_world = None
        self._paired_factor_secondary_offsets_by_world = None
        if response_pair is not None:
            if self._paired_factor_coordinates:
                response_dof_count = self.forest.response_dof_count
                primary_arts = np.flatnonzero(response_dof_count == response_pair[0])
                secondary_arts = np.flatnonzero(response_dof_count == response_pair[1])
                primary_groups = np.empty(self.forest.world_count, dtype=np.int32)
                secondary_groups = np.empty(self.forest.world_count, dtype=np.int32)
                primary_offsets = np.empty(self.forest.world_count, dtype=np.int32)
                secondary_offsets = np.empty(self.forest.world_count, dtype=np.int32)
                articulation_world = self.forest.articulation_world
                articulation_dof_start = self.forest.articulation_dof_start_host
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
            self._arrays.allocate(SPARSE_DIAGONAL_ROWS, world=self.forest.world_count, row=self.dense_max_constraints)
        else:
            self._arrays.allocate(SPARSE_DIAGONAL_ROWS, world=1, row=1)
        if self._sparse_diagonal_contact_triples or self._sparse_diagonal_contact_units:
            self._arrays.allocate(
                SPARSE_CONTACT_GROUPS, world=self.forest.world_count, world_dof=self.forest.max_world_dofs
            )
        else:
            self._arrays.allocate(SPARSE_CONTACT_GROUPS, world=1, world_dof=1)
        if self._sparse_diagonal_speculative_contact_batches:
            # A triple spans three rows, while a unit may be a single normal row.
            rows = self.dense_max_constraints
            serial = rows if self._sparse_diagonal_contact_units else (rows + 2) // 3
            self._arrays.allocate(SPARSE_CONTACT_SERIAL, world=self.forest.world_count, serial=serial)
        else:
            self._arrays.allocate(SPARSE_CONTACT_SERIAL, world=1, serial=1)
        self._allocate_mf_buffers(model)
        self._body_factors_data = self._build_body_factors()
        self.mf_target_velocity = (
            wp.zeros_like(self.mf_rhs, requires_grad=model.requires_grad)
            if self.forest.has_prescribed_response
            else self._arrays.placeholder(wp.float32, (1, 1))
        )
        self._allocate_debug_buffers(model)
        self._scatter_armature_to_groups()
        self._plan_tier_rows()
        self._allocate_packed_rows()
        # The producers of the local owners' rows (case K3) and of the tiers' rows (case K4), each admitted by its
        # own fit: limit prefix and contact packets built in place for worlds of one limited articulation and at
        # most one other (row_packets.py), and keyed rows (keyed_rows.create_owner). Both write the point and the
        # patch row law.
        from . import row_packets  # noqa: PLC0415

        self._row_packets = None
        self._row_packets_reason = (
            (
                None
                if self._route_pins.get("producer") in (None, "local_packets")
                else "A test pin selects another contact-row producer."
            )
            or capable("row_packets", laws)
            or row_packets.decline(self)
        )
        if self._row_packets_reason is None:
            self._row_packets = row_packets.create_owner(self)
            self._row_packets_reason = (
                "Admitted: worlds of one limited articulation and at most one other build packets in place."
                if self._row_packets is not None
                else "The local packet law does not admit this topology."
            )
        self._keyed_rows = None
        if self.laws.regularization_enabled:
            self._keyed_rows_reason = self._parallel_contact_paths_reason
        elif not self._parallel_contact_paths:
            self._keyed_rows_reason = (
                "Keyed rows write the point and patch row laws; torsion and compliance keep ordinary rows."
            )
        elif self._route_pins.get("producer") not in (None, "keyed_rows"):
            self._keyed_rows_reason = "A test pin selects another contact-row producer."
        else:
            from .keyed_rows import create_owner as create_kinetic_rows_owner  # noqa: PLC0415

            self._keyed_rows = create_kinetic_rows_owner(self)
            self._keyed_rows_reason = (
                "Admitted: the in-kernel parallel sweep builds keyed rows for its bounded worlds."
                if self._keyed_rows is not None
                else "Keyed rows need the Jacobi tiers (routes.sweep.tiers) and a plain matrix-free policy."
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
        self._dummy_mf_slot_counter = placeholder(wp.int32, (self.forest.world_count,))
        self._dummy_mf_impulses = placeholder(wp.float32, (1, 1))
        self._dummy_contact_count = placeholder(wp.int32, (1,))
        self._dummy_contact_row_type = placeholder(wp.int32, (1, 1))
        self._reset_dummy_mf_row_type = placeholder(wp.int32, (1, 1), -1)
        self._reset_dummy_mf_row_parent = placeholder(wp.int32, (1, 1), -1)
        self._dummy_contact_row_parent = placeholder(wp.int32, (1, 1), -1)

        # Sticky flags of constraint_capacity_status() (dense and matrix-free rows), the per-world overflow
        # latch and the dropped rows of each family.
        self._arrays.allocate(
            CAPACITY_STATUS, world=self.forest.world_count, status_world=max(self.forest.world_count, 1)
        )
        self._row_dropped_dense = self._row_dropped_all[0]
        self._row_dropped_mf = self._row_dropped_all[1]
        # Constraint-capacity diagnostics are allocated once before CUDA graph
        # capture. The warning path tracks only current-step dropped rows and a
        # one-shot flag per row family; row_watermark additionally accumulates
        # whole-run high-water telemetry for explicit readback.
        self._track_row_capacity = (
            self._row_watermark
            or self.options.warn_constraint_overflow
            or self.laws.contact_torsion_enabled
            or self.options.contact_compliance
        )
        # Dropped-row accounting is unconditional (see _row_dropped_all above);
        # _track_row_capacity now gates only the optional warning/watermark readbacks.
        if self.options.warn_constraint_overflow:
            self._arrays.allocate(OVERFLOW_WARNING)
        else:
            self._arrays.release(OVERFLOW_WARNING)
        # The telemetry words share one array, which constraint_row_watermarks() reads back once.
        self._row_watermark_words = None
        if self._row_watermark:
            self._row_watermark_words = self._arrays.allocate_words(ROW_WATERMARKS)
        else:
            self._arrays.release(ROW_WATERMARKS)

        self._init_double_buffer_stream()

        # Contact torsion's preparation is device-resident on CUDA and on the host on the CPU (the host preparation is
        # the test pin torsion_preparation='host' on CUDA).
        self.contact_torsion_device = bool(
            model.device.is_cuda and self._route_pins.get("torsion_preparation") != "host"
        )
        if self.laws.contact_torsion_enabled and self.contact_torsion_device:
            from .contact_torsion_device import enable_device_torsion  # noqa: PLC0415

            enable_device_torsion(self)

        # Scalar branches take a diagonal-mass group of world-fixed roots over
        # one-coordinate leaves (scalar_branches.py), which the held-tree owner
        # then publishes: the compact diagonal's admission planned them.
        self._scalar_branches, self._scalar_branches_reason = plan_scalar_branches(self)

        held_tree.install(self)
        # Ordinary dynamics take the CPU's drive pass and layout (_stage1_drives: the joint forces and the drives in
        # one launch, K added after the assembly); the per-coordinate drive rows are the held tree's.
        # Their preparation overlaps FK/ID on a stream of its own.
        if self._tree_kinetic_state is None:
            self._parallel_augmented_drive_topology = False
        self._articulation_dynamics_stream = (
            wp.Stream(model.device) if self._parallel_augmented_drive_topology and model.device.is_cuda else None
        )

        # Worlds whose every candidate row separates at the predictor skip row
        # allocation entirely (resolved_worlds.py); the mask is empty when the
        # solver's row law is outside what the classifier evaluates.
        from .resolved_worlds import ResolvedWorlds  # noqa: PLC0415

        classifier = ResolvedWorlds(self)
        self._resolved_worlds_owner = classifier if classifier.enabled else None
        self._resolved_worlds = classifier.resolved
        self._resolved_worlds_reason = classifier.reason

        self._place_dense_response()
        # The route: the row of the route table the class takes. The step plan and the route record read it.
        diagonal = self.routes.case == "K2"
        route = self.routes.finish(
            self.laws.set,
            evaluator="held" if self._tree_kinetic_state is not None else "ordinary",
            producer=contact_row_plan(self).producer,
            packed=self._packed_response,
            # CUDA sweeps the rows matrix-free; the CPU sweeps its assembled rows, interleaved with the free-body
            # rows where a world holds both.
            family="matrix_free"
            if self._matrix_free
            else "split_interleaved"
            if self.forest.has_mixed_contacts
            else "dense_split",
            tier_rows=self._ink_skip_rows(),
            sizes={
                "packed": self._sparse_mass_matrix_size,
                "diagonal": self._sparse_diagonal_response_size if diagonal else None,
                "dense": self._sparse_diagonal_dense_size if diagonal else None,
            },
        )
        # The step's per-stage dispatch tables, which step() and the route record read, and the kernels they launch.
        self._plan = StepPlan.build(route, self)
        self._build_stage_kernels()
        self._setup_contact_update(model)
        # Record the applied options once; get_path_selection() returns detached copies.
        self._applied_options = applied_options(self, SolverFeatherPGS.__init__, requested_options)
        if self.options.contact_update != "sequential":
            for message in contact_update_warnings(route_selection(self)["sweep"]):
                warnings.warn(message, UserWarning, stacklevel=_CONSTRUCTOR_CALLER)

    def _setup_contact_update(self, model):
        """Select the sweeps that solve contact units exactly, allocate their unit storage and build the loop sweeps.

        ``contact_update="unit"`` reaches the CPU's loop sweeps only: its assembled dense rows' sweep and its
        free-body sweep. Every CUDA sweep keeps the sequential law (``routes.sweep.contact_update``).
        ``_contact_unit_patch`` is an internal switch for A/B runs: ``"ring"`` solves a patch region as one ring
        unit, ``"anchor"`` each anchor as a 3x3 unit with the ring's other load held; launches read it.
        """
        unit = self.options.contact_update == "unit" and not self._matrix_free
        self._dense_contact_unit = unit
        self._free_body_contact_unit = bool(unit and self.forest.has_free_rigid_bodies)
        self._contact_unit_patch = "ring"
        units = {"world": self.forest.world_count, "unit_fields": CONTACT_UNIT_STATE_FIELDS}
        units["unit_block"] = self.forest.world_count * RING_BLOCK_SIZE
        if self._dense_contact_unit:
            self._arrays.allocate(DENSE_CONTACT_UNITS, row=self.dense_max_constraints, **units)
        else:
            self._arrays.release(DENSE_CONTACT_UNITS)
        if self._free_body_contact_unit:
            self._arrays.allocate(MF_CONTACT_UNITS, mf_row=self.mf_max_constraints, **units)
        else:
            self._arrays.release(MF_CONTACT_UNITS)
        # The CPU's loop sweeps, one thread per world, under the law chosen above: the dense rows' and the
        # free-body rows'.
        self._pgs_solve_loop_kernel = self._pgs_solve_mf_loop_kernel = None
        if not self._matrix_free:
            self._pgs_solve_loop_kernel = self._sweep_kernel(
                "dense_loop",
                sweep_routes_split.loop_unit_kernel if self._dense_contact_unit else sweep_routes_split.loop_kernel,
            )
        if self.forest.has_free_rigid_bodies and not self._matrix_free:
            self._pgs_solve_mf_loop_kernel = self._sweep_kernel(
                "free_body_loop",
                sweep_routes_split.mf_loop_unit_kernel
                if self._free_body_contact_unit
                else sweep_routes_split.mf_loop_kernel,
            )

    def get_path_selection(self) -> dict:
        """Return the active plans and selection reasons.

        The record is read on the host without device synchronization, except
        under ``pgs_solve_readout``: reading its counters synchronizes once.

        Automatic choices use topology and resource rules, never timing.
        Recreate the solver to change them. The ``reasons`` mapping explains construction-time admission and
        fallbacks; it does not predict dynamic per-world branches.

        ``routes`` records every construction-time owner and variant that decides
        which kernels run: ``routes.row`` the route, the route table's row the
        step plan and the owners' entries read (evaluator, placement case,
        contact-row producer, packed rows, step family, tier rows, the case
        owners' sizes and each case's reason), ``routes.stages`` the step plan's per-size stage routes
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
            "sparse_mass_matrix": "on" if self.routes.row.case == "K1" else "off",
            "cached_dynamics": "on" if cached else "off",
            "publication": "cached" if cached else "ordinary",
            "pgs_kernel": "matrix_free" if self._matrix_free else "loop",
            "cached_prediction": "tree" if self._tree_kinetic_state is not None else None,
            "reasons": {
                "sparse_mass_matrix": self.routes.row.reasons["K1"],
                "cached_dynamics": self._cached_dynamics_selection_reason,
                "publication": "Paired cache owns next-state publication."
                if cached
                else "Ordinary state publication remains active.",
                "pgs_kernel": "CUDA sweeps the rows matrix-free."
                if self._matrix_free
                else "The CPU sweeps the assembled Delassus rows on one thread per world.",
            },
        }
        selection["routes"] = route_selection(self)
        selection["measurements"] = measurement_selection(self)
        selection["options"] = options_selection(self)
        selection["environment"] = copy.deepcopy(self._environment_record)
        return selection

    def prepare_contact_torsion_capture(self, state_in: State, state_out: State) -> None:
        """Prepare experimental torsion rollback buffers before CUDA graph capture.

        Torsion's preparation is device-resident on CUDA. Call
        outside capture with the state layouts used by the graph. Subsequent
        graph replay requires :meth:`validate_contact_torsion` at every batch
        boundary before results are consumed. Eager steps continue to validate
        automatically. This method does not advance physics.
        """
        if not self.laws.contact_torsion_enabled:
            return
        preparation = getattr(self, "_device_torsion", None)
        if preparation is None:
            raise RuntimeError("Contact torsion graph capture requires the device-resident torsion preparation")
        if capturing(self.model.device):
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
            if capturing(self.model.device):
                raise RuntimeError("Validate contact torsion outside CUDA graph capture")
            preparation.validate()

    def _kinematic_state(self, forest: TreePlan) -> tuple[np.ndarray, np.ndarray, np.ndarray, ArmatureFloor]:
        """Derive the kinematic masks and effective joint armature from the body flags, changing nothing.

        The effective armature also holds the least armature that keeps the mass matrix of a deep
        articulation within float32 factorization (:func:`conditioning_armature`).

        The coordinate mask is the rows' prescribed coordinates (:data:`.joint_table.JOINTS`): none takes a row on
        it alone.

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

        if getattr(self, "_sparse_velocity_limits", False) and np.any(dof_mask):
            raise RuntimeError(
                "Sparse response gained prescribed DOFs; reconstruct the solver and recapture its CUDA graph."
            )

        if self.forest is not None:
            selected = np.nonzero(self.forest.prescribed_articulation_host != 0)[0]
            for articulation in selected:
                dof_start = int(self.forest.articulation_dof_start_host[articulation])
                dof_count = int(self.forest.articulation_dof_count_host[articulation])
                if not np.all(dof_mask[dof_start : dof_start + dof_count] != 0):
                    raise RuntimeError(
                        "A prescribed-response articulation changed kinematic membership; "
                        "reconstruct the solver and recapture its CUDA graph."
                    )
        armature, floor = conditioning_armature(model, armature, forest)
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
            self._kinematic_state(self.forest.joint_forest)
            if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES)
            else None
        )
        if self.laws.friction_anchors_enabled:
            # The next captured build sorts its pairs again rather than reusing the
            # sorted contact frame across a model change.
            self.laws.friction_patches.forget_frame()
        if self.laws.friction_anchors_enabled and flags & ModelFlags.SHAPE_PROPERTIES:
            # Geometry edits retire affected material points; unrelated shape
            # properties keep their history and live materials are checked per step.
            self.laws.friction_patches.update_geometry(self.model)
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
        if self.options.contact_compliance:
            self._compliant_contacts = None
            self._compliant_prepared = False
            self.compliance_contact_count = 0
            self.compliance_skipped_contact_count = 0
        if world_mask is not None and world_mask.shape[0] != self.forest.world_count:
            raise ValueError(
                f"world_mask has length {world_mask.shape[0]}, expected {self.forest.world_count} (one entry per world)."
            )
        if self.forest.world_count == 0:
            return

        if getattr(self, "_tree_kinetic_state", None) is not None:
            self._tree_kinetic_state.invalidate(world_mask)

        wp.launch(
            _reset_solver_status,
            dim=max(self.forest.world_count, self.model.articulation_count),
            inputs=[world_mask, self.forest.art_to_world, self._mass_update_requested, self.constraint_overflow],
            device=self.model.device,
        )
        if self.laws.friction_anchors_enabled:
            # Use the world associated with each stored patch anchor.
            self.laws.friction_patches.reset(world_mask)
        if self._fk_id_cache_enabled:
            wp.launch(
                invalidate_articulation_fk_id_cache,
                dim=self.model.articulation_count,
                inputs=[world_mask, self.forest.art_to_world],
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
        if self.forest.world_count == 0:
            return
        prev_mf_impulses = self._ws_prev_mf_impulses
        if not self.options.pgs_warmstart and prev_mf_impulses is None:
            return

        dense_impulses = self.impulses if self.options.pgs_warmstart else self._dummy_contact_impulses
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
            dim=self.forest.world_count,
            inputs=[world_mask, dense_impulses, prev_mf_impulses, prev_mf_row_type, prev_mf_row_parent],
            device=self.model.device,
        )
        if self.options.pgs_warmstart and self._ws_prev_dense_impulses is not None:
            # Neutralize the dense identity-matched carry for reset worlds
            # (the kernel is width-generic: it zeroes impulses and writes -1
            # into the row-type table).
            wp.launch(
                reset_world_warmstart_buffers,
                dim=self.forest.world_count,
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

    def set_loop_joint_enabled(self, joint: int, enabled: bool) -> None:
        """Enable or disable the connect rows of a loop-closing BALL joint at runtime.

        The closure keeps its allocated capacity, so toggling is CUDA-graph-safe and
        takes effect on the next :meth:`step`. Closures disabled through
        :attr:`~newton.Model.joint_enabled` at construction still count as capacity.

        Args:
            joint: Model joint index of a loop-closing BALL joint.
            enabled: ``True`` to enforce the closure, ``False`` to release it.
        """
        self.laws.set_loop_joint_enabled(joint, enabled)

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
        self.laws.set_loop_joint_anchors(joint, parent_anchor, child_anchor)

    def loop_joint_anchors(self, joint: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the current ``(parent_anchor, child_anchor)`` of a loop-closing BALL joint [m]."""
        return self.laws.loop_joint_anchors(joint)

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
        """The bilateral elimination's tables (:func:`.response_builds_rows.elimination`).

        Per articulation that owns mimic or connect rows: an index into the elimination's buffers, which hold up to
        the model's largest block (an articulation's mimic rows plus three per closure; a closure to a prescribed
        parent carries the parent's velocity in its right-hand side). The buffers are allocated once here
        (CUDA-graph-safe); the slot lists and the factor are rebuilt every step on device, since the rows' slots and
        ``J_B`` depend on the state.
        """
        self._preelim_count = 0
        self._preelim_active = False
        self._elimination = None
        if not self.options.enable_bilateral_preelimination or not (self.laws.connect_count or self.laws.mimic_count):
            return
        if not self._matrix_free:
            warnings.warn(
                "SolverFeatherPGS: enable_bilateral_preelimination requires CUDA; falling back to "
                "iterative mimic/connect rows.",
                stacklevel=2,
            )
            return
        # Per-articulation block rows (upper bound: every valid constraint).
        counts = np.zeros(model.articulation_count, dtype=np.int64)
        if self.laws.mimic_count:
            counts += np.diff(self.laws.mimic_art_start_np)
        if self.laws.connect_count:
            counts += 3 * np.bincount(self.laws._connect_art.numpy(), minlength=model.articulation_count)
        arts = np.flatnonzero(counts)
        if not arts.size:
            return
        rows = int(counts.max())
        self._elimination = response_builds_rows.elimination(rows)
        device = model.device
        n_pe = len(arts)
        art_to_idx = np.full(model.articulation_count, -1, dtype=np.int32)
        art_to_idx[arts] = np.arange(n_pe)
        self._preelim_art_to_idx = wp.array(art_to_idx, dtype=wp.int32, device=device)
        self._arrays.allocate(
            PREELIMINATION,
            preelim=n_pe,
            preelim_row=n_pe * rows,
            preelim_block=n_pe * rows * rows,
        )
        self._preelim_count = n_pe
        self._preelim_active = True

    def _stage4_bilateral_preelim(self) -> None:
        """Eliminate the bilateral blocks on the packed rows (:func:`.response_builds_rows.elimination`).

        Form each articulation's block ``G = Z_B^T Z_B`` and its pivoted factor, then correct every other row, its
        right-hand side and its diagonal. Runs once the right-hand sides are final (bias and restitution) and
        before the warm start, whose impulses the corrected rows then carry.
        """
        dense = int(self._packed_dense)
        for size in self.forest.size_groups:
            wp.launch(
                self._elimination.setup,
                dim=self.forest.n_arts_by_size[size],
                inputs=[
                    field_group.bind(Preelim, solver=self),
                    self.forest.group_to_art[size],
                    self.forest.art_to_world,
                    self.forest.articulation_world_dof_offset,
                    dense,
                    self.laws.mimic_slot if self.laws.mimic_slot is not None else self._dummy_is_free_rigid,
                    self.laws._mimic_art_start if self.laws.mimic_count else self._dummy_is_free_rigid,
                    self.laws._mimic_art_list if self.laws.mimic_count else self._dummy_is_free_rigid,
                    int(self.laws.mimic_count or 0),
                    self.laws.connect_slot if self.laws.connect_slot is not None else self._dummy_is_free_rigid,
                    self.laws._connect_art if self.laws.connect_count else self._dummy_is_free_rigid,
                    int(self.laws.connect_count or 0),
                    self._packed_J,
                    self._packed_Z,
                    size,
                ],
                device=self.model.device,
            )
        wp.launch(
            self._elimination.correct,
            dim=self.forest.world_count * self.dense_max_constraints,
            inputs=[
                field_group.bind(Preelim, solver=self),
                self._row_sides,
                self.forest.articulation_response_dof_count,
                self.forest.articulation_world_dof_offset,
                dense,
                self.constraint_count,
                self._local_solve_owner,
                self.row_cfm,
            ],
            outputs=[self._packed_Z, self.rhs, self.diag],
            device=self.model.device,
        )

    def _stage6_project_bilateral_velocity(self) -> None:
        """Enforce J_B v_out = -b_B once, after v_out holds the predictor and every warm start: the block weights, then
        ``v -= P^T L^-T Z_B y`` (:func:`.packed_rows.project_kernel`)."""
        dense = int(self._packed_dense)
        for size in self.forest.size_groups:
            wp.launch(
                self._elimination.weights,
                dim=self.forest.n_arts_by_size[size],
                inputs=[
                    field_group.bind(Preelim, solver=self),
                    self.forest.group_to_art[size],
                    self.forest.art_to_world,
                    self.forest.articulation_dof_start,
                    self.forest.articulation_world_dof_offset,
                    dense,
                    self._packed_J,
                    self.rhs,
                    self.v_out,
                    size,
                ],
                device=self.model.device,
            )
            wp.launch(
                packed_rows.project_kernel(self._elimination.rows),
                dim=self.forest.n_arts_by_size[size],
                inputs=[
                    self.L_by_size[size],
                    self.forest.group_to_art[size],
                    self.forest.art_to_world,
                    self.forest.articulation_dof_start,
                    self.forest.articulation_world_dof_offset,
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
            and self.options.enable_joint_limits
            and self.options.enable_contact_friction
            and not self.options.enable_joint_velocity_limits
            and not self.options.pgs_warmstart
            and not self.options.pgs_warmstart
            and not self._preelim_active
            and not self.forest.has_free_rigid_bodies
            and not self._debug_buffers_enabled
            and not self.laws.regularization_enabled
            and self.laws.mimic_count == 0
            and self.laws.connect_count == 0
        )
        if not supported:
            return

        candidate_sizes = self.forest.diagonal_mass_sizes.intersection(self.laws.joint_limit_sizes)
        if not candidate_sizes:
            return
        response_dof_count = self.forest.response_dof_count
        articulation_world = self.forest.articulation_world
        articulation_dof_start = self.forest.articulation_dof_start_host
        pairs = _world_diagonal_pairs(articulation_world, response_dof_count, self.forest.world_count, candidate_sizes)
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
        valid = self.laws.joint_limit_q_index_host[dof] >= 0
        limited = np.bincount(art[valid], minlength=diagonal_articulations.size) > 0
        self._fused_diagonal_limit_dof_mask_host[dof[limited[art]]] = valid[limited[art]]
        # Add each size where a world first has it, as the per-world set did.
        selected_sizes: set[int] = {int(size) for size in host_index.ordered_unique(diagonal_size[limited])}
        self._fused_diagonal_joint_limit_sizes = frozenset(selected_sizes)
        self._fused_diagonal_joint_limits = bool(selected_sizes)

    def _estimate_dense_internal_rows_per_world(self, model) -> int:
        if not model.articulation_count or not model.joint_count or self.forest.art_to_world is None:
            return 0

        has_position_limit_rows = self.options.enable_joint_limits and model.joint_dof_count > 0
        has_velocity_limit_rows = (
            self.options.enable_joint_velocity_limits
            and model.joint_dof_count > 0
            and not np.isinf(self.options.velocity_limit_activation_fraction)
        )
        has_mimic_rows = self.laws.mimic_count > 0
        has_connect_rows = self.laws.connect_count > 0
        if not (has_position_limit_rows or has_velocity_limit_rows or has_mimic_rows or has_connect_rows):
            return 0

        art_to_world = self.forest.art_to_world.numpy().astype(np.int32, copy=False)
        per_world = np.zeros(max(int(self.forest.world_count), 1), dtype=np.int64)

        if has_mimic_rows and self.laws.mimic_world_np is not None:
            valid_worlds = self.laws.mimic_world_np[self.laws.mimic_valid_np != 0]
            in_range = valid_worlds[(valid_worlds >= 0) & (valid_worlds < per_world.shape[0])]
            np.add.at(per_world, in_range, 1)
        if has_connect_rows and self.laws.connect_world_np is not None:
            valid_worlds = self.laws.connect_world_np[self.laws.connect_valid_np != 0]
            in_range = valid_worlds[(valid_worlds >= 0) & (valid_worlds < per_world.shape[0])]
            np.add.at(per_world, in_range, 3)

        articulation_start = model.articulation_start.numpy().astype(np.int32, copy=False)
        joint_type = model.joint_type.numpy().astype(np.int32, copy=False)
        joint_qd_start = model.joint_qd_start.numpy().astype(np.int32, copy=False)
        joint_dof_dim = model.joint_dof_dim.numpy().astype(np.int32, copy=False)

        joint_limit_lower_arr = getattr(model, "joint_limit_lower", None)
        joint_limit_upper_arr = getattr(model, "joint_limit_upper", None)
        if has_position_limit_rows and (joint_limit_lower_arr is None or joint_limit_upper_arr is None):
            has_position_limit_rows = False
        joint_velocity_limit_arr = getattr(model, "joint_velocity_limit", None)
        if has_velocity_limit_rows and joint_velocity_limit_arr is None:
            has_velocity_limit_rows = False
        if not (has_position_limit_rows or has_velocity_limit_rows):
            return 0

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
        if has_position_limit_rows:
            fuse_position_limits = np.isin(
                self.forest.response_dof_count[art], list(self._fused_diagonal_joint_limit_sizes)
            )
            rows += ~fuse_position_limits & np.isfinite(joint_limit_lower[dof])
            rows += ~fuse_position_limits & np.isfinite(joint_limit_upper[dof])
        if has_velocity_limit_rows:
            rows += 2 * (joint_velocity_limit[dof] > 0.0)
        world = art_to_world[art]
        in_range = (world >= 0) & (world < per_world.size)
        np.add.at(per_world, world[in_range], rows[in_range])

        return int(np.max(per_world)) if per_world.size else 0

    def _setup_sparse_mass_matrix(self, model: Model) -> None:
        """Select sparse articulation factors with independent rigid-body response blocks."""
        self._sparse_mass_matrix_size = None
        self._sparse_velocity_limits = False
        self._sparse_prepare_contacts = False
        self._sparse_row_support = _SparseRowSupport()
        self._sparse_contact_triplets = False
        # Route details for get_path_selection(), set where the sparse owner is admitted.
        self._sparse_active_rows = False
        self._sparse_active_rows_reason = "Native sparse factors are not selected."
        self._sparse_contact_triplets_reason = "Native sparse factors are not selected."
        # The admitted contact units' block records.
        self._sparse_contact_records = None
        self._sparse_max_row_support = None
        # Case K1, a single tree.
        velocity_candidate = self.options.enable_joint_velocity_limits and not self.forest.has_free_rigid_bodies
        declined = (
            self.routes.taken("K1")
            or capable("packed", self.laws.set)
            or first_failed(
                (
                    (
                        self._route_pins.get("storage") != "dense",
                        "A test pin keeps dense storage (storage='dense').",
                    ),
                    (not self._route_pins.get("tiers"), "A test pin builds the Jacobi tiers (tiers) on dense storage."),
                    (
                        self._parallel_augmented_drive_topology,
                        "Drive topology is not supported by native sparse factors.",
                    ),
                    (
                        not self.options.enable_joint_velocity_limits or velocity_candidate,
                        "Velocity-limit rows need worlds without free bodies on native sparse factors.",
                    ),
                    (
                        not self.options.pgs_warmstart or not self.forest.has_free_rigid_bodies,
                        "PGS warm starting with free rigid bodies requires the ordinary factor/row composition.",
                    ),
                    (
                        not self.laws.mimic_count or velocity_candidate,
                        "Mimic rows require the sparse velocity-row composition.",
                    ),
                )
            )
        )
        if declined is not None:
            self.routes.reasons["K1"] = declined
            return
        # Decide ownership before omitting dense buffers. Explicit research
        # recipes retain their own admission and fallback, including when their
        # requested owner later declines the topology. They cannot acquire
        # dense storage after this sparse allocation decision.
        if self.forest.has_free_rigid_bodies:
            # A world that pairs an articulation with free rigid bodies is served
            # by the held-tree owner and the paired response on the dense route
            # (measured: Kuka lift 12.7 ms there against 15.5 on sparse factors).
            # Both friction laws: patch friction walks contact units in the paired
            # sweep and keeps the resolved-world classifier. Native sparse factors never serve free rigid bodies.
            self.routes.reasons["K1"] = (
                "Worlds with free rigid bodies keep the dense route for the held-tree owner and paired response."
            )
            return
        response_dofs = self.forest.response_dof_count
        if velocity_candidate and (self.forest.has_prescribed_response or np.any(self._kinematic_dof_mask_host)):
            self.routes.reasons["K1"] = "Sparse velocity rows do not support prescribed-coordinate topology."
            return
        free_rigid = self.forest.is_free_rigid_host != 0
        articulated = np.flatnonzero((response_dofs > 0) & ~free_rigid)
        sizes = np.unique(response_dofs[articulated])
        if len(sizes) != 1:
            self.routes.reasons["K1"] = "Native sparse factors require one articulated response DOF size."
            return
        size = int(sizes[0])
        # The endpoint support uses one 64-bit ancestry mask, not a robot-specific DOF limit.
        if (
            size > 64
            or size not in self.forest.crba_dof_joint_offset_by_size
            or size in self.forest.diagonal_mass_sizes
        ):
            self.routes.reasons["K1"] = (
                "Articulated topology needs at most 64 response DOFs, a CRBA mapping and non-diagonal inertia."
            )
            return
        articulated_counts = np.bincount(self.forest.articulation_world[articulated], minlength=self.forest.world_count)
        articulation_start = model.articulation_start.numpy()
        joint_ancestor = model.joint_ancestor.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        art = int(self.forest.group_to_art[size].numpy()[0])
        start = int(articulation_start[art])
        end = int(self.forest.articulation_joint_end_host[art])
        tree = TreePlan.build(
            local_parents(joint_ancestor, [start], end - start)[0], np.diff(joint_qd_start[start : end + 1])
        )
        plan = tree.factor
        # Velocity rows can benefit from factor-coordinate response even for a
        # serial chain; this owner does not require zeros in the mass factor.
        if plan.dof_count != size or (not velocity_candidate and plan.nonzero_count >= size * (size + 1) // 2):
            self.routes.reasons["K1"] = "Topology has no usable sparse factor layout for this row policy."
            return
        # Native sparse factors are the held tree's representation: a candidate only where it takes their group.
        from . import held_tree  # noqa: PLC0415

        held = held_tree.sparse_decline(self, size, plan, velocity_limits=velocity_candidate)
        if held is not None:
            self.routes.reasons["K1"] = f"Only the held tree evaluates native sparse factors. {held}"
            return

        mimic_dofs = None
        if self.laws.mimic_count:
            mimic_indices = self.laws._mimic_art_list.numpy()
            mimic_arts, _ = host_index.expand(np.diff(self.laws.mimic_art_start_np))
            starts = self.forest.articulation_dof_start_host[mimic_arts]
            mimic_dofs = (
                self.laws._mimic_dof0.numpy()[mimic_indices] - starts,
                self.laws._mimic_dof1.numpy()[mimic_indices] - starts,
            )
        body_masks, max_support = row_support(self.forest, model, plan, articulated, articulated_counts, mimic_dofs)
        if not max_support:
            self.routes.reasons["K1"] = "Topology has no responsive row support."
            return
        # The sweep's shared working set (sweep_routes_sparse.sparse_staging) against the device's static budget:
        # each shared-memory choice below reads its own sweep's working set, and the route is admitted by the working
        # set of the sweep it launches, which the sweep's factory checks the same way.
        budget = arch_budget(model.device.arch)
        staging = functools.partial(
            sweep_routes_sparse.sparse_staging,
            self.dense_max_constraints,
            self.forest.max_world_dofs,
            int(max_support),
            velocity_limits=velocity_candidate,
            linked_rows=self.laws.friction_anchors_enabled,
        )
        # Patch friction keeps the packed lanes of small worlds, so size them from the
        # full sweep before the active-row scan claims whole warps.
        small_world_lanes = self._sparse_small_world_lanes(staging, budget)
        active_rows, active_rows_reason = self._admit_sparse_active_rows(
            velocity_candidate=velocity_candidate,
            small_world_lanes=small_world_lanes,
            fits=lambda: staging(active_rows=True).fits(budget),
        )
        shape = (self.forest.world_count, self.dense_max_constraints, max_support)
        offsets = self.forest.articulation_world_dof_offset.numpy()
        support_classes = None
        if (
            self._route_pins.get("row_support", True)
            and not self.laws.mimic_count
            and np.all(articulated_counts == 1)
            and size == self.forest.max_world_dofs
            and np.all(offsets[articulated] == 0)
        ):
            classes = tree.support_classes(body_masks, max_support)
            if classes is not None:
                table_bytes = sum(
                    getattr(classes, name).nbytes for name in ("body_class", "pair_class", "class_dof", "limit_class")
                )
                table_bytes += 4 * self.forest.world_count * self.dense_max_constraints
                # Small batches with many support patterns can favor direct indices.
                if table_bytes < 4 * int(np.prod(shape)):
                    support_classes = classes
        # Unsupported mixed worlds retain explicit row indices and their existing kernels.
        use_support_table = support_classes is not None
        # Reuse the local solver's small-world packing without changing this
        # owner's row representation, sweep policy or velocity-limit phase.
        lanes = small_world_lanes if small_world_lanes and not active_rows else 32
        triplet_capacity = self.dense_max_constraints // 3
        contact_triplets, contact_triplets_reason = self._admit_sparse_contact_triplets(
            active_rows=active_rows,
            velocity_candidate=velocity_candidate,
            use_support_table=use_support_table,
            lanes=lanes,
            fits=lambda: (
                triplet_capacity > 0
                and staging(active_rows=active_rows, use_support_table=use_support_table, contact_triplets=True).fits(
                    budget
                )
            ),
        )
        # Large resident workspaces regressed whole-step timing in measured
        # small systems. Keep the global-row consumer when the cache is large.
        local_factors = bool(
            lanes < 32
            and staging(lanes_per_world=lanes, local_factors=True).bytes <= min(budget, _SPARSE_LOCAL_WORKSPACE_BYTES)
        )
        sweep = {
            "active_rows": active_rows,
            "use_support_table": use_support_table,
            "contact_triplets": contact_triplets,
            "lanes_per_world": lanes,
            "local_factors": local_factors,
        }
        solve = staging(**sweep)
        if not solve.fits(budget):
            self.routes.reasons["K1"] = (
                f"Row capacity needs {solve.bytes} bytes of shared memory for sparse solving; limit is {budget}."
            )
            return
        # The factor build's warps share a block within the offline static budget.
        shared_limit = factor_budget(model.device)
        shared_bytes_per_warp = 4 * (2 * plan.nonzero_count + 6 * size)
        self._sparse_factor_warps_per_block = min(
            _CRBA_CHOLESKY_WARPS_PER_BLOCK,
            shared_limit // shared_bytes_per_warp,
        )
        if self._sparse_factor_warps_per_block < 1:
            self.routes.reasons["K1"] = (
                f"One sparse factor warp needs {shared_bytes_per_warp} bytes of shared memory; limit is {shared_limit}."
            )
            return

        self.routes.reasons["K1"] = (
            f"Automatic sparse selection: {size}-DOF topology and row policy admitted; "
            f"solve uses {solve.bytes}/{budget} bytes of shared memory."
        )
        self.routes.claim("K1")
        self._sparse_mass_matrix_size = size
        self._sparse_active_rows = active_rows
        self._sparse_active_rows_reason = active_rows_reason
        self._sparse_contact_triplets = contact_triplets
        self._sparse_contact_triplets_reason = contact_triplets_reason
        self._sparse_solve_lanes = lanes
        self._sparse_local_factors = local_factors
        self._sparse_max_row_support = int(max_support)
        self._sparse_velocity_limits = velocity_candidate
        # The sparse response kernel prepares the contact rows' metadata, CFM, bias and restitution.
        self._sparse_prepare_contacts = True
        # The sparse producers overwrite every live value and retain factors by the mass mask.
        self._double_buffer = False
        self._sparse_mass_matrix_plan = plan
        self._sparse_mass_matrix_indices = factor_indices(plan, model.device)
        self._sparse_body_dof_mask = wp.array(body_masks, dtype=wp.uint64, device=model.device)
        self._arrays.allocate(SPARSE_FACTOR, group=self.forest.n_arts_by_size[size], nonzero=plan.nonzero_count)
        if use_support_table:
            self._sparse_row_support = row_supports(
                support_classes, self.forest.world_count, self.dense_max_constraints, model.device
            )
        if contact_triplets:
            # Static storage for graph capture: the builder writes the block records of each step's
            # contact units; linked sweeps add each anchor's ring members once per call.
            width = CONTACT_RECORD_FLOATS + (CONTACT_RING_FLOATS if self.laws.friction_anchors_enabled else 0)
            self._arrays.allocate(
                SPARSE_CONTACT_RECORDS, world=self.forest.world_count, triplet=triplet_capacity, record=width
            )
        index_shape = (1, 1, 1) if use_support_table else shape
        self._arrays.allocate(
            SPARSE_ROWS,
            index_world=index_shape[0],
            index_row=index_shape[1],
            index_support=index_shape[2],
            world=self.forest.world_count,
            row=self.dense_max_constraints,
            support=max_support,
            world_dof=self.forest.max_world_dofs,
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
        self._pgs_solve_sparse_kernel = self._sweep_kernel(
            "sparse_pgs",
            sweep_routes_sparse.sparse_kernel,
            self.dense_max_constraints,
            self.forest.max_world_dofs,
            max_support,
            **sweep,
            velocity_limits=self._sparse_velocity_limits,
            warm_start=self.options.pgs_warmstart,
            linked_rows=self.laws.friction_anchors_enabled,
            device_arch=model.device.arch,
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
        budget = int(self.options.pgs_iterations if readout_budget is None else readout_budget)
        self._solve_readout.register(
            kernel,
            route,
            readout=policy.readout if policy.counts else None,
            world_count=int(self.forest.world_count),
            budget=max(budget, 1),
            device=self.model.device,
        )
        return built

    def _sparse_small_world_lanes(self, staging, budget: int) -> int:
        """Return the lanes per world (8 or 16) that pack a small complete sparse sweep, or 0 for whole warps.

        Args:
            staging: The sweep's working set for its options (:func:`.sweep_routes_sparse.sparse_staging`).
            budget: Static shared memory available to one block [B].
        """
        if self.forest.max_world_dofs > 16 or self.dense_max_constraints > 64:
            return 0
        lanes = 8 if self.forest.max_world_dofs <= 8 else 16
        return lanes if staging(lanes_per_world=lanes).fits(budget) else 0

    def _admit_sparse_active_rows(
        self, *, velocity_candidate: bool, small_world_lanes: int, fits: Callable[[], bool]
    ) -> tuple[bool, str]:
        """Decide whether the native sparse sweep selects its normal and limit rows at sweep entry, and why.

        Under this law a zero-impulse row that a sweep's earlier updates violate waits for
        the next sweep. Patch friction (``friction_anchor_beta > 0``) adopts it: a friction
        pair is visited while any normal of its patch ring is selected or the pair holds
        impulse, which skips only pairs whose update would be a no-op. Small worlds keep
        their packed lanes. Under either law, a scan whose row bitsets exceed shared memory
        leaves the full ordered sweep.

        Args:
            velocity_candidate: Whether the sweep carries joint velocity-limit rows.
            small_world_lanes: Lanes per world of a packed small-world sweep, 0 for whole warps.
            fits: Whether the active-row scan's working set fits shared memory, evaluated last.
        """
        if not self._route_pins.get("active_rows", True):
            return False, "A test pin keeps the full ordered sweep (active_rows=False)."
        if velocity_candidate:
            return False, "Velocity-limit rows keep full ordered sweeps."
        if self.laws.friction_anchors_enabled and small_world_lanes:
            return False, (
                f"Patch friction (friction_anchor_beta > 0) keeps the packed {small_world_lanes}-lane full sweep "
                "of small worlds."
            )
        if not fits():
            return False, "The full ordered sweep: the active-row scan's row bitsets exceed shared memory."
        if not self.laws.friction_anchors_enabled:
            return True, "Point-contact normal and limit rows are selected at sweep entry."
        return True, (
            "Normal and limit rows are selected at sweep entry; a patch friction pair (friction_anchor_beta > 0) "
            "while any normal of its ring is selected or the pair holds impulse."
        )

    def _admit_sparse_contact_triplets(
        self,
        *,
        active_rows: bool,
        velocity_candidate: bool,
        use_support_table: bool,
        lanes: int,
        fits: Callable[[], bool],
    ) -> tuple[bool, str]:
        """Decide whether the sparse sweep fuses contact triplets, and why: wherever the row structure admits them.

        Point friction fuses them in the active-row scan. Patch friction fuses its anchor
        units, pooling their ring loads as the ordinary pair update does, in the active
        scan or else in the full ordered sweep; velocity-limit, free-body and packed
        small-world sweeps keep ordinary updates. A fused triplet is a contact unit
        on its Gram block (:func:`.sweep_routes_sparse.contact_units`): in exact arithmetic
        the ordinary sweep's normal and pair updates, in one reduction and one apply.

        Args:
            active_rows: Whether the sweep selects its rows at sweep entry.
            velocity_candidate: Whether the sweep carries joint velocity-limit rows.
            use_support_table: Whether rows read the shared support table.
            lanes: Lanes per world of the sweep.
            fits: Whether contact units fit the row capacity and shared memory, evaluated last.
        """
        if "contact_units" in self._route_pins and self._route_pins["contact_units"] != "triplets":
            return False, "A test pin keeps unfused contact rows (contact_units=None)."
        if not active_rows and not self.laws.friction_anchors_enabled:
            return False, "Triplets need the active-row scan."
        if not active_rows and (velocity_candidate or self.forest.has_free_rigid_bodies):
            return False, "Velocity-limit and free-body sweeps keep ordinary updates."
        if lanes < 32:
            return False, f"Small worlds keep their packed {lanes}-lane full sweep."
        if not use_support_table:
            return False, "Triplets need the row support table."
        if not fits():
            return False, "Row capacity or shared memory leaves no room for contact units."
        if self.laws.friction_anchors_enabled:
            sweep = "active-row scan" if active_rows else "full ordered sweep"
            where = f"patch anchor units pool their ring loads in the {sweep}"
        else:
            where = "point contacts fuse in the active-row scan"
        return True, f"Fused wherever the rows admit them: {where}."

    def _detect_jy_world_identity(self) -> bool:
        """Return whether host-side group and compact world J/Y layouts are identical."""
        if len(self.forest.size_groups) != 1 or self.forest.art_to_world is None:
            return False
        size = self.forest.size_groups[0]
        if self.forest.n_arts_by_size[size] != self.forest.world_count:
            return False
        articulation_world = self.forest.articulation_world
        group_to_art = np.flatnonzero(self.forest.response_dof_count == size)
        expected_world = np.arange(len(group_to_art), dtype=articulation_world.dtype)
        return np.array_equal(articulation_world[group_to_art], expected_world)

    def _select_paired_response(self) -> tuple[int, int, np.ndarray] | None:
        """Select one complete robot/free-body response pair per world."""
        if not self._hinv_jt_writes_world or self._hinv_jt_tiled_writes_group or 6 not in self.forest.size_groups:
            return None
        if self._execution_plan.use_tiled_hinv_jt(6):
            return None

        response_dofs = self.forest.response_dof_count
        articulation_world = self.forest.articulation_world
        is_free_rigid = self.forest.is_free_rigid_host
        # Each world's response articulations, in ascending order.
        responding = np.flatnonzero(response_dofs > 0)
        world_bounds, order = host_index.group_by(articulation_world[responding], self.forest.world_count)
        responding = responding[order].tolist()
        response_by_world = [responding[world_bounds[w] : world_bounds[w + 1]] for w in range(self.forest.world_count)]

        secondary_group_arts = np.flatnonzero(response_dofs == 6)
        secondary_group_index = {int(art): group for group, art in enumerate(secondary_group_arts)}
        for primary_size in sorted(self._paired_response_candidate_sizes, reverse=True):
            primary_arts = np.flatnonzero(response_dofs == primary_size)
            if len(primary_arts) != self.forest.world_count:
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

        response_dofs = self.forest.response_dof_count
        articulation_world = self.forest.articulation_world
        is_free_rigid = self.forest.is_free_rigid_host
        offsets = self.forest.articulation_world_dof_offset.numpy()
        group_indices = self.forest.art_group_idx.numpy()
        pairs = _world_diagonal_pairs(
            articulation_world, response_dofs, self.forest.world_count, self._fused_diagonal_joint_limit_sizes
        )
        if pairs is None or self.forest.world_count == 0:
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
        forest = self.forest.joint_forest
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

        self._diagonal_inverse_mass = (
            wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
            if self.forest.diagonal_mass_sizes
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
        if model.articulation_count and model.joint_count and self.forest.articulation_max_dofs:
            self._arrays.allocate(
                AUGMENTED_DRIVE_ROWS,
                articulation=model.articulation_count,
                drive_row=model.articulation_count * self.forest.articulation_max_dofs,
            )
        self._parallel_augmented_drive_topology = False
        self._augmented_drive_row_by_dof = wp.full(
            (max(1, int(model.joint_dof_count)),), -1, dtype=wp.int32, device=model.device
        )
        self._augmented_drive_q_index_by_dof = None
        self._augmented_drive_target_index_by_dof = None
        if not eligible or self.forest.has_loop_joints or not model.joint_dof_count:
            return

        dof_count = int(model.joint_dof_count)
        # Every DOF must lie in exactly one articulation's DOF range.
        range_start = self.forest.articulation_dof_start_host.astype(np.int64)
        owner, local_dof = host_index.expand(self.forest.articulation_dof_count_host)
        owned_dof = range_start[owner] + local_dof
        if np.any(np.bincount(owned_dof[owned_dof < dof_count], minlength=dof_count)[:dof_count] != 1):
            return

        joint_type = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_target_q_start = model.joint_target_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        articulation_start = model.articulation_start.numpy()
        max_dofs = self.forest.articulation_max_dofs

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

    def _memset_clears_jacobian(self, size: int) -> bool:
        """Whether the memset stream clears a size group's active Jacobian rows between uses (double-buffered on CUDA):
        not under the matrix-free sweeps (their producers write the packed rows' sides whole), nor for the
        sparse-diagonal response size (no dense Jacobian) or the compact boundary's six coordinates (it clears its
        rows itself); :attr:`.step_plan.StepPlan.jacobian_clear_sizes` adds the owners of a private Jacobian."""
        return not (
            self._mf_sweeps
            or (self._sparse_diagonal_contact_solve and size == self._sparse_diagonal_response_size)
            or (self._compact_contact_boundary and size == 6)
        )

    def _allocate_buffers(self, model):
        self.H_by_size = {}
        self.L_by_size = {}
        self.J_by_size = {}
        self.Y_by_size = {}
        self.diag_by_size = {}
        self.R_by_size = {}
        self.tau_by_size = {}
        self.qdd_by_size = {}
        if not self.forest.size_groups:
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

        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            # Compact and sparse sizes keep (1, 1, 1) placeholders of the factor and the response.
            compact_diagonal_mass = size in (self._compact_diagonal_mass_size, self._sparse_mass_matrix_size)
            sparse_response = self._sparse_mass_matrix_size is not None or (
                self._sparse_diagonal_contact_solve and size == self._sparse_diagonal_response_size
            )
            factor = 1 if compact_diagonal_mass else 0
            # The matrix-free sweeps' producers write the packed rows (_jacobian_rows): no size group's Jacobian.
            response = 1 if sparse_response or self._mf_sweeps else 0
            # The matrix-free sweeps read the packed rows and the sparse-diagonal owner whitens its dense rows in place
            # (SparsePairZ): no route reads a size group's response.
            whitened = self._sparse_diagonal_contact_solve and size == self._sparse_diagonal_dense_size
            y = 1 if sparse_response or self._mf_sweeps or whitened else 0
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
            # Double buffering gives the mass matrix and Jacobian of a size group two sets; the second Jacobian only
            # where the memset stream clears the rows between uses, else both sets share one.
            systems = (
                zip(self._H_bufs, self._J_bufs, strict=True) if double_buffered else ((self.H_by_size, self.J_by_size),)
            )
            for index, (H_by_size, J_by_size) in enumerate(systems):
                shared = index > 0 and not self._memset_clears_jacobian(size)
                one = {"response_art": 1, "response_row": 1, "response_size": 1} if shared else {}
                system = arrays.allocate(SIZE_SYSTEM, owner={}, **{**sizes, **one})
                H_by_size[size] = system["H"]
                J_by_size[size] = self._J_bufs[0][size] if shared else system["J"]
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
        if self.options.pgs_warmstart:
            contact_fields.append(ContactField("_ws_prev_contact_normal", wp.vec3))
        else:
            self._ws_prev_contact_normal = None
        self._contact_rows = self._contact_pool.rows(contact_fields, on_rows=self._on_contact_rows)
        # The friction patch state and the velocity-limit slots.
        self.laws.allocate(model, self._contact_pool, self._friction_patch_phi)
        # Allocated once here: per-step allocation is forbidden under CUDA graph capture.
        arrays.allocate(WORLD_ROW_STATE, world=self.forest.world_count)
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
        arrays.allocate(CONTACT_RUNS, world=self.forest.world_count, run_block=run_blocks)

    def _allocate_world_buffers(self, model):
        """Allocate world-level constraint system buffers for multi-articulation support."""
        if self.forest.world_count == 0:
            return

        arrays = self._arrays
        # The widest world's DOFs exist where a world response mapping was built (the routes that read them).
        world_dof = self.forest.max_world_dofs
        sizes = {"world": self.forest.world_count, "row": self.dense_max_constraints, "world_dof": world_dof}
        # Per-world constraint matrices and vectors. Matrix-free never assembles C.
        if not self._matrix_free:
            arrays.allocate(DENSE_DELASSUS, **sizes)
        else:
            self.C = None

        # Matrix-free uses world-indexed J/Y for both dense and rigid phases.
        if self._matrix_free:
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
                size = self.forest.size_groups[0]
                self.J_world = self.J_by_size[size]
                self.Y_world = self.Y_by_size[size]
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
        if self.options.pgs_warmstart:
            arrays.allocate(DENSE_WARMSTART, **sizes)
            self._ws_prev_dense_slot_rows = self._contact_pool.rows(
                (ContactField("_ws_prev_dense_slot_sorted", wp.int32, -1),), on_rows=self._publish_contact_rows
            )
        fused = self._fused_diagonal_joint_limits
        arrays.allocate(FUSED_DIAGONAL_LIMITS, **(sizes if fused else {"world": 1, "world_dof": 1}))
        # Per-row weight is needed only when regularization is active because
        # speculative and restitution rows selectively use the rigid law.
        if self.laws.regularization_enabled:
            arrays.allocate(DENSE_ROW_WEIGHTS, **sizes)
        else:
            self.row_w = self._contact_row_w_dummy

    def _allocate_mf_buffers(self, model):
        """Allocate buffers for matrix-free PGS path for free rigid body contacts.

        CUDA without free rigid bodies still passes the
        ``mf_*`` arrays to kernels (the fused GS solve, the PGS diagnostics),
        but ``mf_constraint_count`` stays zero (``_clear_dense_row_state`` and
        ``finalize_constraint_counts_with_status`` write it only when MF is
        active), so every access sits inside a loop bounded by
        ``m_mf == 0`` and no element is ever read or written; :meth:`_pack_mf_meta`
        skips its launch. The rows then hold one slot per world and the per-world
        counts keep their real width.
        """
        if not self.forest.has_free_rigid_bodies and not self._matrix_free:
            return

        arrays = self._arrays
        worlds = self.forest.world_count
        rows = self.mf_max_constraints if self.forest.has_free_rigid_bodies else 1
        arrays.allocate(MF_COUNTERS, world=worlds)
        arrays.allocate(MF_ROWS, world=worlds, mf_row=rows, mf_meta=4 * rows)
        if self.laws.regularization_enabled and self.forest.has_free_rigid_bodies:
            arrays.allocate(MF_ROW_WEIGHTS, world=worlds, mf_row=rows)
        else:
            self.mf_row_w = self._contact_row_w_dummy

        if not self.forest.has_free_rigid_bodies:
            # Only the ``is not None`` checks observe this on the inactive
            # path, where no free rigid body owns a speed row.
            self.rigid_velocity_limit_slot = (
                wp.full((1,), -1, dtype=wp.int32, device=model.device, requires_grad=model.requires_grad)
                if self.laws.has_rigid_body_velocity_limits and model.body_count > 0
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
        if self.options.pgs_warmstart:
            arrays.allocate(MF_WARMSTART, world=worlds, mf_row=rows)
            pool = getattr(self, "_contact_pool", None)
            if pool is None:
                self._ws_prev_slot_sorted = wp.full((1,), -1, dtype=wp.int32, device=model.device)
            else:
                self._ws_prev_slot_rows = pool.rows(
                    (ContactField("_ws_prev_slot_sorted", wp.int32, -1),), on_rows=self._publish_contact_rows
                )

        if self.laws.has_rigid_body_velocity_limits and self.forest.free_rigid_body_count > 0:
            # Two speed rows per free rigid body: its linear and its angular speed.
            self.rigid_velocity_limit_slot = wp.full(
                (RIGID_SPEED_ROWS * self.forest.free_rigid_body_count,),
                -1,
                dtype=wp.int32,
                device=model.device,
                requires_grad=model.requires_grad,
            )
        else:
            self.rigid_velocity_limit_slot = None

    def _allocate_debug_buffers(self, model):
        """Allocate buffers for PGS convergence diagnostics."""
        if not self.options.pgs_debug:
            return
        sizes = {"world": self.forest.world_count, "row": self.dense_max_constraints, "mf_row": self.mf_max_constraints}
        self._arrays.allocate(PGS_DIAGNOSTICS, **sizes)
        if hasattr(self, "mf_impulses"):
            self._arrays.allocate(PGS_MF_DIAGNOSTICS, **sizes)
        else:
            self._diag_prev_mf_impulses = None

    def _scatter_armature_to_groups(self):
        """Copy armature from model (DOF-ordered) to size-grouped storage."""
        if not self.forest.size_groups:
            return

        armature_np = self._joint_armature_effective
        art_dof_start_np = self.forest.articulation_dof_start_host
        art_H_rows_np = self.forest.articulation_dof_count_host

        # R_by_size is sized to actual DOF count (matches H_by_size allocation)
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            R_np = np.zeros((n_arts, size), dtype=np.float32)

            group_to_art_np = np.flatnonzero(self.forest.response_dof_count == size)
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
        self._paired_components_reason = paired_components.decline(self)
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
                    self.forest.max_world_dofs,
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

        # Only free rigid bodies produce matrix-free rows, which the CUDA step packs each step.
        self._pack_mf_meta_kernel = (
            response_builds_rows.free_body_meta_kernel(self.mf_max_constraints, device_arch)
            if self._matrix_free and self.forest.has_free_rigid_bodies and self.mf_max_constraints > 0
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
                self.forest.max_world_dofs,
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
                self.forest.max_world_dofs,
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
            self._local_general_solver_blocks if self._local_internal_fast_path else self.forest.world_count
        )
        # The Jacobi tiers ahead of the matrix-free sweep (SweepTier).
        self._mf_gs_tiers = []
        if model.device.is_cuda and int(model.articulation_count) > 0:
            # The grouped response solve (_stage4_hinv_jt_grouped), for the sizes StepPlan gives it from the data.
            self._grouped_response = True
            self._grouped_dummy_diag = wp.zeros((1, 1), dtype=wp.float32, device=model.device)
        if self._mf_sweeps:
            # A test pin streams the impulses (storage is part of the kernel specialization key); the selection
            # sizes the resident row metadata with its choice.
            force_stream_mf_impulses = self._route_pins.get("mf_storage") == "streamed"
            storage = _MFGSStorage(
                False, force_stream_mf_impulses, False, "Without CUDA no shared-memory storage applies."
            )
            if model.device.is_cuda:
                storage = _execution_plan._select_mfgs_storage(
                    functools.partial(
                        sweep_routes.mf_gs_staging,
                        self.dense_max_constraints,
                        self.mf_max_constraints,
                        self.forest.max_world_dofs,
                    ),
                    arch_budget(device_arch),
                    block_count=self._mf_gs_block_count,
                    sm_count=int(model.device.sm_count),
                    force_stream_mf_impulses=force_stream_mf_impulses,
                )
            self._mfgs_storage = storage
            parallel_rows = self._ink_rows
            # The colored mapping takes the worlds the unit builder's rule picks at every pack
            # (sweep_units.COLORED_RULE), among those whose free bodies and row capacity admit it: built, with a
            # storage slot each, for those worlds only, it runs beside the one-warp mapping. More of them than the
            # device holds colored blocks run in waves, which scales the rule (ColoredRule.at_scale).
            rule = sweep_units.COLORED_RULE
            capable = sweep_units.capable_worlds(self.forest.free_rigid_per_world, self.mf_max_constraints, rule)
            if model.device.is_cuda and capable.size > 0:
                rule = rule.at_scale(capable.size, model.device.sm_count)
                capable = sweep_units.capable_worlds(self.forest.free_rigid_per_world, self.mf_max_constraints, rule)
            colored = capable.size > 0 and _route_table.capable("colored", self.laws.set) is None
            mf_gs_options = {
                "has_dense_velocity_limit_rows": self.options.enable_joint_velocity_limits,
                "shared_metadata": storage.shared_metadata,
                "resident_row_metadata": storage.resident_row_metadata,
                "stream_mf_impulses": storage.stream_mf_impulses,
                "skip_local_internal_worlds": self._local_internal_fast_path,
                "local_internal_max_constraints": self._dense_internal_max_rows,
                "contact_torsion": self.laws.contact_torsion_enabled,
                "factor_coordinates": self._paired_factor_coordinates,
                "skip_tier_rows": parallel_rows,
                "independent_components": self._paired_components is not None,
                "colored_worlds": colored,
                "max_row_dofs": self._packed_support,
            }
            mf_gs_shape = (self.dense_max_constraints, self.mf_max_constraints, self.forest.max_world_dofs, device_arch)
            self._pgs_solve_mf_gs_kernel = self._sweep_kernel(
                "mf_gs",
                sweep_routes.mf_gs_kernel,
                *mf_gs_shape,
                **mf_gs_options,
            )
            if colored:
                self._colored_units = sweep_units.ColoredUnits(
                    self.forest.world_count,
                    self.mf_max_constraints,
                    self.forest.max_world_dofs,
                    model.device,
                    worlds=capable,
                    rule=rule,
                )
                self._pgs_solve_mf_colored_kernel = self._sweep_kernel(
                    "mf_colored",
                    sweep_routes.mf_gs_kernel,
                    *mf_gs_shape,
                    mapping="colored",
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
                if len(tiers) >= _TIER_LIST_MIN_TIERS:
                    dev = model.device
                    self._tier_lo = wp.array([lo for _, lo in tiers], dtype=wp.int32, device=dev)
                    self._tier_hi = wp.array([hi for hi, _ in tiers], dtype=wp.int32, device=dev)
                    self._tier_counts = wp.zeros(len(tiers), dtype=wp.int32, device=dev)
                    self._tier_worlds = wp.empty((len(tiers), self.forest.world_count), dtype=wp.int32, device=dev)
                    self._tier_count_views = [self._tier_counts[t : t + 1] for t in range(len(tiers))]
                    self._tier_world_views = [self._tier_worlds[t] for t in range(len(tiers))]
                    self._tier_blocks = int(self.forest.world_count)
                tier_sweeps, self._par_sweeps_rule, self._par_sweeps_reason = self._parallel_tier_sweeps(tiers)
                for tier_index, ((rows_, min_rows_), sweeps_) in enumerate(zip(tiers, tier_sweeps, strict=True)):
                    tier = self._sweep_kernel(
                        "parallel_tier",
                        parallel_factory,
                        self.dense_max_constraints,
                        self.mf_max_constraints,
                        self.forest.max_world_dofs,
                        device_arch,
                        readout_budget=sweeps_,
                        rows=rows_,
                        sweeps=sweeps_,
                        min_rows=min_rows_,
                        inkernel_response=self._ink_sizes,
                        linked_rows=self.laws.friction_anchors_enabled,
                        velocity_limit_pass=self._parallel_velocity_limit_pass(),
                    )
                    self._mf_gs_tiers.append(dataclasses.replace(tier, tier=tier_index))
            # No matrix-free sweep enters a world without rows; the readout counts those solves as skipped.
            self._solve_readout.register_skipped(world_count=int(self.forest.world_count), device=model.device)

        # The gate of the row builder (_stage4_build_rows): it allocates free-body velocity-limit rows into
        # rigid_velocity_limit_slot when free rigid bodies exist, and an infinite activation fraction reserves
        # none. The route record names the law of the CPU's free-body sweep wherever the builder can allocate them,
        # even when every registered limit is infinite and no row is built.
        self._has_free_body_velocity_limit_rows = bool(
            self.forest.has_free_rigid_bodies
            and getattr(self, "rigid_velocity_limit_slot", None) is not None
            and not np.isinf(self.options.velocity_limit_activation_fraction)
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

        if model.device.is_cuda and len(self.forest.size_groups) > 1:
            for size in self.forest.size_groups:
                self._size_streams[size] = wp.Stream(model.device)
                self._size_events[size] = wp.Event(model.device)
        else:
            # No streams needed for CPU or single size group
            for size in self.forest.size_groups:
                self._size_streams[size] = None
                self._size_events[size] = None
        self._local_internal_stream = (
            wp.Stream(model.device)
            if model.device.is_cuda
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

    def _contact_friction_start_iteration(self, iterations: int) -> int:
        """Return the first local PGS iteration that should solve friction rows.

        ``contact_friction_position_iterations=-1`` preserves the legacy all-iteration behavior.
        """
        if self.options.contact_friction_position_iterations < 0:
            return 0
        if iterations <= 0:
            return 0
        active = min(self.options.contact_friction_position_iterations, iterations)
        return iterations - active

    def _pack_mf_meta(self, mf_rhs: wp.array) -> None:
        """Pack matrix-free row metadata for the GS kernels, or skip when no such row can exist."""
        if not self.forest.has_free_rigid_bodies:
            # Only free rigid bodies produce matrix-free rows, and the model plan fixes
            # them at construction: mf_constraint_count stays zero on the one-slot
            # dummy buffers, so the pack would write nothing, and no pack kernel is built.
            return
        wp.launch_tiled(
            self._pack_mf_meta_kernel,
            dim=[self.forest.world_count],
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
                    self.forest.art_group_idx,
                    self.forest.art_to_world,
                    self.forest.articulation_dof_start,
                    self._local_solve_owner,
                    self.constraint_count,
                    self.L_by_size[primary_size],
                    self.L_by_size[secondary_group_size],
                    self._packed_J,
                    self._row_sides,
                    self.forest.articulation_response_dof_count,
                    self.forest.articulation_world_dof_offset,
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
                    int(self.laws.regularization_enabled),
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
    ) -> None:
        if iterations <= 0:
            return
        if friction_start_iteration is None:
            friction_start_iteration = self._contact_friction_start_iteration(iterations)
        route = self._plan.solve.route.name
        if route == "sparse_pgs":
            worlds_per_block = 64 // self._sparse_solve_lanes
            wp.launch_tiled(
                self._pgs_solve_sparse_kernel,
                dim=[(self.forest.world_count + worlds_per_block - 1) // worlds_per_block],
                inputs=[
                    self.forest.world_count,
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
                    self.forest.world_dof_indices,
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
                    # Contact units read the triplets' block records.
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
                dim=(self.forest.n_arts_by_size[size], size),
                inputs=[
                    self.forest.group_to_art[size],
                    self.forest.art_to_world,
                    self.forest.articulation_world_dof_offset,
                    self.forest.articulation_dof_start,
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
                dim=[self.forest.world_count],
                inputs=[
                    self.constraint_count,
                    self.forest.world_dof_indices,
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
                    self.L_by_size[dense_size],
                    self.J_by_size[dense_size],
                    self._sparse_diagonal_row_dof,
                    self._sparse_diagonal_row_z,
                    self._fused_diagonal_limit_active,
                    self._fused_diagonal_limit_lower_rhs,
                    self._fused_diagonal_limit_upper_rhs,
                    self._diagonal_inverse_mass,
                    self.options.pgs_cfm,
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

        # Under a readout, count the worlds without rows of any family, which no sweep enters.
        self._solve_readout.count_skipped(self.constraint_count, self.mf_constraint_count)
        launch_local = bool(
            self._pgs_solve_local_internal_kernels
            or self._pgs_solve_local_pair_kernels
            or self._pgs_solve_local_residual_kernels
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
            components.prepare(self, dense_rhs, mf_meta)
        factor_stream = self._paired_general_solve_stream
        if factor_stream is not None:
            wp.get_stream(self.model.device).record_event(self._paired_general_ready)
            factor_stream.wait_event(self._paired_general_ready)
        if factor_kernel is not None:
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
                with self._sync_timed(f"factor_pgs_iters{iterations}"):
                    wp.launch_tiled(
                        factor_kernel,
                        dim=[(self.forest.world_count + 1) // 2],
                        inputs=[
                            self.forest.world_count,
                            self.constraint_count,
                            self.dense_phase_bounds,
                            self.forest.world_dof_indices,
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
                            iterations,
                            omega,
                            int(friction_start_iteration),
                            int(iteration_offset),
                        ],
                        outputs=[self.v_out, *self._solve_readout.operands(factor_kernel)],
                        block_dim=64,
                        device=self.model.device,
                        stream=factor_stream,
                    )

            if factor_stream is None:
                launch_factor_solve()
        with self._sync_timed(f"mfgs_iters{iterations}"):
            mf_gs_inputs = [
                self._local_general_world_count,
                self._local_general_worlds,
                general_block_count,
                int(use_general_queue),
                self.constraint_count,
                self.dense_phase_bounds,
                self._local_solve_owner if components is None else components.selector,
                self.forest.world_dof_indices,
                dense_rhs,
                self.diag,
                self.row_w,
                self.impulses,
                self.forest.articulation_response_dof_count,
                self.forest.articulation_world_dof_offset,
                int(self._packed_dense),
                self._packed_Z,
                self._row_sides,
                self.row_type,
                self.row_parent,
                self.row_mu,
                self._contact_torsion_group,
                self.laws.contact_torsion_radius,
                self.mf_constraint_count,
                self.mf_contact_rows_end,
                mf_meta,
                self.mf_impulses,
                self.mf_Z_a,
                self.mf_Z_b,
                self._body_factors(),
                self.mf_row_mu,
                self.mf_row_w,
                iterations,
                omega,
                int(self.laws.regularization_enabled),
                int(friction_start_iteration),
                int(iteration_offset),
            ]
            # The tiers own the worlds with only dense rows inside their row bound; the general sweep returns
            # early for those and solves the rest.
            tier_lists = getattr(self, "_tier_worlds", None) is not None
            if tier_lists:
                self._tier_counts.zero_()
                wp.launch(
                    classify_parallel_sweep_tiers,
                    dim=self.forest.world_count,
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
                            iterations,
                            omega,
                            friction_start_iteration,
                            iteration_offset,
                            solve_single=False,
                            solve_pair=False,
                        )
                    local_done_events.append(residual_stream.record_event())
                if pair_stream is not None:
                    with wp.ScopedStream(pair_stream, sync_enter=False):
                        self._launch_local_internal_solve(
                            dense_rhs,
                            iterations,
                            omega,
                            friction_start_iteration,
                            iteration_offset,
                            solve_single=False,
                            solve_residual=False,
                        )
                    local_done_events.append(pair_stream.record_event())
                with wp.ScopedStream(local_stream, sync_enter=False):
                    self._launch_local_internal_solve(
                        dense_rhs,
                        iterations,
                        omega,
                        friction_start_iteration,
                        iteration_offset,
                        solve_pair=pair_stream is None,
                        solve_residual=residual_stream is None,
                    )
                local_done_events.append(local_stream.record_event())
            else:
                with self._sync_timed(f"mfgs_local_internal_iters{iterations}"):
                    self._launch_local_internal_solve(
                        dense_rhs,
                        iterations,
                        omega,
                        friction_start_iteration,
                        iteration_offset,
                    )
        for local_done_event in local_done_events:
            wp.get_stream(self.model.device).wait_event(local_done_event)

    def _launch_matrix_free_position_solve(
        self,
        *,
        iterations: int,
        friction_start_iteration: int | None = None,
        iteration_offset: int = 0,
    ) -> None:
        if self.options.contact_compliance:
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
            omega=self.options.pgs_omega,
            friction_start_iteration=friction_start_iteration,
            iteration_offset=iteration_offset,
        )

    def _on_contact_rows(self, rows) -> None:
        """Assign the per-contact buffers at the contact pool's ready rows, on creation and after each growth."""
        rows.publish(self)
        self._max_contacts_alloc = rows.rows
        patches = self.laws.friction_patches
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
        if self.laws.contact_torsion_enabled:
            validate_torsion_step(self)
            if getattr(self, "_device_torsion", None) is not None:
                self._device_torsion.begin_step(state_in, state_out)
        if self.options.contact_compliance:
            # Reject incompatible contact preprocessing before it can mutate the stream.
            _contact_compliance.validate_step(self)
        if contacts is not None:
            self._follow_contacts(contacts)
        if self.options.pgs_warmstart:
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
                self.J_world = self.J_by_size[self.forest.size_groups[0]]
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
            for size, ctx in self._for_sizes():
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
                for size, ctx in self._for_sizes():
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
            if self.laws.contact_torsion_enabled:
                prepare_torsion_rows(self, state_in, state_aug, contacts)

        solve = self._plan.solve
        if not solve.assembles_delassus:
            # Compute Y = H^-1 * J^T only (no Delassus C)
            with wp.ScopedTimer("S4_HinvJt_Diag_RHS", print=False, use_nvtx=self._nvtx, synchronize=False):
                for size, ctx in self._for_sizes():
                    with ctx:
                        self._stage4_response(size)

                # Diagonal from J*Y (no full Delassus), or from the packed rows' |Z|^2
                packed_diag = self._stage4_compute_matrix_free_diag()
                self._stage4_pack_rows(write_diag=packed_diag)
                self._stage4_finalize_world_diag_cfm()

                # RHS = bias only (J*v recomputed per iteration)
                self._stage4_compute_rhs_world(dt)
                # NOTE: skip _stage4_accumulate_rhs_world — J*v_hat not baked into rhs

            # MF: whiten the rows (Z = L^-1 J^T), their mf_rhs, mf_eff_mass_inv and incident velocities
            if self.forest.has_free_rigid_bodies:
                with wp.ScopedTimer("S4_MF_Setup", print=False, use_nvtx=self._nvtx, synchronize=False):
                    with self._sync_timed("s4_mf_setup"):
                        self._mf_pgs_setup(state_aug, dt)

                    # MF: compute world-relative DOF offsets for two-phase GS kernel
                    with self._sync_timed("s4_mf_dof_offsets"):
                        wp.launch(
                            response_builds_rows.free_body_coordinates_kernel,
                            dim=self.forest.world_count * self.mf_max_constraints,
                            inputs=[
                                self.mf_constraint_count,
                                self.mf_body_a,
                                self.mf_body_b,
                                self.forest.body_to_articulation,
                                self.forest.articulation_world_dof_offset,
                                self.mf_max_constraints,
                            ],
                            outputs=[self.mf_dof_a, self.mf_dof_b],
                            device=self.model.device,
                        )

        else:
            # The CPU's split mode: the response, then the Delassus operator C assembled per size.
            self._stage4_zero_world_C()
            for size, ctx in self._for_sizes():
                with ctx:
                    self._stage4_response(size)
            for size in self.forest.size_groups:
                self._stage4_delassus(size)
            self._stage4_finalize_world_diag_cfm()

            self._stage4_compute_rhs_world(dt)

            for size in self.forest.size_groups:
                self._stage4_accumulate_rhs_world(size)
            self._stage4_apply_world_contact_restitution(dt, matrix_free=False)

        # ══════════════════════════════════════════════════════════════
        # STAGE 5+6: PGS solve
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S5_PGS_Prep", print=False, use_nvtx=self._nvtx, synchronize=False):
            self._stage5_prepare_impulses_world()
            if self.options.pgs_warmstart and self._ws_prev_dense_impulses is not None and contacts is not None:
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
                        self.options.pgs_warmstart_decay,
                        dt / self._ws_prev_dt if self._ws_prev_dt > 0.0 else 1.0,
                        self.dense_max_constraints,
                    ],
                    outputs=[self.impulses],
                    extent=contacts.rigid_contact_count,
                    max_blocks=self._contact_capacity_blocks,
                )

        if self.laws.friction_anchors_enabled and self._patch_frame_built(contacts):
            ratio = dt / self._ws_prev_dt if self._ws_prev_dt > 0.0 else 1.0
            seeded = [
                (route, parents, mu, impulses, decay * ratio)
                for route, parents, mu, impulses, decay in self._patch_row_arrays()
                if decay > 0.0
            ]
            if seeded:
                self.laws.friction_patches.seed(
                    state_in,
                    self.contact_world,
                    self.contact_slot,
                    self.contact_path,
                    self.contact_slots_needed,
                    patch_routes(seeded),
                )

        if solve.family == "matrix_free":
            with wp.ScopedTimer("S5_GatherJY", print=False, use_nvtx=self._nvtx, synchronize=False):
                self._stage4_apply_world_contact_restitution(dt, matrix_free=True)
                if self._preelim_active:
                    # Bilateral elimination: the other rows take the closed-loop response, their right-hand sides
                    # and diagonals (the rows' closed-loop inertia).
                    self._stage4_bilateral_preelim()

                # Initialize v_out = v_hat before GS loop
                self._stage6_prepare_world_velocity()
                if self.options.pgs_warmstart and self._sparse_mass_matrix_size is None:
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
                if self.options.pgs_debug:
                    self._pgs_convergence_log.append([])
                    self._pgs_ncp_residual_log.append([])
                    friction_start_iteration = self._contact_friction_start_iteration(self.options.pgs_iterations)
                    for _pgs_dbg_iter in range(self.options.pgs_iterations):
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
                            dim=self.forest.world_count,
                            inputs=[
                                bind_dense_rows(self),
                                bind_mf_rows(self),
                                self.constraint_count,
                                self.forest.world_dof_indices,
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
                            dim=self.forest.world_count,
                            inputs=[
                                bind_dense_rows(self),
                                bind_mf_rows(self),
                                self.constraint_count,
                                self.forest.world_dof_indices,
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
                        iterations=self.options.pgs_iterations,
                        friction_start_iteration=self._contact_friction_start_iteration(self.options.pgs_iterations),
                    )

        elif solve.family == "split_interleaved":
            # Split mode with mixed contacts: interleaved dense and MF, 1 iteration each.
            self._mf_pgs_setup(state_aug, dt)
            self.v_mf_accum.zero_()
            self._apply_mf_warmstart_velocity(self.v_mf_accum, set_output_to_delta=True)
            if self.options.pgs_warmstart:
                # Split stores dense constraints in impulse space, so its RHS
                # must include the velocity represented by the cached MF
                # lambdas before the first interleaved dense iteration.
                for size in self.forest.size_groups:
                    self._stage4_accumulate_rhs_world(size, velocity=self.v_mf_accum)

            for _pgs_iter in range(self.options.pgs_iterations):
                # Dense PGS (1 iteration, impulse space)
                self._dispatch_dense_pgs_solve(
                    iterations=1,
                    friction_start_iteration=self._contact_friction_start_iteration(self.options.pgs_iterations),
                    iteration_offset=_pgs_iter,
                )

                # Rebuild v_out = v_hat + Y*impulses + MF_corrections
                self._stage6_prepare_world_velocity()
                for size in self.forest.size_groups:
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
                    friction_start_iteration=self._contact_friction_start_iteration(self.options.pgs_iterations),
                    iteration_offset=_pgs_iter,
                )

                # v_mf_accum += (v_out - v_out_snap); v_out_snap = delta
                wp.launch(
                    compute_delta_and_accumulate,
                    dim=self.v_out.size,
                    inputs=[self.v_out, self.v_out_snap, self.v_mf_accum],
                    device=self.model.device,
                )

                # Update dense rhs: world_rhs += J * delta_v_mf
                for size in self.forest.size_groups:
                    self._stage4_accumulate_rhs_world(size, velocity=self.v_out_snap)

            # v_out is already final (includes both dense and MF corrections)

        else:
            # Split without mixed contacts (dense_split): dense PGS, then optional MF
            if self.options.pgs_debug:
                self._pgs_convergence_log.append([])
                friction_start_iteration = self._contact_friction_start_iteration(self.options.pgs_iterations)
                for _pgs_dbg_iter in range(self.options.pgs_iterations):
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
                    iterations=self.options.pgs_iterations,
                    friction_start_iteration=self._contact_friction_start_iteration(self.options.pgs_iterations),
                )

            self._stage6_prepare_world_velocity()
            for size in self.forest.size_groups:
                self._stage6_apply_impulses_world(size)

            if self.forest.has_free_rigid_bodies:
                self._stage6b_mf_pgs(state_aug, dt)

        self._stage6_clamp_rigid_velocity_limits()

        # ══════════════════════════════════════════════════════════════
        # STAGE 7: Update qdd + integrate
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S7_Integrate", print=False, use_nvtx=self._nvtx, synchronize=False):
            self._stage6_update_qdd(state_in, state_aug, dt)
            self._stage6_integrate(state_in, state_aug, state_out, dt)

        # ── MF warm-start carry: snapshot this step's converged impulses +
        # row-type table + per-(sorted)-contact slot map so step N+1 can seed
        # from them by contact identity. Only runs when the feature is on, so
        # the default-off launch sequence is unchanged.
        # Dense identity-matched carry: snapshot this step's converged dense
        # impulses + row types + per-(sorted)-contact dense-slot map (mirror of
        # the MF carry below).
        if self.laws.friction_anchors_enabled:
            with wp.ScopedTimer("S7_Friction_Patch_Carry", print=False, use_nvtx=self._nvtx, synchronize=False):
                self._finish_friction_patches(state_in, state_out, contacts, dt)

        if self.options.pgs_warmstart and self._ws_prev_dense_impulses is not None:
            with wp.ScopedTimer("S7_Dense_Warmstart_Carry", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                wp.copy(self._ws_prev_dense_impulses, self.impulses)
                wp.copy(self._ws_prev_dense_row_type, self.row_type)
                wp.copy(self._ws_prev_dense_row_parent, self.row_parent)
                if contacts is None:
                    self._ws_prev_dense_row_type.fill_(-1)
                    self._ws_prev_dense_row_parent.fill_(-1)

        if self.options.pgs_warmstart and self.forest.has_free_rigid_bodies:
            with wp.ScopedTimer("S7_MF_Warmstart_Carry", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                wp.copy(self._ws_prev_mf_impulses, self.mf_impulses)
                wp.copy(self._ws_prev_mf_row_type, self.mf_row_type)
                wp.copy(self._ws_prev_mf_row_parent, self.mf_row_parent)
                if contacts is None:
                    self._ws_prev_mf_row_type.fill_(-1)
                    self._ws_prev_mf_row_parent.fill_(-1)

        if self.options.pgs_warmstart:
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
                    for size in self.forest.size_groups:
                        if self._mass_update_global_flag and size not in self._plan.unread_mass_sizes:
                            self._H_bufs[self._buf_idx][size].zero_()
                        if size not in clear:
                            continue
                        wp.launch(
                            clear_grouped_jacobian_active_rows,
                            dim=self.forest.n_arts_by_size[size] * 32,
                            inputs=[
                                self.forest.group_to_art[size],
                                self.forest.art_to_world,
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

        if self.laws.contact_torsion_enabled and getattr(self, "_device_torsion", None) is not None:
            self._device_torsion.end_step(state_out)
            if self._device_torsion.deferred_errors and not capturing(self.model.device):
                self.validate_contact_torsion()
        self._step += 1
        return state_out

    def constraint_capacity_status(self, *, clear: bool = False) -> dict[str, bool]:
        """Return sticky capacity-overflow flags without enabling row telemetry.

        The keys ``dense`` and ``matrix_free`` identify row buffers whose
        capacity omitted constraints. A contact stream's count never exceeds its
        rows: a collision pass clamps it and flags the drop in its own capacity
        status.
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
        result = {"dense": dense, "matrix_free": matrix_free}
        frame_status = self.laws.patch_frame_status()
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
        frame_status = self.laws.patch_frame_status()
        if frame_status is not None and frame_status.numpy()[0]:
            raise RuntimeError(
                "FeatherPGS solved captured steps without patch friction: their contact generation or count changed "
                "since the previous captured step of the same graph without a collision pass or Contacts.clear(). "
                "Call Contacts.clear() before rewriting contacts between captured steps, or set "
                "FEATHER_PGS_PATCH_FRAME_REUSE=0; constraint_capacity_status(clear=True) clears the flag."
            )

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
            for family in ("dense", "mf")
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
        enable_friction_flag = 1 if self.options.enable_contact_friction else 0
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
    def _parallel_size_region(self):
        """Context for parallel dispatch across size groups."""
        if not self.model.device.is_cuda or len(self.forest.size_groups) <= 1:
            yield
            return

        main_stream = wp.get_stream(self.model.device)
        self._init_event = main_stream.record_event()
        try:
            yield
        finally:
            for size in self.forest.size_groups:
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
    def _size_ctx(self, size: int):
        with self._on_size_stream(size):
            yield

    def _for_sizes(self):
        # convenience generator; keeps step code tight
        with self._parallel_size_region():
            for size in self.forest.size_groups:
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
            DynamicsData,
            model=self.model,
            solver=self,
            forest=self.forest,
            state=state,
            aug=state_aug,
            control=control,
            **values,
        )

    def _launch_velocity_prescale(self, joint_qd: wp.array, prescaled_joint_qd: wp.array) -> None:
        """Write every DOF's PhysX-prescaled joint velocity in one launch, without a separate copy."""
        model = self.model
        if not model.articulation_count:
            wp.copy(prescaled_joint_qd, joint_qd)
            return
        wp.launch(
            prescale_joint_velocity_limits,
            dim=self.laws.velocity_prescale_threads,
            inputs=[
                bind_joint_table(self),
                model.articulation_count,
                self.laws.velocity_prescale_lanes,
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
        if self.options.enable_joint_velocity_limits:
            self._launch_velocity_prescale(state_in.joint_qd, self.qd_work)
            stage3_qd = self.qd_work
        else:
            stage3_qd = state_in.joint_qd

        refresh_composite = (self._step % self.options.update_mass_matrix_interval) == 0 or self._force_mass_update
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
                    self.forest.body_to_articulation,
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
        with wp.ScopedStream(dynamics_stream, sync_enter=False):
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
            if self.forest.articulation_max_dofs > 0:
                wp.launch(
                    eval_rigid_tau_and_augmented_drives,
                    dim=model.articulation_count,
                    inputs=[
                        self._dynamics(state_in, state_aug, control),
                        self.forest.articulation_dof_count,
                        model.joint_target_q_start,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        control.joint_target_q,
                        control.joint_target_qd,
                        model.joint_effort_limit,
                        self.forest.articulation_max_dofs,
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

    def _stage1_crba(
        self,
        state_aug: State,
        drive_rows_ready: wp.Event | None,
    ):
        model = self.model
        global_flag = (
            1 if ((self._step % self.options.update_mass_matrix_interval) == 0 or self._force_mass_update) else 0
        )
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
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            if self._H_bufs is None and global_flag:  # not double-buffered
                self.H_by_size[size].zero_()
            wp.launch(
                crba_fill,
                dim=int(n_arts * size),
                inputs=[
                    dynamics,
                    self._body_inertia_terms if size in self.forest.free_body_inertia_sizes else self.body_I_c,
                    self.forest.group_to_art[size],
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
                    self.forest.group_to_art[size],
                    self.forest.articulation_dof_start,
                    size,
                    self.forest.articulation_max_dofs,
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
        sizes = self.forest.size_groups

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
        n_arts = self.forest.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.factor_coordinates_kernel(),
            dim=n_arts * size,
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.forest.group_to_art[size],
                self.mass_update_mask,
                size,
            ],
            outputs=[self.L_by_size[size]],
            device=self.model.device,
        )

    def _stage2_cholesky_tiled(self, size: int):
        """Factor a tiled size's H (factor route ``tiles``)."""
        model = self.model
        n_arts = self.forest.n_arts_by_size[size]
        wp.launch_tiled(
            self._cholesky_kernels_by_size[size],
            dim=[n_arts],
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.forest.group_to_art[size],
                self.mass_update_mask,
            ],
            outputs=[self.L_by_size[size]],
            block_dim=self.tile_threads,
            device=model.device,
        )

    def _stage2_cholesky_loop(self, size: int):
        """Factor a loop size's H (factor route ``loop``)."""
        model = self.model
        n_arts = self.forest.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.factor_articulations_kernel(),
            dim=n_arts,
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.forest.group_to_art[size],
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
            dim=self.forest.n_arts_by_size[size] * size,
            inputs=[
                self.L_by_size[size],
                self.forest.group_to_art[size],
                self.forest.articulation_dof_start,
                size,
                state_aug.joint_tau,
            ],
            outputs=[state_aug.joint_qdd],
            device=self.model.device,
        )

    def _stage3_trisolve_tiled(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.forest.n_arts_by_size[size]

        wp.launch(
            gather_tau_to_groups,
            dim=n_arts,
            inputs=[
                state_aug.joint_tau,
                self.forest.group_to_art[size],
                self.forest.articulation_dof_start,
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
                self.forest.group_to_art[size],
                self.forest.articulation_dof_start,
                size,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_trisolve_loop(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.forest.n_arts_by_size[size]
        wp.launch(
            trisolve_loop,
            dim=n_arts,
            inputs=[
                self.L_by_size[size],
                self.forest.group_to_art[size],
                self.forest.articulation_dof_start,
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
        self._stage3_correct_free_root_velocity(state_in, stage3_qd, dt)

    def _stage3_correct_free_root_velocity(
        self, state_in: State, stage3_qd: wp.array, dt: float, *, apply_transport: bool = True
    ) -> None:
        """Apply canonical gyro physics after either ordinary or private prediction."""
        model = self.model
        # Keep v_hat on jcalc_integrate's free-root convention. The compact
        # launch excludes every non-root joint; a live device mask handles
        # runtime kinematic changes without changing graph topology.
        if not self.forest.free_root_joint_count:
            return
        if self.forest.free_rigid_body_count:
            wp.launch(
                apply_free_root_velocity_corrections,
                dim=self.forest.free_root_joint_count,
                inputs=[
                    self.forest.free_root_joint_indices,
                    model.joint_qd_start,
                    model.joint_child,
                    self.forest.body_to_articulation,
                    self.forest.is_free_rigid,
                    self.forest.art_group_idx,
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
                dim=self.forest.free_root_joint_count,
                inputs=[
                    self.forest.free_root_joint_indices,
                    model.joint_qd_start,
                    self._kinematic_joint_mask,
                    stage3_qd,
                    dt,
                ],
                outputs=[self.v_hat],
                device=model.device,
            )

    def _build_friction_patches(self, state_in, contacts):
        """Construct the shared contact law independently of the dynamics owner."""
        if self.laws.friction_anchors_enabled and contacts is not None:
            self.laws.friction_patches.build(
                self.model,
                state_in,
                contacts,
                body_to_articulation=self.forest.body_to_articulation,
                is_free_rigid=self.forest.is_free_rigid
                if self.forest.is_free_rigid is not None
                else self._dummy_is_free_rigid,
                contact_gap_gate=self.options.contact_gap_gate,
                same_articulation_gap_gate=self.options.same_articulation_contact_gap_gate,
                articulation_pair_gap_gate=self.options.articulation_pair_contact_gap_gate,
                friction_gap=self.options.contact_friction_gap_threshold,
                friction_articulation_pairs_only=self.options.contact_friction_articulation_pairs_only,
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
        if self.laws.friction_anchors_enabled and self._patch_frame_built(contacts):
            self.laws.friction_patches.link(
                self.contact_world,
                self.contact_slot,
                self.contact_path,
                self.contact_slots_needed,
                self._patch_routes(),
            )

    def _finish_friction_patches(self, state_in, state_out, contacts, dt):
        """Publish persistent patch history from the integrated public state."""
        if not self.laws.friction_anchors_enabled:
            return
        if contacts is None:
            # A memset of today's rows: a capture of it is recaptured after growth.
            self._contact_pool.note_unbound()
            self.laws.friction_patches.previous.valid.zero_()
            return
        # A step without a frame holds no patch rows; it stores the last frame again.
        routes = self._patch_routes() if self._patch_frame_built(contacts) else patch_routes(())
        self.laws.friction_patches.finish(
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
        if self.forest.has_non_free_articulations or not self.forest.has_free_rigid_bodies:
            yield (
                0,
                self.row_parent,
                self.row_mu,
                self.impulses,
                self.options.pgs_warmstart_decay if self.options.pgs_warmstart else 0.0,
            )
        if self.forest.has_free_rigid_bodies:
            yield (
                1,
                self.mf_row_parent,
                self.mf_row_mu,
                self.mf_impulses,
                self.options.pgs_warmstart_decay if self.options.pgs_warmstart else 0.0,
            )

    def _finalize_constraint_status(self, contacts: Contacts | None) -> None:
        """Latch current per-world capacity failures for every row owner."""
        # The contact capacity (argument 8) follows the ready rows in a capture whose launches bind.
        self._contact_pool.launch(
            _finalize_constraint_status,
            self.forest.world_count,
            inputs=[
                self.slot_counter,
                self.mf_slot_counter if self.forest.has_free_rigid_bodies else self._dummy_mf_slot_counter,
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
        order_mf = int(self.forest.has_free_rigid_bodies)
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
            dim=(self.forest.world_count, lanes),
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
        if self.options.contact_compliance:
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
        mf_active = self.forest.has_free_rigid_bodies
        if self.forest.has_prescribed_response:
            self.mf_target_velocity.zero_()

        # One launch clears the world-level allocation state and, with free rigid bodies, the
        # matrix-free counters and impulses (PGS reads mf_impulses before its first write; the
        # warm-start seed pass re-gathers after build_mf_contact_rows). Other row buffers are
        # fully overwritten by their producers.
        wp.launch(
            _clear_dense_row_state,
            dim=(self.forest.world_count, self.mf_impulses.shape[1] if mf_active else 1),
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
        is_free_rigid = (
            self.forest.is_free_rigid if self.forest.is_free_rigid is not None else self._dummy_is_free_rigid
        )
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
                dim=(self.forest.world_count, 12),
                inputs=[self._sparse_diagonal_dense_groups, self.J_by_size[6], self._sparse_diagonal_row_dof],
                device=model.device,
            )
            j_buffers_zeroed = True
        if self._row_packets is not None:
            self._row_packets.prepare_prefix(state_in, dt)
        if self._keyed_rows is not None:
            self._keyed_rows.begin_rows(state_in, state_aug, dt)

        # ── Dense row-family layout ──────────────────────────────────────
        # Slots are handed out in the order [mimic][joint-limit]
        # [joint-velocity-limit][contact/friction], so each family owns one
        # contiguous dense row range:
        #   mimic and position limits: [0, bounds[0])
        #   velocity limits:           [bounds[0], bounds[1])
        #   contacts and friction:     [bounds[1], m_dense)
        # The two snapshot_dense_phase_bound launches below record the family
        # boundaries and must stay glued to this allocation sequence
        # (slot_counter is zeroed at the top of this method). Mimic rows come
        # first, ahead of the (activation-gated, flickering) limit family, so
        # warm-started mimic lambdas keep a stable slot index across steps.

        # Mimic (joint coupling) equality rows — static count per model.
        if self.laws.mimic_count > 0 and self._row_packets is None:
            wp.launch(
                allocate_mimic_slots,
                dim=self.laws.mimic_count,
                inputs=[field_group.bind(Mimics, laws=self.laws, model=model), max_constraints],
                outputs=[
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                ],
                device=model.device,
            )
            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.forest.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True
            for size in self.forest.size_groups:
                if size not in self.laws.mimic_sizes:
                    continue
                n_arts = self.forest.n_arts_by_size[size]
                if self._sparse_velocity_limits and size == self._sparse_mass_matrix_size:
                    indices = self._sparse_mass_matrix_indices
                    wp.launch(
                        response_builds_rows.factor_mimic_rows_kernel,
                        dim=n_arts * 32,
                        inputs=[
                            self.forest.group_to_art[size],
                            self.forest.art_to_world,
                            self.forest.articulation_world_dof_offset,
                            self.forest.articulation_dof_start,
                            self.laws.mimic_slot,
                            self.laws._mimic_art_start,
                            self.laws._mimic_art_list,
                            self.laws._mimic_dof0,
                            self.laws._mimic_dof1,
                            self.laws._mimic_q0,
                            self.laws._mimic_q1,
                            model.constraint_mimic_coef0,
                            model.constraint_mimic_coef1,
                            state_in.joint_q,
                            self.options.pgs_beta,
                            self.options.pgs_cfm,
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
                        field_group.bind(Mimics, laws=self.laws, model=model),
                        joints,
                        self.forest.group_to_art[size],
                        state_in.joint_q,
                        self.options.pgs_beta,
                        self.options.pgs_cfm,
                        support,
                    ],
                    outputs=[self._jacobian_rows(size), dense_rows],
                    device=model.device,
                )

        # Connect (loop-closure) equality rows — static count per model.
        if self.laws.connect_count > 0:
            wp.launch(
                allocate_connect_slots,
                dim=self.laws.connect_count,
                inputs=[field_group.bind(Connects, laws=self.laws), max_constraints],
                outputs=[
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                ],
                device=model.device,
            )
            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.forest.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True
            for size in self.forest.size_groups:
                n_arts = self.forest.n_arts_by_size[size]
                wp.launch(
                    populate_connect_J_for_size,
                    dim=n_arts,
                    inputs=[
                        joints,
                        field_group.bind(Connects, laws=self.laws),
                        self.forest.group_to_art[size],
                        size,
                        state_in.body_q,
                        state_in.body_qd,
                        model.body_com,
                        self.forest.body_to_joint,
                        self.forest.body_to_articulation,
                        model.joint_ancestor,
                        state_aug.joint_S_s,
                        self.articulation_origin,
                        self.options.pgs_beta,
                        self.options.pgs_cfm,
                        support,
                    ],
                    outputs=[self._jacobian_rows(size), dense_rows],
                    device=model.device,
                )

        if self._fused_diagonal_joint_limits:
            wp.launch(
                prepare_fused_diagonal_joint_limits,
                dim=self.forest.world_count * self.forest.max_world_dofs,
                inputs=[
                    joints,
                    self.forest.world_dof_indices,
                    self.forest.max_world_dofs,
                    self._fused_diagonal_limit_dof_mask,
                    self.laws.joint_limit_q_index,
                    state_in.joint_q,
                    self.options.joint_limit_activation_gap,
                    self.options.pgs_beta,
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
            self.options.enable_joint_limits
            and self.laws.joint_limit_sizes
            and self._row_packets is None
            and self._keyed_rows is None
        ):
            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.forest.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True
            # The native limit rows (prefix_rows) read the law and write the records through one struct.
            law = field_group.bind(
                prefix_rows.PrefixLaw,
                solver=self,
                laws=self.laws,
                options=self.options,
                model=model,
                state_in=state_in,
                dt=dt,
            )
            for size in self.forest.size_groups:
                if size not in self.laws.joint_limit_sizes or size in self._fused_diagonal_joint_limit_sizes:
                    continue
                n_arts = self.forest.n_arts_by_size[size]
                if size == self._sparse_mass_matrix_size:
                    indices = self._sparse_mass_matrix_indices
                    wp.launch(
                        response_builds_rows.factor_limit_rows_kernel,
                        dim=n_arts * 32,
                        inputs=[
                            self.forest.group_to_art[size],
                            self.forest.art_to_world,
                            self.forest.articulation_world_dof_offset,
                            self.forest.articulation_dof_start,
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
                warp_kernel = self.laws.joint_limit_warp_kernels.get(size)
                if warp_kernel is not None:
                    wp.launch_tiled(
                        warp_kernel,
                        dim=[(n_arts + JOINT_LIMIT_WARPS_PER_BLOCK - 1) // JOINT_LIMIT_WARPS_PER_BLOCK],
                        inputs=[
                            n_arts,
                            self.forest.articulation_dof_start,
                            self.forest.art_to_world,
                            self.forest.group_to_art[size],
                            self._resolved_worlds,
                            law,
                            support,
                        ],
                        outputs=[self._jacobian_rows(size)],
                        block_dim=32 * JOINT_LIMIT_WARPS_PER_BLOCK,
                        device=model.device,
                    )
                else:
                    wp.launch(
                        build_joint_limit_rows_for_size,
                        dim=n_arts,
                        inputs=[
                            joints,
                            state_in.joint_q,
                            self.options.joint_limit_activation_gap,
                            self.forest.group_to_art[size],
                            max_constraints,
                            self.options.pgs_beta,
                            self.options.pgs_cfm,
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

        # Unconditional: the first boundary must include position-limit
        # rows and remain valid when that family is disabled (when it is just
        # the current watermark at the start of the velocity-limit segment).
        if self._row_packets is None and self._keyed_rows is None:
            wp.launch(
                snapshot_dense_phase_bound,
                dim=self.forest.world_count,
                inputs=[self.slot_counter, 0],
                outputs=[self.dense_phase_bounds],
                device=model.device,
            )

        # Allocate + populate joint velocity-limit rows (per-DOF clamp on
        # |qdot_i| against model.joint_velocity_limit). Dense contact rows
        # are allocated after this block so contact/friction rows form the
        # trailing per-world slot segment; the sweeps visit velocity-limit
        # rows in a dedicated final pass regardless of slot order.
        if (
            self.options.enable_joint_velocity_limits
            and self.laws.velocity_limit_slot is not None
            and self._keyed_rows is None
        ):
            wp.launch(
                allocate_joint_velocity_limit_slots,
                dim=model.articulation_count,
                inputs=[
                    joints,
                    self.v_hat,
                    self.options.velocity_limit_activation_fraction,
                    max_constraints,
                ],
                outputs=[
                    self.laws.velocity_limit_slot,
                    self.laws.velocity_limit_sign,
                    self.slot_counter,
                    self._dense_first_rejected_slot,
                ],
                device=model.device,
            )

        # Unconditional velocity-limit boundary snapshot; dense contact rows start here.
        if self._row_packets is None and self._keyed_rows is None:
            wp.launch(
                snapshot_dense_phase_bound,
                dim=self.forest.world_count,
                inputs=[self.slot_counter, 1],
                outputs=[self.dense_phase_bounds],
                device=model.device,
            )

        if self._sparse_diagonal_contact_solve:
            wp.launch(
                clear_sparse_diagonal_response_prefix,
                dim=self.forest.world_count,
                inputs=[self.dense_phase_bounds],
                outputs=[self._sparse_diagonal_row_dof],
                device=model.device,
            )

        if (
            self.options.enable_joint_velocity_limits
            and self.laws.velocity_limit_slot is not None
            and self._keyed_rows is None
        ):
            if self._H_bufs is None and not j_buffers_zeroed:  # not double-buffered
                for size in self.forest.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True
            for size in self.forest.size_groups:
                n_arts = self.forest.n_arts_by_size[size]
                if self._sparse_velocity_limits and size == self._sparse_mass_matrix_size:
                    indices = self._sparse_mass_matrix_indices
                    wp.launch(
                        response_builds_rows.factor_velocity_limit_rows_kernel,
                        dim=n_arts * 32,
                        inputs=[
                            self.forest.group_to_art[size],
                            self.forest.art_to_world,
                            self.forest.articulation_world_dof_offset,
                            self.forest.articulation_dof_start,
                            model.joint_velocity_limit,
                            self.laws.velocity_limit_slot,
                            self.laws.velocity_limit_sign,
                            self.options.pgs_cfm,
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
                        self.laws.velocity_limit_slot,
                        self.laws.velocity_limit_sign,
                        self.forest.group_to_art[size],
                        self.options.pgs_cfm,
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
            enable_friction_flag = 1 if self.options.enable_contact_friction else 0
            pool = self._contact_pool
            count = contacts.rigid_contact_count
            # Sized for the bound in a capture whose launches bind: every replay runs them at the live
            # count and strides with the captured threads (ContactPool.launch); otherwise for the ready rows.
            rows = pool.launch_rows()
            contact_build_threads = min(rows, _CONTACT_BUILD_THREAD_CAP)

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
                    self.forest.body_to_articulation,
                    self.forest.art_to_world,
                    model.body_flags,
                    self.forest.body_has_response_dofs,
                    is_free_rigid,
                    has_free_rigid_flag,
                    self.options.contact_gap_gate,
                    self.options.same_articulation_contact_gap_gate,
                    self.options.articulation_pair_contact_gap_gate,
                    max_constraints,
                    self.mf_max_constraints,
                    enable_friction_flag,
                    self.options.contact_friction_gap_threshold,
                    1 if self.options.contact_friction_articulation_pairs_only else 0,
                    1,
                    self.laws.friction_patches.view,
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
                for size in self.forest.size_groups:
                    self.J_by_size[size].zero_()
                j_buffers_zeroed = True

            # The class's case names the producer that writes every contact row (contact_row_plan):
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
                for size in self.forest.size_groups:
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
                                self.forest.art_group_idx,
                                self.forest.is_free_rigid,
                                self.forest.articulation_world_dof_offset,
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
                        inputs=[contact_input, contact_jacobian_workers, size, self.forest.art_group_idx, law, support],
                        outputs=[self._jacobian_rows(size)],
                        extent=count,
                        clamp=True,
                    )
                if plan.producer == "sparse_diagonal":
                    # The dense builder has 32 lanes per worker. Give the scalar
                    # builder the contact-build thread budget; its live-prefix
                    # stride covers larger counts without visiting unused slots.
                    sparse_contact_response_workers = contact_build_threads
                    pool.launch(
                        response_builds_rows.sparse_pair_contact_rows_kernel,
                        sparse_contact_response_workers,
                        inputs=[
                            contact_input,
                            sparse_contact_response_workers,
                            self._sparse_diagonal_response_size,
                            self.forest.articulation_world_dof_offset,
                            self.forest.body_single_response_dof,
                            self._diagonal_inverse_mass,
                            law,
                        ],
                        outputs=[self._sparse_diagonal_row_dof, self._sparse_diagonal_row_z],
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
                for size in self.forest.size_groups:
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
                                self.forest.art_group_idx,
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
                            self.forest.art_group_idx,
                            self.forest.body_to_joint,
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
                        int(self.forest.has_prescribed_response),
                        law,
                    ],
                    extent=count,
                    clamp=True,
                )

            if mf_active:
                wp.launch(
                    finalize_constraint_counts_with_status,
                    dim=self.forest.world_count,
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
                if self.options.pgs_warmstart:
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
                            self.options.pgs_warmstart_decay,
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
                self.forest.articulation_root_dof_start
                if self.forest.articulation_root_dof_start is not None
                else wp.zeros((1,), dtype=wp.int32, device=model.device)
            )
            wp.launch(
                allocate_rigid_velocity_limit_slots,
                dim=self.forest.free_rigid_body_count,
                inputs=[
                    bind_free_bodies(self),
                    model.body_flags,
                    root_dof_start,
                    self.v_hat,
                    self.options.velocity_limit_activation_fraction,
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
                dim=self.forest.free_rigid_body_count,
                inputs=[bind_free_bodies(self), bind_mf_rows(self), self.rigid_velocity_limit_slot],
                device=model.device,
            )
            wp.launch(
                finalize_constraint_counts_with_status,
                dim=self.forest.world_count,
                inputs=[self.mf_slot_counter, self.mf_max_constraints, 1, self._mf_first_rejected_slot],
                outputs=[self.mf_constraint_count, self._constraint_capacity_status],
                device=model.device,
            )

        if self._row_packets is not None:
            self._row_packets.finish_rows()
        wp.launch(
            finalize_constraint_counts_with_status,
            dim=self.forest.world_count,
            inputs=[self.slot_counter, max_constraints, 0, self._dense_first_rejected_slot],
            outputs=[self.constraint_count, self._constraint_capacity_status],
            device=model.device,
        )
        if self._sparse_prepare_contacts:
            wp.launch(
                finalize_sparse_joint_limit_rows,
                dim=self.forest.world_count,
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
                dim=self.forest.world_count * 32,
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
                dim=self.forest.world_count,
                inputs=[
                    bind_mf_rows(self),
                    field_group.bind(LocalOwners, solver=self),
                    self.constraint_count,
                    self.dense_phase_bounds,
                    self.mf_constraint_count,
                    self.forest.body_to_articulation,
                    self.forest.articulation_dof_count,
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
                            self.forest.art_to_world,
                            self._local_solve_owner,
                            owner,
                        ],
                        outputs=[active_count, active_candidates[size], active_secondaries[size]],
                        device=model.device,
                    )

        if self.options.warn_constraint_overflow:
            mf_raw_counts = self.mf_slot_counter if mf_active else self.slot_counter
            wp.launch(
                _warn_constraint_row_overflow,
                dim=self.forest.world_count,
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
        offsets = (
            self.forest.articulation_world_dof_offset if self._hinv_jt_writes_world else self.forest.group_to_art[size]
        )
        return J_world, Y_world, offsets

    def _stage4_response(self, size: int) -> None:
        """Build one size's contact response along the plan's stage-4 route (none where an owner publishes it)."""
        route = self._plan.response[size].name
        if route == "diagonal":
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
        """Assemble one size's Delassus block (the CPU's split mode), a thread per entry."""
        self._stage4_delassus_par_row_col(size)

    def _stage4_hinv_jt_grouped(self, size: int) -> None:
        """The grouped response solve (grouped dynamics), a warp per articulation; verified up to 32 DOFs."""
        n_arts = self.forest.n_arts_by_size[size]
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
                self.forest.group_to_art[size],
                self.forest.art_to_world,
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
        n_arts = self.forest.n_arts_by_size[size]
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
                self.forest.group_to_art[size],
                self.forest.art_to_world,
                world_dof_offset,
                self.constraint_count,
            ],
            outputs=outputs,
            block_dim=self.tile_threads,
            device=model.device,
        )

    def _stage4_hinv_jt_diagonal(self, size: int) -> None:
        """Apply diagonal generalized inertia to the active response rows."""
        n_arts = self.forest.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_diagonal_kernel(),
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
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

    def _allocate_packed_rows(self) -> None:
        """The matrix-free sweeps' dense rows, packed by support (:mod:`.packed_rows`): every world's row capacity,
        each row as wide as two of its world's articulations hold; a placeholder without those sweeps."""
        if self._mf_sweeps:
            sizes, worlds = self.forest.articulation_response_dof_count.numpy(), self.forest.art_to_world.numpy()
            offsets = self.forest.articulation_world_dof_offset.numpy()
            # A row's sides follow articulation indices (dense_rows.row_sides); they follow the world's coordinates
            # too, so packed entries ascend with them.
            order = np.lexsort((np.arange(len(worlds)), worlds))
            responding = order[sizes[order] > 0]
            same_world = worlds[responding][1:] == worlds[responding][:-1]
            if np.any(same_world & (offsets[responding][1:] < offsets[responding][:-1])):
                raise RuntimeError("Packed rows need each world's coordinates in articulation order")
            self._packed_support = packed_rows.support_width(sizes, worlds, self.forest.world_count)
            # Two articulations hold every coordinate of the widest world: the rows are laid out at their coordinates.
            self._packed_dense = self._packed_support == self.forest.max_world_dofs
            if self._ink_rows and not self._packed_dense:
                raise RuntimeError("The parallel tiers read their worlds' packed rows at their coordinates")
            sizes = {
                "world": self.forest.world_count,
                "row": self.dense_max_constraints,
                "support": self._packed_support,
            }
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
        for size in self.forest.size_groups:
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
        support.dofs = self.forest.articulation_response_dof_count
        support.offset = self.forest.articulation_world_dof_offset if self._mf_sweeps else self._dummy_is_free_rigid
        support.world = self.forest.art_to_world if self._mf_sweeps else self._dummy_is_free_rigid
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
        skip = 0 if self.options.pgs_warmstart else int(self._ink_skip_rows())
        skip_local = int(self._row_packets is not None and not self.options.pgs_warmstart)
        # A compact or sparse size keeps a placeholder factor: its rows have no dense factor.
        sizes = tuple(sorted(size for size in self.forest.size_groups if self.L_by_size[size].shape[1] == size))
        wp.launch(
            packed_rows.pack_kernel(sizes),
            dim=self.forest.world_count * self.dense_max_constraints,
            inputs=[
                *(self.L_by_size[size] for size in sizes),
                self._row_sides,
                self.forest.articulation_response_dof_count,
                self.forest.art_group_idx,
                self.forest.articulation_world_dof_offset,
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
        """Choose the Jacobi tiers from the class (case K4): their row bound and their in-kernel response.

        The tiers serve the dense rows of a tree beside a free body where no earlier case claims the class
        (:meth:`.route_table.RouteTable.taken`). They form each world's response in the kernel, which bounds the class
        (:meth:`_build_inkernel_response_meta`: one articulation of each of one or two sizes per world, at most
        ``_INKERNEL_MAX_DOF`` coordinates each), under the contact laws they implement
        (:meth:`_admit_parallel_contact_paths`). Their row bound is the dense row capacity, at most
        :data:`.sweep_routes_tiers.TIER_MAX_ROWS`, in whole quads; a test pin (``tiers``) sets it on any class whose
        response they form. ``_ink_rows`` is 0 without tiers and the route table's K4 reason says why or why not
        (``routes.row.reasons``). The local packets and the keyed rows read this before the tiers are built.
        """
        pinned = self._route_pins.get("tiers")
        self._ink_rows, self._ink_sizes = 0, None
        reason = (
            self.routes.taken("K4")
            or capable("tiers", self.laws.set)
            or first_failed(
                (
                    (self._mf_sweeps, "No matrix-free sweep runs whose dense rows the tiers could take."),
                    (pinned is None or bool(pinned), "A test pin builds no tier (tiers=0)."),
                    (
                        not self._preelim_active,
                        "Bilateral pre-elimination corrects the packed rows, which the tiers whiten anew.",
                    ),
                    (
                        bool(pinned) or (self.forest.has_free_rigid_bodies and self.forest.has_non_free_articulations),
                        "The tiers serve a tree beside a free body (case K4).",
                    ),
                    (self._parallel_contact_paths, self._parallel_contact_paths_reason),
                )
            )
        )
        if reason is None:
            sizes, reason = self._build_inkernel_response_meta()
            rows = min(int(self.dense_max_constraints), int(pinned or sweep_routes_tiers.TIER_MAX_ROWS))
            rows -= rows % 4
            if self.routes.claim("K4", sizes is not None and rows > 0):
                self._ink_sizes, self._ink_rows = sizes, rows
            elif sizes is not None:
                reason = f"The dense row capacity {self.dense_max_constraints} holds no tier of four rows."
        self.routes.reasons["K4"] = reason

    def _build_inkernel_response_meta(self) -> tuple[tuple[int, int, int, int] | None, str]:
        """Return the in-kernel response's group sizes and offsets ``(NA, NB, OA, OB)``, or ``None``, and why.

        The parallel tiers form the response of up to two size groups, one articulation of each size per world with
        a primary in every world, at one world DOF offset per size when the response is written in world layout.
        """
        if self.forest.art_to_world is None:
            return None, "No articulation->world map."
        if not (1 <= len(self.forest.size_groups) <= 2):
            return None, f"{len(self.forest.size_groups)} size groups; the in-kernel response takes one or two."
        sizes = sorted((int(s) for s in self.forest.size_groups), reverse=True)
        if sizes[0] > _INKERNEL_MAX_DOF:
            return None, f"Size {sizes[0]} exceeds the in-kernel whitening's {_INKERNEL_MAX_DOF} coordinates."
        art_to_world = self.forest.art_to_world.numpy()
        dof_off = self.forest.articulation_world_dof_offset.numpy()
        meta = -np.ones((self.forest.world_count, 4), dtype=np.int32)
        offsets = [0, 0]
        for col, size in enumerate(sizes):
            g2a = self.forest.group_to_art[size].numpy()
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
        if offsets[0] + sizes[0] > self.forest.max_world_dofs or (nb and offsets[1] + nb > self.forest.max_world_dofs):
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
        row law, so they serve patch friction as they serve point friction. Every tier steps each friction pair
        isotropically, so a sliding pair converges to the serial pair solve's maximal
        dissipation under either law; with per-row steps it did not, which moved
        body-velocity RMS by 0.4-0.9% on two contact-rich patch-friction workloads, and
        converged point tiers sat 3.9 times the serial sweep's p90 error away from its
        fixed point on anymal-D's 33-48-row worlds. Both project joint velocity-limit
        rows by one unilateral law, so the tiers share the serial sweep's fixed point
        there too. Torsion and compliance keep ordinary rows. So does contact
        regularization, under either law: the tiers
        solve the unregularized law, so the worlds they own would drop
        ``pgs_contact_regularization`` while the serial sweeps apply it.
        """
        laws = self.laws.set
        if laws.regularized:
            return False, (
                "Contact regularization keeps serial sweeps and ordinary rows: the parallel tiers, which keyed rows "
                "feed, solve the unregularized law, and the serial sweeps apply pgs_contact_regularization."
            )
        if laws.point:
            return True, "Point friction: parallel tiers and keyed rows serve it."
        if not laws.agnostic:
            return False, "Contact torsion and compliance keep serial sweeps and ordinary rows."
        return True, (
            "Patch friction: parallel tiers and keyed rows serve it. In the parallel (Jacobi) sweeps each tangent "
            "pair projects on its patch ring's pooled normal load, as point pairs project on their own normal, with "
            "one isotropic step per pair, the serial pair solve's friction law."
        )

    def _parallel_velocity_limit_pass(self) -> bool:
        """Whether the parallel tiers end with the last-word velocity-limit pass (``routes.sweep.tiers``).

        The linked (patch) tiers give the joint velocity limits the last word, as the serial sweeps' end-of-iteration
        pass gives it to them: after the final sweep one warp pushes every violated velocity-limit row back onto its
        limit, in row order. It changes nothing once a tier converges and bounds the limit overshoot of a tier its
        sweep budget stops early; it has rows to push only with joint velocity limits.
        """
        return bool(self.laws.friction_anchors_enabled and self.options.enable_joint_velocity_limits)

    def _parallel_tier_sweeps(self, tiers) -> tuple[list[int], str, str]:
        """Return each parallel tier's sweep budget, the rule that set it, and why (``routes.sweep.tiers``).

        ``tiers`` lists ``(rows, min_rows)`` per tier. Every tier, under either friction law, sweeps
        :func:`.sweep_routes_tiers.tier_sweeps` of its row bound: the equal-accuracy rule, from ``pgs_iterations``
        at construction. A world stops earlier under the sweeps' exit law. A test pin (``tier_sweeps``) sets every
        tier's budget.
        """
        pinned = self._route_pins.get("tier_sweeps")
        if pinned is not None:
            return [int(pinned)] * len(tiers), "pinned", f"A test pin sets every tier's budget (tier_sweeps={pinned})."
        reason = (
            f"auto: a tier serving up to R rows sweeps ceil({sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION} * "
            f"pgs_iterations * sqrt(R / 32)) times with pgs_iterations={int(self.options.pgs_iterations)}, at most "
            f"{sweep_routes_tiers.TIER_SWEEPS_CAP}; a world stops earlier under the sweeps' exit law."
        )
        return [sweep_routes_tiers.tier_sweeps(rows, self.options.pgs_iterations) for rows, _ in tiers], "auto", reason

    def _mf_count_or_dummy(self):
        """Matrix-free row counts, or a one-element dummy when the model has no contact rows (never indexed then)."""
        arr = getattr(self, "mf_constraint_count", None)
        return arr if arr is not None else self._dummy_is_free_rigid

    def _ink_skip_rows(self) -> int:
        """The dense rows of the worlds the tiers own, whose response they form in-kernel (0 without tiers)."""
        return int(getattr(self, "_ink_rows", 0))

    def _stage4_hinv_jt_lanes(self, size: int) -> None:
        """The response a warp per articulation, skipping the worlds the local owners solve (route ``lanes``)."""
        n_arts = self.forest.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_lanes_kernel(),
            dim=n_arts * 32,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
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
        n_arts = self.forest.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_rows_kernel(),
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
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
        n_arts = self.forest.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.delassus_entries_kernel(),
            dim=n_arts * self.dense_max_constraints * self.dense_max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
            ],
            outputs=[self.C, self.diag],
            device=model.device,
        )

    def _stage4_finalize_world_diag_cfm(self):
        if self._packed_response:
            return  # The packed rows' build adds CFM with the diagonal.
        if self._sparse_prepare_contacts:
            return  # Contact and limit producers add CFM exactly once.
        if self._sparse_diagonal_contact_solve:
            return  # The sparse-diagonal owner adds its rows' CFM with their diagonals.
        model = self.model
        wp.launch(
            response_builds_dense.cfm_worlds_kernel(),
            dim=self.forest.world_count,
            inputs=[self.constraint_count, self.row_cfm, self._mf_count_or_dummy(), int(self._ink_skip_rows())],
            outputs=[self.diag],
            device=model.device,
        )

    def _stage4_diag_from_JY(self, size: int):
        n_arts = self.forest.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.diagonal_rows_kernel(),
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
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
            dim=self.forest.world_count * self.dense_max_constraints,
            inputs=[
                self.diag_by_size[size],
                self.forest.world_response_group_art_start[size],
                self.forest.world_response_group_to_art[size],
                self.forest.art_group_idx,
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
        if self._sparse_diagonal_contact_solve:
            # The sparse-diagonal owner whitens its rows' dense parts in place and stores their diagonals with cfm:
            # every row, or the compact boundary's limit prefix (its producer whitens its contact rows as it writes).
            size = self._sparse_diagonal_dense_size
            rows = 12 if self._compact_contact_boundary else self.dense_max_constraints
            wp.launch(
                response_builds_rows.SPARSE_PAIR.whiten(),
                dim=self.forest.world_count * rows,
                inputs=[
                    self.L_by_size[size],
                    self._sparse_diagonal_dense_groups,
                    self.constraint_count,
                    self.dense_phase_bounds,
                    int(self._compact_contact_boundary),
                    rows,
                    self.dense_max_constraints,
                    size,
                    self._sparse_diagonal_row_dof,
                    self._sparse_diagonal_row_z,
                    self.row_cfm,
                ],
                outputs=[self.J_by_size[size], self.diag],
                device=self.model.device,
            )
            return 0
        if self._sparse_mass_matrix_size is not None:
            # Sparse producers write the diagonal; unfused controls add CFM separately.
            return 0
        self.diag.zero_()
        if self._hinv_jt_diag_sizes:
            for size in self.forest.size_groups:
                if size in self._hinv_jt_diag_sizes:
                    self._stage4_accumulate_hinv_diag(size)
                else:
                    self._stage4_diag_from_JY(size)
        elif self._hinv_jt_writes_world:
            # The packed rows' build stores the diagonal (_stage4_pack_rows); the CFM pass adds CFM after.
            return 1
        else:
            for size in self.forest.size_groups:
                self._stage4_diag_from_JY(size)
        if self._local_internal_fast_path and not self._hinv_jt_writes_world:
            wp.launch(
                clear_local_solve_diag,
                dim=self.forest.world_count * self.dense_max_constraints,
                inputs=[self.constraint_count, self._local_solve_owner, self.dense_max_constraints],
                outputs=[self.diag],
                device=self.model.device,
            )
        return 0

    def _stage4_compute_rhs_world(self, dt: float):
        """Each dense row's rhs bias (``compute_world_contact_bias``)."""
        if self._sparse_prepare_contacts or self._row_packets is not None or self._keyed_rows is not None:
            return  # These rows write their own right-hand sides.
        wp.launch(
            compute_world_contact_bias,
            dim=self.forest.world_count * self.dense_max_constraints,
            inputs=[
                bind_dense_rows(self),
                self.constraint_count,
                self.dense_max_constraints,
                dt,
                self.options.contact_speculative_scale,
                self.laws.contact_w,
            ],
            device=self.model.device,
        )

    def _stage4_accumulate_rhs_world(self, size: int, velocity=None):
        model = self.model
        n_arts = self.forest.n_arts_by_size[size]
        if velocity is None:
            velocity = self.v_hat
        parallel_rows = model.device.is_cuda and not model.requires_grad and self.forest.is_one_solve_art_per_world
        wp.launch(
            rhs_accum_world_par_row if parallel_rows else rhs_accum_world_par_art,
            dim=n_arts * self.dense_max_constraints if parallel_rows else n_arts,
            inputs=[
                self.constraint_count,
                self.dense_max_constraints,
                self.forest.art_to_world,
                self.forest.articulation_dof_start,
                velocity,
                self.forest.group_to_art[size],
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
                dim=self.forest.world_count * self.dense_max_constraints,
                inputs=[
                    bind_dense_rows(self),
                    self.constraint_count,
                    self.dense_max_constraints,
                    self.v_hat,
                    self.forest.world_dof_indices,
                    self._row_support(),
                    self._packed_J,
                    int(sparse),
                    self._sparse_diagonal_dense_offsets,
                    self._sparse_diagonal_dense_groups,
                    dense_size,
                    self.L_by_size[dense_size] if sparse else self._dummy_world_J,
                    self.J_by_size[dense_size] if sparse else self._dummy_world_J,
                    self._sparse_diagonal_row_dof,
                    self._sparse_diagonal_row_z,
                    self._diagonal_inverse_mass,
                    dt,
                    self.laws.effective_restitution_velocity_threshold,
                    int(self.laws.regularization_enabled),
                ],
                device=self.model.device,
            )
            return

        wp.launch(
            apply_world_contact_restitution_accumulated,
            dim=self.forest.world_count * self.dense_max_constraints,
            inputs=[
                bind_dense_rows(self),
                self.constraint_count,
                self.dense_max_constraints,
                dt,
                self.options.contact_speculative_scale,
                self.laws.effective_restitution_velocity_threshold,
                int(self.laws.regularization_enabled),
            ],
            device=self.model.device,
        )

    def _stage5_prepare_impulses_world(self):
        warmstart_flag = 1 if self.options.pgs_warmstart else 0
        wp.launch(
            prepare_world_impulses,
            dim=self.forest.world_count,
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
        self._stage5_pgs_solve_world_loop(iterations, int(friction_start_iteration), int(iteration_offset))

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
            dim=self.forest.world_count,
            inputs=[
                self.constraint_count,
                self.diag,
                self.C,
                self.rhs,
                self.impulses,
                iterations,
                self.options.pgs_omega,
                self.row_type,
                self.row_parent,
                self.row_mu,
                self.row_w,
                int(self.laws.regularization_enabled),
                friction_start_iteration,
                iteration_offset,
                *units,
                *self._solve_readout.operands(self._pgs_solve_loop_kernel),
            ],
            device=self.model.device,
        )

    def _stage6_prepare_world_velocity(self):
        wp.copy(self.v_out, self.v_hat)

    def _apply_mf_warmstart_velocity(self, velocity: wp.array, *, set_output_to_delta: bool) -> None:
        """Install the carried free-body impulses' velocity into a live velocity state, on every free rigid body the
        rows hold (:func:`.response_builds_rows.free_body_publish_kernel`, one thread per free rigid body)."""
        if not self.options.pgs_warmstart or not self.forest.has_free_rigid_bodies:
            return
        wp.launch(
            response_builds_rows.free_body_publish_kernel,
            dim=self.forest.free_rigid_body_count,
            inputs=[
                self.forest.free_rigid_body_indices,
                self.forest.free_rigid_body_world,
                self.mf_body_a,
                self.mf_body_b,
                self.mf_constraint_count,
                self.mf_Z_a,
                self.mf_Z_b,
                self.mf_row_type,
                self.mf_impulses,
                self._body_factors(),
                int(set_output_to_delta),
            ],
            outputs=[velocity],
            device=self.model.device,
        )

    def _stage6_clamp_rigid_velocity_limits(self):
        """Bound the solved speeds of every free root in the CPU's split mode, where its step ends (the root clamp).

        Split mode bounds free rigid bodies inside its free-body sweep with speed rows where the activation
        gate reserved them; this clamp of the solved velocity bounds every free root, floating bases included,
        without changing a predicted velocity the contacts still act on. CUDA bounds free rigid bodies with speed
        rows alone.
        """
        if not self.forest.has_root_free or not self.laws.has_rigid_body_velocity_limits or self._matrix_free:
            return

        wp.launch(
            clamp_free_root_velocity_limits,
            dim=self.model.articulation_count,
            inputs=[
                self.model.articulation_start,
                self.model.joint_child,
                self.model.body_flags,
                self.forest.articulation_root_is_free,
                self.forest.articulation_root_dof_start,
                self.laws.rigid_body_max_linear_velocity,
                self.laws.rigid_body_max_angular_velocity,
            ],
            outputs=[self.v_out],
            device=self.model.device,
        )

    def _stage6_apply_packed_impulses(self) -> None:
        """``v_out = v_hat + P^T L^-T sum_r Z_r lambda_r`` of every articulation with a dense factor, from the packed
        rows (:func:`.packed_rows.apply_kernel`), a launch per size group."""
        for size in self.forest.size_groups:
            if self.L_by_size[size].shape[1] != size:
                continue  # A compact or sparse size keeps a placeholder: its rows have no dense factor.
            wp.launch(
                packed_rows.apply_kernel(),
                dim=self.forest.n_arts_by_size[size],
                inputs=[
                    self.L_by_size[size],
                    self.forest.group_to_art[size],
                    self.forest.art_to_world,
                    self.forest.articulation_dof_start,
                    self.forest.articulation_response_dof_count,
                    size,
                    self._row_sides,
                    self.constraint_count,
                    self.forest.articulation_world_dof_offset,
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
        n_arts = self.forest.n_arts_by_size[size]
        wp.launch(
            response_builds_dense.apply_coordinates_kernel(),
            dim=int(n_arts * size),
            inputs=[
                self.forest.group_to_art[size],
                self.forest.art_to_world,
                self.forest.articulation_dof_start,
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
        self._apply_mf_warmstart_velocity(self.v_out, set_output_to_delta=False)
        self._mf_pgs_solve(
            self.options.pgs_iterations,
            friction_start_iteration=self._contact_friction_start_iteration(self.options.pgs_iterations),
        )

    def _mf_pgs_setup(self, state_aug: State, dt: float):
        """MF PGS setup: whiten the free-body rows through each body's held factor, their diagonals and RHS."""
        model = self.model
        self.mf_rhs.zero_()
        self.mf_eff_mass_inv.zero_()
        wp.launch(
            response_builds_rows.free_body_rows_kernel,
            dim=self.forest.world_count * self.mf_max_constraints,
            inputs=[
                bind_mf_rows(self),
                self.mf_constraint_count,
                self._body_factors(),
                int(self.forest.has_prescribed_response),
                self.v_hat,
                self.laws.rigid_body_max_depenetration_velocity,
                self.options.pgs_cfm,
                self.options.pgs_beta,
                self.laws.contact_w,
                dt,
                self.options.contact_speculative_scale,
                self.laws.effective_restitution_velocity_threshold,
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
        if self.forest.has_free_rigid_bodies:
            free = np.flatnonzero(self.forest.is_free_rigid_host != 0)
            factor_index = self.forest.art_group_idx.numpy()[free]
            starts = self.forest.articulation_dof_start.numpy()[free]
            held = factor_index >= 0
            dof_factor[starts[held]] = factor_index[held]
        factors = BodyFactors()
        factors.body_to_articulation = self.forest.body_to_articulation
        factors.factor_index = self.forest.art_group_idx
        factors.dof_start = self.forest.articulation_dof_start
        factors.dof_factor = wp.array(dof_factor, dtype=wp.int32, device=device)
        lower = self.L_by_size.get(6) if self.forest.has_free_rigid_bodies else None
        factors.L = lower if lower is not None else wp.zeros((1, 6, 6), dtype=wp.float32, device=device)
        # Every articulation's dense factor, by address: the size groups' factors are allocated once, with the buffers.
        art_count = max(int(self.model.articulation_count), 1)
        first_art = np.full(max(int(self.model.joint_dof_count), 1), -1, dtype=np.int32)
        sizes = np.zeros(art_count, dtype=np.int32)
        offsets = np.zeros(art_count, dtype=np.int32)
        addresses = np.zeros(art_count, dtype=np.uint64)
        if int(self.model.articulation_count) > 0:
            starts = self.forest.articulation_dof_start.numpy()
            counts = self.forest.articulation_response_dof_count.numpy()
            world_offsets = self.forest.articulation_world_dof_offset
            offsets[:] = world_offsets.numpy() if world_offsets is not None else 0
            sizes[:] = counts
            responding = np.flatnonzero(counts > 0)
            first_art[starts[responding]] = responding
            for size, lower_group in self.L_by_size.items():
                if lower_group.shape[1] != size or size not in self.forest.group_to_art:
                    continue  # A compact or sparse size keeps a placeholder: no dense factor.
                arts = self.forest.group_to_art[size].numpy()
                addresses[arts] = lower_group.ptr + np.arange(len(arts), dtype=np.uint64) * np.uint64(size * size * 4)
        factors.first_art = wp.array(first_art, dtype=wp.int32, device=device)
        factors.size = wp.array(sizes, dtype=wp.int32, device=device)
        factors.world_offset = wp.array(offsets, dtype=wp.int32, device=device)
        factors.address = wp.array(addresses, dtype=wp.uint64, device=device)
        return factors

    def _body_factors(self) -> BodyFactors:
        """The free bodies' factors and coordinates (:meth:`_build_body_factors`)."""
        return self._body_factors_data

    def _mf_pgs_solve(
        self,
        iterations: int,
        friction_start_iteration: int | None = None,
        iteration_offset: int = 0,
    ):
        """MF PGS solve with given iteration count."""
        model = self.model
        if friction_start_iteration is None:
            friction_start_iteration = self._contact_friction_start_iteration(iterations)

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
            dim=self.forest.world_count,
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
                self.forest.body_to_articulation,
                self.forest.articulation_dof_start,
                iterations,
                self.options.pgs_omega,
                int(self.laws.regularization_enabled),
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
        if not self.forest.free_root_joint_count:
            return
        model = self.model
        wp.launch(
            remove_free_root_transport_from_qdd,
            dim=self.forest.free_root_joint_count,
            inputs=[
                self.forest.free_root_joint_indices,
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
            next_refresh = ((self._step + 1) % self.options.update_mass_matrix_interval) == 0
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
                    self.options.angular_damping,
                ],
                outputs=[state_out.joint_q, state_out.joint_qd],
                device=model.device,
            )

            self._stage7_update_kinematics(state_out, state_aug)

        self.integrate_particles(model, state_in, state_out, dt)

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

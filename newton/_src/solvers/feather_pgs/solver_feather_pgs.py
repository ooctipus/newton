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
from collections.abc import Callable
from contextlib import contextmanager
from functools import cache
from typing import ClassVar, Literal

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags
from ...sim.contacts import contact_capacity_blocks
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
from . import environment_flags as _environment_flags
from . import execution_plan as _execution_plan
from . import (
    field_group,
    host_index,
    kernel_template,
    packed_rows,
    prefix_rows,
    response_builds_rows,
    sweep_routes,
    sweep_routes_diag,
    sweep_routes_local,
    sweep_routes_sparse,
    sweep_routes_tiers,
    sweep_units,
)
from . import route_table as _route_table
from . import row_metadata as _row_metadata
from .body_cut import BodyCut
from .configuration import check_options, kernel_choices
from .contact_input import ContactInput
from .contact_input import bind as bind_contact_input
from .contact_input import bind_routes as bind_contact_routes
from .contact_row_plan import SHARED_METADATA_PRODUCERS, contact_row_plan
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
    PGS_CONSTRAINT_TYPE_TORSION,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
    allocate_connect_slots,
    allocate_joint_velocity_limit_slots,
    allocate_link_slots,
    allocate_mimic_slots,
    allocate_rigid_velocity_limit_slots,
    allocate_world_contact_slots,
    build_joint_limit_rows_for_size,
    build_mf_contact_rows,
    classify_local_solve_worlds,
    classify_parallel_sweep_tiers,
    clear_sparse_diagonal_response_prefix,
    compact_local_pair_candidates,
    compute_contact_linear_force_from_impulses,
    compute_world_contact_bias,
    finalize_constraint_counts_with_status,
    gather_dense_warmstart,
    gather_mf_warmstart,
    local_solve_launch_gate,
    pack_contact_linear_force_as_spatial,
    populate_connect_J_for_size,
    populate_joint_velocity_limit_J_for_size,
    populate_mimic_J_for_size,
    populate_rigid_velocity_limit_rows,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_fused_diagonal_joint_limits,
    prepare_world_contact_rows,
    reset_world_warmstart_buffers,
    snapshot_contact_warmstart,
    snapshot_dense_phase_bound,
    write_link_rows,
)
from .path_selection import (
    applied_options,
    measurement_selection,
    options_selection,
    route_selection,
)
from .placement import Placement
from .route_table import RouteTable, capable
from .row_build import contact_row_prelude, populate_world_J_masked
from .rows import JOINT_LIMIT_WARPS_PER_BLOCK, RowLaws
from .solve_policy import SolvePolicy, SolveReadoutRecord
from .solver_fields import (
    DOF_WORK,
    SPARSE_CONTACT_RECORDS,
    SPARSE_ROWS,
    STAGE3_SNAPSHOTS,
    Connects,
    Links,
    LocalOwners,
    Mimics,
)
from .sparse_contact import (
    CONTACT_RECORD_FLOATS,
    CONTACT_RING_FLOATS,
    finalize_sparse_joint_limit_rows,
)
from .sparse_mass_matrix import (
    _SparseRowSupport,
    row_supports,
)
from .staging import arch_budget, factor_budget
from .step_plan import StepPlan, first_failed
from .sweep_routes_sparse import _SPARSE_LOCAL_WORKSPACE_BYTES
from .tree_factor import TreeFactor
from .tree_plan import TreePlan, local_parents

# Parallel tiers grid-stride over per-tier world lists from this many tiers on: one classify launch against
# (tiers - 1) x worlds idle block entries.
_TIER_LIST_MIN_TIERS = 3
_FPGS_SYNC_TIMINGS = os.environ.get("FEATHER_PGS_SYNC_TIMINGS", "0").lower() in {"1", "true", "yes", "on"}
_FPGS_SYNC_TIMINGS_START = int(os.environ.get("FEATHER_PGS_SYNC_TIMINGS_START", "20"))
_FPGS_SYNC_TIMINGS_COUNT = max(int(os.environ.get("FEATHER_PGS_SYNC_TIMINGS_COUNT", "1")), 0)
_SMALL_DOF_THRESHOLD_DEFAULT = 12
# Threads per block of the tiled factor, triangular-solve and response kernels.
_TILE_THREADS = 64


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
_estimate_hinv_jt_shared_memory = _execution_plan._estimate_hinv_jt_shared_memory
_select_hinv_jt_chunk_size = _execution_plan._select_hinv_jt_chunk_size
_FeatherPGSExecutionPlan = _execution_plan._FeatherPGSExecutionPlan


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

    ``placement.constraint_overflow`` is a device boolean array with one entry per solver
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
    :meth:`get_path_selection`. Every tree likewise takes the least armature that keeps a rotation about a thin body's
    own axis, far from the articulation's origin, above what float32 assembles (:func:`.conditioning.assembly_floor`).
    Give such joints armature to choose it yourself.

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
    #   storage "dense": no packed (native sparse) factor.
    #   grouping "forest" or "per_size": the held families.   star "held": a star's diagonal family in the held tree.
    #   producer "compact_jacobian" or "general": no local packets or keyed rows.
    #   claims: the runtime claims kept, of ("resolved_worlds",).
    #   active_rows False: full-order sparse sweeps.           contact_units None: unfused sparse contact triplets.
    #   transaction "single" or "staged": fused triplets'.     mf_storage "streamed": streamed matrix-free impulses.
    #   repair_skip, repair_classify, predicted_twists False: the held tree's launch mapping without that step.
    #   row_support False: sparse rows on explicit indices.     placement "general": case K6 (route_table.RouteTable).
    #   tiers: the Jacobi tiers' row bound, on any class whose response they form (0: no tiers).
    #   tier_sweeps: every Jacobi tier's sweep budget, in place of the equal-accuracy rule.
    #   hinv_jt_kernel: a dense response kernel (CUDA).
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

    @deprecate_nonkeyword_arguments(last_positional="friction_anchor_beta")
    def __init__(
        self,
        model: Model,
        *,
        angular_damping: float = 0.05,
        update_mass_matrix_interval: int = 1,
        enable_contact_friction: bool = True,
        contact_friction_gap_threshold: float = float("inf"),
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
        enable_restitution: bool = True,
        same_articulation_contact_gap_gate: float = 0.0,
        articulation_pair_contact_gap_gate: float = 0.0,
        pgs_warmstart_decay: float = 1.0,
        warn_constraint_overflow: bool = True,
        friction_anchor_beta: float | None = None,
        pgs_solve_readout: Literal["off", "sweeps", "trace"] = "off",
        contact_torsion: bool = False,
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
            contact_torsion: Resist spin about each contact's normal: a friction contact takes a spin row beside its
                friction pair, and the pair and the spin row share the elliptic cone
                ``|(t1 / mu, t2 / mu, s / mu_t)| <= lambda_n`` (the soft-finger contact). ``mu_t`` [m] is the
                contact's torsional coefficient, the mean of its shapes' ``Model.shape_material_mu_torsional``
                (as ``mu`` is the mean of their ``shape_material_mu``); a zero coefficient pins the spin. Under
                persistent friction patches each anchor's spin row shares the region's capacity as its pair shares
                the region's friction. Point contacts (spheres, fingertips) are its use: a flat contact's several
                points already resist spin through their friction's lever arms. Off by default.
            contact_compliance: Contacts of positive ``Contacts.rigid_contact_stiffness`` [N/m] are material
                contacts: each pushes with ``F = max(0, -k phi - c u)`` at the step's end (backward Euler), with the
                contact's ``rigid_contact_damping`` [N s/m] (no dashpot across an open gap), and its tangents take
                the contact's ``rigid_contact_friction`` scale (zero meaning one). Zero stiffness keeps the rigid
                law. The law is the row law's compliance ``eps = 1 / (dt (dt k + c))``: a material normal takes
                neither restitution nor ``pgs_contact_regularization``. The contacts must carry the three
                per-contact arrays. Every placement takes it: each sweep steps a material normal by the weight
                ``d / (d + eps)`` of the diagonal it forms. Defaults to False.
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
            friction_anchor_beta (float | None, optional): Set the positional correction strength
                for persistent patch friction. ``None`` enables it at 0.2 for the default solver;
                the experimental contact material law instead retains velocity-only point friction and warns.
                Zero explicitly selects velocity-only point friction. Positive values
                group compatible contacts on a body pair into regions,
                select up to two friction locations per region, and share the total
                normal impulse equally between those anchors. Normal contacts are preserved.
                Twisting resistance comes only from the separation between the anchors;
                a single-anchor region has no independent torsional stiction constraint
                unless ``contact_torsion`` gives each anchor a spin row.
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

        Raises:
            ValueError: For a model that records gradients (``requires_grad``): FeatherPGS has no gradients.
        """
        if model.requires_grad:
            raise ValueError("FeatherPGS has no gradients: build the model with requires_grad=False.")
        # Every option as requested, before any default resolution; get_path_selection() reports it.
        requested_options = dict(locals())
        # Read the FEATHER_PGS_* environment once; undeclared names are recorded, not rejected.
        self._environment_record = _environment_flags.record()
        # The test pins in force at construction, which every admission reads and the route record lists.
        self._route_pins = dict(type(self)._route_pins)
        # The solver's own model, a pure forest (:mod:`.body_cut`): everything below reads it.
        self.cut = BodyCut.of(model)
        model = self.cut.model
        super().__init__(model)
        # Every scalar option coerced and checked once (configuration.OPTION_RULES).
        self.options = options = check_options(requested_options)

        # The dense kernels from the device (configuration.kernel_choices).
        choices, self._kernel_choice_reasons = kernel_choices(self._route_pins, is_cpu=model.device.is_cpu)

        # Every sweep compiles the exit law and the policy's readout.
        self._solve_policy = SolvePolicy(readout=options.pgs_solve_readout)
        self._solve_readout = SolveReadoutRecord(self._solve_policy)

        self._ws_prev_dt = 0.0
        self._grouped_response = False

        self._nvtx = nvtx
        self._debug_buffers_enabled = bool(pgs_debug)
        # External NaN dumpers read the stage-3 snapshots; nothing in the solver
        # does. They cost three copies per substep, so they are opt-in.
        self._debug_stage3_snapshot_flag = os.environ.get("FEATHER_PGS_DEBUG_SNAPSHOTS") == "1"
        self._debug_stage3_snapshots = self._debug_stage3_snapshot_flag or self._debug_buffers_enabled
        # The solver's device fields (solver_fields), on the model's device.
        self._arrays = SolverArrays(self, model.device)

        small_dof_threshold = _SMALL_DOF_THRESHOLD_DEFAULT

        self.tile_threads = _TILE_THREADS
        # Launches over the contact capacity (the force export, the warm-start snapshot
        # and gathers, the keyed-row key scatters) grid-stride over about one wave of blocks.
        self._contact_capacity_blocks = contact_capacity_blocks(model.device)

        self.hinv_jt_kernel = choices["hinv_jt_kernel"]
        self.small_dof_threshold = small_dof_threshold

        self._step = 0
        # The tree factor: the augmented state, the mass-update cadence and the FK/ID cache now; its masks, drives,
        # factors and evaluator as construction reaches them.
        self.factor = TreeFactor(model, options, nvtx=nvtx)

        self.forest: ForestPlan | None = None
        # The model's joint forest (Model.joint_ancestor): conditioning depths, body maps, CRBA's ancestor test.
        forest = joint_forest(model)
        self.factor.apply_kinematic_state(self.factor.kinematic_state(forest))
        # The forest plan: topology, groups, root kinds, world mapping, body maps. The matrix-free formulation
        # elides fully-kinematic free bases from the compact response mapping of its rows.
        self.forest = ForestPlan.build(self.cut, forest, self.factor.kinematic_dof_mask_host)
        self._compact_contact_jacobian = bool(
            model.device.is_cuda
            and self.forest.body_response_dof_mask is not None
            and self.forest.size_groups
            and max(self.forest.size_groups) <= _CONTACT_JACOBIAN_MAX_DOF
        )
        # The row laws' tables: the contact law, speed limits, the joint-limit tables, the mimic and connect plans,
        # contact torsion and the materials.
        self.laws = RowLaws.build(model, self.forest, options)
        # The placement: where the rows live, their capacities and status.
        self.placement = Placement(model, options, self.forest, self.laws, self.factor)
        self.factor.plan_blocks(self.forest, self.laws)
        # Axis 2 of the route table, the row laws, which every owner's admission reads (route_table.capable).
        self.laws.decide(preelimination=self.factor.eliminates)
        laws = self.laws.set
        # The passive joint forces and the per-coordinate drive rows the held tree's admission reads.
        self.factor.plan(self.forest, self.laws)
        self._setup_fused_diagonal_joint_limits(model)
        from . import held_tree  # noqa: PLC0415
        from .scalar_branches import plan_scalar_branches  # noqa: PLC0415

        # Axis 3 of the route table, the placement case: the owners below, in precedence order (route_table.CASES),
        # each test their own fit once no earlier case holds the class, and the first that fits claims it.
        self.routes = RouteTable(self._route_pins)
        # Case K2, a tree beside a star: the sparse diagonal.
        declined = self.routes.taken("K2")
        sparse_diagonal_pair = (
            None
            if declined
            else self._select_sparse_diagonal_response_pair(
                factor_dense_contract=self.placement.fused_diagonal_joint_limits
            )
        )
        held = None
        if sparse_diagonal_pair is not None:
            # Its compact diagonal is the held tree's representation: a candidate only where the held tree takes
            # the dense groups with it, the diagonal group as scalar branches or a held family.
            scalar, _ = plan_scalar_branches(self, size=sparse_diagonal_pair[0])
            held = held_tree.compact_decline(self, sparse_diagonal_pair[0], scalar=scalar is not None)
            if held is not None:
                sparse_diagonal_pair = None
        self.placement.sparse_diagonal_contact_solve = self.routes.claim("K2", sparse_diagonal_pair is not None)
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
            self.placement.fused_diagonal_joint_limits = False
            self.placement.fused_diagonal_joint_limit_sizes = frozenset()
            self.placement.fused_diagonal_limit_dof_mask_host.fill(0)
        self.placement.sparse_diagonal_response_size = (
            sparse_diagonal_pair[0] if sparse_diagonal_pair is not None else None
        )
        self.placement.sparse_diagonal_dense_size = sparse_diagonal_pair[1] if sparse_diagonal_pair is not None else 0
        # Persistent patches allocate one tangent pair per surviving anchor, so their contact rows are not
        # uniform normal/tangent/tangent triples; keep the triple schedule for point friction only.
        self.placement.sparse_diagonal_contact_triples = bool(
            self.placement.sparse_diagonal_contact_solve
            and self.options.enable_contact_friction
            and self.options.contact_friction_gap_threshold == math.inf
            and not self.laws.friction_anchors_enabled
        )
        # Patch contacts are units instead: a normal row, followed by its two tangent rows only when the contact
        # owns an anchor. The unit schedule replaces every stride-3 assumption of the triple schedule.
        self.placement.sparse_diagonal_contact_units = bool(
            self.placement.sparse_diagonal_contact_solve
            and self.options.enable_contact_friction
            and self.options.contact_friction_gap_threshold == math.inf
            and self.laws.friction_anchors_enabled
        )
        self.placement.sparse_diagonal_speculative_contact_batches = bool(
            (self.placement.sparse_diagonal_contact_triples or self.placement.sparse_diagonal_contact_units)
            and self.placement.sparse_diagonal_dense_size + 2 <= 16
        )
        self.placement.sparse_diagonal_dense_offsets = wp.array(
            sparse_diagonal_pair[2] if sparse_diagonal_pair is not None else np.zeros(1, dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        self.placement.sparse_diagonal_dense_groups = wp.array(
            sparse_diagonal_pair[3] if sparse_diagonal_pair is not None else np.zeros(1, dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        )
        self.placement.dense_internal_max_rows = self.placement.internal_rows_per_world(model)
        self.placement.hinv_jt_computes_diag = not self.factor.eliminates
        self._execution_plan = _FeatherPGSExecutionPlan.build(
            self.forest.size_groups,
            max_constraints=self.placement.dense_max_constraints,
            max_shared_memory=int(getattr(model.device, "max_shared_memory_per_block", 0)),
            hinv_jt_kernel=self.hinv_jt_kernel,
            small_dof_threshold=self.small_dof_threshold,
            tile_threads=self.tile_threads,
            diagonal_mass_sizes=self.forest.diagonal_mass_sizes,
        )
        self.placement.hinv_jt_diag_sizes = frozenset(
            size
            for size in self.forest.size_groups
            if self.placement.hinv_jt_computes_diag
            and (self._execution_plan.use_diagonal_mass(size) or self._execution_plan.use_tiled_hinv_jt(size))
        )
        self._paired_response_candidate_sizes = frozenset(
            size
            for size in self.forest.size_groups
            if model.device.is_cuda
            and not self.factor.eliminates
            and size >= _PERSISTENT_HINV_JT_MIN_DOF
            and size + 6 <= 32
            and self._execution_plan.use_tiled_hinv_jt(size)
            and self._execution_plan.hinv_jt_chunk_size(size) < self.placement.dense_max_constraints
        )
        self._local_solve_max_rows = min(_LOCAL_SOLVE_MAX_ROWS, self.placement.dense_max_constraints)
        self._local_residual_max_rows = min(_LOCAL_RESIDUAL_MAX_ROWS, self.placement.dense_max_constraints)
        self._local_residual_mf_max_rows = min(_LOCAL_RESIDUAL_MF_MAX_ROWS, self.placement.mf_max_constraints)
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
        self.placement.local_solve_owner = wp.zeros(self.forest.world_count, dtype=wp.int32, device=model.device)
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
        self.placement.local_internal_fast_path = self.routes.claim(
            "K3", local_base_supported and any(count > 0 for count in self._local_internal_art_counts.values())
        )
        self.routes.reasons["K3"] = local_decline or (
            "Worlds with one small non-free articulation (and at most one free body) solve in local shared memory."
            if self.placement.local_internal_fast_path
            else "No world holds one non-free articulation of at most "
            f"{_LOCAL_INTERNAL_MAX_DOF} DOFs whose local staging fits the device."
        )
        self._contact_material_placeholder = self._arrays.placeholder(wp.float32, (1,))
        self._bind_contact_material(None)
        # The parallel matrix-free tiers pool each friction pair's cone over its patch
        # ring and keyed rows write the patch row law, so both can serve point and patch
        # friction (Jacobi sweeps, not the serial patch sweep).
        self._parallel_contact_paths, self._parallel_contact_paths_reason = self._admit_parallel_contact_paths()
        self.placement.jy_world_aliased = self._detect_jy_world_identity()
        self._allocate_common_buffers(model)
        self._compact_diagonal_mass_size = self.placement.sparse_diagonal_response_size
        self.placement.row_watermark = bool(row_watermark)
        self._setup_sparse_mass_matrix(model)
        # The matrix-free sweeps read the dense rows packed (packed_rows): decided before the buffers, since that
        # placement allocates neither a size group's response nor the world layout.
        self.placement.mf_sweeps = bool(
            self.routes.case not in ("K1", "K2")
            and self.forest.world_count > 0
            and self.forest.max_world_dofs > 0
            and self.placement.mf_max_constraints > 0
        )
        if self.placement.mf_sweeps:
            self.placement.jy_world_aliased = False
            self.placement.hinv_jt_diag_sizes = frozenset()
        # The compact boundary owns the rows and response of every world the sparse-diagonal owner pairs with a
        # six-coordinate dense articulation (decided before the buffers: it clears its own rows, so their Jacobian
        # needs no second buffer). It writes every contact row law: point and patch friction, contact torsion's spin
        # rows and material contacts.
        from .compact_contact import supported  # noqa: PLC0415

        declined = (
            None
            if supported(self)
            else "Declined: requires the six-coordinate sparse-diagonal triple or unit contract and no row checks."
        )
        self.placement.compact_contact_boundary = declined is None
        self.placement.compact_contact_reason = (
            declined or "Admitted: the six-coordinate sparse-diagonal triples or units."
        )
        self._allocate_buffers(model)
        self.placement.allocate_world_rows(model)
        # Bilateral pre-elimination corrects the grouped response after H^-1 J^T,
        # so that path must retain the existing post-correction world gather.
        self.placement.hinv_jt_writes_world = (
            not self.placement.jy_world_aliased and not self.factor.eliminates and self.routes.case not in ("K1", "K2")
        )
        self.placement.hinv_jt_tiled_writes_group = (
            not self.placement.hinv_jt_writes_world or self.options.pgs_warmstart
        )
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
        # Patch contacts are units of one row, or three (four with a spin row) when they own an anchor;
        # the factor sweep walks them with the generic path's row order and
        # arithmetic, solving each tangent pair once.
        self._factor_coordinate_contact_units = bool(
            self._paired_factor_coordinates and laws.contact == "patch" and not laws.compliance
        )
        self._paired_factor_primary_groups_by_world = None
        self._paired_factor_secondary_groups_by_world = None
        self._paired_factor_primary_offsets_by_world = None
        self._paired_factor_secondary_offsets_by_world = None
        if response_pair is not None:
            if self._paired_factor_coordinates:
                primary_arts = self.forest.group_to_art_host[response_pair[0]]
                secondary_arts = self.forest.group_to_art_host[response_pair[1]]
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
        if not self.placement.hinv_jt_writes_world and response_pair is None:
            self.placement.hinv_jt_diag_sizes = frozenset()
        self.placement.allocate_sparse_diagonal_rows()
        self.placement.allocate_free_body_rows(model)
        self.factor.build_body_factors()
        self.placement.mf_target_velocity = (
            wp.zeros_like(self.placement.mf_rhs)
            if self.forest.has_prescribed_response
            else self._arrays.placeholder(wp.float32, (1, 1))
        )
        self.factor.scatter_armature()
        self._plan_tier_rows()
        self.placement.allocate_packed_rows()
        # The producers of the local owners' rows (case K3) and of the tiers' rows (case K4), each admitted by its
        # own fit: limit prefix and contact packets built in place for worlds of one limited articulation and at
        # most one other (row_packets.py), and keyed rows (keyed_rows.create_owner). Both write the point and the
        # patch row law.
        from . import row_packets  # noqa: PLC0415

        self.placement.row_packets = None
        self.placement.row_packets_reason = (
            (
                None
                if self._route_pins.get("producer") in (None, "local_packets")
                else "A test pin selects another contact-row producer."
            )
            or capable("row_packets", laws)
            or row_packets.decline(self)
        )
        if self.placement.row_packets_reason is None:
            self.placement.row_packets = row_packets.create_owner(self)
            self.placement.row_packets_reason = (
                "Admitted: worlds of one limited articulation and at most one other build packets in place."
                if self.placement.row_packets is not None
                else "The local packet law does not admit this topology."
            )
        self.placement.keyed_rows = None
        if not self._parallel_contact_paths:
            self.placement.keyed_rows_reason = self._parallel_contact_paths_reason
        elif (declined := capable("keyed_rows", self.laws.set)) is not None:
            self.placement.keyed_rows_reason = declined
        elif self._route_pins.get("producer") not in (None, "keyed_rows"):
            self.placement.keyed_rows_reason = "A test pin selects another contact-row producer."
        else:
            from .keyed_rows import create_owner as create_kinetic_rows_owner  # noqa: PLC0415

            self.placement.keyed_rows = create_kinetic_rows_owner(self)
            self.placement.keyed_rows_reason = (
                "Admitted: the in-kernel parallel sweep builds keyed rows for its bounded worlds."
                if self.placement.keyed_rows is not None
                else "Keyed rows need the Jacobi tiers (routes.sweep.tiers) and a plain matrix-free policy."
            )
        self._init_tiled_kernels(model)
        self._init_size_group_streams(model)
        self.placement.allocate_placeholders()

        self.placement.allocate_status()

        # Scalar branches take a diagonal-mass group of world-fixed roots over
        # one-coordinate leaves (scalar_branches.py), which the held-tree owner
        # then publishes: the compact diagonal's admission planned them.
        self._scalar_branches, self._scalar_branches_reason = plan_scalar_branches(self)

        # The tree factor's evaluator: the held tree.
        self.factor.install(self)

        # Worlds whose every candidate row separates at the predictor skip row
        # allocation entirely (resolved_worlds.py); the mask is empty when the
        # solver's row law is outside what the classifier evaluates.
        from .resolved_worlds import ResolvedWorlds  # noqa: PLC0415

        classifier = ResolvedWorlds(self)
        self._resolved_worlds_owner = classifier if classifier.enabled else None
        self._resolved_worlds = classifier.resolved
        self._resolved_worlds_reason = classifier.reason

        self.placement.place_dense_response(paired_coordinates=self._paired_factor_coordinates)
        # The route: the row of the route table the class takes. The step plan and the route record read it.
        diagonal = self.routes.case == "K2"
        route = self.routes.finish(
            self.laws.set,
            producer=contact_row_plan(self).producer,
            packed=self.placement.packed_response,
            tier_rows=self.placement.ink_skip_rows(),
            sizes={
                "packed": self.placement.sparse_mass_matrix_size,
                "diagonal": self.placement.sparse_diagonal_response_size if diagonal else None,
                "dense": self.placement.sparse_diagonal_dense_size if diagonal else None,
            },
        )
        # The step's per-stage dispatch tables, which step() and the route record read, and the kernels they launch.
        self._plan = StepPlan.build(route, self)
        self.placement.build_response_kernels(self._plan, self._execution_plan, self.tile_threads)
        # Record the applied options once; get_path_selection() returns detached copies.
        self._applied_options = applied_options(self, SolverFeatherPGS.__init__, requested_options)

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
        contact-row producer, packed rows, tier rows, the case owners' sizes and
        each case's reason), ``routes.stages`` the step plan's per-size stage routes
        (factor, predictor solve, response) that ``step()`` launches,
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
        ``contact_consumers.max_blocks`` is the block cap of the launches over the
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
        selection = {
            "sparse_mass_matrix": "on" if self.routes.row.case == "K1" else "off",
            "cached_dynamics": "on",
            "publication": "cached",
            "cached_prediction": "tree",
            "reasons": {
                "sparse_mass_matrix": self.routes.row.reasons["K1"],
                "publication": "Paired cache owns next-state publication.",
            },
        }
        selection["routes"] = route_selection(self)
        selection["measurements"] = measurement_selection(self)
        selection["options"] = options_selection(self)
        selection["environment"] = copy.deepcopy(self._environment_record)
        return selection

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
        self.cut.refresh()
        # Derive the kinematic state first: a body-flag change the solver
        # cannot absorb raises here, before any mask, cache or owner changes.
        kinematic = (
            self.factor.kinematic_state(self.forest.joint_forest, sparse_velocity_limits=self._sparse_velocity_limits)
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
        if self.placement.row_packets is not None:
            self.placement.row_packets.validate_notification(flags)
        if self.placement.keyed_rows is not None:
            self.placement.keyed_rows.validate_notification(flags)
        self.factor.notify_model_changed(flags, kinematic)
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
        if world_mask is not None and world_mask.shape[0] != self.forest.world_count:
            raise ValueError(
                f"world_mask has length {world_mask.shape[0]}, expected {self.forest.world_count} (one entry per world)."
            )
        if self.forest.world_count == 0:
            return

        self.factor.held.invalidate(world_mask)

        self.placement.reset_status(world_mask)
        if self.laws.friction_anchors_enabled:
            # Use the world associated with each stored patch anchor.
            self.laws.friction_patches.reset(world_mask)
        self.factor.reset_cache(self.cut.mirror(state), world_mask)
        self._clear_warmstart_history(world_mask)

    def _clear_warmstart_history(self, world_mask: wp.array | None) -> None:
        """Discard solver impulses independently of latched capacity status."""
        if self.forest.world_count == 0:
            return
        prev_mf_impulses = self.placement.ws_prev_mf_impulses
        if not self.options.pgs_warmstart and prev_mf_impulses is None:
            return

        dense_impulses = (
            self.placement.impulses if self.options.pgs_warmstart else self.placement.dummy_contact_impulses
        )
        if prev_mf_impulses is None:
            prev_mf_impulses = self.placement.dummy_contact_impulses
        prev_mf_row_type = self.placement.ws_prev_mf_row_type
        prev_mf_row_parent = self.placement.ws_prev_mf_row_parent
        if prev_mf_row_type is None:
            # Use the dedicated reset sink, not dummy_contact_row_type: the
            # kernel writes -1 into this buffer and the shared dummy must stay
            # all zeros for its other consumers.
            prev_mf_row_type = self.placement.reset_dummy_mf_row_type
        if prev_mf_row_parent is None:
            prev_mf_row_parent = self.placement.reset_dummy_mf_row_parent
        # ws_prev_slot_sorted is intentionally NOT cleared here:
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
        if self.options.pgs_warmstart and self.placement.ws_prev_dense_impulses is not None:
            # Neutralize the dense identity-matched carry for reset worlds
            # (the kernel is width-generic: it zeroes impulses and writes -1
            # into the row-type table).
            wp.launch(
                reset_world_warmstart_buffers,
                dim=self.forest.world_count,
                inputs=[
                    world_mask,
                    self.placement.dummy_contact_impulses,
                    self.placement.ws_prev_dense_impulses,
                    self.placement.ws_prev_dense_row_type,
                    self.placement.ws_prev_dense_row_parent,
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
        self.cut.set_enabled(joint, enabled)
        self.laws.set_links_enabled()
        self.notify_model_changed(ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)

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
        self.cut.set_anchors(joint, parent_anchor, child_anchor)
        self.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

    def loop_joint_anchors(self, joint: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the current ``(parent_anchor, child_anchor)`` of a loop-closing BALL joint [m]."""
        return self.cut.anchors(joint)

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
        self.factor.notify_state_changed(self.cut.trees(articulations))

    def _setup_fused_diagonal_joint_limits(self, model) -> None:
        """Select independent position limits that can remain outside the dense row system."""
        self.placement.fused_diagonal_joint_limits = False
        self.placement.fused_diagonal_joint_limit_sizes = frozenset()
        self.placement.fused_diagonal_limit_dof_mask_host = np.zeros(model.joint_dof_count, dtype=np.int32)
        supported = bool(
            model.device.is_cuda
            and model.joint_dof_count > 0
            and self.options.enable_joint_limits
            and self.options.enable_contact_friction
            and not self.options.enable_joint_velocity_limits
            and not self.options.pgs_warmstart
            and not self.options.pgs_warmstart
            and not self.factor.eliminates
            and not self.forest.has_free_rigid_bodies
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
        self.placement.fused_diagonal_limit_dof_mask_host[dof[limited[art]]] = valid[limited[art]]
        # Add each size where a world first has it, as the per-world set did.
        selected_sizes: set[int] = {int(size) for size in host_index.ordered_unique(diagonal_size[limited])}
        self.placement.fused_diagonal_joint_limit_sizes = frozenset(selected_sizes)
        self.placement.fused_diagonal_joint_limits = bool(selected_sizes)

    def _setup_sparse_mass_matrix(self, model: Model) -> None:
        """Select sparse articulation factors with independent rigid-body response blocks."""
        self.placement.sparse_mass_matrix_size = None
        self._sparse_velocity_limits = False
        self.placement.sparse_prepare_contacts = False
        self.placement.sparse_row_support = _SparseRowSupport()
        self._sparse_contact_triplets = False
        # Route details for get_path_selection(), set where the sparse owner is admitted.
        self._sparse_active_rows = False
        self._sparse_active_rows_reason = "Native sparse factors are not selected."
        self._sparse_contact_triplets_reason = "Native sparse factors are not selected."
        # The admitted contact units' block records.
        self.placement.sparse_contact_records = None
        self.placement.sparse_max_row_support = None
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
                        self.factor.parallel_drives,
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
        if velocity_candidate and (self.forest.has_prescribed_response or np.any(self.factor.kinematic_dof_mask_host)):
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
            self.placement.dense_max_constraints,
            self.forest.max_world_dofs,
            int(max_support),
            velocity_limits=velocity_candidate,
            linked_rows=self.laws.friction_anchors_enabled,
            torsion=self.laws.contact_torsion_enabled,
        )
        # Patch friction keeps the packed lanes of small worlds, so size them from the
        # full sweep before the active-row scan claims whole warps.
        small_world_lanes = self._sparse_small_world_lanes(staging, budget)
        active_rows, active_rows_reason = self._admit_sparse_active_rows(
            velocity_candidate=velocity_candidate,
            small_world_lanes=small_world_lanes,
            fits=lambda: staging(active_rows=True).fits(budget),
        )
        shape = (self.forest.world_count, self.placement.dense_max_constraints, max_support)
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
                table_bytes += 4 * self.forest.world_count * self.placement.dense_max_constraints
                # Small batches with many support patterns can favor direct indices.
                if table_bytes < 4 * int(np.prod(shape)):
                    support_classes = classes
        # Unsupported mixed worlds retain explicit row indices and their existing kernels.
        use_support_table = support_classes is not None
        # Reuse the local solver's small-world packing without changing this
        # owner's row representation, sweep policy or velocity-limit phase.
        lanes = small_world_lanes if small_world_lanes and not active_rows else 32
        triplet_capacity = self.placement.dense_max_constraints // 3
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
        self.placement.sparse_mass_matrix_size = size
        self._sparse_active_rows = active_rows
        self._sparse_active_rows_reason = active_rows_reason
        self._sparse_contact_triplets = contact_triplets
        self._sparse_contact_triplets_reason = contact_triplets_reason
        self._sparse_solve_lanes = lanes
        self._sparse_local_factors = local_factors
        self.placement.sparse_max_row_support = int(max_support)
        self._sparse_velocity_limits = velocity_candidate
        # The sparse response kernel prepares the contact rows' metadata, CFM, bias and restitution.
        self.placement.sparse_prepare_contacts = True
        self.factor.allocate_sparse(plan, self.forest.n_arts_by_size[size])
        self.placement.sparse_body_dof_mask = wp.array(body_masks, dtype=wp.uint64, device=model.device)
        if use_support_table:
            self.placement.sparse_row_support = row_supports(
                support_classes, self.forest.world_count, self.placement.dense_max_constraints, model.device
            )
        if contact_triplets:
            # Static storage for graph capture: the builder writes the block records of each step's
            # contact units; linked sweeps add each anchor's ring members once per call.
            width = CONTACT_RECORD_FLOATS + (CONTACT_RING_FLOATS if self.laws.friction_anchors_enabled else 0)
            self.placement.arrays.allocate(
                SPARSE_CONTACT_RECORDS, world=self.forest.world_count, triplet=triplet_capacity, record=width
            )
        index_shape = (1, 1, 1) if use_support_table else shape
        self.placement.arrays.allocate(
            SPARSE_ROWS,
            index_world=index_shape[0],
            index_row=index_shape[1],
            index_support=index_shape[2],
            world=self.forest.world_count,
            row=self.placement.dense_max_constraints,
            support=max_support,
            world_dof=self.forest.max_world_dofs,
        )
        # The response kernel's free-body response output, where no contact records take
        # it, and its free-body factor: placeholders, since native sparse factors decline
        # free rigid bodies.
        self._sparse_row_free_response = self._arrays.placeholder(wp.float32, (1, 1, 1))
        self._sparse_free_factor_dummy = self._arrays.placeholder(wp.float32, (1, 6, 6))
        self._sparse_contact_lanes = 8
        # Contact torsion: each friction contact's spin row is the fourth of its unit.
        torsion = {"torsion": True} if self.laws.contact_torsion_enabled else {}
        self._sparse_contact_response_kernel = response_builds_rows.factor_contact_rows_kernel(
            size,
            lanes_per_contact=self._sparse_contact_lanes,
            prepare_rows=True,
            use_support_table=use_support_table,
            **({"contact_records": True} if self.placement.sparse_contact_records is not None else {}),
            **torsion,
        )
        self._pgs_solve_sparse_kernel = self._sweep_kernel(
            "sparse_pgs",
            sweep_routes_sparse.sparse_kernel,
            self.placement.dense_max_constraints,
            self.forest.max_world_dofs,
            max_support,
            **sweep,
            velocity_limits=self._sparse_velocity_limits,
            warm_start=self.options.pgs_warmstart,
            linked_rows=self.laws.friction_anchors_enabled,
            device_arch=model.device.arch,
            weighted=self.laws.set.weighted,
            **torsion,
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
        built = factory(*args, solve_policy=policy, material=bool(self.options.contact_compliance), **kwargs)
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

    def _weight_operands(self) -> list:
        """The row weights and the run-time switch that a sweep without them takes once the laws are weighted
        (:func:`.sweep.weight_arguments`)."""
        return [self.placement.row_w, 1] if self.laws.set.weighted else []

    def _material_operands(self) -> list:
        """The material contacts' compliance per row, which every sweep built with them takes after its own operands
        (:func:`.sweep.material_arguments`)."""
        return [self.placement.row_eps, self.placement.mf_row_eps] if self.options.contact_compliance else []

    def _sparse_small_world_lanes(self, staging, budget: int) -> int:
        """Return the lanes per world (8 or 16) that pack a small complete sparse sweep, or 0 for whole warps.

        Args:
            staging: The sweep's working set for its options (:func:`.sweep_routes_sparse.sparse_staging`).
            budget: Static shared memory available to one block [B].
        """
        if self.forest.max_world_dofs > 16 or self.placement.dense_max_constraints > 64:
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
        group_to_art = self.forest.group_to_art_host[size]
        expected_world = np.arange(len(group_to_art), dtype=articulation_world.dtype)
        return np.array_equal(articulation_world[group_to_art], expected_world)

    def _select_paired_response(self) -> tuple[int, int, np.ndarray] | None:
        """Select one complete robot/free-body response pair per world."""
        if (
            not self.placement.hinv_jt_writes_world
            or self.placement.hinv_jt_tiled_writes_group
            or 6 not in self.forest.size_groups
        ):
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

        secondary_group_arts = self.forest.group_to_art_host[6]
        secondary_group_index = {int(art): group for group, art in enumerate(secondary_group_arts)}
        for primary_size in sorted(self._paired_response_candidate_sizes, reverse=True):
            primary_arts = self.forest.group_to_art_host.get(primary_size, ())
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
        if not factor_dense_contract or not self.placement.fused_diagonal_joint_limits:
            return None

        response_dofs = self.forest.response_dof_count
        articulation_world = self.forest.articulation_world
        is_free_rigid = self.forest.is_free_rigid_host
        offsets = self.forest.articulation_world_dof_offset.numpy()
        group_indices = self.forest.art_group_idx.numpy()
        pairs = _world_diagonal_pairs(
            articulation_world, response_dofs, self.forest.world_count, self.placement.fused_diagonal_joint_limit_sizes
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

    def _allocate_common_buffers(self, model):
        arrays = self._arrays
        if model.joint_count:
            arrays.allocate(DOF_WORK, dof=model.joint_dof_count)
            # Stage-3 snapshots for external NaN dumpers, which read their values
            # by these names: None unless opted in.
            if self._debug_stage3_snapshots:
                arrays.allocate(STAGE3_SNAPSHOTS, dof=model.joint_dof_count)
            else:
                arrays.release(STAGE3_SNAPSHOTS)
        else:
            arrays.release(DOF_WORK)
            arrays.release(STAGE3_SNAPSHOTS)

        self.placement.fused_diagonal_limit_dof_mask = (
            wp.array(self.placement.fused_diagonal_limit_dof_mask_host, dtype=wp.int32, device=model.device)
            if self.placement.fused_diagonal_joint_limits
            else arrays.placeholder(wp.int32)
        )
        # The tree factor's mass-update mask, compact inverse mass, body inertia records and articulation origins.
        self.factor.allocate_common()

    def _allocate_buffers(self, model):
        self.placement.allocate_size_groups()
        if not model.articulation_count:
            return

        # The tree factor's L and R (compact and sparse sizes keep (1, 1, 1) placeholders of L).
        self.factor.allocate_size_groups((self._compact_diagonal_mass_size, self.placement.sparse_mass_matrix_size))
        self.placement.allocate_contact_rows(model)

    def _init_tiled_kernels(self, model):
        """Build the paired response and the sweeps.

        The factor, stage-3 and stage-4 kernels follow the step plan (:meth:`.placement.Placement.build_response_kernels`).
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
                    self.placement.dense_max_constraints,
                    self.forest.max_world_dofs,
                    primary_size,
                    secondary_size,
                    device_arch,
                    contact_triples=self._factor_coordinate_contact_triples,
                    independent_components=self._paired_components is not None,
                    contact_units=self._factor_coordinate_contact_units,
                    weighted=self.laws.set.weighted,
                    torsion=self.laws.contact_torsion_enabled,
                )
                if self._paired_factor_coordinates
                else None
            )
        else:
            self._paired_factor_solve_kernel = None

        # Only free rigid bodies produce matrix-free rows, which the CUDA step packs each step.
        self.placement.pack_mf_meta_kernel = (
            response_builds_rows.free_body_meta_kernel(self.placement.mf_max_constraints, device_arch)
            if self.forest.has_free_rigid_bodies and self.placement.mf_max_constraints > 0
            else None
        )

        self._pgs_solve_local_internal_kernels = {}
        self._pgs_solve_local_pair_kernels = {}
        self._pgs_solve_local_residual_kernels = {}
        self._pgs_solve_local_internal_warps_per_block = {}
        self._pgs_solve_local_internal_worlds_per_block = {}
        self._pgs_solve_local_pair_warps_per_block = {}
        local_factory = sweep_routes_local.local_owned_kernel
        if self.placement.row_packets is not None:
            local_factory = self.placement.row_packets.local_factory()
        # The contact owners solve contact torsion's spin rows with their pairs.
        local_torsion = {"torsion": True} if self.laws.contact_torsion_enabled else {}
        if self.placement.local_internal_fast_path:
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
                        self.placement.dense_max_constraints,
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
                        self.placement.dense_max_constraints,
                        self._local_solve_max_rows,
                        size,
                        device_arch,
                        paired_dof_count=6,
                        persistent_queue=True,
                        warps_per_block=warps_per_block,
                        **local_torsion,
                    )
            for size, count in self._local_residual_art_counts.items():
                if count > 0:
                    warps_per_block = self._local_residual_warps_per_block[size]
                    self._pgs_solve_local_residual_kernels[size] = self._sweep_kernel(
                        "local_residual",
                        local_factory,
                        self.placement.dense_max_constraints,
                        self._local_residual_max_rows,
                        size,
                        device_arch,
                        paired_dof_count=6,
                        persistent_queue=True,
                        warps_per_block=warps_per_block,
                        mf_max_constraints=self.placement.mf_max_constraints,
                        local_mf_max_constraints=self._local_residual_mf_max_rows,
                        **local_torsion,
                    )

        sparse_contact_schedule = (
            self.placement.sparse_diagonal_contact_triples or self.placement.sparse_diagonal_contact_units
        )
        self._mark_independent_sparse_contact_candidates_kernel = (
            _get_mark_independent_sparse_contact_candidates_kernel(
                self.placement.sparse_diagonal_response_size,
                device_arch,
                units=self.placement.sparse_diagonal_contact_units,
                torsion=self.laws.contact_torsion_enabled,
            )
            if sparse_contact_schedule
            else None
        )
        self._build_independent_sparse_contact_groups_kernel = (
            _get_build_independent_sparse_contact_groups_kernel(
                self.placement.dense_max_constraints,
                self.forest.max_world_dofs,
                device_arch,
                build_serial_contacts=self.placement.sparse_diagonal_speculative_contact_batches,
                units=self.placement.sparse_diagonal_contact_units,
                torsion=self.laws.contact_torsion_enabled,
            )
            if sparse_contact_schedule
            else None
        )
        self._pgs_solve_sparse_diagonal_kernel = (
            self._sweep_kernel(
                "sparse_diagonal",
                sweep_routes_diag.sparse_diagonal_kernel,
                self.placement.dense_max_constraints,
                self.forest.max_world_dofs,
                self.placement.sparse_diagonal_dense_size,
                device_arch,
                contact_triples=self.placement.sparse_diagonal_contact_triples,
                speculative_contact_batches=self.placement.sparse_diagonal_speculative_contact_batches,
                contact_units=self.placement.sparse_diagonal_contact_units,
                weighted=self.laws.set.weighted,
                torsion=self.laws.contact_torsion_enabled,
            )
            if self.placement.sparse_diagonal_contact_solve
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
            self._local_general_solver_blocks if self.placement.local_internal_fast_path else self.forest.world_count
        )
        # The Jacobi tiers ahead of the matrix-free sweep (SweepTier).
        self._mf_gs_tiers = []
        if model.device.is_cuda and int(model.articulation_count) > 0:
            # The grouped response solve (Placement._response_grouped), for the sizes StepPlan gives it from the data.
            self._grouped_response = True
            self.placement.grouped_dummy_diag = wp.zeros((1, 1), dtype=wp.float32, device=model.device)
        # The sweeps start from the impulses the rows carry: the warm start's, the closed loop's projection.
        self._seeded_sweep = bool(self.options.pgs_warmstart or self.factor.eliminates)
        if self.placement.mf_sweeps:
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
                        self.placement.dense_max_constraints,
                        self.placement.mf_max_constraints,
                        self.forest.max_world_dofs,
                    ),
                    arch_budget(device_arch),
                    block_count=self._mf_gs_block_count,
                    sm_count=int(model.device.sm_count),
                    force_stream_mf_impulses=force_stream_mf_impulses,
                )
            self._mfgs_storage = storage
            parallel_rows = self.placement.ink_rows
            # The colored mapping takes the worlds the unit builder's rule picks at every pack
            # (sweep_units.COLORED_RULE), among those whose free bodies and row capacity admit it: built, with a
            # storage slot each, for those worlds only, it runs beside the one-warp mapping. More of them than the
            # device holds colored blocks run in waves, which scales the rule (ColoredRule.at_scale).
            rule = sweep_units.COLORED_RULE
            capable = sweep_units.capable_worlds(
                self.forest.free_rigid_per_world, self.placement.mf_max_constraints, rule
            )
            if model.device.is_cuda and capable.size > 0:
                rule = rule.at_scale(capable.size, model.device.sm_count)
                capable = sweep_units.capable_worlds(
                    self.forest.free_rigid_per_world, self.placement.mf_max_constraints, rule
                )
            colored = capable.size > 0 and _route_table.capable("colored", self.laws.set) is None
            mf_gs_options = {
                "has_dense_velocity_limit_rows": self.options.enable_joint_velocity_limits,
                "shared_metadata": storage.shared_metadata,
                "resident_row_metadata": storage.resident_row_metadata,
                "stream_mf_impulses": storage.stream_mf_impulses,
                "skip_local_internal_worlds": self.placement.local_internal_fast_path,
                "local_internal_max_constraints": self.placement.dense_internal_max_rows,
                "contact_torsion": self.laws.contact_torsion_enabled,
                "factor_coordinates": self._paired_factor_coordinates,
                "skip_tier_rows": parallel_rows,
                "independent_components": self._paired_components is not None,
                "colored_worlds": colored,
                "max_row_dofs": self.placement.packed_support,
                "seeded": self._seeded_sweep,
                "wide_rows": self.placement.packed_dense == packed_rows.WIDE,
            }
            mf_gs_shape = (
                self.placement.dense_max_constraints,
                self.placement.mf_max_constraints,
                self.forest.max_world_dofs,
                device_arch,
            )
            self._pgs_solve_mf_gs_kernel = self._sweep_kernel(
                "mf_gs",
                sweep_routes.mf_gs_kernel,
                *mf_gs_shape,
                **mf_gs_options,
            )
            if colored:
                self._colored_units = sweep_units.ColoredUnits(
                    self.forest.world_count,
                    self.placement.mf_max_constraints,
                    self.forest.max_world_dofs,
                    model.device,
                    worlds=capable,
                    rule=rule,
                    torsion=self.laws.contact_torsion_enabled,
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
                if self.placement.keyed_rows is not None:
                    from .keyed_rows import get_parallel_factory  # noqa: PLC0415

                    parallel_factory = get_parallel_factory(parallel_factory, self.placement.keyed_rows.layout)
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
                        self.placement.dense_max_constraints,
                        self.placement.mf_max_constraints,
                        self.forest.max_world_dofs,
                        device_arch,
                        readout_budget=sweeps_,
                        rows=rows_,
                        sweeps=sweeps_,
                        min_rows=min_rows_,
                        inkernel_response=self.placement.ink_sizes,
                        linked_rows=self.laws.friction_anchors_enabled,
                        velocity_limit_pass=self._parallel_velocity_limit_pass(),
                        weighted=self.laws.set.weighted,
                        **({"torsion": True} if self.laws.contact_torsion_enabled else {}),
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
            and getattr(self.placement, "rigid_velocity_limit_slot", None) is not None
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
        # Components are admitted on CUDA, under the factor-coordinate sweep's plain policy.
        paired_overlap = bool(
            self._paired_components is not None
            and self._paired_factor_solve_kernel is not None
            and not getattr(self, "_mf_gs_tiers", ())
            and not _FPGS_SYNC_TIMINGS
        )
        self._paired_general_solve_stream = wp.Stream(model.device) if paired_overlap else None
        self._paired_general_ready = wp.Event(model.device) if paired_overlap else None
        self._paired_general_done = wp.Event(model.device) if paired_overlap else None

    def _launch_local_internal_solve(
        self,
        dense_rhs: wp.array,
        iterations: int,
        omega: float,
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
                    self.placement.local_solve_owner,
                    self.placement.constraint_count,
                    self.factor.L_by_size[primary_size],
                    self.factor.L_by_size[secondary_group_size],
                    self.placement.packed_Z,
                    self.placement.row_sides,
                    self.forest.articulation_response_dof_count,
                    self.forest.articulation_world_dof_offset,
                    int(self.placement.packed_dense),
                    dense_rhs,
                    self.placement.row_type,
                    self.placement.row_parent,
                    self.placement.row_mu,
                    self.placement.mf_constraint_count,
                    self.placement.mf_meta_packed,
                    self.placement.mf_impulses,
                    self.placement.mf_Z_a,
                    self.placement.mf_Z_b,
                    self.placement.mf_row_mu,
                    int(self.laws.set.weighted),
                    self.placement.row_w,
                    self.placement.mf_row_w,
                    iterations,
                    omega,
                ]
            )
            outputs = [self.placement.diag, self.placement.impulses, self.v_out]
            if self.placement.row_packets is not None:
                outputs.extend([self.placement.row_packets.prefix, self.placement.row_packets.packet])
            outputs.extend([*self._material_operands(), *self._solve_readout.operands(kernel)])
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
    ) -> None:
        if iterations <= 0 and not self._seeded_sweep:
            return
        route = self._plan.solve.route.name
        if route == "sparse_pgs":
            worlds_per_block = 64 // self._sparse_solve_lanes
            wp.launch_tiled(
                self._pgs_solve_sparse_kernel,
                dim=[(self.forest.world_count + worlds_per_block - 1) // worlds_per_block],
                inputs=[
                    self.forest.world_count,
                    self.placement.constraint_count,
                    dense_rhs,
                    self.placement.diag,
                    self.placement.impulses,
                    self.placement.sparse_row_dof,
                    self.placement.sparse_row_factor,
                    self.placement.row_incident,
                    self.placement.row_type,
                    self.placement.row_parent,
                    self.placement.row_mu,
                    self.forest.world_dof_indices,
                    self.v_hat,
                    self.placement.mf_constraint_count,
                    mf_meta,
                    self.placement.mf_impulses,
                    self.placement.mf_Z_a,
                    self.placement.mf_Z_b,
                    self.placement.mf_row_mu,
                    iterations,
                    omega,
                    # Contact units read the triplets' block records.
                    *(
                        ()
                        if self.placement.sparse_contact_records is None
                        else (self.placement.sparse_contact_records,)
                    ),
                ],
                outputs=[
                    self.placement.sparse_factor_velocity_delta,
                    self.placement.sparse_row_support,
                    *self._weight_operands(),
                    *self._material_operands(),
                    *self._solve_readout.operands(self._pgs_solve_sparse_kernel),
                ],
                block_dim=64,
                device=self.model.device,
            )
            self.factor.leave(
                self.v_out,
                size=self.placement.sparse_mass_matrix_size,
                indices=self.factor.sparse_mass_matrix_indices,
                Linv=self.factor.sparse_Linv,
                delta=self.placement.sparse_factor_velocity_delta,
                base=self.v_hat,
            )
            return
        if route == "sparse_diagonal":
            kernel = self._pgs_solve_sparse_diagonal_kernel
            if kernel is None:
                raise RuntimeError("Sparse diagonal GS kernel is unavailable for this solver shape")
            dense_size = self.placement.sparse_diagonal_dense_size
            wp.launch_tiled(
                kernel,
                dim=[self.forest.world_count],
                inputs=[
                    self.placement.constraint_count,
                    self.forest.world_dof_indices,
                    dense_rhs,
                    self.placement.row_incident,
                    self.placement.diag,
                    self.placement.impulses,
                    self.placement.row_type,
                    self.placement.row_parent,
                    self.placement.row_mu,
                    self.placement.dense_phase_bounds,
                    self.placement.sparse_contact_group_count,
                    self.placement.sparse_contact_group_heads,
                    self.placement.sparse_contact_serial_count,
                    self.placement.sparse_contact_serial_normals,
                    self.placement.sparse_diagonal_dense_offsets,
                    self.placement.sparse_diagonal_dense_groups,
                    self.factor.L_by_size[dense_size],
                    self.placement.J_by_size[dense_size],
                    self.placement.sparse_diagonal_row_dof,
                    self.placement.sparse_diagonal_row_z,
                    self.placement.fused_diagonal_limit_active,
                    self.placement.fused_diagonal_limit_lower_rhs,
                    self.placement.fused_diagonal_limit_upper_rhs,
                    self.factor.diagonal_inverse_mass,
                    self.options.pgs_cfm,
                    iterations,
                    omega,
                ],
                outputs=[
                    self.placement.fused_diagonal_limit_lower_lambda,
                    self.placement.fused_diagonal_limit_upper_lambda,
                    self.v_out,
                    *self._weight_operands(),
                    *self._material_operands(),
                    *self._solve_readout.operands(kernel),
                ],
                block_dim=32,
                device=self.model.device,
            )
            return
        if not self.forest.max_world_dofs:
            return  # No coordinate holds a row.
        mf_gs_kernel = self._pgs_solve_mf_gs_kernel
        if mf_gs_kernel is None:
            raise RuntimeError("Matrix-free GS kernel is unavailable for this solver shape")

        # Under a readout, count the worlds without rows of any family, which no sweep enters.
        self._solve_readout.count_skipped(self.placement.constraint_count, self.placement.mf_constraint_count)
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
        use_general_queue = bool(self.placement.local_internal_fast_path)
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
                            self.placement.constraint_count,
                            self.placement.dense_phase_bounds,
                            self.forest.world_dof_indices,
                            dense_rhs,
                            self.placement.row_incident,
                            self.placement.diag,
                            self.placement.impulses,
                            # The factor rows: the packed rows at their world coordinates.
                            self.placement.packed_Z,
                            self.placement.row_type,
                            self.placement.row_parent,
                            self.placement.row_mu,
                            self.placement.mf_constraint_count if components is None else components.selector,
                            self.factor.L_by_size[primary_size],
                            self.factor.L_by_size[secondary_size],
                            self._paired_factor_primary_groups_by_world,
                            self._paired_factor_secondary_groups_by_world,
                            self._paired_factor_primary_offsets_by_world,
                            self._paired_factor_secondary_offsets_by_world,
                            iterations,
                            omega,
                        ],
                        outputs=[
                            self.v_out,
                            *self._weight_operands(),
                            *self._material_operands(),
                            *self._solve_readout.operands(factor_kernel),
                        ],
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
                self.placement.constraint_count,
                self.placement.dense_phase_bounds,
                self.placement.local_solve_owner if components is None else components.selector,
                self.forest.world_dof_indices,
                dense_rhs,
                self.placement.row_incident,
                self.placement.diag,
                self.placement.row_w,
                self.placement.impulses,
                self.forest.articulation_response_dof_count,
                self.forest.articulation_world_dof_offset,
                int(self.placement.packed_dense),
                self.placement.packed_Z,
                self.placement.row_sides,
                self.placement.row_type,
                self.placement.row_parent,
                self.placement.row_mu,
                self.placement.mf_constraint_count,
                self.placement.mf_contact_rows_end,
                mf_meta,
                self.placement.mf_impulses,
                self.placement.mf_Z_a,
                self.placement.mf_Z_b,
                self.factor.view(),
                self.placement.mf_row_mu,
                self.placement.mf_row_w,
                iterations,
                omega,
                int(self.laws.set.weighted),
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
                        self.placement.constraint_count,
                        self.placement.mf_constraint_count,
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
                    inputs = inputs + self.placement.inkernel_response_inputs()
                    if sweep.kinetic_rows:
                        inputs = [*inputs, *self.placement.keyed_rows.parallel_arguments()]
                if sweep.colored_worlds:
                    inputs = [*inputs, self._colored_units.world_colored]
                wp.launch_tiled(
                    sweep.kernel,
                    dim=[dim],
                    inputs=inputs,
                    outputs=[self.v_out, *self._material_operands(), *self._solve_readout.operands(sweep.kernel)],
                    block_dim=sweep.block_dim,
                    device=self.model.device,
                )
            if colored_kernel is not None:
                # The worlds the unit builder's rule picked: a block each, a color's units at once.
                wp.launch_tiled(
                    colored_kernel,
                    dim=[general_block_count],
                    inputs=[*mf_gs_inputs, *self._colored_units.operands()],
                    outputs=[self.v_out, *self._material_operands(), *self._solve_readout.operands(colored_kernel)],
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
                            solve_single=False,
                            solve_residual=False,
                        )
                    local_done_events.append(pair_stream.record_event())
                with wp.ScopedStream(local_stream, sync_enter=False):
                    self._launch_local_internal_solve(
                        dense_rhs,
                        iterations,
                        omega,
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
                    )
        for local_done_event in local_done_events:
            wp.get_stream(self.model.device).wait_event(local_done_event)

    def _launch_matrix_free_position_solve(
        self,
        *,
        iterations: int,
    ) -> None:
        self._launch_matrix_free_gs_solve(
            dense_rhs=self.placement.rhs,
            mf_meta=self.placement.mf_meta_packed,
            iterations=iterations,
            omega=self.options.pgs_omega,
        )

    def _bind_contact_material(self, contacts: Contacts | None) -> None:
        """Bind the contacts' per-contact material for the writers (``contact_compliance``), or the placeholders.

        An attribute test, no device read: contacts without the three arrays are refused.
        """
        names = ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction")
        material = [self._contact_material_placeholder] * 3
        if self.options.contact_compliance and contacts is not None:
            material = [getattr(contacts, name, None) for name in names]
            if any(array is None for array in material):
                raise ValueError(f"contact_compliance requires the contacts' per-contact {', '.join(names)}")
        self.contact_stiffness, self.contact_damping, self.contact_friction_weight = material

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
        # The configuration's checks come first; then the states', before any of them is read.
        if state_in.requires_grad or state_out.requires_grad:
            raise ValueError("FeatherPGS has no gradients: step states built with requires_grad=False.")
        forest_in, forest_out, forest_control = self.cut.enter(
            state_in, state_out, control, self.factor.fk_id_cache_valid
        )
        self._step_forest(forest_in, forest_out, forest_control, contacts, dt, collide_done_event)
        self.cut.leave(state_in, state_out)
        return state_out

    def _step_forest(self, state_in, state_out, control, contacts, dt, collide_done_event) -> None:
        """One step of the forest (:mod:`.body_cut`)."""
        self._bind_contact_material(contacts)
        if contacts is not None:
            self.placement.follow_contacts(contacts)
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
        self.factor.track_step_dt(dt)

        model = self.model

        if control is None:
            control = model.control(clone_variables=False)
        # The step's state-derived dynamics are the tree factor's (no state carries them: there are no gradients).
        state_aug = self.factor
        self._last_debug_state_aug = state_aug

        if collide_done_event is not None and state_in.particle_count > 0:
            wp.get_stream(self.model.device).wait_event(collide_done_event)
            collide_done_event = None  # consumed

        self._eval_particle_forces(state_in, control, contacts)

        if not model.articulation_count:
            self.integrate_particles(model, state_in, state_out, dt)
            self._step += 1
            return

        # Stages 1-3: the tree factor (FK/ID, drives, mass refresh, factor), then the predictor v_hat.
        stage3_qd, inverse_dynamics_ready = self.factor.factor(
            state_in, control, state_out, dt, self._step, self.qd_work
        )
        self.factor.predict(state_in, control, stage3_qd, inverse_dynamics_ready, dt, self.v_hat)
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

        # The response Y = H^-1 J^T of each size that builds one, the diagonals (from J Y, or the packed rows'
        # |Z|^2) and the bias (the sweeps compute J v each iteration).
        with wp.ScopedTimer("S4_HinvJt_Diag_RHS", print=False, use_nvtx=self._nvtx, synchronize=False):
            for size in self.forest.size_groups:
                self.placement.response(size, self._plan.response[size].name)
            packed_diag = self.placement.diagonals(self.v_hat)
            self.placement.pack_rows(self.v_hat, write_diag=packed_diag)
            self.placement.add_cfm()
            self._stage4_compute_rhs_world(dt)

        # MF: whiten the rows (Z = L^-1 J^T), their mf_rhs, mf_eff_mass_inv and incident velocities
        if self.forest.has_free_rigid_bodies:
            with wp.ScopedTimer("S4_MF_Setup", print=False, use_nvtx=self._nvtx, synchronize=False):
                with self._sync_timed("s4_mf_setup"):
                    self.placement.whiten_free_bodies(self.v_hat, dt)

                # MF: compute world-relative DOF offsets for two-phase GS kernel
                with self._sync_timed("s4_mf_dof_offsets"):
                    self.placement.free_body_coordinates()

        # ══════════════════════════════════════════════════════════════
        # STAGE 5+6: PGS solve
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S5_PGS_Prep", print=False, use_nvtx=self._nvtx, synchronize=False):
            self.placement.prepare_impulses()
            if (
                self.options.pgs_warmstart
                and self.placement.ws_prev_dense_impulses is not None
                and contacts is not None
            ):
                match_index = contacts.rigid_contact_match_index
                pool = self.placement.contact_pool
                pool.launch(
                    gather_dense_warmstart,
                    pool.launch_rows(),
                    inputs=[
                        bind_contact_routes(self, contacts),
                        bind_dense_rows(self.placement),
                        match_index,
                        self.placement.ws_prev_dense_slot_sorted,
                        self.placement.ws_prev_dense_impulses,
                        self.placement.ws_prev_dense_row_type,
                        self.placement.ws_prev_dense_row_parent,
                        self.placement.constraint_count,
                        self.placement.ws_prev_contact_normal,
                        self.options.pgs_warmstart_decay,
                        dt / self._ws_prev_dt if self._ws_prev_dt > 0.0 else 1.0,
                        self.placement.dense_max_constraints,
                    ],
                    outputs=[self.placement.impulses],
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
                    self.placement.contact_world,
                    self.placement.contact_slot,
                    self.placement.contact_path,
                    self.placement.contact_slots_needed,
                    patch_routes(seeded),
                )

        with wp.ScopedTimer("S5_GatherJY", print=False, use_nvtx=self._nvtx, synchronize=False):
            if self.factor.eliminates:
                # Bilateral elimination: the other rows take the closed-loop response, their right-hand sides
                # and diagonals (the rows' closed-loop inertia).
                self.factor.eliminate(
                    Z=self.placement.packed_Z,
                    sides=self.placement.row_sides,
                    counts=self.placement.constraint_count,
                    owners=self.placement.local_solve_owner,
                    cfm=self.placement.row_cfm,
                    rhs=self.placement.rhs,
                    incident=self.placement.row_incident,
                    diag=self.placement.diag,
                    impulses=self.placement.impulses,
                )

            # The sweeps add their velocity change to the predictor; a world without rows keeps it.
            wp.copy(self.v_out, self.v_hat)

            # Pack MF metadata into int4 structs for coalesced 128-bit loads
            self.placement.pack_free_body_meta(self.placement.mf_rhs)
            if self._colored_units is not None:
                # The colored worlds' units and colors follow the rows the pack just published.
                self._colored_units.build(
                    self.placement.mf_constraint_count,
                    self.placement.mf_contact_rows_end,
                    self.placement.mf_meta_packed,
                    self.placement.mf_Z_a,
                    self.placement.mf_Z_b,
                    self.placement.mf_row_mu,
                    self.placement.mf_row_w,
                    int(self.laws.set.weighted),
                )

        with wp.ScopedTimer("S6_PGS_Solve", print=False, use_nvtx=self._nvtx, synchronize=False):
            self._launch_matrix_free_position_solve(
                iterations=self.options.pgs_iterations,
            )

        # ══════════════════════════════════════════════════════════════
        # STAGE 7: Update qdd + integrate
        # ══════════════════════════════════════════════════════════════
        self.factor.finish(state_in, state_out, dt, self._step)
        self.integrate_particles(model, state_in, state_out, dt)

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

        if self.options.pgs_warmstart and self.placement.ws_prev_dense_impulses is not None:
            with wp.ScopedTimer("S7_Dense_Warmstart_Carry", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                wp.copy(self.placement.ws_prev_dense_impulses, self.placement.impulses)
                wp.copy(self.placement.ws_prev_dense_row_type, self.placement.row_type)
                wp.copy(self.placement.ws_prev_dense_row_parent, self.placement.row_parent)
                if contacts is None:
                    self.placement.ws_prev_dense_row_type.fill_(-1)
                    self.placement.ws_prev_dense_row_parent.fill_(-1)

        if self.options.pgs_warmstart and self.forest.has_free_rigid_bodies:
            with wp.ScopedTimer("S7_MF_Warmstart_Carry", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                wp.copy(self.placement.ws_prev_mf_impulses, self.placement.mf_impulses)
                wp.copy(self.placement.ws_prev_mf_row_type, self.placement.mf_row_type)
                wp.copy(self.placement.ws_prev_mf_row_parent, self.placement.mf_row_parent)
                if contacts is None:
                    self.placement.ws_prev_mf_row_type.fill_(-1)
                    self.placement.ws_prev_mf_row_parent.fill_(-1)

        if self.options.pgs_warmstart:
            if contacts is not None:
                # One thread per row the contact stream ever used: the slot tables of every such row are rewritten.
                pool = self.placement.contact_pool
                used = pool.rows_used if contacts._rigid_contact_pool is pool else None
                pool.launch(
                    snapshot_contact_warmstart,
                    pool.launch_rows(),
                    inputs=[bind_contact_routes(self, contacts)],
                    outputs=[
                        self.placement.ws_prev_dense_slot_sorted,
                        self.placement.ws_prev_slot_sorted,
                        self.placement.ws_prev_contact_normal,
                    ],
                    extent=pool.ready_rows if used is None else used,
                    max_blocks=self._contact_capacity_blocks,
                )
            self._ws_prev_dt = float(dt)

        self.placement.record_watermarks(contacts)

        self._step += 1

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
        return self.placement.capacity_status(clear=clear)

    def check_constraint_capacity(self) -> None:
        """Raise for worlds invalidated by contact or constraint capacity exhaustion.

        Call at a host observation boundary, outside CUDA graph capture. The
        device-resident ``placement.constraint_overflow`` boolean array is always
        maintained, including when ``row_watermark=False``; GPU consumers can
        inspect it without host synchronization. Status remains set until
        :meth:`reset` clears the affected worlds. Increasing capacity requires
        constructing a solver and recapturing graphs before restarting the
        invalidated transition. Also raises while the ``patch_frames`` flag of
        :meth:`constraint_capacity_status` is set, until that method clears it.
        """
        self.placement.check_capacity()

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
        return self.placement.row_watermarks()

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

        dt = self.factor.last_step_dt
        inv_dt = 0.0 if dt is None or dt <= 0.0 else 1.0 / dt
        enable_friction_flag = 1 if self.options.enable_contact_friction else 0
        mf_impulses = getattr(self.placement, "mf_impulses", None)
        if mf_impulses is None:
            mf_impulses = self.placement.dummy_contact_impulses
        mf_constraint_count = getattr(self.placement, "mf_constraint_count", None)
        if mf_constraint_count is None:
            mf_constraint_count = self.placement.dummy_contact_count
        mf_row_type = getattr(self.placement, "mf_row_type", None)
        if mf_row_type is None:
            mf_row_type = self.placement.dummy_contact_row_type
        mf_row_parent = getattr(self.placement, "mf_row_parent", None)
        if mf_row_parent is None:
            mf_row_parent = self.placement.dummy_contact_row_parent

        pool = self.placement.contact_pool
        rows = pool.launch_rows()
        pool.launch(
            compute_contact_linear_force_from_impulses,
            rows,
            inputs=[
                bind_contact_routes(self, contacts),
                self.placement.impulses,
                mf_impulses,
                self.placement.constraint_count,
                mf_constraint_count,
                self.placement.row_type,
                self.placement.row_parent,
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
                else self.placement.dummy_is_free_rigid,
                contact_gap_gate=self.options.contact_gap_gate,
                same_articulation_gap_gate=self.options.same_articulation_contact_gap_gate,
                articulation_pair_gap_gate=self.options.articulation_pair_contact_gap_gate,
                friction_gap=self.options.contact_friction_gap_threshold,
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
                self.placement.contact_world,
                self.placement.contact_slot,
                self.placement.contact_path,
                self.placement.contact_slots_needed,
                self._patch_routes(),
            )

    def _finish_friction_patches(self, state_in, state_out, contacts, dt):
        """Publish persistent patch history from the integrated public state."""
        if not self.laws.friction_anchors_enabled:
            return
        if contacts is None:
            # A memset of today's rows: a capture of it is recaptured after growth.
            self.placement.contact_pool.note_unbound()
            self.laws.friction_patches.previous.valid.zero_()
            return
        # A step without a frame holds no patch rows; it stores the last frame again.
        routes = self._patch_routes() if self._patch_frame_built(contacts) else patch_routes(())
        self.laws.friction_patches.finish(
            state_in,
            state_out,
            self.model.body_com,
            self.placement.contact_world,
            self.placement.contact_slot,
            self.placement.contact_path,
            self.placement.contact_slots_needed,
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
                self.placement.row_parent,
                self.placement.row_mu,
                self.placement.impulses,
                self.options.pgs_warmstart_decay if self.options.pgs_warmstart else 0.0,
            )
        if self.forest.has_free_rigid_bodies:
            yield (
                1,
                self.placement.mf_row_parent,
                self.placement.mf_row_mu,
                self.placement.mf_impulses,
                self.options.pgs_warmstart_decay if self.options.pgs_warmstart else 0.0,
            )

    def _stage4_build_rows(self, state_in: State, state_aug: State, control: Control, contacts: Contacts, dt: float):
        model = self.model
        max_constraints = self.placement.dense_max_constraints
        # Every row producer writes the dense row records through one struct, and the joint row producers read
        # the joint table through another, both bound for this step.
        dense_rows = bind_dense_rows(self.placement)
        joints = bind_joint_table(self)
        mf_active = self.forest.has_free_rigid_bodies
        if self.forest.has_prescribed_response:
            self.placement.mf_target_velocity.zero_()

        self.placement.begin(self._plan.jacobian_clear_sizes)

        has_free_rigid_flag = 1 if mf_active else 0
        # Dummy arrays when MF is not active (kernel still needs valid pointers).
        # Preallocated in __init__: a per-step wp.zeros here would add an
        # allocation + memset every step and, under CUDA graph capture, leave
        # the graph holding a pointer to a freed temporary.
        is_free_rigid = (
            self.forest.is_free_rigid if self.forest.is_free_rigid is not None else self.placement.dummy_is_free_rigid
        )
        mf_slot_counter = self.placement.mf_slot_counter if mf_active else self.placement.dummy_mf_slot_counter
        dense_dropped_rows = self.placement.row_dropped_dense
        mf_dropped_rows = self.placement.row_dropped_mf
        mf_first_rejected_slot = (
            self.placement.mf_first_rejected_slot if mf_active else self.placement.dummy_mf_slot_counter
        )
        support = self.placement.row_support()
        if self.placement.compact_contact_boundary:
            from .compact_contact import clear_limit_prefix  # noqa: PLC0415

            wp.launch(
                clear_limit_prefix,
                dim=(self.forest.world_count, 12),
                inputs=[
                    self.placement.sparse_diagonal_dense_groups,
                    self.placement.J_by_size[6],
                    self.placement.sparse_diagonal_row_dof,
                ],
                device=model.device,
            )
        if self.placement.row_packets is not None:
            self.placement.row_packets.prepare_prefix(state_in, dt)
        if self.placement.keyed_rows is not None:
            self.placement.keyed_rows.begin_rows(state_in, state_aug, dt)

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
        if self.laws.mimic_count > 0 and self.placement.row_packets is None:
            wp.launch(
                allocate_mimic_slots,
                dim=self.laws.mimic_count,
                inputs=[field_group.bind(Mimics, laws=self.laws, model=model), max_constraints],
                outputs=[
                    self.placement.slot_counter,
                    self.placement.dense_first_rejected_slot,
                ],
                device=model.device,
            )
            for size in self.forest.size_groups:
                if size not in self.laws.mimic_sizes:
                    continue
                n_arts = self.forest.n_arts_by_size[size]
                if self._sparse_velocity_limits and size == self.placement.sparse_mass_matrix_size:
                    indices = self.factor.sparse_mass_matrix_indices
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
                            self.factor.sparse_Linv,
                            self.v_hat,
                        ],
                        outputs=[
                            dense_rows,
                            self.placement.sparse_row_dof,
                            self.placement.sparse_row_factor,
                            self.placement.row_incident,
                        ],
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
                    outputs=[self.placement.jacobian_rows(size), dense_rows],
                    device=model.device,
                )

        # The links' rows: static per model, their sides the trees of the link's two pieces.
        if self.laws.link_count > 0:
            links = field_group.bind(Links, laws=self.laws)
            wp.launch(
                allocate_link_slots,
                dim=self.laws.link_count,
                inputs=[links, max_constraints],
                outputs=[self.placement.slot_counter, self.placement.dense_first_rejected_slot],
                device=model.device,
            )
            wp.launch(
                write_link_rows,
                dim=self.laws.link_count,
                inputs=[
                    joints,
                    links,
                    state_in.body_q,
                    model.body_com,
                    self.forest.body_to_joint,
                    self.forest.body_to_articulation,
                    model.joint_ancestor,
                    state_aug.joint_S_s,
                    self.factor.articulation_origin,
                    self.options.pgs_beta,
                    self.options.pgs_cfm,
                    support,
                ],
                outputs=[self.placement.packed_Z, dense_rows],
                device=model.device,
            )

        # Connect (loop-closure) equality rows — static count per model.
        if self.laws.connect_count > 0:
            wp.launch(
                allocate_connect_slots,
                dim=self.laws.connect_count,
                inputs=[field_group.bind(Connects, laws=self.laws), max_constraints],
                outputs=[
                    self.placement.slot_counter,
                    self.placement.dense_first_rejected_slot,
                ],
                device=model.device,
            )
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
                        self.factor.articulation_origin,
                        self.options.pgs_beta,
                        self.options.pgs_cfm,
                        support,
                    ],
                    outputs=[self.placement.jacobian_rows(size), dense_rows],
                    device=model.device,
                )

        if self.placement.fused_diagonal_joint_limits:
            wp.launch(
                prepare_fused_diagonal_joint_limits,
                dim=self.forest.world_count * self.forest.max_world_dofs,
                inputs=[
                    joints,
                    self.forest.world_dof_indices,
                    self.forest.max_world_dofs,
                    self.placement.fused_diagonal_limit_dof_mask,
                    self.laws.joint_limit_q_index,
                    state_in.joint_q,
                    self.options.joint_limit_activation_gap,
                    self.options.pgs_beta,
                    dt,
                    self.v_hat,
                ],
                outputs=[
                    self.placement.fused_diagonal_limit_active,
                    self.placement.fused_diagonal_limit_lower_rhs,
                    self.placement.fused_diagonal_limit_upper_rhs,
                ],
                device=model.device,
            )

        # Allocate and populate joint-limit rows per response-size group.
        if (
            self.options.enable_joint_limits
            and self.laws.joint_limit_sizes
            and self.placement.row_packets is None
            and self.placement.keyed_rows is None
        ):
            # The native limit rows (prefix_rows) read the law and write the records through one struct.
            law = field_group.bind(
                prefix_rows.PrefixLaw,
                solver=self,
                placement=self.placement,
                factor=self.factor,
                laws=self.laws,
                options=self.options,
                model=model,
                state_in=state_in,
                dt=dt,
            )
            for size in self.forest.size_groups:
                if size not in self.laws.joint_limit_sizes or size in self.placement.fused_diagonal_joint_limit_sizes:
                    continue
                n_arts = self.forest.n_arts_by_size[size]
                if size == self.placement.sparse_mass_matrix_size:
                    indices = self.factor.sparse_mass_matrix_indices
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
                            self.factor.sparse_Linv,
                            self.v_hat,
                        ],
                        outputs=[
                            self.placement.sparse_row_dof,
                            self.placement.sparse_row_factor,
                            self.placement.row_incident,
                            self.placement.sparse_row_support,
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
                        outputs=[self.placement.jacobian_rows(size)],
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
                            self.placement.slot_counter,
                            self.placement.dense_first_rejected_slot,
                            self.placement.jacobian_rows(size),
                            dense_rows,
                        ],
                        device=model.device,
                    )

        # Unconditional: the first boundary must include position-limit
        # rows and remain valid when that family is disabled (when it is just
        # the current watermark at the start of the velocity-limit segment).
        if self.placement.row_packets is None and self.placement.keyed_rows is None:
            wp.launch(
                snapshot_dense_phase_bound,
                dim=self.forest.world_count,
                inputs=[self.placement.slot_counter, 0],
                outputs=[self.placement.dense_phase_bounds],
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
            and self.placement.keyed_rows is None
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
                    self.placement.slot_counter,
                    self.placement.dense_first_rejected_slot,
                ],
                device=model.device,
            )

        # Unconditional velocity-limit boundary snapshot; dense contact rows start here.
        if self.placement.row_packets is None and self.placement.keyed_rows is None:
            wp.launch(
                snapshot_dense_phase_bound,
                dim=self.forest.world_count,
                inputs=[self.placement.slot_counter, 1],
                outputs=[self.placement.dense_phase_bounds],
                device=model.device,
            )

        if self.placement.sparse_diagonal_contact_solve:
            wp.launch(
                clear_sparse_diagonal_response_prefix,
                dim=self.forest.world_count,
                inputs=[self.placement.dense_phase_bounds],
                outputs=[self.placement.sparse_diagonal_row_dof],
                device=model.device,
            )

        if (
            self.options.enable_joint_velocity_limits
            and self.laws.velocity_limit_slot is not None
            and self.placement.keyed_rows is None
        ):
            for size in self.forest.size_groups:
                n_arts = self.forest.n_arts_by_size[size]
                if self._sparse_velocity_limits and size == self.placement.sparse_mass_matrix_size:
                    indices = self.factor.sparse_mass_matrix_indices
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
                            self.factor.sparse_Linv,
                            self.v_hat,
                        ],
                        outputs=[
                            dense_rows,
                            self.placement.sparse_row_dof,
                            self.placement.sparse_row_factor,
                            self.placement.row_incident,
                            self.placement.sparse_row_support,
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
                    outputs=[self.placement.jacobian_rows(size), dense_rows],
                    device=model.device,
                )

        if (
            contacts is not None
            and getattr(contacts, "rigid_contact_count", None) is not None
            and contacts._rigid_contact_rows > 0
        ):
            enable_friction_flag = 1 if self.options.enable_contact_friction else 0
            pool = self.placement.contact_pool
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
                    self.placement.mf_max_constraints,
                    enable_friction_flag,
                    self.options.contact_friction_gap_threshold,
                    1,
                    self.laws.friction_patches.view,
                    self._resolved_worlds,
                ],
                outputs=[
                    self.placement.slot_counter,
                    mf_slot_counter,
                    dense_dropped_rows,
                    mf_dropped_rows,
                    self.placement.dense_first_rejected_slot,
                    mf_first_rejected_slot,
                ],
                extent=count,
                clamp=True,
            )
            if contacts.rigid_contacts_pair_sorted:
                self.placement.place_sorted_contact_rows(
                    contacts, contact_build_threads, mf_slot_counter, mf_first_rejected_slot
                )

            # The class's case names the producer that writes every contact row (contact_row_plan):
            # get_path_selection() records the same plan (routes.contact_rows).
            plan = contact_row_plan(self)
            if plan.producer == "local_packets":
                self.placement.row_packets.produce_contacts(contact_input)
            elif plan.producer == "keyed_rows":
                self.placement.keyed_rows.produce_contacts(contact_input)
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
                        indices = self.factor.sparse_mass_matrix_indices
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
                                self.placement.sparse_body_dof_mask,
                                law,
                                indices.permutation,
                                indices.row_offsets,
                                self.factor.sparse_Linv,
                                self._sparse_free_factor_dummy,
                            ],
                            outputs=[
                                self.placement.sparse_row_dof,
                                self.placement.sparse_row_factor,
                                # The record builder writes the triplet records in the free-response
                                # slot, which the support table's routes leave unused.
                                self._sparse_row_free_response
                                if self.placement.sparse_contact_records is None
                                else self.placement.sparse_contact_records,
                                self.placement.row_incident,
                                self.placement.sparse_row_support,
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
                        outputs=[self.placement.jacobian_rows(size)],
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
                            self.placement.sparse_diagonal_response_size,
                            self.forest.articulation_world_dof_offset,
                            self.forest.body_single_response_dof,
                            self.factor.diagonal_inverse_mass,
                            law,
                        ],
                        outputs=[self.placement.sparse_diagonal_row_dof, self.placement.sparse_diagonal_row_z],
                        extent=count,
                        clamp=True,
                    )
                    if self.placement.sparse_diagonal_contact_triples or self.placement.sparse_diagonal_contact_units:
                        marker = self._mark_independent_sparse_contact_candidates_kernel
                        if marker is None:
                            raise RuntimeError("Independent sparse-contact marker kernel is unavailable")
                        pool.launch(
                            marker,
                            contact_build_threads,
                            inputs=[contact_input, contact_build_threads],
                            outputs=[self.placement.sparse_diagonal_row_dof],
                            extent=count,
                            clamp=True,
                        )
            else:
                # The masked builder: a prelude writes the row records and per-contact flags, then one
                # thread per (contact, DOF) writes the Jacobian; one launch pair serves the combined sizes.
                flags = self.placement.rows_prelude_flags
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
                        J_a = self.placement.jacobian_rows(size_a)
                        J_b = self.placement.jacobian_rows(size_b) if size_b > 0 else J_a
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
                        outputs=[self.placement.jacobian_rows(size)],
                        extent=count,
                        clamp=True,
                    )

            # Build MF contact rows
            if mf_active:
                pool.launch(
                    build_mf_contact_rows,
                    contact_build_threads,
                    inputs=[
                        bind_mf_rows(self.placement),
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
                    inputs=[
                        self.placement.mf_slot_counter,
                        self.placement.mf_max_constraints,
                        1,
                        self.placement.mf_first_rejected_slot,
                    ],
                    outputs=[self.placement.mf_constraint_count, self.placement.constraint_capacity_status],
                    device=model.device,
                )

                # ── MF warm-start seed ────────────────────────────────────
                # contact_slot + mf_row_type are now populated for THIS step.
                # mf_impulses was already memset at the top of this method
                # (covers vel-limit rows, padding tail, and any contact that
                # returns early in the gather); here we gather matched contacts'
                # previous impulses into their current slots by identity. The
                # carry itself lives in ws_prev_mf_impulses.
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
                            bind_mf_rows(self.placement),
                            match_index,
                            self.placement.ws_prev_slot_sorted,
                            self.placement.ws_prev_mf_impulses,
                            self.placement.ws_prev_mf_row_type,
                            self.placement.ws_prev_mf_row_parent,
                            self.placement.mf_constraint_count,
                            self.placement.ws_prev_contact_normal,
                            self.options.pgs_warmstart_decay,
                            # Carried support impulses are proportional to dt, so
                            # rescale by the exact step-size ratio (1 at fixed dt).
                            dt / self._ws_prev_dt if self._ws_prev_dt > 0.0 else 1.0,
                            self.placement.mf_max_constraints,
                        ],
                        outputs=[self.placement.mf_impulses],
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
            wp.copy(self.placement.mf_contact_rows_end, self.placement.mf_slot_counter)
        if mf_active and self.placement.rigid_velocity_limit_slot is not None:
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
                    self.placement.mf_max_constraints,
                    self._resolved_worlds,
                ],
                outputs=[
                    self.placement.rigid_velocity_limit_slot,
                    self.placement.mf_slot_counter,
                    self.placement.mf_first_rejected_slot,
                ],
                device=model.device,
            )
            wp.launch(
                populate_rigid_velocity_limit_rows,
                dim=self.forest.free_rigid_body_count,
                inputs=[bind_free_bodies(self), bind_mf_rows(self.placement), self.placement.rigid_velocity_limit_slot],
                device=model.device,
            )
            wp.launch(
                finalize_constraint_counts_with_status,
                dim=self.forest.world_count,
                inputs=[
                    self.placement.mf_slot_counter,
                    self.placement.mf_max_constraints,
                    1,
                    self.placement.mf_first_rejected_slot,
                ],
                outputs=[self.placement.mf_constraint_count, self.placement.constraint_capacity_status],
                device=model.device,
            )

        if self.placement.row_packets is not None:
            self.placement.row_packets.finish_rows()
        wp.launch(
            finalize_constraint_counts_with_status,
            dim=self.forest.world_count,
            inputs=[self.placement.slot_counter, max_constraints, 0, self.placement.dense_first_rejected_slot],
            outputs=[self.placement.constraint_count, self.placement.constraint_capacity_status],
            device=model.device,
        )
        if self.placement.sparse_prepare_contacts:
            wp.launch(
                finalize_sparse_joint_limit_rows,
                dim=self.forest.world_count,
                inputs=[self.placement.dense_phase_bounds, self.placement.constraint_count, dense_rows, dt],
                device=model.device,
            )
        if self.placement.keyed_rows is not None:
            self.placement.keyed_rows.finish_rows()
        # The patch link runs after every row producer: the keyed rows' general worlds take their records last.
        self._link_friction_patches(contacts)
        self.placement.latch_status(contacts)
        if self.placement.sparse_diagonal_contact_triples or self.placement.sparse_diagonal_contact_units:
            schedule = self._build_independent_sparse_contact_groups_kernel
            if schedule is None:
                raise RuntimeError("Independent sparse-contact schedule kernel is unavailable")
            # Unit heads come from the row types and the tangent rows' parents.
            unit_rows = (
                [self.placement.row_type, self.placement.row_parent]
                if self.placement.sparse_diagonal_contact_units
                else []
            )
            wp.launch(
                schedule,
                dim=self.forest.world_count * 32,
                inputs=[
                    self.placement.constraint_count,
                    self.placement.dense_phase_bounds,
                    *unit_rows,
                ],
                outputs=[
                    self.placement.sparse_diagonal_row_dof,
                    self.placement.sparse_contact_group_count,
                    self.placement.sparse_contact_group_heads,
                    self.placement.sparse_contact_serial_count,
                    self.placement.sparse_contact_serial_normals,
                ],
                device=model.device,
            )
        if self.placement.local_internal_fast_path and self.placement.row_packets is None:
            self._local_general_world_count.zero_()
            wp.launch(
                classify_local_solve_worlds,
                dim=self.forest.world_count,
                inputs=[
                    bind_mf_rows(self.placement),
                    field_group.bind(LocalOwners, solver=self, placement=self.placement),
                    self.placement.constraint_count,
                    self.placement.dense_phase_bounds,
                    self.placement.mf_constraint_count,
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
                            self.placement.local_solve_owner,
                            owner,
                        ],
                        outputs=[active_count, active_candidates[size], active_secondaries[size]],
                        device=model.device,
                    )

        self.placement.warn_overflow()

    def _plan_tier_rows(self) -> None:
        """Choose the Jacobi tiers from the class (case K4): their row bound and their in-kernel response.

        The tiers serve the dense rows of a tree beside a free body where no earlier case claims the class
        (:meth:`.route_table.RouteTable.taken`). They form each world's response in the kernel, which bounds the class
        (:meth:`.placement.Placement.inkernel_response_meta`: one articulation of each of one or two sizes per world, at most
        ``_INKERNEL_MAX_DOF`` coordinates each), under the contact laws they implement
        (:meth:`_admit_parallel_contact_paths`). Their row bound is the dense row capacity, at most
        :data:`.sweep_routes_tiers.TIER_MAX_ROWS`, in whole quads; a test pin (``tiers``) sets it on any class whose
        response they form. ``ink_rows`` is 0 without tiers and the route table's K4 reason says why or why not
        (``routes.row.reasons``). The local packets and the keyed rows read this before the tiers are built.
        """
        pinned = self._route_pins.get("tiers")
        self.placement.ink_rows, self.placement.ink_sizes = 0, None
        reason = (
            self.routes.taken("K4")
            or capable("tiers", self.laws.set)
            or first_failed(
                (
                    (self.placement.mf_sweeps, "No matrix-free sweep runs whose dense rows the tiers could take."),
                    (pinned is None or bool(pinned), "A test pin builds no tier (tiers=0)."),
                    (
                        not self.factor.eliminates,
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
            sizes, reason = self.placement.inkernel_response_meta()
            rows = min(int(self.placement.dense_max_constraints), int(pinned or sweep_routes_tiers.TIER_MAX_ROWS))
            rows -= rows % 4
            if self.routes.claim("K4", sizes is not None and rows > 0):
                self.placement.ink_sizes, self.placement.ink_rows = sizes, rows
            elif sizes is not None:
                reason = f"The dense row capacity {self.placement.dense_max_constraints} holds no tier of four rows."
        self.routes.reasons["K4"] = reason

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
        there too, and weighted contact rows (the relative regularization, material
        contacts) by their weights, the serial sweeps' row law. Contact torsion's spin
        rows project with their pairs onto their cone (keyed rows write none: the tiers
        then take the general producer's rows).
        """
        laws = self.laws.set
        if laws.contact == "point":
            return True, "Point friction: parallel tiers and keyed rows serve it."
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

    def _stage4_compute_rhs_world(self, dt: float):
        """Each dense row's rhs: its bias, or a bouncing contact's rebound (``compute_world_contact_bias``)."""
        if (
            self.placement.sparse_prepare_contacts
            or self.placement.row_packets is not None
            or self.placement.keyed_rows is not None
        ):
            return  # These rows write their own right-hand sides.
        wp.launch(
            compute_world_contact_bias,
            dim=self.forest.world_count * self.placement.dense_max_constraints,
            inputs=[
                bind_dense_rows(self.placement),
                self.placement.row_incident,
                self.placement.constraint_count,
                self.placement.dense_max_constraints,
                dt,
                self.options.contact_speculative_scale,
                self.laws.contact_w,
                self.laws.effective_restitution_velocity_threshold,
                int(self.laws.set.weighted),
                int(self.options.contact_compliance),
            ],
            device=self.model.device,
        )

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
def _get_mark_independent_sparse_contact_candidates_kernel(
    sparse_response_dofs: int,
    device_arch: str,
    *,
    units: bool = False,
    torsion: bool = False,
) -> "wp.Kernel":
    """Mark contact triples whose response is confined to one sparse coordinate.

    ``units`` marks patch contact units instead: a normal row, followed by its two
    tangent rows only when the contact owns a friction anchor. Under contact
    ``torsion`` a friction contact's spin row follows its tangents (four rows).
    """
    del device_arch
    full = 3 + int(torsion)
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
            if data.path[contact] != 0 or slot < 0 or (rows != 1 and rows != full):
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

    suffix = "_torsion" if torsion else ""
    if units:
        name = f"mark_independent_sparse_contact_candidates_{sparse_size}_units{suffix}"
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
            if data.path[contact] != 0 or slot < 0 or data.slots_needed[contact] != full:
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

    name = f"mark_independent_sparse_contact_candidates_{sparse_size}{suffix}"
    return python_kernel(mark_independent_sparse_contact_candidates_template, name)


@cache
def _get_build_independent_sparse_contact_groups_kernel(
    max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    build_serial_contacts: bool = False,
    units: bool = False,
    torsion: bool = False,
) -> "wp.Kernel":
    """Build exact independent and serial contact schedules for sparse diagonal response.

    ``units`` schedules patch contact units instead of contact triples (see
    :func:`_build_independent_sparse_contact_units_kernel`). Each world's warp
    links the scalar triples and lists the remaining ones in parallel; the
    schedules equal those of serial walks over the segment's whole triples. Under
    contact ``torsion`` a contact's spin row follows its tangents: four rows apart.
    """
    del device_arch
    M = int(max_constraints)
    D = int(max_world_dofs)
    if units:
        return _build_independent_sparse_contact_units_kernel(M, D, build_serial_contacts, torsion)
    # The serial list's capacity, a triple's (four-row contacts need less).
    S = (M + 2) // 3
    R = 4 if torsion else 3
    serial_schedule = (
        f"""
    // Compact the remaining contacts in original row order. The solve may
    // inspect up to four normals concurrently, but any active batch is still
    // applied through this serial list in exact GS order.
{_list_serial_contacts_source(count="triple_count", stride=R, head="true", serial_capacity=S)}
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
    const int triple_count = contact_end > contact_start ? (contact_end - contact_start) / {R} : 0;
    for (int coordinate = lane; coordinate < {D}; coordinate += 32)
        group_heads.data[head_base + coordinate] = -1;
    SYNC();

    // Reserve every diagonal coordinate used by a coupled contact. A scalar
    // row sharing one must remain in the serial chain to preserve GS order.
    for (int normal = contact_start + lane * {R}; normal + {R - 1} < contact_end; normal += {32 * R}) {{
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
{_link_scalar_contacts_source(count="triple_count", stride=R, head="true", dofs=D)}
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
    if torsion:
        name += "_torsion"
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


def _build_independent_sparse_contact_units_kernel(
    M: int, D: int, build_serial_contacts: bool, torsion: bool = False
) -> "wp.Kernel":
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
    runs the walks for any other row layout. Under contact ``torsion`` a unit's
    spin row follows its tangents and names its head too.
    """
    friction, spin = int(PGS_CONSTRAINT_TYPE_FRICTION), int(PGS_CONSTRAINT_TYPE_TORSION)
    R = 4 if torsion else 3

    def member(row: str) -> str:
        """Whether row ``row`` belongs to the unit of an earlier head (a tangent, or a spin row)."""
        kind = f"world_row_type.data[row_base + {row}]"
        return f"({kind} == {friction} || {kind} == {spin})" if torsion else f"{kind} == {friction}"

    unit_head = f"!{member('normal')}" if torsion else f"world_row_type.data[row_base + normal] != {friction}"
    # The serial walks that define the schedules, which lane 0 runs for any other row layout.
    serial_walks = f"""
    // Prepending candidates in reverse row order makes each negative link
    // reproduce the original ascending GS order. Each step lands on the last
    // row of the previous unit, which is its head or one of its tangent rows.
    if (lane == 0) {{
        int row = contact_end - 1;
        while (row >= contact_start) {{
            int normal = row;
            if ({member("row")}) {{
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
            const bool owns_tangents = normal + {R - 1} < contact_end
                && world_row_type.data[row_base + normal + 1] == {friction}
                && world_row_parent.data[row_base + normal + 1] == normal;
            normal += owns_tangents ? {R} : 1;
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
    # A unit's rows: its head, then nothing or its tangents (and spin row) naming it, then the next head.
    layout = "if (tangent_1 && !(named_1 && tangent_2 && named_2 && !tangent_3)) rows_form_units = false;"
    if torsion:
        layout = f"""const bool spin_3 = normal + 3 < contact_end && world_row_type.data[row_base + normal + 3] == {spin};
        const bool named_3 = normal + 3 < contact_end && world_row_parent.data[row_base + normal + 3] == normal;
        const bool member_4 = normal + 4 < contact_end && {member("normal + 4")};
        if (tangent_1 && !(named_1 && tangent_2 && named_2 && spin_3 && named_3 && !member_4)) rows_form_units = false;"""

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
        || {f"!{member('contact_start')}" if torsion else f"world_row_type.data[row_base + contact_start] != {friction}"};
    for (int normal = contact_start + lane; normal < contact_end; normal += 32) {{
        const int sparse_row = sparse_world_base + normal * 2;
        const int coord_0 = sparse_row_dof.data[sparse_row];
        const int coord_1 = sparse_row_dof.data[sparse_row + 1];
        const bool unit_head = {unit_head};
        const bool tangent_1 = normal + 1 < contact_end && world_row_type.data[row_base + normal + 1] == {friction};
        const bool tangent_2 = normal + 2 < contact_end && world_row_type.data[row_base + normal + 2] == {friction};
        const bool tangent_3 = normal + 3 < contact_end && world_row_type.data[row_base + normal + 3] == {friction};
        const bool named_1 = normal + 1 < contact_end && world_row_parent.data[row_base + normal + 1] == normal;
        const bool named_2 = normal + 2 < contact_end && world_row_parent.data[row_base + normal + 2] == normal;
        if (!unit_head) continue;
        {layout}
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
    name += "_units" + ("_torsion" if torsion else "")
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

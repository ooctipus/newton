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
import functools
import os
import sys
import time
from contextlib import contextmanager
from typing import ClassVar, Literal

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags
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
    prefix_rows,
    response_builds_rows,
)
from . import row_metadata as _row_metadata
from .audit import Audit
from .body_cut import BodyCut
from .configuration import check_options, kernel_choices
from .contact_input import bind as bind_contact_input
from .contact_input import bind_routes as bind_contact_routes
from .contact_row_plan import SHARED_METADATA_PRODUCERS, contact_row_plan
from .dense_rows import bind as bind_dense_rows
from .dense_rows import bind_mf as bind_mf_rows
from .field_group import SolverArrays
from .forest_plan import ForestPlan, joint_forest
from .friction_patches import (
    patch_row_law,
)
from .joint_table import bind as bind_joint_table
from .joint_table import bind_free_bodies
from .kernels import (
    allocate_joint_velocity_limit_slots,
    allocate_link_slots,
    allocate_mimic_slots,
    allocate_rigid_velocity_limit_slots,
    allocate_world_contact_slots,
    build_joint_limit_rows_for_size,
    build_mf_contact_rows,
    clear_sparse_diagonal_response_prefix,
    compute_world_contact_bias,
    finalize_constraint_counts_with_status,
    gather_mf_warmstart,
    populate_joint_velocity_limit_J_for_size,
    populate_mimic_J_for_size,
    populate_rigid_velocity_limit_rows,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_fused_diagonal_joint_limits,
    prepare_world_contact_rows,
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
from .solver_fields import (
    DOF_WORK,
    STAGE3_SNAPSHOTS,
    Links,
    Mimics,
)
from .sparse_contact import (
    finalize_sparse_joint_limit_rows,
)
from .step_plan import StepPlan
from .sweep_plan import _CONTACT_JACOBIAN_MAX_DOF, Sweep
from .tree_factor import TreeFactor

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
        pgs_audit: bool = False,
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
            enable_bilateral_preelimination (bool, optional): Eliminate mimic rows exactly, as a loop joint's rows
                always are (:func:`.response_builds_rows.elimination`): every other row (contacts especially) takes
                the closed-loop response, ``Z' = (I - Q Q^T) Z``, and the predictor velocity is projected onto the
                closures once per step. Blocks are sized from the model; a row that depends on the others leaves
                its block. Defaults to False.
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
                mimic and loop-closing rows. Defaults to 0.2.
            pgs_cfm (float, optional): Compliance/regularization added to the Delassus diagonal. Defaults to 1.0e-6.
            pgs_omega (float, optional): Successive over-relaxation factor for the PGS sweep. Defaults to 1.0.
            pgs_audit (bool, optional): Audit each step after its solve: per world, the energy gap ``G`` [J] the solve
                failed to account for, the dual violation ``D`` [m/s] and the rows' injected work ``W`` [J]
                (:mod:`.audit`), read with :meth:`read_audit`. It reads what the step holds, steers no route and
                changes no computed value. A world whose rows exist only inside its sweep reads NaN
                (``routes.diagnostics.audit`` says which). Defaults to ``False``.
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
        # The row laws' tables: the contact law, speed limits, the joint-limit tables, the mimic and link plans,
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
        from . import held_tree  # noqa: PLC0415
        from .scalar_branches import plan_scalar_branches  # noqa: PLC0415

        # Axis 3 of the route table, the placement case: the owners below, in precedence order (route_table.CASES),
        # each test their own fit once no earlier case holds the class, and the first that fits claims it.
        self.routes = RouteTable(self._route_pins)
        self._execution_plan = _FeatherPGSExecutionPlan.build(
            self.forest.size_groups,
            max_constraints=self.placement.dense_max_constraints,
            max_shared_memory=int(getattr(model.device, "max_shared_memory_per_block", 0)),
            hinv_jt_kernel=self.hinv_jt_kernel,
            small_dof_threshold=self.small_dof_threshold,
            tile_threads=self.tile_threads,
            diagonal_mass_sizes=self.forest.diagonal_mass_sizes,
        )
        # The sweep: its owners claim the class below, in the route table's order.
        self.sweep = Sweep(
            model,
            options,
            self.forest,
            self.laws,
            self.placement,
            self.factor,
            self.routes,
            self._execution_plan,
            self._route_pins,
            timed=self._sync_timed,
            nvtx=nvtx,
        )

        def diagonal_decline(size: int) -> str | None:
            """Its compact diagonal is the held tree's representation: a candidate only where the held tree takes
            the dense groups with it, the diagonal group as scalar branches or a held family."""
            scalar, _ = plan_scalar_branches(self, size=size)
            return held_tree.compact_decline(self, size, scalar=scalar is not None)

        self.sweep.admit_diagonal(diagonal_decline)
        self.placement.dense_internal_max_rows = self.placement.internal_rows_per_world(model)
        self.placement.hinv_jt_computes_diag = not self.factor.eliminates
        self.placement.hinv_jt_diag_sizes = frozenset(
            size
            for size in self.forest.size_groups
            if self.placement.hinv_jt_computes_diag
            and (self._execution_plan.use_diagonal_mass(size) or self._execution_plan.use_tiled_hinv_jt(size))
        )
        self.sweep.admit_local()
        self._contact_material_placeholder = self._arrays.placeholder(wp.float32, (1,))
        self._bind_contact_material(None)
        self.placement.jy_world_aliased = self._detect_jy_world_identity()
        self._allocate_common_buffers(model)
        self._compact_diagonal_mass_size = self.placement.sparse_diagonal_response_size
        self.placement.row_watermark = bool(row_watermark)
        self.sweep.admit_packed(model, functools.partial(held_tree.sparse_decline, self))
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
        self.sweep.admit_paired()
        if not self.placement.hinv_jt_writes_world and self.sweep.paired_response_primary_size is None:
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
        self.sweep.plan_tier_rows()
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
        if (declined := capable("keyed_rows", self.laws.set)) is not None:
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
        self.sweep.build()
        self.sweep.open_streams(overlap=not _FPGS_SYNC_TIMINGS)
        self.placement.allocate_placeholders()

        self._audit = Audit(self) if self.options.pgs_audit else None
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
        self.sweep.resolved_worlds_owner = classifier if classifier.enabled else None
        self.sweep.resolved_worlds = classifier.resolved
        self.sweep.resolved_worlds_reason = classifier.reason

        self.placement.place_dense_response(paired_coordinates=self.sweep.paired_factor_coordinates)
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
            self.factor.kinematic_state(
                self.forest.joint_forest, sparse_velocity_limits=self.sweep.sparse_velocity_limits
            )
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
            self.sweep.clear_history(None)

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
        self.sweep.clear_history(world_mask)

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
        """Enable or disable the closure of a loop joint, of any type, at runtime.

        The closure keeps its allocated capacity and takes effect on the next :meth:`step`, also in a captured graph.
        Closures disabled through :attr:`~newton.Model.joint_enabled` at construction still count as capacity.

        Args:
            joint: Model joint index of a loop joint.
            enabled: ``True`` to enforce the closure, ``False`` to release it.
        """
        self.cut.set_enabled(joint, enabled)
        self.laws.set_links_enabled()
        self.notify_model_changed(ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)

    def set_loop_joint_anchors(self, joint: int, parent_anchor, child_anchor) -> None:
        """Move the anchors of a loop joint, of any type, at runtime: the origins of its frames on the model.

        Both anchors are body-local points [m] (world-frame for a world parent). Use it
        to re-target a closure before enabling it, for example to attach a body to a
        prescribed hand at the relative pose measured at acquisition.

        Args:
            joint: Model joint index of a loop joint.
            parent_anchor: Anchor in the parent body frame [m].
            child_anchor: Anchor in the child body frame [m].
        """
        self.cut.set_anchors(joint, parent_anchor, child_anchor)
        self.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

    def loop_joint_anchors(self, joint: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the current ``(parent_anchor, child_anchor)`` of a loop joint [m]."""
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

    def read_audit(self) -> dict:
        """Return the per-world audit of ``pgs_audit`` by name (:data:`.audit.AUDIT_SLOTS`), each a NumPy array.

        ``gap``, ``violation`` and ``work`` are the last step's ``G``, ``D`` and ``W``, NaN in a world without readable
        rows; ``gap_sum``, ``violation_max``, ``work_sum`` and ``steps`` cover the steps read since construction,
        captured graph replays included. This synchronizes with the device.
        """
        if self._audit is None:
            raise RuntimeError("read_audit() needs a solver built with pgs_audit=True.")
        return self._audit.read()

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
            if self.sweep.resolved_worlds_owner is not None:
                self.sweep.resolved_worlds_owner.classify(state_in, state_aug, contacts, dt)
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
            self.sweep.warm_start(contacts, dt)
        self.sweep.seed_patches(state_in, contacts, dt)

        with wp.ScopedTimer("S5_GatherJY", print=False, use_nvtx=self._nvtx, synchronize=False):
            if self.factor.eliminates:
                # Bilateral elimination: the other rows take the closed-loop response, their right-hand sides
                # and diagonals (the rows' closed-loop inertia).
                self.factor.eliminate(
                    first=self.placement.dense_first,
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
            if self.sweep.colored_units is not None:
                # The colored worlds' units and colors follow the rows the pack just published.
                self.sweep.colored_units.build(
                    self.placement.mf_constraint_count,
                    self.placement.mf_first,
                    self.placement.mf_contact_rows_end,
                    self.placement.mf_meta_packed,
                    self.placement.mf_Z_a,
                    self.placement.mf_Z_b,
                    self.placement.mf_row_mu,
                    self.placement.mf_row_w,
                    int(self.laws.set.weighted),
                )

        with wp.ScopedTimer("S6_PGS_Solve", print=False, use_nvtx=self._nvtx, synchronize=False):
            self.sweep.solve(self.v_hat, self.v_out)

        if self._audit is not None:
            self._audit.launch()

        # ══════════════════════════════════════════════════════════════
        # STAGE 7: Update qdd + integrate
        # ══════════════════════════════════════════════════════════════
        self.factor.finish(state_in, state_out, dt, self._step)
        self.integrate_particles(model, state_in, state_out, dt)

        self.sweep.carry(state_in, state_out, contacts, dt)

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
        self.sweep.forces(contacts)

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

    def _link_friction_patches(self, contacts):
        """Link shared normal loads before any consumer packs row metadata."""
        if self.laws.friction_anchors_enabled and self.sweep.patch_frame_built(contacts):
            self.laws.friction_patches.link(
                self.placement.contact_world,
                self.placement.contact_slot,
                self.placement.contact_path,
                self.placement.contact_slots_needed,
                self.sweep.patch_routes(),
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
                    self.placement.dense_first,
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
                if self.sweep.sparse_velocity_limits and size == self.placement.sparse_mass_matrix_size:
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
                            self.sweep.resolved_worlds,
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
                            self.sweep.resolved_worlds,
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
                inputs=[self.placement.dense_first, self.placement.dense_phase_bounds],
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
                if self.sweep.sparse_velocity_limits and size == self.placement.sparse_mass_matrix_size:
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
                    self.sweep.resolved_worlds,
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
                            contact_build_threads, contact_jacobian_workers * 32 // self.placement.sparse_contact_lanes
                        )
                        pool.launch(
                            self.placement.sparse_contact_response_kernel,
                            sparse_contact_workers * self.placement.sparse_contact_lanes,
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
                                self.placement.sparse_free_factor_dummy,
                            ],
                            outputs=[
                                self.placement.sparse_row_dof,
                                self.placement.sparse_row_factor,
                                # The record builder writes the triplet records in the free-response
                                # slot, which the support table's routes leave unused.
                                self.placement.sparse_row_free_response
                                if self.placement.sparse_contact_records is None
                                else self.placement.sparse_contact_records,
                                self.placement.row_incident,
                                self.placement.sparse_row_support,
                            ],
                            extent=count,
                            scale=self.placement.sparse_contact_lanes,
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
                            self.placement.dense_first,
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
                        pool.launch(
                            self.sweep.sparse_contact_marker,
                            contact_build_threads,
                            inputs=[self.placement.dense_first, contact_input, contact_build_threads],
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
                            dt / self.sweep.prev_dt if self.sweep.prev_dt > 0.0 else 1.0,
                            self.placement.mf_max_constraints,
                        ],
                        outputs=[self.placement.mf_impulses],
                        extent=count,
                        max_blocks=self.placement.contact_capacity_blocks,
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
                    self.sweep.resolved_worlds,
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
        self.sweep.claim()
        self.placement.warn_overflow()

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

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The placement: where each world's constraint rows live, how many fit, and what happens to those that do not
(BACKBONE §2).

:class:`Placement` owns what the rows are stored in across a step:

- the dense rows and their records, the free-body rows, the sparse placements' rows, the packed rows (Z by support) and
  each size group's response;
- the per-contact buffers of the contact pool (each contact's world, slot and path) and the carry's row tables;
- the capacities, the slot counters, the row-family bounds and the sorted placement of contact rows;
- the status: the per-world overflow latch, the sticky flags, dropped rows, warnings and watermarks.

A step calls :meth:`Placement.begin` (clear the allocation state), then the row pass writes into its slots
(:meth:`Placement.place_sorted_contact_rows`, :meth:`Placement.latch_status`, :meth:`Placement.warn_overflow`),
:meth:`Placement.response`, :meth:`Placement.diagonals` and :meth:`Placement.pack_rows` (``Z = L^-1 P J^T`` with
``|Z|^2``), :meth:`Placement.add_cfm`, :meth:`Placement.whiten_free_bodies`, :meth:`Placement.prepare_impulses` and
:meth:`Placement.pack_free_body_meta`.

It also holds which placement the class takes: the native sparse size, the sparse-diagonal pair, the row producers and
the local owners' per-world claim. The row pass and the sweeps read its fields.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.contact_pool import ContactField, model_contact_pool
from ...sim import Contacts
from ...sim.enums import JointType
from . import host_index, packed_rows, response_builds_dense, response_builds_rows
from .dense_rows import RowSupport
from .dense_rows import bind_mf as bind_mf_rows
from .field_group import SolverArrays
from .kernels import (
    RIGID_SPEED_ROWS,
    clear_local_solve_diag,
    place_world_contact_runs,
    prepare_world_impulses,
    record_world_contact_runs,
)
from .row_metadata import _validate_mf_metadata_encoding
from .solver_fields import (
    CAPACITY_STATUS,
    CONTACT_RUNS,
    DENSE_ROW_EPS,
    DENSE_ROW_WEIGHTS,
    DENSE_ROWS,
    DENSE_WARMSTART,
    FUSED_DIAGONAL_LIMITS,
    MF_COUNTERS,
    MF_ROW_EPS,
    MF_ROW_WEIGHTS,
    MF_ROWS,
    MF_WARMSTART,
    OVERFLOW_WARNING,
    ROW_SLOT_UNBOUNDED,
    ROW_WATERMARKS,
    SIZE_JACOBIAN,
    SIZE_RESPONSE,
    SPARSE_CONTACT_GROUPS,
    SPARSE_CONTACT_SERIAL,
    SPARSE_DIAGONAL_ROWS,
    WORLD_RESPONSE,
    WORLD_ROW_STATE,
)

__all__ = ["Placement"]

# Per-world preset of the first rejected row slot: no reservation was rejected.
_ROW_SLOT_UNBOUNDED = ROW_SLOT_UNBOUNDED
# The largest articulation the tiers whiten in-kernel: its factor is staged in the block (case K4).
_INKERNEL_MAX_DOF = 32


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


class Placement:
    """The placement of one solver (see the module docstring); the solver holds it as ``placement``.

    Built in the solver's construction order: the capacities and placeholders here, then
    :meth:`allocate_size_groups`, :meth:`allocate_contact_rows`, :meth:`allocate_world_rows`,
    :meth:`allocate_sparse_diagonal_rows`, :meth:`allocate_free_body_rows`, :meth:`allocate_packed_rows`,
    :meth:`allocate_placeholders`, :meth:`allocate_status`, :meth:`place_dense_response` and
    :meth:`build_response_kernels`. The native sparse route allocates its rows on :attr:`arrays` as it admits them.

    Args:
        model: The solver's model.
        options: The solver's physics options (:class:`.configuration.Options`).
        forest: The forest plan.
        laws: The row laws.
        factor: The tree factor.
    """

    def __init__(self, model, options, forest, laws, factor):
        self.model = model
        self.options = options
        self.forest = forest
        self.laws = laws
        self.factor = factor
        self.arrays = SolverArrays(self, model.device)
        self.dense_max_constraints = options.dense_max_constraints
        self.mf_max_constraints = options.mf_max_constraints
        _validate_mf_metadata_encoding(self.mf_max_constraints)
        # The carry's tables, allocated with their rows only when enabled.
        self.ws_prev_dense_impulses = None
        self.ws_prev_dense_row_type = None
        self.ws_prev_dense_row_parent = None
        self.ws_prev_dense_slot_sorted = None
        self.ws_prev_mf_impulses = None
        self.ws_prev_mf_row_type = None
        self.ws_prev_mf_row_parent = None
        self.ws_prev_slot_sorted = None
        # All inactive weight arguments alias this one element. Their producer kernels skip stores and their consumer
        # kernels skip reads when regularization is disabled.
        self.contact_row_w_dummy = self.arrays.placeholder(wp.float32, (1, 1), 1.0)
        self.contact_row_eps_dummy = self.arrays.placeholder(wp.float32, (1, 1))
        self.row_eps = self.mf_row_eps = self.contact_row_eps_dummy
        self.J_by_size: dict[int, wp.array] = {}
        self.Y_by_size: dict[int, wp.array] = {}
        self.diag_by_size: dict[int, wp.array] = {}

    # ---------------------------------------------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------------------------------------------

    def internal_rows_per_world(self, model) -> int:
        """The most mimic, connect and limit rows any world can hold, from the topology."""
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
                self.forest.response_dof_count[art], list(self.fused_diagonal_joint_limit_sizes)
            )
            rows += ~fuse_position_limits & np.isfinite(joint_limit_lower[dof])
            rows += ~fuse_position_limits & np.isfinite(joint_limit_upper[dof])
        if has_velocity_limit_rows:
            rows += 2 * (joint_velocity_limit[dof] > 0.0)
        world = art_to_world[art]
        in_range = (world >= 0) & (world < per_world.size)
        np.add.at(per_world, world[in_range], rows[in_range])

        return int(np.max(per_world)) if per_world.size else 0

    def allocate_contact_rows(self, model):
        """The per-contact buffers (the contact pool's ready rows), the row reservation and the contact runs."""
        arrays = self.arrays
        device = model.device
        # Every per-contact buffer holds the ready rows of the model's contact pool, which its
        # collision pipelines share: they are sized with the pipeline's measured demand before a graph
        # is captured, and grow with any other contact stream the solver consumes (see follow_contacts).
        self.contact_pool = model_contact_pool(model, device)
        contact_fields = [
            ContactField(name, wp.int32)
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
        contact_fields.append(ContactField("rows_prelude_flags", wp.int32))
        # Row builders always receive a zero phi view when patch friction is off.
        contact_fields.append(ContactField("friction_patch_phi", wp.vec2))
        if self.options.pgs_warmstart:
            contact_fields.append(ContactField("ws_prev_contact_normal", wp.vec3))
        else:
            self.ws_prev_contact_normal = None
        self.contact_rows = self.contact_pool.rows(contact_fields, on_rows=self._on_contact_rows)
        # The friction patch state and the velocity-limit slots.
        self.laws.allocate(model, self.contact_pool, self.friction_patch_phi)
        # Allocated once here: per-step allocation is forbidden under CUDA graph capture.
        arrays.allocate(WORLD_ROW_STATE, world=self.forest.world_count)
        # Contact runs of each world and block under sorted contacts (record_world_contact_runs):
        # blocks are the global shapes before every world's, the worlds' own shapes, and the global
        # shapes after them; 16 bytes per world and block.
        shape_world_start = model.shape_world_start.numpy() if model.shape_world_start is not None else None
        if shape_world_start is None or model.shape_count == 0:
            self.contact_run_lead, self.contact_run_trail_start = 0, int(model.shape_count)
        else:
            self.contact_run_lead = int(shape_world_start[0])
            self.contact_run_trail_start = int(shape_world_start[-2])
        run_blocks = self.contact_run_lead + 1 + int(model.shape_count) - self.contact_run_trail_start
        arrays.allocate(CONTACT_RUNS, world=self.forest.world_count, run_block=run_blocks)

    def allocate_size_groups(self, *, double_buffered: bool) -> None:
        """Each size group's response and Jacobian (two sets where the memset stream clears them) and its diagonal."""
        if not self.model.articulation_count:
            self.dummy_hinv_diag = None
            self.J_bufs = None
            return

        device = self.model.device
        arrays = self.arrays
        self.dummy_hinv_diag = wp.zeros((1, 1), dtype=wp.float32, device=device)
        self.J_bufs = [{}, {}] if double_buffered else None
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            sparse_response = self.sparse_mass_matrix_size is not None or (
                self.sparse_diagonal_contact_solve and size == self.sparse_diagonal_response_size
            )
            # The matrix-free sweeps' producers write the packed rows (jacobian_rows): no size group's Jacobian.
            response = 1 if sparse_response or self.mf_sweeps else 0
            # The matrix-free sweeps read the packed rows and the sparse-diagonal owner whitens its dense rows in place
            # (SparsePairZ): no route reads a size group's response.
            whitened = self.sparse_diagonal_contact_solve and size == self.sparse_diagonal_dense_size
            y = 1 if sparse_response or self.mf_sweeps or whitened else 0
            sizes = {
                "art": n_arts,
                "size": size,
                "response_art": response or n_arts,
                "response_row": response or self.dense_max_constraints,
                "response_size": response or size,
                "y_art": y or n_arts,
                "y_row": y or self.dense_max_constraints,
                "y_size": y or size,
            }
            self.Y_by_size[size] = arrays.allocate(SIZE_RESPONSE, owner={}, **sizes)["Y"]
            # Double buffering gives a size group two Jacobians only where the memset stream clears the rows between
            # uses; else both sets share one.
            J = arrays.allocate(SIZE_JACOBIAN, owner={}, **sizes)["J"]
            if double_buffered:
                self.J_bufs[0][size] = J
                second = self.clears_jacobian(size)
                self.J_bufs[1][size] = arrays.allocate(SIZE_JACOBIAN, owner={}, **sizes)["J"] if second else J
            else:
                self.J_by_size[size] = J
            self.diag_by_size[size] = (
                wp.zeros((n_arts, self.dense_max_constraints), dtype=wp.float32, device=device)
                if size in self.hinv_jt_diag_sizes and not sparse_response
                else self.dummy_hinv_diag
            )

        if self.J_bufs is not None:
            self.J_by_size = self.J_bufs[0]

    def clears_jacobian(self, size: int) -> bool:
        """Whether the memset stream clears a size group's active Jacobian rows between uses (double-buffered on CUDA):
        not under the matrix-free sweeps (their producers write the packed rows' sides whole), nor for the
        sparse-diagonal response size (no dense Jacobian) or the compact boundary's six coordinates (it clears its
        rows itself); :attr:`.step_plan.StepPlan.jacobian_clear_sizes` adds the owners of a private Jacobian."""
        return not (
            self.mf_sweeps
            or (self.sparse_diagonal_contact_solve and size == self.sparse_diagonal_response_size)
            or (self.compact_contact_boundary and size == 6)
        )

    def _on_contact_rows(self, rows) -> None:
        """Assign the per-contact buffers at the contact pool's ready rows, on creation and after each growth."""
        rows.publish(self)
        self.max_contacts_alloc = rows.rows
        patches = self.laws.friction_patches
        if patches is not None:
            patches.view.phi = self.friction_patch_phi

    def _publish_contact_rows(self, rows) -> None:
        rows.publish(self)

    def follow_contacts(self, contacts: Contacts) -> None:
        """Hold at least ``contacts``' rigid rows, and grow with the pool of a pipeline's buffer from now on."""
        pool = getattr(self, "contact_pool", None)
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

    def allocate_world_rows(self, model):
        """Allocate world-level constraint system buffers for multi-articulation support."""
        if self.forest.world_count == 0:
            return

        arrays = self.arrays
        # The widest world's DOFs exist where a world response mapping was built (the routes that read them).
        world_dof = self.forest.max_world_dofs
        sizes = {"world": self.forest.world_count, "row": self.dense_max_constraints, "world_dof": world_dof}
        # The world-indexed J and Y of the routes that build a response in the world layout.
        if self.sparse_diagonal_contact_solve or self.sparse_mass_matrix_size is not None:
            arrays.allocate(WORLD_RESPONSE, world=1, row=1, world_dof=1)
        elif self.mf_sweeps:
            # The matrix-free sweeps read the packed rows (place_dense_response): no world layout.
            arrays.release(WORLD_RESPONSE)
        elif self.jy_world_aliased:
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

        arrays.allocate(DENSE_ROWS, **sizes)
        # Dense identity-matched warm-start carry buffers. Allocated here (not in the MF
        # buffer path) so scenes WITHOUT free rigid bodies still get them —
        # allocate_free_body_rows early-returns for such scenes and the matched
        # carry must never silently degrade to index reuse.
        if self.options.pgs_warmstart:
            arrays.allocate(DENSE_WARMSTART, **sizes)
            self.ws_prev_dense_slot_rows = self.contact_pool.rows(
                (ContactField("ws_prev_dense_slot_sorted", wp.int32, -1),), on_rows=self._publish_contact_rows
            )
        fused = self.fused_diagonal_joint_limits
        arrays.allocate(FUSED_DIAGONAL_LIMITS, **(sizes if fused else {"world": 1, "world_dof": 1}))
        # Per-row weights exist only for a weighted row law (relative regularization or material contacts), and
        # per-row compliance only for material contacts.
        if self.laws.set.weighted:
            arrays.allocate(DENSE_ROW_WEIGHTS, **sizes)
        else:
            self.row_w = self.contact_row_w_dummy
        if self.options.contact_compliance:
            arrays.allocate(DENSE_ROW_EPS, **sizes)
        else:
            self.row_eps = self.contact_row_eps_dummy

    def allocate_sparse_diagonal_rows(self):
        """The sparse-diagonal rows and their contact schedule (case K2)."""
        # The sparse-diagonal schedule's buffers, one row per world when it is off.
        if self.sparse_diagonal_contact_solve:
            self.arrays.allocate(SPARSE_DIAGONAL_ROWS, world=self.forest.world_count, row=self.dense_max_constraints)
        else:
            self.arrays.allocate(SPARSE_DIAGONAL_ROWS, world=1, row=1)
        if self.sparse_diagonal_contact_triples or self.sparse_diagonal_contact_units:
            self.arrays.allocate(
                SPARSE_CONTACT_GROUPS, world=self.forest.world_count, world_dof=self.forest.max_world_dofs
            )
        else:
            self.arrays.allocate(SPARSE_CONTACT_GROUPS, world=1, world_dof=1)
        if self.sparse_diagonal_speculative_contact_batches:
            # A triple spans three rows, while a unit may be a single normal row.
            rows = self.dense_max_constraints
            serial = rows if self.sparse_diagonal_contact_units else (rows + 2) // 3
            self.arrays.allocate(SPARSE_CONTACT_SERIAL, world=self.forest.world_count, serial=serial)
        else:
            self.arrays.allocate(SPARSE_CONTACT_SERIAL, world=1, serial=1)

    def allocate_free_body_rows(self, model):
        """Allocate buffers for matrix-free PGS path for free rigid body contacts.

        CUDA without free rigid bodies still passes the
        ``mf_*`` arrays to kernels (the fused GS solve, the PGS diagnostics),
        but ``mf_constraint_count`` stays zero (``_clear_dense_row_state`` and
        ``finalize_constraint_counts_with_status`` write it only when MF is
        active), so every access sits inside a loop bounded by
        ``m_mf == 0`` and no element is ever read or written; :meth:`pack_free_body_meta`
        skips its launch. The rows then hold one slot per world and the per-world
        counts keep their real width.
        """
        arrays = self.arrays
        worlds = self.forest.world_count
        rows = self.mf_max_constraints if self.forest.has_free_rigid_bodies else 1
        arrays.allocate(MF_COUNTERS, world=worlds)
        arrays.allocate(MF_ROWS, world=worlds, mf_row=rows, mf_meta=4 * rows)
        if self.laws.set.weighted and self.forest.has_free_rigid_bodies:
            arrays.allocate(MF_ROW_WEIGHTS, world=worlds, mf_row=rows)
        else:
            self.mf_row_w = self.contact_row_w_dummy
        if self.options.contact_compliance and self.forest.has_free_rigid_bodies:
            arrays.allocate(MF_ROW_EPS, world=worlds, mf_row=rows)
        else:
            self.mf_row_eps = self.contact_row_eps_dummy

        if not self.forest.has_free_rigid_bodies:
            # Only the ``is not None`` checks observe this on the inactive
            # path, where no free rigid body owns a speed row.
            self.rigid_velocity_limit_slot = (
                wp.full((1,), -1, dtype=wp.int32, device=model.device)
                if self.laws.has_rigid_body_velocity_limits and model.body_count > 0
                else None
            )
            return

        # MF warm-start carry buffers. Allocated ONLY when the feature is on so
        # the default-off path has zero memory / CUDA-graph delta. ``prev_mf_*``
        # snapshot last step's converged impulses + their row-type table;
        # ``ws_prev_slot_sorted`` records, per *current sorted contact index*,
        # the base MF slot that contact occupied last step. Because FeatherPGS
        # consumes the already-sorted contact arrays, the contact array index is
        # the sorted index, so next step's ``rigid_contact_match_index`` (which
        # references the previous frame's *sorted* index) indexes directly into
        # this table — no matcher permutation access required.
        if self.options.pgs_warmstart:
            arrays.allocate(MF_WARMSTART, world=worlds, mf_row=rows)
            pool = getattr(self, "contact_pool", None)
            if pool is None:
                self.ws_prev_slot_sorted = wp.full((1,), -1, dtype=wp.int32, device=model.device)
            else:
                self.ws_prev_slot_rows = pool.rows(
                    (ContactField("ws_prev_slot_sorted", wp.int32, -1),), on_rows=self._publish_contact_rows
                )

        if self.laws.has_rigid_body_velocity_limits and self.forest.free_rigid_body_count > 0:
            # Two speed rows per free rigid body: its linear and its angular speed.
            self.rigid_velocity_limit_slot = wp.full(
                (RIGID_SPEED_ROWS * self.forest.free_rigid_body_count,),
                -1,
                dtype=wp.int32,
                device=model.device,
            )
        else:
            self.rigid_velocity_limit_slot = None

    def allocate_packed_rows(self) -> None:
        """The matrix-free sweeps' dense rows, packed by support (:mod:`.packed_rows`): every world's row capacity,
        each row as wide as two of its world's articulations hold; a placeholder without those sweeps."""
        if self.mf_sweeps:
            sizes, worlds = self.forest.articulation_response_dof_count.numpy(), self.forest.art_to_world.numpy()
            offsets = self.forest.articulation_world_dof_offset.numpy()
            # A row's sides follow articulation indices (dense_rows.row_sides); they follow the world's coordinates
            # too, so packed entries ascend with them.
            order = np.lexsort((np.arange(len(worlds)), worlds))
            responding = order[sizes[order] > 0]
            same_world = worlds[responding][1:] == worlds[responding][:-1]
            if np.any(same_world & (offsets[responding][1:] < offsets[responding][:-1])):
                raise RuntimeError("Packed rows need each world's coordinates in articulation order")
            self.packed_support = packed_rows.support_width(sizes, worlds, self.forest.world_count)
            # Two articulations hold every coordinate of the widest world: the rows are laid out at their coordinates.
            self.packed_dense = self.packed_support == self.forest.max_world_dofs
            if self.ink_rows and not self.packed_dense:
                raise RuntimeError("The parallel tiers read their worlds' packed rows at their coordinates")
            sizes = {
                "world": self.forest.world_count,
                "row": self.dense_max_constraints,
                "support": self.packed_support,
            }
        else:
            self.packed_support, self.packed_dense = 1, False
            sizes = {"world": 1, "row": 1, "support": 1}
        self.arrays.allocate(packed_rows.PACKED_ROWS, **sizes)

    def place_dense_response(self, *, paired_coordinates: bool) -> None:
        """Where the dense rows' response lives (``packed_response`` and ``packed_response_reason``).

        The matrix-free sweeps, and every route beside them, read the dense rows packed by support
        (:mod:`.packed_rows`): the packed build alone serves the step, the response stage builds nothing and the world
        layout and the size groups' responses are not kept.
        """
        self.packed_response_reason = None if self.mf_sweeps else "No matrix-free sweep reads packed rows."
        self.packed_response = self.packed_response_reason is None
        if not self.packed_response:
            if paired_coordinates:
                raise RuntimeError(
                    f"The paired factor-coordinate sweep reads packed rows: {self.packed_response_reason}"
                )
            return
        if paired_coordinates and not self.packed_dense:
            raise RuntimeError("The paired factor-coordinate sweep reads packed rows at their world coordinates")
        self.hinv_jt_writes_world = False
        self.jy_world_aliased = False
        self.arrays.release(WORLD_RESPONSE)
        placeholder = self.arrays.placeholder(wp.float32, (1, 1, 1))
        for size in self.forest.size_groups:
            self.Y_by_size[size] = placeholder
            self.diag_by_size[size] = self.dummy_hinv_diag
        self.hinv_jt_diag_sizes = frozenset()

    def inkernel_response_meta(self) -> tuple[tuple[int, int, int, int] | None, str]:
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
        self.ink_meta = wp.array(meta.reshape(-1), dtype=wp.int32, device=self.model.device)
        return (sizes[0], nb, offsets[0], offsets[1]), (
            f"The parallel tiers whiten the rows of sizes {sizes} in the kernel."
        )

    def allocate_status(self):
        """The capacity status, the overflow warning and the row telemetry."""
        # Sticky flags of constraint_capacity_status() (dense and matrix-free rows), the per-world overflow
        # latch and the dropped rows of each family.
        self.arrays.allocate(
            CAPACITY_STATUS, world=self.forest.world_count, status_world=max(self.forest.world_count, 1)
        )
        self.row_dropped_dense = self.row_dropped_all[0]
        self.row_dropped_mf = self.row_dropped_all[1]
        # Constraint-capacity diagnostics are allocated once before CUDA graph
        # capture. The warning path tracks only current-step dropped rows and a
        # one-shot flag per row family; row_watermark additionally accumulates
        # whole-run high-water telemetry for explicit readback.
        self.track_row_capacity = self.row_watermark or self.options.warn_constraint_overflow
        # Dropped-row accounting is unconditional (see row_dropped_all above);
        # track_row_capacity now gates only the optional warning/watermark readbacks.
        if self.options.warn_constraint_overflow:
            self.arrays.allocate(OVERFLOW_WARNING)
        else:
            self.arrays.release(OVERFLOW_WARNING)
        # The telemetry words share one array, which constraint_row_watermarks() reads back once.
        self.row_watermark_words = None
        if self.row_watermark:
            self.row_watermark_words = self.arrays.allocate_words(ROW_WATERMARKS)
        else:
            self.arrays.release(ROW_WATERMARKS)

    def allocate_placeholders(self):
        """The operands of the features that are off."""
        # Operands of the features that are off (SolverArrays.placeholder). reset_world_warmstart_buffers
        # writes -1 into its row-type and parent arguments, so their sink holds -1; the matrix-free counters
        # stand in at world width, and allocate_world_contact_slots writes none of them when has_free_rigid == 0.
        placeholder = self.arrays.placeholder
        self.dummy_contact_impulses = placeholder(wp.float32, (1, 1))
        self.dummy_world_J = placeholder(wp.float32, (1, 1, 1))
        self.dummy_is_free_rigid = placeholder(wp.int32, (1,))
        self.dummy_mf_slot_counter = placeholder(wp.int32, (self.forest.world_count,))
        self.dummy_mf_impulses = placeholder(wp.float32, (1, 1))
        self.dummy_contact_count = placeholder(wp.int32, (1,))
        self.dummy_contact_row_type = placeholder(wp.int32, (1, 1))
        self.reset_dummy_mf_row_type = placeholder(wp.int32, (1, 1), -1)
        self.reset_dummy_mf_row_parent = placeholder(wp.int32, (1, 1), -1)
        self.dummy_contact_row_parent = placeholder(wp.int32, (1, 1), -1)

    def build_response_kernels(self, plan, execution_plan, tile_threads: int) -> None:
        """Build the stage-4 kernels the plan's response table launches, and only those (one entry per size)."""
        self.tile_threads = threads = tile_threads
        rows = self.dense_max_constraints
        sizes = self.forest.size_groups

        def routed(stage, size, route):
            entry = getattr(plan, stage).get(size)
            return entry is not None and entry.name == route

        self.hinv_jt_kernels_by_size = {}
        self.hinv_jt_chunk_count_by_size = {}
        for size in sizes:
            chunk = execution_plan.hinv_jt_chunk_size(size) if routed("response", size, "tiles") else None
            self.hinv_jt_chunk_count_by_size[size] = 0 if chunk is None else (rows + chunk - 1) // chunk
            self.hinv_jt_kernels_by_size[size] = (
                None
                if chunk is None
                else response_builds_dense.response_tiles_kernel(
                    size,
                    rows,
                    threads,
                    constraint_chunk_size=chunk,
                    write_world=self.hinv_jt_writes_world,
                    write_group=self.hinv_jt_tiled_writes_group,
                    compute_diag=size in self.hinv_jt_diag_sizes,
                )
            )
        # Sizes the plan routes to the per-row kernel (e.g. free bodies) consume the group-layout response (diag
        # from J.Y, world impulse apply), so the grouped kernel writes the group copy for them.
        self.grouped_kernels_by_size = {
            size: response_builds_dense.response_warps_kernel(
                size,
                int(rows),
                write_world=bool(self.hinv_jt_writes_world),
                write_group=bool(self.hinv_jt_tiled_writes_group) or not execution_plan.use_tiled_hinv_jt(size),
                compute_diag=size in self.hinv_jt_diag_sizes,
            )
            for size in plan.sizes("response", "grouped")
        }

    # ---------------------------------------------------------------------------------------------------------------
    # Step: the rows' slots, counts and status
    # ---------------------------------------------------------------------------------------------------------------

    def begin(self):
        """Clear the rows' allocation state before the row pass writes it."""
        # One launch clears the world-level allocation state and, with free rigid bodies, the
        # matrix-free counters and impulses (PGS reads mf_impulses before its first write; the
        # warm-start seed pass re-gathers after build_mf_contact_rows). Other row buffers are
        # fully overwritten by their producers.
        mf_active = self.forest.has_free_rigid_bodies
        wp.launch(
            _clear_dense_row_state,
            dim=(self.forest.world_count, self.mf_impulses.shape[1] if mf_active else 1),
            inputs=[
                self.slot_counter,
                self.row_dropped_all,
                self.dense_first_rejected_slot,
                1 if mf_active else 0,
                self.mf_slot_counter if mf_active else self.dummy_mf_slot_counter,
                self.mf_first_rejected_slot if mf_active else self.dummy_mf_slot_counter,
                self.mf_constraint_count if mf_active else self.dummy_mf_slot_counter,
                self.mf_impulses if mf_active else self.dummy_mf_impulses,
            ],
            device=self.model.device,
        )

    def place_sorted_contact_rows(self, contacts, contact_build_threads, mf_slot_counter, mf_first_rejected_slot):
        """Give each world's contact rows the order of sorted contacts, whatever order the allocator reserved them in.

        Runs only when the contacts come from a deterministic collision pipeline
        (``Contacts.rigid_contacts_pair_sorted``), whose determinism Newton promises when the
        solver is deterministic as well; the default pipeline keeps the allocator's atomic
        reservation, whose order on CUDA follows the warps. A captured step keeps the order of
        its capture. Dense rows are placed unless the local row packets or keyed rows place
        their own; matrix-free rows are placed with free rigid bodies.
        """
        order_dense = int(self.row_packets is None and self.keyed_rows is None)
        order_mf = int(self.forest.has_free_rigid_bodies)
        if not (order_dense or order_mf):
            return
        device = self.model.device
        runs = (self.contact_run_first, self.contact_run_last, self.contact_run_rows)
        blocks = (self.contact_run_lead, self.contact_run_trail_start)
        # Complete warps on CUDA, whose lanes merge each run's contacts before recording it.
        lanes = 32 if device.is_cuda else 1
        record_threads = -(-contact_build_threads // lanes) * lanes
        # Its warps merge runs, so the grid stays whole: no extent follows the count.
        self.contact_pool.launch(
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
                self.dense_first_rejected_slot,
                mf_slot_counter,
                mf_first_rejected_slot,
                *blocks,
                lanes,
            ],
            outputs=[*runs, self.contact_slot],
            device=device,
        )

    def latch_status(self, contacts: Contacts | None) -> None:
        """Latch current per-world capacity failures for every row owner."""
        # The contact capacity (argument 8) follows the ready rows in a capture whose launches bind.
        self.contact_pool.launch(
            _finalize_constraint_status,
            self.forest.world_count,
            inputs=[
                self.slot_counter,
                self.mf_slot_counter if self.forest.has_free_rigid_bodies else self.dummy_mf_slot_counter,
                self.row_dropped_all,
                contacts.rigid_contact_count if contacts is not None else self.dummy_contact_count,
                contacts._reduction_overflow if contacts is not None else self.dummy_contact_count,
                self.dense_max_constraints,
                self.mf_max_constraints,
                contacts._rigid_contact_rows if contacts is not None else 0,
                self.constraint_overflow,
            ],
            row_args=(8,) if contacts is not None else (),
        )

    def warn_overflow(self):
        """One device-side warning per overflowing row family."""
        mf_active = self.forest.has_free_rigid_bodies
        if self.options.warn_constraint_overflow:
            mf_raw_counts = self.mf_slot_counter if mf_active else self.slot_counter
            wp.launch(
                _warn_constraint_row_overflow,
                dim=self.forest.world_count,
                inputs=[
                    self.slot_counter,
                    self.row_dropped_dense,
                    self.dense_max_constraints,
                    mf_raw_counts,
                    self.row_dropped_mf,
                    self.mf_max_constraints,
                    1 if mf_active else 0,
                    self.row_overflow_warning_emitted,
                ],
                device=self.model.device,
            )

    def reset_status(self, world_mask):
        """Clear the selected worlds' overflow latch and request their factors."""
        wp.launch(
            _reset_solver_status,
            dim=max(self.forest.world_count, self.model.articulation_count),
            inputs=[world_mask, self.forest.art_to_world, self.factor.mass_update_requested, self.constraint_overflow],
            device=self.model.device,
        )

    def record_watermarks(self, contacts):
        """Opt-in row high-water telemetry (behavior-neutral): at the end of the step, after the row pass has finalized
        the per-world counts, so the atomic-max runs inside the captured graph every step and catches mid-run peaks."""
        if self.row_watermark:
            wp.launch(
                _accumulate_row_watermark,
                dim=self.constraint_count.shape[0],
                inputs=[self.constraint_count, self.row_watermark_dense],
                device=self.constraint_count.device,
            )
            wp.launch(
                _accumulate_row_capacity_telemetry,
                dim=self.slot_counter.shape[0],
                inputs=[
                    self.slot_counter,
                    self.row_dropped_dense,
                    self.dense_max_constraints,
                    self.row_watermark_dense_raw,
                    self.row_dropped_dense_high_water,
                    self.row_overflow_dense_excess,
                    self.row_overflow_dense_world_steps,
                ],
                device=self.slot_counter.device,
            )
            mf_counts = getattr(self, "mf_constraint_count", None)
            if mf_counts is not None and mf_counts.shape[0] > 0:
                wp.launch(
                    _accumulate_row_watermark,
                    dim=mf_counts.shape[0],
                    inputs=[mf_counts, self.row_watermark_mf],
                    device=mf_counts.device,
                )
                wp.launch(
                    _accumulate_row_capacity_telemetry,
                    dim=self.mf_slot_counter.shape[0],
                    inputs=[
                        self.mf_slot_counter,
                        self.row_dropped_mf,
                        self.mf_max_constraints,
                        self.row_watermark_mf_raw,
                        self.row_dropped_mf_high_water,
                        self.row_overflow_mf_excess,
                        self.row_overflow_mf_world_steps,
                    ],
                    device=self.mf_slot_counter.device,
                )
            if contacts is not None and getattr(contacts, "rigid_contact_count", None) is not None:
                contact_counts = contacts.rigid_contact_count
                wp.launch(
                    _accumulate_row_watermark,
                    dim=contact_counts.shape[0],
                    inputs=[contact_counts, self.row_watermark_contact],
                    device=contact_counts.device,
                )

    def capacity_status(self, *, clear: bool = False) -> dict[str, bool]:
        """See :meth:`.SolverFeatherPGS.constraint_capacity_status`."""
        if wp.get_device(self.model.device).is_capturing:
            raise RuntimeError("Constraint capacity status must be read outside CUDA graph capture")
        dense, matrix_free = map(bool, self.constraint_capacity_status.numpy())
        result = {"dense": dense, "matrix_free": matrix_free}
        frame_status = self.laws.patch_frame_status()
        result["patch_frames"] = frame_status is not None and bool(frame_status.numpy()[0])
        pool = getattr(self, "contact_pool", None)
        result["contact_launches"] = pool is not None and pool.launch_failed(clear=clear)
        if clear:
            self.constraint_capacity_status.zero_()
            if frame_status is not None:
                frame_status.zero_()
        return result

    def check_capacity(self) -> None:
        """See :meth:`.SolverFeatherPGS.check_constraint_capacity`."""
        if self.model.device.is_cuda and self.model.device.is_capturing:
            raise RuntimeError("check_constraint_capacity() must run outside CUDA graph capture")
        # Skipped contact launches come first: they leave the rows the other checks read unbuilt.
        pool = getattr(self, "contact_pool", None)
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

    def row_watermarks(self) -> dict:
        """See :meth:`.SolverFeatherPGS.constraint_row_watermarks`."""
        words = {}
        if self.row_watermark_words is not None:
            words = dict(
                zip((field.name for field in ROW_WATERMARKS), self.row_watermark_words.numpy().tolist(), strict=True)
            )
        fields = {
            "high_water": "row_watermark_{}",
            "raw_high_water": "row_watermark_{}_raw",
            "dropped_contact_rows_high_water": "row_dropped_{}_high_water",
            "overflow_excess_high_water": "row_overflow_{}_excess",
            "overflow_world_steps": "row_overflow_{}_world_steps",
        }
        result = {
            f"{family}_{field}": int(words.get(name.format(family), 0))
            for family in ("dense", "mf")
            for field, name in fields.items()
        }
        result["contact_high_water"] = int(words.get("row_watermark_contact", 0))
        return result

    # ---------------------------------------------------------------------------------------------------------------
    # Step: where the rows' Jacobians live
    # ---------------------------------------------------------------------------------------------------------------

    def row_support(self) -> RowSupport:
        """Where the dense rows' Jacobian entries sit (:class:`.dense_rows.RowSupport`): the matrix-free sweeps' packed
        rows (their sides, the articulations' coordinates, first world coordinates and worlds, and the layout), or
        each size group's rows. Without the matrix-free sweeps there are no packed rows: the articulations' offsets and
        worlds are one-entry placeholders."""
        support = RowSupport()
        support.sides = self.row_sides
        support.dofs = self.forest.articulation_response_dof_count
        support.offset = self.forest.articulation_world_dof_offset if self.mf_sweeps else self.dummy_is_free_rigid
        support.world = self.forest.art_to_world if self.mf_sweeps else self.dummy_is_free_rigid
        support.dense = int(self.packed_dense)
        support.packed = int(self.mf_sweeps)
        return support

    def jacobian_rows(self, size: int) -> wp.array:
        """The Jacobian rows the producers of a size's rows write (:func:`.dense_rows.jacobian_row`): the packed rows
        (:mod:`.packed_rows`) under the matrix-free sweeps, else the size group's rows."""
        return self.packed_J if self.mf_sweeps else self.J_by_size[size]

    def mf_count_or_dummy(self):
        """Matrix-free row counts, or a one-element dummy when the model has no contact rows (never indexed then)."""
        arr = getattr(self, "mf_constraint_count", None)
        return arr if arr is not None else self.dummy_is_free_rigid

    def ink_skip_rows(self) -> int:
        """The dense rows of the worlds the tiers own, whose response they form in-kernel (0 without tiers)."""
        return int(getattr(self, "ink_rows", 0))

    def inkernel_response_inputs(self) -> list:
        """The parallel tiers' in-kernel response operands: the worlds' group indices, the primary and secondary
        groups' factors (the primary's for a model without a secondary), the packed rows' Jacobian and the rows'
        cfm."""
        na, nb = self.ink_sizes[0], self.ink_sizes[1]
        L_b = self.factor.L_by_size[nb] if nb else self.factor.L_by_size[na]
        return [self.ink_meta, self.factor.L_by_size[na], L_b, self.packed_J, self.row_cfm]

    # ---------------------------------------------------------------------------------------------------------------
    # Step: the response, the whitening and the diagonals
    # ---------------------------------------------------------------------------------------------------------------

    def _response_targets(self, size: int) -> tuple:
        """The world-layout J and Y a size's response writes (its group buffers where none is allocated) and the
        DOF offsets it writes them at."""
        J_world = self.J_world if self.J_world is not None else self.J_by_size[size]
        Y_world = self.Y_world if self.Y_world is not None else self.Y_by_size[size]
        offsets = (
            self.forest.articulation_world_dof_offset if self.hinv_jt_writes_world else self.forest.group_to_art[size]
        )
        return J_world, Y_world, offsets

    def response(self, size: int, route: str) -> None:
        """Build one size's contact response along the plan's stage-4 route (none where an owner publishes it)."""
        if route == "diagonal":
            self._response_diagonal(size)
        elif route == "grouped":
            self._response_grouped(size)
        elif route == "tiles":
            self._response_tiled(size)
        elif route == "lanes":
            self._response_lanes(size)
        elif route == "rows":
            self._response_rows(size)

    def _response_grouped(self, size: int) -> None:
        """The grouped response solve (grouped dynamics), a warp per articulation; verified up to 32 DOFs."""
        n_arts = self.forest.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        diag_out = self.diag_by_size[size] if size in self.hinv_jt_diag_sizes else self.grouped_dummy_diag
        wpb = 4
        wp.launch_tiled(
            self.grouped_kernels_by_size[size],
            dim=[(n_arts + wpb - 1) // wpb],
            inputs=[
                wpb,
                n_arts,
                self.factor.L_by_size[size],
                self.J_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
                world_dof_offset,
                self.constraint_count,
                self.Y_by_size[size],
                J_world,
                Y_world,
                diag_out,
                self.mf_count_or_dummy(),
                int(self.ink_skip_rows()),
            ],
            outputs=[],
            block_dim=32 * wpb,
            device=self.model.device,
        )

    def _response_tiled(self, size: int):
        model = self.model
        n_arts = self.forest.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        outputs = [self.Y_by_size[size], J_world, Y_world]
        if size in self.hinv_jt_diag_sizes:
            outputs.append(self.diag_by_size[size])
        wp.launch_tiled(
            self.hinv_jt_kernels_by_size[size],
            dim=[n_arts, self.hinv_jt_chunk_count_by_size[size]],
            inputs=[
                self.factor.L_by_size[size],
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

    def _response_diagonal(self, size: int) -> None:
        """Apply diagonal generalized inertia to the active response rows."""
        n_arts = self.forest.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_diagonal_kernel(),
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.factor.L_by_size[size],
                self.J_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
                world_dof_offset,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
                int(self.hinv_jt_tiled_writes_group),
                int(self.hinv_jt_writes_world),
                int(size in self.hinv_jt_diag_sizes),
            ],
            outputs=[self.Y_by_size[size], J_world, Y_world, self.diag_by_size[size]],
            device=self.model.device,
        )

    def _response_lanes(self, size: int) -> None:
        """The response a warp per articulation, skipping the worlds the local owners solve (route ``lanes``)."""
        n_arts = self.forest.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_lanes_kernel(),
            dim=n_arts * 32,
            inputs=[
                self.factor.L_by_size[size],
                self.J_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
                world_dof_offset,
                self.constraint_count,
                self.local_solve_owner,
                self.row_restitution,
                size,
                self.dense_max_constraints,
                n_arts,
                int(self.hinv_jt_writes_world),
            ],
            outputs=[self.Y_by_size[size], J_world, Y_world],
            device=self.model.device,
            block_dim=256,
        )

    def _response_rows(self, size: int) -> None:
        """The response a thread per row (route ``rows``)."""
        n_arts = self.forest.n_arts_by_size[size]
        J_world, Y_world, world_dof_offset = self._response_targets(size)
        wp.launch(
            response_builds_dense.response_rows_kernel(),
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.factor.L_by_size[size],
                self.J_by_size[size],
                self.forest.group_to_art[size],
                self.forest.art_to_world,
                world_dof_offset,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
                int(self.hinv_jt_writes_world),
            ],
            outputs=[self.Y_by_size[size], J_world, Y_world],
            device=self.model.device,
        )

    def pack_rows(self, v_hat: wp.array, *, write_diag: int) -> None:
        """Pack and whiten every world's dense rows and store the general worlds' diagonals as ``write_diag`` asks
        (:func:`.packed_rows.pack_kernel`, one launch; :meth:`diagonals`)."""
        if not self.mf_sweeps:
            return
        # The parallel tiers whiten their worlds' rows in the kernel, and under the row packets the local owners
        # solve their own and the packet producer applies restitution: the build leaves both, as main's response did,
        # unless the warm start applies their impulses through the packed rows. Elsewhere the local owners' worlds
        # are packed: restitution and the drive descriptor read their J v there.
        skip = 0 if self.options.pgs_warmstart else int(self.ink_skip_rows())
        skip_local = int(self.row_packets is not None and not self.options.pgs_warmstart)
        # A compact or sparse size keeps a placeholder factor: its rows have no dense factor.
        sizes = tuple(sorted(size for size in self.forest.size_groups if self.factor.L_by_size[size].shape[1] == size))
        wp.launch(
            packed_rows.pack_kernel(sizes),
            dim=self.forest.world_count * self.dense_max_constraints,
            inputs=[
                *(self.factor.L_by_size[size] for size in sizes),
                self.row_sides,
                self.forest.articulation_response_dof_count,
                self.forest.art_group_idx,
                self.forest.articulation_world_dof_offset,
                self.forest.articulation_dof_start,
                self.constraint_count,
                self.mf_count_or_dummy(),
                skip,
                self.local_solve_owner,
                skip_local,
                int(self.row_packets is not None or self.keyed_rows is not None),
                self.row_cfm,
                int(self.packed_dense),
                write_diag,
                v_hat,
            ],
            outputs=[self.packed_J, self.packed_Z, self.diag, self.row_incident],
            device=self.model.device,
        )

    def diagonals(self, v_hat: wp.array) -> int:
        """The dense rows' diagonals of the matrix-free route, by their owner; what the packed rows' build owes
        (:meth:`pack_rows`, :func:`.packed_rows.pack_kernel`'s ``write_diag``): 0 nothing, 1 ``|Z|^2``, 2
        ``|Z|^2 + cfm``."""
        if self.packed_response:
            # The packed rows' build stores |Z|^2 + cfm in the general worlds; local row packets keep their worlds'
            # private seeds.
            if self.row_packets is None:
                self.diag.zero_()
            return 2
        if self.sparse_diagonal_contact_solve:
            # The sparse-diagonal owner whitens its rows' dense parts in place and stores their diagonals with cfm:
            # every row, or the compact boundary's limit prefix (its producer whitens its contact rows as it writes).
            size = self.sparse_diagonal_dense_size
            rows = 12 if self.compact_contact_boundary else self.dense_max_constraints
            wp.launch(
                response_builds_rows.SPARSE_PAIR.whiten(),
                dim=self.forest.world_count * rows,
                inputs=[
                    self.factor.L_by_size[size],
                    self.sparse_diagonal_dense_groups,
                    self.sparse_diagonal_dense_offsets,
                    self.forest.world_dof_indices,
                    self.constraint_count,
                    self.dense_phase_bounds,
                    int(self.compact_contact_boundary),
                    rows,
                    self.dense_max_constraints,
                    size,
                    self.sparse_diagonal_row_dof,
                    self.sparse_diagonal_row_z,
                    self.row_cfm,
                    self.factor.diagonal_inverse_mass,
                    v_hat,
                ],
                outputs=[self.J_by_size[size], self.diag, self.row_incident],
                device=self.model.device,
            )
            return 0
        if self.sparse_mass_matrix_size is not None:
            # Sparse producers write the diagonal; unfused controls add CFM separately.
            return 0
        self.diag.zero_()
        if self.hinv_jt_diag_sizes:
            for size in self.forest.size_groups:
                if size in self.hinv_jt_diag_sizes:
                    self._accumulate_response_diagonal(size)
                else:
                    self._diagonal_from_JY(size)
        elif self.hinv_jt_writes_world:
            # The packed rows' build stores the diagonal (pack_rows); the CFM pass adds CFM after.
            return 1
        else:
            for size in self.forest.size_groups:
                self._diagonal_from_JY(size)
        if self.local_internal_fast_path and not self.hinv_jt_writes_world:
            wp.launch(
                clear_local_solve_diag,
                dim=self.forest.world_count * self.dense_max_constraints,
                inputs=[self.constraint_count, self.local_solve_owner, self.dense_max_constraints],
                outputs=[self.diag],
                device=self.model.device,
            )
        return 0

    def _diagonal_from_JY(self, size: int):
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
                self.mf_count_or_dummy(),
                int(self.ink_skip_rows()),
            ],
            outputs=[self.diag],
            device=self.model.device,
        )

    def _accumulate_response_diagonal(self, size: int):
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
                self.mf_count_or_dummy(),
                int(self.ink_skip_rows()),
            ],
            outputs=[self.diag],
            device=self.model.device,
        )

    def add_cfm(self):
        if self.packed_response:
            return  # The packed rows' build adds CFM with the diagonal.
        if self.sparse_prepare_contacts:
            return  # Contact and limit producers add CFM exactly once.
        if self.sparse_diagonal_contact_solve:
            return  # The sparse-diagonal owner adds its rows' CFM with their diagonals.
        model = self.model
        wp.launch(
            response_builds_dense.cfm_worlds_kernel(),
            dim=self.forest.world_count,
            inputs=[self.constraint_count, self.row_cfm, self.mf_count_or_dummy(), int(self.ink_skip_rows())],
            outputs=[self.diag],
            device=model.device,
        )

    def whiten_free_bodies(self, v_hat: wp.array, dt: float):
        """MF PGS setup: whiten the free-body rows through each body's held factor, their diagonals and RHS."""
        model = self.model
        self.mf_eff_mass_inv.zero_()
        wp.launch(
            response_builds_rows.free_body_rows_kernel,
            dim=self.forest.world_count * self.mf_max_constraints,
            inputs=[
                bind_mf_rows(self),
                self.mf_constraint_count,
                self.factor.view(),
                int(self.forest.has_prescribed_response),
                v_hat,
                self.laws.rigid_body_max_depenetration_velocity,
                self.options.pgs_cfm,
                self.options.pgs_beta,
                self.laws.contact_w,
                dt,
                self.options.contact_speculative_scale,
                self.laws.effective_restitution_velocity_threshold,
                self.mf_max_constraints,
                int(self.laws.set.weighted),
                int(self.options.contact_compliance),
            ],
            outputs=[self.mf_eff_mass_inv, self.mf_row_w],
            device=model.device,
        )

    def free_body_coordinates(self):
        """Each free-body row's world-relative coordinates, for the sweeps."""
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

    def pack_free_body_meta(self, mf_rhs: wp.array) -> None:
        """Pack matrix-free row metadata for the GS kernels, or skip when no such row can exist."""
        if not self.forest.has_free_rigid_bodies:
            # Only free rigid bodies produce matrix-free rows, and the model plan fixes
            # them at construction: mf_constraint_count stays zero on the one-slot
            # dummy buffers, so the pack would write nothing, and no pack kernel is built.
            return
        wp.launch_tiled(
            self.pack_mf_meta_kernel,
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

    def prepare_impulses(self):
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

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The placement: where each world's constraint rows live, how many fit, and what happens to those that do not
(BACKBONE §2).

:class:`Placement` owns what the rows are stored in across a step:

- the dense rows and their records, the free-body rows, the sparse placements' rows and the packed rows (Z by
  support);
- the per-contact buffers of the contact pool (each contact's world, slot and path) and the carry's row tables;
- the capacities, the slot counters, the row-family bounds and the sorted placement of contact rows;
- the status: the per-world overflow latch, the sticky flags, dropped rows, warnings and watermarks.

A step calls :meth:`Placement.begin` (clear the allocation state), then the row pass writes into its slots
(:meth:`Placement.place_sorted_contact_rows`, :meth:`Placement.latch_status`, :meth:`Placement.warn_overflow`),
:meth:`Placement.diagonals` and :meth:`Placement.prepare_impulses`; the rows' writers store their records.

It also holds which placement the class takes: the native sparse size, the sparse-diagonal pair and the row producers.
The row pass and the sweeps read its fields.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.contact_pool import ContactField, model_contact_pool
from ...sim import Contacts
from ...sim.contacts import CAPACITY_BLOCKS_PER_SM, contact_capacity_blocks
from ...sim.enums import JointType
from . import host_index, packed_rows
from .field_group import Field, FieldGroup, SolverArrays, fields_of
from .row_kernels import RIGID_SPEED_ROWS
from .row_laws import LINK_ROWS
from .row_metadata import record_layout
from .row_records import DENSE_ROWS, ROW_RECORDS, RowSupport
from .row_slots import place_world_contact_runs, record_world_contact_runs
from .warm_start import ContactCarry, prepare_world_impulses

ROW_SLOT_UNBOUNDED = 2**31 - 1
"""First rejected slot of a row family that rejected none this step."""

SIZE_JACOBIAN = FieldGroup(Field("J", wp.float32, ("response_art", "response_row", "response_size")))
"""One size group's Jacobian. The matrix-free sweeps' producers write the packed rows instead (:mod:`.packed_rows`):
their Jacobian is a (1, 1, 1) placeholder."""

DENSE_ROW_WEIGHTS = FieldGroup(Field("row_w", wp.float32, ("world", "row"), fill=1.0, owner="placement"))
"""Per-row weights of the regularized row law: the relative option's (speculative and restitution rows keep the rigid
law) or a material contact's ``d / (d + eps)``."""

DENSE_ROW_EPS = FieldGroup(Field("row_eps", wp.float32, ("world", "row"), owner="placement"))
"""Per-row compliance of material contact normals (``contact_compliance``); zero elsewhere."""

FUSED_DIAGONAL_LIMITS = FieldGroup(
    Field("fused_diagonal_limit_active", wp.int32, ("world", "world_dof"), owner="placement"),
    Field("fused_diagonal_limit_lower_rhs", wp.float32, ("world", "world_dof"), owner="placement"),
    Field("fused_diagonal_limit_upper_rhs", wp.float32, ("world", "world_dof"), owner="placement"),
    Field("fused_diagonal_limit_lower_lambda", wp.float32, ("world", "world_dof"), owner="placement"),
    Field("fused_diagonal_limit_upper_lambda", wp.float32, ("world", "world_dof"), owner="placement"),
)
"""Position limits fused into the diagonal-mass sweep, outside the dense row system."""

ROW_CLASSES = 2
"""The row store's classes: 0 the dense rows, 1 the free bodies' (from the store's body base)."""

ROW_STATUS = FieldGroup(
    Field("row_count", wp.int32, (ROW_CLASSES, "world"), owner="placement"),
    Field("row_slot_counter", wp.int32, (ROW_CLASSES, "world"), owner="placement"),
    Field("row_first_rejected", wp.int32, (ROW_CLASSES, "world"), fill=ROW_SLOT_UNBOUNDED, owner="placement"),
)
"""Each row class's status by class and world: its rows, its reservation counter and its first rejected slot (a
reservation that does not fit rejects its contact and latches the world). A class's row is its own array: the dense
rows' ``constraint_count``, ``slot_counter`` and ``dense_first_rejected_slot``, the free bodies' ``mf_*``
(:meth:`.placement.Placement.allocate_world_rows`)."""

WORLD_ROW_STATE = FieldGroup(
    # Per-world dense row-family boundaries, snapshotted from slot_counter each step. The dense slot layout is
    # [mimic][joint-limit][joint-vel-limit][contact/friction], so [world, 0] ends the mimic and position-limit
    # prefix, [world, 1] the velocity-limit segment, and contacts/friction occupy [bounds[world, 1], m_dense).
    Field("dense_phase_bounds", wp.int32, ("world", 2), owner="placement"),
    # First matrix-free slot past the contact/friction rows (start of the velocity-limit tail).
    Field("mf_contact_rows_end", wp.int32, ("world",), owner="placement"),
    # The line's first speed row, past every contact row: a free body's speed rows are the line's last rows where
    # they are rows of the line (step_plan.Capacities.speed_rows), so the rows [speed_first, m_dense) are speed rows.
    Field("speed_first", wp.int32, ("world",), owner="placement"),
)
"""Per-world row-family boundaries: the dense rows' phases, the free bodies' contact rows' end and the line's speed
rows' start."""

CONTACT_RUNS = FieldGroup(
    Field("contact_run_first", wp.int32, ("world", "run_block"), fill=2**31 - 1, owner="placement"),
    Field("contact_run_last", wp.int32, ("world", "run_block"), fill=-1, owner="placement"),
    Field("contact_run_rows", wp.vec2i, ("world", "run_block"), owner="placement"),
)
"""Contact runs of each world and shape block under sorted contacts (``record_world_contact_runs``)."""

ROW_WATERMARKS = FieldGroup(
    *(
        Field(name, wp.int32, (1,), owner="placement")
        for name in (
            "row_watermark_dense",
            "row_watermark_mf",
            "row_watermark_contact",
            "row_watermark_resolved",
            "row_watermark_dense_raw",
            "row_watermark_mf_raw",
            "row_dropped_dense_high_water",
            "row_dropped_mf_high_water",
            "row_overflow_dense_excess",
            "row_overflow_mf_excess",
            "row_overflow_dense_world_steps",
            "row_overflow_mf_world_steps",
        )
    )
)
"""Whole-run row high-water telemetry for explicit readback (``Diagnostics.row_watermark``)."""

SPARSE_DIAGONAL_ROWS = FieldGroup(
    Field("sparse_diagonal_row_dof", wp.int32, ("world", "row", 2), fill=-1, owner="placement"),
    Field("sparse_diagonal_row_z", wp.float32, ("world", "row", 2), owner="placement"),
)
"""Each sparse-diagonal row's two response coordinates and their whitened entries ``J / sqrt(m)``."""

SPARSE_CONTACT_LEVELS = FieldGroup(
    Field("sparse_contact_level_count", wp.int32, ("world",), owner="placement"),
    Field("sparse_contact_level_offsets", wp.int32, ("world", "level"), fill=None, owner="placement"),
    Field("sparse_contact_level_units", wp.int32, ("world", "unit"), fill=None, owner="placement"),
)
"""The scalar contacts' levels per world (:func:`.sweep_levels._sparse_contact_levels_kernel`): each level's first
place, and the units' head rows level by level."""

SPARSE_CONTACT_SERIAL = FieldGroup(
    Field("sparse_contact_serial_count", wp.int32, ("world",), owner="placement"),
    Field("sparse_contact_serial_normals", wp.int32, ("world", "serial"), fill=None, owner="placement"),
)
"""Contacts solved serially after the speculative batches, by normal row."""

CAPACITY_STATUS = FieldGroup(
    Field("constraint_capacity_status", wp.int32, (2,), owner="placement"),
    Field("constraint_overflow", wp.bool, ("world",), owner="placement"),
    Field("row_dropped_all", wp.int32, (2, "status_world"), owner="placement"),
)
"""Sticky capacity flags of the dense and matrix-free rows, the per-world overflow latch and dropped rows."""

OVERFLOW_WARNING = FieldGroup(Field("row_overflow_warning_emitted", wp.int32, (2,), owner="placement"))
"""The one-shot overflow warning latch of each row family."""


__all__ = ["Placement"]


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
def _clear_dense_diagonals(diag: wp.array2d[float]):
    """Zero the dense rows' diagonals, ``dim = (lines, body_base)``: the free bodies' after them stay their writers'."""
    line, row = wp.tid()
    diag[line, row] = 0.0


@wp.kernel
def _clear_dense_row_state(
    first: wp.array[int],
    slot_counter: wp.array2d[int],
    first_rejected: wp.array2d[int],
    count: wp.array2d[int],
    dropped: wp.array2d[int],
    has_free_rigid: int,
    body_base: int,
    impulses: wp.array2d[float],
):
    """Clear each row class's allocation state and row-loss counters (by class and world), and the free bodies' rows
    (their count, and impulses from ``body_base`` of each world's line), in one launch. The dense rows' count stands
    until their producers' finalize: the last step's rows are cleared by it."""
    world, column = wp.tid()
    line = first[world]
    if column == 0:
        for row_class in range(2):
            slot_counter[row_class, world] = 0
            first_rejected[row_class, world] = ROW_SLOT_UNBOUNDED
            dropped[row_class, world] = 0
        count[1, world] = 0
    if has_free_rigid != 0:
        impulses[line, body_base + column] = 0.0


@wp.kernel
def _finalize_constraint_status(
    slot_counter: wp.array2d[int],
    dropped: wp.array2d[int],
    contact_count: wp.array[int],
    reduction_overflow: wp.array[int],
    capacity: wp.vec2i,
    contact_capacity: int,
    overflow: wp.array[wp.bool],
):
    """Latch invalid-world status until reset: a row class (by class and world) that reserved past its ``capacity`` or
    dropped rows, or the contacts' overflow; counts come from the per-class finalize kernels."""
    world = wp.tid()
    invalid = contact_count[0] > contact_capacity or reduction_overflow[0] != 0
    for row_class in range(2):
        if slot_counter[row_class, world] > capacity[row_class] or dropped[row_class, world] > 0:
            invalid = True
    if invalid:
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
    :meth:`allocate_placeholders`, :meth:`allocate_status` and :meth:`place_dense_response`. The native sparse route
    allocates its rows on :attr:`arrays` as it admits them.

    Args:
        spec: The step's spec: the forest and the row capacities.
        model: The solver's model.
        options: The solver's physics options (:class:`.configuration.Options`).
        laws: The row laws.
        factor: The tree factor.
    """

    def __init__(self, spec, model, options, laws, factor):
        self.model = model
        self.options = options
        self.forest = forest = spec.forest
        self.classes = spec.classes
        """The class plan (:class:`.class_plan.ClassPlan`): the case whose rows this placement holds."""
        self.laws = laws
        self.factor = factor
        self.arrays = SolverArrays(self, model.device)
        # The line of the row store where each world's rows sit: the dense class's from row 0, the free bodies' from
        # ``body_base``.
        worlds = np.arange(forest.world_count, dtype=np.int32)
        self.dense_first = wp.array(worlds, dtype=wp.int32, device=model.device)
        # Launches over the contact capacity (the force export, the warm start's carry and gathers, the keyed-row key
        # scatters) grid-stride over about one wave of blocks.
        self.contact_capacity_blocks = contact_capacity_blocks(model.device)
        # The spec's capacities (step_plan.Capacities): a link's rows come on top of the rows the option gives the
        # world's contacts.
        self.dense_max_constraints = spec.capacities.dense_rows
        self.mf_max_constraints = spec.capacities.free_body_rows
        # A free body is a one-node tree: its contacts and speed rows keep a run of their own (the free bodies' rows
        # from ``body_base``) only where a world can take the colored mapping; elsewhere they are rows of the line,
        # the speed rows its last (``speed_rows`` a world), and the free bodies' rows hold none (the spec's
        # capacities, step_plan.Capacities).
        self.free_contact_rows = spec.capacities.free_contact_rows
        self.speed_rows = spec.capacities.speed_rows
        # The row store's line: the dense class's rows, then (with free rigid bodies) the free bodies' from
        # ``body_base``.
        self.body_base = self.dense_max_constraints
        self.row_pitch = spec.capacities.row_pitch
        # All inactive weight arguments alias this one element. Their producer kernels skip stores and their consumer
        # kernels skip reads when regularization is disabled.
        self.contact_row_w_dummy = self.arrays.placeholder(wp.float32, (1, 1), 1.0)
        self.contact_row_eps_dummy = self.arrays.placeholder(wp.float32, (1, 1))
        self.row_eps = self.contact_row_eps_dummy
        self.J_by_size: dict[int, wp.array] = {}
        # The general placement's one writer (walk_rows): each row's unit source, by world line and row.
        self.walks = False
        self.dummy_units = self.arrays.placeholder(wp.int32, (1, 1))
        # No unit sources: the contact allocator writes none (a producer other than the one writer fills the rows).
        self.no_units = wp.zeros((0, 0), dtype=wp.int32, device=model.device)
        self.unit_source = None

    # ---------------------------------------------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------------------------------------------

    def internal_rows_per_world(self, model, at_once: bool = False) -> int:
        """The most mimic, link and limit rows any world can hold, from the topology; ``at_once``: the most it can hold
        in one step, where a coordinate's two position limits are one row unless both sides lie within the activation
        gap at once (``upper - lower <= 2 gap``, :func:`.row_laws.joint_limit_side`)."""
        if not model.articulation_count or not model.joint_count or self.forest.art_to_world is None:
            return 0

        has_position_limit_rows = self.options.enable_joint_limits and model.joint_dof_count > 0
        has_velocity_limit_rows = (
            self.options.enable_joint_velocity_limits
            and model.joint_dof_count > 0
            and not np.isinf(self.options.velocity_limit_activation_fraction)
        )
        has_mimic_rows = self.laws.mimic_count > 0
        if not (has_position_limit_rows or has_velocity_limit_rows or has_mimic_rows or self.laws.link_count):
            return 0

        art_to_world = self.forest.art_to_world.numpy().astype(np.int32, copy=False)
        per_world = np.zeros(max(int(self.forest.world_count), 1), dtype=np.int64)

        if has_mimic_rows and self.laws.mimic_world_np is not None:
            valid_worlds = self.laws.mimic_world_np[self.laws.mimic_valid_np != 0]
            in_range = valid_worlds[(valid_worlds >= 0) & (valid_worlds < per_world.shape[0])]
            np.add.at(per_world, in_range, 1)
        if self.laws.link_count:
            np.add.at(per_world, self.laws.link_world_np[self.forest.cut.enabled != 0], LINK_ROWS)

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
            return int(np.max(per_world)) if per_world.size else 0

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
            fuse_position_limits = np.isin(self.forest.response_dof_count[art], list(self.classes.fused_sizes))
            lower, upper = joint_limit_lower[dof], joint_limit_upper[dof]
            rows += ~fuse_position_limits & np.isfinite(lower)
            rows += ~fuse_position_limits & np.isfinite(upper)
            if at_once:
                both = ~fuse_position_limits & np.isfinite(lower) & np.isfinite(upper)
                gap = 2.0 * self.options.joint_limit_activation_gap
                rows -= both & (np.where(both, upper - lower, 0.0) > gap)
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
        self.contact_pool = model_contact_pool(self.forest.cut.source, device)
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
        # Row builders always receive a zero phi view when patch friction is off.
        contact_fields.append(ContactField("friction_patch_phi", wp.vec2))
        # The warm start's carry by contact identity (:data:`.warm_start.ContactCarry`).
        if self.options.pgs_warmstart:
            contact_fields += [ContactField(field.attr, field.dtype) for field in fields_of(ContactCarry)]
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

    def allocate_size_groups(self) -> None:
        """Each size group's Jacobian: the sparse diagonal's dense part (case K2); a placeholder elsewhere."""
        if not self.model.articulation_count:
            return

        arrays = self.arrays
        for size in self.forest.size_groups:
            n_arts = self.forest.n_arts_by_size[size]
            sparse_response = self.classes.packed is not None or size == self.classes.diagonal_size
            # The one writer walks the matrix-free sweeps' rows into the packed rows: no size group's rows.
            response = 1 if sparse_response or self.classes.mf_sweeps else 0
            sizes = {
                "art": n_arts,
                "size": size,
                "response_art": response or n_arts,
                "response_row": response or self.dense_max_constraints,
                "response_size": response or size,
            }
            self.J_by_size[size] = arrays.allocate(SIZE_JACOBIAN, owner={}, **sizes)["J"]

    def _on_contact_rows(self, rows) -> None:
        """Assign the per-contact buffers at the contact pool's ready rows, on creation and after each growth."""
        rows.publish(self)
        self.max_contacts_alloc = rows.rows
        patches = self.laws.friction_patches
        if patches is not None:
            patches.view.phi = self.friction_patch_phi

    def follow_contacts(self, contacts: Contacts) -> None:
        """Hold at least ``contacts``' rigid rows, and grow with the pool of a pipeline's buffer from now on (staged where
        the model holds articulations, whose rows own a contact pool)."""
        pool = self.contact_pool
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
        """Allocate the row store and each row class's status."""
        arrays = self.arrays
        # Each row class's status (its rows, reservation counter and first rejected slot), one array a class; a model
        # without worlds keeps one world's (a class's row is a view, which an empty axis cannot give).
        arrays.allocate(ROW_STATUS, world=max(self.forest.world_count, 1))
        self.constraint_count, self.mf_constraint_count = self.row_count[0], self.row_count[1]
        self.slot_counter, self.mf_slot_counter = self.row_slot_counter[0], self.row_slot_counter[1]
        self.dense_first_rejected_slot = self.row_first_rejected[0]
        self.mf_first_rejected_slot = self.row_first_rejected[1]
        if self.forest.world_count == 0:
            return

        # The widest world's DOFs exist where a world response mapping was built (the routes that read them).
        world_dof = self.forest.max_world_dofs
        sizes = {"world": self.forest.world_count, "row": self.row_pitch, "world_dof": world_dof}
        arrays.allocate(DENSE_ROWS, **sizes)
        fused = bool(self.classes.fused_sizes)
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
        diagonal = self.classes.diagonal
        if diagonal is not None:
            self.arrays.allocate(SPARSE_DIAGONAL_ROWS, world=self.forest.world_count, row=self.dense_max_constraints)
        else:
            self.arrays.allocate(SPARSE_DIAGONAL_ROWS, world=1, row=1)
        if diagonal is not None and diagonal.units:
            # A world's rows hold at most a unit a row, so at most that many levels.
            rows = self.dense_max_constraints
            self.arrays.allocate(SPARSE_CONTACT_LEVELS, world=self.forest.world_count, level=rows + 1, unit=rows)
        else:
            self.arrays.allocate(SPARSE_CONTACT_LEVELS, world=1, level=1, unit=1)
        if diagonal is not None and diagonal.speculative_batches:
            # A unit may be a single normal row: a world lists at most a unit a row.
            self.arrays.allocate(
                SPARSE_CONTACT_SERIAL, world=self.forest.world_count, serial=self.dense_max_constraints
            )
        else:
            self.arrays.allocate(SPARSE_CONTACT_SERIAL, world=1, serial=1)

    def allocate_free_body_rows(self, model):
        """Allocate buffers for matrix-free PGS path for free rigid body contacts.

        CUDA without free rigid bodies still passes the
        ``mf_*`` arrays to kernels (the fused GS solve, the PGS diagnostics),
        but ``mf_constraint_count`` stays zero (``_clear_dense_row_state`` and
        ``finalize_constraint_counts_with_status`` write it only when MF is
        active), so every access sits inside a loop bounded by
        ``m_mf == 0`` and no element is ever read or written. The rows' records
        follow the sweep's plan (:meth:`allocate_row_records`).
        """
        self.row_record, self.writes_records = None, False

        if not self.forest.has_free_rigid_bodies:
            # Only the ``is not None`` checks observe this on the inactive
            # path, where no free rigid body owns a speed row.
            self.rigid_velocity_limit_slot = (
                wp.full((1,), -1, dtype=wp.int32, device=model.device)
                if self.laws.has_rigid_body_velocity_limits and model.body_count > 0
                else None
            )
            return

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
        """The matrix-free sweeps' rows, packed by support (:mod:`.packed_rows`): every row of the world's line, the
        free bodies' included (their two one-node trees of six are two sides like any other), each row as wide as two
        of its world's articulations hold; a placeholder without those sweeps."""
        if self.classes.mf_sweeps:
            sizes, worlds = self.forest.articulation_response_dof_count.numpy(), self.forest.art_to_world.numpy()
            offsets = self.forest.articulation_world_dof_offset.numpy()
            # A row's sides follow articulation indices (row_records.row_sides); they follow the world's coordinates
            # too, so packed entries ascend with them.
            order = np.lexsort((np.arange(len(worlds)), worlds))
            responding = order[sizes[order] > 0]
            same_world = worlds[responding][1:] == worlds[responding][:-1]
            if np.any(same_world & (offsets[responding][1:] < offsets[responding][:-1])):
                raise RuntimeError("Packed rows need each world's coordinates in articulation order")
            self.packed_support = packed_rows.support_width(sizes, worlds, self.forest.world_count)
            # Two articulations hold every coordinate of the widest world: the rows are laid out at their coordinates.
            self.packed_dense = int(self.packed_support == self.forest.max_world_dofs)
            if self.factor.eliminates:
                # A corrected row reaches every tree of its block.
                self.packed_support, self.packed_dense = self.forest.max_world_dofs, packed_rows.WIDE
            if self.classes.tier_rows and not self.packed_dense:
                raise RuntimeError("The parallel tiers read their worlds' packed rows at their coordinates")
            sizes = {"world": self.forest.world_count, "row": self.row_pitch, "support": self.packed_support}
        else:
            self.packed_support, self.packed_dense = 1, 0
            sizes = {"world": 1, "row": 1, "support": 1}
        self.arrays.allocate(packed_rows.PACKED_ROWS, **sizes)

    def place_dense_response(self, *, paired_coordinates: bool) -> None:
        """Where the dense rows' response lives (``packed_response`` and ``packed_response_reason``).

        The matrix-free sweeps, and every route beside them, read the dense rows packed by support
        (:mod:`.packed_rows`): the packed build alone serves the step.
        """
        self.packed_response_reason = None if self.classes.mf_sweeps else "No matrix-free sweep reads packed rows."
        self.packed_response = self.packed_response_reason is None
        if not self.packed_response:
            if paired_coordinates:
                raise RuntimeError(
                    f"The paired factor-coordinate sweep reads packed rows: {self.packed_response_reason}"
                )
            return
        if paired_coordinates and not self.packed_dense:
            raise RuntimeError("The paired factor-coordinate sweep reads packed rows at their world coordinates")

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
        # one-shot flag per row family; the row watermark additionally accumulates
        # whole-run high-water telemetry for explicit readback. Dropped-row
        # accounting is unconditional (see row_dropped_all above).
        if self.options.diagnostics.overflow_warning:
            self.arrays.allocate(OVERFLOW_WARNING)
        else:
            self.arrays.release(OVERFLOW_WARNING)
        # The telemetry words share one array, which constraint_row_watermarks() reads back once.
        self.row_watermark_words = None
        if self.options.diagnostics.row_watermark:
            self.row_watermark_words = self.arrays.allocate_words(ROW_WATERMARKS)
        else:
            self.arrays.release(ROW_WATERMARKS)

    def allocate_row_records(self, first: int, *, skip_rows: int = 0, speed_rows: bool = False) -> None:
        """Allocate the streamed rows' records (:data:`.row_records.ROW_RECORDS`, :func:`.row_metadata.record_layout`)
        from the line's row ``first``; their writers store them in the worlds the general sweep visits, not the tiers'
        (``skip_rows``, with the line's ``speed_rows``)."""
        self.record_first = int(first)
        rows = self.row_pitch - self.record_first
        free_rows = self.mf_max_constraints if self.free_contact_rows else 0
        self.writes_records = bool(rows > 0 and (self.record_first < self.body_base or free_rows > 0))
        self.record_skip_rows, self.record_speed_rows = int(skip_rows), bool(speed_rows)
        self.free_records, words = record_layout(self.record_first, self.body_base, self.row_pitch)
        self.arrays.allocate(ROW_RECORDS, world=self.forest.world_count, record=max(words, 1))

    @property
    def writes_dense_records(self) -> bool:
        """Whether the one writer stores the dense rows' records (:meth:`allocate_row_records`)."""
        return self.writes_records and self.record_first < self.body_base

    def allocate_placeholders(self):
        """The operands of the features that are off."""
        if self.row_record is None:
            # No sweep streams a row (no matrix-free sweep): the records' operand alone.
            self.allocate_row_records(self.row_pitch)
        # Operands of the features that are off (SolverArrays.placeholder).
        placeholder = self.arrays.placeholder
        self.dummy_is_free_rigid = placeholder(wp.int32, (1,))
        self.dummy_mf_impulses = placeholder(wp.float32, (1, 1))
        self.dummy_contact_count = placeholder(wp.int32, (1,))

    # ---------------------------------------------------------------------------------------------------------------
    # Step: the rows' slots, counts and status
    # ---------------------------------------------------------------------------------------------------------------

    def begin(self):
        """Clear the rows' allocation state before the row pass writes them."""
        # One launch clears the world-level allocation state and, with free rigid bodies, the free bodies'
        # counters and impulses (PGS reads their impulses before its first write; the warm start's gather seeds
        # their normals after the row pass). Other row buffers are fully overwritten by their producers.
        mf_active = self.forest.has_free_rigid_bodies
        wp.launch(
            _clear_dense_row_state,
            dim=(self.forest.world_count, self.mf_max_constraints if mf_active else 1),
            inputs=[
                self.dense_first,
                self.row_slot_counter,
                self.row_first_rejected,
                self.row_count,
                self.row_dropped_all,
                1 if mf_active else 0,
                self.body_base,
                self.impulses if mf_active else self.dummy_mf_impulses,
            ],
            device=self.model.device,
        )

    def place_sorted_contact_rows(
        self, contacts, contact_build_threads, mf_slot_counter, mf_first_rejected_slot, units: wp.array
    ):
        """Give each world's contact rows the order of sorted contacts, whatever order the allocator reserved them in.

        Runs only when the contacts come from a deterministic collision pipeline
        (``Contacts.rigid_contacts_pair_sorted``), whose determinism Newton promises when the
        solver is deterministic as well; the default pipeline keeps the allocator's atomic
        reservation, whose order on CUDA follows the warps. A captured step keeps the order of
        its capture. Dense rows are placed unless the keyed rows place their own; matrix-free rows are placed with free
        rigid bodies. ``units``: the one writer's unit sources, which follow the placed slots (empty where the writer
        does not take the contact rows).
        """
        order_dense = int(self.keyed_rows is None)
        order_mf = int(self.forest.has_free_rigid_bodies)
        if not (order_dense or order_mf):
            return
        device = self.model.device
        runs = (self.contact_run_first, self.contact_run_last, self.contact_run_rows)
        blocks = (self.contact_run_lead, self.contact_run_trail_start)
        # Each contact's family: the free bodies' contacts follow the line's other contact rows, in their own run or,
        # where they are rows of the line (free_contact_rows false), after its others from the one counter.
        is_free_rigid = self.forest.is_free_rigid if self.forest.is_free_rigid is not None else self.dummy_is_free_rigid
        free_line = int(self.forest.has_free_rigid_bodies and not self.free_contact_rows)
        families = (
            self.contact_art_a,
            self.contact_art_b,
            is_free_rigid,
            self.forest.articulation_response_dof_count,
            free_line,
        )
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
                *families,
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
                *families,
                self.slot_counter,
                self.dense_first_rejected_slot,
                mf_slot_counter,
                mf_first_rejected_slot,
                self.body_base,
                *blocks,
                lanes,
                self.dense_first,
            ],
            outputs=[*runs, self.contact_slot, units],
            device=device,
        )

    def latch_status(self, contacts: Contacts | None) -> None:
        """Latch current per-world capacity failures for every row owner."""
        # The contact capacity (argument 6) follows the ready rows in a capture whose launches bind.
        self.contact_pool.launch(
            _finalize_constraint_status,
            self.forest.world_count,
            inputs=[
                self.row_slot_counter,
                self.row_dropped_all,
                contacts.rigid_contact_count if contacts is not None else self.dummy_contact_count,
                contacts._reduction_overflow if contacts is not None else self.dummy_contact_count,
                wp.vec2i(self.dense_max_constraints, self.mf_max_constraints),
                contacts._rigid_contact_rows if contacts is not None else 0,
                self.constraint_overflow,
            ],
            row_args=(6,) if contacts is not None else (),
        )

    def warn_overflow(self):
        """One device-side warning per overflowing row family (staged under ``Diagnostics.overflow_warning``)."""
        wp.launch(
            _warn_constraint_row_overflow,
            dim=self.forest.world_count,
            inputs=[
                self.slot_counter,
                self.row_dropped_dense,
                self.dense_max_constraints,
                self.mf_slot_counter,
                self.row_dropped_mf,
                self.mf_max_constraints,
                int(self.forest.has_free_rigid_bodies),
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

    def record(self) -> dict:
        """The block cap of the launches over the contact capacity (``routes.contact_consumers``, 0: uncapped)."""
        blocks = int(self.contact_capacity_blocks)
        reason = (
            "Launches over the contact capacity (force export, warm-start snapshot and gathers, keyed-row key"
            f" scatters) grid-stride over {CAPACITY_BLOCKS_PER_SM} blocks of 256 threads per SM, about one wave."
            if blocks
            else "Uncapped: a CPU launch, or a kernel compiled without the grid-stride loop, covers every slot."
        )
        return {"contact_consumers": {"max_blocks": blocks, "reason": reason}}

    def record_watermarks(self, contacts):
        """The row high-water telemetry (``Diagnostics.row_watermark``, staged only under it; behavior-neutral): at the
        end of the step, after the row pass has finalized the per-world counts, so the atomic-max runs inside the
        captured graph every step and catches mid-run peaks."""
        device = self.model.device
        # Each row class's rows (row_count), reservations (row_slot_counter) and dropped rows: the free bodies' exist
        # only with free bodies.
        for family, name in enumerate(("dense", "mf")[: 1 + int(self.forest.has_free_rigid_bodies)]):
            wp.launch(
                _accumulate_row_watermark,
                dim=self.row_count.shape[1],
                inputs=[self.row_count[family], getattr(self, f"row_watermark_{name}")],
                device=device,
            )
            wp.launch(
                _accumulate_row_capacity_telemetry,
                dim=self.row_slot_counter.shape[1],
                inputs=[
                    self.row_slot_counter[family],
                    self.row_dropped_all[family],
                    (self.dense_max_constraints, self.mf_max_constraints)[family],
                    getattr(self, f"row_watermark_{name}_raw"),
                    getattr(self, f"row_dropped_{name}_high_water"),
                    getattr(self, f"row_overflow_{name}_excess"),
                    getattr(self, f"row_overflow_{name}_world_steps"),
                ],
                device=device,
            )
        if contacts is not None and getattr(contacts, "rigid_contact_count", None) is not None:
            contact_counts = contacts.rigid_contact_count
            wp.launch(
                _accumulate_row_watermark,
                dim=contact_counts.shape[0],
                inputs=[contact_counts, self.row_watermark_contact],
                device=device,
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
        # The predictor's border (tree_kinetic_state.border_solve): bit 0, a tree slipped past its frame's capacity; bit 2,
        # a tree the walk does not hold took a decided step.
        result["slip_frames"] = bool(int(self.factor.walk_release_status.numpy()[0]) & 5)
        if clear:
            self.constraint_capacity_status.zero_()
            self.factor.walk_release_status.zero_()
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
        result["resolved_high_water"] = int(words.get("row_watermark_resolved", 0))
        return result

    # ---------------------------------------------------------------------------------------------------------------
    # Step: where the rows' Jacobians live
    # ---------------------------------------------------------------------------------------------------------------

    def row_support(self) -> RowSupport:
        """Where the dense rows' entries sit and the seeds the one writer walks (:class:`.row_records.RowSupport`): the
        rows' sides, the articulations' coordinates and first world coordinates, the layout, and the unit sources.
        Without the matrix-free sweeps there are no packed rows: the articulations' offsets are a one-entry
        placeholder. A class that walks no rows holds none (:func:`.step_plan._stages`): its unit sources are a
        placeholder."""
        support = RowSupport()
        support.sides = self.row_sides
        support.dofs = self.forest.articulation_response_dof_count
        support.offset = (
            self.forest.articulation_world_dof_offset if self.classes.mf_sweeps else self.dummy_is_free_rigid
        )
        support.dense = int(self.packed_dense)
        support.first = self.dense_first
        support.units = self.unit_source if self.walks else self.dummy_units
        return support

    def ink_skip_rows(self) -> int:
        """The dense rows of the worlds the tiers own, whose response they form in-kernel (0 without tiers)."""
        return self.classes.tier_rows

    def inkernel_response_inputs(self) -> dict:
        """The parallel tiers' operands beside the walk: the worlds' trees (``ink_meta``) and the rows' law's cfm."""
        return {"ink_meta": self.ink_meta, "cfm": self.options.pgs_cfm}

    # ---------------------------------------------------------------------------------------------------------------
    # Step: the whitening and the diagonals
    # ---------------------------------------------------------------------------------------------------------------

    def hold_units(self) -> None:
        """The one writer's unit sources (:mod:`.walk_rows`): an int per row, by world line and row."""
        rows = self.dense_max_constraints
        self.unit_source = wp.full((self.forest.world_count, rows), 0, dtype=wp.int32, device=self.model.device)

    def diagonals(self) -> None:
        """The dense rows' diagonals, by the placement that holds them (:attr:`.step_plan.StepPlan.response`): the packed
        rows' start from zero for the one writer's ``|Z|^2 + cfm``; the sparse diagonal's and native sparse factors'
        writers store theirs with its CFM. The free bodies' rows, from ``body_base``, keep theirs: their writers stored
        them in the row pass (:mod:`.walk_rows`). The stage list runs it where the packed rows hold the response
        (:attr:`clears_diagonals`)."""
        dim = (self.diag.shape[0], self.body_base)
        wp.launch(_clear_dense_diagonals, dim=dim, outputs=[self.diag], device=self.model.device)

    @property
    def clears_diagonals(self) -> bool:
        """Whether the dense rows' diagonals start from zero each step (:meth:`diagonals`): the packed rows hold the
        response."""
        return bool(self.packed_response)

    def prepare_impulses(self):
        warmstart_flag = 1 if self.options.pgs_warmstart else 0
        wp.launch(
            prepare_world_impulses,
            dim=self.forest.world_count,
            inputs=[
                self.dense_first,
                self.constraint_count,
                self.dense_max_constraints,
                warmstart_flag,
            ],
            outputs=[self.impulses],
            device=self.model.device,
        )

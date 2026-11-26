# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The placement: where each world's constraint rows live, how many fit, and what happens to those that do not
(BACKBONE §2).

:class:`Placement` owns what the rows are stored in across a step:

- the dense rows and their records, the free-body rows, the sparse placements' rows and the packed rows (Z by
  support);
- the per-contact buffers of the contact pool (each contact's world, slot and path) and the carry's row tables;
- the capacities, the slot counters, the row-family bounds and each world's span of contacts;
- the status: the per-world overflow latch, the sticky flags, dropped rows, warnings and watermarks.

A step calls :meth:`Placement.begin` (clear the allocation state), then the row pass writes into its slots
(:meth:`Placement.close_rows`),
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
from . import field_group, host_index, packed_rows
from .device_preludes import reserved_count
from .factor.walk import walk_tables
from .field_group import Field, FieldGroup, SolverArrays, fields_of
from .forest_plan import RowLaw
from .friction_patches import frame_kept
from .route_table import first_failed
from .row_laws import LINK_ROWS
from .row_metadata import record_layout, record_words, split_records
from .row_records import DENSE_ROWS, ROW_RECORDS, RowSupport
from .row_slots import ROW_SLOT_UNBOUNDED, RowGroups, RowOrder, row_groups
from .warm_start import ContactCarry, prepare_world_impulses, row_lanes

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})


SIZE_JACOBIAN = FieldGroup(Field("J", wp.float32, ("response_art", "response_row", "response_size")))
"""One size group's Jacobian. The matrix-free sweeps' producers write the packed rows instead (:mod:`.packed_rows`):
their Jacobian is a (1, 1, 1) placeholder."""

DENSE_ROW_WEIGHTS = FieldGroup(Field("row_w", wp.float32, ("line", "row"), fill=1.0, owner="placement"))
"""Per-row weights of the regularized row law: the relative option's (speculative and restitution rows keep the rigid
law) or a material contact's ``d / (d + eps)``."""

DENSE_ROW_EPS = FieldGroup(Field("row_eps", wp.float32, ("line", "row"), owner="placement"))
"""Per-row compliance of material contact normals (:attr:`.route_table.Laws.material`): positive [1/kg] from a
contact's stiffness, negative ``-rho`` (relative, of the row's diagonal; its weight ``1 / (1 + rho)``) from authored
shape materials; zero elsewhere."""

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
    # Per-world dense row-family boundary, snapshotted from slot_counter each step. The dense slot layout is
    # [mimic][link][joint-limit][contact/friction], so a world's topology rows are [0, topology_rows) and its
    # contacts and friction [topology_rows, m_dense).
    Field("topology_rows", wp.int32, ("world",), owner="placement"),
    # The free bodies' run's slot past its contact/friction rows.
    Field("mf_contact_rows_end", wp.int32, ("world",), owner="placement"),
)
"""Per-world row-family boundaries: the dense rows' topology rows and the free bodies' contact rows' end."""

WORLD_ACTS = FieldGroup(Field("world_acts", wp.int32, ("world",), owner="placement"))
"""Whether a row of the world can act at the predictor (:func:`.row_laws.separates`): set by its allocators where every
row is separable (:meth:`Placement.hold_acts`); the claims, the one writer and the patch finish leave a world without it
at its exact solution, zero impulse (P4)."""

WORLD_CONTACTS = FieldGroup(
    Field("world_contact_first", wp.int32, ("world",), fill=2**31 - 1, owner="placement"),
    Field("world_contact_last", wp.int32, ("world",), fill=-1, owner="placement"),
    Field("contact_worlds", wp.int32, ("world",), owner="placement"),
    Field("contact_worlds_listed", wp.int32, (2,), owner="placement"),
)
"""The frame positions each world's contacts span (``admit_world_contacts``), and the worlds whose span starts (listed
in ``contact_worlds``, counted in ``contact_worlds_listed[0]``), from which their rows are reserved in the row order
(``reserve_world_rows``), which resets the spans and the list."""

CONTACT_PASS = FieldGroup(
    Field("contact_rows_kept", wp.int32, (ROW_CLASSES, "world"), owner="placement"),
    Field("world_held", wp.int32, ("world",), owner="placement"),
    Field("pass_words", wp.int32, (2,), owner="placement"),
)
"""The contact set's work once a collision pass (:func:`_bound_topology`): each world's contact rows by class that its
last reservation added (``contact_rows_kept``; -1 in the dense class where it rejected a row), whether the step keeps
the world's contact rows (``world_held``), and two words: whether the step's contact frame is the last step's
(``pass_words[0]``: no collision pass since) and the reservation's launch extent (``pass_words[1]``: the contact count
where a world reserves, else zero, which skips the bound launch)."""

ROW_WATERMARKS = FieldGroup(
    *(
        Field(name, wp.int32, (1,), owner="placement")
        for name in (
            "row_watermark_dense",
            "row_watermark_mf",
            "row_watermark_contact",
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
    Field("sparse_diagonal_row_dof", wp.int32, ("line", "row", 2), fill=-1, owner="placement"),
    Field("sparse_diagonal_row_z", wp.float32, ("line", "row", 2), owner="placement"),
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
    Field("row_demand", wp.int32, (2, "world"), owner="placement"),
)
"""Sticky capacity flags of the dense and matrix-free rows, the per-world overflow latch and dropped rows, and each
world's largest reservation past a class's capacity since its reset (``row_demand``: the rows it asked for, 0 while it
fits), which the capacity check reports."""

ROW_DROP_WORDS = 3
"""The host's record of the row passes that dropped rows since the last reset (:meth:`Placement.fail_on_dropped_rows`):
the worlds' passes that reserved past a class's capacity, then the largest reservation of the dense rows and of the free
bodies' run."""

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
def _bound_topology(
    slot_counter: wp.array2d[int],
    cleared: int,
    count: wp.array[wp.int32],
    generation: wp.array[wp.int32],
    stream: wp.int64,
    built_frame: wp.array[wp.int64],
    home: wp.array[int],
    # outputs
    topology_rows: wp.array[int],
    diag: wp.array2d[float],
    contact_rows_kept: wp.array2d[int],
    world_held: wp.array[int],
    pass_words: wp.array[int],
    world_run: wp.array[wp.vec2i],
    arena_cursor: wp.array[int],
):
    """Record where each world's topology rows (mimics, links, joint limits) end and its contacts start, decide whether
    the step keeps the world's contact rows, and zero its first ``cleared`` dense rows' diagonals: the free bodies'
    after them stay their writers'.

    Launched right after the joint-limit allocation, so the sweeps' and writers' contact phases loop only their own
    contiguous row range. The raw counter is stored; consumers clamp against the per-world constraint count.

    The contact set changes only at a collision pass, so its admission, the rows' reservation and their units are a
    pass's (:data:`CONTACT_PASS`): where the frame is the last step's (:func:`.friction_patches.frame_kept`, read
    before this step's patch build), a world whose topology rows end where they ended last step and whose last
    reservation fit keeps its contacts' slots, and its counters advance by the rows that reservation added. Any other
    world reserves (``reserve_world_rows``), which the reservation's extent follows.

    One linear launch of ``worlds * max(cleared, 1)`` threads: thread ``t`` below the worlds decides world ``t`` (whole
    warps of worlds), and every thread clears row ``t % cleared`` of world ``t // cleared``'s home line. On a new frame
    (every world reserves) the arena's runs start over; on a kept frame a world that reserves anew may keep its run
    (:func:`.row_slots.reserve_world_rows_kernel`).
    """
    t = wp.tid()
    worlds = topology_rows.shape[0]
    if t < worlds:
        world = t
        live = wp.max(count[0], 0)
        kept = frame_kept(generation, live, stream, built_frame)
        dense = slot_counter[0, world]
        held = int(0)
        if kept != 0 and topology_rows[world] == dense and contact_rows_kept[0, world] >= 0:
            held = 1
        world_held[world] = held
        topology_rows[world] = dense
        if held != 0:
            slot_counter[0, world] = dense + contact_rows_kept[0, world]
            slot_counter[1, world] = slot_counter[1, world] + contact_rows_kept[1, world]
        else:
            # The reservation writes a listed world's rows (and claims its run anew); a world without contacts adds
            # none.
            contact_rows_kept[0, world] = 0
            contact_rows_kept[1, world] = 0
        # A new frame: every world reserves, so the arena's runs start over.
        if kept == 0:
            world_run[world] = wp.vec2i(-1, 0)
        # The words, once: on a new frame every world reserves; on a kept frame only a world that does not keep its
        # slots (rare) stores the extent.
        if world == 0:
            pass_words[0] = kept
            if kept == 0:
                arena_cursor[0] = 0
        if (kept == 0 and world == 0) or (kept != 0 and held == 0):
            pass_words[1] = live
    if cleared > 0:
        if t < worlds * cleared:
            diag[home[t // cleared], t % cleared] = 0.0


@wp.func_native("""
    const unsigned char* const from = reinterpret_cast<const unsigned char*>(source);
    unsigned char* const to = reinterpret_cast<unsigned char*>(target);
    for (long long b = 0; b < bytes; ++b) to[b] = from == nullptr ? 0 : from[b];
""")
def _copy_bytes(target: wp.uint64, source: wp.uint64, bytes: wp.int64): ...


@wp.kernel
def _move_to_runs(
    home: wp.array[int],
    run_of: wp.array[wp.vec2i],
    topology_rows: wp.array[int],
    slot_counter: wp.array2d[int],
    capacity: int,
    fields: wp.array[wp.uint64],
    row_bytes: wp.array[wp.int64],
    line_bytes: wp.array[wp.int64],
    diag: int,
    # outputs
    first: wp.array[int],
):
    """Take each world whose rows run in the arena (``run_of``, the reservation's claim, or kept while it holds its
    slots) to its run: its topology rows, which its producers wrote at its home line this step, copied there in every
    field laid out by line and row (``fields``: their addresses, their rows' bytes and their lines' bytes), and its
    contact rows' diagonals zeroed for the one writer's sums (field ``diag``). A thread a world and field."""
    world, field = wp.tid()
    run = run_of[world][0]
    if run < 0:
        return
    if field == 0:
        first[world] = run
    rows = topology_rows[world]
    size = row_bytes[field]
    at = fields[field] + wp.uint64(run) * wp.uint64(line_bytes[field])
    _copy_bytes(at, fields[field] + wp.uint64(home[world]) * wp.uint64(line_bytes[field]), wp.int64(rows) * size)
    if field == diag:
        count = wp.min(slot_counter[0, world], capacity)
        if count > rows:
            _copy_bytes(at + wp.uint64(wp.int64(rows) * size), wp.uint64(0), wp.int64(count - rows) * size)


@wp.kernel
def _clear_dense_row_state(
    slot_counter: wp.array2d[int],
    first_rejected: wp.array2d[int],
    count: wp.array2d[int],
    dropped: wp.array2d[int],
    home: wp.array[int],
    acts: wp.array[int],
    pass_words: wp.array[int],
    first: wp.array[int],
):
    """Clear each row class's allocation state and row-loss counters (by class and world), the free bodies' row count
    and the world's acts word, in one launch, and take each world to its home line, where the step's topology rows are
    written (a world whose rows run in the arena moves to its run once they are, :func:`_move_to_runs`). The dense
    rows' count stands until their producers' finalize: the last step's rows are cleared by it. The rows' impulses
    start where the counts are final (:func:`.warm_start.prepare_world_impulses`)."""
    world = wp.tid()
    first[world] = home[world]
    for row_class in range(2):
        slot_counter[row_class, world] = 0
        first_rejected[row_class, world] = ROW_SLOT_UNBOUNDED
        dropped[row_class, world] = 0
    count[1, world] = 0
    if acts.shape[0] > 0:
        acts[world] = 0
    if world == 0:
        # No world reserves until _bound_topology finds one.
        pass_words[1] = 0


@wp.func
def _latch_world(
    world: int,
    slot_counter: wp.array2d[int],
    dropped: wp.array2d[int],
    contact_count: wp.array[int],
    reduction_overflow: wp.array[int],
    capacity: wp.vec2i,
    contact_capacity: int,
    overflow: wp.array[wp.bool],
    demand: wp.array2d[int],
    record: wp.array[wp.int32],
):
    """Latch ``world``'s invalid status until reset: a row class (by class and world) that reserved past its
    ``capacity`` or dropped rows, or the contacts' overflow; a class past its capacity also keeps its largest
    reservation (``demand``) and counts in the host's drop record (:data:`ROW_DROP_WORDS`): a world's pass that dropped
    rows (past its capacity, or past its line where the arena held no run), the largest reservation of each class."""
    invalid = contact_count[0] > contact_capacity or reduction_overflow[0] != 0
    dropped_rows = bool(False)
    for row_class in range(2):
        asked = slot_counter[row_class, world]
        if asked > capacity[row_class]:
            demand[row_class, world] = wp.max(demand[row_class, world], asked)
            wp.atomic_max(record, 1 + row_class, wp.int32(asked))
            dropped_rows = True
        if asked > capacity[row_class] or dropped[row_class, world] > 0:
            invalid = True
            dropped_rows = True
    if dropped_rows:
        wp.atomic_add(record, 0, wp.int32(1))
    if invalid:
        overflow[world] = True


@wp.kernel
def _reset_solver_status(
    world_mask: wp.array[wp.bool],
    articulation_world: wp.array[int],
    mass_update_requested: wp.array[int],
    overflow: wp.array[wp.bool],
    demand: wp.array2d[int],
):
    """Clear selected world status (its latch and row demand) and request only its articulation factors."""
    tid = wp.tid()
    if tid < overflow.shape[0]:
        if not world_mask or world_mask[tid] != 0:
            overflow[tid] = False
            demand[0, tid] = 0
            demand[1, tid] = 0
    if tid < mass_update_requested.shape[0]:
        world = articulation_world[tid]
        if not world_mask or world_mask[world] != 0:
            mass_update_requested[tid] = 1


@wp.func
def _warn_world(
    world: int,
    raw_counts: wp.array2d[int],
    dropped_rows: wp.array2d[int],
    capacity: wp.vec2i,
    mf_active: int,
    warning_emitted: wp.array[wp.int32],
):
    """Emit one device-side warning per overflowing FeatherPGS row family (``raw_counts``: each class's reservations,
    ``dropped_rows`` its dropped contact rows)."""
    dense_dropped = dropped_rows[0, world]
    dense_requested = raw_counts[0, world]
    if dense_requested > capacity[0] and wp.atomic_exch(warning_emitted, 0, 1) == 0:
        wp.printf(
            "Warning: FeatherPGS dense constraint-row overflow in world %d: requested %d rows, limit %d; "
            "dropped %d contact/friction rows. Increase dense_max_constraints.\n",
            world,
            dense_requested,
            capacity[0],
            dense_dropped,
        )

    if mf_active != 0:
        mf_dropped = dropped_rows[1, world]
        mf_requested = raw_counts[1, world]
        if mf_requested > capacity[1] and wp.atomic_exch(warning_emitted, 1, 1) == 0:
            wp.printf(
                "Warning: FeatherPGS matrix-free constraint-row overflow in world %d: requested %d rows, "
                "limit %d; dropped %d contact/friction rows. Increase mf_max_constraints.\n",
                world,
                mf_requested,
                capacity[1],
                mf_dropped,
            )


@wp.kernel
def _close_row_pass(
    slot_counter: wp.array2d[int],
    first_rejected: wp.array2d[int],
    capacity: wp.vec2i,
    families: int,
    dropped: wp.array2d[int],
    contact_count: wp.array[int],
    reduction_overflow: wp.array[int],
    contact_capacity: int,
    warn: int,
    mf_active: int,
    warning_emitted: wp.array[wp.int32],
    count: wp.array2d[int],
    capacity_status: wp.array[int],
    overflow: wp.array[wp.bool],
    demand: wp.array2d[int],
    record: wp.array[wp.int32],
):
    """Close each world's row pass once every row producer reserved: the first ``families`` row classes' counts (the
    dense rows', then the free bodies' run), each its accepted prefix with sticky overflow (:func:`.row_slots`'
    ``reserved_count``), the world's invalid-status latch (:func:`_latch_world`) and, with ``warn``, one device-side
    warning per overflowing class (:func:`_warn_world`)."""
    world = wp.tid()
    for family in range(families):
        count[family, world] = reserved_count(
            slot_counter[family], world, capacity[family], first_rejected[family], capacity_status, family
        )
    _latch_world(
        world,
        slot_counter,
        dropped,
        contact_count,
        reduction_overflow,
        capacity,
        contact_capacity,
        overflow,
        demand,
        record,
    )
    if warn != 0:
        _warn_world(world, slot_counter, dropped, capacity, mf_active, warning_emitted)


_WARP = 32
"""The general sweep's lanes over a row's entries (a warp a world), which the rows' support classes are chosen by."""


class Placement:
    """The placement of one solver (see the module docstring); the solver holds it as ``placement``.

    Built in the solver's construction order: the capacities and placeholders here, then
    :meth:`allocate_size_groups`, :meth:`allocate_contact_rows`, :meth:`allocate_world_rows`,
    :meth:`allocate_sparse_diagonal_rows`, :meth:`allocate_packed_rows`,
    :meth:`allocate_placeholders`, :meth:`allocate_status` and :meth:`place_dense_response`. Every class that writes
    rows by the walk but the sparse diagonal holds them in the one store, the packed rows (:attr:`packs_rows`).

    Args:
        spec: The step's spec: the forest and the row capacities.
        model: The solver's model.
        options: The solver's physics options (:func:`.configuration.check_options`).
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
        self.support_pin = spec.pins.row_support
        self.arrays = SolverArrays(self, model.device)
        # The line of the row store where each world's rows sit: the dense class's from row 0, the free bodies' from
        # ``body_base``.
        worlds = np.arange(forest.world_count, dtype=np.int32)
        self.home_first = wp.array(worlds, dtype=wp.int32, device=model.device)
        self.dense_first = wp.array(worlds, dtype=wp.int32, device=model.device)
        # Launches over the contact capacity (the force export, the warm start's carry and gathers) grid-stride over
        # about one wave of blocks.
        self.contact_capacity_blocks = contact_capacity_blocks(model.device)
        # The spec's capacities (step_plan.Capacities): a link's rows come on top of the rows the option gives the
        # world's contacts.
        self.dense_max_constraints = spec.capacities.dense_rows
        self.mf_max_constraints = spec.capacities.free_body_rows
        # A free body is a one-node tree: its contacts keep a run of their own (the free bodies' rows from
        # ``body_base``) only where a world can take the colored mapping; elsewhere they are rows of the line, and the
        # free bodies' rows hold none (the spec's capacities, step_plan.Capacities).
        self.free_contact_rows = spec.capacities.free_contact_rows
        # The streamed rows' records, once the sweep's plan places them (allocate_row_records).
        self.row_record, self.writes_records = None, False
        # The row store's line: the dense class's rows, then (with free rigid bodies) the free bodies' from the line's
        # body base (lay_out_lines), at most the capacities' line.
        self.row_pitch = spec.capacities.row_pitch
        self.lay_out_lines(self.row_pitch, 0)
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
        has_mimic_rows = self.laws.mimic_count > 0
        if not (has_position_limit_rows or has_mimic_rows or self.laws.link_count):
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
        if joint_limit_lower_arr is None or joint_limit_upper_arr is None:
            has_position_limit_rows = False
        if not has_position_limit_rows:
            return int(np.max(per_world)) if per_world.size else 0

        joint_limit_lower = joint_limit_lower_arr.numpy()
        joint_limit_upper = joint_limit_upper_arr.numpy()

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
        fuse_position_limits = np.isin(self.forest.response_dof_count[art], list(self.classes.fused_sizes))
        lower, upper = joint_limit_lower[dof], joint_limit_upper[dof]
        rows += ~fuse_position_limits & np.isfinite(lower)
        rows += ~fuse_position_limits & np.isfinite(upper)
        if at_once:
            both = ~fuse_position_limits & np.isfinite(lower) & np.isfinite(upper)
            gap = 2.0 * self.options.joint_limit_activation_gap
            rows -= both & (np.where(both, upper - lower, 0.0) > gap)
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
                "contact_acts",
            )
        ]
        # The row order's records (row_slots.RowOrder).
        contact_fields += [ContactField(name, wp.vec4i) for name in ("contact_rows_frame", "contact_rows_before")]
        contact_fields.append(ContactField("contact_pair_sides", wp.vec2i))
        contact_fields += [
            ContactField(name, wp.int32)
            for name in (
                "contact_pairs_before",
                "contact_pairs",
                "contact_pair_island",
                "contact_pair_rank",
                "contact_row_pairs",
            )
        ]
        # Row builders always receive a zero phi view when patch friction is off.
        contact_fields.append(ContactField("friction_patch_phi", wp.vec2))
        # The warm start's carry by contact identity (:data:`.warm_start.ContactCarry`).
        if self.options.pgs_warmstart:
            contact_fields += [ContactField(field.attr, field.dtype) for field in fields_of(ContactCarry)]
        self.contact_rows = self.contact_pool.rows(contact_fields, on_rows=self._on_contact_rows)
        # The friction patch state.
        # A contact has a region normal only where neither side is a non-free articulation's link: some body must be
        # free (a free rigid body's, or no articulation's) for any contact to hold one.
        body_to_articulation = self.forest.body_to_articulation
        region_normals = (
            body_to_articulation is None
            or bool(self.forest.has_free_rigid_bodies)
            or bool(np.any(body_to_articulation.numpy() < 0))
        )
        self.laws.allocate(model, self.contact_pool, self.friction_patch_phi, region_normals=region_normals)
        # The row order's islands, by articulation (row_slots.RowOrder): each world's reservation writes its own trees'.
        trees = max(int(self.forest.articulation_response_dof_count.shape[0]), 1)
        # Between reservations each tree is its own root and its island is empty.
        self.island_parent = wp.array(np.arange(trees, dtype=np.int32), dtype=wp.int32, device=device)
        self.island_size = wp.zeros(trees, dtype=wp.int32, device=device)
        self.island_base = wp.zeros(trees, dtype=wp.vec2i, device=device)
        # What a batch of a world's frame order leaves the next in its block of warps (row_slots._frame_turns).
        self.reserve_turns = wp.zeros((max(int(self.forest.world_count), 1), 8), dtype=wp.int32, device=device)
        # No world's dense line by colors of groups until the sweep takes them (allocate_row_groups).
        self.row_groups = self._row_groups(np.full(0, -1, dtype=np.int32), 0, 0, 0.0)
        # Allocated once here: per-step allocation is forbidden under CUDA graph capture.
        arrays.allocate(WORLD_ROW_STATE, world=self.forest.world_count)
        arrays.allocate(WORLD_CONTACTS, world=self.forest.world_count)
        arrays.allocate(CONTACT_PASS, world=self.forest.world_count)

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

    def row_order(self, pair_keys, frame_kept) -> RowOrder:
        """The row order's records (:class:`.row_slots.RowOrder`) over the contact rows, with the frame's pair keys and
        whether the patch builder kept the last build's frame."""
        order = RowOrder()
        order.pair_keys = pair_keys
        order.frame_kept = frame_kept
        order.world_contact_first = self.world_contact_first
        order.world_contact_last = self.world_contact_last
        for name in (
            "rows_frame",
            "pairs_before",
            "pairs",
            "pair_sides",
            "pair_island",
            "pair_rank",
            "row_pairs",
            "rows_before",
        ):
            setattr(order, name, getattr(self, f"contact_{name}"))
        for name in ("island_parent", "island_size", "island_base"):
            setattr(order, name, getattr(self, name))
        order.turns = self.reserve_turns
        return order

    def allocate_row_groups(self, worlds: np.ndarray, min_units: int, need: float) -> None:
        """The dense line by colors of groups (:class:`.row_slots.RowGroups`) for ``worlds``, a slot each at the
        line's capacity, by the rule's ``min_units`` and ``need`` (groups a color)."""
        slot = np.full(self.forest.world_count, -1, dtype=np.int32)
        slot[worlds] = np.arange(len(worlds), dtype=np.int32)
        self.row_groups = self._row_groups(slot, len(worlds), min_units, need)

    def _row_groups(self, slot: np.ndarray, slots: int, min_units: int, need: float) -> RowGroups:
        trees = max(int(self.forest.articulation_response_dof_count.shape[0]), 1) if slots else 0
        rows = int(self.dense_max_constraints) if slots else 0
        return row_groups(slot, slots, rows, trees, min_units, need, self.model.device)

    def _on_contact_rows(self, rows) -> None:
        """Assign the per-contact buffers at the contact pool's ready rows, on creation and after each growth."""
        rows.publish(self)
        self.max_contacts_alloc = rows.rows
        patches = self.laws.friction_patches
        if patches is not None:
            patches.view.phi = self.friction_patch_phi

    def follow_contacts(self, contacts: Contacts) -> None:
        """Hold at least ``contacts``' rigid rows, and grow with the pool of a pipeline's buffer from now on (staged where
        the model holds articulations, whose rows own a contact pool). A fixed buffer's count, which keeps every contact
        its pass found, is clamped to its rows first: the solver reads every row below the count.

        Fails closed on lost contacts: outside capture, once the buffer's pool records a collision pass that dropped
        contacts for want of rows (its pinned record, read without synchronizing, so on CUDA up to the passes still in
        flight later), the step raises instead of solving without them. A growable buffer grows before a pass outside
        capture writes, so this is a fixed buffer's overflow; in graph replays the pipeline's
        :meth:`~newton.CollisionPipeline.service_capacity` warns of a drop."""
        pool = self.contact_pool
        rows = contacts._rigid_contact_rows
        source = getattr(contacts, "_rigid_contact_pool", None)
        try:
            if source is not None:
                source.clamp_count(contacts.rigid_contact_count)
                source.follow(pool)
            if rows > pool.ready:
                pool.grow(rows)
        except RuntimeError as error:
            raise RuntimeError(
                f"FeatherPGS received {rows} contact rows during CUDA graph capture, but its per-contact scratch "
                f"holds {pool.ready}: step once outside capture first."
            ) from error
        if source is not None and not wp.get_device(pool.device).is_capturing:
            drops = source.drops()
            if drops.passes:
                raise RuntimeError(
                    f"FeatherPGS: {drops.passes} collision pass(es), the last pass {drops.last_pass}, dropped "
                    f"{drops.contacts} rigid contacts past the contact buffer's {source.ready} rows; the solver does "
                    "not step without them. Build the CollisionPipeline with growable_contacts=True, whose rows grow "
                    "from the measured demand."
                )

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
        # The host's drop record (ROW_DROP_WORDS): pinned memory on CUDA, which the row pass's close writes through
        # the device's mapping of it, so the step reads it without synchronizing (fail_on_dropped_rows).
        self.row_drop_host = wp.zeros(ROW_DROP_WORDS, dtype=wp.int32, device=self.model.device)
        self.row_drop_mirror = self.row_drop_host
        if self.model.device.is_cuda:
            self.row_drop_host = wp.zeros(ROW_DROP_WORDS, dtype=wp.int32, device="cpu", pinned=True)
            self.row_drop_mirror = wp.array(
                ptr=self.row_drop_host.ptr, dtype=wp.int32, shape=(ROW_DROP_WORDS,), device=self.model.device
            )
        if self.forest.world_count == 0:
            return

        # The widest world's DOFs exist where a world response mapping was built (the routes that read them).
        world_dof = self.forest.max_world_dofs
        sizes = {"world": self.forest.world_count, "line": self.lines, "row": self.row_pitch, "world_dof": world_dof}
        arrays.allocate(DENSE_ROWS, **sizes)
        fused = bool(self.classes.fused_sizes)
        arrays.allocate(FUSED_DIAGONAL_LIMITS, **(sizes if fused else {"world": 1, "world_dof": 1}))
        # Per-row weights exist only for a weighted row law (relative regularization or material contacts), and
        # per-row compliance only for material contacts.
        if self.laws.set.weighted:
            arrays.allocate(DENSE_ROW_WEIGHTS, **sizes)
        else:
            self.row_w = self.contact_row_w_dummy
        if self.laws.set.material:
            arrays.allocate(DENSE_ROW_EPS, **sizes)
        else:
            self.row_eps = self.contact_row_eps_dummy

    def allocate_sparse_diagonal_rows(self):
        """The sparse-diagonal rows and their contact schedule (case K2)."""
        # The sparse-diagonal schedule's buffers, one row per world when it is off.
        diagonal = self.classes.diagonal
        if diagonal is not None:
            self.arrays.allocate(SPARSE_DIAGONAL_ROWS, line=self.lines, row=self.dense_max_constraints)
        else:
            self.arrays.allocate(SPARSE_DIAGONAL_ROWS, line=1, row=1)
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

    @property
    def packs_rows(self) -> bool:
        """Whether the dense rows live in the packed rows (:mod:`.packed_rows`): the matrix-free sweeps' and native
        sparse factors' (case K1)."""
        return bool(self.classes.mf_sweeps or self.classes.packed is not None)

    def allocate_packed_rows(self) -> None:
        """The packed rows (:mod:`.packed_rows`, :attr:`packs_rows`): every row of the world's line, by side or by
        support class; a placeholder without them."""
        if self.packs_rows:
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
            self.place_row_classes()
            sizes = {"line": self.lines, "row": self.row_pitch, "support": self.packed_support}
        else:
            self.packed_support, self.packed_dense = 1, 0
            self.row_classes = packed_rows.RowClasses()
            self.row_classes_reason = "No sweep reads packed rows."
            sizes = {"line": 1, "row": 1, "support": 1}
        self.arrays.allocate(packed_rows.PACKED_ROWS, **sizes)

    @property
    def classed(self) -> bool:
        return self.row_classes_reason is None

    def place_row_classes(self) -> None:
        """The rows by their support classes (:func:`.packed_rows.row_classes`, ``row_classes``) where they save the bytes
        that cost: for the general sweep, where a lane of its warp holds fewer of a row's entries (the classes' table
        loads cost more than narrower rows save at the same entries a lane: measured); for native sparse factors (case
        K1), whose unit writer stores and whose records and sweep read every entry of a row, wherever a row by class is
        narrower. ``row_classes_reason`` says why not (``None``: they do)."""
        self.row_classes = packed_rows.RowClasses()
        model, forest, factor = self.model, self.forest, self.factor
        self.row_classes_reason = first_failed(
            (
                (self.support_pin, "A test pin keeps the rows' sides."),
                (not self.packed_dense == packed_rows.WIDE, "Bilateral blocks correct rows past their sides."),
                (not self.classes.tier_rows, "The tiers read their rows at their coordinates."),
                (bool(factor.walk_classes), "No tree walks."),
            )
        )
        if self.row_classes_reason is not None:
            return
        mimic_dofs = None
        if self.laws.mimic_count:
            mimic_dofs = (self.laws._mimic_dof0.numpy(), self.laws._mimic_dof1.numpy())
        shape_body = model.shape_body.numpy()
        contact_body = np.zeros(max(int(model.body_count), 1), dtype=bool)
        contact_body[shape_body[shape_body >= 0]] = True
        joints = (
            model.articulation_start.numpy(),
            forest.articulation_joint_end_host,
            model.joint_child.numpy(),
            model.joint_qd_start.numpy(),
        )
        table = packed_rows.row_classes(
            walk_tables(factor.walk_classes),
            factor.walk_class_host,
            forest.art_to_world.numpy(),
            forest.world_count,
            joints,
            contact_body,
            mimic_dofs,
        )
        if table is None or table.width >= 64:
            self.row_classes_reason = "The support classes exceed a row word's class ids or first side's entries."
        elif self.classes.packed is not None:
            # Native sparse factors' unit writer stores every entry of a row and their records and sweep read it: a
            # narrower row saves its bytes whatever the lanes (measured: anymald-rough's rows of 12 by class, 1.2% of a
            # step faster than 18 by side).
            if table.width >= self.packed_support:
                self.row_classes_reason = (
                    f"Rows of {table.width} entries by class are no narrower than {self.packed_support} by side."
                )
        elif (table.width - 1) // _WARP >= (self.packed_support - 1) // _WARP:
            self.row_classes_reason = (
                f"Rows of {table.width} entries by class take as many entries a lane of a warp as {self.packed_support}"
                " by side."
            )
        if self.row_classes_reason is not None:
            return
        row_class = self.line_array("row_class", wp.int32, fill=-1)
        self.row_classes = field_group.upload(packed_rows.RowClasses, model.device, table, row_class=row_class)
        self.packed_support, self.packed_dense = int(table.width), 0

    def place_dense_response(self) -> None:
        """Where the dense rows' response lives (``packed_response``, ``packed_response_reason``): the packed rows
        (:mod:`.packed_rows`) wherever a matrix-free sweep reads them."""
        self.packed_response_reason = None if self.classes.mf_sweeps else "No matrix-free sweep reads packed rows."
        self.packed_response = self.packed_response_reason is None

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

    @property
    def split_records(self) -> bool:
        """Whether the line's records are split (:func:`.row_metadata.split_records`): a world past a packed record's
        coordinates or a class past its rows."""
        return split_records(self.forest.max_world_dofs, max(self.dense_max_constraints, self.mf_max_constraints))

    def allocate_row_records(self, first: int) -> None:
        """Allocate the streamed rows' records (:data:`.row_records.ROW_RECORDS`, :func:`.row_metadata.record_layout`)
        from the line's row ``first``; the rows' writers store them (:func:`.response_builds_rows.write_row_record`)."""
        self.record_first = int(first)
        rows = self.row_pitch - self.record_first
        free_rows = self.mf_max_constraints if self.free_contact_rows else 0
        dense = self.dense_max_constraints
        self.writes_records = bool(rows > 0 and (self.record_first < dense or free_rows > 0))
        words = record_layout(self.record_first, dense, self.row_pitch, self.split_records)[1]
        # A line's free rows' records follow its dense rows' where those stream: from word ``body_base * words``.
        self.dense_record_words = record_words(False, self.split_records) if self.record_first == 0 else 0
        # A line's records at the store's line length: the most words its rows can need, a dense row's record a row
        # (a line's free rows start at its body base, anywhere in it), whole words of the 16-byte quads records are
        # stored in.
        self.arrays.line_strides["record"] = self.line_rows * record_words(False, self.split_records)
        self.arrays.allocate(ROW_RECORDS, line=self.lines, record=max(words, 1))

    def hold_acts(self, solver) -> None:
        """Hold each world's acts word (:data:`WORLD_ACTS`) where every row is separable (:attr:`.route_table.Laws.separable`,
        no link) and the claims take the worlds. A contact's verdict (``contact_acts``) reads the body twists the
        predictor publishes (``acts_twists``) where a world can be held, a contact can occur and the predictor
        publishes every responding tree's; elsewhere a contact acts (the twists are published only where a consumer
        reads the verdict)."""
        laws, forest = solver.laws, self.forest
        self.acts_reason = first_failed(
            (
                (
                    laws.set.separable and not laws.link_count,
                    "A row can act at a separating residual or starts loaded (a warm start, a weighted contact, a "
                    "mimic, a link or a pre-eliminated row).",
                ),
                (solver.sweep.claims is not None, "Only the claims' sweeps leave a world out."),
                (solver.spec.pins.separated_worlds, "A test pin sweeps every world (separated_worlds=False)."),
            )
        )
        self.arrays.allocate(WORLD_ACTS, world=forest.world_count if self.acts_reason is None else 0)
        published = getattr(solver, "_predicted_twist_arts_host", None)
        contacts = RowLaw.CONTACT & RowLaw(int(np.bitwise_or.reduce(forest.world_laws, initial=0)))
        twists = self.acts_reason is None and bool(contacts) and published is not None
        twists = twists and bool(np.all(published[forest.response_dof_count > 0] != 0))
        self.acts_twists = solver._predicted_twist if twists else self.arrays.placeholder(wp.spatial_vector, (0,))

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
        wp.launch(
            _clear_dense_row_state,
            dim=self.forest.world_count,
            inputs=[
                self.row_slot_counter,
                self.row_first_rejected,
                self.row_count,
                self.row_dropped_all,
                self.home_first,
            ],
            outputs=[self.world_acts, self.pass_words, self.dense_first],
            device=self.model.device,
        )

    def close_rows(self, contacts: Contacts | None, families: int, warn: bool) -> None:
        """Close the row pass once every row producer reserved (:func:`_close_row_pass`): the first ``families`` row
        classes' counts with status (the dense rows', and the free bodies' run where its writer is staged), each world's
        invalid-status latch and, with ``warn`` (``Diagnostics.overflow_warning``), one warning per overflowing class."""
        # The contact capacity (argument 8) follows the ready rows in a capture whose launches bind.
        self.contact_pool.launch(
            _close_row_pass,
            self.forest.world_count,
            inputs=[
                self.row_slot_counter,
                self.row_first_rejected,
                wp.vec2i(self.dense_max_constraints, self.mf_max_constraints),
                int(families),
                self.row_dropped_all,
                contacts.rigid_contact_count if contacts is not None else self.dummy_contact_count,
                contacts._reduction_overflow if contacts is not None else self.dummy_contact_count,
                contacts._rigid_contact_rows if contacts is not None else 0,
                int(warn),
                int(self.forest.has_free_rigid_bodies),
                self.row_overflow_warning_emitted if warn else self.dummy_contact_count,
            ],
            outputs=[
                self.row_count,
                self.constraint_capacity_status,
                self.constraint_overflow,
                self.row_demand,
                self.row_drop_mirror,
            ],
            row_args=(8,) if contacts is not None else (),
        )

    def fail_on_dropped_rows(self) -> None:
        """Fail closed on dropped rows: outside capture, once the drop record (its pinned words, read without
        synchronizing, so on CUDA up to the passes still in flight later) holds a pass that reserved past a class's
        capacity, the step raises with the rows asked for and the option that holds them instead of solving without
        them. In graph replays the latched status (:meth:`check_capacity`) and the device warning report it."""
        if self.model.device.is_cuda and self.model.device.is_capturing:
            return
        record = self.row_drop_host.numpy()
        if record[0] == 0:
            return
        refused = int(self.arena_record_host.numpy()[1])
        arena = f"; the row store's arena held no run for {refused} claim(s)" if refused else ""
        raise RuntimeError(
            f"FeatherPGS: {int(record[0])} world pass(es) reserved rows past capacity and dropped them; "
            f"{self._asked(record[1], record[2])}{arena}; the solver does not step without them. Build the solver "
            "with that capacity, reset and recapture."
        )

    def _asked(self, dense: int, free: int) -> str:
        """The rows asked for past each class's capacity, as the options that would hold them (a world's dense rows hold
        its links' on top of the option's)."""
        links = int(self.dense_max_constraints) - int(self.options.dense_max_constraints)
        asked = []
        if dense > 0:
            asked.append(
                f"requested {int(dense)} dense rows a world against {int(self.dense_max_constraints)}: "
                f"dense_max_constraints >= {int(dense) - links}"
            )
        if free > 0:
            asked.append(
                f"requested {int(free)} matrix-free rows a world against {int(self.mf_max_constraints)}: "
                f"mf_max_constraints >= {int(free)}"
            )
        return "; ".join(asked)

    def reset_status(self, world_mask):
        """Clear the selected worlds' overflow latch and request their factors; the drop record restarts."""
        self.row_drop_mirror.zero_()
        wp.launch(
            _reset_solver_status,
            dim=max(self.forest.world_count, self.model.articulation_count),
            inputs=[
                world_mask,
                self.forest.art_to_world,
                self.factor.mass_update_requested,
                self.constraint_overflow,
                self.row_demand,
            ],
            device=self.model.device,
        )

    def record(self) -> dict:
        """The block cap of the launches over the contact capacity (``routes.contact_consumers``, 0: uncapped); the
        packed rows' width and why they keep their sides (``routes.row_classes``)."""
        blocks = int(self.contact_capacity_blocks)
        reason = (
            "Launches over the contact capacity (force export, warm-start snapshot and gathers)"
            f" grid-stride over {CAPACITY_BLOCKS_PER_SM} blocks of 256 threads per SM, about one wave."
            if blocks
            else "Uncapped: a CPU launch, or a kernel compiled without the grid-stride loop, covers every slot."
        )
        classes = {"support": int(self.packed_support), "reason": self.row_classes_reason}
        return {"contact_consumers": {"max_blocks": blocks, "reason": reason}, "row_classes": classes}

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
        pool = getattr(self, "contact_pool", None)
        result["contact_launches"] = pool is not None and pool.launch_failed(clear=clear)
        # The predictor's border (tree_kinetic_state.border_solve): bit 0, a tree slipped past its frame's capacity; bit 2,
        # a tree the walk does not hold took a decided step.
        result["slip_frames"] = bool(int(self.factor.walk_release_status.numpy()[0]) & 5)
        if clear:
            self.constraint_capacity_status.zero_()
            self.factor.walk_release_status.zero_()
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
            # The rows the invalid worlds asked for past each class's capacity, as the options that would hold them
            # (a world's dense rows hold its links' on top of the option's).
            demand = self.row_demand.numpy()[:, worlds].max(axis=1)
            asked = self._asked(demand[0], demand[1])
            raise RuntimeError(
                f"FeatherPGS constraint/contact capacity exceeded in worlds {worlds[:16].tolist()}"
                f" ({worlds.size} invalid worlds){'; ' + asked if asked else ''}; increase capacities and reset before"
                " accepting transitions."
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
        """Where the dense rows' entries sit and the seeds the one writer walks (:class:`.row_records.RowSupport`): the
        rows' sides, the articulations' coordinates and first world coordinates, the layout, and the unit sources
        (placeholders without packed rows or without a walk, :func:`.step_plan._stages`)."""
        support = RowSupport()
        support.sides = self.row_sides
        support.dofs = self.forest.articulation_response_dof_count
        support.offset = self.forest.articulation_world_dof_offset if self.packs_rows else self.dummy_is_free_rigid
        support.dense = int(self.packed_dense)
        support.first = self.dense_first
        support.units = self.unit_source if self.walks else self.dummy_units
        return support

    def inkernel_response_inputs(self) -> dict:
        """The parallel tiers' operands beside the walk: the worlds' trees (``ink_meta``) and the rows' law's cfm."""
        return {"ink_meta": self.ink_meta, "cfm": self.options.pgs_cfm}

    # ---------------------------------------------------------------------------------------------------------------
    # Step: the whitening and the diagonals
    # ---------------------------------------------------------------------------------------------------------------

    def lay_out_lines(self, rows: int, arena: int) -> None:
        """Lay out the row store, before its fields are allocated: a home line a world (``home_first``, where the step's
        topology rows are written) and ``arena`` lines more, each ``rows`` rows long, the outer stride of the store's
        arrays (which the native bodies read from the arrays themselves, :func:`.row_records.line_size`). A line
        field's arrays are runs of lines at that stride (:meth:`.field_group.SolverArrays.array`): its rows, and its
        friction records a slot per three rows. A world whose rows pass its line takes a run of the arena's lines in
        the reservation (:func:`.row_slots.reserve_world_rows_kernel`); each line's free rows start at its body base,
        a line's own (``body_base``, the dense rows' end, set by the reservation)."""
        model, worlds = self.model, self.forest.world_count
        self.line_rows, self.home_lines, self.arena_lines = int(rows), worlds, int(arena)
        self.lines = worlds + self.arena_lines
        self.line_geometry = wp.vec2i(self.line_rows, self.home_lines)
        # The arena's lines mapped, on the device (the reservation's claims' bound), and its record in pinned memory the
        # device writes and the host reads without synchronizing: the high water of the lines the claims reached and
        # the claims it refused.
        self.arena_bound = wp.array([self.arena_lines], dtype=wp.int32, device=model.device)
        self.arena_record_host = wp.zeros(2, dtype=wp.int32, device="cpu", pinned=model.device.is_cuda)
        self.arena_record = self.arena_record_host
        if model.device.is_cuda:
            self.arena_record = wp.array(
                ptr=self.arena_record_host.ptr, dtype=wp.int32, shape=(2,), device=model.device
            )
        self.arrays.line_strides.update(row=self.line_rows, contact=-(-self.line_rows // 3))
        self.body_base = wp.full(max(self.lines, 1), self.dense_max_constraints, dtype=wp.int32, device=model.device)
        # Each world's run of the arena (-1 for its home line), kept while it holds its slots, and the arena's lines
        # taken this frame.
        self.world_run = wp.full(max(worlds, 1), wp.vec2i(-1, 0), dtype=wp.vec2i, device=model.device)
        self.arena_cursor = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._run_fields = None

    def move_to_runs(self) -> None:
        """After the reservation, take each world whose rows run in the arena to its run (:func:`_move_to_runs`): none
        without an arena."""
        if not self.arena_lines or not self.forest.world_count:
            return
        # The store's fields (a feature's one-line placeholder holds no world's rows).
        fields = [array for array in self.arrays.line_fields if array.shape[0] == self.lines]
        if self._run_fields is None or self._run_fields[0] != len(fields):
            device = self.model.device
            self._run_fields = (
                len(fields),
                wp.array([int(a.ptr) for a in fields], dtype=wp.uint64, device=device),
                wp.array([int(a.strides[1]) for a in fields], dtype=wp.int64, device=device),
                wp.array([int(a.strides[0]) for a in fields], dtype=wp.int64, device=device),
                next(i for i, a in enumerate(fields) if a.ptr == self.diag.ptr),
            )
        count, pointers, row_bytes, line_bytes, diag = self._run_fields
        wp.launch(
            _move_to_runs,
            dim=(self.forest.world_count, count),
            inputs=[
                self.home_first,
                self.world_run,
                self.topology_rows,
                self.row_slot_counter,
                int(self.dense_max_constraints),
                pointers,
                row_bytes,
                line_bytes,
                diag,
            ],
            outputs=[self.dense_first],
            device=self.model.device,
        )

    def line_array(self, name: str, dtype, *, rows: int | None = None, fill=None) -> wp.array:
        """One field of the row store by line (``rows`` a line, the store's line by default), laid out as the store's
        fields are (:attr:`lines` at the store's line length)."""
        rows = self.row_pitch if rows is None else int(rows)
        return self.arrays.array(Field(name, dtype, ("line", "row"), fill=fill), {"line": self.lines, "row": rows})

    def hold_units(self) -> None:
        """The one writer's unit sources (:mod:`.walk_rows`): an int per row, by world line and row."""
        self.unit_source = self.line_array("unit_source", wp.int32, rows=self.dense_max_constraints, fill=0)

    def bound_topology(self, count, generation, stream: int, built_frame) -> None:
        """Snapshot where each world's topology rows end (:meth:`.rows.RowLaws.bound_topology`) and, where the packed
        rows hold the response (:attr:`clears_diagonals`), zero the dense rows' diagonals for the one writer's
        ``|Z|^2 + cfm``, in one launch (:func:`_bound_topology`). The sparse diagonal's and native sparse factors'
        writers store theirs with its CFM. The free bodies' rows, from ``body_base``, keep theirs: their writers stored
        them in the row pass (:mod:`.walk_rows`). No producer between the topology rows and the one writer reads or
        writes the dense rows' diagonals. The same launch decides which worlds keep their contacts' rows this step
        (``count``, ``generation``, ``stream`` and ``built_frame`` name the step's contact frame and the last build's,
        :func:`.friction_patches.frame_kept`)."""
        cleared = min(int(self.dense_max_constraints), int(self.line_rows)) if self.clears_diagonals else 0
        wp.launch(
            _bound_topology,
            dim=self.forest.world_count * max(cleared, 1),
            inputs=[self.row_slot_counter, cleared, count, generation, stream, built_frame, self.home_first],
            outputs=[
                self.topology_rows,
                self.diag,
                self.contact_rows_kept,
                self.world_held,
                self.pass_words,
                self.world_run,
                self.arena_cursor,
            ],
            device=self.model.device,
        )

    @property
    def clears_diagonals(self) -> bool:
        """Whether the dense rows' diagonals start from zero each step (:meth:`bound_topology`): the packed rows hold the
        response."""
        return bool(self.packed_response)

    def prepare_impulses(self):
        """Start the step's rows' impulses (:func:`.warm_start.prepare_world_impulses`): each world's live dense rows,
        and with free rigid bodies the live rows of their run."""
        free_max = self.mf_max_constraints if self.forest.has_free_rigid_bodies else 0
        # No more lanes a world than its rows' capacity: a world of a few rows takes as few; a few worlds of many rows
        # spread them over the GPU (row_lanes).
        lanes = row_lanes(
            self.forest.world_count, max(int(self.dense_max_constraints), int(free_max)), self.model.device
        )
        wp.launch(
            prepare_world_impulses,
            dim=(self.forest.world_count, lanes),
            inputs=[
                self.dense_first,
                self.constraint_count,
                self.dense_max_constraints,
                self.mf_constraint_count,
                self.body_base,
                free_max,
                lanes,
            ],
            outputs=[self.impulses],
            device=self.model.device,
        )

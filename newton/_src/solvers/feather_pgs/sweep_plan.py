# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweep (BACKBONE 2): which worlds each schedule claims, the impulses it starts from and carries, the launches.

:class:`Sweep` owns the solve of one step:

- the sweeps that own their worlds, each a configuration (:class:`.sweep.SweepConfig`) the plan binds from the class
  plan's choices, the laws and the placement, built by the one factory (:func:`.sweep.sweep_kernel`), and their streams
  (:meth:`Sweep.build`, :meth:`Sweep.open_streams`);
- the claims: the worlds the row pass left to each owner (:meth:`Sweep.claim_worlds`), and the resolved worlds;
- the warm start: the normal impulses gathered by contact identity (:meth:`Sweep.warm_start`), and what a step
  carries to the next (:meth:`Sweep.carry`);
- the launches, one bound launch a sweep (:class:`SweepLaunch`), in line or on the claims' streams
  (:meth:`Sweep.solve`): the step holds no case;
- the contact forces the impulses export (:meth:`Sweep.forces`).

The sweeps' mathematics lives in :mod:`.sweep` and the route modules; the schedule of case K2 in :mod:`.sweep_levels`.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import numpy as np
import warp as wp

from ...core.execution import Beside, Buckets, Queue, shape
from ...core.memory import Place, arch_budget, device
from . import (
    class_plan,
    packed_rows,
    sweep_levels,
    sweep_routes,
    sweep_routes_diag,
    sweep_routes_sparse,
    sweep_routes_tiers,
    sweep_units,
    walk_rows,
)
from .contact_forces import compute_contact_linear_force_from_impulses, pack_contact_linear_force_as_spatial
from .contact_input import bind_routes as bind_contact_routes
from .execution_plan import _MFGSStorage, _select_mfgs_storage
from .field_group import Field, FieldGroup
from .friction_patches import patch_routes
from .path_selection import json_value, kernel_key
from .response_builds_rows import bind_row_records, store_row_records
from .row_records import bind as bind_dense_rows
from .solve_policy import QUIET_UNITS, UNIT_ROUNDOFF, SolvePolicy, SolveReadoutRecord
from .sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from .sparse_mass_matrix import _SparseRowSupport, row_supports
from .sweep import EXACT, FRICTION_METRIC_FLOATS, LevelLaw, SweepConfig, sweep_kernel
from .sweep_levels import (
    _get_mark_independent_sparse_contact_candidates_kernel,
    _get_sparse_contact_levels_kernel,
)
from .tree_scan import classes_half
from .warm_start import (
    bind_carry,
    carry_contacts,
    forget_carried_contacts,
    gather_carried_normals,
    reset_world_impulses,
)

_COLORED = {
    "levels": LevelLaw(EXACT),
    "active_rows": False,
    "contact_units": False,
    "lanes_per_world": 32,
    "block_warps": 1,
}
"""The colored mapping's fields over the warp mapping's."""

SPARSE_ROWS = FieldGroup(
    Field("sparse_row_dof", wp.int32, ("index_world", "index_row", "index_support"), fill=None, owner="placement"),
    Field("sparse_row_factor", wp.float32, ("world", "row", "support"), fill=None, owner="placement"),
    Field("sparse_factor_velocity_delta", wp.float32, ("world", "world_dof"), fill=None, owner="placement"),
)
"""Each sparse row's factor coordinates (unless the support table holds them) and factor entries, and each world's
factor velocity change."""

SPARSE_CONTACT_RECORDS = FieldGroup(
    Field("sparse_contact_records", wp.float32, ("world", "triplet", "record"), owner="placement")
)
"""The single-stage transaction's per-triplet block records (``sparse_contact.CONTACT_RECORD_FLOATS``)."""

FRICTION_METRIC = FieldGroup(Field("friction_metric", wp.float32, ("world", "contact", "metric"), owner="placement"))
"""Where spin rows may appear, each friction contact's metric, formed once per launch by the sweeps that visit its rows
one by one, the matrix-free and the sparse-diagonal sweeps (:func:`.sweep.friction_record_stage`,
:func:`.sweep.friction_metric_stage`): a slot per three rows of the row store's line, a contact's rows."""


__all__ = ["Sweep"]


@dataclasses.dataclass(frozen=True)
class SweepLaunch:
    """One sweep's launch, bound once by the plan (:meth:`Sweep.build`): its configuration and kernel, the layer's
    shape of its work over its worlds (``grid`` blocks of ``block`` threads), and the operands it takes beside the
    step's (:meth:`Sweep._operands`), read at each launch (``named``): its claim's bucket, its placement's arrays, the
    walk."""

    config: SweepConfig
    kernel: wp.Kernel
    grid: int
    block: int
    named: Callable[[], dict]


class Sweep:
    """The solve of one step, with the owners that claim their worlds (see the module's docstring).

    Args:
        model: The solver's model.
        options: The checked options.
        forest: The forest plan.
        laws: The row laws.
        placement: The placement.
        factor: The tree factor.
        classes: The class plan (:class:`.class_plan.ClassPlan`): its choices, from which the plan binds the sweeps.
        pins: The test-only route pins (:class:`.route_table.Pins`).
    """

    def __init__(self, model, options, forest, laws, placement, factor, classes, pins):
        self.model = model
        self.options = options
        self.forest = forest
        self.laws = laws
        self.placement = placement
        self.factor = factor
        self.classes = classes
        self.pins = pins
        # Every sweep compiles the exit law and the policy's readout.
        self.policy = SolvePolicy(readout=options.diagnostics.solve_readout)
        self.readout = SolveReadoutRecord(self.policy)

    def place_diagonal(self, choice) -> None:
        """Case K2, a tree beside a star: each world's dense arm's offset and group, from the class plan's choice
        (:class:`.class_plan.Diagonal`, ``None`` where the class is another case)."""
        model, placement, chosen = self.model, self.placement, choice is not None
        placement.sparse_diagonal_dense_offsets = wp.array(
            choice.dense_offsets if chosen else np.zeros(1, dtype=np.int32), dtype=wp.int32, device=model.device
        )
        placement.sparse_diagonal_dense_groups = wp.array(
            choice.dense_groups if chosen else np.zeros(1, dtype=np.int32), dtype=wp.int32, device=model.device
        )

    def place_packed(self, choice) -> None:
        """Case K1, a single tree: native sparse factors with independent rigid-body response blocks, from the class
        plan's choice (:class:`.class_plan.Packed`, ``None`` where the class is another case)."""
        model = self.model
        self.sparse_velocity_limits = False
        self.support_rows = self.support_units_kernel = self._support_sink = None
        self.support_units = None
        self.support_du_base = None
        self.leaves = False
        self.placement.sparse_row_support = _SparseRowSupport()
        # The admitted contact units' block records.
        self.placement.sparse_contact_records = None
        self.placement.friction_metric = None
        self.sparse_sweep = self.sparse_kernel = None
        if choice is None:
            return
        sweep, max_support = dict(choice.sweep), choice.max_support
        self.sparse_velocity_limits = choice.velocity_limits
        self.factor.allocate_sparse(choice.plan)
        self.placement.sparse_body_dof_mask = wp.array(choice.body_masks, dtype=wp.uint64, device=model.device)
        if sweep["use_support_table"]:
            self.placement.sparse_row_support = row_supports(
                choice.support_classes, self.forest.world_count, self.placement.dense_max_constraints, model.device
            )
        if sweep["contact_triplets"] or self.laws.spin_rows:
            # Static storage for graph capture: the builder writes the block records of each step's
            # contact units, every friction triple's metric among them where spin rows may appear; linked sweeps add
            # each anchor's ring members once per call.
            width = CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS
            self.placement.arrays.allocate(
                SPARSE_CONTACT_RECORDS, world=self.forest.world_count, triplet=choice.triplet_capacity, record=width
            )
        index_shape = choice.index_shape
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
        # Contact torsion: each friction contact's spin row is the fourth of its unit (the laws' torsion).
        self.sparse_sweep = sweep_routes_sparse.SparseSweep(
            self.placement.dense_max_constraints,
            self.forest.max_world_dofs,
            max_support,
            **sweep,
            velocity_limits=self.sparse_velocity_limits,
            warm_start=self.options.pgs_warmstart,
            linked_rows=True,
            device_arch=model.device.arch,
            **self.law_fields,
        )
        self.sparse_kernel = self.sweep("sparse_pgs", self.sparse_sweep)

    def plan_support_walk(self) -> None:
        """Case K1 on the walk (FL3.4): the class's writer, release and leave, once the factor holds the walk
        (:meth:`.tree_factor.TreeFactor.plan_walk`).

        The class's tree (every articulation of the factored size shares it) and each coordinate's factor coordinate,
        the support masks' bit, are baked into the writer's walkers (:func:`.walk_rows.support_rows_kernel`) and the
        leave (the step's one leave, :func:`.factor.walk.walk_leave_kernel`, reading the sparse sweep's change at
        ``support_du_base`` and the class's nodes). The writer's first launch (:func:`.walk_rows.support_units_kernel`)
        lists the step's units, a world's lane group of :data:`.walk_rows.SUPPORT_UNIT_LANES` over its rows; its walkers
        stride over the list on lanes sized to the GPU (:func:`.walk_rows.support_threads`). The release is the factor's
        (:attr:`.factor.walk.WalkTrees.release`), built by :func:`.walk_rows.walk_release_kernel`.

        Raises:
            ValueError: For a factored size whose articulations do not share one tree.
        """
        model, forest, factor = self.model, self.forest, self.factor
        size = self.classes.packed_size
        arts = forest.group_to_art_host[size]
        # The group's tree: its articulations' walking classes' one path table.
        classes = factor.walk_classes
        tables = {
            (classes[cls].parent, classes[cls].first) if cls >= 0 else None for cls in factor.walk_class_host[arts]
        }
        if len(tables) != 1 or None in tables:
            raise ValueError("The native sparse factor's articulations must share one tree")
        parent, first = next(iter(tables))
        node = tuple(int(k) for k in factor.sparse_pattern.inverse_permutation)
        torsion = bool(self.laws.spin_rows)
        self.support_lanes = walk_rows.SUPPORT_UNIT_LANES if wp.get_device(model.device).is_cuda else 1
        self.support_units_kernel = walk_rows.support_units_kernel(4 if torsion else 3, self.support_lanes)
        self.support_rows = walk_rows.support_rows_kernel(
            parent,
            first,
            node,
            records=self.placement.sparse_contact_records is not None,
            torsion=torsion,
            half=classes_half(walk for walk in factor.walk_classes if (walk.parent, walk.first) == (parent, first)),
        )
        # Where the step's one leave reads each coordinate's change (factor.walk.walk_leave_kernel): the sparse sweep
        # publishes it in factor coordinates (factor_velocity_delta), so a tree's changes start at its offset in its
        # world's and a coordinate reads its factor coordinate, the support masks' bit.
        width = int(forest.max_world_dofs)
        art_world = forest.art_to_world.numpy()
        offsets = forest.articulation_world_dof_offset.numpy()
        du_base = np.zeros(max(int(model.articulation_count), 1), dtype=np.int32)
        du_base[arts] = art_world[arts] * width + offsets[arts]
        self.support_du_base = wp.array(du_base, dtype=wp.int32, device=model.device)
        factor.plan_leave(
            tuple(
                node if (walk.parent, walk.first) == (parent, first) else tuple(range(walk.coordinates))
                for walk in factor.walk_classes
            )
        )

    def support_sink(self, v_hat: wp.array) -> walk_rows.RowSink:
        """Case K1: what its writer reads and writes besides each unit's seeds (:class:`.walk_rows.RowSink`): the
        support table, the support masks, the articulations' first world coordinates, and the rows' factor entries,
        coordinates and block records, bound once: every array is the placement's for the solver's life."""
        if self._support_sink is None:
            placement, factor, forest = self.placement, self.factor, self.forest
            sink = walk_rows.row_sink(factor, forest, placement, v_hat)
            sink.support = placement.sparse_row_support
            sink.body_mask = placement.sparse_body_dof_mask
            sink.dof_mask = factor.sparse_ancestor_mask
            sink.world_offset = forest.articulation_world_dof_offset
            sink.entries, sink.row_dof = placement.sparse_row_factor, placement.sparse_row_dof
            records = placement.sparse_contact_records
            sink.records = records if records is not None else placement.sparse_row_factor
            self._support_sink = sink
            worlds, capacity = sink.entries.shape[0], sink.entries.shape[1]
            self.support_units = walk_rows.unit_list(
                worlds, capacity, self.model.device, spin_rows=bool(self.laws.spin_rows)
            )
            self.support_threads = walk_rows.support_threads(worlds * capacity, self.model.device)
        return self._support_sink

    def place_tiers(self, choice) -> None:
        """Case K4, the Jacobi tiers: their in-kernel response's trees, from the class plan's choice
        (:class:`.class_plan.Tiers`, ``None`` where the class is another case)."""
        if choice is not None:
            self.placement.ink_meta = wp.array(choice.meta.reshape(-1), dtype=wp.int32, device=self.model.device)

    def _tier_frame(self) -> tuple[int, int]:
        """Each of the tiers' trees' slip capacity (:meth:`.tree_factor.TreeFactor.slip_capacity`): the release its rows
        and leave compile, none where no tree of that size holds a limited drive."""
        na, nb = self.classes.tiers.sizes[0], self.classes.tiers.sizes[1]
        return tuple(
            self.factor.slip_capacity(self.forest.group_to_art[size].numpy()) if size else 0 for size in (na, nb)
        )

    def _parallel_velocity_limit_pass(self) -> bool:
        """Whether the parallel tiers end with the last-word velocity-limit pass (``routes.sweep.tiers``).

        The linked (patch) tiers give the joint velocity limits the last word, as the serial sweeps' end-of-iteration
        pass gives it to them: after the final sweep one warp pushes every violated velocity-limit row back onto its
        limit, in row order. It changes nothing once a tier converges and bounds the limit overshoot of a tier its
        sweep budget stops early; it has rows to push only with joint velocity limits.
        """
        return bool(self.options.enable_joint_velocity_limits)

    def _parallel_tier_sweeps(self, tiers) -> tuple[list[int], str, str]:
        """Return each parallel tier's sweep budget, the rule that set it, and why (``routes.sweep.tiers``).

        ``tiers`` lists ``(rows, min_rows)`` per tier. Every tier, under either friction law, sweeps
        :func:`.sweep_routes_tiers.tier_sweeps` of its row bound: the equal-accuracy rule, from ``pgs_iterations``
        at construction. A world stops earlier under the sweeps' exit law. A test pin (``tier_sweeps``) sets every
        tier's budget.
        """
        pinned = self.pins.tier_sweeps
        if pinned is not None:
            return [int(pinned)] * len(tiers), "pinned", f"A test pin sets every tier's budget (tier_sweeps={pinned})."
        reason = (
            f"auto: a tier serving up to R rows sweeps ceil({sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION} * "
            f"pgs_iterations * sqrt(R / 32)) times with pgs_iterations={int(self.options.pgs_iterations)}, at most "
            f"{sweep_routes_tiers.TIER_SWEEPS_CAP}; a world stops earlier under the sweeps' exit law."
        )
        return [sweep_routes_tiers.tier_sweeps(rows, self.options.pgs_iterations) for rows, _ in tiers], "auto", reason

    @property
    def law_fields(self) -> dict:
        """The row laws every sweep configuration compiles (:class:`.sweep.SweepConfig`): the solve policy, weighted
        rows, material contacts and spin rows."""
        return {
            "solve_policy": self.policy,
            "weighted": bool(self.laws.set.weighted),
            "material": bool(self.options.contact_compliance),
            "torsion": bool(self.laws.spin_rows),
        }

    def sweep(self, route: str, config: SweepConfig, readout_budget: int | None = None) -> wp.Kernel:
        """Build the sweep of ``config`` (:func:`.sweep.sweep_kernel`) and allocate its readout counters, outside any
        graph capture.

        Args:
            route: Owner of the sweep in the records, such as ``"sparse_pgs"``.
            config: The sweep's configuration, its laws :attr:`law_fields`.
            readout_budget: The sweep budget the kernel compiles in (a parallel tier's), or ``None`` for the
                solver's iteration counts.
        """
        kernel = sweep_kernel(config)
        self._register(route, kernel, readout_budget)
        return kernel

    def _register(self, route: str, kernel: wp.Kernel, readout_budget: int | None) -> None:
        policy = self.policy
        budget = int(self.options.pgs_iterations if readout_budget is None else readout_budget)
        self.readout.register(
            kernel,
            route,
            readout=policy.readout if policy.counts else None,
            world_count=int(self.forest.world_count),
            budget=max(budget, 1),
            device=self.model.device,
        )

    def build(self) -> None:
        """Build the sweeps."""
        model = self.model
        device_arch = model.device.arch

        # The sweeps start from the impulses the rows carry: the warm start's, the closed loop's projection.
        self.seeded = bool(self.options.pgs_warmstart or self.factor.eliminates)
        # The line's speed rows (step_plan.Capacities.speed_rows), where a free body's rows are the line's.
        speed_rows = self.placement.speed_rows > 0
        diagonal = self.classes.diagonal
        sparse_contact_schedule = diagonal is not None and diagonal.units
        self.sparse_contact_marker = (
            _get_mark_independent_sparse_contact_candidates_kernel(
                self.classes.diagonal_size,
                device_arch,
                torsion=self.laws.spin_rows,
            )
            if sparse_contact_schedule
            else None
        )
        self.sparse_contact_schedule = (
            _get_sparse_contact_levels_kernel(
                self.placement.dense_max_constraints,
                self.forest.max_world_dofs,
                device_arch,
                build_serial_contacts=diagonal.speculative_batches,
                torsion=self.laws.spin_rows,
            )
            if sparse_contact_schedule
            else None
        )
        self.diagonal_sweep = (
            sweep_routes_diag.DiagonalSweep(
                self.placement.dense_max_constraints,
                self.forest.max_world_dofs,
                diagonal.dense_size,
                device_arch,
                speculative_contact_batches=diagonal.speculative_batches,
                levels=LevelLaw(EXACT) if diagonal.units else LevelLaw(),
                seeded=self.seeded,
                velocity_limits=bool(self.options.enable_joint_velocity_limits),
                **self.law_fields,
            )
            if diagonal is not None
            else None
        )
        self.sparse_diagonal_kernel = (
            self.sweep("sparse_diagonal", self.diagonal_sweep) if self.diagonal_sweep is not None else None
        )
        row_by_row = self.sparse_diagonal_kernel is not None
        if self.laws.spin_rows and (self.classes.mf_sweeps or row_by_row):
            # Static storage for graph capture: the friction contacts' metrics, formed per launch by the sweeps that
            # visit a contact's rows one by one (sweep.friction_record_stage).
            self.placement.arrays.allocate(
                FRICTION_METRIC,
                world=self.forest.world_count,
                contact=(self.placement.row_pitch + 2) // 3,
                metric=FRICTION_METRIC_FLOATS,
            )
        self.mf_gs_kernel = None
        # The colored mapping of the matrix-free sweep and its units, when a world can take it.
        self.mf_colored_kernel = None
        self.colored_units = None
        # Storage classes of the MF-GS kernel (_MFGSStorage); recorded by get_path_selection().
        self.mf_gs_storage = None
        self.general_active_reason = None
        # The Jacobi tiers ahead of the matrix-free sweep (their SweepLaunch each).
        self.tiers = []
        self.par_tiers = []
        # The claims' worlds (sweep_levels.claim_worlds): a bucket a tier, then the general sweep's.
        self.claims = None
        self.claims_beside = None
        # The one writer's EMIT sink (walk_rows.RowSeeds): none until a tier walks its rows.
        self.placement.row_seeds = walk_rows.row_seeds(self.placement, 0, 0, sweep_routes_tiers.TIER_ROWS)
        if self.classes.mf_sweeps:
            # A test pin streams the impulses, the free bodies' or every row's (storage is part of the kernel
            # specialization key); the selection sizes the resident row metadata with its choice.
            stream_line = self.pins.mf_storage == "line"
            force_stream_mf_impulses = stream_line or self.pins.mf_storage == "streamed"
            on_cpu = Place.GLOBAL if force_stream_mf_impulses else Place.SHARED
            storage = _MFGSStorage(
                sweep_routes.RowPlaces(Place.GLOBAL, on_cpu, Place.GLOBAL if stream_line else Place.SHARED),
                "Without CUDA no shared-memory storage applies.",
            )
            if model.device.is_cuda:
                storage = _select_mfgs_storage(
                    functools.partial(
                        sweep_routes.mf_gs_staging,
                        self.placement.dense_max_constraints,
                        self.placement.mf_max_constraints,
                        self.forest.max_world_dofs,
                        free_rows=self.placement.free_contact_rows,
                    ),
                    arch_budget(device_arch),
                    force_stream_mf_impulses=force_stream_mf_impulses,
                    force_stream_line=stream_line,
                )
            self.mf_gs_storage = storage
            parallel_rows = self.classes.tier_rows
            # The streamed rows' records, in the worlds the general sweep and the unit builder read.
            self.placement.allocate_row_records(
                sweep_routes.record_first(self.placement.dense_max_constraints, storage.places)
            )
            # The records' pass: a warp of lanes per 256 rows of the line, at most a block, over a wave of the device.
            self.record_lanes = min(256, 32 * -(-self.placement.row_pitch // 256))
            threads = max(int(model.device.sm_count), 1) * 512
            self.record_grid = min(self.forest.world_count, max(threads // self.record_lanes, 1))
            # The warp mapping: the storage's places, the placement's rows, lanes by the rows' support where a warp's
            # stages fit (two warps a block where theirs do), the sparse sweep's rule for the enumeration and units.
            budget = arch_budget(device_arch)
            staging = functools.partial(
                sweep_routes.mf_gs_staging,
                self.placement.dense_max_constraints,
                self.placement.mf_max_constraints,
                self.forest.max_world_dofs,
                places=storage.places,
                free_rows=self.placement.free_contact_rows,
                wide_rows=self.placement.packed_dense == packed_rows.WIDE,
            )
            support_lanes = sweep_routes.world_lanes(
                self.placement.packed_support, free_rows=bool(self.placement.free_contact_rows)
            )
            small = support_lanes if support_lanes < 32 else 0
            active_rows, self.general_active_reason = class_plan.active_rows_rule(
                self.pins, small_world_lanes=small, fits=lambda: staging(active_rows=True).fits(budget)
            )
            staging = functools.partial(staging, active_rows=active_rows)
            lanes, block_warps = 32, 1
            if model.device.is_cuda:
                if staging(lanes=support_lanes, groups=32 // support_lanes).fits(budget):
                    lanes = support_lanes
                staging = functools.partial(staging, lanes=lanes)
                if staging(groups=32 * sweep_routes.BLOCK_WARPS // lanes).fits(budget):
                    block_warps = sweep_routes.BLOCK_WARPS
            self.general_sweep = sweep_routes.GeneralSweep(
                self.placement.dense_max_constraints,
                self.placement.mf_max_constraints,
                self.forest.max_world_dofs,
                device_arch,
                pitch=self.placement.row_pitch,
                has_dense_velocity_limit_rows=bool(self.options.enable_joint_velocity_limits),
                places=storage.places,
                max_row_dofs=self.placement.packed_support,
                seeded=self.seeded,
                wide_rows=self.placement.packed_dense == packed_rows.WIDE,
                support_classes=self.placement.classed,
                active_rows=active_rows,
                contact_units=self.pins.contact_units == "triplets" and not small,
                lanes_per_world=lanes,
                block_warps=block_warps,
                free_rows=bool(self.placement.free_contact_rows),
                speed_rows=speed_rows,
                **self.law_fields,
            )
            # The colored mapping takes the worlds the unit builder's rule picks at every pack
            # (sweep_units.COLORED_RULE), among those whose free bodies and row capacity admit it: built, with a
            # storage slot each, for those worlds only, it runs beside the one-warp mapping. More of them than the
            # device holds colored blocks run in waves, which scales the rule (ColoredRule.at_scale).
            rule = sweep_units.COLORED_RULE
            # The colored mapping sweeps the free bodies' run, which exists only where a world can take it.
            capable = np.zeros(0, dtype=np.int32)
            if self.placement.free_contact_rows:
                capable = sweep_units.capable_worlds(
                    self.forest.free_rigid_per_world, self.placement.mf_max_constraints, rule
                )
            if model.device.is_cuda and capable.size > 0:
                found = device(model.device)
                colored_work = dataclasses.replace(self.general_sweep, **_COLORED).work()
                rule = rule.at_scale(capable.size, shape(colored_work, found).blocks_per_sm * found.sms)
                capable = sweep_units.capable_worlds(
                    self.forest.free_rigid_per_world, self.placement.mf_max_constraints, rule
                )
            colored = bool(model.device.is_cuda and capable.size > 0)
            self.mf_gs_kernel = self.sweep("mf_gs", self.general_sweep)
            if colored:
                self.colored_sweep = dataclasses.replace(self.general_sweep, **_COLORED)
                self.mf_colored_kernel = self.sweep("mf_colored", self.colored_sweep)
            if parallel_rows > 0:
                # The producer's sink: a tier that takes EMIT walks its rows from their seeds in its prologue.
                emit = self._emits()
                width = sweep_routes_tiers.TIER_ROWS
                tiers = [(min(first + width, parallel_rows), first) for first in range(0, parallel_rows, width)]
                self.par_tiers = list(tiers)
                tier_sweeps, self.par_sweeps_rule, self.par_sweeps_reason = self._parallel_tier_sweeps(tiers)
                for tier_index, ((rows_, _), sweeps_) in enumerate(zip(tiers, tier_sweeps, strict=True)):
                    config = sweep_routes_tiers.TierSweep(
                        self.placement.dense_max_constraints,
                        self.placement.mf_max_constraints,
                        self.forest.max_world_dofs,
                        device_arch,
                        rows=rows_,
                        levels=LevelLaw.split(sweeps_),
                        inkernel_response=self.classes.tiers.sizes,
                        emit=emit,
                        linked_rows=True,
                        velocity_limit_pass=self._parallel_velocity_limit_pass(),
                        frame=self._tier_frame(),
                        classes=self.factor.walk_classes,
                        pitch=self.placement.row_pitch,
                        **self.law_fields,
                    )
                    kernel = self.sweep("parallel_tier", config, readout_budget=sweeps_)
                    # On its own list, a tier takes a block a world of the list's capacity (its work's shape); it
                    # grid-strides over its bucket.
                    layout = shape(config.work(self.forest.world_count), device(model.device))
                    named = functools.partial(self._tier_operands, config, tier_index, layout.grid)
                    self.tiers.append(SweepLaunch(config, kernel, layout.grid, layout.block, named))
                if emit:
                    self.placement.row_seeds = walk_rows.row_seeds(
                        self.placement, (1 << len(tiers)) - 1, parallel_rows, width
                    )
            # Each sweep walks its bucket of the claims' worlds (claim_worlds), a bucket of every world's slots: a tier's,
            # the general sweep's, and with the colored mapping its candidates' (which the unit builder walks) and the
            # colored worlds' (which it appends).
            worlds = self.forest.world_count
            self.claim_general = len(self.tiers)
            self.claim_candidate = self.claim_general + 1 if colored else -1
            self.claim_colored = self.claim_general + 2 if colored else -1
            buckets = self.claim_general + (3 if colored else 1)
            self.claims = Queue(buckets * worlds, model.device, buckets=buckets)
            # The claim stage runs beside the step's stream until its first reader (claim_worlds).
            self.claims_beside = Beside(model.device)
            if colored:
                self.colored_units = sweep_units.ColoredUnits(
                    self.forest.world_count,
                    self.placement.mf_max_constraints,
                    self.forest.max_world_dofs,
                    model.device,
                    worlds=capable,
                    rule=rule,
                    torsion=self.laws.spin_rows,
                    weighted=self.laws.set.weighted,
                    claims=self.claims,
                    buckets=(self.claim_candidate, self.claim_colored, self.claim_general),
                    support=self.placement.packed_support,
                )
            # The general sweep walks its bucket on a persistent grid, a warp a world (its work's shape); the colored
            # bucket a block a world of its slots, a color's units at once.
            general = self._claim_launch(self.general_sweep, self.mf_gs_kernel, worlds, self.claim_general)
            mapped = (general,)
            if colored:
                slots = self.colored_units.slot_count
                mapped += (self._claim_launch(self.colored_sweep, self.mf_colored_kernel, slots, self.claim_colored),)
            # In line, the tiers and then the general sweep; on the claims' streams (open_streams) the first tier in
            # line and the general sweep and the other tiers beside it.
            self.launches = (*self.tiers, *mapped)
            self.claim_groups = (*((tier,) for tier in self.tiers[:1]), mapped, *((tier,) for tier in self.tiers[1:]))
        else:
            # A class whose one sweep owns every world (K1, K2): one launch over the worlds, in line.
            single = (
                (self.sparse_sweep, self.sparse_kernel, self._sparse_operands),
                (self.diagonal_sweep, self.sparse_diagonal_kernel, self._diagonal_operands),
            )
            self.launches = tuple(
                self._world_launch(config, kernel, named) for config, kernel, named in single if config is not None
            )
            self.claim_groups = ()
        if self.classes.mf_sweeps:
            # No matrix-free sweep enters a world without rows; the readout counts those solves as skipped.
            self.readout.register_skipped(world_count=int(self.forest.world_count), device=model.device)
        # The sweeps that walk their rows (and the Jacobi tiers, which always leave by the walk) add their change to the
        # walk's du in its coordinates; the step's one leave walks every tree down (factor.walk.walk_leave_kernel).
        self.leaves = self.factor.walk_arts is not None and bool(self.placement.walks or self.tiers)

    def open_streams(self, *, overlap: bool) -> None:
        """Open the claims' streams: the sweeps that own their worlds run on them beside the matrix-free sweep.

        The claims touch nothing the matrix-free sweep touches, so they fork after the rows are published and join
        before the leave: the Jacobi tiers, whose worlds hold no free-body row (nor a speed row of the line), none of
        the matrix-free sweep's. A tier's worlds are those of its row count, so each tier is a claim of its own.

        The claims' worlds are written before the fork (:func:`.sweep_levels.claim_worlds`), since the matrix-free sweep
        walks its bucket of them too. A stream a tier.

        The other side streams (inertia, dynamics, memset) read no contact rows nor the contact pool's ready count:
        they run factorizations and articulation data. The claims do (the tiers), between the fork and the join
        only, and the launch stream, which the pool registers, waits for the join before it goes on. A trim that waits
        for the launch stream therefore waits for them too (:meth:`ContactPool.add_reader_stream`).
        """
        model = self.model
        claims = len(self.tiers) if wp.get_device(model.device).is_cuda and overlap else 0
        self.claim_streams = tuple(wp.Stream(model.device) for _ in range(claims))
        self.claims_ready = wp.Event(model.device) if claims else None
        self.claims_done = tuple(wp.Event(model.device) for _ in range(claims))

    def claim_worlds(self) -> None:
        """The claim stage: every world with rows into its claim's bucket of :attr:`claims`
        (:func:`.sweep_levels.claim_worlds`), once the row counts are final (after the status latch): a tier's (only dense
        rows, inside the tiers' row bound), the colored candidates' (a world with a colored storage slot) or the
        general sweep's. The one writer, the unit builder and every sweep read their buckets.

        Its first reader is the unit builder or the sweep: it runs beside the stage's stream until
        :meth:`join_claims`."""
        if self.claims_beside is not None:
            self.claims_beside.start(self._claim_worlds)
        else:
            self._claim_worlds()

    def join_claims(self) -> None:
        """Make the step's stream wait for the claim stage (before its first reader)."""
        if self.claims_beside is not None:
            self.claims_beside.join()

    def _claim_worlds(self) -> None:
        claims, units = self.claims, self.colored_units
        claims.count.zero_()
        wp.launch(
            sweep_levels.claim_worlds,
            dim=self.forest.world_count,
            inputs=[
                self.placement.constraint_count,
                self.placement.mf_constraint_count,
                self.placement.speed_first,
                units.units["world_slot"] if units is not None else self.placement.dummy_is_free_rigid,
                self.par_tiers[-1][0] if self.par_tiers else 0,
                sweep_routes_tiers.TIER_ROWS,
                self.claim_general,
                self.claim_candidate,
                claims.region,
            ],
            outputs=[claims.items, claims.count],
            device=self.model.device,
        )

    def general_claims(self) -> Buckets:
        """The claims' worlds the general sweep and the unit builder read: the general bucket and the candidates'
        (consecutive), which the unit builder splits."""
        general = self.claim_general
        return self.claims.span(general, general + (1 if self.colored_units is None else 2))

    def _emits(self) -> bool:
        """Whether the tiers' rows take the one writer's EMIT sink (:class:`.walk_rows.RowSeeds`): where the writer
        closes no bilateral block, whose closure reads every row's stored ``Z``, and no test pins the store."""
        return not (self.factor.eliminates and self.factor.block_capacity) and self.pins.sink != "store"

    def store_records(self) -> None:
        """The streamed rows' records of the worlds :meth:`general_claims` lists, once their writers and the patch link
        left their rows final (:func:`.response_builds_rows.store_row_records`), a lane group a world."""
        self.join_claims()
        placement = self.placement
        wp.launch(
            store_row_records,
            dim=(self.record_grid, self.record_lanes),
            inputs=[
                self.general_claims(),
                self.record_grid,
                self.record_lanes,
                bind_row_records(placement),
                bind_dense_rows(placement),
                placement.row_incident,
                placement.constraint_count,
                placement.mf_constraint_count,
            ],
            device=self.model.device,
        )

    def schedule_diagonal(self) -> None:
        """Case K2: publish the sparse diagonal's schedule of the rows the row pass wrote (run after every producer):
        unit heads from the row types, a world's warp, several a block (their levels' shared working set)."""
        dim, block = sweep_levels.levels_launch(
            self.forest.world_count, self.forest.max_world_dofs, self.model.device.arch
        )
        wp.launch(
            self.sparse_contact_schedule,
            dim=dim,
            inputs=[
                self.placement.constraint_count,
                self.placement.dense_first,
                self.placement.dense_phase_bounds,
                self.placement.row_type,
            ],
            outputs=[
                self.placement.sparse_diagonal_row_dof,
                self.placement.sparse_contact_level_count,
                self.placement.sparse_contact_level_offsets,
                self.placement.sparse_contact_level_units,
                self.placement.sparse_contact_serial_count,
                self.placement.sparse_contact_serial_normals,
            ],
            block_dim=block,
            device=self.model.device,
        )

    def warm_start(self, contacts) -> None:
        """Gather the previous step's normal impulses by contact identity into both row classes' normal rows, one launch
        over the contacts (:func:`.warm_start.gather_carried_normals`)."""
        if contacts is None:
            return
        placement = self.placement
        pool = placement.contact_pool
        pool.launch(
            gather_carried_normals,
            pool.launch_rows(),
            inputs=[
                bind_contact_routes(placement, contacts),
                contacts.rigid_contact_match_index,
                bind_carry(placement),
                placement.dense_first,
                placement.row_type,
                self.options.pgs_warmstart_decay,
                # Carried support impulses are proportional to dt: the step clock rescales them.
                self.factor.clock,
            ],
            outputs=[placement.impulses],
            extent=contacts.rigid_contact_count,
            max_blocks=placement.contact_capacity_blocks,
        )

    def _operands(self, v_hat: wp.array, v_out: wp.array) -> dict:
        """The step's sweep operands by parameter name, which every sweep kernel takes from (:meth:`_launch`). A row
        array is read here, each step: the contact pool's growth rebinds what a build would hold."""
        placement, forest = self.placement, self.forest
        return {
            "world_constraint_count": placement.constraint_count,
            "dense_first": placement.dense_first,
            "dense_phase_bounds": placement.dense_phase_bounds,
            "speed_first": placement.speed_first,
            "world_dof_indices": forest.world_dof_indices,
            "rhs_bias": placement.rhs,
            "row_incident": placement.row_incident,
            "world_diag": placement.diag,
            "world_row_w": placement.row_w,
            "world_row_eps": placement.row_eps,
            "world_impulses": placement.impulses,
            "response_dofs": forest.articulation_response_dof_count,
            "articulation_world_dof_offset": forest.articulation_world_dof_offset,
            "packed_dense": int(placement.packed_dense),
            "packed_Z": placement.packed_Z,
            "row_sides": placement.row_sides,
            "row_classes": placement.row_classes.row_class,
            "class_dof": placement.row_classes.class_dof,
            "class_count": placement.row_classes.class_count,
            "world_row_type": placement.row_type,
            "world_row_parent": placement.row_parent,
            "world_row_mu": placement.row_mu,
            "mf_constraint_count": placement.mf_constraint_count,
            "mf_contact_rows_end": placement.mf_contact_rows_end,
            # The streamed rows' records (the free bodies', and the dense rows' where they stream).
            "row_record": placement.row_record,
            "iterations": self.options.pgs_iterations,
            "omega": self.options.pgs_omega,
            "friction_metric": placement.friction_metric,
            "v_hat": v_hat,
            "v_out": v_out,
        }

    def _world_launch(self, config: SweepConfig, kernel: wp.Kernel, named: Callable[[], dict]) -> SweepLaunch:
        """The launch of a sweep over every world (the layer's shape of its work over them)."""
        layout = shape(config.work(self.forest.world_count), device(self.model.device))
        return SweepLaunch(config, kernel, layout.grid, layout.block, named)

    def _claim_launch(self, config: SweepConfig, kernel: wp.Kernel, items: int, bucket: int) -> SweepLaunch:
        """The launch of a matrix-free mapping over its claim's ``bucket``, its grid the layer's shape of its work over
        ``items`` (the bucket's capacity)."""
        layout = shape(config.work(items), device(self.model.device))
        return SweepLaunch(
            config,
            kernel,
            layout.grid,
            layout.block,
            functools.partial(self._mapped_operands, bucket, layout.grid * layout.groups),
        )

    def _walk(self) -> dict:
        """The walk's view, where the sweeps leave by the walk and find a free body's tree in it."""
        return {} if self.factor.walk_arts is None else {"walk": self.factor.walk_view()}

    def _queued(self, bucket: int, grid: int) -> dict:
        """The world queue of the claims' ``bucket`` (claim_worlds, the step's claim stage), walked at ``grid``'s
        stride (the launch's lane groups)."""
        count, worlds = self.claims.bucket(bucket)
        return {"general_world_count": count, "general_worlds": worlds, "general_world_grid_stride": grid}

    def _tier_operands(self, config, bucket: int, grid: int) -> dict:
        """A Jacobi tier's operands: its bucket, the in-kernel response's trees and, under EMIT, the rows' seeds."""
        named = dict(self._walk(), **self._queued(bucket, grid))
        named.update(self.placement.inkernel_response_inputs())
        if config.emit:
            named["row_seeds"] = self.placement.row_seeds
        return named

    def _mapped_operands(self, bucket: int, grid: int) -> dict:
        """A matrix-free mapping's operands: its bucket and the colored storage's units."""
        colored = {} if self.mf_colored_kernel is None else self.colored_units.operands()
        return {**self._walk(), **colored, **self._queued(bucket, grid)}

    def _sparse_operands(self) -> dict:
        """Case K1's operands: its rows' factor entries, coordinates and records, and its change in factor
        coordinates."""
        placement = self.placement
        return {
            "world_count": self.forest.world_count,
            "row_dof": placement.sparse_row_dof,
            "row_factor": placement.sparse_row_factor,
            "factor_velocity_delta": placement.sparse_factor_velocity_delta,
            "support": placement.sparse_row_support,
            "contact_records": placement.sparse_contact_records,
        }

    def _diagonal_operands(self) -> dict:
        """Case K2's operands: its schedule (:meth:`schedule_diagonal`), its rows' slots, its fused limits."""
        placement = self.placement
        return {
            "sparse_contact_level_count": placement.sparse_contact_level_count,
            "sparse_contact_level_offsets": placement.sparse_contact_level_offsets,
            "sparse_contact_level_units": placement.sparse_contact_level_units,
            "sparse_contact_serial_count": placement.sparse_contact_serial_count,
            "sparse_contact_serial_normals": placement.sparse_contact_serial_normals,
            "dense_offsets": placement.sparse_diagonal_dense_offsets,
            "dense_groups": placement.sparse_diagonal_dense_groups,
            "walk": self.factor.walk_view(),
            "dense_Z": placement.J_by_size[self.classes.dense_size],
            "sparse_row_dof": placement.sparse_diagonal_row_dof,
            "sparse_row_z": placement.sparse_diagonal_row_z,
            "fused_limit_active": placement.fused_diagonal_limit_active,
            "fused_limit_lower_rhs": placement.fused_diagonal_limit_lower_rhs,
            "fused_limit_upper_rhs": placement.fused_diagonal_limit_upper_rhs,
            "diagonal_inverse_mass": self.factor.diagonal_inverse_mass,
            "fused_limit_cfm": self.options.pgs_cfm,
            "fused_limit_lower_lambda": placement.fused_diagonal_limit_lower_lambda,
            "fused_limit_upper_lambda": placement.fused_diagonal_limit_upper_lambda,
        }

    def _launch(self, launch: SweepLaunch, operands: dict, stream=None) -> None:
        """Launch a sweep with each of its parameters read by name: its own operands first, then the step's."""
        kernel, named = launch.kernel, launch.named()
        named.update(zip(("readout",), self.readout.operands(kernel), strict=False))
        wp.launch_tiled(
            kernel,
            dim=[launch.grid],
            inputs=[named[arg.label] if arg.label in named else operands[arg.label] for arg in kernel.adj.args],
            block_dim=launch.block,
            device=self.model.device,
            stream=stream,
        )

    def solve(self, v_hat: wp.array, v_out: wp.array) -> None:
        """Launch the step's sweeps from the predictor ``v_hat`` into ``v_out``: each claim's sweep over its worlds,
        in line, or on the claims' streams where they are open (:meth:`open_streams`), then the step's one leave."""
        self.join_claims()
        if self.options.pgs_iterations <= 0 and not self.seeded:
            return
        operands = self._operands(v_hat, v_out)
        # The sweeps' change: into the walk's du, from zero, or (the support placement's sparse sweep) its own in factor
        # coordinates at each tree's offset (support_du_base).
        factor_change = self.support_du_base is not None
        if self.leaves and not factor_change:
            self.factor.walk_view().du.zero_()
        # Under a readout, count the worlds without rows of any family, which no sweep enters.
        self.readout.count_skipped(self.placement.constraint_count, self.placement.mf_constraint_count)
        streams = self.claim_streams
        if not streams:
            for launch in self.launches:
                self._launch(launch, operands)
        else:
            main = wp.get_stream(self.model.device)
            main.record_event(self.claims_ready)
            first, *beside = self.claim_groups
            for launch in first:
                self._launch(launch, operands)
            for stream, group, done in zip(streams, beside, self.claims_done, strict=True):
                stream.wait_event(self.claims_ready)
                with wp.ScopedStream(stream, sync_enter=False):
                    for launch in group:
                        self._launch(launch, operands)
                stream.record_event(done)
                main.wait_event(done)
        if self.leaves:
            # The step's one leave: every walking tree's change in the walk's coordinates (the sparse sweep's in its
            # factor coordinates).
            if factor_change:
                delta = self.placement.sparse_factor_velocity_delta
                self.factor.leave(v_out, delta.reshape((delta.size,)), self.support_du_base)
            else:
                self.factor.leave(v_out)

    @staticmethod
    def patch_frame_built(contacts):
        """Whether this step builds a patch frame, under the contact row stage's own condition."""
        return (
            contacts is not None
            and getattr(contacts, "rigid_contact_count", None) is not None
            and contacts._rigid_contact_rows > 0
        )

    def finish_patches(self, state_in, state_out, contacts, dt):
        """Publish persistent patch history from the integrated public state."""
        if contacts is None:
            # A memset of today's rows: a capture of it is recaptured after growth.
            self.placement.contact_pool.note_unbound()
            self.laws.friction_patches.previous.valid.zero_()
            return
        # A step without a frame holds no patch rows; it stores the last frame again.
        routes = self.patch_routes() if self.patch_frame_built(contacts) else patch_routes(())
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

    def patch_routes(self):
        """Pack the routes that can hold patch rows into one launch argument."""
        return patch_routes(row for row in self.patch_row_arrays() if row[1] is not None)

    def patch_row_arrays(self):
        """Yield the contact routes that can hold patch rows: ``(route, parents, mu, impulses, first)``."""
        # The line's rows hold every contact but the free bodies' where they keep their own run
        # (Placement.free_contact_rows): with only free response bodies there, the allocator routes every responsive
        # contact through the free bodies' rows. Dense bilateral rows can still exist, but have no patch links or
        # impulses.
        if self.forest.has_non_free_articulations or not self.placement.free_contact_rows:
            yield (
                0,
                self.placement.row_parent,
                self.placement.row_mu,
                self.placement.impulses,
                self.placement.dense_first,
            )
        if self.placement.free_contact_rows:
            # The free bodies' rows: the row store's, where their slots and parents count too.
            yield (
                1,
                self.placement.row_parent,
                self.placement.row_mu,
                self.placement.impulses,
                self.placement.dense_first,
            )

    def carry(self, contacts) -> None:
        """Write the warm start's carry, one record by contact identity which the next step's gather reads: each
        contact's normal and its normal row's impulse (:func:`.warm_start.carry_contacts`), over every row the contact
        stream ever used. Without contacts the record carries nothing."""
        placement = self.placement
        carry = bind_carry(placement)
        pool = placement.contact_pool
        if contacts is None:
            # A memset of today's rows: a capture of it is recaptured after growth.
            pool.note_unbound()
            carry.impulse.zero_()
            return
        used = pool.rows_used if contacts._rigid_contact_pool is pool else None
        pool.launch(
            carry_contacts,
            pool.launch_rows(),
            inputs=[bind_contact_routes(placement, contacts), placement.dense_first, placement.row_type],
            outputs=[placement.impulses, carry],
            extent=pool.ready_rows if used is None else used,
            max_blocks=placement.contact_capacity_blocks,
        )

    def clear_history(self, world_mask: wp.array | None) -> None:
        """Discard the masked worlds' impulses and their contacts' carry (every world's without a mask), independently
        of latched capacity status."""
        if self.forest.world_count == 0 or not self.options.pgs_warmstart:
            return
        placement = self.placement
        wp.launch(
            reset_world_impulses,
            dim=self.forest.world_count,
            inputs=[placement.dense_first, world_mask],
            outputs=[placement.impulses],
            device=self.model.device,
        )
        # Each contact's world is the allocator's of the carried step until the next step allocates.
        pool = placement.contact_pool
        pool.launch(
            forget_carried_contacts,
            pool.launch_rows(),
            inputs=[placement.contact_world, world_mask],
            outputs=[bind_carry(placement)],
            extent=pool.ready_rows,
        )

    def forces(self, contacts, contact_f) -> None:
        """Write the linear contact forces of the step's solve into ``contacts.rigid_contact_force`` and the spatial
        ``contact_f`` (the ``CONTACT_F`` observable; rows past the contacts and soft rows zero), and into the
        deprecated ``contacts.force`` when allocated."""
        contact_f.zero_()
        if contacts is None or contacts.rigid_contact_count is None:
            return

        dt = self.factor.last_step_dt
        inv_dt = 0.0 if dt is None or dt <= 0.0 else 1.0 / dt
        pool = self.placement.contact_pool
        rows = pool.launch_rows()
        pool.launch(
            compute_contact_linear_force_from_impulses,
            rows,
            inputs=[
                self.placement.dense_first,
                self.placement.body_base,
                bind_contact_routes(self.placement, contacts),
                self.placement.impulses,
                self.placement.constraint_count,
                self.placement.mf_constraint_count,
                self.placement.row_type,
                self.placement.row_parent,
                int(self.options.enable_contact_friction),
                inv_dt,
            ],
            outputs=[contacts.rigid_contact_force],
            extent=contacts.rigid_contact_count,
            max_blocks=self.placement.contact_capacity_blocks,
        )

        for output in (contact_f, contacts.force):
            if output is None or (output is contacts.force and output.ptr == contact_f.ptr):
                continue
            pool.launch(
                pack_contact_linear_force_as_spatial,
                rows,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_force,
                ],
                outputs=[output],
                extent=contacts.rigid_contact_count,
                max_blocks=self.placement.contact_capacity_blocks,
            )

    def record(self, plan) -> dict:
        """The sweep's route records, from the step's plan (:class:`.step_plan.StepPlan`)."""
        row, classes, forest = plan.route, self.classes, self.forest
        diagonal, packed = classes.diagonal, classes.packed
        # The class plan's choice holds native sparse factors' options and their reasons (class_plan.Packed).
        options, unselected = dict(packed.sweep) if packed else {}, "Native sparse factors are not selected."
        sparse = {
            "size": json_value(row.sizes["packed"]),
            "reason": row.reasons["packed"],
            "velocity_limits": bool(self.sparse_velocity_limits),
            "active_rows": bool(options.get("active_rows")),
            "active_rows_reason": packed.active_rows_reason if packed else unselected,
            "support_table": bool(self.placement.sparse_row_support.enabled),
            "contact_triplets": bool(options.get("contact_triplets")),
            "contact_triplets_reason": packed.contact_triplets_reason if packed else unselected,
        }
        if packed:
            sparse.update(
                {
                    "max_row_support": json_value(packed.max_support),
                    "solve_lanes": int(options["lanes_per_world"]),
                    "local_factors": bool(options["local_factors"]),
                    "units_kernel": kernel_key(self.support_units_kernel),
                    "writer_kernel": kernel_key(self.support_rows),
                    "leave_kernel": kernel_key(self.factor.walk_leave),
                    "sweep_kernel": kernel_key(self.sparse_kernel),
                    # The sweep's launch bounds from its shared memory and register budget (threads, minimum blocks).
                    "launch_bounds": json_value(self.sparse_kernel.options.get("launch_bounds")),
                }
            )
        owner = self.resolved_worlds_owner
        published = bool(owner is not None and owner.published)
        if owner is None:
            twists = None
        elif published:
            twists = "The held-tree predictor publishes body twists for every responding articulation."
        else:
            twists = "The classifier builds body twists itself: a responding articulation has no twist-publishing tree."
        count = int(forest.free_rigid_body_count)
        owners = []
        if count:
            if row.support == "factor":
                owners.append("sparse_free_rows")
            owners.append("matrix_free_rows")
        return {
            "sparse_diagonal": {
                "enabled": row.support == "diagonal",
                "response_dofs": json_value(row.sizes["diagonal"]),
                "dense_dofs": int(row.sizes["dense"] or 0),
                "contact_units": bool(diagonal is not None and diagonal.units),
                "speculative_contact_batches": bool(diagonal is not None and diagonal.speculative_batches),
                "fused_joint_limits": bool(classes.fused_sizes),
                "reason": row.reasons["diagonal"],
            },
            "sparse_mass_matrix": sparse,
            "sweep": self._record_sweep(plan),
            "resolved_worlds": {
                "enabled": owner is not None,
                "reason": self.resolved_worlds_reason,
                "publishes_predicted_twists": published,
                "twists_reason": twists,
            },
            "free_rigid_bodies": {
                "count": count,
                "mixed_worlds": bool(forest.has_mixed_contacts),
                # Whether their contacts keep a run of their own (the colored worlds'), or are rows of the line.
                "contact_run": bool(self.placement.free_contact_rows),
                "owners": owners,
                "reason": "No free rigid bodies."
                if not count
                else "Owners that can solve free-body rows, in claim order; contacts are routed per world at run time.",
            },
        }

    def _record_sweep(self, plan) -> dict:
        """``routes.sweep``: its kernels, the readout's counted kernels and the exit law (:mod:`.solve_policy`)."""
        placement = self.placement
        counted = self.readout.kernels
        readout = self.options.diagnostics.solve_readout
        return {
            "route": plan.solve.route.name,
            "reason": plan.solve.route.reason,
            "iterations": int(self.options.pgs_iterations),
            # The streamed rows' records, which the rows' writers store where a sweep streams a row.
            "row_records": {
                "written": bool(placement.writes_records),
                "first": int(placement.record_first),
                "reason": "The rows' writers store the streamed rows' records."
                if placement.writes_records
                else "Skipped: no sweep streams a row's metadata.",
            },
            **{name: plan.kernels[name] for name in ("sparse", "sparse_diagonal", "mf_gs")},
            "mf_gs_storage": self._record_storage(),
            "mf_gs_active_reason": self.general_active_reason,
            "mf_colored": self._record_colored(plan),
            "tiers": self._record_tiers(plan.route),
            "readout": {
                "mode": readout if counted else "off",
                "counted": dict(sorted(counted.items())),
                # Whether the solves of worlds without rows, which the matrix-free sweeps do not enter, count as skipped.
                "counts_skipped": bool(counted) and self.readout.counts_skipped,
                "reason": "The solve readout is off."
                if readout == "off"
                else "Each launch of a counted kernel solves its worlds.",
            },
            "exit_law": {
                "law": "rounding_floor",
                "quiet_units": QUIET_UNITS,
                "unit_roundoff": json_value(UNIT_ROUNDOFF),
                "reason": (
                    "Every sweep stops a world once each row update's change of its own velocity, |dlambda| A_ii, is "
                    f"within {QUIET_UNITS} rounding units of the magnitude its update combines: the right-hand side, the "
                    "terms of its velocity sum and its own impulse times A_ii."
                ),
            },
        }

    def _record_storage(self) -> dict | None:
        """The storage classes of the general sweep, chosen by :func:`.execution_plan._select_mfgs_storage`."""
        storage = self.mf_gs_storage
        if storage is None:
            return None
        places = storage.places
        names = {"metadata": places.metadata, "mf_impulses": places.impulses, "dense_impulses": places.dense_impulses}
        record = {name: "resident" if place is Place.SHARED else "streamed" for name, place in names.items()}
        return {**record, "reason": storage.reason}

    def _record_colored(self, plan) -> dict | None:
        """The colored mapping of the matrix-free sweep: its kernel and the worlds it takes, or ``None`` unbuilt."""
        if self.mf_colored_kernel is None:
            return None
        units = self.colored_units
        rule = units.rule
        return {
            "kernel": plan.kernels["mf_colored"],
            "min_units": int(rule.min_units),
            "parallelism": json_value(rule.parallelism),
            # The waves the capable worlds take on the device's SMs, which multiply the units per color a world needs.
            "waves": int(rule.waves),
            "units_per_color": json_value(rule.units_per_color),
            # The worlds whose free bodies admit the rule, each with a storage slot, and that storage's bytes.
            "capable_worlds": int(units.slot_count),
            "storage_bytes": int(units.nbytes),
            "reason": (
                f"At every pack the unit builder gives the colored mapping each capable world (at least "
                f"{rule.units_per_color:g} free bodies: a world's units per color are at most its free bodies) of at "
                f"least {rule.min_units} contact units and {rule.units_per_color:g} units per color: it sweeps a color's "
                "units at once, a block each; the others keep the one-warp mapping (the measured crossover of their "
                f"solve times, {rule.parallelism:g} units per color in one wave of colored worlds on the device, here "
                f"{rule.waves})."
            ),
        }

    def _record_tiers(self, row) -> dict | None:
        tiers = list(self.tiers)
        if not tiers:
            return None
        return {
            "family": "parallel",
            # Why the class takes the tiers (case K4: the in-kernel response's shape).
            "inkernel_response_reason": row.reasons["tiers"],
            # Every built tier projects each friction pair on its patch ring's pooled normal load (patch friction).
            "linked_rows": all(tier.config.linked_rows for tier in tiers),
            # The last-word velocity-limit pass after the final sweep (linked tiers with joint velocity limits).
            "velocity_limit_pass": all(tier.config.velocity_limit_pass for tier in tiers),
            # The largest tier budget; each tier records its own (the equal-accuracy rule "auto", or "pinned").
            "sweeps": max(tier.config.levels.budget for tier in tiers),
            "sweeps_rule": self.par_sweeps_rule,
            "sweeps_reason": self.par_sweeps_reason,
            "tiers": [
                {
                    "rows": int(rows),
                    "min_rows": int(min_rows),
                    "sweeps": tier.config.levels.budget,
                    # The one writer's sink for the tier's worlds: its prologue walks their seeds, or reads stored Z.
                    "sink": "emit" if tier.config.emit else "store",
                    "kernel": kernel_key(tier.kernel),
                }
                for (rows, min_rows), tier in zip(self.par_tiers, tiers, strict=True)
            ],
        }

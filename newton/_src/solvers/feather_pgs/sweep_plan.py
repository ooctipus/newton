# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweep (BACKBONE 2): which worlds each schedule claims, the impulses it starts from and carries, the launches.

:class:`Sweep` owns the solve of one step:

- the owners that sweep their own worlds, built from the class plan's case (:mod:`.class_plan`), with their kernels
  and streams (:meth:`Sweep.build`, :meth:`Sweep.open_streams`);
- the claims: the worlds the row pass left to each owner (:meth:`Sweep.claim`), and the resolved worlds;
- the warm start: the normal impulses gathered by contact identity (:meth:`Sweep.warm_start`), and what a step
  carries to the next (:meth:`Sweep.carry`);
- the launches, the claims in order and then the overflow row (:meth:`Sweep.solve`);
- the contact forces the impulses export (:meth:`Sweep.forces`).

The sweeps' mathematics lives in :mod:`.sweep` and the route modules; the schedule of case K2 in :mod:`.sweep_levels`.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np
import warp as wp

from ...core.execution import Queue, shape
from ...core.memory import Place, arch_budget, device
from . import (
    packed_rows,
    sweep_levels,
    sweep_routes,
    sweep_routes_diag,
    sweep_routes_local,
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
from .kernel_template import SweepTier
from .solve_policy import SolvePolicy, SolveReadoutRecord
from .sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from .sparse_mass_matrix import _SparseRowSupport, row_supports
from .sweep import FRICTION_METRIC_FLOATS
from .sweep_levels import (
    _get_build_independent_sparse_contact_groups_kernel,
    _get_mark_independent_sparse_contact_candidates_kernel,
)
from .tree_scan import classes_half
from .warm_start import (
    bind_carry,
    carry_contacts,
    forget_carried_contacts,
    gather_carried_normals,
    reset_world_impulses,
)

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
"""Where spin rows may appear, each friction triple's metric, formed once per launch by the sweeps that visit a triple's
rows one by one, the matrix-free, the sparse-diagonal and the paired factor sweeps (:func:`.sweep.friction_metric_stage`):
a slot per four rows of the row store's line, a triple's rows."""


__all__ = ["Sweep"]


class Sweep:
    """The solve of one step, with the owners that claim their worlds (see the module's docstring).

    Args:
        model: The solver's model.
        options: The checked options.
        forest: The forest plan.
        laws: The row laws.
        placement: The placement.
        factor: The tree factor.
        classes: The class plan (:class:`.class_plan.ClassPlan`): the case and its choice, which the owners build.
        pins: The test-only route pins.
    """

    def __init__(self, model, options, forest, laws, placement, factor, classes, pins):
        self.model = model
        self.options = options
        self.forest = forest
        self.laws = laws
        self.placement = placement
        self.factor = factor
        self.case = classes.case
        """The class's case (:attr:`.class_plan.ClassPlan.case`)."""
        self.pins = pins
        # Every sweep compiles the exit law and the policy's readout.
        self.policy = SolvePolicy(readout=options.diagnostics.solve_readout)
        self.readout = SolveReadoutRecord(self.policy)

    def place_diagonal(self, choice) -> None:
        """Case K2, a tree beside a star: the sparse diagonal and the fused limits of its star, from the class plan's
        choice (:class:`.class_plan.Diagonal`, ``None`` where the class is another case)."""
        model, placement = self.model, self.placement
        chosen = choice is not None
        placement.sparse_diagonal_contact_solve = chosen
        # Limit rows belong to this specialized owner; without it the ordinary rows keep them.
        placement.fused_diagonal_joint_limit_sizes = choice.fused_sizes if chosen else frozenset()
        placement.fused_diagonal_joint_limits = bool(placement.fused_diagonal_joint_limit_sizes)
        placement.fused_diagonal_limit_dof_mask_host = (
            choice.fused_mask if chosen else np.zeros(model.joint_dof_count, dtype=np.int32)
        )
        placement.sparse_diagonal_response_size = choice.size if chosen else None
        placement.sparse_diagonal_dense_size = choice.dense_size if chosen else 0
        # A contact is a unit: a normal row, followed by its friction rows only when its region holds an anchor.
        placement.sparse_diagonal_contact_units = bool(
            chosen and self.options.enable_contact_friction and self.options.contact_friction_gap_threshold == math.inf
        )
        placement.sparse_diagonal_speculative_contact_batches = bool(
            placement.sparse_diagonal_contact_units and placement.sparse_diagonal_dense_size + 2 <= 16
        )
        placement.sparse_diagonal_dense_offsets = wp.array(
            choice.dense_offsets if chosen else np.zeros(1, dtype=np.int32), dtype=wp.int32, device=model.device
        )
        placement.sparse_diagonal_dense_groups = wp.array(
            choice.dense_groups if chosen else np.zeros(1, dtype=np.int32), dtype=wp.int32, device=model.device
        )
        # Each world's dense arm, the tree the walk whitens and leaves (FL3.5).
        placement.sparse_diagonal_dense_arts_host = choice.dense_arts if chosen else np.zeros(0, dtype=np.int32)

    def place_packed(self, choice) -> None:
        """Case K1, a single tree: native sparse factors with independent rigid-body response blocks, from the class
        plan's choice (:class:`.class_plan.Packed`, ``None`` where the class is another case)."""
        model = self.model
        self.placement.sparse_mass_matrix_size = None
        self.sparse_velocity_limits = False
        self.support_rows = self.support_units_kernel = self._support_sink = None
        self.support_units = None
        self.support_du_base = None
        self.leaves = False
        self.placement.sparse_row_support = _SparseRowSupport()
        self.sparse_contact_triplets = False
        # Route details for get_path_selection(), set where the sparse owner is admitted.
        self.sparse_active_rows = False
        self.sparse_active_rows_reason = "Native sparse factors are not selected."
        self.sparse_contact_triplets_reason = "Native sparse factors are not selected."
        # The admitted contact units' block records.
        self.placement.sparse_contact_records = None
        self.placement.friction_metric = None
        self.placement.sparse_max_row_support = None
        if choice is None:
            return
        sweep, max_support = dict(choice.sweep), choice.max_support
        self.placement.sparse_mass_matrix_size = choice.size
        self.sparse_active_rows = sweep["active_rows"]
        self.sparse_active_rows_reason = choice.active_rows_reason
        self.sparse_contact_triplets = sweep["contact_triplets"]
        self.sparse_contact_triplets_reason = choice.contact_triplets_reason
        self.sparse_solve_lanes = sweep["lanes_per_world"]
        self.sparse_local_factors = sweep["local_factors"]
        self.placement.sparse_max_row_support = max_support
        self.sparse_velocity_limits = choice.velocity_limits
        self.factor.allocate_sparse(choice.plan)
        self.placement.sparse_body_dof_mask = wp.array(choice.body_masks, dtype=wp.uint64, device=model.device)
        if sweep["use_support_table"]:
            self.placement.sparse_row_support = row_supports(
                choice.support_classes, self.forest.world_count, self.placement.dense_max_constraints, model.device
            )
        if self.sparse_contact_triplets or self.laws.spin_rows:
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
        # Contact torsion: each friction contact's spin row is the fourth of its unit.
        torsion = {"torsion": True} if self.laws.spin_rows else {}
        self.sparse_kernel = self.kernel(
            "sparse_pgs",
            sweep_routes_sparse.sparse_kernel,
            self.placement.dense_max_constraints,
            self.forest.max_world_dofs,
            max_support,
            **sweep,
            velocity_limits=self.sparse_velocity_limits,
            warm_start=self.options.pgs_warmstart,
            linked_rows=True,
            device_arch=model.device.arch,
            **torsion,
        )

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
        size = self.placement.sparse_mass_matrix_size
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

    def place_paired(self, choice, reason: str | None) -> None:
        """Case K5, a tree beside one free body: the paired response and its factor coordinates, from the class plan's
        choice (:class:`.class_plan.Paired`, ``None`` where the class is another case, ``reason`` saying why)."""
        model, forest = self.model, self.forest
        self.paired_response_reason = reason
        self.paired_factor_coordinates = choice is not None
        self.paired_response_primary_size = choice.primary if choice is not None else None
        self.paired_response_secondary_size = choice.secondary if choice is not None else None
        # Patch contacts are units of one row, or three (four with a spin row) when they own an anchor;
        # the factor sweep walks them with the generic path's row order and
        # arithmetic, solving each tangent pair once.
        self.factor_coordinate_contact_units = bool(self.paired_factor_coordinates and not self.laws.set.compliance)
        self.paired_factor_primary_offsets_by_world = None
        self.paired_factor_secondary_offsets_by_world = None
        self.paired_factor_primary_arts_by_world = None
        if choice is None:
            return
        primary_arts = forest.group_to_art_host[choice.primary]
        secondary_arts = forest.group_to_art_host[choice.secondary]
        primary_offsets = np.empty(forest.world_count, dtype=np.int32)
        secondary_offsets = np.empty(forest.world_count, dtype=np.int32)
        # Each world's articulation: where the paired components' check reads its walk (paired_components).
        primary_by_world = np.full(forest.world_count, -1, dtype=np.int32)
        for primary_group, primary_art in enumerate(primary_arts):
            world = int(forest.articulation_world[primary_art])
            secondary_art = int(secondary_arts[int(choice.groups[primary_group])])
            primary_by_world[world] = int(primary_art)
            dof_start = forest.articulation_dof_start_host
            primary_first = dof_start[primary_art] < dof_start[secondary_art]
            primary_offsets[world] = 0 if primary_first else choice.secondary
            secondary_offsets[world] = choice.primary if primary_first else 0
        self.paired_factor_primary_offsets_by_world = wp.array(primary_offsets, dtype=wp.int32, device=model.device)
        self.paired_factor_secondary_offsets_by_world = wp.array(secondary_offsets, dtype=wp.int32, device=model.device)
        self.paired_factor_primary_arts_by_world = wp.array(primary_by_world, dtype=wp.int32, device=model.device)

    def place_tiers(self, choice) -> None:
        """Case K4, the Jacobi tiers: their row bound and their in-kernel response's trees, from the class plan's
        choice (:class:`.class_plan.Tiers`, ``None`` where the class is another case). ``ink_rows`` is 0 without
        tiers; the keyed rows read it before the tiers are built."""
        placement = self.placement
        placement.ink_rows, placement.ink_sizes = (choice.rows, choice.sizes) if choice is not None else (0, None)
        if choice is not None:
            placement.ink_meta = wp.array(choice.meta.reshape(-1), dtype=wp.int32, device=self.model.device)

    def _tier_frame(self) -> tuple[int, int]:
        """Each of the tiers' trees' slip capacity (:meth:`.tree_factor.TreeFactor.slip_capacity`): the release its rows
        and leave compile, none where no tree of that size holds a limited drive."""
        na, nb = self.placement.ink_sizes[0], self.placement.ink_sizes[1]
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
        pinned = self.pins.get("tier_sweeps")
        if pinned is not None:
            return [int(pinned)] * len(tiers), "pinned", f"A test pin sets every tier's budget (tier_sweeps={pinned})."
        reason = (
            f"auto: a tier serving up to R rows sweeps ceil({sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION} * "
            f"pgs_iterations * sqrt(R / 32)) times with pgs_iterations={int(self.options.pgs_iterations)}, at most "
            f"{sweep_routes_tiers.TIER_SWEEPS_CAP}; a world stops earlier under the sweeps' exit law."
        )
        return [sweep_routes_tiers.tier_sweeps(rows, self.options.pgs_iterations) for rows, _ in tiers], "auto", reason

    def kernel(self, route: str, factory, *args, readout_budget: int | None = None, **kwargs):
        """Build a sweep under the solve policy and allocate its readout counters, outside any graph capture.

        Args:
            route: Owner of the sweep in the records, such as ``"sparse_pgs"``.
            factory: The sweep factory, which takes the policy as ``solve_policy`` and the laws' constants as
                ``weighted`` and ``material``, and returns a kernel or a :class:`SweepTier`.
            *args: Positional arguments of the factory.
            readout_budget: The sweep budget the kernel compiles in (a parallel tier's), or ``None`` for the
                solver's iteration counts.
            **kwargs: Keyword arguments of the factory.
        """
        policy = self.policy
        # The laws' constants, compiled into every sweep: weighted rows and material contacts.
        laws = {"weighted": self.laws.set.weighted, "material": bool(self.options.contact_compliance)}
        built = factory(*args, solve_policy=policy, **laws, **kwargs)
        kernel = built.kernel if isinstance(built, SweepTier) else built
        budget = int(self.options.pgs_iterations if readout_budget is None else readout_budget)
        self.readout.register(
            kernel,
            route,
            readout=policy.readout if policy.counts else None,
            world_count=int(self.forest.world_count),
            budget=max(budget, 1),
            device=self.model.device,
        )
        return built

    def build(self) -> None:
        """Build the paired response and the sweeps."""
        model = self.model
        device_arch = model.device.arch
        from . import paired_components  # noqa: PLC0415

        # The sweeps start from the impulses the rows carry: the warm start's, the closed loop's projection.
        self.seeded = bool(self.options.pgs_warmstart or self.factor.eliminates)
        self.paired_components = None
        self.paired_components_reason = paired_components.decline(self)
        if self.paired_components_reason is None:
            self.paired_components = paired_components.PairedComponents(self)
            self.paired_components_reason = (
                "Admitted: each paired world splits into independent articulation and free-body components."
            )
        self.paired_factor_kernel = (
            self.kernel(
                "paired_factor",
                sweep_routes_local.paired_factor_kernel,
                self.placement.dense_max_constraints,
                self.forest.max_world_dofs,
                self.paired_response_primary_size,
                self.paired_response_secondary_size,
                device_arch,
                independent_components=self.paired_components is not None,
                contact_units=self.factor_coordinate_contact_units,
                torsion=self.laws.spin_rows,
                pitch=self.placement.row_pitch,
                seeded=self.seeded,
            )
            if self.paired_factor_coordinates
            else None
        )

        sparse_contact_schedule = self.placement.sparse_diagonal_contact_units
        self.sparse_contact_marker = (
            _get_mark_independent_sparse_contact_candidates_kernel(
                self.placement.sparse_diagonal_response_size,
                device_arch,
                torsion=self.laws.spin_rows,
            )
            if sparse_contact_schedule
            else None
        )
        self.sparse_contact_schedule = (
            _get_build_independent_sparse_contact_groups_kernel(
                self.placement.dense_max_constraints,
                self.forest.max_world_dofs,
                device_arch,
                build_serial_contacts=self.placement.sparse_diagonal_speculative_contact_batches,
                torsion=self.laws.spin_rows,
            )
            if sparse_contact_schedule
            else None
        )
        self.sparse_diagonal_kernel = (
            self.kernel(
                "sparse_diagonal",
                sweep_routes_diag.sparse_diagonal_kernel,
                self.placement.dense_max_constraints,
                self.forest.max_world_dofs,
                self.placement.sparse_diagonal_dense_size,
                device_arch,
                speculative_contact_batches=self.placement.sparse_diagonal_speculative_contact_batches,
                contact_units=self.placement.sparse_diagonal_contact_units,
                torsion=self.laws.spin_rows,
            )
            if self.placement.sparse_diagonal_contact_solve
            else None
        )
        row_by_row = self.sparse_diagonal_kernel is not None or self.paired_factor_kernel is not None
        if self.laws.spin_rows and (self.placement.mf_sweeps or row_by_row):
            # Static storage for graph capture: the friction triples' metrics, formed per launch by the sweeps that
            # visit a triple's rows one by one (sweep.friction_metric_stage).
            self.placement.arrays.allocate(
                FRICTION_METRIC,
                world=self.forest.world_count,
                contact=(self.placement.row_pitch + 3) // 4,
                metric=FRICTION_METRIC_FLOATS,
            )
        self.mf_gs_kernel = None
        # The colored mapping of the matrix-free sweep and its units, when a world can take it.
        self.mf_colored_kernel = None
        self.colored_units = None
        # Storage classes of the MF-GS kernel (_MFGSStorage); recorded by get_path_selection().
        self.mf_gs_storage = None
        # The Jacobi tiers ahead of the matrix-free sweep (SweepTier), and their keyed rows' release.
        self.tiers = []
        self.tier_worlds = None
        if self.placement.mf_sweeps:
            # A test pin streams the impulses (storage is part of the kernel specialization key); the selection
            # sizes the resident row metadata with its choice.
            force_stream_mf_impulses = self.pins.get("mf_storage") == "streamed"
            on_cpu = Place.GLOBAL if force_stream_mf_impulses else Place.SHARED
            storage = _MFGSStorage(
                sweep_routes.RowPlaces(Place.GLOBAL, Place.GLOBAL, on_cpu),
                "Without CUDA no shared-memory storage applies.",
            )
            if model.device.is_cuda:
                storage = _select_mfgs_storage(
                    functools.partial(
                        sweep_routes.mf_gs_staging,
                        self.placement.dense_max_constraints,
                        self.placement.mf_max_constraints,
                        self.forest.max_world_dofs,
                    ),
                    arch_budget(device_arch),
                    block_count=self.forest.world_count,
                    sm_count=int(model.device.sm_count),
                    force_stream_mf_impulses=force_stream_mf_impulses,
                )
            self.mf_gs_storage = storage
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
                found = device(model.device)
                rule = rule.at_scale(capable.size, shape(sweep_routes.COLORED_WORK, found).blocks_per_sm * found.sms)
                capable = sweep_units.capable_worlds(
                    self.forest.free_rigid_per_world, self.placement.mf_max_constraints, rule
                )
            colored = bool(model.device.is_cuda and capable.size > 0)
            mf_gs_options = {
                "pitch": self.placement.row_pitch,
                "has_dense_velocity_limit_rows": self.options.enable_joint_velocity_limits,
                "places": storage.places,
                "contact_torsion": self.laws.spin_rows,
                "factor_coordinates": self.paired_factor_coordinates,
                "skip_tier_rows": parallel_rows,
                "independent_components": self.paired_components is not None,
                "colored_worlds": colored,
                "max_row_dofs": self.placement.packed_support,
                "seeded": self.seeded,
                "wide_rows": self.placement.packed_dense == packed_rows.WIDE,
            }
            mf_gs_shape = (
                self.placement.dense_max_constraints,
                self.placement.mf_max_constraints,
                self.forest.max_world_dofs,
                device_arch,
            )
            self.mf_gs_kernel = self.kernel(
                "mf_gs",
                sweep_routes.mf_gs_kernel,
                *mf_gs_shape,
                **mf_gs_options,
            )
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
                )
                self.mf_colored_kernel = self.kernel(
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
                width = sweep_routes_tiers.TIER_ROWS
                tiers = [(min(first + width, parallel_rows), first) for first in range(0, parallel_rows, width)]
                self.par_tiers = list(tiers)
                # Each tier walks its bucket of the level builder's world list (sweep_levels.list_worlds).
                self.tier_worlds = Queue(len(tiers) * self.forest.world_count, model.device, buckets=len(tiers))
                tier_sweeps, self.par_sweeps_rule, self.par_sweeps_reason = self._parallel_tier_sweeps(tiers)
                for tier_index, ((rows_, min_rows_), sweeps_) in enumerate(zip(tiers, tier_sweeps, strict=True)):
                    tier = self.kernel(
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
                        linked_rows=True,
                        velocity_limit_pass=self._parallel_velocity_limit_pass(),
                        frame=self._tier_frame(),
                        classes=self.factor.walk_classes,
                        pitch=self.placement.row_pitch,
                        queued=True,
                        **({"torsion": True} if self.laws.spin_rows else {}),
                    )
                    # On its own list, a tier takes a block a world of the list's capacity (its work's shape).
                    grid = shape(dataclasses.replace(tier.work, items=self.forest.world_count), device(model.device))
                    self.tiers.append(dataclasses.replace(tier, tier=tier_index, grid=grid.grid))
            # No matrix-free sweep enters a world without rows; the readout counts those solves as skipped.
            self.readout.register_skipped(world_count=int(self.forest.world_count), device=model.device)
        # The sweeps that walk their rows (and the Jacobi tiers, which always leave by the walk) add their change to the
        # walk's du in its coordinates; the step's one leave walks every tree down (factor.walk.walk_leave_kernel).
        self.leaves = self.factor.walk_arts is not None and bool(self.placement.walks or self.tiers)

    def open_streams(self, *, overlap: bool) -> None:
        """Open the claims' streams: the sweeps that own their worlds run on them beside the matrix-free sweep.

        The claims touch nothing the matrix-free sweep touches, so they fork after the rows are published and join
        before the leave:

        - the paired factor sweep over independent components (admitted on CUDA, under the factor-coordinate sweep's
          plain policy), whose coordinates of a split world are disjoint from the general sweep's;
        - the Jacobi tiers, whose worlds hold no free-body row, the worlds the matrix-free sweep returns from. A tier's
          worlds are those of its row count, so each tier is a claim of its own.

        A stream a claim: the first takes the paired factor sweep where it forks, then the tiers' lists and the first
        tier; every other tier its own, once the lists are written.

        The other side streams (inertia, dynamics, memset) read no contact rows nor the contact pool's ready count:
        they run factorizations and articulation data. The claims do (the keyed tiers), between the fork and the join
        only, and the launch stream, which the pool registers, waits for the join before it goes on. A trim that waits
        for the launch stream therefore waits for them too (:meth:`ContactPool.add_reader_stream`).
        """
        model = self.model
        self.paired_forks = bool(
            overlap and self.paired_components is not None and self.paired_factor_kernel is not None
        )
        claims = max(int(self.paired_forks), len(self.tiers)) if wp.get_device(model.device).is_cuda and overlap else 0
        self.claim_streams = tuple(wp.Stream(model.device) for _ in range(claims))
        self.claims_ready = wp.Event(model.device) if claims else None
        self.claims_listed = wp.Event(model.device) if claims > 1 else None
        self.claims_done = tuple(wp.Event(model.device) for _ in range(claims))

    def claim(self) -> None:
        """Publish the schedule of the rows the row pass wrote and each owner's worlds (run after every producer)."""
        model = self.model
        if self.placement.sparse_diagonal_contact_units:
            # Unit heads come from the row types and the tangent rows' parents.
            unit_rows = [self.placement.row_type, self.placement.row_parent]
            wp.launch(
                self.sparse_contact_schedule,
                dim=self.forest.world_count * 32,
                inputs=[
                    self.placement.constraint_count,
                    self.placement.dense_first,
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
            # The paired components' signed selector (sweep_routes.mf_gs_source), where they split worlds.
            "component_selector": self.placement.dummy_is_free_rigid
            if self.paired_components is None
            else self.paired_components.selector,
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
            "world_row_type": placement.row_type,
            "world_row_parent": placement.row_parent,
            "world_row_mu": placement.row_mu,
            "mf_constraint_count": placement.mf_constraint_count,
            # The free bodies' headers (their records and packed rows are the line's above).
            "mf_contact_rows_end": placement.mf_contact_rows_end,
            "mf_meta": placement.mf_meta_packed,
            # The sparse sweep's ABI slot of the free bodies' rows, read by no configuration (U10's hunk drops it).
            "mf_Z": placement.packed_Z,
            "iterations": self.options.pgs_iterations,
            "omega": self.options.pgs_omega,
            "friction_metric": placement.friction_metric,
            "v_hat": v_hat,
            "v_out": v_out,
        }

    def _launch(self, kernel, operands: dict, dim: int, block_dim: int, *, stream=None, **named) -> None:
        """Launch a sweep with each of its parameters read by name: ``named`` first, then ``operands``."""
        named.update(zip(("readout",), self.readout.operands(kernel), strict=False))
        wp.launch_tiled(
            kernel,
            dim=[dim],
            inputs=[named[arg.label] if arg.label in named else operands[arg.label] for arg in kernel.adj.args],
            block_dim=block_dim,
            device=self.model.device,
            stream=stream,
        )

    def solve(self, v_hat: wp.array, v_out: wp.array) -> None:
        """Launch the step's sweeps from the predictor ``v_hat`` into ``v_out``: the claims in order, then the rest."""
        if self.options.pgs_iterations <= 0 and not self.seeded:
            return
        operands = self._operands(v_hat, v_out)
        if self.leaves and self.case != "K1":
            # The sweeps add their change into the walk's du from zero.
            self.factor.walk_view().du.zero_()
        if self.case == "K1":
            self._solve_packed(operands)
        elif self.case == "K2":
            self._solve_diagonal(operands)
        elif self.forest.max_world_dofs:
            self._solve_claims(operands)
        if self.leaves:
            # The step's one leave: every walking tree's change in the walk's coordinates (the sparse sweep's in its
            # factor coordinates).
            if self.case == "K1":
                delta = self.placement.sparse_factor_velocity_delta
                self.factor.leave(v_out, delta.reshape((delta.size,)), self.support_du_base)
            else:
                self.factor.leave(v_out)

    def _solve_packed(self, operands: dict) -> None:
        """Case K1: the sparse sweep in factor coordinates, then its velocity change back onto ``v_out``."""
        placement = self.placement
        worlds_per_block = 64 // self.sparse_solve_lanes
        self._launch(
            self.sparse_kernel,
            operands,
            (self.forest.world_count + worlds_per_block - 1) // worlds_per_block,
            64,
            world_count=self.forest.world_count,
            row_dof=placement.sparse_row_dof,
            row_factor=placement.sparse_row_factor,
            factor_velocity_delta=placement.sparse_factor_velocity_delta,
            support=placement.sparse_row_support,
            contact_records=placement.sparse_contact_records,
        )

    def _solve_diagonal(self, operands: dict) -> None:
        """Case K2: the sparse-diagonal sweep over the schedule of :meth:`claim`."""
        placement = self.placement
        dense_size = placement.sparse_diagonal_dense_size
        self._launch(
            self.sparse_diagonal_kernel,
            operands,
            self.forest.world_count,
            32,
            sparse_contact_group_count=placement.sparse_contact_group_count,
            sparse_contact_group_heads=placement.sparse_contact_group_heads,
            sparse_contact_serial_count=placement.sparse_contact_serial_count,
            sparse_contact_serial_normals=placement.sparse_contact_serial_normals,
            dense_offsets=placement.sparse_diagonal_dense_offsets,
            dense_groups=placement.sparse_diagonal_dense_groups,
            walk=self.factor.walk_view(),
            dense_Z=placement.J_by_size[dense_size],
            sparse_row_dof=placement.sparse_diagonal_row_dof,
            sparse_row_z=placement.sparse_diagonal_row_z,
            fused_limit_active=placement.fused_diagonal_limit_active,
            fused_limit_lower_rhs=placement.fused_diagonal_limit_lower_rhs,
            fused_limit_upper_rhs=placement.fused_diagonal_limit_upper_rhs,
            diagonal_inverse_mass=self.factor.diagonal_inverse_mass,
            fused_limit_cfm=self.options.pgs_cfm,
            fused_limit_lower_lambda=placement.fused_diagonal_limit_lower_lambda,
            fused_limit_upper_lambda=placement.fused_diagonal_limit_upper_lambda,
        )

    def _solve_claims(self, operands: dict) -> None:
        """The other cases: the claims that own their worlds (the paired factor sweep's, the Jacobi tiers'), on the
        claims' streams where they are open (:meth:`open_streams`), beside the matrix-free sweep of the rest."""
        mf_gs_kernel = self.mf_gs_kernel
        if mf_gs_kernel is None:
            raise RuntimeError("Matrix-free GS kernel is unavailable for this solver shape")

        # Under a readout, count the worlds without rows of any family, which no sweep enters.
        self.readout.count_skipped(self.placement.constraint_count, self.placement.mf_constraint_count)
        factor_kernel = self.paired_factor_kernel
        components = self.paired_components
        if components is not None:
            components.prepare(self, self.placement.rhs, self.placement.mf_meta_packed)
        streams = self.claim_streams
        main = wp.get_stream(self.model.device) if streams else None

        walked = {}
        if factor_kernel is not None and self.placement.walks:
            walked = {"walk": self.factor.walk_view()}

        def launch_factor_solve():
            self._launch(
                factor_kernel,
                operands,
                (self.forest.world_count + 1) // 2,
                64,
                world_count=self.forest.world_count,
                # The factor rows: the packed rows at their world coordinates.
                factor_rows=self.placement.packed_Z,
                world_mf_constraint_count=(
                    self.placement.mf_constraint_count if components is None else components.selector
                ),
                primary_offset_by_world=self.paired_factor_primary_offsets_by_world,
                secondary_offset_by_world=self.paired_factor_secondary_offsets_by_world,
                **walked,
            )

        # The sweeps leave by the walk (sweep_routes.leave_all) and find a free body's tree in it.
        walk = {} if self.factor.walk_arts is None else {"walk": self.factor.walk_view()}

        # The tiers own the worlds with only dense rows inside their row bound; the general sweep returns early for
        # those and solves the rest.
        lists = self.tier_worlds

        def list_tiers():
            if lists is not None:
                lists.count.zero_()
                wp.launch(
                    sweep_levels.list_worlds,
                    dim=self.forest.world_count,
                    inputs=[
                        self.placement.constraint_count,
                        self.placement.mf_constraint_count,
                        self.par_tiers[-1][0],
                        sweep_routes_tiers.TIER_ROWS,
                        lists.region,
                    ],
                    outputs=[lists.items, lists.count],
                    device=self.model.device,
                )

        def launch_tier(tier):
            # A tier grid-strides over its bucket of the world list.
            count, worlds = lists.bucket(tier.tier)
            named = dict(walk, general_world_count=count, general_worlds=worlds, general_world_grid_stride=tier.grid)
            named.update(self.placement.inkernel_response_inputs())
            if tier.kinetic_rows:
                keyed = self.placement.keyed_rows.parallel_arguments()
                named.update(zip(("kinetic_rows", "patch_law"), keyed, strict=False))
            self._launch(tier.kernel, operands, tier.grid, tier.work.block, **named)

        # The claims in their order, the paired factor sweep's worlds then the tiers'. A paired factor sweep without
        # independent components shares worlds with the general sweep, so it runs ahead of the fork.
        fork_factor = bool(streams) and self.paired_forks
        if factor_kernel is not None and not fork_factor:
            launch_factor_solve()
        if streams:
            main.record_event(self.claims_ready)
            for stream in streams:
                stream.wait_event(self.claims_ready)
        if not streams:
            list_tiers()
            for tier in self.tiers:
                launch_tier(tier)
        colored_kernel = self.mf_colored_kernel
        colored = {} if colored_kernel is None else self.colored_units.operands()
        self._launch(mf_gs_kernel, operands, self.forest.world_count, 32, **walk, **colored)
        if colored_kernel is not None:
            # The worlds the unit builder's rule picked: a block each, a color's units at once.
            self._launch(
                colored_kernel, operands, self.forest.world_count, sweep_routes.COLORED_THREADS, **walk, **colored
            )
        if streams:
            # The general tail is queued first; the claims share only the ready dependency with it.
            with wp.ScopedStream(streams[0], sync_enter=False):
                if fork_factor:
                    launch_factor_solve()
                list_tiers()
            if self.claims_listed is not None:
                streams[0].record_event(self.claims_listed)
            for t, tier in enumerate(self.tiers):
                stream = streams[t]
                if t:
                    stream.wait_event(self.claims_listed)
                with wp.ScopedStream(stream, sync_enter=False):
                    launch_tier(tier)
            for stream, done in zip(streams, self.claims_done, strict=True):
                stream.record_event(done)
                main.wait_event(done)

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
        # With only free response bodies, the allocator routes every responsive
        # contact through matrix-free rows. Dense bilateral rows can still exist,
        # but have no patch links or impulses.
        if self.forest.has_non_free_articulations or not self.forest.has_free_rigid_bodies:
            yield (
                0,
                self.placement.row_parent,
                self.placement.row_mu,
                self.placement.impulses,
                self.placement.dense_first,
            )
        if self.forest.has_free_rigid_bodies:
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

    def forces(self, contacts) -> None:
        """Export the linear contact forces of the last solve (:meth:`.SolverFeatherPGS.update_contacts`)."""
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
                max_blocks=self.placement.contact_capacity_blocks,
            )

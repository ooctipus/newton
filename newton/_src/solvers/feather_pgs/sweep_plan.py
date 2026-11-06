# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweep (BACKBONE 2): which worlds each schedule claims, the impulses it starts from and carries, the launches.

:class:`Sweep` owns the solve of one step:

- the admission of the owners that sweep their own worlds (the route table's cases, in its order), with their kernels
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
from collections.abc import Callable

import numpy as np
import warp as wp

from ...core.execution import shape
from ...core.memory import Place, arch_budget, device
from . import (
    host_index,
    packed_rows,
    route_table,
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
from .forest_plan import row_support
from .friction_patches import patch_routes
from .kernel_template import SweepTier
from .route_table import capable
from .row_records import bind as bind_dense_rows
from .solve_policy import SolvePolicy, SolveReadoutRecord
from .sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from .sparse_mass_matrix import _SparseRowSupport, row_supports
from .step_plan import first_failed
from .sweep import FRICTION_METRIC_FLOATS
from .sweep_levels import (
    _get_build_independent_sparse_contact_groups_kernel,
    _get_mark_independent_sparse_contact_candidates_kernel,
)
from .sweep_routes_sparse import _SPARSE_LOCAL_WORKSPACE_BYTES
from .sweep_routes_tiers import classify_parallel_sweep_tiers
from .tree_plan import TreePlan, local_parents
from .tree_scan import classes_half
from .warm_start import (
    bind_carry,
    carry_contacts,
    forget_carried_contacts,
    gather_dense_warmstart,
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

# Parallel tiers grid-stride over per-tier world lists from this many tiers on: one classify launch against
# (tiers - 1) x worlds idle block entries.
_TIER_LIST_MIN_TIERS = 3
_CONTACT_JACOBIAN_MAX_DOF = 10
# The paired response pairs a tree of _PAIRED_MIN_DOF to (32 - 6) coordinates, no star, with one free body, in a class
# of more than _PAIRED_MIN_ROWS rows a world. The rows bound is the retired tiled H^-1 J^T plan's (execution_plan
# before 0.4): it admitted a size whose 32-row chunk was below dense_max_constraints. Its footprint at 26 coordinates
# and 32 rows, 26 * 26 * 4 + 3 * 26 * 32 * 4 + 64 * 4 = 12,944 B, fits every CUDA device (at least 48 KiB a block), so
# the chunk was min(32, dense_max_constraints) and the test was dense_max_constraints > 32.
_PAIRED_MIN_DOF = 17
_PAIRED_MIN_ROWS = 32


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


class Sweep:
    """The solve of one step, with the owners that claim their worlds (see the module's docstring).

    Args:
        model: The solver's model.
        options: The checked options.
        forest: The forest plan.
        laws: The row laws.
        placement: The placement.
        factor: The tree factor.
        routes: The route table the owners claim the class from.
        pins: The test-only route pins.
    """

    def __init__(self, model, options, forest, laws, placement, factor, routes, pins):
        self.model = model
        self.options = options
        self.forest = forest
        self.laws = laws
        self.placement = placement
        self.factor = factor
        self.routes = routes
        self.pins = pins
        # Every sweep compiles the exit law and the policy's readout.
        self.policy = SolvePolicy(readout=options.diagnostics.solve_readout)
        self.readout = SolveReadoutRecord(self.policy)
        self.paired_response_candidate_sizes = frozenset(
            size
            for size in self.forest.size_groups
            if model.device.is_cuda
            and not self.factor.eliminates
            and size >= _PAIRED_MIN_DOF
            and size + 6 <= 32
            and size not in self.forest.diagonal_mass_sizes
            and self.placement.dense_max_constraints > _PAIRED_MIN_ROWS
        )

    def admit_diagonal(self, held_decline: Callable) -> None:
        """Case K2, a tree beside a star: the sparse diagonal, with the fused limits of its star."""
        model = self.model
        self._select_fused_diagonal_limits()
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
            held = held_decline(sparse_diagonal_pair[0])
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
        # A contact is a unit: a normal row, followed by its friction rows only when its region holds an anchor.
        self.placement.sparse_diagonal_contact_units = bool(
            self.placement.sparse_diagonal_contact_solve
            and self.options.enable_contact_friction
            and self.options.contact_friction_gap_threshold == math.inf
        )
        self.placement.sparse_diagonal_speculative_contact_batches = bool(
            self.placement.sparse_diagonal_contact_units and self.placement.sparse_diagonal_dense_size + 2 <= 16
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
        # Each world's dense arm, the tree the walk whitens and leaves (FL3.5).
        self.placement.sparse_diagonal_dense_arts_host = (
            sparse_diagonal_pair[4] if sparse_diagonal_pair is not None else np.zeros(0, dtype=np.int32)
        )

    def _select_fused_diagonal_limits(self) -> None:
        """Select independent position limits that can remain outside the dense row system."""
        model = self.model
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
            and self.laws.link_count == 0
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

    def _select_sparse_diagonal_response_pair(
        self, *, factor_dense_contract: bool
    ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray] | None:
        """Select one independent wide component plus one compact dense component per world: the two sizes, then per
        world the dense component's first world coordinate, its group index and its articulation."""
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
        return int(diagonal_size[0]), int(dense_size[0]), dense_offsets, dense_groups, dense_arts.astype(np.int32)

    def admit_packed(self, model, held_decline: Callable) -> None:
        """Select sparse articulation factors with independent rigid-body response blocks."""
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
        # Case K1, a single tree.
        velocity_candidate = self.options.enable_joint_velocity_limits and not self.forest.has_free_rigid_bodies
        declined = (
            self.routes.taken("K1")
            or capable("packed", self.laws.set)
            or first_failed(
                (
                    (
                        self.pins.get("storage") != "dense",
                        "A test pin keeps dense storage (storage='dense').",
                    ),
                    (not self.pins.get("tiers"), "A test pin builds the Jacobi tiers (tiers) on dense storage."),
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
        held = held_decline(size, plan, velocity_limits=velocity_candidate)
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
            linked_rows=True,
            torsion=self.laws.spin_rows,
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
            self.pins.get("row_support", True)
            and not self.laws.mimic_count
            and np.all(articulated_counts == 1)
            and size == self.forest.max_world_dofs
            and np.all(offsets[articulated] == 0)
        ):
            # The classes of the bodies a contact can name (those with shapes; the row width is theirs too): a
            # shape-less body takes the empty class.
            contact_bodies = np.zeros(body_masks.shape[0], dtype=bool)
            shape_body = model.shape_body.numpy()
            contact_bodies[shape_body[shape_body >= 0]] = True
            classes = tree.support_classes(np.where(contact_bodies, body_masks, np.uint64(0)), max_support)
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
        # A small world packs a lane group of 8 or 16 (_sparse_small_world_lanes) without changing this owner's row
        # representation, sweep policy or velocity-limit phase.
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
        self.routes.reasons["K1"] = (
            f"Automatic sparse selection: {size}-DOF topology and row policy admitted; "
            f"solve uses {solve.bytes}/{budget} bytes of shared memory."
        )
        self.routes.claim("K1")
        self.placement.sparse_mass_matrix_size = size
        self.sparse_active_rows = active_rows
        self.sparse_active_rows_reason = active_rows_reason
        self.sparse_contact_triplets = contact_triplets
        self.sparse_contact_triplets_reason = contact_triplets_reason
        self.sparse_solve_lanes = lanes
        self.sparse_local_factors = local_factors
        self.placement.sparse_max_row_support = int(max_support)
        self.sparse_velocity_limits = velocity_candidate
        self.factor.allocate_sparse(plan)
        self.placement.sparse_body_dof_mask = wp.array(body_masks, dtype=wp.uint64, device=model.device)
        if use_support_table:
            self.placement.sparse_row_support = row_supports(
                support_classes, self.forest.world_count, self.placement.dense_max_constraints, model.device
            )
        if contact_triplets or self.laws.spin_rows:
            # Static storage for graph capture: the builder writes the block records of each step's
            # contact units, every friction triple's metric among them where spin rows may appear; linked sweeps add
            # each anchor's ring members once per call.
            width = CONTACT_RECORD_FLOATS + CONTACT_RING_FLOATS
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

    def admit_paired(self) -> None:
        """Case K5, a tree beside one free body: the paired response and its factor coordinates."""
        model = self.model
        laws = self.laws.set
        self.paired_response_reason = self.routes.taken("K5")
        response_pair = None if self.paired_response_reason else self._select_paired_response()
        if response_pair is None and self.paired_response_reason is None:
            self.paired_response_reason = (
                f"No world pairs one tree of {_PAIRED_MIN_DOF} to 26 coordinates with exactly one free body."
            )
        self.paired_response_primary_size = response_pair[0] if response_pair is not None else None
        self.paired_response_secondary_size = response_pair[1] if response_pair is not None else None
        self.routes.reasons["K5"] = ("No paired response." if response_pair is None else None) or capable(
            "paired_factor", laws
        )
        self.paired_factor_coordinates = self.routes.claim("K5", self.routes.reasons["K5"] is None)
        # Patch contacts are units of one row, or three (four with a spin row) when they own an anchor;
        # the factor sweep walks them with the generic path's row order and
        # arithmetic, solving each tangent pair once.
        self.factor_coordinate_contact_units = bool(self.paired_factor_coordinates and not laws.compliance)
        self.paired_factor_primary_offsets_by_world = None
        self.paired_factor_secondary_offsets_by_world = None
        self.paired_factor_primary_arts_by_world = None
        if self.paired_factor_coordinates:
            primary_arts = self.forest.group_to_art_host[response_pair[0]]
            secondary_arts = self.forest.group_to_art_host[response_pair[1]]
            primary_offsets = np.empty(self.forest.world_count, dtype=np.int32)
            secondary_offsets = np.empty(self.forest.world_count, dtype=np.int32)
            # Each world's articulation: where the paired components' check reads its walk (paired_components).
            primary_by_world = np.full(self.forest.world_count, -1, dtype=np.int32)
            articulation_world = self.forest.articulation_world
            articulation_dof_start = self.forest.articulation_dof_start_host
            for primary_group, primary_art in enumerate(primary_arts):
                world = int(articulation_world[primary_art])
                secondary_group = int(response_pair[2][primary_group])
                secondary_art = int(secondary_arts[secondary_group])
                primary_by_world[world] = int(primary_art)
                primary_first = articulation_dof_start[primary_art] < articulation_dof_start[secondary_art]
                primary_offsets[world] = 0 if primary_first else response_pair[1]
                secondary_offsets[world] = response_pair[0] if primary_first else 0
            self.paired_factor_primary_offsets_by_world = wp.array(primary_offsets, dtype=wp.int32, device=model.device)
            self.paired_factor_secondary_offsets_by_world = wp.array(
                secondary_offsets, dtype=wp.int32, device=model.device
            )
            self.paired_factor_primary_arts_by_world = wp.array(primary_by_world, dtype=wp.int32, device=model.device)

    def plan_tier_rows(self) -> None:
        """Choose the Jacobi tiers from the class (case K4): their row bound and their in-kernel response.

        The tiers serve the dense rows of a tree beside a free body where no earlier case claims the class
        (:meth:`.route_table.RouteTable.taken`). They form each world's response in the kernel, which bounds the class
        (:meth:`.placement.Placement.inkernel_response_meta`: one articulation of each of one or two sizes per world, at most
        ``_INKERNEL_MAX_DOF`` coordinates each), under the contact laws they implement: they pool each friction pair's
        cone over its patch ring and step it isotropically, so a sliding pair converges to the serial pair solve's maximal
        dissipation under either friction law (per-row steps moved body-velocity RMS by 0.4-0.9% on two contact-rich
        patch workloads, and left point tiers 3.9 times the serial sweep's p90 error from its fixed point on anymal-D's
        33-48-row worlds). Their row bound is the dense row capacity, at most
        :data:`.sweep_routes_tiers.TIER_MAX_ROWS`, in whole quads; a test pin (``tiers``) sets it on any class whose
        response they form. ``ink_rows`` is 0 without tiers and the route table's K4 reason says why or why not
        (``routes.row.reasons``). The keyed rows read this before the tiers are built.
        """
        pinned = self.pins.get("tiers")
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
                        "Bilateral pre-elimination corrects the packed rows, which the tiers walk anew.",
                    ),
                    (
                        bool(pinned) or (self.forest.has_free_rigid_bodies and self.forest.has_non_free_articulations),
                        "The tiers serve a tree beside a free body (case K4).",
                    ),
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
        if not self.pins.get("active_rows", True):
            return False, "A test pin keeps the full ordered sweep (active_rows=False)."
        if velocity_candidate:
            return False, "Velocity-limit rows keep full ordered sweeps."
        if small_world_lanes:
            return False, (
                f"Patch friction (friction_anchor_beta > 0) keeps the packed {small_world_lanes}-lane full sweep "
                "of small worlds."
            )
        if not fits():
            return False, "The full ordered sweep: the active-row scan's row bitsets exceed shared memory."
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
        if "contact_units" in self.pins and self.pins["contact_units"] != "triplets":
            return False, "A test pin keeps unfused contact rows (contact_units=None)."
        if not active_rows and (velocity_candidate or self.forest.has_free_rigid_bodies):
            return False, "Velocity-limit and free-body sweeps keep ordinary updates."
        if lanes < 32:
            return False, f"Small worlds keep their packed {lanes}-lane full sweep."
        if not use_support_table:
            return False, "Triplets need the row support table."
        if not fits():
            return False, "Row capacity or shared memory leaves no room for contact units."
        sweep = "active-row scan" if active_rows else "full ordered sweep"
        return True, f"Fused wherever the rows admit them: patch anchor units pool their ring loads in the {sweep}."

    def _select_paired_response(self) -> tuple[int, int, np.ndarray] | None:
        """Select one complete robot/free-body response pair per world."""
        if (
            self.routes.case in ("K1", "K2")
            or self.factor.eliminates
            or self.options.pgs_warmstart
            or 6 not in self.forest.size_groups
        ):
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
        for primary_size in sorted(self.paired_response_candidate_sizes, reverse=True):
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

    def build(self) -> None:
        """Build the paired response and the sweeps."""
        model = self.model
        device_arch = model.device.arch
        from . import paired_components  # noqa: PLC0415

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
        # The sweeps start from the impulses the rows carry: the warm start's, the closed loop's projection.
        self.seeded = bool(self.options.pgs_warmstart or self.factor.eliminates)
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
            colored = capable.size > 0 and route_table.capable("colored", self.laws.set) is None
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
                tiers = [(min(first + 32, parallel_rows), first) for first in range(0, parallel_rows, 32)]
                self.par_tiers = list(tiers)
                if len(tiers) >= _TIER_LIST_MIN_TIERS:
                    dev = model.device
                    self.tier_lo = wp.array([lo for _, lo in tiers], dtype=wp.int32, device=dev)
                    self.tier_hi = wp.array([hi for hi, _ in tiers], dtype=wp.int32, device=dev)
                    self.tier_counts = wp.zeros(len(tiers), dtype=wp.int32, device=dev)
                    self.tier_worlds = wp.empty((len(tiers), self.forest.world_count), dtype=wp.int32, device=dev)
                    self.tier_count_views = [self.tier_counts[t : t + 1] for t in range(len(tiers))]
                    self.tier_world_views = [self.tier_worlds[t] for t in range(len(tiers))]
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
                        # A tier walks its own list of worlds where the plan keeps one.
                        queued=self.tier_worlds is not None,
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

    def warm_start(self, contacts, dt: float) -> None:
        """Gather the dense rows' normal impulses of the previous step by contact identity (:mod:`.warm_start`)."""
        if self.options.pgs_warmstart and contacts is not None:
            match_index = contacts.rigid_contact_match_index
            pool = self.placement.contact_pool
            pool.launch(
                gather_dense_warmstart,
                pool.launch_rows(),
                inputs=[
                    bind_contact_routes(self.placement, contacts),
                    bind_dense_rows(self.placement),
                    match_index,
                    bind_carry(self.placement),
                    self.placement.constraint_count,
                    self.options.pgs_warmstart_decay,
                    self.factor.clock,
                ],
                outputs=[self.placement.impulses],
                extent=contacts.rigid_contact_count,
                max_blocks=self.placement.contact_capacity_blocks,
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
            # The free bodies' placement: their Z and headers (their records are the row store's above).
            "mf_contact_rows_end": placement.mf_contact_rows_end,
            "mf_meta": placement.mf_meta_packed,
            "mf_Z": placement.mf_Z,
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
        if self.leaves and self.routes.case != "K1":
            # The sweeps add their change into the walk's du from zero.
            self.factor.walk_view().du.zero_()
        if self.routes.case == "K1":
            self._solve_packed(operands)
        elif self.routes.case == "K2":
            self._solve_diagonal(operands)
        elif self.forest.max_world_dofs:
            self._solve_claims(operands)
        if self.leaves:
            # The step's one leave: every walking tree's change in the walk's coordinates (the sparse sweep's in its
            # factor coordinates).
            if self.routes.case == "K1":
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
        tier_lists = self.tier_worlds is not None

        def list_tiers():
            if tier_lists:
                self.tier_counts.zero_()
                wp.launch(
                    classify_parallel_sweep_tiers,
                    dim=self.forest.world_count,
                    inputs=[
                        self.placement.constraint_count,
                        self.placement.mf_constraint_count,
                        self.tier_lo,
                        self.tier_hi,
                        len(self.par_tiers),
                    ],
                    outputs=[self.tier_worlds, self.tier_counts],
                    device=self.model.device,
                )

        def launch_tier(tier):
            named = dict(walk)
            dim = self.forest.world_count
            if tier_lists:
                # This tier grid-strides over its own compacted world list.
                dim = tier.grid
                named.update(
                    general_world_count=self.tier_count_views[tier.tier],
                    general_worlds=self.tier_world_views[tier.tier],
                    general_world_grid_stride=tier.grid,
                )
            named.update(self.placement.inkernel_response_inputs())
            if tier.kinetic_rows:
                keyed = self.placement.keyed_rows.parallel_arguments()
                named.update(zip(("kinetic_rows", "patch_law"), keyed, strict=False))
            self._launch(tier.kernel, operands, dim, tier.work.block, **named)

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

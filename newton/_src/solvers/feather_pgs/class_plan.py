# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The class plan: the placement case of the solver's class, decided by rules over data before any of its owners.

Each case's rule (:data:`RULES`, in precedence order) reads only data known before the owners: the forest, the row
laws, the options, the row capacity, the device's shared budget, the test pins, the factor's drive plan and the held
tree's admission of the representation the case needs. It returns the case's choice (what its owner builds) or why the
class does not fit. :func:`plan_class` applies the rules in order: the first that fits gives the class its case, and
every later case says which case holds the class. Three data keep every class general: a test pin, a device without
the cases' native bodies, and the bilateral pre-elimination (:func:`general_reason`). The owners then build from the
plan (:class:`ClassPlan`) and decide nothing of the case.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ...core.memory import arch_budget
from . import held_tree, host_index, sweep_routes_sparse
from .forest_plan import row_support
from .route_table import Pins, first_failed
from .scalar_branches import ScalarBranchPlan, plan_scalar_branches
from .sweep_routes_sparse import _SPARSE_LOCAL_WORKSPACE_BYTES
from .sweep_routes_tiers import TIER_MAX_ROWS
from .tree_plan import TreePlan, local_parents

__all__ = ["RULES", "ClassInputs", "ClassPlan", "general_reason", "plan_class"]

_CONTACT_JACOBIAN_MAX_DOF = 10
# The paired response pairs a tree of _PAIRED_MIN_DOF to (32 - 6) coordinates, no star, with one free body, in a class
# of more than _PAIRED_MIN_ROWS rows a world. The rows bound is the retired tiled H^-1 J^T plan's (execution_plan
# before 0.4): it admitted a size whose 32-row chunk was below dense_max_constraints. Its footprint at 26 coordinates
# and 32 rows, 26 * 26 * 4 + 3 * 26 * 32 * 4 + 64 * 4 = 12,944 B, fits every CUDA device (at least 48 KiB a block), so
# the chunk was min(32, dense_max_constraints) and the test was dense_max_constraints > 32.
_PAIRED_MIN_DOF = 17
_PAIRED_MIN_ROWS = 32
# The largest articulation the tiers hold (case K4): a lane keeps its row's coordinates in registers.
_INKERNEL_MAX_DOF = 32


@dataclass(frozen=True)
class ClassInputs:
    """The data the class rules read, all known before any owner of the class."""

    model: object
    options: object
    """The checked options (:class:`.configuration.Options`)."""
    forest: object
    """The forest plan (:class:`.forest_plan.ForestPlan`)."""
    laws: object
    """The row laws' tables (:class:`.rows.RowLaws`)."""
    factor: object
    """The tree factor's drive plan and kinematic mask (:class:`.tree_factor.TreeFactor`, once planned)."""
    pins: Pins
    """The test-only route pins (:class:`.route_table.Pins`)."""
    dense_rows: int
    """The dense rows a world (:attr:`.step_plan.Capacities.dense_rows`)."""


@dataclass(frozen=True, eq=False)
class Diagonal:
    """Case K2's choice, a tree beside a star: the two sizes, each world's dense arm, and the star's fused limits."""

    size: int
    """The star's (the diagonal articulation's) coordinates."""
    dense_size: int
    """The dense arm's coordinates."""
    dense_offsets: np.ndarray
    """Each world's dense arm's first world coordinate."""
    dense_groups: np.ndarray
    """Each world's dense arm's index in its size group."""
    dense_arts: np.ndarray
    """Each world's dense arm."""
    fused_sizes: frozenset[int]
    """The star sizes whose limits fuse outside the dense rows."""
    fused_mask: np.ndarray
    """Each coordinate's fused limit (1) or none (0)."""
    units: bool
    """A contact is a unit: a normal row, followed by its friction rows only when its region holds an anchor (with
    contact friction and no row-activation gap)."""
    speculative_batches: bool
    """The contact units run in speculative batches (a dense arm of at most 14 coordinates)."""
    scalar: ScalarBranchPlan | None
    """The star's group as scalar branches (:func:`.scalar_branches.plan_scalar_branches`), or ``None``."""
    scalar_reason: str
    """Why the star's group runs as scalar branches or not."""


@dataclass(frozen=True, eq=False)
class Packed:
    """Case K1's choice, native sparse factors of a single tree: the factored size and pattern, the row support and
    the sweep's options."""

    size: int
    plan: object
    """The tree's factor pattern (:class:`.tree_plan.FactorPattern`)."""
    body_masks: np.ndarray
    max_support: int
    support_classes: object
    """The row support table's classes (:class:`.tree_plan.SupportClasses`), or ``None`` for explicit indices."""
    sweep: Mapping[str, object]
    """The sweep's options: ``active_rows``, ``use_support_table``, ``contact_triplets``, ``lanes_per_world``,
    ``local_factors``."""
    velocity_limits: bool
    active_rows_reason: str
    contact_triplets_reason: str
    triplet_capacity: int
    index_shape: tuple[int, int, int]


@dataclass(frozen=True, eq=False)
class Paired:
    """Case K5's choice, a tree beside one free body: the tree's size, the body's, and each tree's body's group."""

    primary: int
    secondary: int
    groups: np.ndarray


@dataclass(frozen=True, eq=False)
class Tiers:
    """Case K4's choice, the Jacobi tiers: their in-kernel response's sizes and offsets ``(NA, NB, OA, OB)``, each
    world's trees (``meta``: each size's articulation and first world coordinate, ``-1`` for none) and the row bound."""

    sizes: tuple[int, int, int, int]
    meta: np.ndarray
    rows: int


@dataclass(frozen=True, eq=False)
class ClassPlan:
    """The class's case and each case's reason, decided by :func:`plan_class`, with the choice of the case that holds
    the class (the others are ``None``)."""

    case: str
    """``"K1"``, ``"K2"``, ``"K4"``, ``"K5"``, or ``"K6"`` (general) where no rule fits."""
    reasons: Mapping[str, str | None]
    """Why each case holds the class or does not."""
    mf_sweeps: bool
    """The matrix-free sweeps run and read the dense rows packed (:mod:`.packed_rows`): every class but native sparse
    factors (case K1) and the sparse diagonal (case K2), where a world has coordinates."""
    diagonal: Diagonal | None = None
    packed: Packed | None = None
    paired: Paired | None = None
    tiers: Tiers | None = None

    @property
    def diagonal_size(self) -> int | None:
        """The sparse diagonal's star size (case K2), or ``None``."""
        return None if self.diagonal is None else self.diagonal.size

    @property
    def dense_size(self) -> int:
        """The sparse diagonal's dense arm size (case K2), or 0."""
        return 0 if self.diagonal is None else self.diagonal.dense_size

    @property
    def fused_sizes(self) -> frozenset[int]:
        """The star sizes whose limits fuse outside the dense rows (case K2), or none."""
        return frozenset() if self.diagonal is None else self.diagonal.fused_sizes

    @property
    def packed_size(self) -> int | None:
        """The native sparse factor's size (case K1), or ``None``."""
        return None if self.packed is None else self.packed.size

    @property
    def tier_rows(self) -> int:
        """The Jacobi tiers' row bound (case K4), or 0 without tiers."""
        return 0 if self.tiers is None else self.tiers.rows

    @property
    def scalar(self) -> tuple[ScalarBranchPlan | None, str]:
        """The star's group as scalar branches and why (:attr:`Diagonal.scalar`): none without the sparse
        diagonal."""
        if self.diagonal is None:
            return None, "No diagonal-mass group shares the sparse-diagonal response."
        return self.diagonal.scalar, self.diagonal.scalar_reason


def general_reason(pins, *, native: bool, eliminates: bool) -> str | None:
    """Why every class keeps the general placement, or ``None``: the test pin ``placement="general"``; a device without
    the cases' native bodies (``native``: they are written for CUDA, so the CPU runs the general placement); and the
    bilateral pre-elimination (``eliminates``), whose blocks the one writer's block phase closes and whose wide rows
    only the general sweep's exit reads (the cases' writers and sweeps take neither yet)."""
    return first_failed(
        (
            (pins.placement != "general", "A test pin keeps the general placement (placement='general')."),
            (native, "The device runs the general placement: the cases' native bodies are written for CUDA."),
            (not eliminates, "Bilateral pre-elimination closes its blocks in the general row's block phase."),
        )
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


def _fused_diagonal_limits(inputs: ClassInputs) -> tuple[frozenset[int], np.ndarray]:
    """The star sizes whose independent position limits can remain outside the dense row system, and each
    coordinate's fused limit."""
    model, options, forest, laws = inputs.model, inputs.options, inputs.forest, inputs.laws
    mask = np.zeros(model.joint_dof_count, dtype=np.int32)
    supported = bool(
        model.joint_dof_count > 0
        and options.enable_joint_limits
        and options.enable_contact_friction
        and not options.enable_joint_velocity_limits
        and not options.pgs_warmstart
        and not forest.has_free_rigid_bodies
        and laws.mimic_count == 0
        and laws.link_count == 0
    )
    if not supported:
        return frozenset(), mask
    candidate_sizes = forest.diagonal_mass_sizes.intersection(laws.joint_limit_sizes)
    if not candidate_sizes:
        return frozenset(), mask
    response_dof_count = forest.response_dof_count
    pairs = _world_diagonal_pairs(forest.articulation_world, response_dof_count, forest.world_count, candidate_sizes)
    if pairs is None:
        return frozenset(), mask
    diagonal_articulations, dense_articulations = pairs
    dense_size = response_dof_count[dense_articulations]
    if np.any((dense_size <= 0) | (dense_size > _CONTACT_JACOBIAN_MAX_DOF)):
        return frozenset(), mask
    # The diagonal articulations with a limited DOF fuse all their DOFs' limits.
    diagonal_size = response_dof_count[diagonal_articulations]
    art, local_dof = host_index.expand(diagonal_size)
    dof = forest.articulation_dof_start_host[diagonal_articulations][art] + local_dof
    valid = laws.joint_limit_q_index_host[dof] >= 0
    limited = np.bincount(art[valid], minlength=diagonal_articulations.size) > 0
    mask[dof[limited[art]]] = valid[limited[art]]
    # Add each size where a world first has it, as the per-world set did.
    return frozenset(int(size) for size in host_index.ordered_unique(diagonal_size[limited])), mask


def diagonal_rule(inputs: ClassInputs) -> tuple[Diagonal | None, str]:
    """Case K2, a tree beside a star: one independent wide component (the star, its limits fused) plus one compact
    dense component per world, of the same two sizes in every world, whose compact diagonal the held tree takes."""
    forest = inputs.forest
    fused_sizes, fused_mask = _fused_diagonal_limits(inputs)
    missing = "No world pairs a diagonal-mass articulation with fused limits and one compact dense articulation."
    if not fused_sizes:
        return None, missing
    response_dofs = forest.response_dof_count
    is_free_rigid = forest.is_free_rigid_host
    pairs = _world_diagonal_pairs(forest.articulation_world, response_dofs, forest.world_count, fused_sizes)
    if pairs is None or forest.world_count == 0:
        return None, missing
    diagonal_arts, dense_arts = pairs
    if np.any(is_free_rigid[diagonal_arts] != 0) or np.any(is_free_rigid[dense_arts] != 0):
        return None, missing
    # Every world pairs the same two sizes.
    diagonal_size = response_dofs[diagonal_arts]
    dense_size = response_dofs[dense_arts]
    if np.any((dense_size <= 0) | (dense_size > _CONTACT_JACOBIAN_MAX_DOF)):
        return None, missing
    if np.any(diagonal_size != diagonal_size[0]) or np.any(dense_size != dense_size[0]):
        return None, missing
    dense_offsets = forest.articulation_world_dof_offset.numpy()[dense_arts].astype(np.int32)
    dense_groups = forest.art_group_idx.numpy()[dense_arts].astype(np.int32)
    if np.any(dense_offsets < 0) or np.any(dense_groups < 0):
        return None, missing
    # Its compact diagonal is the held tree's representation: a candidate only where the held tree takes the dense
    # groups with it, the diagonal group as scalar branches or a held family.
    scalar, scalar_reason = plan_scalar_branches(inputs, size=int(diagonal_size[0]))
    held = held_tree.compact_decline(inputs, int(diagonal_size[0]), scalar=scalar is not None)
    if held is not None:
        return None, f"Only the held tree evaluates the compact diagonal. {held}"
    units = bool(inputs.options.enable_contact_friction and inputs.options.contact_friction_gap_threshold == math.inf)
    choice = Diagonal(
        size=int(diagonal_size[0]),
        dense_size=int(dense_size[0]),
        dense_offsets=dense_offsets,
        dense_groups=dense_groups,
        dense_arts=dense_arts.astype(np.int32),
        fused_sizes=fused_sizes,
        fused_mask=fused_mask,
        units=units,
        speculative_batches=units and int(dense_size[0]) + 2 <= 16,
        scalar=scalar,
        scalar_reason=scalar_reason,
    )
    return choice, "One diagonal-mass articulation and one compact dense articulation per world share one owner."


def _small_world_lanes(forest, dense_rows: int, staging, budget: int) -> int:
    """The lanes per world (8 or 16) that pack a small complete sparse sweep, or 0 for whole warps.

    Args:
        forest: The forest plan.
        dense_rows: The dense rows a world.
        staging: The sweep's working set for its options (:func:`.sweep_routes_sparse.sparse_staging`).
        budget: Static shared memory available to one block [B].
    """
    if forest.max_world_dofs > 16 or dense_rows > 64:
        return 0
    lanes = 8 if forest.max_world_dofs <= 8 else 16
    return lanes if staging(lanes_per_world=lanes).fits(budget) else 0


def active_rows_rule(pins, *, small_world_lanes: int, fits: Callable[[], bool]) -> tuple[bool, str]:
    """Whether the native sparse sweep selects its normal and limit rows at sweep entry, and why.

    Under this law a zero-impulse row that a sweep's earlier updates violate waits for
    the next sweep. Patch friction (``friction_anchor_beta > 0``) adopts it: a friction
    pair is visited while any normal of its patch ring is selected or the pair holds
    impulse, which skips only pairs whose update would be a no-op. Velocity-limit rows keep
    their last word after the scanned rows. Small worlds keep their packed lanes. A scan
    whose row bitsets exceed shared memory leaves the full ordered sweep.

    Args:
        pins: The test-only route pins (:class:`.route_table.Pins`).
        small_world_lanes: Lanes per world of a packed small-world sweep, 0 for whole warps.
        fits: Whether the active-row scan's working set fits shared memory, evaluated last.
    """
    if not pins.active_rows:
        return False, "A test pin keeps the full ordered sweep (active_rows=False)."
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


def contact_triplets_rule(
    pins, *, active_rows: bool, use_support_table: bool, lanes: int, fits: Callable[[], bool]
) -> tuple[bool, str]:
    """Whether the sparse sweep fuses contact triplets, and why: wherever the row structure admits them.

    Patch friction fuses its anchor units, pooling their ring loads as the ordinary pair
    update does, in the active scan or else in the full ordered sweep; packed small-world
    sweeps keep ordinary updates. A fused triplet is a contact unit on its Gram block
    (:func:`.sweep_routes_sparse.contact_units`): in exact arithmetic the ordinary sweep's
    normal and pair updates, in one reduction and one apply.

    Args:
        pins: The test-only route pins (:class:`.route_table.Pins`).
        active_rows: Whether the sweep selects its rows at sweep entry.
        use_support_table: Whether rows read the shared support table.
        lanes: Lanes per world of the sweep.
        fits: Whether contact units fit the row capacity and shared memory, evaluated last.
    """
    if pins.contact_units != "triplets":
        return False, "A test pin keeps unfused contact rows (contact_units=None)."
    if lanes < 32:
        return False, f"Small worlds keep their packed {lanes}-lane full sweep."
    if not use_support_table:
        return False, "Triplets need the row support table."
    if not fits():
        return False, "Row capacity or shared memory leaves no room for contact units."
    sweep = "active-row scan" if active_rows else "full ordered sweep"
    return True, f"Fused wherever the rows admit them: patch anchor units pool their ring loads in the {sweep}."


def packed_rule(inputs: ClassInputs) -> tuple[Packed | None, str]:
    """Case K1, a single tree: native sparse factors with independent rigid-body response blocks."""
    model, options, forest, laws, factor, pins = (
        inputs.model,
        inputs.options,
        inputs.forest,
        inputs.laws,
        inputs.factor,
        inputs.pins,
    )
    dense_rows = inputs.dense_rows
    velocity_candidate = options.enable_joint_velocity_limits and not forest.has_free_rigid_bodies
    declined = first_failed(
        (
            (pins.storage != "dense", "A test pin keeps dense storage (storage='dense')."),
            (not pins.tiers, "A test pin builds the Jacobi tiers (tiers) on dense storage."),
            (factor.parallel_drives, "Drive topology is not supported by native sparse factors."),
            (
                not laws.mimic_count or velocity_candidate,
                "Mimic rows require the sparse velocity-row composition.",
            ),
        )
    )
    if declined is not None:
        return None, declined
    if forest.has_free_rigid_bodies:
        # A world that pairs an articulation with free rigid bodies is served
        # by the held-tree owner and the paired response on the dense route
        # (measured: Kuka lift 12.7 ms there against 15.5 on sparse factors).
        # Both friction laws: patch friction walks contact units in the paired
        # sweep and keeps the resolved-world classifier. Native sparse factors never serve free rigid bodies.
        return None, "Worlds with free rigid bodies keep the dense route for the held-tree owner and paired response."
    response_dofs = forest.response_dof_count
    if velocity_candidate and (forest.has_prescribed_response or np.any(factor.kinematic_dof_mask_host)):
        return None, "Sparse velocity rows do not support prescribed-coordinate topology."
    free_rigid = forest.is_free_rigid_host != 0
    articulated = np.flatnonzero((response_dofs > 0) & ~free_rigid)
    sizes = np.unique(response_dofs[articulated])
    if len(sizes) != 1:
        return None, "Native sparse factors require one articulated response DOF size."
    size = int(sizes[0])
    # The endpoint support uses one 64-bit ancestry mask, not a robot-specific DOF limit.
    if size > 64 or size not in forest.crba_dof_joint_offset_by_size or size in forest.diagonal_mass_sizes:
        return None, "Articulated topology needs at most 64 response DOFs, a CRBA mapping and non-diagonal inertia."
    articulated_counts = np.bincount(forest.articulation_world[articulated], minlength=forest.world_count)
    articulation_start = model.articulation_start.numpy()
    joint_ancestor = model.joint_ancestor.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    art = int(forest.group_to_art[size].numpy()[0])
    start = int(articulation_start[art])
    end = int(forest.articulation_joint_end_host[art])
    tree = TreePlan.build(
        local_parents(joint_ancestor, [start], end - start)[0], np.diff(joint_qd_start[start : end + 1])
    )
    plan = tree.factor
    # Velocity rows can benefit from factor-coordinate response even for a
    # serial chain; this owner does not require zeros in the mass factor.
    if plan.dof_count != size or (not velocity_candidate and plan.nonzero_count >= size * (size + 1) // 2):
        return None, "Topology has no usable sparse factor layout for this row policy."
    # Native sparse factors are the held tree's representation: a candidate only where it takes their group.
    held = held_tree.sparse_decline(inputs, size, plan, velocity_limits=velocity_candidate)
    if held is not None:
        return None, f"Only the held tree evaluates native sparse factors. {held}"

    mimic_dofs = None
    if laws.mimic_count:
        mimic_indices = laws._mimic_art_list.numpy()
        mimic_arts, _ = host_index.expand(np.diff(laws.mimic_art_start_np))
        starts = forest.articulation_dof_start_host[mimic_arts]
        mimic_dofs = (
            laws._mimic_dof0.numpy()[mimic_indices] - starts,
            laws._mimic_dof1.numpy()[mimic_indices] - starts,
        )
    body_masks, max_support = row_support(forest, model, plan, articulated, articulated_counts, mimic_dofs)
    if not max_support:
        return None, "Topology has no responsive row support."
    # The sweep's shared working set (sweep_routes_sparse.sparse_staging) against the device's static budget:
    # each shared-memory choice below reads its own sweep's working set, and the route is admitted by the working
    # set of the sweep it launches, which the sweep's factory checks the same way.
    budget = arch_budget(model.device.arch)
    staging = functools.partial(
        sweep_routes_sparse.sparse_staging,
        dense_rows,
        forest.max_world_dofs,
        int(max_support),
        velocity_limits=velocity_candidate,
        linked_rows=True,
        torsion=laws.spin_rows,
    )
    # Patch friction keeps the packed lanes of small worlds, so size them from the
    # full sweep before the active-row scan claims whole warps.
    small_world_lanes = _small_world_lanes(forest, dense_rows, staging, budget)
    active_rows, active_rows_reason = active_rows_rule(
        pins,
        small_world_lanes=small_world_lanes,
        fits=lambda: staging(active_rows=True).fits(budget),
    )
    shape = (forest.world_count, dense_rows, max_support)
    offsets = forest.articulation_world_dof_offset.numpy()
    support_classes = None
    if (
        pins.row_support
        and not laws.mimic_count
        and np.all(articulated_counts == 1)
        and size == forest.max_world_dofs
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
            table_bytes += 4 * forest.world_count * dense_rows
            # Small batches with many support patterns can favor direct indices.
            if table_bytes < 4 * int(np.prod(shape)):
                support_classes = classes
    # Unsupported mixed worlds retain explicit row indices and their existing kernels.
    use_support_table = support_classes is not None
    # A small world packs a lane group of 8 or 16 (_small_world_lanes) without changing this owner's row
    # representation, sweep policy or velocity-limit phase.
    lanes = small_world_lanes if small_world_lanes and not active_rows else 32
    triplet_capacity = dense_rows // 3
    contact_triplets, contact_triplets_reason = contact_triplets_rule(
        pins,
        active_rows=active_rows,
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
        return None, f"Row capacity needs {solve.bytes} bytes of shared memory for sparse solving; limit is {budget}."
    choice = Packed(
        size=size,
        plan=plan,
        body_masks=body_masks,
        max_support=int(max_support),
        support_classes=support_classes,
        sweep=MappingProxyType(sweep),
        velocity_limits=bool(velocity_candidate),
        active_rows_reason=active_rows_reason,
        contact_triplets_reason=contact_triplets_reason,
        triplet_capacity=triplet_capacity,
        index_shape=(1, 1, 1) if use_support_table else shape,
    )
    return choice, (
        f"Automatic sparse selection: {size}-DOF topology and row policy admitted; "
        f"solve uses {solve.bytes}/{budget} bytes of shared memory."
    )


def paired_rule(inputs: ClassInputs) -> tuple[Paired | None, str]:
    """Case K5, a tree beside one free body: one complete robot/free-body response pair per world, the tree of
    :data:`_PAIRED_MIN_DOF` to 26 coordinates and no star, in a class of more than :data:`_PAIRED_MIN_ROWS` rows."""
    forest = inputs.forest
    missing = f"No world pairs one tree of {_PAIRED_MIN_DOF} to 26 coordinates with exactly one free body."
    if 6 not in forest.size_groups:
        return None, missing
    candidate_sizes = frozenset(
        size
        for size in forest.size_groups
        if size >= _PAIRED_MIN_DOF
        and size + 6 <= 32
        and size not in forest.diagonal_mass_sizes
        and inputs.dense_rows > _PAIRED_MIN_ROWS
    )
    response_dofs = forest.response_dof_count
    articulation_world = forest.articulation_world
    is_free_rigid = forest.is_free_rigid_host
    # Each world's response articulations, in ascending order.
    responding = np.flatnonzero(response_dofs > 0)
    world_bounds, order = host_index.group_by(articulation_world[responding], forest.world_count)
    responding = responding[order].tolist()
    response_by_world = [responding[world_bounds[w] : world_bounds[w + 1]] for w in range(forest.world_count)]

    secondary_group_arts = forest.group_to_art_host[6]
    secondary_group_index = {int(art): group for group, art in enumerate(secondary_group_arts)}
    for primary_size in sorted(candidate_sizes, reverse=True):
        primary_arts = forest.group_to_art_host.get(primary_size, ())
        if len(primary_arts) != forest.world_count:
            continue
        paired_groups: list[int] = []
        paired_arts: list[int] = []
        for primary_art in primary_arts:
            world_arts = response_by_world[int(articulation_world[primary_art])]
            secondaries = [
                art for art in world_arts if art != primary_art and response_dofs[art] == 6 and is_free_rigid[art] != 0
            ]
            if len(world_arts) != 2 or len(secondaries) != 1:
                break
            secondary_art = secondaries[0]
            paired_arts.append(secondary_art)
            paired_groups.append(secondary_group_index[secondary_art])
        else:
            if len(set(paired_arts)) == len(secondary_group_arts) and set(paired_arts) == set(secondary_group_arts):
                return Paired(primary_size, 6, np.asarray(paired_groups, dtype=np.int32)), None
    return None, missing


def tier_response(forest) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None, str]:
    """The in-kernel response's group sizes and offsets ``(NA, NB, OA, OB)`` and each world's trees, or ``None``,
    and why.

    The parallel tiers walk the rows of up to two size groups, one articulation of each size per world with a
    primary in every world, at one world DOF offset per size when the response is written in world layout: the meta
    holds each world's articulation of each size and its first world coordinate (``-1`` for none).
    """
    if forest.art_to_world is None:
        return None, None, "No articulation->world map."
    if not (1 <= len(forest.size_groups) <= 2):
        return None, None, f"{len(forest.size_groups)} size groups; the in-kernel response takes one or two."
    sizes = sorted((int(s) for s in forest.size_groups), reverse=True)
    if sizes[0] > _INKERNEL_MAX_DOF:
        return None, None, f"Size {sizes[0]} exceeds the tiers' {_INKERNEL_MAX_DOF} coordinates a row."
    art_to_world = forest.art_to_world.numpy()
    dof_off = forest.articulation_world_dof_offset.numpy()
    meta = -np.ones((forest.world_count, 4), dtype=np.int32)
    offsets = [0, 0]
    for col, size in enumerate(sizes):
        g2a = forest.group_to_art[size].numpy()
        worlds = art_to_world[g2a]
        if np.unique(worlds).size != worlds.size:
            return None, None, f"A world has several articulations of size {size}."
        # The rows lie at their world coordinates (the packed rows' dense layout).
        off = dof_off[g2a]
        if off.size and np.any(off != off[0]):
            # The register rows use compile-time offsets.
            return None, None, f"The world DOF offsets of size {size} differ."
        offsets[col] = int(off[0]) if off.size else 0
        meta[worlds, 2 * col] = g2a
        meta[worlds, 2 * col + 1] = off
    if np.any(meta[:, 0] < 0):
        return None, None, "A world has no primary articulation."
    nb = sizes[1] if len(sizes) > 1 else 0
    if offsets[0] + sizes[0] > forest.max_world_dofs or (nb and offsets[1] + nb > forest.max_world_dofs):
        return None, None, "The group DOFs exceed the world DOF count."
    return (sizes[0], nb, offsets[0], offsets[1]), meta, f"The parallel tiers walk the rows of sizes {sizes}."


def tier_rule(inputs: ClassInputs) -> tuple[Tiers | None, str]:
    """Case K4, the Jacobi tiers of the dense rows of a tree beside a free body.

    They form each world's response in the kernel, which bounds the class (:func:`tier_response`: one articulation of
    each of one or two sizes per world, at most :data:`_INKERNEL_MAX_DOF` coordinates each), under the contact laws
    they implement: they pool each friction pair's cone over its patch ring and step it isotropically, so a sliding
    pair converges to the serial pair solve's maximal dissipation under either friction law (per-row steps moved
    body-velocity RMS by 0.4-0.9% on two contact-rich patch workloads, and left point tiers 3.9 times the serial
    sweep's p90 error from its fixed point on anymal-D's 33-48-row worlds). Their row bound is the dense row capacity,
    at most :data:`.sweep_routes_tiers.TIER_MAX_ROWS`, in whole quads; a test pin (``tiers``) sets it on any class
    whose response they form.
    """
    forest, pinned = inputs.forest, inputs.pins.tiers
    reason = first_failed(
        (
            (
                forest.world_count > 0 and forest.max_world_dofs > 0,
                "No matrix-free sweep runs whose dense rows the tiers could take.",
            ),
            (pinned is None or bool(pinned), "A test pin builds no tier (tiers=0)."),
            (
                bool(pinned) or (forest.has_free_rigid_bodies and forest.has_non_free_articulations),
                "The tiers serve a tree beside a free body (case K4).",
            ),
        )
    )
    if reason is not None:
        return None, reason
    sizes, meta, reason = tier_response(forest)
    if sizes is None:
        return None, reason
    rows = min(int(inputs.dense_rows), int(pinned or TIER_MAX_ROWS))
    rows -= rows % 4
    if rows <= 0:
        return None, f"The dense row capacity {inputs.dense_rows} holds no tier of four rows."
    return Tiers(sizes, meta, rows), reason


RULES = (
    ("K2", "the sparse diagonal of a tree beside a star", diagonal_rule),
    ("K1", "native sparse factors of a single tree", packed_rule),
    ("K5", "factor coordinates of a tree beside one free body", paired_rule),
    ("K4", "the Jacobi tiers of dense trees beside a free body", tier_rule),
)
"""The specialized cases (placement and producer by class shape) in precedence order, each with its rule: the first
whose rule fits claims the class. A class no rule claims keeps the general placement, case K6."""

_CHOICE = {"K2": "diagonal", "K1": "packed", "K5": "paired", "K4": "tiers"}


def plan_class(inputs: ClassInputs, *, general: str | None) -> ClassPlan:
    """The class's plan: each rule of :data:`RULES` in order until one fits; every case after it says which case
    holds the class, and ``general`` (:func:`general_reason`), when given, is every case's reason."""
    descriptions = {name: description for name, description, _ in RULES}
    forest = inputs.forest
    case, reasons, choices = "K6", {}, {}
    for name, _, rule in RULES:
        if general is not None:
            reasons[name] = general
            continue
        if case != "K6":
            reasons[name] = f"The class is case {case}: {descriptions[case]}."
            continue
        choice, reasons[name] = rule(inputs)
        if choice is not None:
            case, choices[_CHOICE[name]] = name, choice
    mf_sweeps = case not in ("K1", "K2") and forest.world_count > 0 and forest.max_world_dofs > 0
    return ClassPlan(case=case, reasons=MappingProxyType(reasons), mf_sweeps=mf_sweeps, **choices)

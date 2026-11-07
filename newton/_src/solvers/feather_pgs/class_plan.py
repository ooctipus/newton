# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The class plan: the placement of the solver's class, decided by predicates over data before any of its owners.

Each choice's rule (:data:`RULES`) is a predicate on the class shape (:class:`.forest_plan.ClassShape`) and data known
before the owners: the row laws, the options, the row capacity, the device's shared budget, the test pins, the
factor's drive plan and the held tree's admission of the representation the choice needs. It returns the choice (what
its owner builds) or why the class does not fit. The predicates exclude one another, so at most one holds the class
and none outranks another. One datum keeps every class general: a test pin (:func:`general_reason`). Each body runs
on both devices, the CPU's one thread a lane group, so the device decides nothing. The owners then build from the
plan (:class:`ClassPlan`) and decide nothing of it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ...core.memory import arch_budget
from . import held_tree, host_index, sweep_routes_sparse
from .forest_plan import row_support
from .route_table import Pins, first_failed
from .sweep_routes_sparse import _SPARSE_LOCAL_WORKSPACE_BYTES
from .sweep_routes_tiers import TIER_MAX_ROWS
from .tree_plan import TreePlan, local_parents

__all__ = ["RULES", "ClassInputs", "ClassPlan", "general_reason", "plan_class"]

_CONTACT_JACOBIAN_MAX_DOF = 10
# A warp's lanes: a world that fits them keeps its velocity change a coordinate a lane in registers (case K5).
_LANES = 32
# A hub world of fewer coordinates steps its rows at once (:func:`split_level`).
_SEQUENTIAL_COORDINATES = 23
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
    held: held_tree.HeldPlan
    """The held tree's plan with this compact diagonal, which admitted the case (:func:`.held_tree.plan`)."""


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
    held: held_tree.HeldPlan | None = None
    """The held tree's plan with these factors, which admitted the case (:func:`.held_tree.plan`)."""


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
    """The choice that holds the class (the others are ``None``; none for the general placement) and each rule's reason,
    decided by :func:`plan_class`."""

    reasons: Mapping[str, str | None]
    """Why each rule of :data:`RULES` holds the class or does not."""
    mf_sweeps: bool
    """The matrix-free sweeps run and read the dense rows packed (:mod:`.packed_rows`): every class but native sparse
    factors (case K1) and the sparse diagonal (case K2), where a world has coordinates."""
    held: held_tree.HeldPlan | None = None
    """The held tree's plan: one held plan a world class, from the case's choice (:func:`.held_tree.plan`)."""
    diagonal: Diagonal | None = None
    packed: Packed | None = None
    paired: Paired | None = None
    tiers: Tiers | None = None

    @property
    def support(self) -> str:
        """Where a row's entries live: on its native sparse factor's support, the sparse diagonal's, or dense."""
        return "factor" if self.packed else "diagonal" if self.diagonal else "dense"

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


def general_reason(pins) -> str | None:
    """Why every class keeps the general placement, or ``None``: the test pin ``placement="general"``. Each case's body
    runs on both devices (the CPU's one thread runs its lane group), so the device is no datum of the case.
    The bilateral pre-elimination is no general datum: its blocks close in the one writer's block phase, so each case
    whose writer lacks that phase declines it in its own rule (K1's support writer, the keyed layout)."""
    return first_failed(
        ((pins.placement != "general", "A test pin keeps the general placement (placement='general')."),)
    )


def _star_beside(inputs: ClassInputs) -> tuple[int, int] | None:
    """The class shape's star block and tree block, or ``None``: one of each in every world, a limited star, a tree a
    contact Jacobian wide, no free body, and the laws the fused limits leave out (friction, no mimic or link row)."""
    model, options, forest, laws = inputs.model, inputs.options, inputs.forest, inputs.laws
    shape = forest.shape
    if not (
        model.joint_dof_count > 0
        and options.enable_joint_limits
        and options.enable_contact_friction
        and not forest.has_free_rigid_bodies
        and laws.mimic_count == 0
        and laws.link_count == 0
        and shape.blocks == 2
        and shape.present(0)
        and shape.present(1)
    ):
        return None
    stars = [b for b, size in enumerate(shape.sizes) if size in forest.diagonal_mass_sizes & laws.joint_limit_sizes]
    if len(stars) != 1 or not 0 < shape.sizes[1 - stars[0]] <= _CONTACT_JACOBIAN_MAX_DOF:
        return None
    return stars[0], 1 - stars[0]


def _fused_diagonal_limits(inputs: ClassInputs, star: int) -> tuple[frozenset[int], np.ndarray]:
    """The star sizes whose independent position limits remain outside the dense row system, and each coordinate's
    fused limit: every coordinate of a star tree that holds a limit."""
    forest, shape = inputs.forest, inputs.forest.shape
    mask = np.zeros(inputs.model.joint_dof_count, dtype=np.int32)
    arts, size = shape.arts[:, star], shape.sizes[star]
    art, local_dof = host_index.expand(np.full(arts.size, size))
    dof = forest.articulation_dof_start_host[arts][art] + local_dof
    valid = inputs.laws.joint_limit_q_index_host[dof] >= 0
    limited = np.bincount(art[valid], minlength=arts.size) > 0
    mask[dof[limited[art]]] = valid[limited[art]]
    return frozenset((size,) if limited.any() else ()), mask


def diagonal_rule(inputs: ClassInputs) -> tuple[Diagonal | None, str]:
    """Case K2, a tree beside a star: one independent wide component (the star, its limits fused) plus one compact
    dense component per world, of the same two sizes in every world, whose compact diagonal the held tree takes."""
    forest, shape = inputs.forest, inputs.forest.shape
    missing = "No world pairs a diagonal-mass articulation with fused limits and one compact dense articulation."
    if inputs.pins.tiers:
        return None, "A test pin builds the Jacobi tiers (tiers) on dense storage."
    blocks = _star_beside(inputs)
    if blocks is None:
        return None, missing
    star, dense = blocks
    fused_sizes, fused_mask = _fused_diagonal_limits(inputs, star)
    if not fused_sizes:
        return None, missing
    dense_arts = shape.arts[:, dense]
    # Its compact diagonal is the held tree's representation: a candidate only where the held tree takes the dense
    # groups with it, the diagonal group as a held family (the one-coordinate class where its trees are rigid below
    # their leaves).
    try:
        held = held_tree.plan(inputs, compact=shape.sizes[star])
    except ValueError as error:
        return None, f"Only the held tree evaluates the compact diagonal. {error}"
    units = bool(inputs.options.enable_contact_friction and inputs.options.contact_friction_gap_threshold == math.inf)
    choice = Diagonal(
        size=shape.sizes[star],
        dense_size=shape.sizes[dense],
        dense_offsets=forest.articulation_world_dof_offset.numpy()[dense_arts].astype(np.int32),
        dense_groups=shape.groups[:, dense].astype(np.int32),
        dense_arts=dense_arts.astype(np.int32),
        fused_sizes=fused_sizes,
        fused_mask=fused_mask,
        units=units,
        speculative_batches=units and shape.sizes[dense] + 2 <= 16,
        held=held,
    )
    return choice, "One diagonal-mass articulation and one compact dense articulation per world share one owner."


def _small_world_lanes(forest, dense_rows: int, staging, budget: int) -> int:
    """The lanes per world (8 or 16) that pack a small complete sparse sweep, or 0 for whole warps.

    Args:
        forest: The forest plan.
        dense_rows: The dense rows a world.
        staging: The sweep's working set for its options (:attr:`.sweep_routes_sparse.SparseSweep.staging`).
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
    """Case K1, a single tree: native sparse factors of the class's one block, a tree shape without a free body."""
    model, options, forest, laws, factor, pins = (
        inputs.model,
        inputs.options,
        inputs.forest,
        inputs.laws,
        inputs.factor,
        inputs.pins,
    )
    dense_rows, shape = inputs.dense_rows, inputs.forest.shape
    velocity_candidate = options.enable_joint_velocity_limits
    declined = first_failed(
        (
            (pins.storage != "dense", "A test pin keeps dense storage (storage='dense')."),
            (not pins.tiers, "A test pin builds the Jacobi tiers (tiers) on dense storage."),
            (factor.parallel_drives, "Drive topology is not supported by native sparse factors."),
            (
                not factor.eliminates,
                "Bilateral pre-elimination closes its blocks in the one writer's block phase; the support writer has "
                "none.",
            ),
        )
    )
    if declined is not None:
        return None, declined
    if any(shape.free):
        # A free body beside a tree is the paired response's (measured: Kuka lift 12.7 ms there against 15.5 on sparse
        # factors).
        return None, "Worlds with free rigid bodies keep the dense route for the held-tree owner and paired response."
    if shape.blocks != 1:
        return None, "Native sparse factors require one articulated response DOF size."
    size = shape.sizes[0]
    # The endpoint support uses one 64-bit ancestry mask, not a robot-specific DOF limit.
    if size > 64 or size not in forest.crba_dof_joint_offset_by_size or size in forest.diagonal_mass_sizes:
        return None, "Articulated topology needs at most 64 response DOFs, a CRBA mapping and non-diagonal inertia."
    articulated = forest.group_to_art_host[size]
    joint_ancestor = model.joint_ancestor.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    start = int(model.articulation_start.numpy()[articulated[0]])
    end = int(forest.articulation_joint_end_host[articulated[0]])
    tree = TreePlan.build(
        local_parents(joint_ancestor, [start], end - start)[0], np.diff(joint_qd_start[start : end + 1])
    )
    plan = tree.factor
    # Velocity rows can benefit from factor-coordinate response even for a
    # serial chain; this owner does not require zeros in the mass factor.
    if plan.dof_count != size or (not velocity_candidate and plan.nonzero_count >= size * (size + 1) // 2):
        return None, "Topology has no usable sparse factor layout for this row policy."
    mimic_dofs = None
    if laws.mimic_count:
        mimic_indices = laws._mimic_art_list.numpy()
        mimic_arts, _ = host_index.expand(np.diff(laws.mimic_art_start_np))
        starts = forest.articulation_dof_start_host[mimic_arts]
        mimic_dofs = (
            laws._mimic_dof0.numpy()[mimic_indices] - starts,
            laws._mimic_dof1.numpy()[mimic_indices] - starts,
        )
    body_masks, max_support = row_support(forest, model, plan, articulated, shape.arts is None, mimic_dofs)
    if not max_support:
        return None, "Topology has no responsive row support."
    # The sweep's shared working set (sweep_routes_sparse.SparseSweep.staging) against the device's static budget:
    # each shared-memory choice below reads its own sweep's working set, and the route is admitted by the working
    # set of the sweep it launches, which the one factory checks the same way.
    budget = arch_budget(model.device.arch)

    def staging(**options):
        return sweep_routes_sparse.SparseSweep(
            dense_rows,
            forest.max_world_dofs,
            int(max_support),
            velocity_limits=velocity_candidate,
            linked_rows=True,
            torsion=bool(laws.spin_rows),
            **options,
        ).staging

    # Patch friction keeps the packed lanes of small worlds, so size them from the
    # full sweep before the active-row scan claims whole warps.
    small_world_lanes = _small_world_lanes(forest, dense_rows, staging, budget)
    active_rows, active_rows_reason = active_rows_rule(
        pins,
        small_world_lanes=small_world_lanes,
        fits=lambda: staging(active_rows=True).fits(budget),
    )
    index_shape = (forest.world_count, dense_rows, max_support)
    support_classes = None
    # A world of the one tree alone: its coordinates are the tree's.
    if pins.row_support and not laws.mimic_count and shape.present(0):
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
            if table_bytes < 4 * int(np.prod(index_shape)):
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
        index_shape=(1, 1, 1) if use_support_table else index_shape,
    )
    # Native sparse factors are the held tree's representation: a candidate only where it takes their group.
    try:
        choice = dataclasses.replace(choice, held=held_tree.plan(inputs, packed=choice))
    except ValueError as error:
        return None, f"Only the held tree evaluates native sparse factors. {error}"
    return choice, (
        f"Automatic sparse selection: {size}-DOF topology and row policy admitted; "
        f"solve uses {solve.bytes}/{budget} bytes of shared memory."
    )


def split_level(coordinates: int) -> bool:
    """Whether a hub world of ``coordinates`` steps its rows at once (the split level) rather than in turn."""
    return coordinates < _SEQUENTIAL_COORDINATES


def _in_turn(forest) -> str | None:
    """Why the rows do not step in turn, `u` in registers (a tree beside one free body, in the lanes), or ``None``."""
    shape = forest.shape
    if not (shape.blocks == 2 and shape.present(0) and shape.present(1) and shape.free.count(True) == 1):
        return "No world class pairs one tree with exactly one free body in every world."
    return first_failed(
        (
            (shape.sizes[shape.free.index(False)] not in forest.diagonal_mass_sizes, "The tree is a star."),
            (shape.coordinates <= _LANES, f"{shape.coordinates} coordinates a world exceed a warp's {_LANES} lanes."),
            (
                tier_response(shape)[0] is None or not split_level(shape.coordinates),
                f"A hub world of {shape.coordinates} coordinates steps its rows at once.",
            ),
        )
    )


def paired_rule(inputs: ClassInputs) -> tuple[Paired | None, str]:
    """Case K5, `u` in registers: a tree beside one free body in every world, in the lanes, its rows in turn."""
    forest, shape = inputs.forest, inputs.forest.shape
    reason = _in_turn(forest)
    if reason is not None:
        return None, reason
    tree, body = shape.free.index(False), shape.free.index(True)
    # Each tree's free body's group, by the tree's group.
    groups = np.empty(forest.world_count, dtype=np.int32)
    groups[shape.groups[:, tree]] = shape.groups[:, body]
    return Paired(shape.sizes[tree], shape.sizes[body], groups), None


def tier_response(shape) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None, str]:
    """The in-kernel response's group sizes and offsets ``(NA, NB, OA, OB)`` and each world's trees, or ``None``,
    and why.

    The parallel tiers walk the rows of up to two blocks of the class shape, one articulation of each per world with a
    primary in every world, at one world DOF offset per block when the response is written in world layout: the meta
    holds each world's articulation of each block and its first world coordinate (``-1`` for none).
    """
    if not 1 <= shape.blocks <= 2:
        return None, None, f"{shape.blocks} size groups; the in-kernel response takes one or two."
    if shape.sizes[0] > _INKERNEL_MAX_DOF:
        return None, None, f"Size {shape.sizes[0]} exceeds the tiers' {_INKERNEL_MAX_DOF} coordinates a row."
    if shape.arts is None:
        return None, None, "A world has several articulations of one size."
    if None in shape.offsets:
        # The register rows use compile-time offsets.
        return None, None, "The world DOF offsets of a size differ."
    if not shape.present(0):
        return None, None, "A world has no primary articulation."
    (na, nb), (oa, ob) = (*shape.sizes, 0)[:2], (*shape.offsets, 0)[:2]
    if oa + na > shape.coordinates or (nb and ob + nb > shape.coordinates):
        return None, None, "The group DOFs exceed the world DOF count."
    meta = -np.ones((shape.arts.shape[0], 4), dtype=np.int32)
    meta[:, 0 : 2 * shape.blocks : 2] = shape.arts
    meta[:, 1 : 2 * shape.blocks : 2] = np.where(shape.arts >= 0, shape.offsets, -1)
    return (na, nb, oa, ob), meta, f"The parallel tiers walk the rows of sizes {list(shape.sizes)}."


def tier_rule(inputs: ClassInputs) -> tuple[Tiers | None, str]:
    """Case K4, the Jacobi tiers of the dense rows of a hub: a free body beside a tree in one world.

    They form each world's response in the kernel, which bounds the class (:func:`tier_response`: one articulation of
    each of one or two sizes per world, at most :data:`_INKERNEL_MAX_DOF` coordinates each), under the contact laws
    they implement: they pool each friction pair's cone over its patch ring and step it isotropically, so a sliding
    pair converges to the serial pair solve's maximal dissipation under either friction law (per-row steps moved
    body-velocity RMS by 0.4-0.9% on two contact-rich patch workloads, and left point tiers 3.9 times the serial
    sweep's p90 error from its fixed point on anymal-D's 33-48-row worlds). Their row bound is the dense row capacity,
    at most :data:`.sweep_routes_tiers.TIER_MAX_ROWS`, in whole quads; a test pin (``tiers``) sets it on any class
    whose response they form. A lane a row of a block's group: the CPU's one thread runs the lanes in turn
    (:class:`.device_preludes.VirtualLanes`).
    """
    forest, pinned = inputs.forest, inputs.pins.tiers
    reason = first_failed(
        (
            (
                forest.world_count > 0 and forest.max_world_dofs > 0,
                "No matrix-free sweep runs whose dense rows the tiers could take.",
            ),
            (pinned is None or bool(pinned), "A test pin builds no tier (tiers=0)."),
            (bool(pinned) or forest.shape.beside, "The tiers serve a hub: a free body beside a tree in one world."),
            (_in_turn(forest) is not None, "A world's rows step in turn, `u` in registers."),
        )
    )
    if reason is not None:
        return None, reason
    sizes, meta, reason = tier_response(forest.shape)
    if sizes is None:
        return None, reason
    rows = min(int(inputs.dense_rows), int(pinned or TIER_MAX_ROWS))
    rows -= rows % 4
    if rows <= 0:
        return None, f"The dense row capacity {inputs.dense_rows} holds no tier of four rows."
    return Tiers(sizes, meta, rows), reason


RULES = MappingProxyType({"diagonal": diagonal_rule, "packed": packed_rule, "paired": paired_rule, "tiers": tier_rule})
"""Each choice's rule, by the :class:`ClassPlan` field it fills. Their predicates exclude one another; a class none
holds keeps the general placement."""


def plan_class(inputs: ClassInputs, *, general: str | None) -> ClassPlan:
    """The class's plan: the choice whose rule of :data:`RULES` holds the class, and every rule's reason (``general``,
    :func:`general_reason`, when given). The held tree's plan follows from the choice: the sparse diagonal's or native
    sparse factors' admitted it (``held``), else it is built here (:func:`.held_tree.plan`, which raises ValueError for a model it
    does not hold)."""
    forest = inputs.forest
    reasons, choices = {}, {}
    for name, rule in RULES.items():
        choice, reasons[name] = (None, general) if general is not None else rule(inputs)
        if choice is not None:
            choices[name] = choice
    if len(choices) > 1:
        raise RuntimeError(f"FeatherPGS: the placement rules {sorted(choices)} both hold one class.")
    mf_sweeps = not ({"packed", "diagonal"} & choices.keys()) and forest.world_count > 0 and forest.max_world_dofs > 0
    held = next((choice.held for choice in choices.values() if isinstance(choice, Diagonal | Packed)), None)
    if held is None:
        try:
            held = held_tree.plan(inputs)
        except ValueError as error:
            raise ValueError(f"FeatherPGS cannot hold this model: {error}") from error
    return ClassPlan(reasons=MappingProxyType(reasons), mf_sweeps=mf_sweeps, held=held, **choices)

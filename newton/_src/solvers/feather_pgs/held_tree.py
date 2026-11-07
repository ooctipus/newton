# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The held-tree owner: the evaluator of articulated dynamics, its groups planned from the world classes.

The plan (:func:`plan`, held by the class plan) is data: one :class:`HeldClass` a world class, each a list of group
plans (:class:`HeldGroupPlan`) whose kind the class's data chooses: the sparse class, when the solver admitted one, makes
one complete group that walks its pattern (``"sparse"``); otherwise each tree shape is a group that responds by its walk
or by the compact diagonal (``"tree"``), kinematic free bodies a zero-response group (``"prescribed"``), articulations
without coordinates a group of none (``"rest"``), a compact diagonal family of trees rigid below their leaves the
one-coordinate class (``"coordinate"``, :mod:`.coordinate_trees`), or one group per world spans its articulations
(``"forest"``). The classes' groups of one key compose into one group: trees of one shape share one factor plan,
whatever their class. The owner (:class:`HeldTree`) builds each group (:class:`~.tree_kinetic_owner.TreeKineticState`
over a :class:`HeldGroup`, or the one-coordinate class's :class:`~.coordinate_trees.CoordinateTrees`) and decides
nothing. Every group's collection writes its drives' K-free mass (``row_H``) where it forms the mass diagonal; the
compact diagonal's refresh forms its inverse mass. :func:`decline` names why a representation does not fit;
:func:`install` returns the owner.
"""

import dataclasses
import itertools
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp

from . import coordinate_trees, field_group, host_index
from .device_proof import DeviceProof
from .tree_kinetic_owner import TreeKineticState, proof_runs
from .tree_kinetic_plan import make_plan, tree_joints
from .tree_kinetic_state import GroupTrees, HeldResponse
from .tree_plan import TreePlan, local_parents


def operator_metadata(mass) -> dict:
    """What the held groups read on the host of a tree's mass pattern ``mass`` (:class:`.tree_plan.FactorPattern`):
    its coordinate lookup (``index``), each coordinate's joint (``dof_joint``) and its inverse columns' rows
    (``inverse_nodes``, whose width the layout records)."""
    counts = np.diff(mass.column_offsets).astype(np.int32)
    nodes = np.full((mass.dof_count, int(counts.max(initial=0))), -1, np.int32)
    col, rank = host_index.expand(counts)
    nodes[col, rank] = mass.entry_rows[mass.column_entries]
    return {"index": mass.lookup, "dof_joint": mass.dof_joint, "inverse_nodes": nodes}


def bind_group_trees(solver, host, group_to_art) -> GroupTrees:
    """A held group's trees (:data:`.tree_kinetic_state.GROUP_TREES`): the articulations ``group_to_art`` and the
    template tree's joint of each coordinate from ``host`` (:func:`operator_metadata`)."""
    model = solver.model
    return field_group.upload(
        GroupTrees,
        model.device,
        dof_joint=host["dof_joint"],
        group_to_art=group_to_art,
        art_to_world=solver.forest.art_to_world,
        art_dof_start=solver.forest.articulation_dof_start,
        art_joint_start=model.articulation_start,
        joint_child=model.joint_child,
    )


@wp.kernel(enable_backward=False)
def _diagonal_geometry(
    group_to_art: wp.array[int],
    dof_start: wp.array[int],
    mask: wp.array[int],
    offset: int,
    geometry_group: wp.array[int],
    diagonal: wp.array2d[float],
    drive_row: wp.array[int],
    drive_k: wp.array[float],
    n: int,
    inverse: wp.array[float],
):
    """The compact diagonal's inverse mass ``1 / (H_ii + R + K)`` from ``H_ii + R`` as the collection keeps it (its held
    group's coordinate ``offset + local``), with the ``K`` the drives hold now: the rows' step on a key
    (:func:`.walk_rows.key_step`) and the sweep's leave of it."""
    element = wp.tid()
    group, local = element // n, element % n
    art = group_to_art[group]
    if mask[art] == 0:
        return
    dof = dof_start[art] + local
    mass = diagonal[geometry_group[group], offset + local]
    row = drive_row[dof]
    if row >= 0:
        if drive_k[row] > 0.0:
            mass += drive_k[row]
    inverse[dof] = 1.0 / mass


def decline(forest, factor, *, sparse: bool) -> str | None:
    """Why the held tree does not evaluate a representation of the dynamics of ``forest``, or ``None`` when it does.

    The terms of the representation: the native sparse factor's group (``sparse``), then the drive rows every dense
    group reads (the ``factor``'s drive plan). A term is read only once the earlier ones hold. The owner evaluates
    dynamics, not rows: the row laws (mimic rows, velocity limits) are the writers' and the sweeps'.
    """
    # The owner holds no rows, impulses or patch frames: the patch build, link, seed and finish stay on the solver
    # pipeline around it, so it serves every contact law. Torsion rows read the joint motions and body twists every
    # group publishes (joint_S_s, body_v_s); compliance reads no dynamics.
    if sparse:
        if forest.has_free_rigid_bodies:
            return "The sparse factor's group holds no free rigid bodies."
        if forest.has_prescribed_response:
            return "The sparse factor's group holds no prescribed articulations."
    if not factor.parallel_drives and forest.size_groups:
        return "The held tree needs the per-coordinate drive preparation (augmented drives, owned drive rows)."
    return None


def _prescribed_articulations(forest):
    """Articulations whose response the solver prescribes (kinematic free bodies)."""
    return np.flatnonzero(forest.prescribed_articulation.numpy() != 0).astype(np.int32)


def _coverage(model, forest, compact):
    """Prove disjoint articulation and physical coordinate ownership; ``compact`` is the compact diagonal size the
    groups carry. An articulation no size group holds has no coordinates (a ``"rest"`` group), and a body outside
    the articulations stays outside the solver."""
    prescribed = _prescribed_articulations(forest)
    owned, _ = tree_joints(model)
    arts = np.concatenate([forest.group_to_art[size].numpy() for size in forest.size_groups] + [prescribed])
    if np.unique(arts).size != arts.size:
        raise ValueError("Mixed tree factors must hold an articulation at most once")
    starts = model.articulation_start.numpy()
    if starts[0] != 0 or starts[-1] != model.joint_count or np.any(np.diff(starts) <= 0):
        raise ValueError("Mixed tree cache requires contiguous nonempty articulations")
    qd = model.joint_qd_start.numpy()
    ends = forest.articulation_joint_end_host
    ds = forest.articulation_dof_start.numpy()
    if not np.array_equal(ds, qd[starts[:-1]]):
        raise ValueError("Mixed tree factor starts must preserve physical coordinate ownership")
    for size in forest.size_groups:
        group = forest.group_to_art[size].numpy()
        if np.any(qd[ends[group]] - qd[starts[group]] != size):
            raise ValueError("Mixed tree factor size disagrees with its physical coordinates")
        if forest.n_arts_by_size[size] != len(group):
            raise ValueError("Mixed tree factor count disagrees with its group map")
    if prescribed.size and np.any(qd[ends[prescribed]] - qd[starts[prescribed]] != 6):
        raise ValueError("Prescribed cached roots must be six-coordinate free bodies")
    if np.unique(model.joint_child.numpy()[owned]).size != np.count_nonzero(owned):
        raise ValueError("Mixed tree cache must publish a body at most once")
    return (
        tuple(forest.size_groups),
        tuple(sorted(forest.n_arts_by_size.items())),
        compact,
        int(prescribed.size),
    )


def _template(model, forest, first_art, last_art=None, *, coordinates=True):
    """The mass pattern of the trees of articulations ``first_art`` through ``last_art`` (one tree by default; a
    world's forest in local order), from their local parents and each joint's coordinates (none without
    ``coordinates``)."""
    first = int(model.articulation_start.numpy()[first_art])
    end = int(forest.articulation_joint_end_host[first_art if last_art is None else last_art])
    parents = local_parents(model.joint_ancestor.numpy(), [first], end - first)[0]
    counts = np.diff(model.joint_qd_start.numpy())[first:end] if coordinates else np.zeros(end - first, np.int32)
    return TreePlan.build(parents, counts).factor


def _fit(model, forest, group_to_art, mass, *, prescribed=False, complete=False):
    """The metadata (:func:`operator_metadata`) of a held group of the trees ``group_to_art`` (articulations, on the
    host) on the pattern ``mass``, once its layout fits (:func:`.make_plan`; raises ValueError where it does not)."""
    host = operator_metadata(mass)
    trees = wp.array(group_to_art, dtype=int, device="cpu")
    plan = SimpleNamespace(group_to_art=trees, art_to_world=forest.art_to_world)
    make_plan(model, SimpleNamespace(plan=plan, host=host, prescribed=prescribed), complete=complete)
    return host


def _topologies(forest, size) -> list:
    """The ranges of size group ``size`` that hold one topology each (:attr:`.forest_plan.ForestPlan.size_topologies`)."""
    bounds = forest.size_topologies[size]
    return [(int(first), int(last)) for first, last in itertools.pairwise(bounds)]


def _size_trees(model, forest, size, trees, *, diagonal):
    """The metadata of the trees ``trees`` (articulations of size group ``size``, one topology): their mass pattern's,
    once a held group of them fits. Raises ValueError where they do not."""
    mass = _template(model, forest, int(trees[0]))
    if mass.dof_count != size:
        raise ValueError("Mixed tree physical and factor dimensions differ")
    if diagonal and mass.nonzero_count != size:
        raise ValueError("Compact diagonal cache requires independent physical coordinates")
    return _fit(model, forest, trees, mass)


def sparse_decline(inputs, size, plan) -> str | None:
    """Why the held tree would not take a native sparse factor of ``size`` coordinates and pattern ``plan``, or
    ``None``: :func:`decline` of the candidate, then its one complete group's fit.

    Only the held tree evaluates native sparse factors, so their rule (:func:`.class_plan.packed_rule`) reads this,
    with the class's ``inputs`` (:class:`.class_plan.ClassInputs`), before choosing one.
    """
    reason = decline(inputs.forest, inputs.factor, sparse=True)
    if reason is not None:
        return reason
    try:
        _fit(inputs.model, inputs.forest, inputs.forest.group_to_art_host[size], plan, complete=True)
    except ValueError as error:
        return f"The topology does not fit the held groups: {error}"
    return None


_FREE_BODY = TreePlan.build([-1], [6]).factor
"""The mass pattern of a prescribed free body: one body of six coordinates."""


@dataclass(frozen=True, eq=False)
class HeldGroupPlan:
    """One group of a world class's held plan (:class:`HeldClass`): its trees, their pattern and how they respond.

    The groups of two classes with one ``key`` hold trees of one shape on one factor plan, so the held tree composes
    them into one group (:func:`compose`).
    """

    kind: str
    """``"sparse"``, ``"tree"``, ``"prescribed"``, ``"rest"``, ``"forest"`` or ``"coordinate"`` (the one-coordinate
    class, :mod:`.coordinate_trees`)."""
    key: tuple
    """The group's identity across classes, its kind's rank first: composed groups keep the builders' order."""
    trees: np.ndarray
    """Each tree's articulation; a forest's row holds a world's articulations."""
    order: np.ndarray
    """Each tree's place among the trees of its key in every class (a forest row's: its world)."""
    host: dict | None = None
    """The pattern's metadata (:func:`operator_metadata`); ``None`` for the one-coordinate class, which holds none."""
    size: int | None = None
    response: str = "cholesky"
    """``"sparse"``, ``"cholesky"`` (by the walk), or ``"diagonal"`` (a compact diagonal family, or no response)."""
    root_families: tuple = ()
    """Each root's family: its coordinates, negated for the compact diagonal's (its walk is the compact inverse: no walk
    holds it), 0 when prescribed; a forest's per root, a compact diagonal's tree group's one root, empty for the other
    groups (each tree walks its coordinates)."""

    @property
    def diagonal(self) -> np.ndarray:
        """The articulations of the group's diagonal roots (a negative family), which no walk holds."""
        trees = self.trees if self.kind == "forest" else self.trees.reshape(-1, 1)
        roots = [r for r, family in enumerate(self.root_families) if family < 0]
        return trees[:, roots].reshape(-1).astype(np.int32)


@dataclass(frozen=True, eq=False)
class HeldClass:
    """A world class's held plan: how its trees group, and why (:func:`plan`)."""

    worlds: np.ndarray
    """The class's worlds (:attr:`.forest_plan.WorldClass.worlds`)."""
    grouping: str
    """``"sparse"`` (its share of the one complete group), ``"forest"`` (one group a world) or ``"per_size"`` (a group a
    tree shape, beside the prescribed and rest groups)."""
    reason: str | None
    """Why the class takes a forest or per-size groups (``None`` for ``"sparse"``)."""
    groups: tuple[HeldGroupPlan, ...]


@dataclass(frozen=True, eq=False)
class HeldPlan:
    """The held tree's plan: one :class:`HeldClass` a world class, and their groups composed (:func:`plan`)."""

    classes: tuple[HeldClass, ...]
    groups: tuple[HeldGroupPlan, ...]
    """The classes' groups, one a key (:func:`compose`): what the held tree builds."""
    compact: int | None
    """The compact diagonal's size (case K2's star), or ``None``."""
    coverage: tuple | None
    """The dense groups' coverage (:func:`_coverage`), which the held tree proves on notification; ``None`` for the
    sparse group or no tree."""
    walking: np.ndarray
    """The articulations the walk holds: every one with coordinates but the compact diagonal's, whose walk is the
    compact inverse (:attr:`HeldGroupPlan.diagonal`, and the one-coordinate class's trees)."""
    drive_dofs: np.ndarray | None
    """The coordinates whose drives the drive stage prepares, ``None`` for every one: a one-coordinate tree's predict
    decides its own."""
    coordinate: coordinate_trees.CoordinateTreesPlan | None = None
    """The compact diagonal family as the one-coordinate class (:func:`.coordinate_trees.plan`), or ``None``."""
    coordinate_reason: str = "No compact diagonal family."
    """Why the compact diagonal family takes the one-coordinate class or not."""


def _forest_families(inputs, worlds, member, compact):
    """The per-root families of one group a world of ``worlds`` (0 for a prescribed root) and each world's
    articulations, a row a world, or ``None`` where the layout admits no forest; ``member`` marks the worlds'
    articulations."""
    model, forest = inputs.model, inputs.forest
    if decline(forest, inputs.factor, sparse=False) is not None:
        return None
    art_world = forest.articulation_world
    prescribed = forest.prescribed_articulation_host
    counts = forest.articulation_dof_count_host
    arts = np.flatnonzero(member)
    world_bounds, order = host_index.group_by(np.searchsorted(worlds, art_world[arts]), worlds.size)
    world_arts = np.diff(world_bounds)
    if not worlds.size or np.any(world_arts != world_arts[0]) or world_arts[0] == 0:
        return None
    if world_arts[0] > 4 or world_arts[0] < 2:
        return None
    # Each world's articulations in ascending order, one row per world.
    per_world = arts[order].reshape(worlds.size, world_arts[0])
    if np.any(np.diff(per_world, axis=1) != 1):
        return None
    if np.any(counts[per_world] != counts[per_world[0]]) or np.any(prescribed[per_world] != prescribed[per_world[0]]):
        return None
    # One group holds a world's trees: at most 64 bodies and coordinates, each family a warp's lanes per row.
    bodies = forest.articulation_joint_end_host - model.articulation_start.numpy()[:-1]
    if bodies[per_world[0]].sum() > 64 or counts[per_world[0]].sum() > 64:
        return None
    families = [0 if prescribed[a] else int(counts[a]) for a in per_world[0]]
    for size in sorted({f for f in families if f}):
        group = forest.group_to_art_host[size]
        places = np.flatnonzero(member[group])
        # The class's trees of the size are one topology: one range of the size group.
        if np.unique(np.searchsorted(forest.size_topologies[size], places, side="right")).size > 1:
            return None
        if size != compact and (not 1 <= size <= 32 or size in forest.diagonal_mass_sizes):
            # A star's group other than the compact diagonal's keeps a group of its own, on every device: the CPU plans
            # the held family as CUDA does.
            return None
        if places.size != worlds.size or not np.array_equal(art_world[group[places]], worlds):
            return None
    return families, per_world


def _forest_group(inputs, families, per_world, compact) -> HeldGroupPlan:
    """One group a world over each world's articulations ``per_world``: each root responds by its walk, by the compact
    diagonal or with zero when prescribed. Its key is the layout every row's articulations must share."""
    model, forest = inputs.model, inputs.forest
    # Family per root: its coordinate count for a walking root, negated for a diagonal-mass size (compact inverse), 0
    # when prescribed.
    roots = tuple(-family if family == compact else family for family in families)
    first, last = int(per_world[0, 0]), int(per_world[0, -1])
    mass = _template(model, forest, first, last)
    joints = slice(int(model.articulation_start.numpy()[first]), int(forest.articulation_joint_end_host[last]))
    layout = (
        local_parents(model.joint_ancestor.numpy(), [joints.start], joints.stop - joints.start)[0].tobytes(),
        model.joint_type.numpy()[joints].tobytes(),
        model.joint_dof_dim.numpy()[joints].tobytes(),
    )
    return HeldGroupPlan(
        "forest",
        (1, roots, *layout),
        per_world.astype(np.int32),
        forest.articulation_world[per_world[:, 0]],
        operator_metadata(mass),
        size=mass.dof_count,
        root_families=roots,
    )


def _part(arts, member, offset=0):
    """The members of ``arts`` (a range of the solver's order from ``offset``) and their places in that order."""
    places = np.flatnonzero(member[arts])
    return arts[places].astype(np.int32), offset + places


def _size_groups(inputs, member, compact, coordinate) -> list:
    """The per-size groups of a class's articulations ``member``: a group a tree shape (each size group's topology
    range, the compact diagonal's with its family), the one-coordinate class in place of its size's, the prescribed free
    bodies and a group a rest topology."""
    model, forest = inputs.model, inputs.forest
    groups = []
    for rank, size in enumerate(forest.size_groups):
        if coordinate is not None and size == coordinate.size:
            trees, places = _part(forest.group_to_art_host[size], member)
            if trees.size:
                groups.append(HeldGroupPlan("coordinate", (5,), trees, places, None, size, "diagonal", (-size,)))
            continue
        diagonal = size == compact
        for first, last in _topologies(forest, size):
            trees, places = _part(forest.group_to_art_host[size][first:last], member, first)
            if trees.size:
                host = _size_trees(model, forest, size, trees, diagonal=diagonal)
                if diagonal:
                    # The compact diagonal family: its trees' walk is the compact inverse (a negative family).
                    groups.append(
                        HeldGroupPlan("tree", (2, rank, first), trees, places, host, size, "diagonal", (-size,))
                    )
                else:
                    groups.append(HeldGroupPlan("tree", (2, rank, first), trees, places, host, size))
    trees, places = _part(_prescribed_articulations(forest), member)
    if trees.size:
        host = _fit(model, forest, trees, _FREE_BODY, prescribed=True)
        groups.append(HeldGroupPlan("prescribed", (3,), trees, places, host, size=6, response="diagonal"))
    for first, last in itertools.pairwise(forest.rest_topologies):
        trees, places = _part(forest.rest_articulations[first:last], member, int(first))
        if trees.size:
            host = _fit(model, forest, trees, _template(model, forest, int(trees[0]), coordinates=False))
            groups.append(HeldGroupPlan("rest", (4, int(first)), trees, places, host, size=0, response="diagonal"))
    return groups


def held_class(inputs, worlds, *, packed=None, compact=None, coordinate=None) -> HeldClass:
    """The held plan of the world class of ``worlds``, with the class plan's choices: the native sparse factor
    (``packed``, :class:`.class_plan.Packed`), the compact diagonal's size (``compact``) and its family as the
    one-coordinate class (``coordinate``, :class:`.coordinate_trees.CoordinateTreesPlan`). Raises ValueError where its
    trees do not fit a group.

    The sparse factor makes one complete group of every class. Otherwise the class takes one group a world (a forest)
    where the test pin ``grouping`` or the default rule (:func:`forest_default`) asks and its layout admits one, else a
    group a tree shape beside its prescribed and rest groups.
    """
    forest = inputs.forest
    member = np.isin(forest.articulation_world, worlds)
    if packed is not None:
        trees, places = _part(forest.group_to_art_host[packed.size], member)
        group = HeldGroupPlan("sparse", (0,), trees, places, operator_metadata(packed.plan), response="sparse")
        return HeldClass(worlds, "sparse", None, (group,) if trees.size else ())
    override = inputs.pins.grouping
    forest_layout = None if coordinate is not None else _forest_families(inputs, worlds, member, compact)
    if override is not None:
        wants = override == "forest"
    else:
        wants = forest_default(forest_layout, compact)
    if wants and forest_layout is not None:
        reason = (
            "A test pin selects one group per world (grouping='forest')."
            if override is not None
            else "Default: a prescribed root and no diagonal-mass family make one group per world cheaper."
        )
        return HeldClass(worlds, "forest", reason, (_forest_group(inputs, *forest_layout, compact),))
    if wants and coordinate is not None:
        reason = "The one-coordinate class owns the compact diagonal family, so no group spans a world."
    elif wants:
        reason = "The world layout does not admit one forest group."
    elif override is not None:
        reason = "A test pin keeps per-size groups (grouping='per_size')."
    else:
        reason = (
            "Default: per-size groups (the layout admits no forest, has no prescribed root, or has a "
            "diagonal-mass family the forest applies at a loss)."
        )
    return HeldClass(worlds, "per_size", reason, tuple(_size_groups(inputs, member, compact, coordinate)))


def forest_default(layout, compact) -> bool:
    """Whether a class's forest ``layout`` (:func:`_forest_families`, ``None`` where it admits none) is its default
    grouping: a prescribed root and no diagonal-mass family.

    The forest's gain is one group and one launch set per world where the
    per-size groups would run a third, zero-response group for a prescribed
    root: measured -9% on a 9+6+prescribed world and -1% on 23+6+prescribed.
    Without a prescribed root the per-size groups' lane packing wins (+3% for
    the forest on a 16+6 world), and a diagonal-mass family, which the per-size
    owner applies in a two-lane block, costs the forest 5% (9+4 with a drawer).
    """
    return layout is not None and 0 in layout[0] and compact not in layout[0]


def compose(classes) -> tuple[HeldGroupPlan, ...]:
    """The held tree's groups: the classes' groups of one key as one (trees of one shape share one factor plan), each
    group's trees in the solver's order, the groups in the builders' order."""
    keyed = {}
    for held in classes:
        for group in held.groups:
            keyed.setdefault(group.key, []).append(group)
    groups = []
    for key in sorted(keyed):
        parts = keyed[key]
        order = np.concatenate([part.order for part in parts])
        sort = np.argsort(order, kind="stable")
        trees = np.concatenate([part.trees for part in parts])[sort]
        groups.append(dataclasses.replace(parts[0], trees=trees, order=order[sort]))
    return tuple(groups)


def plan(inputs, *, packed=None, compact=None) -> HeldPlan:
    """The held tree's plan: one :class:`HeldClass` a world class (:attr:`.forest_plan.ForestPlan.world_classes`) with
    the class plan's choices (:func:`held_class`) and the compact diagonal family's class (the one-coordinate class
    where its trees are rigid below their leaves, :func:`.coordinate_trees.plan`), their groups composed, the dense
    groups' coverage, the walk's trees and the drive stage's coordinates.

    The plan's failure is the held tree's decline of the choices: the sparse diagonal's rule
    (:func:`.class_plan.diagonal_rule`) builds the plan with its compact diagonal and keeps it, or declines with why.

    Raises:
        ValueError: For a model the held tree does not hold, naming why (:func:`decline`, or the group that does not
            fit).
    """
    model, forest = inputs.model, inputs.forest
    reason = decline(forest, inputs.factor, sparse=packed is not None)
    if reason is not None:
        raise ValueError(reason)
    coordinate, coordinate_reason = coordinate_trees.plan(inputs, size=None if packed is not None else compact)
    drive_dofs = None
    if coordinate is not None:
        other = np.ones(model.joint_dof_count, dtype=bool)
        other[coordinate.dofs[(coordinate.dofs >= 0) & (coordinate.dofs < model.joint_dof_count)]] = False
        drive_dofs = np.flatnonzero(other).astype(np.int32)
    walking = np.flatnonzero(forest.articulation_dof_count_host > 0)
    if not (model.joint_count and model.articulation_count):
        return HeldPlan((), (), compact, None, walking, drive_dofs, coordinate, coordinate_reason)
    try:
        coverage = None if packed is not None else _coverage(model, forest, compact)
        classes = tuple(
            held_class(inputs, world_class.worlds, packed=packed, compact=compact, coordinate=coordinate)
            for world_class in forest.world_classes
        )
    except ValueError as error:
        raise ValueError(f"The topology does not fit the held groups: {error}") from error
    groups = compose(classes)
    walking = np.setdiff1d(walking, np.concatenate([walking[:0], *(group.diagonal for group in groups)]))
    return HeldPlan(classes, groups, compact, coverage, walking, drive_dofs, coordinate, coordinate_reason)


class HeldGroup:
    """One held group built from its plan (:class:`HeldGroupPlan`): the trees ``group_to_art`` (for a forest, each
    world's first; ``forest_arts`` holds them all) on one pattern, whose metadata is ``host``
    (:func:`operator_metadata`), with their response's status.

    ``kind`` names what the plan chose: ``"sparse"``, ``"tree"``, ``"prescribed"``, ``"rest"`` or ``"forest"``. Every
    group responds through the walk, whose refresh is the factor's, with zero, or on a diagonal root (the compact
    diagonal's) by one division by its mass; its collection writes the drives' K-free mass (``row_H``) where it forms
    ``H_ii``. Its own refresh forms only the inverse mass of its compact diagonal ``families``, each ``(size, offset,
    group_to_art, geometry_group)``: the articulations of one diagonal size and, per articulation, the group whose
    collection keeps its K-free mass from coordinate ``offset``.
    """

    requires_body_velocity = True

    def __init__(self, solver, group: HeldGroupPlan):
        device = solver.model.device
        self.solver, self.kind, self.host, self.response = solver, group.kind, group.host, group.response
        self.size = None if group.kind == "sparse" else group.size
        self.prescribed = group.kind == "prescribed"
        forest = group.kind == "forest"
        self.forest_arts = group.trees if forest else None
        self.root_families = group.root_families
        group_to_art = wp.array(group.trees[:, 0].copy() if forest else group.trees, dtype=int, device=device)
        self.plan = bind_group_trees(solver, group.host, group_to_art)
        self.families = self._families(group)
        # The compact diagonal's predictor and refresh read its K-free mass, which the collection keeps.
        self.diagonal_mass = bool(self.families)
        count = self.plan.group_to_art.size
        self.data = HeldResponse()
        # The walk has no factor status: a pivot that is not positive reaches the predictor as a non-finite response.
        self.data.valid = wp.ones(count, dtype=int, device=device)
        self.data.status = wp.zeros(count, dtype=int, device=device)
        self.kinetic_state = None

    def _families(self, group) -> tuple:
        """The compact diagonal families: each diagonal root's (a negative family: a tree group's one root, or a
        forest's), whose refresh reads its coordinates' K-free mass from the group's collection, from the root's offset
        (the coordinates of the world's trees before it)."""
        device = self.solver.model.device
        if not any(family < 0 for family in group.root_families):
            return ()
        trees = group.trees if group.kind == "forest" else group.trees.reshape(-1, 1)
        counts = self.solver.forest.articulation_dof_count_host[trees[0]]
        offsets = np.concatenate(([0], np.cumsum(counts)))
        rows = wp.array(np.arange(trees.shape[0], dtype=np.int32), dtype=int, device=device)
        return tuple(
            (-family, int(offsets[r]), wp.array(trees[:, r].copy(), dtype=int, device=device), rows)
            for r, family in enumerate(group.root_families)
            if family < 0
        )

    def refresh(self, state_aug):
        """Each compact diagonal family's inverse mass from its groups' current geometry."""
        solver = self.solver
        for size, offset, group_to_art, geometry_group in self.families:
            wp.launch(
                _diagonal_geometry,
                dim=group_to_art.size * size,
                inputs=[
                    group_to_art,
                    solver.forest.articulation_dof_start,
                    solver.factor.mass_update_mask,
                    offset,
                    geometry_group,
                    self.kinetic_state.mass_diagonal,
                    solver.factor.drive_row_by_dof,
                    solver.factor.aug_row_K,
                    size,
                ],
                outputs=[solver.factor.diagonal_inverse_mass],
                device=solver.model.device,
            )


class HeldTree:
    """Evaluate every articulation's dynamics in the held groups of its plan (:class:`HeldPlan`; see the module
    docstring)."""

    def __init__(self, solver, plan: HeldPlan):
        self.solver, self.plan = solver, plan
        # One read of the stage clocks for every group this owner composes.
        self.clocked = solver.options.diagnostics.stage_clocks
        self.coordinate_trees = None
        self.coverage = plan.coverage
        # The drive stage's coordinates: every one (None), or those no one-coordinate tree's predict decides.
        self.drive_dofs = None
        if plan.drive_dofs is not None and plan.drive_dofs.size:
            self.drive_dofs = wp.array(plan.drive_dofs, dtype=int, device=solver.model.device)
        self.prepares_drives = plan.drive_dofs is None or plan.drive_dofs.size > 0
        if self.coverage is not None:
            self._dimensions = (solver.model.joint_count, solver.model.body_count, solver.model.articulation_count)
            # The coverage proof's inputs, compared on the device on notification.
            self._proof = DeviceProof(list(self._coverage_arrays()), solver.model.device)
        self.groups = []
        for group in plan.groups:
            if group.kind == "coordinate":
                # A tree group's proof holds the model topology, the joint axes and art_to_world and watches the joint
                # frames, and the coverage proof every size's group map, so the one-coordinate class needs no proof of
                # its own: the sparse-diagonal pair always leaves a tree group beside it.
                self.coordinate_trees = coordinate_trees.CoordinateTrees(solver, group, plan.coordinate)
                self.groups.append(self.coordinate_trees)
                continue
            operator = HeldGroup(solver, group)
            complete = group.kind == "sparse"
            state = TreeKineticState(
                operator,
                chains=complete,
                complete=complete,
                response=operator.response,
                clocked=self.clocked,
            )
            operator.kinetic_state = state
            self.groups.append(state)

    @property
    def reasons(self) -> tuple:
        """Each world class's grouping and why (:attr:`HeldClass.reason`)."""
        return tuple(held.reason for held in self.plan.classes)

    def _coverage_arrays(self):
        solver, model = self.solver, self.solver.model
        for size in solver.forest.size_groups:
            yield f"group_to_art[{size}]", solver.forest.group_to_art[size]
        yield "articulation_start", model.articulation_start
        yield "joint_qd_start", model.joint_qd_start
        yield "joint_child", model.joint_child
        yield "articulation_dof_start", solver.forest.articulation_dof_start
        yield "prescribed_articulation", solver.forest.prescribed_articulation

    def validate_model(self, flags=None):
        """Reject ownership changes before any cache or notification mutation.

        Returns:
            Whether every group's proof ran and found its inputs unchanged (see
            :meth:`TreeKineticState.validate_model`). The one-coordinate class,
            without a proof of its own, counts as passed: its tree neighbours
            prove the same model arrays.
        """
        if not proof_runs(flags):
            # Numeric notifications cannot change ownership; see TreeKineticState.
            return False
        solver = self.solver
        if self.coverage is not None:
            sizes = (
                tuple(solver.forest.size_groups),
                tuple(sorted(solver.forest.n_arts_by_size.items())),
                self.plan.compact,
            )
            if (
                solver.model.joint_count,
                solver.model.body_count,
                solver.model.articulation_count,
            ) != self._dimensions or sizes != self.coverage[:3]:
                raise RuntimeError("Mixed tree factor groups changed; reconstruct and recapture")
            current = dict(self._coverage_arrays())
            self._proof.validate(current.__getitem__, "Mixed tree factor")
        # Each group's plan binds the same group map it was built from.
        proven = [group.validate_model(flags) for group in self.groups]
        return bool(proven) and all(proven)

    def invalidate(self, world_mask=None):
        """Invalidate all component caches for selected worlds."""
        for group in self.groups:
            group.invalidate(world_mask)

    def begin(self, state, augmented, velocity, dt, *, state_out=None):
        """Repair every current component before any factor or force consumer (see :meth:`TreeKineticState.begin`):
        each group's device decides which of its trees recompute."""
        for group in self.groups:
            group.begin(state, augmented, velocity, dt, state_out=state_out)

    def refresh(self, augmented):
        """Preserve independent factor masks and numerical representations."""
        for group in self.groups:
            group.sparse.refresh(augmented)

    def predict(self, state, augmented, control, velocity, dt):
        """Project current forces and apply each original held response."""
        for group in self.groups:
            group.predict(state, augmented, control, velocity, dt)

    def finish(self, state, augmented, output, dt):
        """Integrate and publish each physical coordinate and body once."""
        for group in self.groups:
            group.finish(state, augmented, output, dt)


def install(solver) -> HeldTree:
    """The owner of the solver's dynamics, built from the class plan's held plan (:attr:`.class_plan.ClassPlan.held`,
    which names why a model does not fit). A group that does not fit raises ``ValueError``, naming why."""
    try:
        return HeldTree(solver, solver.spec.classes.held)
    except ValueError as error:
        raise ValueError(f"FeatherPGS cannot hold this model: {error}") from error

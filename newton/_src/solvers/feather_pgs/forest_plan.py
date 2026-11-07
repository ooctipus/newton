# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The forest plan: the model's articulation forest as every other primitive reads it, built once per model.

:meth:`ForestPlan.build` derives, from the model's topology alone:

- the joint forest (:func:`joint_forest`) and each articulation's tree, coordinates and world;
- one root rule (a FREE or DISTANCE joint without a parent roots a free tree), from which the free rigid bodies, the
  free-rooted articulations and the free-root joints follow;
- the prescribed free bodies the matrix-free formulation elides, and so each articulation's response coordinates;
- the size groups (articulations by response coordinate count) and the world mapping;
- the body maps (each body's tree joint, articulation and response support);
- the root cut: the joint forest with the joints above every coordinate cut away, whose trees are the coordinate
  trees, the vertices of the rows' graph (a fixed-base star is as many one-coordinate trees as it has keys), each
  body's among them and each world's, and each tree's shape;
- the world classes: worlds of equal tree shapes and equal row laws present, which one plan can serve;
- per size group, the mass matrix's coordinate-to-joint map and whether every tree has one coordinate (a star);
- with the matrix-free formulation, the compact per-world response mapping.

Host arrays end in ``_host`` where a device array of the same name exists. The plan is frozen: notifications that
change the model's topology need a new solver.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from ...geometry.types import GeoType
from ...sim import Model, ModelFlags
from ...sim.enums import JointType
from . import host_index
from .body_cut import BodyCut
from .body_mapping import build_body_maps
from .field_group import Field, FieldGroup, SolverArrays
from .tree_plan import TreePlan

BODY_MAPS = FieldGroup(
    Field("body_to_joint", wp.int32, ("body",), owner="forest"),
    Field("body_to_articulation", wp.int32, ("body",), owner="forest"),
    Field("body_has_response_dofs", wp.int32, ("body",), owner="forest"),
    Field("body_response_dof_mask", wp.uint32, ("body",), owner="forest"),
    Field("body_single_response_dof", wp.int32, ("body",), owner="forest"),
)
"""Each body's tree joint, articulation, response presence, low-32 response support and single response DOF."""


__all__ = ["ClassShape", "ForestPlan", "RowLaw", "TopologyPin", "WorldClass", "joint_forest", "row_support"]


NUMERIC_PROPERTIES = (
    ModelFlags.JOINT_DOF_PROPERTIES
    | ModelFlags.BODY_INERTIAL_PROPERTIES
    | ModelFlags.SHAPE_PROPERTIES
    | ModelFlags.MODEL_PROPERTIES
)
"""The model notifications that change only numbers (coefficients, limits, inertias, shape materials), never a row
owner's topology."""

JOINT_TOPOLOGY = ("joint_type", "joint_parent", "joint_child", "joint_q_start", "joint_qd_start", "body_flags")
"""The structural joint and body arrays every row owner's plan reads."""


@dataclass(frozen=True)
class TopologyPin:
    """The structural model arrays a row owner's plan was built from, read at construction (:meth:`read`)."""

    names: tuple[str, ...]
    data: tuple[bytes, ...]

    @classmethod
    def read(cls, model, names) -> TopologyPin:
        names = tuple(names)
        return cls(names, tuple(getattr(model, name).numpy().tobytes() for name in names))

    def validate(self, model, flags, holds, owner: str) -> None:
        """Refuse a notification beyond :data:`NUMERIC_PROPERTIES` that changes a pinned array or after which the
        owner's own check ``holds()`` fails; a numeric notification reads nothing back."""
        if flags and not (int(flags) & ~int(NUMERIC_PROPERTIES)):
            return
        if TopologyPin.read(model, self.names) != self or not holds():
            raise RuntimeError(f"{owner} topology changed; reconstruct the solver")


class RowLaw(enum.IntFlag):
    """The row laws a world's data gives rows, from the model at construction (the options decide which of them run)."""

    CONTACT = 1
    """A shape pair the collision filter admits (``Model.shape_contact_pairs``) on a body a tree moves."""
    LIMIT = 2
    """A joint axis with a finite position limit."""
    VELOCITY_LIMIT = 4
    """A joint axis with a positive velocity limit."""
    MIMIC = 8
    """A mimic constraint on a tree's joint."""
    LINK = 16
    """A link of the body cut."""
    SPIN = 32
    """A body pair the collision filter admits that can touch at two points or more, on a body a tree moves: a friction
    region with a footprint, whose wrench carries a spin row (:func:`.friction_patches.patch_spin_rows`)."""


_LIMITED_JOINTS = (int(JointType.PRISMATIC), int(JointType.REVOLUTE), int(JointType.D6))
"""The joints whose axes take limit rows."""
_ONE_POINT_SHAPES = (int(GeoType.SPHERE), int(GeoType.ELLIPSOID))
"""The shapes that touch a convex shape at one point."""
_MANIFOLD_SHAPES = (int(GeoType.MESH), int(GeoType.HFIELD), int(GeoType.GAUSSIAN))
"""The shapes outside the convex narrow phase: a pair with one can touch at several points whatever the other is."""


@dataclass(frozen=True, eq=False)
class WorldClass:
    """Worlds of equal tree shapes and equal row laws present: what one plan of their trees and rows serves."""

    shapes: tuple[int, ...]
    """The shape of each of a member world's trees (:attr:`ForestPlan.tree_shape`), ascending."""
    laws: RowLaw
    """The row laws present in each member world."""
    worlds: np.ndarray
    """The member worlds, ascending."""


@dataclass(frozen=True, eq=False)
class ClassShape:
    """Each world's responding articulations as blocks, a block a size group (descending), one articulation a world."""

    sizes: tuple[int, ...]
    free: tuple[bool, ...]
    arts: np.ndarray | None
    """``(world, block)``: each world's articulation, ``-1`` for none; ``None`` where a world holds two of one size."""
    groups: np.ndarray | None
    offsets: tuple[int | None, ...]
    """Each block's first world coordinate, ``None`` where its articulations differ."""
    coordinates: int

    @property
    def blocks(self) -> int:
        return len(self.sizes)

    def present(self, block: int) -> bool:
        return self.arts is not None and bool(np.all(self.arts[:, block] >= 0))

    @property
    def beside(self) -> bool:
        if self.arts is None:
            return False
        held, free = self.arts >= 0, np.asarray(self.free, dtype=bool)
        return bool(np.any(held[:, free].any(axis=1) & held[:, ~free].any(axis=1)))


def joint_forest(model: Model) -> TreePlan:
    """The forest of the model's articulation joints by their ancestry (``Model.joint_ancestor``); a joint
    outside every articulation stands alone."""
    if not model.joint_count:
        return TreePlan.build(np.zeros(0, dtype=np.int32))
    member = model.joint_articulation.numpy() >= 0
    try:
        return TreePlan.build(np.where(member, model.joint_ancestor.numpy(), -1))
    except ValueError as error:
        raise ValueError(f"SolverFeatherPGS: the model's joint ancestry is not a forest ({error})") from error


@dataclass(frozen=True)
class ForestPlan:
    """The model's articulation forest: topology, groups, root kinds, world mapping and body maps (frozen)."""

    cut: BodyCut
    """The cut that made the model a forest, and its links (:mod:`.body_cut`)."""
    joint_forest: TreePlan
    """The model's joint forest (:func:`joint_forest`)."""
    world_count: int
    """Worlds that hold an articulation (the largest articulation world plus one)."""

    # Articulations (host).
    articulation_dof_count_host: np.ndarray
    """Each articulation's tree coordinates."""
    articulation_dof_start_host: np.ndarray
    """Each articulation's first coordinate."""
    articulation_world: np.ndarray
    """Each articulation's world (a negative model world reads as world 0)."""
    articulation_joint_end_host: np.ndarray
    """Each articulation's end of its tree joints (a joint outside every tree may follow it)."""
    is_free_rigid_host: np.ndarray
    """Each articulation that is one free body (one FREE joint without a parent)."""
    response_free_rigid_body_indices: np.ndarray
    """The bodies of the free rigid bodies that respond (not prescribed)."""
    prescribed_articulation_host: np.ndarray
    """Each free body the matrix-free formulation elides: all six coordinates kinematic, not its world's only
    response."""
    response_dof_count: np.ndarray
    """Each articulation's response coordinates: its tree coordinates, none where prescribed."""

    # Articulations (device).
    prescribed_articulation: wp.array | None
    articulation_dof_count: wp.array | None
    """Each articulation's tree coordinates (``None`` without joints)."""
    articulation_dof_start: wp.array | None
    """Each articulation's first coordinate (``None`` without joints)."""
    articulation_max_dofs: int
    articulation_joint_end: wp.array | None
    articulation_response_dof_count: wp.array
    is_free_rigid: wp.array | None
    free_rigid_body_indices: wp.array | None
    free_rigid_body_world: wp.array | None
    """Each free rigid body's world, in the order of ``free_rigid_body_indices`` (the warm start's publication)."""

    # Roots.
    free_root_joint_count: int
    free_root_joint_indices: wp.array | None
    """Every joint that roots a free tree (FREE or DISTANCE without a parent), standalone joints included."""
    articulation_root_is_free: wp.array | None
    articulation_root_dof_start: wp.array | None
    has_root_free: bool
    has_prescribed_response: bool

    # Free bodies.
    free_rigid_body_count: int
    has_free_rigid_bodies: bool
    has_mixed_contacts: bool
    """A world holds both a free rigid body and another responding articulation."""
    has_non_free_articulations: bool
    free_rigid_per_world: np.ndarray
    """Responding free rigid bodies per world (at least one entry)."""

    # Coordinate trees (host), ascending by root joint.
    coordinate_forest: TreePlan
    """The root cut: the joint forest with every joint above all coordinates cut from its children, so each of them
    roots a tree. A tree is a root with coordinates and its subtree, which keeps the welds below its root; a joint
    above every coordinate is a root of its own without any."""
    tree_root: np.ndarray
    """Each tree's root joint."""
    tree_shape: np.ndarray
    """Each tree's shape: equal for trees of equal parents, joint types and axes in depth-first order."""
    tree_first: np.ndarray
    """Each tree's first coordinate. A tree's coordinates need not be a range: breadth-first joints interleave trees."""
    tree_size: np.ndarray
    """Each tree's coordinates."""
    tree_articulation: np.ndarray
    """Each tree's articulation."""
    world_tree_start: np.ndarray
    world_trees: np.ndarray
    """Each world's trees: ``world_trees[world_tree_start[w] : world_tree_start[w + 1]]``."""
    body_tree: np.ndarray
    """Each body's tree, ``-1`` where no solved coordinate moves it."""

    # World classes (host).
    world_laws: np.ndarray
    """The row laws present in each world (:class:`RowLaw` bits)."""
    world_class: np.ndarray
    """Each world's class among :attr:`world_classes`."""
    world_classes: tuple[WorldClass, ...]
    """The world classes, by their first world."""
    shape: ClassShape

    # Size groups.
    size_groups: list
    """Response coordinate counts, descending."""
    n_arts_by_size: dict
    group_to_art: dict
    """Each size group's articulations, by topology and ascending within one."""
    art_group_idx: wp.array | None
    group_to_art_host: dict
    """Each size group's articulations by group index, on the host."""
    rest_articulations: np.ndarray
    rest_topologies: np.ndarray
    """The articulations without coordinates, equal trees side by side, and the bounds of each run."""
    size_topologies: dict
    """Each size group's ranges of equal trees: bounds into its group indices, one range for one topology."""
    """Each articulation's index in its size group (``-1`` without response)."""
    free_body_inertia_sizes: frozenset
    """Size groups of free bodies only (the composite inertia is the body's)."""
    crba_dof_joint_offset_by_size: dict
    """Per topology-compatible size group, each coordinate's joint as an offset from the articulation's first."""
    diagonal_mass_sizes: frozenset
    """Size groups whose every tree has one coordinate (stars): their fill-free factor pattern is diagonal."""
    direct_diagonal_inertia_sizes: frozenset
    """Star size groups whose every coordinate's child body has it as its only response coordinate."""

    # Worlds.
    art_to_world: wp.array | None
    world_art_start: wp.array | None
    world_response_group_art_start: dict
    world_response_group_to_art: dict
    max_world_dofs: int
    """The widest world's response coordinates (the matrix-free formulation's world mapping)."""
    articulation_world_dof_offset: wp.array
    world_dof_count: wp.array
    world_dof_indices: wp.array

    # Bodies.
    body_to_joint: wp.array | None
    body_to_articulation: wp.array | None
    body_has_response_dofs: wp.array | None
    body_response_dof_mask: wp.array | None
    body_single_response_dof: wp.array | None
    body_single_response_dof_host: np.ndarray | None

    @classmethod
    def build(cls, cut: BodyCut, joint_forest: TreePlan, kinematic_dof_mask: np.ndarray) -> ForestPlan:
        """Build the plan from the forest model's topology.

        Args:
            cut: The body cut, whose :attr:`~.body_cut.BodyCut.model` is the model.
            joint_forest: The model's joint forest (:func:`joint_forest`).
            kinematic_dof_mask: Each coordinate the body flags make kinematic, at construction.
        """
        model = cut.model
        plan = SimpleNamespace(cut=cut, joint_forest=joint_forest)
        free_root = None
        if model.joint_count:
            # The one root rule: a FREE or DISTANCE joint without a parent roots a free tree.
            joint_type = model.joint_type.numpy()
            free_root = ((joint_type == JointType.FREE) | (joint_type == JointType.DISTANCE)) & (
                model.joint_parent.numpy() < 0
            )
        _articulations(plan, model, kinematic_dof_mask, free_root, cut.linked)
        plan.prescribed_articulation = wp.array(plan.prescribed_articulation_host, dtype=wp.int32, device=model.device)
        plan.has_prescribed_response = bool(np.any(plan.prescribed_articulation_host != 0))
        _coordinates(plan, model)
        _roots(plan, model, free_root)
        _size_groups(plan, model)
        _world_mapping(plan, model)
        _body_maps(plan, model)
        _free_bodies(plan, model)
        _coordinate_trees(plan, model)
        _world_classes(plan, model)
        _mass_schedules(plan, model)
        _world_size_groups(plan, model)
        _world_response_mapping(plan, model)
        _class_shape(plan)
        return cls(**vars(plan))


def _articulations(plan, model, kinematic_dof_mask, free_root, linked):
    """Each articulation's tree joints, coordinates, world, free body and prescribed response. A linked tree's rows are
    the bilateral block's, so a body on one is neither free nor prescribed: its rows and its kinematic armature stay in
    the block's reach."""
    articulation_count = model.articulation_count
    articulation_dof_count = np.zeros(articulation_count, dtype=np.int32)
    articulation_dof_start = np.zeros(articulation_count, dtype=np.int32)
    articulation_world = np.zeros(articulation_count, dtype=np.int32)
    is_free_rigid = np.zeros(articulation_count, dtype=np.int32)
    free_rigid_body_indices = np.zeros(0, dtype=np.int32)

    if articulation_count:
        if model.articulation_world is None:
            articulation_world = np.arange(articulation_count, dtype=np.int32)
        else:
            articulation_world = model.articulation_world.numpy().astype(np.int32, copy=True)
            articulation_world[articulation_world < 0] = 0

    articulation_joint_end = np.zeros(articulation_count, dtype=np.int32)
    if articulation_count and model.joint_count:
        articulation_start = model.articulation_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_type = model.joint_type.numpy()
        joint_child = model.joint_child.numpy()

        first_joint = articulation_start[:articulation_count]
        last_joint = articulation_start[1 : articulation_count + 1]

        # An articulation's range may hold joints outside every tree behind its own, and only those: its members must
        # open it, as many preceding each one as joints do.
        joint_articulation = model.joint_articulation.numpy()
        range_art, position = host_index.expand(np.maximum(last_joint - first_joint, 0))
        member = joint_articulation[first_joint[range_art] + position] == range_art
        member_count = np.bincount(range_art[member], minlength=articulation_count)
        members_before = host_index.segmented_rank(range_art, member)
        stray = np.flatnonzero(member & (members_before != position))
        if stray.size:
            art = int(range_art[stray[0]])
            j = int(first_joint[art] + position[stray[0]])
            tree_end = int(first_joint[art] + members_before[stray[0]])
            raise ValueError(
                f"SolverFeatherPGS: articulation {art}'s joints are not contiguous (joint {j} follows a foreign joint "
                f"at {tree_end}). The ModelBuilder guarantees contiguity; this model is malformed."
            )
        articulation_joint_end[:] = first_joint + member_count

        articulation_dof_start[:] = joint_qd_start[first_joint]
        articulation_dof_count[:] = joint_qd_start[articulation_joint_end] - articulation_dof_start
        single = np.flatnonzero(articulation_joint_end - first_joint == 1)
        root = first_joint[single]
        # A free rigid body: one tree joint, a FREE joint that roots a free tree.
        free = single[free_root[root] & (joint_type[root] == int(JointType.FREE)) & ~linked[single]]
        is_free_rigid[free] = 1
        free_rigid_body_indices = joint_child[first_joint[free]]

    prescribed = np.zeros(articulation_count, dtype=np.int32)
    if articulation_count and model.joint_count:
        candidates = np.zeros(articulation_count, dtype=np.int32)
        free_six = np.flatnonzero((is_free_rigid != 0) & (articulation_dof_count == 6))
        free_dofs = articulation_dof_start[free_six, None] + np.arange(6)
        candidates[free_six[np.all(kinematic_dof_mask[free_dofs] != 0, axis=1)]] = 1

        # Never elide the only response articulation in a world. Such
        # worlds retain the exact large-armature kinematic fallback.
        retained_response = np.zeros(int(np.max(articulation_world)) + 1, dtype=bool)
        retained_response[articulation_world[(articulation_dof_count > 0) & (candidates == 0)]] = True
        prescribed[(candidates != 0) & retained_response[articulation_world]] = 1

    response_dof_count = articulation_dof_count.copy()
    response_dof_count[prescribed != 0] = 0
    free_articulations = np.nonzero(is_free_rigid != 0)[0]
    free_bodies = np.asarray(free_rigid_body_indices, dtype=np.int32)
    response_free_rigid_body_indices = free_bodies[prescribed[free_articulations] == 0]

    arrays = {
        "articulation_dof_count_host": articulation_dof_count,
        "articulation_dof_start_host": articulation_dof_start,
        "articulation_world": articulation_world,
        "is_free_rigid_host": is_free_rigid,
        "response_free_rigid_body_indices": response_free_rigid_body_indices,
        "prescribed_articulation_host": prescribed,
        "response_dof_count": response_dof_count,
        "articulation_joint_end_host": articulation_joint_end,
    }
    for name, array in arrays.items():
        array.setflags(write=False)
        setattr(plan, name, array)
    plan.world_count = int(np.max(articulation_world)) + 1 if articulation_count else 0


def _coordinates(plan, model):
    """Each articulation's tree coordinate count and start on the device, and the largest count."""
    plan.articulation_dof_count = None
    plan.articulation_dof_start = None
    plan.articulation_max_dofs = 0
    if model.joint_count:
        articulation_count = model.articulation_count
        dof_count = plan.articulation_dof_count_host
        plan.articulation_dof_count = wp.array(dof_count, dtype=wp.int32, device=model.device)
        plan.articulation_dof_start = wp.array(plan.articulation_dof_start_host, dtype=wp.int32, device=model.device)
        plan.articulation_max_dofs = int(np.max(dof_count)) if articulation_count else 0


def _roots(plan, model, free_root):
    """The free-root joints and each articulation's root (the one root rule, :meth:`ForestPlan.build`)."""
    if model.joint_count:
        free_root_joint_indices = np.flatnonzero(free_root).astype(np.int32, copy=False)
        plan.free_root_joint_count = int(free_root_joint_indices.size)
        plan.free_root_joint_indices = (
            wp.array(free_root_joint_indices, dtype=wp.int32, device=model.device)
            if plan.free_root_joint_count
            else None
        )
    else:
        plan.free_root_joint_count = 0
        plan.free_root_joint_indices = None

    if not model.articulation_count or not model.joint_count:
        plan.articulation_root_is_free = None
        plan.articulation_root_dof_start = None
        plan.has_root_free = False
        return

    root_joint = model.articulation_start.numpy()[: model.articulation_count]
    root_dof_start = model.joint_qd_start.numpy()[root_joint].astype(np.int32, copy=False)
    root_is_free = free_root[root_joint].astype(np.int32)

    plan.articulation_root_is_free = wp.array(root_is_free, dtype=wp.int32, device=model.device)
    plan.articulation_root_dof_start = wp.array(root_dof_start, dtype=wp.int32, device=model.device)
    plan.has_root_free = bool(np.any(root_is_free != 0))


def _equal_ids(lengths: np.ndarray, rows) -> np.ndarray:
    """Equal ids for equal items whose keys are rows of varying length: ``rows(members, length)`` gives the key rows of
    the items ``members`` of one ``length``. Ids count up by length, then by key."""
    ids = np.zeros(lengths.size, dtype=np.int64)
    found = 0
    for length in np.unique(lengths):
        members = np.flatnonzero(lengths == length)
        _, inverse = np.unique(rows(members, int(length)), axis=0, return_inverse=True)
        ids[members] = found + inverse.reshape(-1)
        found += int(inverse.max()) + 1
    return ids


def _shapes(model, parent: np.ndarray, first, lengths: np.ndarray, order=None) -> np.ndarray:
    """Equal ids for equal trees: tree ``t`` is the joints ``order[first[t] : first[t] + lengths[t]]`` (``order`` the
    identity by default), parents first; its key is each joint's parent as a position in the tree (``-1`` for a parent
    outside it: a tree stops at its articulation), type and axes."""
    kinds, dimensions = model.joint_type.numpy(), model.joint_dof_dim.numpy()
    order = np.arange(parent.size) if order is None else order
    position = np.full(parent.size, -1, dtype=np.int64)
    tree = np.full(parent.size, -1, dtype=np.int64)

    def rows(members, length):
        table = order[first[members, None] + np.arange(length)]
        position[table], tree[table] = np.arange(length), members[:, None]
        above = np.maximum(parent[table], 0)
        local = np.where((parent[table] >= 0) & (tree[above] == members[:, None]), position[above], -1)
        return np.concatenate((local, kinds[table], dimensions[table].reshape(members.size, -1)), axis=1)

    return _equal_ids(lengths, rows)


def _by_topology(plan, model, arts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The articulations ``arts`` of a size group, ascending, with equal trees side by side (ascending among them),
    and the bounds of each run of equal trees. A group of one topology keeps its order."""
    first = model.articulation_start.numpy()[arts]
    topology = _shapes(model, model.joint_ancestor.numpy(), first, plan.articulation_joint_end_host[arts] - first)
    return arts[np.argsort(topology, kind="stable")], host_index.offsets(np.bincount(topology), np.int64)


def _size_groups(plan, model):
    """Response-size groups, without modifying physical topology."""
    plan.art_group_idx = None
    plan.group_to_art = {}
    plan.group_to_art_host = {}
    plan.size_topologies = {}
    plan.rest_articulations, plan.rest_topologies = np.zeros(0, np.int32), np.zeros(1, np.int32)
    if not model.articulation_count or not model.joint_count:
        plan.size_groups = []
        plan.n_arts_by_size = {}
        plan.free_body_inertia_sizes = frozenset()
        plan.articulation_response_dof_count = wp.zeros(model.articulation_count, dtype=wp.int32, device=model.device)
        return

    response_dof_count = plan.response_dof_count
    # Each size group holds its articulations by topology, ascending within one, the groups by descending size.
    negated_sizes, group = np.unique(-response_dof_count, return_inverse=True)
    bounds, order = host_index.group_by(group, negated_sizes.size)
    art_group_idx = np.full(response_dof_count.size, -1, dtype=np.int32)
    group_to_art = {}
    for g, size in enumerate(negated_sizes):
        if size < 0:
            arts, plan.size_topologies[int(-size)] = _by_topology(plan, model, order[bounds[g] : bounds[g + 1]])
            art_group_idx[arts] = np.arange(arts.size)
            group_to_art[int(-size)] = arts.astype(np.int32)
    rest = np.flatnonzero(plan.articulation_dof_count_host == 0)
    if rest.size:
        plan.rest_articulations, plan.rest_topologies = _by_topology(plan, model, rest)
    plan.group_to_art_host = group_to_art
    solve_sizes = list(group_to_art)
    plan.size_groups = solve_sizes
    plan.n_arts_by_size = {size: int(group_to_art[size].size) for size in solve_sizes}

    plan.articulation_response_dof_count = wp.array(response_dof_count, dtype=wp.int32, device=model.device)
    plan.art_group_idx = wp.array(art_group_idx, dtype=wp.int32, device=model.device)
    plan.group_to_art = {
        size: wp.array(group_to_art[size], dtype=wp.int32, device=model.device) for size in solve_sizes
    }
    # A one-body free articulation has I_composite == I_body. Select the direct
    # source only for whole size groups so the CRBA kernel remains branch-free.
    plan.free_body_inertia_sizes = frozenset(
        size for size in solve_sizes if np.all(plan.is_free_rigid_host[group_to_art[size]] != 0)
    )


def _world_mapping(plan, model):
    """Articulation-to-world mappings."""
    plan.articulation_joint_end = None
    if not model.articulation_count:
        plan.art_to_world = None
        plan.world_art_start = None
        return

    articulation_world = plan.articulation_world
    plan.art_to_world = wp.array(articulation_world, dtype=wp.int32, device=model.device)
    plan.articulation_joint_end = wp.array(plan.articulation_joint_end_host, dtype=wp.int32, device=model.device)

    world_art_counts = np.bincount(articulation_world, minlength=plan.world_count).astype(np.int32)
    world_art_start = host_index.offsets(world_art_counts, dtype=np.int32)
    plan.world_art_start = wp.array(world_art_start, dtype=wp.int32, device=model.device)


def _body_maps(plan, model):
    """Each body's tree joint, articulation and response support (:func:`.body_mapping.build_body_maps`)."""
    maps = dict.fromkeys(field.name for field in BODY_MAPS)
    plan.body_single_response_dof_host = None
    if model.body_count and model.articulation_count:
        (
            body_to_joint,
            body_to_articulation,
            body_has_response_dofs,
            body_response_dof_mask,
            body_single_response_dof,
        ) = build_body_maps(
            model.body_count,
            model.joint_child.numpy(),
            plan.joint_forest,
            model.joint_qd_start.numpy(),
            model.articulation_start.numpy(),
            plan.articulation_joint_end_host,
            plan.articulation_dof_start_host,
            plan.response_dof_count,
        )
        SolverArrays(None, model.device).upload(
            BODY_MAPS,
            owner=maps,
            body_to_joint=body_to_joint,
            body_to_articulation=body_to_articulation,
            body_has_response_dofs=body_has_response_dofs,
            body_response_dof_mask=body_response_dof_mask,
            body_single_response_dof=body_single_response_dof,
        )
        plan.body_single_response_dof_host = body_single_response_dof
    for name, array in maps.items():
        setattr(plan, name, array)


def _free_bodies(plan, model):
    """Free-rigid execution metadata."""
    # Free rigid bodies per world: they bound a world's contact units per color (sweep_units.capable_worlds).
    plan.free_rigid_per_world = np.zeros(max(int(model.world_count), 1), dtype=np.int64)
    if not model.articulation_count or not model.joint_count:
        plan.has_free_rigid_bodies = False
        plan.has_mixed_contacts = False
        plan.has_non_free_articulations = False
        plan.is_free_rigid = None
        plan.free_rigid_body_indices = None
        plan.free_rigid_body_world = None
        plan.free_rigid_body_count = 0
        return

    is_free_rigid = plan.is_free_rigid_host
    solve_mask = plan.response_dof_count > 0
    free_mask = is_free_rigid != 0
    plan.free_rigid_body_count = len(plan.response_free_rigid_body_indices)
    plan.has_free_rigid_bodies = plan.free_rigid_body_count > 0

    articulation_world = plan.articulation_world
    free_worlds = set(articulation_world[solve_mask & free_mask])
    nonfree_worlds = set(articulation_world[solve_mask & ~free_mask])
    plan.has_mixed_contacts = bool(free_worlds & nonfree_worlds)
    # Solve-mask-aware: zero-DOF (kinematic-only) articulations are excluded
    # from the solve size groups, so they must not count as non-free
    # articulations when selecting the patch routes.
    plan.has_non_free_articulations = bool(np.any(solve_mask & ~free_mask))
    free_world = articulation_world[solve_mask & free_mask]
    free_world = free_world[free_world >= 0]
    plan.free_rigid_per_world = np.bincount(free_world, minlength=plan.free_rigid_per_world.size)
    plan.is_free_rigid = wp.array(is_free_rigid, dtype=wp.int32, device=model.device)
    plan.free_rigid_body_indices = wp.array(plan.response_free_rigid_body_indices, dtype=wp.int32, device=model.device)
    plan.free_rigid_body_world = wp.array(
        articulation_world[solve_mask & free_mask], dtype=wp.int32, device=model.device
    )


def _coordinate_trees(plan, model):
    """The root cut and its coordinate trees: each articulation's joints once the joints above every coordinate are cut
    away, so a root without coordinates is no node and each joint below it roots a tree (a fixed-base star becomes one
    tree a key). Only a common tree couples two coordinates in the mass matrix, so a tree is a vertex of the rows' graph
    (rows are its edges). The mass matrix stops at an articulation, and so does a tree, whatever the ancestry says. A
    prescribed articulation responds with nothing and holds none."""
    forest = plan.joint_forest
    joint_articulation = model.joint_articulation.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    member = joint_articulation >= 0
    dofs = np.zeros(forest.count, dtype=np.int32)
    dofs[member] = np.where(plan.response_dof_count[joint_articulation[member]] > 0, np.diff(joint_qd_start)[member], 0)

    def rooted_above(above, joint):
        return np.where((dofs[above] > 0) & (joint_articulation[above] == joint_articulation[joint]), above, joint)

    # Each joint's topmost ancestor with coordinates, or itself: its tree's root, or a joint above every coordinate,
    # whose children the cut makes roots.
    joints = np.arange(forest.count)
    root = forest.scan_down(joints, rooted_above)
    cut = TreePlan.build(np.where(root == joints, -1, forest.parent), dofs)
    roots = cut.roots[dofs[cut.roots] > 0]
    solved = dofs > 0
    tree_articulation = joint_articulation[roots].astype(np.int32)
    world_tree_start, world_trees = host_index.group_by(plan.articulation_world[tree_articulation], plan.world_count)
    joint_child = model.joint_child.numpy()
    plan.coordinate_forest, plan.tree_root = cut, roots
    joint_tree = _joint_tree(plan)
    body_tree = np.full(model.body_count, -1, dtype=np.int32)
    owned = member & (joint_child >= 0)
    body_tree[joint_child[owned]] = joint_tree[owned]
    arrays = {
        "tree_root": roots.astype(np.int32),
        "tree_shape": _shapes(model, cut.parent, cut.dfs_position[roots], cut.subtree_size[roots], cut.dfs_order),
        "tree_first": joint_qd_start[roots].astype(np.int32),
        "tree_size": np.bincount(joint_tree[solved], dofs[solved], roots.size).astype(np.int32),
        "tree_articulation": tree_articulation,
        "world_tree_start": world_tree_start.astype(np.int32),
        "world_trees": world_trees.astype(np.int32),
        "body_tree": body_tree,
    }
    for name, array in arrays.items():
        array.setflags(write=False)
        setattr(plan, name, array)


def _joint_tree(plan) -> np.ndarray:
    """Each joint's tree, ``-1`` for a joint above every coordinate or outside every tree."""
    cut = plan.coordinate_forest
    tree = np.full(cut.count, -1, dtype=np.int64)
    tree[plan.tree_root] = np.arange(plan.tree_root.size)
    return tree[cut.root_of]


def _world_classes(plan, model):
    """The row laws present in each world, and the world classes: worlds of equal tree shapes and equal laws."""
    laws = np.zeros(plan.world_count, dtype=np.int32)
    tree_world = plan.articulation_world[plan.tree_articulation]

    def present(law, trees):
        np.bitwise_or.at(laws, tree_world[trees[trees >= 0]], int(law))

    if model.shape_contact_pair_count:
        pairs = model.shape_contact_pairs.numpy().reshape(-1, 2)
        bodies = np.sort(model.shape_body.numpy()[pairs], axis=1)
        trees = np.where(bodies >= 0, plan.body_tree[bodies], -1).max(axis=1)
        present(RowLaw.CONTACT, trees)
        # A sphere or an ellipsoid touches a convex shape at one point (the narrow phase skips its multi-contact
        # manifold); a body pair of one such shape pair has no footprint, every other one can.
        kinds = model.shape_type.numpy()[pairs]
        soft = (model.shape_flags.numpy()[pairs] & int(ShapeFlags.HYDROELASTIC)) != 0
        round_ = np.isin(kinds, _ONE_POINT_SHAPES).any(axis=1)
        convex = ~np.isin(kinds, _MANIFOLD_SHAPES).any(axis=1) & ~soft.any(axis=1)
        _, pair_body, shared = np.unique(bodies, axis=0, return_inverse=True, return_counts=True)
        present(RowLaw.SPIN, trees[(shared[pair_body.reshape(-1)] > 1) | ~(round_ & convex)])
    if model.joint_count:
        joint_tree = _joint_tree(plan)
        limited = np.flatnonzero(np.isin(model.joint_type.numpy(), _LIMITED_JOINTS) & (joint_tree >= 0))
        joint, axis = host_index.expand(model.joint_dof_dim.numpy()[limited].sum(axis=1))
        dof = model.joint_qd_start.numpy()[limited[joint]] + axis
        trees = joint_tree[limited[joint]]
        finite = np.isfinite(model.joint_limit_lower.numpy()[dof]) | np.isfinite(model.joint_limit_upper.numpy()[dof])
        present(RowLaw.LIMIT, trees[finite])
        present(RowLaw.VELOCITY_LIMIT, trees[model.joint_velocity_limit.numpy()[dof] > 0.0])
        if model.constraint_mimic_count:
            for joints in (model.constraint_mimic_joint0.numpy(), model.constraint_mimic_joint1.numpy()):
                present(RowLaw.MIMIC, joint_tree[joints[joints >= 0]])
    if plan.cut.tree_primary.size:
        np.bitwise_or.at(laws, plan.articulation_world[plan.cut.tree_primary], int(RowLaw.LINK))

    # A world's key: its trees' shapes, ascending, then its laws.
    counts = np.diff(plan.world_tree_start)
    by_world = plan.tree_shape[plan.world_trees]
    order = np.lexsort((by_world, np.repeat(np.arange(plan.world_count), counts)))
    ordered = by_world[order]

    def rows(members, length):
        table = ordered[plan.world_tree_start[members, None] + np.arange(length)]
        return np.concatenate((table, laws[members, None]), axis=1)

    # Classes by their first world.
    _, first, inverse = np.unique(_equal_ids(counts, rows), return_index=True, return_inverse=True)
    world_class = np.argsort(np.argsort(first))[inverse.reshape(-1)].astype(np.int32)
    bounds, worlds = host_index.group_by(world_class, first.size)
    classes = []
    for c in range(first.size):
        members = worlds[bounds[c] : bounds[c + 1]].astype(np.int32)
        head = members[0]
        shape = ordered[plan.world_tree_start[head] : plan.world_tree_start[head + 1]]
        members.setflags(write=False)
        classes.append(WorldClass(tuple(int(k) for k in shape), RowLaw(int(laws[head])), members))
    laws.setflags(write=False)
    world_class.setflags(write=False)
    plan.world_laws, plan.world_class, plan.world_classes = laws, world_class, tuple(classes)


def _mass_schedules(plan, model):
    """Per topology-homogeneous size group: the coordinate-to-joint map and the star test."""
    plan.crba_dof_joint_offset_by_size = {}
    diagonal_mass_sizes: set[int] = set()
    direct_diagonal_inertia_sizes: set[int] = set()
    plan.diagonal_mass_sizes = frozenset()
    plan.direct_diagonal_inertia_sizes = frozenset()
    if not model.articulation_count or not model.joint_count:
        return

    articulation_start = model.articulation_start.numpy()
    articulation_dof_start = plan.articulation_dof_start_host
    response_dof_count = plan.response_dof_count
    joint_qd_start = model.joint_qd_start.numpy()
    joint_ancestor = model.joint_ancestor.numpy()
    joint_child = model.joint_child.numpy()
    body_single_response_dof = plan.body_single_response_dof_host
    response_body_counts = np.bincount(
        body_single_response_dof[body_single_response_dof >= 0], minlength=model.joint_dof_count
    )

    for size in plan.size_groups:
        arts = np.flatnonzero(response_dof_count == size)
        joint_start = articulation_start[arts]
        joint_count = np.maximum(articulation_start[arts + 1] - joint_start, 0)
        dof_start = articulation_dof_start[arts]
        # Each DOF's joint, as an offset from the articulation's first joint;
        # a later joint wins a DOF that an earlier one also spans.
        art, position = host_index.expand(joint_count)
        joint = joint_start[art] + position
        local_start = np.maximum(0, joint_qd_start[joint] - dof_start[art])
        local_end = np.minimum(size, joint_qd_start[joint + 1] - dof_start[art])
        entry, local_dof = host_index.expand(np.maximum(local_end - local_start, 0))
        offsets = np.full((arts.size, size), -1, dtype=np.int32)
        np.maximum.at(offsets, (art[entry], local_start[entry] + local_dof), position[entry].astype(np.int32))
        # One schedule serves the group only if every articulation has the first one's topology.
        compatible = bool(
            not np.any(offsets < 0) and np.all(offsets == offsets[0]) and np.all(joint_count == joint_count[0])
        )
        if compatible:
            parents = joint_ancestor[joint_start[:, None] + np.arange(joint_count[0])]
            parents = np.where(parents >= joint_start[:, None], parents - joint_start[:, None], -1).astype(np.int32)
            compatible = bool(np.all(parents == parents[0]))

        if compatible:
            reference_offsets = offsets[0]
            # A star: every tree of the group has one coordinate, so each entry of the mass matrix is a coordinate's own.
            if np.all(plan.tree_size[response_dof_count[plan.tree_articulation] == size] == 1):
                diagonal_mass_sizes.add(int(size))
                # Direct when each DOF's child body has that DOF as its single
                # response DOF, and no other body does.
                child = joint_child[joint_start[:, None] + reference_offsets]
                global_dof = dof_start[:, None] + np.arange(size)
                single_dof = np.where(child >= 0, body_single_response_dof[np.maximum(child, 0)], -1)
                owned = single_dof == global_dof
                owned &= response_body_counts[np.where(owned, global_dof, 0)] == 1
                if np.all(owned):
                    direct_diagonal_inertia_sizes.add(int(size))
            plan.crba_dof_joint_offset_by_size[int(size)] = wp.array(
                reference_offsets, dtype=wp.int32, device=model.device
            )

    plan.diagonal_mass_sizes = frozenset(diagonal_mass_sizes)
    plan.direct_diagonal_inertia_sizes = frozenset(direct_diagonal_inertia_sizes)


def _world_size_groups(plan, model):
    """Each size group's responding articulations by world (the fused diagonal accumulation's order)."""
    plan.world_response_group_art_start = {}
    plan.world_response_group_to_art = {}
    if not model.articulation_count or plan.art_to_world is None or plan.is_free_rigid is None:
        return

    device = model.device
    art_to_world_np = plan.art_to_world.numpy().astype(np.int32, copy=False)
    art_size_np = plan.response_dof_count.astype(np.int32, copy=False)
    world_count = max(int(plan.world_count), 0)
    for size in plan.size_groups:
        # The fused diagonal accumulator must retain every solved articulation:
        # each world's, in ascending order, one world after another.
        arts = np.flatnonzero(art_size_np == int(size))
        worlds = art_to_world_np[arts]
        in_range = (worlds >= 0) & (worlds < world_count)
        arts, worlds = arts[in_range], worlds[in_range]
        response_starts, order = host_index.group_by(worlds, world_count)
        response_flat_np = arts[order].astype(np.int32)
        plan.world_response_group_art_start[int(size)] = wp.array(response_starts, dtype=wp.int32, device=device)
        plan.world_response_group_to_art[int(size)] = wp.array(response_flat_np, dtype=wp.int32, device=device)


def _world_response_mapping(plan, model):
    """Compact per-world response offsets and global-DOF indices (the matrix-free formulation's)."""
    articulation_world = plan.articulation_world
    response_dof_count = plan.response_dof_count
    articulation_dof_start = plan.articulation_dof_start_host
    articulation_world_dof_offset = np.full(model.articulation_count, -1, dtype=np.int32)

    # Each world lists its response articulations' DOFs by ascending DOF start, ties by articulation.
    articulations = np.flatnonzero(response_dof_count > 0)
    articulations = articulations[
        np.lexsort((articulation_dof_start[articulations], articulation_world[articulations]))
    ]
    worlds = articulation_world[articulations]
    counts = response_dof_count[articulations].astype(np.int64)
    world_dof_count = np.zeros(plan.world_count, dtype=np.int64)
    np.add.at(world_dof_count, worlds, counts)
    world_offset = host_index.segmented_rank(worlds, counts)
    articulation_world_dof_offset[articulations] = world_offset

    world_dof_count = world_dof_count.astype(np.int32)
    plan.max_world_dofs = int(np.max(world_dof_count)) if len(world_dof_count) else 0
    padded_indices = np.full((plan.world_count, plan.max_world_dofs), -1, dtype=np.int32)
    entry, local_dof = host_index.expand(counts)
    padded_indices[worlds[entry], world_offset[entry] + local_dof] = (
        articulation_dof_start[articulations][entry] + local_dof
    )

    plan.articulation_world_dof_offset = wp.array(articulation_world_dof_offset, dtype=wp.int32, device=model.device)
    plan.world_dof_count = wp.array(world_dof_count, dtype=wp.int32, device=model.device)
    plan.world_dof_indices = wp.array(padded_indices, dtype=wp.int32, device=model.device)


def _class_shape(plan):
    sizes = tuple(int(size) for size in plan.size_groups)
    arts = np.full((plan.world_count, len(sizes)), -1, dtype=np.int32)
    groups, offsets, world_offset = arts.copy(), [], plan.articulation_world_dof_offset.numpy()
    for block, size in enumerate(sizes):
        members = plan.group_to_art_host[size]
        worlds, offset = plan.articulation_world[members], world_offset[members]
        if np.unique(worlds).size != worlds.size:
            arts = groups = None
        if arts is not None:
            arts[worlds, block], groups[worlds, block] = members, np.arange(members.size)
        offsets.append(int(offset[0]) if np.all(offset == offset[0]) else None)
    free = tuple(size in plan.free_body_inertia_sizes for size in sizes)
    plan.shape = ClassShape(sizes, free, arts, groups, tuple(offsets), int(plan.max_world_dofs))


def row_support(
    forest: ForestPlan, model: Model, factor, articulated: np.ndarray, several: bool, mimic_dofs=None
) -> tuple[np.ndarray, int]:
    """Each body's support and the widest row's on the native sparse factors of one articulated size.

    A body's support is its joint's ancestor mask in factor coordinates. The widest row covers a contact between any
    two bodies' supports, a mimic pair, two articulations of one world, and at most the widest world.

    Args:
        forest: The forest plan.
        model: The model.
        factor: The size's fill-free factor pattern (:meth:`.tree_plan.TreePlan.factor`).
        articulated: The articulated (not free-body) responding articulations.
        several: Whether a world holds several of them.
        mimic_dofs: Each mimic's two coordinates, local to its articulation, or ``None`` without mimic rows.

    Returns:
        ``(body_masks, width)``.
    """
    articulation_start = model.articulation_start.numpy()
    joint_child = model.joint_child.numpy()
    joint_masks = factor.joint_ancestor_mask
    body_masks = np.zeros(model.body_count, dtype=np.uint64)
    first_joint = articulation_start[articulated]
    art, local_joint = host_index.expand(forest.articulation_joint_end_host[articulated] - first_joint)
    children = joint_child[first_joint[art] + local_joint]
    valid = children >= 0
    body_masks[children[valid]] = joint_masks[local_joint[valid]]
    shape_body = model.shape_body.numpy()
    contact_masks = np.unique(body_masks[shape_body[shape_body >= 0]])
    max_support = max((int(mask).bit_count() for mask in joint_masks), default=0)
    for first in contact_masks:
        for second in contact_masks:
            max_support = max(max_support, int(first | second).bit_count())
    if mimic_dofs is not None:
        # Coefficients and enable bits stay live, so retain both endpoint
        # supports even when a coupling is initially disabled or zero.
        first_dofs, second_dofs = mimic_dofs
        masks = factor.ancestor_mask[first_dofs] | factor.ancestor_mask[second_dofs]
        max_support = max(max_support, max((int(mask).bit_count() for mask in masks), default=0))
    if several:
        # Different articulations have disjoint coordinate blocks, even
        # when they share the same topology and local ancestor indices.
        max_support = max(max_support, 2 * max(int(mask).bit_count() for mask in joint_masks))
    # Response groups need not coexist in every world. No row can contain
    # more distinct coordinates than the largest world's response vector.
    return body_masks, min(max_support, forest.max_world_dofs)

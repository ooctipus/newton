# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The forest plan: the model's articulation forest as every other primitive reads it, built once per model.

:meth:`ForestPlan.build` derives, from the model's topology alone:

- the joint forest (:func:`joint_forest`) and each articulation's tree joints, coordinates and world, with the
  loop-closing joints set apart (they become connect rows);
- one root rule (a FREE or DISTANCE joint without a parent roots a free tree), from which the free rigid bodies, the
  free-rooted articulations and the free-root joints follow;
- the prescribed free bodies the matrix-free formulation elides, and so each articulation's response coordinates;
- the size groups (articulations by response coordinate count) and the world mapping;
- the body maps (each body's tree joint, articulation and response support);
- per size group, the mass matrix's coordinate-to-joint map and whether its fill-free factor pattern is diagonal
  (a star);
- with the matrix-free formulation, the compact per-world response mapping.

Host arrays end in ``_host`` where a device array of the same name exists. The plan is frozen: notifications that
change the model's topology need a new solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp

from ...sim import Model
from ...sim.enums import JointType
from . import host_index
from .body_mapping import build_body_maps
from .field_group import SolverArrays
from .solver_fields import BODY_MAPS
from .tree_plan import TreePlan

__all__ = ["ForestPlan", "joint_forest", "row_support"]


def joint_forest(model: Model) -> TreePlan:
    """The forest of the model's articulation joints by their ancestry (``Model.joint_ancestor``); a joint
    outside every articulation, a loop closure, stands alone."""
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
    """Each articulation's end of its tree joints (loop-closing joints follow it)."""
    loop_joint_articulation: np.ndarray
    """Each joint's articulation where it closes a loop, else ``-1``."""
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
    has_loop_joints: bool

    # Free bodies.
    free_rigid_body_count: int
    has_free_rigid_bodies: bool
    has_mixed_contacts: bool
    """A world holds both a free rigid body and another responding articulation."""
    has_non_free_articulations: bool
    free_rigid_per_world: np.ndarray
    """Responding free rigid bodies per world (at least one entry)."""

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
    """Size groups whose fill-free factor pattern is diagonal (stars)."""
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
    def build(cls, model: Model, joint_forest: TreePlan, kinematic_dof_mask: np.ndarray) -> ForestPlan:
        """Build the plan from the model's topology.

        Args:
            model: The model.
            joint_forest: The model's joint forest (:func:`joint_forest`).
            kinematic_dof_mask: Each coordinate the body flags make kinematic, at construction.
        """
        plan = SimpleNamespace(joint_forest=joint_forest)
        free_root = None
        if model.joint_count:
            # The one root rule: a FREE or DISTANCE joint without a parent roots a free tree.
            joint_type = model.joint_type.numpy()
            free_root = ((joint_type == JointType.FREE) | (joint_type == JointType.DISTANCE)) & (
                model.joint_parent.numpy() < 0
            )
        _articulations(plan, model, kinematic_dof_mask, free_root)
        plan.prescribed_articulation = wp.array(plan.prescribed_articulation_host, dtype=wp.int32, device=model.device)
        plan.has_prescribed_response = bool(np.any(plan.prescribed_articulation_host != 0))
        _coordinates(plan, model)
        _roots(plan, model, free_root)
        _size_groups(plan, model)
        _world_mapping(plan, model)
        _body_maps(plan, model)
        _free_bodies(plan, model)
        _mass_schedules(plan, model)
        _world_size_groups(plan, model)
        # Loop-closing joints are excluded from the tree and enforced as connect rows.
        plan.has_loop_joints = bool(np.any(plan.loop_joint_articulation >= 0))
        _world_response_mapping(plan, model)
        return cls(**vars(plan))


def _articulations(plan, model, kinematic_dof_mask, free_root):
    """Each articulation's tree joints, coordinates, world, free body and prescribed response."""
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
    loop_joint_articulation = np.full(model.joint_count, -1, dtype=np.int32)
    if articulation_count and model.joint_count:
        articulation_start = model.articulation_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_type = model.joint_type.numpy()
        joint_child = model.joint_child.numpy()

        joint_articulation_arr = getattr(model, "joint_articulation", None)
        joint_articulation = joint_articulation_arr.numpy() if joint_articulation_arr is not None else None

        first_joint = articulation_start[:articulation_count]
        last_joint = articulation_start[1 : articulation_count + 1]

        # Loop-closing joints (how the MJCF/USD importers close `connect`/`weld`
        # equalities) are not part of the kinematic tree. Excluding them here
        # keeps their DOFs out of the mass matrix (garbage motion subspaces made
        # H singular -> instant NaN) and out of the FK walk (they would overwrite
        # their child's tree pose). They are consumed as CONNECT rows instead.
        if joint_articulation is not None:
            # An unowned joint is a closure only when its child body already
            # belongs to an articulation through a different, tree-owned joint.
            # Standalone world-root joints are also unowned, but their child has
            # no articulation owner and must remain outside the solver plan.
            # A body's last owned joint names its articulation.
            joint_index = np.arange(model.joint_count)
            owned = (joint_articulation >= 0) & (joint_child >= 0)
            body_joint = np.full(model.body_count, -1, dtype=np.int64)
            np.maximum.at(body_joint, joint_child[owned], joint_index[owned])
            body_articulation = np.full(model.body_count, -1, dtype=np.int32)
            has_owner = body_joint >= 0
            body_articulation[has_owner] = joint_articulation[body_joint[has_owner]]
            closing = (joint_articulation < 0) & (joint_child >= 0)
            loop_joint_articulation[closing] = body_articulation[joint_child[closing]]

            # Model.joint_articulation identifies the articulation's own contiguous
            # tree joints; the body-ownership pass above separately distinguishes
            # closures from other unowned joints. The prefix range derived from
            # articulation_start may contain foreign entries after the tree — e.g.
            # multi-source add_builder defers loop joints past a later articulation's
            # free joint — so never adopt range members by position.
            range_art, position = host_index.expand(np.maximum(last_joint - first_joint, 0))
            member = joint_articulation[first_joint[range_art] + position] == range_art
            member_count = np.bincount(range_art[member], minlength=articulation_count)
            # Members must open their range: as many members precede each one as joints do.
            members_before = host_index.segmented_rank(range_art, member)
            stray = np.flatnonzero(member & (members_before != position))
            if stray.size:
                art = int(range_art[stray[0]])
                j = int(first_joint[art] + position[stray[0]])
                tree_end = int(first_joint[art] + members_before[stray[0]])
                raise ValueError(
                    f"SolverFeatherPGS: articulation {art}'s joints are not "
                    f"contiguous (joint {j} follows a foreign joint at "
                    f"{tree_end}). The ModelBuilder guarantees contiguity; "
                    "this model is malformed."
                )
            articulation_joint_end[:] = first_joint + member_count
        else:
            for art in range(articulation_count):
                # Legacy models without joint_articulation: fall back to the
                # second-inbound-joint heuristic (loop joints must be trailing).
                seen_children: set[int] = set()
                tree_end = int(first_joint[art])
                for j in range(int(first_joint[art]), int(last_joint[art])):
                    child = int(joint_child[j])
                    if child in seen_children:
                        break  # first loop joint; the rest of the range must be loops too
                    seen_children.add(child)
                    tree_end = j + 1
                for j in range(tree_end, int(last_joint[art])):
                    if int(joint_child[j]) not in seen_children:
                        raise ValueError(
                            "SolverFeatherPGS: articulation joint range mixes tree joints "
                            f"after loop joints (articulation {art}, joint {j}). "
                            "Loop-closing joints must be trailing."
                        )
                    loop_joint_articulation[j] = art
                articulation_joint_end[art] = tree_end

        articulation_dof_start[:] = joint_qd_start[first_joint]
        articulation_dof_count[:] = joint_qd_start[articulation_joint_end] - articulation_dof_start
        single = np.flatnonzero(articulation_joint_end - first_joint == 1)
        root = first_joint[single]
        # A free rigid body: one tree joint, a FREE joint that roots a free tree.
        free = single[free_root[root] & (joint_type[root] == int(JointType.FREE))]
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
        "loop_joint_articulation": loop_joint_articulation,
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


def _by_topology(plan, model, arts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The articulations ``arts`` of a size group, ascending, with equal trees side by side (ascending among them),
    and the bounds of each run of equal trees. A group of one topology keeps its order."""
    first = model.articulation_start.numpy()[arts]
    joints = plan.articulation_joint_end_host[arts] - first
    ancestor, kinds = model.joint_ancestor.numpy(), model.joint_type.numpy()
    dimensions = model.joint_dof_dim.numpy()
    topology = np.zeros(arts.size, dtype=np.int64)
    found = 0
    for count in np.unique(joints):
        members = np.flatnonzero(joints == count)
        table = first[members, None] + np.arange(count)
        parents = np.where(ancestor[table] >= 0, ancestor[table] - first[members, None], -1)
        rows = np.concatenate((parents, kinds[table], dimensions[table].reshape(members.size, -1)), axis=1)
        _, inverse = np.unique(rows, axis=0, return_inverse=True)
        topology[members] = found + inverse.reshape(-1)
        found += int(inverse.max()) + 1
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
    # Per-articulation end of the kinematic-tree joint prefix: trailing loop-closing
    # joints (imported `connect`/`weld` equalities) are excluded from FK/ID iteration.
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
            # A star: the fill-free factor pattern of the group's coordinates on their joints is diagonal (no
            # coordinate has another on its joint or an ancestor's), so each entry of the mass matrix is a
            # coordinate's own.
            joint_dofs = np.bincount(reference_offsets, minlength=parents[0].size)
            if TreePlan.build(parents[0], joint_dofs).nonzero_count == int(size):
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


def row_support(
    forest: ForestPlan, model: Model, factor, articulated: np.ndarray, articulated_counts: np.ndarray, mimic_dofs=None
) -> tuple[np.ndarray, int]:
    """Each body's support and the widest row's on the native sparse factors of one articulated size.

    A body's support is its joint's ancestor mask in factor coordinates. The widest row covers a contact between any
    two bodies' supports, a mimic pair, two articulations of one world, and at most the widest world.

    Args:
        forest: The forest plan.
        model: The model.
        factor: The size's fill-free factor pattern (:meth:`.tree_plan.TreePlan.factor`).
        articulated: The articulated (not free-body) responding articulations.
        articulated_counts: Articulated responding articulations per world.
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
    if np.any(articulated_counts > 1):
        # Different articulations have disjoint coordinate blocks, even
        # when they share the same topology and local ancestor indices.
        max_support = max(max_support, 2 * max(int(mask).bit_count() for mask in joint_masks))
    # Response groups need not coexist in every world. No row can contain
    # more distinct coordinates than the largest world's response vector.
    return body_masks, min(max_support, forest.max_world_dofs)

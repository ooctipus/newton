# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The body cut: the solver's own model, a pure forest.

The forest plan cuts a body, never a joint. A joint that would give a body a second parent stays a tree joint, of its
parent's tree, and its child becomes a shadow of that body: a leaf past the model's bodies with an equal share of the
body's mass, tied to the primary piece by one 6-row link (:attr:`BodyCut.primary`, :attr:`BodyCut.shadow`). The joint
keeps its coordinates, drive, limits, armature and damping, whatever its type. An articulation of several roots
becomes one tree per root.

:attr:`BodyCut.model` is that forest as a :class:`~newton.Model`, and it is the only model the solver reads:

- every primitive sees joints, bodies and coordinates of pure trees, so no primitive knows what a loop is;
- a model that is already a pure forest is its own forest model, the same object, and nothing here runs;
- otherwise its arrays are the model's, gathered: joints in tree order (a tree's joints must be contiguous, which the
  model's order does not give), coordinates with them, and the shadows past the model's bodies, whose indices stay.

:meth:`BodyCut.enter` and :meth:`BodyCut.leave` carry a step across the boundary. The solver steps mirrors of the
caller's states: the forest's layout, one mirror per state, so that a state keeps its identity across steps. A loop
joint's coordinates are real state: they integrate like any joint's and are published to ``state.joint_q``.

**Live arrays.** A step reads the model's gains, limits, frames and inertias live: a write reaches the rows with no
notification. So the arrays a step reads and the layout moves (:data:`_LIVE`) are gathered at every
:meth:`BodyCut.enter`, and at every notification, whose handlers read them too, by one gather that needs no notice of
what changed. The rest of the model's arrays a step reads keep their layout
and are the same arrays: gravity, the shapes, the mimic coefficients, and the speed limits, which a step reads only for
contact and free bodies, primaries all. The model's topology is derived once, and a notification that finds it changed
raises, as the held tree's proof does on a pure forest.

**Seating.** A caller writes a tree's coordinates and not the loop joints' (a reset, a teleport), and a model's
``joint_q`` never holds them to begin with. So wherever the tree factor is about to re-derive a tree's kinematics
(``fk_id_cache_valid == 0``), :meth:`BodyCut.enter` first puts each of its shadows on its primary piece, and inverse
kinematics reads the loop joint's coordinates from that pose. Steady stepping seats nothing, so a coordinate stays
continuous past a full turn. The test is the device's, so it holds in a captured graph.

**Disabled links.** A disabled link leaves its shadow massless and kinematic: its share returns to the primary piece,
which is then the whole body again, and the shadow's coordinates, under the kinematic armature, cost the tree nothing.
"""

from __future__ import annotations

import copy
import dataclasses
import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from ...core import memory
from ...core.execution import Shape, Work, launch, shape
from ...sim import BodyFlags, Control, Model, State
from ...sim.articulation import eval_fk, eval_ik
from . import host_index
from .tree_plan import TreePlan

__all__ = ["BodyCut"]

_F = Model.AttributeFrequency

_TOPOLOGY = (
    "articulation_end",
    "articulation_start",
    "joint_ancestor",
    "joint_articulation",
    "joint_child",
    "joint_dof_dim",
    "joint_parent",
    "joint_q_start",
    "joint_qd_start",
    "joint_type",
)
"""The model's topology: derived once (:data:`_DERIVED` of it) and held to its value (:meth:`BodyCut.refresh`)."""

_DERIVED = frozenset(_TOPOLOGY) - {"joint_dof_dim", "joint_type"}
"""The arrays whose values are indices of the layout the cut changes: derived, never gathered."""

_BODY_DEFAULTS = frozenset({"body_q", "body_qd", "body_world", "body_label"})
"""The per-body arrays gathered once, beside :data:`_LIVE`'s. A step reads no other per-body array for a shadow."""

_LIVE_DOFS = (
    "joint_axis",
    "joint_armature",
    "joint_damping",
    "joint_stiffness",
    "joint_effort_limit",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_target_ke",
    "joint_target_kd",
    "joint_velocity_limit",
)
"""The per-DOF arrays a step reads live (``joint_armature``: a notification's handlers read it)."""

_LIVE_COORDS = ("joint_rest_q",)
"""The per-coordinate arrays a step reads live: the passive springs' rest configuration."""

_LIVE = frozenset(
    {"joint_X_p", "joint_X_c", *_LIVE_DOFS, *_LIVE_COORDS, "body_com", "body_mass", "body_inertia", "body_flags"},
)
"""Every array the gathers write: the ones a step reads live and the layout moves."""

_DYNAMIC = wp.constant(int(BodyFlags.DYNAMIC))
_KINEMATIC = wp.constant(int(BodyFlags.KINEMATIC))


@wp.struct
class _Items:
    """A map's data, one launch operand: ``count`` items, item ``i`` reading ``outof[i]`` and writing ``into[i]``
    where ``active[i]``."""

    count: int
    active: wp.array[wp.bool]
    outof: wp.array[int]
    into: wp.array[int]


@wp.kernel(enable_backward=False)
def _move(items: _Items, source: wp.array(dtype=Any), target: wp.array(dtype=Any)):
    """The cut's one map (:class:`_Map`): ``target[into[i]] = source[outof[i]]``. A gather reads by its order into the
    identity, a scatter the identity into its order."""
    i = wp.tid()
    if i < items.count:
        if items.active[i]:
            target[items.into[i]] = source[items.outof[i]]


@wp.kernel(enable_backward=False)
def _gather_frames(
    items: _Items,
    source: wp.array[wp.transform],
    anchor: wp.array[int],
    body_q: wp.array[wp.transform],
    target: wp.array[wp.transform],
):
    """The joints' parent frames, by the joints' map. A loop joint on a body no tree holds stands on the world, where
    that body stands."""
    i = wp.tid()
    if i < items.count:
        j = items.outof[i]
        target[i] = source[j]
        if anchor[i] >= 0:
            target[i] = wp.transform_multiply(body_q[anchor[i]], source[j])


@wp.kernel(enable_backward=False)
def _gather_bodies(
    items: _Items,
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    flags: wp.array[int],
    share: wp.array[float],
    primaries: int,
    forest_mass: wp.array[float],
    forest_inertia: wp.array[wp.mat33],
    forest_flags: wp.array[int],
):
    """Each piece's share of its body, by the bodies' map. A shadow is dynamic whatever its body is, as a joint's child
    must be to move, and kinematic where its link is disabled (no share)."""
    i = wp.tid()
    if i < items.count:
        j = items.outof[i]
        forest_mass[i] = mass[j] * share[i]
        forest_inertia[i] = inertia[j] * share[i]
        if i < primaries:
            forest_flags[i] = flags[j]
        elif share[i] > 0.0:
            forest_flags[i] = _DYNAMIC
        else:
            forest_flags[i] = _KINEMATIC


@wp.kernel(enable_backward=False)
def _seat(
    count: int,
    tree_start: wp.array[int],
    trees: wp.array[int],
    link_start: wp.array[int],
    links: wp.array[int],
    valid: wp.array[int],
    seated_trees: wp.array[wp.bool],
    seated_links: wp.array[wp.bool],
):
    """Seat each of ``count`` linked components whose trees' kinematics the step re-derives: every tree of it
    re-derives them, so each clears its cache bit, and each of its links seats its shadow."""
    component = wp.tid()
    if component < count:
        cleared = int(0)
        for k in range(tree_start[component], tree_start[component + 1]):
            if valid[trees[k]] == 0:
                cleared = 1
        seat = cleared != 0
        for k in range(tree_start[component], tree_start[component + 1]):
            seated_trees[trees[k]] = seat
            if seat:
                valid[trees[k]] = 0
        for k in range(link_start[component], link_start[component + 1]):
            seated_links[links[k]] = seat


@dataclasses.dataclass(frozen=True)
class _Map:
    """One map of the cut: its items (:class:`_Items`, one operand, so a launch packs what it did before the map was
    data) and the launch its work declares."""

    items: _Items
    shape: Shape

    def move(self, source: wp.array, target: wp.array) -> None:
        """Move ``source`` into ``target`` by the map (:func:`_move`)."""
        launch(_move, self.shape, inputs=[self.items, source], outputs=[target], device=target.device, tiled=False)


def _components(count: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Each of ``count`` nodes' connected component under the edges ``first[i] - second[i]``, as its lowest node."""
    labels = np.arange(count)
    while True:
        lowest = np.minimum(labels[first], labels[second])
        before = labels.copy()
        np.minimum.at(labels, first, lowest)
        np.minimum.at(labels, second, lowest)
        labels = labels[labels]
        if np.array_equal(labels, before):
            return labels


def _topology(model: Model) -> SimpleNamespace | None:
    """The forest of ``model``'s trees and loop joints, or ``None`` when ``model`` already is one.

    A loop joint is a joint outside every tree whose child a tree holds (any other such joint stands outside the
    solver, as its body does, and stays after every tree). It joins its parent's tree. On the world, or on a body no
    tree holds, it roots a tree of its own beside its child's. Trees keep the model's order, a tree's joints theirs,
    with its loop joints after them, so a tree stays parent-first.

    Returns:
        ``loops`` (the model's loop joints, ascending, one shadow each, past the bodies in that order), ``on_tree``
        (each one's parent is a tree's body), ``order`` (the model's joint at each forest joint), ``position`` (its
        inverse), ``ancestor``, ``child``, ``tree`` (each forest joint's, past the last for a joint outside every
        tree) and ``trees`` (their count), ``owner`` (each tree's articulation of the model).
    """
    articulation = model.joint_articulation.numpy()
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    ancestor = model.joint_ancestor.numpy()
    joints = np.arange(model.joint_count)
    owned = (articulation >= 0) & (child >= 0)
    body_joint = np.full(model.body_count, -1, dtype=np.int64)
    body_joint[child[owned]] = joints[owned]
    loops = np.flatnonzero(~owned & (child >= 0))
    loops = loops[body_joint[child[loops]] >= 0]
    if not loops.size and not np.any(np.bincount(articulation[owned & (ancestor < 0)], minlength=1) > 1):
        return None

    root = TreePlan.build(np.where(owned, ancestor, -1)).root_of
    has_parent = parent[loops] >= 0
    on_tree = np.zeros(loops.size, dtype=bool)
    on_tree[has_parent] = body_joint[parent[loops[has_parent]]] >= 0
    above = np.where(owned, ancestor, -1)
    above[loops[on_tree]] = body_joint[parent[loops[on_tree]]]
    tree = np.where(owned, root, -1)
    tree[loops] = np.where(on_tree, root[above[loops]], loops)
    beside = joints.copy()
    beside[loops] = root[body_joint[child[loops]]]
    heads = np.unique(tree[tree >= 0])
    heads = heads[np.lexsort((heads, beside[heads]))]
    rank = np.zeros(model.joint_count, dtype=np.int64)
    rank[heads] = np.arange(heads.size)
    tree = np.where(tree >= 0, rank[np.maximum(tree, 0)], heads.size)
    order = np.lexsort((joints, ~owned, tree))
    position = np.empty(order.size, dtype=np.int64)
    position[order] = joints
    forest_child = child.copy()
    forest_child[loops] = model.body_count + np.arange(loops.size)
    forest_above = above[order]
    return SimpleNamespace(
        loops=loops,
        on_tree=on_tree,
        order=order,
        position=position,
        ancestor=np.where(forest_above >= 0, position[np.maximum(forest_above, 0)], -1),
        child=forest_child[order],
        tree=tree[order],
        trees=int(heads.size),
        owner=articulation[beside[heads]],
    )


def _coordinates(starts: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The coordinates of the joints ``order``, in that order, and their joints' starts among them."""
    widths = np.diff(starts)[order]
    joint, rank = host_index.expand(widths)
    return (starts[order][joint] + rank).astype(np.int32), host_index.offsets(widths, np.int32)


def _gathered(value, order: np.ndarray):
    """``value``'s entries ``order``, as the type it is."""
    if isinstance(value, wp.array):
        return wp.array(value.numpy()[order], dtype=value.dtype, device=value.device)
    return [value[int(i)] for i in order]


class BodyCut:
    """The solver's own model and the boundary between it and the caller's (see the module docstring)."""

    def __init__(self, model: Model):
        """The cut of a pure forest, which is its own model."""
        self.source = model
        """The caller's model."""
        self.model = model
        """The forest the solver reads."""
        self.joints = np.zeros(0, dtype=np.int64)
        """The model's loop joints, one per link."""
        self.primary = np.zeros(0, dtype=np.int32)
        """Each link's primary piece, the model's body."""
        self.shadow = np.zeros(0, dtype=np.int32)
        """Each link's shadow, a body of :attr:`model` past the source's."""
        self.enabled = np.zeros(0, dtype=np.int32)
        """Each link's enable bit."""
        self.tree_primary = np.zeros(0, dtype=np.int32)
        """Each link's primary piece's tree, an articulation of :attr:`model`."""
        self.tree_shadow = np.zeros(0, dtype=np.int32)
        """Each link's shadow's tree."""
        self.component = None
        """Each tree's linked component, as its first tree, or ``None`` without links."""

    @classmethod
    def of(cls, model: Model) -> BodyCut:
        """The cut of ``model`` (the module docstring)."""
        cut = cls(model)
        topology = _topology(model) if model.joint_count else None
        if topology is not None:
            cut._cut(topology)
        return cut

    @property
    def pure(self) -> bool:
        """Whether the caller's model is itself the forest: no mirrors, nothing to carry."""
        return self.model is self.source

    @property
    def linked(self) -> np.ndarray:
        """Each tree a link joins, as a mask over the forest's articulations."""
        linked = np.zeros(self.model.articulation_count, dtype=bool)
        linked[self.tree_primary] = True
        linked[self.tree_shadow] = True
        return linked

    def _cut(self, topology: SimpleNamespace) -> None:
        source = self.source
        device = source.device
        loops, order, position = topology.loops, topology.order, topology.position
        self.joints = loops
        self.primary = source.joint_child.numpy()[loops].astype(np.int32)
        self.shadow = (source.body_count + np.arange(loops.size)).astype(np.int32)
        self.enabled = (source.joint_enabled.numpy()[loops] != 0).astype(np.int32)
        self._position = position
        self._anchored = loops[~topology.on_tree & (source.joint_parent.numpy()[loops] >= 0)]

        coords, q_start = _coordinates(source.joint_q_start.numpy(), order)
        dofs, qd_start = _coordinates(source.joint_qd_start.numpy(), order)
        bodies = np.concatenate((np.arange(source.body_count), self.primary))
        self._orders = {
            _F.BODY: bodies,
            _F.JOINT: order,
            _F.JOINT_DOF: dofs,
            _F.JOINT_COORD: coords,
            _F.ARTICULATION: topology.owner,
        }

        forest = copy.copy(source)
        forest.body_count = bodies.size
        forest.articulation_count = topology.trees
        # No primitive reads these, and a stale share must not be read: a reader fails loudly.
        forest.body_inv_mass = forest.body_inv_inertia = None
        for name, spec in source.attribute_specs.items():
            value = getattr(source, name, None)
            if spec.frequency not in self._orders or name in _DERIVED:
                continue
            if spec.frequency == _F.BODY and name not in _LIVE and name not in _BODY_DEFAULTS:
                continue
            if name.endswith("_world_start"):
                # Per-world index ranges do not hold over shadows past every world's bodies; no primitive reads them.
                setattr(forest, name, None)
            elif isinstance(value, wp.array | list) and len(value) == source._attribute_frequency_count(spec.frequency):
                setattr(forest, name, _gathered(value, self._orders[spec.frequency]))

        def indices(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        in_tree = topology.tree < topology.trees
        bounds = host_index.offsets(np.bincount(topology.tree[in_tree], minlength=topology.trees))
        parent = source.joint_parent.numpy().copy()
        parent[self._anchored] = -1
        forest.joint_ancestor = indices(topology.ancestor)
        forest.joint_child = indices(topology.child)
        forest.joint_parent = indices(parent[order])
        forest.joint_articulation = indices(np.where(in_tree, topology.tree, -1))
        forest.joint_q_start, forest.joint_qd_start = indices(q_start), indices(qd_start)
        forest.articulation_start = indices(np.append(bounds[:-1], source.joint_count))
        forest.articulation_end = indices(bounds[1:])
        forest.max_joints_per_articulation = int(np.diff(bounds).max())
        forest.max_dofs_per_articulation = int(
            np.bincount(np.repeat(topology.tree, np.diff(qd_start)), minlength=topology.trees)[: topology.trees].max()
        )
        if source.constraint_mimic_count:
            forest.constraint_mimic_joint0 = indices(position[source.constraint_mimic_joint0.numpy()])
            forest.constraint_mimic_joint1 = indices(position[source.constraint_mimic_joint1.numpy()])
        self.model = forest
        self._topology = {name: getattr(source, name).numpy().copy() for name in _TOPOLOGY}

        # The links' trees and components.
        body_tree = np.full(bodies.size, -1, dtype=np.int64)
        body_tree[topology.child[in_tree]] = topology.tree[in_tree]
        self.tree_primary = body_tree[self.primary].astype(np.int32)
        self.tree_shadow = body_tree[self.shadow].astype(np.int32)
        self.component = _components(topology.trees, self.tree_primary, self.tree_shadow).astype(np.int32)

        # The maps: the gathers from the caller's layout, the scatters that publish the step (a joint outside every
        # tree stands after the trees, and the step leaves its coordinates alone) and the seat's moves.
        coords, dofs = indices(coords), indices(dofs)
        items = max(order.size, bodies.size, dofs.shape[0], coords.shape[0], loops.size)
        identity, every = indices(np.arange(items)), wp.array(np.ones(items, dtype=bool), dtype=wp.bool, device=device)
        self._seated_trees = wp.zeros(topology.trees, dtype=wp.bool, device=device)
        self._seated_links = wp.zeros(loops.size, dtype=wp.bool, device=device)

        def mapped(count, outof, into, active=every):
            items = _Items()
            items.count, items.active, items.outof, items.into = int(count), active, outof, into
            return _Map(items, shape(Work(int(count)), memory.device(device)))

        self._joint_map = mapped(order.size, indices(order), identity)
        self._body_map = mapped(bodies.size, indices(bodies), identity)
        self._dof_map = mapped(dofs.shape[0], dofs, identity)
        self._coord_map = mapped(coords.shape[0], coords, identity)
        self._target_map = self._coord_map if source.use_coord_layout_targets else self._dof_map
        self._publish_q = mapped(q_start[bounds[-1]], identity, coords)
        self._publish_qd = mapped(qd_start[bounds[-1]], identity, dofs)
        self._seat_map = mapped(loops.size, indices(self.primary), indices(self.shadow), self._seated_links)
        anchor = np.full(order.size, -1, dtype=np.int32)
        anchor[position[self._anchored]] = source.joint_parent.numpy()[self._anchored]
        self._anchor = indices(anchor)
        self._share = wp.zeros(bodies.size, dtype=wp.float32, device=device)

        # The seat: each linked component's trees and links.
        labels, of_tree = np.unique(self.component, return_inverse=True)
        tree_start, trees = host_index.group_by(of_tree, labels.size)
        link_start, links = host_index.group_by(of_tree[self.tree_shadow], labels.size)
        self._components = (labels.size, *(indices(values) for values in (tree_start, trees, link_start, links)))
        self._seat_shape = shape(Work(labels.size), memory.device(device))

        # The mirrors of the caller's states and controls, and the seat's model: a loop joint is a one-joint
        # articulation to inverse kinematics.
        self._states: weakref.WeakKeyDictionary[State, State] = weakref.WeakKeyDictionary()
        self._controls: weakref.WeakKeyDictionary[Control, Control] = weakref.WeakKeyDictionary()
        self._seats = copy.copy(forest)
        self._seats.articulation_start = indices(position[loops])
        self._seats.articulation_end = indices(position[loops] + 1)
        self._seats.articulation_count = int(loops.size)
        self._seats.max_joints_per_articulation = 1
        self._shares()
        self._gather_model()

    # --- the model's properties ---------------------------------------------------------------------------------
    def _shares(self) -> None:
        """Each piece's share of its body: equal among the body's primary and its enabled links' shadows, none for a
        disabled link's shadow."""
        pieces = 1 + np.bincount(self.primary[self.enabled != 0], minlength=self.source.body_count)
        share = 1.0 / pieces[self._orders[_F.BODY]].astype(np.float32)
        share[self.shadow] *= self.enabled
        self._share.assign(share)

    def _gather_model(self) -> None:
        """The arrays a step reads and the layout moves, from the caller's model as it stands now."""
        source, forest, device = self.source, self.model, self.model.device
        for map_, names in (
            (self._joint_map, ("joint_X_c",)),
            (self._dof_map, _LIVE_DOFS),
            (self._coord_map, _LIVE_COORDS),
            (self._body_map, ("body_com",)),
        ):
            for name in names:
                map_.move(getattr(source, name), getattr(forest, name))
        launch(
            _gather_frames,
            self._joint_map.shape,
            inputs=[self._joint_map.items, source.joint_X_p, self._anchor, source.body_q],
            outputs=[forest.joint_X_p],
            device=device,
            tiled=False,
        )
        launch(
            _gather_bodies,
            self._body_map.shape,
            inputs=[
                self._body_map.items,
                source.body_mass,
                source.body_inertia,
                source.body_flags,
                self._share,
                source.body_count,
            ],
            outputs=[forest.body_mass, forest.body_inertia, forest.body_flags],
            device=device,
            tiled=False,
        )

    def refresh(self) -> None:
        """Take a model notification, whose handlers read the forest next: re-gather it, whatever changed.

        Raises:
            RuntimeError: The model's topology changed, which no notification can carry.
        """
        if self.pure:
            return
        for name, value in self._topology.items():
            if not np.array_equal(getattr(self.source, name).numpy(), value):
                raise RuntimeError(f"The model's {name} changed under the body cut; reconstruct the solver.")
        self._gather_model()

    # --- the links, by the model's joints -----------------------------------------------------------------------
    def link(self, joint: int) -> int:
        """The link of the model's loop joint ``joint``.

        Raises:
            ValueError: ``joint`` is not a loop joint of the model.
        """
        found = np.flatnonzero(self.joints == int(joint))
        if not found.size:
            raise ValueError(f"SolverFeatherPGS: joint {joint} is not a loop joint of this model.")
        return int(found[0])

    def set_enabled(self, joint: int, enabled: bool) -> None:
        """Enable or disable the link of a loop joint; the caller then notifies the body properties."""
        self.enabled[self.link(joint)] = 1 if enabled else 0
        self._shares()

    def set_anchors(self, joint: int, parent_anchor, child_anchor) -> None:
        """Move a loop joint's anchors, its frames' origins on the caller's model, which holds them; the caller then
        notifies the joint properties."""
        self.link(joint)
        for frames, anchor in (
            (self.source.joint_X_p, parent_anchor),
            (self.source.joint_X_c, child_anchor),
        ):
            values = frames.numpy()
            values[int(joint), :3] = np.asarray(anchor, dtype=np.float32)
            frames.assign(values)

    def anchors(self, joint: int) -> tuple[np.ndarray, np.ndarray]:
        """The ``(parent_anchor, child_anchor)`` of a loop joint [m]."""
        self.link(joint)
        return (
            self.source.joint_X_p.numpy()[int(joint), :3].copy(),
            self.source.joint_X_c.numpy()[int(joint), :3].copy(),
        )

    # --- the step's boundary ------------------------------------------------------------------------------------
    def mirror(self, state: State) -> State:
        """``state`` as the solver holds it: its mirror, with its own arrays in the forest's layout and everything else
        shared, or ``state`` itself on a pure forest."""
        if self.pure:
            return state
        mirror = self._states.get(state)
        if mirror is None:
            mirror = copy.copy(state)
            # Inside a graph capture wp.zeros would be a memset that replays every launch. Every entry here but the
            # shadows' wrenches is written before it is read, and a shadow's wrench stays zero through a replayed fill.
            count, device = self.model.body_count, self.model.device
            mirror.body_q = wp.empty(count, dtype=wp.transform, device=device)
            mirror.body_qd = wp.empty(count, dtype=wp.spatial_vector, device=device)
            mirror.body_f = wp.empty(count, dtype=wp.spatial_vector, device=device)
            mirror.body_f.zero_()
            mirror.joint_q = wp.empty_like(state.joint_q)
            mirror.joint_qd = wp.empty_like(state.joint_qd)
            self._states[state] = mirror
        return mirror

    def trees(self, articulations: wp.array | None) -> wp.array | None:
        """The trees of the model's ``articulations``, which name them only on a pure forest: every tree otherwise."""
        return articulations if self.pure else None

    def _control(self, control: Control | None) -> Control | None:
        """``control``'s mirror, its joint inputs gathered."""
        if control is None:
            return None
        mirror = self._controls.get(control)
        if mirror is None:
            mirror = copy.copy(control)
            for name in ("joint_f", "joint_target_q", "joint_target_qd", "joint_act"):
                value = getattr(control, name)
                setattr(mirror, name, None if value is None else wp.empty_like(value))
            self._controls[control] = mirror
        for name, map_ in (
            ("joint_f", self._dof_map),
            ("joint_target_q", self._target_map),
            ("joint_target_qd", self._dof_map),
            ("joint_act", self._dof_map),
        ):
            value = getattr(control, name)
            if value is not None:
                map_.move(value, getattr(mirror, name))
        return mirror

    def enter(
        self, state_in: State, state_out: State, control: Control | None, cache_valid: wp.array
    ) -> tuple[State, State, Control | None]:
        """The step's input in the forest's layout, its output's mirror and its control, seated (the module docstring).

        Args:
            state_in: The caller's input state.
            state_out: The caller's output state.
            control: The caller's control.
            cache_valid: Each tree's kinematics cache bit; every tree of a component that holds a cleared one clears
                its own, since seating moves it.
        """
        self._gather_model()
        forest_in, forest_out = self.mirror(state_in), self.mirror(state_out)
        device, count = self.model.device, self.source.body_count
        self._coord_map.move(state_in.joint_q, forest_in.joint_q)
        self._dof_map.move(state_in.joint_qd, forest_in.joint_qd)
        wp.copy(forest_in.body_q, state_in.body_q, count=count)
        wp.copy(forest_in.body_qd, state_in.body_qd, count=count)
        wp.copy(forest_in.body_f, state_in.body_f, count=count)

        launch(
            _seat,
            self._seat_shape,
            inputs=list(self._components),
            outputs=[cache_valid, self._seated_trees, self._seated_links],
            device=device,
            tiled=False,
        )
        eval_fk(self.model, forest_in.joint_q, forest_in.joint_qd, forest_in, mask=self._seated_trees)
        self._seat_map.move(forest_in.body_q, forest_in.body_q)
        self._seat_map.move(forest_in.body_qd, forest_in.body_qd)
        eval_ik(self._seats, forest_in, forest_in.joint_q, forest_in.joint_qd, mask=self._seated_links)
        return forest_in, forest_out, self._control(control)

    def leave(self, state_in: State, state_out: State) -> None:
        """Publish the step: the output's coordinates, and both states' bodies (the step's primary pieces, the model's
        bodies, whose indices the forest keeps)."""
        count = self.source.body_count
        forest_in, forest_out = self.mirror(state_in), self.mirror(state_out)
        self._publish_q.move(forest_out.joint_q, state_out.joint_q)
        self._publish_qd.move(forest_out.joint_qd, state_out.joint_qd)
        for state, mirror in ((state_in, forest_in), (state_out, forest_out)):
            wp.copy(state.body_q, mirror.body_q, count=count)
            wp.copy(state.body_qd, mirror.body_qd, count=count)

    def follow_shadows(self, state_in: State, state_out: State) -> None:
        """After :meth:`leave`, where links hold shadows: a caller may copy its output over its input and step that, so
        the input mirror's shadows take the output's poses and twists."""
        count = self.source.body_count
        forest_in, forest_out = self.mirror(state_in), self.mirror(state_out)
        for name in ("body_q", "body_qd"):
            wp.copy(
                getattr(forest_in, name),
                getattr(forest_out, name),
                dest_offset=count,
                src_offset=count,
                count=self.shadow.size,
            )

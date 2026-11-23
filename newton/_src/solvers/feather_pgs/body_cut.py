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
and are the same arrays: gravity, the shapes, the constraint mimics' coefficients, and the speed limits, which a step
reads only for contact and free bodies, primaries all. The mimics' coefficients are read at a notification only, so the
joint-owned ones are gathered there. The model's topology is derived once, and a notification that finds it changed
raises, as the held tree's proof does on a pure forest.

**Seating.** A caller writes a tree's coordinates and not the loop joints' (a reset, a teleport), and a model's
``joint_q`` never holds them to begin with. So wherever the tree factor is about to re-derive a tree's kinematics
(``fk_id_cache_valid == 0``), :meth:`BodyCut.enter` first puts each of its shadows on its primary piece, and inverse
kinematics reads the loop joint's coordinates from that pose. Steady stepping seats nothing, so a coordinate stays
continuous past a full turn. The test is the device's, so it holds in a captured graph.

**Loose bodies.** Bodies that joints join outside every articulation (``add_link`` and ``add_joint_*`` with no
``add_articulation``) form trees here, so they move: each group of them joined through such joints is rooted at its
joint to the world, else at a free joint the cut adds on its first body that no such joint enters, and its joints become
the tree's from the root; a joint that would give a body a second parent is a loop joint, as above. A group's state is
its bodies' (``body_q``, ``body_qd``): an added free root takes its body's pose and twist at every step, and the group's
joint coordinates follow its bodies wherever the step re-derives its kinematics (the first step, a reset, a
notification). The caller's model and state keep their layout; the added roots' coordinates live in the mirrors only.

**Disabled links.** A disabled link leaves its shadow massless and kinematic: its share returns to the primary piece,
which is then the whole body again, and the shadow's coordinates, under the kinematic armature, cost the tree nothing.
"""

from __future__ import annotations

import copy
import dataclasses
import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from ...core import memory
from ...core.execution import Shape, Work, launch, shape
from ...sim import BodyFlags, Control, JointType, Model, ModelBuilder, State
from ...sim.articulation import (
    compute_3d_rotational_dofs,
    eval_articulation_ik,
    eval_fk,
    eval_ik,
    invert_3d_rotational_dofs,
)
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
        if items.active[i]:
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
def _take_roots(
    bodies: wp.array[int],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Each added free root's coordinates, its body's pose and twist (``(v_com, omega)`` in the world, the free joint's
    as the solver holds it): the joint's frames are the identity on the world."""
    r = wp.tid()
    body = bodies[r]
    pose = body_q[body]
    twist = body_qd[body]
    for k in range(7):
        joint_q[q_start[r] + k] = pose[k]
    for k in range(6):
        joint_qd[qd_start[r] + k] = twist[k]


@wp.struct
class _Spheres:
    """The free spheres (:func:`_free_spheres`): each one's forest joint and tree, whether its tree is loose, and its
    coordinates' and speeds' starts in the caller's layout (three Euler angles and rates) and the forest's (a ball's
    quaternion and spin)."""

    joint: wp.array[int]
    tree: wp.array[int]
    loose: wp.array[int]
    caller_q: wp.array[int]
    caller_qd: wp.array[int]
    forest_q: wp.array[int]
    forest_qd: wp.array[int]


@wp.kernel(enable_backward=False)
def _spheres_in(
    spheres: _Spheres,
    valid: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    caller_q: wp.array[float],
    caller_qd: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Each free sphere's ball coordinates and spin (in its parent frame) at the enter: where the step re-derives its
    tree's kinematics, from the caller's Euler angles and rates (Newton's D6 forward kinematics, smooth everywhere);
    else from its bodies, as a ball joint's inverse kinematics reads them (a loose tree's always: its state is its
    bodies')."""
    k = wp.tid()
    j = spheres.joint[k]
    rotation = wp.quat_identity()
    spin = wp.vec3(0.0)
    if spheres.loose[k] == 0 and valid[spheres.tree[k]] == 0:
        a = spheres.forest_qd[k]
        q = spheres.caller_q[k]
        qd = spheres.caller_qd[k]
        rotation, spin = compute_3d_rotational_dofs(
            joint_axis[a], joint_axis[a + 1], joint_axis[a + 2],
            caller_q[q], caller_q[q + 1], caller_q[q + 2], caller_qd[qd], caller_qd[qd + 1], caller_qd[qd + 2],
        )  # fmt: skip
    else:
        parent = joint_parent[j]
        frame = joint_X_p[j]
        parent_spin = wp.vec3(0.0)
        if parent >= 0:
            frame = body_q[parent] * frame
            parent_spin = wp.spatial_bottom(body_qd[parent])
        child = joint_child[j]
        q_p = wp.transform_get_rotation(frame)
        rotation = wp.quat_inverse(q_p) * wp.transform_get_rotation(body_q[child] * joint_X_c[j])
        spin = wp.quat_rotate_inv(q_p, wp.spatial_bottom(body_qd[child]) - parent_spin)
    for i in range(4):
        joint_q[spheres.forest_q[k] + i] = rotation[i]
    for i in range(3):
        joint_qd[spheres.forest_qd[k] + i] = spin[i]


@wp.kernel(enable_backward=False)
def _spheres_out(
    spheres: _Spheres,
    joint_axis: wp.array[wp.vec3],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    caller_q: wp.array[float],
    caller_qd: wp.array[float],
):
    """Each free sphere's Euler angles and rates for the caller, from its ball coordinates and spin (Newton's D6
    inverse kinematics): the caller's readout, singular where the middle angle is at a right angle (its rates there)."""
    k = wp.tid()
    q = spheres.forest_q[k]
    qd = spheres.forest_qd[k]
    rotation = wp.quat(joint_q[q], joint_q[q + 1], joint_q[q + 2], joint_q[q + 3])
    spin = wp.vec3(joint_qd[qd], joint_qd[qd + 1], joint_qd[qd + 2])
    angles, rates = invert_3d_rotational_dofs(
        joint_axis[qd], joint_axis[qd + 1], joint_axis[qd + 2], wp.quat_identity(), rotation, spin
    )
    for i in range(3):
        caller_q[spheres.caller_q[k] + i] = angles[i]
        caller_qd[spheres.caller_qd[k] + i] = rates[i]


@wp.kernel(enable_backward=False)
def _loose_rederived(trees: wp.array[int], valid: wp.array[int], rederived: wp.array[wp.bool]):
    """Each loose tree whose kinematics the step re-derives (its cache bit cleared): its joints then follow its
    bodies."""
    k = wp.tid()
    rederived[trees[k]] = valid[trees[k]] == 0


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


def _loose_trees(model: Model) -> SimpleNamespace | None:
    """The trees of the bodies joints join outside every articulation (the module docstring), or ``None`` where no
    such joint has a child no tree holds and a parent that is the world or a body no tree holds, or where no such group
    has a coordinate (a body fixed to the world without an articulation stays outside the solver).

    Each group is rooted at its first joint on the world, else at an added free root on its first body no such joint
    enters (else its first body), and grows from the root along its joints' directions, breadth first in joint order.

    Returns:
        ``order`` (the tree joints in their trees' parent-first order, an added root ``joint_count + r``), ``tree``
        (each one's new tree), ``ancestor`` (each one's tree joint above it, -1 at a root), ``roots`` (each added
        root's body), ``trees`` (the new trees' count) and ``root_body`` (each new tree's root body).
    """
    if not (model.joint_count and model.body_count):
        return None
    articulation = model.joint_articulation.numpy()
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    owned = (articulation >= 0) & (child >= 0)
    held = np.zeros(model.body_count, dtype=bool)
    held[child[owned]] = True
    on_free = parent < 0
    on_free[~on_free] = ~held[parent[~on_free]]
    loose = np.flatnonzero(~owned & (child >= 0) & ~held[np.maximum(child, 0)] & on_free)
    if not loose.size:
        return None
    on_world, between = loose[parent[loose] < 0], loose[parent[loose] >= 0]
    below: dict[int, list[int]] = {}
    for j in between:
        below.setdefault(int(parent[j]), []).append(int(j))
    entered = np.zeros(model.body_count, dtype=bool)
    entered[child[between]] = True
    tree_of_body = np.full(model.body_count, -1, dtype=np.int64)
    joint_of_body = np.full(model.body_count, -1, dtype=np.int64)
    order, roots, root_body = [], [], []

    def plant(body: int, joint: int) -> None:
        tree_of_body[body], joint_of_body[body] = len(root_body), joint
        root_body.append(body)
        order.append(joint)
        queue = deque([body])
        while queue:
            above = queue.popleft()
            for j in below.get(above, ()):
                c = int(child[j])
                if tree_of_body[c] < 0:
                    tree_of_body[c], joint_of_body[c] = tree_of_body[above], j
                    order.append(j)
                    queue.append(c)

    for j in on_world:
        if tree_of_body[child[j]] < 0:
            plant(int(child[j]), int(j))
    members = np.unique(np.concatenate((child[loose], parent[between])))
    for body in members[np.lexsort((members, entered[members]))]:
        if tree_of_body[body] < 0:
            plant(int(body), model.joint_count + len(roots))
            roots.append(int(body))
    order = np.asarray(order, dtype=np.int64)
    added = order >= model.joint_count
    bodies = np.empty(order.size, dtype=np.int64)
    bodies[added] = np.asarray(roots, dtype=np.int64)[order[added] - model.joint_count]
    bodies[~added] = child[order[~added]]
    # A tree joint's ancestor is its parent body's tree joint; a root's (on the world, or added) is none.
    above = np.full(order.size, -1, dtype=np.int64)
    on_body = np.flatnonzero(~added)
    on_body = on_body[parent[order[on_body]] >= 0]
    above[on_body] = joint_of_body[parent[order[on_body]]]
    # A group that cannot move (rooted on the world through joints without coordinates) stays outside the solver, as a
    # body fixed to the world does: only groups with a coordinate become trees.
    tree = tree_of_body[bodies]
    widths = np.where(added, 6, np.diff(model.joint_qd_start.numpy())[np.where(added, 0, order)])
    moving = np.bincount(tree, weights=widths, minlength=len(root_body)) > 0
    if not moving.any():
        return None
    keep = moving[tree]
    renumber = np.cumsum(moving) - 1
    return SimpleNamespace(
        order=order[keep],
        tree=renumber[tree[keep]],
        ancestor=above[keep],
        roots=np.asarray(roots, dtype=np.int64),
        trees=int(moving.sum()),
        root_body=np.asarray(root_body, dtype=np.int64)[moving],
    )


_FREE_LIMIT = 1.0e6
"""A coordinate limit at or past this magnitude bounds nothing: the coordinate is free."""


def _free_spheres(model: Model) -> np.ndarray:
    """The model's D6 joints that are ball joints: no linear axis and three angular axes, each free (no limit, no drive
    target gain, no passive spring or damper). Their Euler coordinates have a singular configuration (the middle angle
    at a right angle) that the joint does not: the forest holds them as ball joints (a quaternion), and the Euler
    coordinates are the caller's readout only (:meth:`BodyCut.enter`, :meth:`BodyCut.leave`)."""
    if not model.joint_count:
        return np.zeros(0, dtype=np.int64)
    dims = model.joint_dof_dim.numpy()
    candidates = np.flatnonzero((model.joint_type.numpy() == int(JointType.D6)) & (dims[:, 0] == 0) & (dims[:, 1] == 3))
    if not candidates.size:
        return candidates
    dofs = model.joint_qd_start.numpy()[candidates][:, None] + np.arange(3)
    free = (model.joint_limit_lower.numpy()[dofs] <= -_FREE_LIMIT) & (
        model.joint_limit_upper.numpy()[dofs] >= _FREE_LIMIT
    )
    for name in ("joint_target_ke", "joint_target_kd", "joint_stiffness", "joint_damping"):
        values = getattr(model, name, None)
        if values is not None:
            free &= values.numpy()[dofs] == 0.0
    return candidates[np.all(free, axis=1)]


def _restructured(model: Model, loose: SimpleNamespace | None, spheres: np.ndarray) -> Model:
    """``model`` as the cut reads it: its loose trees rooted (:func:`_loose_trees`: their joints in new articulations
    past the model's, each added root a free joint, as :meth:`~newton.ModelBuilder.add_joint_free` makes it, on the
    world with identity frames, past the model's joints, its coordinates past theirs), and its free spheres
    (:func:`_free_spheres`) ball joints, their four coordinates past those. A joint's coordinates are then
    ``_coord_first`` and ``_coord_width`` (an added or a sphere's need not follow its neighbour's)."""
    builder = ModelBuilder()
    builder.add_body()
    link = builder.add_link()
    builder.add_articulation([builder.add_joint_ball(-1, link)])
    template = builder.finalize(device="cpu")
    added = 0 if loose is None else loose.roots.size
    trees = 0 if loose is None else loose.trees
    device = model.device
    # Each frequency's appended rows and the template's rows that fill them (the free joint's, then the ball's).
    stencils = {
        _F.JOINT: ((0, 1, added),),
        _F.JOINT_DOF: ((0, 6, added),),
        _F.JOINT_COORD: ((0, 7, added), (7, 11, spheres.size)),
        _F.ARTICULATION: ((0, 1, trees),),
    }
    starts = ("joint_q_start", "joint_qd_start", "articulation_start", "articulation_end")
    rooted = copy.copy(model)
    for name, spec in model.attribute_specs.items():
        if spec.frequency not in stencils or name in starts:
            continue
        value = getattr(model, name, None)
        if value is None or len(value) != model.attribute_frequency_count(spec.frequency):
            continue
        stencil = getattr(template, name, None)
        if isinstance(value, list):
            tail = [
                stencil[first + k % (stop - first)] if isinstance(stencil, list) and len(stencil) >= stop else None
                for first, stop, copies in stencils[spec.frequency]
                for k in range(copies * (stop - first))
            ]
            setattr(rooted, name, [*value, *tail])
            continue
        host = value.numpy()
        tails = []
        for first, stop, copies in stencils[spec.frequency]:
            if isinstance(stencil, wp.array) and len(stencil) >= stop and stencil.dtype == value.dtype:
                rows = stencil.numpy()[first:stop]
                tails.append(np.tile(rows, (copies,) + (1,) * (rows.ndim - 1)))
            else:
                tails.append(np.zeros((copies * (stop - first), *host.shape[1:]), dtype=host.dtype))
        setattr(rooted, name, wp.array(np.concatenate((host, *tails)), dtype=value.dtype, device=device))

    joints, articulations = model.joint_count, model.articulation_count
    count = joints + added

    def edit(name, rows, values, dtype=wp.int32):
        host = getattr(rooted, name).numpy()
        host[rows] = values
        setattr(rooted, name, wp.array(host, dtype=dtype, device=device))

    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    first = np.concatenate((q_start[:-1], model.joint_coord_count + 7 * np.arange(added)))
    width = np.concatenate((np.diff(q_start), np.full(added, 7)))
    if spheres.size:
        edit("joint_type", spheres, int(JointType.BALL))
        first[spheres] = model.joint_coord_count + 7 * added + 4 * np.arange(spheres.size)
        width[spheres] = 4
    rooted._coord_first, rooted._coord_width = first.astype(np.int64), width.astype(np.int64)
    if loose is not None:
        added_rows = np.arange(joints, count)
        edit("joint_child", added_rows, loose.roots)
        edit("joint_parent", added_rows, -1)
        edit("joint_articulation", loose.order, articulations + loose.tree)
        edit("joint_ancestor", loose.order, loose.ancestor)
        identity = np.array(wp.transform_identity(), dtype=np.float32)
        edit("joint_X_p", added_rows, identity, wp.transform)
        edit("joint_X_c", added_rows, identity, wp.transform)
        body_world = model.body_world.numpy()
        edit("joint_world", added_rows, body_world[loose.roots])
        edit("articulation_world", np.arange(articulations, articulations + trees), body_world[loose.root_body])
        dof_tail = model.joint_dof_count + 6 * np.arange(added + 1)
        rooted.joint_qd_start = wp.array(np.concatenate((qd_start[:-1], dof_tail)), dtype=wp.int32, device=device)
    rooted.joint_count = count
    rooted.joint_dof_count = model.joint_dof_count + 6 * added
    rooted.joint_coord_count = model.joint_coord_count + 7 * added + 4 * spheres.size
    rooted.articulation_count = articulations + trees
    return rooted


def _topology(model: Model, loose: SimpleNamespace | None = None, forced: bool = False) -> SimpleNamespace | None:
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
    if not forced and not loops.size and not np.any(np.bincount(articulation[owned & (ancestor < 0)], minlength=1) > 1):
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
    # A tree keeps its joints' order, parent-first: the model's, or a loose tree's growth (:func:`_loose_trees`).
    place = joints.copy()
    if loose is not None:
        place[loose.order] = np.arange(loose.order.size)
    order = np.lexsort((place, ~owned, tree))
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


def _coordinates(first: np.ndarray, width: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The coordinates of the joints ``order`` (each joint's ``width`` from its ``first``), in that order, and their
    joints' starts among them."""
    widths = np.asarray(width)[order]
    joint, rank = host_index.expand(widths)
    return (np.asarray(first)[order][joint] + rank).astype(np.int32), host_index.offsets(widths, np.int32)


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
        loose = _loose_trees(model)
        spheres = _free_spheres(model)
        rooted = model if loose is None and not spheres.size else _restructured(model, loose, spheres)
        topology = _topology(rooted, loose, forced=rooted is not model) if model.joint_count else None
        if topology is not None:
            cut._cut(topology, rooted, loose, spheres)
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

    def _cut(
        self, topology: SimpleNamespace, source: Model, loose: SimpleNamespace | None, spheres: np.ndarray
    ) -> None:
        """Build the forest of ``topology`` from ``source``: the caller's model, or it restructured
        (:func:`_restructured`: its loose trees rooted, its free spheres ball joints), whose added joints' items and
        spheres' state the step's gathers and publications skip (the spheres' state crosses by its conversion)."""
        caller = self.source
        device = source.device
        loops, order, position = topology.loops, topology.order, topology.position
        self.joints = loops
        self.primary = source.joint_child.numpy()[loops].astype(np.int32)
        self.shadow = (source.body_count + np.arange(loops.size)).astype(np.int32)
        self.enabled = (source.joint_enabled.numpy()[loops] != 0).astype(np.int32)
        self._position = position
        self._anchored = loops[~topology.on_tree & (source.joint_parent.numpy()[loops] >= 0)]

        q_starts = source.joint_q_start.numpy()
        coords, q_start = _coordinates(
            getattr(source, "_coord_first", q_starts[:-1]), getattr(source, "_coord_width", np.diff(q_starts)), order
        )
        qd_starts = source.joint_qd_start.numpy()
        dofs, qd_start = _coordinates(qd_starts[:-1], np.diff(qd_starts), order)
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
            elif isinstance(value, wp.array | list) and len(value) == source.attribute_frequency_count(spec.frequency):
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
        # The model's cached forward-kinematics schedules (a serial joint order, the levels of the tiled kernel)
        # describe the source's joints; the forest's joints are parent-first within each tree, so its forward
        # kinematics take the serial path in joint order.
        forest._fk_joint_order = None
        forest._fk_articulation_level_start = forest._fk_level_joint_start = None
        forest._fk_level_joints = forest._fk_level_parent_pos = None
        forest._fk_level_capacity = 0
        forest.max_joints_per_articulation = int(np.diff(bounds).max())
        forest.max_dofs_per_articulation = int(
            np.bincount(np.repeat(topology.tree, np.diff(qd_start)), minlength=topology.trees)[: topology.trees].max()
        )
        if source.constraint_mimic_count:
            forest.constraint_mimic_joint0 = indices(position[source.constraint_mimic_joint0.numpy()])
            forest.constraint_mimic_joint1 = indices(position[source.constraint_mimic_joint1.numpy()])
        reference = source.joint_mimic_joint.numpy()[order]
        forest.joint_mimic_joint = indices(np.where(reference >= 0, position[reference], -1))
        self.model = forest
        self._topology = {name: getattr(caller, name).numpy().copy() for name in _TOPOLOGY}

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

        # An added root's items have no counterpart in the caller's layout: its joint's frames and gains stay the
        # free joint's, its coordinates the mirrors' own (taken from its body at each step).
        def caller_items(values, count):
            return wp.array(np.asarray(values) < count, dtype=wp.bool, device=device)

        in_joints = caller_items(order, caller.joint_count)
        # A free sphere's speeds are its spin as a ball joint's, its Euler rates the caller's: they cross converted.
        sphere_dofs = np.zeros(caller.joint_dof_count + 1, dtype=bool)
        caller_qd = caller.joint_qd_start.numpy()
        for joint in spheres:
            sphere_dofs[caller_qd[joint] : caller_qd[joint + 1]] = True
        forest_dofs = dofs.numpy()
        in_dofs = wp.array(
            (forest_dofs < caller.joint_dof_count) & ~sphere_dofs[np.minimum(forest_dofs, caller.joint_dof_count)],
            dtype=wp.bool,
            device=device,
        )
        in_coords = caller_items(coords.numpy(), caller.joint_coord_count)
        self._joint_map = mapped(order.size, indices(order), identity, in_joints)
        self._body_map = mapped(bodies.size, indices(bodies), identity)
        self._dof_map = mapped(dofs.shape[0], dofs, identity, in_dofs)
        self._coord_map = mapped(coords.shape[0], coords, identity, in_coords)
        self._publish_q = mapped(q_start[bounds[-1]], identity, coords, in_coords)
        self._publish_qd = mapped(qd_start[bounds[-1]], identity, dofs, in_dofs)
        # The loose trees: the added roots (their bodies, their coordinates' starts in the forest) and every loose
        # tree, whose joints follow its bodies where the step re-derives its kinematics.
        self._spheres = None
        if spheres.size:
            sphere_joint = position[spheres]
            loose_tree = np.zeros(topology.trees + 1, dtype=np.int32)
            if loose is not None:
                loose_tree[np.unique(topology.tree[np.isin(order, loose.order)])] = 1
            record = _Spheres()
            record.joint, record.tree = indices(sphere_joint), indices(topology.tree[sphere_joint])
            record.loose = indices(loose_tree[topology.tree[sphere_joint]])
            record.caller_q = indices(caller.joint_q_start.numpy()[spheres])
            record.caller_qd = indices(caller_qd[spheres])
            record.forest_q, record.forest_qd = indices(q_start[sphere_joint]), indices(qd_start[sphere_joint])
            self._spheres = record
        self._roots = None
        if loose is not None:
            forest_joint = position[caller.joint_count + np.arange(loose.roots.size)]
            self._roots = (indices(loose.roots), indices(q_start[forest_joint]), indices(qd_start[forest_joint]))
            loose_trees = np.unique(topology.tree[np.isin(order, loose.order)])
            self._loose = indices(loose_trees)
            self._rederived = wp.zeros(topology.trees, dtype=wp.bool, device=device)
        self._seat_map = mapped(loops.size, indices(self.primary), indices(self.shadow), self._seated_links)
        # The bodies no tree holds stand outside the solver: the step passes their pose and twist through (no finish
        # writes them), so the output's mirror takes the input's at the enter.
        outside = np.flatnonzero(body_tree[: caller.body_count] < 0)
        self._outside = mapped(outside.size, indices(outside), indices(outside)) if outside.size else None
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
        # The mimic coefficients are read at a notification only (:class:`.forest_plan.MimicRelations`).
        self._joint_map.move(self.source.joint_mimic_coeffs, self.model.joint_mimic_coeffs)

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
            mirror.joint_q = wp.empty(self.model.joint_coord_count, dtype=state.joint_q.dtype, device=device)
            mirror.joint_qd = wp.empty(self.model.joint_dof_count, dtype=state.joint_qd.dtype, device=device)
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
            # An added root's entries hold zero: no force, no target.
            for name, count in (
                ("joint_f", self.model.joint_dof_count),
                ("joint_target_q", self.model.joint_coord_count),
                ("joint_target_qd", self.model.joint_dof_count),
                ("joint_act", self.model.joint_dof_count),
            ):
                value = getattr(control, name)
                setattr(
                    mirror, name, None if value is None else wp.zeros(count, dtype=value.dtype, device=value.device)
                )
            self._controls[control] = mirror
        for name, map_ in (
            ("joint_f", self._dof_map),
            ("joint_target_q", self._coord_map),
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
        if self._outside is not None:
            self._outside.move(state_in.body_q, forest_out.body_q)
            self._outside.move(state_in.body_qd, forest_out.body_qd)

        launch(
            _seat,
            self._seat_shape,
            inputs=list(self._components),
            outputs=[cache_valid, self._seated_trees, self._seated_links],
            device=device,
            tiled=False,
        )
        if self._spheres is not None:
            forest = self.model
            wp.launch(
                _spheres_in,
                dim=self._spheres.joint.shape[0],
                inputs=[
                    self._spheres,
                    cache_valid,
                    forest.joint_parent,
                    forest.joint_child,
                    forest.joint_X_p,
                    forest.joint_X_c,
                    forest.joint_axis,
                    forest_in.body_q,
                    forest_in.body_qd,
                    state_in.joint_q,
                    state_in.joint_qd,
                ],
                outputs=[forest_in.joint_q, forest_in.joint_qd],
                device=device,
            )
        if self._roots is not None:
            self._follow_loose(forest_in, cache_valid)
        eval_fk(self.model, forest_in.joint_q, forest_in.joint_qd, forest_in, mask=self._seated_trees)
        self._seat_map.move(forest_in.body_q, forest_in.body_q)
        self._seat_map.move(forest_in.body_qd, forest_in.body_qd)
        eval_ik(self._seats, forest_in, forest_in.joint_q, forest_in.joint_qd, mask=self._seated_links)
        return forest_in, forest_out, self._control(control)

    def _follow_loose(self, forest_in: State, cache_valid: wp.array) -> None:
        """The loose trees' state is their bodies': each added root takes its body's pose and twist, and a loose tree
        whose kinematics the step re-derives takes its joints' coordinates from its bodies (inverse kinematics)."""
        forest, device = self.model, self.model.device
        bodies, q_start, qd_start = self._roots
        wp.launch(
            _take_roots,
            dim=bodies.shape[0],
            inputs=[bodies, q_start, qd_start, forest_in.body_q, forest_in.body_qd],
            outputs=[forest_in.joint_q, forest_in.joint_qd],
            device=device,
        )
        wp.launch(
            _loose_rederived,
            dim=self._loose.shape[0],
            inputs=[self._loose, cache_valid],
            outputs=[self._rederived],
            device=device,
        )
        wp.launch(
            eval_articulation_ik,
            dim=(self._loose.shape[0], forest.max_joints_per_articulation),
            inputs=[
                forest.articulation_start,
                forest.articulation_end,
                forest.articulation_count,
                self._rederived,
                self._loose,
                forest_in.body_q,
                forest_in.body_qd,
                forest.body_com,
                forest.joint_type,
                forest.joint_parent,
                forest.joint_child,
                forest.joint_X_p,
                forest.joint_X_c,
                forest.joint_axis,
                forest.joint_dof_dim,
                forest.joint_q_start,
                forest.joint_qd_start,
                forest.body_flags,
                int(BodyFlags.ALL),
            ],
            outputs=[forest_in.joint_q, forest_in.joint_qd],
            device=device,
        )

    def leave(self, state_in: State, state_out: State) -> None:
        """Publish the step: the output's coordinates, and both states' bodies (the step's primary pieces, the model's
        bodies, whose indices the forest keeps)."""
        count = self.source.body_count
        forest_in, forest_out = self.mirror(state_in), self.mirror(state_out)
        self._publish_q.move(forest_out.joint_q, state_out.joint_q)
        self._publish_qd.move(forest_out.joint_qd, state_out.joint_qd)
        if self._spheres is not None:
            wp.launch(
                _spheres_out,
                dim=self._spheres.joint.shape[0],
                inputs=[self._spheres, self.model.joint_axis, forest_out.joint_q, forest_out.joint_qd],
                outputs=[state_out.joint_q, state_out.joint_qd],
                device=self.model.device,
            )
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

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Armature that keeps an articulation's mass matrix within float32 factorization, of every tree by two rules.

The solver factors each articulation's joint-space mass matrix in float32,
leaf-first (a dense factor of the reversed coordinates, or the sparse factor in
:meth:`.TreePlan.factor`'s order). A pivot of that factorization is the inertia
a coordinate keeps once the coordinates before it move freely; it is the difference of the coordinate's
diagonal and the terms those coordinates take from it, so float32 resolves it
only to about ``n`` units in the last place of the diagonal, ``n`` the
articulation's coordinate count. Along a serial chain the ratio of a pivot to
its diagonal falls roughly as the cube of the chain's length (Featherstone,
"An empirical study of the joint space inertia matrix", 2004): a 100-link
cable without armature reaches 3.5e-7, below the 6e-6 that float32 resolves,
and its factorization meets a negative pivot within a few steps.

:func:`conditioning_armature` adds, where an articulation's mass matrix at the
model's configuration has a pivot below ``4 n u`` (``u`` = 2^-24) times its
diagonal, the least armature that lifts it there, and nothing elsewhere: a
model whose pivots all clear the floor keeps its armature bit for bit. Only
articulations with a root-to-leaf path of :data:`DEEP_CHAIN_JOINTS` joints or
more are evaluated; shallower trees keep pivot ratios far above the floor
unless their masses span many orders of magnitude.

:func:`assembly_floor` is the other rule, for every tree. A rotation's diagonal is assembled in float32 about the
articulation's origin, from terms of order ``m R^2`` that cancel down to the inertia about the axis, so it is resolved
only to about ``n`` units in the last place of those terms: a coordinate about a thin body's own axis, far from the
origin, can round to zero or below, and the factor to NaN. The rule adds the least armature that lifts such a diagonal
to ``4 n u`` of its terms, and a model with no such coordinate keeps its armature bit for bit. Every model takes it at
construction and at every notification, in one vectorized pass over all worlds' joints by depth.
"""

import dataclasses

import numpy as np
import warp as wp

from ...sim import JointType
from . import host_index
from .tree_plan import TreePlan

DEEP_CHAIN_JOINTS = 32
"""Root-to-leaf joint count from which an articulation's mass matrix is evaluated: a uniform chain of 32 links
keeps its pivot ratios near 1e-4, ten times the floor."""

FLOAT32_ROUNDING = 2.0**-24
"""Unit roundoff of float32."""

PIVOT_FLOOR_FACTOR = 4.0
"""The pivot floor is ``PIVOT_FLOOR_FACTOR * n * FLOAT32_ROUNDING`` times a coordinate's diagonal."""


@dataclasses.dataclass(frozen=True)
class ArmatureFloor:
    """The armature :func:`conditioning_armature` added, for the route record."""

    articulations: int
    """Articulations that received armature."""
    coordinates: int
    """Coordinates that received armature."""
    largest: float
    """Largest armature added [kg m^2 or kg]."""
    evaluated: int
    """Articulations whose mass matrix was evaluated (deep chains)."""


def articulation_depths(model, joint_forest: TreePlan) -> np.ndarray:
    """Number of joints on the longest root-to-leaf path of each articulation of ``model``.

    ``joint_forest`` is the forest of the model's joint ancestry (``Model.joint_ancestor``).
    """
    depths = np.zeros(model.articulation_count, dtype=np.int64)
    if model.joint_count and model.articulation_count:
        articulation = model.joint_articulation.numpy()
        member = articulation >= 0
        np.maximum.at(depths, articulation[member], joint_forest.depth[member].astype(np.int64) + 1)
    return depths


def pivot_floor(mass: np.ndarray, rounding: float = FLOAT32_ROUNDING) -> np.ndarray:
    """Least diagonal additions that lift every leaf-first Cholesky pivot of ``mass`` to its floor: the pivots of
    ``P mass P^T`` with the coordinates reversed, the order the dense factors take.

    Args:
        mass: Mass matrices with their armature, shape ``[count, n, n]``.
        rounding: Unit roundoff of the factorization's arithmetic.

    Returns:
        The additions, shape ``[count, n]``: zero wherever a pivot of the
        float64 factorization clears ``PIVOT_FLOOR_FACTOR * n * rounding`` times its diagonal.
    """
    mass = np.asarray(mass, dtype=np.float64)[:, ::-1, ::-1]
    count, n, _ = mass.shape
    ratio = PIVOT_FLOOR_FACTOR * n * rounding
    lower = np.zeros_like(mass)
    added = np.zeros((count, n))
    for j in range(n):
        diagonal = mass[:, j, j]
        pivot = diagonal - np.einsum("bk,bk->b", lower[:, j, :j], lower[:, j, :j])
        # pivot + a >= ratio * (diagonal + a), the floor of the lifted diagonal.
        added[:, j] = np.maximum(ratio * diagonal - pivot, 0.0) / (1.0 - ratio)
        root = np.sqrt(pivot + added[:, j])
        lower[:, j, j] = root
        lower[:, j + 1 :, j] = (
            mass[:, j + 1 :, j] - np.einsum("bik,bk->bi", lower[:, j + 1 :, :j], lower[:, j, :j])
        ) / root[:, None]
    return added[:, ::-1]


def _rotated(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """``vector`` rotated by the unit ``quaternion`` (``xyzw``), broadcasting."""
    axis, scalar = quaternion[..., :3], quaternion[..., 3:]
    twice = 2.0 * np.cross(axis, vector)
    return vector + scalar * twice + np.cross(axis, twice)


def _about_axes(offset: np.ndarray) -> np.ndarray:
    """``|x|^2 E - x x^T`` per row: a unit mass's inertia at ``offset``."""
    return np.einsum("ni,ni->n", offset, offset)[:, None, None] * np.eye(3) - np.einsum("ni,nj->nij", offset, offset)


def assembly_floor(model, armature: np.ndarray, joint_forest: TreePlan) -> np.ndarray:
    """Least diagonal additions that lift every rotation's diagonal to the rounding of the terms it is assembled from,
    at the model's configuration (see the module docstring).

    Args:
        model: The model.
        armature: Effective joint armature per coordinate [kg m^2 or kg].
        joint_forest: The forest of the model's joint ancestry (``Model.joint_ancestor``).

    Returns:
        The additions, one per coordinate: zero unless a rotation's diagonal falls below ``4 n u`` of its tree's terms.
    """
    added = np.zeros(armature.shape, dtype=np.float64)
    if not (model.body_count and model.joint_count and model.articulation_count):
        return added
    articulation = model.joint_articulation.numpy()
    child, kinds = model.joint_child.numpy(), model.joint_type.numpy()
    qd_start = model.joint_qd_start.numpy()
    member = (articulation >= 0) & (child >= 0)
    tree = np.flatnonzero(member)
    pose = model.body_q.numpy().astype(np.float64)[child]
    mass = model.body_mass.numpy().astype(np.float64)[child] * member
    # The origin is the tree's root: as many terms as its first body's distance leaves.
    first = np.full(model.articulation_count, -1, dtype=np.int64)
    first[articulation[tree[::-1]]] = tree[::-1]
    origin = pose[first[articulation], :3]
    terms = np.zeros(model.articulation_count)
    offset = pose[:, :3] - origin
    np.add.at(terms, articulation[tree], (mass * np.einsum("ni,ni->n", offset, offset))[tree])
    widths = np.diff(qd_start)
    coordinates = np.zeros(model.articulation_count, dtype=np.int64)
    np.add.at(coordinates, articulation[tree], widths[tree])
    floor = PIVOT_FLOOR_FACTOR * coordinates * FLOAT32_ROUNDING * terms
    # Each subtree's mass, first moment and inertia about its tree's origin, in world axes, gathered leaf first.
    com = pose[:, :3] + _rotated(pose[:, 3:], model.body_com.numpy().astype(np.float64)[child]) - origin
    rotation = _rotated(pose[:, None, 3:], np.eye(3)).transpose(0, 2, 1)
    principal = model.body_inertia.numpy().astype(np.float64)[child] * member[:, None, None]
    moment = mass[:, None] * com
    inertia = rotation @ principal @ rotation.transpose(0, 2, 1) + mass[:, None, None] * _about_axes(com)
    bounds = joint_forest.level_offsets
    for level in range(bounds.size - 2, 0, -1):
        nodes = joint_forest.level_nodes[bounds[level] : bounds[level + 1]]
        parents = joint_forest.parent[nodes]
        np.add.at(mass, parents, mass[nodes])
        np.add.at(moment, parents, moment[nodes])
        np.add.at(inertia, parents, inertia[nodes])
    # The same inertias about each rotary joint's anchor, by the parallel axis rule.
    rotary = tree[np.isin(kinds[tree], (int(JointType.REVOLUTE), int(JointType.BALL), int(JointType.D6)))]
    frames = model.joint_X_c.numpy().astype(np.float64)[rotary]
    anchor = pose[rotary, :3] + _rotated(pose[rotary, 3:], frames[:, :3]) - origin[rotary]
    outer = np.einsum("ni,nj->nij", moment[rotary], anchor)
    block = inertia[rotary] + mass[rotary, None, None] * _about_axes(anchor) + outer + outer.transpose(0, 2, 1)
    block -= 2.0 * np.einsum("nii->n", outer)[:, None, None] * np.eye(3)
    diagonal = np.linalg.eigvalsh(block)[:, 0]
    # A hinge turns about one axis, so the parent's frame and the child's agree on it in world coordinates.
    hinges = np.flatnonzero(kinds[rotary] == int(JointType.REVOLUTE))
    axes = model.joint_axis.numpy().astype(np.float64)[qd_start[rotary[hinges]]]
    axes = _rotated(pose[rotary[hinges], 3:], _rotated(frames[hinges, 3:], axes))
    diagonal[hinges] = np.einsum("ni,nij,nj->n", axes, block[hinges], axes)
    joint, rank = host_index.expand(widths[rotary])
    dofs = qd_start[rotary[joint]] + rank
    added[dofs] = np.maximum(floor[articulation[rotary[joint]]] - diagonal[joint] - armature[dofs], 0.0)
    return added


def conditioning_armature(model, armature: np.ndarray, joint_forest: TreePlan) -> tuple[np.ndarray, ArmatureFloor]:
    """Return ``armature`` with the least additions that keep every articulation's mass matrix factorable.

    First the diagonals float32 cannot assemble (:func:`assembly_floor`). Then evaluates the mass matrix of every
    articulation with at least :data:`DEEP_CHAIN_JOINTS` joints on a root-to-leaf path at the model's configuration
    (``model.joint_q``), adds ``armature`` to its diagonal and lifts each pivot below the floor (:func:`pivot_floor`).
    Other articulations and coordinates keep ``armature`` unchanged.

    Args:
        model: The model.
        armature: Effective joint armature per coordinate [kg m^2 or kg], prescribed coordinates included.
        joint_forest: The forest of the model's joint ancestry (``Model.joint_ancestor``).

    Returns:
        The armature with the additions, and a summary of them.
    """
    assembled = assembly_floor(model, armature, joint_forest)
    lifted_now = assembled > 0.0
    coordinates = int(np.count_nonzero(lifted_now))
    largest = float(assembled.max(initial=0.0))
    articulations = 0
    if coordinates:
        armature = np.array(armature, dtype=np.float32, copy=True)
        armature[lifted_now] += assembled[lifted_now].astype(np.float32)
        owners = np.repeat(model.joint_articulation.numpy(), np.diff(model.joint_qd_start.numpy()))
        articulations = int(np.unique(owners[lifted_now]).size)
    depths = articulation_depths(model, joint_forest)
    deep = np.flatnonzero(depths >= DEEP_CHAIN_JOINTS)
    if deep.size == 0:
        return armature, ArmatureFloor(articulations, coordinates, largest, 0)
    import newton  # noqa: PLC0415

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    mask = np.zeros(model.articulation_count, dtype=bool)
    mask[deep] = True
    mass = newton.eval_mass_matrix(model, state, mask=wp.array(mask, dtype=wp.bool, device=model.device)).numpy()
    articulation_start = model.articulation_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    armature = np.array(armature, dtype=np.float32, copy=True)
    starts = joint_qd_start[articulation_start[deep]]
    sizes = joint_qd_start[articulation_start[deep + 1]] - starts
    for size in np.unique(sizes):
        group = np.flatnonzero(sizes == size)
        dofs = starts[group][:, None] + np.arange(size)[None, :]
        block = mass[deep[group], :size, :size].astype(np.float64)
        block[:, np.arange(size), np.arange(size)] += armature[dofs]
        with np.errstate(divide="ignore", invalid="ignore"):
            added = pivot_floor(block)
        # A mass matrix without a finite, positive floor (a massless coordinate, a non-finite configuration)
        # keeps its armature: its factorization fails with or without the floor.
        usable = np.isfinite(block).all(axis=(1, 2)) & np.isfinite(added).all(axis=1)
        lifted = (added > 0.0) & usable[:, None]
        armature[dofs[lifted]] += added[lifted].astype(np.float32)
        articulations += int(np.count_nonzero(lifted.any(axis=1)))
        coordinates += int(np.count_nonzero(lifted))
        if lifted.any():
            largest = max(largest, float(added[lifted].max()))
    return armature, ArmatureFloor(articulations, coordinates, largest, int(deep.size))

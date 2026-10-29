# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Armature that keeps a deep articulation's mass matrix within float32 factorization.

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
"""

import dataclasses

import numpy as np
import warp as wp

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


def conditioning_armature(model, armature: np.ndarray, joint_forest: TreePlan) -> tuple[np.ndarray, ArmatureFloor]:
    """Return ``armature`` with the least additions that keep deep articulations' mass matrices factorable.

    Evaluates the mass matrix of every articulation with at least
    :data:`DEEP_CHAIN_JOINTS` joints on a root-to-leaf path at the model's
    configuration (``model.joint_q``), adds ``armature`` to its diagonal and
    lifts each pivot below the floor (:func:`pivot_floor`). Other
    articulations and coordinates keep ``armature`` unchanged.

    Args:
        model: The model.
        armature: Effective joint armature per coordinate [kg m^2 or kg], prescribed coordinates included.
        joint_forest: The forest of the model's joint ancestry (``Model.joint_ancestor``).

    Returns:
        The armature with the additions, and a summary of them.
    """
    depths = articulation_depths(model, joint_forest)
    deep = np.flatnonzero(depths >= DEEP_CHAIN_JOINTS)
    if deep.size == 0:
        return armature, ArmatureFloor(0, 0, 0.0, 0)
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
    articulations = coordinates = 0
    largest = 0.0
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

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Place copies of a block of model rows by world transforms, as the attribute metadata declares."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..core.host_transform import transform_inverse_rows, transform_multiply_rows
from .enums import JointType

if TYPE_CHECKING:
    from .model import Model

_IDENTITY = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)

_WIDTHS = {"pose": 7, "point": 3, "direction": 3, "twist": 6, "rest_frame": 9}
"""Values per placed row of each placement kind."""


def _rotate_vectors(rotation: np.ndarray, vectors: np.ndarray) -> None:
    """Rotate ``(..., 3)`` float32 vectors in place by ``(..., 4)`` quaternions broadcast against them.

    This is the cross-product form copies of velocities have always used; poses compose with Warp's
    ``transform_multiply`` arithmetic instead.
    """
    q = np.asarray(rotation, dtype=np.float32)
    xyz = q[..., :3]
    w = q[..., 3:]
    t = 2.0 * np.cross(xyz, vectors)
    vectors += np.cross(xyz, t) + w * t


def place_copies(
    specs: Mapping[str, Model.AttributeSpec], source: Mapping[str, np.ndarray], xforms: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the values a world transform moves in each of ``len(xforms)`` copies of a block of rows.

    Each attribute of ``specs`` with a :attr:`~newton.Model.AttributeSpec.placement` and present in ``source`` is
    placed: rows in the world frame (every row, or the rows its anchor marks with ``-1``) move with their copy's
    transform; rows relative to an entity keep their values. The copies of the block are laid out one after another,
    as :meth:`~newton.ModelBuilder.replicate` lays out worlds.

    Copies with the identity transform keep their values bit for bit, and a copy that only translates keeps its
    directions and twists. Poses of every row of a domain take the translation as an addition when no copy rotates;
    points of every row of a domain always add it (zero for a copy that does not move).

    Args:
        specs: Attribute metadata, for example :attr:`newton.Model.attribute_specs`.
        source: The block's values by attribute name, as float32 arrays of shape [rows, ...] (transforms as
            ``(px, py, pz, qx, qy, qz, qw)``), with the attributes the placements read (their anchors, owner starts
            relative to the block, frames and ``joint_type`` for joint placements).
        xforms: Transform of each copy, shape [copies, 7].

    Returns:
        For each placed attribute, ``(targets, values)``: indices along the first axis of the attribute's values tiled
        to every copy, and the values to store there (shape ``targets.shape`` followed by the element shape).
    """
    xforms = np.asarray(xforms, dtype=np.float32).reshape(-1, 7)
    copies = len(xforms)
    moved = np.flatnonzero(~np.all(xforms == _IDENTITY, axis=1))
    moved_xforms = xforms[moved]
    rotating = ~np.all(moved_xforms[:, 3:] == _IDENTITY[3:], axis=1)
    rotated, rotated_xforms = moved[rotating], moved_xforms[rotating]
    offsets = np.zeros((copies, 3), dtype=np.float32)
    offsets[moved] = moved_xforms[:, :3]
    placed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, spec in specs.items():
        placement = spec.placement
        if placement is None or name not in source or not copies:
            continue
        kind = placement.kind
        values = np.asarray(source[name], dtype=np.float32)
        if placement.owner_start is not None:
            placed_rows = _owned_rows(placement, source, values, moved_xforms, rotating, moved, rotated)
            if placed_rows is not None:
                placed[name] = placed_rows
            continue
        rows = len(values)
        values = values.reshape(rows, *((3, 3) if kind == "rest_frame" else (_WIDTHS[kind],)))
        if not rows:
            continue
        stride = np.arange(copies, dtype=np.int64)[:, None] * rows
        if placement.anchor is None:
            selected = np.arange(rows, dtype=np.int64)
        else:
            selected = np.flatnonzero(_world_rows(placement, source, rows))
            if not len(selected):
                continue
        local = values[selected]
        if kind == "pose":
            if placement.anchor is None and not len(rotated):
                if not len(moved):
                    continue
                targets = stride + selected[None]
                out = np.broadcast_to(local, (copies, *local.shape)).copy()
                out[..., :3] += offsets[:, None]
            else:
                if not len(moved):
                    continue
                targets = stride[moved] + selected[None]
                out = transform_multiply_rows(moved_xforms[:, None], local[None])
        elif kind == "point":
            # Every copy of a domain-wide point adds its translation, zero for a copy that does not move.
            copied = np.arange(copies) if placement.anchor is None else moved
            if not len(copied):
                continue
            targets = stride[copied] + selected[None]
            out = np.broadcast_to(local, (len(copied), *local.shape)).copy()
            if len(rotated):
                turning = np.isin(copied, rotated)
                turned = out[turning]
                _rotate_vectors(rotated_xforms[:, None, 3:], turned)
                out[turning] = turned
            out += offsets[copied, None]
        else:
            if not len(rotated):
                continue
            targets = stride[rotated] + selected[None]
            out = np.broadcast_to(local, (len(rotated), *local.shape)).copy()
            if kind == "twist":
                twists = out.reshape(len(rotated), len(selected), 2, 3)
                _rotate_vectors(rotated_xforms[:, None, None, 3:], twists)
            elif kind == "rest_frame":
                _rotate_vectors(rotated_xforms[:, None, None, 3:], out)
            else:
                _rotate_vectors(rotated_xforms[:, None, 3:], out)
        placed[name] = (targets, out)
    return placed


def _world_rows(placement: Model.AttributePlacement, source: Mapping[str, np.ndarray], rows: int) -> np.ndarray:
    """Rows of the block in the world frame: anchored to ``-1`` and of the placement's joint types."""
    anchor = np.asarray(source[placement.anchor]).reshape(-1)
    if len(anchor) != rows:
        raise ValueError(f"Placement anchor {placement.anchor!r} has {len(anchor)} rows; the block has {rows}.")
    world = anchor == -1
    if placement.free_joint is not None:
        free = np.asarray(source["joint_type"]).reshape(-1) == int(JointType.FREE)
        world &= free if placement.free_joint else ~free
    return world


def _owned_rows(
    placement: Model.AttributePlacement,
    source: Mapping[str, np.ndarray],
    values: np.ndarray,
    moved_xforms: np.ndarray,
    rotating: np.ndarray,
    moved: np.ndarray,
    rotated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Place values that owner rows (for example free root joints) hold from their start on, in the owner's frame."""
    starts = np.asarray(source[placement.owner_start], dtype=np.int64).reshape(-1)
    owners = np.flatnonzero(_world_rows(placement, source, len(starts)))
    width = _WIDTHS[placement.kind]
    if not len(owners) or not len(moved) or (placement.kind != "pose" and not len(rotated)):
        return None
    index = starts[owners, None] + np.arange(width)
    frame = np.asarray(source[placement.frame], dtype=np.float32).reshape(-1, 7)[owners]
    # A transform X of the world moves an owner's values expressed in its frame F by F^-1 X F.
    local = transform_multiply_rows(
        transform_multiply_rows(transform_inverse_rows(frame)[None], moved_xforms[:, None]), frame[None]
    )
    copy_base = len(values) * moved[:, None, None]
    if placement.kind == "pose":
        return copy_base + index[None], transform_multiply_rows(local, values[index][None])
    twist = np.broadcast_to(values[index], (len(rotated), *index.shape)).reshape(len(rotated), len(owners), 2, 3).copy()
    _rotate_vectors(local[rotating, :, None, 3:], twist)
    return copy_base[rotating] + index[None], twist.reshape(len(rotated), *index.shape)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host construction of tree ownership and compact body response supports."""

import numpy as np

from . import host_index
from .tree_plan import TreePlan, bit_range


def build_body_maps(
    body_count: int,
    joint_child: np.ndarray,
    joint_forest: TreePlan,
    joint_qd_start: np.ndarray,
    articulation_start: np.ndarray,
    articulation_joint_end: np.ndarray,
    articulation_dof_start: np.ndarray,
    response_dof_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ownership, response presence, low-32 support and singleton indices.

    A body's response coordinates are those of its tree joint and the joint's
    ancestors in ``joint_forest`` (the model's joint ancestry) that lie in its
    articulation's response window: their count, their low 32 bits from the
    window's start, and the coordinate itself when there is exactly one.
    """
    body_joint = np.full(body_count, -1, dtype=np.int32)
    body_articulation = np.full(body_count, -1, dtype=np.int32)
    joint_articulation, _ = host_index.expand(np.diff(articulation_start))
    indices = np.arange(len(joint_child), dtype=np.int32)
    tree = (indices < articulation_joint_end[joint_articulation]) & (joint_child >= 0)
    # Preserve the last tree owner.
    np.maximum.at(body_joint, joint_child[tree], indices[tree])
    owned = body_joint >= 0
    body_articulation[owned] = joint_articulation[body_joint[owned]]

    # Each joint's coordinates inside its articulation's response window, accumulated down every root path.
    start = articulation_dof_start[joint_articulation].astype(np.int64)
    end = start + response_dof_count[joint_articulation]
    lower = np.maximum(joint_qd_start[:-1], start)
    upper = np.maximum(lower, np.minimum(joint_qd_start[1:], end))
    width = upper - lower
    count = joint_forest.scan_down(width)
    support = joint_forest.scan_down(
        bit_range(np.clip(lower - start, 0, 32), np.clip(upper - start, 0, 32)), np.bitwise_or
    )
    single = joint_forest.scan_down(np.where(width > 0, lower, -1), np.maximum)

    joint = body_joint[owned]
    dof_count = np.zeros(body_count, dtype=np.int32)
    dof_count[owned] = count[joint]
    body_support = np.zeros(body_count, dtype=np.uint32)
    body_support[owned] = support[joint].astype(np.uint32)
    single_dof = np.full(body_count, -1, dtype=np.int32)
    single_dof[owned] = np.where(count[joint] == 1, single[joint], -1)
    return body_joint, body_articulation, (dof_count > 0).astype(np.int32), body_support, single_dof

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host construction of tree ownership and compact body response supports."""

import numpy as np

from . import host_index


def build_body_maps(
    body_count: int,
    joint_child: np.ndarray,
    joint_ancestor: np.ndarray,
    joint_qd_start: np.ndarray,
    articulation_start: np.ndarray,
    articulation_joint_end: np.ndarray,
    articulation_dof_start: np.ndarray,
    response_dof_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ownership, response presence, low-32 support and singleton indices."""
    body_joint = np.full(body_count, -1, dtype=np.int32)
    body_articulation = np.full(body_count, -1, dtype=np.int32)
    joint_articulation, _ = host_index.expand(np.diff(articulation_start))
    indices = np.arange(len(joint_child), dtype=np.int32)
    tree = (indices < articulation_joint_end[joint_articulation]) & (joint_child >= 0)
    # Preserve the last tree owner without allowing a trailing loop joint to steal it.
    np.maximum.at(body_joint, joint_child[tree], indices[tree])
    owned = body_joint >= 0
    body_articulation[owned] = joint_articulation[body_joint[owned]]

    dof_count = np.zeros(body_count, dtype=np.int32)
    support = np.zeros(body_count, dtype=np.uint32)
    single_dof = np.full(body_count, -1, dtype=np.int32)
    # Bound temporary storage independently of the number of worlds or DOFs.
    for first in range(0, body_count, 65536):
        bodies = np.arange(first, min(first + 65536, body_count))
        bodies = bodies[owned[bodies]]
        current = body_joint[bodies]
        fast = current.copy()
        while len(bodies):
            articulation = body_articulation[bodies]
            start = articulation_dof_start[articulation]
            end = start + response_dof_count[articulation]
            lower = np.maximum(joint_qd_start[current], start)
            upper = np.maximum(lower, np.minimum(joint_qd_start[current + 1], end))
            width = upper - lower
            dof_count[bodies] += width
            single_dof[bodies] = np.where(width > 0, lower, single_dof[bodies])
            lo_bit = np.clip(lower - start, 0, 32).astype(np.uint64)
            hi_bit = np.clip(upper - start, 0, 32).astype(np.uint64)
            bits = np.left_shift(np.uint64(1), hi_bit) - np.left_shift(np.uint64(1), lo_bit)
            support[bodies] |= bits.astype(np.uint32)

            current = joint_ancestor[current]
            # Floyd's check also admits acyclic ancestors stored after their children.
            for _ in range(2):
                live = fast >= 0
                fast[live] = joint_ancestor[fast[live]]
            cyclic = (current >= 0) & (current == fast)
            if np.any(cyclic):
                index = int(np.flatnonzero(cyclic)[0])
                raise ValueError(
                    "SolverFeatherPGS: cyclic joint ancestry while building body response maps "
                    f"(body {bodies[index]}, articulation {articulation[index]}, joint {current[index]})."
                )
            live = current >= 0
            bodies, current, fast = bodies[live], current[live], fast[live]

    single_dof[dof_count != 1] = -1
    return body_joint, body_articulation, (dof_count > 0).astype(np.int32), support, single_dof

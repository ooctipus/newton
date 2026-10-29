# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural layout and unary-prefix generation for compact contact keys."""

from dataclasses import dataclass
from functools import cache

import numpy as np

from ...sim import JointType
from .device_preludes import lane_group
from .kernel_template import WORLDS, KernelTemplate
from .prefix_rows import limits, prefix_rows, velocity_limits
from .row_laws import ROW_LAW


@dataclass(frozen=True)
class KeyedLayout:
    primary: int
    secondary: int
    primary_offset: int
    secondary_offset: int
    parallel_rows: int
    velocity_limits: bool
    velocity_mask: int
    position_mask: int | None = None

    def __post_init__(self):
        d = self.primary + self.secondary
        coordinates = [*range(self.primary_offset, self.primary_offset + self.primary)]
        coordinates += range(self.secondary_offset, self.secondary_offset + self.secondary)
        if self.primary <= 0 or self.secondary < 0 or d > 32 or sorted(coordinates) != list(range(d)):
            raise ValueError("keyed rows require one or two disjoint complete coordinate blocks, at most 32 DOFs")
        if not 0 < self.parallel_rows <= 128 or self.parallel_rows % 4:
            raise ValueError("keyed rows require the existing bounded parallel row policy")
        for mask in (self.velocity_mask, self.position_mask):
            if mask is not None and not 0 <= mask < (1 << d):
                raise ValueError("keyed limit eligibility must lie inside the coordinate layout")

    @property
    def dofs(self):
        return self.primary + self.secondary

    @property
    def position_eligibility(self):
        return (1 << self.dofs) - 1 if self.position_mask is None else self.position_mask

    def primary_test(self, coordinate):
        return f"({coordinate} >= {self.primary_offset} && {coordinate} < {self.primary_offset + self.primary})"

    def local(self, coordinate):
        return f"({self.primary_test(coordinate)} ? {coordinate} - {self.primary_offset} : {coordinate} - {self.secondary_offset})"


def topology_plan(solver):
    """Derive uniform block offsets and supported velocity DOFs from the model: the layout, then each world's group
    index, first coordinate and articulation of each size."""
    sizes = sorted((int(n) for n in solver.size_groups), reverse=True)
    if not 1 <= len(sizes) <= 2 or sum(sizes) != solver.max_world_dofs:
        raise ValueError("keyed rows require at most two complete uniform size groups")
    worlds = solver.art_to_world.numpy()
    offsets = solver.articulation_world_dof_offset.numpy()
    starts = solver.articulation_dof_start.numpy()
    groups = np.zeros((solver.world_count, 2), dtype=np.int32)
    dof_starts = np.zeros_like(groups)
    arts_by_world = np.full_like(groups, -1)
    block_offsets = [0, 0]
    model = solver.model
    art_joint = model.articulation_start.numpy()
    joint_dof = model.joint_qd_start.numpy()
    joint_type = model.joint_type.numpy()
    velocity_mask, position_mask = 0, 0
    q_index = solver._joint_limit_q_index_host
    for column, size in enumerate(sizes):
        arts = solver.group_to_art[size].numpy()
        if len(arts) != solver.world_count or not np.array_equal(np.sort(worlds[arts]), np.arange(solver.world_count)):
            raise ValueError("keyed rows require exactly one articulation of each size per world")
        if np.any(offsets[arts] != offsets[arts[0]]):
            raise ValueError("keyed rows require uniform coordinate offsets")
        block_offsets[column] = int(offsets[arts[0]])
        groups[worlds[arts], column] = np.arange(len(arts), dtype=np.int32)
        dof_starts[worlds[arts], column] = starts[arts]
        arts_by_world[worlds[arts], column] = arts
        masks, position_masks = set(), set()
        for art in arts:
            position_masks.add(
                sum(
                    1 << (int(d) + block_offsets[column])
                    for d in np.flatnonzero(q_index[starts[art] : starts[art] + size] >= 0)
                )
            )
            mask = 0
            for joint in range(art_joint[art], art_joint[art + 1]):
                if joint_type[joint] in (JointType.PRISMATIC, JointType.REVOLUTE, JointType.D6):
                    for dof in range(joint_dof[joint], joint_dof[joint + 1]):
                        mask |= 1 << (int(dof - starts[art]) + block_offsets[column])
                elif solver.enable_joint_velocity_limits and joint_type[joint] not in (JointType.FIXED, JointType.FREE):
                    raise ValueError("unsupported moving joint velocity-limit law")
            masks.add(mask)
        if len(masks) != 1:
            raise ValueError("keyed rows require uniform velocity-limit eligibility")
        if len(position_masks) != 1:
            raise ValueError("keyed rows require uniform position-limit eligibility")
        velocity_mask |= masks.pop()
        position_mask |= position_masks.pop()
    layout = KeyedLayout(
        sizes[0],
        sizes[1] if len(sizes) == 2 else 0,
        *block_offsets,
        solver.mf_gs_parallel_rows,
        solver.enable_joint_velocity_limits,
        velocity_mask,
        position_mask,
    )
    return layout, groups, dof_starts, arts_by_world


@cache
def prefix_kernel(layout, prefix_type, rows_type):
    """Generate a world's ordered unary rows for its eligible coordinates (:func:`.prefix_rows.prefix_rows`).

    A row's key ``~(2 d + side)`` names its coordinate ``d`` and side; a velocity-limit row also records its slot and
    sign per DOF side.
    """
    dof = f"p.starts.data[world * 2 + ({layout.primary_test('d')} ? 0 : 1)] + {layout.local('d')}"
    velocities = layout.velocity_mask if layout.velocity_limits else 0
    sink = "            z.rowkeys.data[at] = ~(d0 * 2 + side);"
    clear = ""
    if velocities:
        sink += """
            if (type == 4) {
                p.velocity_slots.data[dof * 2 + side] = slot;
                p.velocity_signs.data[dof * 2 + side] = w0;
            }"""
        clear = f"""    for (int k = LANE; k < {2 * layout.dofs}; k += STRIDE) {{
        const int d = k / 2;
        const int dof = {dof};
        p.velocity_slots.data[dof * 2 + k % 2] = -1;
        p.velocity_signs.data[dof * 2 + k % 2] = 0.0f;
    }}
"""
    positions, velocities = (
        tuple(d for d in range(layout.dofs) if mask & (1 << d)) for mask in (layout.position_eligibility, velocities)
    )
    source = clear + prefix_rows(
        ((limits(positions, dof, "p."),), (velocity_limits(velocities, dof, "p."),) if velocities else ()),
        law="p.",
        sink=sink,
        ordered=True,
        rhs=True,
    )
    # A warp generates a world's rows on CUDA, one thread on the CPU.
    return KernelTemplate(
        "keyed_limit_prefix",
        (("p", prefix_type), ("z", rows_type)),
        source,
        WORLDS,
        preludes=(lane_group(32), *ROW_LAW.preludes()),
        dual=True,
    ).kernel()

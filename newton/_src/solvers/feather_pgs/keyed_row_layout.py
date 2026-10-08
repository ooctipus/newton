# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural layout and unary-prefix generation for compact contact keys."""

from dataclasses import dataclass
from functools import cache

import numpy as np
import warp as wp

from ...sim import JointType


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

    @property
    def limit_coordinates(self):
        mask = self.position_eligibility | (self.velocity_mask if self.velocity_limits else 0)
        return tuple(d for d in range(self.dofs) if mask & (1 << d))

    def primary_test(self, coordinate):
        return f"({coordinate} >= {self.primary_offset} && {coordinate} < {self.primary_offset + self.primary})"

    def local(self, coordinate):
        return f"({self.primary_test(coordinate)} ? {coordinate} - {self.primary_offset} : {coordinate} - {self.secondary_offset})"


def topology_plan(solver):
    """Derive uniform block offsets and supported velocity DOFs from the model."""
    sizes = sorted((int(n) for n in solver.size_groups), reverse=True)
    if not 1 <= len(sizes) <= 2 or sum(sizes) != solver.max_world_dofs:
        raise ValueError("keyed rows require at most two complete uniform size groups")
    worlds = solver.art_to_world.numpy()
    offsets = solver.articulation_world_dof_offset.numpy()
    starts = solver.articulation_dof_start.numpy()
    groups = np.zeros((solver.world_count, 2), dtype=np.int32)
    dof_starts = np.zeros_like(groups)
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
    return layout, groups, dof_starts


@cache
def prefix_kernel(layout, prefix_type, rows_type):
    """Generate ordered unary rows for eligible coordinates, using the existing laws."""
    source = r"""
    const int M = p.kind.shape[1];
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x;
#endif
#if $VELOCITY
#if defined(__CUDA_ARCH__)
    for (int k = lane; k < $DOUBLE; k += 32) {
#else
    for (int k = 0; k < $DOUBLE; ++k) {
#endif
        const int d = k / 2;
        const int dof = p.starts.data[world * 2 + ($PRIMARY ? 0 : 1)] + $LOCAL;
        p.velocity_slots.data[dof * 2 + k % 2] = -1;
        p.velocity_signs.data[dof * 2 + k % 2] = 0.0f;
    }
#endif
    int base = 0;
    const int coordinates[] = {$COORDINATES};
    for (int phase = 0; phase < 2; ++phase) {
        for (int chunk = 0; chunk < $CANDIDATES; chunk += 32) {
#if !defined(__CUDA_ARCH__)
            int phase_count = 0;
            for (int lane = 0; lane < 32; ++lane) {
#endif
            const int candidate = chunk + lane, side = candidate & 1;
            const int d = candidate < $CANDIDATES ? coordinates[candidate / 2] : 0;
            const int dof = p.starts.data[world * 2 + ($PRIMARY ? 0 : 1)] + $LOCAL;
            const float sign = side == 0 ? 1.0f : -1.0f;
            bool active = false;
            float phi = 0.0f, target = 0.0f;
            if (candidate < $CANDIDATES) {
                if (phase == 0 && ($POSITION & (1u << d)) != 0u) {
                    const int qi = p.q_index.data[dof];
                    if (qi >= 0) {
                        const float bound = side == 0 ? p.lower.data[dof] : p.upper.data[dof];
                        const float q = p.q.data[qi];
                        phi = side == 0 ? q - bound : bound - q;
                        active = isfinite(bound) && (side == 0 ? q <= bound + p.gap : q >= bound - p.gap);
                    }
                }
#if $VELOCITY
                else if (phase == 1 && ($MASK & (1u << d)) != 0u) {
                    const float limit = p.velocity_limit.data[dof];
                    active = !(limit <= 0.0f) && !(p.activation > 0.0f && fabsf(p.vhat.data[dof]) < p.activation * limit);
                    target = -limit;
                }
#endif
            }
#if defined(__CUDA_ARCH__)
            const unsigned mask = __ballot_sync(0xffffffffu, active);
            const int phase_count = __popc(mask);
            const int rank = __popc(mask & (lane == 0 ? 0u : ((1u << lane) - 1u)));
#else
            const int rank = phase_count;
            phase_count += active;
#endif
            const int slot = base + rank;
            if (active && slot < M) {
                const int at = world * M + slot;
                z.rowkeys.data[at] = ~(d * 2 + side);
                p.kind.data[at] = phase == 0 ? 3 : 4;
                p.parent.data[at] = -1; p.mu.data[at] = 0.0f;
                p.beta.data[at] = phase == 0 ? p.pgs_beta : 0.0f;
                p.cfm.data[at] = p.pgs_cfm;
                p.phi.data[at] = phi; p.target.data[at] = target;
                p.restitution.data[at] = 0.0f;
                p.rhs.data[at] = phase == 1 ? -target : ((phi < 0.0f ? p.pgs_beta : 1.0f) * phi / p.dt);
#if $VELOCITY
                if (phase == 1) {
                    p.velocity_slots.data[dof * 2 + side] = slot;
                    p.velocity_signs.data[dof * 2 + side] = sign;
                }
#endif
            }
#if !defined(__CUDA_ARCH__)
            }
#endif
            base += phase_count;
        }
#if defined(__CUDA_ARCH__)
        if (lane == 0)
#endif
        p.bounds.data[world * 2 + phase] = base;
    }
#if defined(__CUDA_ARCH__)
    if (lane == 0)
#endif
    p.counts.data[world] = base;
"""
    for key, value in {
        "$VELOCITY": int(layout.velocity_limits),
        "$DOUBLE": 2 * layout.dofs,
        "$CANDIDATES": 2 * len(layout.limit_coordinates),
        "$COORDINATES": ", ".join(map(str, layout.limit_coordinates)) or "0",
        "$POSITION": f"{layout.position_eligibility}u",
        "$PRIMARY": layout.primary_test("d"),
        "$LOCAL": layout.local("d"),
        "$MASK": f"{layout.velocity_mask}u",
    }.items():
        source = source.replace(key, str(value))

    @wp.func_native(source)
    def native(world: int, logical_lane: int, p: prefix_type, z: rows_type): ...

    @wp.kernel(module="unique", enable_backward=False)
    def keyed_limit_prefix(p: prefix_type, z: rows_type):
        world, lane = wp.tid()
        native(world, lane, p, z)

    return keyed_limit_prefix

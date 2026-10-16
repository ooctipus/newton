# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural admission for complete uniform cached articulation trees."""

import os

import numpy as np

from ...sim import BodyFlags, JointType
from .sparse_mass_matrix import _SparseMassMatrixPlan
from .tree_kinetic_layout import TreeLayout


def cooperative_block_dim(layout, chain_plan=None):
    """Cover all one-body lanes and occupied packed-chain lanes."""
    if layout.bodies > 32 or (chain_plan is not None and np.any(chain_plan["meta"][32:, 0] >= 0)):
        return 64
    return 32


_MIN_BLOCKS = 8192


def lanes_for(bodies, groups, device):
    """Threads per tree, so small trees share a 32-thread block.

    The lane count is the smallest power of two covering the bodies (at least
    four), but the pack is also bounded so that at least ``_MIN_BLOCKS`` blocks
    remain: these kernels are latency-bound, and eight one-body trees per
    block left 2048 blocks on a 16k-world scene, slower than two per block.
    The group count must divide into whole blocks; the pack halves until it
    does. ``FEATHER_PGS_TREE_LANE_PACK=0`` keeps one tree per block.
    """
    if bodies > 32 or not device.is_cuda or os.environ.get("FEATHER_PGS_TREE_LANE_PACK", "1") != "1":
        return 0
    lanes = max(4, 1 << (bodies - 1).bit_length())
    while lanes < 32 and (groups % (32 // lanes) or groups // (32 // lanes) < _MIN_BLOCKS):
        lanes *= 2
    return lanes


def make_plan(model, operator, *, complete=True):
    """Check the tree, scalar laws and packed operator before allocating caches."""
    worlds, arts = int(model.world_count), int(model.articulation_count)
    if worlds < 1 or arts < 1 or model.joint_count != model.body_count:
        raise ValueError("Cached trees require complete articulation and body ownership")
    if model.requires_grad or model.particle_count:
        raise ValueError("Cached trees exclude gradients and particles")
    starts = model.articulation_start.numpy()
    group = operator.plan.group_to_art.numpy()
    art_world = operator.plan.art_to_world.numpy()
    if (
        group.ndim != 1
        or not len(group)
        or np.any(group < 0)
        or np.any(group >= arts)
        or len(np.unique(group)) != len(group)
    ):
        raise ValueError("Cached tree groups require distinct valid articulation IDs")
    if complete and not np.array_equal(np.sort(group), np.arange(arts)):
        raise ValueError("Cached tree groups must cover every articulation exactly once")
    counts_by_art = np.diff(starts)
    bodies = int(counts_by_art[group[0]])
    if not 1 <= bodies <= 64 or not np.all(counts_by_art[group] == bodies):
        raise ValueError("Cached tree groups require uniform body counts up to 64")
    joints = starts[group, None] + np.arange(bodies)
    qd = model.joint_qd_start.numpy()
    counts = np.diff(qd)[joints]
    dofs = int(counts[0].sum())
    if not 1 <= dofs <= 64:
        raise ValueError("Cached trees require at most 64 coordinates")
    if art_world.shape != (arts,) or np.any(art_world < 0) or np.any(art_world >= worlds):
        raise ValueError("Cached tree articulations require valid world ownership")
    child = model.joint_child.numpy()
    if not np.array_equal(np.sort(child), np.arange(model.body_count)):
        raise ValueError("Cached tree publication requires every body exactly once")
    ancestors = model.joint_ancestor.numpy()[joints]
    local = np.where(ancestors >= 0, ancestors - starts[group, None], -1)
    parent = local[0].astype(np.int32)
    if (
        not np.all(local == parent)
        or parent[0] != -1
        or np.any(parent[1:] < 0)
        or np.any(parent[1:] >= np.arange(1, bodies))
    ):
        raise ValueError("Cached trees require one uniform parent-first topology")
    types = model.joint_type.numpy()[joints]
    locked_root = types[0, 0] == int(JointType.D6) and np.all(counts[:, 0] == 0)
    if not np.all(types == types[0]) or (
        types[0, 0] not in (int(JointType.FREE), int(JointType.FIXED)) and not locked_root
    ):
        raise ValueError("Cached trees require a free, fixed or zero-coordinate D6 root")
    if not np.all(np.isin(types[:, 1:], [int(JointType.FIXED), int(JointType.REVOLUTE), int(JointType.PRISMATIC)])):
        raise ValueError("Cached trees require scalar or fixed descendant joints")
    if not np.all(counts == counts[0]) or not np.all(
        qd[joints] - qd[starts[group, None]] == qd[joints[0]] - qd[starts[group[0]]]
    ):
        raise ValueError("Cached trees require uniform coordinate indexing")
    # Kinematic bodies: a prescribed group owns whole free bodies whose six
    # coordinates are all kinematic; any other group must be fully dynamic.
    # (A welded kinematic base of a fixed-root tree carries no coordinate.)
    kinematic = model.body_flags.numpy()[child[joints]] & int(BodyFlags.KINEMATIC)
    coordinate_bodies = counts > 0
    prescribed = bool(getattr(operator, "prescribed", False))
    if prescribed:
        if bodies != 1 or dofs != 6 or not np.all(kinematic[:, 0]):
            raise ValueError("Prescribed cached roots must be single kinematic free bodies")
    elif np.any(kinematic & coordinate_bodies):
        raise ValueError("Cached trees exclude kinematic bodies with coordinates")
    mass = _SparseMassMatrixPlan.build(parent, counts[0])
    if not np.array_equal(mass.lookup, operator.host["index"]) or not np.array_equal(
        mass.dof_joint, operator.host["dof_joint"]
    ):
        raise ValueError("Cached tree and held operator use different coordinates")
    depth = np.zeros(bodies, np.int32)
    for body in range(1, bodies):
        depth[body] = depth[parent[body]] + 1
    layout = TreeLayout(
        bodies,
        dofs,
        mass.nonzero_count,
        operator.host["inverse_nodes"].shape[1],
        int(depth.max()),
        int(counts[0, 0]),
        tuple(map(int, mass.permutation)),
        lanes_for(bodies, len(group), model.device),
    )
    return layout, parent, depth

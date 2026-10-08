# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural admission for complete uniform cached articulation trees."""

import numpy as np

from ...sim import BodyFlags, JointType
from .sparse_mass_matrix import _SparseMassMatrixPlan
from .tree_kinetic_layout import TreeLayout


def cooperative_block_dim(layout, chain_plan=None):
    """Cover all one-body lanes and occupied packed-chain lanes."""
    if layout.bodies > 32 or (chain_plan is not None and np.any(chain_plan["meta"][32:, 0] >= 0)):
        return 64
    return 32


def make_plan(model, operator):
    """Check the tree, scalar laws and packed operator before allocating caches."""
    worlds = int(model.world_count)
    if worlds < 1 or model.articulation_count != worlds or model.joint_count != model.body_count:
        raise ValueError("Cached trees require one complete articulation per world")
    if model.joint_count % worlds or model.joint_dof_count % worlds:
        raise ValueError("Cached trees require uniform body and coordinate counts")
    bodies, dofs = model.joint_count // worlds, model.joint_dof_count // worlds
    if not 1 <= bodies <= 64 or not 1 <= dofs <= 64:
        raise ValueError("Cached trees require at most 64 bodies and coordinates")
    if model.requires_grad or model.particle_count or np.any(model.body_flags.numpy() & int(BodyFlags.KINEMATIC)):
        raise ValueError("Cached trees exclude gradients, particles and prescribed bodies")
    starts = model.articulation_start.numpy()
    if not np.array_equal(starts, np.arange(worlds + 1) * bodies):
        raise ValueError("Cached trees require contiguous uniform articulation joints")
    group = operator.plan.group_to_art.numpy()
    if not np.array_equal(group, np.arange(worlds)) or not np.array_equal(
        operator.plan.art_to_world.numpy(), np.arange(worlds)
    ):
        raise ValueError("Cached tree groups must follow their complete world order")
    child = model.joint_child.numpy()
    if not np.array_equal(np.sort(child), np.arange(model.body_count)):
        raise ValueError("Cached tree publication requires every body exactly once")
    ancestors = model.joint_ancestor.numpy().reshape(worlds, bodies)
    local = np.where(ancestors >= 0, ancestors - np.arange(worlds)[:, None] * bodies, -1)
    parent = local[0].astype(np.int32)
    if (
        not np.all(local == parent)
        or parent[0] != -1
        or np.any(parent[1:] < 0)
        or np.any(parent[1:] >= np.arange(1, bodies))
    ):
        raise ValueError("Cached trees require one uniform parent-first topology")
    types = model.joint_type.numpy().reshape(worlds, bodies)
    if not np.all(types == types[0]) or types[0, 0] not in (int(JointType.FREE), int(JointType.FIXED)):
        raise ValueError("Cached trees require a free or fixed root")
    if not np.all(np.isin(types[:, 1:], [int(JointType.FIXED), int(JointType.REVOLUTE), int(JointType.PRISMATIC)])):
        raise ValueError("Cached trees require scalar or fixed descendant joints")
    qd = model.joint_qd_start.numpy()
    counts = np.diff(qd).reshape(worlds, bodies)
    if not np.all(counts == counts[0]) or not np.all(
        qd[:-1].reshape(worlds, bodies) - np.arange(worlds)[:, None] * dofs == qd[:bodies]
    ):
        raise ValueError("Cached trees require uniform coordinate indexing")
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
    )
    return layout, parent, depth

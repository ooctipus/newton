# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural admission for complete uniform cached articulation trees."""

import numpy as np

from ...sim import BodyFlags, JointType
from .tree_kinetic_layout import TreeLayout
from .tree_plan import TreePlan


def make_plan(model, operator, *, complete=True):
    """Check the tree, scalar laws and packed operator before allocating caches.

    Returns the group's layout, its lanes left to the owner's launch rule, and its template tree's parents.
    """
    worlds, arts = int(model.world_count), int(model.articulation_count)
    if worlds < 1 or arts < 1 or model.joint_count != model.body_count:
        raise ValueError("Cached trees require complete articulation and body ownership")
    if model.particle_count:
        raise ValueError("Cached trees exclude particles")
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
    # A forest group owns several consecutive articulations of one world; the
    # operator names them and the group map holds the first of each.
    forest = getattr(operator, "forest_arts", None)
    forest = None if forest is None else np.asarray(forest, np.int64)
    if forest is not None:
        if forest.ndim != 2 or forest.shape[0] != len(group) or not np.array_equal(forest[:, 0], group):
            raise ValueError("Cached forest groups must start at their group articulation")
        if np.any(starts[forest[:, 1:]] != starts[forest[:, :-1] + 1]):
            raise ValueError("Cached forest articulations must be consecutive")
        bodies = int(np.sum(counts_by_art[forest[0]]))
        if np.any(np.sum(counts_by_art[forest], axis=1) != bodies) or np.any(
            counts_by_art[forest] != counts_by_art[forest[0]][None, :]
        ):
            raise ValueError("Cached forest groups require uniform tree sizes")
        root_locals = tuple(int(v) for v in np.concatenate(([0], np.cumsum(counts_by_art[forest[0]])[:-1])))
    else:
        bodies = int(counts_by_art[group[0]])
        root_locals = (0,)
    if not 1 <= bodies <= 64 or not np.all(counts_by_art[group] == counts_by_art[group[0]]):
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
    is_root = np.zeros(bodies, bool)
    is_root[list(root_locals)] = True
    if (
        not np.all(local == parent)
        or np.any(parent[is_root] != -1)
        or np.any(parent[~is_root] < 0)
        or np.any(parent[1:] >= np.arange(1, bodies))
    ):
        raise ValueError("Cached trees require one uniform parent-first topology")
    types = model.joint_type.numpy()[joints]
    if not np.all(types == types[0]):
        raise ValueError("Cached trees require one uniform parent-first topology")
    root_coordinates = []
    for root in root_locals:
        locked_root = types[0, root] == int(JointType.D6) and np.all(counts[:, root] == 0)
        if types[0, root] not in (int(JointType.FREE), int(JointType.FIXED)) and not locked_root:
            raise ValueError("Cached trees require a free, fixed or zero-coordinate D6 root")
        root_coordinates.append(int(counts[0, root]))
    non_root = np.ones(bodies, bool)
    non_root[list(root_locals)] = False
    # Descendants may carry one or several coordinates: the motion scan stores
    # one axis per coordinate through jcalc_motion and the integrator handles
    # every joint type; the mass plan takes per-body coordinate counts.
    descendants = (
        int(JointType.FIXED),
        int(JointType.REVOLUTE),
        int(JointType.PRISMATIC),
        int(JointType.D6),
        int(JointType.BALL),
    )
    if not np.all(np.isin(types[:, non_root], descendants)):
        raise ValueError("Cached trees require fixed, scalar, D6 or ball descendant joints")
    linear, angular = model.joint_dof_dim.numpy()[joints[0]].T
    own_bias = bool(np.any((types[0] == int(JointType.D6)) & (((linear > 0) & (angular > 0)) | (angular > 1))))
    if not np.all(counts == counts[0]) or not np.all(
        qd[joints] - qd[starts[group, None]] == qd[joints[0]] - qd[starts[group[0]]]
    ):
        raise ValueError("Cached trees require uniform coordinate indexing")
    # Kinematic bodies. A prescribed group owns whole free bodies whose six
    # coordinates are all kinematic, and a forest's zero family a kinematic
    # free root; neither holds a factor. Held groups serve kinematic
    # coordinate bodies as ordinary dynamics does, also after a runtime flag
    # change: the kinematic armature enters the held factor, the live
    # kinematic masks guard the free-root transport and the finish, and the
    # predictor drops a kinematic body's external wrench.
    kinematic = model.body_flags.numpy()[child[joints]] & int(BodyFlags.KINEMATIC)
    prescribed = bool(getattr(operator, "prescribed", False))
    if prescribed:
        if bodies != 1 or dofs != 6 or not np.all(kinematic[:, 0]):
            raise ValueError("Prescribed cached roots must be single kinematic free bodies")
    elif forest is not None:
        families = np.asarray(operator.root_families, np.int64)
        for r, root in enumerate(root_locals):
            if families[r] == 0 and not np.all(kinematic[:, root]):
                raise ValueError("A zero-family forest root must be a kinematic free body")
            # Construction places the forest's held families on dynamic roots
            # only. A root made kinematic later keeps the built forest, which
            # serves it as above, although a new plan would refuse it here.
            if families[r] != 0 and np.any(kinematic[:, root]):
                raise ValueError("A dynamic forest root must not be kinematic")
    tree = TreePlan.build(parent, counts[0])
    mass = tree.factor
    if not np.array_equal(mass.lookup, operator.host["index"]) or not np.array_equal(
        mass.dof_joint, operator.host["dof_joint"]
    ):
        raise ValueError("Cached tree and held operator use different coordinates")
    offsets = tuple(int(v) for v in np.cumsum(np.concatenate(([0], counts[0])))[list(root_locals)])
    layout = TreeLayout(
        bodies,
        dofs,
        mass.nonzero_count,
        operator.host["inverse_nodes"].shape[1],
        tree.max_depth,
        int(counts[0, 0]),
        tuple(map(int, mass.permutation)),
        roots=root_locals,
        root_coordinates=tuple(root_coordinates),
        root_offsets=offsets,
        root_families=tuple(int(v) for v in getattr(operator, "root_families", ())) or (dofs,),
        own_bias=own_bias,
    )
    return layout, parent

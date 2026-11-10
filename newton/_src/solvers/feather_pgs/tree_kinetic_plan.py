# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural admission for complete uniform cached articulation trees."""

import numpy as np

from ...sim import BodyFlags, JointType
from .tree_kinetic_layout import TreeLayout
from .tree_plan import TreePlan

FREE_JOINTS = (int(JointType.FREE), int(JointType.DISTANCE))
"""The joints a root stands on with the free-root transport (the forest plan's root rule)."""


def tree_joints(model):
    """The joints of the articulations' trees (a mask) and their count per articulation: a joint outside every
    articulation stays outside the plan."""
    articulation = model.joint_articulation.numpy()
    return articulation >= 0, np.bincount(articulation[articulation >= 0], minlength=model.articulation_count)


def make_plan(model, operator, *, complete=True):
    """The layout of a held group (its lanes left to the owner's launch rule) and its template tree's parents. The plan
    groups trees of one topology on the template's pattern; what their data can still refuse is checked: coverage, one
    owner a body, uniform parents and joint kinds (a sparse size group, a forest's rest trees), the kinematic roots."""
    owned, counts_by_art = tree_joints(model)
    starts = model.articulation_start.numpy()
    group = operator.plan.group_to_art.numpy()
    if complete and not np.array_equal(np.sort(group), np.arange(model.articulation_count)):
        raise ValueError("Cached tree groups must cover every articulation exactly once")
    # A forest group owns several consecutive articulations of one world; the
    # operator names them and the group map holds the first of each.
    forest = getattr(operator, "forest_arts", None)
    forest = None if forest is None else np.asarray(forest, np.int64)
    if forest is not None:
        if forest.ndim != 2 or forest.shape[0] != len(group) or not np.array_equal(forest[:, 0], group):
            raise ValueError("Cached forest groups must start at their group articulation")
        if np.any(starts[forest[:, 1:]] != starts[forest[:, :-1]] + counts_by_art[forest[:, :-1]]):
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
    if bodies < 1 or not np.all(counts_by_art[group] == counts_by_art[group[0]]):
        raise ValueError("Cached tree groups require uniform body counts")
    joints = starts[group, None] + np.arange(bodies)
    counts = np.diff(model.joint_qd_start.numpy())[joints]
    dofs = int(counts[0].sum())
    child = model.joint_child.numpy()
    if np.unique(child[owned]).size != np.count_nonzero(owned):
        raise ValueError("Cached tree publication requires each body at most once")
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
    # Every joint may carry one or several coordinates: the motion scan stores
    # one axis per coordinate through joint_motion and the integrator handles
    # every joint type; the mass plan takes per-body coordinate counts. A root
    # stands on a free joint, on no coordinates (fixed or a locked D6), or on
    # a joint with coordinates that the stages treat as a descendant's.
    root_coordinates = [int(counts[0, root]) for root in root_locals]
    free_roots = [int(types[0, root]) in FREE_JOINTS for root in root_locals]
    linear, angular = model.joint_dof_dim.numpy()[joints[0]].T
    own_bias = bool(np.any((types[0] == int(JointType.D6)) & (((linear > 0) & (angular > 0)) | (angular > 1))))
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
        tree_ends = (*root_locals[1:], bodies)
        for r, root in enumerate(root_locals):
            if families[r] == 0 and np.any(counts[0, root : tree_ends[r]]) and not np.all(kinematic[:, root]):
                raise ValueError("A zero-family forest tree must be a kinematic free body or hold no coordinates")
            # Construction places the forest's held families on dynamic roots
            # only. A root made kinematic later keeps the built forest, which
            # serves it as above, although a new plan would refuse it here.
            if families[r] != 0 and np.any(kinematic[:, root]):
                raise ValueError("A dynamic forest root must not be kinematic")
    tree = TreePlan.build(parent, counts[0])
    mass = tree.factor
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
        free_roots=tuple(free_roots),
        own_bias=own_bias,
    )
    return layout, parent

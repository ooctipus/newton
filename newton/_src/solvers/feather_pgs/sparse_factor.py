# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed lower-factor plan and values shared by the cached tree owners.

``SparsePlan`` binds a size group's packed factor pattern, its coordinate
permutations and articulation ownership; ``SparseData`` holds the packed factor
values. The sparse-route and dense-route tree caches (``tree_kinetic_sparse``,
``mixed_tree_kinetic``) fill them through :func:`operator_metadata` and
:func:`bind_operator_plan`. The structs keep the layout the retired packed
sparse-factor route also used, so the held-tree kernels that take them keep
their generated code; fields those owners do not bind stay empty.
"""

import numpy as np
import warp as wp


@wp.struct
class SparsePlan:
    index: wp.array2d[int]
    row: wp.array[int]
    col: wp.array[int]
    source: wp.array[int]
    level_offsets: wp.array2d[int]
    level_pivots: wp.array[int]
    level_scales: wp.array[int]
    level_entries: wp.array[int]
    level_term_offsets: wp.array[int]
    level_terms: wp.array2d[int]
    inverse_nodes: wp.array2d[int]
    inverse_count: wp.array[int]
    support_nodes: wp.array2d[int]
    support_count: wp.array[int]
    limit_support: wp.array[int]
    pair_support: wp.array2d[int]
    body_mask: wp.array[wp.uint64]
    body_tag: wp.array[int]
    dof_joint: wp.array[int]
    physical: wp.array[int]
    sparse: wp.array[int]
    group_to_art: wp.array[int]
    art_to_world: wp.array[int]
    art_dof_start: wp.array[int]
    art_joint_start: wp.array[int]
    joint_child: wp.array[int]


@wp.struct
class SparseData:
    W: wp.array2d[float]
    valid: wp.array[int]
    status: wp.array[int]
    Z: wp.array3d[float]
    support: wp.array2d[int]
    incident: wp.array2d[float]


def operator_metadata(mass, *, inverse_column_width=None):
    """Describe topology-derived factor coordinates for every cached tree adapter."""
    counts = np.diff(mass.column_offsets).astype(np.int32)
    width = int(counts.max()) if inverse_column_width is None else inverse_column_width
    if width < int(counts.max()):
        raise ValueError("Inverse-column storage does not cover the factor topology")
    nodes = np.full((mass.dof_count, width), -1, np.int32)
    for col in range(mass.dof_count):
        begin, end = mass.column_offsets[col : col + 2]
        nodes[col, : end - begin] = mass.entry_rows[mass.column_entries[begin:end]]
    return {
        "index": mass.lookup,
        "row": mass.entry_rows,
        "col": mass.columns,
        "source": mass.permutation[mass.columns],
        "inverse_nodes": nodes,
        "inverse_count": counts,
        "dof_joint": mass.dof_joint,
        "physical": mass.permutation,
        "sparse": mass.inverse_permutation,
    }


def bind_operator_plan(solver, size, host, *, group_to_art=None):
    """Bind factor metadata and articulation ownership without choosing a row law.

    ``group_to_art`` names the owned articulations explicitly; by default they
    are the solver's factor group of ``size`` coordinates.
    """
    model = solver.model
    plan = SparsePlan()
    for name, values in host.items():
        setattr(plan, name, wp.array(values, dtype=wp.uint64 if name == "body_mask" else int, device=model.device))
    plan.group_to_art = solver.group_to_art[size] if group_to_art is None else group_to_art
    plan.art_to_world = solver.art_to_world
    plan.art_dof_start = solver.articulation_dof_start
    plan.art_joint_start = model.articulation_start
    plan.joint_child = model.joint_child
    return plan

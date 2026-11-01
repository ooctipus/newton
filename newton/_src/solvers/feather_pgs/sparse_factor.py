# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed lower-factor plan and values shared by the cached tree owners.

``SparsePlan`` binds a size group's packed factor pattern, its coordinate
permutations and articulation ownership; ``SparseData`` holds the packed factor
values. The held tree's sparse and dense operators (:mod:`.held_tree`) fill them
through :func:`operator_metadata` and :func:`bind_operator_plan`. The structs keep the layout the retired packed
sparse-factor route also used, so the held-tree kernels that take them keep
their generated code; fields those owners do not bind stay empty.
"""

import numpy as np
import warp as wp

from . import field_group, host_index
from .field_group import Field
from .sparse_mass_matrix import _SparseMassMatrixIndices

SPARSE_PLAN = (
    Field("index", wp.int32, ("i", "j"), owner="host"),
    Field("row", wp.int32, ("i",), owner="host"),
    Field("col", wp.int32, ("i",), owner="host"),
    Field("source", wp.int32, ("i",), owner="host"),
    Field("level_offsets", wp.int32, ("i", "j"), owner="host"),
    Field("level_pivots", wp.int32, ("i",), owner="host"),
    Field("level_scales", wp.int32, ("i",), owner="host"),
    Field("level_entries", wp.int32, ("i",), owner="host"),
    Field("level_term_offsets", wp.int32, ("i",), owner="host"),
    Field("level_terms", wp.int32, ("i", "j"), owner="host"),
    Field("inverse_nodes", wp.int32, ("i", "j"), owner="host"),
    Field("inverse_count", wp.int32, ("i",), owner="host"),
    Field("support_nodes", wp.int32, ("i", "j"), owner="host"),
    Field("support_count", wp.int32, ("i",), owner="host"),
    Field("limit_support", wp.int32, ("i",), owner="host"),
    Field("pair_support", wp.int32, ("i", "j"), owner="host"),
    Field("body_mask", wp.uint64, ("i",), owner="host"),
    Field("body_tag", wp.int32, ("i",), owner="host"),
    Field("dof_joint", wp.int32, ("i",), owner="host"),
    Field("physical", wp.int32, ("i",), owner="host"),
    Field("sparse", wp.int32, ("i",), owner="host"),
    Field("group_to_art", wp.int32, ("art",), owner="step"),
    Field("art_to_world", wp.int32, ("articulation",), owner="forest"),
    Field("art_dof_start", wp.int32, ("articulation",), owner="forest", attr="articulation_dof_start"),
    Field("art_joint_start", wp.int32, ("articulation_bound",), owner="model", attr="articulation_start"),
    Field("joint_child", wp.int32, ("joint",), owner="model"),
)
"""A size group's packed factor pattern (host metadata), coordinate permutations and articulation ownership."""

SparsePlan = field_group.struct("SparsePlan", SPARSE_PLAN, __name__)


@wp.struct
class SparseData:
    W: wp.array2d[float]
    W_trial: wp.array2d[float]
    valid: wp.array[int]
    status: wp.array[int]
    armature: wp.array2d[float]
    indices: _SparseMassMatrixIndices
    Z: wp.array3d[float]
    support: wp.array2d[int]
    incident: wp.array2d[float]


def operator_metadata(mass, *, inverse_column_width=None):
    """Describe topology-derived factor coordinates for every cached tree adapter."""
    counts = np.diff(mass.column_offsets).astype(np.int32)
    width = int(counts.max(initial=0)) if inverse_column_width is None else inverse_column_width
    if width < int(counts.max(initial=0)):
        raise ValueError("Inverse-column storage does not cover the factor topology")
    nodes = np.full((mass.dof_count, width), -1, np.int32)
    col, rank = host_index.expand(counts)
    nodes[col, rank] = mass.entry_rows[mass.column_entries]
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
    return field_group.upload(
        SparsePlan,
        model.device,
        **host,
        group_to_art=solver.forest.group_to_art[size] if group_to_art is None else group_to_art,
        art_to_world=solver.forest.art_to_world,
        art_dof_start=solver.forest.articulation_dof_start,
        art_joint_start=model.articulation_start,
        joint_child=model.joint_child,
    )

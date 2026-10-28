# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Topology-derived, leaf-first mass factors for small articulated forests.

The sparse coordinate ``i`` names physical DOF ``permutation[i]``. With that
permutation, ``P H P.T = L L.T`` and ``Linv = inverse(L)`` have the same lower
ancestor pattern. Neither assembly nor factorization allocates a dense H.
"""

import warp as wp

from . import field_group
from .field_group import Field
from .tree_plan import FactorPattern, SupportClasses

SPARSE_MASS_INDICES = (
    Field("dof_count", wp.int32),
    Field("nonzero_count", wp.int32),
    Field("factor_level_count", wp.int32),
    Field("permutation", wp.int32, ("dof",)),
    Field("inverse_permutation", wp.int32, ("dof",)),
    Field("row_offsets", wp.int32, ("dof_bound",)),
    Field("columns", wp.int32, ("entry",)),
    Field("entry_rows", wp.int32, ("entry",)),
    Field("lookup", wp.int32, ("dof", "dof")),
    Field("dof_joint", wp.int32, ("dof",)),
    Field("ancestor_mask", wp.uint64, ("dof",)),
    Field("column_offsets", wp.int32, ("dof_bound",)),
    Field("column_entries", wp.int32, ("entry",)),
    Field("factor_level_offsets", wp.int32, ("level_bound",)),
    Field("factor_level_columns", wp.int32, ("dof",)),
    Field("factor_level_entry_offsets", wp.int32, ("level_bound",)),
    Field("factor_level_entries", wp.int32, ("entry",)),
)
"""The packed factor's topology (:class:`.tree_plan.FactorPattern`), uploaded once."""

_SparseMassMatrixIndices = field_group.struct("_SparseMassMatrixIndices", SPARSE_MASS_INDICES, __name__)


def factor_indices(pattern: FactorPattern, device) -> _SparseMassMatrixIndices:
    """Upload a tree's factor pattern once; all per-step values remain caller-owned."""
    return field_group.upload(_SparseMassMatrixIndices, device, pattern)


SPARSE_ROW_SUPPORT = (
    Field("enabled", wp.int32),
    Field("body_class", wp.int32, ("body",)),
    Field("pair_class", wp.int32, ("class", "class")),
    Field("class_dof", wp.int32, ("class", "dof")),
    Field("row_class", wp.int32, ("world", "row")),
    Field("limit_class", wp.int32, ("dof",)),
)
"""Model-owned support patterns and each live row's pattern index."""

_SparseRowSupport = field_group.struct("_SparseRowSupport", SPARSE_ROW_SUPPORT, __name__)


def row_supports(classes: SupportClasses, world_count: int, max_constraints: int, device) -> _SparseRowSupport:
    """Upload a tree's row support catalogue (:meth:`.tree_plan.TreePlan.support_classes`) and allocate only one
    support index per row."""
    row_class = wp.empty((world_count, max_constraints), dtype=int, device=device)
    return field_group.upload(_SparseRowSupport, device, classes, enabled=1, row_class=row_class)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The sparse class's row support catalogue: each body's and row's support class over a tree's coordinates
(:meth:`.tree_plan.TreePlan.support_classes`), uploaded once, and one support index per live row."""

import warp as wp

from . import field_group
from .field_group import Field
from .tree_plan import SupportClasses

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

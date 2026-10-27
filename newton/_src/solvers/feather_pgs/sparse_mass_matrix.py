# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Topology-derived, leaf-first mass factors for small articulated forests.

The sparse coordinate ``i`` names physical DOF ``permutation[i]``. With that
permutation, ``P H P.T = L L.T`` and ``Linv = inverse(L)`` have the same lower
ancestor pattern. Neither assembly nor factorization allocates a dense H.
"""

import warp as wp

from . import field_group, response_operator
from .device_preludes import lane_group
from .field_group import Field
from .kernel_template import THREADS, KernelTemplate, substitute
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


_RESPONSE = response_operator.SparseInverseFactor(
    "Linv_group.data+base",
    response_operator.CompressedSupport(
        "indices.row_offsets.data",
        "indices.columns.data",
        "indices.column_offsets.data",
        "indices.column_entries.data",
        "indices.entry_rows.data",
    ),
)


# Apply physical H inverse as P.T Linv.T Linv P; launch 32 threads per group.
solve_sparse_mass_matrix = KernelTemplate(
    "solve_sparse_mass_matrix",
    (
        ("group_to_art", wp.array[int]),
        ("articulation_dof_start", wp.array[int]),
        ("indices", _SparseMassMatrixIndices),
        ("Linv_group", wp.array2d[float]),
        ("tau", wp.array[float]),
        ("scratch", wp.array2d[float]),
        ("qdd", wp.array[float]),
    ),
    substitute(
        """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if !defined(__CUDA_ARCH__)
    if ((tid & 31) != 0) return;
#endif
    const int n = indices.dof_count;
    const int base = group * indices.nonzero_count;
    const int start = articulation_dof_start.data[group_to_art.data[group]];
    for (int row = LANE; row < n; row += STRIDE) {
        $FORWARD_ACTION
        scratch.data[group * n + row] = value;
    }
    SYNC();
    for (int col = LANE; col < n; col += STRIDE) {
        $TRANSPOSE_ACTION
        qdd.data[start + indices.permutation.data[col]] = value;
    }
""",
        FORWARD_ACTION=_RESPONSE.forward("row", lambda index: f"tau.data[start+indices.permutation.data[{index}]]"),
        TRANSPOSE_ACTION=_RESPONSE.transpose("col", lambda index: f"scratch.data[group*n+{index}]"),
    ),
    THREADS,
    preludes=(lane_group(32),),
    dual=True,
    native_name="_solve_native",
    module=__name__,
).kernel()

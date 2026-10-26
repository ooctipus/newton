# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Topology-derived, leaf-first mass factors for small articulated forests.

The sparse coordinate ``i`` names physical DOF ``permutation[i]``. With that
permutation, ``P H P.T = L L.T`` and ``Linv = inverse(L)`` have the same lower
ancestor pattern. Neither assembly nor factorization allocates a dense H.
"""

from functools import cache

import warp as wp

from . import field_group, response_operator
from .device_preludes import dual_native, lane_group
from .field_group import Field
from .kernel_template import THREADS, KernelTemplate, substitute
from .sparse_factorization import _sparse_cholesky_source, _sparse_inverse_source
from .spatial_laws import INERTIA_RECORDS, SPATIAL, inertia_record
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


@cache
def _get_crba_sparse_factor_kernel(
    dof_count: int, nonzero_count: int, *, warps_per_block: int = 4, geometric: bool = False
) -> wp.Kernel:
    """Assemble CRBA and produce packed Linv, one warp per articulation.

    Launch ``group_count * 32`` threads with ``warps_per_block * 32`` threads per
    block. All value arrays must be contiguous. A zero articulation update mask
    retains the inverse factor and status. Status is zero after a successful refresh,
    or one for a nonpositive/nonfinite pivot; no pivot floor is introduced.
    """
    n, nnz, warps = int(dof_count), int(nonzero_count), int(warps_per_block)
    if not 1 <= n <= 64 or not n <= nnz <= n * (n + 1) // 2 or not 1 <= warps <= 32:
        raise ValueError("Invalid sparse factor dimensions or warp count")
    force_storage_cuda = (
        ""
        if geometric
        else f"""    __shared__ float forces[{warps * n * 6}];
    float* force = forces + SUB * {n * 6};"""
    )
    force_storage_cpu = "" if geometric else f"""    float force[{n * 6}];"""
    prepare_mass = (
        ""
        if geometric
        else f"""    for (int physical = lane; physical < {n}; physical += stride) {{
        const int joint = articulation_start.data[art] + indices.dof_joint.data[physical];
        const auto value = InertiaRecord::apply(body_I_c.data[joint_child.data[joint]].c, joint_S_s.data[start + physical]);
        for (int k = 0; k < 6; ++k) force[physical * 6 + k] = value.c[k];
    }}
"""
    )
    read_mass = (
        "        float value = body_I_c.data[group * " + str(nnz) + "+entry];"
        if geometric
        else """        const auto motion = joint_S_s.data[start + physical_row];
        float value = 0.0f;
        for (int k = 0; k < 6; ++k) value += motion.c[k] * force[physical_col * 6 + k];
"""
    )
    mass_input_type = wp.array2d[float] if geometric else wp.array[inertia_record]
    body = f"""
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
    const int art = group_to_art.data[group];
    if (mass_update_mask.data[art] == 0) return;
    const int lane = LANE, stride = STRIDE;
#if defined(__CUDA_ARCH__)
    __shared__ float factors[{warps * nnz}];
    __shared__ float inverses[{warps * nnz}];
{force_storage_cuda}
    float* factor = factors + SUB * {nnz};
    float* inverse = inverses + SUB * {nnz};

#else
    if ((tid & 31) != 0) return;
    float factor[{nnz}];
    float inverse[{nnz}];
{force_storage_cpu}
#endif
    const int start = articulation_dof_start.data[art];
{prepare_mass}
    if (lane == 0) status.data[group] = 0;
    SYNC();
    for (int entry = lane; entry < {nnz}; entry += stride) {{
        const int row = indices.entry_rows.data[entry];
        const int col = indices.columns.data[entry];
        const int physical_row = indices.permutation.data[row];
        const int physical_col = indices.permutation.data[col];
{read_mass}
        if (row == col) {{
            value += R_group.data[group * {n} + physical_row];
            if (fused_augmented_drive != 0) {{
                const int drive = drive_row_by_dof.data[start + physical_row];
                if (drive >= 0 && row_K.data[drive] > 0.0f) value += row_K.data[drive];
            }}
        }}
        factor[entry] = value;
    }}
    SYNC();
    {_sparse_cholesky_source(n, nnz)}
    {_sparse_inverse_source(n)}
    SYNC();
    const bool valid = status.data[group] == 0;
    for (int entry = lane; entry < {nnz}; entry += stride) {{
        Linv_group.data[group * {nnz} + entry] = valid ? inverse[entry] : NAN;
    }}
"""
    preludes = (lane_group(32),) if geometric else (lane_group(32), *SPATIAL.preludes(), INERTIA_RECORDS)
    snippet = dual_native(body, preludes=preludes)

    name = f"crba_sparse_factor_{n}_{nnz}_{warps}"
    if geometric:
        name = "cached_" + name
    return KernelTemplate(
        name,
        (
            ("group_to_art", wp.array[int]),
            ("mass_update_mask", wp.array[int]),
            ("articulation_start", wp.array[int]),
            ("articulation_dof_start", wp.array[int]),
            ("joint_child", wp.array[int]),
            ("joint_S_s", wp.array[wp.spatial_vector]),
            ("body_I_c", mass_input_type),
            ("R_group", wp.array2d[float]),
            ("fused_augmented_drive", int),
            ("drive_row_by_dof", wp.array[int]),
            ("row_K", wp.array[float]),
            ("indices", _SparseMassMatrixIndices),
            ("Linv_group", wp.array2d[float]),
            ("status", wp.array[int]),
        ),
        snippet,
        THREADS,
        composed=True,
        native_name="factor_native",
    ).kernel()


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

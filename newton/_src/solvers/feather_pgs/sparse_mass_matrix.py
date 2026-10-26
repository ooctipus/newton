# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Topology-derived, leaf-first mass factors for small articulated forests.

The sparse coordinate ``i`` names physical DOF ``permutation[i]``. With that
permutation, ``P H P.T = L L.T`` and ``Linv = inverse(L)`` have the same lower
ancestor pattern. Neither assembly nor factorization allocates a dense H.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import warp as wp

from . import field_group, host_index, response_operator
from .device_preludes import dual_native, lane_group
from .field_group import Field
from .kernel_template import THREADS, KernelTemplate, substitute
from .sparse_factorization import _sparse_cholesky_source, _sparse_inverse_source


def _factor_levels(index):
    """Derive independent pivots from a packed lower triangular pattern."""
    n = index.shape[0]
    levels = np.zeros(n, dtype=np.int32)
    for col in range(n):
        previous = np.flatnonzero(index[col, :col] >= 0)
        levels[col] = 0 if not len(previous) else 1 + int(np.max(levels[previous]))
    return levels


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
"""The packed factor's topology (:class:`_SparseMassMatrixPlan`), uploaded once."""

_SparseMassMatrixIndices = field_group.struct("_SparseMassMatrixIndices", SPARSE_MASS_INDICES, __name__)


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


@dataclass(frozen=True, eq=False)
class _SparseRowSupportPlan:
    """Reuse structural endpoint unions without per-row index vectors.

    Coordinates belong to one responding articulation at world offset zero.
    All joint-block entries remain present, even when a coefficient is zero.
    """

    body_class: np.ndarray
    pair_class: np.ndarray
    class_dof: np.ndarray
    limit_class: np.ndarray

    @classmethod
    def build(cls, mass_plan, body_masks, *, max_row_dofs: int) -> "_SparseRowSupportPlan | None":
        """Enumerate collision endpoint unions and every limit support.

        Every body is a possible endpoint. Shape attachment is topology;
        geometry, materials and live limit activation do not change this plan.
        """
        body_masks = np.asarray(body_masks, dtype=np.uint64)
        if body_masks.ndim != 1 or not 1 <= max_row_dofs <= mass_plan.dof_count:
            raise ValueError("Sparse row supports require one-dimensional masks and a valid row width")
        if any(int(mask) >> mass_plan.dof_count for mask in body_masks):
            raise ValueError("Sparse row support exceeds the articulation coordinate range")
        classes = np.unique(np.concatenate((np.zeros(1, dtype=np.uint64), body_masks)))
        pairs = classes[:, None] | classes[None, :]
        masks = np.unique(np.concatenate((pairs.ravel(), mass_plan.ancestor_mask)))
        if any(int(mask).bit_count() > max_row_dofs for mask in masks):
            return None
        if any(int(mask) >> mass_plan.dof_count for mask in masks):
            raise ValueError("Sparse row support exceeds the articulation coordinate range")
        class_dof = np.full((len(masks), max_row_dofs), -1, dtype=np.int32)
        for index, mask in enumerate(masks):
            entries = [dof for dof in range(mass_plan.dof_count) if int(mask) & (1 << dof)]
            class_dof[index, : len(entries)] = entries
        return cls(
            # Every body's mask is one of the classes.
            body_class=np.searchsorted(classes, body_masks).astype(np.int32),
            pair_class=np.searchsorted(masks, pairs).astype(np.int32),
            class_dof=class_dof,
            # CSC factor support omits structural zeros inside multi-DOF joints.
            limit_class=np.searchsorted(masks, mass_plan.ancestor_mask).astype(np.int32),
        )

    def to_device(self, world_count: int, max_constraints: int, device) -> _SparseRowSupport:
        """Upload the catalogue and allocate only one support index per row."""
        row_class = wp.empty((world_count, max_constraints), dtype=int, device=device)
        support = field_group.upload(_SparseRowSupport, device, self, enabled=1, row_class=row_class)
        return support


@dataclass(frozen=True, eq=False)
class _SparseMassMatrixPlan:
    """Immutable sparsity for one articulation topology, independent of pose.

    Parent indices are local joint indices, not body indices. Fixed joints have
    zero DOFs; physical DOFs are consecutive in input joint order. Mask bits use
    sparse coordinates, including every coordinate in an ancestor joint block.
    The initial device path is bounded to 64 DOFs, not to a robot or task name.
    """

    joint_parent: np.ndarray
    """Local parent joint indices, with -1 for each root."""
    joint_dof_count: np.ndarray
    """Number of physical coordinates in each joint block."""
    permutation: np.ndarray
    """Sparse-coordinate to physical-coordinate permutation."""
    inverse_permutation: np.ndarray
    """Physical-coordinate to sparse-coordinate permutation."""
    row_offsets: np.ndarray
    """CSR offsets into the packed lower factor."""
    columns: np.ndarray
    """Sparse-coordinate column index for each packed entry."""
    entry_rows: np.ndarray
    """Sparse-coordinate row index for each packed entry."""
    lookup: np.ndarray
    """Dense index table containing a packed entry index or -1."""
    dof_joint: np.ndarray
    """Local joint owning each physical coordinate."""
    ancestor_mask: np.ndarray
    """Sparse-coordinate ancestor support for each physical coordinate."""
    joint_ancestor_mask: np.ndarray
    """Sparse-coordinate endpoint support including fixed-joint endpoints."""
    column_offsets: np.ndarray
    """CSC offsets for each factor column's ancestor chain."""
    column_entries: np.ndarray
    """Packed factor indices ordered by column, then increasing row."""
    factor_level_offsets: np.ndarray
    """Offsets of independent Cholesky pivots at each dependency level."""
    factor_level_columns: np.ndarray
    """Sparse-coordinate pivot columns grouped by dependency level."""
    factor_level_entry_offsets: np.ndarray
    """Offsets of off-diagonal factor entries at each dependency level."""
    factor_level_entries: np.ndarray
    """Packed off-diagonal entries grouped by their column's dependency level."""

    @property
    def dof_count(self) -> int:
        return len(self.permutation)

    @property
    def nonzero_count(self) -> int:
        return len(self.columns)

    @property
    def factor_level_count(self) -> int:
        return len(self.factor_level_offsets) - 1

    @classmethod
    def build(cls, joint_parent, joint_dof_count) -> "_SparseMassMatrixPlan":
        """Build a fill-free scalar pattern from a checked joint forest."""
        parents = np.asarray(joint_parent)
        counts = np.asarray(joint_dof_count)
        if parents.ndim != 1 or counts.shape != parents.shape or not len(parents):
            raise ValueError("Joint parents and DOF counts must be nonempty one-dimensional arrays of equal length")
        if not np.issubdtype(parents.dtype, np.integer) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("Joint parents and DOF counts must be integers")
        if np.any(parents < -1) or np.any(parents >= len(parents)) or np.any(counts < 0):
            raise ValueError("Invalid joint parent or negative DOF count")
        if np.any(counts > 64):
            raise ValueError("Sparse mass-matrix factors currently require 1 to 64 DOFs")
        parents = parents.astype(np.int32)
        counts = counts.astype(np.int32)
        n = int(np.sum(counts, dtype=np.int64))
        if n < 1 or n > 64:
            raise ValueError("Sparse mass-matrix factors currently require 1 to 64 DOFs")
        children = [[] for _ in parents]
        roots = []
        for joint, parent in enumerate(parents):
            if parent == -1:
                roots.append(joint)
            else:
                children[parent].append(joint)
        preorder = []
        pending = roots[::-1]
        while pending:
            joint = pending.pop()
            preorder.append(joint)
            pending.extend(children[joint][::-1])
        if len(preorder) != len(parents):
            raise ValueError("Joint parents must form an acyclic forest")
        starts = host_index.offsets(counts)
        permutation = np.array(
            [dof for joint in reversed(preorder) for dof in range(starts[joint + 1] - 1, starts[joint] - 1, -1)],
            dtype=np.int32,
        )
        inverse_permutation = np.argsort(permutation).astype(np.int32)
        dof_joint = host_index.expand(counts)[0].astype(np.int32)
        joint_masks = [0] * len(parents)
        for joint in range(len(parents)):
            ancestor = joint
            while ancestor != -1:
                for dof in range(starts[ancestor], starts[ancestor + 1]):
                    joint_masks[joint] |= 1 << int(inverse_permutation[dof])
                ancestor = int(parents[ancestor])
        row_offsets = [0]
        columns = []
        entry_rows = []
        lookup = np.full((n, n), -1, dtype=np.int32)
        for row in range(n):
            for col in range(row + 1):
                col_joint = int(dof_joint[permutation[col]])
                if joint_masks[col_joint] & (1 << row):
                    lookup[row, col] = len(columns)
                    entry_rows.append(row)
                    columns.append(col)
            row_offsets.append(len(columns))
        column_offsets = [0]
        column_entries = []
        for col in range(n):
            column_entries.extend(int(lookup[row, col]) for row in range(col, n) if lookup[row, col] >= 0)
            column_offsets.append(len(column_entries))

        # Reverse DFS places every descendant subtree in one contiguous row
        # interval. A pivot depends only on those earlier descendants; sibling
        # branches can therefore be factored at the same dependency level.
        levels = _factor_levels(lookup)
        level_count = int(np.max(levels)) + 1
        level_offsets, level_columns = host_index.group_by(levels, level_count)
        level_entries = sorted(
            (entry for entry, (row, col) in enumerate(zip(entry_rows, columns, strict=True)) if row != col),
            key=lambda entry: (levels[columns[entry]], columns[entry], entry_rows[entry]),
        )
        entry_counts = np.bincount([levels[columns[entry]] for entry in level_entries], minlength=level_count)
        values = {
            "joint_parent": parents,
            "joint_dof_count": counts,
            "permutation": permutation,
            "inverse_permutation": inverse_permutation,
            "row_offsets": np.asarray(row_offsets, dtype=np.int32),
            "columns": np.asarray(columns, dtype=np.int32),
            "entry_rows": np.asarray(entry_rows, dtype=np.int32),
            "lookup": lookup,
            "dof_joint": dof_joint,
            "ancestor_mask": np.asarray([joint_masks[joint] for joint in dof_joint], dtype=np.uint64),
            "joint_ancestor_mask": np.asarray(joint_masks, dtype=np.uint64),
            "column_offsets": np.asarray(column_offsets, dtype=np.int32),
            "column_entries": np.asarray(column_entries, dtype=np.int32),
            "factor_level_offsets": level_offsets.astype(np.int32),
            "factor_level_columns": level_columns.astype(np.int32),
            "factor_level_entry_offsets": host_index.offsets(entry_counts, dtype=np.int32),
            "factor_level_entries": np.asarray(level_entries, dtype=np.int32),
        }
        for value in values.values():
            value.setflags(write=False)
        return cls(**values)

    def to_device(self, device) -> _SparseMassMatrixIndices:
        """Upload the topology once; all per-step values remain caller-owned."""
        indices = field_group.upload(_SparseMassMatrixIndices, device, self)
        return indices


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
        const auto value = wp::mul(body_I_c.data[joint_child.data[joint]], joint_S_s.data[start + physical]);
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
    mass_input_type = wp.array2d[float] if geometric else wp.array[wp.spatial_matrix]
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
    snippet = dual_native(body, preludes=(lane_group(32),))

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

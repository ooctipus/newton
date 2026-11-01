# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The dense factor of the held tree and the six-coordinate group's register whitening.

* :func:`dense_factor_kernel`: the leaf-first factor ``P (H + diag(R)) P^T = L L^T`` (``P`` reverses an
  articulation's coordinates, :func:`.leaf_first`) from the held tree's geometry, level by level over a warp's lanes;
* :func:`whiten_six`: one row of the six-coordinate group whitened in registers by the kit's forward solve
  (:class:`.CholeskySolve`), the dense part of a compact boundary row;
* the kit's text the row builds share: its index map (:func:`_element`) and indentation.

Rows are whitened where they are placed (:mod:`.packed_rows`, the in-kernel owners, the sparse placements); no route
builds ``Y = H^-1 J^T``.
"""

from __future__ import annotations

import functools

import numpy as np
import warp as wp

from . import field_group
from .device_preludes import SHARED_STORAGE, lane_group, substitute
from .field_group import Field
from .kernel_template import THREADS, KernelTemplate
from .response_operator import CholeskySolve
from .staging import Buffer, Staging
from .tree_plan import level_schedule, offsets

__all__ = ["dense_factor_kernel", "dense_factor_plan", "whiten_six"]


# ---------------------------------------------------------------------------------------------------------------------
# The kit's text: its index map and indentation
# ---------------------------------------------------------------------------------------------------------------------


def _indent(text: str, pad: str) -> str:
    """Prefix every non-empty line of ``text``."""
    return "\n".join(pad + line if line.strip() else line for line in text.split("\n"))


def _element(array: str, *indices: str) -> str:
    """``array[indices]`` of a contiguous array, flattened through the array's own shape (Warp's indexing)."""
    offset = indices[0]
    if len(indices) > 1:
        offset = f"(size_t)({offset})"
        for axis, index in enumerate(indices[1:], start=1):
            offset = f"({offset}) * {array}.shape[{axis}] + ({index})"
    return f"{array}.data[{offset}]"


# ---------------------------------------------------------------------------------------------------------------------
# Stage "factor": H + diag(R) = L L^T
# ---------------------------------------------------------------------------------------------------------------------

# The held tree's factor, of any size. A warp per articulation assembles the entries of P H P^T in place in L (the
# coordinates reversed; geometry, then R, then K of the physical coordinate on the diagonal: S F + R + K in that order)
# and walks them level by level: each level's pivots, then the entries below them, left-looking, over the lanes. A pivot
# leaves its reciprocal at the end of its row, in the upper triangle. Up to STAGED_FACTOR_SIZE the working L is in shared
# memory and only the pattern's entries are published; past it L is factored in place in global memory, and the
# reciprocals stay above the diagonal, which no reader of L reads. The CPU runs a warp's lanes on its first thread. A
# structural zero of L stays the array's zero. Last, each drive's pivot without its K, row_P: its H_ii less the same
# squares, which the factor's own pivot subtracts after adding K, so that pivot keeps its order (a release's pivot,
# tree_kinetic_state.release).
DENSE_FACTOR_BODY = """
    for (int entry = LANE; entry < plan.entry_count; entry += STRIDE) {
        const int row = plan.rows.data[entry], col = plan.columns.data[entry];
        const int geometry = plan.geometry.data[entry];
        float value = geometry < 0 ? 0.0f : mass[geometry];
        if (row == col) {
            const int physical = n - 1 - row;
            value += R_group.data[factor_group * n + physical];
            const int drive_row = drive_row_by_dof.data[dof_start + physical];
            if (drive_row >= 0) {
                row_H.data[drive_row] = value;
                const float K = row_K.data[drive_row];
                if (K > 0.0f) value += K;
            }
        }
        L[row * n + col] = value;
    }
    SYNC();
    for (int level = 0; level < plan.level_count; ++level) {
        const int pivot_end = plan.factor_level_offsets.data[level + 1];
        for (int pivot = plan.factor_level_offsets.data[level] + LANE; pivot < pivot_end; pivot += STRIDE) {
            const int col = plan.factor_level_columns.data[pivot];
            const int prefix = plan.row_offsets.data[col + 1] - 1;
            float* own = L + col * n;
            float value = own[col];
            for (int entry = plan.row_offsets.data[col]; entry < prefix; ++entry) {
                const float term = own[plan.columns.data[entry]];
                value -= term * term;
            }
            value = sqrtf(value);
            own[col] = value;
            if (col + 1 < n) own[n - 1] = 1.0f / value;
        }
        SYNC();
        const int work_end = plan.factor_level_entry_offsets.data[level + 1];
        for (int work = plan.factor_level_entry_offsets.data[level] + LANE; work < work_end; work += STRIDE) {
            const int target = plan.factor_level_entries.data[work];
            const int col = plan.columns.data[target];
            const int prefix = plan.row_offsets.data[col + 1] - 1;
            float* below = L + plan.rows.data[target] * n;
            const float* own = L + col * n;
            float value = below[col];
            for (int entry = plan.row_offsets.data[col]; entry < prefix; ++entry) {
                const int k = plan.columns.data[entry];
                value -= below[k] * own[k];
            }
            below[col] = value * own[n - 1];
        }
        SYNC();
    }
    for (int col = LANE; col < n; col += STRIDE) {
        const int drive_row = drive_row_by_dof.data[dof_start + n - 1 - col];
        if (drive_row < 0) continue;
        const float* own = L + col * n;
        float pivot = row_H.data[drive_row];
        for (int entry = plan.row_offsets.data[col]; entry < plan.row_offsets.data[col + 1] - 1; ++entry) {
            const float term = own[plan.columns.data[entry]];
            pivot -= term * term;
        }
        row_P.data[drive_row] = pivot;
    }
"""

_DENSE_FACTOR = """
    const int factor_group = tid >> 5;
    if (factor_group >= group_to_art.shape[0]) return;
#if !defined(__CUDA_ARCH__)
    if ((tid & 31) != 0) return;
#endif
    const int articulation = group_to_art.data[factor_group];
    if (mass_update_mask.data[articulation] == 0) return;
    const int n = plan.size;
    const int dof_start = articulation_dof_start.data[articulation];
    const float* mass = geometric.data + geometry_group.data[factor_group] * geometric.shape[1];
$STAGING
$BODY
$PUBLISH
"""

DENSE_FACTOR_PUBLISH = """
    for (int entry = LANE; entry < plan.entry_count; entry += STRIDE) {
        const int element = plan.rows.data[entry] * n + plan.columns.data[entry];
        L_group.data[factor_group * n * n + element] = L[element];
    }
"""

STAGED_FACTOR_SIZE = 32
"""The largest articulation :func:`dense_factor_kernel` factors in shared memory."""

DENSE_FACTOR_PLAN = (
    Field("size", wp.int32),
    Field("entry_count", wp.int32),
    Field("level_count", wp.int32),
    Field("rows", wp.int32, ("entry",)),
    Field("geometry", wp.int32, ("entry",)),
    Field("row_offsets", wp.int32, ("dof_bound",)),
    Field("columns", wp.int32, ("entry",)),
    Field("factor_level_offsets", wp.int32, ("level_bound",)),
    Field("factor_level_columns", wp.int32, ("dof",)),
    Field("factor_level_entry_offsets", wp.int32, ("level_bound",)),
    Field("factor_level_entries", wp.int32, ("entry",)),
    Field("column_offsets", wp.int32, ("dof_bound",)),
    Field("column_rows", wp.int32, ("entry",)),
)
"""The lower pattern of a dense factor in leaf-first coordinates, fill included: each entry's row, column and geometry slot
(``-1`` on fill), the level schedule (:func:`.tree_plan.level_schedule`), and each column's rows below its diagonal in
order: its path to the root, which a release walks (:func:`.tree_kinetic_state.release`)."""

DenseFactorPlan = field_group.struct("DenseFactorPlan", DENSE_FACTOR_PLAN, __name__)


def dense_factor_plan(physical_lookup, device) -> DenseFactorPlan:
    """The dense factor's plan from ``physical_lookup``, a symmetric physical ``(row, col)`` index of the held geometry
    (``-1`` where structurally zero). The dense response's order reverses the coordinates, which is fill-free only for
    joints in depth-first order, so the pattern takes its fill here."""
    index = np.asarray(physical_lookup)[::-1, ::-1]
    n = index.shape[0]
    present = np.tril(index >= 0)
    for col in range(n):
        below = col + 1 + np.flatnonzero(present[col + 1 :, col])
        present[np.ix_(below, below)] = True
    rows, cols = np.nonzero(np.tril(present))
    schedule = level_schedule(rows, cols, n)
    below = np.flatnonzero(rows > cols)
    below = below[np.lexsort((rows[below], cols[below]))]
    # A release walks column i's rows as its path to the root (tree_kinetic_state.RELEASE_DENSE): every column's rows are its
    # parent and the parent's rows. A tree's coupling in reversed topological order has it, and a filled pattern keeps it.
    column = {c: rows[below][cols[below] == c].tolist() for c in range(n)}
    if any(column[c] and column[c][1:] != column[column[c][0]] for c in range(n)):
        raise ValueError("A dense factor's pattern must hold each column's whole path to the root")
    return field_group.upload(
        DenseFactorPlan,
        device,
        size=n,
        entry_count=rows.size,
        level_count=schedule["factor_level_offsets"].size - 1,
        rows=rows.astype(np.int32),
        geometry=index[rows, cols].astype(np.int32),
        row_offsets=offsets(np.bincount(rows, minlength=n), np.int32),
        columns=cols.astype(np.int32),
        column_offsets=offsets(np.bincount(cols[below], minlength=n), np.int32),
        column_rows=rows[below].astype(np.int32),
        **schedule,
    )


@functools.cache
def dense_factor_kernel(size: int) -> wp.Kernel:
    """The held tree's leaf-first factor, one warp per articulation and four per block (``dim = articulations * 32``):
    the pattern of ``plan`` (:func:`dense_factor_plan`) with the geometry of each group's ``geometry_group`` row, into
    ``L_group``. Up to :data:`STAGED_FACTOR_SIZE` the working factor of ``size`` coordinates is in shared memory; past
    it, or at 0, ``L_group`` holds it in place."""
    size = int(size) if size <= STAGED_FACTOR_SIZE else 0
    inputs = (
        ("articulation_dof_start", wp.array[int]),
        ("mass_update_mask", wp.array[int]),
        ("group_to_art", wp.array[int]),
        ("geometry_group", wp.array[int]),
        ("plan", DenseFactorPlan),
        ("R_group", wp.array2d[float]),
        ("drive_row_by_dof", wp.array[int]),
        ("row_K", wp.array[float]),
        ("row_H", wp.array[float]),
        ("row_P", wp.array[float]),
        ("geometric", wp.array2d[float]),
        ("L_group", wp.array3d[float]),
    )
    staging = Staging((Buffer("L", "float", size * size),), groups=4, layout="arrays").declarations() if size else ""
    snippet = substitute(
        _DENSE_FACTOR,
        STAGING=staging or "    float* L = L_group.data + factor_group * n * n;",
        BODY=DENSE_FACTOR_BODY,
        PUBLISH=DENSE_FACTOR_PUBLISH if size else "",
    )
    return KernelTemplate(
        f"dense_factor_{size}",
        inputs,
        snippet,
        THREADS,
        preludes=(lane_group(32), SHARED_STORAGE),
        dual=True,
        native_name="factor_native",
    ).kernel()


_Vec6 = wp.types.vector(length=6, dtype=wp.float32)


def _whiten_six_snippet() -> str:
    solve = CholeskySolve(6, lambda row, col: _element("factor", "group", row, col))

    def jacobian_at(i):
        return f"jacobian[{i}]"

    def whitened_at(i):
        return f"whitened[{i}]"

    return f"""
    wp::vec_t<6, float> whitened(0.0f);
{_indent(solve.whiten(jacobian_at, whitened_at), "    ")}
    return whitened;
"""


@wp.func_native(_whiten_six_snippet())
def whiten_six(factor: wp.array3d[float], group: int, jacobian: _Vec6) -> _Vec6:
    """One row of the six-coordinate group whitened in registers, ``Z = L^-1 P J^T`` by the kit's forward solve (Z0):
    a free rigid body's side of a walked row (:func:`.walk_rows.walk_wrench`) until the free bodies walk."""

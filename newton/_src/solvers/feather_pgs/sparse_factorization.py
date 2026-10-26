# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse triangular operations shared by warp and block factor producers."""


def _sparse_inverse_source(dof_count: int, *, local_column_width: int = 0) -> str:
    """Emit inverse columns while retaining the caller's execution mapping.

    The caller supplies ``indices`` (column topology), ``factor`` and
    ``inverse`` pointers, and ``lane``/``stride``. A positive column width keeps
    each column's prefix in thread-local storage and reports nonfinite values
    through the caller's shared ``bad`` flag. Otherwise prefixes use the shared
    inverse buffer. Synchronization and publication remain caller-owned.

    Columns are CSC: row offsets and packed entry IDs describe each ancestor chain.
    """
    if not 1 <= dof_count <= 64 or not 0 <= local_column_width <= dof_count:
        raise ValueError("Sparse inverse dimensions must fit the topology's 64-coordinate bound")
    bounds = """const int begin = indices.column_offsets.data[col];
        const int end = indices.column_offsets.data[col + 1];"""
    row = """const int target = indices.column_entries.data[position];
            const int row = indices.entry_rows.data[target];
            const int diagonal = indices.row_offsets.data[row + 1] - 1;"""
    entry = """const int other = indices.column_entries.data[previous];
                const int factor_entry = diagonal - row + indices.entry_rows.data[other];"""
    diagonal, target, first = "diagonal", "target", "row == col"
    scratch = f"float values[{local_column_width}];" if local_column_width else ""
    previous_value = "values[previous - begin]" if local_column_width else "inverse[other]"
    publish = ""
    if local_column_width:
        publish = "values[position - begin] = value;\n            if (!isfinite(value)) atomicExch(&bad, 1);"
    return f"""
    // Each inverse column follows only its ancestor chain.
    for (int col = lane; col < {dof_count}; col += stride) {{
        {scratch}
        {bounds}
        for (int position = begin; position < end; ++position) {{
            {row}
            float value = {first} ? 1.0f : 0.0f;
            for (int previous = begin; previous < position; ++previous) {{
                {entry}
                value -= factor[factor_entry] * {previous_value};
            }}
            value /= factor[{diagonal}];
            {publish}
            inverse[{target}] = value;
        }}
    }}
"""


def _sparse_cholesky_source(dof_count: int, nonzero_count: int) -> str:
    """Emit a left-looking packed Cholesky operation with an explicit traversal contract.

    It consumes ancestor-closed packed storage and derives work from topology
    metadata, gathering prior row products using CSR prefixes and factor
    levels, one warp per articulation (or serial on CPU).

    The caller binds ``indices``, mutable ``factor``, ``status``, ``group`` and
    ``lane``/``stride``, and declares a lane group (``SYNC``). Producers, mass assembly, inverse generation and
    publication are external to this operation. No pivot floor or
    numerical-policy change is introduced.
    """
    if not 1 <= dof_count <= 64 or not dof_count <= nonzero_count <= dof_count * (dof_count + 1) // 2:
        raise ValueError("Invalid packed sparse factor dimensions")
    return r"""    for (int level = 0; level < indices.factor_level_count; ++level) {
        const int level_begin = indices.factor_level_offsets.data[level];
        const int level_end = indices.factor_level_offsets.data[level + 1];
        for (int pivot = level_begin + lane; pivot < level_end; pivot += stride) {
            const int col = indices.factor_level_columns.data[pivot];
            const int begin = indices.row_offsets.data[col];
            const int diagonal = indices.row_offsets.data[col + 1] - 1;
            float value = factor[diagonal];
            for (int entry = begin; entry < diagonal; ++entry) value -= factor[entry] * factor[entry];
            if (!(value > 0.0f) || !wp::isfinite(value)) {
#if defined(__CUDA_ARCH__)
                atomicExch(status.data + group, 1);
#else
                status.data[group] = 1;
#endif
            }
            factor[diagonal] = sqrtf(value);
        }
        SYNC();
        const int entry_begin = indices.factor_level_entry_offsets.data[level];
        const int entry_end = indices.factor_level_entry_offsets.data[level + 1];
        for (int work = entry_begin + lane; work < entry_end; work += stride) {
            const int target = indices.factor_level_entries.data[work];
            const int row = indices.entry_rows.data[target];
            const int col = indices.columns.data[target];
            const int begin = indices.row_offsets.data[col];
            const int diagonal = indices.row_offsets.data[col + 1] - 1;
            // Ancestor closure makes the column's whole prefix present in this
            // row. Both prefixes are contiguous under the reversed DFS order.
            const int row_begin = target - (diagonal - begin);
            float value = factor[target];
            for (int entry = begin; entry < diagonal; ++entry) {
                value -= factor[row_begin + entry - begin] * factor[entry];
            }
            factor[target] = value / factor[diagonal];
        }
        SYNC();
    }
"""

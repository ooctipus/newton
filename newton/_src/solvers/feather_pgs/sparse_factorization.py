# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse triangular operations shared by warp and block factor producers."""


def _sparse_inverse_source(dof_count: int, *, local_column_width: int = 0, column_layout: str = "csc") -> str:
    """Emit inverse columns while retaining the caller's execution mapping.

    The caller supplies ``indices`` (column topology), ``factor`` and
    ``inverse`` pointers, and ``lane``/``stride``. A positive column width keeps
    each column's prefix in thread-local storage and reports nonfinite values
    through the caller's shared ``bad`` flag. Otherwise prefixes use the shared
    inverse buffer. Synchronization and publication remain caller-owned.

    CSC columns use row offsets and packed entry IDs. Padded columns use
    ``inverse_count``, ``inverse_nodes`` and ``index`` arrays, with a positive
    local column width as their stride. Both describe the same ancestor chain.
    """
    if not 1 <= dof_count <= 64 or not 0 <= local_column_width <= dof_count:
        raise ValueError("Sparse inverse dimensions must fit the topology's 64-coordinate bound")
    if column_layout == "csc":
        bounds = """const int begin = indices.column_offsets.data[col];
        const int end = indices.column_offsets.data[col + 1];"""
        row = """const int target = indices.column_entries.data[position];
            const int row = indices.entry_rows.data[target];
            const int diagonal = indices.row_offsets.data[row + 1] - 1;"""
        entry = """const int other = indices.column_entries.data[previous];
                const int factor_entry = diagonal - row + indices.entry_rows.data[other];"""
        diagonal, target, first = "diagonal", "target", "row == col"
    elif column_layout == "padded" and local_column_width:
        bounds = "const int begin = 0;\n        const int end = indices.inverse_count.data[col];"
        row = f"const int row = indices.inverse_nodes.data[col*{local_column_width}+position];"
        entry = f"""const int prev = indices.inverse_nodes.data[col*{local_column_width}+previous];
                const int factor_entry = indices.index.data[row*{dof_count}+prev];"""
        diagonal = f"indices.index.data[row*{dof_count}+row]"
        target, first = f"indices.index.data[row*{dof_count}+col]", "position == 0"
    else:
        raise ValueError("Use CSC columns or padded columns with a positive local width")
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


def _sparse_cholesky_source(dof_count: int, nonzero_count: int, *, algorithm: str, level_count: int = 0) -> str:
    """Emit a packed Cholesky operation with an explicit traversal contract.

    All variants consume ancestor-closed packed storage and derive work from
    topology metadata. ``left_looking`` gathers prior row products using CSR
    prefixes and factor levels, one warp per articulation (or serial on CPU).
    ``right_looking`` updates trailing entries in pivot order.
    ``right_looking_levels`` gathers Schur updates with one owner per target.
    The latter two use a cooperative block and a shared ``bad`` status flag.

    The caller binds ``indices``, mutable ``factor``, and ``lane``/``stride``.
    Left-looking also binds ``status`` and ``group``. Right-looking indices
    provide lookup/row/column arrays and optionally the level-update schedule.
    Producers, mass assembly, inverse generation and publication are external
    to this operation. No pivot floor or numerical-policy change is introduced.
    """
    if not 1 <= dof_count <= 64 or not dof_count <= nonzero_count <= dof_count * (dof_count + 1) // 2:
        raise ValueError("Invalid packed sparse factor dimensions")
    if algorithm == "left_looking":
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
#if defined(__CUDA_ARCH__)
        __syncwarp();
#endif
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
#if defined(__CUDA_ARCH__)
        __syncwarp();
#endif
    }
"""
    if algorithm == "right_looking":
        return f"""    for (int k = 0; k < {dof_count}; ++k) {{
        const int diag = indices.index.data[k*{dof_count}+k];
        if (lane == 0) {{
            if (!(factor[diag] > 0.0f) || !isfinite(factor[diag])) bad = 1;
            factor[diag] = sqrtf(factor[diag]);
        }}
        __syncthreads();
        for (int row = k+1+lane; row < {dof_count}; row += stride) {{
            const int entry = indices.index.data[row*{dof_count}+k];
            if (entry >= 0) factor[entry] /= factor[diag];
        }}
        __syncthreads();
        for (int e = lane; e < {nonzero_count}; e += stride) {{
            const int row = indices.row.data[e], col = indices.col.data[e];
            if (col <= k) continue;
            const int x = indices.index.data[row*{dof_count}+k], y = indices.index.data[col*{dof_count}+k];
            if (x >= 0 && y >= 0) factor[e] -= factor[x] * factor[y];
        }}
        __syncthreads();
    }}
"""
    if algorithm == "right_looking_levels":
        if not 1 <= level_count <= dof_count:
            raise ValueError("Sparse level updates require a valid topology level count")
        return f"""    // Columns within a level are independent, but their ancestor Schur
    // destinations overlap. One destination owner gathers this level only.
    for (int level = 0; level < {level_count}; ++level) {{
        const int begin = 3*level, end = 3*(level+1);
        for (int task = indices.level_offsets.data[begin]+lane;
             task < indices.level_offsets.data[end]; task += stride) {{
            const int k = indices.level_pivots.data[task];
            const int diag = indices.index.data[k*{dof_count}+k];
            if (!(factor[diag] > 0.0f) || !isfinite(factor[diag])) atomicExch(&bad, 1);
            factor[diag] = sqrtf(factor[diag]);
        }}
        __syncthreads();
        for (int task = indices.level_offsets.data[begin+1]+lane;
             task < indices.level_offsets.data[end+1]; task += stride) {{
            const int entry = indices.level_scales.data[task];
            const int col = indices.col.data[entry];
            factor[entry] /= factor[indices.index.data[col*{dof_count}+col]];
        }}
        __syncthreads();
        for (int task = indices.level_offsets.data[begin+2]+lane;
             task < indices.level_offsets.data[end+2]; task += stride) {{
            const int entry = indices.level_entries.data[task];
            float value = factor[entry];
            for (int term = indices.level_term_offsets.data[task];
                 term < indices.level_term_offsets.data[task+1]; ++term) {{
                value -= factor[indices.level_terms.data[2*term]] * factor[indices.level_terms.data[2*term+1]];
            }}
            factor[entry] = value;
        }}
        __syncthreads();
    }}
"""
    raise ValueError("Unknown sparse Cholesky traversal")

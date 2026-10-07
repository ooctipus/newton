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

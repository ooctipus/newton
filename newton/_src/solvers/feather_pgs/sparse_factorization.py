# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse triangular operations shared by warp and block factor producers."""


def _sparse_inverse_source(dof_count: int, *, local_column_width: int = 0) -> str:
    """Emit inverse columns while retaining the caller's execution mapping.

    The caller supplies ``indices`` (CSR/CSC topology), ``factor`` and
    ``inverse`` pointers, and ``lane``/``stride``. A positive column width keeps
    each column's prefix in thread-local storage and reports nonfinite values
    through the caller's shared ``bad`` flag. Otherwise prefixes use the shared
    inverse buffer. Synchronization and publication remain caller-owned.
    """
    if not 1 <= dof_count <= 64 or not 0 <= local_column_width <= dof_count:
        raise ValueError("Sparse inverse dimensions must fit the topology's 64-coordinate bound")
    scratch = f"float values[{local_column_width}];" if local_column_width else ""
    previous_value = "values[previous - begin]" if local_column_width else "inverse[other]"
    publish = ""
    if local_column_width:
        publish = "values[position - begin] = value;\n            if (!isfinite(value)) atomicExch(&bad, 1);"
    return f"""
    // Each inverse column follows only its ancestor chain.
    for (int col = lane; col < {dof_count}; col += stride) {{
        {scratch}
        const int begin = indices.column_offsets.data[col];
        const int end = indices.column_offsets.data[col + 1];
        for (int position = begin; position < end; ++position) {{
            const int target = indices.column_entries.data[position];
            const int row = indices.entry_rows.data[target];
            const int diagonal = indices.row_offsets.data[row + 1] - 1;
            float value = row == col ? 1.0f : 0.0f;
            for (int previous = begin; previous < position; ++previous) {{
                const int other = indices.column_entries.data[previous];
                const int factor_entry = diagonal - row + indices.entry_rows.data[other];
                value -= factor[factor_entry] * {previous_value};
            }}
            value /= factor[diagonal];
            {publish}
            inverse[target] = value;
        }}
    }}
"""

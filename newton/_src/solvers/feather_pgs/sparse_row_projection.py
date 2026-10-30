# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate factor-coordinate contact projection with an explicit operand mapping."""

from .device_preludes import substitute


def _sparse_contact_projection_source(
    mapping: str,
    *,
    dof_count: int,
    scratch_stride: int = 0,
    rows: int = 3,
) -> str:
    """Generate the sparse product of a contact's ``rows`` directions (three, four with a spin row), preserving
    reduction order.

    ``shared_mask`` consumes a Jacobian in shared scratch and walks the actual
    supporting factor entries of one output coordinate. Geometry, row
    allocation, norms and publication remain caller-owned.

    The generated code uses the surrounding producer's existing operand names
    and adds no launch, conversion, synchronization or intermediate array.
    """
    if not 1 <= dof_count <= 64:
        raise ValueError("Sparse projection requires 1 to 64 coordinates")
    if mapping != "shared_mask":
        raise ValueError("Use shared_mask sparse projection")
    if scratch_stride < dof_count:
        raise ValueError("Shared Jacobian storage must cover all coordinates")
    initialize = f"auto value = wp::vec{rows}(0.0f);"
    begin = """
        const int begin = factor_row_offsets.data[node];
        const int first = node - (factor_row_offsets.data[node + 1] - begin) + 1;
        // Reverse DFS packs this row's columns into [first, node].
        // Include bit 63 without a shift by 64.
        unsigned long long pending = mask & (~0ull << first) & (~0ull >> (63 - node));
        while (pending != 0) {
            const int column = FIRST(pending);
            const int entry = begin + column - first;
            const float factor = inverse_factor.data[factor_base + entry];
        """
    output, weight = "value.c[row]", "factor"
    operand = f"jacobian[row * {scratch_stride} + column]"
    # Preserve dynamic direction admission in the ordinary producer.
    accumulate = f"""
        #pragma unroll
            for (int row = 0; row < {rows}; ++row) {{
                if (row < row_count) {{ $ACCUMULATE }}
            }}
        """
    end = "pending &= pending - 1;\n        }"

    # One factor-product law over array bindings, which preserve ordinary direction admission.
    product = f"{output} += {weight} * {operand};"
    return initialize + begin + substitute(accumulate, ACCUMULATE=product) + end

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate factor-coordinate contact projection with explicit operand mappings."""


def _sparse_contact_projection_source(
    mapping: str,
    *,
    dof_count: int,
    support_width: int = 0,
    factor_stride: int = 0,
    scratch_stride: int = 0,
) -> str:
    """Generate the three-direction sparse product, preserving reduction order.

    ``shared_mask`` consumes a Jacobian in shared scratch and walks the actual
    supporting factor entries of one output coordinate. ``warp_catalogue``
    broadcasts a contact's catalogue columns across a full warp, with one
    output coordinate per lane. Every lane must enter its broadcasts, including
    lanes outside the output support. Both accumulate the same factor product;
    geometry, row allocation, norms and publication remain caller-owned.

    The generated code uses the surrounding producer's existing operand names
    and adds no launch, conversion, synchronization or intermediate array.
    """
    if not 1 <= dof_count <= 64:
        raise ValueError("Sparse projection requires 1 to 64 coordinates")

    if mapping == "shared_mask":
        if scratch_stride < dof_count:
            raise ValueError("Shared Jacobian storage must cover all coordinates")
        initialize = "auto value = wp::vec3(0.0f);"
        begin = """
        const int begin = factor_row_offsets.data[node];
        const int first = node - (factor_row_offsets.data[node + 1] - begin) + 1;
        // Reverse DFS packs this row's columns into [first, node].
        // Include bit 63 without a shift by 64.
        unsigned long long pending = mask & (~0ull << first) & (~0ull >> (63 - node));
        while (pending != 0) {
            const int column = first_bit(pending);
            const int entry = begin + column - first;
            const float factor = inverse_factor.data[factor_base + entry];
        """
        output, weight = "value.c[row]", "factor"
        operand = f"jacobian[row * {scratch_stride} + column]"
        # Preserve dynamic direction admission in the ordinary producer.
        accumulate = """
        #pragma unroll
            for (int row = 0; row < 3; ++row) {
                if (row < row_count) { $ACCUMULATE }
            }
        """
        end = "pending &= pending - 1;\n        }"
    elif mapping == "warp_catalogue":
        if not 1 <= support_width <= min(32, dof_count) or factor_stride < dof_count:
            raise ValueError("Warp catalogue requires bounded support and a complete factor")
        initialize = "float zn=0.0f, zt0=0.0f, zt1=0.0f;"
        begin = f"""
        for (int k=0; k<length; ++k) {{
            const float x=__shfl_sync(0xffffffff,jn,k), y=__shfl_sync(0xffffffff,jt0,k), z=__shfl_sync(0xffffffff,jt1,k);
            if (lane < length) {{
                const int col=p.support_nodes.data[tpl*{support_width}+k], e=p.index.data[node*{dof_count}+col];
                if (e >= 0) {{ const float a=d.W.data[group*{factor_stride}+e];
        """
        output, weight, operand = "$OUTPUT", "a", "$OPERAND"
        accumulate = "$ACCUMULATE"
        end = "}\n            }\n        }"
    else:
        raise ValueError("Use shared_mask or warp_catalogue sparse projection")

    # One factor-product law. Static scalar bindings preserve the register
    # producer; array bindings preserve ordinary direction admission.
    product = f"{output} += {weight} * {operand};"
    if mapping == "warp_catalogue":
        product = " ".join(
            product.replace("$OUTPUT", out).replace("$OPERAND", value)
            for out, value in zip(("zn", "zt0", "zt1"), ("x", "y", "z"), strict=True)
        )
    return initialize + begin + accumulate.replace("$ACCUMULATE", product) + end

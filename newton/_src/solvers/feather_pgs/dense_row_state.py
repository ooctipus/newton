# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lane-owned dense GS state with direct reads of the original lower C triangle."""

from functools import cache

import warp as wp


@cache
def get_dense_row_state_kernel(device_arch: str, worlds_per_block: int = 2, *, row_budget: int = 32) -> "wp.Kernel":
    """Build one half of the disjoint C64 dense-row solve.

    The caller must launch both budgets. Small worlds with out-of-prefix
    friction metadata retain full-capacity storage semantics in the 64 budget.
    Every lane owns one or two rows of impulse/metadata state. Matrix accesses
    use the same original lower-triangle entries, and the original serial
    terms per lane and full 32-lane reduction are retained.

    ``device_arch`` participates in the factory cache like other dense kernels.
    """
    if worlds_per_block not in (1, 2, 4) or row_budget not in (32, 64):
        raise ValueError("Dense register state requires budget 32/64 and 1, 2, or 4 worlds per block")
    parts = row_budget // 32

    def read(field, index):
        if parts == 1:
            return f"__shfl_sync(MASK, {field}_reg0, {index})"
        return f"__shfl_sync(MASK, ({index} < 32) ? {field}_reg0 : {field}_reg1, {index} & 31)"

    def write(index, value):
        if parts == 1:
            return f"if (lane == {index}) lam_reg0 = {value};"
        return f"if (lane == ({index} & 31)) {{ if ({index} < 32) lam_reg0 = {value}; else lam_reg1 = {value}; }}"

    declarations, loads, columns, dot_terms, stores = [], [], [], [], []
    for kind, name, field in (
        ("float", "lam", "world_impulses"),
        ("float", "rhs", "world_rhs"),
        ("float", "diag", "world_diag"),
        ("int", "rtype", "world_row_type"),
        ("int", "parent", "world_row_parent"),
        ("float", "mu", "world_row_mu"),
    ):
        for part in range(parts):
            declarations.append(f"    {kind} {name}_reg{part} = 0;")
            load = f"{name}_reg{part} = {field}.data[off1 + lane + {part * 32}];"
            loads.append(f"    if (lane < m) {{ {load} }}" if parts == 1 else f"    {load}")
    for part in range(parts):
        columns.append(f"    const int j{part} = lane + {part * 32};")
        dot_terms.append(f"""
            if (j{part} < m) {{
                float cij = (j{part} <= i)
                    ? world_C.data[off2 + i * ROW_STRIDE + j{part}]
                    : world_C.data[off2 + j{part} * ROW_STRIDE + i];
                my_sum += cij * lam_reg{part};
            }}""")
        store = f"world_impulses.data[off1 + lane + {part * 32}] = lam_reg{part};"
        stores.append(f"    if (lane < m) {{ {store} }}" if parts == 1 else f"    {store}")

    small_admission = "    if (m <= 0 || m > 32) return;" if parts == 1 else ""
    budget_admission = "    if (unsupported) return;" if parts == 1 else "    if (m <= 32 && !unsupported) return;"
    snippet = f"""
#if defined(__CUDA_ARCH__)
    const int ROW_STRIDE = 64;
    const int TILE_M_SQ = 4096;
    const unsigned MASK = 0xFFFFFFFF;
    int lane = threadIdx.x & 31;
    int m = world_constraint_count.data[world];
    if (m == 0) return;
{small_admission}

    // Match the existing partition, including the NaN denominator branch.
    bool unsupported = false;
    if (m <= 32) {{
        if (lane < m && !(world_diag.data[world * ROW_STRIDE + lane] <= 0.0f) &&
            world_row_type.data[world * ROW_STRIDE + lane] == 2) {{
            const int p = world_row_parent.data[world * ROW_STRIDE + lane];
            unsupported = p < 0 || p > m - 3 || (lane != p + 1 && lane != p + 2);
        }}
        unsupported = __ballot_sync(MASK, unsupported) != 0u;
    }}
{budget_admission}

{chr(10).join(declarations)}
    int off1 = world * ROW_STRIDE;
    int off2 = world * TILE_M_SQ;
{chr(10).join(loads)}
{chr(10).join(columns)}

    for (int iter = 0; iter < iterations; iter++) {{
        int global_iter = iteration_offset + iter;
        for (int i = 0; i < m; i++) {{
            float my_sum = 0.0f;
{chr(10).join(dot_terms)}
            my_sum += __shfl_down_sync(MASK, my_sum, 16);
            my_sum += __shfl_down_sync(MASK, my_sum, 8);
            my_sum += __shfl_down_sync(MASK, my_sum, 4);
            my_sum += __shfl_down_sync(MASK, my_sum, 2);
            my_sum += __shfl_down_sync(MASK, my_sum, 1);
            float dot_sum = __shfl_sync(MASK, my_sum, 0);

            float denom = {read("diag", "i")};
            if (denom <= 0.0f) continue;
            float rhs_i = {read("rhs", "i")};
            float lam_i = {read("lam", "i")};
            float w_val = rhs_i + dot_sum;
            float delta = -w_val / denom;
            float new_impulse = lam_i + omega * delta;
            int row_type = {read("rtype", "i")};
            if (row_type == 2 && global_iter < friction_start_iteration) {{
                {write("i", "0.0f")}
                continue;
            }}
            if (row_type == 0 || row_type == 3 || row_type == 4) {{
                if (new_impulse < 0.0f) new_impulse = 0.0f;
                {write("i", "new_impulse")}
            }} else if (row_type == 2) {{
                int parent_idx = {read("parent", "i")};
                float lambda_n = {read("lam", "parent_idx")};
                float mu = {read("mu", "i")};
                float radius = fmaxf(mu * lambda_n, 0.0f);
                if (radius <= 0.0f) {{
                    {write("i", "0.0f")}
                }} else {{
                    int sib = (i == parent_idx + 1) ? (parent_idx + 2) : (parent_idx + 1);
                    float a = new_impulse;
                    float b = {read("lam", "sib")};
                    float mag = sqrtf(a * a + b * b);
                    if (mag > radius) {{
                        float scale = radius / mag;
                        {write("i", "a * scale")}
                        {write("sib", "b * scale")}
                    }} else {{
                        {write("i", "a")}
                    }}
                }}
            }} else {{
                {write("i", "new_impulse")}
            }}
        }}
    }}
{chr(10).join(stores)}
#endif
"""

    @wp.func_native(snippet)
    def solve(
        world: int,
        world_constraint_count: wp.array[int],
        world_diag: wp.array2d[float],
        world_C: wp.array3d[float],
        world_rhs: wp.array2d[float],
        world_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        friction_start_iteration: int,
        iteration_offset: int,
    ): ...

    def grouped(
        world_constraint_count: wp.array[int],
        world_diag: wp.array2d[float],
        world_C: wp.array3d[float],
        world_rhs: wp.array2d[float],
        world_impulses: wp.array2d[float],
        iterations: int,
        omega: float,
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        friction_start_iteration: int,
        iteration_offset: int,
    ):
        block, thread = wp.tid()
        world = block * wp.static(worlds_per_block) + thread // 32
        if world >= world_constraint_count.shape[0]:
            return
        solve(
            world,
            world_constraint_count,
            world_diag,
            world_C,
            world_rhs,
            world_impulses,
            iterations,
            omega,
            world_row_type,
            world_row_parent,
            world_row_mu,
            friction_start_iteration,
            iteration_offset,
        )

    grouped.__name__ = f"pgs_solve_dense_row_state_b{row_budget}_w{worlds_per_block}"
    grouped.__qualname__ = grouped.__name__
    return wp.kernel(enable_backward=False, module="unique")(grouped)

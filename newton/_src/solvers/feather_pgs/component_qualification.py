# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared component predicates with explicit physical, held and hybrid bindings."""

_PHYSICAL_PUBLISH = r"""        if (accepted) {
            for (int row = warp; row < m; row += nw) {
                const int base = (world * M + row) * 29 + po;
#if defined(__CUDA_ARCH__)
                const float y = lane < 23 ? data.Y.data[base + lane] : 0.0f;
                float z = 0.0f;
                for (int k = 0; k < 23; ++k) {
                    const float yk = __shfl_sync(0xffffffffu, y, k);
                    if (lane < 23 && lane <= k)
                        z += data.L.data[group * 23 * 23 + k * 23 + lane] * yk;
                }
                if (lane < 23) data.Y.data[base + lane] = z;
#else
                float y[23];
                for (int k = 0; k < 23; ++k) y[k] = data.Y.data[base + k];
                for (int d = 0; d < 23; ++d) {
                    float z = 0.0f;
                    for (int k = d; k < 23; ++k)
                        z += data.L.data[group * 23 * 23 + k * 23 + d] * y[k];
                    data.Y.data[base + d] = z;
                }
#endif
            }
            if (tid == 0) data.selector.data[world] = -1;
        }
"""


def get_source(layout="physical"):
    """Keep one predicate owner while selecting its current representation bindings."""
    if layout not in ("physical", "kinetic", "hybrid"):
        raise ValueError("unknown component qualification representation")
    held = layout != "physical"
    hybrid = layout == "hybrid"
    bindings = {
        "secondary_offset": "data.secondary_offset",
        "primary_offset": "data.primary_offset",
        "primary_group": "data.primary_group",
        "world_count": "data.world_count",
        "row_parent": "data.row_parent",
        "mf_counts": "data.mf_counts",
        "mf_MiJt_a": "data.mf_MiJt_a",
        "mf_MiJt_b": "data.mf_MiJt_b",
        "row_type": "data.row_type",
        "mf_meta": "data.mf_meta",
        "counts": "data.counts",
        "mf_J_a": "data.mf_J_a",
        "mf_J_b": "data.mf_J_b",
        "mf_mu": "data.mf_mu",
        "diag": "data.diag",
        "rhs": "data.rhs",
        "mu": "data.mu",
        "Y": "data.Y",
        "J": "data.J",
    }
    if held:
        bindings.update(
            {
                "world_count": "state.dense_count.shape[0]",
                "counts": "state.dense_count",
                "mf_counts": "state.mf_count",
                "primary_offset": "state.primary_offset",
                "secondary_offset": "state.secondary_offset",
                "primary_group": "plan.primary_group",
                "selector": "solve.selector",
                "J": "rows.physical_J",
                "Y": "rows.response",
                "rhs": "rows.rhs",
                "diag": "rows.diag",
                "mu": "rows.row_mu",
                "row_type": "rows.row_type",
                "row_parent": "rows.row_parent",
                "mf_meta": "mf.meta",
                "mf_mu": "mf.mu",
                "mf_J_a": "mf.J_a",
                "mf_J_b": "mf.J_b",
                "mf_MiJt_a": "mf.MiJt_a",
                "mf_MiJt_b": "mf.MiJt_b",
            }
        )
    mf_count = "mf_count" if held else "mf"
    selector = "solve.selector" if held else "data.selector"
    selector_initialize = f"if (tid == 0) {selector}.data[world] = {mf_count} == 0 ? 0 : wp::max({mf_count}, 1);"
    accepted_initialize = "if (tid == 0) accepted = 1;"
    predicate_variables = "int valid = 1;"
    secondary_predicate = "if (d >= so && d < so + 6) valid &= j == 0.0f && y == 0.0f;"
    mf_kind = "const int tp = " + ("mf.meta" if held else "data.mf_meta") + ".data[r * 4 + 3], type = tp & 65535;"
    cuda_accept = "if (!valid) atomicExch(&accepted, 0);"
    cpu_accept = "accepted = valid;"
    operator_guard = "|| group < 0 || group >= data.L.shape[0]) continue;"
    operator_length = "23 * 23; k += nt"
    operator_value = "data.L.data[group * 23 * 23 + k]"
    publish = _PHYSICAL_PUBLISH
    if held:
        operator_guard = ") continue;\n        if (held.valid.data[world]==0 || held.generation.data[world]!=state.held_generation.data[world] || rows.status.data[world]!=0 || rows.global_status.data[0]!=0) { if(tid==0) solve.status.data[world]=1; continue; }"
        operator_length = "180; k += nt"
        operator_value = "held.T.data[world * 180 + k]"
        publish = "        if (accepted) {\n            if (tid == 0) solve.selector.data[world] = -1;\n        }\n"
    if hybrid:
        selector_initialize = "if (tid == 0) { solve.selector.data[world] = mf_count == 0 ? 0 : wp::max(mf_count, 1); hybrid.eligible.data[world] = 0; }"
        accepted_initialize = "if (tid == 0) accepted = 3;"
        predicate_variables = "int valid = 1, independent = 1, hybrid_route = 1;"
        secondary_predicate = "if (d >= so && d < so + 6) independent &= j == 0.0f && y == 0.0f;"
        mf_kind += "\n            hybrid_route &= (a == so) != (b == so);"
        cuda_accept = "const int keep = valid ? ((independent ? 1 : 0) | (hybrid_route ? 2 : 0)) : 0;\n        if (keep != 3) atomicAnd(&accepted, keep);"
        cpu_accept = "accepted = valid ? ((independent ? 1 : 0) | (hybrid_route ? 2 : 0)) : 0;"
        publish = "        if (tid == 0) {\n            if (accepted & 1) solve.selector.data[world] = -1;\n            else if ((accepted & 2) && m > 0 && state.resolved.data[world] == 0) hybrid.eligible.data[world] = 1;\n        }\n"
    return f"""
#if defined(__CUDA_ARCH__)
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nw = nt / 32;
    __shared__ int accepted;
#else
    if (logical_lane != 0) return;
    const int tid = 0, nt = 1, lane = 0, warp = 0, nw = 1;
    int accepted;
#endif
    const int M = {bindings["J"]}.shape[1];
    const int F = {bindings["mf_meta"]}.shape[1] / 4;
    for (int world = worker; world < {bindings["world_count"]}; world += workers) {{
        const int m = {bindings["counts"]}.data[world];
        const int {mf_count} = {bindings["mf_counts"]}.data[world];
        {selector_initialize}
        if ({mf_count} == 0) continue;
        if (m < 0 || m > M || {mf_count} < 1 || {mf_count} > F) continue;
        const int po = {bindings["primary_offset"]}.data[world];
        const int so = {bindings["secondary_offset"]}.data[world];
        const int group = {bindings["primary_group"]}.data[world];
        if (!((po == 0 && so == 23) || (po == 6 && so == 0))
            {operator_guard}
        {accepted_initialize}
#if defined(__CUDA_ARCH__)
        __syncthreads();
#endif
        {predicate_variables}
        for (int k = tid; k < {operator_length}) {{
            const float x = {operator_value};
            valid &= wp::isfinite(x) && wp::abs(x) <= 1.0e15f;
        }}
        for (int row = tid; row < m; row += nt) {{
            const int r = world * M + row;
            const int type = {bindings["row_type"]}.data[r];
            const float diag = {bindings["diag"]}.data[r];
            const float mu = {bindings["mu"]}.data[r];
            valid &= (type == 0 || type == 2 || type == 3)
                && wp::isfinite(diag) && diag > 0.0f
                && wp::isfinite(mu) && mu >= 0.0f && wp::isfinite({bindings["rhs"]}.data[r]);
            if (type == 2) {{
                const int p = {bindings["row_parent"]}.data[r];
                if (p < 0 || p + 2 >= m || (row != p + 1 && row != p + 2)) valid = 0;
                else {{
                    const int base = world * M + p;
                    valid &= {bindings["row_type"]}.data[base] == 0
                        && {bindings["row_type"]}.data[base + 1] == 2 && {bindings["row_type"]}.data[base + 2] == 2
                        && {bindings["row_parent"]}.data[base + 1] == p && {bindings["row_parent"]}.data[base + 2] == p;
                }}
            }}
            for (int d = 0; d < 29; ++d) {{
                const float j = {bindings["J"]}.data[r * 29 + d], y = {bindings["Y"]}.data[r * 29 + d];
                valid &= wp::isfinite(j) && wp::isfinite(y) && wp::abs(y) <= 1.0e15f;
                {secondary_predicate}
            }}
        }}
        for (int row = tid; row < {mf_count}; row += nt) {{
            const int r = world * F + row;
            const int packed = {bindings["mf_meta"]}.data[r * 4];
            const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535));
            {mf_kind}
            union {{ int bits; float value; }} diag, rhs;
            diag.bits = {bindings["mf_meta"]}.data[r * 4 + 1];
            rhs.bits = {bindings["mf_meta"]}.data[r * 4 + 2];
            const float mu = {bindings["mf_mu"]}.data[r];
            valid &= (a == -1 || a == so) && (b == -1 || b == so) && (a == so || b == so)
                && (type == 0 || type == 2 || type == 4)
                && wp::isfinite(diag.value) && diag.value > 0.0f && wp::isfinite(rhs.value)
                && wp::isfinite(mu) && mu >= 0.0f;
            if (type == 2) {{
                const int encoded_parent = static_cast<unsigned int>(tp) >> 16;
                const int p = encoded_parent == 65535 ? -1 : encoded_parent;
                if (p < 0 || p + 2 >= {mf_count} || (row != p + 1 && row != p + 2)) valid = 0;
                else {{
                    const int base = (world * F + p) * 4 + 3;
                    const int t1 = {bindings["mf_meta"]}.data[base + 4], t2 = {bindings["mf_meta"]}.data[base + 8];
                    valid &= ({bindings["mf_meta"]}.data[base] & 65535) == 0
                        && (t1 & 65535) == 2 && (t2 & 65535) == 2
                        && static_cast<int>(static_cast<unsigned int>(t1) >> 16) == p
                        && static_cast<int>(static_cast<unsigned int>(t2) >> 16) == p;
                }}
            }}
            for (int d = 0; d < 6; ++d) {{
                valid &= wp::isfinite({bindings["mf_J_a"]}.data[r * 6 + d])
                    && wp::isfinite({bindings["mf_J_b"]}.data[r * 6 + d])
                    && wp::isfinite({bindings["mf_MiJt_a"]}.data[r * 6 + d])
                    && wp::isfinite({bindings["mf_MiJt_b"]}.data[r * 6 + d]);
            }}
        }}
#if defined(__CUDA_ARCH__)
        {cuda_accept}
        __syncthreads();
#else
        {cpu_accept}
#endif
{publish}#if defined(__CUDA_ARCH__)
        // Do not reuse the shared admission word while another warp consumes it.
        __syncthreads();
#endif
    }}
"""

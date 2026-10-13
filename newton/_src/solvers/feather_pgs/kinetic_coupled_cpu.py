# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share scalar coupled recurrence with explicit secondary-column preparation."""

from string import Template

from .friction import get_pair_source

_HELD_SECONDARY = r"""        for (int col = 0; col < 6; ++col) {
            float y = 0.0f;
            for (int k = col; k < 6; ++k) y += held.inverse6.data[group * 36 + k * 6 + col]
                * rows.response.data[(world * 192 + r) * 29 + so + k];
            ys[r * 6 + col] = y; good &= isfinite(y);
        }
    }
"""

_CONVERTED_SECONDARY = r"""
        for(int col=0;col<6;++col){const float y=rows.response.data[(world*192+r)*29+so+col];ys[r*6+col]=y;good &= isfinite(y);}
    }
"""

_BODY = Template(r"""
    const int m = world_constraint_count.data[world], n = mf_constraint_count.data[world];
    const int po = state.primary_offset.data[world], so = state.secondary_offset.data[world];
    const int group = plan.secondary_group.data[world], mend = mf_contact_rows_end.data[world];
    float du[29] = {}, lam[192], ml[64], ys[192 * 6], base[6];
    bool good = true;
    for (int k = 0; k < 6; ++k) base[k] = state.v_hat.data[plan.dof_ids.data[world * 35 + 23 + k]];
    for (int r = 0; r < m; ++r) {
        lam[r] = world_impulses.data[world * 192 + r];
${secondary_prepare}    for (int r = 0; r < n; ++r) ml[r] = mf_impulses.data[world * 64 + r];
    if (!good) { solve.status.data[world] = 2; return; }
    auto dense_update = [&](int row, float delta) {
        if (delta != 0.0f) for (int d = 0; d < 29; ++d) {
            const float y = d >= po && d < po + 23 ? rows.response.data[(world * 192 + row) * 29 + d] : ys[row * 6 + d - so];
            du[d] += y * delta;
        }
    };
    auto mf_update = [&](int row, float delta) {
        if (delta == 0.0f) return;
        const int id = world * 64 + row, packed = mf_meta.data[id * 4];
        const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535));
        for (int k = 0; k < 6; ++k) {
            if (a >= 0) du[a + k] += mf_MiJt_a.data[id * 6 + k] * delta;
            if (b >= 0) du[b + k] += mf_MiJt_b.data[id * 6 + k] * delta;
        }
    };
    for (int iter = 0; iter < iterations; ++iter) {
        bool changed = false; const int global_iter = iter + iteration_offset;
        for (int r = 0; r < m; ++r) {
            const int id = world * 192 + r, type = world_row_type.data[id];
            if (type == 2 && global_iter < friction_start_iteration) { lam[r] = 0.0f; continue; }
            if (type == 2 && r != world_row_parent.data[id] + 1) continue;
            const float denom = world_diag.data[id]; if (denom <= 0.0f && type != 2) continue;
            float dot = 0.0f;
            for (int d = 0; d < 29; ++d) {
                const float j = d >= po && d < po + 23 ? rows.response.data[id * 29 + d] : J_world.data[id * 29 + d];
                dot += j * du[d];
            }
            const float residual = dot + rows.r0.data[id], old = lam[r];
            float next = old - (denom > 0.0f ? residual / denom : 0.0f), sibling_delta = 0.0f; int sibling = -1;
            if (type == 0 || type == 3) next = wp::max(next, 0.0f);
            else if (type == 2) {
                const int p = world_row_parent.data[id]; sibling = p + 2;
                float normal_load = lam[p];
                for (int patch = world_row_parent.data[world * 192 + p]; patch >= 0 && patch != p; patch = world_row_parent.data[world * 192 + patch]) normal_load += lam[patch];
                const float radius = wp::max(world_row_mu.data[id] * normal_load, 0.0f);
                const int sid = world * 192 + sibling;
                float other_residual = rows.r0.data[sid], cross = 0.0f;
                for (int d = 0; d < 29; ++d) {
                    const bool robot = d >= po && d < po + 23;
                    const float j = robot ? rows.response.data[id * 29 + d] : J_world.data[id * 29 + d];
                    const float js = robot ? rows.response.data[sid * 29 + d] : J_world.data[sid * 29 + d];
                    const float ysib = robot ? rows.response.data[sid * 29 + d] : ys[sibling * 6 + d - so];
                    other_residual += js * du[d]; cross += j * ysib;
                }
                const auto pair = friction_pair_candidate(denom, cross, world_diag.data[sid], residual, other_residual, old, lam[sibling], radius, omega);
                const float length = sqrtf(pair[0] * pair[0] + pair[1] * pair[1]);
                const float scale = length > radius ? radius / length : 1.0f;
                next = pair[0] * scale; const float other = pair[1] * scale;
                sibling_delta = other - lam[sibling]; lam[sibling] = other;
            }
            if (sibling_delta != 0.0f) { changed = true; dense_update(sibling, sibling_delta); }
            lam[r] = next; const float delta = next - old;
            if (delta != 0.0f) { changed = true; dense_update(r, delta); }
        }
        for (int pass = 0; pass < 2; ++pass) {
            const int lo = pass == 0 ? 0 : mend, hi = n;
            for (int r = lo; r < hi; ++r) {
                const int id = world * 64 + r, packed = mf_meta.data[id * 4], tp = mf_meta.data[id * 4 + 3];
                const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535)), type = tp & 65535;
                if ((pass == 0 && type == 4) || (pass == 1 && type != 4)) continue;
                if (type == 2 && global_iter < friction_start_iteration) { ml[r] = 0.0f; continue; }
                if (type == 2 && r != (tp >> 16) + 1) continue;
                union { int bits; float value; } inv, rhs;
                inv.bits = mf_meta.data[id * 4 + 1]; rhs.bits = mf_meta.data[id * 4 + 2];
                if (inv.value <= 0.0f && type != 2) continue;
                float dot = 0.0f;
                for (int k = 0; k < 6; ++k) {
                    if (a >= 0) dot += mf_J_a.data[id * 6 + k] * (du[a + k] + base[k]);
                    if (b >= 0) dot += mf_J_b.data[id * 6 + k] * (du[b + k] + base[k]);
                }
                const float residual = dot + rhs.value, delta0 = -residual * inv.value, old = ml[r];
                float next = old + delta0, delta = 0.0f;
                if (type == 4) { delta = residual < 0.0f ? delta0 : 0.0f; next = delta; }
                else if (type == 0) next = wp::max(next, 0.0f);
                else if (type == 2) {
                    const int p = tp >> 16, sibling = p + 2, sid = world * 64 + sibling;
                    float normal_load = ml[p];
                    for (int patch = mf_meta.data[(world * 64 + p) * 4 + 3] >> 16; patch >= 0 && patch != p; patch = mf_meta.data[(world * 64 + patch) * 4 + 3] >> 16) normal_load += ml[patch];
                    const float radius = wp::max(mf_row_mu.data[id] * normal_load, 0.0f);
                    union { int bits; float value; } other_inv, other_rhs;
                    other_inv.bits = mf_meta.data[sid * 4 + 1]; other_rhs.bits = mf_meta.data[sid * 4 + 2];
                    float other_residual = other_rhs.value, cross = 0.0f;
                    for (int k = 0; k < 6; ++k) {
                        if (a >= 0) { other_residual += mf_J_a.data[sid * 6 + k] * (du[a + k] + base[k]); cross += mf_J_a.data[id * 6 + k] * mf_MiJt_a.data[sid * 6 + k]; }
                        if (b >= 0) { other_residual += mf_J_b.data[sid * 6 + k] * (du[b + k] + base[k]); cross += mf_J_b.data[id * 6 + k] * mf_MiJt_b.data[sid * 6 + k]; }
                    }
                    const auto pair = friction_pair_candidate(inv.value > 0.0f ? 1.0f / inv.value : 0.0f, cross,
                        other_inv.value > 0.0f ? 1.0f / other_inv.value : 0.0f, residual, other_residual, old, ml[sibling], radius, omega);
                    const float length = sqrtf(pair[0] * pair[0] + pair[1] * pair[1]);
                    const float scale = length > radius ? radius / length : 1.0f;
                    next = pair[0] * scale; const float other = pair[1] * scale, sd = other - ml[sibling]; ml[sibling] = other;
                    if (sd != 0.0f) { changed = true; mf_update(sibling, sd); }
                }
                if (type != 4) delta = next - old;
                ml[r] = next;
                if (delta != 0.0f) { changed = true; mf_update(r, delta); }
            }
        }
        if (global_iter >= friction_start_iteration && !changed) break;
    }
    float output[29];
    for (int d = 0; d < 29; ++d) {
        float delta = 0.0f;
        if (d < 23) for (int k = 0; k < 23; ++k) delta += t(k, d) * du[po + k];
        else delta = du[so + d - 23];
        output[d] = state.v_hat.data[plan.dof_ids.data[world * 35 + d]] + delta;
        good &= isfinite(output[d]);
    }
    for (int r = 0; r < m; ++r) good &= isfinite(lam[r]);
    for (int r = 0; r < n; ++r) good &= isfinite(ml[r]);
    if (!good) { solve.status.data[world] = 2; return; }
    for (int d = 0; d < 29; ++d) v_out.data[plan.dof_ids.data[world * 35 + d]] = output[d];
    for (int r = 0; r < m; ++r) world_impulses.data[world * 192 + r] = lam[r];
    for (int r = 0; r < n; ++r) mf_impulses.data[world * 64 + r] = ml[r];
    solve.status.data[world] = 0;
""")


def get_source(*, converted_secondary=False):
    """Consume current converted columns or perform the original held bridge."""
    secondary = _CONVERTED_SECONDARY if converted_secondary else _HELD_SECONDARY
    return get_pair_source(cpu=True) + _BODY.substitute(secondary_prepare=secondary)

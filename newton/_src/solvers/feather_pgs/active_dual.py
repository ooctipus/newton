# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in active-dual Coulomb solve for the ordinary ANYmal18 parallel tiers."""

import ast
import functools
import hashlib
import inspect
import linecache
import os
import textwrap

_ORIGINAL_FACTORY_SHA = "2d195e7d37f5b9e127a75d2dd2be6185f78faac4f2d5bd29f1603f03d9745a48"


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Active-dual original native seam changed")
    return source.replace(old, new)


_NATIVE = r"""
    // All allocations are per solve. The original factor becomes LU scratch
    // only after whitening, and is reloaded before either final decode path.
    __shared__ float ad_gram[171];
    __shared__ int ad_ids[18], ad_old_ids[18], ad_map[AM];
    __shared__ float ad_q[AM], ad_pg[AM], ad_delta[AM], ad_lu_rhs[18];
    __shared__ int ad_nactive, ad_old_count, ad_rebuild, ad_lu_ok;
    int ad_consumed = 0;
    bool ad_done = false;
    bool ad_factor_dirty = false;
    const int ad_i = lane;
    const bool ad_row = ad_i < n_rows;
    float ad_diagonal = 0.0f, ad_spectral = 0.0f, ad_eta = 0.0f;

    auto ad_any = [&](int value) {
        if (NT == 32) return __any_sync(MASK, value);
        return __syncthreads_or(value);
    };
    auto ad_sum = [&](float value) {
        for (int shift = 16; shift > 0; shift >>= 1)
            value += __shfl_xor_sync(MASK, value, shift);
        if (NT > 32) {
            if ((lane & 31) == 0) s_dot[lane >> 5] = value;
            SYNC();
            value = s_dot[0] + s_dot[1];
            SYNC();
        }
        return value;
    };
    auto ad_max = [&](float value) {
        for (int shift = 16; shift > 0; shift >>= 1)
            value = fmaxf(value, __shfl_xor_sync(MASK, value, shift));
        if (NT > 32) {
            if ((lane & 31) == 0) s_dot[lane >> 5] = value;
            SYNC();
            value = fmaxf(s_dot[0], s_dot[1]);
            SYNC();
        }
        return value;
    };
    auto ad_residual = [&](const float* point, bool with_rhs) {
        // Same chunked factor-space contraction as the retained MF owner.
        constexpr int NCH = NT / 18;
        const int n4 = (n_rows + 3) >> 2;
        const int per = (n4 + NCH - 1) / NCH;
        if (lane < NCH * 18) {
            const int d = lane % 18;
            const int ch = lane / 18;
            const float4* x4 = reinterpret_cast<const float4*>(point);
            const float4* z4 = reinterpret_cast<const float4*>(&s_Yt[d * YS]);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int j = ch * per; j < min(n4, (ch + 1) * per); ++j) {
                const float4 xv = x4[j], zv = z4[j];
                a0 += xv.x * zv.x; a1 += xv.y * zv.y;
                a2 += xv.z * zv.z; a3 += xv.w * zv.w;
            }
            if (NCH == 1) s_dv[d] = (a0 + a1) + (a2 + a3);
            else s_dvp[ch * 18 + d] = (a0 + a1) + (a2 + a3);
        }
        SYNC();
        if (NCH > 1) {
            if (lane < 18) {
                float value = 0.0f;
                #pragma unroll
                for (int ch = 0; ch < NCH; ++ch) value += s_dvp[ch * 18 + lane];
                s_dv[lane] = value;
            }
            SYNC();
        }
        float value = ad_row && with_rhs ? s_rhs[ad_i] : 0.0f;
        if (ad_row) {
            #pragma unroll
            for (int d = 0; d < 18; ++d) value += Jr[d] * s_dv[d];
        }
        // A later invocation may overwrite the shared18-vector.
        SYNC();
        return value;
    };
    auto ad_project = [&](float* point) {
        float value = ad_row ? point[ad_i] : 0.0f;
        if (ad_row && s_kind[ad_i] == 0) value = fmaxf(value, 0.0f);
        if (ad_row) point[ad_i] = value;
        SYNC();
        if (ad_row && s_kind[ad_i] == 1) {
            const int p = s_parent[ad_i];
            const int sibling = ad_i == p + 1 ? p + 2 : p + 1;
            const float radius = fmaxf(s_mu[ad_i] * point[p], 0.0f);
            const float other = point[sibling];
            const float length = sqrtf(value * value + other * other);
            if (radius <= 0.0f) value = 0.0f;
            else if (length > radius) value *= radius / length;
        }
        // Both siblings must finish reading before either overwrites its value.
        SYNC();
        if (ad_row) point[ad_i] = value;
        SYNC();
    };
    auto ad_map_value = [&](const float* point, float residual) {
        if (ad_row) ad_q[ad_i] = point[ad_i] - ad_eta * residual;
        SYNC();
        float value = ad_row ? fmaxf(ad_q[ad_i], 0.0f) : 0.0f;
        if (ad_row && s_kind[ad_i] == 1) {
            const int p = s_parent[ad_i];
            const int sibling = ad_i == p + 1 ? p + 2 : p + 1;
            const float radius = s_mu[ad_i] * fmaxf(ad_q[p], 0.0f);
            const float qi = ad_q[ad_i], qs = ad_q[sibling];
            const float length = sqrtf(qi * qi + qs * qs);
            value = qi;
            if (radius <= 0.0f) value = 0.0f;
            else if (length > radius) value *= radius / length;
        }
        if (ad_row) ad_pg[ad_i] = value;
        SYNC();
        return ad_row ? (point[ad_i] - value) / ad_eta : 0.0f;
    };
    auto ad_g = [&](int a, int b) {
        const int hi = max(a, b), lo = min(a, b);
        return ad_gram[hi * (hi + 1) / 2 + lo];
    };

    // Staging has preserved the original reference. All unsupported states
    // reach the literal original majorizer/recurrence before any impulse change.
    int ad_bad = row_phase != 0 || freeze_drive_rows != 0
        || friction_start_iteration != 0 || iteration_offset != 0
        || regularize != 0 || omega != 1.0f;
    if (ad_row) {
        ad_bad |= !isfinite(s_lam0[ad_i]) || s_lam0[ad_i] != 0.0f
            || !isfinite(s_rhs[ad_i]) || s_kind[ad_i] < 0 || s_kind[ad_i] > 1;
        for (int d = 0; d < 18; ++d) {
            ad_diagonal += Jr[d] * Jr[d];
            ad_bad |= !isfinite(Jr[d]);
        }
        const float cfm = s_row_c[ad_i] >= 0 ? wr_pgs_cfm : world_row_cfm.data[off_dense + ad_i];
        if (s_row_c[ad_i] >= 0) {
            const int contact = s_row_c[ad_i] >> 2;
            const int art_a = contact_art_a.data[contact], art_b = contact_art_b.data[contact];
            ad_bad |= art_a >= 0 && art_a == art_b;
        }
        ad_bad |= !isfinite(cfm) || cfm < 0.0f || !(ad_diagonal > 0.0f) || !isfinite(ad_diagonal);
        ad_eta = 1.0f / (ad_diagonal + cfm);
        if (s_kind[ad_i] == 1) {
            const int p = s_parent[ad_i];
            const bool pair = p >= 0 && p + 2 < n_rows && (ad_i == p + 1 || ad_i == p + 2);
            ad_bad |= !pair;
            if (pair) {
                ad_bad |= s_kind[p] != 0 || s_kind[p + 1] != 1 || s_kind[p + 2] != 1
                    || s_parent[p + 1] != p || s_parent[p + 2] != p
                    || !isfinite(s_mu[ad_i]) || s_mu[ad_i] < 0.0f
                    || s_mu[p + 1] != s_mu[p + 2];
                float a = 0.0f, b = 0.0f, c = 0.0f;
                for (int d = 0; d < 18; ++d) {
                    const float za = s_Yt[d * YS + p + 1], zb = s_Yt[d * YS + p + 2];
                    a += za * za; b += za * zb; c += zb * zb;
                }
                ad_spectral = 0.5f * (a + c + sqrtf((a - c) * (a - c) + 4.0f * b * b));
                const float c0 = s_row_c[p + 1] >= 0 ? wr_pgs_cfm : world_row_cfm.data[off_dense + p + 1];
                const float c1 = s_row_c[p + 2] >= 0 ? wr_pgs_cfm : world_row_cfm.data[off_dense + p + 2];
                ad_eta = 1.0f / (ad_spectral + fmaxf(c0, c1));
                ad_bad |= !(ad_spectral > 0.0f) || !isfinite(ad_spectral);
            }
        }
        ad_bad |= !(ad_eta > 0.0f) || !isfinite(ad_eta);
    }
    const bool ad_admitted = !ad_any(ad_bad);
    if (ad_admitted) {
        if (lane == 0) ad_old_count = -1;
        if (lane < 18) ad_old_ids[lane] = -1;
        for (int j = n_rows + lane; j < ((n_rows + 3) & ~3); j += NT) {
            s_x[j] = 0.0f; s_y[j] = 0.0f;
        }
        // Per-row step is also available to the normal-radius derivative.
        if (ad_row) s_step[ad_i] = ad_eta;
        SYNC();
        for (int ad_outer = 0; ad_outer < 24; ++ad_outer) {
            const float residual = ad_residual(s_x, true);
            const float natural = ad_map_value(s_x, residual);
            const float projected_direction = ad_row ? ad_pg[ad_i] - s_x[ad_i] : 0.0f;
            const float scale = ad_max(ad_row ? fmaxf(fabsf(s_rhs[ad_i]), fabsf(residual - s_rhs[ad_i])) : 0.0f);
            bool physical_bad = false;
            if (ad_row) {
                const float x = s_x[ad_i];
                physical_bad = !isfinite(residual) || !isfinite(x) || !isfinite(natural);
                if (s_kind[ad_i] == 0) {
                    const float defect = ad_diagonal * (x - fmaxf(0.0f, x - residual / ad_diagonal));
                    physical_bad |= fabsf(defect) >= 3.0e-5f * fmaxf(1.0f, scale)
                        || -residual >= 3.0e-5f || fabsf(x * residual) >= 3.0e-5f || -x >= 1.0e-7f;
                }
                // The sibling residual is needed only by the physical pair check.
                ad_delta[ad_i] = residual;
            }
            SYNC();
            if (ad_row && s_kind[ad_i] == 1 && ad_i == s_parent[ad_i] + 1) {
                const int p = s_parent[ad_i], sibling = ad_i + 1;
                const float x = s_x[ad_i], y = s_x[sibling];
                const float r = residual, t = ad_delta[sibling];
                const float radius = fmaxf(0.0f, s_mu[ad_i] * s_x[p]);
                float qx = x - r / ad_spectral, qy = y - t / ad_spectral;
                const float length = sqrtf(qx * qx + qy * qy);
                if (radius <= 0.0f) { qx = 0.0f; qy = 0.0f; }
                else if (length > radius) { const float factor = radius / length; qx *= factor; qy *= factor; }
                const float defect = ad_spectral * fmaxf(fabsf(x - qx), fabsf(y - qy));
                const float mdp = x * r + y * t + radius * sqrtf(r * r + t * t);
                physical_bad |= defect >= 3.0e-5f * fmaxf(1.0f, scale)
                    || mdp >= 3.0e-5f || sqrtf(x * x + y * y) - radius >= 3.0e-5f
                    || !isfinite(defect) || !isfinite(mdp);
            }
            if (!ad_any(physical_bad)) { ad_done = true; break; }
            const float old_merit = ad_sum(natural * natural);
            const bool active = ad_row && (s_kind[ad_i] == 0 ? ad_q[ad_i] > 0.0f
                : s_mu[ad_i] * fmaxf(ad_q[s_parent[ad_i]], 0.0f) > 0.0f);
            if (ad_row) ad_map[ad_i] = active ? 1 : 0;
            SYNC();
            if (lane == 0) {
                int count = 0;
                for (int i = 0; i < n_rows; ++i) {
                    if (ad_map[i]) {
                        if (count < 18) ad_ids[count] = i;
                        ad_map[i] = count++;
                    } else ad_map[i] = -1;
                }
                ad_nactive = count;
                ad_rebuild = count != ad_old_count;
                if (count <= 18) {
                    for (int j = 0; j < count; ++j) ad_rebuild |= ad_ids[j] != ad_old_ids[j];
                    if (ad_rebuild) for (int j = 0; j < count; ++j) ad_old_ids[j] = ad_ids[j];
                    ad_old_count = count;
                }
            }
            SYNC();
            // A late growth handoff has only24-ad_consumed original sweeps.
            if (ad_nactive > 18 || !isfinite(old_merit)) break;
            const int na = ad_nactive;
            if (ad_rebuild) {
                for (int e = lane; e < na * (na + 1) / 2; e += NT) {
                    const int a = static_cast<int>((sqrtf(static_cast<float>(8 * e + 1)) - 1.0f) * 0.5f);
                    const int b = e - a * (a + 1) / 2;
                    const int ia = ad_ids[a], ib = ad_ids[b];
                    float value = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < 18; ++d) value += s_Yt[d * YS + ia] * s_Yt[d * YS + ib];
                    ad_gram[e] = value;
                }
                SYNC();
            }
            // Exact inactive identity elimination, not active-row truncation.
            const bool inactive_nonzero = ad_row && !active && s_x[ad_i] != 0.0f;
            if (ad_any(inactive_nonzero)) {
                if (ad_row) s_y[ad_i] = active ? 0.0f : -s_x[ad_i];
                SYNC();
                const float coupling = ad_residual(s_y, false);
                if (ad_row) ad_delta[ad_i] = coupling;
            } else if (ad_row) ad_delta[ad_i] = 0.0f;
            SYNC();
            if (lane == 0) ad_lu_ok = 1;
            if (active) {
                const int a = ad_map[ad_i];
                float c0 = 1.0f, c1 = 0.0f, cn = 0.0f;
                int t0 = ad_i, t1 = ad_i, p = ad_i;
                if (s_kind[ad_i] == 1) {
                    p = s_parent[ad_i]; t0 = p + 1; t1 = p + 2;
                    const float q0 = ad_q[t0], q1 = ad_q[t1];
                    const float length = sqrtf(q0 * q0 + q1 * q1);
                    const float radius = s_mu[ad_i] * ad_q[p];
                    c0 = ad_i == t0 ? 1.0f : 0.0f;
                    c1 = ad_i == t1 ? 1.0f : 0.0f;
                    if (length > radius) {
                        const float u0 = q0 / length, u1 = q1 / length;
                        const float ui = ad_i == t0 ? u0 : u1;
                        const float ratio = radius / length;
                        c0 = ratio * (c0 - ui * u0);
                        c1 = ratio * (c1 - ui * u1);
                        cn = s_mu[ad_i] * ui;
                    }
                }
                const float normal_ratio = s_step[p] / ad_eta;
                const float coupling = c0 * ad_delta[t0] + c1 * ad_delta[t1]
                    + cn * normal_ratio * ad_delta[p];
                ad_lu_rhs[a] = -natural - coupling;
                for (int b = 0; b < na; ++b) {
                    const int j = ad_ids[b];
                    float value;
                    if (s_kind[ad_i] == 0) value = ad_g(a, b);
                    else {
                        value = ((j == ad_i ? 1.0f : 0.0f) - c0 * (j == t0 ? 1.0f : 0.0f)
                            - c1 * (j == t1 ? 1.0f : 0.0f) - cn * (j == p ? 1.0f : 0.0f)) / ad_eta;
                        value += c0 * ad_g(ad_map[t0], b) + c1 * ad_g(ad_map[t1], b)
                            + cn * normal_ratio * ad_g(ad_map[p], b);
                    }
                    s_L[a * 18 + b] = value;
                }
            }
            SYNC();
            ad_factor_dirty = true;
            // One warp owns the small nonsymmetric LU. The second tier warp
            // waits at the enclosing CTA join, never at these warp-only steps.
            if (lane < 32) {
                float row_scale = 0.0f;
                if (lane < na) for (int j = 0; j < na; ++j) row_scale = fmaxf(row_scale, fabsf(s_L[lane * 18 + j]));
                const int bad = lane < na && (!(row_scale > 0.0f) || !isfinite(row_scale) || !isfinite(ad_lu_rhs[lane]));
                if (__any_sync(MASK, bad)) { if (lane == 0) ad_lu_ok = 0; }
                if (lane < na && !bad) {
                    for (int j = 0; j < na; ++j) s_L[lane * 18 + j] /= row_scale;
                    ad_lu_rhs[lane] /= row_scale;
                }
                __syncwarp(MASK);
                for (int k = 0; k < na && ad_lu_ok; ++k) {
                    float best = lane >= k && lane < na ? fabsf(s_L[lane * 18 + k]) : -1.0f;
                    int pivot = lane;
                    for (int shift = 16; shift > 0; shift >>= 1) {
                        const float other = __shfl_xor_sync(MASK, best, shift);
                        const int index = __shfl_xor_sync(MASK, pivot, shift);
                        if (other > best || (other == best && index < pivot)) { best = other; pivot = index; }
                    }
                    if (!(best > 7.62939453125e-6f) || !isfinite(best)) {
                        if (lane == 0) ad_lu_ok = 0;
                        __syncwarp(MASK);
                        break;
                    }
                    if (pivot != k) {
                        if (lane < na) {
                            const float value = s_L[k * 18 + lane];
                            s_L[k * 18 + lane] = s_L[pivot * 18 + lane];
                            s_L[pivot * 18 + lane] = value;
                        }
                        if (lane == 0) {
                            const float value = ad_lu_rhs[k];
                            ad_lu_rhs[k] = ad_lu_rhs[pivot]; ad_lu_rhs[pivot] = value;
                        }
                    }
                    __syncwarp(MASK);
                    if (lane > k && lane < na) {
                        const float factor = s_L[lane * 18 + k] / s_L[k * 18 + k];
                        for (int j = k + 1; j < na; ++j) s_L[lane * 18 + j] -= factor * s_L[k * 18 + j];
                        s_L[lane * 18 + k] = 0.0f;
                        ad_lu_rhs[lane] -= factor * ad_lu_rhs[k];
                    }
                    __syncwarp(MASK);
                }
                if (ad_lu_ok) for (int k = na - 1; k >= 0; --k) {
                    if (lane == k) {
                        float value = ad_lu_rhs[k];
                        for (int j = k + 1; j < na; ++j) value -= s_L[k * 18 + j] * ad_lu_rhs[j];
                        ad_lu_rhs[k] = value / s_L[k * 18 + k];
                    }
                    __syncwarp(MASK);
                }
            }
            SYNC();
            if (ad_row) ad_delta[ad_i] = active ? ad_lu_rhs[ad_map[ad_i]] : -s_x[ad_i];
            const bool direction_bad = ad_any(ad_row && !isfinite(ad_delta[ad_i]));
            bool accepted = false;
            // A failed Newton direction gets a merit-tested projected direction,
            // not an unconditional diagonal update. Each direction has at most
            // eight trials, and only one accepted correction consumes an outer.
            for (int direction = 0; direction < 2 && !accepted; ++direction) {
                if (direction == 0 && (!ad_lu_ok || direction_bad)) continue;
                const float update = direction == 0 && ad_row ? ad_delta[ad_i] : projected_direction;
                for (int trial = 0; trial < 8; ++trial) {
                    const float alpha = ldexpf(1.0f, -trial);
                    if (ad_row) s_y[ad_i] = s_x[ad_i] + alpha * update;
                    SYNC();
                    ad_project(s_y);
                    const float trial_residual = ad_residual(s_y, true);
                    const float trial_natural = ad_map_value(s_y, trial_residual);
                    const float merit = ad_sum(trial_natural * trial_natural);
                    const float decrease = 1.0f - 1.0e-4f * alpha;
                    if (isfinite(merit) && merit < decrease * decrease * old_merit) {
                        if (ad_row) s_x[ad_i] = s_y[ad_i];
                        accepted = true;
                        break;
                    }
                }
            }
            // If neither direction decreases the all-row merit, retain the
            // literal original EX1 recurrence with only the unused budget.
            if (!accepted) break;
            ++ad_consumed;
            SYNC();
            if (ad_consumed == 24) ad_done = true;
        }
        // Restore original factor/reference ownership even for a late handoff.
        if (ad_factor_dirty) {
            for (int e = lane; e < 324; e += NT) s_L[e] = ink_L_a.data[(size_t)ink_ga * 324 + e];
            SYNC();
        }
        if (!ad_done && ad_row) s_y[ad_i] = s_x[ad_i];
        SYNC();
    }
"""


def _rewrite_native(source, rows):
    """Keep staging, reference and decode while replacing the admitted solve."""
    del rows
    start = "    // b'_i = rhs_i + J_i . (v_in - Y^T lam0)"
    source = _replace_once(source, start, _NATIVE + "\n    if (!ad_done) {\n" + start)
    end = "    SYNC();\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    # The MF branch is the only admitted factory. Close its retained setup
    # before the preprocessor's unused non-MF branch, leaving that source intact.
    source = _replace_once(
        source, end, "    SYNC();\n    }\n#else\n#if 1\n    // In-kernel response in whitened coordinates:"
    )
    source = _replace_once(source, "    float t_k = 1.0f;", "    if (!ad_done) {\n    float t_k = 1.0f;")
    source = _replace_once(source, "sweep < 24;", "sweep < 24 - ad_consumed;")
    source = _replace_once(
        source,
        "const int global_iter = iteration_offset + sweep;",
        "const int global_iter = iteration_offset + ad_consumed + sweep;",
    )
    source = _replace_once(source, "\n#if 1\n    // u = Z^T (x - lam0)", "\n    }\n#if 1\n    // u = Z^T (x - lam0)")
    return source


def _supported(arguments, globals_):
    return (
        arguments["max_world_dofs"] == 18
        and arguments["rows"] in (32, 48)
        and arguments["min_rows"] == (0 if arguments["rows"] == 32 else 32)
        and arguments["sweeps"] == 24
        and arguments["nesterov"]
        and arguments["matrix_free"]
        and arguments["inkernel_response"] == (18, 0, 0, 0)
        and arguments["exact_row_sums"]
        and arguments["world_rows"]
        and not any(
            arguments[name]
            for name in (
                "has_drive_rows",
                "has_dense_velocity_limit_rows",
                "skip_local_internal_worlds",
                "lean_sweep",
                "bound_finger",
            )
        )
        and not any(
            globals_.get(name, False) for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING")
        )
        and int(os.environ.get("FEATHER_PGS_WR_STOP", "0")) == 0
    )


@functools.cache
def get_parallel_factory(original):
    """Preserve the original factory ABI and all unsupported factory owners."""
    original_source = inspect.getsource(original)
    if hashlib.sha256(original_source.encode()).hexdigest() != _ORIGINAL_FACTORY_SHA:
        raise RuntimeError("Active-dual original factory changed; review the complete seam")
    tree = ast.parse(textwrap.dedent(original_source))
    factory = tree.body[0]
    factory.name = "active_dual_parallel_factory"
    factory.decorator_list = []
    native = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.FunctionDef) and node.name == "pgs_solve_parallel_native"
    )
    factory.body[native:native] = ast.parse("snippet = _rewrite_native(snippet, AM)").body
    naming = next(
        i
        for i, node in enumerate(factory.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "__name__" for target in node.targets)
    )
    factory.body[naming:naming] = ast.parse("name += '_ad1'").body
    final = next(i for i, node in enumerate(factory.body) if isinstance(node, ast.Return))
    factory.body[final:final] = ast.parse(
        "kernel._fpgs_active_dual = True\nkernel._fpgs_active_dual_native = snippet"
    ).body
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    filename = f"<active-dual-{hashlib.sha256(generated.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(generated), None, generated.splitlines(keepends=True), filename)
    namespace = dict(original.__globals__)
    namespace["_rewrite_native"] = _rewrite_native
    exec(compile(generated, filename, "exec"), namespace)
    successor = namespace[factory.name]
    signature = inspect.signature(original)

    @functools.wraps(original)
    def selected(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not _supported(bound.arguments, original.__globals__):
            return original(*args, **kwargs)
        return successor(*args, **kwargs)

    return selected


def validate_solver(solver, source):
    """Reject a requested mode that cannot own the ordinary ANYmal tiers."""
    if not (
        solver.model.device.is_cuda
        and solver.grouped_dynamics
        and solver.max_world_dofs == 18
        and solver.mf_gs_parallel_rows == 48
        and solver.mf_gs_parallel_sweeps == 24
        and solver.mf_gs_parallel_nesterov
        and solver.mf_gs_parallel_matrix_free
        and solver.mf_gs_response_block_rows == 0
        and solver.dense_max_constraints >= 48
        and solver.friction_mode == "current"
        and not solver.enable_joint_velocity_limits
        and not solver.fuse_joint_velocity_limits
        and solver.drive_mode != "physx_pgs"
        and not solver.pgs_warmstart
        and not solver.pgs_debug
        and not solver._paired_factor_coordinates
        and not solver._local_internal_fast_path
        and source._INK_ON
        and source._WR_ON
        and source._MF_EXACT_ROWSUM
        and not any(
            getattr(source, name)
            for name in ("_INK_CHECK", "_WR_CHECK", "_WR_WARM", "_REGISTER_WHITENING", "_SHADOW_LEAN")
        )
        and int(os.environ.get("FEATHER_PGS_WR_STOP", "0")) == 0
    ):
        raise ValueError("Active dual requires ordinary ANYmal18 INK/WR/EX1 matrix-free Nesterov24 tiers32/48")

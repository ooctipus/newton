# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-contained linear fragment for the bounded Coulomb Newton experiment.

This function-scope CUDA source has no CTA collective. All 32 lanes of warp
zero call ``kc_solve_warp`` between two caller-owned CTA handoffs. Its 3,496
bytes do not overlay the retained factor or any original fallback scratch.
Both Arnoldi and right-preconditioned bases remain explicitly allocated.
"""

SCRATCH_BYTES = 3496

NATIVE = r"""
    struct KCLinearScratch {
        float gram[171];
        float rhs[18];
        float eta[18];
        float projection[54];
        float basis[162];
        float right_basis[144];
        float hessenberg[72];
        float cosine[8], sine[8], rotated_rhs[9], coeff[8];
        float lu[54], row_scale[18], delta[18];
        int columns[54], block_start[18], block_size[18], permutation[18];
        int ok, reason, steps;
        float true_error;
    };
    static_assert(sizeof(KCLinearScratch) == 3496, "Unexpected Krylov scratch layout");
    __shared__ KCLinearScratch kc_linear;

    // Reasons: 0 okay; 1 invalid input; 2 local preconditioner rank;
    // 3 Arnoldi/Givens breakdown; 4 Hessenberg rank; 5 true forcing;
    // 6 nonfinite direction. Every failure is an uncommitted outer guard.
    auto kc_solve_warp = [&](KCLinearScratch& s, int k, float target, int warp_lane) -> bool {
        const unsigned kc_mask = 0xffffffffu;
        const float kc_rank_tol = 7.62939453125e-6f;  // 64 * FP32 epsilon
        auto kc_fail = [&](int reason) -> bool {
            if (warp_lane == 0) { s.ok = 0; s.reason = reason; }
            __syncwarp(kc_mask);
            return false;
        };
        auto kc_sum = [&](float value) -> float {
            for (int shift = 16; shift > 0; shift >>= 1)
                value += __shfl_xor_sync(kc_mask, value, shift);
            return value;
        };
        auto kc_gram = [&](int a, int b) -> float {
            const int hi = a > b ? a : b, lo = a > b ? b : a;
            return s.gram[hi * (hi + 1) / 2 + lo];
        };

        if (warp_lane == 0) {
            s.ok = 0; s.reason = 0; s.steps = 0;
            s.true_error = __int_as_float(0x7f800000);
        }
        if (warp_lane < 18) s.delta[warp_lane] = 0.0f;
        __syncwarp(kc_mask);
        if (k < 0 || k > 18 || !isfinite(target) || target < 0.0f) return kc_fail(1);

        bool bad = false;
        if (warp_lane < k) {
            const int start = s.block_start[warp_lane], size = s.block_size[warp_lane];
            bad = !isfinite(s.rhs[warp_lane]) || !(s.eta[warp_lane] > 0.0f)
                || !isfinite(s.eta[warp_lane]) || start < 0 || size < 1 || size > 3
                || start > warp_lane || start + size > k || warp_lane >= start + size;
            if (!bad) {
                for (int row = start; row < start + size; ++row)
                    bad |= s.block_start[row] != start || s.block_size[row] != size;
                for (int slot = 0; slot < 3; ++slot) {
                    const float coefficient = s.projection[warp_lane * 3 + slot];
                    const int column = s.columns[warp_lane * 3 + slot];
                    bad |= !isfinite(coefficient) || column < -1 || column >= k;
                    if (coefficient != 0.0f) bad |= column < start || column >= start + size;
                }
            }
        }
        for (int e = warp_lane; e < k * (k + 1) / 2; e += 32) bad |= !isfinite(s.gram[e]);
        if (__any_sync(kc_mask, bad)) return kc_fail(1);

        // One leader factors each disjoint at-most-three-row linear block.
        // Scale remains in original row order; permutation is applied to b/scale.
        bool rank_bad = false;
        if (warp_lane < k && s.block_start[warp_lane] == warp_lane) {
            const int start = warp_lane, size = s.block_size[start];
            for (int row = 0; row < size; ++row) {
                const int a = start + row;
                float scale = 0.0f;
                for (int col = 0; col < size; ++col) {
                    const int b = start + col;
                    float projected = 0.0f;
                    for (int slot = 0; slot < 3; ++slot) {
                        const int c = s.columns[a * 3 + slot];
                        const float p = s.projection[a * 3 + slot];
                        if (c >= 0 && p != 0.0f)
                            projected += p * ((c == b ? 1.0f : 0.0f) - s.eta[c] * kc_gram(c, b));
                    }
                    const float value = ((a == b ? 1.0f : 0.0f) - projected) / s.eta[a];
                    s.lu[a * 3 + col] = value;
                    scale = fmaxf(scale, fabsf(value));
                    rank_bad |= !isfinite(value);
                }
                s.row_scale[a] = scale;
                s.permutation[a] = row;
                rank_bad |= !(scale > 0.0f) || !isfinite(scale);
                if (scale > 0.0f && isfinite(scale))
                    for (int col = 0; col < size; ++col) s.lu[a * 3 + col] /= scale;
            }
            for (int col = 0; col < size && !rank_bad; ++col) {
                int pivot = col;
                float best = fabsf(s.lu[(start + col) * 3 + col]);
                for (int row = col + 1; row < size; ++row) {
                    const float candidate = fabsf(s.lu[(start + row) * 3 + col]);
                    if (candidate > best) { best = candidate; pivot = row; }
                }
                if (!(best > kc_rank_tol) || !isfinite(best)) { rank_bad = true; break; }
                if (pivot != col) {
                    for (int j = 0; j < size; ++j) {
                        const float value = s.lu[(start + col) * 3 + j];
                        s.lu[(start + col) * 3 + j] = s.lu[(start + pivot) * 3 + j];
                        s.lu[(start + pivot) * 3 + j] = value;
                    }
                    const int value = s.permutation[start + col];
                    s.permutation[start + col] = s.permutation[start + pivot];
                    s.permutation[start + pivot] = value;
                }
                for (int row = col + 1; row < size; ++row) {
                    const float factor = s.lu[(start + row) * 3 + col] / s.lu[(start + col) * 3 + col];
                    s.lu[(start + row) * 3 + col] = factor;
                    for (int j = col + 1; j < size; ++j)
                        s.lu[(start + row) * 3 + j] -= factor * s.lu[(start + col) * 3 + j];
                }
            }
        }
        if (__any_sync(kc_mask, rank_bad)) return kc_fail(2);
        __syncwarp(kc_mask);

        auto kc_apply_preconditioner = [&](int col) {
            if (warp_lane < k && s.block_start[warp_lane] == warp_lane) {
                const int start = warp_lane, size = s.block_size[start];
                float value[3] = {0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (int row = 0; row < 3; ++row) if (row < size) {
                    const int original = start + s.permutation[start + row];
                    value[row] = s.basis[col * 18 + original] / s.row_scale[original];
                    #pragma unroll
                    for (int j = 0; j < 3; ++j) if (j < row)
                        value[row] -= s.lu[(start + row) * 3 + j] * value[j];
                }
                #pragma unroll
                for (int row = 2; row >= 0; --row) if (row < size) {
                    #pragma unroll
                    for (int j = 0; j < 3; ++j) if (j > row && j < size)
                        value[row] -= s.lu[(start + row) * 3 + j] * value[j];
                    value[row] /= s.lu[(start + row) * 3 + row];
                    s.right_basis[col * 18 + start + row] = value[row];
                }
            }
            __syncwarp(kc_mask);
        };
        auto kc_action = [&](float value) -> float {
            // Delta is scratch until the final certificate; that invocation
            // stores the final solution itself, so the output survives intact.
            if (warp_lane < k) s.delta[warp_lane] = value;
            __syncwarp(kc_mask);
            float dq = 0.0f;
            if (warp_lane < k) {
                float response = 0.0f;
                for (int j = 0; j < k; ++j) response += kc_gram(warp_lane, j) * s.delta[j];
                dq = value - s.eta[warp_lane] * response;
            }
            float projected = 0.0f;
            #pragma unroll
            for (int slot = 0; slot < 3; ++slot) {
                const int c = warp_lane < k ? s.columns[warp_lane * 3 + slot] : -1;
                const float p = warp_lane < k ? s.projection[warp_lane * 3 + slot] : 0.0f;
                const float other = __shfl_sync(kc_mask, dq, c >= 0 ? c : 0);
                projected += p * other;
            }
            const float result = warp_lane < k ? (value - projected) / s.eta[warp_lane] : 0.0f;
            __syncwarp(kc_mask);
            return result;
        };

        const float local_rhs = warp_lane < k ? s.rhs[warp_lane] : 0.0f;
        const float beta = sqrtf(kc_sum(local_rhs * local_rhs));
        if (!isfinite(beta)) return kc_fail(3);
        const int cap = k < 8 ? k : 8;
        if (beta > target) {
            for (int e = warp_lane; e < 72; e += 32) s.hessenberg[e] = 0.0f;
            if (warp_lane < 9) s.rotated_rhs[warp_lane] = warp_lane == 0 ? beta : 0.0f;
            if (warp_lane < k) s.basis[warp_lane] = local_rhs / beta;
            __syncwarp(kc_mask);
            for (int col = 0; col < cap; ++col) {
                kc_apply_preconditioner(col);
                const float right = warp_lane < k ? s.right_basis[col * 18 + warp_lane] : 0.0f;
                if (__any_sync(kc_mask, !isfinite(right))) return kc_fail(6);
                float value = kc_action(right);
                const float original_length = sqrtf(kc_sum(value * value));
                if (!isfinite(original_length)) return kc_fail(3);
                // Both coefficient batches see the same vector within a pass.
                // Sequentially updating it per coefficient would be MGS, not CGS2.
                for (int pass = 0; pass < 2; ++pass) {
                    for (int j = 0; j <= col; ++j) {
                        const float v = warp_lane < k ? s.basis[j * 18 + warp_lane] : 0.0f;
                        const float coefficient = kc_sum(v * value);
                        if (warp_lane == 0) {
                            s.coeff[j] = coefficient;
                            s.hessenberg[col * 9 + j] += coefficient;
                        }
                    }
                    __syncwarp(kc_mask);
                    if (warp_lane < k) {
                        float update = 0.0f;
                        for (int j = 0; j <= col; ++j) update += s.basis[j * 18 + warp_lane] * s.coeff[j];
                        value -= update;
                    }
                    __syncwarp(kc_mask);
                }
                const float length = sqrtf(kc_sum(value * value));
                if (!isfinite(length)) return kc_fail(3);
                const bool breakdown = length <= kc_rank_tol * original_length;
                if (!breakdown && warp_lane < k) s.basis[(col + 1) * 18 + warp_lane] = value / length;
                if (warp_lane == 0) {
                    s.steps = col + 1;
                    s.hessenberg[col * 9 + col + 1] = length;
                    for (int row = 0; row < col; ++row) {
                        const float a = s.hessenberg[col * 9 + row], b = s.hessenberg[col * 9 + row + 1];
                        s.hessenberg[col * 9 + row] = s.cosine[row] * a + s.sine[row] * b;
                        s.hessenberg[col * 9 + row + 1] = -s.sine[row] * a + s.cosine[row] * b;
                    }
                    const float a = s.hessenberg[col * 9 + col], b = s.hessenberg[col * 9 + col + 1];
                    const float radius = hypotf(a, b);
                    if (!(radius > 0.0f) || !isfinite(radius)) s.reason = 3;
                    else {
                        s.cosine[col] = a / radius; s.sine[col] = b / radius;
                        s.hessenberg[col * 9 + col] = radius;
                        s.hessenberg[col * 9 + col + 1] = 0.0f;
                        s.rotated_rhs[col + 1] = -s.sine[col] * s.rotated_rhs[col];
                        s.rotated_rhs[col] *= s.cosine[col];
                        s.ok = fabsf(s.rotated_rhs[col + 1]) <= target;
                        if (breakdown && !s.ok) s.reason = 3;
                    }
                }
                __syncwarp(kc_mask);
                if (s.reason != 0) return kc_fail(s.reason);
                if (s.ok) break;
            }
            // coeff's CGS2 temporary lifetime has ended; reuse only that array.
            if (warp_lane == 0) {
                for (int row = s.steps - 1; row >= 0; --row) {
                    const float diagonal = s.hessenberg[row * 9 + row];
                    if (diagonal == 0.0f || !isfinite(diagonal)) { s.reason = 4; break; }
                    float value = s.rotated_rhs[row];
                    for (int j = row + 1; j < s.steps; ++j) value -= s.hessenberg[j * 9 + row] * s.coeff[j];
                    s.coeff[row] = value / diagonal;
                }
            }
            __syncwarp(kc_mask);
            if (s.reason != 0) return kc_fail(s.reason);
        }
        float solution = 0.0f;
        if (warp_lane < k)
            for (int j = 0; j < s.steps; ++j) solution += s.right_basis[j * 18 + warp_lane] * s.coeff[j];
        if (__any_sync(kc_mask, !isfinite(solution))) return kc_fail(6);
        const float defect = kc_action(solution) - local_rhs;
        const float error = sqrtf(kc_sum(defect * defect));
        if (warp_lane == 0) {
            s.true_error = error;
            s.ok = isfinite(error) && error <= target;
            s.reason = s.ok ? 0 : 5;
        }
        __syncwarp(kc_mask);
        return s.ok != 0;
    };
"""

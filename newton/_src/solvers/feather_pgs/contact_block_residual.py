# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sequential contact GS through four local response products.

The producer owns current canonical coefficients. This consumer retains only
compact velocities, row state and coordinate maps in shared memory. Contact
projection order and the original prefix/limit recurrence remain unchanged.
"""

from functools import cache

import warp as wp

from . import compact_contact as compact
from . import compact_island_residual as original
from .contact_block_data import BlockContactData
from .fused_contact_solve import BiasData, SolveData, _access
from .private_contact_islands import IslandRouting

ROWS = 160
COORDINATES = 22
_LAYOUT = tuple(item for item in original._LAYOUT if item[1] not in ("s_J", "s_Y", "s_dof", "s_jy"))
SHARED_BYTES = 4 * sum(count for _, _, count in _LAYOUT)


def _aliases():
    """Bind the one explicit shared allocation to its bounded row state."""
    lines = ["float* storage = reinterpret_cast<float*>(address);"]
    offset = 0
    for dtype, name, count in _LAYOUT:
        lines.append(f"{dtype}* {name} = reinterpret_cast<{dtype}*>(storage + {offset});")
        offset += count
    return "\n".join(lines) + "\n"


_INITIALIZE = r"""
    const int lane = threadIdx.x & 31;
    const int prefix = solve.dense_phase_bounds.data[world * 2 + 1];
    const int reserved = routing.residual_key_count.data[world];
    const int count = routing.residual_row_count.data[world];
    for (int c = lane; c < 114; c += 32) s_map[c] = -1;
    if (lane < 22) {
        int physical = -1;
        if (lane < 6) physical = solve.dense_offsets.data[world] + lane;
        else if (lane - 6 < reserved)
            physical = routing.residual_key_coordinates.data[world * 16 + lane - 6];
        s_physical[lane] = physical;
        s_global[lane] = physical >= 0
            ? solve.world_dof_indices.data[world * 114 + physical] : -1;
    }
    __syncwarp(0xffffffffu);
    if (lane < 22 && s_physical[lane] >= 0) s_map[s_physical[lane]] = lane;
    for (int row = lane; row < count; row += 32) {
        const int canonical = routing.residual_row_ids.data[world * 160 + row];
        const int index = world * 704 + canonical;
        const int kind = data.row_type.data[index];
        const int component = row >= prefix ? (row - prefix) % 3 : 0;
        const int parent = row < prefix ? data.row_parent.data[index]
            : (component > 0 ? row - component : -1);
        s_lambda[row] = 0.0f;
        s_diag[row] = data.diag.data[index];
        s_meta[row] = (kind & 7) | ((parent + 1) << 3);
        s_mu[row] = data.row_mu.data[index];
        float rhs = 0.0f;
        if (row < prefix) {
            const float phi = data.phi.data[index];
            const float scale = phi < 0.0f ? data.row_beta.data[index]
                : bias.joint_limit_speculative_scale;
            rhs = scale * phi * (1.0f / bias.dt) - data.target.data[index];
            solve.rhs_bias.data[index] = rhs;
        } else rhs = solve.rhs_bias.data[index];
        s_rhs[row] = rhs;
    }
    __syncwarp(0xffffffffu);
"""

# Every contact visit enters and leaves collectively. The fourth eight-lane
# group contributes zero but participates in every full-mask shuffle.
_CONTACT = r"""
            if (contact_component == 0) {
                const int normal = row;
                const int packet_index = (normal - contact_start) / 3;
                const int raw = routing.residual_contact_ids.data[world * 53 + packet_index];
                const int row_group = lane >> 3;
                const int local_lane = lane & 7;
                int block_coord = -1;
                float block_j = 0.0f;
                float block_y = 0.0f;
                if (row_group < 3) {
                    const int packet_row = normal + row_group;
                    const int canonical = routing.residual_row_ids.data[world * 160 + packet_row];
                    if (local_lane < 6) {
                        block_coord = local_lane;
                        const int index = (canonical_group * 704 + canonical) * 6 + local_lane;
                        block_j = solve.dense_J.data[index];
                        block_y = solve.dense_Y.data[index];
                    } else {
                        const int endpoint = local_lane - 6;
                        const int physical = solve.sparse_row_dof.data[(world * 704 + canonical) * 2 + endpoint];
                        if (physical >= 0) block_coord = s_map[physical];
                        if (block_coord >= 0) {
                            const int index = (world * 704 + canonical) * 4 + endpoint * 2;
                            block_j = solve.sparse_row_jy.data[index];
                            block_y = solve.sparse_row_jy.data[index + 1];
                        }
                    }
                }
                float block_partial = block_coord >= 0 ? block_j * s_v[block_coord] : 0.0f;
                block_partial += __shfl_down_sync(MASK, block_partial, 4, 8);
                block_partial += __shfl_down_sync(MASK, block_partial, 2, 8);
                block_partial += __shfl_down_sync(MASK, block_partial, 1, 8);
                const float velocity0 = __shfl_sync(MASK, block_partial, 0);
                float velocity1 = __shfl_sync(MASK, block_partial, 8);
                float velocity2 = __shfl_sync(MASK, block_partial, 16);
                float delta0 = 0.0f, delta1 = 0.0f, delta2 = 0.0f;
                if (lane == 0) {
                    float lambda0 = s_lambda[normal];
                    float lambda1 = s_lambda[normal + 1];
                    float lambda2 = s_lambda[normal + 2];
                    const float denominator0 = s_diag[normal];
                    if (!(denominator0 <= 0.0f)) {
                        const float next = fmaxf(lambda0 - solve.omega
                            * (velocity0 + s_rhs[normal]) / denominator0, 0.0f);
                        delta0 = next - lambda0;
                        lambda0 = next;
                        if (delta0 != 0.0f) iteration_changed = 1;
                    }
                    if (global_iteration < solve.friction_start_iteration) {
                        // Match the original delayed-friction lambda-only clear.
                        lambda1 = 0.0f;
                        lambda2 = 0.0f;
                    } else if (!(lambda0 <= 0.0f && lambda1 == 0.0f && lambda2 == 0.0f)) {
                        if (delta0 != 0.0f) {
                            velocity1 += block.gram.data[raw * 4] * delta0;
                            velocity2 += block.gram.data[raw * 4 + 1] * delta0;
                        }
                        const float denominator1 = s_diag[normal + 1];
                        if (!(denominator1 <= 0.0f)) {
                            const float old = lambda1;
                            float next = old - solve.omega
                                * (velocity1 + s_rhs[normal + 1]) / denominator1;
                            const float radius = fmaxf(s_mu[normal + 1] * lambda0, 0.0f);
                            if (radius <= 0.0f) next = 0.0f;
                            else {
                                const float magnitude = sqrtf(next * next + lambda2 * lambda2);
                                if (magnitude > radius) {
                                    const float scale = radius / magnitude;
                                    next *= scale;
                                    const float sibling = lambda2 * scale;
                                    const float change = sibling - lambda2;
                                    lambda2 = sibling;
                                    delta2 += change;
                                    if (change != 0.0f) {
                                        iteration_changed = 1;
                                        velocity2 += block.gram.data[raw * 4 + 3] * change;
                                    }
                                }
                            }
                            const float change = next - old;
                            lambda1 = next;
                            delta1 += change;
                            if (change != 0.0f) {
                                iteration_changed = 1;
                                velocity2 += block.gram.data[raw * 4 + 2] * change;
                            }
                        }
                        const float denominator2 = s_diag[normal + 2];
                        if (!(denominator2 <= 0.0f)) {
                            const float old = lambda2;
                            float next = old - solve.omega
                                * (velocity2 + s_rhs[normal + 2]) / denominator2;
                            const float radius = fmaxf(s_mu[normal + 2] * lambda0, 0.0f);
                            if (radius <= 0.0f) next = 0.0f;
                            else {
                                const float magnitude = sqrtf(next * next + lambda1 * lambda1);
                                if (magnitude > radius) {
                                    const float scale = radius / magnitude;
                                    next *= scale;
                                    const float sibling = lambda1 * scale;
                                    const float change = sibling - lambda1;
                                    lambda1 = sibling;
                                    delta1 += change;
                                    if (change != 0.0f) iteration_changed = 1;
                                }
                            }
                            const float change = next - old;
                            lambda2 = next;
                            delta2 += change;
                            if (change != 0.0f) iteration_changed = 1;
                        }
                    }
                    s_lambda[normal] = lambda0;
                    s_lambda[normal + 1] = lambda1;
                    s_lambda[normal + 2] = lambda2;
                }
                delta0 = __shfl_sync(MASK, delta0, 0);
                delta1 = __shfl_sync(MASK, delta1, 0);
                delta2 = __shfl_sync(MASK, delta2, 0);
                const float delta = row_group == 0 ? delta0 : (row_group == 1 ? delta1 : delta2);
                const float contribution = row_group < 3 && block_coord >= 0 && delta != 0.0f
                    ? block_y * delta : 0.0f;
                const float response1 = __shfl_sync(MASK, contribution, local_lane + 8);
                const float response2 = __shfl_sync(MASK, contribution, local_lane + 16);
                // Common geometry preserves component-wise coordinate identities
                // and merges duplicate sparse coordinates. Only group zero writes.
                if (row_group == 0 && block_coord >= 0) {
                    const float change = (contribution + response1) + response2;
                    if (change != 0.0f) s_v[block_coord] += change;
                }
                __syncwarp(MASK);
                serial_row += 2;
                continue;
            }
"""


def sources(device_arch):
    """Keep original limits/prefix/speculation and replace only contact visits."""
    _, source = original.sources(device_arch)
    source = "const int canonical_group = solve.dense_groups.data[world];\n" + source

    def row(index):
        return f"routing.residual_row_ids.data[world * 160 + ({index})]"

    for name, public in (("s_J", "dense_J"), ("s_Y", "dense_Y")):
        alias = f"bridge.{public}"
        source = source.replace(f"{name}[", f"{alias}.data[")
        source = _access(
            source,
            alias,
            lambda i, public=public: (
                f"solve.{public}.data[(canonical_group * 704 + {row(f'({i}) / 6')}) * 6 + ({i}) % 6]"
            ),
        )
    source = source.replace("s_dof[", "bridge.dof.data[")

    def mapped_dof(index):
        canonical = row(f"({index}) / 2")
        physical = f"solve.sparse_row_dof.data[(world * 704 + {canonical}) * 2 + ({index}) % 2]"
        return f"((({index}) / 2 < contact_start || {physical} < 0) ? -1 : s_map[{physical}])"

    source = _access(source, "bridge.dof", mapped_dof)
    source = source.replace("s_jy[", "bridge.jy.data[")
    source = _access(
        source,
        "bridge.jy",
        lambda i: (
            f"(({i}) / 4 < contact_start ? 0.0f : "
            f"solve.sparse_row_jy.data[(world * 704 + {row(f'({i}) / 4')}) * 4 + ({i}) % 4])"
        ),
    )
    marker = "            const int row_type = s_meta[row] & 7;"
    if source.count(marker) != 1:
        raise RuntimeError("The original contact/prefix projection boundary changed")
    source = source.replace(marker, _CONTACT + marker)
    if any(f"{name}[" in source for name in ("s_J", "s_Y", "s_dof", "s_jy")):
        raise RuntimeError("The light contact recurrence retained a private coefficient reader")
    return source


@cache
def get_kernel(device_arch):
    """Build the CUDA-only one-world/one-warp coupled contact consumer."""
    aliases = _aliases()
    allocation = f"""
#if defined(__CUDA_ARCH__)
    __shared__ float storage[{SHARED_BYTES // 4}];
    return reinterpret_cast<wp::uint64>(storage);
#else
    return 0;
#endif
"""

    @wp.func_native(allocation)
    def allocate_shared() -> wp.uint64: ...

    @wp.func_native("#if defined(__CUDA_ARCH__)\n" + aliases + _INITIALIZE + "#endif\n")
    def initialize(
        address: wp.uint64,
        world: int,
        data: compact.ContactBoundaryData,
        solve: SolveData,
        bias: BiasData,
        routing: IslandRouting,
    ): ...

    @wp.func_native("#if defined(__CUDA_ARCH__)\n" + aliases + sources(device_arch) + "#endif\n")
    def solve_contacts(
        address: wp.uint64, world: int, solve: SolveData, routing: IslandRouting, block: BlockContactData
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def contact_block_residual(
        data: compact.ContactBoundaryData,
        solve: SolveData,
        bias: BiasData,
        routing: IslandRouting,
        block: BlockContactData,
    ):
        world, _lane = wp.tid()
        if routing.owner[world] == 1:
            address = allocate_shared()
            initialize(address, world, data, solve, bias, routing)
            solve_contacts(address, world, solve, routing, block)

    return contact_block_residual

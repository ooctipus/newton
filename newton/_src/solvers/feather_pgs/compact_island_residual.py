# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private coupled rows with the original sparse GS law and compact coordinates.

The partitioner admits at most 160 rows and 16 reserved scalar coordinates.
The six articulated coordinates come first locally, irrespective of their
canonical offset. Isolated scalar islands never enter this shared state.
"""

import re
from functools import cache

import warp as wp

from . import compact_contact as compact
from .fused_contact_solve import BiasData, SolveData, _access, _field_names, _native
from .private_contact_islands import IslandRouting, contact_rhs

ROWS = 160
COORDINATES = 22
GLOBAL_ROWS = 704
GLOBAL_COORDINATES = 114

# All elements are four bytes. One native allocation is passed explicitly to
# preparation and recurrence, rather than relying on separate function-local
# shared declarations aliasing each other.
_LAYOUT = (
    ("float", "s_v", COORDINATES),
    ("float", "s_lambda", ROWS),
    ("float", "s_rhs", ROWS),
    ("float", "s_diag", ROWS),
    ("float", "s_mu", ROWS),
    ("int", "s_meta", ROWS),
    ("float", "s_J", ROWS * 6),
    ("float", "s_Y", ROWS * 6),
    ("int", "s_dof", ROWS * 2),
    ("float", "s_jy", ROWS * 4),
    ("int", "s_global", COORDINATES),
    ("int", "s_physical", COORDINATES),
    ("int", "s_map", GLOBAL_COORDINATES),
)
SHARED_BYTES = 4 * sum(count for _, _, count in _LAYOUT)


def _aliases():
    """Give every native phase the same bounded shared allocation layout."""
    result = ["float* storage = reinterpret_cast<float*>(address);"]
    offset = 0
    for dtype, name, count in _LAYOUT:
        result.append(f"{dtype}* {name} = reinterpret_cast<{dtype}*>(storage + {offset});")
        offset += count
    return "\n".join(result) + "\n"


_INITIALIZE = r"""
    const int lane = threadIdx.x & 31;
    const int prefix = solve.dense_phase_bounds.data[world * 2 + 1];
    const int group = solve.dense_groups.data[world];
    const int reserved = routing.residual_key_count.data[world];
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
    for (int row = lane; row < prefix; row += 32) {
        const int canonical = routing.residual_row_ids.data[world * 160 + row];
        const int index = world * 704 + canonical;
        const int kind = data.row_type.data[index];
        const int parent = data.row_parent.data[index];
        s_lambda[row] = 0.0f;
        s_diag[row] = data.diag.data[index];
        s_meta[row] = (kind & 7) | ((parent + 1) << 3);
        s_mu[row] = data.row_mu.data[index];
        const float phi = data.phi.data[index];
        const float scale = phi < 0.0f ? data.row_beta.data[index] : bias.joint_limit_speculative_scale;
        const float rhs = scale * phi * (1.0f / bias.dt) - data.target.data[index];
        s_rhs[row] = rhs;
        solve.rhs_bias.data[index] = rhs;
        for (int c = 0; c < 6; ++c) {
            s_J[row * 6 + c] = data.J.data[(group * 704 + canonical) * 6 + c];
            s_Y[row * 6 + c] = data.Y.data[(group * 704 + canonical) * 6 + c];
        }
        // Admission preserves only the original dense-limit prefix here.
        // Its sparse operands may intentionally still contain old row data.
        s_dof[row * 2] = -1;
        s_dof[row * 2 + 1] = -1;
        for (int c = 0; c < 4; ++c) s_jy[row * 4 + c] = 0.0f;
    }
    __syncwarp(0xffffffffu);
"""

_STORE = r"""
    s_lambda[row] = 0.0f;
    s_rhs[row] = rhs;
    s_diag[row] = packet.diag;
    s_mu[row] = packet.mu;
    const int parent = component > 0 ? row - component : -1;
    s_meta[row] = (packet.kind & 7) | ((parent + 1) << 3);
    for (int c = 0; c < 6; ++c) {
        s_J[row * 6 + c] = packet.J[c];
        s_Y[row * 6 + c] = packet.Y[c];
    }
    s_dof[row * 2] = packet.dof0 >= 0 ? s_map[packet.dof0] : -1;
    s_dof[row * 2 + 1] = packet.dof1 >= 0 ? s_map[packet.dof1] : -1;
    s_jy[row * 4] = packet.j0;
    s_jy[row * 4 + 1] = packet.y0;
    s_jy[row * 4 + 2] = packet.j1;
    s_jy[row * 4 + 3] = packet.y1;
"""


def sources(device_arch):
    """Remap storage access only; retain the original eight-sweep contact law."""
    from .solver_feather_pgs import _get_pgs_solve_sparse_diagonal_kernel  # noqa: PLC0415

    original = _native(
        _get_pgs_solve_sparse_diagonal_kernel(
            ROWS, COORDINATES, 6, device_arch, contact_triples=True, speculative_contact_batches=True
        )
    ).native_snippet
    start_marker = "    for (int row = lane; row < row_count; row += 32) {"
    end_marker = "    for (int coord = lane; coord < 22; coord += 32) {"
    if original.count(start_marker) != 1 or original.count(end_marker) != 2:
        raise RuntimeError("The original sparse staging/publication boundary changed")
    begin = original.index(start_marker)
    end = original.index(end_marker, begin)
    snippet = original[:begin] + original[end:]
    snippet, shared_count = re.subn(r"    __shared__ [^;]+;\n", "", snippet)
    if shared_count != 6 or "__syncthreads" in snippet:
        raise RuntimeError("The original sparse shared layout or warp ownership changed")
    # Names are rebound before inserting expressions containing descriptor names.
    snippet = _field_names(snippet)
    accesses = {
        "solve.world_constraint_count": lambda i: f"routing.residual_row_count.data[{i}]",
        "solve.world_dof_indices": lambda i: f"s_global[({i}) - world * 22]",
        "solve.dense_offsets": lambda i: "0",
        "solve.dense_groups": lambda i: "0",
        "solve.dense_J": lambda i: f"s_J[{i}]",
        "solve.dense_Y": lambda i: f"s_Y[{i}]",
        "solve.sparse_row_dof": lambda i: f"s_dof[({i}) - world * 160 * 2]",
        "solve.sparse_row_jy": lambda i: f"s_jy[({i}) - world * 160 * 4]",
        "solve.sparse_contact_group_count": lambda i: "0",
        "solve.sparse_contact_group_heads": lambda i: "-1",
        "solve.sparse_contact_serial_count": lambda i: "((row_count - contact_start) / 3)",
        "solve.sparse_contact_serial_normals": lambda i: f"(contact_start + 3 * (({i}) - world * 54))",
        "solve.world_impulses": lambda i: (
            "solve.world_impulses.data[world * 704 + "
            f"routing.residual_row_ids.data[world * 160 + (({i}) - world * 160)]]"
        ),
    }
    for name in (
        "fused_limit_active",
        "fused_limit_lower_rhs",
        "fused_limit_upper_rhs",
        "fused_limit_lower_lambda",
        "fused_limit_upper_lambda",
    ):
        accesses[f"solve.{name}"] = lambda i, name=name: (
            f"solve.{name}.data[world * 114 + s_physical[({i}) - world * 22]]"
        )
    for name, transform in accesses.items():
        snippet = _access(snippet, name, transform)
    # Unused local slots must not address the previous world's canonical limit
    # storage. The original native law already initializes their response to 0.
    if snippet.count("    if (lane < 22) {") != 2:
        raise RuntimeError("The original fused-limit load/store guards changed")
    snippet = snippet.replace("    if (lane < 22) {", "    if (lane < 22 && s_physical[lane] >= 0) {")
    snippet = snippet.replace("#if defined(__CUDA_ARCH__)", "").replace("#endif", "")
    if "solve.dense_J.data[" in snippet or "solve.sparse_row_jy.data[" in snippet:
        raise RuntimeError("The residual recurrence retained a canonical contact coefficient reader")
    return original, snippet


@cache
def get_kernel(device_arch):
    """Build one-warp preparation and the original storage-remapped recurrence."""
    _, sweep = sources(device_arch)
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

    @wp.func_native("#if defined(__CUDA_ARCH__)\n" + aliases + _STORE + "#endif\n")
    def store_row(address: wp.uint64, row: int, component: int, packet: compact.ContactRow, rhs: float): ...

    @wp.func_native("#if defined(__CUDA_ARCH__)\n" + aliases + "__syncwarp(0xffffffffu);\n" + sweep + "#endif\n")
    def solve_residual(address: wp.uint64, world: int, solve: SolveData, routing: IslandRouting): ...

    @wp.kernel(module="unique", enable_backward=False)
    def residual(
        data: compact.ContactBoundaryData,
        solve: SolveData,
        bias: BiasData,
        routing: IslandRouting,
    ):
        world, lane = wp.tid()
        if routing.owner[world] == 1:
            address = allocate_shared()
            initialize(address, world, data, solve, bias, routing)
            prefix = solve.dense_phase_bounds[world, 1]
            count = (routing.residual_row_count[world] - prefix) // 3
            for contact_index in range(lane, count, 32):
                contact = routing.residual_contact_ids[world, contact_index]
                geometry = compact.prepare_contact_geometry(data, contact)
                for component in range(3):
                    packet = compact.evaluate_contact_row(data, geometry, component)
                    rhs = contact_rhs(solve, bias, world, packet)
                    canonical = geometry.slot + component
                    compact.publish_row_metadata(data, world, canonical, packet)
                    solve.rhs_bias[world, canonical] = rhs
                    store_row(address, prefix + contact_index * 3 + component, component, packet, rhs)
            solve_residual(address, world, solve, routing)

    return residual

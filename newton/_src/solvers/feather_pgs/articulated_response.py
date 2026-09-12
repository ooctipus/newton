# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental consumers of the shared articulated response factor.

The first bridge retains current canonical row geometry and the existing
nonlinear iteration. A prepared primary Y replaces only local triangular
response preparation; its array occupies the old primary L argument slot.
Secondary free-body factors and every projection/publication remain unchanged.
"""

import re
from functools import cache

import numpy as np
import warp as wp


def _replace_once(pattern: str, replacement: str, source: str) -> str:
    result, count = re.subn(pattern, replacement, source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one articulated-response source seam, found {count}: {pattern}")
    return result


def prepare_local_response_source(source: str, dofs: int) -> str:
    """Replace the exact inherited primary solve by prepared physical Y reads."""
    if dofs <= 0:
        raise ValueError("dofs must be positive")
    load = (
        rf"    for \(int i = lane; i < {dofs * dofs}; i \+= \d+\) \{{\n"
        r"        float factor = L_group\.data\[group_l_base \+ i\];\n"
        r"(?:[^\n]*\n)*?        s_L\[i\] = factor;\n    \}\n"
    )
    source = _replace_once(load, "", source)
    solve = (
        rf"        for \(int i = 0; i < {dofs}; \+\+i\) \{{\n"
        r"            float value = s_J\[row \* \d+ \+ i\];\n"
        r"[\s\S]*?(?=        float diagonal = 0\.0f;)"
    )
    source = _replace_once(
        solve,
        f"        for (int i = 0; i < {dofs}; ++i) {{\n"
        f"            response[i] = L_group.data[group_j_base + row * {dofs} + i];\n"
        "        }\n\n",
        source,
    )
    # The declaration is either standalone shared storage or one struct field.
    source = _replace_once(rf"^\s*(?:__shared__ )?float s_L\[{dofs * dofs}\];\n", "", source)
    source = re.sub(r"^#define s_L local_solve_scratch\[local_group\]\.s_L\n", "", source, flags=re.MULTILINE)
    source = re.sub(r"^#undef s_L\n", "", source, flags=re.MULTILINE)
    if "s_L[" in source:
        raise ValueError("Primary local factor reads remain after response conversion")
    return source


def prepare_upper_local_response_source(source: str, dofs: int) -> str:
    """Keep original row ownership and solve H=Et Et.T with upper Et."""
    if dofs <= 0:
        raise ValueError("dofs must be positive")
    pattern = (
        rf"        for \(int i = 0; i < {dofs}; \+\+i\) \{{\n"
        r"            float value = s_J\[row \* \d+ \+ i\];\n"
        r"[\s\S]*?(?=        float diagonal = 0\.0f;)"
    )
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1:
        raise ValueError("Expected exactly one original primary row solve")
    start, end = matches[0].span()
    block = source[start:end]
    reverse = f"        for (int reverse = 0; reverse < {dofs}; ++reverse) {{\n            const int i = {dofs} - 1 - reverse;"
    forward = f"        for (int i = 0; i < {dofs}; ++i) {{"
    if block.count(reverse) != 1:
        raise ValueError("Original primary transpose solve changed")
    first, second = block.split(reverse)
    if first.count(forward) != 1:
        raise ValueError("Original primary forward solve changed")
    first = first.replace(forward, reverse)
    first = _replace_once(
        rf"for \(int k = 0; k < i; \+\+k\) value -= s_L\[i \* {dofs} \+ k\] \* response\[k\];",
        f"for (int k = i + 1; k < {dofs}; ++k) value -= s_L[i * {dofs} + k] * response[k];",
        first,
    )
    second = _replace_once(
        rf"for \(int k = i \+ 1; k < {dofs}; \+\+k\) value -= s_L\[k \* {dofs} \+ i\] \* response\[k\];",
        f"for (int k = 0; k < i; ++k) value -= s_L[k * {dofs} + i] * response[k];",
        second,
    )
    return source[:start] + first + forward + second + source[end:]


@cache
def get_upper_response_fallback_kernel(dofs: int) -> wp.Kernel:
    """Preserve the original GENERAL/local restitution ownership using upper Et.

    The ABI and ordinary one-thread-per-row launch are identical to
    ``hinv_jt_par_row_contact_fallback``. Launch ``n_arts*32`` threads.
    """
    if dofs <= 0:
        raise ValueError("dofs must be positive")

    def upper_response_fallback(
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        local_solve_owner: wp.array[int],
        world_row_restitution: wp.array2d[float],
        n_dofs: int,
        max_constraints: int,
        n_arts: int,
        write_world: int,
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
    ):
        tid = wp.tid()
        group_index = tid // 32
        lane = tid % 32
        if group_index >= n_arts:
            return
        art = group_to_art[group_index]
        world = art_to_world[art]
        constraint_count = world_constraint_count[world]
        if local_solve_owner[world] != 0:
            if write_world != 0:
                dof_offset = articulation_world_dof_offset[art]
                constraint = lane
                while constraint < constraint_count:
                    if world_row_restitution[world, constraint] > 0.0:
                        for i in range(n_dofs):
                            J_world[world, constraint, dof_offset + i] = J_group[group_index, constraint, i]
                    constraint += 32
            return
        constraint = lane
        while constraint < constraint_count:
            for reverse in range(n_dofs):
                i = n_dofs - 1 - reverse
                value = J_group[group_index, constraint, i]
                for k in range(i + 1, n_dofs):
                    value -= L_group[group_index, i, k] * Y_group[group_index, constraint, k]
                diagonal = L_group[group_index, i, i]
                if diagonal != 0.0:
                    Y_group[group_index, constraint, i] = value / diagonal
                else:
                    Y_group[group_index, constraint, i] = 0.0
            for i in range(n_dofs):
                value = Y_group[group_index, constraint, i]
                for k in range(i):
                    value -= L_group[group_index, k, i] * Y_group[group_index, constraint, k]
                diagonal = L_group[group_index, i, i]
                if diagonal != 0.0:
                    Y_group[group_index, constraint, i] = value / diagonal
                else:
                    Y_group[group_index, constraint, i] = 0.0
            if write_world != 0:
                dof_offset = articulation_world_dof_offset[art]
                for i in range(n_dofs):
                    J_world[world, constraint, dof_offset + i] = J_group[group_index, constraint, i]
                    Y_world[world, constraint, dof_offset + i] = Y_group[group_index, constraint, i]
            constraint += 32

    upper_response_fallback.__name__ = f"upper_response_fallback_d{dofs}"
    upper_response_fallback.__qualname__ = upper_response_fallback.__name__
    return wp.kernel(enable_backward=False, module="unique")(upper_response_fallback)


@cache
def get_group_response_kernel(max_bodies: int, dofs: int, *, write_world: bool = True) -> wp.Kernel:
    """Build one warp per articulation, applying its held factor to current rows.

    Launch tiled ``[groups]`` with block_dim=32. All lanes traverse the same
    current bounded rows. The factor function owns and reuses one RHS workspace.
    Rows beyond the current count are untouched; no new capacity is inferred.
    """
    from .articulated_factor import ArticulatedFactorData, get_apply_function  # noqa: PLC0415

    apply_row = get_apply_function(max_bodies, dofs)
    publish_source = (
        f"constexpr int DOFS = {dofs};\n"
        + r"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    constexpr int WORKERS = 32;
#else
    const int lane = 0;
    constexpr int WORKERS = 1;
#endif
    const int offset = world_dof_offset.data[art];
    for (int dof = lane; dof < DOFS; dof += WORKERS) {
        const int input = (group * current_j.shape[1] + row) * DOFS + dof;
        const int output = (world * world_j.shape[1] + row) * world_j.shape[2] + offset + dof;
        world_j.data[output] = current_j.data[input];
        world_y.data[output] = response_y.data[(group * response_y.shape[1] + row) * DOFS + dof];
    }
"""
    )

    @wp.func_native(publish_source)
    def publish_row(
        group: int,
        row: int,
        art: int,
        world: int,
        world_dof_offset: wp.array[int],
        current_j: wp.array3d[float],
        response_y: wp.array3d[float],
        world_j: wp.array3d[float],
        world_y: wp.array3d[float],
    ): ...

    def articulated_group_response(
        factor: ArticulatedFactorData,
        current_j: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        world_dof_offset: wp.array[int],
        constraint_count: wp.array[int],
        response_y: wp.array3d[float],
        world_j: wp.array3d[float],
        world_y: wp.array3d[float],
    ):
        group, _lane = wp.tid()
        art = group_to_art[group]
        world = art_to_world[art]
        count = wp.min(wp.max(constraint_count[world], 0), current_j.shape[1])
        for row in range(count):
            apply_row(factor, group, row, current_j, response_y)
            if wp.static(write_world):
                publish_row(group, row, art, world, world_dof_offset, current_j, response_y, world_j, world_y)

    name = f"articulated_group_response_b{max_bodies}_d{dofs}_world{int(write_world)}"
    articulated_group_response.__name__ = name
    articulated_group_response.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(articulated_group_response)


def whitening_bridge_source(max_links: int, dofs: int, *, encoding_only: bool = False) -> str:
    """Build upper W and E-transpose, with H = E-transpose E and W = E-transpose^-1.

    The existing paired consumer uses these in its original L/Linv allocations,
    with the primary triangular comparisons reversed. Secondary6 is unchanged.
    This is a charged transitional matrix bridge, not dense-H construction.
    """
    if not 0 < dofs < max_links <= 128:
        raise ValueError("Expected a bounded fixed-root zero/one-DOF tree")
    source = (
        f"constexpr int LINKS = {max_links}, DOFS = {dofs};\n"
        + r"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    constexpr int WORKERS = 32;
    #define BRIDGE_SHARED __shared__
    #define BRIDGE_SYNC() __syncwarp(0xffffffffu)
#else
    const int lane = 0;
    constexpr int WORKERS = 1;
    #define BRIDGE_SHARED
    #define BRIDGE_SYNC() ((void)0)
#endif
    const int owner = factor.art_ids.data[group];
    if (mass_mask.data[owner] == 0) return;
    const int base = group * DOFS * DOFS;
    BRIDGE_SHARED int dof_body[DOFS];
    BRIDGE_SHARED float encoding_transpose[DOFS * DOFS];
    BRIDGE_SHARED float inverse_encoding[DOFS * DOFS];
    for (int item = lane; item < DOFS * DOFS; item += WORKERS) {
        encoding_transpose[item] = 0.0f;
        inverse_encoding[item] = 0.0f;
    }
    for (int item = lane; item < DOFS; item += WORKERS) dof_body[item] = -1;
    BRIDGE_SYNC();
    const int count = factor.counts.data[group];
    for (int link = lane; link < count; link += WORKERS) {
        const int slot = factor.dof_slots.data[group * LINKS + link];
        if (slot >= 0) dof_body[slot] = link;
    }
    BRIDGE_SYNC();
    if (factor.valid.data[group] == 0) {
        union { unsigned int bits; float value; } invalid;
        invalid.bits = 0x7fc00000u;
        for (int item = lane; item < DOFS * DOFS; item += WORKERS) {
            whitener.data[base + item] = invalid.value;
            encode_transpose.data[base + item] = invalid.value;
        }
        BRIDGE_SYNC();
        return;
    }
    // Each column is one joint's kinetic coordinate. Fixed ancestors have no
    // generalized column, but the walk must continue through them.
    for (int column = lane; column < DOFS; column += WORKERS) {
        const int link = dof_body[column];
        const int index = group * LINKS + link;
        const float scale = wp::sqrt(factor.invD.data[index]);
        encoding_transpose[column * DOFS + column] = 1.0f / scale;
        int ancestor = factor.parents.data[index];
        while (ancestor >= 0) {
            const int ancestor_index = group * LINKS + ancestor;
            const int row = factor.dof_slots.data[ancestor_index];
            if (row >= 0) {
                float value = 0.0f;
                for (int k = 0; k < 6; ++k)
                    value += factor.U.data[index * 6 + k] * factor.S.data[ancestor_index * 6 + k];
                encoding_transpose[row * DOFS + column] = scale * value;
            }
            ancestor = factor.parents.data[ancestor_index];
        }
    }
    BRIDGE_SYNC();
    // BEGIN_INVERSE
    // Solve the upper triangular encoding-transpose once per unit column.
    // Each column owns all of its output; no cross-column dependencies exist.
    for (int column = lane; column < DOFS; column += WORKERS) {
        for (int row = column; row >= 0; --row) {
            float value = row == column ? 1.0f : 0.0f;
            for (int k = row + 1; k <= column; ++k)
                value -= encoding_transpose[row * DOFS + k] * inverse_encoding[k * DOFS + column];
            inverse_encoding[row * DOFS + column] = value / encoding_transpose[row * DOFS + row];
        }
    }
    BRIDGE_SYNC();
    for (int item = lane; item < DOFS * DOFS; item += WORKERS)
        whitener.data[base + item] = inverse_encoding[item];
    // END_INVERSE
    for (int item = lane; item < DOFS * DOFS; item += WORKERS)
        encode_transpose.data[base + item] = encoding_transpose[item];
    BRIDGE_SYNC();
#undef BRIDGE_SHARED
#undef BRIDGE_SYNC
"""
    )
    if encoding_only:
        start, end = source.index("    // BEGIN_INVERSE"), source.index("    // END_INVERSE")
        source = source[:start] + source[end + len("    // END_INVERSE\n") :]
        for statement in (
            "    BRIDGE_SHARED float inverse_encoding[DOFS * DOFS];\n",
            "        inverse_encoding[item] = 0.0f;\n",
            "            whitener.data[base + item] = invalid.value;\n",
        ):
            if source.count(statement) != 1:
                raise ValueError("Encoding-only source seam changed")
            source = source.replace(statement, "")
        if "whitener" in source or "inverse_encoding" in source:
            raise ValueError("Encoding-only bridge still accesses whitening storage")
    return source


def validate_whitening_layout(factor, dofs: int) -> None:
    """Reject non-topological local DOF order once, before capture or allocation use."""
    slots = factor.dof_slots.numpy()
    counts = factor.counts.numpy()
    for group, count in enumerate(counts):
        ordered = slots[group, :count]
        if not np.array_equal(ordered[ordered >= 0], np.arange(dofs)):
            raise ValueError("Upper articulated whitening requires parent-first local DOF order")


@cache
def get_whitening_bridge_kernel(max_links: int, dofs: int) -> wp.Kernel:
    """Materialize the two upper transforms only on original factor-refresh masks.

    Validate the plan with ``validate_whitening_layout`` before first use.
    Launch tiled ``[groups]``, block_dim=32, after the shared factor refresh.
    ``encode_transpose`` and ``whitener`` have shape [groups,dofs,dofs].
    """
    from .articulated_factor import ArticulatedFactorData  # noqa: PLC0415

    @wp.func_native(whitening_bridge_source(max_links, dofs))
    def bridge_native(
        factor: ArticulatedFactorData,
        group: int,
        mass_mask: wp.array[int],
        encode_transpose: wp.array3d[float],
        whitener: wp.array3d[float],
    ): ...

    def articulated_whitening_bridge(
        factor: ArticulatedFactorData,
        mass_mask: wp.array[int],
        encode_transpose: wp.array3d[float],
        whitener: wp.array3d[float],
    ):
        group, _lane = wp.tid()
        bridge_native(factor, group, mass_mask, encode_transpose, whitener)

    name = f"articulated_whitening_bridge_b{max_links}_d{dofs}"
    articulated_whitening_bridge.__name__ = name
    articulated_whitening_bridge.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(articulated_whitening_bridge)


@cache
def get_encoding_bridge_kernel(max_links: int, dofs: int) -> wp.Kernel:
    """Publish only upper Et on held refresh; launch tiled [groups], block32."""
    from .articulated_factor import ArticulatedFactorData  # noqa: PLC0415

    @wp.func_native(whitening_bridge_source(max_links, dofs, encoding_only=True))
    def encode_native(
        factor: ArticulatedFactorData, group: int, mass_mask: wp.array[int], encode_transpose: wp.array3d[float]
    ): ...

    def articulated_encoding_bridge(
        factor: ArticulatedFactorData, mass_mask: wp.array[int], encode_transpose: wp.array3d[float]
    ):
        group, _lane = wp.tid()
        encode_native(factor, group, mass_mask, encode_transpose)

    articulated_encoding_bridge.__name__ = f"articulated_encoding_bridge_b{max_links}_d{dofs}"
    articulated_encoding_bridge.__qualname__ = articulated_encoding_bridge.__name__
    return wp.kernel(enable_backward=False, module="unique")(articulated_encoding_bridge)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share small Cholesky arithmetic across current and cached mass producers."""

import hashlib
import linecache
from functools import cache
from string import Template

import warp as wp

_FACTOR = Template("""
#if defined(__CUDA_ARCH__)
    const int lane = tid & 31;
    const int group = tid >> 5;
    if (group >= n_arts) return;

    const int articulation = group_to_art.data[group];
    if (mass_update_mask.data[articulation] == 0) return;

    const int warp = threadIdx.x >> 5;
    __shared__ float factors[$FACTOR_STORAGE];
$FORCE_STORAGE
    float* factor = factors + warp * $ELEMENTS;
$FORCE_POINTER
    const int dof_start = articulation_dof_start.data[articulation];
$FORCE_INIT
    for (int schedule_index = lane; schedule_index < $LOWER; schedule_index += 32) {
        const int packed = lower_schedule.data[schedule_index];
        const int element = packed & 255;
        const int row = element / $DOFS;
        const int col = element - row * $DOFS;
$MASS_INPUT
        if (row == col) {
            value += R_group.data[group * $DOFS + row];
            if (fused_augmented_drive != 0) {
                const int drive_row = drive_row_by_dof.data[dof_start + row];
                if (drive_row >= 0) {
                    const float K = row_K.data[drive_row];
                    if (K > 0.0f) value += K;
                }
            }
        }
        factor[element] = value;
    }
    __syncwarp();

    if (lane == 0) {
#pragma unroll
        for (int col = 0; col < $DOFS; ++col) {
            float diagonal = factor[col * $DOFS + col];
#pragma unroll
            for (int k = 0; k < col; ++k) {
                const float value = factor[col * $DOFS + k];
                diagonal -= value * value;
            }
            diagonal = sqrtf(diagonal);
            factor[col * $DOFS + col] = diagonal;
            const float inverse_diagonal = 1.0f / diagonal;
#pragma unroll
            for (int row = col + 1; row < $DOFS; ++row) {
                float value = factor[row * $DOFS + col];
#pragma unroll
                for (int k = 0; k < col; ++k)
                    value -= factor[row * $DOFS + k] * factor[col * $DOFS + k];
                factor[row * $DOFS + col] = value * inverse_diagonal;
            }
        }
    }
    __syncwarp();

    if (lane < $DOFS) {
        const int factor_base = group * $ELEMENTS;
#pragma unroll
        for (int col = 0; col <= lane; ++col)
            L_group.data[factor_base + lane * $DOFS + col] = factor[lane * $DOFS + col];
    }
#endif
""")

_FORCE_INIT = Template("""
    if (lane < $DOFS) {
        const int joint = articulation_start.data[articulation] + dof_joint_offset.data[lane];
        const auto force = wp::mul(body_I_c.data[joint_child.data[joint]], joint_S_s.data[dof_start + lane]);
        #pragma unroll
        for (int component = 0; component < 6; ++component)
            force_components[lane * 6 + component] = force.c[component];
    }
    __syncwarp();
""")

_COMPOSITE_INPUT = """
        const int source_code = (packed >> 8) & 255;
        float value = 0.0f;
        if (source_code != 0) {
            const int source = source_code - 1;
            const int projection = (packed >> 16) & 255;
            const auto motion = joint_S_s.data[dof_start + projection];
#pragma unroll
            for (int component = 0; component < 6; ++component)
                value += motion.c[component] * force_components[source * 6 + component];
        }
"""

_INPUTS = (
    "n_arts: int",
    "articulation_start: wp.array[int]",
    "articulation_dof_start: wp.array[int]",
    "mass_update_mask: wp.array[int]",
    "joint_child: wp.array[int]",
    "joint_S_s: wp.array[wp.spatial_vector]",
    "body_I_c: wp.array[wp.spatial_matrix]",
    "group_to_art: wp.array[int]",
    "R_group: wp.array2d[float]",
    "dof_joint_offset: wp.array[int]",
    "lower_schedule: wp.array[int]",
    "fused_augmented_drive: int",
    "drive_row_by_dof: wp.array[int]",
    "row_K: wp.array[float]",
)


@cache
def get_kernel(size: int, *, warps_per_block: int = 4, geometric: bool = False, geometric_index=()):
    """Bind composite-body or geometric mass to the same factor stages.

    The packed schedule addresses matrices of at most sixteen coordinates.
    Geometric entries are physical lower-triangle values before regularization
    and augmented drives. An index of -1 denotes an absent geometric entry.
    """
    size, warps_per_block = int(size), int(warps_per_block)
    if not 1 <= size <= 16 or not 1 <= warps_per_block <= 32:
        raise ValueError("Small Cholesky requires 1..16 DOFs and 1..32 warps per block")
    if geometric_index and (not geometric or len(geometric_index) != size * size or min(geometric_index) < -1):
        raise ValueError("Geometric factor requires a complete physical matrix index")
    elements = size * size
    inputs = list(_INPUTS)
    if geometric:
        inputs.append("geometric: wp.array2d[float]")
        mass_input = f"        float value = geometric.data[group * {elements} + element];"
        if geometric_index:
            inputs.append("geometry_lookup: wp.array[int]")
            mass_input = (
                "        const int geometry_entry=geometry_lookup.data[element];\n"
                "        float value = geometry_entry < 0 ? 0.0f : "
                f"geometric.data[group * {max(geometric_index) + 1} + geometry_entry];"
            )
    else:
        mass_input = _COMPOSITE_INPUT
    inputs.append("L_group: wp.array3d[float]")
    snippet = _FACTOR.substitute(
        DOFS=size,
        ELEMENTS=elements,
        FACTOR_STORAGE=warps_per_block * elements,
        LOWER=size * (size + 1) // 2,
        FORCE_STORAGE="" if geometric else f"    __shared__ float forces[{warps_per_block * size * 6}];",
        FORCE_POINTER="" if geometric else f"    float* force_components = forces + warp * {size * 6};",
        FORCE_INIT="" if geometric else _FORCE_INIT.substitute(DOFS=size),
        MASS_INPUT=mass_input,
    )
    name = f"{'geometric_' if geometric else ''}crba_cholesky_warp{warps_per_block}_{size}"
    signature = ",\n    ".join(inputs)
    arguments = ", ".join(item.split(":")[0] for item in inputs)
    # Author both ABIs directly; never inspect or rewrite a producer's Python.
    source = (
        f"@wp.func_native(snippet)\ndef factor_native(tid: int,\n    {signature}\n): ...\n\n"
        f"def {name}(\n    {signature}\n):\n    factor_native(wp.tid(), {arguments})\n"
    )
    filename = f"<small-cholesky-{hashlib.sha256((source + snippet).encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = {"wp": wp, "snippet": snippet}
    exec(compile(source, filename, "exec"), namespace)
    return wp.kernel(enable_backward=False, module="unique")(namespace[name])

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Feed current geometric H9 into the retained primary Cholesky arithmetic."""

import hashlib
import inspect
import linecache
from functools import cache


@cache
def _factory():
    # Keep the original default factory and its ABI untouched. This explicit
    # source seam changes only assembly input, not R/K or factor arithmetic.
    from . import solver_feather_pgs as original  # noqa: PLC0415

    source = inspect.getsource(original._get_crba_cholesky_warp_kernel)
    expected = "5d220f15ca5d48ee5b1feb2430e51d87de792aaa2b98128a2e78566b4fd003c7"
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError("Original Franka primary factor source changed; review the geometric input seam")
    source = source.replace("def _get_crba_cholesky_warp_kernel(", "def _geometric_factor(", 1)
    begin = source.index("    __shared__ float forces[")
    end = source.index("    for (int schedule_index", begin)
    source = (
        source[:begin]
        + "    float* factor = factors + warp * {matrix_elements};\n"
        + "    const int dof_start = articulation_dof_start.data[articulation];\n\n"
        + source[end:]
    )
    begin = source.index("        const int source_code =")
    end = source.index("        if (row == col)", begin)
    source = (
        source[:begin] + "        float value = geometric.data[group * {matrix_elements} + element];\n" + source[end:]
    )
    argument = "        L_group: wp.array3d[float],"
    if source.count(argument) != 2:
        raise ValueError("Original native/kernel factor signatures changed")
    source = source.replace(argument, "        geometric: wp.array2d[float],\n" + argument)
    call = "            row_K,\n            L_group,"
    if source.count(call) != 1:
        raise ValueError("Original factor native call changed")
    source = source.replace(call, "            row_K,\n            geometric,\n            L_group,")
    source = source.replace('name = f"crba_cholesky_warp', 'name = f"franka_kinetic_crba_cholesky_warp', 1)
    filename = "<franka-kinetic-geometric-factor>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(original.__dict__)
    exec(compile(source, filename, "exec"), namespace)
    return namespace["_geometric_factor"]


@cache
def get_kernel(device_arch: str, *, warps_per_block: int = 4):
    """Return only the admitted nine-DOF specialization, retaining original L9."""
    return _factory()(9, device_arch, warps_per_block=warps_per_block)

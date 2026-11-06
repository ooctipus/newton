# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free-body rows' headers and addressing.

The rows are the walk's (:mod:`.walk_rows`: the one writer walks every row's seeds into its whitened form ``Z``). This
module packs the free-body rows (:func:`.sweep_routes.free_body_rows`, rows over at most two free bodies) into the
row word (:func:`free_body_meta_kernel`, :func:`.row_metadata.pack_mf_header`, which the sweeps' row loop decodes:
its sides' coordinates, ``1 / A_ii``); their ``A_ii`` is the walk's (:func:`.walk_rows.write_free_body_rows`), their rhs the rows' law's.
The bilateral closure is the factor's (:mod:`.factor.closure`).
"""

from __future__ import annotations

from functools import cache

import warp as wp

from . import kernel_template
from .device_preludes import CPU_MATH, dual_native, lane_group
from .kernel_template import KernelTemplate
from .row_metadata import _validate_mf_metadata_encoding, pack_mf_header

# ---------------------------------------------------------------------------------------------------------------------
# Free-body rows
# ---------------------------------------------------------------------------------------------------------------------
#
# A free-body row acts on at most two free rigid bodies, ``a`` and ``b`` (a negative body for none), through six
# entries each. A free body is a one-node tree; the one writer walks each side's seed into ``Z = L^-1 J^T`` (its walk's
# factor, armature included), so ``A_ii = cfm + |Z_a|^2 + |Z_b|^2`` and the sweeps hold the change of the bodies'
# velocities in factor coordinates ``u = L^T dv``. A body's six velocity coordinates start at its articulation's first
# coordinate, in the addressing the mapping reads: global (``articulation_dof_start``) or world
# (``articulation_world_dof_offset``).


@cache
def free_body_meta_kernel(mf_max_constraints: int, pitch: int, base: int, lanes: int, device_arch: str) -> wp.Kernel:
    """Pack each world's free-body rows into the 128-bit word the matrix-free sweeps load (:func:`.row_metadata.pack_mf_header`).

    A group of ``lanes`` lanes per world (``wp.launch_tiled`` over the worlds: a warp, or a block when few worlds
    would idle the device, :func:`.execution.fill_lanes`; one thread on the CPU), a lane per row; the word holds the
    rows' sides' first world coordinates (``world_dof_offset`` of their articulations), the inverse of their stored
    diagonal ``A_ii`` (the step's form of the law, cached in the header), right-hand side (the law's plus the row's
    incident velocity: its residual at ``du = 0``), type and parent. The records are the row store's
    (``pitch`` rows a line, the free bodies' from ``base``), whose parents count in the line; the word's count from the
    free bodies' first row, as the sweeps stage their impulses. ``device_arch`` keys the variant.
    """
    _validate_mf_metadata_encoding(mf_max_constraints)
    word = pack_mf_header("packed", "da", "db", "inverse", "rhs_val", "rt", "par", "        ")
    body = f"""    int m_mf = mf_constraint_count.data[world];
    const int mf_line = first.data[world];
    __builtin_assume(mf_line >= 0);
    int off_meta = mf_line * {mf_max_constraints * 4};
    const int off_body = mf_line * {pitch} + {base};

    for (int i = LANE; i < m_mf; i += STRIDE) {{
        const wp::vec2i sides = row_sides.data[off_body + i];
        int da = sides.c[0] >= 0 ? world_dof_offset.data[sides.c[0]] : -1;
        int db = sides.c[1] >= 0 ? world_dof_offset.data[sides.c[1]] : -1;
        const float A = diag.data[off_body + i];
        float inverse = 0.0f;
        if (A > 0.0f) inverse = 1.0f / A;
        float rhs_val = rhs.data[off_body + i] + row_incident.data[off_body + i];
        int rt = row_type.data[off_body + i];
        int par = row_parent.data[off_body + i];
        if (par >= 0) par -= {base};

{word}
        *reinterpret_cast<int4*>(&mf_meta.data[off_meta + i * 4]) = packed;
    }}
"""
    return KernelTemplate(
        f"free_body_meta_{mf_max_constraints}_{pitch}_{lanes}",
        kernel_template.params(
            ("mf_constraint_count", wp.array[int]),
            ("first", wp.array[int]),
            ("world_dof_offset", wp.array[int]),
            ("row_sides", wp.array2d[wp.vec2i]),
            ("diag", wp.array2d[float]),
            ("rhs", wp.array2d[float]),
            ("row_incident", wp.array2d[float]),
            ("row_type", wp.array2d[int]),
            ("row_parent", wp.array2d[int]),
            ("mf_meta", wp.array2d[int]),
        ),
        dual_native(body, preludes=(lane_group(lanes), CPU_MATH)),
        dispatch=kernel_template.items("world"),
        composed=True,
        native_name="free_body_meta_native",
    ).kernel()

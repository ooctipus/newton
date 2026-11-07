# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The streamed rows' records: what the sweeps' row loops load a row ahead.

The rows are the walk's (:mod:`.walk_rows`: the one writer walks every row's seeds into its whitened form ``Z``). This
module packs every row whose metadata a sweep streams into its record (:func:`row_record_kernel`,
:func:`.row_metadata.pack_row_record`, which the sweeps' row loop decodes: its sides' coordinates and widths, its stored
diagonal, right-hand side, kind, parent and friction coefficient): the free bodies' rows (:func:`.sweep_routes.
free_body_rows`, rows over at most two free bodies; their ``A_ii`` is the walk's), and the dense rows where a world's
line holds more of them than the sweep stages. The bilateral closure is the factor's (:mod:`.factor.closure`).
"""

from __future__ import annotations

from functools import cache

import warp as wp

from . import kernel_template
from .device_preludes import CPU_MATH, dual_native, lane_group
from .kernel_template import KernelTemplate
from .row_metadata import RECORD_WORDS, _validate_mf_metadata_encoding, pack_row_record

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


def _pack_rows(count: str, first: str, base: int, inverse: bool, pad: str) -> str:
    """Pack rows ``[0, count)`` of a class whose first row is the line's ``base``, a lane a row; the class keeps the
    inverse of its diagonals (``inverse``: the free bodies') or the diagonals themselves (the dense rows')."""
    diagonal = "inverse" if inverse else "A"
    word = pack_row_record(
        f"&row_record.data[off_rec + ({base - int(first)} + i) * {RECORD_WORDS}]",
        "da", "db", "na", "nb", diagonal, "rhs_val", "rt", "par", "mu", pad + "    ",
    )  # fmt: skip
    inverse_line = f"{pad}    float inverse = 0.0f;\n{pad}    if (A > 0.0f) inverse = 1.0f / A;\n" if inverse else ""
    rebase = f"{pad}    if (par >= 0) par -= {base};\n" if base else ""
    return f"""{pad}for (int i = LANE; i < {count}; i += STRIDE) {{
{pad}    const int row = off_line + {base} + i;
{pad}    const wp::vec2i sides = row_sides.data[row];
{pad}    const int da = sides.c[0] >= 0 ? world_dof_offset.data[sides.c[0]] : -1;
{pad}    const int db = sides.c[1] >= 0 ? world_dof_offset.data[sides.c[1]] : -1;
{pad}    const int na = sides.c[0] >= 0 ? response_dofs.data[sides.c[0]] : 0;
{pad}    const int nb = sides.c[1] >= 0 ? response_dofs.data[sides.c[1]] : 0;
{pad}    const float A = diag.data[row];
{inverse_line}{pad}    const float rhs_val = rhs.data[row] + row_incident.data[row];
{pad}    const int rt = row_type.data[row];
{pad}    int par = row_parent.data[row];
{rebase}{pad}    const float mu = row_mu.data[row];
{word}
{pad}}}"""


@cache
def row_record_kernel(
    pitch: int,
    first: int,
    base: int,
    mf_rows: int,
    lanes: int,
    device_arch: str,
    *,
    skip_rows: int = 0,
    speed_rows: bool = False,
) -> wp.Kernel:
    """Pack each world's streamed rows into their records, which the sweeps' row loops load a row ahead
    (:func:`.row_metadata.pack_row_record`): the line's rows from ``first`` (``pitch`` rows a line), the dense rows'
    from 0 where a sweep streams them (``first`` 0), then the free bodies' from ``base``.

    A group of ``lanes`` lanes per world (``wp.launch_tiled`` over the worlds: a warp, or a block when few worlds
    would idle the device, :func:`.execution.fill_lanes`; one thread on the CPU), a lane per row; a record holds the
    row's sides' first world coordinates and widths, its stored diagonal (the free bodies' its inverse, cached for
    their step's form of the law; the dense rows' ``A_ii``), its right-hand side (the law's plus the row's incident
    velocity: its residual at ``du = 0``), kind, parent (counted from its class's first row, as the sweeps stage the
    class's rows) and friction coefficient. ``mf_rows`` and ``device_arch`` key the variant.

    Only the worlds the matrix-free sweep visits are packed: it returns for a world the tiers own (no free body's
    row, at most ``skip_rows`` rows of the line and, with the line's ``speed_rows``, none from ``speed_first``:
    :func:`.sweep_routes.mf_gs_source`'s skip), whose records no loop reads.
    """
    _validate_mf_metadata_encoding(max(int(mf_rows), int(base) if first < base else 0))
    dense = _pack_rows(f"(m_dense < {base} ? m_dense : {base})", first, 0, False, "    ") if first < base else ""
    free = _pack_rows("mf_constraint_count.data[world]", first, base, True, "    ") if mf_rows else ""
    skip = ""
    if skip_rows > 0:
        # The matrix-free sweep's skip of the tiers' worlds, on the row counts it clamps.
        speed = " && m_dense <= speed_first.data[world]" if speed_rows else ""
        skip = (
            f"    if (mf_constraint_count.data[world] == 0 && (m_dense < {base} ? m_dense : {base}) <= {int(skip_rows)}"
            f"{speed}) return;\n"
        )
    body = f"""    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int off_line = line * {pitch};
    const int off_rec = line * {(pitch - first) * RECORD_WORDS};
    const int m_dense = world_constraint_count.data[world];
{skip}{dense}
{free}
"""
    return KernelTemplate(
        f"row_record_{pitch}_{first}_{base}_{mf_rows}_{lanes}"
        + (f"_skip{int(skip_rows)}" if skip_rows > 0 else "")
        + ("_speed" if skip_rows > 0 and speed_rows else ""),
        kernel_template.params(
            ("world_constraint_count", wp.array[int]),
            ("mf_constraint_count", wp.array[int]),
            ("speed_first", wp.array[int]),
            ("dense_first", wp.array[int]),
            ("world_dof_offset", wp.array[int]),
            ("response_dofs", wp.array[int]),
            ("row_sides", wp.array2d[wp.vec2i]),
            ("diag", wp.array2d[float]),
            ("rhs", wp.array2d[float]),
            ("row_incident", wp.array2d[float]),
            ("row_type", wp.array2d[int]),
            ("row_parent", wp.array2d[int]),
            ("row_mu", wp.array2d[float]),
            ("row_record", wp.array2d[int]),
        ),
        dual_native(body, preludes=(lane_group(lanes), CPU_MATH)),
        dispatch=kernel_template.items("world"),
        composed=True,
        native_name="row_record_native",
    ).kernel()

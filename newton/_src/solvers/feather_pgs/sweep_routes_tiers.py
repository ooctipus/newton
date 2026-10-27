# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The parallel (Jacobi) tiers as configurations of the sweep emitter (:mod:`.sweep`).

:func:`parallel_kernel` takes the arguments of the solver's parallel tier factory and builds a kernel with its ABI
and dispatch (:data:`PARALLEL_ABI`, the world queue), returning the same tier descriptor, so a launch site can take
either kernel unchanged.

A tier owns each world without free-body rows whose dense rows it bounds: one block per world and a lane per row
(:class:`.sweep.JacobiRows`). The rows step together under the Jacobi law with Nesterov momentum
(:func:`.sweep.jacobi_sweeps`) over one of two response operators, each fed by the response kernels' rows or by rows
whitened in the kernel (:class:`.response_operator.WhitenedRows`, which keyed rows feed):

- the response block ``A = J Y`` in row layout (:class:`.response_operator.Block`);
- matrix-free, ``J`` in registers and ``Y^T`` staged (:class:`.response_operator.TransposedResponse`).
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import cache

import warp as wp

from . import kernel_template
from .device_preludes import RING, cuda_native, lane_group
from .kernel_template import KernelTemplate, SweepTier
from .law_source import law_preludes
from .projection_laws import PROJECTION_LAW
from .response_operator import Block, Lanes, TransposedResponse, WhitenedRows
from .row_metadata import _validate_dense_metadata_encoding
from .staging import Buffer, Staging
from .sweep import (
    CONTACT,
    JOINT_LIMIT,
    VELOCITY_LIMIT,
    JacobiRows,
    RowLaws,
    dense_tier_world,
    jacobi_limit_pass,
    jacobi_scale,
    jacobi_sweeps,
)
from .sweep_routes import _PHASE_DENSE_KINDS, MF_GS_ABI

PARALLEL_ABI = kernel_template.params(
    *MF_GS_ABI[:-1],
    ("ink_meta", wp.array[int]),
    ("ink_L_a", wp.array3d[float]),
    ("ink_L_b", wp.array3d[float]),
    ("ink_J_a", wp.array3d[float]),
    ("ink_J_b", wp.array3d[float]),
    ("world_row_cfm", wp.array2d[float]),
    ("v_out", wp.array[float]),
)
"""The parallel tiers' native ABI: the matrix-free sweep's, then the in-kernel response's operands (placeholders when
a tier builds none) before ``v_out``; keyed rows insert theirs after ``world_row_cfm``."""

TIER_SWEEPS_PER_ITERATION = Fraction(7, 2)
"""Sweeps of a 32-row tier per serial iteration (:func:`tier_sweeps`)."""
TIER_SWEEPS_CAP = 128
"""The largest automatic tier budget, unless the minimum is larger."""


def tier_sweeps(rows: int, iterations: int, sweeps: int) -> int:
    """The automatic sweep budget of a parallel tier serving up to ``rows`` rows.

    ``ceil(c * iterations * sqrt(rows / 32))`` for the factor ``c = p / q``
    (:data:`TIER_SWEEPS_PER_ITERATION`), at least ``sweeps`` and at most
    :data:`TIER_SWEEPS_CAP` unless ``sweeps`` is larger, and never fewer than
    one. The ceiling is exact: the least ``n`` with ``32 q^2 n^2 >= (p
    iterations)^2 rows``.
    """
    factor = TIER_SWEEPS_PER_ITERATION
    numerator = (factor.numerator * int(iterations)) ** 2 * int(rows)
    square = -(-numerator // (32 * factor.denominator**2))
    raw = math.isqrt(square - 1) + 1 if square > 0 else 0
    return max(1, int(sweeps), min(raw, TIER_SWEEPS_CAP))


def _phase_admission() -> str:
    """Clear ``admit`` for a row its row phase does not visit (velocity-limit rows stay in every phase but 3 and 4)."""
    return "\n".join(
        f"if (row_phase == {phase} && " + " && ".join(f"row_type != {kind}" for kind in kinds) + ") admit = 0;"
        for phase, kinds in _PHASE_DENSE_KINDS.items()
    )


def _operands(
    max_constraints: int,
    max_world_dofs: int,
    *,
    rows: int,
    min_rows: int,
    matrix_free: bool,
    has_drive_rows: bool,
    inkernel_response,
    exact_row_sums: bool,
    row_inputs,
):
    """Validate the tier's options as the factory it replaces does (same checks, order and messages); return the
    in-kernel response's group sizes and offsets and the keyed rows' operands (``None`` without)."""
    _validate_dense_metadata_encoding(max_constraints)
    M_D, D, AM = max_constraints, max_world_dofs, int(rows)
    if AM <= 0 or AM > M_D or AM > 128:
        raise ValueError("rows must be in [1, min(max_constraints, 128)]")
    if AM > 96 and not matrix_free:
        raise ValueError("rows above 96 need the matrix-free variant (the response block exceeds shared memory)")
    ink = inkernel_response is not None
    groups = tuple(int(v) for v in inkernel_response) if ink else (0, 0, 0, 0)
    NA, NB, OA, OB = groups
    if ink and (NA <= 0 or NB < 0 or OA + NA > D or (NB > 0 and OB + NB > D)):
        raise ValueError("inkernel_response needs group sizes within the world DOF count")
    if ink and not matrix_free and D * (AM + 4) > AM * (AM + 1):
        raise ValueError("in-kernel response (A-based) needs the response block storage to hold the padded Z scratch")
    if AM % 4 != 0:
        raise ValueError("rows must be a multiple of 4")
    if int(min_rows) < 0 or int(min_rows) >= AM:
        raise ValueError("min_rows must be in [0, rows)")
    operands = row_inputs(NA, NB, OA, OB, D, M_D) if row_inputs is not None else None
    if operands is not None:
        if not ink or not matrix_free or has_drive_rows:
            raise ValueError("generated factor rows require the ordinary matrix-free parallel law")
        if operands.stage is not None and matrix_free and exact_row_sums:
            raise ValueError("preprojected row staging does not implement exact row sums")
    return groups, operands


def parallel_source(
    max_constraints: int,
    max_world_dofs: int,
    *,
    rows: int,
    sweeps: int,
    nesterov: bool = True,
    min_rows: int = 0,
    tol: float = 1e-4,
    matrix_free: bool = False,
    has_drive_rows: bool = False,
    skip_local_internal_worlds: bool = False,
    inkernel_response: tuple[int, int, int, int] | None = None,
    exact_row_sums: bool = False,
    row_inputs=None,
    linked_rows: bool = False,
    velocity_limit_pass: bool = False,
) -> tuple[str, Staging]:
    """The native body of a parallel tier: a world's dense rows (at most ``rows``, more than ``min_rows``) under the
    Jacobi law, a thread per row.

    The block stages the rows (:class:`.sweep.JacobiRows`) and the velocity, then the response operator: the block
    ``A = J Y`` (row layout; ``rows`` above 96 need ``matrix_free``) or, ``matrix_free``, ``J`` in registers and
    ``Y^T`` staged, its step scale bounded by the exact row sums (``exact_row_sums``) or the column sums. With
    ``inkernel_response = (NA, NB, OA, OB)`` the rows are whitened in the kernel (:class:`.WhitenedRows`), from the
    group Jacobians or the keyed rows' operands (``row_inputs``: their metadata, physical rows and arguments). It
    runs ``sweeps`` Jacobi sweeps (:func:`.sweep.jacobi_sweeps`: momentum with ``nesterov``, pairs projected on their
    patch ring's pooled load with ``linked_rows``, a quiet change within ``tol``), with ``velocity_limit_pass`` the
    velocity limits' last word, then publishes the velocity and the impulses.

    Returns the body and its staging.
    """
    (NA, NB, OA, OB), operands = _operands(
        max_constraints,
        max_world_dofs,
        rows=rows,
        min_rows=min_rows,
        matrix_free=matrix_free,
        has_drive_rows=has_drive_rows,
        inkernel_response=inkernel_response,
        exact_row_sums=exact_row_sums,
        row_inputs=row_inputs,
    )
    M_D, D, AM = max_constraints, max_world_dofs, int(rows)
    NT = 32 * ((AM + 31) // 32)
    lanes = Lanes(NT)
    velocity_limit_pass = bool(velocity_limit_pass)
    laws = RowLaws(
        drive=has_drive_rows, velocity_limit_law="accumulated", regularize=False, jacobi_tolerance=float(tol)
    )
    store = JacobiRows(
        AM,
        NT,
        drive=has_drive_rows,
        limits=velocity_limit_pass,
        diagonal=None if inkernel_response is not None else "world_diag.data[off_dense + i]",
    )
    whitened = None
    if inkernel_response is not None:
        whitened = WhitenedRows(
            NA,
            NB,
            OA,
            OB,
            D,
            M_D,
            AM + 4,
            lanes,
            read=operands.physical_row if operands is not None else None,
            staged=operands.stage if operands is not None else None,
        )
    if matrix_free:
        op = TransposedResponse(D, AM, lanes, exact=bool(exact_row_sums), whitened=whitened)
    else:
        schedule = "rows" if whitened is not None or D > 32 else "pairs"
        op = Block(D, AM + 1, lanes, rows="n_rows", layout="row", schedule=schedule, whitened=whitened)
    dot = (Buffer("s_dot", "float", NT // 32),) if NT > 32 and nesterov else ()
    staging = Staging(
        (Buffer("s_v", "float", D), *store.buffers(), *op.buffers(AM), *dot, *store.drive_buffers()), threads=NT
    )
    metadata = {
        "rhs": "rhs_bias.data[off_dense + i]",
        "row_type": "world_row_type.data[off_dense + i]",
        "parent": "world_row_parent.data[off_dense + i]",
        "mu": "world_row_mu.data[off_dense + i]",
    }
    if operands is not None:
        metadata = {name: getattr(operands, name) for name in metadata}
    loads = store.loads(
        **metadata,
        unilateral=(CONTACT, JOINT_LIMIT, VELOCITY_LIMIT) if laws.accumulated else (CONTACT, JOINT_LIMIT),
        admission=_phase_admission(),
    )
    cfm = "world_row_cfm.data[off_dense + i]" if whitened is not None else None

    def scale(aii, bound, pad):
        return jacobi_scale(aii, bound, cfm=cfm, pad=pad)

    last_word = jacobi_limit_pass(store, op, laws) if velocity_limit_pass else ""
    body = f"""    const int lane = LANE;
{dense_tier_world(AM, int(min_rows), D, M_D, local=skip_local_internal_worlds, rows="n_rows")}

{staging.declarations("    ")}
{op.registers("    ")}

{loads}
    for (int d = lane; d < {D}; d += {NT}) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;
    }}
    SYNC();

{op.jacobi_stage(store.padding(), scale, "    ")}

{jacobi_sweeps(store, op, laws, sweeps=int(sweeps), momentum=bool(nesterov), pooled=bool(linked_rows))}

{last_word}
{op.publish(indent="    ", applied=lambda row: f"s_x[{row}] - s_lam0[{row}]")}
{store.stores()}
    // The world queue's next world restages the shared rows: every lane has read this world's.
    SYNC();
#endif
"""
    row_preludes = operands.preludes if operands is not None else ()
    preludes = (lane_group(NT), *((RING,) if linked_rows else ()), *row_preludes, *law_preludes(PROJECTION_LAW))
    return cuda_native(body, preludes=preludes), staging


@cache
def parallel_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    rows: int,
    sweeps: int,
    nesterov: bool = True,
    min_rows: int = 0,
    tol: float = 1e-4,
    matrix_free: bool = False,
    has_drive_rows: bool = False,
    has_dense_velocity_limit_rows: bool = False,
    skip_local_internal_worlds: bool = False,
    inkernel_response: tuple[int, int, int, int] | None = None,
    exact_row_sums: bool = False,
    row_inputs=None,
    linked_rows: bool = False,
    velocity_limit_pass: bool = False,
) -> SweepTier:
    """A parallel tier (the arguments, ABI and tier descriptor of the factory it replaces).

    A thread per row: the block's threads cover ``rows`` in whole warps. ``has_dense_velocity_limit_rows`` and
    ``device_arch`` are unused, ``mf_max_constraints`` names the kernel only: the tier owns no world with
    free-body rows and admits its velocity-limit rows as accumulated unilateral rows whatever the serial sweeps'
    law.
    """
    options = {
        "rows": rows,
        "min_rows": min_rows,
        "matrix_free": matrix_free,
        "has_drive_rows": has_drive_rows,
        "inkernel_response": inkernel_response,
        "exact_row_sums": exact_row_sums,
        "row_inputs": row_inputs,
    }
    snippet, _staging = parallel_source(
        max_constraints,
        max_world_dofs,
        **options,
        sweeps=sweeps,
        nesterov=nesterov,
        tol=tol,
        skip_local_internal_worlds=skip_local_internal_worlds,
        linked_rows=linked_rows,
        velocity_limit_pass=velocity_limit_pass,
    )
    (NA, NB, OA, OB), operands = _operands(max_constraints, max_world_dofs, **options)
    AM = int(rows)
    NT = 32 * ((AM + 31) // 32)
    ink = inkernel_response is not None
    name = (
        f"sweep_{'parallel' if row_inputs is None else 'keyed_parallel'}_{max_constraints}_{mf_max_constraints}"
        f"_{max_world_dofs}_rows{AM}_min{int(min_rows)}_sweeps{int(sweeps)}_nes{int(bool(nesterov))}"
        f"_tol{float(tol):g}_nt{NT}_mf{int(bool(matrix_free))}"
        + (f"_ink{NA}_{NB}_o{OA}_{OB}" if ink else "")
        + f"_hd{int(bool(has_drive_rows))}_z{int(ink and not matrix_free)}_ex{int(bool(matrix_free and exact_row_sums))}"
        + ("_local_fallback" if skip_local_internal_worlds else "")
        + ("_linked" if linked_rows else "")
        + ("_lw" if velocity_limit_pass else "")
    )
    kernel = KernelTemplate.from_abi(
        name,
        PARALLEL_ABI,
        snippet,
        kernel_template.WORLD_QUEUE,
        extra=operands.arguments if operands is not None else (),
        composed=True,
        native_name="sweep_parallel_native",
    ).kernel()
    return SweepTier(
        kernel,
        block_dim=NT,
        response_inputs=True,
        kinetic_rows=row_inputs is not None,
        linked_rows=bool(linked_rows),
        velocity_limit_pass=bool(velocity_limit_pass),
        velocity_limit_law="accumulated",
        sweeps=int(sweeps),
    )

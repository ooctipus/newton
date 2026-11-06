# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The parallel (Jacobi) tiers as configurations of the sweep emitter (:mod:`.sweep`).

:func:`parallel_kernel` builds a tier's kernel (:data:`PARALLEL_ABI`, the world queue) and returns its launch
descriptor (:class:`.kernel_template.SweepTier`).

A tier owns each world without free-body rows whose dense rows it bounds: one block per world and a lane per row
(:class:`.sweep.JacobiRows`). The rows are the walk's (:class:`.response_operator.WalkedRows`): the keyed rows walk their
seeds in the kernel, and the other rows are the one writer's stored ``Z`` (:mod:`.walk_rows`). They step together under
the Jacobi law with Nesterov momentum (:func:`.sweep.jacobi_sweeps`) over the matrix-free response, ``Z`` in registers
and staged transposed (:class:`.response_operator.TransposedResponse`), and leave by the walk down.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from functools import cache

import warp as wp

from ...core.execution import Work, device_lanes, shape
from ...core.memory import Buffer, Staging, arch_budget, built_for
from . import kernel_template
from .device_preludes import RING, SHARED_STORAGE, VirtualLanes, group_native, lane_group
from .factor.walk import walk_prelude, walk_row_prelude, walk_tables
from .kernel_template import KernelTemplate, SweepTier
from .law_source import law_preludes
from .projection_laws import CONE_LAW, PROJECTION_LAW
from .response_operator import Lanes, TransposedResponse, WalkedRows
from .row_metadata import _validate_dense_metadata_encoding
from .solve_policy import DEFAULT_POLICY, SolvePolicy, block_argmax
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
    material_arguments,
)
from .sweep_routes import MF_GS_ABI, worlds
from .tree_scan import WALK_BOTH, WalkClass, classes_half

PARALLEL_ABI = kernel_template.params(
    *MF_GS_ABI[:-1],
    ("ink_meta", wp.array[int]),
    ("cfm", float),
    ("v_out", wp.array[float]),
)
"""The parallel tiers' native ABI: the matrix-free sweep's (the walk among it), then the worlds' trees (``ink_meta``:
each world's articulation of each size and its first coordinate) and the rows' law's cfm before ``v_out``; keyed rows
insert theirs after ``cfm``."""

# The parallel tiers' automatic budget (tier_sweeps) grows with the serial sweep's iteration count, whose accuracy
# every tier must match, and with the square root of a tier's row bound, as Jacobi sweeps with momentum need more passes
# where more rows couple through one body. One factor serves both friction laws, since every tier steps each friction
# pair isotropically. It is the smallest, in eighths, that brings each evaluated row bucket (>= 100 world-samples) of
# four contact-rich workloads, allegro (12 iterations) and anymal-D flat (8) under point and patch friction, to the
# serial sweep's accuracy: per bucket, the p90 energy-norm velocity error against a converged serial reference over the
# serial route's, 95% world-bootstrap CI upper bound at most 1, each world's error interpolated log-linearly between
# measured sweep caps under the exact exit law. The point workload's sparse 97-128-row tier
# sets it: 27/8 left its worlds at 0.95x (CI upper 1.06), 7/2 gives 0.87x (0.99); its other tiers reach 0.23x, 0.50x and
# 0.70x, the patch tiers 0.23x, 0.49x and 0.65x, and at 8 iterations every tier stays below 0.62x. The exit law sets the
# small tiers' floor: at a tolerance of 1e-4 the 32-row point tier stalls near 0.87x the serial error whatever its
# budget, which had made the patch factor 15/4. The cap bounds the cost at high iteration counts; the conformance gate's
# 64-iteration scenes pass at it. A fraction keeps the budget in exact integer arithmetic.
TIER_SWEEPS_PER_ITERATION = Fraction(7, 2)
"""Sweeps of a 32-row tier per serial iteration (:func:`tier_sweeps`)."""
TIER_SWEEPS_CAP = 128
"""The largest tier budget."""
TIER_MAX_ROWS = 128
"""The most rows a tier holds: a lane per row in four warps, its staged block within the static budget."""
TIER_ROWS = 32
"""The rows a tier holds past the tier below it, a warp's: tier ``t`` serves ``(32 t, 32 (t + 1)]`` rows
(:func:`.sweep_levels.claim_worlds` buckets its worlds)."""
REGISTER_LINE = 80
"""The register line of a tier: 24 warps a multiprocessor (6 in each of the register file's 4 sub-partitions). It is
no cap: a tier's variants compile on both sides of it unbounded (72 registers plain, 100 to 108 keyed) and spill under
it, so a tier is bound only where its shared memory keeps no more blocks than the line (the layer's ``Work.cap``),
which leaves ptxas spare registers for the sweep's load schedule."""


def tier_sweeps(rows: int, iterations: int) -> int:
    """The sweep budget of a parallel tier serving up to ``rows`` rows: the equal-accuracy rule.

    ``ceil(c * iterations * sqrt(rows / 32))`` for the factor ``c = p / q``
    (:data:`TIER_SWEEPS_PER_ITERATION`), at most :data:`TIER_SWEEPS_CAP`: none
    at no iterations, as the serial sweeps run none. The ceiling is exact: the
    least ``n`` with ``32 q^2 n^2 >= (p iterations)^2 rows``.
    """
    factor = TIER_SWEEPS_PER_ITERATION
    numerator = (factor.numerator * int(iterations)) ** 2 * int(rows)
    square = -(-numerator // (32 * factor.denominator**2))
    raw = math.isqrt(square - 1) + 1 if square > 0 else 0
    return min(raw, TIER_SWEEPS_CAP)


def _operands(
    max_constraints: int,
    max_world_dofs: int,
    *,
    rows: int,
    min_rows: int,
    inkernel_response: tuple[int, int, int, int],
    row_inputs,
):
    """Validate the tier's options; return the worlds' tree sizes and offsets and the keyed rows' operands (``None``
    without)."""
    _validate_dense_metadata_encoding(max_constraints)
    M_D, D, AM = max_constraints, max_world_dofs, int(rows)
    if AM <= 0 or AM > M_D or AM > TIER_MAX_ROWS:
        raise ValueError(f"rows must be in [1, min(max_constraints, {TIER_MAX_ROWS})]")
    groups = tuple(int(v) for v in inkernel_response)
    NA, NB, OA, OB = groups
    if NA <= 0 or NB < 0 or OA + NA > D or (NB > 0 and OB + NB > D):
        raise ValueError("inkernel_response needs group sizes within the world DOF count")
    if AM % 4 != 0:
        raise ValueError("rows must be a multiple of 4")
    if int(min_rows) < 0 or int(min_rows) >= AM:
        raise ValueError("min_rows must be in [0, rows)")
    operands = row_inputs(NA, NB, OA, OB, D, M_D) if row_inputs is not None else None
    return groups, operands


def _stored_row(pitch: int, dofs: int) -> str:
    """Read the lane's row as the one writer stored it (:mod:`.walk_rows`: the packed sink at its world coordinates, the
    slips' frame applied; ``pitch`` rows a world's line) and its incident velocity ``J v_hat`` (``row_incident``)."""
    return (
        f"        const float* Z_row = &packed_Z.data[((size_t)line * {pitch} + i) * packed_Z.shape[2]];\n"
        "        #pragma unroll\n"
        f"        for (int q = 0; q < {dofs}; ++q) Zi[q * ZS] = Z_row[q];\n"
        "        incident = row_incident.data[off_dense + i];"
    )


def parallel_source(
    max_constraints: int,
    max_world_dofs: int,
    *,
    rows: int,
    sweeps: int,
    inkernel_response: tuple[int, int, int, int],
    min_rows: int = 0,
    row_inputs=None,
    linked_rows: bool = False,
    velocity_limit_pass: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
    frame: tuple[int, int] = (0, 0),
    classes: tuple[WalkClass, ...] = (),
    pitch: int | None = None,
    lanes: int = 32,
) -> tuple[str, Staging]:
    """The native body of a parallel tier: a world's dense rows (at most ``rows``, more than ``min_rows``) under the
    Jacobi law, a lane per row; under ``torsion`` a contact's spin row steps and projects with its pair. ``lanes`` is
    the device's (:func:`.execution.device_lanes`): CUDA's lanes are the block's threads, the CPU's one thread runs
    them as virtual lanes (:class:`.device_preludes.VirtualLanes`), each phase between the block's barriers a loop.

    The block stages the rows (:class:`.sweep.JacobiRows`) and the velocity, then the rows in the walk's coordinates
    (:class:`.WalkedRows`, the world's trees ``inkernel_response = (NA, NB, OA, OB)``): the keyed rows' operands
    (``row_inputs``: their metadata, their walked rows and arguments) walk their seeds in the kernel, with the release
    of each tree that can slip (``frame``: the class's slip capacity of each tree; FL3.4's release ``R``, the leave
    ``R^T``); the other rows are the one writer's stored ``Z``, its frame ``F`` included (the leave ``F^T``). They form the matrix-free response (:class:`.TransposedResponse`: ``Z`` in
    registers and staged transposed, its step scale bounded by the column sums). It runs at most ``sweeps`` Jacobi
    sweeps (:func:`.sweep.jacobi_sweeps`: Nesterov momentum, pairs projected on their patch ring's pooled load with
    ``linked_rows``, a world stopped under ``solve_policy``), with ``velocity_limit_pass`` the velocity limits' last
    word, then publishes the velocity by the walk down and the impulses. With ``weighted`` contact rows step by their
    weights (the relative regularization's; ``material``: a material contact's ``d / (d + eps)`` from the diagonal the
    tier forms). Keyed rows walk their trees by the walking ``classes``' path tables in constant memory
    (:func:`.factor.walk.walk_prelude`, the half their records share, :func:`.tree_scan.classes_half`).

    Returns the body and its staging.
    """
    (NA, NB, OA, OB), operands = _operands(
        max_constraints,
        max_world_dofs,
        rows=rows,
        min_rows=min_rows,
        inkernel_response=inkernel_response,
        row_inputs=row_inputs,
    )
    M_D, D, AM = max_constraints, max_world_dofs, int(rows)
    NT = 32 * ((AM + 31) // 32)
    if lanes not in (1, 32):
        raise ValueError(f"a tier runs its rows on a block's warps or the CPU's one thread, got {lanes} lanes")
    serial = lanes == 1
    velocity_limit_pass = bool(velocity_limit_pass)
    laws = RowLaws(regularize=bool(weighted), material=bool(material), torsion=torsion, policy=solve_policy)
    store = JacobiRows(
        AM, NT, limits=velocity_limit_pass, types=solve_policy.traces, weighted=bool(weighted), torsion=torsion
    )
    row_lanes = Lanes(NT)
    walked = WalkedRows(
        NA,
        NB,
        OA,
        OB,
        D,
        AM + 4,
        row_lanes,
        read=operands.read if operands is not None else _stored_row(M_D if pitch is None else int(pitch), D),
        walked=operands is not None,
        frame=tuple(int(cap) for cap in frame),
        classes=len(classes),
    )
    op = TransposedResponse(D, AM, walked, row_lanes)
    # Each lane's row registers; on the CPU arrays over the virtual lanes.
    group = VirtualLanes(NT, serial, (*op.registers(), *store.registers()))
    store = dataclasses.replace(store, group=group)
    op = dataclasses.replace(op, group=group)
    dot = (Buffer("s_dot", "float", NT // 32),) if NT > 32 and not serial else ()
    # A block of several warps reduces the trace's largest change over its warps.
    trace_reduction, trace = "", ()
    if solve_policy.traces and NT > 32 and not serial:
        trace_reduction, trace = block_argmax(NT // 32, warp="WARP", lane="(int)(threadIdx.x & 31)")
    staging = Staging(
        (Buffer("s_v", "float", D), *store.buffers(), *op.buffers(AM), *dot, *trace),
        threads=NT,
    )
    metadata = {
        "rhs": "rhs_bias.data[off_dense + i]",
        "row_type": "world_row_type.data[off_dense + i]",
        "parent": "world_row_parent.data[off_dense + i]",
        "mu": "world_row_mu.data[off_dense + i]",
    }
    if operands is not None:
        metadata = {name: getattr(operands, name) for name in metadata}
    loads = store.loads(**metadata, unilateral=(CONTACT, JOINT_LIMIT, VELOCITY_LIMIT))

    def scale(aii, bound, pad):
        return jacobi_scale(aii, bound, cfm="cfm", pad=pad)

    # The velocity limits' last word closes the sweeps: none without a sweep, as the serial sweeps' at no iterations.
    last_word = jacobi_limit_pass(store, op) if velocity_limit_pass and sweeps else ""
    velocity = f"""    for (int d = lane; d < {D}; d += {NT}) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;
    }}"""
    publish = op.publish(indent="    ", applied=lambda row: f"s_x[{row}]") + "\n" + store.stores()
    body = f"""    const int lane = LANE;
{dense_tier_world(D, M_D, rows="n_rows", support=D, pitch=M_D if pitch is None else int(pitch))}

{staging.declarations("    ")}{trace_reduction}
{op.declarations("    ")}

{group.phase(loads + chr(10) + velocity)}
{group.barrier("    ")}

{op.jacobi_stage(store.padding(), scale, "    ", idle=store.idle(), counted=store.counted)}

{jacobi_sweeps(store, op, laws, sweeps=int(sweeps), pooled=bool(linked_rows), scale=scale)}

{last_word}
{group.phase(publish)}
    // The world queue's next world restages the shared rows: every lane has read this world's.
{group.barrier("    ")}
"""
    row_preludes = operands.preludes if operands is not None else ()
    walk = ()
    if operands is not None:
        if not classes:
            raise ValueError("a tier that walks its rows walks the classes of their trees")
        # The walk last, as the general sweep declares it (sweep_routes.mf_gs_source).
        walk = (walk_prelude(walk_tables(classes), 1, classes_half(classes)), walk_row_prelude(1))
    preludes = (
        lane_group(NT),
        SHARED_STORAGE,
        *((RING,) if linked_rows else ()),
        *row_preludes,
        *law_preludes(PROJECTION_LAW, *((CONE_LAW,) if torsion else ())),
    )
    return group_native(body, preludes=(*preludes, *solve_policy.preludes, *walk), lanes=lanes), staging


@cache
def parallel_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    rows: int,
    sweeps: int,
    inkernel_response: tuple[int, int, int, int],
    min_rows: int = 0,
    row_inputs=None,
    linked_rows: bool = False,
    velocity_limit_pass: bool = False,
    torsion: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    weighted: bool = False,
    material: bool = False,
    frame: tuple[int, int] = (0, 0),
    classes: tuple[WalkClass, ...] = (),
    pitch: int | None = None,
) -> SweepTier:
    """A parallel tier: its kernel (:data:`PARALLEL_ABI`) and its launch descriptor.

    A thread per row: the block's threads cover ``rows`` in whole warps, and its working set fits the static budget
    of ``device_arch`` (:func:`.memory.arch_budget`). The tier owns no world with free-body rows (its kernel's name
    carries the solver's free-body capacity ``mf_max_constraints``) and admits its velocity-limit rows as unilateral
    rows. Its worlds' stored rows (:mod:`.walk_rows`) lie at their coordinates, ``max_world_dofs`` entries a row.
    With ``weighted`` its contact rows step by their weights, ``material`` contacts' from their compliance; ``frame``
    is each tree's slip capacity (:class:`.WalkedRows`); ``classes`` the walking classes whose trees its keyed rows
    walk (:attr:`.tree_factor.TreeFactor.walk_classes`). It walks its bucket of the claims' worlds
    (:func:`.sweep_routes.worlds`, :func:`.sweep_levels.claim_worlds`), which holds its worlds only.
    """
    if material and not weighted:
        raise ValueError("material contacts step by their weights: their tiers are weighted")
    options = {
        "rows": rows,
        "min_rows": min_rows,
        "inkernel_response": inkernel_response,
        "row_inputs": row_inputs,
    }
    snippet, staging = parallel_source(
        max_constraints,
        max_world_dofs,
        **options,
        sweeps=sweeps,
        linked_rows=linked_rows,
        velocity_limit_pass=velocity_limit_pass,
        torsion=torsion,
        solve_policy=solve_policy,
        weighted=weighted,
        material=material,
        frame=frame,
        classes=classes,
        pitch=pitch,
        lanes=device_lanes(device_arch),
    )
    staging.check("the parallel tier", arch_budget(device_arch))
    (NA, NB, OA, OB), operands = _operands(max_constraints, max_world_dofs, **options)
    AM = int(rows)
    NT = 32 * ((AM + 31) // 32)
    walk_half = classes_half(classes) if classes else WALK_BOTH
    # A block a world, a thread a row. Not persistent: an uncapped line does not fix the tier's residency, and a grid
    # of one wave by the line is 1.2 waves at a keyed tier's 96 registers (allegro's 64-row tier 5% slower, PRO).
    work = Work(0, lanes=NT, block=NT, registers=REGISTER_LINE, cap=False, shared=staging.bytes)
    name = (
        f"sweep_{'parallel' if row_inputs is None else 'keyed_parallel'}_{max_constraints}_{mf_max_constraints}"
        f"_{max_world_dofs}_rows{AM}_min{int(min_rows)}_sweeps{int(sweeps)}_nt{NT}_ink{NA}_{NB}_o{OA}_{OB}"
        + (f"_frame{int(frame[0])}_{int(frame[1])}" if any(frame) else "")
        + ("" if walk_half == WALK_BOTH else f"_half{walk_half}")
        + ("" if pitch is None or pitch == max_constraints else f"_p{int(pitch)}")
        + ("_linked" if linked_rows else "")
        + ("_lw" if velocity_limit_pass else "")
        + ("_weighted" if weighted else "")
        + ("_material" if material else "")
        + ("_torsion" if torsion else "")
        + solve_policy.suffix
    )
    kernel = KernelTemplate.from_abi(
        name,
        PARALLEL_ABI,
        snippet,
        worlds(True),
        extra=(
            *(operands.arguments if operands is not None else ()),
            *material_arguments(material),
            *solve_policy.arguments,
        ),
        composed=True,
        native_name="sweep_parallel_native",
        launch_bounds=shape(work, built_for(device_arch)).bounds,
    ).kernel()
    return SweepTier(
        kernel,
        work,
        kinetic_rows=row_inputs is not None,
        linked_rows=bool(linked_rows),
        velocity_limit_pass=bool(velocity_limit_pass),
        sweeps=int(sweeps),
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The parallel (Jacobi) tiers as configurations of the sweep emitter (:mod:`.sweep`).

:class:`TierSweep` is a tier's configuration, which the one factory builds (:func:`.sweep.sweep_kernel`:
:data:`PARALLEL_ABI`, the world queue).

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
from typing import ClassVar

import warp as wp

from ...core.execution import Work, device_lanes
from ...core.memory import Buffer, Staging
from . import kernel_template
from .device_preludes import VirtualLanes
from .factor.walk import walk_prelude, walk_row_prelude, walk_tables
from .kernel_template import Dispatch
from .keyed_row_layout import KeyedLayout
from .keyed_rows import get_direct_parallel_inputs
from .response_operator import Lanes, TransposedResponse, WalkedRows
from .row_inputs import RowArgument
from .row_laws import ROW_CONTACT, ROW_JOINT_LIMIT, ROW_JOINT_VELOCITY_LIMIT
from .row_metadata import _validate_dense_metadata_encoding
from .solve_policy import block_argmax
from .sweep import JacobiRows, SweepConfig, dense_tier_world, jacobi_limit_pass, jacobi_scale, jacobi_sweeps
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


def _stored_row(pitch: int, dofs: int) -> str:
    """Read the lane's row as the one writer stored it (:mod:`.walk_rows`: the packed sink at its world coordinates, the
    slips' frame applied; ``pitch`` rows a world's line) and its incident velocity ``J v_hat`` (``row_incident``)."""
    return (
        f"        const float* Z_row = &packed_Z.data[((size_t)line * {pitch} + i) * packed_Z.shape[2]];\n"
        "        #pragma unroll\n"
        f"        for (int q = 0; q < {dofs}; ++q) Zi[q * ZS] = Z_row[q];\n"
        "        incident = row_incident.data[off_dense + i];"
    )


def parallel_source(tier: TierSweep) -> tuple[str, Staging]:
    """The native body of a parallel tier: a world's dense rows (at most ``rows``: its claim's bucket holds the rest)
    under the Jacobi law, a lane per row (``tier``, :class:`TierSweep`'s source); under ``torsion`` a contact's spin
    row steps and projects with its pair. The lanes are the device's (:func:`.execution.device_lanes`): CUDA's lanes are the block's threads, the CPU's one
    thread runs them as virtual lanes (:class:`.device_preludes.VirtualLanes`), each phase between the block's barriers
    a loop.

    The block stages the rows (:class:`.sweep.JacobiRows`) and the velocity, then the rows in the walk's coordinates
    (:class:`.WalkedRows`, the world's trees ``inkernel_response = (NA, NB, OA, OB)``): the keyed rows' operands
    (:attr:`TierSweep.row_inputs`: their metadata, their walked rows and arguments) walk their seeds in the kernel, with the release
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
    NA, NB, OA, OB = tier.inkernel_response
    operands = tier.row_inputs
    M_D, D, AM, NT = tier.max_constraints, tier.max_world_dofs, tier.rows, tier.threads
    pitch, frame, classes, sweeps = tier.pitch, tier.frame, tier.classes, tier.sweeps
    torsion, solve_policy, linked_rows = tier.torsion, tier.solve_policy, tier.linked_rows
    serial = device_lanes(tier.device_arch) == 1
    velocity_limit_pass = tier.velocity_limit_pass
    laws = tier.laws
    store = JacobiRows(
        AM, NT, limits=velocity_limit_pass, types=solve_policy.traces, weighted=tier.weighted, torsion=torsion
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
    loads = store.loads(**metadata, unilateral=(ROW_CONTACT, ROW_JOINT_LIMIT, ROW_JOINT_VELOCITY_LIMIT))

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
        # The walk last, as the general sweep declares it (sweep_routes.mf_gs_source).
        walk = (walk_prelude(walk_tables(classes), 1, classes_half(classes)), walk_row_prelude(1))
    lanes = device_lanes(tier.device_arch)
    return tier.native(body, lanes=lanes, group=NT, ring=bool(linked_rows), rows=row_preludes, walks=walk), staging


@dataclasses.dataclass(frozen=True)
class TierSweep(SweepConfig):
    """A parallel tier's configuration (case K4): a world's dense rows (at most ``rows``, its claim's bucket) under the
    Jacobi law, a lane a row, ``sweeps`` sweeps at most (:func:`parallel_source`).

    Its switches, each the plan's from the class's shape and the laws (:meth:`.sweep_plan.Sweep.build`):

    - the level law: a world's rows one split (Jacobi) level, stepped at once in the Gershgorin metric with momentum,
      ``sweeps`` its budget (:func:`tier_sweeps` of its row bound); ``rows`` the bucket's bound;
    - the producer's sink: with a ``keyed`` layout (:class:`.keyed_row_layout.KeyedLayout`) the rows are walked from
      their keys in the kernel's prologue (:func:`.keyed_rows.get_direct_parallel_inputs`), else read as the one writer
      stored them (``pitch`` rows a world's line);
    - the rows' trees: ``inkernel_response`` ``(NA, NB, OA, OB)``, their slip capacities ``frame`` and walking
      ``classes``;
    - ``linked_rows``: pairs project on their patch ring's pooled load; ``velocity_limit_pass``: the velocity limits'
      last word.

    A tier owns no world with free-body rows (its kernel's name carries the solver's free-body capacity
    ``mf_max_constraints``) and admits its velocity-limit rows as unilateral rows. Its worlds' stored rows
    (:mod:`.walk_rows`) lie at their coordinates, ``max_world_dofs`` entries a row. A block a world, a thread a row
    (the CPU's one thread runs them as virtual lanes); its working set fits the static budget of ``device_arch``. It
    walks its bucket of the claims' worlds (:func:`.sweep_routes.worlds`, :func:`.sweep_levels.claim_worlds`).
    """

    max_constraints: int
    mf_max_constraints: int
    max_world_dofs: int
    device_arch: str
    _: dataclasses.KW_ONLY
    rows: int
    sweeps: int
    inkernel_response: tuple[int, int, int, int]
    keyed: KeyedLayout | None = None
    linked_rows: bool = False
    velocity_limit_pass: bool = False
    frame: tuple[int, int] = (0, 0)
    classes: tuple[WalkClass, ...] = ()
    pitch: int | None = None

    ABI: ClassVar[tuple] = PARALLEL_ABI
    NATIVE: ClassVar[str] = "sweep_parallel_native"
    LABEL: ClassVar[str] = "the parallel tier"

    def __post_init__(self):
        if self.material and not self.weighted:
            raise ValueError("material contacts step by their weights: their tiers are weighted")
        _validate_dense_metadata_encoding(self.max_constraints)
        M_D, D, AM = self.max_constraints, self.max_world_dofs, int(self.rows)
        if AM <= 0 or AM > M_D or AM > TIER_MAX_ROWS:
            raise ValueError(f"rows must be in [1, min(max_constraints, {TIER_MAX_ROWS})]")
        NA, NB, OA, OB = (int(v) for v in self.inkernel_response)
        if NA <= 0 or NB < 0 or OA + NA > D or (NB > 0 and OB + NB > D):
            raise ValueError("inkernel_response needs group sizes within the world DOF count")
        if AM % 4 != 0:
            raise ValueError("rows must be a multiple of 4")
        if self.keyed is not None and not self.classes:
            raise ValueError("a tier that walks its rows walks the classes of their trees")
        object.__setattr__(self, "inkernel_response", (NA, NB, OA, OB))
        object.__setattr__(self, "frame", tuple(int(cap) for cap in self.frame))

    @property
    def threads(self) -> int:
        """A block's threads: the rows in whole warps."""
        return 32 * ((int(self.rows) + 31) // 32)

    @property
    def row_inputs(self):
        """The keyed rows' operands (:class:`.row_inputs.ParallelRowInputs`), ``None`` for stored rows."""
        if self.keyed is None:
            return None
        return get_direct_parallel_inputs(
            self.keyed,
            *self.inkernel_response,
            self.max_world_dofs,
            self.max_constraints,
            weighted=self.weighted,
            torsion=self.torsion,
        )

    @property
    def name(self) -> str:
        NA, NB, OA, OB = self.inkernel_response
        walk_half = classes_half(self.classes) if self.classes else WALK_BOTH
        frame, pitch = self.frame, self.pitch
        return (
            f"sweep_{'parallel' if self.keyed is None else 'keyed_parallel'}_{self.max_constraints}_"
            f"{self.mf_max_constraints}_{self.max_world_dofs}_rows{int(self.rows)}_sweeps{int(self.sweeps)}"
            f"_nt{self.threads}_ink{NA}_{NB}_o{OA}_{OB}"
            + (f"_frame{int(frame[0])}_{int(frame[1])}" if any(frame) else "")
            + ("" if walk_half == WALK_BOTH else f"_half{walk_half}")
            + ("" if pitch is None or pitch == self.max_constraints else f"_p{int(pitch)}")
            + ("_linked" if self.linked_rows else "")
            + ("_lw" if self.velocity_limit_pass else "")
            + ("_weighted" if self.weighted else "")
            + ("_material" if self.material else "")
            + ("_torsion" if self.torsion else "")
            + self.solve_policy.suffix
        )

    source = parallel_source

    def dispatch(self, staging: Staging) -> Dispatch:
        return worlds()

    def operands(self) -> tuple[RowArgument, ...]:
        inputs = self.row_inputs
        return (*(inputs.arguments if inputs is not None else ()), *self.law_operands())

    def work(self, items: int = 0, shared: int = 0) -> Work:
        """A block a world, a thread a row. Not persistent: an uncapped line does not fix the tier's residency, and a
        grid of one wave by the line is 1.2 waves at a keyed tier's 96 registers (allegro's 64-row tier 5% slower,
        PRO)."""
        NT = self.threads
        return Work(int(items), lanes=NT, block=NT, registers=REGISTER_LINE, cap=False, shared=int(shared))

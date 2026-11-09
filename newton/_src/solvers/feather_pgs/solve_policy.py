# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweeps' exit law, written once for every route, and its opt-in readout.

Every native sweep (Gauss-Seidel and Jacobi alike) stops a world after a sweep in which every row update was
quiet: its change lay within the rounding floor of the update's own arithmetic. For row ``i`` with diagonal
``A_ii``, impulse ``lambda_i`` before the update and change ``dlambda_i``, the law reads the change of the row's own
velocity that the update made, ``phi_i = A_ii |dlambda_i|`` (the row's natural residual in velocity units; a Jacobi
step from extrapolated impulses reads its step from them over its step size), against

    tau_i = 16 u s_i,   s_i = |b_i| + S_i + A_ii |lambda_i|,   u = 2^-24,

where ``b_i`` is the residual's right-hand side and ``S_i = sum_k |t_ik|`` the magnitude of the terms its velocity
sum adds (``Z_ik u_k``, ``J_ik v_k`` or ``W_ij lambda_j``, as the sweep forms it). The residual is computed only to
``gamma_n S_i + u |b_i|``, and the step, projection and difference add a few roundings of the impulse, so a change
within ``tau_i`` is indistinguishable from rounding: a sweep of quiet updates is a fixed point of the floating-point
projected Gauss-Seidel map up to its floor, whatever the schedule. (The impulse after the update bounds the impulse
term within ``phi_i`` of the one before, so the floor reads the one before, at hand where the residual is formed.)
A sweep that visits only the rows it selected at entry also requires every unvisited zero-impulse unilateral row's
would-be change to be quiet.

A vote site computes each row's bound where its residual is reduced (:func:`bound`), in the form its storage gives
without a division, and votes its change against it with one compare (:class:`Change`): ``phi_i`` against ``tau_i``
where the row keeps its diagonal ``A_ii``, ``|dlambda_i|`` against ``tau_i / A_ii`` where it keeps ``1 / A_ii``; a
Jacobi step, whose ``phi_i`` is its change over its step size ``s_i``, votes against ``tau_i s_i``. An operator that
does not form ``S_i`` passes a lower bound of it (by default the reduced velocity term,
``|sum_k t_ik| <= sum_k |t_ik|``), which makes the test stricter: a quiet vote always certifies ``phi_i <= tau_i``. A
:class:`SolvePolicy` names the readout compiled into every sweep and keys their kernel caches.

With ``readout="sweeps"`` each sweep also counts, per world, how many sweeps every solve executed;
``readout="trace"`` adds, per solve, the sweep at which each absolute velocity tolerance of :data:`TRACE_GRID` would
have stopped it, its exit reason, and the largest velocity-scaled change of its final sweep with that row's kind.
The counters are one operand per kernel, appended to the kernel's arguments, and :class:`SolveReadoutRecord` owns
them on the host. The matrix-free sweeps enter no world without rows, so the record also counts the solves of
worlds without dense or matrix-free rows as skipped, in a launch of its own. Neither readout changes a computed
value.
"""

from dataclasses import dataclass, fields

import numpy as np
import warp as wp

from ...core import memory
from ...core.memory import Buffer
from . import field_group
from .device_preludes import DevicePrelude
from .field_group import Field, FieldGroup, SolverArrays
from .row_inputs import RowArgument

SOLVE_READOUTS = ("off", "sweeps", "trace")
"""Accepted readouts: no counter, executed sweeps, or executed sweeps with the exit-law trace."""

TRACE_GRID = tuple(10.0 ** (-7.0 + 0.5 * point) for point in range(13))
"""Velocity tolerances [m/s or rad/s] at which the trace records where each solve would stop: 1e-7 to 1e-1."""

TRACE_REASONS = ("stationary", "tolerance", "budget")
"""Exit reasons of a traced solve: no change, changes within the exit law's floor, or the sweep budget ran out."""

TRACE_KINDS = ("contact", "speed", "friction", "joint_limit", "velocity_limit", "mimic", "link", "torsion", "none")
"""Row kinds of the trace's final-sweep change, indexed by constraint type; ``"none"`` when no row changed."""

TRACE_BINS = 26
"""Magnitude bins of the final-sweep change: 0 for exactly zero, then half decades from 1e-9 to 1e3."""


READOUT_COUNTERS = FieldGroup(
    # Solves by executed sweeps and world: row k counts the solves that executed k sweeps; executed counts beyond the
    # last row add to the last row.
    Field("sweeps", wp.int32, ("budget_row", "world")),
    # Solves per world that executed every sweep of their budget without stopping.
    Field("exhausted", wp.int32, ("world",)),
)
"""The per-world counters of a readout operand (``readout="sweeps"`` and ``"trace"``)."""

TRACE_COUNTERS = FieldGroup(
    # Solves by grid tolerance and stopping sweep: column k counts the solves whose largest velocity-scaled change
    # first fell to the tolerance after sweep k; the last column counts the solves for which it never did within
    # their executed sweeps.
    Field("stops", wp.int64, (len(TRACE_GRID), "stop")),
    # Solves by exit reason (TRACE_REASONS).
    Field("reasons", wp.int64, (len(TRACE_REASONS),)),
    # Final sweeps by the kind of their largest velocity-scaled change and its magnitude bin.
    Field("limits", wp.int64, (len(TRACE_KINDS), TRACE_BINS)),
)
"""The whole-run counters a trace adds: they sum over every world, so they are 64-bit and do not wrap within a
training-length run."""

SolveReadout = field_group.struct("SolveReadout", READOUT_COUNTERS, __name__)
"""Device counters of one sweep kernel under ``readout="sweeps"``."""

SolveTrace = field_group.struct("SolveTrace", (*READOUT_COUNTERS, *TRACE_COUNTERS), __name__)
"""Device counters of one sweep kernel under ``readout="trace"``."""


@wp.kernel
def count_skipped_solves(
    world_constraint_count: wp.array[int],
    mf_constraint_count: wp.array[int],
    skipped: wp.array[wp.int32],
):
    """Count one skipped solve of every world without dense or matrix-free rows."""
    world = wp.tid()
    if world_constraint_count[world] == 0 and mf_constraint_count[world] == 0:
        skipped[world] = skipped[world] + 1


_SWEEP_READOUT = """    const auto solve_readout_sweeps = [&](int readout_world, int executed, bool stopped) {
        const int rows = readout.sweeps.shape[0];
        const int row = executed < rows ? executed : rows - 1;
        ADD(readout.sweeps.data[row * readout.sweeps.shape[1] + readout_world], 1);
        if (!stopped) ADD(readout.exhausted.data[readout_world], 1);
    };"""

_ADD = memory.words("ADD")
"""The readouts' counts: the memory toolkit's ``ADD`` (an atomic add; the CPU's one thread adds)."""

SWEEP_READOUT = DevicePrelude("sweep_readout", *(f"{add}\n{_SWEEP_READOUT}" for add in _ADD))
"""Device helper that records one solve; ``readout`` is the kernel's :class:`SolveReadout` operand."""

_GRID_LITERALS = ", ".join(f"{value:.9e}f" for value in TRACE_GRID)

_SWEEP_TRACE = f"""    float solve_trace_max = 0.0f;
    int solve_trace_kind = -1;
    unsigned solve_trace_open = {(1 << len(TRACE_GRID)) - 1:#x}u;
    const auto solve_trace_change = [&](float change, int kind) {{
        if (change > solve_trace_max) {{
            solve_trace_max = change;
            solve_trace_kind = kind;
        }}
    }};
    const auto solve_trace_sweep = [&](int trace_world, int executed, float change, int kind, bool stopped,
                                       bool final) {{
        const float grid[{len(TRACE_GRID)}] = {{{_GRID_LITERALS}}};
        const int columns = readout.stops.shape[1];
        const int column = executed < columns - 1 ? executed : columns - 2;
        for (int point = 0; point < {len(TRACE_GRID)}; ++point) {{
            if (((solve_trace_open >> point) & 1u) != 0u && change <= grid[point]) {{
                ADD(*reinterpret_cast<unsigned long long*>(&readout.stops.data[point * columns + column]), 1ull);
                solve_trace_open &= ~(1u << point);
            }}
        }}
        if (!final) return;
        const int rows = readout.sweeps.shape[0];
        ADD(readout.sweeps.data[(executed < rows ? executed : rows - 1) * readout.sweeps.shape[1] + trace_world], 1);
        if (!stopped) ADD(readout.exhausted.data[trace_world], 1);
        ADD(*reinterpret_cast<unsigned long long*>(&readout.reasons.data[stopped ? (change > 0.0f ? 1 : 0) : 2]), 1ull);
        const int kinds = readout.limits.shape[0], bins = readout.limits.shape[1];
        const int kind_row = kind >= 0 && kind < kinds - 1 ? kind : kinds - 1;
        int bin = 0;
        if (change > 0.0f) {{
            bin = isfinite(change) ? static_cast<int>(floorf(2.0f * log10f(change))) + 19 : bins - 1;
            bin = bin < 1 ? 1 : (bin > bins - 1 ? bins - 1 : bin);
        }}
        ADD(*reinterpret_cast<unsigned long long*>(&readout.limits.data[kind_row * bins + bin]), 1ull);
        for (int point = 0; point < {len(TRACE_GRID)}; ++point)
            if (((solve_trace_open >> point) & 1u) != 0u)
                ADD(*reinterpret_cast<unsigned long long*>(&readout.stops.data[point * columns + columns - 1]), 1ull);
    }};"""

SWEEP_TRACE = DevicePrelude("sweep_trace", *(f"{add}\n{_SWEEP_TRACE}" for add in _ADD))
"""Device state and helpers of the trace; ``readout`` is the kernel's :class:`SolveTrace` operand."""


QUIET_UNITS = 16
"""Rounding units of the exit law's floor: a change within ``QUIET_UNITS * UNIT_ROUNDOFF`` of its update's scale is
quiet. It covers a lane reduction's depth (5 to 8 roundings) and the applies a velocity carries between two visits
of a row (random-sign roundings of a few dozen applies, about 8)."""

UNIT_ROUNDOFF = 2.0**-24
"""The unit roundoff of float32, the precision every sweep computes in."""

QUIET = QUIET_UNITS * UNIT_ROUNDOFF
"""The exit law's floor relative to an update's scale, ``2^-20``."""

QUIET_LITERAL = f"{QUIET!r}f"
""":data:`QUIET` as a CUDA ``float`` literal (exact: a power of two)."""


def bound(rhs: str, terms: str, impulse: str, *, step: str | None = None, diagonal: str | None = None) -> str:
    """The exit law's bound of a row update, in the form the row's storage gives without a division.

    With ``diagonal`` (a row that keeps ``A_ii``): ``tau_i = 2^-20 (|b_i| + S_i + A_ii |lambda_i|)`` itself, against
    which the vote reads ``phi_i`` (:attr:`Change.scaled`). With ``step`` (a row that keeps ``1 / A_ii``, or a Jacobi
    step's ``s_i``): ``tau_i step = 2^-20 ((|b_i| + S_i) step + A_ii step |lambda_i|)``, against which the vote reads
    the impulse change.

    Args:
        rhs: The right-hand side ``b_i`` of the residual the update reads.
        terms: ``S_i``, the magnitude of the terms the residual's velocity sum adds, or a lower bound of it.
        impulse: The row's impulse before the update; with ``step``, times ``A_ii step`` (the impulse itself for a
            Gauss-Seidel step, its share of one for a Jacobi step). ``"0.0f"`` drops the term.
        step: The row's step size, for a bound in impulse units.
        diagonal: The row's diagonal ``A_ii``, for a bound in velocity units.
    """
    if (step is None) == (diagonal is None):
        raise ValueError("a bound takes either the row's step or its diagonal")
    held = "" if impulse == "0.0f" else impulse
    if diagonal is not None:
        own = f" + ({diagonal}) * fabsf({held})" if held else ""
        return f"{QUIET_LITERAL} * (fabsf({rhs}) + ({terms}){own})"
    own = f" + fabsf({held})" if held else ""
    return f"{QUIET_LITERAL} * ((fabsf({rhs}) + ({terms})) * ({step}){own})"


@dataclass(frozen=True)
class Change:
    """One update's change as the exit law reads it: native expressions in scope at the vote.

    The change is loud beyond its bound (:func:`bound`), which the vote site computed where the residual was reduced;
    a NaN change is loud. The trace records ``phi = |delta| * velocity``, the change of the row's own velocity.
    """

    delta: str
    """The impulse change ``dlambda_i``."""
    bound: str
    """The row's bound: ``tau_i`` (:attr:`scaled`), else ``tau_i / A_ii`` or a Jacobi step's ``tau_i s_i``."""
    kind: str
    """The row's constraint type (the trace names its change by it)."""
    velocity: str
    """Converts the impulse change into the row's velocity change: ``A_ii``, or a Jacobi step's inverse size."""
    scaled: bool = False
    """Whether the bound is in velocity units (a row that keeps its diagonal), so the vote reads ``phi``."""
    exists: str | None = None
    """Where the row may not exist (a friction sibling of a row without a pair), the condition that it does: a
    missing row's change is quiet and reads none of the row's data."""

    @property
    def phi(self) -> str:
        """The change of the row's own velocity, ``|dlambda_i| A_ii``."""
        return f"fabsf({self.delta}) * ({self.velocity})"

    @property
    def loud(self) -> str:
        """Whether the change exceeds its bound."""
        change = self.phi if self.scaled else f"fabsf({self.delta})"
        loud = f"!({change} <= {self.bound})"
        return f"{self.exists} && {loud}" if self.exists else loud


def terms_bound(residual: str, rhs: str) -> str:
    """The default lower bound of ``S_i``: the reduced velocity term ``|r_i - b_i|`` of a residual ``r_i`` read with
    its right-hand side ``b_i`` (``|sum_k t_ik| <= sum_k |t_ik|``)."""
    return f"fabsf(({residual}) - ({rhs}))"


@dataclass(frozen=True, repr=False, kw_only=True)
class SolvePolicy:
    """What a native sweep reports beside the exit law, which every sweep compiles.

    A frozen, hashable value: sweep factories take it as a keyword and cache
    their kernels on it. Its representation names only the fields that differ
    from their defaults, so kernel labels stay stable as fields are added.
    """

    readout: str = "off"
    """``"off"``, ``"sweeps"`` to count the sweeps every solve executes, or ``"trace"`` to add the exit trace."""

    def __post_init__(self):
        if self.readout not in SOLVE_READOUTS:
            raise ValueError(f"readout must be one of {SOLVE_READOUTS}, got {self.readout!r}")

    def __repr__(self) -> str:
        changed = (
            f"{field.name}={getattr(self, field.name)!r}"
            for field in fields(self)
            if getattr(self, field.name) != field.default
        )
        return f"SolvePolicy({', '.join(changed)})"

    @property
    def counts(self) -> bool:
        """Whether the sweep carries a readout operand."""
        return self.readout != "off"

    @property
    def traces(self) -> bool:
        """Whether the sweep records the exit trace."""
        return self.readout == "trace"

    @property
    def suffix(self) -> str:
        """Kernel-name suffix of this policy's variant; empty without readout."""
        return f"_{self.readout}" if self.counts else ""

    @property
    def preludes(self) -> tuple[DevicePrelude, ...]:
        """Device preludes the sweep body declares under this policy, after its own."""
        return {"off": (), "sweeps": (SWEEP_READOUT,), "trace": (SWEEP_TRACE,)}[self.readout]

    @property
    def operands(self) -> dict:
        """Native operands appended after the sweep's own, as ``{name: type}``."""
        return {"off": {}, "sweeps": {"readout": SolveReadout}, "trace": {"readout": SolveTrace}}[self.readout]

    @property
    def arguments(self) -> tuple[RowArgument, ...]:
        """The same operands as a generated kernel's extra operands (:func:`.kernel_template.operands`)."""
        return tuple(RowArgument(name, dtype) for name, dtype in self.operands.items())


DEFAULT_POLICY = SolvePolicy()
"""The policy without readout: every sweep factory's default."""


def vote(policy: SolvePolicy, changes: tuple[Change, ...], *, flag: str, pad: str) -> str:
    """Return the exit law's vote on an update's changes: ``flag`` is set when one of them is loud.

    The sweep keeps applying every nonzero change to the velocity; a quiet change only does not keep the world
    sweeping. The trace records each nonzero change's ``phi``. A zero change is quiet, so the vote may stand outside
    a block entered on a change.

    Args:
        policy: The sweep's solve policy (its readout).
        changes: The update's changes (one per row it changed).
        flag: The sweep's change flag.
        pad: Indentation of the emitted statements.
    """
    statements = []
    if policy.traces:
        statements += [
            f"if ({change.delta} != 0.0f) solve_trace_change({change.phi}, {change.kind});" for change in changes
        ]
    loud = " || ".join(f"({change.loud})" for change in changes)
    statements.append(f"if ({loud}) {flag} = 1;")
    return "\n".join(pad + statement for statement in statements)


def sweep_exit(
    policy: SolvePolicy,
    *,
    quiet: str,
    violation: tuple[str, str, str, str] | None = None,
    world: str = "world",
    iteration: str = "iteration",
    iterations: str = "iterations",
    lane: str = "lane",
    argmax: str | None = "ARGMAX",
    indent: str = "        ",
) -> str:
    """Return the end-of-sweep exit of a native sweep: stop the world once no update of the sweep was loud.

    With a readout, lane 0 records the solve when the world stops and after the last sweep of its budget, so no
    counter stays live through the loop.

    Args:
        policy: The sweep's solve policy.
        quiet: Group-uniform condition that no update of this sweep voted (no loud change).
        violation: For sweeps that visit only rows selected at sweep entry, ``(code, loud, value, kind)``:
            group-uniform statements that set the bool ``loud`` when a visit of some zero-impulse unilateral row
            would make a loud change (:attr:`Change.loud`), the float ``value`` to the largest such change (its
            violation times the relaxation factor) and the int ``kind`` to that row's constraint type.
        world: World index expression.
        iteration: Zero-based sweep index within this launch.
        iterations: Sweep budget of this launch.
        lane: Lane index within the world's lanes; the trace reduces over the sweep's lane group.
        argmax: The reduction that takes the trace's largest change and its row kind over the world's lanes
            (``ARGMAX(value, kind)``; a block of several warps names its own), or ``None`` for a world one thread
            sweeps.
        indent: Indentation of the emitted statements.
    """
    stop = quiet
    inner = indent + "    "

    def block(code):
        return "\n".join(inner + line if line.strip() else line for line in code.strip("\n").split("\n"))

    lines = [f"{indent}{{"]
    if not policy.traces:
        if violation is None:
            lines.append(f"{inner}const bool sweep_stop = {stop};")
        else:
            code, loud, _, _ = violation
            lines += [f"{inner}bool sweep_stop = {stop};", f"{inner}if (sweep_stop) {{", block(code)]
            lines += [f"{inner}    sweep_stop = !{loud};", f"{inner}}}"]
        if policy.counts:
            lines += [
                f"{inner}if ({lane} == 0 && (sweep_stop || {iteration} + 1 == {iterations}))",
                f"{inner}    solve_readout_sweeps({world}, {iteration} + 1, sweep_stop);",
            ]
        lines.append(f"{inner}if (sweep_stop) break;")
        return "\n".join([*lines, f"{indent}}}"]) + "\n"
    lines += [
        f"{inner}float sweep_change = solve_trace_max;",
        f"{inner}int sweep_kind = solve_trace_kind;",
        *([f"{inner}{argmax}(sweep_change, sweep_kind);"] if argmax else []),
        f"{inner}solve_trace_max = 0.0f;",
        f"{inner}solve_trace_kind = -1;",
    ]
    stop_line = f"{inner}const bool sweep_stop = {stop};"
    if violation is not None:
        code, loud, value, kind = violation
        lines += [block(code), f"{inner}if ({value} > sweep_change) {{"]
        lines += [f"{inner}    sweep_change = {value};", f"{inner}    sweep_kind = {kind};", f"{inner}}}"]
        stop_line = f"{inner}const bool sweep_stop = {stop} && !{loud};"
    lines += [
        stop_line,
        f"{inner}if ({lane} == 0)",
        f"{inner}    solve_trace_sweep({world}, {iteration} + 1, sweep_change, sweep_kind, sweep_stop,",
        f"{inner}        sweep_stop || {iteration} + 1 == {iterations});",
        f"{inner}if (sweep_stop) break;",
        f"{indent}}}",
    ]
    return "\n".join(lines) + "\n"


def block_argmax(warps: int, *, warp: str, lane: str) -> tuple[str, tuple[Buffer, ...]]:
    """The trace's reduction over a block of ``warps`` warps, ``BLOCK_ARGMAX(value, kind)`` (two barriers): each
    warp's largest change and its kind (``ARGMAX``), then every warp's over all of theirs, so every thread holds the
    block's. ``warp`` and ``lane`` are the thread's warp and its lane in that warp.

    Returns its declaration, after the block's shared arrays, and those arrays.
    """
    text = f"""
    // The trace's largest change over the block: each warp's, then every warp's over all of theirs.
    const auto BLOCK_ARGMAX = [&](float& value, int& kind) {{
        ARGMAX(value, kind);
        if ({lane} == 0) {{
            s_trace_max[{warp}] = value;
            s_trace_kind[{warp}] = kind;
        }}
        __syncthreads();
        value = {lane} < {warps} ? s_trace_max[{lane}] : 0.0f;
        kind = {lane} < {warps} ? s_trace_kind[{lane}] : -1;
        ARGMAX(value, kind);
        __syncthreads();
    }};"""
    return text, (Buffer("s_trace_max", "float", warps), Buffer("s_trace_kind", "int", warps))


@dataclass
class _ReadoutSlot:
    label: str
    operand: object
    arrays: dict


class SolveReadoutRecord:
    """Host owner of the readout counters of a solver's sweep kernels.

    Each readout kernel gets one slot of counters, allocated at registration
    (outside graph capture) and passed as the kernel's last argument. The
    counters accumulate over every launch, captured graph replays included,
    so a window is the difference of two summaries.

    A sweep that returns before its first sweep for a world records nothing
    for it. The matrix-free sweeps do so for every world without dense or
    matrix-free rows, such as one the resolved-world classifier left without
    rows: the route that launches them registers skip counters
    (:meth:`register_skipped`) and counts, at each launch
    (:meth:`count_skipped`), the worlds without rows of any family, so a
    world's solves are its slots' solves plus its skipped ones. Every sweep
    carries counters, the parallel (Jacobi) tiers against their own sweep
    budget, so every world-solve that a sweep enters lands in a slot.
    """

    def __init__(self, policy: SolvePolicy):
        self.policy = policy
        self._slots: dict[str, _ReadoutSlot] = {}
        self._skipped: wp.array | None = None

    def register(
        self,
        kernel,
        label: str,
        *,
        readout: str | None,
        world_count: int,
        budget: int,
        device,
    ) -> None:
        """Allocate the counters of a readout kernel; kernels built without a readout are ignored.

        Args:
            kernel: A sweep kernel.
            label: Route of the kernel in the record, such as ``"sparse_pgs"``.
            readout: The readout of the policy the kernel was built under (``"sweeps"`` or ``"trace"``), or
                ``None`` for a kernel without a readout operand.
            world_count: Number of worlds the kernel can solve.
            budget: Largest sweep budget a launch of the kernel passes.
            device: Device of the counters.
        """
        if kernel is None or not readout or kernel.key in self._slots:
            return
        rows = max(int(budget), 1) + 1
        trace = readout == "trace"
        operand = SolveTrace() if trace else SolveReadout()
        counters = SolverArrays(operand, device)
        arrays = counters.allocate(READOUT_COUNTERS, budget_row=rows, world=max(int(world_count), 1))
        if trace:
            arrays.update(counters.allocate(TRACE_COUNTERS, stop=rows + 1))
        self._slots[kernel.key] = _ReadoutSlot(label, operand, arrays)

    def register_skipped(self, *, world_count: int, device) -> None:
        """Allocate per-world counters of the solves that enter no sweep because the world has no rows.

        Ignored without a readout, or when already registered.

        Args:
            world_count: Number of worlds the route's launches cover.
            device: Device of the counters.
        """
        if not self.policy.counts or self._skipped is not None or int(world_count) <= 0:
            return
        self._skipped = wp.zeros(int(world_count), dtype=wp.int32, device=device)

    def count_skipped(self, world_constraint_count: wp.array, mf_constraint_count: wp.array) -> None:
        """Count one skipped solve of every world without dense or matrix-free rows.

        A launch of its own, made only under a registered readout; capturable.

        Args:
            world_constraint_count: Dense rows per world.
            mf_constraint_count: Matrix-free rows per world.
        """
        skipped = self._skipped
        if skipped is None:
            return
        wp.launch(
            count_skipped_solves,
            dim=skipped.shape[0],
            inputs=[world_constraint_count, mf_constraint_count],
            outputs=[skipped],
            device=skipped.device,
        )

    def operands(self, kernel) -> list:
        """Return the readout operand to append to a launch of ``kernel``, or nothing."""
        key = getattr(kernel, "key", None) if kernel is not None else None
        slot = self._slots.get(key)
        return [] if slot is None else [slot.operand]

    @property
    def kernels(self) -> dict[str, str]:
        """``{kernel key: route label}`` of the registered readout kernels."""
        return {key: slot.label for key, slot in self._slots.items()}

    @property
    def counts_skipped(self) -> bool:
        """Whether the solves of worlds without rows count as skipped (:meth:`register_skipped`)."""
        return self._skipped is not None

    def summary(self) -> dict:
        """Read the counters and summarize every slot; this synchronizes with the device.

        A slot is one kernel. ``sweeps[k]`` counts
        the solves that executed ``k`` sweeps (the last entry also holds longer
        solves), ``exhausted`` the solves that ran their whole budget without
        stopping. A traced slot adds ``trace``. Counts are cumulative since
        construction.

        A solve whose first sweep changes nothing stops after it: a world
        without rows, which the sparse and sparse-diagonal sweeps enter like
        any other, or one whose rows need no update. No tolerance shortens
        those; ``past_first_sweep`` summarizes the other solves. The
        full-budget fraction among the solves of worlds with rows lies between
        ``full_budget_fraction`` and ``past_first_sweep["full_budget_fraction"]``.

        ``skipped`` holds, for a route that registered skip counters, the
        ``solves`` of worlds without dense or matrix-free rows, which entered
        no sweep, and the number of ``worlds`` they came from (else ``None``):
        the matrix-free sweeps return before their first sweep for such a
        world, so no slot counts it. Each solve of a world that a counted sweep
        owns is counted once, by a slot or here.
        """
        slots = []
        for key, slot in self._slots.items():
            sweeps = slot.arrays["sweeps"].numpy().astype(np.int64)
            exhausted = int(slot.arrays["exhausted"].numpy().astype(np.int64).sum())
            histogram = sweeps.sum(axis=1)
            solves = int(histogram.sum())
            budget = len(histogram) - 1
            executed = np.arange(len(histogram), dtype=np.float64)
            per_world = sweeps.sum(axis=0)
            active = per_world > 0
            world_mean = (executed[:, None] * sweeps).sum(axis=0)[active] / per_world[active]
            record = {
                "kernel": key,
                "route": slot.label,
                "budget": budget,
                "solves": solves,
                "sweeps": [int(count) for count in histogram],
                "exhausted": exhausted,
                "full_budget_fraction": float(histogram[-1] / solves) if solves else None,
                "mean_sweeps": float((executed * histogram).sum() / solves) if solves else None,
                "past_first_sweep": _past_first_sweep(histogram),
                "worlds": int(active.sum()),
                "world_mean_sweeps": {
                    name: float(np.percentile(world_mean, q)) if world_mean.size else None
                    for name, q in (("p10", 10), ("p50", 50), ("p90", 90))
                },
            }
            if "stops" in slot.arrays:
                record["trace"] = _trace_summary(slot.arrays)
            slots.append(record)
        skipped = None
        if self._skipped is not None:
            counts = self._skipped.numpy().astype(np.int64)
            skipped = {"solves": int(counts.sum()), "worlds": int(np.count_nonzero(counts))}
        return {"readout": self.policy.readout, "slots": slots, "skipped": skipped}


def _past_first_sweep(histogram) -> dict:
    """Solves, full-budget fraction and mean sweeps of the solves that executed more than one sweep."""
    longer = histogram[2:]
    solves = int(longer.sum())
    executed = np.arange(2, len(histogram), dtype=np.float64)
    return {
        "solves": solves,
        "full_budget_fraction": float(histogram[-1] / solves) if solves else None,
        "mean_sweeps": float((executed * longer).sum() / solves) if solves else None,
    }


def _trace_summary(arrays) -> dict:
    """Summarize a traced slot: stopping sweeps per grid tolerance, exit reasons and final-sweep changes."""
    stops = arrays["stops"].numpy().astype(np.int64)
    reasons = arrays["reasons"].numpy().astype(np.int64)
    limits = arrays["limits"].numpy().astype(np.int64)
    # JSON-compliant like the route records: the open last bin's upper edge is the string "inf".
    upper = [0.0, *(10.0 ** (-8.5 + 0.5 * (bin - 1)) for bin in range(1, TRACE_BINS - 1)), "inf"]

    def percentiles(counts):
        total = int(counts.sum())
        if not total:
            return None
        cumulative = np.cumsum(counts)
        return {
            name: upper[int(np.searchsorted(cumulative, q * total))]
            for name, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))
        }

    return {
        "grid": list(TRACE_GRID),
        # Row per grid tolerance; column k: solves that would stop after k sweeps; last column: within no sweep.
        "stops": [[int(count) for count in row] for row in stops],
        "reasons": {name: int(count) for name, count in zip(TRACE_REASONS, reasons, strict=True)},
        # Upper edge of each magnitude bin [m/s or rad/s]: bin 0 holds exact zeros.
        "bin_upper": upper,
        "final_change": {
            kind: {"count": int(row.sum()), "bins": [int(count) for count in row], "at_most": percentiles(row)}
            for kind, row in zip(TRACE_KINDS, limits, strict=True)
            if row.sum()
        },
        "final_change_all": percentiles(limits.sum(axis=0)),
    }

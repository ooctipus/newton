# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile-time stopping rule of the Gauss-Seidel sweeps and their opt-in readout.

Every native Gauss-Seidel sweep stops a world after a sweep in which no
impulse changed (the exact law). A :class:`SolvePolicy` is compiled into the
sweeps that admit one and keys their kernel caches: equal policies share
kernels, and the default policy emits exactly the kernels built before
policies existed.

``velocity_change_tolerance`` replaces the exact law by a velocity-scaled
one: a row is quiet when its impulse change times its effective-mass
diagonal, ``|dlambda| * A_ii`` (the change of the row's own velocity), is at
most the tolerance. Sweeps that visit only rows selected at sweep entry also
require every zero-impulse unilateral row to be violated by at most the
tolerance, the change its visit would make. At zero tolerance the emitters
return each sweep's own exact text.

With ``readout="sweeps"`` each sweep also counts, per world, how many sweeps
every solve executed; ``readout="trace"`` adds, per solve, the sweep at which
each tolerance of :data:`TRACE_GRID` would have stopped it, its exit reason,
and the largest velocity-scaled change of its final sweep with that row's
kind. The counters are one operand per kernel, appended to the kernel's
arguments, and :class:`SolveReadoutRecord` owns them on the host. Neither
readout changes a computed value.
"""

import math
from dataclasses import dataclass, fields

import numpy as np
import warp as wp

from .device_preludes import DevicePrelude
from .kernels import PGS_CONSTRAINT_TYPE_COUNT
from .row_inputs import RowArgument

SOLVE_READOUTS = ("off", "sweeps", "trace")
"""Accepted readouts: no counter, executed sweeps, or executed sweeps with the exit-law trace."""

TRACE_GRID = tuple(10.0 ** (-7.0 + 0.5 * point) for point in range(13))
"""Velocity tolerances [m/s or rad/s] at which the trace records where each solve would stop: 1e-7 to 1e-1."""

TRACE_REASONS = ("stationary", "tolerance", "budget")
"""Exit reasons of a traced solve: no change, changes within the tolerance, or the sweep budget ran out."""

TRACE_KINDS = ("contact", "drive", "friction", "joint_limit", "velocity_limit", "mimic", "connect", "torsion", "none")
"""Row kinds of the trace's final-sweep change, by constraint type; ``"none"`` when no row changed."""

TRACE_BINS = 26
"""Magnitude bins of the final-sweep change: 0 for exactly zero, then half decades from 1e-9 to 1e3."""

assert len(TRACE_KINDS) == PGS_CONSTRAINT_TYPE_COUNT + 1


class SolvePolicyUnsupported(ValueError):
    """A sweep variant cannot compile the requested stopping rule or readout."""


@wp.struct
class SolveReadout:
    """Device counters of one sweep kernel under ``readout="sweeps"``."""

    sweeps: wp.array2d[wp.int32]
    """Solves by executed sweeps and world, shape ``[rows, world_count]``.

    Row ``k`` counts the solves that executed ``k`` sweeps; executed counts
    beyond the last row add to the last row.
    """
    exhausted: wp.array[wp.int32]
    """Solves per world that executed every sweep of their budget without stopping."""


@wp.struct
class SolveTrace:
    """Device counters of one sweep kernel under ``readout="trace"``."""

    sweeps: wp.array2d[wp.int32]
    """Solves by executed sweeps and world, as :attr:`SolveReadout.sweeps`."""
    exhausted: wp.array[wp.int32]
    """Solves per world that ran out of budget, as :attr:`SolveReadout.exhausted`."""
    stops: wp.array2d[wp.int32]
    """Solves by grid tolerance and stopping sweep, shape ``[len(TRACE_GRID), rows + 1]``.

    Column ``k`` counts the solves whose largest velocity-scaled change first
    fell to the tolerance after sweep ``k``; the last column counts the solves
    for which it never did within their executed sweeps.
    """
    reasons: wp.array[wp.int32]
    """Solves by exit reason (:data:`TRACE_REASONS`)."""
    limits: wp.array2d[wp.int32]
    """Final sweeps by the kind of their largest velocity-scaled change and its magnitude bin, ``[kinds, bins]``."""


SWEEP_READOUT = DevicePrelude(
    "sweep_readout",
    """    const auto solve_readout_sweeps = [&](int readout_world, int executed, bool stopped) {
        const int rows = readout.sweeps.shape[0];
        const int row = executed < rows ? executed : rows - 1;
        atomicAdd(&readout.sweeps.data[row * readout.sweeps.shape[1] + readout_world], 1);
        if (!stopped) atomicAdd(&readout.exhausted.data[readout_world], 1);
    };""",
)
"""Device helper that records one solve; ``readout`` is the kernel's :class:`SolveReadout` operand."""

_GRID_LITERALS = ", ".join(f"{value:.9e}f" for value in TRACE_GRID)

SWEEP_TRACE = DevicePrelude(
    "sweep_trace",
    f"""    float solve_trace_max = 0.0f;
    int solve_trace_kind = -1;
    unsigned solve_trace_open = {(1 << len(TRACE_GRID)) - 1:#x}u;
    const auto solve_trace_change = [&](float change, int kind) {{
        if (change > solve_trace_max) {{
            solve_trace_max = change;
            solve_trace_kind = kind;
        }}
    }};
    const auto solve_trace_sweep = [&](int trace_world, int executed, float change, int kind, bool eligible,
                                       bool stopped, bool final) {{
        const float grid[{len(TRACE_GRID)}] = {{{_GRID_LITERALS}}};
        const int columns = readout.stops.shape[1];
        if (eligible) {{
            const int column = executed < columns - 1 ? executed : columns - 2;
            for (int point = 0; point < {len(TRACE_GRID)}; ++point) {{
                if (((solve_trace_open >> point) & 1u) != 0u && change <= grid[point]) {{
                    atomicAdd(&readout.stops.data[point * columns + column], 1);
                    solve_trace_open &= ~(1u << point);
                }}
            }}
        }}
        if (!final) return;
        const int rows = readout.sweeps.shape[0];
        atomicAdd(&readout.sweeps.data[(executed < rows ? executed : rows - 1) * readout.sweeps.shape[1] + trace_world], 1);
        if (!stopped) atomicAdd(&readout.exhausted.data[trace_world], 1);
        atomicAdd(&readout.reasons.data[stopped ? (change > 0.0f ? 1 : 0) : 2], 1);
        const int kinds = readout.limits.shape[0], bins = readout.limits.shape[1];
        const int kind_row = kind >= 0 && kind < kinds - 1 ? kind : kinds - 1;
        int bin = 0;
        if (change > 0.0f) {{
            bin = isfinite(change) ? static_cast<int>(floorf(2.0f * log10f(change))) + 19 : bins - 1;
            bin = bin < 1 ? 1 : (bin > bins - 1 ? bins - 1 : bin);
        }}
        atomicAdd(&readout.limits.data[kind_row * bins + bin], 1);
        for (int point = 0; point < {len(TRACE_GRID)}; ++point)
            if (((solve_trace_open >> point) & 1u) != 0u) atomicAdd(&readout.stops.data[point * columns + columns - 1], 1);
    }};""",
)
"""Device state and helpers of the trace; ``readout`` is the kernel's :class:`SolveTrace` operand."""


@dataclass(frozen=True, repr=False, kw_only=True)
class SolvePolicy:
    """How a native Gauss-Seidel sweep stops, and what it reports.

    A frozen, hashable value: sweep factories take it as a keyword and cache
    their kernels on it. Its representation names only the fields that differ
    from their defaults, so kernel labels stay stable as fields are added.
    """

    readout: str = "off"
    """``"off"``, ``"sweeps"`` to count the sweeps every solve executes, or ``"trace"`` to add the exit-law trace."""
    velocity_change_tolerance: float = 0.0
    """Largest velocity-scaled change of a quiet row [m/s or rad/s]; zero keeps the exact stationary-sweep law."""

    def __post_init__(self):
        if self.readout not in SOLVE_READOUTS:
            raise ValueError(f"readout must be one of {SOLVE_READOUTS}, got {self.readout!r}")
        tolerance = float(self.velocity_change_tolerance)
        if not (math.isfinite(tolerance) and tolerance >= 0.0):
            raise ValueError(f"velocity_change_tolerance must be finite and non-negative, got {tolerance!r}")
        object.__setattr__(self, "velocity_change_tolerance", tolerance)

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
        """Whether the sweep records the exit-law trace."""
        return self.readout == "trace"

    @property
    def exact(self) -> bool:
        """Whether the sweep keeps the exact stationary-sweep law."""
        return self.velocity_change_tolerance == 0.0

    @property
    def tolerance_literal(self) -> str:
        """The velocity tolerance as a CUDA ``float`` literal."""
        return f"{self.velocity_change_tolerance!r}f"

    @property
    def suffix(self) -> str:
        """Kernel-name suffix of this policy's variant; empty for the exact policy without readout."""
        tolerance = ""
        if not self.exact:
            tolerance = "_vtol" + repr(self.velocity_change_tolerance).replace(".", "p").replace("-", "m")
        return tolerance + (f"_{self.readout}" if self.counts else "")

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
        """The same operands as generated-kernel arguments (:func:`row_inputs.make_row_kernel`)."""
        return tuple(RowArgument(name, dtype) for name, dtype in self.operands.items())


EXACT_POLICY = SolvePolicy()
"""The exact law without readout: every sweep factory's default."""


def vote(
    policy: SolvePolicy,
    exact: str,
    *,
    flag: str,
    terms: tuple[tuple[str, str, str], ...],
    slack: tuple[str, ...] | None = None,
) -> str:
    """Return a row update's change vote under the policy's exit law.

    Under the exact law without trace the sweep's own vote is returned
    unchanged. A positive tolerance votes when some term's velocity-scaled
    change exceeds it; the sweep keeps applying every nonzero change to the
    velocity. The trace records each term's velocity-scaled change.

    Args:
        policy: The sweep's solve policy.
        exact: The update's vote under the exact law, with its leading indentation.
        flag: The sweep's change flag.
        terms: ``(change, scale, kind)`` per row the update changed: its impulse
            change, its effective-mass diagonal ``A_ii`` and its constraint type.
        slack: Per-term impulse slack a quiet row may also change by (the native
            sparse sweeps' run-time impulse tolerances), or ``None``.
    """
    if policy.exact and not policy.traces:
        return exact
    indent = exact[: len(exact) - len(exact.lstrip())]
    statements = []
    if policy.traces:
        statements += [
            f"if ({change} != 0.0f) solve_trace_change(fabsf({change}) * ({scale}), {kind});"
            for change, scale, kind in terms
        ]
    if policy.exact:
        statements.append(exact.strip())
    else:
        bounds = []
        for index, (change, scale, _) in enumerate(terms):
            bound = policy.tolerance_literal
            if slack is not None:
                bound += f" + ({scale}) * ({slack[index]})"
            bounds.append(f"({change} != 0.0f && !(fabsf({change}) * ({scale}) <= {bound}))")
        statements.append(f"if ({' || '.join(bounds)}) {flag} = 1;")
    text = "\n".join(indent + statement for statement in statements)
    return text + "\n" if exact.endswith("\n") else text


def sweep_exit(
    policy: SolvePolicy,
    exact: str,
    *,
    quiet: str,
    gate: str | None = None,
    violation: tuple[str, str, str] | None = None,
    world: str = "world",
    iteration: str = "iteration",
    iterations: str = "iterations",
    lane: str = "lane",
    lanes: int = 32,
    indent: str = "        ",
) -> str:
    """Return the end-of-sweep exit of a native Gauss-Seidel sweep.

    Under the exact law without readout the sweep's own exit text is returned
    unchanged. With a readout, lane 0 records the solve when the world stops
    and after the last sweep of its budget, so no counter stays live through
    the loop.

    Args:
        policy: The sweep's solve policy.
        exact: The sweep's exit under the exact law without readout.
        quiet: Warp-uniform condition that no row of this sweep voted a change.
        gate: Condition under which a quiet sweep may stop the world (delayed friction), or ``None``.
        violation: For sweeps that visit only rows selected at sweep entry,
            ``(code, value, kind)``: warp-uniform statements that set the float
            ``value`` to the largest violation of a zero-impulse unilateral row
            and the int ``kind`` to its constraint type. A positive tolerance
            stops only when ``value`` is within it; the exact law implies it.
        world: World index expression.
        iteration: Zero-based sweep index within this launch.
        iterations: Sweep budget of this launch.
        lane: Lane index within the world's lanes.
        lanes: Lanes per world, the width of the trace's reductions.
        indent: Indentation of the emitted statements.
    """
    if policy.exact and not policy.counts:
        return exact
    stop = quiet if gate is None else f"{gate} && {quiet}"
    inner = indent + "    "
    tolerance = policy.tolerance_literal

    def block(code):
        return "\n".join(inner + line if line.strip() else line for line in code.strip("\n").split("\n"))

    lines = [f"{indent}{{"]
    if not policy.traces:
        if policy.exact or violation is None:
            lines.append(f"{inner}const bool sweep_stop = {stop};")
        else:
            code, value, _ = violation
            lines += [f"{inner}bool sweep_stop = {stop};", f"{inner}if (sweep_stop) {{", block(code)]
            lines += [f"{inner}    sweep_stop = {value} <= {tolerance};", f"{inner}}}"]
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
        f"{inner}for (int trace_offset = {lanes // 2}; trace_offset > 0; trace_offset >>= 1) {{",
        f"{inner}    const float other_change = __shfl_xor_sync(MASK, sweep_change, trace_offset, {lanes});",
        f"{inner}    const int other_kind = __shfl_xor_sync(MASK, sweep_kind, trace_offset, {lanes});",
        f"{inner}    if (other_change > sweep_change || (other_change == sweep_change && other_kind > sweep_kind)) {{",
        f"{inner}        sweep_change = other_change;",
        f"{inner}        sweep_kind = other_kind;",
        f"{inner}    }}",
        f"{inner}}}",
        f"{inner}solve_trace_max = 0.0f;",
        f"{inner}solve_trace_kind = -1;",
    ]
    stop_line = f"{inner}const bool sweep_stop = {stop};"
    if violation is not None:
        code, value, kind = violation
        lines += [block(code), f"{inner}if ({value} > sweep_change) {{"]
        lines += [f"{inner}    sweep_change = {value};", f"{inner}    sweep_kind = {kind};", f"{inner}}}"]
        if not policy.exact:
            stop_line = f"{inner}const bool sweep_stop = {stop} && {value} <= {tolerance};"
    lines += [
        stop_line,
        f"{inner}if ({lane} == 0)",
        f"{inner}    solve_trace_sweep({world}, {iteration} + 1, sweep_change, sweep_kind, {gate or 'true'}, sweep_stop,",
        f"{inner}        sweep_stop || {iteration} + 1 == {iterations});",
        f"{inner}if (sweep_stop) break;",
        f"{indent}}}",
    ]
    return "\n".join(lines) + "\n"


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
    """

    def __init__(self, policy: SolvePolicy):
        self.policy = policy
        self._slots: dict[str, _ReadoutSlot] = {}

    def register(self, kernel, label: str, *, world_count: int, budget: int, device) -> None:
        """Allocate the counters of a readout kernel; kernels without a readout operand are ignored.

        Args:
            kernel: A sweep kernel built under a policy with a readout.
            label: Route of the kernel in the record, such as ``"sparse_pgs"``.
            world_count: Number of worlds the kernel can solve.
            budget: Largest sweep budget a launch of the kernel passes.
            device: Device of the counters.
        """
        readout = getattr(kernel, "_fpgs_solve_readout", False) if kernel is not None else False
        if not readout or kernel.key in self._slots:
            return
        rows, worlds = max(int(budget), 1) + 1, max(int(world_count), 1)
        arrays = {
            "sweeps": wp.zeros((rows, worlds), dtype=wp.int32, device=device),
            "exhausted": wp.zeros(worlds, dtype=wp.int32, device=device),
        }
        operand = SolveReadout()
        if readout == "trace":
            arrays["stops"] = wp.zeros((len(TRACE_GRID), rows + 1), dtype=wp.int32, device=device)
            arrays["reasons"] = wp.zeros(len(TRACE_REASONS), dtype=wp.int32, device=device)
            arrays["limits"] = wp.zeros((len(TRACE_KINDS), TRACE_BINS), dtype=wp.int32, device=device)
            operand = SolveTrace()
        for name, array in arrays.items():
            setattr(operand, name, array)
        self._slots[kernel.key] = _ReadoutSlot(label, operand, arrays)

    def operands(self, kernel) -> list:
        """Return the readout operand to append to a launch of ``kernel``, or nothing."""
        slot = self._slots.get(getattr(kernel, "key", None)) if kernel is not None else None
        return [] if slot is None else [slot.operand]

    @property
    def kernels(self) -> dict[str, str]:
        """``{kernel key: route label}`` of the registered readout kernels."""
        return {key: slot.label for key, slot in self._slots.items()}

    def summary(self) -> dict:
        """Read the counters and summarize every slot; this synchronizes with the device.

        ``sweeps[k]`` counts the solves that executed ``k`` sweeps (the last
        entry also holds longer solves), ``exhausted`` the solves that ran their
        whole budget without stopping. A traced slot adds ``trace``. Counts are
        cumulative since construction.
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
                "worlds": int(active.sum()),
                "world_mean_sweeps": {
                    name: float(np.percentile(world_mean, q)) if world_mean.size else None
                    for name, q in (("p10", 10), ("p50", 50), ("p90", 90))
                },
            }
            if "stops" in slot.arrays:
                record["trace"] = _trace_summary(slot.arrays)
            slots.append(record)
        return {"readout": self.policy.readout, "slots": slots}


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

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
require the change a visit would make to every zero-impulse unilateral row,
its violation times the relaxation factor, to be at most the tolerance. At
zero tolerance the emitters return each sweep's own exact text.

Without the native sparse sweeps' run-time impulse tolerances
(``pgs_change_tolerance`` and ``pgs_relative_change_tolerance``) a solve
under the velocity law runs the exact law's sweeps and stops at or before the
sweep where the exact law stops, so its outputs are the exact law's at that
sweep, and the trace of an exact solve predicts that sweep for every
tolerance of its grid. The impulse tolerances let the exact law stop with
nonzero changes: the trace records changes without that slack, so it no
longer predicts the stop, and the unvisited-row bound, which has no slack,
can add sweeps.

With ``readout="sweeps"`` each sweep also counts, per world, how many sweeps
every solve executed; ``readout="trace"`` adds, per solve, the sweep at which
each tolerance of :data:`TRACE_GRID` would have stopped it, its exit reason,
and the largest velocity-scaled change of its final sweep with that row's
kind. The counters are one operand per kernel, appended to the kernel's
arguments, and :class:`SolveReadoutRecord` owns them on the host. The
matrix-free sweeps enter no world without rows, so the record also counts
the solves of worlds without dense, matrix-free or propagation rows as
skipped, in a launch of its own. Neither readout changes a computed value.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, fields

import numpy as np
import warp as wp

from .device_preludes import DevicePrelude
from .row_inputs import RowArgument

SOLVE_READOUTS = ("off", "sweeps", "trace")
"""Accepted readouts: no counter, executed sweeps, or executed sweeps with the exit-law trace."""

SOLVE_PASSES = ("position", "velocity")
"""Solve passes a readout counts apart: the position solve and the velocity pass (``pgs_velocity_iterations``)."""

TRACE_GRID = tuple(10.0 ** (-7.0 + 0.5 * point) for point in range(13))
"""Velocity tolerances [m/s or rad/s] at which the trace records where each solve would stop: 1e-7 to 1e-1."""

TRACE_REASONS = ("stationary", "tolerance", "budget")
"""Exit reasons of a traced solve: no change, changes within the tolerance, or the sweep budget ran out."""

TRACE_KINDS = ("contact", "drive", "friction", "joint_limit", "velocity_limit", "mimic", "connect", "torsion", "none")
"""Row kinds of the trace's final-sweep change, indexed by constraint type; ``"none"`` when no row changed."""

TRACE_BINS = 26
"""Magnitude bins of the final-sweep change: 0 for exactly zero, then half decades from 1e-9 to 1e3."""


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
    """Device counters of one sweep kernel under ``readout="trace"``.

    The per-world counts are 32-bit like :class:`SolveReadout`'s; the
    stopping, reason and final-change counts sum over every world, so they
    are 64-bit and do not wrap within a training-length run.
    """

    sweeps: wp.array2d[wp.int32]
    """Solves by executed sweeps and world, as :attr:`SolveReadout.sweeps`."""
    exhausted: wp.array[wp.int32]
    """Solves per world that ran out of budget, as :attr:`SolveReadout.exhausted`."""
    stops: wp.array2d[wp.int64]
    """Solves by grid tolerance and stopping sweep, shape ``[len(TRACE_GRID), rows + 1]``.

    Column ``k`` counts the solves whose largest velocity-scaled change first
    fell to the tolerance after sweep ``k``; the last column counts the solves
    for which it never did within their executed sweeps.
    """
    reasons: wp.array[wp.int64]
    """Solves by exit reason (:data:`TRACE_REASONS`)."""
    limits: wp.array2d[wp.int64]
    """Final sweeps by the kind of their largest velocity-scaled change and its magnitude bin, ``[kinds, bins]``."""


@wp.kernel
def count_skipped_solves(
    world_constraint_count: wp.array[int],
    mf_constraint_count: wp.array[int],
    propagation_constraint_count: wp.array[int],
    skipped: wp.array[wp.int32],
):
    """Count one skipped solve of every world without dense, matrix-free or propagation rows."""
    world = wp.tid()
    if (
        world_constraint_count[world] == 0
        and mf_constraint_count[world] == 0
        and propagation_constraint_count[world] == 0
    ):
        skipped[world] = skipped[world] + 1


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
                    atomicAdd(reinterpret_cast<unsigned long long*>(&readout.stops.data[point * columns + column]), 1ull);
                    solve_trace_open &= ~(1u << point);
                }}
            }}
        }}
        if (!final) return;
        const int rows = readout.sweeps.shape[0];
        atomicAdd(&readout.sweeps.data[(executed < rows ? executed : rows - 1) * readout.sweeps.shape[1] + trace_world], 1);
        if (!stopped) atomicAdd(&readout.exhausted.data[trace_world], 1);
        atomicAdd(reinterpret_cast<unsigned long long*>(&readout.reasons.data[stopped ? (change > 0.0f ? 1 : 0) : 2]), 1ull);
        const int kinds = readout.limits.shape[0], bins = readout.limits.shape[1];
        const int kind_row = kind >= 0 && kind < kinds - 1 ? kind : kinds - 1;
        int bin = 0;
        if (change > 0.0f) {{
            bin = isfinite(change) ? static_cast<int>(floorf(2.0f * log10f(change))) + 19 : bins - 1;
            bin = bin < 1 ? 1 : (bin > bins - 1 ? bins - 1 : bin);
        }}
        atomicAdd(reinterpret_cast<unsigned long long*>(&readout.limits.data[kind_row * bins + bin]), 1ull);
        for (int point = 0; point < {len(TRACE_GRID)}; ++point)
            if (((solve_trace_open >> point) & 1u) != 0u)
                atomicAdd(reinterpret_cast<unsigned long long*>(&readout.stops.data[point * columns + columns - 1]), 1ull);
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
    velocity. The trace records each term's velocity-scaled change, without
    the term's slack.

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
            ``value`` to the largest velocity change a visit of a zero-impulse
            unilateral row would make (its violation times the relaxation
            factor) and the int ``kind`` to that row's constraint type. A
            positive tolerance stops only when ``value`` is within it; without
            run-time impulse tolerances the exact law implies it.
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
    solve_pass: str
    operand: object
    arrays: dict


class SolveReadoutRecord:
    """Host owner of the readout counters of a solver's sweep kernels.

    Each readout kernel gets one slot of counters per solve pass it runs in
    (:data:`SOLVE_PASSES`), allocated at registration (outside graph capture)
    and passed as the kernel's last argument: the position solve's slot, and
    a velocity-pass slot when the solver runs that pass, so that its short
    budgets do not fill the position solve's histogram. Launches inside
    :meth:`velocity_pass` take the velocity-pass slot. The counters accumulate
    over every launch, captured graph replays included, so a window is the
    difference of two summaries.

    A sweep that returns before its first sweep for a world records nothing
    for it. The matrix-free sweeps do so for every world without dense or
    matrix-free rows, such as one the resolved-world classifier left without
    rows: the route that launches them registers skip counters
    (:meth:`register_skipped`) and counts, at each launch
    (:meth:`count_skipped`), the worlds without rows of any family, so a
    world's solves are its slots' solves plus its skipped ones. Sweeps
    without counters can own whole worlds that hold rows: the parallel
    (Jacobi) tiers and the response-block sweeps the worlds they claim from
    the matrix-free sweep, the multi-block driver's dense tiles every world's
    dense rows, and the propagation sweeps a world whose rows are all
    propagation rows. Neither a slot nor the skip counter counts those
    solves; the route record names their owners
    (``routes.sweep.readout.uncounted_worlds``).
    """

    def __init__(self, policy: SolvePolicy):
        self.policy = policy
        self._slots: dict[tuple[str, str], _ReadoutSlot] = {}
        self._labels: dict[str, str] = {}
        self._skipped: dict[str, wp.array] = {}
        # Row counts of a family a route does not build: no world holds such rows.
        self._no_rows: wp.array | None = None
        self._pass = "position"

    def register(self, kernel, label: str, *, world_count: int, budget: int, device, velocity_budget: int = 0) -> None:
        """Allocate the counters of a readout kernel; kernels without a readout operand are ignored.

        Args:
            kernel: A sweep kernel built under a policy with a readout.
            label: Route of the kernel in the record, such as ``"sparse_pgs"``.
            world_count: Number of worlds the kernel can solve.
            budget: Largest sweep budget a position-solve launch of the kernel passes.
            device: Device of the counters.
            velocity_budget: Sweep budget of the velocity pass, or 0 when the solver runs none.
        """
        readout = getattr(kernel, "_fpgs_solve_readout", False) if kernel is not None else False
        if not readout or kernel.key in self._labels:
            return
        self._labels[kernel.key] = label
        budgets = {"position": budget, "velocity": velocity_budget}
        worlds = max(int(world_count), 1)
        for solve_pass in SOLVE_PASSES:
            if solve_pass != "position" and int(budgets[solve_pass]) <= 0:
                continue
            rows = max(int(budgets[solve_pass]), 1) + 1
            arrays = {
                "sweeps": wp.zeros((rows, worlds), dtype=wp.int32, device=device),
                "exhausted": wp.zeros(worlds, dtype=wp.int32, device=device),
            }
            operand = SolveReadout()
            if readout == "trace":
                arrays["stops"] = wp.zeros((len(TRACE_GRID), rows + 1), dtype=wp.int64, device=device)
                arrays["reasons"] = wp.zeros(len(TRACE_REASONS), dtype=wp.int64, device=device)
                arrays["limits"] = wp.zeros((len(TRACE_KINDS), TRACE_BINS), dtype=wp.int64, device=device)
                operand = SolveTrace()
            for name, array in arrays.items():
                setattr(operand, name, array)
            self._slots[(kernel.key, solve_pass)] = _ReadoutSlot(label, solve_pass, operand, arrays)

    def register_skipped(self, *, world_count: int, device, velocity_budget: int = 0) -> None:
        """Allocate per-world counters of the solves that enter no sweep because the world has no rows.

        Ignored without a readout, or when already registered.

        Args:
            world_count: Number of worlds the route's launches cover.
            device: Device of the counters.
            velocity_budget: Sweep budget of the velocity pass, or 0 when the solver runs none.
        """
        if not self.policy.counts or self._skipped or int(world_count) <= 0:
            return
        self._no_rows = wp.zeros(int(world_count), dtype=wp.int32, device=device)
        budgets = {"position": 1, "velocity": int(velocity_budget)}
        for solve_pass in SOLVE_PASSES:
            if budgets[solve_pass] > 0:
                self._skipped[solve_pass] = wp.zeros(int(world_count), dtype=wp.int32, device=device)

    def count_skipped(
        self,
        world_constraint_count: wp.array,
        mf_constraint_count: wp.array,
        *,
        propagation_constraint_count: wp.array | None = None,
    ) -> None:
        """Count, in the current pass, one skipped solve of every world without dense, matrix-free or propagation rows.

        A launch of its own, made only under a registered readout; capturable.

        Args:
            world_constraint_count: Dense rows per world.
            mf_constraint_count: Matrix-free rows per world.
            propagation_constraint_count: Propagation rows per world, which the propagation sweeps solve
                on the propagation contact routes, or ``None`` on a route without them.
        """
        skipped = self._skipped.get(self._pass)
        if skipped is None:
            return
        if propagation_constraint_count is None:
            propagation_constraint_count = self._no_rows
        wp.launch(
            count_skipped_solves,
            dim=skipped.shape[0],
            inputs=[world_constraint_count, mf_constraint_count, propagation_constraint_count],
            outputs=[skipped],
            device=skipped.device,
        )

    @contextmanager
    def velocity_pass(self):
        """Count the launches made inside the context in the velocity-pass slots."""
        previous, self._pass = self._pass, "velocity"
        try:
            yield
        finally:
            self._pass = previous

    def operands(self, kernel) -> list:
        """Return the readout operand to append to a launch of ``kernel`` in the current pass, or nothing."""
        key = getattr(kernel, "key", None) if kernel is not None else None
        slot = self._slots.get((key, self._pass))
        return [] if slot is None else [slot.operand]

    @property
    def kernels(self) -> dict[str, str]:
        """``{kernel key: route label}`` of the registered readout kernels."""
        return dict(self._labels)

    @property
    def skipped_passes(self) -> tuple[str, ...]:
        """Solve passes that count the solves of worlds without rows as skipped (:meth:`register_skipped`)."""
        return tuple(self._skipped)

    def summary(self) -> dict:
        """Read the counters and summarize every slot; this synchronizes with the device.

        A slot is one kernel in one solve pass (``pass``). ``sweeps[k]`` counts
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

        ``skipped`` holds, per solve pass of a route that registered skip
        counters, the ``solves`` of worlds without dense, matrix-free or
        propagation rows, which entered no sweep, and the number of ``worlds``
        they came from: the matrix-free sweeps return before their first sweep
        for such a world, so no slot counts it. Under the direct driver
        (``routes.sweep.driver``) each solve of a world that a counted sweep
        owns is counted once, by a slot or here, and under the
        contact-then-internal driver once per launch. Drivers that launch one
        row phase at a time count per launch, and a launch whose phase holds
        none of a world's rows can return before its first sweep, which counts
        that world nowhere. The worlds that sweeps without counters own, which
        hold rows, are counted by neither
        (``routes.sweep.readout.uncounted_worlds``).
        """
        slots = []
        for (key, _), slot in self._slots.items():
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
                "pass": slot.solve_pass,
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
        skipped = []
        for solve_pass, array in self._skipped.items():
            counts = array.numpy().astype(np.int64)
            skipped.append({"pass": solve_pass, "solves": int(counts.sum()), "worlds": int(np.count_nonzero(counts))})
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

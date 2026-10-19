# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile-time stopping rule of the Gauss-Seidel sweeps and their opt-in readout.

Every native Gauss-Seidel sweep stops a world after a sweep in which no
impulse changed (the exact law). A :class:`SolvePolicy` is compiled into the
sweeps that admit one and keys their kernel caches: equal policies share
kernels, and the default policy emits exactly the kernels built before
policies existed.

With ``readout="sweeps"`` each sweep also counts, per world, how many sweeps
every solve executed. The counters are one :class:`SolveReadout` operand per
kernel, appended to the kernel's arguments, and :class:`SolveReadoutRecord`
owns them on the host. Counting does not change any computed value.
"""

from dataclasses import dataclass, fields

import numpy as np
import warp as wp

from .device_preludes import DevicePrelude
from .row_inputs import RowArgument

SOLVE_READOUTS = ("off", "sweeps")
"""Accepted readouts: ``"off"`` emits no counter, ``"sweeps"`` counts executed sweeps."""


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


@dataclass(frozen=True, repr=False)
class SolvePolicy:
    """How a native Gauss-Seidel sweep stops, and what it reports.

    A frozen, hashable value: sweep factories take it as a keyword and cache
    their kernels on it. Its representation names only the fields that differ
    from their defaults, so kernel labels stay stable as fields are added.
    """

    readout: str = "off"
    """``"off"``, or ``"sweeps"`` to count the sweeps every solve executes, per kernel and world."""

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
        """Whether the sweep carries a :class:`SolveReadout` operand."""
        return self.readout != "off"

    @property
    def suffix(self) -> str:
        """Kernel-name suffix of this policy's variant; empty for the exact policy without readout."""
        return f"_{self.readout}" if self.counts else ""

    @property
    def preludes(self) -> tuple[DevicePrelude, ...]:
        """Device preludes the sweep body declares under this policy, after its own."""
        return (SWEEP_READOUT,) if self.counts else ()

    @property
    def operands(self) -> dict:
        """Native operands appended after the sweep's own, as ``{name: type}``."""
        return {"readout": SolveReadout} if self.counts else {}

    @property
    def arguments(self) -> tuple[RowArgument, ...]:
        """The same operands as generated-kernel arguments (:func:`row_inputs.make_row_kernel`)."""
        return tuple(RowArgument(name, dtype) for name, dtype in self.operands.items())


EXACT_POLICY = SolvePolicy()
"""The exact law without readout: every sweep factory's default."""


def sweep_exit(
    policy: SolvePolicy,
    exact: str,
    *,
    quiet: str,
    gate: str | None = None,
    world: str = "world",
    iteration: str = "iteration",
    iterations: str = "iterations",
    lane: str = "lane",
    indent: str = "        ",
) -> str:
    """Return the end-of-sweep exit of a native Gauss-Seidel sweep.

    Without a readout the sweep's own exit text is returned unchanged. With
    one, lane 0 records the solve when the world stops and after the last
    sweep of its budget, so no counter stays live through the loop.

    Args:
        policy: The sweep's solve policy.
        exact: The sweep's exit under the exact law without readout.
        quiet: Warp-uniform condition that no row of this sweep voted a change.
        gate: Condition under which a quiet sweep may stop the world (delayed friction), or ``None``.
        world: World index expression.
        iteration: Zero-based sweep index within this launch.
        iterations: Sweep budget of this launch.
        lane: Lane index within the world's lanes.
        indent: Indentation of the emitted statements.
    """
    if not policy.counts:
        return exact
    stop = quiet if gate is None else f"{gate} && {quiet}"
    inner = indent + "    "
    return (
        f"{indent}{{\n"
        f"{inner}const bool sweep_stop = {stop};\n"
        f"{inner}if ({lane} == 0 && (sweep_stop || {iteration} + 1 == {iterations}))\n"
        f"{inner}    solve_readout_sweeps({world}, {iteration} + 1, sweep_stop);\n"
        f"{inner}if (sweep_stop) break;\n"
        f"{indent}}}\n"
    )


@dataclass
class _ReadoutSlot:
    label: str
    operand: SolveReadout
    sweeps: wp.array
    exhausted: wp.array


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
            kernel: A sweep kernel built under this record's policy.
            label: Route of the kernel in the record, such as ``"sparse_pgs"``.
            world_count: Number of worlds the kernel can solve.
            budget: Largest sweep budget a launch of the kernel passes.
            device: Device of the counters.
        """
        if kernel is None or not getattr(kernel, "_fpgs_solve_readout", False) or kernel.key in self._slots:
            return
        sweeps = wp.zeros((max(int(budget), 1) + 1, max(int(world_count), 1)), dtype=wp.int32, device=device)
        exhausted = wp.zeros(max(int(world_count), 1), dtype=wp.int32, device=device)
        operand = SolveReadout()
        operand.sweeps = sweeps
        operand.exhausted = exhausted
        self._slots[kernel.key] = _ReadoutSlot(label, operand, sweeps, exhausted)

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
        whole budget without stopping. Counts are cumulative since construction.
        """
        slots = []
        for key, slot in self._slots.items():
            sweeps = slot.sweeps.numpy().astype(np.int64)
            exhausted = int(slot.exhausted.numpy().astype(np.int64).sum())
            histogram = sweeps.sum(axis=1)
            solves = int(histogram.sum())
            budget = len(histogram) - 1
            executed = np.arange(len(histogram), dtype=np.float64)
            per_world = sweeps.sum(axis=0)
            active = per_world > 0
            world_mean = (executed[:, None] * sweeps).sum(axis=0)[active] / per_world[active]
            slots.append(
                {
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
            )
        return {"readout": self.policy.readout, "slots": slots}

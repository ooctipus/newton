# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check and resolve FPGS options before topology admission."""

import functools
import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass, make_dataclass
from typing import Any, Literal

from .solve_policy import SOLVE_READOUTS

MAX_CONTACT_REGULARIZATION = 1.0e6
"""The largest ``pgs_contact_regularization``: larger values are not numerically useful in the float32 solve."""


def _finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


# Each numeric option's type and admissible values: (name, type, admits, requirement, append the given value).
_NUMBER_RULES = (
    ("contact_speculative_scale", float, _finite_nonnegative, "must be finite and non-negative", False),
    ("contact_gap_gate", float, _finite_nonnegative, "must be finite and non-negative", False),
    ("same_articulation_contact_gap_gate", float, _finite_nonnegative, "must be finite and non-negative", False),
    ("articulation_pair_contact_gap_gate", float, _finite_nonnegative, "must be finite and non-negative", False),
    ("joint_limit_activation_gap", float, lambda value: value >= 0.0, "must be non-negative or inf", False),
    # A finite fraction above 1 would leave speeds in (limit, fraction * limit) unclamped: a row allocates only at
    # |qd| >= fraction * limit. inf is the explicit never-activate switch.
    (
        "velocity_limit_activation_fraction",
        float,
        lambda v: 0.0 <= v <= 1.0 or math.isinf(v),
        "must be in [0, 1] or inf",
        False,
    ),
    ("pgs_contact_regularization", float, _finite_nonnegative, "must be finite and non-negative", False),
    ("restitution_velocity_threshold", float, _finite_nonnegative, "must be finite and non-negative", False),
    ("pgs_warmstart_decay", float, _finite_nonnegative, "must be finite and non-negative", True),
    ("dense_max_constraints", int, lambda value: value >= 1, "must be positive", False),
    # The free bodies' rows per world. Every class but native sparse factors and the sparse diagonal runs the
    # matrix-free sweeps, whatever this capacity: none selects a route.
    ("mf_max_constraints", int, lambda value: value >= 1, "must be positive", False),
)


# The switches the solver applies as booleans.
_BOOLEANS = (
    "enable_bilateral_preelimination",
    "enable_restitution",
    "pgs_warmstart",
    "contact_torsion",
    "contact_compliance",
)


@dataclass(frozen=True)
class Diagnostics:
    """What :class:`~newton.solvers.SolverFeatherPGS` reads out beside the step, as its ``diagnostics`` option.

    No field changes a computed value or a choice: a step with any diagnostics on publishes the same state, and builds
    the same rows, as one with all of them off.

    Attributes:
        nvtx: Mark the step's stages as NVTX ranges.
        snapshots: Copy the stage-3 joint velocity, joint acceleration and predicted velocity of every substep into
            ``_debug_stage3_qd_work``, ``_debug_stage3_joint_qdd`` and ``_debug_stage3_v_hat``, which external NaN
            dumpers read; three copies a substep.
        row_watermark: Accumulate each row family's whole-run high-water mark, read back by
            :meth:`~newton.solvers.SolverFeatherPGS.constraint_row_watermarks`.
        overflow_warning: Emit a device-side warning the first time each dense or matrix-free row family exceeds its
            per-world capacity: the world, requested rows, row limit and dropped contact and friction rows, without a
            host synchronization, so it stays compatible with CUDA graph capture.
        solve_readout: Counters of the Gauss-Seidel sweeps. ``"sweeps"`` counts, per sweep kernel and world, how
            many sweeps each solve executed and whether it stopped before its budget; ``"trace"`` adds, per solve,
            the sweep at which each absolute velocity tolerance from 1e-7 to 1e-1 [m/s or rad/s] would have stopped
            it, its exit reason (no change, changes within the exit law's rounding floor, or the budget) and the
            largest velocity-scaled change of its final sweep with that row's kind.
            ``get_path_selection()["measurements"]["solve_readout"]`` summarizes them and synchronizes once. The
            matrix-free sweeps enter no world without rows; the readout counts those solves as ``skipped``.
        audit: Audit each step after its solve: per world, the energy gap ``G`` [J] the solve failed to account for,
            the dual violation ``D`` [m/s] and the rows' injected work ``W`` [J], read by
            :meth:`~newton.solvers.SolverFeatherPGS.read_audit`. A world whose rows exist only inside its sweep
            reads NaN (``routes.diagnostics.audit`` says which).
        stage_clocks: Compile per-stage cycle counters into the held tree's kernels (``_stage_clocks``). It changes the
            emitted code, not what it computes.
        sync_timings: ``(first_step, steps)``: synchronize around each stage of those steps and print its time;
            the step plan then runs each stage as a timed stage. The sweep's streams then run without overlap.
            ``None`` times nothing.
    """

    nvtx: bool = False
    snapshots: bool = False
    row_watermark: bool = False
    overflow_warning: bool = True
    solve_readout: Literal["off", "sweeps", "trace"] = "off"
    audit: bool = False
    stage_clocks: bool = False
    sync_timings: tuple[int, int] | None = None

    def __post_init__(self):
        if self.solve_readout not in SOLVE_READOUTS:
            raise ValueError(f"solve_readout must be one of {SOLVE_READOUTS}, got {self.solve_readout!r}")
        if self.sync_timings is not None and (
            len(self.sync_timings) != 2 or min(int(value) for value in self.sync_timings) < 0
        ):
            raise ValueError(f"sync_timings must be None or (first_step, steps) >= 0, got {self.sync_timings!r}")

    def timed(self, step: int) -> bool:
        """Whether ``step`` is one :attr:`sync_timings` times."""
        if self.sync_timings is None:
            return False
        first, steps = self.sync_timings
        return first <= step < first + steps


@functools.cache
def declared_options(init) -> dict:
    """Every option of the constructor ``init`` with its declared default: the parameters after ``self`` and ``model``,
    but the test pins, which are the spec's (``routes.spec.pins``), not an option."""
    parameters = list(inspect.signature(init).parameters.values())[2:]
    return {parameter.name: parameter.default for parameter in parameters if parameter.name != "pins"}


@functools.cache
def _record(names: tuple[str, ...]) -> type:
    return make_dataclass("Options", names, frozen=True)


def check_options(init, options: Mapping[str, Any]):
    """Return every option of the constructor ``init`` (:func:`declared_options`) as the solver applies it, after
    checking each: a frozen record, numbers coerced by the rule tables and the switches of :data:`_BOOLEANS` booleans,
    ``diagnostics`` a :class:`Diagnostics` (``None`` gives the default), every other option as given
    (``friction_anchor_beta`` the request, ``None`` for the default: the row laws resolve it).

    Args:
        init: The constructor.
        options: The constructor's options by keyword (more keys are ignored).

    Raises:
        ValueError: naming the first option, in table order, that is not admissible, or a regularization above
            :data:`MAX_CONTACT_REGULARIZATION`.
        TypeError: for ``diagnostics`` that is neither ``None`` nor a :class:`Diagnostics`.
    """
    checked = {}
    for name, coerce, admits, requirement, given in _NUMBER_RULES:
        value = options[name]
        message = f"{name} {requirement}" + (f", got {value!r}" if given else "")
        try:
            checked[name] = coerce(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if not admits(checked[name]):
            raise ValueError(message)
    if checked["pgs_contact_regularization"] > MAX_CONTACT_REGULARIZATION:
        raise ValueError(
            f"pgs_contact_regularization must be at most {MAX_CONTACT_REGULARIZATION:g}; "
            "larger values are not numerically useful in the float32 solve"
        )
    checked.update({name: bool(options[name]) for name in _BOOLEANS})
    diagnostics = options["diagnostics"]
    if diagnostics is None:
        diagnostics = Diagnostics()
    elif not isinstance(diagnostics, Diagnostics):
        raise TypeError(f"diagnostics must be None or a SolverFeatherPGS.Diagnostics, got {type(diagnostics).__name__}")
    checked["diagnostics"] = diagnostics
    names = tuple(declared_options(init))
    return _record(names)(**{name: checked.get(name, options[name]) for name in names})

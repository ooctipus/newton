# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check and resolve FPGS options before topology admission."""

import math
from collections.abc import Mapping
from typing import Any

from .solve_policy import SOLVE_READOUTS


def _finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


# Each numeric option's type and admissible values: (name, type, admits, requirement, append the given value).
_NUMBER_RULES = (
    (
        "contact_friction_position_iterations",
        int,
        lambda value: value >= -1,
        "must be -1 for all position iterations, or a non-negative last-N iteration count",
        False,
    ),
    ("contact_friction_scale", float, lambda value: not value < 0.0, "must be non-negative", False),
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
)
# Each option with a fixed set of values.
_CHOICE_RULES = (
    ("pgs_solve_readout", SOLVE_READOUTS),
    ("pgs_velocity_drive_mode", ("active", "freeze")),
    ("pgs_mode", ("split", "matrix_free")),
    ("pgs_schedule", ("interleaved", "contact_then_internal", "physx_grasp")),
    ("drive_mode", ("augmented", "physx_pgs")),
)


def check_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return the numeric options coerced to their types, after checking every option of the rule tables.

    Raises:
        ValueError: naming the first option, numbers first and in table order, that is not admissible.
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
    for name, choices in _CHOICE_RULES:
        if options[name] not in choices:
            raise ValueError(f"{name} must be one of {choices}, got {options[name]!r}")
    return checked


# The kernels the CPU runs, and the choices a test pin may fix on CUDA.
_CPU_KERNELS = {
    "cholesky_kernel": "loop",
    "trisolve_kernel": "loop",
    "hinv_jt_kernel": "par_row",
    "delassus_kernel": "par_row_col",
    "pgs_kernel": "loop",
}
_PINNABLE = {
    "cholesky_kernel": ("tiled", "loop"),
    "trisolve_kernel": ("tiled", "loop"),
    "hinv_jt_kernel": ("tiled", "par_row"),
    "delassus_kernel": ("tiled", "par_row_col"),
    "pgs_kernel": ("loop", "tiled_row"),
}
# Split mode's assembled Delassus is swept by one thread per world up to this many rows (at most 256 floats), by the
# tiled-row sweep beyond.
LOOP_SWEEP_MAX_ROWS = 16


def kernel_choices(pins: Mapping[str, Any], *, is_cpu: bool, split: bool, dense_rows: int):
    """Return each dense kernel's choice and why.

    The CPU runs the loop kernels. On CUDA the factor, triangular-solve, response and Delassus kernels are
    ``"auto"``: the execution plan maps each size by its resource rule. Split mode's dense sweep is the loop for at
    most :data:`LOOP_SWEEP_MAX_ROWS` rows per world and the tiled-row sweep beyond; matrix-free mode builds neither.
    A test pin (``SolverFeatherPGS._route_pins``) fixes a kernel on CUDA where the execution plan maps a size to a
    dense kernel; it claims no worlds from the other owners (the ``placement`` pin does).

    Raises:
        ValueError: for a pin that names no kernel of its stage, or a CUDA kernel on the CPU.
    """
    choices, reasons = {}, {}
    for name, allowed in _PINNABLE.items():
        pinned = pins.get(name)
        if pinned == "auto":
            pinned = None
        if pinned is not None and pinned not in allowed:
            raise ValueError(f"The {name} pin must be one of {allowed}, got {pinned!r}")
        if is_cpu:
            if pinned is not None and pinned != _CPU_KERNELS[name]:
                raise ValueError(f"The {name} pin {pinned!r} requires CUDA; the CPU runs {_CPU_KERNELS[name]!r}")
            choices[name], reasons[name] = _CPU_KERNELS[name], f"The CPU runs {_CPU_KERNELS[name]}."
        elif pinned is not None:
            choices[name], reasons[name] = pinned, f"A test pin selects {name}={pinned!r}."
        elif name == "pgs_kernel":
            loop = split and dense_rows <= LOOP_SWEEP_MAX_ROWS
            choices[name] = "loop" if loop else "tiled_row"
            reasons[name] = (
                f"Split mode sweeps at most {LOOP_SWEEP_MAX_ROWS} dense rows per world on one thread."
                if loop
                else f"The tiled-row sweep serves more than {LOOP_SWEEP_MAX_ROWS} dense rows per world."
            )
        else:
            choices[name], reasons[name] = "auto", "Automatic selection by size and device resources."
    return choices, reasons

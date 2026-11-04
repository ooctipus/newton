# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check and resolve FPGS options before topology admission."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
# Each option with a fixed set of values.
_CHOICE_RULES = (("pgs_solve_readout", SOLVE_READOUTS),)


# The switches the solver applies as booleans.
_BOOLEANS = (
    "enable_bilateral_preelimination",
    "enable_restitution",
    "warn_constraint_overflow",
    "pgs_warmstart",
    "row_watermark",
    "contact_torsion",
    "contact_compliance",
    "pgs_audit",
)


@dataclass(frozen=True)
class Options:
    """Every option of :class:`.SolverFeatherPGS`, as the solver applies it (:func:`check_options`).

    Numbers are coerced and checked by the rule tables, choices checked and the switches of :data:`_BOOLEANS` made
    booleans; every other option is as given.
    ``friction_anchor_beta`` is the request (``None`` for the default): the row laws resolve it.
    """

    angular_damping: Any
    update_mass_matrix_interval: Any
    enable_contact_friction: Any
    contact_friction_gap_threshold: Any
    enable_joint_limits: Any
    enable_bilateral_preelimination: bool
    joint_limit_activation_gap: float
    enable_joint_velocity_limits: Any
    velocity_limit_activation_fraction: float
    pgs_iterations: Any
    pgs_beta: Any
    pgs_cfm: Any
    pgs_omega: Any
    pgs_contact_regularization: float
    dense_max_constraints: int
    pgs_warmstart: bool
    mf_max_constraints: int
    nvtx: Any
    pgs_debug: Any
    row_watermark: bool
    restitution_velocity_threshold: float
    contact_speculative_scale: float
    contact_gap_gate: float
    enable_restitution: bool
    same_articulation_contact_gap_gate: float
    articulation_pair_contact_gap_gate: float
    pgs_warmstart_decay: float
    warn_constraint_overflow: bool
    friction_anchor_beta: Any
    pgs_solve_readout: str
    pgs_audit: bool
    contact_torsion: bool
    contact_compliance: bool


def check_options(options: Mapping[str, Any]) -> Options:
    """Return every option as the solver applies it (:class:`Options`), after checking each.

    Args:
        options: The constructor's options by keyword (more keys are ignored).

    Raises:
        ValueError: naming the first option, numbers first and in table order, then the choices, that is not
            admissible, or a regularization above :data:`MAX_CONTACT_REGULARIZATION`.
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
    if checked["pgs_contact_regularization"] > MAX_CONTACT_REGULARIZATION:
        raise ValueError(
            f"pgs_contact_regularization must be at most {MAX_CONTACT_REGULARIZATION:g}; "
            "larger values are not numerically useful in the float32 solve"
        )
    checked.update({name: bool(options[name]) for name in _BOOLEANS})
    return Options(**{name: checked.get(name, options[name]) for name in Options.__dataclass_fields__})

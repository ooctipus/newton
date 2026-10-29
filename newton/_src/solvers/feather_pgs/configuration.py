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
    ("contact_friction_anchor_limit", int, lambda value: value >= 0, "must be non-negative", False),
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
    ("tile_threads", int, lambda value: value in (32, 64, 128, 256), "must be one of (32, 64, 128, 256)", True),
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


_KERNEL_CHOICES = {
    "cholesky_kernel": ("auto", "tiled", "loop"),
    "trisolve_kernel": ("auto", "tiled", "loop"),
    "hinv_jt_kernel": ("auto", "tiled", "par_row"),
    "delassus_kernel": ("auto", "tiled", "par_row_col"),
    "pgs_kernel": ("auto", "loop", "tiled_row"),
    "sparse_mass_matrix": ("auto", "on", "off"),
    "cached_dynamics": ("auto", "on", "off"),
}
_CPU_KERNELS = {
    "cholesky_kernel": "loop",
    "trisolve_kernel": "loop",
    "hinv_jt_kernel": "par_row",
    "delassus_kernel": "par_row_col",
    "pgs_kernel": "loop",
    "sparse_mass_matrix": "off",
    "cached_dynamics": "off",
}
_DENSE_KERNELS = ("cholesky_kernel", "trisolve_kernel", "hinv_jt_kernel", "delassus_kernel")


def resolve_kernel_choices(
    requested: dict[str, str],
    overrides: dict[str, str | bool],
    *,
    is_cpu: bool,
    pgs_mode: str,
) -> tuple[dict[str, str], bool, dict[str, str], dict[str, str]]:
    """Return effective choices, explicit dense ownership, construction-time reasons and ignored requests.

    Public choices take precedence over private test hooks. A private hook is
    validated before CPU fallback, but does not acquire explicit dense ownership.
    Topology and resource admission remain the responsibility of execution planning.
    Matrix-free mode owns its sweep kernels, so a valid explicit ``pgs_kernel``
    there resolves like ``auto`` on any device; the last mapping holds the warning
    the caller emits for it.
    """
    pgs_kernel = requested["pgs_kernel"]
    matrix_free = pgs_mode == "matrix_free"
    ignored = {}
    for name, value in requested.items():
        if value not in _KERNEL_CHOICES[name]:
            raise ValueError(f"{name} must be one of {_KERNEL_CHOICES[name]}, got {value!r}")
        if matrix_free and name == "pgs_kernel" and value != "auto":
            ignored[name] = (
                f"pgs_kernel={value!r} is ignored in matrix_free mode, which owns its sweep kernels; "
                "omit it or use pgs_mode='split'."
            )
        elif is_cpu and value not in ("auto", _CPU_KERNELS[name]):
            raise ValueError(f"{name}={value!r} requires CUDA; use 'auto' or {_CPU_KERNELS[name]!r}")
    if matrix_free and requested["delassus_kernel"] != "auto":
        raise ValueError("delassus_kernel is unused in matrix_free mode; use 'auto' or pgs_mode='split'")
    explicit_dense = any(requested[name] != "auto" for name in _DENSE_KERNELS)
    for name in ("sparse_mass_matrix", "cached_dynamics"):
        if requested[name] == "on" and explicit_dense:
            raise ValueError(f"{name}='on' conflicts with explicit dense kernel choices; use 'auto'")

    resolved = dict(requested)
    reasons = {
        name: f"Explicit {name}={value!r}." if value != "auto" else "Automatic selection."
        for name, value in requested.items()
    }
    for name in _DENSE_KERNELS:
        if requested[name] == "auto" and name in overrides:
            value = overrides[name]
            if value not in _KERNEL_CHOICES[name]:
                raise ValueError(f"{name} must be one of {sorted(_KERNEL_CHOICES[name])}")
            resolved[name] = value
            reasons[name] = f"Private override selects {name}={value!r}."
    if pgs_kernel == "auto" or "pgs_kernel" in ignored:
        resolved["pgs_kernel"] = "loop" if is_cpu else "tiled_row"
        reasons["pgs_kernel"] = (
            f"Explicit pgs_kernel={pgs_kernel!r} is ignored in matrix_free mode."
            if "pgs_kernel" in ignored
            else f"Automatic {'CPU' if is_cpu else 'CUDA'} default: {resolved['pgs_kernel']}."
        )
    if is_cpu:
        for name in (*_DENSE_KERNELS, "pgs_kernel"):
            resolved[name] = _CPU_KERNELS[name]
            reasons[name] += f" CPU uses {_CPU_KERNELS[name]}."
    return resolved, explicit_dense, reasons, ignored

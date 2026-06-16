# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton model/state parity helpers used by capture-time verification and tests."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from newton import ModelBuilder, eval_fk

_SKIP_MODEL_ATTRS = {
    "actuators",
    "contacts",
    "control",
    "device",
    "particle_grid",
    "shape_source",
    "shape_source_ptr",
}


def finalize_model_state(
    builder: ModelBuilder,
    sim_cfg: Mapping,
    device: str = "cpu",
    num_envs: int | None = None,
):
    """Finalize *builder*, apply gravity, create state, and run FK."""
    model = builder.finalize(device=device)
    model.set_gravity(tuple(float(v) for v in sim_cfg.get("gravity", (0.0, 0.0, -9.81))))
    model.num_envs = num_envs
    state = model.state()
    eval_fk(model, state.joint_q, state.joint_qd, state, None)
    return model, state


def _normalize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "numpy") and hasattr(value, "shape"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and all(
        item is None or isinstance(item, (bool, int, float, str)) for item in value
    ):
        return np.asarray(value, dtype=object if any(isinstance(item, str) for item in value) else None)
    return None


def _public_data_attrs(obj, skip: set[str]) -> dict[str, object]:
    data = {}
    for name in dir(obj):
        if name.startswith("_") or name in skip:
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        normalized = _normalize(value)
        if normalized is not None:
            data[name] = normalized
    return data


def _compare(prefix: str, expected: dict, actual: dict, mismatches: list[str]) -> None:
    exp_keys = set(expected)
    act_keys = set(actual)
    for missing in sorted(exp_keys - act_keys):
        mismatches.append(f"{prefix}.{missing}: missing from actual")
    for extra in sorted(act_keys - exp_keys):
        mismatches.append(f"{prefix}.{extra}: extra in actual")
    for key in sorted(exp_keys & act_keys):
        name = f"{prefix}.{key}"
        exp_val = expected[key]
        act_val = actual[key]
        if isinstance(exp_val, np.ndarray) or isinstance(act_val, np.ndarray):
            exp = np.asarray(exp_val)
            act = np.asarray(act_val)
            if exp.shape != act.shape:
                mismatches.append(f"{name}: shape {exp.shape} != {act.shape}")
            elif exp.dtype.kind == "f" or act.dtype.kind == "f":
                if not np.array_equal(exp, act):
                    diff = np.max(np.abs(exp.astype(np.float64) - act.astype(np.float64))) if exp.size else 0.0
                    mismatches.append(f"{name}: float arrays differ exactly (max_abs={diff:.9e})")
            elif not np.array_equal(exp, act):
                mismatches.append(f"{name}: arrays differ")
        elif exp_val != act_val:
            mismatches.append(f"{name}: {exp_val!r} != {act_val!r}")


def assert_model_state_equal(expected_model, expected_state, actual_model, actual_state) -> None:
    """Raise ``AssertionError`` if normalized Newton model/state data differ."""
    mismatches: list[str] = []
    _compare(
        "model",
        _public_data_attrs(expected_model, _SKIP_MODEL_ATTRS),
        _public_data_attrs(actual_model, _SKIP_MODEL_ATTRS),
        mismatches,
    )
    _compare(
        "state",
        _public_data_attrs(expected_state, set()),
        _public_data_attrs(actual_state, set()),
        mismatches,
    )
    if mismatches:
        head = "\n".join(mismatches[:50])
        suffix = "" if len(mismatches) <= 50 else f"\n... {len(mismatches) - 50} more mismatch(es)"
        raise AssertionError(f"Newton model/state parity failed:\n{head}{suffix}")

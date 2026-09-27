# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kamino accelerated PADMM policy composed with the frozen fixed operator.

Matches padmm/kernels.py's fused acceleration: restart rewinds hats to the
previous unaccelerated y/z, while the current f/y/z remains committed.
"""

import hashlib
from pathlib import Path
from types import FunctionType

import numpy as np

from tools.fpgs_bench import padmm_control as fixed

FIXED_SHA = "71ef26bde2eefa3a32edbaf017dd9122689df7c62f17113d5646c04a80e6717b"
RESTART_TOLERANCE = 0.999
INITIAL_MERIT = float(np.finfo(np.float32).max)


def accelerated_update(data, f_previous, y_previous, z_previous, y_hat, z_hat, a_previous, merit_previous):
    """Advance from hats but retain the unaccelerated proximal f and history."""
    f, y, z, info = fixed.update(data, f_previous, y_hat, z_hat)
    dy = data["scale"] * (y - y_hat)
    dz = (z - z_hat) / data["scale"]
    raw_merit = fixed.RHO * float(dy @ dy) + float(dz @ dz) / fixed.RHO
    restart = not (raw_merit < RESTART_TOLERANCE * merit_previous)
    if restart:
        a, beta, merit = 1.0, 0.0, merit_previous / RESTART_TOLERANCE
        next_y_hat, next_z_hat = y_previous.copy(), z_previous.copy()
    else:
        a = (1 + np.sqrt(1 + 4 * a_previous * a_previous)) / 2
        beta = (a_previous - 1) / a
        merit = raw_merit
        next_y_hat = y + beta * (y - y_previous)
        next_z_hat = z + beta * (z - z_previous)
    info.update(restart=restart, beta=float(beta), raw_merit=raw_merit, retained_merit=float(merit))
    return f, y, z, next_y_hat, next_z_hat, float(a), float(merit), info


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, trace=False):
    """Keep the fixed solver's factor, budget, physical check and publication.

    Only its per-pass update is replaced. Acceleration bookkeeping is computed
    even on the final physically converged pass, conservatively counted; those
    unused terminal hats cannot alter the published current y or velocity.
    """
    if hashlib.sha256(Path(fixed.__file__).read_bytes()).hexdigest() != FIXED_SHA:
        raise RuntimeError("Frozen fixed PADMM source changed")
    state = {}
    work = dict.fromkeys(
        (
            "restarts",
            "acceleration_checks",
            "merit_products",
            "merit_divisions",
            "merit_subtractions",
            "extrapolation_products",
            "extrapolation_values",
            "restart_copy_values",
        ),
        0,
    )

    def update(data, f, y, z):
        if not state:
            state.update(y_hat=np.zeros_like(y), z_hat=np.zeros_like(z), a=1.0, merit=INITIAL_MERIT)
        values = accelerated_update(data, f, y, z, state["y_hat"], state["z_hat"], state["a"], state["merit"])
        next_f, next_y, next_z, next_y_hat, next_z_hat, a, merit, info = values
        n = len(y)
        work["acceleration_checks"] += 1
        work["merit_products"] += 3 * n
        work["merit_divisions"] += n
        work["merit_subtractions"] += 2 * n
        if info["restart"]:
            work["restarts"] += 1
            work["restart_copy_values"] += 2 * n
        else:
            work["extrapolation_products"] += 2 * n
            work["extrapolation_values"] += 2 * n
        state.update(y_hat=next_y_hat, z_hat=next_z_hat, a=a, merit=merit)
        return next_f, next_y, next_z, info

    bound = FunctionType(fixed.solve.__code__, dict(fixed.solve.__globals__, update=update), "accelerated_padmm")
    bound.__kwdefaults__ = fixed.solve.__kwdefaults__
    result = bound(J, L, diagonal, rhs, types, parents, mu, vhat, iterations=iterations, incoming=incoming, trace=trace)
    result.work.update(work)
    result.work["acceleration_state_values"] = 2 * len(J)
    result.work["restart_tolerance"] = RESTART_TOLERANCE
    return result

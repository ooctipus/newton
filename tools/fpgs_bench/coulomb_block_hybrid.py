# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU six-block-pass control with remaining-budget original EX1 continuation."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block

REFERENCE_PATH = Path("/tmp/fpgs-anymal-lazy-reference-zE5yUe/method.py")
REFERENCE_SHA = "553c12619971e9cbeb950e1940797d91057950d70d789c1202861f1e6fcf485f"
if hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest() != REFERENCE_SHA:
    raise RuntimeError("Original parallel CPU reference changed")
_spec = importlib.util.spec_from_file_location("_pinned_anymal_parallel_reference", REFERENCE_PATH)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def parallel_continue(Z, diagonal, rhs, types, parents, mu, incoming, *, iterations):
    """Return final impulses and kinetic DELTA from the supplied incoming state.

    ``rhs`` is the physical residual at zero impulse, not the residual already
    including ``incoming``. Original EX1 steps contain denominator-only CFM.
    FP32 source associations are retained where the pinned CPU helper supports
    them; this does not emulate CUDA FMA or claim byte-identical native output.
    """
    Z = np.asarray(Z)
    dtype = Z.dtype.type
    diagonal, rhs, mu, incoming = (np.asarray(a, dtype=Z.dtype) for a in (diagonal, rhs, mu, incoming))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    n, dofs = Z.shape
    if iterations < 0 or n > 64 or not np.isin(types, (0, 2, 3)).all():
        raise ValueError("Unsupported continuation allowance or row layout")
    work = dict.fromkeys(
        (
            "sweeps",
            "majorizer_products",
            "residual_products",
            "transpose_products",
            "restart_count",
            "publish_products",
        ),
        0,
    )
    x, y = incoming.copy(), incoming.copy()
    if iterations == 0:
        return block.original.Result(np.zeros(dofs, Z.dtype), x, work)
    A = reference.gram(Z)
    if not np.isfinite(A).all() or np.any(np.diag(A) <= 0) or np.any(diagonal <= 0):
        raise ValueError("Unsupported nonpositive/nonfinite response")
    work["majorizer_products"] = n * n * dofs
    rho = np.abs(A).sum(axis=1, dtype=Z.dtype)
    eta = np.minimum(np.diag(A) / rho, dtype(1)) / diagonal
    pairs = reference.pairs_of({"kind": types, "parent": parents, "mu": mu})
    normal, active = types != 2, np.ones(n, bool)
    lanes, t = (32 if n <= 32 else 64), dtype(1)
    for _ in range(iterations):
        dv = reference.transpose_product(Z, y, active, lanes)
        residual = rhs.copy()
        for d in range(dofs):
            residual += Z[:, d] * dv[d]
        work["transpose_products"] += n * dofs
        work["residual_products"] += n * dofs
        value = y - eta * residual
        value[normal] = np.maximum(value[normal], 0)
        for parent, first, second in pairs:
            radius = max(dtype(0), mu[first] * value[parent])
            magnitude = dtype(np.sqrt(value[first] * value[first] + value[second] * value[second]))
            if radius <= 0:
                value[first] = value[second] = 0
            elif magnitude > radius:
                value[[first, second]] *= radius / magnitude
        local = np.zeros(lanes, Z.dtype)
        local[:n] = (y - value) * (value - x)
        for offset in (16, 8, 4, 2, 1):
            local = local + local[np.arange(lanes) ^ offset]
        restart = local[0] + (local[32] if lanes == 64 else dtype(0)) > 0
        if restart:
            t = dtype(1)
            work["restart_count"] += 1
        t_next = dtype(0.5) * (dtype(1) + dtype(np.sqrt(dtype(1) + dtype(4) * t * t)))
        beta = (t - dtype(1)) / t_next
        changed = np.any(np.abs(value - x) > dtype(1e-4) * (np.abs(value) + dtype(1e-4)))
        y, x, t = value + beta * (value - x), value, t_next
        work["sweeps"] += 1
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite continuation")
        if not changed:
            break
    delta = np.zeros(dofs, Z.dtype)
    for row in range(n):
        change = x[row] - incoming[row]
        if change != 0:
            delta += Z[row] * change
            work["publish_products"] += dofs
    return block.original.Result(delta, x, work)


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None):
    """Keep the original total allowance; never reset the current physical state."""
    if iterations < 0:
        raise ValueError("Negative outer allowance")
    result = block.solve(
        J, L, diagonal, rhs, types, parents, mu, vhat, iterations=min(6, iterations), early_stop=True, incoming=incoming
    )
    consumed = result.work["sweeps"]
    stopped = bool(result.work["curve"] and result.work["curve"][-1]["physical"]["stop"])
    work = {
        "coupled_sweeps": consumed,
        "parallel_sweeps": 0,
        "majorizer_products": 0,
        "coupled_work": result.work,
        "physical_stop": stopped,
        "fallback": False,
        "allowance": iterations,
        "unused_allowance": iterations - consumed,
        "cpu_fallback_Z_rebuild_rows": 0,
        "cpu_fallback_Z_triangular_products": 0,
        "handoff_row_dots": 0,
        "handoff_transpose_products": 0,
    }
    remaining = iterations - consumed
    if stopped or remaining == 0:
        return block.original.Result(result.velocity, result.impulses, work)
    J, L = np.asarray(J, float), np.asarray(L, float)
    Z = np.linalg.solve(L, J.T).T
    n, dofs = Z.shape
    # CPU composition repeats this preparation because the frozen block oracle
    # does not export Z. Record it rather than claiming it was cached for free.
    work["cpu_fallback_Z_rebuild_rows"] = n
    work["cpu_fallback_Z_triangular_products"] = n * dofs * (dofs - 1) // 2
    seed = result.impulses
    zero_rhs = J @ result.velocity + rhs - Z @ (Z.T @ seed)
    work["handoff_row_dots"] = 2 * n
    work["handoff_transpose_products"] = n * dofs
    continued = parallel_continue(Z, diagonal, zero_rhs, types, parents, mu, seed, iterations=remaining)
    velocity = result.velocity + np.linalg.solve(L.T, continued.velocity)
    work.update(
        parallel_sweeps=continued.work["sweeps"],
        parallel_work=continued.work,
        majorizer_products=continued.work["majorizer_products"],
        fallback=True,
        unused_allowance=remaining - continued.work["sweeps"],
        cpu_final_backsolve_products=dofs * (dofs - 1) // 2,
    )
    assert work["coupled_sweeps"] + work["parallel_sweeps"] <= iterations
    return block.original.Result(velocity, continued.impulses, work)

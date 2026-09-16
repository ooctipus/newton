# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU fixed spectral natural map with original Nesterov and residual carry.

This is a feasibility control, not a convergent algorithm or native owner.
It deliberately has no line search, AA history, EX1 majorizer or local roots.
The stop is the existing AA4 physical stop (natural 3e-5, not block's 1e-5).
"""

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench.coulomb_block_hybrid import reference


def _score(residual, impulse, rhs, physical_diagonal, spectral, types, pairs, mu):
    normal = types != 2
    projected = np.maximum(0, impulse[normal] - residual[normal] / physical_diagonal[normal])
    natural = float(np.max(np.abs(physical_diagonal[normal] * (impulse[normal] - projected)), initial=0))
    score = {
        "normal": float(np.max(-residual[normal], initial=0)),
        "complementarity": float(np.max(np.abs(impulse[normal] * residual[normal]), initial=0)),
        "negative": float(np.max(-impulse[normal], initial=0)),
        "mdp": 0.0,
        "cone": 0.0,
    }
    for parent, first, second in pairs:
        tangent, rt = impulse[[first, second]], residual[[first, second]]
        radius = max(0.0, float(mu[first] * impulse[parent]))
        trial = tangent - rt / spectral[first]
        length = np.linalg.norm(trial)
        projected = trial * min(1.0, radius / length) if length > 0 else trial
        natural = max(natural, float(np.max(np.abs(spectral[first] * (tangent - projected)))))
        score["mdp"] = max(score["mdp"], float(tangent @ rt + radius * np.linalg.norm(rt)))
        score["cone"] = max(score["cone"], float(np.linalg.norm(tangent)) - radius)
    scale = max(1.0, float(np.max(np.abs(rhs), initial=0)), float(np.max(np.abs(residual - rhs), initial=0)))
    score["natural"] = natural / scale
    score["stop"] = bool(
        np.isfinite(residual).all()
        and np.isfinite(impulse).all()
        and all(score[key] < 3e-5 for key in ("natural", "normal", "complementarity", "mdp", "cone"))
        and score["negative"] < 1e-7
    )
    return score


def natural_continue(Z, diagonal, rhs, types, parents, mu, incoming, *, iterations=24, trace=False):
    """Continue from impulses using zero-impulse physical RHS; return delta-u.

    One transpose/row operator pair is charged per pass, including a pass which
    discovers that current x is already physically converged before proposing.
    The final physical scan is offline reporting only, never another update.
    """
    Z = np.asarray(Z)
    dtype = Z.dtype.type
    diagonal, rhs, mu, incoming = (np.asarray(a, dtype=Z.dtype) for a in (diagonal, rhs, mu, incoming))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    n, dofs = Z.shape
    if iterations < 0 or n > 64 or not np.isin(types, (0, 2, 3)).all():
        raise ValueError("Unsupported allowance or row layout")
    pairs = reference.pairs_of({"kind": types, "parent": parents, "mu": mu})
    physical_diagonal = np.sum(Z * Z, axis=1, dtype=Z.dtype)
    cfm = diagonal - physical_diagonal
    if np.any(physical_diagonal <= 0) or np.any(diagonal <= 0) or not np.isfinite(Z).all():
        raise ValueError("Nonpositive or nonfinite physical response")
    eta, spectral = dtype(1) / diagonal, physical_diagonal.copy()
    for _parent, first, second in pairs:
        a, c = physical_diagonal[[first, second]]
        cross = np.dot(Z[first], Z[second])
        largest = dtype(0.5) * (a + c + dtype(np.sqrt((a - c) ** 2 + dtype(4) * cross * cross)))
        spectral[[first, second]] = largest
        eta[[first, second]] = dtype(1) / (largest + max(cfm[first], cfm[second]))
    if not np.isfinite(eta).all() or np.any(eta <= 0):
        raise ValueError("Invalid spectral step")
    work = dict.fromkeys(
        (
            "sweeps",
            "operator_passes",
            "operator_products",
            "transpose_products",
            "residual_products",
            "majorizer_products",
            "stop_operator_products",
            "stop_row_projections",
            "proposal_row_projections",
            "residual_history_values",
            "residual_recovery_values",
            "restart_count",
            "publish_products",
        ),
        0,
    )
    work.update(
        setup_products=(n + len(pairs)) * dofs,
        physical_stop=False,
        delta_stop=False,
        curve=[],
        trace=[],
        first_stop=None,
        allowance=iterations,
    )
    x, y = incoming.copy(), incoming.copy()
    previous_residual, beta_previous, t = rhs.copy(), dtype(0), dtype(1)
    normal, active, lanes = types != 2, np.ones(n, bool), (32 if n <= 32 else 64)
    for _ in range(iterations):
        action = reference.transpose_product(Z, y, active, lanes)
        residual_y = rhs.copy()
        for dof in range(dofs):
            residual_y += Z[:, dof] * action[dof]
        current = (residual_y + beta_previous * previous_residual) / (dtype(1) + beta_previous)
        work["operator_passes"] += 1
        work["transpose_products"] += n * dofs
        work["residual_products"] += n * dofs
        work["operator_products"] += 2 * n * dofs
        work["residual_recovery_values"] += n
        score = _score(current, x, rhs, physical_diagonal, spectral, types, pairs, mu)
        work["stop_row_projections"] += n
        work["curve"].append({"sweep": work["sweeps"], "physical": score})
        if trace:
            work["trace"].append({"impulse": x.copy(), "residual": current.copy(), "beta": float(beta_previous)})
        if score["stop"]:
            work["physical_stop"], work["first_stop"] = True, work["sweeps"]
            break
        previous_residual = current.copy()
        work["residual_history_values"] += n
        value = y - eta * residual_y
        value[normal] = np.maximum(value[normal], 0)
        for parent, first, second in pairs:
            radius = max(dtype(0), mu[first] * value[parent])
            magnitude = dtype(np.sqrt(value[first] * value[first] + value[second] * value[second]))
            if radius <= 0:
                value[first] = value[second] = 0
            elif magnitude > radius:
                value[[first, second]] *= radius / magnitude
        work["proposal_row_projections"] += n
        local = np.zeros(lanes, Z.dtype)
        local[:n] = (y - value) * (value - x)
        for offset in (16, 8, 4, 2, 1):
            local = local + local[np.arange(lanes) ^ offset]
        if local[0] + (local[32] if lanes == 64 else dtype(0)) > 0:
            t = dtype(1)
            work["restart_count"] += 1
        t_next = dtype(0.5) * (dtype(1) + dtype(np.sqrt(dtype(1) + dtype(4) * t * t)))
        beta = (t - dtype(1)) / t_next
        changed = np.any(np.abs(value - x) > dtype(1e-4) * (np.abs(value) + dtype(1e-4)))
        y, x, t, beta_previous = value + beta * (value - x), value, t_next, beta
        work["sweeps"] += 1
        if not changed:
            work["delta_stop"] = True
            break
    delta = reference.transpose_product(Z, x - incoming, active, lanes)
    work["publish_products"] = n * dofs
    final = rhs + Z @ (Z.T @ x)
    work["offline_final_operator_products"] = 2 * n * dofs
    work["final_physical"] = _score(final, x, rhs, physical_diagonal, spectral, types, pairs, mu)
    work["offline_final_row_projections"] = n
    work["unused_allowance"] = iterations - work["sweeps"]
    return block.original.Result(delta, x, work)


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, trace=False):
    """Preserve current velocity/incoming impulse and publish only their delta."""
    J, L, vhat = (np.asarray(a, float) for a in (J, L, vhat))
    incoming = np.zeros(len(J)) if incoming is None else np.asarray(incoming, float)
    Z = np.linalg.solve(L, J.T).T
    zero_rhs = J @ vhat + rhs - Z @ (Z.T @ incoming)
    result = natural_continue(Z, diagonal, zero_rhs, types, parents, mu, incoming, iterations=iterations, trace=trace)
    result.work.update(
        cpu_Z_triangular_products=len(J) * len(L) * (len(L) - 1) // 2,
        cpu_zero_rhs_products=3 * J.size,
        cpu_decode_products=len(L) * (len(L) - 1) // 2,
    )
    return block.original.Result(vhat + np.linalg.solve(L.T, result.velocity), result.impulses, result.work)

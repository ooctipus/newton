# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only fixed-policy unrelaxed proposal lookahead / permanent Q half latch.

The proof, admission and pre-evaluation policy are recorded in
reports/fpgs/ANYMAL_SPECTRAL_RESIDUAL_20260916.md. This changes the latch metric,
not the local contact proposal or the final physical gates. No native hook.
"""

import hashlib
from pathlib import Path

import numpy as np

from tools.fpgs_bench import spectral_jacobi_control as frozen

FROZEN_SHA = "f3cb2e5e15de8ff7d5b876bbf966d69139c134226a2338d15072ffa998385a70"


def prepare(J, L, diagonal, types, parents, mu, incoming):
    """Prepare unchanged physical coefficients and the proved Q weights once."""
    Z = np.linalg.solve(L, J.T).T
    physical_diagonal = np.sum(Z * Z, axis=1)
    if (
        np.any(physical_diagonal <= 0)
        or np.any(diagonal <= 0)
        or not np.isfinite(diagonal).all()
        or not np.isfinite(Z).all()
    ):
        raise ValueError("invalid_response")
    cfm = diagonal - physical_diagonal
    if not np.isin(types, (0, 2, 3)).all():
        raise ValueError("unsupported_rows")
    pairs = frozen.reference.pairs_of({"kind": types, "parent": parents, "mu": mu})
    if {row for _normal, first, second in pairs for row in (first, second)} != set(np.flatnonzero(types == 2)):
        raise ValueError("noncanonical_triples")
    normals = np.flatnonzero(types != 2)
    if not np.isfinite(incoming).all() or np.any(incoming[normals] < 0):
        raise ValueError("infeasible_incoming")
    cache = []
    for normal, first, second in pairs:
        friction = mu[first]
        if not np.isfinite(friction) or friction < 0 or friction != mu[second]:
            raise ValueError("invalid_friction")
        if np.linalg.norm(incoming[[first, second]]) > friction * incoming[normal]:
            raise ValueError("infeasible_incoming")
        a, c = physical_diagonal[[first, second]]
        cross = np.array([Z[normal] @ Z[first], Z[normal] @ Z[second]])
        a12 = Z[first] @ Z[second]
        spectral = 0.5 * (a + c) + np.hypot(0.5 * (a - c), a12) + max(cfm[first], cfm[second])
        tangent_diagonal = max(diagonal[first], diagonal[second])
        if not np.isfinite(spectral) or spectral < tangent_diagonal or spectral <= 0:
            raise ValueError("invalid_spectral_bound")
        weight_t = spectral / np.sqrt(tangent_diagonal)
        weight_n = (friction * spectral + np.sum(np.abs(cross))) / np.sqrt(tangent_diagonal)
        cache.append((normal, first, second, cross, spectral, friction, weight_t, weight_n))
    return {"Z": Z, "diagonal": diagonal, "normal_rows": normals, "cache": cache, "sqrt_diagonal": np.sqrt(diagonal)}


def propose(prepared, impulse, residual, cold_scale):
    """Return the identical normal-first T and an upper bound on natural defect."""
    normal_rows = prepared["normal_rows"]
    proposed = impulse.copy()
    proposed[normal_rows] = np.maximum(
        0.0, impulse[normal_rows] - residual[normal_rows] / prepared["diagonal"][normal_rows]
    )
    q = float(
        np.max(prepared["sqrt_diagonal"][normal_rows] * np.abs(proposed[normal_rows] - impulse[normal_rows]), initial=0)
    )
    for normal, first, second, cross, spectral, friction, weight_t, weight_n in prepared["cache"]:
        take = np.array([first, second])
        normal_delta = proposed[normal] - impulse[normal]
        trial = impulse[take] - (residual[take] + cross * normal_delta) / spectral
        length = np.linalg.norm(trial)
        radius = friction * proposed[normal]
        proposed[take] = trial * min(1.0, radius / length) if length > 0 else trial
        q = max(q, weight_t * np.sum(np.abs(proposed[take] - impulse[take])) + weight_n * abs(normal_delta))
    q /= cold_scale
    if not np.isfinite(proposed).all() or not np.isfinite(q):
        q = np.inf
    return proposed, float(q)


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, early_stop=True):
    """Reuse unrelaxed lookahead; full checks gate actual stops and every return."""
    if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != FROZEN_SHA:
        raise RuntimeError("Frozen CPU map changed")
    if iterations < 0:
        raise ValueError("Negative iteration allowance")
    J, L, diagonal, rhs, mu, vhat = (np.asarray(a, float) for a in (J, L, diagonal, rhs, mu, vhat))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    n, dofs = J.shape
    impulse = np.zeros(n) if incoming is None else np.asarray(incoming, float).copy()
    try:
        prepared = prepare(J, L, diagonal, types, parents, mu, impulse)
    except ValueError as error:
        result = frozen.solve(
            J,
            L,
            diagonal,
            rhs,
            types,
            parents,
            mu,
            vhat,
            iterations=iterations,
            incoming=incoming,
            early_stop=early_stop,
        )
        result.work.update(
            fallbacks=1,
            fallback_reason=str(error),
            admission_setup_products=(n + n) * dofs,
            admission_cpu_Z_triangular_products=n * dofs * (dofs - 1) // 2,
        )
        return result
    Z, cache, normals = prepared["Z"], prepared["cache"], prepared["normal_rows"]
    residual = J @ vhat + rhs
    initial_residual = residual.copy()
    kinetic_delta = np.zeros(dofs)
    cold_scale = 1 + float(np.max(np.abs(residual) / np.sqrt(diagonal), initial=0))
    work = dict.fromkeys(
        (
            "sweeps",
            "row_transactions",
            "changed_transactions",
            "operator_products",
            "transpose_products",
            "residual_products",
            "proposal_row_projections",
            "stop_row_projections",
            "midpoint_row_projections",
            "midpoint_values",
            "root_probes",
            "majorizer_products",
            "stop_row_dots",
            "initial_merit_row_projections",
            "initial_proposals",
            "lookahead_proposals",
            "midpoint_replacement_proposals",
            "final_lookahead_proposals",
            "q_row_visits",
            "full_checks",
            "q_gated_full_checks",
            "failed_q_gates",
            "final_only_full_checks",
            "fallbacks",
            "proposal_blocks",
            "proposal_singletons",
            "full_check_blocks",
            "full_check_singletons",
        ),
        0,
    )
    work.update(
        setup_products=(n + 3 * len(cache)) * dofs,
        setup_cross_products=3 * len(cache) * dofs,
        setup_blocks=len(cache),
        setup_q_coefficients=2 * len(cache) + n,
        producer_rows=n,
        cold_scale_row_dots=n,
        added_persistent_proposal_values=n,
        cpu_Z_triangular_products=n * dofs * (dofs - 1) // 2,
        curve=[],
        first_stop=None,
        latch_step=None,
        allowance=iterations,
    )

    def proposal(x, r, kind):
        value = propose(prepared, x, r, cold_scale)
        work[kind] += 1
        work["proposal_row_projections"] += n
        work["q_row_visits"] += n
        work["proposal_blocks"] += len(cache)
        work["proposal_singletons"] += len(normals) - len(cache)
        work["row_transactions"] += len(normals)
        work["changed_transactions"] += sum(
            int(
                np.any(
                    value[0][row : row + (3 if row + 1 < n and types[row + 1] == 2 else 1)]
                    != x[row : row + (3 if row + 1 < n and types[row + 1] == 2 else 1)]
                )
            )
            for row in normals
        )
        return value

    def physical(x, r):
        work["full_checks"] += 1
        work["stop_row_projections"] += n
        work["full_check_blocks"] += len(cache)
        work["full_check_singletons"] += len(normals) - len(cache)
        return frozen.physical_score(r, x, diagonal, types, parents, mu, cold_scale)

    relaxation, score, q = 1.0, None, None
    if iterations:
        proposed, q = proposal(impulse, residual, "initial_proposals")
    for iteration in range(iterations):
        next_impulse = impulse + relaxation * (proposed - impulse)
        delta_u = Z.T @ (next_impulse - impulse)
        next_delta = kinetic_delta + delta_u
        next_residual = residual + Z @ delta_u
        work["transpose_products"] += n * dofs
        work["residual_products"] += n * dofs
        work["operator_products"] += 2 * n * dofs
        next_proposed, next_q = proposal(next_impulse, next_residual, "lookahead_proposals")
        trial_q = next_q
        if relaxation == 1.0 and next_q > q:
            relaxation = 0.5
            next_impulse = 0.5 * (impulse + next_impulse)
            next_delta = 0.5 * (kinetic_delta + next_delta)
            next_residual = 0.5 * (residual + next_residual)
            next_proposed, next_q = proposal(next_impulse, next_residual, "midpoint_replacement_proposals")
            work["latch_step"] = iteration + 1
            work["midpoint_values"] += 2 * n + dofs
        old_q = q
        impulse, kinetic_delta, residual, proposed, q = next_impulse, next_delta, next_residual, next_proposed, next_q
        work["sweeps"] += 1
        q_gate = np.isfinite(q) and q <= 1e-5
        final = iteration + 1 == iterations
        score = physical(impulse, residual) if q_gate or final else None
        if q_gate:
            work["q_gated_full_checks"] += 1
            work["failed_q_gates"] += int(not score["stop"])
        elif final:
            work["final_only_full_checks"] += 1
        work["curve"].append(
            {
                "sweep": iteration + 1,
                "current_q": old_q,
                "trial_q": trial_q,
                "accepted_q": q,
                "relaxation": relaxation,
                "physical": score,
            }
        )
        if score is not None and score["stop"] and work["first_stop"] is None:
            work["first_stop"] = iteration + 1
        if early_stop and score is not None and score["stop"]:
            break
    if score is None:
        score = physical(impulse, residual)
        work["final_only_full_checks"] += 1
    work.update(
        relaxation=relaxation,
        physical_stop=bool(score["stop"]),
        final_physical=score,
        final_q=q,
        final_lookahead_proposals=int(work["sweeps"] > 0),
        residual_carry_error=float(np.max(np.abs(residual - (initial_residual + Z @ kinetic_delta)), initial=0)),
        cpu_final_backsolve_products=dofs * (dofs - 1) // 2,
        unused_allowance=iterations - work["sweeps"],
    )
    return frozen.block.original.Result(vhat + np.linalg.solve(L.T, kinetic_delta), impulse, work)

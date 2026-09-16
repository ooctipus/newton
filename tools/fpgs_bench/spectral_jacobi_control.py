# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU simultaneous spectral contact proposals with one permanent half latch.

Only the proposal differs from frozen contact-block Jacobi: no local root or
Schur solve. Its precision-consistent physical merit/stop and delta-operator
dataflow are retained. This is a bounded control, not a native runtime hook.
"""

import hashlib
from pathlib import Path

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench.coulomb_block_hybrid import reference

FROZEN_BLOCK_SHA = "0fc44df283d5284763f7fd4889b4b8cc475b197bb6314714cf1952bd0a3d5295"
BOUNDS = {"natural": 1e-5, "normal": 3e-5, "complementarity": 3e-5, "mdp": 3e-5, "cone": 3e-5}


def physical_score(residual, impulse, diagonal, types, parents, mu, cold_scale):
    """Reuse the original block metrics with its already-tested cone precision."""
    score = block.physical_from_residual(residual, impulse, diagonal, types, parents, mu, cold_scale)
    score["stop"] = all(np.isfinite(score[key]) and score[key] <= bound for key, bound in BOUNDS.items())
    return score


def merit(score):
    """Use only the fixed physical bounds, never a reference solution."""
    values = [score[key] / bound for key, bound in BOUNDS.items()]
    return float(max(values)) if np.isfinite(values).all() else np.inf


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, early_stop=True):
    """Propose locally from old state, then apply one physical operator pair.

    Normal first within each contact; its physical cross terms correct tangent
    residuals before one common spectral step and disk projection. Every other
    contact remains frozen. CFM is used only in proposal denominators. A first
    physical-merit increase permanently switches relaxation from one to half;
    its midpoint residual is affine and needs no second operator evaluation.
    """
    if hashlib.sha256(Path(block.__file__).read_bytes()).hexdigest() != FROZEN_BLOCK_SHA:
        raise RuntimeError("Frozen physical metric source changed")
    J, L, diagonal, rhs, mu, vhat = (np.asarray(a, float) for a in (J, L, diagonal, rhs, mu, vhat))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    n, dofs = J.shape
    if iterations < 0 or not np.isin(types, (0, 2, 3)).all():
        raise ValueError("Unsupported allowance or row type")
    Z = np.linalg.solve(L, J.T).T
    physical_diagonal = np.sum(Z * Z, axis=1)
    if (
        np.any(physical_diagonal <= 0)
        or np.any(diagonal <= 0)
        or not np.isfinite(diagonal).all()
        or not np.isfinite(Z).all()
    ):
        raise ValueError("Nonpositive/nonfinite response")
    cfm = diagonal - physical_diagonal
    pairs = reference.pairs_of({"kind": types, "parent": parents, "mu": mu})
    if {row for _parent, first, second in pairs for row in (first, second)} != set(np.flatnonzero(types == 2)):
        raise ValueError("Only canonical contact triples are supported")
    cache = []
    for normal, first, second in pairs:
        if not np.isfinite(mu[first]) or mu[first] < 0 or mu[first] != mu[second]:
            raise ValueError("Invalid Coulomb coefficient")
        a, c = physical_diagonal[[first, second]]
        a01, a02, a12 = Z[normal] @ Z[first], Z[normal] @ Z[second], Z[first] @ Z[second]
        largest = 0.5 * (a + c) + np.hypot(0.5 * (a - c), a12)
        denominator = largest + max(cfm[first], cfm[second])
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid tangent spectral denominator")
        cache.append((normal, first, second, np.array([a01, a02]), denominator))
    impulse = np.zeros(n) if incoming is None else np.asarray(incoming, float).copy()
    kinetic_delta = np.zeros(dofs)
    residual = J @ vhat + rhs
    cold_scale = 1 + float(np.max(np.abs(residual) / np.sqrt(diagonal), initial=0))
    score = physical_score(residual, impulse, diagonal, types, parents, mu, cold_scale)
    energy, relaxation = merit(score), 1.0
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
        ),
        0,
    )
    work.update(
        setup_products=(n + 3 * len(pairs)) * dofs,
        setup_cross_products=3 * len(pairs) * dofs,
        setup_blocks=len(pairs),
        producer_rows=n,
        cold_scale_row_dots=n,
        initial_merit_row_projections=n,
        cpu_Z_triangular_products=n * dofs * (dofs - 1) // 2,
        curve=[],
        first_stop=None,
        latch_step=None,
        allowance=iterations,
    )
    normal_rows = np.flatnonzero(types != 2)
    for iteration in range(iterations):
        proposed = impulse.copy()
        proposed[normal_rows] = np.maximum(0.0, impulse[normal_rows] - residual[normal_rows] / diagonal[normal_rows])
        for normal, first, second, cross, denominator in cache:
            take = np.array([first, second])
            tangent_residual = residual[take] + cross * (proposed[normal] - impulse[normal])
            trial = impulse[take] - tangent_residual / denominator
            radius = max(0.0, mu[first] * proposed[normal])
            length = np.linalg.norm(trial)
            proposed[take] = trial * min(1.0, radius / length) if length > 0 else trial
        work["row_transactions"] += len(normal_rows)
        work["changed_transactions"] += sum(
            int(
                np.any(
                    proposed[row : row + (3 if row + 1 < n and types[row + 1] == 2 else 1)]
                    != impulse[row : row + (3 if row + 1 < n and types[row + 1] == 2 else 1)]
                )
            )
            for row in normal_rows
        )
        work["proposal_row_projections"] += n
        next_impulse = impulse + relaxation * (proposed - impulse)
        delta_u = Z.T @ (next_impulse - impulse)
        next_delta = kinetic_delta + delta_u
        next_residual = residual + Z @ delta_u
        work["transpose_products"] += n * dofs
        work["residual_products"] += n * dofs
        work["operator_products"] += 2 * n * dofs
        next_score = physical_score(next_residual, next_impulse, diagonal, types, parents, mu, cold_scale)
        next_energy = merit(next_score)
        proposed_energy = next_energy
        work["stop_row_projections"] += n
        if relaxation == 1.0 and next_energy > energy:
            relaxation = 0.5
            next_impulse = 0.5 * (impulse + next_impulse)
            next_delta = 0.5 * (kinetic_delta + next_delta)
            next_residual = 0.5 * (residual + next_residual)
            next_score = physical_score(next_residual, next_impulse, diagonal, types, parents, mu, cold_scale)
            next_energy = merit(next_score)
            work["latch_step"] = iteration + 1
            work["midpoint_row_projections"] += n
            work["midpoint_values"] += 2 * n + dofs
        work["sweeps"] += 1
        work["curve"].append(
            {
                "sweep": iteration + 1,
                "current_merit": energy,
                "proposed_merit": proposed_energy,
                "accepted_merit": next_energy,
                "relaxation": relaxation,
                "physical": next_score,
            }
        )
        impulse, kinetic_delta, residual, score, energy = (
            next_impulse,
            next_delta,
            next_residual,
            next_score,
            next_energy,
        )
        if score["stop"] and work["first_stop"] is None:
            work["first_stop"] = iteration + 1
        if early_stop and score["stop"]:
            break
    work.update(
        relaxation=relaxation,
        physical_stop=bool(score["stop"]),
        final_physical=score,
        cpu_final_backsolve_products=dofs * (dofs - 1) // 2,
        unused_allowance=iterations - work["sweeps"],
    )
    return block.original.Result(vhat + np.linalg.solve(L.T, kinetic_delta), impulse, work)

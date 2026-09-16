# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU simultaneous contact-block proposals with one permanent damping latch."""

import hashlib
import json
from functools import partial
from pathlib import Path
from types import FunctionType

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench.coulomb_block_guarded import merit

FROZEN_SHA = "0fc44df283d5284763f7fd4889b4b8cc475b197bb6314714cf1952bd0a3d5295"


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, early_stop=True):
    """Read one state per pass, combine updates, and reuse its exact residual.

    The sole relaxation change is 1 to 0.5 on the first merit increase.
    Subsequent half steps are not halved again. Neither a midpoint nor a
    physical stopping projection consumes another outer iteration.
    """
    if hashlib.sha256(Path(block.__file__).read_bytes()).hexdigest() != FROZEN_SHA:
        raise RuntimeError("Frozen local contact source changed")
    if iterations < 0:
        raise ValueError("Negative outer allowance")
    J, L, diagonal, rhs, mu, vhat = (np.asarray(a, float) for a in (J, L, diagonal, rhs, mu, vhat))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    n, dofs = J.shape
    Z = np.linalg.solve(L, J.T).T
    impulse = np.zeros(n) if incoming is None else np.asarray(incoming, float).copy()
    kinetic_delta = np.zeros(dofs)
    residual = J @ vhat + rhs
    cold_scale = 1 + float(np.max(np.abs(residual) / np.sqrt(diagonal), initial=0))
    score = block.physical_from_residual(residual, impulse, diagonal, types, parents, mu, cold_scale)
    energy = merit(score)
    relaxation = 1.0
    work = dict.fromkeys(
        (
            "sweeps",
            "row_transactions",
            "changed_transactions",
            "setup_blocks",
            "setup_products",
            "root_probes",
            "sliding_roots",
            "block_open",
            "block_stick",
            "block_normal",
            "block_slip",
            "metric_fallbacks",
            "legacy_root_probes",
            "legacy_cpu_gram_products",
            "fallback_self_gram_products",
            "fallback_residual_products",
            "fallback_response_products",
            "operator_products",
            "residual_products",
            "transpose_products",
            "stop_row_dots",
            "stop_row_projections",
            "midpoint_row_projections",
            "midpoint_values",
        ),
        0,
    )
    work.update(
        curve=[],
        first_stop=None,
        latch_step=None,
        relaxation=1.0,
        producer_rows=n,
        cold_scale_row_dots=n,
        initial_merit_row_projections=n,
        cpu_Z_triangular_products=n * dofs * (dofs - 1) // 2,
    )
    cache = {}
    for iteration in range(iterations):
        proposed = impulse.copy()
        row = 0
        while row < n:
            work["row_transactions"] += 1
            canonical = (
                types[row] == 0
                and row + 2 < n
                and np.all(types[row + 1 : row + 3] == 2)
                and np.all(parents[row + 1 : row + 3] == row)
            )
            if canonical:
                take = slice(row, row + 3)
                old = impulse[take]
                if not np.any(old) and residual[row] >= 0:
                    work["block_open"] += 1
                    row += 3
                    continue
                if row not in cache:
                    matrix = np.diag(diagonal[take])
                    for first, second in ((0, 1), (0, 2), (1, 2)):
                        matrix[first, second] = matrix[second, first] = Z[row + first] @ Z[row + second]
                    prepared = block.prepare_block(matrix)
                    cache[row] = matrix, prepared if prepared is not None else False, None
                    work["setup_blocks"] += 1
                    work["setup_products"] += 3 * dofs
                matrix, prepared, physical = cache[row]
                trial, info = None, {"probes": 0, "kind": "unsafe"}
                if np.isfinite(mu[row + 1]) and mu[row + 1] >= 0 and mu[row + 1] == mu[row + 2]:
                    trial, info = block.local_block(
                        matrix, residual[take] - matrix @ old, mu[row + 1], prepared=prepared
                    )
                work["root_probes"] += info["probes"]
                work["sliding_roots"] += int(info["kind"] == "slip")
                if trial is None:
                    # In local residual coordinates, J=I and Y=G give exactly
                    # the unchanged metric transaction r <- r + G*delta_lambda.
                    if physical is None:
                        physical = matrix.copy()
                        np.fill_diagonal(physical, np.sum(Z[take] * Z[take], axis=1))
                        cache[row] = matrix, prepared, physical
                        work["fallback_self_gram_products"] += 3 * dofs
                    (_local_residual, trial, _stats), counters = block.original.counted_reference(
                        np.eye(3),
                        physical,
                        diagonal[take],
                        np.zeros(3),
                        types[take],
                        np.array([-1, 0, 0]),
                        mu[take],
                        residual[take],
                        iterations=1,
                        incoming=old,
                    )
                    work["metric_fallbacks"] += 1
                    work["legacy_root_probes"] += counters["root_probes"]
                    work["legacy_cpu_gram_products"] += counters["cpu_full_gram_products"]
                    work["fallback_residual_products"] += 3 * counters["residual_row_dots"]
                    work["fallback_response_products"] += 3 * counters["response_row_actions"]
                else:
                    work[f"block_{info['kind']}"] += 1
                proposed[take] = trial
                work["changed_transactions"] += int(np.any(trial != old))
                row += 3
                continue
            if types[row] not in (0, 3):
                raise ValueError("Only scalar normal/limit rows and canonical contact triples are supported")
            if diagonal[row] > 0:
                proposed[row] = max(0.0, impulse[row] - residual[row] / diagonal[row])
                work["changed_transactions"] += int(proposed[row] != impulse[row])
            row += 1
        next_impulse = impulse + relaxation * (proposed - impulse)
        delta_u = Z.T @ (next_impulse - impulse)
        next_delta = kinetic_delta + delta_u
        next_residual = residual + Z @ delta_u
        work["transpose_products"] += n * dofs
        work["residual_products"] += n * dofs
        work["operator_products"] += 2 * n * dofs
        next_score = block.physical_from_residual(next_residual, next_impulse, diagonal, types, parents, mu, cold_scale)
        next_energy = merit(next_score)
        proposed_energy = next_energy
        work["stop_row_projections"] += n
        if relaxation == 1.0 and next_energy > energy:
            relaxation = 0.5
            work["latch_step"] = iteration + 1
            next_impulse = 0.5 * (impulse + next_impulse)
            next_delta = 0.5 * (kinetic_delta + next_delta)
            next_residual = 0.5 * (residual + next_residual)
            next_score = block.physical_from_residual(
                next_residual, next_impulse, diagonal, types, parents, mu, cold_scale
            )
            next_energy = merit(next_score)
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
    )
    velocity = vhat + np.linalg.solve(L.T, kinetic_delta)
    return block.original.Result(velocity, impulse, work)


def main():
    """Falsify the fixed policy on all sixteen pinned G1 cases at eight passes."""
    assess = FunctionType(
        block.original.assess.__code__,
        dict(block.original.assess.__globals__, solve=partial(solve, iterations=8)),
        "assess_jacobi",
    )
    _, records = block.original.saved_records()
    cases = []
    for gpu, record, data in records:
        for index, world in enumerate(data["worlds"]):
            J, L = data[f"J_world_{world}"].astype(float), data["L_by_size"][index].astype(float)
            diagonal, rhs, types, parents, mu = (
                data[f"{key}_{world}"] for key in ("diag", "rhs", "row_type", "row_parent", "row_mu")
            )
            vhat = data["v_hat"].reshape(4, 43)[index].astype(float)
            case = assess(J, L, diagonal, rhs, types, parents, mu, vhat)
            case.update(gpu=gpu, step=record["step"], world=int(world), rows=len(J))
            cases.append(case)
            print("CASE " + json.dumps(case), flush=True)
    print(
        "RESULT "
        + json.dumps(
            {
                "passed": sum(c["passed"] for c in cases),
                "total": len(cases),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()

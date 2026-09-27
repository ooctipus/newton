# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU ordered normal/shared-spectral-tangent GS with fixed physical stop.

This is the existing offline physical reference's local update law, tested at
the original finite allowance. No local root, all-pair Gram or momentum is used.
"""

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench.coulomb_block_hybrid import reference
from tools.fpgs_bench.natural_nesterov_control import _score


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, early_stop=True):
    """Update each normal before its tangent pair, retaining current state.

    CFM appears only in update denominators. Stopping scans the actual physical
    residual at the end of each complete pass, charging all row dots/projections.
    The common tangent denominator is lambda_max(physical Gtt)+max(CFMt).
    """
    J, L, diagonal, rhs, mu, velocity = (np.asarray(a, float) for a in (J, L, diagonal, rhs, mu, vhat))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    if iterations < 0 or not np.isin(types, (0, 2, 3)).all():
        raise ValueError("Unsupported allowance or row type")
    n, dofs = J.shape
    Z = np.linalg.solve(L, J.T).T
    Y = np.linalg.solve(L.T, Z.T).T
    physical_diagonal = np.sum(Z * Z, axis=1)
    if np.any(physical_diagonal <= 0) or np.any(diagonal <= 0) or not np.isfinite(Z).all():
        raise ValueError("Nonpositive/nonfinite physical response")
    cfm = diagonal - physical_diagonal
    spectral = physical_diagonal.copy()
    pairs = reference.pairs_of({"kind": types, "parent": parents, "mu": mu})
    child_by_parent = {parent: (first, second) for parent, first, second in pairs}
    pair_denominator = {}
    for parent, first, second in pairs:
        a, c = physical_diagonal[[first, second]]
        cross = Z[first] @ Z[second]
        largest = 0.5 * (a + c + np.sqrt((a - c) ** 2 + 4 * cross * cross))
        spectral[[first, second]] = largest
        pair_denominator[parent] = largest + max(cfm[first], cfm[second])
    if any(value <= 0 or not np.isfinite(value) for value in pair_denominator.values()):
        raise ValueError("Invalid tangent spectral denominator")
    velocity = velocity.copy()
    impulse = np.zeros(n) if incoming is None else np.asarray(incoming, float).copy()
    zero_rhs = J @ velocity + rhs - Z @ (Z.T @ impulse)
    work = dict.fromkeys(
        (
            "sweeps",
            "residual_row_dots",
            "response_row_actions",
            "projection_rows",
            "changed_transactions",
            "stop_row_dots",
            "stop_row_projections",
            "root_probes",
            "majorizer_products",
        ),
        0,
    )
    work.update(
        setup_products=(n + len(pairs)) * dofs,
        tangent_setup_products=len(pairs) * dofs,
        setup_pairs=len(pairs),
        cpu_Z_Y_triangular_products=n * dofs * (dofs - 1),
        cpu_zero_rhs_products=3 * n * dofs,
        curve=[],
        physical_stop=False,
        first_stop=None,
        exact_unchanged=False,
        allowance=iterations,
    )
    for iteration in range(iterations):
        changed = False
        for row in np.flatnonzero(types != 2):
            old = impulse[row]
            residual = J[row] @ velocity + rhs[row]
            value = max(0.0, old - residual / diagonal[row])
            work["residual_row_dots"] += 1
            work["projection_rows"] += 1
            delta = value - old
            if delta != 0:
                velocity += Y[row] * delta
                work["response_row_actions"] += 1
                work["changed_transactions"] += 1
                changed = True
            impulse[row] = value
            child = child_by_parent.get(row)
            if child is None:
                continue
            take = np.asarray(child)
            tangent_residual = J[take] @ velocity + rhs[take]
            trial = impulse[take] - tangent_residual / pair_denominator[row]
            radius = max(0.0, mu[child[0]] * value)
            length = np.linalg.norm(trial)
            proposed = trial * min(1.0, radius / length) if length > 0 else trial
            tangent_delta = proposed - impulse[take]
            work["residual_row_dots"] += 2
            work["projection_rows"] += 2
            if np.any(tangent_delta != 0):
                velocity += Y[take].T @ tangent_delta
                work["response_row_actions"] += 2
                work["changed_transactions"] += 1
                changed = True
            impulse[take] = proposed
        work["sweeps"] += 1
        residual = J @ velocity + rhs
        work["stop_row_dots"] += n
        work["stop_row_projections"] += n
        score = _score(residual, impulse, zero_rhs, physical_diagonal, spectral, types, pairs, mu)
        work["curve"].append({"sweep": iteration + 1, "physical": score})
        if score["stop"] and work["first_stop"] is None:
            work["first_stop"] = iteration + 1
        if early_stop and score["stop"]:
            work["physical_stop"] = True
            break
        if not changed:
            work["exact_unchanged"] = True
            break
    if not work["curve"]:
        residual = J @ velocity + rhs
        score = _score(residual, impulse, zero_rhs, physical_diagonal, spectral, types, pairs, mu)
        work["offline_zero_budget_row_dots"] = n
    work["final_physical"] = score
    work["unused_allowance"] = iterations - work["sweeps"]
    work["residual_products"] = dofs * work["residual_row_dots"]
    work["response_products"] = dofs * work["response_row_actions"]
    work["stop_products"] = dofs * work["stop_row_dots"]
    return block.original.Result(velocity, impulse, work)

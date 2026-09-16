# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU normal-LCP elimination, one tangent action and full-merit latch.

This is an unqualified CPU control, not a native hook. Physical Gram columns
are built only for entered normal IDs. CFM never enters those columns or the
normal solve. Failure discards the uncommitted trial and calls the pinned
original recurrence from CURRENT impulses with the REMAINING outer allowance.
"""

from collections import Counter

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench import spectral_jacobi_control as spectral
from tools.fpgs_bench.active_wrench_control import RANK_TOL
from tools.fpgs_bench.coulomb_block_hybrid import parallel_continue, reference


class NormalFailure(RuntimeError):
    """Abort only the current uncommitted normal trial, preserving its costs."""


def _transition(work, count, start, kind):
    """Bound active-set transitions by 2m per normal subproblem, not per call."""
    if work["pivot_transitions"] - start >= 2 * count:
        raise NormalFailure("pivot_guard")
    work["pivot_transitions"] += 1
    work["pivot_enters" if kind == "enter" else "pivot_drops"] += 1


def _column(row, context):
    """Cache a complete physical column once per entered row and solve call."""
    cache, z, work = context["columns"], context["Z"], context["work"]
    if row not in cache:
        cache[row] = z @ z[row]
        work["column_builds"] += 1
        work["column_products"] += z.size
    else:
        work["column_hits"] += 1
    return cache[row]


def _factor(ids, context):
    """Factor the scaled physical SPD working set with inherited rank cutoff."""
    work = context["work"]
    key = tuple(ids)
    if key == context["factor_ids"]:
        work["factor_reuses"] += 1
        return context["factor"]
    columns = np.column_stack([_column(row, context)[list(ids)] for row in ids])
    size = len(ids)
    scale = np.sqrt(np.diag(columns))
    normalized = columns / scale[:, None] / scale[None, :]
    work["factor_builds"] += 1
    work["factor_mask_changes"] += int(context["factor_ids"] is not None)
    work["factor_divisions"] += 2 * size * size
    work["factor_square_roots"] += size
    lower = np.zeros((size, size))
    for row in range(size):
        for col in range(row + 1):
            value = normalized[row, col] - lower[row, :col] @ lower[col, :col]
            work["factor_products"] += col
            if row == col:
                if not np.isfinite(value) or value <= RANK_TOL:
                    work["rank_rejections"] += 1
                    raise NormalFailure("rank")
                lower[row, col] = np.sqrt(value)
                work["factor_square_roots"] += 1
            else:
                lower[row, col] = value / lower[col, col]
                work["factor_divisions"] += 1
    context["factor_ids"], context["factor"] = key, (lower, scale)
    return lower, scale


def _equality_direction(ids, residual, context):
    """Solve two tiny triangular systems without a dense inverse or new CFM."""
    lower, scale = _factor(ids, context)
    value = -residual / scale
    size, work = len(ids), context["work"]
    for row in range(size):
        value[row] = (value[row] - lower[row, :row] @ value[:row]) / lower[row, row]
        work["solve_products"] += row
    for row in range(size - 1, -1, -1):
        value[row] = (value[row] - lower[row + 1 :, row] @ value[row + 1 :]) / lower[row, row]
        work["solve_products"] += size - row - 1
    work["solve_divisions"] += 4 * size
    work["equality_solves"] += 1
    return value / scale


def _certificate(normal, residual, context):
    """Check ALL normal and limit rows using the inherited physical gates."""
    diagonal, work = context["diagonal"][context["normal_rows"]], context["work"]
    work["normal_rechecks"] += 1
    work["normal_recheck_rows"] += len(normal)
    projected = np.maximum(0, normal - residual / diagonal)
    natural = float(np.max(np.abs(normal - projected) * np.sqrt(diagonal), initial=0)) / context["cold_scale"]
    negative = float(np.max(np.maximum(-residual, 0), initial=0))
    complementarity = float(np.max(np.abs(normal * residual), initial=0))
    work["normal_check_products"] += 2 * len(normal)
    return bool(
        np.isfinite(normal).all()
        and np.isfinite(residual).all()
        and np.all(normal >= 0)
        and natural <= 1e-5
        and negative <= 3e-5
        and complementarity <= 3e-5
    )


def _normal_step(impulse, residual, context):
    """Use a primal active set; return a complete feasible normal endpoint."""
    normal_rows, work = context["normal_rows"], context["work"]
    normal, gradient = impulse[normal_rows].copy(), residual[normal_rows].copy()
    local_of = {int(row): index for index, row in enumerate(normal_rows)}
    free = sorted(int(row) for row in normal_rows if impulse[row] > 0 or residual[row] < 0)
    start = work["pivot_transitions"]
    dropped = []
    while True:
        if free:
            selected = np.array([local_of[row] for row in free])
            direction = _equality_direction(free, gradient[selected], context)
            if not np.isfinite(direction).all():
                raise NormalFailure("nonfinite_normal")
            negative = np.flatnonzero(direction < 0)
            boundary = [(float(normal[selected[index]] / -direction[index]), free[index], index) for index in negative]
            hit = min(boundary) if boundary else (np.inf, -1, -1)
            step = min(1.0, hit[0])
            before = normal.copy()
            normal[selected] += step * direction
            dropping = step < 1.0 or (hit[0] == 1.0 and normal[selected[hit[2]]] <= 0)
            if dropping:
                normal[local_of[hit[1]]] = 0.0
            for row in free:
                change = normal[local_of[row]] - before[local_of[row]]
                if change != 0:
                    gradient += _column(row, context)[normal_rows] * change
                    work["normal_inner_products"] += len(normal_rows)
            if dropping:
                _transition(work, len(normal_rows), start, "drop")
                free.remove(hit[1])
                dropped.append(hit[1])
                continue
        if _certificate(normal, gradient, context):
            break
        omitted = [int(row) for row in normal_rows if row not in free and gradient[local_of[int(row)]] < 0]
        if not omitted:
            raise NormalFailure("normal_certificate")
        _transition(work, len(normal_rows), start, "enter")
        free.append(min(omitted))
        free.sort()
    proposed, next_residual = impulse.copy(), residual.copy()
    proposed[normal_rows] = normal
    changed = []
    for local, row in enumerate(normal_rows):
        delta = normal[local] - impulse[row]
        if delta != 0:
            next_residual += _column(int(row), context) * delta
            work["normal_publish_products"] += len(impulse)
            changed.append(int(row))
    work["normal_solves"] += 1
    return proposed, next_residual, {"active_ids": free, "dropped_ids": dropped, "changed_normal_ids": changed}


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, early_stop=True):
    """Execute at most the original allowance, including literal late fallback."""
    J, L, diagonal, rhs, mu, vhat = (np.asarray(value, float) for value in (J, L, diagonal, rhs, mu, vhat))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    count, dofs = J.shape
    if iterations < 0 or count > 64 or not np.isin(types, (0, 2, 3)).all():
        raise ValueError("Unsupported original allowance or row layout")
    z = np.linalg.solve(L, J.T).T
    physical_diagonal = np.sum(z * z, axis=1)
    if (
        not np.isfinite(z).all()
        or not np.isfinite(diagonal).all()
        or np.any(diagonal <= 0)
        or np.any(physical_diagonal <= 0)
    ):
        raise ValueError("Unsupported nonpositive/nonfinite physical operator")
    pairs = reference.pairs_of({"kind": types, "parent": parents, "mu": mu})
    if {row for _, first, second in pairs for row in (first, second)} != set(np.flatnonzero(types == 2)):
        raise ValueError("Noncanonical tangent layout")
    tangent_cache = []
    for normal, first, second in pairs:
        if not np.isfinite(mu[first]) or mu[first] < 0 or mu[first] != mu[second]:
            raise ValueError("Invalid Coulomb coefficient")
        a, c = physical_diagonal[[first, second]]
        cross = z[first] @ z[second]
        denominator = 0.5 * (a + c) + np.hypot(0.5 * (a - c), cross)
        denominator += max(diagonal[first] - a, diagonal[second] - c)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid tangent spectral denominator")
        tangent_cache.append((normal, first, second, denominator))
    entry = np.zeros(count) if incoming is None else np.asarray(incoming, float).copy()
    if not np.isfinite(entry).all() or np.any(entry[types != 2] < 0):
        raise ValueError("Invalid incoming impulse")
    impulse = entry.copy()
    residual = J @ vhat + rhs
    initial_residual = residual.copy()
    zero_rhs = residual.copy()
    work = Counter()
    for key in (
        "sweeps",
        "candidate_sweeps",
        "fallback_sweeps",
        "column_builds",
        "pivot_enters",
        "pivot_drops",
        "pivot_transitions",
        "factor_reuses",
        "rank_rejections",
        "fallback_majorizer_products",
    ):
        work[key] = 0
    work.update(
        setup_products=(count + len(pairs)) * dofs,
        setup_cross_products=len(pairs) * dofs,
        initial_rhs_products=count * dofs,
        cpu_Z_triangular_products=count * dofs * (dofs - 1) // 2,
        allowance=iterations,
    )
    cold_scale = 1 + float(np.max(np.abs(residual) / np.sqrt(diagonal), initial=0))
    context = {
        "Z": z,
        "work": work,
        "columns": {},
        "factor_ids": None,
        "factor": None,
        "normal_rows": np.flatnonzero(types != 2),
        "diagonal": diagonal,
        "cold_scale": cold_scale,
    }

    def score_state(value, response):
        """Charge every full physical score, including initial/midpoint/fallback."""
        work["full_scores"] += 1
        work["full_score_row_visits"] += count
        work["stop_row_projections"] += count
        work["full_score_products"] += 2 * len(context["normal_rows"]) + 14 * len(pairs)
        work["full_score_divisions"] += len(context["normal_rows"]) + 3 * len(pairs) + 1
        work["full_score_square_roots"] += count + 3 * len(pairs)
        return spectral.physical_score(response, value, diagonal, types, parents, mu, cold_scale)

    score = score_state(impulse, residual)
    energy, relaxation = spectral.merit(score), 1.0
    curve, reason, first_stop, latch_step = [], None, None, None
    if np.any(entry != 0):
        zero_rhs -= z @ (z.T @ entry)
        work["incoming_rhs_products"] += 2 * count * dofs
        reason = "nonzero_incoming"
    for iteration in range(iterations):
        if reason is not None:
            break
        before_work = dict(work)
        work["normal_trial_attempts"] += 1
        try:
            proposal, proposal_residual, detail = _normal_step(impulse, residual, context)
        except NormalFailure as failure:
            reason = str(failure)
            work["failed_trials"] += 1
            break
        tangent_delta = np.zeros(count)
        for normal, first, second, denominator in tangent_cache:
            take = [first, second]
            trial = impulse[take] - proposal_residual[take] / denominator
            radius = max(0.0, mu[first] * proposal[normal])
            magnitude = np.linalg.norm(trial)
            value = trial * min(1.0, radius / magnitude) if magnitude > 0 else trial
            if radius == 0:
                value[:] = 0
            proposal[take] = value
            tangent_delta[take] = value - impulse[take]
            work["tangent_disk_projections"] += 1
            work["tangent_divisions"] += 2
        changed = np.flatnonzero(tangent_delta != 0)
        if len(changed):
            kinetic = z[changed].T @ tangent_delta[changed]
            proposal_residual += z @ kinetic
            work["tangent_transpose_products"] += len(changed) * dofs
            work["tangent_residual_products"] += count * dofs
            work["tangent_action_pairs"] += 1
        work["changed_tangent_rows"] += len(changed)
        next_impulse = impulse + relaxation * (proposal - impulse)
        next_residual = residual + relaxation * (proposal_residual - residual)
        work["relaxation_products"] += 2 * count
        next_score = score_state(next_impulse, next_residual)
        next_energy = spectral.merit(next_score)
        trial_energy = next_energy
        if relaxation == 1.0 and next_energy > energy:
            relaxation, latch_step = 0.5, iteration + 1
            next_impulse = 0.5 * (impulse + next_impulse)
            next_residual = 0.5 * (residual + next_residual)
            next_score = score_state(next_impulse, next_residual)
            next_energy = spectral.merit(next_score)
            work["midpoint_full_scores"] += 1
            work["midpoint_values"] += 2 * count
        work["candidate_sweeps"] += 1
        work["sweeps"] += 1
        curve.append(
            dict(
                sweep=iteration + 1,
                current_merit=energy,
                proposed_merit=trial_energy,
                accepted_merit=next_energy,
                relaxation=relaxation,
                physical=next_score,
                **detail,
                work_delta={key: value - before_work.get(key, 0) for key, value in work.items()},
            )
        )
        impulse, residual, score, energy = next_impulse, next_residual, next_score, next_energy
        if score["stop"] and first_stop is None:
            first_stop = iteration + 1
        if early_stop and score["stop"]:
            break
    fallback_entry = None
    if reason is not None:
        remaining = iterations - work["candidate_sweeps"]
        fallback_entry = impulse.tolist()
        continued = parallel_continue(z, diagonal, zero_rhs, types, parents, mu, impulse, iterations=remaining)
        work["fallback_allowance"] = remaining
        work["fallback_sweeps"] = continued.work["sweeps"]
        work["sweeps"] += continued.work["sweeps"]
        for key, value in continued.work.items():
            work["fallback_" + key] = value
        impulse = continued.impulses
        # Retain the original continuation's publication work above; the CPU
        # adapter independently reconstructs total committed momentum below.
    delta = impulse - entry
    changed = np.flatnonzero(delta != 0)
    kinetic_delta = z[changed].T @ delta[changed] if len(changed) else np.zeros(dofs)
    work["final_transpose_products"] = len(changed) * dofs
    reconstructed_residual = initial_residual + z @ kinetic_delta
    if reason is not None:
        residual = reconstructed_residual
        score = score_state(impulse, residual)
        work["fallback_final_residual_products"] = count * dofs
    else:
        work["diagnostic_residual_products"] = count * dofs
    dict.update(
        work,
        curve=curve,
        column_ids=list(context["columns"]),
        fallback=reason is not None,
        fallback_reason=reason,
        fallback_entry_impulse=fallback_entry,
        first_stop=first_stop,
        latch_step=latch_step,
        relaxation=relaxation,
        physical_stop=bool(score["stop"]),
        final_physical=score,
        residual_carry_error=float(np.max(np.abs(reconstructed_residual - residual), initial=0)),
        unused_allowance=iterations - work["sweeps"],
        cpu_final_backsolve_products=dofs * (dofs - 1) // 2,
    )
    assert work["sweeps"] <= iterations
    return block.original.Result(vhat + np.linalg.solve(L.T, kinetic_delta), impulse, dict(work))

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only analytic Coulomb Newton/GMRES with feasible-chord globalization."""

from collections import Counter

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench.coulomb_block_hybrid import parallel_continue, reference

RANK_TOL = 64 * np.finfo(np.float32).eps


class KrylovFailure(RuntimeError):
    """Reject an uncommitted direction without discarding its measured work."""


def _prepare(J, L, diagonal, rhs, types, parents, mu, vhat):
    """Keep physical operator and denominator CFM distinct throughout a call."""
    J, L, diagonal, rhs, mu, vhat = (np.asarray(a, float) for a in (J, L, diagonal, rhs, mu, vhat))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    z = np.linalg.solve(L, J.T).T
    count, dofs = z.shape
    physical = np.sum(z * z, axis=1)
    if count > 64 or not np.isin(types, (0, 2, 3)).all() or not np.isfinite(z).all():
        raise ValueError("Unsupported original row layout/operator")
    if np.any(physical <= 0) or np.any(diagonal <= 0) or not np.isfinite(diagonal).all():
        raise ValueError("Unsupported response denominator")
    pairs = reference.pairs_of({"kind": types, "parent": parents, "mu": mu})
    if {row for _, first, second in pairs for row in (first, second)} != set(np.flatnonzero(types == 2)):
        raise ValueError("Noncanonical tangent rows")
    eta = 1 / diagonal
    spectral = {}
    paired = {int(normal): (int(first), int(second)) for normal, first, second in pairs}
    groups = []
    for normal in np.flatnonzero(types != 2):
        ids = [int(normal)]
        if int(normal) in paired:
            first, second = paired[int(normal)]
            if not np.isfinite(mu[first]) or mu[first] < 0 or mu[first] != mu[second]:
                raise ValueError("Invalid Coulomb coefficient")
            a, c = physical[[first, second]]
            cross = z[first] @ z[second]
            largest = 0.5 * (a + c) + np.hypot(0.5 * (a - c), cross)
            denominator = largest + max(diagonal[first] - a, diagonal[second] - c)
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError("Invalid tangent step")
            eta[[first, second]] = 1 / denominator
            spectral[int(normal)] = largest
            ids.extend((first, second))
        groups.append(np.asarray(ids, int))
    work = Counter()
    for key in ("sweeps", "candidate_sweeps", "fallback_sweeps", "krylov_steps", "fresh_checks"):
        work[key] = 0
    work.update(
        setup_products=(count + len(pairs)) * dofs,
        initial_rhs_products=count * dofs,
        cpu_Z_triangular_products=count * dofs * (dofs - 1) // 2,
    )
    return {
        "J": J,
        "L": L,
        "Z": z,
        "diagonal": diagonal,
        "physical": physical,
        "rhs": rhs,
        "types": types,
        "parents": parents,
        "mu": mu,
        "vhat": vhat,
        "eta": eta,
        "spectral": spectral,
        "groups": groups,
        "work": work,
        "gram_ids": None,
        "gram": None,
    }


def _retract(value, context):
    """Return a cone-feasible endpoint, not a Euclidean cone projection."""
    result = value.copy()
    for ids in context["groups"]:
        result[ids[0]] = max(0.0, result[ids[0]])
        if len(ids) == 3:
            radius = context["mu"][ids[1]] * result[ids[0]]
            length = np.linalg.norm(result[ids[1:]])
            if radius <= 0:
                result[ids[1:]] = 0
            elif length > radius:
                result[ids[1:]] *= radius / length
            context["work"]["retraction_disks"] += 1
    context["work"]["retraction_rows"] += len(value)
    return result


def _map(x, residual, context):
    """Evaluate the inherited complete map and cache its current local modes."""
    eta, work = context["eta"], context["work"]
    q = x - eta * residual
    pg = q.copy()
    modes, active = [], []
    for ids in context["groups"]:
        normal = ids[0]
        projection = np.zeros((len(ids), len(ids)))
        pg[normal] = max(q[normal], 0.0)
        projection[0, 0] = float(q[normal] > 0)
        if len(ids) == 3:
            take = ids[1:]
            radius = context["mu"][take[0]] * pg[normal]
            length = np.linalg.norm(q[take])
            if radius <= 0:
                pg[take] = 0
            elif length <= radius:
                projection[1:, 1:] = np.eye(2)
            else:
                unit = q[take] / length
                pg[take] = radius * unit
                projection[1:, 1:] = (radius / length) * (np.eye(2) - np.outer(unit, unit))
                projection[1:, 0] = context["mu"][take[0]] * unit * projection[0, 0]
                work["mode_slide_products"] += 10
            work["map_disks"] += 1
        modes.append(projection)
        active.extend(int(ids[index]) for index in range(len(ids)) if np.any(projection[index] != 0))
    work["map_calls"] += 1
    work["map_rows"] += len(x)
    work["map_products"] += len(x) + 5 * len(context["spectral"])
    work["map_divisions"] += len(x)
    return (x - pg) / eta, pg, modes, np.asarray(sorted(active), int)


def _jvp(vector, response, modes, context):
    """Apply the full nonsymmetric derivative, including moving normal radii."""
    dq = vector - context["eta"] * response
    projected = np.zeros(len(vector))
    for ids, mode in zip(context["groups"], modes, strict=True):
        projected[ids] = mode @ dq[ids]
        context["work"]["derivative_products"] += len(ids) ** 2
    context["work"]["derivative_products"] += len(vector)
    context["work"]["derivative_divisions"] += len(vector)
    return (vector - projected) / context["eta"]


def _lu(matrix, work):
    """Factor a local linear block with the inherited scaled pivot guard."""
    size = len(matrix)
    scale = np.max(np.abs(matrix), axis=1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise KrylovFailure("precondition_rank")
    lower_upper = matrix / scale[:, None]
    permutation = np.arange(size)
    work["precondition_build_divisions"] += size * size
    work["precondition_builds"] += 1
    for col in range(size):
        pivot = col + int(np.argmax(np.abs(lower_upper[col:, col])))
        if abs(lower_upper[pivot, col]) <= RANK_TOL or not np.isfinite(lower_upper[pivot, col]):
            raise KrylovFailure("precondition_rank")
        lower_upper[[col, pivot]] = lower_upper[[pivot, col]]
        permutation[[col, pivot]] = permutation[[pivot, col]]
        for row in range(col + 1, size):
            lower_upper[row, col] /= lower_upper[col, col]
            lower_upper[row, col + 1 :] -= lower_upper[row, col] * lower_upper[col, col + 1 :]
            work["precondition_build_divisions"] += 1
            work["precondition_build_products"] += size - col - 1
    return lower_upper, scale, permutation


def _precondition(vector, blocks, work):
    """Apply only the current right preconditioner, never a physical shift."""
    result = np.zeros_like(vector)
    for indices, (matrix, scale, permutation) in blocks:
        value = (vector[indices] / scale)[permutation].copy()
        size = len(indices)
        for row in range(size):
            value[row] -= matrix[row, :row] @ value[:row]
            work["precondition_apply_products"] += row
        for row in range(size - 1, -1, -1):
            value[row] = (value[row] - matrix[row, row + 1 :] @ value[row + 1 :]) / matrix[row, row]
            work["precondition_apply_products"] += size - row - 1
        work["precondition_apply_divisions"] += 2 * size
        work["precondition_applies"] += 1
        result[indices] = value
    return result


def _direction(x, residual, f, pg, modes, active, initial_norm, context):
    """Solve the shifted active equation with fixed bounded right-GMRES/CGS2."""
    work, z = context["work"], context["Z"]
    size, count = len(active), len(x)
    if size > 18:
        raise KrylovFailure("active_capacity")
    key = tuple(map(int, active))
    if key != context["gram_ids"]:
        context["gram"] = z[active] @ z[active].T
        context["gram_ids"] = key
        work["gram_builds"] += 1
        work["gram_products"] += size * size * z.shape[1]
    else:
        work["gram_reuses"] += 1
    gram = context["gram"]
    delta = pg - x
    delta[active] = 0
    shift = np.zeros(count)
    if np.any(delta != 0):
        shift = z @ (z.T @ delta)
        work["inactive_coupling_products"] += 2 * z.size
        work["inactive_couplings"] += 1
    rhs = -f[active] - _jvp(delta, shift, modes, context)[active]
    full_norm = float(np.linalg.norm(f))
    work["control_norm_products"] += count
    work["control_norm_reductions"] += 1
    work["control_square_roots"] += 2
    if full_norm == 0 or initial_norm <= 0:
        raise KrylovFailure("zero_map_without_physical_stop")
    theta = min(0.5, np.sqrt(full_norm / initial_norm))
    target = theta * full_norm
    lookup = {int(row): position for position, row in enumerate(active)}
    blocks = []
    for ids, mode in zip(context["groups"], modes, strict=True):
        local = [index for index, row in enumerate(ids) if int(row) in lookup]
        if not local:
            continue
        indices = np.array([lookup[int(ids[index])] for index in local])
        local_eta = context["eta"][ids[local]]
        projection = mode[np.ix_(local, local)]
        dimension = len(indices)
        block_gram = gram[np.ix_(indices, indices)]
        matrix = (np.eye(dimension) - projection @ (np.eye(dimension) - local_eta[:, None] * block_gram)) / local_eta[
            :, None
        ]
        work["precondition_build_products"] += dimension**3 + dimension**2
        work["precondition_build_divisions"] += dimension**2
        blocks.append((indices, _lu(matrix, work)))

    def action(value, *, certificate=False):
        """Use the cached physical Gram, not a dense Jacobian or new Z pair."""
        vector, response = np.zeros(count), np.zeros(count)
        vector[active], response[active] = value, gram @ value
        work["true_linear_products" if certificate else "krylov_gram_products"] += size * size
        return _jvp(vector, response, modes, context)[active]

    cap = min(size, 8)
    beta = float(np.linalg.norm(rhs))
    work["control_norm_products"] += size
    work["control_norm_reductions"] += 1
    work["control_square_roots"] += 1
    solution = np.zeros(size)
    steps = 0
    if beta > target:
        basis = np.zeros((size, cap + 1))
        right = np.zeros((size, cap))
        hessenberg = np.zeros((cap + 1, cap))
        cosine, sine = np.zeros(cap), np.zeros(cap)
        transformed = np.zeros(cap + 1)
        transformed[0], basis[:, 0] = beta, rhs / beta
        work["orthogonal_divisions"] += size
        for col in range(cap):
            right[:, col] = _precondition(basis[:, col], blocks, work)
            value = action(right[:, col])
            original_length = np.linalg.norm(value)
            work["krylov_norm_products"] += size
            work["krylov_norm_reductions"] += 1
            work["krylov_square_roots"] += 2
            for _ in range(2):
                coefficient = basis[:, : col + 1].T @ value
                hessenberg[: col + 1, col] += coefficient
                value -= basis[:, : col + 1] @ coefficient
                work["orthogonal_products"] += 2 * size * (col + 1)
                work["orthogonal_reductions"] += col + 1
            length = np.linalg.norm(value)
            work["orthogonal_products"] += size
            work["orthogonal_reductions"] += 1
            work["krylov_steps"] += 1
            work["krylov_q_squared"] += 2 * col + 1
            steps = col + 1
            hessenberg[col + 1, col] = length
            breakdown = length <= RANK_TOL * original_length
            if not breakdown:
                basis[:, col + 1] = value / length
                work["orthogonal_divisions"] += size
            for row in range(col):
                a, b = hessenberg[row : row + 2, col]
                hessenberg[row, col] = cosine[row] * a + sine[row] * b
                hessenberg[row + 1, col] = -sine[row] * a + cosine[row] * b
                work["givens_products"] += 4
            radius = np.hypot(hessenberg[col, col], hessenberg[col + 1, col])
            if not np.isfinite(radius) or radius == 0:
                raise KrylovFailure("linear_breakdown")
            cosine[col], sine[col] = hessenberg[col, col] / radius, hessenberg[col + 1, col] / radius
            hessenberg[col, col], hessenberg[col + 1, col] = radius, 0
            transformed[col + 1] = -sine[col] * transformed[col]
            transformed[col] *= cosine[col]
            work["givens_products"] += 2
            work["givens_divisions"] += 2
            work["givens_square_roots"] += 1
            if abs(transformed[col + 1]) <= target:
                break
            if breakdown:
                raise KrylovFailure("linear_breakdown")
        coefficient = transformed[:steps].copy()
        for row in range(steps - 1, -1, -1):
            diagonal = hessenberg[row, row]
            if diagonal == 0 or not np.isfinite(diagonal):
                raise KrylovFailure("hessenberg_rank")
            coefficient[row] = (coefficient[row] - hessenberg[row, row + 1 : steps] @ coefficient[row + 1 :]) / diagonal
            work["hessenberg_products"] += steps - row - 1
            work["hessenberg_divisions"] += 1
        solution = right[:, :steps] @ coefficient
        work["direction_products"] += size * steps
    linear_error = float(np.linalg.norm(action(solution, certificate=True) - rhs))
    work["control_norm_products"] += size
    work["control_norm_reductions"] += 1
    work["control_square_roots"] += 1
    work["true_linear_checks"] += 1
    if not np.isfinite(linear_error) or linear_error > target:
        raise KrylovFailure("linear_forcing")
    delta[active] = solution
    if not np.isfinite(delta).all():
        raise KrylovFailure("nonfinite_direction")
    return delta, {
        "active_ids": list(key),
        "krylov_steps": steps,
        "forcing_target": target,
        "forcing_theta": theta,
        "full_map_norm": full_norm,
        "active_rhs_norm": beta,
        "true_linear_residual": linear_error,
    }


def _physical(x, residual, context):
    """Literal computed physical gates from the prior active-dual owner."""
    work = context["work"]
    scale = max(
        1.0, np.max(np.abs(context["zero_rhs"]), initial=0), np.max(np.abs(residual - context["zero_rhs"]), initial=0)
    )
    natural = normal = complementarity = mdp = cone = negative = 0.0
    for ids in context["groups"]:
        row = ids[0]
        diagonal = context["physical"][row]
        defect = diagonal * (x[row] - max(0.0, x[row] - residual[row] / diagonal))
        natural = max(natural, abs(defect) / scale)
        normal = max(normal, -residual[row])
        complementarity = max(complementarity, abs(x[row] * residual[row]))
        negative = max(negative, -x[row])
        if len(ids) == 3:
            take = ids[1:]
            radius = max(0.0, context["mu"][take[0]] * x[row])
            spectral = context["spectral"][int(row)]
            trial = x[take] - residual[take] / spectral
            length = np.linalg.norm(trial)
            projected = trial * min(1.0, radius / length) if length > 0 else trial
            natural = max(natural, spectral * np.max(np.abs(x[take] - projected)) / scale)
            mdp = max(mdp, x[take] @ residual[take] + radius * np.linalg.norm(residual[take]))
            cone = max(cone, np.linalg.norm(x[take]) - radius)
    work["full_scores"] += 1
    work["full_score_rows"] += len(x)
    work["full_score_products"] += 2 * len(context["groups"]) + 13 * len(context["spectral"])
    work["full_score_square_roots"] += 3 * len(context["spectral"])
    work["full_score_divisions"] += 2 * len(context["groups"]) + 4 * len(context["spectral"])
    score = {
        "natural": float(natural),
        "normal": float(normal),
        "complementarity": float(complementarity),
        "mdp": float(mdp),
        "cone": float(cone),
        "negative": float(negative),
    }
    score["stop"] = bool(
        np.isfinite(x).all()
        and np.isfinite(residual).all()
        and all(np.isfinite(v) for v in score.values())
        and max(natural, normal, complementarity, mdp, cone) < 3e-5
        and negative < 1e-7
    )
    return score


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, early_stop=True):
    """Bound committed corrections and preserve all failed linear/line costs."""
    if iterations < 0:
        raise ValueError("Negative original allowance")
    context = _prepare(J, L, diagonal, rhs, types, parents, mu, vhat)
    z, work = context["Z"], context["work"]
    count, dofs = z.shape
    entry = np.zeros(count) if incoming is None else np.asarray(incoming, float).copy()
    if not np.isfinite(entry).all():
        raise ValueError("Nonfinite incoming state")
    x = entry.copy()
    initial_residual = context["J"] @ context["vhat"] + context["rhs"]
    if not np.isfinite(initial_residual).all():
        raise ValueError("Unsupported nonfinite supplied residual")
    residual = initial_residual.copy()
    zero_rhs = initial_residual.copy()
    reason = None
    if np.any(entry != 0):
        zero_rhs -= z @ (z.T @ entry)
        work["incoming_rhs_products"] += 2 * z.size
        reason = "nonzero_incoming"
    context["zero_rhs"] = zero_rhs
    f, pg, mode, active = _map(x, residual, context)
    initial_norm = float(np.linalg.norm(f))
    work["control_norm_products"] += count
    work["control_norm_reductions"] += 1
    work["control_square_roots"] += 1
    score = _physical(x, residual, context)
    curve, first_stop, fallback_entry = [], None, None
    fresh_version = -1
    kinetic_delta = np.zeros(dofs)

    def fresh():
        """Reconstruct the complete physical state; never trust a false stop."""
        nonlocal residual, f, pg, mode, active, score, fresh_version, kinetic_delta
        kinetic_delta = z.T @ (x - entry)
        residual = initial_residual + z @ kinetic_delta
        work["fresh_checks"] += 1
        work["fresh_operator_products"] += 2 * z.size
        f, pg, mode, active = _map(x, residual, context)
        score = _physical(x, residual, context)
        fresh_version = work["sweeps"]

    while work["candidate_sweeps"] < iterations and reason is None:
        if early_stop and score["stop"]:
            fresh()
            if score["stop"]:
                first_stop = work["candidate_sweeps"]
                break
            work["false_carried_stops"] += 1
        before = dict(work)
        try:
            delta, detail = _direction(x, residual, f, pg, mode, active, initial_norm, context)
        except KrylovFailure as failure:
            reason = str(failure)
            work["failed_directions"] += 1
            break
        endpoint = _retract(x + delta, context)
        direction = endpoint - x
        if not np.isfinite(direction).all() or not np.any(direction != 0):
            reason = "zero_or_nonfinite_chord"
            break
        response = z @ (z.T @ direction)
        work["chord_operator_products"] += 2 * z.size
        work["chord_directions"] += 1
        old_merit = float(np.linalg.norm(f))
        work["control_norm_products"] += count
        work["control_norm_reductions"] += 1
        work["control_square_roots"] += 1
        accepted = False
        for trial in range(8):
            alpha = 2.0**-trial
            next_x, next_residual = x + alpha * direction, residual + alpha * response
            next_f, next_pg, next_mode, next_active = _map(next_x, next_residual, context)
            merit = float(np.linalg.norm(next_f))
            work["line_trials"] += 1
            work["line_affine_products"] += 2 * count
            work["line_merit_products"] += count
            work["line_merit_reductions"] += 1
            work["line_merit_square_roots"] += 1
            if np.isfinite(merit) and merit < (1 - 1e-4 * alpha) * old_merit:
                accepted = True
                break
            work["rejected_trials"] += 1
        if not accepted:
            reason = "line_search"
            break
        x, residual, f, pg, mode, active = next_x, next_residual, next_f, next_pg, next_mode, next_active
        score = _physical(x, residual, context)
        work["candidate_sweeps"] += 1
        work["sweeps"] += 1
        curve.append(
            {
                "sweep": work["candidate_sweeps"],
                "alpha": alpha,
                "trial_count": trial + 1,
                "old_merit": old_merit,
                "accepted_merit": merit,
                "physical": score,
                **detail,
                "work_delta": {key: value - before.get(key, 0) for key, value in work.items()},
            }
        )
    if reason is not None:
        remaining = iterations - work["candidate_sweeps"]
        fallback_entry = x.tolist()
        continued = parallel_continue(
            z,
            context["diagonal"],
            zero_rhs,
            context["types"],
            context["parents"],
            context["mu"],
            x,
            iterations=remaining,
        )
        x = continued.impulses
        work["fallback_allowance"] = remaining
        work["fallback_sweeps"] = continued.work["sweeps"]
        work["sweeps"] += continued.work["sweeps"]
        for key, value in continued.work.items():
            work["fallback_" + key] = value
        fresh_version = -1
    if fresh_version != work["sweeps"]:
        fresh()
    dict.update(
        work,
        curve=curve,
        allowance=iterations,
        first_stop=first_stop,
        physical_stop=bool(score["stop"]),
        final_physical=score,
        fallback=reason is not None,
        fallback_reason=reason,
        fallback_entry_impulse=fallback_entry,
        unused_allowance=iterations - work["sweeps"],
        cpu_final_backsolve_products=dofs * (dofs - 1) // 2,
    )
    assert work["sweeps"] <= iterations
    return block.original.Result(context["vhat"] + np.linalg.solve(context["L"].T, kinetic_delta), x, dict(work))

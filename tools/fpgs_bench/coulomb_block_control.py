# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU control of ordered full Coulomb blocks, not a native performance claim."""

import hashlib
import json
from pathlib import Path
from types import FunctionType

import numpy as np

from tools.fpgs_bench import active_wrench_control as original


def prepare_block(matrix):
    """Factor the fixed proximal self block without adding physical compliance."""
    matrix = np.asarray(matrix, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0:
        return None
    a, h = float(matrix[0, 0]), matrix[0, 1:].copy()
    schur = matrix[1:, 1:] - np.outer(h, h) / a
    values = np.linalg.eigvalsh(schur)
    if not np.isfinite(values).all() or values[0] <= 1e-7 * max(float(np.max(np.diag(matrix))), 1e-30):
        return None
    return a, h, schur, values


def local_block(matrix, bias, friction, *, prepared=None):
    """Solve normal plus circular friction using a safeguarded, nonmonotone root."""
    matrix, bias = np.asarray(matrix, float), np.asarray(bias, float)
    info = {"kind": "unsafe", "gamma": 0.0, "probes": 0, "reason": None}
    if not np.isfinite(bias).all() or not np.isfinite(friction) or friction < 0:
        info["reason"] = "nonfinite_input"
        return None, info
    if bias[0] >= 0:
        info["kind"] = "open"
        return np.zeros(3), info
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0:
        info["reason"] = "invalid_normal"
        return None, info
    if friction == 0:
        info["kind"] = "normal"
        return np.array([-bias[0] / matrix[0, 0], 0.0, 0.0]), info
    prepared = prepare_block(matrix) if prepared is None else prepared
    if prepared is None or prepared is False:
        info["reason"] = "unsafe_schur"
        return None, info
    a, h, schur, eigenvalues = prepared
    beta = bias[1:] - h * bias[0] / a

    def evaluate(gamma):
        shifted = schur + gamma * np.eye(2)
        tangent = -np.linalg.solve(shifted, beta)
        normal = (-bias[0] - h @ tangent) / a
        length = np.linalg.norm(tangent)
        info["probes"] += 1
        return np.r_[normal, tangent], float(length - friction * normal), shifted

    trial, value, shifted = evaluate(0.0)
    if trial[0] >= 0 and value <= 0:
        info["kind"] = "stick"
        return trial, info
    radius_min = friction * (-bias[0]) / (a + friction * np.linalg.norm(h))
    lower, upper = 0.0, max(0.0, np.linalg.norm(beta) / radius_min - eigenvalues[0])
    if not np.isfinite(upper) or upper <= 0:
        info["reason"] = "invalid_bracket"
        return None, info
    upper_trial, upper_value, _ = evaluate(upper)
    if not np.isfinite(upper_trial).all() or upper_value > 1e-12 * max(radius_min, 1.0):
        info["reason"] = "unbracketed_root"
        return None, info
    gamma = 0.0
    for _ in range(16):
        normal, tangent = trial[0], trial[1:]
        radius, length = friction * normal, np.linalg.norm(tangent)
        if normal >= 0 and radius > 0 and abs(value / radius) <= 2e-6:
            if value > 0:
                # Coupled inward repair retains the normal equation too.
                scale = friction * (-bias[0]) / (a * length + friction * (h @ tangent))
                tangent = scale * tangent
                normal = (-bias[0] - h @ tangent) / a
                trial = np.r_[normal, tangent]
            residual = matrix @ trial + bias + np.r_[0.0, gamma * trial[1:]]
            bound = 8e-6 * (1 + np.linalg.norm(bias) + (np.linalg.norm(matrix, 2) + gamma) * np.linalg.norm(trial))
            if (
                np.isfinite(trial).all()
                and trial[0] >= 0
                and np.linalg.norm(trial[1:]) <= friction * trial[0] * (1 + 4e-7)
                and np.linalg.norm(residual) <= bound
            ):
                info.update(kind="slip", gamma=float(gamma))
                return trial, info
            info["reason"] = "root_law_guard"
            return None, info
        if value > 0:
            lower = gamma
        else:
            upper = gamma
        derivative_t = -np.linalg.solve(shifted, tangent)
        derivative_n = -(h @ derivative_t) / a
        derivative = tangent @ derivative_t / max(length, 1e-300) - friction * derivative_n
        proposal = gamma - value / derivative if derivative != 0 else np.nan
        if not np.isfinite(proposal) or proposal <= lower or proposal >= upper:
            proposal = 0.5 * (lower + upper)
        gamma = float(proposal)
        trial, value, shifted = evaluate(gamma)
    info["reason"] = "root_budget"
    return None, info


def physical_from_residual(residual, impulse, diagonal, types, parents, mu, cold_scale):
    """Apply the same baseline-independent stop to both ordered methods."""
    correction = np.zeros(len(impulse))
    cone, mdp = 0.0, 0.0
    normals = np.flatnonzero(np.isin(types, (0, 3)))
    for row in normals:
        correction[row] = impulse[row] - max(0.0, impulse[row] - residual[row] / diagonal[row])
        cone = max(cone, -impulse[row])
    for row in np.flatnonzero(types == 2):
        if row != parents[row] + 1:
            continue
        pair = slice(row, row + 2)
        radius = max(0.0, mu[row] * impulse[parents[row]])
        trial = impulse[pair] - residual[pair] / max(diagonal[pair])
        projected = trial * min(1.0, radius / max(np.linalg.norm(trial), 1e-300))
        correction[pair] = impulse[pair] - projected
        cone = max(cone, np.linalg.norm(impulse[pair]) - radius)
        mdp = max(mdp, abs(impulse[pair] @ residual[pair] + radius * np.linalg.norm(residual[pair])))
    score = {
        "natural": float(np.max(np.sqrt(diagonal) * np.abs(correction), initial=0) / cold_scale),
        "normal": float(np.max(np.maximum(-residual[normals], 0), initial=0)),
        "complementarity": float(np.max(np.abs(impulse[normals] * residual[normals]), initial=0)),
        "mdp": float(mdp),
        "cone": float(cone),
    }
    score["stop"] = (
        all(np.isfinite(v) for v in score.values())
        and score["natural"] <= 1e-5
        and score["normal"] <= 3e-5
        and score["complementarity"] <= 3e-5
        and score["mdp"] <= 3e-5
        and score["cone"] <= 1e-10
    )
    return score


def solve(
    J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=8, coupled=True, early_stop=False, incoming=None
):
    """Run the original ordered allowance with full-block or original metric updates."""
    if iterations < 0:
        raise ValueError("The caller must supply its unchanged nonnegative allowance")
    J, L, diagonal, rhs, mu, vhat = (np.asarray(a, float) for a in (J, L, diagonal, rhs, mu, vhat))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    n, dofs = J.shape
    Z = np.linalg.solve(L, J.T).T
    Y = np.linalg.solve(L.T, Z.T).T
    velocity = vhat.copy()
    impulse = np.zeros(n) if incoming is None else np.asarray(incoming, float).copy()
    work = dict.fromkeys(
        (
            "sweeps",
            "row_transactions",
            "residual_row_dots",
            "response_row_actions",
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
            "legacy_sliding_roots",
            "legacy_cpu_gram_products",
            "stop_row_dots",
        ),
        0,
    )
    work.update(curve=[], first_stop=None, coupled=bool(coupled), producer_rows=n)
    cold = J @ vhat + rhs
    cold_scale = 1 + float(np.max(np.abs(cold) / np.sqrt(diagonal), initial=0))
    work["cold_scale_row_dots"] = n
    cache = {}
    for iteration in range(iterations):
        row, changed = 0, False
        while row < n:
            work["row_transactions"] += 1
            valid = (
                types[row] == 0
                and row + 2 < n
                and np.all(types[row + 1 : row + 3] == 2)
                and np.all(parents[row + 1 : row + 3] == row)
                and np.isfinite(mu[row + 1])
                and mu[row + 1] >= 0
                and mu[row + 1] == mu[row + 2]
            )
            if valid:
                take = slice(row, row + 3)
                old = impulse[take].copy()
                r_normal = 0.0
                if coupled:
                    r_normal = float(J[row] @ velocity + rhs[row])
                    work["residual_row_dots"] += 1
                if coupled and not np.any(old) and r_normal >= 0:
                    work["block_open"] += 1
                    row += 3
                    continue
                trial, info = None, None
                if coupled:
                    residual = np.r_[r_normal, J[row + 1 : row + 3] @ velocity + rhs[row + 1 : row + 3]]
                    work["residual_row_dots"] += 2
                    if row not in cache:
                        matrix = np.diag(diagonal[take])
                        for first, second in ((0, 1), (0, 2), (1, 2)):
                            matrix[first, second] = matrix[second, first] = Z[row + first] @ Z[row + second]
                        prepared = prepare_block(matrix)
                        cache[row] = matrix, prepared if prepared is not None else False
                        work["setup_blocks"] += 1
                        work["setup_products"] += 3 * dofs
                    matrix, prepared = cache[row]
                    trial, info = local_block(matrix, residual - matrix @ old, mu[row + 1], prepared=prepared)
                    work["root_probes"] += info["probes"]
                    work["sliding_roots"] += int(info["kind"] == "slip")
                if trial is None:
                    # Existing reference is transactionally invoked only after a
                    # rejected proposal; it sees the original incoming state.
                    args = (
                        J[take],
                        Y[take],
                        diagonal[take],
                        rhs[take],
                        types[take],
                        np.array([-1, 0, 0]),
                        mu[take],
                        velocity,
                    )
                    (next_velocity, trial, _stats), counters = original.counted_reference(
                        *args, iterations=1, incoming=old
                    )
                    work["metric_fallbacks"] += int(coupled)
                    work["residual_row_dots"] += counters["residual_row_dots"]
                    work["legacy_root_probes"] += counters["root_probes"]
                    work["legacy_sliding_roots"] += counters["sliding_roots"]
                    work["legacy_cpu_gram_products"] += counters["cpu_full_gram_products"]
                    work["response_row_actions"] += counters["response_row_actions"]
                    velocity = next_velocity
                else:
                    velocity += Y[take].T @ (trial - old)
                    work["response_row_actions"] += 3
                    work[f"block_{info['kind']}"] += 1
                impulse[take] = trial
                difference = bool(np.any(trial != old))
                changed |= difference
                work["changed_transactions"] += int(difference)
                row += 3
                continue
            if diagonal[row] > 0:
                old = impulse[row]
                next_value = old - (J[row] @ velocity + rhs[row]) / diagonal[row]
                work["residual_row_dots"] += 1
                if types[row] in (0, 3):
                    next_value = max(next_value, 0.0)
                elif types[row] == 2:
                    parent = parents[row]
                    radius = max(mu[row] * impulse[parent], 0.0)
                    sibling = parent + (2 if row == parent + 1 else 1)
                    if radius <= 0:
                        next_value = 0.0
                    elif np.hypot(next_value, impulse[sibling]) > radius:
                        scale = radius / np.hypot(next_value, impulse[sibling])
                        next_value *= scale
                        other = impulse[sibling] * scale
                        velocity += Y[sibling] * (other - impulse[sibling])
                        work["response_row_actions"] += 1
                        changed |= other != impulse[sibling]
                        impulse[sibling] = other
                impulse[row] = next_value
                velocity += Y[row] * (next_value - old)
                work["response_row_actions"] += 1
                changed |= next_value != old
                work["changed_transactions"] += int(next_value != old)
            row += 1
        work["sweeps"] += 1
        residual = J @ velocity + rhs
        work["stop_row_dots"] += n
        score = physical_from_residual(residual, impulse, diagonal, types, parents, mu, cold_scale)
        curve = {
            "sweep": iteration + 1,
            "physical": score,
            "counts": {k: v for k, v in work.items() if isinstance(v, int)},
        }
        work["curve"].append(curve)
        if work["first_stop"] is None and score["stop"]:
            work["first_stop"] = iteration + 1
        if not changed or (early_stop and score["stop"]):
            break
    return original.Result(velocity, impulse, work)


def main():
    """Qualify fixed-eight quality and compare same-stop pass/work obligations."""
    assess = FunctionType(original.assess.__code__, dict(original.assess.__globals__, solve=solve), "assess_coulomb")
    _, records = original.saved_records()
    cases = []
    for gpu, record, data in records:
        for index, world in enumerate(data["worlds"]):
            J, L = data[f"J_world_{world}"].astype(float), data["L_by_size"][index].astype(float)
            diagonal, rhs, types, parents, mu = (
                data[f"{name}_{world}"] for name in ("diag", "rhs", "row_type", "row_parent", "row_mu")
            )
            vhat = data["v_hat"].reshape(4, 43)[index].astype(float)
            args = (J, L, diagonal, rhs, types, parents, mu, vhat)
            case = assess(*args)
            baseline = solve(*args, coupled=False)
            stopped = solve(*args, early_stop=True)
            case.update(
                gpu=gpu,
                step=record["step"],
                world=int(world),
                rows=len(J),
                metric_curve=baseline.work,
                stopped_candidate=stopped.work,
            )
            # The stopping rule never reads this reference comparison.
            case["stopped_physical"] = original.physical_metrics(
                J, diagonal, rhs, types, parents, mu, vhat, stopped.velocity, stopped.impulses
            )
            cases.append(case)
            print("CASE " + json.dumps(case), flush=True)
    print(
        "RESULT "
        + json.dumps(
            {
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "passed": sum(c["passed"] for c in cases),
                "cases": cases,
            }
        )
    )


if __name__ == "__main__":
    main()

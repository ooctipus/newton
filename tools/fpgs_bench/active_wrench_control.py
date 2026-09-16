# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU falsification of physical, rank-aware contact working sets.

This is not a native solver or a timing harness. CFM enters only the original
fallback and diagnostic scaling, never the candidate's physical equations.
Sliding contacts use exact polar forces, not a projected natural-map direction.
All rank decisions use one fixed FP32-scale cutoff; actual residual evaluation
always uses the unmodified physical operator. No final active-set oracle is read.
"""

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.fpgs_bench.test_sparse_contact_block import PAYLOAD_SHA
from tools.fpgs_bench.test_sparse_metric_tangents import metric_disk, physical_metrics, reference, saved_records

RANK_TOL = 32.0 * np.finfo(np.float32).eps
EVENT_TOL = 1.0e-10


def counted_reference(*args, **kwargs):
    """Count executed original CPU transactions without modifying its recurrence.

    This source-line trace is a diagnostic, not a runtime proposal or timer.
    The existing CPU oracle's full Gram and 80-probe eigen disk solver are
    reported separately; neither is a claim about native retained work.
    """
    markers = {
        reference: {
            "row = 0": ("sweeps", 1),
            "valid = (": ("row_transactions", 1),
            "normal_residual = J[row] @ v + rhs[row]": ("residual_row_dots", 1),
            "residual = J[pair] @ v + rhs[pair] + gram[pair, row] * (normal - old[0])": ("residual_row_dots", 2),
            "next_value = old - omega * (J[row] @ v + rhs[row]) / diagonal[row]": ("residual_row_dots", 1),
            "v += Y[take].T @ delta": ("response_row_actions", 3),
            "v += Y[sibling] * (other - lam[sibling])": ("response_row_actions", 1),
            "v += Y[row] * (next_value - old)": ("response_row_actions", 1),
        },
        metric_disk: {
            "lower, upper = 0.0, np.linalg.norm(linear) / radius": ("sliding_roots", 1),
            "middle = 0.5 * (lower + upper)": ("root_probes", 1),
        },
    }
    locations, counts = {}, {}
    for function, entries in markers.items():
        lines, first = inspect.getsourcelines(function)
        for marker, counter in entries.items():
            matches = [first + index for index, line in enumerate(lines) if line.strip() == marker]
            if len(matches) != 1:
                raise RuntimeError(f"Original reference counter seam changed: {marker}")
            locations[function.__code__, matches[0]] = counter
            counts[counter[0]] = 0

    def trace(frame, event, _arg):
        if event == "line":
            counter = locations.get((frame.f_code, frame.f_lineno))
            if counter is not None:
                counts[counter[0]] += counter[1]
        return trace if frame.f_code in (reference.__code__, metric_disk.__code__) else None

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        output = reference(*args, **kwargs)
    finally:
        sys.settrace(previous)
    n, dofs = np.shape(args[0])
    counts["cpu_full_gram_products"] = n * n * dofs
    return output, counts


def pivoted_basis(matrix):
    """Reveal independent columns with twice-reorthogonalized pivoted MGS."""
    matrix = np.asarray(matrix, float)
    remaining = matrix.copy()
    selected, vectors = [], []
    products = 0
    scale = float(np.max(np.linalg.norm(matrix, axis=0), initial=0.0))
    for _ in range(min(matrix.shape)):
        norms = np.linalg.norm(remaining, axis=0)
        products += remaining.size
        pivot = int(np.argmax(norms))
        if norms[pivot] <= RANK_TOL * scale:
            break
        vector = remaining[:, pivot].copy()
        for _pass in range(2):
            for basis in vectors:
                vector -= basis * (basis @ vector)
                products += 2 * len(vector)
        length = np.linalg.norm(vector)
        if length <= RANK_TOL * scale:
            remaining[:, pivot] = 0.0
            continue
        vector /= length
        selected.append(pivot)
        vectors.append(vector)
        for _pass in range(2):
            coefficients = vector @ remaining
            remaining -= vector[:, None] * coefficients
            products += 2 * remaining.size
        remaining[:, pivot] = 0.0
    basis = np.stack(vectors, axis=1) if vectors else np.zeros((matrix.shape[0], 0))
    return basis, np.asarray(selected, int), products


def rank_basic_solve(matrix, target):
    """Keep dependent variable deltas zero and check every equation for consistency."""
    matrix, target = np.asarray(matrix, float), np.asarray(target, float)
    basis, selected, products = pivoted_basis(matrix)
    value = np.zeros(matrix.shape[1])
    if len(selected):
        triangle = basis.T @ matrix[:, selected]
        projected = basis.T @ target
        value[selected] = np.linalg.solve(triangle, projected)
        products += basis.size * (len(selected) + 1)
    error = np.linalg.norm(matrix @ value - target, np.inf)
    scale = 1.0 + np.linalg.norm(target, np.inf)
    return value, {
        "rank": len(selected),
        "consistent": bool(error <= RANK_TOL * scale),
        "equation_error": float(error),
        "qr_products": products,
        "triangular_products": len(selected) * (len(selected) - 1) // 2,
    }


def decode(x, blocks, rows):
    """Represent every scalar impulse, with exact circular sliding forces."""
    impulse = np.zeros(rows)
    derivative = np.zeros((rows, len(x)))
    cursor = 0
    for row, mode, friction in blocks:
        normal = x[cursor]
        impulse[row] = normal
        derivative[row, cursor] = 1.0
        if mode == "slide":
            angle = x[cursor + 1]
            direction = np.array((np.cos(angle), np.sin(angle)))
            perpendicular = np.array((-direction[1], direction[0]))
            impulse[row + 1 : row + 3] = -friction * normal * direction
            derivative[row + 1 : row + 3, cursor] = -friction * direction
            derivative[row + 1 : row + 3, cursor + 1] = -friction * normal * perpendicular
            cursor += 2
        elif mode == "stick":
            impulse[row + 1 : row + 3] = x[cursor + 1 : cursor + 3]
            derivative[row + 1 : row + 3, cursor + 1 : cursor + 3] = np.eye(2)
            cursor += 3
        else:
            cursor += 1
    return impulse, derivative


def encode(impulse, blocks, angles):
    """Preserve current represented impulses when the working-set mode changes."""
    values = []
    for row, mode, _friction in blocks:
        values.append(float(impulse[row]))
        if mode == "slide":
            tangent = impulse[row + 1 : row + 3]
            values.append(float(np.arctan2(-tangent[1], -tangent[0])) if np.any(tangent) else angles[row])
        elif mode == "stick":
            values.extend(impulse[row + 1 : row + 3])
    return np.asarray(values)


def equations(x, blocks, residual, residual_derivative):
    """Form normal complementarity and physical sticking/sliding equations."""
    values, rows = [], []
    cursor = 0
    for row, mode, _friction in blocks:
        values.append(residual[row])
        rows.append(residual_derivative[row].copy())
        if mode == "slide":
            angle = x[cursor + 1]
            direction = np.array((np.cos(angle), np.sin(angle)))
            perpendicular = np.array((-direction[1], direction[0]))
            tangent = residual[row + 1 : row + 3]
            values.append(perpendicular @ tangent)
            angular = perpendicular @ residual_derivative[row + 1 : row + 3]
            # The angular equation rotates even when its normal impulse is zero.
            angular[cursor + 1] -= direction @ tangent
            rows.append(angular)
            cursor += 2
        elif mode == "stick":
            values.extend(residual[row + 1 : row + 3])
            rows.extend(residual_derivative[row + 1 : row + 3].copy())
            cursor += 3
        else:
            cursor += 1
    return np.asarray(values), np.asarray(rows)


@dataclass
class Result:
    """Keep physical output and explicit CPU work separate from acceptance."""

    velocity: np.ndarray
    impulses: np.ndarray
    work: dict


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=8):
    """Spend at most eight physical corrections/mode pivots, then only remaining GS."""
    if not 0 <= iterations <= 8:
        raise ValueError("The original allowance is at most eight")
    J, L = np.asarray(J, float), np.asarray(L, float)
    rhs, diagonal, mu = (np.asarray(value, float) for value in (rhs, diagonal, mu))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    vhat = np.asarray(vhat, float)
    n, dofs = J.shape
    normal = np.flatnonzero(np.isin(types, (0, 3)))
    work = dict.fromkeys(
        (
            "committed",
            "fallback_sweeps",
            "mode_pivots",
            "membership_pivots",
            "initial_discoveries",
            "materialized_rows",
            "response_triangular_products",
            "response_divisions",
            "full_residual_scans",
            "full_residual_products",
            "decode_products",
            "basis_products",
            "gram_products",
            "jacobian_products",
            "rank_solves",
            "qr_products",
            "triangular_products",
            "feasibility_probes",
            "stop_checks",
            "fallback_row_visits_upper",
            "stop_full_residual_products",
            "active_rebuilds",
            "omitted_normal_probes",
        ),
        0,
    )
    work.update(reason=None, trace=[], max_active=0, max_rank=0)
    z_cache, y_cache = {}, {}
    impulse = np.zeros(n)
    velocity = vhat.copy()
    modes, angles = {}, {}

    def materialize(rows):
        for input_row in rows:
            row = int(input_row)
            if row not in z_cache:
                z_cache[row] = np.linalg.solve(L, J[row])
                y_cache[row] = np.linalg.solve(L.T, z_cache[row])
                work["materialized_rows"] += 1
                # Literal CPU dense triangular actions, not a claimed native W cost.
                work["response_triangular_products"] += dofs * (dofs - 1)
                work["response_divisions"] += 2 * dofs

    def evaluate(value):
        rows = np.flatnonzero(value)
        materialize(rows)
        out = vhat.copy()
        for row in rows:
            out += y_cache[int(row)] * value[row]
        work["decode_products"] += len(rows) * dofs
        work["full_residual_scans"] += 1
        work["full_residual_products"] += n * dofs
        return out, J @ out + rhs

    def choose_mode(row, residual):
        if row + 2 < n and types[row] == 0 and np.all(types[row + 1 : row + 3] == 2):
            if np.all(parents[row + 1 : row + 3] == row) and mu[row + 1] > 0:
                tangent = residual[row + 1 : row + 3]
                angles[row] = float(np.arctan2(tangent[1], tangent[0]))
                return "slide" if np.linalg.norm(tangent) > EVENT_TOL else "stick"
        return "normal"

    velocity, residual = evaluate(impulse)
    cold_amplitude = float(np.max(np.abs(residual) / np.sqrt(diagonal), initial=0.0))
    cold_scale, cold_work = 1.0 + cold_amplitude, 1.0 + cold_amplitude**2
    for row in normal[residual[normal] < -EVENT_TOL]:
        modes[int(row)] = choose_mode(int(row), residual)
        work["initial_discoveries"] += 1

    for outer in range(iterations):
        if not modes:
            if np.min(residual[normal], initial=0.0) < -EVENT_TOL:
                work["reason"] = "closing_empty_set"
            break
        blocks = [(row, modes[row], float(mu[row + 1]) if modes[row] != "normal" else 0.0) for row in sorted(modes)]
        ids = np.asarray([r for row, mode, _ in blocks for r in (range(row, row + 3) if mode != "normal" else (row,))])
        work["max_active"] = max(work["max_active"], len(ids))
        work["active_rebuilds"] += 1
        materialize(ids)
        Z = np.stack([z_cache[int(row)] for row in ids])
        basis, _selected, products = pivoted_basis(Z.T)
        work["basis_products"] += products + Z.size * basis.shape[1]
        coordinates = Z @ basis
        gram = coordinates @ coordinates.T
        work["gram_products"] += len(ids) ** 2 * basis.shape[1]
        work["max_rank"] = max(work["max_rank"], basis.shape[1])
        x = encode(impulse, blocks, angles)
        _same, derivative = decode(x, blocks, n)
        residual_derivative = np.zeros((n, len(x)))
        residual_derivative[ids] = gram @ derivative[ids]
        work["jacobian_products"] += len(ids) ** 2 * len(x)
        value, jacobian = equations(x, blocks, residual, residual_derivative)
        scale_rows = np.asarray(
            [
                np.sqrt(diagonal[row])
                for row, mode, _ in blocks
                for _k in range(2 if mode == "slide" else 3 if mode == "stick" else 1)
            ]
        )
        delta, info = rank_basic_solve(jacobian / scale_rows[:, None], -value / scale_rows)
        work["rank_solves"] += 1
        work["qr_products"] += info["qr_products"]
        work["triangular_products"] += info["triangular_products"]
        entry = {
            "slot": outer + 1,
            "ids": ids.tolist(),
            "physical_rank": basis.shape[1],
            "jacobian_rank": info["rank"],
            "consistent": info["consistent"],
        }
        work["trace"].append(entry)
        if not info["consistent"] or not np.isfinite(delta).all():
            work["reason"] = "rank_inconsistent" if not info["consistent"] else "nonfinite_direction"
            break

        # Normal impulse events are affine even for the exact polar path.
        alpha, cursor, removed = 1.0, 0, None
        for row, mode, _friction in blocks:
            if delta[cursor] < 0 and -x[cursor] / delta[cursor] < alpha:
                alpha, removed = max(0.0, -x[cursor] / delta[cursor]), row
            cursor += 2 if mode == "slide" else 3 if mode == "stick" else 1
        omitted = np.asarray([row for row in normal if row not in modes], int)

        def probe(fraction, x=x, delta=delta, blocks=blocks, omitted=omitted):
            trial_x = x + fraction * delta
            trial, _ = decode(trial_x, blocks, n)
            trial_v, trial_r = evaluate(trial)
            work["feasibility_probes"] += 1
            work["omitted_normal_probes"] += len(omitted)
            violation = np.min(trial[normal], initial=0.0) < -EVENT_TOL
            for row, mode, friction in blocks:
                if mode == "stick":
                    violation |= np.linalg.norm(trial[row + 1 : row + 3]) > friction * trial[row] + EVENT_TOL
            blocked = omitted[trial_r[omitted] < -EVENT_TOL]
            return trial_x, trial, trial_v, trial_r, bool(violation or len(blocked)), blocked

        proposal = probe(alpha)
        event_rows = proposal[-1]
        if proposal[-2]:
            # Bracket actual reconstructed feasibility, not a linearized normal
            # fraction. This is not an Armijo merit search or a tolerance grid.
            low, high = 0.0, alpha
            for _ in range(24):
                middle = (low + high) * 0.5
                trial = probe(middle)
                if trial[-2]:
                    high = middle
                    event_rows = trial[-1]
                else:
                    low = middle
            alpha = low
            proposal = probe(alpha)
        entry["fraction"] = alpha
        if alpha <= EVENT_TOL:
            if len(event_rows):
                for row in event_rows:
                    modes[int(row)] = choose_mode(int(row), residual)
                    work["membership_pivots"] += 1
                work["committed"] += 1
                continue
            # An outward sticking direction at a disk boundary is a mode event.
            changed = False
            raw, _ = decode(x + delta, blocks, n)
            for row, mode, friction in blocks:
                if mode == "stick" and np.linalg.norm(raw[row + 1 : row + 3]) > friction * raw[row] + EVENT_TOL:
                    modes[row] = "slide"
                    angles[row] = float(np.arctan2(-raw[row + 2], -raw[row + 1]))
                    work["mode_pivots"] += 1
                    changed = True
            if changed:
                work["committed"] += 1
                continue
            work["reason"] = "zero_length_event"
            break
        if proposal[-2]:
            work["reason"] = "feasibility_rejected"
            break
        trial_x, impulse, velocity, residual = proposal[:4]
        work["committed"] += 1
        if removed is not None and impulse[removed] <= EVENT_TOL:
            impulse[removed] = 0.0
            if modes[removed] != "normal":
                impulse[removed + 1 : removed + 3] = 0.0
            velocity, residual = evaluate(impulse)
            modes.pop(removed)
            work["membership_pivots"] += 1
            if residual[removed] < -EVENT_TOL:
                work["reason"] = "closing_removed_row"
                break
        for row in event_rows:
            if row not in modes:
                modes[int(row)] = choose_mode(int(row), residual)
                work["membership_pivots"] += 1
        cursor = 0
        for row, mode, _ in blocks:
            if mode == "slide" and row in modes:
                direction = np.array((np.cos(trial_x[cursor + 1]), np.sin(trial_x[cursor + 1])))
                if direction @ residual[row + 1 : row + 3] < -EVENT_TOL:
                    modes[row] = "stick"
                    work["mode_pivots"] += 1
            cursor += 2 if mode == "slide" else 3 if mode == "stick" else 1
        work["stop_checks"] += 1
        metrics = physical_metrics(J, diagonal, rhs, types, parents, mu, vhat, velocity, impulse)
        # The existing CPU helper evaluates Jv twice and Jvhat once. A future
        # endpoint implementation may reuse residuals, but this control does not.
        work["stop_full_residual_products"] += 3 * n * dofs
        # Fixed diagnostic only, independent of the baseline's eventual output.
        weighted_normal = np.max(np.maximum(-residual[normal], 0.0) / np.sqrt(diagonal[normal]), initial=0.0)
        stop = metrics["natural"] <= 1e-5 and weighted_normal <= 1e-5 * cold_scale
        stop &= metrics["complementarity"] <= 1e-5 * cold_work and metrics["mdp"] <= 1e-5 * cold_work
        stop &= metrics["cone"] <= 1e-10
        if stop:
            work["reason"] = "physical_diagnostic_stop"
            break
    remaining = iterations - work["committed"]
    work["working_set_response_rows"] = work["materialized_rows"]
    work["fallback_new_response_rows"] = 0
    if work["reason"] not in (None, "physical_diagnostic_stop") and remaining:
        materialize(range(n))
        work["fallback_new_response_rows"] = work["materialized_rows"] - work["working_set_response_rows"]
        Y = np.stack([y_cache[row] for row in range(n)])
        (velocity, impulse, fallback_stats), fallback_work = counted_reference(
            J, Y, diagonal, rhs, types, parents, mu, velocity, iterations=remaining, incoming=impulse
        )
        work["fallback_sweeps"] = fallback_work["sweeps"]
        work["fallback_row_visits_upper"] = fallback_work["sweeps"] * n
        work["fallback_work"] = fallback_work
        work["fallback_stats"] = fallback_stats
    return Result(velocity, impulse, work)


def assess(J, L, diagonal, rhs, types, parents, mu, vhat):
    """Compare against unchanged metric-eight physical defects, never as a seed."""
    Y = np.linalg.solve(L.T, np.linalg.solve(L, J.T)).T
    (old_v, old_lam, old_stats), old_work = counted_reference(J, Y, diagonal, rhs, types, parents, mu, vhat)
    result = solve(J, L, diagonal, rhs, types, parents, mu, vhat)
    baseline = physical_metrics(J, diagonal, rhs, types, parents, mu, vhat, old_v, old_lam)
    actual = physical_metrics(J, diagonal, rhs, types, parents, mu, vhat, result.velocity, result.impulses)
    force, delta, H = J.T @ result.impulses, result.velocity - vhat, L @ L.T
    denominator = max(
        float(np.max(np.abs(H) @ np.abs(delta))), float(np.max(np.abs(J).T @ np.abs(result.impulses))), 1e-30
    )
    actual["momentum_uncancelled"] = float(np.max(np.abs(H @ delta - force))) / denominator
    actual["negative_impulse"] = float(np.max(np.maximum(-result.impulses[types != 2], 0.0), initial=0.0))
    failed = [
        name
        for name in ("natural", "normal", "complementarity", "mdp")
        if actual[name] > baseline[name] + 3e-5 * max(1.0, baseline[name])
    ]
    for name, bound in (("cone", 3e-5), ("momentum_uncancelled", 2e-6), ("negative_impulse", 1e-7)):
        if actual[name] > bound:
            failed.append(name)
    if not np.isfinite(result.velocity).all() or not np.isfinite(result.impulses).all():
        failed.append("finite")
    return {
        "passed": not failed,
        "failed": failed,
        "baseline": baseline,
        "actual": actual,
        "baseline_stats": old_stats,
        "baseline_work": old_work,
        "work": result.work,
    }


def main():
    """Read the existing sixteen pinned cases and print complete diagnostic results."""
    _, records = saved_records()
    cases = []
    for gpu, record, data in records:
        for index, world in enumerate(data["worlds"]):
            J = data[f"J_world_{world}"].astype(float)
            L = data["L_by_size"][index].astype(float)
            diagonal, rhs, types, parents, mu = (
                data[f"{name}_{world}"] for name in ("diag", "rhs", "row_type", "row_parent", "row_mu")
            )
            vhat = data["v_hat"].reshape(4, 43)[index].astype(float)
            case = assess(J, L, diagonal, rhs, types, parents, mu, vhat)
            case.update(gpu=gpu, step=record["step"], world=int(world), rows=len(J))
            cases.append(case)
            print("CASE " + json.dumps(case), flush=True)
    totals = {
        key: sum(case["work"][key] for case in cases)
        for key, value in cases[0]["work"].items()
        if isinstance(value, int)
    }
    print(
        "RESULT "
        + json.dumps(
            {
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "payload_sha256": {f"{key[0]}/{key[1]}": value for key, value in PAYLOAD_SHA.items()},
                "scope": "All16 pinned current/held FP64 CPU cases; no native timing or full-population convergence claim",
                "passed": sum(case["passed"] for case in cases),
                "totals": totals,
                "cases": cases,
            }
        )
    )


if __name__ == "__main__":
    main()

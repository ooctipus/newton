# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only processed-set normal driver with counted physical polar restoration.

The incoming normal is a force driver, not an additional equality in its
compensation solve. Previously processed inactive normals are protected; other
unprocessed rows may remain closing until they become the next driver. Every
finite advance, zero-length pivot, or nonlinear restoration consumes one of
the original eight slots. Exact selected response Grams preserve all physical
directions. Rank-basic compensation fixes dependent variable deltas, not forces.
"""

import hashlib
import json
from pathlib import Path
from types import FunctionType

import numpy as np

from tools.fpgs_bench import active_wrench_control as original


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=8):
    """Drive one violated normal, pivot at real constraints, and budget fallback."""
    if not 0 <= iterations <= 8:
        raise ValueError("The original allowance is at most eight")
    J, L = np.asarray(J, float), np.asarray(L, float)
    diagonal, rhs, mu, vhat = (np.asarray(a, float) for a in (diagonal, rhs, mu, vhat))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    n, dofs = J.shape
    normals = np.flatnonzero(np.isin(types, (0, 3)))
    roots = np.sqrt(diagonal)
    work = dict.fromkeys(
        (
            "committed",
            "fallback_sweeps",
            "driver_admissions",
            "membership_pivots",
            "mode_pivots",
            "restorations",
            "materialized_rows",
            "response_triangular_products",
            "response_divisions",
            "full_residual_scans",
            "full_residual_products",
            "decode_products",
            "gram_products",
            "jacobian_products",
            "qr_products",
            "rank_solves",
            "triangular_products",
            "active_rebuilds",
            "feasibility_probes",
            "processed_normal_probes",
            "stop_checks",
            "stop_full_residual_products",
        ),
        0,
    )
    work.update(reason=None, trace=[], max_active=0)
    z_cache, y_cache = {}, {}
    impulse, velocity = np.zeros(n), vhat.copy()
    modes, angles, processed = {}, {}, set()
    driver, settling = None, False

    def materialize(ids):
        for item in ids:
            row = int(item)
            if row not in z_cache:
                z_cache[row] = np.linalg.solve(L, J[row])
                y_cache[row] = np.linalg.solve(L.T, z_cache[row])
                work["materialized_rows"] += 1
                work["response_triangular_products"] += dofs * (dofs - 1)
                work["response_divisions"] += 2 * dofs

    def evaluate(value):
        ids = np.flatnonzero(value)
        materialize(ids)
        out = vhat.copy()
        for row in ids:
            out += y_cache[int(row)] * value[row]
        work["decode_products"] += len(ids) * dofs
        work["full_residual_scans"] += 1
        work["full_residual_products"] += n * dofs
        return out, J @ out + rhs

    def mode_for(row, residual):
        if (
            types[row] == 0
            and row + 2 < n
            and np.all(types[row + 1 : row + 3] == 2)
            and np.all(parents[row + 1 : row + 3] == row)
            and mu[row + 1] > 0
        ):
            tangent = residual[row + 1 : row + 3]
            angles[row] = float(np.arctan2(tangent[1], tangent[0]))
            return "slide" if np.linalg.norm(tangent) > original.EVENT_TOL else "stick"
        return "normal"

    velocity, residual = evaluate(impulse)
    amplitude = float(np.max(np.abs(residual) / roots, initial=0.0))
    amplitude_scale, work_scale = 1.0 + amplitude, 1.0 + amplitude**2
    equation_tolerance = 1e-5 * amplitude_scale

    def stopped():
        work["stop_checks"] += 1
        score = original.physical_metrics(J, diagonal, rhs, types, parents, mu, vhat, velocity, impulse)
        work["stop_full_residual_products"] += 3 * n * dofs
        negative = np.max(np.maximum(-residual[normals], 0.0) / roots[normals], initial=0.0)
        return (
            score["natural"] <= 1e-5
            and negative <= 1e-5 * amplitude_scale
            and score["complementarity"] <= 1e-5 * work_scale
            and score["mdp"] <= 1e-5 * work_scale
            and score["cone"] <= 1e-10
        )

    while work["committed"] < iterations:
        if driver is None:
            candidates = [int(r) for r in normals if r not in modes and residual[r] < -original.EVENT_TOL]
            if candidates:
                driver = min(candidates, key=lambda r: residual[r] / roots[r])
                modes[driver] = mode_for(driver, residual)
                work["driver_admissions"] += 1
                settling = False
            elif not modes or stopped():
                work["reason"] = "physical_diagnostic_stop"
                break
            else:
                settling = True

        blocks = [(r, modes[r], float(mu[r + 1]) if modes[r] != "normal" else 0.0) for r in sorted(modes)]
        ids = np.asarray([i for r, mode, _ in blocks for i in (range(r, r + 3) if mode != "normal" else (r,))], int)
        materialize(ids)
        work["active_rebuilds"] += 1
        work["max_active"] = max(work["max_active"], len(ids))
        Z = np.stack([z_cache[int(row)] for row in ids])
        gram = Z @ Z.T
        work["gram_products"] += len(ids) ** 2 * dofs
        x = original.encode(impulse, blocks, angles)
        _same, derivative = original.decode(x, blocks, n)
        response = np.zeros((n, len(x)))
        response[ids] = gram @ derivative[ids]
        work["jacobian_products"] += len(ids) ** 2 * len(x)
        values, jacobian = original.equations(x, blocks, residual, response)
        offsets, cursor = {}, 0
        scales = []
        for row, mode, _ in blocks:
            offsets[row] = cursor
            width = 2 if mode == "slide" else 3 if mode == "stick" else 1
            scales.extend([roots[row]] * width)
            cursor += width
        scales = np.asarray(scales)
        driver_column = offsets.get(driver)
        equations = np.arange(len(x))
        free = np.arange(len(x))
        if driver is not None and not settling:
            equations = equations[equations != driver_column]
            free = free[free != driver_column]
        scaled_values = values[equations] / scales[equations]
        matrix = jacobian[np.ix_(equations, free)] / scales[equations, None]
        restore = settling or np.max(np.abs(scaled_values), initial=0.0) > equation_tolerance
        rhs_solve = -scaled_values if restore else -jacobian[equations, driver_column] / scales[equations]
        if len(equations):
            basic, info = original.rank_basic_solve(matrix, rhs_solve)
            work["qr_products"] += info["qr_products"]
            work["triangular_products"] += info["triangular_products"]
            work["rank_solves"] += 1
            if not info["consistent"]:
                work["reason"] = "compensation_inconsistent"
                break
        else:
            basic, info = np.zeros(len(free)), {"rank": 0}
        delta = np.zeros(len(x))
        delta[free] = basic
        if not restore:
            delta[driver_column] = 1.0
        if not np.isfinite(delta).all():
            work["reason"] = "nonfinite_direction"
            break

        entry = {
            "slot": work["committed"] + 1,
            "driver": driver,
            "restore": bool(restore),
            "active_rows": ids.tolist(),
            "rank": info["rank"],
            "processed": sorted(processed),
        }
        work["trace"].append(entry)
        bound, release = (1.0 if restore else np.inf), None
        incoming_event = False
        if not restore:
            rate = float(response[driver] @ delta)
            entry["incoming_response_rate"] = rate
            if rate > original.RANK_TOL * diagonal[driver]:
                bound = max(0.0, -residual[driver] / rate)
                incoming_event = True
        for row, mode, friction in blocks:
            offset = offsets[row]
            if delta[offset] < 0:
                hit = max(0.0, -x[offset] / delta[offset])
                if hit <= bound:
                    bound, release, incoming_event = hit, row, False
            if mode == "stick":
                tangent, motion = x[offset + 1 : offset + 3], delta[offset + 1 : offset + 3]
                coefficients = (
                    float(motion @ motion - (friction * delta[offset]) ** 2),
                    float(2 * (tangent @ motion - friction**2 * x[offset] * delta[offset])),
                    float(tangent @ tangent - (friction * x[offset]) ** 2),
                )
                roots_disk = np.roots(np.trim_zeros(coefficients, "f")) if np.any(coefficients) else []
                for candidate_hit in roots_disk:
                    if abs(np.imag(candidate_hit)) <= original.EVENT_TOL and np.real(candidate_hit) >= 0:
                        hit = float(np.real(candidate_hit))
                        after = np.polyval(coefficients, hit + max(1e-8, abs(hit) * 1e-8))
                        if after > 0 and hit < bound:
                            bound, release, incoming_event = hit, None, False
        if not np.isfinite(bound):
            work["reason"] = "no_finite_driver_event"
            break
        protected = np.asarray([r for r in processed if r not in modes], int)

        def probe(fraction, x=x, delta=delta, blocks=blocks, protected=protected):
            trial_x = x + fraction * delta
            trial, _ = original.decode(trial_x, blocks, n)
            out, rr = evaluate(trial)
            work["feasibility_probes"] += 1
            work["processed_normal_probes"] += len(protected)
            blocked = protected[rr[protected] < -original.EVENT_TOL]
            bad = np.min(trial[normals], initial=0.0) < -original.EVENT_TOL
            for row, mode, friction in blocks:
                if mode == "stick":
                    bad |= np.linalg.norm(trial[row + 1 : row + 3]) > friction * trial[row] + original.EVENT_TOL
            return trial_x, trial, out, rr, bool(bad or len(blocked)), blocked

        proposal = probe(bound)
        blocked = proposal[-1]
        if proposal[-2]:
            low, high = 0.0, bound
            for _ in range(24):
                trial = probe((low + high) * 0.5)
                if trial[-2]:
                    high = (low + high) * 0.5
                    blocked = trial[-1]
                else:
                    low = (low + high) * 0.5
            bound, release, incoming_event = low, None, False
            proposal = probe(bound)
        # A finite polar path can cross rn=0 before its linear prediction.
        if driver is not None and not restore and proposal[3][driver] > original.EVENT_TOL:
            low, high = 0.0, bound
            for _ in range(24):
                middle = (low + high) * 0.5
                trial = probe(middle)
                if trial[3][driver] >= 0:
                    high = middle
                else:
                    low = middle
            bound, release, incoming_event = low, None, True
            proposal = probe(bound)
            blocked = proposal[-1]
        entry["advance"] = bound
        if proposal[-2]:
            work["reason"] = "actual_feasibility_rejected"
            break
        if bound <= original.EVENT_TOL:
            changed = False
            if release is not None and release != driver:
                impulse[release] = 0.0
                if modes[release] != "normal":
                    impulse[release + 1 : release + 3] = 0.0
                modes.pop(release)
                processed.add(release)
                velocity, residual = evaluate(impulse)
                work["membership_pivots"] += 1
                changed = True
            for row in blocked:
                modes[int(row)] = mode_for(int(row), residual)
                work["membership_pivots"] += 1
                changed = True
            for row, mode, friction in blocks:
                if row not in modes or mode != "stick":
                    continue
                offset = offsets[row]
                raw = x + delta
                if np.linalg.norm(raw[offset + 1 : offset + 3]) > friction * raw[offset] + original.EVENT_TOL:
                    modes[row] = "slide"
                    angles[row] = float(np.arctan2(-raw[offset + 2], -raw[offset + 1]))
                    work["mode_pivots"] += 1
                    changed = True
            if not changed:
                work["reason"] = "zero_length_without_pivot"
                break
            work["committed"] += 1
            continue
        trial_x, impulse, velocity, residual = proposal[:4]
        work["committed"] += 1
        work["restorations"] += int(restore)
        if release is not None and release != driver and impulse[release] <= original.EVENT_TOL:
            impulse[release] = 0.0
            if modes[release] != "normal":
                impulse[release + 1 : release + 3] = 0.0
            modes.pop(release)
            processed.add(release)
            velocity, residual = evaluate(impulse)
            work["membership_pivots"] += 1
        for row in blocked:
            if row not in modes:
                modes[int(row)] = mode_for(int(row), residual)
                work["membership_pivots"] += 1
        for row, mode, _ in blocks:
            if mode == "slide" and row in modes:
                angle = trial_x[offsets[row] + 1]
                direction = np.array((np.cos(angle), np.sin(angle)))
                if direction @ residual[row + 1 : row + 3] < -original.EVENT_TOL:
                    modes[row] = "stick"
                    work["mode_pivots"] += 1
        if driver is not None:
            # Old equations and the new normal are checked on actual reconstructed
            # motion. Any drift remains unprocessed and needs a counted restore.
            check_blocks = [(r, modes[r], float(mu[r + 1]) if modes[r] != "normal" else 0.0) for r in sorted(modes)]
            check_x = original.encode(impulse, check_blocks, angles)
            check_values, _ = original.equations(check_x, check_blocks, residual, np.zeros((n, len(check_x))))
            check_scales = np.asarray(
                [
                    roots[r]
                    for r, mode, _ in check_blocks
                    for _k in range(2 if mode == "slide" else 3 if mode == "stick" else 1)
                ]
            )
            if np.max(np.abs(check_values) / check_scales, initial=0.0) <= equation_tolerance:
                processed.add(driver)
                driver, settling = None, False
            elif incoming_event or settling:
                settling = True
        if stopped():
            work["reason"] = "physical_diagnostic_stop"
            break

    work["working_set_response_rows"] = work["materialized_rows"]
    work["fallback_new_response_rows"] = 0
    remaining = iterations - work["committed"]
    if work["reason"] not in (None, "physical_diagnostic_stop") and remaining:
        materialize(range(n))
        work["fallback_new_response_rows"] = work["materialized_rows"] - work["working_set_response_rows"]
        Y = np.stack([y_cache[row] for row in range(n)])
        (velocity, impulse, stats), counters = original.counted_reference(
            J, Y, diagonal, rhs, types, parents, mu, velocity, iterations=remaining, incoming=impulse
        )
        work["fallback_sweeps"], work["fallback_stats"], work["fallback_work"] = counters["sweeps"], stats, counters
    return original.Result(velocity, impulse, work)


def main():
    """Reuse the existing exact acceptance helper with a separate solver binding."""
    # This clones only the final assessment binding: the original baseline and
    # all physical gates remain unchanged, and the preserved first module is not
    # modified on disk or in its module globals.
    namespace = dict(original.assess.__globals__, solve=solve)
    assess = FunctionType(original.assess.__code__, namespace, "assess_driver")
    _, records = original.saved_records()
    cases = []
    for gpu, record, data in records:
        for index, world in enumerate(data["worlds"]):
            J, L = data[f"J_world_{world}"].astype(float), data["L_by_size"][index].astype(float)
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
        if isinstance(value, int) and not key.startswith("max_")
    }
    print(
        "RESULT "
        + json.dumps(
            {
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "original_control_sha256": hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                "passed": sum(c["passed"] for c in cases),
                "totals": totals,
                "cases": cases,
            }
        )
    )


if __name__ == "__main__":
    main()

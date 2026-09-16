# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose frozen Jacobi tails; no changed solver, damping or qualification."""

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np

from tools.fpgs_bench import coulomb_block_jacobi as jacobi

FROZEN_SHA = "0ca4b931aec62f2ea7bc5587898e20c8ffd76473038fe5983f25d4fd5852ab00"
COHORT = Path("/tmp/fpgs-coupled24-qualification-Pab7PnYA/jacobi_qualification.json")
COHORT_SHA = "f083139859e513c1a6f14ed0f33eca5349f6e89ce49d8aed86d944fc024a4411"


def make_undamped_map():
    """Disable only the outer latch and expose per-transaction root metadata."""
    tree = ast.parse(inspect.getsource(jacobi.solve))
    function = tree.body[0]
    function.body.insert(0, ast.parse("map_meta = {}").body[0])

    class Instrument(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == "relaxation == 1.0 and next_energy > energy":
                node.test = ast.Constant(False)
            return node

        def visit_AugAssign(self, node):
            if ast.unparse(node.target) == "work['block_open']":
                return [
                    node,
                    *ast.parse("map_meta[row] = dict(kind='open', probes=0, cone_error=0., law_error=0.)").body,
                ]
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and ast.unparse(node.targets[0]) == "proposed[take]":
                return [
                    node,
                    *ast.parse("""
map_meta[row] = dict(kind=info['kind'], probes=info['probes'],
                    cone_error=abs(np.linalg.norm(trial[1:])-mu[row+1]*trial[0]) if info['kind']=='slip' else 0.,
                    law_error=float(np.linalg.norm(matrix @ trial + residual[take] - matrix @ old
                                      + np.r_[0., info.get('gamma', 0.)*trial[1:]]))
                              if info['kind'] in ('slip', 'stick') else 0.)
""").body,
                ]
            return node

        def visit_Return(self, node):
            return [*ast.parse("work['map_meta'] = map_meta").body, node]

    tree = Instrument().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(jacobi.solve.__globals__)
    exec(compile(tree, "<frozen-undamped-jacobi-map>", "exec"), namespace)
    return namespace["solve"]


UNDAMPED = make_undamped_map()


def map_value(z, diagonal, bias, types, parents, mu, impulse):
    """Keep original zero-impulse bias and current velocity consistent."""
    value = UNDAMPED(
        z, np.eye(z.shape[1]), diagonal, bias, types, parents, mu, z.T @ impulse, iterations=1, incoming=impulse
    )
    return value.impulses, value.work["map_meta"]


def derivative(args, impulse, *, factor=1.0):
    """Use central differences with a fixed, local impulse-relative scale."""
    base, metadata = map_value(*args, impulse)
    size = len(impulse)
    central, right, left = (np.zeros((size, size)) for _ in range(3))
    mode_columns, probe_columns = [], []
    max_cone_error, max_law_error = 0.0, 0.0
    base_kind = {k: v["kind"] for k, v in metadata.items()}
    base_probes = {k: v["probes"] for k, v in metadata.items()}
    types, parents = args[3:5]
    for column in range(size):
        normal = int(parents[column]) if types[column] == 2 else column
        stop = normal + 3 if normal + 2 < size and types[normal + 1] == 2 else normal + 1
        step = factor * 1e-6 * max(1.0, abs(impulse[column]), np.linalg.norm(impulse[normal:stop]))
        positive, negative = impulse.copy(), impulse.copy()
        positive[column] += step
        negative[column] -= step
        plus, plus_meta = map_value(*args, positive)
        minus, minus_meta = map_value(*args, negative)
        central[:, column] = (plus - minus) / (2 * step)
        right[:, column] = (plus - base) / step
        left[:, column] = (base - minus) / step
        for details in (plus_meta, minus_meta):
            if {k: v["kind"] for k, v in details.items()} != base_kind:
                mode_columns.append(column)
            if {k: v["probes"] for k, v in details.items()} != base_probes:
                probe_columns.append(column)
            max_cone_error = max(max_cone_error, max((v["cone_error"] for v in details.values()), default=0.0))
            max_law_error = max(max_law_error, max((v["law_error"] for v in details.values()), default=0.0))
    information = {
        "mode_changed_columns": sorted(set(mode_columns)),
        "probe_changed_columns": sorted(set(probe_columns)),
        "one_sided_relative_difference": float(np.linalg.norm(right - left) / max(np.linalg.norm(central), 1e-30)),
        "root_cone_error_max": max_cone_error,
        "root_law_error_max": max_law_error,
        "base_modes": metadata,
    }
    return central, information


def spectrum(matrix):
    """Report implemented-map eigenvalues and their fixed half-map transforms."""
    eigenvalues = np.linalg.eigvals(matrix)
    damped = (1 + eigenvalues) / 2
    order = np.argsort(-np.abs(damped))
    return {
        "radius_undamped": float(np.max(np.abs(eigenvalues), initial=0)),
        "radius_half": float(np.max(np.abs(damped), initial=0)),
        "half_norm2": float(np.linalg.norm((np.eye(len(matrix)) + matrix) / 2, 2)) if len(matrix) else 0.0,
        "eigenvalues": [[float(v.real), float(v.imag)] for v in eigenvalues[order]],
        "half_eigenvalues": [[float(v.real), float(v.imag)] for v in damped[order]],
    }


def analytic_control():
    """Verify the isolated sticking map derivative H^-1 C analytically."""
    physical = np.array([[2.0, 0.2, -0.1], [0.2, 1.5, 0.1], [-0.1, 0.1, 1.0]])
    compliance = np.array([0.1, 0.2, 0.3])
    z = np.linalg.cholesky(physical)
    args = (
        z,
        np.diag(physical) + compliance,
        -physical @ np.array([2.0, 0.1, 0.05]),
        np.array([0, 2, 2]),
        np.array([-1, 0, 0]),
        np.ones(3),
    )
    actual, metadata = derivative(args, np.array([1.9, 0.11, 0.04]))
    expected = np.linalg.solve(physical + np.diag(compliance), np.diag(compliance))
    error = float(np.max(np.abs(actual - expected)))
    assert error < 1e-8, (actual, expected)
    assert not metadata["mode_changed_columns"]
    return {"max_absolute_error": error, "expected": expected.tolist(), "actual": actual.tolist()}


def main():
    """Analyze exactly the eighteen frozen failures at passes twenty-three/four."""
    assert hashlib.sha256(Path(jacobi.__file__).read_bytes()).hexdigest() == FROZEN_SHA
    assert hashlib.sha256(COHORT.read_bytes()).hexdigest() == COHORT_SHA
    print("ANALYTIC " + json.dumps(analytic_control()), flush=True)
    loader_path = Path("/tmp/fpgs-anymal-lazy-reference-zE5yUe/rows.py")
    specification = importlib.util.spec_from_file_location("saved_any_rows", loader_path)
    rows = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(rows)
    cohort = json.loads(COHORT.read_text())
    loaded = {}
    records = []
    for saved in cohort["failed_records"]:
        gpu, step, tier, world = saved["id"]
        key = (gpu, step, tier)
        if key not in loaded:
            loaded[key] = rows.load(*key)
        a, original, scalars, _owned, _pins = loaded[key]
        p = rows.world(a, original, scalars, world)
        z = p["Z"]
        diagonal = np.sum(z * z, axis=1) + p["cfm"]
        arguments = (z, diagonal, p["b"], p["kind"], p["parent"], p["mu"])
        record = {"id": saved["id"], "states": []}
        for count in (23, 24):
            result = jacobi.solve(
                z, np.eye(18), diagonal, p["b"], p["kind"], p["parent"], p["mu"], np.zeros(18), iterations=count
            )
            assert result.work["sweeps"] == count
            if count == 24:
                assert result.work["final_physical"] == saved["work"]["final_physical"]
            derivative_h, details = derivative(arguments, result.impulses)
            derivative_2h, details_2h = derivative(arguments, result.impulses, factor=2.0)
            trial, _ = map_value(*arguments, result.impulses)
            active = []
            for normal in np.flatnonzero(p["kind"] != 2):
                if result.impulses[normal] > 0 or trial[normal] > 0:
                    active.append(normal)
                    active.extend(np.flatnonzero((p["kind"] == 2) & (p["parent"] == normal)))
            active = np.asarray(active, int)
            record["states"].append(
                {
                    "pass": count,
                    "physical": result.work["final_physical"],
                    "fixed_point_distance_inf": float(np.max(np.abs(trial - result.impulses))),
                    "full": spectrum(derivative_h),
                    "full_2h": spectrum(derivative_2h),
                    "active_rows": active.tolist(),
                    "active": spectrum(derivative_h[np.ix_(active, active)]),
                    "derivative_scale_relative_difference": float(
                        np.linalg.norm(derivative_h - derivative_2h) / max(np.linalg.norm(derivative_h), 1e-30)
                    ),
                    "finite_difference": details,
                    "finite_difference_2h": details_2h,
                }
            )
        records.append(record)
        print("CASE " + json.dumps(record), flush=True)
    assert hashlib.sha256(Path(jacobi.__file__).read_bytes()).hexdigest() == FROZEN_SHA
    print(
        "RESULT "
        + json.dumps(
            {
                "cases": len(records),
                "source_guard_pass": True,
                "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

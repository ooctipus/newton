# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-rho, nonaccelerated CPU PADMM Algorithm 1 feasibility control.

Primary source: https://arxiv.org/html/2405.17020v1#alg1 . The iteration order
matches Kamino's previous-dual De Saxce correction. No physical compliance is
introduced. ETA/RHO are fixed algorithmic parameters, not physical CFM.
"""

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench.coulomb_block_hybrid import reference
from tools.fpgs_bench.natural_nesterov_control import _score

ETA, RHO = 1e-5, 1.0
SCALE_EPS = float(np.finfo(np.float32).eps)


def project_cone(value, mu):
    """Project normal-first coordinates onto the exact Euclidean Coulomb cone."""
    normal, length = float(value[0]), float(np.linalg.norm(value[1:]))
    if normal + mu * length <= 0:
        return np.zeros(3)
    if length <= mu * normal:
        return value.copy()
    next_normal = (normal + mu * length) / (1 + mu * mu)
    return np.r_[next_normal, value[1:] * (mu * next_normal / length)]


def prepare(Z, rhs, types, parents, mu):
    """Build only the once-factored DOF-space augmented operator."""
    Z, rhs, mu = (np.asarray(a, float) for a in (Z, rhs, mu))
    types, parents = np.asarray(types, int), np.asarray(parents, int)
    if not np.isin(types, (0, 2, 3)).all():
        raise ValueError("Unsupported row type")
    physical_diagonal = np.sum(Z * Z, axis=1)
    if not np.isfinite(Z).all() or np.any(physical_diagonal <= 0):
        raise ValueError("Invalid physical row response")
    pairs = reference.pairs_of({"kind": types, "parent": parents, "mu": mu})
    scale = np.clip(1 / np.sqrt(physical_diagonal + SCALE_EPS), 0.02, 50.0)
    spectral = physical_diagonal.copy()
    for parent, first, second in pairs:
        take = [parent, first, second]
        scale[take] = np.clip(1 / np.sqrt(np.max(physical_diagonal[take]) + SCALE_EPS), 0.02, 50.0)
        a, c = physical_diagonal[[first, second]]
        cross = Z[first] @ Z[second]
        spectral[[first, second]] = 0.5 * (a + c + np.sqrt((a - c) ** 2 + 4 * cross * cross))
    U = scale[:, None] * Z
    alpha = ETA + RHO
    lower = np.linalg.cholesky(alpha * np.eye(Z.shape[1]) + U.T @ U)
    return {
        "U": U,
        "scale": scale,
        "b": scale * rhs,
        "physical_rhs": rhs,
        "physical_diagonal": physical_diagonal,
        "spectral": spectral,
        "pairs": pairs,
        "mu": mu,
        "types": types,
        "lower": lower,
        "alpha": alpha,
        "Z": Z,
    }


def woodbury(data, rhs):
    """Apply (U U.T + alpha I)^-1 with the unchanged prepared factor."""
    U, lower, alpha = data["U"], data["lower"], data["alpha"]
    reduced = U.T @ rhs
    value = np.empty_like(reduced)
    for row in range(len(value)):
        value[row] = (reduced[row] - lower[row, :row] @ value[:row]) / lower[row, row]
    for row in range(len(value) - 1, -1, -1):
        value[row] = (value[row] - lower[row + 1 :, row] @ value[row + 1 :]) / lower[row, row]
    return (rhs - U @ value) / alpha


def update(data, f, y, z):
    """Perform one ordered previous-state correction, solve, cone, dual update."""
    correction = np.zeros_like(z)
    for parent, first, second in data["pairs"]:
        correction[parent] = data["mu"][first] * np.linalg.norm(z[[first, second]])
    h = -data["b"] - correction + ETA * f + RHO * y + z
    next_f = woodbury(data, h)
    argument = next_f - z / RHO
    next_y = np.maximum(argument, 0)
    for parent, first, second in data["pairs"]:
        take = [parent, first, second]
        next_y[take] = project_cone(argument[take], data["mu"][first])
    next_z = z + RHO * (next_y - next_f)
    primal = float(np.max(np.abs(next_f - next_y), initial=0)) / max(
        1.0, float(np.max(np.abs(next_f), initial=0)), float(np.max(np.abs(next_y), initial=0))
    )
    dual = float(np.max(np.abs(ETA * (next_f - f) + RHO * (next_y - y)), initial=0)) / max(
        1.0, float(np.max(np.abs(data["b"]), initial=0)), float(np.max(np.abs(next_z), initial=0))
    )
    return next_f, next_y, next_z, {"primal": primal, "dual": dual, "cheap_gate": primal <= 1e-6 and dual <= 1e-6}


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None, trace=False):
    """Run the fixed cold policy and publish velocity from feasible impulses y.

    ``diagonal`` belongs to the common replay ABI; its CFM does not enter the
    physical operator, scaling or fixed-point law. Nonzero incoming is rejected
    because this particular experiment is explicitly cold f=y=z=0.
    """
    if iterations < 0 or (incoming is not None and np.any(incoming)):
        raise ValueError("This control requires a nonnegative allowance and cold impulses")
    J, L, rhs, vhat = (np.asarray(a, float) for a in (J, L, rhs, vhat))
    Z = np.linalg.solve(L, J.T).T
    physical_rhs = J @ vhat + rhs
    data = prepare(Z, physical_rhs, types, parents, mu)
    n, dofs = Z.shape
    f, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
    work = dict.fromkeys(
        (
            "sweeps",
            "operator_products",
            "triangular_products",
            "triangular_divisions",
            "projection_rows",
            "correction_pairs",
            "residual_reduction_rows",
            "physical_scans",
            "physical_scan_products",
            "physical_projection_rows",
        ),
        0,
    )
    work.update(
        setup_gram_products=n * dofs * dofs,
        setup_diagonal_products=n * dofs,
        setup_scale_products=n * dofs,
        setup_spectral_products=len(data["pairs"]) * dofs,
        factor_count=1,
        factor_flop_proxy=dofs**3 / 3,
        cpu_whitening_products=n * dofs * (dofs - 1) // 2,
        cpu_rhs_products=n * dofs,
        curve=[],
        trace=[],
        physical_stop=False,
        first_stop=None,
        allowance=iterations,
        eta=ETA,
        rho=RHO,
        scale_epsilon=SCALE_EPS,
    )
    for iteration in range(iterations):
        f, y, z, info = update(data, f, y, z)
        work["sweeps"] += 1
        work["operator_products"] += 2 * n * dofs
        work["triangular_products"] += dofs * (dofs - 1)
        work["triangular_divisions"] += 2 * dofs
        work["projection_rows"] += n
        work["correction_pairs"] += len(data["pairs"])
        work["residual_reduction_rows"] += 2 * n
        if info["cheap_gate"]:
            impulse = data["scale"] * y
            residual = physical_rhs + Z @ (Z.T @ impulse)
            score = _score(
                residual,
                impulse,
                physical_rhs,
                data["physical_diagonal"],
                data["spectral"],
                data["types"],
                data["pairs"],
                data["mu"],
            )
            work["physical_scans"] += 1
            work["physical_scan_products"] += 2 * n * dofs
            work["physical_projection_rows"] += n
            info["physical"] = score
        work["curve"].append(dict(sweep=iteration + 1, **info))
        if trace:
            work["trace"].append({"f": f.copy(), "y": y.copy(), "z": z.copy()})
        if info.get("physical", {}).get("stop", False):
            work["physical_stop"] = True
            work["first_stop"] = iteration + 1
            break
    impulse = data["scale"] * y
    kinetic = Z.T @ impulse
    velocity = vhat + np.linalg.solve(L.T, kinetic)
    final_residual = physical_rhs + Z @ kinetic
    work["final_physical"] = _score(
        final_residual,
        impulse,
        physical_rhs,
        data["physical_diagonal"],
        data["spectral"],
        data["types"],
        data["pairs"],
        data["mu"],
    )
    work.update(
        publish_products=n * dofs,
        cpu_decode_products=dofs * (dofs - 1) // 2,
        offline_final_scan_products=n * dofs,
        offline_final_projection_rows=n,
        unused_allowance=iterations - work["sweeps"],
    )
    return block.original.Result(velocity, impulse, work)

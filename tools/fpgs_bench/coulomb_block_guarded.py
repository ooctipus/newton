# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU global-progress safeguard; frozen local law and original parallel tail."""

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np

from tools.fpgs_bench import coulomb_block_control as block
from tools.fpgs_bench.coulomb_block_hybrid import parallel_continue

FROZEN_SHA = "0fc44df283d5284763f7fd4889b4b8cc475b197bb6314714cf1952bd0a3d5295"


def merit(score):
    """Use the fixed physical stopping bounds, never the baseline solution."""
    values = [
        score[k] / bound
        for k, bound in (("natural", 1e-5), ("normal", 3e-5), ("complementarity", 3e-5), ("mdp", 3e-5), ("cone", 1e-10))
    ]
    return float(max(values)) if np.isfinite(values).all() else np.inf


def build_prefix():
    """Insert snapshots and a post-pass acceptance guard without changing math."""
    if hashlib.sha256(Path(block.__file__).read_bytes()).hexdigest() != FROZEN_SHA:
        raise RuntimeError("Frozen coupled source changed")
    tree = ast.parse(inspect.getsource(block.solve))
    function = tree.body[0]
    sweep = next(node for node in function.body if isinstance(node, ast.For) and node.target.id == "iteration")
    position = function.body.index(sweep)
    function.body[position:position] = ast.parse(
        """
accepted_score = physical_from_residual(cold, impulse, diagonal, types, parents, mu, cold_scale)
accepted_energy = merit(accepted_score)
work.update(accepted_coupled_sweeps=0, snapshot_values=0, rollback_values=0,
            guard_rejected=False, initial_merit_row_projections=n, cold_scale=cold_scale,
            guarded_handoff_physical=accepted_score, guarded_handoff_merit=accepted_energy)
"""
    ).body
    sweep.body[0:0] = ast.parse(
        """
saved_velocity, saved_impulse = velocity.copy(), impulse.copy()
work['snapshot_values'] += n + dofs
"""
    ).body
    append = next(
        index
        for index, node in enumerate(sweep.body)
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("work['curve'].append")
    )
    sweep.body[append + 1 : append + 1] = ast.parse(
        """
energy = merit(score)
curve['merit'] = energy
curve['preceding_merit'] = accepted_energy
curve['accepted'] = energy < accepted_energy
if not curve['accepted']:
    velocity, impulse = saved_velocity, saved_impulse
    work['rollback_values'] += n + dofs
    work['guard_rejected'] = True
    break
accepted_score, accepted_energy = score, energy
work['accepted_coupled_sweeps'] += 1
work['guarded_handoff_physical'] = accepted_score
work['guarded_handoff_merit'] = accepted_energy
"""
    ).body
    ast.fix_missing_locations(tree)
    namespace = dict(block.solve.__globals__, merit=merit)
    exec(compile(tree, "<coulomb-guarded-prefix>", "exec"), namespace)
    assert namespace["solve"].__globals__["local_block"] is block.local_block
    return namespace["solve"]


prefix_solve = build_prefix()


def solve(J, L, diagonal, rhs, types, parents, mu, vhat, *, iterations=24, incoming=None):
    """Charge attempted passes, rollback, remaining continuation and final scan."""
    if iterations < 0:
        raise ValueError("Negative outer allowance")
    result = prefix_solve(
        J,
        L,
        diagonal,
        rhs,
        types,
        parents,
        mu,
        vhat,
        iterations=min(6, iterations),
        early_stop=True,
        incoming=incoming,
    )
    consumed = result.work["sweeps"]
    handoff_score = result.work["guarded_handoff_physical"]
    handoff_merit = result.work["guarded_handoff_merit"]
    work = {
        "coupled_sweeps": consumed,
        "parallel_sweeps": 0,
        "majorizer_products": 0,
        "coupled_work": result.work,
        "physical_stop": bool(handoff_score["stop"]),
        "fallback": False,
        "allowance": iterations,
        "unused_allowance": iterations - consumed,
        "snapshot_values": result.work["snapshot_values"],
        "rollback_values": result.work["rollback_values"],
        "guard_rejected": result.work["guard_rejected"],
        "handoff_merit": handoff_merit,
        "tail_merit": None,
        "selected_handoff": False,
        "post_tail_scan_row_dots": 0,
        "cpu_fallback_Z_rebuild_rows": 0,
        "cpu_fallback_Z_triangular_products": 0,
        "handoff_row_dots": 0,
        "handoff_transpose_products": 0,
    }
    remaining = iterations - consumed
    if handoff_score["stop"] or remaining == 0:
        return block.original.Result(result.velocity, result.impulses, work)
    J, L = np.asarray(J, float), np.asarray(L, float)
    Z = np.linalg.solve(L, J.T).T
    n, dofs = Z.shape
    work["cpu_fallback_Z_rebuild_rows"] = n
    work["cpu_fallback_Z_triangular_products"] = n * dofs * (dofs - 1) // 2
    zero_rhs = J @ result.velocity + rhs - Z @ (Z.T @ result.impulses)
    work["handoff_row_dots"] = 2 * n
    work["handoff_transpose_products"] = n * dofs
    continued = parallel_continue(Z, diagonal, zero_rhs, types, parents, mu, result.impulses, iterations=remaining)
    velocity = result.velocity + np.linalg.solve(L.T, continued.velocity)
    tail_residual = J @ velocity + rhs
    tail_score = block.physical_from_residual(
        tail_residual, continued.impulses, diagonal, types, parents, mu, result.work["cold_scale"]
    )
    tail_merit = merit(tail_score)
    keep_handoff = not (tail_merit < handoff_merit)
    work.update(
        parallel_sweeps=continued.work["sweeps"],
        parallel_work=continued.work,
        majorizer_products=continued.work["majorizer_products"],
        fallback=True,
        unused_allowance=remaining - continued.work["sweeps"],
        cpu_final_backsolve_products=dofs * (dofs - 1) // 2,
        post_tail_scan_row_dots=n,
        tail_physical=tail_score,
        tail_merit=tail_merit,
        selected_handoff=keep_handoff,
        final_physical_stop=bool((handoff_score if keep_handoff else tail_score)["stop"]),
        endpoint_selection_values=n + dofs,
    )
    assert consumed + continued.work["sweeps"] <= iterations
    if keep_handoff:
        return block.original.Result(result.velocity, result.impulses, work)
    return block.original.Result(velocity, continued.impulses, work)

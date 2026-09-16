# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU-only fixed forward/reverse transaction policy over the frozen block law."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType

from tools.fpgs_bench import coulomb_block_control as block

FROZEN_SHA = "0fc44df283d5284763f7fd4889b4b8cc475b197bb6314714cf1952bd0a3d5295"


def build_solve():
    """Change traversal only; preserve indices, preparation, budgets and all math."""
    actual = hashlib.sha256(Path(block.__file__).read_bytes()).hexdigest()
    if actual != FROZEN_SHA:
        raise RuntimeError(f"Frozen source changed: {actual}")
    tree = ast.parse(inspect.getsource(block.solve))
    function = tree.body[0]
    sweep = next(node for node in function.body if isinstance(node, ast.For) and node.target.id == "iteration")
    while_loop = next(node for node in sweep.body if isinstance(node, ast.While))
    assert ast.unparse(while_loop.test) == "row < n"
    # Groups depend only on the canonical row structure, not residuals or cases.
    # Keep malformed/unsupported friction triplets internally in scalar order too.
    preparation = ast.parse(
        """
transaction_groups = []
group_row = 0
while group_row < n:
    group_size = 1
    if (types[group_row] == 0 and group_row + 2 < n
        and np.all(types[group_row + 1:group_row + 3] == 2)
        and np.all(parents[group_row + 1:group_row + 3] == group_row)):
        group_size = 3
    transaction_groups.append((group_row, group_row + group_size))
    group_row += group_size
work['transaction_group_tests'] = len(transaction_groups)
work['transaction_group_ints'] = 2 * len(transaction_groups)
"""
    ).body
    index = function.body.index(sweep)
    function.body[index:index] = preparation
    grouped = ast.parse(
        """
for group_bounds in (transaction_groups if iteration % 2 == 0 else reversed(transaction_groups)):
    row = group_bounds[0]
    while row < group_bounds[1]:
        pass
"""
    ).body[0]
    grouped.body[1].body = while_loop.body
    sweep.body[sweep.body.index(while_loop)] = grouped
    ast.fix_missing_locations(tree)
    namespace = dict(block.solve.__globals__)
    exec(compile(tree, "<frozen-coulomb-alternating>", "exec"), namespace)
    result = namespace["solve"]
    assert result.__globals__["local_block"] is block.local_block
    return result


solve = build_solve()


def main():
    """Use all sixteen existing cases; each forward or reverse pass costs one."""
    original = block.original
    assess = FunctionType(
        original.assess.__code__, dict(original.assess.__globals__, solve=solve), "assess_alternating"
    )
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
            forward = block.solve(*args)
            forward_physical = original.physical_metrics(
                J, diagonal, rhs, types, parents, mu, vhat, forward.velocity, forward.impulses
            )
            case.update(
                gpu=gpu,
                step=record["step"],
                world=int(world),
                forward_physical=forward_physical,
                forward_work=forward.work,
                failed_vs_forward=[
                    name
                    for name in ("natural", "normal", "complementarity", "mdp")
                    if case["actual"][name] > forward_physical[name] + 3e-5 * max(1, forward_physical[name])
                ],
                stopped_work=solve(*args, early_stop=True).work,
            )
            cases.append(case)
            print("CASE " + json.dumps(case), flush=True)
    print(
        "RESULT "
        + json.dumps(
            {
                "policy": "forward on odd passes, reverse on even; canonical triplets internally forward",
                "primitive_sha256": FROZEN_SHA,
                "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "passed": sum(c["passed"] for c in cases),
                "cases": cases,
            }
        )
    )


if __name__ == "__main__":
    main()

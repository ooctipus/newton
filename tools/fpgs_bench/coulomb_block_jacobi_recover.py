# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One per-pass midpoint safeguard; no permanent relaxation latch or new root."""

import ast
import hashlib
import inspect
from pathlib import Path
from types import FunctionType

from tools.fpgs_bench import coulomb_block_jacobi as frozen

FROZEN_SHA = "0ca4b931aec62f2ea7bc5587898e20c8ffd76473038fe5983f25d4fd5852ab00"


def build_solve():
    """Restore alpha one each pass without changing proposals or physical math."""
    if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != FROZEN_SHA:
        raise RuntimeError("Frozen Jacobi source changed")
    tree = ast.parse(inspect.getsource(frozen.solve))
    function = tree.body[0]
    sweep = next(node for node in function.body if isinstance(node, ast.For) and node.target.id == "iteration")
    position = function.body.index(sweep)
    function.body[position:position] = ast.parse(
        "work.update(midpoint_steps=[], relaxation_policy='per_pass_single_midpoint')"
    ).body
    sweep.body[0:0] = ast.parse("relaxation = 1.0").body
    guard = next(
        node
        for node in sweep.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "relaxation == 1.0 and next_energy > energy"
    )
    guard.body.extend(ast.parse("work['midpoint_steps'].append(iteration + 1)").body)

    class Relabel(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "latch_step":
                node.value = "last_midpoint_step"
            return node

        def visit_keyword(self, node):
            self.generic_visit(node)
            if node.arg == "latch_step":
                node.arg = "last_midpoint_step"
            return node

    tree = Relabel().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(frozen.solve.__globals__)
    exec(compile(tree, "<jacobi-per-pass-midpoint>", "exec"), namespace)
    assert namespace["solve"].__globals__["block"] is frozen.block
    return namespace["solve"]


solve = build_solve()


def main():
    """Reuse the exact G1 sixteen-case driver with only the candidate replaced."""
    FunctionType(frozen.main.__code__, dict(frozen.main.__globals__, solve=solve, __file__=__file__), "main_recover")()


if __name__ == "__main__":
    main()

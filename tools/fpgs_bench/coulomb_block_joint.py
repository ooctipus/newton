# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU joint impulse/multiplier iteration with one local evaluation per pass."""

import ast
import hashlib
import inspect
from pathlib import Path
from types import FunctionType

import numpy as np

from tools.fpgs_bench import coulomb_block_jacobi as frozen

FROZEN_SHA = "0ca4b931aec62f2ea7bc5587898e20c8ffd76473038fe5983f25d4fd5852ab00"


def physical_from_residual(*args):
    """Keep the declared physical stop with the native-aligned cone bound."""
    score = frozen.block.physical_from_residual(*args)
    score["stop"] = (
        all(np.isfinite(score[key]) for key in ("natural", "normal", "complementarity", "mdp", "cone"))
        and score["natural"] <= 1e-5
        and score["normal"] <= 3e-5
        and score["complementarity"] <= 3e-5
        and score["mdp"] <= 3e-5
        and score["cone"] <= 3e-5
    )
    return score


def merit(score):
    """Use the same native-aligned bounds for the permanent relaxation latch."""
    values = [
        score[key] / bound
        for key, bound in (
            ("natural", 1e-5),
            ("normal", 3e-5),
            ("complementarity", 3e-5),
            ("mdp", 3e-5),
            ("cone", 3e-5),
        )
    ]
    return float(max(values)) if np.isfinite(values).all() else np.inf


def joint_block(matrix, bias, friction, gamma, *, prepared=None):
    """Publish the current multiplier trial and take one auxiliary Newton step.

    Progress is not local convergence. An unfinished but feasible trial never
    fails the old tight KKT gate; the common-velocity physical stop decides.
    """
    matrix, bias = np.asarray(matrix, float), np.asarray(bias, float)
    info = {"kind": "unsafe", "probes": 0, "derivative_evaluations": 0, "reason": None}

    def unsafe(reason):
        info["reason"] = reason
        return None, 0.0, info

    if not np.isfinite(matrix).all() or not np.isfinite(bias).all() or not np.isfinite(gamma):
        return unsafe("nonfinite_input")
    if not np.isfinite(friction) or friction < 0:
        return unsafe("invalid_friction")
    if bias[0] >= 0:
        info["kind"] = "open"
        return np.zeros(3), 0.0, info
    a = float(matrix[0, 0])
    if a <= 0:
        return unsafe("invalid_normal")
    if friction == 0:
        value = np.array([-bias[0] / a, 0.0, 0.0])
        if not np.isfinite(value).all():
            return unsafe("nonfinite_normal")
        info["kind"] = "normal"
        return value, 0.0, info
    prepared = frozen.block.prepare_block(matrix) if prepared is None else prepared
    if prepared is None or prepared is False:
        return unsafe("unsafe_schur")
    a, h, schur, eigenvalues = prepared
    beta = bias[1:] - h * bias[0] / a
    radius_min = friction * (-bias[0]) / (a + friction * np.linalg.norm(h))
    if not np.isfinite(radius_min) or radius_min <= 0:
        return unsafe("invalid_radius_bound")
    upper = max(0.0, np.linalg.norm(beta) / radius_min - eigenvalues[0])
    if not np.isfinite(upper):
        return unsafe("nonfinite_upper")
    seed = float(np.clip(gamma, 0.0, upper))
    aa, bb, dd = schur[0, 0] + seed, schur[0, 1], schur[1, 1] + seed
    determinant = aa * dd - bb * bb
    if not np.isfinite(determinant) or determinant <= 0:
        return unsafe("invalid_shifted_schur")
    reciprocal = 1.0 / determinant
    tangent = np.array([-dd * beta[0] + bb * beta[1], bb * beta[0] - aa * beta[1]]) * reciprocal
    normal = float((-bias[0] - h @ tangent) / a)
    length = float(np.linalg.norm(tangent))
    gap = length - friction * normal
    info["probes"] = 1
    if not np.isfinite(tangent).all() or not np.isfinite(normal) or not np.isfinite(gap):
        return unsafe("nonfinite_trial")
    if seed == 0 and normal >= 0 and gap <= 0:
        info["kind"] = "stick"
        return np.r_[normal, tangent], 0.0, info
    # Reuse the same inverse; this is the derivative of the one evaluation,
    # not a second multiplier query or an inner iteration loop.
    derivative_t = np.array([-dd * tangent[0] + bb * tangent[1], bb * tangent[0] - aa * tangent[1]]) * reciprocal
    derivative_n = float(-(h @ derivative_t) / a)
    derivative = float(tangent @ derivative_t / max(length, 1e-300) - friction * derivative_n)
    info["derivative_evaluations"] = 1
    if not np.isfinite(derivative):
        return unsafe("nonfinite_derivative")
    if derivative >= 0:
        return unsafe("nonnegative_derivative")
    proposal = seed - gap / derivative
    if not np.isfinite(proposal):
        return unsafe("nonfinite_multiplier")
    next_gamma = float(np.clip(proposal, 0.0, upper))
    if gap > 0:
        denominator = a * length + friction * (h @ tangent)
        if not np.isfinite(denominator) or denominator <= 0:
            return unsafe("invalid_inward_repair")
        scale = friction * (-bias[0]) / denominator
        tangent = scale * tangent
        normal = float((-bias[0] - h @ tangent) / a)
    value = np.r_[normal, tangent]
    if not np.isfinite(value).all() or normal < 0:
        return unsafe("nonfinite_repair")
    info.update(kind="progress", gamma_evaluated=seed, gamma_next=next_gamma, gap=gap)
    return value, next_gamma, info


def build_solve():
    """Replace only local proposals and carry their matched multiplier state."""
    if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != FROZEN_SHA:
        raise RuntimeError("Frozen Jacobi driver changed")
    tree = ast.parse(inspect.getsource(frozen.solve))
    function = tree.body[0]
    function.args.kwonlyargs.append(ast.arg(arg="incoming_gamma"))
    function.args.kw_defaults.append(ast.Constant(None))
    sweep = next(node for node in function.body if isinstance(node, ast.For) and node.target.id == "iteration")
    position = function.body.index(sweep)
    function.body[position:position] = ast.parse("""
gamma = np.zeros(n) if incoming_gamma is None else np.asarray(incoming_gamma,float).copy()
if gamma.shape != (n,) or not np.isfinite(gamma).all() or np.any(gamma < 0):
    raise ValueError('Initial multiplier state must be finite, nonnegative and row-aligned')
work.update(block_progress=0, derivative_evaluations=0, gamma_blend_values=0,
            gamma_midpoint_values=0, joint_unsafe_reasons={}, joint_state=True, cone_tolerance=3e-5)
""").body
    sweep.body[0:0] = ast.parse("proposed_gamma = gamma.copy()").body

    class Joint(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) == "block.physical_from_residual":
                node.func = ast.Name(id="physical_from_residual", ctx=ast.Load())
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "block.local_block":
                node.value.func = ast.Name(id="joint_block", ctx=ast.Load())
                node.value.args.append(ast.parse("gamma[row]", mode="eval").body)
                node.targets[0] = ast.parse("trial, proposed_gamma[row], info = None").body[0].targets[0]
                return [node, *ast.parse("work['derivative_evaluations'] += info['derivative_evaluations']").body]
            if len(node.targets) == 1 and ast.unparse(node.targets[0]) == "next_impulse":
                if isinstance(node.value, ast.BinOp) and ast.unparse(node.value).startswith("impulse + relaxation"):
                    return [
                        node,
                        *ast.parse("""
next_gamma = gamma + relaxation*(proposed_gamma-gamma)
work['gamma_blend_values'] += n
""").body,
                    ]
                return [
                    node,
                    *ast.parse("""
next_gamma = 0.5*gamma + 0.5*next_gamma
work['gamma_midpoint_values'] += n
""").body,
                ]
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
                if ast.unparse(node.targets[0]).startswith("(impulse, kinetic_delta,"):
                    return [node, *ast.parse("gamma = next_gamma").body]
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == "trial is None":
                node.body[0:0] = ast.parse("""
proposed_gamma[row] = 0.0
reason = info.get('reason', 'unsupported_friction')
work['joint_unsafe_reasons'][reason] = work['joint_unsafe_reasons'].get(reason,0)+1
""").body
            return node

        def visit_AugAssign(self, node):
            if ast.unparse(node.target) == "work['block_open']":
                return [node, *ast.parse("proposed_gamma[row] = 0.0").body]
            return node

        def visit_Return(self, node):
            return [*ast.parse("work['gamma'] = gamma.tolist()").body, node]

    tree = Joint().visit(tree)
    ast.fix_missing_locations(tree)
    namespace = dict(
        frozen.solve.__globals__, joint_block=joint_block, physical_from_residual=physical_from_residual, merit=merit
    )
    exec(compile(tree, "<joint-impulse-multiplier>", "exec"), namespace)
    return namespace["solve"]


solve = build_solve()


def main():
    """Reuse the fixed G1 sixteen-case comparison at its original eight passes."""
    FunctionType(frozen.main.__code__, dict(frozen.main.__globals__, solve=solve, __file__=__file__), "main_joint")()


if __name__ == "__main__":
    main()

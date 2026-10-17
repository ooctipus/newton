# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tree kernels' reference and stage-clock switches are factory arguments, read once by the owner."""

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import warp as wp

from newton._src.solvers.feather_pgs import tree_kinetic_state
from newton._src.solvers.feather_pgs.mixed_tree_kinetic import MixedTreeKineticState
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton.tests.test_feather_pgs_mixed_tree_free_body import _build

# A packed, warp-sized fixed-root tree: both switches change every kernel built for it.
_LAYOUT = TreeLayout(4, 3, 6, 3, 2, 0, (2, 1, 0), lanes=8)
# Modules that generate the tree kernels' code; owners read the environment for them, at construction.
_GENERATION_MODULES = (
    "cached_force.py",
    "cached_state.py",
    "cached_state_forest.py",
    "response_operator.py",
    "tree_chain_scan.py",
    "tree_kinetic_layout.py",
    "tree_kinetic_state.py",
)


def _switches(reference, clocked):
    return patch.dict(
        os.environ,
        {"FEATHER_PGS_TREE_REFERENCE": str(int(reference)), "FEATHER_PGS_STAGE_CLOCKS": str(int(clocked))},
    )


def _kernels(reference, clocked):
    return (
        tree_kinetic_state.get_state_kernel(False, layout=_LAYOUT, reference=reference, clocked=clocked),
        tree_kinetic_state.get_state_kernel(True, layout=_LAYOUT, reference=reference, clocked=clocked),
        tree_kinetic_state.get_predictor_kernel(
            layout=_LAYOUT, response="cholesky", reference=reference, clocked=clocked
        ),
    )


def _environment_reads(node):
    """Lines that read os.environ or os.getenv under an AST node."""
    return [
        sub.lineno
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute)
        and isinstance(sub.value, ast.Name)
        and sub.value.id == "os"
        and sub.attr in ("environ", "getenv")
    ]


def _is_cache(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Attribute):
        return (
            isinstance(target.value, ast.Name)
            and target.value.id == "functools"
            and target.attr in ("cache", "lru_cache")
        )
    return isinstance(target, ast.Name) and target.id in ("cache", "lru_cache")


class TestTreeKernelSwitches(unittest.TestCase):
    def test_switches_select_the_kernels_by_argument(self):
        """Build four different repair, finish and predictor kernels for the four switch combinations."""
        built = [_kernels(reference, clocked) for reference in (False, True) for clocked in (False, True)]
        for position in range(3):
            self.assertEqual(len({kernels[position].module.name for kernels in built}), 4)

    def test_factories_never_read_the_environment(self):
        """Rebuild identical kernels from the same arguments whatever the environment holds."""
        expected = _kernels(False, False)
        with _switches(True, True):
            for factory in (
                tree_kinetic_state.get_helpers,
                tree_kinetic_state.get_state_kernel,
                tree_kinetic_state.get_predictor_kernel,
            ):
                factory.cache_clear()
            rebuilt = _kernels(False, False)
        # Warp keeps one kernel object per emitted content, so equal code is the same object.
        for kernel, again in zip(expected, rebuilt, strict=True):
            self.assertIs(again, kernel)

    def test_kernel_generation_reads_no_environment(self):
        """Keep environment reads out of tree-kernel generation and out of every cached factory body."""
        package = Path(tree_kinetic_state.__file__).parent
        for name in _GENERATION_MODULES:
            with self.subTest(module=name):
                self.assertEqual(_environment_reads(ast.parse((package / name).read_text())), [])
        for path in sorted(package.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.FunctionDef) and any(_is_cache(d) for d in node.decorator_list):
                    with self.subTest(factory=f"{path.name}:{node.name}"):
                        self.assertEqual(_environment_reads(node), [])

    def test_owners_in_one_process_build_their_own_kernels(self):
        """Bind each owner's own kernels when owners are built one after another under each switch.

        On CPU the layouts are fixed (no lane timing), so the switch is the only
        difference between the owners; the held-solve and subtree-reduce A/B tests
        build their arms in this order.
        """
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, pgs_mode="split", friction_anchor_beta=0.0)
        owners = {}
        for name, reference, clocked in (
            ("reference", True, False),
            ("default", False, False),
            ("clocked", False, True),
        ):
            with _switches(reference, clocked):
                owners[name] = MixedTreeKineticState(solver)
            self.assertEqual((owners[name].reference, owners[name].clocked), (reference, clocked))
            for group in owners[name].groups:
                self.assertEqual((group.reference, group.clocked), (reference, clocked))
        # Outside the patch: the owners keep what they read at construction.
        self.assertTrue(owners["reference"].reference)
        self.assertIsNotNone(getattr(solver, "_stage_clocks", None))
        self.assertTrue(owners["default"].groups)
        for reference, default, clocked in zip(
            owners["reference"].groups, owners["default"].groups, owners["clocked"].groups, strict=True
        ):
            self.assertEqual(reference.layout, default.layout)
            for kernel in ("repair_kernel", "finish_kernel", "predictor_kernel"):
                with self.subTest(layout=default.layout.root_families, kernel=kernel):
                    self.assertIsNot(getattr(reference, kernel), getattr(default, kernel))
                    self.assertIsNot(getattr(clocked, kernel), getattr(default, kernel))


if __name__ == "__main__":
    unittest.main()

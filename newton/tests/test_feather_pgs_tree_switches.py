# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The tree kernels' reference and stage-clock switches are factory arguments, set once by the owner."""

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import tree_kinetic_state
from newton._src.solvers.feather_pgs.held_tree import HeldTree
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS
from newton._src.solvers.feather_pgs.tree_kinetic_layout import TreeLayout
from newton._src.solvers.feather_pgs.tree_plan import TreePlan
from newton.tests.test_feather_pgs_mixed_tree_free_body import _build

# A packed, warp-sized fixed-root tree: both switches change every kernel built for it.
_LAYOUT = TreeLayout(4, 3, 6, 3, 2, 0, (2, 1, 0), lanes=8)
_TREE = TreePlan.build([-1, 0, 1, 0], [0, 1, 1, 1])
# Modules that generate the tree kernels' code; owners read the environment for them, at construction.
_GENERATION_MODULES = (
    "cached_force.py",
    "response_operator.py",
    "tree_kinetic_layout.py",
    "tree_kinetic_state.py",
    "tree_scan.py",
)


def _switches(reference, clocked):
    return patch.dict(
        os.environ,
        {"FEATHER_PGS_TREE_REFERENCE": str(int(reference)), "FEATHER_PGS_STAGE_CLOCKS": str(int(clocked))},
    )


def _kernels(reference, clocked):
    schedules = tree_kinetic_state.held_schedules(_TREE, _LAYOUT.lanes, chains=False, reference=reference)
    return (
        tree_kinetic_state.get_state_kernel(False, schedules, layout=_LAYOUT, clocked=clocked),
        tree_kinetic_state.get_state_kernel(True, schedules, layout=_LAYOUT, clocked=clocked),
        tree_kinetic_state.get_predictor_kernel(
            schedules,
            layout=_LAYOUT,
            response="cholesky",
            reference=reference,
            clocked=clocked,
            twists=tree_kinetic_state.publishes_twists(_LAYOUT, schedules, cuda=True),
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


def _package_functions(package):
    """Every function of a flat package by (module, qualified name), and each module's relative imports."""
    functions, imports = {}, {}

    def collect(module, body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[(module, prefix + node.name)] = node
                collect(module, node.body, f"{prefix}{node.name}.")
            elif isinstance(node, ast.ClassDef):
                collect(module, node.body, f"{prefix}{node.name}.")

    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text())
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                for alias in node.names:
                    # "from . import module" binds a module; "from .module import name" binds a member.
                    target = (alias.name, None) if node.module is None else (node.module, alias.name)
                    names.setdefault(alias.asname or alias.name, target)
        imports[path.stem] = names
        collect(path.stem, tree.body, "")
    return functions, imports


def _callees(module, node, functions, imports):
    """Package functions a body calls through a name or an imported module that resolves exactly."""
    for call in ast.walk(node):
        func = call.func if isinstance(call, ast.Call) else None
        if isinstance(func, ast.Name):
            source, member = imports[module].get(func.id, (None, None))
            for key in ((module, func.id), (source, member)):
                if key in functions:
                    yield key
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            source, member = imports[module].get(func.value.id, (None, None))
            if member is None and (source, func.attr) in functions:
                yield source, func.attr


def _cached_environment_reads(package):
    """Environment reads any cached factory of the package reaches through resolvable calls."""
    functions, imports = _package_functions(package)
    reads = []
    for factory, node in functions.items():
        if not any(_is_cache(decorator) for decorator in node.decorator_list):
            continue
        seen, pending = {factory}, [factory]
        while pending:
            key = pending.pop()
            lines = _environment_reads(functions[key])
            if lines:
                reads.append(f"{'.'.join(factory)} reaches {'.'.join(key)}, lines {lines}")
            for callee in _callees(key[0], functions[key], functions, imports):
                if callee not in seen:
                    seen.add(callee)
                    pending.append(callee)
    return sorted(reads)


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

    def test_cached_factories_reach_no_environment_read(self):
        """Follow every cached factory's calls through the package: none reaches an environment read."""
        self.assertEqual(_cached_environment_reads(Path(tree_kinetic_state.__file__).parent), [])

    def test_reach_scan_follows_helpers_across_modules(self):
        """Report a read that a cached factory reaches through a helper, by name or through its module."""
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "switches.py").write_text(
                'import os\n\n\ndef reference():\n    return os.environ.get("FEATHER_PGS_TREE_REFERENCE") == "1"\n'
            )
            (package / "factories.py").write_text(
                "import functools\n\nfrom . import switches\nfrom .switches import reference\n\n\n"
                "@functools.cache\ndef by_name(layout):\n    return layout, reference()\n\n\n"
                "@functools.cache\ndef by_module(layout):\n    return layout, switches.reference()\n\n\n"
                "@functools.cache\ndef clean(layout):\n    return layout\n"
            )
            self.assertEqual(
                _cached_environment_reads(package),
                [
                    "factories.by_module reaches switches.reference, lines [5]",
                    "factories.by_name reaches switches.reference, lines [5]",
                ],
            )

    def test_owners_in_one_process_build_their_own_kernels(self):
        """Bind each owner's own kernels when owners are built one after another under each switch.

        On CPU the layouts are fixed (no lane timing), so the switch is the only
        difference between the owners; the held-solve and subtree-reduce A/B tests
        build their arms in this order.
        """
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
        owners = {}
        for name, reference, clocked in (
            ("reference", True, False),
            ("default", False, False),
            ("clocked", False, True),
        ):
            with _switches(reference, clocked):
                owners[name] = HeldTree(solver)
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

    def test_cpu_owner_publishes_forward_kinematics(self):
        """Repair and finish a CPU owner's trees and publish the poses and twists forward kinematics gives."""
        model = _build(wp.get_device("cpu"), worlds=2)
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
        owner = HeldTree(solver)
        self.assertTrue(owner.groups)
        dt = 1.0 / 240.0
        state, expected = model.state(), model.state()
        joint_q, joint_qd = model.joint_q.numpy(), np.zeros(model.joint_dof_count, np.float32)
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        revolute = np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE))
        rng = np.random.default_rng(29)
        for joint in revolute:
            joint_q[q_start[joint]] += rng.uniform(-0.6, 0.6)
            joint_qd[qd_start[joint]] = rng.uniform(-1.0, 1.0)
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, expected)
        self.assertGreater(np.max(np.abs(state.body_q.numpy() - expected.body_q.numpy())), 1e-2)
        owner.begin(state, solver.factor, state.joint_qd, dt, True)
        np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=2e-6)
        np.testing.assert_allclose(state.body_qd.numpy(), expected.body_qd.numpy(), atol=2e-5)
        # Finish integrates a solved velocity once and publishes every coordinate and body of the next state.
        velocity = rng.uniform(-1.0, 1.0, model.joint_dof_count).astype(np.float32)
        solver.v_out.assign(velocity)
        output = model.state()
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            array = getattr(output, name)
            array.assign(np.full(array.numpy().shape, np.nan, np.float32))
        owner.finish(state, solver.factor, output, dt, False)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            self.assertTrue(np.isfinite(getattr(output, name).numpy()).all(), name)
        coordinates, dofs = q_start[revolute], qd_start[revolute]
        np.testing.assert_allclose(output.joint_qd.numpy()[dofs], velocity[dofs], atol=1e-6)
        np.testing.assert_allclose(
            output.joint_q.numpy()[coordinates], joint_q[coordinates] + dt * velocity[dofs], atol=1e-6
        )
        published = model.state()
        newton.eval_fk(model, output.joint_q, output.joint_qd, published)
        np.testing.assert_allclose(output.body_q.numpy(), published.body_q.numpy(), atol=2e-6)
        np.testing.assert_allclose(output.body_qd.numpy(), published.body_qd.numpy(), atol=2e-5)
        for group in owner.groups:
            np.testing.assert_array_equal(group.data.status.numpy(), 0)


if __name__ == "__main__":
    unittest.main()

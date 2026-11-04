# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep the physics contract of SolverFeatherPGS on Newton's public API.

The contract modules judge physics, not implementation: they build scenes, step and read through ``newton`` and
``newton.solvers`` only, so a refactor below the API changes none of them. This lint fails when one of them, or a test
helper it imports (followed through ``newton.tests``), imports from ``newton._src``. Newton's shared test harness
(:mod:`newton.tests.unittest_utils`) is infrastructure, not the solver, and is not followed.

A new contract module joins :data:`CONTRACT`. The route matrix still selects its routes through the test-only pins
(``SolverFeatherPGS._route_pins``), which go with the cases.
"""

import ast
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent

CONTRACT = (
    "test_feather_pgs_contract_api",
    "test_feather_pgs_contract_bodies",
    "test_feather_pgs_contract_joints",
    "test_feather_pgs_contract_loops",
    "test_feather_pgs_conformance",
    "test_feather_pgs_conformance_cpu",
    "test_feather_pgs_notify_inertial",
)
"""The contract modules, by name in ``newton/tests``."""

INFRASTRUCTURE = frozenset({"unittest_utils"})
"""Shared test harness modules the lint does not follow."""


def _imports(path: Path) -> list[tuple[int, str]]:
    """``(line, module)`` of every import in ``path``, relative imports resolved against ``newton.tests``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            package = ("newton.tests", "newton")[min(node.level, 2) - 1] if node.level else ""
            module = ".".join(part for part in (package, node.module or "") if part)
            found.append((node.lineno, module))
            # ``from newton import _src`` and ``from newton.tests import helper`` name the module in the alias.
            found.extend((node.lineno, f"{module}.{alias.name}") for alias in node.names)
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) in (
            "import_module",
            "__import__",
        ):
            found.extend((node.lineno, arg.value) for arg in node.args[:1] if isinstance(arg, ast.Constant))
    return found


def private_imports(name: str, root: Path = TESTS, seen: set[str] | None = None) -> list[str]:
    """Each import of ``newton._src`` reached from the test module ``name`` of ``root``, as ``module:line imported``."""
    seen = set() if seen is None else seen
    if name in seen or name in INFRASTRUCTURE:
        return []
    seen.add(name)
    path = root / f"{name}.py"
    found = []
    for line, module in _imports(path):
        if module == "newton._src" or module.startswith("newton._src."):
            found.append(f"{name}:{line} {module}")
        elif module.startswith("newton.tests."):
            helper = module.split(".")[2]
            if (root / f"{helper}.py").is_file():
                found.extend(private_imports(helper, root, seen))
    return found


class TestFeatherPGSContractLint(unittest.TestCase):
    def test_contract_imports_only_the_public_api(self):
        """No contract module, nor a helper it imports, imports from ``newton._src``."""
        for name in CONTRACT:
            with self.subTest(module=name):
                self.assertTrue((TESTS / f"{name}.py").is_file(), f"{name} is not a test module")
                self.assertEqual(private_imports(name), [])

    def test_lint_sees_a_private_import(self):
        """Report a private import made directly, relatively and through a helper."""
        sources = {
            "_lint_direct": "from newton._src.solvers.feather_pgs import placement\n",
            "_lint_relative": "from .._src import solvers\n",
            "_lint_helper": "import newton\nfrom newton.tests import _lint_direct\n",
            "_lint_dynamic": 'import importlib\nimportlib.import_module("newton._src.sim")\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, source in sources.items():
                (root / f"{name}.py").write_text(source, encoding="utf-8")
            for name in sources:
                with self.subTest(module=name):
                    self.assertTrue(private_imports(name, root), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)

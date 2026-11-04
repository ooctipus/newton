# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The FeatherPGS package's modules import one another without a cycle.

The graph is read from the source, without importing it: an edge for every import of a package module, at module level
or inside a function, since a deferred import that breaks a cycle at import time still couples the two modules both
ways. A module importing a subpackage depends on its ``__init__``; ``from . import name`` depends on the module
``name`` when it is one.
"""

import ast
import unittest
from pathlib import Path

import newton._src.solvers.feather_pgs as feather_pgs

_ROOT = Path(feather_pgs.__file__).resolve().parent
_PACKAGE = feather_pgs.__name__


def _modules() -> dict[str, Path]:
    """``{dotted name: path}`` of every module of the package, ``__init__`` as its package's name."""
    modules = {}
    for path in sorted(_ROOT.rglob("*.py")):
        parts = path.relative_to(_ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join((_PACKAGE, *parts))] = path
    return modules


def _imports(name: str, path: Path, modules: dict[str, Path]) -> set[str]:
    """The package modules ``name`` imports."""
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    targets = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                source = f"{base}.{node.module}" if node.module else base
            else:
                source = node.module or ""
            for alias in node.names:
                # ``from package import module`` imports the module, ``from module import name`` the module.
                targets.add(f"{source}.{alias.name}" if f"{source}.{alias.name}" in modules else source)
    # A module's own packages run before it whatever it imports: they are not its imports.
    ancestors = {name.rsplit(".", depth)[0] for depth in range(1, name.count(".") + 1)}
    return {target for target in targets if target in modules and target != name and target not in ancestors}


def import_graph() -> dict[str, set[str]]:
    """``{module: the package modules it imports}``."""
    modules = _modules()
    return {name: _imports(name, path, modules) for name, path in modules.items()}


def cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """The strongly connected components of more than one module (Tarjan), each sorted."""
    index, low, stack, on_stack, found = {}, {}, [], set(), []

    def visit(node):
        # Iterative Tarjan: (node, iterator over its successors).
        work = [(node, iter(sorted(graph[node])))]
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        while work:
            current, successors = work[-1]
            for successor in successors:
                if successor not in index:
                    index[successor] = low[successor] = len(index)
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, iter(sorted(graph[successor]))))
                    break
                if successor in on_stack:
                    low[current] = min(low[current], index[successor])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[current])
                if low[current] == index[current]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == current:
                            break
                    if len(component) > 1:
                        found.append(sorted(component))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    return sorted(found)


class TestFeatherPGSImportGraph(unittest.TestCase):
    def test_graph_reads_the_package(self):
        """The graph holds the package's modules and their imports (a reader that found nothing passes trivially)."""
        graph = import_graph()
        self.assertIn(f"{_PACKAGE}.solver_feather_pgs", graph)
        self.assertIn(f"{_PACKAGE}.factor.walk", graph[f"{_PACKAGE}.tree_factor"])

    def test_cycles_are_found(self):
        """The component search reports a cycle through deferred edges, and none in a chain."""
        self.assertEqual(cycles({"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}), [["a", "b", "c"]])
        self.assertEqual(cycles({"a": {"b"}, "b": {"c"}, "c": set()}), [])

    def test_no_import_cycle(self):
        """No module of the package imports, directly or through others, a module that imports it."""
        self.assertEqual(cycles(import_graph()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

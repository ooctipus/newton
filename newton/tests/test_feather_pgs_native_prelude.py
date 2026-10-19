# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose FeatherPGS native sweeps from declared device preludes only.

A native sweep body declares the device helpers it calls, such as the paired
friction law, and ``device_preludes.cuda_native`` places them after the body's
one architecture guard. These tests check that layout on every native snippet
of the point-stream inventory and scan the package for text splices of
preludes or guards elsewhere.
"""

import ast
import importlib
import pkgutil
import unittest
from pathlib import Path

from newton._src.solvers.feather_pgs import device_preludes, friction, mf_gs_source
from newton._src.solvers.feather_pgs.device_preludes import ARCHITECTURE_GUARD, DevicePrelude, cuda_native

_PACKAGE = "newton._src.solvers.feather_pgs"
# Helper names a body may call only after declaring the prelude that defines them.
_HELPERS = {
    "friction_pair_candidate(": "friction_pair",
    "friction_pair_metric(": "friction_pair_cached",
    "friction_pair_from_metric(": "friction_pair_cached",
}


def _modules():
    package = importlib.import_module(_PACKAGE)
    return {
        info.name: importlib.import_module(f"{_PACKAGE}.{info.name}") for info in pkgutil.iter_modules(package.__path__)
    }


def _preludes(modules):
    """Return the distinct module-level preludes of the package and every module-level name bound to one."""
    preludes, names = {}, set()
    for module in modules.values():
        for key, value in vars(module).items():
            if isinstance(value, DevicePrelude):
                preludes[id(value)] = value
                names.add(key)
    return list(preludes.values()), names


def _native_snippets(kernel):
    """Return the native snippets of the user functions a kernel calls, directly or through user functions."""
    snippets, pending, seen = [], [kernel.adj], set()
    while pending:
        adj = pending.pop()
        if id(adj) in seen:
            continue
        seen.add(id(adj))
        for function in adj.get_references()[2]:
            if function.is_builtin():
                continue
            for overload in {**function.user_overloads, **function.user_templates}.values():
                if overload.native_snippet:
                    snippets.append(overload.native_snippet)
                pending.append(overload.adj)
            if function.native_snippet:
                snippets.append(function.native_snippet)
    return snippets


def split_preludes(snippet, preludes):
    """Return ``(lead, declared prelude names, body)`` of a snippet composed by ``cuda_native``.

    Raises:
        AssertionError: The snippet does not open with one guard after an empty or newline lead.
    """
    lead, guard, rest = snippet.partition(ARCHITECTURE_GUARD)
    if not guard or lead not in ("", "\n"):
        raise AssertionError("the snippet does not open with its architecture guard")
    names = []
    matched = True
    while matched:
        matched = False
        # Longest first: no prelude may hide another one that it begins with.
        for prelude in sorted(preludes, key=lambda item: len(item.source), reverse=True):
            if rest.startswith(prelude.source + "\n"):
                names.append(prelude.name)
                rest = rest[len(prelude.source) + 1 :]
                matched = True
                break
    return lead, names, rest


class TestDevicePreludeComposition(unittest.TestCase):
    """The composition rule itself and the declared preludes of the package."""

    @classmethod
    def setUpClass(cls):
        cls.preludes, _ = _preludes(_modules())

    def test_cuda_native_places_preludes_after_one_guard(self):
        """Compose the lead, one guard, each prelude and its newline in order, then the body."""
        first, second = DevicePrelude("first", "    int a;"), DevicePrelude("second", "    int b;")
        snippet = cuda_native("    body();\n#endif\n", preludes=(first, second))
        self.assertEqual(snippet, "\n#if defined(__CUDA_ARCH__)\n    int a;\n    int b;\n    body();\n#endif\n")
        self.assertEqual(cuda_native("x", lead=""), "#if defined(__CUDA_ARCH__)\nx")

    def test_preludes_are_named_uniquely(self):
        """Give every declared prelude its own name, so a composed snippet names its preludes unambiguously."""
        names = [prelude.name for prelude in self.preludes]
        self.assertEqual(len(names), len(set(names)), names)
        self.assertIn(friction.FRICTION_PAIR, self.preludes)
        self.assertIn(friction.FRICTION_PAIR_CACHED, self.preludes)

    def test_dense_tile_entry_precedes_the_friction_law(self):
        """Keep the dense-tile entry trap ahead of the friction law in the dense-tile variant."""
        source = mf_gs_source.get_source(64, 32, 12, "120", dense_support_tiles=True).native
        lead, names, body = split_preludes(source, self.preludes)
        self.assertEqual((lead, names), ("\n", ["dense_tile_entry", "friction_pair"]))
        plain = mf_gs_source.get_source(64, 32, 12, "120").native
        self.assertEqual(split_preludes(plain, self.preludes)[1], ["friction_pair"])
        self.assertNotIn("trap", body)


class TestNativeSweepLayout(unittest.TestCase):
    """Every native snippet of the point-stream inventory that declares a prelude keeps the composed layout."""

    @classmethod
    def setUpClass(cls):
        from newton.tests import test_feather_pgs_point_streams as pins  # noqa: PLC0415

        cls.preludes, _ = _preludes(_modules())
        cls.snippets = {}
        for name, kernel in pins._factory_kernels().items():
            for snippet in _native_snippets(kernel):
                cls.snippets.setdefault(snippet, name)

    def test_inventory_reaches_every_prelude_site(self):
        """Reach the friction law in both forms, so the layout checks below cover their sweeps."""
        text = "".join(self.snippets)
        self.assertIn(friction.FRICTION_PAIR.source, text)
        self.assertIn(friction.FRICTION_PAIR_CACHED.source, text)

    def test_preludes_follow_one_guard(self):
        """Open every snippet holding a prelude with one guard and its preludes, then a body without preludes."""
        checked = 0
        for snippet, kernel in self.snippets.items():
            if not any(prelude.source in snippet for prelude in self.preludes):
                continue
            checked += 1
            with self.subTest(kernel=kernel):
                self.assertEqual(snippet.count(ARCHITECTURE_GUARD), 1)
                _, names, body = split_preludes(snippet, self.preludes)
                self.assertTrue(names)
                for prelude in self.preludes:
                    self.assertNotIn(prelude.source, body, f"{prelude.name} is spliced into the body")
                for helper, prelude in _HELPERS.items():
                    if helper in body:
                        self.assertIn(prelude, names, f"the body calls {helper} without declaring {prelude}")
        self.assertGreater(checked, 0)

    def test_helper_calls_declare_their_prelude(self):
        """Declare the defining prelude in every snippet that calls a friction helper."""
        for snippet, kernel in self.snippets.items():
            calls = [helper for helper in _HELPERS if helper in snippet]
            if not calls:
                continue
            with self.subTest(kernel=kernel):
                _, names, _ = split_preludes(snippet, self.preludes)
                self.assertTrue(any(_HELPERS[helper] in names for helper in calls))


class TestNoPreludeSplices(unittest.TestCase):
    """Join prelude text to native bodies in ``device_preludes`` only."""

    def test_no_module_splices_a_prelude_or_guard(self):
        """Find no guard replace, code concatenated after an opening guard or prelude read outside the helper."""
        modules = _modules()
        _, prelude_names = _preludes(modules)
        helper_file = Path(device_preludes.__file__).resolve()
        violations = []
        for module in modules.values():
            path = Path(module.__file__).resolve()
            if path == helper_file:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                location = f"{path.name}:{getattr(node, 'lineno', 0)}"
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "replace"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and "__CUDA_ARCH__" in node.args[0].value
                ):
                    violations.append(f"{location}: replace() at an architecture guard")
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    # A guard followed by code held in a variable is the splice idiom cuda_native replaces.
                    if (
                        isinstance(node.left, ast.Constant)
                        and isinstance(node.left.value, str)
                        and node.left.value.endswith(ARCHITECTURE_GUARD)
                        and isinstance(node.right, (ast.Name, ast.Attribute))
                    ):
                        violations.append(f"{location}: code concatenated after an opening guard")
                    for operand in (node.left, node.right):
                        if isinstance(operand, ast.Name) and operand.id in prelude_names:
                            violations.append(f"{location}: prelude {operand.id} concatenated")
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "source"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in prelude_names
                ):
                    violations.append(f"{location}: prelude source of {node.value.id} read")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()

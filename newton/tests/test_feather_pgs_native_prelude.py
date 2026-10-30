# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose FeatherPGS native sources from declared device preludes only.

A native body declares the device helpers it calls, such as a law family's
struct, a lane group or the patch ring, and ``device_preludes.cuda_native``
places them after the body's one architecture guard; ``dual_native`` places
their CPU twins after an ``#else`` for a body that also runs on the CPU. These
tests check that layout on every native snippet of the point-stream inventory
and scan the package for text splices of preludes or guards elsewhere. No
snippet compiles to nothing on the CPU: it runs there, or a CPU build of it
fails (``device_preludes.CUDA_ONLY``).
"""

import ast
import importlib
import pkgutil
import unittest
from pathlib import Path

from newton._src.solvers.feather_pgs import device_preludes
from newton._src.solvers.feather_pgs.contact_laws import CONTACT_LAW
from newton._src.solvers.feather_pgs.device_preludes import (
    ARCHITECTURE_GUARD,
    CUDA_ONLY,
    DevicePrelude,
    cuda_native,
    device_branches,
    dual_native,
    lane_group,
)
from newton._src.solvers.feather_pgs.law_source import LawFamily
from newton._src.solvers.feather_pgs.projection_laws import PROJECTION_LAW
from newton._src.solvers.feather_pgs.row_laws import ROW_LAW

_PACKAGE = "newton._src.solvers.feather_pgs"
# Law families whose structs native sweeps call (``Family::law(...)``).
_FAMILIES = (CONTACT_LAW, ROW_LAW, PROJECTION_LAW)
# Helper names a body may call only after declaring the prelude that defines them.
_HELPERS = {
    **{f"{family.name}::": family.name for family in _FAMILIES},
    "RING_LOAD(": "ring",
    "RING_ANY(": "ring",
}
# The lane groups the package composes: warp groups, block groups and the run-time width of lane_minimum.
_LANE_GROUPS = tuple(lane_group(lanes) for lanes in (1, 2, 4, 8, 16, 32, 64, 96, 128, "lanes"))


def _modules():
    package = importlib.import_module(_PACKAGE)
    return {
        info.name: importlib.import_module(f"{_PACKAGE}.{info.name}") for info in pkgutil.iter_modules(package.__path__)
    }


def _preludes(modules):
    """Return the distinct module-level preludes of the package and every module-level name bound to one.

    A module-level law family contributes its prelude.
    """
    preludes, names = {}, set()
    for module in modules.values():
        for key, value in vars(module).items():
            prelude = value.prelude() if isinstance(value, LawFamily) else value
            if isinstance(prelude, DevicePrelude):
                preludes[id(prelude)] = prelude
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


def _splices(module_file, text, prelude_names):
    """Return the prelude and guard splices of one module's source, which only ``device_preludes`` may make.

    Prelude text reaches a body only through ``cuda_native``: no other module names the guard constant,
    renders a law family (``law_source`` renders each once, to declare its prelude), reads a prelude's
    ``source`` or concatenates a prelude, and none replaces or extends an opening guard.

    Args:
        module_file: File name of the module, for the messages and the producer's own module.
        text: Python source of the module.
        prelude_names: Module-level names bound to a prelude anywhere in the package.
    """
    violations = []
    for node in ast.walk(ast.parse(text)):
        location = f"{module_file}:{getattr(node, 'lineno', 0)}"
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
            and (
                (isinstance(node.value, ast.Name) and node.value.id in prelude_names)
                or "prelude" in ast.unparse(node.value).lower()
            )
        ):
            violations.append(f"{location}: prelude source of {ast.unparse(node.value)} read")
        if (
            (isinstance(node, ast.Name) and node.id == "ARCHITECTURE_GUARD")
            or (isinstance(node, ast.Attribute) and node.attr == "ARCHITECTURE_GUARD")
            or (isinstance(node, ast.alias) and node.name == "ARCHITECTURE_GUARD")
        ):
            violations.append(f"{location}: the architecture guard named outside the helper")
        if isinstance(node, ast.Call) and module_file != "law_source.py":
            function = node.func
            called = function.id if isinstance(function, ast.Name) else getattr(function, "attr", None)
            if called == "render_family":
                violations.append(f"{location}: a law family rendered outside its prelude")
    return violations


def split_preludes(snippet, preludes):
    """Return ``(lead, declared prelude names, body, dual)`` of a snippet composed by ``cuda_native`` or ``dual_native``.

    A ``dual_native`` snippet follows its CUDA preludes with ``#else``, their CPU twins in the same order and
    ``#endif``.

    Raises:
        AssertionError: The snippet does not open with one guard after an empty or newline lead, or a dual
            snippet's CPU twins do not repeat its preludes.
    """
    lead, guard, rest = snippet.partition(ARCHITECTURE_GUARD)
    if not guard or lead not in ("", "\n"):
        raise AssertionError("the snippet does not open with its architecture guard")
    declared = []
    matched = True
    while matched:
        matched = False
        # Longest first: no prelude may hide another one that it begins with.
        for prelude in sorted(preludes, key=lambda item: len(item.source), reverse=True):
            if rest.startswith(prelude.source + "\n"):
                declared.append(prelude)
                rest = rest[len(prelude.source) + 1 :]
                matched = True
                break
    dual = rest.startswith("#else\n")
    if dual:
        twins = "#else\n" + "".join(f"{prelude.cpu}\n" for prelude in declared) + "#endif\n"
        if any(prelude.cpu is None for prelude in declared) or not rest.startswith(twins):
            raise AssertionError("the CPU twins of a dual snippet do not follow its preludes")
        rest = rest[len(twins) :]
    return lead, [prelude.name for prelude in declared], rest, dual


def cpu_text(snippet, *, cuda=False):
    """Return the statements a CPU build of ``snippet`` compiles (a CUDA build with ``cuda``): that side of its
    architecture guards (other conditionals keep both sides), without comments or blank lines. A CUDA-only body
    yields its ``#error`` on the CPU."""
    lines, stack = [], []
    for line in snippet.splitlines():
        directive = line.strip().replace(" ", "")
        if directive.startswith(("#ifdefined(__CUDA_ARCH__)", "#ifdef__CUDA_ARCH__")):
            stack.append(cuda)
        elif directive.startswith(("#if!defined(__CUDA_ARCH__)", "#ifndef__CUDA_ARCH__")):
            stack.append(not cuda)
        elif directive.startswith("#if"):
            stack.append(None)
        elif directive.startswith("#else"):
            stack[-1] = None if stack[-1] is None else not stack[-1]
        elif directive.startswith("#endif"):
            stack.pop()
        elif all(state is not False for state in stack):
            code = line.split("//", 1)[0].strip()
            if code:
                lines.append(code)
    return "\n".join(lines)


class TestDevicePreludeComposition(unittest.TestCase):
    """The composition rule itself and the declared preludes of the package."""

    @classmethod
    def setUpClass(cls):
        cls.preludes, _ = _preludes(_modules())

    def test_cuda_native_places_preludes_after_one_guard(self):
        """Compose the lead, one guard, each prelude and its newline in order, then the body."""
        first, second = DevicePrelude("first", "    int a;"), DevicePrelude("second", "    int b;")
        snippet = cuda_native("    body();\n#endif\n", preludes=(first, second))
        self.assertEqual(
            snippet,
            "\n#if defined(__CUDA_ARCH__)\n    int a;\n    int b;\n    body();\n#else\n" + CUDA_ONLY + "#endif\n",
        )
        self.assertEqual(
            cuda_native("x\n#endif", lead=""), "#if defined(__CUDA_ARCH__)\nx\n#else\n" + CUDA_ONLY + "#endif\n"
        )
        with self.assertRaisesRegex(ValueError, "#endif"):
            cuda_native("x", lead="")

    def test_no_branch_compiles_to_nothing_on_the_cpu(self):
        """Give a CUDA-only body a CPU branch that fails the build, and refuse an empty CPU branch."""
        self.assertEqual(cpu_text(cuda_native("    body();\n#endif\n")).strip(), CUDA_ONLY.strip())
        self.assertEqual(cpu_text(device_branches("    a();\n", "    b();\n")).strip(), "b();")
        for empty in ("", "\n", "    \n"):
            with self.subTest(cpu=empty), self.assertRaisesRegex(ValueError, "CPU statements"):
                device_branches("    a();\n", empty)

    def test_dual_native_places_cpu_twins_after_the_preludes(self):
        """Compose the guard, the CUDA preludes, ``#else``, their CPU twins and ``#endif``, then the shared body."""
        first = DevicePrelude("first", "    int a = 1;", "    int a = 0;")
        second = DevicePrelude("second", "    int b = 1;", "    int b = 0;")
        snippet = dual_native("    body(a, b);\n", preludes=(first, second))
        self.assertEqual(
            snippet,
            "\n#if defined(__CUDA_ARCH__)\n    int a = 1;\n    int b = 1;\n#else\n    int a = 0;\n    int b = 0;\n"
            "#endif\n    body(a, b);\n",
        )
        self.assertEqual(
            split_preludes(snippet, (first, second)), ("\n", ["first", "second"], "    body(a, b);\n", True)
        )
        with self.assertRaisesRegex(ValueError, "CPU twin"):
            dual_native("x", preludes=(DevicePrelude("cuda_only", "    int c;"),))

    def test_lane_groups_differ_only_in_their_width(self):
        """Name each lane group by its width, reject widths it cannot hold and keep one text per width."""
        self.assertIs(lane_group(16), lane_group(16))
        self.assertEqual(lane_group(16).name, "lane_group_16")
        self.assertEqual(lane_group(16).cpu, lane_group(64).cpu)
        self.assertEqual(lane_group(16).source.replace("LANES = 16,", "LANES = 64,"), lane_group(64).source)
        for lanes in (0, 3, 24, 48):
            with self.subTest(lanes=lanes), self.assertRaises(ValueError):
                lane_group(lanes)

    def test_preludes_are_named_uniquely(self):
        """Give every declared prelude its own name, so a composed snippet names its preludes unambiguously."""
        names = [prelude.name for prelude in self.preludes]
        self.assertEqual(len(names), len(set(names)), names)
        for family in _FAMILIES:
            self.assertIn(family.prelude(), self.preludes)


class TestNativeSweepLayout(unittest.TestCase):
    """Every native snippet of the point-stream inventory that declares a prelude keeps the composed layout."""

    @classmethod
    def setUpClass(cls):
        from newton.tests import test_feather_pgs_point_streams as pins  # noqa: PLC0415

        cls.preludes, _ = _preludes(_modules())
        cls.preludes += list(_LANE_GROUPS)
        cls.snippets = {}
        for name, kernel in pins._factory_kernels().items():
            for snippet in _native_snippets(kernel):
                cls.snippets.setdefault(snippet, name)

    def test_inventory_reaches_every_prelude_site(self):
        """Reach every law family's struct, so the layout checks below cover the sweeps that call it."""
        text = "".join(self.snippets)
        for family in _FAMILIES:
            self.assertIn(family.prelude().source, text, family.name)

    def test_preludes_follow_one_guard(self):
        """Open every snippet holding a prelude with one guard and its preludes, then a body without preludes.

        A CUDA-only body keeps that one guard; a body ``dual_native`` composes may guard its own CUDA code.
        """
        checked = 0
        for snippet, kernel in self.snippets.items():
            if not any(prelude.source in snippet for prelude in self.preludes):
                continue
            checked += 1
            with self.subTest(kernel=kernel):
                _, names, body, dual = split_preludes(snippet, self.preludes)
                if not dual:
                    self.assertEqual(snippet.count(ARCHITECTURE_GUARD), 1)
                self.assertTrue(names)
                for prelude in self.preludes:
                    self.assertNotIn(prelude.source, body, f"{prelude.name} is spliced into the body")
                for helper, prelude in _HELPERS.items():
                    if helper in body:
                        self.assertIn(prelude, names, f"the body calls {helper} without declaring {prelude}")
        self.assertGreater(checked, 0)

    def test_no_snippet_compiles_to_nothing_on_the_cpu(self):
        """Give every native snippet of the inventory and of the package's modules a CPU build: CPU statements, or
        the ``#error`` of a CUDA-only body (BACKBONE rule 6)."""
        from newton.tests import test_feather_pgs_point_streams as pins  # noqa: PLC0415

        snippets = {}
        for name, kernel in pins._inventory().items():
            for snippet in _native_snippets(kernel):
                snippets.setdefault(snippet, name)
        for module_name, module in _modules().items():
            for key, value in vars(module).items():
                snippet = getattr(value, "native_snippet", None)
                if isinstance(snippet, str):
                    snippets.setdefault(snippet, f"{module_name}.{key}")
        self.assertGreater(len(snippets), 100)
        for snippet, owner in snippets.items():
            if not cpu_text(snippet, cuda=True):
                continue  # Empty on both devices: no device holds the other's code.
            with self.subTest(owner=owner):
                self.assertTrue(cpu_text(snippet), "the snippet compiles to nothing on the CPU")

    def test_helper_calls_declare_their_prelude(self):
        """Declare the defining prelude in every snippet that calls a law or a ring helper."""
        for snippet, kernel in self.snippets.items():
            calls = [helper for helper in _HELPERS if helper in snippet]
            if not calls:
                continue
            with self.subTest(kernel=kernel):
                _, names, _, _ = split_preludes(snippet, self.preludes)
                self.assertTrue(any(_HELPERS[helper] in names for helper in calls))


class TestNoPreludeSplices(unittest.TestCase):
    """Join prelude text to native bodies in ``device_preludes`` only."""

    def test_no_module_splices_a_prelude_or_guard(self):
        """Find no prelude or guard splice in any module of the package but the composition helper."""
        modules = _modules()
        _, prelude_names = _preludes(modules)
        helper_file = Path(device_preludes.__file__).resolve()
        violations = []
        for module in modules.values():
            path = Path(module.__file__).resolve()
            if path != helper_file:
                violations += _splices(path.name, path.read_text(encoding="utf-8"), prelude_names)
        self.assertEqual(violations, [])

    def test_scan_finds_every_splice_idiom(self):
        """Flag each way of joining prelude text or a guard to a body by hand, and not a composed body."""
        idioms = {
            "guard replace": 'snippet = snippet.replace("#if defined(__CUDA_ARCH__)", "#if defined(__CUDA_ARCH__)\\n" + law)',
            "guard then code": 'snippet = "\\n#if defined(__CUDA_ARCH__)\\n" + law',
            "prelude concatenated": "snippet = RING + body",
            "prelude source": "snippet = RING.source + body",
            "guard then a family's source": 'snippet = "#if defined(__CUDA_ARCH__)\\n" + render_family(LAW) + body',
            "f-string of a family's source": 'snippet = f"#if defined(__CUDA_ARCH__)\\n{render_family(LAW)}{body}"',
            "joined guard": 'snippet = "".join([ARCHITECTURE_GUARD, law, body])',
            "imported guard": "from .device_preludes import ARCHITECTURE_GUARD",
            "declared prelude source": "snippet = policy.preludes[0].source + body",
        }
        for idiom, text in idioms.items():
            with self.subTest(idiom=idiom):
                self.assertTrue(_splices("module.py", text, {"RING"}))
        composed = "snippet = cuda_native(body, preludes=(RING, *LAW.preludes()))"
        self.assertEqual(_splices("module.py", composed, {"RING"}), [])
        declared = "self._prelude = DevicePrelude(self.name, render_family(self), cpu=render_family(self, cpu=True))"
        self.assertEqual(_splices("law_source.py", declared, set()), [])


if __name__ == "__main__":
    unittest.main()

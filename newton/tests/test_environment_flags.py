# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep the declared environment variables equal to the variables Newton reads."""

import ast
import re
import unittest
from pathlib import Path

import newton
from newton._src.core import environment_flags as core_flags
from newton._src.solvers.feather_pgs import environment_flags as fpgs_flags

_SRC = Path(newton.__file__).parent / "_src"
_REGISTRIES = {
    "core.environment_flags": core_flags.NEWTON_FLAGS,
    "solvers.feather_pgs.environment_flags": fpgs_flags.FLAGS,
}
# A complete variable name, and the literal start of a constructed one (a bare prefix included).
_NAME = re.compile(r"(?:FEATHER_PGS|IL_NEWTON_FPGS|NEWTON)_[A-Z0-9](?:[A-Z0-9_]*[A-Z0-9])?")
_HEAD = re.compile(r"(?:FEATHER_PGS|IL_NEWTON_FPGS|NEWTON)_[A-Z0-9_]*")
_GETTERS = ("os.environ.get", "os.getenv")
_RETIRED_ORDER = [
    "FEATHER_PGS_SINGLE_FACTOR",
    "FEATHER_PGS_BODY_BASIS_ROWS",
    "FEATHER_PGS_SPARSE_PACKETS",
    "FEATHER_PGS_SPARSE_PAIRED_GS",
    "FEATHER_PGS_COUPLED_CONTACT",
    "FEATHER_PGS_COUPLED_JACOBI",
]


def _parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _in_dunder_all(node, parents):
    parent = parents.get(node)
    while isinstance(parent, (ast.List, ast.Tuple)):
        owner = parents.get(parent)
        if isinstance(owner, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = owner.targets if isinstance(owner, ast.Assign) else [owner.target]
            return any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets)
        parent = owner
    return False


def _reads_environment(node, parents):
    """Return whether ``node`` is the key of an ``os.environ`` or ``os.getenv`` read."""
    parent = parents.get(node)
    if isinstance(parent, ast.Call):
        return bool(parent.args) and parent.args[0] is node and ast.unparse(parent.func) in _GETTERS
    if isinstance(parent, ast.Subscript):
        return parent.slice is node and isinstance(parent.ctx, ast.Load) and ast.unparse(parent.value) == "os.environ"
    if isinstance(parent, ast.Compare):
        return (
            parent.left is node
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and ast.unparse(parent.comparators[0]) == "os.environ"
        )
    return False


def _literal_strings(node):
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(
        isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.elts
    ):
        return [item.value for item in node.elts]
    return None


def _values(name, parents, tree):
    """Return the literal strings a name can hold, or None when they cannot be listed."""
    scope = parents.get(name)
    while scope is not None:
        if isinstance(scope, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for generator in scope.generators:
                if isinstance(generator.target, ast.Name) and generator.target.id == name.id:
                    return _literal_strings(generator.iter)
        if isinstance(scope, ast.For) and isinstance(scope.target, ast.Name) and scope.target.id == name.id:
            return _literal_strings(scope.iter)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [arg.arg for arg in scope.args.posonlyargs + scope.args.args]
            if name.id not in positional and name.id not in [arg.arg for arg in scope.args.kwonlyargs]:
                return None
            values = []
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                callee = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
                if callee != scope.name:
                    continue
                argument = None
                if name.id in positional and positional.index(name.id) < len(call.args):
                    argument = call.args[positional.index(name.id)]
                for keyword in call.keywords:
                    if keyword.arg == name.id:
                        argument = keyword.value
                if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
                    return None
                values.append(argument.value)
            return values or None
        scope = parents.get(scope)
    return None


def _head(node):
    """Return the literal start of a name construction such as ``"FEATHER_PGS_" + x``, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and _HEAD.fullmatch(node.value):
        return node.value
    return None


def _names(node, parents, tree):
    """Return the variable names an expression spells, or None when it spells a name that cannot be listed.

    A literal name and a name built from a literal prefix and a listable suffix count;
    so does a variable that keys an environment read and holds listable names.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parent = parents.get(node)
        heads_construction = isinstance(parent, ast.JoinedStr) or (
            isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add) and parent.left is node
        )
        if (heads_construction and _head(node)) or _in_dunder_all(node, parents):
            return []
        if _NAME.fullmatch(node.value):
            return [node.value]
        return None if _head(node) else []
    head, suffix = None, None
    if isinstance(node, ast.JoinedStr) and node.values and _head(node.values[0]):
        head = node.values[0].value
        tail = node.values[1:]
        suffix = tail[0].value if len(tail) == 1 and isinstance(tail[0], ast.FormattedValue) else None
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and _head(node.left):
        head, suffix = node.left.value, node.right
    elif isinstance(node, ast.Name) and _reads_environment(node, parents):
        return [value for value in _values(node, parents, tree) or () if _NAME.fullmatch(value)]
    if head is None:
        return []
    values = _values(suffix, parents, tree) if isinstance(suffix, ast.Name) else None
    return None if values is None else [head + value for value in values]


def _scan_source(source, module):
    """Return the ``(name, module)`` pairs one source reads, those it only cites, and unlistable names."""
    tree = ast.parse(source)
    parents = _parents(tree)
    reads, cited, unresolved = set(), set(), []
    for node in ast.walk(tree):
        names = _names(node, parents, tree)
        if names is None:
            unresolved.append(f"{module}:{node.lineno}")
            continue
        (reads if _reads_environment(node, parents) else cited).update((name, module) for name in names)
    return reads, cited, unresolved


def _scan_tree():
    reads, cited, unresolved = set(), set(), []
    for path in sorted(_SRC.rglob("*.py")):
        relative = path.relative_to(_SRC)
        module = ".".join(relative.with_suffix("").parts)
        # In-tree tests set variables; the registries read them by declaration.
        if module in _REGISTRIES or "tests" in relative.parts:
            continue
        source_reads, source_cited, source_unresolved = _scan_source(path.read_text(encoding="utf-8"), module)
        reads |= source_reads
        cited |= source_cited
        unresolved += source_unresolved
    return reads, cited, unresolved


def _declared():
    declared = {}
    for registry in _REGISTRIES.values():
        declared.update(registry)
    return declared


class TestEnvironmentFlagRegistry(unittest.TestCase):
    def test_declarations_match_read_sites(self):
        """Require every FEATHER_PGS_/NEWTON_ name under newton._src to be declared with its readers, and back."""
        reads, cited, unresolved = _scan_tree()
        self.assertEqual(unresolved, [], "build environment names from literal tuples or literal call arguments")
        declared = _declared()
        self.assertEqual(
            sorted({name for name, _ in reads | cited} - set(declared)),
            [],
            "declare these variables in an environment_flags module",
        )
        readers = {}
        for name, module in reads:
            readers.setdefault(name, set()).add(module)
        for name, flag in declared.items():
            with self.subTest(name=name):
                if flag.owner == "isaac_lab":
                    self.assertEqual(
                        readers.get(name, set()), set(), "Newton reads this name; declare it with owner 'newton'"
                    )
                    continue
                expected = set(flag.readers) - set(_REGISTRIES)
                if flag.kind != "retired":
                    self.assertTrue(expected, "only a retired name may be read by the registry alone")
                self.assertEqual(readers.get(name, set()), expected, "readers must list the modules that read it")

    def test_declarations_are_well_formed(self):
        """Keep each family in its module, each meaning on one line and each reader resolvable."""
        self.assertFalse(set(core_flags.NEWTON_FLAGS) & set(fpgs_flags.FLAGS))
        for registry, prefixes in (
            (core_flags.NEWTON_FLAGS, core_flags.NEWTON_PREFIXES),
            (fpgs_flags.FLAGS, fpgs_flags.PREFIXES),
        ):
            for name, flag in registry.items():
                with self.subTest(name=name):
                    self.assertTrue(name.startswith(prefixes))
                    self.assertTrue(flag.meaning.endswith(".") and "\n" not in flag.meaning)
                    self.assertTrue(flag.readers)
                    for reader in flag.readers:
                        if flag.owner == "isaac_lab":
                            self.assertTrue(reader.startswith("isaaclab"), reader)
                        else:
                            self.assertTrue((_SRC / Path(*reader.split("."))).with_suffix(".py").is_file(), reader)
        retired = [name for name, flag in fpgs_flags.FLAGS.items() if flag.kind == "retired"]
        self.assertEqual(retired, _RETIRED_ORDER)

    def test_scan_separates_reads_from_citations(self):
        """List literal and constructed names, split environment reads from citations, refuse the rest."""
        source = "\n".join(
            (
                "import os",
                'os.environ.get("FEATHER_PGS_LITERAL")',
                'any(os.environ.get(f"FEATHER_PGS_{option}") for option in ("ONE", "TWO"))',
                'for suffix in ("THREE",):',
                '    os.environ.get("NEWTON_" + suffix)',
                "def enabled(option, *, default=True):",
                '    return os.environ.get(f"FEATHER_PGS_{option}")',
                'enabled("FOUR")',
                'enabled(option="FIVE")',
                'for flag in ("NEWTON_LOOPED",):',
                "    os.getenv(flag)",
                'present = "NEWTON_PRESENT" in os.environ',
                'value = os.environ["IL_NEWTON_FPGS_INDEXED"]',
                'os.environ["NEWTON_WRITTEN"] = "1"',
                'reason = gate("NEWTON_CITED")',
                '__all__ = ["NEWTON_EXPORTED_CONSTANT"]',
                'message = "NEWTON_LITERAL_IN_TEXT is not a name"',
            )
        )
        reads, cited, unresolved = _scan_source(source, "example")
        self.assertEqual(unresolved, [])
        self.assertEqual(
            {name for name, _ in reads},
            {
                "FEATHER_PGS_LITERAL",
                "FEATHER_PGS_ONE",
                "FEATHER_PGS_TWO",
                "NEWTON_THREE",
                "FEATHER_PGS_FOUR",
                "FEATHER_PGS_FIVE",
                "NEWTON_LOOPED",
                "NEWTON_PRESENT",
                "IL_NEWTON_FPGS_INDEXED",
            },
        )
        self.assertEqual({name for name, _ in cited}, {"NEWTON_LOOPED", "NEWTON_WRITTEN", "NEWTON_CITED"})
        for source in ('f"FEATHER_PGS_{name()}"', '"NEWTON_" + name.upper()', 'PREFIX = "FEATHER_PGS_"'):
            with self.subTest(source=source):
                self.assertEqual(len(_scan_source(source, "example")[2]), 1)

    def test_retired_selector_raises_only_for_one(self):
        """Raise for a retired selector set to 1, naming the first in the original order; accept other values."""
        message = (
            "^FEATHER_PGS_SINGLE_FACTOR selects a retired experiment; reproduce it from the original research checkout$"
        )
        with self.assertRaisesRegex(ValueError, message):
            fpgs_flags.reject_retired(dict.fromkeys(_RETIRED_ORDER, "1"))
        for value in ("0", "", "true"):
            with self.subTest(value=value):
                fpgs_flags.reject_retired({"FEATHER_PGS_COUPLED_CONTACT": value, "FEATHER_PGS_NOT_DECLARED": "1"})


if __name__ == "__main__":
    unittest.main()

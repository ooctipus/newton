# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep the declared environment variables equal to the variables Newton reads, and record them."""

import ast
import os
import re
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.core import environment_flags as core_flags
from newton._src.solvers.feather_pgs import environment_flags as fpgs_flags

_SRC = Path(newton.__file__).parent / "_src"
_REGISTRIES = {
    "core.environment_flags": core_flags.NEWTON_FLAGS,
    "solvers.feather_pgs.environment_flags": fpgs_flags.FLAGS,
}
_PREFIXES = ("FEATHER_PGS_", "IL_NEWTON_FPGS_", "NEWTON_")
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
    "FEATHER_PGS_DENSE_ROW_REGISTERS",
]
# Retired names that recipes still set: ignored, recorded and warned about, never rejected.
_RETIRED_IGNORED = [
    "FEATHER_PGS_ALLEGRO_KINETIC_ROWS",
    "FEATHER_PGS_CAPTURE_COUNT",
    "FEATHER_PGS_CAPTURE_DIR",
    "FEATHER_PGS_CAPTURE_KERNEL",
    "FEATHER_PGS_CAPTURE_PHASE",
    "FEATHER_PGS_CAPTURE_STEP",
    "FEATHER_PGS_CHECK_ROWS",
    "FEATHER_PGS_FRANKA_COMPACT_WORKSPACE",
    "FEATHER_PGS_G1_CHAIN_SCAN",
    "FEATHER_PGS_GROUPED_HINV_FREE",
    "FEATHER_PGS_GROUP_LANES",
    "FEATHER_PGS_INDEPENDENT_COMPONENTS",
    "FEATHER_PGS_INK_CHECK",
    "FEATHER_PGS_KUKA_CURRENT_CONTACT",
    "FEATHER_PGS_KUKA_JOINT_WORLD",
    "FEATHER_PGS_LOCAL_ROW_PACKETS",
    "FEATHER_PGS_PACKED_TREE_KINETIC_STATE",
    "FEATHER_PGS_PAIRED_GENERAL_OVERLAP",
    "FEATHER_PGS_PRISMATIC_LINEAR_STATE",
    "FEATHER_PGS_PRISMATIC_PUBLICATION",
    "FEATHER_PGS_SIMPLE_WORLD_ZERO",
    "FEATHER_PGS_SPARSE_BRANCH_SWEEPS",
    "FEATHER_PGS_SPARSE_CONTACT_BLOCK",
    "FEATHER_PGS_SPARSE_FACTOR",
    "FEATHER_PGS_SPARSE_LEVEL_UPDATE",
    "FEATHER_PGS_SPARSE_LIMIT_JACOBI",
    "FEATHER_PGS_SPARSE_METRIC_TANGENTS",
    "FEATHER_PGS_SPARSE_PARALLEL_LIMITS",
    "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS",
    "FEATHER_PGS_WORLD_ROWS",
    "FEATHER_PGS_WORLD_SCAN_PUBLICATION",
]
# The recipe variables that the first construction record (64d35720) listed as unknown: read nowhere
# in Newton or Isaac Lab.
_DEAD_RECIPE_NAMES = [
    "FEATHER_PGS_ALLEGRO_KINETIC_ROWS",
    "FEATHER_PGS_FRANKA_COMPACT_WORKSPACE",
    "FEATHER_PGS_G1_CHAIN_SCAN",
    "FEATHER_PGS_INDEPENDENT_COMPONENTS",
    "FEATHER_PGS_KUKA_CURRENT_CONTACT",
    "FEATHER_PGS_KUKA_JOINT_WORLD",
    "FEATHER_PGS_LOCAL_ROW_PACKETS",
    "FEATHER_PGS_PAIRED_GENERAL_OVERLAP",
    "FEATHER_PGS_SIMPLE_WORLD_ZERO",
    "FEATHER_PGS_SPARSE_BRANCH_SWEEPS",
    "FEATHER_PGS_WORLD_SCAN_PUBLICATION",
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
    so does a variable that keys an environment read and holds listable names. Any
    other key of an environment read (a format call, a join, a case transform or a
    variable whose values cannot be listed) cannot be checked and is refused.
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
        values = _values(node, parents, tree)
        return None if values is None else [value for value in values if _NAME.fullmatch(value)]
    if head is None:
        return None if _reads_environment(node, parents) else []
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


@contextmanager
def _environment(flags):
    """Replace every flag-prefixed variable of the process with ``flags`` for the duration."""
    with patch.dict(os.environ):
        for name in [name for name in os.environ if name.startswith(_PREFIXES)]:
            del os.environ[name]
        os.environ.update(flags)
        yield


def _cpu_model():
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    builder.add_ground_plane()
    return builder.finalize(device="cpu")


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
        self.assertEqual([name for name in retired if name not in fpgs_flags.IGNORED_RETIRED], _RETIRED_ORDER)
        self.assertEqual(sorted(fpgs_flags.IGNORED_RETIRED), _RETIRED_IGNORED)
        self.assertTrue(fpgs_flags.IGNORED_RETIRED <= set(retired))

    def test_rejected_retired_selectors_lead_the_declarations(self):
        """Declare the retired selectors that raise first, in check order, and every other name in name order."""
        names = list(fpgs_flags.FLAGS)
        self.assertEqual(names[: len(_RETIRED_ORDER)], _RETIRED_ORDER)
        rest = names[len(_RETIRED_ORDER) :]
        self.assertEqual(rest, sorted(rest))

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
        for source in (
            'f"FEATHER_PGS_{name()}"',
            '"NEWTON_" + name.upper()',
            'PREFIX = "FEATHER_PGS_"',
            # Environment reads keyed by an expression the scan cannot list.
            'import os\nos.environ.get("FEATHER_PGS_{}".format(option))',
            'import os\nos.environ.get("FEATHER_PGS_%s" % option)',
            'import os\nos.environ.get("_".join(("FEATHER", "PGS", option)))',
            'import os\nos.getenv(f"feather_pgs_{option}".upper())',
            'import os\nos.environ.get("FEATHER_PGS" + "_" + option)',
            'import os\nvalue = os.environ["_".join(("NEWTON", option))]',
            'import os\npresent = "_".join(("NEWTON", option)) in os.environ',
            "import os\nkey = make_key()\nos.environ.get(key)",
        ):
            with self.subTest(source=source):
                self.assertEqual(len(_scan_source(source, "example")[2]), 1)


class TestEnvironmentRecord(unittest.TestCase):
    def test_record_classifies_names(self):
        """Record declared values, undeclared names, retired names and Isaac Lab-owned names separately."""
        environ = {
            "FEATHER_PGS_ROWS_MASKED": "1",
            "FEATHER_PGS_NOT_DECLARED": "0",
            "FEATHER_PGS_COUPLED_JACOBI": "0",
            "IL_NEWTON_FPGS_MF_WARMSTART": "0",
            "NEWTON_NARROW_PHASE_THREADS_X": "4",
            "PATH": "/usr/bin",
        }
        self.assertEqual(
            fpgs_flags.record(environ),
            {
                "applied": {"FEATHER_PGS_ROWS_MASKED": "1", "IL_NEWTON_FPGS_MF_WARMSTART": "0"},
                "unknown": ["FEATHER_PGS_NOT_DECLARED"],
                "retired": ["FEATHER_PGS_COUPLED_JACOBI"],
                "isaac_lab": {},
            },
        )
        environ = {
            "NEWTON_NARROW_PHASE_THREADS_X": "4",
            "NEWTON_CONTACT_MATCHING": "latest",
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
            "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
            "NEWTON_UNDECLARED_EXAMPLE": "1",
            "FEATHER_PGS_ROWS_MASKED": "1",
        }
        self.assertEqual(
            core_flags.record_newton_environment(environ),
            {
                "applied": {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1"},
                "unknown": ["NEWTON_UNDECLARED_EXAMPLE"],
                "retired": ["NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD"],
                "isaac_lab": {"NEWTON_CONTACT_MATCHING": "latest", "NEWTON_NARROW_PHASE_THREADS_X": "4"},
            },
        )

    def test_retired_selector_raises_only_for_one(self):
        """Raise for a retired selector set to 1, naming the first in the original order; record other values."""
        message = (
            "^FEATHER_PGS_SINGLE_FACTOR selects a retired experiment; reproduce it from the original research checkout$"
        )
        for check in (fpgs_flags.reject_retired, fpgs_flags.record):
            with self.subTest(check=check.__name__), self.assertRaisesRegex(ValueError, message):
                check(dict.fromkeys(_RETIRED_ORDER, "1"))
        for value in ("0", "", "true"):
            with self.subTest(value=value):
                record = fpgs_flags.record({"FEATHER_PGS_COUPLED_CONTACT": value, "FEATHER_PGS_NOT_DECLARED": "1"})
                self.assertEqual(record["retired"], ["FEATHER_PGS_COUPLED_CONTACT"])
                self.assertEqual(record["unknown"], ["FEATHER_PGS_NOT_DECLARED"])

    def test_ignored_retired_selectors_record_without_raising(self):
        """Record the retired selectors that recipes still set, whatever their value, instead of raising."""
        for value in ("1", "0"):
            with self.subTest(value=value):
                environ = dict.fromkeys(_RETIRED_IGNORED, value)
                fpgs_flags.reject_retired(environ)
                self.assertEqual(
                    fpgs_flags.record(environ),
                    {"applied": {}, "unknown": [], "retired": _RETIRED_IGNORED, "isaac_lab": {}},
                )
        # An ignored selector does not mask a rejected one set beside it.
        with self.assertRaisesRegex(ValueError, "^FEATHER_PGS_COUPLED_JACOBI selects a retired experiment"):
            fpgs_flags.record({**environ, "FEATHER_PGS_COUPLED_JACOBI": "1"})

    def test_dead_recipe_names_are_retired_not_unknown(self):
        """Declare the recipe variables nothing reads as ignored retired names, never unknown ones."""
        for name in _DEAD_RECIPE_NAMES:
            with self.subTest(name=name):
                self.assertEqual(fpgs_flags.FLAGS[name].kind, "retired")
                self.assertIn(name, fpgs_flags.IGNORED_RETIRED)
        record = fpgs_flags.record(dict.fromkeys(_DEAD_RECIPE_NAMES, "1") | {"FEATHER_PGS_SPARSE_BRANCH_SWEEPS": "0"})
        self.assertEqual((record["unknown"], record["retired"]), ([], _DEAD_RECIPE_NAMES))

    def test_retired_notice_names_each_retired_name_once(self):
        """Return one warning per process for each retired name a record lists, and none for other names."""
        with patch.object(core_flags, "_WARNED_RETIRED", set()):
            first = fpgs_flags.record(
                {"FEATHER_PGS_SPARSE_BRANCH_SWEEPS": "0", "FEATHER_PGS_KUKA_JOINT_WORLD": "1", "FEATHER_PGS_X": "1"}
            )
            self.assertEqual(
                core_flags.retired_notice(first),
                "Ignoring retired environment variables that select nothing: FEATHER_PGS_KUKA_JOINT_WORLD, "
                "FEATHER_PGS_SPARSE_BRANCH_SWEEPS. Remove them from the environment; the construction record lists "
                "them under environment['retired'].",
            )
            self.assertIsNone(core_flags.retired_notice(first))
            second = fpgs_flags.record({"FEATHER_PGS_SPARSE_BRANCH_SWEEPS": "0", "FEATHER_PGS_COUPLED_JACOBI": "0"})
            self.assertIn(": FEATHER_PGS_COUPLED_JACOBI. Remove", core_flags.retired_notice(second))
            self.assertIsNone(core_flags.retired_notice(fpgs_flags.record({"FEATHER_PGS_X": "1"})))


class TestSelectionEnvironment(unittest.TestCase):
    def test_solver_reports_construction_environment(self):
        """Report the FEATHER_PGS_ environment read at construction, detached from later changes."""
        flags = {
            "FEATHER_PGS_RESOLVED_WORLDS": "1",
            "FEATHER_PGS_NOT_DECLARED": "0",
            "FEATHER_PGS_SPARSE_BRANCH_SWEEPS": "0",
            "FEATHER_PGS_COUPLED_JACOBI": "0",
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
        }
        model = _cpu_model()
        with (
            _environment(flags),
            patch.object(core_flags, "_WARNED_RETIRED", set()),
            self.assertWarnsRegex(UserWarning, "FEATHER_PGS_COUPLED_JACOBI, FEATHER_PGS_SPARSE_BRANCH_SWEEPS"),
        ):
            solver = newton.solvers.SolverFeatherPGS(model)
        expected = {
            "applied": {"FEATHER_PGS_RESOLVED_WORLDS": "1"},
            "unknown": ["FEATHER_PGS_NOT_DECLARED"],
            "retired": ["FEATHER_PGS_COUPLED_JACOBI", "FEATHER_PGS_SPARSE_BRANCH_SWEEPS"],
            "isaac_lab": {},
        }
        selected = solver.get_path_selection()
        self.assertEqual(selected["environment"], expected)
        selected["environment"]["unknown"].append("edited")
        selected["environment"]["applied"].clear()
        with _environment({"FEATHER_PGS_KEYED_ROWS": "0"}):
            self.assertEqual(solver.get_path_selection()["environment"], expected)

    def test_solver_warns_once_per_process_about_retired_names(self):
        """Warn at the caller once per process for each retired name, keep recording it and never raise.

        Isaac Lab-owned NEWTON_ variables are not the solver's family and stay silent, also for the
        collision pipeline, which records them apart.
        """
        model = _cpu_model()
        recipe = {
            "FEATHER_PGS_SPARSE_BRANCH_SWEEPS": "0",
            "FEATHER_PGS_KUKA_JOINT_WORLD": "1",
            "NEWTON_NARROW_PHASE_THREADS_X": "4",
        }
        with patch.object(core_flags, "_WARNED_RETIRED", set()):
            with (
                _environment(recipe),
                self.assertWarnsRegex(
                    UserWarning,
                    r"^Ignoring retired environment variables that select nothing: FEATHER_PGS_KUKA_JOINT_WORLD, "
                    r"FEATHER_PGS_SPARSE_BRANCH_SWEEPS\.",
                ) as caught,
            ):
                first = newton.solvers.SolverFeatherPGS(model)
            self.assertEqual(os.path.basename(caught.filename), os.path.basename(__file__))
            with (
                _environment(recipe | {"FEATHER_PGS_LOCAL_ROW_PACKETS": "1"}),
                warnings.catch_warnings(record=True) as seen,
            ):
                warnings.simplefilter("always")
                second = newton.solvers.SolverFeatherPGS(model)
                third = newton.solvers.SolverFeatherPGS(model)
                pipeline = newton.CollisionPipeline(model)
        self.assertEqual(
            [str(item.message).split(": ")[1].split(".")[0] for item in seen], ["FEATHER_PGS_LOCAL_ROW_PACKETS"]
        )
        self.assertEqual(
            first.get_path_selection()["environment"]["retired"],
            ["FEATHER_PGS_KUKA_JOINT_WORLD", "FEATHER_PGS_SPARSE_BRANCH_SWEEPS"],
        )
        for solver in (second, third):
            self.assertEqual(
                solver.get_path_selection()["environment"]["retired"],
                ["FEATHER_PGS_KUKA_JOINT_WORLD", "FEATHER_PGS_LOCAL_ROW_PACKETS", "FEATHER_PGS_SPARSE_BRANCH_SWEEPS"],
            )
        self.assertEqual(
            pipeline.get_collision_selection()["environment"]["isaac_lab"], {"NEWTON_NARROW_PHASE_THREADS_X": "4"}
        )

    def test_collision_pipeline_reports_construction_environment(self):
        """Report the NEWTON_ environment read at construction, with Isaac Lab-owned names kept apart."""
        flags = {
            "NEWTON_NARROW_PHASE_THREADS_X": "4",
            "NEWTON_COLLISION_BOX_SAT": "0",
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
            "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
            "NEWTON_UNDECLARED_EXAMPLE": "1",
            "FEATHER_PGS_ROWS_MASKED": "1",
        }
        model = _cpu_model()
        with _environment(flags):
            pipeline = newton.CollisionPipeline(model)
        expected = {
            "applied": {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1"},
            "unknown": ["NEWTON_UNDECLARED_EXAMPLE"],
            "retired": ["NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD"],
            "isaac_lab": {"NEWTON_COLLISION_BOX_SAT": "0", "NEWTON_NARROW_PHASE_THREADS_X": "4"},
            "overridden": {},
        }
        selected = pipeline.get_collision_selection()
        self.assertEqual(selected["environment"], expected)
        selected["environment"]["isaac_lab"].clear()
        with _environment({"NEWTON_HEIGHTFIELD_CELL_REJECT": "0"}):
            self.assertEqual(pipeline.get_collision_selection()["environment"], expected)

    def test_collision_pipeline_warns_once_per_process_about_retired_names(self):
        """Warn at the caller once per process for each retired NEWTON_ name, keep recording it and never raise.

        ``NEWTON_TERRAIN_LAW_RULE_X`` selected the terrain law's removed rule X: any value builds the pipeline, which
        lists it as retired.
        """
        model = _cpu_model()
        with patch.object(core_flags, "_WARNED_RETIRED", set()):
            with (
                _environment({"NEWTON_TERRAIN_LAW_RULE_X": "1"}),
                self.assertWarnsRegex(
                    UserWarning,
                    r"^Ignoring retired environment variables that select nothing: NEWTON_TERRAIN_LAW_RULE_X\.",
                ) as caught,
            ):
                first = newton.CollisionPipeline(model)
            self.assertEqual(os.path.basename(caught.filename), os.path.basename(__file__))
            with (
                _environment({"NEWTON_TERRAIN_LAW_RULE_X": "0", "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0"}),
                warnings.catch_warnings(record=True) as seen,
            ):
                warnings.simplefilter("always")
                second = newton.CollisionPipeline(model)
                third = newton.CollisionPipeline(model)
        self.assertEqual(
            [str(item.message).split(": ")[1].split(".")[0] for item in seen], ["NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD"]
        )
        self.assertEqual(first.get_collision_selection()["environment"]["retired"], ["NEWTON_TERRAIN_LAW_RULE_X"])
        for pipeline in (second, third):
            self.assertEqual(
                pipeline.get_collision_selection()["environment"]["retired"],
                ["NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD", "NEWTON_TERRAIN_LAW_RULE_X"],
            )

    def test_collision_pipeline_reports_overridden_environment(self):
        """Report the NEWTON_ names an explicit argument superseded apart from the applied ones."""
        flags = {
            # Invalid on their own: an explicit argument replaces them before validation.
            "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "not_a_mode",
            "NEWTON_NARROW_PHASE_CONVEX_CELLS": "maybe",
            "NEWTON_REDUCER_COMPACT_EXPORT": "1",
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
        }
        with _environment(flags):
            pipeline = newton.CollisionPipeline(_cpu_model(), convex_collision="auto", contact_export="auto")
        selected = pipeline.get_collision_selection()
        self.assertEqual(selected["environment"]["applied"], {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1"})
        self.assertEqual(
            selected["environment"]["overridden"],
            {
                "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "not_a_mode",
                "NEWTON_NARROW_PHASE_CONVEX_CELLS": "maybe",
                "NEWTON_REDUCER_COMPACT_EXPORT": "1",
            },
        )
        self.assertEqual(selected["narrow_phase"]["coherent_convex"]["source"], "argument")
        with _environment({"NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0", "NEWTON_REDUCER_COMPACT_EXPORT": "0"}):
            environment = newton.CollisionPipeline(_cpu_model()).get_collision_selection()["environment"]
        self.assertEqual(
            environment["applied"], {"NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0", "NEWTON_REDUCER_COMPACT_EXPORT": "0"}
        )
        self.assertEqual(environment["overridden"], {})


if __name__ == "__main__":
    unittest.main()

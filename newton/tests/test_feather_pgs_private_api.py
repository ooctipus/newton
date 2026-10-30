# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from pathlib import Path

_FEATHER_PGS = Path(__file__).parents[1] / "_src" / "solvers" / "feather_pgs"

# Kernel factories: cached, so that a variant is built once per process.
_KERNEL_FACTORIES = {
    "tree_ordinary.py": ("triangular_solve_kernel",),
    "response_builds_dense.py": (
        "factor_warps_kernel",
        "factor_tiles_kernel",
        "response_warps_kernel",
        "response_tiles_kernel",
    ),
    "response_builds_rows.py": (
        "free_body_meta_kernel",
        "factor_contact_rows_kernel",
        "sparse_factor_kernel",
        "paired_components_kernel",
    ),
}


def _top_level_functions(module: str) -> dict:
    tree = ast.parse((_FEATHER_PGS / module).read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


class TestFeatherPGSPrivateApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver_module = ast.parse((_FEATHER_PGS / "solver_feather_pgs.py").read_text())
        cls.top_level_functions = {
            node.name: node for node in cls.solver_module.body if isinstance(node, ast.FunctionDef)
        }
        cls.solver_class = next(
            node
            for node in cls.solver_module.body
            if isinstance(node, ast.ClassDef) and node.name == "SolverFeatherPGS"
        )
        cls.solver_methods = {node.name: node for node in cls.solver_class.body if isinstance(node, ast.FunctionDef)}

    def test_prescribed_response_is_not_a_public_execution_knob(self):
        init_method = self.solver_methods["__init__"]
        parameters = {
            argument.arg
            for argument in [*init_method.args.posonlyargs, *init_method.args.args, *init_method.args.kwonlyargs]
        }
        self.assertNotIn("exclude_fully_kinematic_free_articulations", parameters)

    def test_tiled_kernel_factory_is_not_exported(self):
        class_names = {node.name for node in self.solver_module.body if isinstance(node, ast.ClassDef)}
        self.assertNotIn("TiledKernelFactory", class_names)

    def test_kernel_factories_are_cached(self):
        for module, names in _KERNEL_FACTORIES.items():
            functions = _top_level_functions(module)
            for name in names:
                with self.subTest(module=module, factory=name):
                    self.assertIn(name, functions)
                    decorators = {ast.unparse(decorator) for decorator in functions[name].decorator_list}
                    self.assertTrue(decorators & {"cache", "functools.cache"}, decorators)

    def test_plain_response_tiles_kernel_omits_diagonal_output(self):
        factory = _top_level_functions("response_builds_dense.py")["response_tiles_kernel"]
        templates = {node.name: node for node in factory.body if isinstance(node, ast.FunctionDef)}
        diagonal_branch = next(
            node for node in factory.body if isinstance(node, ast.If) and ast.unparse(node.test) == "compute_diag"
        )
        diagonal_template = next(node for node in diagonal_branch.body if isinstance(node, ast.FunctionDef))
        plain_template = templates["hinv_jt_plain_template"]

        factory_parameters = {argument.arg for argument in factory.args.args}
        plain_parameters = {argument.arg for argument in plain_template.args.args}
        diagonal_parameters = {argument.arg for argument in diagonal_template.args.args}
        plain_names = {node.id for node in ast.walk(plain_template) if isinstance(node, ast.Name)}
        diagonal_names = {node.id for node in ast.walk(diagonal_branch) if isinstance(node, ast.Name)}

        self.assertNotIn("diag_group", plain_parameters)
        self.assertNotIn("diag_tile", plain_names)
        self.assertIn("compute_diag", factory_parameters)
        self.assertIn("COMPUTE_DIAG", diagonal_names)
        self.assertIn("diag_group", diagonal_parameters)
        self.assertIn("diag_tile", diagonal_names)

    def test_cholesky_and_triangular_kernels_are_not_dense_constraint_gated(self):
        # The plan's factor and triangular-solve kernels (OrdinaryTree.build_kernels) do not depend on the dense row
        # capacity.
        tree = ast.parse((_FEATHER_PGS / "tree_ordinary.py").read_text())
        ordinary = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OrdinaryTree")
        methods = {node.name: node for node in ordinary.body if isinstance(node, ast.FunctionDef)}
        for helper_name in ("factor_tiles_kernel", "triangular_solve_kernel"):
            with self.subTest(helper_name=helper_name):
                node = methods["build_kernels"]
                call_line = self._first_call_line(node, helper_name)
                first_dense_continue = self._first_dense_guard_continue_line(node)
                if first_dense_continue is not None:
                    self.assertLess(call_line, first_dense_continue)
                dense_guarded_none_assignments = self._dense_guarded_none_assignments(node)
                self.assertNotIn("cholesky_kernels_by_size", dense_guarded_none_assignments)
                self.assertNotIn("triangular_solve_kernels_by_size", dense_guarded_none_assignments)

    @staticmethod
    def _references_dense_max_constraints(node):
        return any(
            isinstance(child, ast.Attribute) and child.attr == "dense_max_constraints" for child in ast.walk(node)
        )

    @classmethod
    def _first_dense_guard_continue_line(cls, node):
        continue_lines = []
        for child in ast.walk(node):
            if isinstance(child, ast.If) and cls._references_dense_max_constraints(child.test):
                for body_node in child.body:
                    continue_lines.extend(
                        descendant.lineno for descendant in ast.walk(body_node) if isinstance(descendant, ast.Continue)
                    )
        return min(continue_lines) if continue_lines else None

    @staticmethod
    def _first_call_line(node, function_name):
        call_lines = [
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and function_name
            in (
                getattr(child.func, "id", None),  # a module-level helper
                getattr(child.func, "attr", None),  # a build module's factory
            )
        ]
        if not call_lines:
            raise AssertionError(f"{function_name} is not called while initializing tiled kernels")
        return min(call_lines)

    @classmethod
    def _dense_guarded_none_assignments(cls, node):
        assigned_attrs = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.If) or not cls._references_dense_max_constraints(child.test):
                continue
            for body_node in child.body:
                for descendant in ast.walk(body_node):
                    if not isinstance(descendant, ast.Assign):
                        continue
                    if not isinstance(descendant.value, ast.Constant) or descendant.value.value is not None:
                        continue
                    for target in descendant.targets:
                        attr_name = cls._assigned_self_collection_attr(target)
                        if attr_name is not None:
                            assigned_attrs.add(attr_name)
        return assigned_attrs

    @staticmethod
    def _assigned_self_collection_attr(target):
        if isinstance(target, ast.Subscript):
            target = target.value
        if isinstance(target, ast.Attribute):
            return target.attr
        return None


if __name__ == "__main__":
    unittest.main()

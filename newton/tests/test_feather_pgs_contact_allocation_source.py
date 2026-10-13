# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One reservation law with explicit original and prevalidated geometry inputs."""

import ast
import importlib
import unittest


class TestContactAllocationSource(unittest.TestCase):
    def test_cached_owner_extends_original_contract(self):
        factory = importlib.import_module("newton._src.solvers.feather_pgs.contact_allocation_source")
        original = ast.parse(factory.get_source()).body
        cached = ast.parse(factory.get_source(cached_geometry=True)).body
        self.assertEqual(len(original), 2)
        self.assertEqual(len(cached), 2)
        for i, extras in enumerate((["cache"], ["cache", "raw", "settings"])):
            original_args = [a.arg for a in original[i].args.args]
            cached_args = [a.arg for a in cached[i].args.args]
            self.assertEqual(cached_args, original_args + extras)
        self.assertIn("dense_first_rejected_slot", ast.unparse(cached[0]))
        self.assertIn("_stage_row(c, total_contacts, cache, raw, settings)", ast.unparse(cached[1]))
        self.assertNotIn("wp.transform_point", ast.unparse(cached[0]))

    def test_current_prefix_reuses_all_output_initialization(self):
        factory = importlib.import_module("newton._src.solvers.feather_pgs.allocation_count_source")
        original = ast.parse(factory.get_source()).body[0]
        cached = ast.parse(factory.get_source(cached_geometry=True)).body[0]

        def publication(node):
            start = next(
                i
                for i, statement in enumerate(node.body)
                if isinstance(statement, ast.Assign) and ast.unparse(statement).startswith("allocation.dense_counter")
            )
            return [ast.dump(statement, include_attributes=False) for statement in node.body[start:]]

        self.assertEqual(publication(original), publication(cached))
        self.assertIn("state.active_worlds[index] = world", ast.unparse(cached))
        self.assertNotIn("range(46)", ast.unparse(cached))


if __name__ == "__main__":
    unittest.main()

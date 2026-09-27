# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the narrow current-resolved publication source adapter."""

import ast
import hashlib
import inspect
import unittest

from newton._src.solvers.feather_pgs import world_scan_publication as publication

STANDALONE_FACTORY_SHA256 = "f17722bb377512bcf86a2b800902910c4fa135e04f0fe00dac19d038e102dd91"


def native_function(source):
    """Extract the single nested publication kernel from its factory."""
    functions = [node for node in ast.parse(source).body[0].body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise AssertionError("Expected exactly one nested native kernel")
    return functions[0]


class TestWorldScanResolvedSource(unittest.TestCase):
    def test_current_resolved_api(self):
        """Require the new three-operand variant while preserving standalone ABI."""
        original = publication.get_kernel("cpu")
        resolved = publication.get_resolved_kernel("cpu")
        self.assertEqual([arg.label for arg in original.adj.args], ["plan", "data"])
        self.assertEqual([arg.label for arg in resolved.adj.args], ["plan", "data", "resolved"])
        self.assertIs(publication.get_resolved_kernel("cpu"), resolved)

    def test_standalone_factory_exact(self):
        """Keep the previously tested standalone factory source byte-identical."""
        source = inspect.getsource(publication.get_kernel)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), STANDALONE_FACTORY_SHA256)

    def test_only_generalized_segment_is_guarded(self):
        """Wrap exactly three generalized loops and their joins, leaving body law intact."""
        source = inspect.getsource(publication.get_kernel)
        original = native_function(source)
        changed = native_function(publication._resolved_source(source))
        guard = next(
            node
            for node in changed.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "resolved[world] == 0"
        )
        self.assertEqual(len(guard.body), 6)
        self.assertEqual([type(node) for node in guard.body], [ast.For, ast.Expr] * 3)
        self.assertEqual([ast.unparse(node.value) for node in guard.body[1::2]], ["_sync()"] * 3)
        self.assertEqual(guard.orelse, [])
        offset = changed.body.index(guard)
        changed.body[offset : offset + 1] = guard.body
        changed.name = original.name
        changed.args.args.pop()
        self.assertEqual(ast.dump(changed), ast.dump(original))

    def test_changed_or_duplicated_seams_reject(self):
        """Reject a changed generalized owner instead of silently wrapping different work."""
        source = inspect.getsource(publication.get_kernel)
        first = "        for local in range(lane, 35, stride):\n"
        body = "            transform = kernels.jcalc_transform(\n"
        for malformed in (source.replace(first, first.replace("35", "36")), source + first, source.replace(body, "")):
            with self.subTest(malformed=malformed[-80:]), self.assertRaises(ValueError):
                publication._resolved_source(malformed)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the kernel identity record: code normalization, comparison and agreement with the pins."""

import contextlib
import io
import json
import unittest
from pathlib import Path

from tools.fpgs_bench import kernel_identity as identity


def _record(**kernels):
    return {"kernels": {name: dict(zip(identity.COLUMNS, row, strict=True)) for name, row in kernels.items()}}


class TestKernelIdentity(unittest.TestCase):
    def test_normalize_ignores_comments_line_directives_and_layout(self):
        """Ignore comments, line directives and layout whitespace; keep code and string literals."""
        source = 'int a = 1;  // x = f(y)  <L 12>\n#line 40 "k.py"\n/* block\n comment */  int b = a +  2;\n'
        moved = '    int a = 1;   // x = f(y)  <L 98>\n\n#line 7 "k.py"\nint b = a + 2;\n'
        self.assertEqual(identity.normalize(source), "int a = 1;\nint b = a + 2;")
        self.assertEqual(identity.normalize(source), identity.normalize(moved))
        self.assertNotEqual(identity.normalize("int a = 1;"), identity.normalize("int a = 2;"))
        self.assertNotEqual(identity.normalize('printf("a  b");'), identity.normalize('printf("a b");'))
        self.assertEqual(identity.normalize('s = "// kept";  // dropped'), 's = "// kept";')

    def test_normalize_replaces_only_the_kernel_hash_suffix(self):
        """Replace the kernel's hash suffix in entry points and argument structs, and nothing else."""
        code = (
            "void k_0123abcd_cuda_kernel_forward(wp_args_k_0123abcd* a);\n"
            "Patches_cf9b8dab p; int k_0123abcd9; int x0123abcd;"
        )
        normalized = identity.normalize(code, ("0123abcd",))
        self.assertIn("void k_HASH_cuda_kernel_forward(wp_args_k_HASH* a);", normalized)
        # A struct's layout hash and longer names keep their digits.
        self.assertIn("Patches_cf9b8dab p; int k_0123abcd9; int x0123abcd;", normalized)

    def test_normalize_names_functions_by_build_order(self):
        """Rename user functions by key and build order, with their adjoint, replay and callable names."""

        class Function:
            def __init__(self, key, native_func):
                self.key, self.native_func = key, native_func

            def is_builtin(self):
                return False

        names = identity.stable_function_names(
            [Function("native", "native_26"), Function("f", "f_3"), Function("native", "native_4")]
        )
        self.assertEqual(names, {"native_26": "native__fn0", "f_3": "f__fn0", "native_4": "native__fn1"})
        code = (
            "native_26(); adj_native_26(); replay_native_4(); native_26_callable_0a1b(); my_native_26(); native_260();"
        )
        self.assertEqual(
            identity.normalize(code, (), names),
            "native__fn0(); adj_native__fn0(); replay_native__fn1(); native__fn0_callable_0a1b(); my_native_26(); "
            "native_260();",
        )

    def test_compare_reports_columns_additions_removals_and_rejections(self):
        """Report each changed column, added and removed kernels and changed rejections."""
        parent = _record(same=("1", "c", "p"), moved=("2", "c", "p"), edited=("3", "c", "p"), gone=("4", "c", "p"))
        child = _record(same=("1", "c", "p"), moved=("5", "c", "p"), edited=("6", "d", "p"), new=("7", "c", "p"))
        parent["rejected"], child["rejected"] = {"r": "ValueError: a"}, {"r": "ValueError: b"}
        result = identity.compare(parent, child)
        self.assertEqual(result["changed"], {"edited": ["hash", "cuda"], "moved": ["hash"]})
        self.assertEqual((result["added"], result["removed"], result["rejections"]), (["new"], ["gone"], ["r"]))
        self.assertEqual(identity.compare(parent, child, ("cuda", "cpu"))["changed"], {"edited": ["cuda"]})

    def test_report_fails_on_changes_and_removals_only(self):
        """Pass a record that only adds kernels; fail one that changes or drops a kernel."""
        parent = _record(a=("1", "c", "p"))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(identity.report(parent, _record(a=("1", "c", "p"), b=("2", "c", "p"))), 0)
            self.assertEqual(identity.report(parent, _record(a=("9", "c", "p"))), 1)
            self.assertEqual(identity.report(parent, _record(a=("9", "c", "p")), ("cuda", "cpu")), 0)
            self.assertEqual(identity.report(parent, _record(b=("1", "c", "p"))), 1)
        self.assertIn("changed (hash only, generated code identical): a", output.getvalue())

    def test_report_claims_identical_code_only_when_compared(self):
        """Name a hash change of records without code digests as not compared, not as identical code."""
        parent, child = {"kernels": {"a": {"hash": "1"}}}, {"kernels": {"a": {"hash": "2"}}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(identity.report(parent, child), 1)
        self.assertIn("changed (hash changed, generated code not compared): a", output.getvalue())
        self.assertNotIn("generated code identical", output.getvalue())

    def test_label_names_calls_like_the_pins(self):
        """Label calls with positional values, sorted keywords and named layouts."""
        layout = identity.Named("chain9", object())
        self.assertEqual(identity.label("f", (1, "120"), {"b": True, "a": layout}), "f(1, '120', a=chain9, b=True)")

    def test_hash_column_is_the_pinned_content_hash(self):
        """Record a kernel's point-stream pin as its hash, beside its CUDA and CPU code digests."""
        import warp as wp  # noqa: PLC0415

        wp.init()
        kernels, rejected = identity.inventory(("pins",), ("kernels.allocate_world_contact_slots",))
        self.assertEqual(rejected, {})
        pins = Path(__file__).resolve().parents[2] / "newton/tests/golden_data/test_feather_pgs_point_streams/pins.json"
        stored = json.loads(pins.read_text(encoding="utf-8"))["kernels"]
        self.assertEqual(len(kernels), 2)
        for name, (family, kernel) in kernels.items():
            with self.subTest(kernel=name):
                row = identity.kernel_digests(kernel)
                self.assertEqual((family, row["key"], row["hash"]), ("pins", name.split(".", 1)[1], stored[name]))
                self.assertRegex(row["cuda"], "^[0-9a-f]{64}$")
                self.assertRegex(row["cpu"], "^[0-9a-f]{64}$")
                self.assertNotEqual(row["cuda"], row["cpu"])


if __name__ == "__main__":
    unittest.main()

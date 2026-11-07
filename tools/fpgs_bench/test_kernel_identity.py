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

    def test_records_by_key_match_renamed_calls(self):
        """Match a factory's renamed calls by the kernels they build: one row a kernel key, rejections by summary."""

        def row(key, digest):
            return {"key": key, "hash": digest}

        parent = {
            "kernels": {"old(1)": row("k", "1"), "old(2)": row("k", "1"), "pin": row("p", "2")},
            "rejected": {"old [census]": "2 rejected, labels 0123: 2 x ValueError: a"},
        }
        child = {
            "kernels": {"New(1)": row("k", "1"), "pin": row("p", "2")},
            "rejected": {"New [census]": "2 rejected, labels 4567: 2 x ValueError: a"},
        }
        self.assertEqual(identity.by_key(parent)["kernels"], {"k": row("k", "1"), "p": row("p", "2")})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(identity.report(identity.by_key(parent), identity.by_key(child), ("hash",)), 0)
            moved = {**child, "kernels": {"New(1)": row("k", "3"), "pin": row("p", "2")}}
            self.assertEqual(identity.report(identity.by_key(parent), identity.by_key(moved), ("hash",)), 1)

    def test_report_claims_identical_code_only_when_compared(self):
        """Name a hash change of records without code digests as not compared, not as identical code."""
        parent, child = {"kernels": {"a": {"hash": "1"}}}, {"kernels": {"a": {"hash": "2"}}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(identity.report(parent, child), 1)
        self.assertIn("changed (hash changed, generated code not compared): a", output.getvalue())
        self.assertNotIn("generated code identical", output.getvalue())

    def test_code_diff_fails_kernels_without_code_digests(self):
        """Fail a generated-code diff of a changed hash that no code digest covers; pass an unchanged hash."""
        full = _record(a=("1", "c", "p"), b=("2", "c", "p"))
        hash_only = {"kernels": {"a": {"hash": "1"}, "b": {"hash": "3"}}}
        for parent, child in ((hash_only, {"kernels": {"a": {"hash": "1"}, "b": {"hash": "4"}}}), (full, hash_only)):
            with self.subTest(parent=parent), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(identity.report(parent, child, ("cuda", "cpu")), 1)
                self.assertEqual(identity.compare(parent, child, ("cuda", "cpu"))["uncompared"], {"b": ["cuda", "cpu"]})
            self.assertIn("1 not compared", output.getvalue())
            self.assertIn(
                "not compared (no cuda, cpu on both sides, and the hash changed or is missing): b", output.getvalue()
            )
        # The content hash covers every input of code generation.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(identity.report(hash_only, hash_only, ("cuda", "cpu")), 0)

    def test_held_tree_grid_records_the_kernels_as_they_run(self):
        """Record every layout's state kernels with body velocity, which every held-tree operator requires.

        Every layout records its finish and its repair. Layouts of ordinary
        motion, at every lane width, also record the repair that compares input
        banks. The state kernels without body velocity stay recorded as a switch of four
        layouts, both forests among them.
        """
        calls = [
            (name, args, kwargs["layout"].name, {key: value for key, value in kwargs.items() if key != "layout"})
            for name, _, args, kwargs in identity.held_tree_calls()
        ]
        state = "tree_kinetic_state.get_state_kernel"
        for name, (tree, chain, _, options) in identity.held_tree_layouts().items():
            schedules = identity._schedules(tree, chain)
            with self.subTest(layout=name):
                for finish in (False, True):
                    running = {
                        key: value for key, value in options.items() if not (finish and key == "motion_velocity")
                    }
                    self.assertIn((state, (finish, schedules), name, {**running, "body_velocity": True}), calls)
                ordinary = not options.get("motion_velocity", False)
                coordinates = tree.dofs + tree.free_root_count
                call = (state, (False, schedules), name, {"body_velocity": True, "bank_coordinates": coordinates})
                self.assertEqual(call in calls, ordinary, call)
        without = {layout for _, _, layout, options in calls if options.get("body_velocity") is False}
        self.assertEqual(without, {"chain_free13", "kinetic_fixed11_l16", "forest13_l16", "forest32"})
        layouts = identity.held_tree_layouts().values()
        widths = {tree.lanes for tree, _, _, options in layouts if not options.get("motion_velocity", False)}
        self.assertEqual(widths, {4, 8, 16, 32, 64})

    def test_walk_family_covers_the_walk_shapes(self):
        """Record the walk's readers on a class with joins at g1's size, the loop form, side loads and every held width."""
        from newton._src.solvers.feather_pgs import tree_scan  # noqa: PLC0415

        classes = identity.walk_classes_of_the_suite()
        self.assertEqual((len(classes["g1"].parent), classes["g1"].coordinates), (38, 43))
        self.assertGreater(classes["chain106"].coordinates, tree_scan.WALK_UNROLL)
        self.assertIn(tree_scan.WALK_BOTH, classes["d6"].half)
        calls = list(identity.walk_calls())
        trees = [
            kwargs["layout"].value for name, _, _, kwargs in calls if name == "tree_kinetic_state.get_predictor_kernel"
        ]
        words = {-(-max(tree.dofs, tree.bodies + 1) // min(tree.lanes, 32)) for tree in trees}
        self.assertEqual(words, {1, 2, 4})
        names = {name for name, _, _, _ in calls}
        for name in ("walk_leave_kernel", "walk_enter_kernel", "walk_step_kernel", "get_walk_solve_kernel"):
            self.assertIn(f"factor.walk.{name}" if name.startswith("walk") else f"tree_kinetic_state.{name}", names)
        self.assertIn("walk_rows.support_rows_kernel", names)
        self.assertIn("walk_rows.walked_rows_kernel", names)
        labels = [identity.label(name, args, kwargs) for name, _, args, kwargs in calls]
        self.assertEqual(len(labels), len(set(labels)))

    def test_combinations_take_one_option_of_every_axis(self):
        """Merge one option of every census axis, for every combination of options."""
        axes = (identity._axis("a"), ({}, {"b": 1, "c": 2}))
        self.assertEqual(
            list(identity._combinations(axes)),
            [{"a": False}, {"a": False, "b": 1, "c": 2}, {"a": True}, {"a": True, "b": 1, "c": 2}],
        )

    def test_census_hashes_admitted_combinations_and_summarizes_rejections(self):
        """Hash every admitted flag combination of a factory, labeled by its non-default flags; count rejections."""
        rows, rejected = identity.census(("sweep_routes_diag.DiagonalSweep",))
        self.assertEqual(len(rows), 12)
        self.assertIn("sweep_routes_diag.DiagonalSweep(64, 10, 9, '120')", rows)
        self.assertTrue(
            all(set(row) == {"family", "key", "hash"} and row["family"] == "census" for row in rows.values())
        )
        self.assertEqual(list(rejected), ["sweep_routes_diag.DiagonalSweep [census]"])
        self.assertRegex(next(iter(rejected.values())), r"^4 rejected, labels [0-9a-f]{16}: 4 x ValueError: ")
        # A call another family records keeps its code digests there.
        skipped, _ = identity.census(("sweep_routes_diag.DiagonalSweep",), skip=rows.keys())
        self.assertEqual(skipped, {})

    def test_census_crosses_resident_row_metadata(self):
        """Cross the matrix-free sweep with resident row metadata, which rewrites its row loads."""
        calls = [kwargs for name, _, _, kwargs in identity.census_calls() if name.endswith("sweep_routes.GeneralSweep")]
        self.assertTrue(any(kwargs["places"].staging == "resident" for kwargs in calls))

    def test_coordinate_tree_kernels_are_recorded_with_the_pins(self):
        """Record every kernel of the held-tree owner's one-coordinate class in the pins family.

        They are module-level kernels without a tree layout, so the held-tree
        layout grid builds none of them; the kernel-pin inventory holds
        them, and an identity diff compares their generated code there.
        """
        from newton._src.solvers.feather_pgs import coordinate_trees  # noqa: PLC0415

        kernels, rejected = identity.inventory(("pins",), ("coordinate_trees.",))
        self.assertEqual(rejected, {})
        kinds = ("slides", "turns", "welded_slides", "welded_turns")
        names = sorted(f"{stage}_{kind}" for stage in ("repair", "predict", "finish") for kind in kinds)
        self.assertEqual(sorted(kernels), [f"coordinate_trees.{name}" for name in names])
        for name in names:
            self.assertEqual(kernels[f"coordinate_trees.{name}"], ("pins", getattr(coordinate_trees, name)))

    def test_label_names_calls_like_the_pins(self):
        """Label calls with positional values, sorted keywords and named layouts."""
        layout = identity.Named("chain9", object())
        self.assertEqual(identity.label("f", (1, "120"), {"b": True, "a": layout}), "f(1, '120', a=chain9, b=True)")

    def test_hash_column_is_the_pinned_content_hash(self):
        """Record a kernel's kernel pin as its hash, beside its CUDA and CPU code digests."""
        import warp as wp  # noqa: PLC0415

        pins = Path(__file__).resolve().parents[2] / "newton/tests/golden_data/test_feather_pgs_kernel_pins/pins.json"
        record = json.loads(pins.read_text(encoding="utf-8"))
        # As the kernel-pin test: the pins use Warp's private hasher of one version and no switches.
        if record["warp"] != wp.config.version:
            self.skipTest(f"Pins use Warp's private ModuleHasher of {record['warp']}; running Warp {wp.config.version}")
        if identity._switches():
            self.skipTest(f"Import-time switches select other generated kernels: {', '.join(identity._switches())}")
        stored = record["kernels"]
        wp.init()
        kernels, rejected = identity.inventory(("pins",), ("row_slots.allocate_world_contact_slots",))
        self.assertEqual(rejected, {})
        self.assertEqual(len(kernels), 1)
        for name, (family, kernel) in kernels.items():
            with self.subTest(kernel=name):
                row = identity.kernel_digests(kernel)
                self.assertEqual((family, row["key"], row["hash"]), ("pins", name.split(".", 1)[1], stored[name]))
                self.assertRegex(row["cuda"], "^[0-9a-f]{64}$")
                self.assertRegex(row["cpu"], "^[0-9a-f]{64}$")
                self.assertNotEqual(row["cuda"], row["cpu"])


if __name__ == "__main__":
    unittest.main()

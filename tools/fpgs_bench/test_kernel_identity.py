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

    def test_held_factor_calls_index_like_their_owners(self):
        """Read a tree's own geometry record, and each held forest family's root block of the forest record."""
        observed = {
            (
                args[0],
                kwargs.get("geometric_stride") or max(args[1].value) + 1,
                kwargs.get("geometric_group", False),
            )
            for _, _, args, kwargs in identity.held_factor_calls()
        }
        # Stride: a tree's entry count, or the forest's; a prescribed root holds no factor.
        expected = {(16, 40, False), (6, 21, False), (9, 86, True), (6, 86, True), (23, 222, True), (6, 222, True)}
        self.assertEqual(observed, expected)

    def test_held_tree_grid_records_the_kernels_as_they_run(self):
        """Record every layout's state kernels with body velocity, which every held-tree operator requires.

        Every layout records its finish and its repair. Layouts of ordinary
        motion, at every lane width, also record the one-pass repair that
        compares input banks and the kernels of a launch large enough to
        classify: the classification, the listed repair and the comparison
        pass. Owners classify no motion-velocity layout, so the grid records no
        classification or listed repair of an alternate motion. The state
        kernels without body velocity stay recorded as a switch of four
        layouts, both forests among them.
        """
        calls = [
            (name, args, kwargs["layout"].name, {key: value for key, value in kwargs.items() if key != "layout"})
            for name, _, args, kwargs in identity.held_tree_calls()
        ]
        state = "tree_kinetic_state.get_state_kernel"
        classify = "tree_kinetic_state.get_classify_kernel"
        for name, (tree, chain, _, options) in identity.held_tree_layouts().items():
            schedules = identity._schedules(tree, chain)
            with self.subTest(layout=name):
                for finish in (False, True):
                    running = {
                        key: value for key, value in options.items() if not (finish and key == "motion_velocity")
                    }
                    self.assertIn((state, (finish, schedules), name, {**running, "body_velocity": True}), calls)
                ordinary = not options.get("motion_velocity", False)
                coordinates = tree.dofs + tree.free_roots
                for call in (
                    (state, (False, schedules), name, {"body_velocity": True, "bank_coordinates": coordinates}),
                    (classify, (), name, {}),
                    (state, (False, schedules), name, {"body_velocity": True, "listed": True}),
                    ("tree_kinetic_state.get_bank_kernel", (), name, {"coordinates": coordinates}),
                ):
                    self.assertEqual(call in calls, ordinary, call)
        motion_classified = [
            call for call in calls if call[3].get("motion_velocity") and (call[0] == classify or call[3].get("listed"))
        ]
        self.assertEqual(motion_classified, [])
        without = {layout for _, _, layout, options in calls if options.get("body_velocity") is False}
        self.assertEqual(without, {"chain_free13", "kinetic_fixed11_l16", "forest13_l16", "forest32"})
        layouts = identity.held_tree_layouts().values()
        widths = {tree.lanes for tree, _, _, options in layouts if not options.get("motion_velocity", False)}
        self.assertEqual(widths, {4, 8, 16, 32, 64})

    def test_combinations_take_one_option_of_every_axis(self):
        """Merge one option of every census axis, for every combination of options."""
        axes = (identity._axis("a"), ({}, {"b": 1, "c": 2}))
        self.assertEqual(
            list(identity._combinations(axes)),
            [{"a": False}, {"a": False, "b": 1, "c": 2}, {"a": True}, {"a": True, "b": 1, "c": 2}],
        )

    def test_census_hashes_admitted_combinations_and_summarizes_rejections(self):
        """Hash every admitted flag combination of a factory, labeled by its non-default flags; count rejections."""
        rows, rejected = identity.census(("sweep_routes_local.paired_factor_kernel",))
        self.assertEqual(len(rows), 6)
        self.assertIn("sweep_routes_local.paired_factor_kernel(192, 29, 23, 6, '120')", rows)
        self.assertTrue(
            all(set(row) == {"family", "key", "hash"} and row["family"] == "census" for row in rows.values())
        )
        self.assertEqual(list(rejected), ["sweep_routes_local.paired_factor_kernel [census]"])
        self.assertRegex(next(iter(rejected.values())), r"^2 rejected, labels [0-9a-f]{16}: 2 x ValueError: ")
        # A call another family records keeps its code digests there.
        skipped, _ = identity.census(("sweep_routes_local.paired_factor_kernel",), skip=rows.keys())
        self.assertEqual(skipped, {})

    def test_census_crosses_the_velocity_limit_laws(self):
        """Cross every factory that takes a velocity-limit law with both laws, and the sweep with resident metadata.

        The laws rewrite the velocity-limit visits and the matrix-free sweep's resident row metadata
        its row loads, so a refactor can be exact under the default and change the other variant.
        Labels omit a keyword at its default, so the default variants keep their labels.
        """
        factories = ("sweep_routes_sparse.sparse_kernel",)
        rows, _ = identity.census(factories)
        for factory in factories:
            with self.subTest(factory=factory):
                labels = [key for key in rows if factory in key]
                accumulated = [key for key in labels if "velocity_limit_law='accumulated'" in key]
                self.assertTrue(accumulated)
                self.assertEqual(2 * len(accumulated), len(labels))
                self.assertFalse([key for key in labels if "velocity_limit_law='stateless'" in key])
        calls = [kwargs for name, _, _, kwargs in identity.census_calls() if name.endswith("sweep_routes.mf_gs_kernel")]
        self.assertTrue(any(kwargs["resident_row_metadata"] and not kwargs["shared_metadata"] for kwargs in calls))
        self.assertEqual({kwargs["velocity_limit_law"] for kwargs in calls}, {"stateless", "accumulated"})

    def test_scalar_branch_kernels_are_recorded_with_the_pins(self):
        """Record every kernel of the held-tree owner's scalar-branch group in the pins family.

        They are module-level kernels without a tree layout, so the held-tree
        layout grid builds none of them; the point-stream pin inventory holds
        them, and an identity diff compares their generated code there.
        """
        from newton._src.solvers.feather_pgs import scalar_branches  # noqa: PLC0415

        kernels, rejected = identity.inventory(("pins",), ("scalar_branches.",))
        self.assertEqual(rejected, {})
        names = (
            "finish_prismatic_branches",
            "predict_prismatic_branches",
            "prepare_listed_augmented_drives",
            "repair_scalar_branches",
        )
        self.assertEqual(sorted(kernels), [f"scalar_branches.{name}" for name in names])
        for name in names:
            self.assertEqual(kernels[f"scalar_branches.{name}"], ("pins", getattr(scalar_branches, name)))

    def test_label_names_calls_like_the_pins(self):
        """Label calls with positional values, sorted keywords and named layouts."""
        layout = identity.Named("chain9", object())
        self.assertEqual(identity.label("f", (1, "120"), {"b": True, "a": layout}), "f(1, '120', a=chain9, b=True)")

    def test_hash_column_is_the_pinned_content_hash(self):
        """Record a kernel's point-stream pin as its hash, beside its CUDA and CPU code digests."""
        import warp as wp  # noqa: PLC0415

        pins = Path(__file__).resolve().parents[2] / "newton/tests/golden_data/test_feather_pgs_point_streams/pins.json"
        record = json.loads(pins.read_text(encoding="utf-8"))
        # As the point-stream test: the pins use Warp's private hasher of one version and no switches.
        if record["warp"] != wp.config.version:
            self.skipTest(f"Pins use Warp's private ModuleHasher of {record['warp']}; running Warp {wp.config.version}")
        if identity._switches():
            self.skipTest(f"Import-time switches select other generated kernels: {', '.join(identity._switches())}")
        stored = record["kernels"]
        wp.init()
        kernels, rejected = identity.inventory(("pins",), ("kernels.allocate_world_contact_slots",))
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

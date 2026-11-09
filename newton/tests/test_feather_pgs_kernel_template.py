# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build FeatherPGS's generated kernels through one template.

The kernel pins hold every production kernel's content; these tests hold
the mechanism: strict source slots, one parameter declaration for the native
ABI and the Warp signature, the dispatches, and kernels that run.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.core.execution import Work
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.device_preludes import CUDA_ONLY, RING
from newton._src.solvers.feather_pgs.field_group import Field
from newton._src.solvers.feather_pgs.kernel_template import (
    Dispatch,
    KernelTemplate,
    Phase,
    compose,
    items,
    operands,
    params,
    python_kernel,
    substitute,
)
from newton._src.solvers.feather_pgs.row_inputs import RowArgument

Scaled = field_group.struct("Scaled", (Field("scale", wp.float32), Field("values", wp.float32, ("row",))), __name__)

# One native call per thread of a 1D launch, guarded by a count only the wrapper reads.
_GUARDED = Dispatch("tid = wp.tid()\nif tid < count:\n    $CALL", ("tid",), (("tid", int),), private=("count",))


class TestKernelTemplate(unittest.TestCase):
    """The template's slots, parameters, dispatches and kernels."""

    def test_substitute_needs_every_slot_and_every_value(self):
        """Fill each slot, and refuse a slot without a value or a value without a slot."""
        self.assertEqual(substitute("a[$N] = ${VALUE}f; $$x", N=3, VALUE=1.5), "a[3] = 1.5f; $x")
        with self.assertRaisesRegex(ValueError, r"without values \['N'\]"):
            substitute("a[$N] = $VALUE", VALUE=1)
        with self.assertRaisesRegex(ValueError, r"without slots \['M'\]"):
            substitute("a[$N]", N=1, M=2)
        # A vocabulary offers values a source may leave unused, and a value is never searched for slots.
        self.assertEqual(substitute("$N", {"N": "$M", "M": 2}), "$M")

    def test_params_declare_pairs_fields_and_structs(self):
        """Take parameters from pairs, field declarations and generated structs, by distinct identifiers."""
        declared = params(("count", int), Field("rows", wp.int32, ("row",)), Scaled)
        self.assertEqual([name for name, _ in declared], ["count", "rows", "scale", "values"])
        self.assertEqual((declared[1][1].dtype, declared[1][1].ndim), (wp.int32, 1))
        self.assertIs(declared[2][1], wp.float32)
        with self.assertRaisesRegex(ValueError, "distinct identifiers"):
            params(("scale", float), Scaled)

    def test_operands_insert_after_the_parameter_they_name(self):
        """Insert a variant's operands after their anchor, or last, and refuse a name the ABI holds."""
        base = (("a", int), ("b", int))
        inserted = operands(base, (RowArgument("x", float, "a"), RowArgument("y", float)))
        self.assertEqual([name for name, _ in inserted], ["a", "x", "b", "y"])
        for extra in (RowArgument("a", int), RowArgument("bad name", int)):
            with self.subTest(operand=extra.name), self.assertRaisesRegex(ValueError, "distinct identifier"):
                operands(base, (extra,))

    def test_signature_and_native_abi_derive_from_one_declaration(self):
        """Derive the wrapper's signature and the native ABI from the parameters and the dispatch."""
        queued = KernelTemplate.from_abi(
            "queued",
            (("candidate", int), ("first", int), ("second", wp.array[float]), ("third", float)),
            "",
            items("candidate", groups=2, lanes=16, queue=("active_candidate_count", "candidate_grid_stride")),
        )
        # The queue's operands lead the kernel's parameters; the native function takes the candidate.
        self.assertEqual(
            [name for name, _ in queued.signature],
            ["active_candidate_count", "candidate_grid_stride", "first", "second", "third"],
        )
        self.assertEqual([name for name, _ in queued.native_params], ["candidate", "first", "second", "third"])
        guarded = KernelTemplate("guarded", (("count", int), ("out", wp.array[float])), "", _GUARDED)
        self.assertEqual([name for name, _ in guarded.signature], ["count", "out"])
        self.assertEqual([name for name, _ in guarded.native_params], ["tid", "out"])
        self.assertIn("native(tid, out)", guarded.declaration())
        with self.assertRaisesRegex(ValueError, "world"):
            KernelTemplate.from_abi("unqueued", (("candidate", int),), "", items("world", queue=("count", "stride")))

    def test_snippet_composes_preludes_once(self):
        """Compose the preludes ahead of the body, or take a composed snippet as it is."""
        body = "    // body\n#endif\n"
        snippet = KernelTemplate("ringed", (), body, preludes=(RING,)).snippet
        self.assertEqual(snippet.count(RING.source), 1)
        self.assertTrue(snippet.endswith("    // body\n#else\n" + CUDA_ONLY + "#endif\n"))
        self.assertEqual(KernelTemplate("composed", (), snippet, composed=True).snippet, snippet)
        with self.assertRaisesRegex(ValueError, "composed"):
            _ = KernelTemplate("composed", (), snippet, preludes=(RING,), composed=True).snippet

    def test_kernel_runs_a_struct_operand_on_the_cpu(self):
        """Build a dual-device native kernel that reads a generated struct, and launch it."""
        template = KernelTemplate(
            "scaled_values",
            params(("count", int), ("data", Scaled), ("out", wp.array[float])),
            substitute("    out.data[tid] = data.scale * data.values.data[tid] * $FACTOR;\n", FACTOR="2.0f"),
            _GUARDED,
            dual=True,
        )
        kernel = template.kernel()
        self.assertEqual(kernel.key, "scaled_values")
        data = field_group.upload(Scaled, "cpu", scale=3.0, values=np.arange(4, dtype=np.float32))
        out = wp.full(4, -1.0, dtype=float, device="cpu")
        wp.launch(kernel, dim=4, inputs=[3, data, out], device="cpu")
        np.testing.assert_array_equal(out.numpy(), [0.0, 6.0, 12.0, -1.0])

    def test_python_kernel_names_a_variant(self):
        """Rename a Python kernel per variant, in a module of its own."""
        factor = 2.0

        def scale(values: wp.array[float]):
            tid = wp.tid()
            values[tid] = values[tid] * factor

        kernel = python_kernel(scale, "scale_by_two")
        self.assertEqual(kernel.key, "scale_by_two")
        values = wp.array([1.0, 2.0], dtype=float, device="cpu")
        wp.launch(kernel, dim=2, inputs=[values], device="cpu")
        np.testing.assert_array_equal(values.numpy(), [2.0, 4.0])


@wp.func
def _add_row(world: int, sub: int, rows: wp.array2d[float], out: wp.array[float]):
    wp.atomic_add(out, world, rows[world, sub])


class TestItems(unittest.TestCase):
    """The one dispatch: a thread's item directly, behind a tail guard, or from a queue, over a native or Warp body."""

    def test_dispatch_forms(self):
        """The default dispatch (a block's item), a flat thread's, a block's lane groups and a flat lane group's."""
        self.assertEqual(KernelTemplate("k", (), "").dispatch.body, "block, _lane = wp.tid()\n$CALL")
        self.assertEqual(items("tid", flat=True).arguments, ("wp.tid()",))
        self.assertEqual(
            items("world", groups=2, lanes=16, guard="world_count").body,
            "block, lane = wp.tid()\nworld = block * 2 + lane // 16\nif world < world_count:\n    $CALL",
        )
        self.assertEqual(items("world", lanes=32, flat=True).body, "world = wp.tid() // 32\n$CALL")
        with self.assertRaisesRegex(ValueError, "their lanes"):
            items("world", groups=2)

    def test_guard_and_listed_queue_run_on_the_cpu(self):
        """A guarded flat launch stops at its count; a listed queue strides over its items."""
        out = wp.zeros(8, dtype=float, device="cpu")
        guarded = KernelTemplate(
            "items_guarded",
            (("out", wp.array[float]),),
            "    out.data[tid] = 1.0f;\n",
            items("tid", flat=True, guard="n"),
            dual=True,
        ).kernel()
        wp.launch(guarded, dim=8, inputs=[5, out], device="cpu")
        np.testing.assert_array_equal(out.numpy(), [1, 1, 1, 1, 1, 0, 0, 0])
        listed = items("world", queue=("count", "stride", "worlds"))
        self.assertEqual([name for name, _ in listed.extra], ["count", "stride", "worlds"])
        queued = KernelTemplate(
            "items_listed", (("out", wp.array[float]),), "    out.data[world] += 2.0f;\n", listed, dual=True
        ).kernel()
        worlds = wp.array([6, 1, 3, 7, 0], dtype=int, device="cpu")
        count = wp.array([4], dtype=int, device="cpu")
        # Two blocks stride over the list's first four items: worlds 6, 1, 3, 7.
        wp.launch(queued, dim=(2, 1), inputs=[count, 2, worlds, out], device="cpu")
        np.testing.assert_array_equal(out.numpy(), [1, 3, 1, 3, 1, 0, 2, 2])

    def test_python_body_loops_its_sub_items(self):
        """A Warp function body under the dispatch: each world's lanes share its rows."""
        dispatch = items("world", sub="rows.shape[1]", lanes=4)
        self.assertEqual(dispatch.body, "world, lane = wp.tid()\nfor sub in range(lane, rows.shape[1], 4):\n    $CALL")
        kernel = KernelTemplate(
            "items_rows",
            (("rows", wp.array2d[float]), ("out", wp.array[float])),
            "",
            dispatch,
            function=_add_row,
        ).kernel()
        rows = wp.array(np.arange(12, dtype=np.float32).reshape(2, 6), dtype=float, device="cpu")
        out = wp.zeros(2, dtype=float, device="cpu")
        wp.launch(kernel, dim=(2, 4), inputs=[rows, out], device="cpu")
        np.testing.assert_array_equal(out.numpy(), [15.0, 51.0])


@wp.func
def _count_world(world: int, lane: int, lanes: int, visits: wp.array[float]):
    visits[world] = visits[world] + 1.0


def _phase(name, body="", function=None, params=(("out", wp.array[float]),)):
    leading = (("world", int), ("lane", int), ("lanes", int)) if function else (("world", int),)
    dispatch = Dispatch("", leading=leading)
    template = KernelTemplate(name, params, body, dispatch, dual=True, native_name=name, function=function)
    return Phase(template, ("world", "lane % 1", "1") if function else ("world",), Work(0, lanes=1, block=1))


class TestCompose(unittest.TestCase):
    """The composer: phases of one item in turn, the lanes' barrier between them, their operands bound once."""

    def test_phases_run_each_world_in_turn(self):
        """Two native phases and a Warp function phase on each world of a guarded grid."""
        first = _phase("place", "    out.data[world] = (float)world;\n")
        second = _phase("scale", "    out.data[world] = 2.0f * out.data[world] + 1.0f;\n")
        visit = _phase("visit", function=_count_world, params=(("visits", wp.array[float]),))
        dispatch = items("world", groups=1, lanes=1, guard="count")
        fused = compose("fused_cpu", (first, second, visit), dispatch, line=0)
        self.assertEqual([name for name, _ in fused.template.signature], ["count", "out", "visits"])
        calls = [row.strip().split("(")[0] for row in fused.template.dispatch.body.splitlines()[3:]]
        self.assertEqual(calls, ["place", "barrier", "scale", "barrier", "visit"])
        out, seen = wp.zeros(5, dtype=float, device="cpu"), wp.zeros(5, dtype=float, device="cpu")
        wp.launch_tiled(fused.template.kernel(), dim=[6], inputs=[5, out, seen], block_dim=1, device="cpu")
        np.testing.assert_array_equal(out.numpy(), [1, 3, 5, 7, 9])
        np.testing.assert_array_equal(seen.numpy(), [1, 1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "operand 'out'"):
            compose("clash", (first, _phase("other", params=(("out", wp.array[int]),))), dispatch, line=0)


if __name__ == "__main__":
    unittest.main()

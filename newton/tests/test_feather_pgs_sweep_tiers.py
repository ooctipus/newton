# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source checks of the parallel (Jacobi) tiers as configurations of the sweep emitter.

No GPU is needed: each tier takes its legacy factory's arguments and declares the same ABI, dispatch and tier
descriptor, rejects the same options, emits the Jacobi law the core composes, and, with the CUDA toolkit on ``PATH``,
compiles offline next to the legacy kernel.
"""

import contextlib
import functools
import importlib.util
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

from newton._src.solvers.feather_pgs import kernel_template, sweep_routes, sweep_routes_tiers
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout
from newton._src.solvers.feather_pgs.keyed_rows import get_direct_parallel_inputs
from newton._src.solvers.feather_pgs.response_operator import Block, CholeskySolve, Lanes
from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy
from newton._src.solvers.feather_pgs.sweep import (
    EXACT_EXIT,
    JacobiRows,
    RowLaws,
    change_vote,
    dense_tier_world,
    end_sweep,
    jacobi_sweeps,
)

_TIER = sweep_routes_tiers.parallel_kernel.__wrapped__

_LAYOUT = KeyedLayout(12, 6, 0, 12, 32, False, 0, None)


def _keyed(linked: bool):
    """The keyed rows' operands as the solver's keyed parallel factory binds them."""
    return functools.partial(get_direct_parallel_inputs, _LAYOUT, patch=linked)


_INK = {"matrix_free": True, "inkernel_response": (12, 6, 0, 12)}

CALLS = (
    # The conformance routes' configurations (32 rows, one warp), then the wider tiers and the other variants.
    ("block", (64, 32, 18, "120"), {"rows": 32, "sweeps": 112}),
    ("block_linked", (64, 32, 18, "120"), {"rows": 32, "sweeps": 112, "linked_rows": True}),
    (
        "matrix_free_exact",
        (64, 32, 18, "120"),
        {"rows": 32, "sweeps": 112, "matrix_free": True, "exact_row_sums": True},
    ),
    (
        "keyed",
        (64, 32, 18, "120"),
        {"rows": 32, "sweeps": 112, **_INK, "exact_row_sums": True, "row_inputs": _keyed(False)},
    ),
    (
        "keyed_linked",
        (64, 32, 18, "120"),
        {"rows": 32, "sweeps": 112, **_INK, "exact_row_sums": True, "row_inputs": _keyed(True), "linked_rows": True},
    ),
    ("block_rows", (64, 32, 40, "120"), {"rows": 32, "sweeps": 42}),
    ("block_drive", (64, 32, 18, "120"), {"rows": 32, "sweeps": 42, "has_drive_rows": True}),
    ("block_64", (96, 32, 18, "120"), {"rows": 64, "min_rows": 32, "sweeps": 60}),
    (
        "block_96_last_word",
        (96, 32, 18, "120"),
        {"rows": 96, "min_rows": 64, "sweeps": 73, "linked_rows": True, "velocity_limit_pass": True},
    ),
    ("matrix_free_bound", (64, 32, 18, "120"), {"rows": 32, "sweeps": 42, "matrix_free": True}),
    (
        "matrix_free_128",
        (128, 32, 18, "120"),
        {
            "rows": 128,
            "min_rows": 96,
            "sweeps": 84,
            "matrix_free": True,
            "linked_rows": True,
            "velocity_limit_pass": True,
        },
    ),
    ("whitened", (64, 32, 18, "120"), {"rows": 32, "sweeps": 42, **_INK}),
    ("whitened_block", (64, 32, 18, "120"), {"rows": 32, "sweeps": 42, "inkernel_response": (12, 6, 0, 12)}),
    (
        "local_without_momentum",
        (64, 32, 18, "120"),
        {"rows": 32, "sweeps": 24, "nesterov": False, "skip_local_internal_worlds": True},
    ),
)
"""``(label, arguments, options)`` of the factory: the conformance routes' tiers and the other variants."""


@contextlib.contextmanager
def _templates():
    """Build kernel templates without building their kernels."""
    with mock.patch.object(kernel_template.KernelTemplate, "kernel", lambda self: self):
        yield


def _tier(factory, args, kwargs):
    with _templates():
        return factory(*args, **kwargs)


def _source(args, kwargs) -> str:
    """The tier's native body, after its preludes."""
    snippet = _tier(_TIER, args, kwargs).kernel.snippet
    return snippet[snippet.index("    const int lane = LANE;\n") :]


class TestParallelTier(unittest.TestCase):
    """The tier's options, sources and votes."""

    def test_invalid_options_raise(self):
        args = (64, 32, 18, "120")
        invalid = (
            {"rows": 0, "sweeps": 1},
            {"rows": 72, "sweeps": 1},
            {"rows": 30, "sweeps": 1},
            {"rows": 32, "sweeps": 1, "min_rows": 32},
            {"rows": 32, "sweeps": 1, "inkernel_response": (12, 7, 0, 12)},
            {"rows": 4, "sweeps": 1, "inkernel_response": (12, 6, 0, 12)},
            {"rows": 32, "sweeps": 1, "row_inputs": _keyed(False)},
            {"rows": 32, "sweeps": 1, **_INK, "has_drive_rows": True, "row_inputs": _keyed(False)},
            {"rows": 32, "sweeps": 1, "inkernel_response": (12, 6, 0, 12), "row_inputs": _keyed(False)},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    _tier(_TIER, args, kwargs)
        with self.assertRaisesRegex(ValueError, "rows above 96 need the matrix-free variant"):
            _tier(_TIER, (128, 32, 18, "120"), {"rows": 100, "sweeps": 1})

    def test_worlds_are_entered_as_the_incremental_tier_enters_them(self):
        incremental, _ = sweep_routes.incremental_source(64, 18, incremental_rows=32, skip_local_internal_worlds=True)
        self.assertIn(dense_tier_world(32, 0, 18, 64, local=True, rows="n_rows_total"), incremental)

    def test_every_world_ends_with_a_barrier(self):
        """The world queue's next world restages the shared rows only after every lane has read this world's."""
        for label, args, kwargs in CALLS:
            with self.subTest(label):
                source = _source(args, kwargs)
                self.assertTrue(source.rstrip().endswith("SYNC();\n#endif"), source[-200:])

    def test_drive_rows_take_the_drive_law_only_with_drive_rows(self):
        args = (64, 32, 18, "120")
        driven = _source(args, {"rows": 32, "sweeps": 1, "has_drive_rows": True})
        self.assertIn("ProjectionLaw::physx_drive(s_y[i], r - s_drive_rhs[i]", driven)
        undriven = _source(args, {"rows": 32, "sweeps": 1})
        self.assertNotIn("physx_drive", undriven)
        self.assertNotIn("s_drive", undriven)
        self.assertIn("diag > 0.0f && row_type != 1) ? 1 : 0;", undriven)

    def test_formed_diagonals_freeze_their_rows_after_the_staging(self):
        """A row the formed diagonal freezes changes its kind only once every lane has read the kinds."""
        source = _source((64, 32, 18, "120"), {"rows": 32, "sweeps": 1, **_INK, "exact_row_sums": True})
        self.assertIn("else { s_step[i] = 0.0f; frozen = 1; }", source)
        self.assertNotIn("s_kind[i] = -1", source)
        freeze = source.index("if (frozen) s_kind[lane] = -1;\n    SYNC();")
        self.assertLess(source.index("bound += fabsf(acc);"), freeze)
        self.assertLess(freeze, source.index("step_pair = fminf(step_a, step_b)"))

    def test_several_warps_stop_on_the_block_vote(self):
        one = _source((64, 32, 18, "120"), {"rows": 32, "sweeps": 1})
        self.assertIn(EXACT_EXIT.rstrip(), one)
        block = _source((96, 32, 18, "120"), {"rows": 64, "min_rows": 32, "sweeps": 1})
        self.assertIn("if (!ANY(iteration_changed != 0)) break;", block)
        self.assertNotIn("BALLOT", block)


class TestJacobiLaw(unittest.TestCase):
    """The core's Jacobi law: its vote, its exit and its policy."""

    def test_the_jacobi_vote_follows_the_policy(self):
        """A row votes every change under the exact law and changes beyond the policy's slack or velocity bound."""
        rows = JacobiRows(32, 32)
        op = Block(18, 33, Lanes(32), rows="n_rows", layout="row")
        exact = jacobi_sweeps(rows, op, RowLaws(), sweeps=1, momentum=True, pooled=False)
        self.assertIn("if (x_new != x_old) {\n", exact)
        self.assertIn("iteration_changed = 1;", exact)
        slack = RowLaws(policy=SolvePolicy(relative_change_tolerance=1.0e-4))
        self.assertIn(
            "if (!(fabsf(x_new - x_old) <= 0.0001f * fabsf(x_new))) iteration_changed = 1;",
            jacobi_sweeps(rows, op, slack, sweeps=1, momentum=True, pooled=False),
        )
        scaled = JacobiRows(32, 32, scales=True)
        traced = RowLaws(policy=SolvePolicy(readout="trace", velocity_change_tolerance=1.0e-3))
        text = jacobi_sweeps(scaled, op, traced, sweeps=7, momentum=True, pooled=False)
        self.assertIn("solve_trace_change(fabsf(x_new - x_old) * (s_diag[i]), s_type[i]);", text)
        self.assertIn("!(fabsf(x_new - x_old) * (s_diag[i]) <= 0.001f)", text)
        self.assertIn("sweep + 1 == 7", text)

    def test_the_default_exit_and_votes_are_unchanged(self):
        self.assertEqual(end_sweep(SolvePolicy(), iteration="iter"), EXACT_EXIT)
        self.assertEqual(change_vote(RowLaws(), (("d", "A", "0"),), "  "), "  iteration_changed = 1;")
        block = end_sweep(SolvePolicy(), iteration="sweep", block=True)
        self.assertEqual(
            block,
            "        if (global_iter >= friction_start_iteration) {\n"
            "            if (!ANY(iteration_changed != 0)) break;\n        }\n",
        )

    def test_the_velocity_law_and_trace_need_staged_scales(self):
        rows = JacobiRows(32, 32)
        op = Block(18, 33, Lanes(32), rows="n_rows", layout="row")
        for policy in (SolvePolicy(readout="trace"), SolvePolicy(velocity_change_tolerance=1.0e-3)):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "staged diagonals"):
                jacobi_sweeps(rows, op, RowLaws(policy=policy), sweeps=1, momentum=True, pooled=False)
        counted = jacobi_sweeps(
            rows, op, RowLaws(policy=SolvePolicy(readout="sweeps")), sweeps=3, momentum=True, pooled=False
        )
        self.assertIn("solve_readout_sweeps(world, sweep + 1, sweep_stop);", counted)

    def test_a_pooled_pair_reads_its_ring_from_the_staged_impulses(self):
        source = _source((64, 32, 18, "120"), {"rows": 32, "sweeps": 1, "linked_rows": True})
        self.assertIn("RING_LOAD(s_x[p], p, n_rows,", source)
        self.assertIn("return next < n_rows ? next : -1;", source)
        point = _source((64, 32, 18, "120"), {"rows": 32, "sweeps": 1})
        self.assertIn("const float xp = READ(x_new, p & 31);", point)
        self.assertNotIn("RING_LOAD", point)


class TestResponseOperators(unittest.TestCase):
    """The tiers' response operators: one block build for both layouts, the whitened rows' triangular solves."""

    def test_one_build_serves_both_layouts(self):
        column = Block(18, 32, Lanes(32)).assemble()
        self.assertIn("const float* Jr = &J_world.data[jy_world_base + r * 18];", column)
        self.assertIn("const float* Yi = &Y_world.data[jy_world_base + i * 18];", column)
        row = Block(18, 33, Lanes(32), rows="n_rows", layout="row").assemble()
        self.assertIn("const float* Ji = &J_world.data[jy_world_base + i * 18];", row)
        self.assertIn("const float* Yr = &Y_world.data[jy_world_base + r * 18];", row)
        for text in (column, row):
            self.assertIn("for (int d = 0; d < 18; ++d) a += J", text)
        self.assertEqual(Block(18, 32).entry("s", "t")[1], "s * 32 + t")
        self.assertEqual(Block(18, 33, layout="row").entry("s", "t")[1], "t * 33 + s")
        with self.assertRaisesRegex(ValueError, "row layout"):
            Block(18, 32, schedule="rows")

    def test_the_rows_schedule_holds_each_lanes_row(self):
        rows = Block(40, 33, Lanes(32), rows="n_rows", layout="row", schedule="rows").assemble()
        self.assertIn("float Ji[40];", rows)
        self.assertIn("a += Ji[d] * Yr[d];", rows)

    def test_a_separate_reciprocal_diagonal(self):
        solve = CholeskySolve(6, "s_L + 144", inverse="s_Dinv + 12", masked=True)
        forward = solve.whiten(lambda k: f"z[{k}]", lambda k: f"z[{k}]")
        self.assertTrue(forward.startswith("#pragma unroll 1\n"))
        # The leaf-first factor: factor coordinate ``k`` of a vector sits at position ``6 - 1 - k``.
        self.assertIn("for(int k=0;k<6;++k)if(k<row)value-=(s_L + 144)[row*6+k]*z[(6-1-(k))];", forward)
        self.assertIn("z[(6-1-(row))]=value*(s_Dinv + 12)[row];", forward)
        self.assertNotIn("diagonal", forward)
        transpose = solve.leave(lambda k: f"v[{k}]", lambda k: f"v[{k}]")
        self.assertIn("v[(6-1-(row))]=value*(s_Dinv + 12)[row];\n        }", transpose)
        self.assertEqual(
            solve.stage_inverse("lane"), "if (lane < 6) (s_Dinv + 12)[lane] = 1.0f / (s_L + 144)[lane * 6 + lane];"
        )


def _kernel_sass():
    """The offline compiler of ``tools/fpgs_bench`` when the CUDA toolkit is on ``PATH``, else ``None``."""
    if not (shutil.which("ptxas") and shutil.which("cuobjdump")):
        return None
    root = Path(__file__).resolve().parents[2]
    path = root / "tools" / "fpgs_bench" / "kernel_sass.py"
    if not path.exists():
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("fpgs_kernel_sass", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(_kernel_sass() is None, "the CUDA toolkit (ptxas, cuobjdump) is not on PATH")
class TestOfflineCompile(unittest.TestCase):
    """The tiers compile for the RTX PRO 6000 (sm_120) without stack or local memory."""

    def test_tiers_compile(self):
        sass = _kernel_sass()
        for label, args, kwargs in CALLS[:5]:
            with self.subTest(label):
                tier = sweep_routes_tiers.parallel_kernel(*args, **kwargs)
                new = sass.compile_kernel(tier.kernel, arch=120, block_dim=tier.block_dim)
                self.assertEqual(new["stack"], 0)
                self.assertEqual(new["local"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

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

from newton._src.solvers.feather_pgs import kernel_template, sweep_routes_tiers
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout
from newton._src.solvers.feather_pgs.keyed_rows import get_direct_parallel_inputs
from newton._src.solvers.feather_pgs.response_operator import CholeskySolve, Lanes, TransposedResponse, WhitenedRows
from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy, bound
from newton._src.solvers.feather_pgs.sweep import (
    JacobiRows,
    RowLaws,
    end_sweep,
    jacobi_sweeps,
)

_TIER = sweep_routes_tiers.parallel_kernel.__wrapped__

# The exit of a warp's sweep and of a block's, without readout: stop a quiet sweep once friction is active.
_EXIT = "const bool sweep_stop = global_iter >= friction_start_iteration && BALLOT(iteration_changed != 0) == 0u;"
_BLOCK_EXIT = "const bool sweep_stop = global_iter >= friction_start_iteration && !ANY(iteration_changed != 0);"
_EXIT_TEXT = f"        {{\n            {_EXIT}\n            if (sweep_stop) break;\n        }}\n"

_LAYOUT = KeyedLayout(12, 6, 0, 12, 32, False, 0, None)


def _keyed(linked: bool):
    """The keyed rows' operands as the solver's keyed parallel factory binds them."""
    return functools.partial(get_direct_parallel_inputs, _LAYOUT, patch=linked)


_INK = {"inkernel_response": (12, 6, 0, 12)}

CALLS = (
    # The conformance routes' configurations (32 rows, one warp), then the wider tiers and the other variants.
    ("whitened", (64, 32, 18, "120"), {"rows": 32, "sweeps": 112, **_INK}),
    ("whitened_linked", (64, 32, 18, "120"), {"rows": 32, "sweeps": 112, **_INK, "linked_rows": True}),
    ("keyed", (64, 32, 18, "120"), {"rows": 32, "sweeps": 112, **_INK, "row_inputs": _keyed(False)}),
    (
        "keyed_linked",
        (64, 32, 18, "120"),
        {"rows": 32, "sweeps": 112, **_INK, "row_inputs": _keyed(True), "linked_rows": True},
    ),
    ("one_group", (64, 32, 18, "120"), {"rows": 32, "sweeps": 42, "inkernel_response": (18, 0, 0, 0)}),
    ("rows_64", (96, 32, 18, "120"), {"rows": 64, "min_rows": 32, "sweeps": 60, **_INK}),
    (
        "rows_96_last_word",
        (96, 32, 18, "120"),
        {"rows": 96, "min_rows": 64, "sweeps": 73, **_INK, "linked_rows": True, "velocity_limit_pass": True},
    ),
    (
        "rows_128",
        (128, 32, 18, "120"),
        {"rows": 128, "min_rows": 96, "sweeps": 84, **_INK, "linked_rows": True, "velocity_limit_pass": True},
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


def _operator() -> TransposedResponse:
    """A 32-row tier's response operator over 18 coordinates, whitened from groups of 12 and 6."""
    lanes = Lanes(32)
    return TransposedResponse(18, 32, WhitenedRows(12, 6, 0, 12, 18, 64, 36, lanes), lanes)


class TestParallelTier(unittest.TestCase):
    """The tier's options, sources and votes."""

    def test_invalid_options_raise(self):
        args = (64, 32, 18, "120")
        invalid = (
            {"rows": 0, "sweeps": 1, **_INK},
            {"rows": 72, "sweeps": 1, **_INK},
            {"rows": 30, "sweeps": 1, **_INK},
            {"rows": 32, "sweeps": 1, "min_rows": 32, **_INK},
            {"rows": 32, "sweeps": 1, "inkernel_response": (12, 7, 0, 12)},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    _tier(_TIER, args, kwargs)
        with self.assertRaisesRegex(ValueError, "rows must be in"):
            _tier(_TIER, (256, 32, 18, "120"), {"rows": 132, "sweeps": 1, **_INK})

    def test_every_world_ends_with_a_barrier(self):
        """The world queue's next world restages the shared rows only after every lane has read this world's."""
        for label, args, kwargs in CALLS:
            with self.subTest(label):
                source = _source(args, kwargs)
                # The CUDA body ends at the CPU branch, which fails a CPU build (device_preludes.CUDA_ONLY).
                body = source[: source.rindex("#else\n")]
                self.assertTrue(body.rstrip().endswith("SYNC();"), body[-200:])

    def test_formed_diagonals_freeze_their_rows_after_the_staging(self):
        """A row the formed diagonal freezes changes its kind only once every lane has read the kinds; a friction row
        is never frozen: it takes its block's step."""
        source = _source((64, 32, 18, "120"), {"rows": 32, "sweeps": 1, **_INK})
        self.assertIn("else { s_step[i] = 0.0f; frozen = s_kind[i] != 1; }", source)
        self.assertNotIn("s_kind[i] = -1", source)
        freeze = source.index("if (frozen) s_kind[lane] = -1;\n    SYNC();")
        self.assertLess(source.index("bound += fabsf(Ji[d]) * s_colsum[d];"), freeze)
        self.assertLess(freeze, source.index("float step_block = step_a > 0.0f ? step_a : step_b;"))

    def test_several_warps_stop_on_the_block_vote(self):
        one = _source((64, 32, 18, "120"), {"rows": 32, "sweeps": 1, **_INK})
        self.assertIn(_EXIT, one)
        block = _source((96, 32, 18, "120"), {"rows": 64, "min_rows": 32, "sweeps": 1, **_INK})
        self.assertIn(_BLOCK_EXIT, block)
        self.assertNotIn("BALLOT", block)


class TestJacobiLaw(unittest.TestCase):
    """The core's Jacobi law: its vote, its exit and its policy."""

    def test_the_jacobi_vote_follows_the_exit_law(self):
        """A row's step from the extrapolated impulse votes against its bound: the floor of its residual times its
        step size (its right-hand side and terms, and its impulse before the step times its step's share of a
        Gauss-Seidel step, ``A_ii s_i``)."""
        rows = JacobiRows(32, 32)
        op = _operator()
        text = jacobi_sweeps(rows, op, RowLaws(regularize=False), sweeps=1, pooled=False)
        self.assertNotIn("if (x_new != x_old) {", text)
        self.assertIn("const float row_share = row_diag * row_step;", text)
        self.assertIn(
            f"row_bound = {bound('row_rhs', 'fabsf((r) - (row_rhs))', 'row_share * xi', step='row_step')};", text
        )
        self.assertIn("x_from = s_y[i];", text)
        self.assertIn("if (x_new != x_from) {", text)
        self.assertIn("(!(fabsf(x_new - x_from) <= row_bound))", text)
        typed = JacobiRows(32, 32, types=True)
        traced = RowLaws(regularize=False, policy=SolvePolicy(readout="trace"))
        text = jacobi_sweeps(typed, op, traced, sweeps=7, pooled=False)
        self.assertIn("solve_trace_change(fabsf(x_new - x_from) * (row_inv_step), s_type[i]);", text)
        self.assertIn("sweep + 1 == 7", text)

    def test_the_exit_stops_a_quiet_sweep_once_friction_is_active(self):
        self.assertEqual(end_sweep(SolvePolicy(), iteration="iter"), _EXIT_TEXT)
        block = end_sweep(SolvePolicy(), iteration="sweep", block=True)
        self.assertIn(_BLOCK_EXIT, block)

    def test_the_trace_needs_staged_types(self):
        rows = JacobiRows(32, 32)
        op = _operator()
        with self.assertRaisesRegex(ValueError, "staged types"):
            jacobi_sweeps(
                rows, op, RowLaws(regularize=False, policy=SolvePolicy(readout="trace")), sweeps=1, pooled=False
            )
        counted = jacobi_sweeps(
            rows, op, RowLaws(regularize=False, policy=SolvePolicy(readout="sweeps")), sweeps=3, pooled=False
        )
        self.assertIn("solve_readout_sweeps(world, sweep + 1, sweep_stop);", counted)

    def test_a_pooled_pair_reads_its_ring_from_the_staged_impulses(self):
        source = _source((64, 32, 18, "120"), {"rows": 32, "sweeps": 1, **_INK, "linked_rows": True})
        self.assertIn("RING_LOAD(s_x[p], p, n_rows,", source)
        self.assertIn("return next < n_rows ? next : -1;", source)
        point = _source((64, 32, 18, "120"), {"rows": 32, "sweeps": 1, **_INK})
        self.assertIn("const float xp = READ(x_new, p & 31);", point)
        self.assertNotIn("RING_LOAD", point)


class TestResponseOperators(unittest.TestCase):
    """The tiers' response operator: the whitened rows' triangular solves."""

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

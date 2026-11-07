# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source checks of the parallel (Jacobi) tiers as configurations of the sweep emitter.

No GPU is needed: each tier takes its legacy factory's arguments and declares the same ABI, dispatch and tier
descriptor, rejects the same options, emits the Jacobi law the core composes, and, with the CUDA toolkit on ``PATH``,
compiles offline next to the legacy kernel.
"""

import contextlib
import importlib.util
import re
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

from newton._src.core.execution import Work, shape
from newton._src.core.memory import Buffer, Staging, built_for
from newton._src.solvers.feather_pgs import kernel_template, sweep_routes_tiers
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout
from newton._src.solvers.feather_pgs.response_operator import Lanes, TransposedResponse, WalkedRows
from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy, bound
from newton._src.solvers.feather_pgs.sweep import (
    JacobiRows,
    RowLaws,
    end_sweep,
    jacobi_scale,
    jacobi_sweeps,
    sweep_kernel,
)
from newton.tests.feather_pgs_law_harness import built, keyed_classes

_TIER = built(sweep_routes_tiers.TierSweep)


def _scale(aii, bound, pad):
    """The tiers' step scale (:func:`jacobi_scale`), with the rows' law's cfm as the tiers read it."""
    return jacobi_scale(aii, bound, cfm="cfm", pad=pad)


# The exit of a warp's sweep and of a block's, without readout: stop a quiet sweep once friction is active.
_EXIT = "const bool sweep_stop = BALLOT(iteration_changed != 0) == 0u;"
_BLOCK_EXIT = "const bool sweep_stop = !ANY(iteration_changed != 0);"
_EXIT_TEXT = f"        {{\n            {_EXIT}\n            if (sweep_stop) break;\n        }}\n"

_LAYOUT = KeyedLayout(12, 6, 0, 12, 32, False, 0, None)
_CLASSES = keyed_classes(_LAYOUT)


_INK = {"inkernel_response": (12, 6, 0, 12)}

CALLS = (
    # The conformance routes' configurations (32 rows, one warp), then the wider tiers and the other variants.
    ("whitened", (64, 32, 18, "120"), {"rows": 32, "sweeps": 112, **_INK}),
    ("whitened_linked", (64, 32, 18, "120"), {"rows": 32, "sweeps": 112, **_INK, "linked_rows": True}),
    (
        "keyed",
        (64, 32, 18, "120"),
        {"rows": 32, "sweeps": 112, **_INK, "keyed": _LAYOUT, "classes": _CLASSES},
    ),
    (
        "keyed_linked",
        (64, 32, 18, "120"),
        {"rows": 32, "sweeps": 112, **_INK, "keyed": _LAYOUT, "classes": _CLASSES, "linked_rows": True},
    ),
    ("one_group", (64, 32, 18, "120"), {"rows": 32, "sweeps": 42, "inkernel_response": (18, 0, 0, 0)}),
    ("rows_64", (96, 32, 18, "120"), {"rows": 64, "sweeps": 60, **_INK}),
    (
        "rows_96_last_word",
        (96, 32, 18, "120"),
        {"rows": 96, "sweeps": 73, **_INK, "linked_rows": True, "velocity_limit_pass": True},
    ),
    (
        "rows_128",
        (128, 32, 18, "120"),
        {"rows": 128, "sweeps": 84, **_INK, "linked_rows": True, "velocity_limit_pass": True},
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
    snippet = _tier(_TIER, args, kwargs).snippet
    return snippet[snippet.index("    const int lane = LANE;\n") :]


def _operator() -> TransposedResponse:
    """A 32-row tier's response operator over 18 coordinates, the walked rows of trees of 12 and 6."""
    lanes = Lanes(32)
    return TransposedResponse(18, 32, WalkedRows(12, 6, 0, 12, 18, 36, lanes, read=""), lanes)


class TestParallelTier(unittest.TestCase):
    """The tier's options, sources and votes."""

    def test_invalid_options_raise(self):
        args = (64, 32, 18, "120")
        invalid = (
            {"rows": 0, "sweeps": 1, **_INK},
            {"rows": 72, "sweeps": 1, **_INK},
            {"rows": 30, "sweeps": 1, **_INK},
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

    def test_the_cpu_runs_the_lanes_in_turn(self):
        """On the CPU one thread runs the block's lanes as virtual lanes: each phase a loop over the lanes with their
        registers bound by name, no block barrier, the body on both devices (no CUDA-only branch)."""
        for label, args, kwargs in CALLS:
            with self.subTest(label):
                lanes = 32 * ((kwargs["rows"] + 31) // 32)
                source = _source((*args[:3], "0"), kwargs)
                self.assertIsNone(re.search(r"(?<!WARP_)SYNC\(\)", source))
                self.assertNotIn("#error", source)
                self.assertIn(f"float x_new_lanes[{lanes}];", source)
                self.assertIn(f"for (int lane = 0; lane < {lanes}; ++lane) {{", source)
                self.assertIn("auto& Jr = Jr_lanes[lane];", source)

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
        block = _source((96, 32, 18, "120"), {"rows": 64, "sweeps": 1, **_INK})
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
        text = jacobi_sweeps(rows, op, RowLaws(regularize=False), sweeps=1, pooled=False, scale=_scale)
        self.assertNotIn("if (x_new != x_old) {", text)
        self.assertIn("float row_share = row_diag * row_step;", text)
        self.assertIn(
            f"row_bound = {bound('row_rhs', 'fabsf((r) - (row_rhs))', 'row_share * xi', step='row_step')};", text
        )
        self.assertIn("x_from = s_y[i];", text)
        self.assertIn("if (x_new != x_from) {", text)
        self.assertIn("(!(fabsf(x_new - x_from) <= row_bound))", text)
        typed = JacobiRows(32, 32, types=True)
        traced = RowLaws(regularize=False, policy=SolvePolicy(readout="trace"))
        text = jacobi_sweeps(typed, op, traced, sweeps=7, pooled=False, scale=_scale)
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
                rows,
                op,
                RowLaws(regularize=False, policy=SolvePolicy(readout="trace")),
                sweeps=1,
                pooled=False,
                scale=_scale,
            )
        counted = jacobi_sweeps(
            rows,
            op,
            RowLaws(regularize=False, policy=SolvePolicy(readout="sweeps")),
            sweeps=3,
            pooled=False,
            scale=_scale,
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
    """The tiers' response operator: rows by the walk, the change to the step's one leave, no factor staged."""

    def test_the_tiers_walk_their_rows_and_leave_by_the_walk(self):
        for label, args, kwargs in CALLS:
            with self.subTest(label):
                source = _source(args, kwargs)
                self.assertNotIn("s_L", source)
                # The change goes to the step's one leave in the walk's coordinates (no walk down in the sweep); the
                # keyed rows walk their seeds, the others read the one writer's.
                self.assertIn("if (global_dof >= 0) walk.du.data[global_dof] += u;", source)
                self.assertNotIn("WALK_DOWN", source)
                keyed = "keyed" in kwargs
                self.assertEqual("TIER_WALK(" in source, keyed)
                self.assertEqual("incident = row_incident.data[off_dense + i];" in source, not keyed)
                self.assertIn("s_rhs[i] += incident;", source)

    def test_the_release_compiles_where_a_tree_can_slip(self):
        """A slipping drive's release ``R`` (the walk's ``WALK_RELEASE``) on the keyed rows, only on the trees of the
        capacity the tier names; the stored rows carry ``R`` from their writer. Its transpose is the step's one leave's
        (``factor.walk.walk_leave_kernel``), never the tier's."""
        for label, args, kwargs in (CALLS[0], CALLS[2]):
            with self.subTest(label):
                self.assertNotIn("WALK_RELEASE(walk.release", _source(args, kwargs))
                framed = _source(args, {**kwargs, "frame": (4, 0)})
                self.assertNotIn("WALK_RELEASE_T(", framed)
                keyed = "keyed" in kwargs
                self.assertEqual(framed.count("WALK_RELEASE(walk.release, art_a"), 1 if keyed else 0)


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

    def test_a_tier_states_the_blocks_shared_memory_keeps(self):
        """Bound each tier by the blocks its shared memory keeps; a staging that leaves no registers to spare has none."""
        for label, args, kwargs in CALLS:
            with self.subTest(tier=label):
                threads, blocks = _tier(_TIER, args, kwargs).launch_bounds
                self.assertEqual(threads, 32 * ((kwargs["rows"] + 31) // 32))
                self.assertLessEqual(blocks * threads // 32, 24)  # 24 warps at a line of 80 registers

        def bounds(staging):
            line = sweep_routes_tiers.REGISTER_LINE
            work = Work(0, block=staging.threads, registers=line, cap=False, shared=staging.bytes)
            return shape(work, built_for("sm_120")).bounds

        self.assertEqual(bounds(Staging((Buffer("s_x", "float", 4400),), threads=128)), (128, 5))
        # 25 blocks of a warp: more than the line keeps, where a bound could only take registers from the sweep.
        self.assertIsNone(bounds(Staging((Buffer("s_x", "float", 736),), threads=32)))

    def test_tiers_compile(self):
        sass = _kernel_sass()
        for label, args, kwargs in CALLS[:5]:
            with self.subTest(label):
                config = sweep_routes_tiers.TierSweep(*args, **kwargs)
                new = sass.compile_kernel(sweep_kernel(config), arch=120, block_dim=config.threads)
                self.assertEqual(new["stack"], 0)
                self.assertEqual(new["local"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

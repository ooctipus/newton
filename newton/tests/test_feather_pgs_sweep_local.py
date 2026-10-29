# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source checks of the local owners' and the paired factor sweeps as configurations of the sweep emitter.

No GPU is needed: each kernel rejects the options it does not admit, declares its staging and, with the CUDA toolkit
on ``PATH``, compiles offline without local memory.
"""

import importlib.util
import itertools
import shutil
import sys
import unittest
from pathlib import Path

from newton._src.solvers.feather_pgs import row_packets, sweep_routes_local
from newton._src.solvers.feather_pgs.response_operator import (
    CholeskySolve,
    FactorComponent,
    FactorRows,
    FreeBodyZ,
    Lanes,
    LocalBlock,
)
from newton._src.solvers.feather_pgs.solve_policy import EXACT_POLICY, SolvePolicy

_LOCAL = sweep_routes_local.local_owned_kernel.__wrapped__
_PAIRED = sweep_routes_local.paired_factor_kernel.__wrapped__

_PAIR = {"paired_dof_count": 6, "persistent_queue": True, "warps_per_block": 2}

LOCAL_CALLS = (
    # The single, pair and residual owners as the solver builds them (the gripper scenes' sizes), and the other lane
    # layouts of the single owner.
    ("single", (64, 3, 3, "120"), {"lanes_per_world": 16, "contact_capable": False}),
    (
        "single_l8_w2",
        (192, 9, 9, "120"),
        {"lanes_per_world": 8, "contact_capable": False, "warps_per_block": 2},
    ),
    ("single_l32", (64, 16, 16, "120"), {"contact_capable": False}),
    ("pair", (64, 20, 3, "120"), _PAIR),
    ("pair_w1", (64, 20, 3, "120"), {"paired_dof_count": 6}),
    ("residual", (64, 40, 3, "120"), {**_PAIR, "mf_max_constraints": 512, "local_mf_max_constraints": 12}),
)
"""``(label, arguments, options)`` of the local owners' live configurations."""

PAIRED_SCHEDULES = ({}, {"contact_triples": True}, {"contact_units": True})


def _row_inputs(primary: int):
    """The row packets' row inputs for a primary articulation of ``primary`` coordinates and a free body."""
    return row_packets.get_local_factory.__wrapped__(primary, 6, 40).keywords["row_inputs"]


def _local_variants():
    for (label, args, options), packets, policy in itertools.product(
        LOCAL_CALLS, (False, True), (EXACT_POLICY, SolvePolicy(readout="sweeps"))
    ):
        kwargs = {**options, "solve_policy": policy}
        if packets:
            kwargs["row_inputs"] = _row_inputs(args[2])
        yield f"{label}{'_packets' if packets else ''}{policy.suffix}", args, kwargs


def _paired_variants():
    for schedule, components, policy in itertools.product(
        PAIRED_SCHEDULES, (False, True), (EXACT_POLICY, SolvePolicy(readout="sweeps"))
    ):
        kwargs = {**schedule, "independent_components": components, "solve_policy": policy}
        yield f"{list(schedule) or 'general'} components={components}{policy.suffix}", (64, 23, 17, 6, "120"), kwargs


def _block(body: str, head: str) -> str:
    """The braced block that opens with ``head`` in ``body``."""
    start = body.index(head)
    depth = 0
    for index in range(start, len(body)):
        depth += {"{": 1, "}": -1}.get(body[index], 0)
        if depth == 0 and body[index] == "}":
            return body[start : index + 1]
    raise ValueError(f"unbalanced block {head!r}")


class TestLocalOwnedRoute(unittest.TestCase):
    """The local owners' sweeps (``sweep_routes_local.local_owned_kernel``)."""

    def test_variants_build(self):
        """Every live configuration builds, over packet rows too and with a counted readout."""
        for label, args, kwargs in _local_variants():
            with self.subTest(label):
                self.assertTrue(_LOCAL(*args, **kwargs).key.startswith("sweep_local"))

    def test_invalid_options_raise(self):
        invalid = (
            {"warps_per_block": 3},
            {"lanes_per_world": 4},
            {"contact_capable": False, "paired_dof_count": 6},
            {"local_mf_max_constraints": 12, "mf_max_constraints": 64},
            {"paired_dof_count": 6, "local_mf_max_constraints": 12},
            {"lanes_per_world": 8},
            {"lanes_per_world": 16, "contact_capable": False, "paired_dof_count": 0, "warps_per_block": 1},
        )
        for options in invalid:
            args = (64, 20, 3 if options.get("lanes_per_world") != 16 else 17, "120")
            with self.subTest(options), self.assertRaises(ValueError):
                _LOCAL(*args, **options)

    def test_every_row_ends_with_a_barrier(self):
        """Lanes read residuals other lanes updated: every row's update ends with the group's barrier."""
        for label, args, kwargs in LOCAL_CALLS:
            with self.subTest(label):
                options = {key: value for key, value in kwargs.items() if key != "persistent_queue"}
                body, _ = sweep_routes_local.local_owned_source(*args[:3], **options)
                loops = ["for (int i = 0; i < row_count; ++i) {"]
                if options.get("local_mf_max_constraints"):
                    loops.append("for (int i = 0; i < mf_count; ++i) {")
                for head in loops:
                    rows = _block(body, head)
                    self.assertTrue(rows[: rows.rindex("}")].rstrip().endswith("SYNC();"), head)

    def test_staging_declares_one_struct_per_group(self):
        _, staging = sweep_routes_local.local_owned_source(
            192, 9, 9, lanes_per_world=8, contact_capable=False, warps_per_block=2
        )
        self.assertEqual(staging.groups, 8)
        body, _ = sweep_routes_local.local_owned_source(
            192, 9, 9, lanes_per_world=8, contact_capable=False, warps_per_block=2
        )
        self.assertIn("__shared__ LocalScratch localscratch[8];", body)
        self.assertIn("#define s_A localscratch[SUB].s_A", body)
        self.assertIn("#undef s_A", body)


class TestPairedFactorRoute(unittest.TestCase):
    """The paired factor sweep (``sweep_routes_local.paired_factor_kernel``)."""

    def test_variants_build(self):
        """Every schedule builds, with and without independent components and with a counted readout."""
        for label, args, kwargs in _paired_variants():
            with self.subTest(label):
                self.assertTrue(_PAIRED(*args, **kwargs).key)

    def test_invalid_options_raise(self):
        invalid = (
            ((0, 23, 17, 6, "120"), {}),
            ((64, 24, 17, 6, "120"), {}),
            ((64, 36, 30, 6, "120"), {}),
            ((64, 23, 23, 0, "120"), {}),
            ((64, 23, 17, 6, "120"), {"contact_triples": True, "contact_units": True}),
        )
        for args, kwargs in invalid:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                _PAIRED(*args, **kwargs)

    def test_unit_schedules_reduce_a_pair_in_one_ladder(self):
        for schedule in PAIRED_SCHEDULES[1:]:
            body, _ = sweep_routes_local.paired_factor_source(64, 23, 17, 6, **schedule)
            self.assertIn("SUM(residual, sibling_residual, cross);", body)
        body, _ = sweep_routes_local.paired_factor_source(64, 23, 17, 6)
        self.assertIn("SUM(sibling_residual, cross);", body)
        self.assertNotIn("SUM(residual, sibling_residual, cross);", body)

    def test_triple_prefix_holds_no_friction_pairs(self):
        body, _ = sweep_routes_local.paired_factor_source(64, 23, 17, 6, contact_triples=True)
        prefix = _block(body, "for (int i = 0; i < contact_start; ++i) {")
        self.assertNotIn("RING_LOAD", prefix)
        self.assertNotIn("friction_gate", prefix)


class TestResponseOperators(unittest.TestCase):
    """The operators the two routes add: the local response block and the factor-coordinate rows."""

    def test_reciprocal_cholesky_multiplies_by_the_staged_inverse_diagonal(self):
        solve = CholeskySolve(3, "s_L", reciprocal=True, ascending=True)
        staged = solve.stage(lambda i: f"L[{i}]", "lane", 32)
        self.assertIn("if (i % 4 == 0) factor = factor != 0.0f ? 1.0f / factor : 0.0f;", staged)
        out = lambda i: f"x[{i}]"  # noqa: E731
        self.assertIn("x[(row)]=value*diagonal;", solve.forward(out, out))
        transposed = solve.transpose(out, out)
        self.assertIn("for(int step=0;step<3;++step)", transposed)
        self.assertIn("const int row=3-1-step;", transposed)
        self.assertNotIn("1.0f / factor", CholeskySolve(3, "s_L").stage(lambda i: f"L[{i}]", "lane", 32))

    def test_local_block_views_and_packed_entries(self):
        block = LocalBlock(15, (CholeskySolve(9, "s_L"), CholeskySolve(6, "s_S")), Lanes(32), free=FreeBodyZ(Lanes(32)))
        self.assertEqual(block.rows, "row_count + mf_count")
        self.assertEqual(block.residual("jv", ""), "float jv = s_res[i];")
        free = LocalBlock(15, block.factors, Lanes(32), free=FreeBodyZ(Lanes(32)), first="row_count")
        self.assertEqual(free.residual("jv", ""), "float jv = s_res[row_count + i];")
        coupling = free.coupling("cross", "i", "sib", "")
        self.assertIn("(row_count + sib) > (row_count + i)", coupling)
        self.assertIn("s_A[hi * (hi + 1) / 2 + lo]", coupling)
        with self.assertRaises(ValueError):
            LocalBlock(16, block.factors, Lanes(32)).respond(("a", "b"), "")

    def test_factor_rows_fuse_a_pair_ladder(self):
        rows = FactorRows(
            23, (FactorComponent(17, "o", "l", "L", "Linv", "b"),), Lanes(32), diagonal=lambda r: f"s_diag[{r}]"
        )
        rhs = lambda r: f"s_rhs[{r}]"  # noqa: E731
        fused = rows.pair_terms("i", "sib", rhs, "", own=True)
        self.assertIn("SUM(residual, sibling_residual, cross);", fused)
        self.assertIn("residual += s_rhs[i];", fused)
        self.assertEqual(rows.step("r", "i"), "-r / s_diag[i]")
        self.assertIn("factor_velocity += row_z * d;", rows.apply("d", ""))
        self.assertIn("factor_velocity += sibling_z * d;", rows.apply_row("sib", "d", ""))
        self.assertIn("const int l = lane - o;", rows.enter("v", ""))
        self.assertIn("Linv.data[b + k * 17 + l]", rows.leave("out", ""))


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
    """The kernels compile for the RTX PRO 6000 (sm_120) without local memory."""

    def _compile(self, kernel, block):
        self.assertEqual(_kernel_sass().compile_kernel(kernel, arch=120, block_dim=block)["local"], 0)

    def test_local_owners_compile(self):
        for label, args, kwargs in LOCAL_CALLS[3:]:
            with self.subTest(label):
                self._compile(_LOCAL(*args, **kwargs), 32 * kwargs.get("warps_per_block", 1))

    def test_paired_schedules_compile(self):
        for schedule in PAIRED_SCHEDULES:
            with self.subTest(schedule):
                args = (64, 23, 17, 6, "120")
                kwargs = {**schedule, "independent_components": True}
                self._compile(_PAIRED(*args, **kwargs), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)

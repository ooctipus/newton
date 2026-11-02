# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source checks of the local owners' and the paired factor sweeps as configurations of the sweep emitter.

No GPU is needed: each kernel rejects the options it does not admit, declares its staging and, with the CUDA toolkit
on ``PATH``, compiles offline without local memory.
"""

import importlib.util
import itertools
import re
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
from newton._src.solvers.feather_pgs.solve_policy import DEFAULT_POLICY, SolvePolicy
from newton._src.solvers.feather_pgs.tree_factor import WALK_NOW

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
    return row_packets.get_local_factory.__wrapped__(primary, 6, 40, primary + 2).keywords["row_inputs"]


def _local_variants():
    for (label, args, options), packets, policy in itertools.product(
        LOCAL_CALLS, (False, True), (DEFAULT_POLICY, SolvePolicy(readout="sweeps"))
    ):
        kwargs = {**options, "solve_policy": policy}
        if packets:
            kwargs["row_inputs"] = _row_inputs(args[2])
        yield f"{label}{'_packets' if packets else ''}{policy.suffix}", args, kwargs


def _paired_variants():
    for schedule, components, policy in itertools.product(
        PAIRED_SCHEDULES, (False, True), (DEFAULT_POLICY, SolvePolicy(readout="sweeps"))
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
        self.assertIn("SHARED LocalScratch localscratch[8];", body)
        self.assertIn("#define s_A localscratch[SUB].s_A", body)
        self.assertIn("#undef s_A", body)
        # The trees' factor is the walk's (FL3.6): no staged factor; the rows are the walk's, contactless owners
        # included, and the change goes to the step's one leave in the walk's coordinates.
        self.assertNotIn("s_L", body)
        self.assertIn("WALK_TREE(", body)
        self.assertIn("walk.du.data[dof] += s_v[d];", body)
        self.assertNotIn("WALK_DOWN", body)
        self.assertNotIn("value/diagonal", body)
        # A paired owner's free body goes the same way; the packets' owner stages the walk in shared memory.
        paired, _ = sweep_routes_local.local_owned_source(192, 20, 9, 6, warps_per_block=2)
        self.assertNotIn("s_L", paired)
        self.assertIn("walk.du.data[dof] += s_v[d];", paired)
        _, staged = sweep_routes_local.local_owned_source(192, 20, 9, 6, warps_per_block=2, row_inputs=_row_inputs(9))
        self.assertGreater(
            staged.bytes, sweep_routes_local.local_owned_source(192, 20, 9, 6, warps_per_block=2)[1].bytes
        )


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

    def test_unit_schedules_run_contact_units_through_the_row_law(self):
        """A loaded contact unit reduces its three rows' velocity terms, the pair's Gram couplings with the normal
        and their terms' magnitudes (the exit law's scale) in one ladder, then runs its rows by the row law
        (:func:`.sweep.contact_unit`): the pair's residuals moved by the normal's change, a moved tangent's terms
        bounded below, the changes deferred to one apply. An unloaded unit reduces its normal alone and its pair on
        the pair's own ladder; a pair at its first tangent in the general sweep reduces its sibling's and the
        coupling."""
        unit = (
            "SUM(unit_jv0, unit_jv0_terms, unit_jv1, unit_jv1_terms, unit_jv2, unit_jv2_terms, gram_n1, gram_n1_terms, "
            "gram_n2, gram_n2_terms, unit_cross);"
        )
        for schedule in PAIRED_SCHEDULES[1:]:
            body, _ = sweep_routes_local.paired_factor_source(64, 23, 17, 6, **schedule)
            held = "const float unit_lam0 = s_lam[head], unit_lam1 = s_lam[head + 1], unit_lam2 = s_lam[head + 2];"
            self.assertIn(held, body)
            units = _block(body, "if (unit_lam0 != 0.0f || unit_lam1 != 0.0f || unit_lam2 != 0.0f) {")
            self.assertIn(unit, units)
            self.assertIn("float jv = r == 0 ? unit_jv0 : unit_jv1 + gram_n1 * unit_dn;", units)
            self.assertIn("fmaxf(fabsf(jv), unit_jv1_terms - fabsf(unit_dn) * gram_n1_terms)", units)
            apply = "factor_velocity += (row_z * unit_dn + tangent1_z * unit_dt1) + tangent2_z * unit_dt2;"
            self.assertIn(apply, units)
            # The rows read the impulses the unit held and pool the ring from the normal's new impulse in a register,
            # so they read no store of the unit: each row's stores wait for the lanes (a skipped normal leaves the
            # pair's first), no row synchronizes after its stores, and the unit does once after its apply.
            self.assertEqual(re.findall(r"s_lam\[[^\]]*\](?! = )", units), [])
            self.assertIn("RING_LOAD(unit_normal, parent, m, ring_parent, ring_impulse)", units)
            # Nor do they read a staged right-hand side or diagonal after a barrier: the unit holds them too, loaded
            # before its ladder (the normal's check reads its declared diagonal before its barrier).
            for held in ("unit_rhs0 = s_rhs[head]", "unit_diag2 = s_diag[head + 2]"):
                self.assertLess(units.index(held), units.index(unit))
            rows = units[units.index("for (int r = 0; r < 2; ++r) {") : units.index(apply)]
            self.assertEqual(re.findall(r"s_(?:rhs|diag)\[[^\]]*\]", rows), ["s_diag[i]"])
            self.assertEqual(units.count("SYNC();"), 2)
            self.assertRegex(units, re.escape(apply) + r"\s+SYNC\(\);\s+}$")
            rest = body[body.index(units) + len(units) :]
            self.assertTrue(rest.startswith(" else {"))
            unloaded = _block(rest, "else {")
            self.assertIn("SUM(jv, jv_terms);", unloaded)
            self.assertIn("SUM(residual, residual_terms, sibling_residual, sibling_residual_terms, cross);", unloaded)
            self.assertNotIn("unit_jv", unloaded)
        body, _ = sweep_routes_local.paired_factor_source(64, 23, 17, 6)
        self.assertIn("SUM(sibling_residual, sibling_residual_terms, cross);", body)
        self.assertNotIn(unit, body)

    def test_triple_prefix_holds_no_friction_pairs(self):
        body, _ = sweep_routes_local.paired_factor_source(64, 23, 17, 6, contact_triples=True)
        prefix = _block(body, "for (int i = 0; i < contact_start; ++i) {")
        self.assertNotIn("RING_LOAD", prefix)


class TestResponseOperators(unittest.TestCase):
    """The operators the two routes add: the local response block and the factor-coordinate rows."""

    def test_reciprocal_cholesky_multiplies_by_the_staged_inverse_diagonal(self):
        solve = CholeskySolve(3, "s_L", reciprocal=True, ascending=True)
        staged = solve.stage(lambda i: f"L[{i}]", "lane", 32)
        # The reciprocals take a loop of their own: the copy's loads carry no division.
        self.assertIn("if (i % 4 != 0) (s_L)[i] = L[(i)];", staged)
        self.assertIn("(s_L)[i * 4] = factor != 0.0f ? 1.0f / factor : 0.0f;", staged)
        out = lambda i: f"x[{i}]"  # noqa: E731
        # The leaf-first factor: factor row ``row`` writes the vector's physical position ``3 - 1 - row``.
        self.assertIn("x[(3-1-(row))]=value*diagonal;", solve.whiten(out, out))
        transposed = solve.leave(out, out)
        self.assertIn("for(int step=0;step<3;++step)", transposed)
        self.assertIn("const int row=3-1-step;", transposed)
        self.assertNotIn("1.0f / factor", CholeskySolve(3, "s_L").stage(lambda i: f"L[{i}]", "lane", 32))
        # Entering factor coordinates with the staged reciprocal divides the diagonal term by it (a zero gives zero).
        entered = solve.enter(out, out)
        self.assertIn("float value=(s_L)[row*3+row]!=0.0f?x[(3-1-(row))]/(s_L)[row*3+row]:0.0f;", entered)
        self.assertIn("for(int k=row+1;k<3;++k)", entered)

    def test_local_block_views_and_packed_entries(self):
        block = LocalBlock(
            15,
            9,
            11,
            seeds=True,
            staged=True,
            wrenches=True,
            release=True,
            lanes=Lanes(32),
            free=FreeBodyZ(Lanes(32)),
        )
        self.assertEqual(block.rows, "row_count + mf_count")
        self.assertEqual(block.residual("jv", ""), "float jv = s_res[i];")
        free = LocalBlock(15, 9, lanes=Lanes(32), free=FreeBodyZ(Lanes(32)), first="row_count")
        self.assertEqual(free.residual("jv", ""), "float jv = s_res[row_count + i];")
        coupling = free.coupling("cross", "i", "sib", "")
        self.assertIn("(row_count + sib) > (row_count + i)", coupling)
        self.assertIn("s_A[hi * (hi + 1) / 2 + lo]", coupling)
        packets = row_packets.get_local_factory(9, 6, 20, 11).keywords["row_inputs"]
        with self.assertRaises(ValueError):
            LocalBlock(16, 9, 11, seeds=True, wrenches=True, lanes=Lanes(32)).respond(packets, "")
        # The packets' seeds walked in the kernel (FL3.6) on the world's staged walk: the tree's and the paired free
        # body's (a one-body tree), the release of the tree's slips along their paths, then |z|^2 in coordinate order.
        prologue = block.prologue()
        self.assertIn("WALK_LOAD_RECORD(r, walk.data.data + (size_t)d * ", prologue)
        self.assertIn("const auto walk_free = [&]() { WALK_TREE(free_path, 1, 6); };", prologue)
        self.assertNotIn("walk.data.data + (size_t)walk_d0", prologue)
        respond = block.respond(packets, "")
        self.assertIn(
            "walk_wrench(&s_Z[staged], s_v, walk_data, walk_link, walk_now, walk_now_link, walk_shift,", respond
        )
        self.assertIn(
            "walk_wrench(&s_Z[staged + 9], s_v + 9, free_data, free_link, free_now, free_now_link,\n"
            "                                    free_shift, 0, wrench);",
            respond,
        )
        self.assertIn("incident += walk_covector(&s_Z[staged], dof, weight);", respond)
        self.assertIn("WALK_RELEASE(walk.release, art, &s_Z[staged]);", respond)
        self.assertIn("row_diagonal += z * z;", respond)
        self.assertNotIn("s_L", respond)
        # Read in place, the same walks take the device walk's data (a data-chosen layout).
        in_place = LocalBlock(15, 9, 11, seeds=True, wrenches=True, lanes=Lanes(32)).prologue()
        self.assertIn(f"const float* walk_now = walk.now.data + (size_t)walk_d0 * {WALK_NOW};", in_place)
        self.assertNotIn("s_walk", in_place)
        # The one writer's stored rows: read, not whitened again; the walk read in place for the leave.
        stored_block = LocalBlock(15, 9, release=True, lanes=Lanes(32))
        stored = stored_block.respond(sweep_routes_local.StoredLocalRows(), "")
        self.assertIn("Z_row[first_p + c]", stored)
        self.assertIn("const float incident = row_incident.data[world_row_base + row];", stored)
        self.assertNotIn("WALK_", stored)
        self.assertNotIn("s_walk", stored_block.prologue())
        assembled = block.assemble(lambda r: f"rhs[{r}]", "")
        self.assertIn("a += s_Z[hi * 15 + d] * s_Z[lo * 15 + d];", assembled)
        self.assertNotIn("s_v", assembled)
        self.assertIn("side_term += mf_Z_a.data[row6 + d] * s_Z[lo * 15 + row_a + d];", assembled)
        commit = block.commit("")
        self.assertIn("u += s_Z[r * 15 + d] * s_app[r];", commit)
        self.assertIn("u += mf_Z_a.data[row6 + d - row_a] * applied;", commit)
        # No leave in the block, for either row source: the change goes to the step's one leave.
        for source in (block, stored_block):
            self.assertFalse(hasattr(source, "leave"))

    def test_factor_rows_fuse_a_pair_ladder(self):
        rows = FactorRows(23, (FactorComponent(17, "o", "l", "L", "b"),), Lanes(32), diagonal=lambda r: f"s_diag[{r}]")
        rhs = lambda r: f"s_rhs[{r}]"  # noqa: E731
        fused = rows.pair_terms("i", "sib", rhs, "", own=True)
        self.assertIn("SUM(residual, residual_terms, sibling_residual, sibling_residual_terms, cross);", fused)
        self.assertIn("float residual_terms = fabsf(residual);", fused)
        self.assertEqual(rows.terms("residual", "s_rhs[i]", whole=True), "residual_terms")
        self.assertIn("residual += s_rhs[i];", fused)
        self.assertEqual(rows.step("r", "i"), "-r / s_diag[i]")
        self.assertIn("factor_velocity += row_z * d;", rows.apply("d", ""))
        self.assertIn("factor_velocity += sibling_z * d;", rows.apply_row("sib", "d", ""))
        self.assertIn("const int l = lane - o;", rows.lanes_of("")[0])
        # Leaving runs the factor's back substitution on the lanes: no explicit inverse.
        leave = rows.leave("out", "")
        self.assertIn("value -= L.data[b + ((17-1-(s))) * 17 + ((17-1-(l)))] * x;", leave)
        self.assertIn("const float x = READ(out, source);", leave)


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

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source checks of the sweep emitter's core: the response operators' shared interface, the row visit and the row tail.

No GPU is needed: the checks read the native statements the core emits for small configurations.
"""

import dataclasses
import re
import unittest

from newton._src.solvers.feather_pgs.response_operator import Lanes, ResidualBlock, ResponseOperator, SparseFactor
from newton._src.solvers.feather_pgs.solve_policy import Change, SolvePolicy, bound
from newton._src.solvers.feather_pgs.sweep import (
    FRICTION,
    RowContext,
    RowLaws,
    UnitOperator,
    change_vote,
    contact_unit,
    friction_gate,
    friction_radius,
    row_update,
    row_visit,
)


@dataclasses.dataclass(frozen=True)
class _Rows(ResponseOperator):
    """An operator of named sums: the least an operator implements for a friction pair."""

    diagonal: object = None
    inverse: object = None
    name = "rows"

    def residual(self, out, indent):
        return f"{indent}float {out} = J_i_v;"

    def row_velocity(self, out, row, indent):
        return f"{indent}float {out} = J_v({row});"

    def coupling(self, out, row, sibling, indent):
        return f"{indent}float {out} = J_Y({row}, {sibling});"

    def apply(self, delta, indent):
        return f"\n{indent}v += Y_i * {delta};"

    def apply_row(self, row, delta, indent):
        return f"\n{indent}v += Y({row}) * {delta};"


@dataclasses.dataclass(frozen=True)
class _Block(ResidualBlock):
    """A staged block whose rows keep their whole residual, in column layout: the least a maintained block adds."""

    capacity: int
    lanes: Lanes
    rows: str = "n_rows"
    diagonal: object = None
    inverse: object = None
    name = "block"

    def entry(self, source, target):
        return "", f"{source} * {self.capacity} + {target}"


@dataclasses.dataclass(frozen=True)
class _FusedRows(_Rows):
    """The same operator applying a pair in one pass."""

    fused_pair_apply = True

    def apply_pair(self, sibling_delta, delta, indent):
        return f"\n{indent}apply_pair({sibling_delta}, {delta});"


@dataclasses.dataclass(frozen=True)
class _UnitRows(UnitOperator, _Rows):
    """The same operator's rows in a contact unit."""


def _rhs(row):
    return f"rhs[{row}]"


def _context(operator, lanes=None, **options):
    lanes = lanes or Lanes(32)
    return RowContext(
        row="i",
        kind="row_type",
        operator=operator,
        load=lambda row: f"lam[{row}]",
        store=lambda row, value: lanes.first(f"lam[{row}] = {value};"),
        rhs=_rhs,
        mu=lambda row: f"mu[{row}]",
        weight=lambda row: f"w[{row}]",
        parent="parent",
        ring_load=lambda parent: f"RING_LOAD(lam[{parent}])",
        lanes=lanes,
        **options,
    )


class TestResponseOperatorInterface(unittest.TestCase):
    """The base class's defaults: the sibling's residual, the pair's terms and the end passes' apply."""

    def test_sibling_residual_is_the_row_velocity_then_its_rhs(self):
        self.assertEqual(_Rows().sibling_residual("s", "sib", "", "rhs[sib]"), "float s = J_v(sib);\ns += rhs[sib];")

    def test_pair_terms_compose_the_sibling_residual_and_the_coupling(self):
        rows = _Rows(diagonal=lambda row: f"A[{row}]")
        self.assertEqual(
            rows.pair_terms("i", "sib", _rhs, ""),
            "float sibling_residual = J_v(sib);\nsibling_residual += rhs[sib];\nfloat cross = J_Y(i, sib);",
        )
        own = rows.pair_terms("i", "sib", _rhs, "", own=True)
        self.assertTrue(own.startswith("float residual = J_i_v;\nresidual += rhs[i];\n"))

    def test_maintained_residuals_hold_their_rhs(self):
        block = _Block(8, Lanes(32))
        terms = block.pair_terms("i", "sib", _rhs, "", own=True)
        self.assertNotIn("rhs", terms)
        self.assertIn("float sibling_residual = s_res[sib];", terms)

    def test_end_passes_apply_a_row_through_apply_row(self):
        self.assertEqual(_Rows().apply_strided("r", "d", ""), _Rows().apply_row("r", "d", ""))
        # Packed rows apply the entries their row velocity kept.
        packed = SparseFactor(16, lambda row, e: f"dof[{row}][{e}]", lambda row, e: f"Z[{row}][{e}]", Lanes(32))
        self.assertEqual(packed.apply_strided("r", "d", ""), packed.apply("d", ""))

    def test_an_operator_names_the_action_it_lacks(self):
        with self.assertRaisesRegex(NotImplementedError, "_Rows has no pair apply"):
            _Rows().apply_pair("a", "b", "")
        with self.assertRaisesRegex(NotImplementedError, "_Block has no row velocity"):
            _Block(8, Lanes(32)).row_velocity("v", "r", "")


class TestRowVisit(unittest.TestCase):
    """The admission of a row of a row loop before the row law."""

    def test_rows_without_friction_pairs_have_no_gate_admission_or_disk(self):
        ctx = _context(_Rows(diagonal=lambda row: f"A[{row}]"), friction=False)
        visit = row_visit(ctx, RowLaws(regularize=False), parent="p[i]", check="denom", pad="")
        self.assertNotIn("friction_gate", visit)
        self.assertNotIn("parent", visit)
        self.assertNotIn("sib", visit)
        self.assertIn("if (denom <= 0.0f) continue;", visit)

    def test_admitted_rows_set_the_disk_at_admission(self):
        ctx = _context(_Rows(diagonal=lambda row: f"A[{row}]"))
        visit = row_visit(ctx, RowLaws(regularize=False), parent="p[i]", check=None, pad="")
        admission = visit[visit.index(f"if (row_type == {FRICTION}) {{") : visit.index("float jv")]
        self.assertIn("if (i != parent + 1) continue;\n    const float lambda_n = RING_LOAD(lam[parent]);", admission)
        self.assertEqual(admission.count("{"), admission.count("}"))
        self.assertNotIn("<= 0.0f && row_type", visit)

    def test_a_held_second_tangent_impulse_is_kept_by_the_disk(self):
        ctx = _context(_Rows(diagonal=lambda row: f"A[{row}]"))
        visit = row_visit(
            ctx,
            RowLaws(regularize=False),
            parent="p[i]",
            check="denom",
            declare="float denom = A[i];",
            hold="float pair_impulse = 0.0f;",
            keep_next="pair_impulse",
            pad="",
        )
        self.assertLess(visit.index("<= 0.0f && row_type"), visit.index("float pair_impulse = 0.0f;"))
        self.assertIn("pair_impulse = lam[i + 1];", visit)
        self.assertIn("if (radius == 0.0f && lam[i] == 0.0f && pair_impulse == 0.0f) continue;", visit)
        self.assertEqual(
            friction_radius("k", "L", "m", "a", "b", "", keep_next="h"),
            f"if (k == {FRICTION}) {{\n    const float lambda_n = L;\n    radius = fmaxf(m * lambda_n, 0.0f);\n"
            "    h = b;\n    if (radius == 0.0f && a == 0.0f && h == 0.0f) continue;\n}",
        )

    def test_the_gate_skips_the_row_unless_asked_not_to(self):
        gate = friction_gate("k", "lam[i] = 0.0f;", "SYNC();", "")
        self.assertTrue(gate.endswith("    SYNC();\n    continue;\n}"))
        self.assertNotIn("continue", friction_gate("k", "lam[i] = 0.0f;", "", "", skip=False))


class TestContactUnit(unittest.TestCase):
    """A contact unit's rows by the row law (:func:`.sweep.contact_unit`): where the lanes synchronize."""

    def _unit(self, lanes=None, check=None, held=("lam0", "lam1", "lam2"), **options):
        ctx = dataclasses.replace(
            _context(_UnitRows(diagonal=lambda row: f"A[{row}]"), lanes, **options),
            ring_load=lambda parent, load=None: f"RING_LOAD({load or f'lam[{parent}]'})",
        )
        declare = "const int i = head + r;\nconst int row_type = r == 0 ? 0 : 2;\nconst float denom = A[i];"
        laws = RowLaws(regularize=False)
        return contact_unit(ctx, laws, rows="3", declare=declare, parent="head", check=check, held=held, pad="")

    def test_held_rows_synchronize_only_before_the_normal_stores(self):
        """The rows read the unit's impulses from the registers the placement loaded and pool the ring from the
        normal's new impulse in a register, so only the normal's stores wait for every lane's reads (besides the
        delayed-friction gate's own barrier) and no row synchronizes after its stores: the placement does after the
        unit's apply."""
        unit = self._unit()
        self.assertEqual(re.findall(r"lam\[[^\]]*\](?! = )", unit), [])
        self.assertIn("float unit_normal = lam0;", unit)
        self.assertIn("RING_LOAD(unit_normal)", unit)
        self.assertRegex(
            unit,
            r"if \(r == 0\) SYNC\(\);\s+if \(lane == 0\) lam\[i\] = new_impulse;\s+if \(r == 0\) unit_normal = new_impulse;",
        )
        self.assertEqual(unit.count("SYNC();"), 2)
        self.assertNotIn("if (denom <= 0.0f", unit)

    def test_a_normal_the_visit_may_skip_leaves_every_rows_stores_waiting(self):
        """Where the placement did not admit the normal, a skipped normal leaves the pair's stores first."""
        unit = self._unit(check="denom")
        self.assertIn("if (denom <= 0.0f && row_type != 2) continue;", unit)
        self.assertNotIn("if (r == 0) SYNC();", unit)
        self.assertRegex(unit, r"SYNC\(\);\s+if \(lane == 0\) lam\[i\] = new_impulse;")
        self.assertEqual(unit.count("SYNC();"), 2)

    def test_rows_every_lane_stores_read_their_own_stores(self):
        """Rows that every lane stores alike, or one thread's, hold nothing and synchronize only at the gate."""
        for lanes, owned in ((Lanes(32), False), (Lanes(1), True)):
            with self.subTest(width=lanes.width):
                unit = self._unit(lanes, check="denom", held=None, owned=owned)
                self.assertNotIn("unit_normal", unit)
                self.assertIn("RING_LOAD(lam[parent])", unit)
                self.assertEqual(unit.count("SYNC();"), int(lanes.parallel))

    def test_owned_rows_hold_their_impulses_exactly_where_the_lanes_share_the_stores(self):
        with self.assertRaises(ValueError):
            self._unit(held=None)
        with self.assertRaises(ValueError):
            self._unit(Lanes(1))


class TestRowTail(unittest.TestCase):
    """The stores, votes and applies after a row's projection."""

    def _update(self, operator, laws=None, **options):
        ctx = _context(operator, **options)
        return row_update(ctx, laws or RowLaws(regularize=False), residual="jv", pad="")

    def test_the_sibling_applies_before_the_row(self):
        update = self._update(_Rows(diagonal=lambda row: f"A[{row}]"))
        self.assertLess(update.index("v += Y(sib) * sib_delta;"), update.index("v += Y_i * delta_impulse;"))
        self.assertEqual(update.count("SYNC();"), 3)

    def test_a_fused_pair_apply_votes_and_applies_both_changes_at_once(self):
        update = self._update(_FusedRows(diagonal=lambda row: f"A[{row}]"))
        self.assertIn("if (delta_impulse != 0.0f || sib_delta != 0.0f) {", update)
        self.assertIn("apply_pair(sib_delta, delta_impulse);", update)
        self.assertNotIn("Y(sib)", update)

    def test_a_fused_vote_reads_the_sibling_only_where_the_row_has_one(self):
        """A row without a pair (``sib < 0``) has no sibling whose diagonal the vote could read: its zero change is
        quiet without reading ``A[-1]`` (another row's or buffer's slot, possibly not a finite number)."""
        update = self._update(_FusedRows(diagonal=lambda row: f"A[{row}]"))
        self.assertIn("(sib >= 0 && !(fabsf(sib_delta) * (A[sib]) <= sib_bound))", update)

    def test_rows_without_friction_pairs_store_and_apply_the_row_only(self):
        update = self._update(_FusedRows(diagonal=lambda row: f"A[{row}]"), friction=False)
        self.assertNotIn("sib", update)
        self.assertIn("v += Y_i * delta_impulse;", update)

    def test_an_unfenced_row_synchronizes_before_its_stores_and_after_its_apply(self):
        update = self._update(_Rows(diagonal=lambda row: f"A[{row}]"), fenced=False)
        self.assertEqual(update.count("SYNC();"), 2)

    def test_the_exit_law_quiets_changes_at_the_rounding_floor(self):
        """A change is loud beyond its row's bound at the old impulse, in its storage's form: phi against
        2^-20 (|b| + S + A |lambda|) from a stored diagonal."""
        vote = change_vote(RowLaws(regularize=False), (Change("d", "tau", "0", "A"),), "  ")
        self.assertEqual(vote, "  if ((!(fabsf(d) <= tau))) iteration_changed = 1;")
        self.assertEqual(bound("b", "S", "l", step="s"), "9.5367431640625e-07f * ((fabsf(b) + (S)) * (s) + fabsf(l))")
        update = self._update(_Rows(diagonal=lambda row: f"A[{row}]"))
        self.assertIn(f"float row_bound = {bound('rhs[i]', 'fabsf(jv)', 'old_impulse', diagonal='A[i]')};", update)
        self.assertIn("(!(fabsf(delta_impulse) * (A[i]) <= row_bound))", update)
        traced = change_vote(RowLaws(policy=SolvePolicy(readout="trace")), (Change("d", "tau", "0", "A"),), "")
        self.assertTrue(traced.startswith("if (d != 0.0f) solve_trace_change(fabsf(d) * (A), 0);"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

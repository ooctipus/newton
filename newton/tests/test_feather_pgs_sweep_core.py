# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source checks of the sweep emitter's core: the response operators' shared interface, the row visit and the row tail.

No GPU is needed: the checks read the native statements the core emits for small configurations.
"""

import dataclasses
import re
import unittest

from newton._src.solvers.feather_pgs.response_operator import Lanes, ResponseOperator, SparseFactor
from newton._src.solvers.feather_pgs.solve_policy import Change, SolvePolicy, bound
from newton._src.solvers.feather_pgs.sweep import (
    FRICTION,
    UNIT_HELD,
    Diagonal,
    RowContext,
    RowLaws,
    UnitZ,
    change_vote,
    contact_unit,
    friction_radius,
    row_sums,
    row_update,
    row_visit,
    unit_held_loads,
)


@dataclasses.dataclass(frozen=True)
class _Rows(ResponseOperator):
    """An operator of named sums: the two actions the row law takes, ``dot`` and ``axpy``."""

    def dot(self, sums, rows, indent, *, rhs=None, **_):
        rhs = rhs or {}
        lines = [
            f"{indent}float {out} = "
            + (f"J_v({rows[row]})" if other is None else f"J_Y({rows[row]}, {rows[other]})")
            + ";"
            for out, row, other in sums
        ]
        return "\n".join(lines + [f"{indent}{out} += {rhs[out]};" for out, _, _ in sums if out in rhs])

    def axpy(self, changes, indent, *, row=None, **_):
        ((name, delta),) = changes
        return f"\n{indent}v += Y({row or name}) * {delta};"


@dataclasses.dataclass(frozen=True)
class _FusedRows(_Rows):
    """The same operator applying a pair in one pass."""

    fused_pair_apply = True

    def axpy(self, changes, indent, *, row=None, **_):
        if len(changes) == 1:
            return super().axpy(changes, indent, row=row)
        return f"\n{indent}apply_pair({', '.join(delta for _, delta in changes)});"


def _rhs(row):
    return f"rhs[{row}]"


def _context(operator, lanes=None, **options):
    lanes = lanes or Lanes(32)
    return RowContext(
        row="i",
        kind="row_type",
        operator=operator,
        diagonal=Diagonal(lambda row: f"A[{row}]"),
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


class TestRowSums(unittest.TestCase):
    """The row law's sums by the operator's ``dot``: the visit's residual, its pair's terms and a contact unit's."""

    _PACKED = SparseFactor(16, lambda row, e: f"dof[{row}][{e}]", lambda row, e: f"Z[{row}][{e}]", Lanes(32))

    def test_a_visit_in_memory_reduces_its_pair_in_its_own_ladder(self):
        self.assertTrue(self._PACKED.pair_with_visit)
        sums = row_sums(self._PACKED, "jv", "", sibling="parent + 2", paired="k == 2")
        self.assertIn("SUM(jv, sibling_jv, pair_coupling);", sums)
        self.assertIn("const float sib_value = k == 2 && entry < 16 ? Z[(parent + 2)][entry] : 0.0f;", sums)
        self.assertEqual(row_sums(self._PACKED, "jv", ""), self._PACKED.dot((("jv", "row", None),), {"row": "i"}, ""))

    def test_sums_an_earlier_ladder_reduced_are_declared_from_it(self):
        sums = (("residual", "row", None), ("sibling_residual", "sib", None), ("cross", "row", "sib"))
        terms = self._PACKED.dot(
            sums,
            {"row": "i", "sib": "sib"},
            "",
            rhs={"residual": "rhs[i]", "sibling_residual": "rhs[sib]"},
            reduced={"sibling_residual": "sibling_jv", "cross": "pair_coupling"},
        )
        self.assertNotIn("sib_value", terms)
        self.assertTrue(
            terms.endswith(
                "SUM(residual);\nresidual += rhs[i];\nfloat sibling_residual = sibling_jv;\n"
                "sibling_residual += rhs[sib];\nfloat cross = pair_coupling;"
            )
        )

    def test_a_unit_dots_moved_terms_and_its_gram_entries(self):
        sums = (("spin_residual", "spin", None), ("spin_cross", "row", "spin"), ("spin_cross_sibling", "sib", "spin"))
        self.assertEqual(
            UnitZ().dot(sums, {}, "", rhs={"spin_residual": "b"}),
            "float spin_residual = (unit_jv3 + gram_n3 * unit_dn) + b;\nconst float spin_cross = unit_cross13;\n"
            "const float spin_cross_sibling = unit_cross23;",
        )
        self.assertTrue(
            UnitZ(cached=True)
            .dot(sums, {}, "")
            .endswith("const float spin_cross = 0.0f, spin_cross_sibling = 0.0f;  // the cached metric replaces them")
        )
        self.assertEqual(
            UnitZ().axpy((("sib", "a"), ("row", "b")), ""),
            "\nunit_dt2 += a;\nif (r == 0) unit_dn += b; else unit_dt1 += b;",
        )
        with self.assertRaisesRegex(ValueError, "moved residual"):
            UnitZ().dot((("residual", "row", None),), {}, "", rhs={"residual": "b"})


class TestDiagonal(unittest.TestCase):
    """The row law reads a row's diagonal in the form its store keeps: a value, or a reciprocal it multiplies by."""

    def test_the_step_divides_by_a_value_and_multiplies_by_a_reciprocal(self):
        value = Diagonal(lambda row: f"d[{row}]")
        self.assertEqual(value.step("r", "i"), "-r / d[i]")
        self.assertEqual(value.scale("r", "i"), "r / d[i]")
        self.assertEqual(value.value("i"), "d[i]")
        reciprocal = Diagonal(inverse=lambda row: f"inv[{row}]")
        self.assertEqual(reciprocal.step("r", "i"), "-r * inv[i]")
        self.assertEqual(reciprocal.value("i"), "inv[i] > 0.0f ? 1.0f / inv[i] : 0.0f")
        # A stored value with a staged reciprocal: the step multiplies, the vote reads the value (velocity units).
        both = Diagonal(lambda row: f"d[{row}]", lambda row: f"inv[{row}]")
        self.assertEqual(both.step("r", "i"), "-r * inv[i]")
        self.assertTrue(both.change("x", "b", "k", "i").scaled)
        self.assertFalse(reciprocal.change("x", "b", "k", "i").scaled)


class TestRowVisit(unittest.TestCase):
    """The admission of a row of a row loop before the row law."""

    def test_rows_without_friction_pairs_have_no_gate_admission_or_disk(self):
        ctx = _context(_Rows(), friction=False)
        visit = row_visit(ctx, RowLaws(regularize=False), parent="p[i]", check="denom", pad="")
        self.assertNotIn("parent", visit)
        self.assertNotIn("sib", visit)
        self.assertIn("if (denom <= 0.0f) continue;", visit)

    def test_admitted_rows_set_the_disk_at_admission(self):
        ctx = _context(_Rows())
        visit = row_visit(ctx, RowLaws(regularize=False), parent="p[i]", check=None, pad="")
        admission = visit[visit.index(f"if (row_type == {FRICTION}) {{") : visit.index("float jv")]
        self.assertIn("if (i != parent + 1) continue;\n    const float lambda_n = RING_LOAD(lam[parent]);", admission)
        self.assertEqual(admission.count("{"), admission.count("}"))
        self.assertNotIn("<= 0.0f && row_type", visit)

    def test_a_held_second_tangent_impulse_is_kept_by_the_disk(self):
        ctx = _context(_Rows())
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


class TestContactUnit(unittest.TestCase):
    """A contact unit's rows by the row law (:func:`.sweep.contact_unit`): where the lanes synchronize."""

    def _context(self, lanes=None, **options):
        return dataclasses.replace(
            _context(UnitZ(), lanes, **options),
            ring_load=lambda parent, load=None: f"RING_LOAD({load or f'lam[{parent}]'})",
        )

    def _unit(self, lanes=None, check=None, held=True, **options):
        ctx = self._context(lanes, **options)
        declare = "const int i = head + r;\nconst int row_type = r == 0 ? 0 : 2;\nconst float denom = A[i];"
        laws = RowLaws(regularize=False)
        return contact_unit(ctx, laws, rows="3", declare=declare, parent="head", check=check, held=held, pad="")

    def test_held_rows_synchronize_only_before_the_normal_stores(self):
        """The rows read the unit's impulses from the registers the placement loaded and pool the ring from the
        normal's new impulse in a register, so only the normal's stores wait for every lane's reads and no row
        synchronizes after its stores: the placement does after the unit's apply."""
        unit = self._unit()
        self.assertEqual(re.findall(r"lam\[[^\]]*\](?! = )", unit), [])
        self.assertIn("float unit_normal = unit_lam0;", unit)
        self.assertIn("RING_LOAD(unit_normal)", unit)
        self.assertRegex(
            unit,
            r"if \(r == 0\) SYNC\(\);\s+if \(lane == 0\) lam\[i\] = new_impulse;\s+if \(r == 0\) unit_normal = new_impulse;",
        )
        self.assertEqual(unit.count("SYNC();"), 1)
        self.assertNotIn("if (denom <= 0.0f", unit)

    def test_held_rows_read_no_staged_value_after_a_barrier(self):
        """A barrier orders the reads after it, so the held rows read their right-hand sides and diagonals, like their
        impulses, from the registers the placement loaded with the unit (:func:`.sweep.unit_held_loads`), the pair's
        too: no staged read waits for the normal's stores."""
        unit = self._unit()
        self.assertEqual(re.findall(r"\b(?:rhs|A)\[[^\]]*\]", unit), ["A[i]"])  # the declared diagonal, unread
        self.assertIn("float residual = jv + (r == 0 ? unit_rhs0 : unit_rhs1);", unit)
        self.assertIn("(unit_jv2 + gram_n2 * unit_dn) + unit_rhs2;", unit)
        self.assertIn("fabsf(delta_impulse) * ((r == 0 ? unit_diag0 : unit_diag1))", unit)
        ctx = self._context()
        self.assertEqual(
            [unit_held_loads(ctx, name, "h", "") for name in UNIT_HELD],
            [
                "const float unit_lam0 = lam[h], unit_lam1 = lam[h + 1], unit_lam2 = lam[h + 2];",
                "const float unit_rhs0 = rhs[h], unit_rhs1 = rhs[h + 1], unit_rhs2 = rhs[h + 2];",
                "const float unit_diag0 = A[h], unit_diag1 = A[h + 1], unit_diag2 = A[h + 2];",
            ],
        )

    def test_a_normal_the_visit_may_skip_leaves_every_rows_stores_waiting(self):
        """Where the placement did not admit the normal, a skipped normal leaves the pair's stores first."""
        unit = self._unit(check="denom")
        self.assertIn("if (denom <= 0.0f && row_type != 2) continue;", unit)
        self.assertNotIn("if (r == 0) SYNC();", unit)
        self.assertRegex(unit, r"SYNC\(\);\s+if \(lane == 0\) lam\[i\] = new_impulse;")
        self.assertEqual(unit.count("SYNC();"), 1)

    def test_rows_every_lane_stores_read_their_own_stores(self):
        """Rows that every lane stores alike, or one thread's, hold nothing and do not synchronize."""
        for lanes, owned in ((Lanes(32), False), (Lanes(1), True)):
            with self.subTest(width=lanes.width):
                unit = self._unit(lanes, check="denom", held=False, owned=owned)
                self.assertNotIn("unit_normal", unit)
                self.assertIn("RING_LOAD(lam[parent])", unit)
                self.assertEqual(unit.count("SYNC();"), 0)

    def test_owned_rows_hold_their_impulses_exactly_where_the_lanes_share_the_stores(self):
        with self.assertRaises(ValueError):
            self._unit(held=False)
        with self.assertRaises(ValueError):
            self._unit(Lanes(1))


class TestRowTail(unittest.TestCase):
    """The stores, votes and applies after a row's projection."""

    def _update(self, operator, laws=None, **options):
        ctx = _context(operator, **options)
        return row_update(ctx, laws or RowLaws(regularize=False), residual="jv", pad="")

    def test_the_sibling_applies_before_the_row(self):
        update = self._update(_Rows())
        self.assertLess(update.index("v += Y(sib) * sib_delta;"), update.index("v += Y(row) * delta_impulse;"))
        self.assertEqual(update.count("SYNC();"), 3)

    def test_a_fused_pair_apply_votes_and_applies_both_changes_at_once(self):
        update = self._update(_FusedRows())
        self.assertIn("if (delta_impulse != 0.0f || sib_delta != 0.0f) {", update)
        self.assertIn("apply_pair(sib_delta, delta_impulse);", update)
        self.assertNotIn("Y(sib)", update)

    def test_a_fused_vote_reads_the_sibling_only_where_the_row_has_one(self):
        """A row without a pair (``sib < 0``) has no sibling whose diagonal the vote could read: its zero change is
        quiet without reading ``A[-1]`` (another row's or buffer's slot, possibly not a finite number)."""
        update = self._update(_FusedRows())
        self.assertIn("(sib >= 0 && !(fabsf(sib_delta) * (A[sib]) <= sib_bound))", update)

    def test_rows_without_friction_pairs_store_and_apply_the_row_only(self):
        update = self._update(_FusedRows(), friction=False)
        self.assertNotIn("sib", update)
        self.assertIn("v += Y(row) * delta_impulse;", update)

    def test_an_unfenced_row_synchronizes_before_its_stores_and_after_its_apply(self):
        update = self._update(_Rows(), fenced=False)
        self.assertEqual(update.count("SYNC();"), 2)

    def test_the_exit_law_quiets_changes_at_the_rounding_floor(self):
        """A change is loud beyond its row's bound at the old impulse, in its storage's form: phi against
        2^-20 (|b| + S + A |lambda|) from a stored diagonal."""
        vote = change_vote(RowLaws(regularize=False), (Change("d", "tau", "0", "A"),), "  ")
        self.assertEqual(vote, "  if ((!(fabsf(d) <= tau))) iteration_changed = 1;")
        self.assertEqual(bound("b", "S", "l", step="s"), "9.5367431640625e-07f * ((fabsf(b) + (S)) * (s) + fabsf(l))")
        update = self._update(_Rows())
        self.assertIn(f"float row_bound = {bound('rhs[i]', 'fabsf(jv)', 'old_impulse', diagonal='A[i]')};", update)
        self.assertIn("(!(fabsf(delta_impulse) * (A[i]) <= row_bound))", update)
        traced = change_vote(RowLaws(policy=SolvePolicy(readout="trace")), (Change("d", "tau", "0", "A"),), "")
        self.assertTrue(traced.startswith("if (d != 0.0f) solve_trace_change(fabsf(d) * (A), 0);"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

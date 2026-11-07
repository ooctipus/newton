# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source checks of the paired factor sweep as a configuration of the sweep emitter.

No GPU is needed: the kernel rejects the options it does not admit and, with the CUDA toolkit on ``PATH``, compiles
offline without local memory.
"""

import importlib.util
import itertools
import re
import shutil
import sys
import unittest
from pathlib import Path

from newton._src.solvers.feather_pgs import sweep_routes_local
from newton._src.solvers.feather_pgs.response_operator import Lanes
from newton._src.solvers.feather_pgs.solve_policy import DEFAULT_POLICY, SolvePolicy
from newton._src.solvers.feather_pgs.sweep import sweep_kernel


def _PAIRED(*args, **kwargs):
    return sweep_kernel.__wrapped__(sweep_routes_local.PairedSweep(*args, **kwargs))


PAIRED_SCHEDULES = ({}, {"contact_units": True})


def _paired_variants():
    for schedule, speed, policy in itertools.product(
        PAIRED_SCHEDULES, (False, True), (DEFAULT_POLICY, SolvePolicy(readout="sweeps"))
    ):
        kwargs = {**schedule, "speed_rows": speed, "solve_policy": policy}
        yield f"{list(schedule) or 'general'} speed={speed}{policy.suffix}", (64, 23, 17, 6, "120"), kwargs


def _block(body: str, head: str) -> str:
    """The braced block that opens with ``head`` in ``body``."""
    start = body.index(head)
    depth = 0
    for index in range(start, len(body)):
        depth += {"{": 1, "}": -1}.get(body[index], 0)
        if depth == 0 and body[index] == "}":
            return body[start : index + 1]
    raise ValueError(f"unbalanced block {head!r}")


class TestPairedFactorRoute(unittest.TestCase):
    """The paired factor sweep (``sweep_routes_local.PairedSweep``)."""

    def test_variants_build(self):
        """Every schedule builds, with and without the line's speed rows and with a counted readout."""
        for label, args, kwargs in _paired_variants():
            with self.subTest(label):
                self.assertTrue(_PAIRED(*args, **kwargs).key)

    def test_invalid_options_raise(self):
        invalid = (
            ((0, 23, 17, 6, "120"), {}),
            ((64, 24, 17, 6, "120"), {}),
            ((64, 36, 30, 6, "120"), {}),
            ((64, 23, 23, 0, "120"), {}),
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
            body, _ = sweep_routes_local.PairedSweep(64, 23, 17, 6, "120", **schedule).source()
            held = "const float unit_lam0 = s_impulse[head], unit_lam1 = s_impulse[head + 1], unit_lam2 = s_impulse[head + 2];"
            self.assertIn(held, body)
            units = _block(body, "if (unit_lam0 != 0.0f || unit_lam1 != 0.0f || unit_lam2 != 0.0f) {")
            self.assertIn(unit, units)
            self.assertIn("float jv = r == 0 ? unit_jv0 : unit_jv1 + gram_n1 * unit_dn;", units)
            self.assertIn("fmaxf(fabsf(jv), unit_jv1_terms - fabsf(unit_dn) * gram_n1_terms)", units)
            # One block apply: the unit's changes summed in its rows' order, then added once.
            apply = "factor_velocity = factor_velocity + ((row_z * unit_dn + tangent1_z * unit_dt1) + tangent2_z * unit_dt2);"
            self.assertIn(apply, units)
            # The rows read the impulses the unit held and pool the ring from the normal's new impulse in a register,
            # so they read no store of the unit: each row's stores wait for the lanes (a skipped normal leaves the
            # pair's first), no row synchronizes after its stores, and the unit does once after its apply.
            self.assertEqual(re.findall(r"s_impulse\[[^\]]*\](?! = )", units), [])
            self.assertIn("RING_LOAD(unit_normal, parent, m, ring_parent, ring_impulse)", units)
            # Nor do they read a staged right-hand side or diagonal after a barrier: the unit holds them too, loaded
            # before its ladder (the normal's check reads its declared diagonal before its barrier).
            for held in ("unit_rhs0 = s_rhs[head]", "unit_diag2 = s_diag[head + 2]"):
                self.assertLess(units.index(held), units.index(unit))
            rows = units[units.index("for (int r = 0; r < 2; ++r) {") : units.index(apply)]
            self.assertEqual(re.findall(r"s_(?:rhs|diag)\[[^\]]*\]", rows), ["s_diag[i]"])
            self.assertEqual(units.count("SYNC();"), 2)
            self.assertRegex(units, re.escape(apply) + r"\s+}\s+}\s+SYNC\(\);\s+}$")
            rest = body[body.index(units) + len(units) :]
            self.assertTrue(rest.startswith(" else {"))
            unloaded = _block(rest, "else {")
            self.assertIn("SUM(jv, jv_terms);", unloaded)
            self.assertIn("SUM(residual, residual_terms, sibling_residual, sibling_residual_terms, cross);", unloaded)
            self.assertNotIn("unit_jv", unloaded)
        body, _ = sweep_routes_local.PairedSweep(64, 23, 17, 6, "120").source()
        self.assertIn("SUM(sibling_residual, sibling_residual_terms, cross);", body)
        self.assertNotIn(unit, body)


class TestResponseOperators(unittest.TestCase):
    """The paired route's rows: dense packed rows (``SparseFactor``) held in registers, ``u`` in a register a lane."""

    def test_register_rows_fuse_a_pair_ladder(self):
        rows = sweep_routes_local._rows(23, Lanes(32))
        rhs = lambda r: f"s_rhs[{r}]"  # noqa: E731
        sums = (("residual", "row", None), ("sibling_residual", "sib", None), ("cross", "row", "sib"))
        fused = rows.dot(
            sums, {"row": "i", "sib": "sib"}, "", rhs={"residual": rhs("i"), "sibling_residual": rhs("sib")}
        )
        self.assertFalse(rows.pair_with_visit)
        self.assertIn("SUM(residual, residual_terms, sibling_residual, sibling_residual_terms, cross);", fused)
        self.assertIn("residual_terms += fabsf(row_z * velocity);", fused)
        self.assertIn("const float velocity = factor_velocity;", fused)
        self.assertNotIn("indices", fused)
        self.assertEqual(rows.terms("residual", "s_rhs[i]", whole=True), "residual_terms")
        self.assertIn("residual += s_rhs[i];", fused)
        self.assertIn("if (index >= 0) factor_velocity += row_z * d;", rows.axpy((("row", "d"),), ""))
        self.assertIn("if (index >= 0) factor_velocity += spin_z * d;", rows.axpy((("spin", "d"),), ""))
        pair = rows.axpy((("sib", "sd"), ("row", "d")), "")
        self.assertLess(pair.index("value += sibling_z * sd;"), pair.index("value += row_z * d;"))


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

    def test_paired_schedules_compile(self):
        for schedule in PAIRED_SCHEDULES:
            with self.subTest(schedule):
                args = (64, 23, 17, 6, "120")
                kwargs = {**schedule, "speed_rows": True}
                self._compile(_PAIRED(*args, **kwargs), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)

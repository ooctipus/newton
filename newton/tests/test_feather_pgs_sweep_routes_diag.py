# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse-diagonal, tiled-row and split free-body sweeps as configurations of the sweep emitter (CPU checks).

Each configuration raises for the options it does not admit, declares its staging and generates a balanced body; the
operators' lane layouts emit the arithmetic order of the sweeps. With ``ptxas`` on ``PATH`` the kernels are also
assembled offline.
"""

import itertools
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import warp as wp

from newton._src.solvers.feather_pgs import sweep_routes_diag as routes
from newton._src.solvers.feather_pgs.response_operator import FreeBodyZ, Lanes, SparsePairZ, Velocity
from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy

_POLICIES = (SolvePolicy(), SolvePolicy(readout="sweeps"), SolvePolicy(readout="trace"))
_SCHEDULES = (
    {},
    {"contact_triples": True},
    {"contact_triples": True, "speculative_contact_batches": True},
    {"contact_units": True},
    {"contact_units": True, "speculative_contact_batches": True},
)


def _balanced(test, text):
    test.assertEqual(text.count("{"), text.count("}"))
    test.assertEqual(text.count("("), text.count(")"))
    test.assertNotIn("$", text)


class TestSweepRoutesDiagOptions(unittest.TestCase):
    """The configurations refuse the options they do not admit."""

    def test_invalid_options_raise(self):
        cases = [
            (routes.sparse_diagonal_kernel, (32, 14, 6, "sm_120"), {"contact_triples": True, "contact_units": True}),
            (routes.sparse_diagonal_kernel, (32, 14, 6, "sm_120"), {"speculative_contact_batches": True}),
            (
                routes.sparse_diagonal_kernel,
                (32, 30, 15, "sm_120"),
                {"contact_triples": True, "speculative_contact_batches": True},
            ),
        ]
        for factory, args, kwargs in cases:
            with self.subTest(factory=factory.__name__, kwargs=kwargs), self.assertRaises(ValueError):
                factory(*args, **kwargs)


class TestSweepRoutesDiagSources(unittest.TestCase):
    """Every option combination generates a body; the stagings are the declared working sets."""

    def test_sparse_diagonal_sources(self):
        for schedule, policy, (rows, dofs, dense) in itertools.product(
            _SCHEDULES, _POLICIES, ((32, 14, 6), (64, 40, 10), (128, 70, 14))
        ):
            with self.subTest(schedule=schedule, policy=policy, rows=rows, dofs=dofs, dense=dense):
                text, staging = routes.sparse_diagonal_source(rows, dofs, dense, solve_policy=policy, **schedule)
                _balanced(self, text)
                self.assertEqual(staging.bytes, 4 * (dofs + 5 * rows) + 4 * 32)
                # The coupled rows: every lane stores what it computes; the warp synchronizes after the apply only.
                self.assertIn("s_lam_dense[i] = new_impulse;", text)
                self.assertNotIn("if (lane == 0) s_lam_dense", text)
                # The sweep runs on the velocity change from the predictor, and nothing enters factor coordinates.
                self.assertIn(f"for (int d = lane; d < {dofs}; d += 32) s_v[d] = 0.0f;", text)

    def test_sparse_diagonal_schedules(self):
        triples, _ = routes.sparse_diagonal_source(32, 14, 6, contact_triples=True, speculative_contact_batches=True)
        units, _ = routes.sparse_diagonal_source(32, 14, 6, contact_units=True, speculative_contact_batches=True)
        plain, _ = routes.sparse_diagonal_source(32, 14, 6)
        # A batch tests one contact per segment of eight lanes (P + 2 = 8 response entries).
        self.assertIn("SEGMENT_SUM(8, candidate_velocity);", triples)
        self.assertIn("sparse_contact_serial_normals.data[world * 11 + candidate_index]", triples)
        self.assertIn("sparse_contact_serial_normals.data[world * 32 + candidate_index]", units)
        # The phase bound is clamped to the world's rows; the plain schedule has no contact segment.
        self.assertIn("const int contact_start = min(dense_phase_bounds.data[world * 2 + 1], row_count);", triples)
        self.assertIn("const int contact_start = row_count;", plain)
        self.assertNotIn("independent_group_count", plain)
        # Independent contacts: one product rounded alone, as the warp's sum of it rounds.
        self.assertIn("float jv = __fmul_rn(z, s_v[scalar_coord]);", triples)
        # A chain visits a contact's normal and its pair's first tangent with their kinds (and the pair's parent) known,
        # so no visit reads row metadata; the second tangent is only zeroed while friction waits. A unit has a pair
        # only when it owns an anchor.
        for text in (triples, units):
            self.assertRegex(text, r"const int i = head;\s+const int row_type = 0;")
            self.assertRegex(text, r"const int i = head \+ 1;\s+const int row_type = 2;")
            self.assertIn("parent = head;", text)
            self.assertNotIn("pair_visits", text)
        self.assertIn("if (head + 2 < row_count", units)
        self.assertNotIn("if (head + 2 < row_count", triples)
        # A pair loads its sibling's entries once and reduces its two terms in one ladder; its apply reuses them.
        for text in (triples, units):
            self.assertEqual(text.count("SUM(sibling_residual, cross);"), 1)
            self.assertEqual(text.count("const float sib_z = sparse_row_z.data[sib_entry];"), 1)
            self.assertIn("s_v[scalar_coord] += sib_z * sib_delta;", text)
            self.assertIn("if (sib_coord >= 0) s_v[sib_coord] += sib_z * sib_delta;", text)
        # Every schedule solves a loaded coupled contact unit by the unit law in the warp's layout, before the row visits
        # an unloaded unit falls through to: the unit's three rows on a lane's coordinate, their velocity terms and the
        # pair's couplings in one ladder, the rows by the row law, one apply, the loop past the unit. A chain's contacts
        # keep their row visits (one lane per contact: a unit branch there would split the warp's lanes).
        loaded = "if (s_lam_dense[i] != 0.0f || s_lam_dense[i + 1] != 0.0f || s_lam_dense[i + 2] != 0.0f) {"
        for text in (plain, triples, units):
            self.assertEqual(text.count("SUM(unit_jv0, unit_jv1, unit_jv2, gram_n1, gram_n2, unit_cross);"), 1)
            self.assertEqual(text.count(loaded), 1)
            self.assertIn("float jv = r == 0 ? unit_jv0 : unit_jv1 + gram_n1 * unit_dn;", text)
            self.assertIn(
                "s_v[unit_coord] = unit_u + ((unit_z0 * unit_dn + unit_z1 * unit_dt1) + unit_z2 * unit_dt2);", text
            )
            self.assertLess(text.index(loaded), text.rindex("const int row_type = s_meta_dense[i]"))
        for text in (triples, units):
            start = text.index("const int independent_group_count")
            chains = text[start : text.index("SYNC();", start)]
            self.assertNotIn("unit_jv", chains)
            self.assertIn("float jv = __fmul_rn(z, s_v[scalar_coord]);", chains)
        # The plain schedule has no chain phase: a contact a chain's link marks keeps its row visits there too, so every
        # schedule solves the same rows by the same law (the CUDA twins compare the schedules bit for bit).
        marked = "!(sparse_row_dof.data[(row_base + i) * 2] < -1 || sparse_row_dof.data[(row_base + i) * 2 + 1] < -1)"
        self.assertLess(plain.index(marked), plain.index(loaded))
        # The rows act in factor coordinates on the velocity change: it leaves once before it adds to the predictor,
        # the dense articulation by the walk (in its rows' frame), a diagonal coordinate scaled by its m^-1/2.
        for text in (plain, triples, units):
            self.assertIn("rhs_bias.data[off_dense + (i)] + row_incident.data[off_dense + (i)]", text)
            self.assertIn("v_out.data[global_dof] += diagonal_scale > 0.0f ? s_v[d] * diagonal_scale : s_v[d];", text)
            self.assertIn("sqrtf(diagonal_inverse_mass.data[global_dof])", text)
            self.assertLess(text.index("for (int iteration = 0;"), text.index("dv = P^T L^-T R^T u"))
            self.assertLess(text.index("dv = P^T L^-T R^T u"), text.rindex("WALK_DOWN("))


class TestSweepRoutesDiagOperators(unittest.TestCase):
    """The operators' layouts emit the order of the sweeps they replace."""

    def test_free_body_one_thread(self):
        op = FreeBodyZ(
            Lanes(1), velocity=Velocity("s_vel"), inverse=lambda row: f"inv[{row}]", factor=lambda f: f"F[{f}]"
        )
        residual = op.residual("jv", "")
        # Each side's six products as one expression, added to the residual: a's, then b's.
        terms = " + ".join(
            f"mf_Z_a.data[mf6_base + i * 6 + {k}] * s_vel[dof_a{f' + {k}' if k else ''}]" for k in range(6)
        )
        self.assertIn(f"jv += {terms};", residual)
        self.assertLess(residual.index("dof_a >= 0"), residual.index("dof_b >= 0"))
        # The sibling residual starts from its right-hand side; a and b alternate per coordinate.
        sibling = op.sibling_residual("sibling_residual", "sib", "", "rhs[sib]")
        self.assertIn("float sibling_residual = rhs[sib];", sibling)
        a = sibling.index("if (dof_a >= 0) sibling_residual += mf_Z_a.data[mf6_base + sib * 6 + k] * s_vel[dof_a + k];")
        b = sibling.index("if (dof_b >= 0) sibling_residual += mf_Z_b.data[mf6_base + sib * 6 + k] * s_vel[dof_b + k];")
        self.assertLess(a, b)
        coupling = op.coupling("cross", "i", "sib", "")
        self.assertIn("float cross = 0.0f;", coupling)
        self.assertIn("cross += mf_Z_a.data[mf6_base + i * 6 + k] * mf_Z_a.data[mf6_base + sib * 6 + k];", coupling)
        # A speed row reads the body's physical velocity v = L^-T u through its factor, then pushes u by L^T dv.
        speed = op.speed("speed", "f", "axis", "")
        self.assertIn("speed_v[k] = s_vel[f + k];", speed)
        self.assertIn("(factors.L.data + (F[f]) * 36)", speed)
        self.assertIn("sqrtf((speed_v[axis] * speed_v[axis] + speed_v[axis + 1] * speed_v[axis + 1])", speed)
        self.assertIn("s_vel[f + k] += speed_dv[k];", op.scale_speed("s", "f", "axis", ""))
        self.assertEqual(op.step("r", "i"), "-r * inv[i]")

    def test_free_body_lanes_keep_their_text(self):
        op = FreeBodyZ(Lanes(32), factor=lambda f: f"F[{f}]")
        self.assertIn(
            "SUM(sibling_residual);\nsibling_residual += rhs;", op.sibling_residual("sibling_residual", "s", "", "rhs")
        )
        self.assertIn("if (lane == 0) {", op.scale_speed("s", "speed_dof", "axis", ""))

    def test_sparse_pair_layouts(self):
        warp = SparsePairZ(6, 32, Lanes(32), diagonal=lambda row: f"d[{row}]")
        self.assertIn(
            "float jv_partial = coord >= 0 ? z * s_v[coord] : 0.0f;\nSUM(jv_partial);", warp.residual("jv", "")
        )
        consumed = warp.consume("i", "")
        self.assertIn("lane <= 7", consumed)
        # One whitened entry per lane: the dense part's Z, or a slot's Z_s, read where the coordinate is.
        self.assertIn("z = dense_Z.data[(dense_group * 32 + i) * 6 + lane];", consumed)
        self.assertIn("const int sparse_entry = (row_base + i) * 2 + lane - 6;", consumed)
        self.assertIn("z = sparse_row_z.data[sparse_entry];", consumed)
        # W = Z^T Z: the coupling and the apply read the same entries the residual does.
        self.assertIn("float cross = coord >= 0 ? z * cross_here : 0.0f;", warp.coupling("cross", "i", "s", ""))
        self.assertIn("if (coord >= 0) s_v[coord] += z * delta;", warp.apply("delta", ""))
        segment = SparsePairZ(6, 32, Lanes(8, "batch_lane", segment=True))
        self.assertIn("SEGMENT_SUM(8, v);", segment.row_velocity("v", "c", "", active="valid"))
        self.assertIn("if (valid && batch_lane < 6)", segment.row_velocity("v", "c", "", active="valid"))
        scalar = SparsePairZ(6, 32, Lanes(1))
        self.assertIn("__fmul_rn(z, s_v[scalar_coord])", scalar.residual("jv", ""))
        self.assertEqual(warp.step("r", "i"), "-r / d[i]")


def _assemble(kernel, arch: int = 120) -> dict:
    """Assemble a kernel's module offline (Warp's PTX, then ``ptxas``) and return its entry's resources."""
    with tempfile.TemporaryDirectory() as directory:
        options = kernel.module.resolve_options(wp.config, block_dim=32)
        kernel.module._compile(None, directory, output_arch=arch, use_ptx=True, options=options)
        ptx = sorted(Path(directory).rglob("*.ptx"))[0]
        source = Path(directory) / "module.ptx"
        source.write_bytes(ptx.read_bytes().rstrip(b"\0"))
        result = subprocess.run(
            ["ptxas", f"-arch=sm_{arch}", "-v", str(source), "-o", str(Path(directory) / "module.cubin")],
            check=True,
            capture_output=True,
            text=True,
        )
    log = result.stderr
    entry = log[log.index("_cuda_kernel_forward") :]
    return {
        "registers": int(re.search(r"Used (\d+) registers", entry).group(1)),
        "shared": int(re.search(r"(\d+) bytes smem", entry).group(1)),
        "spills": int(re.search(r"(\d+) bytes spill stores", entry).group(1)),
    }


@unittest.skipUnless(shutil.which("ptxas"), "ptxas is not on PATH")
class TestSweepRoutesDiagAssembly(unittest.TestCase):
    """The kernels assemble offline without spills."""

    def test_kernels_assemble(self):
        cases = [
            (
                routes.sparse_diagonal_kernel,
                (32, 14, 6, "sm_120"),
                {"contact_triples": True, "speculative_contact_batches": True},
            ),
        ]
        for factory, args, kwargs in cases:
            with self.subTest(factory=factory.__name__, kwargs=kwargs):
                self.assertEqual(_assemble(factory(*args, **kwargs))["spills"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

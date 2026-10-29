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
from newton._src.solvers.feather_pgs.response_operator import Assembled, FreeBodyZ, Lanes, SparsePairJY, Velocity
from newton._src.solvers.feather_pgs.solve_policy import SolvePolicy
from newton._src.solvers.feather_pgs.staging import arch_budget

_POLICIES = (SolvePolicy(), SolvePolicy(readout="sweeps"), SolvePolicy(velocity_change_tolerance=1.0e-4))
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
            (routes.tiled_row_kernel, (32, "sm_120"), {"staging": "dense"}),
            (routes.tiled_row_kernel, (32, "sm_120"), {"row_budget": 16}),
            (routes.tiled_row_kernel, (64, "sm_120"), {"row_budget": 24}),
            (routes.tiled_row_kernel, (64, "sm_120"), {"row_budget": 32, "min_rows": 32}),
            (routes.tiled_row_kernel, (64, "sm_120"), {"row_budget": 32, "staging": "global"}),
            (routes.tiled_row_kernel, (64, "sm_120"), {"min_rows": 8}),
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
        self.assertIn("float jv = __fmul_rn(jacobian, s_v[scalar_coord]);", triples)
        # A chain visits a contact's normal and its pair's first tangent with their kinds (and the pair's parent) known,
        # so no visit reads row metadata; the second tangent is only zeroed while friction waits. A unit has a pair
        # only when it owns an anchor.
        for text in (triples, units):
            self.assertRegex(text, r"const int i = head;\s+const int row_type = 0;")
            self.assertRegex(text, r"const int i = head \+ 1;\s+const int row_type = 2;")
            self.assertIn("parent = head;", text)
            self.assertRegex(
                text,
                r"friction_gate\(2, global_iter, friction_start_iteration\)\) \{\s+s_lam_dense\[head \+ 2\] = 0\.0f;",
            )
            self.assertNotIn("pair_visits", text)
        self.assertIn("if (head + 2 < row_count", units)
        self.assertNotIn("if (head + 2 < row_count", triples)
        # A pair loads its sibling's entries once and reduces its two terms in one ladder; its apply reuses them.
        for text in (triples, units):
            self.assertEqual(text.count("SUM(sibling_residual, cross);"), 1)
            self.assertEqual(text.count("const float sib_response = sparse_row_jy.data[sib_entry + 1];"), 1)
            self.assertIn("s_v[scalar_coord] += sib_response * sib_delta;", text)
            self.assertIn("if (sib_coord >= 0) s_v[sib_coord] += sib_response * sib_delta;", text)

    def test_tiled_row_stagings(self):
        for rows, staging in itertools.product((32, 64, 128), ("shared", "global")):
            with self.subTest(rows=rows, staging=staging):
                text, declared = routes.tiled_row_source(rows, staging=staging)
                _balanced(self, text)
                # Six row vectors, the packed triangle when shared, and Warp's tile storage of a warp.
                triangle = rows * (rows + 1) // 2 if staging == "shared" else 0
                self.assertEqual(declared.bytes, 4 * (6 * rows + triangle + 32))
                self.assertEqual("s_Ctri" in declared, staging == "shared")
                # Impulse space: every lane keeps its own impulses, nothing is synchronized in the sweeps.
                sweeps = text[text.index("for (int iter = 0;") :]
                self.assertNotIn("SYNC()", sweeps)

    def test_tiled_row_refuses_a_triangle_past_the_static_budget(self):
        # The triangle's static shared memory against the device's budget (staging.arch_budget): 150 rows fill an
        # offline build's 48 KiB, 217 the RTX PRO 6000's 100,352 B.
        budget = arch_budget("sm_120")
        rows = max(r for r in range(64, 320) if routes.tiled_row_source(r)[1].bytes <= budget)
        routes.tiled_row_kernel(rows, "sm_120")
        with self.assertRaisesRegex(ValueError, "static shared memory"):
            routes.tiled_row_kernel(rows + 1, "sm_120")

    def test_tiled_row_budget_sources(self):
        for rows, budget, minimum in ((64, 16, 0), (64, 32, 0), (64, 64, 32), (128, 32, 0), (128, 128, 32)):
            with self.subTest(rows=rows, budget=budget, minimum=minimum):
                text, declared = routes.tiled_row_source(rows, row_budget=budget, min_rows=minimum)
                _balanced(self, text)
                self.assertIn(f"const int off2 = world * {rows * rows};", text)
                self.assertEqual(declared.bytes, 4 * (6 * budget + budget * (budget + 1) // 2 + 32))

    def test_mf_split_sources(self):
        for vlim in (False, True):
            with self.subTest(has_velocity_limit_rows=vlim):
                text, declared = routes.mf_split_source(512, 64, has_velocity_limit_rows=vlim)
                _balanced(self, text)
                # Velocities, impulses, the bodies' first coordinates and their factors' indices.
                self.assertEqual(declared.bytes, 4 * (64 * 6 + 512 + 64 + 64) + 4 * 32)
                self.assertEqual("ProjectionLaw::speed_row" in text, vlim)


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
        warp = SparsePairJY(6, 32, Lanes(32), diagonal=lambda row: f"d[{row}]")
        self.assertIn(
            "float jv_partial = coord >= 0 ? jacobian * s_v[coord] : 0.0f;\nSUM(jv_partial);", warp.residual("jv", "")
        )
        self.assertIn("lane <= 7", warp.consume("i", ""))
        segment = SparsePairJY(6, 32, Lanes(8, "batch_lane", segment=True))
        self.assertIn("SEGMENT_SUM(8, v);", segment.row_velocity("v", "c", "", active="valid"))
        self.assertIn("if (valid && batch_lane < 6)", segment.row_velocity("v", "c", "", active="valid"))
        scalar = SparsePairJY(6, 32, Lanes(1))
        self.assertIn("__fmul_rn(jacobian, s_v[scalar_coord])", scalar.residual("jv", ""))
        self.assertEqual(warp.step("r", "i"), "-r / d[i]")

    def test_assembled_stagings_read_their_entries(self):
        shared = Assembled(64, "64", "shared")
        stored = Assembled(64, "64", "global")
        self.assertIn("j0 <= i ? s_Ctri[base_i + j0] : s_Ctri[jb0 + i]", shared.residual("jv", ""))
        self.assertIn("world_C.data[row_i + j1]", stored.residual("jv", ""))
        # The coupling is C(s, i) below the diagonal, the entry both stagings hold.
        self.assertIn("s_Ctri[((sib * (sib + 1)) >> 1) + i]", shared.coupling("cross", "i", "sib", ""))
        self.assertIn("world_C.data[off2 + sib * 64 + i]", stored.coupling("cross", "i", "sib", ""))
        self.assertEqual(shared.apply("d", ""), "")
        self.assertFalse(shared.applies)
        with self.assertRaises(ValueError):
            Assembled(64, "64", "dense")


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
            (routes.tiled_row_kernel, (64, "sm_120"), {"staging": "global"}),
            (routes.mf_split_kernel, (512, 64, "sm_120"), {"has_velocity_limit_rows": True}),
        ]
        for factory, args, kwargs in cases:
            with self.subTest(factory=factory.__name__, kwargs=kwargs):
                self.assertEqual(_assemble(factory(*args, **kwargs))["spills"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

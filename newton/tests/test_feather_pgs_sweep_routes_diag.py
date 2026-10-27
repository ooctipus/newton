# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse-diagonal, tiled-row and split free-body sweeps as configurations of the sweep emitter (CPU checks).

Each configuration replaces a legacy factory of ``solver_feather_pgs``: it takes the same arguments, builds a kernel
of the same signature and raises for the same options. The operators' lane layouts emit the arithmetic order of the
sweeps they replace; with ``ptxas`` on ``PATH`` the legacy and emitter kernels are also assembled offline.
"""

import itertools
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs as legacy
from newton._src.solvers.feather_pgs import sweep_routes_diag as routes
from newton._src.solvers.feather_pgs.response_operator import Assembled, FreeBody6, Lanes, SparsePairJY, Velocity
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


def _signature(kernel):
    return [(argument.label, wp.types.type_repr(argument.type)) for argument in kernel.adj.args]


def _balanced(test, text):
    test.assertEqual(text.count("{"), text.count("}"))
    test.assertEqual(text.count("("), text.count(")"))
    test.assertNotIn("$", text)


class TestSweepRoutesDiagABI(unittest.TestCase):
    """The configurations' kernels take the legacy factories' arguments and keep their signatures."""

    def test_sparse_diagonal_operands_are_the_legacy_operands(self):
        self.assertEqual(routes.SPARSE_DIAGONAL_ABI, legacy._SPARSE_DIAGONAL_SWEEP_PARAMS)

    def test_sparse_diagonal_signatures(self):
        for schedule, policy in itertools.product(_SCHEDULES, (*_POLICIES, SolvePolicy(readout="trace"))):
            with self.subTest(schedule=schedule, policy=policy):
                args = (32, 14, 6, "sm_120")
                old = legacy._get_pgs_solve_sparse_diagonal_kernel(*args, solve_policy=policy, **schedule)
                new = routes.sparse_diagonal_kernel(*args, solve_policy=policy, **schedule)
                self.assertEqual(_signature(new), _signature(old))

    def test_tiled_row_signatures(self):
        cases = [((32, "sm_120"), {}), ((64, "sm_120"), {"staging": "global"})]
        cases += [((128, "sm_120"), {"row_budget": 32}), ((128, "sm_120"), {"row_budget": 128, "min_rows": 32})]
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                old = legacy._get_pgs_solve_tiled_row_kernel(*args, **kwargs)
                new = routes.tiled_row_kernel(*args, **kwargs)
                self.assertEqual(_signature(new), _signature(old))

    def test_mf_split_signatures(self):
        for vlim in (False, True):
            with self.subTest(has_velocity_limit_rows=vlim):
                old = legacy._get_pgs_solve_mf_kernel(512, 64, "sm_120", has_velocity_limit_rows=vlim)
                new = routes.mf_split_kernel(512, 64, "sm_120", has_velocity_limit_rows=vlim)
                self.assertEqual(_signature(new), _signature(old))

    def test_options_raise_like_the_legacy_factories(self):
        cases = [
            (
                legacy._get_pgs_solve_sparse_diagonal_kernel,
                routes.sparse_diagonal_kernel,
                (32, 14, 6, "sm_120"),
                {"contact_triples": True, "contact_units": True},
            ),
            (
                legacy._get_pgs_solve_sparse_diagonal_kernel,
                routes.sparse_diagonal_kernel,
                (32, 14, 6, "sm_120"),
                {"speculative_contact_batches": True},
            ),
            (
                legacy._get_pgs_solve_sparse_diagonal_kernel,
                routes.sparse_diagonal_kernel,
                (32, 30, 15, "sm_120"),
                {"contact_triples": True, "speculative_contact_batches": True},
            ),
            (legacy._get_pgs_solve_tiled_row_kernel, routes.tiled_row_kernel, (32, "sm_120"), {"staging": "dense"}),
            (legacy._get_pgs_solve_tiled_row_kernel, routes.tiled_row_kernel, (32, "sm_120"), {"row_budget": 16}),
            (legacy._get_pgs_solve_tiled_row_kernel, routes.tiled_row_kernel, (64, "sm_120"), {"row_budget": 24}),
            (
                legacy._get_pgs_solve_tiled_row_kernel,
                routes.tiled_row_kernel,
                (64, "sm_120"),
                {"row_budget": 32, "min_rows": 32},
            ),
            (
                legacy._get_pgs_solve_tiled_row_kernel,
                routes.tiled_row_kernel,
                (64, "sm_120"),
                {"row_budget": 32, "staging": "global"},
            ),
            (legacy._get_pgs_solve_tiled_row_kernel, routes.tiled_row_kernel, (64, "sm_120"), {"min_rows": 8}),
        ]
        for old, new, args, kwargs in cases:
            with self.subTest(factory=new.__name__, kwargs=kwargs):
                with self.assertRaises(ValueError) as expected:
                    old(*args, **kwargs)
                with self.assertRaisesRegex(ValueError, re.escape(str(expected.exception))):
                    new(*args, **kwargs)


class TestSweepRoutesDiagSources(unittest.TestCase):
    """Every option combination generates a body; the stagings are the legacy working sets."""

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

    def test_tiled_row_stagings_are_the_legacy_bytes(self):
        for rows, staging in itertools.product((32, 64, 128), ("shared", "global")):
            with self.subTest(rows=rows, staging=staging):
                text, declared = routes.tiled_row_source(rows, staging=staging)
                _balanced(self, text)
                self.assertEqual(declared.bytes, legacy._tiled_row_shared_bytes(rows, staging))
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
                self.assertEqual(declared.bytes, 4 * (64 * 6 + 512 + 64) + 4 * 32)
                self.assertEqual("ProjectionLaw::speed_row" in text, vlim)


class TestSweepRoutesDiagOperators(unittest.TestCase):
    """The operators' layouts emit the order of the sweeps they replace."""

    def test_free_body_one_thread(self):
        op = FreeBody6(Lanes(1), velocity=Velocity("s_vel"), inverse=lambda row: f"inv[{row}]")
        residual = op.residual("jv", "")
        # Each side's six products as one expression, added to the residual: a's, then b's.
        terms = " + ".join(
            f"mf_J_a.data[mf6_base + i * 6 + {k}] * s_vel[dof_a{f' + {k}' if k else ''}]" for k in range(6)
        )
        self.assertIn(f"jv += {terms};", residual)
        self.assertLess(residual.index("dof_a >= 0"), residual.index("dof_b >= 0"))
        # The sibling residual starts from its right-hand side; a and b alternate per coordinate.
        sibling = op.sibling_residual("sibling_residual", "sib", "", "rhs[sib]")
        self.assertIn("float sibling_residual = rhs[sib];", sibling)
        a = sibling.index("if (dof_a >= 0) sibling_residual += mf_J_a.data[mf6_base + sib * 6 + k] * s_vel[dof_a + k];")
        b = sibling.index("if (dof_b >= 0) sibling_residual += mf_J_b.data[mf6_base + sib * 6 + k] * s_vel[dof_b + k];")
        self.assertLess(a, b)
        coupling = op.coupling("cross", "i", "sib", "")
        self.assertIn("float cross = 0.0f;", coupling)
        self.assertIn("cross += mf_J_a.data[mf6_base + i * 6 + k] * mf_MiJt_a.data[mf6_base + sib * 6 + k];", coupling)
        speed = op.speed("speed", "f", "")
        self.assertIn("sqrtf(s_vel[f] * s_vel[f] + s_vel[f + 1] * s_vel[f + 1] + s_vel[f + 2] * s_vel[f + 2])", speed)
        self.assertEqual(op.step("r", "i"), "-r * inv[i]")

    def test_free_body_lanes_keep_their_text(self):
        op = FreeBody6(Lanes(32))
        self.assertIn(
            "SUM(sibling_residual);\nsibling_residual += rhs;", op.sibling_residual("sibling_residual", "s", "", "rhs")
        )
        self.assertIn("READ(speed_sq, 0)", op.speed("speed", "speed_dof", ""))

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
    """Legacy and emitter kernels assemble offline with the same shared memory and no spills."""

    def test_kernels_assemble(self):
        cases = [
            (
                legacy._get_pgs_solve_sparse_diagonal_kernel,
                routes.sparse_diagonal_kernel,
                (32, 14, 6, "sm_120"),
                {"contact_triples": True, "speculative_contact_batches": True},
            ),
            (legacy._get_pgs_solve_tiled_row_kernel, routes.tiled_row_kernel, (64, "sm_120"), {"staging": "global"}),
            (
                legacy._get_pgs_solve_mf_kernel,
                routes.mf_split_kernel,
                (512, 64, "sm_120"),
                {"has_velocity_limit_rows": True},
            ),
        ]
        for old, new, args, kwargs in cases:
            with self.subTest(factory=new.__name__, kwargs=kwargs):
                reference, assembled = _assemble(old(*args, **kwargs)), _assemble(new(*args, **kwargs))
                self.assertEqual(assembled["shared"], reference["shared"])
                self.assertEqual(assembled["spills"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

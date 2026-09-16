# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check fixed-cohort staging without changing the ordered spectral law."""

import ast
import inspect
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_spectral_tangents, sparse_staged_rows
from tools.fpgs_bench import test_sparse_spectral_tangents as spectral
from tools.fpgs_bench.test_sparse_factor import fixture

ENV = {**spectral.ENV, "FEATHER_PGS_SPARSE_STAGED_ROWS": "1"}


def _check_native():
    """Reuse the independent policy controls, changing only their owner key."""
    source = inspect.getsource(spectral.check_native)
    old = '"sparse_spectral_tangent43_s18_c100"'
    if source.count(old) != 1:
        raise RuntimeError("Original spectral test owner seam changed")
    namespace = dict(spectral.check_native.__globals__)
    exec(compile(source.replace(old, '"sparse_staged_spectral_tangent43_s18_c100"'), __file__, "exec"), namespace)
    return namespace["check_native"]


def _native_source(kernel):
    return inspect.getclosurevars(kernel.func).nonlocals["native"].native_snippet


class TestSparseStagedRows(unittest.TestCase):
    def test_factory(self):
        """Require the distinct default-off staged solver factory."""
        kernel = sparse_staged_rows.get_solve_kernel()
        self.assertEqual(kernel.key, "sparse_staged_spectral_tangent43_s18_c100")
        self.assertIs(kernel, sparse_staged_rows.get_solve_kernel())
        self.assertEqual(
            [arg.label for arg in kernel.adj.args],
            [arg.label for arg in sparse_spectral_tangents.get_solve_kernel().adj.args],
        )

    def test_owner_and_admission(self):
        """Keep original ownership off by default and reject incompatible flags."""
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_STAGED_ROWS": "0"}):
            ordinary = fixture("cpu")["owner"]
        self.assertFalse(ordinary.staged_rows)
        self.assertIs(ordinary.kernels.solve, sparse_spectral_tangents.get_solve_kernel())
        with patch.dict(os.environ, ENV):
            staged = fixture("cpu")["owner"]
        self.assertTrue(staged.staged_rows)
        self.assertIs(staged.kernels.solve, sparse_staged_rows.get_solve_kernel())
        for flags in (
            {"FEATHER_PGS_SPARSE_STAGED_ROWS": "bad"},
            {"FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS": "0"},
            {"FEATHER_PGS_SPARSE_METRIC_TANGENTS": "0"},
        ):
            with self.subTest(flags=flags), patch.dict(os.environ, {**ENV, **flags}), self.assertRaises(ValueError):
                fixture("cpu")

    def test_original_fallback_and_staged_reads(self):
        """Prove that the large cohort changes only shared storage declarations."""
        original = _native_source(sparse_spectral_tangents.get_solve_kernel())
        source = _native_source(sparse_staged_rows.get_solve_kernel())
        anchor = "    __shared__ float du[43], lam[100];"
        prefix, body = original.split(anchor)
        body = anchor + body[: -len("#endif\n")]
        expected = sparse_staged_rows._body_aliases(body, "full", fast=False)
        self.assertTrue(source.startswith(prefix))
        self.assertTrue(source.endswith("    } else {\n" + expected + "    }\n#endif\n"))
        fast = source.split("    for(int iteration=0;iteration<iterations;++iteration) {", 1)[1]
        fast = fast.split("    } else {\n" + expected, 1)[0]
        for retired in (
            "d.Z.data",
            "d.support.data",
            "p.support_nodes.data",
            "p.support_count.data",
            "row_type.data",
            "parent.data",
            "rhs.data",
            "diagonal.data",
            "mu.data",
            "d.incident.data",
        ):
            self.assertNotIn(retired, fast)
        self.assertIn("cfm.data[base+row+1]", fast)
        self.assertIn("+staged.incident[row]+staged.rhs[row]", fast)
        self.assertIn("du[node]+=z0*change0+z1*change1+z2*change2", fast)
        self.assertIn("du[static_cast<int>(staged.nodes[sibling][lane])]", fast)
        self.assertIn("sizeof(StagedSolveStorage)==3952", source)
        self.assertEqual(source.count("__shared__"), 1)
        self.assertEqual(source.count("if(count<=32)"), 1)
        self.assertNotIn("__syncthreads", source)
        self.assertIn("if(lane==0)contact_ready[0]=0u;", source)
        self.assertEqual(source.count("if(lane<4)contact_ready[lane]=0u;"), 1)
        with self.assertRaises(RuntimeError):
            sparse_staged_rows._staged_source(original.replace(anchor, "changed"))

    def test_actual_observer_seam(self):
        """Exercise the exact observer insertion against actual CPU owners."""
        path = Path(__file__).parent / "chain_capture_20260916" / "checked_sparse.py"
        assignments = [
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "spectral_replacement" for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        insertion = ast.literal_eval(assignments[0].value)
        source = 'def observe(sparse, solver):\n    solve="sparse_metric_tangent43_s18_c100"\n    result={}\n'
        namespace = {"os": os}
        exec(compile(source + insertion + "    return result\n", str(path), "exec"), namespace)
        observe = namespace["observe"]
        for flag in ("0", "1"):
            with self.subTest(flag=flag), patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_STAGED_ROWS": flag}):
                f = fixture("cpu")
                result = observe(f["owner"], f["solver"])
                self.assertEqual(result["staged_rows"]["observed"], flag == "1")
                self.assertEqual(result["solve_key"], f["owner"].kernels.solve.key)
                f["owner"].kernels.solve = (
                    sparse_staged_rows.get_solve_kernel()
                    if flag == "0"
                    else sparse_spectral_tangents.get_solve_kernel()
                )
                with self.assertRaises(RuntimeError):
                    observe(f["owner"], f["solver"])
                f["owner"].kernels.solve = (
                    sparse_spectral_tangents.get_solve_kernel()
                    if flag == "0"
                    else sparse_staged_rows.get_solve_kernel()
                )
                with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_STAGED_ROWS": "1" if flag == "0" else "0"}):
                    with self.assertRaises(RuntimeError):
                        observe(f["owner"], f["solver"])
                del os.environ["FEATHER_PGS_SPARSE_STAGED_ROWS"]
                with self.assertRaises(RuntimeError):
                    observe(f["owner"], f["solver"])


@unittest.skipUnless(wp.is_cuda_available(), "Root owns native execution")
class TestSparseStagedRowsCUDA(unittest.TestCase):
    def test_native_transactions(self):
        """Reuse all frozen spectral local-law and incoming-delta controls."""
        with patch.dict(os.environ, ENV), patch.object(spectral, "check_native", _check_native()):
            spectral.TestSparseSpectralTangentsCUDA.test_native_transactions(self)

    def test_native_current_held_graph_and_empty(self):
        """Keep current/held physical momentum and graph empty/regrow checks."""
        with patch.dict(os.environ, ENV), patch.object(spectral, "check_native", _check_native()):
            spectral.TestSparseSpectralTangentsCUDA.test_native_current_held_graph_and_empty(self)

    def test_native_saved_sixteen(self):
        """Reuse all 16 saved independent J/H physical cases without new tolerance."""
        with patch.dict(os.environ, ENV), patch.object(spectral, "check_native", _check_native()):
            spectral.TestSparseSpectralTangentsCUDA.test_native_saved_sixteen(self)

    def test_native_fixed_cohort_boundaries(self):
        """Compare 0/32/33/100 rows, triplet tails and exact fallback to the original."""
        with patch.dict(os.environ, ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        original = sparse_spectral_tangents.get_solve_kernel()
        staged = sparse_staged_rows.get_solve_kernel()
        rng = np.random.default_rng(328)
        lengths = owner.host["support_count"]
        primary = int(np.flatnonzero(lengths == 18)[0])
        secondary = int(np.argsort(lengths)[-2])
        self.assertNotEqual(primary, secondary)
        vhat = rng.normal(0, 0.1, 43).astype(np.float32)
        s.v_hat.assign(vhat)
        for count in (0, 32, 33, 100, 0, 32):
            for mode in ("spectral", "delayed", "scalar_fallback"):
                with self.subTest(count=count, mode=mode):
                    z = np.full((1, 100, 18), np.nan, np.float32)
                    z[0, :count] = rng.normal(0, 0.15, (count, 18))
                    support = np.full((1, 100), primary, np.int32)
                    types = np.full((1, 100), 3, np.int32)
                    parents = np.full((1, 100), -1, np.int32)
                    mu = np.zeros((1, 100), np.float32)
                    incoming = np.zeros((1, 100), np.float32)
                    incoming[0, :count] = 0.15
                    # End exactly at rows31/32 so both ownership boundaries
                    # exercise complete normal+tangent transactions.
                    for row in range(count - 3, -1, -3):
                        types[0, row : row + 3] = (0, 2, 2)
                        parents[0, row + 1 : row + 3] = row
                        mu[0, row + 1 : row + 3] = 0.6
                        incoming[0, row + 1 : row + 3] = (0.01, -0.01)
                        if mode == "scalar_fallback":
                            support[0, row + 2] = secondary
                    incident = rng.normal(0, 0.2, (1, 100)).astype(np.float32)
                    rhs = rng.normal(-0.1, 0.3, (1, 100)).astype(np.float32)
                    diagonal = np.ones((1, 100), np.float32)
                    for row in range(count):
                        length = lengths[support[0, row]]
                        diagonal[0, row] = np.sum(z[0, row, :length] ** 2) + 0.01
                        z[0, row, length:] = np.nan
                    owner.data.Z.assign(z)
                    owner.data.support.assign(support)
                    owner.data.incident.assign(incident)
                    s.constraint_count.assign(np.array([count], np.int32))
                    for name, values in (
                        ("row_type", types),
                        ("row_parent", parents),
                        ("row_mu", mu),
                        ("rhs", rhs),
                        ("diag", diagonal),
                    ):
                        getattr(s, name).assign(values)
                    s.row_cfm.fill_(0.01)
                    results = []
                    for kernel in (original, staged):
                        owner.kernels.solve = kernel
                        s.impulses.assign(incoming)
                        s.v_out.fill_(np.nan)
                        owner.solve(s.rhs, 8, 1.0, int(mode == "delayed"))
                        owner.check()
                        results.append((s.v_out.numpy().copy(), s.impulses.numpy().copy()))
                    for old, actual in zip(*results, strict=True):
                        np.testing.assert_allclose(actual, old, rtol=3e-5, atol=3e-6)
                    self.assertTrue(np.isfinite(results[1][0]).all())
                    if count == 0:
                        np.testing.assert_array_equal(results[1][0], vhat)
        owner.kernels.solve = staged
        s.constraint_count.assign(np.array([101], np.int32))
        s.v_out.fill_(123.0)
        owner.solve(s.rhs, 8, 1.0, 0)
        np.testing.assert_array_equal(s.v_out.numpy(), np.full(43, 123.0, np.float32))


if __name__ == "__main__":
    unittest.main()

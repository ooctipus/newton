# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Require actual experimental owner identity without claiming convergence."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.fpgs_bench.krylov_capture_20260916.checked_capture import snapshot
from tools.fpgs_bench.krylov_capture_20260916.run import load_variants


def solver(candidate):
    """Supply only the original observer's required configuration fields."""
    kernels = []
    for tier in (32, 48):
        kernels.append(
            SimpleNamespace(
                key=f"original_{tier}" + ("_kc24" if candidate else ""),
                _fpgs_krylov_chord=candidate,
                _fpgs_krylov_chord_native=f"native {tier}",
            )
        )
    return SimpleNamespace(
        _pgs_solve_mf_gs_incremental_kernels=kernels,
        _par_tiers=[(32, 0), (48, 32)],
        max_world_dofs=18,
        mf_gs_parallel_sweeps=24,
        mf_gs_parallel_matrix_free=True,
        mf_gs_parallel_nesterov=True,
        pgs_warmstart=False,
        _mf_warmstart_enabled=False,
        _ink_sizes=(18, 0, 0, 0),
    )


class TestKrylovCapture(unittest.TestCase):
    def test_one_available_gpu_preserves_other_cli_guards(self):
        adapter = SimpleNamespace(
            TOOLS=Path("/home/octi/Projects/newton-fpgs-structural-bench-20260913/tools/fpgs_bench")
        )
        variants = load_variants(adapter)
        common = [
            "--isaaclab",
            "/tmp/lab",
            "--baseline",
            "/tmp/base",
            "--candidate",
            "/tmp/candidate",
            "--output-dir",
            "/tmp/output",
            "--task",
            "anymald",
        ]
        for selected in (["1"], ["0", "1"]):
            args = variants.parse_args([*common, "--gpus", *selected])
            self.assertEqual(args.gpus, [int(value) for value in selected])
            self.assertTrue(args.check_overflow)
            self.assertEqual(args.seed, 0)
        for selected in (["1", "1"], ["-1"], ["0", "1", "2"]):
            with self.assertRaises(SystemExit):
                variants.parse_args([*common, "--gpus", *selected])

    def setUp(self):
        self.enterContext(
            patch.dict(
                "os.environ",
                {
                    "FEATHER_PGS_KRYLOV_CHORD": "1",
                    "FEATHER_PGS_SPECTRAL_RESIDUAL": "0",
                    "FEATHER_PGS_SPECTRAL_JACOBI": "0",
                    "FEATHER_PGS_SPECTRAL_GS": "0",
                },
            )
        )

    def test_actual_owner_and_baseline(self):
        result = snapshot(solver(True))
        self.assertTrue(result["check_pass"])
        self.assertEqual(len(result["native_source_sha256"]), 2)
        self.assertIn("not", result["scope"])
        with patch.dict("os.environ", {"FEATHER_PGS_KRYLOV_CHORD": "0"}):
            self.assertEqual(snapshot(solver(False))["native_source_sha256"], [])

    def test_wrong_owner_or_budget_rejected(self):
        with self.assertRaises(RuntimeError):
            snapshot(solver(False))
        for field, value in (("mf_gs_parallel_sweeps", 25), ("pgs_warmstart", True)):
            supplied = solver(True)
            setattr(supplied, field, value)
            with self.assertRaises(RuntimeError):
                snapshot(supplied)
        supplied = solver(True)
        supplied._pgs_solve_mf_gs_incremental_kernels[0].key = "original_32"
        with self.assertRaises(RuntimeError):
            snapshot(supplied)

    def test_conflicting_experiment_rejected(self):
        with patch.dict("os.environ", {"FEATHER_PGS_SPECTRAL_RESIDUAL": "1"}):
            with self.assertRaises(RuntimeError):
                snapshot(solver(True))


if __name__ == "__main__":
    unittest.main()

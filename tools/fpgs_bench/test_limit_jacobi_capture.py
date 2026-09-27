# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Require actual G1 limit-only dispatch without changing capture guards."""

import os
import runpy
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.fpgs_bench.test_sparse_factor import fixture
from tools.fpgs_bench.test_sparse_spectral_tangents import ENV

HERE = Path(__file__).with_name("chain_capture_20260916")
FLAG = "FEATHER_PGS_SPARSE_LIMIT_JACOBI"


class TestLimitJacobiCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the existing source-pinned observer outside native execution."""
        cls.observer = runpy.run_path(str(HERE / "checked_sparse.py"))

    def owner(self, enabled):
        """Construct the actual sparse owner from the existing physical CPU fixture."""
        with patch.dict(os.environ, {**ENV, FLAG: str(int(enabled))}):
            return fixture("cpu")["owner"]

    def test_explicit_flag_and_original_owner(self):
        """Accept the original owner and reject omitted or inconsistent flags."""
        owner = self.owner(False)
        snapshot = self.observer["limit_snapshot"]
        with patch.dict(os.environ, {FLAG: "0"}):
            self.assertFalse(snapshot(owner)["observed"])
        for value in (None, "yes", "1"):
            with patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                with self.assertRaises(RuntimeError):
                    snapshot(owner)

    def test_actual_candidate_factory_and_marker(self):
        """Require the real constructor-selected candidate kernel and strict Boolean."""
        owner = self.owner(True)
        snapshot = self.observer["limit_snapshot"]
        with patch.dict(os.environ, {FLAG: "1"}):
            result = snapshot(owner)
            self.assertTrue(result["observed"])
            self.assertEqual(result["kernel_key"], "sparse_spectral_limit_jacobi43_s18_c100")
            with patch.object(owner.kernels, "solve", object()), self.assertRaises(RuntimeError):
                snapshot(owner)
            with patch.object(owner, "limit_jacobi", 1), self.assertRaises(RuntimeError):
                snapshot(owner)
            with patch.object(owner, "spectral_tangents", False), self.assertRaises(RuntimeError):
                snapshot(owner)

    def test_original_factory_is_retained(self):
        """Reject a changed solve factory even when the new policy is disabled."""
        owner = self.owner(False)
        with patch.dict(os.environ, {FLAG: "0"}):
            with patch.object(owner.kernels, "solve", object()), self.assertRaises(RuntimeError):
                self.observer["limit_snapshot"](owner)

    def test_one_or_two_gpus_only(self):
        """Retain original argument validation apart from GPU cardinality."""
        runner = runpy.run_path(str(HERE / "run.py"))
        tools = Path("/home/octi/Projects/newton-fpgs-structural-bench-20260913/tools/fpgs_bench")
        with patch.object(sys, "path", [str(tools), *sys.path]):
            import compare_backends  # noqa: PLC0415

            module = SimpleNamespace(adapter=SimpleNamespace(TOOLS=tools), owner=compare_backends)
            variants = runner["selected_variants"](module)
        base = [
            "--isaaclab",
            "/tmp/lab",
            "--baseline",
            "/tmp/a",
            "--candidate",
            "/tmp/b",
            "--output-dir",
            "/tmp/out",
            "--task",
            "g1",
            "--gpus",
        ]
        for selected in (["0"], ["1"], ["0", "1"]):
            self.assertEqual(variants.parse_args([*base, *selected]).gpus, [int(i) for i in selected])
        for selected in (["0", "0"], ["-1"], ["0", "1", "2"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                variants.parse_args([*base, *selected])


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The untimed grouped-world observer must reject silent dispatch fallback."""

import os
import runpy
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from newton._src.solvers.feather_pgs import sparse_fullwarp_group


class TestFullwarpCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observer = runpy.run_path(str(Path(__file__).with_name("chain_capture_20260916") / "checked_sparse.py"))

    def test_exact_grouped_factories(self):
        owner = SimpleNamespace(
            fullwarp_group=True,
            kernels=SimpleNamespace(
                solve=sparse_fullwarp_group.get_solve_kernel(), prefix=sparse_fullwarp_group.get_prefix_kernel()
            ),
        )
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FULLWARP_GROUP": "1"}):
            result = self.observer["fullwarp_snapshot"](owner)
            self.assertTrue(result["check_pass"])
            self.assertTrue(result["observed"])
            owner.kernels.prefix = object()
            with self.assertRaises(RuntimeError):
                self.observer["fullwarp_snapshot"](owner)

    def test_marker_does_not_prove_dispatch(self):
        owner = SimpleNamespace(fullwarp_group=True, kernels=SimpleNamespace(solve=object(), prefix=object()))
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FULLWARP_GROUP": "1"}), self.assertRaises(RuntimeError):
            self.observer["fullwarp_snapshot"](owner)
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FULLWARP_GROUP": "yes"}), self.assertRaises(RuntimeError):
            self.observer["fullwarp_snapshot"](owner)

    def test_explicit_flag_and_baseline(self):
        owner = SimpleNamespace()
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FULLWARP_GROUP": "0"}):
            self.assertFalse(self.observer["fullwarp_snapshot"](owner)["observed"])
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_FULLWARP_GROUP": "1"}), self.assertRaises(RuntimeError):
            self.observer["fullwarp_snapshot"](owner)

    def test_one_or_two_gpus_only(self):
        runner = runpy.run_path(str(Path(__file__).with_name("chain_capture_20260916") / "run.py"))
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

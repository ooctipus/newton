# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check compact-workspace observation against the actual saved-input owner."""

import os
import runpy
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import test_franka_kinetic_state as retained

HERE = Path(__file__).with_name("compact_workspace_capture_20260917")
FLAG = "FEATHER_PGS_FRANKA_COMPACT_WORKSPACE"


class TestCompactWorkspaceCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the observer before constructing the actual native owner."""
        cls.observer = runpy.run_path(str(HERE / "checked_compact_workspace.py"))

    def bind(self, enabled):
        """Reuse real constructor assignments and saved shapes, not fake factory fields."""
        capture = next(retained.captures())
        with (
            np.load(capture["path"], allow_pickle=False) as snapshot,
            patch.dict(os.environ, {FLAG: str(int(enabled))}),
        ):
            case = retained.bind_saved(snapshot, capture, "cpu")
        # The saved-input binder does not run the outer solver constructor.
        case.solver._franka_kinetic_state = case.owner
        return case

    def test_actual_owners_and_explicit_modes(self):
        """Accept both real constructor arms and unchanged cache/factor shapes."""
        for enabled in (False, True):
            case = self.bind(enabled)
            with patch.dict(os.environ, {FLAG: str(int(enabled)), "FEATHER_PGS_FRANKA_KINETIC_STATE": "1"}):
                result = self.observer["snapshot"](case.solver)
            self.assertIs(result["observed"], enabled)
            self.assertEqual(result["status_nonzero"], 0)
            suffix = "_c376_p16" if enabled else "_p16"
            self.assertEqual(result["keys"]["repair"], "franka_kinetic_repair13_h81" + suffix)

    def test_missing_mode_wrong_factory_and_status(self):
        """Reject omitted/invalid modes, false owner selection and sticky failures."""
        case = self.bind(True)
        snapshot = self.observer["snapshot"]
        with patch.dict(os.environ, {"FEATHER_PGS_FRANKA_KINETIC_STATE": "1"}):
            for value in (None, "yes", "0"):
                with patch.dict(os.environ):
                    if value is None:
                        os.environ.pop(FLAG, None)
                    else:
                        os.environ[FLAG] = value
                    with self.assertRaises(RuntimeError):
                        snapshot(case.solver)
        with patch.dict(os.environ, {FLAG: "1", "FEATHER_PGS_FRANKA_KINETIC_STATE": "1"}):
            with patch.object(case.owner, "finish_kernel", case.owner.repair_kernel), self.assertRaises(RuntimeError):
                snapshot(case.solver)
            with patch.object(case.owner, "predictor_kernel", object()), self.assertRaises(RuntimeError):
                snapshot(case.solver)
            case.owner.status.fill_(1)
            with self.assertRaises(RuntimeError):
                snapshot(case.solver)
            case.owner.status.zero_()
            case.owner.data.current_valid.fill_(2)
            with self.assertRaises(RuntimeError):
                snapshot(case.solver)

    def test_one_or_two_gpu_cli(self):
        """Retain original argument guards except selected GPU cardinality."""
        runner = runpy.run_path(str(HERE / "run.py"))
        tools = Path("/home/octi/Projects/newton-fpgs-structural-bench-20260913/tools/fpgs_bench")
        with patch.object(sys, "path", [str(tools), *sys.path]):
            variants = runner["selected_variants"](tools)
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
            "franka",
            "--gpus",
        ]
        for selected in (["0"], ["1"], ["0", "1"]):
            self.assertEqual(variants.parse_args([*base, *selected]).gpus, list(map(int, selected)))
        for selected in (["0", "0"], ["-1"], ["0", "1", "2"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                variants.parse_args([*base, *selected])


if __name__ == "__main__":
    unittest.main()

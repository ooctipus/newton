# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate actual joint-world partitions at existing untimed boundaries."""

import unittest
from types import SimpleNamespace

import checked_capture as checked
from test_checked_capture import FlagArray


def fixture():
    """Provide current active and ZERO outputs with a poisoned unused list tail."""
    solver = SimpleNamespace(world_count=3)
    output = SimpleNamespace(
        resolved=FlagArray([1, 0, 1]),
        active_worlds=FlagArray([1, -999, -999]),
        active_count=FlagArray([1]),
        predictor_fallback=FlagArray([0, 1, 0]),
    )
    solver._joint_world = SimpleNamespace(
        solver=solver,
        output=output,
        buckets=SimpleNamespace(data=SimpleNamespace(invalid=FlagArray([0])), storage_bytes=16000100),
    )
    return solver, output


class TestJointWorldMetadata(unittest.TestCase):
    def test_actual_and_absent(self):
        """Report current output counts without claiming whole-run or physical acceptance."""
        solver, _ = fixture()
        result = checked.joint_world_metadata(solver)
        self.assertEqual((result["zero_worlds"], result["active_worlds"]), (2, 1))
        self.assertEqual(result["original_predictor_worlds"], 1)
        self.assertFalse(result["whole_run_admission"])
        self.assertFalse(checked.joint_world_metadata(object())["configured"])

    def test_all_and_empty(self):
        """Allow both valid extremes while ignoring unused list values."""
        solver, output = fixture()
        for active in (0, 3):
            output.resolved = FlagArray([int(i >= active) for i in range(3)])
            output.active_count = FlagArray([active])
            output.active_worlds = FlagArray(list(range(active)) + [-999] * (3 - active))
            output.predictor_fallback = FlagArray([0, 0, 0])
            self.assertEqual(checked.joint_world_metadata(solver)["active_worlds"], active)

    def test_invalid_output_rejected(self):
        """Reject malformed masks, prefixes, alias owners and invalid raw input."""
        for fault in ("count", "mask", "list", "capture", "raw", "fallback", "owner"):
            solver, output = fixture()
            if fault == "count":
                output.active_count.values[0] = 4
            elif fault == "mask":
                output.resolved.values[0] = 2
            elif fault == "list":
                output.active_worlds.values[0] = 0
            elif fault == "capture":
                output.resolved.device.is_capturing = True
            elif fault == "raw":
                solver._joint_world.buckets.data.invalid.values[0] = 1
            elif fault == "fallback":
                output.predictor_fallback.values[0] = 1
            else:
                solver._joint_world.solver = object()
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                checked.joint_world_metadata(solver)


if __name__ == "__main__":
    unittest.main()

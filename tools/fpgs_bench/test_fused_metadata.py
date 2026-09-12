# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test current fusion ownership readback without a simulator or GPU."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import checked_capture as checked
from test_checked_capture import FlagArray, collision_pipeline, fpgs


def fixture():
    """Expose current owners with mixed empty/admitted/fallback worlds."""
    solver = fpgs()
    solver.world_count = 4
    solver.constraint_count = FlagArray([0, 3, 384, 402])
    fusion = SimpleNamespace(owner=FlagArray([1, 1, 1, 0]), fallback_counts=FlagArray([0, 0, 0, 402]))
    solver._fused_contact_solve = fusion
    solver._fused_contact_solve_active = fusion
    return solver


class TestFusedMetadata(unittest.TestCase):
    def test_current_histogram_is_read_only_and_counts_rows(self):
        """Distinguish all admitted worlds from nonempty admitted work."""
        solver = fixture()
        arrays = (
            solver.constraint_count,
            solver._fused_contact_solve.owner,
            solver._fused_contact_solve.fallback_counts,
        )
        original = [value.values.copy() for value in arrays]
        result = checked.fused_contact_metadata(solver)
        self.assertEqual(result["fused_worlds"], 3)
        self.assertEqual(result["fused_nonempty_worlds"], 2)
        self.assertEqual(result["fallback_nonempty_worlds"], 1)
        self.assertEqual((result["fused_rows"], result["fallback_rows"]), (387, 402))
        self.assertEqual((result["fused_row_max"], result["fallback_row_max"]), (384, 402))
        self.assertEqual([value.values for value in arrays], original)
        json.dumps(result, allow_nan=False)

    def test_absent_or_inactive_owner_never_reads_stale_arrays(self):
        """Do not turn stale classifier contents into current-path evidence."""
        self.assertEqual(checked.fused_contact_metadata(fpgs()), {"configured": False, "active": False})
        solver = fixture()
        solver._fused_contact_solve_active = None
        solver._fused_contact_solve.owner.numpy = Mock(side_effect=AssertionError("stale read"))
        self.assertEqual(checked.fused_contact_metadata(solver), {"configured": True, "active": False})
        solver._fused_contact_solve.owner.numpy.assert_not_called()

    def test_invalid_current_ownership_is_rejected(self):
        """Reject malformed masks, counts, capture reads and mismatched owners."""
        for problem in ("mask", "shape", "negative", "overflow", "fallback", "capture", "identity"):
            with self.subTest(problem=problem):
                solver = fixture()
                fusion = solver._fused_contact_solve
                if problem == "mask":
                    fusion.owner.values[0] = 2
                elif problem == "shape":
                    fusion.owner.shape = (3,)
                elif problem == "negative":
                    solver.constraint_count.values[0] = -1
                elif problem == "overflow":
                    solver.constraint_count.values[3] = 705
                elif problem == "fallback":
                    fusion.fallback_counts.values[3] = 401
                elif problem == "capture":
                    fusion.owner.device.is_capturing = True
                else:
                    solver._fused_contact_solve_active = object()
                with self.assertRaises(RuntimeError):
                    checked.fused_contact_metadata(solver)

    def test_only_existing_metadata_boundaries_read_current_owner(self):
        """Persist evidence at the existing boundaries without adding any step hook."""
        solver = fixture()
        manager = SimpleNamespace(_solver=solver, _collision_pipeline=collision_pipeline())
        harness = SimpleNamespace(_model_meta=Mock(return_value={"state_finite": True}))
        original = harness._model_meta
        report = {"boundaries": [], "boundary_count": 0}
        checked.install_boundary_check(harness, report, lambda: None, lambda: manager)
        for _ in range(2):
            harness._model_meta("feather_pgs")
        self.assertEqual(original.call_count, 2)
        self.assertEqual([row["fused_contact_solve"]["fused_rows"] for row in report["boundaries"]], [387, 387])


if __name__ == "__main__":
    unittest.main()

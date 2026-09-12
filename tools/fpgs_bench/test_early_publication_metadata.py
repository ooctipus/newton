# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check untimed early-publication ownership without importing a GPU runtime."""

import unittest
from types import SimpleNamespace

import checked_capture as checked
from test_checked_capture import FlagArray


def fixture():
    """Provide a mixed partition with deliberately invalid unused list tails."""
    solver = SimpleNamespace(world_count=2, model=SimpleNamespace(articulation_count=6))
    solver.art_to_world = FlagArray([0, 0, 0, 1, 1, 1])
    data = SimpleNamespace(
        counts=FlagArray([2, 4]),
        art_mask=FlagArray([0, 1, 0, 0, 1, 0]),
        early_arts=FlagArray([4, 1, -71]),
        late_arts=FlagArray([5, 0, 3, 2, -71, -71]),
        invalid_raw=FlagArray([0]),
    )
    solver._early_franka = SimpleNamespace(solver=solver, data=data)
    return solver, data


class TestEarlyPublicationMetadata(unittest.TestCase):
    """Reject invalid ownership while allowing real empty/all partitions."""

    def test_mixed_and_absent(self):
        """Report actual selected work, not a host active flag or list capacity."""
        solver, _ = fixture()
        result = checked.early_publication_metadata(solver)
        self.assertEqual(result["franka"]["early_articulations"], 2)
        self.assertEqual(result["franka"]["late_articulations"], 4)
        self.assertEqual(result["franka"]["worlds_with_early_articulations"], 2)
        self.assertFalse(result["kuka"]["configured"])
        self.assertFalse(result["franka"]["whole_run_admission"])

    def test_empty_and_all(self):
        """Do not read unused tails or reject legitimate zero-cohort execution."""
        solver, data = fixture()
        for selected in (0, 6):
            data.counts = FlagArray([selected, 6 - selected])
            data.art_mask = FlagArray([int(i < selected) for i in range(6)])
            data.early_arts = FlagArray(list(range(selected)) + [-71] * (6 - selected))
            data.late_arts = FlagArray(list(range(selected, 6)) + [-71] * selected)
            result = checked.early_publication_metadata(solver)["franka"]
            self.assertEqual(result["early_articulations"], selected)

    def test_reject_invalid_partition_and_capture(self):
        """A malformed, aliased, unreadable or in-capture owner cannot certify work."""
        for failure in ("duplicate", "range", "count", "mask", "capture", "raw", "owner", "world"):
            solver, data = fixture()
            if failure == "duplicate":
                data.late_arts.values[0] = 4
            elif failure == "range":
                data.early_arts.values[0] = 6
            elif failure == "count":
                data.counts.values[0] = 4
            elif failure == "mask":
                data.art_mask.values[0] = 1
            elif failure == "capture":
                data.counts.device.is_capturing = True
            elif failure == "raw":
                data.invalid_raw.values[0] = 1
            elif failure == "owner":
                solver._early_franka.solver = object()
            else:
                solver.art_to_world.values[4] = -1
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                checked.early_publication_metadata(solver)


if __name__ == "__main__":
    unittest.main()

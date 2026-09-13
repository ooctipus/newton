# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check current structural ownership without a simulator or device import."""

import unittest
from types import SimpleNamespace

import checked_capture as checked
from test_checked_capture import FlagArray


def fixture():
    """Provide exactly the current live metadata fields used at a boundary."""
    zero, valid = FlagArray([0, 0]), FlagArray([1, 1])
    guard = SimpleNamespace(**dict.fromkeys(("predictor_status", "row_status", "solve_status", "error"), zero))
    for name in ("frame_status", "row_global_status", "raw_invalid"):
        setattr(guard, name, FlagArray([0]))
    state = SimpleNamespace(active_count=FlagArray([1]), active_worlds=FlagArray([1, 0]), resolved=FlagArray([1, 0]))
    slots = SimpleNamespace(
        schedule=SimpleNamespace(status=zero), next_schedule=SimpleNamespace(status=zero), refresh_status=zero
    )
    solver = SimpleNamespace(world_count=2, _joint_world=object())
    owner = SimpleNamespace(
        solver=solver,
        last_private=True,
        ever_admitted=True,
        clock=FlagArray([8, 9]),
        held=SimpleNamespace(valid=valid, generation=FlagArray([6, 7])),
        calls={1: slots},
        states={1: object(), 2: object()},
        last_call=SimpleNamespace(rows=SimpleNamespace(state=state), solve=SimpleNamespace(guard=guard)),
    )
    solver._kinetic_world = owner
    return solver


class TestStructuralMetadata(unittest.TestCase):
    """Reject bad current ownership without reading retired producer buffers."""

    def test_retired_joint_owner_not_read_when_kinetic_is_current(self):
        """Avoid treating stale joint-world outputs as the current partition."""
        result = checked.joint_world_metadata(fixture())
        self.assertEqual(result["delegated_to"], "kinetic_world")
        self.assertFalse(result["whole_run_admission"])

    def test_current_kinetic_partition_and_errors(self):
        """Validate actual current status while preserving failed flags."""
        solver = fixture()
        result = checked.kinetic_world_metadata(solver)
        self.assertEqual(result["active_worlds"], 1)
        self.assertEqual(result["zero_worlds"], 1)
        self.assertEqual(result["state_slots"], 2)
        solver._kinetic_world.last_call.solve.guard.frame_status = FlagArray([4])
        with self.assertRaisesRegex(RuntimeError, "status"):
            checked.kinetic_world_metadata(solver)
        self.assertEqual(solver._kinetic_world.last_call.solve.guard.frame_status.values, [4])

    def test_unconfigured_original_uses_no_candidate_arrays(self):
        """Keep the actual original backend free of candidate imports or work."""
        self.assertFalse(checked.kinetic_world_metadata(SimpleNamespace())["configured"])

    def test_capture_and_partition_rejected(self):
        """Reject in-capture readbacks and malformed active queues."""
        solver = fixture()
        solver._kinetic_world.clock.device.is_capturing = True
        with self.assertRaisesRegex(RuntimeError, "capture"):
            checked.kinetic_world_metadata(solver)
        solver = fixture()
        solver._kinetic_world.last_call.rows.state.active_worlds.values = [0, 1]
        with self.assertRaisesRegex(RuntimeError, "partition"):
            checked.kinetic_world_metadata(solver)


if __name__ == "__main__":
    unittest.main()

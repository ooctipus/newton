# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the CUDA driver query that tells whether a stream capture can pause for a conditional node."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import capture_joins
from newton._src.solvers.feather_pgs.capture_joins import capture_has_unjoined_work, capture_queries_available


class TestCaptureJoins(unittest.TestCase):
    def test_devices_without_streams_hold_no_unjoined_work(self):
        """Report no unjoined work on the CPU, which has no stream capture to pause."""
        self.assertFalse(capture_has_unjoined_work(wp.get_device("cpu")))

    def test_unanswered_queries_report_unjoined_work(self):
        """Report unjoined work, so callers keep the capture running, when the driver queries cannot load."""
        with mock.patch.object(capture_joins, "_driver", return_value=None):
            self.assertFalse(capture_queries_available())
            self.assertTrue(capture_has_unjoined_work(SimpleNamespace(is_cuda=True)))

    @unittest.skipUnless(wp.is_cuda_available(), "stream capture needs CUDA")
    def test_forked_work_stays_unjoined_until_the_stream_waits_for_it(self):
        """Find work a forked stream captured only until the capturing stream joins it."""
        device = wp.get_device("cuda:0")
        self.assertTrue(capture_queries_available())
        main_values = wp.zeros(64, dtype=float, device=device)
        side_values = wp.zeros(64, dtype=float, device=device)
        side = wp.Stream(device)
        states = []
        with wp.ScopedCapture(device=device) as capture:
            main = wp.get_stream(device)
            states.append(capture_has_unjoined_work(device))
            main_values.fill_(1.0)
            side.wait_event(main.record_event())
            states.append(capture_has_unjoined_work(device))
            with wp.ScopedStream(side, sync_enter=False):
                side_values.fill_(2.0)
            states.append(capture_has_unjoined_work(device))
            main_values.fill_(3.0)
            states.append(capture_has_unjoined_work(device))
            main.wait_event(side.record_event())
            states.append(capture_has_unjoined_work(device))
        self.assertEqual(states, [False, False, True, True, False])
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(main_values.numpy(), 3.0)
        np.testing.assert_array_equal(side_values.numpy(), 2.0)


if __name__ == "__main__":
    unittest.main()

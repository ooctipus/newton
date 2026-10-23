# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the capture query: which capture records a stream."""

import unittest
from types import SimpleNamespace
from unittest import mock

import warp as wp

from newton._src.solvers.feather_pgs import capture_joins


class TestCaptureJoins(unittest.TestCase):
    def test_capture_id_reports_zero_where_the_registry_cannot_answer(self):
        """Name the capture recording the current stream, and 0 wherever Warp's capture registry cannot answer.

        A CPU device, a device without a capture registry, a stream the registry does not hold and
        a graph without a capture id all report 0, which no recorded capture matches.
        """
        stream = object()
        cases = (
            ("cpu device", SimpleNamespace(is_cuda=False, captures={stream: SimpleNamespace(capture_id=9)}), 0),
            ("no registry", SimpleNamespace(is_cuda=True), 0),
            (
                "stream not registered",
                SimpleNamespace(is_cuda=True, captures={object(): SimpleNamespace(capture_id=9)}),
                0,
            ),
            ("graph without an id", SimpleNamespace(is_cuda=True, captures={stream: SimpleNamespace()}), 0),
            (
                "graph with an empty id",
                SimpleNamespace(is_cuda=True, captures={stream: SimpleNamespace(capture_id=None)}),
                0,
            ),
            (
                "registered capture",
                SimpleNamespace(is_cuda=True, captures={stream: SimpleNamespace(capture_id=42)}),
                42,
            ),
        )
        for name, device, expected in cases:
            with self.subTest(name), mock.patch.object(wp, "get_stream", return_value=stream):
                self.assertEqual(capture_joins.capture_id(device), expected)

    @unittest.skipUnless(wp.is_cuda_available(), "stream capture needs CUDA")
    def test_successive_captures_report_distinct_ids(self):
        """Report 0 outside a capture and a distinct nonzero id inside each of two successive captures."""
        device = wp.get_device("cuda:0")
        values = wp.zeros(4, dtype=float, device=device)
        ids = [capture_joins.capture_id(device)]
        for _ in range(2):
            with wp.ScopedCapture(device=device):
                values.fill_(1.0)
                ids.append(capture_joins.capture_id(device))
        ids.append(capture_joins.capture_id(device))
        self.assertEqual((ids[0], ids[3]), (0, 0))
        self.assertTrue(ids[1] != 0 and ids[2] != 0 and ids[1] != ids[2], ids)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the CUDA driver query that tells whether a stream capture can pause for a conditional node."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import capture_joins
from newton._src.solvers.feather_pgs.capture_joins import capture_has_unjoined_work, capture_queries_available


def _fake_driver(edges, dependencies, *, status=1, failing=None):
    """Return driver queries over a capture graph of integer node handles 1, 2, ... and directed ``edges``.

    ``failing`` names a query (``"capture"``, ``"nodes"`` or ``"edges"``) that
    returns an error code.
    """
    nodes = sorted({node for edge in edges for node in edge} | set(dependencies))
    kept = []

    def capture_info(stream, status_out, id_out, graph_out, dependencies_out, count_out):
        if failing == "capture":
            return 1
        status_out._obj.value = status
        graph_out._obj.value = 1
        array = (ctypes.c_void_p * len(dependencies))(*dependencies)
        kept.append(array)
        pointer = ctypes.cast(array, ctypes.POINTER(ctypes.c_void_p))
        ctypes.memmove(ctypes.addressof(dependencies_out._obj), ctypes.addressof(pointer), ctypes.sizeof(pointer))
        count_out._obj.value = len(dependencies)
        return 0

    def graph_nodes(graph, out, count_out):
        if failing == "nodes":
            return 1
        if out is not None:
            for index, node in enumerate(nodes):
                out[index] = node
        count_out._obj.value = len(nodes)
        return 0

    def graph_edges(graph, sources, targets, count_out):
        if failing == "edges":
            return 1
        if sources is not None:
            for index, (source, target) in enumerate(edges):
                sources[index], targets[index] = source, target
        count_out._obj.value = len(edges)
        return 0

    return capture_info, graph_nodes, graph_edges


class TestCaptureJoins(unittest.TestCase):
    def test_leaves_outside_the_stream_dependencies_are_unjoined(self):
        """Report unjoined work exactly when a graph leaf is not a capture dependency, and when a query fails."""
        device = SimpleNamespace(is_cuda=True, stream=SimpleNamespace(cuda_stream=7))
        # Node 1 forks 2 (main stream) and 3 (a forked stream); 4 joins 2 and 3 when it is listed.
        cases = (
            ("main stream only", [(1, 2)], [2], {}, False),
            ("forked work outstanding", [(1, 2), (1, 3)], [2], {}, True),
            ("forked work joined", [(1, 2), (1, 3)], [2, 3], {}, False),
            ("joined by a later node", [(1, 2), (1, 3), (2, 4), (3, 4)], [4], {}, False),
            ("no node captured yet", [], [], {}, False),
            ("capture not active", [(1, 2)], [2], {"status": 0}, True),
            ("capture query fails", [(1, 2)], [2], {"failing": "capture"}, True),
            ("node query fails", [(1, 2)], [2], {"failing": "nodes"}, True),
            ("edge query fails", [(1, 2)], [2], {"failing": "edges"}, True),
        )
        for name, edges, dependencies, options, unjoined in cases:
            with self.subTest(name):
                driver = _fake_driver(edges, dependencies, **options)
                with mock.patch.object(capture_joins, "_driver", return_value=driver):
                    self.assertEqual(capture_has_unjoined_work(device), unjoined)

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

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep solver teardown independent of cyclically finalized stream objects."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import warp as wp

from newton.solvers import SolverFeatherPGS


class TestFeatherPGSTeardown(unittest.TestCase):
    def test_drain_context_without_dereferencing_finalized_streams(self):
        """Drain queued work without trusting a stream's surviving Python wrapper."""
        owner = SimpleNamespace(
            model=SimpleNamespace(device=sentinel.device),
            _memset_stream=sentinel.finalized_stream,
            _size_streams={9: sentinel.finalized_stream},
        )
        with (
            patch.object(wp, "synchronize_device") as drain,
            patch.object(wp, "synchronize_stream", side_effect=AssertionError("finalized stream handle")),
        ):
            SolverFeatherPGS.__del__(owner)
        drain.assert_called_once_with(sentinel.device)

    def test_partial_construction_and_interpreter_shutdown(self):
        """Allow missing construction state and already unavailable CUDA runtime."""
        with patch.object(wp, "synchronize_device") as drain:
            SolverFeatherPGS.__del__(SimpleNamespace())
        drain.assert_not_called()
        for error in (AttributeError, RuntimeError, TypeError):
            with self.subTest(error=error), patch.object(wp, "synchronize_device", side_effect=error):
                SolverFeatherPGS.__del__(SimpleNamespace(model=SimpleNamespace(device=sentinel.device)))


if __name__ == "__main__":
    unittest.main()

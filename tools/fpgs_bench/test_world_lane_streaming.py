# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse complete world-lane physical controls for the streaming correction."""

import importlib
import os
import unittest
from unittest.mock import patch

import warp as wp

from tools.fpgs_bench import test_world_lane_state as original


class _Streaming:
    def setUp(self):
        """Select only the corrected owner and its fixed 32-thread launch."""
        super().setUp()
        module = importlib.import_module("newton._src.solvers.feather_pgs.world_lane_streaming")
        self.addCleanup(patch.stopall)
        patch.object(original, "WorldLaneState", module.StreamingWorldLaneState).start()
        patch.dict(os.environ, {"FEATHER_PGS_WORLD_LANE_STREAMING": "1"}).start()
        launch = wp.launch

        def corrected(kernel, *args, **kwargs):
            if "_stream13_" in kernel.key or kernel.key.startswith("world_lane_factor_predict"):
                kwargs["block_dim"] = 32
                if "dim" in kwargs and "finish" in kernel.key:
                    kwargs["dim"] = (int(kwargs["dim"]) + 31) // 32 * 32
            return launch(kernel, *args, **kwargs)

        patch.object(wp, "launch", side_effect=corrected).start()


class TestWorldLaneStreamingCPU(_Streaming, original.TestWorldLaneStateCPU):
    """Keep current/held, force/factor, reset and notification CPU controls."""


class TestWorldLaneStreamingCUDA(_Streaming, original.TestWorldLaneStateCUDA):
    """Keep the saved and actual finite-eight native p16 comparison controls."""


if __name__ == "__main__":
    unittest.main()

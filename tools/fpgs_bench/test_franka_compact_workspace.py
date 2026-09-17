# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check compact paired16 workspace ownership without changing physical work."""

import importlib
import os
import unittest
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench import test_franka_kinetic_state as retained


class TestFrankaCompactWorkspaceFactory(unittest.TestCase):
    def test_compact_factory_exists(self):
        """Require a distinct default-off factory before accepting the feature."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.franka_compact_workspace")
        self.assertTrue(callable(module.get_state_kernel))

    def test_actual_factory_selection_and_default_off(self):
        """Select only repair/finish while keeping the original owner and predictor."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.franka_compact_workspace")
        original = module.original
        capture = next(retained.captures())
        with np.load(capture["path"], allow_pickle=False) as snapshot:
            for enabled in (False, True):
                with (
                    self.subTest(enabled=enabled),
                    patch.dict(os.environ, {"FEATHER_PGS_FRANKA_COMPACT_WORKSPACE": str(int(enabled))}),
                ):
                    case = retained.bind_saved(snapshot, capture, "cpu")
                    self.assertIs(type(case.owner), original.FrankaKineticState)
                    self.assertIs(case.owner.compact_workspace, enabled)
                    factory = module.get_state_kernel if enabled else original.get_state_kernel
                    self.assertIs(case.owner.repair_kernel, factory(False))
                    self.assertIs(case.owner.finish_kernel, factory(True))
                    self.assertIs(case.owner.predictor_kernel, original.get_predictor_kernel())

    def test_workspace_lifetimes_and_original_scan_arithmetic(self):
        """Keep all live fields disjoint and bound CPU padding to sixteen slots."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.franka_compact_workspace")
        slots = [set(range(19 * body, 19 * (body + 1))) for body in range(16)]
        actions = set(range(module.ACTION_OFFSET, module.ACTION_OFFSET + 54))
        self.assertTrue(actions <= set.union(*slots[13:]))
        self.assertFalse(actions & set.union(*slots[:13]))
        axes = set(range(module.AXIS_OFFSET, module.AXIS_OFFSET + 54))
        origins = set(range(module.ORIGIN_OFFSET, module.ORIGIN_OFFSET + 9))
        shifts = set(range(module.SHIFT_OFFSET, module.SHIFT_OFFSET + 9))
        fields = [set.union(*slots), axes, origins, shifts]
        for index, first in enumerate(fields):
            self.assertLess(max(first), module.FLOATS_PER_WORLD)
            for second in fields[index + 1 :]:
                self.assertFalse(first & second)
        for motion in (False, True):
            source = module.scan_source(motion)
            self.assertNotIn("lane<32", source)
            self.assertNotIn("[32]", source)
            self.assertIn("world*32+lane", source)
            self.assertIn("round<4", source)
            self.assertIn("__syncwarp(mask)", source)
            self.assertNotIn("__syncthreads", source)
        self.assertEqual(
            module.collect_source().count("__syncwarp(mask)"),
            module.original._collect.native_snippet.count("__syncwarp(mask)"),
        )
        self.assertNotIn("s[6*dof", module.collect_source())


class _CompactEnabled:
    def setUp(self):
        """Run the inherited physical/lifecycle controls against the actual new owner."""
        flags = patch.dict(os.environ, {"FEATHER_PGS_FRANKA_COMPACT_WORKSPACE": "1"})
        flags.start()
        self.addCleanup(flags.stop)


class TestFrankaCompactWorkspaceCPU(_CompactEnabled, retained.TestFrankaKineticStateCPU):
    """Reuse all independent current/held/integration/notification CPU controls."""


class TestFrankaCompactWorkspaceCUDA(_CompactEnabled, retained.TestFrankaKineticStateCUDA):
    """Reuse saved physical controls and five-world loaded/reset/graph controls."""


if __name__ == "__main__":
    unittest.main()

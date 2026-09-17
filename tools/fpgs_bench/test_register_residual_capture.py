# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check actual register-residual dispatch and the retained capture boundary."""

import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from tools.fpgs_bench.test_sparse_factor import fixture
from tools.fpgs_bench.test_sparse_limit_jacobi import ENV

HERE = Path(__file__).with_name("chain_capture_20260916")
FLAG = "FEATHER_PGS_SPARSE_REGISTER_RESIDUAL"


class TestRegisterResidualCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the unchanged inherited observer with its explicit extension."""
        cls.observer = runpy.run_path(str(HERE / "checked_sparse.py"))

    def owner(self, enabled):
        """Construct the real owner from the existing independent CPU fixture."""
        with patch.dict(os.environ, {**ENV, FLAG: str(int(enabled))}):
            owner = fixture("cpu")["owner"]
        if enabled:
            owner.solver.constraint_count.fill_(17)
            owner.register_residual_routing.assign(np.array([1], dtype=np.int32))
        return owner

    def test_off_owner_and_explicit_flag(self):
        """Retain original dispatch and reject absent or inconsistent requests."""
        owner = self.owner(False)
        snapshot = self.observer["register_snapshot"]
        with patch.dict(os.environ, {FLAG: "0"}):
            self.assertEqual(snapshot(owner), {"requested": False, "observed": False, "check_pass": True})
        for value in (None, "yes", "1"):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = value
                with self.assertRaises(RuntimeError):
                    snapshot(owner)

    def test_actual_route_and_two_factories(self):
        """Require constructor-selected factories and complete typed routing."""
        owner = self.owner(True)
        with patch.dict(os.environ, {FLAG: "1", "FEATHER_PGS_SPARSE_LIMIT_JACOBI": "1"}):
            result = self.observer["register_snapshot"](owner)
            self.assertEqual(result["solve_key"], "sparse_register_residual43_s18_c100")
            self.assertEqual(result["fallback_key"], "sparse_register_residual_fallback43_s18_c100")
            self.assertEqual(result["routing_shape"], [1])
            self.assertEqual(result["routing_logical_bytes"], 4)
            self.assertEqual((result["small_worlds"], result["fallback_worlds"]), (1, 0))
            self.assertEqual(self.observer["limit_snapshot"](owner)["kernel_key"], result["solve_key"])

    def test_false_factories_markers_and_routes(self):
        """Reject false ownership, unwritten routes and out-of-class publication."""
        owner = self.owner(True)
        snapshot = self.observer["register_snapshot"]
        with patch.dict(os.environ, {FLAG: "1"}):
            for target, attribute, value in (
                (owner.kernels, "solve", object()),
                (owner.kernels, "solve_fallback", object()),
                (owner, "register_residual", 1),
                (owner, "limit_jacobi", False),
                (owner, "spectral_tangents", False),
            ):
                with self.subTest(attribute=attribute), patch.object(target, attribute, value):
                    with self.assertRaises(RuntimeError):
                        snapshot(owner)
            owner.register_residual_routing.fill_(2)
            with self.assertRaises(RuntimeError):
                snapshot(owner)
            owner.register_residual_routing.fill_(1)
            owner.solver.constraint_count.fill_(33)
            with self.assertRaises(RuntimeError):
                snapshot(owner)
            owner.register_residual_routing.fill_(0)
            self.assertEqual(snapshot(owner)["fallback_worlds"], 1)

    def test_same_stream_two_launch_contract(self):
        """Launch small then original fallback with the same complete inputs."""
        for enabled in (False, True):
            owner = self.owner(enabled)
            with self.subTest(enabled=enabled), patch.object(wp, "launch_tiled") as launch:
                owner.solve(owner.solver.rhs, 8, 1.0, 0)
            self.assertEqual(launch.call_count, 2 if enabled else 1)
            first = launch.call_args_list[0]
            self.assertIs(first.args[0], owner.kernels.solve)
            self.assertEqual(first.kwargs["block_dim"], 32)
            self.assertEqual(first.kwargs["dim"], [1])
            self.assertIs(first.kwargs["inputs"][5], owner.solver.row_cfm)
            self.assertEqual(len(first.kwargs["inputs"]), 16 if enabled else 15)
            if enabled:
                second = launch.call_args_list[1]
                self.assertIs(second.args[0], owner.kernels.solve_fallback)
                self.assertIs(second.kwargs["inputs"], first.kwargs["inputs"])
                self.assertIs(first.kwargs["inputs"][-1], owner.register_residual_routing)
                self.assertEqual(second.kwargs["device"], first.kwargs["device"])
                self.assertNotIn("stream", first.kwargs)
                self.assertNotIn("stream", second.kwargs)


if __name__ == "__main__":
    unittest.main()

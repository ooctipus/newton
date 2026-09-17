# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the real packet owner and its same-stream producer retirement."""

import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from tools.fpgs_bench.test_sparse_factor import fixture
from tools.fpgs_bench.test_sparse_limit_jacobi import ENV

FLAGS = {
    **ENV,
    "FEATHER_PGS_SPARSE_REGISTER_RESIDUAL": "1",
    "FEATHER_PGS_SPARSE_REGISTER_PACKETS": "1",
}


class TestRegisterPacketCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the existing observer rather than a duplicate test boundary."""
        cls.observer = runpy.run_path(str(Path(__file__).with_name("chain_capture_20260916") / "checked_sparse.py"))

    def owner(self):
        """Construct actual factories using the original CPU fixture."""
        with patch.dict(os.environ, FLAGS):
            owner = fixture("cpu")["owner"]
        owner.solver.constraint_count.fill_(17)
        owner.register_residual_routing.assign(np.array([1], dtype=np.int32))
        return owner

    def test_actual_factories_and_abi(self):
        """Observe both key-only producers and every physical fallback."""
        owner = self.owner()
        with patch.dict(os.environ, FLAGS):
            packet = self.observer["packet_snapshot"](owner)
            route = self.observer["register_snapshot"](owner)
            policy = self.observer["limit_snapshot"](owner)
        self.assertTrue(packet["observed"])
        self.assertEqual(packet["kernels"]["contacts"], "packet_contacts")
        self.assertEqual(route["solve_key"], "sparse_register_packets43_s18_c100")
        self.assertEqual(policy["kernel_key"], route["solve_key"])
        self.assertEqual(owner.data.Z.shape, (1, 100, 18))
        self.assertFalse(owner.packet_rows)

    def test_reject_false_factories_and_flags(self):
        """An enabled flag cannot conceal retained producers or a false owner."""
        owner = self.owner()
        snapshot = self.observer["packet_snapshot"]
        with patch.dict(os.environ, FLAGS):
            for name in (
                "prefix",
                "contacts",
                "solve",
                "materialize_prefix",
                "materialize_contacts",
                "materialize_restitution",
            ):
                with self.subTest(name=name), patch.object(owner.kernels, name, object()):
                    with self.assertRaises(RuntimeError):
                        snapshot(owner)
            with patch.object(owner, "register_packets", 1), self.assertRaises(RuntimeError):
                snapshot(owner)
        with patch.dict(os.environ, {**FLAGS, "FEATHER_PGS_SPARSE_REGISTER_PACKETS": "0"}):
            with self.assertRaises(RuntimeError):
                snapshot(owner)

    def test_transaction_launch_order(self):
        """Small transaction precedes filtered producers and original GS."""
        owner = self.owner()
        current_contacts = [object()]
        owner._register_packet_contacts = (8, current_contacts)
        calls = []

        def launched(kernel, **kwargs):
            calls.append((kernel, kwargs))

        with patch.object(wp, "launch", side_effect=launched), patch.object(wp, "launch_tiled", side_effect=launched):
            owner.solve(owner.solver.rhs, 8, 1.0, 0)
        self.assertEqual(
            [kernel for kernel, _ in calls],
            [
                owner.kernels.solve,
                owner.kernels.materialize_prefix,
                owner.kernels.materialize_contacts,
                owner.kernels.materialize_restitution,
                owner.kernels.solve_fallback,
            ],
        )
        small, prefix, contacts, restitution, fallback = [kwargs for _, kwargs in calls]
        self.assertEqual(len(small["inputs"]), 17)
        self.assertEqual(len(fallback["inputs"]), 16)
        self.assertEqual(small["inputs"][:-1], fallback["inputs"])
        self.assertIs(small["inputs"][-1], owner.packet_input)
        self.assertEqual(prefix["dim"], (1, 96))
        self.assertEqual(contacts["inputs"][:-1], current_contacts)
        for kwargs in (small, prefix, contacts, restitution, fallback):
            self.assertNotIn("stream", kwargs)
            self.assertEqual(kwargs["device"], owner.solver.model.device)

    def test_constructor_requires_corrected_register_owner(self):
        """Reject the new mode without its numerical-policy prerequisites."""
        with patch.dict(os.environ, {**FLAGS, "FEATHER_PGS_SPARSE_REGISTER_RESIDUAL": "0"}):
            with self.assertRaisesRegex(ValueError, "register-residual owner"):
                fixture("cpu")
        with patch.dict(os.environ, {**FLAGS, "FEATHER_PGS_SPARSE_REGISTER_PACKETS": "yes"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                fixture("cpu")


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify complete generalized laws and arbitrary physical-index mapping."""

import ast
import hashlib
import importlib
import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels, kuka_joint_world

_EXPECTED = {
    "apply_free_root_velocity_corrections": {
        "kernel": "148b80db948334d0c58c5b5fbadead1d01c26b8de4e607a6cc9c7768108f9c5a",
        "indexed": "7c97dd4aa333ed88f93ec5576be0e43ba8603098fc68cad53c7afd24930c2a03",
    },
    "remove_free_root_transport_from_qdd": {
        "kernel": "b282ef7e93d066a1a49de7ccce0cb1e8a38b61d5a006de8e1c12547d2fa4ffaa",
        "indexed": "c231978b689c1f76a3155c9cf76618435dfba5419c34ccb1786966a8baeb11ce",
    },
    "integrate_generalized_joints": {
        "kernel": "fbfd6007ca45443dd637c749cdc749aaeb1c36798ec59f87d0933799e6e7f573",
        "indexed": "2d9a165c7871d46c428bc09d3b21aaeab754f7a1594748196f3723150e10245f",
    },
    "update_qdd_from_velocity": {
        "kernel": "3f7dc187fe7c378c11879a7964aa3c51a27209e7a4feca6fc15ac2ae68906476",
        "indexed": "479a4f7012a060a0b4ca73c0cebe965307aa152c00bf959fcccb2337c2c32104",
    },
}


class TestGeneralizedOperationSource(unittest.TestCase):
    def test_indexed_operations_without_kernel_introspection(self):
        """Construct retained indexed laws without recovering kernel source."""
        with mock.patch("inspect.getsource", side_effect=RuntimeError("kernel extraction forbidden")):
            for name, expected in _EXPECTED.items():
                with self.subTest(name=name):
                    body = kuka_joint_world.operation_source(name)
                    self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), expected["indexed"])

    def test_original_kernels_and_unknown_operation(self):
        """Keep the original complete kernel form and reject unsupported operations."""
        factory = importlib.import_module("newton._src.solvers.feather_pgs.generalized_operation_source")
        for name, expected in _EXPECTED.items():
            with self.subTest(name=name):
                source = factory.get_source(name)
                canonical = ast.unparse(ast.parse(source)) + "\n"
                self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), expected["kernel"])
        with self.assertRaises(ValueError):
            factory.get_source("unsupported_operation")

    def test_permuted_physical_indices_and_prescribed_dofs(self):
        """Update each physical DOF once under an unseen nonidentity permutation."""
        factory = importlib.import_module("newton._src.solvers.feather_pgs.generalized_operation_source")
        operation = factory.bind("update_qdd_from_velocity", dict(kernels.__dict__), indexed=True)

        @wp.kernel
        def mapped(
            indices: wp.array[int],
            qd: wp.array[float],
            mask: wp.array[int],
            velocity: wp.array[float],
            acceleration: wp.array[float],
        ):
            operation(indices[wp.tid()], qd, mask, 2.0, velocity, acceleration)

        device = os.environ.get("FPGS_TEST_DEVICE", "cpu")
        n = 37
        old = np.linspace(-1.5, 2.5, n, dtype=np.float32)
        velocity = np.sin(np.arange(n)).astype(np.float32)
        mask = (np.arange(n) % 5 == 1).astype(np.int32)
        expected_velocity = np.where(mask != 0, old, velocity)
        expected_acceleration = np.where(mask != 0, 0.0, (velocity - old) * 2.0)
        arrays = [
            wp.array((np.arange(n) * 13 % n).astype(np.int32), dtype=int, device=device),
            wp.array(old, device=device),
            wp.array(mask, dtype=int, device=device),
            wp.array(velocity, device=device),
            wp.full(n, float("nan"), device=device),
        ]
        wp.launch(mapped, dim=n, inputs=arrays, device=device)
        np.testing.assert_array_equal(arrays[3].numpy(), expected_velocity)
        np.testing.assert_array_equal(arrays[4].numpy(), expected_acceleration)


if __name__ == "__main__":
    unittest.main()

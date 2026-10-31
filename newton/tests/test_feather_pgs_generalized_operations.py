# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared generalized laws are plain Warp functions with an explicit physical index."""

import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels


class TestGeneralizedOperations(unittest.TestCase):
    def test_laws_are_indexed_functions(self):
        """Every generalized law is an indexed function without generated source."""
        for name in (
            "apply_free_root_velocity_corrections",
            "remove_free_root_transport_from_qdd",
            "integrate_generalized_joints",
            "update_qdd_from_velocity",
        ):
            with self.subTest(name=name):
                self.assertIsInstance(getattr(kernels, "_light_" + name), wp.Function)

    def test_permuted_physical_indices_and_prescribed_dofs(self):
        """Update each physical DOF once under an unseen nonidentity permutation."""

        @wp.kernel
        def mapped(
            indices: wp.array[int],
            qd: wp.array[float],
            mask: wp.array[int],
            velocity: wp.array[float],
            acceleration: wp.array[float],
        ):
            kernels._light_update_qdd_from_velocity(indices[wp.tid()], qd, mask, 2.0, velocity, acceleration)

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

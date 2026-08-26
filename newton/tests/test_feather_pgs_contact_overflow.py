# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import allocate_world_contact_slots


class TestFeatherPGSContactOverflow(unittest.TestCase):
    def test_contact_routing_rejects_overflowed_materialized_prefix(self):
        """Ignore the whole contact frame when its raw count exceeds capacity."""
        device = "cpu"
        capacity = 2
        contact_slot = wp.full((capacity,), -7, dtype=wp.int32, device=device)
        contact_path = wp.full((capacity,), -7, dtype=wp.int32, device=device)
        world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

        wp.launch(
            allocate_world_contact_slots,
            dim=capacity,
            inputs=[
                wp.array([capacity + 1], dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                wp.full((capacity,), -1, dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.vec3, device=device),
                wp.zeros((capacity,), dtype=wp.vec3, device=device),
                wp.array([wp.vec3(0.0, 0.0, 1.0)] * capacity, dtype=wp.vec3, device=device),
                wp.zeros((capacity,), dtype=wp.float32, device=device),
                wp.zeros((capacity,), dtype=wp.float32, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                0,
                0,
                0,
                0,
                8,
                8,
                8,
                0,
                0.0,
                0,
            ],
            outputs=[
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                contact_slot,
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                world_slot_counter,
                contact_path,
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.int32, device=device),
            ],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_array_equal(contact_slot.numpy(), np.full(capacity, -1, dtype=np.int32))
        np.testing.assert_array_equal(contact_path.numpy(), np.full(capacity, -1, dtype=np.int32))
        self.assertEqual(int(world_slot_counter.numpy()[0]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

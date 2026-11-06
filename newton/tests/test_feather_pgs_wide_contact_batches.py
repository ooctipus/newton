# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sweep_routes_diag import sparse_diagonal_kernel
from newton.tests.feather_pgs_law_harness import diagonal_walk, leave_walk, row_velocity


@unittest.skipUnless(wp.is_cuda_available(), "contact batch screening requires CUDA")
class TestWideContactBatches(unittest.TestCase):
    def test_wide_batches_preserve_ordered_contacts(self):
        """Preserve active, inactive, patch-linked and partial contact batches."""
        for width in (6, 7, 9, 14):
            with self.subTest(dense_dofs=width):
                self._check_width(width)

    def test_reject_unsupported_batch_layouts(self):
        """Reject response widths beyond sixteen lanes and schedules without contact units."""
        for width, units in ((15, True), (9, False)):
            with self.subTest(dense_dofs=width, contact_units=units), self.assertRaises(ValueError):
                sparse_diagonal_kernel(
                    24, width + 2, width, "120", contact_units=units, speculative_contact_batches=True
                )

    def _check_width(self, width):
        worlds, capacity, dofs, prefix, contacts = 4, 24, width + 2, 2, 5
        rng = np.random.default_rng(7612 + width)
        mass = rng.uniform(0.5, 1.5, (worlds, dofs)).astype(np.float32)
        jacobian = np.zeros((worlds, capacity, dofs), dtype=np.float32)
        jacobian[:, 0, 0] = 1.0
        jacobian[:, 1, width - 1] = -1.0
        jacobian[:, prefix : prefix + 3 * contacts] = rng.normal(0.0, 0.2, (worlds, 3 * contacts, dofs))
        response = jacobian * mass[:, None, :]
        diagonal = np.sum(jacobian * response, axis=2) + np.float32(1.0e-6)
        rhs = np.zeros((worlds, capacity), dtype=np.float32)
        rhs[:, :prefix] = 0.5
        types = np.full((worlds, capacity), 2, dtype=np.int32)
        types[:, :prefix] = 3
        parents = np.full((worlds, capacity), -1, dtype=np.int32)
        mu = np.full((worlds, capacity), 0.7, dtype=np.float32)
        normal_rows = prefix + np.arange(contacts, dtype=np.int32) * 3
        for row in normal_rows:
            types[:, row] = 0
            parents[:, row + 1 : row + 3] = row
            rhs[:, row] = 0.5
        rhs[1:, normal_rows[2:]] = -0.15
        # A linked patch's positive load prevents an apparently inactive normal
        # from certifying its tangent pair as a no-op.
        parents[2, normal_rows[0]] = normal_rows[1]
        parents[2, normal_rows[1]] = normal_rows[0]
        impulses = np.zeros((worlds, capacity), dtype=np.float32)
        impulses[2, normal_rows[1]] = 0.2
        impulses[2, normal_rows[0] + 1] = 0.03
        velocity = rng.uniform(-0.01, 0.01, (worlds, dofs)).astype(np.float32)
        velocity += np.einsum("wrd,wr->wd", response, impulses)
        row_counts = np.full(worlds, prefix + 3 * contacts, dtype=np.int32)
        row_counts[-1] = prefix + 9
        serial_counts = np.array((5, 5, 5, 3), dtype=np.int32)
        # A world lists at most a unit a row (the placement's capacity).
        serial = np.full((worlds, capacity), -1, dtype=np.int32)
        serial[:, :contacts] = normal_rows
        sparse_dof = np.broadcast_to(np.array((width, width + 1), np.int32), (worlds, capacity, 2)).copy()
        sparse_dof[:, :prefix] = -1
        sparse_jy = np.stack(
            (jacobian[:, :, width], response[:, :, width], jacobian[:, :, width + 1], response[:, :, width + 1]),
            axis=2,
        )
        limit_active = np.zeros((worlds, dofs), np.int32)
        limit_active[-1, width] = 1
        lower_rhs = np.zeros((worlds, dofs), np.float32)
        lower_rhs[-1, width] = -0.05 + velocity[-1, width]
        host = [
            row_counts,
            np.arange(worlds, dtype=np.int32),
            np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs),
            rhs,
            row_velocity(jacobian, velocity),
            diagonal,
            impulses,
            types,
            parents,
            mu,
            np.full((worlds, 2), prefix, dtype=np.int32),
            # No scalar contact: no level.
            np.zeros(worlds, np.int32),
            np.zeros((worlds, capacity + 1), np.int32),
            np.full((worlds, capacity), -1, np.int32),
            serial_counts,
            serial,
            np.zeros(worlds, np.int32),
            np.arange(worlds, dtype=np.int32),
            # The dense part's walk (a diagonal one: both schedules add their change to its du, and the step's one
            # leave walks it onto the velocity alike).
            "dense_walk",
            response[:, :, :width].copy(),
            sparse_dof,
            sparse_jy,
            limit_active,
            lower_rhs,
            np.zeros((worlds, dofs), np.float32),
            mass.ravel(),
            1.0e-6,
            8,
            1.0,
            np.zeros((worlds, dofs), np.float32),
            np.zeros((worlds, dofs), np.float32),
            velocity.ravel(),
        ]
        device = wp.get_device("cuda:0")
        results = []
        for batch in (False, True):
            arrays = []
            walk = diagonal_walk(np.ones((worlds, width), np.float32), device, dofs)
            for value in host:
                if isinstance(value, str):
                    arrays.append(walk)
                elif isinstance(value, np.ndarray):
                    dtype = wp.int32 if value.dtype == np.int32 else wp.float32
                    arrays.append(wp.array(value, dtype=dtype, device=device))
                else:
                    arrays.append(value)
            kernel = sparse_diagonal_kernel(
                capacity, dofs, width, str(device.arch), contact_units=True, speculative_contact_batches=batch
            )
            wp.launch_tiled(kernel, dim=[worlds], inputs=arrays[:29], outputs=arrays[29:], block_dim=32, device=device)
            leave_walk(walk, arrays[31], device)
            results.append([arrays[i].numpy() for i in (6, 29, 30, 31)])
        for reference, candidate in zip(*results, strict=True):
            np.testing.assert_allclose(candidate, reference, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_array_equal(results[1][0][0, : row_counts[0]], 0.0)


if __name__ == "__main__":
    unittest.main()

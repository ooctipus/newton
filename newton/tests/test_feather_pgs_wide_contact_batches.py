# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sweep_routes_diag import sparse_diagonal_kernel
from newton.tests.feather_pgs_law_harness import row_velocity


@unittest.skipUnless(wp.is_cuda_available(), "contact batch screening requires CUDA")
class TestWideContactBatches(unittest.TestCase):
    def test_wide_batches_preserve_ordered_contacts(self):
        """Preserve active, inactive, patch-linked and partial contact batches."""
        for width in (6, 7, 9, 14):
            with self.subTest(dense_dofs=width):
                self._check_width(width)

    def test_reject_unsupported_batch_layouts(self):
        """Reject response widths beyond sixteen lanes and nontriple row schedules."""
        for width, triples in ((15, True), (9, False)):
            with self.subTest(dense_dofs=width, contact_triples=triples), self.assertRaises(ValueError):
                sparse_diagonal_kernel(
                    24, width + 2, width, "120", contact_triples=triples, speculative_contact_batches=True
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
        serial = np.full((worlds, (capacity + 2) // 3), -1, dtype=np.int32)
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
            np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs),
            rhs,
            row_velocity(jacobian, velocity),
            diagonal,
            impulses,
            types,
            parents,
            mu,
            np.full((worlds, 2), prefix, dtype=np.int32),
            np.zeros(worlds, np.int32),
            np.full((worlds, dofs), -1, np.int32),
            serial_counts,
            serial,
            np.zeros(worlds, np.int32),
            np.arange(worlds, dtype=np.int32),
            jacobian[:, :, :width].copy(),
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
            arrays = [
                wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else wp.float32, device=device)
                if isinstance(value, np.ndarray)
                else value
                for value in host
            ]
            kernel = sparse_diagonal_kernel(
                capacity, dofs, width, str(device.arch), contact_triples=True, speculative_contact_batches=batch
            )
            wp.launch_tiled(kernel, dim=[worlds], inputs=arrays[:27], outputs=arrays[27:], block_dim=32, device=device)
            results.append([arrays[i].numpy() for i in (5, 27, 28, 29)])
        for reference, candidate in zip(*results, strict=True):
            np.testing.assert_allclose(candidate, reference, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_array_equal(results[1][0][0, : row_counts[0]], 0.0)


if __name__ == "__main__":
    unittest.main()

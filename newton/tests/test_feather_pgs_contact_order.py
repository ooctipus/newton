# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise contact scheduling without changing allocated row ownership."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import compact_contact as compact
from newton.tests import test_feather_pgs_compact_contact as compact_fixtures
from newton.tests.test_feather_pgs_compact_contact import (
    OUTPUTS,
    fixture,
    install_prefix,
    run_candidate,
)


@unittest.skipUnless(wp.is_cuda_available(), "contact ordering requires CUDA")
class TestContactOrder(unittest.TestCase):
    def test_ordered_contacts_and_graph_replay(self):
        """Preserve poisoned, empty and sparse contacts through reused sort buffers and graphs."""
        device = wp.get_device("cuda:0")
        candidate, aux = fixture(device)
        order = compact._ContactOrder(16, 2, 40, device)
        pointers = order.keys.ptr, order.order.ptr
        fields = (
            "point0",
            "point1",
            "normal",
            "shape0",
            "shape1",
            "margin0",
            "margin1",
            "world",
            "slot",
            "art0",
            "art1",
            "path",
            "slots_needed",
        )
        permutation = np.random.default_rng(457).permutation(16)

        def launch():
            wp.launch(
                compact.clear_contact_response,
                dim=(2, 256),
                inputs=[aux.counts, aux.bounds, candidate.dense_group, candidate.J, candidate.Y],
                block_dim=256,
                device=device,
            )
            order.launch(candidate, 16)
            wp.launch(
                compact.produce_limit_response,
                dim=(2, 12),
                inputs=[
                    aux.bounds,
                    aux.counts,
                    candidate.dense_group,
                    candidate.factor,
                    candidate.J,
                    candidate.Y,
                    candidate.row_cfm,
                    candidate.diag,
                ],
                device=device,
            )

        for kept in ((7, 7), (0, 0), (2, 7), (7, 1)):
            original, reference_aux = fixture(device, kept=kept)
            for field in fields:
                getattr(candidate, field).assign(getattr(original, field).numpy()[permutation].copy())
            aux.counts.assign(reference_aux.counts)
            aux.bounds.assign(reference_aux.bounds)
            for field in OUTPUTS:
                getattr(candidate, field).fill_(
                    -777 if field in ("row_type", "row_parent", "sparse_dof") else float("nan")
                )
            run_candidate(original, reference_aux)
            install_prefix(candidate, aux)
            launch()
            with wp.ScopedCapture(device=device) as capture:
                launch()
            for _ in range(2):
                wp.capture_launch(capture.graph)
                compact_fixtures.TestCompactContact.compare(self, original, candidate, aux)
            indices = order.order.numpy()[:16]
            np.testing.assert_array_equal(np.sort(indices), np.arange(16))
            keys = order.keys.numpy()[:16]
            self.assertTrue(np.all(keys[:-1] <= keys[1:]))
            self.assertEqual(pointers, (order.keys.ptr, order.order.ptr))

    def test_sort_capacity_and_key_range(self):
        """Reject insufficient scratch or key range before launching contact work."""
        device = wp.get_device("cuda:0")
        with self.assertRaises(ValueError):
            compact._ContactOrder(16, 1 << 30, 40, device)
        data, _ = fixture(device)
        order = compact._ContactOrder(16, 2, 40, device)
        with self.assertRaises(ValueError):
            order.launch(data, 17)
        order.launch(data, 0)


if __name__ == "__main__":
    unittest.main()

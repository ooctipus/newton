# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check active, padded, and wide-chunk fused dense response publication."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_hinv_jt_fused_kernel


class TestFeatherPGSFusedResponse(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Fused tiled response requires CUDA")
    def test_exact_response_and_padding(self):
        """Preserve exact response, diagonal shifts, mappings, and inactive bytes."""
        device = wp.get_device("cuda:0")
        for dofs, capacity, threads in ((14, 64, 64), (27, 64, 64), (4, 5, 32), (2, 65, 64)):
            with self.subTest(dofs=dofs, capacity=capacity, threads=threads):
                self._check_case(device, dofs, capacity, threads)

    def _check_case(self, device, dofs, capacity, threads):
        """Use power-of-two factors so the complete reference is exactly representable."""
        counts = np.array(
            sorted(
                {
                    0,
                    1,
                    min(15, capacity),
                    min(16, capacity),
                    min(17, capacity),
                    min(31, capacity),
                    min(32, capacity),
                    min(33, capacity),
                    capacity - 1,
                    capacity,
                }
            ),
            dtype=np.int32,
        )
        worlds = len(counts)
        rng = np.random.default_rng(719)
        group_to_art = rng.permutation(worlds).astype(np.int32)
        art_to_world = rng.permutation(worlds).astype(np.int32)
        factors = np.tile(2 * np.eye(dofs, dtype=np.float32), (worlds, 1, 1))
        jacobians = rng.integers(-2, 3, (worlds, capacity, dofs)).astype(np.float32)
        cfm = np.broadcast_to(np.arange(1, capacity + 1, dtype=np.float32) / 128, (worlds, capacity)).copy()
        initial_c = np.full((worlds, capacity, capacity), -1234, dtype=np.float32)
        initial_diag = np.full((worlds, capacity), -2345, dtype=np.float32)
        initial_y = np.full_like(jacobians, -3456)
        expected_c, expected_diag, expected_y = initial_c.copy(), initial_diag.copy(), initial_y.copy()
        chunk = min(capacity, 16)
        if capacity % chunk:
            chunk = capacity
        for group in range(worlds):
            world = art_to_world[group_to_art[group]]
            count = counts[world]
            jacobians[group, count:] = 0
            stop = (int(count) + chunk - 1) // chunk * chunk
            if not stop:
                continue
            j = jacobians[group, :stop]
            y = j / 4
            c = j @ y.T
            expected_y[group, :stop] = y
            expected_c[world, :stop, :stop] = c
            expected_diag[world, :count] = np.diag(c)[:count] + cfm[world, :count]
        values = (factors, jacobians, group_to_art, art_to_world, counts, cfm, initial_c, initial_diag, initial_y)
        kernel = _get_hinv_jt_fused_kernel(dofs, capacity, str(device.arch), threads)
        stream = wp.Stream(device)
        with wp.ScopedStream(stream):
            arrays = [
                wp.array(value, dtype=wp.int32 if value.dtype == np.int32 else wp.float32, device=device)
                for value in values
            ]
            initial_outputs = [wp.clone(array) for array in arrays[6:]]

            def launch():
                wp.launch_tiled(
                    kernel,
                    dim=[worlds],
                    inputs=arrays[:6],
                    outputs=arrays[6:],
                    block_dim=threads,
                    device=device,
                    stream=stream,
                )

            launch()
            for actual, expected in zip(arrays[6:], (expected_c, expected_diag, expected_y), strict=True):
                self.assertEqual(actual.numpy().tobytes(), expected.tobytes())
            for actual, expected in zip(arrays[:6], values[:6], strict=True):
                self.assertEqual(actual.numpy().tobytes(), expected.tobytes())
            wp.capture_begin(device=device, stream=stream)
            launch()
            graph = wp.capture_end(device=device, stream=stream)
            for _ in range(2):
                for actual, initial in zip(arrays[6:], initial_outputs, strict=True):
                    wp.copy(actual, initial, stream=stream)
                wp.capture_launch(graph, stream=stream)
                for actual, expected in zip(arrays[6:], (expected_c, expected_diag, expected_y), strict=True):
                    self.assertEqual(actual.numpy().tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()

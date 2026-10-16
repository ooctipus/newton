# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The device proof's checksums see any changed or moved word in a frozen array."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.device_proof import _WORDS_PER_THREAD, DeviceProof


@unittest.skipUnless(wp.is_cuda_available(), "the device proof sums on CUDA")
class TestFeatherPGSDeviceProof(unittest.TestCase):
    def _arrays(self, device):
        rng = np.random.default_rng(3)
        # One array shorter than the thread count, one spanning many strides, and a float array.
        return {
            "short": wp.array(rng.integers(0, 1 << 30, 7, dtype=np.int32), dtype=wp.int32, device=device),
            "long": wp.array(rng.integers(0, 1 << 30, 40 * _WORDS_PER_THREAD + 5, dtype=np.int32), device=device),
            "floats": wp.array(rng.normal(size=(300, 3)).astype(np.float32), dtype=wp.vec3, device=device),
        }

    def test_unchanged_arrays_validate_and_any_change_is_seen(self):
        device = wp.get_device("cuda:0")
        arrays = self._arrays(device)
        proof = DeviceProof(list(arrays.items()), device)
        proof.validate(arrays.__getitem__, "test")
        for label in arrays:
            with self.subTest(label=label, change="word"):
                original = arrays[label].numpy().copy()
                changed = original.copy()
                flat = changed.reshape(-1)
                flat[len(flat) // 2] += 1
                wp.copy(arrays[label], wp.array(changed, dtype=arrays[label].dtype, device=device))
                with self.assertRaisesRegex(RuntimeError, label):
                    proof.validate(arrays.__getitem__, "test")
                wp.copy(arrays[label], wp.array(original, dtype=arrays[label].dtype, device=device))
            proof.validate(arrays.__getitem__, "test")
        with self.subTest(change="swap"):
            original = arrays["long"].numpy().copy()
            swapped = original.copy()
            swapped[[1, 1000]] = swapped[[1000, 1]]
            wp.copy(arrays["long"], wp.array(swapped, dtype=wp.int32, device=device))
            with self.assertRaisesRegex(RuntimeError, "long"):
                proof.validate(arrays.__getitem__, "test")

    def test_resized_array_is_rejected(self):
        device = wp.get_device("cuda:0")
        arrays = self._arrays(device)
        proof = DeviceProof(list(arrays.items()), device)
        arrays["short"] = wp.zeros(8, dtype=wp.int32, device=device)
        with self.assertRaisesRegex(RuntimeError, "short"):
            proof.validate(arrays.__getitem__, "test")


if __name__ == "__main__":
    unittest.main()

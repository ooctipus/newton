# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general route's joint-limit warp (:func:`.rows.joint_limit_warp_kernel`): one text on both devices, a warp
per articulation on CUDA and one thread running a block's articulations in turn on the CPU."""

import unittest

import numpy as np
import warp as wp

from newton.tests.feather_pgs_law_harness import PREFIX_RECORDS, prefix_warp_rows


def _rows(device, resolved=None, warps_per_block=4, articulation_count=257, size=6):
    """One D6 joint per articulation and world, with positions drawn around its [-1, 1] limits: each world's row count,
    Jacobian rows and records."""
    dof_count = articulation_count * size
    rng = np.random.default_rng(42)
    rows = prefix_warp_rows(
        device,
        articulation_count,
        size,
        2 * size,
        resolved=resolved,
        warps_per_block=warps_per_block,
        lower=np.full(dof_count, -1.0),
        upper=np.full(dof_count, 1.0),
        q=rng.uniform(-1.2, 1.2, dof_count),
        activation_gap=0.25,
        beta=0.2,
        cfm=1.0e-6,
    )
    return {name: rows[name] for name in ("count", "J", *PREFIX_RECORDS)}


class TestFeatherPGSJointLimitWarp(unittest.TestCase):
    def test_resolved_worlds_take_no_rows(self):
        """An all-zero mask builds the same rows as the empty mask, and a resolved world builds none, on every device.

        Mixed decisions share a block: a resolved group must leave its neighbors' ballots and row order unchanged.
        """
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                unmasked = _rows(device)
                self.assertGreater(int(unmasked["count"].sum()), 0)
                for name, actual in _rows(device, np.zeros(257)).items():
                    np.testing.assert_array_equal(actual, unmasked[name], err_msg=name)
                resolved = np.arange(257) % 3 == 0
                for name, actual in _rows(device, resolved).items():
                    expected = unmasked[name].copy()
                    expected[resolved] = 0
                    np.testing.assert_array_equal(actual, expected, err_msg=name)

    @unittest.skipUnless(wp.is_cuda_available(), "the warp-collective form runs on CUDA")
    def test_cuda_warps_match_the_cpu_row_order_and_values(self):
        """The CUDA warps' ballot reservation places each world's rows as the CPU's serial run of the same text does."""
        cpu = _rows("cpu")
        for warps_per_block in (1, 4):
            for name, actual in _rows("cuda:0", warps_per_block=warps_per_block).items():
                np.testing.assert_array_equal(actual, cpu[name], err_msg=f"{name}, {warps_per_block} warps a block")


if __name__ == "__main__":
    unittest.main()

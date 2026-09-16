# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete sparse panel factorization and canonical inverse publication."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from tools.fpgs_bench.test_sparse_factor import fixture, unpack


class TestSparseSupernodal(unittest.TestCase):
    device = "cpu"

    def make_fixture(self):
        """Enable only the complete panel owner on the retained actual USD."""
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SUPERNODAL": "1"}):
            return fixture(self.device)

    def launch(self, f, *, geometric=False, matrix=None, serial=False, missing_parallel=False):
        """Bind the actual refresh inputs without changing their ownership."""
        from newton._src.solvers.feather_pgs.sparse_supernodal import get_refresh_kernel  # noqa: PLC0415

        s, o = f["solver"], f["owner"]
        R = s.R_by_size[43].numpy()[0]
        K = s.aug_row_K.numpy()
        if geometric:
            original = f["H"] - np.diag(R + K)
            if matrix is not None:
                original = matrix
            reverse = original[::-1, ::-1]
            inertia = wp.array(
                reverse[o.host["row"], o.host["col"]][None].astype(np.float32), dtype=float, device=self.device
            )
        else:
            inertia = s.body_I_c
        if serial or missing_parallel:
            drive_map = wp.full(43, -1, dtype=int, device=self.device)
            drive_counts = wp.array(np.array([43], np.int32), dtype=int, device=self.device)
            drive_dofs = wp.array(np.arange(43, dtype=np.int32), dtype=int, device=self.device)
        else:
            drive_map = s._augmented_drive_row_by_dof
            drive_counts = s.aug_row_counts
            drive_dofs = s.aug_row_dof_index
        wp.launch_tiled(
            get_refresh_kernel(geometric=geometric),
            dim=[1],
            inputs=[
                o.plan,
                o.data,
                s.mass_update_mask,
                s.joint_S_s,
                inertia,
                s.R_by_size[43],
                drive_map,
                s.aug_row_K,
                drive_counts,
                drive_dofs,
                43,
                int(not serial),
            ],
            block_dim=128,
            device=self.device,
        )

    def test_factory_contract(self):
        """Require separate default-off physical and geometric refresh factories."""
        from newton._src.solvers.feather_pgs import sparse_supernodal  # noqa: PLC0415

        self.assertEqual(sparse_supernodal.get_refresh_kernel().key, "sparse_supernodal43_434")
        self.assertEqual(sparse_supernodal.get_refresh_kernel(geometric=True).key, "g1_kinetic_sparse_supernodal43_434")

    def test_actual_operator_current_held_and_drive_readiness(self):
        """Check all branch contributions against independent physical H/action."""
        f = self.make_fixture()
        o, s = f["owner"], f["solver"]
        self.assertTrue(o.supernodal)
        self.assertEqual(o.kernels.refresh.key, "sparse_supernodal43_434")
        rhs = np.random.default_rng(38).normal(0, 0.2, 43)
        original = f["H"][::-1, ::-1]
        for geometric, serial, missing_parallel in (
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, False, True),
        ):
            with self.subTest(geometric=geometric, serial=serial, missing_parallel=missing_parallel):
                o.data.W.fill_(-999)
                self.launch(f, geometric=geometric, serial=serial, missing_parallel=missing_parallel)
                self.assertEqual(int(o.data.status.numpy()[0]), 0)
                self.assertEqual(int(o.data.valid.numpy()[0]), 1)
                W = unpack(o)
                H = original.copy()
                if missing_parallel:
                    H -= np.diag(s.aug_row_K.numpy()[::-1])
                np.testing.assert_allclose(W.T @ (W @ rhs), np.linalg.solve(H, rhs), rtol=3e-4, atol=3e-5)
                np.testing.assert_allclose(np.linalg.inv(W) @ np.linalg.inv(W).T, H, rtol=3e-5, atol=3e-6)
        # A held generation never consumes changed/invalid current geometry.
        held = o.data.W.numpy().copy()
        s.mass_update_mask.zero_()
        self.launch(f, geometric=True, matrix=np.full((43, 43), np.nan))
        np.testing.assert_array_equal(o.data.W.numpy(), held)
        self.assertEqual(int(o.data.valid.numpy()[0]), 1)
        self.assertEqual(int(o.data.status.numpy()[0]), 0)

    def test_failed_refresh_and_default_dispatch(self):
        """Latch invalid factors and retain the original flag-off factory."""
        from newton._src.solvers.feather_pgs import sparse_factor as sf  # noqa: PLC0415
        from newton._src.solvers.feather_pgs.sparse_supernodal import schedule, validate_plan  # noqa: PLC0415

        f = self.make_fixture()
        o = f["owner"]
        np.testing.assert_array_equal(schedule()["index"], o.host["index"])
        validate_plan(o.host["index"])
        broken = o.host["index"].copy()
        broken[42, 42] = -1
        with self.assertRaises(ValueError):
            validate_plan(broken)
        bad_matrix = np.zeros((43, 43))
        bad_matrix[42, 42] = -100.0
        self.launch(f, geometric=True, matrix=bad_matrix)
        self.assertEqual(int(o.data.valid.numpy()[0]), 0)
        self.assertEqual(int(o.data.status.numpy()[0]), 1)
        self.launch(f, geometric=True)
        self.assertEqual(int(o.data.status.numpy()[0]), 1)
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SUPERNODAL": "0"}):
            plan, host = sf.make_plan(f["model"], f["solver"])
            original = sf.SparseFactor(f["solver"], plan, host)
        self.assertFalse(original.supernodal)
        self.assertIs(original.kernels.refresh, sf.get_refresh_kernel(original.level_update))
        self.assertIs(original.kernels.contacts, o.kernels.contacts)
        self.assertIs(original.kernels.solve, o.kernels.solve)
        self.assertEqual(o.data.W.shape, (1, 434))


@unittest.skipUnless(wp.is_cuda_available(), "Require the root-owned native device lease")
class TestSparseSupernodalCUDA(TestSparseSupernodal):
    device = "cuda:0"

    def test_original_physical_lifecycle(self):
        """Reuse independent original full steps, held reuse and graph replay."""
        from tools.fpgs_bench.test_sparse_factor import TestSparseFactorCUDA  # noqa: PLC0415

        # That retained test constructs a dense original with SPARSE_FACTOR=0,
        # so enabling this flag cannot silently change its reference factor.
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SUPERNODAL": "1", "FEATHER_PGS_SPARSE_LEVEL_UPDATE": "0"}):
            TestSparseFactorCUDA("test_complete_owner_two_steps_and_graph").test_complete_owner_two_steps_and_graph()


if __name__ == "__main__":
    unittest.main()

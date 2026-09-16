# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the opt-in hybrid contact owner's ABI and generated lifecycle."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import coupled_contact
from newton._src.solvers.feather_pgs import solver_feather_pgs as original


@wp.kernel
def _local_blocks(
    matrices: wp.array[wp.mat33],
    biases: wp.array[wp.vec3],
    friction: wp.array[float],
    output: wp.array[coupled_contact.BlockResult],
):
    i = wp.tid()
    output[i] = coupled_contact.project_contact_block(matrices[i], biases[i], friction[i])


class TestCoupledContactCPU(unittest.TestCase):
    def test_factory_abi_and_remaining_allowance(self):
        """Retain both live tier ABIs and charge every completed coupled sweep."""
        for arch in (120, 103):
            for tier in (32, 48):
                with self.subTest(arch=arch, tier=tier), patch.object(original, "_REGISTER_WHITENING", True):
                    coupled_contact.get_parallel_factory.cache_clear()
                    factory = coupled_contact.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
                    arguments = {
                        "rows": tier,
                        "min_rows": 0 if tier == 32 else 32,
                        "sweeps": 24,
                        "matrix_free": True,
                        "inkernel_response": (18, 0, 0, 0),
                        "exact_row_sums": True,
                        "world_rows": True,
                    }
                    old = original._get_pgs_solve_parallel_kernel(72, 32, 18, arch, **arguments)
                    new = factory(72, 32, 18, arch, **arguments)
                    self.assertTrue(new._fpgs_coupled_contact)
                    self.assertEqual([a.label for a in old.adj.args], [a.label for a in new.adj.args])
                    self.assertEqual(old._fpgs_block_dim, new._fpgs_block_dim)
                    self.assertEqual(
                        inspect.signature(factory), inspect.signature(original._get_pgs_solve_parallel_kernel)
                    )
                    source = new._fpgs_coupled_contact_native
                    self.assertIn("sweep < 24 - cc_consumed", source)
                    self.assertIn("cc_consumed < 6", source)
                    self.assertIn("iteration_offset + cc_consumed + sweep", source)
                    self.assertEqual(source.count("++cc_consumed;"), 1)
                    self.assertLess(source.index("if (!cc_done)"), source.index("// b'_i ="))
                    self.assertIn("s_y[lane] = s_x[lane]", source)
                    self.assertIn("s_x[i] - s_lam0[i]", source)
                    if arch == 103 and tier == 32:
                        self.assertIn("float* s_L = s_Yt", source)
                        self.assertIn("s_L[e] = ink_L_a.data", source)

    def test_unsupported_factory_retains_original(self):
        """Return the established factory for unsupported dimensions or iteration modes."""
        factory = coupled_contact.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
        arguments = {
            "rows": 32,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "exact_row_sums": True,
            "world_rows": True,
        }
        for changed in ({"sweeps": 8}, {"nesterov": False}, {"exact_row_sums": False}, {"has_drive_rows": True}):
            with self.subTest(changed=changed):
                kernel = factory(72, 32, 18, 120, **(arguments | changed))
                self.assertFalse(getattr(kernel, "_fpgs_coupled_contact", False))

    def test_literal_native_local_law_on_cpu(self):
        """Check coupled slip, open, singular rejection and proximal-delta CFM."""
        h = np.array([0.9977495887413199, -0.2281392022580242])
        multi = np.block([[np.ones((1, 1)), h[None, :]], [h[:, None], np.diag([0.1, 10.0]) + np.outer(h, h)]])
        matrix = np.array([[2.0, 0.8, 0.3], [0.8, 1.5, 0.2], [0.3, 0.2, 0.9]])
        target = np.array([1.0, -0.3, 0.4])
        bias = np.array([0.0, 0.6, -0.8]) - matrix @ target
        old = np.array([0.3, -0.1, 0.05])
        physical = matrix.copy()
        proximal = physical + 0.2 * np.eye(3)
        current = physical @ old + bias
        matrices = np.array([matrix, multi, np.eye(3), np.ones((3, 3)), proximal], dtype=np.float32)
        bn = -0.02731990326961045
        biases = np.array(
            [bias, np.r_[bn, np.ones(2) + h * bn], [1.0, 2.0, 3.0], [-1.0, 2.0, 3.0], current - proximal @ old],
            dtype=np.float32,
        )
        friction = np.array([0.5, 1.0, 0.5, 0.5, 0.5], dtype=np.float32)
        output = wp.zeros(5, dtype=coupled_contact.BlockResult, device="cpu")
        wp.launch(
            _local_blocks,
            dim=5,
            inputs=[
                wp.array(matrices, dtype=wp.mat33, device="cpu"),
                wp.array(biases, dtype=wp.vec3, device="cpu"),
                wp.array(friction, device="cpu"),
            ],
            outputs=[output],
            device="cpu",
        )
        values = output.numpy()
        self.assertEqual(values[3, 3], 0)
        for i in (0, 1, 2, 4):
            with self.subTest(case=i):
                value = values[i, :3].astype(float)
                self.assertEqual(values[i, 3], 1)
                residual = matrices[i].astype(float) @ value + biases[i]
                self.assertGreaterEqual(value[0], 0)
                self.assertLessEqual(np.linalg.norm(value[1:]) - friction[i] * value[0], 3e-5)
                self.assertGreaterEqual(residual[0], -3e-5)
                self.assertLessEqual(abs(value[0] * residual[0]), 3e-5)
                radius = friction[i] * value[0]
                self.assertLessEqual(abs(value[1:] @ residual[1:] + radius * np.linalg.norm(residual[1:])), 3e-5)


if __name__ == "__main__":
    unittest.main()

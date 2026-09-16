# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the bounded native translation of frozen ordered spectral GS."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs as original
from newton._src.solvers.feather_pgs import spectral_contact
from tools.fpgs_bench import spectral_gs_control as control


@wp.kernel
def _local_iterations(
    matrix: wp.array[wp.mat33],
    bias: wp.array[wp.vec3],
    coefficients: wp.array[wp.vec4],
    mu: wp.array[float],
    count: int,
    output: wp.array[wp.vec3],
):
    case = wp.tid()
    value = wp.vec3()
    for _ in range(count):
        residual = matrix[case] * value + bias[case]
        value = spectral_contact.project_contact(value, residual, coefficients[case], mu[case])
    output[case] = value


class TestSpectralContactCPU(unittest.TestCase):
    def test_factory_and_physical_stop(self):
        """Preserve both original owner ABIs and the frozen physical stop law."""
        with patch.object(original, "_REGISTER_WHITENING", True):
            factory = spectral_contact.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
            for arch in (120, 103):
                for tier in (32, 48):
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
                    self.assertTrue(new._fpgs_spectral_contact)
                    self.assertIn("_sgs24", new.key)
                    self.assertEqual([a.label for a in old.adj.args], [a.label for a in new.adj.args])
                    self.assertEqual(old._fpgs_block_dim, new._fpgs_block_dim)
                    source = new._fpgs_spectral_contact_native
                    self.assertIn("pass<24", source)
                    self.assertIn("while(row<n_rows)", source)
                    self.assertIn("sg_spectral", source)
                    self.assertIn("natural<3.0e-5f*scale", source)
                    self.assertIn("s_x[i] - s_lam0[i]", source)
                    self.assertNotIn("cc_schur", source)
                    self.assertNotIn("cc_block", source)
                    self.assertNotIn("probe<", source)
            self.assertEqual(inspect.signature(factory), inspect.signature(original._get_pgs_solve_parallel_kernel))

    def test_unsupported_factory_keeps_original(self):
        """Keep unsupported dimensions and policies on the unchanged complete owner."""
        factory = spectral_contact.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
        arguments = {
            "rows": 32,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "exact_row_sums": True,
            "world_rows": True,
        }
        for changed in ({"sweeps": 8}, {"has_drive_rows": True}, {"nesterov": False}, {"exact_row_sums": False}):
            kernel = factory(72, 32, 18, 120, **(arguments | changed))
            self.assertFalse(getattr(kernel, "_fpgs_spectral_contact", False))

    def test_literal_native_update_matches_frozen_cpu(self):
        """Preserve normal-before-tangent coupling and denominator-only unequal CFM."""
        matrices = np.array(
            [
                [[2.0, 0.8, 0.3], [0.8, 1.5, 0.2], [0.3, 0.2, 0.9]],
                [[2.0, 0.8, 0.3], [0.8, 1.5, 0.2], [0.3, 0.2, 0.9]],
                np.diag([1.0, 1.0, 2.0]),
            ],
            dtype=np.float32,
        )
        biases = np.array([[-1, 0.8, -0.9], [1, 0.2, -0.4], [-1, 0.3, 0.4]], np.float32)
        cfms = np.array([[0.2, 0.1, 0.4], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], np.float32)
        mus = np.array([0.7, 0.5, 0], np.float32)
        coefficients = []
        for h, cfm in zip(matrices, cfms, strict=True):
            denominator = np.linalg.eigvalsh(h[1:, 1:].astype(float))[-1] + max(cfm[1:])
            coefficients.append([h[0, 0] + cfm[0], denominator, h[0, 1], h[0, 2]])
        output = wp.zeros(3, dtype=wp.vec3, device="cpu")
        inputs = [
            wp.array(matrices, dtype=wp.mat33, device="cpu"),
            wp.array(biases, dtype=wp.vec3, device="cpu"),
            wp.array(np.asarray(coefficients, np.float32), dtype=wp.vec4, device="cpu"),
            wp.array(mus, device="cpu"),
        ]
        for iterations in (1, 24):
            wp.launch(_local_iterations, dim=3, inputs=[*inputs, iterations], outputs=[output], device="cpu")
            actual = output.numpy()
            for case, h in enumerate(matrices):
                jacobian = np.linalg.cholesky(h.astype(float))
                expected = control.solve(
                    jacobian,
                    np.eye(3),
                    np.diag(h).astype(float) + cfms[case],
                    biases[case],
                    np.array([0, 2, 2]),
                    np.array([-1, 0, 0]),
                    np.array([0, mus[case], mus[case]]),
                    np.zeros(3),
                    iterations=iterations,
                    early_stop=False,
                )
                np.testing.assert_allclose(actual[case], expected.impulses, atol=4e-7, rtol=4e-7)
                self.assertLessEqual(np.linalg.norm(actual[case, 1:]) - mus[case] * actual[case, 0], 3e-5)


if __name__ == "__main__":
    unittest.main()

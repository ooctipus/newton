# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check directed responses in the incremental-residual contact solve."""

import inspect
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_mf_gs_incremental_kernel


@unittest.skipUnless(wp.is_cuda_available(), "Incremental native solve requires CUDA")
class TestFeatherPGSIncremental(unittest.TestCase):
    def test_preeliminated_response_does_not_reintroduce_bilateral_velocity(self):
        """Preserve an eliminated bilateral row when only contact responses are projected."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Select a CUDA device for native incremental execution")

        def array(value, dtype=wp.float32):
            return wp.array(value, dtype=dtype, device=device)

        zero_int = array([0], wp.int32)
        zero_float = array([[0.0]])
        zero_spatial = wp.zeros((1, 1, 6), dtype=wp.float32, device=device)
        # H=I, J_B=(1,0), J_C=(1,1). Pre-elimination preserves Y_B but
        # projects Y_C to (0,1), so J Y^T=[[1,0],[1,1]] is not symmetric.
        # Applying the contact impulse cannot change the bilateral velocity.
        values = {
            "general_world_count": array([1], wp.int32),
            "general_worlds": zero_int,
            "general_world_grid_stride": 1,
            "use_general_world_queue": 0,
            "rb_class": zero_int,
            "world_constraint_count": array([2], wp.int32),
            "dense_phase_bounds": array([[0, 0]], wp.int32),
            "local_solve_owner": zero_int,
            "world_dof_indices": array([[0, 1]], wp.int32),
            "world_deferred_dof_mask": array([[0, 0]], wp.int32),
            "rhs_bias": array([[0.0, -1.0]]),
            "world_diag": array([[1.0, 1.0]]),
            "world_row_w": array([[1.0, 1.0]]),
            "world_impulses": array([[0.0, 0.0]]),
            "J_world": array([[[1.0, 0.0], [1.0, 1.0]]]),
            "Y_world": array([[[1.0, 0.0], [0.0, 1.0]]]),
            "world_row_type": array([[5, 0]], wp.int32),
            "world_row_parent": array([[-1, -1]], wp.int32),
            "world_row_mu": array([[0.0, 0.0]]),
            "world_torsion_group": array([[-1]], wp.int32),
            "contact_torsion_radius": 0.0,
            "mf_constraint_count": zero_int,
            "mf_contact_rows_end": zero_int,
            "mf_meta": array([[0]], wp.int32),
            "mf_impulses": zero_float,
            "mf_J_a": zero_spatial,
            "mf_J_b": zero_spatial,
            "mf_MiJt_a": zero_spatial,
            "mf_MiJt_b": zero_spatial,
            "mf_row_mu": zero_float,
            "mf_row_w": zero_float,
            "regularize": 1,
            "row_phase": 0,
            "friction_start_iteration": 0,
            "iteration_offset": 0,
            "freeze_drive_rows": 0,
            "defer_dense_response": 0,
            "v_out": array([0.0, 0.0]),
        }
        for name in ("target_vel_bias", "vel_multiplier", "impulse_multiplier", "max_impulse", "vel_limit"):
            values[f"world_drive_{name}"] = zero_float
        kernel = _get_pgs_solve_mf_gs_incremental_kernel(
            2, 1, 2, str(device.arch), incremental_rows=2, has_drive_rows=False, has_dense_velocity_limit_rows=False
        )
        for omega in (1.0, 0.65):
            for weight in (1.0, 0.8):
                for iterations in (1, 2, 8):
                    with self.subTest(omega=omega, weight=weight, iterations=iterations):
                        values["v_out"].zero_()
                        values["world_impulses"].zero_()
                        values["world_row_w"].assign(np.array([[1.0, weight]], dtype=np.float32))
                        values.update(omega=omega, iterations=iterations)
                        wp.launch_tiled(
                            kernel,
                            dim=[1],
                            inputs=[values[name] for name in inspect.signature(kernel.func).parameters],
                            block_dim=32,
                            device=device,
                        )
                        # The scalar contact recurrence is lambda'=lambda+omega*(weight-lambda).
                        expected = [0.0, weight * (1.0 - (1.0 - omega) ** iterations)]
                        np.testing.assert_allclose(values["v_out"].numpy(), expected, rtol=1.0e-6, atol=1.0e-7)
                        np.testing.assert_allclose(
                            values["world_impulses"].numpy()[0], expected, rtol=1.0e-6, atol=1.0e-7
                        )


if __name__ == "__main__":
    unittest.main()

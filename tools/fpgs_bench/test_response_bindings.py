# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve owner fallback and publication when composing response services."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kinetic_predictor import get_predictor_kernel
from tools.fpgs_bench.test_cached_force_bindings import copy_arguments
from tools.fpgs_bench.test_kinetic_live_bindings import bound_call


class TestResponseBindings(unittest.TestCase):
    def check_free_fallback(self, device):
        _solver, _owner, state, _, _, _, call = bound_call()
        state.joint_qd.assign(np.linspace(-0.07, 0.12, state.joint_qd.size, dtype=np.float32))
        call.inputs.joint_f.assign(np.linspace(-0.13, 0.17, call.inputs.joint_f.size, dtype=np.float32))
        call.construct_launch()
        call.refresh_launch()
        call.free.launch()
        call.predict_launch()
        arguments = copy_arguments(call.kernel_arguments["predict"], device)
        plan, _, held, _inputs, output = arguments
        kernel = get_predictor_kernel(str(wp.get_device(device).arch))

        def launch():
            wp.launch_tiled(kernel, dim=[2], inputs=arguments, block_dim=32, device=device)

        launch()
        names = ("joint_qdd", "v_hat", "endpoint_twists")
        expected = {name: getattr(output, name).numpy().copy() for name in names}
        groups = plan.secondary_group.numpy()
        inverse = held.inverse6.numpy()
        inverse[groups[0]] = np.nan
        held.inverse6.assign(inverse)
        launch()
        np.testing.assert_array_equal(output.status.numpy(), 0)
        for name in names:
            np.testing.assert_allclose(getattr(output, name).numpy(), expected[name], atol=3e-6, rtol=3e-5)
        if device.startswith("cuda"):
            with wp.ScopedCapture(device=device) as capture:
                launch()
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(output.status.numpy(), 0)
            np.testing.assert_allclose(output.v_hat.numpy(), expected["v_hat"], atol=3e-6, rtol=3e-5)

        # A bad primary response still rejects the complete world's publication.
        primary = held.T.numpy()
        primary[0, 40] = np.nan
        held.T.assign(primary)
        for name in names:
            getattr(output, name).fill_(731.0)
        launch()
        status = output.status.numpy()
        self.assertNotEqual(status[0], 0)
        self.assertEqual(status[1], 0)
        dofs = plan.dof_ids.numpy()
        bodies = plan.body_ids.numpy()
        for name in names:
            indices = bodies if name == "endpoint_twists" else dofs
            actual = getattr(output, name).numpy()
            np.testing.assert_array_equal(actual[indices[0]], 731.0)
            np.testing.assert_allclose(actual[indices[1]], expected[name][indices[1]], atol=3e-6, rtol=3e-5)

    def test_free_root_fallback_and_atomic_publication_cpu(self):
        """Recover an invalid free inverse while rejecting a bad primary response."""
        self.check_free_fallback("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_free_root_fallback_and_atomic_publication_cuda(self):
        """Keep fallback, failure isolation and replay inside the original warp owner."""
        self.check_free_fallback("cuda:0")


if __name__ == "__main__":
    unittest.main()

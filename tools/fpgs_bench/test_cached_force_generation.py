# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared force operands against independent physical arithmetic."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import cached_force
from newton._src.solvers.feather_pgs.device_preludes import dual_native
from newton._src.solvers.feather_pgs.spatial_laws import SPATIAL


def make_probe():
    operands = cached_force.ForceOperands(
        q="q.data",
        state_qd="qd.data",
        predictor_qd="predictor.data",
        stiffness="stiffness.data",
        reference="reference.data",
        damping="damping.data",
        dt="dt",
    )
    source = dual_native("", preludes=SPATIAL.preludes())
    source += "out.data[0]=" + operands.passive("1", "0") + ";\n"
    source += "for(int i=0;i<6;++i)out.data[1+i]=" + operands.velocity("i", "acceleration.data[i]") + ";\n"
    source += operands.transport("0", "out.data[1+k]")

    @wp.func_native(source)
    def evaluate(
        q: wp.array[float],
        qd: wp.array[float],
        predictor: wp.array[float],
        stiffness: wp.array[float],
        reference: wp.array[float],
        damping: wp.array[float],
        acceleration: wp.array[float],
        dt: float,
        out: wp.array[float],
    ): ...

    @wp.kernel(module="unique", enable_backward=False)
    def probe(
        q: wp.array[float],
        qd: wp.array[float],
        predictor: wp.array[float],
        stiffness: wp.array[float],
        reference: wp.array[float],
        damping: wp.array[float],
        acceleration: wp.array[float],
        dt: float,
        out: wp.array[float],
    ):
        evaluate(q, qd, predictor, stiffness, reference, damping, acceleration, dt, out)

    return probe


class TestCachedForceGeneration(unittest.TestCase):
    def check_physics(self, device):
        kernel = make_probe()
        rng = np.random.default_rng(442)
        q, qd, predictor = rng.normal(size=6), rng.normal(size=6), rng.normal(size=6)
        stiffness, reference, damping, acceleration = (rng.normal(size=6) for _ in range(4))
        dt = 0.013
        values = [q, qd, predictor, stiffness, reference, damping, acceleration]
        inputs = [wp.array(value, dtype=float, device=device) for value in values]
        out = wp.empty(7, dtype=float, device=device)
        wp.launch(kernel, dim=1, inputs=[*inputs, dt, out], device=device)
        velocity = predictor + dt * acceleration
        velocity[:3] += dt * np.cross(predictor[3:], predictor[:3])
        expected = np.r_[stiffness[0] * (reference[0] - q[1]) - damping[0] * qd[0], velocity]
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-6, atol=2e-6)

    def test_physical_operands_cpu(self):
        """Check q indexing, passive state velocity and predictor transport."""
        self.check_physics("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_physical_operands_cuda(self):
        """Run the same independent arithmetic oracle on the first visible GPU."""
        self.check_physics("cuda:0")


if __name__ == "__main__":
    unittest.main()

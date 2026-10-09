# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared force operands against independent physical arithmetic."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import cached_force


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
    source = "float wrench[6]={};\n"
    source += cached_force.shift_wrench("external.data[0]", "offset.data[0]", "wrench[k]", "wrench[k+3]")
    source += "float value=0.0f;\n"
    source += cached_force.project_wrench("axes.data[0]", "wrench[k]", "value")
    source += "out.data[0]=value;\n"
    source += "out.data[1]=" + operands.passive("1", "0") + ";\n"
    source += "for(int i=0;i<6;++i)out.data[2+i]=" + operands.velocity("i", "acceleration.data[i]") + ";\n"
    source += operands.transport("0", "out.data[2+k]")
    source += "float free_force=0.0f;\n"
    source += cached_force.project_wrench("axes.data[0]", "wrench[k]", "free_force", bias="bias.data[0]")
    source += "out.data[8]=free_force;\n"

    @wp.func_native(source)
    def evaluate(
        external: wp.array[wp.spatial_vector],
        offset: wp.array[wp.vec3],
        axes: wp.array[wp.spatial_vector],
        bias: wp.array[wp.spatial_vector],
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
        external: wp.array[wp.spatial_vector],
        offset: wp.array[wp.vec3],
        axes: wp.array[wp.spatial_vector],
        bias: wp.array[wp.spatial_vector],
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
        evaluate(external, offset, axes, bias, q, qd, predictor, stiffness, reference, damping, acceleration, dt, out)

    return probe


class TestCachedForceGeneration(unittest.TestCase):
    def check_physics(self, device):
        kernel = make_probe()
        rng = np.random.default_rng(442)
        force, radius, axis, bias = rng.normal(size=6), rng.normal(size=3), rng.normal(size=6), rng.normal(size=6)
        q, qd, predictor = rng.normal(size=6), rng.normal(size=6), rng.normal(size=6)
        stiffness, reference, damping, acceleration = (rng.normal(size=6) for _ in range(4))
        dt = 0.013
        values = [
            force[None],
            radius[None],
            axis[None],
            bias[None],
            q,
            qd,
            predictor,
            stiffness,
            reference,
            damping,
            acceleration,
        ]
        types = [wp.spatial_vector, wp.vec3, wp.spatial_vector, wp.spatial_vector] + [float] * 7
        inputs = [wp.array(value, dtype=dtype, device=device) for value, dtype in zip(values, types, strict=True)]
        out = wp.empty(9, dtype=float, device=device)
        for sign in (1, -1, 0):
            inputs[0].assign((sign * force)[None])
            wp.launch(kernel, dim=1, inputs=[*inputs, dt, out], device=device)
            wrench = sign * np.r_[force[:3], force[3:] + np.cross(radius, force[:3])]
            velocity = predictor + dt * acceleration
            velocity[:3] += dt * np.cross(predictor[3:], predictor[:3])
            expected = np.r_[
                axis @ wrench,
                stiffness[0] * (reference[0] - q[1]) - damping[0] * qd[0],
                velocity,
                axis @ (wrench - bias),
            ]
            np.testing.assert_allclose(out.numpy(), expected, rtol=2e-6, atol=2e-6)

    def test_physical_operands_cpu(self):
        """Check wrench origin, q indexing, passive state velocity and predictor transport."""
        self.check_physics("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
    def test_physical_operands_cuda(self):
        """Run the same independent arithmetic oracle on the first visible GPU."""
        self.check_physics("cuda:0")


if __name__ == "__main__":
    unittest.main()

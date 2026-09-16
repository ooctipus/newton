# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Factory/dataflow controls for concurrent spectral proposals, without CUDA."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import coupled_jacobi, spectral_contact, spectral_jacobi
from newton._src.solvers.feather_pgs import solver_feather_pgs as original


def _merit_native(cached):
    """Execute actual generated merit expressions with a three-row host shim."""
    source = spectral_jacobi._RECURRENCE if cached else coupled_jacobi._JACOBI_NATIVE
    for before, after in (("cj_", "sj_"), ("cc_diag", "sg_diag"), ("cc_scale", "sj_scale"), ("cc_max", "sg_max")):
        source = source.replace(before, after)
    measure = source.split("        const float cone_tolerance=3.0e-5f;", 1)[1].split(
        "        // The original float4 transpose", 1
    )[0]
    measure = "const float cone_tolerance=3.0e-5f;" + measure
    measure = measure.replace("__int_as_float(0x7f800000)", "INFINITY")
    initialization = "sg_diag[i]=diagonal[i];"
    if cached:
        initialization = """
        sg_diag[i]=1.0f/(i==0?diagonal[0]:fmaxf(diagonal[1],diagonal[2]));
        sg_physical[i]=sqrtf(diagonal[i])/(1.0e-5f*scale);
"""
    return (
        """
    auto fmaxf=[](float x,float y){return wp::max(x,y);};
    auto hypotf=[](float x,float y){return wp::sqrt(x*x+y*y);};
    float sg_diag[3],sg_physical[3],impulse[3],residual[3];
    for(int i=0;i<3;++i){impulse[i]=value[i];residual[i]=velocity[i];
"""
        + initialization
        + """
    }
    const int n_rows=3,s_kind[3]={0,1,1},s_parent[3]={-1,0,0};
    const float s_mu[3]={0.0f,friction,friction},sj_scale=scale;
    auto sg_max=[](float energy){return energy;};
    float result=0.0f;
    for(int lane=0;lane<3;++lane){
"""
        + measure
        + """
        result=fmaxf(result,sj_measure(impulse,residual));
    }
    return result;
"""
    )


@wp.func_native(_merit_native(False))
def _original_merit(value: wp.vec3, velocity: wp.vec3, diagonal: wp.vec3, friction: float, scale: float) -> float:
    """Evaluate the original literal native physical merit."""


@wp.func_native(_merit_native(True))
def _cached_merit(value: wp.vec3, velocity: wp.vec3, diagonal: wp.vec3, friction: float, scale: float) -> float:
    """Evaluate the corrected native merit with setup-produced coefficients."""


@wp.kernel
def _merit_comparison(
    impulse: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    diagonal: wp.array[wp.vec3],
    friction: wp.array[float],
    scale: wp.array[float],
    output: wp.array[wp.vec2],
):
    i = wp.tid()
    output[i] = wp.vec2(
        _original_merit(impulse[i], residual[i], diagonal[i], friction[i], scale[i]),
        _cached_merit(impulse[i], residual[i], diagonal[i], friction[i], scale[i]),
    )


@wp.kernel
def _cached_proposal(
    old: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    coefficients: wp.array[wp.vec4],
    friction: wp.array[float],
    original_output: wp.array[wp.vec3],
    cached_output: wp.array[wp.vec3],
):
    case = wp.tid()
    c = coefficients[case]
    cached = wp.vec4(1.0 / c[0], 1.0 / c[1], c[2], c[3])
    original_output[case] = spectral_contact.project_contact(old[case], residual[case], c, friction[case])
    cached_output[case] = spectral_jacobi.project_contact_cached(old[case], residual[case], cached, friction[case])


class TestSpectralJacobiNative(unittest.TestCase):
    def test_actual_merit_weights_and_thresholds(self):
        """Compare generated full merit on feasible/infeasible and near-stop states."""
        rng = np.random.default_rng(16)
        size = 256
        impulse = rng.normal(size=(size, 3)).astype(np.float32)
        impulse[:, 0] = np.abs(impulse[:, 0])
        residual = rng.normal(size=(size, 3)).astype(np.float32)
        diagonal = np.exp(rng.uniform(-4, 4, (size, 3))).astype(np.float32)
        friction = rng.uniform(0, 1, size).astype(np.float32)
        scale = rng.uniform(1, 10, size).astype(np.float32)
        # Keep both sides of the unchanged physical stop boundary explicit.
        impulse[:16] = 0
        residual[:16] = 0
        residual[:8, 0] = -np.linspace(0.1, 0.9, 8) * 3e-5
        residual[8:16, 0] = -np.linspace(1.1, 1.9, 8) * 3e-5
        diagonal[:16] = 1
        scale[:16] = 10
        inputs = [
            wp.array(impulse, dtype=wp.vec3, device="cpu"),
            wp.array(residual, dtype=wp.vec3, device="cpu"),
            wp.array(diagonal, dtype=wp.vec3, device="cpu"),
            wp.array(friction, device="cpu"),
            wp.array(scale, device="cpu"),
        ]
        output = wp.empty(size, dtype=wp.vec2, device="cpu")
        wp.launch(_merit_comparison, dim=size, inputs=inputs, outputs=[output], device="cpu")
        result = output.numpy()
        np.testing.assert_allclose(result[:, 1], result[:, 0], rtol=3e-6, atol=3e-6)
        np.testing.assert_array_equal(result[:16, 1] <= 1, result[:16, 0] <= 1)
        self.assertTrue(np.all(result[:8] <= 1))
        self.assertTrue(np.all(result[8:16] > 1))

    def test_cached_coefficients_retire_only_invariant_math(self):
        """Move reciprocal/weight production outside the unchanged pass loop."""
        self.assertTrue(hasattr(spectral_jacobi, "_CACHE_PREP"))
        measure = spectral_jacobi._RECURRENCE.split("auto sj_measure", 1)[1].split(
            "// The original float4 transpose", 1
        )[0]
        self.assertNotIn("sqrtf(sg_diag", measure)
        self.assertNotIn("/3.0e-5f", measure)
        self.assertNotIn("/cone_tolerance", measure)
        self.assertNotIn("/diag", measure)
        self.assertIn("radius/fmaxf(length,1.0e-30f)", measure)
        self.assertEqual(measure.count("hypotf("), 3)
        self.assertIn("sj_alpha==1.0f&&candidate_energy>sj_energy", spectral_jacobi._RECURRENCE)

    def test_cached_local_update_matches_literal_denominators(self):
        """Keep normal cross correction, disk feasibility and unequal denominators."""
        self.assertTrue(hasattr(spectral_jacobi, "project_contact_cached"))
        rng = np.random.default_rng(6)
        size = 128
        friction = rng.uniform(0.0, 1.0, size).astype(np.float32)
        old = rng.normal(size=(size, 3)).astype(np.float32)
        old[:, 0] = np.abs(old[:, 0])
        radius = friction * old[:, 0]
        old[:, 1:] *= np.minimum(1, radius / np.maximum(np.linalg.norm(old[:, 1:], axis=1), 1e-20))[:, None]
        residual = rng.normal(size=(size, 3)).astype(np.float32)
        coefficients = rng.normal(size=(size, 4)).astype(np.float32)
        coefficients[:, :2] = np.exp(rng.uniform(-4, 4, (size, 2))).astype(np.float32)
        friction[:3] = 0.0
        first, second = (wp.zeros(size, dtype=wp.vec3, device="cpu") for _ in range(2))
        inputs = [
            wp.array(old, dtype=wp.vec3, device="cpu"),
            wp.array(residual, dtype=wp.vec3, device="cpu"),
            wp.array(coefficients, dtype=wp.vec4, device="cpu"),
            wp.array(friction, device="cpu"),
        ]
        wp.launch(_cached_proposal, dim=size, inputs=inputs, outputs=[first, second], device="cpu")
        before, after = first.numpy(), second.numpy()
        np.testing.assert_allclose(after, before, rtol=3e-6, atol=3e-6)
        self.assertTrue(np.all(after[:, 0] >= 0))
        self.assertLessEqual(float(np.max(np.linalg.norm(after[:, 1:], axis=1) - friction * after[:, 0])), 3e-5)

    def test_actual_owner_abis_and_concurrent_dataflow(self):
        with patch.object(original, "_REGISTER_WHITENING", True):
            factory = spectral_jacobi.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
            for arch in (120, 103):
                for tier in (32, 48):
                    args = {
                        "rows": tier,
                        "min_rows": 0 if tier == 32 else 32,
                        "sweeps": 24,
                        "matrix_free": True,
                        "inkernel_response": (18, 0, 0, 0),
                        "exact_row_sums": True,
                        "world_rows": True,
                    }
                    baseline = original._get_pgs_solve_parallel_kernel(72, 32, 18, arch, **args)
                    kernel = factory(72, 32, 18, arch, **args)
                    self.assertTrue(kernel._fpgs_spectral_jacobi)
                    self.assertIn("_sj24", kernel.key)
                    self.assertEqual([a.label for a in baseline.adj.args], [a.label for a in kernel.adj.args])
                    self.assertEqual(baseline._fpgs_block_dim, kernel._fpgs_block_dim)
                    source = kernel._fpgs_spectral_jacobi_native
                    self.assertIn("pass<24", source)
                    self.assertIn("if(lane<n_rows&&s_kind[lane]==0)", source)
                    self.assertIn("sj_project_cached(old,residual,coefficients,s_mu[row+1])", source)
                    self.assertIn("sj_alpha==1.0f&&candidate_energy>sj_energy", source)
                    self.assertIn("0.5f*(s_rhs[lane]+s_step[lane])", source)
                    self.assertIn("1.0e-5f*sj_scale", source)
                    self.assertIn("cone_tolerance=3.0e-5f", source)
                    self.assertIn("s_x[i] - s_lam0[i]", source)
                    self.assertNotIn("while(row<n_rows)", source)
                    self.assertNotIn("cc_block", source)
                    self.assertNotIn("cc_metric", source)
                    self.assertNotIn("cc_schur", source)
                    self.assertNotIn("probe<", source)
            self.assertEqual(inspect.signature(factory), inspect.signature(original._get_pgs_solve_parallel_kernel))

    def test_unsupported_policies_keep_original_factory(self):
        factory = spectral_jacobi.get_parallel_factory(original._get_pgs_solve_parallel_kernel)
        args = {
            "rows": 32,
            "sweeps": 24,
            "matrix_free": True,
            "inkernel_response": (18, 0, 0, 0),
            "exact_row_sums": True,
            "world_rows": True,
        }
        for changed in ({"sweeps": 8}, {"has_drive_rows": True}, {"nesterov": False}, {"exact_row_sums": False}):
            kernel = factory(72, 32, 18, 120, **(args | changed))
            self.assertFalse(getattr(kernel, "_fpgs_spectral_jacobi", False))
            self.assertNotIn("_sj24", kernel.key)


if __name__ == "__main__":
    unittest.main()

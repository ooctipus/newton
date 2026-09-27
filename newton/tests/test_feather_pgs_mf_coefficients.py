# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check immutable mass-split projector coefficients and captured refreshes."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.mf_multiblock import MultiBlockContactSolver
from newton.tests.test_feather_pgs_mf_coloring import fixture


def prepared_fixture(device, *, direct_static=True, parallel=True):
    """Reuse mixed contact rows, including a self endpoint and an empty world."""
    contacts = [[(0, -1, True), (0, 1, True), (2, 0, True), (0, 0, True)], [], [(1, 0, True)]]
    reference, host, arrays, mapping, _ = fixture(
        contacts, device, mode="mass_split", direct_static_pairs=direct_static, parallel_prepare=parallel
    )
    # Give the self pair a nonzero combined Jacobian, and preserve its cross terms.
    host["J_b"][0, 9:12] = 0.5 * host["J_a"][0, 9:12]
    host["MiJt_b"][0, 9:12] = 0.5 * host["MiJt_a"][0, 9:12]
    host["meta"][0, 9 * 4 + 1 : 12 * 4 : 4] = np.float32(1.0 / 2.25).view(np.int32)
    # CFM is preconditioning, not an added lambda term in the physical residual.
    for world, count in enumerate(host["counts"]):
        inverse = host["meta"][world, 1 : 4 * count : 4].copy().view(np.float32)
        inverse = (1.0 / (1.0 / inverse + 0.03)).astype(np.float32)
        host["meta"][world, 1 : 4 * count : 4] = inverse.view(np.int32)
    host["w"][:] = np.resize(np.array([0.0, 1.0, 0.4, 0.2, 0.8, 1.0, 1.0e-20], np.float32), host["w"].shape)
    for name in ("meta", "J_b", "MiJt_b", "w"):
        arrays[name].assign(host[name])
    reference.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
    cached = MultiBlockContactSolver(
        reference.world_count,
        reference.max_rows,
        reference.max_world_dofs,
        reference.world_dof_indices,
        device,
        mode="mass_split",
        direct_static_pairs=direct_static,
        parallel_prepare=parallel,
        cache_coefficients=True,
    )
    cached.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
    return reference, cached, host, arrays, mapping


def prepare_coefficients(owner, arrays, regularize=1):
    """Prepare the same row inputs used by the production call site."""
    owner.prepare_coefficients(*(arrays[name] for name in ("meta", "J_a", "J_b", "MiJt_a", "MiJt_b", "w")), regularize)


def sweep(owner, arrays, *, regularize=1, friction_start=0):
    """Run exactly 24 sweeps without changing the scalar contact law."""
    arguments = [arrays[name] for name in ("meta", "impulses", "J_a", "J_b", "MiJt_a", "MiJt_b", "mu", "w", "v")]
    for iteration in range(24):
        owner.solve_mass_split(*arguments, 1.0, regularize, int(iteration >= friction_start))


class TestFeatherPGSMFCoefficients(unittest.TestCase):
    """Validate the actual preparation kernel on CPU without requiring a GPU."""

    device = "cpu"

    def test_preparation_preserves_cfm_self_cross_terms_and_direct_static(self):
        """Match independently computed projector coefficients for every admitted row."""
        for direct_static in (False, True):
            for regularize in (0, 1):
                with self.subTest(direct_static=direct_static, regularize=regularize):
                    _, owner, host, arrays, mapping = prepared_fixture(self.device, direct_static=direct_static)
                    prepare_coefficients(owner, arrays, regularize)
                    degree = owner.body_degree.numpy()
                    actual = owner.coefficients.numpy()
                    for world, count in enumerate(host["counts"]):
                        for row in range(int(count)):
                            packed = int(host["meta"][world, row * 4])
                            a, b = packed >> 16, packed & 65535
                            if b >= 32768:
                                b -= 65536
                            inverse = host["meta"][world, row * 4 + 1].view(np.float32)
                            old_diagonal = 1.0 / float(inverse)
                            split_diagonal = old_diagonal
                            for endpoint in {a, b} - {-1}:
                                J = np.zeros(6)
                                Y = np.zeros(6)
                                for side, body in (("a", a), ("b", b)):
                                    if body == endpoint:
                                        J += host["J_" + side][world, row]
                                        Y += host["MiJt_" + side][world, row]
                                valid = mapping[world, endpoint : endpoint + 6] >= 0
                                n = 1 if direct_static and min(a, b) < 0 else degree[world, endpoint]
                                split_diagonal += (n - 1) * np.dot(J[valid], Y[valid])
                            expected = [1.0 / split_diagonal, 0.0]
                            if regularize and host["kind"][world, row] == 0:
                                weight = float(host["w"][world, row])
                                g_weighted = old_diagonal * (1.0 - weight)
                                denominator = weight * split_diagonal + g_weighted
                                expected = [weight / denominator, g_weighted / denominator]
                            np.testing.assert_allclose(actual[world, row], expected, rtol=3e-6, atol=1e-8)
                    np.testing.assert_array_equal(owner.coefficient_ready.numpy(), 1 + regularize)
                    np.testing.assert_array_equal(owner.coefficient_cache_misses.numpy(), 0)

    def test_refresh_clears_empty_rows_and_recomputes_response(self):
        """Invalidate old degrees and coefficients as the graph becomes empty then loaded."""
        for parallel in (False, True):
            _, owner, host, arrays, _ = prepared_fixture(self.device, parallel=parallel)
            prepare_coefficients(owner, arrays)
            before = owner.coefficients.numpy().copy()
            arrays["counts"].zero_()
            owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
            np.testing.assert_array_equal(owner.coefficient_ready.numpy(), 0)
            prepare_coefficients(owner, arrays)
            np.testing.assert_array_equal(owner.coefficients.numpy(), 0)
            arrays["counts"].assign(host["counts"])
            arrays["MiJt_a"].assign(1.2 * host["MiJt_a"])
            owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
            prepare_coefficients(owner, arrays)
            self.assertGreater(np.max(np.abs(owner.coefficients.numpy() - before)), 1e-5)

    def test_host_guard_rejects_unprepared_and_changed_regularization(self):
        """Require refresh before a host-launched cached sweep consumes changed inputs."""
        _, owner, _, arrays, _ = prepared_fixture(self.device)
        with self.assertRaisesRegex(RuntimeError, "Refresh projection"):
            sweep(owner, arrays)
        prepare_coefficients(owner, arrays, 0)
        with self.assertRaisesRegex(RuntimeError, "Refresh projection"):
            sweep(owner, arrays, regularize=1)
        owner.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
        with self.assertRaisesRegex(RuntimeError, "Refresh projection"):
            sweep(owner, arrays, regularize=0)

    def test_uncached_owner_does_not_allocate_coefficient_buffers(self):
        """Keep causal controls free of coefficient preparation and extra row storage."""
        reference, _, _, arrays, _ = prepared_fixture(self.device)
        self.assertIsNone(reference.coefficients)
        self.assertIsNone(reference.coefficient_ready)
        with self.assertRaisesRegex(ValueError, "cache_coefficients=True"):
            prepare_coefficients(reference, arrays)
        with self.assertRaisesRegex(ValueError, "mode='mass_split'"):
            MultiBlockContactSolver(1, 1, 6, reference.world_dof_indices, self.device, cache_coefficients=True)


@unittest.skipUnless(wp.is_cuda_available(), "Cached mass-split solves require CUDA")
class TestFeatherPGSMFCoefficientsCUDA(unittest.TestCase):
    """Compare the same 24-sweep contact map and captured topology transitions."""

    device = "cuda:0"

    def test_cached_matches_uncached_with_friction_delay_and_regularization(self):
        """Preserve coupled updates with direct support, self endpoints and friction delays."""
        for direct_static in (False, True):
            for regularize, delay in ((0, 0), (1, 0), (1, 7), (1, 24)):
                with self.subTest(direct_static=direct_static, regularize=regularize, delay=delay):
                    reference, cached, host, arrays, _ = prepared_fixture(self.device, direct_static=direct_static)
                    prepare_coefficients(cached, arrays, regularize)
                    sweep(reference, arrays, regularize=regularize, friction_start=delay)
                    expected = {name: arrays[name].numpy().copy() for name in ("v", "impulses")}
                    arrays["v"].assign(host["v"])
                    arrays["impulses"].zero_()
                    sweep(cached, arrays, regularize=regularize, friction_start=delay)
                    for name, value in expected.items():
                        np.testing.assert_allclose(arrays[name].numpy(), value, rtol=5e-5, atol=5e-6)
                    np.testing.assert_array_equal(cached.coefficient_cache_misses.numpy(), 0)

    def test_captured_refresh_and_stale_graph_fallback(self):
        """Refresh captured coefficients and detect a solve-only graph after topology invalidation."""
        reference, cached, host, arrays, _ = prepared_fixture(self.device)

        def cached_step():
            cached.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
            prepare_coefficients(cached, arrays)
            sweep(cached, arrays, friction_start=3)

        cached_step()
        with wp.ScopedCapture(self.device) as complete:
            cached_step()
        with wp.ScopedCapture(self.device) as solve_only:
            sweep(cached, arrays, friction_start=3)
        for loaded in (False, True, False, True):
            arrays["counts"].assign(host["counts"] if loaded else np.zeros_like(host["counts"]))
            arrays["v"].assign(host["v"])
            arrays["impulses"].zero_()
            reference.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
            sweep(reference, arrays, friction_start=3)
            expected = {name: arrays[name].numpy().copy() for name in ("v", "impulses")}
            arrays["v"].assign(host["v"])
            arrays["impulses"].zero_()
            wp.capture_launch(complete.graph)
            for name, value in expected.items():
                np.testing.assert_allclose(arrays[name].numpy(), value, rtol=5e-5, atol=5e-6)
        np.testing.assert_array_equal(cached.coefficient_cache_misses.numpy(), 0)
        # Host-side guards cannot run inside an already captured solve-only graph.
        cached.prepare(arrays["counts"], arrays["counts"], arrays["meta"])
        arrays["v"].assign(host["v"])
        arrays["impulses"].zero_()
        wp.capture_launch(solve_only.graph)
        for name, value in expected.items():
            np.testing.assert_allclose(arrays[name].numpy(), value, rtol=5e-5, atol=5e-6)
        np.testing.assert_array_equal(cached.coefficient_cache_misses.numpy(), [1, 0, 1])


if __name__ == "__main__":
    unittest.main()

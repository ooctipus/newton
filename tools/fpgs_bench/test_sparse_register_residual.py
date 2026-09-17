# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the experimental static-register small-world representation."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_limit_jacobi as jacobi
from newton._src.solvers.feather_pgs import sparse_register_residual as register
from tools.fpgs_bench import test_sparse_limit_jacobi as limits
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench.test_sparse_contact_block import full_rows, simple_case
from tools.fpgs_bench.test_sparse_factor import fixture, unpack

ENV = {**limits.ENV, "FEATHER_PGS_SPARSE_REGISTER_RESIDUAL": "1"}


def solve_inputs(f, impulses, vout, *, iterations=8, omega=1.0, friction_start=0):
    """Use the exact original fifteen-argument current-row solve ABI."""
    s, owner = f["solver"], f["owner"]
    return [
        owner.plan,
        owner.data,
        s.constraint_count,
        s.rhs,
        s.diag,
        s.row_cfm,
        impulses,
        s.row_type,
        s.row_parent,
        s.row_mu,
        iterations,
        omega,
        friction_start,
        s.v_hat,
        vout,
    ]


def check_native(test, f, *, iterations=8, omega=1.0, friction_start=0, momentum=True):
    """Compare private original outputs and retain independent physical checks."""
    s, owner = f["solver"], f["owner"]
    test.assertTrue(owner.register_residual and owner.limit_jacobi)
    test.assertIs(owner.kernels.solve, register.get_solve_kernel())
    test.assertIs(owner.kernels.solve_fallback, register.get_fallback_kernel())
    incoming = s.impulses.numpy().copy()
    vhat = s.v_hat.numpy().copy()
    baseline_impulses = wp.clone(s.impulses)
    baseline_vout = wp.empty_like(s.v_out)
    groups = owner.plan.group_to_art.shape[0]
    wp.launch_tiled(
        jacobi.get_solve_kernel(),
        dim=[groups],
        block_dim=32,
        inputs=solve_inputs(
            f, baseline_impulses, baseline_vout, iterations=iterations, omega=omega, friction_start=friction_start
        ),
        device=s.model.device,
    )
    expected_lam, expected_v = baseline_impulses.numpy(), baseline_vout.numpy()
    owner.solve(s.rhs, iterations, omega, friction_start)
    owner.check()
    actual, lam = s.v_out.numpy(), s.impulses.numpy()
    np.testing.assert_allclose(lam, expected_lam, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(actual, expected_v, rtol=3e-4, atol=3e-5)
    test.assertTrue(np.isfinite(actual).all() and np.isfinite(lam).all())
    count = int(s.constraint_count.numpy()[0])
    if momentum and (friction_start == 0 or not np.any(incoming)):
        z, _ = full_rows(owner, count)
        response = (unpack(owner).T @ (z.T @ (lam[0, :count].astype(float) - incoming[0, :count])))[::-1]
        np.testing.assert_allclose(actual, vhat + response, rtol=3e-5, atol=3e-6)
    stats = {"register_routing": owner.register_residual_routing.numpy().tolist()}
    return actual.copy(), lam[0, :count].copy(), stats


def seed_contact(f, kind, *, incoming=None):
    """Place a small independent contact matrix in a real support template."""
    s, owner = f["solver"], f["owner"]
    z, diagonal, rhs, types, parents, mu = simple_case(kind)
    template = int(np.flatnonzero(owner.host["support_count"] >= 3)[0])
    packed = np.zeros((1, 100, 18), np.float32)
    packed[0, :3, :3] = z
    owner.data.Z.assign(packed)
    owner.data.support.fill_(template)
    owner.data.incident.zero_()
    s.constraint_count.assign(np.array([3], np.int32))
    cfm = diagonal - np.sum(z * z, axis=1)
    for name, values in (
        ("diag", diagonal),
        ("rhs", rhs),
        ("row_type", types),
        ("row_parent", parents),
        ("row_mu", mu),
        ("row_cfm", cfm),
        ("impulses", np.zeros(3) if incoming is None else incoming),
    ):
        array = getattr(s, name)
        data = np.zeros(array.shape, dtype=array.numpy().dtype)
        data[0, :3] = values
        array.assign(data)
    s.v_hat.zero_()
    s.v_out.fill_(np.nan)


class TestSparseRegisterResidual(unittest.TestCase):
    def test_factory(self):
        """Require separate cached small and original-fallback owners."""
        small = register.get_solve_kernel()
        fallback = register.get_fallback_kernel()
        self.assertEqual(small.key, "sparse_register_residual43_s18_c100")
        self.assertEqual(fallback.key, "sparse_register_residual_fallback43_s18_c100")
        self.assertIs(small, register.get_solve_kernel())
        self.assertIs(fallback, register.get_fallback_kernel())

    def test_actual_owner_and_static_storage(self):
        """Require explicit ownership, named Gram scalars and separate fallback resources."""
        with patch.dict(os.environ, ENV):
            owner = fixture("cpu")["owner"]
        self.assertTrue(owner.register_residual)
        self.assertEqual(owner.register_residual_routing.shape, (1,))
        self.assertEqual(owner.plan.group_to_art.shape, (1,))
        self.assertIs(owner.kernels.solve, register.get_solve_kernel())
        self.assertIs(owner.kernels.solve_fallback, register.get_fallback_kernel())
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_REGISTER_RESIDUAL": "0"}):
            old = fixture("cpu")["owner"]
        self.assertFalse(old.register_residual)
        self.assertIs(old.kernels.solve, jacobi.get_solve_kernel())
        self.assertIsNone(old.register_residual_routing)
        for flags in ({"FEATHER_PGS_SPARSE_REGISTER_RESIDUAL": "bad"}, {"FEATHER_PGS_SPARSE_LIMIT_JACOBI": "0"}):
            with self.subTest(flags=flags), patch.dict(os.environ, {**ENV, **flags}):
                with self.assertRaises(ValueError):
                    fixture("cpu")
        source = register.native_source()
        for row in range(32):
            self.assertIn(f"float g{row}=0.0f", source)
            self.assertIn(f"if({row}<count", source)
        self.assertNotIn("float gram[", source)
        self.assertNotIn("__syncthreads", source)
        self.assertIn("__shared__ float staged[32*44]", source)
        self.assertIn("applied_row+=delta", source)
        self.assertIn("routing.data[world]=0", source)
        self.assertIn("isfinite(omega) && omega>=0.0f", source)
        self.assertLess(source.index("group>=p.group_to_art.shape[0]"), source.index("p.group_to_art.data[group]"))

    def test_cpu_routing_reset(self):
        """Clear the routing word even though the inherited CPU solve is unsupported."""
        with patch.dict(os.environ, ENV):
            f = fixture("cpu")
        owner = f["owner"]
        owner.register_residual_routing.fill_(7)
        wp.launch_tiled(
            register.get_solve_kernel(),
            dim=[1],
            block_dim=32,
            inputs=[*solve_inputs(f, f["solver"].impulses, f["solver"].v_out), owner.register_residual_routing],
            device="cpu",
        )
        np.testing.assert_array_equal(owner.register_residual_routing.numpy(), [0])

    def test_cpu_prefix_transaction_controls(self):
        """Reuse the fixed accepted, real-ascent rejection and incoming-delta policy controls."""
        limits.TestSparseLimitJacobi.test_fixed_cpu_map_and_transactional_rejection(self)
        limits.TestSparseLimitJacobi.test_cpu_zero_incoming_and_nonfinite(self)


@unittest.skipUnless(wp.is_cuda_available(), "Root owns native execution")
class TestSparseRegisterResidualCUDA(unittest.TestCase):
    def test_native_thresholds_prefix_and_fallback(self):
        """Test31/32/33 routing, current prefix acceptance, rejection and unsupported inputs."""
        with patch.dict(os.environ, ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        saved_w = owner.data.W.numpy().copy()
        for count in (0, 1, 13, 31, 32, 33, 86, 87, 100):
            with self.subTest(count=count):
                limits.seed_limits(f, count, incoming=True, signed=True)
                _, _, stats = check_native(self, f, iterations=2)
                self.assertEqual(stats["register_routing"], [int(count <= 32)])
        # Put both tangent rows exactly inside / across the admission threshold.
        for prefix in (28, 29, 30):
            limits.seed_limits(f, prefix, contacts=True, incoming=True, signed=True)
            _, _, stats = check_native(self, f, iterations=3)
            self.assertEqual(stats["register_routing"], [int(prefix + 3 <= 32)])
        limits.seed_limits(f, 3, ascending=True)
        _, lam, stats = check_native(self, f, iterations=1)
        np.testing.assert_allclose(lam, [1.0, 0.2, 0.04], rtol=3e-5, atol=3e-6)
        self.assertEqual(stats["register_routing"], [1])
        owner.data.W.assign(saved_w)
        limits.seed_limits(f, 2)
        z = np.zeros((1, 100, 18), np.float32)
        z[0, 0, 0] = 1.0
        z[0, 1, :2] = [0.3, np.sqrt(0.91)]
        owner.data.Z.assign(z)
        owner.data.support.zero_()
        s.diag.fill_(1.0)
        s.rhs.fill_(-1.0)
        s.row_cfm.zero_()
        _, lam, _ = check_native(self, f, iterations=1)
        np.testing.assert_allclose(lam, [1.0, 1.0], rtol=3e-5, atol=3e-6)
        # Finite Z with overflowing Gram must leave original input untouched,
        # route out, and let the ordinary one-pass action publish its result.
        limits.seed_limits(f, 1)
        z = owner.data.Z.numpy()
        z[0, 0, 0] = 1e20
        owner.data.Z.assign(z)
        s.diag.fill_(1.0)
        s.rhs.fill_(-1.0)
        _, _, stats = check_native(self, f, iterations=1, momentum=False)
        self.assertEqual(stats["register_routing"], [0])
        limits.seed_limits(f, 1)
        owner.data.status.fill_(4)
        owner.register_residual_routing.fill_(1)
        before = s.impulses.numpy().copy()
        owner.solve(s.rhs, 8, 1.0, 0)
        np.testing.assert_array_equal(owner.register_residual_routing.numpy(), [0])
        np.testing.assert_array_equal(s.impulses.numpy(), before)
        owner.data.status.zero_()
        owner.data.valid.zero_()
        owner.register_residual_routing.fill_(1)
        owner.solve(s.rhs, 8, 1.0, 0)
        np.testing.assert_array_equal(owner.register_residual_routing.numpy(), [0])

    def test_native_contact_law_and_applied_delta(self):
        """Preserve normal-first disks, CFM-only denominators, delay and scalar fallthrough."""
        with patch.dict(os.environ, ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        for kind in ("stick", "open", "slip", "cfm", "singular"):
            with self.subTest(kind=kind):
                seed_contact(f, kind)
                check_native(self, f, iterations=8)
        seed_contact(f, "cfm", incoming=np.array([0.3, 0.01, -0.01]))
        check_native(self, f, iterations=2, omega=1.2)
        seed_contact(f, "cfm", incoming=np.array([0.3, 0.01, -0.01]))
        initial = s.impulses.numpy()[0, :3].copy()
        z, _ = full_rows(owner, 3)
        actual, lam, _ = check_native(self, f, iterations=1, friction_start=1)
        naive = (unpack(owner).T @ (z.T @ (lam - initial)))[::-1]
        self.assertGreater(np.linalg.norm(actual - naive), 1e-5)
        for modify in ("zero_mu", "mismatched_mu", "mismatched_support", "negative_cfm"):
            seed_contact(f, "cfm", incoming=np.array([0.3, 0.01, -0.01]))
            if modify in ("zero_mu", "mismatched_mu"):
                values = s.row_mu.numpy()
                values[0, 1:3] = 0 if modify == "zero_mu" else [0.2, 0.3]
                s.row_mu.assign(values)
            elif modify == "negative_cfm":
                s.row_cfm.fill_(-0.1)
            else:
                values = owner.data.support.numpy()
                values[0, 2] = len(owner.host["support_count"]) - 1
                owner.data.support.assign(values)
            check_native(self, f, iterations=3)
        seed_contact(f, "cfm", incoming=np.array([0.3, 0.01, -0.01]))
        check_native(self, f, iterations=0)
        limits.seed_limits(f, 1)
        s.impulses.zero_()
        s.rhs.fill_(1.0)
        _, lam, _ = check_native(self, f, iterations=1, omega=-1.0)
        self.assertGreater(lam[0], 0.0)

    def test_native_graph_small_large_transitions(self):
        """Reuse one graph through31/33/32/empty worlds with poisoned old routing."""
        with patch.dict(os.environ, ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        graph = None
        for count in (31, 33, 32, 0, 31):
            limits.seed_limits(f, count, incoming=True, signed=True)
            initial = s.impulses.numpy().copy()
            expected, expected_lam, _ = check_native(self, f, iterations=2)
            s.impulses.assign(initial)
            if graph is None:
                with wp.ScopedCapture(device="cuda:0") as capture:
                    owner.solve(s.rhs, 2, 1.0, 0)
                graph = capture.graph
            owner.register_residual_routing.fill_(7)
            wp.capture_launch(graph)
            np.testing.assert_array_equal(s.v_out.numpy(), expected)
            np.testing.assert_array_equal(s.impulses.numpy()[0, :count], expected_lam)
            np.testing.assert_array_equal(owner.register_residual_routing.numpy(), [int(count <= 32)])
        s.constraint_count.fill_(101)
        s.v_out.fill_(17.0)
        owner.register_residual_routing.fill_(7)
        wp.capture_launch(graph)
        np.testing.assert_array_equal(owner.register_residual_routing.numpy(), [0])
        np.testing.assert_array_equal(s.v_out.numpy(), np.full(43, 17.0))

    def test_native_current_held_graph_and_empty(self):
        """Reuse independent current/held geometry and original graph-lifecycle controls."""
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check_native):
            metric.TestSparseMetricTangentsCUDA.test_native_current_held_graph_and_empty(self)

    def test_native_saved_sixteen(self):
        """Reuse independent physical J/H checks on all sixteen saved current/held cases."""
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check_native):
            metric.TestSparseMetricTangentsCUDA.test_native_saved_sixteen_current_held_epochs(self)


if __name__ == "__main__":
    unittest.main()

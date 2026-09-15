# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check solve-local zero-row expiry against the unchanged metric owner."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.sparse_factor_rows import get_solve_kernel
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench.test_sparse_factor import fixture
from tools.fpgs_bench.test_sparse_metric_tangents import METRIC_ENV

EXPIRY_ENV = {**METRIC_ENV, "FEATHER_PGS_SPARSE_ZERO_EXPIRY": "1"}
EXPIRY_KEY = "sparse_metric_expiry43_s18_c100"


def assign_rows(f, rows, rhs, *, kinds=None, incoming=None, diagonal=None, mu=None, parents=None):
    """Assign authored sparse rows without changing the held physical operator."""
    s, owner = f["solver"], f["owner"]
    rows = np.asarray(rows, np.float32)
    count, width = rows.shape
    template = int(np.flatnonzero(owner.host["support_count"] >= width)[0])
    packed = np.zeros(owner.data.Z.shape, np.float32)
    packed[0, :count, :width] = rows
    owner.data.Z.assign(packed)
    owner.data.support.fill_(template)
    owner.data.incident.zero_()
    s.v_hat.zero_()
    s.constraint_count.assign(np.array([count], np.int32))
    if diagonal is None:
        diagonal = np.maximum(np.sum(rows.astype(float) ** 2, axis=1), 1e-3)
    values = {
        "diag": diagonal,
        "rhs": rhs,
        "row_type": np.full(count, 3) if kinds is None else kinds,
        "row_parent": np.full(count, -1) if parents is None else parents,
        "row_mu": np.zeros(count) if mu is None else mu,
        "impulses": np.zeros(count) if incoming is None else incoming,
    }
    for name, value in values.items():
        array = getattr(s, name)
        data = np.zeros(array.shape, dtype=array.numpy().dtype)
        data[0, :count] = value
        array.assign(data)


def matched_solve(test, f, *, iterations=8, omega=1.0, friction_start=0, finite=True):
    """Run original and expiry from the identical incoming lambda and du=0 state."""
    s, owner = f["solver"], f["owner"]
    candidate = owner.kernels.solve
    test.assertEqual(candidate.key, EXPIRY_KEY)
    incoming = s.impulses.numpy().copy()
    try:
        owner.kernels.solve = get_solve_kernel(100, metric_tangents=True)
        owner.solve(s.rhs, iterations, omega, friction_start)
        baseline_v, baseline_lam = s.v_out.numpy().copy(), s.impulses.numpy().copy()
    finally:
        owner.kernels.solve = candidate
    s.impulses.assign(incoming)
    owner.solve(s.rhs, iterations, omega, friction_start)
    actual_v, actual_lam = s.v_out.numpy(), s.impulses.numpy()
    # Preserve the original metric-control allowances; this is not an
    # elementwise bit-identity requirement on native floating-point execution.
    np.testing.assert_allclose(actual_v, baseline_v, rtol=3e-4, atol=3e-5, equal_nan=not finite)
    np.testing.assert_allclose(actual_lam, baseline_lam, rtol=3e-4, atol=3e-5, equal_nan=not finite)
    np.testing.assert_array_equal(np.isfinite(actual_v), np.isfinite(baseline_v))
    np.testing.assert_array_equal(np.isfinite(actual_lam), np.isfinite(baseline_lam))
    if finite:
        test.assertTrue(np.isfinite(actual_v).all() and np.isfinite(actual_lam).all())
    owner.check()
    return actual_v.copy(), actual_lam.copy()


class TestSparseZeroExpiryCPU(unittest.TestCase):
    def test_default_off_and_exclusive_admission(self):
        """Keep the original factory default and require exclusive metric100 expiry."""
        original = get_solve_kernel(100, metric_tangents=True)
        explicit = get_solve_kernel(100, metric_tangents=True, zero_expiry=False)
        self.assertEqual(original.key, explicit.key)
        candidate = get_solve_kernel(100, metric_tangents=True, zero_expiry=True)
        self.assertEqual(candidate.key, EXPIRY_KEY)
        self.assertEqual(original.key, "sparse_metric_tangent43_s18_c100")
        with patch.dict(os.environ, EXPIRY_ENV):
            owner = fixture("cpu")["owner"]
        self.assertTrue(owner.zero_expiry)
        self.assertTrue(owner.metric_tangents)
        self.assertEqual(owner.kernels.solve.key, EXPIRY_KEY)
        with patch.dict(os.environ, {**EXPIRY_ENV, "FEATHER_PGS_SPARSE_ZERO_EXPIRY": "0"}):
            ordinary = fixture("cpu")["owner"]
        self.assertFalse(ordinary.zero_expiry)
        self.assertEqual(ordinary.kernels.solve.key, original.key)
        for options in (
            {"metric_tangents": False},
            {"metric_tangents": True, "block_contacts": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                get_solve_kernel(100, zero_expiry=True, **options)
        with self.assertRaises(ValueError):
            get_solve_kernel(101, metric_tangents=True, zero_expiry=True)
        for flag in ("FEATHER_PGS_SPARSE_PACKETS", "FEATHER_PGS_SPARSE_CONTACT_BLOCK"):
            with self.subTest(flag=flag), patch.dict(os.environ, {**EXPIRY_ENV, flag: "1"}):
                with self.assertRaises(ValueError):
                    fixture("cpu")
        with patch.dict(os.environ, {**EXPIRY_ENV, "FEATHER_PGS_SPARSE_METRIC_TANGENTS": "0"}):
            with self.assertRaises(ValueError):
                fixture("cpu")
        with patch.dict(os.environ, EXPIRY_ENV):
            with self.assertRaises(ValueError):
                fixture("cpu", capacity=101)

    def test_roundoff_expiry_accounts_for_future_error(self):
        """Bound future residual error as well as the residual that armed expiry."""
        c = 128 * np.finfo(np.float32).eps
        rng = np.random.default_rng(805)
        for _ in range(300):
            norm = 10.0 ** rng.uniform(-4, 4)
            t0 = 10.0 ** rng.uniform(-5, 3)
            incident, rhs = rng.normal(0, 100, 2)
            constant = 1 + abs(incident) + abs(rhs)
            error0 = c * (constant + norm * t0)
            residual = 2 * error0 + 10.0 ** rng.uniform(-5, 2)
            expiry = t0 + (residual - 2 * error0) / (norm * (1 + c))
            current = t0 + 0.99 * (expiry - t0)
            # Worst saved error, worst opposing displacement, and worst
            # future reduction/addition error must still leave positivity.
            lower = residual - error0 - norm * (current - t0) - c * (constant + norm * current)
            self.assertGreater(lower, 0)
        # Ignoring future-T error admits a false-positive interval near expiry.
        norm, t0, residual, constant = 1.0, 0.0, 1.0, 1.0
        naive = t0 + (residual - 2 * c * constant) / norm
        corrected = t0 + (residual - 2 * c * constant) / (norm * (1 + c))
        self.assertLess(corrected, naive)

    def test_clock_covers_rounded_combined_and_sibling_updates(self):
        """Include vector-add rounding even when large metric terms nearly cancel."""
        c = 128 * np.finfo(np.float32).eps
        rng = np.random.default_rng(141)
        for scale in (1e-12, 1.0, 1e12):
            du = np.zeros(18, np.float32)
            clock = 0.0
            for transaction in range(80):
                rows = (scale * rng.normal(size=(3, 18))).astype(np.float32)
                delta = rng.normal(size=3).astype(np.float32)
                if transaction % 4 == 0:
                    rows[1] = rows[0]
                    delta[1] = -delta[0]
                # A metric update commits the combined expression once;
                # scalar sibling and own contributions commit separately.
                groups = ([0, 1, 2],) if transaction % 2 == 0 else ([0], [1], [2])
                for group in groups:
                    old = du.copy()
                    contribution = np.zeros(18, np.float32)
                    for k in group:
                        contribution = np.float32(contribution + np.float32(rows[k] * delta[k]))
                    du = np.float32(du + contribution)
                    bound = sum(np.linalg.norm(rows[k].astype(float)) * abs(float(delta[k])) for k in group)
                    next_clock = clock + bound + c * (1 + clock + bound)
                    self.assertLessEqual(np.linalg.norm(du.astype(float) - old.astype(float)), next_clock - clock)
                    self.assertLessEqual(np.linalg.norm(du.astype(float)), next_clock)
                    clock = next_clock


@unittest.skipUnless(wp.is_cuda_available(), "Native zero-row expiry controls require CUDA")
class TestSparseZeroExpiryCUDA(unittest.TestCase):
    def test_native_zero_reactivation_and_scalar_rounding(self):
        """Expire zero limits after opposing updates and preserve invalid-input fallback."""
        with patch.dict(os.environ, EXPIRY_ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        for kind in (
            "safe_zero",
            "reactivate",
            "negative_cfm",
            "negative_denominator",
            "zero_z",
            "tiny_z",
            "overflow_norm",
            "nan_z",
            "nan_incident",
            "near_boundary",
            "negative_omega",
            "zero_omega",
            "nonzero_incoming",
            "iterations_zero",
        ):
            with self.subTest(kind=kind):
                rows = np.array([[1.0, 0.0], [-0.5, 1.0], [0.25, 1.0]], np.float32)
                rhs = np.array([0.05, -1.0, 0.2], np.float32)
                diagonal = np.sum(rows.astype(float) ** 2, axis=1)
                incoming = np.zeros(3)
                omega, iterations = 1.0, 8
                if kind == "safe_zero":
                    rows[0] = [0.0, 1.0]
                    rhs[0] = 100.0
                elif kind == "negative_cfm":
                    diagonal *= 0.8
                elif kind == "negative_denominator":
                    diagonal[0] = -1.0
                elif kind == "zero_z":
                    rows[0] = 0
                elif kind == "tiny_z":
                    rows[0] = np.float32(1e-30)
                elif kind == "overflow_norm":
                    rows[0] = np.float32(1e20)
                elif kind == "nan_z":
                    rows[0, 0] = np.nan
                elif kind == "near_boundary":
                    rhs[0] = np.nextafter(np.float32(0), np.float32(1))
                elif kind == "negative_omega":
                    omega = -0.25
                elif kind == "zero_omega":
                    omega = 0.0
                elif kind == "nonzero_incoming":
                    incoming[:] = [0.3, 0.1, 0.2]
                elif kind == "iterations_zero":
                    iterations = 0
                assign_rows(f, rows, rhs, diagonal=diagonal, incoming=incoming)
                if kind == "nan_incident":
                    incident = owner.data.incident.numpy()
                    incident[0, 0] = np.nan
                    owner.data.incident.assign(incident)
                _, lam = matched_solve(
                    self,
                    f,
                    iterations=iterations,
                    omega=omega,
                    finite=kind not in ("overflow_norm", "nan_z", "nan_incident"),
                )
                if kind == "safe_zero":
                    self.assertEqual(lam[0, 0], 0)
                elif kind == "reactivate":
                    self.assertGreater(lam[0, 0], 0.01)

    def test_native_metric_fallback_and_combined_clock(self):
        """Reuse all original metric guards and exercise repeated coupled tangent updates."""
        with patch.object(metric, "METRIC_ENV", EXPIRY_ENV):
            metric.TestSparseMetricTangentsCUDA.test_native_stick_open_slip_and_guarded_fallback(self)
        with patch.dict(os.environ, EXPIRY_ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        for kind in ("metric", "delayed", "overrelaxed", "incoming", "sibling", "bad_support"):
            with self.subTest(kind=kind):
                rows = np.array([[1, 0, 0], [0, 2, 0.1], [0, 0.2, 1], [-1, 0.2, 0]], np.float32)
                rhs = np.array([0.1, 2, -3, -1], np.float32)
                incoming = np.array([0.3, 0.1, -0.05, 0]) if kind == "incoming" else np.zeros(4)
                assign_rows(
                    f,
                    rows,
                    rhs,
                    kinds=[0, 2, 2, 3],
                    parents=[-1, 0, 0, -1],
                    mu=[0, 0.5, 0.5, 0],
                    incoming=incoming,
                )
                if kind == "bad_support":
                    support = owner.data.support.numpy()
                    support[0, 2] = int(np.flatnonzero(owner.host["support_count"] >= 3)[-1])
                    self.assertNotEqual(support[0, 2], support[0, 0])
                    owner.data.support.assign(support)
                if kind == "sibling":
                    mu = s.row_mu.numpy()
                    mu[0, 2] = 0.7
                    s.row_mu.assign(mu)
                matched_solve(
                    self,
                    f,
                    friction_start=2 if kind == "delayed" else 0,
                    omega=1.2 if kind == "overrelaxed" else 1.0,
                )

    def test_native_saved_sixteen_current_held_epochs(self):
        """Keep every existing loaded-state momentum, cone and metric-reference gate."""
        with patch.object(metric, "METRIC_ENV", EXPIRY_ENV):
            metric.TestSparseMetricTangentsCUDA.test_native_saved_sixteen_current_held_epochs(self)

    def test_native_current_held_graph_and_empty(self):
        """Reinitialize expiry on each replay with changed rows and empty/regrown counts."""
        with patch.object(metric, "METRIC_ENV", EXPIRY_ENV):
            metric.TestSparseMetricTangentsCUDA.test_native_current_held_graph_and_empty(self)


if __name__ == "__main__":
    unittest.main()

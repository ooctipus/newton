# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the existing spectral-GS policy in the original sparse owner."""

import hashlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sparse_metric_tangents
from newton._src.solvers.feather_pgs.sparse_spectral_tangents import get_fragments, get_solve_kernel
from tools.fpgs_bench import spectral_gs_control as control
from tools.fpgs_bench import test_sparse_metric_tangents as metric
from tools.fpgs_bench import test_spectral_gs_control as policy_tests
from tools.fpgs_bench.test_sparse_contact_block import full_rows, simple_case
from tools.fpgs_bench.test_sparse_factor import fixture, unpack

ENV = {**metric.METRIC_ENV, "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS": "1"}


def check_native(test, f, *, iterations=8, omega=1.0, friction_start=0, scalar=False):
    """Compare the native translation with the frozen independent CPU policy."""
    s, owner = f["solver"], f["owner"]
    test.assertTrue(owner.spectral_tangents)
    test.assertEqual(owner.kernels.solve.key, "sparse_spectral_tangent43_s18_c100")
    count = int(s.constraint_count.numpy()[0])
    z, templates = full_rows(owner, count)
    diagonal, rhs, types, parents, mu, incoming = (
        getattr(s, name).numpy()[0, :count].copy()
        for name in ("diag", "rhs", "row_type", "row_parent", "row_mu", "impulses")
    )
    seed = rhs.astype(float) + owner.data.incident.numpy()[0, :count]
    if scalar or omega != 1.0 or friction_start != 0:
        scalar_iterations = (
            min(iterations, friction_start) if friction_start and not scalar and omega == 1.0 else iterations
        )
        du, expected_lam, stats = metric.scalar_reference(
            z,
            z,
            diagonal,
            seed,
            types,
            parents,
            mu,
            np.zeros(43),
            iterations=scalar_iterations,
            omega=omega,
            friction_start=friction_start,
            incoming=incoming,
            templates=templates,
            block=False,
        )
        if scalar_iterations < iterations:
            result = control.solve(
                z,
                np.eye(43),
                diagonal,
                seed,
                types,
                parents,
                mu,
                du,
                iterations=iterations - scalar_iterations,
                incoming=expected_lam,
                early_stop=False,
            )
            du, expected_lam, stats = result.velocity, result.impulses, result.work
    else:
        result = control.solve(
            z,
            np.eye(43),
            diagonal,
            seed,
            types,
            parents,
            mu,
            np.zeros(43),
            iterations=iterations,
            incoming=incoming,
            early_stop=False,
        )
        du, expected_lam, stats = result.velocity, result.impulses, result.work
    vhat, W = s.v_hat.numpy().copy(), unpack(owner)
    owner.solve(s.rhs, iterations, omega, friction_start)
    owner.check()
    actual, lam = s.v_out.numpy(), s.impulses.numpy()[0, :count]
    np.testing.assert_allclose(lam, expected_lam, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(actual, vhat + (W.T @ du)[::-1], rtol=3e-4, atol=3e-5)
    test.assertTrue(np.isfinite(actual).all() and np.isfinite(lam).all())
    if friction_start == 0 or not np.any(incoming):
        np.testing.assert_allclose(actual, vhat + (W.T @ (z.T @ (lam - incoming)))[::-1], rtol=3e-5, atol=3e-6)
    return actual.copy(), lam.copy(), stats


class TestSparseSpectralTangents(unittest.TestCase):
    def test_factory(self):
        """Require the separately keyed root-free native owner."""
        self.assertEqual(get_solve_kernel().key, "sparse_spectral_tangent43_s18_c100")

    def test_owner_and_source(self):
        """Keep default ownership unchanged and retire only the metric proposal."""
        with patch.dict(os.environ, ENV):
            owner = fixture("cpu")["owner"]
        self.assertTrue(owner.spectral_tangents)
        self.assertEqual(owner.kernels.solve.key, "sparse_spectral_tangent43_s18_c100")
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_SPECTRAL_TANGENTS": "0"}):
            ordinary = fixture("cpu")["owner"]
        self.assertFalse(ordinary.spectral_tangents)
        self.assertEqual(ordinary.kernels.solve.key, "sparse_metric_tangent43_s18_c100")
        setup, update = get_fragments(100)
        self.assertEqual(setup, sparse_metric_tangents.get_fragments(100)[0])
        self.assertNotIn("probe<16", update)
        self.assertNotIn("beta1", update)
        self.assertIn("c1=cfm.data", update)
        self.assertIn("contact_cross[row+2]", update)
        self.assertIn("du[node]+=z0*change0+z1*change1+z2*change2", update)
        with patch.dict(os.environ, {**ENV, "FEATHER_PGS_SPARSE_METRIC_TANGENTS": "0"}):
            with self.assertRaises(ValueError):
                fixture("cpu")

    def test_frozen_policy_controls(self):
        """Preserve the previously tested anisotropic, duplicate and incoming law."""
        self.assertEqual(
            hashlib.sha256(Path(control.__file__).read_bytes()).hexdigest(),
            "a39eb83e4f5b240a24416f96683df0cc5e7453c8f180cbf539f9d752141256c7",
        )
        policy_tests.TestSpectralGSControl.test_duplicate_normals_see_preceding_response(self)
        policy_tests.TestSpectralGSControl.test_isolated_anisotropic_sliding(self)
        policy_tests.TestSpectralGSControl.test_incoming_state_applies_only_delta(self)


@unittest.skipUnless(wp.is_cuda_available(), "Root owns native execution")
class TestSparseSpectralTangentsCUDA(unittest.TestCase):
    def test_native_transactions(self):
        """Check scalar fallback, anisotropy, CFM and incoming delta semantics."""
        with patch.dict(os.environ, ENV):
            f = fixture("cuda:0")
        s, owner = f["solver"], f["owner"]
        owner.refresh(s)
        template = int(np.flatnonzero(owner.host["support_count"] >= 3)[0])
        owner.data.incident.zero_()
        s.v_hat.zero_()
        for kind in (
            "stick",
            "open",
            "slip",
            "cfm",
            "incoming",
            "delayed",
            "overrelaxed",
            "zero_mu",
            "closing_old_tangents",
            "limit",
            "bad_cfm",
        ):
            with self.subTest(kind=kind):
                z, diagonal, rhs, types, parents, mu = simple_case(kind)
                incoming = np.zeros(3)
                if kind in ("incoming", "delayed", "zero_mu", "closing_old_tangents"):
                    incoming[:] = [0.3, 0.01, -0.01]
                if kind == "zero_mu":
                    mu[1:] = 0.0
                elif kind == "closing_old_tangents":
                    rhs[0] = 1.0
                elif kind == "limit":
                    types[:] = 3
                cfm = diagonal - np.sum(z * z, axis=1)
                if kind == "bad_cfm":
                    cfm[1:] = -0.01
                packed = np.zeros((1, 100, 18), np.float32)
                packed[0, :3, :3] = z
                owner.data.Z.assign(packed)
                owner.data.support.fill_(template)
                s.constraint_count.assign(np.array([3], np.int32))
                for name, values in (
                    ("diag", diagonal),
                    ("rhs", rhs),
                    ("row_type", types),
                    ("row_parent", parents),
                    ("row_mu", mu),
                    ("impulses", incoming),
                    ("row_cfm", cfm),
                ):
                    array = getattr(s, name)
                    data = np.zeros(array.shape, dtype=array.numpy().dtype)
                    data[0, :3] = values
                    array.assign(data)
                check_native(
                    self,
                    f,
                    iterations=2 if kind == "delayed" else 1,
                    friction_start=int(kind == "delayed"),
                    omega=1.2 if kind == "overrelaxed" else 1.0,
                    scalar=kind == "bad_cfm",
                )

    def test_native_current_held_graph_and_empty(self):
        """Reuse the independent current/held, graph and empty-state controls."""
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check_native):
            metric.TestSparseMetricTangentsCUDA.test_native_current_held_graph_and_empty(self)

    def test_native_saved_sixteen(self):
        """Retain saved16 independent J/H momentum and current-row law checks."""
        with patch.dict(os.environ, ENV), patch.object(metric, "check_native", check_native):
            metric.TestSparseMetricTangentsCUDA.test_native_saved_sixteen_current_held_epochs(self)


if __name__ == "__main__":
    unittest.main()

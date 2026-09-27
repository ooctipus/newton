# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public optional helper with genuine elliptic contact rows."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mjwarp_linesearch_compat as compat


def _available():
    """Require the explicitly reviewed optional backend for native CPU tests."""
    return (
        importlib.util.find_spec("mujoco_warp") is not None
        and compat.importlib.metadata.version("mujoco-warp") == "3.12.0"
    )


@unittest.skipUnless(_available(), "Require optional reviewed MJWarp 3.12.0")
class EllipticTests(unittest.TestCase):
    def setUp(self):
        """Isolate the installed driver and upstream cache for each CPU test."""
        import mjwarp_elliptic_fixture as fixture  # noqa: PLC0415 - optional CPU test dependencies
        from mujoco_warp._src import solver, warp_util

        self.fixture = fixture
        self.solver, self.cache = solver, warp_util._KERNEL_CACHE
        self.original = solver._linesearch_iterative
        self.original_factory = solver._linesearch_iterative_kernel
        self.previous_cache = dict(self.cache)
        self.previous_install = compat._INSTALLED
        self.cache.clear()
        compat._INSTALLED = None

    def tearDown(self):
        """Restore original bindings even after a failing native test."""
        self.solver._linesearch_iterative = self.original
        self.solver._linesearch_iterative_kernel = self.original_factory
        self.cache.clear()
        self.cache.update(self.previous_cache)
        compat._INSTALLED = self.previous_install

    def test_elliptic_scope(self):
        """Admit the explicitly selected elliptic Newton compatibility scope."""
        from mujoco_warp._src import types

        model = SimpleNamespace(opt=SimpleNamespace(cone=types.ConeType.ELLIPTIC, solver=types.SolverType.NEWTON))
        self.assertTrue(compat.supports_model(model))

    def test_native_contact_cost_and_force(self):
        """Match native MuJoCo cost and force in every zone of real elliptic contacts."""
        import mujoco
        import numpy as np  # noqa: PLC0415 - optional CPU test dependency

        for dim in (3, 4, 6):
            _, _, _, _, model, data = self.fixture.fixture(dim)
            mu = data.contact[0].friction[0] / np.sqrt(model.opt.impratio)
            for normal, expected in ((20.0, "satisfied"), (-20.0, "quadratic"), (-0.2, "cone")):
                with self.subTest(dim=dim, zone=expected):
                    residual = np.r_[normal, np.linspace(0.3, 1.0, dim - 1)]
                    cost = np.zeros(1)
                    mujoco.mj_constraintUpdate(model, data, residual, cost, 1)
                    own, gradient, hessian, zone = self.fixture.cone_point(
                        residual, data.efc_D, data.contact[0].friction, mu
                    )
                    self.assertEqual(zone, expected)
                    self.assertAlmostEqual(cost[0], own, delta=2e-12 * max(1, abs(own)))
                    np.testing.assert_allclose(data.efc_force, -gradient, rtol=2e-13, atol=1e-12)
                    self.assertGreaterEqual(np.linalg.eigvalsh(hessian).min(), -1e-10)

    def test_selected_driver_real_contacts(self):
        """Exercise all native contact dimensions at the unchanged task budget of fifty."""
        compat.install()
        for dim, impratio, slip in ((3, 10, 1.0), (4, 10, 1.0), (6, 100, 1.0), (3, 10, 0.0), (6, 100, 8.0)):
            with self.subTest(dim=dim, impratio=impratio, slip=slip):
                with patch.object(self.solver, "_linesearch_iterative_kernel", wraps=self.original_factory) as original:
                    result = self.fixture.run_case(self.solver._linesearch_iterative, dim, impratio, slip)
                    original.assert_not_called()
                self.assertTrue(result["finite"])
                self.assertEqual(result["actual_elliptic_rows"], dim)
                self.assertEqual(result["overflow"], 0)
                self.assertLess(result["cost_change"], 0)
                self.assertLess(abs(result["scaled_cost_regret"]), 2e-10)
                self.assertLess(result["scaled_gradient"], 2e-5)

    def test_dense_sparse_fused_jv(self):
        """Preserve real row mapping through dense and CSR fused-Jv dispatch."""
        compat.install()
        for dim in (3, 4, 6):
            for sparse in (False, True):
                with self.subTest(dim=dim, sparse=sparse):
                    result = self.fixture.run_case(
                        self.solver._linesearch_iterative, dim, 100, sparse=sparse, fuse_jv=True
                    )
                    self.assertTrue(result["finite"])
                    self.assertEqual(result["overflow"], 0)
                    self.assertLess(abs(result["scaled_cost_regret"]), 2e-10)
                    self.assertLess(result["scaled_gradient"], 2e-5)

    def test_genuine_budget_exhaustion(self):
        """Retain the actual budget-one warning instead of relabeling it convergence."""
        compat.install()
        result = self.fixture.run_case(self.solver._linesearch_iterative, 6, 100, budget=1)
        self.assertTrue(result["finite"])
        self.assertEqual(result["overflow"], 1024)
        self.assertGreater(result["scaled_gradient"], 2e-5)

    def test_convex_line_reference(self):
        """Check the independent line derivatives and positive curvature numerically."""
        import numpy as np  # noqa: PLC0415 - optional CPU test dependency

        _, _, _, reference, _, _ = self.fixture.fixture(6, 100)
        derivatives = np.array([reference.point(alpha)[1] for alpha in np.linspace(0, 2, 129)])
        self.assertGreaterEqual(np.diff(derivatives).min(), -1e-8)
        for alpha in (0.1, 0.5, 1.2):
            cost, gradient, hessian, _ = reference.point(alpha)
            delta = 1e-5
            finite_difference = (reference.point(alpha + delta)[0] - reference.point(alpha - delta)[0]) / (2 * delta)
            self.assertAlmostEqual(gradient, finite_difference, delta=1e-7 * max(1, abs(gradient)))
            self.assertTrue(np.isfinite(cost))
            self.assertGreater(hessian, 0)


if __name__ == "__main__":
    unittest.main()

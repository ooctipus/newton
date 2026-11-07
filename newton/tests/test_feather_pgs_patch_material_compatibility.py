# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The material laws (contact compliance, contact torsion) run under the friction law's stick."""

import unittest
import warnings

import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_compliance import run_fixture


class TestPatchMaterialCompatibility(unittest.TestCase):
    def test_disabled_material_options_keep_default_patches(self):
        """Keep ordinary patch defaults when neither experimental material law is enabled."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.05)
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, contact_compliance=False, contact_torsion=False)
        self.assertEqual(solver.laws.friction_anchor_beta, 0.2)

    def test_torsion_opt_in_keeps_default_patches(self):
        """Resolve omitted patch gain to default patch friction alongside torsion, without a warning."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.05)
        model = builder.finalize(device="cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = SolverFeatherPGS(model, contact_torsion=True)
        self.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
        self.assertTrue(solver.laws.contact_torsion_enabled)
        self.assertAlmostEqual(solver.laws.friction_anchor_beta, 0.2)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_compliance_opt_in_keeps_default_patches(self):
        """Resolve omitted patch gain to default patch friction alongside material contacts, without a warning, and
        rest at the material's indentation on dense and MF rows."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trace, _, _, mass = run_fixture(
                    articulated=articulated, enabled=True, solver_options={"friction_anchor_beta": None}
                )
                self.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
                self.assertLess(abs(trace[-1, 2] - (0.05 - mass * 9.81 / 3000)), 2e-6)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve explicit material-law opt-ins alongside default patch friction."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_compliance import run_fixture
from newton.tests.test_feather_pgs_contact_torsion import fixture


class TestPatchMaterialCompatibility(unittest.TestCase):
    def test_disabled_material_options_keep_default_patches(self):
        """Keep ordinary patch defaults when neither experimental material law is enabled."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.05)
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, contact_compliance=False, contact_torsion_radius=0.0)
        self.assertEqual(solver.laws.friction_anchor_beta, 0.2)
        self.assertTrue(solver.laws.friction_anchors_enabled)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_torsion_opt_in_keeps_default_patches(self):
        """Resolve omitted patch gain to default patch friction alongside torsion, without a warning."""
        options = {"center_only": True, "contact_shared_anchor": False, "contact_friction_shared_anchor": False}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual, solver, *_ = fixture(0.01, friction_anchor_beta=None, **options)
        self.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
        reference, explicit, *_ = fixture(0.01, friction_anchor_beta=0.2, **options)
        self.assertTrue(solver.laws.friction_anchors_enabled)
        self.assertTrue(explicit.laws.friction_anchors_enabled)
        self.assertAlmostEqual(solver.laws.friction_anchor_beta, 0.2)
        self.assertGreater(solver.laws.torsion_stats["rows"], 0)
        for name in reference:
            np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_compliance_opt_in_keeps_default_patches(self):
        """Resolve omitted patch gain to default patch friction alongside material contacts, without a warning, and
        rest at the material's indentation on dense and MF rows."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trace, _, solver, mass = run_fixture(
                    articulated=articulated, enabled=True, solver_options={"friction_anchor_beta": None}
                )
                self.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
                self.assertTrue(solver.laws.friction_anchors_enabled)
                self.assertLess(abs(trace[-1, 2] - (0.05 - mass * 9.81 / 3000)), 2e-6)

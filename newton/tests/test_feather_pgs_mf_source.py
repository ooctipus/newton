# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve complete matrix-free recurrence and explicit representation boundaries."""

import unittest
from unittest import mock

from newton._src.solvers.feather_pgs import kinetic_compact_coupled, kinetic_hybrid, mf_gs_source, solver_feather_pgs
from newton._src.solvers.feather_pgs import kinetic_solve as original


class TestMatrixFreeSource(unittest.TestCase):
    def test_shared_physical_and_held_recurrence(self):
        """Bind both representations without inspecting a generated closure."""

        options = {
            "has_drive_rows": False,
            "has_dense_velocity_limit_rows": False,
            "factor_coordinates": True,
            "independent_components": True,
        }
        kernel = solver_feather_pgs._get_pgs_solve_mf_gs_kernel(192, 64, 29, "cpu", **options)
        source = mf_gs_source.get_source(192, 64, 29, "cpu", **options)
        self.assertEqual(source.native, kernel.mf_native.native_snippet)
        with mock.patch("inspect.getclosurevars", side_effect=AssertionError("generated-source extraction")):
            held, baseline, _ = kinetic_hybrid.general_source(kernel)
        expected = mf_gs_source.get_source(
            192,
            64,
            29,
            "cpu",
            **options,
            held_policy=(original._T, kinetic_hybrid._STAGE, kinetic_hybrid._STORE),
        )
        self.assertEqual(held, expected.native)
        self.assertEqual(baseline, source.native)
        self.assertIn(expected.mf_phase, held)
        self.assertIn("hybrid_base[lane]", expected.mf_phase)

    def test_shared_arena_has_exact_fallback_arithmetic(self):
        """Bind all seven storage views without rewriting native declarations."""

        options = {
            "has_drive_rows": False,
            "has_dense_velocity_limit_rows": False,
            "factor_coordinates": True,
            "independent_components": True,
        }
        physical = mf_gs_source.get_source(192, 64, 29, "cpu", **options)
        arena = mf_gs_source.get_source(192, 64, 29, "cpu", **options, shared_arena=True)
        expected, _ = kinetic_compact_coupled.fallback_source(physical.native)
        self.assertEqual(arena.native, expected)
        self.assertNotIn("__shared__", arena.native)
        self.assertEqual(arena.arena_floats, 1053)

    def test_held_and_arena_reject_other_recipes(self):
        """Prevent specialized representation storage from escaping its ABI."""

        for options in ({"held_policy": ("", "", "")}, {"shared_arena": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                mf_gs_source.get_source(16, 12, 9, "cpu", **options)


if __name__ == "__main__":
    unittest.main()

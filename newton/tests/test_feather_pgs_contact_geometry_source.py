# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retain one traversal and material law across scalar, triplet and cached rows."""

import importlib
import unittest


class TestContactGeometrySource(unittest.TestCase):
    def test_original_and_cached_consumers_share_prelude(self):
        factory = importlib.import_module("newton._src.solvers.feather_pgs.contact_geometry_source")
        rows = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_rows")
        triplet = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_rows_triplet")
        current = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_current_contact")
        original = factory.get_prelude(rows._COMMON)
        cached = factory.get_prelude(rows._COMMON, cached_geometry=True)
        self.assertTrue(rows._CONTACT.startswith(original))
        self.assertEqual(triplet._PRELUDE, original)
        self.assertEqual(current._contact_source(), cached + triplet._TRIPLET)
        self.assertIn("wp::transform_point", original)
        self.assertNotIn("wp::transform_point", cached)
        self.assertIn("state.held_generation.data[world]", cached)

    def test_row_staging_has_explicit_offset_and_geometry(self):
        factory = importlib.import_module("newton._src.solvers.feather_pgs.contact_geometry_source")
        current = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_current_contact")
        source = factory.get_stage_source()
        self.assertEqual(source, current._stage_source())
        self.assertIn("const float phi=s[15]", source)
        self.assertIn("s[19]=friction?3.0f:1.0f", source)
        self.assertNotIn("s[83]", source)
        self.assertNotIn("wp::transform_point", source)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject retired heightfield recipes without changing the disabled route."""

import os
import unittest
from unittest.mock import patch

from newton._src.geometry.contact_reduction_global import create_export_reduced_contacts_kernel
from newton._src.geometry.narrow_phase import NarrowPhase, write_contact_simple

FLAG = "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD"


class TestHeightfieldRetiredExperiments(unittest.TestCase):
    def test_reject_retired_flag_before_buffer_setup(self):
        """Identify the retired selector before validating or allocating buffers."""
        with patch.dict(os.environ, {FLAG: "1"}):
            with self.assertRaisesRegex(ValueError, FLAG + ".*retired"):
                NarrowPhase(max_candidate_pairs=1, max_mesh_mesh_pairs=-1, device="cpu")

    def test_unset_and_zero_keep_original_export(self):
        """Preserve original reduced export when the obsolete option is disabled."""
        flags = {
            "NEWTON_HEIGHTFIELD_CELL_REJECT": "0",
            "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
            "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "0",
            "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
            "NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL": "0",
        }
        for value in (None, "0"):
            with self.subTest(value=value), patch.dict(os.environ, flags):
                os.environ.pop(FLAG, None)
                if value is not None:
                    os.environ[FLAG] = value
                owner = NarrowPhase(
                    max_candidate_pairs=4,
                    max_triangle_pairs=8,
                    has_meshes=False,
                    has_heightfields=True,
                    device="cpu",
                )
                expected = create_export_reduced_contacts_kernel(write_contact_simple)
                self.assertEqual(owner.export_reduced_contacts_kernel.key, expected.key)
                self.assertFalse(owner._heightfield_pair_csr)
                self.assertFalse(owner._heightfield_pair_csr_shell)


if __name__ == "__main__":
    unittest.main()

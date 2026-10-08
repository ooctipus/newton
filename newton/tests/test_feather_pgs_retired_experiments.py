# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject removed research selectors before constructing a replacement owner."""

import os
import unittest
from unittest.mock import patch

import newton
from newton._src.solvers.feather_pgs import sparse_factor

RETIRED_FLAGS = (
    "FEATHER_PGS_SINGLE_FACTOR",
    "FEATHER_PGS_BODY_BASIS_ROWS",
    "FEATHER_PGS_SPARSE_PACKETS",
    "FEATHER_PGS_SPARSE_PAIRED_GS",
    "FEATHER_PGS_COUPLED_CONTACT",
    "FEATHER_PGS_COUPLED_JACOBI",
)


class TestFeatherPGSRetiredExperiments(unittest.TestCase):
    """Keep an obsolete recipe from silently selecting a different solver."""

    def test_solver_rejects_removed_flags_before_model_access(self):
        """Reject each retired selector before allocating or reading model data."""
        for flag in RETIRED_FLAGS:
            flags = dict.fromkeys(RETIRED_FLAGS, "0") | {flag: "1"}
            with self.subTest(flag=flag), patch.dict(os.environ, flags):
                with self.assertRaisesRegex(ValueError, flag + ".*retired"):
                    newton.solvers.SolverFeatherPGS(None)

    def test_sparse_owner_rejects_removed_flags_before_admission(self):
        """Reject retired recipes even when sparse admission would otherwise fail."""
        for flag in RETIRED_FLAGS:
            flags = dict.fromkeys(RETIRED_FLAGS, "0") | {flag: "1"}
            with self.subTest(flag=flag), patch.dict(os.environ, flags):
                with self.assertRaisesRegex(ValueError, flag + ".*retired"):
                    sparse_factor.create_owner(None)
                with self.assertRaisesRegex(ValueError, flag + ".*retired"):
                    sparse_factor.SparseFactor(None, None, None)


if __name__ == "__main__":
    unittest.main()

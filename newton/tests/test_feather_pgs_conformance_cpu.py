# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The conformance catalogue on the CPU.

Each problem of :func:`.test_feather_pgs_conformance.cpu_placements` (the CUDA matrix) runs every route of it, each
case's body at one lane (the Jacobi tiers' as virtual lanes), and passes the conformance module's checks against the CPU reference; the
incline also slides steadily at every budget. A module of its own, so that the CPU suite runs it beside the
conformance module.
"""

import unittest

import warp as wp

from newton.tests import test_feather_pgs_conformance as conformance


class TestFeatherPGSConformanceCPU(conformance._ConformanceCase):
    """The CUDA matrix's routes on the CPU against the converged CPU reference."""

    device = wp.get_device("cpu")


conformance._add_group_tests(TestFeatherPGSConformanceCPU, conformance.cpu_placements())
conformance.add_incline_budget_tests(TestFeatherPGSConformanceCPU, conformance.CPU)


if __name__ == "__main__":
    unittest.main(verbosity=2)

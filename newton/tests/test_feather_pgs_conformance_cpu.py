# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The conformance catalogue's CUDA problems on the CPU's matrix-free formulation.

Each problem of :func:`.test_feather_pgs_conformance.cpu_matrix_free_groups` runs the held tree and the general row
on the CPU (the test pin ``formulation="matrix_free"``) and passes the conformance module's checks against the CPU
reference. A module of its own, so that the CPU suite runs it beside the conformance module.
"""

import unittest

import warp as wp

from newton.tests import test_feather_pgs_conformance as conformance


class TestFeatherPGSConformanceCPUMatrixFree(conformance._ConformanceCase):
    """The CUDA problems through the CPU's matrix-free formulation against the CPU reference."""

    device = wp.get_device("cpu")


conformance._add_group_tests(TestFeatherPGSConformanceCPUMatrixFree, conformance.cpu_matrix_free_groups())


if __name__ == "__main__":
    unittest.main(verbosity=2)

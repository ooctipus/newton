# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard how the held tree's warp factor build reads its geometry record."""

import inspect
import unittest

from newton._src.solvers.feather_pgs.response_builds_dense import factor_warps_kernel


def native(kernel, name="factor_native"):
    """Read the actual generated native definition, not a second implementation."""
    variables = inspect.getclosurevars(kernel.func)
    function = variables.nonlocals.get(name) or variables.globals[name]
    return function.native_snippet


class TestGeometricFactor(unittest.TestCase):
    def test_indexed_record_reads_its_row(self):
        """A group reads its own record row; a forest family its block of a wider row through the group-to-row map."""
        own = native(factor_warps_kernel(18, tuple(range(324))))
        self.assertIn("geometric.data[group * 324 + geometry_entry]", own)
        self.assertIn("const int element = lower_schedule.data[schedule_index];", own)
        family = native(factor_warps_kernel(3, tuple(range(9)), geometric_stride=20, geometric_group=True))
        self.assertIn("geometric.data[geometry_group.data[group] * 20 + geometry_entry]", family)
        self.assertIn("geometry_lookup.data[element]", family)


if __name__ == "__main__":
    unittest.main()

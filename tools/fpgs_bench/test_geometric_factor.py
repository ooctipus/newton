# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard the assembly-only seam between the warp factor build's geometric and composite variants."""

import inspect
import unittest

from newton._src.solvers.feather_pgs.response_builds_dense import factor_warps_kernel


def native(kernel, name="factor_native"):
    """Read the actual generated native definition, not a second implementation."""
    variables = inspect.getclosurevars(kernel.func)
    function = variables.nonlocals.get(name) or variables.globals[name]
    return function.native_snippet


class TestGeometricFactor(unittest.TestCase):
    def test_only_assembly_changes(self):
        """R/K, factor recurrence, guards and final L writes are the composite variant's source."""
        old = native(factor_warps_kernel(9, warps_per_block=4))
        new = native(factor_warps_kernel(9, geometric=True))
        tail = "        if (row == col)"
        self.assertEqual(old[old.index(tail) :], new[new.index(tail) :])
        self.assertIn("float value = geometric.data[group * 81 + element];", new)
        self.assertNotIn("body_I_c.data[", new)
        self.assertNotIn("forces[", new)

    def test_original_abi_is_unchanged(self):
        """The geometric entry adds its record before the L output; the composite ABI has none."""
        old = factor_warps_kernel(9, warps_per_block=4)
        new = factor_warps_kernel(9, geometric=True)
        old_args, new_args = list(inspect.signature(old.func).parameters), list(inspect.signature(new.func).parameters)
        self.assertEqual(new_args, [*old_args[:-1], "geometric", "L_group"])
        free = factor_warps_kernel(6, warps_per_block=4)
        self.assertNotIn("geometric", inspect.signature(free.func).parameters)
        self.assertIn("body_I_c.data[", native(free))

    def test_indexed_record_reads_its_row(self):
        """An indexed family reads its block of a wider record through the group-to-row map."""
        index = tuple(range(9))
        new = native(
            factor_warps_kernel(3, geometric=True, geometric_index=index, geometric_stride=20, geometric_group=True),
            "factor_native",
        )
        self.assertIn("geometric.data[geometry_group.data[group] * 20 + geometry_entry]", new)
        self.assertIn("geometry_lookup.data[element]", new)
        wide = native(factor_warps_kernel(18, geometric=True, geometric_index=tuple(range(324))), "factor_native")
        self.assertIn("const int element = packed & 1023;", wide)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard the assembly-only seam to the original primary factor arithmetic."""

import inspect
import unittest
from unittest.mock import patch

from newton._src.solvers.feather_pgs import franka_kinetic_factor as candidate
from newton._src.solvers.feather_pgs import solver_feather_pgs as original


def native(kernel):
    """Read the actual generated native definition, not a second implementation."""
    return inspect.getclosurevars(kernel.func).nonlocals["crba_cholesky_warp_native"].native_snippet


class TestFrankaKineticFactor(unittest.TestCase):
    def test_only_assembly_changes(self):
        """R/K, factor recurrence, guards and final L writes retain exact source."""
        for architecture in ("120", "100"):
            old = native(original._get_crba_cholesky_warp_kernel(9, architecture, warps_per_block=4))
            new = native(candidate.get_kernel(architecture))
            self.assertEqual(old[: old.index("    __shared__ float forces[")], new[: new.index("    float* factor =")])
            tail = "        if (row == col)"
            self.assertEqual(old[old.index(tail) :], new[new.index(tail) :])
            self.assertIn("float value = geometric.data[group * 81 + element];", new)
            self.assertNotIn("body_I_c.data[", new)
            self.assertNotIn("forces[", new)

    def test_original_abi_is_unchanged(self):
        """Only the new nine-DOF entry adds an input, before the original L output."""
        old = original._get_crba_cholesky_warp_kernel(9, "120", warps_per_block=4)
        new = candidate.get_kernel("120")
        old_args, new_args = list(inspect.signature(old.func).parameters), list(inspect.signature(new.func).parameters)
        self.assertEqual(new_args, [*old_args[:-1], "geometric", "L_group"])
        free = original._get_crba_cholesky_warp_kernel(6, "120", warps_per_block=4)
        self.assertNotIn("geometric", inspect.signature(free.func).parameters)
        self.assertIn("body_I_c.data[", native(free))

    def test_unreviewed_source_change_fails_closed(self):
        """Future original factory edits cannot silently alter this specialization."""
        with patch.object(candidate.inspect, "getsource", return_value="unreviewed original"):
            with self.assertRaisesRegex(ValueError, "Original Franka primary factor source changed"):
                candidate._factory.__wrapped__()


if __name__ == "__main__":
    unittest.main()

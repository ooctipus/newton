# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve complete queue/guard dispatch while retiring generated AST adapters."""

import ast
import hashlib
import inspect
import textwrap
import unittest

from newton._src.solvers.feather_pgs import kinetic_compact_coupled, kinetic_guard, kinetic_hybrid, mf_dispatch
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_mf_gs_kernel


class TestFeatherPGSMFDispatch(unittest.TestCase):
    """Check the shared ABI, original dispatch bodies and negative admission."""

    def test_dispatch_policies_preserve_complete_retained_ast(self):
        """Match dispatches captured from 4534ffb4 before removing extraction."""
        original = _get_pgs_solve_mf_gs_kernel(
            192,
            64,
            29,
            "cpu",
            has_drive_rows=False,
            has_dense_velocity_limit_rows=False,
            factor_coordinates=True,
            independent_components=True,
        )
        cases = (
            ("plain", original, "4073c76b80171645d46ad8b123ef61ae690a469b2fdd9a449735aaa3f8e0d553"),
            (
                "guard",
                kinetic_guard.get_general_kernel(original),
                "f0ba2e9a1986d775bf8f4f7fdf648c2023f9e2eb25e1374eaae300fba69a2d39",
            ),
            (
                "hybrid",
                kinetic_hybrid.get_general_kernel(original),
                "81de0b043d12bbb793934056de7e950aa6a0cd473487a539937bc41f737e87e4",
            ),
            (
                "compact",
                kinetic_compact_coupled.get_general_kernel(original),
                "db217adc16acfa733daa2c2aaf20ef9c1ea2414da44e0930c645fd0fc9ad2f5e",
            ),
        )
        for policy, kernel, expected in cases:
            with self.subTest(policy=policy):
                self.assertEqual(kernel.func.mf_dispatch_policy, policy)
                node = ast.parse(textwrap.dedent(inspect.getsource(kernel.func))).body[0]
                node.decorator_list = []
                actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                self.assertEqual(actual, expected)

    def test_private_operands_cannot_shadow_original_abi(self):
        """Reject duplicated operands before producing a native declaration."""
        with self.assertRaisesRegex(ValueError, "distinct"):
            mf_dispatch.make_native("", extra={"world": int})
        with self.assertRaisesRegex(ValueError, "valid name"):
            mf_dispatch.make_native("", name="bad name")

    def test_guarded_dispatch_requires_complete_bindings(self):
        """Reject incomplete owner contracts before compilation or publication."""
        with self.assertRaisesRegex(ValueError, "Unknown"):
            mf_dispatch.make_dispatch(None, policy="unknown")
        with self.assertRaisesRegex(ValueError, "no private"):
            mf_dispatch.make_dispatch(None, extra={"guard": kinetic_guard.BoundaryGuard})
        with self.assertRaisesRegex(ValueError, "boundary"):
            mf_dispatch.make_dispatch(None, policy="guard")
        with self.assertRaisesRegex(ValueError, "required owner"):
            mf_dispatch.make_dispatch(
                None,
                policy="compact",
                extra={"guard": kinetic_guard.BoundaryGuard},
                bindings={"_guard_world_check": kinetic_guard._guard_world_check},
            )


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source and descriptor controls for the opt-in first-hit representation."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import warp as wp

from newton._src.solvers.feather_pgs import kinetic_first_hit, kinetic_solve


class TestFirstHit(unittest.TestCase):
    def test_factory_api(self):
        """Expose the coupled producer and consumer without new capacity panels."""
        for name in ("get_contact_kernel", "get_prefix_kernel", "get_solve_kernel", "install"):
            self.assertTrue(callable(getattr(kinetic_first_hit, name)))

    def test_install_reuses_all_panels(self):
        """Replace both getter seams without allocating another state or panel."""
        settings = SimpleNamespace(enable_friction=1, friction_gap_threshold=float("inf"), friction_anchor_limit=0)
        rows = SimpleNamespace(
            settings=settings,
            device=SimpleNamespace(arch="cpu"),
            kernels=[None] * 4,
            plan=object(),
            held=object(),
            state=object(),
            out=object(),
            current=object(),
        )
        solve = SimpleNamespace(kernels={}, arguments={}, solve_descriptor=object())
        call = SimpleNamespace(rows=rows, solve=solve)
        with (
            patch.object(wp, "zeros", side_effect=AssertionError("Unexpected new panel")),
            patch.object(wp, "empty", side_effect=AssertionError("Unexpected new panel")),
        ):
            self.assertTrue(kinetic_first_hit.install(call))
        self.assertIs(solve.arguments["offset_eight"][-1], rows.current)
        self.assertEqual(len(solve.arguments["offset_eight"]), 6)
        self.assertTrue(call.first_hit)
        self.assertIsNone(rows.kernels[0])
        self.assertIsNone(rows.kernels[3])

    def test_unsupported_recipe_keeps_eager_before_any_write(self):
        """Keep unsupported nontriplet recipes entirely on the original representation."""
        settings = SimpleNamespace(enable_friction=0, friction_gap_threshold=float("inf"), friction_anchor_limit=0)
        rows = SimpleNamespace(settings=settings, kernels=[None] * 4)
        call = SimpleNamespace(rows=rows, solve=SimpleNamespace(kernels={}, arguments={}))
        self.assertFalse(kinetic_first_hit.install(call))
        self.assertEqual(rows.kernels, [None] * 4)
        self.assertFalse(hasattr(call, "first_hit"))

    def test_shared_impulse_transactions_survive(self):
        """Keep all reviewed read-complete fences and the original projection order."""
        source = kinetic_first_hit.cuda_source()
        recovered = kinetic_solve.recover_impulse_transactions(source)
        self.assertNotIn("IMPULSE_READ_COMPLETE", recovered)
        self.assertNotIn("SIBLING_READ_COMPLETE", recovered)
        for statement in (
            "const float magnitude = sqrtf(",
            "new_tangent1 *= scale;",
            "new_tangent2 *= scale;",
            "factor_velocity += tangent2_factor * sibling_delta;",
            "factor_velocity += tangent1_factor * sibling_delta;",
        ):
            self.assertIn(statement, source)
        self.assertLess(source.index("float tangent1_sum"), source.index("float tangent2_sum"))

    def test_first_demand_state_is_not_final_impulse(self):
        """Count cache construction at valid1-to2, independent of final lambda."""
        self.assertIn("rows.valid.data[id]=2", kinetic_first_hit._HELPERS)
        self.assertIn("rows.valid.data[id]=2", kinetic_first_hit._CPU_LAZY)
        self.assertNotIn("out.response", kinetic_first_hit._PACKET)
        self.assertNotIn("__shared__ float", kinetic_first_hit._PACKET)

    def test_cpu_stateless_velocity_limit_preserves_original_law(self):
        """Keep type4 raw-delta projection distinct from accumulated bilateral rows."""
        branch = "else if(type==4){if(residual<0.0f){delta=raw_delta;next=raw_delta;}else next=0.0f;}"
        self.assertIn(branch, kinetic_solve._CPU)
        self.assertIn(branch.replace("residual", "r"), kinetic_first_hit._CPU_LAZY)

    def test_live_hook_binds_both_new_owners(self):
        """Bind the actual live constructor to the new producer and consumer together."""
        from tools.fpgs_bench.test_kinetic_live_bindings import bound_call  # noqa: PLC0415

        with patch.dict(os.environ, {"FEATHER_PGS_KUKA_FIRST_HIT": "1"}):
            *_, call = bound_call(3)
        self.assertTrue(call.first_hit)
        self.assertIn("first_hit_packets", call.rows.kernels[2].key)
        self.assertIn("first_hit_eight", call.solve.kernels["offset_eight"].key)
        self.assertIs(call.solve.arguments["offset_eight"][-1], call.slots.current)


if __name__ == "__main__":
    unittest.main()

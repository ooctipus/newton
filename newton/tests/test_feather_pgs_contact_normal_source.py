# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The three admission owners share one normal law with explicit storage policy."""

import ast
import hashlib
import importlib
import unittest


class TestContactNormalSource(unittest.TestCase):
    def test_normal_law_uses_explicit_policies(self):
        factory = importlib.import_module("newton._src.solvers.feather_pgs.contact_normal")
        sources = {policy: factory.get_source(policy) for policy in ("scratch", "light", "current")}
        for source in sources.values():
            self.assertEqual(len(ast.parse(source).body), 2)
            self.assertNotIn("inspect.", source)
        self.assertIn("scratch.minimum_residual", sources["scratch"])
        self.assertNotIn("scratch.", sources["light"])
        self.assertIn("state.endpoint_twists[body]", sources["current"])
        self.assertNotIn("data.body_q[", sources["current"])
        with self.assertRaises(ValueError):
            factory.get_source("unknown")

    def test_body_validation_keeps_original_native_bytes(self):
        factory = importlib.import_module("newton._src.solvers.feather_pgs.body_validation")
        kinetic_zero = importlib.import_module("newton._src.solvers.feather_pgs.kinetic_zero")
        kuka_joint_world = importlib.import_module("newton._src.solvers.feather_pgs.kuka_joint_world")

        original = factory.get_source(produce_twists=True)
        current = factory.get_source(produce_twists=False)
        self.assertTrue(kuka_joint_world._TWISTS.endswith(original))
        self.assertTrue(kinetic_zero.load_source().endswith(current))
        self.assertIn("data.body_v_s", original)
        self.assertNotIn("data.body_v_s", current)
        self.assertNotIn("data.joint_S_s", current)
        for value in (original, current):
            self.assertIn("data.max_linear_velocity", value)
            self.assertIn("isfinite(s[96+b*6+k])", value)
        self.assertEqual(
            hashlib.sha256(kinetic_zero._load.native_snippet.encode()).hexdigest(),
            "99c23ee48e1ab48f71af20f42142b8bd585c6254156dbb162e39ac85d3abb609",
        )


if __name__ == "__main__":
    unittest.main()

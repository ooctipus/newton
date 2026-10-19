# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the held-tree output A/B comparison and its environment handling."""

import contextlib
import io
import os
import unittest

import numpy as np

from tools.fpgs_bench import held_tree_ab as ab


class TestHeldTreeAB(unittest.TestCase):
    def test_compare_is_bitwise_and_skips_stage_clocks(self):
        """Report arrays whose bytes, dtype or shape differ, and arrays one record lacks; skip cycle counts."""
        parent = {
            "s/0/begin/cache0.bias": np.array([0.0, 1.0], np.float32),
            "s/0/begin/cache0.status": np.array([0, 1], np.int32),
            "s/0/begin/cache0.stage_clocks": np.array([[5]], np.int64),
            "s/0/begin/v_hat": np.array([1.0, np.nan], np.float32),
            "s/0/begin/gone": np.zeros(1, np.float32),
        }
        child = {
            # Equal values, other bits: a negative zero.
            "s/0/begin/cache0.bias": np.array([-0.0, 1.0], np.float32),
            "s/0/begin/cache0.status": np.array([0, 1], np.int64),
            "s/0/begin/cache0.stage_clocks": np.array([[9]], np.int64),
            "s/0/begin/v_hat": np.array([1.0, np.nan], np.float32),
            "s/0/begin/new": np.zeros(1, np.float32),
        }
        differing, only_parent, only_child = ab.compare(parent, child)
        self.assertEqual(differing, ["s/0/begin/cache0.bias", "s/0/begin/cache0.status"])
        self.assertEqual((only_parent, only_child), (["s/0/begin/gone"], ["s/0/begin/new"]))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(ab.report(parent, child), 1)
            self.assertEqual(ab.report(parent, dict(parent)), 0)
        self.assertIn("5 shared arrays: 0 differ bitwise", output.getvalue())

    def test_environment_is_restored(self):
        """Set switches for one scene and restore the values, or the absence, they had."""
        name, other = "FEATHER_PGS_TEST_AB_SWITCH", "FEATHER_PGS_TEST_AB_OTHER"
        os.environ[other] = "7"
        try:
            with ab._environment(**{name: 1, other: 0}):
                self.assertEqual((os.environ[name], os.environ[other]), ("1", "0"))
            self.assertNotIn(name, os.environ)
            self.assertEqual(os.environ[other], "7")
        finally:
            os.environ.pop(other, None)


if __name__ == "__main__":
    unittest.main()

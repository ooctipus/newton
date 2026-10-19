# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the held-tree output A/B record, its comparison and its environment handling."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import tree_kinetic_owner
from tools.fpgs_bench import held_tree_ab as ab


class TestHeldTreeAB(unittest.TestCase):
    def test_compare_is_bitwise_and_skips_stage_clocks(self):
        """Report arrays whose bytes, dtype or shape differ, and arrays one record lacks; skip cycle counts."""
        parent = {
            "s/0/begin/cache0.bias": np.array([0.0, 1.0], np.float32),
            "s/0/begin/cache0.status": np.array([0, 1], np.int32),
            "s/0/begin/cache0.stage_clocks": np.array([[5]], np.int64),
            "s/0/begin/v_hat": np.array([1.0, np.nan], np.float32),
            "s/0/begin/launched": np.array(["tree_kinetic_repair4", "tree_kinetic_finish4"]),
            "s/0/begin/gone": np.zeros(1, np.float32),
        }
        child = {
            # Equal values, other bits: a negative zero.
            "s/0/begin/cache0.bias": np.array([-0.0, 1.0], np.float32),
            "s/0/begin/cache0.status": np.array([0, 1], np.int64),
            "s/0/begin/cache0.stage_clocks": np.array([[9]], np.int64),
            "s/0/begin/v_hat": np.array([1.0, np.nan], np.float32),
            # Another kernel launched: the comparing repair in place of the ordinary one.
            "s/0/begin/launched": np.array(["tree_kinetic_repair4_banked", "tree_kinetic_finish4"]),
            "s/0/begin/new": np.zeros(1, np.float32),
        }
        differing, only_parent, only_child = ab.compare(parent, child)
        self.assertEqual(differing, ["s/0/begin/cache0.bias", "s/0/begin/cache0.status", "s/0/begin/launched"])
        self.assertEqual((only_parent, only_child), (["s/0/begin/gone"], ["s/0/begin/new"]))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(ab.report(parent, child), 1)
            self.assertEqual(ab.report(parent, dict(parent)), 0)
        self.assertIn(
            "s/0/begin/launched: ['tree_kinetic_repair4'] against ['tree_kinetic_repair4_banked']", output.getvalue()
        )
        self.assertIn("6 shared arrays: 0 differ bitwise", output.getvalue())

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

    def test_record_names_the_kernels_each_copy_back_step_launched(self):
        """Record the comparing one-pass repair in the first copy-back step and the classified repair in the second.

        The second step forces classification on a one-block grid: it lists the
        blocks (classification), compares them in a pass of its own and repairs
        over the list. One mixed scene on CPU; the record also holds its layouts.
        """
        environment = {name: value for name, value in os.environ.items() if not name.startswith("FEATHER_PGS_")}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            records = ab.record(Path(directory) / "record.npz", scenes=("arm-box",))
            self.assertTrue((Path(directory) / "record.npz").exists())
        self.assertEqual({name.split("/")[0] for name in records}, {"mixed-arm-box"})
        layout = records["mixed-arm-box/layout"]
        # A free box and a four-body arm, one tree per block, both comparing banks.
        self.assertEqual(layout[:, [0, 2, 5]].tolist(), [[1, 1, 1], [4, 1, 1]])
        copied = records["mixed-arm-box/copied/begin/launched"].tolist()
        classified = records["mixed-arm-box/copied-classified/begin/launched"].tolist()
        for bodies in layout[:, 0].tolist():
            with self.subTest(bodies=bodies):
                self.assertIn(f"tree_kinetic_repair{bodies}_banked", copied)
                self.assertNotIn(f"tree_kinetic_classify{bodies}", copied)
                for kernel in ("classify{}", "bank{}", "repair{}_listed"):
                    self.assertIn("tree_kinetic_" + kernel.format(bodies), classified)
                self.assertNotIn(f"tree_kinetic_repair{bodies}_banked", classified)
        # Every call records its launches; the hooks are gone afterwards.
        self.assertIn("tree_kinetic_finish4", records["mixed-arm-box/0/finish/launched"].tolist())
        self.assertEqual(tree_kinetic_owner.TreeKineticState._publication.__name__, "_publication")
        self.assertEqual((wp.launch.__name__, wp.launch_tiled.__name__), ("launch", "launch_tiled"))

    def test_worlds_set_the_mixed_scenes_world_count(self):
        """Build the mixed scenes over the requested world count; the single trees keep their fixtures' two."""
        environment = {name: value for name, value in os.environ.items() if not name.startswith("FEATHER_PGS_")}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            records = ab.record(
                Path(directory) / "record.npz", worlds=3, scenes=("short-arm-box", "single-floating0-chains0")
            )
        # The short arm's three-body tree beside the free box, one of each per world.
        self.assertEqual(records["mixed-short-arm-box/layout"][:, 0].tolist(), [1, 3])
        for index in range(2):
            self.assertEqual(records[f"mixed-short-arm-box/0/finish/cache{index}.status"].shape, (3,))
        self.assertEqual(records["single-floating0-chains0/0/finish/cache0.status"].shape, (2,))


if __name__ == "__main__":
    unittest.main()

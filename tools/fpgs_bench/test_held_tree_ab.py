# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the held-tree output A/B record, its comparison and its environment handling."""

import contextlib
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import tree_kinetic_owner
from newton.tests.test_feather_pgs_cache_validity import _input_mismatches
from tools.fpgs_bench import held_tree_ab as ab

# The owners the per-size groups of test_feather_pgs_cache_validity's launch check leave out, by scene:
# (forest, chain scan, alternate motion) of the scene's one group.
_OWNER_SCENES = {
    "single-floating1-chains0": (False, False, False),
    "prescaled-floating1-chains0": (False, False, True),
    "single-floating1-chains1": (False, True, False),
    "prescaled-floating1-chains1": (False, True, True),
    "forest-arm-box": (True, False, False),
    "forest-speed-limit": (True, False, True),
}


def _form(key):
    """The launch form a held-tree kernel key names: its stage, or a repair's form."""
    stage = re.match(r"tree_kinetic_([a-z]+)\d", key).group(1)
    if stage != "repair":
        return stage
    return next((form for form in ("banked", "listed", "motion") if key.endswith(f"_{form}")), "repair")


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

        def launched(names, kernel, bodies, suffix=""):
            """Whether ``names`` holds the ``kernel`` of ``bodies`` bodies, its scans' schedule tag aside."""
            pattern = re.compile(rf"tree_kinetic_{kernel}{bodies}(_[a-z_]+?)?{suffix}")
            return any(pattern.fullmatch(name) for name in names)

        for bodies in layout[:, 0].tolist():
            with self.subTest(bodies=bodies):
                self.assertTrue(launched(copied, "repair", bodies, "_banked"), copied)
                self.assertFalse(launched(copied, "classify", bodies), copied)
                for kernel, suffix in (("classify", ""), ("bank", ""), ("repair", "_listed")):
                    self.assertTrue(launched(classified, kernel, bodies, suffix), classified)
                self.assertFalse(launched(classified, "repair", bodies, "_banked"), classified)
        # Every call records its launches; the hooks are gone afterwards.
        self.assertTrue(launched(records["mixed-arm-box/0/finish/launched"].tolist(), "finish", 4))
        self.assertEqual(tree_kinetic_owner.TreeKineticState._publication.__name__, "_publication")
        self.assertEqual((wp.launch.__name__, wp.launch_tiled.__name__), ("launch", "launch_tiled"))

    def test_single_tree_and_forest_launches_match_their_kernels(self):
        """Launch every held-tree kernel of sparse single trees and forests with one input of each parameter's type.

        Warp checks an array input at launch but packs whatever it is given for a struct
        parameter, so a kinetic plan in a chain-scan plan's place would launch. The scenes cover the
        owners that the per-size groups of test_feather_pgs_cache_validity's launch check leave out:
        a sparse single tree on a kinetic and on a chain-scan plan and a forest group, each with
        ordinary and alternate motion. Each launch is checked before it runs, and each scene runs
        every form its group binds: the finish and the predictor; for ordinary motion the one-pass,
        comparing and listed repairs, the classification and the comparison pass; for alternate
        motion its one-pass motion repair.
        """
        checked = []

        def checking(launch):
            def launch_checked(kernel, *args, **kwargs):
                if kernel.key.startswith("tree_kinetic_"):
                    self.assertEqual(_input_mismatches(kernel, kwargs.get("inputs", ())), [], kernel.key)
                    checked.append(kernel.key)
                return launch(kernel, *args, **kwargs)

            return launch_checked

        environment = {name: value for name, value in os.environ.items() if not name.startswith("FEATHER_PGS_")}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
            patch.multiple(wp, launch=checking(wp.launch), launch_tiled=checking(wp.launch_tiled)),
        ):
            records = ab.record(Path(directory) / "record.npz", scenes=tuple(_OWNER_SCENES))
        ordinary = {"finish", "predict", "repair", "banked", "listed", "classify", "bank"}
        for scene, (forest, chains, motion) in _OWNER_SCENES.items():
            tag = scene if scene.startswith(("single-", "prescaled-")) else f"mixed-{scene}"
            with self.subTest(scene=tag):
                layout = records[f"{tag}/layout"]
                self.assertEqual(layout[:, [3, 6]].tolist(), [[forest, motion]])
                # A chains1 scene runs every scan on the heavy chains; otherwise the tree's data chooses.
                if chains:
                    self.assertEqual(layout[:, 4].tolist(), [1])
                launched = {
                    key
                    for name, keys in records.items()
                    if name.startswith(f"{tag}/") and name.endswith("/launched")
                    for key in keys.tolist()
                }
                self.assertLessEqual(launched, set(checked))
                self.assertEqual(
                    {_form(key) for key in launched}, {"finish", "predict", "motion"} if motion else ordinary
                )

    def test_mixed_scenes_request_no_scalar_branches(self):
        """Build every mixed scene's solver with its diagonal-mass group pinned to the held tree (``star="held"``), so
        that its owner composes tree groups only.

        A scalar-branch group holds no tree layout or cache data, so a record could not describe or
        drive it; ``check_owner_launches`` of the scalar-branch tests checks its launches instead.
        """
        from newton._src.solvers.feather_pgs import scalar_branches  # noqa: PLC0415

        plan = Mock(wraps=scalar_branches.plan_scalar_branches)
        environment = {name: value for name, value in os.environ.items() if not name.startswith("FEATHER_PGS_")}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, environment, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(scalar_branches, "plan_scalar_branches", plan),
        ):
            records = ab.record(Path(directory) / "record.npz", scenes=("cabinet", "forest-cabinet"))
        self.assertEqual([call.args[0]._route_pins.get("star") for call in plan.call_args_list], ["held"] * 2)
        self.assertEqual({name.split("/")[0] for name in records}, {"mixed-cabinet", "mixed-forest-cabinet"})

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

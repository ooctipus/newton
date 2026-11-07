# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise convex terrain scheduling with the retained support and contact law."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.heightfield_test_utils import Case, build_scene
from newton.tests.test_convex_cells import FACES, VERTICES
from newton.tests.test_heightfield_convex_cells import pipeline
from newton.tests.test_heightfield_primitive_query import TestHeightfieldPrimitiveQuery


def make_scene(device, **options):
    with patch(
        "newton.tests.heightfield_test_utils.cube_mesh",
        lambda: newton.Mesh(VERTICES + np.array([0.02, -0.01, 0.015], np.float32), FACES.ravel()),
    ):
        return build_scene(
            Case("offset_hull", "support", hull=True, **options), False, device, make_solver=False, hull_scale=4.0
        )


def split_pipeline(model, enabled, *, automatic=False):
    with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CONVEX_SPLIT": str(int(enabled))}):
        return pipeline(model, True, automatic=automatic)


class TestHeightfieldConvexSplit(unittest.TestCase):
    def test_admission_cpu(self):
        """Expose the bounded three-pass owner only when explicitly requested."""
        scene = make_scene("cpu")
        actual, ordinary = split_pipeline(scene.model, True), split_pipeline(scene.model, False)
        self.assertEqual(len(actual.narrow_phase._heightfield_convex_split_kernels), 3)
        self.assertFalse(ordinary.narrow_phase._heightfield_convex_split_kernels)
        self.assertEqual(actual.get_collision_selection()["heightfield_support"]["query_schedule"], "split")
        self.assertEqual(ordinary.get_collision_selection()["heightfield_support"]["query_schedule"], "fused")
        self.assertEqual(
            len(actual.narrow_phase._heightfield_convex_query_results), len(actual.narrow_phase.triangle_pairs)
        )

    def check_contacts(self, device):
        checker = TestHeightfieldPrimitiveQuery()
        for options in ({}, {"tilt": 0.08}, {"terrain_angle": 0.15}, {"x": 1.05}, {"gap": -0.01}):
            with self.subTest(device=device, options=options):
                scene = make_scene(device, **options)
                state = scene.states[0]
                reference, actual = split_pipeline(scene.model, False), split_pipeline(scene.model, True)
                before, after = reference.contacts(), actual.contacts()
                reference.collide(state, before)
                actual.collide(state, after)
                checker.assert_current_contacts(reference, actual, state, before, after)
                if wp.get_device(device).is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        actual.collide(state, after)
                    wp.capture_launch(capture.graph)
                    checker.assert_current_contacts(reference, actual, state, before, after)
                reference.invalidate_convex_cells()
                actual.invalidate_convex_cells()
                reference.collide(state, before)
                if wp.get_device(device).is_cuda:
                    wp.capture_launch(capture.graph)
                else:
                    actual.collide(state, after)
                checker.assert_current_contacts(reference, actual, state, before, after)

    def test_contacts_cpu(self):
        """Preserve offset-hull contacts across separated and penetrating queries."""
        self.check_contacts("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_contacts_cuda(self):
        """Preserve contacts, graph replay and invalidation on the CUDA owner."""
        self.check_contacts("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_conditional_graph_cuda(self):
        """Keep both support branches and empty-to-loaded graph transitions valid."""
        scene = make_scene("cuda:0", tilt=0.08)
        reference, actual = (split_pipeline(scene.model, split, automatic=True) for split in (False, True))
        before, after = reference.contacts(), actual.contacts()
        state = scene.states[0]
        checker = TestHeightfieldPrimitiveQuery()
        reference.collide(state, before)
        actual.collide(state, after)
        with wp.ScopedCapture(device="cuda:0") as capture:
            actual.collide(state, after)
        poses = state.body_q.numpy().copy()
        for empty, cells in ((False, 0), (False, 1), (True, 1), (False, 1), (False, 0)):
            current = poses.copy()
            if empty:
                current[:, 2] += 10.0
            state.body_q.assign(current)
            for owner in (reference, actual):
                choice = owner.narrow_phase._heightfield_method_choice
                choice.epoch.assign(np.array([1], np.int32))
                choice.choice.assign(np.array([cells], np.int32))
            reference.collide(state, before)
            wp.capture_launch(capture.graph)
            if empty:
                self.assertFalse(checker.raw_contacts(reference))
                self.assertFalse(checker.raw_contacts(actual))
                self.assertEqual(int(actual.narrow_phase._heightfield_convex_gjk_count.numpy()[0]), 0)
            else:
                checker.assert_current_contacts(reference, actual, state, before, after)
        reference.invalidate_convex_cells()
        actual.invalidate_convex_cells()
        reference.collide(state, before)
        wp.capture_launch(capture.graph)
        checker.assert_current_contacts(reference, actual, state, before, after)


if __name__ == "__main__":
    unittest.main()

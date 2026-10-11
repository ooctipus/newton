# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the same terrain contact law with immutable convex support cells."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.heightfield_test_utils import Case, build_scene
from newton.tests.test_convex_cells import FACES, VERTICES
from newton.tests.test_heightfield_primitive_query import FLAGS, TestHeightfieldPrimitiveQuery


def pipeline(model, enabled, *, automatic=False):
    with patch.dict(
        os.environ,
        {
            **FLAGS,
            "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0",
            "NEWTON_HEIGHTFIELD_CONVEX_CELLS": "0",
        },
    ):
        return newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=4096,
            max_triangle_pairs=4096,
            convex_collision="auto" if automatic else "convex_cells" if enabled else "ordinary",
        )


class TestHeightfieldConvexCells(unittest.TestCase):
    def test_public_admission_cpu(self):
        """Force supported terrain cells without environment configuration."""
        builder = newton.ModelBuilder()
        builder.add_shape_heightfield(
            heightfield=newton.Heightfield(data=np.zeros((5, 5), np.float32), nrow=5, ncol=5, hx=0.4, hy=0.4)
        )
        body = builder.add_body(xform=wp.transform((0, 0, 0.25), wp.quat_identity()))
        builder.add_shape_convex_hull(body, mesh=newton.Mesh(VERTICES, FACES.ravel()))
        model = builder.finalize(device="cpu")
        actual, ordinary = pipeline(model, True), pipeline(model, False)
        self.assertEqual(actual.get_collision_selection()["method"], "convex_cells")
        self.assertEqual(ordinary.get_collision_selection()["method"], "ordinary")
        self.assertTrue(actual.narrow_phase._heightfield_primitive_query)
        with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0"}):
            for options in ({"reduce_contacts": False}, {"deterministic": True}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    newton.CollisionPipeline(model, convex_collision="convex_cells", **options)

    def check_contacts(self, device):
        checker = TestHeightfieldPrimitiveQuery()
        for options in ({}, {"tilt": 0.08}, {"terrain_angle": 0.15}, {"x": 1.05}, {"gap": -0.01}):
            with self.subTest(device=device, options=options):
                case = Case("hull", "support", hull=True, **options)
                # The retained table admission requires compact mesh coordinates;
                # preserve the authored geometry and AABBs through builder scale.
                with (
                    patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CONVEX_CELLS": "0"}),
                    patch(
                        "newton.tests.heightfield_test_utils.cube_mesh",
                        lambda: newton.Mesh(VERTICES + np.array([0.02, -0.01, 0.015], np.float32), FACES.ravel()),
                    ),
                ):
                    scene = build_scene(case, False, device, make_solver=False, hull_scale=4.0)
                state = scene.states[0]
                reference, actual = pipeline(scene.model, False), pipeline(scene.model, True)
                self.assertIsNotNone(actual.narrow_phase._heightfield_cell_owner)
                before, after = reference.contacts(), actual.contacts()
                reference.collide(state, before)
                actual.collide(state, after)
                checker.assert_current_contacts(reference, actual, state, before, after)
                if wp.get_device(device).is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        actual.collide(state, after)
                    wp.capture_launch(capture.graph)
                    checker.assert_current_contacts(reference, actual, state, before, after)
                actual.invalidate_convex_cells()
                self.assertTrue(actual._heightfield_convex_cells.metadata["invalidated"])
                if wp.get_device(device).is_cuda:
                    wp.capture_launch(capture.graph)
                else:
                    actual.collide(state, after)
                checker.assert_current_contacts(reference, actual, state, before, after)

    def test_contacts_cpu(self):
        self.check_contacts("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_contacts_cuda(self):
        self.check_contacts("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_automatic_graph_choices_cuda(self):
        """Switch actual graph support branches and withdraw invalidated geometry."""
        with (
            patch.dict(
                os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_CONVEX_CELLS": "0", "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0"}
            ),
            patch("newton.tests.heightfield_test_utils.cube_mesh", lambda: newton.Mesh(VERTICES, FACES.ravel())),
        ):
            scene = build_scene(
                Case("wide_hull", "support", hull=True),
                False,
                "cuda:0",
                make_solver=False,
                hull_scale=4.0,
                terrain_resolution=65,
            )
            actual = newton.CollisionPipeline(
                scene.model,
                convex_collision="auto",
                reduce_contacts=True,
                rigid_contact_max=4096,
                max_triangle_pairs=4096,
            )
        reference = pipeline(scene.model, False, automatic=True)
        before, after = reference.contacts(), actual.contacts()
        state = scene.states[0]
        choice = actual.narrow_phase._heightfield_method_choice
        self.assertIsNotNone(choice)
        checker = TestHeightfieldPrimitiveQuery()
        # Deliberately separated synthetic costs test both decisions independently
        # of hardware calibration noise; scene timing never trains the selector.
        cheap = np.array([0, 100, 0, 0, 0, 0, 0, 0, 0, 0], np.float32)
        expensive = np.array([0, 0, 100, 100, 100, 0, 0, 0, 0, 0], np.float32)
        # Compare both support branches within the same compiled query owner.
        # The general query can choose another coplanar witness on this exact
        # flat grid; the separate offset/tilted controls compare those owners.
        reference.narrow_phase._heightfield_method_choice.costs.assign(expensive)
        reference.collide(state, before)
        choice.costs.assign(cheap)
        actual.collide(state, after)
        self.assertGreaterEqual(choice.describe()["eligible_samples"], 32)
        self.assertEqual(int(choice.choice.numpy()[0]), 1)
        checker.assert_current_contacts(reference, actual, state, before, after)
        with wp.ScopedCapture(device="cuda:0") as capture:
            actual.collide(state, after)
        for costs, expected in ((expensive, 0), (cheap, 1)):
            choice.costs.assign(costs)
            choice.epoch.zero_()
            wp.capture_launch(capture.graph)
            self.assertEqual(int(choice.choice.numpy()[0]), expected)
            checker.assert_current_contacts(reference, actual, state, before, after)
        actual.invalidate_convex_cells()
        wp.capture_launch(capture.graph)
        self.assertEqual(int(choice.choice.numpy()[0]), 0)
        checker.assert_current_contacts(reference, actual, state, before, after)


if __name__ == "__main__":
    unittest.main()

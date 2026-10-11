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


def pipeline(model, enabled):
    with patch.dict(
        os.environ,
        {
            **FLAGS,
            "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
            "NEWTON_HEIGHTFIELD_CONVEX_CELLS": str(int(enabled)),
        },
    ):
        return newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=4096, max_triangle_pairs=4096)


class TestHeightfieldConvexCells(unittest.TestCase):
    def check_contacts(self, device):
        checker = TestHeightfieldPrimitiveQuery()
        for options in ({}, {"tilt": 0.08}, {"terrain_angle": 0.15}, {"x": 1.05}, {"gap": -0.01}):
            with self.subTest(device=device, options=options):
                case = Case("hull", "support", hull=True, **options)
                # The retained table admission requires compact mesh coordinates;
                # preserve the authored geometry through the public live scale.
                with (
                    patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CONVEX_CELLS": "0"}),
                    patch(
                        "newton.tests.heightfield_test_utils.cube_mesh",
                        lambda: newton.Mesh(VERTICES + np.array([0.02, -0.01, 0.015], np.float32), FACES.ravel()),
                    ),
                ):
                    scene = build_scene(case, False, device, make_solver=False)
                scale = scene.model.shape_scale.numpy()
                scale[1] *= 4.0
                scene.model.shape_scale.assign(scale)
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


if __name__ == "__main__":
    unittest.main()

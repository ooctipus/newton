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


def _hull_box_capsule_model():
    """A flat heightfield under an admitted cube hull, a box and a lying capsule, each resting 5 mm deep."""
    builder = newton.ModelBuilder()
    builder.add_shape_heightfield(
        heightfield=newton.Heightfield(data=np.zeros((17, 17), np.float32), nrow=17, ncol=17, hx=3.0, hy=3.0)
    )
    lying = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)
    labels = {}
    for kind, x in (("hull", -1.5), ("box", 0.0), ("capsule", 1.5)):
        body = builder.add_body(xform=wp.transform((x, 0.3, 0.245), wp.quat_identity()))
        if kind == "hull":
            shape = builder.add_shape_convex_hull(body, mesh=newton.Mesh(VERTICES, FACES.ravel()))
        elif kind == "box":
            shape = builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
        else:
            shape = builder.add_shape_capsule(body, xform=wp.transform(wp.vec3(), lying), radius=0.25, half_height=0.25)
        labels[shape] = kind
    return builder.finalize(device="cpu"), labels


def _contacts_by_shape(model, labels, convex_collision, environment):
    """Collide once on the terrain route (``NEWTON_TERRAIN_LAW=none``, else Newton's defaults) plus ``environment``;
    return each shape's sorted contact points and normals."""
    with patch.dict(os.environ):
        for name in [name for name in os.environ if name.startswith("NEWTON_")]:
            del os.environ[name]
        os.environ.update({"NEWTON_TERRAIN_LAW": "none", **environment})
        collider = newton.CollisionPipeline(
            model, rigid_contact_max=4096, max_triangle_pairs=16384, convex_collision=convex_collision
        )
    contacts = collider.contacts()
    collider.collide(model.state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    shapes = contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]
    rows = np.concatenate(
        [
            contacts.rigid_contact_point0.numpy()[:count],
            contacts.rigid_contact_point1.numpy()[:count],
            contacts.rigid_contact_normal.numpy()[:count],
        ],
        axis=1,
    )
    found = {}
    for shape, kind in labels.items():
        mine = rows[(shapes[0] == shape) | (shapes[1] == shape)]
        found[kind] = mine[np.lexsort(mine.T[::-1])]
    return collider, found


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

    def test_cells_owner_keeps_typed_primitive_rows_cpu(self):
        """Keep box and capsule terrain contacts when the cells owner takes the hulls out of the fallback scan."""
        model, labels = _hull_box_capsule_model()
        typed_table = {"NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"}
        _, reference = _contacts_by_shape(model, labels, "ordinary", typed_table)
        for convex_collision, environment in (
            ("convex_cells", {}),
            (None, {**typed_table, "NEWTON_HEIGHTFIELD_CONVEX_CELLS": "1"}),
        ):
            with self.subTest(convex_collision=convex_collision):
                collider, actual = _contacts_by_shape(model, labels, convex_collision, environment)
                self.assertIsNotNone(collider.narrow_phase._heightfield_cell_owner)
                self.assertGreater(len(actual["hull"]), 0)
                for kind in ("box", "capsule"):
                    self.assertGreater(len(reference[kind]), 0, kind)
                    # The typed rows are the same cached kernels in both pipelines.
                    np.testing.assert_array_equal(actual[kind], reference[kind], err_msg=kind)
                rows = collider.get_collision_selection()["heightfield"]["primitive_query"]["rows"]
                self.assertEqual([row["row"] for row in rows], ["sphere", "cylinder", "capsule", "box", "fallback"])
                self.assertEqual(rows[-1]["hull"], "excluded")

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

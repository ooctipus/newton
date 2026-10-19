# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-world pair templates that ModelBuilder keeps next to the explicit pair list it replays them into."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.geometry.broad_phase_nxn import ShapePairTemplates, replay_shape_pair_templates


def _robot(variant=0):
    """A small source: overlapping bodies, a kinematic body, two collision groups, a visual shape,
    a static shape inside the world and filters between neighbours."""
    robot = newton.ModelBuilder()
    second_group = newton.ModelBuilder.ShapeConfig(collision_group=2)
    visual = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    previous = None
    for i in range(4 + variant):
        body = robot.add_body(
            xform=wp.transform((0.3 * i, 0.0, 0.25 + 0.05 * i), wp.quat_identity()), is_kinematic=i == 1
        )
        box = robot.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
        robot.add_shape_sphere(body, radius=0.1, cfg=visual)
        if i % 2:
            other = robot.add_shape_capsule(body, radius=0.08, half_height=0.1, cfg=second_group)
        else:
            other = robot.add_shape_sphere(body, radius=0.18)
        if previous is not None:
            robot.add_shape_collision_filter_pair(previous, box)
        previous = other
    robot.add_shape_box(-1, xform=wp.transform((0.5, 0.6, 0.2), wp.quat_identity()), hx=0.4, hy=0.2, hz=0.2)
    return robot


def _replicated_builder(worlds=11, *, global_pair=True, residual=True, mixed=True, empty=True):
    builder = newton.ModelBuilder()
    ground = builder.add_ground_plane()
    if global_pair:
        # A global kinematic box that touches the ground: one global-global pair.
        body = builder.add_body(xform=wp.transform((0.0, 0.0, -0.4), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(body, hx=30.0, hy=3.0, hz=0.5)
    robot, other = _robot(), _robot(1)
    first_shape = []
    for world in range(worlds):
        first_shape.append(builder.shape_count)
        xform = wp.transform((2.5 * world, 0.0, 0.0), wp.quat_identity())
        if empty and world == 3:
            # A world without shapes, and below one whose only shape does not collide.
            shapeless = newton.ModelBuilder()
            shapeless.add_body()
            builder.add_world(shapeless, xform=xform)
        elif empty and world == 7:
            visual = newton.ModelBuilder()
            visual.add_shape_sphere(
                visual.add_body(), radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
            )
            builder.add_world(visual, xform=xform)
        elif mixed and world % 4 == 2:
            builder.add_world(other, xform=xform)
        else:
            builder.add_world(robot, xform=xform)
    if residual:
        # World-specific filters give worlds 1 and 5 a template of their own.
        for world in (1, 5):
            builder.add_shape_collision_filter_pair(ground, first_shape[world])
    builder.first_shape = first_shape
    return builder


def _finalize(builder, device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return builder.finalize(device=device)


class TestShapePairTemplateRecord(unittest.TestCase):
    def test_record_replays_the_model_pairs(self):
        """Keep a record whose replay equals the model's explicit pairs for each builder layout."""
        layouts = {
            "replicate": {"global_pair": False, "residual": False, "mixed": False, "empty": False},
            "global pair": {"global_pair": True, "residual": False, "mixed": False, "empty": False},
            "residual filters": {"global_pair": True, "residual": True, "mixed": False, "empty": False},
            "mixed worlds": {"global_pair": True, "residual": True, "mixed": True, "empty": True},
        }
        for name, options in layouts.items():
            with self.subTest(layout=name):
                model = _finalize(_replicated_builder(**options), "cpu")
                record = model._shape_contact_pair_templates
                self.assertIsInstance(record, ShapePairTemplates)
                self.assertTrue(record.describes(model.shape_contact_pairs))
                np.testing.assert_array_equal(record.replay(), model.shape_contact_pairs.numpy())
                self.assertEqual(record.global_pairs.shape[0], int(options["global_pair"]))
        model = _finalize(_replicated_builder(), "cpu")
        record = model._shape_contact_pair_templates
        # Robot, variant, filtered and pair-free worlds use four templates.
        self.assertEqual(len(record.templates), 4)
        self.assertGreater(len(record.runs), len(record.templates))

    def test_replicate_uses_one_template(self):
        """Record one template and one run for replicated worlds without world-specific filters."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(_robot(), 6)
        model = _finalize(builder, "cpu")
        record = model._shape_contact_pair_templates
        self.assertEqual(len(record.templates), 1)
        self.assertEqual(len(record.runs), 1)
        np.testing.assert_array_equal(record.runs[0][1], model.shape_world_start.numpy()[:6])
        np.testing.assert_array_equal(record.replay(), model.shape_contact_pairs.numpy())

    def test_general_pair_lists_drop_the_record(self):
        """Keep no record for lists the builder does not replay from templates."""
        builder = _replicated_builder()
        # The public list form of the filters disables the template path.
        builder.shape_collision_filter_pairs.append((0, builder.first_shape[2]))
        self.assertIsNone(_finalize(builder, "cpu")._shape_contact_pair_templates)

        builder = _replicated_builder()
        model = _finalize(builder, "cpu")
        self.assertIsNotNone(model._shape_contact_pair_templates)
        with self.assertWarns(DeprecationWarning):
            builder.find_shape_contact_pairs(model)
        self.assertIsNone(model._shape_contact_pair_templates)

    def test_record_identity_is_the_bound_array(self):
        """Describe only the array the replay was bound to, not a copy or a different length."""
        model = _finalize(_replicated_builder(), "cpu")
        record = model._shape_contact_pair_templates
        self.assertTrue(record.describes(model.shape_contact_pairs))
        self.assertFalse(record.describes(wp.clone(model.shape_contact_pairs)))
        self.assertFalse(record.describes(model.shape_contact_pairs.numpy()))
        self.assertFalse(ShapePairTemplates(record.global_pairs, record.templates, record.runs).describes(None))

    def test_replay_matches_builder_order(self):
        """Lead with global pairs, then each world's global-local pairs as (min, max) and its local pairs."""
        global_local = np.array([[0, 2], [7, 0]], dtype=np.int32)
        local = np.array([[0, 1], [1, 2]], dtype=np.int32)
        replay = replay_shape_pair_templates(
            np.array([[0, 7]], dtype=np.int32), ((global_local, local),), ((0, np.array([1, 4], dtype=np.int32)),)
        )
        expected = [[0, 7], [0, 3], [1, 7], [1, 2], [2, 3], [0, 6], [4, 7], [4, 5], [5, 6]]
        self.assertEqual(replay.tolist(), expected)
        self.assertEqual(replay.dtype, np.int32)


if __name__ == "__main__":
    unittest.main(verbosity=2)

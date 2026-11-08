# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""World-template explicit broad phase: the builder's template record and the kernel that tests it.

The template kernel must test exactly the pairs of the flat explicit list. Counts
and sorted candidate sets are compared with :class:`BroadPhaseExplicit` on every
device; on CPU, where both run serially in list order, the arrays must be equal.
"""

import contextlib
import functools
import itertools
import json
import os
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.geometry.broad_phase_nxn import (
    BroadPhaseExplicit,
    BroadPhaseExplicitTemplates,
    ShapePairTemplates,
    _nxn_broadphase_precomputed_pairs,
    _pair_template_kernel,
    replay_shape_pair_templates,
)
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    scoped_capture_after_gc,
)

_FLAG = "NEWTON_BROAD_PHASE_TEMPLATES"


def _synthetic(
    seed,
    *,
    worlds,
    world_shapes,
    template_specs,
    global_shapes=3,
    global_pairs=2,
    gap=2,
    empty_template=False,
    globals_last=False,
):
    """Random templates over worlds of ``world_shapes`` shapes, ``gap`` spare shapes after each world.

    Args:
        template_specs: Per template, (global-local pair count, local pair count).
        empty_template: Add a template without pairs, which the list skips.
        globals_last: Number the global shapes after the last world instead of before the first, so
            every global-local pair stores its global shape second.

    Returns:
        (record, shape_count): the unbound record and the shapes it spans.
    """
    rng = np.random.default_rng(seed)
    stride = world_shapes + gap
    first_global = worlds * stride if globals_last else 0
    templates = []
    for global_local_count, local_count in template_specs:
        local_pairs = set()
        while len(local_pairs) < local_count:
            a, b = sorted(rng.choice(world_shapes, size=2, replace=False).tolist())
            local_pairs.add((a, b))
        global_local = set()
        while len(global_local) < global_local_count:
            global_local.add((first_global + int(rng.integers(global_shapes)), int(rng.integers(world_shapes))))
        templates.append(
            (
                np.asarray(sorted(global_local), dtype=np.int32).reshape((-1, 2)),
                np.asarray(sorted(local_pairs), dtype=np.int32).reshape((-1, 2)),
            )
        )
    if empty_template:
        templates.append((np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)))
    starts = (0 if globals_last else global_shapes) + np.arange(worlds, dtype=np.int64) * stride
    assignment = rng.integers(len(templates), size=worlds)
    runs = []
    for world, template in enumerate(assignment.tolist()):
        if runs and runs[-1][0] == template:
            runs[-1][1].append(int(starts[world]))
        else:
            runs.append((template, [int(starts[world])]))
    global_list = set()
    while len(global_list) < global_pairs:
        a, b = sorted(rng.choice(global_shapes, size=2, replace=False).tolist())
        global_list.add((first_global + a, first_global + b))
    record = ShapePairTemplates(
        global_pairs=np.asarray(sorted(global_list), dtype=np.int32).reshape((-1, 2)),
        templates=tuple(templates),
        runs=tuple((template, np.asarray(run, dtype=np.int32)) for template, run in runs),
    )
    return record, global_shapes + worlds * stride


def _global_second_count(record):
    """Count list pairs whose global shape follows the local one: the kernel must swap them to (min, max)."""
    count = 0
    for template, starts in record.runs:
        global_local = record.templates[template][0]
        local = global_local[None, :, 1] + np.asarray(starts, dtype=np.int64)[:, None]
        count += int((global_local[None, :, 0] > local).sum())
    return count


def _random_bounds(seed, shape_count, device):
    """Quantized boxes, so bounds often touch exactly, with some NaN and empty boxes."""
    rng = np.random.default_rng(seed)
    center = rng.integers(-4, 5, size=(shape_count, 3)) * 0.25
    half = rng.integers(0, 4, size=(shape_count, 3)) * 0.25
    lower = (center - half).astype(np.float32)
    upper = (center + half).astype(np.float32)
    lower[3::17, 1] = np.nan
    lower[5::19] = upper[5::19] + 1.0  # empty boxes
    upper[7::23, 2] = -0.0
    lower[7::23, 2] = 0.0
    return wp.array(lower, dtype=wp.vec3, device=device), wp.array(upper, dtype=wp.vec3, device=device)


def _random_bodies(seed, shape_count, device):
    """Shape bodies with static (-1), dynamic and kinematic bodies."""
    rng = np.random.default_rng(seed)
    body_count = max(1, shape_count // 3)
    shape_body = rng.integers(-1, body_count, size=shape_count).astype(np.int32)
    body_flags = np.where(rng.random(body_count) < 0.4, 2, 1).astype(np.int32)
    return wp.array(shape_body, dtype=wp.int32, device=device), wp.array(body_flags, dtype=wp.int32, device=device)


def _launch_both(pairs, layout, lower, upper, *, capacity=None, shape_body=None, body_flags=None, include=True):
    device = pairs.device
    capacity = max(1, pairs.shape[0]) if capacity is None else capacity
    flat_pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)
    flat_count = wp.zeros(1, dtype=wp.int32, device=device)
    BroadPhaseExplicit().launch(
        lower,
        upper,
        None,
        pairs,
        pairs.shape[0],
        flat_pairs,
        flat_count,
        shape_body=shape_body,
        body_flags=body_flags,
        include_static_kinematic_pairs=include,
    )
    template_pairs = wp.zeros(capacity, dtype=wp.vec2i, device=device)
    template_count = wp.zeros(1, dtype=wp.int32, device=device)
    layout.launch(
        lower,
        upper,
        pairs,
        template_pairs,
        template_count,
        shape_body=shape_body,
        body_flags=body_flags,
        include_static_kinematic_pairs=include,
    )
    return (
        (flat_pairs.numpy(), int(flat_count.numpy()[0])),
        (template_pairs.numpy(), int(template_count.numpy()[0])),
    )


def _sorted_codes(pairs, count, capacity):
    written = pairs[: min(count, capacity)].astype(np.int64)
    return np.sort(written[:, 0] << 32 | written[:, 1])


def _assert_same_candidates(test, device, flat, template, capacity, everything=None):
    """Equal counts and candidate sets; on CPU equal arrays. Saturated CUDA runs write distinct real hits."""
    (flat_pairs, flat_count), (template_pairs, template_count) = flat, template
    test.assertEqual(template_count, flat_count)
    if device.is_cpu:
        # Both kernels run serially in list order on CPU: the arrays match, order included.
        np.testing.assert_array_equal(template_pairs, flat_pairs)
    elif flat_count <= capacity:
        np.testing.assert_array_equal(
            _sorted_codes(template_pairs, template_count, capacity), _sorted_codes(flat_pairs, flat_count, capacity)
        )
    else:
        # Which hits fill the capacity depends on scheduling in both kernels.
        written = _sorted_codes(template_pairs, template_count, capacity)
        test.assertEqual(np.unique(written).shape[0], capacity)
        test.assertTrue(np.isin(written, everything).all())


def test_synthetic_templates_match_flat_kernel(test, device):
    """Test the same candidates as the flat kernel over random templates, runs, globals and bounds."""
    cases = (
        # Small templates pack 8 worlds per block.
        {"template_specs": ((3, 20), (0, 9), (2, 0)), "worlds": 21, "world_shapes": 12, "empty_template": True},
        # One large template: several warps per world, a global block, a tail block.
        {"template_specs": ((5, 300),), "worlds": 7, "world_shapes": 40},
        # Over 256 staged shapes per world: shared memory, not lanes, limits worlds per block (3 x 64 lanes).
        {"template_specs": ((40, 260), (0, 120)), "worlds": 9, "world_shapes": 400},
        # 1100 pairs per world: one 256-lane world per block, as on the largest templates.
        {"template_specs": ((8, 1100),), "worlds": 5, "world_shapes": 60},
        # Global shapes after the worlds: every global-local pair is swapped to (min, max).
        {"template_specs": ((6, 40), (3, 0), (0, 15)), "worlds": 13, "world_shapes": 16, "globals_last": True},
        # 2500 global-global pairs fill two leading blocks of 2048 (256 lanes x 8 pairs).
        {
            "template_specs": ((4, 30), (0, 12)),
            "worlds": 10,
            "world_shapes": 14,
            "global_shapes": 80,
            "global_pairs": 2500,
        },
    )
    for case, options in enumerate(cases):
        record, shape_count = _synthetic(case, **options)
        test.assertEqual(_global_second_count(record) > 0, options.get("globals_last", False))
        pairs = wp.array(record.replay(), dtype=wp.vec2i, device=device)
        layout = BroadPhaseExplicitTemplates(record, device=device)
        test.assertIsNone(layout.unsupported_reason)
        test.assertTrue(layout.verify(pairs))
        test.assertEqual(layout.pair_count, pairs.shape[0])
        if case == 2:
            test.assertGreater(layout.staged_shapes, 256)
            test.assertLess(layout.worlds_per_block, 256 // layout.threads_per_world)
        test.assertEqual(layout.global_blocks, 2 if case == 5 else 1)
        lower, upper = _random_bounds(100 + case, shape_count, device)
        shape_body, body_flags = _random_bodies(200 + case, shape_count, device)
        for include in (True, False):
            for bodies in ((None, None), (shape_body, body_flags), (shape_body, None)):
                with test.subTest(
                    case=case, include=include, bodies=bodies[0] is not None, flags=bodies[1] is not None
                ):
                    flat, template = _launch_both(
                        pairs, layout, lower, upper, shape_body=bodies[0], body_flags=bodies[1], include=include
                    )
                    test.assertGreater(flat[1], 0)
                    _assert_same_candidates(test, device, flat, template, pairs.shape[0])
        # Saturation: the count still reaches every hit, as in write_pair.
        everything = _sorted_codes(*_launch_both(pairs, layout, lower, upper)[0], pairs.shape[0])
        capacity = 5
        with test.subTest(case=case, capacity=capacity):
            flat, template = _launch_both(pairs, layout, lower, upper, capacity=capacity)
            test.assertGreater(flat[1], capacity)
            _assert_same_candidates(test, device, flat, template, capacity, everything)


def _exact_template(pair_count, staged):
    """One template with exactly ``pair_count`` local pairs over exactly ``staged`` shapes, two worlds."""
    pairs = [(0, k) for k in range(1, staged)]
    pairs += [(i, j) for i in range(1, staged) for j in range(i + 1, staged)][: pair_count - len(pairs)]
    return ShapePairTemplates(
        global_pairs=np.empty((0, 2), dtype=np.int32),
        templates=((np.empty((0, 2), dtype=np.int32), np.asarray(pairs[:pair_count], dtype=np.int32)),),
        runs=((0, np.array([0, staged], dtype=np.int32)),),
    )


_LAUNCH_TILED = wp.launch_tiled


def _launch_tiled_one_block(*args, **kwargs):
    """``wp.launch_tiled`` with the CUDA grid capped at one block."""
    return _LAUNCH_TILED(*args, **kwargs, max_blocks=1)


def test_layout_shapes_match_flat_kernel(test, device):
    """Choose lanes per world and worlds per block, and test each launch shape like the flat kernel."""
    expectations = (
        (23, 12, 32, 8),  # one warp per world, eight worlds per block
        (305, 41, 64, 4),  # two warps per world
        (300, 300, 64, 3),  # 300 staged shapes leave shared memory for three worlds: 192-thread blocks
        (1108, 61, 256, 1),  # one 256-lane world per block, as on the largest templates
    )
    for case, (pair_count, staged, threads, worlds_per_block) in enumerate(expectations):
        record = _exact_template(pair_count, staged)
        layout = BroadPhaseExplicitTemplates(record, device=device)
        with test.subTest(pairs=pair_count, staged=staged):
            test.assertEqual((layout.max_pairs_per_world, layout.staged_shapes), (pair_count, staged))
            test.assertEqual(layout.threads_per_world, threads)
            test.assertEqual(layout.worlds_per_block, worlds_per_block)
            pairs = wp.array(record.replay(), dtype=wp.vec2i, device=device)
            test.assertTrue(layout.verify(pairs))
            lower, upper = _random_bounds(300 + case, 2 * staged, device)
            flat, template = _launch_both(pairs, layout, lower, upper)
            test.assertGreater(flat[1], 0)
            _assert_same_candidates(test, device, flat, template, pairs.shape[0])
            # Each thread runs one logical block (shared memory is staged once per block), so the
            # kernel has no grid-stride loop and a capped CUDA grid is refused.
            test.assertIs(layout._kernel.options.get("grid_stride"), False)
            if device.is_cuda:
                with (
                    mock.patch.object(wp, "launch_tiled", _launch_tiled_one_block),
                    test.assertRaisesRegex(RuntimeError, "grid_stride"),
                ):
                    _launch_both(pairs, layout, lower, upper)


def test_static_pairs_filtered_without_body_flags(test, device):
    """Drop static-static pairs when shape bodies are given without body flags, as the flat kernel does.

    The flat kernel drops them with and without static-kinematic pairs.
    """
    record, shape_count = _synthetic(11, worlds=6, world_shapes=20, template_specs=((3, 60), (2, 30)), global_pairs=3)
    pairs = wp.array(record.replay(), dtype=wp.vec2i, device=device)
    layout = BroadPhaseExplicitTemplates(record, device=device)
    test.assertTrue(layout.verify(pairs))
    # Every box overlaps every other one, and every other shape (globals included) is static.
    lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
    upper = wp.array(np.ones((shape_count, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    bodies = np.arange(shape_count, dtype=np.int32)
    bodies[::2] = -1
    shape_body = wp.array(bodies, dtype=wp.int32, device=device)
    for include in (True, False):
        with test.subTest(include=include):
            flat, template = _launch_both(pairs, layout, lower, upper, shape_body=shape_body, include=include)
            # Not vacuous: the static-static rule drops pairs without body flags.
            test.assertLess(flat[1], pairs.shape[0])
            _assert_same_candidates(test, device, flat, template, pairs.shape[0])


def test_verification_rejects_a_changed_list(test, device):
    """Refuse a list whose length or any entry differs from the templates' replay."""
    record, _ = _synthetic(7, worlds=6, world_shapes=10, template_specs=((2, 12), (1, 5)))
    replay = record.replay()
    layout = BroadPhaseExplicitTemplates(record, device=device)
    test.assertTrue(layout.verify(wp.array(replay, dtype=wp.vec2i, device=device)))
    changed = replay.copy()
    changed[-1] = changed[-2]
    test.assertFalse(layout.verify(wp.array(changed, dtype=wp.vec2i, device=device)))
    test.assertFalse(layout.verify(wp.array(replay[:-1], dtype=wp.vec2i, device=device)))
    swapped = replay.copy()
    row = record.global_pairs.shape[0] + 1
    swapped[row] = swapped[row, ::-1]
    test.assertFalse(layout.verify(wp.array(swapped, dtype=wp.vec2i, device=device)))


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


_GLOBAL_ORDERS = ("first", "last", "split")


def _replicated_builder(worlds=11, *, global_pair=True, residual=True, mixed=True, empty=True, global_order="first"):
    """Replicated robots between global shapes.

    Args:
        global_pair: Add a global kinematic box that touches the ground: one global-global pair.
        residual: Filter the ground against worlds 1 and 5, which gives them templates of their own.
        mixed: Use the second robot variant in every fourth world.
        empty: Make world 3 shapeless and world 7 non-colliding.
        global_order: Add the global shapes before the worlds (``"first"``), after them (``"last"``,
            as ``replicate()`` followed by ``add_ground_plane()``), or the ground before and the box
            after (``"split"``). Global shapes after the worlds are stored second in their pairs.
    """
    builder = newton.ModelBuilder()
    ground = None

    def add_ground():
        nonlocal ground
        ground = builder.add_ground_plane()

    def add_box():
        if global_pair:
            body = builder.add_body(xform=wp.transform((0.0, 0.0, -0.4), wp.quat_identity()), is_kinematic=True)
            builder.add_shape_box(body, hx=30.0, hy=3.0, hz=0.5)

    leading, trailing = {
        "first": ((add_ground, add_box), ()),
        "last": ((), (add_ground, add_box)),
        "split": ((add_ground,), (add_box,)),
    }[global_order]
    for add in leading:
        add()
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
    for add in trailing:
        add()
    if residual:
        # World-specific filters give worlds 1 and 5 a template of their own.
        for world in (1, 5):
            builder.add_shape_collision_filter_pair(ground, first_shape[world])
    builder.first_shape = first_shape
    return builder


def _finalize(builder, device):
    """Finalize and build the explicit pair list, which records its world templates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = builder.finalize(device=device)
    _ = model.shape_contact_pairs
    return model


class TestShapePairTemplateRecord(unittest.TestCase):
    def test_record_replays_the_model_pairs(self):
        """Keep a record whose replay equals the model's explicit pairs for each builder layout."""
        layouts = {
            "replicate": {"global_pair": False, "residual": False, "mixed": False, "empty": False},
            "global pair": {"global_pair": True, "residual": False, "mixed": False, "empty": False},
            "residual filters": {"global_pair": True, "residual": True, "mixed": False, "empty": False},
            "mixed worlds": {"global_pair": True, "residual": True, "mixed": True, "empty": True},
            "globals last": {"global_order": "last"},
            "globals split": {"global_order": "split"},
            "replicate, ground last": {"global_pair": False, "residual": False, "mixed": False, "global_order": "last"},
        }
        for name, options in layouts.items():
            with self.subTest(layout=name):
                model = _finalize(_replicated_builder(**options), "cpu")
                record = model._shape_contact_pair_templates
                self.assertIsInstance(record, ShapePairTemplates)
                self.assertTrue(record.describes(model.shape_contact_pairs))
                np.testing.assert_array_equal(record.replay(), model.shape_contact_pairs.numpy())
                self.assertEqual(record.global_pairs.shape[0], int(options.get("global_pair", True)))
                # Global shapes added after the worlds are stored second in their pairs.
                order = options.get("global_order", "first")
                self.assertEqual(_global_second_count(record) > 0, order != "first")
        model = _finalize(_replicated_builder(), "cpu")
        record = model._shape_contact_pair_templates
        # Robot, variant and filtered worlds use three templates; pair-free worlds replay nothing.
        self.assertEqual(len(record.templates), 3)
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
        """Keep no record before the list is built or after it is replaced."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = _replicated_builder().finalize(device="cpu")
        self.assertIsNone(model._shape_contact_pair_templates)
        pairs = model.shape_contact_pairs
        self.assertIsNotNone(model._shape_contact_pair_templates)
        model.shape_contact_pairs = wp.clone(pairs)
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


class TestBroadPhaseTemplateRule(unittest.TestCase):
    def _single(self, pairs_per_world, worlds, staged=None, global_pairs=0):
        staged = pairs_per_world + 1 if staged is None else staged
        local = np.stack((np.zeros(pairs_per_world), 1 + np.arange(pairs_per_world) % (staged - 1)), axis=1)
        record = ShapePairTemplates(
            global_pairs=np.zeros((global_pairs, 2), dtype=np.int32),
            templates=((np.empty((0, 2), dtype=np.int32), local.astype(np.int32)),),
            runs=((0, (np.arange(worlds) * staged).astype(np.int32)),),
        )
        return BroadPhaseExplicitTemplates(record, device="cpu")

    def test_automatic_rule_admits_large_templates(self):
        """Admit automatically only at 2**20 template pairs or more and 128 or more pairs per world."""
        self.assertTrue(self._single(128, 8192).automatic)
        self.assertFalse(self._single(127, 8257).automatic)
        self.assertFalse(self._single(128, 8191).automatic)
        self.assertEqual(self._single(128, 8192).pair_count, 1 << 20)
        # At 64 pairs per world the template kernel only ties the flat one on the device.
        self.assertFalse(self._single(64, 16384).automatic)

    def test_global_pairs_do_not_count_toward_admission(self):
        """Keep the flat kernel automatically when only global-global pairs make the list large."""
        # Global-global pairs are streamed from the list as in the flat kernel, so they save nothing.
        layout = self._single(128, 16, global_pairs=1 << 20)
        self.assertGreaterEqual(layout.pair_count, layout.AUTO_MIN_PAIRS)
        self.assertGreaterEqual(layout.mean_pairs_per_world, layout.AUTO_MIN_PAIRS_PER_WORLD)
        self.assertFalse(layout.automatic)
        # They spread over leading blocks of the launch's lanes x 8 pairs instead of running in one block.
        self.assertEqual(layout.global_chunk, layout.block_dim * 8)
        self.assertEqual(layout.global_blocks, -(-(1 << 20) // layout.global_chunk))
        self.assertGreater(layout.global_blocks, 1)
        self.assertEqual(layout.describe()["global_blocks"], layout.global_blocks)

    def test_staging_capacity_bounds_admission(self):
        """Refuse templates referencing more shapes than a world can stage."""
        self.assertIsNone(self._single(1023, 2).unsupported_reason)
        too_many = self._single(1024, 2)
        self.assertIn("1025 shapes", too_many.unsupported_reason)
        self.assertFalse(too_many.automatic)
        self.assertFalse(too_many.verify(wp.zeros(too_many.pair_count, dtype=wp.vec2i, device="cpu")))

    def test_describe_reports_largest_and_mean_templates(self):
        """Report the largest template's pairs, which set the lanes, next to the mean the rule uses."""
        record, _ = _synthetic(3, worlds=12, world_shapes=20, template_specs=((2, 70), (1, 9)))
        described = BroadPhaseExplicitTemplates(record, device="cpu").describe()
        per_world = [sum(map(len, record.templates[template])) for template, starts in record.runs for _ in starts]
        self.assertEqual(len(set(per_world)), 2)
        self.assertEqual(described["max_pairs_per_world"], 72)
        self.assertAlmostEqual(described["mean_pairs_per_world"], float(np.mean(per_world)))
        self.assertLess(described["mean_pairs_per_world"], 72)
        self.assertNotIn("pairs_per_world", described)
        json.dumps(described, allow_nan=False)

    def test_flat_kernel_order_follows_the_module_option(self):
        """Report ordered flat-kernel writes only when its module runs Warp's deterministic mode."""
        options = _nxn_broadphase_precomputed_pairs.module.options
        self.assertFalse(BroadPhaseExplicitTemplates.flat_kernel_orders_writes())
        with mock.patch.dict(options, {"deterministic": wp.DeterministicMode.RUN_TO_RUN}):
            self.assertTrue(BroadPhaseExplicitTemplates.flat_kernel_orders_writes())
        self.assertFalse(BroadPhaseExplicitTemplates.flat_kernel_orders_writes())

    def test_verification_uploads_the_layout(self):
        """Allocate the device layout in verify(), so a verified layout allocates nothing in launch()."""
        empty = (np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32))
        records = {
            "template pairs": _synthetic(4, worlds=5, world_shapes=10, template_specs=((2, 12),))[0],
            # Worlds without template pairs: verify() returned before uploading.
            "global pairs only": ShapePairTemplates(
                np.array([[0, 1]], dtype=np.int32), (empty,), ((0, np.array([2, 3], dtype=np.int32)),)
            ),
        }
        for name, record in records.items():
            with self.subTest(case=name):
                layout = BroadPhaseExplicitTemplates(record, device="cpu")
                self.assertIsNone(layout.data)
                self.assertTrue(layout.verify(wp.array(record.replay(), dtype=wp.vec2i, device="cpu")))
                self.assertIsNotNone(layout.data)


def _contacts(pipeline, model, state):
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return contacts


_CONTACT_ARRAYS = (
    "rigid_contact_count",
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_offset0",
    "rigid_contact_offset1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
)


def _contact_rows(contacts):
    """Contacts as rows (shapes, points, offsets, normal, margins), sorted so slot order is ignored."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    columns = [
        contacts.rigid_contact_shape0.numpy()[:count, None].astype(np.float64),
        contacts.rigid_contact_shape1.numpy()[:count, None].astype(np.float64),
        contacts.rigid_contact_point0.numpy()[:count].astype(np.float64),
        contacts.rigid_contact_point1.numpy()[:count].astype(np.float64),
        contacts.rigid_contact_offset0.numpy()[:count].astype(np.float64),
        contacts.rigid_contact_offset1.numpy()[:count].astype(np.float64),
        contacts.rigid_contact_normal.numpy()[:count].astype(np.float64),
        contacts.rigid_contact_margin0.numpy()[:count, None].astype(np.float64),
        contacts.rigid_contact_margin1.numpy()[:count, None].astype(np.float64),
    ]
    rows = np.concatenate(columns, axis=1)
    return rows[np.lexsort(rows.T[::-1])]


def _pipeline(model, setting, **kwargs):
    environ = {} if setting is None else {_FLAG: setting}
    with mock.patch.dict(os.environ, environ):
        if setting is None:
            os.environ.pop(_FLAG, None)
        return newton.CollisionPipeline(model, **kwargs)


def _kernel_record(pipeline):
    return pipeline.get_collision_selection()["narrow_phase"]["broad_phase_kernel"]


def _launched(run):
    """Call ``run()`` and count the template and flat explicit broad-phase launches it makes (both still run)."""
    with (
        mock.patch.object(
            BroadPhaseExplicitTemplates, "launch", autospec=True, side_effect=BroadPhaseExplicitTemplates.launch
        ) as templates,
        mock.patch.object(BroadPhaseExplicit, "launch", autospec=True, side_effect=BroadPhaseExplicit.launch) as flat,
    ):
        result = run()
    return result, {"templates": templates.call_count, "flat": flat.call_count}


def _global_order_models(test, device):
    """Yield (global order, model, posed state); global shapes after the worlds make the kernel swap pairs."""
    for global_order in _GLOBAL_ORDERS:
        model = _finalize(_replicated_builder(global_order=global_order), device)
        test.assertEqual(_global_second_count(model._shape_contact_pair_templates) > 0, global_order != "first")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        yield global_order, model, state


def test_pipeline_templates_match_flat_contacts(test, device):
    """Produce the flat kernel's candidates and contacts with the world templates, globals first or last."""
    for (global_order, model, state), deterministic in itertools.product(
        _global_order_models(test, device), (False, True)
    ):
        with test.subTest(global_order=global_order, deterministic=deterministic):
            templated = _pipeline(model, "1", deterministic=deterministic)
            flat = _pipeline(model, "0", deterministic=deterministic)
            record = _kernel_record(templated)
            test.assertEqual(record["kernel"], "world_templates")
            test.assertEqual(record["source"], "environment")
            test.assertEqual(record["templates"]["global_pairs"], 1)
            test.assertEqual(_kernel_record(flat)["kernel"], "flat_pairs")
            templated_contacts, launches = _launched(functools.partial(_contacts, templated, model, state))
            test.assertEqual(launches, {"templates": 1, "flat": 0})
            flat_contacts, launches = _launched(functools.partial(_contacts, flat, model, state))
            test.assertEqual(launches, {"templates": 0, "flat": 1})
            count = int(flat.broad_phase_pair_count.numpy()[0])
            test.assertGreater(count, 0)
            test.assertEqual(int(templated.broad_phase_pair_count.numpy()[0]), count)
            test.assertGreater(int(flat_contacts.rigid_contact_count.numpy()[0]), 0)
            templated_pairs = templated.broad_phase_shape_pairs.numpy()
            flat_pairs = flat.broad_phase_shape_pairs.numpy()
            if device.is_cpu:
                # Serial CPU runs keep list order.
                np.testing.assert_array_equal(templated_pairs[:count], flat_pairs[:count])
            np.testing.assert_array_equal(
                _sorted_codes(templated_pairs, count, count), _sorted_codes(flat_pairs, count, count)
            )
            if device.is_cpu or deterministic:
                # Serial CPU runs keep list order, and deterministic pipelines sort contacts.
                for name in _CONTACT_ARRAYS:
                    np.testing.assert_array_equal(
                        getattr(templated_contacts, name).numpy(), getattr(flat_contacts, name).numpy(), err_msg=name
                    )
            np.testing.assert_array_equal(_contact_rows(templated_contacts), _contact_rows(flat_contacts))


def test_pipeline_immovable_filter_matches_flat(test, device):
    """Filter static and kinematic pairs exactly as the flat kernel does, globals first or last."""
    for global_order, model, state in _global_order_models(test, device):
        with test.subTest(global_order=global_order):
            templated = _pipeline(model, "1", include_static_kinematic_pairs=False)
            flat = _pipeline(model, "0", include_static_kinematic_pairs=False)
            unfiltered = _pipeline(model, "0")
            templated_contacts, launches = _launched(functools.partial(_contacts, templated, model, state))
            test.assertEqual(launches, {"templates": 1, "flat": 0})
            flat_contacts = _contacts(flat, model, state)
            _contacts(unfiltered, model, state)
            count = int(flat.broad_phase_pair_count.numpy()[0])
            # Not vacuous: the fixture's static and kinematic shapes overlap.
            test.assertLess(count, int(unfiltered.broad_phase_pair_count.numpy()[0]))
            test.assertEqual(int(templated.broad_phase_pair_count.numpy()[0]), count)
            templated_pairs = templated.broad_phase_shape_pairs.numpy()
            flat_pairs = flat.broad_phase_shape_pairs.numpy()
            if device.is_cpu:
                np.testing.assert_array_equal(templated_pairs[:count], flat_pairs[:count])
            np.testing.assert_array_equal(
                _sorted_codes(templated_pairs, count, count), _sorted_codes(flat_pairs, count, count)
            )
            np.testing.assert_array_equal(_contact_rows(templated_contacts), _contact_rows(flat_contacts))


class TestBroadPhaseTemplateSelection(unittest.TestCase):
    def test_fallbacks_keep_the_flat_kernel(self):
        """Keep the flat kernel, with a reason, wherever the templates do not stand for the list."""
        model = _finalize(_replicated_builder(), "cpu")
        cases = {
            "automatic, small": (None, {}, "Automatic: below"),
            "environment 0": ("0", {}, "NEWTON_BROAD_PHASE_TEMPLATES=0"),
            "user pairs": ("1", {"shape_pairs_filtered": wp.clone(model.shape_contact_pairs)}, "not the model-built"),
            "speculative": (
                "1",
                {"speculative_contact_gap_max": 0.1},
                "Speculative",
            ),
        }
        for name, (setting, kwargs, reason) in cases.items():
            with self.subTest(case=name):
                record = _kernel_record(_pipeline(model, setting, **kwargs))
                self.assertEqual(record["kernel"], "flat_pairs")
                self.assertIn(reason, record["reason"])
                json.dumps(record, allow_nan=False)
        record = _kernel_record(_pipeline(model, None))
        self.assertEqual(record["source"], "automatic")
        self.assertEqual(record["templates"]["templates"], 3)
        # Lanes follow the largest template, the automatic rule the mean.
        layout = record["templates"]
        self.assertGreater(layout["max_pairs_per_world"], layout["mean_pairs_per_world"])
        self.assertIsNone(_kernel_record(_pipeline(model, "1", broad_phase="nxn")))

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        with self.subTest(case="prebuilt broad phase"):
            # An expert pipeline launches its own broad phase, even over the model-built list.
            narrow_phase = _pipeline(model, "0").narrow_phase
            expert = _pipeline(model, "1", broad_phase=BroadPhaseExplicit(), narrow_phase=narrow_phase)
            record = _kernel_record(expert)
            self.assertEqual(record["kernel"], "flat_pairs")
            self.assertIn("prebuilt broad phase", record["reason"])
            _, launches = _launched(functools.partial(_contacts, expert, model, state))
            self.assertEqual(launches, {"templates": 0, "flat": 1})
        with (
            self.subTest(case="staging capacity"),
            mock.patch.object(BroadPhaseExplicitTemplates, "MAX_STAGED_SHAPES", 4),
        ):
            # Templates that reference more shapes than a world can stage keep the flat kernel.
            record = _kernel_record(_pipeline(model, "1"))
            self.assertEqual(record["kernel"], "flat_pairs")
            self.assertIn("the template kernel stages at most 4", record["reason"])
            self.assertIsNotNone(record["templates"])

    def test_automatic_route_matches_flat_kernel(self):
        """Admit large templates with the flag unset, launch them in collide() and match the flat kernel."""
        model = _finalize(_replicated_builder(), "cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        # The fixture is far below the automatic thresholds; lower them to take the default route.
        with (
            mock.patch.object(BroadPhaseExplicitTemplates, "AUTO_MIN_PAIRS", 1),
            mock.patch.object(BroadPhaseExplicitTemplates, "AUTO_MIN_PAIRS_PER_WORLD", 1),
        ):
            automatic = _pipeline(model, None)
        record = _kernel_record(automatic)
        self.assertEqual(record["kernel"], "world_templates")
        self.assertEqual(record["source"], "automatic")
        self.assertIn("Automatic: large world templates", record["reason"])
        contacts, launches = _launched(functools.partial(_contacts, automatic, model, state))
        self.assertEqual(launches, {"templates": 1, "flat": 0})
        flat = _pipeline(model, "0")
        flat_contacts = _contacts(flat, model, state)
        count = int(flat.broad_phase_pair_count.numpy()[0])
        self.assertGreater(count, 0)
        self.assertEqual(int(automatic.broad_phase_pair_count.numpy()[0]), count)
        # Serial CPU runs keep list order: the arrays match.
        np.testing.assert_array_equal(
            automatic.broad_phase_shape_pairs.numpy()[:count], flat.broad_phase_shape_pairs.numpy()[:count]
        )
        for name in _CONTACT_ARRAYS:
            np.testing.assert_array_equal(
                getattr(contacts, name).numpy(), getattr(flat_contacts, name).numpy(), err_msg=name
            )

    def test_global_pairs_only_keep_the_flat_kernel(self):
        """Keep the flat kernel, even when forced, for a list whose world templates hold no pairs."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)  # a global body: one global-global pair
        visual = newton.ModelBuilder()
        visual.add_shape_sphere(
            visual.add_body(), radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
        )
        for world in range(3):
            builder.add_world(visual, xform=wp.transform((2.0 * world, 0.0, 0.0), wp.quat_identity()))
        model = _finalize(builder, "cpu")
        self.assertEqual(model.shape_contact_pair_count, 1)
        pipeline = _pipeline(model, "1")
        record = _kernel_record(pipeline)
        self.assertEqual(record["kernel"], "flat_pairs")
        self.assertIn("hold no pairs", record["reason"])
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        _, launches = _launched(functools.partial(_contacts, pipeline, model, state))
        self.assertEqual(launches, {"templates": 0, "flat": 1})
        self.assertEqual(int(pipeline.broad_phase_pair_count.numpy()[0]), 1)

    def test_deterministic_mode_keeps_the_flat_kernel(self):
        """Keep the flat kernel when Warp orders its writes, unless the pipeline sorts contacts anyway."""
        model = _finalize(_replicated_builder(), "cpu")
        with mock.patch.object(BroadPhaseExplicitTemplates, "flat_kernel_orders_writes", return_value=True):
            record = _kernel_record(_pipeline(model, "1"))
            self.assertEqual(record["kernel"], "flat_pairs")
            self.assertIn("deterministic mode", record["reason"])
            self.assertEqual(_kernel_record(_pipeline(model, "1", deterministic=True))["kernel"], "world_templates")

    def test_changed_list_keeps_the_flat_kernel(self):
        """Fall back when the model-built list was changed in place after finalization."""
        model = _finalize(_replicated_builder(), "cpu")
        pairs = model.shape_contact_pairs.numpy()
        pairs[-1] = pairs[-2]
        model.shape_contact_pairs.assign(pairs)
        record = _kernel_record(_pipeline(model, "1"))
        self.assertEqual(record["kernel"], "flat_pairs")
        self.assertIn("no longer match", record["reason"])

    def test_general_list_keeps_the_flat_kernel(self):
        """Keep the flat kernel for an assigned list, which has no world templates."""
        model = _finalize(_replicated_builder(), "cpu")
        model.shape_contact_pairs = wp.clone(model.shape_contact_pairs)
        record = _kernel_record(_pipeline(model, "1"))
        self.assertEqual(record["kernel"], "flat_pairs")
        self.assertIn("not the model-built", record["reason"])

    def test_invalid_setting_raises(self):
        """Reject values other than 0 and 1."""
        model = _finalize(_replicated_builder(), "cpu")
        with self.assertRaisesRegex(ValueError, "NEWTON_BROAD_PHASE_TEMPLATES must be 0 or 1"):
            _pipeline(model, "auto")


def _capture_collide(pipeline, state, contacts, **capture_options):
    """Capture one ``collide()`` in a graph and return the graph."""
    with scoped_capture_after_gc(device=pipeline.device, **capture_options) as capture:
        pipeline.collide(state, contacts)
    return capture.graph


def _replay_matches_flat(test, graph, templated, contacts, flat, state):
    """Replay ``graph`` into cleared outputs, compare with ``flat`` run eagerly on ``state``; return the count."""
    flat_contacts = _contacts(flat, flat.model, state)
    count = int(flat.broad_phase_pair_count.numpy()[0])
    templated.broad_phase_shape_pairs.zero_()
    templated.broad_phase_pair_count.zero_()
    for _ in range(3):
        wp.capture_launch(graph)
    test.assertEqual(int(templated.broad_phase_pair_count.numpy()[0]), count)
    np.testing.assert_array_equal(
        _sorted_codes(templated.broad_phase_shape_pairs.numpy(), count, count),
        _sorted_codes(flat.broad_phase_shape_pairs.numpy(), count, count),
    )
    np.testing.assert_array_equal(_contact_rows(contacts), _contact_rows(flat_contacts))
    return count


def test_graph_replays_template_broad_phase(test, device):
    """Capture the template broad phase in a CUDA graph and replay the flat kernel's candidates on live state."""
    for global_order, model, state in _global_order_models(test, device):
        with test.subTest(global_order=global_order):
            templated = _pipeline(model, "1")
            flat = _pipeline(model, "0")
            test.assertEqual(_kernel_record(templated)["kernel"], "world_templates")
            contacts = _contacts(templated, model, state)  # load modules before capture
            graph, launches = _launched(
                functools.partial(_capture_collide, templated, state, contacts, force_module_load=False)
            )
            # The capture holds the kernel's plain and forecasting variants, one in each branch of the conditional
            # node that picks the forecasting pass.
            test.assertEqual(launches, {"templates": 2, "flat": 0})
            count = _replay_matches_flat(test, graph, templated, contacts, flat, state)
            # Lift world 4 clear of the global box and its static shape: the replay must read the new poses.
            body_q = state.body_q.numpy()
            body_q[model.body_world.numpy() == 4, 2] += 3.0
            state.body_q.assign(body_q)
            test.assertLess(_replay_matches_flat(test, graph, templated, contacts, flat, state), count)


@contextlib.contextmanager
def _recorded_while_capturing(device):
    """Record host reads, synchronizations and template-launch allocations made while ``device`` captures a graph.

    Reads and synchronizations are seen through the ``wp.*`` entry points Newton calls:
    ``wp.array.numpy()`` (which ``list()`` and ``str()`` read through) and ``wp.synchronize()``,
    ``synchronize_device()``, ``synchronize_stream()`` and ``synchronize_event()``. A call into
    ``warp._src`` that bypasses them is not seen. Inside :meth:`BroadPhaseExplicitTemplates.launch`,
    every Warp array that allocates memory is recorded too.
    """
    calls = []
    template_launches = []

    def recorded(name, function):
        def call(*args, **kwargs):
            if device.is_capturing:
                calls.append(name)
            return function(*args, **kwargs)

        return call

    array_init = wp.array.__init__
    template_launch = BroadPhaseExplicitTemplates.launch

    def init(array, *args, **kwargs):
        array_init(array, *args, **kwargs)
        # Arrays that own their memory carry its deleter; views and wrapped pointers do not.
        if template_launches and device.is_capturing and array.deleter is not None:
            calls.append(f"allocation in the template launch, shape {array.shape}")

    def launch(layout, *args, **kwargs):
        template_launches.append(layout)
        try:
            return template_launch(layout, *args, **kwargs)
        finally:
            template_launches.pop()

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(wp.array, "numpy", recorded("numpy", wp.array.numpy)))
        for name in ("synchronize", "synchronize_device", "synchronize_stream", "synchronize_event"):
            stack.enter_context(mock.patch.object(wp, name, recorded(name, getattr(wp, name))))
        stack.enter_context(mock.patch.object(wp.array, "__init__", init))
        stack.enter_context(mock.patch.object(BroadPhaseExplicitTemplates, "launch", launch))
        yield calls


def test_first_captured_collide_matches_eager(test, device):
    """Capture a pipeline's first collide(), templates on and off, and replay it like eager runs on live state.

    Each broad-phase mode is captured without a warm-up and reads nothing back from the device
    inside the capture; the world templates are verified and uploaded, and their kernel loaded,
    when the pipeline is built, so the captured template launch neither allocates nor loads a
    module (which drivers older than CUDA 12.3 refuse during capture).
    """
    model = _finalize(_replicated_builder(), device)
    layout = BroadPhaseExplicitTemplates(model._shape_contact_pair_templates, device=device)
    kernel = _pair_template_kernel(layout.staged_capacity, layout.threads_per_world, layout.worlds_per_block)
    for broad_phase, setting in itertools.product(("explicit", "nxn", "sap"), ("1", "0")):
        with test.subTest(broad_phase=broad_phase, templates=setting):
            # Earlier tests may have launched the same kernel: start unloaded to see where it loads.
            kernel.module.unload()
            state = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            # Growable contacts: their passes forecast, so a capture holds the forecasting variant as well.
            captured = _pipeline(model, setting, broad_phase=broad_phase, growable_contacts=True)
            eager = _pipeline(model, setting, broad_phase=broad_phase, growable_contacts=True)
            templated = broad_phase == "explicit" and setting == "1"
            record = _kernel_record(captured)
            if broad_phase == "explicit":
                test.assertEqual(record["kernel"], "world_templates" if templated else "flat_pairs")
            else:
                test.assertIsNone(record)
            # A pipeline that launches the template kernel loads it when built; the others leave it unloaded.
            loaded = set(kernel.module.execs)
            test.assertEqual(bool(loaded), templated)
            if templated:
                test.assertIsNotNone(captured._pair_templates.data)
            contacts = captured.contacts()
            with _recorded_while_capturing(wp.get_device(device)) as recorded:
                graph, launches = _launched(functools.partial(_capture_collide, captured, state, contacts))
            test.assertEqual(recorded, [])
            test.assertEqual(set(kernel.module.execs), loaded, "the capture loaded the template kernel")
            # A templated capture holds the kernel's plain and forecasting variants, one in each branch of the
            # conditional node that picks the forecasting pass.
            test.assertEqual(
                launches, {"templates": 2 * int(templated), "flat": int(broad_phase == "explicit" and not templated)}
            )
            eager_contacts = eager.contacts()
            counts = []
            for lift in (0.0, 3.0):
                if lift:
                    # Lift world 4 clear of the global box and its static shape: the replay must read the new poses.
                    body_q = state.body_q.numpy()
                    body_q[model.body_world.numpy() == 4, 2] += lift
                    state.body_q.assign(body_q)
                captured.broad_phase_shape_pairs.zero_()
                captured.broad_phase_pair_count.zero_()
                contacts.rigid_contact_count.zero_()
                wp.capture_launch(graph)
                eager.collide(state, eager_contacts)
                count = int(eager.broad_phase_pair_count.numpy()[0])
                test.assertGreater(count, 0)
                test.assertEqual(int(captured.broad_phase_pair_count.numpy()[0]), count)
                np.testing.assert_array_equal(
                    _sorted_codes(captured.broad_phase_shape_pairs.numpy(), count, count),
                    _sorted_codes(eager.broad_phase_shape_pairs.numpy(), count, count),
                )
                np.testing.assert_array_equal(_contact_rows(contacts), _contact_rows(eager_contacts))
                counts.append(count)
            test.assertLess(counts[1], counts[0])


class TestBroadPhaseTemplates(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
for _name, _function in (
    ("test_synthetic_templates_match_flat_kernel", test_synthetic_templates_match_flat_kernel),
    ("test_layout_shapes_match_flat_kernel", test_layout_shapes_match_flat_kernel),
    ("test_static_pairs_filtered_without_body_flags", test_static_pairs_filtered_without_body_flags),
    ("test_verification_rejects_a_changed_list", test_verification_rejects_a_changed_list),
    ("test_pipeline_templates_match_flat_contacts", test_pipeline_templates_match_flat_contacts),
    ("test_pipeline_immovable_filter_matches_flat", test_pipeline_immovable_filter_matches_flat),
    ("test_first_captured_collide_matches_eager", test_first_captured_collide_matches_eager),
):
    add_function_test(TestBroadPhaseTemplates, _name, _function, devices=devices)
add_function_test(
    TestBroadPhaseTemplates,
    "test_graph_replays_template_broad_phase",
    test_graph_replays_template_broad_phase,
    devices=get_cuda_test_devices(mode="basic"),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

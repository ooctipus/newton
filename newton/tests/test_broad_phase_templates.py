# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""World-template explicit broad phase: the builder's template record and the kernel that tests it.

The template kernel must test exactly the pairs of the flat explicit list. Counts
and sorted candidate sets are compared with :class:`BroadPhaseExplicit` on every
device; on CPU, where both run serially in list order, the arrays must be equal.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.geometry.broad_phase_nxn import (
    BroadPhaseExplicit,
    BroadPhaseExplicitTemplates,
    ShapePairTemplates,
    replay_shape_pair_templates,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


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
):
    """Random templates over worlds of ``world_shapes`` shapes, globals first, ``gap`` spare shapes between worlds.

    Args:
        template_specs: Per template, (global-local pair count, local pair count).
        empty_template: Add a template without pairs, which the list skips.

    Returns:
        (record, shape_count): the unbound record and the shapes it spans.
    """
    rng = np.random.default_rng(seed)
    templates = []
    for global_local_count, local_count in template_specs:
        local_pairs = set()
        while len(local_pairs) < local_count:
            a, b = sorted(rng.choice(world_shapes, size=2, replace=False).tolist())
            local_pairs.add((a, b))
        global_local = set()
        while len(global_local) < global_local_count:
            global_local.add((int(rng.integers(global_shapes)), int(rng.integers(world_shapes))))
        templates.append(
            (
                np.asarray(sorted(global_local), dtype=np.int32).reshape((-1, 2)),
                np.asarray(sorted(local_pairs), dtype=np.int32).reshape((-1, 2)),
            )
        )
    if empty_template:
        templates.append((np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)))
    starts = global_shapes + np.arange(worlds, dtype=np.int64) * (world_shapes + gap)
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
        global_list.add((a, b))
    record = ShapePairTemplates(
        global_pairs=np.asarray(sorted(global_list), dtype=np.int32).reshape((-1, 2)),
        templates=tuple(templates),
        runs=tuple((template, np.asarray(run, dtype=np.int32)) for template, run in runs),
    )
    return record, int(starts[-1] + world_shapes + gap) if worlds else global_shapes


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
        # (template specs, worlds, world shapes): small templates pack 8 worlds per block.
        (((3, 20), (0, 9), (2, 0)), 21, 12),
        # One large template: several warps per world, global block, tail block.
        (((5, 300),), 7, 40),
        # Many shapes per world with few pairs: shared memory limits worlds per block.
        (((40, 260), (0, 120)), 9, 300),
        # 1100 pairs per world: one 256-lane world per block, as on the largest templates.
        (((8, 1100),), 5, 60),
    )
    for case, (specs, worlds, world_shapes) in enumerate(cases):
        record, shape_count = _synthetic(
            case, worlds=worlds, world_shapes=world_shapes, template_specs=specs, empty_template=case == 0
        )
        pairs = wp.array(record.replay(), dtype=wp.vec2i, device=device)
        layout = BroadPhaseExplicitTemplates(record, device=device)
        test.assertIsNone(layout.unsupported_reason)
        test.assertTrue(layout.verify(pairs))
        test.assertEqual(layout.pair_count, pairs.shape[0])
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


def test_layout_shapes_launches(test, device):
    """Choose lanes per world from the largest template and worlds per block from shared memory."""
    expectations = (
        (23, 12, 32, 8),  # one warp per world, eight worlds per block
        (305, 41, 64, 4),  # two warps per world
        (300, 300, 64, 3),  # 300 staged shapes leave shared memory for three worlds
        (1108, 61, 256, 1),  # one 256-lane world per block, as on the largest templates
    )
    for pair_count, staged, threads, worlds_per_block in expectations:
        layout = BroadPhaseExplicitTemplates(_exact_template(pair_count, staged), device=device)
        with test.subTest(pairs=pair_count, staged=staged):
            test.assertEqual((layout.pairs_per_world, layout.staged_shapes), (pair_count, staged))
            test.assertEqual(layout.threads_per_world, threads)
            test.assertEqual(layout.worlds_per_block, worlds_per_block)


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


class TestBroadPhaseTemplateRule(unittest.TestCase):
    def _single(self, pairs_per_world, worlds, staged=None):
        staged = pairs_per_world + 1 if staged is None else staged
        local = np.stack((np.zeros(pairs_per_world), 1 + np.arange(pairs_per_world) % (staged - 1)), axis=1)
        record = ShapePairTemplates(
            global_pairs=np.empty((0, 2), dtype=np.int32),
            templates=((np.empty((0, 2), dtype=np.int32), local.astype(np.int32)),),
            runs=((0, (np.arange(worlds) * staged).astype(np.int32)),),
        )
        return BroadPhaseExplicitTemplates(record, device="cpu")

    def test_automatic_rule_admits_large_templates(self):
        """Admit automatically only at 2**20 pairs or more and 64 or more pairs per world."""
        self.assertTrue(self._single(64, 16384).automatic)
        self.assertFalse(self._single(63, 16645).automatic)
        self.assertFalse(self._single(64, 16383).automatic)
        self.assertEqual(self._single(64, 16384).pair_count, 1 << 20)

    def test_staging_capacity_bounds_admission(self):
        """Refuse templates referencing more shapes than a world can stage."""
        self.assertIsNone(self._single(1023, 2).unsupported_reason)
        too_many = self._single(1024, 2)
        self.assertIn("1025 shapes", too_many.unsupported_reason)
        self.assertFalse(too_many.automatic)
        self.assertFalse(too_many.verify(wp.zeros(too_many.pair_count, dtype=wp.vec2i, device="cpu")))


class TestBroadPhaseTemplates(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
for _name, _function in (
    ("test_synthetic_templates_match_flat_kernel", test_synthetic_templates_match_flat_kernel),
    ("test_layout_shapes_launches", test_layout_shapes_launches),
    ("test_verification_rejects_a_changed_list", test_verification_rejects_a_changed_list),
):
    add_function_test(TestBroadPhaseTemplates, _name, _function, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

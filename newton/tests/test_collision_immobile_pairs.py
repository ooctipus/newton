# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop candidate pairs whose shapes are both structurally immobile, and keep every other pair."""

import json
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import BodyFlags, ModelFlags
from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from newton._src.geometry.broad_phase_sap import BroadPhaseSAP
from newton.tests.unittest_utils import add_function_test, get_test_devices

_FLAG = "NEWTON_BROAD_PHASE_IMMOBILE_FILTER"
_MODES = ("explicit", "nxn", "sap")
# Every broad-phase launch the pipeline can select, as a mode and the environment selecting it.
# NEWTON_BROAD_PHASE_TEMPLATES=1 selects a world-template kernel for explicit pairs where one
# exists, which must apply the filter as well; elsewhere the variable is recorded as unknown.
_LAUNCHES = (("explicit", {}), ("explicit", {"NEWTON_BROAD_PHASE_TEMPLATES": "1"}), ("nxn", {}), ("sap", {}))
_WORLDS = 2
# Roles whose shapes cannot move: static, welded to the world, or welded through a fixed chain.
_IMMOBILE = frozenset(("ground", "table", "base", "mount", "post"))


@contextmanager
def _flag(value):
    """Run with the immobile-pair flag set to ``value``, or unset for ``None``."""
    with patch.dict(os.environ):
        os.environ.pop(_FLAG, None)
        if value is not None:
            os.environ[_FLAG] = value
        yield


def _pipeline(model, flag=None, **kwargs):
    """Construct a collision pipeline with the immobile-pair flag set to ``flag``, or unset for ``None``."""
    with _flag(flag):
        return newton.CollisionPipeline(model, **kwargs)


def _box(builder, body, x, y, z=0.05, half=0.05):
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    if body < 0:
        return builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()), hx=half, hy=half, hz=half, cfg=cfg
        )
    return builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)


def _pose(x, y, z=0.05):
    return wp.transform(wp.vec3(x, y, z), wp.quat_identity())


def _pair_scene(device):
    """Build one world of every shape role on a global ground and replicate it, as scene builders do.

    Returns:
        The model and, per world, the shape index of each role. Every role rests on the
        ground; ``base`` overlaps ``post``, and ``kinematic`` overlaps ``free``.
    """
    proto = newton.ModelBuilder()
    proto.rigid_gap = 0.01
    roles = {}

    def link(role, x, y, *, kinematic=False):
        body = proto.add_link(xform=_pose(x, y), is_kinematic=kinematic, label=role)
        roles[role] = _box(proto, body, x, y)
        return body

    roles["table"] = _box(proto, -1, 0.6, 0.0, z=0.1, half=0.1)
    base = link("base", 0.0, 0.0)
    mount = link("mount", 0.0, 0.3)
    proto.add_articulation(
        [
            proto.add_joint_fixed(-1, base, parent_xform=_pose(0.0, 0.0)),
            proto.add_joint_fixed(base, mount, parent_xform=_pose(0.0, 0.3, 0.0)),
        ]
    )
    post = link("post", 0.09, 0.0)
    proto.add_articulation([proto.add_joint_fixed(-1, post, parent_xform=_pose(0.09, 0.0))])
    arm = link("arm", -0.6, 0.0)
    proto.add_articulation([proto.add_joint_revolute(-1, arm, axis=(0.0, 0.0, 1.0), parent_xform=_pose(-0.6, 0.0))])
    held = link("welded_kinematic", -0.3, 0.6, kinematic=True)
    carried = link("on_kinematic", -0.3, 0.9)
    proto.add_articulation(
        [
            proto.add_joint_fixed(-1, held, parent_xform=_pose(-0.3, 0.6)),
            proto.add_joint_fixed(held, carried, parent_xform=_pose(0.0, 0.3, 0.0)),
        ]
    )
    loose = link("loose", 0.3, 0.6)
    proto.add_articulation([proto.add_joint_fixed(-1, loose, parent_xform=_pose(0.3, 0.6), enabled=False)])
    roles["free"] = _box(proto, proto.add_body(xform=_pose(0.0, -0.6), label="free"), 0.0, -0.6)
    kinematic = proto.add_body(xform=_pose(0.09, -0.6), is_kinematic=True, label="kinematic")
    roles["kinematic"] = _box(proto, kinematic, 0.09, -0.6)

    scene = newton.ModelBuilder()
    ground = scene.add_ground_plane()
    shape_start = scene.shape_count
    scene.replicate(proto, _WORLDS, spacing=(3.0, 0.0, 0.0))
    model = scene.finalize(device=device)
    worlds = [
        {"ground": ground, **{role: shape_start + world * proto.shape_count + index for role, index in roles.items()}}
        for world in range(_WORLDS)
    ]
    return model, worlds


def _role_of(worlds):
    return {shape: role for world in worlds for role, shape in world.items()}


def _candidate_pairs(pipeline, model):
    """Collide once from the model's initial state and return the broad phase's candidate pairs in order."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(pipeline.broad_phase_pair_count.numpy()[0])
    return [tuple(pair) for pair in pipeline.broad_phase_shape_pairs.numpy()[:count].tolist()], contacts


def _unfiltered_candidates(pipeline, model, mode):
    """Launch the pipeline's broad phase on its current AABBs with the arguments collide() passed before the filter."""
    device = model.device
    pairs = wp.zeros(pipeline.shape_pairs_max, dtype=wp.vec2i, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    lower, upper = pipeline.narrow_phase.shape_aabb_lower, pipeline.narrow_phase.shape_aabb_upper
    unfiltered = {
        "shape_body": model.shape_body,
        "body_flags": model.body_flags,
        "include_static_kinematic_pairs": pipeline.include_static_kinematic_pairs,
    }
    if mode == "explicit":
        BroadPhaseExplicit().launch(
            lower, upper, None, pipeline.shape_pairs_filtered, len(pipeline.shape_pairs_filtered), pairs, count,
            device=device, **unfiltered,
        )  # fmt: skip
    else:
        broad_phase = (BroadPhaseAllPairs if mode == "nxn" else BroadPhaseSAP)(
            model.shape_world, shape_flags=model.shape_flags, device=device
        )
        broad_phase.launch(
            lower, upper, None, model.shape_collision_group, model.shape_world, model.shape_count, pairs, count,
            device=device, filter_pairs=pipeline.shape_pairs_excluded,
            num_filter_pairs=pipeline.shape_pairs_excluded_count, **unfiltered,
        )  # fmt: skip
    return [tuple(pair) for pair in pairs.numpy()[: int(count.numpy()[0])].tolist()]


def _ordered(pairs, device):
    """Pair order is the input order on CPU and depends on scheduling on CUDA."""
    return pairs if wp.get_device(device).is_cpu else sorted(pairs)


def test_broad_phases_drop_only_immobile_pairs(test, device):
    """Drop exactly the pairs of two static or welded shapes in every broad phase, by default."""
    model, worlds = _pair_scene(device)
    role = _role_of(worlds)
    for mode, environment in _LAUNCHES:
        with test.subTest(broad_phase=mode, **environment), patch.dict(os.environ, environment):
            kept, _ = _candidate_pairs(_pipeline(model, broad_phase=mode), model)
            unfiltered, _ = _candidate_pairs(_pipeline(model, "0", broad_phase=mode), model)
            immobile = [pair for pair in unfiltered if role[pair[0]] in _IMMOBILE and role[pair[1]] in _IMMOBILE]
            # The remaining pairs keep their candidate order on CPU.
            test.assertEqual(
                _ordered(kept, device), _ordered([pair for pair in unfiltered if pair not in immobile], device)
            )
            for world in worlds:
                dropped = {tuple(sorted((world[a], world[b]))) for a, b in (
                    ("ground", "base"),  # welded to the world, on a static shape
                    ("ground", "mount"),  # welded through a chain of two fixed joints
                    ("ground", "table"),  # two static shapes of different builders
                    ("base", "post"),  # two welded bodies
                )}  # fmt: skip
                retained = {tuple(sorted((world[a], world[b]))) for a, b in (
                    ("ground", "kinematic"),  # a kinematic body can move
                    ("kinematic", "free"),
                    ("ground", "free"),  # free root
                    ("ground", "arm"),  # revolute root
                    ("ground", "welded_kinematic"),  # welded but kinematic
                    ("ground", "on_kinematic"),  # welded to a kinematic body
                    ("ground", "loose"),  # its fixed joint is disabled
                )}  # fmt: skip
                test.assertLessEqual(dropped, set(unfiltered))
                test.assertLessEqual(retained, set(unfiltered))
                test.assertFalse(dropped & set(kept))
                test.assertLessEqual(retained, set(kept))


def test_opt_out_keeps_the_unfiltered_broad_phase(test, device):
    """Restore the previous candidates with the flag at 0 or include_immobile_pairs=True, in every broad phase."""
    model, _ = _pair_scene(device)
    for mode in _MODES:
        for label, flag, argument in (("environment", "0", None), ("argument", None, True), ("both", "1", True)):
            with test.subTest(broad_phase=mode, opt_out=label), _flag(flag):
                options = {} if argument is None else {"include_immobile_pairs": argument}
                pipeline = newton.CollisionPipeline(model, broad_phase=mode, **options)
                pairs, _ = _candidate_pairs(pipeline, model)
                test.assertEqual(
                    _ordered(pairs, device), _ordered(_unfiltered_candidates(pipeline, model, mode), device)
                )
                test.assertFalse(pipeline.get_collision_selection()["immobile_pairs"]["enabled"])


def test_static_kinematic_filter_composes(test, device):
    """Drop kinematic pairs as well when include_static_kinematic_pairs=False, and keep dynamic ones."""
    model, worlds = _pair_scene(device)
    role = _role_of(worlds)
    movable = {"arm", "free", "on_kinematic", "loose"}
    for mode in _MODES:
        with test.subTest(broad_phase=mode):
            kept, _ = _candidate_pairs(_pipeline(model, broad_phase=mode, include_static_kinematic_pairs=False), model)
            unfiltered, _ = _candidate_pairs(_pipeline(model, "0", broad_phase=mode), model)
            expected = [pair for pair in unfiltered if role[pair[0]] in movable or role[pair[1]] in movable]
            test.assertEqual(_ordered(kept, device), _ordered(expected, device))


def _welded_against_kinematic(device, *, ground):
    """Replicate a welded box that overlaps a kinematic box, optionally with a global ground far below both.

    Returns:
        The model and, per world, the shape pair of the two boxes.
    """
    proto = newton.ModelBuilder()
    base = proto.add_link(xform=_pose(0.0, 0.0, 1.0), label="base")
    base_shape = _box(proto, base, 0.0, 0.0)
    proto.add_articulation([proto.add_joint_fixed(-1, base, parent_xform=_pose(0.0, 0.0, 1.0))])
    kinematic = proto.add_body(xform=_pose(0.09, 0.0, 1.0), is_kinematic=True, label="kinematic")
    kinematic_shape = _box(proto, kinematic, 0.09, 0.0)
    scene = newton.ModelBuilder()
    if ground:
        scene.add_ground_plane()
    shape_start = scene.shape_count
    scene.replicate(proto, _WORLDS, spacing=(3.0, 0.0, 0.0))
    offsets = [shape_start + world * proto.shape_count for world in range(_WORLDS)]
    return scene.finalize(device=device), [(offset + base_shape, offset + kinematic_shape) for offset in offsets]


def test_static_kinematic_composition_ignores_unrelated_pairs(test, device):
    """Drop welded-kinematic pairs under include_static_kinematic_pairs=False whether or not a ground is listed."""
    for ground in (False, True):
        model, welded_kinematic = _welded_against_kinematic(device, ground=ground)
        for mode in _MODES:
            with test.subTest(ground=ground, broad_phase=mode), _flag(None):
                pipeline = newton.CollisionPipeline(model, broad_phase=mode, include_static_kinematic_pairs=False)
                pairs, _ = _candidate_pairs(pipeline, model)
                test.assertFalse(set(welded_kinematic) & set(pairs))
                record = pipeline.get_collision_selection()["immobile_pairs"]
                test.assertTrue(record["enabled"])
                # Each world adds its welded-kinematic pair to the static-kinematic filter, and with
                # a ground its welded-static pair; the static-kinematic pairs were dropped before.
                test.assertEqual(record["listed_pairs"], _WORLDS if ground else 0)
                test.assertEqual(record["excluded_pairs"], (2 if ground else 1) * _WORLDS)
                kept = newton.CollisionPipeline(
                    model, broad_phase=mode, include_static_kinematic_pairs=False, include_immobile_pairs=True
                )
                test.assertLessEqual(set(welded_kinematic), set(_candidate_pairs(kept, model)[0]))


def test_record_reports_excluded_pairs(test, device):
    """Report the choice, its source and the welded bodies, immobile shapes and excluded pairs, as launched."""
    model, worlds = _pair_scene(device)
    role = _role_of(worlds)
    # Base, mount and post of each world are welded; the ground and every table are static.
    welded, shapes = 3 * _WORLDS, 1 + 4 * _WORLDS
    # Listed pairs count whether or not they overlap; every mode lists the model's pairs here.
    listed = model.shape_contact_pairs.numpy().tolist()
    pairs = sum(role.get(a) in _IMMOBILE and role.get(b) in _IMMOBILE for a, b in listed)
    test.assertGreater(pairs, 4 * _WORLDS)
    for mode, environment in _LAUNCHES:
        for flag, argument, source, enabled, reason in (
            (None, None, "default", True, "Default: "),
            ("1", None, "environment", True, f"{_FLAG}=1: "),
            ("0", None, "environment", False, f"Off: {_FLAG}=0 "),
            ("0", False, "argument", True, "include_immobile_pairs=False: "),
            (None, True, "argument", False, "Off: include_immobile_pairs=True "),
        ):
            subtest = test.subTest(broad_phase=mode, flag=flag, argument=argument, **environment)
            with subtest, patch.dict(os.environ, environment), _flag(flag):
                pipeline = newton.CollisionPipeline(model, broad_phase=mode, include_immobile_pairs=argument)
                # The launch agrees with the record: overlapping immobile pairs reach it only when off.
                candidates, _ = _candidate_pairs(pipeline, model)
                joined = [
                    pair for pair in candidates if role.get(pair[0]) in _IMMOBILE and role.get(pair[1]) in _IMMOBILE
                ]
                test.assertEqual(not joined, enabled)
                record = pipeline.get_collision_selection()["immobile_pairs"]
                test.assertEqual(
                    {key: value for key, value in record.items() if key != "reason"},
                    {
                        "enabled": enabled,
                        "source": source,
                        "welded_bodies": welded,
                        "immobile_shapes": shapes,
                        "listed_pairs": pairs,
                        "excluded_pairs": pairs if enabled else 0,
                    },
                )
                test.assertTrue(record["reason"].startswith(reason), record["reason"])
                json.dumps(record)
                # The record is a copy.
                record["enabled"] = not enabled
                test.assertEqual(pipeline.get_collision_selection()["immobile_pairs"]["enabled"], enabled)


def test_long_explicit_lists_count_every_block(test, device):
    """Count the immobile pairs of an explicit list that spans several scan blocks, as of a shorter copy."""
    model, _ = _pair_scene(device)
    listed = model.shape_contact_pairs.numpy()
    # The scan reads 65536 pairs at a time; end the tiled list inside a block.
    tiles = 65536 // len(listed) + 2
    tiled = wp.array(np.tile(listed, (tiles, 1)), dtype=wp.vec2i, device=device)
    test.assertGreater(len(tiled), 65536)
    for include_static_kinematic_pairs in (True, False):
        with test.subTest(include_static_kinematic_pairs=include_static_kinematic_pairs), _flag(None):
            options = {"include_static_kinematic_pairs": include_static_kinematic_pairs}
            single = newton.CollisionPipeline(model, **options).get_collision_selection()["immobile_pairs"]
            pipeline = newton.CollisionPipeline(model, shape_pairs_filtered=tiled, **options)
            record = pipeline.get_collision_selection()["immobile_pairs"]
            test.assertGreater(single["excluded_pairs"], 0)
            test.assertEqual(record["listed_pairs"], tiles * single["listed_pairs"])
            test.assertEqual(record["excluded_pairs"], tiles * single["excluded_pairs"])


def test_models_without_immobile_pairs_keep_unfiltered_arguments(test, device):
    """Leave the broad phase unfiltered when no candidate pair can join two immobile shapes."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    base = builder.add_link(xform=_pose(0.0, 0.0), label="base")
    _box(builder, base, 0.0, 0.0)
    # The builder filters a fixed root against the world's shapes, so no immobile pair is listed.
    builder.add_articulation([builder.add_joint_fixed(-1, base, parent_xform=_pose(0.0, 0.0))])
    _box(builder, builder.add_body(xform=_pose(0.3, 0.0), label="free"), 0.3, 0.0)
    model = builder.finalize(device=device)
    for mode in _MODES:
        with test.subTest(broad_phase=mode), _flag(None):
            pipeline = newton.CollisionPipeline(model, broad_phase=mode)
            record = pipeline.get_collision_selection()["immobile_pairs"]
            test.assertFalse(record["enabled"])
            test.assertEqual(record["listed_pairs"], 0)
            test.assertEqual(
                record["reason"], "No candidate pair can join two shapes that are static or welded to the world."
            )
            pairs, _ = _candidate_pairs(pipeline, model)
            test.assertEqual(_ordered(pairs, device), _ordered(_unfiltered_candidates(pipeline, model, mode), device))


def test_every_parent_joint_must_weld(test, device):
    """Keep a fixed root mobile when a loop joint outside its tree also names it as the child."""
    for loop in (False, True):
        proto = newton.ModelBuilder()
        base = proto.add_link(xform=_pose(0.0, 0.0), label="base")
        base_shape = _box(proto, base, 0.0, 0.0)
        proto.add_articulation([proto.add_joint_fixed(-1, base, parent_xform=_pose(0.0, 0.0))])
        arm = proto.add_link(xform=_pose(0.0, 0.5), label="arm")
        _box(proto, arm, 0.0, 0.5)
        proto.add_articulation([proto.add_joint_revolute(-1, arm, axis=(0.0, 0.0, 1.0), parent_xform=_pose(0.0, 0.5))])
        if loop:
            proto.add_joint_revolute(arm, base, axis=(0.0, 0.0, 1.0), parent_xform=_pose(0.0, -0.5, 0.0))
        scene = newton.ModelBuilder()
        ground = scene.add_ground_plane()
        shape_start = scene.shape_count
        scene.replicate(proto, _WORLDS, spacing=(3.0, 0.0, 0.0))
        model = scene.finalize(device=device)
        ground_base = {(ground, shape_start + world * proto.shape_count + base_shape) for world in range(_WORLDS)}
        for mode in _MODES:
            with test.subTest(loop=loop, broad_phase=mode), _flag(None):
                pipeline = newton.CollisionPipeline(model, broad_phase=mode)
                pairs, _ = _candidate_pairs(pipeline, model)
                record = pipeline.get_collision_selection()["immobile_pairs"]
                if loop:
                    test.assertLessEqual(ground_base, set(pairs))
                    test.assertEqual(record["welded_bodies"], 0)
                else:
                    test.assertFalse(ground_base & set(pairs))
                    test.assertEqual(record["welded_bodies"], _WORLDS)


def test_notifications_refresh_the_immobile_set(test, device):
    """Follow body-flag and joint-enable changes once the pipeline is notified."""
    model, worlds = _pair_scene(device)
    world = worlds[0]
    ground_base = tuple(sorted((world["ground"], world["base"])))
    ground_mount = tuple(sorted((world["ground"], world["mount"])))
    ground_loose = tuple(sorted((world["ground"], world["loose"])))
    base_body = int(model.shape_body.numpy()[world["base"]])
    loose_joint = int(np.flatnonzero(model.joint_child.numpy() == model.shape_body.numpy()[world["loose"]])[0])
    for mode in _MODES:
        with test.subTest(broad_phase=mode), _flag(None):
            pipeline = newton.CollisionPipeline(model, broad_phase=mode)
            # numpy() aliases a CPU array; keep copies to restore from.
            flags = model.body_flags.numpy().copy()
            enabled = model.joint_enabled.numpy().copy()
            try:
                pairs, _ = _candidate_pairs(pipeline, model)
                test.assertNotIn(ground_base, pairs)
                test.assertIn(ground_loose, pairs)

                # A kinematic base can move, and so can the mount welded to it.
                changed = flags.copy()
                changed[base_body] = int(BodyFlags.KINEMATIC)
                model.body_flags.assign(changed)
                pipeline.notify_model_changed(ModelFlags.BODY_PROPERTIES)
                pairs, _ = _candidate_pairs(pipeline, model)
                test.assertIn(ground_base, pairs)
                test.assertIn(ground_mount, pairs)
                test.assertEqual(pipeline.get_collision_selection()["immobile_pairs"]["welded_bodies"], 3 * _WORLDS - 2)

                # Re-enabling the loose joint welds its body.
                changed_enabled = enabled.copy()
                changed_enabled[loose_joint] = True
                model.joint_enabled.assign(changed_enabled)
                pipeline.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
                pairs, _ = _candidate_pairs(pipeline, model)
                test.assertNotIn(ground_loose, pairs)

                model.body_flags.assign(flags)
                model.joint_enabled.assign(enabled)
                pipeline.notify_model_changed(ModelFlags.BODY_PROPERTIES | ModelFlags.JOINT_PROPERTIES)
                pairs, _ = _candidate_pairs(pipeline, model)
                test.assertNotIn(ground_base, pairs)
                test.assertIn(ground_loose, pairs)
                test.assertEqual(pipeline.get_collision_selection()["immobile_pairs"]["welded_bodies"], 3 * _WORLDS)
            finally:
                model.body_flags.assign(flags)
                model.joint_enabled.assign(enabled)


def test_unchanged_notifications_skip_the_refresh(test, device):
    """Rewrite the immobile set only after a notified array changed, reading only the notified arrays."""
    model, worlds = _pair_scene(device)
    base_body = int(model.shape_body.numpy()[worlds[0]["base"]])
    flags = model.body_flags.numpy().copy()
    with _flag(None):
        pipeline = newton.CollisionPipeline(model)
    mask = pipeline._immobile_shape_body
    try:
        with patch.object(mask, "assign", wraps=mask.assign) as assign:
            pipeline.notify_model_changed(ModelFlags.BODY_PROPERTIES | ModelFlags.JOINT_PROPERTIES)
            test.assertEqual(assign.call_count, 0)
            changed = flags.copy()
            changed[base_body] = int(BodyFlags.KINEMATIC)
            model.body_flags.assign(changed)
            # A joint notification re-reads the joint enables only.
            pipeline.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
            test.assertEqual(assign.call_count, 0)
            test.assertEqual(pipeline.get_collision_selection()["immobile_pairs"]["welded_bodies"], 3 * _WORLDS)
            pipeline.notify_model_changed(ModelFlags.BODY_PROPERTIES)
            test.assertEqual(assign.call_count, 1)
            test.assertEqual(pipeline.get_collision_selection()["immobile_pairs"]["welded_bodies"], 3 * _WORLDS - 2)
    finally:
        model.body_flags.assign(flags)


def test_captured_collide_reads_notified_set(test, device):
    """Replay a captured collide with the immobile set of the latest notification, and refuse one in capture."""
    model, worlds = _pair_scene(device)
    world = worlds[0]
    ground_base = tuple(sorted((world["ground"], world["base"])))
    base_body = int(model.shape_body.numpy()[world["base"]])
    flags = model.body_flags.numpy().copy()
    with _flag(None):
        pipeline = newton.CollisionPipeline(model)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    try:
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(state, contacts)
            with test.assertRaises(RuntimeError):
                pipeline.notify_model_changed(ModelFlags.BODY_PROPERTIES)

        def replayed_pairs():
            wp.capture_launch(capture.graph)
            count = int(pipeline.broad_phase_pair_count.numpy()[0])
            return {tuple(pair) for pair in pipeline.broad_phase_shape_pairs.numpy()[:count].tolist()}

        test.assertNotIn(ground_base, replayed_pairs())
        changed = flags.copy()
        changed[base_body] = int(BodyFlags.KINEMATIC)
        model.body_flags.assign(changed)
        pipeline.notify_model_changed(ModelFlags.BODY_PROPERTIES)
        test.assertIn(ground_base, replayed_pairs())
    finally:
        model.body_flags.assign(flags)


def test_environment_flag_values(test, device):
    """Reject flag values other than 0 and 1, unless include_immobile_pairs supersedes the flag."""
    model, _ = _pair_scene(device)
    with _flag("2"), test.assertRaisesRegex(ValueError, _FLAG):
        newton.CollisionPipeline(model)
    for argument in (True, False):
        with test.subTest(argument=argument), _flag("2"):
            selection = newton.CollisionPipeline(model, include_immobile_pairs=argument).get_collision_selection()
            test.assertEqual(selection["environment"]["overridden"].get(_FLAG), "2")
            test.assertNotIn(_FLAG, selection["environment"]["applied"])
            test.assertEqual(selection["immobile_pairs"]["enabled"], not argument)
    with _flag("1"):
        selection = newton.CollisionPipeline(model).get_collision_selection()
        test.assertEqual(selection["environment"]["applied"].get(_FLAG), "1")
        test.assertNotIn(_FLAG, selection["environment"]["overridden"])


def _arm_scene(device):
    """A fixed-base arm whose base rests in the ground and whose link presses a free cube, in two worlds."""
    proto = newton.ModelBuilder()
    proto.rigid_gap = 0.01
    base_pose = wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity())
    base = proto.add_link(xform=base_pose, label="base")
    proto.add_shape_box(base, hx=0.1, hy=0.1, hz=0.05)
    root = proto.add_joint_fixed(-1, base, parent_xform=base_pose)
    arm = proto.add_link(label="arm")
    proto.add_shape_box(arm, hx=0.25, hy=0.03, hz=0.02)
    hinge = proto.add_joint_revolute(
        base,
        arm,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.27, 0.0, 0.0), wp.quat_identity()),
        target_ke=50.0,
        target_kd=2.0,
        target_pos=0.15,
    )
    proto.add_articulation([root, hinge])
    cube = proto.add_body(xform=wp.transform(wp.vec3(0.42, 0.0, 0.05), wp.quat_identity()), label="cube")
    proto.add_shape_box(cube, hx=0.05, hy=0.05, hz=0.05)
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(proto, _WORLDS, spacing=(2.0, 0.0, 0.0))
    return scene.finalize(device=device)


def _arm_trajectory(model, *, flag, friction_anchor_beta, steps=60):
    """Step the arm scene; return its joint trajectory, base-ground and arm-cube contact counts and friction law."""
    solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=friction_anchor_beta)
    law = solver.get_path_selection()["routes"]["friction"]["law"]
    # Deterministic contacts keep the remaining contacts in one order on CUDA as well.
    pipeline = _pipeline(model, flag, deterministic=model.device.is_cuda)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    shape_body = model.shape_body.numpy()
    trajectory, static_contacts, arm_contacts = [], 0, 0
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        bodies = (
            shape_body[contacts.rigid_contact_shape0.numpy()[:count]],
            shape_body[contacts.rigid_contact_shape1.numpy()[:count]],
        )
        # Body 0 of each world is the welded base, body 1 the arm and body 2 the cube.
        local = [np.where(body < 0, -1, body % 3) for body in bodies]
        static_contacts += int(
            np.count_nonzero(((local[0] == -1) & (local[1] == 0)) | ((local[0] == 0) & (local[1] == -1)))
        )
        arm_contacts += int(np.count_nonzero(((local[0] == 1) & (local[1] == 2)) | ((local[0] == 2) & (local[1] == 1))))
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
        trajectory.append(np.concatenate((state_0.joint_q.numpy(), state_0.joint_qd.numpy())))
    return np.asarray(trajectory), static_contacts, arm_contacts, law


def test_fixed_base_arm_trajectories_match(test, device):
    """Step FeatherPGS alike without the base's ground contacts, under point and patch friction.

    CPU kernels run serially, so the remaining contacts keep their order and the trajectories
    are bitwise equal there. CUDA reserves contact rows with per-world atomics, whose order can
    follow the dropped contacts, so on CUDA they agree within ten times the spread of two
    identical runs.
    """
    model = _arm_scene(device)
    for law, beta in (("point", 0.0), ("patch", 0.2)):
        with test.subTest(friction=law):
            reference, reference_static, reference_arm, reference_law = _arm_trajectory(
                model, flag="0", friction_anchor_beta=beta
            )
            trajectory, static, arm, selected_law = _arm_trajectory(model, flag=None, friction_anchor_beta=beta)
            test.assertEqual((reference_law, selected_law), (law, law))
            # The base rests in the ground and the arm presses the cube.
            test.assertGreater(reference_static, 0)
            test.assertEqual(static, 0)
            test.assertGreater(arm, 0)
            test.assertEqual(arm, reference_arm)
            test.assertTrue(np.all(np.isfinite(trajectory)))
            if wp.get_device(device).is_cpu:
                np.testing.assert_array_equal(trajectory, reference)
            else:
                repeat = _arm_trajectory(model, flag="0", friction_anchor_beta=beta)[0]
                spread = float(np.max(np.abs(repeat - reference)))
                np.testing.assert_allclose(trajectory, reference, rtol=0.0, atol=max(1.0e-5, 10.0 * spread))


class TestCollisionImmobilePairs(unittest.TestCase):
    pass


devices = get_test_devices()
for name, function in (
    ("test_broad_phases_drop_only_immobile_pairs", test_broad_phases_drop_only_immobile_pairs),
    ("test_opt_out_keeps_the_unfiltered_broad_phase", test_opt_out_keeps_the_unfiltered_broad_phase),
    ("test_static_kinematic_filter_composes", test_static_kinematic_filter_composes),
    (
        "test_static_kinematic_composition_ignores_unrelated_pairs",
        test_static_kinematic_composition_ignores_unrelated_pairs,
    ),
    ("test_record_reports_excluded_pairs", test_record_reports_excluded_pairs),
    ("test_long_explicit_lists_count_every_block", test_long_explicit_lists_count_every_block),
    (
        "test_models_without_immobile_pairs_keep_unfiltered_arguments",
        test_models_without_immobile_pairs_keep_unfiltered_arguments,
    ),
    ("test_every_parent_joint_must_weld", test_every_parent_joint_must_weld),
    ("test_notifications_refresh_the_immobile_set", test_notifications_refresh_the_immobile_set),
    ("test_unchanged_notifications_skip_the_refresh", test_unchanged_notifications_skip_the_refresh),
    # Warp captures graphs on CPU devices as well.
    ("test_captured_collide_reads_notified_set", test_captured_collide_reads_notified_set),
    ("test_environment_flag_values", test_environment_flag_values),
    ("test_fixed_base_arm_trajectories_match", test_fixed_base_arm_trajectories_match),
):
    add_function_test(TestCollisionImmobilePairs, name, function, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

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
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_test_devices

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
        return newton.CollisionPipeline(model, **kwargs, growable_contacts=True)


def _box(builder, body, x, y, z=0.05, half=0.05):
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    if body < 0:
        return builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()), hx=half, hy=half, hz=half, cfg=cfg
        )
    return builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)


def _pose(x, y, z=0.05):
    return wp.transform(wp.vec3(x, y, z), wp.quat_identity())


def _pair_scene(device, *, locked_d6=False):
    """Build one world of every shape role on a global ground and replicate it, as scene builders do.

    Args:
        device: The device to finalize the model on.
        locked_d6: Use D6 joints instead: one without axes for every fixed joint, and one with
            the revolute joint's axis for the revolute joint.

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

    def weld(parent, child, **kwargs):
        if locked_d6:
            # Filter the parent's shapes as a fixed joint does, so that both scenes list the same pairs.
            return proto.add_joint_d6(parent, child, collision_filter_parent=True, **kwargs)
        return proto.add_joint_fixed(parent, child, **kwargs)

    roles["table"] = _box(proto, -1, 0.6, 0.0, z=0.1, half=0.1)
    base = link("base", 0.0, 0.0)
    mount = link("mount", 0.0, 0.3)
    proto.add_articulation(
        [weld(-1, base, parent_xform=_pose(0.0, 0.0)), weld(base, mount, parent_xform=_pose(0.0, 0.3, 0.0))]
    )
    post = link("post", 0.09, 0.0)
    proto.add_articulation([weld(-1, post, parent_xform=_pose(0.09, 0.0))])
    arm = link("arm", -0.6, 0.0)
    if locked_d6:
        hinge = newton.ModelBuilder.JointDofConfig(axis=(0.0, 0.0, 1.0))
        root = proto.add_joint_d6(-1, arm, angular_axes=[hinge], parent_xform=_pose(-0.6, 0.0))
    else:
        root = proto.add_joint_revolute(-1, arm, axis=(0.0, 0.0, 1.0), parent_xform=_pose(-0.6, 0.0))
    proto.add_articulation([root])
    held = link("welded_kinematic", -0.3, 0.6, kinematic=True)
    carried = link("on_kinematic", -0.3, 0.9)
    proto.add_articulation(
        [weld(-1, held, parent_xform=_pose(-0.3, 0.6)), weld(held, carried, parent_xform=_pose(0.0, 0.3, 0.0))]
    )
    loose = link("loose", 0.3, 0.6)
    proto.add_articulation([weld(-1, loose, parent_xform=_pose(0.3, 0.6), enabled=False)])
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
                    ("base", "post"),  # two welded bodies
                )}  # fmt: skip
                # Two static shapes of different builders never pair: every broad phase drops static pairs.
                static = tuple(sorted((world["ground"], world["table"])))
                test.assertNotIn(static, set(unfiltered))
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
                pipeline = newton.CollisionPipeline(model, broad_phase=mode, **options, growable_contacts=True)
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
                pipeline = newton.CollisionPipeline(
                    model, broad_phase=mode, include_static_kinematic_pairs=False, growable_contacts=True
                )
                pairs, _ = _candidate_pairs(pipeline, model)
                test.assertFalse(set(welded_kinematic) & set(pairs))
                record = pipeline.get_collision_selection()["immobile_pairs"]
                test.assertTrue(record["enabled"])
                # Each world adds its welded-kinematic pair to the static-kinematic filter, and with
                # a ground its welded-static pair; the static-kinematic pairs were dropped before.
                test.assertEqual(record["listed_pairs"], _WORLDS if ground else 0)
                test.assertEqual(record["excluded_pairs"], (2 if ground else 1) * _WORLDS)
                kept = newton.CollisionPipeline(
                    model,
                    broad_phase=mode,
                    include_static_kinematic_pairs=False,
                    include_immobile_pairs=True,
                    growable_contacts=True,
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
                pipeline = newton.CollisionPipeline(
                    model, broad_phase=mode, include_immobile_pairs=argument, growable_contacts=True
                )
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
            single = newton.CollisionPipeline(model, **options, growable_contacts=True).get_collision_selection()[
                "immobile_pairs"
            ]
            pipeline = newton.CollisionPipeline(model, shape_pairs_filtered=tiled, **options, growable_contacts=True)
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
            pipeline = newton.CollisionPipeline(model, broad_phase=mode, growable_contacts=True)
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
                pipeline = newton.CollisionPipeline(model, broad_phase=mode, growable_contacts=True)
                pairs, _ = _candidate_pairs(pipeline, model)
                record = pipeline.get_collision_selection()["immobile_pairs"]
                if loop:
                    test.assertLessEqual(ground_base, set(pairs))
                    test.assertEqual(record["welded_bodies"], 0)
                else:
                    test.assertFalse(ground_base & set(pairs))
                    test.assertEqual(record["welded_bodies"], _WORLDS)


def test_joints_without_freedom_weld_whatever_their_type(test, device):
    """Weld through D6 joints without axes as through fixed joints, and not through a D6 joint with an axis."""
    fixed, _ = _pair_scene(device)
    locked, worlds = _pair_scene(device, locked_d6=True)
    joint_type = locked.joint_type.numpy()
    dofs = np.diff(locked.joint_qd_start.numpy())
    test.assertNotIn(int(newton.JointType.FIXED), joint_type.tolist())
    # Every D6 joint is a weld without axes, except the arm's root, which has one.
    d6 = joint_type == int(newton.JointType.D6)
    arm_bodies = locked.shape_body.numpy()[[world["arm"] for world in worlds]]
    arm_root = np.isin(locked.joint_child.numpy(), arm_bodies)
    test.assertEqual(int(np.count_nonzero(arm_root)), _WORLDS)
    test.assertTrue(np.all(d6[arm_root] & (dofs[arm_root] == 1)))
    test.assertEqual(int(np.count_nonzero(d6 & ~arm_root)), 6 * _WORLDS)
    test.assertTrue(np.all(dofs[d6 & ~arm_root] == 0))
    ground_base = {tuple(sorted((world["ground"], world["base"]))) for world in worlds}
    for mode, environment in _LAUNCHES:
        with test.subTest(broad_phase=mode, **environment), patch.dict(os.environ, environment), _flag(None):
            expected = newton.CollisionPipeline(fixed, broad_phase=mode, growable_contacts=True)
            pipeline = newton.CollisionPipeline(locked, broad_phase=mode, growable_contacts=True)
            kept, _ = _candidate_pairs(pipeline, locked)
            test.assertEqual(_ordered(kept, device), _ordered(_candidate_pairs(expected, fixed)[0], device))
            record = pipeline.get_collision_selection()["immobile_pairs"]
            test.assertEqual(record, expected.get_collision_selection()["immobile_pairs"])
            test.assertEqual(record["welded_bodies"], 3 * _WORLDS)
            # The welds drop pairs that the opt-out keeps, such as the base's on the ground.
            test.assertTrue(record["enabled"])
            test.assertGreater(record["excluded_pairs"], 0)
            unfiltered, _ = _candidate_pairs(_pipeline(locked, "0", broad_phase=mode), locked)
            test.assertLessEqual(ground_base, set(unfiltered))
            test.assertFalse(ground_base & set(kept))


_ROOT_JOINT_ARM_USD = """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "arm" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def PhysicsJoint "root"
    {
        rel physics:body0 = </arm/base>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
    }

    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 4

        def Cylinder "collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            uniform token axis = "Z"
            double height = 0.03
            double radius = 0.075
            double3 xformOp:translate = (0, 0, 0.01)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }

    def Xform "link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 2
        double3 xformOp:translate = (0, 0, 0.2)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Sphere "collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.05
        }
    }

    def PhysicsRevoluteJoint "hinge"
    {
        rel physics:body0 = </arm/base>
        rel physics:body1 = </arm/link>
        uniform token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, 0.2)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
    }
}
"""


def _arm_on_table_scene(device, *, source):
    """Replicate an arm whose base rests on a static table, beside a free box on the table.

    The arm's root is a D6 joint without axes: ``source="usd"`` imports it from a USD
    ``PhysicsJoint`` that names the base and no other body, and ``source="builder"`` adds it
    directly. A global ground lies well below the table.
    """
    proto = newton.ModelBuilder()
    proto.rigid_gap = 0.01
    _box(proto, -1, 0.3, 0.0, z=-0.5, half=0.5)
    if source == "usd":
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(_ROOT_JOINT_ARM_USD)
        proto.add_usd(stage)
    else:
        base = proto.add_link(label="base")
        proto.add_shape_cylinder(base, xform=_pose(0.0, 0.0, 0.01), radius=0.075, half_height=0.015)
        link = proto.add_link(xform=_pose(0.0, 0.0, 0.2), label="link")
        proto.add_shape_sphere(link, radius=0.05)
        root = proto.add_joint_d6(-1, base)
        hinge = proto.add_joint_revolute(base, link, axis=(0.0, 0.0, 1.0), parent_xform=_pose(0.0, 0.0, 0.2))
        proto.add_articulation([root, hinge])
    _box(proto, proto.add_body(xform=_pose(0.4, 0.3), label="box"), 0.4, 0.3)
    scene = newton.ModelBuilder()
    scene.add_ground_plane(height=-2.0)
    scene.replicate(proto, _WORLDS, spacing=(3.0, 0.0, 0.0))
    return scene.finalize(device=device)


def test_census_drops_exactly_the_contacts_without_response(test, device):
    """Drop exactly the contacts FeatherPGS gives neither side a response DOF, for an arm on a table.

    The arm's root joint is a D6 joint without axes, as the USD importer makes of a
    ``PhysicsJoint`` that names one body. A census of the contacts with the pairs kept classifies
    each by whether FeatherPGS gives either side a response DOF, and a FeatherPGS step on those
    contacts must reserve rows for exactly the responding ones: the free box's table contacts get
    rows, the base's get none. The default must drop exactly the latter and count the base as
    welded.
    """
    for source in ("builder", "usd"):
        with test.subTest(source=source):
            if source == "usd" and not USD_AVAILABLE:
                test.skipTest("Requires usd-core")
            model = _arm_on_table_scene(device, source=source)
            # The arm's root joint has no degree of freedom, whatever made it.
            roots = np.flatnonzero(model.joint_parent.numpy() == -1)
            dofs = np.diff(model.joint_qd_start.numpy())[roots]
            types = model.joint_type.numpy()[roots]
            test.assertEqual(int(np.count_nonzero((types == int(newton.JointType.D6)) & (dofs == 0))), _WORLDS)

            solver = newton.solvers.SolverFeatherPGS(model)
            shape_body = model.shape_body.numpy()
            body = np.maximum(shape_body, 0)
            articulation = solver.forest.body_to_articulation.numpy()[body]
            responds = (
                (shape_body >= 0)
                & (articulation >= 0)
                & (solver.forest.articulation_response_dof_count.numpy()[np.maximum(articulation, 0)] > 0)
                & (solver.forest.body_has_response_dofs.numpy()[body] != 0)
                & ((model.body_flags.numpy()[body] & int(BodyFlags.KINEMATIC)) == 0)
            )
            state = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            census = {}
            for flag in ("0", None):
                pipeline = _pipeline(model, flag)
                contacts = pipeline.contacts()
                pipeline.collide(state, contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                pairs = np.sort(
                    np.stack(
                        (contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count])
                    ),
                    axis=0,
                )
                census[flag] = (pairs, pipeline.get_collision_selection()["immobile_pairs"], contacts)
            pairs, _, listed = census["0"]
            silent = ~responds[pairs[0]] & ~responds[pairs[1]]
            # The base's table contacts and the box's, in every world; the base is the cylinder.
            base = model.shape_type.numpy() == int(newton.GeoType.CYLINDER)
            on_base = base[pairs[0]] | base[pairs[1]]
            test.assertGreaterEqual(int(np.count_nonzero(on_base)), _WORLDS)
            test.assertTrue(np.all(silent[on_base]))
            test.assertGreaterEqual(int(np.count_nonzero(~silent)), _WORLDS)
            # The census is FeatherPGS's own: it reserves no row for exactly the silent contacts.
            solver.step(state, model.state(), model.control(), listed, 1.0 / 120.0)
            path = solver.placement.contact_path.numpy()[: len(silent)]
            np.testing.assert_array_equal(path[silent], -1)
            test.assertTrue(np.all(path[~silent] >= 0), path)
            kept, record, _ = census[None]
            test.assertEqual(sorted(map(tuple, kept.T.tolist())), sorted(map(tuple, pairs[:, ~silent].T.tolist())))
            test.assertTrue(record["enabled"])
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
            pipeline = newton.CollisionPipeline(model, broad_phase=mode, growable_contacts=True)
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
        pipeline = newton.CollisionPipeline(model, growable_contacts=True)
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
        pipeline = newton.CollisionPipeline(model, growable_contacts=True)
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
        newton.CollisionPipeline(model, growable_contacts=True)
    for argument in (True, False):
        with test.subTest(argument=argument), _flag("2"):
            selection = newton.CollisionPipeline(
                model, include_immobile_pairs=argument, growable_contacts=True
            ).get_collision_selection()
            test.assertEqual(selection["environment"]["overridden"].get(_FLAG), "2")
            test.assertNotIn(_FLAG, selection["environment"]["applied"])
            test.assertEqual(selection["immobile_pairs"]["enabled"], not argument)
    with _flag("1"):
        selection = newton.CollisionPipeline(model, growable_contacts=True).get_collision_selection()
        test.assertEqual(selection["environment"]["applied"].get(_FLAG), "1")
        test.assertNotIn(_FLAG, selection["environment"]["overridden"])


def _arm_scene(device, *, locked_d6=False):
    """A fixed-base arm whose base rests in the ground and whose link presses a free cube, in two worlds.

    ``locked_d6`` roots the base with a D6 joint without axes instead of a fixed joint.
    """
    proto = newton.ModelBuilder()
    proto.rigid_gap = 0.01
    base_pose = wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity())
    base = proto.add_link(xform=base_pose, label="base")
    proto.add_shape_box(base, hx=0.1, hy=0.1, hz=0.05)
    root = (proto.add_joint_d6 if locked_d6 else proto.add_joint_fixed)(-1, base, parent_xform=base_pose)
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


# What the narrow phase computes per contact; thread ids are left out, they follow the candidate list.
_CONTACT_GEOMETRY = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1")


def _per_contact_arrays(contacts):
    """Return every per-contact array of ``contacts``."""
    return [
        array
        for name, array in vars(contacts).items()
        if name.startswith("rigid_contact_")
        and isinstance(array, wp.array)
        and array.shape == (contacts.rigid_contact_max,)
    ]


def _list_immobile_contacts_last(contacts, immobile):
    """Stably move the contacts between two ``immobile`` shapes behind the others; return how many others."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shapes = (contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count])
    dropped = immobile[shapes[0]] & immobile[shapes[1]]
    order = np.concatenate((np.flatnonzero(~dropped), np.flatnonzero(dropped)))
    for array in _per_contact_arrays(contacts):
        values = array.numpy().copy()
        values[:count] = values[order]
        array.assign(values)
    return count - int(np.count_nonzero(dropped))


def _arm_shape_roles(model):
    """Return each shape's role in the arm scene: -1 static, 0 the welded base, 1 the arm and 2 the cube."""
    shape_body = model.shape_body.numpy()
    # Body 0 of each world is the base, body 1 the arm and body 2 the cube.
    return np.where(shape_body < 0, -1, shape_body % 3)


def _contact_roles(contacts, roles):
    """Return the roles of each live contact's two shapes, the smaller first, shape [2, count]."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shapes = np.stack((contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]))
    return np.sort(roles[shapes], axis=0)


def _check_arm_steps(test, model, law, friction_anchor_beta, device):
    """Step the arm scene with and without the base's ground pairs from the same state, and compare every step."""
    roles = _arm_shape_roles(model)
    immobile = roles <= 0
    control = model.control()
    solvers = [newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=friction_anchor_beta) for _ in range(2)]
    # Sorted contacts compare element by element: the reference keeps the base's ground pairs.
    pipelines = (_pipeline(model, "0", deterministic=True), _pipeline(model, None, deterministic=True))
    reference, contacts = (pipeline.contacts() for pipeline in pipelines)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    # Every step gets new states, kept alive so that no address recurs: a solver must not take a
    # copied input for its own last output.
    states = [state]
    dropped_contacts = arm_contacts = 0
    for _ in range(60):
        pipelines[0].collide(state, reference)
        pipelines[1].collide(state, contacts)
        listed = int(reference.rigid_contact_count.numpy()[0])
        kept = _list_immobile_contacts_last(reference, immobile)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertEqual(count, kept)
        for name in _CONTACT_GEOMETRY:
            np.testing.assert_array_equal(
                getattr(contacts, f"rigid_contact_{name}").numpy()[:count],
                getattr(reference, f"rigid_contact_{name}").numpy()[:kept],
            )
        dropped_contacts += listed - kept
        ends = _contact_roles(contacts, roles)
        arm_contacts += int(np.count_nonzero((ends[0] == 1) & (ends[1] == 2)))
        outputs = []
        for solver, buffer in zip(solvers, (reference, contacts), strict=True):
            state_in, state_out = model.state(), model.state()
            state_in.assign(state)
            state_in.clear_forces()
            solver.step(state_in, state_out, control, buffer, 1.0 / 120.0)
            states += [state_in, state_out]
            outputs.append(state_out)
        # FeatherPGS gives the base's ground contacts no row and every other contact the same rows:
        # the same route, row count, world and first row.
        rows_of = {
            name: [getattr(solver.placement, f"contact_{name}").numpy().copy() for solver in solvers]
            for name in ("path", "slots_needed", "world", "slot")
        }
        paths = rows_of.pop("path")
        np.testing.assert_array_equal(paths[0][kept:listed], -1)
        np.testing.assert_array_equal(paths[1][:count], paths[0][:kept])
        rows = paths[0][:kept] >= 0
        for name, values in rows_of.items():
            np.testing.assert_array_equal(values[1][:count][rows], values[0][:kept][rows], err_msg=f"contact_{name}")
        expected, actual = (np.concatenate((output.joint_q.numpy(), output.joint_qd.numpy())) for output in outputs)
        test.assertTrue(np.all(np.isfinite(actual)))
        if wp.get_device(device).is_cpu:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-5)
        state = outputs[0]
    # The base rests in the ground and the arm presses the cube.
    test.assertGreater(dropped_contacts, 0)
    test.assertGreater(arm_contacts, 0)


def test_fixed_base_arm_steps_match(test, device):
    """Step FeatherPGS alike without the base's ground contacts, from the same state every step.

    Under point and patch friction, with the base rooted by a fixed joint and by a D6 joint
    without axes. FeatherPGS reserves a world's contact rows with atomics, which CUDA can serve in
    warp order, so the dropped contacts must not move the others. Left in place, they push world
    1's last arm-cube contacts into a second warp; on the GPU, states then ended up to 1.2e-2
    rad/s apart after 12 PGS iterations, and a CPU run that reserves those contacts' rows first
    reproduces that. The reference therefore lists the dropped contacts last, so that every other
    contact keeps its index. Contact geometry and rows then agree exactly, and states bitwise on
    CPU and within 1e-5 on CUDA, which allows float atomics to accumulate in another order.
    """
    for root in ("fixed", "locked_d6"):
        model = _arm_scene(device, locked_d6=root == "locked_d6")
        for law, beta in (("patch", 0.2),):
            with test.subTest(root=root, friction=law):
                _check_arm_steps(test, model, law, beta, device)


def _arm_trajectory(test, model, flag, law, friction_anchor_beta, steps=60):
    """Free-run the arm scene as applications step it, with the immobile-pair flag at ``flag``.

    Two states take turns as input and output, which FeatherPGS's caches follow, and the contacts
    keep the order in which the pipeline lists them.

    Returns:
        The joint coordinates and velocities after each step, and the number of contacts in all,
        between two immobile shapes and between the arm and the cube, summed over the steps.
    """
    roles = _arm_shape_roles(model)
    solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=friction_anchor_beta)
    pipeline = _pipeline(model, flag)
    contacts = pipeline.contacts()
    control = model.control()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    trajectory, counts = [], {"all": 0, "immobile": 0, "arm_cube": 0}
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        ends = _contact_roles(contacts, roles)
        counts["all"] += ends.shape[1]
        counts["immobile"] += int(np.count_nonzero(ends[1] <= 0))
        counts["arm_cube"] += int(np.count_nonzero((ends[0] == 1) & (ends[1] == 2)))
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
        trajectory.append(np.concatenate((state_0.joint_q.numpy(), state_0.joint_qd.numpy())))
    return np.asarray(trajectory), counts


def test_fixed_base_arm_trajectories_match(test, device):
    """Free-run FeatherPGS bitwise alike without the base's ground contacts, on CPU.

    Both runs step as applications do, with two states in turn and the contacts of the default
    and the opt-out pipeline in their own order, the dropped contacts among the others. CPU
    kernels run in order, so the remaining contacts reserve their rows in the same order. CUDA
    reserves rows with atomics and can order them otherwise; test_fixed_base_arm_steps_match
    covers it. The base is rooted by a fixed joint and by a D6 joint without axes.
    """
    for root in ("fixed", "locked_d6"):
        model = _arm_scene(device, locked_d6=root == "locked_d6")
        for law, beta in (("patch", 0.2),):
            with test.subTest(root=root, friction=law):
                expected, listed = _arm_trajectory(test, model, "0", law, beta)
                actual, counts = _arm_trajectory(test, model, None, law, beta)
                # The base rests in the ground and the arm presses the cube; only the base's
                # ground contacts are dropped.
                test.assertGreater(listed["immobile"], 0)
                test.assertGreater(listed["arm_cube"], 0)
                test.assertEqual(counts, {**listed, "all": listed["all"] - listed["immobile"], "immobile": 0})
                test.assertTrue(np.all(np.isfinite(actual)))
                np.testing.assert_array_equal(actual, expected)


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
    ("test_joints_without_freedom_weld_whatever_their_type", test_joints_without_freedom_weld_whatever_their_type),
    (
        "test_census_drops_exactly_the_contacts_without_response",
        test_census_drops_exactly_the_contacts_without_response,
    ),
    ("test_notifications_refresh_the_immobile_set", test_notifications_refresh_the_immobile_set),
    ("test_unchanged_notifications_skip_the_refresh", test_unchanged_notifications_skip_the_refresh),
    # Warp captures graphs on CPU devices as well.
    ("test_captured_collide_reads_notified_set", test_captured_collide_reads_notified_set),
    ("test_environment_flag_values", test_environment_flag_values),
    ("test_fixed_base_arm_steps_match", test_fixed_base_arm_steps_match),
):
    add_function_test(TestCollisionImmobilePairs, name, function, devices=devices)
# CUDA can reserve the rows of renumbered contacts in another order.
add_function_test(
    TestCollisionImmobilePairs,
    "test_fixed_base_arm_trajectories_match",
    test_fixed_base_arm_trajectories_match,
    devices=[device for device in devices if device.is_cpu],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

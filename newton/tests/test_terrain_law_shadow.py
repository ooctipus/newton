# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's shadow route: the flag, the side buffer, the pair policy, capture and today's contacts.

``NEWTON_TERRAIN_LAW_SHADOW`` lists partner classes whose terrain-law kernels run after the narrow phase into a
side buffer that nothing reads, while the terrain route writes every heightfield contact (comparison mode). With it set,
today's contacts must stay bit for bit what the terrain route writes without it (``NEWTON_TERRAIN_LAW=none``); the
shadow must follow the pipeline's pair policy, run inside CUDA graph capture, give stale features no row and raise
``terrain_stale``, keep every support of a pair however many it has, and raise ``terrain_rows`` only when the side
buffer's spill region, sized at the scene level (``NEWTON_TERRAIN_LAW_SHADOW_ROWS``), has no room for them.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import terrain_law_point
from newton._src.geometry.terrain_contacts import (
    TERRAIN_LANE_THREAD,
    TERRAIN_LANE_TILE,
    TERRAIN_PARTNER_BOX,
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_PARTNER_HULL,
    TERRAIN_PARTNER_POINT,
    TERRAIN_PARTNER_SCAN_RUN,
    TERRAIN_PARTNER_WALK_SLOTS,
)
from newton._src.geometry.terrain_law import (
    TERRAIN_LAW_TILE_BLOCKS,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_ROW_OVERFLOW,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    terrain_finish_blocks,
    terrain_law_block_dim,
)
from newton._src.geometry.terrain_law_point import TERRAIN_POINT_WALK_SLOTS
from newton._src.geometry.terrain_law_solid import (
    TERRAIN_SOLID_JOB_LISTS,
    TERRAIN_SOLID_JOB_THREADS_CPU,
    TERRAIN_SOLID_LAUNCHES,
    TERRAIN_SOLID_RESERVE_PAIRS,
    terrain_solid_job_capacities,
    terrain_solid_job_reserves,
)
from newton._src.geometry.terrain_shadow import (
    TERRAIN_LAW_JOB_COUNTERS,
    TERRAIN_LAW_JOB_NAMES,
    TERRAIN_LAW_KERNELS,
    TERRAIN_LAW_SPILL_SHARE,
    TerrainLawShadow,
    parse_terrain_law_shadow,
    parse_terrain_law_shadow_jobs,
    parse_terrain_law_shadow_rows,
    terrain_law_spill_check,
)
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_solid_utils as su
from newton.tests import terrain_law_test_utils as tu
from newton.tests.unittest_utils import (
    StdOutCapture,
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    scoped_capture_after_gc,
)

PIPELINE = {"rigid_contact_max": 1 << 14, "max_triangle_pairs": 1 << 16}


def _mixed_scene(device):
    """Rough terrain under spheres, capsules, boxes and cylinders, each body in its own world, some resting."""
    rng = np.random.default_rng(6)
    n = 33
    heights = np.round(rng.uniform(0.0, 0.1, (n, n)) / 0.005) * 0.005
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
    data = heights.astype(np.float32)
    terrain = builder.add_shape_heightfield(
        heightfield=newton.Heightfield(data=data, nrow=n, ncol=n, hx=1.6, hy=1.6), cfg=cfg
    )
    for i in range(24):
        builder.begin_world()
        x, y = rng.uniform(-1.2, 1.2, 2)
        z = hf.height(x, y) + rng.uniform(0.02, 0.09)
        q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(size=3))), float(rng.uniform(0.0, 3.0)))
        body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z), q))
        kind = i % 4
        if kind == 0:
            builder.add_shape_sphere(body, radius=0.03, cfg=cfg)
        elif kind == 1:
            builder.add_shape_capsule(body, radius=0.025, half_height=0.06, cfg=cfg)
        elif kind == 2:
            builder.add_shape_box(body, hx=0.05, hy=0.04, hz=0.03, cfg=cfg)
        else:
            builder.add_shape_cylinder(body, radius=0.03, half_height=0.04, cfg=cfg)
        builder.end_world()
    return builder.finalize(device=device), terrain


def _contacts(pipeline, model, state=None):
    state = model.state() if state is None else state
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return contacts


def _contact_arrays(contacts) -> dict:
    """Every ``rigid_contact_*`` array of a :class:`newton.Contacts`, the per-contact ones cut to the contact count."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    out = {"count": count}
    for name in sorted(vars(contacts)):
        array = getattr(contacts, name)
        if name.startswith("rigid_contact_") and isinstance(array, wp.array):
            values = array.numpy()
            out[name] = values[:count] if values.shape[0] == contacts.rigid_contact_max else values
    return out


def _unset():
    """An environment without NEWTON_TERRAIN_LAW_SHADOW or NEWTON_TERRAIN_LAW: the default, where the law writes."""
    environ = {k: v for k, v in os.environ.items() if k not in ("NEWTON_TERRAIN_LAW_SHADOW", "NEWTON_TERRAIN_LAW")}
    return patch.dict(os.environ, environ, clear=True)


def test_shadow_keeps_contacts_bitwise(test, device):
    """Write today's contacts bit for bit with the shadow running beside the terrain route.

    Every ``rigid_contact_*`` array is compared in place. On CUDA both pipelines are deterministic: by default the
    global reducer breaks score ties by atomic order, so two pipelines without the shadow could already differ.
    """
    model, _terrain = _mixed_scene(device)
    state = model.state()
    for deterministic in (False, True) if device.is_cpu else (True,):
        with test.subTest(deterministic=deterministic):
            today = tu.terrain_route_pipeline(model, deterministic=deterministic, **PIPELINE)
            shadow = tu.shadow_pipeline(model, deterministic=deterministic, **PIPELINE)
            test.assertIsNone(today._terrain_law_shadow)
            test.assertIsNotNone(shadow._terrain_law_shadow)
            a = _contact_arrays(_contacts(today, model, state))
            b = _contact_arrays(_contacts(shadow, model, state))
            test.assertGreater(a["count"], 20)
            test.assertEqual(sorted(a), sorted(b))
            test.assertGreater(len(a), 12)
            for name, values in a.items():
                np.testing.assert_array_equal(values, b[name], err_msg=name)
            test.assertGreater(int(shadow._terrain_law_shadow.rows()["count"].sum()), 0)


def test_shadow_runs_under_graph_capture(test, device):
    """Capture the shadow with the collide in a CUDA graph and replay it, twice, to the rows of an uncaptured collide.

    The shadow's launches are also captured alone with the device's memory pool off, so that an allocation inside
    them raises instead of becoming a graph allocation node.
    """
    model, _terrain = _mixed_scene(device)
    pipeline = tu.shadow_pipeline(model, **PIPELINE)
    shadow = pipeline._terrain_law_shadow
    state = model.state()
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    expected = shadow.rows()
    spilled = shadow.rows_spilled()
    # Move the bodies, so that the replay must recompute the rows.
    moved = model.state()
    q = moved.body_q.numpy()
    q[:, 2] += 0.5
    moved.body_q.assign(q)
    pipeline.collide(moved, contacts)
    test.assertEqual(int(shadow.rows()["count"].sum()), 0)
    with scoped_capture_after_gc(device=device) as capture:
        pipeline.collide(state, contacts)
    for _replay in range(2):
        wp.capture_launch(capture.graph)
        replayed = shadow.rows()
        for name in ("count", "state", "q", "n", "d", "feature"):
            np.testing.assert_array_equal(replayed[name], expected[name], err_msg=name)
        test.assertEqual(shadow.rows_spilled(), spilled)
    # The shadow alone without a memory pool; its replay recomputes rows forgotten after an uncaptured collide.
    pooled = wp.is_mempool_enabled(device)
    wp.set_mempool_enabled(device, False)
    try:
        with scoped_capture_after_gc(device=device) as capture:
            shadow.launch(
                pipeline.geom_transform,
                pipeline.geom_data,
                shape_type=model.shape_type,
                shape_gap=model.shape_gap,
                shape_heightfield_index=model.shape_heightfield_index,
                **pipeline._broad_phase_pair_filter(model),
            )
    finally:
        wp.set_mempool_enabled(device, pooled)
    pipeline.collide(state, contacts)
    shadow._out.count.zero_()
    wp.capture_launch(capture.graph)
    replayed = shadow.rows()
    for name in ("count", "state", "q", "n", "d", "feature"):
        np.testing.assert_array_equal(replayed[name], expected[name], err_msg=name)


def test_stale_features_give_no_rows(test, device):
    """Give no row to pairs over heights edited and not rebuilt, and raise ``terrain_stale``; rebuild restores them."""
    model, terrain = _mixed_scene(device)
    pipeline = tu.shadow_pipeline(model, **PIPELINE)
    shadow = pipeline._terrain_law_shadow
    narrow = pipeline.narrow_phase
    _contacts(pipeline, model)
    before = shadow.rows()
    test.assertGreater(int(before["count"].sum()), 0)
    narrow.check_buffer_capacity()
    shadow.features.mark_elevations_changed(terrain)
    _contacts(pipeline, model)
    stale = shadow.rows()
    test.assertEqual(int(stale["count"].sum()), 0)
    test.assertTrue(np.all(stale["state"] == int(TERRAIN_PAIR_STALE)))
    test.assertEqual([k for k, v in narrow.buffer_capacity_status().items() if v], ["terrain_stale"])
    with test.assertRaisesRegex(RuntimeError, "terrain_stale"):
        narrow.check_buffer_capacity()
    shadow.features.rebuild_terrain_features(terrain)
    narrow.buffer_capacity_status(clear=True)
    _contacts(pipeline, model)
    after = shadow.rows()
    for name in ("count", "state", "q", "n", "d", "feature"):
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)
    narrow.check_buffer_capacity()


def _corrugated(rng):
    """Ridges every second column at random heights, valleys between: 0.1 m cells."""
    n = 41
    heights = np.zeros((n, n))
    ridge = rng.uniform(0.02, 0.035, size=n)
    heights[:, 1::2] = ridge[1::2]
    return heights


def _assert_law_rows(test, expected, got, what):
    """Assert that a pair's float32 rows and the float64 law's match one for one (terrain point within 10 um, normal
    within 1e-3 rad, core distance within 10 um)."""
    test.assertEqual(len(got), len(expected), msg=what)
    test.assertEqual(tu.missing_rows(expected, got, q_tol=1.0e-5, angle_tol=1.0e-3, d_tol=1.0e-5), [], msg=what)
    test.assertEqual(tu.missing_rows(got, expected, q_tol=1.0e-5, angle_tol=1.0e-3, d_tol=1.0e-5), [], msg=what)


def test_long_capsule_keeps_every_support(test, device):
    """Keep every support of a pair with more rows than its walk slots, as the float64 law gives them; raise nothing.

    A long capsule lies across ridges of different heights: each ridge in reach is a support (its own plane), more than
    the point class's eight walk slots. Its class's finish of deferred pairs finishes it and writes its rows to the side
    buffer's spill region; the walk slots bound no row. (With a per-class limit of eight rows the law kept eight of the
    ten, each plane's deepest first, and raised ``terrain_rows``.)
    """
    rng = np.random.default_rng(2)
    heights = _corrugated(rng)
    hf = tu.reference_surface(tu.heightfield_model(heights, 2.0, 2.0, device="cpu"))
    lying = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    dims = {"radius": 0.03, "half_height": 0.9}
    top = max(hf.height(x, 0.05) for x in np.linspace(-0.95, 0.95, 400))
    pose = np.array([0.0, 0.05, top + 0.03 + 0.002, *lying], np.float32).astype(np.float64)
    model, _terrain, shapes = tu.body_scene(heights, 2.0, 2.0, [("capsule", dims, pose)], device=device)
    pipeline = tu.shadow_pipeline(model, **PIPELINE)
    _contacts(pipeline, model)
    shadow = pipeline._terrain_law_shadow
    rows = shadow.rows()
    core = ref.make_core("capsule", dims, pose)
    unlimited = ref.pair_rows(hf, core, dims["radius"] + 0.04)
    test.assertEqual(len(unlimited), 10)
    test.assertGreater(len(unlimited), TERRAIN_POINT_WALK_SLOTS)
    test.assertEqual(int(rows["count"][0]), len(unlimited))
    test.assertEqual(int(rows["state"][0]), 0)
    test.assertEqual(shadow.rows_spilled(), len(unlimited))
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
    test.assertEqual(shadow.overflow_counts(), (0, 0, 0))
    _assert_law_rows(test, unlimited, tu.shadow_rows(pipeline)[shapes[0]], "capsule")


CAPTURED_PATCH = (
    np.array(
        [
            [30, 20, 40, 20, 80, 100, 60, 80, 30, 40, 60, 60],
            [40, 40, 20, 60, 80, 40, 30, 80, 100, 80, 100, 60],
            [80, 100, 80, 100, 80, 30, 40, 40, 80, 100, 80, 40],
            [40, 40, 60, 100, 60, 100, 30, 40, 30, 100, 46.7, 40],
            [60, 100, 80, 40, 60, 100, 60, 40, 80, 40, 100, 60],
            [40, 40, 20, 30, 25, 20, 20, 60, 80, 100, 30, 80],
            [30, 80, 60, 100, 60, 30, 20, 30, 100, 40, 30, 40],
            [100, 60, 80, 80, 80, 40, 20, 40, 20, 40, 20, 40],
            [100, 80, 40, 100, 30, 40, 100, 80, 40, 20, 60, 80],
            [40, 50, 50, 80, 40, 25, 80, 100, 60, 100, 100, 100],
            [33.3, 40, 80, 100, 100, 100, 100, 40, 26.7, 100, 30, 60],
            [100, 30, 40, 100, 30, 40, 80, 50, 20, 80, 60, 40],
        ]
    )
    * 1.0e-3
)
"""Heights [m] of the rough terrain (0.1 m cells) under a 0.25 m sphere whose shadow rows overflowed a per-class row
limit on the GPU in a recorded rough-terrain run (an overflow probe's capture, at x -35.63, y 19.58 of the recorded
terrain)."""
CAPTURED_CENTRE = (-0.55 + 0.573985, -0.55 + 0.58171, 0.300879)
"""The sphere's centre over :data:`CAPTURED_PATCH` (half extent 0.55 m), 2.5-3.3 cm above its nine supports."""


def _captured_sphere(device):
    pose = np.array([*CAPTURED_CENTRE, 0.0, 0.0, 0.0, 1.0])
    dims = {"radius": 0.25}
    model, _terrain, shapes = tu.body_scene(CAPTURED_PATCH, 0.55, 0.55, [("sphere", dims, pose)], device=device)
    hf = tu.reference_surface(model)
    return model, shapes, ref.pair_rows(hf, ref.make_core("sphere", dims, pose), dims["radius"] + 0.04)


def test_sphere_keeps_every_support_over_rough_terrain(test, device):
    """Keep all nine supports of a 0.25 m sphere hovering over rough terrain, as the float64 law gives them.

    A pose a GPU overflow probe captured in a recorded rough-terrain run: every support a different feature and plane,
    all within reach, one more than the point class's eight walk slots. (With a per-class limit of eight rows the law
    kept eight, flagged a row overflow and raised ``terrain_rows``, which failed every such run with the shadow on.)
    """
    model, shapes, law = _captured_sphere(device)
    test.assertEqual(len(law), 9)
    pipeline = tu.shadow_pipeline(model, **PIPELINE)
    _contacts(pipeline, model)
    shadow = pipeline._terrain_law_shadow
    rows = shadow.rows()
    test.assertEqual(int(rows["count"][0]), 9)
    test.assertEqual(int(rows["state"][0]), 0)
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
    pipeline.narrow_phase.check_buffer_capacity()
    _assert_law_rows(test, law, tu.shadow_rows(pipeline)[shapes[0]], "sphere")


def _policy_scene(device):
    """Spheres resting on a flat heightfield: free, filtered against it, in a group that does not touch it, welded to
    the world by a fixed joint (structurally immobile), and one an explicit pair list leaves out."""
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.01, gap=0.01, collision_group=1)
    data = np.zeros((21, 21), np.float32)
    terrain = builder.add_shape_heightfield(
        heightfield=newton.Heightfield(data=data, nrow=21, ncol=21, hx=1.0, hy=1.0), cfg=cfg
    )
    shapes = {}
    for i, name in enumerate(("free", "filtered", "other_group", "welded", "listed_out")):
        builder.begin_world()
        position = wp.vec3(-0.6 + 0.3 * i, 0.0, 0.035)
        group = 2 if name == "other_group" else 1
        sphere_cfg = builder.ShapeConfig(margin=0.01, gap=0.01, collision_group=group)
        if name == "welded":
            body = builder.add_link(xform=wp.transform(position))
            builder.add_articulation(
                [builder.add_joint_fixed(-1, body, parent_xform=wp.transform(position), collision_filter_parent=False)]
            )
        else:
            body = builder.add_body(xform=wp.transform(position))
        shapes[name] = builder.add_shape_sphere(body, radius=0.03, cfg=sphere_cfg)
        if name == "filtered":
            builder.add_shape_collision_filter_pair(terrain, shapes[name])
        builder.end_world()
    return builder.finalize(device=device), terrain, shapes


def test_shadow_follows_the_pair_policy(test, device):
    """Run the shadow on exactly the pairs the pipeline collides: filter pairs, groups, explicit list, immobile pairs."""
    model, terrain, shapes = _policy_scene(device)
    pipeline = tu.shadow_pipeline(model, **PIPELINE)
    _contacts(pipeline, model)
    rows = pipeline._terrain_law_shadow.rows()
    partners = {int(p[1]): i for i, p in enumerate(rows["pairs"])}
    test.assertNotIn(shapes["filtered"], partners)
    test.assertNotIn(shapes["other_group"], partners)
    test.assertEqual(int(rows["count"][partners[shapes["free"]]]), 1)
    # The broad phase drops a welded body's pair with the static heightfield by default; so does the shadow.
    test.assertEqual(int(rows["count"][partners[shapes["welded"]]]), 0)
    test.assertEqual(int(rows["state"][partners[shapes["welded"]]]), int(TERRAIN_PAIR_FILTERED))
    kept = tu.shadow_pipeline(model, include_immobile_pairs=True, **PIPELINE)
    _contacts(kept, model)
    rows = kept._terrain_law_shadow.rows()
    partners = {int(p[1]): i for i, p in enumerate(rows["pairs"])}
    test.assertEqual(int(rows["count"][partners[shapes["welded"]]]), 1)
    # An explicit pair list without a pair: no shadow pair either.
    listed = model.shape_contact_pairs.numpy()
    listed = listed[~np.any(listed == shapes["listed_out"], axis=1)]
    explicit = tu.shadow_pipeline(
        model, shape_pairs_filtered=wp.array(listed, dtype=wp.vec2i, device=device), **PIPELINE
    )
    _contacts(explicit, model)
    pairs = explicit._terrain_law_shadow.rows()["pairs"]
    test.assertNotIn(shapes["listed_out"], set(pairs[:, 1].tolist()))
    test.assertIn(shapes["free"], set(pairs[:, 1].tolist()))
    test.assertTrue(np.all(pairs[:, 0] == terrain))


def test_high_cores_leave_at_the_tile_max(test, device):
    """Give bodies far above the terrain no row through the tile-max exit, before any window is walked."""
    model, _terrain = _mixed_scene(device)
    pipeline = tu.shadow_pipeline(model, **PIPELINE)
    state = model.state()
    q = state.body_q.numpy()
    q[:, 2] += 1.0
    state.body_q.assign(q)
    _contacts(pipeline, model, state)
    rows = pipeline._terrain_law_shadow.rows()
    test.assertEqual(int(rows["count"].sum()), 0)
    test.assertTrue(np.all(rows["state"] == int(TERRAIN_PAIR_TILE_EXIT)))


def _tiled_scene(device):
    """Large spheres and long capsules (windows of more than 16 cells) on rough terrain, every second one 1 m above it."""
    rng = np.random.default_rng(9)
    heights = np.round(rng.uniform(0.0, 0.1, (33, 33)) / 0.005) * 0.005
    lying = np.array([0.0, np.sin(np.pi / 4.0), 0.0, np.cos(np.pi / 4.0)])
    bodies = []
    for i in range(12):
        x, y = rng.uniform(-1.0, 1.0, 2)
        lift = 1.0 if i % 2 else 0.0
        if i % 4 < 2:
            bodies.append(("sphere", {"radius": 0.25}, np.array([x, y, 0.35 + lift, 0.0, 0.0, 0.0, 1.0])))
        else:
            bodies.append(("capsule", {"radius": 0.06, "half_height": 0.3}, np.array([x, y, 0.16 + lift, *lying])))
    return tu.body_scene(heights, 1.6, 1.6, bodies, device=device)[0]


def test_tiled_pairs_that_exit_are_not_queued(test, device):
    """Queue for the tiled kernel exactly the larger-window pairs that pass the whole-pair rejects.

    The one-thread kernel takes every pair of the class through the rejects. A larger-window pair that leaves (here at
    the tile max, high above the terrain) is recorded there and never occupies a tile of lanes; the tiled kernel's
    fixed grid of blocks takes the queued pairs in turn.
    """
    model = _tiled_scene(device)
    pipeline = tu.shadow_pipeline(model, **PIPELINE)
    shadow = pipeline._terrain_law_shadow
    tiled_launches = []
    launch_tiled = wp.launch_tiled

    def record(kernel, *args, **kwargs):
        if kernel is terrain_law_point.terrain_point_rows_tiled:
            tiled_launches.append((kwargs["dim"], kwargs["block_dim"]))
        return launch_tiled(kernel, *args, **kwargs)

    with patch.object(wp, "launch_tiled", record):
        _contacts(pipeline, model)
    rows = shadow.rows()
    tiled = rows["lane_class"] == TERRAIN_LANE_TILE
    left = (rows["state"] & int(TERRAIN_PAIR_TILE_EXIT)) != 0
    test.assertEqual(int(tiled.sum()), 12)
    test.assertEqual(int((tiled & left).sum()), 6)
    test.assertGreater(int(rows["count"][tiled & ~left].sum()), 0)
    work = next(extra[0] for kind, *_launch, extra in shadow._launches if kind == "thread" and not extra[1])
    queued = work.pairs.numpy()[: int(work.count.numpy()[0])]
    np.testing.assert_array_equal(np.sort(queued), shadow._listed[tiled & ~left])
    test.assertEqual(tiled_launches, [(min(12, TERRAIN_LAW_TILE_BLOCKS), terrain_law_block_dim(device))])


def test_shadow_launches_take_one_block_size(test, device):
    """Launch every kernel of the shadow with one block size, so that the point module builds once per device.

    Warp builds a module once per block size it is launched with: the one-thread kernel took Warp's default 256-thread
    blocks while the tiled kernels took their lanes, two builds of the module on every device. Every launch now takes
    ``terrain_law_block_dim`` (one warp on CUDA, where the one-thread kernel's warps then also spread evenly over the
    SMs; one on the CPU).
    """
    model = _tiled_scene(device)
    pipeline = tu.shadow_pipeline(model, **PIPELINE)
    blocks = {}
    launch, launch_tiled = wp.launch, wp.launch_tiled

    def recorder(original):
        def record(kernel, *args, **kwargs):
            if kernel.module.name == terrain_law_point.__name__:
                blocks.setdefault(kernel.key, set()).add(kwargs.get("block_dim"))
            return original(kernel, *args, **kwargs)

        return record

    with patch.object(wp, "launch", recorder(launch)), patch.object(wp, "launch_tiled", recorder(launch_tiled)):
        _contacts(pipeline, model)
    test.assertEqual(
        sorted(blocks), ["terrain_point_rows", "terrain_point_rows_finish", "terrain_point_rows_tiled"], msg=blocks
    )
    test.assertEqual(set().union(*blocks.values()), {terrain_law_block_dim(device)}, msg=blocks)


class TestTerrainLawShadow(unittest.TestCase):
    def test_flag_values_cpu(self):
        """Parse the flag's partner classes: names, ``all``, blanks; refuse unknown names and classes without kernels."""
        self.assertEqual(parse_terrain_law_shadow(None), ())
        self.assertEqual(parse_terrain_law_shadow(""), ())
        self.assertEqual(parse_terrain_law_shadow(" , "), ())
        self.assertEqual(parse_terrain_law_shadow("point"), (TERRAIN_PARTNER_POINT,))
        self.assertEqual(parse_terrain_law_shadow("point, point"), (TERRAIN_PARTNER_POINT,))
        self.assertEqual(parse_terrain_law_shadow("all"), tuple(sorted(TERRAIN_LAW_KERNELS)))
        self.assertEqual(
            parse_terrain_law_shadow("hull, box,cylinder"),
            (TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL),
        )
        self.assertEqual(
            parse_terrain_law_shadow("all"),
            (TERRAIN_PARTNER_POINT, TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL),
        )
        with self.assertRaisesRegex(ValueError, "unknown partner class 'sphere'"):
            parse_terrain_law_shadow("sphere")
        model, _terrain = _mixed_scene("cpu")
        with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW": "spheres"}), self.assertRaises(ValueError):
            newton.CollisionPipeline(model, **PIPELINE)
        # The spill region's rows: a non-negative count, blank for the default.
        self.assertIsNone(parse_terrain_law_shadow_rows(None))
        self.assertIsNone(parse_terrain_law_shadow_rows(" "))
        self.assertEqual(parse_terrain_law_shadow_rows("0"), 0)
        self.assertEqual(parse_terrain_law_shadow_rows(" 4096 "), 4096)
        for value in ("-1", "1e6", "many"):
            with self.assertRaisesRegex(ValueError, "NEWTON_TERRAIN_LAW_SHADOW_ROWS"):
                parse_terrain_law_shadow_rows(value)
        with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW_ROWS": "-3"}), self.assertRaises(ValueError):
            tu.shadow_pipeline(model, **PIPELINE)
        # The solid classes' job capacities: a positive factor, blank for the default.
        self.assertIsNone(parse_terrain_law_shadow_jobs(None))
        self.assertIsNone(parse_terrain_law_shadow_jobs(" "))
        self.assertEqual(parse_terrain_law_shadow_jobs("2"), 2.0)
        self.assertEqual(parse_terrain_law_shadow_jobs(" 0.5 "), 0.5)
        for value in ("0", "-1", "inf", "nan", "many"):
            with self.assertRaisesRegex(ValueError, "NEWTON_TERRAIN_LAW_SHADOW_JOBS"):
                parse_terrain_law_shadow_jobs(value)
        with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW_JOBS": "0"}), self.assertRaises(ValueError):
            tu.shadow_pipeline(model, **PIPELINE)
        # The shadow's own option takes the same factors: none of these would size a list.
        for value in (0, -1.0, float("inf"), float("nan")):
            with self.subTest(job_scale=value), self.assertRaisesRegex(ValueError, "positive finite factor"):
                TerrainLawShadow(model, (TERRAIN_PARTNER_BOX,), job_scale=value)

    def test_job_check_names_lists_and_classes_cpu(self):
        """Warn of a solid class's job list without room by the class's and the list's names, with the entries asked
        for against the room and the factor to raise, and of the scan's runs by the window bound they hold (a pipeline
        to rebuild); warn again of a reserve without room (its pairs keep no row); stay silent within the room."""
        jobs = 4
        # (entries asked, room, reserve): the cylinder class's candidates outgrow their reserve too
        overflowing = {
            (TERRAIN_PARTNER_BOX, "faces"): (9, 5, 10),
            (TERRAIN_PARTNER_CYLINDER, "candidates"): (70, 64, 2),
            (TERRAIN_PARTNER_HULL, "scans"): (12, 11, 4),
        }
        capacities = np.zeros(4 * TERRAIN_LAW_JOB_COUNTERS, np.int32)
        reserves = np.zeros_like(capacities)
        counters = np.zeros(jobs + len(capacities), np.int32)
        for (partner_class, name), (asked, room, reserve) in overflowing.items():
            k = partner_class * TERRAIN_LAW_JOB_COUNTERS + TERRAIN_LAW_JOB_NAMES.index(name)
            capacities[k], reserves[k], counters[jobs + k] = room, reserve, asked
        # within the room: no warning
        k = TERRAIN_PARTNER_HULL * TERRAIN_LAW_JOB_COUNTERS + TERRAIN_LAW_JOB_NAMES.index("edges")
        capacities[k], counters[jobs + k] = 8, 8
        capture = StdOutCapture()
        capture.begin()
        try:
            wp.launch(
                terrain_law_spill_check,
                dim=1,
                inputs=[
                    wp.array(counters, dtype=wp.int32, device="cpu"),
                    0,
                    1,
                    jobs,
                    wp.array(capacities, dtype=wp.int32, device="cpu"),
                    wp.array(reserves, dtype=wp.int32, device="cpu"),
                ],
                device="cpu",
            )
            wp.synchronize()
        finally:
            message = capture.end()
        lines = [line for line in message.splitlines() if line.startswith("Warning: Terrain-law")]
        self.assertEqual(len(lines), len(overflowing) + 1, msg=message)
        self.assertIn("Terrain-law box class: its rule F face job list overflowed 9 > 5", message)
        self.assertIn("Terrain-law cylinder class: its candidate pool overflowed 70 > 64", message)
        self.assertIn("took its reserve (rows unchanged)", message)
        self.assertIn("Raise NEWTON_TERRAIN_LAW_SHADOW_JOBS", message)
        self.assertIn("Terrain-law hull class: its scan's runs overflowed 12 > 11", message)
        self.assertIn("Rebuild the collision pipeline", message)
        self.assertIn("Terrain-law cylinder class: the reserve of 2 overflowed too", message)
        self.assertEqual(message.count("overflowed too"), 1, msg=message)
        self.assertNotIn("edge job list", message)

    def test_spill_overflow_raises_terrain_rows_cpu(self):
        """Raise ``terrain_rows`` when the spill region, sized at the scene level, has no room for a pair's rows, and
        keep the pair's first rows by priority in its walk slots.

        The captured sphere's nine rows exceed its walk slots: with ``NEWTON_TERRAIN_LAW_SHADOW_ROWS=8`` the spill
        region is one row short, the pair keeps eight of its rows (every plane group's deepest first, the law's eight
        deepest here, every row its own plane) in its walk slots with a ``TERRAIN_PAIR_ROW_OVERFLOW`` state, the check
        warns with the rows asked for against the region's, and the capacity check raises; with 9 the rows fit.
        (Before, the pair kept no row.)
        """
        model, shapes, law = _captured_sphere("cpu")
        with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW_ROWS": "8"}):
            pipeline = tu.shadow_pipeline(model, **PIPELINE)
        shadow = pipeline._terrain_law_shadow
        self.assertEqual(shadow.spill_rows, 8)
        capture = StdOutCapture()
        capture.begin()
        try:
            _contacts(pipeline, model)
            rows = shadow.rows()
            # flush the device's printf before the capture ends
            wp.synchronize()
        finally:
            message = capture.end()
        self.assertIn("Terrain-law spill region overflowed 9 > 8", message)
        self.assertEqual(int(rows["count"][0]), TERRAIN_POINT_WALK_SLOTS)
        self.assertEqual(int(rows["state"][0]), int(TERRAIN_PAIR_ROW_OVERFLOW))
        self.assertLess(int(shadow._out.first.numpy()[shadow._listed[0]]), shadow.slot_count)
        deepest = sorted(law, key=lambda r: r.d)[:TERRAIN_POINT_WALK_SLOTS]
        _assert_law_rows(self, deepest, tu.shadow_rows(pipeline)[shapes[0]], "sphere")
        self.assertEqual(shadow.overflow_counts(), (1, 0, 0))
        self.assertEqual(shadow.rows_spilled(), 9)
        with self.assertRaisesRegex(RuntimeError, "terrain_rows"):
            pipeline.narrow_phase.check_buffer_capacity()
        with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW_ROWS": "9"}):
            pipeline = tu.shadow_pipeline(model, **PIPELINE)
        _contacts(pipeline, model)
        pipeline.narrow_phase.check_buffer_capacity()
        _assert_law_rows(self, law, tu.shadow_rows(pipeline)[shapes[0]], "sphere")

    def test_spill_room_goes_to_pairs_that_fit_cpu(self):
        """Give a later pair whose rows fit the room an earlier, larger pair found too small for its own.

        Two capsules across ridges defer: the first has 12 rows, the second 10, and the spill region holds 10. The
        first finds no room (it keeps its eight walk slots' worth, flagged) and reserves nothing, so the second keeps
        all its rows: whichever asks first, the rows asked for are 22. (Before, the failed reservation took the room
        too, and the second pair kept no row either.)
        """
        rng = np.random.default_rng(2)
        heights = _corrugated(rng)
        hf = tu.reference_surface(tu.heightfield_model(heights, 2.0, 2.0, device="cpu"))
        lying = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
        bodies, laws = [], []
        for half_height, y in ((0.8, 0.05), (0.9, 0.35)):
            dims = {"radius": 0.03, "half_height": half_height}
            top = max(hf.height(x, y) for x in np.linspace(-0.95, 0.95, 400))
            pose = np.array([0.0, y, top + 0.03 + 0.002, *lying], np.float32).astype(np.float64)
            bodies.append(("capsule", dims, pose))
            laws.append(ref.pair_rows(hf, ref.make_core("capsule", dims, pose), dims["radius"] + 0.04))
        self.assertEqual([len(law) for law in laws], [12, 10])
        model, _terrain, shapes = tu.body_scene(heights, 2.0, 2.0, bodies, device="cpu")
        with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW_ROWS": "10"}):
            pipeline = tu.shadow_pipeline(model, **PIPELINE)
        _contacts(pipeline, model)
        shadow = pipeline._terrain_law_shadow
        rows = shadow.rows()
        self.assertEqual(rows["count"].tolist(), [TERRAIN_POINT_WALK_SLOTS, 10])
        self.assertEqual(rows["state"].tolist(), [int(TERRAIN_PAIR_ROW_OVERFLOW), 0])
        self.assertEqual(shadow.rows_spilled(), 22)
        self.assertEqual(shadow.overflow_counts(), (1, 0, 0))
        got = tu.shadow_rows(pipeline)
        _assert_law_rows(self, laws[1], got[shapes[1]], "the capsule that fits")
        self.assertEqual(len(got[shapes[0]]), TERRAIN_POINT_WALK_SLOTS)
        self.assertEqual(tu.missing_rows(got[shapes[0]], laws[0], q_tol=1.0e-5, angle_tol=1.0e-3, d_tol=1.0e-5), [])

    def test_unset_flag_runs_no_shadow_cpu(self):
        """Run no shadow without the flag, and record why: by default the law writes the contacts instead, and with
        its opt-out (``NEWTON_TERRAIN_LAW=none``) no terrain features, table or buffer are built."""
        model, _terrain = _mixed_scene("cpu")
        off = {"enabled": False, "reason": "Off: NEWTON_TERRAIN_LAW_SHADOW lists no partner class."}
        with _unset():
            default = newton.CollisionPipeline(model, **PIPELINE)
        selection = default.get_collision_selection()
        self.assertEqual(selection["heightfield"]["law_shadow"], off)
        self.assertTrue(selection["heightfield"]["law"]["writes"])
        self.assertNotIn("NEWTON_TERRAIN_LAW", selection["environment"]["applied"])
        opted_out = tu.terrain_route_pipeline(model, **PIPELINE)
        self.assertIsNone(opted_out._terrain_law_shadow)
        selection = opted_out.get_collision_selection()
        self.assertEqual(selection["heightfield"]["law_shadow"], off)
        self.assertFalse(selection["heightfield"]["law"]["writes"])
        self.assertNotIn("NEWTON_TERRAIN_LAW_SHADOW", selection["environment"]["applied"])

    def test_deferred_finish_launch_is_a_fixed_grid_cpu(self):
        """Launch the finish of deferred pairs with a fixed grid whatever the class's size, striding over the deferred
        list.

        The deferred pairs are found on the device, during the collide; ``TERRAIN_LAW_DEFERRED_BLOCKS`` blocks of lanes
        with their global scratch take them in turn, so that a collide with nothing deferred costs one near-empty
        launch instead of one thread per pair of the class (each of which, before, initialized the finish's locals).
        """
        model, _terrain = _mixed_scene("cpu")
        pipeline = tu.shadow_pipeline(model, **PIPELINE)
        shadow = pipeline._terrain_law_shadow
        launches = []
        launch_tiled = wp.launch_tiled

        def record(kernel, *args, **kwargs):
            if kernel is terrain_law_point.terrain_point_rows_finish:
                launches.append((kwargs["dim"], kwargs["block_dim"]))
            return launch_tiled(kernel, *args, **kwargs)

        with patch.object(wp, "launch_tiled", record):
            _contacts(pipeline, model)
        begin = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_THREAD)[0]
        end = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_TILE)[1]
        blocks = terrain_finish_blocks(int(shadow.table.candidates[begin:end].max()))
        self.assertEqual(launches, [(blocks, terrain_law_block_dim("cpu"))])
        self.assertNotEqual(sum(shadow.describe()["pairs"]["point"].values()), blocks)
        self.assertGreater(shadow.describe()["bytes"]["finish_scratch"], 0)

    def test_route_record_cpu(self):
        """Record the shadow's classes, pairs per lane class, launch shapes and memory in the collision selection."""
        model, _terrain = _mixed_scene("cpu")
        pipeline = tu.shadow_pipeline(model, **PIPELINE)
        selection = pipeline.get_collision_selection()
        record = selection["heightfield"]["law_shadow"]
        self.assertTrue(record["enabled"])
        self.assertEqual(record["classes"], ["point"])
        table = pipeline._terrain_law_shadow.table
        lanes = {
            name: table.segment(TERRAIN_PARTNER_POINT, lane)[1] - table.segment(TERRAIN_PARTNER_POINT, lane)[0]
            for name, lane in (("thread", TERRAIN_LANE_THREAD), ("tiled", TERRAIN_LANE_TILE))
        }
        self.assertEqual(record["pairs"], {"point": lanes})
        self.assertEqual(sum(lanes.values()), 12)
        self.assertEqual(record["slots"], 12 * TERRAIN_POINT_WALK_SLOTS)
        # The spill region: from the pairs' own bounds on their rows, the largest and a share of their sum.
        begin = table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_THREAD)[0]
        bounds = table.candidates[begin : begin + 12]
        candidates = int(bounds.max())
        self.assertEqual(record["spill_rows"], candidates + -(-int(bounds.sum()) // TERRAIN_LAW_SPILL_SHARE))
        # The one-thread kernel over the class, the tiled kernel over its queue when it has a tiled segment, the
        # class's finish of deferred pairs and the spill check, with the dimensions fixed at construction.
        self.assertEqual(record["launches"], 3 + int(lanes["tiled"] > 0))
        self.assertEqual(
            record["launch_shapes"],
            {
                "point": {
                    "block_dim": terrain_law_block_dim("cpu"),
                    "thread_dim": 12,
                    "tiled_blocks": min(lanes["tiled"], TERRAIN_LAW_TILE_BLOCKS),
                    "deferred_blocks": terrain_finish_blocks(candidates),
                    "deferred_candidates": candidates,
                    "deferred_limited": False,
                    "walk_slots": TERRAIN_POINT_WALK_SLOTS,
                }
            },
        )
        self.assertEqual(selection["environment"]["applied"]["NEWTON_TERRAIN_LAW_SHADOW"], "point")
        # Without heightfield pairs the flag builds nothing.
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_shape_sphere(builder.add_body(), radius=0.1)
        flat = tu.shadow_pipeline(builder.finalize(device="cpu"), **PIPELINE)
        self.assertIsNone(flat._terrain_law_shadow)
        self.assertIsNone(flat.get_collision_selection()["heightfield"])
        # Nor without a terrain partner of a listed class (no terrain features are built).
        builder = newton.ModelBuilder()
        terrain = newton.Heightfield(data=np.zeros((5, 5), np.float32), nrow=5, ncol=5, hx=0.2, hy=0.2)
        builder.add_shape_heightfield(heightfield=terrain)
        builder.add_shape_box(builder.add_body(), hx=0.05, hy=0.05, hz=0.05)
        boxes = tu.shadow_pipeline(builder.finalize(device="cpu"), **PIPELINE)
        self.assertIsNone(boxes._terrain_law_shadow)
        self.assertEqual(
            boxes.get_collision_selection()["heightfield"]["law_shadow"],
            {"enabled": False, "reason": "No heightfield pair has a partner of the listed classes."},
        )

    def test_route_record_of_solid_classes_cpu(self):
        """Record the solid classes' pairs per lane class, launches, data, job scale, kernel grids, job capacities and
        reserves (a barrel cylinder's in the cylinder class): one block size, one thread per pair for the whole-pair rejects, the job threads for
        the others, the grid of the finish of deferred pairs, and job lists and a candidate pool sized from the pairs'
        own bounds, the scan's jobs exactly, each with a reserve of the class's largest pairs' work."""
        rng = np.random.default_rng(8)
        heights = np.round(rng.uniform(0.0, 0.05, (17, 17)) / 0.005) * 0.005
        bodies = [
            ("box", {"half": (0.05, 0.04, 0.03)}, [-0.4, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0]),
            ("box", {"half": (0.5, 0.04, 0.03)}, [0.0, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0]),
            ("cylinder", {"radius": 0.03, "half_height": 0.04}, [0.4, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0]),
            ("cone", {"radius": 0.03, "half_height": 0.04}, [-0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
            ("ellipsoid", {"semi": (0.05, 0.04, 0.03)}, [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
            ("hull", {"verts": su.sole_vertices()}, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        ]
        model, _terrain, _shapes = su.solid_scene(heights, 0.8, 0.8, bodies, device="cpu")
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
        terrain = newton.Heightfield(data=np.zeros((9, 9), np.float32), nrow=9, ncol=9, hx=0.4, hy=0.4)
        builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
        builder.add_shape_box(builder.add_body(xform=wp.transform(wp.vec3(-0.2, 0.0, 0.05))), hx=0.03, hy=0.03, hz=0.03)
        barrel = builder.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, 0.05)))
        builder.add_shape(body=barrel, type=newton.GeoType.CYLINDER, scale=wp.vec3(0.03, 0.04, 0.2), cfg=cfg)
        for scene in (model, builder.finalize(device="cpu")):
            pipeline = tu.shadow_pipeline(scene, classes="box,cylinder,hull", **PIPELINE)
            record = pipeline.get_collision_selection()["heightfield"]["law_shadow"]
            self.assertTrue(record["enabled"])
            self.assertEqual(record["classes"], ["box", "cylinder", "hull"])
            table = pipeline._terrain_law_shadow.table
            # the spill check
            launches = 1
            for partner_class, name in (
                (TERRAIN_PARTNER_BOX, "box"),
                (TERRAIN_PARTNER_CYLINDER, "cylinder"),
                (TERRAIN_PARTNER_HULL, "hull"),
            ):
                lanes = {}
                for lane, lane_name in ((TERRAIN_LANE_THREAD, "thread"), (TERRAIN_LANE_TILE, "tiled")):
                    begin, end = table.segment(partner_class, lane)
                    lanes[lane_name] = end - begin
                self.assertEqual(record["pairs"][name], lanes)
                # with pairs: the class's launches (whole-pair rejects, scan, job kernels, close, rule R, small-pair
                # finish, finish of deferred pairs)
                count = sum(lanes.values())
                if count:
                    begin = table.segment(partner_class, TERRAIN_LANE_THREAD)[0]
                    end = begin + count
                    candidates = int(table.candidates[begin:end].max())
                    walk = int(TERRAIN_PARTNER_WALK_SLOTS[partner_class])
                    capacities = terrain_solid_job_capacities(
                        table.vertices[begin:end],
                        table.candidates[begin:end],
                        walk,
                        scan_run=int(TERRAIN_PARTNER_SCAN_RUN[partner_class]),
                        sides=table.sides[begin:end],
                    )
                    names = (*TERRAIN_SOLID_JOB_LISTS, "candidates", "scans")
                    self.assertEqual(sorted(capacities), sorted(names))
                    # the scan's jobs: every pair's runs of window vertices at its largest window
                    run = int(TERRAIN_PARTNER_SCAN_RUN[partner_class])
                    scans = int(np.sum(-(-table.vertices[begin:end].astype(np.int64) // run)))
                    self.assertEqual(capacities["scans"], scans)
                    reserves = terrain_solid_job_reserves(
                        table.vertices[begin:end],
                        table.candidates[begin:end],
                        scan_run=run,
                        sides=table.sides[begin:end],
                    )
                    # the reserve: the work of up to TERRAIN_SOLID_RESERVE_PAIRS pairs of the class's largest bounds,
                    # corner lines among the edges and creases, the border's walls per scan run and border edges'
                    # clips in its own list
                    vertices = int(table.vertices[begin:end].max())
                    v = table.vertices[begin:end].astype(np.int64)
                    sides = table.sides[begin:end].astype(np.int64)
                    reserve_pairs = reserves["candidates"] // candidates
                    self.assertTrue(1 <= reserve_pairs <= TERRAIN_SOLID_RESERVE_PAIRS, msg=reserves)
                    bounds = {
                        "faces": int(np.max(2 * v)),
                        "edges": int(np.max(3 * v + 4)),
                        "thetas": int(np.max(3 * v + 4)),
                        "vertices": vertices,
                        "border": int(np.max(4 * -(-v // run) + 2 * sides)),
                    }
                    expected = {name: reserve_pairs * bound for name, bound in bounds.items()}
                    expected["candidates"] = reserve_pairs * candidates
                    expected["scans"] = reserve_pairs * -(-vertices // run)
                    self.assertEqual(reserves, expected)
                    # the job kernels' threads: at most the longest list's room (they stride over its reserve too)
                    longest = max(capacities[n] for n in (*TERRAIN_SOLID_JOB_LISTS, "scans"))
                    launches += len(TERRAIN_SOLID_LAUNCHES)
                    self.assertEqual(
                        record["launch_shapes"][name],
                        {
                            "block_dim": terrain_law_block_dim("cpu"),
                            "pair_dim": count,
                            "job_threads": min(TERRAIN_SOLID_JOB_THREADS_CPU, longest),
                            "border_threads": min(TERRAIN_SOLID_JOB_THREADS_CPU, capacities["border"]),
                            "pair_job_threads": min(TERRAIN_SOLID_JOB_THREADS_CPU, longest, count),
                            "finish_blocks": min(TERRAIN_SOLID_JOB_THREADS_CPU, count),
                            "jobs": capacities,
                            "jobs_reserve": reserves,
                            "deferred_blocks": terrain_finish_blocks(candidates),
                            "deferred_candidates": candidates,
                            "deferred_limited": False,
                            "walk_slots": walk,
                        },
                    )
            self.assertEqual(record["launches"], launches)
            self.assertEqual(record["job_scale"], 1.0)
            self.assertGreater(record["bytes"]["solid_data"], 0)
            self.assertGreater(record["bytes"]["jobs"], 0)
            self.assertEqual(record["bytes"]["work_lists"], 0)
            self.assertGreater(record["bytes"]["finish_scratch"], 0)
        # both lane classes of the box class are in the first scene
        self.assertTrue(
            all(
                v > 0
                for v in tu.shadow_pipeline(model, classes="box", **PIPELINE)
                .get_collision_selection()["heightfield"]["law_shadow"]["pairs"]["box"]
                .values()
            )
        )


devices = get_test_devices()
for _name, _func in (
    ("test_shadow_keeps_contacts_bitwise", test_shadow_keeps_contacts_bitwise),
    ("test_stale_features_give_no_rows", test_stale_features_give_no_rows),
    ("test_long_capsule_keeps_every_support", test_long_capsule_keeps_every_support),
    ("test_sphere_keeps_every_support_over_rough_terrain", test_sphere_keeps_every_support_over_rough_terrain),
    ("test_shadow_follows_the_pair_policy", test_shadow_follows_the_pair_policy),
    ("test_high_cores_leave_at_the_tile_max", test_high_cores_leave_at_the_tile_max),
    ("test_tiled_pairs_that_exit_are_not_queued", test_tiled_pairs_that_exit_are_not_queued),
    ("test_shadow_launches_take_one_block_size", test_shadow_launches_take_one_block_size),
):
    add_function_test(TestTerrainLawShadow, _name, _func, devices=devices)
add_function_test(
    TestTerrainLawShadow,
    "test_shadow_runs_under_graph_capture",
    test_shadow_runs_under_graph_capture,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's shadow route: the flag, the side buffer, the pair policy, capture and today's contacts.

``NEWTON_TERRAIN_LAW_SHADOW`` lists partner classes whose terrain-law kernels run after the narrow phase into a
side buffer that nothing reads. With it set, today's contacts must stay bit for bit what they are without it; the
shadow must follow the pipeline's pair policy, run inside CUDA graph capture, give stale features no row and raise
``terrain_stale``, and raise ``terrain_rows`` on a pair whose rows overflow.
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
)
from newton._src.geometry.terrain_law import (
    TERRAIN_LAW_FULL_BLOCKS,
    TERRAIN_LAW_TILE_BLOCKS,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_ROW_OVERFLOW,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    terrain_law_block_dim,
)
from newton._src.geometry.terrain_law_point import TERRAIN_POINT_ROWS
from newton._src.geometry.terrain_law_solid import (
    TERRAIN_SOLID_FULL_THREADS,
    TERRAIN_SOLID_THREAD_VERTICES,
    TERRAIN_SOLID_TILE_BLOCKS,
)
from newton._src.geometry.terrain_shadow import TERRAIN_LAW_KERNELS, parse_terrain_law_shadow
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_solid_utils as su
from newton.tests import terrain_law_test_utils as tu
from newton.tests.unittest_utils import (
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
    """An environment without NEWTON_TERRAIN_LAW_SHADOW."""
    environ = {k: v for k, v in os.environ.items() if k != "NEWTON_TERRAIN_LAW_SHADOW"}
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
            with _unset():
                today = newton.CollisionPipeline(model, deterministic=deterministic, **PIPELINE)
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
    """Capture the shadow with the collide in a CUDA graph and replay it to the rows of an uncaptured collide.

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
    # Move the bodies, so that the replay must recompute the rows.
    moved = model.state()
    q = moved.body_q.numpy()
    q[:, 2] += 0.5
    moved.body_q.assign(q)
    pipeline.collide(moved, contacts)
    test.assertEqual(int(shadow.rows()["count"].sum()), 0)
    with scoped_capture_after_gc(device=device) as capture:
        pipeline.collide(state, contacts)
    wp.capture_launch(capture.graph)
    replayed = shadow.rows()
    for name in ("count", "state", "q", "n", "d", "feature"):
        np.testing.assert_array_equal(replayed[name], expected[name], err_msg=name)
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


def test_row_overflow_raises_terrain_rows(test, device):
    """Keep a pair's supports first when its rows overflow the slots, and raise ``terrain_rows``.

    A long capsule lies across ridges of different heights: each ridge in reach is a support (its own plane), more
    than the point class's eight slots. The kept rows are the float64 finish's under the same row limit.
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
    rows = pipeline._terrain_law_shadow.rows()
    test.assertEqual(int(rows["count"][0]), TERRAIN_POINT_ROWS)
    test.assertEqual(int(rows["state"][0]) & int(TERRAIN_PAIR_ROW_OVERFLOW), int(TERRAIN_PAIR_ROW_OVERFLOW))
    status = pipeline.narrow_phase.buffer_capacity_status()
    test.assertEqual([k for k, v in status.items() if v], ["terrain_rows"])
    test.assertEqual(pipeline._terrain_law_shadow.overflow_counts(), (1, 0))
    core = ref.make_core("capsule", dims, pose)
    unlimited = ref.pair_rows(hf, core, dims["radius"] + 0.04)
    test.assertGreater(len(unlimited), TERRAIN_POINT_ROWS)
    raw = ref.capsule_law(hf, core, dims["radius"] + 0.04, hf.window_features(hf.cells_in_box(*_box(core, 0.07))))
    band = tu.kernel_band(hf, core, dims["radius"] + 0.04)
    limited = ref.finish(raw, row_limit=TERRAIN_POINT_ROWS, group_offset=4.0 * band)
    got = tu.shadow_rows(pipeline)[shapes[0]]
    test.assertEqual(tu.missing_rows(limited, got, q_tol=1.0e-5, angle_tol=1.0e-3, d_tol=1.0e-5), [])


def _box(core, reach):
    lo, hi = core.aabb()
    return lo[:2] - reach, hi[:2] + reach


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
    work = next(extra for kind, *_launch, extra in shadow._launches if kind == "thread")
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
        sorted(blocks), ["terrain_point_rows", "terrain_point_rows_full", "terrain_point_rows_tiled"], msg=blocks
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

    def test_unset_flag_builds_nothing_cpu(self):
        """Build no terrain features, table or buffer without the flag, and record why in the route record."""
        model, _terrain = _mixed_scene("cpu")
        with _unset():
            pipeline = newton.CollisionPipeline(model, **PIPELINE)
        self.assertIsNone(pipeline._terrain_law_shadow)
        record = pipeline.get_collision_selection()["heightfield"]["law_shadow"]
        self.assertEqual(record, {"enabled": False, "reason": "Off: NEWTON_TERRAIN_LAW_SHADOW lists no partner class."})
        self.assertNotIn("NEWTON_TERRAIN_LAW_SHADOW", pipeline.get_collision_selection()["environment"]["applied"])

    def test_full_finish_launch_is_a_fixed_grid_cpu(self):
        """Launch the full-finish kernel with a fixed grid whatever the class's size, striding over the deferred list.

        The deferred pairs are found on the device, during the collide; ``TERRAIN_LAW_FULL_BLOCKS`` blocks of lanes
        with their global scratch take them in turn, so that a collide with nothing deferred costs one near-empty
        launch instead of one thread per pair of the class (each of which, before, initialized the finish's locals).
        """
        model, _terrain = _mixed_scene("cpu")
        pipeline = tu.shadow_pipeline(model, **PIPELINE)
        shadow = pipeline._terrain_law_shadow
        launches = []
        launch_tiled = wp.launch_tiled

        def record(kernel, *args, **kwargs):
            if kernel is terrain_law_point.terrain_point_rows_full:
                launches.append((kwargs["dim"], kwargs["block_dim"]))
            return launch_tiled(kernel, *args, **kwargs)

        with patch.object(wp, "launch_tiled", record):
            _contacts(pipeline, model)
        self.assertEqual(launches, [(TERRAIN_LAW_FULL_BLOCKS, terrain_law_block_dim("cpu"))])
        self.assertNotEqual(sum(shadow.describe()["pairs"]["point"].values()), TERRAIN_LAW_FULL_BLOCKS)
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
        self.assertEqual(record["slots"], 12 * TERRAIN_POINT_ROWS)
        # The one-thread kernel over the class, the tiled kernel over its queue when it has a tiled segment, and the
        # class's full-finish kernel, with the dimensions fixed at construction.
        self.assertEqual(record["launches"], 2 + int(lanes["tiled"] > 0))
        self.assertEqual(
            record["launch_shapes"],
            {
                "point": {
                    "block_dim": terrain_law_block_dim("cpu"),
                    "thread_dim": 12,
                    "tiled_blocks": min(lanes["tiled"], TERRAIN_LAW_TILE_BLOCKS),
                    "full_blocks": TERRAIN_LAW_FULL_BLOCKS,
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
        """Record the solid classes' pairs per lane class, launches, data, lane choice, kernel grids and coreless
        pairs."""
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
        for scene, unsupported in ((model, 0), (builder.finalize(device="cpu"), 1)):
            pipeline = tu.shadow_pipeline(scene, classes="box,cylinder,hull", **PIPELINE)
            record = pipeline.get_collision_selection()["heightfield"]["law_shadow"]
            self.assertTrue(record["enabled"])
            self.assertEqual(record["classes"], ["box", "cylinder", "hull"])
            table = pipeline._terrain_law_shadow.table
            launches = 0
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
                # with pairs: the one-thread kernel over the class, the tiled grid when the class has tiled-lane
                # partners (only their windows can exceed the one-thread kernel's), and the full kernel
                if sum(lanes.values()):
                    launches += 2 + int(lanes["tiled"] > 0)
            self.assertEqual(record["launches"], launches)
            self.assertEqual(record["thread_vertices"], TERRAIN_SOLID_THREAD_VERTICES)
            self.assertEqual(record["tile_blocks"], TERRAIN_SOLID_TILE_BLOCKS)
            self.assertEqual(record["full_threads"], TERRAIN_SOLID_FULL_THREADS)
            self.assertEqual(record["unsupported_pairs"], unsupported)
            self.assertGreater(record["bytes"]["solid_data"], 0)
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
    ("test_row_overflow_raises_terrain_rows", test_row_overflow_raises_terrain_rows),
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

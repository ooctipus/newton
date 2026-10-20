# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's static pair table: filter, partner classes, lane classes, segments and output slots."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.terrain_contacts import (
    TERRAIN_LANE_CELLS,
    TERRAIN_LANE_CLASSES,
    TERRAIN_LANE_THREAD,
    TERRAIN_LANE_TILE,
    TERRAIN_PARTNER_BOX,
    TERRAIN_PARTNER_CLASSES,
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_PARTNER_HULL,
    TERRAIN_PARTNER_POINT,
    TERRAIN_PARTNER_ROWS,
    TerrainPairTable,
    heightfield_pair_table,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

_CLASS = {
    "sphere": TERRAIN_PARTNER_POINT,
    "big_sphere": TERRAIN_PARTNER_POINT,
    "capsule": TERRAIN_PARTNER_POINT,
    "box": TERRAIN_PARTNER_BOX,
    "cylinder": TERRAIN_PARTNER_CYLINDER,
    "cone": TERRAIN_PARTNER_CYLINDER,
    "barrel": TERRAIN_PARTNER_HULL,
    "ellipsoid": TERRAIN_PARTNER_HULL,
    "hull": TERRAIN_PARTNER_HULL,
}


def _cube():
    vertices = np.array([(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)], np.float32) * 0.04
    faces = [
        (0, 1, 3),
        (0, 3, 2),
        (4, 6, 7),
        (4, 7, 5),
        (0, 4, 5),
        (0, 5, 1),
        (2, 3, 7),
        (2, 7, 6),
        (0, 2, 6),
        (0, 6, 4),
        (1, 5, 7),
        (1, 7, 3),
    ]
    return newton.Mesh(vertices=vertices, indices=np.asarray(faces, np.int32).ravel())


def _scene(device, *, worlds=3):
    """A global heightfield (0.1 m cells) and, per world, one partner of every kind plus excluded ones.

    Excluded per world: a triangle mesh, a sphere that takes no part in shape collision, one in collision group
    2 (the heightfield is in group 1) and one filtered against the heightfield.
    """
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
    terrain = newton.Heightfield(data=np.zeros((21, 21), np.float32), nrow=21, ncol=21, hx=1.0, hy=1.0)
    heightfield = builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
    cube = _cube()
    kinds = {}
    excluded = []
    for w in range(worlds):
        builder.begin_world()
        body = builder.add_body(xform=wp.transform((0.1 * w, 0.0, 0.5), wp.quat_identity()))
        shapes = {
            "sphere": builder.add_shape_sphere(body, radius=0.03, cfg=cfg),
            "big_sphere": builder.add_shape_sphere(body, radius=0.25, cfg=cfg),
            "capsule": builder.add_shape_capsule(body, radius=0.03, half_height=0.1, cfg=cfg),
            "box": builder.add_shape_box(body, hx=0.05, hy=0.04, hz=0.03, cfg=cfg),
            "cylinder": builder.add_shape_cylinder(body, radius=0.02, half_height=0.06, cfg=cfg),
            "cone": builder.add_shape_cone(body, radius=0.03, half_height=0.04, cfg=cfg),
            "barrel": builder.add_shape_cylinder(body, radius=0.02, half_height=0.06, barrel_radius=0.2, cfg=cfg),
            "ellipsoid": builder.add_shape_ellipsoid(body, rx=0.03, ry=0.02, rz=0.01, cfg=cfg),
            "hull": builder.add_shape_convex_hull(body, mesh=cube, cfg=cfg),
        }
        for kind, shape in shapes.items():
            kinds[shape] = (kind, w)
        excluded.append(builder.add_shape_mesh(body, mesh=cube, cfg=cfg))
        excluded.append(builder.add_shape_sphere(body, radius=0.03, cfg=builder.ShapeConfig(has_shape_collision=False)))
        excluded.append(
            builder.add_shape_sphere(
                body, radius=0.03, cfg=builder.ShapeConfig(margin=0.01, gap=0.01, collision_group=2)
            )
        )
        filtered = builder.add_shape_sphere(body, radius=0.03, cfg=cfg)
        builder.add_shape_collision_filter_pair(heightfield, filtered)
        excluded.append(filtered)
        builder.end_world()
    return builder.finalize(device=device), heightfield, kinds, excluded


def _window_cells(radius, reach, cell):
    """Cells of a partner's largest window: its bounding sphere plus reach, at any alignment."""
    side = math.ceil(2.0 * (radius + reach) / cell) + 1
    return side * side


def test_pair_table_from_the_collision_filter(test, device):
    """List every (heightfield, convex partner) pair the filter admits, by partner class, lane class and world."""
    model, heightfield, kinds, excluded = _scene(device)
    table = TerrainPairTable(model)
    pairs = table.pairs_host
    test.assertEqual(table.pair_count, len(kinds))
    test.assertEqual({tuple(p) for p in pairs.tolist()}, {(heightfield, s) for s in kinds})
    test.assertFalse(set(pairs[:, 1].tolist()) & set(excluded))
    np.testing.assert_array_equal(table.pairs.numpy(), pairs)
    radius = model.shape_collision_radius.numpy()
    for i, (a, b) in enumerate(pairs):
        kind, world = kinds[int(b)]
        test.assertEqual(int(a), heightfield)
        test.assertEqual(table.partner_class[i], _CLASS[kind], msg=kind)
        test.assertEqual(table.world[i], world)
        lane = TERRAIN_LANE_THREAD if _window_cells(radius[b], 0.04, 0.1) <= TERRAIN_LANE_CELLS else TERRAIN_LANE_TILE
        test.assertEqual(table.lane_class[i], lane, msg=kind)
    # The scene has partners in both lane classes.
    test.assertEqual(
        table.lane_class[[kinds[int(b)][0] == "sphere" for b in pairs[:, 1]]].tolist(), [TERRAIN_LANE_THREAD] * 3
    )
    test.assertEqual(
        table.lane_class[[kinds[int(b)][0] == "big_sphere" for b in pairs[:, 1]]].tolist(), [TERRAIN_LANE_TILE] * 3
    )
    # Sorted by (partner class, lane class, world): every segment is one contiguous range.
    key = table.partner_class * TERRAIN_LANE_CLASSES + table.lane_class
    test.assertTrue(np.all(np.diff(key) >= 0))
    rows_total = 0
    for partner in range(TERRAIN_PARTNER_CLASSES):
        for lane in range(TERRAIN_LANE_CLASSES):
            begin, end = table.segment(partner, lane)
            segment = slice(begin, end)
            test.assertTrue(np.all(table.partner_class[segment] == partner))
            test.assertTrue(np.all(table.lane_class[segment] == lane))
            test.assertTrue(np.all(np.diff(table.world[segment]) >= 0))
            test.assertEqual(end - begin, int(np.sum((table.partner_class == partner) & (table.lane_class == lane))))
            for pair in range(begin, end):
                slots = table.slots(pair)
                test.assertEqual(slots[0], rows_total)
                test.assertEqual(slots[1] - slots[0], TERRAIN_PARTNER_ROWS[partner])
                rows_total = slots[1]
    test.assertEqual(table.slot_count, rows_total)
    test.assertEqual(table.memory_bytes()["pairs"], 8 * len(kinds))


def test_pair_table_without_a_pair_list(test, device):
    """Without the model's pair list, apply the same filter: worlds, collision groups and filter pairs.

    The law's pairs bypass the broad phase, so a superset would make contacts the filter excludes.
    """
    model, heightfield, kinds, excluded = _scene(device)
    with_list = TerrainPairTable(model)
    model.shape_contact_pairs = None
    model.shape_contact_pair_count = 0
    without = TerrainPairTable(model)
    np.testing.assert_array_equal(without.pairs_host, with_list.pairs_host)
    test.assertEqual({tuple(p) for p in without.pairs_host.tolist()}, {(heightfield, s) for s in kinds})
    test.assertFalse(set(without.pairs_host[:, 1].tolist()) & set(excluded))


def test_pair_table_of_a_filtered_heightfield_is_empty(test, device):
    """Give no pair to a heightfield whose every partner is filtered: the model's pair list is empty, not absent."""
    builder = newton.ModelBuilder()
    terrain = newton.Heightfield(data=np.zeros((5, 5), np.float32), nrow=5, ncol=5, hx=0.2, hy=0.2)
    heightfield = builder.add_shape_heightfield(heightfield=terrain)
    sphere = builder.add_shape_sphere(builder.add_body(), radius=0.03)
    builder.add_shape_collision_filter_pair(heightfield, sphere)
    model = builder.finalize(device=device)
    test.assertEqual(model.shape_contact_pair_count, 0)
    table = TerrainPairTable(model)
    test.assertEqual(table.pair_count, 0)
    test.assertEqual(table.slot_count, 0)


class TestTerrainLawPairs(unittest.TestCase):
    def test_heightfield_pair_table_explicit_pairs(self):
        """Orient explicit pairs heightfield-first and drop meshes, non-heightfield and two-heightfield pairs."""
        hfield, mesh, sphere, box = (
            int(newton.GeoType.HFIELD),
            int(newton.GeoType.MESH),
            int(newton.GeoType.SPHERE),
            int(newton.GeoType.BOX),
        )
        types = np.array([hfield, sphere, mesh, box, hfield])
        explicit = np.array([[1, 0], [0, 2], [3, 0], [1, 3], [0, 4], [3, 4], [9, 0]])
        table = heightfield_pair_table(types, None, np.ones(5, bool), explicit_pairs=explicit)
        np.testing.assert_array_equal(table, [[0, 1], [0, 3], [4, 3]])


devices = get_test_devices()
add_function_test(
    TestTerrainLawPairs,
    "test_pair_table_from_the_collision_filter",
    test_pair_table_from_the_collision_filter,
    devices=devices,
)
add_function_test(
    TestTerrainLawPairs, "test_pair_table_without_a_pair_list", test_pair_table_without_a_pair_list, devices=devices
)
add_function_test(
    TestTerrainLawPairs,
    "test_pair_table_of_a_filtered_heightfield_is_empty",
    test_pair_table_of_a_filtered_heightfield_is_empty,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

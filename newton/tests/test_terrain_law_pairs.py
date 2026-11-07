# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's static pair table: filter, partner classes, lane classes, segments, walk slots and the
bounds on every pair's window and candidates."""

import math
import unittest
from types import SimpleNamespace

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
    TERRAIN_PARTNER_WALK_SLOTS,
    TerrainPairTable,
    heightfield_pair_table,
    terrain_window_vertices,
)
from newton._src.geometry.terrain_law import TERRAIN_CELL_WIDEN, TERRAIN_FINISH_MASK_ROWS
from newton._src.geometry.types import GeoType
from newton._src.geometry.utils import compute_shape_radius
from newton.tests import terrain_law_reference as ref
from newton.tests.unittest_utils import add_function_test, get_test_devices

_CLASS = {
    "sphere": TERRAIN_PARTNER_POINT,
    "big_sphere": TERRAIN_PARTNER_POINT,
    "capsule": TERRAIN_PARTNER_POINT,
    "box": TERRAIN_PARTNER_BOX,
    "cylinder": TERRAIN_PARTNER_CYLINDER,
    "cone": TERRAIN_PARTNER_CYLINDER,
    "barrel": TERRAIN_PARTNER_CYLINDER,
    "plane": TERRAIN_PARTNER_BOX,
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
    """A global heightfield (0.1 m cells) and, per world, one partner of every kind (a finite plane on the body
    among them) plus excluded ones.

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
            "plane": builder.add_shape_plane(body=body, width=0.1, length=0.08, cfg=cfg),
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


_ROWS_PER_VERTEX = {"sphere": (6, 1), "big_sphere": (6, 1), "capsule": (16, 3)}
"""Rows a window vertex's features give a point core at most, and the rows its ends' exits and its axis's recovery
add (any other partner: a solid's 15 per vertex, and its 4 recovery rows)."""


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
    for partner in range(TERRAIN_PARTNER_CLASSES):
        for lane in range(TERRAIN_LANE_CLASSES):
            begin, end = table.segment(partner, lane)
            segment = slice(begin, end)
            test.assertTrue(np.all(table.partner_class[segment] == partner))
            test.assertTrue(np.all(table.lane_class[segment] == lane))
            test.assertTrue(np.all(np.diff(table.world[segment]) >= 0))
            test.assertEqual(end - begin, int(np.sum((table.partner_class == partner) & (table.lane_class == lane))))
    # Every pair's bound on its candidates: its largest window's vertices (21 x 21 at most) times the rows a vertex's
    # features give its partner, the rows added after the walk and for a solid the rows of the heightfield's border
    # (four per vertex along the window's sides, 24 more).
    for i, b in enumerate(pairs[:, 1]):
        kind = kinds[int(b)][0]
        side = min(math.floor(2.0 * (radius[b] + 0.04) / 0.1) + 4, 21)
        per_vertex, per_pair = _ROWS_PER_VERTEX.get(kind, (15, 4))
        border = 0 if kind in _ROWS_PER_VERTEX else 4 * 2 * side + 24
        test.assertEqual(table.sides[i], 2 * side, msg=kind)
        test.assertEqual(table.candidates[i], per_vertex * side * side + per_pair + border, msg=kind)
    test.assertEqual(table.memory_bytes()["pairs"], 8 * len(kinds))


def test_pair_table_without_a_pair_list(test, device):
    """Without the model's pair list, apply the same filter: worlds, collision groups and filter pairs.

    The law's pairs bypass the broad phase, so a superset would make contacts the filter excludes.
    """
    model, heightfield, kinds, excluded = _scene(device)
    with_list = TerrainPairTable(model)
    model.shape_contact_pairs = None
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
    test.assertEqual(len(table.candidates), 0)


def _box_cells(hf, lo, hi):
    """The cells ``terrain_box_cells`` gives a footprint box (float32 cell coordinates widened by
    ``TERRAIN_CELL_WIDEN`` cells, clamped to the grid)."""
    u0 = np.float32((np.float32(lo[0]) + np.float32(hf.hx)) / np.float32(hf.dx)) - np.float32(TERRAIN_CELL_WIDEN)
    u1 = np.float32((np.float32(hi[0]) + np.float32(hf.hx)) / np.float32(hf.dx)) + np.float32(TERRAIN_CELL_WIDEN)
    v0 = np.float32((np.float32(lo[1]) + np.float32(hf.hy)) / np.float32(hf.dy)) - np.float32(TERRAIN_CELL_WIDEN)
    v1 = np.float32((np.float32(hi[1]) + np.float32(hf.hy)) / np.float32(hf.dy)) + np.float32(TERRAIN_CELL_WIDEN)
    c0, c1 = max(int(np.floor(u0)), 0), min(int(np.floor(u1)), hf.cols - 1)
    r0, r1 = max(int(np.floor(v0)), 0), min(int(np.floor(v1)), hf.rows - 1)
    return r0, r1, c0, c1


def test_window_bound_holds_for_every_pose(test, device):
    """Bound every pair's window, whatever its pose, by the vertices its shape extent, reach and cells give.

    The scratch of the finish of deferred pairs holds the candidates such a window can give, so this bound is what
    makes it complete. Random positions (on cell lines and off them, 60 m from the origin of a 200 m grid, where
    float32 cell coordinates round coarsest) and orientations of every kind, a convex hull off its shape's origin among
    them (its radius half its local box's diagonal, the model's), each window taken as the kernels take it: the core's
    bounds (its axis-aligned box or its ends) widened by the reach. The sampling checks the argument of
    :func:`~newton._src.geometry.terrain_contacts.terrain_window_vertices`, which holds for every pose: a core lies in
    a ball of its radius, so its box spans at most twice the radius along each axis.
    """
    rng = np.random.default_rng(5)
    # a 200 m grid of 0.1 m cells: its layout is all a window reads
    hf = SimpleNamespace(hx=100.0, hy=100.0, dx=0.1, dy=0.1, rows=2000, cols=2000)
    # a hull whose points lie off its shape's origin: its radius is half its local box's diagonal, not centred there
    blob = rng.normal(size=(24, 3))
    blob = blob / np.linalg.norm(blob, axis=1, keepdims=True) * np.array([0.09, 0.05, 0.03])
    blob = (blob + np.array([0.12, -0.07, 0.05])).astype(np.float32).astype(np.float64)
    shapes = (
        ("sphere", {"radius": 0.25}),
        ("capsule", {"radius": 0.12, "half_height": 0.35}),
        ("box", {"half": (0.0465, 0.0315, 0.1425)}),
        ("cylinder", {"radius": 0.0175, "half_height": 0.1979}),
        ("cone", {"radius": 0.07, "half_height": 0.0816}),
        ("ellipsoid", {"semi": (0.12, 0.22, 0.29)}),
        ("hull", {"verts": blob}),
    )
    reach = 0.04
    for kind, dims in shapes:
        if kind == "hull":
            radius = compute_shape_radius(int(GeoType.CONVEX_MESH), (1.0, 1.0, 1.0), SimpleNamespace(vertices=blob))
            # the points lie off the origin by more than the radius's slack would hide
            test.assertGreater(float(np.linalg.norm(blob, axis=1).max()), radius)
        else:
            radius = {
                "sphere": dims.get("radius", 0.0),
                "capsule": dims.get("radius", 0.0) + dims.get("half_height", 0.0),
                "box": float(np.linalg.norm(dims.get("half", (0.0, 0.0, 0.0)))),
                "cylinder": dims.get("radius", 0.0) + dims.get("half_height", 0.0),
                "cone": dims.get("radius", 0.0) + dims.get("half_height", 0.0),
                "ellipsoid": max(dims.get("semi", (0.0,))),
            }[kind]
        bound = int(terrain_window_vertices(radius, reach, hf.dx, hf.dy, 2001, 2001))
        side = math.floor(2.0 * (radius + reach) / hf.dx) + 4
        test.assertEqual(bound, side * side)
        largest = 0
        widest = 0
        for _ in range(2000):
            x, y = rng.uniform(-60.0, 60.0, 2)
            if rng.uniform() < 0.3:
                x = np.round(x / hf.dx) * hf.dx + rng.choice([0.0, 1e-6, -1e-6])
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            core = ref.make_core(kind, dims, np.r_[x, y, 0.1, q])
            if kind in ("sphere", "capsule"):
                ends = [core.a, core.b] if kind == "capsule" else [core.c]
                lo = np.min(ends, axis=0) - (dims["radius"] + reach)
                hi = np.max(ends, axis=0) + (dims["radius"] + reach)
            else:
                lo, hi = core.aabb()
                lo, hi = lo - reach, hi + reach
            r0, r1, c0, c1 = _box_cells(hf, lo, hi)
            largest = max(largest, (r1 - r0 + 2) * (c1 - c0 + 2))
            widest = max(widest, r1 - r0 + 2, c1 - c0 + 2)
        test.assertLessEqual(largest, bound, msg=kind)
        test.assertLessEqual(widest, side, msg=kind)
        # Not vacuous: windows come within two vertices of the bound's side.
        test.assertGreaterEqual(widest, side - 2, msg=(kind, widest, side))


class TestTerrainLawPairs(unittest.TestCase):
    def test_walk_slots_fit_the_in_place_finish(self):
        """Give every partner class no more walk slots than the in-place finish of a walk takes (its masks' bits)."""
        self.assertLessEqual(max(TERRAIN_PARTNER_WALK_SLOTS), TERRAIN_FINISH_MASK_ROWS)
        self.assertEqual(len(TERRAIN_PARTNER_WALK_SLOTS), TERRAIN_PARTNER_CLASSES)

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
add_function_test(
    TestTerrainLawPairs,
    "test_window_bound_holds_for_every_pose",
    test_window_bound_holds_for_every_pose,
    devices=["cpu"],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

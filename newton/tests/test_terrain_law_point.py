# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's point-core kernels (spheres and capsules), run in shadow beside today's terrain route.

The same checks run on today's contacts (as expected failures) and on the law's shadow rows: today's contacts fail
them, the shadow rows pass. Measured when these tests were written (default terrain flags):

* minimal: spheres resting at flat seams get 2.58 rows and 0.92 ghost normals each today, lying capsules 3.92 rows
  and 1.08 ghosts, where the plane has one row per sphere and one per capsule end;
* complete: today misses 13 of the 24 end supports of the lying capsules, and on rough, pitted and stepped terrain
  supports of spheres and capsules the float64 law finds (9 of 20 spheres and 15 of 20 capsules on the rough tile
  lose one).

The float32 rows equal the float64 law's (:mod:`newton.tests.terrain_law_reference`) on seam-adversarial
placements up to 100 m out, but for decisions within the predicates' band
(:func:`~newton.tests.terrain_law_test_utils.band_decided`).
"""

import glob
import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry import terrain_law_point
from newton._src.geometry.terrain_contacts import TERRAIN_LANE_THREAD, TERRAIN_LANE_TILE, TERRAIN_PARTNER_POINT
from newton._src.geometry.terrain_law import (
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TerrainFeatureData,
    TerrainLawRows,
    TerrainPairInputs,
    TerrainPlanes,
    TerrainWindow,
    terrain_claims,
    terrain_edge_kind,
    terrain_edge_sides,
    terrain_eps,
    terrain_face_values,
    terrain_feature_id,
    terrain_pair_exit,
    terrain_plane,
    terrain_planes,
    terrain_window,
    terrain_window_extent,
    terrain_window_vertex,
)
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_test_utils as tu
from newton.tests.unittest_utils import add_function_test, get_test_devices

R31 = 0.031455
R22 = 0.022
REACH = 4 * tu.SCENE_MARGIN
"""Margins and gaps of a pair: the reach beyond the rounding radius [m]."""
SITES = {
    "interior": (0.0333, 0.0667),
    "diagonal": (0.05, 0.05),
    "axis edge": (0.05, 0.0),
    "vertex": (0.0, 0.0),
    "vertex+0.5mm": (0.0005, 0.0003),
    "vertex+3mm": (0.003, 0.002),
}
"""Sphere sites on a 0.1 m grid, offsets from a vertex [m] (the symptom bench's resting sites)."""
CAPSULE_SITES = {"vertex": (0.0, 0.0), "edge": (0.05, 0.0005), "interior": (0.033, 0.067)}
PIPELINE = {"rigid_contact_max": 1 << 15, "max_triangle_pairs": 1 << 17}
GHOST_ANGLE = np.radians(1.0)
EXITS = int(TERRAIN_PAIR_TILE_EXIT | TERRAIN_PAIR_STALE | TERRAIN_PAIR_FILTERED | TERRAIN_PAIR_OUTSIDE)
"""Pair states of a pair that leaves before its window is walked."""


def _quat_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _axis_angle(axis, angle):
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])


LYING = _axis_angle((0.0, 1.0, 0.0), np.pi / 2.0)
"""A capsule's axis (local z) onto world x."""


def _collide(model, *, shadow: bool):
    pipeline = tu.shadow_pipeline(model, **PIPELINE) if shadow else newton.CollisionPipeline(model, **PIPELINE)
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    return pipeline, contacts, state


# ----------------------------------------------------------------------------------------------------------------
# minimal at flat seams
# ----------------------------------------------------------------------------------------------------------------


def _flat_seams():
    """Spheres resting at seams of a flat 0.1 m grid and capsules lying across them, with the plane's rows."""
    bodies, expected = [], []
    for r in (R31, R22):
        for dx, dy in SITES.values():
            z = r + 2 * tu.SCENE_MARGIN - 0.0005
            bodies.append(("sphere", {"radius": r}, np.array([dx, dy, z, 0.0, 0.0, 0.0, 1.0])))
            expected.append([(np.array([dx, dy, 0.0]), np.array([0.0, 0.0, 1.0]), z)])
    for yaw in (0.0, 30.0, 45.0, 90.0):
        q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), np.radians(yaw)), LYING)
        axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
        for dx, dy in CAPSULE_SITES.values():
            z = R31 + 2 * tu.SCENE_MARGIN - 0.0005
            bodies.append(("capsule", {"radius": R31, "half_height": 0.08}, np.array([dx, dy, z, *q])))
            ends = [np.array([dx, dy, 0.0]) + s * 0.08 * axis for s in (1.0, -1.0)]
            expected.append([(e, np.array([0.0, 0.0, 1.0]), z) for e in ends])
    return bodies, expected


def _minimal_violations(rows, shapes, expected, *, d_tol: float | None = 1.0e-5) -> dict[str, list[str]]:
    """Rows that are not the plane's, by kind: ``count`` (a wrong number of rows), ``missing`` (a plane row missing:
    terrain point within 10 um, normal within 1e-4 rad and, given ``d_tol``, core distance within it) and ``ghost``
    (a normal over 1 deg off the plane's)."""
    violations = {"count": [], "missing": [], "ghost": []}
    for i, (shape, plane) in enumerate(zip(shapes, expected, strict=True)):
        got = rows.get(shape, [])
        if len(got) != len(plane):
            violations["count"].append(f"body {i}: {len(got)} rows, the plane has {len(plane)}")
        missing = tu.missing_rows(plane, got, q_tol=1.0e-5, angle_tol=1.0e-4, d_tol=d_tol)
        if missing:
            violations["missing"].append(f"body {i}: {len(missing)} plane rows missing")
        ghosts = sum(1 for g in got if tu.angle_between(g[1], (0.0, 0.0, 1.0)) > GHOST_ANGLE)
        if ghosts:
            violations["ghost"].append(f"body {i}: {ghosts} ghost normals")
    return violations


def _assert_minimal(test, violations):
    for kind in ("count", "missing", "ghost"):
        test.assertEqual(violations[kind], [], msg=kind)


def test_point_rows_are_minimal_at_flat_seams(test, device):
    """Give a sphere at a flat seam the plane's one row and a lying capsule its two end rows, with no ghost normal."""
    bodies, expected = _flat_seams()
    model, _hfield, shapes = tu.body_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    _assert_minimal(test, _minimal_violations(tu.shadow_rows(pipeline), shapes, expected))


NOISY_FLAT_HEIGHT = 0.05
"""Mean height [m] of :func:`_noisy_flat_seams`' tile."""


def _noisy_flat_seams():
    """Spheres and lying capsules at the seams of a noisy flat tile (+-1.4 um, as the recorded suite terrain's flat
    border), with the plane's rows; returns ``(heights, bodies, expected)``."""
    rng = np.random.default_rng(8)
    heights = NOISY_FLAT_HEIGHT + rng.integers(-6, 7, size=(41, 41)) * 2.4e-7
    sites = [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05), (0.05, 0.05), (0.0005, 0.0003), (0.033, 0.067)]
    bodies, expected = [], []
    for _ in range(4):
        for fx, fy in sites:
            j, i = rng.integers(4, 36, size=2)
            x, y = -2.0 + j * 0.1 + fx, -2.0 + i * 0.1 + fy
            z = NOISY_FLAT_HEIGHT + R31 + 2 * tu.SCENE_MARGIN - 0.0005
            bodies.append(("sphere", {"radius": R31}, np.array([x, y, z, 0.0, 0.0, 0.0, 1.0])))
            expected.append([(np.array([x, y, NOISY_FLAT_HEIGHT]), np.array([0.0, 0.0, 1.0]), z - NOISY_FLAT_HEIGHT)])
            yaw = np.radians(float(rng.choice([0.0, 30.0, 45.0, 90.0])))
            q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), yaw), LYING)
            bodies.append(("capsule", {"radius": R31, "half_height": 0.08}, np.array([x, y, z, *q])))
            axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            ends = [np.array([x, y, NOISY_FLAT_HEIGHT]) + s * 0.08 * axis for s in (1.0, -1.0)]
            expected.append([(e, np.array([0.0, 0.0, 1.0]), z - NOISY_FLAT_HEIGHT) for e in ends])
    return heights, bodies, expected


def test_point_rows_have_the_plane_at_noisy_flat_seams(test, device):
    """Give spheres and lying capsules at the seams of a noisy flat tile the plane's rows, with no ghost normal.

    Both the shadow rows and the float64 law's; their count is a known failure
    (:meth:`TestTerrainLawPoint.test_point_rows_are_minimal_at_noisy_flat_seams_cpu`).
    """
    heights, bodies, expected = _noisy_flat_seams()
    model, _hfield, shapes = tu.body_scene(heights, 2.0, 2.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    violations = _minimal_violations(tu.shadow_rows(pipeline), shapes, expected)
    test.assertEqual(violations["missing"], [])
    test.assertEqual(violations["ghost"], [])
    hf = tu.reference_surface(tu.heightfield_model(heights, 2.0, 2.0, device="cpu"))
    rows = {i: [(r.q, r.n, r.d) for r in _law_rows(hf, *body)] for i, body in enumerate(bodies)}
    violations = _minimal_violations(rows, range(len(bodies)), expected)
    test.assertEqual(violations["missing"], [])
    test.assertEqual(violations["ghost"], [])


# ----------------------------------------------------------------------------------------------------------------
# complete on rough terrain
# ----------------------------------------------------------------------------------------------------------------


def _rough_terrains(rng):
    """A rough tile (5 mm steps), pits and peaks on a plateau, and stairs: 32 x 32 cells of 0.1 m."""
    n = 33
    x = np.linspace(-1.6, 1.6, n)
    X, _Y = np.meshgrid(x, x)
    rough = np.round(rng.uniform(0.0, 0.1, (n, n)) / 0.005) * 0.005
    pits = np.full((n, n), 0.3)
    for i in range(4, n - 4, 6):
        for j in range(4, n - 4, 6):
            pits[i, j] -= 0.04
            pits[i + 2, j + 3] += 0.04
    stairs = np.floor((X + 1.6) / 0.3) * 0.15
    return {"rough": rough, "pits": pits, "stairs": stairs}


def _placements(rng, hf, count):
    """Spheres and capsules on a terrain, a third snapped onto grid lines, their lowest point -2 to +12 mm off it."""
    bodies = []
    for kind in ("sphere", "capsule"):
        for i in range(count):
            r = R31 if i % 2 else R22
            dims = {"radius": r} if kind == "sphere" else {"radius": r, "half_height": 0.08}
            x, y = rng.uniform(-1.2, 1.2, 2)
            if i % 3 == 0:
                x = np.round((x + hf.hx) / hf.dx) * hf.dx - hf.hx
            if i % 4 == 0:
                y = np.round((y + hf.hy) / hf.dy) * hf.dy - hf.hy
            q = np.array([0.0, 0.0, 0.0, 1.0])
            if kind == "capsule":
                tilt = _axis_angle((1.0, 0.0, 0.0), rng.uniform(0.0, 0.4))
                q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)), _quat_mul(tilt, LYING))
            lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
            top = max(
                hf.height(xx, yy)
                for xx in np.linspace(lo[0] - r, hi[0] + r, 7)
                for yy in np.linspace(lo[1] - r, hi[1] + r, 7)
            )
            z = top - lo[2] + r + rng.uniform(-0.002, 0.012)
            bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
    return bodies


def _law_rows(hf, kind, dims, pose):
    """The float64 law's rows of a pair, its plane groups at the float32 finish's offset (four kernel bands)."""
    core = ref.make_core(kind, dims, pose)
    reach = dims["radius"] + REACH
    return ref.pair_rows(hf, core, reach, group_offset=4.0 * tu.kernel_band(hf, core, reach))


def _incomplete(hf, rows, shapes, bodies) -> list[str]:
    """The float64 law's supports that no row has (terrain point within 0.1 mm, normal within 1 deg)."""
    violations = []
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        law = _law_rows(hf, kind, dims, pose)
        missing = tu.missing_rows(law, rows.get(shape, []), q_tol=1.0e-4, angle_tol=np.radians(1.0))
        violations.extend(f"{kind} at {np.round(pose[:3], 4).tolist()}: {row}" for row in missing)
    return violations


def _rough_scenes(device, count):
    rng = np.random.default_rng(3)
    for name, heights in _rough_terrains(rng).items():
        terrain_only = tu.heightfield_model(heights, 1.6, 1.6, device="cpu")
        hf = tu.reference_surface(terrain_only)
        bodies = _placements(rng, hf, count)
        model, hfield, shapes = tu.body_scene(heights, 1.6, 1.6, bodies, device=device)
        yield name, hf, model, hfield, shapes, bodies


def test_point_rows_are_complete(test, device):
    """Keep every support of the float64 law for spheres and capsules on rough, pitted and stepped terrain."""
    for name, hf, model, _hfield, shapes, bodies in _rough_scenes(device, 20):
        with test.subTest(terrain=name):
            pipeline, _contacts, _state = _collide(model, shadow=True)
            test.assertEqual(_incomplete(hf, tu.shadow_rows(pipeline), shapes, bodies), [])


# ----------------------------------------------------------------------------------------------------------------
# float32 against float64
# ----------------------------------------------------------------------------------------------------------------


def _adversarial(rng, hf, count, kind):
    """Seam-adversarial placements: 15% on x-lines, 15% y-lines, 10% vertices, 10% diagonals, 50% generic."""
    bodies = []
    for _ in range(count):
        r = float(rng.choice([R22, R31]))
        dims = {"radius": r} if kind == "sphere" else {"radius": r, "half_height": float(rng.choice([0.04, 0.08]))}
        j = int(rng.integers(5, hf.cols - 5))
        i = int(rng.integers(5, hf.rows - 5))
        u = rng.uniform()
        fx, fy = rng.uniform(0.0, 1.0, size=2)
        if u < 0.15:
            fx = 0.0
        elif u < 0.30:
            fy = 0.0
        elif u < 0.40:
            fx, fy = 0.0, 0.0
        elif u < 0.50:
            fy = fx
        x, y = -hf.hx + (j + fx) * hf.dx, -hf.hy + (i + fy) * hf.dy
        q = np.array([0.0, 0.0, 0.0, 1.0])
        if kind == "capsule":
            if rng.uniform() < 0.5:
                q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)), LYING)
            else:
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
        lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
        top = max(
            hf.height(xx, yy)
            for xx in np.linspace(lo[0] - r, hi[0] + r, 7)
            for yy in np.linspace(lo[1] - r, hi[1] + r, 7)
        )
        z = top - lo[2] + r + rng.uniform(-0.003, 0.012)
        bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
    return bodies


def test_point_rows_match_the_float64_law(test, device):
    """Reproduce the float64 law's rows in float32 at seams up to 100 m out, but for decisions within the band.

    Spheres and capsules snapped onto grid lines, vertices and diagonals of a suite-like mosaic and of 200 m rough
    strips: every float64 row has a float32 row that matches it or carries its constraint (no pair loses a support),
    every float32 row matches or carries a float64 row but those whose deciding predicate lies within
    :data:`~newton.tests.terrain_law_test_utils.DECISION_BAND` of its threshold (at most 2% of the rows here, where
    half the placements sit exactly on seams; 0.2-0.5% over 600 placements per terrain), and every pair's deepest core
    distance agrees within 10 um. Carried rows (at most 5%; 1.9% on the mosaic over 600 placements) lie on the same
    plane at the same core distance: lying capsules' axis rows at the noisy flat tile's tiny creases, decided within
    the band, and an axis-edge row on an edge nearly parallel to the axis, whose closest point is ill-conditioned along
    it.
    """
    rng = np.random.default_rng(21)
    strip = np.round(rng.uniform(0.0, 0.1, size=(21, 2001)) / 0.005) * 0.005
    terrains = {
        "mosaic": (tu.suite_like_heights(rng, 64, 64), 3.2, 3.2, 60),
        "x-strip": (strip, 100.0, 1.0, 40),
        "y-strip": (strip.T.copy(), 1.0, 100.0, 40),
    }
    for name, (heights, hx, hy, count) in terrains.items():
        with test.subTest(terrain=name):
            hf = tu.reference_surface(tu.heightfield_model(heights, hx, hy, device="cpu"))
            bodies = _adversarial(rng, hf, count, "sphere") + _adversarial(rng, hf, count, "capsule")
            model, _hfield, shapes = tu.body_scene(heights, hx, hy, bodies, device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            rows = tu.shadow_rows(pipeline)
            totals = {"f64": 0, "covered": 0, "tolerated": 0}
            defects = []
            for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
                law = _law_rows(hf, kind, dims, pose)
                got = rows.get(shape, [])
                frame_tol = 1.0e-5 + 2.0e-7 * float(np.abs(pose[:2]).max())
                result = tu.compare_rows(hf, law, got, frame_tol=frame_tol)
                totals["f64"] += len(law)
                totals["covered"] += result.covered
                totals["tolerated"] += result.tolerated
                if result.missing or result.extra or result.deepest > 1.0e-5:
                    defects.append((kind, pose.tolist(), result))
            test.assertEqual(defects, [])
            test.assertGreater(totals["f64"], len(bodies))
            test.assertLessEqual(totals["covered"], 0.05 * totals["f64"], msg=totals)
            test.assertLessEqual(totals["tolerated"], 0.02 * totals["f64"], msg=totals)


# ----------------------------------------------------------------------------------------------------------------
# cores inside the solid (G2, G3)
# ----------------------------------------------------------------------------------------------------------------


def _surface_distance(hf, p, radius):
    """Exact distance from ``p`` to the triangulated surface within ``radius`` (float64), and the closest point."""
    best, closest = np.inf, None
    box = hf.cells_in_box(p[:2] - radius, p[:2] + radius)
    for t in hf.window_features(box)[0]:
        a, b, c = hf.tri_verts(t)
        q = ref._closest_on_tri(p, a, b, c)
        dist = float(np.linalg.norm(p - q))
        if dist < best:
            best, closest = dist, q
    return best, closest


def test_buried_cores_exit_by_the_minimum_translation(test, device):
    """Give every buried core point one exit, its exact minimum translation when that exit lies in its window (G2, G3).

    Sphere centres and lying capsules 3-60 mm inside rough, pitted and stepped terrain. A sphere gets one row; a buried
    sphere centre or capsule end gets exactly one row whose core point it is, an exit. When the exact distance from it
    to the surface is within the reach (its closest surface point then lies in its window), the exit's depth is that
    distance within 10 um, the minimum translation (G2); otherwise it is the face above it (G3). Every exit equals the
    float64 law's, and no buried core is left without a row.
    """
    rng = np.random.default_rng(12)
    checked = {"exact": 0, "law": 0}
    for name, heights in _rough_terrains(rng).items():
        with test.subTest(terrain=name):
            hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
            bodies = []
            for i in range(40):
                kind = "sphere" if i % 2 == 0 else "capsule"
                dims = {"radius": R31} if kind == "sphere" else {"radius": R31, "half_height": 0.08}
                x, y = rng.uniform(-1.2, 1.2, 2)
                depth = float(rng.choice([0.003, 0.01, 0.03, 0.06]))
                q = np.array([0.0, 0.0, 0.0, 1.0])
                if kind == "capsule":
                    q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)), LYING)
                z = hf.height(x, y) - depth
                bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
            model, _hfield, shapes = tu.body_scene(heights, 1.6, 1.6, bodies, device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            rows = tu.shadow_rows(pipeline)
            for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
                got = rows.get(shape, [])
                where = f"{kind} at {pose[:3].tolist()}"
                test.assertTrue(got, msg=f"{where} has no row")
                if kind == "sphere":
                    test.assertEqual(len(got), 1, msg=f"{where}: {got}")
                core = ref.make_core(kind, dims, pose)
                reach = dims["radius"] + REACH
                law = _law_rows(hf, kind, dims, pose)
                for point in (core.c,) if kind == "sphere" else (core.a, core.b):
                    if not ref._below_surface(hf, point):
                        continue
                    exits = [
                        row for row in got if row[2] < 0.0 and np.linalg.norm(row[0] + row[2] * row[1] - point) < 1e-5
                    ]
                    test.assertEqual(len(exits), 1, msg=f"{where}, core point {point.tolist()}: {got}")
                    law_exits = [row for row in law if row.d < 0.0 and np.linalg.norm(row.k - point) < 1e-9]
                    test.assertEqual(len(law_exits), 1, msg=f"{where}: {law}")
                    test.assertAlmostEqual(exits[0][2], law_exits[0].d, delta=1.0e-5, msg=where)
                    checked["law"] += 1
                    exact, _closest = _surface_distance(hf, point, reach)
                    if exact <= reach:
                        test.assertAlmostEqual(-exits[0][2], exact, delta=1.0e-5, msg=f"{where}: {point.tolist()}")
                        checked["exact"] += 1
    test.assertGreater(checked["exact"], 60, msg=checked)


# ----------------------------------------------------------------------------------------------------------------
# a capsule's axis inside the solid with both ends outside
# ----------------------------------------------------------------------------------------------------------------

RIDGE_HEIGHT = 0.2
"""Crest height [m] of :func:`_ridge_terrain`."""
RIDGE_SLOPE = 2.0
"""Slope of the ridge's faces (63.4 deg): its normals span +-atan(2) from vertical."""


def _ridge_terrain():
    """A flat 0.1 m grid with one raised vertex column: a ridge along y at x = 0, its faces one cell wide."""
    heights = np.zeros((21, 21))
    heights[:, 10] = RIDGE_HEIGHT
    return heights


def _ridge_translation(a, b):
    """The exact minimum translation of segment ``ab`` out of the ridge (its wedge of two half-spaces), and its
    direction.

    Along a direction ``u`` of the ridge's normal arc the wedge's support is the crest, so freeing the segment takes
    ``u . apex - min(u . a, u . b)``, a maximum of two sinusoids of the arc angle; outside the arc the wedge is
    unbounded. The minimum lies where the two are equal (``u`` normal to the segment, if in the arc) or at an arc end.
    """
    half = np.arctan(RIDGE_SLOPE)
    candidates = [np.array([np.sin(t), 0.0, np.cos(t)]) for t in (-half, half)]
    w = (b - a)[[0, 2]]
    u = np.array([-w[1], 0.0, w[0]]) / np.linalg.norm(w)
    u = -u if u[2] < 0.0 else u
    if abs(np.arctan2(u[0], u[2])) <= half:
        candidates.append(u)
    apex = np.array([0.0, 0.0, RIDGE_HEIGHT])
    lifts = [float(u @ apex - min(u @ a, u @ b)) for u in candidates]
    i = int(np.argmin(lifts))
    return lifts[i], candidates[i]


def _ridge_capsules():
    """Capsules across the ridge, their axis 2-50 mm below the crest and both ends outside the solid."""
    bodies = []
    for depth in (0.002, 0.005, 0.02, 0.05):
        for yaw, tilt in ((0.0, 0.0), (20.0, 0.0), (0.0, 4.0), (25.0, -3.0)):
            q = _quat_mul(
                _axis_angle((0.0, 0.0, 1.0), np.radians(yaw)),
                _quat_mul(_axis_angle((0.0, 1.0, 0.0), np.radians(tilt)), LYING),
            )
            pose = np.array([0.0, 0.05, RIDGE_HEIGHT - depth, *q], np.float32).astype(np.float64)
            bodies.append(("capsule", {"radius": 0.03, "half_height": 0.15}, pose))
    return bodies


def test_capsule_axis_exits_over_a_ridge(test, device):
    """Give a capsule whose axis passes below a ridge one exit, the segment's exact minimum translation.

    Both ends lie outside the solid, so neither has an exit (G2); the axis interior, inside, exits over the crease:
    one row with the normal from the axis to the crease and the distance of the two lines, which is the smallest
    lift over the ridge's normal arc that frees the segment (:func:`_ridge_translation`). A point exit of the deepest
    axis point through either face would push the capsule sideways by less than it needs.
    """
    heights = _ridge_terrain()
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.0, device="cpu"))
    bodies = _ridge_capsules()
    model, _hfield, shapes = tu.body_scene(heights, 1.0, 1.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        core = ref.make_core(kind, dims, pose)
        for end in (core.a, core.b):
            test.assertFalse(ref._below_surface(hf, end))
        translation, direction = _ridge_translation(core.a, core.b)
        got = rows.get(shape, [])
        with test.subTest(pose=np.round(pose[:3], 4).tolist()):
            test.assertEqual(len(got), 1, msg=got)
            _q, n, d, feature = got[0]
            test.assertEqual((feature >> 5) & 3, int(TERRAIN_FEATURE_EDGE))
            test.assertAlmostEqual(-d, translation, delta=1.0e-5)
            test.assertLess(tu.angle_between(n, direction), 1.0e-3)
            law = _law_rows(hf, kind, dims, pose)
            test.assertEqual(len(law), 1)
            test.assertAlmostEqual(law[0].d, -translation, delta=1.0e-6)


def test_capsule_axis_inside_without_exit_gets_recovery(test, device):
    """Give an axis inside the solid with no exit the face over its deepest crossing under a crease (G3).

    The axis descends 40 deg across a ridge of 63 deg faces, 10 mm below the crest where its footprint crosses it.
    The point of the axis closest to the crest lies outside the solid, above a face, so no exit lifts it over the
    crease; the axis interior gets the plane of the crest's first triangle at its crossing, its normal and depth.
    """
    heights = _ridge_terrain()
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.0, device="cpu"))
    depth = 0.01
    q = _axis_angle((0.0, 1.0, 0.0), np.radians(90.0 + 40.0))
    pose = np.array([0.0, 0.05, RIDGE_HEIGHT - depth, *q], np.float32).astype(np.float64)
    dims = {"radius": 0.02, "half_height": 0.13}
    model, _hfield, shapes = tu.body_scene(heights, 1.0, 1.0, [("capsule", dims, pose)], device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    got = tu.shadow_rows(pipeline).get(shapes[0], [])
    core = ref.make_core("capsule", dims, pose)
    for end in (core.a, core.b):
        test.assertFalse(ref._below_surface(hf, end))
    exits = [row for row in got if row[2] < 0.0]
    test.assertEqual(len(exits), 1, msg=got)
    _q, n, d, feature = exits[0]
    test.assertEqual((feature >> 5) & 3, int(TERRAIN_FEATURE_RECOVERY))
    # The crest's first triangle is the face on its -x side; the crossing lies `depth` below the crest.
    face = np.array([-RIDGE_SLOPE, 0.0, 1.0]) / np.sqrt(1.0 + RIDGE_SLOPE**2)
    crossing = core.a + (0.0 - core.a[0]) / (core.b[0] - core.a[0]) * (core.b - core.a)
    test.assertLess(tu.angle_between(n, face), 1.0e-3)
    test.assertAlmostEqual(d, float(face @ (crossing - np.array([0.0, 0.0, RIDGE_HEIGHT]))), delta=1.0e-5)
    law = [row for row in _law_rows(hf, "capsule", dims, pose) if row.d < 0.0]
    test.assertEqual(len(law), 1)
    test.assertAlmostEqual(law[0].d, d, delta=1.0e-6)


def test_point_rows_point_up_on_steep_terrain(test, device):
    """Give spheres and capsules on steep, spiky terrain only rows whose normal points up.

    Every outward normal of a heightfield lies in a cone spanned by its faces' upward normals, and an exit points
    from the core point to its closest surface point across a valley or a face, or (an axis) to a ridge's crease
    within its arc. An axis point inside the solid but beyond both faces of a ridge lies above the ridge, where an
    exit to its crease would point down and push the capsule in.
    """
    rng = np.random.default_rng(5)
    heights = np.round(rng.uniform(0.0, 0.25, (33, 33)) / 0.01) * 0.01
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
    bodies = []
    for i in range(600):
        kind = "sphere" if i % 2 == 0 else "capsule"
        dims = {"radius": 0.03} if kind == "sphere" else {"radius": 0.03, "half_height": 0.08}
        x, y = rng.uniform(-1.3, 1.3, 2)
        q = np.array([0.0, 0.0, 0.0, 1.0])
        if kind == "capsule":
            q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)), LYING)
        z = hf.height(x, y) + rng.uniform(-0.05, 0.08)
        bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
    model, _hfield, shapes = tu.body_scene(heights, 1.6, 1.6, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    down = [
        (kind, pose[:3].tolist(), row[1].tolist(), row[2])
        for shape, (kind, _dims, pose) in zip(shapes, bodies, strict=True)
        for row in rows.get(shape, [])
        if row[1][2] <= 0.0
    ]
    test.assertEqual(down, [])
    test.assertGreater(sum(len(v) for v in rows.values()), len(bodies))
    for kind, dims, pose in bodies[1::10]:
        for row in _law_rows(hf, kind, dims, pose):
            test.assertGreater(row.n[2], 0.0, msg=f"{kind} at {pose[:3].tolist()}: {row}")


# ----------------------------------------------------------------------------------------------------------------
# the tiled kernel
# ----------------------------------------------------------------------------------------------------------------


def _launch_segment(shadow, model, pipeline, kernel, lane, begin, count, out):
    """Run one walk kernel over table pairs ``[begin, begin + count)`` into ``out``, then the full-finish kernel."""
    inputs = TerrainPairInputs()
    inputs.pairs = shadow.table.pairs
    inputs.shape_type = model.shape_type
    inputs.shape_transform = pipeline.geom_transform
    inputs.shape_data = pipeline.geom_data
    inputs.shape_gap = model.shape_gap
    inputs.shape_heightfield_index = model.shape_heightfield_index
    inputs.shape_body = model.shape_body
    inputs.body_flags = model.body_flags
    inputs.include_static_kinematic_pairs = 1
    args = [begin, 0, inputs, shadow.features.data(), out]
    out.deferred_count.zero_()
    if lane == TERRAIN_LANE_THREAD:
        wp.launch(kernel, dim=count, inputs=args, device=model.device)
    else:
        wp.launch_tiled(kernel, dim=count, inputs=args, block_dim=16, device=model.device)
    wp.launch(terrain_law_point.terrain_point_rows_full, dim=count, inputs=args, device=model.device)


def _launch_full(shadow, model, pipeline, begin, count, out):
    """Run the full-finish kernel over the pairs of ``out``'s deferred list (``deferred_count``) from table pair
    ``begin``."""
    inputs = TerrainPairInputs()
    inputs.pairs = shadow.table.pairs
    inputs.shape_type = model.shape_type
    inputs.shape_transform = pipeline.geom_transform
    inputs.shape_data = pipeline.geom_data
    inputs.shape_gap = model.shape_gap
    inputs.shape_heightfield_index = model.shape_heightfield_index
    inputs.shape_body = model.shape_body
    inputs.body_flags = model.body_flags
    inputs.include_static_kinematic_pairs = 1
    args = [begin, 0, inputs, shadow.features.data(), out]
    wp.launch(terrain_law_point.terrain_point_rows_full, dim=count, inputs=args, device=model.device)


def _side_buffer(begin, count, pair_count, device):
    """A side buffer for ``count`` pairs from table pair ``begin`` (slots from 0), deferred list from ``begin``."""
    out = TerrainLawRows()
    slots = count * terrain_law_point.TERRAIN_POINT_ROWS
    out.q = wp.zeros(slots, dtype=wp.vec3, device=device)
    out.n = wp.zeros(slots, dtype=wp.vec3, device=device)
    out.d = wp.zeros(slots, dtype=float, device=device)
    out.feature = wp.zeros(slots, dtype=wp.int32, device=device)
    out.count = wp.zeros(pair_count, dtype=wp.int32, device=device)
    out.state = wp.zeros(pair_count, dtype=wp.int32, device=device)
    out.status = wp.zeros(1, dtype=wp.int32, device=device)
    out.overflow = wp.zeros(2, dtype=wp.int32, device=device)
    out.deferred = wp.zeros(pair_count, dtype=wp.int32, device=device)
    out.deferred_first = wp.full(4, begin, dtype=wp.int32, device=device)
    out.deferred_count = wp.zeros(4, dtype=wp.int32, device=device)
    return out


MARKED_BIT_31 = (
    (0.8063436150550842, -0.8646434545516968, 0.34505870938301086),
    (-0.3956831395626068, 0.1805560439825058, 0.33991286158561707),
    (-0.7925142645835876, -0.43028581142425537, 0.3399025499820709),
)
"""Centres of 0.25 m spheres on :func:`_tile_scenes`' rough tile whose rows include one owned by the window vertex of
index 31: bit 31, the sign bit, of the tiled kernel's first mask word."""


def _tile_scenes(rng):
    """Scenes whose pairs all take the tiled kernel (windows over 16 cells): ``(name, heights, half extent, bodies)``.

    Large spheres and long capsules rest on rough terrain; long capsules lie across ridges of one height, whose ridge
    rows share a plane, so that the finish may drop some of them (the full kernel's pairs); spheres whose rows mark
    bit 31 of a mask word (:data:`MARKED_BIT_31`); a sphere and a capsule whose windows hold over 256 vertices, which
    the tiled kernel's first lane walks unmasked.
    """
    heights = _rough_terrains(rng)["rough"]
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
    bodies = []
    for i in range(24):
        kind = "sphere" if i % 2 == 0 else "capsule"
        dims = {"radius": 0.25} if kind == "sphere" else {"radius": 0.06, "half_height": 0.3}
        x, y = rng.uniform(-1.0, 1.0, 2)
        q = np.array([0.0, 0.0, 0.0, 1.0])
        if kind == "capsule":
            q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)), LYING)
        lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
        r = dims["radius"]
        top = max(
            hf.height(xx, yy)
            for xx in np.linspace(lo[0] - r, hi[0] + r, 9)
            for yy in np.linspace(lo[1] - r, hi[1] + r, 9)
        )
        z = top - lo[2] + r + rng.uniform(-0.004, 0.01)
        bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
    ridges = np.zeros((41, 41))
    ridges[:, 1::2] = 0.03
    lying = []
    for _ in range(8):
        dims = {"radius": 0.03, "half_height": 0.6}
        q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(-0.4, 0.4)), LYING)
        x, y = rng.uniform(-0.6, 0.6, 2)
        z = 0.03 + dims["radius"] + rng.uniform(0.0, 0.005)
        lying.append(("capsule", dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
    marked = [("sphere", {"radius": 0.25}, np.array([*c, 0.0, 0.0, 0.0, 1.0])) for c in MARKED_BIT_31]
    diagonal = _quat_mul(_axis_angle((0.0, 0.0, 1.0), np.pi / 4.0), LYING)
    wide = [
        ("sphere", {"radius": 0.8}, np.array([0.0, 0.0, hf.height(0.0, 0.0) + 0.81, 0.0, 0.0, 0.0, 1.0])),
        ("capsule", {"radius": 0.05}, np.array([0.1, -0.1, 0.15, *diagonal])),
    ]
    wide[1][1]["half_height"] = 1.0
    return [
        ("rough", heights, 1.6, bodies),
        ("ridges", ridges, 2.0, lying),
        ("marked bit 31", heights, 1.6, marked),
        ("unmasked", heights, 1.6, wide),
    ]


def _owner_vertex_indices(hf, rows, bodies, shapes):
    """Per body, the window index ``(r - r0) * (c1 - c0 + 2) + (c - c0)`` of the vertex owning each of its walked
    rows (recovery rows, added after the walk, excluded), and its window's vertex count."""
    out = []
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        core = ref.make_core(kind, dims, pose)
        r0, r1, c0, c1 = tu.window_cells(hf, core, dims["radius"] + REACH)
        width = c1 - c0 + 2
        owners = []
        for _q, _n, _d, feature in rows.get(shape, []):
            if (feature >> 5) & 3 == int(TERRAIN_FEATURE_RECOVERY):
                continue
            r = r0 + (((feature >> 13) & 63) - r0) % 64
            c = c0 + (((feature >> 7) & 63) - c0) % 64
            owners.append((r - r0) * width + (c - c0))
        out.append((owners, (r1 - r0 + 2) * width))
    return out


def test_tiled_kernel_matches_one_thread_kernel(test, device):
    """Give a pair the same rows, bit for bit, whether one thread or a tile of lanes evaluates its window.

    Both kernels evaluate the same table segment of tile-lane pairs (:func:`_tile_scenes`), the full kernel after
    them; some pairs go through the full kernel. On the CPU Warp runs ``wp.launch_tiled`` with one lane per pair
    (``block_dim`` 1), so the CPU run checks one lane's walk, its masked and unmasked paths and bit 31 of a mask word;
    the 16-lane path (the window extent's ``tile_max``, the union of the lanes' mask words by ``tile_sum`` and the
    lanes' striding) runs on CUDA only, in this test's CUDA variant.
    """
    deferred = 0
    for name, heights, extent, bodies in _tile_scenes(np.random.default_rng(4)):
        with test.subTest(scene=name):
            model, _hfield, shapes = tu.body_scene(heights, extent, extent, bodies, device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            shadow = pipeline._terrain_law_shadow
            begin, end = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_TILE)
            test.assertEqual(end - begin, len(bodies))
            outputs = []
            for kernel, lane in (
                (terrain_law_point.terrain_point_rows, TERRAIN_LANE_THREAD),
                (terrain_law_point.terrain_point_rows_tiled, TERRAIN_LANE_TILE),
            ):
                out = _side_buffer(begin, end - begin, shadow.table.pair_count, device)
                _launch_segment(shadow, model, pipeline, kernel, lane, begin, end - begin, out)
                outputs.append(out)
            one, tiled = outputs
            np.testing.assert_array_equal(one.count.numpy(), tiled.count.numpy())
            np.testing.assert_array_equal(one.state.numpy(), tiled.state.numpy())
            test.assertGreaterEqual(int(one.count.numpy().sum()), len(bodies))
            test.assertEqual(int(one.deferred_count.numpy()[0]), int(tiled.deferred_count.numpy()[0]))
            deferred += int(one.deferred_count.numpy()[0])
            for array in ("q", "n", "d", "feature"):
                np.testing.assert_array_equal(getattr(one, array).numpy(), getattr(tiled, array).numpy(), err_msg=array)
            hf = tu.reference_surface(tu.heightfield_model(heights, extent, extent, device="cpu"))
            owners = _owner_vertex_indices(hf, tu.shadow_rows(pipeline), bodies, shapes)
            if name == "marked bit 31":
                # The scene covers what it is for: a row of every body owned by window vertex 31.
                test.assertTrue(all(31 in rows for rows, _vertices in owners), msg=owners)
            elif name == "unmasked":
                test.assertTrue(all(vertices > 256 and rows for rows, vertices in owners), msg=owners)
    # Both paths finished pairs in the full kernel too.
    test.assertGreater(deferred, 0)


# ----------------------------------------------------------------------------------------------------------------
# the kernels' rejects against the plain walk
# ----------------------------------------------------------------------------------------------------------------

# The plain walk compiles here, with the point module's functions it calls: without contraction, as that module
# compiles, so that it rounds as the kernels do on CUDA too.
wp.set_module_options({"fuse_fp": False})

_Claim = terrain_law_point._Claim
_PLAIN_ROWS = wp.constant(terrain_law_point.TERRAIN_POINT_ROWS)
_PLAIN_HELD = wp.constant(int(terrain_law_point._HELD))


@wp.func
def _plain_end_face(
    data: TerrainFeatureData,
    window: TerrainWindow,
    core: terrain_law_point._PointCore,
    end: int,
    k: wp.vec3,
    below: int,
    r: int,
    c: int,
    sub: int,
    n: wp.vec3,
    a: wp.vec3,
) -> _Claim:
    """``terrain_law_point._end_face`` without its off-cell reject."""
    d = wp.dot(n, k - a)
    if d > core.reach:
        return _Claim()
    if below != 0:
        if d > core.eps:
            return _Claim()
    elif d < -core.eps:
        return _Claim()
    q = k - d * n
    s = terrain_face_values(data, window, r, c, sub, n, k)
    if not (terrain_claims(s[0], core.eps) and terrain_claims(s[1], core.eps) and terrain_claims(s[2], core.eps)):
        return _Claim()
    feature = terrain_feature_id(r, c, TERRAIN_FEATURE_FACE, sub, end)
    return terrain_law_point._offer(end, below, q, k, n, d, core.reach, feature)


@wp.func
def _plain_point_item(
    data: TerrainFeatureData,
    window: TerrainWindow,
    planes: TerrainPlanes,
    core: terrain_law_point._PointCore,
    r: int,
    c: int,
    r1: int,
    c1: int,
    item: int,
):
    """``terrain_law_point._point_item`` with :func:`_plain_end_face`."""
    end = item // 6
    feature = item - 6 * end
    if end == 2:
        feature = item - 10
    k = core.a
    below = core.below_a
    if end == 1:
        k = core.b
        below = core.below_b
    inside = r <= r1 and c <= c1
    if feature < 2:
        if not inside:
            return planes, _Claim()
        planes, n, a = terrain_plane(data, window, planes, r, c, feature)
        return planes, _plain_end_face(data, window, core, end, k, below, r, c, feature, n, a)
    if feature == 5:
        if end == 2:
            return terrain_law_point._axis_vertex(data, window, planes, core, r, c)
        return planes, terrain_law_point._end_vertex(data, window, core, end, k, below, r, c)
    which = feature - 2
    if (which == 0 and c > c1) or (which == 1 and r > r1) or (which == 2 and not inside):
        return planes, _Claim()
    if terrain_edge_kind(data, window.surface, r, c, which) >= 0:
        return planes, _Claim()
    v0i, v1i, _ta, _tb, _wa, _wb = terrain_edge_sides(r, c, which)
    v0 = terrain_window_vertex(data, window, v0i[0], v0i[1])
    v1 = terrain_window_vertex(data, window, v1i[0], v1i[1])
    if end == 2:
        return terrain_law_point._axis_edge(data, window, planes, core, r, c, which, v0, v1)
    return terrain_law_point._end_edge(data, window, planes, core, end, k, below, r, c, which, v0, v1)


@wp.kernel(enable_backward=False)
def _plain_point_rows(
    begin: int, slot_begin: int, inputs: TerrainPairInputs, data: TerrainFeatureData, out: TerrainLawRows
):
    """``terrain_point_rows`` without its rejects: every item of every window vertex in window order (no vertex
    drop, no vertex mask, no off-cell face reject), then the same close."""
    i = wp.tid()
    pair = begin + i
    slot = slot_begin + i * _PLAIN_ROWS
    state, frame, heightfield, cells, a, b, segment, reach = terrain_law_point._pair_setup(inputs, data, pair)
    if state != 0:
        terrain_pair_exit(out, pair, state)
        return
    r1 = cells[1]
    c1 = cells[3]
    window = terrain_window(data, heightfield, cells[0], cells[2])
    eps = terrain_eps(terrain_window_extent(data, window, r1, c1, terrain_law_point._core_extent(window, a, b)))
    planes = terrain_planes()
    planes, core = terrain_law_point._pair_core(data, window, planes, a, b, segment, reach, eps)
    terrain_law_point._open_pair(out, slot)
    held = int(0)
    candidates = int(0)
    items = terrain_law_point._vertex_items(core)
    for r in range(cells[0], r1 + 2):
        for c in range(cells[2], c1 + 2):
            if held <= _PLAIN_HELD:
                for item in range(items):
                    planes, claim = _plain_point_item(data, window, planes, core, r, c, r1, c1, item)
                    held, candidates = terrain_law_point._walk_claim(out, slot, held, candidates, claim, eps)
    terrain_law_point._close_pair(
        out, pair, slot, terrain_law_point._wide(cells), held, candidates, data, window, planes, core, frame
    )


def _reject_scenes(device):
    """Scenes for :func:`test_rejects_keep_the_plain_walk_rows`: ``(name, model, bodies)``."""
    for name, _hf, model, _hfield, _shapes, bodies in _rough_scenes(device, 20):
        yield name, model, bodies
    rng = np.random.default_rng(23)
    mosaic = tu.suite_like_heights(rng, 64, 64)
    hf = tu.reference_surface(tu.heightfield_model(mosaic, 3.2, 3.2, device="cpu"))
    bodies = _adversarial(rng, hf, 60, "sphere") + _adversarial(rng, hf, 60, "capsule")
    yield "seams", tu.body_scene(mosaic, 3.2, 3.2, bodies, device=device)[0], bodies
    rough = _rough_terrains(rng)["rough"]
    hf = tu.reference_surface(tu.heightfield_model(rough, 1.6, 1.6, device="cpu"))
    bodies = []
    for i in range(40):
        kind = "sphere" if i % 2 == 0 else "capsule"
        dims = {"radius": R31} if kind == "sphere" else {"radius": R31, "half_height": 0.08}
        x, y = rng.uniform(-1.2, 1.2, 2)
        q = np.array([0.0, 0.0, 0.0, 1.0])
        if kind == "capsule":
            q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)), LYING)
        z = hf.height(x, y) - float(rng.choice([0.003, 0.01, 0.03, 0.06]))
        bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
    yield "buried", tu.body_scene(rough, 1.6, 1.6, bodies, device=device)[0], bodies
    bodies = _ridge_capsules()
    yield "ridge", tu.body_scene(_ridge_terrain(), 1.0, 1.0, bodies, device=device)[0], bodies
    # Rows at the admission limit: the lowest point 28-40 mm (the reach beyond the radius) above the terrain.
    bodies = []
    for i in range(80):
        kind = "sphere" if i % 2 == 0 else "capsule"
        r = R31 if i % 4 < 2 else R22
        dims = {"radius": r} if kind == "sphere" else {"radius": r, "half_height": 0.08}
        x, y = rng.uniform(-1.2, 1.2, 2)
        q = np.array([0.0, 0.0, 0.0, 1.0])
        if kind == "capsule":
            tilt = _axis_angle((1.0, 0.0, 0.0), rng.uniform(0.0, 0.4))
            q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)), _quat_mul(tilt, LYING))
        lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
        top = max(
            hf.height(xx, yy)
            for xx in np.linspace(lo[0] - r, hi[0] + r, 7)
            for yy in np.linspace(lo[1] - r, hi[1] + r, 7)
        )
        z = top - lo[2] + r + rng.uniform(0.028, REACH)
        bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
    yield "reach", tu.body_scene(rough, 1.6, 1.6, bodies, device=device)[0], bodies


def test_rejects_keep_the_plain_walk_rows(test, device):
    """Give every pair, bit for bit, the rows of the walk over every window vertex without the kernels' rejects.

    The one-thread kernel drops the vertices whose features lie out of reach (``_vertex_dropped``), walks a bit mask of
    the rest, and refuses a face whose projection lies off its cell (``_off_cell``) before its canonical values; the
    plain walk (:func:`_plain_point_rows`) evaluates every item of every window vertex in window order and closes the
    pair the same way. On rough, pitted, stepped, seam-adversarial, buried and ridge scenes the count, state, q, n, d
    and feature of every pair agree bit for bit; so do those of the full-finish kernel (which walks with the drop and
    runs the finish) run on every walked pair, which also checks that the walks write a pair's rows as the finish keeps
    them.
    """
    walked = 0
    for name, model, bodies in _reject_scenes(device):
        with test.subTest(scene=name):
            pipeline, _contacts, _state = _collide(model, shadow=True)
            shadow = pipeline._terrain_law_shadow
            begin = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_THREAD)[0]
            end = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_TILE)[1]
            count = end - begin
            test.assertEqual(count, len(bodies))
            results = {}
            for label, kernel in (("kernel", terrain_law_point.terrain_point_rows), ("plain", _plain_point_rows)):
                out = _side_buffer(begin, count, shadow.table.pair_count, device)
                _launch_segment(shadow, model, pipeline, kernel, TERRAIN_LANE_THREAD, begin, count, out)
                results[label] = out
            # The full-finish kernel on every pair.
            full = _side_buffer(begin, count, shadow.table.pair_count, device)
            full.deferred.assign(np.arange(shadow.table.pair_count, dtype=np.int32))
            full.deferred_count.assign(np.full(4, count, np.int32))
            _launch_full(shadow, model, pipeline, begin, count, full)
            kernel = {f: getattr(results["kernel"], f).numpy() for f in ("count", "state", "q", "n", "d", "feature")}
            plain = {f: getattr(results["plain"], f).numpy() for f in ("count", "state", "q", "n", "d", "feature")}
            finished = {f: getattr(full, f).numpy() for f in ("count", "state", "q", "n", "d", "feature")}
            np.testing.assert_array_equal(kernel["count"], plain["count"])
            np.testing.assert_array_equal(kernel["state"], plain["state"])
            for i in range(count):
                pair = begin + i
                rows = slice(
                    i * terrain_law_point.TERRAIN_POINT_ROWS,
                    i * terrain_law_point.TERRAIN_POINT_ROWS + int(kernel["count"][pair]),
                )
                for f in ("q", "n", "d", "feature"):
                    np.testing.assert_array_equal(kernel[f][rows], plain[f][rows], err_msg=f"{name} pair {i} {f}")
                if kernel["state"][pair] & EXITS:
                    continue
                walked += 1
                test.assertEqual(kernel["count"][pair], finished["count"][pair], msg=f"{name} pair {i}")
                test.assertEqual(kernel["state"][pair], finished["state"][pair], msg=f"{name} pair {i}")
                for f in ("q", "n", "d", "feature"):
                    np.testing.assert_array_equal(kernel[f][rows], finished[f][rows], err_msg=f"{name} pair {i} {f}")
    test.assertGreater(walked, 200)


class TestTerrainLawPoint(unittest.TestCase):
    # Today's terrain route is held to the shadow rows' checks and fails them (the defects the law removes). Each
    # expected failure turns into an unexpected success once the law writes these contacts (Tracks 2 and 7): remove
    # them then.

    @unittest.expectedFailure
    def test_today_route_is_minimal_at_flat_seams_cpu(self):
        """Give today's spheres and capsules at flat seams the plane's rows only, as the shadow rows have them.

        Today's rows are compared on terrain point and normal (:func:`terrain_law_test_utils.contact_rows`). When this
        test was written today gave 2.58 rows and 0.92 ghost normals per sphere, 3.92 rows and 1.08 ghosts per
        lying capsule, and lacked 13 of the 24 capsule end rows.
        """
        bodies, expected = _flat_seams()
        model, hfield, shapes = tu.body_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device="cpu")
        _pipeline, contacts, state = _collide(model, shadow=False)
        rows = tu.contact_rows(model, state, contacts, hfield)
        _assert_minimal(self, _minimal_violations(rows, shapes, expected, d_tol=None))

    @unittest.expectedFailure
    def test_today_route_is_complete_cpu(self):
        """Keep every float64-law support on rough, pitted and stepped terrain in today's contacts, as the shadow does.

        When this test was written today missed supports on every terrain: 9 of the 20 spheres and 15 of the 20
        capsules on the rough tile lost one, 1 and 15 on the pits, 0 and 9 on the stairs.
        """
        for _name, hf, terrain, hfield, shapes, bodies in _rough_scenes("cpu", 20):
            _pipeline, contacts, state = _collide(terrain, shadow=False)
            self.assertEqual(_incomplete(hf, tu.contact_rows(terrain, state, contacts, hfield), shapes, bodies), [])

    @unittest.expectedFailure
    def test_point_rows_are_minimal_at_noisy_flat_seams_cpu(self):
        """Give spheres and lying capsules at the seams of a noisy flat tile the plane's rows only (a known failure).

        The tile carries +-1.4 um of height noise, as the recorded suite terrain's flat border: the law's plane
        clusters break it into near-coplanar pieces, and a body over a seam gets near-duplicate rows (normals 18-34 urad
        and terrain points 0.8-1.7 um apart, beyond the finish's merge of 4 bands and its 1e-5 plane groups). When this
        test was written the shadow gave these bodies 1.17 rows per sphere and 2.67 per capsule (the plane's 1 and 2).
        A merge distance of about the cluster tolerance would remove them; their rows are complete and without ghosts
        (:func:`test_point_rows_have_the_plane_at_noisy_flat_seams`).
        """
        heights, bodies, expected = _noisy_flat_seams()
        model, _hfield, shapes = tu.body_scene(heights, 2.0, 2.0, bodies, device="cpu")
        pipeline, _contacts, _state = _collide(model, shadow=True)
        self.assertEqual(_minimal_violations(tu.shadow_rows(pipeline), shapes, expected)["count"], [])

    @unittest.expectedFailure
    def test_law_rows_are_minimal_at_noisy_flat_seams_cpu(self):
        """Give the float64 law's rows at the noisy flat tile's seams the plane's rows only (a known failure).

        The same near-duplicates as the shadow rows' (:meth:`test_point_rows_are_minimal_at_noisy_flat_seams_cpu`), in
        the law itself: 1.08 rows per sphere and 2.63 per capsule when this test was written.
        """
        heights, bodies, expected = _noisy_flat_seams()
        hf = tu.reference_surface(tu.heightfield_model(heights, 2.0, 2.0, device="cpu"))
        rows = {i: [(r.q, r.n, r.d) for r in _law_rows(hf, *body)] for i, body in enumerate(bodies)}
        self.assertEqual(_minimal_violations(rows, range(len(bodies)), expected)["count"], [])

    def test_comparison_flags_lost_rows_cpu(self):
        """Count a lost support as missed in the float32 against float64 comparison, at seams too.

        A float64 row is tolerated as unmatched only when a float32 row carries its constraint; a pair whose float32
        rows are gone, or lack one row, fails however close to a seam its supports lie.
        """
        rng = np.random.default_rng(21)
        heights = tu.suite_like_heights(rng, 64, 64)
        hf = tu.reference_surface(tu.heightfield_model(heights, 3.2, 3.2, device="cpu"))
        pairs = 0
        for fx, fy in ((0.0, 0.0), (0.0, 0.37), (0.41, 0.0), (0.3, 0.3), (0.37, 0.61)):
            for _ in range(40):
                j, i = int(rng.integers(5, hf.cols - 5)), int(rng.integers(5, hf.rows - 5))
                x, y = -hf.hx + (j + fx) * hf.dx, -hf.hy + (i + fy) * hf.dy
                z = hf.height(x, y) + R31 + rng.uniform(-0.003, 0.012)
                pose = np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], np.float32).astype(np.float64)
                law = _law_rows(hf, "sphere", {"radius": R31}, pose)
                if not law:
                    continue
                pairs += 1
                lost = tu.compare_rows(hf, law, [])
                self.assertEqual(len(lost.missing), len(law))
                self.assertEqual(lost.deepest, float("inf"))
                got = [(row.q, row.n, row.d) for row in law]
                self.assertEqual(tu.compare_rows(hf, law, got).missing, [])
                for drop in range(len(got)):
                    others = got[:drop] + got[drop + 1 :]
                    if not tu.carries_half_space(law[drop], others):
                        self.assertEqual(len(tu.compare_rows(hf, law, others).missing), 1)
        self.assertGreater(pairs, 150)

    def test_point_kernels_compile_without_contraction_cpu(self):
        """Compile the point-core kernels to PTX without fused multiply-adds (NVRTC, any host).

        Every float32 operation then rounds as on the CPU, so the CPU decides every boundary as the GPU does.
        """
        if 86 not in wp.get_cuda_supported_archs():
            self.skipTest("NVRTC cannot compile for sm_86 here")
        with tempfile.TemporaryDirectory() as directory:
            wp.compile_aot_module(terrain_law_point, arch=86, module_dir=directory, use_ptx=True)
            paths = glob.glob(os.path.join(directory, "*.ptx"))
            self.assertEqual(len(paths), 1)
            with open(paths[0], encoding="latin-1") as f:
                ptx = f.read()
        self.assertEqual(ptx.count("fma.rn"), 0)
        self.assertGreater(ptx.count("mul.rn.f32"), 100)


devices = get_test_devices()
for _name, _func in (
    ("test_point_rows_are_minimal_at_flat_seams", test_point_rows_are_minimal_at_flat_seams),
    ("test_point_rows_are_complete", test_point_rows_are_complete),
    ("test_point_rows_match_the_float64_law", test_point_rows_match_the_float64_law),
    ("test_buried_cores_exit_by_the_minimum_translation", test_buried_cores_exit_by_the_minimum_translation),
    ("test_point_rows_have_the_plane_at_noisy_flat_seams", test_point_rows_have_the_plane_at_noisy_flat_seams),
    ("test_capsule_axis_exits_over_a_ridge", test_capsule_axis_exits_over_a_ridge),
    ("test_capsule_axis_inside_without_exit_gets_recovery", test_capsule_axis_inside_without_exit_gets_recovery),
    ("test_point_rows_point_up_on_steep_terrain", test_point_rows_point_up_on_steep_terrain),
    ("test_tiled_kernel_matches_one_thread_kernel", test_tiled_kernel_matches_one_thread_kernel),
    ("test_rejects_keep_the_plain_walk_rows", test_rejects_keep_the_plain_walk_rows),
):
    add_function_test(TestTerrainLawPoint, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

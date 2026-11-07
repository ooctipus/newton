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

import functools
import glob
import os
import re
import tempfile
import unittest

import numpy as np
import warp as wp

from newton._src.geometry import terrain_law_point
from newton._src.geometry.terrain_contacts import TERRAIN_LANE_THREAD, TERRAIN_LANE_TILE, TERRAIN_PARTNER_POINT
from newton._src.geometry.terrain_law import (
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_FINISH_CANDIDATE_OVERFLOW,
    TERRAIN_FINISH_ROW_OVERFLOW,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TerrainFeatureData,
    TerrainFinishScratch,
    TerrainLawRows,
    TerrainPairInputs,
    TerrainPairWork,
    TerrainPlanes,
    TerrainWindow,
    terrain_claims,
    terrain_eps,
    terrain_face_values,
    terrain_feature_id,
    terrain_finish_list,
    terrain_finish_overflow,
    terrain_finish_scratch,
    terrain_finish_state,
    terrain_law_block_dim,
    terrain_law_rows,
    terrain_pair_exit,
    terrain_pair_work,
    terrain_plane,
    terrain_planes,
    terrain_report,
    terrain_spill_rows,
    terrain_store_row,
    terrain_window,
    terrain_window_extent,
    terrain_window_point,
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
    """Collide once with the point class in shadow, or on today's terrain route alone (``NEWTON_TERRAIN_LAW=none``)."""
    pipeline = tu.shadow_pipeline(model, **PIPELINE) if shadow else tu.terrain_route_pipeline(model, **PIPELINE)
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
    half the placements sit exactly on seams; 0.2-0.9% at 600 placements per terrain), and every pair's deepest core
    distance agrees within 10 um. Carried rows (at most 5%; up to 2.1% on the mosaic at 600 placements) lie on the
    same plane at the same core distance: lying capsules' axis rows at the noisy flat tile's tiny creases, decided
    within the band, and an axis-edge row on an edge nearly parallel to the axis, whose closest point is ill-conditioned
    along it.
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
# the border
# ----------------------------------------------------------------------------------------------------------------


def test_point_rows_match_the_float64_law_at_the_border(test, device):
    """Reproduce the float64 law's rows at the heightfield's border in float32 for spheres and capsules.

    Spheres and capsules in any orientation within 8 cm of a wall's plane or a corner of flat, rough and sloped tiles,
    their lowest point up to 3 cm into the top: the walls, border edges, border vertices and corner lines give the
    float64 law's rows (none missed or extra, none carried or decided within the band here), every pair's deepest row
    within 10 um, every penetrating row's core point inside the block.
    """
    rng = np.random.default_rng(31)
    kinds = [("sphere", {"radius": 0.03}), ("capsule", {"radius": 0.02, "half_height": 0.06})]
    for name, heights in tu.border_terrains(rng).items():
        with test.subTest(terrain=name):
            hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.0, device="cpu"))
            bodies = tu.border_bodies(rng, hf, kinds, 20)
            model, _hfield, shapes = tu.body_scene(heights, 1.0, 1.0, bodies, device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            rows = tu.shadow_rows(pipeline)
            total, defects = 0, []
            for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
                law = _law_rows(hf, kind, dims, pose)
                got = rows.get(shape, [])
                result = tu.compare_rows(hf, law, got)
                total += len(law)
                if result.missing or result.extra or result.covered or result.tolerated or result.deepest > 1.0e-5:
                    defects.append((kind, pose.tolist(), result))
                if tu.outside_face_rows(hf, got):
                    defects.append((kind, pose.tolist(), "outside", tu.outside_face_rows(hf, got)))
            test.assertEqual(defects, [])
            test.assertGreater(total, len(bodies))


def _block_row(centre, reach):
    """The one row of a point core at ``centre`` about the block ``[-1, 1] x [-1, 1] x [-1, 0]`` (a flat unit
    heightfield and its walls) within ``reach``, or None: inside the footprint the top face's (a point below it exits
    through the top), outside it the block's closest point, its wall's above the walls' base."""
    x, y, z = centre
    if abs(x) <= 1.0 and abs(y) <= 1.0:
        return (np.array([x, y, 0.0]), np.array([0.0, 0.0, 1.0]), z) if z <= reach else None
    if z < -1.0:
        return None
    q = np.array([np.clip(x, -1.0, 1.0), np.clip(y, -1.0, 1.0), min(z, 0.0)])
    d = float(np.linalg.norm(centre - q))
    return (q, (centre - q) / d, d) if d <= reach else None


def test_point_rows_take_the_block_at_the_border(test, device):
    """Give a sphere about a corner of a flat heightfield's block its one exact row: the block's closest point.

    The heightfield is the solid block ``[-1, 1] x [-1, 1] x [-1, 0]``: its top face, its walls from the border down
    1 m, the border edges and vertices between them and the corner lines where two walls meet. Sphere centres about
    every corner, 3-3.5 cm in or out across each wall: over the top (the top face), beyond one wall above the top (a
    border edge) or beside it below the top (the wall, also 2 cm above its base), beyond both walls above the top (the
    corner vertex) or below it (the corner line), and inside the footprint below the top (which exits through the
    top). Each gets one row, the block's closest point with its normal toward the centre and its distance; centres
    beside the block below the walls' base get none.
    """
    radius = 0.02
    reach = radius + REACH
    bodies, expected = [], []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for u in (0.97, 1.03):
                for w in (0.965, 1.035):
                    for z in (0.025, -0.3, -0.98, -1.05):
                        centre = np.array([sx * u, sy * w, z])
                        if u < 1.0 and w < 1.0 and z < -0.5:
                            continue
                        row = _block_row(centre, reach)
                        bodies.append(("sphere", {"radius": radius}, np.r_[centre, 0.0, 0.0, 0.0, 1.0]))
                        expected.append([] if row is None else [row])
    model, _hfield, shapes = tu.body_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    defects = []
    for shape, (_kind, _dims, pose), want in zip(shapes, bodies, expected, strict=True):
        got = rows.get(shape, [])
        missing = tu.missing_rows(want, got, q_tol=1.0e-5, angle_tol=1.0e-5, d_tol=1.0e-5)
        if len(got) != len(want) or missing:
            defects.append((pose[:3].tolist(), want, got))
    test.assertEqual(defects, [])
    test.assertEqual(sum(len(w) for w in expected), len(bodies) - 12)


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


def _recovery_capsules():
    """Capsules descending 30-50 deg across the ridge, their axis 5-20 mm below the crest where its footprint crosses
    it and both ends outside the solid: no axis exit, the axis interior's G3 row instead
    (:func:`test_capsule_axis_inside_without_exit_gets_recovery`)."""
    bodies = []
    for tilt in (30.0, 40.0, 50.0):
        for depth in (0.005, 0.02):
            for yaw in (0.0, -20.0):
                q = _quat_mul(
                    _axis_angle((0.0, 0.0, 1.0), np.radians(yaw)), _axis_angle((0.0, 1.0, 0.0), np.radians(90.0 + tilt))
                )
                pose = np.array([0.0, 0.05, RIDGE_HEIGHT - depth, *q], np.float32).astype(np.float64)
                bodies.append(("capsule", {"radius": 0.02, "half_height": 0.13}, pose))
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


def _pair_inputs(shadow, model, pipeline):
    """The shadow's per-collide pair inputs, from the pipeline's current shape transforms."""
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
    return inputs


def _launch_segment(shadow, model, pipeline, lane, begin, count, out, *, kernel=None, blocks=None):
    """Walk table pairs ``[begin, begin + count)`` into ``out``, then finish the pairs the walk deferred.

    With ``lane`` the one-thread class, the one-thread kernel (or ``kernel``, of its signature) walks every pair; with
    the tiled class, it queues them past the whole-pair rejects and the tiled kernel takes the queue, with ``blocks``
    blocks (the work list's own when ``None``). The class's finish of deferred pairs finishes the deferred pairs; after
    the plain walk (:func:`_plain_point_rows`) the plain finish does (:func:`_plain_point_rows_finish`), so that the
    plain side's deferred pairs are finished independently of the kernels' marks, scratch and grid too.
    """
    device = model.device
    args = [begin, 0, _pair_inputs(shadow, model, pipeline), shadow.features.data(), out]
    out.deferred_count.zero_()
    block = terrain_law_block_dim(device)
    if lane == TERRAIN_LANE_THREAD:
        work = terrain_pair_work(device, begin + count, 0)
        kernel = kernel or terrain_law_point.terrain_point_rows
        wp.launch(kernel, dim=count, inputs=[*args, work], block_dim=block, device=device)
    else:
        work = terrain_pair_work(device, begin, count)
        if blocks is not None:
            work.blocks = blocks
        wp.launch(terrain_law_point.terrain_point_rows, dim=count, inputs=[*args, work], block_dim=block, device=device)
        wp.launch_tiled(
            terrain_law_point.terrain_point_rows_tiled,
            dim=work.blocks,
            inputs=[*args, work],
            block_dim=block,
            device=device,
        )
    if kernel is _plain_point_rows:
        _launch_plain_finish(shadow, model, pipeline, begin, count, out)
    else:
        _launch_finish(shadow, model, pipeline, begin, out)


def _candidates(shadow) -> int:
    """The distinct candidates a block of the point class's finish of deferred pairs holds (the most any of its pairs
    can give)."""
    return shadow.describe()["launch_shapes"]["point"]["deferred_candidates"]


def _launch_finish(shadow, model, pipeline, begin, out, *, blocks=None):
    """Run the class's finish of deferred pairs, with ``blocks`` blocks (the shadow's grid when ``None``), over the
    pairs of ``out``'s deferred list (``deferred_count``) from table pair ``begin``."""
    scratch = terrain_finish_scratch(model.device, _candidates(shadow), blocks=blocks)
    args = [begin, 0, _pair_inputs(shadow, model, pipeline), shadow.features.data(), out, scratch]
    wp.launch_tiled(
        terrain_law_point.terrain_point_rows_finish,
        dim=scratch.blocks,
        inputs=args,
        block_dim=terrain_law_block_dim(model.device),
        device=model.device,
    )


def _side_buffer(begin, count, pair_count, device):
    """A side buffer for ``count`` pairs from table pair ``begin`` (walk slots from 0, then a spill region of as many
    rows), deferred list from ``begin``."""
    slots = count * terrain_law_point.TERRAIN_POINT_WALK_SLOTS
    return terrain_law_rows(device, pair_count, slots, max(slots, 4096), [begin] * 4)


def _host_rows(out) -> dict:
    """A side buffer's per-pair first rows, counts and states and its rows (host arrays), its spill rows asked for
    (``spilled``), and its point class's deferred pairs: how many (``deferred``) and which (``deferred_pairs``, in
    list order)."""
    rows = {f: getattr(out, f).numpy() for f in ("first", "count", "state", "q", "n", "d", "feature")}
    deferred = int(out.deferred_count.numpy()[TERRAIN_PARTNER_POINT])
    first = int(out.deferred_first.numpy()[TERRAIN_PARTNER_POINT])
    rows["deferred"] = deferred
    rows["deferred_pairs"] = out.deferred.numpy()[first : first + deferred]
    rows["slots"] = out.slots
    rows["spilled"] = int(out.spilled.numpy()[0])
    return rows


def _pair_rows(rows: dict, pair: int) -> slice:
    """The rows of a table pair in a side buffer (:func:`_host_rows`): its walk slots or its spill rows."""
    first = int(rows["first"][pair])
    return slice(first, first + int(rows["count"][pair]))


def _bits(values: np.ndarray) -> np.ndarray:
    """The bits of an array: float32 values as uint32, so that equal means bit for bit (``-0.0`` is not ``0.0``)."""
    values = np.ascontiguousarray(values)
    return values.view(np.uint32) if values.dtype == np.float32 else values


def _assert_same_rows(a: dict, b: dict, begin: int, count: int, what: str):
    """Assert that two side buffers (:func:`_host_rows`) give table pairs ``[begin, begin + count)`` the same count,
    state and rows, bit for bit, wherever each keeps them (walk slots or spill rows, which take the order the pairs are
    finished in)."""
    pairs = slice(begin, begin + count)
    np.testing.assert_array_equal(a["count"][pairs], b["count"][pairs], err_msg=f"{what}: count")
    np.testing.assert_array_equal(a["state"][pairs], b["state"][pairs], err_msg=f"{what}: state")
    for i in range(count):
        rows_a, rows_b = _pair_rows(a, begin + i), _pair_rows(b, begin + i)
        for f in ("q", "n", "d", "feature"):
            np.testing.assert_array_equal(_bits(a[f][rows_a]), _bits(b[f][rows_b]), err_msg=f"{what}: pair {i} {f}")


def _finish_on_every_pair(shadow, model, pipeline, begin, count) -> dict:
    """Run the class's finish of deferred pairs on every one of table pairs ``[begin, begin + count)``, seven blocks
    taking them in turn through the same scratch, and return its side buffer (:func:`_host_rows`)."""
    finished = _side_buffer(begin, count, shadow.table.pair_count, model.device)
    finished.deferred.assign(np.arange(shadow.table.pair_count, dtype=np.int32))
    finished.deferred_count.assign(np.full(4, count, np.int32))
    _launch_finish(shadow, model, pipeline, begin, finished, blocks=7)
    return _host_rows(finished)


def _assert_finished_rows(test, walked: dict, finished: dict, begin: int, count: int, what: str) -> int:
    """Assert that the class's finish of deferred pairs run on every pair (:func:`_finish_on_every_pair`) gives each
    pair that reaches its window the count, state and rows of its walk, bit for bit: so the walks write a pair's rows
    as the finish keeps them. Returns the pairs compared."""
    compared = 0
    for i in range(count):
        pair = begin + i
        if walked["state"][pair] & EXITS:
            continue
        compared += 1
        test.assertEqual(walked["count"][pair], finished["count"][pair], msg=f"{what} pair {i}")
        test.assertEqual(walked["state"][pair], finished["state"][pair], msg=f"{what} pair {i}")
        rows_walked, rows_finished = _pair_rows(walked, pair), _pair_rows(finished, pair)
        for f in ("q", "n", "d", "feature"):
            np.testing.assert_array_equal(
                _bits(walked[f][rows_walked]), _bits(finished[f][rows_finished]), err_msg=f"{what} pair {i} {f}"
            )
    return compared


def _production_rows(shadow, model, pipeline) -> dict:
    """Launch the shadow as a collide does (:meth:`TerrainLawShadow.launch`, the production protocol: the one-thread
    kernel over the class, queueing the tiled segment's pairs that pass the whole-pair rejects; the tiled kernel's
    fixed grid over the queue; the fixed grid of the finish of deferred pairs over the deferred list), and return its
    side buffer (:func:`_host_rows`) and its queued pairs (``queued``)."""
    shadow.launch(
        pipeline.geom_transform,
        pipeline.geom_data,
        shape_type=model.shape_type,
        shape_gap=model.shape_gap,
        shape_heightfield_index=model.shape_heightfield_index,
        **pipeline._broad_phase_pair_filter(model),
    )
    rows = _host_rows(shadow._out)
    rows["queued"] = sum(
        int(w[0].count.numpy()[0]) for kind, *_launch, w in shadow._launches if kind == "thread" and not w[1]
    )
    return rows


MARKED_BIT_31 = (
    (0.8063436150550842, -0.8646434545516968, 0.34505870938301086),
    (-0.3956831395626068, 0.1805560439825058, 0.33991286158561707),
    (-0.7925142645835876, -0.43028581142425537, 0.3399025499820709),
)
"""Centres of 0.25 m spheres on :func:`_tile_scenes`' rough tile whose rows include one owned by the window vertex of
index 31: bit 31, the sign bit, of the tiled kernel's first mask word."""

AFTER_EMPTY_WORD = (
    (0.8946513533592224, -0.5399307012557983, 0.34645599126815796),
    (0.8948346972465515, 0.07594582438468933, 0.3388945758342743),
)
"""Centres of 0.25 m spheres on :func:`_tile_scenes`' rough tile whose window vertices 0-31 yield nothing and whose
vertex 32 yields: the tiled kernels' first lane skips the empty first mask word onto a marked vertex (2 of 600 random
placements)."""


def _tile_scenes(rng):
    """Scenes whose pairs all take the tiled kernel (windows over 16 cells): ``(name, heights, half extent, bodies)``.

    Large spheres and long capsules rest on rough terrain; long capsules lie across ridges of one height, whose ridge
    rows share a plane, so that the finish may drop some of them (deferred pairs); spheres whose rows mark
    bit 31 of a mask word (:data:`MARKED_BIT_31`) and spheres whose first mask word is empty and whose next word's first
    vertex yields (:data:`AFTER_EMPTY_WORD`); a sphere and a capsule whose windows hold over 256 vertices, which the
    tiled kernel's first lane walks unmasked.
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
    skipped = [("sphere", {"radius": 0.25}, np.array([*c, 0.0, 0.0, 0.0, 1.0])) for c in AFTER_EMPTY_WORD]
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
        ("after an empty word", heights, 1.6, skipped),
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

    Both kernels evaluate the same table segment of tile-lane pairs (:func:`_tile_scenes`), the finish of deferred pairs
    after them; some pairs go through that finish. The tiled kernel takes the pairs the one-thread kernel queues: with a
    block per pair, and with three blocks that take the queue in turn. The finish of deferred pairs, which marks and
    walks a window with the tiled kernel's code, also runs on every pair (seven blocks taking them in turn) and gives
    each walked pair its walk's rows. On the CPU Warp runs ``wp.launch_tiled`` with one lane per pair (``block_dim`` 1),
    so the CPU run checks one lane's walk, its masked and unmasked paths, bit 31 of a mask word, the skip of an empty
    mask word onto a marked vertex and the blocks' turns; the 32-lane path (the window extent's ``tile_max``, the union
    of the lanes' mask words by ``tile_sum`` and the lanes' striding) runs on CUDA only, in this test's CUDA variant.
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
            for lane, blocks in ((TERRAIN_LANE_THREAD, None), (TERRAIN_LANE_TILE, None), (TERRAIN_LANE_TILE, 3)):
                out = _side_buffer(begin, end - begin, shadow.table.pair_count, device)
                _launch_segment(shadow, model, pipeline, lane, begin, end - begin, out, blocks=blocks)
                outputs.append(out)
            one = outputs[0]
            test.assertGreaterEqual(int(one.count.numpy().sum()), len(bodies))
            deferred += int(one.deferred_count.numpy()[0])
            for tiled in outputs[1:]:
                test.assertEqual(int(one.deferred_count.numpy()[0]), int(tiled.deferred_count.numpy()[0]))
                _assert_same_rows(_host_rows(one), _host_rows(tiled), begin, end - begin, f"{name}: tiled")
            finished = _finish_on_every_pair(shadow, model, pipeline, begin, end - begin)
            _assert_finished_rows(test, _host_rows(one), finished, begin, end - begin, name)
            hf = tu.reference_surface(tu.heightfield_model(heights, extent, extent, device="cpu"))
            owners = _owner_vertex_indices(hf, tu.shadow_rows(pipeline), bodies, shapes)
            if name == "marked bit 31":
                # The scene covers what it is for: a row of every body owned by window vertex 31.
                test.assertTrue(all(31 in rows for rows, _vertices in owners), msg=owners)
            elif name == "after an empty word":
                # Every body's window vertices 0-31 yield nothing and its vertex 32 yields (the kernels' own marks).
                for shape in shapes:
                    marks = _window_masks(shadow, model, pipeline, shape)[:, 3]
                    test.assertTrue(len(marks) <= 256 and not marks[:32].any() and marks[32] != 0, msg=marks)
            elif name == "unmasked":
                test.assertTrue(all(vertices > 256 and rows for rows, vertices in owners), msg=owners)
    # Both paths deferred pairs to the finish of deferred pairs too.
    test.assertGreater(deferred, 0)


# ----------------------------------------------------------------------------------------------------------------
# the kernels' rejects against the plain walk
# ----------------------------------------------------------------------------------------------------------------

# The plain walk compiles here, with the point module's functions it calls: without contraction, as that module
# compiles, so that it rounds as the kernels do on CUDA too.
wp.set_module_options({"fuse_fp": False})

_Claim = terrain_law_point._Claim
_PLAIN_ROWS = wp.constant(terrain_law_point.TERRAIN_POINT_WALK_SLOTS)
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
    """``terrain_law_point._point_item_border`` (every item of any window, the border's too) with
    :func:`_plain_end_face`."""
    end = item // 6
    feature = item - 6 * end
    if end == 2:
        feature = item - 10
    k = core.a
    below = core.below_a
    if end == 1:
        k = core.b
        below = core.below_b
    if feature < 2:
        if not (r <= r1 and c <= c1):
            return planes, _Claim()
        planes, n, a = terrain_plane(data, window, planes, r, c, feature)
        return planes, _plain_end_face(data, window, core, end, k, below, r, c, feature, n, a)
    if end == 2:
        return terrain_law_point._axis_item_border(data, window, planes, core, r, c, r1, c1, feature)
    return terrain_law_point._end_item_border(data, window, planes, core, end, k, below, r, c, r1, c1, feature)


@wp.kernel(enable_backward=False)
def _plain_point_rows(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    work: TerrainPairWork,
):
    """``terrain_point_rows`` without its rejects: every item of every window vertex in window order (no vertex
    drop, no vertex mask, no off-cell face reject), then the same close; every pair is walked (``work`` unused)."""
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
    ka = terrain_window_point(window, a)
    kb = terrain_window_point(window, b)
    extent = wp.max(
        wp.max(wp.max(wp.abs(ka[0]), wp.abs(ka[1])), wp.abs(ka[2])),
        wp.max(wp.max(wp.abs(kb[0]), wp.abs(kb[1])), wp.abs(kb[2])),
    )
    eps = terrain_eps(terrain_window_extent(data, window, r1, c1, extent))
    planes = terrain_planes()
    planes, core = terrain_law_point._pair_core(data, window, planes, ka, kb, segment, reach, eps)
    terrain_law_point._open_pair(out, slot)
    held = int(0)
    items = terrain_law_point._vertex_items(core)
    for r in range(cells[0], r1 + 2):
        for c in range(cells[2], c1 + 2):
            if held <= _PLAIN_HELD:
                for item in range(items):
                    planes, claim = _plain_point_item(data, window, planes, core, r, c, r1, c1, item)
                    held = terrain_law_point._walk_claim(out, slot, held, claim, eps)
    terrain_law_point._close_pair(
        out, pair, slot, terrain_law_point._wide(cells), held, data, window, planes, core, frame
    )


_PLAIN_CLASS = wp.constant(TERRAIN_PARTNER_POINT)
_PLAIN_ROW = wp.constant(int(terrain_law_point._ROW))
_PLAIN_EXIT_A = wp.constant(int(terrain_law_point._EXIT_A))
_PLAIN_EXIT_B = wp.constant(int(terrain_law_point._EXIT_B))
_PLAIN_SLOT_AXIS = wp.constant(int(terrain_law_point._SLOT_AXIS))


@wp.func
def _plain_candidate(scratch: TerrainFinishScratch, base: int, count: int, claim: _Claim) -> int:
    """A pair's raw candidate list with a claim's row appended (counted past the list, which the bound on a pair's
    candidates makes room for)."""
    if count < scratch.capacity:
        scratch.q[base + count] = claim.q
        scratch.k[base + count] = claim.k
        scratch.n[base + count] = claim.n
        scratch.d[base + count] = claim.d
        scratch.feature[base + count] = claim.feature
    return count + 1


@wp.kernel(enable_backward=False)
def _plain_point_rows_finish(
    begin: int,
    slot_begin: int,
    inputs: TerrainPairInputs,
    data: TerrainFeatureData,
    out: TerrainLawRows,
    scratch: TerrainFinishScratch,
):
    """The finish of the pairs a walk deferred, as the plain walk takes them: ``terrain_point_rows_finish`` as one
    thread per deferred pair (one lane, no fixed grid), without the kernels' rejects and marks, and with every raw
    candidate held (the finish merges them, where the kernel merges them as it holds them).

    Every item of every window vertex in window order (:func:`_plain_point_item`: no vertex drop, no item marks, no
    off-cell face reject) goes into the entry's candidate list, then the ends' exits (G2, G3) and the axis interior's
    G3, then the law's finish (``terrain_finish_list``); rows beyond the walk slots go to the spill region.
    """
    entry = wp.tid()
    if entry >= out.deferred_count[_PLAIN_CLASS]:
        return
    base = entry * scratch.capacity
    pair = out.deferred[out.deferred_first[_PLAIN_CLASS] + entry]
    _state, frame, heightfield, cells, a, b, segment, reach = terrain_law_point._pair_setup(inputs, data, pair)
    r1 = cells[1]
    c1 = cells[3]
    window = terrain_window(data, heightfield, cells[0], cells[2])
    ka = terrain_window_point(window, a)
    kb = terrain_window_point(window, b)
    extent = wp.max(
        wp.max(wp.max(wp.abs(ka[0]), wp.abs(ka[1])), wp.abs(ka[2])),
        wp.max(wp.max(wp.abs(kb[0]), wp.abs(kb[1])), wp.abs(kb[2])),
    )
    eps = terrain_eps(terrain_window_extent(data, window, r1, c1, extent))
    planes = terrain_planes()
    planes, core = terrain_law_point._pair_core(data, window, planes, ka, kb, segment, reach, eps)
    count = int(0)
    exit_a = _Claim()
    exit_b = _Claim()
    items = terrain_law_point._vertex_items(core)
    for r in range(cells[0], r1 + 2):
        for c in range(cells[2], c1 + 2):
            for item in range(items):
                planes, claim = _plain_point_item(data, window, planes, core, r, c, r1, c1, item)
                # G2: an end's exit with the smallest translation, the first in window order on ties.
                if claim.kind == _PLAIN_EXIT_A and (exit_a.kind == 0 or claim.d > exit_a.d):
                    exit_a = claim
                elif claim.kind == _PLAIN_EXIT_B and (exit_b.kind == 0 or claim.d > exit_b.d):
                    # With both ends outside: the axis's deepest crossing under a crease.
                    exit_b = claim
                elif claim.kind == _PLAIN_ROW:
                    count = _plain_candidate(scratch, base, count, claim)
    # G2 and G3: one exit per end inside the solid, after the window's rows.
    for end in range(2):
        if (end == 0 and core.below_a != 0) or (end == 1 and core.segment != 0 and core.below_b != 0):
            claim = exit_a
            k = core.a
            if end == 1:
                claim = exit_b
                k = core.b
            if claim.kind != 0:
                claim.kind = _PLAIN_ROW
                claim.k = k
            else:
                planes, claim = terrain_law_point._exit_row(data, window, planes, k, end)
            if claim.kind != 0:
                count = _plain_candidate(scratch, base, count, claim)
    # G3 for the axis interior: both ends outside, no axis exit, a crossing of the axis under a crease.
    if core.segment != 0 and core.below_a == 0 and core.below_b == 0 and exit_b.kind != 0:
        axis_exit = bool(False)
        for row in range(wp.min(count, scratch.capacity)):
            if (scratch.feature[base + row] & 7) == _PLAIN_SLOT_AXIS and scratch.d[base + row] < 0.0:
                axis_exit = True
        if not axis_exit:
            planes, claim = terrain_law_point._axis_recovery(data, window, planes, exit_b.q, exit_b.feature)
            if claim.kind != 0:
                count = _plain_candidate(scratch, base, count, claim)
    flags = int(0)
    if count > scratch.capacity:
        flags = TERRAIN_FINISH_CANDIDATE_OVERFLOW
        count = scratch.capacity
    kept = terrain_finish_list(scratch, base, count, eps, 0, 1)
    slot = slot_begin + (pair - begin) * _PLAIN_ROWS
    if kept > _PLAIN_ROWS:
        first = terrain_spill_rows(out, kept)
        if first >= 0:
            slot = first
        else:
            kept = terrain_finish_overflow(scratch, base, count, _PLAIN_ROWS)
            flags = flags | TERRAIN_FINISH_ROW_OVERFLOW
    written = int(0)
    for row in range(count):
        if scratch.mark[base + row] != 0:
            terrain_store_row(
                out,
                slot + written,
                scratch.q[base + row],
                scratch.n[base + row],
                scratch.d[base + row],
                scratch.feature[base + row],
                window,
                frame,
            )
            written += 1
    out.first[pair] = slot
    out.count[pair] = kept
    out.state[pair] = terrain_law_point._wide(cells) | terrain_finish_state(flags)
    terrain_report(out.status, out.overflow, flags, True)


def _launch_plain_finish(shadow, model, pipeline, begin, count, out):
    """Run the plain finish (:func:`_plain_point_rows_finish`) over ``out``'s deferred list, one thread per entry
    (at most ``count``, each with a candidate list of the class's bound), from table pair ``begin``."""
    scratch = terrain_finish_scratch(model.device, _candidates(shadow), blocks=count)
    args = [begin, 0, _pair_inputs(shadow, model, pipeline), shadow.features.data(), out, scratch]
    wp.launch(
        _plain_point_rows_finish,
        dim=count,
        inputs=args,
        block_dim=terrain_law_block_dim(model.device),
        device=model.device,
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
    bodies = _recovery_capsules()
    yield "recovery", tu.body_scene(_ridge_terrain(), 1.0, 1.0, bodies, device=device)[0], bodies
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
    pair the same way, and finishes the pairs it defers the same way too (:func:`_plain_point_rows_finish`). On rough,
    pitted, stepped, seam-adversarial, buried, ridge, axis-recovery and reach-limit scenes the count, state, q, n, d and
    feature of every pair agree bit for bit with the plain walk's: the one-thread kernel's walking every pair, and the
    production launches' (the shadow's own: the larger windows queued for the tiled kernel, whose lanes mark the items
    that yield anything, a capsule's far ends dropped, and whose first lane walks the marked items). So do those of the
    class's finish of deferred pairs (which walks with the drop and runs the finish) run on every walked pair, which
    also checks that the walks write a pair's rows as the finish keeps them. That finish runs with fewer blocks than
    deferred pairs, as its fixed grid does when many pairs are deferred: each block finishes several pairs in turn,
    through the same scratch.
    """
    walked = 0
    queued = 0
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
                _launch_segment(shadow, model, pipeline, TERRAIN_LANE_THREAD, begin, count, out, kernel=kernel)
                results[label] = _host_rows(out)
            results["production"] = _production_rows(shadow, model, pipeline)
            queued += results["production"]["queued"]
            for label in ("kernel", "production"):
                _assert_same_rows(results[label], results["plain"], begin, count, f"{name}: {label} against plain")
            finished = _finish_on_every_pair(shadow, model, pipeline, begin, count)
            walked += _assert_finished_rows(test, results["kernel"], finished, begin, count, name)
    test.assertGreater(walked, 200)
    # The production launches walked larger windows in the tiled kernel: capsules and spheres of every scene.
    test.assertGreater(queued, 80)


# ----------------------------------------------------------------------------------------------------------------
# the finish of deferred pairs
# ----------------------------------------------------------------------------------------------------------------


RIDGE_CREST = 0.03
"""Crest height [m] of :func:`_deferring_scene`'s ridges, between its raised rows."""


def _deferring_scene(rng):
    """A scene whose walks defer dozens of pairs to the finish of deferred pairs: ``(heights, half extent, bodies)``.

    Ridges of one height every second column of 0.1 m cells, every fourth row raised 15 mm. Capsules 0.6-1.2 m long lie
    across the ridges or at random yaws, their lowest points 2 cm below to 2 cm above the crests (the raised rows poke
    into them), and spheres of 0.05-0.25 m lie 3 cm below to 2 cm above the raised rows' crests: rows on several faces
    of one plane, rows of every ridge in reach (more than the slots: they overflow) and exits. Six capsules 2 m long
    lie diagonally (windows of more than 256 vertices, which the tiled kernels walk unmasked), and three 2.4-3.6 m long
    lie across every ridge.
    """
    heights = np.zeros((41, 41))
    heights[:, 1::2] = RIDGE_CREST
    heights[::4, :] += 0.015

    def lying(dims, x, y, yaw, offsets):
        z = RIDGE_CREST + dims["radius"] + float(rng.choice(offsets))
        q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), yaw), LYING)
        return "capsule", dims, np.array([x, y, z, *q], np.float32).astype(np.float64)

    bodies = []
    for i in range(72):
        x, y = rng.uniform(-1.2, 1.2, 2)
        if i % 3 == 2:
            dims = {"radius": float(rng.choice([0.05, 0.12, 0.25]))}
            z = RIDGE_CREST + 0.015 + dims["radius"] + float(rng.choice([-0.03, -0.01, 0.0, 0.005, 0.02]))
            bodies.append(("sphere", dims, np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], np.float32).astype(np.float64)))
            continue
        dims = {"radius": float(rng.choice([0.02, 0.03])), "half_height": float(rng.choice([0.3, 0.6]))}
        yaw = rng.uniform(-0.5, 0.5) if i % 3 == 0 else rng.uniform(0.0, 2.0 * np.pi)
        bodies.append(lying(dims, x, y, yaw, (-0.02, -0.005, 0.0, 0.003, 0.01, 0.02)))
    for i in range(6):
        yaw = np.pi / 4.0 + rng.uniform(-0.2, 0.2) + (np.pi / 2.0 if i % 2 else 0.0)
        x, y = rng.uniform(-0.4, 0.4, 2)
        bodies.append(lying({"radius": 0.03, "half_height": 1.0}, x, y, yaw, (-0.005, 0.0, 0.003)))
    for half in (1.2, 1.5, 1.8):
        bodies.append(lying({"radius": 0.03, "half_height": half}, 0.0, 0.05, rng.uniform(0.0, 0.3), (-0.005, 0.0)))
    return heights, 2.0, bodies


def _deferring_rows(heights, extent, bodies, device):
    """A scene's point pairs through the production launches and through the plain walk and its finish:
    ``(production, plain, begin, count)``, the side buffers as :func:`_host_rows` and the class's table pairs."""
    model = tu.body_scene(heights, extent, extent, bodies, device=device)[0]
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    begin = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_THREAD)[0]
    count = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_TILE)[1] - begin
    production = _production_rows(shadow, model, pipeline)
    plain = _side_buffer(begin, count, shadow.table.pair_count, device)
    _launch_segment(shadow, model, pipeline, TERRAIN_LANE_THREAD, begin, count, plain, kernel=_plain_point_rows)
    return production, _host_rows(plain), begin, count


def test_deferred_pairs_keep_the_plain_finish_rows(test, device):
    """Give the pairs the walks defer the rows of the plain walk's own finish, bit for bit.

    A walk defers a pair whose rows the finish may change (a plane group of three rows, more rows than the walk slots)
    to the class's finish of deferred pairs, whose fixed grid of lane blocks marks the window's productive items, walks
    them on the first lane into its block's distinct candidates and runs the finish on them with its lanes. The
    production launches (the shadow's own) are compared with the plain walk and its finish
    (:func:`_plain_point_rows_finish`: one thread per deferred pair, every item of every window vertex, every raw
    candidate held) on a scene that defers dozens of pairs (:func:`_deferring_scene`), some with more rows than their
    walk slots (written to the spill region), some with windows that finish walks unmasked. On CUDA the rows also equal
    a CPU run's, bit for bit (there the lanes split the windows, mark the items, take their union and split the
    finish's implied-row tests, where the CPU's one lane runs alone).
    """
    heights, extent, bodies = _deferring_scene(np.random.default_rng(77))
    production, plain, begin, count = _deferring_rows(heights, extent, bodies, device)
    test.assertEqual(count, len(bodies))
    _assert_same_rows(production, plain, begin, count, "production against plain")
    # Not vacuous: dozens of deferred pairs (both sides defer the same ones), some with more rows than their walk slots,
    # and some with windows of more than 256 vertices.
    test.assertGreater(production["deferred"], 20)
    test.assertEqual(production["deferred"], plain["deferred"])
    counts = production["count"][begin : begin + count]
    test.assertGreater(int(np.count_nonzero(counts > terrain_law_point.TERRAIN_POINT_WALK_SLOTS)), 2)
    test.assertEqual(production["spilled"], int(counts[counts > terrain_law_point.TERRAIN_POINT_WALK_SLOTS].sum()))
    test.assertEqual(int(np.count_nonzero(production["state"][begin : begin + count] & 48)), 0)
    hf = tu.reference_surface(tu.heightfield_model(heights, extent, extent, device="cpu"))
    unmasked = 0
    for pair in production["deferred_pairs"]:
        kind, dims, pose = bodies[pair - begin]
        r0, r1, c0, c1 = tu.window_cells(hf, ref.make_core(kind, dims, pose), dims["radius"] + REACH)
        unmasked += (r1 - r0 + 2) * (c1 - c0 + 2) > terrain_law_point.TERRAIN_POINT_MASK_VERTICES
    test.assertGreater(unmasked, 0)
    if device.is_cuda:
        cpu, _plain, _begin, _count = _deferring_rows(heights, extent, bodies, "cpu")
        _assert_same_rows(production, cpu, begin, count, f"{device} against cpu")


def test_fixed_grids_take_each_pair_once(test, device):
    """Take every queued and every deferred pair once, even in a launch with more blocks than the grid.

    The tiled kernel and the finish of deferred pairs stride over their lists by the grids fixed in the work list and
    in the finish scratch (``work.blocks``, ``scratch.blocks``); a block past that grid returns at once. Launched with
    three blocks more (and scratch room for them), the tiled kernel would otherwise take queued pairs twice, deferring
    them twice, and the finish would finish deferred pairs twice, reserving their spill rows twice.
    """
    heights, extent, bodies = _deferring_scene(np.random.default_rng(77))
    model = tu.body_scene(heights, extent, extent, bodies, device=device)[0]
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    begin = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_THREAD)[0]
    tiled_begin, end = shadow.table.segment(TERRAIN_PARTNER_POINT, TERRAIN_LANE_TILE)
    count = end - begin
    args = [_pair_inputs(shadow, model, pipeline), shadow.features.data()]
    block = terrain_law_block_dim(device)
    blocks = 2
    results = []
    for extra in (0, 3):
        out = _side_buffer(begin, count, shadow.table.pair_count, device)
        # Room for pairs deferred more than once, so that blocks past the grid fail the counts below instead of
        # writing past the deferred list.
        out.deferred = wp.zeros(4 * shadow.table.pair_count, dtype=wp.int32, device=device)
        work = terrain_pair_work(device, tiled_begin, end - tiled_begin)
        work.blocks = blocks
        thread_args = [begin, 0, *args, out, work]
        wp.launch(terrain_law_point.terrain_point_rows, dim=count, inputs=thread_args, block_dim=block, device=device)
        tiled_args = [tiled_begin, (tiled_begin - begin) * terrain_law_point.TERRAIN_POINT_WALK_SLOTS, *args, out, work]
        wp.launch_tiled(
            terrain_law_point.terrain_point_rows_tiled,
            dim=blocks + extra,
            inputs=tiled_args,
            block_dim=block,
            device=device,
        )
        deferred = int(out.deferred_count.numpy()[TERRAIN_PARTNER_POINT])
        # The deferred list in table order, so that the entries each block takes do not depend on the device.
        listed = out.deferred.numpy()
        listed[begin : begin + deferred] = np.sort(listed[begin : begin + deferred])
        out.deferred.assign(listed)
        scratch = terrain_finish_scratch(device, _candidates(shadow), blocks=blocks + extra)
        scratch.blocks = blocks
        wp.launch_tiled(
            terrain_law_point.terrain_point_rows_finish,
            dim=blocks + extra,
            inputs=[begin, 0, *args, out, scratch],
            block_dim=block,
            device=device,
        )
        rows = _host_rows(out)
        rows["queued"] = int(work.count.numpy()[0])
        results.append(rows)
    grid, wider = results
    # Not vacuous: more queued pairs defer than the grid has blocks (so that some are taken past its first blocks),
    # and deferred pairs past its first blocks have more rows than their walk slots (spill rows).
    listed = grid["deferred_pairs"]
    test.assertGreater(int(np.count_nonzero(listed >= tiled_begin)), blocks)
    spilling = grid["count"][listed] > terrain_law_point.TERRAIN_POINT_WALK_SLOTS
    test.assertGreater(int(np.count_nonzero(spilling[blocks:])), 0)
    test.assertEqual(wider["queued"], grid["queued"])
    test.assertEqual(wider["deferred"], grid["deferred"])
    test.assertEqual(wider["spilled"], grid["spilled"])
    test.assertEqual(grid["spilled"], int(grid["count"][listed][spilling].sum()))
    _assert_same_rows(wider, grid, begin, count, "three blocks more")


@wp.kernel(enable_backward=False)
def _vertex_item_masks(inputs: TerrainPairInputs, data: TerrainFeatureData, pair: int, masks: wp.array2d[wp.int32]):
    """Per window vertex ``v`` of table pair ``pair`` (window order; rows past the window get -1): its row and column,
    the items ``_kept_items`` keeps, the items ``_vertex_marks`` marks, the items whose claim is not nothing (every
    item of the vertex evaluated by the plain walk's :func:`_plain_point_item`), and the pair's whole-pair state."""
    v = wp.tid()
    state, _frame, heightfield, cells, a, b, segment, reach = terrain_law_point._pair_setup(inputs, data, pair)
    r0 = cells[0]
    r1 = cells[1]
    c0 = cells[2]
    c1 = cells[3]
    width = c1 + 2 - c0
    if v >= (r1 + 2 - r0) * width:
        masks[v, 0] = -1
        return
    window = terrain_window(data, heightfield, r0, c0)
    ka, kb = terrain_law_point._window_points(window, a, b, segment)
    eps = terrain_eps(terrain_window_extent(data, window, r1, c1, terrain_law_point._point_extent(ka, kb)))
    planes = terrain_planes()
    planes, core = terrain_law_point._pair_core(data, window, planes, ka, kb, segment, reach, eps)
    margins = terrain_law_point._cell_margins(window, core)
    r = r0 + v // width
    c = c0 + v % width
    items = terrain_law_point._vertex_items(core)
    planes, marks = terrain_law_point._vertex_marks(data, window, planes, core, margins, r, c, r1, c1, items)
    claims = int(0)
    for item in range(items):
        planes, claim = _plain_point_item(data, window, planes, core, r, c, r1, c1, item)
        if claim.kind != 0:
            claims = claims | (1 << item)
    masks[v, 0] = r
    masks[v, 1] = c
    masks[v, 2] = terrain_law_point._kept_items(data, window, core, margins, r, c)
    masks[v, 3] = marks
    masks[v, 4] = claims
    masks[v, 5] = state


def _window_masks(shadow, model, pipeline, shape) -> np.ndarray:
    """Per window vertex of the pair of partner ``shape``, in window order: ``(r, c, kept, marks, claims, state)`` of
    :func:`_vertex_item_masks`."""
    device = model.device
    masks = wp.full((4096, 6), -1, dtype=wp.int32, device=device)
    pair = shadow.table.pairs_host[:, 1].tolist().index(shape)
    wp.launch(
        _vertex_item_masks,
        dim=masks.shape[0],
        inputs=[_pair_inputs(shadow, model, pipeline), shadow.features.data(), pair, masks],
        block_dim=terrain_law_block_dim(device),
        device=device,
    )
    masks = masks.numpy()
    return masks[masks[:, 0] >= 0]


ITEM_CASES = {
    # name: (kind, dims, pose, {cell containing (x, y): items kept}, every vertex's items or None)
    "lying capsule": (
        "capsule",
        {"radius": 0.03, "half_height": 0.3},
        (0.05, 0.05, 0.035, *LYING),
        {(0.35, 0.05): 0xF03F, (-0.25, 0.05): 0xFFC0, (0.05, 0.05): 0xF000},
        None,
    ),
    "short capsule": (
        "capsule",
        {"radius": 0.03, "half_height": 0.02},
        (0.05, 0.05, 0.035, *LYING),
        {(0.05, 0.05): 0xFFFF},
        None,
    ),
    "capsule end inside": (
        "capsule",
        {"radius": 0.03, "half_height": 0.1},
        (0.05, 0.05, 0.02, *_axis_angle((0.0, 1.0, 0.0), np.pi / 2.0 - np.arcsin(0.3))),
        {},
        0xFFFF,
    ),
    "sphere": (
        "sphere",
        {"radius": 0.25},
        (0.05, 0.05, 0.255, 0.0, 0.0, 0.0, 1.0),
        {(-0.25, -0.25): 0x0000, (0.05, 0.05): 0x003F},
        None,
    ),
}
"""Cores over a flat tile of 0.1 m cells (half extent 2 m) and the items ``_kept_items`` keeps at chosen vertices.

The vertex owning the cell under a lying capsule's end ``a`` (local +z, world +x) keeps its axis's and end ``a``'s
items, not end ``b``'s 0.6 m away; the cell under end ``b`` the reverse; the cell midway the axis's alone. Both ends of
a short capsule are near (every item); with an end inside the solid (end ``b`` 1 cm below the tile) every vertex keeps
every item, the reach bounding no exit. A sphere centre keeps its items at the cell under it and none at a window
corner cell 0.35 m off in xy."""


def test_kept_items_drop_far_core_ends(test, device):
    """Keep at a window vertex exactly the items within reach of the core, and mark exactly the items that yield.

    ``_kept_items`` passes by a capsule end's six items at a vertex whose box lies beyond the reach of the end (both
    ends outside the solid), and every item of a vertex the drop passes by (:data:`ITEM_CASES`); ``_vertex_marks``
    marks, of a vertex's kept items, those whose claim is not nothing, which must be every item whose claim is not
    nothing (each evaluated by the plain walk's item, without the off-cell reject). A drop that kept every item or marks
    that set every kept item would undo the walks' savings without changing a row.
    """
    bodies = [(kind, dims, np.asarray(pose, np.float64)) for kind, dims, pose, _cells, _every in ITEM_CASES.values()]
    model, _hfield, shapes = tu.body_scene(np.zeros((41, 41)), 2.0, 2.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    for (name, (_kind, _dims, _pose, cells, every)), shape in zip(ITEM_CASES.items(), shapes, strict=True):
        with test.subTest(case=name):
            got = _window_masks(shadow, model, pipeline, shape)
            test.assertTrue(np.all(got[:, 5] == 0), msg="the pair leaves before its window")
            kept = {(int(r), int(c)): int(k) for r, c, k in got[:, :3]}
            for (x, y), items in cells.items():
                cell = (int(np.floor((y + 2.0) / 0.1)), int(np.floor((x + 2.0) / 0.1)))
                test.assertEqual(hex(kept[cell]), hex(items), msg=f"cell {cell} under ({x}, {y})")
            if every is not None:
                test.assertEqual({hex(k) for k in kept.values()}, {hex(every)})
            np.testing.assert_array_equal(got[:, 3], got[:, 4], err_msg="marks against the items that yield")
            test.assertTrue(np.any(got[:, 3] != 0), msg="no vertex yields anything")


@functools.cache
def _point_ptx() -> str | None:
    """The point module's PTX for sm_86 (NVRTC, any host), or ``None`` where NVRTC cannot compile for it."""
    if 86 not in wp.get_cuda_supported_archs():
        return None
    with tempfile.TemporaryDirectory() as directory:
        wp.compile_aot_module(terrain_law_point, arch=86, module_dir=directory, use_ptx=True)
        paths = glob.glob(os.path.join(directory, "*.ptx"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one PTX file, got {paths}")
        with open(paths[0], encoding="latin-1") as f:
            return f.read()


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
        ptx = _point_ptx()
        if ptx is None:
            self.skipTest("NVRTC cannot compile for sm_86 here")
        self.assertEqual(ptx.count("fma.rn"), 0)
        self.assertGreater(ptx.count("mul.rn.f32"), 100)

    def test_deferred_finish_keeps_its_candidates_off_the_stack_cpu(self):
        """Keep the finish of deferred pairs' per-thread local frame under 512 bytes (its PTX local depot, sm_86,
        NVRTC).

        The driver reserves local memory for the largest stack a context launches, for every thread the device can
        hold (about 289,000 on a 188-SM GPU), launched or not. The kernel's candidates and the finish's state live in
        global scratch (:class:`~newton._src.geometry.terrain_law.TerrainFinishScratch`,
        :func:`~newton._src.geometry.terrain_law.terrain_finish_list`): the depot was 8,584 bytes with the candidate
        list on the stack (10,088 bytes of sm_120 stack, measured at +2.5 GB of device memory), then under 4.5 KB with
        the finish's 32-candidate frame (5,960 bytes of sm_120 stack, 1,342 MiB), 32 bytes of sm_120 stack since.
        """
        ptx = _point_ptx()
        if ptx is None:
            self.skipTest("NVRTC cannot compile for sm_86 here")
        depots = {}
        for entry in re.finditer(r"\.entry (\w+)\(", ptx):
            body = ptx[entry.end() : ptx.find("\n}\n", entry.end())]
            depot = re.search(r"__local_depot\d+\[(\d+)\]", body)
            depots[entry.group(1)] = int(depot.group(1)) if depot else 0
        finish = [size for name, size in depots.items() if name.startswith("terrain_point_rows_finish_")]
        self.assertEqual(len(finish), 1, msg=depots)
        self.assertLess(finish[0], 512, msg=depots)


devices = get_test_devices()
for _name, _func in (
    ("test_point_rows_are_minimal_at_flat_seams", test_point_rows_are_minimal_at_flat_seams),
    ("test_point_rows_are_complete", test_point_rows_are_complete),
    ("test_point_rows_match_the_float64_law", test_point_rows_match_the_float64_law),
    ("test_point_rows_match_the_float64_law_at_the_border", test_point_rows_match_the_float64_law_at_the_border),
    ("test_point_rows_take_the_block_at_the_border", test_point_rows_take_the_block_at_the_border),
    ("test_buried_cores_exit_by_the_minimum_translation", test_buried_cores_exit_by_the_minimum_translation),
    ("test_point_rows_have_the_plane_at_noisy_flat_seams", test_point_rows_have_the_plane_at_noisy_flat_seams),
    ("test_capsule_axis_exits_over_a_ridge", test_capsule_axis_exits_over_a_ridge),
    ("test_capsule_axis_inside_without_exit_gets_recovery", test_capsule_axis_inside_without_exit_gets_recovery),
    ("test_point_rows_point_up_on_steep_terrain", test_point_rows_point_up_on_steep_terrain),
    ("test_tiled_kernel_matches_one_thread_kernel", test_tiled_kernel_matches_one_thread_kernel),
    ("test_rejects_keep_the_plain_walk_rows", test_rejects_keep_the_plain_walk_rows),
    ("test_deferred_pairs_keep_the_plain_finish_rows", test_deferred_pairs_keep_the_plain_finish_rows),
    ("test_fixed_grids_take_each_pair_once", test_fixed_grids_take_each_pair_once),
    ("test_kept_items_drop_far_core_ends", test_kept_items_drop_far_core_ends),
):
    add_function_test(TestTerrainLawPoint, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

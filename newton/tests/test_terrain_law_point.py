# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's point-core kernels (spheres and capsules), run in shadow beside today's terrain route.

The same checks run on today's contacts and on the law's shadow rows: today's contacts fail them, the shadow rows
pass. Measured at this tree's parent (6d9c648b, default terrain flags):

* minimal: spheres resting at flat seams get 2.58 rows and 0.92 ghost normals each today, lying capsules 3.92 rows
  and 1.08 ghosts, where the plane has one row per sphere and one per capsule end;
* complete: today misses 13 of the 24 end supports of the lying capsules, and on rough and pitted terrain the
  supports of spheres and capsules the float64 law finds (5 of 40 rough spheres lose one).

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
from newton._src.geometry.terrain_law import TerrainLawRows, TerrainPairInputs
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


def _minimal_violations(rows, shapes, expected) -> list[str]:
    """Rows that are not the plane's: a wrong count, a plane row missing, or a ghost normal (over 1 deg off)."""
    violations = []
    for i, (shape, plane) in enumerate(zip(shapes, expected, strict=True)):
        got = rows.get(shape, [])
        if len(got) != len(plane):
            violations.append(f"body {i}: {len(got)} rows, the plane has {len(plane)}")
        missing = tu.missing_rows(plane, got, q_tol=1.0e-5, angle_tol=1.0e-4, d_tol=1.0e-5)
        if missing:
            violations.append(f"body {i}: {len(missing)} plane rows missing")
        ghosts = sum(1 for g in got if tu.angle_between(g[1], (0.0, 0.0, 1.0)) > GHOST_ANGLE)
        if ghosts:
            violations.append(f"body {i}: {ghosts} ghost normals")
    return violations


def test_point_rows_are_minimal_at_flat_seams(test, device):
    """Give a sphere at a flat seam the plane's one row and a lying capsule its two end rows, with no ghost normal."""
    bodies, expected = _flat_seams()
    model, _hfield, shapes = tu.body_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    test.assertEqual(_minimal_violations(tu.shadow_rows(pipeline), shapes, expected), [])


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


def _incomplete(hf, rows, shapes, bodies) -> list[str]:
    """The float64 law's supports that no row has (terrain point within 0.1 mm, normal within 1 deg)."""
    violations = []
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        law = ref.pair_rows(hf, ref.make_core(kind, dims, pose), dims["radius"] + REACH)
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
    strips: no float64 row is missed, no float32 row is extra, except rows whose deciding predicate lies within
    :data:`~newton.tests.terrain_law_test_utils.DECISION_BAND` of its threshold (at most 5% of the rows here, where
    half the placements sit exactly on seams; 0.4-1.2% over 600 placements per terrain), and every
    pair's deepest core distance agrees within 10 um.
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
            totals = {"f64": 0, "tolerated": 0}
            defects = []
            for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
                law = ref.pair_rows(hf, ref.make_core(kind, dims, pose), dims["radius"] + REACH)
                got = rows.get(shape, [])
                frame_tol = 1.0e-5 + 2.0e-7 * float(np.abs(pose[:2]).max())
                missing, extra, tolerated, deepest = tu.compare_rows(hf, law, got, frame_tol=frame_tol)
                totals["f64"] += len(law)
                totals["tolerated"] += tolerated
                if missing or extra or deepest > 1.0e-5:
                    defects.append((kind, pose.tolist(), missing, extra, deepest))
            test.assertEqual(defects, [])
            test.assertGreater(totals["f64"], len(bodies))
            test.assertLessEqual(totals["tolerated"], 0.05 * totals["f64"], msg=totals)


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
    """Give a buried sphere one row, its exact minimum translation; give a buried capsule end one exit (G2, G3).

    Sphere centres 3-60 mm inside rough, pitted and stepped terrain: one row, its depth the exact distance from the
    centre to the surface within 10 um (the rows push by exactly the minimum translation, as the float64 law does);
    no buried core is left without a row.
    """
    rng = np.random.default_rng(12)
    for name, heights in _rough_terrains(rng).items():
        with test.subTest(terrain=name):
            hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
            bodies, depths = [], []
            for i in range(40):
                kind = "sphere" if i % 2 == 0 else "capsule"
                r = R31
                dims = {"radius": r} if kind == "sphere" else {"radius": r, "half_height": 0.08}
                x, y = rng.uniform(-1.2, 1.2, 2)
                depth = float(rng.choice([0.003, 0.01, 0.03, 0.06]))
                q = np.array([0.0, 0.0, 0.0, 1.0])
                if kind == "capsule":
                    q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)), LYING)
                z = hf.height(x, y) - depth
                bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
                depths.append(depth)
            model, _hfield, shapes = tu.body_scene(heights, 1.6, 1.6, bodies, device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            rows = tu.shadow_rows(pipeline)
            for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
                got = rows.get(shape, [])
                test.assertTrue(got, msg=f"{kind} at {pose[:3].tolist()} has no row")
                if kind != "sphere":
                    continue
                test.assertEqual(len(got), 1, msg=f"sphere at {pose[:3].tolist()}: {got}")
                exact, _closest = _surface_distance(hf, pose[:3], dims["radius"] + REACH)
                law = ref.pair_rows(hf, ref.make_core(kind, dims, pose), dims["radius"] + REACH)
                test.assertEqual(len(law), 1)
                test.assertAlmostEqual(got[0][2], law[0].d, delta=1.0e-5)
                if exact < dims["radius"] + REACH - 0.05:
                    # The exit lies well inside the window: G2 takes it, the minimum translation.
                    test.assertAlmostEqual(-got[0][2], exact, delta=1.0e-5, msg=f"sphere at {pose[:3].tolist()}")


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


def _tile_scenes(rng):
    """Scenes whose pairs all take the tiled kernel (windows over 16 cells): ``(heights, half extent, bodies)``.

    Large spheres and long capsules rest on rough terrain; long capsules lie across ridges of one height, whose ridge
    rows share a plane, so that the finish may drop some of them (the full kernel's pairs).
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
    return [(heights, 1.6, bodies), (ridges, 2.0, lying)]


def test_tiled_kernel_matches_one_thread_kernel(test, device):
    """Give a pair the same rows, bit for bit, whether one thread or a tile of lanes evaluates its window.

    Both kernels evaluate the same table segment of tile-lane pairs (:func:`_tile_scenes`), the full kernel after
    them; some pairs go through the full kernel.
    """
    deferred = 0
    for heights, extent, bodies in _tile_scenes(np.random.default_rng(4)):
        model, _hfield, _shapes = tu.body_scene(heights, extent, extent, bodies, device=device)
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
        test.assertGreater(int(one.count.numpy().sum()), len(bodies))
        test.assertEqual(int(one.deferred_count.numpy()[0]), int(tiled.deferred_count.numpy()[0]))
        deferred += int(one.deferred_count.numpy()[0])
        for name in ("q", "n", "d", "feature"):
            np.testing.assert_array_equal(getattr(one, name).numpy(), getattr(tiled, name).numpy(), err_msg=name)
    # Both paths finished pairs in the full kernel too.
    test.assertGreater(deferred, 0)


class TestTerrainLawPoint(unittest.TestCase):
    def test_today_route_is_not_minimal_at_flat_seams_cpu(self):
        """Duplicate and tilt today's rows of spheres and capsules at flat seams (the defect the law removes).

        Measured at 6d9c648b: 2.58 rows and 0.92 ghost normals per sphere, 3.92 rows and 1.08 ghosts per capsule.
        """
        bodies, expected = _flat_seams()
        model, hfield, shapes = tu.body_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device="cpu")
        _pipeline, contacts, state = _collide(model, shadow=False)
        rows = tu.contact_rows(model, state, contacts, hfield)
        violations = _minimal_violations(rows, shapes, expected)
        spheres = len(SITES) * 2
        per_sphere = sum(len(rows.get(s, [])) for s in shapes[:spheres]) / spheres
        ghosts = sum(
            1
            for s in shapes[:spheres]
            for g in rows.get(s, [])
            if tu.angle_between(g[1], (0.0, 0.0, 1.0)) > GHOST_ANGLE
        )
        self.assertGreater(per_sphere, 2.0)
        self.assertGreater(ghosts, 5)
        self.assertGreater(len(violations), 20)

    def test_today_route_misses_point_supports_cpu(self):
        """Miss supports of the float64 law today: lying capsules' ends at flat seams, spheres on rough terrain."""
        bodies, expected = _flat_seams()
        model, hfield, shapes = tu.body_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device="cpu")
        _pipeline, contacts, state = _collide(model, shadow=False)
        rows = tu.contact_rows(model, state, contacts, hfield)
        capsule_ends = [
            row
            for shape, plane in zip(shapes, expected, strict=True)
            if len(plane) == 2
            for row in tu.missing_rows(plane, rows.get(shape, []), q_tol=1.0e-4, angle_tol=np.radians(1.0))
        ]
        self.assertGreater(len(capsule_ends), 5)
        missed = {}
        for name, hf, terrain, hfield, terrain_shapes, terrain_bodies in _rough_scenes("cpu", 20):
            _pipeline, contacts, state = _collide(terrain, shadow=False)
            missed[name] = _incomplete(
                hf, tu.contact_rows(terrain, state, contacts, hfield), terrain_shapes, terrain_bodies
            )
        self.assertTrue(any(v.startswith("sphere") for v in missed["rough"] + missed["pits"]), msg=missed)
        self.assertTrue(all(missed.values()), msg={k: len(v) for k, v in missed.items()})

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
    ("test_tiled_kernel_matches_one_thread_kernel", test_tiled_kernel_matches_one_thread_kernel),
):
    add_function_test(TestTerrainLawPoint, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

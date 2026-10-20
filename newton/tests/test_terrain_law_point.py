# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the rows a terrain pair of a sphere or a capsule needs: minimal at flat seams, complete on rough terrain.

Today's terrain route fails both checks. Measured at this tree's parent (6d9c648b, default terrain flags):

* minimal: spheres resting at flat seams get 2.58 rows and 0.92 ghost normals each today, lying capsules 3.92 rows
  and 1.08 ghosts, where the plane has one row per sphere and one per capsule end;
* complete: today misses 13 of the 24 end supports of the lying capsules, and on rough and pitted terrain the
  supports of spheres and capsules the float64 law finds (5 of 40 rough spheres lose one).
"""

import unittest

import numpy as np

import newton
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_test_utils as tu

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


def _collide(model):
    pipeline = newton.CollisionPipeline(model, **PIPELINE)
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


class TestTerrainLawPoint(unittest.TestCase):
    def test_today_route_is_not_minimal_at_flat_seams_cpu(self):
        """Duplicate and tilt today's rows of spheres and capsules at flat seams (the defect the law removes).

        Measured at 6d9c648b: 2.58 rows and 0.92 ghost normals per sphere, 3.92 rows and 1.08 ghosts per capsule.
        """
        bodies, expected = _flat_seams()
        model, hfield, shapes = tu.body_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device="cpu")
        _pipeline, contacts, state = _collide(model)
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
        _pipeline, contacts, state = _collide(model)
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
            _pipeline, contacts, state = _collide(terrain)
            missed[name] = _incomplete(
                hf, tu.contact_rows(terrain, state, contacts, hfield), terrain_shapes, terrain_bodies
            )
        self.assertTrue(any(v.startswith("sphere") for v in missed["rough"] + missed["pits"]), msg=missed)
        self.assertTrue(all(missed.values()), msg={k: len(v) for k, v in missed.items()})


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's solid-core kernels (boxes, cylinders and cones, convex hulls and ellipsoids) in shadow.

The same checks run on today's contacts and on the law's shadow rows: today's contacts fail them, the shadow rows
pass. Measured on this tree with the shadow unset (today's terrain route as the suite runs it, :data:`SUITE_ROUTE`):

* minimal: solids resting at flat seams (tilted boxes and cylinders, upright cylinders on a vertex, cones,
  ellipsoids, bevelled and flat soles) get rows that are not the analytic plane's (the plane's manifold is the law's
  exact answer on a plane): 106 and 102 violations among 50 bodies on 5 and 10 cm cells, 17 among 12 upright
  cylinders on a vertex without margins (the shadow rows: none);
* complete: of 120 solids buried in rough terrain (lying cylinders, cones and ellipsoids) 9 keep no contact and 14
  only ones shallower than their deepest point, and 4 of 40 cylinders through an exposed cell (the per-cell cases of
  the cylinder query's exact LP reference) get none as deep as theirs (the shadow rows: none without a row, and 1 and
  0 short, within the law's known limit for penetrating solids);
* exact geometry: random poses of every solid kind on rough, pitted and stepped terrain get a deepest row other than
  their exact separation (34 of 126 poses) or rows with normals outside the surface's normal cone (25 ghost rows).

The float32 rows equal the float64 law's (:mod:`newton.tests.terrain_law_reference`) on seam-adversarial placements
up to 100 m out, but for rows the other side implies (:func:`~newton.tests.terrain_law_solid_utils.compare_solid_rows`).
"""

import glob
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import terrain_law_box, terrain_law_cylinder, terrain_law_hull, terrain_law_solid
from newton._src.geometry.terrain_contacts import (
    TERRAIN_LANE_THREAD,
    TERRAIN_LANE_TILE,
    TERRAIN_PARTNER_BOX,
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_PARTNER_HULL,
    TERRAIN_PARTNER_WALK_SLOTS,
)
from newton._src.geometry.terrain_law import (
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_FINISH_MASK_ROWS,
    TERRAIN_LAW_TILE_BLOCKS,
    TERRAIN_PAIR_CANDIDATE_OVERFLOW,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_UNSUPPORTED,
    TerrainFinishScratch,
    TerrainPairInputs,
    terrain_finish_list,
    terrain_finish_scratch,
    terrain_finish_slots,
    terrain_law_block_dim,
    terrain_law_rows,
    terrain_pair_work,
)
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_solid_utils as su
from newton.tests import terrain_law_test_utils as tu
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    scoped_capture_after_gc,
)

PIPELINE = {"rigid_contact_max": 1 << 15, "max_triangle_pairs": 1 << 17}
SUITE_ROUTE = {
    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
    "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
    "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "1",
    "NEWTON_REDUCER_EXPORT_CLEANUP": "1",
}
"""Today's terrain route as the recorded suite states ran it (its primitive queries and cylinder split)."""
GHOST_ANGLE = 1.0
"""A row's normal more than this outside the surface's normal cone at its terrain point is a ghost [deg]."""
BOX = {"half": (0.08, 0.06, 0.04)}
CYLINDER = {"radius": 0.05, "half_height": 0.04}
CONE = {"radius": 0.05, "half_height": 0.045}
ELLIPSOID = {"semi": (0.07, 0.05, 0.03)}


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
"""A cylinder's axis (local z) onto world x."""


def _collide(model, *, shadow: bool, **kwargs):
    """Collide once on today's route as the suite runs it (:data:`SUITE_ROUTE`), with every class in shadow or none."""
    options = dict(PIPELINE, **kwargs)
    with patch.dict(os.environ, SUITE_ROUTE):
        if shadow:
            pipeline = tu.shadow_pipeline(model, classes="all", **options)
        else:
            pipeline = newton.CollisionPipeline(model, **options)
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    return pipeline, contacts, state


def _f32_pose(pose):
    """A pose as a model holds it (float32)."""
    return np.asarray(pose, np.float32).astype(np.float64)


# ----------------------------------------------------------------------------------------------------------------
# the finish in place
# ----------------------------------------------------------------------------------------------------------------

# Both finishes compile here: without contraction, as the solid modules compile them, so that the bitwise comparison
# holds in the mode the kernels run in on CUDA too.
wp.set_module_options({"fuse_fp": False})


@wp.kernel(enable_backward=False)
def _finish_both(
    q: wp.array[wp.vec3],
    n: wp.array[wp.vec3],
    d: wp.array[float],
    counts: wp.array[int],
    prefixes: wp.array[int],
    eps: wp.array[float],
    scratch: TerrainFinishScratch,
    in_slots: wp.array2d[int],
    in_list: wp.array2d[int],
):
    """Per list of rows at ``40 i``: ``terrain_finish_slots`` in place (its masks take the first ``prefixes[i]``, at
    most 32) and ``terrain_finish_list`` on all of them (core points ``q + d n``, in the list's block of a scratch),
    each as per-row keep flags and the kept count last."""
    i = wp.tid()
    base = i * 40
    count = counts[i]
    keep, kept = terrain_finish_slots(q, n, d, base, prefixes[i], eps[i])
    for j in range(prefixes[i]):
        in_slots[i, j] = (keep >> j) & 1
    in_slots[i, 40] = kept
    block = i * scratch.capacity
    for j in range(count):
        s = base + j
        scratch.q[block + j] = q[s]
        scratch.k[block + j] = q[s] + d[s] * n[s]
        scratch.n[block + j] = n[s]
        scratch.d[block + j] = d[s]
        scratch.feature[block + j] = j
    in_list[i, 40] = terrain_finish_list(scratch, block, count, eps[i])
    for j in range(count):
        in_list[i, j] = scratch.mark[block + j]


def _candidate_lists(rng, count):
    """Candidate lists that exercise every step of the finish: coincident rows, several plane groups, implied rows
    (inside or on the lower hull of their group), groups of more than four rows, and lists of more rows than the
    in-place finish's masks hold (copies of a shorter list). Returns ``(rows, prefix)`` per list: the list and how many
    of its first rows are distinct (the rest copies)."""
    lists = []
    for i in range(count):
        rows = []
        groups = int(rng.integers(1, 5))
        for _g in range(groups):
            normal = rng.normal(size=3) * np.array([0.3, 0.3, 1.0])
            normal[2] = abs(normal[2]) + 0.5
            normal /= np.linalg.norm(normal)
            e1 = np.cross(normal, [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(normal, e1)
            origin = rng.uniform(-0.05, 0.05, size=3)
            size = int(rng.integers(1, 9))
            corners = [(-0.04, -0.03), (0.04, -0.03), (0.04, 0.03), (-0.04, 0.03)]
            tilt = rng.uniform(-0.05, 0.05, size=2)
            for j in range(size):
                if j < 4 and rng.uniform() < 0.7:
                    u, v = corners[j]
                else:
                    u, v = rng.uniform(-0.05, 0.05, size=2)
                    if rng.uniform() < 0.3:
                        # on an edge between two corners (implied when not deeper)
                        a, b = corners[j % 4], corners[(j + 1) % 4]
                        t = rng.uniform()
                        u, v = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
                q = origin + u * e1 + v * e2
                dist = 0.01 + tilt[0] * u + tilt[1] * v + rng.choice([0.0, 0.0, rng.uniform(-0.002, 0.004)])
                rows.append((q, normal, dist))
                if rng.uniform() < 0.1:
                    # an inclusive claim of the same constraint: a coincident row
                    rows.append((q + rng.uniform(-1e-7, 1e-7, size=3), normal, dist))
        order = rng.permutation(len(rows))
        rows = [rows[k] for k in order]
        prefix = len(rows)
        if i % 17 == 0:
            rows = (rows * 6)[:36]
        lists.append((rows, min(prefix, len(rows))))
    return lists


def test_finish_in_place_equals_finish(test, device):
    """Keep the same rows, bit for bit, with the finish in a walk's slots as with the finish on a full-finish scratch,
    and finish any number of candidates there.

    Lists with coincident rows, several plane groups, implied rows and groups of more than four rows; lists of more
    rows than the in-place finish's masks hold (:data:`TERRAIN_FINISH_MASK_ROWS`) are copies of a shorter list, whose
    copies merge into their first occurrence: the scratch finish keeps the rows the in-place finish keeps of the
    shorter list.
    """
    rng = np.random.default_rng(31)
    lists = _candidate_lists(rng, 400)
    q = np.zeros((len(lists) * 40, 3), np.float32)
    n = np.zeros((len(lists) * 40, 3), np.float32)
    d = np.zeros(len(lists) * 40, np.float32)
    counts = np.zeros(len(lists), np.int32)
    prefixes = np.zeros(len(lists), np.int32)
    for i, (rows, prefix) in enumerate(lists):
        counts[i] = len(rows)
        prefixes[i] = prefix
        for j, (qq, nn, dd) in enumerate(rows):
            q[i * 40 + j] = qq
            n[i * 40 + j] = nn
            d[i * 40 + j] = dd
    test.assertTrue(np.all(prefixes <= TERRAIN_FINISH_MASK_ROWS))
    eps = np.full(len(lists), 8.0 * 1.1920928955078125e-7 * 0.2, np.float32)
    in_slots = wp.zeros((len(lists), 41), dtype=int, device=device)
    in_list = wp.zeros((len(lists), 41), dtype=int, device=device)
    wp.launch(
        _finish_both,
        dim=len(lists),
        inputs=[
            wp.array(q, dtype=wp.vec3, device=device),
            wp.array(n, dtype=wp.vec3, device=device),
            wp.array(d, dtype=float, device=device),
            wp.array(counts, dtype=int, device=device),
            wp.array(prefixes, dtype=int, device=device),
            wp.array(eps, dtype=float, device=device),
            terrain_finish_scratch(device, 40, blocks=len(lists)),
        ],
        outputs=[in_slots, in_list],
        device=device,
    )
    a = in_slots.numpy()
    b = in_list.numpy()
    np.testing.assert_array_equal(a[:, 40], b[:, 40])
    for i in range(len(lists)):
        np.testing.assert_array_equal(a[i, : prefixes[i]], b[i, : prefixes[i]], err_msg=f"list {i}")
        test.assertFalse(np.any(b[i, prefixes[i] : counts[i]]), msg=f"list {i}: a copy kept")
    # every step acted somewhere: rows dropped, and lists longer than the masks
    test.assertGreater(int(np.sum(b[:, 40] < counts)), 100)
    test.assertGreater(int(np.sum(counts > TERRAIN_FINISH_MASK_ROWS)), 5)


# ----------------------------------------------------------------------------------------------------------------
# minimal on flat seams (the symptom bench's tilted, vertex and sole placements, as rows)
# ----------------------------------------------------------------------------------------------------------------


def _lowest(kind, dims, q):
    core = ref.make_core(kind, dims, np.r_[0.0, 0.0, 0.0, q])
    return float(core.aabb()[0][2])


def _flat_placements(rng, cell):
    """Solids pressing 0.3-2 mm into the margins of a flat grid at seams, tilted 0-2 deg, and their plane rows.

    Boxes, cylinders, cones (base down) and ellipsoids at random sites (a third on grid lines, a tenth on vertices),
    upright cylinders on a vertex and 1 um, 1 mm and 10 mm off it, and bevelled and flat-bottomed soles at seams.
    The plane rows are the core's analytic plane manifold against ``z = 0`` (the law's answer on a plane).
    """
    reach = su.REACH
    bodies = []
    sole = su.sole_vertices()
    prism = su.sole_vertices(bevel=0.0)
    for i in range(32):
        kind = ("box", "cylinder", "cone", "ellipsoid")[i % 4]
        dims = {"box": BOX, "cylinder": CYLINDER, "cone": CONE, "ellipsoid": ELLIPSOID}[kind]
        tilt = np.radians(rng.uniform(0.0, 2.0)) if i % 8 else 0.0
        q = _quat_mul(_axis_angle((np.cos(i), np.sin(i), 0.0), tilt), _axis_angle((0.0, 0.0, 1.0), rng.uniform(0, 6.3)))
        x, y = rng.uniform(0.0, cell, size=2)
        if i % 3 == 0:
            x = 0.0
        if i % 10 == 0:
            x, y = 0.0, 0.0
        z = -_lowest(kind, dims, q) + 2.0 * su.MARGIN - rng.uniform(0.0003, 0.002)
        bodies.append((kind, dims, _f32_pose(np.r_[x - 0.4, y - 0.3 + 0.1 * (i % 7), z, q])))
    for r, hh in ((0.05, 0.04), (0.02, 0.0625), (0.07, 0.08156)):
        for k, off in enumerate((0.0, 1.0e-6, 0.001, 0.01)):
            dims = {"radius": r, "half_height": hh}
            z = hh + 2.0 * su.MARGIN - 0.0005
            bodies.append(("cylinder", dims, _f32_pose([0.3 + off, 0.2 + 0.6 * off + 0.2 * k, z, 0.0, 0.0, 0.0, 1.0])))
    for verts in (sole, prism):
        for yaw in (0.0, 30.0, 45.0):
            q = _axis_angle((0.0, 0.0, 1.0), np.radians(yaw))
            z = 0.035 / 2 + 2.0 * su.MARGIN - 0.0005
            bodies.append(
                (
                    "hull",
                    {"verts": verts},
                    _f32_pose([0.013 - 0.5 + yaw / 40.0, 0.507 + 0.25 * (verts is prism), z, *q]),
                )
            )
    up = np.array([0.0, 0.0, 1.0])
    expected = []
    for kind, dims, pose in bodies:
        core = ref.make_core(kind, dims, pose)
        expected.append([(k - h * up, up, h) for k, h in core.plane_manifold(up, np.zeros(3), reach)])
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
        ghosts = sum(1 for g in got if tu.angle_between(g[1], (0.0, 0.0, 1.0)) > np.radians(GHOST_ANGLE))
        if ghosts:
            violations.append(f"body {i}: {ghosts} ghost normals")
    return violations


def _flat_scenes(device):
    """The flat placements on 5 cm and 10 cm grids (the bench's small-terrain cells and the suite's)."""
    for cell, half in ((0.05, 1.0), (0.1, 1.0)):
        rng = np.random.default_rng(int(cell * 1000))
        bodies, expected = _flat_placements(rng, cell)
        n = int(round(2 * half / cell)) + 1
        model, hfield, shapes = su.solid_scene(np.zeros((n, n)), half, half, bodies, device=device)
        yield cell, model, hfield, shapes, expected


def test_solid_rows_are_minimal_at_flat_seams(test, device):
    """Give every solid at a flat seam the analytic plane's rows: the same points and depths, no ghost normal."""
    for cell, model, _hfield, shapes, expected in _flat_scenes(device):
        with test.subTest(cell=cell):
            pipeline, _contacts, _state = _collide(model, shadow=True)
            test.assertEqual(_minimal_violations(tu.shadow_rows(pipeline), shapes, expected), [])


def _vertex_m0_scene(device):
    """Upright cylinders dropped 2 mm onto a vertex of a flat 5 cm grid with margins 0 (the terrain track's physics
    test): three sizes, on the vertex and 1 um, 1 mm and 10 mm off it."""
    bodies = []
    for r, hh in ((0.05, 0.04), (0.02, 0.0625), (0.07, 0.08156)):
        for k, off in enumerate((0.0, 1.0e-6, 0.001, 0.01)):
            pose = [off - 0.3 + 0.3 * k, 0.6 * off - 0.3 + 0.2 * (r > 0.03) + 0.2 * (r > 0.06), hh + 0.002, 0, 0, 0, 1]
            bodies.append(("cylinder", {"radius": r, "half_height": hh}, _f32_pose(pose)))
    model, hfield, shapes = su.solid_scene(np.zeros((65, 65)), 1.6, 1.6, bodies, device=device, margin=0.0)
    up = np.array([0.0, 0.0, 1.0])
    expected = []
    for kind, dims, pose in bodies:
        core = ref.make_core(kind, dims, pose)
        expected.append([(k - h * up, up, h) for k, h in core.plane_manifold(up, np.zeros(3), 2.0 * su.GAP)])
    return model, hfield, shapes, expected


def test_solid_rows_are_minimal_on_a_vertex_without_margins(test, device):
    """Give upright cylinders over a grid vertex (margins 0, 5 cm cells) the plane's rows and no ghost normal."""
    model, _hfield, shapes, expected = _vertex_m0_scene(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    test.assertEqual(_minimal_violations(tu.shadow_rows(pipeline), shapes, expected), [])


# ----------------------------------------------------------------------------------------------------------------
# complete and free of ghost normals against exact geometry (check_solid)
# ----------------------------------------------------------------------------------------------------------------

TOE = {"radius": 0.02, "half_height": 0.0625}


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


def _solid_kinds(rng):
    return (
        ("box", BOX),
        ("cylinder", CYLINDER),
        ("cylinder", TOE),
        ("cone", CONE),
        ("ellipsoid", ELLIPSOID),
        ("hull", {"verts": su.sole_vertices()}),
        ("hull", {"verts": su.random_hull_vertices(rng)}),
    )


def _random_quat(rng, tilt=None):
    if tilt is None:
        q = rng.normal(size=4)
        return q / np.linalg.norm(q)
    axis = rng.normal(size=2)
    return _quat_mul(_axis_angle((axis[0], axis[1], 0.0), tilt), _axis_angle((0.0, 0.0, 1.0), rng.uniform(0, 6.3)))


def _top_under(hf, lo, hi, margin=0.05):
    xs = np.linspace(lo[0] - margin, hi[0] + margin, 9)
    ys = np.linspace(lo[1] - margin, hi[1] + margin, 9)
    X, Y = np.meshgrid(xs, ys)
    return float(np.nanmax(su.surface_height(hf, X.ravel(), Y.ravel())))


def _exact_poses(rng, hf, count):
    """Random poses of every solid kind: half any rotation, half tilted 2, 10 or 30 deg; a third over a grid line;
    the lowest point 5 mm into to 25 mm above the highest terrain point under the body."""
    bodies = []
    for kind, dims in _solid_kinds(rng):
        for i in range(count):
            x, y = rng.uniform(-1.1, 1.1, 2)
            if i % 3 == 0:
                x = np.round((x + hf.hx) / hf.dx) * hf.dx - hf.hx
            q = _random_quat(rng, None if i % 2 else np.radians(rng.choice([2.0, 10.0, 30.0])))
            lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
            z = _top_under(hf, lo, hi) - lo[2] + rng.uniform(-0.005, 0.025)
            bodies.append((kind, dims, _f32_pose(np.r_[x, y, z, q])))
    return bodies


SEPARATED_TOL = 5.0e-5
"""A separated core's deepest row equals its exact separation within this [m] (DESIGN's check_solid)."""
PENETRATING_TOL = 2.0e-5
"""A penetrating core's deepest row is at least as deep as its deepest point, within this [m]."""


def _exact_violations(hf, rows, shapes, bodies):
    """Incomplete pairs and ghost rows against exact geometry (DESIGN's check_solid).

    Against the core's exact separation ``s`` (:func:`~newton.tests.terrain_law_solid_utils.refined_separation`)
    when within the reach: separated, the deepest row must equal ``s`` (:data:`SEPARATED_TOL`; a deeper row is
    stricter than the surface); penetrating, it must be at least as deep (:data:`PENETRATING_TOL`). A ghost row's
    normal lies over 1 deg outside the surface's normal cone at its terrain point.
    """
    incomplete, ghosts = [], []
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        core = ref.make_core(kind, dims, pose)
        got = rows.get(shape, [])
        separation = su.refined_separation(hf, core)
        if separation <= su.REACH - 1.0e-4:
            deepest = min((g[2] for g in got), default=np.inf)
            if separation > 0.0:
                bad = abs(deepest - separation) > SEPARATED_TOL
            else:
                bad = deepest > separation + PENETRATING_TOL
            if bad:
                incomplete.append(f"{kind} at {np.round(pose[:3], 4).tolist()}: {deepest} against {separation}")
        for g in got:
            angle = su.normal_cone_angle(hf, g[0], g[1])
            if np.isfinite(angle) and angle > GHOST_ANGLE:
                ghosts.append(f"{kind} at {np.round(pose[:3], 4).tolist()}: {angle:.2f} deg")
    return incomplete, ghosts


def _exact_scenes(device, count):
    rng = np.random.default_rng(5)
    for name, heights in _rough_terrains(rng).items():
        hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
        bodies = _exact_poses(rng, hf, count)
        model, hfield, shapes = su.solid_scene(heights, 1.6, 1.6, bodies, device=device)
        yield name, hf, model, hfield, shapes, bodies


def test_solid_rows_are_complete_and_free_of_ghosts(test, device):
    """Give every solid within reach a deepest row equal to its exact separation (as deep as its deepest point when
    penetrating), and every normal inside the surface's normal cone (check_solid): boxes, cylinders, cones,
    ellipsoids, a bevelled sole and a random hull on rough, pitted and stepped terrain, the lowest point 5 mm into to
    25 mm above the terrain."""
    for name, hf, model, _hfield, shapes, bodies in _exact_scenes(device, 6):
        with test.subTest(terrain=name):
            pipeline, _contacts, _state = _collide(model, shadow=True)
            incomplete, ghosts = _exact_violations(hf, tu.shadow_rows(pipeline), shapes, bodies)
            test.assertEqual(incomplete, [])
            test.assertEqual(ghosts, [])


# ----------------------------------------------------------------------------------------------------------------
# minimal and complete on rough terrain and at a concave crease (the symptom bench's g_thin and c_crease20, as rows)
# ----------------------------------------------------------------------------------------------------------------


def _thin_rods(device, seed=11):
    """Thin rods lying on a rough tile (the symptom bench's g_thin): radii 10 and 20 mm, half heights 62.5 and 150 mm,
    any yaw, resting 0.2-2 mm into their margin shells on 5 mm steps of 0.1 m cells."""
    rng = np.random.default_rng(seed)
    heights = np.round(rng.uniform(0.0, 0.1, (33, 33)) / 0.005) * 0.005
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
    bodies = []
    for k in range(16):
        dims = {"radius": (0.01, 0.02)[k % 2], "half_height": (0.0625, 0.15)[(k // 2) % 2]}
        x, y = rng.uniform(-1.1, 1.1, 2)
        q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(0.0, 6.3)), LYING)
        lo, hi = ref.make_core("cylinder", dims, np.r_[x, y, 0.0, q]).aabb()
        z0 = _top_under(hf, lo, hi, 0.0) - lo[2]
        separation = su.refined_separation(hf, ref.make_core("cylinder", dims, np.r_[x, y, z0, q]))
        z = z0 - separation + 2.0 * su.MARGIN - rng.uniform(0.0002, 0.002)
        bodies.append(("cylinder", dims, _f32_pose(np.r_[x, y, z, q])))
    model, hfield, shapes = su.solid_scene(heights, 1.6, 1.6, bodies, device=device)
    return hf, model, hfield, shapes, bodies


def _crease_bodies(device, seed=11):
    """Solids resting in a 20 deg concave crease (the symptom bench's c_crease20: a flat floor meets a planar slope at
    a cell boundary, 5 cm cells): boxes, cylinders, a toe and a cone, upright and leaning 10 and 20 deg toward the
    slope, yawed 0, 4 or 17 deg, each 0.2-2 mm into the margin shells of both half-spaces."""
    rng = np.random.default_rng(seed)
    th, z0 = np.radians(20.0), 0.2
    x = np.linspace(-1.0, 1.0, 41)
    heights = np.tile(np.where(x < 0.0, z0 + np.minimum(np.tan(th) * -x, 0.35), z0), (65, 1))
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.6, device="cpu"))
    floor = np.array([0.0, 0.0, 1.0])
    slope = np.array([np.sin(th), 0.0, np.cos(th)])
    kinds = (
        ("box", BOX),
        ("box", {"half": (0.05, 0.09, 0.03)}),
        ("cylinder", CYLINDER),
        ("cylinder", {"radius": 0.06, "half_height": 0.03}),
        ("cylinder", TOE),
        ("cone", CONE),
    )
    bodies = []
    for k in range(18):
        kind, dims = kinds[k % len(kinds)]
        lean = np.radians((0.0, 10.0, 20.0)[k // len(kinds)])
        yaw = np.radians(rng.choice([0.0, 4.0, 17.0]))
        q = _quat_mul(_axis_angle((0.0, 1.0, 0.0), -lean), _axis_angle((0.0, 0.0, 1.0), yaw))
        core = ref.make_core(kind, dims, np.r_[0.0, 0.0, 0.0, q])
        # the translation (x, z) that sets the core's distance to both planes (through (0, 0, z0)) in the margins
        gap_floor, gap_slope = 2.0 * su.MARGIN - rng.uniform(0.0002, 0.002, size=2)
        z = gap_floor - (float(floor @ core.lowest(floor)) - z0)
        x = (gap_slope - (float(slope @ core.lowest(slope)) - slope[2] * z0) - slope[2] * z) / slope[0]
        bodies.append((kind, dims, _f32_pose(np.r_[x, -1.35 + 2.7 * k / 17, z, q])))
    model, hfield, shapes = su.solid_scene(heights, 1.0, 1.6, bodies, device=device)
    return hf, model, hfield, shapes, bodies


def _duplicates(rows) -> int:
    """Near-duplicate row pairs: core points within 1 mm and normals within 2 deg (the symptom bench's dup metric)."""
    count = 0
    for a in range(len(rows)):
        qa, na, da = tu.row_qnd(rows[a])
        for b in range(a + 1, len(rows)):
            qb, nb, db = tu.row_qnd(rows[b])
            if np.linalg.norm(qa + da * na - qb - db * nb) < 1.0e-3 and tu.angle_between(na, nb) < np.radians(2.0):
                count += 1
    return count


def _support_violations(hf, rows, shapes, bodies) -> dict:
    """Per body against the float64 law and exact geometry: rows other than the law's, near-duplicates beyond the
    law's own (at a concave crease the law keeps a point's two half-space rows, 20 deg apart: not duplicates), ghost
    normals and incomplete bodies (:func:`_exact_violations`)."""
    differ, duplicates = [], []
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        law = su.law_rows(hf, kind, dims, pose)
        got = rows.get(shape, [])
        missing, extra, deepest = su.compare_solid_rows(law, got)
        if missing or extra or deepest > 1.0e-5:
            differ.append((kind, pose.tolist(), [repr(m) for m in missing], extra, deepest))
        if _duplicates(got) > _duplicates(law):
            duplicates.append((kind, pose.tolist(), _duplicates(got), _duplicates(law)))
    incomplete, ghosts = _exact_violations(hf, rows, shapes, bodies)
    return {"differ": differ, "duplicates": duplicates, "ghosts": ghosts, "incomplete": incomplete}


def test_thin_rods_and_crease_bodies_keep_their_supports(test, device):
    """Give thin rods on rough terrain and solids in a concave crease the float64 law's rows: no ghost normal, no
    near-duplicate the law lacks, the deepest row at the exact separation.

    The symptom bench's g_thin and c_crease20 as rows (DESIGN 6.1, Track 3's must-fail-first cases). Today's route
    loses 10 of the 16 rods' supports and adds a ghost row, and in the crease loses 7 of 18 bodies' supports with 12
    ghost rows and a duplicate.
    """
    for name, scene in (("g_thin", _thin_rods), ("c_crease20", _crease_bodies)):
        with test.subTest(case=name):
            hf, model, _hfield, shapes, bodies = scene(device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            violations = _support_violations(hf, tu.shadow_rows(pipeline), shapes, bodies)
            test.assertEqual(violations, {"differ": [], "duplicates": [], "ghosts": [], "incomplete": []})
            test.assertGreater(sum(len(v) for v in tu.shadow_rows(pipeline).values()), 3 * len(bodies))


# ----------------------------------------------------------------------------------------------------------------
# complete: cylinders through one exposed cell (the exact LP reference's per-cell cases) and buried lying cylinders
# ----------------------------------------------------------------------------------------------------------------


def _lp_cases(rng, count):
    """Cylinders through one exposed cell, as the cylinder query's exact LP reference draws them: any axis (vertical,
    along x, along a diagonal, within 20 deg of horizontal, random), radius 5-60 mm, half height 5-150 mm, cells of
    5-20 cm whose face rises up to 60 deg, half of them 60 m out. The cell sits 1-10 mm above its neighbours (an exposed
    plateau) and the cylinder's deepest point over the cell lies 0.5-40 mm below its face.

    Returns ``(heights, half extent, xform, (kind, dims, pose))`` per case (the pose in the heightfield's frame).
    """
    cases = []
    while len(cases) < count:
        dx = float(rng.choice([0.05, 0.1, 0.2]))
        cells = 9
        slope = np.tan(np.radians(rng.uniform(0.0, 60.0)))
        direction = rng.uniform(0.0, 2.0 * np.pi)
        gx, gy = slope * np.cos(direction), slope * np.sin(direction)
        n = np.array([-gx, -gy, 1.0]) / np.sqrt(1.0 + gx * gx + gy * gy)
        half = 0.5 * cells * dx
        xs = np.linspace(-half, half, cells + 1)
        X, Y = np.meshgrid(xs, xs)
        heights = gx * X + gy * Y - rng.uniform(0.001, 0.01, size=X.shape)
        mid = cells // 2
        heights[mid : mid + 2, mid : mid + 2] = (
            gx * X[mid : mid + 2, mid : mid + 2] + gy * Y[mid : mid + 2, mid : mid + 2]
        )
        radius = float(np.exp(rng.uniform(np.log(0.005), np.log(0.06))))
        half_height = float(np.exp(rng.uniform(np.log(0.005), np.log(0.15))))
        kind = rng.uniform()
        if kind < 0.1:
            axis = np.array([0.0, 0.0, 1.0])
        elif kind < 0.2:
            axis = np.array([1.0, 0.0, 0.0])
        elif kind < 0.3:
            axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        elif kind < 0.6:
            el, az = np.radians(rng.uniform(-20.0, 20.0)), rng.uniform(0.0, 2.0 * np.pi)
            axis = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        else:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
        z = np.array([0.0, 0.0, 1.0])
        if axis @ z < -0.999999:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            cr = np.cross(z, axis)
            q = np.array([*cr, 1.0 + axis @ z])
            q /= np.linalg.norm(q)
        cx, cy = xs[mid] + rng.uniform(-0.5, 1.5) * dx, xs[mid] + rng.uniform(-0.5, 1.5) * dx
        dims = {"radius": radius, "half_height": half_height}
        core = ref.make_core("cylinder", dims, np.r_[cx, cy, 0.0, q])
        pts = su.boundary_samples(core, 0.001)
        over = (pts[:, 0] >= xs[mid]) & (pts[:, 0] <= xs[mid + 1]) & (pts[:, 1] >= xs[mid]) & (pts[:, 1] <= xs[mid + 1])
        if not np.any(over):
            continue
        plane = pts[over] @ n - (n @ np.array([0.0, 0.0, 0.0]))
        depth = rng.uniform(0.0005, 0.04)
        dz = (-depth - float(plane.min())) / n[2]
        offset = (
            np.array([rng.uniform(-60.0, 60.0), rng.uniform(-60.0, 60.0), 0.0]) if rng.uniform() < 0.5 else np.zeros(3)
        )
        cases.append((heights, half, offset, ("cylinder", dims, _f32_pose(np.r_[cx, cy, dz, q]))))
    return cases


def _lp_pairs(cases, device, *, shadow):
    """Per exposed-cell case: ``(rows, law, pose, separation, frame_tol)``: today's contacts or the shadow rows
    (heightfield frame), the float64 law's rows for the pose the kernels see (the float32 world pose less the float32
    heightfield offset), that pose, the core's exact separation and the rows' rounding in the world frame."""
    out = []
    for heights, half, offset, (kind, dims, pose) in cases:
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=su.MARGIN, gap=su.GAP)
        data = np.asarray(heights, np.float32)
        terrain = newton.Heightfield(data=data, nrow=data.shape[0], ncol=data.shape[1], hx=half, hy=half)
        hfield = builder.add_shape_heightfield(
            heightfield=terrain, xform=wp.transform(wp.vec3(*offset), wp.quat_identity()), cfg=cfg
        )
        world = pose.copy()
        world[:3] += offset
        shape = su.add_solid(builder, kind, dims, world, cfg)
        model = builder.finalize(device=device)
        pipeline, contacts, state = _collide(model, shadow=shadow)
        origin = np.asarray(offset, np.float32).astype(np.float64)
        seen = pose.copy()
        seen[:3] = np.asarray(world[:3], np.float32).astype(np.float64) - origin
        if shadow:
            got = tu.shadow_rows(pipeline).get(shape, [])
        else:
            got = tu.contact_rows(model, state, contacts, hfield).get(shape, [])
        got = [(g[0] - origin, *g[1:]) for g in got]
        hf = tu.reference_surface(model)
        separation = su.refined_separation(hf, ref.make_core(kind, dims, seen))
        frame_tol = 1.0e-5 + 2.0e-7 * float(np.abs(world[:2]).max())
        out.append((got, su.law_rows(hf, kind, dims, seen), seen, separation, frame_tol))
    return out


def _penetration_violations(pairs):
    """Of penetrating pairs ``(rows, law, pose, separation, frame_tol)``: rows other than the float64 law's (terrain
    points beyond ``frame_tol``), pairs with no row, and pairs whose deepest row is shallower than the core's deepest
    point (exact separation, :data:`PENETRATING_TOL`)."""
    differ, rowless, short = [], [], []
    for got, law, pose, separation, frame_tol in pairs:
        missing, extra, deepest = su.compare_solid_rows(law, got, frame_tol=frame_tol)
        if missing or extra or deepest > 1.0e-5:
            differ.append((pose.tolist(), [repr(m) for m in missing], extra, deepest))
        if not got:
            rowless.append(pose.tolist())
        elif min(g[2] for g in got) > separation + PENETRATING_TOL:
            short.append((pose.tolist(), min(g[2] for g in got), separation))
    return differ, rowless, short


LP_SHORT = 3
"""Exposed-cell cases of 40 whose deepest row may be short of the core's depth (a known limitation of the law)."""
BURIED_SHORT = 6
"""Buried solids of 120 whose deepest row may be short of the core's depth (a known limitation of the law)."""


def test_exposed_cell_rows_follow_the_law(test, device):
    """Give every cylinder through an exposed cell the float64 law's rows and a row, and all but a few their depth.

    The exposed-cell cases of the cylinder query's exact LP reference (DESIGN 7, Track 1: any axis and size, steep
    faces, 60 m out), adapted per pair. The reference requires every triangle prism's row to reach the exact minimum
    over that prism (today's route misses 1,569 of 2,000 cases), which a per-pair law that emits one support set per
    pair, not one row per triangle, does not do by design; here a pair's deepest row must reach its core's depth.

    The law does not guarantee that for every penetrating solid (DESIGN 1.3 and 8.5: no rule covers some deep cores
    and cores pierced where their footprints leave a cluster; rule R only fills pairs with no row at all). Over 4,000
    cases of this generator (100 seeds) the shadow rows equal the float64 law's in every case and leave 65 short
    (1.6%, from 3.9 mm of penetration on; in 27 of them today's route reaches the depth), none rowless; today's route
    leaves 537 short and 16 rowless. The check allows :data:`LP_SHORT` of its 40.
    """
    pairs = _lp_pairs(_lp_cases(np.random.default_rng(17), 40), device, shadow=True)
    differ, rowless, short = _penetration_violations(pairs)
    test.assertEqual(differ, [])
    test.assertEqual(rowless, [])
    test.assertLessEqual(len(short), LP_SHORT, msg=short)


def _buried_cylinders(rng, hf, count, dims=TOE):
    """Lying cylinders within 20 deg of horizontal over rough terrain, their deepest boundary point 1-40 mm below
    the surface (the recorded rough state's buried lying cylinders)."""
    bodies = []
    for _ in range(count):
        x, y = rng.uniform(-1.1, 1.1, 2)
        el, yaw = np.radians(rng.uniform(-20.0, 20.0)), rng.uniform(0.0, 2.0 * np.pi)
        q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), yaw), _quat_mul(_axis_angle((1.0, 0.0, 0.0), el), LYING))
        core = ref.make_core("cylinder", dims, np.r_[x, y, 0.0, q])
        lo, hi = core.aabb()
        z0 = _top_under(hf, lo, hi, 0.0) - lo[2]
        core = ref.make_core("cylinder", dims, np.r_[x, y, z0, q])
        separation = su.sampled_separation(hf, core, step=0.002)
        z = z0 - separation - rng.uniform(0.001, 0.04)
        bodies.append(("cylinder", dims, _f32_pose(np.r_[x, y, z, q])))
    return bodies


def _buried_solids(rng, hf, kind, dims, count):
    """Solids of one kind in any orientation over rough terrain, their deepest boundary point 1-40 mm below the
    surface."""
    bodies = []
    for _ in range(count):
        x, y = rng.uniform(-1.1, 1.1, 2)
        q = _random_quat(rng)
        core = ref.make_core(kind, dims, np.r_[x, y, 0.0, q])
        lo, hi = core.aabb()
        z0 = _top_under(hf, lo, hi, 0.0) - lo[2]
        core = ref.make_core(kind, dims, np.r_[x, y, z0, q])
        separation = su.sampled_separation(hf, core, step=0.002)
        z = z0 - separation - rng.uniform(0.001, 0.04)
        bodies.append((kind, dims, _f32_pose(np.r_[x, y, z, q])))
    return bodies


def _burial_scene(device, count=40):
    """Buried lying cylinders (toes), cones and ellipsoids on a rough tile, ``count`` of each."""
    rng = np.random.default_rng(23)
    heights = np.round(rng.uniform(0.0, 0.1, (33, 33)) / 0.005) * 0.005
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
    bodies = _buried_cylinders(rng, hf, count)
    bodies += _buried_solids(rng, hf, "cone", CONE, count)
    bodies += _buried_solids(rng, hf, "ellipsoid", ELLIPSOID, count)
    model, hfield, shapes = su.solid_scene(heights, 1.6, 1.6, bodies, device=device)
    return hf, model, hfield, shapes, bodies


def _burial_pairs(hf, rows, shapes, bodies):
    """``(rows, law, pose, separation, frame_tol)`` of every buried body: its rows, the float64 law's, its exact
    separation."""
    return [
        (
            rows.get(shape, []),
            su.law_rows(hf, kind, dims, pose),
            pose,
            su.refined_separation(hf, ref.make_core(kind, dims, pose)),
            1.0e-5,
        )
        for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True)
    ]


def test_buried_solids_follow_the_law(test, device):
    """Give every buried solid the float64 law's rows and a row, and all but a few of them their depth.

    Lying cylinders within 20 deg of horizontal (the recorded rough state's buried toes), cones and ellipsoids in any
    orientation, 40 of each, their deepest point 1-40 mm below a rough tile (the burial census, DESIGN 7 Track 1).

    The law does not guarantee the depth of every penetrating solid (DESIGN 1.3 and 8.5). Over 4,000 of each (100
    seeds) the shadow rows equal the float64 law's in every case, none rowless, and leave short 34 cylinders (0.85%,
    from 4.2 mm on), 92 cones (2.3%, from 4.4 mm) and 12 ellipsoids (0.3%, from 22 mm); today's route leaves 2,052,
    425 and 73 short, and 929 cylinders without a contact. The check allows :data:`BURIED_SHORT` of its 120.
    """
    hf, model, _hfield, shapes, bodies = _burial_scene(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    differ, rowless, short = _penetration_violations(_burial_pairs(hf, tu.shadow_rows(pipeline), shapes, bodies))
    test.assertEqual(differ, [])
    test.assertEqual(rowless, [])
    test.assertLessEqual(len(short), BURIED_SHORT, msg=short)


def test_recovery_rows_have_distinct_feature_ids(test, device):
    """Give a buried pair's recovery rows distinct feature ids (contact matching keys on them).

    An ellipsoid lies 3 cm deep in a valley of a sawtooth terrain (56 deg faces, every edge a crease): only rule R
    gives it rows, two of them from boundary points 0 and 8 under one triangle, whose indices are equal in the
    feature id's three-bit point slot.
    """
    x = np.linspace(-1.6, 1.6, 33)
    heights = np.tile(np.abs(((x + 1.6) % 0.2) - 0.1) * 1.5, (33, 1))
    pose = _f32_pose([0.992, 0.1249, 0.1397, 0.4019, -0.4489, -0.7217, -0.3407])
    model, _hfield, shapes = su.solid_scene(heights, 1.6, 1.6, [("ellipsoid", ELLIPSOID, pose)], device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    features = [g[3] for g in tu.shadow_rows(pipeline)[shapes[0]]]
    recovery = [f for f in features if (f >> 5) & 3 == int(TERRAIN_FEATURE_RECOVERY)]
    test.assertGreaterEqual(len(recovery), 2)
    test.assertEqual(len(set(features)), len(features), msg=features)


# ----------------------------------------------------------------------------------------------------------------
# float32 against float64
# ----------------------------------------------------------------------------------------------------------------


def _adversarial(rng, hf, count):
    """Seam-adversarial placements of every solid kind: 15% of reference points on x-lines, 15% y-lines, 10% on
    vertices, 10% on diagonals, 50% generic; the lowest point 3 mm into to 12 mm above the terrain."""
    bodies = []
    kinds = _solid_kinds(rng)
    for i in range(count):
        kind, dims = kinds[i % len(kinds)]
        j = int(rng.integers(6, hf.cols - 6))
        k = int(rng.integers(6, hf.rows - 6))
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
        x, y = -hf.hx + (j + fx) * hf.dx, -hf.hy + (k + fy) * hf.dy
        q = _random_quat(rng, None if i % 3 else np.radians(rng.choice([0.0, 2.0, 15.0])))
        lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
        # put the lowest point (often a corner or rim) over the seam point
        z = _top_under(hf, lo, hi) - lo[2] + rng.uniform(-0.003, 0.012)
        bodies.append((kind, dims, _f32_pose(np.r_[x, y, z, q])))
    return bodies


def test_solid_rows_match_the_float64_law(test, device):
    """Reproduce the float64 law's rows in float32 at seams up to 100 m out, for every solid kind.

    Seam-adversarial placements on a suite-like mosaic and on 200 m rough strips: no float64 row is missed and no
    float32 row is extra but those the other side's rows imply, every pair's deepest row agrees within 10 um, and the
    row counts agree within 1%.
    """
    rng = np.random.default_rng(29)
    strip = np.round(rng.uniform(0.0, 0.1, size=(21, 2001)) / 0.005) * 0.005
    terrains = {
        "mosaic": (tu.suite_like_heights(rng, 64, 64), 3.2, 3.2, 70),
        "x-strip": (strip, 100.0, 1.0, 35),
        "y-strip": (strip.T.copy(), 1.0, 100.0, 35),
    }
    for name, (heights, hx, hy, count) in terrains.items():
        with test.subTest(terrain=name):
            hf = tu.reference_surface(tu.heightfield_model(heights, hx, hy, device="cpu"))
            bodies = _adversarial(rng, hf, count)
            model, _hfield, shapes = su.solid_scene(heights, hx, hy, bodies, device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            rows = tu.shadow_rows(pipeline)
            totals = {"f64": 0, "f32": 0}
            defects = []
            for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
                law = su.law_rows(hf, kind, dims, pose)
                got = rows.get(shape, [])
                frame_tol = 1.0e-5 + 2.0e-7 * float(np.abs(pose[:2]).max())
                missing, extra, deepest = su.compare_solid_rows(law, got, frame_tol=frame_tol)
                totals["f64"] += len(law)
                totals["f32"] += len(got)
                if missing or extra or deepest > 1.0e-5:
                    defects.append((kind, pose.tolist(), [repr(m) for m in missing], extra, deepest))
            test.assertEqual(defects, [])
            test.assertGreater(totals["f64"], len(bodies))
            test.assertLessEqual(abs(totals["f32"] - totals["f64"]), 0.01 * totals["f64"] + 1, msg=totals)


# ----------------------------------------------------------------------------------------------------------------
# the tiled kernels, row overflow, unsupported shapes and the shadow
# ----------------------------------------------------------------------------------------------------------------

_SOLID_KERNELS = {
    TERRAIN_PARTNER_BOX: (terrain_law_box, "terrain_box_rows", "terrain_box_rows_tiled", "terrain_box_rows_full"),
    TERRAIN_PARTNER_CYLINDER: (
        terrain_law_cylinder,
        "terrain_cylinder_rows",
        "terrain_cylinder_rows_tiled",
        "terrain_cylinder_rows_full",
    ),
    TERRAIN_PARTNER_HULL: (terrain_law_hull, "terrain_hull_rows", "terrain_hull_rows_tiled", "terrain_hull_rows_full"),
}


def _side_buffer(begin, count, rows, pair_count, device):
    """A side buffer for ``count`` pairs from table pair ``begin`` with ``rows`` walk slots each (from 0), then a spill
    region of as many rows (at least 4096), deferred lists from ``begin``."""
    slots = count * rows
    return terrain_law_rows(device, pair_count, slots, max(slots, 4096), [begin] * 4)


def _pair_rows(out, pair: int) -> dict:
    """A table pair's rows in a side buffer (its walk slots or its spill rows), host arrays."""
    first, count = int(out.first.numpy()[pair]), int(out.count.numpy()[pair])
    return {name: getattr(out, name).numpy()[first : first + count] for name in ("q", "n", "d", "feature")}


def _full_candidates(shadow, partner_class) -> int:
    """The distinct candidates a full-finish block of a solid class holds (the most any of its pairs can give)."""
    name = {TERRAIN_PARTNER_BOX: "box", TERRAIN_PARTNER_CYLINDER: "cylinder", TERRAIN_PARTNER_HULL: "hull"}
    return shadow.describe()["launch_shapes"][name[partner_class]]["full_candidates"]


TILE_LARGE = (
    ("box", {"half": (0.6, 0.3, 0.02)}),
    ("cylinder", {"radius": 0.05, "half_height": 0.6}),
    ("cone", {"radius": 0.4, "half_height": 0.05}),
    ("ellipsoid", {"semi": (0.6, 0.3, 0.05)}),
)
"""Solids whose windows hold more vertices (over 256 on 5 cm cells) than the tiled kernels' marks cover."""


def _tile_scene(rng, device):
    """Solids whose windows exceed 16 cells (5 cm cells) over rough terrain, every class, and of every class one whose
    window exceeds 256 vertices (:data:`TILE_LARGE`): ``(heightfield, model, shapes, bodies)``."""
    n = 49
    heights = np.round(rng.uniform(0.0, 0.08, (n, n)) / 0.005) * 0.005
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.2, 1.2, device="cpu"))
    bodies = []
    for i in range(36 + len(TILE_LARGE)):
        kind, dims = _solid_kinds(rng)[i % 7] if i < 36 else TILE_LARGE[i - 36]
        x, y = rng.uniform(-0.9, 0.9, 2) if i < 36 else rng.uniform(-0.3, 0.3, 2)
        q = _random_quat(rng) if i < 36 else _random_quat(rng, np.radians(2.0))
        if i >= 36 and kind == "cylinder":
            q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), np.pi / 4.0), LYING)
        lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
        z = _top_under(hf, lo, hi) - lo[2] + rng.uniform(-0.004, 0.01)
        bodies.append((kind, dims, _f32_pose(np.r_[x, y, z, q])))
    model, _hfield, shapes = su.solid_scene(heights, 1.2, 1.2, bodies, device=device)
    return hf, model, shapes, bodies


def _window_vertices(hf, kind, dims, pose) -> int:
    """Vertices of a solid pair's window: its core's bounds widened by the reach, in cells."""
    lo, hi = ref.make_core(kind, dims, pose).aabb()
    r0, r1, c0, c1 = hf.cells_in_box(lo[:2] - su.REACH, hi[:2] + su.REACH)
    return (r1 - r0 + 2) * (c1 - c0 + 2)


def _queued(shadow, partner_class) -> np.ndarray:
    """The pairs a shadow's last launch queued for a solid class's tiled kernel (its one-thread launch's work list)."""
    module, thread, _tiled, _full = _SOLID_KERNELS[partner_class]
    work = next(extra for _kind, kernel, *_launch, extra in shadow._launches if kernel is getattr(module, thread))
    return work.pairs.numpy()[: int(work.count.numpy()[0])]


def _pair_inputs(model, pipeline) -> TerrainPairInputs:
    """The pair inputs of a shadow pipeline's last collide (its pair table and shape transforms)."""
    inputs = TerrainPairInputs()
    inputs.pairs = pipeline._terrain_law_shadow.table.pairs
    inputs.shape_type = model.shape_type
    inputs.shape_transform = pipeline.geom_transform
    inputs.shape_data = pipeline.geom_data
    inputs.shape_gap = model.shape_gap
    inputs.shape_heightfield_index = model.shape_heightfield_index
    inputs.shape_body = model.shape_body
    inputs.body_flags = model.body_flags
    inputs.include_static_kinematic_pairs = 1
    return inputs


def test_tiled_kernels_match_one_thread_kernels(test, device):
    """Give a solid pair the same rows, bit for bit, whether one thread or a tile of lanes walks its window, and so
    whichever lane the shadow's run-time choice takes.

    One arm walks every pair of the classes' tiled segments in one thread; the other queues every walking pair for the
    tiled kernel (``thread_vertices`` 1) and runs it on a grid of 3 blocks, so that blocks stride over the queue, the
    full-finish kernel on 3 blocks too. Windows of up to 256 vertices are marked by the lanes and replayed by the
    first; larger ones (one solid per class here) are walked by the first lane alone. The shadow's own rows (windows
    over 64 vertices tiled, the others in one thread) are those rows too. Every launch takes the shadow's block size;
    the CPU runs one lane.
    """
    hf, model, shapes, bodies = _tile_scene(np.random.default_rng(37), device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    table = shadow.table
    shape_class = {int(p[1]): int(c) for p, c in zip(table.pairs_host, table.partner_class, strict=True)}
    large = {
        shape_class[shape] for shape, body in zip(shapes, bodies, strict=True) if _window_vertices(hf, *body) > 256
    }
    test.assertEqual(large, {TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL})
    # the shadow's run-time choice took both lanes in every class
    queued = {c: len(_queued(shadow, c)) for c in _SOLID_KERNELS}
    test.assertTrue(all(v > 0 for v in queued.values()), msg=queued)
    expected = shadow.rows()
    first_row = np.concatenate([[0], np.cumsum(expected["count"])])
    inputs = _pair_inputs(model, pipeline)
    partners = table.pairs_host[:, 1]
    walked = 0
    for partner_class, (module, thread, tiled, full) in _SOLID_KERNELS.items():
        begin, end = table.segment(partner_class, TERRAIN_LANE_TILE)
        test.assertGreater(end - begin, 0)
        rows = TERRAIN_PARTNER_WALK_SLOTS[partner_class]
        outputs = []
        block = terrain_law_block_dim(device)
        for thread_vertices in (0, 1):
            solids = terrain_law_solid.terrain_solid_data(model, partners, thread_vertices=thread_vertices)
            out = _side_buffer(begin, end - begin, rows, table.pair_count, device)
            work = terrain_pair_work(device, begin, end - begin)
            work.blocks = 3
            scratch = terrain_finish_scratch(device, _full_candidates(shadow, partner_class), blocks=3)
            walk = [begin, 0, inputs, shadow.features.data(), out, work, solids]
            wp.launch(getattr(module, thread), dim=end - begin, inputs=walk, block_dim=block, device=device)
            if thread_vertices:
                test.assertGreater(int(work.count.numpy()[0]), 3)
                wp.launch_tiled(getattr(module, tiled), dim=work.blocks, inputs=walk, block_dim=block, device=device)
            finish = [begin, 0, inputs, shadow.features.data(), out, scratch, solids]
            wp.launch_tiled(getattr(module, full), dim=scratch.blocks, inputs=finish, block_dim=block, device=device)
            outputs.append(out)
        one, lanes = outputs
        for name in ("count", "state"):
            np.testing.assert_array_equal(getattr(one, name).numpy(), getattr(lanes, name).numpy(), err_msg=name)
        # the shadow's rows of the segment: the same, pair by pair, wherever each side keeps them
        count = one.count.numpy()[begin:end]
        np.testing.assert_array_equal(expected["count"][begin:end], count)
        np.testing.assert_array_equal(expected["state"][begin:end], one.state.numpy()[begin:end])
        for i in range(end - begin):
            a = first_row[begin + i]
            got, other = _pair_rows(one, begin + i), _pair_rows(lanes, begin + i)
            for name in ("q", "n", "d", "feature"):
                np.testing.assert_array_equal(expected[name][a : a + count[i]], got[name], err_msg=name)
                np.testing.assert_array_equal(other[name], got[name], err_msg=name)
        walked += int(count.sum())
    test.assertGreater(walked, 20)


def _ridge_boxes(device, half, ys):
    """Long boxes lying across ridges of different heights (every ridge in reach its own plane group), one per world:
    ``(heightfield, model, shapes, bodies)``. With half extents ``(0.9, 0.05, 0.02)`` a box has more distinct
    candidates (28) and rows (20) than the box class's 16 walk slots; with ``(0.9, 0.15, 0.02)``, more distinct
    candidates (48) than a full-finish list held before (32)."""
    rng = np.random.default_rng(2)
    heights = np.zeros((41, 41))
    heights[:, 1::2] = rng.uniform(0.02, 0.035, size=41)[1::2]
    hf = tu.reference_surface(tu.heightfield_model(heights, 2.0, 2.0, device="cpu"))
    dims = {"half": half}
    top = max(hf.height(x, 0.05) for x in np.linspace(-0.95, 0.95, 400))
    bodies = [
        ("box", dims, _f32_pose([0.0, y, top + half[2] + 0.002, 0.0, 0.0, np.sin(0.01), np.cos(0.01)])) for y in ys
    ]
    model, _hfield, shapes = su.solid_scene(heights, 2.0, 2.0, bodies, device=device)
    return hf, model, shapes, bodies


def test_long_box_keeps_every_support(test, device):
    """Keep every support of a box with more rows than its walk slots, as the float64 law gives them, raising nothing.

    A long box lies across ridges of different heights (every ridge in reach its own plane group): 20 rows, more than
    the box class's 16 walk slots. Its class's full-finish kernel finishes it and writes them to the spill region.
    (1cf34769 kept 16, each plane's deepest first, and raised ``terrain_rows``.)
    """
    hf, model, shapes, bodies = _ridge_boxes(device, (0.9, 0.05, 0.02), (0.05,))
    _kind, dims, pose = bodies[0]
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = pipeline._terrain_law_shadow.rows()
    law = su.law_rows(hf, "box", dims, pose)
    test.assertEqual(len(law), 20)
    test.assertGreater(len(law), terrain_law_box.TERRAIN_BOX_WALK_SLOTS)
    test.assertEqual(int(rows["count"][0]), len(law))
    test.assertEqual(int(rows["state"][0]), 0)
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
    missing, extra, _deepest = su.compare_solid_rows(law, tu.shadow_rows(pipeline)[shapes[0]])
    test.assertEqual((missing, extra), ([], []))


def test_deferred_pairs_keep_their_rows_beyond_the_grid(test, device):
    """Finish every deferred pair from its own rows, however many defer: one full-finish block for three boxes.

    Three long boxes have more rows than the box class's walk slots and defer; a full kernel of one block walks them
    again one after the other, each into the block's scratch. Every box keeps the float64 law's rows, as with the
    shadow's own grid, bit for bit, none flagged, each in rows of the spill region.
    """
    hf, model, _shapes, bodies = _ridge_boxes(device, (0.9, 0.05, 0.02), (-0.55, 0.0, 0.55))
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    expected = shadow.rows()
    test.assertTrue(np.all(expected["count"] > terrain_law_box.TERRAIN_BOX_WALK_SLOTS))
    test.assertTrue(np.all(expected["state"] == 0))
    inputs = _pair_inputs(model, pipeline)
    solids = terrain_law_solid.terrain_solid_data(model, shadow.table.pairs_host[:, 1], thread_vertices=0)
    rows = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_BOX]
    begin, end = shadow.table.segment(TERRAIN_PARTNER_BOX, TERRAIN_LANE_TILE)
    test.assertEqual(end - begin, len(bodies))
    out = _side_buffer(begin, end - begin, rows, shadow.table.pair_count, device)
    block = terrain_law_block_dim(device)
    work = terrain_pair_work(device, begin, end - begin)
    walk = [begin, 0, inputs, shadow.features.data(), out, work, solids]
    wp.launch(terrain_law_box.terrain_box_rows, dim=end - begin, inputs=walk, block_dim=block, device=device)
    test.assertEqual(int(out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), len(bodies))
    scratch = terrain_finish_scratch(device, _full_candidates(shadow, TERRAIN_PARTNER_BOX), blocks=1)
    finish = [begin, 0, inputs, shadow.features.data(), out, scratch, solids]
    wp.launch_tiled(terrain_law_box.terrain_box_rows_full, dim=1, inputs=finish, block_dim=block, device=device)
    state = out.state.numpy()
    test.assertEqual(int(out.spilled.numpy()[0]), int(expected["count"].sum()))
    for i, (_kind, dims, pose) in enumerate(bodies):
        with test.subTest(box=i):
            test.assertEqual(int(state[begin + i]), int(expected["state"][i]))
            test.assertGreaterEqual(int(out.first.numpy()[begin + i]), out.slots)
            mine = _pair_rows(out, begin + i)
            got = [
                (q.astype(np.float64), n.astype(np.float64), float(d))
                for q, n, d in zip(mine["q"], mine["n"], mine["d"], strict=True)
            ]
            first = int(np.sum(expected["count"][:i]))
            np.testing.assert_array_equal(mine["d"], expected["d"][first : first + len(got)])
            missing, extra, _deepest = su.compare_solid_rows(su.law_rows(hf, "box", dims, pose), got)
            test.assertEqual((missing, extra), ([], []))


def test_wide_box_keeps_every_support(test, device):
    """Keep every support of a pair whose distinct candidates outnumber what a full-finish list held before (32).

    A wide box across ridges has 48 distinct candidates: its class's full-finish kernel holds them all (its scratch
    holds every candidate a window of the class can give) and keeps the float64 law's 22 rows, raising nothing.
    (1cf34769 kept the first 32 candidates in window order, kept 16 rows, and flagged a candidate and a row overflow.)
    """
    hf, model, shapes, bodies = _ridge_boxes(device, (0.9, 0.15, 0.02), (0.05,))
    _kind, dims, pose = bodies[0]
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    rows = shadow.rows()
    core = ref.make_core("box", dims, pose)
    lo, hi = core.aabb()
    window = hf.window_features(hf.cells_in_box(lo[:2] - su.REACH, hi[:2] + su.REACH))
    test.assertGreater(len(ref.solid_law(hf, core, su.REACH, window)), 64)
    law = su.law_rows(hf, "box", dims, pose)
    test.assertEqual(len(law), 22)
    test.assertEqual(int(rows["count"][0]), len(law))
    test.assertEqual(int(rows["state"][0]), 0)
    test.assertEqual(shadow.overflow_counts(), (0, 0))
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
    missing, extra, _deepest = su.compare_solid_rows(law, tu.shadow_rows(pipeline)[shapes[0]])
    test.assertEqual((missing, extra), ([], []))


def test_window_past_its_bound_is_flagged(test, device):
    """Flag a pair whose window gives more distinct candidates than its full-finish scratch holds, and raise.

    Only a window its shape data no longer bounds can: the scratch holds every candidate a window of the class can give
    as the shapes were at construction. A scratch made smaller than the wide box's 48 candidates stands for such a
    window: the pair keeps the first 32 in window order, is flagged :data:`TERRAIN_PAIR_CANDIDATE_OVERFLOW`, counts a
    candidate overflow and raises ``terrain_rows``; every row it keeps is one of the float64 law's candidates.
    """
    hf, model, _shapes, bodies = _ridge_boxes(device, (0.9, 0.15, 0.02), (0.05,))
    _kind, dims, pose = bodies[0]
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    inputs = _pair_inputs(model, pipeline)
    solids = terrain_law_solid.terrain_solid_data(model, shadow.table.pairs_host[:, 1], thread_vertices=0)
    rows = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_BOX]
    begin, end = shadow.table.segment(TERRAIN_PARTNER_BOX, TERRAIN_LANE_TILE)
    out = _side_buffer(begin, end - begin, rows, shadow.table.pair_count, device)
    block = terrain_law_block_dim(device)
    walk = [begin, 0, inputs, shadow.features.data(), out, terrain_pair_work(device, begin, end - begin), solids]
    wp.launch(terrain_law_box.terrain_box_rows, dim=end - begin, inputs=walk, block_dim=block, device=device)
    scratch = terrain_finish_scratch(device, 32, blocks=1)
    finish = [begin, 0, inputs, shadow.features.data(), out, scratch, solids]
    wp.launch_tiled(terrain_law_box.terrain_box_rows_full, dim=1, inputs=finish, block_dim=block, device=device)
    state = int(out.state.numpy()[begin])
    test.assertEqual(state & int(TERRAIN_PAIR_CANDIDATE_OVERFLOW), int(TERRAIN_PAIR_CANDIDATE_OVERFLOW))
    test.assertEqual(out.overflow.numpy().tolist(), [0, 1])
    test.assertNotEqual(int(out.status.numpy()[0]), 0)
    core = ref.make_core("box", dims, pose)
    lo, hi = core.aabb()
    candidates = ref.solid_law(
        hf, core, su.REACH, hf.window_features(hf.cells_in_box(lo[:2] - su.REACH, hi[:2] + su.REACH))
    )
    mine = _pair_rows(out, begin)
    got = [
        (q.astype(np.float64), n.astype(np.float64), float(d))
        for q, n, d in zip(mine["q"], mine["n"], mine["d"], strict=True)
    ]
    test.assertGreater(len(got), 0)
    test.assertEqual(tu.missing_rows(got, candidates, q_tol=1.0e-5, angle_tol=1.0e-3, d_tol=1.0e-5), [])


def _capture_scene(device):
    """Every solid kind on ridges of different heights, in both lane classes (small solids one thread per pair, long
    ones a tile of lanes), and a long box whose rows overflow its slots (its class's full kernel finishes it)."""
    rng = np.random.default_rng(2)
    heights = np.zeros((41, 41))
    heights[:, 1::2] = rng.uniform(0.02, 0.035, size=41)[1::2]
    top = float(heights.max())
    small = (("box", BOX), ("cylinder", CYLINDER), ("cone", CONE), ("ellipsoid", ELLIPSOID))
    large = (
        ("box", {"half": (0.3, 0.06, 0.04)}),
        ("cylinder", {"radius": 0.05, "half_height": 0.3}),
        ("cone", {"radius": 0.25, "half_height": 0.1}),
        ("ellipsoid", {"semi": (0.3, 0.06, 0.04)}),
        ("hull", {"verts": su.sole_vertices()}),
    )
    bodies = []
    for i, (kind, dims) in enumerate(small + large):
        q = _random_quat(rng, np.radians(rng.uniform(0.0, 10.0)))
        if kind == "cylinder" and dims["half_height"] > 0.1:
            q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), 0.3), LYING)
        x, y = -1.5 + 0.33 * i, rng.uniform(-1.0, 1.0)
        lo, _hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
        bodies.append((kind, dims, _f32_pose(np.r_[x, y, top - lo[2] - 0.002, q])))
    bodies.append(("box", {"half": (0.9, 0.05, 0.02)}, _f32_pose([0.0, 1.4, top + 0.02 + 0.002, 0.0, 0.0, 0.005, 1.0])))
    # windows of more than the one-thread kernels' 64 vertices, which the tiled kernels walk
    for kind, dims in (("box", {"half": (0.5, 0.4, 0.03)}), ("ellipsoid", {"semi": (0.5, 0.4, 0.06)})):
        lo, _hi = ref.make_core(kind, dims, np.r_[0.0, -1.2, 0.0, 0.0, 0.0, 0.0, 1.0]).aabb()
        x = -0.9 if kind == "box" else 0.9
        bodies.append((kind, dims, _f32_pose([x, -1.2, top - lo[2] - 0.002, 0.0, 0.0, 0.0, 1.0])))
    return su.solid_scene(heights, 2.0, 2.0, bodies, device=device)


def test_solid_shadow_runs_under_graph_capture(test, device):
    """Capture the solid classes' shadow with the collide in a CUDA graph and replay it to an uncaptured collide's rows.

    Every solid class runs, with both lane classes of the pair table; a box and an ellipsoid whose windows exceed the
    one-thread kernels' 64 vertices go to the tiled kernels (the queue and its count the capture zeroes, the tiles'
    mark sums), and a long box defers to its class's full kernel, whose grid reads the deferred count. The shadow's launches
    are also captured alone with the device's memory pool off, so that an allocation inside them raises instead of
    becoming a graph allocation node.
    """
    model, _hfield, _shapes = _capture_scene(device)
    pipeline = tu.shadow_pipeline(model, classes="all", **PIPELINE)
    shadow = pipeline._terrain_law_shadow
    table = shadow.table
    for partner_class in (TERRAIN_PARTNER_BOX, TERRAIN_PARTNER_CYLINDER, TERRAIN_PARTNER_HULL):
        for lane in (TERRAIN_LANE_THREAD, TERRAIN_LANE_TILE):
            begin, end = table.segment(partner_class, lane)
            test.assertGreater(end - begin, 0, msg=(partner_class, lane))
    state = model.state()
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    expected = shadow.rows()
    test.assertTrue(np.all(expected["count"] > 0), msg=expected["count"])
    # the long box's rows exceed its walk slots: the full kernel writes them to the spill region
    test.assertGreaterEqual(shadow.rows_spilled(), int(expected["count"].max()))
    test.assertTrue(np.all(expected["state"] & 48 == 0))
    # the tiled kernels walk the windows over 64 vertices (a box and an ellipsoid)
    test.assertGreater(len(_queued(shadow, TERRAIN_PARTNER_BOX)), 0)
    test.assertGreater(len(_queued(shadow, TERRAIN_PARTNER_HULL)), 0)
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
    pipeline.collide(moved, contacts)
    test.assertEqual(int(shadow.rows()["count"].sum()), 0)
    pipeline.collide(state, contacts)
    shadow._out.count.zero_()
    wp.capture_launch(capture.graph)
    replayed = shadow.rows()
    for name in ("count", "state", "q", "n", "d", "feature"):
        np.testing.assert_array_equal(replayed[name], expected[name], err_msg=name)


def test_solid_launches_share_the_point_grids(test, device):
    """Launch every solid kernel on the shadow's shared mechanism, as the point class's: one block size, the one-thread
    kernel over the class, the tiled kernel on its work list's fixed grid, the full-finish kernel on the fixed grid of
    its scratch.

    Every solid class has pairs of both lane classes here, windows over the one-thread kernel's 64 vertices (a box and
    an ellipsoid) and a deferring box. The one-thread kernel queues exactly once each pair it leaves to the tiled
    kernel, every one a tiled-lane pair past the whole-pair rejects.
    """
    model, _hfield, _shapes = _capture_scene(device)
    pipeline = tu.shadow_pipeline(model, classes="all", **PIPELINE)
    shadow = pipeline._terrain_law_shadow
    table = shadow.table
    launched = {}
    launch, launch_tiled = wp.launch, wp.launch_tiled
    modules = {module.__name__ for module, *_names in _SOLID_KERNELS.values()}

    def recorder(original, how):
        def record(kernel, *args, **kwargs):
            if kernel.module.name in modules:
                launched.setdefault(kernel.key, []).append((how, kwargs["dim"], kwargs.get("block_dim")))
            return original(kernel, *args, **kwargs)

        return record

    with patch.object(wp, "launch", recorder(launch, "launch")):
        with patch.object(wp, "launch_tiled", recorder(launch_tiled, "tiled")):
            pipeline.collide(model.state(), pipeline.contacts())
    block = terrain_law_block_dim(device)
    rows = shadow.rows()
    exits = int(TERRAIN_PAIR_TILE_EXIT | TERRAIN_PAIR_STALE | TERRAIN_PAIR_FILTERED | TERRAIN_PAIR_OUTSIDE)
    exits |= int(TERRAIN_PAIR_UNSUPPORTED)
    for partner_class, (module, thread, tiled, full) in _SOLID_KERNELS.items():
        begin = table.segment(partner_class, TERRAIN_LANE_THREAD)[0]
        tiled_begin, end = table.segment(partner_class, TERRAIN_LANE_TILE)
        test.assertGreater(tiled_begin, begin)
        test.assertGreater(end, tiled_begin)
        test.assertEqual(launched[getattr(module, thread).key], [("launch", end - begin, block)])
        tiled_blocks = min(end - tiled_begin, TERRAIN_LAW_TILE_BLOCKS)
        test.assertEqual(launched[getattr(module, tiled).key], [("tiled", tiled_blocks, block)])
        full_blocks = shadow.describe()["launch_shapes"][module.__name__.rsplit("_", 1)[1]]["full_blocks"]
        test.assertEqual(launched[getattr(module, full).key], [("tiled", full_blocks, block)])
        queued = _queued(shadow, partner_class)
        test.assertEqual(len(np.unique(queued)), len(queued))
        test.assertTrue(np.all((queued >= tiled_begin) & (queued < end)), msg=queued)
        state = rows["state"][np.searchsorted(shadow._listed, queued)]
        test.assertFalse(np.any(state & exits), msg=state)
    test.assertGreater(len(_queued(shadow, TERRAIN_PARTNER_BOX)), 0)
    test.assertGreater(len(_queued(shadow, TERRAIN_PARTNER_HULL)), 0)
    test.assertGreater(int(shadow._out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), 0)


def test_solid_fixed_grids_take_each_pair_once(test, device):
    """Take every queued and every deferred solid pair once, even in a launch with more blocks than the grid.

    The solid tiled and full-finish kernels stride over their lists by the grids fixed in the work list and in the
    finish scratch (``work.blocks``, ``scratch.blocks``); a block past that grid returns at once. Long boxes lie across
    ridges: their windows exceed the one-thread kernel's vertices (queued) and their rows the box class's slots
    (deferred). Launched with three blocks more (and scratch room for them), the tiled kernel would otherwise take
    queued pairs twice, deferring them twice, and the full-finish kernel would finish deferred pairs twice, counting
    their row overflow twice.
    """
    _hf, model, _shapes, bodies = _ridge_boxes(device, (0.9, 0.05, 0.02), (-0.75, -0.45, -0.15, 0.15, 0.45, 0.75))
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    table = shadow.table
    begin = table.segment(TERRAIN_PARTNER_BOX, TERRAIN_LANE_THREAD)[0]
    tiled_begin, end = table.segment(TERRAIN_PARTNER_BOX, TERRAIN_LANE_TILE)
    test.assertEqual(end - tiled_begin, len(bodies))
    rows = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_BOX]
    inputs = _pair_inputs(model, pipeline)
    data = shadow.features.data()
    solids = terrain_law_solid.terrain_solid_data(model, table.pairs_host[:, 1])
    block = terrain_law_block_dim(device)
    blocks = 2
    results = []
    for extra in (0, 3):
        out = _side_buffer(begin, end - begin, rows, table.pair_count, device)
        # Room for pairs deferred more than once, so that blocks past the grid fail the counts below instead of
        # writing past the deferred list.
        out.deferred = wp.zeros(4 * table.pair_count, dtype=wp.int32, device=device)
        work = terrain_pair_work(device, tiled_begin, end - tiled_begin)
        work.blocks = blocks
        walk = [begin, 0, inputs, data, out, work, solids]
        wp.launch(terrain_law_box.terrain_box_rows, dim=end - begin, inputs=walk, block_dim=block, device=device)
        walk[:2] = [tiled_begin, (tiled_begin - begin) * rows]
        wp.launch_tiled(
            terrain_law_box.terrain_box_rows_tiled, dim=blocks + extra, inputs=walk, block_dim=block, device=device
        )
        deferred = int(out.deferred_count.numpy()[TERRAIN_PARTNER_BOX])
        # The deferred list in table order, so that the entries each block takes do not depend on the device.
        listed = out.deferred.numpy()
        listed[begin : begin + deferred] = np.sort(listed[begin : begin + deferred])
        out.deferred.assign(listed)
        scratch = terrain_finish_scratch(device, _full_candidates(shadow, TERRAIN_PARTNER_BOX), blocks=blocks + extra)
        scratch.blocks = blocks
        finish = [begin, 0, inputs, data, out, scratch, solids]
        wp.launch_tiled(
            terrain_law_box.terrain_box_rows_full, dim=blocks + extra, inputs=finish, block_dim=block, device=device
        )
        results.append(
            {
                "queued": int(work.count.numpy()[0]),
                "deferred": deferred,
                "overflow": out.overflow.numpy().tolist(),
                "spilled": int(out.spilled.numpy()[0]),
                **{name: getattr(out, name).numpy() for name in ("count", "state")},
                "rows": [_pair_rows(out, pair) for pair in range(tiled_begin, end)],
            }
        )
    grid, wider = results
    # Not vacuous: more pairs are queued and deferred than the grid has blocks, and every deferred pair has more rows
    # than its walk slots (spill rows).
    test.assertEqual(grid["queued"], len(bodies))
    test.assertEqual(grid["deferred"], len(bodies))
    test.assertGreater(grid["deferred"], blocks)
    test.assertEqual(grid["overflow"], [0, 0])
    test.assertTrue(np.all(grid["count"][tiled_begin:end] > rows))
    test.assertEqual(grid["spilled"], int(grid["count"][tiled_begin:end].sum()))
    for name in ("queued", "deferred", "overflow", "spilled"):
        test.assertEqual(wider[name], grid[name], msg=name)
    for name in ("count", "state"):
        np.testing.assert_array_equal(wider[name], grid[name], err_msg=name)
    for a, b in zip(wider["rows"], grid["rows"], strict=True):
        for name in ("q", "n", "d", "feature"):
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_shadow_keeps_contacts_bitwise_with_every_class(test, device):
    """Write today's contacts bit for bit with every class's kernels running beside the terrain route (the default
    route and the suite's)."""
    rng = np.random.default_rng(41)
    heights = np.round(rng.uniform(0.0, 0.1, (33, 33)) / 0.005) * 0.005
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
    bodies = _exact_poses(rng, hf, 3)
    model, _hfield, _shapes = su.solid_scene(heights, 1.6, 1.6, bodies, device=device)

    def rows(arrays):
        columns = [
            np.ascontiguousarray(a).view(np.int32).reshape(len(arrays["rigid_contact_shape0"]), -1)
            for a in arrays.values()
        ]
        stacked = np.concatenate(columns, axis=1)
        return stacked[np.lexsort(stacked.T[::-1])]

    for name, route in (("default", {}), ("suite", SUITE_ROUTE)):
        with test.subTest(route=name), patch.dict(os.environ, route):
            unset = {k: v for k, v in os.environ.items() if k != "NEWTON_TERRAIN_LAW_SHADOW"}
            with patch.dict(os.environ, unset, clear=True):
                today = newton.CollisionPipeline(model, **PIPELINE)
            shadow = tu.shadow_pipeline(model, classes="all", **PIPELINE)
            state = model.state()
            contacts = []
            for pipeline in (today, shadow):
                c = pipeline.contacts()
                pipeline.collide(state, c)
                count = int(c.rigid_contact_count.numpy()[0])
                contacts.append(
                    {
                        key: getattr(c, key).numpy()[:count]
                        for key in (
                            "rigid_contact_shape0",
                            "rigid_contact_shape1",
                            "rigid_contact_point0",
                            "rigid_contact_point1",
                            "rigid_contact_normal",
                        )
                    }
                )
            test.assertGreater(len(contacts[0]["rigid_contact_shape0"]), 10)
            if device.is_cpu:
                for key in contacts[0]:
                    np.testing.assert_array_equal(contacts[0][key], contacts[1][key], err_msg=key)
            else:
                np.testing.assert_array_equal(rows(contacts[0]), rows(contacts[1]))
            test.assertGreater(int(shadow._terrain_law_shadow.rows()["count"].sum()), 0)


class TestTerrainLawSolid(unittest.TestCase):
    def test_today_route_is_not_minimal_for_solids_cpu(self):
        """Duplicate, tilt or drop today's rows of solids at flat seams (the defect the law removes)."""
        counts = {}
        for cell, model, hfield, shapes, expected in _flat_scenes("cpu"):
            _pipeline, contacts, state = _collide(model, shadow=False)
            rows = tu.contact_rows(model, state, contacts, hfield)
            counts[cell] = len(_minimal_violations(rows, shapes, expected))
        model, hfield, shapes, expected = _vertex_m0_scene("cpu")
        _pipeline, contacts, state = _collide(model, shadow=False)
        counts["vertex_m0"] = len(
            _minimal_violations(tu.contact_rows(model, state, contacts, hfield), shapes, expected)
        )
        self.assertTrue(all(v > 5 for v in counts.values()), msg=counts)

    def test_today_route_misses_solid_supports_cpu(self):
        """Miss supports today: rows shallower than the exact separation or with normals outside the surface's cone,
        cylinders through an exposed cell left without their depth, buried lying cylinders without a contact."""
        found = {}
        for name, hf, model, hfield, shapes, bodies in _exact_scenes("cpu", 6):
            _pipeline, contacts, state = _collide(model, shadow=False)
            incomplete, ghosts = _exact_violations(hf, tu.contact_rows(model, state, contacts, hfield), shapes, bodies)
            found[f"{name} incomplete"] = len(incomplete)
            found[f"{name} ghosts"] = len(ghosts)
        _differ, _rowless, short = _penetration_violations(
            _lp_pairs(_lp_cases(np.random.default_rng(17), 40), "cpu", shadow=False)
        )
        found["exposed cells short"] = len(short)
        hf, model, hfield, shapes, bodies = _burial_scene("cpu")
        _pipeline, contacts, state = _collide(model, shadow=False)
        _differ, rowless, short = _penetration_violations(
            _burial_pairs(hf, tu.contact_rows(model, state, contacts, hfield), shapes, bodies)
        )
        found["buried rowless"] = len(rowless)
        found["buried short"] = len(short) + len(rowless)
        self.assertGreater(found["exposed cells short"], LP_SHORT, msg=found)
        self.assertGreater(found["buried rowless"], 5, msg=found)
        self.assertGreater(found["buried short"], 2 * BURIED_SHORT, msg=found)
        self.assertGreater(sum(v for k, v in found.items() if k.endswith("incomplete")), 5, msg=found)
        self.assertGreater(sum(v for k, v in found.items() if k.endswith("ghosts")), 5, msg=found)

    def test_today_route_fails_thin_rods_and_creases_cpu(self):
        """Lose thin rods' and crease bodies' supports today, and add ghost normals and duplicates there."""
        found = {}
        for name, scene in (("g_thin", _thin_rods), ("c_crease20", _crease_bodies)):
            hf, model, hfield, shapes, bodies = scene("cpu")
            _pipeline, contacts, state = _collide(model, shadow=False)
            violations = _support_violations(hf, tu.contact_rows(model, state, contacts, hfield), shapes, bodies)
            for key in ("duplicates", "ghosts", "incomplete"):
                found[f"{name} {key}"] = len(violations[key])
        self.assertGreater(found["g_thin incomplete"], 5, msg=found)
        self.assertGreater(found["c_crease20 incomplete"] + found["c_crease20 ghosts"], 5, msg=found)

    def test_barrel_cylinders_are_left_out_cpu(self):
        """Give a barrel cylinder (a hull-class shape with no core in the law yet) no row and say why."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=su.MARGIN, gap=su.GAP)
        terrain = newton.Heightfield(data=np.zeros((9, 9), np.float32), nrow=9, ncol=9, hx=0.4, hy=0.4)
        builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05)))
        builder.add_shape(body=body, type=newton.GeoType.CYLINDER, scale=wp.vec3(0.03, 0.04, 0.2), cfg=cfg)
        model = builder.finalize(device="cpu")
        pipeline, _contacts, _state = _collide(model, shadow=True)
        rows = pipeline._terrain_law_shadow.rows()
        self.assertEqual(int(rows["count"][0]), 0)
        self.assertEqual(int(rows["state"][0]), int(TERRAIN_PAIR_UNSUPPORTED))

    def test_solid_kernels_compile_without_contraction_cpu(self):
        """Compile the solid classes' kernels to PTX without fused multiply-adds (NVRTC, any host).

        Every float32 operation then rounds as on the CPU, so the CPU decides every boundary as the GPU does.
        """
        if 86 not in wp.get_cuda_supported_archs():
            self.skipTest("NVRTC cannot compile for sm_86 here")
        for module in (terrain_law_box, terrain_law_cylinder, terrain_law_hull):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                wp.compile_aot_module(module, arch=86, module_dir=directory, use_ptx=True)
                paths = glob.glob(os.path.join(directory, "*.ptx"))
                self.assertEqual(len(paths), 1)
                with open(paths[0], encoding="latin-1") as f:
                    ptx = f.read()
                self.assertEqual(ptx.count("fma.rn"), 0)
                self.assertGreater(ptx.count("mul.rn.f32"), 100)


devices = get_test_devices()
for _name, _func in (
    ("test_finish_in_place_equals_finish", test_finish_in_place_equals_finish),
    ("test_solid_rows_are_minimal_at_flat_seams", test_solid_rows_are_minimal_at_flat_seams),
    (
        "test_solid_rows_are_minimal_on_a_vertex_without_margins",
        test_solid_rows_are_minimal_on_a_vertex_without_margins,
    ),
    ("test_solid_rows_are_complete_and_free_of_ghosts", test_solid_rows_are_complete_and_free_of_ghosts),
    ("test_thin_rods_and_crease_bodies_keep_their_supports", test_thin_rods_and_crease_bodies_keep_their_supports),
    ("test_exposed_cell_rows_follow_the_law", test_exposed_cell_rows_follow_the_law),
    ("test_buried_solids_follow_the_law", test_buried_solids_follow_the_law),
    ("test_recovery_rows_have_distinct_feature_ids", test_recovery_rows_have_distinct_feature_ids),
    ("test_solid_rows_match_the_float64_law", test_solid_rows_match_the_float64_law),
    ("test_tiled_kernels_match_one_thread_kernels", test_tiled_kernels_match_one_thread_kernels),
    ("test_long_box_keeps_every_support", test_long_box_keeps_every_support),
    ("test_deferred_pairs_keep_their_rows_beyond_the_grid", test_deferred_pairs_keep_their_rows_beyond_the_grid),
    ("test_wide_box_keeps_every_support", test_wide_box_keeps_every_support),
    ("test_window_past_its_bound_is_flagged", test_window_past_its_bound_is_flagged),
    ("test_solid_launches_share_the_point_grids", test_solid_launches_share_the_point_grids),
    ("test_solid_fixed_grids_take_each_pair_once", test_solid_fixed_grids_take_each_pair_once),
    ("test_shadow_keeps_contacts_bitwise_with_every_class", test_shadow_keeps_contacts_bitwise_with_every_class),
):
    add_function_test(TestTerrainLawSolid, _name, _func, devices=devices)
add_function_test(
    TestTerrainLawSolid,
    "test_solid_shadow_runs_under_graph_capture",
    test_solid_shadow_runs_under_graph_capture,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

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
    TERRAIN_PARTNER_SCAN_RUN,
    TERRAIN_PARTNER_WALK_SLOTS,
    TerrainFeatures,
)
from newton._src.geometry.terrain_law import (
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_FINISH_MASK_ROWS,
    TERRAIN_PAIR_CANDIDATE_OVERFLOW,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_FULL,
    TERRAIN_PAIR_INCOMPLETE,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_ROW_OVERFLOW,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_UNSUPPORTED,
    TerrainFeatureData,
    TerrainFinishScratch,
    TerrainLawRows,
    TerrainPairInputs,
    terrain_eps,
    terrain_finish_list,
    terrain_finish_scratch,
    terrain_finish_slots,
    terrain_hold_row,
    terrain_law_block_dim,
    terrain_law_rows,
    terrain_planes,
    terrain_window,
    terrain_window_point,
)
from newton._src.geometry.terrain_law_cores import TerrainBoxCore, TerrainCylinderCore
from newton._src.geometry.terrain_law_solid import TERRAIN_SOLID_LAUNCHES, solid_recovery, solid_sink
from newton._src.geometry.terrain_shadow import TERRAIN_LAW_JOB_NAMES
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
    keep, kept = terrain_finish_slots(q, n, d, base, prefixes[i], eps[i], 0, 1, False)
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


@wp.kernel(enable_backward=False)
def _finish_held(
    q: wp.array[wp.vec3],
    n: wp.array[wp.vec3],
    d: wp.array[float],
    counts: wp.array[int],
    eps: wp.array[float],
    out: TerrainLawRows,
    merged: wp.array2d[int],
    skipped: wp.array2d[int],
):
    """Per list of rows at ``40 i``: the rows held one by one as a pair holds its candidates (``terrain_hold_row``,
    into ``out``'s slots from ``40 i``), then the in-place finish of the held rows with its coincidence step run and
    skipped, each as per-row keep flags, the held count and the kept count last."""
    i = wp.tid()
    base = i * 40
    held = int(0)
    for j in range(counts[i]):
        held = terrain_hold_row(out, base, held, q[base + j], n[base + j], d[base + j], j, eps[i], 32)
    held = wp.min(held, 32)
    keep, kept = terrain_finish_slots(out.q, out.n, out.d, base, held, eps[i], 0, 1, False)
    for j in range(held):
        merged[i, j] = (keep >> j) & 1
    merged[i, 39] = held
    merged[i, 40] = kept
    keep, kept = terrain_finish_slots(out.q, out.n, out.d, base, held, eps[i], 0, 1, True)
    for j in range(held):
        skipped[i, j] = (keep >> j) & 1
    skipped[i, 39] = held
    skipped[i, 40] = kept


def test_held_rows_need_no_coincidence_step(test, device):
    """Keep the same rows, bit for bit, finishing held rows with the finish's coincidence step skipped (``distinct``,
    as every solid pair's finish runs) as with it run: rows held with ``terrain_hold_row`` at the finish's ``eps``
    include no coincident pair, its test dropping each row that coincides with a held one.

    The finish's candidate lists (coincident rows, plane groups, implied rows, groups of more than four rows), held in
    list order first.
    """
    rng = np.random.default_rng(59)
    lists = _candidate_lists(rng, 400)
    q = np.zeros((len(lists) * 40, 3), np.float32)
    n = np.zeros((len(lists) * 40, 3), np.float32)
    d = np.zeros(len(lists) * 40, np.float32)
    counts = np.zeros(len(lists), np.int32)
    for i, (rows, _prefix) in enumerate(lists):
        counts[i] = len(rows)
        for j, (qq, nn, dd) in enumerate(rows):
            q[i * 40 + j] = qq
            n[i * 40 + j] = nn
            d[i * 40 + j] = dd
    eps = np.full(len(lists), 8.0 * 1.1920928955078125e-7 * 0.2, np.float32)
    out = terrain_law_rows(device, len(lists), len(lists) * 40, 0, [0] * 4)
    merged = wp.zeros((len(lists), 41), dtype=int, device=device)
    skipped = wp.zeros((len(lists), 41), dtype=int, device=device)
    wp.launch(
        _finish_held,
        dim=len(lists),
        inputs=[
            wp.array(q, dtype=wp.vec3, device=device),
            wp.array(n, dtype=wp.vec3, device=device),
            wp.array(d, dtype=float, device=device),
            wp.array(counts, dtype=int, device=device),
            wp.array(eps, dtype=float, device=device),
            out,
        ],
        outputs=[merged, skipped],
        device=device,
    )
    a = merged.numpy()
    b = skipped.numpy()
    np.testing.assert_array_equal(a, b)
    # the hold merged coincident rows, and the finish dropped rows of the held ones
    test.assertGreater(int(np.sum(a[:, 39] < counts)), 30)
    test.assertGreater(int(np.sum(a[:, 40] < a[:, 39])), 100)


@wp.kernel(enable_backward=False)
def _finish_lanes(
    q: wp.array[wp.vec3],
    n: wp.array[wp.vec3],
    d: wp.array[float],
    prefixes: wp.array[int],
    eps: wp.array[float],
    one_lane: wp.array2d[int],
    lanes: wp.array2d[int],
):
    """Per list of rows at ``40 i``, one block: ``terrain_finish_slots`` by the first lane alone (``0, 1``), then by
    the block's lanes together, each as per-row keep flags and the kept count last."""
    i, lane = wp.tid()
    base = i * 40
    if lane == 0:
        keep, kept = terrain_finish_slots(q, n, d, base, prefixes[i], eps[i], 0, 1, False)
        for j in range(prefixes[i]):
            one_lane[i, j] = (keep >> j) & 1
        one_lane[i, 40] = kept
    keep, kept = terrain_finish_slots(q, n, d, base, prefixes[i], eps[i], lane, wp.block_dim(), False)
    if lane == 0:
        for j in range(prefixes[i]):
            lanes[i, j] = (keep >> j) & 1
        lanes[i, 40] = kept


def _large_group_lists(rng, count):
    """Candidate lists dominated by one plane group of 9-24 rows, as a flat sole's rows with its crossings at creases
    give: corners of a rectangle, points on its edges and inside, most on the sole's plane (implied when inside), some
    deeper or shallower, plus a few rows of other groups."""
    lists = []
    for _i in range(count):
        rows = []
        normal = rng.normal(size=3) * np.array([0.2, 0.2, 1.0])
        normal[2] = abs(normal[2]) + 0.5
        normal /= np.linalg.norm(normal)
        e1 = np.cross(normal, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        origin = rng.uniform(-0.05, 0.05, size=3)
        tilt = rng.uniform(-0.05, 0.05, size=2) * rng.choice([0.0, 1.0])
        half = rng.uniform(0.02, 0.08, size=2)
        size = int(rng.integers(9, 25))
        for j in range(size):
            if j < 4:
                u, v = (-half[0] if j in (0, 3) else half[0]), (-half[1] if j < 2 else half[1])
            elif rng.uniform() < 0.5:
                side = int(rng.integers(0, 4))
                t = rng.uniform()
                corners = [(-half[0], -half[1]), (half[0], -half[1]), (half[0], half[1]), (-half[0], half[1])]
                a, b = corners[side], corners[(side + 1) % 4]
                u, v = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            else:
                u, v = rng.uniform(-1.0, 1.0, size=2) * half
            dist = 0.01 + tilt[0] * u + tilt[1] * v
            if rng.uniform() < 0.2:
                dist += rng.uniform(-0.002, 0.003)
            rows.append((origin + u * e1 + v * e2, normal, dist))
        for _k in range(int(rng.integers(0, 5))):
            other = rng.normal(size=3)
            other[2] = abs(other[2]) + 0.3
            other /= np.linalg.norm(other)
            rows.append((rng.uniform(-0.05, 0.05, size=3), other, float(rng.uniform(0.0, 0.02))))
        order = rng.permutation(len(rows))
        rows = [rows[k] for k in order][:TERRAIN_FINISH_MASK_ROWS]
        lists.append((rows, len(rows)))
    return lists


def test_finish_lanes_keep_one_lanes_rows(test, device):
    """Keep the same rows, bit for bit, with a block's lanes splitting the finish's implied-row tests as with one lane
    taking them all: the lanes OR their shares, each test the same float32 operations.

    The candidate lists of :func:`test_finish_in_place_equals_finish` (up to the masks' rows) and lists dominated by one
    plane group of 9-24 rows, whose implied-row tests cost the most. On the CPU a block is one lane.
    """
    rng = np.random.default_rng(47)
    lists = [
        entry
        for entry in _candidate_lists(rng, 300)
        if entry[1] == len(entry[0]) and entry[1] <= TERRAIN_FINISH_MASK_ROWS
    ]
    lists += _large_group_lists(rng, 300)
    q = np.zeros((len(lists) * 40, 3), np.float32)
    n = np.zeros((len(lists) * 40, 3), np.float32)
    d = np.zeros(len(lists) * 40, np.float32)
    prefixes = np.zeros(len(lists), np.int32)
    for i, (rows, prefix) in enumerate(lists):
        prefixes[i] = prefix
        for j, (qq, nn, dd) in enumerate(rows):
            q[i * 40 + j] = qq
            n[i * 40 + j] = nn
            d[i * 40 + j] = dd
    test.assertTrue(np.all(prefixes <= TERRAIN_FINISH_MASK_ROWS))
    eps = np.full(len(lists), 8.0 * 1.1920928955078125e-7 * 0.2, np.float32)
    one_lane = wp.zeros((len(lists), 41), dtype=int, device=device)
    lanes = wp.zeros((len(lists), 41), dtype=int, device=device)
    wp.launch_tiled(
        _finish_lanes,
        dim=len(lists),
        inputs=[
            wp.array(q, dtype=wp.vec3, device=device),
            wp.array(n, dtype=wp.vec3, device=device),
            wp.array(d, dtype=float, device=device),
            wp.array(prefixes, dtype=int, device=device),
            wp.array(eps, dtype=float, device=device),
        ],
        outputs=[one_lane, lanes],
        block_dim=terrain_law_block_dim(device),
        device=device,
    )
    a = one_lane.numpy()
    b = lanes.numpy()
    for i in range(len(lists)):
        np.testing.assert_array_equal(b[i, : prefixes[i]], a[i, : prefixes[i]], err_msg=f"list {i}")
        test.assertEqual(b[i, 40], a[i, 40], msg=f"list {i}")
    # the large groups lose rows to the implied-row tests and the group caps
    large = slice(len(lists) - 300, len(lists))
    test.assertGreater(int(np.sum(a[large, 40] < prefixes[large])), 250)


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


LP_SHORT = 0
"""Exposed-cell cases of 40 whose deepest row may be short of the core's depth: none of this seed's (the law leaves
ties short, 5 of 4,000 cases of the generator)."""
BURIED_SHORT = 0
"""Buried solids of 120 whose deepest row may be short of the core's depth: none of this seed's (the law leaves ties
short, 27 of 12,000 of the generators)."""


def test_exposed_cell_rows_follow_the_law(test, device):
    """Give every cylinder through an exposed cell the float64 law's rows and a row, and all but a few their depth.

    The exposed-cell cases of the cylinder query's exact LP reference (DESIGN 7, Track 1: any axis and size, steep
    faces, 60 m out), adapted per pair. The reference requires every triangle prism's row to reach the exact minimum
    over that prism (today's route misses 1,569 of 2,000 cases), which a per-pair law that emits one support set per
    pair, not one row per triangle, does not do by design; here a pair's deepest row must reach its core's depth.

    Rule X exits the core's deepest candidate point (its rims' points farthest from the valleys and vertices under
    them), so a pair reaches its depth unless its deepest point is a tie: equally far from two or more surface points,
    inside a cap or the side (DESIGN 1.3 and 8.5). Over 4,000 cases of this generator (100 seeds) the shadow rows
    equal the float64 law's in all but one case (a decision within float32 resolution, as without rule X) and leave 4
    short, all ties, by at most 0.45 mm, none rowless; without rule X the law left 74 short (by up to 161 mm). Today's
    route leaves 526 short. The check allows :data:`LP_SHORT` of its 40 (none of this seed's needs rule X; see
    :func:`test_exposed_cell_exits_reach_their_depth`).
    """
    pairs = _lp_pairs(_lp_cases(np.random.default_rng(17), 40), device, shadow=True)
    differ, rowless, short = _penetration_violations(pairs)
    test.assertEqual(differ, [])
    test.assertEqual(rowless, [])
    test.assertLessEqual(len(short), LP_SHORT, msg=short)


LP_EXITS = ((49, 7), (74, 33), (54, 28), (2, 12), (0, 23), (10, 0), (90, 4))
"""Exposed-cell cases (seed, index of :func:`_lp_cases`) whose deepest row only rule X gives, from 160 down to 5 mm
deeper than the other rules' rows. In the first the deepest point's closest surface point lies outside the pair's
window, two cell rows past it (a rod 166 mm deep under a 60 degree face, its reach 40 mm)."""


def test_exposed_cell_exits_reach_their_depth(test, device):
    """Give cylinders through an exposed cell whose depth only rule X reaches an exit row, the float64 law's rows and
    their depth.

    Deep rods through steep cells (:data:`LP_EXITS`), from the generator of
    :func:`test_exposed_cell_rows_follow_the_law`: each exit is the point law's over the cells within the deepest
    candidate's bound, inside the pair's window or not (the first case's lies outside it). Without rule X these pairs
    stop 5 to 160 mm short; with an exit searched only inside the window the first stops 0.3 mm short, on a farther
    minimum.
    """
    cases = []
    for seed, index in LP_EXITS:
        cases.append(_lp_cases(np.random.default_rng(seed), index + 1)[index])
    pairs = _lp_pairs(cases, device, shadow=True)
    differ, rowless, short = _penetration_violations(pairs)
    test.assertEqual(differ, [])
    test.assertEqual(rowless, [])
    test.assertEqual(short, [])
    for (seed, index), (got, law, _pose, _separation, _frame_tol) in zip(LP_EXITS, pairs, strict=True):
        with test.subTest(case=(seed, index)):
            test.assertEqual(sum(1 for g in got if su.is_exit_row(g)), 1)
            test.assertEqual(sum(1 for r in law if su.is_exit_row(r)), 1)


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

    The law reaches a pair's depth unless its deepest point is a tie (rule X; DESIGN 1.3 and 8.5). Over 4,000 of each
    (100 seeds) the shadow rows equal the float64 law's in every case, none rowless, and leave short 7 cylinders (by
    at most 3.4 mm), 14 cones (16 mm) and 6 ellipsoids (16 mm), all ties; without rule X the law left 32, 106 and 16
    short. Today's route leaves 2,088, 345 and 11 short. The check allows :data:`BURIED_SHORT` of its 120.
    """
    hf, model, _hfield, shapes, bodies = _burial_scene(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    differ, rowless, short = _penetration_violations(_burial_pairs(hf, tu.shadow_rows(pipeline), shapes, bodies))
    test.assertEqual(differ, [])
    test.assertEqual(rowless, [])
    test.assertLessEqual(len(short), BURIED_SHORT, msg=short)


def _valley_bodies(device):
    """Solids sunk into a V-valley (30 deg walls meeting along y at x = 0, 5 cm cells), each with its deepest point in
    the wedge below the valley's line, whose closest surface point it is: a lying cylinder and an ellipsoid along the
    valley with their axes 5 mm below its line, a box with an edge down along it 20 mm deep and a cone apex down
    10 mm deep. Rule F's supports there project across the valley onto the other wall."""
    th = np.radians(30.0)
    x = np.linspace(-1.0, 1.0, 41)
    heights = np.tile(np.abs(x) * np.tan(th), (65, 1))
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.6, device="cpu"))
    along = _axis_angle((1.0, 0.0, 0.0), -np.pi / 2.0)
    diamond = _quat_mul(_axis_angle((0.0, 1.0, 0.0), np.pi / 4.0), _axis_angle((0.0, 0.0, 1.0), np.pi / 2.0))
    bodies = [
        ("cylinder", TOE, _f32_pose(np.r_[0.0, -0.9137, -0.005, along])),
        ("ellipsoid", {"semi": (0.03, 0.07, 0.03)}, _f32_pose(np.r_[0.0, -0.3113, -0.005, 0.0, 0.0, 0.0, 1.0])),
        ("box", {"half": (0.08, 0.03, 0.03)}, _f32_pose(np.r_[0.0, 0.2911, 0.03 * np.sqrt(2.0) - 0.02, diamond])),
        ("cone", CONE, _f32_pose(np.r_[0.0, 0.8871, -0.01 + CONE["half_height"], 1.0, 0.0, 0.0, 0.0])),
    ]
    model, hfield, shapes = su.solid_scene(heights, 1.0, 1.6, bodies, device=device)
    return hf, model, hfield, shapes, bodies


def test_valley_cores_reach_their_depth(test, device):
    """Give solids sunk into a valley a row as deep as their deepest point, the float64 law's rows and no ghost.

    Each deepest point lies in the wedge below the valley's line and is closest to it (a cylinder's or a cone's rim,
    a box's corner, a cone's apex, an ellipsoid's support point): no rule but X reaches it. Without rule X their deepest
    rows were the walls' plane depths of their recovery points (rule R), short by up to 13% of the depth.
    """
    hf, model, _hfield, shapes, bodies = _valley_bodies(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        with test.subTest(kind=kind):
            got = rows.get(shape, [])
            law = su.law_rows(hf, kind, dims, pose)
            missing, extra, deepest = su.compare_solid_rows(law, got)
            test.assertEqual((missing, extra), ([], []))
            test.assertLess(deepest, 1.0e-5)
            separation = su.refined_separation(hf, ref.make_core(kind, dims, pose))
            test.assertLess(separation, -0.008)
            test.assertLessEqual(min(g[2] for g in got), separation + PENETRATING_TOL)
            exits = [g for g in got if su.is_exit_row(g)]
            test.assertEqual(len(exits), 1)
            # the exit's terrain point lies on the valley's line, its normal in the line's normal cone
            test.assertLessEqual(su.normal_cone_angle(hf, exits[0][0], exits[0][1]), GHOST_ANGLE)


NEEDLES = (
    ({"semi": (0.03, 0.025, 0.12)}, (-2.88e-05, 0.0017035, 0.05, -0.0035235, 0.0199778, -0.0137802, 0.9996992)),
    ({"semi": (0.02, 0.02, 0.1)}, (0.000511, -0.0001721, 0.04, -0.0067381, -0.0060868, 0.0250489, 0.999645)),
    ({"semi": (0.01, 0.01, 0.1)}, (0.0002699, 0.0012358, 0.03, -0.0199863, -0.0199853, 0.0342181, 0.9990146)),
    ({"semi": (0.015, 0.01, 0.15)}, (-0.0009437, -0.0014296, 0.08, 0.0191654, 0.0038809, 0.0439472, 0.9988425)),
    ({"semi": (0.02, 0.02, 0.1)}, (0.0, 0.0005, 0.04, 0.0, 0.0, 0.0, 1.0)),
)
"""Elongated ellipsoids standing in a one-vertex pit 50 mm deep (5 cm cells), their lower ends 10 to 20 mm below the
pit's vertex, which lies inside them."""


def _needle_bodies(device):
    """A flat terrain with one vertex 50 mm low and the needles of :data:`NEEDLES` standing in it."""
    heights = np.zeros((41, 41))
    heights[20, 20] = -0.05
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.0, device="cpu"))
    bodies = [("ellipsoid", dims, _f32_pose(np.asarray(pose, np.float64))) for dims, pose in NEEDLES]
    model, hfield, shapes = su.solid_scene(heights, 1.0, 1.0, bodies, device=device)
    return hf, model, hfield, shapes, bodies


def test_needle_cores_reach_their_depth(test, device):
    """Give elongated ellipsoids standing in a pit a row as deep as their lower ends, the float64 law's rows.

    The pit's vertex lies inside each needle, whose lower end is its deepest point, closest to that vertex alone. The
    end is a local maximum of the distance from the vertex, not the farthest point (the upper end, above the surface):
    rule X takes both, an ellipsoid's every local maximum of the distance from a terrain vertex. With the farthest point
    alone the law stopped 0.1 to 0.45 mm short, on a valley's support point.
    """
    hf, model, _hfield, shapes, bodies = _needle_bodies(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    pit = hf.V(20, 20)
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        with test.subTest(semi=dims["semi"]):
            got = rows.get(shape, [])
            law = su.law_rows(hf, kind, dims, pose)
            missing, extra, deepest = su.compare_solid_rows(law, got)
            test.assertEqual((missing, extra), ([], []))
            test.assertLess(deepest, 1.0e-5)
            separation = su.refined_separation(hf, ref.make_core(kind, dims, pose))
            test.assertLess(separation, -0.009)
            test.assertLessEqual(min(g[2] for g in got), separation + PENETRATING_TOL)
            exits = [g for g in got if su.is_exit_row(g)]
            test.assertEqual(len(exits), 1)
            # the lower end's exit is the pit's vertex
            test.assertLess(float(np.linalg.norm(exits[0][0] - pit)), 1.0e-6)
            test.assertLessEqual(su.normal_cone_angle(hf, exits[0][0], exits[0][1]), GHOST_ANGLE)


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


def test_hovering_core_skips_rule_r(test, device):
    """Leave a core without a candidate that clears its window's highest vertex without a row, not listed for rule R.

    A box hovers 2 cm beyond its reach over flat ground, next to a spike outside its window but in a fine mip block
    under it: the tile-max and fine exits keep the pair, and its scan lists nothing. Rule R can give it no row (every
    point of the core lies above the surface), so its close leaves it without one instead of listing it for rule R's
    kernel.
    """
    heights = np.zeros((33, 33))
    heights[3, 6] = 0.5
    lift = BOX["half"][2] + su.REACH + 0.02
    pose = _f32_pose([-1.3, -1.3, lift, 0.0, 0.0, 0.0, 1.0])
    model, _hfield, _shapes = su.solid_scene(heights, 1.6, 1.6, [("box", BOX, pose)], device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    counts = shadow.job_counts()["box"]
    # past the tile-max exit, with no candidate
    test.assertEqual(counts["live"], 1)
    test.assertEqual(counts["candidates"], 0)
    test.assertEqual(counts["recovery"], 0)
    test.assertEqual(int(shadow.rows()["count"].sum()), 0)


def test_hovering_core_leaves_at_the_fine_mip(test, device):
    """Reject a core whose window lies below it by more than its reach at the fine mip, though its tile rises past it.

    A box hovers 2 cm beyond its reach over flat ground; a spike in the same 8 x 8-cell tile but four cells from its
    window keeps the tile-max exit from taking it. The fine mip's 2 x 2-cell blocks under its window do: the pair
    never goes live (no scan, no close).
    """
    heights = np.zeros((33, 33))
    heights[7, 7] = 0.5
    lift = BOX["half"][2] + su.REACH + 0.02
    pose = _f32_pose([-1.3, -1.3, lift, 0.0, 0.0, 0.0, 1.0])
    model, _hfield, _shapes = su.solid_scene(heights, 1.6, 1.6, [("box", BOX, pose)], device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    test.assertEqual(shadow.job_counts()["box"]["live"], 0)
    test.assertEqual(int(shadow.rows()["count"].sum()), 0)


@wp.kernel(enable_backward=False)
def _recovery_rows(
    data: TerrainFeatureData,
    boxes: wp.array[TerrainBoxCore],
    cylinders: wp.array[TerrainCylinderCore],
    reference: wp.vec2i,
    out: TerrainLawRows,
    held: wp.array[int],
):
    """Rule R's rows of cores given in the heightfield frame, in the window of reference vertex ``reference``: block
    ``i`` takes box ``i``, then the cylinders, its lanes splitting the core's points as rule R's kernel does (one lane
    on the CPU), its rows held in the eight slots from ``8 i``."""
    i, lane = wp.tid()
    window = terrain_window(data, 0, reference[0], reference[1])
    window.eps = terrain_eps(1.0)
    planes = terrain_planes()
    sink = solid_sink(8 * i, 8, window.eps)
    if i < boxes.shape[0]:
        box = boxes[i]
        box.c = terrain_window_point(window, box.c)
        planes, sink = solid_recovery(box, data, window, planes, sink, out, lane, wp.block_dim())
    else:
        cylinder = cylinders[i - boxes.shape[0]]
        cylinder.c = terrain_window_point(window, cylinder.c)
        planes, sink = solid_recovery(cylinder, data, window, planes, sink, out, lane, wp.block_dim())
    if lane == 0:
        held[i] = sink.held


def test_recovery_picks_follow_depth_then_index(test, device):
    """Pick rule R's four deepest boundary points in (depth, index) order when their depths tie: the lowest indices
    first, rank by rank.

    A level box and an upright cylinder buried under flat terrain: the box's four bottom corners and the cylinder's 16
    bottom rim points lie equally deep, bit for bit in float32. Rule R's one-pass selection keeps the picks of four
    passes that each take the deepest point not yet taken, the first in index order on ties: the box's corners 0 to 3
    and the cylinder's rim points 16 to 19 (its bottom cap's first), each row's rank (its feature id's point slot)
    following the index. A selection that put a later point before equally deep kept ones would pick the cylinder's
    last four rim points and reverse the ranks. On CUDA the block's lanes split the points (the cylinder's 32 one per
    lane) and merge their picks by block minima of depth, then index: the same picks in the same ranks.
    """
    n = 17
    model = tu.heightfield_model(np.zeros((n, n)), 0.4, 0.4, device=device)
    features = TerrainFeatures(model)
    box = TerrainBoxCore()
    box.c = wp.vec3(0.013, -0.021, -0.07)
    box.rot = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    box.h = wp.vec3(0.08, 0.06, 0.04)
    cylinder = TerrainCylinderCore()
    cylinder.c = wp.vec3(-0.017, 0.029, -0.06)
    cylinder.a = wp.vec3(0.0, 0.0, 1.0)
    cylinder.r = 0.05
    cylinder.h = 0.04
    cylinder.cone = 0
    reference = (4, 4)
    out = terrain_law_rows(device, 2, 16, 0, [0] * 4)
    held = wp.zeros(2, dtype=int, device=device)
    wp.launch_tiled(
        _recovery_rows,
        dim=2,
        inputs=[
            features.data(),
            wp.array([box], dtype=TerrainBoxCore, device=device),
            wp.array([cylinder], dtype=TerrainCylinderCore, device=device),
            wp.vec2i(*reference),
        ],
        outputs=[out, held],
        block_dim=terrain_law_block_dim(device),
        device=device,
    )
    # the window frame's origin: the reference vertex (heights 0)
    origin = np.array([-0.4 + reference[1] * 0.05, -0.4 + reference[0] * 0.05, 0.0])
    angles = np.arange(16) * (2.0 * np.pi / 16.0)
    rim = np.stack([np.cos(angles), np.sin(angles)], axis=1) * cylinder.r
    corners = np.array([[(-1, 1)[(i >> k) & 1] * box.h[k] for k in range(3)] for i in range(8)])
    points = {
        "box": (np.array(box.c) + corners, 0),
        "cylinder": (
            np.concatenate([np.array(cylinder.c) + np.c_[rim, np.full(16, s * cylinder.h)] for s in (1, -1)]),
            1,
        ),
    }
    q, nn, d, feature = (getattr(out, name).numpy() for name in ("q", "n", "d", "feature"))
    for name, (local, i) in points.items():
        with test.subTest(core=name):
            test.assertEqual(int(held.numpy()[i]), 4)
            rows = slice(8 * i, 8 * i + 4)
            # the four rows' depths tie, bit for bit
            test.assertEqual(len(np.unique(d[rows])), 1, msg=d[rows])
            np.testing.assert_array_equal(nn[rows], np.tile([0.0, 0.0, 1.0], (4, 1)))
            np.testing.assert_array_equal((feature[rows] >> 5) & 3, int(TERRAIN_FEATURE_RECOVERY))
            # each row's boundary point (its core point, q + d n) and rank (its point slot)
            frame = local - origin
            core = q[rows].astype(np.float64) + d[rows, None].astype(np.float64) * nn[rows]
            gaps = np.linalg.norm(frame[None, :, :] - core[:, None, :], axis=2)
            picked = np.argmin(gaps, axis=1).tolist()
            test.assertLess(float(gaps.min(axis=1).max()), 1.0e-6)
            ranks = (feature[rows] & 7).tolist()
            # four passes, each the deepest point not yet taken, the first in index order on ties
            depth = local[:, 2]
            expected = sorted(range(len(depth)), key=lambda k: (depth[k], k))[:4]
            test.assertEqual([picked[ranks.index(rank)] for rank in range(4)], expected)


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
# the job pipeline, row overflow, unsupported shapes and the shadow
# ----------------------------------------------------------------------------------------------------------------


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


LARGE_WINDOWS = (
    ("box", {"half": (0.6, 0.3, 0.02)}),
    ("cylinder", {"radius": 0.05, "half_height": 0.6}),
    ("cone", {"radius": 0.4, "half_height": 0.05}),
    ("ellipsoid", {"semi": (0.6, 0.3, 0.05)}),
)
"""Solids whose windows hold over 256 vertices on 5 cm cells (their scans take several jobs)."""


def _window_scene(rng, device):
    """Solids whose windows exceed 16 cells (5 cm cells) over rough terrain, every class, and of every class one whose
    window exceeds 256 vertices (:data:`LARGE_WINDOWS`): ``(heightfield, model, shapes, bodies)``."""
    n = 49
    heights = np.round(rng.uniform(0.0, 0.08, (n, n)) / 0.005) * 0.005
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.2, 1.2, device="cpu"))
    bodies = []
    for i in range(36 + len(LARGE_WINDOWS)):
        kind, dims = _solid_kinds(rng)[i % 7] if i < 36 else LARGE_WINDOWS[i - 36]
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


_SOLID_MODULES = {
    TERRAIN_PARTNER_BOX: (terrain_law_box, "box"),
    TERRAIN_PARTNER_CYLINDER: (terrain_law_cylinder, "cylinder"),
    TERRAIN_PARTNER_HULL: (terrain_law_hull, "hull"),
}
_WHOLE_PAIR_EXITS = int(
    TERRAIN_PAIR_TILE_EXIT
    | TERRAIN_PAIR_STALE
    | TERRAIN_PAIR_FILTERED
    | TERRAIN_PAIR_OUTSIDE
    | TERRAIN_PAIR_UNSUPPORTED
)
"""States of a pair the whole-pair rejects take (no row)."""
_LIST_OF_KERNEL = {
    "scan": "scans",
    "faces": "faces",
    "edges": "edges",
    "thetas": "thetas",
    "vertices": "vertices",
    "close": "live",
    "recovery": "recover",
    "xitems": "xitems",
    "xvalleys": "xvalleys",
    "small": "small",
}
"""The list each job kernel that first takes a list strides over
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs`; the creases kernel takes the thetas kernel's list
entry by entry, its results kept per entry, and the descents kernel rule X's candidates by their slots in the pool)."""
_COUNTER_OF_LIST = {name: k for k, name in enumerate(TERRAIN_LAW_JOB_NAMES)}
_COUNTER_OF_LIST.update(
    {"live": TERRAIN_LAW_JOB_NAMES.index("live"), "recover": TERRAIN_LAW_JOB_NAMES.index("recovery")}
)


def _kernel(partner_class, name):
    """A solid class's kernel ``name`` of :data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES`."""
    module, class_name = _SOLID_MODULES[partner_class]
    return getattr(module, f"terrain_{class_name}_rows" + ("" if name == "rows" else f"_{name}"))


def _class_pairs(table, partner_class) -> tuple[int, int]:
    """A solid class's table pairs ``[begin, end)`` (both lane classes)."""
    return table.segment(partner_class, TERRAIN_LANE_THREAD)[0], table.segment(partner_class, TERRAIN_LANE_TILE)[1]


def _class_jobs(model, shadow, partner_class, *, threads=None, **capacities):
    """A solid class's jobs, sized as the shadow sizes them but for ``capacities`` (entries per list), with ``threads``
    job threads (by default the shadow's)."""
    table = shadow.table
    begin, end = _class_pairs(table, partner_class)
    points, takes = terrain_law_solid.terrain_solid_exit_items(model, table.pairs_host[begin:end, 1])
    sizes = terrain_law_solid.terrain_solid_job_capacities(
        table.vertices[begin:end],
        points,
        takes,
        table.candidates[begin:end],
        int(TERRAIN_PARTNER_WALK_SLOTS[partner_class]),
        scan_run=int(TERRAIN_PARTNER_SCAN_RUN[partner_class]),
    )
    sizes.update(capacities)
    jobs = terrain_law_solid.terrain_solid_jobs(model.device, partner_class, begin, end - begin, sizes)
    if threads is not None:
        jobs.threads = int(threads)
        jobs.pair_threads = min(int(threads), end - begin)
    return jobs


def _reverse_list(jobs, name):
    """Reverse the filled entries of job list ``name`` (the order its kernel takes them in)."""
    entries = getattr(jobs, name)
    count = min(int(jobs.counts.numpy()[_COUNTER_OF_LIST[name]]), entries.shape[0])
    values = entries.numpy()
    values[:count] = values[:count][::-1].copy()
    entries.assign(values)


def _run_class(
    pipeline,
    model,
    partner_class,
    *,
    jobs=None,
    reverse=False,
    until=None,
    full=False,
    blocks=None,
    extra_blocks=0,
    extra_threads=0,
    candidates=None,
):
    """Run a solid class's launches (:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES`) for a
    shadow pipeline's last collide into a side buffer of their own (:func:`_side_buffer`), with ``jobs`` (by default
    sized as the shadow's), every list reversed before its kernel takes it with ``reverse``, up to the launch ``until``
    (not run). With ``full``, every pair the shadow did not reject whole goes straight to the full-finish kernel, which
    walks it (as a pair some of whose work found no room). The full-finish kernel's grid holds ``blocks`` blocks (by default the shadow's) of ``candidates``
    candidates (by default the class's bound), launched with ``extra_blocks`` more; the job kernels are launched with
    ``extra_threads`` threads more than the jobs' grid. Returns ``(out, jobs, solids)``."""
    shadow = pipeline._terrain_law_shadow
    table = shadow.table
    device = model.device
    begin, end = _class_pairs(table, partner_class)
    if jobs is None:
        jobs = _class_jobs(model, shadow, partner_class)
    solids = terrain_law_solid.terrain_solid_data(model, table.pairs_host[:, 1], pairs=table.pair_count)
    out = _side_buffer(begin, end - begin, TERRAIN_PARTNER_WALK_SLOTS[partner_class], table.pair_count, device)
    capacity = _full_candidates(shadow, partner_class) if candidates is None else candidates
    scratch = terrain_finish_scratch(device, capacity, blocks=blocks)
    grid = scratch.blocks
    if extra_blocks:
        # room for the extra blocks, which must return at once
        scratch = terrain_finish_scratch(device, capacity, blocks=grid + extra_blocks)
        scratch.blocks = grid
    args = [begin, 0, _pair_inputs(model, pipeline), shadow.features.data(), out]
    block = terrain_law_block_dim(device)
    launches = TERRAIN_SOLID_LAUNCHES
    if full:
        walked = np.arange(begin, end, dtype=np.int32)
        walked = walked[(shadow._out.state.numpy()[begin:end] & _WHOLE_PAIR_EXITS) == 0]
        deferred = out.deferred.numpy()
        deferred[begin : begin + len(walked)] = walked
        out.deferred.assign(deferred)
        counts = out.deferred_count.numpy()
        counts[partner_class] = len(walked)
        out.deferred_count.assign(counts)
        state = out.state.numpy()
        state[walked] = int(TERRAIN_PAIR_INCOMPLETE | TERRAIN_PAIR_FULL)
        out.state.assign(state)
        launches = launches[-1:]
    for kind, name in launches:
        if name == until:
            break
        if reverse and name in _LIST_OF_KERNEL:
            _reverse_list(jobs, _LIST_OF_KERNEL[name])
        kernel = _kernel(partner_class, name)
        if kind == "full":
            inputs = [*args, scratch, solids, jobs]
            wp.launch_tiled(kernel, dim=grid + extra_blocks, inputs=inputs, block_dim=block, device=device)
        elif kind == "pair_tiles":
            dim = jobs.finish_blocks + extra_blocks
            wp.launch_tiled(kernel, dim=dim, inputs=[*args, jobs, solids], block_dim=block, device=device)
        else:
            dim = {"pairs": end - begin, "jobs": jobs.threads, "pair_jobs": jobs.pair_threads}[kind]
            if kind != "pairs":
                dim += extra_threads
            wp.launch(kernel, dim=dim, inputs=[*args, jobs, solids], block_dim=block, device=device)
    return out, jobs, solids


def _assert_same_rows(test, a, b, pairs, what):
    """Assert two side buffers hold the same rows, bit for bit, for table pairs ``pairs``, wherever each keeps them."""
    for name in ("count", "state"):
        np.testing.assert_array_equal(
            getattr(a, name).numpy()[pairs], getattr(b, name).numpy()[pairs], err_msg=f"{what}: {name}"
        )
    for pair in pairs:
        x, y = _pair_rows(a, pair), _pair_rows(b, pair)
        for name in ("q", "n", "d", "feature"):
            np.testing.assert_array_equal(x[name], y[name], err_msg=f"{what}: pair {pair} {name}")


def _counts(jobs) -> dict:
    """A solid class's job counters by name (:data:`~newton._src.geometry.terrain_shadow.TERRAIN_LAW_JOB_NAMES`)."""
    return dict(zip(TERRAIN_LAW_JOB_NAMES, (int(v) for v in jobs.counts.numpy()), strict=True))


def test_job_pipeline_rows_are_the_walks(test, device):
    """Give a solid pair the rows its window's walk gives, bit for bit, whatever order its jobs run in.

    Solids of every class over rough terrain, among them of every class one whose window holds over 256 vertices (its
    scan takes several jobs, its class's :data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`
    window vertices each). Three arms per class: its launches as the shadow runs them; the same with one job thread and every
    list reversed before its kernel takes it (the scan's jobs, the features, the creases, the live pairs, the pairs left
    to rule R, rule X's items and valleys), so that the candidate pool holds every pair's rows in another order; and
    every pair walked again by the full-finish kernel alone, in window order. The close holds a pair's candidates by
    their place in window order (a cluster plane two scan jobs list is one offer), so the arms agree with each other and
    with the shadow's own rows.
    """
    hf, model, shapes, bodies = _window_scene(np.random.default_rng(37), device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    table = shadow.table
    shape_class = {int(p[1]): int(c) for p, c in zip(table.pairs_host, table.partner_class, strict=True)}
    large = {
        shape_class[shape] for shape, body in zip(shapes, bodies, strict=True) if _window_vertices(hf, *body) > 256
    }
    test.assertEqual(large, set(_SOLID_MODULES))
    walked = 0
    for partner_class in _SOLID_MODULES:
        with test.subTest(partner_class=partner_class):
            begin, end = _class_pairs(table, partner_class)
            pairs = np.arange(begin, end)
            jobs_out, jobs, _solids = _run_class(pipeline, model, partner_class)
            counts = _counts(jobs)
            # windows of several scan jobs; pairs the jobs finished (not deferred), some by rule X; a deferred pair is
            # finished from its candidates, not walked again
            test.assertGreater(counts["scans"], counts["live"])
            test.assertLess(int(jobs_out.deferred_count.numpy()[partner_class]), counts["live"])
            test.assertEqual(counts["walks"], 0)
            reversed_out, reversed_jobs, _solids = _run_class(
                pipeline, model, partner_class, jobs=_class_jobs(model, shadow, partner_class, threads=1), reverse=True
            )
            pool = counts["candidates"]
            test.assertEqual(_counts(reversed_jobs)["candidates"], pool)
            test.assertFalse(np.array_equal(jobs.key.numpy()[:pool], reversed_jobs.key.numpy()[:pool]))
            full_out, full_jobs, _solids = _run_class(pipeline, model, partner_class, full=True)
            test.assertEqual(_counts(full_jobs)["walks"], int(full_out.deferred_count.numpy()[partner_class]))
            _assert_same_rows(test, shadow._out, jobs_out, pairs, "the shadow's launch")
            _assert_same_rows(test, jobs_out, reversed_out, pairs, "the lists reversed")
            _assert_same_rows(test, jobs_out, full_out, pairs, "the full walk")
            walked += int(jobs_out.count.numpy()[begin:end].sum())
    test.assertGreater(walked, 20)


def _runs_of(entries, run_of):
    """The blocks of consecutive list entries that share a run (``run_of``: an entry's run key), as ``(run, items)``."""
    blocks = []
    for pair, item in entries.tolist():
        run = run_of(pair, item)
        if blocks and blocks[-1][0] == run:
            blocks[-1][1].append(item)
        else:
            blocks.append((run, [item]))
    return blocks


def test_job_lists_hold_each_run_together(test, device):
    """List a scan run's survivors, a pair's rule X items per run of window vertices and an item's valleys as one
    block of consecutive entries each, in window order, whatever order the threads append in.

    Each scan run, rule X's opening of each run of its pair's window vertices and each rule X item reserve their
    entries of a list with one atomic and write them in window order, so that a warp of a job kernel takes one pair's
    features (its record, core and heights). Solids of every class over rough terrain, of every class one whose window
    takes several scan runs: no run's entries are split or out of order in the faces, edges, vertices, rule X item and
    valley lists. (Appending entry by entry, the lanes of a CUDA warp interleave their runs' entries.)
    """
    _hf, model, _shapes, _bodies = _window_scene(np.random.default_rng(37), device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    item_run = terrain_law_solid.TERRAIN_SOLID_SCAN_VERTICES
    checked = {}
    for partner_class in _SOLID_MODULES:
        scan = int(TERRAIN_PARTNER_SCAN_RUN[partner_class])
        with test.subTest(partner_class=partner_class):
            _out, jobs, _solids = _run_class(pipeline, model, partner_class, until="xdescents")
            counts = _counts(jobs)
            lists = {
                # a scan run: its pair's window vertices [k, k + 1) times the run length (six items per vertex)
                "faces": lambda pair, item, scan=scan: (pair, item // 6 // scan),
                "edges": lambda pair, item, scan=scan: (pair, item // 6 // scan),
                "vertices": lambda pair, item, scan=scan: (pair, item // 6 // scan),
                # rule X's items: the core's points together, then each run of window vertices
                "xitems": lambda pair, item: (pair, -1 if item < 0 else item // item_run),
                # an item's valleys: window vertex times three plus the edge
                "xvalleys": lambda pair, item: (pair, item // 3),
            }
            for name, run_of in lists.items():
                entries = getattr(jobs, name).numpy()[: counts[name]]
                blocks = _runs_of(entries, run_of)
                runs = [run for run, _items in blocks]
                test.assertEqual(len(runs), len(set(runs)), msg=f"{name}: a run's entries are split")
                for run, items in blocks:
                    order = sorted(items, key=lambda item: (item >= 0, -item if item < 0 else item))
                    test.assertEqual(items, order, msg=f"{name}: run {run} out of window order")
                checked[name] = checked.get(name, 0) + len(blocks)
    # not vacuous: every list holds runs, a scan list runs from pairs of several runs
    test.assertTrue(all(count > 1 for count in checked.values()), msg=checked)


def _valley_large(device):
    """Long solids sunk along a V-valley (30 deg walls meeting along y at x = 0, 5 cm cells), each deepest along the
    valley's line, which only rule X reaches (axes 5 mm below it, a box's edge 20 mm): windows of over 64 vertices and
    over 256 (their scans take several jobs)."""
    th = np.radians(30.0)
    x = np.linspace(-1.0, 1.0, 41)
    heights = np.tile(np.abs(x) * np.tan(th), (65, 1))
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.6, device="cpu"))
    along = _axis_angle((1.0, 0.0, 0.0), -np.pi / 2.0)
    diamond = _quat_mul(_axis_angle((0.0, 1.0, 0.0), np.pi / 4.0), _axis_angle((0.0, 0.0, 1.0), np.pi / 2.0))
    bodies = [
        ("cylinder", {"radius": 0.03, "half_height": 0.5}, _f32_pose(np.r_[0.0, -0.6137, -0.005, along])),
        ("ellipsoid", {"semi": (0.03, 1.4, 0.03)}, _f32_pose(np.r_[0.0, 0.0113, -0.005, 0.0, 0.0, 0.0, 1.0])),
        ("ellipsoid", {"semi": (0.03, 0.6, 0.03)}, _f32_pose(np.r_[0.0, -0.3, -0.005, 0.0, 0.0, 0.0, 1.0])),
        ("box", {"half": (0.4, 0.03, 0.03)}, _f32_pose(np.r_[0.0, 0.5911, 0.03 * np.sqrt(2.0) - 0.02, diamond])),
    ]
    model, _hfield, shapes = su.solid_scene(heights, 1.0, 1.6, bodies, device=device)
    return hf, model, shapes, bodies


def test_rule_x_lists_give_the_full_passes(test, device):
    """Give a solid pair the same rule X row and rows, bit for bit, whether its exit kernel takes its listed candidates
    or the full-finish kernel walks it again and runs rule X over every candidate of its window.

    Long solids sunk along a valley, each deepest along its line (only an exit reaches it): a cylinder, ellipsoids and
    a box. Rule X's item and valley jobs list the candidates whose bounds pass the pair's target with their near
    bounds' descents, and the exit kernel runs the rule's passes over those alone (any other is a no-op in both
    passes). With room for one item and one valley, or for one listed candidate in rule X's region of the candidate
    pool, a pair's lists lack room and it is left to the full-finish kernel, which runs rule X over every candidate of
    its window; every pair walked by it alone gives the same rows too.
    """
    hf, model, shapes, bodies = _valley_large(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    table = shadow.table
    shape_class = {int(p[1]): int(c) for p, c in zip(table.pairs_host, table.partner_class, strict=True)}
    sizes = sorted(_window_vertices(hf, *body) for body in bodies)
    test.assertGreater(sizes[0], 64)
    test.assertGreater(sizes[-1], 256)
    exits = [g for shape in shapes for g in tu.shadow_rows(pipeline)[shape] if su.is_exit_row(g)]
    test.assertEqual(len(exits), len(bodies))
    for partner_class in sorted({shape_class[s] for s in shapes}):
        with test.subTest(partner_class=partner_class):
            begin, end = _class_pairs(table, partner_class)
            pairs = np.arange(begin, end)
            listed_out, listed, solids = _run_class(pipeline, model, partner_class)
            test.assertGreater(int(solids.exit_count.numpy()[partner_class]), 0)
            test.assertGreater(int(solids.exit_work.numpy()[0]), 0)
            # the listed passes: no pair's lists lacked room
            test.assertFalse(np.any(listed.xall.numpy()))
            heads = listed.xhead.numpy()
            test.assertTrue(np.any(heads >= 0))
            every_out, every, solids = _run_class(
                pipeline, model, partner_class, jobs=_class_jobs(model, shadow, partner_class, xitems=1, xvalleys=1)
            )
            test.assertTrue(np.any(every.xall.numpy()))
            test.assertGreater(int(every_out.deferred_count.numpy()[partner_class]), 0)
            test.assertGreater(int(solids.exit_work.numpy()[0]), 0)
            # one slot in rule X's region of the pool: the pairs listing more take every candidate too
            region_out, region, _solids = _run_class(
                pipeline, model, partner_class, jobs=_class_jobs(model, shadow, partner_class, xdescents=1)
            )
            test.assertGreater(_counts(region)["xdescents"], 1)
            test.assertTrue(np.any(region.xall.numpy()))
            full_out, _jobs, _solids = _run_class(pipeline, model, partner_class, full=True)
            _assert_same_rows(test, shadow._out, listed_out, pairs, "the shadow's launch")
            _assert_same_rows(test, listed_out, every_out, pairs, "lists without room")
            _assert_same_rows(test, listed_out, region_out, pairs, "rule X's region without room")
            _assert_same_rows(test, listed_out, full_out, pairs, "the full walk")
            features = np.concatenate([_pair_rows(listed_out, p)["feature"] for p in pairs])
            test.assertEqual(int(np.sum((features & 7) == 7)), sum(shape_class[s] == partner_class for s in shapes))


def test_exit_work_stays_bounded(test, device):
    """Search few exits for solids buried under a rough slope, over few cells, and keep the float64 law's rows.

    Ellipsoids of 0.3 x 0.3 x 0.1 m and lying cylinders of 0.6 m buried 0.2 to 0.3 m under a 0.6 slope (0.05 m cells,
    1 cm roughness): many candidates' closest surface points lie cells up the slope, outside the 3 x 3 cells around
    them. Rule X's bounds descend along the closest triangles' planes to them, the largest bound is searched first and
    each search covers the cells within its bound: here at most one exit search per pair, covering fewer vertices than
    the pairs' windows. A walk of the whole window per candidate whose 3 x 3 bound passes took 50 to 170 walks per pair.
    """
    rng = np.random.default_rng(3)
    xs = np.linspace(-2.0, 2.0, 81)
    _grid_x, grid_y = np.meshgrid(xs, xs)
    heights = 0.6 * (-grid_y) + rng.uniform(0.0, 0.01, grid_y.shape)
    hf = tu.reference_surface(tu.heightfield_model(heights, 2.0, 2.0, device="cpu"))
    bodies = []
    for kind, dims in (("ellipsoid", {"semi": (0.3, 0.3, 0.1)}), ("cylinder", {"radius": 0.05, "half_height": 0.3})):
        for _ in range(6):
            q = _random_quat(rng)
            if kind == "cylinder":
                q = _quat_mul(_axis_angle((0.0, 0.0, 1.0), rng.uniform(-0.2, 0.2)), LYING)
            x, y = rng.uniform(-0.5, 0.5, 2)
            bodies.append((kind, dims, _f32_pose(np.r_[x, y, float(hf.height(x, y)) - rng.uniform(0.01, 0.05), q])))
    model, _hfield, shapes = su.solid_scene(heights, 2.0, 2.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    searches, vertices = pipeline._terrain_law_shadow.exit_work()
    windows = sum(_window_vertices(hf, *body) for body in bodies)
    test.assertGreater(searches, 0)
    test.assertLessEqual(searches, len(bodies))
    test.assertLess(vertices, windows)
    rows = tu.shadow_rows(pipeline)
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        with test.subTest(kind=kind, pose=pose.tolist()):
            missing, extra, deepest = su.compare_solid_rows(su.law_rows(hf, kind, dims, pose), rows.get(shape, []))
            test.assertEqual((missing, extra), ([], []))
            test.assertLess(deepest, 1.0e-5)


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
    the box class's 16 walk slots. Its class's full-finish kernel finishes it from its candidates in the candidate pool,
    without walking its window again (every job of it found room), and writes them to the spill region. (With a
    per-class limit of 16 rows the law kept 16, each plane's deepest first, and raised ``terrain_rows``.)
    """
    hf, model, shapes, bodies = _ridge_boxes(device, (0.9, 0.05, 0.02), (0.05,))
    _kind, dims, pose = bodies[0]
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    rows = shadow.rows()
    test.assertEqual(int(shadow._out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), 1)
    test.assertEqual(shadow.job_counts()["box"]["walks"], 0)
    law = su.law_rows(hf, "box", dims, pose)
    test.assertEqual(len(law), 20)
    test.assertGreater(len(law), terrain_law_box.TERRAIN_BOX_WALK_SLOTS)
    test.assertEqual(int(rows["count"][0]), len(law))
    test.assertEqual(int(rows["state"][0]), 0)
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
    missing, extra, _deepest = su.compare_solid_rows(law, tu.shadow_rows(pipeline)[shapes[0]])
    test.assertEqual((missing, extra), ([], []))


def test_long_box_keeps_its_first_rows_without_room(test, device):
    """Keep a box's first rows by priority in its walk slots, flagged, when the spill region has no room for all.

    The long box's 20 rows exceed the box class's 16 walk slots, and a spill region of 19 rows cannot hold them: the
    full finish keeps 16 in the box's own walk slots, every plane group's deepest row first (the deepest first), then
    the groups' farthest ones, flags a row overflow and raises ``terrain_rows``. (Before, the pair kept no row.)
    """
    hf, model, shapes, bodies = _ridge_boxes(device, (0.9, 0.05, 0.02), (0.05,))
    _kind, dims, pose = bodies[0]
    capture = StdOutCapture()
    capture.begin()
    try:
        with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW_ROWS": "19"}):
            pipeline, _contacts, _state = _collide(model, shadow=True)
        # flush the device's printf before the capture ends
        wp.synchronize()
    finally:
        message = capture.end()
    test.assertIn("Terrain-law spill region overflowed 20 > 19", message)
    shadow = pipeline._terrain_law_shadow
    rows = shadow.rows()
    law = su.law_rows(hf, "box", dims, pose)
    test.assertEqual(len(law), 20)
    slots = terrain_law_box.TERRAIN_BOX_WALK_SLOTS
    test.assertEqual(int(rows["count"][0]), slots)
    test.assertEqual(int(rows["state"][0]), int(TERRAIN_PAIR_ROW_OVERFLOW))
    test.assertEqual(shadow.overflow_counts(), (1, 0, 0))
    test.assertEqual(shadow.rows_spilled(), 20)
    test.assertTrue(pipeline.narrow_phase.buffer_capacity_status()["terrain_rows"])
    got = tu.shadow_rows(pipeline)[shapes[0]]
    test.assertEqual(tu.missing_rows(got, law, q_tol=1.0e-5, angle_tol=1.0e-3, d_tol=1.0e-5), [])
    # every plane group of the law keeps its deepest row
    groups = {}
    for r in law:
        key = (*np.round(r.n, 4).tolist(), round(float(r.n @ r.q), 4))
        groups.setdefault(key, []).append(r)
    deepest = [min(group, key=lambda r: r.d) for group in groups.values()]
    test.assertLessEqual(len(deepest), slots)
    test.assertEqual(tu.missing_rows(deepest, got, q_tol=1.0e-5, angle_tol=1.0e-3, d_tol=1.0e-5), [])


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
    begin, end = _class_pairs(shadow.table, TERRAIN_PARTNER_BOX)
    test.assertEqual(end - begin, len(bodies))
    out, _jobs, _solids = _run_class(pipeline, model, TERRAIN_PARTNER_BOX, blocks=1)
    test.assertEqual(int(out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), len(bodies))
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
    (With a list of 32 candidates the law kept the first 32 in window order, then 16 rows, and flagged a candidate and
    a row overflow.)
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
    test.assertEqual(shadow.overflow_counts(), (0, 0, 0))
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
    missing, extra, _deepest = su.compare_solid_rows(law, tu.shadow_rows(pipeline)[shapes[0]])
    test.assertEqual((missing, extra), ([], []))


def test_large_plane_group_follows_the_law(test, device):
    """Give a box over a lattice of equal-height pillars the float64 law's rows: one plane group of over 100 rows.

    A flat box 2 mm above one-cell pillars every third cell (5 cm cells): every pillar top under it gives rule F rows
    in one plane, a plane group of 169 candidates, which the finish's implied-row step takes whole (no candidate count
    bounds it). The step runs in one pass, the shallowest row first, and each test stops at the first interpolation of
    the other rows that implies it: the same rows as testing again after every drop and taking the lowest
    interpolation, and on the CPU 0.01 s of a collide here instead of 0.12 s.
    """
    n = 61
    r = np.arange(n)
    heights = np.where((r[:, None] % 3 != 0) & (r[None, :] % 3 != 0), 0.05, 0.0)
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.5, 1.5, device="cpu"))
    bodies = [
        ("box", {"half": (half, half, 0.02)}, _f32_pose(np.r_[0.013, 0.007, 0.05 + 0.02 + 0.002, 0.0, 0.0, 0.0, 1.0]))
        for half in (0.2, 0.45)
    ]
    model, _hfield, shapes = su.solid_scene(heights, 1.5, 1.5, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        with test.subTest(half=dims["half"][0]):
            core = ref.make_core(kind, dims, pose)
            lo, hi = core.aabb()
            window = hf.window_features(hf.cells_in_box(lo[:2] - su.REACH, hi[:2] + su.REACH))
            candidates = ref.solid_law(hf, core, su.REACH, window)
            if dims["half"][0] > 0.4:
                test.assertGreater(len(candidates), 100)
            missing, extra, deepest = su.compare_solid_rows(su.law_rows(hf, kind, dims, pose), rows[shape])
            test.assertEqual((missing, extra), ([], []))
            test.assertLess(deepest, 1.0e-5)
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))


def test_plane_group_of_many_rows_takes_the_lanes(test, device):
    """Finish a pair holding many rows with a block's lanes, one holding few in a plane group of three with one thread of
    the small-pair finish kernel and one holding few others in place, all with the float64 law's rows.

    A flat box 2 mm above the pillar lattice of :func:`test_large_plane_group_follows_the_law`: 6 cm across it holds 15
    candidates in one plane group (:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LANE_ROWS` or more),
    which its close lists for the finish kernel and whose implied-row step drops 6; 3 cm across it holds 4 on four
    planes, which its close writes as they are; 2 cm across over one pillar's top it holds its 4 corners on that plane,
    which its close lists for the small-pair finish kernel.
    """
    n = 61
    r = np.arange(n)
    heights = np.where((r[:, None] % 3 != 0) & (r[None, :] % 3 != 0), 0.05, 0.0)
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.5, 1.5, device="cpu"))
    # (half size, centre, pairs for the lanes, pairs for the small-pair finish); a pillar's top spans 5 cm about 0.075
    for half, centre, lanes, small in (
        (0.06, (0.013, 0.007), 1, 0),
        (0.03, (0.013, 0.007), 0, 0),
        (0.01, (0.073, 0.077), 0, 1),
    ):
        with test.subTest(half=half):
            dims = {"half": (half, half, 0.02)}
            pose = _f32_pose(np.r_[centre[0], centre[1], 0.05 + 0.02 + 0.002, 0.0, 0.0, 0.0, 1.0])
            model, _hfield, shapes = su.solid_scene(heights, 1.5, 1.5, [("box", dims, pose)], device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            counts = pipeline._terrain_law_shadow.job_counts()["box"]
            test.assertEqual((counts["lanes"], counts["small"]), (lanes, small))
            rows = tu.shadow_rows(pipeline)[shapes[0]]
            if lanes:
                # the lanes' implied-row step dropped some of its candidates
                test.assertLess(len(rows), counts["candidates"])
            if small:
                test.assertEqual(len(rows), 4)
            missing, extra, deepest = su.compare_solid_rows(su.law_rows(hf, "box", dims, pose), rows)
            test.assertEqual((missing, extra), ([], []))
            test.assertLess(deepest, 1.0e-5)


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
    begin, _end = _class_pairs(shadow.table, TERRAIN_PARTNER_BOX)
    out, _jobs, _solids = _run_class(pipeline, model, TERRAIN_PARTNER_BOX, blocks=1, candidates=32)
    test.assertEqual(int(out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), 1)
    state = int(out.state.numpy()[begin])
    test.assertEqual(state & int(TERRAIN_PAIR_CANDIDATE_OVERFLOW), int(TERRAIN_PAIR_CANDIDATE_OVERFLOW))
    test.assertEqual(out.overflow.numpy().tolist(), [0, 1, 0])
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
    """Every solid kind on ridges of different heights, small and long, a long box whose rows overflow its slots (its
    class's full kernel finishes it), and a box and an ellipsoid whose windows take several scan jobs."""
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
    # windows of more than one scan job's vertices
    for kind, dims in (("box", {"half": (0.5, 0.4, 0.03)}), ("ellipsoid", {"semi": (0.5, 0.4, 0.06)})):
        lo, _hi = ref.make_core(kind, dims, np.r_[0.0, -1.2, 0.0, 0.0, 0.0, 0.0, 1.0]).aabb()
        x = -0.9 if kind == "box" else 0.9
        bodies.append((kind, dims, _f32_pose([x, -1.2, top - lo[2] - 0.002, 0.0, 0.0, 0.0, 1.0])))
    return su.solid_scene(heights, 2.0, 2.0, bodies, device=device)


def test_solid_shadow_runs_under_graph_capture(test, device):
    """Capture the solid classes' shadow with the collide in a CUDA graph and replay it, again and again, to an
    uncaptured collide's rows.

    Every solid class runs: windows of several scan jobs (a box and an ellipsoid), the job lists and the candidate pool
    (their counters the capture zeroes), pairs reaching below their windows' highest vertices through rule X's lists and
    exit kernels, and a long box deferred to its class's full kernel, whose grid reads the deferred count, and spilled.
    The spill region holds exactly the rows the scene asks for, so that a replay that did not zero the spill counters
    would find no room on its second run. The shadow's launches are also captured alone with the device's memory pool
    off, so that an allocation inside them raises instead of becoming a graph allocation node.
    """
    model, _hfield, _shapes = _capture_scene(device)
    probe = tu.shadow_pipeline(model, classes="all", **PIPELINE)
    probe.collide(model.state(), probe.contacts())
    need = probe._terrain_law_shadow.rows_spilled()
    test.assertGreater(need, 0)
    with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW_ROWS": str(need)}):
        pipeline = tu.shadow_pipeline(model, classes="all", **PIPELINE)
    shadow = pipeline._terrain_law_shadow
    test.assertEqual(shadow.spill_rows, need)
    state = model.state()
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    expected = shadow.rows()
    counts = shadow.job_counts()
    test.assertTrue(np.all(expected["count"] > 0), msg=expected["count"])
    # the long box's rows exceed its walk slots: the full kernel writes them to the spill region
    test.assertGreaterEqual(shadow.rows_spilled(), int(expected["count"].max()))
    test.assertTrue(np.all(expected["state"] & 48 == 0))
    # windows of several scan jobs (a box and an ellipsoid), and rule X's lists
    test.assertGreater(counts["box"]["scans"], counts["box"]["live"])
    test.assertGreater(counts["hull"]["scans"], counts["hull"]["live"])
    test.assertGreater(sum(c["xitems"] for c in counts.values()), 0)
    # Move the bodies, so that the replay must recompute the rows.
    moved = model.state()
    q = moved.body_q.numpy()
    q[:, 2] += 0.5
    moved.body_q.assign(q)
    pipeline.collide(moved, contacts)
    test.assertEqual(int(shadow.rows()["count"].sum()), 0)
    with scoped_capture_after_gc(device=device) as capture:
        pipeline.collide(state, contacts)
    for _replay in range(3):
        wp.capture_launch(capture.graph)
        replayed = shadow.rows()
        for name in ("count", "state", "q", "n", "d", "feature"):
            np.testing.assert_array_equal(replayed[name], expected[name], err_msg=name)
        test.assertEqual(shadow.rows_spilled(), need)
        test.assertEqual(shadow.overflow_counts(), (0, 0, 0))
        test.assertEqual(shadow.job_counts(), counts)
    test.assertFalse(any(pipeline.narrow_phase.buffer_capacity_status().values()))
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


def test_solid_launches_take_fixed_grids(test, device):
    """Launch every solid kernel with the dimensions fixed at construction, as the route record holds them: one block
    size; the whole-pair rejects one thread per pair of the class; the scan and the job kernels the class's job threads,
    each striding over its list, but for a job kernel whose list none of the class's pairs can fill (not launched: rule
    X's valleys of the box class); the close, rule R's kernel and rule X's exit kernel, which stride over lists of the
    class's pairs, the job threads up to one per pair; the finish kernel, blocks of lanes striding over the live pairs,
    up to one block per pair; the full-finish kernel the fixed grid of its scratch.

    Every solid class has pairs here, windows of several scan jobs (a box and an ellipsoid) and a deferring box. The
    live list holds every pair past the whole-pair rejects once.
    """
    model, _hfield, _shapes = _capture_scene(device)
    pipeline = tu.shadow_pipeline(model, classes="all", **PIPELINE)
    shadow = pipeline._terrain_law_shadow
    launched = {}
    launch, launch_tiled = wp.launch, wp.launch_tiled
    modules = {module.__name__ for module, _name in _SOLID_MODULES.values()}

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
    shapes = shadow.describe()["launch_shapes"]
    states = shadow._out.state.numpy()
    for partner_class, (_module, name) in _SOLID_MODULES.items():
        with test.subTest(partner_class=name):
            begin, end = _class_pairs(shadow.table, partner_class)
            jobs = shadow._jobs[partner_class]
            record = shapes[name]
            test.assertEqual(record["pair_dim"], end - begin)
            test.assertEqual(record["job_threads"], jobs.threads)
            test.assertEqual(record["pair_job_threads"], min(jobs.threads, end - begin))
            test.assertEqual(record["finish_blocks"], jobs.finish_blocks)
            test.assertLessEqual(jobs.finish_blocks, end - begin)
            test.assertEqual(record["skipped"], ["xvalleys"] if partner_class == TERRAIN_PARTNER_BOX else [])
            for kind, kernel_name in TERRAIN_SOLID_LAUNCHES:
                key = _kernel(partner_class, kernel_name).key
                if kernel_name in record["skipped"]:
                    test.assertNotIn(key, launched)
                    continue
                dim = {
                    "pairs": ("launch", end - begin),
                    "jobs": ("launch", jobs.threads),
                    "pair_jobs": ("launch", jobs.pair_threads),
                    "pair_tiles": ("tiled", jobs.finish_blocks),
                }.get(kind)
                if dim is None:
                    dim = ("tiled", record["full_blocks"])
                test.assertEqual(launched[key], [(*dim, block)], msg=key)
            counts = shadow.job_counts()[name]
            live = jobs.live.numpy()[: counts["live"]] + begin
            test.assertEqual(
                sorted(live.tolist()), (begin + np.flatnonzero((states[begin:end] & _WHOLE_PAIR_EXITS) == 0)).tolist()
            )
    test.assertGreater(shadow.job_counts()["box"]["scans"], shadow.job_counts()["box"]["live"])
    test.assertGreater(int(shadow._out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), 0)


def test_job_grids_take_each_entry_once(test, device):
    """Take every job and every deferred pair once, whatever the job kernels' grid, and with launches of more threads
    or blocks than their grids.

    The scan, job, close and rule R kernels stride over their lists by the class's job threads (``jobs.threads``, fixed
    at construction), a thread past them returning at once; the full-finish kernel strides over the deferred list by its
    scratch's grid (``scratch.blocks``), a block past it returning at once. Long boxes lie across ridges: their rows
    exceed the box class's walk slots (deferred, spilled). One job thread, three and the shadow's give the same rows, as
    do job kernels launched with three threads more than the jobs' grid, which would otherwise take entries twice
    (closing a pair twice), and a full-finish launch of three blocks more than its grid of two, which would otherwise
    finish deferred pairs twice, counting their spill rows twice.
    """
    _hf, model, _shapes, bodies = _ridge_boxes(device, (0.9, 0.05, 0.02), (-0.75, -0.45, -0.15, 0.15, 0.45, 0.75))
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    begin, end = _class_pairs(shadow.table, TERRAIN_PARTNER_BOX)
    test.assertEqual(end - begin, len(bodies))
    rows = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_BOX]
    results = []
    for threads, extra, extra_threads in ((None, 0, 0), (1, 0, 0), (3, 0, 0), (None, 3, 0), (2, 0, 3)):
        jobs = _class_jobs(model, shadow, TERRAIN_PARTNER_BOX, threads=threads)
        out, jobs, _solids = _run_class(
            pipeline, model, TERRAIN_PARTNER_BOX, jobs=jobs, blocks=2, extra_blocks=extra, extra_threads=extra_threads
        )
        results.append(
            {
                "counts": _counts(jobs),
                "deferred": int(out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]),
                "overflow": out.overflow.numpy().tolist(),
                "spilled": int(out.spilled.numpy()[0]),
                "out": out,
            }
        )
    grid = results[0]
    # Not vacuous: more pairs are deferred than the grid has blocks, and every deferred pair has more rows than its
    # walk slots (spill rows).
    test.assertEqual(grid["deferred"], len(bodies))
    test.assertEqual(grid["overflow"], [0, 0, 0])
    count = grid["out"].count.numpy()[begin:end]
    test.assertTrue(np.all(count > rows))
    test.assertEqual(grid["spilled"], int(count.sum()))
    for other in results[1:]:
        for name in ("counts", "deferred", "overflow", "spilled"):
            test.assertEqual(other[name], grid[name], msg=name)
        _assert_same_rows(test, grid["out"], other["out"], np.arange(begin, end), "another grid")


def test_job_overflow_takes_the_full_walk(test, device):
    """Keep every pair's rows, bit for bit, when its class's job lists, candidate pool or scan jobs have no room for its
    work, count those pairs, raise ``terrain_jobs`` and warn with the entries asked for, by class and list.

    ``NEWTON_TERRAIN_LAW_SHADOW_JOBS`` scales the solid classes' job lists and candidate pools, sized at the scene level
    from the pairs' own bounds: at 1e-6 each holds one entry. A pair whose work finds no room is left to its class's
    full-finish kernel, which walks it again; it is counted (the shadow's third overflow count) and raises the narrow
    phase's sticky ``terrain_jobs`` bit, which the capacity check raises on, as ``rigid_contact_max`` does for
    contacts, though no row is lost; the job check prints the entries asked for against the room. The scan's jobs,
    sized from every pair's bound on its window, lack room only for a window its shape data no longer bounds: with room
    for one, the other pairs are deferred from the whole-pair rejects on, before their scan.
    """
    model, _hfield, _shapes = _capture_scene(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    expected = pipeline._terrain_law_shadow.rows()
    capture = StdOutCapture()
    capture.begin()
    try:
        with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW_JOBS": "1e-6"}):
            small, _contacts, _state = _collide(model, shadow=True)
        # flush the device's printf before the capture ends
        wp.synchronize()
    finally:
        message = capture.end()
    shadow = small._terrain_law_shadow
    record = shadow.describe()
    test.assertEqual(record["job_scale"], 1.0e-6)
    for name in ("box", "cylinder", "hull"):
        capacities = record["launch_shapes"][name]["jobs"]
        lists = {k: v for k, v in capacities.items() if k != "scans"}
        # one entry each, but rule X's valleys of the box class, which none of its pairs can list (no room, no launch)
        test.assertEqual(lists, {k: int(name != "box" or k != "xvalleys") for k in lists})
        counts = shadow.job_counts()[name]
        test.assertGreater(counts["faces"], 1)
    test.assertIn("Terrain-law box class: its rule F face job list overflowed", message)
    test.assertIn("Terrain-law hull class: its rule E edge job list overflowed", message)
    rows = shadow.rows()
    for name in ("count", "state", "q", "n", "d", "feature"):
        np.testing.assert_array_equal(rows[name], expected[name], err_msg=name)
    test.assertGreater(int(shadow._out.deferred_count.numpy()[TERRAIN_PARTNER_CYLINDER]), 1)
    # the pairs that lost work are walked again, counted and flagged
    test.assertGreater(shadow.job_counts()["cylinder"]["walks"], 1)
    rows_lost, candidates_lost, jobs_lost = shadow.overflow_counts()
    test.assertEqual((rows_lost, candidates_lost), (0, 0))
    test.assertGreater(jobs_lost, 1)
    test.assertLessEqual(jobs_lost, sum(int(v) for v in shadow._out.deferred_count.numpy()[1:]))
    test.assertEqual([k for k, v in small.narrow_phase.buffer_capacity_status().items() if v], ["terrain_jobs"])
    with test.assertRaisesRegex(RuntimeError, "terrain_jobs"):
        small.narrow_phase.check_buffer_capacity()
    # the default lists have room for every pair's work
    pipeline.narrow_phase.check_buffer_capacity()
    test.assertEqual(pipeline._terrain_law_shadow.overflow_counts(), (0, 0, 0))
    # room for one scan job
    table = pipeline._terrain_law_shadow.table
    for partner_class in _SOLID_MODULES:
        with test.subTest(partner_class=partner_class):
            begin, end = _class_pairs(table, partner_class)
            jobs = _class_jobs(model, pipeline._terrain_law_shadow, partner_class, scans=1)
            out, jobs, _solids = _run_class(pipeline, model, partner_class, jobs=jobs)
            counts = _counts(jobs)
            test.assertGreater(counts["scans"], 1)
            test.assertGreaterEqual(int(out.deferred_count.numpy()[partner_class]), counts["live"] - 1)
            _assert_same_rows(test, pipeline._terrain_law_shadow._out, out, np.arange(begin, end), "one scan job")


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
    ("test_finish_lanes_keep_one_lanes_rows", test_finish_lanes_keep_one_lanes_rows),
    ("test_held_rows_need_no_coincidence_step", test_held_rows_need_no_coincidence_step),
    ("test_solid_rows_are_minimal_at_flat_seams", test_solid_rows_are_minimal_at_flat_seams),
    (
        "test_solid_rows_are_minimal_on_a_vertex_without_margins",
        test_solid_rows_are_minimal_on_a_vertex_without_margins,
    ),
    ("test_solid_rows_are_complete_and_free_of_ghosts", test_solid_rows_are_complete_and_free_of_ghosts),
    ("test_thin_rods_and_crease_bodies_keep_their_supports", test_thin_rods_and_crease_bodies_keep_their_supports),
    ("test_exposed_cell_rows_follow_the_law", test_exposed_cell_rows_follow_the_law),
    ("test_exposed_cell_exits_reach_their_depth", test_exposed_cell_exits_reach_their_depth),
    ("test_buried_solids_follow_the_law", test_buried_solids_follow_the_law),
    ("test_valley_cores_reach_their_depth", test_valley_cores_reach_their_depth),
    ("test_needle_cores_reach_their_depth", test_needle_cores_reach_their_depth),
    ("test_recovery_rows_have_distinct_feature_ids", test_recovery_rows_have_distinct_feature_ids),
    ("test_hovering_core_skips_rule_r", test_hovering_core_skips_rule_r),
    ("test_hovering_core_leaves_at_the_fine_mip", test_hovering_core_leaves_at_the_fine_mip),
    ("test_recovery_picks_follow_depth_then_index", test_recovery_picks_follow_depth_then_index),
    ("test_solid_rows_match_the_float64_law", test_solid_rows_match_the_float64_law),
    ("test_job_pipeline_rows_are_the_walks", test_job_pipeline_rows_are_the_walks),
    ("test_job_lists_hold_each_run_together", test_job_lists_hold_each_run_together),
    ("test_rule_x_lists_give_the_full_passes", test_rule_x_lists_give_the_full_passes),
    ("test_exit_work_stays_bounded", test_exit_work_stays_bounded),
    ("test_long_box_keeps_every_support", test_long_box_keeps_every_support),
    ("test_deferred_pairs_keep_their_rows_beyond_the_grid", test_deferred_pairs_keep_their_rows_beyond_the_grid),
    ("test_wide_box_keeps_every_support", test_wide_box_keeps_every_support),
    ("test_large_plane_group_follows_the_law", test_large_plane_group_follows_the_law),
    ("test_plane_group_of_many_rows_takes_the_lanes", test_plane_group_of_many_rows_takes_the_lanes),
    ("test_window_past_its_bound_is_flagged", test_window_past_its_bound_is_flagged),
    ("test_solid_launches_take_fixed_grids", test_solid_launches_take_fixed_grids),
    ("test_job_grids_take_each_entry_once", test_job_grids_take_each_entry_once),
    ("test_shadow_keeps_contacts_bitwise_with_every_class", test_shadow_keeps_contacts_bitwise_with_every_class),
):
    add_function_test(TestTerrainLawSolid, _name, _func, devices=devices)
# The spill check's warning is the expected output; the test captures it itself.
add_function_test(
    TestTerrainLawSolid,
    "test_long_box_keeps_its_first_rows_without_room",
    test_long_box_keeps_its_first_rows_without_room,
    devices=devices,
    check_output=False,
)
# The job check's warning is the expected output; the test captures it itself.
add_function_test(
    TestTerrainLawSolid,
    "test_job_overflow_takes_the_full_walk",
    test_job_overflow_takes_the_full_walk,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestTerrainLawSolid,
    "test_solid_shadow_runs_under_graph_capture",
    test_solid_shadow_runs_under_graph_capture,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

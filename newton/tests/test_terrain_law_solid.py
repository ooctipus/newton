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
  their exact separation (34 of 126 poses). Since seam witnesses take the face beneath their partner, no row has a
  normal outside the surface's normal cone (there were 25 such ghost rows).

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
from newton._src.geometry import (
    terrain_law_barrel,
    terrain_law_box,
    terrain_law_cylinder,
    terrain_law_hull,
    terrain_law_solid,
)
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
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_RECOVERY,
    TERRAIN_FINISH_MASK_ROWS,
    TERRAIN_PAIR_CANDIDATE_OVERFLOW,
    TERRAIN_PAIR_DEFERRED,
    TERRAIN_PAIR_FILTERED,
    TERRAIN_PAIR_INCOMPLETE,
    TERRAIN_PAIR_JOBS,
    TERRAIN_PAIR_OUTSIDE,
    TERRAIN_PAIR_ROW_OVERFLOW,
    TERRAIN_PAIR_STALE,
    TERRAIN_PAIR_TILE_EXIT,
    TERRAIN_PAIR_WIDE,
    TERRAIN_STATUS_JOBS,
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
    """Collide once on today's route as the suite runs it (:data:`SUITE_ROUTE`), with every class in shadow or none
    (``NEWTON_TERRAIN_LAW=none``: the terrain route writes every heightfield contact)."""
    options = dict(PIPELINE, **kwargs)
    with patch.dict(os.environ, SUITE_ROUTE):
        if shadow:
            pipeline = tu.shadow_pipeline(model, classes="all", **options)
        else:
            pipeline = tu.terrain_route_pipeline(model, **options)
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
    most 32) and ``terrain_finish_list`` on all of them (core points ``q + d n``, in the list's block of a scratch, one
    lane), each as per-row keep flags and the kept count last."""
    i = wp.tid()
    base = i * 40
    count = counts[i]
    keep, kept = terrain_finish_slots(q, n, d, base, prefixes[i], eps[i], False)
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
    in_list[i, 40] = terrain_finish_list(scratch, block, count, eps[i], 0, 1)
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
        # at most the in-place finish's rows (a walk holds at most its class's walk slots)
        rows = [rows[k] for k in order][:TERRAIN_FINISH_MASK_ROWS]
        prefix = len(rows)
        if i % 17 == 0:
            rows = (rows * 6)[:36]
        lists.append((rows, min(prefix, len(rows))))
    return lists


def _chain_lists(rng, count, eps):
    """Candidate lists whose merge and plane groups only candidate order decides: chains of rows each coinciding with
    the next but not with the one after (0.55 merge tolerances apart, ``4 eps``) and chains of rows each on the next
    one's plane but not on the one after's (normals 0.55 group tolerances apart), on and around a plane group of
    corners, edge and inner rows, in random order. Returns ``(rows, prefix)`` per list, every row its own."""
    step = 0.55 * 4.0 * eps
    lists = []
    for _i in range(count):
        rows = []
        normal = rng.normal(size=3) * np.array([0.3, 0.3, 1.0])
        normal[2] = abs(normal[2]) + 0.5
        normal /= np.linalg.norm(normal)
        e1 = np.cross(normal, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        origin = rng.uniform(-0.05, 0.05, size=3)
        tilt = rng.uniform(-0.05, 0.05, size=2) * rng.choice([0.0, 1.0])

        def depth(u, v, tilt=tilt):
            return 0.01 + tilt[0] * u + tilt[1] * v + rng.choice([0.0, 0.0, rng.uniform(-0.002, 0.004)])

        for _j in range(int(rng.integers(3, 9))):
            u, v = rng.uniform(-0.02, 0.02, size=2)
            rows.append((origin + u * e1 + v * e2, normal, depth(u, v)))
        for _chain in range(int(rng.integers(1, 3))):
            # a merge chain on the group's plane
            u, v = rng.uniform(-0.02, 0.02, size=2)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            direction = np.cos(angle) * e1 + np.sin(angle) * e2
            dist = depth(u, v)
            for j in range(int(rng.integers(3, 6))):
                rows.append((origin + u * e1 + v * e2 + j * step * direction, normal, dist))
        for _chain in range(int(rng.integers(1, 3))):
            # a plane chain: normals turned 0.55 group tolerances a row, positions within 2 cm (offsets of a few
            # tenths of a band from each other's planes)
            axis = e1 if rng.uniform() < 0.5 else e2
            for j in range(int(rng.integers(3, 6))):
                nj = normal + j * 0.55e-5 * axis
                nj /= np.linalg.norm(nj)
                u, v = rng.uniform(-0.01, 0.01, size=2)
                rows.append((origin + u * e1 + v * e2, nj, depth(u, v)))
        order = rng.permutation(len(rows))
        rows = [rows[k] for k in order][:TERRAIN_FINISH_MASK_ROWS]
        lists.append((rows, len(rows)))
    return lists


def test_finish_in_place_equals_finish(test, device):
    """Keep the same rows, bit for bit, with the finish in a walk's slots as with the finish on a scratch (the finish
    of deferred pairs), and finish any number of candidates there.

    Lists with coincident rows, several plane groups, implied rows and groups of more than four rows, and lists whose
    merge and plane groups only candidate order decides (:func:`_chain_lists`); lists of more rows than the in-place
    finish's masks hold (:data:`TERRAIN_FINISH_MASK_ROWS`) are copies of a shorter list, whose copies merge into their
    first occurrence: the scratch finish keeps the rows the in-place finish keeps of the shorter list.
    """
    rng = np.random.default_rng(31)
    lists = _candidate_lists(rng, 400) + _chain_lists(rng, 200, 8.0 * 1.1920928955078125e-7 * 0.2)
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
    keep, kept = terrain_finish_slots(out.q, out.n, out.d, base, held, eps[i], False)
    for j in range(held):
        merged[i, j] = (keep >> j) & 1
    merged[i, 39] = held
    merged[i, 40] = kept
    keep, kept = terrain_finish_slots(out.q, out.n, out.d, base, held, eps[i], True)
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
    counts: wp.array[int],
    eps: wp.array[float],
    scratch: TerrainFinishScratch,
    one_lane: wp.array2d[int],
    lanes: wp.array2d[int],
):
    """Per list of rows at ``40 i``, one block and the list's block of a scratch (core points ``q + d n``):
    ``terrain_finish_list`` by the first lane alone (``0, 1``), then by the block's lanes together, each as per-row keep
    flags and the kept count last."""
    i, lane = wp.tid()
    base = i * 40
    block = i * scratch.capacity
    count = counts[i]
    if lane == 0:
        for j in range(count):
            s = base + j
            scratch.q[block + j] = q[s]
            scratch.k[block + j] = q[s] + d[s] * n[s]
            scratch.n[block + j] = n[s]
            scratch.d[block + j] = d[s]
            scratch.feature[block + j] = j
        kept = terrain_finish_list(scratch, block, count, eps[i], 0, 1)
        for j in range(count):
            one_lane[i, j] = scratch.mark[block + j]
        one_lane[i, 40] = kept
    # the lanes split every step: they start after the first lane's reads (a block barrier)
    wp.tile_extract(wp.tile_max(wp.tile(0)), 0)
    kept = terrain_finish_list(scratch, block, count, eps[i], lane, wp.block_dim())
    if lane == 0:
        for j in range(count):
            lanes[i, j] = scratch.mark[block + j]
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
    """Keep the same rows, bit for bit, with a block's lanes splitting every step of the finish as with one lane taking
    them all (the finish of deferred pairs, on a scratch): the merge, the plane groups, the implied-row tests and the
    groups' ranks, each test the same float32 operations, the lanes' answers combined across the block.

    The candidate lists of :func:`test_finish_in_place_equals_finish` (with the chains only candidate order decides,
    which the first lane decides) and lists dominated by one plane group of 9-24 rows, whose implied-row tests cost the
    most. On the CPU a block is one lane.
    """
    rng = np.random.default_rng(47)
    lists = (
        _candidate_lists(rng, 300)
        + _chain_lists(rng, 300, 8.0 * 1.1920928955078125e-7 * 0.2)
        + _large_group_lists(rng, 300)
    )
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
    one_lane = wp.zeros((len(lists), 41), dtype=int, device=device)
    lanes = wp.zeros((len(lists), 41), dtype=int, device=device)
    wp.launch_tiled(
        _finish_lanes,
        dim=len(lists),
        inputs=[
            wp.array(q, dtype=wp.vec3, device=device),
            wp.array(n, dtype=wp.vec3, device=device),
            wp.array(d, dtype=float, device=device),
            wp.array(counts, dtype=int, device=device),
            wp.array(eps, dtype=float, device=device),
            terrain_finish_scratch(device, 40, blocks=len(lists)),
        ],
        outputs=[one_lane, lanes],
        block_dim=terrain_law_block_dim(device),
        device=device,
    )
    a = one_lane.numpy()
    b = lanes.numpy()
    for i in range(len(lists)):
        np.testing.assert_array_equal(b[i, : counts[i]], a[i, : counts[i]], err_msg=f"list {i}")
        test.assertEqual(b[i, 40], a[i, 40], msg=f"list {i}")
    # the large groups lose rows to the implied-row tests and the group caps
    large = slice(len(lists) - 300, len(lists))
    test.assertGreater(int(np.sum(a[large, 40] < counts[large])), 250)


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
74 of 4,000 cases of the generator short)."""
BURIED_SHORT = 1
"""Buried solids of 120 whose deepest row may be short of the core's depth: one of this seed's, a cone 14 mm short (the
law leaves 154 of 12,000 of the generators short)."""


def test_exposed_cell_rows_follow_the_law(test, device):
    """Give every cylinder through an exposed cell the float64 law's rows and a row, and all but a few their depth.

    The exposed-cell cases of the cylinder query's exact LP reference (DESIGN 7, Track 1: any axis and size, steep
    faces, 60 m out), adapted per pair. The reference requires every triangle prism's row to reach the exact minimum
    over that prism (today's route misses 1,569 of 2,000 cases), which a per-pair law that emits one support set per
    pair, not one row per triangle, does not do by design; here a pair's deepest row must reach its core's depth.

    A pair reaches its depth unless its deepest point's closest surface point lies on no feature that gives it a row
    (a valley, or a vertex its rules leave out, or a tie: equally far from two or more surface points; DESIGN 1.3 and
    8.5). Over 4,000 cases of this generator (100 seeds) the shadow rows equal the float64 law's in all but one case
    (a decision within float32 resolution) and leave 74 short (by up to 161 mm); the removed rule X, which exited the
    core's deepest candidate point, left 4. Today's route leaves 526 short. The check allows :data:`LP_SHORT` of its
    40.
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

    The law reaches a pair's depth unless its deepest point's closest surface point lies on no feature that gives it a
    row (DESIGN 1.3 and 8.5). Over 4,000 of each (100 seeds) the law left 32, 106 and 16 short, none rowless; the
    removed rule X, which exited the core's deepest candidate point, left 7, 14 and 6, all ties. Today's route leaves
    2,088, 345 and 11 short. The check allows :data:`BURIED_SHORT` of its 120.
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


def test_valley_cores_follow_the_law(test, device):
    """Give solids sunk into a valley, each deepest in the wedge below the valley's line, the float64 law's rows.

    Their deepest points' closest surface point is the valley's line, which no rule offers: rule F's supports project
    across the valley onto the other wall, and rule R's recovery points give the walls' plane depths (short by up to 13%
    of the depth).
    """
    hf, model, _hfield, shapes, bodies = _valley_bodies(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        with test.subTest(kind=kind):
            got = rows.get(shape, [])
            test.assertGreater(len(got), 0)
            missing, extra, deepest = su.compare_solid_rows(su.law_rows(hf, kind, dims, pose), got)
            test.assertEqual((missing, extra), ([], []))
            test.assertLess(deepest, 1.0e-5)


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
# the border
# ----------------------------------------------------------------------------------------------------------------


def test_solid_rows_match_the_float64_law_at_the_border(test, device):
    """Reproduce the float64 law's rows at the heightfield's border in float32 for every solid kind.

    Boxes, cylinders, cones, ellipsoids and hulls in any orientation within 8 cm of a wall's plane or a corner of flat,
    rough and sloped tiles, their lowest point up to 3 cm into the top: the faces clipped to the footprint, the walls,
    the border edges and vertices and the corner lines give the float64 law's rows. No float64 row is missed and no
    float32 row is extra but those the other side's rows imply or carry (the same plane and depth: a single clipped
    point's place along its edge is ill-conditioned; at most 2% of the rows); every pair's deepest row agrees within
    10 um, and every penetrating row's core point lies inside the block.
    """
    rng = np.random.default_rng(37)
    kinds = [kind for kind in _solid_kinds(rng) if kind[1] is not TOE]
    totals = {"f64": 0, "carried": 0}
    defects = []
    for name, heights in tu.border_terrains(rng).items():
        with test.subTest(terrain=name):
            hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.0, device="cpu"))
            bodies = tu.border_bodies(rng, hf, kinds, 8)
            model, _hfield, shapes = su.solid_scene(heights, 1.0, 1.0, bodies, device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            rows = tu.shadow_rows(pipeline)
            for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
                law = su.law_rows(hf, kind, dims, pose)
                got = rows.get(shape, [])
                missing, extra, deepest = su.compare_solid_rows(law, got)
                carried = [m for m in missing if tu.carries_half_space(m, got)]
                carried += [e for e in extra if tu.carries_half_space(e, law)]
                totals["f64"] += len(law)
                totals["carried"] += len(carried)
                if len(missing) + len(extra) > len(carried) or deepest > 1.0e-5 or tu.outside_face_rows(hf, got):
                    defects.append((name, kind, pose.tolist(), [repr(m) for m in missing], extra, deepest))
    test.assertEqual(defects, [])
    test.assertGreater(totals["f64"], 100)
    test.assertLessEqual(totals["carried"], 0.02 * totals["f64"], msg=totals)


def test_clip_rows_are_front_facing(test, device):
    """Keep a face's clipped rows to points inside the solid, as its supported points are.

    An ellipsoid buried under a ridge of a sawtooth terrain (56 deg faces), straddling the border's edge: the core's
    part inside the footprint on a border edge's clip plane lies below each face's plane, but above the surface, under
    the other face; the clipped rows (7.8 and 4.4 cm deep along the faces' normals before the test) would push it out
    of the ridge's faces it does not penetrate there. The border edge's crease row remains, the float64 law's.
    """
    x = np.linspace(-1.6, 1.6, 33)
    heights = np.tile(np.abs(((x + 1.6) % 0.2) - 0.1) * 1.5, (33, 1))
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
    quat = [0.4019, -0.4489, -0.7217, -0.3407]
    bodies = [("ellipsoid", ELLIPSOID, _f32_pose([0.992, y, 0.1397, *quat])) for y in (1.6, 1.62)]
    model, _hfield, shapes = su.solid_scene(heights, 1.6, 1.6, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
        with test.subTest(y=float(pose[1])):
            got = rows.get(shape, [])
            law = su.law_rows(hf, kind, dims, pose)
            test.assertEqual(len(got), 1)
            test.assertEqual(su.compare_solid_rows(law, got)[:2], ([], []))
            test.assertEqual(tu.outside_face_rows(hf, got), [])
            test.assertEqual((got[0][3] >> 5) & 3, int(TERRAIN_FEATURE_EDGE))


def _feature_parts(feature):
    """A feature id's ``(r mod 64, c mod 64, kind, sub-feature, point slot)``."""
    return (feature >> 13) & 63, (feature >> 7) & 63, (feature >> 5) & 3, (feature >> 3) & 3, feature & 7


def test_border_rows_carry_their_feature_codes(test, device):
    """Give the border's rows their feature ids: walls and clipped faces under the face kind, corner lines under the
    edge kind, each row of a pair distinct.

    On a flat unit heightfield of 0.1 m cells: a level box beside the wall ``x = 1`` gets its face's four corners on the
    wall (the face kind, sub-feature ``TERRAIN_FACE_WALL + 1``, the wall of a y-edge, slots 0-3); a box beside a
    corner's line, two rows on it (the edge kind, sub-feature ``TERRAIN_EDGE_CORNER``); a box tilted over the wall's
    edge, the top face clipped at the wall (sub-feature ``TERRAIN_FACE_WALL + 1``, the crossing's ends in slots 4 and 5,
    each from the edge it lies on) and the border edges' crease rows (the edge kind, the y-edges); a level box over the
    corner, 1 cm deep, the top face's row inside (sub-feature 0 or 1) and the clipped row at the corner vertex.
    """
    level = np.array([0.0, 0.0, 0.0, 1.0])
    tilted = np.array([0.0, np.sin(np.pi / 8.0), 0.0, np.cos(np.pi / 8.0)])
    bodies = [
        ("box", {"half": (0.05, 0.05, 0.05)}, _f32_pose(np.r_[1.055, 0.33, -0.1, level])),
        ("box", {"half": (0.05, 0.05, 0.05)}, _f32_pose(np.r_[1.055, 1.055, -0.1, level])),
        ("box", {"half": (0.2, 0.05, 0.05)}, _f32_pose(np.r_[0.95, -0.33, 0.11, tilted])),
        ("box", {"half": (0.1, 0.1, 0.05)}, _f32_pose(np.r_[1.0, 1.0, 0.04, level])),
    ]
    model, _hfield, shapes = su.solid_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    wall, corner, hanging, over = (rows.get(shape, []) for shape in shapes)
    face, edge = int(TERRAIN_FEATURE_FACE), int(TERRAIN_FEATURE_EDGE)
    test.assertEqual(sorted(_feature_parts(f)[2:] for _q, _n, _d, f in wall), [(face, 3, j) for j in range(4)])
    test.assertEqual(sorted(_feature_parts(f)[2:4] for _q, _n, _d, f in corner), [(edge, 3)] * 2)
    clipped = sorted(_feature_parts(f)[2:] for _q, n, _d, f in hanging if n[2] > 0.999)
    test.assertEqual(clipped, [(face, 3, 4), (face, 3, 5)])
    creases = sorted(_feature_parts(f)[2:4] for _q, n, _d, f in hanging if n[2] < 0.999)
    test.assertEqual(creases, [(edge, 1)] * 2)
    inside = [_feature_parts(f) for q, _n, _d, f in over if np.allclose(q[:2], (0.9, 0.9), atol=1.0e-5)]
    test.assertEqual([p[2:4] in ((face, 0), (face, 1)) for p in inside], [True])
    at_corner = [_feature_parts(f) for q, _n, _d, f in over if np.allclose(q[:2], (1.0, 1.0), atol=1.0e-5)]
    test.assertEqual([p[2] == face and p[3] >= 2 and p[4] >= 4 for p in at_corner], [True])
    for name, got in (("wall", wall), ("corner", corner), ("hanging", hanging), ("over", over)):
        features = [f for _q, _n, _d, f in got]
        test.assertEqual(len(set(features)), len(features), msg=name)


# ----------------------------------------------------------------------------------------------------------------
# barrel cylinders and finite planes
# ----------------------------------------------------------------------------------------------------------------

BARRELS = (
    {"radius": 0.04, "half_height": 0.05, "barrel": 0.12},
    {"radius": 0.03, "half_height": 0.06, "barrel": 0.06},
    {"radius": 0.05, "half_height": 0.03, "barrel": 2.0},
)
"""Barrel cylinders: a bulging one, one whose side is a half circle (``R = h``), a nearly straight one."""
PLATE = {"size": (0.16, 0.1)}
"""A finite plane, 16 cm by 10 cm (a box of half that width and length and no thickness)."""


def _barrel_kinds():
    return [("barrel", dims) for dims in BARRELS] + [("plane", PLATE)]


def _free_poses(rng, hf, kinds, count):
    """Random poses over the footprint, any orientation, the lowest point 3 cm into to 12 mm above the terrain under
    the body."""
    bodies = []
    for i in range(count * len(kinds)):
        kind, dims = kinds[i % len(kinds)]
        x, y = rng.uniform(-0.8, 0.8, 2)
        q = _random_quat(rng)
        lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
        top = max(hf.height(xx, yy) for xx in np.linspace(lo[0], hi[0], 7) for yy in np.linspace(lo[1], hi[1], 7))
        bodies.append((kind, dims, _f32_pose(np.r_[x, y, top - lo[2] + rng.uniform(-0.03, 0.012), q])))
    return bodies


def test_barrels_and_planes_match_the_float64_law(test, device):
    """Reproduce the float64 law's rows of barrel cylinders and finite planes in float32.

    Barrels of three profiles and a plate in any orientation over flat, rough and sloped tiles, over the footprint and
    at its border, the lowest point up to 3 cm into the terrain: no float64 row is missed and no float32 row is extra
    but those the other side implies, every pair's deepest row agrees within 10 um, and every penetrating face row's
    core point lies inside the block.
    """
    rng = np.random.default_rng(43)
    kinds = _barrel_kinds()
    defects, total = [], 0
    for name, heights in tu.border_terrains(rng).items():
        hf = tu.reference_surface(tu.heightfield_model(heights, 1.0, 1.0, device="cpu"))
        bodies = _free_poses(rng, hf, kinds, 6) + tu.border_bodies(rng, hf, kinds, 4)
        model, _hfield, shapes = su.solid_scene(heights, 1.0, 1.0, bodies, device=device)
        pipeline, _contacts, _state = _collide(model, shadow=True)
        rows = tu.shadow_rows(pipeline)
        for shape, (kind, dims, pose) in zip(shapes, bodies, strict=True):
            law = su.law_rows(hf, kind, dims, pose)
            got = rows.get(shape, [])
            missing, extra, deepest = su.compare_solid_rows(law, got)
            total += len(law)
            if missing or extra or deepest > 1.0e-5 or tu.outside_face_rows(hf, got):
                defects.append((name, kind, pose.tolist(), [repr(m) for m in missing], extra, deepest))
    test.assertEqual(defects, [])
    test.assertGreater(total, 100)


def test_barrels_follow_exact_geometry(test, device):
    """Give every barrel cylinder within reach a deepest row equal to its exact separation (as deep as its deepest
    point when penetrating) and every normal inside the surface's normal cone (check_solid), on rough, pitted and
    stepped terrain, half of them in any orientation, half tilted 2, 10 or 30 deg."""
    rng = np.random.default_rng(5)
    for name, heights in _rough_terrains(rng).items():
        with test.subTest(terrain=name):
            hf = tu.reference_surface(tu.heightfield_model(heights, 1.6, 1.6, device="cpu"))
            bodies = []
            for dims in BARRELS:
                for i in range(6):
                    x, y = rng.uniform(-1.1, 1.1, 2)
                    q = _random_quat(rng, None if i % 2 else np.radians(rng.choice([2.0, 10.0, 30.0])))
                    lo, hi = ref.make_core("barrel", dims, np.r_[x, y, 0.0, q]).aabb()
                    z = _top_under(hf, lo, hi) - lo[2] + rng.uniform(-0.005, 0.025)
                    bodies.append(("barrel", dims, _f32_pose(np.r_[x, y, z, q])))
            model, _hfield, shapes = su.solid_scene(heights, 1.6, 1.6, bodies, device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            incomplete, ghosts = _exact_violations(hf, tu.shadow_rows(pipeline), shapes, bodies)
            test.assertEqual(incomplete, [])
            test.assertEqual(ghosts, [])


def _flat_rows(device, bodies):
    """The shadow rows of bodies over a flat unit heightfield of 0.1 m cells, per body."""
    model, _hfield, shapes = su.solid_scene(np.zeros((21, 21)), 1.0, 1.0, bodies, device=device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    return [rows.get(shape, []) for shape in shapes]


def test_barrel_rows_follow_its_profile(test, device):
    """Give a barrel cylinder on flat ground the rows of its own shape, 2 mm into it.

    Lying on its side: one row at its side's lowest point, the bulge (a curved side has no rolling points). Upright:
    the cylinder's tripod on its cap. Tilted 80 deg: one row at its lowest point on the side, Newton's support point.
    """
    dims = BARRELS[0]
    core = ref.make_core("barrel", dims, np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    bulge = core.rho(0.0)
    tilted = _axis_angle((1.0, 0.0, 0.0), np.radians(80.0))
    lowest = ref.make_core("barrel", dims, np.r_[0.0, 0.0, 0.0, tilted]).lowest(np.array([0.0, 0.0, 1.0]))
    bodies = [
        ("barrel", dims, _f32_pose(np.r_[0.33, 0.37, bulge - 0.002, LYING])),
        ("barrel", dims, _f32_pose(np.r_[-0.33, 0.37, dims["half_height"] - 0.002, 0.0, 0.0, 0.0, 1.0])),
        ("barrel", dims, _f32_pose(np.r_[0.33, -0.37, -lowest[2] - 0.002, tilted])),
    ]
    lying, upright, tilt = _flat_rows(device, bodies)
    up = np.array([0.0, 0.0, 1.0])
    expected = {
        "lying": [(np.array([0.33, 0.37, 0.0]), up, -0.002)],
        "tilted": [(np.array([0.33 + lowest[0], -0.37 + lowest[1], 0.0]), up, -0.002)],
    }
    for name, got in (("lying", lying), ("tilted", tilt)):
        with test.subTest(pose=name):
            test.assertEqual(len(got), 1, msg=got)
            test.assertEqual(tu.missing_rows(expected[name], got, q_tol=1.0e-5, angle_tol=1.0e-5, d_tol=1.0e-6), [])
    with test.subTest(pose="upright"):
        test.assertEqual(len(upright), 3, msg=upright)
        for q, n, d, _f in upright:
            test.assertAlmostEqual(d, -0.002, delta=1.0e-6)
            test.assertLess(tu.angle_between(n, up), 1.0e-5)
            test.assertAlmostEqual(float(np.hypot(q[0] + 0.33, q[1] - 0.37)), dims["radius"], delta=1.0e-5)


def test_barrel_kernels_only_with_a_barrel(test, device):
    """Take the cylinder class's barrel kernels only for a table that holds a barrel cylinder, and give flat-sided
    cylinders and cones the same rows from both kernel sets, bit for bit.

    The class's own kernels set every barrel radius to a literal 0, so that the compiler drops the core functions'
    barrel paths; with a barrel among its partners the class takes :mod:`terrain_law_barrel`'s for every pair. Cylinders
    and cones of random poses over rough terrain, with and without a barrel beside them.
    """
    rng = np.random.default_rng(29)
    heights = np.round(rng.uniform(0.0, 0.06, size=(31, 31)) / 0.005) * 0.005
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.5, 1.5, device="cpu"))
    kinds = [("cylinder", CYLINDER), ("cone", CONE)]
    bodies = _free_poses(rng, hf, kinds, 12)
    barrel = [("barrel", BARRELS[0], _f32_pose(np.r_[1.2, 1.2, 0.2, 0.0, 0.0, 0.0, 1.0]))]
    rows, names = [], []
    for scene in (bodies, bodies + barrel):
        model, _hfield, shapes = su.solid_scene(heights, 1.5, 1.5, scene, device=device)
        pipeline, _contacts, _state = _collide(model, shadow=True)
        shadow = pipeline._terrain_law_shadow
        names.append({launch[1].key.split("_rows")[0] for launch in shadow._launches if "_rows" in launch[1].key})
        got = tu.shadow_rows(pipeline)
        rows.append([got.get(shape, []) for shape in shapes[: len(bodies)]])
    test.assertIn("terrain_cylinder", names[0])
    test.assertNotIn("terrain_barrel", names[0])
    test.assertIn("terrain_barrel", names[1])
    test.assertNotIn("terrain_cylinder", names[1])
    total = 0
    for i, (plain, with_barrel) in enumerate(zip(rows[0], rows[1], strict=True)):
        test.assertEqual(len(plain), len(with_barrel), msg=f"body {i}")
        for (qa, na, da, fa), (qb, nb, db, fb) in zip(plain, with_barrel, strict=True):
            bits_a = np.float32(qa).tobytes() + np.float32(na).tobytes() + np.float32(da).tobytes()
            bits_b = np.float32(qb).tobytes() + np.float32(nb).tobytes() + np.float32(db).tobytes()
            test.assertEqual((bits_a, fa), (bits_b, fb), msg=f"body {i}")
        total += len(plain)
    test.assertGreater(total, 20)


def test_finite_planes_rest_as_boxes_without_thickness(test, device):
    """Give a finite plane on a body the rows of a box of its width and length and no thickness.

    Level and 2 mm into flat ground: its four corners (half its width and length about its centre). Tilted 20 deg about
    its length: the two corners of its lower edge. Over the border's edge: its corners over the footprint and its
    rectangle clipped at the edge.
    """
    half = np.array(PLATE["size"]) / 2.0
    tilt = _axis_angle((0.0, 1.0, 0.0), np.radians(20.0))
    drop = half[0] * np.sin(np.radians(20.0))
    level = np.array([0.0, 0.0, 0.0, 1.0])
    bodies = [
        ("plane", PLATE, _f32_pose(np.r_[0.33, 0.37, -0.002, level])),
        ("plane", PLATE, _f32_pose(np.r_[-0.33, 0.37, drop - 0.002, tilt])),
        ("plane", PLATE, _f32_pose(np.r_[1.0, -0.37, -0.002, level])),
    ]
    flat, tilted, border = _flat_rows(device, bodies)
    up = np.array([0.0, 0.0, 1.0])

    def corners(cx, cy, xs, ys):
        return [(np.array([cx + x, cy + y, 0.0]), up, -0.002) for x in xs for y in ys]

    cases = (
        ("level", flat, corners(0.33, 0.37, (-half[0], half[0]), (-half[1], half[1]))),
        ("tilted", tilted, corners(-0.33 + half[0] * np.cos(np.radians(20.0)), 0.37, (0.0,), (-half[1], half[1]))),
        ("border", border, corners(1.0, -0.37, (-half[0], 0.0), (-half[1], half[1]))),
    )
    for name, got, want in cases:
        with test.subTest(pose=name):
            test.assertEqual(len(got), len(want), msg=got)
            test.assertEqual(tu.missing_rows(want, got, q_tol=1.0e-5, angle_tol=1.0e-5, d_tol=2.0e-6), [])


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


def _deferred_candidates(shadow, partner_class) -> int:
    """The distinct candidates a block of a solid class's finish of deferred pairs holds (the most any of its pairs can
    give)."""
    name = {TERRAIN_PARTNER_BOX: "box", TERRAIN_PARTNER_CYLINDER: "cylinder", TERRAIN_PARTNER_HULL: "hull"}
    return shadow.describe()["launch_shapes"][name[partner_class]]["deferred_candidates"]


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
    inputs.shape_surface = pipeline._terrain_law_shadow.shape_surface
    inputs.shape_body = model.shape_body
    inputs.body_flags = model.body_flags
    inputs.include_static_kinematic_pairs = 1
    return inputs


_SOLID_MODULES = {
    TERRAIN_PARTNER_BOX: (terrain_law_box, "box"),
    TERRAIN_PARTNER_CYLINDER: (terrain_law_cylinder, "cylinder"),
    TERRAIN_PARTNER_HULL: (terrain_law_hull, "hull"),
}
_WHOLE_PAIR_EXITS = int(TERRAIN_PAIR_TILE_EXIT | TERRAIN_PAIR_STALE | TERRAIN_PAIR_FILTERED | TERRAIN_PAIR_OUTSIDE)
"""States of a pair the whole-pair rejects take (no row)."""
_LIST_OF_KERNEL = {
    "scan": "scans",
    "faces": "faces",
    "edges": "edges",
    "thetas": "thetas",
    "vertices": "vertices",
    "border": "border",
    "close": "live",
    "recovery": "recover",
    "small": "small",
}
"""The list each job kernel that first takes a list strides over
(:class:`~newton._src.geometry.terrain_law_solid.TerrainSolidJobs`; the creases kernel takes the thetas kernel's list
entry by entry, its results kept per entry)."""
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


def _class_jobs(model, shadow, partner_class, *, threads=None, reserve=True, **capacities):
    """A solid class's jobs, sized as the shadow sizes them but for ``capacities`` (room per list), with the shadow's
    reserves beyond the room (none without ``reserve``) and ``threads`` job threads (by default the shadow's)."""
    table = shadow.table
    begin, end = _class_pairs(table, partner_class)
    run = int(TERRAIN_PARTNER_SCAN_RUN[partner_class])
    sizes = terrain_law_solid.terrain_solid_job_capacities(
        table.vertices[begin:end],
        table.candidates[begin:end],
        int(TERRAIN_PARTNER_WALK_SLOTS[partner_class]),
        scan_run=run,
        sides=table.sides[begin:end],
    )
    sizes.update(capacities)
    reserves = None
    if reserve:
        reserves = terrain_law_solid.terrain_solid_job_reserves(
            table.vertices[begin:end], table.candidates[begin:end], scan_run=run, sides=table.sides[begin:end]
        )
    jobs = terrain_law_solid.terrain_solid_jobs(model.device, partner_class, begin, end - begin, sizes, reserves)
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


def _defer_every_pair(out, jobs, partner_class, begin, end):
    """Defer every pair of a solid class holding a candidate before its close, as the close defers a pair holding more
    rows than its walk slots (:func:`~newton._src.geometry.terrain_law.terrain_defer_pair`), so that the finish of
    deferred pairs takes it; the close passes it by."""
    state = out.state.numpy()
    heads = jobs.head.numpy()[: end - begin]
    pairs = begin + np.flatnonzero(((state[begin:end] & int(TERRAIN_PAIR_JOBS)) != 0) & (heads >= 0))
    deferred = out.deferred.numpy()
    deferred[begin : begin + len(pairs)] = pairs
    out.deferred.assign(deferred)
    counts = out.deferred_count.numpy()
    counts[partner_class] = len(pairs)
    out.deferred_count.assign(counts)
    state[pairs] = (state[pairs] & int(TERRAIN_PAIR_WIDE)) | int(TERRAIN_PAIR_DEFERRED)
    out.state.assign(state)
    return len(pairs)


def _run_class(
    pipeline,
    model,
    partner_class,
    *,
    jobs=None,
    reverse=False,
    until=None,
    defer=False,
    blocks=None,
    extra_blocks=0,
    extra_threads=0,
    candidates=None,
):
    """Run a solid class's launches (:data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_LAUNCHES`) for a
    shadow pipeline's last collide into a side buffer of their own (:func:`_side_buffer`), with ``jobs`` (by default
    sized as the shadow's), every list reversed before its kernel takes it with ``reverse``, up to the launch ``until``
    (not run). With ``defer``, every pair holding a candidate is deferred before the close (:func:`_defer_every_pair`),
    so that the finish of deferred pairs finishes it. That finish's grid holds ``blocks`` blocks (by default the
    shadow's) of ``candidates`` candidates (by default the class's bound), launched with ``extra_blocks`` more; the job
    kernels are launched with ``extra_threads`` threads more than the jobs' grid. Returns ``(out, jobs, solids)``."""
    shadow = pipeline._terrain_law_shadow
    table = shadow.table
    device = model.device
    begin, end = _class_pairs(table, partner_class)
    if jobs is None:
        jobs = _class_jobs(model, shadow, partner_class)
    solids = terrain_law_solid.terrain_solid_data(model, table.pairs_host[:, 1])
    out = _side_buffer(begin, end - begin, TERRAIN_PARTNER_WALK_SLOTS[partner_class], table.pair_count, device)
    capacity = _deferred_candidates(shadow, partner_class) if candidates is None else candidates
    scratch = terrain_finish_scratch(device, capacity, blocks=blocks)
    grid = scratch.blocks
    if extra_blocks:
        # room for the extra blocks, which must return at once
        scratch = terrain_finish_scratch(device, capacity, blocks=grid + extra_blocks)
        scratch.blocks = grid
    args = [begin, 0, _pair_inputs(model, pipeline), shadow.features.data(), out]
    block = terrain_law_block_dim(device)
    for kind, name in TERRAIN_SOLID_LAUNCHES:
        if name == until:
            break
        if defer and name == "close":
            _defer_every_pair(out, jobs, partner_class, begin, end)
        if reverse and name in _LIST_OF_KERNEL:
            _reverse_list(jobs, _LIST_OF_KERNEL[name])
        kernel = _kernel(partner_class, name)
        if kind == "deferred":
            inputs = [*args, scratch, solids, jobs]
            wp.launch_tiled(kernel, dim=grid + extra_blocks, inputs=inputs, block_dim=block, device=device)
        elif kind == "pair_tiles":
            dim = jobs.finish_blocks + extra_blocks
            wp.launch_tiled(kernel, dim=dim, inputs=[*args, jobs, solids], block_dim=block, device=device)
        else:
            dim = {
                "pairs": end - begin,
                "jobs": jobs.threads,
                "border_jobs": jobs.border_threads,
                "pair_jobs": jobs.pair_threads,
            }[kind]
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


def test_job_pipeline_rows_depend_on_the_pair_alone(test, device):
    """Give a solid pair the same rows, bit for bit, whatever order its jobs run in and whichever finish takes it.

    Solids of every class over rough terrain, among them of every class one whose window holds over 256 vertices (its
    scan takes several jobs, its class's :data:`~newton._src.geometry.terrain_contacts.TERRAIN_PARTNER_SCAN_RUN`
    window vertices each). Three arms per class: its launches as the shadow runs them; the same with one job thread and
    every list reversed before its kernel takes it (the scan's jobs, the features, the creases, the live pairs and the
    pairs left to rule R), so that the candidate pool holds every pair's rows in another order; and every pair holding a
    candidate deferred before its close, so that the finish of deferred pairs finishes it from its candidates, a block's
    lanes splitting its implied-row tests. The close and that finish hold a pair's candidates by their place in window
    order (a cluster plane two scan jobs list is one offer), and the finish in a pair's walk slots keeps the rows the
    finish on a scratch keeps, so the arms agree with each other and with the shadow's own rows.
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
            # windows of several scan jobs; pairs the close finished (not deferred)
            test.assertGreater(counts["scans"], counts["live"])
            test.assertLess(int(jobs_out.deferred_count.numpy()[partner_class]), counts["live"])
            reversed_out, reversed_jobs, _solids = _run_class(
                pipeline, model, partner_class, jobs=_class_jobs(model, shadow, partner_class, threads=1), reverse=True
            )
            pool = counts["candidates"]
            test.assertEqual(_counts(reversed_jobs)["candidates"], pool)
            test.assertFalse(np.array_equal(jobs.key.numpy()[:pool], reversed_jobs.key.numpy()[:pool]))
            deferred_out, _jobs, _solids = _run_class(pipeline, model, partner_class, defer=True)
            # every pair holding a candidate went through the finish of deferred pairs, more than defer by themselves
            deferred = int(deferred_out.deferred_count.numpy()[partner_class])
            test.assertEqual(deferred, int(np.count_nonzero(jobs.head.numpy() >= 0)))
            test.assertGreater(deferred, int(jobs_out.deferred_count.numpy()[partner_class]))
            _assert_same_rows(test, shadow._out, jobs_out, pairs, "the shadow's launch")
            _assert_same_rows(test, jobs_out, reversed_out, pairs, "the lists reversed")
            _assert_same_rows(test, jobs_out, deferred_out, pairs, "every pair deferred")
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
    """List a scan run's survivors as one block of consecutive entries per list, in window order, whatever order the
    threads append in.

    Each scan run reserves its entries of a list with one atomic and writes them in window order, so that a warp of a
    job kernel takes one pair's features (its record, core and heights). Solids of every class over rough terrain, of
    every class one whose window takes several scan runs: no run's entries are split or out of order in the faces,
    edges and vertices lists. (Appending entry by entry, the lanes of a CUDA warp interleave their runs' entries.)
    """
    _hf, model, _shapes, _bodies = _window_scene(np.random.default_rng(37), device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    checked = {}
    for partner_class in _SOLID_MODULES:
        scan = int(TERRAIN_PARTNER_SCAN_RUN[partner_class])
        with test.subTest(partner_class=partner_class):
            _out, jobs, _solids = _run_class(pipeline, model, partner_class, until="close")
            counts = _counts(jobs)
            per_vertex = terrain_law_solid.TERRAIN_SOLID_ITEMS
            lists = {
                # a scan run: its pair's window vertices [k, k + 1) times the run length (TERRAIN_SOLID_ITEMS items per
                # vertex)
                "faces": lambda pair, item, scan=scan, per_vertex=per_vertex: (pair, item // per_vertex // scan),
                "edges": lambda pair, item, scan=scan, per_vertex=per_vertex: (pair, item // per_vertex // scan),
                "vertices": lambda pair, item, scan=scan, per_vertex=per_vertex: (pair, item // per_vertex // scan),
            }
            for name, run_of in lists.items():
                entries = getattr(jobs, name).numpy()[: counts[name]]
                blocks = _runs_of(entries, run_of)
                runs = [run for run, _items in blocks]
                test.assertEqual(len(runs), len(set(runs)), msg=f"{name}: a run's entries are split")
                for run, items in blocks:
                    test.assertEqual(items, sorted(items), msg=f"{name}: run {run} out of window order")
                checked[name] = checked.get(name, 0) + len(blocks)
    # not vacuous: every list holds runs, a scan list runs from pairs of several runs
    test.assertTrue(all(count > 1 for count in checked.values()), msg=checked)


def _slope_resting(device):
    """Solids resting 0.5 mm deep on a planar 15 deg slope (rising along x, 5 cm cells), each flat on it: a box, a
    cylinder lying across the slope and an ellipsoid. Every window's uphill vertices rise above its core's lowest
    point."""
    th = np.radians(15.0)
    x = np.linspace(-1.0, 1.0, 41)
    heights = np.tile(x * np.tan(th), (65, 1))
    n = np.array([-np.sin(th), 0.0, np.cos(th)])
    tilt = _axis_angle((0.0, 1.0, 0.0), -th)
    across = _quat_mul(tilt, _axis_angle((1.0, 0.0, 0.0), -np.pi / 2.0))
    sink = 0.0005

    def on_slope(x0, y0, up):
        return np.r_[x0, y0, x0 * np.tan(th)] + n * (up - sink)

    bodies = [
        ("box", {"half": (0.1, 0.05, 0.03)}, _f32_pose(np.r_[on_slope(-0.2, -0.8, 0.03), tilt])),
        ("cylinder", {"radius": 0.03, "half_height": 0.15}, _f32_pose(np.r_[on_slope(0.1, 0.0, 0.03), across])),
        ("ellipsoid", {"semi": (0.08, 0.04, 0.03)}, _f32_pose(np.r_[on_slope(0.3, 0.8, 0.03), tilt])),
    ]
    model, _hfield, shapes = su.solid_scene(heights, 1.0, 1.6, bodies, device=device)
    return model, shapes, bodies


def test_slope_resting_solids_keep_their_sink(test, device):
    """Give solids resting 0.5 mm deep on a planar slope rows whose deepest is their sink, within float32.

    Their lowest points lie below their windows' uphill vertices, and over every cell under them the slope's plane lies
    their sink above them: a box, a cylinder lying across the slope and an ellipsoid, each flat on it.
    """
    model, shapes, bodies = _slope_resting(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    rows = tu.shadow_rows(pipeline)
    for shape, (kind, _dims, _pose) in zip(shapes, bodies, strict=True):
        with test.subTest(kind=kind):
            got = rows[shape]
            test.assertGreater(len(got), 0)
            # the deepest row is the sink (the slope's plane), within float32
            test.assertAlmostEqual(min(g[2] for g in got), -0.0005, delta=2.0e-6)


def test_slope_buried_solids_follow_the_law(test, device):
    """Give solids buried under a rough slope the float64 law's rows.

    Ellipsoids of 0.3 x 0.3 x 0.1 m and lying cylinders of 0.6 m buried 0.2 to 0.3 m under a 0.6 slope (0.05 m cells,
    1 cm roughness): many of their points' closest surface points lie cells up the slope, outside their windows.
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
    candidates (48) than a finish list held before (32)."""
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
    the box class's 16 walk slots. Its class's finish of deferred pairs finishes it from its candidates in the
    candidate pool and writes its rows to the spill region. (With a per-class limit of 16 rows the law kept 16, each
    plane's deepest first, and raised ``terrain_rows``.)
    """
    hf, model, shapes, bodies = _ridge_boxes(device, (0.9, 0.05, 0.02), (0.05,))
    _kind, dims, pose = bodies[0]
    pipeline, _contacts, _state = _collide(model, shadow=True)
    shadow = pipeline._terrain_law_shadow
    rows = shadow.rows()
    test.assertEqual(int(shadow._out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), 1)
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
    finish of deferred pairs keeps 16 in the box's own walk slots, every plane group's deepest row first (the deepest
    first), then the groups' farthest ones, flags a row overflow and raises ``terrain_rows``. (Before, the pair kept no
    row.)
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
    """Finish every deferred pair from its own rows, however many defer: one block of the finish of deferred pairs for
    three boxes.

    Three long boxes have more rows than the box class's walk slots and defer; a finish of one block takes them one
    after the other, each from its candidates into the block's scratch. Every box keeps the float64 law's rows, as with
    the shadow's own grid, bit for bit, none flagged, each in rows of the spill region.
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
    """Keep every support of a pair whose distinct candidates outnumber what a finish list held before (32).

    A wide box across ridges has 48 distinct candidates: its class's finish of deferred pairs holds them all (its
    scratch holds every candidate a window of the class can give) and keeps the float64 law's 22 rows, raising nothing.
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


def test_plane_groups_finish_in_one_thread(test, device):
    """Finish a pair holding many rows in a plane group and one holding few in a plane group of three with one thread of
    the small-pair finish kernel, and one holding few others in place, all with the float64 law's rows.

    A flat box 2 mm above the pillar lattice of :func:`test_large_plane_group_follows_the_law`: 6 cm across it holds 15
    candidates in one plane group, within the box class's walk slots, which its close lists for the small-pair finish
    kernel and whose implied-row step drops 6; 3 cm across it holds 4 on four planes, which its close writes as they
    are; 2 cm across over one pillar's top it holds its 4 corners on that plane, which its close lists for the
    small-pair finish kernel too. No pair is deferred.
    """
    n = 61
    r = np.arange(n)
    heights = np.where((r[:, None] % 3 != 0) & (r[None, :] % 3 != 0), 0.05, 0.0)
    hf = tu.reference_surface(tu.heightfield_model(heights, 1.5, 1.5, device="cpu"))
    # (half size, centre, pairs for the small-pair finish); a pillar's top spans 5 cm about 0.075
    for half, centre, small in (
        (0.06, (0.013, 0.007), 1),
        (0.03, (0.013, 0.007), 0),
        (0.01, (0.073, 0.077), 1),
    ):
        with test.subTest(half=half):
            dims = {"half": (half, half, 0.02)}
            pose = _f32_pose(np.r_[centre[0], centre[1], 0.05 + 0.02 + 0.002, 0.0, 0.0, 0.0, 1.0])
            model, _hfield, shapes = su.solid_scene(heights, 1.5, 1.5, [("box", dims, pose)], device=device)
            pipeline, _contacts, _state = _collide(model, shadow=True)
            shadow = pipeline._terrain_law_shadow
            counts = shadow.job_counts()["box"]
            test.assertEqual(counts["small"], small)
            test.assertEqual(int(shadow._out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), 0)
            rows = tu.shadow_rows(pipeline)[shapes[0]]
            if half == 0.06:
                # the finish's implied-row step dropped some of its candidates
                test.assertLess(len(rows), counts["candidates"])
            elif small:
                test.assertEqual(len(rows), 4)
            missing, extra, deepest = su.compare_solid_rows(su.law_rows(hf, "box", dims, pose), rows)
            test.assertEqual((missing, extra), ([], []))
            test.assertLess(deepest, 1.0e-5)


def test_window_past_its_bound_is_flagged(test, device):
    """Flag a pair whose window gives more distinct candidates than its class's scratch holds, and raise.

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
    class's finish of deferred pairs takes it), and a box and an ellipsoid whose windows take several scan jobs."""
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
    (their counters the capture zeroes), and a long box deferred to its class's finish of deferred pairs, whose grid
    strides over the deferred count, and spilled.
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
    # the long box's rows exceed its walk slots: the finish of deferred pairs writes them to the spill region
    test.assertGreaterEqual(shadow.rows_spilled(), int(expected["count"].max()))
    test.assertTrue(np.all(expected["state"] & 48 == 0))
    # windows of several scan jobs (a box and an ellipsoid)
    test.assertGreater(counts["box"]["scans"], counts["box"]["live"])
    test.assertGreater(counts["hull"]["scans"], counts["hull"]["live"])
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
    each striding over its list; the close and the small-pair finish, which stride over lists of the class's pairs, the
    job threads up to one per pair; rule R's kernel, blocks of lanes striding over its list, up to one block per pair;
    the finish of deferred pairs the fixed grid of its scratch.

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
            test.assertEqual(record["border_threads"], jobs.border_threads)
            test.assertEqual(record["pair_job_threads"], min(jobs.threads, end - begin))
            test.assertEqual(record["finish_blocks"], jobs.finish_blocks)
            test.assertLessEqual(jobs.finish_blocks, end - begin)
            for kind, kernel_name in TERRAIN_SOLID_LAUNCHES:
                key = _kernel(partner_class, kernel_name).key
                dim = {
                    "pairs": ("launch", end - begin),
                    "jobs": ("launch", jobs.threads),
                    "border_jobs": ("launch", jobs.border_threads),
                    "pair_jobs": ("launch", jobs.pair_threads),
                    "pair_tiles": ("tiled", jobs.finish_blocks),
                    "deferred": ("tiled", record["deferred_blocks"]),
                }[kind]
                test.assertEqual(launched[key], [(*dim, block)], msg=key)
            counts = shadow.job_counts()[name]
            live = jobs.live.numpy()[: counts["live"]] + begin
            test.assertEqual(
                sorted(live.tolist()), (begin + np.flatnonzero((states[begin:end] & _WHOLE_PAIR_EXITS) == 0)).tolist()
            )
    # every launch of the solid modules is one of their classes' launches
    test.assertEqual(len(launched), len(_SOLID_MODULES) * len(TERRAIN_SOLID_LAUNCHES), msg=sorted(launched))
    test.assertGreater(shadow.job_counts()["box"]["scans"], shadow.job_counts()["box"]["live"])
    test.assertGreater(int(shadow._out.deferred_count.numpy()[TERRAIN_PARTNER_BOX]), 0)


def test_job_grids_take_each_entry_once(test, device):
    """Take every job and every deferred pair once, whatever the job kernels' grid, and with launches of more threads
    or blocks than their grids.

    The scan, job, close and rule R kernels stride over their lists by the class's job threads (``jobs.threads``, fixed
    at construction), a thread past them returning at once; the finish of deferred pairs strides over the deferred list
    by its scratch's grid (``scratch.blocks``), a block past it returning at once. Long boxes lie across ridges: their
    rows exceed the box class's walk slots (deferred, spilled). One job thread, three and the shadow's give the same
    rows, as do job kernels launched with three threads more than the jobs' grid, which would otherwise take entries
    twice (closing a pair twice), and a finish of deferred pairs launched with three blocks more than its grid of two,
    which would otherwise finish deferred pairs twice, counting their spill rows twice.
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


def test_job_overflow_takes_the_reserve(test, device):
    """Keep every pair's rows, bit for bit, when its class's job lists, candidate pool or scan's runs have no room for
    its work, count those pairs, raise ``terrain_jobs`` and warn with the entries asked for, by class and list; past the
    reserve, a pair that lost work keeps no row.

    ``NEWTON_TERRAIN_LAW_SHADOW_JOBS`` scales the solid classes' job lists and candidate pools, sized at the scene level
    from the pairs' own bounds: at 1e-6 each has room for one entry. Work past a list's room takes its reserve (the work
    of :data:`~newton._src.geometry.terrain_law_solid.TERRAIN_SOLID_RESERVE_PAIRS` pairs of the class's largest
    bounds), which the same kernels run: its pair's rows are unchanged; the pair is counted (the shadow's third overflow
    count) and raises the narrow phase's sticky ``terrain_jobs`` bit, which the capacity check raises on, as
    ``rigid_contact_max`` does for contacts, though no row is lost; the job check prints the entries asked for against
    the room. The scan's runs, sized from every pair's bound on its window, lack room only for a window its shape data
    no longer bounds: with room for one run, the other pairs' runs take the reserve, their rows unchanged. With room for
    one entry fewer than a launch asks of each list and no reserve, a pair some of whose work found no room keeps no row
    (:data:`TERRAIN_PAIR_INCOMPLETE`, counted, ``terrain_jobs``), every other pair its rows.
    """
    model, _hfield, _shapes = _capture_scene(device)
    pipeline, _contacts, _state = _collide(model, shadow=True)
    expected = pipeline._terrain_law_shadow.rows()
    default_counts = pipeline._terrain_law_shadow.job_counts()
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
    live = 0
    for name in ("box", "cylinder", "hull"):
        capacities = record["launch_shapes"][name]["jobs"]
        reserves = record["launch_shapes"][name]["jobs_reserve"]
        lists = {k: v for k, v in capacities.items() if k != "scans"}
        # one entry each, and the reserve beyond it held all the work
        test.assertEqual(lists, dict.fromkeys(lists, 1))
        counts = shadow.job_counts()[name]
        test.assertGreater(counts["faces"], 1)
        for k in lists:
            test.assertLessEqual(counts[k], 1 + reserves[k], msg=f"{name} {k}")
        live += counts["live"]
    test.assertIn("Terrain-law box class: its rule F face job list overflowed", message)
    test.assertIn("Terrain-law hull class: its rule E edge job list overflowed", message)
    test.assertNotIn("overflowed too", message)
    rows = shadow.rows()
    for name in ("count", "state", "q", "n", "d", "feature"):
        np.testing.assert_array_equal(rows[name], expected[name], err_msg=name)
    # the pairs whose work took the reserve are counted
    rows_lost, candidates_lost, jobs_taken = shadow.overflow_counts()
    test.assertEqual((rows_lost, candidates_lost), (0, 0))
    test.assertGreater(jobs_taken, 1)
    test.assertLessEqual(jobs_taken, live)
    test.assertEqual([k for k, v in small.narrow_phase.buffer_capacity_status().items() if v], ["terrain_jobs"])
    with test.assertRaisesRegex(RuntimeError, "terrain_jobs"):
        small.narrow_phase.check_buffer_capacity()
    # the default lists have room for every pair's work
    pipeline.narrow_phase.check_buffer_capacity()
    test.assertEqual(pipeline._terrain_law_shadow.overflow_counts(), (0, 0, 0))
    table = pipeline._terrain_law_shadow.table
    names = {TERRAIN_PARTNER_BOX: "box", TERRAIN_PARTNER_CYLINDER: "cylinder", TERRAIN_PARTNER_HULL: "hull"}
    lost_pairs = 0
    kept_pairs = 0
    for partner_class in _SOLID_MODULES:
        with test.subTest(partner_class=partner_class):
            begin, end = _class_pairs(table, partner_class)
            pairs = np.arange(begin, end)
            # room for one scan run: the other pairs' runs take the reserve
            jobs = _class_jobs(model, pipeline._terrain_law_shadow, partner_class, scans=1)
            out, jobs, _solids = _run_class(pipeline, model, partner_class, jobs=jobs)
            counts = _counts(jobs)
            test.assertGreater(counts["scans"], 1)
            test.assertGreaterEqual(int(out.overflow.numpy()[2]), counts["live"] - 1)
            _assert_same_rows(test, pipeline._terrain_law_shadow._out, out, pairs, "one scan run")
            # room for one entry fewer than a launch asks of each list, no reserve: a pair that lost work keeps no row
            asked = default_counts[names[partner_class]]
            sizes = {k: max(1, asked[k] - 1) for k in (*terrain_law_solid.TERRAIN_SOLID_JOB_LISTS, "candidates")}
            jobs = _class_jobs(model, pipeline._terrain_law_shadow, partner_class, reserve=False, **sizes)
            out, jobs, _solids = _run_class(pipeline, model, partner_class, jobs=jobs)
            state = out.state.numpy()[pairs]
            lost = (state & int(TERRAIN_PAIR_INCOMPLETE)) != 0
            test.assertGreater(int(np.count_nonzero(lost)), 0)
            test.assertTrue(np.all(out.count.numpy()[pairs][lost] == 0))
            test.assertTrue(np.all(out.first.numpy()[pairs][lost] == -1))
            test.assertEqual(int(out.overflow.numpy()[2]), int(np.count_nonzero(lost)))
            test.assertNotEqual(int(out.status.numpy()[0]) & int(TERRAIN_STATUS_JOBS), 0)
            _assert_same_rows(test, pipeline._terrain_law_shadow._out, out, pairs[~lost], "no reserve")
            lost_pairs += int(np.count_nonzero(lost))
            kept_pairs += int(np.count_nonzero(out.count.numpy()[pairs][~lost] > 0))
    # not vacuous: pairs of every class lost work, and others kept their rows
    test.assertGreaterEqual(lost_pairs, len(_SOLID_MODULES))
    test.assertGreater(kept_pairs, 2)


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
            today = tu.terrain_route_pipeline(model, **PIPELINE)
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
        """Miss supports today: rows shallower than the exact separation, cylinders through an exposed cell left
        without their depth, buried lying cylinders without a contact. No row's normal leaves the surface's cone: a
        seam witness takes the face beneath its partner."""
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
        self.assertEqual(sum(v for k, v in found.items() if k.endswith("ghosts")), 0, msg=found)

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

    def test_solid_kernels_compile_without_contraction_cpu(self):
        """Compile the solid classes' kernels to PTX without fused multiply-adds (NVRTC, any host).

        Every float32 operation then rounds as on the CPU, so the CPU decides every boundary as the GPU does.
        """
        if 86 not in wp.get_cuda_supported_archs():
            self.skipTest("NVRTC cannot compile for sm_86 here")
        for module in (terrain_law_barrel, terrain_law_box, terrain_law_cylinder, terrain_law_hull):
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
    ("test_buried_solids_follow_the_law", test_buried_solids_follow_the_law),
    ("test_valley_cores_follow_the_law", test_valley_cores_follow_the_law),
    ("test_recovery_rows_have_distinct_feature_ids", test_recovery_rows_have_distinct_feature_ids),
    ("test_hovering_core_skips_rule_r", test_hovering_core_skips_rule_r),
    ("test_hovering_core_leaves_at_the_fine_mip", test_hovering_core_leaves_at_the_fine_mip),
    ("test_recovery_picks_follow_depth_then_index", test_recovery_picks_follow_depth_then_index),
    ("test_solid_rows_match_the_float64_law", test_solid_rows_match_the_float64_law),
    ("test_solid_rows_match_the_float64_law_at_the_border", test_solid_rows_match_the_float64_law_at_the_border),
    ("test_clip_rows_are_front_facing", test_clip_rows_are_front_facing),
    ("test_border_rows_carry_their_feature_codes", test_border_rows_carry_their_feature_codes),
    ("test_barrels_and_planes_match_the_float64_law", test_barrels_and_planes_match_the_float64_law),
    ("test_barrels_follow_exact_geometry", test_barrels_follow_exact_geometry),
    ("test_barrel_rows_follow_its_profile", test_barrel_rows_follow_its_profile),
    ("test_barrel_kernels_only_with_a_barrel", test_barrel_kernels_only_with_a_barrel),
    ("test_finite_planes_rest_as_boxes_without_thickness", test_finite_planes_rest_as_boxes_without_thickness),
    ("test_job_pipeline_rows_depend_on_the_pair_alone", test_job_pipeline_rows_depend_on_the_pair_alone),
    ("test_job_lists_hold_each_run_together", test_job_lists_hold_each_run_together),
    ("test_slope_resting_solids_keep_their_sink", test_slope_resting_solids_keep_their_sink),
    ("test_slope_buried_solids_follow_the_law", test_slope_buried_solids_follow_the_law),
    ("test_long_box_keeps_every_support", test_long_box_keeps_every_support),
    ("test_deferred_pairs_keep_their_rows_beyond_the_grid", test_deferred_pairs_keep_their_rows_beyond_the_grid),
    ("test_wide_box_keeps_every_support", test_wide_box_keeps_every_support),
    ("test_large_plane_group_follows_the_law", test_large_plane_group_follows_the_law),
    ("test_plane_groups_finish_in_one_thread", test_plane_groups_finish_in_one_thread),
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
    "test_job_overflow_takes_the_reserve",
    test_job_overflow_takes_the_reserve,
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

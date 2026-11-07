# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's shared per-pair finish and its output layout (feature ids, overflow flags).

The finish keeps every row the law selects: no count bounds a pair's rows, and it takes any number of candidates
(:func:`~newton._src.geometry.terrain_law.terrain_finish_list`, its state in a scratch).
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import make_contact_sort_key
from newton._src.geometry.narrow_phase import record_narrow_phase_buffers
from newton._src.geometry.terrain_law import (
    TERRAIN_FEATURE_BITS,
    TERRAIN_FINISH_CANDIDATE_OVERFLOW,
    TERRAIN_FINISH_ROW_OVERFLOW,
    TERRAIN_STATUS_ROWS,
    TerrainFinishScratch,
    terrain_feature_id,
    terrain_feature_parts,
    terrain_finish_list,
    terrain_finish_scratch,
    terrain_report,
)
from newton.tests import terrain_law_reference as ref
from newton.tests.unittest_utils import add_function_test, get_test_devices

EPS = 4.0e-7
"""A window band of about 0.4 m [m]."""
MAX_ROWS = 48


@wp.kernel(enable_backward=False)
def _finish(
    q: wp.array2d[wp.vec3],
    k: wp.array2d[wp.vec3],
    n: wp.array2d[wp.vec3],
    d: wp.array2d[float],
    counts: wp.array[int],
    eps: float,
    scratch: TerrainFinishScratch,
    order: wp.array2d[int],
    kept: wp.array[int],
):
    """Per list: the candidates into its block of a finish scratch (feature id: the candidate's index), the finish,
    and the kept candidates' indices in candidate order."""
    i = wp.tid()
    base = i * scratch.capacity
    for j in range(counts[i]):
        scratch.q[base + j] = q[i, j]
        scratch.k[base + j] = k[i, j]
        scratch.n[base + j] = n[i, j]
        scratch.d[base + j] = d[i, j]
        scratch.feature[base + j] = j
    kept[i] = terrain_finish_list(scratch, base, counts[i], eps, 0, 1)
    written = int(0)
    for j in range(counts[i]):
        if scratch.mark[base + j] != 0:
            order[i, written] = scratch.feature[base + j]
            written += 1


@wp.kernel(enable_backward=False)
def _report(status: wp.array[wp.int32], overflow: wp.array[wp.int32], flags: int, current: int):
    terrain_report(status, overflow, flags, current != 0)


@wp.kernel(enable_backward=False)
def _feature_ids(
    parts: wp.array[wp.vec4i],
    slots: wp.array[int],
    ids: wp.array[int],
    back: wp.array[wp.vec4i],
    keys: wp.array[wp.int64],
):
    i = wp.tid()
    p = parts[i]
    ids[i] = terrain_feature_id(p[0], p[1], p[2], p[3], slots[i])
    back[i] = terrain_feature_parts(ids[i])
    keys[i] = make_contact_sort_key(5, 77, ids[i], 20)


def _frame(n):
    e1 = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _random_rows(rng):
    """Rows of one pair: plane groups (corners, interior points above or below their hull) and single rows.

    Every decision of the finish has a margin of at least 0.1 mm here, far beyond float32 rounding.
    """
    rows = []
    for _group in range(int(rng.integers(1, 4))):
        n = rng.normal(size=3)
        n[2] = abs(n[2]) + 1.0
        n /= np.linalg.norm(n)
        e1, e2 = _frame(n)
        origin = rng.uniform(-0.3, 0.3, size=3)
        count = int(rng.integers(1, 9))
        uv = rng.uniform(-0.1, 0.1, size=(count, 2))
        depth = rng.uniform(-0.004, 0.01, size=count)
        corners = []
        for (u, v), dd in zip(uv, depth, strict=True):
            q = origin + u * e1 + v * e2
            corners.append(ref.Row(q, q + dd * n, n.copy(), dd, None))
        rows.extend(corners)
        if count >= 3:
            # An interior point: on, above (implied) or clearly below (kept) the corners' affine interpolation.
            w = rng.dirichlet(np.ones(3))
            w = 0.1 + 0.7 * w
            w /= w.sum()
            q = origin + sum(wi * (uv[i, 0] * e1 + uv[i, 1] * e2) for i, wi in enumerate(w))
            base = float(sum(wi * depth[i] for i, wi in enumerate(w)))
            dd = base + float(rng.choice([0.0, 0.002, -0.002]))
            rows.append(ref.Row(q, q + dd * n, n.copy(), dd, None))
    for _single in range(int(rng.integers(0, 4))):
        n = rng.normal(size=3)
        n[2] = abs(n[2]) + 0.5
        n /= np.linalg.norm(n)
        q = rng.uniform(-0.3, 0.3, size=3)
        dd = float(rng.uniform(-0.004, 0.02))
        rows.append(ref.Row(q, q + dd * n, n, dd, None))
    # exact duplicates
    for _dup in range(int(rng.integers(0, 3))):
        src = rows[int(rng.integers(0, len(rows)))]
        rows.append(ref.Row(src.q.copy(), src.k.copy(), src.n.copy(), src.d, None))
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def _run(device, row_sets):
    count = len(row_sets)
    q = np.zeros((count, MAX_ROWS, 3), np.float32)
    k = np.zeros_like(q)
    n = np.zeros_like(q)
    d = np.zeros((count, MAX_ROWS), np.float32)
    counts = np.zeros(count, np.int32)
    for i, rows in enumerate(row_sets):
        counts[i] = len(rows)
        for j, row in enumerate(rows):
            q[i, j], k[i, j], n[i, j], d[i, j] = row.q, row.k, row.n, row.d
    order = wp.zeros((count, MAX_ROWS), dtype=int, device=device)
    kept = wp.zeros(count, dtype=int, device=device)
    wp.launch(
        _finish,
        dim=count,
        inputs=[
            wp.array(q, dtype=wp.vec3, device=device),
            wp.array(k, dtype=wp.vec3, device=device),
            wp.array(n, dtype=wp.vec3, device=device),
            wp.array(d, dtype=float, device=device),
            wp.array(counts, dtype=int, device=device),
            EPS,
            terrain_finish_scratch(device, MAX_ROWS, blocks=count),
        ],
        outputs=[order, kept],
        device=device,
    )
    kept = kept.numpy()
    order = order.numpy()
    return [order[i, : kept[i]].tolist() for i in range(count)]


def test_finish_matches_reference(test, device):
    """Keep the rows the float64 finish keeps: coincident rows merged, implied rows dropped, four per plane group."""
    rng = np.random.default_rng(8)
    row_sets = [_random_rows(rng) for _ in range(300)]
    kept = _run(device, row_sets)
    implied = merged = reduced = 0
    for rows, got in zip(row_sets, kept, strict=True):
        stats = ref.Stats()
        expected = ref.finish(rows, stats=stats)
        implied += stats.implied
        merged += stats.merged
        reduced += stats.reduced
        test.assertEqual(sorted(got), sorted(next(i for i, r in enumerate(rows) if r is e) for e in expected))
        test.assertEqual(got, sorted(got))
    # The cases exercise every rule.
    test.assertGreater(implied, 50)
    test.assertGreater(merged, 50)
    test.assertGreater(reduced, 20)


def test_plane_groups_form_by_tolerance(test, device):
    """Group rows of one plane whose offsets straddle a rounding boundary, and drop the row the others imply.

    Two rows at depth 10 mm and one between them at 12 mm on one plane, the middle one 2e-13 m higher: a
    grouping by rounded offsets (8 decimals) split them and kept all three.
    """
    n = np.array([0.0, 0.0, 1.0])
    rows = []
    for x, depth, z in ((0.0, 0.01, 0.1234567849999), (0.1, 0.01, 0.1234567849999), (0.05, 0.012, 0.1234567850001)):
        q = np.array([x, 0.0, z])
        rows.append(ref.Row(q, q + depth * n, n.copy(), depth, None))
    kept = _run(device, [rows])
    test.assertEqual(kept[0], [0, 1])
    test.assertEqual([next(i for i, r in enumerate(rows) if r is e) for e in ref.finish(rows)], [0, 1])


def _corner_rows(rng, groups, corners):
    """Rows of one pair: ``groups`` plane groups of ``corners`` corners of a convex polygon each (none implied)."""
    rows = []
    for _group in range(groups):
        n = rng.normal(size=3)
        n[2] = abs(n[2]) + 1.0
        n /= np.linalg.norm(n)
        e1, e2 = _frame(n)
        origin = rng.uniform(-0.3, 0.3, size=3)
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=corners))
        for a in angles:
            q = origin + 0.08 * (np.cos(a) * e1 + np.sin(a) * e2)
            dd = float(rng.uniform(-0.004, 0.01))
            rows.append(ref.Row(q, q + dd * n, n.copy(), dd, None))
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def test_finish_keeps_every_plane_group(test, device):
    """Keep every plane group's rows, four at most per group, however many groups and candidates a pair has.

    Five groups of four rows (20 rows) and four groups of six rows (each reduced to four, 16 rows): the float64
    finish keeps the same rows. No count bounds a pair's rows: the finish once kept 16 and 12 of them, each group's
    deepest first, and flagged the rest. A line of 40 equal-depth points (more candidates than its list once held)
    keeps its two ends.
    """
    rng = np.random.default_rng(9)
    for groups, corners in ((5, 4), (4, 6)):
        row_sets = [_corner_rows(rng, groups, corners) for _ in range(40)]
        kept = _run(device, row_sets)
        for rows, got in zip(row_sets, kept, strict=True):
            stats = ref.Stats()
            expected = ref.finish(rows, stats=stats)
            test.assertEqual(len(got), groups * min(corners, 4))
            test.assertEqual(got, sorted(next(i for i, r in enumerate(rows) if r is e) for e in expected))
            test.assertEqual(stats.limited, 0)
    many = [
        ref.Row(np.array([0.01 * i, 0.0, 0.0]), np.array([0.01 * i, 0.0, 0.02]), np.array([0.0, 0.0, 1.0]), 0.02, None)
        for i in range(40)
    ]
    test.assertEqual(_run(device, [many])[0], [0, 39])


def test_feature_ids_fit_the_sort_key(test, device):
    """Pack a row's window-local feature in 19 bits, unique within 64 cells, unchanged by the contact sort key."""
    r, c, kind, sub, slot = np.meshgrid(
        np.arange(64), np.arange(64), np.arange(4), np.arange(4), np.arange(8), indexing="ij"
    )
    parts = np.stack([r, c, kind, sub], -1).reshape(-1, 4) + np.array([128, 64, 0, 0])
    slots = slot.reshape(-1)
    n = len(slots)
    ids = wp.zeros(n, dtype=int, device=device)
    back = wp.zeros(n, dtype=wp.vec4i, device=device)
    keys = wp.zeros(n, dtype=wp.int64, device=device)
    wp.launch(
        _feature_ids,
        dim=n,
        inputs=[
            wp.array(parts.astype(np.int32), dtype=wp.vec4i, device=device),
            wp.array(slots.astype(np.int32), dtype=int, device=device),
        ],
        outputs=[ids, back, keys],
        device=device,
    )
    ids = ids.numpy()
    test.assertEqual(len(np.unique(ids)), n)
    test.assertEqual(int(ids.max()), (1 << TERRAIN_FEATURE_BITS) - 1)
    np.testing.assert_array_equal(back.numpy(), parts % np.array([64, 64, 4, 4]))
    np.testing.assert_array_equal(keys.numpy() & 0x7FFFFF, ids)


def test_terrain_report_raises_at_the_capacity_check(test, device):
    """Raise at the narrow phase's capacity check, by name, after a terrain pair's rows find no room in the spill
    region, its window outgrows its scratch or it reads stale features; the buffer verifier keeps the bit."""
    builder = newton.ModelBuilder()
    builder.add_shape_sphere(builder.add_body(), radius=0.1)
    narrow = newton.CollisionPipeline(builder.finalize(device=device), verify_buffers=True).narrow_phase
    overflow = wp.zeros(2, dtype=wp.int32, device=device)
    for flags, current, name in (
        (int(TERRAIN_FINISH_ROW_OVERFLOW), 1, "terrain_rows"),
        (int(TERRAIN_FINISH_CANDIDATE_OVERFLOW), 1, "terrain_rows"),
        (0, 0, "terrain_stale"),
    ):
        with test.subTest(name=name, flags=flags):
            narrow.buffer_capacity_status(clear=True)
            narrow.check_buffer_capacity()
            wp.launch(_report, dim=1, inputs=[narrow._buffer_capacity_status, overflow, flags, current], device=device)
            test.assertEqual([key for key, value in narrow.buffer_capacity_status().items() if value], [name])
            with test.assertRaisesRegex(RuntimeError, name):
                narrow.check_buffer_capacity()
    test.assertEqual(overflow.numpy().tolist(), [1, 1])
    # The buffer verifier starts from the status word and keeps the bit.
    status = wp.array([int(TERRAIN_STATUS_ROWS)], dtype=wp.int32, device=device)
    zero = wp.zeros(1, dtype=wp.int32, device=device)
    active = wp.zeros(2, dtype=wp.int32, device=device)
    args = [zero, 1, zero, -1, zero, 1, zero, -1, zero, -1, zero, 1, zero, 1, zero, 1, zero, 1, zero, 1, zero, 1]
    wp.launch(record_narrow_phase_buffers, dim=1, inputs=[*args, active, 0, zero, zero, 80, status], device=device)
    test.assertEqual(int(status.numpy()[0]), int(TERRAIN_STATUS_ROWS))


class TestTerrainLawFinish(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _func in (
    ("test_finish_matches_reference", test_finish_matches_reference),
    ("test_finish_keeps_every_plane_group", test_finish_keeps_every_plane_group),
    ("test_plane_groups_form_by_tolerance", test_plane_groups_form_by_tolerance),
    ("test_feature_ids_fit_the_sort_key", test_feature_ids_fit_the_sort_key),
    ("test_terrain_report_raises_at_the_capacity_check", test_terrain_report_raises_at_the_capacity_check),
):
    add_function_test(TestTerrainLawFinish, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

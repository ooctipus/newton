# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's shared per-pair finish and its output layout (feature ids, overflow flags)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import make_contact_sort_key
from newton._src.geometry.narrow_phase import verify_narrow_phase_buffers
from newton._src.geometry.terrain_law import (
    TERRAIN_FEATURE_BITS,
    TERRAIN_FINISH_CANDIDATE_OVERFLOW,
    TERRAIN_FINISH_ROW_OVERFLOW,
    TERRAIN_ROW_CANDIDATES,
    TERRAIN_STATUS_ROWS,
    TerrainRows,
    terrain_feature_id,
    terrain_feature_parts,
    terrain_finish,
    terrain_report,
    terrain_rows_push,
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
    row_limit: int,
    order: wp.array2d[int],
    kept: wp.array[int],
    flags: wp.array[int],
    status: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    i = wp.tid()
    rows = TerrainRows()
    for j in range(counts[i]):
        rows = terrain_rows_push(rows, q[i, j], k[i, j], n[i, j], d[i, j], j)
    selected, count, flag = terrain_finish(rows, eps, row_limit)
    kept[i] = count
    flags[i] = flag
    for j in range(count):
        order[i, j] = rows.feature[selected[j]]
    terrain_report(status, overflow, flag, True)


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
    keys[i] = make_contact_sort_key(5, 77, ids[i])


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


def _run(device, row_sets, row_limit):
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
    order = wp.zeros((count, TERRAIN_ROW_CANDIDATES), dtype=int, device=device)
    kept = wp.zeros(count, dtype=int, device=device)
    flags = wp.zeros(count, dtype=int, device=device)
    status = wp.zeros(1, dtype=wp.int32, device=device)
    overflow = wp.zeros(2, dtype=wp.int32, device=device)
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
            row_limit,
        ],
        outputs=[order, kept, flags, status, overflow],
        device=device,
    )
    kept = kept.numpy()
    order = order.numpy()
    return [order[i, : kept[i]].tolist() for i in range(count)], flags.numpy(), status, overflow


def test_finish_matches_reference(test, device):
    """Keep the rows the float64 finish keeps: coincident rows merged, implied rows dropped, four per plane group."""
    rng = np.random.default_rng(8)
    row_sets = [_random_rows(rng) for _ in range(300)]
    kept, flags, _status, _overflow = _run(device, row_sets, 32)
    implied = merged = reduced = 0
    for rows, got, flag in zip(row_sets, kept, flags, strict=True):
        stats = ref.Stats()
        expected = ref.finish(rows, stats=stats)
        implied += stats.implied
        merged += stats.merged
        reduced += stats.reduced
        test.assertEqual(sorted(got), sorted(next(i for i, r in enumerate(rows) if r is e) for e in expected))
        test.assertEqual(flag, 0)
        test.assertEqual(got, sorted(got))
    # The cases exercise every rule.
    test.assertGreater(implied, 50)
    test.assertGreater(merged, 50)
    test.assertGreater(reduced, 20)


def _overflow_expected(rows, row_limit):
    """The overflow rule, independently: every group's deepest row first, then its farthest, triangle, quadrilateral."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(ref.plane_group(row), []).append(i)
    ranked = []
    for members in groups.values():
        order = ref.reduce4([rows[i].k for i in members], [rows[i].d for i in members], rows[members[0]].n)
        if len(members) <= 4:
            order = _rank4(rows, members)
        for rank, local in enumerate(order[:4]):
            ranked.append((rank, rows[members[local]].d, members[local]))
    ranked.sort()
    return sorted(i for _rank, _d, i in ranked[:row_limit])


def _rank4(rows, members):
    """The order reduce4 would give a group of at most four rows (it returns them unordered)."""
    n = rows[members[0]].n
    P = np.array([rows[i].k - (rows[i].k @ n) * n for i in members])
    h = np.array([rows[i].d for i in members])
    i0 = int(np.argmin(h))
    rest = [j for j in range(len(members)) if j != i0]
    if not rest:
        return [i0]
    i1 = max(rest, key=lambda j: (np.linalg.norm(P[j] - P[i0]), -j))
    rest = [j for j in rest if j != i1]
    if not rest:
        return [i0, i1]
    i2 = max(rest, key=lambda j: (abs(np.cross(P[i1] - P[i0], P[j] - P[i0]) @ n), -j))
    return [i0, i1, i2] + [j for j in rest if j != i2]


def test_finish_overflow_keeps_supports(test, device):
    """On overflow keep every plane group's deepest row first, then its extremes, and raise a sticky status bit."""
    rng = np.random.default_rng(9)
    row_sets = []
    for _ in range(40):
        rows = []
        for _group in range(5):
            n = rng.normal(size=3)
            n[2] = abs(n[2]) + 1.0
            n /= np.linalg.norm(n)
            e1, e2 = _frame(n)
            origin = rng.uniform(-0.3, 0.3, size=3)
            # four corners of a convex quadrilateral: none implied
            angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=4))
            for a in angles:
                q = origin + 0.08 * (np.cos(a) * e1 + np.sin(a) * e2)
                dd = float(rng.uniform(-0.004, 0.01))
                rows.append(ref.Row(q, q + dd * n, n.copy(), dd, None))
        row_sets.append(rows)
    kept, flags, status, overflow = _run(device, row_sets, 16)
    for rows, got, flag in zip(row_sets, kept, flags, strict=True):
        test.assertEqual(len(got), 16)
        test.assertEqual(flag, int(TERRAIN_FINISH_ROW_OVERFLOW))
        test.assertEqual(got, _overflow_expected(rows, 16))
    test.assertEqual(int(status.numpy()[0]), int(TERRAIN_STATUS_ROWS))
    test.assertEqual(overflow.numpy().tolist(), [40, 0])
    # The buffer verifier starts from the status word and keeps the bit.
    zero = wp.zeros(1, dtype=wp.int32, device=device)
    active = wp.zeros(2, dtype=wp.int32, device=device)
    args = [zero, 1, zero, -1, zero, 1, zero, -1, zero, -1, zero, 1, zero, 1, zero, 1, zero, 1, zero, 1, zero, 1]
    wp.launch(verify_narrow_phase_buffers, dim=1, inputs=[*args, active, 0, zero, 80, status], device=device)
    test.assertEqual(int(status.numpy()[0]), int(TERRAIN_STATUS_ROWS))
    # More candidates than the finish holds: the extra ones are counted and flagged.
    many = [
        ref.Row(np.array([0.01 * i, 0.0, 0.0]), np.array([0.01 * i, 0.0, 0.02]), np.array([0.0, 0.0, 1.0]), 0.02, None)
        for i in range(40)
    ]
    kept, flags, status, overflow = _run(device, [many], 8)
    test.assertEqual(flags[0], int(TERRAIN_FINISH_CANDIDATE_OVERFLOW))
    test.assertEqual(overflow.numpy().tolist(), [0, 1])
    # A line of equal-depth points keeps its two ends.
    test.assertEqual(kept[0], [0, TERRAIN_ROW_CANDIDATES - 1])


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
    """Raise at the narrow phase's capacity check, by name, after a terrain pair overflows or reads stale features."""
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


class TestTerrainLawFinish(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _func in (
    ("test_finish_matches_reference", test_finish_matches_reference),
    ("test_finish_overflow_keeps_supports", test_finish_overflow_keeps_supports),
    ("test_feature_ids_fit_the_sort_key", test_feature_ids_fit_the_sort_key),
    ("test_terrain_report_raises_at_the_capacity_check", test_terrain_report_raises_at_the_capacity_check),
):
    add_function_test(TestTerrainLawFinish, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

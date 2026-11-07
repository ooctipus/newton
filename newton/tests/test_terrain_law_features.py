# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's static surface features: plane clusters, edge classes, vertex features, mip, rebuild.

The Warp cluster build must reproduce the float64 reference (:mod:`newton.tests.terrain_law_reference`) exactly,
on every device, keep a planar slope decoded in float32 in one cluster, and give after an edit and a rebuild
exactly what a fresh build gives; stale features must be reported, not read.
"""

import glob
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.geometry import terrain_contacts
from newton._src.geometry.narrow_phase import record_narrow_phase_buffers
from newton._src.geometry.terrain_contacts import TerrainFeatures
from newton._src.geometry.terrain_law import (
    TERRAIN_EDGE_BORDER,
    TERRAIN_EDGE_CONCAVE,
    TERRAIN_EDGE_CONVEX,
    TERRAIN_EDGE_FLAT,
    TERRAIN_FINE_CELLS,
    TERRAIN_STATUS_STALE,
    TERRAIN_TILE_CELLS,
    TERRAIN_VERTEX_FEATURE_BIT,
    TerrainFeatureData,
    terrain_decode,
    terrain_features_current,
    terrain_fine_bound,
    terrain_report,
    terrain_tile_bound,
    terrain_vertex_bits_of_clusters,
    terrain_window,
)
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_test_utils as tu
from newton.tests.unittest_utils import add_function_test, get_test_devices

_CLASSES = {
    "flat": TERRAIN_EDGE_FLAT,
    "convex": TERRAIN_EDGE_CONVEX,
    "concave": TERRAIN_EDGE_CONCAVE,
    "border": TERRAIN_EDGE_BORDER,
}


@wp.kernel(enable_backward=False)
def _stale_check(data: TerrainFeatureData, heightfield: int, status: wp.array[wp.int32], overflow: wp.array[wp.int32]):
    terrain_report(status, overflow, 0, terrain_features_current(data, heightfield))


@wp.kernel(enable_backward=False)
def _tile_bounds(data: TerrainFeatureData, heightfield: int, out: wp.array2d[float]):
    i, j = wp.tid()
    out[i, j] = terrain_tile_bound(data, data.surfaces[heightfield], i, j)


@wp.kernel(enable_backward=False)
def _fine_bounds(data: TerrainFeatureData, heightfield: int, out: wp.array2d[float]):
    i, j = wp.tid()
    out[i, j] = terrain_fine_bound(data, data.surfaces[heightfield], i, j)


# A module of its own with Warp's default contraction, so that its PTX can be checked alone.
@wp.kernel(enable_backward=False, module="unique")
def _decode_heights(elevations: wp.array[wp.float32], min_z: float, max_z: float, out: wp.array[wp.float32]):
    i = wp.tid()
    out[i] = terrain_decode(min_z, max_z, elevations[i])


@wp.kernel(enable_backward=False)
def _count_every_parent(active: wp.array[wp.int32], parent: wp.array[wp.int32], counts: wp.array[wp.int32]):
    wp.atomic_add(counts, 1, 1)


class _StaleParents:
    """The parent array of a replayed launch whose threads run one after another.

    A thread replayed stale loads the parents as the launch found them, apart from its own stores, as on CUDA a
    load issued before another thread's store (or served from a non-coherent cache) may; every store lands in
    replay order.
    """

    def __init__(self, parent):
        self.values = list(parent)
        self._initial = list(parent)
        self.own = None

    def __getitem__(self, i):
        if self.own is None:
            return self.values[i]
        return self.own.get(i, self._initial[i])

    def __setitem__(self, i, value):
        if self.own is not None:
            self.own[i] = value
        self.values[i] = value


def _python_kernel(kernel):
    """Return a kernel's own Python body as ``body(t, *args)``: its Warp functions run as Python, ``wp.tid()`` is t."""
    thread = [0]

    class ThreadScope:
        def __getattr__(self, name):
            return getattr(wp, name)

        def tid(self):
            return thread[0]

    scope = {k: v.func if isinstance(v, wp.Function) else v for k, v in kernel.func.__globals__.items()}
    scope["wp"] = ThreadScope()
    function = types.FunctionType(kernel.func.__code__, scope)

    def body(t, *args):
        thread[0] = t
        function(*args)

    return body


def _replay_compress(body, parent, order, stale):
    """Replay a compression of ``parent`` (every triangle active), thread by thread in ``order``."""
    memory = _StaleParents(parent)
    active = [1] * len(parent)
    for t in order:
        memory.own = {} if t in stale else None
        body(t, active, memory)
    return memory.values


def test_height_decode_rounds_like_the_cpu(test, device):
    """Decode heights with each float32 operation rounded once, as the CPU decode and the float64 reference do.

    The elevations are drawn where one fused multiply-add rounds differently from the product and the sum
    rounded apart (CUDA's default contraction fused them even through a float64 round trip).
    """
    rng = np.random.default_rng(10)
    elevations = rng.uniform(0.0625, 1.0, 4096).astype(np.float32)
    min_z, max_z = np.float32(-1.5), np.float32(0.9)
    out = wp.zeros(len(elevations), dtype=wp.float32, device=device)
    wp.launch(
        _decode_heights,
        dim=len(elevations),
        inputs=[wp.array(elevations, dtype=wp.float32, device=device), float(min_z), float(max_z)],
        outputs=[out],
        device=device,
    )
    expected = ref.decode_heights(elevations, 1, len(elevations), float(min_z), float(max_z)).reshape(-1)
    np.testing.assert_array_equal(out.numpy().astype(np.float64), expected)
    # The premise: the product of two float32 values is exact in float64 and, with min_z = -1.5 and elevations of
    # at least 1/16, so is its sum with min_z; one rounding to float32 is the fused value.
    span = np.float64(np.float32(max_z - min_z))
    fused = (elevations.astype(np.float64) * span + np.float64(min_z)).astype(np.float32)
    test.assertGreater(np.count_nonzero(fused.astype(np.float64) != expected), 500)


def _planar_slope(n: int, kind: str):
    """A planar 12 deg slope on n x n vertices of 0.1 m cells, as raw heights (decoded in float32 by Newton)."""
    x = np.arange(n) * 0.1
    X, Y = np.meshgrid(x, x)
    t = np.tan(np.radians(12.0))
    return {
        "up_x": t * X,
        "down_x": -t * X,
        "down_xy": -t * (0.8 * X + 0.6 * Y),
        "offset": 5.0 + t * X,
    }[kind]


def _assert_matches_reference(test, features, model, *, heightfield=0, shape=0):
    hf = tu.reference_surface(model, heightfield=heightfield)
    ids = ref.cluster_ids(hf.cluster)
    info = features.surface(shape)
    got = features.clusters.numpy()[info.triangle_offset : info.triangle_offset + len(ids)]
    np.testing.assert_array_equal(got, ids)
    planes = features.planes.numpy()[info.plane_offset : info.plane_offset + features.plane_count(shape)]
    records = [hf.records[k] for k in sorted(hf.records)]
    test.assertEqual(len(planes), len(records))
    for plane, record in zip(planes, records, strict=True):
        test.assertEqual(int(plane["anchor"]), record[0])
        test.assertEqual(np.float32(plane["gx"]), record[1])
        test.assertEqual(np.float32(plane["gy"]), record[2])
        test.assertEqual(np.float32(plane["offset"]), record[3])
    return hf


def test_planar_slope_is_one_cluster(test, device):
    """Keep a planar 12 deg slope decoded in float32 in one plane cluster, whatever its orientation and height.

    The float32 decode leaves every triangle up to ~0.24 um off the slope, so the triangles' own planes differ;
    the plane of any single triangle tilts by up to ~2e-6 rad and carries no large cluster (the prototype's
    plane of the lowest-id triangle split the 12.7 m down-slope into thousands of clusters).
    """
    for kind in ("up_x", "down_x", "down_xy", "offset"):
        with test.subTest(kind=kind):
            raw = _planar_slope(128, kind)
            model = tu.heightfield_model(raw, 6.35, 6.35, device=device)
            features = TerrainFeatures(model)
            clusters = features.clusters.numpy()[: 2 * 127 * 127]
            test.assertEqual(len(np.unique(clusters)), 1)
            test.assertEqual(int(clusters[0]), 0)
            test.assertEqual(features.plane_count(0), 1)
            hf = _assert_matches_reference(test, features, model)
            # The premise: the decoded triangles are not one plane on their own.
            normals = np.array([hf.triangle_plane(t)[0] for t in range(0, 2 * 127 * 127, 997)])
            test.assertGreater(np.abs(normals - normals[0]).max(), 1.0e-9)
            # Every vertex lies within 1 um of the cluster plane.
            n, a = hf.plane(0)
            nrow = ncol = 128
            P = np.stack(np.meshgrid(hf.vx(np.arange(ncol)), hf.vy(np.arange(nrow))), -1).reshape(-1, 2)
            dev = np.abs((P[:, 0] - a[0]) * n[0] + (P[:, 1] - a[1]) * n[1] + (hf.Z.reshape(-1) - a[2]) * n[2])
            test.assertLessEqual(dev.max(), ref.FLAT_EPS)
            if kind == "down_x":
                test.assertGreater(len(np.unique(ref.build_prototype_clusters(hf.Z, hf.hx, hf.hy))), 1000)


def test_clusters_match_reference(test, device):
    """Build the clusters of a mosaic of the suite's sub-terrains exactly as the float64 reference does."""
    rng = np.random.default_rng(2)
    for _ in range(2):
        heights = tu.suite_like_heights(rng, 96, 128)
        model = tu.heightfield_model(heights, 6.4, 4.8, device=device)
        features = TerrainFeatures(model)
        _assert_matches_reference(test, features, model)
        again = TerrainFeatures(model)
        np.testing.assert_array_equal(again.clusters.numpy(), features.clusters.numpy())
        np.testing.assert_array_equal(again.planes.numpy(), features.planes.numpy())


def test_compression_check_counts_non_root_parents(test, device):
    """Count the active triangles under a non-root parent, which every round of a build checks, and raise on any.

    The count is read with the round's existing readback; a build whose check counts any triangle raises instead of
    returning clusters split over non-root ids.
    """
    # 1 and 2 under the root 0, 3 under the non-root 2, 4 inactive, 5 alone, 8 under the non-root 7.
    parent = wp.array([0, 0, 0, 2, 3, 5, 6, 6, 7], dtype=wp.int32, device=device)
    active = wp.array([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=wp.int32, device=device)
    counts = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(terrain_contacts._cluster_check_roots, dim=9, inputs=[active, parent, counts], device=device)
    test.assertEqual(counts.numpy().tolist(), [0, 2])
    wp.launch(terrain_contacts._cluster_compress, dim=9, inputs=[active, parent], device=device)
    np.testing.assert_array_equal(parent.numpy(), [0, 0, 0, 0, 3, 5, 6, 6, 6])
    counts.zero_()
    wp.launch(terrain_contacts._cluster_check_roots, dim=9, inputs=[active, parent, counts], device=device)
    test.assertEqual(counts.numpy().tolist(), [0, 0])
    model = tu.heightfield_model(np.zeros((3, 3)), 0.1, 0.1, device=device, min_z=0.0, max_z=1.0)
    with mock.patch.object(terrain_contacts, "_cluster_check_roots", _count_every_parent):
        with test.assertRaisesRegex(RuntimeError, "not their component's root"):
            TerrainFeatures(model)


def test_edge_classes_and_vertex_features_match_reference(test, device):
    """Derive every edge class and vertex feature in float32 on demand as the float64 reference does.

    On the CPU every decision matches. Under CUDA's default contraction the decisions within float32 resolution
    of their threshold may differ: a crease whose far vertex lies within 0.1 um of the other cluster's plane (the
    fixture has 26, three of them at exactly 0), or a triple product within a factor of two of the span. Either
    class or ownership there moves no row by more than that.
    """
    rng = np.random.default_rng(3)
    heights = tu.suite_like_heights(rng, 64, 64)
    model = tu.heightfield_model(heights, 3.2, 3.2, device=device)
    features = TerrainFeatures(model)
    hf = tu.reference_surface(model)
    queries, expected_class, expected_feature, clear = [], [], [], []
    for r in range(hf.nrow):
        for c in range(hf.ncol):
            span = hf.vertex_span(r, c)
            for which, kind in enumerate(("H", "V", "D")):
                if (
                    (kind == "H" and c >= hf.cols)
                    or (kind == "V" and r >= hf.rows)
                    or (kind == "D" and (r >= hf.rows or c >= hf.cols))
                ):
                    continue
                window = hf.window_features(
                    (min(r, hf.rows - 1), min(r, hf.rows - 1), min(c, hf.cols - 1), min(c, hf.cols - 1))
                )
                edge = next(e for e in window[1] if e[0] == kind and e[1] == r and e[2] == c)
                queries.append((r, c, which))
                expected_class.append(_CLASSES[hf.edge_class(edge)])
                expected_feature.append(int(span > ref.VERTEX_SPAN))
                margin = np.inf
                if expected_class[-1] in (TERRAIN_EDGE_CONVEX, TERRAIN_EDGE_CONCAVE):
                    n, a = hf.plane(edge[5])
                    margin = abs(float(n @ (hf.V(*edge[8]) - a)))
                clear.append(margin >= 1.0e-7 and not 0.5 * ref.VERTEX_SPAN < span < 2.0 * ref.VERTEX_SPAN)
    answers = tu.feature_queries(features, 0, queries)
    expected_class = np.asarray(expected_class)
    expected_feature = np.asarray(expected_feature)
    clear = np.asarray(clear)
    test.assertLess(np.count_nonzero(~clear), 0.03 * len(clear))
    if device.is_cpu:
        clear[:] = True
    np.testing.assert_array_equal(answers[clear, 0], expected_class[clear])
    np.testing.assert_array_equal(answers[clear, 1], expected_feature[clear])
    test.assertGreater(np.count_nonzero(expected_class == TERRAIN_EDGE_CONVEX), 1000)
    test.assertGreater(np.count_nonzero(expected_class == TERRAIN_EDGE_CONCAVE), 1000)
    test.assertGreater(np.count_nonzero(expected_class == TERRAIN_EDGE_FLAT), 1000)
    test.assertGreater(np.count_nonzero(expected_feature), 500)


def test_weak_convex_vertex_keeps_its_support(test, device):
    """Keep the support of a box corner over a convex vertex whose incident normals lie 3e-4 rad apart.

    The apex of a shallow pyramid has a normal cone 6e-4 rad wide and a triple product of about 4e-7. No face or
    edge claims a support whose normal lies inside that cone, so the vertex must be a feature in float64 and in
    float32. At a span of 1e-6 it was not, and a box corner 5 mm above it had no row there.
    """
    slope = 3.0e-4
    x = (np.arange(9) - 4) * 0.1
    X, Y = np.meshgrid(x, x)
    model = tu.heightfield_model(0.05 - slope * (np.abs(X) + np.abs(Y)), 0.4, 0.4, device=device)
    features = TerrainFeatures(model)
    hf = tu.reference_surface(model)
    test.assertTrue(hf.vertex_feature(4, 4))
    test.assertLess(hf.vertex_span(4, 4), 1.0e-6)
    up = np.array([0.0, 0.0, 1.0])
    answers = tu.feature_queries(features, 0, [(4, 4, 0)], directions=[up])
    test.assertEqual(answers[0, 1:].tolist(), [1, 1])
    q = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7), np.float64)
    dims = {"half": (0.08, 0.06, 0.04)}
    corner = ref.make_core("box", dims, np.array([0.0, 0.0, 0.0, *q])).lowest(up)
    gap = 0.005
    core = ref.make_core("box", dims, np.array([*(hf.V(4, 4) + gap * up - corner), *q]))
    rows = ref.pair_rows(hf, core, 0.04)
    deepest = min(rows, key=lambda row: row.d)
    test.assertAlmostEqual(deepest.d, gap, delta=1.0e-9)
    test.assertLess(float(np.linalg.norm(deepest.q - hf.V(4, 4))), 1.0e-9)
    test.assertEqual(deepest.feat, ("V", 4, 4))


def test_tile_max_bounds_every_height(test, device):
    """Hold the largest decoded height of every 8 x 8 cell tile and of every 2 x 2 cell block of the fine mip, and a
    bound above every height of each."""
    rng = np.random.default_rng(4)
    heights = tu.suite_like_heights(rng, 70, 45)
    model = tu.heightfield_model(heights, 2.25, 3.5, device=device)
    features = TerrainFeatures(model)
    hf = tu.reference_surface(model)
    info = features.surface(0)
    tile_rows = -(-hf.rows // TERRAIN_TILE_CELLS)
    tile_cols = -(-hf.cols // TERRAIN_TILE_CELLS)
    test.assertEqual(info.tile_cols, tile_cols)
    got = features.tile_max.numpy()[: tile_rows * tile_cols].reshape(tile_rows, tile_cols)
    bounds = wp.zeros((tile_rows, tile_cols), dtype=float, device=device)
    wp.launch(_tile_bounds, dim=(tile_rows, tile_cols), inputs=[features.data(), 0], outputs=[bounds], device=device)
    bounds = bounds.numpy()
    for i in range(tile_rows):
        for j in range(tile_cols):
            block = hf.Z[i * 8 : min(i * 8 + 8, hf.rows) + 1, j * 8 : min(j * 8 + 8, hf.cols) + 1]
            test.assertEqual(np.float32(got[i, j]), np.float32(block.max()))
            test.assertGreater(bounds[i, j], block.max())
    # the fine mip: every 2 x 2 cell block
    fine_rows = -(-hf.rows // TERRAIN_FINE_CELLS)
    fine_cols = -(-hf.cols // TERRAIN_FINE_CELLS)
    test.assertEqual(info.fine_cols, fine_cols)
    got = features.fine_max.numpy()[: fine_rows * fine_cols].reshape(fine_rows, fine_cols)
    bounds = wp.zeros((fine_rows, fine_cols), dtype=float, device=device)
    wp.launch(_fine_bounds, dim=(fine_rows, fine_cols), inputs=[features.data(), 0], outputs=[bounds], device=device)
    bounds = bounds.numpy()
    for i in range(fine_rows):
        for j in range(fine_cols):
            block = hf.Z[i * 2 : min(i * 2 + 2, hf.rows) + 1, j * 2 : min(j * 2 + 2, hf.cols) + 1]
            test.assertEqual(np.float32(got[i, j]), np.float32(block.max()))
            test.assertGreater(bounds[i, j], block.max())


@wp.kernel(enable_backward=False)
def _bits_of_clusters(data: TerrainFeatureData, heightfield: int, out: wp.array2d[int]):
    """Every vertex's bits as its clusters and heights give them, in the window of vertex (0, 0)."""
    r, c = wp.tid()
    out[r, c] = terrain_vertex_bits_of_clusters(data, terrain_window(data, heightfield, 0, 0), r, c)


def test_vertex_bits_match_their_clusters(test, device):
    """Record every vertex's edge kinds and feature bit as its clusters and heights give them, whatever the window.

    The build computes a vertex's bits in its own window; here they are computed again in another (the feature test
    reads cluster normals, which no window changes) and must be equal, bit for bit. Every edge code (crease, flat,
    border, a crease concave in every window frame) and both feature answers occur.
    """
    rng = np.random.default_rng(9)
    heights = tu.suite_like_heights(rng, 64, 64)
    model = tu.heightfield_model(heights, 3.2, 3.2, device=device)
    features = TerrainFeatures(model)
    info = features.surface(features.heightfield_shapes[0])
    out = wp.zeros((info.nrow, info.ncol), dtype=int, device=device)
    wp.launch(_bits_of_clusters, dim=(info.nrow, info.ncol), inputs=[features.data(), 0, out], device=device)
    stored = features.vertex_bits.numpy()[info.data_offset : info.data_offset + info.nrow * info.ncol]
    stored = stored.reshape(info.nrow, info.ncol).astype(np.int64)
    np.testing.assert_array_equal(stored, out.numpy())
    codes = np.concatenate([((stored >> (2 * which)) & 3).ravel() for which in range(3)])
    test.assertEqual(set(np.unique(codes).tolist()), {0, 1, 2, 3})
    feature = (stored & int(TERRAIN_VERTEX_FEATURE_BIT)) != 0
    test.assertTrue(np.any(feature) and not np.all(feature))


def test_rebuild_after_edit_equals_fresh_build(test, device):
    """Report stale features after an edit, and rebuild them to exactly what a fresh build of the new heights gives.

    The edits raise a bump in a flat tile (splitting a large cluster far beyond the edited cells), flatten a rough
    patch and tilt part of a slope, on the second of two heightfields so that the first must stay as it was.
    """
    rng = np.random.default_rng(6)
    first = tu.suite_like_heights(rng, 40, 40)
    second = tu.suite_like_heights(rng, 64, 96)
    model = tu.heightfield_model(
        first, 2.0, 2.0, device=device, extra=((second, 4.8, 3.2, wp.transform((20.0, 0.0, 0.0), wp.quat_identity())),)
    )
    features = TerrainFeatures(model)
    before_first = features.clusters.numpy()[: 2 * 40 * 40].copy()
    shape = features.heightfield_shapes[1]
    info = features.surface(shape)
    elevations = model.heightfield_elevations.numpy()
    grid = elevations[info.data_offset : info.data_offset + info.nrow * info.ncol].reshape(info.nrow, info.ncol).copy()
    edits = (
        ((4, 9, 4, 9), lambda g: g + 0.05),
        ((40, 50, 10, 20), lambda g: np.full_like(g, g.mean())),
        ((30, 60, 70, 90), lambda g: g + 0.001 * np.arange(g.shape[1])[None, :]),
    )
    for region, change in edits:
        r0, r1, c0, c1 = region
        grid[r0:r1, c0:c1] = change(grid[r0:r1, c0:c1])
        elevations[info.data_offset : info.data_offset + info.nrow * info.ncol] = grid.reshape(-1)
        model.heightfield_elevations.assign(elevations)
        features.mark_elevations_changed(shape, region=region)
        with test.assertRaises(RuntimeError):
            features.check_current()
        features.rebuild_terrain_features(shape)
        features.check_current()
        _assert_equals_fresh_build(test, features, model)
        _assert_matches_reference(test, features, model, heightfield=1, shape=shape)
    np.testing.assert_array_equal(features.clusters.numpy()[: 2 * 40 * 40], before_first)


def _assert_equals_fresh_build(test, features, model):
    fresh = TerrainFeatures(model)
    np.testing.assert_array_equal(features.clusters.numpy(), fresh.clusters.numpy())
    np.testing.assert_array_equal(features.tile_max.numpy(), fresh.tile_max.numpy())
    np.testing.assert_array_equal(features.fine_max.numpy(), fresh.fine_max.numpy())
    np.testing.assert_array_equal(features.vertex_bits.numpy(), fresh.vertex_bits.numpy())
    for s in features.heightfield_shapes:
        a, b = features.surface(s), fresh.surface(s)
        test.assertEqual(features.plane_count(s), fresh.plane_count(s))
        np.testing.assert_array_equal(
            features.planes.numpy()[a.plane_offset : a.plane_offset + features.plane_count(s)],
            fresh.planes.numpy()[b.plane_offset : b.plane_offset + fresh.plane_count(s)],
        )


def test_rebuild_covers_every_reported_edit(test, device):
    """Rebuild every edit reported since the last rebuild, also when the rebuild names another region.

    A patch raised to the top of the height range and reported, then another patch lowered and reported, then a
    rebuild given only the second region: the tiles over the first patch must not keep their old maxima while the
    version stamps say current (a pair's tile-max exit would drop every pair over it).
    """
    rng = np.random.default_rng(16)
    model = tu.heightfield_model(tu.suite_like_heights(rng, 64, 64), 3.2, 3.2, device=device)
    features = TerrainFeatures(model)
    shape = features.heightfield_shapes[0]
    info = features.surface(shape)
    elevations = model.heightfield_elevations.numpy()
    grid = elevations[: info.nrow * info.ncol].reshape(info.nrow, info.ncol).copy()
    for region, value in (((4, 9, 4, 9), 1.0), ((40, 45, 40, 45), 0.0)):
        r0, r1, c0, c1 = region
        grid[r0:r1, c0:c1] = value
        model.heightfield_elevations.assign(grid.reshape(-1).astype(np.float32))
        features.mark_elevations_changed(shape, region=region)
    features.rebuild_terrain_features(shape, region=(40, 45, 40, 45))
    features.check_current()
    _assert_equals_fresh_build(test, features, model)


def test_stale_features_are_reported(test, device):
    """Set a sticky status bit when a kernel reads features older than the heights; the buffer verifier keeps it."""
    rng = np.random.default_rng(7)
    model = tu.heightfield_model(tu.suite_like_heights(rng, 16, 16), 0.8, 0.8, device=device)
    features = TerrainFeatures(model)
    shape = features.heightfield_shapes[0]
    status = wp.zeros(1, dtype=wp.int32, device=device)
    overflow = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(_stale_check, dim=1, inputs=[features.data(), 0, status, overflow], device=device)
    test.assertEqual(int(status.numpy()[0]), 0)
    features.mark_elevations_changed(shape)
    wp.launch(_stale_check, dim=1, inputs=[features.data(), 0, status, overflow], device=device)
    test.assertEqual(int(status.numpy()[0]), int(TERRAIN_STATUS_STALE))
    # record_narrow_phase_buffers starts from the status word and keeps the bit.
    zero = wp.zeros(1, dtype=wp.int32, device=device)
    active = wp.zeros(2, dtype=wp.int32, device=device)
    args = [zero, 1, zero, -1, zero, 1, zero, -1, zero, -1, zero, 1, zero, 1, zero, 1, zero, 1, zero, 1, zero, 1]
    wp.launch(record_narrow_phase_buffers, dim=1, inputs=[*args, active, 0, zero, zero, 80, status], device=device)
    test.assertEqual(int(status.numpy()[0]), int(TERRAIN_STATUS_STALE))
    features.rebuild_terrain_features(shape)
    status.zero_()
    wp.launch(_stale_check, dim=1, inputs=[features.data(), 0, status, overflow], device=device)
    test.assertEqual(int(status.numpy()[0]), 0)


def test_reallocated_plane_records_are_reported(test, device):
    """Report a struct that predates a rebuild which reallocated the plane records as stale, not current.

    A flat heightfield has one plane record; an edit into 2 x 2-cell treads of distinct heights outgrows the
    record capacity, so the rebuild moves the records to a new array while the cluster ids index into it. A struct
    fetched earlier (one baked into a captured graph) still points at the old array.
    """
    model = tu.heightfield_model(np.zeros((65, 65)), 3.2, 3.2, device=device, min_z=0.0, max_z=1.0)
    features = TerrainFeatures(model)
    shape = features.heightfield_shapes[0]
    before, layout = features.data(), features.layout_version
    r, c = np.meshgrid(np.arange(65), np.arange(65), indexing="ij")
    treads = ((r // 2) * 7 + (c // 2) * 3) % 50 / 50.0
    model.heightfield_elevations.assign(treads.reshape(-1).astype(np.float32))
    features.mark_elevations_changed(shape)
    features.rebuild_terrain_features(shape)
    test.assertGreater(features.layout_version, layout)
    test.assertGreater(features.plane_count(shape), 1000)
    _assert_equals_fresh_build(test, features, model)
    status = wp.zeros(1, dtype=wp.int32, device=device)
    overflow = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(_stale_check, dim=1, inputs=[before, 0, status, overflow], device=device)
    test.assertEqual(int(status.numpy()[0]), int(TERRAIN_STATUS_STALE))
    status.zero_()
    wp.launch(_stale_check, dim=1, inputs=[features.data(), 0, status, overflow], device=device)
    test.assertEqual(int(status.numpy()[0]), 0)


class TestTerrainLawFeatures(unittest.TestCase):
    def test_build_and_decode_compile_without_contraction_cpu(self):
        """Compile the height decode and the cluster build to PTX without fused multiply-adds (NVRTC, any host).

        CUDA's default contraction fuses a float32 product and sum into one rounding, even through a float64 round
        trip, which the compiler narrows back to float32: the decode must not fuse in a module that contracts,
        and the build module must not fuse at all, so that a build gives the CPU's bits on every device.
        """
        if 86 not in wp.get_cuda_supported_archs():
            self.skipTest("NVRTC cannot compile for sm_86 here")
        with tempfile.TemporaryDirectory() as directory:
            wp.compile_aot_module(_decode_heights.module, arch=86, module_dir=directory, use_ptx=True)
            wp.compile_aot_module(terrain_contacts, arch=86, module_dir=directory, use_ptx=True)
            ptx = {}
            for path in glob.glob(os.path.join(directory, "*.ptx")):
                with open(path, encoding="latin-1") as f:
                    ptx["build" if "terrain_contacts" in os.path.basename(path) else "decode"] = f.read()
        self.assertEqual(sorted(ptx), ["build", "decode"])
        counts = {(name, op): text.count(op) for name, text in ptx.items() for op in ("fma.rn", "mul.rn", "add.rn")}
        self.assertEqual(counts["decode", "fma.rn"], 0, msg=counts)
        self.assertGreater(counts["decode", "mul.rn"], 0, msg=counts)
        self.assertGreater(counts["decode", "add.rn"], 0, msg=counts)
        self.assertEqual(counts["build", "fma.rn"], 0, msg=counts)
        self.assertGreater(counts["build", "mul.rn"], 0, msg=counts)

    def test_compression_leaves_roots_in_any_thread_order_cpu(self):
        """Replay the cluster compression in adversarial thread orders: every parent it stores is a root.

        With path halving, the compression stored into other triangles' slots, and on CUDA a halving store computed
        from a triangle's old parent landed after that triangle's thread stored its root, leaving a non-root parent
        (53 to 88 triangles of a 12 deg slope). Warp's CPU backend runs the threads one after another in id order,
        so no CPU build shows it. The replay runs the kernel's own Python body thread by thread in any order, each
        thread's loads current or as the launch found them.
        """
        compress = _python_kernel(terrain_contacts._cluster_compress)

        def halving(t, active, parent):
            # The compression as it was: the union pass's find, which halves paths.
            if active[t] != 0:
                parent[t] = terrain_contacts._find.func(parent, t)

        chain = [0, 0, 1, 2, 3, 4, 5]
        # Thread 4 stores its root, then thread 6, whose loads predate that store, halves slot 4 back to 2.
        self.assertEqual(_replay_compress(halving, chain, range(7), stale={6}), [0, 0, 0, 0, 2, 0, 0])
        self.assertEqual(_replay_compress(compress, chain, range(7), stale={6}), [0] * 7)
        rng = np.random.default_rng(12)
        forest = [int(rng.integers(0, t + 1)) for t in range(40)]
        roots = []
        for t in range(40):
            root = t
            while forest[root] != root:
                root = forest[root]
            roots.append(root)
        broken = 0
        for _ in range(300):
            order = rng.permutation(40).tolist()
            stale = set(np.flatnonzero(rng.random(40) < 0.5).tolist())
            self.assertEqual(_replay_compress(compress, forest, order, stale), roots)
            old = _replay_compress(halving, forest, order, stale)
            broken += any(old[old[t]] != old[t] for t in range(40))
        # The premise: these orders break the halving compression.
        self.assertGreater(broken, 30)

    def test_region_and_shape_arguments_cpu(self):
        """Refuse a shape that is no heightfield and a region outside the grid."""
        model = tu.heightfield_model(np.zeros((5, 5)), 0.2, 0.2, device="cpu")
        features = TerrainFeatures(model)
        with self.assertRaises(ValueError):
            features.rebuild_terrain_features(7)
        with self.assertRaises(ValueError):
            features.mark_elevations_changed(0, region=(0, 6, 0, 2))
        with self.assertRaises(TypeError):
            features.mark_elevations_changed(0, (1, 3, 1, 3))
        with self.assertRaises(TypeError):
            features.rebuild_terrain_features(0, (1, 3, 1, 3))
        features.mark_elevations_changed(0, region=(1, 3, 1, 3))
        features.rebuild_terrain_features(0)
        features.check_current()
        self.assertEqual(features.memory_bytes()["clusters"], 2 * 4 * 4 * 4)


devices = get_test_devices()
for _name, _func in (
    ("test_height_decode_rounds_like_the_cpu", test_height_decode_rounds_like_the_cpu),
    ("test_planar_slope_is_one_cluster", test_planar_slope_is_one_cluster),
    ("test_clusters_match_reference", test_clusters_match_reference),
    ("test_compression_check_counts_non_root_parents", test_compression_check_counts_non_root_parents),
    ("test_edge_classes_and_vertex_features_match_reference", test_edge_classes_and_vertex_features_match_reference),
    ("test_weak_convex_vertex_keeps_its_support", test_weak_convex_vertex_keeps_its_support),
    ("test_vertex_bits_match_their_clusters", test_vertex_bits_match_their_clusters),
    ("test_tile_max_bounds_every_height", test_tile_max_bounds_every_height),
    ("test_rebuild_after_edit_equals_fresh_build", test_rebuild_after_edit_equals_fresh_build),
    ("test_rebuild_covers_every_reported_edit", test_rebuild_covers_every_reported_edit),
    ("test_stale_features_are_reported", test_stale_features_are_reported),
    ("test_reallocated_plane_records_are_reported", test_reallocated_plane_records_are_reported),
):
    add_function_test(TestTerrainLawFeatures, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's static surface features: plane clusters, edge classes, vertex features, mip, rebuild.

The Warp cluster build must reproduce the float64 reference (:mod:`newton.tests.terrain_law_reference`) exactly,
keep a planar slope decoded in float32 in one cluster, and give after an edit and a rebuild exactly what a fresh
build gives; stale features must be reported, not read.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.narrow_phase import verify_narrow_phase_buffers
from newton._src.geometry.terrain_contacts import TerrainFeatures
from newton._src.geometry.terrain_law import (
    TERRAIN_STATUS_STALE,
    TERRAIN_TILE_CELLS,
    TerrainFeatureData,
    terrain_features_current,
    terrain_report,
    terrain_tile_bound,
)
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_test_utils as tu
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _stale_check(data: TerrainFeatureData, heightfield: int, status: wp.array[wp.int32], overflow: wp.array[wp.int32]):
    terrain_report(status, overflow, 0, terrain_features_current(data, heightfield))


@wp.kernel(enable_backward=False)
def _tile_bounds(data: TerrainFeatureData, heightfield: int, out: wp.array2d[float]):
    i, j = wp.tid()
    out[i, j] = terrain_tile_bound(data, data.surfaces[heightfield], i, j)


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


def _assert_matches_reference(test, features, model, heightfield=0, shape=0):
    hf = tu.reference_surface(model, heightfield)
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


def test_tile_max_bounds_every_height(test, device):
    """Hold the largest decoded height of every 8 x 8 cell tile, and a bound above every height of the tile."""
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
        features.mark_elevations_changed(shape, region)
        with test.assertRaises(RuntimeError):
            features.check_current()
        features.rebuild_terrain_features(shape)
        features.check_current()
        fresh = TerrainFeatures(model)
        np.testing.assert_array_equal(features.clusters.numpy(), fresh.clusters.numpy())
        np.testing.assert_array_equal(features.tile_max.numpy(), fresh.tile_max.numpy())
        for s in features.heightfield_shapes:
            a, b = features.surface(s), fresh.surface(s)
            test.assertEqual(features.plane_count(s), fresh.plane_count(s))
            np.testing.assert_array_equal(
                features.planes.numpy()[a.plane_offset : a.plane_offset + features.plane_count(s)],
                fresh.planes.numpy()[b.plane_offset : b.plane_offset + fresh.plane_count(s)],
            )
        _assert_matches_reference(test, features, model, heightfield=1, shape=shape)
    np.testing.assert_array_equal(features.clusters.numpy()[: 2 * 40 * 40], before_first)


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
    # verify_narrow_phase_buffers starts from the status word and keeps the bit.
    zero = wp.zeros(1, dtype=wp.int32, device=device)
    active = wp.zeros(2, dtype=wp.int32, device=device)
    args = [zero, 1, zero, -1, zero, 1, zero, -1, zero, -1, zero, 1, zero, 1, zero, 1, zero, 1, zero, 1, zero, 1]
    wp.launch(verify_narrow_phase_buffers, dim=1, inputs=[*args, active, 0, zero, 80, status], device=device)
    test.assertEqual(int(status.numpy()[0]), int(TERRAIN_STATUS_STALE))
    features.rebuild_terrain_features(shape)
    status.zero_()
    wp.launch(_stale_check, dim=1, inputs=[features.data(), 0, status, overflow], device=device)
    test.assertEqual(int(status.numpy()[0]), 0)


class TestTerrainLawFeatures(unittest.TestCase):
    def test_region_and_shape_arguments_cpu(self):
        """Refuse a shape that is no heightfield and a region outside the grid."""
        model = tu.heightfield_model(np.zeros((5, 5)), 0.2, 0.2, device="cpu")
        features = TerrainFeatures(model)
        with self.assertRaises(ValueError):
            features.rebuild_terrain_features(7)
        with self.assertRaises(ValueError):
            features.mark_elevations_changed(0, (0, 6, 0, 2))
        features.mark_elevations_changed(0, (1, 3, 1, 3))
        features.rebuild_terrain_features(0)
        features.check_current()
        self.assertEqual(features.memory_bytes()["clusters"], 2 * 4 * 4 * 4)


devices = get_test_devices()
for _name, _func in (
    ("test_planar_slope_is_one_cluster", test_planar_slope_is_one_cluster),
    ("test_clusters_match_reference", test_clusters_match_reference),
    ("test_tile_max_bounds_every_height", test_tile_max_bounds_every_height),
    ("test_rebuild_after_edit_equals_fresh_build", test_rebuild_after_edit_equals_fresh_build),
    ("test_stale_features_are_reported", test_stale_features_are_reported),
):
    add_function_test(TestTerrainLawFeatures, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

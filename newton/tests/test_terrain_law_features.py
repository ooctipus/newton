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
import unittest

import numpy as np
import warp as wp

from newton._src.geometry import terrain_contacts
from newton._src.geometry.narrow_phase import verify_narrow_phase_buffers
from newton._src.geometry.terrain_contacts import TerrainFeatures
from newton._src.geometry.terrain_law import (
    TERRAIN_EDGE_BORDER,
    TERRAIN_EDGE_CONCAVE,
    TERRAIN_EDGE_CONVEX,
    TERRAIN_EDGE_FLAT,
    TERRAIN_STATUS_STALE,
    TERRAIN_TILE_CELLS,
    TerrainFeatureData,
    terrain_decode,
    terrain_features_current,
    terrain_report,
    terrain_tile_bound,
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


# A module of its own with Warp's default contraction, so that its PTX can be checked alone.
@wp.kernel(enable_backward=False, module="unique")
def _decode_heights(elevations: wp.array[wp.float32], min_z: float, max_z: float, out: wp.array[wp.float32]):
    i = wp.tid()
    out[i] = terrain_decode(min_z, max_z, elevations[i])


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


def test_edge_classes_and_vertex_features_match_reference(test, device):
    """Derive every edge class and vertex feature in float32 on demand as the float64 reference does."""
    rng = np.random.default_rng(3)
    heights = tu.suite_like_heights(rng, 64, 64)
    model = tu.heightfield_model(heights, 3.2, 3.2, device=device)
    features = TerrainFeatures(model)
    hf = tu.reference_surface(model)
    queries, expected_class, expected_feature = [], [], []
    for r in range(hf.nrow):
        for c in range(hf.ncol):
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
                expected_feature.append(int(hf.vertex_feature(r, c)))
    answers = tu.feature_queries(features, 0, queries)
    np.testing.assert_array_equal(answers[:, 0], np.asarray(expected_class))
    np.testing.assert_array_equal(answers[:, 1], np.asarray(expected_feature))
    test.assertGreater(sum(k == TERRAIN_EDGE_CONVEX for k in expected_class), 1000)
    test.assertGreater(sum(k == TERRAIN_EDGE_CONCAVE for k in expected_class), 1000)
    test.assertGreater(sum(k == TERRAIN_EDGE_FLAT for k in expected_class), 1000)
    test.assertGreater(sum(expected_feature), 500)


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
    ("test_edge_classes_and_vertex_features_match_reference", test_edge_classes_and_vertex_features_match_reference),
    ("test_tile_max_bounds_every_height", test_tile_max_bounds_every_height),
    ("test_rebuild_after_edit_equals_fresh_build", test_rebuild_after_edit_equals_fresh_build),
    ("test_rebuild_covers_every_reported_edit", test_rebuild_covers_every_reported_edit),
    ("test_stale_features_are_reported", test_stale_features_are_reported),
    ("test_reallocated_plane_records_are_reported", test_reallocated_plane_records_are_reported),
):
    add_function_test(TestTerrainLawFeatures, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

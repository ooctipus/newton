# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direction cells: a small convex hull's directional table for support queries.

A hull of 4 to 64 vertices carries, per direction cell (6 cube faces of 16 x 16
cells), a mask of every vertex that can be the float32 support of a direction
in that cell. A support query scans only those candidates in vertex order, so
it returns the full scan's vertex, bit for bit.
"""

import unittest
from fractions import Fraction

import numpy as np
import warp as wp

import newton
from newton._src.geometry.support_function import (
    _CONVEX_SUPPORT_CELLS,
    _CONVEX_SUPPORT_RESOLUTION_MASK,
    CONVEX_SUPPORT_CELL_COUNT,
    AcceleratedSupportMapDataProvider,
    GenericShapeData,
    SupportMapDataProvider,
    _direction_cell,
    pack_mesh_ptr,
    support_map,
    support_map_accelerated,
    support_map_lean_accelerated,
)
from newton._src.geometry.utils import remesh_convex_hull
from newton.tests.unittest_utils import add_function_test, get_test_devices

CUBE = np.array([[x, y, z] for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)], np.float32)


def cell_boundary_rays():
    """Cell corners, their adjacent float32 values, face ties and both signs."""
    grid = np.linspace(-1, 1, 17, dtype=np.float32)
    rays = []
    for axis in range(3):
        other = [k for k in range(3) if k != axis]
        for sign in (-1, 1):
            for x in grid:
                for y in grid:
                    for shift in (-np.inf, None, np.inf):
                        ray = np.zeros(3, np.float32)
                        ray[axis] = sign
                        ray[other[0]] = x if shift is None else np.nextafter(x, np.float32(shift))
                        ray[other[1]] = y
                        rays.append(ray)
    return np.asarray(rays)


def _sphere_points(count, radius, seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return (points * radius).astype(np.float32)


def _hull_model(device, point_sets, scales=None):
    builder = newton.ModelBuilder()
    body = builder.add_body()
    for i, points in enumerate(point_sets):
        vertices, faces = remesh_convex_hull(points)
        mesh = newton.Mesh(vertices, np.asarray(faces, dtype=np.int32).reshape(-1), compute_inertia=False)
        builder.add_shape_convex_hull(
            body,
            mesh=mesh,
            scale=(1.0, 1.0, 1.0) if scales is None else scales[i],
            xform=wp.transform(wp.vec3(float(i), 0.0, 0.0), wp.quat_identity()),
        )
    return builder.finalize(device=device)


@wp.kernel
def _cell_check(rays: wp.array[wp.vec3], output: wp.array[int]):
    i = wp.tid()
    output[i] = _direction_cell(rays[i])


@wp.kernel
def _support_pairs(
    shape_index: int,
    shape_source: wp.array[wp.uint64],
    shape_scale: wp.array[wp.vec3],
    shape_support_data: wp.array[wp.vec4i],
    support_lut: wp.array[int],
    rays: wp.array[wp.vec3],
    cells: wp.array[wp.vec3],
    scan: wp.array[wp.vec3],
    cells_lean: wp.array[wp.vec3],
):
    i = wp.tid()
    geom = GenericShapeData()
    geom.shape_type = int(newton.GeoType.CONVEX_MESH)
    geom.scale = shape_scale[shape_index]
    geom.auxiliary = pack_mesh_ptr(shape_source[shape_index])
    geom.shape_index = shape_index
    provider = AcceleratedSupportMapDataProvider()
    provider.shape_support_data = shape_support_data
    provider.support_lut = support_lut
    cells[i] = support_map_accelerated(geom, rays[i], provider)
    cells_lean[i] = support_map_lean_accelerated(geom, rays[i], provider)
    # Without the tables the query scans every vertex.
    scan[i] = support_map(geom, rays[i], SupportMapDataProvider())


def _rays(seed):
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(4096, 3)).astype(np.float32)
    axes = np.concatenate([np.eye(3), -np.eye(3), [[1, 1, 0], [1, -1, 0], [0, 1, 1], [1, 1, 1], [-1, 1, -1]]])
    return np.concatenate([cell_boundary_rays(), random, axes.astype(np.float32)])


def test_cell_support_matches_scan(test, device):
    """Every hull with a cell table returns the full scan's support vertex, bit for bit."""
    point_sets = [CUBE, _sphere_points(12, 0.05, 1), _sphere_points(40, 0.1, 2), _sphere_points(120, 0.05, 3)]
    scales = [(1.0, 1.0, 1.0), (2.0, 0.5, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
    model = _hull_model(device, point_sets, scales)
    support_data = model._shape_support_data.numpy()
    flags = support_data[:, 3]
    cell_hulls = [i for i in range(model.shape_count) if flags[i] > 0 and flags[i] & int(_CONVEX_SUPPORT_CELLS)]
    # The cube and the hulls of at most 64 vertices take cells; larger ones keep the scan.
    vertex_counts = [len(model.shape_source[i].vertices) for i in range(model.shape_count)]
    for i, count in enumerate(vertex_counts):
        test.assertEqual(i in cell_hulls, count <= 64, f"hull {i} with {count} vertices")
        test.assertEqual(int(flags[i]) & int(_CONVEX_SUPPORT_RESOLUTION_MASK), 0)
    rays = wp.array(_rays(7), dtype=wp.vec3, device=device)
    for shape in range(model.shape_count):
        out = [wp.zeros(rays.shape[0], dtype=wp.vec3, device=device) for _ in range(3)]
        wp.launch(
            _support_pairs,
            dim=rays.shape[0],
            inputs=[
                shape,
                model.shape_source_ptr,
                model.shape_scale,
                model._shape_support_data,
                model._convex_support_lut,
                rays,
                *out,
            ],
            device=device,
        )
        cells, scan, cells_lean = (a.numpy() for a in out)
        np.testing.assert_array_equal(cells, scan, err_msg=f"hull {shape}")
        np.testing.assert_array_equal(cells_lean, scan, err_msg=f"hull {shape} (lean)")


def test_cell_table_layout(test, device):
    """A cell hull's table holds one nonzero 64-bit mask per cell, as two int32 words."""
    model = _hull_model(device, [CUBE])
    start = int(model._shape_support_data.numpy()[0][0])
    words = model._convex_support_lut.numpy()[start : start + 2 * CONVEX_SUPPORT_CELL_COUNT]
    masks = words.astype("<i4").view("<u8")
    test.assertEqual(len(masks), CONVEX_SUPPORT_CELL_COUNT)
    test.assertTrue(np.all(masks != 0))
    test.assertTrue(np.all(masks < (1 << len(CUBE))))
    # The +x face's center cell can only be supported by the four +x vertices.
    center = 0 * 256 + 8 * 16 + 8
    candidates = [i for i in range(len(CUBE)) if int(masks[center]) >> i & 1]
    test.assertTrue(all(CUBE[i][0] > 0 for i in candidates))


class TestConvexCells(unittest.TestCase):
    def test_cell_rectangle_intersections(self):
        """Match exact clipping for touching, degenerate and oblique convex regions."""
        from newton._src.geometry.convex_cells_build import DELTA, _clip, _covered_cells  # noqa: PLC0415

        square = [
            (Fraction(-1), Fraction(-1)),
            (Fraction(1), Fraction(-1)),
            (Fraction(1), Fraction(1)),
            (Fraction(-1), Fraction(1)),
        ]
        regions = [
            square,
            [(Fraction(0), Fraction(0))],
            [(Fraction(-1, 8), Fraction(0)), (Fraction(1, 8), Fraction(0))],
            [(Fraction(0), Fraction(-1, 8)), (Fraction(0), Fraction(1, 8))],
            [(Fraction(-1, 8), Fraction(-1, 8)), (Fraction(1, 8), Fraction(1, 8))],
        ]
        rng = np.random.default_rng(438)
        for _ in range(12):
            poly = square
            for _ in range(8):
                a, b = (int(v) for v in rng.integers(-9, 10, 2))
                poly = _clip(poly, a, b, Fraction(int(rng.integers(0, 10)), 8))
            if poly:
                regions.append(poly)
        regions.extend([[(x + DELTA, y - DELTA) for x, y in p] for p in regions[:5]])
        for poly in regions:
            expected = []
            for x in range(16):
                for y in range(16):
                    clipped = _clip(poly, 1, 0, -(Fraction(x, 8) - 1 - DELTA))
                    clipped = _clip(clipped, -1, 0, Fraction(x + 1, 8) - 1 + DELTA)
                    clipped = _clip(clipped, 0, 1, -(Fraction(y, 8) - 1 - DELTA))
                    clipped = _clip(clipped, 0, -1, Fraction(y + 1, 8) - 1 + DELTA)
                    if clipped:
                        expected.append((x, y))
            self.assertEqual(list(_covered_cells(poly)), expected)

    def test_direction_range_cpu(self):
        """Nonfinite, zero and extreme directions leave the cells for the scan."""
        rays = np.array(
            [
                [0, 0, 0],
                [np.inf, 0, 0],
                [np.nan, 1, 0],
                [2**-61, 0, 0],
                [2**61, 0, 0],
                [2**-60, 0, 0],
                [2**60, 0, 0],
                [1, 1e-40, 0],
            ],
            np.float32,
        )
        output = wp.zeros(len(rays), dtype=int, device="cpu")
        wp.launch(_cell_check, len(rays), inputs=[wp.array(rays, dtype=wp.vec3, device="cpu"), output], device="cpu")
        np.testing.assert_array_equal(output.numpy()[:5], -np.ones(5, dtype=int))
        self.assertTrue(np.all(output.numpy()[5:] >= 0))

    def test_unsupported_hulls_keep_the_scan(self):
        """Hulls past the cells' coordinate range carry no cell table."""
        model = _hull_model("cpu", [CUBE * 4.0])
        flags = int(model._shape_support_data.numpy()[0][3])
        self.assertEqual(flags & int(_CONVEX_SUPPORT_CELLS), 0)


devices = get_test_devices()
add_function_test(TestConvexCells, "test_cell_support_matches_scan", test_cell_support_matches_scan, devices=devices)
add_function_test(TestConvexCells, "test_cell_table_layout", test_cell_table_layout, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain cells of a static mesh: the lattice, the cell classes, the corners and the mixed CSR.

Every case builds a small mesh the way terrain generators do (heightfield patches as two triangles per cell, boxes as
closed triangle soups) and checks the classes against the construction, then checks the whole table against every
triangle with :func:`verify_terrain_mesh_cells`.
"""

import unittest

import numpy as np

from newton._src.geometry import terrain_cells_mesh as tcm
from newton.tests.unittest_utils import add_function_test, get_test_devices

H = 0.1


def _heightfield_mesh(heights, origin=(0.0, 0.0), diagonal=0):
    """Two triangles per cell, as Isaac Lab's ``convert_height_field_to_mesh`` writes them (x along rows of heights)."""
    nx, ny = heights.shape
    xs = origin[0] + H * np.arange(nx)
    ys = origin[1] + H * np.arange(ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack([xx.ravel(), yy.ravel(), heights.ravel()], 1).astype(np.float32)
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00, v01 = i * ny + j, i * ny + j + 1  # v01: +y
            v10, v11 = v00 + ny, v01 + ny  # v10: +x
            if diagonal == 0:
                tris += [(v00, v11, v01), (v00, v10, v11)]
            else:
                tris += [(v00, v10, v01), (v10, v11, v01)]
    return vertices, np.asarray(tris, dtype=np.int32)


def _box(x0, x1, y0, y1, z0, z1):
    """A closed box as 12 outward triangles."""
    p = np.array(
        [[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=np.float32
    )  # index = x + 2 y + 4 z
    quads = [
        (0, 2, 3, 1),  # bottom (down)
        (4, 5, 7, 6),  # top (up)
        (0, 1, 5, 4),  # y0 side
        (2, 6, 7, 3),  # y1 side
        (0, 4, 6, 2),  # x0 side
        (1, 3, 7, 5),  # x1 side
    ]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    return p, np.asarray(tris, dtype=np.int32)


def _merge(*meshes):
    vs, fs, off = [], [], 0
    for v, f in meshes:
        vs.append(v)
        fs.append(f + off)
        off += v.shape[0]
    return np.concatenate(vs), np.concatenate(fs)


def _ground(nx, ny, z=0.0, origin=(0.0, 0.0)):
    return _heightfield_mesh(np.full((nx + 1, ny + 1), z, dtype=np.float32), origin)


def _cell(table, x, y):
    (x0, y0), (dx, dy), (_rows, cols) = table.origin, table.cell_size, table.shape
    return int(np.floor((y - y0) / dy)) * cols + int(np.floor((x - x0) / dx))


class TestTerrainCellsMesh(unittest.TestCase):
    def _verified(self, v, f, **kwargs):
        table = tcm.build_terrain_mesh_cells(v, f, **kwargs)
        report = tcm.verify_terrain_mesh_cells(table, v, f)
        self.assertEqual(report["failed"], 0, report)
        self.assertEqual(report["csr_misses"], 0, report)
        return table, report

    def test_lattice_from_edges(self):
        rng = np.random.default_rng(1)
        v, f = _heightfield_mesh(rng.uniform(0.0, 0.1, (7, 5)).astype(np.float32), origin=(-3.3, 12.4))
        lat = tcm.terrain_mesh_lattice(v, f)
        self.assertAlmostEqual(lat["x"]["size"], H, places=6)
        self.assertAlmostEqual(lat["y"]["size"], H, places=6)
        self.assertEqual(lat["x"]["vertex_share"], 1.0)
        self.assertEqual(lat["y"]["vertex_share"], 1.0)

    def test_heightfield_mesh_is_grid_cells_with_its_heights(self):
        """A converted heightfield is the special case: grid cells, the heightfield's diagonal, corners = heights."""
        rng = np.random.default_rng(2)
        heights = rng.uniform(-0.2, 0.3, (6, 5)).astype(np.float32)
        v, f = _heightfield_mesh(heights, origin=(-35.4, 60.1))
        table, report = self._verified(v, f)
        self.assertEqual(table.shape, (4, 5))
        self.assertEqual(table.counts()["grid"], 20)
        self.assertFalse(np.any(table.cell_flags & tcm.TERRAIN_CELL_DIAGONAL))
        z = table.cell_z.reshape(4, 5, 4)
        np.testing.assert_array_equal(z[:, :, 0], heights[:-1, :-1].T)
        np.testing.assert_array_equal(z[:, :, 1], heights[1:, :-1].T)
        np.testing.assert_array_equal(z[:, :, 2], heights[:-1, 1:].T)
        np.testing.assert_array_equal(z[:, :, 3], heights[1:, 1:].T)
        # the special case: the shared vertex grid is the height grid, and no cell needs a record
        np.testing.assert_array_equal(table.vertex_z, heights.T)
        self.assertEqual(table.record_z.shape[0], 0)
        inner = (table.cell_flags.reshape(4, 5)[1:-1, 1:-1] >> 4) & 0xF
        self.assertFalse(np.any(inner))
        self.assertEqual(report["reproduced"], f.shape[0])
        self.assertLessEqual(report["max_height_error_ulp"], 4.0)

    def test_planar_cells_copy_their_vertex_heights(self):
        """Quantized heights (exact zeros, planar cells, ramps): every corner is its mesh vertex's height bit for bit.

        A planar cell's plane evaluated at a corner of its other triangle rounds (a height of 0 became 5.6e-17, a
        float32 height its neighbour), which gave a heightfield's cells records of their own.
        """
        rng = np.random.default_rng(6)
        heights = (np.round(rng.uniform(0.0, 0.1, (12, 9)) / 0.005) * 0.005).astype(np.float32)
        heights[:4, :4] = 0.0
        heights[6:, :] = (0.0137 * np.arange(9, dtype=np.float32))[None, :] + np.float32(0.03)
        v, f = _heightfield_mesh(heights, origin=(-1.6, -1.6))
        table, report = self._verified(v, f)
        np.testing.assert_array_equal(table.vertex_z, heights.T)
        self.assertEqual(table.record_z.shape[0], 0)
        self.assertLessEqual(report["max_height_error_ulp"], 2.0)

    def test_other_diagonal(self):
        rng = np.random.default_rng(3)
        v, f = _heightfield_mesh(rng.uniform(0.0, 0.3, (5, 5)).astype(np.float32), diagonal=1)
        table, report = self._verified(v, f)
        self.assertEqual(table.counts()["grid"], 16)
        self.assertTrue(np.all(table.cell_flags & tcm.TERRAIN_CELL_DIAGONAL))
        self.assertEqual(report["reproduced"], f.shape[0])

    def test_stairs_on_lattice_lines_are_grid_cells_with_edge_risers(self):
        """Risers on lattice lines: each side is a grid cell, the riser is the edge between them."""
        v, f = _merge(
            _box(0.0, 0.3, 0.0, 0.4, -1.0, 0.0),
            _box(0.3, 0.6, 0.0, 0.4, -1.0, 0.15),
            _box(0.6, 0.9, 0.0, 0.4, -1.0, 0.3),
        )
        table, report = self._verified(v, f, cell_size=(H, H), origin=(0.0, 0.0))
        self.assertEqual(table.counts(), {"grid": 36, "step": 0, "mixed": 0, "empty": 0})
        c = _cell(table, 0.25, 0.15)
        self.assertTrue(table.cell_flags[c] & tcm.TERRAIN_CELL_EDGE_X1)
        self.assertTrue(table.cell_flags[c + 1] & tcm.TERRAIN_CELL_EDGE_X0)
        self.assertFalse(table.cell_flags[c] & tcm.TERRAIN_CELL_EDGE_X0)
        np.testing.assert_array_equal(table.cell_z[c + 1], np.float32(0.15))
        self.assertEqual(report["in_mixed"], 0)
        self.assertGreater(table.stats["wall_checks"]["edge_ok"], 0)
        self.assertAlmostEqual(table.stats["bottoms"]["min_thickness"], 1.0, places=6)

    def test_riser_inside_a_cell_is_a_step(self):
        """Boxes 0.45 m wide put risers at quarter cells: step cells with their offset and levels."""
        for axis in (0, 1):
            with self.subTest(axis=axis):
                lo = (0.0, 0.025, 0.0, 0.4, -1.0, 0.1)
                hi = (0.025, 0.4, 0.0, 0.4, -1.0, 0.2)
                if axis == 1:
                    lo = (0.0, 0.4, 0.0, 0.025, -1.0, 0.1)
                    hi = (0.0, 0.4, 0.025, 0.4, -1.0, 0.2)
                bx = _box(-0.4, 0.0, 0.0, 0.4, -1.0, 0.1)
                if axis == 1:
                    bx = _box(0.0, 0.4, -0.4, 0.0, -1.0, 0.1)
                v, f = _merge(bx, _box(*lo), _box(*hi))
                table, report = self._verified(v, f, cell_size=(H, H), origin=(0.0, 0.0))
                counts = table.counts()
                self.assertEqual(counts["step"], 4, counts)
                self.assertEqual(counts["mixed"], 0, counts)
                step = np.flatnonzero(table.kind == tcm.TERRAIN_CELL_STEP)
                expect = [0.25, 0.0] if axis == 0 else [0.0, 0.25]
                np.testing.assert_allclose(table.offsets()[step], np.tile(expect, (4, 1)), atol=1e-6)
                z = table.cell_z[step[0]]
                np.testing.assert_array_equal(z[0], np.float32(0.1))
                np.testing.assert_array_equal(z[3], np.float32(0.2))
                zs, _kind = table.surface_z(
                    np.array([0.01, 0.04, 0.025]) if axis == 0 else np.array([0.1, 0.1, 0.1]),
                    np.array([0.1, 0.1, 0.1]) if axis == 0 else np.array([0.01, 0.04, 0.025]),
                )
                np.testing.assert_array_equal(zs, np.float32([0.1, 0.2, 0.2]))
                self.assertEqual(report["in_mixed"], 0)

    def test_shared_riser_rounded_apart_is_one_riser(self):
        """Two boxes' shared side rounded one float32 ulp apart (as the rough terrain's boxes are) is one riser."""
        x = np.float32(-35.475)
        x_next = np.nextafter(x, np.float32(1.0))
        v, f = _merge(_box(-35.6, float(x), 0.0, 0.4, -1.0, 0.1), _box(float(x_next), -35.2, 0.0, 0.4, -1.0, 0.2))
        table, report = self._verified(v, f, cell_size=(H, H), origin=(0.0, 0.0))
        self.assertEqual(table.counts()["mixed"], 0, table.counts())
        self.assertEqual(table.counts()["step"], 4)
        self.assertLessEqual(report["max_lattice_snap"], 1.0e-5)

    def test_two_crossing_risers_are_a_step(self):
        """Four boxes meeting inside a cell (a box grid's corner): two risers cross, four quadrant levels."""
        v, f = _merge(
            _box(-0.1, 0.025, -0.1, 0.075, -1.0, 0.0),
            _box(0.025, 0.2, -0.1, 0.075, -1.0, 0.05),
            _box(-0.1, 0.025, 0.075, 0.2, -1.0, 0.02),
            _box(0.025, 0.2, 0.075, 0.2, -1.0, 0.1),
        )
        table, report = self._verified(v, f, cell_size=(H, H), origin=(0.0, 0.0))
        corner = _cell(table, 0.03, 0.08)
        self.assertEqual(table.kind[corner], tcm.TERRAIN_CELL_STEP)
        np.testing.assert_allclose(table.offsets()[corner], [0.25, 0.75], atol=1e-6)
        np.testing.assert_array_equal(table.cell_z[corner], np.float32([0.0, 0.05, 0.02, 0.1]))
        zs, _ = table.surface_z(np.array([0.01, 0.05, 0.01, 0.05, 0.025]), np.array([0.05, 0.05, 0.09, 0.09, 0.05]))
        np.testing.assert_array_equal(zs, np.float32([0.0, 0.05, 0.02, 0.1, 0.05]))
        self.assertEqual(table.counts()["mixed"], 0)
        self.assertEqual(report["in_mixed"], 0)
        self.assertGreater(table.stats["wall_checks"]["interior_riser"], 0)

    def test_box_over_ground_buries_it(self):
        """Ground under a box's whole cells is inside the box: buried, the box top is the surface."""
        v, f = _merge(_ground(4, 4, origin=(-0.2, -0.2)), _box(0.0, 0.3, 0.0, 0.3, -0.5, 0.1))
        table, report = self._verified(v, f)
        self.assertEqual(report["buried_up_facing"], 8)
        np.testing.assert_array_equal(table.cell_z[_cell(table, 0.15, 0.15)], np.float32(0.1))
        self.assertEqual(table.counts()["mixed"], 0)

    def test_two_parallel_risers_in_one_cell_are_mixed(self):
        """Three levels across one cell: the cell keeps the mesh's triangles."""
        v, f = _merge(
            _box(0.0, 0.02, 0.0, 0.1, -1.0, 0.1),
            _box(0.02, 0.05, 0.0, 0.1, -1.0, 0.2),
            _box(0.05, 0.1, 0.0, 0.1, -1.0, 0.3),
        )
        table, report = self._verified(v, f, cell_size=(H, H), origin=(0.0, 0.0))
        self.assertEqual(table.kind[0], tcm.TERRAIN_CELL_MIXED)
        k = int(table.cell_flags[0] >> tcm.TERRAIN_CELL_PAYLOAD_SHIFT)
        tris = table.mixed_triangles[table.mixed_start[k] : table.mixed_start[k + 1]]
        self.assertEqual(tris.size, f.shape[0])  # with the side walls on its edges, beside a mixed cell
        np.testing.assert_array_equal(table.cell_z[0], np.float32(0.3))
        self.assertEqual(table.mixed_z[k, 1], np.float32(0.3))
        self.assertLessEqual(table.mixed_z[k, 0], -1.0 + 1e-6)
        self.assertEqual(report["in_mixed"], f.shape[0])

    def test_overhang_and_fin_are_mixed(self):
        """A plate floating over the ground (two surfaces over a column) and a wall standing above both sides."""
        plate = _box(0.1, 0.2, 0.1, 0.2, 0.3, 0.35)
        fin_v = np.array([[0.2, 0.0, 0.0], [0.2, 0.1, 0.0], [0.2, 0.1, 0.5], [0.2, 0.0, 0.5]], dtype=np.float32)
        fin_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        v, f = _merge(_ground(4, 4), plate, (fin_v, fin_f))
        table, _ = self._verified(v, f)
        self.assertEqual(table.kind[_cell(table, 0.15, 0.15)], tcm.TERRAIN_CELL_MIXED)
        self.assertEqual(table.kind[_cell(table, 0.15, 0.05)], tcm.TERRAIN_CELL_MIXED)
        self.assertEqual(table.kind[_cell(table, 0.25, 0.05)], tcm.TERRAIN_CELL_MIXED)
        self.assertEqual(table.kind[_cell(table, 0.35, 0.35)], tcm.TERRAIN_CELL_GRID)
        self.assertGreater(table.stats["wall_checks"]["edge_bad"], 0)

    def test_hole_is_empty_and_its_edges_are_risers(self):
        v, f = _ground(3, 3)
        a = v[f]
        centre = np.all((np.abs(a[:, :, 0] - 0.15) < 0.051) & (np.abs(a[:, :, 1] - 0.15) < 0.051), axis=1)
        table, _ = self._verified(v, f[~centre])
        middle = _cell(table, 0.15, 0.15)
        self.assertEqual(table.kind[middle], tcm.TERRAIN_CELL_EMPTY)
        self.assertTrue(table.cell_flags[middle - 1] & tcm.TERRAIN_CELL_EDGE_X1)

    def test_qualifying_rule(self):
        rng = np.random.default_rng(4)
        v, f = _heightfield_mesh(rng.uniform(0.0, 0.1, (8, 8)).astype(np.float32))
        table = tcm.build_terrain_mesh_cells(v, f)
        ok, why = tcm.terrain_mesh_cells_qualify(table, f.shape[0])
        self.assertTrue(ok, why)
        # an unstructured soup: its lattice holds few of its vertices
        pts = rng.uniform(0.0, 1.0, (60, 3)).astype(np.float32)
        pts[:, 2] *= 0.1
        tri = np.array([(i, i + 1, i + 2) for i in range(0, 57, 3)], dtype=np.int32)
        upward = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])[:, 2] < 0
        tri[upward] = tri[upward][:, [0, 2, 1]]
        pts[0, :2] = pts[1, :2] + [0.37, 0.0]
        soup = tcm.build_terrain_mesh_cells(pts, tri, cell_size=(0.37, 0.37), origin=(0.0, 0.0))
        ok, why = tcm.terrain_mesh_cells_qualify(soup, tri.shape[0])
        self.assertFalse(ok, why)


def test_devices_build_one_table(test, device):
    """Every device runs the same kernels without floating-point contraction: the same table, bit for bit."""
    rng = np.random.default_rng(5)
    meshes = [
        (_heightfield_mesh(rng.uniform(-0.2, 0.3, (6, 5)).astype(np.float32), origin=(-35.4, 60.1)), {}),
        (
            _merge(
                _box(-0.1, 0.025, -0.1, 0.075, -1.0, 0.0),
                _box(0.025, 0.2, -0.1, 0.075, -1.0, 0.05),
                _box(-0.1, 0.025, 0.075, 0.2, -1.0, 0.02),
                _box(0.025, 0.2, 0.075, 0.2, -1.0, 0.1),
                _box(0.2, 0.22, -0.1, 0.2, -1.0, 0.3),
            ),
            {"cell_size": (H, H), "origin": (0.0, 0.0)},
        ),
    ]
    for (v, f), kwargs in meshes:
        cpu = tcm.build_terrain_mesh_cells(v, f, device="cpu", **kwargs)
        here = tcm.build_terrain_mesh_cells(v, f, device=device, **kwargs)
        for name in (
            "vertex_z",
            "cell_flags",
            "record_z",
            "record_offsets",
            "mixed_cell",
            "mixed_start",
            "mixed_triangles",
            "mixed_z",
        ):
            np.testing.assert_array_equal(getattr(here, name), getattr(cpu, name), err_msg=name)
        test.assertEqual(here.origin, cpu.origin)
        test.assertEqual(here.cell_size, cpu.cell_size)
    test.assertGreater(cpu.counts()["mixed"], 0)
    test.assertGreater(cpu.counts()["step"], 0)


add_function_test(
    TestTerrainCellsMesh, "test_devices_build_one_table", test_devices_build_one_table, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

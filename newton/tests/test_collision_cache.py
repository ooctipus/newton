# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the disk cache of the collision tables a model's finalize derives from each mesh."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.geometry import _collision_cache


def _terrain(cells: int = 12) -> newton.Mesh:
    """A rough open height sheet of ``cells`` x ``cells`` quads: boundary edges, flat and tilted triangles."""
    rng = np.random.default_rng(3)
    x, y = np.meshgrid(np.arange(cells + 1, dtype=np.float32), np.arange(cells + 1, dtype=np.float32), indexing="ij")
    z = np.round(rng.random(x.shape) * 4.0).astype(np.float32) * np.float32(0.05)
    vertices = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    base = (np.arange(cells)[:, None] * (cells + 1) + np.arange(cells)[None, :]).ravel()
    quads = np.stack((base, base + cells + 1, base + cells + 2, base + 1), axis=1)
    indices = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]])).astype(np.int32).reshape(-1)
    return newton.Mesh(vertices, indices, compute_inertia=False)


def _collision_arrays(model) -> dict[str, np.ndarray]:
    """The model's mesh-derived collision tables (structured arrays field by field)."""
    owners = (
        (
            model,
            "",
            (
                "_mesh_boundary_flags",
                "_mesh_edge_order",
                "_mesh_edge_clusters",
                "_mesh_edge_cluster_reach",
                "_shape_mesh_properties",
                "_convex_support_lut",
            ),
        ),
        (model._mesh_cells, "cells.", None),
        (model._mesh_topology.data, "topology.", None),
    )
    arrays = {}
    for owner, prefix, names in owners:
        for name in names or [name for name in vars(owner) if isinstance(getattr(owner, name), wp.array)]:
            data = getattr(owner, name).numpy()
            if data.dtype.names is None:
                arrays[prefix + name] = data
            else:
                arrays.update({f"{prefix}{name}.{field}": data[field] for field in data.dtype.names})
    return arrays


def _finalize(cache_dir):
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=_terrain())
    body = builder.add_body(xform=wp.transform((3.0, 3.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
    hull = newton.Mesh.create_box(0.2, 0.2, 0.2, duplicate_vertices=False, compute_normals=False, compute_uvs=False)
    builder.add_shape_convex_hull(builder.add_body(xform=wp.transform((5.0, 5.0, 1.0), wp.quat_identity())), mesh=hull)
    saved = newton.config.cache_dir_collision
    newton.config.cache_dir_collision = cache_dir
    try:
        return builder.finalize(device="cpu")
    finally:
        newton.config.cache_dir_collision = saved


class TestCollisionCache(unittest.TestCase):
    def test_round_trip(self):
        """Each table layout comes back with its dtypes, structured fields, NumPy scalars and plain values."""
        record = np.zeros(3, dtype=[("key", "<u8"), ("box", "<f4", (3,))])
        record["key"] = (1, 2, 3)
        tables = {
            "array": np.arange(5, dtype=np.int16),
            "tuple": (np.ones(2, np.float32), record),
            "dict": {"cell": np.float32(0.25), "levels": 3, "rows": record, "origin": np.zeros(3, np.float64)},
        }
        with tempfile.TemporaryDirectory() as directory:
            saved = newton.config.cache_dir_collision
            newton.config.cache_dir_collision = directory
            try:
                for kind, table in tables.items():
                    stored = _collision_cache.cached(kind, (np.arange(3), 1.5), lambda table=table: table)
                    loaded = _collision_cache.cached(kind, (np.arange(3), 1.5), lambda: self.fail("not cached"))
                    self.assertIs(stored, table)
                    if kind == "dict":
                        self.assertEqual(loaded.keys(), table.keys())
                        pairs = [(loaded[name], table[name]) for name in table]
                    else:
                        pairs = list(zip(loaded, table, strict=True)) if kind == "tuple" else [(loaded, table)]
                    for got, want in pairs:
                        self.assertEqual(type(got), type(want))
                        np.testing.assert_array_equal(got, want)
                        if isinstance(want, (np.ndarray, np.generic)):
                            self.assertEqual(got.dtype, want.dtype)
            finally:
                newton.config.cache_dir_collision = saved

    def test_finalize_loads_equal_tables(self):
        """A finalize that fills the cache and one that loads it build the tables an uncached finalize builds."""
        reference = _collision_arrays(_finalize(None))
        with tempfile.TemporaryDirectory() as directory:
            filled = _collision_arrays(_finalize(directory))
            entries = sorted(path.name.split(".")[1] for path in Path(directory).rglob("*.npz"))
            loaded = _collision_arrays(_finalize(directory))
        self.assertEqual(
            sorted(set(entries)),
            ["boundary_flags", "convex_support_cells", "edge_clusters", "mesh_grid", "mesh_topology", "watertight"],
        )
        self.assertTrue(any(name.startswith("cells.") for name in reference))
        for tables in (filled, loaded):
            self.assertEqual(tables.keys(), reference.keys())
            for name, data in reference.items():
                np.testing.assert_array_equal(tables[name], data, err_msg=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)

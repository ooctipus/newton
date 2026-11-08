# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that the terrain law reads a static mesh's cell table as its one surface type.

A static triangle mesh whose cell table the law reads whole (:func:`terrain_mesh_surface`) is a surface like a
heightfield: the law writes its partners' contacts and the narrow phase drops those pairs. The same surface given as a
heightfield and as its triangle mesh must give the same rows, within the float32 rounding of the lattice the mesh's
float32 vertices give (the mesh table's origin and cell size are fitted in float64 from them). A mesh the law cannot
read whole (risers, no lattice) keeps the generic mesh path, and its reason is recorded.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.narrow_phase import TERRAIN_LAW_MESH_SURFACE, TERRAIN_LAW_PARTNER
from newton._src.geometry.terrain_contacts import TerrainFeatures, TerrainSurfaces
from newton.tests.unittest_utils import add_function_test, get_test_devices

PIPELINE = {"rigid_contact_max": 1 << 14, "max_triangle_pairs": 1 << 16}
N = 33
HX = 1.6


def _heights(seed=6):
    rng = np.random.default_rng(seed)
    return (np.round(rng.uniform(0.0, 0.1, (N, N)) / 0.005) * 0.005).astype(np.float32)


def _decoded(heights):
    """The heights the law reads for a heightfield of ``heights`` (decoded once, as the narrow phase decodes)."""
    builder = newton.ModelBuilder()
    builder.add_shape_heightfield(heightfield=newton.Heightfield(data=heights, nrow=N, ncol=N, hx=HX, hy=HX))
    model = builder.finalize(device="cpu")
    return TerrainFeatures(model).heights.numpy()[: N * N].reshape(N, N)


def _grid_mesh(z, hx, hy):
    """Two triangles per cell on a heightfield's diagonal, vertices at its lattice positions (shape frame)."""
    nrow, ncol = z.shape
    xs = (np.float64(-hx) + np.arange(ncol) * (2.0 * hx / (ncol - 1))).astype(np.float32)
    ys = (np.float64(-hy) + np.arange(nrow) * (2.0 * hy / (nrow - 1))).astype(np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.stack([xx.ravel(), yy.ravel(), z.ravel()], 1).astype(np.float32)
    tris = []
    for r in range(nrow - 1):
        for c in range(ncol - 1):
            p00, p10 = r * ncol + c, r * ncol + c + 1
            p01, p11 = p00 + ncol, p10 + ncol
            tris += [(p00, p10, p11), (p00, p11, p01)]
    return vertices, np.asarray(tris, np.int32)


def _stairs_mesh():
    """Two stair treads with a riser on a lattice line, as closed boxes (the law cannot read risers yet)."""

    def box(x0, x1, y0, y1, z0, z1):
        p = np.array([[x, y, zz] for zz in (z0, z1) for y in (y0, y1) for x in (x0, x1)], np.float32)
        quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        return p, np.asarray([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], np.int32)

    v0, f0 = box(-1.0, 0.0, -1.0, 1.0, -1.0, 0.0)
    v1, f1 = box(0.0, 1.0, -1.0, 1.0, -1.0, 0.1)
    return np.concatenate([v0, v1]), np.concatenate([f0, f1 + len(v0)])


def _scene(device, *, mesh: bool, heights=None, surface=None, seed=6):
    """The surface under spheres, capsules, boxes and cylinders, each body in its own world."""
    rng = np.random.default_rng(seed)
    heights = _heights(seed) if heights is None else heights
    z = _decoded(heights)
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
    if surface is not None:
        vertices, triangles = surface
        terrain = builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices, triangles, compute_inertia=False), cfg=cfg)
    elif mesh:
        vertices, triangles = _grid_mesh(z, HX, HX)
        terrain = builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices, triangles, compute_inertia=False), cfg=cfg)
    else:
        terrain = builder.add_shape_heightfield(
            heightfield=newton.Heightfield(data=heights, nrow=N, ncol=N, hx=HX, hy=HX), cfg=cfg
        )
    dx = 2.0 * HX / (N - 1)
    for i in range(24):
        builder.begin_world()
        x, y = rng.uniform(-1.2, 1.2, 2)
        c, r = int((x + HX) / dx), int((y + HX) / dx)
        z0 = float(z[r : r + 2, c : c + 2].max())
        q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(size=3))), float(rng.uniform(0.0, 3.0)))
        body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z0 + rng.uniform(-0.01, 0.05)), q))
        kind = i % 4
        if kind == 0:
            builder.add_shape_sphere(body, radius=0.03, cfg=cfg)
        elif kind == 1:
            builder.add_shape_capsule(body, radius=0.025, half_height=0.06, cfg=cfg)
        elif kind == 2:
            builder.add_shape_box(body, hx=0.05, hy=0.04, hz=0.03, cfg=cfg)
        else:
            builder.add_shape_cylinder(body, radius=0.03, half_height=0.04, cfg=cfg)
        builder.end_world()
    return builder.finalize(device=device), terrain


def _law_rows(model):
    pipeline = newton.CollisionPipeline(model, **PIPELINE)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    law = pipeline._terrain_law_shadow
    return pipeline, contacts, (law.rows() if law is not None else None)


def test_mesh_surface_rows_match_heightfield(test, device):
    """The same surface as a heightfield and as its static triangle mesh: the law takes both, with the same rows."""
    model_h, _ = _scene(device, mesh=False)
    model_m, terrain = _scene(device, mesh=True)
    surfaces = TerrainSurfaces(model_m)
    test.assertEqual(surfaces.mesh_shapes, [terrain], surfaces.mesh_reasons)
    test.assertEqual(int(surfaces.shape_surface[terrain]), 0)
    table = surfaces.tables[0]
    np.testing.assert_allclose(table.origin, (-HX, -HX), atol=1.0e-6)
    np.testing.assert_allclose(table.cell_size, (2.0 * HX / (N - 1),) * 2, rtol=1.0e-6)
    np.testing.assert_array_equal(table.vertex_z, _decoded(_heights()))
    _pipe_h, contacts_h, rows_h = _law_rows(model_h)
    pipe_m, contacts_m, rows_m = _law_rows(model_m)
    test.assertTrue(pipe_m._terrain_law_writes)
    test.assertEqual(len(rows_m["pairs"]), len(rows_h["pairs"]))
    np.testing.assert_array_equal(rows_m["pairs"], rows_h["pairs"])
    np.testing.assert_array_equal(rows_m["count"], rows_h["count"])
    test.assertGreater(int(rows_h["count"].sum()), 12)
    np.testing.assert_allclose(rows_m["q"], rows_h["q"], atol=2.0e-6)
    np.testing.assert_allclose(rows_m["n"], rows_h["n"], atol=2.0e-5)
    np.testing.assert_allclose(rows_m["d"], rows_h["d"], atol=2.0e-6)
    np.testing.assert_array_equal(rows_m["feature"], rows_h["feature"])
    # The narrow phase drops the mesh pairs the law writes: the contacts are the law's alone, as for the heightfield.
    count_h = int(contacts_h.rigid_contact_count.numpy()[0])
    count_m = int(contacts_m.rigid_contact_count.numpy()[0])
    test.assertEqual(count_m, count_h)
    role = pipe_m.narrow_phase._terrain_law_role.numpy()
    test.assertEqual(int(role[terrain]), int(TERRAIN_LAW_MESH_SURFACE))
    test.assertTrue(np.all(role[np.arange(model_m.shape_count) != terrain] == int(TERRAIN_LAW_PARTNER)))


def test_mesh_with_risers_keeps_mesh_path(test, device):
    """A mesh with a riser inside its table is not a law surface yet: its pairs keep the generic mesh path."""
    model, terrain = _scene(device, mesh=True, surface=_stairs_mesh())
    surfaces = TerrainSurfaces(model)
    test.assertEqual(surfaces.mesh_shapes, [])
    test.assertEqual(int(surfaces.shape_surface[terrain]), -1)
    test.assertIn("riser", surfaces.mesh_reasons[terrain])
    pipeline, contacts, _rows = _law_rows(model)
    test.assertFalse(pipeline._terrain_law_writes)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_mesh_without_lattice_keeps_mesh_path(test, device):
    """A triangle soup has no lattice: the rule rejects it before building any table."""
    rng = np.random.default_rng(3)
    vertices = rng.uniform(-1.0, 1.0, (300, 3)).astype(np.float32)
    vertices[:, 2] *= 0.05
    triangles = rng.integers(0, 300, (100, 3)).astype(np.int32)
    builder = newton.ModelBuilder()
    terrain = builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices, triangles, compute_inertia=False))
    model = builder.finalize(device=device)
    surfaces = TerrainSurfaces(model)
    test.assertEqual(int(surfaces.shape_surface[terrain]), -1)
    test.assertIn("lattice", surfaces.mesh_reasons[terrain])


class TestTerrainLawMesh(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestTerrainLawMesh,
    "test_mesh_surface_rows_match_heightfield",
    test_mesh_surface_rows_match_heightfield,
    devices=devices,
)
add_function_test(
    TestTerrainLawMesh, "test_mesh_with_risers_keeps_mesh_path", test_mesh_with_risers_keeps_mesh_path, devices=devices
)
add_function_test(
    TestTerrainLawMesh,
    "test_mesh_without_lattice_keeps_mesh_path",
    test_mesh_without_lattice_keeps_mesh_path,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

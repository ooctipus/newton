# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The mesh cell index and the mesh-vs-convex midphase it serves, against brute force.

For every shape type and random poses near the surface of terrain-like and fully 3D meshes (a two-storey room with
a ceiling, a closed box room, a huge two-triangle ground, a long thin wall), the midphase must emit each triangle that
meets the shape's bound exactly once, and no other: the same set as the bound and back-face tests applied to every
triangle. Independently, in float64, every triangle within the contact threshold of the shape (exact for spheres and
capsules, sampled for the rest) must be emitted.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.collision_core import (
    _mesh_query_bound,
    _mesh_triangle_is_front_facing_local,
    aabb_to_unscaled,
    mesh_query_bound_meets_triangle,
    mesh_vs_convex_local_box,
    mesh_vs_convex_query_box,
)
from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.mesh_cells import (
    MAX_BOX_CELLS,
    _triangles_meet_boxes,
    build_mesh_cells,
    build_mesh_grid,
    triangle_meets_box,
)
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.geometry.support_function import GenericShapeData, pack_mesh_ptr
from newton._src.geometry.types import GeoType
from newton.tests.unittest_utils import add_function_test, get_test_devices

# ----------------------------------------------------------------------------------------------------------------
# Meshes (unwelded triangle soups like the terrain generator's, and welded closed solids)


def _quad(a, b, c, d):
    return [a, b, c, a, c, d]


def _box_faces(lo, hi, inward=False):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    )
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = []
    for quad in quads:
        tris += _quad(*(quad[::-1] if inward else quad))
    return v, np.array(tris)


def _soup(parts):
    verts, faces = [], []
    for v, f in parts:
        faces.append(np.asarray(f).reshape(-1, 3) + sum(len(x) for x in verts))
        verts.append(np.asarray(v, dtype=np.float64))
    return np.concatenate(verts).astype(np.float32), np.concatenate(faces).astype(np.int32).ravel()


def terrain_mesh(rng, n=24, step=0.1):
    """A lattice patch with random heights, a box with vertical walls and stair treads (an unwelded soup)."""
    xs = np.arange(n + 1) * step - 0.5 * n * step
    z = 0.04 * rng.standard_normal((n + 1, n + 1))
    parts = []
    for i in range(n):
        for j in range(n):
            p = [
                [xs[i], xs[j], z[i, j]],
                [xs[i + 1], xs[j], z[i + 1, j]],
                [xs[i + 1], xs[j + 1], z[i + 1, j + 1]],
                [xs[i], xs[j + 1], z[i, j + 1]],
            ]
            parts.append((p, [0, 1, 2, 0, 2, 3]))
    parts.append(_box_faces((0.3, -0.6, -0.5), (0.9, 0.2, 0.25)))
    for k in range(5):
        parts.append(_box_faces((-1.2 + 0.25 * k, -1.0, -0.5), (-0.95 + 0.25 * k, 1.0, 0.05 + 0.08 * k)))
    return _soup(parts)


def two_storey_room():
    """A closed room (inward faces) with a middle floor slab and an inner wall: floors, a ceiling, walls."""
    parts = [_box_faces((-2.0, -1.5, 0.0), (2.0, 1.5, 5.0), inward=True)]
    parts.append(_box_faces((-2.0, -1.5, 2.4), (1.0, 1.5, 2.6)))  # the upper floor, open on one side
    parts.append(_box_faces((0.0, -1.5, 0.0), (0.1, 0.5, 2.4)))  # an inner wall with a doorway
    return _soup(parts)


def closed_box_room():
    v, f = _box_faces((-1.0, -1.0, 0.0), (1.0, 1.0, 2.5), inward=True)
    return v.astype(np.float32), f.astype(np.int32).ravel()


def huge_ground():
    v = np.array([[-100.0, -100.0, 0.0], [100.0, -100.0, 0.0], [100.0, 100.0, 0.0], [-100.0, 100.0, 0.0]])
    return v.astype(np.float32), np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)


def long_wall():
    v, f = _box_faces((-25.0, -0.05, 0.0), (25.0, 0.05, 2.0))
    return v.astype(np.float32), f.astype(np.int32).ravel()


MESHES = {
    "terrain": terrain_mesh,
    "two_storey_room": lambda rng: two_storey_room(),
    "closed_box_room": lambda rng: closed_box_room(),
    "huge_ground": lambda rng: huge_ground(),
    "long_wall": lambda rng: long_wall(),
}

SHAPES = (
    (GeoType.SPHERE, (0.06, 0.0, 0.0)),
    (GeoType.CAPSULE, (0.04, 0.12, 0.0)),
    (GeoType.BOX, (0.15, 0.05, 0.03)),
    (GeoType.CYLINDER, (0.04, 0.1, 0.0)),
    (GeoType.CONE, (0.06, 0.08, 0.0)),
    (GeoType.ELLIPSOID, (0.1, 0.05, 0.04)),
    (GeoType.CONVEX_MESH, (1.5, 0.8, 1.2)),
    (GeoType.PLANE, (0.1, 0.07, 0.0)),
)


def _quat_np(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ----------------------------------------------------------------------------------------------------------------
# Brute force


@wp.func
def mesh_query_box_kernel_helper(
    mesh_shape: int,
    other: int,
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_source: wp.array[wp.uint64],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    thr: float,
) -> tuple[wp.vec3, wp.vec3]:
    return mesh_vs_convex_query_box(
        mesh_shape,
        other,
        shape_transform[mesh_shape],
        shape_transform[other],
        shape_type,
        shape_data,
        shape_source,
        shape_lower,
        shape_upper,
        thr,
    )


@wp.kernel(enable_backward=False)
def _dense_kernel(
    mesh_shape: int,
    shape_type: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_transform: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_gap: wp.array[float],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_interior: wp.array[wp.vec3],
    watertight: int,
    out: wp.array2d[wp.int32],
):
    s, t = wp.tid()
    other = s + 1
    mesh = wp.mesh_get(shape_source[mesh_shape])
    convex = GenericShapeData()
    convex.shape_type = shape_type[other]
    d = shape_data[other]
    convex.scale = wp.vec3(d[0], d[1], d[2])
    if convex.shape_type == int(GeoType.CONVEX_MESH):
        convex.auxiliary = pack_mesh_ptr(shape_source[other])
    rel = wp.transform_multiply(wp.transform_inverse(shape_transform[mesh_shape]), shape_transform[other])
    md = shape_data[mesh_shape]
    scale = wp.vec3(md[0], md[1], md[2])
    thr = shape_gap[other] + shape_gap[mesh_shape] + d[3] + md[3]
    local_lower, local_upper = mesh_vs_convex_local_box(convex, other, shape_lower, shape_upper)
    bound = _mesh_query_bound(convex, local_lower, local_upper, rel, thr)
    v0 = mesh.points[mesh.indices[3 * t + 0]]
    v1 = mesh.points[mesh.indices[3 * t + 1]]
    v2 = mesh.points[mesh.indices[3 * t + 2]]
    hit = mesh_query_bound_meets_triangle(bound, wp.cw_mul(v0, scale), wp.cw_mul(v1, scale), wp.cw_mul(v2, scale))
    lo, hi = mesh_query_box_kernel_helper(
        mesh_shape, other, shape_transform, shape_type, shape_data, shape_source, shape_lower, shape_upper, thr
    )
    center_box = 0.5 * (lo + hi)
    hit = hit and triangle_meets_box(v0 - center_box, v1 - center_box, v2 - center_box, 0.5 * (hi - lo))
    if hit:
        _lo, _hi, inv = aabb_to_unscaled(wp.vec3(0.0), wp.vec3(0.0), scale)
        center = wp.cw_mul(wp.transform_point(rel, shape_interior[other]), inv)
        hit = _mesh_triangle_is_front_facing_local(
            shape_source[mesh_shape], center, v0, v1, v2, watertight != 0, convex, rel, scale
        )
    out[s, t] = wp.where(hit, 1, 0)


def _segment_triangle_distance_np(p, q, a, b, c, samples=48):
    """Distance from a segment to a triangle, by sampling the segment densely against exact point distances."""
    ts = np.linspace(0.0, 1.0, samples)
    return min(_point_triangle_distance_np(p + t * (q - p), a, b, c) for t in ts)


def _point_triangle_distance_np(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(ap)
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return np.linalg.norm(p - (a + ab * d1 / (d1 - d3)))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return np.linalg.norm(p - (a + ac * d2 / (d2 - d6)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return np.linalg.norm(p - (b + (c - b) * (d4 - d3) / ((d4 - d3) + (d5 - d6))))
    den = 1.0 / (va + vb + vc)
    return np.linalg.norm(p - (a + ab * vb * den + ac * vc * den))


def _sampled_points(a, b, c, n=12):
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    m = (u + v) <= 1.0
    u, v = u[m], v[m]
    return a + u[:, None] * (b - a) + v[:, None] * (c - a)


def _local_distance(kind, scale, pts):
    """Distance [m] from local points to a shape (box-like shapes exactly; ellipsoids only inside or not)."""
    if kind == GeoType.BOX or kind == GeoType.CONVEX_MESH:
        half = np.asarray(scale)
        return np.linalg.norm(np.maximum(np.abs(pts) - half, 0.0), axis=1)
    if kind == GeoType.CYLINDER:
        r, hh = scale[0], scale[1]
        dr = np.maximum(np.linalg.norm(pts[:, :2], axis=1) - r, 0.0)
        dz = np.maximum(np.abs(pts[:, 2]) - hh, 0.0)
        return np.hypot(dr, dz)
    if kind == GeoType.ELLIPSOID:
        inside = np.sum((pts / np.asarray(scale)) ** 2, axis=1) <= 1.0
        return np.where(inside, 0.0, np.inf)
    if kind == GeoType.CONE:
        # Inside the cone (apex +z, base -hh) or not.
        r, hh = scale[0], scale[1]
        frac = (hh - pts[:, 2]) / (2.0 * hh)
        inside = (np.abs(pts[:, 2]) <= hh) & (np.linalg.norm(pts[:, :2], axis=1) <= r * frac)
        return np.where(inside, 0.0, np.inf)
    if kind == GeoType.PLANE:
        q = np.stack(
            [np.maximum(np.abs(pts[:, 0]) - scale[0], 0), np.maximum(np.abs(pts[:, 1]) - scale[1], 0), pts[:, 2]], 1
        )
        return np.linalg.norm(q, axis=1)
    raise AssertionError(kind)


# ----------------------------------------------------------------------------------------------------------------
# The scene


def _run(test, device, name, seed, mesh_scale=(1.0, 1.0, 1.0), mesh_pose=None, count=48, cell=None):
    rng = np.random.default_rng(seed)
    verts, faces = MESHES[name](rng)
    mesh = newton.Mesh(verts, faces, compute_inertia=False)
    hull = newton.Mesh.create_box(0.06, 0.04, 0.03, compute_inertia=False)
    with wp.ScopedDevice(device):
        mesh_id = mesh.finalize()
        hull_id = hull.finalize()
    watertight = int(mesh.is_watertight)
    if mesh_pose is None:
        mesh_pose = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    mesh_rot = _quat_np(mesh_pose[1])
    scale = np.asarray(mesh_scale, dtype=np.float64)
    tri = faces.reshape(-1, 3)
    world = (mesh_rot @ (verts.astype(np.float64) * scale).T).T + mesh_pose[0]
    gap = 0.01

    types, datas, poses, sources, kinds = [GeoType.MESH], [(*mesh_scale, 0.0)], [mesh_pose], [mesh_id], []
    for k in range(count):
        kind, shape_scale = SHAPES[k % len(SHAPES)]
        t = tri[rng.integers(tri.shape[0])]
        u, v = rng.random(2)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        p = world[t[0]] + u * (world[t[1]] - world[t[0]]) + v * (world[t[2]] - world[t[0]])
        direction = rng.standard_normal(3)
        p = p + direction / np.linalg.norm(direction) * rng.uniform(-0.1, 0.3)
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        types.append(kind)
        datas.append((*shape_scale, float(rng.choice([0.0, 0.005]))))
        poses.append((p, q))
        sources.append(hull_id if kind == GeoType.CONVEX_MESH else 0)
        kinds.append((kind, shape_scale))
    n = len(types)
    shape_type = np.array([int(x) for x in types], dtype=np.int32)
    shape_source = np.array(sources, dtype=np.uint64)
    lower = np.zeros((n, 3))
    upper = np.zeros((n, 3))
    for i in range(1, n):
        if types[i] == GeoType.CONVEX_MESH:
            # The model's local box of a scaled hull (as the builder computes it).
            lo, hi = hull.vertices.min(0) * datas[i][:3], hull.vertices.max(0) * datas[i][:3]
            lower[i], upper[i] = np.minimum(lo, hi), np.maximum(lo, hi)
    arr = lambda x, t: wp.array(x, dtype=t, device=device)  # noqa: E731
    xf = arr([wp.transform(wp.vec3(*p), wp.quat(*q)) for p, q in poses], wp.transform)
    kw = {
        "candidate_pair": arr(np.stack([np.zeros(n - 1), np.arange(1, n)], 1).astype(np.int32), wp.vec2i),
        "candidate_pair_count": arr([n - 1], wp.int32),
        "shape_types": arr(shape_type, wp.int32),
        "shape_data": arr(np.array(datas, dtype=np.float32), wp.vec4),
        "shape_transform": xf,
        "shape_source": arr(shape_source, wp.uint64),
        "shape_sdf_index": arr(np.full(n, -1), wp.int32),
        "shape_gap": arr(np.full(n, gap, dtype=np.float32), wp.float32),
        "shape_collision_radius": arr(np.ones(n), wp.float32),
        "shape_flags": arr(np.full(n, int(ShapeFlags.COLLIDE_SHAPES)), wp.int32),
        "shape_collision_aabb_lower": arr(lower, wp.vec3),
        "shape_collision_aabb_upper": arr(upper, wp.vec3),
        "shape_voxel_resolution": arr(np.full((n, 3), 4), wp.vec3i),
        "shape_interior_point": arr(np.zeros((n, 3)), wp.vec3),
        "shape_mesh_properties": arr(np.array([1 if watertight else 0] + [0] * (n - 1)), wp.int32),
    }
    narrow = NarrowPhase(max_candidate_pairs=n, max_triangle_pairs=200000, device=device, reduce_contacts=False)
    mesh_cells = build_mesh_cells(shape_type, shape_source, device, cell=cell)
    m = 4 * n * 64
    narrow.launch(
        **kw,
        mesh_cells=mesh_cells,
        contact_pair=wp.zeros(m, dtype=wp.vec2i, device=device),
        contact_position=wp.zeros(m, dtype=wp.vec3, device=device),
        contact_normal=wp.zeros(m, dtype=wp.vec3, device=device),
        contact_penetration=wp.zeros(m, dtype=float, device=device),
        contact_count=wp.zeros(1, dtype=int, device=device),
    )
    count_tp = int(narrow.triangle_pairs_count.numpy()[0])
    test.assertLessEqual(count_tp, narrow.triangle_pairs.shape[0])
    emitted = narrow.triangle_pairs.numpy()[:count_tp]
    test.assertTrue(np.all(emitted[:, 0] == 0))
    keys = emitted[:, 1].astype(np.int64) * (1 << 32) + emitted[:, 2]
    test.assertEqual(len(np.unique(keys)), len(keys), f"{name}: a triangle was emitted twice")

    dense = wp.zeros((n - 1, tri.shape[0]), dtype=wp.int32, device=device)
    wp.launch(
        _dense_kernel,
        dim=(n - 1, tri.shape[0]),
        inputs=[
            0,
            kw["shape_types"],
            kw["shape_data"],
            xf,
            kw["shape_source"],
            kw["shape_gap"],
            kw["shape_collision_aabb_lower"],
            kw["shape_collision_aabb_upper"],
            kw["shape_interior_point"],
            watertight,
        ],
        outputs=[dense],
        device=device,
    )
    dense = dense.numpy()
    want = set(zip(*np.nonzero(dense), strict=True))
    got = {(int(s) - 1, int(t)) for _, s, t in emitted}
    test.assertEqual(got, want, f"{name}: cell midphase differs from the dense bound test")

    # Independent float64 check: triangles within the threshold that face the shape must be emitted.
    missed = 0
    for i, (kind, shape_scale) in enumerate(kinds):
        thr = 2.0 * gap + datas[i + 1][3]
        p, q = poses[i + 1]
        rot = _quat_np(q)
        for t in range(tri.shape[0]):
            a, b, c = world[tri[t, 0]], world[tri[t, 1]], world[tri[t, 2]]
            if kind == GeoType.SPHERE:
                d = _point_triangle_distance_np(p, a, b, c) - shape_scale[0]
            elif kind == GeoType.CAPSULE:
                axis = rot[:, 2] * shape_scale[1]
                d = _segment_triangle_distance_np(p - axis, p + axis, a, b, c) - shape_scale[0]
            else:
                local = (rot.T @ (_sampled_points(a, b, c) - p).T).T
                d = float(
                    np.min(
                        _local_distance(
                            kind,
                            np.multiply((0.06, 0.04, 0.03), shape_scale)
                            if kind == GeoType.CONVEX_MESH
                            else shape_scale,
                            local,
                        )
                    )
                )
            if d > thr - 1.0e-4:
                continue
            # The back-face rule for open meshes, in the unscaled mesh frame: the shape's center in front.
            v = verts[tri[t]].astype(np.float64)
            center = (mesh_rot.T @ (p - mesh_pose[0])) / scale
            if not watertight and np.cross(v[1] - v[0], v[2] - v[0]) @ (center - v[0]) < 0.0:
                continue
            if watertight:
                continue
            if (i, t) not in got:
                missed += 1
    test.assertEqual(missed, 0, f"{name}: {missed} triangles within the threshold were not emitted")
    return len(got)


class TestMeshCells(unittest.TestCase):
    pass


def test_build_registers_surface_cells(test, device):
    """Each triangle registers in exactly the cells of its box that its surface meets, at most MAX_BOX_CELLS."""
    rng = np.random.default_rng(3)
    for name, make in MESHES.items():
        verts, faces = make(rng)
        g = build_mesh_grid(verts, faces)
        tri = faces.reshape(-1, 3)
        a, b, c = (verts[tri[:, k]].astype(np.float64) for k in range(3))
        masks = g["triangles"]["cells_mask"]
        counts = np.array([bin(int(m) & 0xFFFFFFFFFFFFFFFF).count("1") for m in masks])
        test.assertTrue(np.all(counts >= 1), name)
        test.assertTrue(np.all(counts <= MAX_BOX_CELLS), name)
        test.assertEqual(int(counts.sum()), g["entries"].shape[0], name)
        # The cell keys of the entries meet their triangles.
        slots = g["sorted_slots"]
        level = (slots["key"] >> np.uint64(60)).astype(np.int64)
        cells = np.stack(
            [((slots["key"] >> np.uint64(s)) & np.uint64(0xFFFFF)).astype(np.int64) for s in (40, 20, 0)], 1
        )
        owner = np.repeat(np.arange(slots.size), slots["count"])
        t = g["entries"][:, 0]
        size = float(g["cell"]) * np.left_shift(1, 2 * level[owner]).astype(np.float64)
        center = g["origin"].astype(np.float64) + (cells[owner] + 0.5) * size[:, None]
        half = np.repeat(0.5 * size[:, None] * (1.0 + 1.0e-6), 3, axis=1)
        test.assertTrue(np.all(_triangles_meet_boxes(a[t], b[t], c[t], center, half)), name)


def test_two_triangle_ground_is_small(test, device):
    """A 200 m ground of two triangles costs a few entries."""
    verts, faces = huge_ground()
    g = build_mesh_grid(verts, faces, cell=0.2)
    test.assertLessEqual(g["entries"].shape[0], 2 * MAX_BOX_CELLS)


def test_midphase_matches_brute_force(test, device):
    """Every mesh, every shape type, random poses: the emitted set is the dense bound test's, once each."""
    for seed, name in enumerate(MESHES):
        _run(test, device, name, seed)


def test_midphase_scaled_and_moved_mesh(test, device):
    """A rotated, translated, non-uniformly (and mirror-) scaled mesh."""
    q = np.array([0.2, -0.3, 0.1, 0.9])
    q /= np.linalg.norm(q)
    _run(test, device, "terrain", 11, mesh_scale=(1.5, 0.7, 2.0), mesh_pose=(np.array([3.0, -2.0, 1.0]), q))
    _run(test, device, "two_storey_room", 12, mesh_scale=(-1.0, 1.0, 1.0))


def test_midphase_cell_sizes(test, device):
    """The emitted set does not depend on the cell edge."""
    sizes = {_run(test, device, "terrain", 21, cell=c) for c in (0.05, 0.2, 1.0)}
    test.assertEqual(len(sizes), 1)


def test_non_finite_pose_meets_nothing(test, device):
    """A shape with a non-finite pose gets no triangles (and its query stays in bounds)."""
    verts, faces = terrain_mesh(np.random.default_rng(5))
    mesh = newton.Mesh(verts, faces, compute_inertia=False)
    with wp.ScopedDevice(device):
        mesh_id = mesh.finalize()
    arr = lambda x, t: wp.array(x, dtype=t, device=device)  # noqa: E731
    shape_type = np.array([int(GeoType.MESH), int(GeoType.SPHERE), int(GeoType.SPHERE)], dtype=np.int32)
    shape_source = np.array([mesh_id, 0, 0], dtype=np.uint64)
    poses = [(0.0, 0.0, 0.0), (np.nan, 0.0, 0.0), (0.0, 0.0, np.inf)]
    narrow = NarrowPhase(max_candidate_pairs=2, max_triangle_pairs=1000, device=device, reduce_contacts=False)
    m = 64
    narrow.launch(
        candidate_pair=arr([[0, 1], [0, 2]], wp.vec2i),
        candidate_pair_count=arr([2], wp.int32),
        shape_types=arr(shape_type, wp.int32),
        shape_data=arr([[1.0, 1.0, 1.0, 0.0], [0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]], wp.vec4),
        shape_transform=arr([wp.transform(wp.vec3(*p), wp.quat_identity()) for p in poses], wp.transform),
        shape_source=arr(shape_source, wp.uint64),
        shape_sdf_index=arr(np.full(3, -1), wp.int32),
        shape_gap=arr(np.full(3, 0.01), wp.float32),
        shape_collision_radius=arr(np.ones(3), wp.float32),
        shape_flags=arr(np.full(3, int(ShapeFlags.COLLIDE_SHAPES)), wp.int32),
        shape_collision_aabb_lower=arr(np.zeros((3, 3)), wp.vec3),
        shape_collision_aabb_upper=arr(np.zeros((3, 3)), wp.vec3),
        shape_voxel_resolution=arr(np.full((3, 3), 4), wp.vec3i),
        shape_interior_point=arr(np.zeros((3, 3)), wp.vec3),
        mesh_cells=build_mesh_cells(shape_type, shape_source, device),
        contact_pair=wp.zeros(m, dtype=wp.vec2i, device=device),
        contact_position=wp.zeros(m, dtype=wp.vec3, device=device),
        contact_normal=wp.zeros(m, dtype=wp.vec3, device=device),
        contact_penetration=wp.zeros(m, dtype=float, device=device),
        contact_count=wp.zeros(1, dtype=int, device=device),
    )
    test.assertEqual(int(narrow.shape_pairs_mesh_count.numpy()[0]), 2)
    test.assertEqual(int(narrow.triangle_pairs_count.numpy()[0]), 0)
    test.assertTrue(np.all(narrow._mesh_query_grid.numpy()[:2] == -1))


for _test in (
    test_non_finite_pose_meets_nothing,
    test_build_registers_surface_cells,
    test_two_triangle_ground_is_small,
    test_midphase_matches_brute_force,
    test_midphase_scaled_and_moved_mesh,
    test_midphase_cell_sizes,
):
    add_function_test(TestMeshCells, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()

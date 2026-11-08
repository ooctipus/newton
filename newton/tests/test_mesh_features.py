# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact contacts on triangle meshes: one contact per local minimum of the shape-surface distance.

A sphere's contacts with a mesh are the local minima of the distance from its center to the surface: a triangle's
closest point is one exactly when it is also the closest point of every triangle that contains it. These tests
compare the mesh path with that brute-force reference on soups (unwelded, as the rough terrain is): flat lattices,
creases, corners, ceilings, stacked floors, a closed room, convex edges and vertices, and a random heightfield's
exact triangulation. Capsules, boxes and cylinders on a fine lattice must give the contacts they get on a plane.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.mesh_topology import build_mesh_topology
from newton.tests.unittest_utils import add_function_test, get_test_devices

GAP = 0.01
RADIUS = 0.05


def _soup(vertices, faces):
    """An unwelded triangle soup of a mesh (every triangle its own three vertices)."""
    v = np.asarray(vertices, np.float32)[np.asarray(faces, np.int64).reshape(-1)]
    return v, np.arange(v.shape[0], dtype=np.int32).reshape(-1, 3)


def _lattice(n, size, z=0.0, heights=None, flip=False):
    """An ``n`` x ``n`` lattice over ``[-size, size]^2`` at height ``z`` (or ``heights``), facing up (down if flip)."""
    xs = np.linspace(-size, size, n + 1, dtype=np.float32)
    x, y = np.meshgrid(xs, xs, indexing="xy")
    h = np.full(x.shape, z, np.float32) if heights is None else np.asarray(heights, np.float32)
    v = np.stack([x.ravel(), y.ravel(), h.ravel()], 1)
    f = []
    for r in range(n):
        for c in range(n):
            a = r * (n + 1) + c
            b, d = a + 1, a + n + 1
            e = d + 1
            f += [(a, d, e), (a, e, b)] if flip else [(a, b, e), (a, e, d)]
    return v, np.array(f, np.int64)


def _quad(p0, p1, p2, p3, n=4, toward=None):
    """A planar quad ``p0 p1 p2 p3`` split into ``2 n^2`` triangles, facing ``toward`` (else its winding's front)."""
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    s = np.linspace(0.0, 1.0, n + 1)
    pts = []
    for j in s:
        for i in s:
            pts.append((1 - i) * (1 - j) * p0 + i * (1 - j) * p1 + i * j * p2 + (1 - i) * j * p3)
    f = []
    for r in range(n):
        for c in range(n):
            a = r * (n + 1) + c
            b, d = a + 1, a + n + 1
            e = d + 1
            f += [(a, b, e), (a, e, d)]
    f = np.array(f, np.int64)
    if toward is not None and np.cross(p1 - p0, p3 - p0) @ (np.asarray(toward, np.float64) - p0) < 0.0:
        f = f[:, [0, 2, 1]]
    return np.array(pts, np.float32), f


def _merge(*parts):
    vs, fs, base = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + base)
        base += v.shape[0]
    return np.concatenate(vs), np.concatenate(fs)


def _closest(p, a, b, c):
    """Closest point of triangle ``abc`` to ``p`` (float64)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + d1 / (d1 - d3) * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + d2 / (d2 - d6) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def _reference_minima(vertices, faces, center, reach):
    """The local minima of the distance from ``center`` to the one-sided surface within ``reach``.

    Faces with the center behind their plane are not part of the surface the center sees (the midphase culls
    them for open meshes).
    """
    tri = np.asarray(vertices, np.float64)[np.asarray(faces).reshape(-1, 3)]
    front = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), center - tri[:, 0]) >= 0.0
    tri = tri[front]
    if tri.shape[0] == 0:
        return []
    feet = np.array([_closest(center, *t) for t in tri])
    dist = np.linalg.norm(feet - center, axis=1)
    minima = []
    for i in np.flatnonzero(dist <= reach):
        f = feet[i]
        # Triangles containing the foot: their closest points must all be the foot.
        contain = [j for j in range(tri.shape[0]) if np.linalg.norm(_closest(f, *tri[j]) - f) < 1e-6]
        if all(np.linalg.norm(feet[j] - f) < 1e-5 for j in contain):
            if not any(np.linalg.norm(m[0] - f) < 1e-5 for m in minima):
                minima.append((f, dist[i]))
    return minima


def _collide(device, vertices, faces, shape, xform, reduce=False, scale=None):
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.0, gap=GAP)
    mesh = newton.Mesh(np.asarray(vertices, np.float32), np.asarray(faces, np.int32).ravel(), compute_inertia=False)
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg, scale=scale)
    body = builder.add_body(xform=xform)
    kind, args = shape
    getattr(builder, f"add_shape_{kind}")(body, cfg=cfg, **args)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=reduce)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    normal = contacts.rigid_contact_normal.numpy()[:count]
    p0 = contacts.rigid_contact_point0.numpy()[:count]
    p1 = contacts.rigid_contact_point1.numpy()[:count].astype(np.float64)
    # Shape 0 is the static mesh (world frame); shape 1's point is in its body frame.
    q = np.array(xform, np.float64)
    t, (x, y, z, w) = q[:3], q[3:]
    u = np.array([x, y, z])
    p1 = p1 + 2.0 * w * np.cross(u, p1) + 2.0 * np.cross(u, np.cross(u, p1)) + t
    margin = contacts.rigid_contact_margin0.numpy()[:count] + contacts.rigid_contact_margin1.numpy()[:count]
    separation = np.einsum("ij,ij->i", normal, p1 - p0) - margin
    return count, normal, p0, separation


def _sphere_case(test, device, vertices, faces, center, scale=None):
    """Mesh sphere contacts against the brute-force local minima of the (scaled) soup."""
    v, f = _soup(vertices, faces)
    world = v.astype(np.float64) * (1.0 if scale is None else np.asarray(scale, np.float64))
    # A mirroring scale reverses the winding; the triangle path swaps two corners to keep faces outward.
    outward = f if scale is None or np.prod(scale) > 0.0 else f[:, [0, 2, 1]]
    expected = _reference_minima(world, outward, np.asarray(center, np.float64), RADIUS + 2.0 * GAP)
    count, normal, p0, _p1 = _collide(
        device, v, f, ("sphere", {"radius": RADIUS}), wp.transform(wp.vec3(*center), wp.quat_identity()), scale=scale
    )
    test.assertEqual(count, len(expected), f"center {center}: {count} contacts, {len(expected)} local minima")
    for foot, _distance in expected:
        k = int(np.argmin(np.linalg.norm(p0 - foot, axis=1)))
        test.assertLess(np.linalg.norm(p0[k] - foot), 2e-5)
        want = (np.asarray(center) - foot) / np.linalg.norm(np.asarray(center) - foot)
        test.assertGreater(float(abs(normal[k] @ want)), 1.0 - 1e-5)


def test_flat_lattice_spheres(test, device):
    """A finely triangulated plane: one contact anywhere, never a ghost normal at an internal edge."""
    v, f = _lattice(8, 0.4)
    rng = np.random.default_rng(1)
    centers = [(0.0, 0.0), (0.1, 0.0), (0.05, 0.05), (0.1, 0.1), (0.033, 0.017)]
    centers += [tuple(rng.uniform(-0.3, 0.3, 2)) for _ in range(6)]
    for x, y in centers:
        _sphere_case(test, device, v, f, (x, y, RADIUS + 0.004))


def test_concave_spheres(test, device):
    """Crease, corner, ceiling, two floors and a closed room: one contact per face touched."""
    floor = _quad((-0.4, -0.4, 0.0), (0.4, -0.4, 0.0), (0.4, 0.4, 0.0), (-0.4, 0.4, 0.0))
    wall_x = _quad((0.2, -0.4, 0.0), (0.2, 0.4, 0.0), (0.2, 0.4, 0.5), (0.2, -0.4, 0.5), toward=(0, 0, 0))
    wall_y = _quad((-0.4, 0.2, 0.0), (-0.4, 0.2, 0.5), (0.4, 0.2, 0.5), (0.4, 0.2, 0.0), toward=(0, 0, 0))
    ceiling = _quad((-0.4, -0.4, 0.12), (-0.4, 0.4, 0.12), (0.4, 0.4, 0.12), (0.4, -0.4, 0.12))  # faces down
    upper = _quad((-0.4, -0.4, 0.3), (0.4, -0.4, 0.3), (0.4, 0.4, 0.3), (-0.4, 0.4, 0.3))
    r = RADIUS + 0.003
    _sphere_case(test, device, *_merge(floor, wall_x), (0.2 - r, 0.0, r))
    _sphere_case(test, device, *_merge(floor, wall_x, wall_y), (0.2 - r, 0.2 - r, r))
    _sphere_case(test, device, *ceiling, (0.0, 0.0, 0.12 - r))
    _sphere_case(test, device, *_merge(floor, ceiling), (0.05, 0.05, 0.06))
    _sphere_case(test, device, *_merge(floor, ceiling, upper), (0.05, 0.05, 0.3 + r))
    room = _merge(
        floor,
        _quad((-0.4, -0.4, 0.4), (-0.4, 0.4, 0.4), (0.4, 0.4, 0.4), (0.4, -0.4, 0.4)),
        _quad((0.4, -0.4, 0.0), (0.4, 0.4, 0.0), (0.4, 0.4, 0.4), (0.4, -0.4, 0.4), toward=(0, 0, 0.2)),
        _quad((-0.4, -0.4, 0.0), (-0.4, -0.4, 0.4), (-0.4, 0.4, 0.4), (-0.4, 0.4, 0.0), toward=(0, 0, 0.2)),
        _quad((-0.4, 0.4, 0.0), (-0.4, 0.4, 0.4), (0.4, 0.4, 0.4), (0.4, 0.4, 0.0), toward=(0, 0, 0.2)),
        _quad((-0.4, -0.4, 0.0), (0.4, -0.4, 0.0), (0.4, -0.4, 0.4), (-0.4, -0.4, 0.4), toward=(0, 0, 0.2)),
    )
    _sphere_case(test, device, *room, (0.4 - r, 0.4 - r, r))
    _sphere_case(test, device, *room, (0.0, -0.4 + r, 0.4 - r))


def test_convex_spheres(test, device):
    """A stair nose and a box corner from outside: one contact, its normal inside the feature's cone."""
    tread = _quad((-0.4, -0.4, 0.0), (0.0, -0.4, 0.0), (0.0, 0.4, 0.0), (-0.4, 0.4, 0.0))
    riser = _quad((0.0, -0.4, -0.3), (0.0, 0.4, -0.3), (0.0, 0.4, 0.0), (0.0, -0.4, 0.0))  # faces +x
    for angle in (0.2, 0.7, 1.2):
        d = RADIUS + 0.004
        _sphere_case(test, device, *_merge(tread, riser), (d * np.sin(angle), 0.03, d * np.cos(angle)))
    s = 0.2
    box = _merge(
        _quad((-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s), 2),
        _quad((s, -s, -s), (s, s, -s), (s, s, s), (s, -s, s), 2),
        _quad((-s, s, -s), (-s, s, s), (s, s, s), (s, s, -s), 2),
    )
    d = (RADIUS + 0.004) / np.sqrt(3.0)
    _sphere_case(test, device, *box, (s + d, s + d, s + d))
    _sphere_case(test, device, *box, (s + 0.02, s + 0.03, s + 0.035))
    _sphere_case(test, device, *box, (s + 0.04, 0.05, s + 0.03))


def test_heightfield_triangulation_spheres(test, device):
    """A random heightfield's exact triangulation (the law's diagonal): the brute-force local minima."""
    rng = np.random.default_rng(7)
    heights = rng.uniform(0.0, 0.06, (9, 9)).astype(np.float32)
    v, f = _lattice(8, 0.4, heights=heights)
    for _ in range(12):
        x, y = rng.uniform(-0.3, 0.3, 2)
        _sphere_case(test, device, v, f, (x, y, 0.06 + RADIUS - 0.01))


def test_heightfield_equivalence(test, device):
    """A heightfield and its exact triangulation as a mesh: spheres and capsules get the same contacts."""
    rng = np.random.default_rng(7)
    heights = rng.uniform(0.0, 0.06, (9, 9)).astype(np.float32)
    v, f = _soup(*_lattice(8, 0.4, heights=heights))

    def collide(kind, center, heightfield):
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=0.0, gap=GAP)
        if heightfield:
            terrain = newton.Heightfield(heights.copy(), nrow=9, ncol=9, hx=0.4, hy=0.4)
            builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
        else:
            builder.add_shape_mesh(body=-1, mesh=newton.Mesh(v, f.ravel(), compute_inertia=False), cfg=cfg)
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.2)), 0.7)
        body = builder.add_body(xform=wp.transform(wp.vec3(*center), rotation))
        if kind == "sphere":
            builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        else:
            builder.add_shape_capsule(body, radius=0.03, half_height=0.08, cfg=cfg)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        point = contacts.rigid_contact_point0.numpy()[:count]
        order = np.lexsort(point.T[::-1])
        return count, point[order], contacts.rigid_contact_normal.numpy()[:count][order]

    reported = 0
    for kind in ("sphere", "capsule"):
        for _ in range(10):
            # The core stays in front of the faces (penetration below the radius): the closed forms' regime.
            center = (*rng.uniform(-0.3, 0.3, 2), 0.1 if kind == "sphere" else 0.125)
            want = collide(kind, center, True)
            got = collide(kind, center, False)
            test.assertEqual(got[0], want[0], f"{kind} at {center}")
            np.testing.assert_allclose(got[1], want[1], atol=2e-5)
            np.testing.assert_allclose(got[2], want[2], atol=2e-5)
            reported += got[0]
    test.assertGreater(reported, 10)


def test_mirrored_scale_spheres(test, device):
    """A mirrored mesh scale keeps outward faces and the same local minima."""
    tread = _quad((-0.4, -0.4, 0.0), (0.0, -0.4, 0.0), (0.0, 0.4, 0.0), (-0.4, 0.4, 0.0))
    riser = _quad((0.0, -0.4, -0.3), (0.0, 0.4, -0.3), (0.0, 0.4, 0.0), (0.0, -0.4, 0.0))
    v, f = _merge(tread, riser)
    d = RADIUS + 0.004
    _sphere_case(test, device, v, f, (-d * np.sin(0.6), 0.03, d * np.cos(0.6)), scale=(-1.0, 1.0, 1.0))


def test_topology_build(test, device):
    """Welding a soup recovers the lattice's edges; a fold's coincident triangles are non-manifold."""
    v, f = _soup(*_lattice(3, 1.0))
    topo = build_mesh_topology(v, f)
    test.assertEqual(topo["vertex_start"].shape[0] - 1, 16)
    test.assertEqual(int((topo["neighbor"] < 0).sum()), 12)
    # A doubled triangle: its edges are non-manifold, hence boundaries.
    v2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 2, np.float32)
    topo2 = build_mesh_topology(v2, np.arange(6).reshape(2, 3))
    test.assertTrue((topo2["neighbor"] < 0).all())


def test_flat_lattice_shapes(test, device):
    """Capsules, boxes and cylinders on a fine flat lattice get the plane's contacts: no seam point, no ghost.

    Under the shapes every lattice vertex is a flat vertex, so the manifold is the partner's own: a capsule's two
    ends, a box face's four corners, a cylinder's rim samples, each reported once, at the plane depth.
    """
    fine = _soup(*_lattice(16, 0.8))
    shapes = (
        ("capsule", {"radius": 0.03, "half_height": 0.12}, 0.031, wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 1.5707964)),
        ("box", {"hx": 0.13, "hy": 0.07, "hz": 0.05}, 0.051, wp.quat_identity()),
        ("cylinder", {"radius": 0.06, "half_height": 0.05}, 0.051, wp.quat_identity()),
    )
    for kind, args, z, rot in shapes:
        for x, y in ((0.0, 0.0), (0.05, 0.0), (0.033, 0.017)):
            xform = wp.transform(wp.vec3(x, y, z), rot)
            got = _collide(device, *fine, (kind, args), xform)
            test.assertTrue(np.all(np.abs(got[1][:, 2]) > 1.0 - 1e-5), f"{kind}: a ghost normal")
            np.testing.assert_allclose(got[3], np.full(got[0], 0.001), atol=2e-5)
            local = got[2][:, :2] - np.array([x, y])
            if kind == "capsule":
                want = np.array([[-0.12, 0.0], [0.12, 0.0]])
            elif kind == "box":
                want = np.array([[sx * 0.13, sy * 0.07] for sx in (-1, 1) for sy in (-1, 1)])
            else:
                want = None
            if want is None:
                # Rim samples of the disc (the manifold keeps four).
                test.assertEqual(got[0], 4)
                np.testing.assert_allclose(np.linalg.norm(local, axis=1), 0.06, atol=2e-5)
            else:
                test.assertEqual(got[0], want.shape[0], f"{kind} at {(x, y)}")
                np.testing.assert_allclose(np.sort(local, axis=0), np.sort(want, axis=0), atol=2e-5)
            # The reducer keeps a subset of the same points.
            reduced = _collide(device, *fine, (kind, args), xform, reduce=True)
            test.assertTrue(np.all(np.abs(reduced[1][:, 2]) > 1.0 - 1e-5))
            test.assertLessEqual(reduced[0], got[0])


def test_capsule_over_ridge(test, device):
    """A capsule across a gable ridge: one contact at the ridge, from the core's interior, normal in the cone."""
    left = _quad((-0.4, -0.4, -0.2), (0.0, -0.4, 0.0), (0.0, 0.4, 0.0), (-0.4, 0.4, -0.2))
    right = _quad((0.0, -0.4, 0.0), (0.4, -0.4, -0.2), (0.4, 0.4, -0.2), (0.0, 0.4, 0.0))
    v, f = _soup(*_merge(left, right))
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 1.5707964)
    for y in (0.0, 0.05, 0.1):
        xform = wp.transform(wp.vec3(0.01, y, 0.033), rot)
        count, normal, p0, separation = _collide(
            device, v, f, ("capsule", {"radius": 0.03, "half_height": 0.15}), xform
        )
        test.assertEqual(count, 1)
        np.testing.assert_allclose(p0[0], (0.0, y, 0.0), atol=1e-5)
        test.assertGreater(float(normal[0, 2]), 0.9)
        test.assertAlmostEqual(float(separation[0]), 0.033 - 0.03, delta=2e-5)


def test_box_on_stair_nose(test, device):
    """A box resting across a stair nose: its patch on the tread, cut at the nose, with no ghost normal.

    The patch's corners are the box's two corners over the tread and the two crossings of its edges with the nose;
    each is reported once. The tread's coplanar edges ending at the nose (a facet corner line) keep their seam
    points; every contact lies in the patch.
    """
    tread = _quad((-0.4, -0.4, 0.0), (0.0, -0.4, 0.0), (0.0, 0.4, 0.0), (-0.4, 0.4, 0.0), 8)
    riser = _quad((0.0, -0.4, -0.3), (0.0, 0.4, -0.3), (0.0, 0.4, 0.0), (0.0, -0.4, 0.0), 8)
    xform = wp.transform(wp.vec3(-0.05, 0.013, 0.051), wp.quat_identity())
    count, normal, p0, separation = _collide(
        device, *_soup(*_merge(tread, riser)), ("box", {"hx": 0.13, "hy": 0.07, "hz": 0.05}), xform
    )
    test.assertTrue(np.all(normal[:, 2] > 1.0 - 1e-5))
    np.testing.assert_allclose(separation, np.full(count, 0.001), atol=2e-5)
    xy = p0[:, :2]
    for corner in ((-0.18, -0.057), (-0.18, 0.083), (0.0, -0.057), (0.0, 0.083)):
        test.assertEqual(int(np.sum(np.linalg.norm(xy - np.array(corner), axis=1) < 2e-5)), 1, f"corner {corner}")
    test.assertTrue(np.all((xy[:, 0] > -0.18 - 2e-5) & (xy[:, 0] < 2e-5)))
    test.assertTrue(np.all((xy[:, 1] > -0.057 - 2e-5) & (xy[:, 1] < 0.083 + 2e-5)))


def test_small_facet_keeps_its_corners(test, device):
    """A box covering a small two-triangle pad (a finger pad) rests on the pad's four corners."""
    v = np.array([[-0.02, -0.01, 0.0], [0.02, -0.01, 0.0], [0.02, 0.01, 0.0], [-0.02, 0.01, 0.0]], np.float32)
    v, f = _soup(v, np.array([[0, 1, 2], [0, 2, 3]]))
    for reduce in (False, True):
        xform = wp.transform(wp.vec3(0.003, 0.002, 0.0199), wp.quat_identity())
        count, _normal, p0, _separation = _collide(
            device, v, f, ("box", {"hx": 0.05, "hy": 0.05, "hz": 0.02}), xform, reduce=reduce
        )
        test.assertEqual(count, 4)
        np.testing.assert_allclose(np.sort(np.abs(p0[:, :2]), axis=0), np.full((4, 2), (0.02, 0.01)), atol=1e-6)


class TestMeshFeatures(unittest.TestCase):
    pass


devices = get_test_devices()
for _name, _fn in (
    ("test_flat_lattice_spheres", test_flat_lattice_spheres),
    ("test_concave_spheres", test_concave_spheres),
    ("test_convex_spheres", test_convex_spheres),
    ("test_heightfield_triangulation_spheres", test_heightfield_triangulation_spheres),
    ("test_heightfield_equivalence", test_heightfield_equivalence),
    ("test_mirrored_scale_spheres", test_mirrored_scale_spheres),
    ("test_flat_lattice_shapes", test_flat_lattice_shapes),
    ("test_capsule_over_ridge", test_capsule_over_ridge),
    ("test_box_on_stair_nose", test_box_on_stair_nose),
    ("test_small_facet_keeps_its_corners", test_small_facet_keeps_its_corners),
):
    add_function_test(TestMeshFeatures, _name, _fn, devices=devices)
add_function_test(TestMeshFeatures, "test_topology_build", test_topology_build, devices=["cpu"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

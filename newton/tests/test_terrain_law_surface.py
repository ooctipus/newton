# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law's float32 surface predicates: canonical values and watertightness at seams.

Every support the float64 oracle (:mod:`newton.tests.terrain_law_reference`) finds must be claimed by the float32
predicates of :mod:`newton._src.geometry.terrain_law`, in the window-local frame, however exactly the shape sits
on a grid line, vertex or diagonal and however far from the origin. Today's heightfield route builds a float32
frame per cell (``x1 = x0 + dx``), which leaves cracks of up to 9.5 um between neighbouring cells on a 200 m
terrain; an upright cylinder on a vertex whose lines both crack loses every contact under a 3e-14 rad tilt.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.terrain_contacts import TerrainFeatures
from newton._src.geometry.terrain_law import (
    TERRAIN_EDGE_CONVEX,
    TerrainFeatureData,
    terrain_cluster_plane,
    terrain_edge_values,
    terrain_eps,
    terrain_face_values,
    terrain_window,
    terrain_window_point,
)
from newton._src.utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_test_utils as tu
from newton.tests.unittest_utils import add_function_test, get_test_devices

SUITE_FLAGS = {
    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
    "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
    "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "1",
    "NEWTON_REDUCER_EXPORT_CLEANUP": "1",
}
"""The terrain flags the suite's rough recipes set."""

LONG_HALF = 100.0
LONG_COUNT = 2001
SHORT_HALF = 0.4
SHORT_COUNT = 9
DISTANCES = (0.5, 40.0, 99.9)
TILT = 3.0e-14
TILT_AXIS = (-1.0, 1.0, 0.0)
"""A tilt about this axis moves the lower cap of an upright cylinder toward -x and -y."""
SPHERE_RADIUS = 0.031455
CYLINDER = {"radius": 0.05, "half_height": 0.04}
PENETRATION = 0.001
REACH = 0.04
"""Margins and gaps of the pair [m]."""
WINDOW_BAND = 8.0 * float(np.finfo(np.float32).eps) * 0.4
"""The closed predicates' band of a 0.4 m window [m] (:func:`terrain_eps`)."""
CANONICAL_BAND = 8.0 * float(np.finfo(np.float32).eps) * 1.0
"""The band the canonical-value kernel sets (``terrain_eps(1.0)``) [m]."""


@wp.kernel(enable_backward=False)
def _per_cell_corners(
    heightfield: wp.array[HeightfieldData],
    elevations: wp.array[wp.float32],
    corners: wp.array[wp.vec2i],
    out: wp.array[wp.vec3],
):
    """Corner ``corners[i][1]`` of triangle ``corners[i][0]`` as today's per-cell frame places it."""
    i = wp.tid()
    triangle = corners[i][0]
    corner = corners[i][1]
    geom, v0 = get_triangle_shape_from_heightfield(heightfield[0], elevations, wp.transform_identity(), triangle)
    p = v0
    if corner == 1:
        p = v0 + geom.scale
    elif corner == 2:
        p = v0 + geom.auxiliary
    out[i] = p


@wp.kernel(enable_backward=False)
def _canonical_values(
    data: TerrainFeatureData,
    edges: wp.array[wp.vec3i],
    points: wp.array[wp.vec3],
    faces: wp.array[wp.vec2],
    values: wp.array[wp.vec4],
    planes: wp.array[wp.vec2],
):
    """Per (edge, point): the face values both triangles read at the edge, the edge's values, the plane offsets."""
    i = wp.tid()
    e = edges[i]
    r = e[0]
    c = e[1]
    which = e[2]
    window = terrain_window(data, 0, wp.max(r - 1, 0), wp.max(c - 1, 0))
    window.eps = terrain_eps(1.0)
    k = terrain_window_point(window, points[i])
    # The two triangles of the edge and the component of their face values at it (terrain_face_values order).
    ra = r - 1
    ca = c
    sa = int(1)
    ia = int(1)
    sb = int(0)
    ib = int(0)
    if which == 1:
        ra = r
        ca = c - 1
        sa = 0
        ia = 1
        sb = 1
        ib = 2
    elif which == 2:
        ra = r
        ca = c
        sa = 0
        ia = 2
        sb = 1
        ib = 0
    rb = r
    cb = c
    na, pa = terrain_cluster_plane(data, window, ra, ca, sa)
    nb, pb = terrain_cluster_plane(data, window, rb, cb, sb)
    face_a = terrain_face_values(data, window, ra, ca, sa, na, k)
    face_b = terrain_face_values(data, window, rb, cb, sb, nb, k)
    faces[i] = wp.vec2(face_a[ia], face_b[ib])
    values[i] = terrain_edge_values(data, window, r, c, which, k)
    planes[i] = wp.vec2(wp.dot(na, pa), wp.dot(nb, pb))


def _strip(long_axis: str, heights=None):
    """(heights, hx, hy) of a strip 200 m long along ``long_axis`` with 0.1 m cells."""
    shape = (SHORT_COUNT, LONG_COUNT) if long_axis == "x" else (LONG_COUNT, SHORT_COUNT)
    if heights is None:
        heights = np.zeros(shape)
    hx, hy = (LONG_HALF, SHORT_HALF) if long_axis == "x" else (SHORT_HALF, LONG_HALF)
    return heights, hx, hy


def _frame_gaps(model, long_axis: str):
    """Gaps of today's per-cell frame at every grid line: (along x by column, along y by row) [m], > 0 a crack.

    The gap at column line ``c`` is the x of cell ``(r, c)``'s p00 minus the x of cell ``(r, c - 1)``'s p10.
    """
    row = model.heightfield_data.numpy()[0]
    nrow, ncol = int(row["nrow"]), int(row["ncol"])
    cols = ncol - 1
    r_mid, c_mid = nrow // 2, ncol // 2
    corners = []
    for c in range(1, ncol - 1):
        corners.append((2 * (r_mid * cols + c), 0))  # p00 of cell (r, c)
        corners.append((2 * (r_mid * cols + c - 1), 1))  # p10 of cell (r, c - 1)
    for r in range(1, nrow - 1):
        corners.append((2 * (r * cols + c_mid), 0))  # p00 of cell (r, c)
        corners.append((2 * ((r - 1) * cols + c_mid) + 1, 2))  # p01 of cell (r - 1, c)
    out = wp.zeros(len(corners), dtype=wp.vec3, device=model.device)
    wp.launch(
        _per_cell_corners,
        dim=len(corners),
        inputs=[
            model.heightfield_data,
            model.heightfield_elevations,
            wp.array(corners, dtype=wp.vec2i, device=model.device),
        ],
        outputs=[out],
        device=model.device,
    )
    p = out.numpy().astype(np.float64)
    nx = ncol - 2
    gx = np.zeros(ncol)
    gy = np.zeros(nrow)
    gx[1 : ncol - 1] = p[0 : 2 * nx : 2, 0] - p[1 : 2 * nx : 2, 0]
    gy[1 : nrow - 1] = p[2 * nx :: 2, 1] - p[2 * nx + 1 :: 2, 1]
    return gx, gy


def _lines(gaps_long, distances, half, count, *, gap_only):
    """Grid lines of the long axis nearest the distances (only cracked ones with ``gap_only``)."""
    d = 2.0 * half / (count - 1)
    # A cylinder of 5 cm radius on an edge midpoint of the line stays inside the heightfield.
    lines = np.arange(1, count - 2)
    if gap_only:
        lines = lines[gaps_long[lines] > 0.0]
    return [int(lines[np.argmin(np.abs(-half + lines * d - t))]) for t in distances]


def _placements(model, long_axis: str, *, gap_only: bool):
    """Vertex, x-edge, y-edge and diagonal placements near the distances: (label, x, y, line gaps) in float32."""
    gx, gy = _frame_gaps(model, long_axis)
    row = model.heightfield_data.numpy()[0]
    nrow, ncol = int(row["nrow"]), int(row["ncol"])
    hx, hy = np.float32(row["hx"]), np.float32(row["hy"])
    dx = np.float32(np.float32(2.0) * hx / np.float32(ncol - 1))
    dy = np.float32(np.float32(2.0) * hy / np.float32(nrow - 1))
    if long_axis == "x":
        longs = _lines(gx, DISTANCES, float(hx), ncol, gap_only=gap_only)
        shorts = [r for r in range(1, nrow - 1) if gy[r] > 0.0] if gap_only else [nrow // 2]
        cells = [(shorts[0], c) for c in longs]
    else:
        longs = _lines(gy, DISTANCES, float(hy), nrow, gap_only=gap_only)
        shorts = [c for c in range(1, ncol - 1) if gx[c] > 0.0] if gap_only else [ncol // 2]
        cells = [(r, shorts[0]) for r in longs]
    out = []
    for (r, c), distance in zip(cells, DISTANCES, strict=True):
        # As today's per-cell frame places them: x0 = -hx + c dx in float32.
        x0 = np.float32(-hx + np.float32(c) * dx)
        x1 = np.float32(-hx + np.float32(c + 1) * dx)
        y0 = np.float32(-hy + np.float32(r) * dy)
        y1 = np.float32(-hy + np.float32(r + 1) * dy)
        xm, ym = np.float32(0.5) * (x0 + x1), np.float32(0.5) * (y0 + y1)
        for label, x, y in (("vertex", x0, y0), ("x-edge", xm, y0), ("y-edge", x0, ym), ("diagonal", xm, ym)):
            out.append((f"{long_axis}-strip {label} at {distance} m (r {r}, c {c})", float(x), float(y), gx[c], gy[r]))
    return out


def _cylinder_pose(x, y, tilt):
    q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*TILT_AXIS)), float(tilt)) if tilt else wp.quat_identity()
    pose = np.array([x, y, CYLINDER["half_height"] - PENETRATION, *q], dtype=np.float32)
    return pose.astype(np.float64)


def _check_supports(test, features, hf, placements, tilt):
    """New predicates: every oracle support of the sphere and the cylinder is claimed in float32."""
    sphere_points = np.array([[x, y, SPHERE_RADIUS - PENETRATION] for _l, x, y, _gx, _gy in placements], np.float32)
    sphere_rows = tu.point_rows(features, 0, sphere_points, SPHERE_RADIUS + REACH)
    clusters = ref.cluster_ids(hf.cluster)
    for (label, _x, _y, _gx, _gy), p, got in zip(placements, sphere_points, sphere_rows, strict=True):
        expected = ref.pair_rows(hf, ref.Sphere(p.astype(np.float64), SPHERE_RADIUS), SPHERE_RADIUS + REACH)
        test.assertTrue(expected, msg=label)
        for row in expected:
            test.assertTrue(
                any(np.linalg.norm(g[0] - row.q) <= 1.0e-5 and tu.angle_deg(g[1], row.n) <= 1.0e-3 for g in got),
                msg=f"sphere {label}: support {row} missing from {got}",
            )
        # Flat terrain: one support, the plane's, and no ghost normal.
        test.assertEqual(len(got), 1, msg=f"sphere {label}: {got}")
        test.assertLess(tu.angle_deg(got[0][1], np.array([0.0, 0.0, 1.0])), 1.0e-3, msg=label)
        test.assertAlmostEqual(got[0][2], SPHERE_RADIUS - PENETRATION, delta=1.0e-6, msg=label)
    points, owners = [], []
    for label, x, y, _gx, _gy in placements:
        core = ref.make_core("cylinder", CYLINDER, _cylinder_pose(x, y, tilt))
        rows = ref.pair_rows(hf, core, REACH)
        test.assertGreaterEqual(len(rows), 3, msg=f"cylinder {label}: {rows}")
        for row in rows:
            test.assertEqual(row.feat[0], "F", msg=f"cylinder {label}: {row}")
            points.append(row.k)
            owners.append((label, row, int(clusters[row.feat[1]])))
    claims = tu.face_claims(features, 0, points)
    for (label, row, cluster), claimed in zip(owners, claims, strict=True):
        test.assertTrue(
            any(
                cid == cluster
                and np.linalg.norm(q - row.q) <= 1.0e-5
                and tu.angle_deg(n, row.n) <= 1.0e-3
                and abs(d - row.d) <= 1.0e-6
                for cid, q, n, d in claimed
            ),
            msg=f"cylinder {label}: support {row} unclaimed ({claimed})",
        )


def test_seam_placements_keep_support(test, device):
    """Keep a sphere's and an upright cylinder's supports on grid vertices, edges and diagonals up to 100 m out.

    The placements sit on the grid lines nearest 0.5, 40 and 100 m from the origin, those where today's
    per-cell frame cracks and the exact ones, with the cylinder upright and tilted by 3e-14 rad into the
    cracks. Every float64 support must be claimed by the float32 predicates of the window-local frame.
    """
    for long_axis in ("x", "y"):
        heights, hx, hy = _strip(long_axis)
        model = tu.heightfield_model(heights, hx, hy, device=device)
        features = TerrainFeatures(model)
        hf = tu.reference_surface(model)
        placements = _placements(model, long_axis, gap_only=True) + _placements(model, long_axis, gap_only=False)
        for tilt in (0.0, TILT):
            with test.subTest(long_axis=long_axis, tilt=tilt):
                _check_supports(test, features, hf, placements, tilt)


class TestTerrainLawSurface(unittest.TestCase):
    def test_today_route_loses_vertex_cylinder_cpu(self):
        """Lose every terrain contact of a vertex cylinder tilted 3e-14 rad on today's route (the defect).

        Today's per-cell float32 frame cracks between neighbouring cells by 6-9.5 um on a 200 m terrain. A cylinder
        upright on a vertex whose two grid lines both crack, tilted by 3e-14 rad so that its cap centre moves
        into the cracks, gets no contact from any cell under the suite's terrain flags (the closed-form cylinder
        row clips every rim sample onto a crack). The same placements keep their supports under the law's
        predicates (``test_seam_placements_keep_support``).
        """
        device = "cpu"
        for long_axis in ("x", "y"):
            heights, hx, hy = _strip(long_axis)
            terrain_only = tu.heightfield_model(heights, hx, hy, device=device)
            placements = [p for p in _placements(terrain_only, long_axis, gap_only=True) if " vertex " in p[0]]
            gaps_long = [p[3] if long_axis == "x" else p[4] for p in placements]
            gaps_short = [p[4] if long_axis == "x" else p[3] for p in placements]
            # The per-cell frame cracks: 6.1-7.6 um at these lines, up to 9.5 um on the strip.
            self.assertGreater(min(gaps_long), 6.0e-6, msg=f"{long_axis}: {gaps_long}")
            self.assertGreater(min(gaps_short), 0.0)
            gx, gy = _frame_gaps(terrain_only, long_axis)
            self.assertGreater(max(np.abs(gx).max(), np.abs(gy).max()), 9.0e-6)
            builder = newton.ModelBuilder()
            cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
            data = np.zeros(heights.shape, np.float32)
            builder.add_shape_heightfield(
                heightfield=newton.Heightfield(data=data, nrow=data.shape[0], ncol=data.shape[1], hx=hx, hy=hy), cfg=cfg
            )
            shapes = []
            for label, x, y, _gx, _gy in placements:
                for tilt in (0.0, TILT):
                    pose = _cylinder_pose(x, y, tilt).astype(np.float32)
                    body = builder.add_body(xform=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])))
                    shape = builder.add_shape_cylinder(body, cfg=cfg, **CYLINDER)
                    shapes.append((shape, tilt, label))
            model = builder.finalize(device=device)
            with patch.dict(os.environ, SUITE_FLAGS):
                pipeline = tu.terrain_route_pipeline(
                    model, reduce_contacts=True, rigid_contact_max=1 << 12, max_triangle_pairs=1 << 14
                )
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            s0 = contacts.rigid_contact_shape0.numpy()[:count]
            s1 = contacts.rigid_contact_shape1.numpy()[:count]
            normal = contacts.rigid_contact_normal.numpy()[:count]
            for shape, tilt, label in shapes:
                mine = np.flatnonzero(((s0 == 0) & (s1 == shape)) | ((s1 == 0) & (s0 == shape)))
                support = [i for i in mine if abs(abs(normal[i][2]) - 1.0) < 1.0e-4]
                if tilt:
                    self.assertEqual(len(mine), 0, msg=f"today's route kept contacts of the tilted cylinder at {label}")
                else:
                    self.assertGreaterEqual(len(support), 3, msg=f"upright cylinder at {label}")


def _canonical(features, edges, points):
    device = features.device
    n = len(edges)
    faces = wp.zeros(n, dtype=wp.vec2, device=device)
    values = wp.zeros(n, dtype=wp.vec4, device=device)
    planes = wp.zeros(n, dtype=wp.vec2, device=device)
    wp.launch(
        _canonical_values,
        dim=n,
        inputs=[
            features.data(),
            wp.array(np.asarray(edges, np.int32), dtype=wp.vec3i, device=device),
            wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device=device),
        ],
        outputs=[faces, values, planes],
        device=device,
    )
    return faces.numpy(), values.numpy(), planes.numpy()


def _assert_canonical(test, device, faces, values, same):
    """Face and edge read one value at an edge, negated across it inside one cluster.

    Bit for bit on the CPU. CUDA's default contraction may fuse the call sites differently; the values must then
    agree within a quarter of the band (:func:`terrain_eps`), so a closed claim on either side covers the other's.
    """
    tolerance = 0.0 if device.is_cpu else 0.25 * CANONICAL_BAND
    for (fa, fb), v, flat in zip(faces, values, same, strict=True):
        test.assertLessEqual(abs(float(fa) - float(v[0])), tolerance)
        test.assertLessEqual(abs(float(fb) - float(v[1])), tolerance)
        if flat:
            test.assertLessEqual(abs(float(fa) + float(fb)), tolerance)


def test_window_frame_has_no_cracks(test, device):
    """Compute a vertex shared by neighbouring cells identically in the window-local frame.

    Today's per-cell frame places the same vertex up to 9.5 um apart in neighbouring cells; the window frame
    derives every vertex from integer index differences, so all cells of a window read one value. The face values
    of the two cells meeting at the worst line, read for points on it, are exact negatives inside one cluster, and
    the edge's own values equal the faces' (on CUDA, within a quarter of the band).
    """
    heights, hx, hy = _strip("x")
    model = tu.heightfield_model(heights, hx, hy, device=device)
    features = TerrainFeatures(model)
    # The worst crack of today's per-cell frame, as the CPU computes it.
    gx, _gy = _frame_gaps(tu.heightfield_model(heights, hx, hy, device="cpu"), "x")
    line = int(np.argmax(np.abs(gx)))
    test.assertGreater(abs(gx[line]), 9.0e-6)
    r = SHORT_COUNT // 2
    row = model.heightfield_data.numpy()[0]
    x = np.float32(-np.float32(row["hx"]) + np.float32(line) * np.float32(0.1))
    edges = [(r, line, 1), (r, line, 1), (r, line - 1, 2), (r, line, 0)]
    points = [(x, -0.03, 0.02), (x, 0.05, -0.01), (x - 0.05, 0.05, 0.0), (x + 0.03, 0.0, 0.1)]
    faces, values, _planes = _canonical(features, edges, points)
    _assert_canonical(test, device, faces, values, [True] * len(edges))


def test_canonical_values_on_rough_terrain(test, device):
    """Read one value on both sides of every boundary: face and edge share ``s``, a cluster negates it."""
    rng = np.random.default_rng(5)
    heights = tu.suite_like_heights(rng, 64, 64)
    model = tu.heightfield_model(heights, 3.2, 3.2, device=device)
    features = TerrainFeatures(model)
    hf = tu.reference_surface(model)
    clusters = ref.cluster_ids(hf.cluster)
    edges, points, same = [], [], []
    for _ in range(600):
        r, c = (int(v) for v in rng.integers(1, 63, size=2))
        which = int(rng.integers(0, 3))
        a = {0: (r - 1, c, 1), 1: (r, c - 1, 0), 2: (r, c, 0)}[which]
        b = {0: (r, c, 0), 1: (r, c, 1), 2: (r, c, 1)}[which]
        v0 = hf.V(r, c)
        v1 = hf.V(*{0: (r, c + 1), 1: (r + 1, c), 2: (r + 1, c + 1)}[which])
        p = v0 + rng.uniform(-0.2, 1.2) * (v1 - v0) + np.array([0.0, 0.0, rng.uniform(-0.02, 0.05)])
        edges.append((r, c, which))
        points.append(p)
        same.append(clusters[hf.tri_id(*a)] == clusters[hf.tri_id(*b)])
    faces, values, _planes = _canonical(features, edges, points)
    _assert_canonical(test, device, faces, values, same)
    test.assertGreater(sum(same), 50)
    test.assertLess(sum(same), len(same) - 50)


def _rough_strip(rng, long_axis):
    """A 200 m strip of the suite's random-rough sub-terrain (uniform heights in 5 mm steps on 0.1 m cells)."""
    shape = (21, LONG_COUNT) if long_axis == "x" else (LONG_COUNT, 21)
    heights = np.round(rng.uniform(0.0, 0.1, size=shape) / 0.005) * 0.005
    hx, hy = (LONG_HALF, 1.0) if long_axis == "x" else (1.0, LONG_HALF)
    return heights, hx, hy


def _seam_poses(rng, hf, kind, count):
    """Seam-adversarial placements: 15% on x-lines, 15% y-lines, 10% vertices, 10% diagonals, 50% generic."""
    poses = []
    for _ in range(count):
        j = int(rng.integers(5, hf.cols - 5))
        i = int(rng.integers(5, hf.rows - 5))
        u = rng.uniform()
        fx, fy = rng.uniform(0.0, 1.0, size=2)
        if u < 0.15:
            fx = 0.0
        elif u < 0.30:
            fy = 0.0
        elif u < 0.40:
            fx, fy = 0.0, 0.0
        elif u < 0.50:
            fy = fx
        x, y = -hf.hx + (j + fx) * hf.dx, -hf.hy + (i + fy) * hf.dy
        if kind == "sphere":
            q = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            mode = rng.choice(["upright", "lying", "free"])
            if mode == "upright":
                q = np.array([0.0, 0.0, 0.0, 1.0])
            elif mode == "lying":
                yaw = rng.uniform(0.0, 2.0 * np.pi)
                q = np.asarray(
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))
                    * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.pi / 2))
                )
            else:
                q = rng.normal(size=4)
                q /= np.linalg.norm(q)
        poses.append(np.array([x, y, 0.0, *q], np.float32).astype(np.float64))
    return poses


def test_seam_adversarial_placements_are_watertight(test, device):
    """Claim every float64 support in float32 for spheres and cylinders snapped onto seams up to 100 m out.

    The port of the design study's float32 seam check: on the suite's random-rough sub-terrain, half of the
    placements sit exactly on x-lines, y-lines, vertices or diagonals. Sphere supports must be reproduced by the
    float32 point rows; a cylinder's face supports must be claimed by a triangle of their cluster, its crease
    supports must sit on edges the float32 classes call convex, and its vertex supports on float32 vertex features
    whose cone holds the support normal.
    """
    rng = np.random.default_rng(11)
    margin, gap = 0.01, 0.01
    misses = []
    supports = 0
    for long_axis in ("x", "y"):
        heights, hx, hy = _rough_strip(rng, long_axis)
        model = tu.heightfield_model(heights, hx, hy, device=device)
        features = TerrainFeatures(model)
        hf = tu.reference_surface(model)
        clusters = ref.cluster_ids(hf.cluster)
        # spheres: the point rows
        cores, points, reach = [], [], []
        for pose in _seam_poses(rng, hf, "sphere", 300):
            radius = float(rng.choice([0.022, 0.03146]))
            x, y = pose[:2]
            z = np.float32(max(hf.height(x + dx, y + dy) for dx in (-radius, 0, radius) for dy in (-radius, 0, radius)))
            p = np.array([x, y, z + radius + rng.uniform(-0.001, 0.010)], np.float32).astype(np.float64)
            cores.append(ref.Sphere(p, radius))
            points.append(p)
            reach.append(radius + 2 * margin + 2 * gap)
        got_all = tu.point_rows(features, 0, points, reach)
        for core, d_reach, got in zip(cores, reach, got_all, strict=True):
            for row in ref.pair_rows(hf, core, d_reach):
                if row.d > core.r_eff + 2 * margin + 0.001:
                    continue
                supports += 1
                if not any(np.linalg.norm(g[0] - row.q) <= 1.0e-4 and tu.angle_deg(g[1], row.n) <= 1.0 for g in got):
                    misses.append(("sphere", long_axis, row, got))
        # cylinders: face claims, edge classes, vertex features and cones
        face_points, face_rows, queries, query_rows = [], [], [], []
        for pose in _seam_poses(rng, hf, "cylinder", 150):
            core = ref.make_core("cylinder", {"radius": 0.02, "half_height": 0.0625}, pose)
            lo, hi = core.aabb()
            top = max(hf.height(x, y) for x in np.linspace(lo[0], hi[0], 5) for y in np.linspace(lo[1], hi[1], 5))
            placed = pose.copy()
            placed[2] = np.float32(top - lo[2] + rng.uniform(-0.001, 0.010))
            core = ref.make_core("cylinder", {"radius": 0.02, "half_height": 0.0625}, placed)
            for row in ref.pair_rows(hf, core, 2 * margin + 2 * gap):
                if row.d > 2 * margin + 0.001:
                    continue
                supports += 1
                if row.feat[0] == "F":
                    face_points.append(row.k)
                    face_rows.append(row)
                elif row.feat[0] == "E":
                    queries.append((row.feat[2], row.feat[3], {"H": 0, "V": 1, "D": 2}[row.feat[1]]))
                    query_rows.append(row)
                elif row.feat[0] == "V":
                    queries.append((row.feat[1], row.feat[2], 0))
                    query_rows.append(row)
                else:
                    misses.append(("cylinder recovery row on a placement above the surface", long_axis, row, None))
        for row, claimed in zip(face_rows, tu.face_claims(features, 0, face_points), strict=True):
            cluster = int(clusters[row.feat[1]])
            if not any(
                cid == cluster and np.linalg.norm(q - row.q) <= 1.0e-4 and tu.angle_deg(n, row.n) <= 1.0
                for cid, q, n, _d in claimed
            ):
                misses.append(("cylinder face", long_axis, row, claimed))
        if queries:
            answers = tu.feature_queries(features, 0, queries, directions=[row.n for row in query_rows])
            for row, (kind, feature, cone) in zip(query_rows, answers, strict=True):
                if row.feat[0] == "E" and kind != TERRAIN_EDGE_CONVEX:
                    misses.append(("cylinder edge class", long_axis, row, kind))
                if row.feat[0] == "V" and not (feature and cone):
                    misses.append(("cylinder vertex", long_axis, row, (feature, cone)))
    test.assertGreater(supports, 500)
    test.assertEqual(misses, [], msg=f"{len(misses)} of {supports} supports missed")


def test_plane_distances_stay_window_precise_on_large_clusters(test, device):
    """Keep float32 plane distances far inside the band on a 16 m planar slope that is one cluster.

    The cluster's plane record is anchored at its lowest-id vertex, up to 16 m from a window. Read at that anchor,
    the plane carried the float32 rounding of the anchor's distance into every distance: up to 1.6 um off,
    against a band of 0.38 um for these windows.
    """
    count = 161
    x = np.arange(count) * 0.1
    X, Y = np.meshgrid(x, x)
    half = 0.05 * (count - 1)
    model = tu.heightfield_model(0.4 * (X + 0.5 * Y), half, half, device=device)
    features = TerrainFeatures(model)
    hf = tu.reference_surface(model)
    test.assertEqual(features.plane_count(0), 1)
    test.assertEqual(len(np.unique(features.clusters.numpy())), 1)
    rng = np.random.default_rng(17)
    points = []
    for _ in range(400):
        px, py = rng.uniform(-half + 0.2, half - 0.2, size=2)
        points.append((px, py, hf.height(px, py) + rng.uniform(-0.002, 0.002)))
    points = np.asarray(points, np.float32).astype(np.float64)
    worst = 0.0
    for p, claimed in zip(points, tu.face_claims(features, 0, points), strict=True):
        test.assertTrue(claimed, msg=f"{p} unclaimed")
        n, a = hf.plane(hf.locate(p[0], p[1]))
        for _cluster, _q, _n, d in claimed:
            worst = max(worst, abs(d - float(n @ (p - a))))
    test.assertLess(worst, 0.25 * WINDOW_BAND, msg=f"worst {worst * 1e6:.3f} um")


devices = get_test_devices()
add_function_test(
    TestTerrainLawSurface, "test_window_frame_has_no_cracks", test_window_frame_has_no_cracks, devices=devices
)
add_function_test(
    TestTerrainLawSurface,
    "test_canonical_values_on_rough_terrain",
    test_canonical_values_on_rough_terrain,
    devices=devices,
)
add_function_test(
    TestTerrainLawSurface,
    "test_plane_distances_stay_window_precise_on_large_clusters",
    test_plane_distances_stay_window_precise_on_large_clusters,
    devices=devices,
)
add_function_test(
    TestTerrainLawSurface, "test_seam_placements_keep_support", test_seam_placements_keep_support, devices=devices
)
add_function_test(
    TestTerrainLawSurface,
    "test_seam_adversarial_placements_are_watertight",
    test_seam_adversarial_placements_are_watertight,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixtures of the terrain law's solid-core tests: scenes of solids over a heightfield, the float64 law's rows of a
solid pair, comparisons of row sets and an exact-geometry view of a pose (sampled separation, normal cones).

The geometry here (boundary samples, closest points on the triangulated surface, normal cones by non-negative least
squares) is independent of both the float32 kernels and the float64 law (:mod:`newton.tests.terrain_law_reference`).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
from newton._src.geometry.terrain_law import (
    TERRAIN_CELL_WIDEN,
    TERRAIN_EPS32,
    TERRAIN_EPS_FACTOR,
    TERRAIN_EPS_MIN_EXTENT,
)
from newton.tests import terrain_law_reference as ref
from newton.tests import terrain_law_test_utils as tu

MARGIN = 0.01
"""Margin of every shape of :func:`solid_scene` [m] (the suite's)."""
GAP = 0.01
"""Contact gap of every shape of :func:`solid_scene` [m] (the suite's)."""
REACH = 4 * MARGIN
"""A solid pair's reach: both margins and both gaps [m]."""
SOLID_KINDS = ("box", "cylinder", "cone", "ellipsoid", "hull")


def sole_vertices(*, length=0.22, width=0.09, height=0.035, bevel=0.008, n=12) -> np.ndarray:
    """A convex sole: a stadium outline 22 x 9 x 3.5 cm whose flat bottom is inset by ``bevel`` (a prism with
    ``bevel`` 0), float32 vertices (as a model holds them)."""
    half_l, rad = length / 2, width / 2
    straight = half_l - rad
    pts = []
    for k in range(n + 1):
        a = -np.pi / 2 + np.pi * k / n
        pts.append((straight + rad * np.cos(a), rad * np.sin(a)))
    for k in range(n + 1):
        a = np.pi / 2 + np.pi * k / n
        pts.append((-straight + rad * np.cos(a), rad * np.sin(a)))
    outline = np.asarray(pts)
    centre = outline.mean(0)
    if bevel > 0.0:
        inset = centre + (outline - centre) * (1.0 - bevel / rad)
        verts = [(x, y, -height / 2) for x, y in inset]
        verts += [(x, y, -height / 2 + bevel) for x, y in outline]
    else:
        verts = [(x, y, -height / 2) for x, y in outline]
    verts += [(x, y, height / 2) for x, y in outline]
    return np.asarray(verts, np.float32).astype(np.float64)


def random_hull_vertices(rng, count=14, size=(0.06, 0.05, 0.03)) -> np.ndarray:
    """Points of a random convex blob (float32 values), its hull's vertices among them."""
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return (d * np.asarray(size) * rng.uniform(0.7, 1.0, size=(count, 1))).astype(np.float32).astype(np.float64)


def add_solid(builder, kind, dims, pose, cfg):
    """Add a body at ``pose`` (px, py, pz, qx, qy, qz, qw) with one shape of ``kind``; return the shape."""
    p = np.asarray(pose, np.float64)
    body = builder.add_body(xform=wp.transform(wp.vec3(*p[:3]), wp.quat(*p[3:7])))
    if kind == "box":
        h = dims["half"]
        return builder.add_shape_box(body, hx=h[0], hy=h[1], hz=h[2], cfg=cfg)
    if kind == "cylinder":
        return builder.add_shape_cylinder(body, radius=dims["radius"], half_height=dims["half_height"], cfg=cfg)
    if kind == "barrel":
        return builder.add_shape_cylinder(
            body, radius=dims["radius"], half_height=dims["half_height"], barrel_radius=dims["barrel"], cfg=cfg
        )
    if kind == "plane":
        return builder.add_shape_plane(body=body, width=dims["size"][0], length=dims["size"][1], cfg=cfg)
    if kind == "cone":
        return builder.add_shape_cone(body, radius=dims["radius"], half_height=dims["half_height"], cfg=cfg)
    if kind == "ellipsoid":
        s = dims["semi"]
        return builder.add_shape_ellipsoid(body, rx=s[0], ry=s[1], rz=s[2], cfg=cfg)
    if kind == "hull":
        mesh = newton.Mesh(
            vertices=np.asarray(dims["verts"], np.float32), indices=np.array([0, 1, 2], np.int32), compute_inertia=False
        )
        return builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg)
    if kind == "sphere":
        return builder.add_shape_sphere(body, radius=dims["radius"], cfg=cfg)
    if kind == "capsule":
        return builder.add_shape_capsule(body, radius=dims["radius"], half_height=dims["half_height"], cfg=cfg)
    raise ValueError(kind)


def solid_scene(heights, hx, hy, bodies, *, device, margin=MARGIN, gap=GAP):
    """A heightfield at the origin and bodies, each alone in a world over it: ``(model, heightfield, shapes)``.

    ``bodies``: ``(kind, dims, pose)`` per body (:func:`add_solid`).
    """
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=margin, gap=gap)
    data = np.asarray(heights, np.float32)
    terrain = newton.Heightfield(data=data, nrow=data.shape[0], ncol=data.shape[1], hx=hx, hy=hy)
    hfield = builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
    shapes = []
    for kind, dims, pose in bodies:
        builder.begin_world()
        shapes.append(add_solid(builder, kind, dims, pose, cfg))
        builder.end_world()
    return builder.finalize(device=device), hfield, shapes


def solid_kernel_band(hf: ref.ReferenceHeightfield, core, reach: float) -> float:
    """The float32 kernels' closed-predicate band of a solid pair's window: the core's bounds widened by the reach
    (cell coordinates widened by ``TERRAIN_CELL_WIDEN``), the extent of the window's size and heights and of the
    core's bounds about the window's reference vertex."""
    lo, hi = core.aabb()
    c0 = max(int(np.floor((lo[0] - reach + hf.hx) / hf.dx - TERRAIN_CELL_WIDEN)), 0)
    c1 = min(int(np.floor((hi[0] + reach + hf.hx) / hf.dx + TERRAIN_CELL_WIDEN)), hf.cols - 1)
    r0 = max(int(np.floor((lo[1] - reach + hf.hy) / hf.dy - TERRAIN_CELL_WIDEN)), 0)
    r1 = min(int(np.floor((hi[1] + reach + hf.hy) / hf.dy + TERRAIN_CELL_WIDEN)), hf.rows - 1)
    origin = hf.V(r0, c0)
    extent = max(
        (c1 + 1 - c0) * hf.dx,
        (r1 + 1 - r0) * hf.dy,
        float(np.abs(hf.Z[r0 : r1 + 2, c0 : c1 + 2] - origin[2]).max()),
        float(np.abs(lo - origin).max()),
        float(np.abs(hi - origin).max()),
    )
    return TERRAIN_EPS_FACTOR * TERRAIN_EPS32 * max(extent, TERRAIN_EPS_MIN_EXTENT)


def law_rows(hf: ref.ReferenceHeightfield, kind, dims, pose, *, reach=REACH, row_limit=None):
    """The float64 law's rows of a solid pair, its plane groups at the float32 finish's offset (four kernel bands)."""
    core = ref.make_core(kind, dims, pose)
    lo, hi = core.aabb()
    box = hf.cells_in_box(lo[:2] - reach, hi[:2] + reach)
    if box is None:
        return []
    window = hf.window_features(box)
    raw = ref.solid_law(hf, core, reach, window)
    if not raw:
        raw = ref._solid_recovery(hf, core)
    return ref.finish(raw, row_limit=row_limit, group_offset=4.0 * solid_kernel_band(hf, core, reach))


def implied_by(row, others, *, normal_tol=1.0e-4, plane_tol=5.0e-6, d_tol=5.0e-6) -> bool:
    """Whether rows ``others`` imply ``row``: the lower convex hull of (tangent position, distance) of the rows of its
    plane among them is at most its distance at its tangent position (it adds no constraint they lack)."""
    q, n, d = tu.row_qnd(row)
    same = [tu.row_qnd(o) for o in others]
    same = [o for o in same if np.linalg.norm(o[1] - n) <= normal_tol and abs(float(n @ (o[0] - q))) <= plane_tol]
    if not same:
        return False
    k = q + d * n
    points = np.array([o[0] + o[2] * o[1] for o in same])
    distances = np.array([o[2] for o in same])
    value = ref.lower_hull_value(points, distances, k, n, tol=1.0e-6)
    return value is not None and value <= d + d_tol


def compare_solid_rows(law, got, *, frame_tol=1.0e-5, angle_tol=1.0e-3, d_tol=1.0e-5):
    """Compare float32 rows with the float64 law's rows of one solid pair.

    Rows match with terrain points within ``frame_tol``, normals within ``angle_tol`` and distances within ``d_tol``;
    an unmatched row that the other side's rows imply (:func:`implied_by`: the finish kept an equivalent subset of a
    flat support, its ties decided by rounding) is no difference. Returns ``(missing, extra, deepest)``: unmatched
    float64 rows, unmatched float32 rows and the difference of the deepest distances (infinite when one side has rows
    and the other none).
    """
    missing = tu.missing_rows(law, got, q_tol=frame_tol, angle_tol=angle_tol, d_tol=d_tol)
    extra = tu.missing_rows(got, law, q_tol=frame_tol, angle_tol=angle_tol, d_tol=d_tol)
    missing = [m for m in missing if not implied_by(m, got)]
    extra = [e for e in extra if not implied_by(e, law)]
    deepest = 0.0
    if law and got:
        deepest = abs(min(tu.row_qnd(r)[2] for r in law) - min(tu.row_qnd(g)[2] for g in got))
    elif law or got:
        deepest = math.inf
    return missing, extra, deepest


# ----------------------------------------------------------------------------------------------------------------
# exact geometry of a pose
# ----------------------------------------------------------------------------------------------------------------


def _grid(n):
    return np.linspace(0.0, 1.0, max(int(n), 2))


def boundary_samples(core, step=0.002) -> np.ndarray:
    """Points on a core's boundary, about ``step`` apart, every vertex, edge and rim included."""
    pts = []
    kind = core.kind
    if kind in ("box", "hull"):
        verts = core.verts
        for face, normal in zip(core.faces, core.fnormals, strict=True):
            loop = verts[face]
            centre = loop.mean(axis=0)
            # the face's vertices around its normal, fanned from its centre (the face is convex)
            e1 = loop[0] - centre
            e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-12)
            e2 = np.cross(normal, e1)
            loop = loop[np.argsort(np.arctan2((loop - centre) @ e2, (loop - centre) @ e1))]
            for i in range(len(loop)):
                a, b = loop[i], loop[(i + 1) % len(loop)]
                n_ab = int(np.ceil(np.linalg.norm(b - a) / step)) + 1
                n_ac = int(np.ceil(np.linalg.norm(centre - a) / step)) + 1
                u, v = np.meshgrid(_grid(n_ab), _grid(n_ac), indexing="ij")
                u, v = u.ravel(), v.ravel()
                keep = u + v <= 1.0 + 1e-12
                pts.append(a + u[keep, None] * (b - a) + v[keep, None] * (centre - a))
        return np.concatenate(pts)
    if kind == "ellipsoid":
        nth = int(np.ceil(np.pi * float(core.s.max()) / step)) + 1
        th = np.linspace(0.0, np.pi, nth)
        ph = np.linspace(0.0, 2.0 * np.pi, 2 * nth, endpoint=False)
        T, P = np.meshgrid(th, ph, indexing="ij")
        unit = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
        return core.c + (unit * core.s) @ core.R.T
    ax = core.axis
    u, w = ref._disk_basis(ax)
    m = int(np.ceil(2.0 * np.pi * core.r / step)) + 1
    phi = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    ring = np.outer(np.cos(phi), u) + np.outer(np.sin(phi), w)
    if kind in ("cylinder", "barrel"):
        # the side by arc length along its profile
        zs, radii = _side_profile(core, np.linspace(0.0, 1.0, int(np.ceil(_side_length(core) / step)) + 1))
        pts.append((core.c + zs[:, None, None] * ax + radii[:, None, None] * ring[None]).reshape(-1, 3))
        for sign in (-1.0, 1.0):
            for rho in np.linspace(0.0, core.r, int(np.ceil(core.r / step)) + 1):
                pts.append(core.c + sign * core.h * ax + rho * ring)
        return np.concatenate(pts)
    # cone: lateral surface from the base rim to the apex, and the base disk
    for t in np.linspace(0.0, 1.0, int(np.ceil(math.hypot(core.r, 2.0 * core.h) / step)) + 1):
        pts.append(core.base + t * (core.apex - core.base) + (1.0 - t) * core.r * ring)
    for rho in np.linspace(0.0, core.r, int(np.ceil(core.r / step)) + 1):
        pts.append(core.base + rho * ring)
    return np.concatenate(pts)


def _side_profile(core, v) -> tuple[np.ndarray, np.ndarray]:
    """A cylinder's side profile at parameters ``v`` in ``[0, 1]``, by arc length: ``(z, radius)``. A barrel's arc
    (:class:`terrain_law_reference.Barrel`) by its angle, so that its steep ends near the rims are sampled as finely as
    its middle."""
    v = np.asarray(v, np.float64)
    if core.kind != "barrel":
        return (2.0 * v - 1.0) * core.h, np.full(v.shape, core.r)
    span = math.asin(min(core.h / core.R, 1.0))
    phi = (2.0 * v - 1.0) * span
    x0 = core.r - math.sqrt(max(core.R**2 - core.h**2, 0.0))
    return core.R * np.sin(phi), x0 + core.R * np.cos(phi)


def _side_length(core) -> float:
    """The length of a cylinder's side profile (a barrel's arc)."""
    if core.kind != "barrel":
        return 2.0 * core.h
    return 2.0 * core.R * math.asin(min(core.h / core.R, 1.0))


def _surface_triangles(hf: ref.ReferenceHeightfield, lo, hi):
    box = hf.cells_in_box(np.asarray(lo[:2]), np.asarray(hi[:2]))
    if box is None:
        return np.zeros((0, 3, 3))
    tris = hf.window_features(box)[0]
    return np.array([hf.tri_verts(t) for t in tris])


def surface_height(hf: ref.ReferenceHeightfield, x, y) -> np.ndarray:
    """Heights of the triangulated surface (cell diagonal p00-p11) under points ``(x, y)`` (vectorized; NaN outside)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    u = (x + hf.hx) / hf.dx
    v = (y + hf.hy) / hf.dy
    c = np.clip(np.floor(u).astype(np.int64), 0, hf.cols - 1)
    r = np.clip(np.floor(v).astype(np.int64), 0, hf.rows - 1)
    fx = u - c
    fy = v - r
    z00, z10 = hf.Z[r, c], hf.Z[r, c + 1]
    z01, z11 = hf.Z[r + 1, c], hf.Z[r + 1, c + 1]
    lower = fx >= fy
    z = np.where(lower, z00 + fx * (z10 - z00) + fy * (z11 - z10), z00 + fy * (z01 - z00) + fx * (z11 - z01))
    outside = (u < 0.0) | (v < 0.0) | (u > hf.cols) | (v > hf.rows)
    return np.where(outside, np.nan, z)


def signed_surface_distance(hf: ref.ReferenceHeightfield, points, *, reach=0.1, prune=True) -> np.ndarray:
    """Signed Euclidean distance of points to the triangulated surface (negative below it), numpy.

    With ``prune``, only the points whose distance can be the smallest get an exact distance (the others get their
    lower bound, the vertical gap times the steepest triangle's cosine): the minimum is exact. The triangles searched
    lie within ``reach`` of the points, or within their largest vertical gap where that is more (a point's closest
    surface point lies no farther than the surface point above or below it).
    """
    points = np.asarray(points, np.float64).reshape(-1, 3)
    gap = points[:, 2] - surface_height(hf, points[:, 0], points[:, 1])
    finite = np.abs(gap[np.isfinite(gap)])
    if finite.size:
        reach = max(reach, float(finite.max()))
    v = _surface_triangles(hf, points.min(axis=0) - reach, points.max(axis=0) + reach)
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    cos_steep = float(np.clip(np.abs(normals[:, 2] / np.linalg.norm(normals, axis=1)).min(), 0.05, 1.0))
    low = np.where(gap < 0.0, gap, gap * cos_steep)
    up = np.where(gap < 0.0, gap * cos_steep, gap)
    out = low.copy()
    exact = low <= np.nanmin(up) + 1.0e-9 if prune else np.ones(len(points), bool)
    sel = points[exact]
    best = np.full(len(sel), np.inf)
    for start in range(0, len(v), 64):
        chunk = v[start : start + 64]
        a = np.broadcast_to(chunk[None, :, 0], (len(sel), len(chunk), 3))
        b = np.broadcast_to(chunk[None, :, 1], a.shape)
        c = np.broadcast_to(chunk[None, :, 2], a.shape)
        p = np.broadcast_to(sel[:, None, :], a.shape)
        q = _closest_on_triangles(p.reshape(-1, 3), a.reshape(-1, 3), b.reshape(-1, 3), c.reshape(-1, 3))
        dist = np.linalg.norm(q - p.reshape(-1, 3), axis=1).reshape(len(sel), len(chunk))
        best = np.minimum(best, dist.min(axis=1))
    out[exact] = np.where(gap[exact] < 0.0, -best, best)
    return out


def _closest_on_triangles(p, a, b, c):
    """Closest points of triangles ``abc`` to points ``p`` (vectorized Voronoi regions)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1.0 / (va + vb + vc)
        out = a + ab * (vb * den)[:, None] + ac * (vc * den)[:, None]
        edge_ab = a + ab * (d1 / (d1 - d3))[:, None]
        edge_ac = a + ac * (d2 / (d2 - d6))[:, None]
        edge_bc = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]
    sel = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    out = np.where(sel[:, None], edge_bc, out)
    sel = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    out = np.where(sel[:, None], edge_ac, out)
    sel = (d6 >= 0) & (d5 <= d6)
    out = np.where(sel[:, None], c, out)
    sel = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    out = np.where(sel[:, None], edge_ab, out)
    sel = (d3 >= 0) & (d4 <= d3)
    out = np.where(sel[:, None], b, out)
    sel = (d1 <= 0) & (d2 <= 0)
    return np.where(sel[:, None], a, out)


def sampled_separation(hf: ref.ReferenceHeightfield, core, *, step=0.002) -> float:
    """The smallest signed surface distance over a core's boundary samples: no less than the exact separation (or
    minus the deepest point's depth); every row of a complete law is at least this deep."""
    return float(signed_surface_distance(hf, boundary_samples(core, step)).min())


def _fan_patch(a, b, c):
    """Triangle ``abc`` over the unit square (``u + v > 1`` folded back), its sides' lengths as the patch's scales."""

    def fn(u, v):
        fold = u + v > 1.0
        uu = np.where(fold, 1.0 - u, u)
        vv = np.where(fold, 1.0 - v, v)
        return a[None] + uu[:, None] * (b - a)[None] + vv[:, None] * (c - a)[None]

    return fn, float(np.linalg.norm(b - a)), float(np.linalg.norm(c - a))


def boundary_patches(core) -> list:
    """A core's boundary as parametric patches over the unit square: ``(fn, su, sv)``, ``fn(u, v)`` the points (world)
    and ``su``, ``sv`` the lengths the parameters span [m]. Their union is the boundary, edges and vertices included."""
    kind = core.kind
    patches = []
    if kind in ("box", "hull"):
        verts = core.verts
        for face, normal in zip(core.faces, core.fnormals, strict=True):
            loop = verts[face]
            centre = loop.mean(axis=0)
            e1 = loop[0] - centre
            e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-12)
            e2 = np.cross(normal, e1)
            loop = loop[np.argsort(np.arctan2((loop - centre) @ e2, (loop - centre) @ e1))]
            for i in range(len(loop)):
                patches.append(_fan_patch(centre, loop[i], loop[(i + 1) % len(loop)]))
        return patches
    if kind == "ellipsoid":

        def shell(u, v):
            th, ph = np.pi * u, 2.0 * np.pi * v
            unit = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
            return core.c + (unit * core.s) @ core.R.T

        smax = float(core.s.max())
        return [(shell, np.pi * smax, 2.0 * np.pi * smax)]
    ax = core.axis
    w1, w2 = ref._disk_basis(ax)

    def ring(u):
        th = 2.0 * np.pi * u
        return np.cos(th)[:, None] * w1[None] + np.sin(th)[:, None] * w2[None]

    if kind in ("cylinder", "barrel"):

        def side(u, v):
            zs, radii = _side_profile(core, v)
            return core.c + zs[:, None] * ax + radii[:, None] * ring(u)

        patches.append((side, 2 * np.pi * float(_side_profile(core, np.full(1, 0.5))[1][0]), _side_length(core)))
        for sign in (-1.0, 1.0):
            patches.append(
                (
                    lambda u, v, sign=sign: core.c + sign * core.h * ax + (core.r * v)[:, None] * ring(u),
                    2 * np.pi * core.r,
                    core.r,
                )
            )
        return patches
    # cone: the lateral surface from the base rim (v = 0) to the apex (v = 1), and the base disk
    slant = math.hypot(core.r, 2.0 * core.h)
    patches.append(
        (
            lambda u, v: core.base + v[:, None] * (core.apex - core.base) + ((1.0 - v) * core.r)[:, None] * ring(u),
            2 * np.pi * core.r,
            slant,
        )
    )
    patches.append((lambda u, v: core.base + (core.r * v)[:, None] * ring(u), 2 * np.pi * core.r, core.r))
    return patches


def refined_separation(hf: ref.ReferenceHeightfield, core, *, step=0.003, levels=3) -> float:
    """A core's exact signed separation from the surface (minus the deepest point's depth when penetrating), within
    about a micrometre: the boundary patches (:func:`boundary_patches`) sampled about ``step`` apart, then each patch
    that can hold the minimum refined around its four best samples on 9 x 9 grids, each level a quarter of the last
    one's span (as the symptom bench's exact separation does)."""
    patches = []
    best = math.inf
    for fn, su, sv in boundary_patches(core):
        nu, nv = int(np.ceil(su / step)) + 1, int(np.ceil(sv / step)) + 1
        u, v = np.meshgrid(_grid(nu), _grid(nv), indexing="ij")
        u, v = u.ravel(), v.ravel()
        patches.append((fn, u, v, 1.0 / (nu - 1), 1.0 / (nv - 1)))
    values = signed_surface_distance(hf, np.concatenate([fn(u, v) for fn, u, v, _du, _dv in patches]))
    best = float(values.min())
    first = 0
    g = np.linspace(-1.0, 1.0, 9)
    for fn, u, v, du, dv in patches:
        pruned = values[first : first + len(u)]
        first += len(u)
        # a patch whose samples all lie more than its spacing above the best cannot hold a lower point
        if pruned.min() > best + step:
            continue
        order = np.argsort(pruned)[:4]
        exact = signed_surface_distance(hf, fn(u[order], v[order]), prune=False)
        best = min(best, float(exact.min()))
        cu, cv = u[order], v[order]
        for level in range(levels):
            span_u, span_v = du / (4.0**level), dv / (4.0**level)
            uu = np.clip(cu[:, None, None] + g[None, :, None] * span_u, 0.0, 1.0).repeat(9, axis=2).reshape(-1)
            vv = np.clip(cv[:, None, None] + g[None, None, :] * span_v, 0.0, 1.0).repeat(9, axis=1).reshape(-1)
            vals = signed_surface_distance(hf, fn(uu, vv), prune=False)
            best = min(best, float(vals.min()))
            keep = np.argsort(vals)[:4]
            cu, cv = uu[keep], vv[keep]
    return best


def _nnls(A, b, *, iterations=200):
    """Non-negative least squares (Lawson-Hanson active set) for small dense problems."""
    m = A.shape[1]
    x = np.zeros(m)
    passive = np.zeros(m, bool)
    for _ in range(iterations):
        w = A.T @ (b - A @ x)
        candidates = np.flatnonzero(~passive & (w > 1.0e-12))
        if not len(candidates):
            break
        passive[candidates[np.argmax(w[candidates])]] = True
        while True:
            z = np.zeros(m)
            z[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if np.all(z[passive] > 0.0):
                x = z
                break
            neg = passive & (z <= 0.0)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x)
            passive &= x > 1.0e-15
    return x


def normal_cone_angle(hf: ref.ReferenceHeightfield, q, n, *, on_tol=1.0e-6) -> float:
    """Angle [deg] from ``n`` to the cone spanned by the normals of the surface triangles containing ``q`` (0 when it
    lies inside: a legal support direction of the true surface at ``q``); ``nan`` off the surface."""
    q = np.asarray(q, np.float64)
    v = _surface_triangles(hf, q - 0.2, q + 0.2)
    cl = _closest_on_triangles(np.broadcast_to(q, (len(v), 3)), v[:, 0], v[:, 1], v[:, 2])
    inc = v[np.linalg.norm(cl - q, axis=1) <= on_tol]
    if not len(inc):
        return math.nan
    normals = np.cross(inc[:, 1] - inc[:, 0], inc[:, 2] - inc[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals *= np.sign(normals[:, 2:3])
    n = np.asarray(n, np.float64) / np.linalg.norm(n)
    lam = _nnls(normals.T, n)
    proj = normals.T @ lam
    norm = float(np.linalg.norm(proj))
    if norm < 1.0e-12:
        return 90.0
    return float(np.degrees(np.arccos(np.clip(n @ proj / norm, -1.0, 1.0))))

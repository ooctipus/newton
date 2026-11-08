# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures of the terrain law tests: terrains, the float32 predicate kernels and the float64 oracle's view.

The float32 kernels here evaluate only the shared predicates of
:mod:`newton._src.geometry.terrain_law` (window frame, canonical edge values, closed predicates, cluster planes,
edge classes, vertex features and the per-pair finish); they are the point-core part of the law, enough to check
that the predicates claim every support the float64 oracle (:mod:`newton.tests.terrain_law_reference`) finds.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import warp as wp

import newton
from newton._src.geometry.terrain_contacts import TerrainFeatures
from newton._src.geometry.terrain_law import (
    TERRAIN_CELL_WIDEN,
    TERRAIN_EDGE_BORDER,
    TERRAIN_EDGE_CONCAVE,
    TERRAIN_EDGE_CORNER,
    TERRAIN_EDGE_FLAT,
    TERRAIN_EPS32,
    TERRAIN_EPS_FACTOR,
    TERRAIN_EPS_MIN_EXTENT,
    TERRAIN_FACE_WALL,
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_VERTEX,
    TerrainFeatureData,
    TerrainFinishScratch,
    terrain_border_point,
    terrain_border_side,
    terrain_cluster,
    terrain_corner_point,
    terrain_edge_class,
    terrain_edge_point,
    terrain_eps,
    terrain_face_point,
    terrain_feature_id,
    terrain_finish_list,
    terrain_finish_scratch,
    terrain_interior_vertex,
    terrain_point_below,
    terrain_vertex_feature,
    terrain_vertex_in_cone,
    terrain_vertex_point,
    terrain_window,
    terrain_window_point,
    terrain_window_to_local,
    terrain_window_vertex,
)
from newton.tests import terrain_law_reference as ref

CLAIMS = 16
"""Face claims recorded per point."""


# ----------------------------------------------------------------------------------------------------------------
# terrains
# ----------------------------------------------------------------------------------------------------------------


def heightfield_model(heights, hx, hy, *, device, min_z=None, max_z=None, extra=()):
    """A model holding heightfields (each ``(heights, hx, hy[, xform])``) and nothing else."""
    builder = newton.ModelBuilder()
    for item in ((heights, hx, hy), *extra):
        data, sx, sy = item[:3]
        data = np.asarray(data, np.float32)
        terrain = newton.Heightfield(
            data=data, nrow=data.shape[0], ncol=data.shape[1], hx=sx, hy=sy, min_z=min_z, max_z=max_z
        )
        builder.add_shape_heightfield(heightfield=terrain, xform=item[3] if len(item) > 3 else None)
    return builder.finalize(device=device)


def reference_surface(model, *, heightfield: int = 0) -> ref.ReferenceHeightfield:
    """The float64 oracle's surface of heightfield index ``heightfield`` of a model (narrow-phase decode)."""
    row = model.heightfield_data.numpy()[heightfield]
    nrow, ncol, offset = int(row["nrow"]), int(row["ncol"]), int(row["data_offset"])
    e = model.heightfield_elevations.numpy()[offset : offset + nrow * ncol]
    Z = ref.decode_heights(e, nrow, ncol, float(row["min_z"]), float(row["max_z"]))
    return ref.ReferenceHeightfield(Z, float(row["hx"]), float(row["hy"]))


def suite_like_heights(rng, rows: int = 64, cols: int = 64, *, tile: int = 32):
    """A mosaic of the suite's sub-terrains on 0.1 m cells: noisy flat, quantized rough, stairs, slope, boxes.

    Returns float64 heights ``(rows + 1, cols + 1)`` [m]. The flat tile carries +-1.4 um of noise, as the recorded
    suite terrain's flat border does; the rough tile is uniform noise in 5 mm steps.
    """
    nrow, ncol = rows + 1, cols + 1
    r, c = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
    Z = np.zeros((nrow, ncol))
    kind = ((r // tile) * 7 + (c // tile) * 3) % 5
    noise = rng.integers(-6, 7, size=(nrow, ncol)) * 2.4e-7
    rough = np.round(rng.uniform(0.0, 0.08, size=(nrow, ncol)) / 0.005) * 0.005
    stairs = np.floor((c % tile) / 4) * 0.05
    slope = np.tan(np.radians(12.0)) * (0.06 * (c % tile) - 0.08 * (r % tile)) + 0.3
    boxes = np.where(((r // 6) + (c // 6)) % 3 == 0, 0.12, 0.0)
    Z = np.where(kind == 0, noise, Z)
    Z = np.where(kind == 1, rough, Z)
    Z = np.where(kind == 2, stairs, Z)
    Z = np.where(kind == 3, slope, Z)
    Z = np.where(kind == 4, boxes + noise, Z)
    return Z


def border_terrains(rng) -> dict:
    """2 m tiles of 0.1 m cells for the heightfield's border, heights ``(21, 21)`` [m] by name: flat, rough (uniform
    noise in 5 mm steps up to 8 cm) and a slope (0.2 along y), so that the walls stand under a level, a jagged and a
    tilted rim."""
    return {
        "flat": np.zeros((21, 21)),
        "rough": np.round(rng.uniform(0.0, 0.08, size=(21, 21)) / 0.005) * 0.005,
        "slope": np.broadcast_to(0.2 * np.linspace(-1.0, 1.0, 21), (21, 21)).T + 0.1,
    }


def border_bodies(rng, hf: ref.ReferenceHeightfield, kinds, count: int) -> list:
    """Bodies ``(kind, dims, pose)`` at a heightfield's border, ``count`` of each ``(kind, dims)`` of ``kinds``: the
    pose's point within 8 cm of a wall's plane (a sixth of them at each wall) or of both walls at a corner (a third), in
    any orientation, the lowest point (a sphere's or capsule's rounded) 3 cm into to 12 mm above the highest terrain
    under the body's bounds."""
    bodies = []
    for i in range(count * len(kinds)):
        kind, dims = kinds[i % len(kinds)]
        x = rng.uniform(-hf.hx + 0.1, hf.hx - 0.1)
        y = rng.uniform(-hf.hy + 0.1, hf.hy - 0.1)
        side = int(rng.integers(0, 6))
        if side < 2:
            x = (1 - 2 * side) * hf.hx + rng.uniform(-0.08, 0.08)
        elif side < 4:
            y = (5 - 2 * side) * hf.hy + rng.uniform(-0.08, 0.08)
        else:
            x = rng.choice([-1.0, 1.0]) * (hf.hx + rng.uniform(-0.08, 0.08))
            y = rng.choice([-1.0, 1.0]) * (hf.hy + rng.uniform(-0.08, 0.08))
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        lo, hi = ref.make_core(kind, dims, np.r_[x, y, 0.0, q]).aabb()
        xs = np.clip(np.linspace(lo[0], hi[0], 5), -hf.hx, hf.hx)
        ys = np.clip(np.linspace(lo[1], hi[1], 5), -hf.hy, hf.hy)
        top = max(hf.height(xx, yy) for xx in xs for yy in ys)
        radius = dims["radius"] if kind in ("sphere", "capsule") else 0.0
        z = top - lo[2] + radius + rng.uniform(-0.03, 0.012)
        bodies.append((kind, dims, np.array([x, y, z, *q], np.float32).astype(np.float64)))
    return bodies


def outside_face_rows(hf: ref.ReferenceHeightfield, rows, *, tol: float = 1.0e-6) -> list:
    """The float32 face rows (faces, clipped faces and walls) deeper than ``tol`` whose core point ``q + d n`` lies
    outside the solid: above the surface over the footprint's closure (the point taken into it within ``tol``) or
    beyond it. A face's supported point is front-facing: below its plane, it lies inside the block."""
    out = []
    for row in rows:
        q, n, d = row_qnd(row)
        if d >= -tol or (int(row[3]) >> 5) & 3 != int(TERRAIN_FEATURE_FACE):
            continue
        k = q + d * n
        inside = abs(k[0]) <= hf.hx + tol and abs(k[1]) <= hf.hy + tol
        height = hf.height(float(np.clip(k[0], -hf.hx, hf.hx)), float(np.clip(k[1], -hf.hy, hf.hy)))
        if not inside or k[2] > height + tol:
            out.append(row)
    return out


# ----------------------------------------------------------------------------------------------------------------
# float32 predicate kernels
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _window_cells(data: TerrainFeatureData, heightfield: int, p: wp.vec3, reach: float) -> wp.vec4i:
    surface = data.surfaces[heightfield]
    cols = surface.ncol - 1
    rows = surface.nrow - 1
    c0 = wp.clamp(int(wp.floor((p[0] - reach - surface.x0) / surface.dx)), 0, cols - 1)
    c1 = wp.clamp(int(wp.floor((p[0] + reach - surface.x0) / surface.dx)), 0, cols - 1)
    r0 = wp.clamp(int(wp.floor((p[1] - reach - surface.y0) / surface.dy)), 0, rows - 1)
    r1 = wp.clamp(int(wp.floor((p[1] + reach - surface.y0) / surface.dy)), 0, rows - 1)
    return wp.vec4i(r0, r1, c0, c1)


@wp.func
def _push(scratch: TerrainFinishScratch, base: int, count: int, q: wp.vec3, k: wp.vec3, n: wp.vec3, d: float, f: int):
    """Append a candidate to a point's list in the finish scratch (raw: the finish merges coincident ones)."""
    if count < scratch.capacity:
        scratch.q[base + count] = q
        scratch.k[base + count] = k
        scratch.n[base + count] = n
        scratch.d[base + count] = d
        scratch.feature[base + count] = f
    return count + 1


@wp.kernel(enable_backward=False)
def _point_rows_kernel(
    data: TerrainFeatureData,
    heightfield: int,
    points: wp.array[wp.vec3],
    reach: wp.array[float],
    scratch: TerrainFinishScratch,
    out_count: wp.array[int],
    out_q: wp.array2d[wp.vec3],
    out_n: wp.array2d[wp.vec3],
    out_d: wp.array2d[float],
    out_feature: wp.array2d[int],
):
    """The point-core rows of the law from the float32 predicates (faces, crease edges, vertices and at the border the
    border edges, walls and corner lines; no G2/G3), every candidate into the point's list of the finish scratch, then
    the law's finish."""
    i = wp.tid()
    base = i * scratch.capacity
    p = points[i]
    D = reach[i]
    box = _window_cells(data, heightfield, p, D)
    r0 = box[0]
    r1 = box[1]
    c0 = box[2]
    c1 = box[3]
    window = terrain_window(data, heightfield, r0, c0)
    k = terrain_window_point(window, p)
    extent = wp.max(wp.max(wp.abs(k[0]), wp.abs(k[1])), wp.abs(k[2]))
    extent = wp.max(extent, wp.max(float(c1 - c0 + 2) * window.surface.dx, float(r1 - r0 + 2) * window.surface.dy))
    for r in range(r0, r1 + 2):
        for c in range(c0, c1 + 2):
            extent = wp.max(extent, wp.abs(terrain_window_vertex(data, window, r, c)[2]))
    window.eps = terrain_eps(extent)
    eps = window.eps
    below = terrain_point_below(data, window, k)
    count = int(0)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            for sub in range(2):
                claims, q, n, d = terrain_face_point(data, window, r, c, sub, k)
                if claims and d <= D and not ((d > eps and below) or (d < -eps and not below)):
                    count = _push(
                        scratch, base, count, q, k, n, d, terrain_feature_id(r, c, TERRAIN_FEATURE_FACE, sub, 0)
                    )
    for r in range(r0, r1 + 2):
        for c in range(c0, c1 + 2):
            for which in range(3):
                if which == 0 and c > c1:
                    continue
                if which == 1 and r > r1:
                    continue
                if which == 2 and (r > r1 or c > c1):
                    continue
                kind = terrain_edge_class(data, window, r, c, which)
                if kind == TERRAIN_EDGE_BORDER:
                    side = terrain_border_side(window.surface, r, c, which)
                    if side >= 0 and not below:
                        claim, q, n, d = terrain_border_point(data, window, r, c, which, side, k)
                        if claim != 0 and d <= D:
                            f = terrain_feature_id(r, c, TERRAIN_FEATURE_EDGE, which, 0)
                            if claim == 2:
                                f = terrain_feature_id(r, c, TERRAIN_FEATURE_FACE, TERRAIN_FACE_WALL + which, 0)
                            count = _push(scratch, base, count, q, k, n, d, f)
                    continue
                if kind == TERRAIN_EDGE_FLAT:
                    continue
                if kind == TERRAIN_EDGE_CONCAVE and not below:
                    continue
                claims, q = terrain_edge_point(data, window, r, c, which, k)
                dist = wp.length(k - q)
                if claims and dist > 1.0e-9:
                    n = (k - q) / dist
                    d = dist
                    if below:
                        n = -n
                        d = -dist
                    if d <= D:
                        count = _push(
                            scratch, base, count, q, k, n, d, terrain_feature_id(r, c, TERRAIN_FEATURE_EDGE, which, 0)
                        )
            if not terrain_interior_vertex(window.surface, r, c):
                v = terrain_window_vertex(data, window, r, c)
                if below:
                    continue
                if v[2] - k[2] > 0.0:
                    ok, q, n, d = terrain_corner_point(data, window, r, c, k)
                    if ok and d <= D:
                        f = terrain_feature_id(r, c, TERRAIN_FEATURE_EDGE, TERRAIN_EDGE_CORNER, 0)
                        count = _push(scratch, base, count, q, k, n, d, f)
                    continue
            if terrain_vertex_point(data, window, r, c, k):
                v = terrain_window_vertex(data, window, r, c)
                dist = wp.length(k - v)
                if dist > 1.0e-9:
                    n = (k - v) / dist
                    d = dist
                    if below:
                        n = -n
                        d = -dist
                    if d <= D:
                        count = _push(
                            scratch, base, count, v, k, n, d, terrain_feature_id(r, c, TERRAIN_FEATURE_VERTEX, 0, 0)
                        )
    count = wp.min(count, scratch.capacity)
    kept = terrain_finish_list(scratch, base, count, eps, 0, 1)
    out_count[i] = kept
    j = int(0)
    for row in range(count):
        if scratch.mark[base + row] != 0:
            out_q[i, j] = terrain_window_to_local(window, scratch.q[base + row])
            out_n[i, j] = scratch.n[base + row]
            out_d[i, j] = scratch.d[base + row]
            out_feature[i, j] = scratch.feature[base + row]
            j += 1


@wp.kernel(enable_backward=False)
def _face_claims_kernel(
    data: TerrainFeatureData,
    heightfield: int,
    points: wp.array[wp.vec3],
    out_count: wp.array[int],
    out_cluster: wp.array2d[int],
    out_q: wp.array2d[wp.vec3],
    out_n: wp.array2d[wp.vec3],
    out_d: wp.array2d[float],
):
    """Every triangle near point ``k`` that claims ``k``'s projection on its cluster plane (closed predicates)."""
    i = wp.tid()
    p = points[i]
    box = _window_cells(data, heightfield, p, 0.15)
    r0 = box[0]
    r1 = box[1]
    c0 = box[2]
    c1 = box[3]
    window = terrain_window(data, heightfield, r0, c0)
    k = terrain_window_point(window, p)
    extent = wp.max(wp.max(wp.abs(k[0]), wp.abs(k[1])), wp.abs(k[2]))
    extent = wp.max(extent, wp.max(float(c1 - c0 + 2) * window.surface.dx, float(r1 - r0 + 2) * window.surface.dy))
    for r in range(r0, r1 + 2):
        for c in range(c0, c1 + 2):
            extent = wp.max(extent, wp.abs(terrain_window_vertex(data, window, r, c)[2]))
    window.eps = terrain_eps(extent)
    count = int(0)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            for sub in range(2):
                claims, q, n, d = terrain_face_point(data, window, r, c, sub, k)
                if claims and count < 16:
                    out_cluster[i, count] = terrain_cluster(data, window.surface, r, c, sub)
                    out_q[i, count] = terrain_window_to_local(window, q)
                    out_n[i, count] = n
                    out_d[i, count] = d
                    count += 1
    out_count[i] = count


@wp.kernel(enable_backward=False)
def _feature_queries_kernel(
    data: TerrainFeatureData,
    heightfield: int,
    queries: wp.array[wp.vec4i],
    directions: wp.array[wp.vec3],
    out: wp.array[wp.vec3i],
):
    """Per query (r, c, which, -): the edge class, whether the vertex is a feature and whether ``n`` is in its cone."""
    i = wp.tid()
    query = queries[i]
    window = terrain_window(data, heightfield, query[0], query[1])
    window.eps = terrain_eps(1.0)
    kind = terrain_edge_class(data, window, query[0], query[1], query[2])
    feature = int(terrain_vertex_feature(data, window, query[0], query[1]))
    cone = int(terrain_vertex_in_cone(data, window, query[0], query[1], directions[i]))
    out[i] = wp.vec3i(kind, feature, cone)


def point_rows(features: TerrainFeatures, heightfield: int, points, reach):
    """Float32 point-core rows per point: lists of ``(q, n, d, feature)`` in heightfield-local coordinates."""
    device = features.device
    points = np.asarray(points, np.float32).reshape(-1, 3)
    n = len(points)
    reach = np.broadcast_to(np.asarray(reach, np.float32), (n,)).copy()
    # every feature of the window once: six per vertex of a window of at most 2 D / cell + 2 cells a side
    surface = features.data().surfaces.numpy()[heightfield]
    cells = int(np.floor(2.0 * float(reach.max()) / min(float(surface["dx"]), float(surface["dy"])))) + 2
    candidates = 6 * (cells + 1) ** 2
    scratch = terrain_finish_scratch(device, candidates, blocks=n)
    out_count = wp.zeros(n, dtype=int, device=device)
    out_q = wp.zeros((n, candidates), dtype=wp.vec3, device=device)
    out_n = wp.zeros((n, candidates), dtype=wp.vec3, device=device)
    out_d = wp.zeros((n, candidates), dtype=float, device=device)
    out_f = wp.zeros((n, candidates), dtype=int, device=device)
    wp.launch(
        _point_rows_kernel,
        dim=n,
        inputs=[
            features.data(),
            heightfield,
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(reach, dtype=float, device=device),
            scratch,
        ],
        outputs=[out_count, out_q, out_n, out_d, out_f],
        device=device,
    )
    counts, q, nn, d, f = out_count.numpy(), out_q.numpy(), out_n.numpy(), out_d.numpy(), out_f.numpy()
    return [
        [
            (q[i, j].astype(np.float64), nn[i, j].astype(np.float64), float(d[i, j]), int(f[i, j]))
            for j in range(counts[i])
        ]
        for i in range(n)
    ]


def face_claims(features: TerrainFeatures, heightfield: int, points):
    """Float32 face claims per point: lists of ``(cluster id, q, n, d)``."""
    device = features.device
    points = np.asarray(points, np.float32).reshape(-1, 3)
    n = len(points)
    out_count = wp.zeros(n, dtype=int, device=device)
    out_cluster = wp.zeros((n, CLAIMS), dtype=int, device=device)
    out_q = wp.zeros((n, CLAIMS), dtype=wp.vec3, device=device)
    out_n = wp.zeros((n, CLAIMS), dtype=wp.vec3, device=device)
    out_d = wp.zeros((n, CLAIMS), dtype=float, device=device)
    wp.launch(
        _face_claims_kernel,
        dim=n,
        inputs=[features.data(), heightfield, wp.array(points, dtype=wp.vec3, device=device)],
        outputs=[out_count, out_cluster, out_q, out_n, out_d],
        device=device,
    )
    counts, cl, q, nn, d = out_count.numpy(), out_cluster.numpy(), out_q.numpy(), out_n.numpy(), out_d.numpy()
    return [
        [
            (int(cl[i, j]), q[i, j].astype(np.float64), nn[i, j].astype(np.float64), float(d[i, j]))
            for j in range(counts[i])
        ]
        for i in range(n)
    ]


def feature_queries(features: TerrainFeatures, heightfield: int, queries, *, directions=None):
    """Per ``(r, c, which)``: (edge class, vertex feature 0/1, direction in the vertex's cone 0/1)."""
    device = features.device
    queries = np.asarray(queries, np.int32).reshape(-1, 3)
    n = len(queries)
    q4 = np.zeros((n, 4), np.int32)
    q4[:, :3] = queries
    if directions is None:
        directions = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (n, 1))
    out = wp.zeros(n, dtype=wp.vec3i, device=device)
    wp.launch(
        _feature_queries_kernel,
        dim=n,
        inputs=[
            features.data(),
            heightfield,
            wp.array(q4, dtype=wp.vec4i, device=device),
            wp.array(np.asarray(directions, np.float32), dtype=wp.vec3, device=device),
        ],
        outputs=[out],
        device=device,
    )
    return out.numpy()


def angle_deg(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


# ----------------------------------------------------------------------------------------------------------------
# scenes and the rows a collision pipeline produces: today's contacts and the law's shadow rows
# ----------------------------------------------------------------------------------------------------------------

SCENE_MARGIN = 0.01
"""Margin of every shape of :func:`body_scene` [m] (the suite's)."""
SCENE_GAP = 0.01
"""Contact gap of every shape of :func:`body_scene` [m] (the suite's)."""


def body_scene(heights, hx, hy, bodies, *, device, margin=SCENE_MARGIN, gap=SCENE_GAP):
    """A heightfield at the origin and bodies, each alone in a world over it, as the suite's robots are.

    Args:
        heights: Heights ``(nrow, ncol)`` [m].
        hx: Half-extent along x [m].
        hy: Half-extent along y [m].
        bodies: ``(kind, dims, pose)`` per body: ``"sphere"`` (``radius``) or ``"capsule"`` (``radius``,
            ``half_height``), pose ``(px, py, pz, qx, qy, qz, qw)``.
        device: Model device.
        margin: Margin of every shape [m].
        gap: Contact gap of every shape [m].

    Returns:
        ``(model, heightfield shape, partner shapes)``.
    """
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=margin, gap=gap)
    data = np.asarray(heights, np.float32)
    terrain = newton.Heightfield(data=data, nrow=data.shape[0], ncol=data.shape[1], hx=hx, hy=hy)
    heightfield = builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
    shapes = []
    for kind, dims, pose in bodies:
        builder.begin_world()
        p = np.asarray(pose, np.float64)
        body = builder.add_body(xform=wp.transform(wp.vec3(*p[:3]), wp.quat(*p[3:7])))
        if kind == "sphere":
            shapes.append(builder.add_shape_sphere(body, radius=dims["radius"], cfg=cfg))
        elif kind == "capsule":
            shapes.append(
                builder.add_shape_capsule(body, radius=dims["radius"], half_height=dims["half_height"], cfg=cfg)
            )
        else:
            raise ValueError(kind)
        builder.end_world()
    return builder.finalize(device=device), heightfield, shapes


def shadow_pipeline(model, *, classes: str = "point", **kwargs):
    """A collision pipeline built with ``NEWTON_TERRAIN_LAW_SHADOW=classes`` (keyword arguments pass through)."""
    import os  # noqa: PLC0415
    from unittest.mock import patch  # noqa: PLC0415

    environ = {"NEWTON_TERRAIN_LAW_SHADOW": classes}
    with patch.dict(os.environ, environ):
        return newton.CollisionPipeline(model, **kwargs)


def terrain_route_pipeline(model, **kwargs):
    """A collision pipeline whose terrain route writes every heightfield contact, today's route without the law
    (``NEWTON_TERRAIN_LAW=none``, the law's temporary opt-out, and no shadow), for comparisons with the law (keyword
    arguments pass through)."""
    import os  # noqa: PLC0415
    from unittest.mock import patch  # noqa: PLC0415

    environ = {k: v for k, v in os.environ.items() if k != "NEWTON_TERRAIN_LAW_SHADOW"}
    environ["NEWTON_TERRAIN_LAW"] = "none"
    with patch.dict(os.environ, environ, clear=True):
        return newton.CollisionPipeline(model, **kwargs)


def shadow_rows(pipeline) -> dict[int, list]:
    """The law's shadow rows of the last collide per partner shape: ``[(q, n, d, feature)]``, world frame, float64."""
    rows = pipeline._terrain_law_shadow.rows()
    out = {int(p[1]): [] for p in rows["pairs"]}
    for j, i in enumerate(rows["row_pair"]):
        out[int(rows["pairs"][i][1])].append(
            (
                rows["q"][j].astype(np.float64),
                rows["n"][j].astype(np.float64),
                float(rows["d"][j]),
                int(rows["feature"][j]),
            )
        )
    return out


def contact_rows(model, state, contacts, heightfield: int) -> dict[int, list]:
    """Today's contacts of a heightfield per partner shape: ``[(q, n, d)]``, world frame, float64.

    ``q`` is the contact's terrain point, ``n`` its normal from the terrain to the partner and ``d = n . (k - q)``
    with ``k`` its partner point. Today's narrow phase puts that point within about 0.1 mm of the partner's core (a
    sphere's centre, a point of a capsule's axis), so ``d`` approximates the law's core distance to that much only:
    compare today's rows with the law's on ``q`` and ``n``.
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    point0 = contacts.rigid_contact_point0.numpy()[:count].astype(np.float64)
    point1 = contacts.rigid_contact_point1.numpy()[:count].astype(np.float64)
    normal = contacts.rigid_contact_normal.numpy()[:count].astype(np.float64)
    body_q = state.body_q.numpy().astype(np.float64)
    shape_body = model.shape_body.numpy()

    def world(shape, p):
        body = shape_body[shape]
        if body < 0:
            return p
        return body_q[body, :3] + ref.quat_rotate(body_q[body, 3:7], p)

    out = {}
    for i in range(count):
        a, b = int(shape0[i]), int(shape1[i])
        if a == heightfield:
            partner, q, k, n = b, world(a, point0[i]), world(b, point1[i]), normal[i]
        elif b == heightfield:
            partner, q, k, n = a, world(b, point1[i]), world(a, point0[i]), -normal[i]
        else:
            continue
        out.setdefault(partner, []).append((q, n, float(n @ (k - q))))
    return out


def angle_between(a, b) -> float:
    """Angle between two directions [rad]."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))


def row_qnd(row) -> tuple:
    """``(q, n, d)`` of a reference row or of a ``(q, n, d, ...)`` tuple."""
    return (row.q, row.n, row.d) if hasattr(row, "q") else tuple(row[:3])


def missing_rows(expected, got, *, q_tol: float, angle_tol: float, d_tol: float | None = None) -> list:
    """The rows of ``expected`` that no row of ``got`` matches (reference rows or ``(q, n, d, ...)`` tuples)."""
    candidates = [row_qnd(g) for g in got]
    missing = []
    for row in expected:
        q, n, d = row_qnd(row)
        if not any(
            np.linalg.norm(gq - q) <= q_tol
            and angle_between(gn, n) <= angle_tol
            and (d_tol is None or abs(gd - d) <= d_tol)
            for gq, gn, gd in candidates
        ):
            missing.append(row)
    return missing


# ----------------------------------------------------------------------------------------------------------------
# float32 against float64: decisions within the predicates' band
# ----------------------------------------------------------------------------------------------------------------

DECISION_BAND = 2.0e-6
"""Margin [m] within which a float32 boundary decision may differ from the float64 one (two bands of a 1 m window)."""


def core_ends(core) -> list:
    """A reference core's points: a sphere's centre, a capsule's two ends."""
    return [core.a, core.b] if hasattr(core, "a") else [core.c]


def window_cells(hf: ref.ReferenceHeightfield, core, reach: float) -> tuple[int, int, int, int]:
    """The cells ``(r0, r1, c0, c1)`` of a point-core pair's window as ``terrain_box_cells`` takes them: the core's
    ends widened by ``reach``, cell coordinates widened by ``TERRAIN_CELL_WIDEN``."""
    ends = core_ends(core)
    lo = np.min(ends, axis=0) - reach
    hi = np.max(ends, axis=0) + reach
    c0 = max(int(np.floor((lo[0] + hf.hx) / hf.dx - TERRAIN_CELL_WIDEN)), 0)
    c1 = min(int(np.floor((hi[0] + hf.hx) / hf.dx + TERRAIN_CELL_WIDEN)), hf.cols - 1)
    r0 = max(int(np.floor((lo[1] + hf.hy) / hf.dy - TERRAIN_CELL_WIDEN)), 0)
    r1 = min(int(np.floor((hi[1] + hf.hy) / hf.dy + TERRAIN_CELL_WIDEN)), hf.rows - 1)
    return r0, r1, c0, c1


def kernel_band(hf: ref.ReferenceHeightfield, core, reach: float) -> float:
    """The closed-predicate band [m] the float32 kernels use for a pair's window (heightfield-local core).

    The window of :func:`window_cells` and the extent of ``terrain_window_extent``: the core's ends in the window
    frame, the window's size and its heights about its reference vertex. Four of these bands are the float32 finish's
    plane-group offset, which a float64 comparison passes to :func:`terrain_law_reference.finish` (``group_offset``).
    """
    ends = core_ends(core)
    r0, r1, c0, c1 = window_cells(hf, core, reach)
    origin = hf.V(r0, c0)
    extent = max(
        (c1 + 1 - c0) * hf.dx,
        (r1 + 1 - r0) * hf.dy,
        float(np.abs(hf.Z[r0 : r1 + 2, c0 : c1 + 2] - origin[2]).max()),
        *(float(np.abs(np.asarray(p) - origin).max()) for p in ends),
    )
    return TERRAIN_EPS_FACTOR * TERRAIN_EPS32 * max(extent, TERRAIN_EPS_MIN_EXTENT)


def _seam_edges(hf: ref.ReferenceHeightfield, q, tol: float) -> list:
    """Grid edges ``(kind, r, c)`` (``"H"``, ``"V"``, ``"D"`` as the reference names them) whose segment passes within
    ``tol`` of ``q``'s footprint."""
    u = (q[0] + hf.hx) / hf.dx
    v = (q[1] + hf.hy) / hf.dy
    out = []
    c, r = int(np.floor(u)), int(np.floor(v))
    for rr in (r - 1, r, r + 1):
        for cc in (c - 1, c, c + 1):
            # H edge (rr, cc)-(rr, cc + 1), V edge (rr, cc)-(rr + 1, cc), D edge (rr, cc)-(rr + 1, cc + 1)
            for kind, (dr, dc) in (("H", (0, 1)), ("V", (1, 0)), ("D", (1, 1))):
                if not (hf.valid_vertex(rr, cc) and hf.valid_vertex(rr + dr, cc + dc)):
                    continue
                a = np.array([cc * hf.dx, rr * hf.dy])
                b = np.array([(cc + dc) * hf.dx, (rr + dr) * hf.dy])
                p = np.array([u * hf.dx, v * hf.dy])
                t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
                if np.linalg.norm(a + t * (b - a) - p) <= tol:
                    out.append((kind, rr, cc))
    return out


def _edge_triangles(hf: ref.ReferenceHeightfield, kind: str, r: int, c: int):
    """The two triangles of a reference edge (None outside the grid), as ``window_features`` lists them."""
    if kind == "H":
        return (hf.tri_id(r - 1, c, 1) if hf.valid_cell(r - 1, c) else None), (
            hf.tri_id(r, c, 0) if hf.valid_cell(r, c) else None
        )
    if kind == "V":
        return (hf.tri_id(r, c - 1, 0) if hf.valid_cell(r, c - 1) else None), (
            hf.tri_id(r, c, 1) if hf.valid_cell(r, c) else None
        )
    return hf.tri_id(r, c, 0), hf.tri_id(r, c, 1)


def _face_values(hf: ref.ReferenceHeightfield, t: int, k) -> list:
    """Float64 face values ``s = (k - a) . m`` of triangle ``t``'s three edges for point ``k``."""
    vs = hf.tri_verts(t)
    n, _ = hf.plane(t)
    out = []
    for i in range(3):
        a, b, w = vs[i], vs[(i + 1) % 3], vs[(i + 2) % 3]
        m = np.cross(b - a, n)
        if m @ (w - a) > 0.0:
            m = -m
        out.append(float((k - a) @ (m / np.linalg.norm(m))))
    return out


def band_decided(hf: ref.ReferenceHeightfield, q, k, *, band: float = DECISION_BAND) -> bool:
    """Whether a row (terrain point ``q``, core point ``k``, heightfield-local) was decided within ``band``.

    True when the row's terrain point lies on a seam and a float64 predicate deciding a feature there is within
    ``band`` of its threshold: a face's value at that edge (a face claim, or an edge claim beyond its faces), an
    edge's class (a crease within ``band`` of flat) or the vertex's Voronoi test. Rows decided within the band
    may be claimed in float32 and not in float64, or the reverse; any other difference is a defect.
    """
    q = np.asarray(q, np.float64)
    k = np.asarray(k, np.float64)
    for kind, r, c in _seam_edges(hf, q, band):
        ta, tb = _edge_triangles(hf, kind, r, c)
        for t in (ta, tb):
            if t is not None and min(abs(s) for s in _face_values(hf, t, k)) <= band:
                return True
        if ta is not None and tb is not None:
            n, a = hf.plane(ta)
            vb = hf.tri_verts(tb)
            far = [v for v in vb if all(np.linalg.norm(v - w) > 1e-12 for w in hf.tri_verts(ta))]
            if far and abs(float(n @ (far[0] - a))) <= band:
                return True
    # a vertex's Voronoi boundary: the test's largest value within the band of zero
    u = (q[0] + hf.hx) / hf.dx
    v = (q[1] + hf.hy) / hf.dy
    c, r = int(np.rint(u)), int(np.rint(v))
    if abs(u - c) * hf.dx <= band and abs(v - r) * hf.dy <= band and hf.ring(r, c) is not None:
        vertex = hf.V(r, c)
        values = [float((k - vertex) @ (hf.V(*w) - vertex) / np.linalg.norm(hf.V(*w) - vertex)) for w in hf.ring(r, c)]
        if abs(max(values)) <= band:
            return True
    return False


def carries_half_space(row, got, *, angle_tol: float = 1.0e-3, tol: float = 1.0e-5) -> bool:
    """Whether a row of ``got`` carries ``row``'s constraint: normal within ``angle_tol``, terrain point on its plane
    (``|n . dq|``) and core distance within ``tol``."""
    q, n, d = row_qnd(row)
    return any(
        angle_between(gn, n) <= angle_tol and abs(float(n @ (gq - q))) <= tol and abs(gd - d) <= tol
        for gq, gn, gd in (row_qnd(g) for g in got)
    )


class RowComparison(NamedTuple):
    """The float32 rows of one pair against its float64 rows (:func:`compare_rows`)."""

    missing: list
    """Float64 rows no float32 row matches or carries (defects)."""
    extra: list
    """Float32 rows no float64 row matches, not decided within the band (defects)."""
    covered: int
    """Unmatched rows whose constraint a row of the other side carries (:func:`carries_half_space`): the same plane and
    core distance with the terrain point elsewhere on it, as an edge's closest point on a line nearly parallel to the
    axis lands."""
    tolerated: int
    """Float32 rows no float64 row matches or carries, decided within the band (:func:`band_decided`)."""
    deepest: float
    """Difference of the deepest core distances [m] (``inf`` when only float64 has rows)."""


def compare_rows(
    hf: ref.ReferenceHeightfield, expected, got, *, frame_tol: float = 1.0e-5, band: float = DECISION_BAND
) -> RowComparison:
    """Compare float32 rows with float64 rows of one pair (heightfield-local frame).

    Rows match with terrain points within ``frame_tol``, normals within 1e-3 rad and core distances within 10 um. An
    unmatched row is a defect unless a row of the other side carries its constraint, or (a float32 row) it was decided
    within the band. A pair with float64 rows and no float32 row misses them all.
    """
    missing = missing_rows(expected, got, q_tol=frame_tol, angle_tol=1.0e-3, d_tol=1.0e-5)
    extra = missing_rows(got, expected, q_tol=frame_tol, angle_tol=1.0e-3, d_tol=1.0e-5)
    defects_missing = [row for row in missing if not carries_half_space(row, got)]
    defects_extra = []
    covered = len(missing) - len(defects_missing)
    tolerated = 0
    for row in extra:
        q, n, d = row_qnd(row)
        if carries_half_space(row, expected):
            covered += 1
        elif band_decided(hf, q, q + d * n, band=band):
            tolerated += 1
        else:
            defects_extra.append(row)
    deepest = 0.0
    if expected and got:
        deepest = abs(min(row_qnd(r)[2] for r in expected) - min(row_qnd(g)[2] for g in got))
    elif expected:
        deepest = float("inf")
    return RowComparison(defects_missing, defects_extra, covered, tolerated, deepest)

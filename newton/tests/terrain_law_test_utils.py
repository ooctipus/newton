# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures of the terrain law tests: terrains, the float32 predicate kernels and the float64 oracle's view.

The float32 kernels here evaluate only the shared predicates of
:mod:`newton._src.geometry.terrain_law` (window frame, canonical edge values, closed predicates, cluster planes,
edge classes, vertex features and the per-pair finish); they are the point-core part of the law, enough to check
that the predicates claim every support the float64 oracle (:mod:`newton.tests.terrain_law_reference`) finds.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton._src.geometry.terrain_contacts import TerrainFeatures
from newton._src.geometry.terrain_law import (
    TERRAIN_EDGE_BORDER,
    TERRAIN_EDGE_CONCAVE,
    TERRAIN_EDGE_FLAT,
    TERRAIN_FEATURE_EDGE,
    TERRAIN_FEATURE_FACE,
    TERRAIN_FEATURE_VERTEX,
    TerrainFeatureData,
    TerrainRows,
    terrain_cluster,
    terrain_edge_class,
    terrain_edge_point,
    terrain_eps,
    terrain_face_point,
    terrain_feature_id,
    terrain_finish,
    terrain_point_below,
    terrain_row_n,
    terrain_row_q,
    terrain_rows_push,
    terrain_vertex_feature,
    terrain_vertex_in_cone,
    terrain_vertex_point,
    terrain_window,
    terrain_window_point,
    terrain_window_to_local,
    terrain_window_vertex,
)
from newton.tests import terrain_law_reference as ref

POINT_ROWS = 8
"""Rows a point-core pair keeps (the law's limit for spheres and capsules)."""
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


# ----------------------------------------------------------------------------------------------------------------
# float32 predicate kernels
# ----------------------------------------------------------------------------------------------------------------


@wp.func
def _window_cells(data: TerrainFeatureData, heightfield: int, p: wp.vec3, reach: float) -> wp.vec4i:
    surface = data.surfaces[heightfield]
    cols = surface.ncol - 1
    rows = surface.nrow - 1
    c0 = wp.clamp(int(wp.floor((p[0] - reach + surface.hx) / surface.dx)), 0, cols - 1)
    c1 = wp.clamp(int(wp.floor((p[0] + reach + surface.hx) / surface.dx)), 0, cols - 1)
    r0 = wp.clamp(int(wp.floor((p[1] - reach + surface.hy) / surface.dy)), 0, rows - 1)
    r1 = wp.clamp(int(wp.floor((p[1] + reach + surface.hy) / surface.dy)), 0, rows - 1)
    return wp.vec4i(r0, r1, c0, c1)


@wp.kernel(enable_backward=False)
def _point_rows_kernel(
    data: TerrainFeatureData,
    heightfield: int,
    points: wp.array[wp.vec3],
    reach: wp.array[float],
    out_count: wp.array[int],
    out_q: wp.array2d[wp.vec3],
    out_n: wp.array2d[wp.vec3],
    out_d: wp.array2d[float],
    out_feature: wp.array2d[int],
):
    """The point-core rows of the law from the float32 predicates (faces, crease edges, vertices; no G2/G3)."""
    i = wp.tid()
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
    rows = TerrainRows()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            for sub in range(2):
                claims, q, n, d = terrain_face_point(data, window, r, c, sub, k)
                if claims and d <= D and not ((d > eps and below) or (d < -eps and not below)):
                    rows = terrain_rows_push(rows, q, k, n, d, terrain_feature_id(r, c, TERRAIN_FEATURE_FACE, sub, 0))
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
                if kind == TERRAIN_EDGE_FLAT or kind == TERRAIN_EDGE_BORDER:
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
                        rows = terrain_rows_push(
                            rows, q, k, n, d, terrain_feature_id(r, c, TERRAIN_FEATURE_EDGE, which, 0)
                        )
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
                        rows = terrain_rows_push(
                            rows, v, k, n, d, terrain_feature_id(r, c, TERRAIN_FEATURE_VERTEX, 0, 0)
                        )
    order, kept, _flags = terrain_finish(rows, eps, POINT_ROWS)
    out_count[i] = kept
    for j in range(kept):
        row = order[j]
        out_q[i, j] = terrain_window_to_local(window, terrain_row_q(rows, row))
        out_n[i, j] = terrain_row_n(rows, row)
        out_d[i, j] = rows.d[row]
        out_feature[i, j] = rows.feature[row]


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
    out_count = wp.zeros(n, dtype=int, device=device)
    out_q = wp.zeros((n, POINT_ROWS), dtype=wp.vec3, device=device)
    out_n = wp.zeros((n, POINT_ROWS), dtype=wp.vec3, device=device)
    out_d = wp.zeros((n, POINT_ROWS), dtype=float, device=device)
    out_f = wp.zeros((n, POINT_ROWS), dtype=int, device=device)
    wp.launch(
        _point_rows_kernel,
        dim=n,
        inputs=[
            features.data(),
            heightfield,
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(reach, dtype=float, device=device),
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

    with patch.dict(os.environ, {"NEWTON_TERRAIN_LAW_SHADOW": classes}):
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

    ``q`` is the contact's terrain point, ``n`` its normal from the terrain to the partner and ``d`` its core
    distance (the margins and the partner's rounding radius taken out), as the law's rows have them.
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    point0 = contacts.rigid_contact_point0.numpy()[:count].astype(np.float64)
    point1 = contacts.rigid_contact_point1.numpy()[:count].astype(np.float64)
    normal = contacts.rigid_contact_normal.numpy()[:count].astype(np.float64)
    margin0 = contacts.rigid_contact_margin0.numpy()[:count].astype(np.float64)
    margin1 = contacts.rigid_contact_margin1.numpy()[:count].astype(np.float64)
    body_q = state.body_q.numpy().astype(np.float64)
    shape_body = model.shape_body.numpy()
    shape_margin = model.shape_margin.numpy().astype(np.float64)

    def world(shape, p):
        body = shape_body[shape]
        if body < 0:
            return p
        return body_q[body, :3] + ref.quat_rotate(body_q[body, 3:7], p)

    out = {}
    for i in range(count):
        a, b = int(shape0[i]), int(shape1[i])
        if a == heightfield:
            partner, q, k, n, offset = b, world(a, point0[i]), world(b, point1[i]), normal[i], margin1[i]
        elif b == heightfield:
            partner, q, k, n, offset = a, world(b, point1[i]), world(a, point0[i]), -normal[i], margin0[i]
        else:
            continue
        radius = offset - shape_margin[partner]
        out.setdefault(partner, []).append((q, n, float(n @ (k - q)) - radius))
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

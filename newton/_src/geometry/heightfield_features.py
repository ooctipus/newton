# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Terrain surface normals for heightfield prism queries.

Each cell triangle is queried on its own, so its boundary edges and vertices
answer with the normal of a lone triangle: toward a partner beside the
triangle, an oblique edge normal, and under a tilted flat partner, the
partner's face normal. Where the terrain continues past that boundary, the
surface's normal there is a neighbouring face's: :func:`seam_surface_normal`
gives a separated witness the terrain's normal at its feature.
"""

import warp as wp

from ..utils.heightfield import HeightfieldData
from .support_function import GenericShapeData, GeoTypeEx

SEAM_DIRECTION_TOLERANCE = 1.0e-3
"""Direction cosine [-] by which a neighbouring terrain vertex must face the witness normal to retire the witness.

It exceeds the witness normals' rounding, so a witness whose nearer neighbour is not resolved keeps its contact.
"""


@wp.struct
class HeightfieldFeatureContext:
    heightfield: HeightfieldData
    elevations: wp.array[float]
    triangle: int


@wp.func
def _vertex(context: HeightfieldFeatureContext, row: int, col: int) -> wp.vec3:
    """Heightfield-local position of grid vertex ``(row, col)``, as the cell triangles decode it."""
    h = context.heightfield
    dx = 2.0 * h.hx / wp.float32(h.ncol - 1)
    dy = 2.0 * h.hy / wp.float32(h.nrow - 1)
    z = h.min_z + context.elevations[h.data_offset + row * h.ncol + col] * (h.max_z - h.min_z)
    return wp.vec3(-h.hx + wp.float32(col) * dx, -h.hy + wp.float32(row) * dy, z)


@wp.func
def _faces_witness(
    context: HeightfieldFeatureContext, row: int, col: int, origin: wp.vec3, point: wp.vec3, normal: wp.vec3
) -> bool:
    """Whether grid vertex ``(row, col)`` lies in front of the witness plane (absent vertices never do)."""
    h = context.heightfield
    if row < 0 or row >= h.nrow or col < 0 or col >= h.ncol:
        return False
    toward = _vertex(context, row, col) - origin - point
    return wp.dot(normal, toward) > SEAM_DIRECTION_TOLERANCE * wp.length(toward)


@wp.func
def _cell_normal(context: HeightfieldFeatureContext, row: int, col: int, sub: int) -> wp.vec3:
    """Upward unit normal of cell ``(row, col)``'s triangle ``sub``: (p00, p10, p11) or (p00, p11, p01)."""
    p00 = _vertex(context, row, col)
    p11 = _vertex(context, row + 1, col + 1)
    if sub == 0:
        return wp.normalize(wp.cross(_vertex(context, row, col + 1) - p00, p11 - p00))
    return wp.normalize(wp.cross(p11 - p00, _vertex(context, row + 1, col) - p00))


@wp.func
def seam_surface_normal(
    triangle: GenericShapeData,
    point: wp.vec3,
    normal: wp.vec3,
    distance: float,
    context: HeightfieldFeatureContext,
) -> tuple[wp.vec3, float]:
    """The terrain's normal at a separated witness, and the partner's distance along it.

    The point (on the triangle) and the normal are in the triangle-origin heightfield-local frame;
    the partner's witness point is ``point + distance * normal``. The terrain's normals at a
    boundary edge or vertex span the incident faces' normals, and a normal lies among them exactly
    when no terrain vertex adjacent to the feature lies in front of the witness plane (the
    neighbour's opposite vertex for an edge, the six incident grid vertices for a vertex). Such a
    witness, a face witness, a witness off the top face and non-heightfield triangles keep their
    normal. Otherwise the normal leaves the terrain toward a neighbour and becomes that
    neighbour's face normal (for a vertex, the incident face normal nearest to it); the distance
    is the partner point's along it.

    Returns:
        ``(normal, distance)``: unchanged for a terrain-surface normal.
    """
    h = context.heightfield
    if triangle.shape_type != int(GeoTypeEx.TRIANGLE_PRISM) or h.nrow < 2 or h.ncol < 2:
        return normal, distance
    if not (wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])):
        return normal, distance
    if not (wp.isfinite(normal[0]) and wp.isfinite(normal[1]) and wp.isfinite(normal[2])):
        return normal, distance
    e1, e2 = triangle.scale, triangle.auxiliary
    face = wp.cross(e1, e2)
    scale = wp.max(wp.length(e1), wp.length(e2))
    tolerance = 2.0e-6 * scale
    # Witnesses on the extruded walls or bottom lie off the top face; only its boundary has neighbours.
    if wp.abs(wp.dot(point, face)) > tolerance * wp.length(face):
        return normal, distance
    determinant = e1[0] * e2[1] - e1[1] * e2[0]
    if determinant <= 0.0:
        return normal, distance
    u = (point[0] * e2[1] - point[1] * e2[0]) / determinant
    v = (e1[0] * point[1] - e1[1] * point[0]) / determinant
    bary = wp.vec3(1.0 - u - v, u, v)
    epsilon = float(2.0e-6)
    if wp.min(bary) < -epsilon or wp.max(bary) > 1.0 + epsilon:
        return normal, distance
    zeros = int(0)
    largest = int(0)
    opposite = int(0)
    for k in range(3):
        if bary[k] <= epsilon:
            zeros += 1
            opposite = k
        if bary[k] > bary[largest]:
            largest = k
    if zeros == 0:
        return normal, distance
    cols = h.ncol - 1
    rows = h.nrow - 1
    cell, sub = context.triangle // 2, context.triangle % 2
    row, col = cell // cols, cell % cols
    # Both triangle variants start at p00 = (row, col); the others are p10, p11 (sub 0) and p11, p01 (sub 1).
    origin = _vertex(context, row, col)
    surface = wp.vec3(normal)
    if zeros >= 2:
        vr, vc = row, col
        if largest > 0:
            vr += int(sub == 1 or largest == 2)
            vc += int(sub == 0 or largest == 1)
        # The diagonal p00--p11 triangulation joins a vertex to six grid neighbours.
        ghost = bool(False)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr * dc != -1 and (dr != 0 or dc != 0):
                    if _faces_witness(context, vr + dr, vc + dc, origin, point, normal):
                        ghost = True
        if not ghost:
            return normal, distance
        # Its six faces: both triangles of cells (vr, vc) and (vr - 1, vc - 1), sub 0 of (vr, vc - 1)
        # and sub 1 of (vr - 1, vc).
        alignment = float(-2.0)
        for k in range(6):
            rr = vr - int(k >= 3)
            cc = vc - int(k == 2 or k >= 4)
            s = int(k == 1 or k == 3 or k == 5)
            if rr >= 0 and rr < rows and cc >= 0 and cc < cols:
                candidate = _cell_normal(context, rr, cc, s)
                if wp.dot(candidate, normal) > alignment:
                    alignment = wp.dot(candidate, normal)
                    surface = candidate
    else:
        # bary[k] == 0 is the edge opposite vertex k: the neighbour across it (cell nr, nc, triangle ns)
        # has one vertex (or, oc) off that edge.
        nr, nc, ns, orow, ocol = row, col, 1 - sub, row, col
        if sub == 0:
            if opposite == 0:
                nr, nc, orow, ocol = row, col + 1, row + 1, col + 2
            elif opposite == 1:
                orow, ocol = row + 1, col
            else:
                nr, nc, orow, ocol = row - 1, col, row - 1, col
        else:
            if opposite == 0:
                nr, nc, orow, ocol = row + 1, col, row + 2, col + 1
            elif opposite == 1:
                nr, nc, orow, ocol = row, col - 1, row, col - 1
            else:
                orow, ocol = row, col + 1
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            return normal, distance
        if not _faces_witness(context, orow, ocol, origin, point, normal):
            return normal, distance
        surface = _cell_normal(context, nr, nc, ns)
    return surface, wp.dot(distance * normal, surface)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Terrain surface witnesses for heightfield prism queries.

Each cell triangle is queried on its own, so its boundary edges and vertices
answer as a lone triangle would: toward a partner beside the triangle with an
oblique edge normal, and under a tilted flat partner with the partner's face
normal. Where the terrain continues past that boundary beneath the partner,
the surface there is the neighbouring face's: :func:`seam_surface_witness`
gives such a separated witness the contact of the face beneath the partner.
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
def faces_witness(toward: wp.vec3, normal: wp.vec3) -> bool:
    """The feature rule's test: whether a surface vertex at offset ``toward`` from a witness lies in front of it.

    A witness on an edge or vertex of a surface triangle is a local minimum of the distance to the surface only
    if no vertex adjacent to the feature lies in front of the plane through the witness with the witness normal
    (by more than ``SEAM_DIRECTION_TOLERANCE``, the witness normals' rounding). Heightfield triangles
    (:func:`seam_surface_witness`) and mesh triangles (``mesh_topology.mesh_contact_filter``) apply this one test.
    """
    return wp.dot(normal, toward) > SEAM_DIRECTION_TOLERANCE * wp.length(toward)


@wp.func
def _faces_witness(
    context: HeightfieldFeatureContext, row: int, col: int, origin: wp.vec3, point: wp.vec3, normal: wp.vec3
) -> bool:
    """Whether grid vertex ``(row, col)`` lies in front of the witness plane (absent vertices never do)."""
    h = context.heightfield
    if row < 0 or row >= h.nrow or col < 0 or col >= h.ncol:
        return False
    return faces_witness(_vertex(context, row, col) - origin - point, normal)


@wp.func
def _owner_face(context: HeightfieldFeatureContext, point: wp.vec3) -> tuple[int, wp.vec3, wp.vec3]:
    """Cell triangle under a heightfield-local point: its index (-1 off the grid), a vertex and its upward normal."""
    h = context.heightfield
    cols = h.ncol - 1
    rows = h.nrow - 1
    dx = 2.0 * h.hx / wp.float32(cols)
    dy = 2.0 * h.hy / wp.float32(rows)
    fx = (point[0] + h.hx) / dx
    fy = (point[1] + h.hy) / dy
    if not (wp.isfinite(fx) and wp.isfinite(fy)):
        return -1, wp.vec3(0.0), wp.vec3(0.0)
    col = int(wp.floor(fx))
    row = int(wp.floor(fy))
    if col < 0 or col >= cols or row < 0 or row >= rows:
        return -1, wp.vec3(0.0), wp.vec3(0.0)
    p00 = _vertex(context, row, col)
    p11 = _vertex(context, row + 1, col + 1)
    # Sub 0 is (p00, p10, p11), below the p00--p11 diagonal; sub 1 is (p00, p11, p01).
    sub = int(fy - wp.float32(row) > fx - wp.float32(col))
    face = wp.cross(_vertex(context, row, col + 1) - p00, p11 - p00)
    if sub == 1:
        face = wp.cross(p11 - p00, _vertex(context, row + 1, col) - p00)
    return (row * cols + col) * 2 + sub, p00, wp.normalize(face)


@wp.func
def seam_surface_witness(
    triangle: GenericShapeData,
    point: wp.vec3,
    normal: wp.vec3,
    distance: float,
    context: HeightfieldFeatureContext,
) -> tuple[wp.vec3, wp.vec3, float]:
    """The terrain's contact for a separated witness: unchanged, or the face beneath the partner.

    The point (on the triangle) and the normal are in the triangle-origin heightfield-local frame;
    the partner's witness point is ``point + distance * normal``. A face witness keeps its contact,
    and so does a boundary edge or vertex whose normal lies among the terrain's normals there:
    exactly when no terrain vertex adjacent to the feature lies in front of the witness plane (the
    neighbour's opposite vertex for an edge, the six incident grid vertices for a vertex). Otherwise
    the terrain continues past the boundary toward the partner, and the contact is the one of the
    cell triangle beneath the partner point: its upward normal, the partner's distance from its
    plane and the partner's foot on it. Witnesses off the top face, partners off the grid or over
    the triangle itself, and non-heightfield triangles keep theirs.

    Returns:
        ``(point, normal, distance)``: unchanged unless the face beneath the partner owns the contact.
    """
    h = context.heightfield
    if triangle.shape_type != int(GeoTypeEx.TRIANGLE_PRISM) or h.nrow < 2 or h.ncol < 2:
        return point, normal, distance
    if not (wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])):
        return point, normal, distance
    if not (wp.isfinite(normal[0]) and wp.isfinite(normal[1]) and wp.isfinite(normal[2])):
        return point, normal, distance
    e1, e2 = triangle.scale, triangle.auxiliary
    face = wp.cross(e1, e2)
    scale = wp.max(wp.length(e1), wp.length(e2))
    tolerance = 2.0e-6 * scale
    # Witnesses on the extruded walls or bottom lie off the top face; only its boundary has neighbours.
    if wp.abs(wp.dot(point, face)) > tolerance * wp.length(face):
        return point, normal, distance
    determinant = e1[0] * e2[1] - e1[1] * e2[0]
    if determinant <= 0.0:
        return point, normal, distance
    u = (point[0] * e2[1] - point[1] * e2[0]) / determinant
    v = (e1[0] * point[1] - e1[1] * point[0]) / determinant
    bary = wp.vec3(1.0 - u - v, u, v)
    epsilon = float(2.0e-6)
    if wp.min(bary) < -epsilon or wp.max(bary) > 1.0 + epsilon:
        return point, normal, distance
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
        return point, normal, distance
    cols = h.ncol - 1
    cell, sub = context.triangle // 2, context.triangle % 2
    row, col = cell // cols, cell % cols
    # Both triangle variants start at p00 = (row, col); the others are p10, p11 (sub 0) and p11, p01 (sub 1).
    origin = _vertex(context, row, col)
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
            return point, normal, distance
    else:
        # bary[k] == 0 is the edge opposite vertex k; the neighbour across it has one vertex off that edge.
        orow, ocol = row, col
        if sub == 0:
            if opposite == 0:
                orow, ocol = row + 1, col + 2
            elif opposite == 1:
                orow, ocol = row + 1, col
            else:
                orow, ocol = row - 1, col
        else:
            if opposite == 0:
                orow, ocol = row + 2, col + 1
            elif opposite == 1:
                orow, ocol = row, col - 1
            else:
                orow, ocol = row, col + 1
        if not _faces_witness(context, orow, ocol, origin, point, normal):
            return point, normal, distance
    partner = origin + point + distance * normal
    owner, owner_vertex, owner_normal = _owner_face(context, partner)
    if owner < 0 or owner == context.triangle:
        return point, normal, distance
    owner_distance = wp.dot(partner - owner_vertex, owner_normal)
    return partner - origin - owner_distance * owner_normal, owner_normal, owner_distance

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current flat-surface feature qualification for heightfield prism queries.

An interior triangulation edge is not a physical obstacle on a flat surface.
This deliberately narrow experiment leaves slopes, creases, exterior borders
and uncertain witnesses unchanged. It never caches dynamic elevations.
"""

import os

import warp as wp

from ..utils.heightfield import HeightfieldData
from .support_function import GenericShapeData, GeoTypeEx

WELD_FLAT_SEAMS = os.environ.get("NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS", "0") == "1"


@wp.struct
class HeightfieldFeatureContext:
    heightfield: HeightfieldData
    elevations: wp.array[float]
    triangle: int


@wp.func
def _height(context: HeightfieldFeatureContext, row: int, col: int) -> float:
    h = context.heightfield
    return h.min_z + context.elevations[h.data_offset + row * h.ncol + col] * (h.max_z - h.min_z)


@wp.func
def _flat_triangle(context: HeightfieldFeatureContext, triangle: int, height: float) -> bool:
    if triangle < 0:
        return False
    h = context.heightfield
    cell = triangle // 2
    row = cell // (h.ncol - 1)
    col = cell % (h.ncol - 1)
    if row < 0 or row >= h.nrow - 1 or col < 0 or col >= h.ncol - 1:
        return False
    same = _height(context, row, col) == height and _height(context, row + 1, col + 1) == height
    if triangle % 2 == 0:
        return same and _height(context, row, col + 1) == height
    return same and _height(context, row + 1, col) == height


@wp.func
def flat_seam_query_allowed(
    triangle: GenericShapeData,
    point: wp.vec3,
    normal: wp.vec3,
    context: HeightfieldFeatureContext,
) -> bool:
    """Reject only an oblique witness at an interior flat edge or complete fan.

    The point and normal are in the triangle-origin heightfield-local frame. The
    small witness-location allowance does not smooth terrain: flatness itself
    requires exact equality of current decoded vertex heights.
    """
    h = context.heightfield
    if triangle.shape_type != int(GeoTypeEx.TRIANGLE_PRISM) or h.nrow < 2 or h.ncol < 2:
        return True
    if not (wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])):
        return True
    if not (wp.isfinite(normal[0]) and wp.isfinite(normal[1]) and wp.isfinite(normal[2])):
        return True
    if normal[2] >= 1.0 - 1.0e-6 or normal[2] <= 0.0:
        return True
    e1, e2 = triangle.scale, triangle.auxiliary
    if e1[2] != 0.0 or e2[2] != 0.0:
        return True
    scale = wp.max(wp.length(e1), wp.length(e2))
    tolerance = 2.0e-6 * scale
    if wp.abs(point[2]) > tolerance:
        return True
    determinant = e1[0] * e2[1] - e1[1] * e2[0]
    if determinant <= 0.0:
        return True
    u = (point[0] * e2[1] - point[1] * e2[0]) / determinant
    v = (e1[0] * point[1] - e1[1] * point[0]) / determinant
    bary = wp.vec3(1.0 - u - v, u, v)
    epsilon = float(2.0e-6)
    if wp.min(bary) < -epsilon or wp.max(bary) > 1.0 + epsilon:
        return True
    zeros = int(0)
    largest = int(0)
    for k in range(3):
        if bary[k] <= epsilon:
            zeros += 1
        if bary[k] > bary[largest]:
            largest = k
    if zeros == 0:
        return True
    cols = h.ncol - 1
    cell, sub = context.triangle // 2, context.triangle % 2
    row, col = cell // cols, cell % cols
    height = _height(context, row, col)
    if not _flat_triangle(context, context.triangle, height):
        return True
    if zeros >= 2:
        # Both triangle variants start at p00. Their other vertices are
        # p10,p11 and p11,p01 respectively.
        vr, vc = row, col
        if largest > 0:
            vr += int(sub == 1 or largest == 2)
            vc += int(sub == 0 or largest == 1)
        if vr <= 0 or vr >= h.nrow - 1 or vc <= 0 or vc >= h.ncol - 1:
            return True
        # The diagonal p00--p11 triangulation has six incident triangles.
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr * dc != -1 and _height(context, vr + dr, vc + dc) != height:
                    return True
        return False
    # bary[k]==0 is the edge opposite vertex k.
    opposite = int(0)
    for k in range(3):
        if bary[k] <= epsilon:
            opposite = k
    neighbor = int(-1)
    if sub == 0:
        if opposite == 0 and col < cols - 1:
            neighbor = (cell + 1) * 2 + 1
        elif opposite == 1:
            neighbor = context.triangle + 1
        elif opposite == 2 and row > 0:
            neighbor = (cell - cols) * 2 + 1
    else:
        if opposite == 0 and row < h.nrow - 2:
            neighbor = (cell + cols) * 2
        elif opposite == 1 and col > 0:
            neighbor = (cell - 1) * 2
        elif opposite == 2:
            neighbor = context.triangle - 1
    return not _flat_triangle(context, neighbor, height)

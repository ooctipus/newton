# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental cuboid separation inside the existing heightfield append owner."""

import warp as wp

from ..utils.heightfield import HeightfieldData
from .contact_reduction_global import BETA_THRESHOLD
from .heightfield_cells import _abs_sum, _cell_is_above, _heightfield_cell_midphase
from .types import GeoType

Axes = wp.types.matrix(shape=(9, 3), dtype=wp.float32)
Bounds = wp.types.vector(length=9, dtype=wp.float32)


@wp.struct
class PairProjection:
    enabled: bool
    axes: Axes
    lower: Bounds
    upper: Bounds
    center: wp.vec3
    half: wp.vec3
    basis: wp.mat33
    threshold: float


@wp.func
def _finite3(value: wp.vec3) -> bool:
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _radius(pair: PairProjection, axis: wp.vec3) -> float:
    return (
        wp.abs(wp.dot(pair.basis[0], axis)) * pair.half[0]
        + wp.abs(wp.dot(pair.basis[1], axis)) * pair.half[1]
        + wp.abs(wp.dot(pair.basis[2], axis)) * pair.half[2]
    )


@wp.func
def _prepare_pair(
    a: int,
    b: int,
    hfd: HeightfieldData,
    types: wp.array[int],
    transforms: wp.array[wp.transform],
    data: wp.array[wp.vec4],
    sources: wp.array[wp.uint64],
    gaps: wp.array[float],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    bounds: wp.array2d[wp.vec3],
    bound_source: wp.array[wp.uint64],
) -> PairProjection:
    result = PairProjection()
    if types[a] != GeoType.HFIELD or (types[b] != GeoType.BOX and types[b] != GeoType.CONVEX_MESH):
        return result
    scale = wp.vec3(data[b][0], data[b][1], data[b][2])
    half = wp.cw_mul(bounds[b, 1], scale)
    local_center = wp.cw_mul(bounds[b, 0], scale)
    if bound_source[b] != sources[b] or not _finite3(scale) or not _finite3(half) or not _finite3(local_center):
        return result
    if scale[0] <= 0.0 or scale[1] <= 0.0 or scale[2] <= 0.0 or half[0] <= 0.0 or half[1] <= 0.0 or half[2] <= 0.0:
        return result
    if not _finite3(lower[a]) or not _finite3(upper[a]) or not _finite3(lower[b]) or not _finite3(upper[b]):
        return result
    for k in range(3):
        if (
            lower[a][k] > upper[a][k]
            or lower[b][k] > local_center[k] - half[k]
            or upper[b][k] < local_center[k] + half[k]
        ):
            return result
    if hfd.nrow < 2 or hfd.ncol < 2 or hfd.hx <= 0.0 or hfd.hy <= 0.0:
        return result
    if not (wp.isfinite(hfd.hx) and wp.isfinite(hfd.hy) and wp.isfinite(hfd.min_z) and wp.isfinite(hfd.max_z)):
        return result
    if hfd.max_z < hfd.min_z or data[a][3] < 0.0 or data[b][3] < 0.0 or gaps[a] < 0.0 or gaps[b] < 0.0:
        return result
    if not (wp.isfinite(data[a][3]) and wp.isfinite(data[b][3]) and wp.isfinite(gaps[a]) and wp.isfinite(gaps[b])):
        return result
    if data[a][3] + data[b][3] < 1.0e-4:
        return result
    xa, xb = transforms[a], transforms[b]
    qa0, qb0 = wp.transform_get_rotation(xa), wp.transform_get_rotation(xb)
    qa2, qb2 = wp.dot(qa0, qa0), wp.dot(qb0, qb0)
    if not (wp.isfinite(qa2) and wp.isfinite(qb2)) or wp.abs(qa2 - 1.0) > 2.0e-6 or wp.abs(qb2 - 1.0) > 2.0e-6:
        return result
    qa, qb = wp.normalize(qa0), wp.normalize(qb0)
    pa, pb = wp.transform_get_translation(xa), wp.transform_get_translation(xb)
    result.center = wp.quat_rotate_inv(qa, pb + wp.quat_rotate(qb, local_center) - pa)
    result.half = half
    rotation = wp.normalize(wp.quat_inverse(qa) * qb)
    result.basis = wp.matrix_from_rows(
        wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0)),
        wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0)),
        wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0)),
    )
    shell = (gaps[a] + gaps[b]) + (data[a][3] + data[b][3])
    beta = wp.static(BETA_THRESHOLD) * wp.length(upper[a] - lower[a])
    coordinate_scale = (
        1.0
        + _abs_sum(pa)
        + _abs_sum(pb)
        + _abs_sum(lower[b])
        + _abs_sum(upper[b])
        + wp.abs(hfd.min_z)
        + wp.abs(hfd.max_z - hfd.min_z)
        + wp.abs(shell)
    )
    padding = 0.001 + (64.0 * 1.1920928955078125e-7) * coordinate_scale
    result.threshold = wp.max(shell, beta) + 2.0 * padding
    if not _finite3(result.center) or not wp.isfinite(result.threshold) or result.threshold < 0.0:
        return result
    dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
    for k in range(3):
        horizontal = wp.cross(result.basis[k], wp.vec3(0.0, 0.0, 1.0))
        axis = result.basis[k]
        if axis[2] < 0.0:
            axis = -axis
        for j in range(3):
            result.axes[k, j] = horizontal[j]
            result.axes[k + 6, j] = axis[j]
    # Every cell has these three horizontal triangle-edge directions. They are
    # valid separating axes even when reconstructed FP32 grid edges round apart.
    result.axes[3, 0] = 1.0
    result.axes[4, 1] = 1.0
    result.axes[5, 0] = dy
    result.axes[5, 1] = -dx
    for k in range(9):
        axis = result.axes[k]
        n2 = wp.dot(axis, axis)
        if not wp.isfinite(n2):
            return result
        if n2 > 1.0e-20:
            axis = axis / wp.sqrt(n2)
        else:
            axis = wp.vec3(0.0)
        for j in range(3):
            result.axes[k, j] = axis[j]
        radius = _radius(result, axis)
        center = wp.dot(result.center, axis)
        result.lower[k] = center - radius - result.threshold
        result.upper[k] = center + radius + result.threshold
        if not (wp.isfinite(result.lower[k]) and wp.isfinite(result.upper[k])):
            return result
    result.enabled = True
    return result


@wp.func
def _triangle_is_separated(pair: PairProjection, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> bool:
    """Separate only horizontally or upward, retaining the entire downward prism."""
    if not (_finite3(v0) and _finite3(v1) and _finite3(v2)):
        return False
    for k in range(9):
        axis = pair.axes[k]
        a, b, c = wp.dot(axis, v0), wp.dot(axis, v1), wp.dot(axis, v2)
        hi, lo = wp.max(a, wp.max(b, c)), wp.min(a, wp.min(b, c))
        if k < 6:
            if pair.lower[k] > hi or pair.upper[k] < lo:
                return True
        elif axis[2] > 0.0 and pair.lower[k] > hi:
            return True
    axis = wp.cross(v1 - v0, v2 - v0)
    n2 = wp.dot(axis, axis)
    if wp.isfinite(n2) and n2 > 1.0e-20 and axis[2] > 0.0:
        gap = wp.dot(pair.center - v0, axis) - _radius(pair, axis)
        if wp.isfinite(gap) and gap > pair.threshold * wp.sqrt(n2):
            return True
    return False


@wp.func
def _geometric_midphase(
    a: int,
    b: int,
    hfd: HeightfieldData,
    pair: PairProjection,
    transforms: wp.array[wp.transform],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    data: wp.array[wp.vec4],
    gaps: wp.array[float],
    elevations: wp.array[float],
    triples: wp.array[wp.vec3i],
    count: wp.array[int],
):
    # Preserve the original range construction and its existing current-height
    # cull. Only the additional triangle rejection precedes the same append.
    xa, xb = transforms[a], transforms[b]
    relative = wp.transform_multiply(wp.transform_inverse(xa), xb)
    position, rotation = wp.transform_get_translation(relative), wp.transform_get_rotation(relative)
    center_local, half_local = 0.5 * (lower[b] + upper[b]), 0.5 * (upper[b] - lower[b])
    center = wp.quat_rotate(rotation, center_local) + position
    r0 = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
    r1 = wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0))
    r2 = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    half = wp.vec3(
        wp.abs(r0[0]) * half_local[0] + wp.abs(r1[0]) * half_local[1] + wp.abs(r2[0]) * half_local[2],
        wp.abs(r0[1]) * half_local[0] + wp.abs(r1[1]) * half_local[1] + wp.abs(r2[1]) * half_local[2],
        wp.abs(r0[2]) * half_local[0] + wp.abs(r1[2]) * half_local[1] + wp.abs(r2[2]) * half_local[2],
    )
    shell = (gaps[a] + gaps[b]) + (data[a][3] + data[b][3])
    lo, hi = center - half - wp.vec3(shell), center + half + wp.vec3(shell)
    dx, dy = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1), 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
    cmin = wp.max(wp.int32(wp.floor((lo[0] + hfd.hx) / dx)), 0)
    cmax = wp.min(wp.int32(wp.floor((hi[0] + hfd.hx) / dx)), hfd.ncol - 2)
    rmin = wp.max(wp.int32(wp.floor((lo[1] + hfd.hy) / dy)), 0)
    rmax = wp.min(wp.int32(wp.floor((hi[1] + hfd.hy) / dy)), hfd.nrow - 2)
    coordinate_scale = (
        1.0
        + _abs_sum(wp.transform_get_translation(xa))
        + _abs_sum(wp.transform_get_translation(xb))
        + _abs_sum(center_local)
        + _abs_sum(half_local)
        + _abs_sum(center)
        + _abs_sum(half)
        + wp.abs(hfd.min_z)
        + wp.abs(hfd.max_z - hfd.min_z)
        + wp.abs(shell)
    )
    padding = 0.0002 + (64.0 * 1.1920928955078125e-7) * coordinate_scale
    can_reject = wp.isfinite(lo[2]) and wp.isfinite(padding)
    height_range = hfd.max_z - hfd.min_z
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if can_reject and _cell_is_above(lo[2], padding, hfd, elevations, r, c):
                continue
            index = hfd.data_offset + r * hfd.ncol + c
            x0, y0 = -hfd.hx + wp.float32(c) * dx, -hfd.hy + wp.float32(r) * dy
            v0 = wp.vec3(x0, y0, hfd.min_z + elevations[index] * height_range)
            v10 = wp.vec3(x0 + dx, y0, hfd.min_z + elevations[index + 1] * height_range)
            v11 = wp.vec3(x0 + dx, y0 + dy, hfd.min_z + elevations[index + hfd.ncol + 1] * height_range)
            v01 = wp.vec3(x0, y0 + dy, hfd.min_z + elevations[index + hfd.ncol] * height_range)
            for sub in range(2):
                v1, v2 = v10, v11
                if sub == 1:
                    v1, v2 = v11, v01
                if _triangle_is_separated(pair, v0, v1, v2):
                    continue
                tri = (r * (hfd.ncol - 1) + c) * 2 + sub
                out = wp.atomic_add(count, 0, 1)
                if out < triples.shape[0]:
                    triples[out] = wp.vec3i(a, b, tri)


@wp.kernel(enable_backward=False, module="unique")
def heightfield_geometric_overlaps_kernel(
    transforms: wp.array[wp.transform],
    gaps: wp.array[float],
    data: wp.array[wp.vec4],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    heightfield_index: wp.array[int],
    heightfield: wp.array[HeightfieldData],
    elevations: wp.array[float],
    pairs: wp.array[wp.vec2i],
    pair_count: wp.array[int],
    total_num_threads: int,
    types: wp.array[int],
    sources: wp.array[wp.uint64],
    bounds: wp.array2d[wp.vec3],
    bound_source: wp.array[wp.uint64],
    triples: wp.array[wp.vec3i],
    count: wp.array[int],
):
    """Compact surviving triangles directly in the original pair/stride owner."""
    tid, j = wp.tid()
    if j != 0:
        return
    for i in range(tid, pair_count[0], total_num_threads):
        a, b = pairs[i][0], pairs[i][1]
        hfd = heightfield[heightfield_index[a]]
        projection = _prepare_pair(
            a, b, hfd, types, transforms, data, sources, gaps, lower, upper, bounds, bound_source
        )
        if projection.enabled:
            _geometric_midphase(a, b, hfd, projection, transforms, lower, upper, data, gaps, elevations, triples, count)
        else:
            _heightfield_cell_midphase(a, b, hfd, transforms, lower, upper, data, gaps, elevations, triples, count)

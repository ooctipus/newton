# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in current-height rejection before heightfield triangle emission."""

import warp as wp

from ..utils.heightfield import HeightfieldData


@wp.func
def _abs_sum(value: wp.vec3) -> float:
    return wp.abs(value[0]) + wp.abs(value[1]) + wp.abs(value[2])


@wp.func
def _cell_is_above(
    lower_z: float,
    padding: float,
    hfd: HeightfieldData,
    elevations: wp.array[float],
    row: int,
    col: int,
) -> bool:
    """Reject only above all four current corners; retain the entire downward prism."""
    i = hfd.data_offset + row * hfd.ncol + col
    height_range = hfd.max_z - hfd.min_z
    z00 = hfd.min_z + elevations[i] * height_range
    z10 = hfd.min_z + elevations[i + 1] * height_range
    z01 = hfd.min_z + elevations[i + hfd.ncol] * height_range
    z11 = hfd.min_z + elevations[i + hfd.ncol + 1] * height_range
    if not (wp.isfinite(z00) and wp.isfinite(z10) and wp.isfinite(z01) and wp.isfinite(z11)):
        return False
    top = wp.max(wp.max(z00, z10), wp.max(z01, z11))
    return lower_z > top + padding + (16.0 * 1.1920928955078125e-7) * wp.abs(top)


@wp.func
def _heightfield_cell_midphase(
    hfield_shape: int,
    other_shape: int,
    hfd: HeightfieldData,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_data: wp.array[wp.vec4],
    shape_gap: wp.array[float],
    elevations: wp.array[float],
    triangle_pairs: wp.array[wp.vec3i],
    triangle_pairs_count: wp.array[int],
):
    """Keep the original scaled AABB, XY range and surviving triangle emission law."""
    X_hfield_ws = shape_transform[hfield_shape]
    X_other_ws = shape_transform[other_shape]
    X_other_in_hfield = wp.transform_multiply(wp.transform_inverse(X_hfield_ws), X_other_ws)
    other_pos = wp.transform_get_translation(X_other_in_hfield)
    other_rot = wp.transform_get_rotation(X_other_in_hfield)
    local_lo = shape_collision_aabb_lower[other_shape]
    local_hi = shape_collision_aabb_upper[other_shape]
    local_center = 0.5 * (local_lo + local_hi)
    local_half = 0.5 * (local_hi - local_lo)
    center_in_hfield = wp.quat_rotate(other_rot, local_center) + other_pos
    r0 = wp.quat_rotate(other_rot, wp.vec3(1.0, 0.0, 0.0))
    r1 = wp.quat_rotate(other_rot, wp.vec3(0.0, 1.0, 0.0))
    r2 = wp.quat_rotate(other_rot, wp.vec3(0.0, 0.0, 1.0))
    half_in_hfield = wp.vec3(
        wp.abs(r0[0]) * local_half[0] + wp.abs(r1[0]) * local_half[1] + wp.abs(r2[0]) * local_half[2],
        wp.abs(r0[1]) * local_half[0] + wp.abs(r1[1]) * local_half[1] + wp.abs(r2[1]) * local_half[2],
        wp.abs(r0[2]) * local_half[0] + wp.abs(r1[2]) * local_half[1] + wp.abs(r2[2]) * local_half[2],
    )
    gap_sum = shape_gap[hfield_shape] + shape_gap[other_shape]
    margin_sum = shape_data[hfield_shape][3] + shape_data[other_shape][3]
    contact_threshold = gap_sum + margin_sum
    threshold_vec = wp.vec3(contact_threshold, contact_threshold, contact_threshold)
    aabb_lower = center_in_hfield - half_in_hfield - threshold_vec
    aabb_upper = center_in_hfield + half_in_hfield + threshold_vec
    dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
    col_min_f = (aabb_lower[0] + hfd.hx) / dx
    col_max_f = (aabb_upper[0] + hfd.hx) / dx
    row_min_f = (aabb_lower[1] + hfd.hy) / dy
    row_max_f = (aabb_upper[1] + hfd.hy) / dy
    col_min = wp.max(wp.int32(wp.floor(col_min_f)), 0)
    col_max = wp.min(wp.int32(wp.floor(col_max_f)), hfd.ncol - 2)
    row_min = wp.max(wp.int32(wp.floor(row_min_f)), 0)
    row_max = wp.min(wp.int32(wp.floor(row_max_f)), hfd.nrow - 2)
    cols = hfd.ncol - 1

    # Include current world-coordinate cancellation, transformed extent and height
    # reconstruction roundoff. The additional 0.2 mm also exceeds the original
    # MPR query inflation; it can only retain extra candidates. No below-cell cull.
    scale = (
        1.0
        + _abs_sum(wp.transform_get_translation(X_hfield_ws))
        + _abs_sum(wp.transform_get_translation(X_other_ws))
        + _abs_sum(local_center)
        + _abs_sum(local_half)
        + _abs_sum(center_in_hfield)
        + _abs_sum(half_in_hfield)
        + wp.abs(hfd.min_z)
        + wp.abs(hfd.max_z - hfd.min_z)
        + wp.abs(contact_threshold)
    )
    padding = 0.0002 + (64.0 * 1.1920928955078125e-7) * scale
    can_reject = wp.isfinite(aabb_lower[2]) and wp.isfinite(padding)
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            if can_reject and _cell_is_above(aabb_lower[2], padding, hfd, elevations, r, c):
                continue
            for tri_sub in range(2):
                tri_idx = (r * cols + c) * 2 + tri_sub
                out_idx = wp.atomic_add(triangle_pairs_count, 0, 1)
                if out_idx < triangle_pairs.shape[0]:
                    triangle_pairs[out_idx] = wp.vec3i(hfield_shape, other_shape, tri_idx)


@wp.kernel(enable_backward=False, module="unique")
def heightfield_cell_overlaps_kernel(
    shape_transform: wp.array[wp.transform],
    shape_gap: wp.array[float],
    shape_data: wp.array[wp.vec4],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_heightfield_index: wp.array[int],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[float],
    shape_pairs_mesh: wp.array[wp.vec2i],
    shape_pairs_mesh_count: wp.array[int],
    total_num_threads: int,
    triangle_pairs: wp.array[wp.vec3i],
    triangle_pairs_count: wp.array[int],
):
    """Use the original thread/stride schedule for normalized heightfield-only pairs."""
    tid, j = wp.tid()
    if j != 0:
        return
    for i in range(tid, shape_pairs_mesh_count[0], total_num_threads):
        pair = shape_pairs_mesh[i]
        hfd = heightfield_data[shape_heightfield_index[pair[0]]]
        _heightfield_cell_midphase(
            pair[0],
            pair[1],
            hfd,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_data,
            shape_gap,
            heightfield_elevations,
            triangle_pairs,
            triangle_pairs_count,
        )

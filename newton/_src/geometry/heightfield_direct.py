# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pair-owned current heightfield queries without an intermediate triangle stream."""

from functools import cache
from typing import Any

import warp as wp

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_core import create_compute_gjk_mpr_contacts
from .heightfield_cells import _abs_sum, _cell_is_above
from .support_function import create_triangle_prism_penetration_refiner, extract_shape_data, support_map


@wp.struct
class CellRange:
    """Current original bounds, not a persistent terrain or convex cache."""

    col_min: int
    col_max: int
    row_min: int
    row_max: int
    grid_cols: int
    lower_z: float
    padding: float
    can_reject: bool


@wp.func
def _current_cell_range(
    hfield_shape: int,
    other_shape: int,
    hfd: HeightfieldData,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_data: wp.array[wp.vec4],
    shape_gap: wp.array[float],
) -> CellRange:
    """Preserve the accepted current-height path's exact AABB and padding arithmetic."""
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
    result = CellRange()
    result.col_min = col_min
    result.col_max = col_max
    result.row_min = row_min
    result.row_max = row_max
    result.grid_cols = cols
    result.lower_z = aabb_lower[2]
    result.padding = padding
    result.can_reject = can_reject
    return result


@cache
def create_heightfield_direct_kernel(writer_func: Any):
    """Call the original triangle query/writer directly from a pair-owned traversal."""
    query = create_compute_gjk_mpr_contacts(
        writer_func, penetration_refiner=create_triangle_prism_penetration_refiner(support_map)
    )

    @wp.kernel(enable_backward=False, module="unique")
    def heightfield_direct_contacts(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_heightfield_index: wp.array[int],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[float],
        shape_pairs_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_count: wp.array[int],
        triangle_pairs_count: wp.array[int],
        writer_data: Any,
        pair_workers: int,
    ):
        block, lane = wp.tid()
        triangle_demand = int(0)
        count = wp.min(shape_pairs_mesh_count[0], shape_pairs_mesh.shape[0])
        for pair_index in range(block, count, pair_workers):
            pair = shape_pairs_mesh[pair_index]
            shape_a = pair[0]
            shape_b = pair[1]
            hfd = heightfield_data[shape_heightfield_index[shape_a]]
            bounds = _current_cell_range(
                shape_a,
                shape_b,
                hfd,
                shape_transform,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_data,
                shape_gap,
            )
            pos_b, quat_b, shape_data_b, _scale_b, margin_offset_b = extract_shape_data(
                shape_b, shape_transform, shape_types, shape_data, shape_source
            )
            X_ws_a = shape_transform[shape_a]
            quat_a = wp.transform_get_rotation(X_ws_a)
            margin_offset_a = shape_data[shape_a][3]
            gap_sum = shape_gap[shape_a] + shape_gap[shape_b]
            width = wp.max(bounds.col_max - bounds.col_min + 1, 0)
            height = wp.max(bounds.row_max - bounds.row_min + 1, 0)
            for cell in range(lane, width * height, wp.block_dim()):
                row = bounds.row_min + cell // width
                col = bounds.col_min + cell % width
                if bounds.can_reject and _cell_is_above(
                    bounds.lower_z, bounds.padding, hfd, heightfield_elevations, row, col
                ):
                    continue
                triangle_demand += 2
                for tri_sub in range(2):
                    tri_idx = (row * bounds.grid_cols + col) * 2 + tri_sub
                    shape_data_a, v0_world = get_triangle_shape_from_heightfield(
                        hfd, heightfield_elevations, X_ws_a, tri_idx
                    )
                    wp.static(query)(
                        shape_data_a,
                        shape_data_b,
                        quat_a,
                        quat_b,
                        v0_world,
                        pos_b,
                        gap_sum,
                        shape_a,
                        shape_b,
                        margin_offset_a,
                        margin_offset_b,
                        writer_data,
                        (tri_idx << 1) | 1,
                    )
        # The intermediate stream is retired, but the original logical demand
        # and calibrated overflow check remain. No per-triangle reservations.
        total = wp.tile_sum(wp.tile(triangle_demand))[0]
        if lane == 0 and total > 0:
            wp.atomic_add(triangle_pairs_count, 0, total)

    return heightfield_direct_contacts

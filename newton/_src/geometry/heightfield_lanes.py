# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-pass terrain midphase: one thread per pair, then lanes across the cells of long pairs.

The original midphase gives each heightfield pair one thread that walks every
cell of the pair's window. A warp then runs as long as its longest window, and
a few pairs with many cells (long capsules, large feet, wide hulls) idle the
other 31 lanes. Here pass 1 still runs one thread per pair and builds the same
window, height bound and enclosing-box projection. A pair with at most
``small_cells`` cells walks its cells in place, in the original order. A longer
pair is deferred: pass 1 stores its window and projection, and pass 2 splits
its cells over ``lanes`` threads of one warp. When the deferred storage is full,
pass 1 walks the pair in place.

Every cell runs the original tests on the same inputs, so the emitted triangle
set is unchanged; only its order changes. Survivors of one cell are appended
with one reservation, which CUDA aggregates across the lanes of a warp. With
``small_cells < 0`` nothing is deferred and the triangle stream is identical to
the original, including its order on CPU.
"""

import functools

import warp as wp

from ..utils.heightfield import HeightfieldData
from .heightfield_cells import _abs_sum, _cell_is_above
from .heightfield_geometric import PairProjection, _prepare_pair, _triangle_is_separated
from .types import GeoType

# Deferred pairs index their cells with one flat 32-bit counter.
_MAX_DEFERRED_CELLS = wp.constant(1 << 30)


@wp.struct
class TerrainPairs:
    """Current geometry of the heightfield pairs that the midphase reads."""

    transforms: wp.array[wp.transform]
    gaps: wp.array[float]
    data: wp.array[wp.vec4]
    lower: wp.array[wp.vec3]
    upper: wp.array[wp.vec3]
    heightfield_index: wp.array[int]
    heightfield: wp.array[HeightfieldData]
    elevations: wp.array[float]
    pairs: wp.array[wp.vec2i]
    pair_count: wp.array[int]
    types: wp.array[int]
    sources: wp.array[wp.uint64]
    bounds: wp.array2d[wp.vec3]
    bound_source: wp.array[wp.uint64]


@wp.struct
class TerrainTriangles:
    """The compacted triangle stream: (heightfield, shape, triangle) triples and optional pair tags."""

    triples: wp.array[wp.vec3i]
    count: wp.array[int]
    tags: wp.array[int]


@wp.struct
class CellWindow:
    """One pair's cell range and height-rejection bound in heightfield coordinates."""

    pair: int
    a: int
    b: int
    row: int
    col: int
    rows: int
    cols: int
    reject: bool
    geometric: bool
    lower_z: float
    padding: float


@wp.struct
class DeferredPairs:
    """Windows and projections of the pairs that pass 2 splits over lanes."""

    windows: wp.array[CellWindow]
    projections: wp.array[PairProjection]
    count: wp.array[int]


@wp.func_native("""
#if defined(__CUDA_ARCH__)
const unsigned int active = __activemask();
unsigned int lane;
asm("mov.u32 %0, %%laneid;" : "=r"(lane));
const unsigned int below = (1u << lane) - 1u;
const unsigned int one = __ballot_sync(active, amount > 0);
const unsigned int two = __ballot_sync(active, amount > 1);
const int leader = __ffs(active) - 1;
int base = 0;
if (static_cast<int>(lane) == leader)
    base = atomicAdd(count.data, __popc(one) + __popc(two));
base = __shfl_sync(active, base, leader);
return base + __popc(one & below) + __popc(two & below);
#else
return wp::atomic_add(count, 0, amount);
#endif
""")
def _reserve_triangles(count: wp.array[int], amount: int) -> int:
    """Reserve ``amount`` (one or two) consecutive slots; the lanes that call together share one atomic."""
    ...


@wp.func
def _cell_window(
    pair: int,
    a: int,
    b: int,
    hfd: HeightfieldData,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_data: wp.array[wp.vec4],
    shape_gap: wp.array[float],
    reject_cells: bool,
) -> CellWindow:
    """Build the original midphase range and height bound.

    The statements up to ``padding`` repeat ``_heightfield_cell_midphase``'s range
    construction statement for statement; ``_geometric_midphase`` computes the
    same window with identical arithmetic under other names. Every cell therefore
    sees the bound of the midphase it replaces. test_heightfield_lane_midphase
    pins both copies.
    """
    X_hfield_ws = shape_transform[a]
    X_other_ws = shape_transform[b]
    X_other_in_hfield = wp.transform_multiply(wp.transform_inverse(X_hfield_ws), X_other_ws)
    other_pos = wp.transform_get_translation(X_other_in_hfield)
    other_rot = wp.transform_get_rotation(X_other_in_hfield)
    local_lo = shape_collision_aabb_lower[b]
    local_hi = shape_collision_aabb_upper[b]
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
    gap_sum = shape_gap[a] + shape_gap[b]
    margin_sum = shape_data[a][3] + shape_data[b][3]
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
    window = CellWindow()
    window.pair = pair
    window.a = a
    window.b = b
    window.row = row_min
    window.col = col_min
    # Saturated float-to-int bounds of an inverted window could wrap ``last - first + 1``;
    # 0 <= first and last <= n - 2 keep the difference of an ordered range in int32.
    window.rows = wp.where(row_max >= row_min, row_max - row_min + 1, 0)
    window.cols = wp.where(col_max >= col_min, col_max - col_min + 1, 0)
    window.reject = reject_cells and wp.isfinite(aabb_lower[2]) and wp.isfinite(padding)
    window.lower_z = aabb_lower[2]
    window.padding = padding
    return window


@wp.func
def _cell_survivors(
    window: CellWindow,
    projection: PairProjection,
    hfd: HeightfieldData,
    elevations: wp.array[float],
    row: int,
    col: int,
    dx: float,
    dy: float,
    height_range: float,
) -> int:
    """Return the cell's surviving triangles as bits 0 and 1, with the original tests and vertices."""
    if window.reject and _cell_is_above(window.lower_z, window.padding, hfd, elevations, row, col):
        return 0
    if not window.geometric:
        return 3
    index = hfd.data_offset + row * hfd.ncol + col
    x0, y0 = -hfd.hx + wp.float32(col) * dx, -hfd.hy + wp.float32(row) * dy
    v0 = wp.vec3(x0, y0, hfd.min_z + elevations[index] * height_range)
    v10 = wp.vec3(x0 + dx, y0, hfd.min_z + elevations[index + 1] * height_range)
    v11 = wp.vec3(x0 + dx, y0 + dy, hfd.min_z + elevations[index + hfd.ncol + 1] * height_range)
    v01 = wp.vec3(x0, y0 + dy, hfd.min_z + elevations[index + hfd.ncol] * height_range)
    survivors = int(0)
    if not _triangle_is_separated(projection, v0, v10, v11):
        survivors = 1
    if not _triangle_is_separated(projection, v0, v11, v01):
        survivors = survivors | 2
    return survivors


@functools.cache
def create_lane_midphase_kernels(geometric: bool, tagged: bool):
    """Create the per-pair pass and the lane pass for one midphase.

    Args:
        geometric: Apply the enclosing-box triangle rejection of the geometric
            cull to the pairs it admits; otherwise every pair takes cell rejection.
        tagged: Record each triangle's pair index (the pair-CSR ownership tag).

    Returns:
        ``(pairs_kernel, lanes_kernel)``. Both take ``(scene, triangles, deferred,
        parameter, total_num_threads)``, where ``parameter`` is the largest cell
        count walked in place (negative: never defer) for the first and the lane
        count per deferred pair for the second.
    """

    @wp.func
    def emit(
        window: CellWindow,
        projection: PairProjection,
        hfd: HeightfieldData,
        elevations: wp.array[float],
        triangles: TerrainTriangles,
        row: int,
        col: int,
        dx: float,
        dy: float,
        height_range: float,
    ):
        survivors = _cell_survivors(window, projection, hfd, elevations, row, col, dx, dy, height_range)
        if survivors == 0:
            return
        base = _reserve_triangles(triangles.count, (survivors & 1) + (survivors >> 1))
        first = (row * (hfd.ncol - 1) + col) * 2
        for sub in range(2):
            if ((survivors >> sub) & 1) != 0:
                if base < triangles.triples.shape[0]:
                    triangles.triples[base] = wp.vec3i(window.a, window.b, first + sub)
                    if wp.static(tagged):
                        triangles.tags[base] = window.pair
                base += 1

    def heightfield_pairs(
        scene: TerrainPairs,
        triangles: TerrainTriangles,
        deferred: DeferredPairs,
        small_cells: int,
        total_num_threads: int,
    ):
        tid = wp.tid()
        for i in range(tid, wp.min(scene.pair_count[0], scene.pairs.shape[0]), total_num_threads):
            a, b = scene.pairs[i][0], scene.pairs[i][1]
            hfd = scene.heightfield[scene.heightfield_index[a]]
            projection = PairProjection()
            if wp.static(geometric):
                projection = _prepare_pair(
                    a,
                    b,
                    hfd,
                    scene.types,
                    scene.transforms,
                    scene.data,
                    scene.sources,
                    scene.gaps,
                    scene.lower,
                    scene.upper,
                    scene.bounds,
                    scene.bound_source,
                )
            window = _cell_window(
                i,
                a,
                b,
                hfd,
                scene.transforms,
                scene.lower,
                scene.upper,
                scene.data,
                scene.gaps,
                # A plane's cached local AABB does not bound its full surface.
                scene.types[b] != GeoType.PLANE,
            )
            window.geometric = projection.enabled
            if small_cells >= 0 and window.cols > 0:
                if window.rows > small_cells / window.cols and window.rows <= _MAX_DEFERRED_CELLS / window.cols:
                    slot = wp.atomic_add(deferred.count, 0, 1)
                    if slot < deferred.windows.shape[0]:
                        deferred.windows[slot] = window
                        if window.geometric:
                            deferred.projections[slot] = projection
                        continue
            dx, dy = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1), 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
            height_range = hfd.max_z - hfd.min_z
            for row in range(window.row, window.row + window.rows):
                for col in range(window.col, window.col + window.cols):
                    emit(window, projection, hfd, scene.elevations, triangles, row, col, dx, dy, height_range)

    def heightfield_lanes(
        scene: TerrainPairs,
        triangles: TerrainTriangles,
        deferred: DeferredPairs,
        lanes: int,
        total_num_threads: int,
    ):
        tid = wp.tid()
        groups = total_num_threads / lanes
        if tid >= groups * lanes:
            return
        lane = tid % lanes
        count = wp.min(deferred.count[0], deferred.windows.shape[0])
        for slot in range(tid / lanes, count, groups):
            window = deferred.windows[slot]
            hfd = scene.heightfield[scene.heightfield_index[window.a]]
            projection = PairProjection()
            if wp.static(geometric):
                if window.geometric:
                    projection = deferred.projections[slot]
            dx, dy = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1), 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
            height_range = hfd.max_z - hfd.min_z
            for cell in range(lane, window.rows * window.cols, lanes):
                row, col = window.row + cell / window.cols, window.col + cell % window.cols
                emit(window, projection, hfd, scene.elevations, triangles, row, col, dx, dy, height_range)

    # Profiles group these with the one-pass midphase kernels they replace.
    stem = "heightfield_pair_csr_overlaps" if tagged else "heightfield_geometric_overlaps"
    if not geometric:
        stem = "heightfield_cell_overlaps_tagged" if tagged else "heightfield_cell_overlaps"
    kernels = []
    for kernel, suffix in ((heightfield_pairs, "pairs"), (heightfield_lanes, "lanes")):
        kernel.__name__ = kernel.__qualname__ = f"{stem}_{suffix}"
        kernels.append(wp.kernel(kernel, enable_backward=False, module="unique"))
    return tuple(kernels)

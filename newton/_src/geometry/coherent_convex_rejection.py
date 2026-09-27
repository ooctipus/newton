# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental current-shape rejection only; retained queries stay cold."""

import warp as wp

from .coherent_convex import _CacheData, _find_slot, _finite_vector, _plane_lower_bound
from .collision_convex import ConvexQueryResult
from .mpr import create_solve_mpr, create_support_map_function
from .simplex_solver import create_solve_closest_distance
from .support_function import SupportMapDataProvider, support_map_lean


@wp.struct
class _PairInputs:
    shape_types: wp.array[int]
    shape_data: wp.array[wp.vec4]
    shape_transform: wp.array[wp.transform]
    shape_source: wp.array[wp.uint64]
    shape_gap: wp.array[float]
    shape_collision_radius: wp.array[float]
    shape_aabb_lower: wp.array[wp.vec3]
    shape_aabb_upper: wp.array[wp.vec3]
    shape_collision_aabb_lower: wp.array[wp.vec3]
    shape_collision_aabb_upper: wp.array[wp.vec3]


@wp.func
def _store_state(cache: _CacheData, slot: int, pair: wp.vec2i, inputs: _PairInputs, mode: int, axis: wp.vec3):
    if slot >= 0:
        cache.source_a[slot] = inputs.shape_source[pair[0]]
        cache.source_b[slot] = inputs.shape_source[pair[1]]
        cache.types[slot] = wp.vec2i(inputs.shape_types[pair[0]], inputs.shape_types[pair[1]])
        cache.mode[slot] = mode
        cache.axis[slot] = axis
        cache.entry_epoch[slot] = cache.world_epoch[cache.world[slot]]
        cache.entry_generation[slot] = cache.generation[0]


def _create_rejection_query_kernels(*, diagnostics=False):
    """Create a certificate filter and the original cold MPR/GJK queue owners.

    No cached witness, simplex or warm solver can produce a retained result.
    The existing manifold kernel owns all contact publication afterward.
    """
    from .narrow_phase import _append_work_index_compacted, create_prepare_convex_pair  # noqa: PLC0415

    prepare = create_prepare_convex_pair(True, False)
    maps = create_support_map_function(support_map_lean, use_precomputed_center=True)
    solve_mpr = create_solve_mpr(support_map_lean, _support_funcs=maps).core
    solve_gjk = create_solve_closest_distance(support_map_lean, _support_funcs=maps).core

    @wp.func
    def prepare_pair(pair: wp.vec2i, data: _PairInputs):
        return prepare(
            pair,
            data.shape_types,
            data.shape_data,
            data.shape_transform,
            data.shape_source,
            data.shape_gap,
            data.shape_collision_radius,
            data.shape_aabb_lower,
            data.shape_aabb_upper,
            data.shape_collision_aabb_lower,
            data.shape_collision_aabb_upper,
        )

    @wp.kernel(enable_backward=False, module="unique")
    def classify_rejections(
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        inputs: _PairInputs,
        cache: _CacheData,
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        gjk_work_items: wp.array[int],
        gjk_work_count: wp.array[int],
        manifold_work_items: wp.array[int],
        manifold_work_count: wp.array[int],
    ):
        tid = wp.tid()
        count = wp.min(query_results.shape[0], wp.min(candidate_pair.shape[0], candidate_pair_count[0]))
        block_dim = wp.block_dim()
        num_blocks = total_num_threads // block_dim
        block_index = tid // block_dim
        lane = tid - block_index * block_dim
        items_per_block = (count + num_blocks - 1) // num_blocks
        block_start = block_index * items_per_block
        for local_index in range(lane, items_per_block, block_dim):
            index = block_start + local_index
            needs_cold = False
            if index < count:
                pair = candidate_pair[index]
                cache.query_slot[index] = -1
                cache.query_done[index] = 0
                valid, query = prepare_pair(pair, inputs)
                if not valid:
                    cache.query_done[index] = 1
                else:
                    slot = _find_slot(cache, pair)
                    cache.query_slot[index] = slot
                    if slot >= 0:
                        if wp.static(diagnostics):
                            wp.atomic_add(cache.stats, 0, 1)
                        prior = cache.entry_generation[slot]
                        same_source = (
                            cache.source_a[slot] == inputs.shape_source[pair[0]]
                            and cache.source_b[slot] == inputs.shape_source[pair[1]]
                            and cache.types[slot] == wp.vec2i(query.type_a, query.type_b)
                        )
                        current = (
                            prior != wp.uint64(0)
                            and prior + wp.uint64(1) == cache.generation[0]
                            and cache.entry_epoch[slot] == cache.world_epoch[cache.world[slot]]
                            and same_source
                        )
                        mode = cache.mode[slot] if current else 0
                        if mode == 1 or mode == 2:
                            if wp.static(diagnostics):
                                wp.atomic_add(cache.stats, 1, 1)
                            axis = cache.axis[slot]
                            bound = _plane_lower_bound(
                                query.geom_a,
                                query.geom_b,
                                query.relative_orientation_b,
                                query.relative_position_b,
                                axis,
                            )
                            if bound.valid != 0 and bound.lower > query.contact_threshold:
                                cache.query_done[index] = 1
                                _store_state(cache, slot, pair, inputs, 1, axis)
                                if wp.static(diagnostics):
                                    wp.atomic_add(cache.stats, 2, 1)
                    needs_cold = cache.query_done[index] == 0
            _append_work_index_compacted(needs_cold, index, cache.unresolved_work_items, cache.unresolved_work_count)

    @wp.kernel(enable_backward=False, module="unique")
    def cold_mpr(
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        inputs: _PairInputs,
        cache: _CacheData,
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        gjk_work_items: wp.array[int],
        gjk_work_count: wp.array[int],
        manifold_work_items: wp.array[int],
        manifold_work_count: wp.array[int],
    ):
        tid = wp.tid()
        count = wp.min(cache.unresolved_work_items.shape[0], cache.unresolved_work_count[0])
        for work_index in range(tid, count, total_num_threads):
            pair_index = cache.unresolved_work_items[work_index]
            pair = candidate_pair[pair_index]
            valid, query = prepare_pair(pair, inputs)
            needs_gjk = False
            needs_manifold = False
            if valid:
                if wp.static(diagnostics):
                    wp.atomic_add(cache.stats, 7, 1)
                provider = SupportMapDataProvider()
                collision, point_a, point_b, normal, penetration = solve_mpr(
                    query.geom_a,
                    query.geom_b,
                    query.relative_orientation_b,
                    query.relative_position_b,
                    query.enlarge,
                    provider,
                )
                if collision:
                    signed_distance = -penetration + query.enlarge
                    if signed_distance <= query.contact_threshold:
                        half_enlarge = 0.5 * query.enlarge
                        result = ConvexQueryResult()
                        result.point_a = point_a - normal * half_enlarge
                        result.point_b = point_b + normal * half_enlarge
                        result.normal = normal
                        result.signed_distance = signed_distance
                        query_results[pair_index] = result
                        needs_manifold = True
                    _store_state(cache, cache.query_slot[pair_index], pair, inputs, 0, normal)
                else:
                    needs_gjk = True
            _append_work_index_compacted(needs_gjk, pair_index, gjk_work_items, gjk_work_count)
            _append_work_index_compacted(needs_manifold, pair_index, manifold_work_items, manifold_work_count)

    @wp.kernel(enable_backward=False, module="unique")
    def cold_gjk(
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        inputs: _PairInputs,
        cache: _CacheData,
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        gjk_work_items: wp.array[int],
        gjk_work_count: wp.array[int],
        manifold_work_items: wp.array[int],
        manifold_work_count: wp.array[int],
    ):
        tid = wp.tid()
        count = wp.min(gjk_work_items.shape[0], gjk_work_count[0])
        for work_index in range(tid, count, total_num_threads):
            pair_index = gjk_work_items[work_index]
            pair = candidate_pair[pair_index]
            valid, query = prepare_pair(pair, inputs)
            needs_manifold = False
            if valid:
                if wp.static(diagnostics):
                    wp.atomic_add(cache.stats, 6, 1)
                provider = SupportMapDataProvider()
                _separated, point_a, point_b, normal, signed_distance = solve_gjk(
                    query.geom_a,
                    query.geom_b,
                    query.relative_orientation_b,
                    query.relative_position_b,
                    0.0,
                    provider,
                )
                if signed_distance <= query.contact_threshold:
                    result = ConvexQueryResult()
                    result.point_a = point_a
                    result.point_b = point_b
                    result.normal = normal
                    result.signed_distance = signed_distance
                    query_results[pair_index] = result
                    needs_manifold = True
                mode = int(0)
                if signed_distance > 1.0e-4 and wp.isfinite(signed_distance) and _finite_vector(normal):
                    mode = 1
                _store_state(cache, cache.query_slot[pair_index], pair, inputs, mode, normal)
            _append_work_index_compacted(needs_manifold, pair_index, manifold_work_items, manifold_work_count)

    return classify_rejections, cold_mpr, cold_gjk

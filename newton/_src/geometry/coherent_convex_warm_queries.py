# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental split convex-query dispatch with current-geometry cache checks."""

import warp as wp

from .coherent_convex_warm import (
    _ROUND_GUARD,
    _CacheData,
    _CacheProvider,
    _create_cached_gjk,
    _find_slot,
    _finite_vector,
    _plane_lower_bound,
)
from .collision_convex import ConvexQueryResult
from .convex_bsp import _bind_shape, _BspData, _create_support
from .mpr import create_solve_mpr, create_support_map_function


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


def _create_query_kernels(*, diagnostics=False):
    # This factory is called after NarrowPhase has finished importing; avoid
    # a module-level dependency cycle with its shared preparation/queue helpers.
    from .narrow_phase import _append_work_index_compacted, create_prepare_convex_pair  # noqa: PLC0415

    prepare = create_prepare_convex_pair(True, False)
    bsp_support = _create_support(True)
    maps = create_support_map_function(bsp_support, use_precomputed_center=True)
    solve_mpr = create_solve_mpr(bsp_support, _support_funcs=maps).core
    solve_gjk = _create_cached_gjk()

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
    def classify_queries(
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
        bsp: _BspData,
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
            needs_warm = False
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
                        if mode == 1:
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
                        elif mode == 2:
                            needs_warm = True
                    needs_cold = cache.query_done[index] == 0 and not needs_warm
            _append_work_index_compacted(needs_warm, index, cache.warm_work_items, cache.warm_work_count)
            _append_work_index_compacted(needs_cold, index, cache.unresolved_work_items, cache.unresolved_work_count)

    @wp.kernel(enable_backward=False, module="unique")
    def warm_queries(
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
        bsp: _BspData,
    ):
        tid = wp.tid()
        count = wp.min(cache.warm_work_items.shape[0], cache.warm_work_count[0])
        for work_index in range(tid, count, total_num_threads):
            index = cache.warm_work_items[work_index]
            pair = candidate_pair[index]
            slot = cache.query_slot[index]
            needs_cold = True
            valid, query = prepare_pair(pair, inputs)
            if valid and slot >= 0:
                if wp.static(diagnostics):
                    wp.atomic_add(cache.stats, 3, 1)
                provider = _CacheProvider()
                provider.bsp = bsp
                provider.cache = cache
                provider.slot = slot
                provider.use_initial = 1
                _separated, a, b, normal, distance = solve_gjk(
                    _bind_shape(query.geom_a, bsp),
                    _bind_shape(query.geom_b, bsp),
                    query.relative_orientation_b,
                    query.relative_position_b,
                    0.0,
                    provider,
                    query.contact_threshold,
                    4,
                )
                if cache.mask[slot] != wp.uint32(0) and distance > 1.0e-4 and _finite_vector(a) and _finite_vector(b):
                    bound = _plane_lower_bound(
                        query.geom_a,
                        query.geom_b,
                        query.relative_orientation_b,
                        query.relative_position_b,
                        b - a,
                    )
                    upper = distance + _ROUND_GUARD * wp.max(bound.magnitude, 1.0e-12)
                    if bound.valid != 0 and bound.lower > query.contact_threshold:
                        cache.query_done[index] = 1
                        needs_cold = False
                        _store_state(cache, slot, pair, inputs, 1, normal)
                        if wp.static(diagnostics):
                            wp.atomic_add(cache.stats, 5, 1)
                    elif (
                        bound.valid != 0
                        and bound.lower > 0.0
                        and upper <= query.contact_threshold
                        and upper - bound.lower <= 1.0e-4
                    ):
                        result = ConvexQueryResult()
                        result.point_a = a
                        result.point_b = b
                        result.normal = normal
                        result.signed_distance = distance
                        query_results[index] = result
                        cache.query_done[index] = 2
                        needs_cold = False
                        _store_state(cache, slot, pair, inputs, 2, normal)
                        if wp.static(diagnostics):
                            wp.atomic_add(cache.stats, 4, 1)
            # The shell owner commits these results or sends them to cold MPR.
            # No warm contact is exposed to the original fitted manifold.
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
        bsp: _BspData,
    ):
        tid = wp.tid()
        count = wp.min(cache.unresolved_work_items.shape[0], cache.unresolved_work_count[0])
        for work_index in range(tid, count, total_num_threads):
            index = cache.unresolved_work_items[work_index]
            needs_gjk = False
            needs_manifold = False
            if cache.query_done[index] == 0:
                pair = candidate_pair[index]
                valid, query = prepare_pair(pair, inputs)
                if valid:
                    if wp.static(diagnostics):
                        wp.atomic_add(cache.stats, 7, 1)
                    collision, a, b, normal, penetration = solve_mpr(
                        _bind_shape(query.geom_a, bsp),
                        _bind_shape(query.geom_b, bsp),
                        query.relative_orientation_b,
                        query.relative_position_b,
                        query.enlarge,
                        bsp,
                    )
                    if collision:
                        distance = -penetration + query.enlarge
                        if distance <= query.contact_threshold:
                            result = ConvexQueryResult()
                            result.point_a = a - normal * (0.5 * query.enlarge)
                            result.point_b = b + normal * (0.5 * query.enlarge)
                            result.normal = normal
                            result.signed_distance = distance
                            query_results[index] = result
                            needs_manifold = True
                        _store_state(cache, cache.query_slot[index], pair, inputs, 0, normal)
                    else:
                        needs_gjk = True
            _append_work_index_compacted(needs_gjk, index, gjk_work_items, gjk_work_count)
            _append_work_index_compacted(needs_manifold, index, manifold_work_items, manifold_work_count)

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
        bsp: _BspData,
    ):
        tid = wp.tid()
        count = wp.min(gjk_work_items.shape[0], gjk_work_count[0])
        for work_index in range(tid, count, total_num_threads):
            index = gjk_work_items[work_index]
            pair = candidate_pair[index]
            valid, query = prepare_pair(pair, inputs)
            needs_manifold = False
            if valid:
                if wp.static(diagnostics):
                    wp.atomic_add(cache.stats, 6, 1)
                provider = _CacheProvider()
                provider.bsp = bsp
                provider.cache = cache
                provider.slot = cache.query_slot[index]
                provider.use_initial = 0
                _separated, a, b, normal, distance = solve_gjk(
                    _bind_shape(query.geom_a, bsp),
                    _bind_shape(query.geom_b, bsp),
                    query.relative_orientation_b,
                    query.relative_position_b,
                    0.0,
                    provider,
                )
                mode = int(0)
                if distance > 1.0e-4 and wp.isfinite(distance) and _finite_vector(normal):
                    mode = 1 if distance > query.contact_threshold else 2
                _store_state(cache, provider.slot, pair, inputs, mode, normal)
                if distance <= query.contact_threshold:
                    result = ConvexQueryResult()
                    result.point_a = a
                    result.point_b = b
                    result.normal = normal
                    result.signed_distance = distance
                    query_results[index] = result
                    needs_manifold = True
            _append_work_index_compacted(needs_manifold, index, manifold_work_items, manifold_work_count)

    return classify_queries, warm_queries, cold_mpr, cold_gjk

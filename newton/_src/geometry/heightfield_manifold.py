# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional adaptive four-witness terrain discretization at reducer export.

This changes the finite manifold, not the witnesses or the Coulomb law. The
input pool is exactly the old per-key survivor pool. Unsupported pairs retain
that complete pool. No geometric deduplication is performed across keys.
"""

from functools import cache
from typing import Any

import warp as wp

from .contact_data import SHAPE_PAIR_HFIELD_BIT, ContactData
from .contact_reduction import float_flip
from .contact_reduction_global import (
    PAIRS_PER_KEY,
    PREDICTIVE_BIN_ID,
    VALUES_PER_KEY,
    _roundoff_duplicate_bit_for_slot_pair,
    compute_effective_radius,
    make_contact_key,
    unpack_contact,
    unpack_contact_id,
)
from .hashtable import hashtable_find
from .types import GeoType

KEYS_PER_PAIR = PREDICTIVE_BIN_ID + 1
POOL_SIZE = KEYS_PER_PAIR * VALUES_PER_KEY
UINT32_MAX = wp.constant(wp.uint64(0xFFFFFFFF))


@wp.func
def _stock_writer_accepts(
    contact_id: int,
    position_depth: wp.array[wp.vec4],
    normal: wp.array[wp.vec2],
    shape_pairs: wp.array[wp.vec2i],
    shape_types: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_gap: wp.array[float],
) -> bool:
    # Preserve the actual stock writer's normalized endpoint arithmetic and
    # inclusive boundary, rather than substituting a raw-depth comparison.
    position, n, depth = unpack_contact(contact_id, position_depth, normal)
    pair = shape_pairs[contact_id]
    a = pair[0]
    b = pair[1]
    ra = compute_effective_radius(shape_types[a], shape_data[a])
    rb = compute_effective_radius(shape_types[b], shape_data[b])
    total = ra + rb + shape_data[a][3] + shape_data[b][3]
    n = wp.normalize(n)
    point_a = position - n * (0.5 * depth + ra)
    point_b = position + n * (0.5 * depth + rb)
    distance = wp.dot(point_b - point_a, n)
    return not (distance - total > shape_gap[a] + shape_gap[b])


@wp.func
def _selection_score(stage: int, p: wp.vec3, depth: float, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, n: wp.vec3) -> float:
    score = -depth
    if stage == 1:
        score = wp.length_sq(p - p0)
    elif stage == 2:
        score = wp.abs(wp.dot(p - p0, wp.cross(n, p0 - p1)))
    elif stage == 3:
        score = wp.abs(wp.dot(p - p0, wp.cross(n, p0 - p2))) + wp.abs(wp.dot(p1 - p, wp.cross(n, p1 - p2)))
    return score


@cache
def create_export_kernel(writer_func: Any):
    """Create the one replacement export; caller admits only the stock writer."""

    @wp.func
    def export_contact_id(
        contact_id: int,
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],
        shape_pairs: wp.array[wp.vec2i],
        contact_fingerprints: wp.array[wp.int32],
        exported_flags: wp.array[wp.int32],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_gap: wp.array[float],
        writer_data: Any,
    ):
        # Identical writer reconstruction and exact-ID dedup as old export.
        old_flag = wp.atomic_add(exported_flags, contact_id, 1)
        if old_flag > 0:
            return
        position, contact_normal, depth = unpack_contact(contact_id, position_depth, normal)
        pair = shape_pairs[contact_id]
        shape_a = pair[0]
        shape_b = pair[1]
        contact_data = ContactData()
        contact_data.contact_point_center = position
        contact_data.contact_normal_a_to_b = contact_normal
        contact_data.contact_distance = depth
        contact_data.radius_eff_a = compute_effective_radius(shape_types[shape_a], shape_data[shape_a])
        contact_data.radius_eff_b = compute_effective_radius(shape_types[shape_b], shape_data[shape_b])
        contact_data.margin_a = shape_data[shape_a][3]
        contact_data.margin_b = shape_data[shape_b][3]
        contact_data.shape_a = shape_a
        contact_data.shape_b = shape_b
        contact_data.gap_sum = shape_gap[shape_a] + shape_gap[shape_b]
        contact_data.sort_sub_key = contact_fingerprints[contact_id]
        writer_func(contact_data, writer_data, -1)

    @wp.kernel(enable_backward=False, module=f"heightfield_adaptive_manifold_{writer_func.__name__}")
    def export_adaptive_heightfield_manifold(
        ht_keys: wp.array[wp.uint64],
        ht_values: wp.array[wp.uint64],
        ht_active_slots: wp.array[wp.int32],
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],
        shape_pairs: wp.array[wp.vec2i],
        contact_fingerprints: wp.array[wp.int32],
        exported_flags: wp.array[wp.int32],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_gap: wp.array[float],
        writer_data: Any,
        total_num_blocks: int,
        parallel_pairs: int,
        deterministic: int,
        mesh_pairs: wp.array[wp.vec2i],
        mesh_count: wp.array[int],
        plane_pairs: wp.array[wp.vec2i],
        plane_count: wp.array[int],
        mesh_mesh_pairs: wp.array[wp.vec2i],
        mesh_mesh_count: wp.array[int],
    ):
        block, lane = wp.tid()
        width = int(1)
        if parallel_pairs != 0:
            width = 32
        count0 = int(0)
        count1 = int(0)
        count2 = int(0)
        if mesh_count.shape[0] != 0:
            count0 = wp.min(mesh_count[0], mesh_pairs.shape[0])
        if plane_count.shape[0] != 0:
            count1 = wp.min(plane_count[0], plane_pairs.shape[0])
        if mesh_mesh_count.shape[0] != 0:
            count2 = wp.min(mesh_mesh_count[0], mesh_mesh_pairs.shape[0])
        entries = wp.tile_zeros(shape=wp.static(KEYS_PER_PAIR), dtype=int, storage="shared")
        masks = wp.tile_zeros(shape=wp.static(KEYS_PER_PAIR), dtype=int, storage="shared")
        pool = wp.tile_zeros(shape=wp.static(POOL_SIZE), dtype=int, storage="shared")
        score_keys = wp.tile_zeros(shape=32, dtype=wp.uint64, storage="shared")

        for index in range(block, count0 + count1 + count2, total_num_blocks):
            pair = wp.vec2i(0)
            if index < count0:
                pair = mesh_pairs[index]
                # Match the mesh/convex midphase's oriented keys.
                if shape_types[pair[1]] == GeoType.MESH and shape_types[pair[0]] != GeoType.HFIELD:
                    pair = wp.vec2i(pair[1], pair[0])
            elif index < count0 + count1:
                pair = plane_pairs[index - count0]
            else:
                encoded = mesh_mesh_pairs[index - count0 - count1]
                pair = wp.vec2i(encoded[0] & ~SHAPE_PAIR_HFIELD_BIT, encoded[1])
            admitted = shape_types[pair[0]] == GeoType.HFIELD and (
                shape_types[pair[1]] == GeoType.BOX or shape_types[pair[1]] == GeoType.CONVEX_MESH
            )
            for key_round in range((wp.static(KEYS_PER_PAIR) + width - 1) // width):
                bin_id = key_round * width + lane
                entry = int(-1)
                mask = int(0)
                if bin_id < wp.static(KEYS_PER_PAIR):
                    entry = hashtable_find(make_contact_key(pair[0], pair[1], bin_id), ht_keys)
                    if entry >= 0:
                        for pair_idx in range(wp.static(PAIRS_PER_KEY)):
                            mask = mask | _roundoff_duplicate_bit_for_slot_pair(
                                pair_idx,
                                entry,
                                ht_keys.shape[0],
                                ht_values,
                                position_depth,
                                normal,
                                contact_fingerprints,
                                deterministic,
                            )
                wp.tile_scatter_masked(entries, bin_id, entry, bin_id < wp.static(KEYS_PER_PAIR))
                wp.tile_scatter_masked(masks, bin_id, mask, bin_id < wp.static(KEYS_PER_PAIR))
            # Each scatter is a tile synchronization; all keys are published
            # before any lane stages candidates (including bin 32..35).
            for slot_round in range((wp.static(POOL_SIZE) + width - 1) // width):
                slot = slot_round * width + lane
                bin_id = slot // wp.static(VALUES_PER_KEY)
                within = slot % wp.static(VALUES_PER_KEY)
                contact_id = int(0)
                if slot < wp.static(POOL_SIZE):
                    entry = entries[bin_id]
                    if entry >= 0 and masks[bin_id] & (1 << within) == 0:
                        value = ht_values[within * ht_keys.shape[0] + entry]
                        if value != wp.uint64(0):
                            contact_id = unpack_contact_id(value, deterministic)
                if admitted and contact_id != 0:
                    if not _stock_writer_accepts(
                        contact_id, position_depth, normal, shape_pairs, shape_types, shape_data, shape_gap
                    ):
                        contact_id = 0
                wp.tile_scatter_masked(pool, slot, contact_id, slot < wp.static(POOL_SIZE))

            if admitted:
                selected = wp.vec4i(0)
                p0 = wp.vec3(0.0)
                p1 = wp.vec3(0.0)
                p2 = wp.vec3(0.0)
                n0 = wp.vec3(0.0)
                for stage in range(4):
                    best = wp.uint64(0)
                    for slot in range(lane, wp.static(POOL_SIZE), width):
                        contact_id = pool[slot]
                        if (
                            contact_id == 0
                            or contact_id == selected[0]
                            or contact_id == selected[1]
                            or contact_id == selected[2]
                            or contact_id == selected[3]
                        ):
                            continue
                        p, n, depth = unpack_contact(contact_id, position_depth, normal)
                        score = _selection_score(stage, p, depth, p0, p1, p2, n0)
                        # Float-flip is monotone; low contact ID wins a tie.
                        key = (wp.uint64(float_flip(score)) << wp.uint64(32)) | (UINT32_MAX - wp.uint64(contact_id))
                        best = wp.max(best, key)
                    wp.tile_scatter_masked(score_keys, lane, best, True)
                    winner_key = wp.tile_reduce(wp.max, score_keys)[0]
                    winner = int(0)
                    if winner_key != wp.uint64(0):
                        winner = int(UINT32_MAX - (winner_key & UINT32_MAX))
                    selected[stage] = winner
                    if winner != 0:
                        p, n, depth = unpack_contact(winner, position_depth, normal)
                        if stage == 0:
                            p0 = p
                            n0 = n
                        elif stage == 1:
                            p1 = p
                        elif stage == 2:
                            p2 = p
                        # One lane serializes the four selected writer calls;
                        # duplicate pair blocks choose the same IDs independently.
                        if lane == 0:
                            export_contact_id(
                                winner,
                                position_depth,
                                normal,
                                shape_pairs,
                                contact_fingerprints,
                                exported_flags,
                                shape_types,
                                shape_data,
                                shape_gap,
                                writer_data,
                            )
            else:
                # Complete old survivor set, including distinct equivalent IDs
                # in different keys. The writer atomics deduplicate IDs only.
                for slot in range(lane, wp.static(POOL_SIZE), width):
                    contact_id = pool[slot]
                    if contact_id != 0:
                        export_contact_id(
                            contact_id,
                            position_depth,
                            normal,
                            shape_pairs,
                            contact_fingerprints,
                            exported_flags,
                            shape_types,
                            shape_data,
                            shape_gap,
                            writer_data,
                        )
            # Uniform fence before the CTA reuses its pair-local pool.
            wp.tile_scatter_masked(score_keys, lane, wp.uint64(0), True)

    return export_adaptive_heightfield_manifold

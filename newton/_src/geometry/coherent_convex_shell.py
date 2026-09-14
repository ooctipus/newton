# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete pre-write shell commit between warm query and original cold work."""

from typing import Any

import warp as wp

from .coherent_convex_warm import _CacheData
from .coherent_convex_warm_queries import _PairInputs
from .collision_convex import ConvexQueryResult
from .collision_core import post_process_minkowski_only
from .contact_data import ContactData
from .convex_shell import _mesh_index, _release, _shell_patch, _ShellData, _ShellResult, _storage


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
return wp::vec2i(threadIdx.x & 31,32);
#else
return wp::vec2i(0,1);
#endif
""")
def _lane() -> wp.vec2i: ...


def _create_shell_kernel(writer):
    from .narrow_phase import _append_work_index_compacted, create_prepare_convex_pair  # noqa: PLC0415

    prepare = create_prepare_convex_pair(True, False)

    @wp.func
    def contact_at(query: Any, normal: wp.vec3, point_a: wp.vec3, point_b: wp.vec3, index: int) -> ContactData:
        contact = ContactData()
        contact.radius_eff_a = query.radius_eff_a
        contact.radius_eff_b = query.radius_eff_b
        contact.margin_a = query.margin_a
        contact.margin_b = query.margin_b
        contact.shape_a = query.shape_a
        contact.shape_b = query.shape_b
        contact.gap_sum = query.rigid_gap
        contact.sort_sub_key = index
        contact.contact_normal_a_to_b = wp.quat_rotate(query.orientation_a, normal)
        contact.contact_point_center = wp.quat_rotate(query.orientation_a, 0.5 * (point_a + point_b)) + query.position_a
        contact.contact_distance = wp.dot(point_b - point_a, normal)
        return post_process_minkowski_only(
            contact,
            query.geom_a,
            query.position_a,
            query.orientation_a,
            query.geom_b,
            query.position_b,
            query.orientation_b,
        )

    @wp.func
    def passes(contact: ContactData) -> bool:
        # This is the actual simple writer's strict gap computation, not a
        # widened local-plane surrogate. The production writer accepts <=;
        # the strict subset is shared safely with either original writer.
        normal = wp.normalize(contact.contact_normal_a_to_b)
        a = contact.contact_point_center - normal * (0.5 * contact.contact_distance + contact.radius_eff_a)
        b = contact.contact_point_center + normal * (0.5 * contact.contact_distance + contact.radius_eff_b)
        total = contact.radius_eff_a + contact.radius_eff_b + contact.margin_a + contact.margin_b
        gap = wp.dot(b - a, normal) - total
        return wp.isfinite(gap) and gap < contact.gap_sum

    @wp.kernel(enable_backward=False, module="unique")
    def current_shell_commit(
        candidate_pairs: wp.array[wp.vec2i],
        inputs: _PairInputs,
        cache: _CacheData,
        workers: int,
        query_results: wp.array[ConvexQueryResult],
        shell_data: _ShellData,
        writer_data: Any,
    ):
        worker, logical = wp.tid()
        lane = _lane()
        if lane[1] == 1 and logical != 0:
            return
        address = _storage()
        count = wp.min(cache.warm_work_items.shape[0], cache.warm_work_count[0])
        for work in range(worker, count, workers):
            index = cache.warm_work_items[work]
            needs_cold = False
            if cache.query_done[index] == 2:
                pair = candidate_pairs[index]
                valid, query = prepare(
                    pair,
                    inputs.shape_types,
                    inputs.shape_data,
                    inputs.shape_transform,
                    inputs.shape_source,
                    inputs.shape_gap,
                    inputs.shape_collision_radius,
                    inputs.shape_aabb_lower,
                    inputs.shape_aabb_upper,
                    inputs.shape_collision_aabb_lower,
                    inputs.shape_collision_aabb_upper,
                )
                patch = _ShellResult()
                result = query_results[index]
                normal = wp.normalize(result.normal)
                if valid:
                    patch = _shell_patch(
                        address,
                        shell_data,
                        _mesh_index(shell_data, query.geom_a),
                        _mesh_index(shell_data, query.geom_b),
                        query.geom_a.scale,
                        query.geom_b.scale,
                        query.relative_orientation_b,
                        query.relative_position_b,
                        normal,
                        query.contact_threshold,
                        _ShellResult(),
                    )
                if lane[0] == 0:
                    accepted = patch.count >= 3 and patch.count <= 4
                    for k in range(patch.count):
                        contact = contact_at(query, normal, patch.point_a[k], patch.point_b[k], k)
                        accepted = accepted and passes(contact)
                    # The original feasible deepest witness is retained as the
                    # fifth point. No public maximum reservation is changed.
                    deepest = contact_at(query, normal, result.point_a, result.point_b, 4)
                    accepted = accepted and passes(deepest)
                    if accepted:
                        for k in range(patch.count):
                            contact = contact_at(query, normal, patch.point_a[k], patch.point_b[k], k)
                            writer(contact, writer_data, -1)
                        writer(deepest, writer_data, -1)
                        cache.query_done[index] = 3
                    else:
                        # Only failed warm results arrive here: they were not
                        # previously queued by warm_queries, so append once.
                        cache.query_done[index] = 0
                        needs_cold = True
            _append_work_index_compacted(needs_cold, index, cache.unresolved_work_items, cache.unresolved_work_count)
        _release(address)

    return current_shell_commit

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share MPR, compacted GJK and contact publication across terrain shape providers."""

from functools import cache

import warp as wp

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_convex import ConvexQueryResult, create_write_convex_query_result, gjk_separation_cutoff
from .collision_core import ENABLE_MULTI_CONTACT, post_process_axial_on_discrete_contact
from .contact_data import ContactData
from .contact_reduction_global import GlobalContactReducerData, write_contact_to_reducer
from .heightfield_features import HeightfieldFeatureContext, seam_surface_normal
from .mpr import create_solve_mpr, create_support_map_function
from .simplex_solver import create_solve_closest_distance
from .support_function import (
    GenericShapeData,
    create_triangle_prism_penetration_refiner,
    extract_shape_data,
)
from .types import GeoType


@wp.struct
class _TerrainQuery:
    geom_a: GenericShapeData
    geom_b: GenericShapeData
    position_a: wp.vec3
    position_b: wp.vec3
    orientation_a: wp.quat
    orientation_b: wp.quat
    relative_orientation_b: wp.quat
    relative_position_b: wp.vec3
    contact_template: ContactData
    contact_threshold: float
    enlarge: float


@cache
def create_heightfield_split_kernels(shape_type, support_func, data_type, *, precomputed_center=False):
    """Bind one terrain query schedule to shape, support and data providers.

    Preserve current triangle capacity, failed-MPR queue bounds and all reducer
    callbacks. Convex meshes retain the original nonzero AABB center computation;
    the precomputed-center option is used only by the centered primitive owner.
    """
    if shape_type not in (int(GeoType.CYLINDER), int(GeoType.CONVEX_MESH)):
        raise ValueError("Split terrain queries support regular cylinders or convex meshes")

    @wp.func
    def is_query(triple: wp.vec3i, shape_types: wp.array[int], shape_data: wp.array[wp.vec4]) -> bool:
        return (
            shape_types[triple[0]] == GeoType.HFIELD
            and shape_types[triple[1]] == wp.static(shape_type)
            and (wp.static(shape_type != int(GeoType.CYLINDER)) or shape_data[triple[1]][2] == 0.0)
        )

    @wp.func
    def prepare_query(
        triple: wp.vec3i,
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
    ) -> _TerrainQuery:
        shape_a = triple[0]
        shape_b = triple[1]
        query = _TerrainQuery()
        hfd = heightfield_data[shape_heightfield_index[shape_a]]
        geom_a, position_a = get_triangle_shape_from_heightfield(
            hfd, heightfield_elevations, shape_transform[shape_a], triple[2]
        )
        position_b, orientation_b, geom_b, _scale_b, margin_b = extract_shape_data(
            shape_b, shape_transform, shape_types, shape_data, shape_source
        )
        # Keep both facts from current-pair admission visible across array reloads.
        # These assignments only change the local descriptor, never shape storage.
        geom_b.shape_type = wp.static(shape_type)
        if wp.static(shape_type == int(GeoType.CYLINDER)):
            geom_b.scale[2] = 0.0
        query.geom_a = geom_a
        query.geom_b = geom_b
        query.position_a = position_a
        query.position_b = position_b
        query.orientation_b = orientation_b
        query.orientation_a = wp.transform_get_rotation(shape_transform[shape_a])
        query.relative_orientation_b = wp.quat_inverse(query.orientation_a) * query.orientation_b
        query.relative_position_b = wp.quat_rotate_inv(query.orientation_a, query.position_b - query.position_a)

        margin_a = shape_data[shape_a][3]
        gap_sum = shape_gap[shape_a] + shape_gap[shape_b]
        contact_template = ContactData()
        contact_template.margin_a = margin_a
        contact_template.margin_b = margin_b
        contact_template.shape_a = shape_a
        contact_template.shape_b = shape_b
        contact_template.gap_sum = gap_sum
        contact_template.sort_sub_key = (triple[2] << 1) | 1
        query.contact_template = contact_template
        query.contact_threshold = gap_sum + margin_a + margin_b

        margin_sum = margin_a + margin_b
        if margin_sum <= 0.0:
            query.enlarge = 1.0e-4
        elif margin_sum < 1.0e-4:
            query.enlarge = 2.0e-4
        else:
            query.enlarge = 0.0
        return query

    support_funcs = create_support_map_function(support_func, use_precomputed_center=precomputed_center)
    solve_mpr = create_solve_mpr(support_func, _support_funcs=support_funcs)
    solve_gjk = create_solve_closest_distance(support_func, _support_funcs=support_funcs)
    refine_penetration = create_triangle_prism_penetration_refiner(support_func)
    write_result = create_write_convex_query_result(
        support_func,
        write_contact_to_reducer,
        post_process_axial_on_discrete_contact,
        use_precomputed_center=precomputed_center,
    )

    def heightfield_cylinder_mpr_kernel(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
        reducer_data: GlobalContactReducerData,
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        gjk_indices: wp.array[int],
        gjk_count: wp.array[int],
        support_data: data_type,
    ):
        for i in range(wp.tid(), triangle_pairs_count[0], total_num_threads):
            if i >= triangle_pairs.shape[0]:
                break
            triple = triangle_pairs[i]
            if not is_query(triple, shape_types, shape_data):
                continue
            query = prepare_query(
                triple,
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_gap,
                shape_heightfield_index,
                heightfield_data,
                heightfield_elevations,
            )
            provider = support_data
            collision, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(
                query.geom_a,
                query.geom_b,
                query.relative_orientation_b,
                query.relative_position_b,
                query.enlarge,
                provider,
            )
            if collision:
                point_a, point_b, normal, penetration = wp.static(refine_penetration)(
                    query.geom_a,
                    query.geom_b,
                    query.relative_orientation_b,
                    query.relative_position_b,
                    query.enlarge,
                    provider,
                    point_a,
                    point_b,
                    normal,
                    penetration,
                )
                result = ConvexQueryResult()
                result.signed_distance = -penetration + query.enlarge
                half_enlarge = query.enlarge * 0.5
                result.point_a = point_a - normal * half_enlarge
                result.point_b = point_b + normal * half_enlarge
                result.normal = normal
                query_results[i] = result
            else:
                # At most one append per materialized triangle bounds this queue.
                queue_index = wp.atomic_add(gjk_count, 0, 1)
                if queue_index < gjk_indices.shape[0]:
                    gjk_indices[queue_index] = i

    def heightfield_cylinder_gjk_kernel(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
        reducer_data: GlobalContactReducerData,
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        gjk_indices: wp.array[int],
        gjk_count: wp.array[int],
        support_data: data_type,
    ):
        for queue_index in range(wp.tid(), wp.min(gjk_count[0], gjk_indices.shape[0]), total_num_threads):
            i = gjk_indices[queue_index]
            query = prepare_query(
                triangle_pairs[i],
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_gap,
                shape_heightfield_index,
                heightfield_data,
                heightfield_elevations,
            )
            provider = support_data
            _separated, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                query.geom_a,
                query.geom_b,
                query.relative_orientation_b,
                query.relative_position_b,
                0.0,
                provider,
                max_dist=gjk_separation_cutoff(query.contact_threshold, query.position_a, query.position_b),
            )
            result = ConvexQueryResult()
            result.point_a = point_a
            result.point_b = point_b
            result.normal = normal
            result.signed_distance = signed_distance
            query_results[i] = result

    def heightfield_cylinder_write_contacts_kernel(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
        reducer_data: GlobalContactReducerData,
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        gjk_indices: wp.array[int],
        gjk_count: wp.array[int],
        support_data: data_type,
    ):
        for i in range(wp.tid(), triangle_pairs_count[0], total_num_threads):
            if i >= triangle_pairs.shape[0]:
                break
            triple = triangle_pairs[i]
            if not is_query(triple, shape_types, shape_data):
                continue
            query = prepare_query(
                triple,
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_gap,
                shape_heightfield_index,
                heightfield_data,
                heightfield_elevations,
            )
            result = query_results[i]
            context = HeightfieldFeatureContext()
            context.elevations = heightfield_elevations
            context.triangle = triple[2]
            context.heightfield = heightfield_data[shape_heightfield_index[triple[0]]]
            if result.signed_distance >= 0.0:
                # The surface's own normal at this witness, and the distance along it.
                result.normal, result.signed_distance = seam_surface_normal(
                    query.geom_a, result.point_a, result.normal, result.signed_distance, context
                )

            # Preserve the shared single-witness callback. The raw writer screens
            # clearly separated contacts before allocating reducer storage.
            skip_multi_contact = result.signed_distance > query.contact_threshold
            if wp.static(not ENABLE_MULTI_CONTACT):
                skip_multi_contact = True
            provider = support_data
            wp.static(write_result)(
                query.geom_a,
                query.geom_b,
                query.orientation_a,
                query.orientation_b,
                query.position_a,
                query.position_b,
                query.relative_orientation_b,
                query.relative_position_b,
                result,
                provider,
                skip_multi_contact,
                reducer_data,
                query.contact_template,
            )

    kind = "cylinder" if shape_type == int(GeoType.CYLINDER) else "convex"
    kernels = (
        heightfield_cylinder_mpr_kernel,
        heightfield_cylinder_gjk_kernel,
        heightfield_cylinder_write_contacts_kernel,
    )
    for kernel in kernels:
        kernel.__name__ = kernel.__qualname__ = kernel.__name__.replace("cylinder", kind)
    return tuple(wp.kernel(enable_backward=False, module="unique")(kernel) for kernel in kernels)

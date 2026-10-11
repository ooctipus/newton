# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind immutable convex support candidates to the retained terrain contact law."""

from functools import cache

import warp as wp

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_core import create_compute_gjk_mpr_contacts
from .contact_reduction_global import GlobalContactReducerData, write_contact_to_reducer
from .convex_cells import _bind_shape, _CellData, _create_support
from .heightfield_features import WELD_FLAT_SEAMS, HeightfieldFeatureContext, flat_seam_query_allowed
from .support_function import GenericShapeData, create_triangle_prism_penetration_refiner, extract_shape_data
from .types import GeoType

_cell_support = _create_support(False)


@wp.func
def support_heightfield_cells(geom: GenericShapeData, direction: wp.vec3, data: _CellData) -> wp.vec3:
    """Retain original support for prisms and unadmitted or invalidated hulls."""
    return _cell_support(_bind_shape(geom, data), direction, data)


@cache
def get_kernel():
    compute = create_compute_gjk_mpr_contacts(
        write_contact_to_reducer,
        support_func=support_heightfield_cells,
        penetration_refiner=create_triangle_prism_penetration_refiner(support_heightfield_cells),
        query_filter=flat_seam_query_allowed if WELD_FLAT_SEAMS else None,
        external_support_data=True,
    )

    @wp.kernel(enable_backward=False, module="unique")
    def heightfield_convex_cell_contacts(
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
        support_data: _CellData,
    ):
        for index in range(wp.tid(), wp.min(triangle_pairs_count[0], triangle_pairs.shape[0]), total_num_threads):
            triple = triangle_pairs[index]
            shape_a, shape_b, triangle = triple[0], triple[1], triple[2]
            if shape_types[shape_a] != GeoType.HFIELD or shape_types[shape_b] != GeoType.CONVEX_MESH:
                continue
            heightfield = heightfield_data[shape_heightfield_index[shape_a]]
            transform_a = shape_transform[shape_a]
            geom_a, position_a = get_triangle_shape_from_heightfield(
                heightfield, heightfield_elevations, transform_a, triangle
            )
            position_b, orientation_b, geom_b, _scale_b, margin_b = extract_shape_data(
                shape_b, shape_transform, shape_types, shape_data, shape_source
            )
            geom_b.shape_type = wp.static(int(GeoType.CONVEX_MESH))
            context = HeightfieldFeatureContext()
            if wp.static(WELD_FLAT_SEAMS):
                context.elevations = heightfield_elevations
                context.triangle = triangle
                context.heightfield = heightfield
            compute(
                geom_a,
                geom_b,
                wp.transform_get_rotation(transform_a),
                orientation_b,
                position_a,
                position_b,
                shape_gap[shape_a] + shape_gap[shape_b],
                shape_a,
                shape_b,
                shape_data[shape_a][3],
                margin_b,
                reducer_data,
                (triangle << 1) | 1,
                context,
                support_data,
            )

    return heightfield_convex_cell_contacts

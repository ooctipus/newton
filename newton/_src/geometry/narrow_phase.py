# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import math
import os
from typing import Any, Literal

import warp as wp

from ..core.types import MAXVAL, Devicelike
from ..geometry.collision_convex import ConvexQueryResult, create_write_convex_query_result, gjk_separation_cutoff
from ..geometry.collision_core import (
    MINKOWSKI_CORE_RADIUS,
    _mesh_query_bound,
    _mesh_vs_convex_shape_data,
    check_infinite_plane_bsphere_overlap,
    compute_bounding_sphere_from_aabb,
    compute_tight_aabb_from_support,
    convert_infinite_plane_to_cube,
    convex_mesh_interior_point,
    create_compute_gjk_mpr_contacts,
    create_find_contacts,
    mesh_query_bound_slab,
    mesh_query_bound_unscaled_box,
    mesh_vs_convex_cell_hits,
    mesh_vs_convex_hit_candidates,
    mesh_vs_convex_local_box,
    mesh_vs_convex_midphase,
    mesh_vs_convex_pair_shapes,
    mesh_vs_convex_query_range,
    post_process_axial_on_discrete_contact,
    post_process_minkowski_only,
)
from ..geometry.collision_primitive import (
    _collide_capsule_cylinder_barrel,
    _collide_plane_capsule_contacts,
    collide_box_box_features,
    collide_capsule_capsule,
    collide_plane_box,
    collide_plane_cylinder,
    collide_plane_ellipsoid,
    collide_plane_sphere,
    collide_sphere_box,
    collide_sphere_capsule,
    collide_sphere_cylinder,
    collide_sphere_sphere,
)
from ..geometry.contact_data import (
    CONTACT_SORT_CONVEX_SUB_KEY_BITS,
    CONTACT_SORT_MAX_SHAPE_INDEX_BITS,
    CONTACT_SORT_SUB_KEY_BITS,
    SHAPE_PAIR_HFIELD_BIT,
    ContactData,
    _contact_passes_gap_check_precomputed,
    contact_passes_speculative_gap_check,
    contact_sort_shape_index_bits,
    make_contact_sort_key_with_bits,
    prepare_speculative_contact,
)
from ..geometry.contact_reduction_global import (
    EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
    EXPORT_REDUCED_CONTACTS_THREAD_BUDGET_MULTIPLIER,
    HASHTABLE_WARN_LOAD_PERCENT,
    MESH_ROUND_NONE,
    MESH_ROUND_ONLY,
    SURFACES_HEIGHTFIELD,
    SURFACES_MESH,
    GlobalContactReducer,
    create_export_reduced_contacts_kernel,
    create_mesh_face_contacts_kernel,
    create_mesh_triangle_contacts_to_reducer_kernel,
    mesh_triangle_contacts_to_reducer_kernel,
    reduce_buffered_contacts_kernel,
    reduce_buffered_contacts_speculative_kernel,
    write_contact_to_reducer,
)
from ..geometry.contact_sort import ContactSorter
from ..geometry.flags import ShapeFlags
from ..geometry.heightfield_cells import heightfield_cell_overlaps_kernel
from ..geometry.heightfield_cylinder import create_heightfield_cylinder_split_kernels
from ..geometry.heightfield_features import HeightfieldFeatureContext
from ..geometry.heightfield_primitive_contact import ANALYTIC_PRIMITIVE_QUERIES, analytic_primitives_default
from ..geometry.heightfield_row_gate import HEIGHTFIELD_ROW_COUNT, HEIGHTFIELD_TYPED_ROWS, heightfield_row_gate_kernel
from ..geometry.mesh_cells import MeshCells, build_mesh_cells, read_mesh_buffers
from ..geometry.mesh_topology import (
    MeshTopology,
    MeshTopologyData,
    create_mesh_capsule_contacts,
    create_mesh_sphere_contact,
    create_surface_penetration_refiner,
    empty_mesh_topology,
    mesh_contact_filter,
    mesh_deepest_once,
    mesh_manifold_triangle,
    mesh_surface_context,
    mesh_surface_planes,
    mesh_triangle_corners,
    surface_query_filter,
    triangle_plane_clears,
    triangle_shape_from_corners,
)
from ..geometry.mpr import create_solve_mpr, create_support_map_function
from ..geometry.sdf_contact import (
    MESH_SDF_BLOCK_DIM,
    compute_block_counts_from_weights,
    compute_mesh_mesh_block_offsets_scan,
    create_narrow_phase_process_mesh_mesh_contacts_kernel,
)
from ..geometry.sdf_hydroelastic import HydroelasticSDF
from ..geometry.sdf_texture import TextureSDFData
from ..geometry.simplex_solver import create_solve_closest_distance
from ..geometry.support_function import (
    AcceleratedSupportMapDataProvider,
    GenericShapeData,
    SupportMapDataProvider,
    extract_shape_data,
    support_map,
    support_map_accelerated,
    support_map_accelerated_inline,
    support_map_inline,
    support_map_lean,
    support_map_lean_accelerated,
)
from ..geometry.types import GeoType
from ..utils.heightfield import (
    HeightfieldData,
    get_triangle_shape_from_heightfield,
    heightfield_vs_convex_midphase,
)

_SPARSE_GJK_PAIR_CAPACITY_THRESHOLD = 1_000_000
# CUDA lane midphase: pairs with more cells than this move to the lane pass, which
# gives each of them this many lanes. Replayed on recorded rough-terrain pair
# lists, these values cut the warp iterations of the height test by 40-69% and the
# warp-level triangle appends by 43-70%, and bound a warp's in-place walk at 16 cells.
_LANE_SMALL_CELLS = 16
_LANE_WIDTH = 16
# Deferred lane-pass pairs stored when NarrowPhase gets no max_heightfield_pairs bound
# (288 bytes each with the geometric cull); pass 1 walks any further long pair in place.
_LANE_DEFAULT_CAPACITY = 1 << 16


@wp.func
def _reserve_compacted_slot(predicate: int, work_count: wp.array[int]) -> int:
    """Reserve one compacted output slot for a selected work item."""
    if predicate == 0:
        return -1
    return wp.atomic_add(work_count, 0, 1)


@wp.func
def _append_work_index_compacted(
    predicate: bool,
    value: int,
    work_items: wp.array[int],
    work_count: wp.array[int],
):
    """Append one selected work item to a compacted queue."""
    index = _reserve_compacted_slot(int(predicate), work_count)
    if index >= 0 and index < work_items.shape[0]:
        work_items[index] = value


@wp.func
def _append_pair_compacted(
    predicate: bool,
    value: wp.vec2i,
    work_items: wp.array[wp.vec2i],
    work_count: wp.array[int],
):
    """Append one selected pair to a compacted queue."""
    index = _reserve_compacted_slot(int(predicate), work_count)
    if index >= 0 and index < work_items.shape[0]:
        work_items[index] = value


@wp.struct
class ContactWriterData:
    contact_max: int
    shape_index_bits: int
    sub_key_bits: int
    contact_count: wp.array[int]
    contact_pair: wp.array[wp.vec2i]
    contact_position: wp.array[wp.vec3]
    contact_normal: wp.array[wp.vec3]
    contact_penetration: wp.array[float]
    contact_tangent: wp.array[wp.vec3]
    contact_sort_key: wp.array[wp.int64]
    shape_gap: wp.array[float]
    shape_transform: wp.array[wp.transform]
    shape_linear_velocity: wp.array[wp.vec3]
    shape_angular_velocity: wp.array[wp.vec3]
    collision_update_dt: float
    max_speculative_extension: float


@wp.struct
class ConvexPairQueryData:
    """Prepared shape-pair data shared by split MPR, GJK, and manifold passes."""

    shape_a: int
    shape_b: int
    type_a: int
    type_b: int
    geom_a: GenericShapeData
    geom_b: GenericShapeData
    position_a: wp.vec3
    position_b: wp.vec3
    orientation_a: wp.quat
    orientation_b: wp.quat
    relative_position_b: wp.vec3
    relative_orientation_b: wp.quat
    radius_eff_a: float
    radius_eff_b: float
    margin_a: float
    margin_b: float
    rigid_gap: float
    contact_threshold: float
    enlarge: float


def create_prepare_convex_pair(external_aabb: bool, speculative: bool = False):
    """Create pair preparation shared by the split convex-contact kernels."""

    @wp.func
    def prepare_convex_pair(
        pair: wp.vec2i,
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_collision_radius: wp.array[float],
        shape_aabb_lower: wp.array[wp.vec3],
        shape_aabb_upper: wp.array[wp.vec3],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
    ) -> tuple[bool, ConvexPairQueryData]:
        result = ConvexPairQueryData()
        shape_a = pair[0]
        shape_b = pair[1]
        if shape_a == shape_b or shape_a < 0 or shape_b < 0:
            return False, result

        type_a = shape_types[shape_a]
        type_b = shape_types[shape_b]

        pos_a, quat_a, geom_a, _scale_a, margin_a = extract_shape_data(
            shape_a, shape_transform, shape_types, shape_data, shape_source
        )
        pos_b, quat_b, geom_b, _scale_b, margin_b = extract_shape_data(
            shape_b, shape_transform, shape_types, shape_data, shape_source
        )
        if type_a == GeoType.CONVEX_MESH:
            geom_a.center = 0.5 * (shape_collision_aabb_lower[shape_a] + shape_collision_aabb_upper[shape_a])
            geom_a.bounds_half = 0.5 * (shape_collision_aabb_upper[shape_a] - shape_collision_aabb_lower[shape_a])
        if type_b == GeoType.CONVEX_MESH:
            geom_b.center = 0.5 * (shape_collision_aabb_lower[shape_b] + shape_collision_aabb_upper[shape_b])
            geom_b.bounds_half = 0.5 * (shape_collision_aabb_upper[shape_b] - shape_collision_aabb_lower[shape_b])

        is_infinite_plane_a = (type_a == GeoType.PLANE) and (geom_a.scale[0] == 0.0 and geom_a.scale[1] == 0.0)
        is_infinite_plane_b = (type_b == GeoType.PLANE) and (geom_b.scale[0] == 0.0 and geom_b.scale[1] == 0.0)
        if is_infinite_plane_a and is_infinite_plane_b:
            return False, result

        rigid_gap = shape_gap[shape_a] + shape_gap[shape_b]
        bsphere_radius_a = float(0.0)
        bsphere_radius_b = float(0.0)
        if is_infinite_plane_a or is_infinite_plane_b:
            if wp.static(external_aabb):
                aabb_a_lower = shape_aabb_lower[shape_a]
                aabb_a_upper = shape_aabb_upper[shape_a]
                aabb_b_lower = shape_aabb_lower[shape_b]
                aabb_b_upper = shape_aabb_upper[shape_b]
            else:
                gap_vec_a = wp.vec3(shape_gap[shape_a])
                gap_vec_b = wp.vec3(shape_gap[shape_b])
                if is_infinite_plane_a:
                    radius_a = shape_collision_radius[shape_a]
                    aabb_a_lower = pos_a - wp.vec3(radius_a) - gap_vec_a
                    aabb_a_upper = pos_a + wp.vec3(radius_a) + gap_vec_a
                else:
                    provider = SupportMapDataProvider()
                    aabb_a_lower, aabb_a_upper = compute_tight_aabb_from_support(geom_a, quat_a, pos_a, provider)
                    aabb_a_lower = aabb_a_lower - gap_vec_a
                    aabb_a_upper = aabb_a_upper + gap_vec_a
                if is_infinite_plane_b:
                    radius_b = shape_collision_radius[shape_b]
                    aabb_b_lower = pos_b - wp.vec3(radius_b) - gap_vec_b
                    aabb_b_upper = pos_b + wp.vec3(radius_b) + gap_vec_b
                else:
                    provider = SupportMapDataProvider()
                    aabb_b_lower, aabb_b_upper = compute_tight_aabb_from_support(geom_b, quat_b, pos_b, provider)
                    aabb_b_lower = aabb_b_lower - gap_vec_b
                    aabb_b_upper = aabb_b_upper + gap_vec_b

            center_a, bsphere_radius_a = compute_bounding_sphere_from_aabb(aabb_a_lower, aabb_a_upper)
            center_b, bsphere_radius_b = compute_bounding_sphere_from_aabb(aabb_b_lower, aabb_b_upper)
            overlap_radius_a = bsphere_radius_a
            overlap_radius_b = bsphere_radius_b
            if wp.static(external_aabb and speculative):
                pair_search_extension = rigid_gap
                if is_infinite_plane_a:
                    overlap_radius_b += pair_search_extension
                else:
                    overlap_radius_a += pair_search_extension
            if not check_infinite_plane_bsphere_overlap(
                geom_a,
                geom_b,
                pos_a,
                pos_b,
                quat_a,
                quat_b,
                center_a,
                center_b,
                overlap_radius_a,
                overlap_radius_b,
            ):
                return False, result

        else:
            # Bounding-sphere reject for non-plane pairs: AABB overlap admits many pairs whose hulls cannot come
            # within the gap (elongated finger links next to each other); GJK on those is wasted work.
            reject_radius = (
                shape_collision_radius[shape_a] + shape_collision_radius[shape_b] + rigid_gap + margin_a + margin_b
            )
            # Mesh radii enclose the scaled local AABB around its center,
            # which need not coincide with the imported shape's origin.
            center_a = pos_a
            center_b = pos_b
            if type_a == GeoType.CONVEX_MESH:
                center_a += wp.quat_rotate(quat_a, geom_a.center)
            if type_b == GeoType.CONVEX_MESH:
                center_b += wp.quat_rotate(quat_b, geom_b.center)
            if wp.length_sq(center_a - center_b) > reject_radius * reject_radius:
                return False, result
        if is_infinite_plane_a:
            geom_a, pos_a = convert_infinite_plane_to_cube(geom_a, quat_a, pos_a, pos_b, bsphere_radius_b + rigid_gap)
        if is_infinite_plane_b:
            geom_b, pos_b = convert_infinite_plane_to_cube(geom_b, quat_b, pos_b, pos_a, bsphere_radius_a + rigid_gap)

        radius_eff_a = float(0.0)
        radius_eff_b = float(0.0)
        if type_a == GeoType.SPHERE or type_a == GeoType.CAPSULE:
            radius_eff_a = geom_a.scale[0]
            geom_a.scale[0] = MINKOWSKI_CORE_RADIUS
        if type_b == GeoType.SPHERE or type_b == GeoType.CAPSULE:
            radius_eff_b = geom_b.scale[0]
            geom_b.scale[0] = MINKOWSKI_CORE_RADIUS

        margin_sum = margin_a + margin_b
        eps = 1.0e-4
        enlarge = float(0.0)
        if margin_sum <= 0.0:
            enlarge = eps
        elif margin_sum < eps:
            enlarge = 2.0 * eps

        result.shape_a = shape_a
        result.shape_b = shape_b
        result.type_a = type_a
        result.type_b = type_b
        result.geom_a = geom_a
        result.geom_b = geom_b
        result.position_a = pos_a
        result.position_b = pos_b
        result.orientation_a = quat_a
        result.orientation_b = quat_b
        result.relative_orientation_b = wp.quat_inverse(quat_a) * quat_b
        result.relative_position_b = wp.quat_rotate_inv(quat_a, pos_b - pos_a)
        result.radius_eff_a = radius_eff_a
        result.radius_eff_b = radius_eff_b
        result.margin_a = margin_a
        result.margin_b = margin_b
        result.rigid_gap = rigid_gap
        result.contact_threshold = rigid_gap + radius_eff_a + radius_eff_b + margin_sum
        result.enlarge = enlarge
        return True, result

    return prepare_convex_pair


@wp.func
def _write_contact_simple_at_index(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    index: int,
    normal_a_to_b: wp.vec3,
    separation: float,
):
    """Write an accepted contact through the simple narrow-phase API."""
    if index >= writer_data.contact_max:
        return

    writer_data.contact_pair[index] = wp.vec2i(contact_data.shape_a, contact_data.shape_b)
    writer_data.contact_position[index] = contact_data.contact_point_center
    writer_data.contact_normal[index] = normal_a_to_b
    writer_data.contact_penetration[index] = separation

    if writer_data.contact_tangent.shape[0] > 0:
        world_x = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(wp.dot(normal_a_to_b, world_x)) > 0.99:
            world_x = wp.vec3(0.0, 1.0, 0.0)
        writer_data.contact_tangent[index] = wp.normalize(world_x - wp.dot(world_x, normal_a_to_b) * normal_a_to_b)

    if writer_data.contact_sort_key.shape[0] > 0:
        writer_data.contact_sort_key[index] = make_contact_sort_key_with_bits(
            contact_data.shape_a,
            contact_data.shape_b,
            contact_data.sort_sub_key,
            writer_data.shape_index_bits,
            writer_data.sub_key_bits,
        )


@wp.func
def write_contact_simple(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    output_index: int,
):
    """
    Write a contact to the output arrays using the simplified API format.

    Args:
        contact_data: ContactData struct containing contact information
        writer_data: ContactWriterData struct containing output arrays
        output_index: If -1, use atomic_add to get the next available index if contact distance is less than gap_sum. If >= 0, use this index directly and skip gap check.
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )

    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    if output_index < 0:
        if d >= contact_data.gap_sum:
            return
        index = wp.atomic_add(writer_data.contact_count, 0, 1)
    else:
        index = output_index
    _write_contact_simple_at_index(contact_data, writer_data, index, contact_normal_a_to_b, d)


@wp.func
def _write_contact_simple_speculative(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    output_index: int,
):
    """Write a present or exactly predicted contact through the simple API."""
    contact_data.gap_sum = writer_data.shape_gap[contact_data.shape_a] + writer_data.shape_gap[contact_data.shape_b]
    normal, _point_a_world, _point_b_world, separation = prepare_speculative_contact(contact_data)

    index = output_index
    if index < 0:
        if not contact_passes_speculative_gap_check(
            contact_data,
            writer_data.shape_transform,
            writer_data.shape_linear_velocity,
            writer_data.shape_angular_velocity,
            writer_data.collision_update_dt,
            writer_data.max_speculative_extension,
        ):
            return
        index = wp.atomic_add(writer_data.contact_count, 0, 1)

    _write_contact_simple_at_index(contact_data, writer_data, index, normal, separation)


def create_narrow_phase_primitive_kernel(
    writer_func: Any,
    speculative: bool = False,
    sparse_gjk_pairs: bool = False,
    hydroelastic_enabled: bool = False,
    box_box_sat: bool = False,
):
    """
    Create a kernel for fast analytical collision detection of primitive shapes.

    This kernel handles lightweight primitive pairs (sphere-sphere, sphere-capsule,
    capsule-capsule, plane-sphere, plane-capsule) using direct analytical formulas
    instead of iterative GJK/MPR. Remaining pairs are routed to specialized buffers
    for mesh handling or to the GJK/MPR kernel for complex convex pairs.

    Args:
        writer_func: Contact writer function (e.g., write_contact_simple).
        speculative: Enable predictive contact admission.
        sparse_gjk_pairs: Preserve broad-phase pair indices in the GJK buffer.
        hydroelastic_enabled: Route hydroelastic pairs to the SDF-SDF pipeline.
        box_box_sat: Route box-box pairs through the SAT reference-face
            primitive instead of GJK/MPR (4-slot feature-identity manifold).

    Returns:
        A warp kernel for primitive collision detection
    """
    _module = f"narrow_phase_primitive_{writer_func.__name__}_{speculative}_{sparse_gjk_pairs}_{hydroelastic_enabled}_sat{int(box_box_sat)}"

    @wp.func(module=_module)
    def _check_unit_scale_sdf(
        shape_id: int,
        scale: wp.vec3,
        shape_unit_scale_sdf: wp.array[wp.int32],
        unit_scale_sdf_rescaled: wp.array[wp.int32],
    ):
        if shape_unit_scale_sdf.shape[0] > 0:
            if shape_unit_scale_sdf[shape_id] != 0 and (scale[0] != 1.0 or scale[1] != 1.0 or scale[2] != 1.0):
                if wp.atomic_max(unit_scale_sdf_rescaled, 0, 1) == 0:
                    wp.printf(
                        "Warning: shape_scale of shape %d changed, but its texture SDF is queried at unit scale. "
                        "Construct CollisionPipeline with dynamic_shape_scale=True, which is slower.\n",
                        shape_id,
                    )

    @wp.func(module=_module)
    def _admit(
        contact_data: ContactData,
        position: wp.vec3,
        distance: float,
        normalized_contact_normal: wp.vec3,
        total_separation_needed: float,
        shape_transform: wp.array[wp.transform],
        shape_linear_velocity: wp.array[wp.vec3],
        shape_angular_velocity: wp.array[wp.vec3],
        collision_update_dt: float,
        max_speculative_extension: float,
    ) -> bool:
        contact_data.contact_point_center = position
        contact_data.contact_distance = distance
        if wp.static(speculative):
            return contact_passes_speculative_gap_check(
                contact_data,
                shape_transform,
                shape_linear_velocity,
                shape_angular_velocity,
                collision_update_dt,
                max_speculative_extension,
            )
        return _contact_passes_gap_check_precomputed(
            contact_data,
            normalized_contact_normal,
            total_separation_needed,
        )

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_primitive_kernel(
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_linear_velocity: wp.array[wp.vec3],
        shape_angular_velocity: wp.array[wp.vec3],
        collision_update_dt: float,
        max_speculative_extension: float,
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_flags: wp.array[wp.int32],
        shape_sdf_index: wp.array[wp.int32],
        shape_edge_range: wp.array[wp.vec2i],
        shape_unit_scale_sdf: wp.array[wp.int32],
        terrain_law_partner: wp.array[wp.int32],
        writer_data: Any,
        total_num_threads: int,
        # Output: pairs that need GJK/MPR processing
        gjk_candidate_pairs: wp.array[wp.vec2i],
        gjk_candidate_pairs_count: wp.array[int],
        # Output: mesh collision pairs (for mesh processing)
        shape_pairs_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_count: wp.array[int],
        # Output: mesh-plane collision pairs
        shape_pairs_mesh_plane: wp.array[wp.vec2i],
        shape_pairs_mesh_plane_cumsum: wp.array[int],
        shape_pairs_mesh_plane_count: wp.array[int],
        mesh_plane_vertex_total_count: wp.array[int],
        # Output: mesh-mesh collision pairs
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_mesh_count: wp.array[int],
        # Output: sdf-sdf hydroelastic collision pairs
        shape_pairs_sdf_sdf: wp.array[wp.vec2i],
        shape_pairs_sdf_sdf_count: wp.array[int],
        unit_scale_sdf_rescaled: wp.array[wp.int32],
    ):
        """
        Fast narrow phase kernel for primitive shape collisions.

        Handles sphere-sphere, sphere-capsule, capsule-capsule, plane-sphere, and
        plane-capsule collisions analytically. Routes mesh pairs and complex convex
        pairs to specialized processing pipelines.
        """
        tid = wp.tid()

        num_work_items = wp.min(candidate_pair.shape[0], candidate_pair_count[0])

        # Early exit if no work
        if num_work_items == 0:
            return

        for t in range(tid, num_work_items, total_num_threads):
            # Get shape pair
            pair = candidate_pair[t]
            if wp.static(sparse_gjk_pairs) and t < gjk_candidate_pairs.shape[0]:
                gjk_candidate_pairs[t] = wp.vec2i(-1, -1)
            shape_a = pair[0]
            shape_b = pair[1]

            # Safety: ignore self-collision and invalid pairs
            if shape_a == shape_b or shape_a < 0 or shape_b < 0:
                continue

            # Get shape types
            type_a = shape_types[shape_a]
            type_b = shape_types[shape_b]

            # Sort shapes by type to ensure consistent collision handling order
            if type_a > type_b:
                shape_a, shape_b = shape_b, shape_a
                type_a, type_b = type_b, type_a

            # Check if both shapes are hydroelastic - route to SDF-SDF pipeline
            if wp.static(hydroelastic_enabled):
                is_hydro_a = (shape_flags[shape_a] & ShapeFlags.HYDROELASTIC) != 0
                is_hydro_b = (shape_flags[shape_b] & ShapeFlags.HYDROELASTIC) != 0
                if is_hydro_a and is_hydro_b and shape_pairs_sdf_sdf:
                    idx = wp.atomic_add(shape_pairs_sdf_sdf_count, 0, 1)
                    if idx < shape_pairs_sdf_sdf.shape[0]:
                        shape_pairs_sdf_sdf[idx] = wp.vec2i(shape_a, shape_b)
                    continue

            # Get shape data
            data_a = shape_data[shape_a]
            data_b = shape_data[shape_b]
            scale_a = wp.vec3(data_a[0], data_a[1], data_a[2])
            scale_b = wp.vec3(data_b[0], data_b[1], data_b[2])

            # =====================================================================
            # Route heightfield pairs.
            # Heightfield-vs-mesh and heightfield-vs-heightfield go through the
            # mesh-mesh SDF kernel (on-the-fly triangle + SDF evaluation).
            # Other heightfield combinations (convex, plane) use the dedicated
            # heightfield midphase with GJK/MPR per cell.
            # =====================================================================
            is_hfield_a = type_a == GeoType.HFIELD
            is_hfield_b = type_b == GeoType.HFIELD

            if is_hfield_a or is_hfield_b:
                is_mesh_like_a = type_a == GeoType.MESH or is_hfield_a
                is_mesh_like_b = type_b == GeoType.MESH or is_hfield_b

                if is_mesh_like_a and is_mesh_like_b:
                    # Heightfield-vs-heightfield is unsupported in this path.
                    if is_hfield_a and is_hfield_b:
                        continue
                    # Normalize order so heightfield (if present) is always pair[0],
                    # and mark pair[0] with a high-bit flag consumed by the SDF kernel.
                    if is_hfield_b:
                        encoded_a = shape_b | SHAPE_PAIR_HFIELD_BIT
                        encoded_b = shape_a
                    elif is_hfield_a:
                        encoded_a = shape_a | SHAPE_PAIR_HFIELD_BIT
                        encoded_b = shape_b
                    else:
                        encoded_a = shape_a
                        encoded_b = shape_b
                    idx = wp.atomic_add(shape_pairs_mesh_mesh_count, 0, 1)
                    if idx < shape_pairs_mesh_mesh.shape[0]:
                        shape_pairs_mesh_mesh[idx] = wp.vec2i(encoded_a, encoded_b)
                        _check_unit_scale_sdf(shape_a, scale_a, shape_unit_scale_sdf, unit_scale_sdf_rescaled)
                        _check_unit_scale_sdf(shape_b, scale_b, shape_unit_scale_sdf, unit_scale_sdf_rescaled)
                    continue

                # All other heightfield pairs: route through mesh midphase + GJK/MPR.
                # Normalize so the heightfield is always pair[0].
                if is_hfield_a:
                    hf_pair = wp.vec2i(shape_a, shape_b)
                else:
                    hf_pair = wp.vec2i(shape_b, shape_a)
                # The terrain law writes the contacts of its partners' heightfield pairs.
                if terrain_law_partner.shape[0] > 0:
                    if terrain_law_partner[hf_pair[1]] != 0:
                        continue
                idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)
                if idx < shape_pairs_mesh.shape[0]:
                    shape_pairs_mesh[idx] = hf_pair
                continue

            # =====================================================================
            # Route mesh pairs to specialized buffers
            # =====================================================================
            is_mesh_a = type_a == GeoType.MESH
            is_mesh_b = type_b == GeoType.MESH
            is_box_a = type_a == GeoType.BOX
            is_box_b = type_b == GeoType.BOX
            is_plane_a = type_a == GeoType.PLANE
            is_infinite_plane_a = is_plane_a and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
            has_sdf_edges_a = shape_sdf_index[shape_a] >= 0 and shape_edge_range[shape_a][1] > 0
            has_sdf_edges_b = shape_sdf_index[shape_b] >= 0 and shape_edge_range[shape_b][1] > 0

            # Existing mesh-mesh pairs keep their legacy SDF/BVH fallback
            # behavior. New planar SDF cases require both shapes to have
            # texture SDF data and edges; otherwise the old routing is cheaper.
            # Keep box-box on its existing GJK/MPR path even when SDFs are
            # present. The output buffer must exist for the SDF edge route.
            if (is_mesh_a and is_mesh_b) or (
                shape_pairs_mesh_mesh.shape[0] > 0
                and has_sdf_edges_a
                and has_sdf_edges_b
                and not (is_box_a and is_box_b)
            ):
                idx = wp.atomic_add(shape_pairs_mesh_mesh_count, 0, 1)
                if idx < shape_pairs_mesh_mesh.shape[0]:
                    shape_pairs_mesh_mesh[idx] = wp.vec2i(shape_a, shape_b)
                    _check_unit_scale_sdf(shape_a, scale_a, shape_unit_scale_sdf, unit_scale_sdf_rescaled)
                    _check_unit_scale_sdf(shape_b, scale_b, shape_unit_scale_sdf, unit_scale_sdf_rescaled)
                continue

            # Mesh-plane collision (infinite plane only)
            if is_infinite_plane_a and is_mesh_b:
                mesh_id = shape_source[shape_b]
                if mesh_id != wp.uint64(0):
                    mesh_obj = wp.mesh_get(mesh_id)
                    vertex_count = mesh_obj.points.shape[0]
                    mesh_plane_idx = wp.atomic_add(shape_pairs_mesh_plane_count, 0, 1)
                    if mesh_plane_idx < shape_pairs_mesh_plane.shape[0]:
                        # Store (mesh, plane)
                        shape_pairs_mesh_plane[mesh_plane_idx] = wp.vec2i(shape_b, shape_a)
                        cumulative_count_before = wp.atomic_add(mesh_plane_vertex_total_count, 0, vertex_count)
                        shape_pairs_mesh_plane_cumsum[mesh_plane_idx] = cumulative_count_before + vertex_count
                continue

            # Mesh-convex collision
            if is_mesh_a or is_mesh_b:
                idx = wp.atomic_add(shape_pairs_mesh_count, 0, 1)
                if idx < shape_pairs_mesh.shape[0]:
                    shape_pairs_mesh[idx] = wp.vec2i(shape_a, shape_b)
                continue

            # Route pairs without an analytical path before allocating contact
            # storage and walking the primitive dispatch chain. SAT box-box is
            # analytic: exempt it here or it never reaches its dispatch branch.
            if (
                (type_a >= GeoType.ELLIPSOID and not (wp.static(box_box_sat) and is_box_a and is_box_b))
                or type_b == GeoType.CONE
                or (type_a == GeoType.CAPSULE and type_b > GeoType.CAPSULE and type_b != GeoType.CYLINDER)
            ):
                if wp.static(sparse_gjk_pairs):
                    wp.atomic_add(gjk_candidate_pairs_count, 0, 1)
                    if t < gjk_candidate_pairs.shape[0]:
                        gjk_candidate_pairs[t] = wp.vec2i(shape_a, shape_b)
                else:
                    _append_pair_compacted(
                        True, wp.vec2i(shape_a, shape_b), gjk_candidate_pairs, gjk_candidate_pairs_count
                    )
                continue

            margin_offset_a = data_a[3]
            margin_offset_b = data_b[3]

            # Get transforms
            X_a = shape_transform[shape_a]
            X_b = shape_transform[shape_b]
            pos_a = wp.transform_get_translation(X_a)
            pos_b = wp.transform_get_translation(X_b)
            quat_a = wp.transform_get_rotation(X_a)
            quat_b = wp.transform_get_rotation(X_b)
            gap_a = shape_gap[shape_a]
            gap_b = shape_gap[shape_b]
            gap_sum = gap_a + gap_b

            # =====================================================================
            # Handle lightweight primitives analytically
            # =====================================================================
            is_sphere_a = type_a == GeoType.SPHERE
            is_sphere_b = type_b == GeoType.SPHERE
            is_capsule_a = type_a == GeoType.CAPSULE
            is_capsule_b = type_b == GeoType.CAPSULE
            is_ellipsoid_b = type_b == GeoType.ELLIPSOID
            is_cylinder_b = type_b == GeoType.CYLINDER
            is_box_b = type_b == GeoType.BOX

            use_plane_cylinder = is_plane_a and is_cylinder_b
            if use_plane_cylinder and scale_b[2] > 0.0:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                cylinder_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                use_plane_cylinder = wp.abs(wp.dot(plane_normal, cylinder_axis)) * scale_b[2] >= scale_b[1]

            # Compute effective radii for spheres and capsules
            # (radius that can be represented as Minkowski sum with a sphere)
            radius_eff_a = float(0.0)
            radius_eff_b = float(0.0)
            if is_sphere_a or is_capsule_a:
                radius_eff_a = scale_a[0]
            if is_sphere_b or is_capsule_b:
                radius_eff_b = scale_b[0]

            # Initialize contact result storage (up to 4 contacts).
            # Distances default to MAXVAL so unused slots are automatically
            # excluded by the unified num_contacts count after the if/elif chain.
            contact_dist_0 = float(MAXVAL)
            contact_dist_1 = float(MAXVAL)
            contact_dist_2 = float(MAXVAL)
            contact_dist_3 = float(MAXVAL)
            contact_pos_0 = wp.vec3()
            contact_pos_1 = wp.vec3()
            contact_pos_2 = wp.vec3()
            contact_pos_3 = wp.vec3()
            contact_normal = wp.vec3()

            # -----------------------------------------------------------------
            # Plane-Sphere collision (type_a=PLANE=0, type_b=SPHERE=2)
            # -----------------------------------------------------------------
            if is_plane_a and is_sphere_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                sphere_radius = scale_b[0]
                contact_dist_0, contact_pos_0 = collide_plane_sphere(plane_normal, pos_a, pos_b, sphere_radius)
                contact_normal = plane_normal

            # -----------------------------------------------------------------
            # Plane-Ellipsoid collision (type_a=PLANE=0, type_b=ELLIPSOID=4)
            # Produces 1 contact
            # -----------------------------------------------------------------
            elif is_plane_a and is_ellipsoid_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                ellipsoid_rot = wp.quat_to_matrix(quat_b)
                ellipsoid_size = scale_b
                contact_dist_0, contact_pos_0, contact_normal = collide_plane_ellipsoid(
                    plane_normal, pos_a, pos_b, ellipsoid_rot, ellipsoid_size
                )

            # -----------------------------------------------------------------
            # Plane-Box collision (type_a=PLANE=0, type_b=BOX=6)
            # Produces up to 4 contacts
            # -----------------------------------------------------------------
            elif is_plane_a and is_box_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                box_rot = wp.quat_to_matrix(quat_b)
                box_size = scale_b

                dists4_box, positions4_box, contact_normal = collide_plane_box(
                    plane_normal,
                    pos_a,
                    pos_b,
                    box_rot,
                    box_size,
                    gap_sum + margin_offset_a + margin_offset_b,
                )

                contact_dist_0 = dists4_box[0]
                contact_dist_1 = dists4_box[1]
                contact_dist_2 = dists4_box[2]
                contact_dist_3 = dists4_box[3]
                contact_pos_0 = wp.vec3(positions4_box[0, 0], positions4_box[0, 1], positions4_box[0, 2])
                contact_pos_1 = wp.vec3(positions4_box[1, 0], positions4_box[1, 1], positions4_box[1, 2])
                contact_pos_2 = wp.vec3(positions4_box[2, 0], positions4_box[2, 1], positions4_box[2, 2])
                contact_pos_3 = wp.vec3(positions4_box[3, 0], positions4_box[3, 1], positions4_box[3, 2])

            # -----------------------------------------------------------------
            # Box-Box collision via SAT reference-face clipping (opt-in).
            # The manifold-quality fix for GJK/MPR witness-point teleports
            # (measured up to ~a face width per frame under micro-wobble).
            # -----------------------------------------------------------------
            elif wp.static(box_box_sat) and is_box_a and is_box_b:
                box_rot_a = wp.quat_to_matrix(quat_a)
                box_rot_b = wp.quat_to_matrix(quat_b)
                # SAT clips the raw box surfaces, while admission subtracts
                # the per-shape collision margins. Include those margins in
                # candidate generation so an inflated contact is not rejected
                # before the shared admission gate can see it.
                sat_margin = gap_sum + margin_offset_a + margin_offset_b
                if wp.static(speculative):
                    # Candidates must exist beyond the static gap for
                    # predictive admission to see approaching boxes.
                    sat_margin += max_speculative_extension
                sat_dist, sat_pos, sat_normals, sat_feat = collide_box_box_features(
                    pos_a,
                    box_rot_a,
                    wp.vec3(scale_a[0], scale_a[1], scale_a[2]),
                    pos_b,
                    box_rot_b,
                    wp.vec3(scale_b[0], scale_b[1], scale_b[2]),
                    sat_margin,
                )
                # Reduce to at most 4 by feature persistence: incident-face
                # corners are the durable support points of a resting face
                # (family 2), then reference corners (1), then edge clips (0),
                # deepest first within a family. The generation-time feature
                # id -- not the output slot or any center binning -- becomes
                # sort_sub_key, so a surviving contact keeps its per-pair
                # identity across frames however the raw manifold compacts.
                # Dedup tolerance scaled by the smaller box so millimeter
                # parts keep distinct manifold points; duplicates from
                # different clip lines are (near-)exactly coincident.
                min_ext_a = wp.min(scale_a[0], wp.min(scale_a[1], scale_a[2]))
                min_ext_b = wp.min(scale_b[0], wp.min(scale_b[1], scale_b[2]))
                min_ext = wp.min(min_ext_a, min_ext_b)
                dedup_eps_sq = (1.0e-3 * min_ext) * (1.0e-3 * min_ext)
                chosen = wp.vec4i(-1, -1, -1, -1)
                for slot4 in range(4):
                    best = int(-1)
                    best_key = float(-3.4e38)
                    for ci in range(8):
                        if sat_dist[ci] >= MAXVAL:
                            continue
                        skip = False
                        for c4 in range(4):
                            cj = chosen[c4]
                            if cj == ci:
                                skip = True
                            elif cj >= 0:
                                # Aligned boxes emit coincident duplicates
                                # from different clip lines; a duplicated
                                # corner plus a missing one is a degenerate
                                # support under load.
                                dx = sat_pos[ci, 0] - sat_pos[cj, 0]
                                dy = sat_pos[ci, 1] - sat_pos[cj, 1]
                                dz = sat_pos[ci, 2] - sat_pos[cj, 2]
                                if dx * dx + dy * dy + dz * dz < dedup_eps_sq:
                                    skip = True
                        if skip:
                            continue
                        key = float((sat_feat[ci] >> 4) & 3) * 1.0e3 - sat_dist[ci]
                        if key > best_key:
                            best_key = key
                            best = ci
                    chosen[slot4] = best
                bb_data = ContactData()
                bb_data.margin_a = margin_offset_a
                bb_data.margin_b = margin_offset_b
                bb_data.shape_a = shape_a
                bb_data.shape_b = shape_b
                bb_data.gap_sum = gap_sum
                num_valid = int(0)
                for c4 in range(4):
                    ci = chosen[c4]
                    if ci >= 0:
                        n_ci = wp.vec3(sat_normals[ci, 0], sat_normals[ci, 1], sat_normals[ci, 2])
                        bb_data.contact_normal_a_to_b = n_ci
                        if _admit(
                            bb_data,
                            wp.vec3(sat_pos[ci, 0], sat_pos[ci, 1], sat_pos[ci, 2]),
                            sat_dist[ci],
                            n_ci,
                            margin_offset_a + margin_offset_b,
                            shape_transform,
                            shape_linear_velocity,
                            shape_angular_velocity,
                            collision_update_dt,
                            max_speculative_extension,
                        ):
                            num_valid += 1
                        else:
                            chosen[c4] = -1
                if num_valid > 0:
                    bb_base = wp.atomic_add(writer_data.contact_count, 0, num_valid)
                    if bb_base + num_valid <= writer_data.contact_max:
                        for c4 in range(4):
                            ci = chosen[c4]
                            if ci >= 0:
                                bb_data.contact_point_center = wp.vec3(sat_pos[ci, 0], sat_pos[ci, 1], sat_pos[ci, 2])
                                bb_data.contact_distance = sat_dist[ci]
                                bb_data.contact_normal_a_to_b = wp.vec3(
                                    sat_normals[ci, 0], sat_normals[ci, 1], sat_normals[ci, 2]
                                )
                                bb_data.sort_sub_key = sat_feat[ci]
                                writer_func(bb_data, writer_data, bb_base)
                                bb_base += 1
                continue

            # -----------------------------------------------------------------
            # Sphere-Sphere collision (type_a=SPHERE=2, type_b=SPHERE=2)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_sphere_b:
                radius_a = scale_a[0]
                radius_b = scale_b[0]
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_sphere(pos_a, radius_a, pos_b, radius_b)

            # -----------------------------------------------------------------
            # Plane-Capsule collision (type_a=PLANE=0, type_b=CAPSULE=3)
            # Produces 2 contacts (both share same normal)
            # -----------------------------------------------------------------
            elif is_plane_a and is_capsule_b:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                capsule_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                capsule_radius = scale_b[0]
                capsule_half_length = scale_b[1]

                contact_dist_0, contact_pos_0, contact_dist_1, contact_pos_1 = _collide_plane_capsule_contacts(
                    plane_normal,
                    pos_a,
                    pos_b,
                    capsule_axis,
                    capsule_radius,
                    capsule_half_length,
                )

                contact_normal = plane_normal

            # -----------------------------------------------------------------
            # Plane-Cylinder collision (type_a=PLANE=0, type_b=CYLINDER=5)
            # Produces up to 4 contacts
            # -----------------------------------------------------------------
            elif use_plane_cylinder:
                plane_normal = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                cylinder_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                cylinder_radius = scale_b[0]
                cylinder_half_height = scale_b[1]

                dists4, positions4, contact_normal = collide_plane_cylinder(
                    plane_normal, pos_a, pos_b, cylinder_axis, cylinder_radius, cylinder_half_height
                )

                contact_dist_0 = dists4[0]
                contact_dist_1 = dists4[1]
                contact_dist_2 = dists4[2]
                contact_dist_3 = dists4[3]
                contact_pos_0 = wp.vec3(positions4[0, 0], positions4[0, 1], positions4[0, 2])
                contact_pos_1 = wp.vec3(positions4[1, 0], positions4[1, 1], positions4[1, 2])
                contact_pos_2 = wp.vec3(positions4[2, 0], positions4[2, 1], positions4[2, 2])
                contact_pos_3 = wp.vec3(positions4[3, 0], positions4[3, 1], positions4[3, 2])

            # -----------------------------------------------------------------
            # Sphere-Capsule collision (type_a=SPHERE=2, type_b=CAPSULE=3)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_capsule_b:
                sphere_radius = scale_a[0]
                capsule_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                capsule_radius = scale_b[0]
                capsule_half_length = scale_b[1]
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_capsule(
                    pos_a, sphere_radius, pos_b, capsule_axis, capsule_radius, capsule_half_length
                )

            # -----------------------------------------------------------------
            # Capsule-Capsule collision (type_a=CAPSULE=3, type_b=CAPSULE=3)
            # Produces 1 contact (non-parallel) or 2 contacts (parallel axes)
            # -----------------------------------------------------------------
            elif is_capsule_a and is_capsule_b:
                axis_a = wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0))
                axis_b = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                radius_a = scale_a[0]
                half_length_a = scale_a[1]
                radius_b = scale_b[0]
                half_length_b = scale_b[1]

                dists, positions, contact_normal = collide_capsule_capsule(
                    pos_a, axis_a, radius_a, half_length_a, pos_b, axis_b, radius_b, half_length_b
                )

                contact_dist_0 = dists[0]
                contact_pos_0 = wp.vec3(positions[0, 0], positions[0, 1], positions[0, 2])
                contact_dist_1 = dists[1]
                contact_pos_1 = wp.vec3(positions[1, 0], positions[1, 1], positions[1, 2])

            # -----------------------------------------------------------------
            # Sphere-Cylinder collision (type_a=SPHERE=2, type_b=CYLINDER=5)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_cylinder_b and scale_b[2] == 0.0:
                sphere_radius = scale_a[0]
                cylinder_axis = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0))
                cylinder_radius = scale_b[0]
                cylinder_half_height = scale_b[1]
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_cylinder(
                    pos_a, sphere_radius, pos_b, cylinder_axis, cylinder_radius, cylinder_half_height
                )

            # -----------------------------------------------------------------
            # Capsule-Cylinder collision: only exterior barrel witnesses.
            # Unsupported cases leave contact_dist_0 unset and reach MPR/GJK.
            # -----------------------------------------------------------------
            elif is_capsule_a and is_cylinder_b and scale_b[2] == 0.0:
                barrel_handled, barrel_dist, barrel_pos, barrel_normal = _collide_capsule_cylinder_barrel(
                    pos_a,
                    wp.quat_rotate(quat_a, wp.vec3(0.0, 0.0, 1.0)),
                    scale_a[0],
                    scale_a[1],
                    pos_b,
                    wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, 1.0)),
                    scale_b[0],
                    scale_b[1],
                )
                if barrel_handled:
                    contact_dist_0 = barrel_dist
                    contact_pos_0 = barrel_pos
                    contact_normal = barrel_normal

            # -----------------------------------------------------------------
            # Sphere-Box collision (type_a=SPHERE=2, type_b=BOX=6)
            # -----------------------------------------------------------------
            elif is_sphere_a and is_box_b:
                sphere_radius = scale_a[0]
                box_rot = wp.quat_to_matrix(quat_b)
                box_size = scale_b
                contact_dist_0, contact_pos_0, contact_normal = collide_sphere_box(
                    pos_a, sphere_radius, pos_b, box_rot, box_size
                )

            # =====================================================================
            # Write all contacts (single write block for 0 to 4 contacts)
            # =====================================================================
            num_contacts = (
                int(contact_dist_0 < MAXVAL)
                + int(contact_dist_1 < MAXVAL)
                + int(contact_dist_2 < MAXVAL)
                + int(contact_dist_3 < MAXVAL)
            )
            if num_contacts > 0:
                # Prepare contact data (shared fields for both contacts)
                contact_data = ContactData()
                contact_data.contact_normal_a_to_b = contact_normal
                contact_data.radius_eff_a = radius_eff_a
                contact_data.radius_eff_b = radius_eff_b
                contact_data.margin_a = margin_offset_a
                contact_data.margin_b = margin_offset_b
                contact_data.shape_a = shape_a
                contact_data.shape_b = shape_b
                if wp.static(speculative):
                    contact_data.gap_sum = writer_data.shape_gap[shape_a] + writer_data.shape_gap[shape_b]
                else:
                    contact_data.gap_sum = gap_sum

                total_separation_needed = radius_eff_a + radius_eff_b + margin_offset_a + margin_offset_b
                normalized_contact_normal = wp.normalize(contact_normal)

                # Check margin for all possible contacts
                contact_0_valid = False
                if contact_dist_0 < MAXVAL:
                    contact_0_valid = _admit(
                        contact_data,
                        contact_pos_0,
                        contact_dist_0,
                        normalized_contact_normal,
                        total_separation_needed,
                        shape_transform,
                        shape_linear_velocity,
                        shape_angular_velocity,
                        collision_update_dt,
                        max_speculative_extension,
                    )

                contact_1_valid = False
                if contact_dist_1 < MAXVAL:
                    contact_1_valid = _admit(
                        contact_data,
                        contact_pos_1,
                        contact_dist_1,
                        normalized_contact_normal,
                        total_separation_needed,
                        shape_transform,
                        shape_linear_velocity,
                        shape_angular_velocity,
                        collision_update_dt,
                        max_speculative_extension,
                    )

                contact_2_valid = False
                if contact_dist_2 < MAXVAL:
                    contact_2_valid = _admit(
                        contact_data,
                        contact_pos_2,
                        contact_dist_2,
                        normalized_contact_normal,
                        total_separation_needed,
                        shape_transform,
                        shape_linear_velocity,
                        shape_angular_velocity,
                        collision_update_dt,
                        max_speculative_extension,
                    )

                contact_3_valid = False
                if contact_dist_3 < MAXVAL:
                    contact_3_valid = _admit(
                        contact_data,
                        contact_pos_3,
                        contact_dist_3,
                        normalized_contact_normal,
                        total_separation_needed,
                        shape_transform,
                        shape_linear_velocity,
                        shape_angular_velocity,
                        collision_update_dt,
                        max_speculative_extension,
                    )

                # Count valid contacts and allocate consecutive indices
                num_valid = int(contact_0_valid) + int(contact_1_valid) + int(contact_2_valid) + int(contact_3_valid)
                if num_valid > 0:
                    base_index = wp.atomic_add(writer_data.contact_count, 0, num_valid)
                    # Do not invoke the writer callback for overflowing batches.
                    # This keeps user-provided writers safe while still preserving
                    # overflow visibility via contact_count > contact_max.
                    if base_index + num_valid > writer_data.contact_max:
                        continue

                    # Write first contact if valid
                    if contact_0_valid:
                        contact_data.contact_point_center = contact_pos_0
                        contact_data.contact_distance = contact_dist_0
                        contact_data.sort_sub_key = 0
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write second contact if valid
                    if contact_1_valid:
                        contact_data.contact_point_center = contact_pos_1
                        contact_data.contact_distance = contact_dist_1
                        contact_data.sort_sub_key = 1
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write third contact if valid
                    if contact_2_valid:
                        contact_data.contact_point_center = contact_pos_2
                        contact_data.contact_distance = contact_dist_2
                        contact_data.sort_sub_key = 2
                        writer_func(contact_data, writer_data, base_index)
                        base_index += 1

                    # Write fourth contact if valid
                    if contact_3_valid:
                        contact_data.contact_point_center = contact_pos_3
                        contact_data.contact_distance = contact_dist_3
                        contact_data.sort_sub_key = 3
                        writer_func(contact_data, writer_data, base_index)

                continue

            if (
                (is_plane_a and (is_sphere_b or is_capsule_b or is_ellipsoid_b or use_plane_cylinder or is_box_b))
                or (is_sphere_a and (is_sphere_b or is_capsule_b or (is_cylinder_b and scale_b[2] == 0.0) or is_box_b))
                or (is_capsule_a and is_capsule_b)
            ):
                continue

            # =====================================================================
            # Route remaining pairs to GJK/MPR kernel
            # =====================================================================
            if wp.static(sparse_gjk_pairs):
                wp.atomic_add(gjk_candidate_pairs_count, 0, 1)
                if t < gjk_candidate_pairs.shape[0]:
                    gjk_candidate_pairs[t] = wp.vec2i(shape_a, shape_b)
            else:
                _append_pair_compacted(True, wp.vec2i(shape_a, shape_b), gjk_candidate_pairs, gjk_candidate_pairs_count)

    return narrow_phase_primitive_kernel


def create_narrow_phase_kernel_gjk_mpr(
    external_aabb: bool,
    writer_func: Any,
    support_func: Any = None,
    post_process_contact: Any = None,
    speculative: bool = False,
):
    """
    Create a GJK/MPR narrow phase kernel for complex convex shape collisions.

    This kernel is called AFTER the primitive kernel has already:
    - Sorted pairs by type (type_a <= type_b)
    - Routed mesh pairs to specialized buffers
    - Routed hydroelastic pairs to SDF-SDF buffer
    - Handled primitive collisions analytically

    The remaining pairs are complex convex-convex (plane-box, plane-cylinder,
    plane-cone, box-box, cylinder-cylinder, etc.) that need GJK/MPR.
    """
    _sf = support_func.__name__ if support_func is not None else "default"
    _ppc = post_process_contact.__name__ if post_process_contact is not None else "default"
    _module = f"narrow_phase_gjk_mpr_{external_aabb}_{speculative}_{writer_func.__name__}_{_sf}_{_ppc}"
    accelerated_support = support_func in (support_map_accelerated, support_map_lean_accelerated)
    provider_type = AcceleratedSupportMapDataProvider if accelerated_support else SupportMapDataProvider

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_kernel_gjk_mpr(
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_support_data: wp.array[wp.vec4i],
        support_lut: wp.array[int],
        support_vertex_offsets: wp.array[int],
        support_neighbors: wp.array[int],
        shape_gap: wp.array[float],
        shape_collision_radius: wp.array[float],
        shape_aabb_lower: wp.array[wp.vec3],
        shape_aabb_upper: wp.array[wp.vec3],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        writer_data: Any,
        total_num_threads: int,
    ):
        """
        GJK/MPR collision detection for complex convex pairs.

        Pairs are pre-filtered (no meshes, no hydroelastic, no simple
        primitives) and pre-sorted by shape type.
        """
        tid = wp.tid()

        num_work_items = wp.min(candidate_pair.shape[0], candidate_pair_count[0])

        # Early exit if no work (fast path for primitive-only scenes)
        if num_work_items == 0:
            return

        block_dim = wp.block_dim()
        num_blocks = total_num_threads // block_dim
        block_index = tid // block_dim
        lane = tid - block_index * block_dim

        items_per_block = (num_work_items + num_blocks - 1) // num_blocks
        block_start = block_index * items_per_block

        for local_index in range(lane, items_per_block, block_dim):
            t = block_start + local_index
            if t >= num_work_items:
                continue
            # Get a pre-routed pair.
            pair = candidate_pair[t]
            shape_a = pair[0]
            shape_b = pair[1]

            # Safety checks
            if shape_a == shape_b or shape_a < 0 or shape_b < 0:
                continue

            type_a = shape_types[shape_a]
            type_b = shape_types[shape_b]

            # Extract shape data
            pos_a, quat_a, shape_data_a, scale_a, margin_offset_a = extract_shape_data(
                shape_a, shape_transform, shape_types, shape_data, shape_source
            )
            pos_b, quat_b, shape_data_b, scale_b, margin_offset_b = extract_shape_data(
                shape_b, shape_transform, shape_types, shape_data, shape_source
            )
            if type_a == GeoType.CONVEX_MESH:
                shape_data_a.center = 0.5 * (shape_collision_aabb_lower[shape_a] + shape_collision_aabb_upper[shape_a])
                shape_data_a.bounds_half = 0.5 * (
                    shape_collision_aabb_upper[shape_a] - shape_collision_aabb_lower[shape_a]
                )
            if type_b == GeoType.CONVEX_MESH:
                shape_data_b.center = 0.5 * (shape_collision_aabb_lower[shape_b] + shape_collision_aabb_upper[shape_b])
                shape_data_b.bounds_half = 0.5 * (
                    shape_collision_aabb_upper[shape_b] - shape_collision_aabb_lower[shape_b]
                )
            data_provider = provider_type()
            data_provider.shape_support_data = shape_support_data
            data_provider.support_neighbors = support_neighbors
            if wp.static(accelerated_support):
                data_provider.support_lut = support_lut
                data_provider.support_vertex_offsets = support_vertex_offsets

            # Check for infinite planes
            is_infinite_plane_a = (type_a == GeoType.PLANE) and (scale_a[0] == 0.0 and scale_a[1] == 0.0)
            is_infinite_plane_b = (type_b == GeoType.PLANE) and (scale_b[0] == 0.0 and scale_b[1] == 0.0)

            # Early exit: both infinite planes can't collide
            if is_infinite_plane_a and is_infinite_plane_b:
                continue

            # Bounding sphere check is only needed for infinite plane pairs.
            # For non-plane pairs with external AABBs, SAP already verified AABB overlap.
            bsphere_radius_a = float(0.0)
            bsphere_radius_b = float(0.0)
            has_infinite_plane = is_infinite_plane_a or is_infinite_plane_b

            if has_infinite_plane:
                # Compute or fetch AABBs for bounding sphere overlap check
                if wp.static(external_aabb):
                    aabb_a_lower = shape_aabb_lower[shape_a]
                    aabb_a_upper = shape_aabb_upper[shape_a]
                    aabb_b_lower = shape_aabb_lower[shape_b]
                    aabb_b_upper = shape_aabb_upper[shape_b]
                if wp.static(not external_aabb):
                    gap_a = shape_gap[shape_a]
                    gap_b = shape_gap[shape_b]
                    gap_vec_a = wp.vec3(gap_a, gap_a, gap_a)
                    gap_vec_b = wp.vec3(gap_b, gap_b, gap_b)

                    # Shape A AABB
                    if is_infinite_plane_a:
                        radius_a = shape_collision_radius[shape_a]
                        half_extents_a = wp.vec3(radius_a, radius_a, radius_a)
                        aabb_a_lower = pos_a - half_extents_a - gap_vec_a
                        aabb_a_upper = pos_a + half_extents_a + gap_vec_a
                    else:
                        aabb_a_lower, aabb_a_upper = compute_tight_aabb_from_support(
                            shape_data_a, quat_a, pos_a, data_provider
                        )
                        aabb_a_lower = aabb_a_lower - gap_vec_a
                        aabb_a_upper = aabb_a_upper + gap_vec_a

                    # Shape B AABB
                    if is_infinite_plane_b:
                        radius_b = shape_collision_radius[shape_b]
                        half_extents_b = wp.vec3(radius_b, radius_b, radius_b)
                        aabb_b_lower = pos_b - half_extents_b - gap_vec_b
                        aabb_b_upper = pos_b + half_extents_b + gap_vec_b
                    else:
                        aabb_b_lower, aabb_b_upper = compute_tight_aabb_from_support(
                            shape_data_b, quat_b, pos_b, data_provider
                        )
                        aabb_b_lower = aabb_b_lower - gap_vec_b
                        aabb_b_upper = aabb_b_upper + gap_vec_b

                # Compute bounding spheres and check for overlap (early rejection)
                bsphere_center_a, bsphere_radius_a = compute_bounding_sphere_from_aabb(aabb_a_lower, aabb_a_upper)
                bsphere_center_b, bsphere_radius_b = compute_bounding_sphere_from_aabb(aabb_b_lower, aabb_b_upper)
                overlap_radius_a = bsphere_radius_a
                overlap_radius_b = bsphere_radius_b

                # External AABBs describe the current geometry. Include both shapes' search
                # extensions so relative translational motion cannot be culled before GJK.
                if wp.static(external_aabb and speculative):
                    pair_search_extension = shape_gap[shape_a] + shape_gap[shape_b]
                    if is_infinite_plane_a:
                        overlap_radius_b += pair_search_extension
                    else:
                        overlap_radius_a += pair_search_extension

                if not check_infinite_plane_bsphere_overlap(
                    shape_data_a,
                    shape_data_b,
                    pos_a,
                    pos_b,
                    quat_a,
                    quat_b,
                    bsphere_center_a,
                    bsphere_center_b,
                    overlap_radius_a,
                    overlap_radius_b,
                ):
                    continue

            # Compute pairwise gap sum for contact detection
            gap_a = shape_gap[shape_a]
            gap_b = shape_gap[shape_b]
            gap_sum = gap_a + gap_b

            # Find and write contacts using GJK/MPR
            wp.static(
                create_find_contacts(writer_func, support_func=support_func, post_process_contact=post_process_contact)
            )(
                pos_a,
                pos_b,
                quat_a,
                quat_b,
                shape_data_a,
                shape_data_b,
                is_infinite_plane_a,
                is_infinite_plane_b,
                bsphere_radius_a,
                bsphere_radius_b,
                gap_sum,
                shape_a,
                shape_b,
                margin_offset_a,
                margin_offset_b,
                data_provider,
                writer_data,
            )

    return narrow_phase_kernel_gjk_mpr


def create_narrow_phase_kernels_gjk_mpr_split(
    external_aabb: bool,
    writer_func: Any,
    support_func: Any = None,
    post_process_contact: Any = None,
    speculative: bool = False,
):
    """Create graph-capturable MPR, GJK, and manifold work-queue kernels."""
    if support_func is None:
        support_func = support_map
    if post_process_contact is None:
        post_process_contact = post_process_axial_on_discrete_contact

    support_funcs = create_support_map_function(support_func, use_precomputed_center=True)
    solve_mpr = create_solve_mpr(support_func, _support_funcs=support_funcs)
    solve_gjk = create_solve_closest_distance(support_func, _support_funcs=support_funcs)
    prepare_pair = create_prepare_convex_pair(external_aabb, speculative)
    write_result = create_write_convex_query_result(
        support_func, writer_func, post_process_contact, use_precomputed_center=True
    )
    suffix = (
        f"{external_aabb}_{writer_func.__name__}_{support_func.__name__}_{post_process_contact.__name__}_{speculative}"
    )
    accelerated_support = support_func in (support_map_accelerated, support_map_lean_accelerated)
    provider_type = AcceleratedSupportMapDataProvider if accelerated_support else SupportMapDataProvider

    @wp.kernel(enable_backward=False, module=f"narrow_phase_mpr_{suffix}")
    def narrow_phase_mpr_kernel(
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_support_data: wp.array[wp.vec4i],
        support_lut: wp.array[int],
        support_vertex_offsets: wp.array[int],
        support_neighbors: wp.array[int],
        shape_gap: wp.array[float],
        shape_collision_radius: wp.array[float],
        shape_aabb_lower: wp.array[wp.vec3],
        shape_aabb_upper: wp.array[wp.vec3],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        gjk_work_items: wp.array[int],
        gjk_work_count: wp.array[int],
        manifold_work_items: wp.array[int],
        manifold_work_count: wp.array[int],
    ):
        tid = wp.tid()
        num_work_items = wp.min(query_results.shape[0], wp.min(candidate_pair.shape[0], candidate_pair_count[0]))
        block_dim = wp.block_dim()
        num_blocks = total_num_threads // block_dim
        block_index = tid // block_dim
        lane = tid - block_index * block_dim
        items_per_block = (num_work_items + num_blocks - 1) // num_blocks
        block_start = block_index * items_per_block

        for local_index in range(lane, items_per_block, block_dim):
            pair_index = block_start + local_index
            valid = pair_index < num_work_items
            query = ConvexPairQueryData()
            if valid:
                valid, query = wp.static(prepare_pair)(
                    candidate_pair[pair_index],
                    shape_types,
                    shape_data,
                    shape_transform,
                    shape_source,
                    shape_gap,
                    shape_collision_radius,
                    shape_aabb_lower,
                    shape_aabb_upper,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                )

            needs_gjk = False
            needs_manifold = False
            if valid:
                provider = provider_type()
                provider.shape_support_data = shape_support_data
                provider.support_neighbors = support_neighbors
                if wp.static(accelerated_support):
                    provider.support_lut = support_lut
                    provider.support_vertex_offsets = support_vertex_offsets
                collision, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(
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
                else:
                    needs_gjk = True

            _append_work_index_compacted(needs_gjk, pair_index, gjk_work_items, gjk_work_count)
            _append_work_index_compacted(needs_manifold, pair_index, manifold_work_items, manifold_work_count)

    @wp.kernel(enable_backward=False, module=f"narrow_phase_gjk_{suffix}")
    def narrow_phase_gjk_kernel(
        candidate_pair: wp.array[wp.vec2i],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_support_data: wp.array[wp.vec4i],
        support_lut: wp.array[int],
        support_vertex_offsets: wp.array[int],
        support_neighbors: wp.array[int],
        shape_gap: wp.array[float],
        shape_collision_radius: wp.array[float],
        shape_aabb_lower: wp.array[wp.vec3],
        shape_aabb_upper: wp.array[wp.vec3],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        gjk_work_items: wp.array[int],
        gjk_work_count: wp.array[int],
        manifold_work_items: wp.array[int],
        manifold_work_count: wp.array[int],
    ):
        tid = wp.tid()
        num_work_items = wp.min(gjk_work_items.shape[0], gjk_work_count[0])
        for work_index in range(tid, num_work_items, total_num_threads):
            pair_index = gjk_work_items[work_index]
            valid, query = wp.static(prepare_pair)(
                candidate_pair[pair_index],
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_gap,
                shape_collision_radius,
                shape_aabb_lower,
                shape_aabb_upper,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
            )
            needs_manifold = False
            if valid:
                provider = provider_type()
                provider.shape_support_data = shape_support_data
                provider.support_neighbors = support_neighbors
                if wp.static(accelerated_support):
                    provider.support_lut = support_lut
                    provider.support_vertex_offsets = support_vertex_offsets
                # Pairs whose support-plane bound clears the contact threshold stop early; no contact is kept.
                _separated, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                    query.geom_a,
                    query.geom_b,
                    query.relative_orientation_b,
                    query.relative_position_b,
                    0.0,
                    provider,
                    max_dist=gjk_separation_cutoff(query.contact_threshold, query.position_a, query.position_b),
                )
                if signed_distance <= query.contact_threshold:
                    result = ConvexQueryResult()
                    result.point_a = point_a
                    result.point_b = point_b
                    result.normal = normal
                    result.signed_distance = signed_distance
                    query_results[pair_index] = result
                    needs_manifold = True
            _append_work_index_compacted(needs_manifold, pair_index, manifold_work_items, manifold_work_count)

    @wp.kernel(enable_backward=False, module=f"narrow_phase_manifold_{suffix}")
    def narrow_phase_manifold_kernel(
        candidate_pair: wp.array[wp.vec2i],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_support_data: wp.array[wp.vec4i],
        support_lut: wp.array[int],
        support_vertex_offsets: wp.array[int],
        support_neighbors: wp.array[int],
        shape_gap: wp.array[float],
        shape_collision_radius: wp.array[float],
        shape_aabb_lower: wp.array[wp.vec3],
        shape_aabb_upper: wp.array[wp.vec3],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        writer_data: Any,
        total_num_threads: int,
        query_results: wp.array[ConvexQueryResult],
        manifold_work_items: wp.array[int],
        manifold_work_count: wp.array[int],
    ):
        tid = wp.tid()
        num_work_items = wp.min(manifold_work_items.shape[0], manifold_work_count[0])
        for work_index in range(tid, num_work_items, total_num_threads):
            pair_index = manifold_work_items[work_index]
            valid, query = wp.static(prepare_pair)(
                candidate_pair[pair_index],
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_gap,
                shape_collision_radius,
                shape_aabb_lower,
                shape_aabb_upper,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
            )
            if not valid:
                continue

            contact_template = ContactData()
            contact_template.radius_eff_a = query.radius_eff_a
            contact_template.radius_eff_b = query.radius_eff_b
            contact_template.margin_a = query.margin_a
            contact_template.margin_b = query.margin_b
            contact_template.shape_a = query.shape_a
            contact_template.shape_b = query.shape_b
            contact_template.gap_sum = query.rigid_gap
            contact_template.sort_sub_key = 0
            skip_multi_contact = (
                query.type_a == GeoType.SPHERE
                or query.type_b == GeoType.SPHERE
                or query.type_a == GeoType.ELLIPSOID
                or query.type_b == GeoType.ELLIPSOID
            )
            provider = provider_type()
            provider.shape_support_data = shape_support_data
            provider.support_neighbors = support_neighbors
            if wp.static(accelerated_support):
                provider.support_lut = support_lut
                provider.support_vertex_offsets = support_vertex_offsets
            wp.static(write_result)(
                query.geom_a,
                query.geom_b,
                query.orientation_a,
                query.orientation_b,
                query.position_a,
                query.position_b,
                query.relative_orientation_b,
                query.relative_position_b,
                query_results[pair_index],
                provider,
                skip_multi_contact,
                writer_data,
                contact_template,
            )

    return narrow_phase_mpr_kernel, narrow_phase_gjk_kernel, narrow_phase_manifold_kernel


@wp.kernel(enable_backward=False)
def narrow_phase_find_heightfield_triangle_overlaps_kernel(
    shape_types: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_gap: wp.array[float],
    shape_data: wp.array[wp.vec4],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_pairs: wp.array[wp.vec2i],
    shape_pairs_count: wp.array[int],
    total_num_threads: int,
    # outputs
    triangle_pairs: wp.array[wp.vec3i],
    triangle_pairs_count: wp.array[int],
):
    """Find heightfield triangles that overlap with a convex shape, one pair per thread.

    Used when the scene has heightfields but no meshes: every pair in
    ``shape_pairs`` is then a heightfield pair, whose cell enumeration is
    serial, so packing pairs into every lane avoids the idle lanes of the
    tiled mesh BVH launch.
    """
    for i in range(wp.tid(), shape_pairs_count[0], total_num_threads):
        pair = shape_pairs[i]
        hfd = heightfield_data[shape_heightfield_index[pair[0]]]
        heightfield_vs_convex_midphase(
            pair[0],
            pair[1],
            hfd,
            heightfield_elevations,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_data,
            shape_gap,
            triangle_pairs,
            triangle_pairs_count,
            # A plane's cached local AABB does not bound its surface.
            shape_types[pair[1]] != GeoType.PLANE,
        )


@wp.kernel(enable_backward=False)
def compute_shape_interior_points_kernel(
    shape_types: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_source: wp.array[wp.uint64],
    # outputs
    shape_interior_point: wp.array[wp.vec3],
):
    """Store an interior point of each shape in its scaled local frame.

    Primitive origins are interior. A convex hull's origin and AABB center can
    both lie outside it, so hulls use their volume centroid instead.
    """
    shape = wp.tid()
    point = wp.vec3(0.0, 0.0, 0.0)
    if shape_types[shape] == GeoType.CONVEX_MESH:
        data = shape_data[shape]
        point = convex_mesh_interior_point(shape_source[shape], wp.vec3(data[0], data[1], data[2]))
    shape_interior_point[shape] = point


@wp.kernel(enable_backward=False)
def narrow_phase_find_mesh_triangle_overlaps_kernel(
    shape_types: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_gap: wp.array[float],  # Per-shape contact gaps
    shape_data: wp.array[wp.vec4],  # Shape data (scale xyz, margin w)
    shape_collision_aabb_lower: wp.array[wp.vec3],  # Local-space AABB lower bounds
    shape_collision_aabb_upper: wp.array[wp.vec3],  # Local-space AABB upper bounds
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_pairs_mesh: wp.array[wp.vec2i],
    shape_pairs_mesh_count: wp.array[int],
    mesh_cells: MeshCells,
    total_num_threads: int,
    # outputs
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    query_range: wp.array2d[wp.vec3i],
    query_grid: wp.array[wp.int32],
    query_slab0: wp.array[wp.vec4],
    query_slab1: wp.array[wp.vec4],
    query_slab2: wp.array[wp.vec4],
    query_slab_half: wp.array[wp.vec3],
    triangle_pairs: wp.array[wp.vec3i],  # (shape_a, shape_b, triangle_idx)
    triangle_pairs_count: wp.array[int],
):
    """Midphase pass 1 of 3, one thread per mesh-like pair.

    A heightfield pair projects the convex shape's bounding sphere onto the heightfield grid and emits its triangle
    pairs here. A mesh-vs-convex pair stores its bound's box in the mesh frame (:func:`mesh_vs_convex_query_box`),
    the bound's oriented box as three slabs (:func:`mesh_query_bound_slab`), its level-0 cell range and its mesh's
    cell index; other pairs, and pairs whose box misses the mesh, store -1.
    """
    for i in range(wp.tid(), shape_pairs_mesh_count[0], total_num_threads):
        pair = shape_pairs_mesh[i]
        query_grid[i] = -1
        # Pairs are normalized so the heightfield is always shape_a.
        if shape_types[pair[0]] == GeoType.HFIELD:
            hfd = heightfield_data[shape_heightfield_index[pair[0]]]
            heightfield_vs_convex_midphase(
                pair[0],
                pair[1],
                hfd,
                heightfield_elevations,
                shape_transform,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_data,
                shape_gap,
                triangle_pairs,
                triangle_pairs_count,
                # A plane's cached local AABB does not bound its surface.
                shape_types[pair[1]] != GeoType.PLANE,
            )
            continue
        mesh_shape, non_mesh_shape = mesh_vs_convex_pair_shapes(pair, shape_types)
        # Mesh-mesh pairs take their own route.
        if mesh_shape < 0 or shape_source[mesh_shape] == wp.uint64(0):
            continue
        # The margin+gap shell the narrow phase accepts contacts in.
        contact_threshold = (
            shape_gap[non_mesh_shape]
            + shape_gap[mesh_shape]
            + shape_data[non_mesh_shape][3]
            + shape_data[mesh_shape][3]
        )
        grid = mesh_cells.shape_grid[mesh_shape]
        if grid < 0:
            continue
        convex = _mesh_vs_convex_shape_data(non_mesh_shape, shape_types, shape_data, shape_source)
        convex_in_mesh = wp.transform_multiply(
            wp.transform_inverse(shape_transform[mesh_shape]), shape_transform[non_mesh_shape]
        )
        local_lower, local_upper = mesh_vs_convex_local_box(
            convex, non_mesh_shape, shape_collision_aabb_lower, shape_collision_aabb_upper
        )
        bound = _mesh_query_bound(convex, local_lower, local_upper, convex_in_mesh, contact_threshold)
        mesh_data = shape_data[mesh_shape]
        mesh_scale = wp.vec3(mesh_data[0], mesh_data[1], mesh_data[2])
        lower, upper = mesh_query_bound_unscaled_box(bound, mesh_scale)
        lo0, hi0, inside = mesh_vs_convex_query_range(mesh_cells, grid, lower, upper)
        if inside:
            query_slab0[i] = mesh_query_bound_slab(bound, 0, mesh_scale)
            query_slab1[i] = mesh_query_bound_slab(bound, 1, mesh_scale)
            query_slab2[i] = mesh_query_bound_slab(bound, 2, mesh_scale)
            query_slab_half[i] = bound.half
            query_lower[i] = lower
            query_upper[i] = upper
            query_range[i, 0] = lo0
            query_range[i, 1] = hi0
            query_grid[i] = grid


# The pass is bound by its loads' latency: 64 registers keep more warps in flight.
@wp.kernel(enable_backward=False, cuda_max_registers=64)
def narrow_phase_mesh_cell_hits_kernel(
    shape_pairs_mesh_count: wp.array[int],
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    query_range: wp.array2d[wp.vec3i],
    query_grid: wp.array[wp.int32],
    query_slab0: wp.array[wp.vec4],
    query_slab1: wp.array[wp.vec4],
    query_slab2: wp.array[wp.vec4],
    query_slab_half: wp.array[wp.vec3],
    mesh_cells: MeshCells,
    total_num_threads: int,
    # outputs
    hits: wp.array[wp.vec2i],  # (mesh pair, cell in mesh_cells.sorted_slots)
    hit_count: wp.array[int],
):
    """Midphase pass 2a of 3, one thread per mesh pair (launch ``[blocks, block_dim]``): its hit cells
    (:func:`mesh_vs_convex_cell_hits`)."""
    block, j = wp.tid()
    for i in range(block * wp.block_dim() + j, shape_pairs_mesh_count[0], total_num_threads):
        grid = query_grid[i]
        if grid >= 0:
            mesh_vs_convex_cell_hits(
                i,
                grid,
                query_lower[i],
                query_upper[i],
                query_range[i, 0],
                query_range[i, 1],
                query_slab0[i],
                query_slab1[i],
                query_slab2[i],
                query_slab_half[i],
                mesh_cells,
                hits,
                hit_count,
            )


@wp.kernel(enable_backward=False, cuda_max_registers=64)
def narrow_phase_mesh_hit_candidates_kernel(
    hits: wp.array[wp.vec2i],
    hit_count: wp.array[int],
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    query_range: wp.array2d[wp.vec3i],
    query_grid: wp.array[wp.int32],
    query_slab0: wp.array[wp.vec4],
    query_slab1: wp.array[wp.vec4],
    query_slab2: wp.array[wp.vec4],
    query_slab_half: wp.array[wp.vec3],
    mesh_cells: MeshCells,
    total_num_threads: int,
    # outputs
    candidates: wp.array[wp.vec4i],  # (mesh pair, triangle, cell)
    candidate_count: wp.array[int],
):
    """Midphase pass 2b of 3, one thread per hit cell: its candidate triangles (:func:`mesh_vs_convex_hit_candidates`).

    A pair's cells run on as many threads as it has hits, so a pair with many entries no longer holds its lanes."""
    tid = wp.tid()
    for h in range(tid, wp.min(hit_count[0], hits.shape[0]), total_num_threads):
        i = hits[h][0]
        mesh_vs_convex_hit_candidates(
            i,
            mesh_cells.sorted_slots[hits[h][1]],
            query_lower[i],
            query_upper[i],
            query_range[i, 0],
            query_range[i, 1],
            query_slab0[i],
            query_slab1[i],
            query_slab2[i],
            query_slab_half[i],
            mesh_cells.grids[query_grid[i]],
            mesh_cells,
            candidates,
            candidate_count,
        )


@wp.kernel(enable_backward=False)
def narrow_phase_mesh_triangle_candidates_kernel(
    shape_types: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[wp.int32],
    shape_gap: wp.array[float],
    shape_data: wp.array[wp.vec4],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_interior_point: wp.array[wp.vec3],  # Local interior points for back-face culling
    shape_pairs_mesh: wp.array[wp.vec2i],
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    candidates: wp.array[wp.vec4i],
    candidate_count: wp.array[int],
    total_num_threads: int,
    # outputs
    triangle_pairs: wp.array[wp.vec3i],  # (shape_a, shape_b, triangle_idx)
    triangle_pairs_count: wp.array[int],
):
    """Midphase pass 3 of 3, one thread per candidate: exact bound and back-face tests (:func:`mesh_vs_convex_midphase`)."""
    tid = wp.tid()
    count = candidate_count[0]
    for c in range(tid, wp.min(count, candidates.shape[0]), total_num_threads):
        candidate = candidates[c]
        mesh_shape, non_mesh_shape = mesh_vs_convex_pair_shapes(shape_pairs_mesh[candidate[0]], shape_types)
        contact_threshold = (
            shape_gap[non_mesh_shape]
            + shape_gap[mesh_shape]
            + shape_data[non_mesh_shape][3]
            + shape_data[mesh_shape][3]
        )
        mesh_vs_convex_midphase(
            mesh_shape,
            non_mesh_shape,
            candidate,
            query_lower[candidate[0]],
            query_upper[candidate[0]],
            shape_transform[mesh_shape],
            shape_transform[non_mesh_shape],
            shape_source[mesh_shape],
            shape_types,
            shape_data,
            shape_source,
            shape_mesh_properties,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            contact_threshold,
            shape_interior_point[non_mesh_shape],
            triangle_pairs,
            triangle_pairs_count,
        )


@wp.kernel(enable_backward=False)
def narrow_phase_mesh_buffer_report_kernel(
    triangle_pairs_count: wp.array[int],
    max_triangle_pairs: int,
    candidate_count: wp.array[int],
    max_candidates: int,
    hit_count: wp.array[int],
    max_hits: int,
    report: wp.array[int],
):
    """The triangle pair count the buffer check reads: past capacity when the mesh midphase's hits or candidates were
    too."""
    count = triangle_pairs_count[0]
    if candidate_count[0] > max_candidates:
        wp.printf("Warning: Mesh midphase candidate buffer overflowed %d > %d.\n", candidate_count[0], max_candidates)
        count = wp.max(count, max_triangle_pairs + 1)
    if hit_count[0] > max_hits:
        wp.printf("Warning: Mesh midphase hit buffer overflowed %d > %d.\n", hit_count[0], max_hits)
        count = wp.max(count, max_triangle_pairs + 1)
    report[0] = count


def create_narrow_phase_process_mesh_triangle_contacts_kernel(
    writer_func: Any,
    convex_support_acceleration: bool = False,
    *,
    skip_marked: bool = False,
    heightfields_only: bool = False,
):
    suffix = "_finite_fallback" if skip_marked else ""
    if heightfields_only:
        # Heightfield triangles only, beside the mesh triangles' own kernels.
        suffix += "_heightfields"
    _module = f"narrow_phase_mesh_tri_{writer_func.__name__}_{convex_support_acceleration}{suffix}"
    support_func = support_map_accelerated_inline if convex_support_acceleration else support_map_inline
    provider_type = AcceleratedSupportMapDataProvider if convex_support_acceleration else SupportMapDataProvider
    compute_contacts = create_compute_gjk_mpr_contacts(
        writer_func,
        support_func=support_func,
        use_precomputed_center=True,
        penetration_refiner=create_surface_penetration_refiner(support_func),
        query_filter=surface_query_filter,
        point_filter=mesh_contact_filter,
        manifold_shape=mesh_manifold_triangle,
        deepest_once=mesh_deepest_once,
    )
    mesh_sphere_contact = create_mesh_sphere_contact(writer_func)
    mesh_capsule_contacts = create_mesh_capsule_contacts(writer_func)

    def narrow_phase_process_mesh_triangle_contacts_kernel(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],  # Per-shape contact gaps
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
        writer_data: Any,
        total_num_threads: int,
        # Convex-hull support acceleration and interior seeds (the support inputs of the launch).
        shape_support_data: wp.array[wp.vec4i],
        support_lut: wp.array[int],
        support_vertex_offsets: wp.array[int],
        support_neighbors: wp.array[int],
        shape_interior_point: wp.array[wp.vec3],
        mesh_topology: MeshTopologyData,
    ):
        """
        Process triangle pairs to generate contacts: GJK/MPR, spheres on mesh triangles in closed form,
        and the surface's feature rule on every mesh contact.
        """
        tid = wp.tid()

        num_triangle_pairs = triangle_pairs_count[0]

        for i in range(tid, num_triangle_pairs, total_num_threads):
            if i >= triangle_pairs.shape[0]:
                break

            triple = triangle_pairs[i]
            shape_a = triple[0]
            shape_b = triple[1]
            tri_idx = triple[2]
            if wp.static(skip_marked):
                if tri_idx < 0:
                    continue

            type_a = shape_types[shape_a]
            if wp.static(heightfields_only):
                if type_a != GeoType.HFIELD:
                    continue

            corner_0 = wp.vec3(0.0)
            corner_1 = wp.vec3(0.0)
            corner_2 = wp.vec3(0.0)
            if type_a == GeoType.HFIELD:
                # Heightfield triangle
                hfd = heightfield_data[shape_heightfield_index[shape_a]]
                X_ws_a = shape_transform[shape_a]
                shape_data_a, v0_world = get_triangle_shape_from_heightfield(
                    hfd, heightfield_elevations, X_ws_a, tri_idx
                )
            else:
                # Mesh triangle
                mesh_id_a = shape_source[shape_a]
                scale_data_a = shape_data[shape_a]
                mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
                X_ws_a = shape_transform[shape_a]
                corner_0, corner_1, corner_2 = mesh_triangle_corners(mesh_id_a, mesh_scale_a, X_ws_a, tri_idx)
                shape_data_a = triangle_shape_from_corners(corner_0, corner_1, corner_2, mesh_scale_a)
                v0_world = corner_0

            # Extract shape B data
            pos_b, quat_b, shape_data_b, _scale_b, margin_offset_b = extract_shape_data(
                shape_b,
                shape_transform,
                shape_types,
                shape_data,
                shape_source,
            )
            if shape_data_b.shape_type == GeoType.CONVEX_MESH:
                # MPR/GJK need an interior seed. A hull's AABB center can lie
                # outside it and flip the contact against thin triangles.
                shape_data_b.center = shape_interior_point[shape_b]

            # Triangle position is vertex A in world space.
            # For heightfield prisms, edges are in heightfield-local space
            # so we pass the heightfield rotation to let MPR/GJK work in
            # that frame (where -Z is always the down axis).
            pos_a = v0_world
            if type_a == GeoType.HFIELD:
                quat_a = wp.transform_get_rotation(X_ws_a)
            else:
                quat_a = wp.quat_identity()

            # Extract margin offset for shape A (signed distance padding)
            margin_offset_a = shape_data[shape_a][3]

            # Sum per-shape contact gaps for consistent pairwise thresholding
            gap_a = shape_gap[shape_a]
            gap_b = shape_gap[shape_b]
            gap_sum = gap_a + gap_b

            feature_context = HeightfieldFeatureContext()
            if type_a == GeoType.HFIELD:
                feature_context.elevations = heightfield_elevations
                feature_context.triangle = tri_idx
                feature_context.heightfield = heightfield_data[shape_heightfield_index[shape_a]]

            data_provider = provider_type()
            data_provider.shape_support_data = shape_support_data
            data_provider.support_neighbors = support_neighbors
            if wp.static(convex_support_acceleration):
                data_provider.support_lut = support_lut
                data_provider.support_vertex_offsets = support_vertex_offsets

            # Heightfield triangles have no mesh: their context reads no mesh data.
            surface_mesh = wp.uint64(0)
            if type_a == GeoType.MESH:
                surface_mesh = shape_source[shape_a]
            if type_a == GeoType.MESH and shape_data_b.shape_type != GeoType.SPHERE:
                if shape_data_b.shape_type != GeoType.CAPSULE:
                    # Skip a triangle whose plane the shape clears by more than the contact threshold, before
                    # anything else is read (spheres and capsules have their own exits).
                    face = wp.cross(shape_data_a.scale, shape_data_a.auxiliary)
                    support_b = wp.static(support_func)(shape_data_b, wp.quat_rotate_inv(quat_b, -face), data_provider)
                    if triangle_plane_clears(
                        shape_data_a,
                        pos_a,
                        wp.quat_rotate(quat_b, support_b) + pos_b,
                        gap_sum + margin_offset_a + margin_offset_b,
                    ):
                        continue
            surface = mesh_surface_context(
                mesh_topology,
                shape_a,
                tri_idx,
                surface_mesh,
                wp.vec3(shape_data[shape_a][0], shape_data[shape_a][1], shape_data[shape_a][2]),
                X_ws_a,
                corner_0,
                corner_1,
                corner_2,
                pos_b + wp.quat_rotate(quat_b, shape_data_b.center),
                shape_b,
            )
            surface.heightfield = feature_context
            if type_a != GeoType.HFIELD:
                # Mesh triangles: spheres and capsules in closed form; other shapes skip a triangle whose plane the
                # shape clears by more than the contact threshold.
                threshold = gap_sum + margin_offset_a + margin_offset_b
                if shape_data_b.shape_type == GeoType.SPHERE:
                    mesh_sphere_contact(
                        surface,
                        pos_b,
                        shape_data_b.scale[0],
                        threshold,
                        margin_offset_a,
                        margin_offset_b,
                        shape_a,
                        shape_b,
                        gap_sum,
                        (tri_idx << 1) | 1,
                        writer_data,
                    )
                    continue
                if shape_data_b.shape_type == GeoType.CAPSULE:
                    core = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, shape_data_b.scale[1]))
                    mesh_capsule_contacts(
                        surface,
                        pos_b - core,
                        pos_b + core,
                        shape_data_b.scale[0],
                        threshold,
                        margin_offset_a,
                        margin_offset_b,
                        shape_a,
                        shape_b,
                        gap_sum,
                        (tri_idx << 1) | 1,
                        writer_data,
                    )
                    continue
            surface = mesh_surface_planes(surface)
            wp.static(compute_contacts)(
                shape_data_a,
                shape_data_b,
                quat_a,
                quat_b,
                pos_a,
                pos_b,
                gap_sum,
                shape_a,
                shape_b,
                margin_offset_a,
                margin_offset_b,
                data_provider,
                writer_data,
                (tri_idx << 1) | 1,
                surface,
            )

    name = "narrow_phase_process_mesh_triangle_contacts_kernel" + suffix
    narrow_phase_process_mesh_triangle_contacts_kernel.__name__ = name
    narrow_phase_process_mesh_triangle_contacts_kernel.__qualname__ = name
    return wp.kernel(narrow_phase_process_mesh_triangle_contacts_kernel, enable_backward=False, module=_module)


@wp.kernel(enable_backward=False)
def compute_mesh_plane_vert_counts(
    shape_pairs_mesh_plane: wp.array[wp.vec2i],
    shape_pairs_mesh_plane_count: wp.array[int],
    shape_source: wp.array[wp.uint64],
    vert_counts: wp.array[wp.int32],
    total_vert_count: wp.array[wp.int32],
    total_num_threads: int,
):
    """Compute per-pair vertex counts in parallel for mesh-plane pairs.

    Threads stride over the active pair prefix. Values beyond ``pair_count``
    cannot affect the consumed prefix of the subsequent exclusive scan.
    """
    pair_count = wp.min(shape_pairs_mesh_plane_count[0], shape_pairs_mesh_plane.shape[0])
    for i in range(wp.tid(), pair_count, total_num_threads):
        pair = shape_pairs_mesh_plane[i]
        mesh_shape = pair[0]
        mesh_id = shape_source[mesh_shape]
        pair_verts = int(0)  # noqa: RUF046, RUF100 - explicit cast required by Warp codegen
        if mesh_id != wp.uint64(0):
            pair_verts = wp.mesh_get(mesh_id).points.shape[0]
        vert_counts[i] = wp.int32(pair_verts)
        wp.atomic_add(total_vert_count, 0, pair_verts)


def compute_mesh_plane_block_offsets_scan(
    shape_pairs_mesh_plane: wp.array[wp.vec2i],
    shape_pairs_mesh_plane_count: wp.array[int],
    shape_source: wp.array[wp.uint64],
    target_blocks: int,
    block_offsets: wp.array[wp.int32],
    block_counts: wp.array[wp.int32],
    total_vert_count: wp.array[wp.int32],
    total_num_threads: int,
    device: str | None = None,
    record_tape: bool = True,
):
    """Compute mesh-plane block offsets using parallel kernels and array_scan."""
    n = block_counts.shape[0]
    launch_threads = min(n, total_num_threads)
    # Step 1: compute per-pair vertex counts in parallel
    wp.launch(
        kernel=compute_mesh_plane_vert_counts,
        dim=launch_threads,
        inputs=[
            shape_pairs_mesh_plane,
            shape_pairs_mesh_plane_count,
            shape_source,
            block_counts,  # reuse as temp storage for vert counts
            total_vert_count,
            launch_threads,
        ],
        device=device,
        record_tape=record_tape,
    )
    # Step 2: compute per-pair block counts using the scalar total.
    wp.launch(
        kernel=compute_block_counts_from_weights,
        dim=launch_threads,
        inputs=[
            total_vert_count,
            block_counts,  # still holds vert counts
            shape_pairs_mesh_plane_count,
            shape_pairs_mesh_plane.shape[0],
            target_blocks,
            block_offsets,  # reuse as temp for block counts
            launch_threads,
        ],
        device=device,
        record_tape=record_tape,
    )
    # Step 3: exclusive scan of block counts → block_offsets
    wp.utils.array_scan(block_offsets, block_offsets, inclusive=False)


def create_narrow_phase_process_mesh_plane_contacts_kernel(
    writer_func: Any,
    reduce_contacts: bool = False,
):
    """
    Create a mesh-plane collision kernel.

    Args:
        writer_func: Contact writer function (e.g., write_contact_simple)
        reduce_contacts: If True, return multi-block load-balanced variant for global reduction.

    Returns:
        A warp kernel that processes mesh-plane collisions
    """
    _module = f"narrow_phase_mesh_plane_{writer_func.__name__}_{reduce_contacts}"

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_process_mesh_plane_contacts_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        _shape_collision_aabb_lower: wp.array[wp.vec3],  # Unused but kept for API compatibility
        _shape_collision_aabb_upper: wp.array[wp.vec3],  # Unused but kept for API compatibility
        _shape_voxel_resolution: wp.array[wp.vec3i],  # Unused but kept for API compatibility
        shape_pairs_mesh_plane: wp.array[wp.vec2i],
        shape_pairs_mesh_plane_count: wp.array[int],
        writer_data: Any,
        total_num_blocks: int,
    ):
        """
        Process mesh-plane collisions without contact reduction.

        Each thread processes vertices in a strided manner and writes contacts directly.
        """
        tid = wp.tid()

        pair_count = shape_pairs_mesh_plane_count[0]

        # Iterate over all mesh-plane pairs
        for pair_idx in range(pair_count):
            pair = shape_pairs_mesh_plane[pair_idx]
            mesh_shape = pair[0]
            plane_shape = pair[1]

            # Get mesh
            mesh_id = shape_source[mesh_shape]
            if mesh_id == wp.uint64(0):
                continue

            mesh_obj = wp.mesh_get(mesh_id)
            num_vertices = mesh_obj.points.shape[0]

            # Get mesh world transform
            X_mesh_ws = shape_transform[mesh_shape]

            # Get plane world transform
            X_plane_ws = shape_transform[plane_shape]
            X_plane_sw = wp.transform_inverse(X_plane_ws)

            # Get plane normal in world space (plane normal is along local +Z, pointing upward)
            plane_normal = wp.transform_vector(X_plane_ws, wp.vec3(0.0, 0.0, 1.0))

            # Get mesh scale
            scale_data = shape_data[mesh_shape]
            mesh_scale = wp.vec3(scale_data[0], scale_data[1], scale_data[2])

            # Extract per-shape margin offsets (stored in shape_data.w)
            margin_offset_mesh = shape_data[mesh_shape][3]
            margin_offset_plane = shape_data[plane_shape][3]
            total_margin_offset = margin_offset_mesh + margin_offset_plane

            # Use per-shape contact gap for contact detection threshold
            gap_mesh = shape_gap[mesh_shape]
            gap_plane = shape_gap[plane_shape]
            gap_sum = gap_mesh + gap_plane

            # Strided loop over vertices across all threads in the launch
            total_num_threads = total_num_blocks * wp.block_dim()
            for vertex_idx in range(tid, num_vertices, total_num_threads):
                # Get vertex position in mesh local space and transform to world space
                vertex_local = wp.cw_mul(mesh_obj.points[vertex_idx], mesh_scale)
                vertex_world = wp.transform_point(X_mesh_ws, vertex_local)

                # Project vertex onto plane to get closest point
                vertex_in_plane_space = wp.transform_point(X_plane_sw, vertex_world)
                point_on_plane_local = wp.vec3(vertex_in_plane_space[0], vertex_in_plane_space[1], 0.0)
                point_on_plane = wp.transform_point(X_plane_ws, point_on_plane_local)

                # Compute distance
                diff = vertex_world - point_on_plane
                distance = wp.dot(diff, plane_normal)

                # Check if this vertex generates a contact
                if distance < gap_sum + total_margin_offset:
                    # Contact position is the midpoint
                    contact_pos = (vertex_world + point_on_plane) * 0.5

                    # Normal points from mesh to plane (negate plane normal since plane normal points up/away from plane)
                    contact_normal = -plane_normal

                    # Create contact data - contacts are already in world space
                    contact_data = ContactData()
                    contact_data.contact_point_center = contact_pos
                    contact_data.contact_normal_a_to_b = contact_normal
                    contact_data.contact_distance = distance
                    contact_data.radius_eff_a = 0.0
                    contact_data.radius_eff_b = 0.0
                    contact_data.margin_a = margin_offset_mesh
                    contact_data.margin_b = margin_offset_plane
                    contact_data.shape_a = mesh_shape
                    contact_data.shape_b = plane_shape
                    contact_data.gap_sum = gap_sum
                    contact_data.sort_sub_key = vertex_idx

                    if writer_data.contact_count[0] < writer_data.contact_max:
                        writer_func(contact_data, writer_data, -1)

    # Return early if contact reduction is disabled
    if not reduce_contacts:
        return narrow_phase_process_mesh_plane_contacts_kernel

    @wp.kernel(enable_backward=False, module=_module)
    def narrow_phase_process_mesh_plane_contacts_reduce_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        _shape_collision_aabb_lower: wp.array[wp.vec3],
        _shape_collision_aabb_upper: wp.array[wp.vec3],
        _shape_voxel_resolution: wp.array[wp.vec3i],
        shape_pairs_mesh_plane: wp.array[wp.vec2i],
        shape_pairs_mesh_plane_count: wp.array[int],
        block_offsets: wp.array[wp.int32],
        writer_data: Any,
        total_num_blocks: int,
    ):
        """Process mesh-plane collisions with dynamic load balancing.

        Multiple GPU blocks can be assigned to the same mesh-plane pair
        based on vertex count.  Contacts are written directly to the
        global contact reducer buffer via ``writer_func``; reduction into
        the hashtable happens in a separate pass.  This avoids per-block
        shared-memory reduction and unifies the contact reduction path
        with the one used for mesh-mesh contacts.
        """
        block_id, t = wp.tid()

        pair_count = wp.min(shape_pairs_mesh_plane_count[0], shape_pairs_mesh_plane.shape[0])
        total_combos = block_offsets[pair_count]

        # Grid stride loop over (pair, sub-block) combos for multi-block load balancing.
        for combo_idx in range(block_id, total_combos, total_num_blocks):
            # Binary search block_offsets to find the pair for this block
            lo = int(0)
            hi = int(pair_count)
            while lo < hi:
                mid = (lo + hi) // 2
                if block_offsets[mid + 1] <= combo_idx:
                    lo = mid + 1
                else:
                    hi = mid
            pair_idx = int(lo)
            pair_block_start = block_offsets[pair_idx]
            block_in_pair = combo_idx - pair_block_start
            blocks_for_pair = block_offsets[pair_idx + 1] - pair_block_start

            # Get the mesh-plane pair
            pair = shape_pairs_mesh_plane[pair_idx]
            mesh_shape = pair[0]
            plane_shape = pair[1]

            # Get mesh
            mesh_id = shape_source[mesh_shape]
            if mesh_id == wp.uint64(0):
                continue

            mesh_obj = wp.mesh_get(mesh_id)
            num_vertices = mesh_obj.points.shape[0]

            # Compute vertex range for this sub-block
            chunk_size = (num_vertices + blocks_for_pair - 1) // blocks_for_pair
            vert_start = block_in_pair * chunk_size
            vert_end = wp.min(vert_start + chunk_size, num_vertices)

            # Get mesh world transform
            X_mesh_ws = shape_transform[mesh_shape]

            # Get plane world transform
            X_plane_ws = shape_transform[plane_shape]
            X_plane_sw = wp.transform_inverse(X_plane_ws)

            # Get plane normal in world space (plane normal is along local +Z)
            plane_normal = wp.transform_vector(X_plane_ws, wp.vec3(0.0, 0.0, 1.0))

            # Get mesh scale
            scale_data = shape_data[mesh_shape]
            mesh_scale = wp.vec3(scale_data[0], scale_data[1], scale_data[2])

            # Extract per-shape margin offsets (stored in shape_data.w)
            margin_offset_mesh = scale_data[3]
            margin_offset_plane = shape_data[plane_shape][3]
            total_margin_offset = margin_offset_mesh + margin_offset_plane

            # Use per-shape contact gap for contact detection threshold
            gap_mesh = shape_gap[mesh_shape]
            gap_plane = shape_gap[plane_shape]
            gap_sum = gap_mesh + gap_plane

            # Process this block's chunk of vertices — write contacts directly
            # to the global reducer buffer (no per-block shared memory reduction).
            for vertex_idx in range(vert_start + t, vert_end, wp.block_dim()):
                # Get vertex position in mesh local space and transform to world space
                vertex_local = wp.cw_mul(mesh_obj.points[vertex_idx], mesh_scale)
                vertex_world = wp.transform_point(X_mesh_ws, vertex_local)

                # Project vertex onto plane to get closest point
                vertex_in_plane_space = wp.transform_point(X_plane_sw, vertex_world)
                point_on_plane_local = wp.vec3(vertex_in_plane_space[0], vertex_in_plane_space[1], 0.0)
                point_on_plane = wp.transform_point(X_plane_ws, point_on_plane_local)

                # Compute distance
                diff = vertex_world - point_on_plane
                distance = wp.dot(diff, plane_normal)

                # Check if this vertex generates a contact
                if distance < gap_sum + total_margin_offset:
                    # Contact position is the midpoint
                    contact_pos = (vertex_world + point_on_plane) * 0.5

                    # Normal points from mesh to plane
                    contact_normal = -plane_normal

                    contact_data = ContactData()
                    contact_data.contact_point_center = contact_pos
                    contact_data.contact_normal_a_to_b = contact_normal
                    contact_data.contact_distance = distance
                    contact_data.radius_eff_a = 0.0
                    contact_data.radius_eff_b = 0.0
                    contact_data.margin_a = margin_offset_mesh
                    contact_data.margin_b = margin_offset_plane
                    contact_data.shape_a = mesh_shape
                    contact_data.shape_b = plane_shape
                    contact_data.gap_sum = gap_sum
                    contact_data.sort_sub_key = vertex_idx

                    writer_func(contact_data, writer_data, -1)

    return narrow_phase_process_mesh_plane_contacts_reduce_kernel


# =============================================================================
# Verification kernel
# =============================================================================


@wp.func
def _check_narrow_phase_buffers(
    broad_phase_count: wp.array[int],
    max_broad_phase: int,
    split_query_count: wp.array[int],
    max_split_query: int,
    gjk_count: wp.array[int],
    max_gjk: int,
    split_gjk_count: wp.array[int],
    max_split_gjk: int,
    split_manifold_count: wp.array[int],
    max_split_manifold: int,
    mesh_count: wp.array[int],
    max_mesh: int,
    triangle_count: wp.array[int],
    max_triangle: int,
    mesh_plane_count: wp.array[int],
    max_mesh_plane: int,
    mesh_mesh_count: wp.array[int],
    max_mesh_mesh: int,
    sdf_sdf_count: wp.array[int],
    max_sdf_sdf: int,
    contact_count: wp.array[int],
    max_contacts: int,
    reduction_ht_active_slots: wp.array[int],
    reduction_ht_capacity: int,
    reduction_ht_insert_failures: wp.array[int],
    reduction_buffer_overflows: wp.array[int],
    reduction_ht_warn_load_percent: int,
    flags: int,
) -> int:
    """Print each buffer that overflowed; return ``flags`` with the status bit of each set."""
    if broad_phase_count[0] > max_broad_phase:
        flags = flags | 1
        wp.printf(
            "Warning: Broad phase pair buffer overflowed %d > %d.\n",
            broad_phase_count[0],
            max_broad_phase,
        )
    if max_split_query >= 0 and split_query_count[0] > max_split_query:
        flags = flags | 2
        wp.printf(
            "Warning: Split query-result buffer overflowed %d > %d.\n",
            split_query_count[0],
            max_split_query,
        )
    if gjk_count[0] > max_gjk:
        flags = flags | 4
        wp.printf(
            "Warning: GJK candidate pair buffer overflowed %d > %d.\n",
            gjk_count[0],
            max_gjk,
        )
    if max_split_gjk >= 0:
        split_count = split_gjk_count[0]
        if split_count > max_split_gjk:
            flags = flags | 8
            wp.printf(
                "Warning: Split GJK work-item buffer overflowed %d > %d.\n",
                split_count,
                max_split_gjk,
            )
    if max_split_manifold >= 0 and split_manifold_count[0] > max_split_manifold:
        flags = flags | 16
        wp.printf(
            "Warning: Split manifold work-item buffer overflowed %d > %d.\n",
            split_manifold_count[0],
            max_split_manifold,
        )
    if mesh_count:
        if mesh_count[0] > max_mesh:
            flags = flags | 32
            wp.printf(
                "Warning: Mesh-convex shape pair buffer overflowed %d > %d.\n",
                mesh_count[0],
                max_mesh,
            )
    if triangle_count:
        if triangle_count[0] > max_triangle:
            flags = flags | 64
            wp.printf(
                "Warning: Triangle pair buffer overflowed %d > %d.\n",
                triangle_count[0],
                max_triangle,
            )
    if mesh_plane_count:
        if mesh_plane_count[0] > max_mesh_plane:
            flags = flags | 128
            wp.printf(
                "Warning: Mesh-plane shape pair buffer overflowed %d > %d.\n",
                mesh_plane_count[0],
                max_mesh_plane,
            )
    if mesh_mesh_count:
        if mesh_mesh_count[0] > max_mesh_mesh:
            flags = flags | 256
            wp.printf(
                "Warning: Mesh-mesh shape pair buffer overflowed %d > %d.\n",
                mesh_mesh_count[0],
                max_mesh_mesh,
            )
    if sdf_sdf_count:
        if sdf_sdf_count[0] > max_sdf_sdf:
            flags = flags | 512
            wp.printf(
                "Warning: SDF-SDF shape pair buffer overflowed %d > %d.\n",
                sdf_sdf_count[0],
                max_sdf_sdf,
            )
    if contact_count[0] > max_contacts:
        flags = flags | 1024
        wp.printf(
            "Warning: Contact buffer overflowed %d > %d.\n",
            contact_count[0],
            max_contacts,
        )
    if reduction_ht_capacity > 0:
        reduction_ht_active_count = reduction_ht_active_slots[reduction_ht_capacity]
        # Promote before multiplying: large tables can overflow either int32 product.
        if wp.int64(reduction_ht_active_count) * wp.int64(100) >= wp.int64(reduction_ht_capacity) * wp.int64(
            reduction_ht_warn_load_percent
        ):
            flags = flags | 2048
            wp.printf(
                "Warning: Contact reduction hashtable fill ratio exceeded %d%% (%d / %d). "
                "Increase contact_reduction_hashtable_size_factor or max_triangle_pairs.\n",
                reduction_ht_warn_load_percent,
                reduction_ht_active_count,
                reduction_ht_capacity,
            )
        if reduction_ht_insert_failures[0] > 0:
            flags = flags | 4096
            wp.printf(
                "Warning: Contact reduction hashtable insert failures %d. "
                "Increase contact_reduction_hashtable_size_factor or max_triangle_pairs.\n",
                reduction_ht_insert_failures[0],
            )
        if reduction_buffer_overflows[0] > 0:
            flags = flags | 65536
            wp.printf(
                "Warning: Contact reduction buffer overflowed; %d contact candidates were dropped. "
                "Increase max_triangle_pairs.\n",
                reduction_buffer_overflows[0],
            )
    return flags


@wp.kernel(enable_backward=False)
def verify_narrow_phase_buffers(
    broad_phase_count: wp.array[int],
    max_broad_phase: int,
    split_query_count: wp.array[int],
    max_split_query: int,
    gjk_count: wp.array[int],
    max_gjk: int,
    split_gjk_count: wp.array[int],
    max_split_gjk: int,
    split_manifold_count: wp.array[int],
    max_split_manifold: int,
    mesh_count: wp.array[int],
    max_mesh: int,
    triangle_count: wp.array[int],
    max_triangle: int,
    mesh_plane_count: wp.array[int],
    max_mesh_plane: int,
    mesh_mesh_count: wp.array[int],
    max_mesh_mesh: int,
    sdf_sdf_count: wp.array[int],
    max_sdf_sdf: int,
    contact_count: wp.array[int],
    max_contacts: int,
    reduction_ht_active_slots: wp.array[int],
    reduction_ht_capacity: int,
    reduction_ht_insert_failures: wp.array[int],
    reduction_buffer_overflows: wp.array[int],
    reduction_ht_warn_load_percent: int,
):
    """Check for buffer overflows in the collision pipeline."""
    _check_narrow_phase_buffers(
        broad_phase_count,
        max_broad_phase,
        split_query_count,
        max_split_query,
        gjk_count,
        max_gjk,
        split_gjk_count,
        max_split_gjk,
        split_manifold_count,
        max_split_manifold,
        mesh_count,
        max_mesh,
        triangle_count,
        max_triangle,
        mesh_plane_count,
        max_mesh_plane,
        mesh_mesh_count,
        max_mesh_mesh,
        sdf_sdf_count,
        max_sdf_sdf,
        contact_count,
        max_contacts,
        reduction_ht_active_slots,
        reduction_ht_capacity,
        reduction_ht_insert_failures,
        reduction_buffer_overflows,
        reduction_ht_warn_load_percent,
        0,
    )


@wp.kernel(enable_backward=False)
def record_narrow_phase_buffers(
    broad_phase_count: wp.array[int],
    max_broad_phase: int,
    split_query_count: wp.array[int],
    max_split_query: int,
    gjk_count: wp.array[int],
    max_gjk: int,
    split_gjk_count: wp.array[int],
    max_split_gjk: int,
    split_manifold_count: wp.array[int],
    max_split_manifold: int,
    mesh_count: wp.array[int],
    max_mesh: int,
    triangle_count: wp.array[int],
    max_triangle: int,
    mesh_plane_count: wp.array[int],
    max_mesh_plane: int,
    mesh_mesh_count: wp.array[int],
    max_mesh_mesh: int,
    sdf_sdf_count: wp.array[int],
    max_sdf_sdf: int,
    contact_count: wp.array[int],
    max_contacts: int,
    reduction_ht_active_slots: wp.array[int],
    reduction_ht_capacity: int,
    reduction_ht_insert_failures: wp.array[int],
    reduction_buffer_overflows: wp.array[int],
    reduction_ht_warn_load_percent: int,
    capacity_status: wp.array[int],
):
    """Check buffers and retain warning history across subsequent collision calls."""
    capacity_status[0] = _check_narrow_phase_buffers(
        broad_phase_count,
        max_broad_phase,
        split_query_count,
        max_split_query,
        gjk_count,
        max_gjk,
        split_gjk_count,
        max_split_gjk,
        split_manifold_count,
        max_split_manifold,
        mesh_count,
        max_mesh,
        triangle_count,
        max_triangle,
        mesh_plane_count,
        max_mesh_plane,
        mesh_mesh_count,
        max_mesh_mesh,
        sdf_sdf_count,
        max_sdf_sdf,
        contact_count,
        max_contacts,
        reduction_ht_active_slots,
        reduction_ht_capacity,
        reduction_ht_insert_failures,
        reduction_buffer_overflows,
        reduction_ht_warn_load_percent,
        capacity_status[0],
    )


class _HostMesh:
    """A mesh's host buffers, in the form :class:`MeshTopology` reads (``vertices``, ``indices``)."""

    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices


class NarrowPhase:
    """Resolve broad-phase shape pairs into simulation contacts.

    This class orchestrates the narrow-phase collision pipeline by launching the
    appropriate Warp kernels for primitive, mesh, heightfield, and SDF shape
    pairs. It owns the intermediate counters and pair buffers used while
    processing candidate pairs, then writes final contacts through a configurable
    contact writer function.

    Heightfield-only scenes reject cells wholly below the convex contact search
    envelope before triangle queries and pack independent pairs across lanes by
    default. This uses current elevations, scaled bounds and contact/speculative
    gaps without changing the solver or buffer capacities. Set
    ``NEWTON_HEIGHTFIELD_CELL_REJECT=0`` before construction to retain the
    unculled tiled midphase, or ``NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS=0`` to
    disable only the packed mapping. Mixed triangle-mesh and heightfield scenes
    retain the upstream tiled midphase and its cell rejection. Contact ordering
    and redundant manifold point selection may change. On CUDA, a pair whose
    window holds more than 16 cells moves to a second pass that splits its cells
    over 16 lanes of a warp; every cell runs the same tests, so the emitted
    triangles are unchanged and only their order differs. On CPU every pair keeps
    one thread and the original order. ``NEWTON_HEIGHTFIELD_LANE_MIDPHASE=0``
    keeps one thread per pair everywhere.
    Rejection pays when many XY-overlapping cells lie below bodies; profile the
    opt-out for densely touching terrain where these extra bound tests rarely reject.
    Prefer this default for capsule/cylinder terrain contacts; finite-query
    manifold changes target box/convex geometry and need workload validation.

    ``NEWTON_HEIGHTFIELD_FINITE_QUERY=1`` is a separate experimental finite-box
    query/manifold owner for non-speculative heightfield scenes. It preserves
    the compacted triangle stream, generic fallback, reducer and capacities,
    but may choose different physically supported manifold points. Construction
    through CollisionPipeline binds immutable convex hull bounds; refitting a
    bound convex hull requires rebuilding this experimental pipeline.

    Experimental, default-off ``NEWTON_HEIGHTFIELD_GEOMETRIC_CULL=1`` additionally
    rejects separated triangles using enclosing boxes for spheres, capsules,
    regular cylinders and supported cuboids inside the existing cell append
    owner. It requires cell rejection and the stock nonpredictive global
    reducer, independently of the finite-query option. Surviving triangles use
    the selected original generic or finite contact query. Current scale, poses,
    heights, gaps and scaled reducer threshold remain live; unsupported geometry
    and custom writers retain the old path.

    Experimental, default-off ``NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY=1`` splits
    heightfield/sphere and heightfield/regular-cylinder queries into typed kernels
    with the original support, GJK/MPR and manifold math. Other current pair types
    retain the generic query, including barrel cylinders. Admission requires stock
    buffered contact writing without finite queries, deterministic mode,
    speculative contacts or hydroelastic SDFs; no geometry is cached by this option.
    Each typed row scans the triangle stream only when a current heightfield
    pair reaches it: a small kernel reads the current pair list, shape types and
    dimensions every pass and hands rows without pairs a zero stream length.
    ``NEWTON_HEIGHTFIELD_ROW_GATE=0`` restores full scans; contacts are identical.

    Experimental, default-off ``NEWTON_HEIGHTFIELD_CYLINDER_SPLIT=1`` additionally
    separates the admitted regular-cylinder query into MPR, queued GJK and
    manifold/write passes. It requires the primitive query option and preserves
    its geometry and writer restrictions. Scratch results and GJK indices use
    44 bytes per triangle capacity; only the queue counter is reset per call.

    Experimental, default-off ``NEWTON_REDUCER_EXPORT_CLEANUP=1`` retires consumed
    winner slots and keys during stock buffered export, then resets only counters
    at the next collision. Active counts and failure evidence remain available
    through verification. Custom writers, deterministic, speculative, hydroelastic,
    finite-query and pair-CSR routes retain the original clear/export lifecycle.

    Experimental ``NEWTON_HEIGHTFIELD_PAIR_CSR=1`` groups stock-writer-admitted
    raw witnesses by shape pair and publishes at most four for supported
    heightfield/box and heightfield/convex pairs. Selection prioritizes active
    witnesses and then the nearest positive separation intervals. This changes
    the contact manifold; it does not promise an equivalent pointwise Coulomb
    system. Unsupported pairs retain the original global reduction and export.
    Admission requires geometric culling, stock contact writing, no triangle
    meshes, speculative contacts, deterministic mode or hydroelastic SDFs, and
    internal CollisionPipeline construction with unique explicit or generated
    NxN pairs. Raw-pool and membership failures remain sticky capacity errors.

    Experimental ``NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL=1`` additionally permits
    finite-query positive-clearance face witnesses after pair-CSR admission.
    Without admitted CSR, direct and reduced finite queries retain their
    conservative generic fallback. Both options are disabled by default and
    are read during pipeline construction; geometry limitations and immutable
    hull binding requirements of the finite query continue to apply.
    """

    def __init__(
        self,
        *,
        max_candidate_pairs: int,
        max_triangle_pairs: int = 1000000,
        max_mesh_mesh_pairs: int | None = None,
        max_mesh_plane_pairs: int | None = None,
        max_heightfield_pairs: int | None = None,
        reduce_contacts: bool = True,
        device: Devicelike | None = None,
        shape_aabb_lower: wp.array[wp.vec3] | None = None,
        shape_aabb_upper: wp.array[wp.vec3] | None = None,
        shape_voxel_resolution: wp.array[wp.vec3i] | None = None,
        contact_writer_warp_func: Any | None = None,
        box_box_sat: bool = False,
        hydroelastic_sdf: HydroelasticSDF | None = None,
        has_meshes: bool = True,
        has_heightfields: bool = False,
        use_lean_gjk_mpr: bool = False,
        convex_support_acceleration: bool = True,
        has_generic_convex_pairs: bool = True,
        sparse_gjk_pairs: bool | None = None,
        split_gjk_mpr: bool = False,
        candidate_pair_work_estimate: int | None = None,
        mesh_sdf_texture_only: bool = False,
        mesh_sdf_identity_scale_only: bool = False,
        sdf_texture_paired_samples: bool = True,
        deterministic: bool = False,
        contact_max: int | None = None,
        verify_buffers: bool = True,
        contact_reduction_hashtable_size_factor: float = 0.25,
        speculative: bool = False,
        contact_writer_supports_speculative: bool = False,
        contact_export: Literal["auto", "tiled", "compact"] | None = None,
        heightfield_primitive_query: bool | None = None,
    ) -> None:
        """
        Initialize NarrowPhase with pre-allocated buffers.

        Args:
            max_candidate_pairs: Maximum number of candidate pairs from broad phase
            max_triangle_pairs: Maximum number of triangle pairs for mesh and
                heightfield collisions (conservative estimate).
            max_mesh_mesh_pairs: Maximum number of routed mesh-SDF pairs. Defaults
                to ``max_candidate_pairs``.
            max_mesh_plane_pairs: Maximum number of routed mesh-plane pairs. Defaults
                to ``max_candidate_pairs``.
            max_heightfield_pairs: Upper bound on routed heightfield pairs. On CUDA
                the lane midphase stores up to half this many long pairs for its lane
                pass, at most 65536 when no bound is given, and walks further ones in
                place. Defaults to ``max_candidate_pairs``.
            reduce_contacts: Whether to reduce contacts for mesh-mesh and mesh-plane collisions.
                When True, uses shared memory contact reduction to select representative contacts.
                This improves performance and stability for meshes with many vertices. Defaults to True.
            contact_export: Experimental reduced-contact export mapping. ``auto`` uses compact
                workers for supported CUDA reducers; ``tiled`` retains cooperative export.
                ``compact`` requires admission. ``None`` preserves the legacy environment choice.
            heightfield_primitive_query: Request typed reduced terrain queries subject to
                writer and pipeline admission. ``None`` preserves the environment choice.
            device: Device to allocate buffers on
            shape_aabb_lower: Optional external AABB lower bounds array (if provided, AABBs won't be computed internally)
            shape_aabb_upper: Optional external AABB upper bounds array (if provided, AABBs won't be computed internally)
            shape_voxel_resolution: Optional per-shape voxel resolution array used for mesh/SDF and
                hydroelastic contact processing. When omitted, :class:`~newton.CollisionPipeline`
                supplies the model's table. A supplied table must match that model's shape count and device.
            contact_writer_warp_func: Optional custom contact writer function (first arg: ContactData, second arg: custom struct type)
            hydroelastic_sdf: Optional SDF hydroelastic instance. Set is_hydroelastic=True on shapes to enable hydroelastic collisions.
            has_meshes: Whether the scene contains any mesh shapes (GeoType.MESH). When False, mesh-related
                kernel launches are skipped, improving performance for scenes with only primitive shapes.
                Defaults to True for safety. Set to False when constructing from a model with no meshes.
            has_heightfields: Whether the scene contains any heightfield shapes (GeoType.HFIELD). When True,
                heightfield collision buffers and kernels are allocated. Defaults to False.
            convex_support_acceleration: Whether the model contains cooked directional support data for at least
                one convex mesh. When False, retain the original exhaustive support map without acceleration metadata loads. Defaults to True for custom launch compatibility.
            has_generic_convex_pairs: Whether any candidate pair can require
                generic GJK/MPR processing. Set to False only from a complete
                scene-topology proof; this omits the GJK/MPR launch entirely.
            sparse_gjk_pairs: Whether GJK routing preserves broad-phase pair
                indices instead of compacting its work buffer. Defaults to
                automatic enablement for large CUDA candidate buffers.
            split_gjk_mpr: Whether to split generic convex collision into MPR,
                GJK, and manifold kernels. Ignored on CPU and when generic
                convex processing is unnecessary.
            candidate_pair_work_estimate: Static upper bound on pairs that the
                broad phase can emit. Defaults to ``max_candidate_pairs``.
            mesh_sdf_texture_only: Whether every participating mesh SDF has a texture representation,
                allowing BVH fallback branches to be removed from mesh/SDF kernels.
            mesh_sdf_identity_scale_only: Whether every participating texture SDF is queried with
                identity scale, allowing scale conversion branches to be removed. Shapes whose SDF
                has no baked scale must then keep unit ``shape_scale``.
            sdf_texture_paired_samples: Whether texture SDFs store adjacent x samples together.
                This is model-wide so mesh-SDF kernels require only two bounded static variants.
            deterministic: Make contact generation and ordering independent of
                GPU thread scheduling. Adds deterministic hydroelastic atomics
                and a radix sort + gather pass.
            contact_max: Maximum number of contacts for the deterministic sort buffer.
                Must match the ``contact_pair`` array size passed to :meth:`launch`.
                Defaults to ``max_candidate_pairs``.  Set this to a larger value when
                a single candidate pair can emit multiple contacts (e.g. up to 4 for
                primitive multi-contact paths). ``0`` allocates no sort buffer, for
                callers that only use :meth:`launch_custom_write` and sort themselves.
            verify_buffers: When True (the default), launch a ``dim=[1]``
                diagnostic kernel (:func:`record_narrow_phase_buffers`) at the
                end of :meth:`launch` that compares each public counter on this
                class (``gjk_candidate_pairs_count``, ``split_gjk_work_count``,
                ``split_manifold_work_count``, ``shape_pairs_mesh_count``,
                ``triangle_pairs_count``, ``shape_pairs_mesh_plane_count``,
                ``shape_pairs_mesh_mesh_count``, ``shape_pairs_sdf_sdf_count``)
                and the output ``contact_count`` against the capacity of its
                backing array, and checks the global contact reducer hashtable
                fill/failure counters when reduction is enabled, printing
                ``wp.printf`` warnings on overflow or critical hashtable load.
                These warnings are also latched for :meth:`buffer_capacity_status`
                and :meth:`check_buffer_capacity`, including during graph replay.
                The hash-load warning does not necessarily indicate lost work.
                Overhead is one extra kernel
                launch per collision pass (roughly a few µs of launch latency on
                CUDA; the kernel body is a handful of scalar comparisons on one
                thread).  Disable in hot loops or CUDA graph capture once buffer
                sizes are known to be adequate.
            contact_reduction_hashtable_size_factor: Multiplier applied to
                ``max_triangle_pairs`` when allocating the global contact
                reduction hashtable. Increase this if hashtable fill/failure
                warnings appear. Defaults to ``0.25`` for memory compatibility.
            speculative: Whether the caller provides velocity-expanded search
                gaps and exact speculative-contact data. Defaults to False.
            contact_writer_supports_speculative: Whether a custom contact writer performs exact speculative-contact
                admission. Required when ``speculative`` and ``contact_writer_warp_func`` are both provided.
        """
        if os.environ.get("NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD", "0") != "0":
            raise ValueError(
                "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD is retired; unset it or set it to 0. "
                "Use the original experiment checkpoint to reproduce that exporter."
            )
        self.max_candidate_pairs = max_candidate_pairs
        self.max_triangle_pairs = max_triangle_pairs
        if max_mesh_mesh_pairs is not None and max_mesh_mesh_pairs < 0:
            raise ValueError("max_mesh_mesh_pairs must be non-negative or None")
        if max_mesh_plane_pairs is not None and max_mesh_plane_pairs < 0:
            raise ValueError("max_mesh_plane_pairs must be non-negative or None")
        if max_heightfield_pairs is not None and max_heightfield_pairs < 0:
            raise ValueError("max_heightfield_pairs must be non-negative or None")
        self.max_mesh_mesh_pairs = min(
            max_candidate_pairs,
            max_candidate_pairs if max_mesh_mesh_pairs is None else max_mesh_mesh_pairs,
        )
        self.max_mesh_plane_pairs = min(
            max_candidate_pairs,
            max_candidate_pairs if max_mesh_plane_pairs is None else max_mesh_plane_pairs,
        )
        self.max_heightfield_pairs = min(
            max_candidate_pairs,
            max_candidate_pairs if max_heightfield_pairs is None else max_heightfield_pairs,
        )
        self.device = device
        # The welded topology of the meshes (see mesh_topology.py), set by each launch: the model's, passed by the
        # pipelines, or this narrow phase's own (see :meth:`_own_mesh_data`).
        self.mesh_topology = empty_mesh_topology(device)
        self.reduce_contacts = reduce_contacts
        self.has_meshes = has_meshes
        self._empty_mesh_cells = None
        self._own_mesh_key = None
        self._own_mesh_data_value = None
        self.has_heightfields = has_heightfields
        # Mixed mesh/heightfield scenes retain the original full midphase;
        # heightfield-only rejection adds no buffers or cached geometry.
        cell_reject = os.environ.get("NEWTON_HEIGHTFIELD_CELL_REJECT", "1")
        if cell_reject not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_CELL_REJECT must be 0 or 1")
        self._heightfield_cell_reject_requested = cell_reject == "1"
        self._heightfield_cell_reject = self._heightfield_cell_reject_requested and has_heightfields and not has_meshes
        packed_pairs = os.environ.get("NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS", "1")
        if packed_pairs not in ("0", "1"):
            raise ValueError("NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS must be 0 or 1")
        self._heightfield_packed_pairs = packed_pairs == "1" and self._heightfield_cell_reject
        finite_query = os.environ.get("NEWTON_HEIGHTFIELD_FINITE_QUERY", "0")
        if finite_query not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_FINITE_QUERY must be 0 or 1")
        self._heightfield_finite_query = finite_query == "1" and has_heightfields and not has_meshes and not speculative
        geometric_cull = os.environ.get("NEWTON_HEIGHTFIELD_GEOMETRIC_CULL", "0")
        if geometric_cull not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_GEOMETRIC_CULL must be 0 or 1")
        geometric_writer_supported = contact_writer_warp_func is None
        if geometric_cull == "1" and not geometric_writer_supported:
            from ..sim.collide import write_contact as stock_contact_writer  # noqa: PLC0415

            geometric_writer_supported = contact_writer_warp_func is stock_contact_writer
        self._heightfield_geometric_cull = (
            geometric_cull == "1"
            and self._heightfield_cell_reject
            and not speculative
            and reduce_contacts
            and geometric_writer_supported
        )
        pair_csr = os.environ.get("NEWTON_HEIGHTFIELD_PAIR_CSR", "0")
        if pair_csr not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_PAIR_CSR must be 0 or 1")
        pair_csr_writer_supported = False
        if pair_csr == "1" and contact_writer_warp_func is not None:
            from ..sim.collide import write_contact as pair_csr_stock_writer  # noqa: PLC0415

            pair_csr_writer_supported = contact_writer_warp_func is pair_csr_stock_writer
        self._heightfield_pair_csr_requested = (
            pair_csr == "1"
            and self._heightfield_geometric_cull
            and self._heightfield_finite_query
            and not has_meshes
            and not speculative
            and not deterministic
            and hydroelastic_sdf is None
            and pair_csr_writer_supported
        )
        self._heightfield_pair_csr = False
        pair_csr_shell = os.environ.get("NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL", "0")
        if pair_csr_shell not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL must be 0 or 1")
        self._heightfield_pair_csr_shell_requested = pair_csr_shell == "1"
        self._heightfield_pair_csr_shell = False
        self._pair_csr = None
        primitive_query = os.environ.get("NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY", "0")
        if heightfield_primitive_query is not None:
            if not isinstance(heightfield_primitive_query, bool):
                raise ValueError("heightfield_primitive_query must be a bool or None")
            primitive_query = str(int(heightfield_primitive_query))
        if primitive_query not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY must be 0 or 1")
        primitive_writer_supported = contact_writer_warp_func is None
        if primitive_query == "1" and not primitive_writer_supported:
            from ..sim.collide import write_contact as primitive_stock_writer  # noqa: PLC0415

            primitive_writer_supported = contact_writer_warp_func is primitive_stock_writer
        self._heightfield_primitive_query = (
            primitive_query == "1"
            and has_heightfields
            and reduce_contacts
            and not self._heightfield_finite_query
            and not self._heightfield_pair_csr_requested
            and not speculative
            and not deterministic
            and hydroelastic_sdf is None
            and primitive_writer_supported
        )
        self._heightfield_primitive_kernels = ()
        if self._heightfield_primitive_query:
            # The typed rows follow HEIGHTFIELD_TYPED_ROWS, so index 1 is the cylinder
            # query (the cylinder split replaces it); the generic fallback comes last
            # and excludes every typed primitive.
            self._heightfield_primitive_kernels = (
                *(create_mesh_triangle_contacts_to_reducer_kernel(row) for row in HEIGHTFIELD_TYPED_ROWS),
                # Convex hulls stay in the fallback scan (a typed kernel is one more
                # pass over every candidate pair: Go2 rough +2%); with the closed-form
                # row on, the fallback runs that row for them.
                create_mesh_triangle_contacts_to_reducer_kernel(exclude_primitives=True),
            )
        row_gate = os.environ.get("NEWTON_HEIGHTFIELD_ROW_GATE", "1")
        if row_gate not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_ROW_GATE must be 0 or 1")
        # Each typed row scans the triangle stream only when a current pair can reach it.
        self._heightfield_row_gate = row_gate == "1" and self._heightfield_primitive_query
        cylinder_split = os.environ.get("NEWTON_HEIGHTFIELD_CYLINDER_SPLIT", "0")
        if cylinder_split not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_CYLINDER_SPLIT must be 0 or 1")
        # The closed-form cylinder row replaces the GJK split of the typed cylinder query.
        analytic_cylinder = (
            self._heightfield_primitive_query
            and analytic_primitives_default()
            and int(GeoType.CYLINDER) in ANALYTIC_PRIMITIVE_QUERIES
        )
        self._heightfield_cylinder_split = (
            cylinder_split == "1" and self._heightfield_primitive_query and not analytic_cylinder
        )
        self._heightfield_cylinder_split_kernels = (
            create_heightfield_cylinder_split_kernels() if self._heightfield_cylinder_split else ()
        )
        export_cleanup = os.environ.get("NEWTON_REDUCER_EXPORT_CLEANUP", "0")
        if export_cleanup not in ("0", "1"):
            raise ValueError("NEWTON_REDUCER_EXPORT_CLEANUP must be 0 or 1")
        cleanup_writer_supported = contact_writer_warp_func is None
        if not cleanup_writer_supported:
            from ..sim.collide import write_contact as cleanup_stock_writer  # noqa: PLC0415

            cleanup_writer_supported = contact_writer_warp_func is cleanup_stock_writer
        self._reducer_export_cleanup = (
            export_cleanup == "1"
            and reduce_contacts
            and (has_meshes or has_heightfields)
            and cleanup_writer_supported
            and not deterministic
            and not speculative
            and hydroelastic_sdf is None
            and not self._heightfield_finite_query
            and not self._heightfield_pair_csr_requested
        )
        if contact_export not in (None, "auto", "tiled", "compact"):
            raise ValueError(f"contact_export must be 'auto', 'tiled', 'compact' or None, got {contact_export!r}")
        self.contact_export = contact_export
        compact_supported = (
            wp.get_device(device).is_cuda
            and reduce_contacts
            and (has_meshes or has_heightfields)
            and not deterministic
            and not self._heightfield_pair_csr_requested
        )
        if contact_export is None:
            compact_export = os.environ.get("NEWTON_REDUCER_COMPACT_EXPORT", "0")
            if compact_export not in ("0", "1"):
                raise ValueError("NEWTON_REDUCER_COMPACT_EXPORT must be 0 or 1")
            self._reducer_compact_export = compact_export == "1" and compact_supported
        else:
            compact_supported = (
                compact_supported
                and cleanup_writer_supported
                and not speculative
                and hydroelastic_sdf is None
                and not self._heightfield_finite_query
            )
            if contact_export == "compact" and not compact_supported:
                raise ValueError(
                    "contact_export='compact' requires a CUDA mesh/heightfield reducer with a stock writer, "
                    "without deterministic, speculative, hydroelastic or alternate heightfield export"
                )
            self._reducer_compact_export = contact_export != "tiled" and compact_supported
        # Requested terrain and reducer routes (environment or explicit argument), reported by
        # CollisionPipeline.get_collision_selection(); the admitted routes are the fields above.
        self._route_requests = {
            "finite_query": finite_query == "1",
            "geometric_cull": geometric_cull == "1",
            "pair_csr": pair_csr == "1",
            "primitive_query": primitive_query == "1",
            "primitive_query_source": "environment" if heightfield_primitive_query is None else "argument",
            "cylinder_split": cylinder_split == "1",
            "export_cleanup": export_cleanup == "1",
            "row_gate": row_gate == "1",
        }
        lane_midphase = os.environ.get("NEWTON_HEIGHTFIELD_LANE_MIDPHASE", "1")
        if lane_midphase not in ("0", "1"):
            raise ValueError("NEWTON_HEIGHTFIELD_LANE_MIDPHASE must be 0 or 1")
        self._route_requests["lane_midphase"] = lane_midphase == "1"
        # Packed cell rejection owns every heightfield pair with one thread; the
        # lane midphase keeps that pass and moves the cells of long pairs to lanes.
        self._heightfield_lane_midphase = lane_midphase == "1" and self._heightfield_packed_pairs
        # CPU kernels run one lane per block, in order (Warp GH-1413): defer nothing
        # there, which keeps the original triangle order and needs no deferred storage.
        lane_device = wp.get_device(device)
        self._heightfield_lane_small_cells = _LANE_SMALL_CELLS if lane_device.is_cuda else -1
        self._heightfield_lane_width = _LANE_WIDTH if lane_device.is_cuda else 1
        # Deferred pairs that fit when the schedule defers: half the routed-pair bound, or
        # at most _LANE_DEFAULT_CAPACITY when the caller gave none.
        self._heightfield_lane_capacity = (self.max_heightfield_pairs + 1) // 2
        if max_heightfield_pairs is None:
            self._heightfield_lane_capacity = min(self._heightfield_lane_capacity, _LANE_DEFAULT_CAPACITY)
        self._heightfield_lane_deferred = None
        # The lane midphase's kernels and struct arguments, with the route and arrays they were built for.
        self._heightfield_lane_launch = None
        self._finite_bounds = None
        self._finite_source = None
        self.convex_support_acceleration = convex_support_acceleration
        self.mesh_sdf_texture_only = mesh_sdf_texture_only
        self.has_generic_convex_pairs = has_generic_convex_pairs
        self.sdf_texture_paired_samples = sdf_texture_paired_samples
        self.mesh_sdf_identity_scale_only = mesh_sdf_identity_scale_only
        self.deterministic = deterministic
        self.verify_buffers = verify_buffers
        self.contact_pool = None
        """The contact rows of the collision pipeline using this narrow phase, set by the pipeline; their captured
        launches report through :meth:`buffer_capacity_status` (``contact_launches``)."""
        self.speculative = speculative
        if speculative and contact_writer_warp_func is not None and not contact_writer_supports_speculative:
            raise ValueError(
                "A custom contact writer used with speculative=True must set "
                "contact_writer_supports_speculative=True and perform exact speculative-contact admission"
            )
        if speculative and hydroelastic_sdf is not None:
            raise NotImplementedError("Speculative contact generation does not yet support hydroelastic SDF contacts")
        device_obj = wp.get_device(device)
        # Contact reduction requires either meshes or heightfields (the
        # mesh/heightfield-triangle path feeds the global reducer, so
        # heightfield-only scenes still benefit from reduction).
        if reduce_contacts and not (has_meshes or has_heightfields):
            self.reduce_contacts = False

        # Determine if we're using external AABBs
        self.external_aabb = shape_aabb_lower is not None and shape_aabb_upper is not None

        if self.external_aabb:
            # Use provided AABB arrays
            self.shape_aabb_lower = shape_aabb_lower
            self.shape_aabb_upper = shape_aabb_upper
        else:
            # Create empty AABB arrays (won't be used)
            with wp.ScopedDevice(device):
                self.shape_aabb_lower = wp.zeros(0, dtype=wp.vec3, device=device)
                self.shape_aabb_upper = wp.zeros(0, dtype=wp.vec3, device=device)
        self._contact_sort_shape_index_bits = (
            contact_sort_shape_index_bits(self.shape_aabb_lower.shape[0])
            if self.external_aabb
            else CONTACT_SORT_MAX_SHAPE_INDEX_BITS
        )
        self._contact_sort_sub_key_bits = (
            CONTACT_SORT_SUB_KEY_BITS
            if has_meshes or has_heightfields or hydroelastic_sdf is not None
            else CONTACT_SORT_CONVEX_SUB_KEY_BITS
        )
        self.shape_voxel_resolution = shape_voxel_resolution

        # Determine the writer function
        if contact_writer_warp_func is None:
            writer_func = _write_contact_simple_speculative if speculative else write_contact_simple
        else:
            writer_func = contact_writer_warp_func
        # Retain the resolved writer for optional typed support successors.
        self._convex_writer_func = writer_func

        # CPU kernels currently observe ``wp.block_dim() == 1`` regardless
        # of the plain ``wp.launch(..., block_dim=N)`` parameter (Warp
        # GH-1413). Keep the mesh-convex midphase launch grid, tile shape,
        # and kernel-side ``wp.block_dim()`` in sync on CPU.
        # Dense mesh queries benefit from wider cooperative result batches;
        # heightfields only use lane zero and retain the leaner launch.
        self.tile_size_mesh_convex = 1 if device_obj.is_cpu else (128 if has_heightfields else 256)
        # Must match ``MESH_SDF_BLOCK_DIM`` in sdf_contact.py: the mesh-SDF
        # kernels assume ``wp.block_dim()`` equals that constant so the
        # tile-stack overflow margin (``STACK_CAPACITY = 2 *
        # MESH_SDF_BLOCK_DIM``) is correctly sized. Re-use the constant
        # rather than duplicating the value so the two can't drift.
        self.tile_size_mesh_mesh = MESH_SDF_BLOCK_DIM
        assert self.tile_size_mesh_mesh == MESH_SDF_BLOCK_DIM, (
            "mesh-SDF tile launches must use block_dim == MESH_SDF_BLOCK_DIM"
        )
        self.tile_size_mesh_plane = 512
        # Generic block dim for non-tile-stack kernels (primitive /
        # GJK-MPR / export). Not used for the mesh-SDF tile launches,
        # which use ``self.tile_size_mesh_mesh`` above.
        #
        # Plain ``wp.launch`` does not auto-clamp ``block_dim`` on CPU like
        # ``wp.launch_tiled`` does. Match the kernel-observed value so
        # strided-loop and tile-index calculations cannot run past the CPU
        # launch geometry.
        self.block_dim = 1 if device_obj.is_cpu else 128
        if sparse_gjk_pairs is None:
            sparse_gjk_pairs = device_obj.is_cuda and max_candidate_pairs >= _SPARSE_GJK_PAIR_CAPACITY_THRESHOLD
        self.sparse_gjk_pairs = sparse_gjk_pairs
        if candidate_pair_work_estimate is None:
            candidate_pair_work_estimate = max_candidate_pairs
        if candidate_pair_work_estimate < 0:
            raise ValueError("candidate_pair_work_estimate must be non-negative or None")
        self.split_gjk_mpr = device_obj.is_cuda and has_generic_convex_pairs and split_gjk_mpr
        self._use_lean_gjk_mpr = use_lean_gjk_mpr
        self._coherent_cache = None
        self._convex_method_choice = None
        self._coherent_query_kernels = None
        # Create the appropriate kernel variants
        # Primitive kernel handles lightweight primitives and routes remaining pairs
        self.primitive_kernel = create_narrow_phase_primitive_kernel(
            writer_func,
            speculative=speculative,
            sparse_gjk_pairs=self.sparse_gjk_pairs,
            hydroelastic_enabled=hydroelastic_sdf is not None,
            box_box_sat=box_box_sat,
        )
        # GJK/MPR kernel handles remaining convex-convex pairs. Only models with cooked
        # support data select the accelerated support function; other models
        # retain the original exhaustive path.
        if use_lean_gjk_mpr:
            # Use lean support function (CONVEX_MESH, BOX, SPHERE only) and lean post-processing
            # (skip axial shape rolling stabilization) to reduce GPU i-cache pressure
            self.narrow_phase_kernel = create_narrow_phase_kernel_gjk_mpr(
                self.external_aabb,
                writer_func,
                support_func=support_map_lean_accelerated if convex_support_acceleration else support_map_lean,
                post_process_contact=post_process_minkowski_only,
                speculative=speculative,
            )
        else:
            self.narrow_phase_kernel = create_narrow_phase_kernel_gjk_mpr(
                self.external_aabb,
                writer_func,
                support_func=support_map_accelerated if convex_support_acceleration else support_map,
                speculative=speculative,
            )

        if self.split_gjk_mpr:
            if use_lean_gjk_mpr:
                split_support = support_map_lean_accelerated if convex_support_acceleration else support_map_lean
            else:
                split_support = support_map_accelerated if convex_support_acceleration else support_map
            split_post_process = post_process_minkowski_only if use_lean_gjk_mpr else None
            (
                self.narrow_phase_mpr_kernel,
                self.narrow_phase_gjk_kernel,
                self.narrow_phase_manifold_kernel,
            ) = create_narrow_phase_kernels_gjk_mpr_split(
                self.external_aabb,
                writer_func,
                support_func=split_support,
                post_process_contact=split_post_process,
                speculative=speculative,
            )
        else:
            self.narrow_phase_mpr_kernel = None
            self.narrow_phase_gjk_kernel = None
            self.narrow_phase_manifold_kernel = None
        # Mesh triangles against spheres and capsules take their closed form in a lean kernel of their own; the
        # generic reducing query (or the heightfield primitive route's fallback) takes every other pair.
        self._mesh_round_kernel = None
        self._mesh_generic_kernel = mesh_triangle_contacts_to_reducer_kernel
        # The generic route's other pairs: a first pass with the plane cull and the closed-form face case, which
        # compacts the rest into a queue for GJK/MPR (the generic query without its cull).
        self._mesh_face_kernel = None
        self._mesh_queued_kernel = None
        self._mesh_queued_pairs = None
        self._mesh_queued_count = None
        if has_meshes and self.reduce_contacts:
            self._mesh_round_kernel = create_mesh_triangle_contacts_to_reducer_kernel(mesh_round=MESH_ROUND_ONLY)
            self._mesh_generic_kernel = create_mesh_triangle_contacts_to_reducer_kernel(mesh_round=MESH_ROUND_NONE)
            self._mesh_face_kernel = create_mesh_face_contacts_kernel()
            self._mesh_queued_kernel = create_mesh_triangle_contacts_to_reducer_kernel(
                mesh_round=MESH_ROUND_NONE, plane_cull=False
            )
            if self._heightfield_primitive_kernels:
                self._heightfield_primitive_kernels = (
                    *self._heightfield_primitive_kernels[:-1],
                    create_mesh_triangle_contacts_to_reducer_kernel(
                        exclude_primitives=True, mesh_round=MESH_ROUND_NONE
                    ),
                )
        # Mesh triangles that write their contacts directly (the reducer is for soups): the same three passes with
        # the narrow phase's writer, and heightfield triangles in a query of their own beside them.
        self._mesh_round_direct = None
        self._mesh_face_direct = None
        self._mesh_queued_direct = None
        self._heightfield_triangle_kernel = None
        self._heightfield_reducer_kernel = None
        if has_meshes:
            self._mesh_round_direct = create_mesh_triangle_contacts_to_reducer_kernel(
                mesh_round=MESH_ROUND_ONLY,
                writer_func=writer_func,
                writer_data_type=Any,
                kernel_name="mesh_round_contacts_kernel",
            )
            self._mesh_face_direct = create_mesh_face_contacts_kernel(
                writer_func, Any, queue_heightfields=False, kernel_name="mesh_face_contacts_direct_kernel"
            )
            self._mesh_queued_direct = create_mesh_triangle_contacts_to_reducer_kernel(
                mesh_round=MESH_ROUND_NONE,
                plane_cull=False,
                surfaces=SURFACES_MESH,
                writer_func=writer_func,
                writer_data_type=Any,
                kernel_name="mesh_triangle_contacts_queued_kernel",
            )
            if has_heightfields:
                self._heightfield_triangle_kernel = create_narrow_phase_process_mesh_triangle_contacts_kernel(
                    writer_func, convex_support_acceleration=convex_support_acceleration, heightfields_only=True
                )
                if self.reduce_contacts:
                    self._heightfield_reducer_kernel = create_mesh_triangle_contacts_to_reducer_kernel(
                        exclude_primitives=bool(self._heightfield_primitive_kernels),
                        mesh_round=MESH_ROUND_NONE,
                        surfaces=SURFACES_HEIGHTFIELD,
                    )
            self._mesh_queued_pairs = wp.empty(max_triangle_pairs, dtype=wp.vec3i, device=device)
            self._mesh_queued_count = wp.zeros(1, dtype=int, device=device)
        # Create triangle contacts kernel when meshes or heightfields are present
        if has_meshes or has_heightfields:
            self.mesh_triangle_contacts_kernel = create_narrow_phase_process_mesh_triangle_contacts_kernel(
                writer_func,
                convex_support_acceleration=convex_support_acceleration,
            )
        else:
            self.mesh_triangle_contacts_kernel = None
        if self._heightfield_finite_query:
            from .heightfield_finite import create_query_kernel  # noqa: PLC0415

            self._finite_direct = create_query_kernel(writer_func)
            self._finite_reducer = create_query_kernel(write_contact_to_reducer)
            self._finite_direct_fallback = create_narrow_phase_process_mesh_triangle_contacts_kernel(
                writer_func, convex_support_acceleration=convex_support_acceleration, skip_marked=True
            )
            self._finite_reducer_fallback = create_mesh_triangle_contacts_to_reducer_kernel(skip_marked=True)

        # Create mesh-specific kernels only when has_meshes=True
        if has_meshes:
            # Create mesh-plane kernel.
            # When reducing, use multi-block load balancing and write contacts to the
            # global reducer buffer (same path as mesh-mesh and mesh-triangle).
            if self.reduce_contacts:
                self.mesh_plane_contacts_kernel = create_narrow_phase_process_mesh_plane_contacts_kernel(
                    write_contact_to_reducer,
                    reduce_contacts=True,
                )
            else:
                self.mesh_plane_contacts_kernel = create_narrow_phase_process_mesh_plane_contacts_kernel(
                    writer_func,
                )
            if self.reduce_contacts:
                self.mesh_mesh_contacts_kernel = create_narrow_phase_process_mesh_mesh_contacts_kernel(
                    write_contact_to_reducer,
                    enable_heightfields=has_heightfields,
                    reduce_contacts=True,
                    speculative=speculative,
                    use_texture_sdf_only=self.mesh_sdf_texture_only,
                    use_identity_sdf_scale=self.mesh_sdf_identity_scale_only,
                    deterministic_reduction=deterministic,
                )
                self.mesh_mesh_contacts_kernel_precomputed = create_narrow_phase_process_mesh_mesh_contacts_kernel(
                    write_contact_to_reducer,
                    enable_heightfields=has_heightfields,
                    reduce_contacts=True,
                    use_precomputed_edge_data=True,
                    use_texture_sdf_only=self.mesh_sdf_texture_only,
                    use_identity_sdf_scale=self.mesh_sdf_identity_scale_only,
                    speculative=speculative,
                    deterministic_reduction=deterministic,
                )
            else:
                self.mesh_mesh_contacts_kernel = create_narrow_phase_process_mesh_mesh_contacts_kernel(
                    writer_func,
                    enable_heightfields=has_heightfields,
                    speculative=speculative,
                    use_texture_sdf_only=self.mesh_sdf_texture_only,
                    use_identity_sdf_scale=self.mesh_sdf_identity_scale_only,
                )
                self.mesh_mesh_contacts_kernel_precomputed = create_narrow_phase_process_mesh_mesh_contacts_kernel(
                    writer_func,
                    enable_heightfields=has_heightfields,
                    use_precomputed_edge_data=True,
                    use_texture_sdf_only=self.mesh_sdf_texture_only,
                    use_identity_sdf_scale=self.mesh_sdf_identity_scale_only,
                    speculative=speculative,
                )
        else:
            self.mesh_plane_contacts_kernel = None
            self.mesh_mesh_contacts_kernel = None
            self.mesh_mesh_contacts_kernel_precomputed = None

        # Create global contact reduction kernels for mesh/heightfield-triangle
        # contacts (mirror the predicate used to gate ``self.reduce_contacts``
        # above so heightfield-only scenes also get the reducer allocated).
        if self.reduce_contacts and (has_meshes or has_heightfields):
            # Global contact reducer uses hardcoded BETA_THRESHOLD (0.1mm) same as shared-memory reduction
            # Slot layout: NUM_SPATIAL_DIRECTIONS spatial + 1 max-depth = VALUES_PER_KEY slots per key
            self.export_reduced_contacts_kernel = create_export_reduced_contacts_kernel(
                writer_func, clear_after_export=self._reducer_export_cleanup, compact=self._reducer_compact_export
            )
            # Global contact reducer for all mesh contact types
            self.global_contact_reducer = GlobalContactReducer(
                max_triangle_pairs,
                device=device,
                deterministic=deterministic,
                hashtable_size_factor=contact_reduction_hashtable_size_factor,
                enable_contact_reclamation=speculative,
            )
        else:
            self.export_reduced_contacts_kernel = None
            self.global_contact_reducer = None

        self.hydroelastic_sdf = hydroelastic_sdf
        if self.hydroelastic_sdf is not None:
            self.hydroelastic_sdf._validate_deterministic(deterministic)

        # Pre-allocate all intermediate buffers.
        # Counters live in one consolidated array for efficient zeroing.
        with wp.ScopedDevice(device):
            has_mesh_like = has_meshes or has_heightfields
            n = 0  # counter index
            gjk_idx = n
            n += 1
            split_gjk_idx = n if self.split_gjk_mpr else None
            n += 2 if self.split_gjk_mpr else 0
            cylinder_gjk_idx = n if self._heightfield_cylinder_split else None
            n += 1 if self._heightfield_cylinder_split else 0
            sdf_sdf_idx = n
            n += 1
            mesh_like_idx = n if has_mesh_like else None
            n += 4 if has_mesh_like else 0  # mesh_like pairs, triangle pairs, mesh triangle candidates, hit cells
            mesh_only_idx = n if has_meshes else None
            n += 3 if has_meshes else 0  # mesh_plane, mesh_plane_vtx, mesh_mesh
            mesh_weight_idx = n if has_meshes and self.reduce_contacts else None
            n += 2 if mesh_weight_idx is not None else 0  # mesh-plane vertices, mesh-mesh edges
            row_gate_idx = n if self._heightfield_row_gate else None
            n += HEIGHTFIELD_ROW_COUNT if row_gate_idx is not None else 0  # typed terrain row lengths
            lane_idx = n if self._heightfield_lane_midphase else None
            n += 1 if lane_idx is not None else 0  # deferred terrain pairs
            c = wp.zeros(n, dtype=wp.int32, device=device)
            self._counter_array = c
            # Separate storage: per-frame counter resets must not erase history.
            self._buffer_capacity_status = wp.zeros(1, dtype=wp.int32, device=device)

            self.gjk_candidate_pairs_count = c[gjk_idx : gjk_idx + 1]
            self.split_gjk_work_count = c[split_gjk_idx : split_gjk_idx + 1] if self.split_gjk_mpr else None
            self.split_manifold_work_count = c[split_gjk_idx + 1 : split_gjk_idx + 2] if self.split_gjk_mpr else None
            self._heightfield_cylinder_gjk_count = (
                c[cylinder_gjk_idx : cylinder_gjk_idx + 1] if self._heightfield_cylinder_split else None
            )
            self.shape_pairs_sdf_sdf_count = c[sdf_sdf_idx : sdf_sdf_idx + 1]
            self.shape_pairs_mesh_count = c[mesh_like_idx : mesh_like_idx + 1] if has_mesh_like else None
            self.triangle_pairs_count = c[mesh_like_idx + 1 : mesh_like_idx + 2] if has_mesh_like else None
            self._mesh_candidate_count = c[mesh_like_idx + 2 : mesh_like_idx + 3] if has_mesh_like else None
            self._mesh_hit_count = c[mesh_like_idx + 3 : mesh_like_idx + 4] if has_mesh_like else None
            self.shape_pairs_mesh_plane_count = c[mesh_only_idx : mesh_only_idx + 1] if has_meshes else None
            self.mesh_plane_vertex_total_count = c[mesh_only_idx + 1 : mesh_only_idx + 2] if has_meshes else None
            self.mesh_plane_total_weight = (
                c[mesh_weight_idx : mesh_weight_idx + 1] if mesh_weight_idx is not None else None
            )
            self.mesh_mesh_total_weight = (
                c[mesh_weight_idx + 1 : mesh_weight_idx + 2] if mesh_weight_idx is not None else None
            )
            self.shape_pairs_mesh_mesh_count = c[mesh_only_idx + 2 : mesh_only_idx + 3] if has_meshes else None
            # One triangle-stream length per typed terrain row, in row order; zero skips the row's scan.
            self._heightfield_row_counts = (
                c[row_gate_idx : row_gate_idx + HEIGHTFIELD_ROW_COUNT] if row_gate_idx is not None else None
            )
            self._heightfield_row_count_views = (
                tuple(c[row_gate_idx + row : row_gate_idx + row + 1] for row in range(HEIGHTFIELD_ROW_COUNT))
                if row_gate_idx is not None
                else ()
            )
            if lane_idx is not None:
                from .heightfield_lanes import DeferredPairs  # noqa: PLC0415

                deferred = DeferredPairs()
                deferred.count = c[lane_idx : lane_idx + 1]
                self._heightfield_lane_deferred = deferred
                self._set_lane_schedule(self._heightfield_lane_small_cells, self._heightfield_lane_width)
                self._heightfield_lane_empty = (
                    wp.empty(0, dtype=wp.uint64, device=device),
                    wp.empty((0, 2), dtype=wp.vec3, device=device),
                    wp.empty(0, dtype=wp.int32, device=device),
                )

            # Pair and work buffers
            self.gjk_candidate_pairs = wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device)
            self.split_query_results = (
                wp.zeros(candidate_pair_work_estimate, dtype=ConvexQueryResult, device=device)
                if self.split_gjk_mpr
                else None
            )
            self.split_gjk_work_items = (
                wp.zeros(candidate_pair_work_estimate, dtype=wp.int32, device=device) if self.split_gjk_mpr else None
            )
            self.split_manifold_work_items = (
                wp.zeros(candidate_pair_work_estimate, dtype=wp.int32, device=device) if self.split_gjk_mpr else None
            )
            # Generic convex pairs run beside mesh and SDF stages when the scene has them.
            overlap_convex = (
                wp.get_device(device).is_cuda
                and self.has_generic_convex_pairs
                and (has_meshes or has_heightfields or hydroelastic_sdf is not None)
            )
            self._convex_stream = wp.Stream(device) if overlap_convex else None
            self._convex_events = (wp.Event(device), wp.Event(device)) if overlap_convex else None
            self.shape_pairs_mesh = (
                wp.zeros(max_candidate_pairs, dtype=wp.vec2i, device=device) if has_mesh_like else None
            )
            self.triangle_pairs = (
                wp.zeros(max_triangle_pairs, dtype=wp.vec3i, device=device) if has_meshes or has_heightfields else None
            )
            # The mesh midphase's per-pair query boxes and its candidate triangles (as many as triangle pairs).
            pair_rows = max_candidate_pairs if has_mesh_like else 0
            self._mesh_query_lower = wp.empty(pair_rows, dtype=wp.vec3, device=device)
            self._mesh_query_upper = wp.empty(pair_rows, dtype=wp.vec3, device=device)
            self._mesh_query_range = wp.empty((pair_rows, 2), dtype=wp.vec3i, device=device)
            self._mesh_query_grid = wp.empty(pair_rows, dtype=wp.int32, device=device)
            self._mesh_query_slab0 = wp.empty(pair_rows, dtype=wp.vec4, device=device)
            self._mesh_query_slab1 = wp.empty(pair_rows, dtype=wp.vec4, device=device)
            self._mesh_query_slab2 = wp.empty(pair_rows, dtype=wp.vec4, device=device)
            self._mesh_query_slab_half = wp.empty(pair_rows, dtype=wp.vec3, device=device)
            # Candidates include triangles the exact pass drops: twice the triangle pairs' room.
            self._mesh_candidates = wp.empty(2 * max_triangle_pairs if has_meshes else 0, dtype=wp.vec4i, device=device)
            # The cells whose triangles may meet a pair's bound, as (pair, cell). A hit can yield no candidate, so the
            # hits are not bounded by the candidates; they get the candidates' room (0.6-0.67 hits a candidate on the
            # rough tasks' states).
            self._mesh_hits = wp.empty(2 * max_triangle_pairs if has_meshes else 0, dtype=wp.vec2i, device=device)
            self._mesh_buffer_report = wp.zeros(1, dtype=wp.int32, device=device)
            # Every current admitted slot is written by MPR or GJK before the
            # write pass. Full input capacity bounds both arrays without clears.
            self._heightfield_cylinder_query_results = (
                wp.empty(max_triangle_pairs, dtype=ConvexQueryResult, device=device)
                if self._heightfield_cylinder_split
                else None
            )
            self._heightfield_cylinder_gjk_indices = (
                wp.empty(max_triangle_pairs, dtype=wp.int32, device=device)
                if self._heightfield_cylinder_split
                else None
            )
            self.shape_pairs_mesh_plane = (
                wp.zeros(self.max_mesh_plane_pairs, dtype=wp.vec2i, device=device) if has_meshes else None
            )
            self.shape_pairs_mesh_plane_cumsum = (
                wp.zeros(self.max_mesh_plane_pairs, dtype=wp.int32, device=device) if has_meshes else None
            )
            self.shape_pairs_mesh_mesh = (
                wp.zeros(self.max_mesh_mesh_pairs, dtype=wp.vec2i, device=device) if has_meshes else None
            )

            self.empty_tangent = None
            self._empty_sort_key = wp.zeros(0, dtype=wp.int64, device=device)
            self._empty_vec3 = wp.zeros(0, dtype=wp.vec3, device=device)
            # CollisionPipeline enables this diagnostic only for unit-scale SDF queries.
            self._shape_unit_scale_sdf = wp.zeros(0, dtype=wp.int32, device=device)
            self._unit_scale_sdf_rescaled = wp.zeros(0, dtype=wp.int32, device=device)
            # Per shape, nonzero for a partner whose heightfield pairs the terrain law writes (empty: none).
            self._terrain_law_partner = wp.zeros(0, dtype=wp.int32, device=device)
            det_capacity = contact_max if contact_max is not None else max_candidate_pairs
            if deterministic and det_capacity > 0:
                self._sort_key_array = wp.zeros(det_capacity, dtype=wp.int64, device=device)
                # The live sort compares whole keys; it has no radix passes for a key width to limit.
                self._contact_sorter = ContactSorter(det_capacity, device=device)
            else:
                self._sort_key_array = wp.zeros(0, dtype=wp.int64, device=device)
                self._contact_sorter = None
            # Sentinel edge buffers used when no edge data is provided.
            # _empty_edge_range is indexed by shape id, so it must have one
            # slot per shape (not per candidate pair).
            num_shapes = shape_aabb_lower.shape[0] if shape_aabb_lower is not None else max_candidate_pairs
            self._empty_edge_indices = wp.zeros(1, dtype=wp.vec2i, device=device)
            self._empty_edge_centers = wp.zeros(1, dtype=wp.vec4, device=device)
            self._empty_edge_halves = wp.zeros(1, dtype=wp.vec4, device=device)
            self._empty_edge_order = wp.zeros(1, dtype=wp.int32, device=device)
            self._empty_edge_clusters = wp.zeros(1, dtype=wp.vec4, device=device)
            self._empty_edge_cluster_reach = wp.zeros(1, dtype=wp.float32, device=device)
            self._empty_edge_range = wp.full(max(num_shapes, 1), (-1, 0), dtype=wp.vec2i, device=device)
            # Indexed by shape id; all-zero means "no watertight bit set" so the
            # sign method falls back to automatic selection (see resolve_mesh_sign_method).
            self._empty_mesh_properties = wp.zeros(max(num_shapes, 1), dtype=wp.int32, device=device)
            # A -1 start keeps pseudo-normal signs when no boundary flags are given.
            self._empty_mesh_boundary_flags = wp.zeros(1, dtype=wp.uint8, device=device)
            self._empty_shape_mesh_boundary_start = wp.full(max(num_shapes, 1), -1, dtype=wp.int32, device=device)

            self._empty_shape_support_data = wp.full(max(num_shapes, 1), (-1, -1, -1, 0), dtype=wp.vec4i, device=device)
            # Per-shape interior points for mesh/heightfield triangle culling and MPR seeds.
            self._shape_interior_point = (
                wp.zeros(max(num_shapes, 1), dtype=wp.vec3, device=device) if has_meshes or has_heightfields else None
            )
            self._empty_support_lut = wp.zeros(1, dtype=wp.int32, device=device)
            self._empty_support_vertex_offsets = wp.zeros(1, dtype=wp.int32, device=device)
            self._empty_support_neighbors = wp.zeros(1, dtype=wp.int32, device=device)

            if hydroelastic_sdf is not None:
                self.shape_pairs_sdf_sdf = wp.zeros(hydroelastic_sdf.max_num_shape_pairs, dtype=wp.vec2i, device=device)
            else:
                # Empty arrays for when hydroelastic is disabled
                self.shape_pairs_sdf_sdf = None

        # Fixed thread count for kernel launches
        # Use a reasonable minimum for GPU occupancy (256 blocks = 32K threads)
        # but scale with expected workload to avoid massive overprovisioning.
        # 256 blocks provides good occupancy on most GPUs (2-4 blocks per SM).

        # Query GPU properties to compute appropriate thread limits
        if device_obj.is_cuda:
            # Use 4 blocks per SM as a reasonable upper bound for occupancy
            # This balances parallelism with resource utilization
            max_blocks_limit = device_obj.sm_count * 4
        else:
            # CPU fallback: use a conservative limit
            max_blocks_limit = 256

        candidate_blocks = (max_candidate_pairs + self.block_dim - 1) // self.block_dim
        min_blocks = 256  # 32K threads minimum for reasonable GPU occupancy on CUDA
        num_blocks = max(min_blocks, min(candidate_blocks, max_blocks_limit))
        self.total_num_threads = self.block_dim * num_blocks
        self.num_tile_blocks = num_blocks
        # The mesh midphase's passes: query boxes per pair, hit cells per pair, candidates per hit, exact tests per
        # candidate. The cell lookups are latency bound: launch for full residency, or a thread per pair.
        if device_obj.is_cuda:
            self._mesh_midphase_block_dim = 128
            self._mesh_midphase_blocks = max(
                1,
                min(
                    (max_candidate_pairs + self._mesh_midphase_block_dim - 1) // self._mesh_midphase_block_dim,
                    device_obj.sm_count * 32,
                ),
            )
            self._mesh_pair_threads = max(
                self.block_dim, min(max_candidate_pairs, device_obj.sm_count * 2048) // self.block_dim * self.block_dim
            )
        else:
            self._mesh_midphase_block_dim = 1
            self._mesh_midphase_blocks = num_blocks
            self._mesh_pair_threads = num_blocks
        # Split-convex blocks distribute independent serial pair queries across
        # lanes. Use one warp while preserving the block distribution; lanes
        # grid-stride over larger block queues. That suits the accelerated
        # hill-climb support map, whose per-pair cost is long and uneven. The
        # exhaustive support scan is shorter and far more uniform, and does
        # better with the full block: num_blocks is sized against self.block_dim,
        # so the narrower block also launches a quarter of the threads used by
        # the rest of the narrow phase, which throttles large replicated scenes.
        if device_obj.is_cuda:
            self.split_convex_block_dim = 32 if convex_support_acceleration else self.block_dim
        else:
            self.split_convex_block_dim = self.block_dim
        self.split_convex_total_num_threads = self.split_convex_block_dim * num_blocks
        # One-warp blocks spread sparse, serial-per-lane triangle solves across
        # more SMs without reducing the total number of launched threads.
        self.mesh_triangle_block_dim = 32 if device_obj.is_cuda else self.block_dim

        # Dynamic block allocation for mesh-mesh and mesh-plane contacts.
        # On CUDA we partition toward ~4 blocks per SM. The 128-thread mesh-mesh
        # kernel keeps four blocks resident per SM, and launching four times
        # that many blocks (several waves of small chunks) balances the uneven
        # per-chunk edge work: with only two waves, scenes around 100 pairs
        # measured 25% slower because busy chunks dominated the tail. On CPU
        # there is no SM notion so we pick 64 as a modest parallelism target
        # that splits pair work across OpenMP threads without over-subscribing
        # on small scenes.
        if self.reduce_contacts:
            target_blocks = device_obj.sm_count * 4 if device_obj.is_cuda else 64
            # Mesh-mesh
            self.num_mesh_mesh_blocks = target_blocks * 4 if device_obj.is_cuda else target_blocks
            self.mesh_mesh_target_blocks = target_blocks
            mesh_mesh_scan_size = self.max_mesh_mesh_pairs + 1
            self.mesh_mesh_block_offsets = wp.zeros(mesh_mesh_scan_size, dtype=wp.int32, device=device)
            self.mesh_mesh_block_counts = wp.zeros(mesh_mesh_scan_size, dtype=wp.int32, device=device)
            # Mesh-plane
            self.num_mesh_plane_blocks = target_blocks
            self.mesh_plane_target_blocks = target_blocks
            mesh_plane_scan_size = self.max_mesh_plane_pairs + 1
            self.mesh_plane_block_offsets = wp.zeros(mesh_plane_scan_size, dtype=wp.int32, device=device)
            self.mesh_plane_block_counts = wp.zeros(mesh_plane_scan_size, dtype=wp.int32, device=device)
        else:
            self.num_mesh_mesh_blocks = self.num_tile_blocks
            self.mesh_mesh_target_blocks = self.num_tile_blocks
            self.mesh_mesh_block_offsets = None
            self.mesh_mesh_block_counts = None
            self.num_mesh_plane_blocks = self.num_tile_blocks
            self.mesh_plane_target_blocks = self.num_tile_blocks
            self.mesh_plane_block_offsets = None
            self.mesh_plane_block_counts = None

    def _launch_generic_convex(
        self,
        device: Devicelike,
        candidate_pair_count: wp.array[int],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_support_data: wp.array[wp.vec4i],
        support_lut: wp.array[int],
        support_vertex_offsets: wp.array[int],
        support_neighbors: wp.array[int],
        shape_gap: wp.array[float],
        shape_collision_radius: wp.array[float],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        writer_data: Any,
    ):
        """Launch the GJK/MPR stage for the convex pairs routed by the primitive kernel.

        With the coherent-convex owner installed, rejection screening (and, under the
        sampled chooser, the choice between it and the ordinary queries) replaces the
        ordinary MPR/GJK queue owners; the manifold stage is shared.
        """
        if not self.has_generic_convex_pairs:
            return
        convex_pairs = self.gjk_candidate_pairs
        convex_pair_count = candidate_pair_count if self.sparse_gjk_pairs else self.gjk_candidate_pairs_count
        if not self.split_gjk_mpr:
            wp.launch(
                kernel=self.narrow_phase_kernel,
                dim=self.total_num_threads,
                inputs=[
                    convex_pairs,
                    convex_pair_count,
                    shape_types,
                    shape_data,
                    shape_transform,
                    shape_source,
                    shape_support_data,
                    support_lut,
                    support_vertex_offsets,
                    support_neighbors,
                    shape_gap,
                    shape_collision_radius,
                    self.shape_aabb_lower,
                    self.shape_aabb_upper,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    writer_data,
                    self.total_num_threads,
                ],
                device=device,
                block_dim=self.block_dim,
                record_tape=False,
            )
            return

        common_inputs = [
            shape_types,
            shape_data,
            shape_transform,
            shape_source,
            shape_support_data,
            support_lut,
            support_vertex_offsets,
            support_neighbors,
            shape_gap,
            shape_collision_radius,
            self.shape_aabb_lower,
            self.shape_aabb_upper,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
        ]

        def ordinary(query_count=None):
            wp.launch(
                kernel=self.narrow_phase_mpr_kernel,
                dim=self.split_convex_total_num_threads,
                inputs=[
                    convex_pairs,
                    convex_pair_count if query_count is None else query_count,
                    *common_inputs,
                    self.split_convex_total_num_threads,
                    self.split_query_results,
                    self.split_gjk_work_items,
                    self.split_gjk_work_count,
                    self.split_manifold_work_items,
                    self.split_manifold_work_count,
                ],
                device=device,
                block_dim=self.split_convex_block_dim,
                record_tape=False,
            )
            wp.launch(
                kernel=self.narrow_phase_gjk_kernel,
                dim=self.split_convex_total_num_threads,
                inputs=[
                    convex_pairs,
                    *common_inputs,
                    self.split_convex_total_num_threads,
                    self.split_query_results,
                    self.split_gjk_work_items,
                    self.split_gjk_work_count,
                    self.split_manifold_work_items,
                    self.split_manifold_work_count,
                ],
                device=device,
                block_dim=self.split_convex_block_dim,
                record_tape=False,
            )

        pair_inputs = None
        if self._coherent_cache is not None:
            from .coherent_convex_rejection import _PairInputs  # noqa: PLC0415

            pair_inputs = _PairInputs()
            pair_inputs.shape_types = shape_types
            pair_inputs.shape_data = shape_data
            pair_inputs.shape_transform = shape_transform
            pair_inputs.shape_source = shape_source
            pair_inputs.shape_support_data = shape_support_data
            pair_inputs.support_lut = support_lut
            pair_inputs.support_vertex_offsets = support_vertex_offsets
            pair_inputs.support_neighbors = support_neighbors
            pair_inputs.shape_gap = shape_gap
            pair_inputs.shape_collision_radius = shape_collision_radius
            pair_inputs.shape_aabb_lower = self.shape_aabb_lower
            pair_inputs.shape_aabb_upper = self.shape_aabb_upper
            pair_inputs.shape_collision_aabb_lower = shape_collision_aabb_lower
            pair_inputs.shape_collision_aabb_upper = shape_collision_aabb_upper
            coherent_inputs = [
                convex_pairs,
                convex_pair_count,
                pair_inputs,
                self._coherent_cache.data,
                self.total_num_threads,
                self.split_query_results,
                self.split_gjk_work_items,
                self.split_gjk_work_count,
                self.split_manifold_work_items,
                self.split_manifold_work_count,
            ]

            def rejection(query_count=None, gjk_count=None):
                if query_count is not None:
                    coherent_inputs[1] = query_count
                    coherent_inputs[7] = gjk_count
                if self._convex_method_choice is None:
                    self._coherent_cache.begin()
                for coherent_kernel in self._coherent_query_kernels:
                    wp.launch(
                        kernel=coherent_kernel,
                        dim=self.total_num_threads,
                        inputs=coherent_inputs,
                        device=device,
                        block_dim=self.block_dim,
                        record_tape=False,
                    )

        if self._convex_method_choice is not None:
            self._coherent_cache.begin()
            self._convex_method_choice.launch(convex_pairs, convex_pair_count, pair_inputs, ordinary, rejection)
        elif self._coherent_cache is not None:
            rejection()
        else:
            ordinary()
        wp.launch(
            kernel=self.narrow_phase_manifold_kernel,
            dim=self.split_convex_total_num_threads,
            inputs=[
                convex_pairs,
                *common_inputs,
                writer_data,
                self.split_convex_total_num_threads,
                self.split_query_results,
                self.split_manifold_work_items,
                self.split_manifold_work_count,
            ],
            device=device,
            block_dim=self.split_convex_block_dim,
            record_tape=False,
        )

    def _set_lane_schedule(self, small_cells: int, lanes: int, *, capacity: int | None = None) -> None:
        """Set the lane midphase schedule and allocate its deferred-pair storage.

        Pairs with more than ``small_cells`` cells (none when negative) move to the
        lane pass, ``lanes`` lanes each. ``capacity`` deferred pairs fit, by default
        none when nothing is deferred and ``_heightfield_lane_capacity`` otherwise;
        pass 1 walks any further long pair in place, so the triangle set stays exact.
        Call it only between collisions, never while a captured graph is reused.
        """
        from .heightfield_geometric import PairProjection  # noqa: PLC0415
        from .heightfield_lanes import CellWindow  # noqa: PLC0415

        if capacity is None:
            capacity = 0 if small_cells < 0 else self._heightfield_lane_capacity
        deferred = self._heightfield_lane_deferred
        deferred.windows = wp.empty(capacity, dtype=CellWindow, device=self.device)
        deferred.projections = wp.empty(
            capacity if self._heightfield_geometric_cull else 0, dtype=PairProjection, device=self.device
        )
        self._heightfield_lane_small_cells = small_cells
        self._heightfield_lane_width = lanes

    def _launch_lane_midphase(
        self,
        device,
        shape_types,
        shape_data,
        shape_transform,
        shape_source,
        shape_gap,
        shape_collision_aabb_lower,
        shape_collision_aabb_upper,
        shape_heightfield_index,
        heightfield_data,
        heightfield_elevations,
    ):
        """Run the per-pair pass and, on CUDA, the lane pass of the terrain midphase.

        The kernels and their struct arguments are built on the first call and
        again only when the route or a bound array changes, so a collision does
        not rebuild them.
        """
        geometric = self._heightfield_geometric_cull
        tagged = self._heightfield_pair_csr
        no_sources, no_bounds, no_tags = self._heightfield_lane_empty
        arrays = (
            shape_transform,
            shape_gap,
            shape_data,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_heightfield_index,
            heightfield_data,
            heightfield_elevations,
            self.shape_pairs_mesh,
            self.shape_pairs_mesh_count,
            shape_types,
            shape_source if geometric else no_sources,
            self._finite_bounds if geometric else no_bounds,
            self._finite_source if geometric else no_sources,
            self.triangle_pairs,
            self.triangle_pairs_count,
            self._pair_csr.triangle_pair if tagged else no_tags,
        )
        cached = self._heightfield_lane_launch
        if (
            cached is None
            or cached[0] != (geometric, tagged)
            or any(old is not new for old, new in zip(cached[1], arrays, strict=True))
        ):
            from .heightfield_lanes import TerrainPairs, TerrainTriangles, create_lane_midphase_kernels  # noqa: PLC0415

            scene = TerrainPairs()
            triangles = TerrainTriangles()
            (
                scene.transforms,
                scene.gaps,
                scene.data,
                scene.lower,
                scene.upper,
                scene.heightfield_index,
                scene.heightfield,
                scene.elevations,
                scene.pairs,
                scene.pair_count,
                scene.types,
                scene.sources,
                scene.bounds,
                scene.bound_source,
                triangles.triples,
                triangles.count,
                triangles.tags,
            ) = arrays
            cached = ((geometric, tagged), arrays, scene, triangles, create_lane_midphase_kernels(geometric, tagged))
            self._heightfield_lane_launch = cached
        _route, _arrays, scene, triangles, (pairs_kernel, lanes_kernel) = cached
        passes = [(pairs_kernel, self._heightfield_lane_small_cells)]
        if self._heightfield_lane_small_cells >= 0:
            passes.append((lanes_kernel, self._heightfield_lane_width))
        for kernel, parameter in passes:
            wp.launch(
                kernel=kernel,
                dim=self.total_num_threads,
                inputs=[scene, triangles, self._heightfield_lane_deferred, parameter, self.total_num_threads],
                device=device,
                block_dim=self.tile_size_mesh_convex,
                record_tape=False,
            )

    def _own_mesh_data(self, shape_types, shape_source, shape_collision_radius, device):
        """The cells and topology of the meshes in ``shape_source``, for a narrow phase not given its model's.

        Built once from the device meshes' buffers (one read for both) and kept for launches with the same
        ``shape_source`` array; a launch with another array builds them for it. The topology reads the launch's
        ``shape_collision_radius`` in place, as a model's reads the model's.
        """
        key = (shape_source.ptr, shape_source.shape[0], shape_types.ptr)
        if self._own_mesh_key != key:
            types, sources, buffers = read_mesh_buffers(shape_types, shape_source, device)
            cells = build_mesh_cells(types, sources, device, meshes=buffers)
            meshes = {mid: _HostMesh(points, indices) for mid, (points, indices) in buffers.items()}
            shape_meshes = [
                meshes.get(int(source)) if t == int(GeoType.MESH) else None
                for t, source in zip(types, sources, strict=True)
            ]
            topology = MeshTopology(types, shape_meshes, device, shape_collision_radius).data
            self._own_mesh_data_value = (cells, topology)
            self._own_mesh_key = key
        cells, topology = self._own_mesh_data_value
        topology.shape_radius = shape_collision_radius
        return cells, topology

    def launch_custom_write(
        self,
        *,
        candidate_pair: wp.array[wp.vec2i],  # Maybe colliding pairs
        candidate_pair_count: wp.array[wp.int32],  # Size one array
        shape_types: wp.array[wp.int32],  # All shape types, pairs index into it
        shape_data: wp.array[wp.vec4],  # Shape data (scale xyz, margin w)
        shape_transform: wp.array[wp.transform],  # In world space
        shape_source: wp.array[wp.uint64],  # The index into the source array, type define by shape_types
        shape_support_data: wp.array[wp.vec4i] | None = None,
        support_lut: wp.array[wp.int32] | None = None,
        support_vertex_offsets: wp.array[wp.int32] | None = None,
        support_neighbors: wp.array[wp.int32] | None = None,
        shape_mesh_properties: wp.array[wp.int32] | None = None,  # Per-shape mesh property bitfield
        mesh_boundary_flags: wp.array[wp.uint8] | None = None,  # Packed per-face open-boundary bits
        shape_mesh_boundary_start: wp.array[wp.int32] | None = None,  # Per-shape start into the bits
        shape_sdf_index: wp.array[wp.int32],  # Per-shape index into texture_sdf_data (-1 for none)
        shape_gap: wp.array[wp.float32],  # per-shape contact gap (detection threshold)
        shape_base_gap: wp.array[wp.float32] | None = None,
        shape_collision_radius: wp.array[wp.float32],  # per-shape collision radius for AABB fallback
        shape_flags: wp.array[wp.int32],  # per-shape flags (includes ShapeFlags.HYDROELASTIC)
        shape_collision_aabb_lower: wp.array[wp.vec3],  # Local-space AABB lower bounds
        shape_collision_aabb_upper: wp.array[wp.vec3],  # Local-space AABB upper bounds
        shape_voxel_resolution: wp.array[wp.vec3i],  # Voxel grid resolution per shape
        shape_interior_point: wp.array[wp.vec3] | None = None,  # Local-space interior points
        texture_sdf_data: wp.array[TextureSDFData] | None = None,  # Compact texture SDF data table
        shape_heightfield_index: wp.array[wp.int32] | None = None,
        heightfield_data: wp.array[HeightfieldData] | None = None,
        heightfield_elevations: wp.array[wp.float32] | None = None,
        mesh_edge_indices: wp.array[wp.vec2i] | None = None,
        mesh_edge_centers: wp.array[wp.vec4] | None = None,
        mesh_edge_halves: wp.array[wp.vec4] | None = None,
        mesh_edge_order: wp.array[wp.int32] | None = None,
        mesh_edge_clusters: wp.array[wp.vec4] | None = None,
        mesh_edge_cluster_reach: wp.array[wp.float32] | None = None,
        shape_edge_range: wp.array[wp.vec2i] | None = None,
        mesh_cells: MeshCells | None = None,
        mesh_topology: MeshTopologyData | None = None,
        hydroelastic_shape_sdf_data_prepared: bool = False,
        shape_linear_velocity: wp.array[wp.vec3] | None = None,
        shape_angular_velocity: wp.array[wp.vec3] | None = None,
        collision_update_dt: float = 0.0,
        max_speculative_extension: float = 0.0,
        writer_data: Any,
        verify_contacts: bool = True,
        device: Devicelike | None = None,  # Device to launch on
    ) -> None:
        """
        Launch narrow phase collision detection with a custom contact writer struct.

        All internal kernel launches use ``record_tape=False`` so that calls
        are safe inside a :class:`warp.Tape` context.

        Args:
            candidate_pair: Array of potentially colliding shape pairs from broad phase
            candidate_pair_count: Single-element array containing the number of candidate pairs
            shape_types: Array of geometry types for all shapes
            shape_data: Array of vec4 containing scale (xyz) and margin (w) for each shape
            shape_transform: Array of world-space transforms for each shape
            shape_source: Array of source pointers (mesh IDs, etc.) for each shape
            shape_support_data: Optional per-shape ``wp.vec4i`` metadata containing
                packed support-LUT, vertex-offset, and ``support_neighbors`` block starts
                plus the LUT resolution. The block start is also set for verified hulls
                without acceleration. The final entry packs resolution in its low 16 bits
                and strict hull validity in bit 16. Zero resolution disables acceleration.
            support_lut: Optional packed octahedral-direction lookup tables whose entries
                select seed vertices for convex support walks.
            support_vertex_offsets: Optional packed CSR offsets delimiting each convex
                vertex's entries in ``support_neighbors``. They index past the hull's
                edge records.
            support_neighbors: Optional packed per-hull blocks. A verified hull's block
                starts with its edge count and one ``(a, b, c1, c2)`` vertex record per
                edge, whose faces ``cross(b - a, c1 - a)`` and ``cross(c2 - a, b - a)``
                point outward; exact box-hull depth certificates read these records.
                CSR vertex-adjacency indices for support walks follow. Without the edge
                records, box-hull depths fall back to uncertified support-plane bounds.
            shape_mesh_properties: Per-shape mesh property bitfield.
            mesh_boundary_flags: Optional packed per-face bits marking open-boundary
                edges and vertices of meshes queried with pseudo-normal signs.
            shape_mesh_boundary_start: Optional per-shape offset into
                ``mesh_boundary_flags``, or -1 for shapes without boundary bits.
                Without both arrays, pseudo-normal signs extend past open boundaries.
            shape_sdf_index: Per-shape SDF table index (-1 for shapes without SDF)
            texture_sdf_data: Compact array of TextureSDFData structs
            shape_gap: Array of per-shape contact gaps (detection threshold) for each shape
            shape_base_gap: Array of authored per-shape contact gaps used for
                exact admission and regular SDF reduction lanes.
            shape_collision_radius: Array of collision radii for each shape (for AABB fallback for planes/meshes)
            shape_flags: Array of shape flags for each shape (includes ShapeFlags.HYDROELASTIC)
            shape_collision_aabb_lower: Local-space AABB lower bounds for each shape (for voxel binning)
            shape_collision_aabb_upper: Local-space AABB upper bounds for each shape (for voxel binning)
            shape_voxel_resolution: Voxel grid resolution for each shape (for voxel binning)
            shape_interior_point: Optional scaled local-space point inside each shape [m],
                used to cull back-facing mesh triangles and seed triangle contacts. If
                ``None``, convex-mesh volume centroids are computed on each launch.
            shape_heightfield_index: Per-shape heightfield index; required when ``has_heightfields`` is True.
            heightfield_data: Heightfield descriptors; required when ``has_heightfields`` is True.
            heightfield_elevations: Current packed elevations; required when ``has_heightfields`` is True.
            mesh_edge_indices: Packed array of mesh edge vertex pairs for all shapes.
                Required with ``shape_edge_range`` for mesh edge contacts; omitting these
                arrays disables mesh edge contacts rather than extracting edges from meshes.
            mesh_edge_centers: Packed precomputed mesh edge centers (xyz) and radii (w) in
                unscaled mesh coordinates [m]; the shape scale in ``shape_data`` is applied when read.
                Each radius is the length of the corresponding unscaled half-vector.
            mesh_edge_halves: Packed precomputed mesh edge half-vectors (xyz) in unscaled mesh
                coordinates [m] and endpoint-ownership codes (w). Set the ownership code to zero
                to allow both endpoints.
            mesh_edge_order: Packed per-shape edge indices in cluster order (see
                :func:`~newton._src.geometry.sdf_contact.build_mesh_edge_clusters`).
            mesh_edge_clusters: Packed per-shape edge-cluster centers and spreads [m], at the
                shape's edge offset.
            mesh_edge_cluster_reach: Packed per-shape largest edge radius per cluster [m].
            shape_edge_range: Per-shape (start, count) into mesh_edge_indices.
                Required with ``mesh_edge_indices`` for mesh edge contacts.
            mesh_cells: The cell indices of the triangle meshes in ``shape_source``
                (:func:`~newton._src.geometry.mesh_cells.build_mesh_cells`), which the mesh-vs-convex midphase
                reads. Pipelines pass the one their model built at finalize.
            mesh_topology: The welded topology of the same meshes
                (:class:`~newton._src.geometry.mesh_topology.MeshTopology`'s ``data``), which the triangle contact
                law reads; it becomes :attr:`mesh_topology`. Pipelines pass the one their model built with the cells.
                A narrow phase given neither builds both once, from the meshes of ``shape_source`` read on the host
                (not inside a CUDA graph capture), and keeps them for later launches with the same ``shape_source``
                array.
            hydroelastic_shape_sdf_data_prepared: Whether finalized hydroelastic SDF descriptors were cached upstream.
            shape_linear_velocity: Shape-origin linear velocities [m/s]. Required in speculative mode.
            shape_angular_velocity: Shape angular velocities [rad/s]. Required in speculative mode.
            collision_update_dt: Collision prediction horizon [s].
            max_speculative_extension: Maximum speculative clearance [m].
            writer_data: Custom struct instance for contact writing (type must match the custom writer function)
            verify_contacts: Whether the buffer verifier flags contacts past ``writer_data.contact_max``. A
                caller that grows its contact rows and reruns the pass on overflow turns it off.
            device: Device to launch on

        Raises:
            ValueError: If a heightfield pipeline omits a required heightfield array.
        """
        if self.has_heightfields:
            for name, value in (
                ("shape_heightfield_index", shape_heightfield_index),
                ("heightfield_data", heightfield_data),
                ("heightfield_elevations", heightfield_elevations),
            ):
                if value is None:
                    raise ValueError(f"Heightfield NarrowPhase requires {name}")
        if device is None:
            device = self.device if self.device is not None else candidate_pair.device
        if shape_mesh_properties is None:
            shape_mesh_properties = self._empty_mesh_properties
        if mesh_boundary_flags is None or shape_mesh_boundary_start is None:
            mesh_boundary_flags = self._empty_mesh_boundary_flags
            shape_mesh_boundary_start = self._empty_shape_mesh_boundary_start
        if shape_support_data is None:
            shape_support_data = self._empty_shape_support_data
        if support_lut is None:
            support_lut = self._empty_support_lut
        if support_vertex_offsets is None:
            support_vertex_offsets = self._empty_support_vertex_offsets
        if support_neighbors is None:
            support_neighbors = self._empty_support_neighbors
        if shape_edge_range is None:
            shape_edge_range = self._empty_edge_range
        # The triangle pair count the buffer check reads (the mesh midphase's report when it runs).
        mesh_buffer_report = self.triangle_pairs_count
        if self.has_meshes and (mesh_cells is None or mesh_topology is None):
            own_cells, own_topology = self._own_mesh_data(shape_types, shape_source, shape_collision_radius, device)
            mesh_cells = own_cells if mesh_cells is None else mesh_cells
            mesh_topology = own_topology if mesh_topology is None else mesh_topology
        if mesh_topology is not None:
            self.mesh_topology = mesh_topology
        if mesh_cells is None:
            if self._empty_mesh_cells is None:
                self._empty_mesh_cells = build_mesh_cells([], [], device)
            mesh_cells = self._empty_mesh_cells
        if self.speculative and shape_base_gap is None:
            raise ValueError("Speculative NarrowPhase requires the authored shape_base_gap array")
        if shape_base_gap is None:
            shape_base_gap = shape_gap
        if self.speculative and (shape_linear_velocity is None or shape_angular_velocity is None):
            raise ValueError("Speculative NarrowPhase requires per-shape linear/angular velocity arrays")
        if self.speculative:
            for name, value in (
                ("collision_update_dt", collision_update_dt),
                ("max_speculative_extension", max_speculative_extension),
            ):
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
        shape_linear_velocity = self._empty_vec3 if shape_linear_velocity is None else shape_linear_velocity
        shape_angular_velocity = self._empty_vec3 if shape_angular_velocity is None else shape_angular_velocity

        # Clear counters only when a routed or split path consumes them.
        if (
            self.has_generic_convex_pairs
            or self.has_meshes
            or self.has_heightfields
            or self.hydroelastic_sdf is not None
        ):
            self._counter_array.zero_()

        # Stage 1: Launch primitive kernel for fast analytical collisions
        # This handles sphere-sphere, sphere-capsule, capsule-capsule, plane-sphere, plane-capsule
        # and routes remaining pairs to gjk_candidate_pairs and mesh buffers
        wp.launch(
            kernel=self.primitive_kernel,
            dim=self.total_num_threads,
            inputs=[
                candidate_pair,
                candidate_pair_count,
                shape_types,
                shape_data,
                shape_transform,
                shape_linear_velocity,
                shape_angular_velocity,
                collision_update_dt,
                max_speculative_extension,
                shape_source,
                shape_gap,
                shape_flags,
                shape_sdf_index,
                shape_edge_range,
                self._shape_unit_scale_sdf,
                self._terrain_law_partner,
                writer_data,
                self.total_num_threads,
            ],
            outputs=[
                self.gjk_candidate_pairs,
                self.gjk_candidate_pairs_count,
                self.shape_pairs_mesh,
                self.shape_pairs_mesh_count,
                self.shape_pairs_mesh_plane,
                self.shape_pairs_mesh_plane_cumsum,
                self.shape_pairs_mesh_plane_count,
                self.mesh_plane_vertex_total_count,
                self.shape_pairs_mesh_mesh,
                self.shape_pairs_mesh_mesh_count,
                self.shape_pairs_sdf_sdf,
                self.shape_pairs_sdf_sdf_count,
                self._unit_scale_sdf_rescaled,
            ],
            device=device,
            block_dim=self.block_dim,
            record_tape=False,
        )

        # Stage 2: Launch GJK/MPR kernel for remaining convex pairs
        # These are pairs that couldn't be handled analytically (box, cylinder, cone, convex hull, etc.)
        # All routing has been done by the primitive kernel, so this kernel just does GJK/MPR.
        # Convex pairs only append contacts, so they overlap the mesh and SDF
        # stages on a second stream; a few slow pairs no longer serialize them.
        main_stream = None
        if self._convex_stream is not None and self.has_generic_convex_pairs:
            main_stream = wp.get_stream(device)
            self._convex_stream.wait_event(main_stream.record_event(self._convex_events[0]))
        # The explicit event already forks the stream, so skip ScopedStream's own sync.
        with (
            wp.ScopedStream(self._convex_stream, sync_enter=False)
            if main_stream is not None
            else contextlib.nullcontext()
        ):
            self._launch_generic_convex(
                device,
                candidate_pair_count,
                shape_types,
                shape_data,
                shape_transform,
                shape_source,
                shape_support_data,
                support_lut,
                support_vertex_offsets,
                support_neighbors,
                shape_gap,
                shape_collision_radius,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                writer_data,
            )

        # Skip mesh/heightfield kernels when no meshes or heightfields are present
        if self.has_meshes or self.has_heightfields:
            # Launch mesh-plane contact processing kernel (meshes only)
            if self.has_meshes and not self.reduce_contacts and self.max_mesh_plane_pairs > 0:
                wp.launch(
                    kernel=self.mesh_plane_contacts_kernel,
                    dim=self.total_num_threads,
                    inputs=[
                        shape_data,
                        shape_transform,
                        shape_source,
                        shape_gap,
                        shape_collision_aabb_lower,
                        shape_collision_aabb_upper,
                        shape_voxel_resolution,
                        self.shape_pairs_mesh_plane,
                        self.shape_pairs_mesh_plane_count,
                        writer_data,
                        self.num_tile_blocks,
                    ],
                    device=device,
                    block_dim=self.block_dim,
                    record_tape=False,
                )

            if shape_interior_point is None:
                if self._shape_interior_point.shape[0] < shape_types.shape[0]:
                    self._shape_interior_point = wp.zeros(shape_types.shape[0], dtype=wp.vec3, device=device)
                wp.launch(
                    kernel=compute_shape_interior_points_kernel,
                    dim=shape_types.shape[0],
                    inputs=[shape_types, shape_data, shape_source],
                    outputs=[self._shape_interior_point],
                    device=device,
                    record_tape=False,
                )
                shape_interior_point = self._shape_interior_point

            # Launch midphase: finds overlapping triangles for both mesh and heightfield pairs.
            # Heightfield-only scenes take the cell-rejection terrain routes; scenes with meshes
            # take the mesh cell midphase (three passes).
            if self._heightfield_cell_reject:
                second_dim = 1 if self._heightfield_packed_pairs else self.tile_size_mesh_convex
                midphase_workers = self.total_num_threads if self._heightfield_packed_pairs else self.num_tile_blocks
                midphase_kernel = heightfield_cell_overlaps_kernel
                midphase_inputs = [
                    shape_transform,
                    shape_gap,
                    shape_data,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_heightfield_index,
                    heightfield_data,
                    heightfield_elevations,
                    self.shape_pairs_mesh,
                    self.shape_pairs_mesh_count,
                    midphase_workers,
                ]
                if self._heightfield_geometric_cull:
                    from .heightfield_geometric import heightfield_geometric_overlaps_kernel  # noqa: PLC0415

                    if self._finite_bounds is None:
                        raise RuntimeError("Geometric cull requires CollisionPipeline finite model binding")
                    midphase_kernel = heightfield_geometric_overlaps_kernel
                    midphase_inputs.extend([shape_types, shape_source, self._finite_bounds, self._finite_source])
                else:
                    midphase_inputs.insert(-1, shape_types)
                midphase_outputs = [self.triangle_pairs, self.triangle_pairs_count]
                if self._heightfield_pair_csr:
                    midphase_kernel = self._pair_csr.midphase
                    midphase_outputs.append(self._pair_csr.triangle_pair)
                # The lane passes replace only the packed per-pair launch.
                if self._heightfield_lane_midphase and self._heightfield_packed_pairs:
                    self._launch_lane_midphase(
                        device,
                        shape_types,
                        shape_data,
                        shape_transform,
                        shape_source,
                        shape_gap,
                        shape_collision_aabb_lower,
                        shape_collision_aabb_upper,
                        shape_heightfield_index,
                        heightfield_data,
                        heightfield_elevations,
                    )
                else:
                    wp.launch(
                        kernel=midphase_kernel,
                        dim=[midphase_workers, second_dim],
                        inputs=midphase_inputs,
                        outputs=midphase_outputs,
                        device=device,
                        block_dim=self.tile_size_mesh_convex,
                        record_tape=False,
                    )
            else:
                wp.launch(
                    kernel=narrow_phase_find_mesh_triangle_overlaps_kernel,
                    dim=self._mesh_pair_threads,
                    inputs=[
                        shape_types,
                        shape_transform,
                        shape_source,
                        shape_gap,
                        shape_data,
                        shape_collision_aabb_lower,
                        shape_collision_aabb_upper,
                        shape_heightfield_index,
                        heightfield_data,
                        heightfield_elevations,
                        self.shape_pairs_mesh,
                        self.shape_pairs_mesh_count,
                        mesh_cells,
                        self._mesh_pair_threads,
                    ],
                    outputs=[
                        self._mesh_query_lower,
                        self._mesh_query_upper,
                        self._mesh_query_range,
                        self._mesh_query_grid,
                        self._mesh_query_slab0,
                        self._mesh_query_slab1,
                        self._mesh_query_slab2,
                        self._mesh_query_slab_half,
                        self.triangle_pairs,
                        self.triangle_pairs_count,
                    ],
                    device=device,
                    record_tape=False,
                )
                if self.has_meshes:
                    query_arrays = [
                        self._mesh_query_lower,
                        self._mesh_query_upper,
                        self._mesh_query_range,
                        self._mesh_query_grid,
                        self._mesh_query_slab0,
                        self._mesh_query_slab1,
                        self._mesh_query_slab2,
                        self._mesh_query_slab_half,
                    ]
                    wp.launch(
                        kernel=narrow_phase_mesh_cell_hits_kernel,
                        dim=[self._mesh_midphase_blocks, self._mesh_midphase_block_dim],
                        inputs=[
                            self.shape_pairs_mesh_count,
                            *query_arrays,
                            mesh_cells,
                            self._mesh_midphase_blocks * self._mesh_midphase_block_dim,
                        ],
                        outputs=[self._mesh_hits, self._mesh_hit_count],
                        device=device,
                        block_dim=self._mesh_midphase_block_dim,
                        record_tape=False,
                    )
                    wp.launch(
                        kernel=narrow_phase_mesh_hit_candidates_kernel,
                        dim=self._mesh_pair_threads,
                        inputs=[
                            self._mesh_hits,
                            self._mesh_hit_count,
                            *query_arrays,
                            mesh_cells,
                            self._mesh_pair_threads,
                        ],
                        outputs=[self._mesh_candidates, self._mesh_candidate_count],
                        device=device,
                        record_tape=False,
                    )
                    wp.launch(
                        kernel=narrow_phase_mesh_triangle_candidates_kernel,
                        dim=self._mesh_pair_threads,
                        inputs=[
                            shape_types,
                            shape_transform,
                            shape_source,
                            shape_mesh_properties,
                            shape_gap,
                            shape_data,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_interior_point,
                            self.shape_pairs_mesh,
                            self._mesh_query_lower,
                            self._mesh_query_upper,
                            self._mesh_candidates,
                            self._mesh_candidate_count,
                            self._mesh_pair_threads,
                        ],
                        outputs=[self.triangle_pairs, self.triangle_pairs_count],
                        device=device,
                        record_tape=False,
                    )
                    wp.launch(
                        kernel=narrow_phase_mesh_buffer_report_kernel,
                        dim=1,
                        inputs=[
                            self.triangle_pairs_count,
                            self.triangle_pairs.shape[0],
                            self._mesh_candidate_count,
                            self._mesh_candidates.shape[0],
                            self._mesh_hit_count,
                            self._mesh_hits.shape[0],
                        ],
                        outputs=[self._mesh_buffer_report],
                        device=device,
                        record_tape=False,
                    )
                    mesh_buffer_report = self._mesh_buffer_report

            # Hull support acceleration and interior seeds, appended to every triangle query's inputs.
            support_inputs = [
                shape_support_data,
                support_lut,
                support_vertex_offsets,
                support_neighbors,
                shape_interior_point,
                self.mesh_topology,
            ]

            # Mesh triangles that publish directly (see _mesh_triangles_direct): spheres and capsules in closed form,
            # the face case in closed form, then GJK/MPR on the compacted rest.
            direct_meshes = self._mesh_triangles_direct()
            if direct_meshes:
                direct_inputs = [
                    shape_types,
                    shape_data,
                    shape_transform,
                    shape_source,
                    shape_gap,
                    shape_heightfield_index,
                    heightfield_data,
                    heightfield_elevations,
                    self.triangle_pairs,
                    self.triangle_pairs_count,
                    writer_data,
                    self.total_num_threads,
                ]
                self._launch_mesh_triangle_passes(
                    self._mesh_round_direct,
                    self._mesh_face_direct,
                    self._mesh_queued_direct,
                    direct_inputs,
                    support_inputs,
                    device,
                )

            # Launch contact processing for triangle pairs
            if self.reduce_contacts:
                # Unified global reduction for all mesh contact types.
                assert self.global_contact_reducer is not None
                if self._reducer_export_cleanup:
                    # The preceding stock export retired its entries. The first
                    # call uses the constructor's cleared table; no host state
                    # or device-count read is needed during graph capture.
                    self.global_contact_reducer.reset_after_export()
                else:
                    self.global_contact_reducer.clear_active()
                reducer_data = self.global_contact_reducer.get_data_struct()

                # Mesh-plane contacts → global reducer (meshes only)
                if self.has_meshes and self.max_mesh_plane_pairs > 0:
                    compute_mesh_plane_block_offsets_scan(
                        shape_pairs_mesh_plane=self.shape_pairs_mesh_plane,
                        shape_pairs_mesh_plane_count=self.shape_pairs_mesh_plane_count,
                        shape_source=shape_source,
                        target_blocks=self.mesh_plane_target_blocks,
                        block_offsets=self.mesh_plane_block_offsets,
                        block_counts=self.mesh_plane_block_counts,
                        total_vert_count=self.mesh_plane_total_weight,
                        total_num_threads=self.total_num_threads,
                        device=device,
                        record_tape=False,
                    )
                    wp.launch_tiled(
                        kernel=self.mesh_plane_contacts_kernel,
                        dim=(self.num_mesh_plane_blocks,),
                        inputs=[
                            shape_data,
                            shape_transform,
                            shape_source,
                            shape_gap,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            self.shape_pairs_mesh_plane,
                            self.shape_pairs_mesh_plane_count,
                            self.mesh_plane_block_offsets,
                            reducer_data,
                            self.num_mesh_plane_blocks,
                        ],
                        device=device,
                        block_dim=self.tile_size_mesh_plane,
                        record_tape=False,
                    )

                # Mesh/heightfield-triangle contacts → same global reducer
                triangle_inputs = [
                    shape_types,
                    shape_data,
                    shape_transform,
                    shape_source,
                    shape_gap,
                    shape_heightfield_index,
                    heightfield_data,
                    heightfield_elevations,
                    self.triangle_pairs,
                    self.triangle_pairs_count,
                    self._pair_csr.data if self._heightfield_pair_csr else reducer_data,
                    self.total_num_threads,
                ]
                if self._heightfield_finite_query:
                    if self._finite_bounds is None:
                        raise RuntimeError("Finite query requires CollisionPipeline model binding")
                    wp.launch(
                        kernel=self._pair_csr.finite if self._heightfield_pair_csr else self._finite_reducer,
                        dim=self.total_num_threads,
                        inputs=[*triangle_inputs, self._finite_bounds, self._finite_source],
                        device=device,
                        block_dim=self.mesh_triangle_block_dim,
                        record_tape=False,
                    )
                if (
                    self._mesh_round_kernel is not None
                    and not direct_meshes
                    and not (self._heightfield_pair_csr or self._heightfield_finite_query)
                ):
                    wp.launch(
                        kernel=self._mesh_round_kernel,
                        dim=self.total_num_threads,
                        inputs=[*triangle_inputs, *support_inputs],
                        device=device,
                        block_dim=self.mesh_triangle_block_dim,
                        record_tape=False,
                    )
                if self._heightfield_primitive_query:
                    if self._heightfield_row_gate:
                        wp.launch(
                            kernel=heightfield_row_gate_kernel,
                            dim=self.total_num_threads,
                            inputs=[
                                self.shape_pairs_mesh,
                                self.shape_pairs_mesh_count,
                                shape_types,
                                shape_data,
                                self.triangle_pairs_count,
                                self.total_num_threads,
                            ],
                            outputs=[self._heightfield_row_counts],
                            device=device,
                            block_dim=self.block_dim,
                            record_tape=False,
                        )
                    primitive_kernels = self._heightfield_primitive_kernels
                    if direct_meshes:
                        # Mesh pairs published directly: the fallback takes the heightfield pairs only.
                        primitive_kernels = (*primitive_kernels[:-1], self._heightfield_reducer_kernel)
                    for primitive_index, primitive_kernel in enumerate(primitive_kernels):
                        row_inputs = triangle_inputs
                        if self._heightfield_row_gate:
                            # The same row kernel reads its gated stream length as triangle_pairs_count.
                            length = self._heightfield_row_count_views[primitive_index]
                            row_inputs = [
                                length if value is self.triangle_pairs_count else value for value in triangle_inputs
                            ]
                        kernels = (primitive_kernel,)
                        inputs = [*row_inputs, *support_inputs]
                        if self._heightfield_cylinder_split and primitive_index == 1:
                            kernels = self._heightfield_cylinder_split_kernels
                            inputs = [
                                *row_inputs,
                                self._heightfield_cylinder_query_results,
                                self._heightfield_cylinder_gjk_indices,
                                self._heightfield_cylinder_gjk_count,
                                SupportMapDataProvider(),
                            ]
                        for kernel in kernels:
                            wp.launch(
                                kernel=kernel,
                                dim=self.total_num_threads,
                                inputs=inputs,
                                device=device,
                                block_dim=self.mesh_triangle_block_dim,
                                record_tape=False,
                            )
                elif direct_meshes:
                    if self._heightfield_reducer_kernel is not None:
                        wp.launch(
                            kernel=self._heightfield_reducer_kernel,
                            dim=self.total_num_threads,
                            inputs=[*triangle_inputs, *support_inputs],
                            device=device,
                            block_dim=self.mesh_triangle_block_dim,
                            record_tape=False,
                        )
                elif self._mesh_face_kernel is not None and not (
                    self._heightfield_pair_csr or self._heightfield_finite_query
                ):
                    self._launch_mesh_triangle_passes(
                        None, self._mesh_face_kernel, self._mesh_queued_kernel, triangle_inputs, support_inputs, device
                    )
                else:
                    wp.launch(
                        kernel=self._pair_csr.generic
                        if self._heightfield_pair_csr
                        else self._finite_reducer_fallback
                        if self._heightfield_finite_query
                        else self._mesh_generic_kernel,
                        dim=self.total_num_threads,
                        inputs=[*triangle_inputs, *support_inputs],
                        device=device,
                        block_dim=self.mesh_triangle_block_dim,
                        record_tape=False,
                    )
            else:
                # Direct contact processing without reduction
                triangle_inputs = [
                    shape_types,
                    shape_data,
                    shape_transform,
                    shape_source,
                    shape_gap,
                    shape_heightfield_index,
                    heightfield_data,
                    heightfield_elevations,
                    self.triangle_pairs,
                    self.triangle_pairs_count,
                    writer_data,
                    self.total_num_threads,
                ]
                if self._heightfield_finite_query:
                    if self._finite_bounds is None:
                        raise RuntimeError("Finite query requires CollisionPipeline model binding")
                    wp.launch(
                        kernel=self._finite_direct,
                        dim=self.total_num_threads,
                        inputs=[*triangle_inputs, self._finite_bounds, self._finite_source],
                        device=device,
                        block_dim=self.mesh_triangle_block_dim,
                        record_tape=False,
                    )
                triangle_kernel = (
                    self._finite_direct_fallback
                    if self._heightfield_finite_query
                    else self._heightfield_triangle_kernel
                    if direct_meshes
                    else self.mesh_triangle_contacts_kernel
                )
                if triangle_kernel is not None:
                    wp.launch(
                        kernel=triangle_kernel,
                        dim=self.total_num_threads,
                        inputs=[*triangle_inputs, *support_inputs],
                        device=device,
                        block_dim=self.mesh_triangle_block_dim,
                        record_tape=False,
                    )

            # Register mesh-plane/mesh-triangle contacts in hashtable BEFORE mesh-mesh.
            # Mesh-mesh does inline hashtable registration in its kernel.
            if self.reduce_contacts:
                if self._heightfield_pair_csr:
                    from .heightfield_pair_csr import reduce_fallback_contacts  # noqa: PLC0415

                    self._pair_csr.build(
                        self.shape_pairs_mesh,
                        self.shape_pairs_mesh_count,
                        shape_types,
                        shape_data,
                        shape_gap,
                        self.total_num_threads,
                    )
                    wp.launch(
                        kernel=reduce_fallback_contacts,
                        dim=self.total_num_threads,
                        inputs=[
                            self._pair_csr.data,
                            shape_transform,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            self.total_num_threads,
                        ],
                        device=device,
                        block_dim=self.block_dim,
                        record_tape=False,
                    )
                elif self.speculative:
                    wp.launch(
                        kernel=reduce_buffered_contacts_speculative_kernel,
                        dim=self.total_num_threads,
                        inputs=[
                            reducer_data,
                            shape_types,
                            shape_data,
                            shape_base_gap,
                            shape_transform,
                            shape_linear_velocity,
                            shape_angular_velocity,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            collision_update_dt,
                            max_speculative_extension,
                            self.total_num_threads,
                        ],
                        device=device,
                        block_dim=self.block_dim,
                        record_tape=False,
                    )
                else:
                    wp.launch(
                        kernel=reduce_buffered_contacts_kernel,
                        dim=self.total_num_threads,
                        inputs=[
                            reducer_data,
                            shape_transform,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            self.total_num_threads,
                        ],
                        device=device,
                        block_dim=self.block_dim,
                        record_tape=False,
                    )

            # Launch mesh-mesh contact processing kernel.
            # The kernel uses texture SDF for fast sampling, with BVH fallback via shape_sdf_index,
            # as well as on-the-fly heightfield evaluation via heightfield_data.
            if texture_sdf_data is None:
                texture_sdf_data = wp.zeros(0, dtype=TextureSDFData, device=device)
            if mesh_edge_indices is None:
                mesh_edge_indices = self._empty_edge_indices
            precomputed_edge_data = (
                mesh_edge_centers,
                mesh_edge_halves,
                mesh_edge_order,
                mesh_edge_clusters,
                mesh_edge_cluster_reach,
            )
            has_precomputed_edge_data = all(array is not None for array in precomputed_edge_data)
            if mesh_edge_centers is None:
                mesh_edge_centers = self._empty_edge_centers
            if mesh_edge_halves is None:
                mesh_edge_halves = self._empty_edge_halves
            if mesh_edge_order is None:
                mesh_edge_order = self._empty_edge_order
            if mesh_edge_clusters is None:
                mesh_edge_clusters = self._empty_edge_clusters
            if mesh_edge_cluster_reach is None:
                mesh_edge_cluster_reach = self._empty_edge_cluster_reach
            if shape_edge_range is None:
                shape_edge_range = self._empty_edge_range

            if self.mesh_mesh_contacts_kernel is not None and self.max_mesh_mesh_pairs > 0:
                mesh_mesh_contacts_kernel = (
                    self.mesh_mesh_contacts_kernel_precomputed
                    if has_precomputed_edge_data
                    else self.mesh_mesh_contacts_kernel
                )
                if self.reduce_contacts and self.mesh_mesh_block_offsets is not None:
                    # Mesh-mesh contacts → buffer + inline hashtable registration
                    compute_mesh_mesh_block_offsets_scan(
                        shape_pairs_mesh_mesh=self.shape_pairs_mesh_mesh,
                        shape_pairs_mesh_mesh_count=self.shape_pairs_mesh_mesh_count,
                        shape_edge_range=shape_edge_range,
                        shape_heightfield_index=shape_heightfield_index,
                        heightfield_data=heightfield_data,
                        target_blocks=self.num_mesh_mesh_blocks,
                        block_offsets=self.mesh_mesh_block_offsets,
                        block_counts=self.mesh_mesh_block_counts,
                        total_edge_count=self.mesh_mesh_total_weight,
                        total_num_threads=self.total_num_threads,
                        device=device,
                        record_tape=False,
                    )

                    wp.launch_tiled(
                        kernel=mesh_mesh_contacts_kernel,
                        dim=(self.num_mesh_mesh_blocks,),
                        inputs=[
                            shape_data,
                            shape_transform,
                            shape_source,
                            texture_sdf_data,
                            shape_sdf_index,
                            shape_mesh_properties,
                            mesh_boundary_flags,
                            shape_mesh_boundary_start,
                            shape_gap,
                            shape_base_gap,
                            shape_linear_velocity,
                            shape_angular_velocity,
                            collision_update_dt,
                            max_speculative_extension,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            self.shape_pairs_mesh_mesh,
                            self.shape_pairs_mesh_mesh_count,
                            shape_heightfield_index,
                            heightfield_data,
                            heightfield_elevations,
                            mesh_edge_indices,
                            mesh_edge_centers,
                            mesh_edge_halves,
                            mesh_edge_order,
                            mesh_edge_clusters,
                            mesh_edge_cluster_reach,
                            shape_edge_range,
                            self.mesh_mesh_block_offsets,
                            reducer_data,
                            self.num_mesh_mesh_blocks,
                        ],
                        device=device,
                        block_dim=self.tile_size_mesh_mesh,
                        record_tape=False,
                    )
                else:
                    # Non-reduce fallback: direct contact write, no dynamic allocation
                    wp.launch_tiled(
                        kernel=mesh_mesh_contacts_kernel,
                        dim=(self.num_tile_blocks,),
                        inputs=[
                            shape_data,
                            shape_transform,
                            shape_source,
                            texture_sdf_data,
                            shape_sdf_index,
                            shape_mesh_properties,
                            mesh_boundary_flags,
                            shape_mesh_boundary_start,
                            shape_gap,
                            shape_base_gap,
                            shape_linear_velocity,
                            shape_angular_velocity,
                            collision_update_dt,
                            max_speculative_extension,
                            shape_collision_aabb_lower,
                            shape_collision_aabb_upper,
                            shape_voxel_resolution,
                            self.shape_pairs_mesh_mesh,
                            self.shape_pairs_mesh_mesh_count,
                            shape_heightfield_index,
                            heightfield_data,
                            heightfield_elevations,
                            mesh_edge_indices,
                            mesh_edge_centers,
                            mesh_edge_halves,
                            mesh_edge_order,
                            mesh_edge_clusters,
                            mesh_edge_cluster_reach,
                            shape_edge_range,
                            writer_data,
                            self.num_tile_blocks,
                        ],
                        device=device,
                        block_dim=self.tile_size_mesh_mesh,
                        record_tape=False,
                    )

            # Export reduced contacts from hashtable
            if self.reduce_contacts:
                # Export has only one writer lane per block, so use a wider grid than
                # the contact-generation kernels. On CPU, tiled kernels expose one lane.
                effective_block_dim = min(self.block_dim, EXPORT_REDUCED_CONTACTS_BLOCK_DIM)
                export_num_blocks = max(
                    1,
                    EXPORT_REDUCED_CONTACTS_THREAD_BUDGET_MULTIPLIER * self.total_num_threads // effective_block_dim,
                )
                if self._reducer_compact_export:
                    # The compact kernel strides by threads rather than blocks.
                    export_num_blocks = self.total_num_threads
                export_inputs = [
                    self.global_contact_reducer.hashtable.keys,
                    self.global_contact_reducer.ht_values,
                    self.global_contact_reducer.hashtable.active_slots,
                    self.global_contact_reducer.position_depth,
                    self.global_contact_reducer.normal,
                    self.global_contact_reducer.shape_pairs,
                    self.global_contact_reducer.contact_fingerprints,
                    self.global_contact_reducer.exported_flags,
                    shape_types,
                    shape_data,
                    shape_gap,
                    writer_data,
                    export_num_blocks,
                    int(self.block_dim > 1),
                    int(self.global_contact_reducer.deterministic),
                ]
                if self._heightfield_pair_csr:
                    wp.launch_tiled(
                        kernel=self._pair_csr.export_kernel,
                        dim=export_num_blocks,
                        inputs=[
                            self._pair_csr.data,
                            self.shape_pairs_mesh_count,
                            shape_types,
                            shape_data,
                            shape_gap,
                            writer_data,
                            export_num_blocks,
                            int(self.block_dim > 1),
                        ],
                        device=device,
                        block_dim=EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
                        record_tape=False,
                    )
                export_launch = wp.launch if self._reducer_compact_export else wp.launch_tiled
                export_launch(
                    kernel=self._pair_csr.fallback_export
                    if self._heightfield_pair_csr
                    else self.export_reduced_contacts_kernel,
                    dim=export_num_blocks,
                    inputs=export_inputs,
                    device=device,
                    block_dim=self.block_dim if self._reducer_compact_export else EXPORT_REDUCED_CONTACTS_BLOCK_DIM,
                    record_tape=False,
                )
        if self.hydroelastic_sdf is not None:
            self.hydroelastic_sdf.launch(
                texture_sdf_data,
                shape_sdf_index,
                shape_data,
                shape_transform,
                shape_gap,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
                self.shape_pairs_sdf_sdf,
                self.shape_pairs_sdf_sdf_count,
                writer_data,
                hydroelastic_shape_sdf_data_prepared,
                verify_contacts=verify_contacts,
            )

        if main_stream is not None:
            main_stream.wait_event(self._convex_stream.record_event(self._convex_events[1]))

        # Verify no collision pipeline buffers overflowed
        if self.verify_buffers:
            if self.global_contact_reducer is not None:
                reduction_ht_active_slots = self.global_contact_reducer.hashtable.active_slots
                reduction_ht_capacity = self.global_contact_reducer.hashtable.capacity
                reduction_ht_insert_failures = self.global_contact_reducer.ht_insert_failures
                reduction_buffer_overflows = self.global_contact_reducer.buffer_overflows
            else:
                reduction_ht_active_slots = self.gjk_candidate_pairs_count
                reduction_ht_capacity = 0
                reduction_ht_insert_failures = self.gjk_candidate_pairs_count
                reduction_buffer_overflows = self.gjk_candidate_pairs_count

            if self.split_gjk_mpr:
                split_query_count = candidate_pair_count if self.sparse_gjk_pairs else self.gjk_candidate_pairs_count
                max_split_query = self.split_query_results.shape[0]
            else:
                split_query_count = self.gjk_candidate_pairs_count
                max_split_query = -1

            wp.launch(
                kernel=record_narrow_phase_buffers,
                dim=[1],
                inputs=[
                    candidate_pair_count,
                    candidate_pair.shape[0],
                    split_query_count,
                    max_split_query,
                    self.gjk_candidate_pairs_count,
                    self.gjk_candidate_pairs.shape[0],
                    self.split_gjk_work_count
                    if self.split_gjk_work_count is not None
                    else self.gjk_candidate_pairs_count,
                    self.split_gjk_work_items.shape[0] if self.split_gjk_work_items is not None else -1,
                    self.split_manifold_work_count
                    if self.split_manifold_work_count is not None
                    else self.gjk_candidate_pairs_count,
                    self.split_manifold_work_items.shape[0] if self.split_manifold_work_items is not None else -1,
                    self.shape_pairs_mesh_count,
                    self.shape_pairs_mesh.shape[0] if self.shape_pairs_mesh is not None else 0,
                    mesh_buffer_report,
                    self.triangle_pairs.shape[0] if self.triangle_pairs is not None else 0,
                    self.shape_pairs_mesh_plane_count,
                    self.shape_pairs_mesh_plane.shape[0] if self.shape_pairs_mesh_plane is not None else 0,
                    self.shape_pairs_mesh_mesh_count,
                    self.shape_pairs_mesh_mesh.shape[0] if self.shape_pairs_mesh_mesh is not None else 0,
                    self.shape_pairs_sdf_sdf_count,
                    self.shape_pairs_sdf_sdf.shape[0] if self.shape_pairs_sdf_sdf is not None else 0,
                    writer_data.contact_count,
                    writer_data.contact_max if verify_contacts else 2**31 - 1,
                    reduction_ht_active_slots,
                    reduction_ht_capacity,
                    reduction_ht_insert_failures,
                    reduction_buffer_overflows,
                    HASHTABLE_WARN_LOAD_PERCENT,
                    self._buffer_capacity_status,
                ],
                device=device,
                record_tape=False,
            )

    def buffer_capacity_status(self, *, clear: bool = False) -> dict[str, bool]:
        """Read sticky flags from the existing narrow-phase buffer verifier.

        Keep ``verify_buffers=True`` for the entire observed run. Call outside
        graph capture on the execution stream or after joining it. Flags survive
        later clean calls and graph replays. Clearing acknowledges prior warnings;
        it does not repair results, resize buffers, or recapture graphs.

        This covers :func:`record_narrow_phase_buffers` and the optional pair-CSR
        raw-witness/membership guard: not soft contacts, hydroelastic internal
        storage, or post-narrow-phase owners such as body pair reduction/contact
        matching. ``reduction_hash_load`` is a load warning,
        not proof that contacts were lost. Other flags identify exceeded buffers
        or failed reducer insertions. No flag is an estimate of required capacity.
        ``terrain_rows`` (the terrain law's spill region overflowed, see
        ``NEWTON_TERRAIN_LAW_SHADOW_ROWS``, or a pair's window outgrew the bound
        its shape data gave at construction), ``terrain_stale`` (terrain
        features older than the heights were read) and ``terrain_jobs`` (a
        solid class's job lists or candidate pool had no room for a pair's
        work, see ``NEWTON_TERRAIN_LAW_SHADOW_JOBS``: the pair's rows are
        unchanged while the lists' reserve takes it; past the reserve the
        pair keeps no row) are set by the terrain law's
        kernels, which write their partners' heightfield contacts (by
        default, see ``NEWTON_TERRAIN_LAW``; the ``contacts`` bit then also
        covers them) or run in shadow beside the terrain route
        (``NEWTON_TERRAIN_LAW_SHADOW``). ``contact_launches`` (a narrow phase of a
        :class:`~newton.CollisionPipeline`): a contact-sized launch of a captured
        graph, the pipeline's or a solver's sharing its contact rows, that the
        graph's last replay skipped, because its device graph updater failed
        closed after the capture emitted it or its count passed the captured
        bound; clearing acknowledges it until a replay reports it again.
        """
        if not self.verify_buffers:
            raise RuntimeError("Narrow-phase capacity status requires verify_buffers=True throughout the run")
        if self._buffer_capacity_status.device.is_capturing:
            raise RuntimeError("Narrow-phase capacity status must be read outside CUDA graph capture")
        mask = int(self._buffer_capacity_status.numpy()[0])
        names = (
            "broad_phase",
            "split_query",
            "gjk",
            "split_gjk",
            "split_manifold",
            "mesh",
            "triangle",
            "mesh_plane",
            "mesh_mesh",
            "sdf_sdf",
            "contacts",
            "reduction_hash_load",
            "reduction_hash_insert",
            "terrain_rows",
            "terrain_stale",
            "terrain_jobs",
            "reduction_buffer",
        )
        result = {name: bool(mask & (1 << bit)) for bit, name in enumerate(names)}
        if self._heightfield_pair_csr:
            result["heightfield_pair_csr_raw_or_membership"] = int(self._pair_csr.status.numpy()[0]) != 0
        if self.contact_pool is not None:
            result["contact_launches"] = self.contact_pool.launch_failed(clear=clear)
        if clear:
            self._buffer_capacity_status.zero_()
            if self._heightfield_pair_csr:
                self._pair_csr.status.zero_()
        return result

    def _mesh_triangles_direct(self) -> bool:
        """Whether mesh triangle pairs publish their contacts directly, without the global reducer.

        Without reduction they always do. The heightfield pair-CSR and finite routes keep their own mesh passes.
        """
        if self._mesh_face_direct is None or self._heightfield_pair_csr or self._heightfield_finite_query:
            return False
        return not self.reduce_contacts

    def _launch_mesh_triangle_passes(self, round_kernel, face_kernel, queued_kernel, inputs, support_inputs, device):
        """Launch the mesh triangle passes over ``inputs`` (the triangle query's): the round-shape kernel (if any),
        the face kernel, which compacts the rest into the queue, and the GJK/MPR kernel over the queue."""
        launches = []
        if round_kernel is not None:
            launches.append((round_kernel, [*inputs, *support_inputs]))
        launches.append((face_kernel, [*inputs, *support_inputs, self._mesh_queued_pairs, self._mesh_queued_count]))
        queued_inputs = [*inputs]
        queued_inputs[8] = self._mesh_queued_pairs
        queued_inputs[9] = self._mesh_queued_count
        launches.append((queued_kernel, [*queued_inputs, *support_inputs]))
        self._mesh_queued_count.zero_()
        for kernel, kernel_inputs in launches:
            wp.launch(
                kernel=kernel,
                dim=self.total_num_threads,
                inputs=kernel_inputs,
                device=device,
                block_dim=self.mesh_triangle_block_dim,
                record_tape=False,
            )

    def set_terrain_law_partners(self, partner: wp.array | None) -> None:
        """Leave the heightfield pairs of some partner shapes to the terrain law, which writes their contacts.

        Set outside graph capture, before the first capture: a captured collide keeps the array it launched with.

        Args:
            partner: Per shape, nonzero for a partner whose heightfield pairs the primitive kernel drops instead of
                routing them to the terrain route (int32, one entry per shape); ``None`` routes every pair.
        """
        if partner is None:
            self._terrain_law_partner = wp.zeros(0, dtype=wp.int32, device=self.device)
            return
        if partner.dtype != wp.int32 or partner.ndim != 1:
            raise ValueError("The terrain law's partner mask must be a one-dimensional int32 array")
        self._terrain_law_partner = partner

    def check_buffer_capacity(self) -> None:
        """Raise at a safe host boundary if any narrow-phase verification flag is set.

        This adds no device work beyond the enabled verifier. An error does not
        automatically abort a captured graph; discard affected results and correct
        the allocation before rebuilding and recapturing. Hash-load warnings may
        precede actual loss, so inspect the named condition before sizing storage.
        """
        failed = [name for name, flagged in self.buffer_capacity_status().items() if flagged]
        if failed:
            raise RuntimeError("Narrow-phase buffer verification failed: " + ", ".join(failed))

    def launch(
        self,
        *,
        candidate_pair: wp.array[wp.vec2i],  # Maybe colliding pairs
        candidate_pair_count: wp.array[wp.int32],  # Size one array
        shape_types: wp.array[wp.int32],  # All shape types, pairs index into it
        shape_data: wp.array[wp.vec4],  # Shape data (scale xyz, margin w)
        shape_transform: wp.array[wp.transform],  # In world space
        shape_source: wp.array[wp.uint64],  # The index into the source array, type define by shape_types
        shape_support_data: wp.array[wp.vec4i] | None = None,
        support_lut: wp.array[wp.int32] | None = None,
        support_vertex_offsets: wp.array[wp.int32] | None = None,
        support_neighbors: wp.array[wp.int32] | None = None,
        shape_mesh_properties: wp.array[wp.int32] | None = None,  # Per-shape mesh property bitfield
        mesh_boundary_flags: wp.array[wp.uint8] | None = None,  # Packed per-face open-boundary bits
        shape_mesh_boundary_start: wp.array[wp.int32] | None = None,  # Per-shape start into the bits
        shape_sdf_index: wp.array[wp.int32] | None = None,  # Per-shape index into texture_sdf_data (-1 for none)
        texture_sdf_data: wp.array[TextureSDFData] | None = None,  # Compact texture SDF data table
        shape_gap: wp.array[wp.float32],  # per-shape contact gap (detection threshold)
        shape_base_gap: wp.array[wp.float32] | None = None,
        shape_collision_radius: wp.array[wp.float32],  # per-shape collision radius for AABB fallback
        shape_flags: wp.array[wp.int32],  # per-shape flags (includes ShapeFlags.HYDROELASTIC)
        shape_collision_aabb_lower: wp.array[wp.vec3] | None = None,  # Local-space AABB lower bounds
        shape_collision_aabb_upper: wp.array[wp.vec3] | None = None,  # Local-space AABB upper bounds
        shape_voxel_resolution: wp.array[wp.vec3i],  # Voxel grid resolution per shape
        shape_interior_point: wp.array[wp.vec3] | None = None,  # Local-space interior points
        # Outputs
        contact_pair: wp.array[wp.vec2i],
        contact_position: wp.array[wp.vec3],
        contact_normal: wp.array[
            wp.vec3
        ],  # Pointing from pairId.x to pairId.y, represents z axis of local contact frame
        contact_penetration: wp.array[float],  # negative if bodies overlap
        contact_count: wp.array[int],  # Number of active contacts after narrow
        contact_tangent: wp.array[wp.vec3] | None = None,  # Represents x axis of local contact frame (None to disable)
        shape_linear_velocity: wp.array[wp.vec3] | None = None,
        shape_angular_velocity: wp.array[wp.vec3] | None = None,
        collision_update_dt: float = 0.0,
        max_speculative_extension: float = 0.0,
        device: Devicelike | None = None,  # Device to launch on
        **kwargs: Any,
    ) -> None:
        """
        Launch narrow phase collision detection on candidate pairs from broad phase.

        Args:
            candidate_pair: Array of potentially colliding shape pairs from broad phase
            candidate_pair_count: Single-element array containing the number of candidate pairs
            shape_types: Array of geometry types for all shapes
            shape_data: Array of vec4 containing scale (xyz) and margin (w) for each shape
            shape_transform: Array of world-space transforms for each shape
            shape_source: Array of source pointers (mesh IDs, etc.) for each shape
            shape_support_data: Optional per-shape ``wp.vec4i`` metadata containing
                packed support-LUT, vertex-offset, and ``support_neighbors`` block starts
                plus the LUT resolution. The block start is also set for verified hulls
                without acceleration. The final entry packs resolution in its low 16 bits
                and strict hull validity in bit 16. Zero resolution disables acceleration.
            support_lut: Optional packed octahedral-direction lookup tables whose entries
                select seed vertices for convex support walks.
            support_vertex_offsets: Optional packed CSR offsets delimiting each convex
                vertex's entries in ``support_neighbors``. They index past the hull's
                edge records.
            support_neighbors: Optional packed per-hull blocks. A verified hull's block
                starts with its edge count and one ``(a, b, c1, c2)`` vertex record per
                edge, whose faces ``cross(b - a, c1 - a)`` and ``cross(c2 - a, b - a)``
                point outward; exact box-hull depth certificates read these records.
                CSR vertex-adjacency indices for support walks follow. Without the edge
                records, box-hull depths fall back to uncertified support-plane bounds.
            shape_mesh_properties: Per-shape mesh property bitfield.
            mesh_boundary_flags: Optional packed per-face bits marking open-boundary
                edges and vertices of meshes queried with pseudo-normal signs.
            shape_mesh_boundary_start: Optional per-shape offset into
                ``mesh_boundary_flags``, or -1 for shapes without boundary bits.
                Without both arrays, pseudo-normal signs extend past open boundaries.
            shape_sdf_index: Per-shape SDF table index (-1 for shapes without SDF)
            texture_sdf_data: Compact array of TextureSDFData structs
            shape_gap: Array of per-shape contact gaps (detection threshold) for each shape
            shape_base_gap: Authored per-shape contact gaps. Required when
                this narrow phase was constructed with ``speculative=True``.
            shape_collision_radius: Array of collision radii for each shape (for AABB fallback for planes/meshes)
            shape_collision_aabb_lower: Local-space AABB lower bounds for each shape (for voxel binning)
            shape_collision_aabb_upper: Local-space AABB upper bounds for each shape (for voxel binning)
            shape_voxel_resolution: Voxel grid resolution for each shape (for voxel binning)
            shape_interior_point: Optional scaled local-space point inside each shape [m],
                used to cull back-facing mesh triangles and seed triangle contacts. If
                ``None``, convex-mesh volume centroids are computed on each launch.
            mesh_edge_indices: Packed mesh edge vertex pairs, passed through ``kwargs``.
                Required with ``shape_edge_range`` for mesh edge contacts; omitting either
                disables mesh edge contacts.
            shape_edge_range: Per-shape (start, count) into ``mesh_edge_indices``,
                passed through ``kwargs``.
            mesh_edge_centers: Optional packed unscaled edge centers (xyz) and half-lengths
                (w) [m], passed through ``kwargs``. Omit with ``mesh_edge_halves`` to read
                endpoints from meshes using the supplied indices and ranges.
            mesh_edge_halves: Optional packed unscaled edge half-vectors (xyz) [m] and
                endpoint-ownership codes (w), passed through ``kwargs``. Zero allows both endpoints.
            contact_pair: Output array for contact shape pairs
            contact_position: Output array for contact positions (center point)
            contact_normal: Output array for contact normals
            contact_penetration: Output array for penetration depths
            contact_tangent: Output array for contact tangents, or None to disable tangent computation
            contact_count: Output array (single element) for contact count
            shape_linear_velocity: World-space shape-origin linear velocity [m/s].
            shape_angular_velocity: World-space shape angular velocity [rad/s].
            collision_update_dt: Speculative collision horizon [s].
            max_speculative_extension: Maximum speculative clearance [m].
            device: Device to launch on
        """
        if device is None:
            device = self.device if self.device is not None else candidate_pair.device

        # Backward compatibility for older call sites/tests that still pass
        # shape_local_aabb_lower/upper.
        shape_local_aabb_lower = kwargs.pop("shape_local_aabb_lower", None)
        shape_local_aabb_upper = kwargs.pop("shape_local_aabb_upper", None)
        mesh_edge_indices = kwargs.pop("mesh_edge_indices", None)
        mesh_edge_centers = kwargs.pop("mesh_edge_centers", None)
        mesh_edge_halves = kwargs.pop("mesh_edge_halves", None)
        mesh_edge_order = kwargs.pop("mesh_edge_order", None)
        mesh_edge_clusters = kwargs.pop("mesh_edge_clusters", None)
        mesh_edge_cluster_reach = kwargs.pop("mesh_edge_cluster_reach", None)
        shape_edge_range = kwargs.pop("shape_edge_range", None)
        mesh_cells = kwargs.pop("mesh_cells", None)
        mesh_topology = kwargs.pop("mesh_topology", None)
        if kwargs:
            unknown_keys = sorted(kwargs.keys())
            if len(unknown_keys) == 1:
                raise TypeError(f"NarrowPhase.launch() got an unexpected keyword argument '{unknown_keys[0]}'")
            unknown = ", ".join(unknown_keys)
            raise TypeError(f"NarrowPhase.launch() got unexpected keyword arguments: {unknown}")

        if shape_collision_aabb_lower is None:
            shape_collision_aabb_lower = shape_local_aabb_lower
        if shape_collision_aabb_upper is None:
            shape_collision_aabb_upper = shape_local_aabb_upper
        if shape_collision_aabb_lower is None or shape_collision_aabb_upper is None:
            raise TypeError(
                "NarrowPhase.launch() missing required AABB bounds: provide either "
                "shape_collision_aabb_lower/shape_collision_aabb_upper or "
                "shape_local_aabb_lower/shape_local_aabb_upper"
            )
        if shape_sdf_index is None:
            shape_sdf_index = wp.full(shape_types.shape[0], -1, dtype=wp.int32, device=device)
        if self.speculative and shape_base_gap is None:
            raise ValueError("Speculative NarrowPhase requires the authored shape_base_gap array")
        if shape_base_gap is None:
            shape_base_gap = shape_gap
        if self.speculative and (shape_linear_velocity is None or shape_angular_velocity is None):
            raise ValueError("Speculative NarrowPhase requires per-shape linear/angular velocity arrays")
        shape_linear_velocity = self._empty_vec3 if shape_linear_velocity is None else shape_linear_velocity
        shape_angular_velocity = self._empty_vec3 if shape_angular_velocity is None else shape_angular_velocity

        contact_max = contact_pair.shape[0]

        # Handle optional tangent array - use empty array if None
        if contact_tangent is None:
            contact_tangent = self.empty_tangent

        # Clear external contact count (internal counters are cleared in launch_custom_write)
        contact_count.zero_()

        # Verify sort-key buffer and sorter match the contact output capacity.
        # Raising instead of silently reallocating keeps this path
        # CUDA-graph-capturable and consistent with CollisionPipeline.collide().
        if self.deterministic and self._sort_key_array.shape[0] != contact_max:
            raise ValueError(
                f"Contact output capacity ({contact_max}) does not match the "
                f"deterministic sort buffer size ({self._sort_key_array.shape[0]}). "
                f"The sorter operates over fixed-capacity buffers for CUDA graph capture "
                f"compatibility, so the sizes must match exactly."
            )

        # Create ContactWriterData struct
        sort_key_arr = self._sort_key_array if self.deterministic else self._empty_sort_key

        writer_data = ContactWriterData()
        writer_data.contact_max = contact_max
        writer_data.shape_index_bits = self._contact_sort_shape_index_bits
        writer_data.sub_key_bits = self._contact_sort_sub_key_bits
        writer_data.contact_count = contact_count
        writer_data.contact_pair = contact_pair
        writer_data.contact_position = contact_position
        writer_data.contact_normal = contact_normal
        writer_data.contact_penetration = contact_penetration
        writer_data.contact_tangent = contact_tangent
        writer_data.contact_sort_key = sort_key_arr
        writer_data.shape_gap = shape_base_gap
        writer_data.shape_transform = shape_transform
        writer_data.shape_linear_velocity = shape_linear_velocity
        writer_data.shape_angular_velocity = shape_angular_velocity
        writer_data.collision_update_dt = collision_update_dt
        writer_data.max_speculative_extension = max_speculative_extension

        # Delegate to launch_custom_write
        self.launch_custom_write(
            candidate_pair=candidate_pair,
            candidate_pair_count=candidate_pair_count,
            shape_types=shape_types,
            shape_data=shape_data,
            shape_transform=shape_transform,
            shape_source=shape_source,
            shape_support_data=shape_support_data,
            support_lut=support_lut,
            support_vertex_offsets=support_vertex_offsets,
            support_neighbors=support_neighbors,
            shape_mesh_properties=shape_mesh_properties,
            mesh_boundary_flags=mesh_boundary_flags,
            shape_mesh_boundary_start=shape_mesh_boundary_start,
            shape_sdf_index=shape_sdf_index,
            texture_sdf_data=texture_sdf_data,
            shape_gap=shape_gap,
            shape_base_gap=shape_base_gap,
            shape_collision_radius=shape_collision_radius,
            shape_flags=shape_flags,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_interior_point=shape_interior_point,
            shape_voxel_resolution=shape_voxel_resolution,
            mesh_edge_indices=mesh_edge_indices,
            mesh_edge_centers=mesh_edge_centers,
            mesh_edge_halves=mesh_edge_halves,
            mesh_edge_order=mesh_edge_order,
            mesh_edge_clusters=mesh_edge_clusters,
            mesh_edge_cluster_reach=mesh_edge_cluster_reach,
            shape_edge_range=shape_edge_range,
            mesh_cells=mesh_cells,
            mesh_topology=mesh_topology,
            shape_linear_velocity=shape_linear_velocity,
            shape_angular_velocity=shape_angular_velocity,
            collision_update_dt=collision_update_dt,
            max_speculative_extension=max_speculative_extension,
            writer_data=writer_data,
            device=device,
        )

        if self.deterministic:
            self._contact_sorter.sort_simple(
                sort_key_arr,
                contact_count,
                contact_pair=contact_pair,
                contact_position=contact_position,
                contact_normal=contact_normal,
                contact_penetration=contact_penetration,
                contact_tangent=contact_tangent,
                device=device,
            )

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import warnings
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

from ..core.contact_pool import ContactDrops, ContactField, ContactPool, _capture_owner, capturing, model_contact_pool
from ..core.environment_flags import record_newton_environment, retired_notice
from ..core.reset import normalize_reset_world_mask
from ..exceptions import NewtonDeprecationWarning, NewtonWarning
from ..geometry.broad_phase_common import (
    FORECAST_FLOATS,
    FORECAST_HYDROELASTIC,
    FORECAST_REDUCED,
    FORECAST_SLOTS_SHIFT,
    FORECAST_TERRAIN,
    ShapeForecast,
    forecast_shape,
    forecasting,
)
from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit, BroadPhaseExplicitTemplates
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.collision_core import compute_tight_aabb_from_support
from ..geometry.contact_data import (
    CONTACT_SORT_SUB_KEY_BITS,
    ContactData,
    contact_passes_speculative_gap_check,
    contact_sort_shape_index_bits,
    make_contact_sort_key_with_bits,
    prepare_speculative_contact,
)
from ..geometry.contact_match import ContactMatcher
from ..geometry.contact_reduction import MAX_CONTACTS_PER_PAIR, NUM_NORMAL_BINS
from ..geometry.contact_reduction_body_pairs import (
    MAX_GROUP_ID,
    BodyPairContactReducer,
    build_reduction_group_bound,
    build_reduction_groups,
)
from ..geometry.contact_reduction_global import CONTACT_ID_BITS
from ..geometry.contact_sort import ContactSorter
from ..geometry.differentiable_contacts import launch_differentiable_contact_augment
from ..geometry.flags import MeshProperties, ShapeFlags
from ..geometry.kernels import create_soft_contacts
from ..geometry.narrow_phase import NarrowPhase
from ..geometry.sdf_hydroelastic import HydroelasticSDF
from ..geometry.soft_contacts_heightfield import _HEIGHTFIELD_CELLS_PER_TASK, launch_soft_heightfield_contacts
from ..geometry.soft_contacts_mesh import (
    MeshContactData,
    allocate_soft_mesh_contact_buffers,
    launch_soft_mesh_contacts,
)
from ..geometry.soft_contacts_sdf import _SDF_COMPACTION_MIN_PAIRS, _SDF_SPECIALIZED_GEO_TYPES, launch_soft_ef_contacts
from ..geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    pack_mesh_ptr,
)
from ..geometry.terrain_contacts import (
    TERRAIN_PARTNER_BOX,
    TERRAIN_PARTNER_CYLINDER,
    TERRAIN_PARTNER_HULL,
    TERRAIN_PARTNER_POINT,
    TERRAIN_PARTNER_WALK_SLOTS,
)
from ..geometry.tri_mesh_collision import TriMeshCollisionDetector
from ..geometry.types import GeoType
from ..sim.contacts import GENERATION_SENTINEL, Contacts, note_contact_frame_write
from ..sim.enums import BodyFlags, JointType, ModelFlags
from ..sim.model import Model
from ..sim.shape_contact_pairs import (
    _shape_contact_pair_count_for_mask,
    _shape_contact_pair_counts,
    _shape_contact_pair_weight_sum,
)
from ..sim.state import State

if TYPE_CHECKING:
    from ..solvers.coupled.model_view import ModelView

logger = logging.getLogger(__name__)


@wp.kernel
def _reduction_reset_mask_from_matching(
    mask: wp.array[wp.bool],
    world_count: int,
    reduction_mask: wp.array[wp.int32],
):
    world = wp.tid()
    # A changed global collider may participate in every world's patches.
    reduction_mask[world] = int(mask[world_count] or (world < world_count and mask[world]))


def _shape_collide_mask(model: Model, shape_count: int | None = None) -> np.ndarray:
    """Return a host mask for shapes participating in shape-shape collision.

    A variant row counts when any variant shape it can hold collides, since a selection can
    change its flags after construction.
    """
    shape_flags = getattr(model, "shape_flags", None)
    if shape_flags is None:
        count = model.shape_count if shape_count is None else shape_count
        return np.ones(count, dtype=bool)

    flags = model._shape_option_flags() if model._has_variant_rows else shape_flags.numpy()
    if shape_count is not None and len(flags) != shape_count:
        raise ValueError("model.shape_flags and model.shape_type must have the same length")
    return (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0


@dataclasses.dataclass(frozen=True)
class _VariantOptions:
    """The variant shapes that variant rows can hold, so that routing and capacities cover all of them.

    A row of a body can hold the variant shape of its role at its position in any of the body's variants.
    """

    shape_count: int
    row_shapes: np.ndarray
    """Variant row of each option."""
    variant_shapes: np.ndarray
    """Variant shape of each option."""
    variant_types: np.ndarray
    """Shape type of each variant shape."""
    variant_collides: np.ndarray
    """Whether each variant shape collides with shapes."""
    row_mask: np.ndarray
    """Whether each shape is a variant row."""

    def any(self, variant_mask: np.ndarray) -> np.ndarray:
        """Per shape, whether it is a variant row that can hold a variant shape in ``variant_mask``."""
        mask = np.zeros(self.shape_count, dtype=bool)
        mask[self.row_shapes[variant_mask[self.variant_shapes]]] = True
        return mask

    def may_be(self, shape_types: np.ndarray, types: tuple[int, ...]) -> np.ndarray:
        """Per shape, whether it is or can become one of ``types``, counting only colliding variant shapes."""
        return np.isin(shape_types, types) | self.any(self.variant_collides & np.isin(self.variant_types, types))

    def reachable(self, shape_mask: np.ndarray) -> np.ndarray:
        """Per variant shape, whether a variant row in ``shape_mask`` can hold it."""
        mask = np.zeros(len(self.variant_types), dtype=bool)
        mask[self.variant_shapes[shape_mask[self.row_shapes]]] = True
        return mask

    def type_bits(self) -> tuple[np.ndarray, np.ndarray]:
        """Variant rows and, for each, a bit per type of the colliding variant shapes it can hold."""
        bits = np.zeros(self.shape_count, dtype=np.int64)
        option_bits = np.where(
            self.variant_collides[self.variant_shapes],
            np.left_shift(1, self.variant_types[self.variant_shapes].astype(np.int64)),
            0,
        )
        np.bitwise_or.at(bits, self.row_shapes, option_bits)
        rows = np.flatnonzero(self.row_mask)
        return rows, bits[rows]


def _variant_options(model: Model) -> _VariantOptions | None:
    """The variant shapes the model's variant rows can hold, or ``None`` without variant rows."""
    if not model._has_variant_rows:
        return None
    row_shapes, variant_shapes = model._variant_row_options()
    row_mask = np.zeros(model.shape_count, dtype=bool)
    row_mask[row_shapes] = True
    geometry = model._variant_shape_geometry
    return _VariantOptions(
        shape_count=model.shape_count,
        row_shapes=row_shapes,
        variant_shapes=variant_shapes,
        variant_types=geometry["shape_type"].numpy(),
        variant_collides=(geometry["shape_flags"].numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0,
        row_mask=row_mask,
    )


def _shape_may_be_mesh(model: Model, shape_types: np.ndarray, variant_options: _VariantOptions | None) -> np.ndarray:
    """Shapes that are, or as variant rows can become, meshes or heightfields."""
    mesh_types = (int(GeoType.MESH), int(GeoType.HFIELD))
    return (
        np.isin(shape_types, mesh_types) if variant_options is None else variant_options.may_be(shape_types, mesh_types)
    )


_ANALYTIC_PRIMITIVE_PAIRS = frozenset(
    {
        (int(GeoType.PLANE), int(GeoType.SPHERE)),
        (int(GeoType.PLANE), int(GeoType.CAPSULE)),
        (int(GeoType.PLANE), int(GeoType.ELLIPSOID)),
        (int(GeoType.PLANE), int(GeoType.BOX)),
        (int(GeoType.SPHERE), int(GeoType.SPHERE)),
        (int(GeoType.SPHERE), int(GeoType.CAPSULE)),
        (int(GeoType.SPHERE), int(GeoType.BOX)),
        (int(GeoType.CAPSULE), int(GeoType.CAPSULE)),
    }
)


def _pair_requires_generic_convex_narrow_phase(
    type_a: int,
    type_b: int,
) -> bool:
    """Return whether a sorted shape-type pair can reach GJK/MPR."""
    type_a, type_b = min(type_a, type_b), max(type_a, type_b)
    if type_a in (int(GeoType.HFIELD), int(GeoType.MESH)):
        return False
    if type_b in (int(GeoType.HFIELD), int(GeoType.MESH)):
        return False
    return (type_a, type_b) not in _ANALYTIC_PRIMITIVE_PAIRS


# Indexed by raw shape type, so GeoType values must be their own positions.
_GENERIC_CONVEX_PAIR_LOOKUP = np.array(
    [
        [_pair_requires_generic_convex_narrow_phase(int(type_a), int(type_b)) for type_b in GeoType]
        for type_a in GeoType
    ],
    dtype=bool,
)


_PAIR_SCAN_BLOCK = 65536


def _iter_pair_blocks(pairs: wp.array[wp.vec2i] | np.ndarray) -> Iterator[np.ndarray]:
    """Yield a pair list as host blocks of shape ``(n, 2)``.

    A device list is copied one block at a time, so a scan's host memory stays bounded
    however many worlds replicate the list.
    """
    if not isinstance(pairs, wp.array):
        pairs = (pairs.numpy() if hasattr(pairs, "numpy") else np.asarray(pairs)).reshape(-1, 2)
    count = len(pairs)
    for begin in range(0, count, _PAIR_SCAN_BLOCK):
        # Warp slices do not clamp their end.
        block = pairs[begin : min(begin + _PAIR_SCAN_BLOCK, count)]
        yield (block.numpy() if isinstance(block, wp.array) else block).reshape(-1, 2)


@wp.struct
class ContactWriterData:
    """Contact writer data for collide write_contact function.

    ``contact_max`` is the most rows the buffer's addresses span (its pool's
    bound), which only stops the narrow phase's work early; ``contact_limit``
    holds the rows writable now (its pool's ready rows, on the device), and the
    writer drops every row past it. A writer without ``contact_limit`` (an empty
    array) drops the rows past ``contact_max``. A writer that drops a contact
    latches its world in ``dropped_world`` with the pass (one store), so no drop
    is silent (:meth:`~newton._src.core.contact_pool.ContactPool.drops`).
    """

    contact_max: int
    shape_index_bits: int
    sub_key_bits: int
    contact_limit: wp.array[wp.int32]
    # The pool's per-world status of dropped contacts (empty: none), the shapes' worlds and the passes recorded.
    dropped_world: wp.array[wp.int32]
    shape_world: wp.array[int]
    pass_count: wp.array[wp.int32]
    # Body information arrays (for transforming to body-local coordinates)
    body_q: wp.array[wp.transform]
    shape_body: wp.array[int]
    shape_gap: wp.array[float]
    # Output arrays
    contact_count: wp.array[int]
    out_shape0: wp.array[int]
    out_shape1: wp.array[int]
    out_point0: wp.array[wp.vec3]
    out_point1: wp.array[wp.vec3]
    out_offset0: wp.array[wp.vec3]
    out_offset1: wp.array[wp.vec3]
    out_normal: wp.array[wp.vec3]
    out_margin0: wp.array[float]
    out_margin1: wp.array[float]
    out_tids: wp.array[int]
    # Per-contact shape properties, empty arrays if not enabled.
    # Zero-values indicate that no per-contact shape properties are set for this contact
    out_stiffness: wp.array[float]
    out_damping: wp.array[float]
    out_friction: wp.array[float]
    out_sort_key: wp.array[wp.int64]
    # Speculative-contact inputs. Empty arrays and zero scalars when disabled.
    shape_transform: wp.array[wp.transform]
    shape_linear_velocity: wp.array[wp.vec3]
    shape_angular_velocity: wp.array[wp.vec3]
    collision_update_dt: float
    max_speculative_extension: float


_NO_FORECAST = ShapeForecast()
"""A forecast with no arrays: the pass forecasts nothing."""


@wp.func
def _latch_dropped_world(writer_data: ContactWriterData, shape_a: int, shape_b: int):
    """Latch the world of a dropped contact with the pass that dropped it; the last entry takes contacts of no world."""
    last = writer_data.dropped_world.shape[0] - 1
    world = writer_data.shape_world[shape_a]
    if world < 0:
        world = writer_data.shape_world[shape_b]
    if world < 0 or world >= last:
        world = last
    writer_data.dropped_world[world] = writer_data.pass_count[0] + 1


@wp.func
def _write_contact_at_index(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    index: int,
    point_a_world: wp.vec3,
    point_b_world: wp.vec3,
    normal_a_to_b: wp.vec3,
):
    """Write a previously accepted contact at a reserved output index, or latch its world if the rows ran out."""
    limit = writer_data.contact_max
    if writer_data.contact_limit.shape[0] > 0:
        limit = writer_data.contact_limit[0]
    if index >= limit:
        if writer_data.dropped_world.shape[0] > 0:
            _latch_dropped_world(writer_data, contact_data.shape_a, contact_data.shape_b)
        return

    writer_data.out_shape0[index] = contact_data.shape_a
    writer_data.out_shape1[index] = contact_data.shape_b

    body0 = writer_data.shape_body[contact_data.shape_a]
    body1 = writer_data.shape_body[contact_data.shape_b]
    X_bw_a = wp.transform_identity() if body0 == -1 else wp.transform_inverse(writer_data.body_q[body0])
    X_bw_b = wp.transform_identity() if body1 == -1 else wp.transform_inverse(writer_data.body_q[body1])

    writer_data.out_point0[index] = wp.transform_point(X_bw_a, point_a_world)
    writer_data.out_point1[index] = wp.transform_point(X_bw_b, point_b_world)

    offset_mag_a = contact_data.radius_eff_a + contact_data.margin_a
    offset_mag_b = contact_data.radius_eff_b + contact_data.margin_b
    writer_data.out_offset0[index] = wp.transform_vector(X_bw_a, offset_mag_a * normal_a_to_b)
    writer_data.out_offset1[index] = wp.transform_vector(X_bw_b, -offset_mag_b * normal_a_to_b)
    writer_data.out_normal[index] = normal_a_to_b
    writer_data.out_margin0[index] = offset_mag_a
    writer_data.out_margin1[index] = offset_mag_b
    writer_data.out_tids[index] = 0

    if writer_data.out_stiffness.shape[0] > 0:
        writer_data.out_stiffness[index] = contact_data.contact_stiffness
        writer_data.out_damping[index] = contact_data.contact_damping
        writer_data.out_friction[index] = contact_data.contact_friction_scale

    if writer_data.out_sort_key.shape[0] > 0:
        writer_data.out_sort_key[index] = make_contact_sort_key_with_bits(
            contact_data.shape_a,
            contact_data.shape_b,
            contact_data.sort_sub_key,
            writer_data.shape_index_bits,
            writer_data.sub_key_bits,
        )


@wp.func
def write_contact(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    output_index: int,
):
    """
    Write a contact to the output arrays using ContactData and ContactWriterData.

    Args:
        contact_data: ContactData struct containing contact information
        writer_data: ContactWriterData struct containing body info and output arrays
        output_index: If -1, use atomic_add to get the next available index if contact distance is less than margin. If >= 0, use this index directly and skip margin check.
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )

    # Distance calculation matching box_plane_collision
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

    # Use per-shape contact gaps (sum of both shapes)
    gap_a = writer_data.shape_gap[contact_data.shape_a]
    gap_b = writer_data.shape_gap[contact_data.shape_b]
    contact_gap = gap_a + gap_b

    index = output_index

    if index < 0:
        # compute index using atomic counter
        if d > contact_gap:
            return
        index = wp.atomic_add(writer_data.contact_count, 0, 1)
    _write_contact_at_index(contact_data, writer_data, index, a_contact_world, b_contact_world, contact_normal_a_to_b)


@wp.func
def write_contact_speculative(
    contact_data: ContactData,
    writer_data: ContactWriterData,
    output_index: int,
):
    """Write a present or exactly predicted contact to the output arrays."""
    contact_data.gap_sum = writer_data.shape_gap[contact_data.shape_a] + writer_data.shape_gap[contact_data.shape_b]
    normal, point_a_world, point_b_world, _separation = prepare_speculative_contact(contact_data)

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

    _write_contact_at_index(contact_data, writer_data, index, point_a_world, point_b_world, normal)


@wp.func
def _mesh_surface_velocity(
    shape: int,
    point_world: wp.vec3,
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[int],
) -> wp.vec3:
    """Interpolate a mesh shape's local surface velocity at a world-space point."""
    if shape_type[shape] != GeoType.MESH or not (shape_mesh_properties[shape] & int(MeshProperties.SURFACE_VELOCITY)):
        return wp.vec3(0.0)

    mesh_id = shape_source_ptr[shape]
    if mesh_id == wp.uint64(0):
        return wp.vec3(0.0)

    scale = shape_scale[shape]
    X_ws = shape_transform[shape]
    point_local = wp.cw_div(wp.transform_point(wp.transform_inverse(X_ws), point_world), scale)
    query = wp.mesh_query_point_no_sign(mesh_id, point_local, 1.0e11)
    if not query.result:
        return wp.vec3(0.0)

    velocity_local = wp.mesh_eval_velocity(mesh_id, query.face, query.u, query.v)
    return wp.transform_vector(X_ws, wp.cw_mul(velocity_local, scale))


@wp.kernel(enable_backward=False)
def eval_rigid_contact_surface_velocities(
    contact_count: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[int],
    contact_surface_velocity: wp.array[wp.vec3],
):
    """Evaluate world-space relative mesh surface velocity for active rigid contacts."""
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    shape0 = contact_shape0[tid]
    shape1 = contact_shape1[tid]
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    X_wb0 = wp.transform_identity() if body0 < 0 else body_q[body0]
    X_wb1 = wp.transform_identity() if body1 < 0 else body_q[body1]
    point0_world = wp.transform_point(X_wb0, contact_point0[tid])
    point1_world = wp.transform_point(X_wb1, contact_point1[tid])
    velocity0_world = _mesh_surface_velocity(
        shape0, point0_world, shape_type, shape_scale, shape_transform, shape_source_ptr, shape_mesh_properties
    )
    velocity1_world = _mesh_surface_velocity(
        shape1, point1_world, shape_type, shape_scale, shape_transform, shape_source_ptr, shape_mesh_properties
    )
    contact_surface_velocity[tid] = velocity1_world - velocity0_world


@wp.kernel(enable_backward=False)
def _record_reduction_overflow(
    insert_failures: wp.array[wp.int32],
    buffer_overflows: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    """Raise the contact stream's overflow flag if the reducer lost contacts; the pass start clears it."""
    if insert_failures[0] > 0 or buffer_overflows[0] > 0:
        overflow[0] = 1


_EMPTY_AABB_EXTENT = wp.constant(1.0e30)


@wp.kernel(enable_backward=False)
def _clear_match_index(count: wp.array[wp.int32], match_index: wp.array[wp.int32]):
    """Mark every live contact unmatched, for a buffer whose producer does not match contacts."""
    i = wp.tid()
    if i < count[0]:
        match_index[i] = -1


@wp.kernel(enable_backward=False)
def _record_reduction_overflow(
    insert_failures: wp.array[int], buffer_overflows: wp.array[int], overflow: wp.array[int]
):
    overflow[0] = int(insert_failures[0] > 0 or buffer_overflows[0] > 0)


@wp.kernel(enable_backward=False)
def compute_shape_aabbs(
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_collision_radius: wp.array[float],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_gap: wp.array[float],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_flags: wp.array[wp.int32],
    shape_variant_row: wp.array[wp.bool],
    # Fused counter arrays — zeroed by thread 0 to avoid separate kernel launches.
    contact_counters: wp.array[wp.int32],
    contact_generation: wp.array[wp.int32],
    broad_phase_pair_count: wp.array[wp.int32],
    num_contact_counters: int,
    body_qd: wp.array[wp.spatial_vector],
    forecast: ShapeForecast,
    # outputs
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
    geom_data: wp.array[wp.vec4],
    geom_xform: wp.array[wp.transform],
):
    """Compute AABBs, narrow-phase geometry data, and zero collision counters.

    Fuses AABB computation, narrow-phase data preparation, contact counter
    zeroing, and generation bumping into a single kernel launch.
    """
    shape_id = wp.tid()
    # Read the forecast's flag first: its load overlaps the shape's, not the stores below.
    forecast_pass = forecasting(forecast)

    # Thread 0: zero contact counters, bump contact generation, and zero the
    # broad phase candidate-pair count in a single fused step.
    if shape_id == 0:
        for c in range(num_contact_counters):
            contact_counters[c] = 0
        g = contact_generation[0]
        if g == 2147483647:
            g = 0
        else:
            g = g + 1
        contact_generation[0] = g
        broad_phase_pair_count[0] = 0

    # Empty variant rows have their collision flags cleared; give them an empty AABB because broad
    # phases include every row that could hold a colliding variant shape.
    if shape_variant_row.shape[0] > 0 and shape_variant_row[shape_id]:
        if (shape_flags[shape_id] & (ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)) == 0:
            aabb_lower[shape_id] = wp.vec3(_EMPTY_AABB_EXTENT)
            aabb_upper[shape_id] = wp.vec3(-_EMPTY_AABB_EXTENT)
            return

    rigid_id = shape_body[shape_id]
    geo_type = shape_type[shape_id]

    # Compute world transform
    if rigid_id == -1:
        X_ws = shape_transform[shape_id]
    else:
        X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])

    pos = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    margin = shape_margin[shape_id]

    # Enlarge AABB by per-shape effective gap for contact detection
    effective_gap = margin + shape_gap[shape_id]
    margin_vec = wp.vec3(effective_gap, effective_gap, effective_gap)

    # Check if this is an infinite plane or a shape with a pre-computed local AABB
    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == GeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)
    has_local_aabb = geo_type == GeoType.MESH or geo_type == GeoType.HFIELD or geo_type == GeoType.CONVEX_MESH

    geom_scale = scale

    if is_infinite_plane:
        # Clamp to the half space the plane bounds, replacing a bounding-sphere
        # fallback whose 1e6 m cube made every shape a permanent ground-plane
        # candidate. A nearly-aligned normal's surface rises by
        # (|n_j| + |n_k|) * d / |n_i| at lateral offset d from the anchor, so
        # bounding d by the reach this AABB itself admits keeps the clamp
        # conservative for every shape it does not already prune laterally; a
        # tilted plane's rise exceeds that reach and the bound stays unbounded.
        normal = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
        # Matches compute_shape_radius's infinite-plane radius.
        HALF_SPACE_EXTENT = 1.0e6
        half_extents = wp.vec3(HALF_SPACE_EXTENT, HALF_SPACE_EXTENT, HALF_SPACE_EXTENT)
        lo = pos - half_extents - margin_vec
        hi = pos + half_extents + margin_vec
        for i in range(3):
            n_i = normal[i]
            # Below this the rise exceeds HALF_SPACE_EXTENT anyway, and the division stays well conditioned.
            if wp.abs(n_i) > 0.5:
                lateral = wp.abs(normal[(i + 1) % 3]) + wp.abs(normal[(i + 2) % 3])
                rise = lateral * HALF_SPACE_EXTENT / wp.abs(n_i)
                if n_i > 0.0:
                    hi[i] = wp.min(hi[i], pos[i] + rise + effective_gap)
                else:
                    lo[i] = wp.max(lo[i], pos[i] - rise - effective_gap)
        aabb_lower[shape_id] = lo
        aabb_upper[shape_id] = hi
    elif geo_type == GeoType.SPHERE:
        radius = scale[0]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif geo_type == GeoType.BOX:
        # The absolute rotation maps local half-extents to exact world AABB extents.
        r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
        r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
        r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
        half_extents = wp.vec3(
            wp.abs(r0[0]) * scale[0] + wp.abs(r1[0]) * scale[1] + wp.abs(r2[0]) * scale[2],
            wp.abs(r0[1]) * scale[0] + wp.abs(r1[1]) * scale[1] + wp.abs(r2[1]) * scale[2],
            wp.abs(r0[2]) * scale[0] + wp.abs(r1[2]) * scale[1] + wp.abs(r2[2]) * scale[2],
        )
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif geo_type == GeoType.CAPSULE:
        radius = scale[0]
        half_height = scale[1]
        axis = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
        half_extents = wp.vec3(radius, radius, radius) + wp.abs(axis) * half_height
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif geo_type == GeoType.CYLINDER:
        radius = scale[0]
        half_height = scale[1]
        barrel_radius = scale[2]
        # Imported MuJoCo site display sizes may use scale[2] without barrel semantics.
        if barrel_radius >= half_height and barrel_radius > 0.0:
            radius += (half_height * half_height) / (
                barrel_radius + wp.sqrt(barrel_radius * barrel_radius - half_height * half_height)
            )
        r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
        r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
        r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
        half_extents = wp.vec3(
            radius * wp.sqrt(r0[0] * r0[0] + r1[0] * r1[0]) + half_height * wp.abs(r2[0]),
            radius * wp.sqrt(r0[1] * r0[1] + r1[1] * r1[1]) + half_height * wp.abs(r2[1]),
            radius * wp.sqrt(r0[2] * r0[2] + r1[2] * r1[2]) + half_height * wp.abs(r2[2]),
        )
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif has_local_aabb:
        # Pre-computed local AABB transformed to world space.
        # Scale is already baked into shape_collision_aabb by the builder,
        # so we only need to handle the rotation here.
        local_lo = shape_collision_aabb_lower[shape_id]
        local_hi = shape_collision_aabb_upper[shape_id]

        center = (local_lo + local_hi) * 0.5
        half = (local_hi - local_lo) * 0.5

        # Rotate center to world frame
        world_center = wp.quat_rotate(orientation, center) + pos

        # Rotated AABB half-extents via abs of rotation matrix columns
        r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
        r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
        r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))

        world_half = wp.vec3(
            wp.abs(r0[0]) * half[0] + wp.abs(r1[0]) * half[1] + wp.abs(r2[0]) * half[2],
            wp.abs(r0[1]) * half[0] + wp.abs(r1[1]) * half[1] + wp.abs(r2[1]) * half[2],
            wp.abs(r0[2]) * half[0] + wp.abs(r1[2]) * half[1] + wp.abs(r2[2]) * half[2],
        )

        aabb_lower[shape_id] = world_center - world_half - margin_vec
        aabb_upper[shape_id] = world_center + world_half + margin_vec
    else:
        # Use support function to compute tight AABB
        # Create generic shape data
        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        if geo_type == GeoType.PLANE:
            geom_scale = wp.vec3(scale[0] * 0.5, scale[1] * 0.5, 0.0)
        shape_data.scale = geom_scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)
        shape_data.center = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == GeoType.CONVEX_MESH:
            shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[shape_id])

        data_provider = SupportMapDataProvider()

        # Compute tight AABB using helper function
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, orientation, pos, data_provider)

        aabb_lower[shape_id] = aabb_min_world - margin_vec
        aabb_upper[shape_id] = aabb_max_world + margin_vec

    # Narrow-phase geometry data (reuses X_ws and scale already computed above)
    geom_data[shape_id] = wp.vec4(geom_scale[0], geom_scale[1], geom_scale[2], margin)
    geom_xform[shape_id] = X_ws

    if forecast_pass:
        forecast_shape(forecast, shape_id, rigid_id, X_ws, body_q, body_qd, aabb_lower[shape_id], aabb_upper[shape_id])


@wp.kernel(enable_backward=False)
def _compute_pair_shape_aabbs(
    shape_indices: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_collision_radius: wp.array[float],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    shape_gap: wp.array[float],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_flags: wp.array[wp.int32],
    shape_variant_row: wp.array[wp.bool],
    # Fused counter arrays — zeroed by thread 0 to avoid separate kernel launches.
    contact_counters: wp.array[wp.int32],
    contact_generation: wp.array[wp.int32],
    broad_phase_pair_count: wp.array[wp.int32],
    num_contact_counters: int,
    body_qd: wp.array[wp.spatial_vector],
    forecast: ShapeForecast,
    # outputs
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
    geom_data: wp.array[wp.vec4],
    geom_xform: wp.array[wp.transform],
):
    """Prepare the fixed explicit-pair participants without renumbering shapes.

    The per-shape arithmetic mirrors ``compute_shape_aabbs`` verbatim. Keep
    the full kernel unchanged until exact device replay gates this opt-in path.
    """
    work_id = wp.tid()
    # Read the forecast's flag first: its load overlaps the shape's, not the stores below.
    forecast_pass = forecasting(forecast)

    # Thread 0: zero contact counters, bump contact generation, and zero the
    # broad phase candidate-pair count in a single fused step.
    if work_id == 0:
        for c in range(num_contact_counters):
            contact_counters[c] = 0
        g = contact_generation[0]
        if g == 2147483647:
            g = 0
        else:
            g = g + 1
        contact_generation[0] = g
        broad_phase_pair_count[0] = 0

    # One thread still performs counter maintenance for an empty pair set.
    if work_id >= shape_indices.shape[0]:
        return
    shape_id = shape_indices[work_id]

    # Empty variant rows have their collision flags cleared; give them an empty AABB because broad
    # phases include every row that could hold a colliding variant shape.
    if shape_variant_row.shape[0] > 0 and shape_variant_row[shape_id]:
        if (shape_flags[shape_id] & (ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)) == 0:
            aabb_lower[shape_id] = wp.vec3(_EMPTY_AABB_EXTENT)
            aabb_upper[shape_id] = wp.vec3(-_EMPTY_AABB_EXTENT)
            return

    rigid_id = shape_body[shape_id]
    geo_type = shape_type[shape_id]

    # Compute world transform
    if rigid_id == -1:
        X_ws = shape_transform[shape_id]
    else:
        X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])

    pos = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    margin = shape_margin[shape_id]

    # Enlarge AABB by per-shape effective gap for contact detection
    effective_gap = margin + shape_gap[shape_id]
    margin_vec = wp.vec3(effective_gap, effective_gap, effective_gap)

    # Check if this is an infinite plane or a shape with a pre-computed local AABB
    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == GeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)
    has_local_aabb = geo_type == GeoType.MESH or geo_type == GeoType.HFIELD or geo_type == GeoType.CONVEX_MESH

    geom_scale = scale

    if is_infinite_plane:
        # Clamp to the half space the plane bounds, replacing a bounding-sphere
        # fallback whose 1e6 m cube made every shape a permanent ground-plane
        # candidate. A nearly-aligned normal's surface rises by
        # (|n_j| + |n_k|) * d / |n_i| at lateral offset d from the anchor, so
        # bounding d by the reach this AABB itself admits keeps the clamp
        # conservative for every shape it does not already prune laterally; a
        # tilted plane's rise exceeds that reach and the bound stays unbounded.
        normal = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
        # Matches compute_shape_radius's infinite-plane radius.
        HALF_SPACE_EXTENT = 1.0e6
        half_extents = wp.vec3(HALF_SPACE_EXTENT, HALF_SPACE_EXTENT, HALF_SPACE_EXTENT)
        lo = pos - half_extents - margin_vec
        hi = pos + half_extents + margin_vec
        for i in range(3):
            n_i = normal[i]
            # Below this the rise exceeds HALF_SPACE_EXTENT anyway, and the division stays well conditioned.
            if wp.abs(n_i) > 0.5:
                lateral = wp.abs(normal[(i + 1) % 3]) + wp.abs(normal[(i + 2) % 3])
                rise = lateral * HALF_SPACE_EXTENT / wp.abs(n_i)
                if n_i > 0.0:
                    hi[i] = wp.min(hi[i], pos[i] + rise + effective_gap)
                else:
                    lo[i] = wp.max(lo[i], pos[i] - rise - effective_gap)
        aabb_lower[shape_id] = lo
        aabb_upper[shape_id] = hi
    elif geo_type == GeoType.SPHERE:
        radius = scale[0]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif geo_type == GeoType.BOX:
        # The absolute rotation maps local half-extents to exact world AABB extents.
        r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
        r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
        r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
        half_extents = wp.vec3(
            wp.abs(r0[0]) * scale[0] + wp.abs(r1[0]) * scale[1] + wp.abs(r2[0]) * scale[2],
            wp.abs(r0[1]) * scale[0] + wp.abs(r1[1]) * scale[1] + wp.abs(r2[1]) * scale[2],
            wp.abs(r0[2]) * scale[0] + wp.abs(r1[2]) * scale[1] + wp.abs(r2[2]) * scale[2],
        )
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif geo_type == GeoType.CAPSULE:
        radius = scale[0]
        half_height = scale[1]
        axis = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
        half_extents = wp.vec3(radius, radius, radius) + wp.abs(axis) * half_height
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif geo_type == GeoType.CYLINDER:
        radius = scale[0]
        half_height = scale[1]
        barrel_radius = scale[2]
        # Imported MuJoCo site display sizes may use scale[2] without barrel semantics.
        if barrel_radius >= half_height and barrel_radius > 0.0:
            radius += (half_height * half_height) / (
                barrel_radius + wp.sqrt(barrel_radius * barrel_radius - half_height * half_height)
            )
        r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
        r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
        r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
        half_extents = wp.vec3(
            radius * wp.sqrt(r0[0] * r0[0] + r1[0] * r1[0]) + half_height * wp.abs(r2[0]),
            radius * wp.sqrt(r0[1] * r0[1] + r1[1] * r1[1]) + half_height * wp.abs(r2[1]),
            radius * wp.sqrt(r0[2] * r0[2] + r1[2] * r1[2]) + half_height * wp.abs(r2[2]),
        )
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    elif has_local_aabb:
        # Pre-computed local AABB transformed to world space.
        # Scale is already baked into shape_collision_aabb by the builder,
        # so we only need to handle the rotation here.
        local_lo = shape_collision_aabb_lower[shape_id]
        local_hi = shape_collision_aabb_upper[shape_id]

        center = (local_lo + local_hi) * 0.5
        half = (local_hi - local_lo) * 0.5

        # Rotate center to world frame
        world_center = wp.quat_rotate(orientation, center) + pos

        # Rotated AABB half-extents via abs of rotation matrix columns
        r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
        r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
        r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))

        world_half = wp.vec3(
            wp.abs(r0[0]) * half[0] + wp.abs(r1[0]) * half[1] + wp.abs(r2[0]) * half[2],
            wp.abs(r0[1]) * half[0] + wp.abs(r1[1]) * half[1] + wp.abs(r2[1]) * half[2],
            wp.abs(r0[2]) * half[0] + wp.abs(r1[2]) * half[1] + wp.abs(r2[2]) * half[2],
        )

        aabb_lower[shape_id] = world_center - world_half - margin_vec
        aabb_upper[shape_id] = world_center + world_half + margin_vec
    else:
        # Use support function to compute tight AABB
        # Create generic shape data
        shape_data = GenericShapeData()
        shape_data.shape_type = geo_type
        if geo_type == GeoType.PLANE:
            geom_scale = wp.vec3(scale[0] * 0.5, scale[1] * 0.5, 0.0)
        shape_data.scale = geom_scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)
        shape_data.center = wp.vec3(0.0, 0.0, 0.0)

        # For CONVEX_MESH, pack the mesh pointer
        if geo_type == GeoType.CONVEX_MESH:
            shape_data.auxiliary = pack_mesh_ptr(shape_source_ptr[shape_id])

        data_provider = SupportMapDataProvider()

        # Compute tight AABB using helper function
        aabb_min_world, aabb_max_world = compute_tight_aabb_from_support(shape_data, orientation, pos, data_provider)

        aabb_lower[shape_id] = aabb_min_world - margin_vec
        aabb_upper[shape_id] = aabb_max_world + margin_vec

    # Narrow-phase geometry data (reuses X_ws and scale already computed above)
    geom_data[shape_id] = wp.vec4(geom_scale[0], geom_scale[1], geom_scale[2], margin)
    geom_xform[shape_id] = X_ws

    if forecast_pass:
        forecast_shape(forecast, shape_id, rigid_id, X_ws, body_q, body_qd, aabb_lower[shape_id], aabb_upper[shape_id])


# Per-pair worst-case contact counts established by the narrow phase:
#
# * Convex-convex pairs (all primitive GeoTypes, CONVEX_MESH, and planes) are
#   handled either by the analytic primitive fast path, which writes at most 4
#   contacts per pair (``narrow_phase.py::narrow_phase_primitive_kernel``,
#   contact slots 0-3), or by the GJK/MPR multi-contact path, which writes at
#   most 4 manifold points plus 1 deepest contact = 5 per pair
#   (``multicontact.py::build_manifold``: ``count_out = min(num_manifold_points, 4)``
#   followed by an optional deepest-contact write).
_CONVEX_PAIR_MAX_CONTACTS = 5
#
# * Mesh/heightfield-involved pairs (mesh-convex, mesh-plane, mesh-mesh,
#   heightfield-*) and hydroelastic SDF-SDF pairs route their per-triangle /
#   per-vertex contacts through contact reduction, which retains at most
#   ``NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS + 1) + NUM_VOXEL_DEPTH_SLOTS``
#   slots per pair (240 with the default icosahedron configuration).
#   ``contact_reduction.py`` asserts at import time that the slot count never
#   exceeds ``MAX_CONTACTS_PER_PAIR`` (255, the hard architectural limit that
#   keeps per-pair contact indices representable in 8 bits), so
#   MAX_CONTACTS_PER_PAIR bounds every reduced mesh path. The bound assumes
#   contact reduction is enabled (the default); with ``reduce_contacts=False``
#   mesh paths write unreduced per-triangle/per-vertex contacts and no static
#   estimate is possible.
_MESH_PAIR_MAX_CONTACTS = MAX_CONTACTS_PER_PAIR
#
# * Hydroelastic SDF-SDF pairs additionally export one synthetic anchor
#   contact PER ACTIVE NORMAL BIN when ``anchor_contact`` (or
#   ``moment_matching``, which implies it) is enabled
#   (``contact_reduction_hydroelastic.py::export_hydroelastic_reduced_contacts_kernel``),
#   on top of the reduced slots — worst case ``MAX_CONTACTS_PER_PAIR +
#   NUM_NORMAL_BINS`` per pair.
_HYDRO_PAIR_MAX_CONTACTS = MAX_CONTACTS_PER_PAIR + NUM_NORMAL_BINS


FORECAST_TILE_CELLS = 8
"""Heightfield cells a side of a forecast tile holds: small enough that a shape's box meets at most four."""


@wp.kernel(enable_backward=False)
def compute_shape_velocities(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_collision_radius: wp.array[float],
    shape_gap: wp.array[float],
    collision_update_dt: float,
    max_speculative_extension: float,
    # outputs
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_search_gap: wp.array[float],
    shape_displacement: wp.array[wp.vec3],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
):
    """Compute shape motion and expand its AABB over the prediction horizon.

    ``shape_displacement`` is the world-space shape-origin velocity, including
    the ``angular_velocity x COM_offset`` contribution, multiplied by
    ``collision_update_dt``. Angular travel expands the AABB separately.
    ``angular_speed_bound`` is the resulting conservative linear speed [m/s]
    at the shape bound, not an angular speed [rad/s].
    """
    shape_id = wp.tid()
    body_id = shape_body[shape_id]
    if body_id == -1:
        shape_linear_velocity[shape_id] = wp.vec3(0.0)
        shape_angular_velocity[shape_id] = wp.vec3(0.0)
        shape_search_gap[shape_id] = shape_gap[shape_id]
        shape_displacement[shape_id] = wp.vec3(0.0)
        return

    X_wb = body_q[body_id]
    X_ws = wp.transform_multiply(X_wb, shape_transform[shape_id])
    shape_origin_world = wp.transform_get_translation(X_ws)
    com_world = wp.transform_point(X_wb, body_com[body_id])
    twist = body_qd[body_id]
    com_velocity = wp.spatial_top(twist)
    angular_velocity = wp.spatial_bottom(twist)
    shape_origin_velocity = com_velocity + wp.cross(angular_velocity, shape_origin_world - com_world)
    shape_linear_velocity[shape_id] = shape_origin_velocity
    shape_angular_velocity[shape_id] = angular_velocity

    local_lower = shape_collision_aabb_lower[shape_id]
    local_upper = shape_collision_aabb_upper[shape_id]
    furthest = wp.max(wp.abs(local_lower), wp.abs(local_upper))
    angular_radius = wp.max(wp.length(furthest), shape_collision_radius[shape_id])
    angular_speed_bound = wp.length(angular_velocity) * angular_radius
    search_extension = wp.min(
        (wp.length(shape_origin_velocity) + angular_speed_bound) * collision_update_dt,
        max_speculative_extension,
    )
    shape_search_gap[shape_id] = shape_gap[shape_id] + search_extension

    displacement = shape_origin_velocity * collision_update_dt
    angular_extension = angular_speed_bound * collision_update_dt
    cap = wp.vec3(max_speculative_extension)
    # Preserve absolute motion so pairwise subtraction retains relative velocity.
    shape_displacement[shape_id] = displacement
    angular_extension_vec = wp.min(wp.vec3(angular_extension), cap)
    shape_aabb_lower[shape_id] = shape_aabb_lower[shape_id] - angular_extension_vec
    shape_aabb_upper[shape_id] = shape_aabb_upper[shape_id] + angular_extension_vec


# Primitive pairs (GJK/MPR) produce up to 5 manifold contacts.
# Mesh-involved pairs (SDF + contact reduction) typically retain about 40.
_RIGID_CONTACTS_PER_PRIMITIVE_PAIR = 5
_RIGID_CONTACTS_PER_MESH_PAIR = 40
_RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE = 20
_RIGID_CONTACT_MIN_CAPACITY = 1000
# Base Contacts storage: four int32, six vec3, and two float32 values per slot.
_RIGID_CONTACT_BASE_BYTES_PER_SLOT = 96
_RIGID_CONTACT_LARGE_BUFFER_BYTES = 256 * 1024 * 1024


def _shape_capsules(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return each shape's bounding capsule in its own frame: (half axis, radius) and (center, farthest point).

    A box, a mesh by its local box, or a cylinder takes the capsule along its longest side; a capsule and a sphere
    are their own; a shape of no simpler form takes its bounding sphere; a plane or a heightfield has none (radius
    -1), so only its bounding box tells a gap. The farthest point of the shape from its origin is its spin's lever.
    """
    types = model.shape_type.numpy()
    scale = model.shape_scale.numpy()
    center = np.zeros((len(types), 3), dtype=np.float32)
    axis = np.zeros((len(types), 3), dtype=np.float32)
    radius = model.shape_collision_radius.numpy().copy()
    lower, upper = model.shape_collision_aabb_lower.numpy(), model.shape_collision_aabb_upper.numpy()
    farthest = np.maximum(np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper)), axis=1), radius)

    capsule = types == int(GeoType.CAPSULE)
    axis[capsule, 2] = scale[capsule, 1]
    radius[capsule] = scale[capsule, 0]
    sphere = types == int(GeoType.SPHERE)
    radius[sphere] = scale[sphere, 0]

    meshes = (types == int(GeoType.MESH)) | (types == int(GeoType.CONVEX_MESH))
    boxes = (types == int(GeoType.BOX)) | meshes
    half = np.zeros((len(types), 3), dtype=np.float32)
    half[types == int(GeoType.BOX)] = scale[types == int(GeoType.BOX)]
    center[meshes] = (0.5 * (lower + upper))[meshes]
    half[meshes] = (0.5 * (upper - lower))[meshes]
    cylinder = (types == int(GeoType.CYLINDER)) & ~(scale[:, 2] >= scale[:, 1])
    half[cylinder] = scale[cylinder][:, [0, 0, 1]]
    sided = boxes | cylinder
    shapes = np.flatnonzero(sided)
    longest = half[shapes].argmax(axis=1)
    axis[shapes, longest] = half[shapes, longest]
    sides = half[shapes].copy()
    sides[np.arange(len(shapes)), longest] = 0.0
    radius[shapes] = np.sqrt((sides**2).sum(axis=1))
    radius[(types == int(GeoType.PLANE)) | (types == int(GeoType.HFIELD))] = -1.0
    capsules = np.concatenate([axis, radius[:, None]], axis=1).astype(np.float32)
    middles = np.concatenate([center, farthest[:, None]], axis=1).astype(np.float32)
    return capsules, middles


@dataclass
class _ForecastTerrain:
    """What the forecast reads of a model's heightfields (:func:`_forecast_terrain`)."""

    shapes: np.ndarray
    """Each shape's terrain, ``-1`` for a shape that is no upright heightfield."""
    walk_slots: np.ndarray
    """The most rows the terrain law's walk holds for a pair, by its partner's :class:`GeoType`."""
    origins: np.ndarray
    """Each terrain's corner in the world and its height origin [m]."""
    scales: np.ndarray
    """Each terrain's tiles per meter along x and y."""
    layouts: np.ndarray
    """Each terrain's first tile, tile rows and tile columns."""
    tiles: np.ndarray
    """The highest height of each tile [m]."""


def _has_static_meshes(model: Model) -> bool:
    """Whether the model has a triangle mesh on no body: a surface the terrain law may read (its cell table)."""
    if not model.shape_count:
        return False
    return bool(np.any((model.shape_type.numpy() == int(GeoType.MESH)) & (model.shape_body.numpy() < 0)))


def _forecast_terrain(model: Model) -> _ForecastTerrain:
    """Return the tiles of the model's upright heightfields and the shapes they belong to."""
    types = model.shape_type.numpy()
    fields = np.full(len(types), -1, dtype=np.int32)
    origins, scales, layouts, tiles, first = [], [], [], [], 0
    transforms = model.shape_transform.numpy()
    bodies = model.shape_body.numpy()
    for shape in np.flatnonzero(types == int(GeoType.HFIELD)):
        field = model.shape_source[shape]
        upright = np.allclose(np.abs(transforms[shape, 3:7]), (0.0, 0.0, 0.0, 1.0))
        if field is None or bodies[shape] >= 0 or not upright:
            continue
        heights = field.min_z + np.asarray(field.data).reshape(field.nrow, field.ncol) * (field.max_z - field.min_z)
        rows = max((field.nrow - 1 + FORECAST_TILE_CELLS - 1) // FORECAST_TILE_CELLS, 1)
        columns = max((field.ncol - 1 + FORECAST_TILE_CELLS - 1) // FORECAST_TILE_CELLS, 1)
        top = np.zeros((rows, columns), dtype=np.float32)
        for row in range(rows):
            for column in range(columns):
                cells = heights[
                    row * FORECAST_TILE_CELLS : (row + 1) * FORECAST_TILE_CELLS + 1,
                    column * FORECAST_TILE_CELLS : (column + 1) * FORECAST_TILE_CELLS + 1,
                ]
                top[row, column] = cells.max()
        fields[shape] = len(origins)
        position = transforms[shape, :3]
        origins.append((position[0] - field.hx, position[1] - field.hy, position[2]))
        scales.append((columns / (2.0 * field.hx), rows / (2.0 * field.hy)))
        layouts.append((first, rows, columns))
        tiles.append(top.reshape(-1))
        first += top.size
    slots = np.full(len(GeoType), TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_HULL], dtype=np.int64)
    slots[int(GeoType.SPHERE)] = slots[int(GeoType.CAPSULE)] = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_POINT]
    slots[int(GeoType.BOX)] = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_BOX]
    slots[int(GeoType.CYLINDER)] = TERRAIN_PARTNER_WALK_SLOTS[TERRAIN_PARTNER_CYLINDER]
    return _ForecastTerrain(
        fields,
        slots.astype(np.int32),
        np.asarray(origins, dtype=np.float32).reshape(-1, 3) if origins else np.zeros((1, 3), dtype=np.float32),
        np.asarray(scales, dtype=np.float32).reshape(-1, 2) if scales else np.zeros((1, 2), dtype=np.float32),
        np.asarray(layouts, dtype=np.int32).reshape(-1, 3) if layouts else np.zeros((1, 3), dtype=np.int32),
        np.concatenate(tiles) if tiles else np.zeros(1, dtype=np.float32),
    )


def _shape_forecast_words(model: Model, terrain: _ForecastTerrain) -> np.ndarray:
    """Return each shape's forecast word: its forecast bits, and the terrain law's walk rows for a heightfield pair
    of it (:class:`~newton._src.geometry.broad_phase_common.ShapeForecast`)."""
    types = model.shape_type.numpy()
    words = np.zeros(len(types), dtype=np.int32)
    words[(types == int(GeoType.MESH)) | (types == int(GeoType.HFIELD))] |= int(FORECAST_REDUCED)
    shape_flags = getattr(model, "shape_flags", None)
    if shape_flags is not None:
        words[(shape_flags.numpy() & int(ShapeFlags.HYDROELASTIC)) != 0] |= int(FORECAST_HYDROELASTIC)
    floating = np.ones(max(model.body_count, 1), dtype=bool)
    if model.joint_count:
        joint_articulation = model.joint_articulation.numpy()
        roots = model.joint_type.numpy()[model.articulation_start.numpy()[:-1]]
        anchored = np.flatnonzero((roots != int(JointType.FREE)) & (roots != int(JointType.DISTANCE)))
        held = np.isin(joint_articulation, anchored)
        floating[model.joint_child.numpy()[held]] = False
    bodies = model.shape_body.numpy()
    words[floating[np.maximum(bodies, 0)] & (bodies >= 0)] |= int(FORECAST_FLOATS)
    words[terrain.shapes >= 0] |= int(FORECAST_TERRAIN)
    words |= terrain.walk_slots[types].astype(np.int32) << int(FORECAST_SLOTS_SHIFT)
    return words


def _shape_forecast(model: Model, pool, poses: wp.array, *, terrain_law: bool, estimate: int | None) -> ShapeForecast:
    """Return the forecast of a pipeline's passes over ``model``, adding into ``pool``'s forecast.

    ``poses`` are the shapes' world transforms, which the bounding-box pass writes. Without the terrain law a
    heightfield pair's contacts come from contact reduction, which bounds them instead of the law's walk rows.
    """
    device = pool.device
    terrain = _forecast_terrain(model)
    if not terrain_law:
        terrain.walk_slots[:] = _MESH_PAIR_MAX_CONTACTS
    capsules, middles = _shape_capsules(model)
    forecast = ShapeForecast()
    forecast.on = pool.forecast_on
    forecast.slots = pool.forecast_slots
    forecast.horizon = pool.forecast_reach
    forecast.word = wp.array(_shape_forecast_words(model, terrain), dtype=wp.int32, device=device)
    forecast.capsule = wp.array(capsules, dtype=wp.vec4, device=device)
    forecast.middle = wp.array(middles, dtype=wp.vec4, device=device)
    forecast.pose = poses
    forecast.body_com = model.body_com if model.body_count else wp.zeros(1, dtype=wp.vec3, device=device)
    forecast.reach = wp.zeros(model.shape_count, dtype=wp.float32, device=device)
    forecast.terrain_count = int(terrain.shapes.max(initial=-1)) + 1
    if forecast.terrain_count:
        forecast.top = wp.zeros(model.shape_count, dtype=wp.float32, device=device)
    forecast.terrain_origin = wp.array(terrain.origins, dtype=wp.vec3, device=device)
    forecast.terrain_scale = wp.array(terrain.scales, dtype=wp.vec2, device=device)
    forecast.terrain_layout = wp.array(terrain.layouts, dtype=wp.vec3i, device=device)
    forecast.terrain_tiles = wp.array(terrain.tiles, dtype=wp.float32, device=device)
    forecast.bound_convex = _CONVEX_PAIR_MAX_CONTACTS
    forecast.bound_reduced = _MESH_PAIR_MAX_CONTACTS
    forecast.bound_hydroelastic = _HYDRO_PAIR_MAX_CONTACTS
    pool.forecast_limit = _estimate_rigid_contact_max(model) if estimate is None else estimate
    return forecast


@dataclasses.dataclass(frozen=True)
class _RigidContactCountEstimate:
    """Diagnostic details for an automatically selected rigid-contact capacity."""

    capacity: int
    world_count: int
    primitive_count: int
    mesh_count: int
    plane_count: int
    pair_count: int
    pair_estimate_contacts_per_pair: int
    source: Literal["fallback", "minimum", "neighbor heuristic", "precomputed pairs"]


def _estimate_rigid_contact_details(model: Model) -> _RigidContactCountEstimate:
    """Estimate rigid-contact capacity and retain its diagnostic inputs.

    When precomputed contact pairs are available (``model.shape_contact_pairs``,
    produced by ``ModelBuilder.find_shape_contact_pairs`` for every finalized
    model), the estimate is the exact sum of per-pair worst-case contact counts
    over the actual pair list: each pair is classified by the GeoTypes (and
    hydroelastic flags) of its two shapes and assigned the narrow phase's hard
    per-pair cap. Every broad phase mode (explicit, nxn, sap) applies the same
    world/group/filter-pair logic as ``find_shape_contact_pairs``, so any
    candidate pair reaching the narrow phase is contained in the precomputed
    list and the sum is a provable worst case. For dense pair graphs the sum is
    capped by a spatial-locality estimate (see the inline comment); where that
    cap binds, capacity is physically motivated rather than provable, matching
    the pre-pair-aware behavior. Crucially, shapes that participate in no
    contact pair (e.g. visual-only meshes) contribute nothing to either term.
    The per-pair caps assume contact reduction is enabled (the default for
    mesh/hydroelastic paths); with ``reduce_contacts=False`` no static bound
    exists and overflow remains detectable via ``contact_count > contact_max``.

    Otherwise falls back to a linear neighbor-budget estimate assuming each
    non-plane shape contacts at most ``_RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE`` others (spatial
    locality).  The non-plane term is additive across independent worlds so a
    single-pool computation is correct.  The plane term (each plane vs all
    non-planes in its world) would be quadratic if computed globally, so it is
    evaluated per world when metadata is available.

    Args:
        model: The simulation model.

    Returns:
        The estimated capacity and its diagnostic inputs.
    """
    world_count = int(getattr(model, "world_count", 0) or 0)
    pair_count = int(getattr(model, "shape_contact_pair_count", 0) or 0)
    if not hasattr(model, "shape_type") or model.shape_type is None:
        return _RigidContactCountEstimate(
            capacity=_RIGID_CONTACT_MIN_CAPACITY,
            world_count=world_count,
            primitive_count=0,
            mesh_count=0,
            plane_count=0,
            pair_count=pair_count,
            pair_estimate_contacts_per_pair=0,
            source="fallback",
        )

    shape_types = model.shape_type.numpy()
    # Slot rows count as meshes if they can hold one.
    variant_options = _variant_options(model)
    colliding_mask = _shape_collide_mask(model, len(shape_types))

    mesh_mask = colliding_mask & _shape_may_be_mesh(model, shape_types, variant_options)

    # ------------------------------------------------------------------
    # Pair-aware exact bound: sum of per-pair worst cases over the actual
    # precomputed contact pairs.
    # ------------------------------------------------------------------
    if pair_count > 0:
        # Each pair's cap follows from its two shapes' kinds, so sums over the pairs are category-weighted pair
        # counts: no pair list is enumerated or built.
        plane_shape = shape_types == int(GeoType.PLANE)
        shape_flags = getattr(model, "shape_flags", None)
        if shape_flags is not None:
            hydro_mask = (shape_flags.numpy() & int(ShapeFlags.HYDROELASTIC)) != 0
        else:
            hydro_mask = np.zeros(len(shape_types), dtype=bool)
        # Category bits: mesh 1, hydroelastic 2, plane 4.
        categories = (
            mesh_mask.astype(np.int32) | (hydro_mask.astype(np.int32) << 1) | (plane_shape.astype(np.int32) << 2)
        )
        kinds = np.arange(8)
        mesh_kind, hydro_kind, plane_kind = (kinds & 1) != 0, (kinds & 2) != 0, (kinds & 4) != 0
        # Pairs where both shapes are hydroelastic route to the SDF-SDF hydroelastic path, whose anchor export can
        # exceed the reduction slots by one contact per normal bin; other mesh/heightfield pairs are reduced.
        caps = np.full((8, 8), _CONVEX_PAIR_MAX_CONTACTS, dtype=np.int64)
        caps[mesh_kind[:, None] | mesh_kind[None, :]] = _MESH_PAIR_MAX_CONTACTS
        caps[hydro_kind[:, None] & hydro_kind[None, :]] = _HYDRO_PAIR_MAX_CONTACTS
        pair_contacts = _shape_contact_pair_weight_sum(model, categories, caps)
    if pair_count > 0 and pair_contacts is not None:
        # The pair sum is the combinatorial worst case, but for dense pair
        # graphs (e.g. thousands of mutually collidable shapes piled in one
        # world) it explodes quadratically while only a bounded number of
        # neighbors can touch a shape simultaneously. Cap it with a
        # spatial-locality estimate, mirroring the pre-pair-aware behavior:
        # plane-involved pairs keep their full per-pair budget (a ground plane
        # really can contact every shape resting on it at once), and each
        # colliding non-plane shape is budgeted for at most MAX_NEIGHBORS_PER_SHAPE
        # simultaneous neighbors at its own per-pair cap (halved to avoid
        # double-counting both shapes of a pair). Where this cap binds the
        # capacity is physically motivated rather than a provable bound -
        # exactly as before this estimator existed; where the pair sum binds
        # (typical multi-world robot scenes) it is a provable worst case.
        plane_caps = np.where(plane_kind[:, None] | plane_kind[None, :], caps, 0)
        plane_pair_contacts = _shape_contact_pair_weight_sum(model, categories, plane_caps)
        active = colliding_mask & ~plane_shape
        shape_cap = np.full(len(shape_types), _CONVEX_PAIR_MAX_CONTACTS, dtype=np.int64)
        shape_cap[mesh_mask] = _MESH_PAIR_MAX_CONTACTS
        shape_cap[hydro_mask] = _HYDRO_PAIR_MAX_CONTACTS
        nonplane_locality = int(shape_cap[active].sum()) * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE // 2
        locality_cap = plane_pair_contacts + nonplane_locality

        plane_mask = colliding_mask & plane_shape
        num_meshes = int(np.count_nonzero(mesh_mask))
        num_non_planes = int(np.count_nonzero(colliding_mask & ~plane_shape))
        return _RigidContactCountEstimate(
            capacity=max(_RIGID_CONTACT_MIN_CAPACITY, min(pair_contacts, locality_cap)),
            world_count=world_count,
            primitive_count=num_non_planes - num_meshes,
            mesh_count=num_meshes,
            plane_count=int(np.count_nonzero(plane_mask)),
            pair_count=pair_count,
            pair_estimate_contacts_per_pair=pair_contacts // max(pair_count, 1),
            source="precomputed pairs",
        )

    # ------------------------------------------------------------------
    # Fallback: neighbor-budget heuristic (no precomputed pairs available).
    # ------------------------------------------------------------------
    plane_mask = colliding_mask & (shape_types == int(GeoType.PLANE))
    non_plane_mask = colliding_mask & ~plane_mask
    num_meshes = int(np.count_nonzero(mesh_mask))
    num_non_planes = int(np.count_nonzero(non_plane_mask))
    num_primitives = num_non_planes - num_meshes
    num_planes = int(np.count_nonzero(plane_mask))

    # Weighted contacts from non-plane shape types.
    # Each shape's neighbor pairs are weighted by its type's contacts-per-pair.
    # Divide by 2 to avoid double-counting pairs.
    non_plane_contacts = (
        num_primitives * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR
        + num_meshes * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE * _RIGID_CONTACTS_PER_MESH_PAIR
    ) // 2

    # Weighted average contacts-per-pair based on the scene's shape mix.
    avg_cpp = (
        (num_primitives * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR + num_meshes * _RIGID_CONTACTS_PER_MESH_PAIR)
        // max(num_non_planes, 1)
        if num_non_planes > 0
        else 0
    )

    # Plane contacts: each plane contacts all non-plane shapes *in its world*.
    # The naive global formula (num_planes * num_non_planes) is O(worlds²) when
    # both counts grow with the number of worlds.  Use per-world counts instead.
    plane_contacts = 0
    if num_planes > 0 and num_non_planes > 0:
        has_world_info = (
            hasattr(model, "shape_world")
            and model.shape_world is not None
            and hasattr(model, "world_count")
            and model.world_count > 0
        )
        shape_world = model.shape_world.numpy() if has_world_info else None

        if shape_world is not None and len(shape_world) == len(shape_types):
            global_mask = shape_world == -1
            local_mask = ~global_mask
            n_worlds = world_count

            global_planes = int(np.count_nonzero(global_mask & plane_mask))
            global_non_planes = int(np.count_nonzero(global_mask & non_plane_mask))

            local_plane_counts = np.bincount(shape_world[local_mask & plane_mask], minlength=n_worlds)[:n_worlds]
            local_non_plane_counts = np.bincount(shape_world[local_mask & non_plane_mask], minlength=n_worlds)[
                :n_worlds
            ]

            per_world_planes = local_plane_counts + global_planes
            per_world_non_planes = local_non_plane_counts + global_non_planes

            # Global-global pairs appear in every world slice; keep one copy.
            plane_pair_count = int(np.sum(per_world_planes * per_world_non_planes))
            if n_worlds > 1:
                plane_pair_count -= (n_worlds - 1) * global_planes * global_non_planes
            plane_contacts = plane_pair_count * avg_cpp
        else:
            # Fallback: exact type-weighted sum (correct for single-world models).
            plane_contacts = num_planes * (
                num_primitives * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR + num_meshes * _RIGID_CONTACTS_PER_MESH_PAIR
            )

    heuristic_contacts = non_plane_contacts + plane_contacts
    total_contacts = heuristic_contacts
    source = "neighbor heuristic"

    # Legacy fallback: a pair count is known but the pair list itself is not
    # available (only possible for hand-assembled models; ModelBuilder.finalize
    # always populates shape_contact_pairs). Use the count as a tighter bound.
    if pair_count > 0:
        weighted_cpp = max(avg_cpp, _RIGID_CONTACTS_PER_PRIMITIVE_PAIR)
        pair_contacts = pair_count * weighted_cpp
        total_contacts = min(total_contacts, pair_contacts)
        if pair_contacts < heuristic_contacts:
            source = "precomputed pairs"

    # Ensure minimum allocation
    capacity = max(_RIGID_CONTACT_MIN_CAPACITY, total_contacts)
    if capacity == _RIGID_CONTACT_MIN_CAPACITY and total_contacts < capacity:
        source = "minimum"
    return _RigidContactCountEstimate(
        capacity=capacity,
        world_count=world_count,
        primitive_count=num_primitives,
        mesh_count=num_meshes,
        plane_count=num_planes,
        pair_count=pair_count,
        pair_estimate_contacts_per_pair=max(avg_cpp, _RIGID_CONTACTS_PER_PRIMITIVE_PAIR),
        source=source,
    )


def _estimate_rigid_contact_max(model: Model) -> int:
    """Estimate the rigid-contact capacity for the collision pipeline."""
    return _estimate_rigid_contact_details(model).capacity


def _warn_large_rigid_contact_estimate(estimate: _RigidContactCountEstimate) -> None:
    """Warn when an automatic estimate implies a large base contact allocation."""
    allocation_bytes = estimate.capacity * _RIGID_CONTACT_BASE_BYTES_PER_SLOT
    if allocation_bytes < _RIGID_CONTACT_LARGE_BUFFER_BYTES:
        return

    allocation_mib = allocation_bytes / (1024 * 1024)
    warnings.warn(
        f"CollisionPipeline automatically selected {estimate.capacity:,} rigid contact slots, "
        f"requiring approximately {allocation_mib:,.1f} MiB for the base rigid-contact buffers "
        f"({_RIGID_CONTACT_BASE_BYTES_PER_SLOT} bytes per slot). Estimate inputs -- "
        f"world count: {estimate.world_count:,}; colliding shapes: "
        f"{estimate.primitive_count:,} primitives, {estimate.mesh_count:,} meshes/heightfields, "
        f"and {estimate.plane_count:,} planes; precomputed contact pairs: {estimate.pair_count:,}; "
        f"contact weighting: {_RIGID_CONTACTS_PER_PRIMITIVE_PAIR} per primitive pair and "
        f"{_RIGID_CONTACTS_PER_MESH_PAIR} per mesh-involved pair; pair-estimate average: "
        f"{estimate.pair_estimate_contacts_per_pair}; selected estimate: {estimate.source}. "
        "Optional pipeline and solver features may allocate additional memory. Pass rigid_contact_max "
        "explicitly to CollisionPipeline to choose the capacity and silence this warning.",
        NewtonWarning,
        stacklevel=3,
    )


def _estimate_rigid_contact_max_per_world(model: Model | ModelView, rigid_contact_max: int) -> int:
    """Estimate the busiest world's share of a global rigid-contact capacity.

    The collision pipeline stores contacts in one heterogeneous global buffer,
    while downstream solvers may allocate constraint rows per world. Bound that
    per-world allocation from the model's world-compatible shape pairs instead
    of duplicating the entire global contact capacity into every world.

    Args:
        model: The simulation model.
        rigid_contact_max: Global rigid-contact capacity.

    Returns:
        Maximum estimated contacts that one world can contribute, capped by the
        global rigid-contact capacity.
    """
    if rigid_contact_max <= 0 or model.shape_contact_pair_count == 0:
        return 0

    counts = _shape_contact_pair_counts(model)
    is_mesh = _shape_may_be_mesh(model, model.shape_type.numpy(), _variant_options(model))
    primitive_counts = _shape_contact_pair_counts(model, shape_mask=~is_mesh) if is_mesh.any() else counts
    contacts = (
        primitive_counts * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR
        + (counts - primitive_counts) * _RIGID_CONTACTS_PER_MESH_PAIR
    )
    return min(rigid_contact_max, int(contacts[0] + np.max(contacts[1:], initial=0)))


# Mesh-like pairs (mesh, heightfield, planar SDF) feed the triangle-pair buffer and the
# global contact reducer. Measured mesh/SDF piles emit about 10-120 reduction candidates
# per touching pair before reduction; convex shapes on dense meshes can emit more.
_TRIANGLE_PAIRS_PER_MESH_PAIR = 128
_TRIANGLE_PAIR_NEIGHBORS_PER_SHAPE = 6
# Legacy fixed default; automatic sizing never allocates less.
_TRIANGLE_PAIRS_MIN_CAPACITY = 1_000_000
_TRIANGLE_PAIRS_LARGE_BUFFER_BYTES = 256 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class _TrianglePairCountEstimate:
    """Diagnostic details for an automatically selected triangle-pair capacity."""

    capacity: int
    world_count: int
    mesh_pair_count: int
    budgeted_pair_count: int
    busiest_world_pair_count: int
    source: Literal["none", "shape pairs", "shape counts"]


def _estimate_triangle_pair_details(
    model: Model,
    shape_pairs: np.ndarray | None,
    collide_mask: np.ndarray,
    mesh_like_mask: np.ndarray,
    planar_sdf_mask: np.ndarray,
    hydroelastic: bool,
    variant_row_mask: np.ndarray | None = None,
) -> _TrianglePairCountEstimate:
    """Estimate the triangle-pair and global contact-reducer capacity from the model.

    The estimate sums per-world budgets, so a batch in which only some worlds
    contain meshes is sized by those worlds rather than ``world_count`` times the
    busiest one. For each world it counts the shape pairs routed to the mesh path
    (at least one mesh or heightfield, or two planar-SDF shapes), limits that
    count to ``_TRIANGLE_PAIR_NEIGHBORS_PER_SHAPE`` times half the participating
    shapes, and budgets ``_TRIANGLE_PAIRS_PER_MESH_PAIR`` candidates per pair.
    Global shapes count once in every world they can touch, and global-global
    pairs count once.

    With ``shape_pairs`` the routed pairs are counted exactly. Without them
    (NXN/SAP) they are counted per world from
    shape-type counts, skipping static-static pairs; this ignores group,
    filter-pair, kinematic, and hydroelastic culling and so can only overcount.

    This is a heuristic, not a worst-case bound. The neighbor limit is an
    aggregate budget, not a per-shape guarantee, and the per-pair budget assumes
    small contact patches. Dense meshes or heightfields under convex shapes, large
    detection gaps, or worlds where most pairs touch at once can exceed it; the
    overflow warning then names ``max_triangle_pairs``.

    Args:
        model: The simulation model.
        shape_pairs: ``(N, 2)`` candidate shape pairs from the configured explicit
            list, or ``None`` to count from masks.
        collide_mask: Shapes the broad phase pairs; used only without ``shape_pairs``.
        mesh_like_mask: Routed mesh and heightfield shapes.
        planar_sdf_mask: Routed shapes with SDF edge data.
        hydroelastic: Whether hydroelastic-hydroelastic pairs use their own reducer.
        variant_row_mask: Variant rows, whose current type does not limit the pairs they can route.

    Returns:
        The estimated capacity, never below ``_TRIANGLE_PAIRS_MIN_CAPACITY``, and its inputs.
    """
    world_count = int(getattr(model, "world_count", 0) or 0)
    shape_world_array = getattr(model, "shape_world", None)
    if shape_pairs is None and shape_world_array is None:
        return _TrianglePairCountEstimate(_TRIANGLE_PAIRS_MIN_CAPACITY, world_count, 0, 0, 0, "none")
    shape_world = (
        shape_world_array.numpy().astype(np.int64)
        if shape_world_array is not None
        else np.zeros(len(mesh_like_mask), dtype=np.int64)
    )

    if shape_pairs is not None:
        pairs_per_bucket, participants_per_bucket = _count_routed_pairs(
            model, shape_pairs, shape_world, mesh_like_mask, planar_sdf_mask, hydroelastic, variant_row_mask
        )
        source = "shape pairs"
    else:
        pairs_per_bucket, participants_per_bucket = _count_routed_pairs_from_masks(
            shape_world,
            model.shape_body.numpy(),
            collide_mask,
            mesh_like_mask,
            planar_sdf_mask,
            model.shape_type.numpy(),
        )
        source = "shape counts"

    neighbor_budget = (participants_per_bucket * _TRIANGLE_PAIR_NEIGHBORS_PER_SHAPE + 1) // 2
    budgeted_pairs = np.minimum(pairs_per_bucket, neighbor_budget)
    budgeted_pair_count = int(budgeted_pairs.sum())
    return _TrianglePairCountEstimate(
        capacity=max(_TRIANGLE_PAIRS_MIN_CAPACITY, budgeted_pair_count * _TRIANGLE_PAIRS_PER_MESH_PAIR),
        world_count=world_count,
        mesh_pair_count=int(pairs_per_bucket.sum()),
        budgeted_pair_count=budgeted_pair_count,
        busiest_world_pair_count=int(budgeted_pairs[1:].max(initial=0)),
        source=source,
    )


def _count_routed_pairs(
    model: Model,
    shape_pairs: np.ndarray,
    shape_world: np.ndarray,
    mesh_like_mask: np.ndarray,
    planar_sdf_mask: np.ndarray,
    hydroelastic: bool,
    variant_row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Count routed pairs and their participating shapes per bucket from an explicit pair list.

    Bucket 0 holds global-global pairs and bucket ``w + 1`` holds pairs of world ``w``;
    an explicit cross-world pair is counted once, in the bucket of its larger world.
    """
    shape_types = model.shape_type.numpy()
    shape_a = shape_pairs[:, 0]
    shape_b = shape_pairs[:, 1]
    # Explicit pair lists may carry entries the narrow phase skips.
    valid = (shape_a >= 0) & (shape_b >= 0) & (shape_a < len(shape_types)) & (shape_b < len(shape_types))
    valid &= shape_a != shape_b
    shape_a = shape_a[valid]
    shape_b = shape_b[valid]
    heightfield = shape_types == int(GeoType.HFIELD)
    box = shape_types == int(GeoType.BOX)
    if variant_row_mask is not None:
        box &= ~variant_row_mask
    routed = (mesh_like_mask[shape_a] | mesh_like_mask[shape_b]) & ~(heightfield[shape_a] & heightfield[shape_b])
    routed |= planar_sdf_mask[shape_a] & planar_sdf_mask[shape_b] & ~(box[shape_a] & box[shape_b])
    if hydroelastic:
        hydro = (model.shape_flags.numpy() & int(ShapeFlags.HYDROELASTIC)) != 0
        routed &= ~(hydro[shape_a] & hydro[shape_b])
    shape_a = shape_a[routed].astype(np.int64)
    shape_b = shape_b[routed].astype(np.int64)
    if len(shape_a) == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)

    bucket = np.maximum(shape_world[shape_a], shape_world[shape_b]) + 1
    bucket_count = int(bucket.max()) + 1
    pairs_per_bucket = np.bincount(bucket, minlength=bucket_count)
    participant_keys = np.unique(np.concatenate((shape_a, shape_b)) * bucket_count + np.concatenate((bucket, bucket)))
    participants_per_bucket = np.bincount(participant_keys % bucket_count, minlength=bucket_count)
    return pairs_per_bucket, participants_per_bucket


def _count_routed_pairs_from_masks(
    shape_world: np.ndarray,
    shape_body: np.ndarray,
    collide_mask: np.ndarray,
    mesh_like_mask: np.ndarray,
    planar_sdf_mask: np.ndarray,
    shape_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound routed pairs and their participating shapes per bucket from per-world shape counts.

    Uses the same buckets as :func:`_count_routed_pairs`. Each world's segment holds its own
    shapes plus the global ones; global-global pairs are removed from it and counted once.
    """
    heightfield = mesh_like_mask & (shape_types == int(GeoType.HFIELD))
    planar_only = planar_sdf_mask & ~mesh_like_mask
    bucket_count = int(shape_world.max(initial=-1)) + 2

    def counts(mask: np.ndarray) -> np.ndarray:
        selected = collide_mask & mask
        per_bucket = np.bincount(shape_world[selected] + 1, minlength=bucket_count).astype(np.int64)
        # Every world segment also contains the global shapes.
        per_bucket[1:] += per_bucket[0]
        return per_bucket

    def pairs(n: np.ndarray) -> np.ndarray:
        return n * (n - 1) // 2

    def routed_pairs(mask: np.ndarray) -> np.ndarray:
        # At least one mesh-like shape, minus heightfield-heightfield, plus planar-SDF pairs.
        n = counts(mask)
        return (
            pairs(n)
            - pairs(n - counts(mask & mesh_like_mask))
            - pairs(counts(mask & heightfield))
            + pairs(counts(mask & planar_only))
        )

    shapes = counts(np.ones_like(collide_mask))
    mesh_like = counts(mesh_like_mask)
    planar = counts(planar_only)
    # Pairs between two static shapes never reach the narrow phase.
    routed = routed_pairs(np.ones_like(collide_mask)) - routed_pairs(shape_body < 0)
    participants = np.where(mesh_like > 0, shapes, np.where(planar > 1, planar, 0))
    participants = np.where(routed > 0, participants, 0)
    # Remove the global-global pairs repeated in every world segment.
    routed[1:] -= routed[0]
    participants[1:] = np.where(routed[1:] > 0, participants[1:], 0)
    return routed, participants


def _resolve_auto_triangle_pairs(
    estimate: _TrianglePairCountEstimate,
    *,
    has_reducer: bool,
    deterministic: bool,
    speculative: bool,
    hashtable_size_factor: float,
) -> int:
    """Apply route limits to an automatic estimate and warn about clipping or large allocations."""
    capacity = estimate.capacity
    # Deterministic reduction packs reducer buffer indices into CONTACT_ID_BITS.
    packing_limit = (1 << int(CONTACT_ID_BITS)) - 1
    if has_reducer and deterministic and capacity > packing_limit:
        warnings.warn(
            f"CollisionPipeline estimated max_triangle_pairs={capacity:,} but deterministic contact reduction "
            f"(deterministic=True or contact matching) supports at most {packing_limit:,}, so mesh contacts may "
            "be dropped. Split the worlds across pipelines, or disable determinism and contact matching if "
            "acceptable.",
            NewtonWarning,
            stacklevel=3,
        )
        capacity = packing_limit

    allocation_bytes = _triangle_pair_allocation_bytes(capacity, hashtable_size_factor, has_reducer, speculative)
    if allocation_bytes >= _TRIANGLE_PAIRS_LARGE_BUFFER_BYTES:
        legacy_bytes = _triangle_pair_allocation_bytes(
            _TRIANGLE_PAIRS_MIN_CAPACITY, hashtable_size_factor, has_reducer, speculative
        )
        warnings.warn(
            f"CollisionPipeline automatically selected max_triangle_pairs={capacity:,}, requiring about "
            f"{allocation_bytes / (1024 * 1024):,.1f} MiB for triangle-pair and contact-reduction buffers "
            f"(the minimum of {_TRIANGLE_PAIRS_MIN_CAPACITY:,} needs {legacy_bytes / (1024 * 1024):,.1f} MiB). "
            f"Estimate inputs -- {estimate.budgeted_pair_count:,} budgeted of {estimate.mesh_pair_count:,} "
            f"mesh-routed pairs from {estimate.source}, at most {estimate.busiest_world_pair_count:,} in one of "
            f"{estimate.world_count:,} worlds, {_TRIANGLE_PAIRS_PER_MESH_PAIR} candidates per pair. Pass "
            "max_triangle_pairs explicitly to choose the memory budget and silence this warning.",
            NewtonWarning,
            stacklevel=3,
        )
    return capacity


def _triangle_pair_allocation_bytes(
    capacity: int, hashtable_size_factor: float, has_reducer: bool, speculative: bool
) -> int:
    """Return the bytes the narrow phase allocates for a triangle-pair capacity."""
    # Triangle-pair slot (12 B). With reduction: reducer contact slot (40 B), and per hashtable
    # entry a key, an active-slot index, and one value per reduction slot (68 B).
    if not has_reducer:
        return capacity * 12
    hashtable_entries = 1 << (max(int(capacity * hashtable_size_factor), 1024) - 1).bit_length()
    reclaim_bytes = (capacity // 32 + 1) * 4 if speculative else 0
    return capacity * 52 + hashtable_entries * 68 + reclaim_bytes


def _compute_per_world_shape_pairs_max(model: Model) -> int:
    """Compute the maximum number of candidate shape pairs using per-world counts.

    For multi-world scenes the global formula ``N*(N-1)/2`` is O(W^2 * S^2)
    where W is the number of worlds and S is shapes per world.  The correct
    upper bound is the sum of per-world lower-triangular counts which is
    O(W * S^2).

    The result mirrors the segment layout produced by
    :func:`precompute_world_map`: each regular world's segment contains the
    world's local shapes **plus** all global shapes (world == -1), and a
    dedicated final segment contains only the global shapes.  Each segment
    contributes ``n*(n-1)/2`` candidate pairs independently.
    """
    shape_world = getattr(model, "shape_world", None)
    shape_count = model.shape_count
    if shape_world is None or shape_count <= 1:
        return max(0, (shape_count * (shape_count - 1)) // 2)

    sw = shape_world.numpy()
    colliding = _shape_collide_mask(model, len(sw))

    global_count = int(np.count_nonzero((sw == -1) & colliding))
    _, world_counts = np.unique(sw[(sw >= 0) & colliding], return_counts=True)

    total = 0
    for count in world_counts:
        n = int(count) + global_count
        total += (n * (n - 1)) // 2

    # Dedicated global-vs-global segment (appended by precompute_world_map).
    total += (global_count * (global_count - 1)) // 2

    return max(0, total)


def _compute_per_world_mask_pair_max(
    model: Model,
    first_mask: np.ndarray,
    second_mask: np.ndarray | None = None,
) -> int:
    """Compute a world-compatible pair bound for selected shape sets."""
    if second_mask is None:
        second_mask = first_mask

    shape_world = getattr(model, "shape_world", None)
    if shape_world is None:
        overlap = int(np.count_nonzero(first_mask & second_mask))
        return int(np.count_nonzero(first_mask)) * int(np.count_nonzero(second_mask)) - overlap * (overlap + 1) // 2

    sw = shape_world.numpy()
    colliding = _shape_collide_mask(model, len(sw))
    # Group once instead of scanning all shapes for every world's segment.
    # Compact indices also avoid allocating by the largest (possibly sparse) world ID.
    world_ids, inverse = np.unique(sw, return_inverse=True)
    active = (world_ids >= 0) & (np.bincount(inverse[colliding], minlength=len(world_ids)) > 0)

    def segment_counts(mask: np.ndarray) -> np.ndarray:
        counts = np.bincount(inverse[mask], minlength=len(world_ids)).astype(np.int64, copy=False)
        global_count = counts[world_ids == -1].sum()
        return np.append(counts[active] + global_count, global_count)

    first_counts = segment_counts(first_mask)
    second_counts = segment_counts(second_mask)
    overlaps = segment_counts(first_mask & second_mask)
    pairs = first_counts * second_counts - overlaps * (overlaps + 1) // 2
    return max(0, int(pairs.sum()))


def _resolve_shape_pairs_max(model: Model, override: int | None) -> int:
    """Pick the broad-phase candidate-pair buffer capacity.

    ``override`` lets the caller cap the SAP/NXN pair buffer, which is
    otherwise sized to the worst-case ``N*(N-1)/2`` per-world bound.
    SAP and NXN scenes with thousands of bodies typically emit only a
    tiny fraction of that bound, so the default sizing is grossly
    wasteful (multi-GB on 10k+ shape scenes). ``None`` keeps the legacy
    behaviour; a positive integer overrides it. ``0`` is rejected --
    use ``None`` instead.  Values larger than the natural bound are
    accepted as-is: allocating beyond the bound never produces more
    pairs, but we honour the user's explicit capacity request rather
    than silently shrinking it.
    """
    if override is None:
        return _compute_per_world_shape_pairs_max(model)
    if override <= 0:
        raise ValueError(f"shape_pairs_max must be a positive integer or None, got {override}")
    return int(override)


BROAD_PHASE_MODES = ("nxn", "sap", "explicit")
# The fused kernel inlines overlap refinement for every pair, while split
# kernels refine on a concurrent stream. Splitting pays off once the extra
# launches amortize: measured on the RTX PRO 6000 at 16,384 worlds, from about
# 28k lean or 64k full-support pairs (cassie-rough fused is 1.3% faster at
# upstream's 4k/8k; cassie-flat and g1-flat split under both).
_SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD = 27_776
_SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD = 65_536


def _compute_generic_convex_pair_stats(
    model: Model,
    *,
    broad_phase_mode: str,
    shape_pairs_filtered: wp.array[wp.vec2i] | np.ndarray | None,
    candidate_pair_work_estimate: int,
    shape_pairs_host: np.ndarray | None = None,
    variant_options: _VariantOptions | None = None,
) -> tuple[bool, int]:
    """Determine whether generic convex pairs exist and estimate their work.

    ``shape_pairs_host`` is an optional ``(N, 2)`` host copy of
    ``shape_pairs_filtered`` that spares another device-to-host read. With
    ``variant_options``, a variant row counts as every colliding variant shape it can hold.
    """
    shape_types_array = getattr(model, "shape_type", None)
    if shape_types_array is None:
        return True, candidate_pair_work_estimate
    shape_types = shape_types_array.numpy()
    # Bit t of a shape's entry is set if the shape is, or as a variant row can become, of type t.
    type_bits = np.left_shift(1, shape_types.astype(np.int64))
    if variant_options is not None:
        row_shapes, row_bits = variant_options.type_bits()
        type_bits[row_shapes] |= row_bits

    if broad_phase_mode == "explicit":
        if shape_pairs_filtered is None:
            return True, candidate_pair_work_estimate
        pairs = shape_pairs_host if shape_pairs_host is not None else shape_pairs_filtered
        generic_pair_count = 0
        # Bound temporary gathers independently of the replicated world count.
        for block in _iter_pair_blocks(pairs):
            if variant_options is None:
                pair_types = shape_types[block]
                generic = _GENERIC_CONVEX_PAIR_LOOKUP[pair_types[:, 0], pair_types[:, 1]]
            else:
                bits_a = type_bits[block[:, 0]]
                bits_b = type_bits[block[:, 1]]
                generic = np.zeros(len(block), dtype=bool)
                for type_a, type_b in zip(*np.nonzero(_GENERIC_CONVEX_PAIR_LOOKUP), strict=True):
                    generic |= ((bits_a >> type_a) & 1).astype(bool) & ((bits_b >> type_b) & 1).astype(bool)
            generic_pair_count += int(np.count_nonzero(generic))
        return generic_pair_count > 0, min(candidate_pair_work_estimate, generic_pair_count)

    colliding_mask = _shape_collide_mask(model, len(shape_types))
    generic_pair_bound = 0
    unique_types = [t for t in range(len(GeoType)) if np.any(colliding_mask & ((type_bits >> t) & 1).astype(bool))]
    for index, type_a in enumerate(unique_types):
        first_mask = colliding_mask & ((type_bits >> type_a) & 1).astype(bool)
        for type_b in unique_types[index:]:
            if not _pair_requires_generic_convex_narrow_phase(int(type_a), int(type_b)):
                continue
            second_mask = colliding_mask & ((type_bits >> type_b) & 1).astype(bool)
            generic_pair_bound += _compute_per_world_mask_pair_max(model, first_mask, second_mask)

    return generic_pair_bound > 0, min(candidate_pair_work_estimate, generic_pair_bound)


def _normalize_broad_phase_mode(mode: str) -> str:
    mode_str = str(mode).lower()
    if mode_str not in BROAD_PHASE_MODES:
        raise ValueError(f"Unsupported broad phase mode: {mode!r}")
    return mode_str


def _infer_broad_phase_mode_from_instance(broad_phase: BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit) -> str:
    if isinstance(broad_phase, BroadPhaseAllPairs):
        return "nxn"
    if isinstance(broad_phase, BroadPhaseSAP):
        return "sap"
    if isinstance(broad_phase, BroadPhaseExplicit):
        return "explicit"
    raise TypeError(
        "broad_phase must be a BroadPhaseAllPairs, BroadPhaseSAP, or BroadPhaseExplicit instance "
        f"(got {type(broad_phase)!r})"
    )


def _world_compatible_pairs(
    feature_world: np.ndarray,
    shape_world: np.ndarray,
    world_count: int,
    device,
    shape_ok: np.ndarray | None = None,
) -> wp.array[wp.vec2i]:
    """Emit ``(feature, shape)`` index pairs whose worlds are compatible: same world, or either is
    global (``-1``). ``feature_world[i]`` / ``shape_world[s]`` give each entity's world (-1 == global).

    Pairs are stably sorted by shape index so consecutive candidates process the same shape: on CUDA
    a warp then reads one shape's transform/scale/SDF data and takes one type-dispatch branch. The
    sort runs on the host at construction; each contact record stores its candidate tid, so
    downstream mapping does not depend on candidate order.

    Worlds are immutable after :meth:`~newton.ModelBuilder.finalize`, so this filtering is safe to
    precompute; mutable per-entity flags (ACTIVE / COLLIDE_PARTICLES) are deliberately left to the
    per-thread kernel. The compatibility predicate splits into three disjoint groups, each a
    vectorized cross product (disjoint => no de-duplication; no Python loop over features or shapes).
    Reads host arrays, so it is not graph-capture-safe; call at pipeline construction.
    """
    n_features = len(feature_world)
    n_shapes = len(shape_world)

    def _pairs(f_idx: np.ndarray, s_idx: np.ndarray) -> wp.array[wp.vec2i]:
        if len(s_idx):
            order = np.argsort(s_idx, kind="stable")
            f_idx, s_idx = f_idx[order], s_idx[order]
        stacked = np.column_stack((f_idx, s_idx)).astype(np.int32) if len(f_idx) else np.empty((0, 2), np.int32)
        return wp.array(stacked, dtype=wp.vec2i, device=device)

    if n_features == 0 or n_shapes == 0:
        return _pairs(np.empty(0), np.empty(0))

    features = np.arange(n_features)
    shapes = np.arange(n_shapes)
    # Filter before the Cartesian products to avoid materializing pairs for other shape passes.
    if shape_ok is not None:
        shapes = shapes[shape_ok]
        shape_world = shape_world[shape_ok]
    f_local = (feature_world >= 0) & (feature_world < world_count)
    s_local = (shape_world >= 0) & (shape_world < world_count)

    f_cols: list[np.ndarray] = []
    s_cols: list[np.ndarray] = []

    # 1. Global features pair with every shape (any world).
    global_features = features[feature_world < 0]
    if len(global_features):
        f_cols.append(np.tile(global_features, len(shapes)))
        s_cols.append(np.repeat(shapes, len(global_features)))

    # 2. Local-world features additionally pair with every global shape.
    local_features = features[f_local]
    global_shapes = shapes[shape_world < 0]
    if len(local_features) and len(global_shapes):
        f_cols.append(np.tile(local_features, len(global_shapes)))
        s_cols.append(np.repeat(global_shapes, len(local_features)))

    # 3. Group local features by world, then slice out each local shape's compatible block.
    local_feature_world = feature_world[f_local]
    local_shapes = shapes[s_local]
    local_shape_world = shape_world[s_local]
    features_per_world = np.bincount(local_feature_world, minlength=world_count)
    reps = features_per_world[local_shape_world] if len(local_shape_world) else np.zeros(0, np.intp)
    if reps.sum():
        features_by_world = local_features[np.argsort(local_feature_world, kind="stable")]
        world_start = np.cumsum(features_per_world) - features_per_world
        within = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
        f_cols.append(features_by_world[np.repeat(world_start[local_shape_world], reps) + within])
        s_cols.append(np.repeat(local_shapes, reps))

    if not f_cols:
        return _pairs(np.empty(0), np.empty(0))
    f_idx = np.concatenate(f_cols)
    s_idx = np.concatenate(s_cols)
    return _pairs(f_idx, s_idx)


def _build_soft_particle_rigid_contact_pairs(model: Model, *, shape_ok: np.ndarray | None = None) -> wp.array[wp.vec2i]:
    """Build the soft-rigid (particle-shape) candidate pairs for ``model``.

    Emits every particle-shape pair whose worlds are compatible (see :func:`_world_compatible_pairs`).
    :attr:`~newton.ParticleFlags.ACTIVE` and :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES` are applied
    per-thread in :func:`~newton._src.geometry.kernels.create_soft_contacts`, not here, so the
    candidate set stays valid when those flags change after the pipeline is constructed.
    """
    particle_count = int(getattr(model, "particle_count", 0) or 0)
    shape_count = int(getattr(model, "shape_count", 0) or 0)
    if particle_count == 0 or shape_count == 0:
        return wp.array(np.empty((0, 2), np.int32), dtype=wp.vec2i, device=model.device)
    world_count = int(getattr(model, "world_count", 0) or 0)
    return _world_compatible_pairs(
        model.particle_world.numpy(), model.shape_world.numpy(), world_count, model.device, shape_ok
    )


def _count_soft_particle_rigid_contact_pairs(model: Model) -> int:
    """Count exactly how many pairs :func:`_build_soft_particle_rigid_contact_pairs` emits for ``model``.

    Reads only the per-world start offsets, so solvers can pre-size soft-contact buffers without
    downloading per-entity world ids. This is not :attr:`CollisionPipeline.soft_contact_max`, which
    additionally reserves edge/face headroom when ``enable_rigid_soft_full_surface_contact`` is set.
    Reads host arrays, so it is not graph-capture-safe; call at solver construction.
    """
    particle_start = model.particle_world_start.numpy()
    shape_start = model.shape_world_start.numpy()
    global_particles = int(particle_start[-1] - particle_start[-2] + particle_start[0])
    global_shapes = int(shape_start[-1] - shape_start[-2] + shape_start[0])
    # Global particles pair with every shape; local particles additionally pair with global shapes.
    total = global_particles * model.shape_count
    total += (model.particle_count - global_particles) * global_shapes
    # Local particles pair with the shapes sharing their world.
    per_world = slice(0, model.world_count + 1)
    return total + int(
        np.dot(np.diff(particle_start[per_world]).astype(np.int64), np.diff(shape_start[per_world]).astype(np.int64))
    )


def _build_soft_face_rigid_contact_pairs(
    model: Model, capable_shape_mask: np.ndarray | None = None
) -> wp.array[wp.vec2i]:
    """World-compatible ``(soft triangle, shape)`` candidate pairs for the full-surface FACE pass,
    mirroring :func:`_build_soft_particle_rigid_contact_pairs`. A triangle's world is the world of
    its first vertex (all three share it). Empty when there are no triangles or no shapes.
    """
    device = model.device
    empty = wp.array(np.empty((0, 2), np.int32), dtype=wp.vec2i, device=device)
    shape_count = int(getattr(model, "shape_count", 0) or 0)
    n_tris = int(getattr(model, "tri_count", 0) or 0)
    if shape_count == 0 or n_tris == 0:
        return empty
    world_count = int(getattr(model, "world_count", 0) or 0)
    face_world = model.particle_world.numpy()[model.tri_indices.numpy()[:, 0]]
    return _world_compatible_pairs(
        face_world,
        model.shape_world.numpy(),
        world_count,
        device,
        shape_ok=capable_shape_mask,
    )


def _build_soft_edge_rigid_contact_pairs(
    model: Model, capable_shape_mask: np.ndarray | None = None
) -> wp.array[wp.vec2i]:
    """World-compatible ``(soft edge, shape)`` candidate pairs for the full-surface EDGE pass,
    mirroring :func:`_build_soft_particle_rigid_contact_pairs`. An edge's world is that of one of its
    endpoints. Endpoints come straight from ``model.edge_indices`` (no mesh adjacency needed). Empty
    when there are no edges or no shapes.
    """
    device = model.device
    empty = wp.array(np.empty((0, 2), np.int32), dtype=wp.vec2i, device=device)
    shape_count = int(getattr(model, "shape_count", 0) or 0)
    n_edges = int(getattr(model, "edge_count", 0) or 0)
    if shape_count == 0 or n_edges == 0:
        return empty
    world_count = int(getattr(model, "world_count", 0) or 0)
    # edge_indices rows are [o0, o1, v0, v1]; col 2 (v0) is an endpoint, so its world is the edge's.
    edge_world = model.particle_world.numpy()[model.edge_indices.numpy()[:, 2]]
    return _world_compatible_pairs(
        edge_world,
        model.shape_world.numpy(),
        world_count,
        device,
        shape_ok=capable_shape_mask,
    )


def _full_surface_capable_shape_mask(model: Model) -> np.ndarray:
    """Boolean mask over shapes: ``True`` where the shape can generate full-surface edge/face contacts.

    Analytic primitives and triangle meshes are supported. Heightfields use a
    separate structured-grid pass. Mesh contacts do not require a volume SDF.
    """
    stype = model.shape_type.numpy()
    analytic = np.isin(
        stype,
        (
            int(GeoType.SPHERE),
            int(GeoType.BOX),
            int(GeoType.CAPSULE),
            int(GeoType.CYLINDER),
            int(GeoType.CONE),
            int(GeoType.ELLIPSOID),
            int(GeoType.PLANE),
        ),
    )
    is_mesh = np.isin(stype, (int(GeoType.MESH), int(GeoType.CONVEX_MESH)))
    return analytic | is_mesh


def _warn_full_surface_fallbacks(model: Model, capable: np.ndarray) -> None:
    """Warn about participating shape types that cannot do edge/face contacts and fall back to
    per-particle soft contact. Meshes use exact feature contacts without an SDF."""
    stype = model.shape_type.numpy()
    is_mesh = np.isin(stype, (int(GeoType.MESH), int(GeoType.CONVEX_MESH)))
    collide_particles = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
    fallback = np.where(collide_particles & ~capable & ~is_mesh)[0]
    if fallback.size == 0:
        return
    labels = getattr(model, "shape_key", None)

    def _label(i: int) -> str:
        return labels[i] if labels is not None and i < len(labels) else f"shape {int(i)}"

    fallback_labels = [_label(i) for i in fallback]
    warnings.warn(
        "enable_rigid_soft_full_surface_contact=True: these participating shapes cannot generate "
        f"edge/face contacts and fall back to per-particle soft contact only -- {fallback_labels} "
        "(shape types without an analytic signed-distance field)"
        + ". Full-surface contacts still apply to the rest of the scene.",
        NewtonWarning,
        stacklevel=3,
    )


_TYPED_TERRAIN_QUERY = "_contacts_to_reducer_kernel"


def _heightfield_row(kernel) -> dict:
    """Describe one typed terrain query from its factory-generated kernel key.

    ``_create_mesh_triangle_contacts_to_reducer_kernel`` names typed queries
    ``heightfield_<type>_contacts_to_reducer_kernel[_analytic]``; the fallback
    appends ``_without_hfield_convex`` or ``_analytic_convex``.
    """
    key = kernel.key
    if key.startswith("heightfield_") and _TYPED_TERRAIN_QUERY in key:
        row = key[len("heightfield_") : key.index(_TYPED_TERRAIN_QUERY)]
        return {"row": row, "kernel": key, "closed_form": key.endswith("_analytic")}
    hull = (
        "excluded" if "_without_hfield_convex" in key else "closed_form" if key.endswith("_analytic_convex") else "gjk"
    )
    return {"row": "fallback", "kernel": key, "hull": hull}


def _route_gate(enabled: bool, requested: bool, flag: str, blockers, request: str | None = None) -> str:
    """Explain an opt-in route: admitted, off, or the first admission gate that declined it.

    ``request`` names an argument request; the default names the environment flag.
    """
    request = request or f"{flag}=1"
    if enabled:
        return f"{request}: admitted."
    if not requested:
        return f"Off: {flag} is not 1."
    for blocked, reason in blockers:
        if blocked:
            return f"{request} declined: {reason}."
    return f"{request} declined: a custom contact writer."


class CollisionPipeline:
    """
    Full-featured collision pipeline with GJK/MPR narrow phase and pluggable broad phase.

    Key features:
        - GJK/MPR algorithms for convex-convex collision detection
        - Multiple broad phase options: NXN (all-pairs), SAP (sweep-and-prune), EXPLICIT (precomputed pairs)
        - Mesh-mesh collision via SDF with contact reduction
        - Optional hydroelastic contact model for compliant surfaces

    For most users, construct with ``CollisionPipeline(model, ...)``.

    .. experimental::

        Differentiable rigid-contact kinematics computed by
        :func:`newton.eval_rigid_contact_kinematics` may change
        without prior notice. The narrow phase stays frozen and gradients are
        a tangent approximation; validate accuracy and usefulness on your
        workflow before relying on them in optimization loops.
    """

    @dataclass(frozen=True, kw_only=True)
    class BodyPairReductionConfig:
        """Configure body-pair contact reduction.

        Bodies built from many colliders produce one manifold per shape pair.
        After the narrow phase has written the ``Contacts`` buffer (and after
        the deterministic sort), the pipeline compacts it per group of
        ``(body or static shape, other body or static shape, normal bin,
        spatial cell)``: each group keeps one depth representative, up to six
        sampled footprint extremes over all its contacts and up to six more over
        its touching contacts (canonical separation ``<= 0``), so
        ``rigid_contact_count`` itself drops.  Shapes on one body share a group
        only when their ``shape_material_*`` values and ``material_attributes``
        match exactly.  With ``hysteresis > 0`` the previous step's winners keep
        their slots while they trail the instantaneous winner by at most the
        margin, so a resting multi-collider body stops rocking between
        near-symmetric contacts.

        It is an approximation with scene-dependent support error and possible
        upward torsional-friction bias; validate tipping- and yaw-sensitive
        tasks.  Only solvers with
        :attr:`~newton.solvers.SolverBase.supports_body_pair_reduced_contacts`
        accept the reduced buffers.  Sticky contact matching and hydroelastic
        contacts are rejected at construction.  Call
        :meth:`CollisionPipeline.reset_contact_matching` (or
        :meth:`CollisionPipeline.reset_body_pair_reduction_history`) after
        episode resets or teleports, and
        :meth:`CollisionPipeline.refresh_body_pair_reduction_groups` after
        runtime material changes.  A buffer that replaces the pipeline's first
        one is collided once outside CUDA graph capture before it is captured.

        .. experimental::
        """

        cell_size: float = 0.25
        """Spatial cell edge [m] that keeps separated same-normal patches of one body pair
        independently represented."""
        hysteresis: float = 0.001
        """Previous-winner preference [m].  Also how far a touching-slot winner may lift off and
        still compete in the touching family, without the preference.  ``0`` gives memoryless
        selection."""
        hashtable_size_factor: float = 1.0
        """Multiplier on the group-table capacity, which is derived from the model's contact-pair topology
        rather than from the contact rows.  If the table is too small, the affected frames keep every
        contact (``fallback_frames`` in :meth:`CollisionPipeline.body_pair_reduction_stats`)."""
        material_attributes: tuple[str, ...] | None = None
        """Per-shape custom attributes, by full name such as ``"steel:min_torsional_patch_radius"``, that
        must also match for shapes on one body to share a group.  ``None`` uses every per-shape custom
        attribute the model registers."""
        verify: bool = False
        """Recheck the reducer's invariants each frame (tests and debugging)."""

        def __post_init__(self):
            """Validate configuration before any device allocation."""
            if not isinstance(self.verify, bool):
                raise TypeError("BodyPairReductionConfig.verify must be bool")
            try:
                cell_size = float(self.cell_size)
                hysteresis = float(self.hysteresis)
                hashtable_size_factor = float(self.hashtable_size_factor)
            except (TypeError, ValueError) as error:
                raise TypeError("BodyPairReductionConfig numeric fields must be real numbers") from error
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                cell_size_f32 = float(np.float32(cell_size))
                hysteresis_f32 = float(np.float32(hysteresis))
            if not np.isfinite(cell_size_f32) or cell_size_f32 <= 0.0:
                raise ValueError("BodyPairReductionConfig.cell_size must be finite and > 0 at float32 precision")
            if (
                not np.isfinite(hysteresis_f32)
                or hysteresis < 0.0
                or hysteresis_f32 < 0.0
                or (hysteresis > 0.0 and hysteresis_f32 == 0.0)
            ):
                raise ValueError(
                    "BodyPairReductionConfig.hysteresis must be finite and >= 0 without underflow at float32 precision"
                )
            if not np.isfinite(hashtable_size_factor) or hashtable_size_factor <= 0.0:
                raise ValueError("BodyPairReductionConfig.hashtable_size_factor must be finite and > 0")
            if self.material_attributes is not None:
                if not isinstance(self.material_attributes, (tuple, list)) or not all(
                    isinstance(name, str) for name in self.material_attributes
                ):
                    raise TypeError(
                        "BodyPairReductionConfig.material_attributes must be a tuple or list of str, or None"
                    )
                object.__setattr__(self, "material_attributes", tuple(self.material_attributes))

    def __init__(
        self,
        model: Model,
        *,
        reduce_contacts: bool = True,
        body_pair_reduction: CollisionPipeline.BodyPairReductionConfig | None = None,
        rigid_contact_max: int | None = None,
        max_triangle_pairs: int | None = None,
        shape_pairs_filtered: wp.array[wp.vec2i] | None = None,
        include_static_kinematic_pairs: bool = True,
        include_immobile_pairs: bool | None = None,
        soft_contact_max: int | None = None,
        soft_contact_gap: float | None = None,
        soft_contact_margin: float | None = None,
        enable_rigid_soft_full_surface_contact: bool = False,
        requires_grad: bool | None = None,
        broad_phase: Literal["nxn", "sap", "explicit"]
        | BroadPhaseAllPairs
        | BroadPhaseSAP
        | BroadPhaseExplicit
        | None = None,
        narrow_phase: NarrowPhase | None = None,
        sdf_hydroelastic_config: HydroelasticSDF.Config | None = None,
        shape_pairs_max: int | None = None,
        broad_phase_output_max: int | None = None,
        deterministic: bool = False,
        box_box_sat: bool = False,
        contact_matching: Literal["disabled", "latest", "sticky"] = "disabled",
        contact_matching_pos_threshold: float = 0.0005,
        contact_matching_normal_dot_threshold: float = 0.995,
        contact_report: bool = False,
        verify_buffers: bool = True,
        contact_reduction_hashtable_size_factor: float = 0.25,
        speculative_contact_gap_max: float | None = None,
        dynamic_shape_scale: bool = False,
        publish_capacity: bool = True,
        growable_contacts: bool = False,
        terrain_meshes: bool = False,
        convex_collision: Literal["auto", "ordinary", "rejection"] | None = None,
        contact_export: Literal["auto", "tiled", "compact"] | None = None,
    ):
        """
        Initialize the CollisionPipeline (expert API).

        ``convex_collision`` is an experimental per-instance implementation choice.
        ``auto`` chooses ordinary queries or rejection screening using
        independently calibrated costs; unsupported pipelines use ordinary collision. Explicit
        ``ordinary`` and ``rejection`` force a supported method, or raise ValueError.
        These choices override the legacy coherent-convex environment variable. ``None`` preserves
        that environment setting for existing callers.
        Inspect :meth:`get_collision_selection` outside graph capture. This
        experimental parameter may change without the normal deprecation period.

        Automatic CUDA selection samples support work during graph replay and
        prices it with independent hardware primitives. Deterministic, speculative and
        alternate terrain query owners retain their original paths.

        Heightfield pairs with spheres, capsules, boxes, cylinders, cones,
        ellipsoids and convex hulls get their contacts from the terrain law by
        default: after the narrow phase, one contact per support the law finds
        in the pair's window of the grid (a face, edge or vertex within reach;
        the heightfield's border has no features in the law yet).
        Other partners, and the configurations the law does not support yet
        (speculative contacts, the ``NEWTON_TERRAIN_LAW_SHADOW`` comparison
        mode, a prebuilt narrow phase with its own contact writer), take the
        terrain route: per-cell triangle queries with contact reduction.
        ``NEWTON_TERRAIN_LAW=none`` set before construction keeps the terrain
        route for every heightfield pair, a temporary opt-out until that route
        is removed; a class list or ``all`` requests the law and raises on an
        unsupported configuration. ``get_collision_selection()["heightfield"]
        ["law"]`` records whether the law writes and why. The law builds its
        terrain features from the heights here: rebuild the pipeline after
        editing :attr:`~newton.Model.heightfield_elevations` in place.

        Experimental performance mode: setting the environment variable
        ``NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP=1`` before construction prepares
        geometry only for the fixed explicit pair list. In this opt-in mode,
        only shape IDs in ``pipeline.prepared_shape_indices`` have current
        entries in ``geom_data``, ``geom_transform``, and
        ``pipeline.narrow_phase.shape_aabb_lower/upper`` after ``collide``.
        Other entries are unspecified and must not be consumed. Shape IDs,
        pairs, and contact ordering are not remapped. The pair list must not
        change during the pipeline lifetime; rebuild after changing it.
        The exposed ``prepared_shape_indices`` array is read-only by contract
        and must not be modified or replaced during the pipeline lifetime.
        This mode requires internal explicit broad/narrow phases, no particles,
        no hydroelastic or speculative contacts, and no gradients. The default
        remains full preparation, including unpaired and visual-only shapes.

        Args:
            model: The simulation model.
            reduce_contacts: Whether to reduce contacts for mesh-mesh collisions. Defaults to True.
            body_pair_reduction: Reduce the written contacts per body pair, after
                the narrow phase. ``None`` (default) disables it. See
                :class:`CollisionPipeline.BodyPairReductionConfig`.
            rigid_contact_max: The rigid contact rows of the pipeline's contact buffers: a fixed
                capacity, past which a pass drops contacts and flags it, unless ``growable_contacts``
                makes it the rows they start from. Resolved as this value, else
                ``model.rigid_contact_max`` when it is not ``None`` (including zero), else an
                estimate from the model's shape pairs, and published to ``model.rigid_contact_max``
                unless ``publish_capacity`` is ``False``.
            max_triangle_pairs:
                Maximum number of triangle pairs allocated by narrow phase
                for mesh and heightfield collisions; also the capacity of the
                global contact reducer shared by all worlds.  Increase this when
                scenes with large/complex meshes or heightfields report
                triangle-pair or contact-reduction overflow warnings.
                If None (the default), estimate it from the model: each world
                budgets 128 candidates per candidate mesh, heightfield, or
                planar-SDF pair, limited in aggregate to 6 neighbors per
                participating shape, and the per-world budgets are summed.
                Explicit mode uses its configured pair list. NXN and SAP
                count pairs per world from shape masks, which can overcount,
                even when ``model.shape_contact_pairs`` has already been built.
                The estimate never falls below 1,000,000 and warns when its
                buffers require at least 256 MiB. With deterministic contact
                reduction (``deterministic=True`` or contact matching) it is
                limited to ``2**20 - 1`` and warns when that drops part of the
                estimate. It is a heuristic that assumes small contact patches,
                so dense meshes under convex shapes may still need an explicit
                value. Ignored when ``narrow_phase`` is provided.
            contact_reduction_hashtable_size_factor: Multiplier applied to
                ``max_triangle_pairs`` when allocating the global contact
                reduction hashtable. Increase this if hashtable fill/failure
                warnings appear. This sizes only the mesh/heightfield producer
                stage; body-pair sizing lives in
                ``BodyPairReductionConfig.hashtable_size_factor``.
                Defaults to ``0.25`` for memory compatibility.
            soft_contact_max: Maximum number of soft contacts to allocate.
                If None, defaults to ``soft_contact_pair_count``, the number
                of precomputed soft-rigid (particle-shape) pairs launched for soft
                contact generation, plus the full-surface edge/face headroom when
                ``enable_rigid_soft_full_surface_contact`` is set.
                Full-surface mesh storage is estimated; dense patches may require
                a larger explicit capacity to avoid incomplete contacts on overflow.
            soft_contact_gap: Detection-only distance [m] added to the
                per-particle radius for particle-shape (soft) contact queries.
                Defaults to 0.01.
            soft_contact_margin: Deprecated alias of ``soft_contact_gap`` (the
                value is detection-only slack on top of the particle radius,
                i.e. a gap under the margin/gap convention).
            enable_rigid_soft_full_surface_contact: Generate soft contacts over the full soft-mesh
                surface -- the edges and triangle interiors -- against rigid surfaces, in addition to
                the per-vertex (particle) contacts. Catches rigid features that pass between soft
                vertices (e.g. a thin box edge or heightfield cell inside a coarse cloth triangle),
                which the per-particle path misses. Meshes use exact feature contacts without
                requiring a volume SDF and report every vertex-face, face-vertex, and edge-edge
                pair within the contact band, including redundant representations of one surface
                patch, inactive soft features, and pairs with unresolved mesh sign. All rigid
                triangle edges participate, including edges omitted from SDF preprocessing.
                The consuming solver selects the pairs it applies forces to. Only meshes
                with :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES` set at construction get them,
                others keep per-particle contact. These full-surface contacts are consumed only by
                :class:`~newton.solvers.SolverVBD`; other solvers raise on such contacts. Records are
                emitted into :attr:`Contacts.soft_contact_indices`. Defaults to False. Fixed at
                construction because it sizes the soft-contact buffer headroom.
                Fine meshes and large contact bands can generate many more records than
                particle-only contacts or SDF minimization, increasing contact storage and
                collision/solver time. Set ``soft_contact_max`` to size storage explicitly;
                this does not reduce the number of feature pairs queried.
                See :ref:`full-surface-mesh-contact-costs`.
            requires_grad: Whether pipeline-generated soft contacts and the
                deprecated automatic rigid-contact outputs require gradients.
                If None, uses ``model.requires_grad``. Explicit calls to
                :func:`newton.eval_rigid_contact_kinematics` do not
                depend on this flag.
            broad_phase:
                Either a broad phase mode string ("explicit", "nxn", "sap") or
                a prebuilt broad phase instance for expert usage.
            box_box_sat: Route box-box pairs through the SAT reference-face
                clipping primitive instead of GJK/MPR, with a feature-identity-reduced
                4-slot manifold. Stable multi-point box manifolds (no witness-
                point teleports). Cannot be combined with a prebuilt
                ``narrow_phase``. Defaults to False.
            convex_collision: Experimental per-instance convex query implementation. ``auto`` chooses
                ordinary queries or rejection screening; explicit choices
                force a supported method. ``None`` retains the legacy environment selection.
                Supported bulk CUDA terrain queries use staged MPR/GJK publication when
                their scratch fits the memory budget. Small models retain fused queries.
            contact_export: Experimental reduced-contact export mapping. ``auto`` selects compact
                workers for supported CUDA mesh/heightfield reducers. ``tiled`` forces the original
                mapping; ``compact`` requires admission. Automatic compact export excludes
                deterministic, speculative, hydroelastic and alternate heightfield export modes.
                Explicit choices override ``NEWTON_REDUCER_COMPACT_EXPORT``; ``None`` preserves it.
                Cannot be combined with a prebuilt narrow phase; configure that instance directly.
                This parameter may change without the normal deprecation period.
            narrow_phase: Optional prebuilt narrow phase instance. Must be
                provided together with a broad phase instance for expert usage.
                If its voxel-resolution table is omitted, use the model's table.
                A supplied table must match the model's shape count and device.
                Its effective ``reduce_contacts`` state is authoritative for
                the mesh/heightfield producer stage; the pipeline exposes that
                state through ``pipeline.mesh_contact_reduction_enabled``.
            shape_pairs_filtered: Precomputed shape pairs for EXPLICIT mode.
                When broad_phase is "explicit", uses model.shape_contact_pairs if not provided. For
                "nxn"/"sap" modes, ignored. The pair count and shape-type routing are used to size
                and specialize internal buffers at construction, so do not modify or resize the
                array while the pipeline is in use. Rebuild the pipeline after changing the pairs.
                When the pairs are the ``model.shape_contact_pairs`` that :meth:`ModelBuilder.finalize`
                replayed from per-world templates, and the templates are large, the broad phase tests
                each world's template instead of streaming the list; the candidate pairs are the same
                set. Set ``NEWTON_BROAD_PHASE_TEMPLATES=0`` before construction to keep the flat
                kernel, or ``1`` to test templates of any size.
            include_static_kinematic_pairs: Whether to generate contacts for static-kinematic and
                kinematic-kinematic pairs. Set to ``False`` to filter those pairs. Static-static pairs are
                always filtered. Defaults to ``True`` for backward compatibility. With ``True``, pairs of
                two structurally immobile shapes follow ``include_immobile_pairs``. While the broad phase
                drops immobile pairs, a body welded to the world counts as static here, so with ``False``
                its pairs with static, welded and kinematic shapes are dropped as well.
            include_immobile_pairs: Whether to generate contacts for pairs whose
                shapes are both structurally immobile. A shape is structurally
                immobile when it is static or belongs to a body welded to the
                world: a dynamic body whose every parent joint is an enabled
                joint without degrees of freedom, from the world or from another
                welded body. The joint's degree-of-freedom count decides, not its
                type: a fixed joint welds, and so does a D6 joint without axes,
                which the USD importer makes of a generic root joint. Neither
                shape of such a pair can move, so its contacts have no dynamics
                in a solver that gives welded bodies no degrees of freedom.
                Kinematic bodies can move, so they and the bodies welded to them
                are never immobile. ``None`` (default) follows the
                ``NEWTON_BROAD_PHASE_IMMOBILE_FILTER`` environment variable:
                unset or ``1`` drops these pairs in the broad phase, ``0`` keeps
                them. ``False`` drops them. ``True`` keeps them, as before: for
                solvers that resolve such joints as constraints and respond to
                such contacts, and for contact sensors that should report them.
                The immobile set is computed at construction; code that changes
                :attr:`~newton.Model.body_flags` or
                :attr:`~newton.Model.joint_enabled` at runtime must also call
                :meth:`notify_model_changed`, or a body released from its
                welding joint keeps no contacts with static shapes.
            sdf_hydroelastic_config: Configuration for hydroelastic collision
                handling. Defaults to None.
            shape_pairs_max: Override for the broad-phase candidate-pair
                buffer capacity used by the ``"nxn"`` and ``"sap"`` modes.
                Defaults to the worst-case ``N*(N-1)/2`` per-world bound,
                which is rarely hit by either ``"nxn"`` or ``"sap"`` in
                practice -- ``"nxn"`` still applies AABB overlap, group,
                and excluded-pair filtering inside ``BroadPhaseAllPairs``
                before writing, and ``"sap"`` is sparse by design -- so
                the default sizing is typically 10-100x larger than what
                gets emitted on real scenes. Set this to a tighter value
                (e.g. measured peak with ~25% headroom) to avoid multi-GB
                allocations on large scenes; a too-small value triggers
                a buffer overflow warning at runtime. Ignored for the
                ``"explicit"`` mode and for expert paths that pass a
                pre-built ``narrow_phase``.
            broad_phase_output_max: Optional positive integer limiting materialized
                broad-phase output pairs and dependent narrow-phase candidate
                buffers for internally constructed ``"explicit"`` pipelines.
                The entire ``shape_pairs_filtered`` input is still traversed.
                ``None`` retains the full input-list capacity; larger requests
                are capped to that list's length. Unsupported for ``"nxn"``,
                ``"sap"``, or prebuilt broad/narrow phases. This is a capacity
                bound, not a collision filter: undersizing loses pairs and
                ``verify_buffers=True`` reports the unclamped current-frame
                demand. Counters reset each collision call, so sample or record
                every call when calibrating captured runs. Rebuild the pipeline
                and recapture graphs to change this bound; an overflow invalidates
                the affected collision results.
            deterministic: Sort contacts after the narrow phase so that results
                are independent of GPU thread scheduling. This also enables
                deterministic hydroelastic accumulation and contact allocation.
                Adds a radix sort + gather pass.
            contact_matching: Frame-to-frame contact matching mode.  One of
                ``"disabled"``, ``"latest"``, or ``"sticky"``.  Any
                non-disabled mode implies ``deterministic=True`` and
                populates :attr:`Contacts.rigid_contact_match_index`.
                Defaults to ``"disabled"``.

                .. experimental::

                    The ``"sticky"`` mode may change without prior notice.
            contact_matching_pos_threshold: World-space distance threshold [m]
                between the previous and current contact midpoints
                ``0.5 * (world(point0) + world(point1))``.  Contacts whose
                midpoint moves more than this are considered broken. In sticky mode,
                keep fresh geometry when either saved witness moves farther than this
                from its fresh contact point under the current body transforms.
                Defaults to ``0.0005``.
            contact_matching_normal_dot_threshold: Minimum dot product between
                old and new contact normals for a match.
            contact_report: Allocate ``rigid_contact_new_indices`` /
                ``rigid_contact_new_count`` / ``rigid_contact_broken_indices``
                / ``rigid_contact_broken_count`` on the :class:`Contacts`
                container, populated each frame.  Requires a non-disabled
                ``contact_matching`` mode.
            verify_buffers: Run a ``dim=[1]`` diagnostic kernel at the end of
                the narrow phase that prints warnings on any intermediate
                candidate-pair or final rigid contact buffer overflow; see
                :class:`NarrowPhase` for the full counter list.  Defaults to
                ``True``.  Overhead is one extra kernel launch per collision
                pass; disable in hot loops or CUDA graph capture once buffer
                sizes are known to be adequate.
            speculative_contact_gap_max: Cap on the velocity-derived rigid-contact
                detection gap [m]. The effective gap is the larger of the authored
                gap and the capped velocity-derived gap. Must be a non-negative
                finite number or ``None``. ``None`` disables speculative contacts;
                ``0.0`` enables them without enlarging authored gaps. Defaults to
                ``None``. See
                :ref:`Speculative contacts <speculative-contacts>`.
            dynamic_shape_scale: Query texture SDFs built without a baked scale with the current
                :attr:`Model.shape_scale <newton.Model.shape_scale>` on every call, which is slower.
                Enable this when such a shape's scale changes after construction. When ``False`` and
                every such shape has unit scale at construction, these queries are compiled for unit
                scale; routing a rescaled shape through mesh-SDF collision makes the device print a
                warning once and a later :meth:`collide` call outside graph capture raise
                :class:`RuntimeError`. Mesh-plane and mesh-convex contacts are unaffected.
                This option does not update derived collision bounds. After rescaling meshes,
                convex meshes, or heightfields, also update :attr:`Model.shape_collision_radius`,
                :attr:`Model.shape_collision_aabb_lower`, and :attr:`Model.shape_collision_aabb_upper`
                to match the new scale, or broadphase culling can miss contacts. It does not rebuild
                baked SDFs or heightfield collision data. Recreate affected shapes at the new scale
                on a :class:`ModelBuilder`, then finalize a new model and construct a new collision pipeline.
                For explicitly prebuilt mesh SDFs, first rerun :meth:`Mesh.build_sdf` with the new scale.
                Ignored when ``narrow_phase`` is provided. Defaults to ``False``.
            publish_capacity: Whether to publish the resolved capacities as the model's
                default :attr:`~newton.Model.rigid_contact_max` and
                :attr:`~newton.Model.soft_contact_max`. Pass ``False`` for a private
                pipeline, such as a solver's internal one. Defaults to ``True``.
            growable_contacts: Whether the rigid contact buffers grow from demand, for a solver whose
                per-contact state follows them (:class:`~newton.solvers.SolverFeatherPGS`). With
                ``False`` (the default) the pipeline's buffers, its :meth:`contacts` among them, hold
                the resolved ``rigid_contact_max`` rows from construction, of their own, and never
                grow. With ``True`` they share the model's contact pool with the other growable
                pipelines of the model and the solvers that follow it, start at a few thousand rows
                (fewer when ``rigid_contact_max`` is smaller), grow before anything reads them in a
                :meth:`collide` outside graph capture and between graph replays in
                :meth:`service_capacity`, and trim once the demand settles.
            terrain_meshes: Whether the terrain law also reads static triangle meshes as surfaces, each one whose
                structure qualifies (:class:`~newton._src.geometry.terrain_contacts.TerrainSurfaces`: today a height
                sheet on a lattice, a heightfield in triangle form), and writes its partners' contacts with them as it
                writes their heightfield contacts. For a solver that takes the law's contacts
                (:class:`~newton.solvers.SolverFeatherPGS`). With ``False`` (the default) every mesh keeps the mesh
                path, as other solvers expect.

        .. experimental::

            Rigid-contact autodiff via
            :func:`newton.eval_rigid_contact_kinematics` may change
            without prior notice; see :meth:`collide`.

            ``NEWTON_NARROW_PHASE_COHERENT_CONVEX=reject_only`` enables an
            experimental, default-off temporal rejection filter. It requires
            immutable model-built unique explicit pairs and the internal lean
            CUDA split pipeline, without gradients, speculative contacts,
            particles, triangle meshes, heightfields, hydroelastic contacts,
            or body-pair reduction. Cached BOX/CONVEX_MESH directions are only
            hints: current full-shape support must certify separation beyond
            the original contact shell. Every retained or ambiguous pair uses
            the original cold MPR/GJK and unchanged manifold writer.

            Serialize collisions and resets, including stream handoffs and
            graph replay. Only one live CUDA graph may own this cache. Rebuild
            after changing pair membership or shape types; transforms, scale
            and hull vertices are reread for each certificate. Resetting
            contact matching invalidates the selected worlds' hints (a global
            reset invalidates all buckets). An unreported reset cannot reuse
            a cached witness as a contact answer. Optional diagnostic counters
            use ``NEWTON_NARROW_PHASE_COHERENT_STATS=1``; keep them off for timing.
        """
        if contact_matching not in ("disabled", "latest", "sticky"):
            raise ValueError(
                f"contact_matching must be one of 'disabled', 'latest', 'sticky', got {contact_matching!r}"
            )
        if contact_matching_pos_threshold < 0.0:
            raise ValueError(
                f"contact_matching_pos_threshold must be non-negative, got {contact_matching_pos_threshold}"
            )
        if not -1.0 <= contact_matching_normal_dot_threshold <= 1.0:
            raise ValueError(
                f"contact_matching_normal_dot_threshold must be in [-1, 1], got {contact_matching_normal_dot_threshold}"
            )
        matching_enabled = contact_matching != "disabled"
        matching_sticky = contact_matching == "sticky"
        if contact_report and not matching_enabled:
            raise ValueError('contact_report=True requires contact_matching != "disabled"')
        if body_pair_reduction is not None and not isinstance(body_pair_reduction, self.BodyPairReductionConfig):
            raise TypeError(
                "body_pair_reduction must be CollisionPipeline.BodyPairReductionConfig or None, "
                f"got {type(body_pair_reduction).__name__}"
            )
        if body_pair_reduction is not None and matching_sticky:
            raise ValueError('body_pair_reduction requires contact_matching="latest" or "disabled"')
        if speculative_contact_gap_max is not None and (
            not np.isfinite(speculative_contact_gap_max) or speculative_contact_gap_max < 0.0
        ):
            raise ValueError(
                "speculative_contact_gap_max must be a non-negative finite number or None, "
                f"got {speculative_contact_gap_max!r}"
            )

        # Any non-disabled matching mode implies deterministic sorting.
        if matching_enabled:
            deterministic = True

        mode_from_broad_phase: str | None = None
        broad_phase_instance: BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit | None = None
        if broad_phase is not None:
            if isinstance(broad_phase, str):
                mode_from_broad_phase = _normalize_broad_phase_mode(broad_phase)
            else:
                broad_phase_instance = broad_phase

        shape_count = model.shape_count
        # A variant row can take a surface-velocity mesh after construction.
        mesh_properties = (
            getattr(model, "_shape_mesh_properties", None),
            getattr(model, "_variant_shape_geometry", {}).get("_shape_mesh_properties"),
        )
        self._rigid_contact_surface_velocity = any(
            properties is not None and np.any(properties.numpy() & int(MeshProperties.SURFACE_VELOCITY))
            for properties in mesh_properties
        )
        self._contact_sort_shape_index_bits = contact_sort_shape_index_bits(shape_count)
        device = model.device
        using_expert_components = broad_phase_instance is not None or narrow_phase is not None
        if contact_export not in (None, "auto", "tiled", "compact"):
            raise ValueError(f"contact_export must be 'auto', 'tiled', 'compact' or None, got {contact_export!r}")
        if contact_export is not None and narrow_phase is not None:
            raise ValueError("contact_export cannot be used with a prebuilt narrow_phase; configure that instance")
        if broad_phase_output_max is not None:
            if (
                isinstance(broad_phase_output_max, (bool, np.bool_))
                or not isinstance(broad_phase_output_max, (int, np.integer))
                or broad_phase_output_max <= 0
            ):
                raise ValueError(
                    f"broad_phase_output_max must be a positive integer or None, got {broad_phase_output_max!r}"
                )
            if using_expert_components or mode_from_broad_phase not in (None, "explicit"):
                raise ValueError("broad_phase_output_max requires internal explicit broad/narrow phases")
        self.broad_phase_output_max = None if broad_phase_output_max is None else int(broad_phase_output_max)
        # Record the NEWTON_* environment once, before the reads below; undeclared names
        # are recorded, not rejected, names only Isaac Lab reads are kept apart, and
        # retired names select nothing and warn once per process.
        self._environment_record = record_newton_environment()
        retired_warning = retired_notice(self._environment_record)
        if retired_warning is not None:
            warnings.warn(retired_warning, UserWarning, stacklevel=2)
        pair_shape_prep = os.environ.get("NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP", "0")
        if pair_shape_prep not in ("0", "1"):
            raise ValueError("NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP must be 0 or 1")
        self._pair_shape_prep = pair_shape_prep == "1"
        self.prepared_shape_indices = None
        pair_templates = os.environ.get("NEWTON_BROAD_PHASE_TEMPLATES")
        if pair_templates not in (None, "0", "1"):
            raise ValueError("NEWTON_BROAD_PHASE_TEMPLATES must be 0 or 1")
        coherent_convex = os.environ.get("NEWTON_NARROW_PHASE_COHERENT_CONVEX", "0")
        if convex_collision is not None:
            methods = {
                "auto": "auto",
                "ordinary": "0",
                "rejection": "reject_only",
            }
            if convex_collision not in methods:
                raise ValueError(f"convex_collision must be one of {tuple(methods)}, got {convex_collision!r}")
            coherent_convex = methods[convex_collision]
        self.convex_collision = convex_collision
        # The coherent-convex owner requested by convex_collision or the environment, before admission.
        self._coherent_convex_requested = coherent_convex
        coherent_stats = os.environ.get("NEWTON_NARROW_PHASE_COHERENT_STATS", "0")
        if coherent_convex not in ("0", "reject_only", "auto") or coherent_stats not in ("0", "1"):
            raise ValueError(
                "NEWTON_NARROW_PHASE_COHERENT_CONVEX must be 0 or reject_only or auto; COHERENT_STATS must be 0 or 1"
            )
        if coherent_stats == "1" and coherent_convex == "0":
            raise ValueError("coherent convex diagnostics require NEWTON_NARROW_PHASE_COHERENT_CONVEX=reject_only")
        # Who writes heightfield contacts (NEWTON_TERRAIN_LAW): by default the terrain law, for every partner class it
        # has a kernel for, where the configuration is one it supports (else the terrain route); "none" keeps the
        # terrain route (a temporary opt-out); a class list or "all" requests the law strictly. NEWTON_TERRAIN_LAW_SHADOW
        # instead runs the law beside the terrain route, which writes every contact (comparison mode).
        terrain_law_value = os.environ.get("NEWTON_TERRAIN_LAW", "")
        terrain_law_shadow = os.environ.get("NEWTON_TERRAIN_LAW_SHADOW", "")
        terrain_law_request, terrain_law_classes, terrain_shadow_classes = "default", (), ()
        if terrain_law_value.strip() or terrain_law_shadow.strip():
            from ..geometry.terrain_shadow import parse_terrain_law, parse_terrain_law_shadow  # noqa: PLC0415

            terrain_law_request, terrain_law_classes = parse_terrain_law(terrain_law_value)
            if terrain_law_request == "explicit" and terrain_law_shadow.strip():
                raise ValueError(
                    "NEWTON_TERRAIN_LAW and NEWTON_TERRAIN_LAW_SHADOW are both set; the law either writes its partners' "
                    "contacts or runs in shadow: unset one (unset NEWTON_TERRAIN_LAW to compare the shadow with the "
                    "terrain route)"
                )
            terrain_shadow_classes = parse_terrain_law_shadow(terrain_law_shadow)
        self._terrain_law_shadow = None
        """The terrain law (``TerrainLawShadow``), built when its kernels run: writing contacts or in shadow."""
        self._terrain_law_shadow_reason = None
        self._terrain_law_writes = False
        """Whether the law writes the contacts of its partners' heightfield pairs (the default; ``NEWTON_TERRAIN_LAW``)."""
        self._terrain_law_decision = None
        """Why the law writes or not: ``default``, ``explicit``, ``opted_out``, ``unsupported`` or ``no_partner``."""
        self._terrain_law_reason = None
        self._terrain_law_listed = None

        # Resolve rigid contact capacity with explicit > model > estimated precedence: the rows of fixed buffers,
        # or those growable buffers start from.
        estimate = None
        if rigid_contact_max is None:
            model_rigid_contact_max = model.rigid_contact_max
            if model_rigid_contact_max is not None:
                rigid_contact_max = model_rigid_contact_max
            else:
                rigid_contact_estimate = _estimate_rigid_contact_details(model)
                rigid_contact_max = estimate = rigid_contact_estimate.capacity
                if not growable_contacts:
                    _warn_large_rigid_contact_estimate(rigid_contact_estimate)
        self._rigid_contact_max = rigid_contact_max
        if soft_contact_margin is not None:
            if soft_contact_gap is not None:
                raise ValueError("soft_contact_margin is a deprecated alias of soft_contact_gap; pass only one")
            warnings.warn(
                "The soft_contact_margin parameter of CollisionPipeline is deprecated; "
                "use soft_contact_gap (same value: detection-only distance added to the particle radius).",
                NewtonDeprecationWarning,
                stacklevel=2,
            )
            soft_contact_gap = soft_contact_margin
        elif soft_contact_gap is None:
            soft_contact_gap = 0.01
        if soft_contact_gap < 0.0:
            raise ValueError(f"soft_contact_gap must be >= 0, got {soft_contact_gap}")

        if max_triangle_pairs is not None and max_triangle_pairs <= 0:
            raise ValueError("max_triangle_pairs must be > 0 or None")
        if requires_grad is None:
            requires_grad = model.requires_grad
        # Every per-contact buffer of this pipeline and its Contacts holds the pool's ready rows: a fixed buffer of
        # the capacity of the pipeline's own, or the model's pool, which the solvers that follow it share and which
        # grows from measured and forecast demand.
        if growable_contacts:
            self._contact_pool = model_contact_pool(model, device, rigid_contact_max)
        else:
            self._contact_pool = ContactPool(
                device=device, rows=rigid_contact_max, fixed=True, worlds=model.world_count
            )
        # The pool's generation the last service_capacity() reported.
        self._service_generation = self._contact_pool.generation
        # The writers' row limit for a buffer whose pool was created during a capture (see _detect_rigid_contacts).
        self._contact_limit = wp.zeros(1, dtype=wp.int32, device=device)
        # The narrow phase's capacity flags before a pass outside capture, restored if the pass reruns.
        self._capacity_status_before_pass = wp.zeros(1, dtype=wp.int32, device=device)

        shape_world = getattr(model, "shape_world", None)
        shape_flags = getattr(model, "shape_flags", None)
        # Variant rows can hold variant shapes with other types and flags; routing and capacities cover all.
        variant_options = _variant_options(model)
        if variant_options is not None:
            shape_flags = model._shape_option_flags()
        with wp.ScopedDevice(device):
            shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)

        self.model = model
        self.shape_count = shape_count
        self.device = device
        self.reduce_contacts = reduce_contacts
        self._body_pair_reduction = body_pair_reduction
        self.requires_grad = requires_grad
        self.include_static_kinematic_pairs = include_static_kinematic_pairs
        if include_immobile_pairs is None:
            immobile_filter = os.environ.get("NEWTON_BROAD_PHASE_IMMOBILE_FILTER")
            if immobile_filter not in (None, "0", "1"):
                raise ValueError("NEWTON_BROAD_PHASE_IMMOBILE_FILTER must be 0 or 1")
            immobile_source = "default" if immobile_filter is None else "environment"
            include_immobile_pairs = immobile_filter == "0"
        else:
            immobile_source = "argument"
        self.speculative_contact_gap_max = speculative_contact_gap_max
        self._speculative_enabled = speculative_contact_gap_max is not None
        contact_writer = write_contact_speculative if self._speculative_enabled else write_contact

        unit_scale_sdf_shapes = None
        # Host copy of the explicit pair list, read at most once during setup.
        explicit_pairs_host = None
        if using_expert_components:
            if broad_phase_instance is None or narrow_phase is None:
                raise ValueError("Provide both broad_phase and narrow_phase for expert component construction")
            if body_pair_reduction is not None:
                raise ValueError("body_pair_reduction requires the pipeline to build its own narrow phase")
            if sdf_hydroelastic_config is not None:
                raise ValueError("sdf_hydroelastic_config cannot be used when narrow_phase is provided")
            if box_box_sat:
                raise ValueError(
                    "box_box_sat cannot be used when narrow_phase is provided; "
                    "construct the NarrowPhase with box_box_sat=True instead"
                )
            if contact_reduction_hashtable_size_factor != 0.25:
                raise ValueError(
                    "contact_reduction_hashtable_size_factor cannot be used when narrow_phase is provided; "
                    "construct the NarrowPhase with that value instead"
                )
            inferred_mode = _infer_broad_phase_mode_from_instance(broad_phase_instance)
            self.broad_phase_mode = inferred_mode
            self.broad_phase = broad_phase_instance

            if self.broad_phase_mode == "explicit":
                if shape_pairs_filtered is None:
                    shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)
                if shape_pairs_filtered is None:
                    raise ValueError(
                        "shape_pairs_filtered must be provided for explicit broad phase "
                        "(or set model.shape_contact_pairs)"
                    )
                self.shape_pairs_filtered = shape_pairs_filtered
                self.shape_pairs_max = len(shape_pairs_filtered)
                self.shape_pairs_excluded = None
                self.shape_pairs_excluded_count = 0
                self.broad_phase._warn_large_pair_count(shape_pairs_filtered, shape_world, model.world_count)
            else:
                self.shape_pairs_filtered = None
                self.shape_pairs_max = _compute_per_world_shape_pairs_max(model)
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )

            if deterministic and not narrow_phase.deterministic:
                raise ValueError(
                    "CollisionPipeline(deterministic=True) requires a deterministic "
                    "NarrowPhase. Either omit narrow_phase or construct it with "
                    "deterministic=True."
                )
            if bool(getattr(narrow_phase, "speculative", False)) != self._speculative_enabled:
                raise ValueError(
                    "Provided narrow_phase speculative mode must match "
                    "CollisionPipeline(speculative_contact_gap_max=...)."
                )
            if narrow_phase.max_candidate_pairs < self.shape_pairs_max:
                raise ValueError(
                    "Provided narrow_phase.max_candidate_pairs is too small for this model and broad phase mode "
                    f"(required at least {self.shape_pairs_max}, got {narrow_phase.max_candidate_pairs})"
                )
            self.narrow_phase = narrow_phase
            self.hydroelastic_sdf = self.narrow_phase.hydroelastic_sdf
        else:
            self.broad_phase_mode = mode_from_broad_phase if mode_from_broad_phase is not None else "explicit"

            if self.broad_phase_mode == "explicit":
                if shape_pairs_filtered is None:
                    shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)
                if shape_pairs_filtered is None:
                    raise ValueError(
                        "shape_pairs_filtered must be provided for broad_phase=EXPLICIT "
                        "(or set model.shape_contact_pairs)"
                    )
                self.broad_phase = BroadPhaseExplicit()
                self.shape_pairs_filtered = shape_pairs_filtered
                self.shape_pairs_max = len(shape_pairs_filtered)
                if self.broad_phase_output_max is not None:
                    self.shape_pairs_max = min(self.shape_pairs_max, self.broad_phase_output_max)
                self.shape_pairs_excluded = None
                self.shape_pairs_excluded_count = 0
                explicit_pairs_host = shape_pairs_filtered.numpy().reshape(-1, 2)
                self.broad_phase._warn_large_pair_count(explicit_pairs_host, shape_world, model.world_count)
            elif self.broad_phase_mode == "nxn":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=NXN")
                self.broad_phase = BroadPhaseAllPairs(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = _resolve_shape_pairs_max(model, shape_pairs_max)
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            elif self.broad_phase_mode == "sap":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=SAP")
                self.broad_phase = BroadPhaseSAP(shape_world, shape_flags=shape_flags, sort_type="auto", device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = _resolve_shape_pairs_max(model, shape_pairs_max)
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            else:
                raise ValueError(f"Unsupported broad phase mode: {self.broad_phase_mode}")

            if self._speculative_enabled:
                shape_flags_np = model.shape_flags.numpy()
                is_hydroelastic = (shape_flags_np & int(ShapeFlags.HYDROELASTIC)) != 0
                if np.count_nonzero(is_hydroelastic) > 1 and _shape_contact_pair_count_for_mask(
                    model, is_hydroelastic, shape_pairs=self.shape_pairs_filtered
                ):
                    raise NotImplementedError(
                        "Speculative contact generation does not yet support hydroelastic SDF contacts"
                    )

            # Initialize SDF hydroelastic (returns None if no hydroelastic shape pairs in the model)
            hydroelastic_sdf = HydroelasticSDF._from_model(
                model,
                config=sdf_hydroelastic_config,
                writer_func=contact_writer,
                deterministic=deterministic,
                shape_pairs_filtered=self.shape_pairs_filtered,
            )

            # Detect shape classes to optimize narrow-phase kernel launches.
            # Keep mesh and heightfield flags independent: heightfield-only scenes
            # should not trigger mesh-only kernel setup/launches.
            has_meshes = False
            has_heightfields = model.heightfield_count > 0
            use_lean_gjk_mpr = False
            mesh_sdf_texture_only = False
            mesh_sdf_identity_scale_only = False
            max_mesh_mesh_pairs = self.shape_pairs_max
            max_mesh_plane_pairs = self.shape_pairs_max
            max_heightfield_pairs = None
            if hasattr(model, "shape_type") and model.shape_type is not None:
                shape_types = model.shape_type.numpy()
                shape_count_routed = len(shape_types)
                route_mask = _shape_collide_mask(model, shape_count_routed)
                # Shapes that can reach the narrow phase decide which stages it builds.
                # NXN/SAP pair only shapes that collide with shapes. The explicit list is
                # authoritative: every listed shape reaches the narrow phase, including
                # one with shape collision disabled, and an unlisted shape never does.
                if self.broad_phase_mode == "explicit" and self.shape_pairs_filtered is not None:
                    route_mask = np.zeros(len(shape_types), dtype=bool)
                    pair_shapes = explicit_pairs_host.ravel()
                    route_mask[pair_shapes[(pair_shapes >= 0) & (pair_shapes < len(shape_types))]] = True
                # A variant row can take the geometry of any variant shape it can hold at runtime, so
                # the stages cover every variant shape a routed row can hold, appended as extra
                # rows, and capacities count each row as any of them.
                routed_variant_shapes = np.zeros(0, dtype=np.int64)
                if variant_options is not None:
                    reachable = variant_options.reachable(route_mask)
                    if self.broad_phase_mode != "explicit" or self.shape_pairs_filtered is None:
                        reachable &= variant_options.variant_collides
                    routed_variant_shapes = np.flatnonzero(reachable)

                def with_variant_shapes(name: str, shape_values: np.ndarray) -> np.ndarray:
                    """Per-shape values followed by those of the reachable variant shapes."""
                    if variant_options is None:
                        return shape_values
                    variant_values = model._variant_shape_geometry[name].numpy()[routed_variant_shapes]
                    return np.concatenate((shape_values[:shape_count_routed], variant_values))

                def capacity(mask: np.ndarray) -> np.ndarray:
                    """Per shape, whether it or a variant shape it can hold as a routed variant row is in ``mask``."""
                    if variant_options is None:
                        return mask
                    variant_mask = np.zeros(len(variant_options.variant_types), dtype=bool)
                    variant_mask[routed_variant_shapes] = mask[shape_count_routed:]
                    return mask[:shape_count_routed] | (route_mask & variant_options.any(variant_mask))

                route_types = with_variant_shapes("shape_type", shape_types)
                route_rows = np.concatenate((route_mask, np.ones(len(routed_variant_shapes), dtype=bool)))
                route_shape_types = route_types[route_rows]
                mesh_mask = route_rows & (route_types == int(GeoType.MESH))
                heightfield_mask = route_rows & (route_types == int(GeoType.HFIELD))
                plane_mask = route_rows & (route_types == int(GeoType.PLANE))
                mesh_sdf_pair_mask = mesh_mask | heightfield_mask
                planar_sdf_mask = np.zeros(len(route_types), dtype=bool)
                has_meshes = bool(np.any(mesh_mask))
                has_heightfields = bool(np.any(heightfield_mask))
                if (
                    hasattr(model, "_shape_sdf_index")
                    and model._shape_sdf_index is not None
                    and hasattr(model, "_shape_edge_range")
                    and model._shape_edge_range is not None
                ):
                    shape_sdf_index = with_variant_shapes("_shape_sdf_index", model._shape_sdf_index.numpy())
                    shape_edge_range = with_variant_shapes("_shape_edge_range", model._shape_edge_range.numpy())
                    planar_sdf_mask = route_rows & (shape_sdf_index >= 0) & (shape_edge_range[:, 1] > 0)
                    has_planar_sdf_shapes = bool(np.any(planar_sdf_mask))
                    has_meshes = has_meshes or has_planar_sdf_shapes
                    mesh_sdf_pair_mask |= planar_sdf_mask
                    mesh_sdf_shapes = route_rows & (
                        (route_types != int(GeoType.HFIELD))
                        & ((route_types == int(GeoType.MESH)) | (shape_edge_range[:, 1] > 0))
                    )
                    coarse_textures = getattr(model, "_texture_sdf_coarse_textures", None)
                    if coarse_textures is not None and len(coarse_textures) > 0:
                        has_texture_sdf = np.array(
                            [
                                sdf_idx >= 0 and sdf_idx < len(coarse_textures) and coarse_textures[sdf_idx] is not None
                                for sdf_idx in shape_sdf_index
                            ],
                            dtype=bool,
                        )
                        mesh_sdf_texture_only = bool(
                            np.any(mesh_sdf_shapes) and np.all(has_texture_sdf[mesh_sdf_shapes])
                        )
                    if mesh_sdf_texture_only:
                        texture_sdf_data = model._texture_sdf_data.numpy()
                        scale_baked = texture_sdf_data["scale_baked"]
                        mesh_sdf_shape_indices = np.flatnonzero(mesh_sdf_shapes)
                        runtime_scale_sdf_shapes = mesh_sdf_shape_indices[
                            ~scale_baked[shape_sdf_index[mesh_sdf_shape_indices]].astype(bool)
                        ]
                        if dynamic_shape_scale:
                            mesh_sdf_identity_scale_only = runtime_scale_sdf_shapes.size == 0
                        else:
                            shape_scale = with_variant_shapes("shape_scale", model.shape_scale.numpy())
                            mesh_sdf_identity_scale_only = bool(
                                np.all(shape_scale[runtime_scale_sdf_shapes] == np.float32(1.0))
                            )
                            # Variant rows change scale with their variant, which the unit-scale check would
                            # report as a rescale.
                            if variant_options is not None:
                                on_shapes = runtime_scale_sdf_shapes[runtime_scale_sdf_shapes < shape_count_routed]
                                if on_shapes.size < runtime_scale_sdf_shapes.size or np.any(
                                    variant_options.row_mask[on_shapes]
                                ):
                                    mesh_sdf_identity_scale_only = False
                            if mesh_sdf_identity_scale_only and runtime_scale_sdf_shapes.size > 0:
                                unit_scale_sdf_shapes = runtime_scale_sdf_shapes
                # Per-shape capacity masks: a variant row counts as any variant shape it can hold.
                mesh_capacity = capacity(mesh_mask)
                heightfield_capacity = heightfield_mask[:shape_count_routed]
                plane_capacity = plane_mask[:shape_count_routed]
                planar_sdf_capacity = capacity(planar_sdf_mask)
                if self.broad_phase_mode == "explicit" and self.shape_pairs_filtered is not None:
                    # Explicit pair types are fixed at pipeline construction, including
                    # intentional cross-world pairs, so size only the stages they can reach.
                    # Both stages require a mesh or planar SDF, even in heightfield scenes.
                    if not has_meshes or self.shape_pairs_max == 0:
                        max_mesh_mesh_pairs = 0
                        max_mesh_plane_pairs = 0
                    else:
                        shape_a = explicit_pairs_host[:, 0]
                        shape_b = explicit_pairs_host[:, 1]
                        box_mask = route_mask & (shape_types == int(GeoType.BOX))
                        if variant_options is not None:
                            box_mask &= ~variant_options.row_mask
                        mesh_mesh_routes = (
                            (mesh_capacity[shape_a] & mesh_capacity[shape_b])
                            | (heightfield_capacity[shape_a] & mesh_capacity[shape_b])
                            | (mesh_capacity[shape_a] & heightfield_capacity[shape_b])
                            | (
                                planar_sdf_capacity[shape_a]
                                & planar_sdf_capacity[shape_b]
                                & ~(box_mask[shape_a] & box_mask[shape_b])
                            )
                        )
                        shape_scale = model.shape_scale.numpy()
                        infinite_plane_mask = plane_capacity & (shape_scale[:, 0] == 0.0) & (shape_scale[:, 1] == 0.0)
                        mesh_plane_routes = (mesh_capacity[shape_a] & infinite_plane_mask[shape_b]) | (
                            infinite_plane_mask[shape_a] & mesh_capacity[shape_b]
                        )
                        max_mesh_mesh_pairs = int(np.count_nonzero(mesh_mesh_routes))
                        max_mesh_plane_pairs = int(np.count_nonzero(mesh_plane_routes))
                else:
                    max_mesh_mesh_pairs = min(
                        self.shape_pairs_max,
                        _compute_per_world_mask_pair_max(model, capacity(mesh_sdf_pair_mask)),
                    )
                    max_mesh_plane_pairs = min(
                        self.shape_pairs_max,
                        _compute_per_world_mask_pair_max(model, mesh_capacity, plane_capacity),
                    )
                if has_heightfields:
                    # Heightfield pairs with any shape other than a mesh or a heightfield
                    # reach the terrain midphase. This static upper bound on its routed
                    # pairs (not the per-pass count) sizes its per-pair storage.
                    hfield, mesh = int(GeoType.HFIELD), int(GeoType.MESH)
                    if self.broad_phase_mode == "explicit" and self.shape_pairs_filtered is not None:
                        explicit = explicit_pairs_host.astype(np.int64)
                        valid = np.all((explicit >= 0) & (explicit < len(shape_types)), axis=1)
                        first, second = shape_types[explicit[valid, 0]], shape_types[explicit[valid, 1]]
                        partner = np.where(first == hfield, second, first)
                        routed = ((first == hfield) != (second == hfield)) & (partner != mesh)
                        max_heightfield_pairs = int(np.count_nonzero(routed))
                    else:
                        partners = route_mask & (shape_types != hfield) & (shape_types != mesh)
                        max_heightfield_pairs = _compute_per_world_mask_pair_max(model, heightfield_capacity, partners)
                    max_heightfield_pairs = min(self.shape_pairs_max, max_heightfield_pairs)
                # Use lean GJK/MPR kernel when scene has no capsules, ellipsoids,
                # cylinders, or cones (which need full support function and axial
                # rolling post-processing)
                lean_unsupported = {
                    int(GeoType.CAPSULE),
                    int(GeoType.ELLIPSOID),
                    int(GeoType.CYLINDER),
                    int(GeoType.CONE),
                }
                use_lean_gjk_mpr = not bool(lean_unsupported & set(route_shape_types.tolist()))

                if max_triangle_pairs is None:
                    # NXN/SAP count from shape masks without materializing the model's lazy pair array.
                    triangle_pair_estimate = _estimate_triangle_pair_details(
                        model,
                        explicit_pairs_host,
                        route_mask,
                        mesh_capacity | heightfield_capacity,
                        planar_sdf_capacity,
                        hydroelastic=hydroelastic_sdf is not None,
                        variant_row_mask=None if variant_options is None else variant_options.row_mask,
                    )
                    max_triangle_pairs = _resolve_auto_triangle_pairs(
                        triangle_pair_estimate,
                        # Mirrors NarrowPhase: the global reducer exists only for reduced mesh routes.
                        has_reducer=self.reduce_contacts and (has_meshes or has_heightfields),
                        deterministic=deterministic,
                        speculative=self._speculative_enabled,
                        hashtable_size_factor=contact_reduction_hashtable_size_factor,
                    )
            if max_triangle_pairs is None:
                max_triangle_pairs = _TRIANGLE_PAIRS_MIN_CAPACITY

            if self.broad_phase_mode == "explicit":
                candidate_pair_work_estimate = self.shape_pairs_max
            else:
                candidate_pair_work_estimate = min(self.shape_pairs_max, _compute_per_world_shape_pairs_max(model))
            has_generic_convex_pairs, generic_convex_pair_work_estimate = _compute_generic_convex_pair_stats(
                model,
                broad_phase_mode=self.broad_phase_mode,
                shape_pairs_filtered=self.shape_pairs_filtered,
                candidate_pair_work_estimate=candidate_pair_work_estimate,
                shape_pairs_host=explicit_pairs_host,
                variant_options=variant_options,
            )
            split_pair_count_threshold = (
                _SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD
                if use_lean_gjk_mpr
                else _SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD
            )
            split_gjk_mpr = (
                device.is_cuda
                and has_generic_convex_pairs
                and generic_convex_pair_work_estimate >= split_pair_count_threshold
            )
            # Initialize narrow phase with pre-allocated buffers
            # max_triangle_pairs is a conservative estimate for mesh collision triangle pairs
            # Pass write_contact as custom writer to write directly to final Contacts format
            #
            # contact_max is passed explicitly so NarrowPhase sizes its internal
            # deterministic sort buffers to rigid_contact_max (the same capacity
            # the Contacts buffer uses) rather than falling back to the default
            # max_candidate_pairs.  On SAP/NXN scenes with thousands of shapes
            # the candidate-pair bound (N*(N-1)/2 per world) is orders of
            # magnitude larger than the neighbor-budget contact estimate and
            # allocating sorter scratch at that size burns multi-GB of VRAM.
            if body_pair_reduction is not None and hydroelastic_sdf is not None:
                raise ValueError("body_pair_reduction does not support hydroelastic contacts")
            self.narrow_phase = NarrowPhase(
                max_candidate_pairs=self.shape_pairs_max,
                max_triangle_pairs=max_triangle_pairs,
                max_mesh_mesh_pairs=max_mesh_mesh_pairs,
                max_mesh_plane_pairs=max_mesh_plane_pairs,
                max_heightfield_pairs=max_heightfield_pairs,
                reduce_contacts=self.reduce_contacts,
                device=device,
                shape_aabb_lower=shape_aabb_lower,
                shape_aabb_upper=shape_aabb_upper,
                contact_writer_warp_func=contact_writer,
                shape_voxel_resolution=model._shape_voxel_resolution,
                hydroelastic_sdf=hydroelastic_sdf,
                has_meshes=has_meshes,
                has_heightfields=has_heightfields,
                use_lean_gjk_mpr=use_lean_gjk_mpr,
                convex_support_acceleration=model._convex_support_lut.shape[0] > 1,
                box_box_sat=box_box_sat,
                has_generic_convex_pairs=has_generic_convex_pairs,
                split_gjk_mpr=split_gjk_mpr,
                candidate_pair_work_estimate=candidate_pair_work_estimate,
                mesh_sdf_identity_scale_only=mesh_sdf_identity_scale_only,
                mesh_sdf_texture_only=mesh_sdf_texture_only,
                sdf_texture_paired_samples=model._sdf_texture_paired_samples,
                deterministic=deterministic,
                # The pipeline sorts with its own sorter; the narrow phase's standalone sort buffers stay empty.
                contact_max=0,
                verify_buffers=verify_buffers,
                contact_reduction_hashtable_size_factor=contact_reduction_hashtable_size_factor,
                speculative=self._speculative_enabled,
                contact_writer_supports_speculative=self._speculative_enabled,
                contact_export=contact_export,
                heightfield_primitive_query=True
                if convex_collision == "auto" and has_heightfields and int(GeoType.CONVEX_MESH) in route_shape_types
                else None,
            )
            self.hydroelastic_sdf = self.narrow_phase.hydroelastic_sdf

        # Marks shapes whose texture SDF queries assume the unit scale they had at construction.
        shape_unit_scale_sdf = np.zeros(shape_count if unit_scale_sdf_shapes is not None else 0, dtype=np.int32)
        if unit_scale_sdf_shapes is not None:
            shape_unit_scale_sdf[unit_scale_sdf_shapes] = 1
        self._shape_unit_scale_sdf = wp.array(shape_unit_scale_sdf, dtype=wp.int32, device=device)
        self._unit_scale_sdf_rescaled = wp.zeros(min(len(shape_unit_scale_sdf), 1), dtype=wp.int32, device=device)
        self._unit_scale_sdf_rescaled_host = None
        if unit_scale_sdf_shapes is not None:
            self.narrow_phase._shape_unit_scale_sdf = self._shape_unit_scale_sdf
            self.narrow_phase._unit_scale_sdf_rescaled = self._unit_scale_sdf_rescaled
            self._unit_scale_sdf_rescaled_host = wp.zeros(
                1, dtype=wp.int32, device="cpu", pinned=wp.get_device(device).is_cuda
            )

        # Analytic and convex manifolds use compact unique sub-keys even when
        # matching; complex contact families retain the full fingerprint width.
        self._contact_sort_sub_key_bits = getattr(
            self.narrow_phase, "_contact_sort_sub_key_bits", CONTACT_SORT_SUB_KEY_BITS
        )

        self._select_explicit_pair_kernel(model, pair_templates, using_expert_components, deterministic)

        # A declared variable that an explicit argument superseded was present but not applied.
        superseded = []
        if convex_collision is not None:  # it replaces both environment reads above, unvalidated
            superseded.append("NEWTON_NARROW_PHASE_COHERENT_CONVEX")
        if getattr(self.narrow_phase, "_route_requests", {}).get("primitive_query_source") == "argument":
            superseded.append("NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY")
        if getattr(self.narrow_phase, "contact_export", None) is not None:
            # NarrowPhase reads NEWTON_REDUCER_COMPACT_EXPORT only when contact_export is None.
            superseded.append("NEWTON_REDUCER_COMPACT_EXPORT")
        if immobile_source == "argument":  # include_immobile_pairs replaces the read, unvalidated
            superseded.append("NEWTON_BROAD_PHASE_IMMOBILE_FILTER")
        applied = self._environment_record["applied"]
        self._environment_record["overridden"] = {
            name: applied.pop(name) for name in sorted(superseded) if name in applied
        }

        self._configure_immobile_pairs(model, not include_immobile_pairs, immobile_source)

        if getattr(self.narrow_phase, "_heightfield_finite_query", False) or getattr(
            self.narrow_phase, "_heightfield_geometric_cull", False
        ):
            from ..geometry.heightfield_finite import bind_model  # noqa: PLC0415

            bind_model(self.narrow_phase, model)

        if (
            not using_expert_components
            and self.broad_phase_mode in ("explicit", "nxn")
            and getattr(self.narrow_phase, "_heightfield_pair_csr_requested", False)
        ):
            from ..geometry.heightfield_pair_csr import bind as bind_pair_csr  # noqa: PLC0415

            bind_pair_csr(
                self.narrow_phase,
                explicit_pairs_host if self.broad_phase_mode == "explicit" else None,
                shape_types,
                unique_generated=self.broad_phase_mode == "nxn",
            )

        # NarrowPhase is authoritative for the producer stage: it disables
        # mesh/heightfield reduction when no such collision path exists, and
        # expert construction may provide a preconfigured instance.  Publish
        # the effective state separately from the released requested-policy
        # ``reduce_contacts`` attribute.
        self.mesh_contact_reduction_enabled = bool(self.narrow_phase.reduce_contacts)

        self._hydro_shape_sdf_data_prepared = self.hydroelastic_sdf is not None
        if self.hydroelastic_sdf is not None:
            # Model SDF descriptors are finalized here; only shape transforms change per frame.
            self.hydroelastic_sdf._prepare_shape_sdf_data(model._texture_sdf_data, model._shape_sdf_index)

        # Allocate buffers
        with wp.ScopedDevice(device):
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.geom_data = wp.zeros(shape_count, dtype=wp.vec4, device=device)
            self.geom_transform = wp.zeros(shape_count, dtype=wp.transform, device=device)
            # Updated in place by Model.select_body_variants(); skipped for models without variants.
            has_variant_rows = model._has_variant_rows
            self._shape_variant_row = (
                model._shape_variant_row if has_variant_rows else wp.empty(0, dtype=wp.bool, device=device)
            )
            if self._speculative_enabled:
                self._shape_linear_velocity = wp.zeros(shape_count, dtype=wp.vec3, device=device)
                self._shape_angular_velocity = wp.zeros(shape_count, dtype=wp.vec3, device=device)
                self._shape_search_gap = wp.zeros(shape_count, dtype=wp.float32, device=device)
                self._shape_displacement = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            else:
                self._shape_linear_velocity = wp.empty(0, dtype=wp.vec3, device=device)
                self._shape_angular_velocity = wp.empty(0, dtype=wp.vec3, device=device)
                self._shape_search_gap = wp.empty(0, dtype=wp.float32, device=device)
                self._shape_displacement = wp.empty(0, dtype=wp.vec3, device=device)

        if (
            getattr(self.narrow_phase, "shape_aabb_lower", None) is None
            or getattr(self.narrow_phase, "shape_aabb_upper", None) is None
        ):
            raise ValueError("narrow_phase must expose shape_aabb_lower and shape_aabb_upper arrays")
        if self.narrow_phase.shape_aabb_lower.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_aabb_lower must have one entry per model shape "
                f"(expected {shape_count}, got {self.narrow_phase.shape_aabb_lower.shape[0]})"
            )
        if self.narrow_phase.shape_aabb_upper.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_aabb_upper must have one entry per model shape "
                f"(expected {shape_count}, got {self.narrow_phase.shape_aabb_upper.shape[0]})"
            )

        # The mesh contact kernels index the voxel-resolution table per shape. Expert
        # construction rarely supplies it, and the pipeline holds the authoritative model
        # data, so bind it here; leaving the narrow phase without one faults inside those
        # kernels. A supplied table must match the model it is used with.
        voxel_resolution = getattr(self.narrow_phase, "shape_voxel_resolution", None)
        if voxel_resolution is None:
            self.narrow_phase.shape_voxel_resolution = model._shape_voxel_resolution
        elif voxel_resolution.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_voxel_resolution must have one entry per model shape "
                f"(expected {shape_count}, got {voxel_resolution.shape[0]})"
            )
        elif voxel_resolution.device != model.device:
            raise ValueError(
                "narrow_phase.shape_voxel_resolution must be on the model device "
                f"(expected {model.device}, got {voxel_resolution.device})"
            )

        if self._pair_shape_prep:
            if (
                using_expert_components
                or self.broad_phase_mode != "explicit"
                or model.particle_count != 0
                or self.hydroelastic_sdf is not None
                or self._speculative_enabled
                or requires_grad
            ):
                raise ValueError(
                    "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP=1 requires internal explicit broad/narrow phases, "
                    "no particles, no hydroelastic or speculative contacts, and no gradients"
                )
            # The explicit-pair constructor contract already prohibits pair mutation.
            # Do not intersect with shape flags: explicit pairs are authoritative.
            if explicit_pairs_host is None:
                explicit_pairs_host = self.shape_pairs_filtered.numpy().reshape(-1, 2)
            pair_indices = np.unique(explicit_pairs_host.reshape(-1))
            if np.any(pair_indices < 0) or np.any(pair_indices >= shape_count):
                raise ValueError("Explicit pair shape IDs must be within model.shape_count")
            self.prepared_shape_indices = wp.array(pair_indices, dtype=wp.int32, device=device)

        self._convex_selection_reason = "explicit_ordinary"
        # Built here (not in finalize) so models/tasks that never collide don't pay for it.
        if coherent_convex != "0":
            unsupported = (
                using_expert_components
                or self.broad_phase_mode != "explicit"
                or self.shape_pairs_filtered is not getattr(model, "shape_contact_pairs", None)
                or not self.narrow_phase.split_gjk_mpr
                or not self.narrow_phase._use_lean_gjk_mpr
                or model.particle_count != 0
                or self.narrow_phase.has_meshes
                or self.narrow_phase.has_heightfields
                or self.hydroelastic_sdf is not None
                or self._speculative_enabled
                or body_pair_reduction is not None
                or requires_grad
            )
            if coherent_convex == "auto":
                self._convex_selection_reason = "unsupported_pipeline"
                if unsupported:
                    coherent_convex = "0"
                else:
                    types = model.shape_type.numpy()
                    if explicit_pairs_host is None:
                        explicit_pairs_host = self.shape_pairs_filtered.numpy().reshape(-1, 2)
                    pairs = explicit_pairs_host
                    supported = np.isin(types, (int(GeoType.BOX), int(GeoType.CONVEX_MESH)))
                    eligible = int(np.count_nonzero(supported[pairs[:, 0]] & supported[pairs[:, 1]]))
                    if eligible == 0:
                        self._convex_selection_reason = "no_eligible_pairs"
                        coherent_convex = "0"
            elif unsupported:
                raise ValueError(
                    "experimental coherent convex_collision rejection requires model-built unique explicit pairs, "
                    "the internal lean CUDA split pipeline, no mesh/heightfield/particle/hydroelastic or "
                    "speculative contacts, no body-pair reduction, and no gradients"
                )
        # The admitted coherent-convex owner ("0" when none), reported by get_collision_selection().
        self._coherent_convex_mode = coherent_convex
        if coherent_convex != "0":
            from ..geometry.coherent_convex import _ConvexQueryCache  # noqa: PLC0415
            from ..geometry.coherent_convex_rejection import _create_rejection_query_kernels  # noqa: PLC0415

            if explicit_pairs_host is None:
                explicit_pairs_host = self.shape_pairs_filtered.numpy().reshape(-1, 2)
            self.narrow_phase._coherent_cache = _ConvexQueryCache(
                pairs=explicit_pairs_host,
                shape_types=model.shape_type.numpy(),
                shape_world=model.shape_world.numpy(),
                world_count=model.world_count,
                query_capacity=self.narrow_phase.split_query_results.shape[0],
                device=device,
            )
            # The split queue's lean support: accelerated when the model has cooked hull support data.
            self.narrow_phase._coherent_query_kernels = _create_rejection_query_kernels(
                model._convex_support_lut.shape[0] > 1, diagnostics=coherent_stats == "1"
            )
            self._convex_selection_reason = "explicit_rejection"
            if coherent_convex == "auto":
                from ..geometry.convex_query_cost import _ConvexMethodChoice  # noqa: PLC0415

                self.narrow_phase._convex_method_choice = _ConvexMethodChoice(device=device, slots=eligible)
                self._convex_selection_reason = "sampled_cost"

        # The terrain law, writing its partners' heightfield contacts (by default) or in shadow: decided and built once
        # here, outside capture; collide() launches it after the narrow phase.
        self._build_terrain_law(
            model,
            terrain_law_request,
            terrain_law_value,
            terrain_law_classes,
            terrain_shadow_classes,
            explicit_pairs_host if self.broad_phase_mode == "explicit" else None,
            terrain_meshes,
        )

        # Built here (not in finalize) so models/tasks that never collide don't pay for it.
        # Host-side, so not graph-capture-safe -- construct the pipeline before any capture.
        is_mesh = np.zeros(model.shape_count, dtype=bool)
        mesh_mask = is_mesh
        if enable_rigid_soft_full_surface_contact and model.shape_count:
            is_mesh = np.isin(model.shape_type.numpy(), (int(GeoType.MESH), int(GeoType.CONVEX_MESH)))
            # Mesh feature precompute is costly, so only meshes that collide with particles at
            # construction use it. The rest keep the per-particle path, which still honors
            # COLLIDE_PARTICLES being enabled later.
            mesh_mask = is_mesh & ((model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
        self.soft_rigid_contact_pairs = _build_soft_particle_rigid_contact_pairs(model, shape_ok=~mesh_mask)
        self._soft_contact_pair_count = len(self.soft_rigid_contact_pairs)
        self.enable_rigid_soft_full_surface_contact = enable_rigid_soft_full_surface_contact
        self._soft_mesh_contact_data = None
        empty_pairs = wp.empty(0, dtype=wp.vec2i, device=model.device)

        if enable_rigid_soft_full_surface_contact and model.shape_count:
            shape_types = model.shape_type.numpy()
            common_capable = _full_surface_capable_shape_mask(model) & ~is_mesh
            heightfield_capable = shape_types == int(GeoType.HFIELD)
            _warn_full_surface_fallbacks(model, common_capable | mesh_mask | heightfield_capable)
            self._soft_face_sdf_geo_types = tuple(
                int(geo) for geo in _SDF_SPECIALIZED_GEO_TYPES if np.any(common_capable & (shape_types == int(geo)))
            )
            self._soft_edge_sdf_geo_types = self._soft_face_sdf_geo_types
            self.soft_edge_rigid_pairs = _build_soft_edge_rigid_contact_pairs(model, common_capable)
            self.soft_face_rigid_pairs = _build_soft_face_rigid_contact_pairs(model, common_capable)
            self._soft_heightfield_face_pairs = _build_soft_face_rigid_contact_pairs(model, heightfield_capable)
            mesh_vertex_pairs = _build_soft_particle_rigid_contact_pairs(model, shape_ok=mesh_mask)
            if len(mesh_vertex_pairs):
                self._soft_mesh_contact_data = MeshContactData(
                    model,
                    mesh_mask,
                    mesh_vertex_pairs,
                )
        else:
            self.soft_edge_rigid_pairs = empty_pairs
            self.soft_face_rigid_pairs = empty_pairs
            self._soft_heightfield_face_pairs = empty_pairs
            self._soft_face_sdf_geo_types = ()
            self._soft_edge_sdf_geo_types = ()

        self._soft_contact_tids_size = self.soft_contact_pair_count + sum(
            len(pairs)
            for pairs in (
                self.soft_edge_rigid_pairs,
                self.soft_face_rigid_pairs,
                self._soft_heightfield_face_pairs,
            )
        )
        if self._soft_contact_tids_size > np.iinfo(np.int32).max:
            raise ValueError("Soft contact candidates exceed the 32-bit contact indexing capacity.")
        self._soft_heightfield_large_scan = False
        if len(self._soft_heightfield_face_pairs):
            terrain = model.heightfield_data.numpy()
            self._soft_heightfield_large_scan = bool(
                np.any((terrain["nrow"] - 1) * (terrain["ncol"] - 1) > _HEIGHTFIELD_CELLS_PER_TASK)
            )
        sdf_pairs = max(len(self.soft_edge_rigid_pairs), len(self.soft_face_rigid_pairs))
        if self.requires_grad or not model.device.is_cuda or sdf_pairs < _SDF_COMPACTION_MIN_PAIRS:
            sdf_pairs = 0
        self._soft_sdf_fallback_tids = wp.empty(sdf_pairs, dtype=wp.int32, device=model.device)
        self._soft_sdf_fallback_count = wp.zeros(1, dtype=wp.int32, device=model.device)
        if soft_contact_max is None:
            soft_contact_max = self._soft_contact_tids_size
            if self._soft_mesh_contact_data is not None:
                soft_contact_max += self._soft_mesh_contact_data.contact_capacity_hint
        if soft_contact_max >= np.iinfo(np.int32).max:
            raise ValueError("Soft contact capacity exceeds 32-bit contact indexing.")
        self.soft_contact_gap = soft_contact_gap
        # Soft (cloth) self-contact tuning values, populated by
        # init_soft_self_contact(); consumed at detection time like
        # soft_contact_gap (detection query radius = margin + gap; pairs
        # closer than the exclusion radius in the rest shape are skipped).
        self.soft_self_contact_margin = 0.0
        self.soft_self_contact_gap = 0.0
        self.soft_self_contact_rest_shape_exclusion_radius = 0.0
        self._soft_contact_max = soft_contact_max
        model._validate_contact_capacity(rigid_contact_max, soft_contact_max)

        self.requires_grad = requires_grad
        if (
            (requires_grad or enable_rigid_soft_full_surface_contact)
            and variant_options is not None
            and model.particle_count > 0
        ):
            # Differentiable and full-surface soft contacts do not cull shapes by their AABBs.
            raise NotImplementedError(
                "Soft contacts with requires_grad=True or enable_rigid_soft_full_surface_contact=True "
                "do not support body variants."
            )
        self.deterministic = deterministic
        # Per-contact material rows exist only with hydroelastic contacts; a body-pair reducing
        # pipeline takes only the buffers its contacts() allocates, which carry none.
        per_contact_props = self.narrow_phase.hydroelastic_sdf is not None
        if deterministic:
            self._sort_key_rows = self._contact_pool.rows(
                (ContactField("_sort_key_array", wp.int64),), on_rows=self._on_sort_key_rows
            )
        else:
            self._sort_key_array = wp.zeros(0, dtype=wp.int64, device=device)
        if deterministic:
            self._contact_sorter = ContactSorter(
                self._contact_pool.ready,
                per_contact_shape_properties=per_contact_props,
                allocate_simple_scratch=False,
                device=device,
                pool=self._contact_pool,
            )
        else:
            self._contact_sorter = None

        self.contact_matching = contact_matching
        self._matching_enabled = matching_enabled
        self._matching_sticky = matching_sticky
        self.contact_report = contact_report
        if matching_enabled:
            self._contact_matcher = ContactMatcher(
                self._contact_pool.ready,
                pool=self._contact_pool,
                sorter=self._contact_sorter,
                shape_world=model.shape_world,
                world_count=model.world_count,
                shape_index_bits=self._contact_sort_shape_index_bits,
                sub_key_bits=self._contact_sort_sub_key_bits,
                pos_threshold=contact_matching_pos_threshold,
                normal_dot_threshold=contact_matching_normal_dot_threshold,
                contact_report=contact_report,
                sticky=matching_sticky,
                device=device,
            )
        else:
            self._contact_matcher = None

        if body_pair_reduction is not None:
            # Material-equivalence grouping: shapes on one body merge only when every solver-visible
            # material field matches exactly (see build_reduction_groups). Group ids must pack exactly
            # into the reduction key: aliasing two groups could evict a patch's deepest contact.
            shape_group, group_count = build_reduction_groups(model, body_pair_reduction.material_attributes)
            if group_count > MAX_GROUP_ID + 1:
                raise ValueError(f"body_pair_reduction supports at most {MAX_GROUP_ID + 1} groups, got {group_count}")
            self._body_pair_reducer = BodyPairContactReducer(
                self._contact_pool.ready,
                body_pair_reduction.cell_size,
                device,
                shape_group=shape_group,
                up_axis=int(getattr(model, "up_axis", 2)),
                shape_world=model.shape_world.numpy(),
                world_count=max(int(model.world_count), 1),
                borrowed_scratch=(self._contact_sorter.borrow_full_scratch if self._contact_sorter else None),
                pool=self._contact_pool,
                preserve_sort_keys=deterministic and matching_enabled,
                verify=body_pair_reduction.verify,
                hysteresis=body_pair_reduction.hysteresis,
                hashtable_size_factor=body_pair_reduction.hashtable_size_factor,
                # Sized from the broad phase's own candidate pairs; an explicit list is enumerated.
                group_pair_bound=build_reduction_group_bound(model, self.shape_pairs_filtered, shape_group),
            )
            self._reduction_reset_mask = wp.zeros(max(int(model.world_count), 1), dtype=wp.int32, device=device)
        else:
            self._body_pair_reducer = None
            self._reduction_reset_mask = None
        self._reduces_body_pairs = self._body_pair_reducer is not None
        # The buffer the reducer's history belongs to (a weak reference).
        self._last_contacts_ref = None

        # Soft (cloth) self-contact: disabled until init_soft_self_contact() creates
        # the shared detector (re-pointed per Contacts buffer; see
        # _get_soft_self_contact_detector).
        self._soft_self_contact_detector: TriMeshCollisionDetector | None = None

        # The forecast every pass of this pipeline adds into, on the passes the pool picks (ShapeForecast).
        pool = self._contact_pool
        self._forecast = None
        self._no_velocities = wp.empty(0, dtype=wp.spatial_vector, device=device)
        if pool.forecast_on is not None and pool.growable and model.shape_count:
            self._forecast = _shape_forecast(
                model, pool, self.geom_transform, terrain_law=self._terrain_law_listed is not None, estimate=estimate
            )
        pool.publish_reach(float(np.linalg.norm(model.gravity.numpy().reshape(-1, 3), axis=1).max()))
        # Captured contact launches that a replay skipped surface with the narrow phase's capacity flags.
        self.narrow_phase.contact_pool = pool

        # Publish only after all construction succeeds.
        if publish_capacity:
            model.rigid_contact_max = rigid_contact_max
            model.soft_contact_max = soft_contact_max
            model._contact_capacity_initialized = True

    @property
    def body_pair_reduction(self) -> CollisionPipeline.BodyPairReductionConfig | None:
        """Body-pair reduction configuration, fixed at construction, or ``None`` when disabled.

        .. experimental::
        """
        return self._body_pair_reduction

    def refresh_body_pair_reduction_groups(self):
        """Rebuild body-pair reduction groups from the model's current shape materials.

        Groups are built at construction.  Call this outside CUDA graph capture
        after changing ``shape_material_*`` or a grouped per-shape attribute.
        No-op when body-pair reduction is disabled.

        .. experimental::
        """
        if self._body_pair_reducer is None:
            return
        if self.device.is_cuda and self.device.is_capturing:
            raise RuntimeError("refresh_body_pair_reduction_groups() must be called outside CUDA graph capture")
        shape_group, group_count = build_reduction_groups(self.model, self.body_pair_reduction.material_attributes)
        if group_count > MAX_GROUP_ID + 1:
            raise ValueError(f"body_pair_reduction supports at most {MAX_GROUP_ID + 1} groups, got {group_count}")
        self._body_pair_reducer.shape_group.assign(shape_group)
        # Winners recorded under the old classes are not comparable to the new
        # grouping; severing history is the conservative continuation.
        self._body_pair_reducer.reset_history()
        if self._contact_matcher is not None:
            self._contact_matcher.reset()

    def reset_body_pair_reduction_history(self, world_mask=None):
        """Erase the body-pair reduction's hysteresis history.

        With ``BodyPairReductionConfig.hysteresis > 0`` the pipeline carries last
        step's slot winners between :meth:`collide` calls. This clears only that
        history; :meth:`reset_contact_matching` clears it too, with the matched
        contacts.  No-op when reduction or hysteresis is disabled.

        Args:
            world_mask: ``None`` erases everything (host-side; call outside CUDA
                graph capture).  Otherwise a length-``model.world_count`` mask whose
                nonzero entries select the worlds to reset: an int32 device array
                (one fixed-size launch, capturable, with the caller rewriting the
                mask each step) or a host integer array (uploaded, outside capture).
        """
        if self._body_pair_reducer is not None:
            self._body_pair_reducer.reset_history(world_mask)

    def body_pair_reduction_stats(self) -> dict:
        """Whole-run telemetry of the body-pair contact reduction.

        Synchronizes the device; do not call during CUDA graph capture.  Values
        accumulate since construction or :meth:`clear_body_pair_reduction_stats`:

        * ``invariant_violations`` / ``outranked_discards`` -- implementation-invariant
          disagreements (``verify`` only; any nonzero value is a reducer bug).
        * ``probe_failures`` -- group-table keys not found or created within the
          bounded probe budget; each makes its whole frame keep every contact.
        * ``cell_clamp_events`` -- contacts whose spatial cell hit the packed range.
        * ``max_contacts_in`` / ``max_contacts_kept`` -- peak watermarks before and
          after reduction.
        * ``max_hashtable_entries`` / ``hashtable_capacity`` / ``hashtable_load``.
        * ``input_overflow_frames`` / ``fallback_frames`` / ``identity_frames``.
        * ``total_frames`` / ``sum_contacts_in`` / ``sum_contacts_kept``;
          ``sum_contacts_kept / sum_contacts_in`` is the achieved reduction ratio.

        Raises:
            RuntimeError: If body-pair contact reduction is not enabled.
        """
        if self._body_pair_reducer is None:
            raise RuntimeError("body-pair contact reduction is not enabled on this pipeline")
        return self._body_pair_reducer.stats()

    def clear_body_pair_reduction_stats(self):
        """Zero the reduction telemetry accumulators (host-side, outside CUDA graph capture).

        Raises:
            RuntimeError: If body-pair contact reduction is not enabled.
        """
        if self._body_pair_reducer is None:
            raise RuntimeError("body-pair contact reduction is not enabled on this pipeline")
        self._body_pair_reducer.clear_stats()

    def body_pair_reduction_description(self) -> dict:
        """Currently allocated buffer footprint of the body-pair contact reduction by role.

        Raises:
            RuntimeError: If body-pair contact reduction is not enabled.
        """
        if self._body_pair_reducer is None:
            raise RuntimeError("body-pair contact reduction is not enabled on this pipeline")
        return self._body_pair_reducer.describe()

    def _terrain_law_unsupported(self) -> tuple[str, str, str] | None:
        """Return why the terrain law cannot write this pipeline's heightfield contacts, or ``None`` when it can.

        Returns:
            ``(configuration, what the law lacks, how else to leave the configuration)``, or ``None``.
        """
        if self._speculative_enabled:
            return "speculative contacts", "does not write speculative contacts yet", " or speculative_contact_gap_max"
        if getattr(self.narrow_phase, "_convex_writer_func", write_contact) is not write_contact:
            # The law writes through the stock writer; a prebuilt narrow phase's own writer would not see its contacts.
            return (
                "a prebuilt narrow phase with its own contact writer",
                "writes through the stock contact writer, not the narrow phase's",
                "",
            )
        return None

    def _law_pairs(self, explicit_pairs_host: np.ndarray | None) -> np.ndarray | None:
        """The explicit pairs the terrain law takes, from the setup's one host copy; ``None`` without a list."""
        if self.broad_phase_mode != "explicit":
            return None
        if explicit_pairs_host is None:
            return self.shape_pairs_filtered.numpy().reshape(-1, 2)
        return explicit_pairs_host

    def _build_terrain_law(
        self,
        model: Model,
        request: str,
        value: str,
        classes: tuple[int, ...],
        shadow_classes: tuple[int, ...],
        explicit_pairs_host: np.ndarray | None,
        terrain_meshes: bool,
    ) -> None:
        """Decide whether the terrain law writes the heightfield contacts, and build it where its kernels run.

        Sets the law (writing contacts or in shadow), whether it writes, the decision and its reason (the route
        record's ``heightfield["law"]``) and the shadow's reason (``heightfield["law_shadow"]``), and gives the narrow
        phase the partners whose heightfield pairs it drops. Called once, at construction, outside capture.

        Args:
            model: The finalized model.
            request: What ``NEWTON_TERRAIN_LAW`` requests: ``default``, ``explicit`` or ``none``.
            value: The ``NEWTON_TERRAIN_LAW`` value, for the record and the errors.
            classes: The partner classes ``NEWTON_TERRAIN_LAW`` lists (empty but for an explicit request).
            shadow_classes: The partner classes ``NEWTON_TERRAIN_LAW_SHADOW`` lists.
            explicit_pairs_host: The setup's host copy of the explicit pair list, when it read one.
            terrain_meshes: Whether the law also reads the static meshes that qualify as surfaces.

        Raises:
            ValueError: When ``NEWTON_TERRAIN_LAW`` requests the law for a configuration it does not support yet, or a
                tuning variable of the law has an invalid value.
        """
        narrow = self.narrow_phase
        # A prebuilt narrow phase may still drop the pairs another pipeline gave the law: this pipeline decides.
        narrow.set_terrain_law_roles(None)
        self._terrain_law_shadow_reason = "Off: NEWTON_TERRAIN_LAW_SHADOW lists no partner class."
        surfaces = None
        if narrow.has_heightfields or (terrain_meshes and _has_static_meshes(model)):
            # The law's surfaces (TerrainSurfaces): the heightfields, and with the opt-in the static meshes that qualify.
            from ..geometry.terrain_contacts import TerrainSurfaces  # noqa: PLC0415

            surfaces = TerrainSurfaces(model, meshes=terrain_meshes)
        if surfaces is None or surfaces.count == 0:
            self._terrain_law_decision = None
            self._terrain_law_reason = "No heightfield pair."
            if shadow_classes:
                self._terrain_law_shadow_reason = "No heightfield pair: the listed classes have nothing to run on."
            return
        keep = "the terrain route writes every heightfield contact."
        unsupported = self._terrain_law_unsupported()
        writes = False
        if request == "none":
            decision = "opted_out"
            reason = f"NEWTON_TERRAIN_LAW=none, a temporary opt-out until the terrain route is removed: {keep}"
        elif request == "explicit" and unsupported is not None:
            configuration, lack, remedy = unsupported
            raise ValueError(
                f"NEWTON_TERRAIN_LAW={value!r} requests the terrain law, which {lack}; unset NEWTON_TERRAIN_LAW (the "
                f"default keeps the terrain route for {configuration}){remedy}"
            )
        elif shadow_classes:
            decision = "unsupported"
            reason = (
                f"NEWTON_TERRAIN_LAW_SHADOW runs the law in shadow beside the terrain route (comparison mode): {keep}"
            )
        elif unsupported is not None:
            decision = "unsupported"
            configuration, lack, _remedy = unsupported
            reason = f"{configuration[0].upper()}{configuration[1:]}: the terrain law {lack}, so {keep}"
        else:
            writes = True
            decision = request
        if not writes and not shadow_classes:
            self._terrain_law_decision, self._terrain_law_reason = decision, reason
            return
        from ..geometry.terrain_shadow import (  # noqa: PLC0415
            TERRAIN_LAW_KERNELS,
            TerrainLawShadow,
            parse_terrain_law_shadow_jobs,
            parse_terrain_law_shadow_rows,
        )

        law = TerrainLawShadow(
            model,
            (classes if request == "explicit" else tuple(sorted(TERRAIN_LAW_KERNELS))) if writes else shadow_classes,
            pairs=self._law_pairs(explicit_pairs_host),
            surfaces=surfaces,
            status=narrow._buffer_capacity_status,
            # The rows of its scene-level spill region (default: TerrainLawShadow.default_spill_rows()).
            spill_rows=parse_terrain_law_shadow_rows(os.environ.get("NEWTON_TERRAIN_LAW_SHADOW_ROWS", "")),
            # The factor on its solid classes' job capacities (default 1: terrain_solid_job_capacities()).
            job_scale=parse_terrain_law_shadow_jobs(os.environ.get("NEWTON_TERRAIN_LAW_SHADOW_JOBS", "")),
        )
        if not writes:
            self._terrain_law_decision, self._terrain_law_reason = decision, reason
            if law.features is None:
                self._terrain_law_shadow_reason = "No heightfield pair has a partner of the listed classes."
            else:
                self._terrain_law_shadow = law
                self._terrain_law_shadow_reason = (
                    "NEWTON_TERRAIN_LAW_SHADOW: the law's kernels run after the narrow phase into a side buffer that "
                    "nothing reads; the terrain route writes every contact."
                )
            return
        partner, written = law.written_pairs(model)
        if law.features is None or not len(written):
            self._terrain_law_decision = "no_partner"
            listed = "of the listed classes " if request == "explicit" else ""
            self._terrain_law_reason = f"No heightfield pair has a partner {listed}with a core in the law: {keep}"
            return
        self._terrain_law_shadow = law
        self._terrain_law_writes = True
        self._terrain_law_listed = wp.array(written, dtype=wp.int32, device=model.device)
        narrow.set_terrain_law_roles(wp.array(partner, dtype=wp.int32, device=model.device))
        self._terrain_law_decision = decision
        writer = (
            "Default: the terrain law writes the contacts of every heightfield pair whose partner it has a core for "
            "(spheres, capsules, boxes, cylinders, cones, ellipsoids and convex hulls)"
            if request == "default"
            else f"NEWTON_TERRAIN_LAW={value.strip()}: the terrain law writes the contacts of the listed classes' "
            "heightfield pairs"
        )
        self._terrain_law_reason = (
            f"{writer} after the narrow phase, which drops those pairs; other partners keep their routes. "
            "NEWTON_TERRAIN_LAW=none keeps the terrain route (a temporary opt-out)."
        )

    def _select_explicit_pair_kernel(
        self, model: Model, setting: str | None, using_expert_components: bool, deterministic: bool
    ) -> None:
        """Choose the explicit broad-phase kernel once, at construction, and record why.

        The world-template kernel replaces the flat one only where it tests the same
        pairs: the explicit list must be the model-built replay of per-world templates,
        and the templates are verified against its contents on the device. Verification
        also allocates the layout, so ``collide()`` allocates nothing for it.
        """
        self._pair_templates = None
        if self.broad_phase_mode != "explicit":
            self._explicit_pair_kernel = None
            return
        layout = None
        templates = getattr(model, "_shape_contact_pair_templates", None)
        if using_expert_components:
            reason = "A prebuilt broad phase keeps its own kernel."
        elif setting == "0":
            reason = "NEWTON_BROAD_PHASE_TEMPLATES=0 keeps the flat pair kernel."
        elif self._speculative_enabled:
            reason = "Speculative contacts sweep AABBs, which only the flat pair kernel supports."
        elif templates is None or not templates.describes(self.shape_pairs_filtered):
            reason = "The explicit pairs are not the model-built replay of per-world templates."
        elif not deterministic and BroadPhaseExplicitTemplates.flat_kernel_orders_writes():
            # A deterministic pipeline sorts its contacts, so only then is the candidate order free.
            reason = (
                "Warp's deterministic mode orders the flat pair kernel's candidate writes; the world-template "
                "kernel does not."
            )
        else:
            layout = BroadPhaseExplicitTemplates(templates, device=self.device)
            if layout.unsupported_reason is not None:
                reason = layout.unsupported_reason
            elif layout.slot_count == 0:
                reason = "The world templates hold no pairs; the flat pair kernel tests the global pairs."
            elif setting is None and not layout.automatic:
                reason = (
                    f"Automatic: below {layout.AUTO_MIN_PAIRS} template pairs or {layout.AUTO_MIN_PAIRS_PER_WORLD} "
                    "pairs per world, the flat pair kernel is kept."
                )
            elif not layout.verify(self.shape_pairs_filtered):
                reason = "The explicit pairs no longer match the model's world templates."
            else:
                self._pair_templates = layout
                reason = (
                    "NEWTON_BROAD_PHASE_TEMPLATES=1 tests the world templates."
                    if setting == "1"
                    else "Automatic: large world templates replace the flat pair stream."
                )
        self._explicit_pair_kernel = {
            "kernel": "world_templates" if self._pair_templates is not None else "flat_pairs",
            "source": "automatic" if setting is None else "environment",
            "reason": reason,
            "templates": None if layout is None else layout.describe(),
        }

    def get_collision_selection(self) -> dict:
        """Read the requested and current convex method outside CUDA graph capture.

        Automatic selection can change during replay. This reports the latest
        observation, not the fraction of calls using each method. With an
        automatic chooser installed (``convex_collision="auto"`` on CUDA), the
        top-level ``method`` and ``reason`` and the ``heightfield_support``
        fields read the chooser's device state and synchronize: never call this
        inside a timed window or a graph capture.

        ``narrow_phase`` records the coherent-convex mode, the broad phase and the
        generic convex query variants. Its ``broad_phase_kernel`` (``None`` unless
        the broad phase is explicit) records whether the flat pair kernel or the
        world-template kernel tests the explicit pairs, why, and the template
        layout. ``heightfield`` (``None`` without heightfield collision)
        records the terrain midphase, the typed primitive table and its
        closed-form rows, the route convex-hull pairs take, the terrain
        convex-cells owner and its chooser, each opt-in terrain route with
        its reason, ``law``: whether the terrain law writes the heightfield
        contacts (``writes``), why (``decision``: ``"default"``,
        ``"explicit"`` for a ``NEWTON_TERRAIN_LAW`` class list,
        ``"opted_out"`` for ``NEWTON_TERRAIN_LAW=none``, ``"unsupported"``
        for a configuration it does not support yet, ``"no_partner"`` when no
        heightfield pair has a partner it writes), ``reason`` and, when it
        writes, its partner classes, pairs per lane class, launches and
        memory, and ``law_shadow``: whether it runs in shadow instead
        (``NEWTON_TERRAIN_LAW_SHADOW``), with the same details.
        ``environment`` records the ``NEWTON_*`` variables present at
        construction: ``applied`` maps declared names Newton reads to their
        values, ``overridden`` maps declared names whose value an explicit
        argument superseded to their unapplied, unvalidated values
        (``convex_collision`` for the coherent-convex and convex-cells
        variables, the pipeline's typed-table request for
        ``NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY`` and ``contact_export`` for
        ``NEWTON_REDUCER_COMPACT_EXPORT``), ``unknown`` lists undeclared names
        (recorded, not rejected), ``retired`` lists retired names whose value
        did not raise and ``isaac_lab`` maps the names Isaac Lab reads and
        Newton does not. These three sections are construction-time host
        records: they read no device memory and are JSON-compliant.

        ``immobile_pairs`` records whether the broad phase drops pairs whose
        shapes are both structurally immobile (``enabled``; see
        ``include_immobile_pairs``), where that choice came from (``source``:
        ``"default"``, ``"environment"`` or ``"argument"``) and why
        (``reason``), and, as of construction or the last
        :meth:`notify_model_changed` that changed them (notifications refresh
        only while ``enabled``), how many bodies are welded to the world
        (``welded_bodies``), how many colliding shapes are immobile
        (``immobile_shapes``), how many listed pairs join two of them
        (``listed_pairs``: pairs of the explicit list, or for ``"nxn"`` and
        ``"sap"`` of the model's pair list; ``None`` without one) and how many
        listed pairs the broad phase drops that ``include_immobile_pairs=True``
        would keep (``excluded_pairs``): the listed immobile pairs, or with
        ``include_static_kinematic_pairs=False`` the pairs of a welded shape
        with a static, welded or kinematic one.
        """
        owner = getattr(self.narrow_phase, "_convex_method_choice", None)
        result = {"requested": self.convex_collision, "reason": self._convex_selection_reason}
        result["contact_export"] = {
            "requested": self.narrow_phase.contact_export,
            "method": "disabled"
            if not self.narrow_phase.reduce_contacts
            else "compact"
            if self.narrow_phase._reducer_compact_export
            else "tiled",
        }
        if owner is not None:
            result.update(owner.describe())
            if result["method"] == "reject_only":
                result["method"] = "rejection"
        else:
            result["method"] = (
                "rejection" if getattr(self.narrow_phase, "_coherent_cache", None) is not None else "ordinary"
            )
        result["narrow_phase"] = self._narrow_phase_selection()
        result["heightfield"] = self._heightfield_selection()
        result["immobile_pairs"] = dict(self._immobile_pairs)
        result["environment"] = copy.deepcopy(self._environment_record)
        return result

    def _narrow_phase_selection(self) -> dict:
        """Host-only record of the coherent-convex mode, the broad phase and the generic convex query variants."""
        narrow = self.narrow_phase

        def mode(value):
            return "off" if value == "0" else value

        reason = self._convex_selection_reason
        return {
            "coherent_convex": {
                "requested": mode(self._coherent_convex_requested),
                "source": "environment" if self.convex_collision is None else "argument",
                "mode": mode(self._coherent_convex_mode),
                "reason": reason,
            },
            "broad_phase": self.broad_phase_mode,
            "broad_phase_kernel": copy.deepcopy(self._explicit_pair_kernel),
            "split_gjk_mpr": bool(narrow.split_gjk_mpr),
            "lean_gjk_mpr": bool(narrow._use_lean_gjk_mpr),
            "sparse_gjk_pairs": bool(narrow.sparse_gjk_pairs),
            "mesh_reduction": bool(self.mesh_contact_reduction_enabled),
            "body_pair_reduction": self._reduces_body_pairs,
            "deterministic": bool(self.deterministic),
            "speculative": bool(self._speculative_enabled),
            "hydroelastic": self.hydroelastic_sdf is not None,
            "contact_matching": self.contact_matching,
            "pair_shape_prep": bool(self._pair_shape_prep),
        }

    def _heightfield_selection(self) -> dict | None:
        """Host-only record of the terrain routes; ``None`` without heightfield collision."""
        narrow = self.narrow_phase
        if not narrow.has_heightfields:
            return None
        requests = narrow._route_requests
        rows = [_heightfield_row(kernel) for kernel in narrow._heightfield_primitive_kernels]
        if narrow._heightfield_cylinder_split and len(rows) > 1:
            # The launch replaces index 1 (the typed cylinder query) with the staged GJK split.
            rows[1]["split_kernels"] = [kernel.key for kernel in narrow._heightfield_cylinder_split_kernels]
        if not narrow.reduce_contacts:
            hull = "finite_query_direct" if narrow._heightfield_finite_query else "generic_gjk_direct"
        elif narrow._heightfield_pair_csr:
            hull = "pair_csr"
        elif narrow._heightfield_finite_query:
            hull = "finite_query"
        elif narrow._heightfield_primitive_query:
            if rows and rows[-1].get("hull") == "closed_form":
                hull = "closed_form_hull_row"
            else:
                hull = "gjk_fallback"
        else:
            hull = "generic_gjk"
        if narrow._heightfield_pair_csr:
            midphase = "pair_csr"
        elif narrow._heightfield_geometric_cull:
            midphase = "geometric_cull"
        elif narrow._heightfield_cell_reject:
            midphase = "cell_reject_packed" if narrow._heightfield_packed_pairs else "cell_reject"
        else:
            midphase = "full"
        common = (
            (narrow.speculative, "speculative contacts"),
            (narrow.deterministic, "deterministic contacts"),
            (narrow.hydroelastic_sdf is not None, "hydroelastic contacts"),
        )
        # The pipeline requests the table by argument only for convex hulls on heightfields under
        # convex_collision 'auto'; otherwise the environment flag decides.
        typed_request = (
            "The typed-table request of convex_collision" if requests["primitive_query_source"] == "argument" else None
        )
        if not narrow._heightfield_primitive_query and not requests["primitive_query"]:
            primitive_reason = (
                "Off: NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY is not 1 and convex_collision requests no typed table "
                "(only 'auto' with convex hulls on heightfields does)."
            )
        else:
            primitive_reason = _route_gate(
                narrow._heightfield_primitive_query,
                True,
                "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY",
                (
                    (not narrow.reduce_contacts, "contact reduction is off"),
                    (narrow._heightfield_finite_query, "the finite query owns terrain pairs"),
                    (narrow._heightfield_pair_csr_requested, "the pair-CSR route owns terrain pairs"),
                    *common,
                ),
                request=typed_request,
            )
        if (
            requests["cylinder_split"]
            and narrow._heightfield_primitive_query
            and not narrow._heightfield_cylinder_split
        ):
            cylinder_reason = "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT=1 declined: the closed-form cylinder row replaces it."
        else:
            cylinder_reason = _route_gate(
                narrow._heightfield_cylinder_split,
                requests["cylinder_split"],
                "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT",
                ((not narrow._heightfield_primitive_query, "the typed primitive table is off"),),
            )
        if narrow._heightfield_pair_csr:
            pair_reason = "NEWTON_HEIGHTFIELD_PAIR_CSR=1: admitted and bound to unique model pairs."
        elif narrow._heightfield_pair_csr_requested:
            pair_reason = "NEWTON_HEIGHTFIELD_PAIR_CSR=1 admitted but not bound: the pairs are not unique model pairs."
        else:
            pair_reason = _route_gate(
                False,
                requests["pair_csr"],
                "NEWTON_HEIGHTFIELD_PAIR_CSR",
                (
                    (not narrow._heightfield_geometric_cull, "requires the geometric cull"),
                    (not narrow._heightfield_finite_query, "requires the finite query"),
                    (narrow.has_meshes, "meshes share the terrain path"),
                    *common,
                ),
            )
        deferred = narrow._heightfield_lane_deferred
        if narrow._heightfield_lane_midphase and narrow._heightfield_lane_small_cells >= 0:
            lane_reason = (
                "Default: one thread per pair; pairs with more than small_cells cells split their cells over lanes."
            )
        elif narrow._heightfield_lane_midphase:
            lane_reason = "Default on CPU: one thread per pair walks every cell in the original order; no lanes split."
        elif not requests["lane_midphase"]:
            lane_reason = "NEWTON_HEIGHTFIELD_LANE_MIDPHASE=0: one thread per pair walks every cell."
        else:
            lane_reason = "Off: the lane midphase needs packed heightfield cell rejection (no triangle meshes)."
        lane_midphase = {
            "enabled": bool(narrow._heightfield_lane_midphase),
            "small_cells": narrow._heightfield_lane_small_cells,
            "lanes": narrow._heightfield_lane_width,
            "deferred_capacity": None if deferred is None else int(deferred.windows.shape[0]),
            # Device bytes of the deferred windows and enclosing-box projections.
            "deferred_bytes": None
            if deferred is None
            else int(deferred.windows.capacity + deferred.projections.capacity),
            "reason": lane_reason,
        }
        if narrow._heightfield_row_gate:
            row_gate_reason = "Default: each typed row scans the triangle stream only when a current pair reaches it."
        elif not narrow._heightfield_primitive_query:
            row_gate_reason = "Off: the typed primitive table is off."
        else:
            row_gate_reason = "NEWTON_HEIGHTFIELD_ROW_GATE=0: every typed row scans the whole triangle stream."
        shadow = self._terrain_law_shadow is not None and not self._terrain_law_writes
        return {
            "midphase": midphase,
            "lane_midphase": lane_midphase,
            "hull_route": hull,
            "primitive_query": {
                "enabled": bool(narrow._heightfield_primitive_query),
                "source": requests["primitive_query_source"],
                "reason": primitive_reason,
                "rows": rows,
                "row_gate": {"enabled": bool(narrow._heightfield_row_gate), "reason": row_gate_reason},
            },
            "cylinder_split": {"enabled": bool(narrow._heightfield_cylinder_split), "reason": cylinder_reason},
            "geometric_cull": {
                "enabled": bool(narrow._heightfield_geometric_cull),
                "reason": _route_gate(
                    narrow._heightfield_geometric_cull,
                    requests["geometric_cull"],
                    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL",
                    (
                        (not narrow._heightfield_cell_reject, "cell rejection is off or meshes share the midphase"),
                        (narrow.speculative, "speculative contacts"),
                        (not narrow.reduce_contacts, "contact reduction is off"),
                    ),
                ),
            },
            "finite_query": {
                "enabled": bool(narrow._heightfield_finite_query),
                "reason": _route_gate(
                    narrow._heightfield_finite_query,
                    requests["finite_query"],
                    "NEWTON_HEIGHTFIELD_FINITE_QUERY",
                    (
                        (narrow.has_meshes, "meshes share the terrain path"),
                        (narrow.speculative, "speculative contacts"),
                    ),
                ),
            },
            "pair_csr": {
                "enabled": bool(narrow._heightfield_pair_csr),
                "shell": bool(narrow._heightfield_pair_csr_shell),
                "reason": pair_reason,
            },
            "reducer_export_cleanup": {
                "enabled": bool(narrow._reducer_export_cleanup),
                "reason": _route_gate(
                    narrow._reducer_export_cleanup,
                    requests["export_cleanup"],
                    "NEWTON_REDUCER_EXPORT_CLEANUP",
                    (
                        (not narrow.reduce_contacts, "contact reduction is off"),
                        (narrow._heightfield_finite_query, "the finite query owns export"),
                        (narrow._heightfield_pair_csr_requested, "the pair-CSR route owns export"),
                        *common,
                    ),
                ),
            },
            "law": {
                "writes": self._terrain_law_writes,
                "decision": self._terrain_law_decision,
                "classes": [],
                "reason": self._terrain_law_reason,
                **(self._terrain_law_shadow.describe() if self._terrain_law_writes else {}),
            },
            "law_shadow": {
                "enabled": shadow,
                "reason": self._terrain_law_shadow_reason,
                **(self._terrain_law_shadow.describe() if shadow else {}),
            },
        }

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh the immobile shapes after body-flag or joint-enable changes.

        This method handles two flags, and only while the broad phase drops pairs of
        structurally immobile shapes (see ``include_immobile_pairs``):
        ``ModelFlags.BODY_PROPERTIES`` copies :attr:`~newton.Model.body_flags` from the
        device and ``ModelFlags.JOINT_PROPERTIES`` copies :attr:`~newton.Model.joint_enabled`.
        The immobile shapes are recomputed only when a copied array changed, and are
        written in place, so graphs captured earlier use them on their next replay. Other
        properties these flags cover, such as :attr:`~newton.Model.joint_q` or
        :attr:`~newton.Model.joint_X_p`, do not affect the pipeline and need no
        notification here. Other flags are ignored: after material changes call
        :meth:`refresh_body_pair_reduction_groups`.

        Args:
            flags: Bit-mask of :class:`~newton.ModelFlags` or custom ``int`` bits
                indicating which model properties changed.

        Raises:
            RuntimeError: If a handled flag arrives while the broad phase drops immobile
                pairs and the model's device is capturing a graph.
        """
        body = bool(flags & ModelFlags.BODY_PROPERTIES)
        joint = bool(flags & ModelFlags.JOINT_PROPERTIES)
        if self._immobile_shape_body is None or not (body or joint):
            return
        if self.model.device.is_capturing:
            raise RuntimeError("notify_model_changed() reads the model on the host; call it outside graph capture")
        self._refresh_immobile_pairs(body=body, joint=joint)

    def _on_sort_key_rows(self, rows):
        rows.publish(self)

    @property
    def rigid_contact_max(self) -> int:
        """The rigid contact capacity resolved at construction: the rows of the pipeline's buffers.

        With ``growable_contacts`` the buffers start from it and grow on demand:
        read :attr:`newton.Contacts.rigid_contact_max` of :meth:`contacts` for the
        rows they hold now.
        """
        return self._rigid_contact_max

    def service_capacity(self, state: State | None = None, *, dt: float | None = None) -> bool:
        """Size the rigid contact buffers from measured demand; return whether CUDA graphs must be recaptured.

        Fixed buffers (without ``growable_contacts``) never grow, and this returns
        ``False``. Growable buffers, of this pipeline, of its :meth:`contacts` and of
        the solvers that step its model (e.g. :class:`~newton.solvers.SolverFeatherPGS`),
        start at a few thousand rows. Call this
        once per step outside capture, e.g. right after launching the step's
        graph. Every
        :meth:`collide` records its contact count in a sticky high-water mark,
        written to pinned host memory also when replayed from a graph; this reads
        it without synchronizing, so it lags the device by the steps still in
        flight, and when it passes
        :data:`~newton._src.core.contact_pool.GROW_THRESHOLD` of the rows the
        buffers grow to :data:`~newton._src.core.contact_pool.HEADROOM` rows per
        contact of it. One collision pass in
        :data:`~newton._src.core.contact_pool.FORECAST_PASSES` of every graph replay
        forecasts, inside its bounding-box and broad-phase launches, the contacts its
        pairs can make within the service's lag; while the demand is in transit (from
        the start, a measured state or a pass that dropped contacts, until the mark
        holds within :data:`~newton._src.core.contact_pool.TRIM_BAND` for
        :data:`~newton._src.core.contact_pool.TRIM_AFTER_PASSES` passes) the buffers
        grow ahead of it. On CUDA with ``gpu_components`` growth maps more memory
        behind the buffers' fixed addresses without synchronizing the device,
        and graphs whose contact launches follow the rows (this pipeline's and
        FeatherPGS's) use the new rows from their next replay. A replay whose
        contacts outnumbered the rows dropped the excess, flagged its capacity
        status and latched the worlds it dropped them from (:meth:`contact_drops`);
        this warns. Once the mark has stayed within
        :data:`~newton._src.core.contact_pool.TRIM_BAND` for
        :data:`~newton._src.core.contact_pool.TRIM_AFTER_PASSES` collision
        passes, the rows trim to
        :data:`~newton._src.core.contact_pool.HEADROOM` rows per contact of it:
        the ready rows drop in stream order and the pages past them are unmapped
        once the replays submitted before have completed, without synchronizing
        the device or recapturing (CUDA virtual memory only).

        With ``state``, the state a graph starts from, the state's rigid
        contacts are counted first (into scratch rows, writing no
        :class:`~newton.Contacts` and none of the pipeline's state across passes:
        contact matching, reduction history, capacity status) and the buffers
        grow for them too. A :meth:`collide` outside graph capture grows the
        buffers itself before anything reads them. The buffers hold at most what
        the device memory can back: a demand past it raises :class:`MemoryError`.

        Args:
            state: A state to measure before a graph starts from it, or ``None``.
                Ignored during graph capture.
            dt: The time step [s] of the collision passes the graph runs. The
                forecast's horizon follows it, and speculative contacts extend a
                measured state by it; without it the state is measured without
                speculative extension.

        Returns:
            True when the buffers grew since this pipeline's last call (here, in
            a :meth:`collide` outside capture, or through another owner of the
            model's pool) and a live graph sized work by their old rows: a
            consumer that reads :attr:`newton.Contacts.rigid_contact_max` while
            capturing (MuJoCo, XPBD, VBD and other solvers), or buffers without
            CUDA virtual memory, whose growth reallocates. Such graphs must be
            recaptured before their next replay.

        .. experimental::
        """
        pool = self._contact_pool
        if capturing(self.device):
            return False
        if dt is not None:
            pool.set_pass_dt(dt)
        if state is not None and pool.growable:
            pool.publish_reach()
            # The measuring pass forecasts too, and the graph's first replays are in transit: the service below and
            # the ones after it hold the rows the forecast bounds until the demand settles.
            pool.request_forecast()
            pool.restart_transit()
            demand = self._measure_rigid_contacts(state, dt=dt)
            pool.size(max(demand, pool.high_water))
            logger.debug(
                "Measured %d rigid contacts in a state, %d forecast; the contact buffers hold %d rows",
                demand,
                pool.forecast,
                pool.ready,
            )
        pool.service()
        # Each pipeline learns of a growth once, also of one another owner of the model's pool caused.
        generation = pool.generation
        grew = generation != self._service_generation
        self._service_generation = generation
        return grew and pool.needs_recapture

    def contact_drops(self) -> ContactDrops:
        """Return the rigid contacts this pipeline's passes dropped for want of rows, and by which worlds.

        Reads pinned memory that every collision pass writes, also in graph
        replays, without synchronizing: it reports the passes that completed.
        A pass drops contacts only when its demand outran the rows the service
        held (see :meth:`service_capacity`), and flags its capacity status as
        well. Nothing clears the record: compare the pass numbers in
        :attr:`ContactDrops.worlds <newton._src.core.contact_pool.ContactDrops.worlds>`
        with the last one read.

        .. experimental::
        """
        return self._contact_pool.drops()

    def _measure_rigid_contacts(self, state: State, *, dt: float | None) -> int:
        """Count the rigid contacts of ``state`` into scratch rows, recording them as demand, and return the count.

        Writes no :class:`~newton.Contacts` of the caller and leaves the
        pipeline's state across passes as it was. The scratch rows grow (and
        the pass reruns) only when the contacts do not fit them.
        """
        scratch = Contacts(
            0,
            0,
            device=self.device,
            per_contact_shape_properties=self.narrow_phase.hydroelastic_sdf is not None,
            rigid_contact_pool=self._contact_pool,
        )
        self._detect_rigid_contacts(state, scratch, dt=0.0 if dt is None else dt)
        wp.copy(self.narrow_phase._buffer_capacity_status, self._capacity_status_before_pass)
        return int(scratch.rigid_contact_count.numpy()[0])

    @property
    def soft_contact_max(self) -> int:
        """Maximum soft contact buffer capacity used by this pipeline."""
        return self._soft_contact_max

    @property
    def soft_contact_margin(self) -> float:
        """Deprecated alias of :attr:`soft_contact_gap`."""
        warnings.warn(
            "CollisionPipeline.soft_contact_margin is deprecated; use soft_contact_gap.",
            NewtonDeprecationWarning,
            stacklevel=2,
        )
        return self.soft_contact_gap

    @soft_contact_margin.setter
    def soft_contact_margin(self, value: float) -> None:
        warnings.warn(
            "CollisionPipeline.soft_contact_margin is deprecated; use soft_contact_gap.",
            NewtonDeprecationWarning,
            stacklevel=2,
        )
        self.soft_contact_gap = value

    @property
    def soft_contact_pair_count(self) -> int:
        """Number of precomputed (particle, shape) pairs launched for soft contacts.

        This is the base of the default ``soft_contact_max``, which additionally reserves
        edge/face headroom when ``enable_rigid_soft_full_surface_contact`` is set. With that
        flag, mesh and convex-mesh shapes are excluded here: their particle contacts come from
        the full-surface mesh pass.
        """
        return self._soft_contact_pair_count

    @property
    def _has_rigid_soft_queries(self) -> bool:
        """Whether any rigid-soft contact pass has candidates, including full-surface passes."""
        return self._soft_contact_tids_size > 0 or self._soft_mesh_contact_data is not None

    @property
    def soft_rigid_contact_pair_count(self) -> int:
        """Deprecated alias of :attr:`soft_contact_pair_count`."""
        warnings.warn(
            "CollisionPipeline.soft_rigid_contact_pair_count is deprecated; use soft_contact_pair_count.",
            NewtonDeprecationWarning,
            stacklevel=2,
        )
        return self.soft_contact_pair_count

    def contacts(self) -> Contacts:
        """
        Allocate and return a new :class:`newton.Contacts` object for this pipeline.

        The returned buffer uses this pipeline's ``requires_grad`` flag (resolved at
        construction from the argument or ``model.requires_grad``).

        Returns:
            A newly allocated contacts buffer sized for this pipeline.

        .. experimental::

            If ``requires_grad`` is true, deprecated rigid-contact distance and
            point compatibility arrays are allocated. New code should allocate
            only the outputs it needs and pass them to
            :func:`newton.eval_rigid_contact_kinematics`.
        """
        detector = self._soft_self_contact_detector
        soft_self_contact = detector is not None
        contacts = Contacts(
            self._contact_pool.ready,
            self.soft_contact_max,
            # Self-contact buffer sizing mirrors the detector configured by
            # init_soft_self_contact(); Contacts ignores it when the flag is False.
            soft_self_contact=soft_self_contact,
            particle_count=self.model.particle_count,
            tri_count=self.model.tri_count,
            edge_count=self.model.edge_count,
            soft_self_contact_vertex_buffer_pre_alloc=(
                detector.vertex_collision_buffer_pre_alloc if soft_self_contact else 0
            ),
            soft_self_contact_edge_buffer_pre_alloc=(
                detector.edge_collision_buffer_pre_alloc if soft_self_contact else 0
            ),
            soft_self_contact_record_triangle_vertices=(
                detector.record_triangle_contacting_vertices if soft_self_contact else False
            ),
            # The per-thread replay array must span every soft candidate-pair thread (particle + edge +
            # face), independent of soft_contact_max (which the caller may set smaller). See E2 fix.
            soft_contact_tids_size=self._soft_contact_tids_size,
            requires_grad=self.requires_grad,
            device=self.model.device,
            per_contact_shape_properties=self.narrow_phase.hydroelastic_sdf is not None,
            requested_attributes=self.model.get_requested_contact_attributes(),
            contact_matching=self._matching_enabled,
            contact_report=self.contact_report,
            rigid_contact_surface_velocity=self._rigid_contact_surface_velocity,
            rigid_contact_pool=self._contact_pool,
        )
        contacts._contact_matching_mode = self.contact_matching
        contacts._rigid_contact_body_pair_reduced = self._reduces_body_pairs
        if self._soft_mesh_contact_data is not None:
            allocate_soft_mesh_contact_buffers(contacts)
            contacts._soft_contact_mesh_data = self._soft_mesh_contact_data
        # Keep scan scratch with the output buffers. The extra count is a zero scan sentinel.
        # Differentiable contacts retain the serial feature loop's existing replay behavior.
        contacts._soft_heightfield_work = None
        if self._soft_heightfield_large_scan and not self.requires_grad:
            count = len(self._soft_heightfield_face_pairs)
            contacts._soft_heightfield_work = (
                wp.zeros(count + 1, dtype=wp.int64, device=self.model.device),
                wp.empty(count + 1, dtype=wp.int64, device=self.model.device),
                wp.empty(count, dtype=wp.uint64, device=self.model.device),
            )
        # The buffer's rows exist now: hold every row their pages cover.
        self._contact_pool.settle()
        # Flag the buffer so solvers that only consume particle contacts can refuse it (see
        # Contacts._enable_rigid_soft_full_surface_contact); edge/face records appear only when this is set.
        contacts._enable_rigid_soft_full_surface_contact = self.enable_rigid_soft_full_surface_contact

        # attach custom attributes with assignment==CONTACT
        self.model._add_custom_attributes(contacts, Model.AttributeAssignment.CONTACT, requires_grad=self.requires_grad)
        return contacts

    def init_soft_self_contact(
        self,
        *,
        margin: float = 0.2,
        gap: float = 0.0,
        rest_shape_exclusion_radius: float = 0.0,
        vertex_buffer_pre_alloc: int = 32,
        edge_buffer_pre_alloc: int = 64,
        edge_edge_parallel_epsilon: float = 1e-5,
        record_triangle_contacting_vertices: bool = False,
        topological_filter_threshold: int = 2,
        external_vertex_filter_map: dict | None = None,
        external_edge_filter_map: dict | None = None,
    ) -> None:
        """Configure soft (cloth) self-contact detection on this pipeline.

        After configuration, :meth:`contacts` allocates the self-contact result
        buffers (:attr:`Contacts.soft_self_contact_data`) on every returned
        buffer, and :meth:`collide` runs vertex-triangle and edge-edge
        detection into them when called with ``soft_self_contact=True``.

        This is the configuration entry point for standalone pipeline use; a
        solver that owns the pipeline calls this internally, seeded from its
        own self-contact parameters.

        Args:
            margin: Self-contact interaction distance [m] (surface offset at
                which force terms begin to act), consumed by solver force terms.
            gap: Additional detection-only distance [m]; detection queries use
                ``margin + gap``, mirroring the ``ShapeConfig.margin`` /
                ``ShapeConfig.gap`` convention.
            rest_shape_exclusion_radius: Pairs closer than this distance [m]
                in the rest shape (``model.particle_q``) are excluded from
                detection — for meshes whose regions are close by design
                (layered cloth, seams). ``0`` disables the filter.
            vertex_buffer_pre_alloc: Per-vertex collision buffer capacity;
                pairs beyond it are silently dropped during detection.
            edge_buffer_pre_alloc: Per-edge collision buffer capacity;
                pairs beyond it are silently dropped during detection.
            edge_edge_parallel_epsilon: Near-parallel edge-pair threshold.
            record_triangle_contacting_vertices: Also record per-triangle
                contacting vertices.
            topological_filter_threshold: Ring distance under which candidate
                pairs are filtered out.
            external_vertex_filter_map: Extra vertex-triangle exclusions.
            external_edge_filter_map: Extra edge-edge exclusions.
        """
        if margin < 0.0:
            raise ValueError(f"soft self-contact margin must be >= 0, got {margin}")
        if gap < 0.0:
            raise ValueError(f"soft self-contact gap must be >= 0, got {gap}")
        if rest_shape_exclusion_radius < 0.0:
            raise ValueError(f"rest_shape_exclusion_radius must be >= 0, got {rest_shape_exclusion_radius}")
        if self.model.tri_count == 0:
            raise ValueError("init_soft_self_contact() requires a model with triangles (cloth/soft mesh).")
        self.soft_self_contact_margin = margin
        self.soft_self_contact_gap = gap
        self.soft_self_contact_rest_shape_exclusion_radius = rest_shape_exclusion_radius
        # The explicit opt-in is what creates the detector (its BVHs are built
        # from model.particle_q); the result struct stays unallocated until the
        # first Contacts buffer is bound. Re-configuring rebuilds the detector.
        self._soft_self_contact_detector = TriMeshCollisionDetector(
            self.model,
            record_triangle_contacting_vertices=record_triangle_contacting_vertices,
            vertex_collision_buffer_pre_alloc=vertex_buffer_pre_alloc,
            edge_collision_buffer_pre_alloc=edge_buffer_pre_alloc,
            edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
            topological_contact_filter_threshold=topological_filter_threshold,
            external_vertex_triangle_filtering_map=external_vertex_filter_map,
            external_edge_edge_filtering_map=external_edge_filter_map,
        )
        if self._soft_mesh_contact_data is not None:
            self._soft_mesh_contact_data.detector = self._soft_self_contact_detector

    def set_collision_detection_range(
        self,
        *,
        soft_contact_gap: float | None = None,
        soft_self_contact_margin: float | None = None,
        soft_self_contact_gap: float | None = None,
    ) -> None:
        """Update the detection ranges consumed by :meth:`collide`.

        Only the values provided are changed; ``None`` keeps the current
        setting, and changes take effect at the next :meth:`collide` call.
        Rigid (shape-shape) ranges are per-shape model data
        (:attr:`Model.shape_margin`, :attr:`Model.shape_gap`) and are not
        covered here. A solver that owns the pipeline drives self-contact
        detection from its own parameters, so this setter affects standalone
        :meth:`collide` use.

        Args:
            soft_contact_gap: Detection-only distance [m] added to the
                per-particle radius for particle-shape contact queries.
            soft_self_contact_margin: Self-contact interaction distance [m];
                requires :meth:`init_soft_self_contact` to have been called.
            soft_self_contact_gap: Additional detection-only self-contact
                distance [m] (queries use ``margin + gap``); requires
                :meth:`init_soft_self_contact` to have been called.
        """
        for name, value in (
            ("soft_contact_gap", soft_contact_gap),
            ("soft_self_contact_margin", soft_self_contact_margin),
            ("soft_self_contact_gap", soft_self_contact_gap),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if soft_self_contact_margin is not None or soft_self_contact_gap is not None:
            self._ensure_soft_self_contact_detector()
        if soft_contact_gap is not None:
            self.soft_contact_gap = soft_contact_gap
        if soft_self_contact_margin is not None:
            self.soft_self_contact_margin = soft_self_contact_margin
        if soft_self_contact_gap is not None:
            self.soft_self_contact_gap = soft_self_contact_gap

    def _ensure_soft_self_contact_detector(self) -> TriMeshCollisionDetector:
        """Return the shared detector created by :meth:`init_soft_self_contact`."""
        if self._soft_self_contact_detector is None:
            raise ValueError("configure the pipeline with init_soft_self_contact() first.")
        return self._soft_self_contact_detector

    def _get_soft_self_contact_detector(self, contacts: Contacts) -> TriMeshCollisionDetector:
        """Return the shared detector re-pointed at ``contacts.soft_self_contact_data``."""
        data = contacts.soft_self_contact_data
        if data is None:
            raise ValueError(
                "This Contacts buffer has no soft_self_contact_data; allocate it with "
                "CollisionPipeline.contacts() after init_soft_self_contact()."
            )
        detector = self._ensure_soft_self_contact_detector()
        if detector.collision_info is not data:
            detector._bind_external_buffers(data)
        return detector

    def refit_soft_self_contact_bvh(self, new_pos: wp.array[wp.vec3], *, rebuild: bool = False) -> None:
        """Refit (or fully rebuild) the soft self-contact BVHs to ``new_pos``.

        :meth:`collide` automatically refits before self-contact detection.
        Call this method directly to update the trees without detecting, or
        pass ``rebuild=True`` to rebuild them from scratch when repeated
        refitting has degraded their quality under large deformation.

        Args:
            new_pos: Particle positions [m] to fit the BVHs to, e.g.
                ``state.particle_q``.
            rebuild: Rebuild the trees instead of refitting them.
        """
        detector = self._ensure_soft_self_contact_detector()
        if rebuild:
            detector.rebuild(new_pos)
        else:
            detector.refit(new_pos)

    def _detect_soft_self_contact(self, particle_q: wp.array[wp.vec3], contacts: Contacts) -> None:
        """Detect tri-mesh self-contact into ``contacts`` at ``particle_q``."""
        detector = self._get_soft_self_contact_detector(contacts)
        detector.refit(particle_q)
        query_radius = self.soft_self_contact_margin + self.soft_self_contact_gap
        detector.vertex_triangle_collision_detection(
            query_radius,
            min_query_radius=self.soft_self_contact_rest_shape_exclusion_radius,
            min_distance_filtering_ref_pos=self.model.particle_q,
        )
        detector.edge_edge_collision_detection(
            query_radius,
            min_query_radius=self.soft_self_contact_rest_shape_exclusion_radius,
            min_distance_filtering_ref_pos=self.model.particle_q,
        )

    def reset_contact_matching(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear all or reset-selected previous-frame contact history.

        Masked selections accumulate until the next :meth:`collide` call
        consumes them. Call the solver's reset method too when starting a new episode.

        .. experimental::

        Args:
            world_mask: Optional one-dimensional Warp boolean mask on the
                model device with shape ``(model.world_count + 1,)``. The final
                entry selects global entities whose world index is ``-1``. If
                ``None``, clear all previous-frame contact history immediately.
        """
        world_mask = normalize_reset_world_mask(
            world_mask,
            world_count=int(self.model.world_count),
            device=self.model.device,
        )
        if self._contact_matcher is not None:
            self._contact_matcher.reset(world_mask)
        if self._body_pair_reducer is not None:
            if world_mask is None:
                self._body_pair_reducer.reset_history()
            else:
                wp.launch(
                    _reduction_reset_mask_from_matching,
                    dim=self._reduction_reset_mask.shape[0],
                    inputs=[world_mask, int(self.model.world_count), self._reduction_reset_mask],
                    device=self.device,
                )
                self._body_pair_reducer.reset_history(self._reduction_reset_mask)
        if self.narrow_phase._coherent_cache is not None:
            self.narrow_phase._coherent_cache.reset(world_mask)

    @staticmethod
    def _build_excluded_pairs(model: Model) -> wp.array[wp.vec2i] | None:
        sorted_pairs = model.shape_collision_filter_pairs_array()
        if sorted_pairs.shape[0] == 0:
            return None
        return wp.array(
            sorted_pairs,
            dtype=wp.vec2i,
            device=model.device,
        )

    class _WeldTopology:
        """Host copy of the joint topology that welds bodies to the world.

        A joint without degrees of freedom welds its child to its parent, whatever its type: a
        fixed joint, or a D6 joint without axes, which the USD importer makes of a generic
        root joint. A body is welded when it has a parent joint and every joint that names it
        as the child welds it to the world or to another welded body, so it has no degree of
        freedom of its own. A shape is structurally immobile when it is static or belongs to a
        welded body. Joints and shape ownership are fixed once a model is finalized and are
        read here once; body flags and joint enables are runtime state, which
        :meth:`read_body_flags` and :meth:`read_joint_enables` copy from the device.
        """

        def __init__(self, model: Model):
            self.body_count = int(model.body_count)
            empty = np.empty(0, dtype=np.int32)
            self.shape_body = model.shape_body.numpy() if model.shape_count else empty
            joint_count = int(model.joint_count) if self.body_count > 0 else 0
            parent = model.joint_parent.numpy() if joint_count else empty
            child = model.joint_child.numpy() if joint_count else empty
            # Judge the DOF count, not the type; joint_qd_start ends with a sentinel.
            dofs = np.diff(model.joint_qd_start.numpy()[: joint_count + 1]) if joint_count else empty
            attached = (child >= 0) & (child < self.body_count)
            self.weld_joints = np.flatnonzero(attached & (dofs == 0))
            self.weld_parent = parent[self.weld_joints]
            self.weld_child = child[self.weld_joints]
            self.parent_joints = np.bincount(child[attached], minlength=self.body_count)

        def read_joint_enables(self, model: Model) -> np.ndarray | None:
            """Return whether each joint of ``weld_joints`` is enabled, or ``None`` when all are."""
            if model.joint_enabled is None or not len(self.weld_joints):
                return None
            return model.joint_enabled.numpy()[self.weld_joints].astype(bool)

        def read_body_flags(self, model: Model) -> np.ndarray | None:
            """Return a host copy of the body flags, or ``None`` without them."""
            if model.body_flags is None or not self.body_count:
                return None
            # numpy() aliases the memory of a CPU array, which the next refresh compares against.
            return np.array(model.body_flags.numpy())

        def kinematic_shapes(self, flags: np.ndarray | None) -> np.ndarray:
            """Return a host mask of the shapes of kinematic bodies under :meth:`read_body_flags`."""
            kinematic = np.zeros(len(self.shape_body), dtype=bool)
            if flags is not None:
                on_body = self.shape_body >= 0
                kinematic[on_body] = (flags[self.shape_body[on_body]] & int(BodyFlags.KINEMATIC)) != 0
            return kinematic

        def welded_bodies(self, enabled: np.ndarray | None, flags: np.ndarray | None) -> np.ndarray:
            """Return a host mask of the welded bodies.

            Args:
                enabled: :meth:`read_joint_enables`; a disabled joint does not weld.
                flags: :meth:`read_body_flags`; only a dynamic body welds, because the user
                    can move a kinematic body and everything welded to it. With both ``None``
                    the result bounds the welded set over every state of the two arrays.
            """
            weld_parent, weld_child = self.weld_parent, self.weld_child
            if enabled is not None:
                weld_parent, weld_child = weld_parent[enabled], weld_child[enabled]
            eligible = self.parent_joints > 0
            if flags is not None:
                eligible &= ((flags & int(BodyFlags.DYNAMIC)) != 0) & ((flags & int(BodyFlags.KINEMATIC)) == 0)
            # Every parent joint must weld; a body with another parent joint keeps its freedom.
            eligible &= np.bincount(weld_child, minlength=self.body_count) == self.parent_joints
            welded = np.zeros(self.body_count, dtype=bool)
            from_world = weld_parent < 0
            parent_index = np.where(from_world, 0, weld_parent)
            # Weld outward from the world, one joint per pass, until the set stops growing.
            while True:
                anchored = from_world | welded[parent_index]
                grown = eligible & (np.bincount(weld_child[anchored], minlength=self.body_count) == self.parent_joints)
                if np.array_equal(grown, welded):
                    return welded
                welded = grown

        def immobile_shapes(self, welded: np.ndarray) -> np.ndarray:
            """Return a host mask of the static shapes and the shapes of ``welded`` bodies."""
            immobile = self.shape_body < 0
            on_body = ~immobile
            immobile[on_body] = welded[self.shape_body[on_body]]
            return immobile

    def _configure_immobile_pairs(self, model: Model, requested: bool, source: str) -> None:
        """Decide once, at construction, whether the broad phase drops pairs of structurally immobile shapes.

        The filter is installed only when it can change a candidate pair in some state of the
        body flags and joint enables, so other models keep the unfiltered broad-phase
        arguments: with ``include_static_kinematic_pairs=True``, a pair that can join two
        immobile shapes; with ``False``, where a welded body also reads as static against
        kinematic ones and any body can become kinematic, a pair that touches a shape that can
        be welded. That makes the composed rule independent of unrelated pairs. Candidates
        come from the explicit pairs or, for ``"nxn"`` and ``"sap"``, from the model's pair
        list, which applies the same world, group and filter-pair rules as those broad phases.
        """
        self._weld_topology = topology = self._WeldTopology(model)
        self._colliding_shapes = _shape_collide_mask(model, len(topology.shape_body))
        self._immobile_shape_body = None
        self._no_body_flags = None
        self._immobile_inputs = None
        # An explicit list is scanned; nxn and sap pipelines count the model's pairs by shape mask instead, never
        # enumerating them (_shape_contact_pair_count_for_mask).
        pairs = self.shape_pairs_filtered if self.broad_phase_mode == "explicit" else None
        self._immobile_pair_counts = pairs is None and model.shape_contact_pair_count > 0
        # Topology alone bounds the immobile shapes over every body-flag and joint-enable state.
        reachable = topology.immobile_shapes(topology.welded_bodies(None, None))
        weldable = None if self.include_static_kinematic_pairs else reachable & (topology.shape_body >= 0)
        if self._immobile_pair_counts:
            self._immobile_pair_candidates = None
            if weldable is None:
                possible = _shape_contact_pair_count_for_mask(model, reachable) > 0
            else:
                every = np.ones(len(weldable), dtype=bool)
                unweldable = _shape_contact_pair_count_for_mask(model, ~weldable)
                possible = _shape_contact_pair_count_for_mask(model, every) > unweldable
        elif pairs is None:
            self._immobile_pair_candidates = None
            colliding = self._colliding_shapes
            if weldable is None:
                possible = int(np.count_nonzero(reachable & colliding)) > 1
            else:
                possible = bool(np.any(weldable & colliding)) and int(np.count_nonzero(colliding)) > 1
        else:
            candidates = []
            for block in _iter_pair_blocks(pairs):
                listed = reachable[block[:, 0]] & reachable[block[:, 1]]
                if weldable is not None:
                    listed |= weldable[block[:, 0]] | weldable[block[:, 1]]
                candidates.append(block[listed])
            candidates = np.concatenate(candidates) if candidates else np.empty((0, 2), dtype=np.int32)
            self._immobile_pair_candidates = candidates
            if weldable is None:
                possible = len(candidates) > 0
            else:
                possible = bool(np.any(weldable[candidates[:, 0]] | weldable[candidates[:, 1]]))
        if requested and possible:
            self._immobile_shape_body = wp.zeros(len(topology.shape_body), dtype=wp.int32, device=self.device)
            self._no_body_flags = wp.empty(0, dtype=wp.int32, device=self.device)
        if not requested:
            setting = "include_immobile_pairs=True" if source == "argument" else "NEWTON_BROAD_PHASE_IMMOBILE_FILTER=0"
            reason = f"Off: {setting} keeps pairs whose shapes are both static or welded to the world."
        elif not possible:
            reason = (
                "No candidate pair can join two shapes that are static or welded to the world."
                if weldable is None
                else "No candidate pair can touch a body welded to the world, and include_static_kinematic_pairs=False "
                "already drops static-static pairs."
            )
        else:
            setting = {
                "default": "Default",
                "environment": "NEWTON_BROAD_PHASE_IMMOBILE_FILTER=1",
                "argument": "include_immobile_pairs=False",
            }[source]
            dropped = (
                "both static or welded to the world"
                if weldable is None
                else "each static, welded to the world or kinematic (include_static_kinematic_pairs=False)"
            )
            reason = f"{setting}: the broad phase drops pairs whose shapes are {dropped}."
        self._immobile_pairs = {"enabled": self._immobile_shape_body is not None, "source": source, "reason": reason}
        self._refresh_immobile_pairs()

    def _refresh_immobile_pairs(self, *, body: bool = True, joint: bool = True) -> None:
        """Recompute the immobile shapes from the model's body flags (``body``) and joint enables (``joint``).

        Each named array is copied from the device; when neither differs from the previous
        refresh, nothing is recomputed or written.
        """
        topology = self._weld_topology
        previous = self._immobile_inputs
        enabled = topology.read_joint_enables(self.model) if joint or previous is None else previous[0]
        flags = topology.read_body_flags(self.model) if body or previous is None else previous[1]
        if previous is not None and all(
            np.array_equal(new, old) if new is not None and old is not None else new is old
            for new, old in zip((enabled, flags), previous, strict=True)
        ):
            return
        self._immobile_inputs = (enabled, flags)
        welded = topology.welded_bodies(enabled, flags)
        immobile = topology.immobile_shapes(welded)
        if self._immobile_shape_body is not None:
            # Immobile shapes read as static in the broad phase's immovable-pair test.
            self._immobile_shape_body.assign(np.where(immobile, -1, topology.shape_body).astype(np.int32))
        pairs = self._immobile_pair_candidates
        listed = excluded = None
        if self._immobile_pair_counts:
            listed = excluded = _shape_contact_pair_count_for_mask(self.model, immobile)
            if not self.include_static_kinematic_pairs:
                # The test drops static and kinematic pairs anyway; welded shapes join them.
                immovable = (topology.shape_body < 0) | topology.kinematic_shapes(flags)
                excluded = _shape_contact_pair_count_for_mask(
                    self.model, immovable | immobile
                ) - _shape_contact_pair_count_for_mask(self.model, immovable)
        elif pairs is not None:
            first, second = pairs[:, 0], pairs[:, 1]
            listed = excluded = int(np.count_nonzero(immobile[first] & immobile[second]))
            if not self.include_static_kinematic_pairs:
                # The test drops static and kinematic pairs anyway; welded shapes join them.
                immovable = (topology.shape_body < 0) | topology.kinematic_shapes(flags)
                masked = immovable | immobile
                dropped = masked[first] & masked[second] & ~(immovable[first] & immovable[second])
                excluded = int(np.count_nonzero(dropped))
        self._immobile_pairs.update(
            welded_bodies=int(np.count_nonzero(welded)),
            immobile_shapes=int(np.count_nonzero(immobile & self._colliding_shapes)),
            listed_pairs=listed,
            excluded_pairs=excluded if self._immobile_shape_body is not None else 0,
        )

    def _broad_phase_pair_filter(self, model: Model) -> dict:
        """Return the shape metadata of the broad phase's immovable-pair test, as launch keywords."""
        if self._immobile_shape_body is None:
            return {
                "shape_body": model.shape_body,
                "body_flags": model.body_flags,
                "include_static_kinematic_pairs": self.include_static_kinematic_pairs,
            }
        # Immobile shapes read as static, so the test drops pairs of two of them. Without
        # body flags it cannot tell kinematic bodies apart and keeps every other pair.
        return {
            "shape_body": self._immobile_shape_body,
            "body_flags": self._no_body_flags if self.include_static_kinematic_pairs else model.body_flags,
            "include_static_kinematic_pairs": False,
        }

    def _detect_rigid_contacts(self, state: State, contacts: Contacts, *, dt: float | None) -> None:
        """Write and count the rigid contacts of ``state`` in ``contacts``; outside capture, grow until they fit.

        Every rigid writer counts its contacts, also those past the rows, and
        drops the rows past the ready rows it reads on the device. The count is
        then recorded as the demand that grows the buffer's pool (this
        pipeline's, for its own :meth:`contacts`) and clamped to the rows, with
        the overflow flagged in the capacity status. Outside graph capture the
        count is one synchronization away: a pass that filled the rows grows
        them and runs again before anything reads them, so no contact is
        dropped, and its capacity flags do not outlive it. In a capture, the
        pass then records a device graph updater, so the launches that consume
        its contacts follow the count on every replay; graph replays grow
        through :meth:`service_capacity` and flag the replay that overflowed.
        """
        pool = contacts._rigid_contact_pool
        grows = pool.growable and pool.ready_rows is not None and not capturing(self.device)
        if grows:
            wp.copy(self._capacity_status_before_pass, self.narrow_phase._buffer_capacity_status)
        while True:
            # Counter zeroing and the generation bump are fused into the AABB launch; clear_buffers
            # (debug) also zeroes the rows, without a second generation bump.
            if contacts.clear_buffers:
                contacts.clear(bump_generation=False)
            limit = pool.ready_rows
            if limit is None:
                # A pool created during a capture holds its rows until a recapture: publish them here.
                limit = self._contact_limit
                limit.fill_(contacts._rigid_contact_rows)
            # A pass that grows its rows and runs again drops nothing, so only a pass that cannot records drops.
            # In a graph the record picks the forecasting passes on the device; outside one a host asks for one.
            in_graph = capturing(self.device)
            requested = not in_graph and pool.take_forecast_request()
            self._launch_rigid_detection(
                state, contacts, limit, dt=dt, latch_drops=not grows, forecast_requested=requested
            )
            pool.record_demand(
                contacts.rigid_contact_count,
                limit,
                self.narrow_phase._buffer_capacity_status,
                drops=not grows,
                beat=in_graph,
            )
            if not (grows and pool.fit()):
                break
            wp.copy(self.narrow_phase._buffer_capacity_status, self._capacity_status_before_pass)
        pool.begin_updates()

    def _launch_rigid_detection(
        self,
        state: State,
        contacts: Contacts,
        limit: wp.array,
        *,
        dt: float | None,
        latch_drops: bool = True,
        forecast_requested: bool = False,
    ) -> None:
        """Launch the broad and narrow phases, and the terrain law's writes, into ``contacts``' rigid rows.

        ``limit`` holds the rows the writers may fill, on the device; with ``latch_drops`` a writer that drops a
        contact past them latches its world in the pool's status. The launches also forecast the demand on a
        forecasting pass (:attr:`~newton._src.core.contact_pool.ContactPool.forecast_on`): in a graph the device
        flag picks it, outside one ``forecast_requested`` says the host asked for it.
        """
        model = self.model
        if self._speculative_enabled:
            if dt is None:
                raise ValueError("dt must be provided when speculative contacts are enabled")
            collision_update_dt = dt
            if not np.isfinite(collision_update_dt) or collision_update_dt < 0.0:
                raise ValueError(f"dt must be a non-negative finite number, got {collision_update_dt!r}")
            max_speculative_extension = float(self.speculative_contact_gap_max)
            speculative_active = collision_update_dt > 0.0 and max_speculative_extension > 0.0
            search_gap = self._shape_search_gap if speculative_active else model.shape_gap
        else:
            collision_update_dt = 0.0
            max_speculative_extension = 0.0
            speculative_active = False
            search_gap = model.shape_gap

        # Rigid contact detection -- broad phase + narrow phase.
        # These kernels hardcode record_tape=False internally so they are
        # never captured on an active wp.Tape.  The differentiable
        # augmentation and soft-contact kernels that follow are tape-safe
        # and recorded normally.

        # Eager calls copy the sticky rescale flag to the host and raise on a copy from an earlier call; never syncs.
        check_unit_scale_sdf = self._shape_unit_scale_sdf.shape[0] > 0 and not wp.get_device(self.device).is_capturing
        if check_unit_scale_sdf and self._unit_scale_sdf_rescaled_host.numpy()[0]:
            raise RuntimeError("Rescaled a unit-scale texture SDF shape; set dynamic_shape_scale=True (slower).")

        # The forecast this pass adds into on a forecasting pass: none without velocities to read.
        forecast = self._forecast
        body_qd = state.body_qd
        if forecast is None or body_qd is None or contacts._rigid_contact_pool is not self._contact_pool:
            forecast = None
            body_qd = self._no_velocities

        # Compute AABBs for all shapes, zero counters, bump generation.
        # Fuses contacts.clear() + broad_phase_pair_count.zero_() + AABB update.
        note_contact_frame_write(contacts.contact_generation)
        aabb_kernel = compute_shape_aabbs
        aabb_dim = model.shape_count
        aabb_index_inputs = []
        if self._pair_shape_prep:
            aabb_kernel = _compute_pair_shape_aabbs
            aabb_dim = max(1, len(self.prepared_shape_indices)) if model.shape_count else 0
            aabb_index_inputs = [self.prepared_shape_indices]
        wp.launch(
            kernel=aabb_kernel,
            dim=aabb_dim,
            inputs=[
                *aabb_index_inputs,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_collision_radius,
                model.shape_source_ptr,
                model.shape_margin,
                model.shape_gap,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                model.shape_flags,
                self._shape_variant_row,
                contacts.contact_counters,
                contacts.contact_generation,
                self.broad_phase_pair_count,
                contacts.contact_counters.shape[0],
                body_qd,
                _NO_FORECAST if forecast is None else forecast,
            ],
            outputs=[
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                self.geom_data,
                self.geom_transform,
            ],
            device=self.device,
            record_tape=False,
        )

        if speculative_active:
            wp.launch(
                kernel=compute_shape_velocities,
                dim=model.shape_count,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    model.body_com,
                    model.shape_body,
                    model.shape_transform,
                    model.shape_collision_aabb_lower,
                    model.shape_collision_aabb_upper,
                    model.shape_collision_radius,
                    model.shape_gap,
                    collision_update_dt,
                    max_speculative_extension,
                ],
                outputs=[
                    self._shape_linear_velocity,
                    self._shape_angular_velocity,
                    self._shape_search_gap,
                    self._shape_displacement,
                    self.narrow_phase.shape_aabb_lower,
                    self.narrow_phase.shape_aabb_upper,
                ],
                device=self.device,
                record_tape=False,
            )

        # Run broad phase (AABBs are already expanded by effective gaps, so pass None). Every
        # broad phase applies one immovable-pair test, which drops structurally immobile pairs.
        pair_filter = self._broad_phase_pair_filter(model)
        # Expert broad phases of earlier releases take no displacement: pass it only to a speculative pass.
        speculative_kwargs = {"shape_displacement": self._shape_displacement} if speculative_active else {}
        if isinstance(self.broad_phase, BroadPhaseAllPairs):
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,  # AABBs are pre-expanded, no additional margin needed
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                **pair_filter,
                device=self.device,
                filter_pairs=self.shape_pairs_excluded,
                num_filter_pairs=self.shape_pairs_excluded_count,
                skip_count_zero=True,  # Already zeroed by compute_shape_aabbs
                **speculative_kwargs,
                forecast=forecast,
            )
        elif isinstance(self.broad_phase, BroadPhaseSAP):
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,  # AABBs are pre-expanded, no additional margin needed
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                **pair_filter,
                device=self.device,
                filter_pairs=self.shape_pairs_excluded,
                num_filter_pairs=self.shape_pairs_excluded_count,
                skip_count_zero=True,  # Already zeroed by compute_shape_aabbs
                **speculative_kwargs,
                **({"sort_axis_displacement_limit": max_speculative_extension} if speculative_active else {}),
                forecast=forecast,
            )
        elif self._pair_templates is not None:
            # Same pairs as the flat list (verified at construction), tested per world template. A forecasting pass
            # launches the kernel's forecasting variant, every other pass the plain one: in a graph Warp captures a
            # conditional node picks it on the pool's flag, in a capture of its own the variant reads the flag.
            def test_templates(forecasting: bool = False):
                self._pair_templates.launch(
                    self.narrow_phase.shape_aabb_lower,
                    self.narrow_phase.shape_aabb_upper,
                    self.shape_pairs_filtered,
                    self.broad_phase_shape_pairs,
                    self.broad_phase_pair_count,
                    **pair_filter,
                    forecast=forecast if forecasting else None,
                )

            if forecast is None or not capturing(self.device):
                test_templates(forecast is not None and forecast_requested)
            elif self.device.is_cuda and _capture_owner(self.device) is None:
                test_templates(True)
            else:
                wp.capture_if(
                    self._contact_pool.forecast_on,
                    on_true=lambda: test_templates(True),
                    on_false=test_templates,
                )
        else:  # BroadPhaseExplicit
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,  # AABBs are pre-expanded, no additional margin needed
                self.shape_pairs_filtered,
                len(self.shape_pairs_filtered),
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                **pair_filter,
                device=self.device,
                skip_count_zero=True,  # Already zeroed by compute_shape_aabbs
                shape_displacement=self._shape_displacement if speculative_active else None,
                forecast=forecast,
            )

        # Create ContactWriterData struct for custom contact writing
        writer_data = ContactWriterData()
        writer_data.contact_max = contacts._rigid_contact_pool.bound
        writer_data.contact_limit = limit
        writer_data.shape_index_bits = self._contact_sort_shape_index_bits
        writer_data.sub_key_bits = self._contact_sort_sub_key_bits
        if latch_drops and contacts._rigid_contact_pool.dropped_worlds is not None:
            writer_data.dropped_world = contacts._rigid_contact_pool.dropped_worlds
            writer_data.shape_world = model.shape_world
            writer_data.pass_count = contacts._rigid_contact_pool._passes
        writer_data.body_q = state.body_q
        writer_data.shape_body = model.shape_body
        writer_data.shape_gap = model.shape_gap
        writer_data.contact_count = contacts.rigid_contact_count
        writer_data.out_shape0 = contacts.rigid_contact_shape0
        writer_data.out_shape1 = contacts.rigid_contact_shape1
        writer_data.out_point0 = contacts.rigid_contact_point0
        writer_data.out_point1 = contacts.rigid_contact_point1
        writer_data.out_offset0 = contacts.rigid_contact_offset0
        writer_data.out_offset1 = contacts.rigid_contact_offset1
        writer_data.out_normal = contacts.rigid_contact_normal
        writer_data.out_margin0 = contacts.rigid_contact_margin0
        writer_data.out_margin1 = contacts.rigid_contact_margin1
        writer_data.out_tids = contacts.rigid_contact_tids

        writer_data.out_stiffness = contacts.rigid_contact_stiffness
        writer_data.out_damping = contacts.rigid_contact_damping
        writer_data.out_friction = contacts.rigid_contact_friction
        if self.deterministic and contacts._rigid_contact_rows != self._sort_key_array.shape[0]:
            raise ValueError(
                f"Contacts buffer capacity ({contacts._rigid_contact_rows}) does not match the "
                f"deterministic sort buffer size ({self._sort_key_array.shape[0]}). "
                f"The sorter operates over fixed-capacity buffers for CUDA graph capture "
                f"compatibility, so the sizes must match exactly. Use CollisionPipeline.contacts()."
            )
        writer_data.out_sort_key = self._sort_key_array
        writer_data.shape_transform = self.geom_transform
        writer_data.shape_linear_velocity = self._shape_linear_velocity
        writer_data.shape_angular_velocity = self._shape_angular_velocity
        writer_data.collision_update_dt = collision_update_dt
        writer_data.max_speculative_extension = max_speculative_extension
        # Run narrow phase with custom contact writer (writes directly to Contacts format)
        narrow_phase_extension_kwargs = {}
        if type(self.narrow_phase).launch_custom_write is NarrowPhase.launch_custom_write:
            narrow_phase_extension_kwargs.update(
                mesh_edge_centers=model._mesh_edge_centers,
                mesh_edge_halves=model._mesh_edge_halves,
                mesh_edge_order=model._mesh_edge_order,
                mesh_edge_clusters=model._mesh_edge_clusters,
                mesh_edge_cluster_reach=model._mesh_edge_cluster_reach,
                shape_support_data=model._shape_support_data,
                support_lut=model._convex_support_lut,
                support_vertex_offsets=model._convex_support_vertex_offsets,
                support_neighbors=model._convex_support_neighbors,
                hydroelastic_shape_sdf_data_prepared=self._hydro_shape_sdf_data_prepared,
                # The demand record flags a pass whose contacts outnumbered the rows.
                verify_contacts=False,
            )
        if self._speculative_enabled:
            narrow_phase_extension_kwargs.update(
                shape_base_gap=model.shape_gap,
                shape_linear_velocity=self._shape_linear_velocity,
                shape_angular_velocity=self._shape_angular_velocity,
                collision_update_dt=collision_update_dt,
                max_speculative_extension=max_speculative_extension,
            )
        self.narrow_phase.launch_custom_write(
            candidate_pair=self.broad_phase_shape_pairs,
            candidate_pair_count=self.broad_phase_pair_count,
            shape_types=model.shape_type,
            shape_data=self.geom_data,
            shape_transform=self.geom_transform,
            shape_source=model.shape_source_ptr,
            shape_mesh_properties=model._shape_mesh_properties,
            mesh_boundary_flags=model._mesh_boundary_flags,
            shape_mesh_boundary_start=model._shape_mesh_boundary_start,
            shape_sdf_index=model._shape_sdf_index,
            texture_sdf_data=model._texture_sdf_data,
            shape_gap=search_gap,
            shape_collision_radius=model.shape_collision_radius,
            shape_flags=model.shape_flags,
            shape_collision_aabb_lower=model.shape_collision_aabb_lower,
            shape_collision_aabb_upper=model.shape_collision_aabb_upper,
            shape_voxel_resolution=self.narrow_phase.shape_voxel_resolution,
            shape_interior_point=model._shape_interior_point,
            shape_heightfield_index=model.shape_heightfield_index,
            heightfield_data=model.heightfield_data,
            heightfield_elevations=model.heightfield_elevations,
            mesh_edge_indices=model._mesh_edge_indices,
            shape_edge_range=model._shape_edge_range,
            writer_data=writer_data,
            device=self.device,
            **narrow_phase_extension_kwargs,
        )
        if check_unit_scale_sdf:
            wp.copy(self._unit_scale_sdf_rescaled_host, self._unit_scale_sdf_rescaled)

        if self._terrain_law_shadow is not None:
            # The terrain law's rows, under the broad phase's pair policy (in shadow, beside the terrain route).
            self._terrain_law_shadow.launch(
                self.geom_transform,
                self.geom_data,
                shape_type=model.shape_type,
                shape_gap=model.shape_gap,
                **pair_filter,
            )
            if self._terrain_law_writes:
                # The pairs the narrow phase dropped: their rows become contacts, before sorting and reduction.
                self._terrain_law_shadow.write_contacts(
                    self._terrain_law_listed, model.shape_type, self.geom_data, write_contact, writer_data
                )

    def collide(
        self,
        state: State,
        contacts: Contacts,
        *,
        soft_contact_margin: float | None = None,
        soft_self_contact: bool = False,
        dt: float | None = None,
    ):
        """Run the collision pipeline using NarrowPhase.

        Safe to call inside a :class:`wp.Tape` context.  The non-differentiable
        broad-phase and narrow-phase kernels are launched with tape recording
        hardcoded ``record_tape=False`` internally.  The differentiable kernels
        (soft-contact generation and rigid-contact augmentation) are recorded on
        the tape so that gradients flow through ``state.body_q`` and
        ``state.particle_q``.

        For backward compatibility, when ``requires_grad=True`` the deprecated
        ``contacts.rigid_contact_diff_*`` arrays are populated by a lightweight
        augmentation kernel. New code should call
        :func:`newton.eval_rigid_contact_kinematics` explicitly
        after collision detection to reconstruct only the quantities it needs.

        .. experimental::

            This rigid-contact gradient path may change without prior notice.
            Usefulness and numerical behaviour are still being assessed across
            real-world scenarios.

        Args:
            state: The current simulation state.
            contacts: The contacts buffer to populate (will be cleared first).
            soft_contact_margin: Deprecated; set ``soft_contact_gap`` on the
                :class:`CollisionPipeline` constructor instead. When not
                ``None``, the value is still honored for this call and a
                :class:`DeprecationWarning` is emitted.
            soft_self_contact: Also run soft (cloth) self-contact detection
                into ``contacts.soft_self_contact_data``. Requires
                :meth:`init_soft_self_contact` to have been called. The
                self-contact BVHs are refitted to ``state.particle_q`` before
                detection. Use :meth:`refit_soft_self_contact_bvh` directly
                when an explicit full rebuild is needed.
            dt: Collision-update horizon [s]. Required when speculative
                contacts are enabled. ``0.0`` disables velocity adaptation for
                this call. Ignored when speculative contacts are disabled. See
                :ref:`Speculative contacts <speculative-contacts>`.
        """
        if contacts.rigid_contact_body_pair_reduced != self._reduces_body_pairs:
            raise ValueError(
                "body-pair-reduced Contacts buffers and pipelines with body_pair_reduction must be used together; "
                "allocate the buffer with this pipeline's contacts()"
            )
        if self._reduces_body_pairs and (
            contacts._rigid_contact_pool is not self._contact_pool or str(contacts.device) != str(self.device)
        ):
            raise ValueError("body_pair_reduction requires a Contacts buffer allocated by this pipeline's contacts()")
        if self._reduces_body_pairs and self.device.is_cuda and self.device.is_capturing:
            # Replay would repeat a buffer switch's history reset: a pipeline's first buffer needs none,
            # another one is collided once outside capture before it is captured.
            last = self._last_contacts_ref
            if last is not None and last() is not contacts:
                raise RuntimeError(
                    "body_pair_reduction: collide this Contacts buffer once outside capture before capturing "
                    "it; it replaces the buffer the reducer's history belongs to"
                )
        # Keep the buffer's full-surface capability marker in sync with this pipeline on every call.
        # collide() may be handed a Contacts created elsewhere (or by a flag-off pipeline); the edge/
        # face passes below would otherwise populate records while the marker stayed False, so
        # particle-only solvers (XPBD, semi-implicit, Style3D) would not raise and would silently
        # ignore them. Mirrors the assignment in CollisionPipeline.contacts().
        contacts._enable_rigid_soft_full_surface_contact = self.enable_rigid_soft_full_surface_contact

        # A reused buffer can retain mesh tags from a full-surface pipeline. The mesh pass
        # resets them itself; without that pass, replacement particle rows need invalid tags.
        if self._soft_mesh_contact_data is None and contacts._soft_contact_mesh_features is not None:
            contacts._soft_contact_mesh_features.view(wp.int32).fill_(-1)
        if not contacts.clear_buffers and (
            self._soft_mesh_contact_data is not None or contacts._soft_contact_mesh_features is not None
        ):
            # Non-mesh emitters must not inherit rigid primitive IDs from mesh rows.
            # Fresh particle-only buffers already contain -1, and clear() resets them.
            # Scalar fills preserve every -1 component and avoid slower vector tiling.
            contacts.soft_contact_rigid_indices.view(wp.int32).fill_(-1)

        self._detect_rigid_contacts(state, contacts, dt=dt)

        model = self.model
        if soft_contact_margin is not None:
            warnings.warn(
                "The soft_contact_margin argument of CollisionPipeline.collide() is deprecated; "
                "set soft_contact_gap on the CollisionPipeline constructor instead.",
                NewtonDeprecationWarning,
                stacklevel=2,
            )
            soft_contact_gap = soft_contact_margin
        else:
            soft_contact_gap = self.soft_contact_gap

        # The reducers' counters are reused by the next pass, so latch losses on this contact stream.
        reducer = self.narrow_phase.global_contact_reducer
        if reducer is not None:
            wp.launch(
                _record_reduction_overflow,
                dim=1,
                inputs=[reducer.ht_insert_failures, reducer.buffer_overflows, contacts._reduction_overflow],
                device=self.device,
                record_tape=False,
            )

        pool = contacts._rigid_contact_pool
        if self.deterministic and self._contact_sorter is not None:
            self._contact_sorter.sort_full(
                self._sort_key_array,
                contacts.rigid_contact_count,
                shape0=contacts.rigid_contact_shape0,
                shape1=contacts.rigid_contact_shape1,
                point0=contacts.rigid_contact_point0,
                point1=contacts.rigid_contact_point1,
                offset0=contacts.rigid_contact_offset0,
                offset1=contacts.rigid_contact_offset1,
                normal=contacts.rigid_contact_normal,
                margin0=contacts.rigid_contact_margin0,
                margin1=contacts.rigid_contact_margin1,
                tids=contacts.rigid_contact_tids,
                stiffness=contacts.rigid_contact_stiffness,
                damping=contacts.rigid_contact_damping,
                friction=contacts.rigid_contact_friction,
                match_index=contacts.rigid_contact_match_index,
                device=self.device,
            )

        contacts.rigid_contacts_pair_sorted = bool(self.deterministic)

        # Body-pair contact reduction: compact patch-redundant candidates so rigid_contact_count
        # reflects the contact structure, not the collider decomposition, before matching.
        if self._body_pair_reducer is not None:
            # A different Contacts instance means a different stream of states: winners recorded
            # for the previous buffer must not bias this one (a weak reference, since a raw id()
            # can be recycled after the old buffer dies).
            last = self._last_contacts_ref
            if last is not None and last() is not contacts:
                self._body_pair_reducer.reset_history()
            if last is None or last() is not contacts:
                self._last_contacts_ref = weakref.ref(contacts)
            self._body_pair_reducer.reduce(
                model,
                state,
                contacts,
                sort_keys=self._contact_sorter.sorted_keys_view if self._contact_matcher is not None else None,
            )
            # The reduction rewrote the count: later launches follow the reduced count.
            pool.begin_updates()

        # Match and save the retained stream so all indices name solver contacts.
        # Matching now sees the final retained and sorted stream.
        if self._contact_matcher is not None:
            if contacts.rigid_contact_match_index is None:
                raise ValueError(
                    "CollisionPipeline has contact_matching enabled but the "
                    "Contacts buffer was created without contact_matching. "
                    "Use pipeline.contacts() to create a compatible buffer."
                )
            self._contact_matcher.match(
                sort_keys=self._contact_sorter.sorted_keys_view,
                contact_count=contacts.rigid_contact_count,
                point0=contacts.rigid_contact_point0,
                point1=contacts.rigid_contact_point1,
                shape0=contacts.rigid_contact_shape0,
                shape1=contacts.rigid_contact_shape1,
                normal=contacts.rigid_contact_normal,
                body_q=state.body_q,
                shape_body=model.shape_body,
                match_index_out=contacts.rigid_contact_match_index,
                buffer_id=self._contact_matcher.buffer_id(contacts),
                match_generation_out=contacts.rigid_contact_match_generation,
                device=self.device,
            )
        elif contacts.rigid_contact_match_index is not None:
            # A buffer may come from a matching pipeline. Its current producer
            # cannot validate that previous identity, including on graph replay.
            pool.launch(
                _clear_match_index,
                pool.launch_rows(),
                inputs=[contacts.rigid_contact_count, contacts.rigid_contact_match_index],
                extent=contacts.rigid_contact_count,
            )
            contacts.rigid_contact_match_generation.fill_(GENERATION_SENTINEL)

        # Sticky mode: overwrite matched rows with the saved previous-frame
        # contact geometry.  Must run after matching (so match_index points at
        # the sorted prev-frame layout *and* we target the final sorted rows)
        # and before save_sorted_state (we save the record we actually used
        # this frame, carrying the sticky history forward).
        if self._matching_sticky:
            self._contact_matcher.replay_matched(
                contact_count=contacts.rigid_contact_count,
                match_index=contacts.rigid_contact_match_index,
                point0=contacts.rigid_contact_point0,
                point1=contacts.rigid_contact_point1,
                offset0=contacts.rigid_contact_offset0,
                offset1=contacts.rigid_contact_offset1,
                normal=contacts.rigid_contact_normal,
                shape0=contacts.rigid_contact_shape0,
                shape1=contacts.rigid_contact_shape1,
                margin0=contacts.rigid_contact_margin0,
                margin1=contacts.rigid_contact_margin1,
                body_q=state.body_q,
                shape_body=model.shape_body,
                device=self.device,
            )

        if self._rigid_contact_surface_velocity and contacts.rigid_contact_max > 0:
            if (
                contacts.rigid_contact_surface_velocity is None
                or len(contacts.rigid_contact_surface_velocity) < contacts.rigid_contact_max
            ):
                raise ValueError(
                    "contacts must allocate rigid surface velocities for this model; use CollisionPipeline.contacts()"
                )
            wp.launch(
                kernel=eval_rigid_contact_surface_velocities,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    state.body_q,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    self.geom_transform,
                    model.shape_source_ptr,
                    model._shape_mesh_properties,
                ],
                outputs=[contacts.rigid_contact_surface_velocity],
                device=self.device,
                record_tape=False,
            )

        # Build the contact report before saving state, because save
        # overwrites _prev_count and the report needs the old value.
        if self._contact_matcher is not None:
            if self._contact_matcher.has_report:
                if contacts.rigid_contact_new_indices is None:
                    raise ValueError(
                        "CollisionPipeline has contact_report enabled but the Contacts "
                        "buffer was created without contact_report=True. "
                        "Use pipeline.contacts() to create a compatible buffer."
                    )
                self._contact_matcher.build_report(
                    contacts.rigid_contact_match_index,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_new_indices,
                    contacts.rigid_contact_new_count,
                    contacts.rigid_contact_broken_indices,
                    contacts.rigid_contact_broken_count,
                    device=self.device,
                )
            sticky_offsets: dict[str, wp.array] = (
                {
                    "sorted_offset0": contacts.rigid_contact_offset0,
                    "sorted_offset1": contacts.rigid_contact_offset1,
                }
                if self._matching_sticky
                else {}
            )
            self._contact_matcher.save_sorted_state(
                sorted_keys=self._contact_sorter.sorted_keys_view,
                contact_count=contacts.rigid_contact_count,
                sorted_point0=contacts.rigid_contact_point0,
                sorted_point1=contacts.rigid_contact_point1,
                sorted_shape0=contacts.rigid_contact_shape0,
                sorted_shape1=contacts.rigid_contact_shape1,
                sorted_normal=contacts.rigid_contact_normal,
                body_q=state.body_q,
                shape_body=model.shape_body,
                buffer_id=self._contact_matcher.buffer_id(contacts),
                contact_generation=contacts.contact_generation,
                device=self.device,
                **sticky_offsets,
            )

        # Differentiable contact augmentation: reconstruct world-space contact
        # quantities through body_q so that gradients flow via wp.Tape.
        if self.requires_grad and contacts._rigid_contact_diff_distance is not None:
            launch_differentiable_contact_augment(
                contacts=contacts,
                body_q=state.body_q,
                shape_body=model.shape_body,
                device=self.device,
            )

        # Generate soft contacts for particles and shapes
        if state.particle_q and self.soft_contact_max > 0 and self.soft_contact_pair_count > 0:
            wp.launch(
                kernel=create_soft_contacts,
                dim=self.soft_contact_pair_count,
                inputs=[
                    self.soft_rigid_contact_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.particle_flags,
                    model.particle_world,
                    state.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_source_ptr,
                    model._shape_mesh_properties,
                    model.shape_world,
                    # Persistent bounds may be overwritten before tape backward replays this call.
                    None if self.requires_grad else self.narrow_phase.shape_aabb_lower,
                    None if self.requires_grad else self.narrow_phase.shape_aabb_upper,
                    model.shape_gap,
                    soft_contact_gap,
                    model.shape_margin,
                    self.soft_contact_max,
                    model.shape_flags,
                    model.shape_heightfield_index,
                    model.heightfield_data,
                    model.heightfield_elevations,
                    self.enable_rigid_soft_full_surface_contact,
                ],
                outputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    contacts.soft_contact_tids,
                ],
                device=self.device,
            )

        # Full-surface EDGE/FACE passes (opt-in, set at construction): add the soft edge/face contacts
        # the per-particle path cannot detect. Run after the particle launch on the same stream, so
        # edge/face records append after the active particle-contact prefix.
        # The flag is fixed at construction because soft_contact_max headroom is sized there.
        if self.enable_rigid_soft_full_surface_contact and state.particle_q:
            launch_soft_ef_contacts(
                model=model,
                state=state,
                contacts=contacts,
                margin=soft_contact_gap,
                device=self.device,
                edge_pairs=self.soft_edge_rigid_pairs,
                face_pairs=self.soft_face_rigid_pairs,
                sdf_fallback_tids=self._soft_sdf_fallback_tids,
                sdf_fallback_count=self._soft_sdf_fallback_count,
                n_particle_pairs=self.soft_contact_pair_count,
                edge_sdf_geo_types=self._soft_edge_sdf_geo_types,
                face_sdf_geo_types=self._soft_face_sdf_geo_types,
                # The AABB cull reads a persistent buffer rewritten every collide(); a tape
                # backward replay would see the LAST step's bounds, changing which contact
                # kernels early-return versus the forward pass. The cull is a pure optimization,
                # so differentiable pipelines skip it (empty arrays disable the test in-kernel).
                shape_aabb_lower=None if self.requires_grad else self.narrow_phase.shape_aabb_lower,
                shape_aabb_upper=None if self.requires_grad else self.narrow_phase.shape_aabb_upper,
            )
            launch_soft_heightfield_contacts(
                model=model,
                state=state,
                contacts=contacts,
                margin=soft_contact_gap,
                device=self.device,
                face_pairs=self._soft_heightfield_face_pairs,
                tid_base=self.soft_contact_pair_count
                + len(self.soft_edge_rigid_pairs)
                + len(self.soft_face_rigid_pairs),
            )
            # Run the variable-output mesh pass last so its saturated overflow
            # sentinel cannot be incremented by a subsequent contact producer.
            if self._soft_mesh_contact_data is not None:
                launch_soft_mesh_contacts(
                    model=model,
                    state=state,
                    contacts=contacts,
                    gap=soft_contact_gap,
                    data=self._soft_mesh_contact_data,
                )

        # Preserve the previous provenance if validation or collision setup fails.
        contacts._contact_matching_mode = self.contact_matching

        # Soft (cloth) self-contact detection (opt-in per call; results land in
        # contacts.soft_self_contact_data).
        if soft_self_contact:
            self._detect_soft_self_contact(state.particle_q, contacts)

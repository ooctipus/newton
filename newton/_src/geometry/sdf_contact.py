# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp

from ..geometry.broad_phase_common import BODY_FLAG_KINEMATIC
from ..geometry.contact_data import (
    CONTACT_NORMAL_OWNER_SHAPE_A,
    CONTACT_NORMAL_OWNER_SHAPE_B,
    CONTACT_STRICT_GUARD_ONLY,
    SHAPE_PAIR_HFIELD_BIT,
    SHAPE_PAIR_INDEX_MASK,
    ContactData,
    compute_contact_swept_separation_lower_bound,
    pack_contact_is_canonical_endpoint,
    pack_contact_is_strict_guard,
    pack_contact_normal_owner,
    unpack_contact_normal_owner,
)
from ..geometry.sdf_texture import (
    TextureSDFData,
    _texture_sample_sdf_grad_only_hw_paired,
    _texture_sample_sdf_grad_only_hw_scalar,
    _texture_sample_sdf_hw_clamped,
    _texture_sample_sdf_hw_clamped_paired,
    _texture_sample_sdf_hw_clamped_scalar,
    _texture_sample_sdf_hw_pair,
    _texture_sample_sdf_hw_pair_paired,
    _texture_sample_sdf_hw_pair_scalar,
    _texture_sample_sdf_hw_paired,
    _texture_sample_sdf_hw_scalar,
    texture_sample_sdf_grad_only_hw,
    texture_sample_sdf_hw,
)
from ..geometry.types import GeoType
from ..utils.heightfield import HeightfieldData, sample_sdf_grad_heightfield, sample_sdf_heightfield
from .contact_reduction_global import (
    GlobalContactReducerData,
    export_and_reduce_contact_centered_two_spatial_depths,
    export_and_reduce_predictive_contact,
)
from .flags import MeshProperties, MeshSignMethod
from .kernels import mesh_query_point_sign, resolve_mesh_sign_method
from .raycast import map_ray_to_local, ray_intersect_mesh

# Upper bound (mesh-local units) on the closest-point search for the mesh sign
# queries below. It is not a physical contact range: culling uses the tight
# ``_SDF_QUERY_RADIUS_SLACK * threshold`` bounds, and these queries only run at
# points the narrow phase already placed near the surface, so the bound is
# never binding. The queries operate in mesh-local coordinates (per-shape scale
# is divided out), so there is no global unit to tie it to; 1e11 is effectively
# unbounded for any asset under any unit convention (~8x Earth's diameter in
# millimeters). Going much larger is unsafe: Warp's mesh queries square
# ``max_dist`` internally, and the no-hit branches return it as a sentinel that
# contact math multiplies by per-shape scale (budgeted up to 1e6 for extreme
# unit conversions): (1e11 * 1e6)^2 = 1e34 clears float32 max (~3.4e38) by
# four orders of magnitude, where values near sqrt(FLT_MAX) ~ 1.8e19 would
# overflow. Kept finite (rather than ``wp.inf``) for the same reason.
_MESH_QUERY_MAX_DIST = 1.0e11

# Search-radius slack over the narrow-band culling threshold for midpoint SDF
# queries. Slightly exceeding the threshold guarantees points right at the
# threshold are classified by a real closest-point query instead of falling
# into the no-hit sentinel branch.
_SDF_QUERY_RADIUS_SLACK = 1.01

# Launch-side block size for the mesh-SDF narrow-phase kernels. Must match
# the ``block_dim`` used in ``wp.launch_tiled`` for
# ``mesh_sdf_collision_kernel`` and ``mesh_sdf_collision_global_reduce_kernel``.
# Both kernels assume ``wp.block_dim() == MESH_SDF_BLOCK_DIM`` so that the
# tile-stack capacity below correctly sizes the cooperative push overflow
# margin.
MESH_SDF_BLOCK_DIM = 256

# One cooperative block owns each strict mesh-pair query. Raw vertex and
# authored-edge coverage is exhaustive but distributed across the block lanes.
MESH_SDF_ORACLE_BLOCK_DIM = 256

# Empty-key sentinel and hash multiplier for the strict oracle's exact
# directed-vertex feature table.  The table stores keys with bit 63 clear, so
# the all-ones sentinel cannot alias a valid feature.
_MESH_SDF_ORACLE_EMPTY_FEATURE = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))
_MESH_SDF_ORACLE_FEATURE_HASH_MULTIPLIER = wp.constant(wp.uint64(0xFF51AFD7ED558CCD))

# Capacity of the cooperative edge-selection tile stack. Sized to
# ``2 * MESH_SDF_BLOCK_DIM`` so that the inner push loop can never
# overflow: the loop gate ``count < MESH_SDF_BLOCK_DIM`` caps pre-push
# ``count`` at ``MESH_SDF_BLOCK_DIM - 1``, and a single cooperative push
# from ``MESH_SDF_BLOCK_DIM`` threads adds at most ``MESH_SDF_BLOCK_DIM``
# more — fits within ``2 * MESH_SDF_BLOCK_DIM`` regardless of how many
# edges pass the culling test. The consumer-side invariant — "every
# pushed edge is eventually processed" — is maintained by draining the
# stack completely (inner ``while count > 0`` pop loop) before the next
# outer iteration runs.
STACK_CAPACITY = 2 * MESH_SDF_BLOCK_DIM

# Segments reuse the triangle-pair scratch buffer after triangle contacts finish.
# The two-int header is followed by packed ``(edge index, midpoint SDF)`` pairs.
_SDF_WORK_SEGMENT_HEADER_INT32 = 2
SDF_WORK_SEGMENT_STRIDE_INT32 = _SDF_WORK_SEGMENT_HEADER_INT32 + 2 * MESH_SDF_BLOCK_DIM
SDF_WORK_STATE_SIZE = 2
_SDF_WORK_SEGMENT_COUNT = 0
_SDF_WORK_OVERFLOWED = 1

# Covers float32 endpoint integration and quaternion normalization when a
# constant-twist certificate is checked and converted into a curvature bound.
_CONSTANT_TWIST_ANGLE_TOLERANCE = wp.constant(1.0e-5)

# Ambiguous constant-twist triangle pairs are retried over successively finer
# dyadic intervals. Five levels cap the fallback at 32 intervals while the
# common full-interval proof remains unchanged.
_CONSTANT_TWIST_SUBDIVISION_LEVELS = 5

# A texture SDF edge may cross several separated minima. Bound each Brent
# search to at most one voxel diameter, but cap the solve-local partition so
# one edge/mode emits at most 32 minima plus its two authored endpoints. The
# partition is reconstructed in registers and does not enlarge work buffers.
MESH_SDF_EDGE_SEGMENT_CAP = wp.constant(32)


@wp.func
def _mesh_sdf_nonpenetration_oracle_owns_pair(
    shape_a: int,
    shape_b: int,
    movable_capable_shape: wp.array[wp.uint8],
    anchor_capable_shape: wp.array[wp.uint8],
    shape_source: wp.array[wp.uint64],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
) -> bool:
    """Return whether the raw-mesh oracle owns one live shape pair."""
    if (
        movable_capable_shape.shape[0] == 0
        or anchor_capable_shape.shape[0] == 0
        or shape_source.shape[0] == 0
        or shape_body.shape[0] == 0
        or shape_source[shape_a] == wp.uint64(0)
        or shape_source[shape_b] == wp.uint64(0)
    ):
        return False

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]
    immovable_a = body_a < 0
    immovable_b = body_b < 0
    if body_a >= 0:
        if body_flags.shape[0] == 0:
            return False
        immovable_a = (body_flags[body_a] & BODY_FLAG_KINEMATIC) != 0
    if body_b >= 0:
        if body_flags.shape[0] == 0:
            return False
        immovable_b = (body_flags[body_b] & BODY_FLAG_KINEMATIC) != 0
    movable_a = not immovable_a and movable_capable_shape[shape_a] != wp.uint8(0)
    movable_b = not immovable_b and movable_capable_shape[shape_b] != wp.uint8(0)
    anchor_a = immovable_a and anchor_capable_shape[shape_a] != wp.uint8(0)
    anchor_b = immovable_b and anchor_capable_shape[shape_b] != wp.uint8(0)
    return (movable_a and anchor_b) or (movable_b and anchor_a)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __frsqrt_rn(value);
#else
return 1.0f / sqrtf(value);
#endif
""")
def _sdf_rsqrt_rn(value: float) -> float:
    """Return a round-to-nearest reciprocal square root."""
    ...


@wp.func
def _normalize_shortest_quaternion_pair(a: wp.quat, b: wp.quat) -> tuple[bool, wp.quat, wp.quat]:
    """Normalize two finite quaternions and select their shortest relative arc."""
    norm_a_sq = wp.dot(a, a)
    norm_b_sq = wp.dot(b, b)
    if not wp.isfinite(norm_a_sq) or not wp.isfinite(norm_b_sq) or norm_a_sq <= 1.0e-20 or norm_b_sq <= 1.0e-20:
        return False, wp.quat_identity(), wp.quat_identity()
    normalized_a = a / wp.sqrt(norm_a_sq)
    normalized_b = b / wp.sqrt(norm_b_sq)
    if wp.dot(normalized_a, normalized_b) < 0.0:
        normalized_b = -normalized_b
    return True, normalized_a, normalized_b


@wp.func
def _rotation_angle_between(a: wp.quat, b: wp.quat) -> tuple[bool, float]:
    """Return the principal rotation angle between two finite quaternions."""
    valid, normalized_a, normalized_b = _normalize_shortest_quaternion_pair(a, b)
    if not valid:
        return False, 0.0
    quaternion_delta_sq = wp.clamp(wp.dot(normalized_a - normalized_b, normalized_a - normalized_b), 0.0, 4.0)
    angle = 4.0 * wp.asin(wp.clamp(0.5 * wp.sqrt(quaternion_delta_sq), 0.0, 1.0))
    return wp.isfinite(angle), angle


@wp.func
def _rotation_displacement_bound(a: wp.quat, b: wp.quat, radius: float) -> tuple[bool, float]:
    """Bound surface displacement from a rotation over one shape radius."""
    angle_valid, angle = _rotation_angle_between(a, b)
    if not angle_valid or not wp.isfinite(radius) or radius < 0.0:
        return False, 0.0
    displacement = 2.0 * radius * wp.sin(0.5 * angle)
    return wp.isfinite(displacement), displacement


@wp.func
def _angular_path_displacement_bound(angular_path_length: float, radius: float) -> tuple[bool, float]:
    """Bound a material point's displacement over an orientation path.

    Unlike an endpoint quaternion difference, total angular variation retains
    full turns and other pose loops. The chord bound remains proportional to a
    small rotation and saturates only after a half turn.
    """
    valid = wp.isfinite(angular_path_length) and wp.isfinite(radius) and angular_path_length >= 0.0 and radius >= 0.0
    if not valid:
        return False, 0.0
    angle = wp.min(angular_path_length, wp.pi)
    displacement = 2.0 * radius * wp.sin(0.5 * angle)
    return wp.isfinite(displacement), displacement


@wp.func
def mesh_sdf_contact_search_precision(
    inner_contact_threshold: float,
    min_sdf_scale: float,
    voxel_radius: float,
    use_texture_sdf: bool,
) -> float:
    """Return SDF edge-search precision without letting contact gap loosen it."""
    search_precision = inner_contact_threshold / min_sdf_scale
    if use_texture_sdf:
        search_precision = wp.min(search_precision, voxel_radius)
    return search_precision


@wp.func
def mesh_sdf_contact_segment_count(edge_length: float, voxel_radius: float, use_texture_sdf: bool) -> int:
    """Return the bounded number of texture-SDF searches for one edge.

    Short texture edges and non-texture backends preserve the legacy single
    search. Longer texture edges use voxel-diameter intervals, capped at 32 to
    keep candidate work bounded independently of asset scale.
    """
    if not use_texture_sdf or edge_length <= 0.0 or voxel_radius <= 0.0 or not wp.isfinite(voxel_radius):
        return 1
    return wp.clamp(int(wp.ceil(edge_length / (2.0 * voxel_radius))), 1, MESH_SDF_EDGE_SEGMENT_CAP)


@wp.func
def mesh_sdf_contact_segment_bounds(
    v0: wp.vec3, v1: wp.vec3, segment_idx: int, segment_count: int
) -> tuple[wp.vec3, wp.vec3]:
    """Return one closed sub-interval of an edge's uniform voxel-scale partition."""
    edge_dir = v1 - v0
    inv_count = 1.0 / float(segment_count)
    segment_v0 = v0 + edge_dir * (float(segment_idx) * inv_count)
    segment_v1 = v0 + edge_dir * (float(segment_idx + 1) * inv_count)
    return segment_v0, segment_v1


@wp.func
def mesh_sdf_contact_passes_inner_cull_consistency(
    distance_world: float,
    inner_contact_threshold: float,
    midpoint_sdf: float,
    bsphere_center: wp.vec3,
    bsphere_radius: float,
    sdf_aabb_lower: wp.vec3,
    sdf_aabb_upper: wp.vec3,
    min_sdf_scale: float,
    use_texture_bounds: bool,
) -> bool:
    """Reject gap-found penetrations that fail the inner edge cull."""
    if distance_world >= inner_contact_threshold:
        return True

    inner_threshold_unscaled = inner_contact_threshold / min_sdf_scale
    culling_radius = bsphere_radius + inner_threshold_unscaled
    if use_texture_bounds:
        clamped = wp.min(wp.max(bsphere_center, sdf_aabb_lower), sdf_aabb_upper)
        if wp.length_sq(bsphere_center - clamped) > culling_radius * culling_radius:
            return False

    return midpoint_sdf <= culling_radius


@wp.func
def mesh_sdf_contact_sort_sub_key(edge_idx: int, segment_idx: int, mode: int, endpoint: int) -> int:
    """Return a unique deterministic key for an edge feature contact.

    The public low 23 bits retain the legacy edge/mode key. Interior minima
    store their 0--31 segment in otherwise internal feature bits 23--27; sort
    and reducer key builders fold those bits into their narrower public keys.
    Explicit authored endpoints retain their legacy keys.
    """
    if endpoint == 0:
        feature_key = (((edge_idx << 2) | (mode << 1)) & 0x7FFFFF) | (segment_idx << 23)
    else:
        feature_key = (edge_idx << 3) | (mode << 2) | (2 * endpoint - 1)
    normal_owner = CONTACT_NORMAL_OWNER_SHAPE_B if mode == 0 else CONTACT_NORMAL_OWNER_SHAPE_A
    return pack_contact_normal_owner(feature_key, normal_owner)


@wp.func
def mesh_sdf_contact_is_owned_endpoint(
    candidate_endpoint: int, best_endpoint: int, segment_idx: int, segment_count: int
) -> bool:
    """Return whether an emitted edge candidate represents an owned endpoint.

    The candidate-zero lane keeps the legacy edge feature key, so the endpoint
    selected by the edge minimizer must be carried separately.
    """
    if candidate_endpoint != 0:
        return True
    return (best_endpoint == 1 and segment_idx == 0) or (best_endpoint == 2 and segment_idx == segment_count - 1)


@wp.func
def mesh_sdf_contact_endpoint_owned(corner_ownership: int, endpoint: int) -> bool:
    """Return whether an edge owns one authored endpoint.

    Legacy edge buffers encode ownership as zero and therefore retain both
    endpoints. Explicit buffers set bit two and use bits zero and one for the
    first and second endpoints, respectively.
    """
    return corner_ownership == 0 or ((corner_ownership & 4) != 0 and (corner_ownership & endpoint) != 0)


@wp.func
def mesh_sdf_endpoint_guard_passes_admission(
    distance_world: float,
    margin_sum: float,
    base_gap_sum: float,
    max_speculative_extension: float,
) -> bool:
    """Admit a canonical endpoint in the bounded current-geometry guard shell."""
    clearance = distance_world - margin_sum
    return wp.isfinite(clearance) and clearance <= wp.max(base_gap_sum, max_speculative_extension)


@wp.func
def mesh_sdf_contact_segment_minimum_is_unique(best_endpoint: int, segment_idx: int) -> bool:
    """Assign an internal partition boundary to its lower segment only."""
    return best_endpoint != 1 or segment_idx == 0


@wp.func
def mesh_sdf_contact_voxel_owner(
    shape_a: int,
    shape_b: int,
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
) -> int:
    """Choose one deterministic, fine voxel frame shared by both SDF modes."""
    extent_a = shape_collision_aabb_upper[shape_a] - shape_collision_aabb_lower[shape_a]
    extent_b = shape_collision_aabb_upper[shape_b] - shape_collision_aabb_lower[shape_b]
    res_a = shape_voxel_resolution[shape_a]
    res_b = shape_voxel_resolution[shape_b]

    valid = (
        res_a[0] > 0
        and res_a[1] > 0
        and res_a[2] > 0
        and res_b[0] > 0
        and res_b[1] > 0
        and res_b[2] > 0
        and extent_a[0] >= 0.0
        and extent_a[1] >= 0.0
        and extent_a[2] >= 0.0
        and extent_b[0] >= 0.0
        and extent_b[1] >= 0.0
        and extent_b[2] >= 0.0
        and wp.isfinite(extent_a[0])
        and wp.isfinite(extent_a[1])
        and wp.isfinite(extent_a[2])
        and wp.isfinite(extent_b[0])
        and wp.isfinite(extent_b[1])
        and wp.isfinite(extent_b[2])
    )
    if not valid:
        return shape_a

    cell_a = wp.cw_div(extent_a, wp.vec3(float(res_a[0]), float(res_a[1]), float(res_a[2])))
    cell_b = wp.cw_div(extent_b, wp.vec3(float(res_b[0]), float(res_b[1]), float(res_b[2])))
    error_a = wp.dot(cell_a, cell_a)
    error_b = wp.dot(cell_b, cell_b)
    if not wp.isfinite(error_a) or not wp.isfinite(error_b) or error_a <= 0.0 or error_b <= 0.0:
        return shape_a
    if error_b < error_a:
        return shape_b
    if error_a < error_b:
        return shape_a
    return wp.min(shape_a, shape_b)


@wp.func
def safe_sdf_scale_inverse(sdf_scale: wp.vec3) -> tuple[wp.vec3, float]:
    """Sign-preserving safe inverse of an SDF shape's per-axis scale.

    Returns ``(inv_sdf_scale, min_abs_sdf_scale)``. Negative components are
    preserved (mirroring an SDF reflects its gradient field), but components
    near zero are guarded with a small epsilon to avoid divide-by-zero. The
    minimum is taken on magnitudes because it is used as a conservative
    distance scaling factor and must always be positive.
    """
    eps = float(1.0e-10)
    sx = wp.where(wp.abs(sdf_scale[0]) > eps, sdf_scale[0], wp.where(sdf_scale[0] >= 0.0, eps, -eps))
    sy = wp.where(wp.abs(sdf_scale[1]) > eps, sdf_scale[1], wp.where(sdf_scale[1] >= 0.0, eps, -eps))
    sz = wp.where(wp.abs(sdf_scale[2]) > eps, sdf_scale[2], wp.where(sdf_scale[2] >= 0.0, eps, -eps))
    inv = wp.vec3(1.0 / sx, 1.0 / sy, 1.0 / sz)
    min_abs = wp.min(wp.min(wp.abs(sx), wp.abs(sy)), wp.abs(sz))
    return inv, min_abs


@wp.struct
class EdgeCullResult:
    """Packed result from the mesh-SDF midphase edge-culling pass.

    Stores the edge index together with the midpoint SDF value computed
    during culling, so a single cooperative stack can carry both values
    atomically. Splitting them across two separate stacks would break
    the pairing because ``wp.tile_stack_pop`` races for slots
    independently on each stack.
    """

    edge_idx: int
    midpoint_sdf: float


@wp.struct
class MeshSDFCullContext:
    context_id: int
    block_in_pair: int
    blocks_for_pair: int
    sdf_index: int
    edge_range: wp.vec2i
    mesh_to_sdf: wp.transform
    contact_threshold: float


@wp.struct
class MeshSDFSearchContext:
    sdf_index: int
    edge_range: wp.vec2i
    mesh_to_sdf: wp.transform
    contact_threshold: float
    search_precision: float
    margin_sum: float


@wp.struct
class MeshSDFExportContext:
    inner_spatial_depth: float
    outer_spatial_depth: float


@wp.func
def scale_sdf_result_to_world(
    distance: float,
    gradient: wp.vec3,
    sdf_scale: wp.vec3,
    inv_sdf_scale: wp.vec3,
    min_sdf_scale: float,
) -> tuple[float, wp.vec3]:
    """
    Convert SDF distance and gradient from unscaled space to scaled space.

    Args:
        distance: Signed distance in unscaled SDF local space
        gradient: Gradient direction in unscaled SDF local space
        sdf_scale: The SDF shape's scale vector
        inv_sdf_scale: Precomputed 1.0 / sdf_scale
        min_sdf_scale: Precomputed min(sdf_scale) for distance scaling

    Returns:
        Tuple of (scaled_distance, scaled_gradient)
    """
    # Use min scale for conservative distance (won't miss contacts)
    scaled_distance = distance * min_sdf_scale

    # Apply inverse scale here; callers normalize after rotating the gradient
    # into world space, so normalizing before the rigid rotation is redundant.
    scaled_grad = wp.cw_mul(gradient, inv_sdf_scale)

    return scaled_distance, scaled_grad


@wp.func
def sample_sdf_using_mesh(
    mesh_id: wp.uint64,
    world_pos: wp.vec3,
    max_dist: float = _MESH_QUERY_MAX_DIST,
    sign_method: int = MeshSignMethod.NORMAL,
) -> float:
    """
    Sample signed distance to mesh surface using mesh query.

    Uses a mesh sign query to find the closest point on the mesh and compute
    the signed distance. This is compatible with the return type of
    sample_sdf_extrapolated.

    Args:
        mesh_id: The mesh ID (from wp.Mesh.id)
        world_pos: Query position in mesh local coordinates
        max_dist: Maximum distance to search for closest point
        sign_method: Method used to determine the mesh query sign.

    Returns:
        The signed distance value (negative inside, positive outside)
    """
    res = mesh_query_point_sign(mesh_id, world_pos, max_dist, sign_method)

    if res.result:
        closest = wp.mesh_eval_position(mesh_id, res.face, res.u, res.v)
        return wp.length(world_pos - closest) * res.sign

    return max_dist


@wp.func
def sample_sdf_grad_using_mesh(
    mesh_id: wp.uint64,
    world_pos: wp.vec3,
    max_dist: float = _MESH_QUERY_MAX_DIST,
    sign_method: int = MeshSignMethod.NORMAL,
) -> tuple[float, wp.vec3]:
    """
    Sample signed distance and gradient to mesh surface using mesh query.

    Uses a mesh sign query to find the closest point on the mesh and compute
    both the signed distance and the gradient direction. This is compatible
    with the return type of sample_sdf_grad_extrapolated.

    The gradient points in the direction of increasing distance (away from the surface
    when outside, toward the surface when inside).

    Args:
        mesh_id: The mesh ID (from wp.Mesh.id)
        world_pos: Query position in mesh local coordinates
        max_dist: Maximum distance to search for closest point
        sign_method: Method used to determine the mesh query sign.

    Returns:
        Tuple of (distance, gradient) where:
        - distance: Signed distance value (negative inside, positive outside)
        - gradient: Normalized direction of increasing distance
    """
    gradient = wp.vec3(0.0, 0.0, 0.0)

    res = mesh_query_point_sign(mesh_id, world_pos, max_dist, sign_method)

    if res.result:
        closest = wp.mesh_eval_position(mesh_id, res.face, res.u, res.v)
        diff = world_pos - closest
        dist = wp.length(diff)

        if dist > 0.0:
            # Gradient points from surface toward query point, scaled by sign
            # When outside (sign > 0): gradient points away from surface (correct for SDF)
            # When inside (sign < 0): gradient points toward surface (correct for SDF)
            gradient = (diff / dist) * res.sign
        else:
            # Point is exactly on surface - use face normal
            mesh = wp.mesh_get(mesh_id)
            i0 = mesh.indices[res.face * 3 + 0]
            i1 = mesh.indices[res.face * 3 + 1]
            i2 = mesh.indices[res.face * 3 + 2]
            v0 = mesh.points[i0]
            v1 = mesh.points[i1]
            v2 = mesh.points[i2]
            face_normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))
            gradient = face_normal * res.sign

        return dist * res.sign, gradient

    # No hit found - return max distance with arbitrary gradient
    return max_dist, wp.vec3(0.0, 0.0, 1.0)


@wp.func
def closest_pt_point_bary_triangle(c: wp.vec3) -> wp.vec3:
    """
    Find the closest point to `c` on the standard barycentric triangle.

    This function projects a barycentric coordinate point onto the valid barycentric
    triangle defined by vertices (1,0,0), (0,1,0), (0,0,1) in barycentric space.
    The valid region is where all coordinates are non-negative and sum to 1.

    This is a specialized version of the general closest-point-on-triangle algorithm
    optimized for the barycentric simplex.

    Args:
        c: Input barycentric coordinates (may be outside valid triangle region)

    Returns:
        The closest valid barycentric coordinates. All components will be >= 0
        and sum to 1.0.

    Note:
        This is used in optimization algorithms that work in barycentric space,
        where gradient descent may produce invalid coordinates that need projection.
    """
    third = 1.0 / 3.0  # constexpr
    c = c - wp.vec3(third * (c[0] + c[1] + c[2] - 1.0))

    # two negative: return positive vertex
    if c[1] < 0.0 and c[2] < 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    if c[0] < 0.0 and c[2] < 0.0:
        return wp.vec3(0.0, 1.0, 0.0)

    if c[0] < 0.0 and c[1] < 0.0:
        return wp.vec3(0.0, 0.0, 1.0)

    # one negative: return projection onto line if it is on the edge, or the largest vertex otherwise
    if c[0] < 0.0:
        d = c[0] * 0.5
        y = c[1] + d
        z = c[2] + d
        if y > 1.0:
            return wp.vec3(0.0, 1.0, 0.0)
        if z > 1.0:
            return wp.vec3(0.0, 0.0, 1.0)
        return wp.vec3(0.0, y, z)
    if c[1] < 0.0:
        d = c[1] * 0.5
        x = c[0] + d
        z = c[2] + d
        if x > 1.0:
            return wp.vec3(1.0, 0.0, 0.0)
        if z > 1.0:
            return wp.vec3(0.0, 0.0, 1.0)
        return wp.vec3(x, 0.0, z)
    if c[2] < 0.0:
        d = c[2] * 0.5
        x = c[0] + d
        y = c[1] + d
        if x > 1.0:
            return wp.vec3(1.0, 0.0, 0.0)
        if y > 1.0:
            return wp.vec3(0.0, 1.0, 0.0)
        return wp.vec3(x, y, 0.0)
    return c


@wp.func
def get_triangle_from_mesh(
    mesh_id: wp.uint64,
    mesh_scale: wp.vec3,
    X_mesh_ws: wp.transform,
    tri_idx: int,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """
    Extract a triangle from a mesh and transform it to world space.

    This function retrieves a specific triangle from a mesh by its index,
    applies scaling and transformation, and returns the three vertices
    in world space coordinates.

    Args:
        mesh_id: The mesh ID (use wp.mesh_get to retrieve the mesh object)
        mesh_scale: Scale to apply to mesh vertices (component-wise)
        X_mesh_ws: Mesh world-space transform (position and rotation)
        tri_idx: Triangle index in the mesh (0-based)

    Returns:
        Tuple of (v0_world, v1_world, v2_world) - the three triangle vertices
        in world space after applying scale and transform.

    Note:
        The mesh indices array stores triangle vertex indices as a flat array:
        [tri0_v0, tri0_v1, tri0_v2, tri1_v0, tri1_v1, tri1_v2, ...]
    """

    mesh = wp.mesh_get(mesh_id)

    # Extract triangle vertices from mesh (indices are stored as flat array: i0, i1, i2, i0, i1, i2, ...)
    idx0 = mesh.indices[tri_idx * 3 + 0]
    idx1 = mesh.indices[tri_idx * 3 + 1]
    idx2 = mesh.indices[tri_idx * 3 + 2]

    # Get vertex positions in mesh local space (with scale applied)
    v0_local = wp.cw_mul(mesh.points[idx0], mesh_scale)
    v1_local = wp.cw_mul(mesh.points[idx1], mesh_scale)
    v2_local = wp.cw_mul(mesh.points[idx2], mesh_scale)

    # Transform vertices to world space
    v0_world = wp.transform_point(X_mesh_ws, v0_local)
    v1_world = wp.transform_point(X_mesh_ws, v1_local)
    v2_world = wp.transform_point(X_mesh_ws, v2_local)

    return v0_world, v1_world, v2_world


@wp.func
def get_bounding_sphere(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> tuple[wp.vec3, float]:
    """
    Compute a conservative bounding sphere for a triangle.

    This uses the triangle centroid as the sphere center and the maximum
    distance from the centroid to any vertex as the radius. This is a
    conservative (potentially larger than optimal) but fast bounding sphere.

    Args:
        v0, v1, v2: Triangle vertices in world space

    Returns:
        Tuple of (center, radius) where:
        - center: The centroid of the triangle
        - radius: The maximum distance from centroid to any vertex

    Note:
        This is not the minimal bounding sphere, but it's fast to compute
        and adequate for broad-phase culling.
    """
    center = (v0 + v1 + v2) * (1.0 / 3.0)
    radius = wp.max(wp.max(wp.length_sq(v0 - center), wp.length_sq(v1 - center)), wp.length_sq(v2 - center))
    return center, wp.sqrt(radius)


@wp.func
def get_edge_from_mesh(
    mesh_id: wp.uint64,
    mesh_edge_indices: wp.array[wp.vec2i],
    edge_range: wp.vec2i,
    mesh_scale: wp.vec3,
    X_mesh_ws: wp.transform,
    edge_idx: int,
) -> tuple[wp.vec3, wp.vec3]:
    """Extract an edge from a mesh and transform it to world space.

    Reads the edge vertex pair from the packed ``mesh_edge_indices`` array
    using the per-shape ``edge_range`` offset, and returns both endpoints
    in world space after applying scale and transform.

    Args:
        mesh_id: The mesh ID (use wp.mesh_get to retrieve the mesh object)
        mesh_edge_indices: Packed array of all mesh edge vertex pairs.
        edge_range: ``(start, count)`` slice for this shape into ``mesh_edge_indices``.
        mesh_scale: Scale to apply to mesh vertices (component-wise)
        X_mesh_ws: Mesh world-space transform (position and rotation)
        edge_idx: Edge index within this shape (0-based)

    Returns:
        Tuple of (v0_world, v1_world) - the two edge endpoints in world space.
    """
    mesh = wp.mesh_get(mesh_id)
    edge = mesh_edge_indices[edge_range[0] + edge_idx]

    idx0 = edge[0]
    idx1 = edge[1]

    v0_local = wp.cw_mul(mesh.points[idx0], mesh_scale)
    v1_local = wp.cw_mul(mesh.points[idx1], mesh_scale)

    v0_world = wp.transform_point(X_mesh_ws, v0_local)
    v1_world = wp.transform_point(X_mesh_ws, v1_local)

    return v0_world, v1_world


@wp.func
def get_mesh_edge_precomputed(
    mesh_edge_centers: wp.array[wp.vec4],
    mesh_edge_halves: wp.array[wp.vec4],
    edge_range: wp.vec2i,
    X_mesh_ws: wp.transform,
    edge_idx: int,
) -> tuple[wp.vec3, wp.vec3, int]:
    """Extract an edge and endpoint ownership from precomputed data.

    A zero ownership code preserves legacy packed arrays by allowing both
    endpoints. Builder-generated codes use bits zero and one for the first and
    second endpoint, respectively, plus bit two to mark the encoding explicit.
    """
    packed_center = mesh_edge_centers[edge_range[0] + edge_idx]
    center_local = wp.vec3(packed_center[0], packed_center[1], packed_center[2])
    center = wp.transform_point(X_mesh_ws, center_local)
    packed_half = mesh_edge_halves[edge_range[0] + edge_idx]
    half_local = wp.vec3(packed_half[0], packed_half[1], packed_half[2])
    half = wp.transform_vector(X_mesh_ws, half_local)
    return center - half, center + half, int(packed_half[3])


def _create_mesh_edge_accessor_func(use_precomputed_edge_data: bool):
    """Create a mesh-edge accessor with its storage path compiled in."""

    @wp.func
    def get_edge_from_mesh_func(
        mesh_id: wp.uint64,
        mesh_edge_indices: wp.array[wp.vec2i],
        mesh_edge_centers: wp.array[wp.vec4],
        mesh_edge_halves: wp.array[wp.vec4],
        edge_range: wp.vec2i,
        mesh_scale: wp.vec3,
        X_mesh_ws: wp.transform,
        edge_idx: int,
    ) -> tuple[wp.vec3, wp.vec3, int]:
        if wp.static(use_precomputed_edge_data):
            return get_mesh_edge_precomputed(mesh_edge_centers, mesh_edge_halves, edge_range, X_mesh_ws, edge_idx)
        v0, v1 = get_edge_from_mesh(mesh_id, mesh_edge_indices, edge_range, mesh_scale, X_mesh_ws, edge_idx)
        return v0, v1, 0

    return get_edge_from_mesh_func


@wp.func
def get_edge_from_heightfield(
    hfd: HeightfieldData,
    elevation_data: wp.array[wp.float32],
    X_ws: wp.transform,
    edge_idx: int,
) -> tuple[wp.vec3, wp.vec3]:
    """Extract an edge from a heightfield by linear edge index.

    Heightfield edges are enumerated in three groups:

    - Horizontal edges: ``nrow * (ncol - 1)`` edges along rows.
    - Vertical edges: ``(nrow - 1) * ncol`` edges along columns.
    - Diagonal edges: ``(nrow - 1) * (ncol - 1)`` edges across cells.

    ``hfd`` already carries the per-instance scale baked into ``hx``, ``hy``,
    ``min_z``, and ``max_z`` by the builder, so the returned vertices do not
    need a further scale multiplication.

    Args:
        hfd: Heightfield descriptor (extents are scale-baked).
        elevation_data: Flat elevation array.
        X_ws: World-space transform.
        edge_idx: Linear edge index (0-based).

    Returns:
        Tuple of (v0_world, v1_world) - the two edge endpoints in world space.
    """
    nrow = hfd.nrow
    ncol = hfd.ncol

    dx = 2.0 * hfd.hx / wp.float32(ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(nrow - 1)
    z_range = hfd.max_z - hfd.min_z
    base = hfd.data_offset

    num_h = nrow * (ncol - 1)
    num_v = (nrow - 1) * ncol

    r0 = int(0)
    c0 = int(0)
    r1 = int(0)
    c1 = int(0)

    if edge_idx < num_h:
        # Horizontal edge
        r0 = edge_idx // (ncol - 1)
        c0 = edge_idx - r0 * (ncol - 1)
        r1 = r0
        c1 = c0 + 1
    elif edge_idx < num_h + num_v:
        # Vertical edge
        local = edge_idx - num_h
        r0 = local // ncol
        c0 = local - r0 * ncol
        r1 = r0 + 1
        c1 = c0
    else:
        # Diagonal edge
        local = edge_idx - num_h - num_v
        r0 = local // (ncol - 1)
        c0 = local - r0 * (ncol - 1)
        r1 = r0 + 1
        c1 = c0 + 1

    x0 = -hfd.hx + wp.float32(c0) * dx
    y0 = -hfd.hy + wp.float32(r0) * dy
    h0 = elevation_data[base + r0 * ncol + c0]
    p0 = wp.vec3(x0, y0, hfd.min_z + h0 * z_range)

    x1 = -hfd.hx + wp.float32(c1) * dx
    y1 = -hfd.hy + wp.float32(r1) * dy
    h1 = elevation_data[base + r1 * ncol + c1]
    p1 = wp.vec3(x1, y1, hfd.min_z + h1 * z_range)

    v0_world = wp.transform_point(X_ws, p0)
    v1_world = wp.transform_point(X_ws, p1)

    return v0_world, v1_world


@wp.func
def get_edge_bounding_sphere(v0: wp.vec3, v1: wp.vec3) -> tuple[wp.vec3, float]:
    """Compute the bounding sphere for an edge (midpoint and half-length).

    Args:
        v0: First edge endpoint.
        v1: Second edge endpoint.

    Returns:
        Tuple of (midpoint, half_length).
    """
    midpoint = (v0 + v1) * 0.5
    half_length = wp.length(v1 - v0) * 0.5
    return midpoint, half_length


def _create_get_mesh_edge_bounding_sphere_func(use_precomputed_edge_data: bool):
    """Create a mesh-edge bounding-sphere accessor with its storage path compiled in."""

    @wp.func
    def get_mesh_edge_bounding_sphere_func(
        mesh_id: wp.uint64,
        mesh_edge_indices: wp.array[wp.vec2i],
        mesh_edge_centers: wp.array[wp.vec4],
        edge_range: wp.vec2i,
        mesh_scale: wp.vec3,
        X_mesh_ws: wp.transform,
        inv_sdf_scale: wp.vec3,
        radius_scale: float,
        edge_idx: int,
    ) -> tuple[wp.vec3, float]:
        if wp.static(use_precomputed_edge_data):
            center_radius = mesh_edge_centers[edge_range[0] + edge_idx]
            center_local = wp.vec3(center_radius[0], center_radius[1], center_radius[2])
            center_scaled = wp.transform_point(X_mesh_ws, center_local)
            center = wp.cw_mul(center_scaled, inv_sdf_scale)
            return center, center_radius[3] * radius_scale

        v0, v1 = get_edge_from_mesh(mesh_id, mesh_edge_indices, edge_range, mesh_scale, X_mesh_ws, edge_idx)
        center, radius = get_edge_bounding_sphere(wp.cw_mul(v0, inv_sdf_scale), wp.cw_mul(v1, inv_sdf_scale))
        return center, radius

    return get_mesh_edge_bounding_sphere_func


@wp.func
def get_triangle_count(shape_type: int, mesh_id: wp.uint64, hfd: HeightfieldData) -> int:
    """Return the number of triangles for a mesh or heightfield shape."""
    if shape_type == GeoType.HFIELD:
        if hfd.nrow <= 1 or hfd.ncol <= 1:
            return 0
        return 2 * (hfd.nrow - 1) * (hfd.ncol - 1)
    return wp.mesh_get(mesh_id).indices.shape[0] // 3


@wp.func
def get_edge_count(shape_type: int, edge_range: wp.vec2i, hfd: HeightfieldData) -> int:
    """Return the number of edges for a mesh or heightfield shape."""
    if shape_type == GeoType.HFIELD:
        if hfd.nrow <= 1 or hfd.ncol <= 1:
            return 0
        return hfd.nrow * (hfd.ncol - 1) + (hfd.nrow - 1) * hfd.ncol + (hfd.nrow - 1) * (hfd.ncol - 1)
    return edge_range[1]


def _create_sdf_contact_funcs(
    enable_heightfields: bool,
    use_texture_sdf_only: bool,
    sample_sdf: Any,
    sample_pair: Any,
):
    """Generate SDF contact functions with heightfield branches eliminated at compile time.

    When ``enable_heightfields`` is False, ``wp.static`` strips all heightfield code
    paths from the generated functions, reducing register pressure and instruction
    cache footprint — especially in the bounded edge-minimization loop of
    ``do_edge_sdf_collision``.

    Args:
        enable_heightfields: When False, all heightfield code paths are compiled out.

    Returns:
        The edge minimizer and its backend-specialized point sampler.
    """

    @wp.func
    def _sample_sdf_at_t(
        texture_sdf: TextureSDFData,
        sdf_mesh_id: wp.uint64,
        v0: wp.vec3,
        edge_dir: wp.vec3,
        tt: float,
        use_bvh_for_sdf: bool,
        sdf_mesh_query_type: int,
        sdf_is_heightfield: bool,
        hfd_sdf: HeightfieldData,
        elevation_data: wp.array[wp.float32],
    ) -> float:
        """Sample SDF at the point ``v0 + tt * edge_dir``."""
        pp = v0 + edge_dir * tt
        if wp.static(enable_heightfields):
            if sdf_is_heightfield:
                return sample_sdf_heightfield(hfd_sdf, elevation_data, pp)
            elif wp.static(use_texture_sdf_only):
                return wp.static(sample_sdf)(texture_sdf, pp)
            elif use_bvh_for_sdf:
                return sample_sdf_using_mesh(sdf_mesh_id, pp, _MESH_QUERY_MAX_DIST, sdf_mesh_query_type)
            else:
                return wp.static(sample_sdf)(texture_sdf, pp)
        else:
            if wp.static(use_texture_sdf_only):
                return wp.static(sample_sdf)(texture_sdf, pp)
            elif use_bvh_for_sdf:
                return sample_sdf_using_mesh(sdf_mesh_id, pp, _MESH_QUERY_MAX_DIST, sdf_mesh_query_type)
            else:
                return wp.static(sample_sdf)(texture_sdf, pp)

    @wp.func
    def do_edge_sdf_collision_func(
        texture_sdf: TextureSDFData,
        sdf_mesh_id: wp.uint64,
        v0: wp.vec3,
        v1: wp.vec3,
        midpoint_sdf: float,
        use_bvh_for_sdf: bool,
        sdf_mesh_query_type: int,
        sdf_is_heightfield: bool,
        hfd_sdf: HeightfieldData,
        elevation_data: wp.array[wp.float32],
        precision_target: float,
    ) -> tuple[float, wp.vec3, int]:
        """Find the deepest point on an edge relative to an SDF volume.

        Minimizes the SDF value along the edge parameterized as
        ``p(t) = v0 + t * edge_dir`` for t in [0, 1]. The initial midpoint
        SDF value is provided by the caller (cached from culling). Heightfield-free
        texture searches issue the first symmetric pair together, use the three values
        to tighten the bracket, and spend the remaining three-query budget with
        Brent's method. Other SDF backends use five adaptive Brent queries.
        Both paths therefore use at most five interior queries beyond the
        cached midpoint.

        ``precision_target`` is the unscaled SDF space precision the caller
        cares about. Brent's tolerance floor is set
        to ``precision_target / edge_length / 2`` in parametric space so
        edges much shorter than the target precision exit Brent in 0
        queries (the midpoint is already accurate enough).

        After the interior search, evaluates the more promising endpoint
        (the one closer to the unconverged bracket boundary) so that vertex
        contacts at edge corners are not missed.

        The endpoint result is zero for an interior winner, one for ``v0``,
        and two for ``v1`` so duplicate corner contacts can honor edge ownership.

        Returns:
            Tuple of (distance, contact_point, endpoint).
        """
        golden = 0.3819660112501051  # (3 - sqrt(5)) / 2
        edge_dir = v1 - v0
        edge_length_sq = wp.length_sq(edge_dir)
        inv_edge_length = float(1.0e12)
        if edge_length_sq > 0.0:
            inv_edge_length = _sdf_rsqrt_rn(edge_length_sq)

        # Parametric tolerance floor: skip Brent for edges where the
        # midpoint already meets ``precision_target``. Zero-length edges
        # trivially meet any positive precision.
        tol_floor = 0.5 * precision_target * inv_edge_length

        # Initialize Brent's method at the midpoint (SDF value from culling)
        a = float(0.0)
        b = float(1.0)
        x = float(0.5)
        w = float(0.5)
        v_brent = float(0.5)
        fx = midpoint_sdf
        fw = fx
        fv = fx
        d_step = float(0.0)
        e_step = float(0.0)

        if wp.static(use_texture_sdf_only and not enable_heightfields) and tol_floor < 0.25:
            offset = 0.5 * golden
            left = 0.5 - offset
            right = 0.5 + offset
            pair_values = wp.static(sample_pair)(
                texture_sdf,
                v0 + edge_dir * left,
                v0 + edge_dir * right,
            )
            f_left = pair_values[0]
            f_right = pair_values[1]

            if f_left < fx and f_left <= f_right:
                b = 0.5
                x = left
                fx = f_left
                w = 0.5
                fw = midpoint_sdf
                v_brent = right
                fv = f_right
            elif f_right < fx:
                a = 0.5
                x = right
                fx = f_right
                w = 0.5
                fw = midpoint_sdf
                v_brent = left
                fv = f_left
            else:
                a = left
                b = right
                w = left
                fw = f_left
                v_brent = right
                fv = f_right
        for _iter in range(wp.static(3 if use_texture_sdf_only and not enable_heightfields else 5)):
            m = 0.5 * (a + b)
            tol = wp.max(1.0e-2 * wp.abs(x) + 1.0e-8, tol_floor)
            tol2 = 2.0 * tol

            if wp.abs(x - m) <= tol2 - 0.5 * (b - a):
                break

            # Try inverse parabolic interpolation
            use_parabolic = False
            p_num = float(0.0)
            q_denom = float(0.0)
            trial = float(0.0)

            if wp.abs(e_step) > tol:
                r = (x - w) * (fx - fv)
                q_denom = (x - v_brent) * (fx - fw)
                p_num = (x - v_brent) * q_denom - (x - w) * r
                q_denom = 2.0 * (q_denom - r)
                if q_denom > 0.0:
                    p_num = -p_num
                else:
                    q_denom = -q_denom

                # Check if parabolic step is acceptable
                if wp.abs(p_num) < 0.5 * wp.abs(q_denom * e_step):
                    trial = p_num / q_denom
                    u_trial = x + trial
                    if u_trial - a >= tol2 and b - u_trial >= tol2:
                        use_parabolic = True

            if use_parabolic:
                e_step = d_step
                d_step = trial
            else:
                # Golden section step
                if x >= m:
                    e_step = a - x
                else:
                    e_step = b - x
                d_step = golden * e_step

            # Evaluate new point
            if wp.abs(d_step) >= tol:
                u = x + d_step
            else:
                if d_step > 0.0:
                    u = x + tol
                else:
                    u = x - tol

            fu = _sample_sdf_at_t(
                texture_sdf,
                sdf_mesh_id,
                v0,
                edge_dir,
                u,
                use_bvh_for_sdf,
                sdf_mesh_query_type,
                sdf_is_heightfield,
                hfd_sdf,
                elevation_data,
            )

            # Update bracket
            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v_brent = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v_brent = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v_brent == x or v_brent == w:
                    v_brent = u
                    fv = fu

        # Check endpoints only while Brent's bracket still includes them.
        # Once a bound has moved inward, Brent has already excluded that
        # endpoint from containing the minimum.
        best_endpoint = int(0)
        best_t = x
        best_f = fx
        if a == 0.0:
            f_end = _sample_sdf_at_t(
                texture_sdf,
                sdf_mesh_id,
                v0,
                edge_dir,
                0.0,
                use_bvh_for_sdf,
                sdf_mesh_query_type,
                sdf_is_heightfield,
                hfd_sdf,
                elevation_data,
            )
            if f_end < best_f:
                best_t = 0.0
                best_f = f_end
                best_endpoint = 1
        if b == 1.0:
            f_end = _sample_sdf_at_t(
                texture_sdf,
                sdf_mesh_id,
                v0,
                edge_dir,
                1.0,
                use_bvh_for_sdf,
                sdf_mesh_query_type,
                sdf_is_heightfield,
                hfd_sdf,
                elevation_data,
            )
            if f_end < best_f:
                best_t = 1.0
                best_f = f_end
                best_endpoint = 2

        p = v0 + edge_dir * best_t

        return best_f, p, best_endpoint

    return do_edge_sdf_collision_func, _sample_sdf_at_t


@wp.kernel(enable_backward=False)
def compute_mesh_mesh_edge_counts(
    shape_pairs_mesh_mesh: wp.array[wp.vec2i],
    shape_pairs_mesh_mesh_count: wp.array[int],
    shape_edge_range: wp.array[wp.vec2i],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    edge_counts: wp.array[wp.int32],
    total_edge_count: wp.array[wp.int32],
    total_num_threads: int,
):
    """Compute per-pair edge counts for mesh-mesh (or heightfield-mesh) pairs.

    Sums the edge counts of both shapes in each pair — each shape may be
    a triangle mesh or a heightfield. Threads stride over the active pair
    prefix and accumulate its total edge count.
    """
    pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])
    for i in range(wp.tid(), pair_count, total_num_threads):
        pair_encoded = shape_pairs_mesh_mesh[i]
        has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
        pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
        pair_edges = int(0)
        for mode in range(2):
            is_hfield = has_hfield and mode == 0
            shape_idx = pair[mode]
            if is_hfield:
                hfd = heightfield_data[shape_heightfield_index[shape_idx]]
                pair_edges += get_edge_count(GeoType.HFIELD, wp.vec2i(-1, 0), hfd)
            else:
                pair_edges += shape_edge_range[shape_idx][1]
        edge_counts[i] = wp.int32(pair_edges)
        wp.atomic_add(total_edge_count, 0, pair_edges)


@wp.kernel(enable_backward=False)
def compute_block_counts_from_weights(
    total_weight_arr: wp.array[wp.int32],
    weights: wp.array[wp.int32],
    pair_count_arr: wp.array[int],
    max_pairs: int,
    target_blocks: int,
    block_counts: wp.array[wp.int32],
    total_num_threads: int,
):
    """Convert per-pair weights to block counts using adaptive load balancing.

    Reads the scalar total weight to compute the adaptive
    ``weight_per_block`` threshold, then assigns each active pair a block
    count proportional to its weight.
    """
    pair_count = wp.min(pair_count_arr[0], max_pairs)
    for i in range(wp.tid(), pair_count, total_num_threads):
        total_weight = total_weight_arr[0]
        weight_per_block = int(total_weight)
        if target_blocks > 0 and total_weight > 0:
            weight_per_block = wp.max(256, total_weight // target_blocks)

        w = int(weights[i])
        if weight_per_block > 0:
            blocks = wp.max(1, (w + weight_per_block - 1) // weight_per_block)
        else:
            blocks = 1
        block_counts[i] = wp.int32(blocks)


def compute_mesh_mesh_block_offsets_scan(
    shape_pairs_mesh_mesh: wp.array[wp.vec2i],
    shape_pairs_mesh_mesh_count: wp.array[int],
    shape_edge_range: wp.array[wp.vec2i],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    target_blocks: int,
    block_offsets: wp.array[wp.int32],
    block_counts: wp.array[wp.int32],
    total_edge_count: wp.array[wp.int32],
    total_num_threads: int,
    device: str | None = None,
    record_tape: bool = True,
) -> None:
    """Compute mesh-mesh block offsets using parallel kernels and array_scan.

    Runs a three-stage parallel pipeline: per-pair edge counts and scalar
    accumulation, adaptive block counts, then an exclusive scan into
    ``block_offsets``.
    """
    n = block_counts.shape[0]
    launch_threads = min(n, total_num_threads)
    # Step 1: compute per-pair edge counts in parallel
    wp.launch(
        kernel=compute_mesh_mesh_edge_counts,
        dim=launch_threads,
        inputs=[
            shape_pairs_mesh_mesh,
            shape_pairs_mesh_mesh_count,
            shape_edge_range,
            shape_heightfield_index,
            heightfield_data,
            block_counts,  # reuse as temp storage for edge counts
            total_edge_count,
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
            total_edge_count,
            block_counts,  # still holds per-pair edge counts, including heightfield edges
            shape_pairs_mesh_mesh_count,
            shape_pairs_mesh_mesh.shape[0],
            target_blocks,
            block_offsets,  # reuse as temp for block counts
            launch_threads,
        ],
        device=device,
        record_tape=record_tape,
    )
    # Step 3: exclusive scan of block counts → block_offsets
    wp.utils.array_scan(block_offsets, block_offsets, inclusive=False)


def create_mesh_sdf_owned_endpoint_guard_kernel(
    writer_func: Any,
    enable_heightfields: bool = True,
    reduce_contacts: bool = False,
    use_precomputed_edge_data: bool = False,
    use_texture_sdf_only: bool = False,
    use_identity_sdf_scale: bool = False,
):
    """Create the canonical mesh-endpoint strict-guard pass.

    This pass intentionally does not share the ordinary edge search's midpoint,
    bounding-sphere, or segmentation culls. Builder-authored endpoint ownership
    makes every mesh vertex appear exactly once per mode. Reduced variants
    compete inline for bounded spatial and predictive manifold slots.
    """
    if use_identity_sdf_scale and not use_texture_sdf_only:
        raise ValueError("identity SDF scale specialization requires texture-only SDFs")

    _, sample_sdf_at_t = _create_sdf_contact_funcs(
        enable_heightfields, use_texture_sdf_only, texture_sample_sdf_hw, _texture_sample_sdf_hw_pair
    )
    sample_grad = texture_sample_sdf_grad_only_hw
    get_mesh_edge_specialized = _create_mesh_edge_accessor_func(use_precomputed_edge_data)
    _module = (
        f"sdf_endpoint_guard_{writer_func.__name__}_{enable_heightfields}_{use_precomputed_edge_data}_"
        f"{use_texture_sdf_only}_{use_identity_sdf_scale}_{reduce_contacts}"
    )

    @wp.kernel(enable_backward=False, module=_module)
    def mesh_sdf_owned_endpoint_guard_kernel(
        shape_data: wp.array[wp.vec4],
        nonpenetration_oracle_eligible_shape: wp.array[wp.uint8],
        nonpenetration_oracle_anchor_shape: wp.array[wp.uint8],
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        texture_sdf_table: wp.array[TextureSDFData],
        shape_sdf_index: wp.array[wp.int32],
        shape_mesh_properties: wp.array[wp.int32],
        shape_base_gap: wp.array[float],
        shape_linear_velocity: wp.array[wp.vec3],
        shape_angular_velocity: wp.array[wp.vec3],
        shape_rotation_center_offset: wp.array[wp.vec3],
        collision_update_dt: float,
        max_speculative_extension: float,
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_voxel_resolution: wp.array[wp.vec3i],
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_mesh_count: wp.array[int],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        mesh_edge_indices: wp.array[wp.vec2i],
        mesh_edge_centers: wp.array[wp.vec4],
        mesh_edge_halves: wp.array[wp.vec4],
        shape_edge_range: wp.array[wp.vec2i],
        writer_data: Any,
        total_num_blocks: int,
    ):
        block_id, lane = wp.tid()
        pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])

        for pair_idx in range(block_id, pair_count, total_num_blocks):
            pair_encoded = shape_pairs_mesh_mesh[pair_idx]
            if wp.static(enable_heightfields):
                has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
                pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
            else:
                has_hfield = False
                pair = pair_encoded

            if _mesh_sdf_nonpenetration_oracle_owns_pair(
                pair[0],
                pair[1],
                nonpenetration_oracle_eligible_shape,
                nonpenetration_oracle_anchor_shape,
                shape_source,
                shape_body,
                body_flags,
            ):
                continue

            base_gap_sum = shape_base_gap[pair[0]] + shape_base_gap[pair[1]]
            for mode in range(2):
                tri_shape = pair[mode]
                sdf_shape = pair[1 - mode]
                if wp.static(enable_heightfields):
                    tri_is_hfield = has_hfield and mode == 0
                    sdf_is_hfield = has_hfield and mode == 1
                else:
                    tri_is_hfield = False
                    sdf_is_hfield = False

                # Heightfield edges have no canonical endpoint ownership map.
                if tri_is_hfield:
                    continue

                mesh_id_tri = shape_source[tri_shape]
                mesh_id_sdf = shape_source[sdf_shape]
                if mesh_id_tri == wp.uint64(0):
                    continue

                hfd_sdf = HeightfieldData()
                if wp.static(enable_heightfields):
                    if sdf_is_hfield:
                        hfd_sdf = heightfield_data[shape_heightfield_index[sdf_shape]]
                sdf_mesh_query_type = resolve_mesh_sign_method(shape_mesh_properties[sdf_shape])

                sdf_idx = int(-1)
                use_bvh_for_sdf = False
                if not sdf_is_hfield:
                    sdf_idx = shape_sdf_index[sdf_shape]
                    if wp.static(not use_texture_sdf_only):
                        use_bvh_for_sdf = sdf_idx < 0 or sdf_idx >= texture_sdf_table.shape[0]
                        if not use_bvh_for_sdf:
                            use_bvh_for_sdf = texture_sdf_table[sdf_idx].coarse_texture.width == 0
                        if use_bvh_for_sdf and mesh_id_sdf == wp.uint64(0):
                            continue

                scale_data_tri = shape_data[tri_shape]
                scale_data_sdf = shape_data[sdf_shape]
                mesh_scale_tri = wp.vec3(scale_data_tri[0], scale_data_tri[1], scale_data_tri[2])
                mesh_scale_sdf = wp.vec3(scale_data_sdf[0], scale_data_sdf[1], scale_data_sdf[2])
                X_tri_ws = shape_transform[tri_shape]
                X_sdf_ws = shape_transform[sdf_shape]

                texture_sdf = TextureSDFData()
                if sdf_is_hfield:
                    sdf_scale = wp.vec3(1.0)
                else:
                    if not use_bvh_for_sdf:
                        texture_sdf = texture_sdf_table[sdf_idx]
                    if wp.static(use_identity_sdf_scale):
                        sdf_scale = wp.vec3(1.0)
                    else:
                        sdf_scale = mesh_scale_sdf
                        if not use_bvh_for_sdf and texture_sdf.scale_baked:
                            sdf_scale = wp.vec3(1.0)

                if wp.static(use_identity_sdf_scale):
                    inv_sdf_scale = wp.vec3(1.0)
                    min_sdf_scale = float(1.0)
                else:
                    inv_sdf_scale, min_sdf_scale = safe_sdf_scale_inverse(sdf_scale)

                X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_sdf_ws), X_tri_ws)
                margin_sum = scale_data_tri[3] + scale_data_sdf[3]
                edge_range_tri = shape_edge_range[tri_shape]
                for edge_idx in range(lane, edge_range_tri[1], wp.block_dim()):
                    v0s, v1s, corner_ownership = get_mesh_edge_specialized(
                        mesh_id_tri,
                        mesh_edge_indices,
                        mesh_edge_centers,
                        mesh_edge_halves,
                        edge_range_tri,
                        mesh_scale_tri,
                        X_mesh_to_sdf,
                        edge_idx,
                    )
                    v0 = wp.cw_mul(v0s, inv_sdf_scale)
                    v1 = wp.cw_mul(v1s, inv_sdf_scale)
                    edge_dir = v1 - v0

                    for endpoint in range(1, 3):
                        if not mesh_sdf_contact_endpoint_owned(corner_ownership, endpoint):
                            continue

                        endpoint_t = float(endpoint - 1)
                        point_unscaled = v0 if endpoint == 1 else v1
                        distance_unscaled = sample_sdf_at_t(
                            texture_sdf,
                            mesh_id_sdf,
                            v0,
                            edge_dir,
                            endpoint_t,
                            use_bvh_for_sdf,
                            sdf_mesh_query_type,
                            sdf_is_hfield,
                            hfd_sdf,
                            heightfield_elevations,
                        )
                        distance_world = distance_unscaled
                        if wp.static(not use_identity_sdf_scale):
                            distance_world = distance_unscaled * min_sdf_scale
                        if not mesh_sdf_endpoint_guard_passes_admission(
                            distance_world, margin_sum, base_gap_sum, max_speculative_extension
                        ):
                            continue

                        if wp.static(enable_heightfields):
                            if sdf_is_hfield:
                                distance_unscaled, direction_unscaled = sample_sdf_grad_heightfield(
                                    hfd_sdf, heightfield_elevations, point_unscaled
                                )
                            elif wp.static(not use_texture_sdf_only) and use_bvh_for_sdf:
                                distance_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                    mesh_id_sdf, point_unscaled, _MESH_QUERY_MAX_DIST, sdf_mesh_query_type
                                )
                            else:
                                direction_unscaled = wp.static(sample_grad)(texture_sdf, point_unscaled)
                        else:
                            if wp.static(not use_texture_sdf_only) and use_bvh_for_sdf:
                                distance_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                    mesh_id_sdf, point_unscaled, _MESH_QUERY_MAX_DIST, sdf_mesh_query_type
                                )
                            else:
                                direction_unscaled = wp.static(sample_grad)(texture_sdf, point_unscaled)

                        if wp.static(use_identity_sdf_scale):
                            distance_world = distance_unscaled
                            direction = direction_unscaled
                            point = point_unscaled
                        else:
                            distance_world, direction = scale_sdf_result_to_world(
                                distance_unscaled, direction_unscaled, sdf_scale, inv_sdf_scale, min_sdf_scale
                            )
                            point = wp.cw_mul(point_unscaled, sdf_scale)
                        point_world = wp.transform_point(X_sdf_ws, point)
                        direction_world = wp.transform_vector(X_sdf_ws, direction)
                        direction_len_sq = wp.length_sq(direction_world)
                        if direction_len_sq > 0.0:
                            direction_world = direction_world * _sdf_rsqrt_rn(direction_len_sq)
                        else:
                            fallback_dir = point_world - wp.transform_get_translation(X_sdf_ws)
                            fallback_len_sq = wp.length_sq(fallback_dir)
                            if fallback_len_sq > 0.0:
                                direction_world = fallback_dir * _sdf_rsqrt_rn(fallback_len_sq)
                            else:
                                direction_world = wp.vec3(0.0, 1.0, 0.0)

                        contact_normal = -direction_world if mode == 0 else direction_world
                        contact_center = point_world - 0.5 * distance_world * direction_world
                        sort_sub_key = mesh_sdf_contact_sort_sub_key(edge_idx, 0, mode, endpoint)
                        sort_sub_key = pack_contact_is_canonical_endpoint(sort_sub_key, True)
                        sort_sub_key = pack_contact_is_strict_guard(sort_sub_key, True)

                        if wp.static(reduce_contacts):
                            voxel_shape = mesh_sdf_contact_voxel_owner(
                                pair[0],
                                pair[1],
                                shape_collision_aabb_lower,
                                shape_collision_aabb_upper,
                                shape_voxel_resolution,
                            )
                            X_voxel_ws = shape_transform[voxel_shape]
                            position_local_voxel = wp.quat_rotate_inv(
                                wp.transform_get_rotation(X_voxel_ws),
                                contact_center - wp.transform_get_translation(X_voxel_ws),
                            )
                            pair_midpoint = 0.5 * (
                                wp.transform_get_translation(shape_transform[pair[0]])
                                + wp.transform_get_translation(shape_transform[pair[1]])
                            )
                            clearance = distance_world - margin_sum
                            is_speculative_shell = clearance > 0.0
                            point_a = contact_center - 0.5 * distance_world * contact_normal
                            point_b = contact_center + 0.5 * distance_world * contact_normal
                            swept_separation_lower_bound = compute_contact_swept_separation_lower_bound(
                                pair[0],
                                pair[1],
                                point_a,
                                point_b,
                                contact_normal,
                                unpack_contact_normal_owner(sort_sub_key),
                                margin_sum,
                                shape_transform,
                                shape_linear_velocity,
                                shape_angular_velocity,
                                shape_rotation_center_offset,
                                collision_update_dt,
                            )
                            contact_id = export_and_reduce_contact_centered_two_spatial_depths(
                                pair[0],
                                pair[1],
                                contact_center,
                                contact_normal,
                                distance_world,
                                sort_sub_key,
                                contact_center - pair_midpoint,
                                margin_sum,
                                margin_sum + base_gap_sum,
                                is_speculative_shell,
                                True,
                                swept_separation_lower_bound,
                                position_local_voxel,
                                shape_collision_aabb_lower[voxel_shape],
                                shape_collision_aabb_upper[voxel_shape],
                                shape_voxel_resolution[voxel_shape],
                                writer_data,
                            )
                            export_and_reduce_predictive_contact(
                                pair[0],
                                pair[1],
                                contact_center,
                                contact_normal,
                                distance_world,
                                margin_sum,
                                base_gap_sum,
                                0.0,
                                0.0,
                                sort_sub_key,
                                shape_transform,
                                shape_linear_velocity,
                                shape_angular_velocity,
                                shape_rotation_center_offset,
                                collision_update_dt,
                                max_speculative_extension,
                                contact_id,
                                True,
                                writer_data,
                            )
                        else:
                            contact_data = ContactData()
                            contact_data.contact_point_center = contact_center
                            contact_data.contact_normal_a_to_b = contact_normal
                            contact_data.contact_distance = distance_world
                            contact_data.radius_eff_a = 0.0
                            contact_data.radius_eff_b = 0.0
                            contact_data.margin_a = shape_data[pair[0]][3]
                            contact_data.margin_b = shape_data[pair[1]][3]
                            contact_data.shape_a = pair[0]
                            contact_data.shape_b = pair[1]
                            contact_data.gap_sum = base_gap_sum
                            contact_data.sort_sub_key = sort_sub_key
                            contact_data.strict_guard_provenance_finalized = 2
                            writer_func(contact_data, writer_data, -1)

    return mesh_sdf_owned_endpoint_guard_kernel


def create_mesh_sdf_nonpenetration_oracle_kernels(materialize: bool):
    """Create swept-path and cooperative endpoint nonpenetration queries.

    One cooperative block visits all source ``wp.Mesh`` points, authored triangle
    edges, and swept triangles in both directed modes of a shape pair. Raw signed-
    point queries detect vertex containment, while finite segment-to-mesh rays
    detect transverse boundary intersections whose vertices all remain outside
    and construct reference-connected material-point guards.
    Any raw vertex containment or authored-edge intersection at the reference
    pose marks the owned pair infeasible and prevents candidate materialization
    or sweep validation. Invalid reference geometry also fails closed. For a
    stationary target and a source carrying an explicit linear-translation
    path certificate, raw triangle BVH queries and continuous separating-axis
    intervals certify the complete reference-to-candidate translation.
    Solver-provided origin and angular path bounds conservatively route all
    other supported motion; overlapping swept triangle envelopes report
    incomplete. The endpoint query
    has no predictive-shell or contact-reduction dependency and emits at most one
    feature guard from each direction.
    """
    _module = f"sdf_nonpenetration_oracle_{materialize}"

    @wp.func(module=_module)
    def _oracle_feature_key(
        source_shape: int,
        target_shape: int,
        source_vertex: int,
        shape_key_bits: int,
        vertex_key_max: wp.uint64,
    ) -> tuple[bool, wp.uint64]:
        """Pack one directed vertex-to-mesh separation constraint exactly."""
        if source_vertex < 0 or wp.uint64(source_vertex) > vertex_key_max:
            return False, wp.uint64(0)
        shape_mask = (wp.uint64(1) << wp.uint64(shape_key_bits)) - wp.uint64(1)
        if wp.uint64(source_shape) > shape_mask or wp.uint64(target_shape) > shape_mask:
            return False, wp.uint64(0)
        key = wp.uint64(source_shape)
        key |= wp.uint64(target_shape) << wp.uint64(shape_key_bits)
        key |= wp.uint64(source_vertex) << wp.uint64(2 * shape_key_bits)
        return key != _MESH_SDF_ORACLE_EMPTY_FEATURE, key

    @wp.func(module=_module)
    def _upsert_oracle_feature(
        world_id: int,
        key: wp.uint64,
        slots_per_world: int,
        feature_keys: wp.array[wp.uint64],
        feature_contact_indices: wp.array[int],
        feature_update_generations: wp.array[int],
        update_generation: int,
        contact_count: wp.array[int],
        contact_max: int,
    ) -> int:
        """Return the dense contact row for a new or retained exact feature."""
        mixed = key
        mixed = mixed ^ (mixed >> wp.uint64(33))
        mixed = mixed * _MESH_SDF_ORACLE_FEATURE_HASH_MULTIPLIER
        mixed = mixed ^ (mixed >> wp.uint64(33))
        slot_mask = slots_per_world - 1
        slot = int(mixed) & slot_mask
        world_offset = world_id * slots_per_world
        for probe in range(slots_per_world):
            table_index = world_offset + ((slot + probe) & slot_mask)
            stored_key = feature_keys[table_index]
            if stored_key == key:
                # A previous refinement owns this nonlinear feature.  Update
                # its linearization once in place instead of adding another
                # row. Duplicate/reversed explicit pair blocks may reach the
                # same directed key concurrently.
                previous_generation = wp.atomic_exch(
                    feature_update_generations,
                    table_index,
                    update_generation,
                )
                if previous_generation == update_generation:
                    return -1
                return feature_contact_indices[table_index]
            if stored_key == _MESH_SDF_ORACLE_EMPTY_FEATURE:
                previous = wp.atomic_cas(
                    feature_keys,
                    table_index,
                    _MESH_SDF_ORACLE_EMPTY_FEATURE,
                    key,
                )
                if previous == _MESH_SDF_ORACLE_EMPTY_FEATURE:
                    contact_index = wp.atomic_add(contact_count, 0, 1)
                    if contact_index >= contact_max:
                        feature_update_generations[table_index] = update_generation
                        feature_contact_indices[table_index] = -2
                        return -2
                    feature_update_generations[table_index] = update_generation
                    feature_contact_indices[table_index] = contact_index
                    return contact_index
                if previous == key:
                    # Candidate routing emits each directed pair once.  This
                    # branch only protects duplicate input pairs in the same
                    # launch; the claiming lane will materialize the row.
                    return -1
        return -2

    _reference_pair_infeasible = wp.uint8(1)
    _reference_pair_invalid = wp.uint8(2)

    @wp.func(module=_module)
    def _query_endpoint(
        source_shape: int,
        target_shape: int,
        source_point_local: wp.vec3,
        shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_source: wp.array[wp.uint64],
        shape_mesh_properties: wp.array[wp.int32],
    ) -> tuple[bool, float, wp.vec3, wp.vec3]:
        X_source_ws = shape_transform[source_shape]
        X_target_ws = shape_transform[target_shape]
        source_point_world = wp.transform_point(X_source_ws, source_point_local)
        source_point_target = wp.transform_point(wp.transform_inverse(X_target_ws), source_point_world)

        target_mesh_id = shape_source[target_shape]
        if target_mesh_id == wp.uint64(0):
            return False, 0.0, wp.vec3(0.0), source_point_world

        sdf_scale = shape_scale[target_shape]
        inv_sdf_scale, min_sdf_scale = safe_sdf_scale_inverse(sdf_scale)
        query_point = wp.cw_mul(source_point_target, inv_sdf_scale)
        distance_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
            target_mesh_id,
            query_point,
            _MESH_QUERY_MAX_DIST,
            resolve_mesh_sign_method(shape_mesh_properties[target_shape]),
        )

        distance_world, direction_target = scale_sdf_result_to_world(
            distance_unscaled,
            direction_unscaled,
            sdf_scale,
            inv_sdf_scale,
            min_sdf_scale,
        )
        direction_world = wp.transform_vector(X_target_ws, direction_target)
        direction_length_sq = wp.length_sq(direction_world)
        valid = (
            wp.isfinite(distance_world)
            and wp.isfinite(direction_world[0])
            and wp.isfinite(direction_world[1])
            and wp.isfinite(direction_world[2])
            and direction_length_sq > 1.0e-20
        )
        if valid:
            direction_world *= _sdf_rsqrt_rn(direction_length_sq)
        return valid, distance_world, direction_world, source_point_world

    @wp.func(module=_module)
    def _valid_mesh_scale(scale: wp.vec3) -> bool:
        """Return whether an affine mesh scale has a finite, invertible linear part."""
        return (
            wp.isfinite(scale[0])
            and wp.isfinite(scale[1])
            and wp.isfinite(scale[2])
            and wp.abs(scale[0]) > 1.0e-10
            and wp.abs(scale[1]) > 1.0e-10
            and wp.abs(scale[2]) > 1.0e-10
        )

    @wp.func(module=_module)
    def _finite_point(point: wp.vec3) -> bool:
        """Return whether all coordinates of one point are finite."""
        return wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])

    @wp.func(module=_module)
    def _triangle_edge(point0: wp.vec3, point1: wp.vec3, point2: wp.vec3, edge: int) -> wp.vec3:
        """Return one directed triangle edge."""
        if edge == 0:
            return point1 - point0
        if edge == 1:
            return point2 - point1
        return point0 - point2

    @wp.func(module=_module)
    def _swept_obb_axis_separation(
        axis: wp.vec3,
        source_reference_center: wp.vec3,
        source_candidate_center: wp.vec3,
        source_axis0: wp.vec3,
        source_axis1: wp.vec3,
        source_axis2: wp.vec3,
        source_inflation: float,
        target_center: wp.vec3,
        target_axis0: wp.vec3,
        target_axis1: wp.vec3,
        target_axis2: wp.vec3,
        target_inflation: float,
        tolerance: float,
    ) -> int:
        """Test one world-space axis against conservative swept OBB intervals."""
        axis_length_sq = wp.length_sq(axis)
        if not wp.isfinite(axis_length_sq):
            return -1
        if axis_length_sq <= 1.0e-20:
            return 0
        unit_axis = axis * _sdf_rsqrt_rn(axis_length_sq)
        source_radius = (
            wp.abs(wp.dot(unit_axis, source_axis0))
            + wp.abs(wp.dot(unit_axis, source_axis1))
            + wp.abs(wp.dot(unit_axis, source_axis2))
            + source_inflation
        )
        target_radius = (
            wp.abs(wp.dot(unit_axis, target_axis0))
            + wp.abs(wp.dot(unit_axis, target_axis1))
            + wp.abs(wp.dot(unit_axis, target_axis2))
            + target_inflation
        )
        source_reference_projection = wp.dot(unit_axis, source_reference_center)
        source_candidate_projection = wp.dot(unit_axis, source_candidate_center)
        target_projection = wp.dot(unit_axis, target_center)
        finite = (
            wp.isfinite(source_radius)
            and wp.isfinite(target_radius)
            and wp.isfinite(source_reference_projection)
            and wp.isfinite(source_candidate_projection)
            and wp.isfinite(target_projection)
        )
        if not finite:
            return -1
        source_min = wp.min(source_reference_projection, source_candidate_projection) - source_radius
        source_max = wp.max(source_reference_projection, source_candidate_projection) + source_radius
        target_min = target_projection - target_radius
        target_max = target_projection + target_radius
        if source_max < target_min - tolerance or target_max < source_min - tolerance:
            return 1
        return 0

    @wp.func(module=_module)
    def _query_swept_obb_separation(
        source_shape: int,
        target_shape: int,
        reference_shape_transform: wp.array[wp.transform],
        candidate_shape_transform: wp.array[wp.transform],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_margin: wp.array[float],
        shape_motion_radius: wp.array[float],
        tolerance: float,
    ) -> int:
        """Certify separation of a swept source OBB from a stationary target OBB."""
        source_lower = shape_collision_aabb_lower[source_shape]
        source_upper = shape_collision_aabb_upper[source_shape]
        target_lower = shape_collision_aabb_lower[target_shape]
        target_upper = shape_collision_aabb_upper[target_shape]
        source_local_center = 0.5 * (source_lower + source_upper)
        source_half = 0.5 * (source_upper - source_lower)
        target_local_center = 0.5 * (target_lower + target_upper)
        target_half = 0.5 * (target_upper - target_lower)
        if (
            not _finite_point(source_local_center)
            or not _finite_point(source_half)
            or not _finite_point(target_local_center)
            or not _finite_point(target_half)
            or wp.min(source_half) < 0.0
            or wp.min(target_half) < 0.0
        ):
            return -1

        X_source_reference = reference_shape_transform[source_shape]
        X_source_candidate = candidate_shape_transform[source_shape]
        X_target_reference = reference_shape_transform[target_shape]
        X_target_candidate = candidate_shape_transform[target_shape]
        source_reference_center = wp.transform_point(X_source_reference, source_local_center)
        source_candidate_center = wp.transform_point(X_source_candidate, source_local_center)
        target_center = wp.transform_point(X_target_reference, target_local_center)
        source_axis0 = wp.transform_vector(X_source_reference, wp.vec3(source_half[0], 0.0, 0.0))
        source_axis1 = wp.transform_vector(X_source_reference, wp.vec3(0.0, source_half[1], 0.0))
        source_axis2 = wp.transform_vector(X_source_reference, wp.vec3(0.0, 0.0, source_half[2]))
        target_axis0 = wp.transform_vector(X_target_reference, wp.vec3(target_half[0], 0.0, 0.0))
        target_axis1 = wp.transform_vector(X_target_reference, wp.vec3(0.0, target_half[1], 0.0))
        target_axis2 = wp.transform_vector(X_target_reference, wp.vec3(0.0, 0.0, target_half[2]))
        finite = (
            _finite_point(source_reference_center)
            and _finite_point(source_candidate_center)
            and _finite_point(target_center)
            and _finite_point(source_axis0)
            and _finite_point(source_axis1)
            and _finite_point(source_axis2)
            and _finite_point(target_axis0)
            and _finite_point(target_axis1)
            and _finite_point(target_axis2)
        )
        if not finite:
            return -1

        source_radius = shape_motion_radius[source_shape]
        target_radius = shape_motion_radius[target_shape]
        source_rotation_valid, source_inflation = _rotation_displacement_bound(
            wp.transform_get_rotation(X_source_reference),
            wp.transform_get_rotation(X_source_candidate),
            source_radius,
        )
        target_rotation_valid, target_inflation = _rotation_displacement_bound(
            wp.transform_get_rotation(X_target_reference),
            wp.transform_get_rotation(X_target_candidate),
            target_radius,
        )
        target_inflation += wp.length(
            wp.transform_get_translation(X_target_candidate) - wp.transform_get_translation(X_target_reference)
        )
        source_inflation += shape_margin[source_shape]
        target_inflation += shape_margin[target_shape]
        if (
            not source_rotation_valid
            or not target_rotation_valid
            or not wp.isfinite(source_inflation)
            or not wp.isfinite(target_inflation)
        ):
            return -1

        for axis_family in range(4):
            for axis_index in range(3):
                source_axis = source_axis0 if axis_index == 0 else source_axis1 if axis_index == 1 else source_axis2
                target_axis = target_axis0 if axis_index == 0 else target_axis1 if axis_index == 1 else target_axis2
                if axis_family == 0:
                    axis = wp.cross(
                        source_axis1 if axis_index == 0 else source_axis2 if axis_index == 1 else source_axis0,
                        source_axis2 if axis_index == 0 else source_axis0 if axis_index == 1 else source_axis1,
                    )
                elif axis_family == 1:
                    axis = wp.cross(
                        target_axis1 if axis_index == 0 else target_axis2 if axis_index == 1 else target_axis0,
                        target_axis2 if axis_index == 0 else target_axis0 if axis_index == 1 else target_axis1,
                    )
                elif axis_family == 2:
                    axis = wp.cross(source_axis, source_candidate_center - source_reference_center)
                else:
                    axis = wp.cross(target_axis, source_candidate_center - source_reference_center)
                separation = _swept_obb_axis_separation(
                    axis,
                    source_reference_center,
                    source_candidate_center,
                    source_axis0,
                    source_axis1,
                    source_axis2,
                    source_inflation,
                    target_center,
                    target_axis0,
                    target_axis1,
                    target_axis2,
                    target_inflation,
                    tolerance,
                )
                if separation != 0:
                    return separation

        for source_index in range(3):
            source_axis = source_axis0 if source_index == 0 else source_axis1 if source_index == 1 else source_axis2
            for target_index in range(3):
                target_axis = target_axis0 if target_index == 0 else target_axis1 if target_index == 1 else target_axis2
                separation = _swept_obb_axis_separation(
                    wp.cross(source_axis, target_axis),
                    source_reference_center,
                    source_candidate_center,
                    source_axis0,
                    source_axis1,
                    source_axis2,
                    source_inflation,
                    target_center,
                    target_axis0,
                    target_axis1,
                    target_axis2,
                    target_inflation,
                    tolerance,
                )
                if separation != 0:
                    return separation
        return 0

    @wp.func(module=_module)
    def _update_translation_overlap_intervals(
        axis: wp.vec3,
        source0: wp.vec3,
        source1: wp.vec3,
        source2: wp.vec3,
        target0: wp.vec3,
        target1: wp.vec3,
        target2: wp.vec3,
        translation: wp.vec3,
        padding: float,
        physical_start: float,
        physical_end: float,
        padded_start: float,
        padded_end: float,
    ) -> tuple[int, float, float, float, float]:
        """Intersect physical and padded SAT overlap intervals along one axis."""
        axis_length_sq = wp.length_sq(axis)
        if not wp.isfinite(axis_length_sq):
            return -1, physical_start, physical_end, padded_start, padded_end
        if axis_length_sq <= 1.0e-20:
            return 1, physical_start, physical_end, padded_start, padded_end
        unit_axis = axis * _sdf_rsqrt_rn(axis_length_sq)
        source_projection0 = wp.dot(source0, unit_axis)
        source_projection1 = wp.dot(source1, unit_axis)
        source_projection2 = wp.dot(source2, unit_axis)
        target_projection0 = wp.dot(target0, unit_axis)
        target_projection1 = wp.dot(target1, unit_axis)
        target_projection2 = wp.dot(target2, unit_axis)
        speed = wp.dot(translation, unit_axis)
        finite = (
            wp.isfinite(source_projection0)
            and wp.isfinite(source_projection1)
            and wp.isfinite(source_projection2)
            and wp.isfinite(target_projection0)
            and wp.isfinite(target_projection1)
            and wp.isfinite(target_projection2)
            and wp.isfinite(speed)
        )
        if not finite:
            return -1, physical_start, physical_end, padded_start, padded_end

        source_min = wp.min(source_projection0, wp.min(source_projection1, source_projection2))
        source_max = wp.max(source_projection0, wp.max(source_projection1, source_projection2))
        target_min = wp.min(target_projection0, wp.min(target_projection1, target_projection2))
        target_max = wp.max(target_projection0, wp.max(target_projection1, target_projection2))
        if speed == 0.0:
            if source_max < target_min or target_max < source_min:
                return 0, physical_start, physical_end, padded_start, padded_end
            return 1, physical_start, physical_end, padded_start, padded_end

        physical_crossing0 = (target_min - source_max) / speed
        physical_crossing1 = (target_max - source_min) / speed
        physical_axis_start = wp.min(physical_crossing0, physical_crossing1)
        physical_axis_end = wp.max(physical_crossing0, physical_crossing1)
        physical_start = wp.max(physical_start, physical_axis_start)
        physical_end = wp.min(physical_end, physical_axis_end)
        if physical_end < physical_start:
            return 0, physical_start, physical_end, padded_start, padded_end

        padded_crossing0 = (target_min - padding - source_max) / speed
        padded_crossing1 = (target_max + padding - source_min) / speed
        padded_axis_start = wp.min(padded_crossing0, padded_crossing1)
        padded_axis_end = wp.max(padded_crossing0, padded_crossing1)
        padded_start = wp.max(padded_start, padded_axis_start)
        padded_end = wp.min(padded_end, padded_axis_end)
        if padded_end < padded_start:
            return 0, physical_start, physical_end, padded_start, padded_end
        return 1, physical_start, physical_end, padded_start, padded_end

    _translation_fraction_retreat = float(8.0 * 1.1920928955078125e-7)

    @wp.func(module=_module)
    def _continuous_triangle_translation_overlap(
        source0: wp.vec3,
        source1: wp.vec3,
        source2: wp.vec3,
        source0_candidate: wp.vec3,
        source1_candidate: wp.vec3,
        source2_candidate: wp.vec3,
        target0: wp.vec3,
        target1: wp.vec3,
        target2: wp.vec3,
        consistency_tolerance: float,
        padding: float,
    ) -> tuple[int, float]:
        """Classify overlap and return a certified clear translation prefix.

        Returns ``1`` for a positive-duration interior-path overlap, ``0`` for
        a certified clear interval, and ``-1`` for invalid or degenerate data.
        The fraction retreats from the first padded SAT overlap entry.
        """
        translation = source0_candidate - source0
        translation1 = source1_candidate - source1
        translation2 = source2_candidate - source2
        consistency_tolerance = wp.max(consistency_tolerance, 1.0e-7)
        if (
            wp.length_sq(translation1 - translation) > consistency_tolerance * consistency_tolerance
            or wp.length_sq(translation2 - translation) > consistency_tolerance * consistency_tolerance
        ):
            return -1, 0.0
        motion_length_sq = wp.length_sq(translation)
        if not wp.isfinite(motion_length_sq):
            return -1, 0.0
        if motion_length_sq == 0.0:
            return 0, 1.0

        source_edge0 = source1 - source0
        target_edge0 = target1 - target0
        source_normal = wp.cross(source_edge0, source2 - source0)
        target_normal = wp.cross(target_edge0, target2 - target0)
        source_normal_sq = wp.length_sq(source_normal)
        target_normal_sq = wp.length_sq(target_normal)
        if (
            not wp.isfinite(source_normal_sq)
            or not wp.isfinite(target_normal_sq)
            or source_normal_sq <= 1.0e-20
            or target_normal_sq <= 1.0e-20
        ):
            return -1, 0.0

        source_unit_normal = source_normal * _sdf_rsqrt_rn(source_normal_sq)
        target_unit_normal = target_normal * _sdf_rsqrt_rn(target_normal_sq)
        source_normal_displacement = wp.abs(wp.dot(translation, source_unit_normal))
        target_normal_displacement = wp.abs(wp.dot(translation, target_unit_normal))
        normal_cross = wp.cross(source_unit_normal, target_unit_normal)
        normal_cross_sq = wp.length_sq(normal_cross)
        if (
            not wp.isfinite(source_normal_displacement)
            or not wp.isfinite(target_normal_displacement)
            or not wp.isfinite(normal_cross_sq)
        ):
            return -1, 0.0

        physical_start = float(0.0)
        physical_end = float(1.0)
        padded_start = float(0.0)
        padded_end = float(1.0)
        status, physical_start, physical_end, padded_start, padded_end = _update_translation_overlap_intervals(
            source_normal,
            source0,
            source1,
            source2,
            target0,
            target1,
            target2,
            translation,
            padding,
            physical_start,
            physical_end,
            padded_start,
            padded_end,
        )
        if status <= 0:
            return status, 0.0 if status < 0 else 1.0
        status, physical_start, physical_end, padded_start, padded_end = _update_translation_overlap_intervals(
            target_normal,
            source0,
            source1,
            source2,
            target0,
            target1,
            target2,
            translation,
            padding,
            physical_start,
            physical_end,
            padded_start,
            padded_end,
        )
        if status <= 0:
            return status, 0.0 if status < 0 else 1.0

        # Coplanar triangle separation needs the six face-normal cross edge
        # axes in addition to the two face normals and nine edge cross axes.
        for edge_index in range(3):
            source_edge = _triangle_edge(source0, source1, source2, edge_index)
            target_edge = _triangle_edge(target0, target1, target2, edge_index)
            status, physical_start, physical_end, padded_start, padded_end = _update_translation_overlap_intervals(
                wp.cross(source_normal, source_edge),
                source0,
                source1,
                source2,
                target0,
                target1,
                target2,
                translation,
                padding,
                physical_start,
                physical_end,
                padded_start,
                padded_end,
            )
            if status <= 0:
                return status, 0.0 if status < 0 else 1.0
            status, physical_start, physical_end, padded_start, padded_end = _update_translation_overlap_intervals(
                wp.cross(target_normal, target_edge),
                source0,
                source1,
                source2,
                target0,
                target1,
                target2,
                translation,
                padding,
                physical_start,
                physical_end,
                padded_start,
                padded_end,
            )
            if status <= 0:
                return status, 0.0 if status < 0 else 1.0

        for source_edge_index in range(3):
            source_edge = _triangle_edge(source0, source1, source2, source_edge_index)
            for target_edge_index in range(3):
                target_edge = _triangle_edge(target0, target1, target2, target_edge_index)
                status, physical_start, physical_end, padded_start, padded_end = _update_translation_overlap_intervals(
                    wp.cross(source_edge, target_edge),
                    source0,
                    source1,
                    source2,
                    target0,
                    target1,
                    target2,
                    translation,
                    padding,
                    physical_start,
                    physical_end,
                    padded_start,
                    padded_end,
                )
                if status <= 0:
                    return status, 0.0 if status < 0 else 1.0

        physical_interior_start = wp.max(physical_start, 0.0)
        physical_interior_end = wp.min(physical_end, 1.0)
        if (
            physical_interior_end <= physical_interior_start
            or physical_interior_end <= 0.0
            or physical_interior_start >= 1.0
        ):
            return 0, 1.0
        padded_interior_start = wp.max(padded_start, 0.0)
        padded_interior_end = wp.min(padded_end, 1.0)
        if padded_interior_end <= padded_interior_start:
            return -1, 0.0
        coordinate_scale = wp.max(
            wp.max(wp.max(wp.length(source0), wp.length(source1)), wp.length(source2)),
            wp.max(wp.max(wp.length(target0), wp.length(target1)), wp.max(wp.length(target2), wp.length(translation))),
        )
        geometric_tolerance = 4.0 * 1.1920928955078125e-7 * wp.max(coordinate_scale, 1.0e-6)
        source_target_distance0 = wp.dot(source0 - target0, target_unit_normal)
        source_target_distance1 = wp.dot(source1 - target0, target_unit_normal)
        source_target_distance2 = wp.dot(source2 - target0, target_unit_normal)
        source_target_min = wp.min(source_target_distance0, wp.min(source_target_distance1, source_target_distance2))
        source_target_max = wp.max(source_target_distance0, wp.max(source_target_distance1, source_target_distance2))
        target_source_distance0 = wp.dot(target0 - source0, source_unit_normal)
        target_source_distance1 = wp.dot(target1 - source0, source_unit_normal)
        target_source_distance2 = wp.dot(target2 - source0, source_unit_normal)
        target_source_min = wp.min(target_source_distance0, wp.min(target_source_distance1, target_source_distance2))
        target_source_max = wp.max(target_source_distance0, wp.max(target_source_distance1, target_source_distance2))
        tangent_one_sided = (
            target_normal_displacement <= geometric_tolerance
            and (source_target_min >= -geometric_tolerance or source_target_max <= geometric_tolerance)
        ) or (
            source_normal_displacement <= geometric_tolerance
            and (target_source_min >= -geometric_tolerance or target_source_max <= geometric_tolerance)
        )
        if tangent_one_sided and (
            normal_cross_sq > 1.0e-12 or physical_interior_start <= _translation_fraction_retreat
        ):
            # The triangle already touches a supporting plane, stays on one
            # side of it, and moves tangent to it. Nonparallel faces can only
            # retain boundary contact; parallel coplanar faces are ignored
            # only when that contact exists at the reference pose.
            return 0, 1.0
        if normal_cross_sq > 1.0e-12:
            intersection_axis = normal_cross * _sdf_rsqrt_rn(normal_cross_sq)
            source_axis0 = wp.dot(source0, intersection_axis)
            source_axis1 = wp.dot(source1, intersection_axis)
            source_axis2 = wp.dot(source2, intersection_axis)
            target_axis0 = wp.dot(target0, intersection_axis)
            target_axis1 = wp.dot(target1, intersection_axis)
            target_axis2 = wp.dot(target2, intersection_axis)
            source_axis_min = wp.min(source_axis0, wp.min(source_axis1, source_axis2))
            source_axis_max = wp.max(source_axis0, wp.max(source_axis1, source_axis2))
            target_axis_min = wp.min(target_axis0, wp.min(target_axis1, target_axis2))
            target_axis_max = wp.max(target_axis0, wp.max(target_axis1, target_axis2))
            intersection_overlap = wp.min(source_axis_max, target_axis_max) - wp.max(source_axis_min, target_axis_min)
            intersection_displacement = wp.abs(wp.dot(translation, intersection_axis))
            if intersection_displacement <= geometric_tolerance and intersection_overlap <= geometric_tolerance:
                # The planes intersect, but their finite triangles can meet
                # only at a stationary endpoint of that intersection line.
                return 0, 1.0
        if padded_interior_start <= _translation_fraction_retreat:
            return 1, 0.0
        return 1, wp.clamp(padded_interior_start - _translation_fraction_retreat, 0.0, 1.0)

    @wp.func(module=_module)
    def _query_pure_translation_overlap(
        source_shape: int,
        target_shape: int,
        source_triangle_start: int,
        source_triangle_stride: int,
        reference_shape_transform: wp.array[wp.transform],
        candidate_shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_margin: wp.array[float],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_motion_radius: wp.array[float],
        shape_source: wp.array[wp.uint64],
        tolerance: float,
        target_relative: bool,
    ) -> tuple[int, int, float]:
        """Return the earliest padded overlap for one strided translation query."""
        source_mesh_id = shape_source[source_shape]
        target_mesh_id = shape_source[target_shape]
        source_scale = shape_scale[source_shape]
        target_scale = shape_scale[target_shape]
        source_margin = shape_margin[source_shape]
        target_margin = shape_margin[target_shape]
        if (
            source_mesh_id == wp.uint64(0)
            or target_mesh_id == wp.uint64(0)
            or not _valid_mesh_scale(source_scale)
            or not _valid_mesh_scale(target_scale)
            or not wp.isfinite(source_margin)
            or not wp.isfinite(target_margin)
            or source_margin < 0.0
            or target_margin < 0.0
        ):
            return -1, -1, 0.0
        source_mesh = wp.mesh_get(source_mesh_id)
        target_mesh = wp.mesh_get(target_mesh_id)
        source_index_count = source_mesh.indices.shape[0]
        target_index_count = target_mesh.indices.shape[0]
        if (
            source_index_count < 3
            or source_index_count % 3 != 0
            or target_index_count < 3
            or target_index_count % 3 != 0
        ):
            return -1, -1, 0.0

        X_source_reference = reference_shape_transform[source_shape]
        X_source_candidate = candidate_shape_transform[source_shape]
        X_target = reference_shape_transform[target_shape]
        if target_relative:
            X_source_candidate = wp.transform_multiply(
                X_target,
                wp.transform_multiply(
                    wp.transform_inverse(candidate_shape_transform[target_shape]),
                    X_source_candidate,
                ),
            )
        X_target_inverse = wp.transform_inverse(X_target)
        target_scale_inverse, target_min_scale = safe_sdf_scale_inverse(target_scale)
        local_tolerance = tolerance / target_min_scale
        local_padding = (source_margin + target_margin + tolerance) / target_min_scale
        if not target_relative:
            bounds_separation = _query_swept_obb_separation(
                source_shape,
                target_shape,
                reference_shape_transform,
                candidate_shape_transform,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_margin,
                shape_motion_radius,
                tolerance,
            )
            if bounds_separation < 0:
                return -1, -1, 0.0
            if bounds_separation > 0:
                return 0, 2147483647, 1.0
        source_triangle_count = source_index_count // 3
        decisive_triangle = int(2147483647)
        certified_fraction = float(1.0)
        for source_triangle in range(source_triangle_start, source_triangle_count, source_triangle_stride):
            source_index0 = source_mesh.indices[3 * source_triangle]
            source_index1 = source_mesh.indices[3 * source_triangle + 1]
            source_index2 = source_mesh.indices[3 * source_triangle + 2]
            if (
                source_index0 < 0
                or source_index0 >= source_mesh.points.shape[0]
                or source_index1 < 0
                or source_index1 >= source_mesh.points.shape[0]
                or source_index2 < 0
                or source_index2 >= source_mesh.points.shape[0]
            ):
                return -1, source_triangle, 0.0

            source_local0 = wp.cw_mul(source_mesh.points[source_index0], source_scale)
            source_local1 = wp.cw_mul(source_mesh.points[source_index1], source_scale)
            source_local2 = wp.cw_mul(source_mesh.points[source_index2], source_scale)
            source_reference0 = wp.cw_mul(
                wp.transform_point(X_target_inverse, wp.transform_point(X_source_reference, source_local0)),
                target_scale_inverse,
            )
            source_reference1 = wp.cw_mul(
                wp.transform_point(X_target_inverse, wp.transform_point(X_source_reference, source_local1)),
                target_scale_inverse,
            )
            source_reference2 = wp.cw_mul(
                wp.transform_point(X_target_inverse, wp.transform_point(X_source_reference, source_local2)),
                target_scale_inverse,
            )
            source_candidate0 = wp.cw_mul(
                wp.transform_point(X_target_inverse, wp.transform_point(X_source_candidate, source_local0)),
                target_scale_inverse,
            )
            source_candidate1 = wp.cw_mul(
                wp.transform_point(X_target_inverse, wp.transform_point(X_source_candidate, source_local1)),
                target_scale_inverse,
            )
            source_candidate2 = wp.cw_mul(
                wp.transform_point(X_target_inverse, wp.transform_point(X_source_candidate, source_local2)),
                target_scale_inverse,
            )
            if (
                not _finite_point(source_reference0)
                or not _finite_point(source_reference1)
                or not _finite_point(source_reference2)
                or not _finite_point(source_candidate0)
                or not _finite_point(source_candidate1)
                or not _finite_point(source_candidate2)
            ):
                return -1, source_triangle, 0.0
            query_lower = wp.min(
                wp.min(source_reference0, wp.min(source_reference1, source_reference2)),
                wp.min(source_candidate0, wp.min(source_candidate1, source_candidate2)),
            ) - wp.vec3(local_padding)
            query_upper = wp.max(
                wp.max(source_reference0, wp.max(source_reference1, source_reference2)),
                wp.max(source_candidate0, wp.max(source_candidate1, source_candidate2)),
            ) + wp.vec3(local_padding)
            if not _finite_point(query_lower) or not _finite_point(query_upper):
                return -1, source_triangle, 0.0
            query = wp.mesh_query_aabb(target_mesh_id, query_lower, query_upper)
            target_triangle = wp.int32(0)
            while wp.mesh_query_aabb_next(query, target_triangle):
                target_index0 = target_mesh.indices[3 * target_triangle]
                target_index1 = target_mesh.indices[3 * target_triangle + 1]
                target_index2 = target_mesh.indices[3 * target_triangle + 2]
                if (
                    target_index0 < 0
                    or target_index0 >= target_mesh.points.shape[0]
                    or target_index1 < 0
                    or target_index1 >= target_mesh.points.shape[0]
                    or target_index2 < 0
                    or target_index2 >= target_mesh.points.shape[0]
                ):
                    return -1, source_triangle, 0.0
                target0 = target_mesh.points[target_index0]
                target1 = target_mesh.points[target_index1]
                target2 = target_mesh.points[target_index2]
                if not _finite_point(target0) or not _finite_point(target1) or not _finite_point(target2):
                    return -1, source_triangle, 0.0
                overlap, pair_certified_fraction = _continuous_triangle_translation_overlap(
                    source_reference0,
                    source_reference1,
                    source_reference2,
                    source_candidate0,
                    source_candidate1,
                    source_candidate2,
                    target0,
                    target1,
                    target2,
                    local_tolerance,
                    local_padding,
                )
                if overlap < 0:
                    return -1, source_triangle, 0.0
                if overlap > 0 and pair_certified_fraction < certified_fraction:
                    decisive_triangle = source_triangle
                    certified_fraction = pair_certified_fraction
        if decisive_triangle != 2147483647:
            return 1, decisive_triangle, certified_fraction
        return 0, 2147483647, 1.0

    @wp.func(module=_module)
    def _triangle_swept_bounds(
        shape: int,
        index0: int,
        index1: int,
        index2: int,
        mesh_id: wp.uint64,
        shape_local_transform: wp.array[wp.transform],
        reference_shape_transform: wp.array[wp.transform],
        candidate_shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_body: wp.array[int],
        body_origin_path_length: wp.array[float],
        shape_effective_angular_path_length: wp.array[float],
        tolerance: float,
    ) -> tuple[bool, wp.vec3, wp.vec3]:
        """Return a conservative world AABB for one swept raw triangle."""
        mesh = wp.mesh_get(mesh_id)
        scale = shape_scale[shape]
        if not _valid_mesh_scale(scale):
            return False, wp.vec3(0.0), wp.vec3(0.0)
        local0 = wp.cw_mul(mesh.points[index0], scale)
        local1 = wp.cw_mul(mesh.points[index1], scale)
        local2 = wp.cw_mul(mesh.points[index2], scale)
        X_reference = reference_shape_transform[shape]
        X_candidate = candidate_shape_transform[shape]
        reference0 = wp.transform_point(X_reference, local0)
        reference1 = wp.transform_point(X_reference, local1)
        reference2 = wp.transform_point(X_reference, local2)
        candidate0 = wp.transform_point(X_candidate, local0)
        candidate1 = wp.transform_point(X_candidate, local1)
        candidate2 = wp.transform_point(X_candidate, local2)
        valid = (
            _finite_point(reference0)
            and _finite_point(reference1)
            and _finite_point(reference2)
            and _finite_point(candidate0)
            and _finite_point(candidate1)
            and _finite_point(candidate2)
        )
        if not valid:
            return False, wp.vec3(0.0), wp.vec3(0.0)

        reference_lower = wp.min(reference0, wp.min(reference1, reference2))
        reference_upper = wp.max(reference0, wp.max(reference1, reference2))
        candidate_lower = wp.min(candidate0, wp.min(candidate1, candidate2))
        candidate_upper = wp.max(candidate0, wp.max(candidate1, candidate2))
        path_displacement = 0.0
        body = shape_body[shape]
        if body >= 0:
            X_body_shape = shape_local_transform[shape]
            body_local0 = wp.transform_point(X_body_shape, local0)
            body_local1 = wp.transform_point(X_body_shape, local1)
            body_local2 = wp.transform_point(X_body_shape, local2)
            radius = wp.max(wp.length(body_local0), wp.max(wp.length(body_local1), wp.length(body_local2)))
            if not wp.isfinite(radius):
                return False, wp.vec3(0.0), wp.vec3(0.0)
            origin_path_length = body_origin_path_length[body]
            angular_valid, angular_displacement = _angular_path_displacement_bound(
                shape_effective_angular_path_length[shape], radius
            )
            if not wp.isfinite(origin_path_length) or origin_path_length < 0.0 or not angular_valid:
                return False, wp.vec3(0.0), wp.vec3(0.0)
            path_displacement = origin_path_length + angular_displacement
        lower = wp.min(candidate_lower, reference_lower - wp.vec3(path_displacement))
        upper = wp.max(candidate_upper, reference_upper + wp.vec3(path_displacement))
        return True, lower - wp.vec3(tolerance), upper + wp.vec3(tolerance)

    @wp.func(module=_module)
    def _shape_sweep_envelope_displacement_bound(
        shape: int,
        shape_motion_radius: wp.array[float],
        shape_body: wp.array[int],
        body_origin_path_length: wp.array[float],
        shape_effective_angular_path_length: wp.array[float],
    ) -> tuple[bool, float]:
        """Bound the unsupported-motion envelope from the reference geometry."""
        body = shape_body[shape]
        if body < 0:
            return True, 0.0
        origin_path_length = body_origin_path_length[body]
        angular_valid, angular_displacement = _angular_path_displacement_bound(
            shape_effective_angular_path_length[shape], shape_motion_radius[shape]
        )
        displacement = origin_path_length + angular_displacement
        valid = wp.isfinite(origin_path_length) and origin_path_length >= 0.0 and angular_valid
        return valid and wp.isfinite(displacement), displacement

    @wp.func(module=_module)
    def _triangle_projection_gap(
        axis: wp.vec3,
        triangle_a0: wp.vec3,
        triangle_a1: wp.vec3,
        triangle_a2: wp.vec3,
        triangle_b0: wp.vec3,
        triangle_b1: wp.vec3,
        triangle_b2: wp.vec3,
    ) -> tuple[bool, float]:
        """Return a conservative separation lower bound along one axis."""
        axis_length_sq = wp.length_sq(axis)
        if not wp.isfinite(axis_length_sq):
            return False, 0.0
        if axis_length_sq <= 1.0e-20:
            return True, 0.0
        normalized_axis = axis / wp.sqrt(axis_length_sq)
        projection_a0 = wp.dot(normalized_axis, triangle_a0)
        projection_a1 = wp.dot(normalized_axis, triangle_a1)
        projection_a2 = wp.dot(normalized_axis, triangle_a2)
        projection_b0 = wp.dot(normalized_axis, triangle_b0)
        projection_b1 = wp.dot(normalized_axis, triangle_b1)
        projection_b2 = wp.dot(normalized_axis, triangle_b2)
        lower_a = wp.min(projection_a0, wp.min(projection_a1, projection_a2))
        upper_a = wp.max(projection_a0, wp.max(projection_a1, projection_a2))
        lower_b = wp.min(projection_b0, wp.min(projection_b1, projection_b2))
        upper_b = wp.max(projection_b0, wp.max(projection_b1, projection_b2))
        gap = wp.max(0.0, wp.max(lower_b - upper_a, lower_a - upper_b))
        return wp.isfinite(gap), gap

    @wp.func(module=_module)
    def _triangle_pair_separation_lower_bound(
        triangle_a0: wp.vec3,
        triangle_a1: wp.vec3,
        triangle_a2: wp.vec3,
        triangle_b0: wp.vec3,
        triangle_b1: wp.vec3,
        triangle_b2: wp.vec3,
    ) -> tuple[bool, float]:
        """Bound triangle separation from below using finite projection gaps."""
        if (
            not _finite_point(triangle_a0)
            or not _finite_point(triangle_a1)
            or not _finite_point(triangle_a2)
            or not _finite_point(triangle_b0)
            or not _finite_point(triangle_b1)
            or not _finite_point(triangle_b2)
        ):
            return False, 0.0

        # Projection gaps are translation invariant. Centering avoids
        # cancellation when cloned worlds carry a large common offset.
        common_origin = triangle_b0
        triangle_a0 = triangle_a0 - common_origin
        triangle_a1 = triangle_a1 - common_origin
        triangle_a2 = triangle_a2 - common_origin
        triangle_b0 = triangle_b0 - common_origin
        triangle_b1 = triangle_b1 - common_origin
        triangle_b2 = triangle_b2 - common_origin

        edge_a0 = triangle_a1 - triangle_a0
        edge_a1 = triangle_a2 - triangle_a1
        edge_a2 = triangle_a0 - triangle_a2
        edge_b0 = triangle_b1 - triangle_b0
        edge_b1 = triangle_b2 - triangle_b1
        edge_b2 = triangle_b0 - triangle_b2
        lower_bound = float(0.0)
        normal_a = wp.cross(edge_a0, triangle_a2 - triangle_a0)
        normal_b = wp.cross(edge_b0, triangle_b2 - triangle_b0)
        for axis_index in range(5):
            candidate_axis = wp.vec3(1.0, 0.0, 0.0)
            if axis_index == 1:
                candidate_axis = wp.vec3(0.0, 1.0, 0.0)
            elif axis_index == 2:
                candidate_axis = wp.vec3(0.0, 0.0, 1.0)
            elif axis_index == 3:
                candidate_axis = normal_a
            elif axis_index == 4:
                candidate_axis = normal_b
            valid, gap = _triangle_projection_gap(
                candidate_axis, triangle_a0, triangle_a1, triangle_a2, triangle_b0, triangle_b1, triangle_b2
            )
            if not valid:
                return False, 0.0
            lower_bound = wp.max(lower_bound, gap)

        for edge_index in range(3):
            edge_a = edge_a0
            edge_b = edge_b0
            if edge_index == 1:
                edge_a = edge_a1
                edge_b = edge_b1
            elif edge_index == 2:
                edge_a = edge_a2
                edge_b = edge_b2
            for normal_edge_index in range(2):
                separating_axis = wp.cross(normal_a, edge_a)
                if normal_edge_index == 1:
                    separating_axis = wp.cross(normal_b, edge_b)
                valid, gap = _triangle_projection_gap(
                    separating_axis,
                    triangle_a0,
                    triangle_a1,
                    triangle_a2,
                    triangle_b0,
                    triangle_b1,
                    triangle_b2,
                )
                if not valid:
                    return False, 0.0
                lower_bound = wp.max(lower_bound, gap)

            for other_edge_index in range(3):
                other_edge = edge_b0
                if other_edge_index == 1:
                    other_edge = edge_b1
                elif other_edge_index == 2:
                    other_edge = edge_b2
                valid, gap = _triangle_projection_gap(
                    wp.cross(edge_a, other_edge),
                    triangle_a0,
                    triangle_a1,
                    triangle_a2,
                    triangle_b0,
                    triangle_b1,
                    triangle_b2,
                )
                if not valid:
                    return False, 0.0
                lower_bound = wp.max(lower_bound, gap)
        return True, lower_bound

    @wp.func(module=_module)
    def _quadratic_chord_clearance_lower_bound(
        start_clearance: float,
        end_clearance: float,
        curvature_coefficient: float,
    ) -> tuple[bool, float]:
        """Bound one oriented projection clearance over a constant-twist chord."""
        if (
            not wp.isfinite(start_clearance)
            or not wp.isfinite(end_clearance)
            or not wp.isfinite(curvature_coefficient)
            or curvature_coefficient < 0.0
        ):
            return False, 0.0
        lower_bound = wp.min(start_clearance, end_clearance)
        if curvature_coefficient > 0.0:
            delta = end_clearance - start_clearance
            minimum_time = wp.clamp(
                (curvature_coefficient - delta) / (2.0 * curvature_coefficient),
                0.0,
                1.0,
            )
            interior_bound = (
                (1.0 - minimum_time) * start_clearance
                + minimum_time * end_clearance
                - curvature_coefficient * minimum_time * (1.0 - minimum_time)
            )
            lower_bound = wp.min(lower_bound, interior_bound)
        return wp.isfinite(lower_bound), lower_bound

    @wp.func(module=_module)
    def _fixed_axis_separates_constant_twist(
        axis: wp.vec3,
        source_reference0: wp.vec3,
        source_reference1: wp.vec3,
        source_reference2: wp.vec3,
        source_candidate0: wp.vec3,
        source_candidate1: wp.vec3,
        source_candidate2: wp.vec3,
        target0: wp.vec3,
        target1: wp.vec3,
        target2: wp.vec3,
        margin_sum: float,
        tolerance: float,
        curvature_coefficient: float,
        rotation_axis: wp.vec3,
        rotation_axis_valid: bool,
    ) -> bool:
        """Return whether one fixed axis separates a complete constant-twist path."""
        axis_length_sq = wp.length_sq(axis)
        if not wp.isfinite(axis_length_sq) or axis_length_sq <= 1.0e-20:
            return False
        normalized_axis = axis / wp.sqrt(axis_length_sq)
        axis_curvature_coefficient = curvature_coefficient
        if rotation_axis_valid:
            # A rigid rotation has no projected curvature along its rotation
            # axis. Scaling the radius-angle bound by the separator's
            # orthogonal component remains conservative for every vertex.
            rotation_axis_projection = wp.clamp(wp.dot(normalized_axis, rotation_axis), -1.0, 1.0)
            axis_curvature_coefficient *= wp.sqrt(
                wp.max(0.0, 1.0 - rotation_axis_projection * rotation_axis_projection)
            )

        # Projection gaps are translation invariant. Centering keeps cloned
        # worlds with large common offsets numerically equivalent.
        source_reference0 -= target0
        source_reference1 -= target0
        source_reference2 -= target0
        source_candidate0 -= target0
        source_candidate1 -= target0
        source_candidate2 -= target0
        target1 -= target0
        target2 -= target0
        target0 = wp.vec3(0.0)

        source_reference_projection0 = wp.dot(normalized_axis, source_reference0)
        source_reference_projection1 = wp.dot(normalized_axis, source_reference1)
        source_reference_projection2 = wp.dot(normalized_axis, source_reference2)
        source_candidate_projection0 = wp.dot(normalized_axis, source_candidate0)
        source_candidate_projection1 = wp.dot(normalized_axis, source_candidate1)
        source_candidate_projection2 = wp.dot(normalized_axis, source_candidate2)
        target_projection0 = wp.dot(normalized_axis, target0)
        target_projection1 = wp.dot(normalized_axis, target1)
        target_projection2 = wp.dot(normalized_axis, target2)

        source_reference_lower = wp.min(
            source_reference_projection0,
            wp.min(source_reference_projection1, source_reference_projection2),
        )
        source_reference_upper = wp.max(
            source_reference_projection0,
            wp.max(source_reference_projection1, source_reference_projection2),
        )
        source_candidate_lower = wp.min(
            source_candidate_projection0,
            wp.min(source_candidate_projection1, source_candidate_projection2),
        )
        source_candidate_upper = wp.max(
            source_candidate_projection0,
            wp.max(source_candidate_projection1, source_candidate_projection2),
        )
        target_lower = wp.min(target_projection0, wp.min(target_projection1, target_projection2))
        target_upper = wp.max(target_projection0, wp.max(target_projection1, target_projection2))

        # Apply the same finite-precision tolerance used by endpoint validity.
        # Otherwise an accepted shallow contact cannot certify even tangential
        # motion from its reference state. Keeping one orientation across both
        # endpoints still rejects through-and-out paths and bounds any admitted
        # overlap by the configured tolerance.
        reference_before = target_lower - source_reference_upper - margin_sum + tolerance
        candidate_before = target_lower - source_candidate_upper - margin_sum + tolerance
        valid, path_clearance = _quadratic_chord_clearance_lower_bound(
            reference_before,
            candidate_before,
            axis_curvature_coefficient,
        )
        if valid and path_clearance >= 0.0:
            return True

        reference_after = source_reference_lower - target_upper - margin_sum + tolerance
        candidate_after = source_candidate_lower - target_upper - margin_sum + tolerance
        valid, path_clearance = _quadratic_chord_clearance_lower_bound(
            reference_after,
            candidate_after,
            axis_curvature_coefficient,
        )
        return valid and path_clearance >= 0.0

    @wp.func(module=_module)
    def _triangle_pair_separated_over_constant_twist(
        source_reference0: wp.vec3,
        source_reference1: wp.vec3,
        source_reference2: wp.vec3,
        source_candidate0: wp.vec3,
        source_candidate1: wp.vec3,
        source_candidate2: wp.vec3,
        target0: wp.vec3,
        target1: wp.vec3,
        target2: wp.vec3,
        margin_sum: float,
        tolerance: float,
        curvature_coefficient: float,
        rotation_axis: wp.vec3,
        rotation_axis_valid: bool,
    ) -> bool:
        """Prove a swept moving triangle remains on one side of a fixed axis."""
        if (
            not _finite_point(source_reference0)
            or not _finite_point(source_reference1)
            or not _finite_point(source_reference2)
            or not _finite_point(source_candidate0)
            or not _finite_point(source_candidate1)
            or not _finite_point(source_candidate2)
            or not _finite_point(target0)
            or not _finite_point(target1)
            or not _finite_point(target2)
            or not wp.isfinite(margin_sum)
            or not wp.isfinite(tolerance)
            or not wp.isfinite(curvature_coefficient)
            or margin_sum < 0.0
            or tolerance < 0.0
            or curvature_coefficient < 0.0
        ):
            return False

        source_reference_edge0 = source_reference1 - source_reference0
        source_reference_edge1 = source_reference2 - source_reference1
        source_reference_edge2 = source_reference0 - source_reference2
        source_candidate_edge0 = source_candidate1 - source_candidate0
        source_candidate_edge1 = source_candidate2 - source_candidate1
        source_candidate_edge2 = source_candidate0 - source_candidate2
        target_edge0 = target1 - target0
        target_edge1 = target2 - target1
        target_edge2 = target0 - target2
        source_reference_normal = wp.cross(source_reference_edge0, source_reference2 - source_reference0)
        source_candidate_normal = wp.cross(source_candidate_edge0, source_candidate2 - source_candidate0)
        target_normal = wp.cross(target_edge0, target2 - target0)

        for axis_index in range(6):
            axis = wp.vec3(1.0, 0.0, 0.0)
            if axis_index == 1:
                axis = wp.vec3(0.0, 1.0, 0.0)
            elif axis_index == 2:
                axis = wp.vec3(0.0, 0.0, 1.0)
            elif axis_index == 3:
                axis = source_reference_normal
            elif axis_index == 4:
                axis = source_candidate_normal
            elif axis_index == 5:
                axis = target_normal
            if _fixed_axis_separates_constant_twist(
                axis,
                source_reference0,
                source_reference1,
                source_reference2,
                source_candidate0,
                source_candidate1,
                source_candidate2,
                target0,
                target1,
                target2,
                margin_sum,
                tolerance,
                curvature_coefficient,
                rotation_axis,
                rotation_axis_valid,
            ):
                return True

        for edge_index in range(3):
            source_reference_edge = source_reference_edge0
            source_candidate_edge = source_candidate_edge0
            target_edge = target_edge0
            if edge_index == 1:
                source_reference_edge = source_reference_edge1
                source_candidate_edge = source_candidate_edge1
                target_edge = target_edge1
            elif edge_index == 2:
                source_reference_edge = source_reference_edge2
                source_candidate_edge = source_candidate_edge2
                target_edge = target_edge2

            for axis_kind in range(3):
                axis = wp.cross(source_reference_normal, source_reference_edge)
                if axis_kind == 1:
                    axis = wp.cross(source_candidate_normal, source_candidate_edge)
                elif axis_kind == 2:
                    axis = wp.cross(target_normal, target_edge)
                if _fixed_axis_separates_constant_twist(
                    axis,
                    source_reference0,
                    source_reference1,
                    source_reference2,
                    source_candidate0,
                    source_candidate1,
                    source_candidate2,
                    target0,
                    target1,
                    target2,
                    margin_sum,
                    tolerance,
                    curvature_coefficient,
                    rotation_axis,
                    rotation_axis_valid,
                ):
                    return True

            for target_edge_index in range(3):
                other_target_edge = target_edge0
                if target_edge_index == 1:
                    other_target_edge = target_edge1
                elif target_edge_index == 2:
                    other_target_edge = target_edge2
                for source_pose in range(2):
                    source_edge = source_reference_edge
                    if source_pose == 1:
                        source_edge = source_candidate_edge
                    if _fixed_axis_separates_constant_twist(
                        wp.cross(source_edge, other_target_edge),
                        source_reference0,
                        source_reference1,
                        source_reference2,
                        source_candidate0,
                        source_candidate1,
                        source_candidate2,
                        target0,
                        target1,
                        target2,
                        margin_sum,
                        tolerance,
                        curvature_coefficient,
                        rotation_axis,
                        rotation_axis_valid,
                    ):
                        return True
        return False

    @wp.func(module=_module)
    def _triangle_pair_constant_twist_certified_fraction(
        source_body_local0: wp.vec3,
        source_body_local1: wp.vec3,
        source_body_local2: wp.vec3,
        source_body_reference: wp.transform,
        source_body_candidate: wp.transform,
        target0: wp.vec3,
        target1: wp.vec3,
        target2: wp.vec3,
        margin_sum: float,
        tolerance: float,
        motion_radius: float,
        effective_angular_path_length: float,
    ) -> float:
        """Return the longest certified-safe dyadic prefix for one triangle pair."""
        rotations_valid, reference_rotation, candidate_rotation = _normalize_shortest_quaternion_pair(
            wp.transform_get_rotation(source_body_reference),
            wp.transform_get_rotation(source_body_candidate),
        )
        reference_translation = wp.transform_get_translation(source_body_reference)
        candidate_translation = wp.transform_get_translation(source_body_candidate)
        if (
            not rotations_valid
            or not _finite_point(source_body_local0)
            or not _finite_point(source_body_local1)
            or not _finite_point(source_body_local2)
            or not _finite_point(reference_translation)
            or not _finite_point(candidate_translation)
            or not wp.isfinite(motion_radius)
            or not wp.isfinite(effective_angular_path_length)
            or motion_radius < 0.0
            or effective_angular_path_length < 0.0
        ):
            return 0.0

        translation_delta = candidate_translation - reference_translation
        relative_rotation = candidate_rotation * wp.quat_inverse(reference_rotation)
        rotation_axis = wp.vec3(relative_rotation[0], relative_rotation[1], relative_rotation[2])
        rotation_axis_length_sq = wp.length_sq(rotation_axis)
        rotation_axis_valid = wp.isfinite(rotation_axis_length_sq) and rotation_axis_length_sq > 1.0e-20
        if rotation_axis_valid:
            rotation_axis *= _sdf_rsqrt_rn(rotation_axis_length_sq)
        certified_fraction = float(0.0)
        interval_count = int(2)
        for _subdivision_level in range(_CONSTANT_TWIST_SUBDIVISION_LEVELS):
            interval_angle = effective_angular_path_length / float(interval_count) + (_CONSTANT_TWIST_ANGLE_TOLERANCE)
            interval_curvature_coefficient = 0.5 * motion_radius * interval_angle * interval_angle
            all_intervals_separated = bool(True)
            for interval_index in range(interval_count):
                interval_start = float(interval_index) / float(interval_count)
                interval_end = float(interval_index + 1) / float(interval_count)
                start_translation = reference_translation + interval_start * translation_delta
                end_translation = reference_translation + interval_end * translation_delta
                start_rotation = wp.quat_slerp(reference_rotation, candidate_rotation, interval_start)
                end_rotation = wp.quat_slerp(reference_rotation, candidate_rotation, interval_end)
                source_start0 = start_translation + wp.quat_rotate(start_rotation, source_body_local0)
                source_start1 = start_translation + wp.quat_rotate(start_rotation, source_body_local1)
                source_start2 = start_translation + wp.quat_rotate(start_rotation, source_body_local2)
                source_end0 = end_translation + wp.quat_rotate(end_rotation, source_body_local0)
                source_end1 = end_translation + wp.quat_rotate(end_rotation, source_body_local1)
                source_end2 = end_translation + wp.quat_rotate(end_rotation, source_body_local2)
                interval_separated = _triangle_pair_separated_over_constant_twist(
                    source_start0,
                    source_start1,
                    source_start2,
                    source_end0,
                    source_end1,
                    source_end2,
                    target0,
                    target1,
                    target2,
                    margin_sum,
                    tolerance,
                    interval_curvature_coefficient,
                    rotation_axis,
                    rotation_axis_valid,
                )
                if not interval_separated:
                    all_intervals_separated = False
            if all_intervals_separated:
                return 1.0

            # A fallback prefix remains governed by the oracle's acceptance
            # tolerance. Requiring zero overlap here prevents any progress
            # from a valid reference that already lies inside that tolerance.
            # Return only complete dyadic intervals whose conservative sweep
            # bound satisfies that same tolerance.
            prefix_separated = bool(True)
            level_certified_fraction = float(0.0)
            for interval_index in range(interval_count):
                interval_start = float(interval_index) / float(interval_count)
                interval_end = float(interval_index + 1) / float(interval_count)
                start_translation = reference_translation + interval_start * translation_delta
                end_translation = reference_translation + interval_end * translation_delta
                start_rotation = wp.quat_slerp(reference_rotation, candidate_rotation, interval_start)
                end_rotation = wp.quat_slerp(reference_rotation, candidate_rotation, interval_end)
                source_start0 = start_translation + wp.quat_rotate(start_rotation, source_body_local0)
                source_start1 = start_translation + wp.quat_rotate(start_rotation, source_body_local1)
                source_start2 = start_translation + wp.quat_rotate(start_rotation, source_body_local2)
                source_end0 = end_translation + wp.quat_rotate(end_rotation, source_body_local0)
                source_end1 = end_translation + wp.quat_rotate(end_rotation, source_body_local1)
                source_end2 = end_translation + wp.quat_rotate(end_rotation, source_body_local2)
                interval_prefix_separated = _triangle_pair_separated_over_constant_twist(
                    source_start0,
                    source_start1,
                    source_start2,
                    source_end0,
                    source_end1,
                    source_end2,
                    target0,
                    target1,
                    target2,
                    margin_sum,
                    tolerance,
                    interval_curvature_coefficient,
                    rotation_axis,
                    rotation_axis_valid,
                )
                if not interval_prefix_separated:
                    prefix_separated = False
                elif prefix_separated:
                    level_certified_fraction = interval_end
            certified_fraction = wp.max(certified_fraction, level_certified_fraction)
            interval_count *= 2
        return certified_fraction

    @wp.func(module=_module)
    def _world_aabb_in_raw_mesh_frame(
        lower: wp.vec3,
        upper: wp.vec3,
        X_mesh_reference: wp.transform,
        mesh_scale_inverse: wp.vec3,
        inflation: float,
    ) -> tuple[bool, wp.vec3, wp.vec3]:
        """Map an inflated world AABB conservatively into raw mesh coordinates."""
        if not _finite_point(lower) or not _finite_point(upper) or not wp.isfinite(inflation) or inflation < 0.0:
            return False, wp.vec3(0.0), wp.vec3(0.0)
        center = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower) + wp.vec3(inflation)
        if not _finite_point(center) or not _finite_point(half) or wp.min(half) < 0.0:
            return False, wp.vec3(0.0), wp.vec3(0.0)
        X_world_mesh = wp.transform_inverse(X_mesh_reference)
        mesh_center_scaled = wp.transform_point(X_world_mesh, center)
        axis0 = wp.transform_vector(X_world_mesh, wp.vec3(half[0], 0.0, 0.0))
        axis1 = wp.transform_vector(X_world_mesh, wp.vec3(0.0, half[1], 0.0))
        axis2 = wp.transform_vector(X_world_mesh, wp.vec3(0.0, 0.0, half[2]))
        mesh_half_scaled = wp.abs(axis0) + wp.abs(axis1) + wp.abs(axis2)
        mesh_center = wp.cw_mul(mesh_center_scaled, mesh_scale_inverse)
        mesh_half = wp.cw_mul(mesh_half_scaled, wp.abs(mesh_scale_inverse))
        query_lower = mesh_center - mesh_half
        query_upper = mesh_center + mesh_half
        valid = _finite_point(query_lower) and _finite_point(query_upper)
        return valid, query_lower, query_upper

    @wp.func(module=_module)
    def _query_bounded_sweep_ambiguity(
        shape_a: int,
        shape_b: int,
        stationary_a: bool,
        stationary_b: bool,
        source_triangle_start: int,
        source_triangle_stride: int,
        reference_body_q: wp.array[wp.transform],
        candidate_body_q: wp.array[wp.transform],
        shape_local_transform: wp.array[wp.transform],
        reference_shape_transform: wp.array[wp.transform],
        candidate_shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_margin: wp.array[float],
        shape_source: wp.array[wp.uint64],
        shape_motion_radius: wp.array[float],
        shape_body: wp.array[int],
        body_origin_path_length: wp.array[float],
        shape_effective_angular_path_length: wp.array[float],
        constant_twist: bool,
        tolerance: float,
    ) -> tuple[int, int, float]:
        """Certify bounded motion and return its conservative constant-twist prefix."""
        mesh_a_id = shape_source[shape_a]
        mesh_b_id = shape_source[shape_b]
        if mesh_a_id == wp.uint64(0) or mesh_b_id == wp.uint64(0):
            return -1, -1, 0.0
        mesh_a = wp.mesh_get(mesh_a_id)
        mesh_b = wp.mesh_get(mesh_b_id)
        index_count_a = mesh_a.indices.shape[0]
        index_count_b = mesh_b.indices.shape[0]
        if index_count_a < 3 or index_count_a % 3 != 0 or index_count_b < 3 or index_count_b % 3 != 0:
            return -1, -1, 0.0

        source_shape = shape_a
        target_shape = shape_b
        source_mesh_id = mesh_a_id
        target_mesh_id = mesh_b_id
        source_mesh = mesh_a
        target_mesh = mesh_b
        source_index_count = index_count_a
        swap_roles = (stationary_a and not stationary_b) or (
            stationary_a == stationary_b and index_count_b < index_count_a
        )
        if swap_roles:
            source_shape = shape_b
            target_shape = shape_a
            source_mesh_id = mesh_b_id
            target_mesh_id = mesh_a_id
            source_mesh = mesh_b
            target_mesh = mesh_a
            source_index_count = index_count_b

        source_scale = shape_scale[source_shape]
        target_scale = shape_scale[target_shape]
        source_margin = shape_margin[source_shape]
        target_margin = shape_margin[target_shape]
        margin_sum = source_margin + target_margin
        if (
            not _valid_mesh_scale(source_scale)
            or not _valid_mesh_scale(target_scale)
            or not wp.isfinite(source_margin)
            or not wp.isfinite(target_margin)
            or source_margin < 0.0
            or target_margin < 0.0
        ):
            return -1, -1, 0.0
        target_scale_inverse, _target_min_scale = safe_sdf_scale_inverse(target_scale)
        source_motion_valid, source_motion = _shape_sweep_envelope_displacement_bound(
            source_shape,
            shape_motion_radius,
            shape_body,
            body_origin_path_length,
            shape_effective_angular_path_length,
        )
        target_motion_valid, target_motion = _shape_sweep_envelope_displacement_bound(
            target_shape,
            shape_motion_radius,
            shape_body,
            body_origin_path_length,
            shape_effective_angular_path_length,
        )
        relative_displacement = source_motion + target_motion + margin_sum + tolerance
        if not source_motion_valid or not target_motion_valid or not wp.isfinite(relative_displacement):
            return -1, -1, 0.0
        X_source_reference = reference_shape_transform[source_shape]
        X_source_candidate = candidate_shape_transform[source_shape]
        X_target_reference = reference_shape_transform[target_shape]
        source_body = shape_body[source_shape]
        curvature_coefficient = 0.0
        if constant_twist:
            if source_body < 0:
                return -1, -1, 0.0
            effective_angular_path_length = shape_effective_angular_path_length[source_shape]
            curvature_coefficient = (
                0.5 * shape_motion_radius[source_shape] * effective_angular_path_length * effective_angular_path_length
            )
            if not wp.isfinite(curvature_coefficient) or curvature_coefficient < 0.0:
                return -1, -1, 0.0

        sweep_status = int(0)
        decisive_triangle = int(2147483647)
        certified_fraction = float(1.0)

        for source_triangle in range(source_triangle_start, source_index_count // 3, source_triangle_stride):
            source_index0 = source_mesh.indices[3 * source_triangle]
            source_index1 = source_mesh.indices[3 * source_triangle + 1]
            source_index2 = source_mesh.indices[3 * source_triangle + 2]
            if (
                source_index0 < 0
                or source_index0 >= source_mesh.points.shape[0]
                or source_index1 < 0
                or source_index1 >= source_mesh.points.shape[0]
                or source_index2 < 0
                or source_index2 >= source_mesh.points.shape[0]
            ):
                return -1, source_triangle, 0.0
            source_valid, source_lower, source_upper = _triangle_swept_bounds(
                source_shape,
                source_index0,
                source_index1,
                source_index2,
                source_mesh_id,
                shape_local_transform,
                reference_shape_transform,
                candidate_shape_transform,
                shape_scale,
                shape_body,
                body_origin_path_length,
                shape_effective_angular_path_length,
                tolerance,
            )
            if not source_valid:
                return -1, source_triangle, 0.0
            source_reference0 = wp.transform_point(
                X_source_reference, wp.cw_mul(source_mesh.points[source_index0], source_scale)
            )
            source_reference1 = wp.transform_point(
                X_source_reference, wp.cw_mul(source_mesh.points[source_index1], source_scale)
            )
            source_reference2 = wp.transform_point(
                X_source_reference, wp.cw_mul(source_mesh.points[source_index2], source_scale)
            )
            source_candidate0 = wp.transform_point(
                X_source_candidate, wp.cw_mul(source_mesh.points[source_index0], source_scale)
            )
            source_candidate1 = wp.transform_point(
                X_source_candidate, wp.cw_mul(source_mesh.points[source_index1], source_scale)
            )
            source_candidate2 = wp.transform_point(
                X_source_candidate, wp.cw_mul(source_mesh.points[source_index2], source_scale)
            )
            query_valid, query_lower, query_upper = _world_aabb_in_raw_mesh_frame(
                source_lower,
                source_upper,
                X_target_reference,
                target_scale_inverse,
                target_motion + margin_sum + tolerance,
            )
            if not query_valid:
                return -1, source_triangle, 0.0
            query = wp.mesh_query_aabb(target_mesh_id, query_lower, query_upper)
            target_triangle = wp.int32(0)
            while wp.mesh_query_aabb_next(query, target_triangle):
                target_index0 = target_mesh.indices[3 * target_triangle]
                target_index1 = target_mesh.indices[3 * target_triangle + 1]
                target_index2 = target_mesh.indices[3 * target_triangle + 2]
                if (
                    target_index0 < 0
                    or target_index0 >= target_mesh.points.shape[0]
                    or target_index1 < 0
                    or target_index1 >= target_mesh.points.shape[0]
                    or target_index2 < 0
                    or target_index2 >= target_mesh.points.shape[0]
                ):
                    return -1, source_triangle, 0.0
                target_reference0 = wp.transform_point(
                    X_target_reference, wp.cw_mul(target_mesh.points[target_index0], target_scale)
                )
                target_reference1 = wp.transform_point(
                    X_target_reference, wp.cw_mul(target_mesh.points[target_index1], target_scale)
                )
                target_reference2 = wp.transform_point(
                    X_target_reference, wp.cw_mul(target_mesh.points[target_index2], target_scale)
                )
                if constant_twist:
                    if _triangle_pair_separated_over_constant_twist(
                        source_reference0,
                        source_reference1,
                        source_reference2,
                        source_candidate0,
                        source_candidate1,
                        source_candidate2,
                        target_reference0,
                        target_reference1,
                        target_reference2,
                        margin_sum,
                        tolerance,
                        curvature_coefficient,
                        wp.vec3(0.0),
                        False,
                    ):
                        continue
                    X_body_shape = shape_local_transform[source_shape]
                    source_body_local0 = wp.transform_point(
                        X_body_shape, wp.cw_mul(source_mesh.points[source_index0], source_scale)
                    )
                    source_body_local1 = wp.transform_point(
                        X_body_shape, wp.cw_mul(source_mesh.points[source_index1], source_scale)
                    )
                    source_body_local2 = wp.transform_point(
                        X_body_shape, wp.cw_mul(source_mesh.points[source_index2], source_scale)
                    )
                    pair_certified_fraction = _triangle_pair_constant_twist_certified_fraction(
                        source_body_local0,
                        source_body_local1,
                        source_body_local2,
                        reference_body_q[source_body],
                        candidate_body_q[source_body],
                        target_reference0,
                        target_reference1,
                        target_reference2,
                        margin_sum,
                        tolerance,
                        shape_motion_radius[source_shape],
                        effective_angular_path_length,
                    )
                    if pair_certified_fraction >= 1.0:
                        continue
                    sweep_status = 1
                    decisive_triangle = wp.min(decisive_triangle, source_triangle)
                    certified_fraction = wp.min(certified_fraction, pair_certified_fraction)
                    continue
                bound_valid, separation_lower_bound = _triangle_pair_separation_lower_bound(
                    source_reference0,
                    source_reference1,
                    source_reference2,
                    target_reference0,
                    target_reference1,
                    target_reference2,
                )
                if not bound_valid:
                    return -1, source_triangle, 0.0
                if separation_lower_bound <= relative_displacement:
                    return 1, source_triangle, 0.0
        return sweep_status, decisive_triangle, certified_fraction

    @wp.func(module=_module)
    def _query_reference_connected_guard(
        source_shape: int,
        target_shape: int,
        source_point_local: wp.vec3,
        separation_tolerance: float,
        margin_sum: float,
        reference_shape_transform: wp.array[wp.transform],
        candidate_shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_source: wp.array[wp.uint64],
        shape_mesh_properties: wp.array[wp.int32],
    ) -> tuple[int, float, wp.vec3, wp.vec3, wp.vec3]:
        """Connect one material sample to its nearest clear-reference target plane."""
        empty_point = wp.vec3(0.0)
        reference_valid, reference_distance, reference_direction, reference_source_point = _query_endpoint(
            source_shape,
            target_shape,
            source_point_local,
            reference_shape_transform,
            shape_scale,
            shape_source,
            shape_mesh_properties,
        )
        if not reference_valid:
            return -1, 0.0, empty_point, empty_point, empty_point
        if reference_distance - margin_sum < -separation_tolerance:
            return 0, 0.0, empty_point, empty_point, empty_point

        X_source_candidate = candidate_shape_transform[source_shape]
        X_target_reference = reference_shape_transform[target_shape]
        X_target_candidate = candidate_shape_transform[target_shape]
        reference_target_point = reference_source_point - reference_distance * reference_direction
        target_point_local = wp.transform_point(wp.transform_inverse(X_target_reference), reference_target_point)
        normal_target_local = wp.transform_vector(wp.transform_inverse(X_target_reference), reference_direction)
        candidate_source_point = wp.transform_point(X_source_candidate, source_point_local)
        candidate_target_point = wp.transform_point(X_target_candidate, target_point_local)
        candidate_normal = wp.transform_vector(X_target_candidate, normal_target_local)
        candidate_normal_length_sq = wp.length_sq(candidate_normal)
        if (
            not _finite_point(candidate_source_point)
            or not _finite_point(candidate_target_point)
            or not _finite_point(candidate_normal)
            or not wp.isfinite(candidate_normal_length_sq)
            or candidate_normal_length_sq <= 1.0e-20
        ):
            return -1, 0.0, empty_point, empty_point, empty_point
        candidate_normal *= _sdf_rsqrt_rn(candidate_normal_length_sq)
        candidate_separation = wp.dot(candidate_source_point - candidate_target_point, candidate_normal) - margin_sum
        if not wp.isfinite(candidate_separation):
            return -1, 0.0, empty_point, empty_point, empty_point
        if candidate_separation < -separation_tolerance:
            return 1, candidate_separation, candidate_source_point, candidate_target_point, candidate_normal
        return 0, 0.0, empty_point, empty_point, empty_point

    @wp.func(module=_module)
    def _query_transverse_edge_hit(
        source_shape: int,
        target_shape: int,
        source_point0_local: wp.vec3,
        source_point1_local: wp.vec3,
        separation_tolerance: float,
        margin_sum: float,
        reference_shape_transform: wp.array[wp.transform],
        candidate_shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_source: wp.array[wp.uint64],
        shape_mesh_properties: wp.array[wp.int32],
    ) -> tuple[int, float, wp.vec3, wp.vec3, wp.vec3]:
        """Return a transverse hit and its deepest reference-connected guard.

        The integer result is ``2`` for a hit with a guard, ``1`` for a hit
        without a sound reference-connected guard, ``0`` for a clean segment,
        and ``-1`` when finite bounded traversal cannot classify the segment.
        The remaining values describe the most violated candidate endpoint,
        its target point, and the target-to-source normal connected to the
        clear reference pose. Tangential hits are skipped so resting surface
        contact is not reported as volume penetration. A bounded resume loop
        handles endpoint or tangent hits that precede a transverse crossing
        and fails closed if it exhausts that bound.
        """
        empty_point = wp.vec3(0.0)
        target_mesh_id = shape_source[target_shape]
        source_scale = shape_scale[source_shape]
        target_scale = shape_scale[target_shape]
        if target_mesh_id == wp.uint64(0) or not _valid_mesh_scale(source_scale) or not _valid_mesh_scale(target_scale):
            return -1, 0.0, empty_point, empty_point, empty_point
        target_mesh = wp.mesh_get(target_mesh_id)

        X_source_ws = candidate_shape_transform[source_shape]
        X_target_ws = candidate_shape_transform[target_shape]
        point0_world = wp.transform_point(X_source_ws, source_point0_local)
        point1_world = wp.transform_point(X_source_ws, source_point1_local)
        edge_world = point1_world - point0_world
        edge_length_sq = wp.length_sq(edge_world)
        valid_edge = (
            wp.isfinite(point0_world[0])
            and wp.isfinite(point0_world[1])
            and wp.isfinite(point0_world[2])
            and wp.isfinite(point1_world[0])
            and wp.isfinite(point1_world[1])
            and wp.isfinite(point1_world[2])
            and wp.isfinite(edge_length_sq)
        )
        if not valid_edge:
            return -1, 0.0, empty_point, empty_point, empty_point
        if edge_length_sq <= 1.0e-20:
            return 0, 0.0, empty_point, empty_point, empty_point

        edge_length_inv = _sdf_rsqrt_rn(edge_length_sq)
        edge_direction_world = edge_world * edge_length_inv
        ray_origin, ray_direction = map_ray_to_local(X_target_ws, point0_world, edge_world, target_scale)
        valid_ray = (
            wp.isfinite(ray_origin[0])
            and wp.isfinite(ray_origin[1])
            and wp.isfinite(ray_origin[2])
            and wp.isfinite(ray_direction[0])
            and wp.isfinite(ray_direction[1])
            and wp.isfinite(ray_direction[2])
            and wp.length_sq(ray_direction) > 1.0e-20
        )
        if not valid_ray:
            return -1, 0.0, empty_point, empty_point, empty_point

        # ``ray_direction`` spans the complete source segment, hence ``t`` is
        # the affine segment parameter even under nonuniform target scaling.
        param_advance = wp.clamp(4.0 * separation_tolerance * edge_length_inv, 1.0e-6, 1.0e-3)
        cursor = float(0.0)
        saturated = bool(False)
        crossing_found = bool(False)
        best_separation = float(1.0e30)
        best_source_point = empty_point
        best_target_point = empty_point
        best_normal = empty_point
        # Advancing beyond every reported hit prevents revisiting the same triangle. A segment therefore needs at
        # most one hit per target triangle; the extra query proves that no surface remains instead of relying on an
        # arbitrary traversal cap for layered or threaded meshes.
        max_hit_count = target_mesh.indices.shape[0] // 3 + 1
        for attempt in range(max_hit_count):
            remaining = 1.0 - cursor
            if remaining <= param_advance:
                break
            hit_t, hit_normal_target, _u, _v, _face = ray_intersect_mesh(
                ray_origin + cursor * ray_direction,
                ray_direction,
                target_scale,
                target_mesh_id,
                False,
                remaining,
            )
            if hit_t < 0.0:
                break
            if not wp.isfinite(hit_t):
                return -1, 0.0, empty_point, empty_point, empty_point

            segment_t = cursor + hit_t
            hit_normal_world = wp.transform_vector(X_target_ws, hit_normal_target)
            normal_length_sq = wp.length_sq(hit_normal_world)
            valid_normal = (
                wp.isfinite(hit_normal_world[0])
                and wp.isfinite(hit_normal_world[1])
                and wp.isfinite(hit_normal_world[2])
                and wp.isfinite(normal_length_sq)
                and normal_length_sq > 1.0e-20
            )
            if not valid_normal:
                return -1, 0.0, empty_point, empty_point, empty_point
            hit_normal_world *= _sdf_rsqrt_rn(normal_length_sq)

            transverse = wp.abs(wp.dot(edge_direction_world, hit_normal_world)) > 1.0e-5
            interior = segment_t > param_advance and segment_t < 1.0 - param_advance
            if interior and transverse:
                # Triangle rays alone also report shared-surface and edge
                # coincidences. Confirm that one adjacent material sample is
                # actually inside the target before treating this as a volume
                # crossing. Connect the same affine material sample to its
                # clear-reference plane so its shallow candidate SDF depth
                # does not limit the guard displacement.
                before_t = wp.max(param_advance, segment_t - param_advance)
                after_t = wp.min(1.0 - param_advance, segment_t + param_advance)
                before_local = source_point0_local + before_t * (source_point1_local - source_point0_local)
                after_local = source_point0_local + after_t * (source_point1_local - source_point0_local)
                for sample in range(2):
                    sample_local = before_local if sample == 0 else after_local
                    candidate_valid, candidate_distance, candidate_direction, candidate_source_point = _query_endpoint(
                        source_shape,
                        target_shape,
                        sample_local,
                        candidate_shape_transform,
                        shape_scale,
                        shape_source,
                        shape_mesh_properties,
                    )
                    if not candidate_valid:
                        return -1, 0.0, empty_point, empty_point, empty_point
                    if candidate_distance - margin_sum < -separation_tolerance:
                        crossing_found = True
                        (
                            guard_status,
                            guard_separation,
                            guard_source_point,
                            guard_target_point,
                            guard_normal,
                        ) = _query_reference_connected_guard(
                            source_shape,
                            target_shape,
                            sample_local,
                            separation_tolerance,
                            margin_sum,
                            reference_shape_transform,
                            candidate_shape_transform,
                            shape_scale,
                            shape_source,
                            shape_mesh_properties,
                        )
                        if guard_status < 0:
                            return -1, 0.0, empty_point, empty_point, empty_point
                        if guard_status > 0 and guard_separation < best_separation:
                            best_separation = guard_separation
                            best_source_point = guard_source_point
                            best_target_point = guard_target_point
                            best_normal = guard_normal
                        elif guard_status == 0:
                            # A rotating nearest feature need not stay connected to the reference nearest plane.
                            # Retain its exact candidate material pair as a correction row; the final raw-mesh scan,
                            # not this local linearization, remains the acceptance gate.
                            candidate_separation = candidate_distance - margin_sum
                            if candidate_separation < best_separation:
                                best_separation = candidate_separation
                                best_source_point = candidate_source_point
                                best_target_point = candidate_source_point - candidate_distance * candidate_direction
                                best_normal = candidate_direction

            next_cursor = segment_t + param_advance
            if not wp.isfinite(next_cursor) or next_cursor <= cursor:
                return -1, 0.0, empty_point, empty_point, empty_point
            cursor = next_cursor
            if attempt == max_hit_count - 1 and cursor < 1.0 - param_advance:
                saturated = True

        if saturated:
            return -1, 0.0, empty_point, empty_point, empty_point
        if not crossing_found:
            return 0, 0.0, empty_point, empty_point, empty_point

        # Also connect each authored endpoint to its nearest plane at the clear
        # reference pose. Interior and endpoint guards use the same transported
        # plane construction and the deepest one owns this edge identity.
        for endpoint in range(2):
            endpoint_local = source_point0_local if endpoint == 0 else source_point1_local
            (
                guard_status,
                guard_separation,
                guard_source_point,
                guard_target_point,
                guard_normal,
            ) = _query_reference_connected_guard(
                source_shape,
                target_shape,
                endpoint_local,
                separation_tolerance,
                margin_sum,
                reference_shape_transform,
                candidate_shape_transform,
                shape_scale,
                shape_source,
                shape_mesh_properties,
            )
            if guard_status < 0:
                return -1, 0.0, empty_point, empty_point, empty_point
            if guard_status > 0 and guard_separation < best_separation:
                best_separation = guard_separation
                best_source_point = guard_source_point
                best_target_point = guard_target_point
                best_normal = guard_normal

        if best_separation < float(1.0e30):
            return 2, best_separation, best_source_point, best_target_point, best_normal
        return 1, 0.0, empty_point, empty_point, empty_point

    @wp.func(module=_module)
    def _point_in_body_frame(
        point_world: wp.vec3,
        body: int,
        body_q: wp.array[wp.transform],
    ) -> wp.vec3:
        if body < 0:
            return point_world
        return wp.transform_point(wp.transform_inverse(body_q[body]), point_world)

    @wp.func(module=_module)
    def _vector_in_body_frame(
        vector_world: wp.vec3,
        body: int,
        body_q: wp.array[wp.transform],
    ) -> wp.vec3:
        if body < 0:
            return vector_world
        return wp.transform_vector(wp.transform_inverse(body_q[body]), vector_world)

    @wp.kernel(enable_backward=False, module=_module)
    def mesh_sdf_nonpenetration_sweep_oracle_kernel(
        reference_body_q: wp.array[wp.transform],
        candidate_body_q: wp.array[wp.transform],
        shape_local_transform: wp.array[wp.transform],
        reference_shape_transform: wp.array[wp.transform],
        candidate_shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_margin: wp.array[float],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_motion_radius: wp.array[float],
        body_origin_path_length: wp.array[float],
        shape_effective_angular_path_length: wp.array[float],
        body_path_kind: wp.array[wp.uint8],
        stationary_path_kind: int,
        linear_translation_path_kind: int,
        bounded_path_kind: int,
        constant_twist_path_kind: int,
        shape_source: wp.array[wp.uint64],
        shape_body: wp.array[int],
        shape_world: wp.array[int],
        shape_pairs: wp.array[wp.vec2i],
        shape_pair_count: wp.array[int],
        reference_pair_state: wp.array[wp.uint8],
        separation_tolerance: float,
        violation_bit: int,
        incomplete_bit: int,
        sweep_incomplete_bit: int,
        world_status: wp.array[wp.int32],
        certified_path_fraction: wp.array[float],
        query_incomplete: wp.array[int],
        total_num_blocks: int,
    ):
        block_idx, lane = wp.tid()
        pair_count = wp.min(shape_pair_count[0], shape_pairs.shape[0])
        for pair_idx in range(block_idx, pair_count, total_num_blocks):
            if reference_pair_state[pair_idx] != wp.uint8(0):
                continue
            pair = shape_pairs[pair_idx]
            shape_a = pair[0]
            shape_b = pair[1]
            world_a = shape_world[shape_a]
            world_b = shape_world[shape_b]
            world_id = world_a if world_a >= 0 else world_b
            if world_id < 0 and world_status.shape[0] == 1:
                world_id = 0
            if (
                world_id < 0
                or world_id >= world_status.shape[0]
                or (world_a >= 0 and world_b >= 0 and world_a != world_b)
            ):
                if lane == 0:
                    wp.atomic_or(query_incomplete, 0, incomplete_bit)
                continue

            body_a = shape_body[shape_a]
            body_b = shape_body[shape_b]
            path_kind_a = stationary_path_kind if body_a < 0 else int(body_path_kind[body_a])
            path_kind_b = stationary_path_kind if body_b < 0 else int(body_path_kind[body_b])
            valid_path_a = (
                path_kind_a == stationary_path_kind
                or path_kind_a == linear_translation_path_kind
                or path_kind_a == bounded_path_kind
                or path_kind_a == constant_twist_path_kind
            )
            valid_path_b = (
                path_kind_b == stationary_path_kind
                or path_kind_b == linear_translation_path_kind
                or path_kind_b == bounded_path_kind
                or path_kind_b == constant_twist_path_kind
            )
            if not valid_path_a or not valid_path_b:
                if lane == 0:
                    wp.atomic_or(world_status, world_id, sweep_incomplete_bit)
                    wp.atomic_min(certified_path_fraction, world_id, 0.0)
                continue

            stationary_a = path_kind_a == stationary_path_kind
            stationary_b = path_kind_b == stationary_path_kind
            if stationary_a and stationary_b:
                continue

            local_sweep_status = int(-1)
            local_decisive_triangle = int(-1)
            local_certified_fraction = float(1.0)
            certified_translation = bool(False)
            if stationary_a and path_kind_b == linear_translation_path_kind:
                certified_translation = True
                (
                    local_sweep_status,
                    local_decisive_triangle,
                    local_certified_fraction,
                ) = _query_pure_translation_overlap(
                    shape_b,
                    shape_a,
                    lane,
                    wp.block_dim(),
                    reference_shape_transform,
                    candidate_shape_transform,
                    shape_scale,
                    shape_margin,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_motion_radius,
                    shape_source,
                    separation_tolerance,
                    False,
                )
            elif stationary_b and path_kind_a == linear_translation_path_kind:
                certified_translation = True
                (
                    local_sweep_status,
                    local_decisive_triangle,
                    local_certified_fraction,
                ) = _query_pure_translation_overlap(
                    shape_a,
                    shape_b,
                    lane,
                    wp.block_dim(),
                    reference_shape_transform,
                    candidate_shape_transform,
                    shape_scale,
                    shape_margin,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_motion_radius,
                    shape_source,
                    separation_tolerance,
                    False,
                )
            elif not stationary_a and not stationary_b:
                if lane == 0:
                    wp.atomic_or(world_status, world_id, sweep_incomplete_bit)
                    wp.atomic_min(certified_path_fraction, world_id, 0.0)
                continue
            else:
                (
                    local_sweep_status,
                    local_decisive_triangle,
                    local_certified_fraction,
                ) = _query_bounded_sweep_ambiguity(
                    shape_a,
                    shape_b,
                    stationary_a,
                    stationary_b,
                    lane,
                    wp.block_dim(),
                    reference_body_q,
                    candidate_body_q,
                    shape_local_transform,
                    reference_shape_transform,
                    candidate_shape_transform,
                    shape_scale,
                    shape_margin,
                    shape_source,
                    shape_motion_radius,
                    shape_body,
                    body_origin_path_length,
                    shape_effective_angular_path_length,
                    (
                        (not stationary_a and path_kind_a == constant_twist_path_kind)
                        or (not stationary_b and path_kind_b == constant_twist_path_kind)
                    ),
                    separation_tolerance,
                )

            pair_certified_fraction = wp.tile_reduce(wp.min, wp.tile(local_certified_fraction, preserve_type=True))[0]
            if certified_translation:
                local_invalid = int(0)
                local_overlap = int(0)
                if local_sweep_status < 0:
                    local_invalid = 1
                elif local_sweep_status > 0:
                    local_overlap = 1
                invalid_count = wp.tile_reduce(wp.add, wp.tile(local_invalid, preserve_type=True))[0]
                overlap_count = wp.tile_reduce(wp.add, wp.tile(local_overlap, preserve_type=True))[0]
                if lane == 0:
                    if invalid_count > 0:
                        wp.atomic_or(world_status, world_id, sweep_incomplete_bit)
                        wp.atomic_min(certified_path_fraction, world_id, 0.0)
                    elif overlap_count > 0:
                        wp.atomic_or(world_status, world_id, violation_bit)
                        wp.atomic_min(certified_path_fraction, world_id, pair_certified_fraction)
                continue

            decisive_triangle = wp.tile_reduce(wp.min, wp.tile(local_decisive_triangle, preserve_type=True))[0]
            winning_lane = int(0)
            if decisive_triangle >= 0 and decisive_triangle != 2147483647:
                winning_lane = decisive_triangle % wp.block_dim()
            selected_status = int(0)
            if lane == winning_lane:
                selected_status = local_sweep_status
            sweep_status = wp.tile_reduce(wp.add, wp.tile(selected_status, preserve_type=True))[0]
            if lane == 0:
                if sweep_status < 0:
                    wp.atomic_or(world_status, world_id, sweep_incomplete_bit)
                    wp.atomic_min(certified_path_fraction, world_id, pair_certified_fraction)
                elif sweep_status > 0:
                    wp.atomic_or(world_status, world_id, sweep_incomplete_bit)
                    wp.atomic_min(certified_path_fraction, world_id, pair_certified_fraction)

    @wp.kernel(enable_backward=False, module=_module)
    def mesh_sdf_nonpenetration_oracle_kernel(
        reference_body_q: wp.array[wp.transform],
        candidate_body_q: wp.array[wp.transform],
        reference_shape_transform: wp.array[wp.transform],
        candidate_shape_transform: wp.array[wp.transform],
        shape_scale: wp.array[wp.vec3],
        shape_margin: wp.array[float],
        shape_source: wp.array[wp.uint64],
        shape_body: wp.array[int],
        shape_world: wp.array[int],
        shape_mesh_properties: wp.array[wp.int32],
        shape_pairs: wp.array[wp.vec2i],
        shape_pair_count: wp.array[int],
        reference_pair_state: wp.array[wp.uint8],
        separation_tolerance: float,
        reference_validation: bool,
        violation_bit: int,
        reference_infeasible_bit: int,
        incomplete_bit: int,
        capacity_bit: int,
        world_status: wp.array[wp.int32],
        world_min_separation: wp.array[float],
        query_incomplete: wp.array[int],
        shape_key_bits: int,
        vertex_key_max: wp.uint64,
        guard_slots_per_world: int,
        guard_feature_keys: wp.array[wp.uint64],
        guard_feature_contact_indices: wp.array[int],
        guard_feature_update_generations: wp.array[int],
        contact_generation: wp.array[int],
        contact_max: int,
        contact_count: wp.array[int],
        out_point_id: wp.array[int],
        out_shape0: wp.array[int],
        out_shape1: wp.array[int],
        out_point0: wp.array[wp.vec3],
        out_point1: wp.array[wp.vec3],
        out_offset0: wp.array[wp.vec3],
        out_offset1: wp.array[wp.vec3],
        out_normal: wp.array[wp.vec3],
        out_normal_owner: wp.array[wp.int32],
        out_is_predictive: wp.array[wp.uint8],
        out_is_strict_guard: wp.array[wp.uint8],
        out_margin0: wp.array[float],
        out_margin1: wp.array[float],
        out_tids: wp.array[int],
        total_num_blocks: int,
    ):
        block_idx, lane = wp.tid()
        pair_count = wp.min(shape_pair_count[0], shape_pairs.shape[0])
        for pair_idx in range(block_idx, pair_count, total_num_blocks):
            if reference_validation:
                if lane == 0:
                    reference_pair_state[pair_idx] = wp.uint8(0)
            elif reference_pair_state[pair_idx] != wp.uint8(0):
                continue
            pair = shape_pairs[pair_idx]
            shape_a = pair[0]
            shape_b = pair[1]
            world_a = shape_world[shape_a]
            world_b = shape_world[shape_b]
            world_id = world_a if world_a >= 0 else world_b
            if world_id < 0 and world_status.shape[0] == 1:
                world_id = 0
            if (
                world_id < 0
                or world_id >= world_status.shape[0]
                or (world_a >= 0 and world_b >= 0 and world_a != world_b)
            ):
                if lane == 0:
                    if reference_validation:
                        reference_pair_state[pair_idx] = _reference_pair_invalid
                    wp.atomic_or(query_incomplete, 0, incomplete_bit)
                continue

            margin_sum = shape_margin[shape_a] + shape_margin[shape_b]

            for mode in range(2):
                source_shape = shape_a if mode == 0 else shape_b
                target_shape = shape_b if mode == 0 else shape_a
                source_mesh_id = shape_source[source_shape]
                if source_mesh_id == wp.uint64(0):
                    if lane == 0:
                        if reference_validation:
                            reference_pair_state[pair_idx] = _reference_pair_invalid
                        wp.atomic_or(world_status, world_id, incomplete_bit)
                    continue

                mesh = wp.mesh_get(source_mesh_id)
                source_scale = shape_scale[source_shape]
                target_scale = shape_scale[target_shape]
                target_watertight = (shape_mesh_properties[target_shape] & int(MeshProperties.WATERTIGHT)) != 0
                if not _valid_mesh_scale(source_scale) or not _valid_mesh_scale(target_scale):
                    if lane == 0:
                        if reference_validation:
                            reference_pair_state[pair_idx] = _reference_pair_invalid
                        wp.atomic_or(world_status, world_id, incomplete_bit)
                    continue

                local_min_separation = float(1.0e30)
                local_best_score = float(1.0e30)
                local_best_distance = float(0.0)
                local_best_direction = wp.vec3(0.0)
                local_best_source_point = wp.vec3(0.0)
                local_best_vertex = int(-1)
                local_invalid_sample = int(0)
                for vertex in range(lane, mesh.points.shape[0], wp.block_dim()):
                    source_point_local = wp.cw_mul(mesh.points[vertex], source_scale)
                    valid, distance, direction, source_point_world = _query_endpoint(
                        source_shape,
                        target_shape,
                        source_point_local,
                        candidate_shape_transform,
                        shape_scale,
                        shape_source,
                        shape_mesh_properties,
                    )
                    if not valid:
                        local_invalid_sample = 1
                        continue
                    separation = distance - margin_sum
                    local_min_separation = wp.min(local_min_separation, separation)
                    score = separation
                    if score < local_best_score:
                        local_best_score = score
                        local_best_distance = distance
                        local_best_direction = direction
                        local_best_source_point = source_point_world
                        local_best_vertex = vertex

                best_separation = wp.tile_reduce(wp.min, wp.tile(local_min_separation, preserve_type=True))[0]
                best_score = wp.tile_reduce(wp.min, wp.tile(local_best_score, preserve_type=True))[0]
                winning_vertex_candidate = int(2147483647)
                if local_best_vertex >= 0 and local_best_score == best_score:
                    winning_vertex_candidate = local_best_vertex
                best_vertex = wp.tile_reduce(wp.min, wp.tile(winning_vertex_candidate, preserve_type=True))[0]
                if best_vertex == 2147483647:
                    best_vertex = -1

                if lane == 0 and best_vertex >= 0:
                    wp.atomic_min(world_min_separation, world_id, best_separation)

                # Vertex containment proves overlap for a watertight target.
                # Do not use ``shape_edge_range`` here: that buffer may contain SDF-specific
                # dihedral/absorption simplifications and is not exhaustive.
                local_edge_hit = int(0)
                local_edge_unresolved = int(0)
                local_edge_best_score = float(1.0e30)
                local_edge_best_source_point = wp.vec3(0.0)
                local_edge_best_target_point = wp.vec3(0.0)
                local_edge_best_normal = wp.vec3(0.0)
                local_edge_best_feature = int(-1)
                if best_vertex >= 0 and (not target_watertight or best_separation >= -separation_tolerance):
                    index_count = mesh.indices.shape[0]
                    if index_count < 3 or index_count % 3 != 0:
                        local_invalid_sample = 1
                    else:
                        for edge in range(lane, index_count, wp.block_dim()):
                            triangle = edge // 3
                            edge_in_triangle = edge % 3
                            index0 = mesh.indices[edge]
                            index1 = mesh.indices[3 * triangle + (edge_in_triangle + 1) % 3]
                            if (
                                index0 < 0
                                or index0 >= mesh.points.shape[0]
                                or index1 < 0
                                or index1 >= mesh.points.shape[0]
                            ):
                                local_invalid_sample = 1
                                continue
                            source_point0_local = wp.cw_mul(mesh.points[index0], source_scale)
                            source_point1_local = wp.cw_mul(mesh.points[index1], source_scale)
                            (
                                hit_status,
                                edge_score,
                                edge_source_point,
                                edge_target_point,
                                edge_normal,
                            ) = _query_transverse_edge_hit(
                                source_shape,
                                target_shape,
                                source_point0_local,
                                source_point1_local,
                                separation_tolerance,
                                margin_sum,
                                reference_shape_transform,
                                candidate_shape_transform,
                                shape_scale,
                                shape_source,
                                shape_mesh_properties,
                            )
                            if hit_status < 0:
                                local_invalid_sample = 1
                            elif hit_status > 0:
                                local_edge_hit = 1
                                if hit_status == 1:
                                    local_edge_unresolved = 1
                                elif hit_status == 2 and edge_score < local_edge_best_score:
                                    local_edge_best_score = edge_score
                                    local_edge_best_source_point = edge_source_point
                                    local_edge_best_target_point = edge_target_point
                                    local_edge_best_normal = edge_normal
                                    local_edge_best_feature = edge

                invalid_sample = wp.tile_reduce(wp.bit_or, wp.tile(local_invalid_sample, preserve_type=True))[0]
                unresolved_edge = wp.tile_reduce(wp.bit_or, wp.tile(local_edge_unresolved, preserve_type=True))[0]
                edge_violation = wp.tile_reduce(wp.bit_or, wp.tile(local_edge_hit, preserve_type=True))[0] != 0
                best_edge_score = wp.tile_reduce(wp.min, wp.tile(local_edge_best_score, preserve_type=True))[0]
                winning_edge_candidate = int(2147483647)
                if local_edge_best_feature >= 0 and local_edge_best_score == best_edge_score:
                    winning_edge_candidate = local_edge_best_feature
                best_edge_feature = wp.tile_reduce(wp.min, wp.tile(winning_edge_candidate, preserve_type=True))[0]
                if best_edge_feature == 2147483647:
                    best_edge_feature = -1
                edge_guard_available = best_edge_feature >= 0
                vertex_violation = (
                    target_watertight and not edge_violation and best_vertex >= 0 and best_score < -separation_tolerance
                )
                invalid_geometry = invalid_sample != 0 or best_vertex < 0
                if lane == 0:
                    if edge_violation:
                        edge_min_separation = best_edge_score if edge_guard_available else -margin_sum
                        wp.atomic_min(world_min_separation, world_id, edge_min_separation)
                    if reference_validation:
                        if invalid_geometry:
                            reference_pair_state[pair_idx] = _reference_pair_invalid
                            wp.atomic_or(world_status, world_id, incomplete_bit)
                        elif vertex_violation or edge_violation:
                            if reference_pair_state[pair_idx] != _reference_pair_invalid:
                                reference_pair_state[pair_idx] = _reference_pair_infeasible
                            wp.atomic_or(world_status, world_id, violation_bit | reference_infeasible_bit)
                    elif invalid_geometry:
                        wp.atomic_or(world_status, world_id, incomplete_bit)
                    elif vertex_violation or edge_violation:
                        status = violation_bit
                        if edge_violation and (not edge_guard_available or unresolved_edge != 0):
                            status |= incomplete_bit
                        wp.atomic_or(world_status, world_id, status)

                if reference_validation:
                    continue
                if not vertex_violation and not edge_violation:
                    continue
                if wp.static(not materialize):
                    continue

                feature_id = int(-1)
                source_point = wp.vec3(0.0)
                target_point = wp.vec3(0.0)
                direction = wp.vec3(0.0)
                if edge_violation:
                    if not edge_guard_available or local_edge_best_feature != best_edge_feature:
                        continue
                    # Keep edge and vertex identities disjoint across active-set
                    # refinements for the same directed shape pair.
                    feature_id = (best_edge_feature << 1) | 1
                    source_point = local_edge_best_source_point
                    target_point = local_edge_best_target_point
                    direction = local_edge_best_normal
                else:
                    if local_best_vertex != best_vertex:
                        continue
                    feature_id = best_vertex << 1
                    source_point = local_best_source_point
                    target_point = local_best_source_point - local_best_distance * local_best_direction
                    direction = local_best_direction

                point_a = source_point if mode == 0 else target_point
                point_b = target_point if mode == 0 else source_point
                normal_candidate = -direction if mode == 0 else direction
                normal_owner = CONTACT_NORMAL_OWNER_SHAPE_B if mode == 0 else CONTACT_NORMAL_OWNER_SHAPE_A
                body_a = shape_body[shape_a]
                body_b = shape_body[shape_b]
                owner_body = body_b if normal_owner == CONTACT_NORMAL_OWNER_SHAPE_B else body_a
                normal_reference = normal_candidate
                if owner_body >= 0:
                    owner_local = wp.quat_rotate_inv(
                        wp.transform_get_rotation(candidate_body_q[owner_body]), normal_candidate
                    )
                    normal_reference = wp.quat_rotate(
                        wp.transform_get_rotation(reference_body_q[owner_body]), owner_local
                    )
                normal_reference_length_sq = wp.length_sq(normal_reference)
                if (
                    not _finite_point(normal_reference)
                    or not wp.isfinite(normal_reference_length_sq)
                    or normal_reference_length_sq <= 1.0e-20
                ):
                    wp.atomic_or(world_status, world_id, incomplete_bit)
                    continue
                normal_reference *= _sdf_rsqrt_rn(normal_reference_length_sq)

                key_valid, feature_key = _oracle_feature_key(
                    source_shape,
                    target_shape,
                    feature_id,
                    shape_key_bits,
                    vertex_key_max,
                )
                if not key_valid:
                    wp.atomic_or(world_status, world_id, incomplete_bit)
                    continue
                contact_index = _upsert_oracle_feature(
                    world_id,
                    feature_key,
                    guard_slots_per_world,
                    guard_feature_keys,
                    guard_feature_contact_indices,
                    guard_feature_update_generations,
                    contact_generation[0],
                    contact_count,
                    contact_max,
                )
                if contact_index == -2:
                    wp.atomic_or(world_status, world_id, incomplete_bit | capacity_bit)
                if contact_index < 0:
                    continue

                out_point_id[contact_index] = (feature_id << 2) | mode
                out_shape0[contact_index] = shape_a
                out_shape1[contact_index] = shape_b
                out_point0[contact_index] = _point_in_body_frame(point_a, body_a, candidate_body_q)
                out_point1[contact_index] = _point_in_body_frame(point_b, body_b, candidate_body_q)
                out_offset0[contact_index] = _vector_in_body_frame(
                    shape_margin[shape_a] * normal_candidate, body_a, candidate_body_q
                )
                out_offset1[contact_index] = _vector_in_body_frame(
                    -shape_margin[shape_b] * normal_candidate, body_b, candidate_body_q
                )
                out_normal[contact_index] = normal_reference
                out_normal_owner[contact_index] = normal_owner
                out_is_predictive[contact_index] = wp.uint8(1)
                out_is_strict_guard[contact_index] = wp.uint8(CONTACT_STRICT_GUARD_ONLY)
                out_margin0[contact_index] = shape_margin[shape_a]
                out_margin1[contact_index] = shape_margin[shape_b]
                out_tids[contact_index] = 0

    return mesh_sdf_nonpenetration_sweep_oracle_kernel, mesh_sdf_nonpenetration_oracle_kernel


def create_narrow_phase_process_mesh_mesh_contacts_kernel(
    writer_func: Any,
    enable_heightfields: bool = True,
    reduce_contacts: bool = False,
    speculative: bool = False,
    use_precomputed_edge_data: bool = False,
    use_texture_sdf_only: bool = False,
    use_identity_sdf_scale: bool = False,
    run_on_work_overflow: bool = False,
):
    if use_identity_sdf_scale and not use_texture_sdf_only:
        raise ValueError("identity SDF scale specialization requires texture-only SDFs")
    do_edge_sdf_collision, sample_sdf_at_t = _create_sdf_contact_funcs(
        enable_heightfields, use_texture_sdf_only, texture_sample_sdf_hw, _texture_sample_sdf_hw_pair
    )
    sample_clamped = _texture_sample_sdf_hw_clamped
    sample_grad = texture_sample_sdf_grad_only_hw
    get_mesh_edge_specialized = _create_mesh_edge_accessor_func(use_precomputed_edge_data)
    get_mesh_edge_bounding_sphere_specialized = _create_get_mesh_edge_bounding_sphere_func(use_precomputed_edge_data)

    # Derive a stable module name from the factory arguments so that
    # identical configurations share the compiled CUDA kernel.  This is
    # critical for deterministic contact generation: two CollisionPipeline
    # instances with the same writer_func must execute the exact same
    # compiled code, otherwise FMA-fusion or register-allocation
    # differences between independent JIT compilations can produce subtly
    # different floating-point results, breaking bit-exact reproducibility.
    _module = (
        f"sdf_contact_{writer_func.__name__}_{enable_heightfields}_{reduce_contacts}_"
        f"{speculative}_{use_precomputed_edge_data}_{use_texture_sdf_only}_{use_identity_sdf_scale}"
    )
    if run_on_work_overflow:
        _module += "_overflow_fallback"

    @wp.kernel(enable_backward=False, module=_module)
    def mesh_sdf_collision_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        texture_sdf_table: wp.array[TextureSDFData],
        shape_sdf_index: wp.array[wp.int32],
        shape_mesh_properties: wp.array[wp.int32],
        shape_gap: wp.array[float],
        shape_base_gap: wp.array[float],
        shape_linear_velocity: wp.array[wp.vec3],
        shape_angular_velocity: wp.array[wp.vec3],
        shape_rotation_center_offset: wp.array[wp.vec3],
        collision_update_dt: float,
        max_speculative_extension: float,
        _shape_collision_aabb_lower: wp.array[wp.vec3],
        _shape_collision_aabb_upper: wp.array[wp.vec3],
        _shape_voxel_resolution: wp.array[wp.vec3i],
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_mesh_count: wp.array[int],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        mesh_edge_indices: wp.array[wp.vec2i],
        mesh_edge_centers: wp.array[wp.vec4],
        mesh_edge_halves: wp.array[wp.vec4],
        shape_edge_range: wp.array[wp.vec2i],
        writer_data: Any,
        total_num_blocks: int,
    ):
        """Process mesh-mesh and mesh-heightfield collisions using SDF-based detection."""
        block_id, t = wp.tid()

        pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])

        edge_stack = wp.tile_stack(capacity=STACK_CAPACITY, dtype=EdgeCullResult)
        # ``progress[0]`` is the next edge index the upcoming cooperative
        # culling pass should start from (a high-water mark, not a count):
        # each thread ``t`` evaluates ``progress[0] + t`` and the counter
        # advances by ``wp.block_dim()`` per pass.
        progress = wp.tile_zeros(shape=1, dtype=int, storage="shared")

        # Strided loop over pairs
        for pair_idx in range(block_id, pair_count, total_num_blocks):
            pair_encoded = shape_pairs_mesh_mesh[pair_idx]
            if wp.static(enable_heightfields):
                has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
                pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
            else:
                has_hfield = False
                pair = pair_encoded

            gap_sum = shape_gap[pair[0]] + shape_gap[pair[1]]
            base_gap_sum = shape_base_gap[pair[0]] + shape_base_gap[pair[1]]

            for mode in range(2):
                tri_shape = pair[mode]
                sdf_shape = pair[1 - mode]

                if wp.static(enable_heightfields):
                    tri_is_hfield = has_hfield and mode == 0
                    sdf_is_hfield = has_hfield and mode == 1
                else:
                    tri_is_hfield = False
                    sdf_is_hfield = False
                tri_type = GeoType.HFIELD if tri_is_hfield else GeoType.MESH

                mesh_id_tri = shape_source[tri_shape]
                mesh_id_sdf = shape_source[sdf_shape]

                # Edge carriers need a mesh source unless they are heightfields.
                if not tri_is_hfield and mesh_id_tri == wp.uint64(0):
                    continue

                hfd_tri = HeightfieldData()
                hfd_sdf = HeightfieldData()
                if wp.static(enable_heightfields):
                    if tri_is_hfield:
                        hfd_tri = heightfield_data[shape_heightfield_index[tri_shape]]
                    if sdf_is_hfield:
                        hfd_sdf = heightfield_data[shape_heightfield_index[sdf_shape]]
                sdf_mesh_query_type = resolve_mesh_sign_method(shape_mesh_properties[sdf_shape])

                # SDF availability: heightfields always use on-the-fly evaluation
                use_bvh_for_sdf = False
                if not sdf_is_hfield:
                    sdf_idx = shape_sdf_index[sdf_shape]
                    if wp.static(not use_texture_sdf_only):
                        use_bvh_for_sdf = sdf_idx < 0 or sdf_idx >= texture_sdf_table.shape[0]
                        if not use_bvh_for_sdf:
                            use_bvh_for_sdf = texture_sdf_table[sdf_idx].coarse_texture.width == 0
                        if use_bvh_for_sdf and mesh_id_sdf == wp.uint64(0):
                            continue

                scale_data_tri = shape_data[tri_shape]
                scale_data_sdf = shape_data[sdf_shape]
                mesh_scale_tri = wp.vec3(scale_data_tri[0], scale_data_tri[1], scale_data_tri[2])
                mesh_scale_sdf = wp.vec3(scale_data_sdf[0], scale_data_sdf[1], scale_data_sdf[2])

                X_tri_ws = shape_transform[tri_shape]
                X_sdf_ws = shape_transform[sdf_shape]

                # Determine sdf_scale for the SDF query.
                # Heightfields always use scale=identity, since SDF is directly sampled
                # from elevation grid. For texture SDF, override to identity when scale
                # is already baked. For BVH fallback, use the shape scale.
                texture_sdf = TextureSDFData()
                if sdf_is_hfield:
                    sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                else:
                    if not use_bvh_for_sdf:
                        texture_sdf = texture_sdf_table[sdf_idx]
                    if wp.static(use_identity_sdf_scale):
                        sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                    else:
                        sdf_scale = mesh_scale_sdf
                        if not use_bvh_for_sdf and texture_sdf.scale_baked:
                            sdf_scale = wp.vec3(1.0, 1.0, 1.0)

                X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_sdf_ws), X_tri_ws)

                triangle_mesh_margin = scale_data_tri[3]
                sdf_mesh_margin = scale_data_sdf[3]

                if wp.static(use_identity_sdf_scale):
                    inv_sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                    min_sdf_scale = float(1.0)
                    edge_radius_scale = float(1.0)
                else:
                    inv_sdf_scale, min_sdf_scale = safe_sdf_scale_inverse(sdf_scale)
                    edge_radius_scale = wp.max(
                        wp.max(wp.abs(inv_sdf_scale[0]), wp.abs(inv_sdf_scale[1])), wp.abs(inv_sdf_scale[2])
                    )

                contact_threshold = gap_sum + triangle_mesh_margin + sdf_mesh_margin
                contact_threshold_unscaled = contact_threshold / min_sdf_scale
                use_texture_sdf_for_search = False
                texture_voxel_radius = float(0.0)
                if wp.static(enable_heightfields):
                    if not sdf_is_hfield and not use_bvh_for_sdf:
                        use_texture_sdf_for_search = True
                        texture_voxel_radius = texture_sdf.voxel_radius
                elif not use_bvh_for_sdf:
                    use_texture_sdf_for_search = True
                    texture_voxel_radius = texture_sdf.voxel_radius
                search_precision_unscaled = mesh_sdf_contact_search_precision(
                    triangle_mesh_margin + sdf_mesh_margin,
                    min_sdf_scale,
                    texture_voxel_radius,
                    use_texture_sdf_for_search,
                )

                edge_range_tri = shape_edge_range[tri_shape]
                num_edges = get_edge_count(tri_type, edge_range_tri, hfd_tri)

                wp.tile_scatter_masked(progress, 0, 0, t == 0)

                sdf_is_heightfield = sdf_is_hfield
                sdf_aabb_lower = texture_sdf.sdf_box_lower
                sdf_aabb_upper = texture_sdf.sdf_box_upper

                # Cooperative edge-culling + processing. Each outer
                # iteration (a) fills the tile stack with up to
                # ``block_dim`` accepted edges via cooperative pushes,
                # (b) fully drains the stack, processing every accepted
                # edge through ``do_edge_sdf_collision``, and
                # (c) explicitly clears the stack as a defensive,
                # uniformly-called cooperative barrier before the next
                # outer iteration. Draining is essential: a single
                # ``tile_stack_pop`` only removes ``block_dim`` items, so
                # if the inner push loop overshot (the push gate caps
                # pre-push count at ``block_dim - 1`` but the cooperative
                # push itself adds up to ``block_dim`` more) the
                # remainder must be popped before we advance the progress
                # counter — otherwise those edges would be silently
                # dropped by the trailing ``tile_stack_clear``.
                # This block is duplicated in
                # ``mesh_sdf_collision_global_reduce_kernel`` (different
                # edge range and contact writer) — keep the two in sync.
                while wp.tile_extract(progress, 0) < num_edges:
                    capacity = wp.block_dim()
                    while wp.tile_extract(progress, 0) < num_edges and wp.tile_stack_count(edge_stack) < capacity:
                        base_edge_idx = wp.tile_extract(progress, 0)
                        edge_idx = base_edge_idx + t
                        add_edge = False
                        midpoint_sdf = float(0.0)

                        if edge_idx < num_edges:
                            if wp.static(enable_heightfields):
                                if tri_type == GeoType.HFIELD:
                                    v0_scaled, v1_scaled = get_edge_from_heightfield(
                                        hfd_tri, heightfield_elevations, X_mesh_to_sdf, edge_idx
                                    )
                                    v0_cull = wp.cw_mul(v0_scaled, inv_sdf_scale)
                                    v1_cull = wp.cw_mul(v1_scaled, inv_sdf_scale)
                                    bsphere_center, bsphere_radius = get_edge_bounding_sphere(v0_cull, v1_cull)
                                else:
                                    bsphere_center, bsphere_radius = get_mesh_edge_bounding_sphere_specialized(
                                        mesh_id_tri,
                                        mesh_edge_indices,
                                        mesh_edge_centers,
                                        edge_range_tri,
                                        mesh_scale_tri,
                                        X_mesh_to_sdf,
                                        inv_sdf_scale,
                                        edge_radius_scale,
                                        edge_idx,
                                    )
                            else:
                                bsphere_center, bsphere_radius = get_mesh_edge_bounding_sphere_specialized(
                                    mesh_id_tri,
                                    mesh_edge_indices,
                                    mesh_edge_centers,
                                    edge_range_tri,
                                    mesh_scale_tri,
                                    X_mesh_to_sdf,
                                    inv_sdf_scale,
                                    edge_radius_scale,
                                    edge_idx,
                                )

                            threshold = bsphere_radius + contact_threshold_unscaled

                            if wp.static(enable_heightfields) and sdf_is_heightfield:
                                midpoint_sdf = sample_sdf_heightfield(hfd_sdf, heightfield_elevations, bsphere_center)
                                add_edge = midpoint_sdf <= threshold
                            elif wp.static(not use_texture_sdf_only) and use_bvh_for_sdf:
                                midpoint_sdf = sample_sdf_using_mesh(
                                    mesh_id_sdf,
                                    bsphere_center,
                                    _SDF_QUERY_RADIUS_SLACK * threshold,
                                    sdf_mesh_query_type,
                                )
                                add_edge = midpoint_sdf <= threshold
                            else:
                                culling_radius = threshold
                                clamped = wp.min(wp.max(bsphere_center, sdf_aabb_lower), sdf_aabb_upper)
                                aabb_dist_sq = wp.length_sq(bsphere_center - clamped)
                                if aabb_dist_sq > culling_radius * culling_radius:
                                    add_edge = False
                                else:
                                    diff_mag = float(0.0)
                                    if aabb_dist_sq > 0.0:
                                        diff_mag = wp.sqrt(aabb_dist_sq)
                                    midpoint_sdf = wp.static(sample_clamped)(texture_sdf, clamped, diff_mag)
                                    add_edge = midpoint_sdf <= culling_radius

                        cull_result = EdgeCullResult()
                        cull_result.edge_idx = edge_idx
                        cull_result.midpoint_sdf = midpoint_sdf
                        wp.tile_stack_push(edge_stack, cull_result, add_edge)
                        wp.tile_scatter_masked(progress, 0, base_edge_idx + capacity, t == 0)

                    # Drain the stack completely. ``tile_stack_pop`` only
                    # removes up to ``block_dim`` items per call, so we
                    # loop until empty — a single pop followed by
                    # ``tile_stack_clear`` would silently discard any
                    # accepted edges that overflowed the prior push. The
                    # trailing ``tile_stack_clear`` (after this drain) is
                    # a defensive no-op barrier; see the comment block
                    # above the outer ``while``.
                    while wp.tile_stack_count(edge_stack) > 0:
                        popped, edge_slot = wp.tile_stack_pop(edge_stack)
                        my_edge_idx = popped.edge_idx
                        cached_sdf_val = popped.midpoint_sdf
                        has_edge = edge_slot >= 0

                        if has_edge:
                            corner_ownership = int(0)
                            if wp.static(enable_heightfields):
                                if tri_type == GeoType.HFIELD:
                                    v0s, v1s = get_edge_from_heightfield(
                                        hfd_tri,
                                        heightfield_elevations,
                                        X_mesh_to_sdf,
                                        my_edge_idx,
                                    )
                                else:
                                    v0s, v1s, corner_ownership = get_mesh_edge_specialized(
                                        mesh_id_tri,
                                        mesh_edge_indices,
                                        mesh_edge_centers,
                                        mesh_edge_halves,
                                        edge_range_tri,
                                        mesh_scale_tri,
                                        X_mesh_to_sdf,
                                        my_edge_idx,
                                    )
                            else:
                                v0s, v1s, corner_ownership = get_mesh_edge_specialized(
                                    mesh_id_tri,
                                    mesh_edge_indices,
                                    mesh_edge_centers,
                                    mesh_edge_halves,
                                    edge_range_tri,
                                    mesh_scale_tri,
                                    X_mesh_to_sdf,
                                    my_edge_idx,
                                )
                            v0 = wp.cw_mul(v0s, inv_sdf_scale)
                            v1 = wp.cw_mul(v1s, inv_sdf_scale)

                            edge_segment_count = mesh_sdf_contact_segment_count(
                                wp.length(v1 - v0), texture_voxel_radius, use_texture_sdf_for_search
                            )

                            # Each segment contributes at most one interior
                            # minimum. The final two lanes retain the authored
                            # endpoint behavior exactly once per original edge.
                            first_segment_best_endpoint = int(0)
                            last_segment_best_endpoint = int(0)
                            for contact_feature in range(edge_segment_count + 2):
                                candidate_endpoint = int(0)
                                edge_segment_idx = contact_feature
                                if contact_feature >= edge_segment_count:
                                    candidate_endpoint = contact_feature - edge_segment_count + 1
                                    edge_segment_idx = 0 if candidate_endpoint == 1 else edge_segment_count - 1

                                segment_v0, segment_v1 = mesh_sdf_contact_segment_bounds(
                                    v0, v1, edge_segment_idx, edge_segment_count
                                )
                                bsphere_center_inner, bsphere_radius_inner = get_edge_bounding_sphere(
                                    segment_v0, segment_v1
                                )
                                segment_midpoint_sdf = cached_sdf_val
                                if edge_segment_count != 1 and 2 * edge_segment_idx + 1 != edge_segment_count:
                                    segment_midpoint_sdf = sample_sdf_at_t(
                                        texture_sdf,
                                        mesh_id_sdf,
                                        segment_v0,
                                        segment_v1 - segment_v0,
                                        0.5,
                                        use_bvh_for_sdf,
                                        sdf_mesh_query_type,
                                        sdf_is_hfield,
                                        hfd_sdf,
                                        heightfield_elevations,
                                    )

                                candidate_dist_unscaled = float(0.0)
                                candidate_point_unscaled = wp.vec3(0.0)
                                best_endpoint = int(0)
                                emit_candidate = segment_midpoint_sdf <= (
                                    bsphere_radius_inner + contact_threshold_unscaled
                                )
                                if candidate_endpoint == 0 and emit_candidate:
                                    candidate_dist_unscaled, candidate_point_unscaled, best_endpoint = (
                                        do_edge_sdf_collision(
                                            texture_sdf,
                                            mesh_id_sdf,
                                            segment_v0,
                                            segment_v1,
                                            segment_midpoint_sdf,
                                            use_bvh_for_sdf,
                                            sdf_mesh_query_type,
                                            sdf_is_hfield,
                                            hfd_sdf,
                                            heightfield_elevations,
                                            search_precision_unscaled,
                                        )
                                    )
                                    if edge_segment_idx == 0:
                                        first_segment_best_endpoint = best_endpoint
                                    if edge_segment_idx == edge_segment_count - 1:
                                        last_segment_best_endpoint = best_endpoint
                                    best_is_authored_endpoint = (best_endpoint == 1 and edge_segment_idx == 0) or (
                                        best_endpoint == 2 and edge_segment_idx == edge_segment_count - 1
                                    )
                                    emit_candidate = mesh_sdf_contact_segment_minimum_is_unique(
                                        best_endpoint, edge_segment_idx
                                    ) and (
                                        not best_is_authored_endpoint
                                        or mesh_sdf_contact_endpoint_owned(corner_ownership, best_endpoint)
                                    )
                                elif candidate_endpoint != 0 and emit_candidate:
                                    boundary_best_endpoint = (
                                        first_segment_best_endpoint
                                        if candidate_endpoint == 1
                                        else last_segment_best_endpoint
                                    )
                                    emit_candidate = (
                                        mesh_sdf_contact_endpoint_owned(corner_ownership, candidate_endpoint)
                                        and boundary_best_endpoint != candidate_endpoint
                                    )
                                    if emit_candidate:
                                        candidate_point_unscaled = v0 if candidate_endpoint == 1 else v1
                                        candidate_dist_unscaled = sample_sdf_at_t(
                                            texture_sdf,
                                            mesh_id_sdf,
                                            v0,
                                            v1 - v0,
                                            float(candidate_endpoint - 1),
                                            use_bvh_for_sdf,
                                            sdf_mesh_query_type,
                                            sdf_is_hfield,
                                            hfd_sdf,
                                            heightfield_elevations,
                                        )

                                # Gap may widen the edge cull enough to find
                                # SDF minima that the inner contact shell would
                                # not have considered. An alleged inner contact
                                # must still pass the inner 1-Lipschitz cull.
                                dist_approx = candidate_dist_unscaled * min_sdf_scale
                                inner_cull_consistent = False
                                if emit_candidate:
                                    inner_cull_consistent = mesh_sdf_contact_passes_inner_cull_consistency(
                                        dist_approx,
                                        triangle_mesh_margin + sdf_mesh_margin,
                                        segment_midpoint_sdf,
                                        bsphere_center_inner,
                                        bsphere_radius_inner,
                                        sdf_aabb_lower,
                                        sdf_aabb_upper,
                                        min_sdf_scale,
                                        use_texture_sdf_for_search,
                                    )
                                if emit_candidate and dist_approx < contact_threshold and inner_cull_consistent:
                                    if wp.static(enable_heightfields):
                                        if sdf_is_hfield:
                                            candidate_dist_unscaled, direction_unscaled = sample_sdf_grad_heightfield(
                                                hfd_sdf, heightfield_elevations, candidate_point_unscaled
                                            )
                                        elif wp.static(not use_texture_sdf_only) and use_bvh_for_sdf:
                                            candidate_dist_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                                mesh_id_sdf,
                                                candidate_point_unscaled,
                                                _MESH_QUERY_MAX_DIST,
                                                sdf_mesh_query_type,
                                            )
                                        else:
                                            direction_unscaled = wp.static(sample_grad)(
                                                texture_sdf, candidate_point_unscaled
                                            )
                                    else:
                                        if wp.static(not use_texture_sdf_only) and use_bvh_for_sdf:
                                            candidate_dist_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                                mesh_id_sdf,
                                                candidate_point_unscaled,
                                                _MESH_QUERY_MAX_DIST,
                                                sdf_mesh_query_type,
                                            )
                                        else:
                                            direction_unscaled = wp.static(sample_grad)(
                                                texture_sdf, candidate_point_unscaled
                                            )

                                    if wp.static(use_identity_sdf_scale):
                                        dist = candidate_dist_unscaled
                                        direction = direction_unscaled
                                        point = candidate_point_unscaled
                                    else:
                                        dist, direction = scale_sdf_result_to_world(
                                            candidate_dist_unscaled,
                                            direction_unscaled,
                                            sdf_scale,
                                            inv_sdf_scale,
                                            min_sdf_scale,
                                        )
                                        point = wp.cw_mul(candidate_point_unscaled, sdf_scale)
                                    point_world = wp.transform_point(X_sdf_ws, point)

                                    direction_world = wp.transform_vector(X_sdf_ws, direction)
                                    direction_len_sq = wp.length_sq(direction_world)
                                    if direction_len_sq > 0.0:
                                        direction_world = direction_world * _sdf_rsqrt_rn(direction_len_sq)
                                    else:
                                        fallback_dir = point_world - wp.transform_get_translation(X_sdf_ws)
                                        fallback_len_sq = wp.length_sq(fallback_dir)
                                        if fallback_len_sq > 0.0:
                                            direction_world = fallback_dir * _sdf_rsqrt_rn(fallback_len_sq)
                                        else:
                                            direction_world = wp.vec3(0.0, 1.0, 0.0)

                                    contact_data = ContactData()
                                    contact_data.contact_point_center = point_world - 0.5 * dist * direction_world
                                    contact_data.contact_normal_a_to_b = (
                                        -direction_world if mode == 0 else direction_world
                                    )
                                    contact_data.contact_distance = dist
                                    contact_data.radius_eff_a = 0.0
                                    contact_data.radius_eff_b = 0.0
                                    contact_data.margin_a = shape_data[pair[0]][3]
                                    contact_data.margin_b = shape_data[pair[1]][3]
                                    contact_data.shape_a = pair[0]
                                    contact_data.shape_b = pair[1]
                                    if wp.static(speculative):
                                        contact_data.gap_sum = base_gap_sum
                                    else:
                                        contact_data.gap_sum = gap_sum
                                    sort_sub_key = mesh_sdf_contact_sort_sub_key(
                                        my_edge_idx, edge_segment_idx, mode, candidate_endpoint
                                    )
                                    is_owned_endpoint = mesh_sdf_contact_is_owned_endpoint(
                                        candidate_endpoint, best_endpoint, edge_segment_idx, edge_segment_count
                                    )
                                    contact_data.sort_sub_key = pack_contact_is_canonical_endpoint(
                                        sort_sub_key, is_owned_endpoint
                                    )
                                    contact_data.strict_guard_provenance_finalized = int(is_owned_endpoint)

                                    writer_func(contact_data, writer_data, -1)

                    # Defensive cooperative reset before the next outer
                    # iteration. The drain loop above already left the
                    # stack empty, so this is logically a no-op, but it
                    # is a uniformly-called barrier that pairs cleanly
                    # with the inner push loop and matches the original
                    # ``push -> pop -> clear`` pattern that empirically
                    # avoided a deadlock in deterministic mesh-mesh
                    # scenes (see ``example_basic_shapes6_determinism``).
                    wp.tile_stack_clear(edge_stack)

    # Return early if contact reduction is disabled
    if not reduce_contacts:
        return mesh_sdf_collision_kernel

    # =========================================================================
    # Global reduction variant: uses hashtable instead of shared-memory reduction.
    # Same block_offsets load balancing and shared-memory triangle selection,
    # but contacts are written directly to global buffer + hashtable.
    # =========================================================================

    @wp.kernel(enable_backward=False, launch_bounds=(256, 2), module=_module)
    def mesh_sdf_collision_global_reduce_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        texture_sdf_table: wp.array[TextureSDFData],
        shape_sdf_index: wp.array[wp.int32],
        shape_mesh_properties: wp.array[wp.int32],
        shape_gap: wp.array[float],
        shape_base_gap: wp.array[float],
        shape_linear_velocity: wp.array[wp.vec3],
        shape_angular_velocity: wp.array[wp.vec3],
        shape_rotation_center_offset: wp.array[wp.vec3],
        collision_update_dt: float,
        max_speculative_extension: float,
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_voxel_resolution: wp.array[wp.vec3i],
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_mesh_count: wp.array[int],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        mesh_edge_indices: wp.array[wp.vec2i],
        mesh_edge_centers: wp.array[wp.vec4],
        mesh_edge_halves: wp.array[wp.vec4],
        shape_edge_range: wp.array[wp.vec2i],
        block_offsets: wp.array[wp.int32],
        reducer_data: GlobalContactReducerData,
        work_state: wp.array[wp.int32],
        total_num_blocks: int,
    ):
        """Process mesh-mesh collisions with global hashtable contact reduction.

        Same load balancing and triangle selection as the thread-block reduce kernel,
        but contacts are written directly to the global buffer and registered in the
        hashtable inline, matching thread-block reduction contact quality:

        - Midpoint-centered position for spatial extreme projection
        - Fixed beta threshold (0.0001 m)
        - Pair-first shape frame and AABB for mode-independent voxel computation
        """
        block_id, t = wp.tid()
        if wp.static(run_on_work_overflow):
            if work_state[_SDF_WORK_OVERFLOWED] == 0:
                return
        pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])
        total_combos = block_offsets[pair_count]

        edge_stack = wp.tile_stack(capacity=STACK_CAPACITY, dtype=EdgeCullResult)
        # ``progress[0]`` is the next edge index the upcoming cooperative
        # culling pass should start from (a high-water mark, not a count):
        # each thread ``t`` evaluates ``progress[0] + t`` and the counter
        # advances by ``wp.block_dim()`` per pass.
        progress = wp.tile_zeros(shape=1, dtype=int, storage="shared")

        for combo_idx in range(block_id, total_combos, total_num_blocks):
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
            pair_encoded = shape_pairs_mesh_mesh[pair_idx]
            if wp.static(enable_heightfields):
                has_hfield = (pair_encoded[0] & SHAPE_PAIR_HFIELD_BIT) != 0
                pair = wp.vec2i(pair_encoded[0] & SHAPE_PAIR_INDEX_MASK, pair_encoded[1])
            else:
                has_hfield = False
                pair = pair_encoded

            gap_sum = shape_gap[pair[0]] + shape_gap[pair[1]]
            base_gap_sum = shape_base_gap[pair[0]] + shape_base_gap[pair[1]]

            for mode in range(2):
                tri_shape = pair[mode]
                sdf_shape = pair[1 - mode]

                if wp.static(enable_heightfields):
                    tri_is_hfield = has_hfield and mode == 0
                    sdf_is_hfield = has_hfield and mode == 1
                else:
                    tri_is_hfield = False
                    sdf_is_hfield = False
                tri_type = GeoType.HFIELD if tri_is_hfield else GeoType.MESH

                mesh_id_tri = shape_source[tri_shape]
                mesh_id_sdf = shape_source[sdf_shape]

                if not tri_is_hfield and mesh_id_tri == wp.uint64(0):
                    continue

                hfd_tri = HeightfieldData()
                hfd_sdf = HeightfieldData()
                if wp.static(enable_heightfields):
                    if tri_is_hfield:
                        hfd_tri = heightfield_data[shape_heightfield_index[tri_shape]]
                    if sdf_is_hfield:
                        hfd_sdf = heightfield_data[shape_heightfield_index[sdf_shape]]
                sdf_mesh_query_type = resolve_mesh_sign_method(shape_mesh_properties[sdf_shape])

                use_bvh_for_sdf = False
                if not sdf_is_hfield:
                    sdf_idx = shape_sdf_index[sdf_shape]
                    if wp.static(not use_texture_sdf_only):
                        use_bvh_for_sdf = sdf_idx < 0 or sdf_idx >= texture_sdf_table.shape[0]
                        if not use_bvh_for_sdf:
                            use_bvh_for_sdf = texture_sdf_table[sdf_idx].coarse_texture.width == 0
                        if use_bvh_for_sdf and mesh_id_sdf == wp.uint64(0):
                            continue

                scale_data_tri = shape_data[tri_shape]
                scale_data_sdf = shape_data[sdf_shape]
                mesh_scale_tri = wp.vec3(scale_data_tri[0], scale_data_tri[1], scale_data_tri[2])
                mesh_scale_sdf = wp.vec3(scale_data_sdf[0], scale_data_sdf[1], scale_data_sdf[2])

                X_tri_ws = shape_transform[tri_shape]
                X_sdf_ws = shape_transform[sdf_shape]

                texture_sdf = TextureSDFData()
                if sdf_is_hfield:
                    sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                else:
                    if not use_bvh_for_sdf:
                        texture_sdf = texture_sdf_table[sdf_idx]
                    if wp.static(use_identity_sdf_scale):
                        sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                    else:
                        sdf_scale = mesh_scale_sdf
                        if not use_bvh_for_sdf and texture_sdf.scale_baked:
                            sdf_scale = wp.vec3(1.0, 1.0, 1.0)

                X_mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(X_sdf_ws), X_tri_ws)

                triangle_mesh_margin = scale_data_tri[3]
                sdf_mesh_margin = scale_data_sdf[3]

                if wp.static(use_identity_sdf_scale):
                    inv_sdf_scale = wp.vec3(1.0, 1.0, 1.0)
                    min_sdf_scale = float(1.0)
                    edge_radius_scale = float(1.0)
                else:
                    inv_sdf_scale, min_sdf_scale = safe_sdf_scale_inverse(sdf_scale)
                    edge_radius_scale = wp.max(
                        wp.max(wp.abs(inv_sdf_scale[0]), wp.abs(inv_sdf_scale[1])), wp.abs(inv_sdf_scale[2])
                    )

                contact_threshold = gap_sum + triangle_mesh_margin + sdf_mesh_margin
                contact_threshold_unscaled = contact_threshold / min_sdf_scale
                use_texture_sdf_for_search = False
                texture_voxel_radius = float(0.0)
                if wp.static(enable_heightfields):
                    if not sdf_is_hfield and not use_bvh_for_sdf:
                        use_texture_sdf_for_search = True
                        texture_voxel_radius = texture_sdf.voxel_radius
                elif not use_bvh_for_sdf:
                    use_texture_sdf_for_search = True
                    texture_voxel_radius = texture_sdf.voxel_radius
                search_precision_unscaled = mesh_sdf_contact_search_precision(
                    triangle_mesh_margin + sdf_mesh_margin,
                    min_sdf_scale,
                    texture_voxel_radius,
                    use_texture_sdf_for_search,
                )

                edge_range_tri = shape_edge_range[tri_shape]
                num_edges = get_edge_count(tri_type, edge_range_tri, hfd_tri)
                chunk_size = (num_edges + blocks_for_pair - 1) // blocks_for_pair
                edge_start = block_in_pair * chunk_size
                edge_end = wp.min(edge_start + chunk_size, num_edges)

                wp.tile_scatter_masked(progress, 0, edge_start, t == 0)

                sdf_is_heightfield = sdf_is_hfield
                sdf_aabb_lower = texture_sdf.sdf_box_lower
                sdf_aabb_upper = texture_sdf.sdf_box_upper

                # Cooperative edge-culling + processing. See the matching
                # loop in ``mesh_sdf_collision_kernel`` for the invariant
                # discussion; the drain-until-empty pop is essential so
                # that edges overflowing the prior push are not silently
                # dropped. Keep this block in sync with its twin.
                while wp.tile_extract(progress, 0) < edge_end:
                    capacity = wp.block_dim()
                    while wp.tile_extract(progress, 0) < edge_end and wp.tile_stack_count(edge_stack) < capacity:
                        base_edge_idx = wp.tile_extract(progress, 0)
                        edge_idx = base_edge_idx + t
                        add_edge = False
                        midpoint_sdf = float(0.0)

                        if edge_idx < edge_end:
                            if wp.static(enable_heightfields):
                                if tri_type == GeoType.HFIELD:
                                    v0_scaled, v1_scaled = get_edge_from_heightfield(
                                        hfd_tri, heightfield_elevations, X_mesh_to_sdf, edge_idx
                                    )
                                    v0_cull = wp.cw_mul(v0_scaled, inv_sdf_scale)
                                    v1_cull = wp.cw_mul(v1_scaled, inv_sdf_scale)
                                    bsphere_center, bsphere_radius = get_edge_bounding_sphere(v0_cull, v1_cull)
                                else:
                                    bsphere_center, bsphere_radius = get_mesh_edge_bounding_sphere_specialized(
                                        mesh_id_tri,
                                        mesh_edge_indices,
                                        mesh_edge_centers,
                                        edge_range_tri,
                                        mesh_scale_tri,
                                        X_mesh_to_sdf,
                                        inv_sdf_scale,
                                        edge_radius_scale,
                                        edge_idx,
                                    )
                            else:
                                bsphere_center, bsphere_radius = get_mesh_edge_bounding_sphere_specialized(
                                    mesh_id_tri,
                                    mesh_edge_indices,
                                    mesh_edge_centers,
                                    edge_range_tri,
                                    mesh_scale_tri,
                                    X_mesh_to_sdf,
                                    inv_sdf_scale,
                                    edge_radius_scale,
                                    edge_idx,
                                )

                            threshold = bsphere_radius + contact_threshold_unscaled

                            if wp.static(enable_heightfields) and sdf_is_heightfield:
                                midpoint_sdf = sample_sdf_heightfield(hfd_sdf, heightfield_elevations, bsphere_center)
                                add_edge = midpoint_sdf <= threshold
                            elif wp.static(not use_texture_sdf_only) and use_bvh_for_sdf:
                                midpoint_sdf = sample_sdf_using_mesh(
                                    mesh_id_sdf,
                                    bsphere_center,
                                    _SDF_QUERY_RADIUS_SLACK * threshold,
                                    sdf_mesh_query_type,
                                )
                                add_edge = midpoint_sdf <= threshold
                            else:
                                culling_radius = threshold
                                clamped = wp.min(wp.max(bsphere_center, sdf_aabb_lower), sdf_aabb_upper)
                                aabb_dist_sq = wp.length_sq(bsphere_center - clamped)
                                if aabb_dist_sq > culling_radius * culling_radius:
                                    add_edge = False
                                else:
                                    diff_mag = float(0.0)
                                    if aabb_dist_sq > 0.0:
                                        diff_mag = wp.sqrt(aabb_dist_sq)
                                    midpoint_sdf = wp.static(sample_clamped)(texture_sdf, clamped, diff_mag)
                                    add_edge = midpoint_sdf <= culling_radius

                        cull_result = EdgeCullResult()
                        cull_result.edge_idx = edge_idx
                        cull_result.midpoint_sdf = midpoint_sdf
                        wp.tile_stack_push(edge_stack, cull_result, add_edge)
                        wp.tile_scatter_masked(progress, 0, base_edge_idx + capacity, t == 0)

                    # Drain the stack completely — see the matching loop
                    # in ``mesh_sdf_collision_kernel`` for why a single
                    # pop would silently drop overflowed accepted edges.
                    # The trailing ``tile_stack_clear`` is a defensive
                    # no-op barrier (see that same comment block).
                    while wp.tile_stack_count(edge_stack) > 0:
                        popped, edge_slot = wp.tile_stack_pop(edge_stack)
                        my_edge_idx = popped.edge_idx
                        cached_sdf_val = popped.midpoint_sdf
                        has_edge = edge_slot >= 0

                        if has_edge:
                            corner_ownership = int(0)
                            if wp.static(enable_heightfields):
                                if tri_type == GeoType.HFIELD:
                                    v0s, v1s = get_edge_from_heightfield(
                                        hfd_tri,
                                        heightfield_elevations,
                                        X_mesh_to_sdf,
                                        my_edge_idx,
                                    )
                                else:
                                    v0s, v1s, corner_ownership = get_mesh_edge_specialized(
                                        mesh_id_tri,
                                        mesh_edge_indices,
                                        mesh_edge_centers,
                                        mesh_edge_halves,
                                        edge_range_tri,
                                        mesh_scale_tri,
                                        X_mesh_to_sdf,
                                        my_edge_idx,
                                    )
                            else:
                                v0s, v1s, corner_ownership = get_mesh_edge_specialized(
                                    mesh_id_tri,
                                    mesh_edge_indices,
                                    mesh_edge_centers,
                                    mesh_edge_halves,
                                    edge_range_tri,
                                    mesh_scale_tri,
                                    X_mesh_to_sdf,
                                    my_edge_idx,
                                )
                            v0 = wp.cw_mul(v0s, inv_sdf_scale)
                            v1 = wp.cw_mul(v1s, inv_sdf_scale)

                            edge_segment_count = mesh_sdf_contact_segment_count(
                                wp.length(v1 - v0), texture_voxel_radius, use_texture_sdf_for_search
                            )
                            voxel_shape = mesh_sdf_contact_voxel_owner(
                                pair[0],
                                pair[1],
                                shape_collision_aabb_lower,
                                shape_collision_aabb_upper,
                                shape_voxel_resolution,
                            )
                            X_voxel_ws = shape_transform[voxel_shape]
                            aabb_lower_voxel = shape_collision_aabb_lower[voxel_shape]
                            aabb_upper_voxel = shape_collision_aabb_upper[voxel_shape]
                            voxel_res = shape_voxel_resolution[voxel_shape]
                            margin_sum = triangle_mesh_margin + sdf_mesh_margin
                            midpoint = (
                                wp.transform_get_translation(X_tri_ws) + wp.transform_get_translation(X_sdf_ws)
                            ) * 0.5
                            inner_spatial_depth = margin_sum
                            if use_texture_sdf_for_search:
                                inner_spatial_depth += wp.min(texture_voxel_radius * min_sdf_scale, base_gap_sum)
                            outer_spatial_depth = margin_sum + gap_sum
                            if wp.static(speculative):
                                # Velocity-expanded candidates compete only in the bounded predictive manifold.
                                outer_spatial_depth = margin_sum + base_gap_sum

                            first_segment_best_endpoint = int(0)
                            last_segment_best_endpoint = int(0)
                            for contact_feature in range(edge_segment_count + 2):
                                candidate_endpoint = int(0)
                                edge_segment_idx = contact_feature
                                if contact_feature >= edge_segment_count:
                                    candidate_endpoint = contact_feature - edge_segment_count + 1
                                    edge_segment_idx = 0 if candidate_endpoint == 1 else edge_segment_count - 1

                                segment_v0, segment_v1 = mesh_sdf_contact_segment_bounds(
                                    v0, v1, edge_segment_idx, edge_segment_count
                                )
                                bsphere_center_inner, bsphere_radius_inner = get_edge_bounding_sphere(
                                    segment_v0, segment_v1
                                )
                                segment_midpoint_sdf = cached_sdf_val
                                if edge_segment_count != 1 and 2 * edge_segment_idx + 1 != edge_segment_count:
                                    segment_midpoint_sdf = sample_sdf_at_t(
                                        texture_sdf,
                                        mesh_id_sdf,
                                        segment_v0,
                                        segment_v1 - segment_v0,
                                        0.5,
                                        use_bvh_for_sdf,
                                        sdf_mesh_query_type,
                                        sdf_is_hfield,
                                        hfd_sdf,
                                        heightfield_elevations,
                                    )

                                candidate_dist_unscaled = float(0.0)
                                candidate_point_unscaled = wp.vec3(0.0)
                                best_endpoint = int(0)
                                emit_candidate = segment_midpoint_sdf <= (
                                    bsphere_radius_inner + contact_threshold_unscaled
                                )
                                if candidate_endpoint == 0 and emit_candidate:
                                    candidate_dist_unscaled, candidate_point_unscaled, best_endpoint = (
                                        do_edge_sdf_collision(
                                            texture_sdf,
                                            mesh_id_sdf,
                                            segment_v0,
                                            segment_v1,
                                            segment_midpoint_sdf,
                                            use_bvh_for_sdf,
                                            sdf_mesh_query_type,
                                            sdf_is_hfield,
                                            hfd_sdf,
                                            heightfield_elevations,
                                            search_precision_unscaled,
                                        )
                                    )
                                    if edge_segment_idx == 0:
                                        first_segment_best_endpoint = best_endpoint
                                    if edge_segment_idx == edge_segment_count - 1:
                                        last_segment_best_endpoint = best_endpoint
                                    best_is_authored_endpoint = (best_endpoint == 1 and edge_segment_idx == 0) or (
                                        best_endpoint == 2 and edge_segment_idx == edge_segment_count - 1
                                    )
                                    emit_candidate = mesh_sdf_contact_segment_minimum_is_unique(
                                        best_endpoint, edge_segment_idx
                                    ) and (
                                        not best_is_authored_endpoint
                                        or mesh_sdf_contact_endpoint_owned(corner_ownership, best_endpoint)
                                    )
                                elif candidate_endpoint != 0 and emit_candidate:
                                    boundary_best_endpoint = (
                                        first_segment_best_endpoint
                                        if candidate_endpoint == 1
                                        else last_segment_best_endpoint
                                    )
                                    emit_candidate = (
                                        mesh_sdf_contact_endpoint_owned(corner_ownership, candidate_endpoint)
                                        and boundary_best_endpoint != candidate_endpoint
                                    )
                                    if emit_candidate:
                                        candidate_point_unscaled = v0 if candidate_endpoint == 1 else v1
                                        candidate_dist_unscaled = sample_sdf_at_t(
                                            texture_sdf,
                                            mesh_id_sdf,
                                            v0,
                                            v1 - v0,
                                            float(candidate_endpoint - 1),
                                            use_bvh_for_sdf,
                                            sdf_mesh_query_type,
                                            sdf_is_hfield,
                                            hfd_sdf,
                                            heightfield_elevations,
                                        )

                                dist_approx = candidate_dist_unscaled * min_sdf_scale
                                inner_cull_consistent = False
                                if emit_candidate:
                                    inner_cull_consistent = mesh_sdf_contact_passes_inner_cull_consistency(
                                        dist_approx,
                                        margin_sum,
                                        segment_midpoint_sdf,
                                        bsphere_center_inner,
                                        bsphere_radius_inner,
                                        sdf_aabb_lower,
                                        sdf_aabb_upper,
                                        min_sdf_scale,
                                        use_texture_sdf_for_search,
                                    )
                                if emit_candidate and dist_approx < contact_threshold and inner_cull_consistent:
                                    if wp.static(enable_heightfields):
                                        if sdf_is_hfield:
                                            candidate_dist_unscaled, direction_unscaled = sample_sdf_grad_heightfield(
                                                hfd_sdf, heightfield_elevations, candidate_point_unscaled
                                            )
                                        elif wp.static(not use_texture_sdf_only) and use_bvh_for_sdf:
                                            candidate_dist_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                                mesh_id_sdf,
                                                candidate_point_unscaled,
                                                _MESH_QUERY_MAX_DIST,
                                                sdf_mesh_query_type,
                                            )
                                        else:
                                            direction_unscaled = wp.static(sample_grad)(
                                                texture_sdf, candidate_point_unscaled
                                            )
                                    else:
                                        if wp.static(not use_texture_sdf_only) and use_bvh_for_sdf:
                                            candidate_dist_unscaled, direction_unscaled = sample_sdf_grad_using_mesh(
                                                mesh_id_sdf,
                                                candidate_point_unscaled,
                                                _MESH_QUERY_MAX_DIST,
                                                sdf_mesh_query_type,
                                            )
                                        else:
                                            direction_unscaled = wp.static(sample_grad)(
                                                texture_sdf, candidate_point_unscaled
                                            )

                                    if wp.static(use_identity_sdf_scale):
                                        dist = candidate_dist_unscaled
                                        direction = direction_unscaled
                                        point = candidate_point_unscaled
                                    else:
                                        dist, direction = scale_sdf_result_to_world(
                                            candidate_dist_unscaled,
                                            direction_unscaled,
                                            sdf_scale,
                                            inv_sdf_scale,
                                            min_sdf_scale,
                                        )
                                        point = wp.cw_mul(candidate_point_unscaled, sdf_scale)
                                    point_world = wp.transform_point(X_sdf_ws, point)

                                    direction_world = wp.transform_vector(X_sdf_ws, direction)
                                    direction_len_sq = wp.length_sq(direction_world)
                                    if direction_len_sq > 0.0:
                                        direction_world = direction_world * _sdf_rsqrt_rn(direction_len_sq)
                                    else:
                                        fallback_dir = point_world - wp.transform_get_translation(X_sdf_ws)
                                        fallback_len_sq = wp.length_sq(fallback_dir)
                                        if fallback_len_sq > 0.0:
                                            direction_world = fallback_dir * _sdf_rsqrt_rn(fallback_len_sq)
                                        else:
                                            direction_world = wp.vec3(0.0, 1.0, 0.0)

                                    contact_normal = -direction_world if mode == 0 else direction_world
                                    contact_center = point_world - 0.5 * dist * direction_world
                                    position_local_voxel = wp.quat_rotate_inv(
                                        wp.transform_get_rotation(X_voxel_ws),
                                        contact_center - wp.transform_get_translation(X_voxel_ws),
                                    )
                                    sort_sub_key = mesh_sdf_contact_sort_sub_key(
                                        my_edge_idx, edge_segment_idx, mode, candidate_endpoint
                                    )
                                    is_owned_endpoint = mesh_sdf_contact_is_owned_endpoint(
                                        candidate_endpoint, best_endpoint, edge_segment_idx, edge_segment_count
                                    )
                                    sort_sub_key = pack_contact_is_canonical_endpoint(sort_sub_key, is_owned_endpoint)
                                    retain_current_separation_guard = (
                                        candidate_endpoint == 0 and edge_segment_count > 1 and not is_owned_endpoint
                                    )
                                    is_speculative_shell = bool(False)
                                    swept_separation_lower_bound = float(1.0)
                                    if wp.static(speculative):
                                        clearance = dist - margin_sum
                                        if 0.0 < clearance and clearance <= wp.max(
                                            base_gap_sum, max_speculative_extension
                                        ):
                                            is_speculative_shell = True
                                            point_a = contact_center - 0.5 * dist * contact_normal
                                            point_b = contact_center + 0.5 * dist * contact_normal
                                            swept_separation_lower_bound = compute_contact_swept_separation_lower_bound(
                                                pair[0],
                                                pair[1],
                                                point_a,
                                                point_b,
                                                contact_normal,
                                                unpack_contact_normal_owner(sort_sub_key),
                                                margin_sum,
                                                shape_transform,
                                                shape_linear_velocity,
                                                shape_angular_velocity,
                                                shape_rotation_center_offset,
                                                collision_update_dt,
                                            )
                                    contact_id = export_and_reduce_contact_centered_two_spatial_depths(
                                        pair[0],
                                        pair[1],
                                        contact_center,
                                        contact_normal,
                                        dist,
                                        sort_sub_key,
                                        contact_center - midpoint,
                                        inner_spatial_depth,
                                        outer_spatial_depth,
                                        is_speculative_shell,
                                        is_owned_endpoint,
                                        swept_separation_lower_bound,
                                        position_local_voxel,
                                        aabb_lower_voxel,
                                        aabb_upper_voxel,
                                        voxel_res,
                                        reducer_data,
                                    )
                                    if wp.static(speculative):
                                        export_and_reduce_predictive_contact(
                                            pair[0],
                                            pair[1],
                                            contact_center,
                                            contact_normal,
                                            dist,
                                            margin_sum,
                                            base_gap_sum,
                                            0.0,
                                            0.0,
                                            sort_sub_key,
                                            shape_transform,
                                            shape_linear_velocity,
                                            shape_angular_velocity,
                                            shape_rotation_center_offset,
                                            collision_update_dt,
                                            max_speculative_extension,
                                            contact_id,
                                            retain_current_separation_guard,
                                            reducer_data,
                                        )

                    # Defensive cooperative reset before the next outer
                    # iteration — see the matching ``tile_stack_clear``
                    # call in ``mesh_sdf_collision_kernel`` for
                    # rationale.
                    wp.tile_stack_clear(edge_stack)

    return mesh_sdf_collision_global_reduce_kernel


def create_mesh_sdf_two_stage_kernels(
    writer_func: Any,
    speculative: bool = False,
    sdf_texture_paired_samples: bool = True,
):
    """Create texture-SDF cull and solve kernels for global contact reduction."""
    if sdf_texture_paired_samples:
        sample_sdf = _texture_sample_sdf_hw_paired
        sample_pair = _texture_sample_sdf_hw_pair_paired
        sample_clamped = _texture_sample_sdf_hw_clamped_paired
        sample_grad = _texture_sample_sdf_grad_only_hw_paired
    else:
        sample_sdf = _texture_sample_sdf_hw_scalar
        sample_pair = _texture_sample_sdf_hw_pair_scalar
        sample_clamped = _texture_sample_sdf_hw_clamped_scalar
        sample_grad = _texture_sample_sdf_grad_only_hw_scalar
    do_edge_sdf_collision, sample_sdf_at_t = _create_sdf_contact_funcs(False, True, sample_sdf, sample_pair)
    get_mesh_edge = _create_mesh_edge_accessor_func(True)
    get_mesh_edge_bounding_sphere = _create_get_mesh_edge_bounding_sphere_func(True)
    module = f"sdf_contact_two_stage_{writer_func.__name__}_{speculative}_{sdf_texture_paired_samples}"

    @wp.kernel(enable_backward=False, launch_bounds=(256, 2), module=module)
    def mesh_sdf_cull_kernel(
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        texture_sdf_table: wp.array[TextureSDFData],
        shape_sdf_index: wp.array[wp.int32],
        shape_gap: wp.array[float],
        shape_base_gap: wp.array[float],
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_mesh_count: wp.array[int],
        mesh_edge_indices: wp.array[wp.vec2i],
        mesh_edge_centers: wp.array[wp.vec4],
        shape_edge_range: wp.array[wp.vec2i],
        block_offsets: wp.array[wp.int32],
        search_contexts: wp.array[MeshSDFSearchContext],
        export_contexts: wp.array[MeshSDFExportContext],
        work_ints: wp.array[wp.int32],
        work_floats: wp.array[wp.float32],
        work_state: wp.array[wp.int32],
        work_segment_capacity: int,
        total_num_blocks: int,
    ):
        block_id, t = wp.tid()
        pair_count = wp.min(shape_pairs_mesh_mesh_count[0], shape_pairs_mesh_mesh.shape[0])
        total_combos = block_offsets[pair_count]
        edge_stack = wp.tile_stack(capacity=STACK_CAPACITY, dtype=EdgeCullResult)
        cull_context = wp.tile_empty(shape=1, dtype=MeshSDFCullContext, storage="shared")
        progress = wp.tile_zeros(shape=1, dtype=int, storage="shared")
        segment_slot = wp.tile_zeros(shape=1, dtype=int, storage="shared")

        for combo_idx in range(block_id, total_combos, total_num_blocks):
            mode = int(0)
            while mode < 2:
                context = MeshSDFCullContext()
                if t == 0:
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
                    pair = shape_pairs_mesh_mesh[pair_idx]
                    tri_shape = pair[mode]
                    sdf_shape = pair[1 - mode]
                    scale_data_tri = shape_data[tri_shape]
                    scale_data_sdf = shape_data[sdf_shape]
                    margin_sum = scale_data_tri[3] + scale_data_sdf[3]
                    gap_sum = shape_gap[pair[0]] + shape_gap[pair[1]]
                    base_gap_sum = shape_base_gap[pair[0]] + shape_base_gap[pair[1]]
                    tri_transform = shape_transform[tri_shape]
                    sdf_transform = shape_transform[sdf_shape]
                    sdf_index = shape_sdf_index[sdf_shape]
                    texture_sdf = texture_sdf_table[sdf_index]
                    context.context_id = 2 * pair_idx + mode
                    context.block_in_pair = combo_idx - pair_block_start
                    context.blocks_for_pair = block_offsets[pair_idx + 1] - pair_block_start
                    context.sdf_index = sdf_index
                    context.edge_range = shape_edge_range[tri_shape]
                    context.mesh_to_sdf = wp.transform_multiply(wp.transform_inverse(sdf_transform), tri_transform)
                    context.contact_threshold = gap_sum + margin_sum
                    search_precision = mesh_sdf_contact_search_precision(
                        margin_sum, 1.0, texture_sdf.voxel_radius, True
                    )
                    if context.block_in_pair == 0:
                        search = MeshSDFSearchContext()
                        search.sdf_index = sdf_index
                        search.edge_range = context.edge_range
                        search.mesh_to_sdf = context.mesh_to_sdf
                        search.contact_threshold = context.contact_threshold
                        search.search_precision = search_precision
                        search.margin_sum = margin_sum
                        export = MeshSDFExportContext()
                        export.inner_spatial_depth = margin_sum + wp.min(texture_sdf.voxel_radius, base_gap_sum)
                        export.outer_spatial_depth = margin_sum + gap_sum
                        if wp.static(speculative):
                            export.outer_spatial_depth = margin_sum + base_gap_sum
                        search_contexts[context.context_id] = search
                        export_contexts[context.context_id] = export
                wp.tile_scatter_masked(cull_context, 0, context, t == 0)
                context = wp.tile_extract(cull_context, 0)

                texture_sdf = texture_sdf_table[context.sdf_index]
                X_mesh_to_sdf = context.mesh_to_sdf
                contact_threshold = context.contact_threshold
                edge_range_tri = context.edge_range
                num_edges = edge_range_tri[1]
                chunk_size = (num_edges + context.blocks_for_pair - 1) // context.blocks_for_pair
                edge_start = context.block_in_pair * chunk_size
                edge_end = wp.min(edge_start + chunk_size, num_edges)
                wp.tile_scatter_masked(progress, 0, edge_start, t == 0)

                while wp.tile_extract(progress, 0) < edge_end or wp.tile_stack_count(edge_stack) > 0:
                    while (
                        wp.tile_extract(progress, 0) < edge_end and wp.tile_stack_count(edge_stack) < MESH_SDF_BLOCK_DIM
                    ):
                        base_edge_idx = wp.tile_extract(progress, 0)
                        edge_idx = base_edge_idx + t
                        add_edge = False
                        midpoint_sdf = float(0.0)
                        if edge_idx < edge_end:
                            center, radius = get_mesh_edge_bounding_sphere(
                                wp.uint64(0),
                                mesh_edge_indices,
                                mesh_edge_centers,
                                edge_range_tri,
                                wp.vec3(1.0, 1.0, 1.0),
                                X_mesh_to_sdf,
                                wp.vec3(1.0, 1.0, 1.0),
                                1.0,
                                edge_idx,
                            )
                            culling_radius = radius + contact_threshold
                            clamped = wp.min(wp.max(center, texture_sdf.sdf_box_lower), texture_sdf.sdf_box_upper)
                            aabb_dist_sq = wp.length_sq(center - clamped)
                            if aabb_dist_sq <= culling_radius * culling_radius:
                                diff_mag = float(0.0)
                                if aabb_dist_sq > 0.0:
                                    diff_mag = wp.sqrt(aabb_dist_sq)
                                midpoint_sdf = wp.static(sample_clamped)(texture_sdf, clamped, diff_mag)
                                add_edge = midpoint_sdf <= culling_radius

                        cull_result = EdgeCullResult()
                        cull_result.edge_idx = edge_idx
                        cull_result.midpoint_sdf = midpoint_sdf
                        wp.tile_stack_push(edge_stack, cull_result, add_edge)
                        wp.tile_scatter_masked(progress, 0, base_edge_idx + wp.block_dim(), t == 0)

                    stack_count = wp.tile_stack_count(edge_stack)
                    fill = wp.min(stack_count, MESH_SDF_BLOCK_DIM)
                    if fill > 0:
                        segment = int(0)
                        if t == 0:
                            segment = wp.atomic_add(work_state, _SDF_WORK_SEGMENT_COUNT, 1)
                        wp.tile_scatter_masked(segment_slot, 0, segment, t == 0)
                        segment = wp.tile_extract(segment_slot, 0)
                        if t == 0:
                            if segment >= work_segment_capacity:
                                wp.atomic_max(work_state, _SDF_WORK_OVERFLOWED, 1)
                        popped, edge_slot = wp.tile_stack_pop(edge_stack)
                        if segment < work_segment_capacity:
                            base = segment * SDF_WORK_SEGMENT_STRIDE_INT32
                            if t == 0:
                                work_ints[base] = context.context_id
                                work_ints[base + 1] = fill
                            if edge_slot >= 0:
                                item = base + _SDF_WORK_SEGMENT_HEADER_INT32 + 2 * (edge_slot - (stack_count - fill))
                                work_ints[item] = popped.edge_idx
                                work_floats[item + 1] = popped.midpoint_sdf

                wp.tile_stack_clear(edge_stack)
                mode += 1

    @wp.kernel(enable_backward=False, launch_bounds=(256, 3), module=module)
    def mesh_sdf_solve_kernel(
        shape_transform: wp.array[wp.transform],
        texture_sdf_table: wp.array[TextureSDFData],
        shape_linear_velocity: wp.array[wp.vec3],
        shape_angular_velocity: wp.array[wp.vec3],
        shape_rotation_center_offset: wp.array[wp.vec3],
        shape_base_gap: wp.array[float],
        collision_update_dt: float,
        max_speculative_extension: float,
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_voxel_resolution: wp.array[wp.vec3i],
        shape_pairs_mesh_mesh: wp.array[wp.vec2i],
        mesh_edge_indices: wp.array[wp.vec2i],
        mesh_edge_centers: wp.array[wp.vec4],
        mesh_edge_halves: wp.array[wp.vec4],
        heightfield_elevations: wp.array[wp.float32],
        reducer_data: GlobalContactReducerData,
        search_contexts: wp.array[MeshSDFSearchContext],
        export_contexts: wp.array[MeshSDFExportContext],
        work_ints: wp.array[wp.int32],
        work_floats: wp.array[wp.float32],
        work_state: wp.array[wp.int32],
        work_segment_capacity: int,
        total_num_blocks: int,
    ):
        block_id, t = wp.tid()
        if work_state[_SDF_WORK_OVERFLOWED] != 0:
            return
        segment_count = wp.min(work_state[_SDF_WORK_SEGMENT_COUNT], work_segment_capacity)
        solve_context = wp.tile_empty(shape=1, dtype=MeshSDFSearchContext, storage="shared")

        for segment in range(block_id, segment_count, total_num_blocks):
            base = segment * SDF_WORK_SEGMENT_STRIDE_INT32
            context_id = work_ints[base]
            context = MeshSDFSearchContext()
            if t == 0:
                context = search_contexts[context_id]
            wp.tile_scatter_masked(solve_context, 0, context, t == 0)
            context = wp.tile_extract(solve_context, 0)
            count = work_ints[base + 1]
            if t < count:
                texture_sdf = texture_sdf_table[context.sdf_index]
                item = base + _SDF_WORK_SEGMENT_HEADER_INT32 + 2 * t
                edge_idx = work_ints[item]
                cached_sdf_val = work_floats[item + 1]
                v0, v1, corner_ownership = get_mesh_edge(
                    wp.uint64(0),
                    mesh_edge_indices,
                    mesh_edge_centers,
                    mesh_edge_halves,
                    context.edge_range,
                    wp.vec3(1.0, 1.0, 1.0),
                    context.mesh_to_sdf,
                    edge_idx,
                )
                result_context = wp.tile_extract(solve_context, 0)
                export = export_contexts[context_id]
                pair = shape_pairs_mesh_mesh[context_id >> 1]
                base_gap_sum = shape_base_gap[pair[0]] + shape_base_gap[pair[1]]
                mode = context_id & 1
                tri_shape = pair[mode]
                sdf_shape = pair[1 - mode]
                tri_transform = shape_transform[tri_shape]
                sdf_transform = shape_transform[sdf_shape]
                voxel_shape = mesh_sdf_contact_voxel_owner(
                    pair[0],
                    pair[1],
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_voxel_resolution,
                )
                voxel_transform = shape_transform[voxel_shape]
                aabb_lower_voxel = shape_collision_aabb_lower[voxel_shape]
                aabb_upper_voxel = shape_collision_aabb_upper[voxel_shape]
                voxel_res = shape_voxel_resolution[voxel_shape]
                pair_midpoint = (
                    wp.transform_get_translation(tri_transform) + wp.transform_get_translation(sdf_transform)
                ) * 0.5
                edge_segment_count = mesh_sdf_contact_segment_count(wp.length(v1 - v0), texture_sdf.voxel_radius, True)

                # Each voxel-scale interval owns one interior minimum. Authored
                # endpoints are considered only by the first/last interval, so
                # segmentation does not multiply vertex candidates.
                for edge_segment_idx in range(edge_segment_count):
                    segment_v0, segment_v1 = mesh_sdf_contact_segment_bounds(
                        v0, v1, edge_segment_idx, edge_segment_count
                    )
                    center, radius = get_edge_bounding_sphere(segment_v0, segment_v1)
                    segment_midpoint_sdf = cached_sdf_val
                    if edge_segment_count != 1 and 2 * edge_segment_idx + 1 != edge_segment_count:
                        segment_midpoint_sdf = sample_sdf_at_t(
                            texture_sdf,
                            wp.uint64(0),
                            segment_v0,
                            segment_v1 - segment_v0,
                            0.5,
                            False,
                            0,
                            False,
                            HeightfieldData(),
                            heightfield_elevations,
                        )
                    if segment_midpoint_sdf <= radius + result_context.contact_threshold:
                        dist, point, best_endpoint = do_edge_sdf_collision(
                            texture_sdf,
                            wp.uint64(0),
                            segment_v0,
                            segment_v1,
                            segment_midpoint_sdf,
                            False,
                            0,
                            False,
                            HeightfieldData(),
                            heightfield_elevations,
                            context.search_precision,
                        )

                        for contact_feature in range(3):
                            candidate_endpoint = int(contact_feature)
                            candidate_dist = dist
                            candidate_point = point
                            emit_candidate = True
                            if candidate_endpoint == 0:
                                best_is_authored_endpoint = (best_endpoint == 1 and edge_segment_idx == 0) or (
                                    best_endpoint == 2 and edge_segment_idx == edge_segment_count - 1
                                )
                                emit_candidate = mesh_sdf_contact_segment_minimum_is_unique(
                                    best_endpoint, edge_segment_idx
                                ) and (
                                    not best_is_authored_endpoint
                                    or mesh_sdf_contact_endpoint_owned(corner_ownership, best_endpoint)
                                )
                            else:
                                owns_segment_endpoint = (candidate_endpoint == 1 and edge_segment_idx == 0) or (
                                    candidate_endpoint == 2 and edge_segment_idx == edge_segment_count - 1
                                )
                                emit_candidate = (
                                    owns_segment_endpoint
                                    and mesh_sdf_contact_endpoint_owned(corner_ownership, candidate_endpoint)
                                    and best_endpoint != candidate_endpoint
                                )
                                if emit_candidate:
                                    candidate_point = v0 if candidate_endpoint == 1 else v1
                                    candidate_dist = sample_sdf_at_t(
                                        texture_sdf,
                                        wp.uint64(0),
                                        v0,
                                        v1 - v0,
                                        float(candidate_endpoint - 1),
                                        False,
                                        0,
                                        False,
                                        HeightfieldData(),
                                        heightfield_elevations,
                                    )

                            inner_cull_consistent = False
                            if emit_candidate:
                                inner_cull_consistent = mesh_sdf_contact_passes_inner_cull_consistency(
                                    candidate_dist,
                                    result_context.margin_sum,
                                    segment_midpoint_sdf,
                                    center,
                                    radius,
                                    texture_sdf.sdf_box_lower,
                                    texture_sdf.sdf_box_upper,
                                    1.0,
                                    True,
                                )
                            if (
                                emit_candidate
                                and candidate_dist < result_context.contact_threshold
                                and inner_cull_consistent
                            ):
                                direction = wp.static(sample_grad)(texture_sdf, candidate_point)
                                point_world = wp.transform_point(sdf_transform, candidate_point)
                                direction_world = wp.transform_vector(sdf_transform, direction)
                                direction_len_sq = wp.length_sq(direction_world)
                                if direction_len_sq > 0.0:
                                    direction_world = direction_world * _sdf_rsqrt_rn(direction_len_sq)
                                else:
                                    fallback_dir = point_world - wp.transform_get_translation(sdf_transform)
                                    fallback_len_sq = wp.length_sq(fallback_dir)
                                    if fallback_len_sq > 0.0:
                                        direction_world = fallback_dir * _sdf_rsqrt_rn(fallback_len_sq)
                                    else:
                                        direction_world = wp.vec3(0.0, 1.0, 0.0)

                                contact_normal = -direction_world if mode == 0 else direction_world
                                contact_center = point_world - 0.5 * candidate_dist * direction_world
                                position_local_voxel = wp.quat_rotate_inv(
                                    wp.transform_get_rotation(voxel_transform),
                                    contact_center - wp.transform_get_translation(voxel_transform),
                                )
                                sort_sub_key = mesh_sdf_contact_sort_sub_key(
                                    edge_idx, edge_segment_idx, mode, candidate_endpoint
                                )
                                is_owned_endpoint = mesh_sdf_contact_is_owned_endpoint(
                                    candidate_endpoint, best_endpoint, edge_segment_idx, edge_segment_count
                                )
                                sort_sub_key = pack_contact_is_canonical_endpoint(sort_sub_key, is_owned_endpoint)
                                retain_current_separation_guard = (
                                    candidate_endpoint == 0 and edge_segment_count > 1 and not is_owned_endpoint
                                )
                                is_speculative_shell = bool(False)
                                swept_separation_lower_bound = float(1.0)
                                if wp.static(speculative):
                                    clearance = candidate_dist - result_context.margin_sum
                                    if 0.0 < clearance and clearance <= wp.max(base_gap_sum, max_speculative_extension):
                                        is_speculative_shell = True
                                        point_a = contact_center - 0.5 * candidate_dist * contact_normal
                                        point_b = contact_center + 0.5 * candidate_dist * contact_normal
                                        swept_separation_lower_bound = compute_contact_swept_separation_lower_bound(
                                            pair[0],
                                            pair[1],
                                            point_a,
                                            point_b,
                                            contact_normal,
                                            unpack_contact_normal_owner(sort_sub_key),
                                            result_context.margin_sum,
                                            shape_transform,
                                            shape_linear_velocity,
                                            shape_angular_velocity,
                                            shape_rotation_center_offset,
                                            collision_update_dt,
                                        )
                                contact_id = export_and_reduce_contact_centered_two_spatial_depths(
                                    pair[0],
                                    pair[1],
                                    contact_center,
                                    contact_normal,
                                    candidate_dist,
                                    sort_sub_key,
                                    contact_center - pair_midpoint,
                                    export.inner_spatial_depth,
                                    export.outer_spatial_depth,
                                    is_speculative_shell,
                                    is_owned_endpoint,
                                    swept_separation_lower_bound,
                                    position_local_voxel,
                                    aabb_lower_voxel,
                                    aabb_upper_voxel,
                                    voxel_res,
                                    reducer_data,
                                )
                                if wp.static(speculative):
                                    export_and_reduce_predictive_contact(
                                        pair[0],
                                        pair[1],
                                        contact_center,
                                        contact_normal,
                                        candidate_dist,
                                        result_context.margin_sum,
                                        base_gap_sum,
                                        0.0,
                                        0.0,
                                        sort_sub_key,
                                        shape_transform,
                                        shape_linear_velocity,
                                        shape_angular_velocity,
                                        shape_rotation_center_offset,
                                        collision_update_dt,
                                        max_speculative_extension,
                                        contact_id,
                                        retain_current_separation_guard,
                                        reducer_data,
                                    )

    return mesh_sdf_cull_kernel, mesh_sdf_solve_kernel

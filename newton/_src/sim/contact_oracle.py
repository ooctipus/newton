# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private collision-pipeline interface for solver-side contact separation queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import warp as wp

from ..geometry.broad_phase_common import BODY_FLAG_KINEMATIC
from ..geometry.broad_phase_nxn import BroadPhaseAllPairs, BroadPhaseExplicit
from ..geometry.broad_phase_sap import BroadPhaseSAP
from ..geometry.sdf_contact import (
    MESH_SDF_ORACLE_BLOCK_DIM,
    _CONSTANT_TWIST_ANGLE_TOLERANCE,
    _angular_path_displacement_bound,
    _mesh_sdf_nonpenetration_oracle_owns_pair,
    _rotation_angle_between,
    _rotation_displacement_bound,
    create_mesh_sdf_nonpenetration_oracle_kernels,
)
from ..geometry.types import GeoType
from .contacts import Contacts, _increment_contact_generation
from .enums import JointType

if TYPE_CHECKING:
    from .model import Model

CONTACT_ORACLE_VIOLATION = 1
"""At least one queried physical surface separation is negative."""

CONTACT_ORACLE_INCOMPLETE = 2
"""The query could not prove complete geometry coverage."""

CONTACT_ORACLE_CAPACITY = 4
"""A fixed-capacity query buffer overflowed."""

CONTACT_ORACLE_REFERENCE_INFEASIBLE = 8
"""An oracle-owned pair overlaps at the solver composition pose."""

CONTACT_ORACLE_SWEEP_INCOMPLETE = 16
"""The endpoint geometry is valid, but the complete swept path was not certified."""

_STRICT_ORACLE_GUARDS_PER_WORLD = 32
_STRICT_ORACLE_EMPTY_FEATURE_KEY = 0xFFFFFFFFFFFFFFFF

RIGID_BODY_PATH_UNKNOWN = 0
"""No sound path bound is available for this body."""

RIGID_BODY_PATH_STATIONARY = 1
"""The body pose is constant over the complete integration interval."""

RIGID_BODY_PATH_LINEAR_TRANSLATION = 2
"""The body follows the endpoint translation segment without rotating."""

RIGID_BODY_PATH_BOUNDED = 3
"""The body has sound origin- and angular-path bounds but no exact sweep."""

RIGID_BODY_PATH_CONSTANT_TWIST = 4
"""The body follows the shortest constant-twist path between its endpoints."""


def _mesh_sdf_nonpenetration_shape_capabilities(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable movable and anchor capability masks for raw meshes."""
    shape_type = model.shape_type.numpy()
    shape_body = model.shape_body.numpy()
    raw_mesh = shape_type == int(GeoType.MESH)

    joint_type = model.joint_type.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    direct_free_joint = (joint_type == int(JointType.FREE)) & (joint_parent == -1)
    free_child = joint_child[direct_free_joint]
    free_child = free_child[(free_child >= 0) & (free_child < model.body_count)]
    certifiable_body = np.zeros(model.body_count, dtype=bool)
    certifiable_body[free_child] = True

    movable_capable = np.zeros(model.shape_count, dtype=bool)
    attached = shape_body >= 0
    movable_capable[attached] = certifiable_body[shape_body[attached]]
    return raw_mesh & movable_capable, raw_mesh


@dataclass(frozen=True)
class RigidBodyPathCertificate:
    """Solver-owned bounds for one candidate rigid-body path.

    The collision oracle consumes these arrays synchronously and never owns or
    mutates them. Bounds describe the complete integration path rather than the
    shortest transform difference between its endpoints. A
    ``RIGID_BODY_PATH_CONSTANT_TWIST`` value additionally certifies that the
    origin follows its endpoint chord and the orientation follows the shortest
    constant-angular-velocity arc between its endpoints.

    Attributes:
        endpoint_body_q: Candidate end-of-substep body poses, shape
            ``[body_count]``.
        origin_path_length: Upper bound on body-origin path length [m], shape
            ``[body_count]``.
        angular_path_length: Upper bound on orientation total variation [rad],
            shape ``[body_count]``.
        motion_kind: One of the ``RIGID_BODY_PATH_*`` values, shape
            ``[body_count]``.
    """

    endpoint_body_q: wp.array
    origin_path_length: wp.array
    angular_path_length: wp.array
    motion_kind: wp.array


@wp.func
def _shape_aabb_at_pose(
    body_q: wp.array[wp.transform],
    shape: int,
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
) -> tuple[bool, wp.transform, wp.vec3, wp.vec3]:
    """Return one finite world-space shape AABB at a body pose."""
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = wp.transform_multiply(body_q[body], X_ws)
    position = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)
    local_lower = shape_collision_aabb_lower[shape]
    local_upper = shape_collision_aabb_upper[shape]
    valid = (
        wp.isfinite(position[0])
        and wp.isfinite(position[1])
        and wp.isfinite(position[2])
        and wp.isfinite(orientation[0])
        and wp.isfinite(orientation[1])
        and wp.isfinite(orientation[2])
        and wp.isfinite(orientation[3])
        and wp.isfinite(local_lower[0])
        and wp.isfinite(local_lower[1])
        and wp.isfinite(local_lower[2])
        and wp.isfinite(local_upper[0])
        and wp.isfinite(local_upper[1])
        and wp.isfinite(local_upper[2])
    )
    if not valid:
        return False, wp.transform_identity(), wp.vec3(1.0e30), wp.vec3(-1.0e30)

    center = 0.5 * (local_lower + local_upper)
    half = 0.5 * (local_upper - local_lower)
    world_center = wp.quat_rotate(orientation, center) + position
    r0 = wp.quat_rotate(orientation, wp.vec3(1.0, 0.0, 0.0))
    r1 = wp.quat_rotate(orientation, wp.vec3(0.0, 1.0, 0.0))
    r2 = wp.quat_rotate(orientation, wp.vec3(0.0, 0.0, 1.0))
    world_half = wp.vec3(
        wp.abs(r0[0]) * half[0] + wp.abs(r1[0]) * half[1] + wp.abs(r2[0]) * half[2],
        wp.abs(r0[1]) * half[0] + wp.abs(r1[1]) * half[1] + wp.abs(r2[1]) * half[2],
        wp.abs(r0[2]) * half[0] + wp.abs(r1[2]) * half[1] + wp.abs(r2[2]) * half[2],
    )
    margin = wp.vec3(shape_margin[shape])
    return True, X_ws, world_center - world_half - margin, world_center + world_half + margin


@wp.kernel(enable_backward=False)
def _compute_swept_mesh_aabbs(
    reference_body_q: wp.array[wp.transform],
    candidate_body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    endpoint_domain_shape: wp.array[wp.uint8],
    movable_capable_shape: wp.array[wp.uint8],
    body_flags: wp.array[int],
    shape_world: wp.array[int],
    body_origin_path_length: wp.array[float],
    body_angular_path_length: wp.array[float],
    body_path_kind: wp.array[wp.uint8],
    separation_tolerance: float,
    incomplete_bit: int,
    sweep_incomplete_bit: int,
    reference_shape_transform: wp.array[wp.transform],
    candidate_shape_transform: wp.array[wp.transform],
    swept_aabb_lower: wp.array[wp.vec3],
    swept_aabb_upper: wp.array[wp.vec3],
    shape_motion_radius: wp.array[float],
    shape_effective_angular_path_length: wp.array[float],
    world_status: wp.array[wp.int32],
    query_incomplete: wp.array[int],
):
    """Build reference-to-candidate routing AABBs for endpoint-bearing meshes."""
    shape = wp.tid()
    body = shape_body[shape]
    immovable = body < 0
    if body >= 0:
        immovable = (body_flags[body] & BODY_FLAG_KINEMATIC) != 0
    active = immovable or movable_capable_shape[shape] != wp.uint8(0)
    if endpoint_domain_shape[shape] == wp.uint8(0) or not active:
        reference_shape_transform[shape] = wp.transform_identity()
        candidate_shape_transform[shape] = wp.transform_identity()
        swept_aabb_lower[shape] = wp.vec3(1.0e30)
        swept_aabb_upper[shape] = wp.vec3(-1.0e30)
        shape_motion_radius[shape] = 0.0
        shape_effective_angular_path_length[shape] = 0.0
        return

    reference_valid, X_reference_ws, reference_lower, reference_upper = _shape_aabb_at_pose(
        reference_body_q,
        shape,
        shape_transform,
        shape_body,
        shape_margin,
        shape_collision_aabb_lower,
        shape_collision_aabb_upper,
    )
    candidate_valid, X_candidate_ws, candidate_lower, candidate_upper = _shape_aabb_at_pose(
        candidate_body_q,
        shape,
        shape_transform,
        shape_body,
        shape_margin,
        shape_collision_aabb_lower,
        shape_collision_aabb_upper,
    )
    if not reference_valid or not candidate_valid:
        world_id = shape_world[shape]
        if world_id >= 0 and world_id < world_status.shape[0]:
            wp.atomic_or(world_status, world_id, incomplete_bit)
        elif world_id < 0 and world_status.shape[0] == 1:
            wp.atomic_or(world_status, 0, incomplete_bit)
        else:
            wp.atomic_or(query_incomplete, 0, incomplete_bit)
        reference_shape_transform[shape] = wp.transform_identity()
        candidate_shape_transform[shape] = wp.transform_identity()
        swept_aabb_lower[shape] = wp.vec3(1.0e30)
        swept_aabb_upper[shape] = wp.vec3(-1.0e30)
        shape_motion_radius[shape] = 0.0
        shape_effective_angular_path_length[shape] = 0.0
        return

    local_lower = shape_collision_aabb_lower[shape]
    local_upper = shape_collision_aabb_upper[shape]
    local_radius = wp.length(wp.max(wp.abs(local_lower), wp.abs(local_upper))) + shape_margin[shape]
    if not wp.isfinite(local_radius) or local_radius < 0.0:
        world_id = shape_world[shape]
        if world_id >= 0 and world_id < world_status.shape[0]:
            wp.atomic_or(world_status, world_id, incomplete_bit)
        elif world_id < 0 and world_status.shape[0] == 1:
            wp.atomic_or(world_status, 0, incomplete_bit)
        else:
            wp.atomic_or(query_incomplete, 0, incomplete_bit)
        shape_motion_radius[shape] = 0.0
        shape_effective_angular_path_length[shape] = 0.0
        return

    reference_shape_transform[shape] = X_reference_ws
    candidate_shape_transform[shape] = X_candidate_ws
    swept_aabb_lower[shape] = wp.min(reference_lower, candidate_lower)
    swept_aabb_upper[shape] = wp.max(reference_upper, candidate_upper)
    shape_motion_radius[shape] = local_radius
    shape_effective_angular_path_length[shape] = 0.0

    if body >= 0:
        reference_body_transform = reference_body_q[body]
        candidate_body_transform = candidate_body_q[body]
        local_center = 0.5 * (local_lower + local_upper)
        local_half = 0.5 * (local_upper - local_lower)
        X_bs = shape_transform[shape]
        shape_rotation = wp.transform_get_rotation(X_bs)
        center_body = wp.transform_point(X_bs, local_center)
        r0 = wp.quat_rotate(shape_rotation, wp.vec3(1.0, 0.0, 0.0))
        r1 = wp.quat_rotate(shape_rotation, wp.vec3(0.0, 1.0, 0.0))
        r2 = wp.quat_rotate(shape_rotation, wp.vec3(0.0, 0.0, 1.0))
        half_body = wp.vec3(
            wp.abs(r0[0]) * local_half[0] + wp.abs(r1[0]) * local_half[1] + wp.abs(r2[0]) * local_half[2],
            wp.abs(r0[1]) * local_half[0] + wp.abs(r1[1]) * local_half[1] + wp.abs(r2[1]) * local_half[2],
            wp.abs(r0[2]) * local_half[0] + wp.abs(r1[2]) * local_half[1] + wp.abs(r2[2]) * local_half[2],
        )
        body_radius = wp.length(wp.abs(center_body) + half_body) + shape_margin[shape]
        if wp.isfinite(body_radius) and body_radius >= 0.0:
            shape_motion_radius[shape] = body_radius
        origin_path_length = body_origin_path_length[body]
        angular_path_length = body_angular_path_length[body]
        path_kind = int(body_path_kind[body])
        angular_valid, angular_displacement = _angular_path_displacement_bound(
            angular_path_length, body_radius
        )
        effective_angular_path_length = angular_path_length
        swept_angular_displacement = angular_displacement
        endpoint_rotation_valid, endpoint_rotation_displacement = _rotation_displacement_bound(
            wp.transform_get_rotation(reference_body_transform),
            wp.transform_get_rotation(candidate_body_transform),
            body_radius,
        )
        endpoint_angle_valid, endpoint_rotation_angle = _rotation_angle_between(
            wp.transform_get_rotation(reference_body_transform),
            wp.transform_get_rotation(candidate_body_transform),
        )
        reference_origin = wp.transform_get_translation(reference_body_transform)
        candidate_origin = wp.transform_get_translation(candidate_body_transform)
        endpoint_translation = wp.length(candidate_origin - reference_origin)
        valid_path = (
            wp.isfinite(origin_path_length)
            and origin_path_length >= 0.0
            and angular_valid
            and endpoint_rotation_valid
            and endpoint_angle_valid
            and wp.isfinite(endpoint_translation)
            and endpoint_translation <= origin_path_length + separation_tolerance
            and endpoint_rotation_displacement <= angular_displacement + separation_tolerance
            and path_kind >= RIGID_BODY_PATH_STATIONARY
            and path_kind <= RIGID_BODY_PATH_CONSTANT_TWIST
        )
        if path_kind == RIGID_BODY_PATH_STATIONARY:
            valid_path = valid_path and origin_path_length <= separation_tolerance
            valid_path = valid_path and angular_displacement <= separation_tolerance
        elif path_kind == RIGID_BODY_PATH_LINEAR_TRANSLATION:
            valid_path = valid_path and angular_displacement <= separation_tolerance
            valid_path = valid_path and origin_path_length <= endpoint_translation + separation_tolerance
        elif path_kind == RIGID_BODY_PATH_CONSTANT_TWIST:
            valid_path = valid_path and angular_path_length <= wp.pi
            valid_path = valid_path and origin_path_length <= endpoint_translation + separation_tolerance
            valid_path = valid_path and angular_displacement <= endpoint_rotation_displacement + separation_tolerance
            valid_path = valid_path and wp.abs(angular_path_length - endpoint_rotation_angle) <= (
                _CONSTANT_TWIST_ANGLE_TOLERANCE
            )
            effective_angular_path_length = wp.max(angular_path_length, endpoint_rotation_angle) + (
                _CONSTANT_TWIST_ANGLE_TOLERANCE
            )
            swept_angular_valid, swept_angular_displacement = _angular_path_displacement_bound(
                effective_angular_path_length, body_radius
            )
            valid_path = valid_path and swept_angular_valid

        shape_effective_angular_path_length[shape] = effective_angular_path_length

        if not valid_path:
            world_id = shape_world[shape]
            if world_id >= 0 and world_id < world_status.shape[0]:
                wp.atomic_or(world_status, world_id, sweep_incomplete_bit)
            elif world_id < 0 and world_status.shape[0] == 1:
                wp.atomic_or(world_status, 0, sweep_incomplete_bit)
            else:
                wp.atomic_or(query_incomplete, 0, sweep_incomplete_bit)
        else:
            path_displacement = origin_path_length + swept_angular_displacement
            swept_aabb_lower[shape] = wp.min(candidate_lower, reference_lower - wp.vec3(path_displacement))
            swept_aabb_upper[shape] = wp.max(candidate_upper, reference_upper + wp.vec3(path_displacement))


@wp.kernel(enable_backward=False)
def _route_candidate_mesh_pairs(
    candidate_pairs: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[int],
    endpoint_domain_shape: wp.array[wp.uint8],
    nonpenetration_oracle_eligible_shape: wp.array[wp.uint8],
    nonpenetration_oracle_anchor_shape: wp.array[wp.uint8],
    shape_source: wp.array[wp.uint64],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    shape_world: wp.array[int],
    endpoint_pairs: wp.array[wp.vec2i],
    endpoint_pair_count: wp.array[int],
    world_status: wp.array[wp.int32],
    query_incomplete: wp.array[int],
    incomplete_bit: int,
    capacity_bit: int,
    total_num_threads: int,
):
    """Compact candidate-pose pairs whose two shapes expose mesh vertices."""
    pair_count = wp.min(candidate_pair_count[0], candidate_pairs.shape[0])
    for pair_index in range(wp.tid(), pair_count, total_num_threads):
        pair = candidate_pairs[pair_index]
        shape_a = pair[0]
        shape_b = pair[1]
        if endpoint_domain_shape[shape_a] == wp.uint8(0) or endpoint_domain_shape[shape_b] == wp.uint8(0):
            continue
        if not _mesh_sdf_nonpenetration_oracle_owns_pair(
            shape_a,
            shape_b,
            nonpenetration_oracle_eligible_shape,
            nonpenetration_oracle_anchor_shape,
            shape_source,
            shape_body,
            body_flags,
        ):
            continue
        if shape_b < shape_a:
            swap = shape_a
            shape_a = shape_b
            shape_b = swap
            pair = wp.vec2i(shape_a, shape_b)
        world_a = shape_world[shape_a]
        world_b = shape_world[shape_b]
        world_id = world_a if world_a >= 0 else world_b
        if world_id < 0 and world_status.shape[0] == 1:
            world_id = 0
        if world_id < 0 or world_id >= world_status.shape[0] or (world_a >= 0 and world_b >= 0 and world_a != world_b):
            wp.atomic_or(query_incomplete, 0, incomplete_bit)
            continue
        output_index = wp.atomic_add(endpoint_pair_count, 0, 1)
        if output_index < endpoint_pairs.shape[0]:
            endpoint_pairs[output_index] = pair
        else:
            wp.atomic_or(world_status, world_id, incomplete_bit | capacity_bit)


@wp.kernel(enable_backward=False)
def _finalize_query_status(
    candidate_pair_count: wp.array[int],
    candidate_pair_max: int,
    query_incomplete: wp.array[int],
    world_status: wp.array[wp.int32],
    incomplete_bit: int,
    capacity_bit: int,
):
    """Fail every world closed when query coverage cannot be localized."""
    world = wp.tid()
    status = query_incomplete[0]
    if candidate_pair_count[0] > candidate_pair_max:
        status |= incomplete_bit | capacity_bit
    if status != 0:
        wp.atomic_or(world_status, world, status)


class RigidContactOracle(Protocol):
    """Topology-free solver boundary for a pipeline-owned rigid-contact oracle."""

    model_token: object
    """Opaque identity token for the model whose collision topology the oracle owns."""

    device: wp.Device
    """Device where query inputs and fixed-capacity outputs must reside."""

    body_count: int
    """Number of body poses required by each query input."""

    shape_count: int
    """Number of shapes in the collision topology owned by the oracle."""

    world_count: int
    """Number of independent worlds represented by :attr:`world_status`."""

    guard_contacts: Contacts
    """One compact cumulative guard set retained across correction rounds."""

    world_status: wp.array[wp.int32]
    """Per-world bit mask composed from the ``CONTACT_ORACLE_*`` constants."""

    certified_path_fraction: wp.array[wp.float32]
    """Largest certified-safe prefix of the candidate path per world."""

    @property
    def separation_tolerance(self) -> float:
        """Return the geometric tolerance used for acceptance and guard clearance [m]."""

    def begin_refinement(self) -> None:
        """Clear cumulative guards before one solver substep."""

    def refine(
        self,
        reference_body_q: wp.array[wp.transform],
        path: RigidBodyPathCertificate,
    ) -> None:
        """Evaluate candidate geometry and update cumulative guards.

        Args:
            reference_body_q: Body poses at the solver composition point.
            path: Solver-owned endpoint and bounds for the complete candidate
                path.
        """

    def scan(
        self,
        reference_body_q: wp.array[wp.transform],
        path: RigidBodyPathCertificate,
    ) -> None:
        """Evaluate candidate geometry without modifying cumulative guards."""


class _MeshSDFNonpenetrationOracle:
    """Pipeline-owned mesh/SDF nonpenetration oracle.

    Construction and geometry traversal stay private to the collision pipeline;
    solvers consume only :class:`RigidContactOracle`. The oracle checks source
    vertices against the opposite raw mesh and raw triangle edges against the
    opposite surface at the reference and candidate poses. It owns only pairs
    with one direct free movable body and one static or kinematic anchor, whose
    complete paths the solver certifies. Articulated and dynamic--dynamic mesh
    pairs retain collision-time endpoint guards. The oracle additionally
    certifies stationary-target, unchanged-orientation pure translations with a
    fixed-cost swept-bounds certificate followed by continuous triangle SAT.
    Bounded motion is certified when reference triangle separation exceeds the
    complete relative-displacement envelope; remaining ambiguity reports
    incomplete, as do transverse edge-only overlaps without a usable guard face.
    This is not a general continuous-collision detector. Other shape families
    retain their existing collision and strict-guard paths.
    """

    @staticmethod
    def supports_model(model: Model) -> bool:
        """Return whether a structurally eligible movable--anchor pair can exist."""
        # CONVEX_MESH keeps its convex/GJK routing and legacy strict-contact path;
        # this oracle owns only explicit MESH bodies. Runtime body flags select
        # which capable mesh is the movable and which is the anchor.
        movable_capable, anchor_capable = _mesh_sdf_nonpenetration_shape_capabilities(model)
        shape_body = model.shape_body.numpy()
        for movable_shape in np.flatnonzero(movable_capable):
            if np.any(anchor_capable & (shape_body != shape_body[movable_shape])):
                return True
        return False

    def __init__(
        self,
        *,
        model: Model,
        broad_phase: BroadPhaseAllPairs | BroadPhaseSAP | BroadPhaseExplicit,
        shape_pairs_filtered: wp.array[wp.vec2i] | None,
        shape_pairs_excluded: wp.array[wp.vec2i] | None,
        shape_pairs_excluded_count: int,
        include_static_kinematic_pairs: bool,
        candidate_pairs: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        endpoint_pairs: wp.array[wp.vec2i],
        endpoint_pair_count: wp.array[int],
        separation_tolerance: float = 1.0e-6,
    ):
        self.model_token = model
        self.device = model.device
        self.body_count = model.body_count
        self.shape_count = model.shape_count
        self.world_count = model.world_count
        self._guard_slots_per_world = _STRICT_ORACLE_GUARDS_PER_WORLD
        contact_max = model.world_count * self._guard_slots_per_world
        self.guard_contacts = Contacts(contact_max, 0, device=model.device)
        self.guard_contacts._velocity_speculation_active = True
        self.guard_contacts._strict_nonpenetration_active = True
        self._guard_feature_keys = wp.full(
            contact_max,
            _STRICT_ORACLE_EMPTY_FEATURE_KEY,
            dtype=wp.uint64,
            device=model.device,
        )
        self._guard_feature_contact_indices = wp.full(contact_max, -1, dtype=wp.int32, device=model.device)
        self._guard_feature_update_generations = wp.full(contact_max, -1, dtype=wp.int32, device=model.device)
        self._shape_key_bits = max(1, (model.shape_count - 1).bit_length())
        vertex_key_bits = 63 - 2 * self._shape_key_bits
        if vertex_key_bits <= 0:
            raise ValueError(
                "Strict mesh nonpenetration requires enough uint64 key space for two shape indices "
                f"({model.shape_count} shapes require {self._shape_key_bits} bits each)."
            )
        self._vertex_key_max = (1 << vertex_key_bits) - 1
        self.world_status = wp.zeros(model.world_count, dtype=wp.int32, device=model.device)
        self.certified_path_fraction = wp.ones(model.world_count, dtype=wp.float32, device=model.device)
        self.world_min_separation = wp.full(model.world_count, float("inf"), dtype=wp.float32, device=model.device)
        self._query_incomplete = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._reference_shape_transform = wp.zeros(model.shape_count, dtype=wp.transform, device=model.device)
        self._candidate_shape_transform = wp.zeros(model.shape_count, dtype=wp.transform, device=model.device)
        self._swept_aabb_lower = wp.zeros(model.shape_count, dtype=wp.vec3, device=model.device)
        self._swept_aabb_upper = wp.zeros(model.shape_count, dtype=wp.vec3, device=model.device)
        self._shape_motion_radius = wp.zeros(model.shape_count, dtype=wp.float32, device=model.device)
        self._shape_effective_angular_path_length = wp.zeros(
            model.shape_count, dtype=wp.float32, device=model.device
        )
        movable_capable, anchor_capable = _mesh_sdf_nonpenetration_shape_capabilities(model)
        self._endpoint_domain_shape = wp.array(
            anchor_capable.astype(np.uint8), dtype=wp.uint8, device=model.device
        )
        self._anchor_capable_shape = self._endpoint_domain_shape
        self._eligible_shape = wp.array(
            movable_capable.astype(np.uint8), dtype=wp.uint8, device=model.device
        )

        self._model = model
        self._broad_phase = broad_phase
        self._shape_pairs_filtered = shape_pairs_filtered
        self._shape_pairs_excluded = shape_pairs_excluded
        self._shape_pairs_excluded_count = shape_pairs_excluded_count
        self._include_static_kinematic_pairs = include_static_kinematic_pairs
        self._candidate_pairs = candidate_pairs
        self._candidate_pair_count = candidate_pair_count
        self._endpoint_pairs = endpoint_pairs
        self._endpoint_pair_count = endpoint_pair_count
        self._reference_pair_state = wp.zeros(endpoint_pairs.shape[0], dtype=wp.uint8, device=model.device)
        self._separation_tolerance = separation_tolerance
        self._pair_threads = max(1, min(candidate_pairs.shape[0], 65_536))
        self._endpoint_blocks = max(1, min(endpoint_pairs.shape[0], 65_536))
        self._sweep_kernel, self._scan_kernel = create_mesh_sdf_nonpenetration_oracle_kernels(False)
        _, self._materialize_kernel = create_mesh_sdf_nonpenetration_oracle_kernels(True)

    @property
    def separation_tolerance(self) -> float:
        """Return the geometric tolerance used for acceptance and guard clearance [m]."""
        return self._separation_tolerance

    def begin_refinement(self) -> None:
        """Clear cumulative feature ownership before one solver substep."""
        self.guard_contacts.clear()
        self.guard_contacts._velocity_speculation_active = True
        self.guard_contacts._strict_nonpenetration_active = True
        self._guard_feature_keys.fill_(_STRICT_ORACLE_EMPTY_FEATURE_KEY)
        self._guard_feature_contact_indices.fill_(-1)
        self._guard_feature_update_generations.fill_(-1)

    def refine(
        self,
        reference_body_q: wp.array[wp.transform],
        path: RigidBodyPathCertificate,
    ) -> None:
        """Append novel features and relinearize retained features in place."""
        wp.launch(
            kernel=_increment_contact_generation,
            dim=1,
            inputs=[self.guard_contacts.contact_generation],
            device=self.device,
            record_tape=False,
        )
        self._query(reference_body_q, path, self.guard_contacts, materialize=True)

    def scan(
        self,
        reference_body_q: wp.array[wp.transform],
        path: RigidBodyPathCertificate,
    ) -> None:
        """Validate candidate geometry without modifying retained guards."""
        self._query(reference_body_q, path, self.guard_contacts, materialize=False)

    def _query(
        self,
        reference_body_q: wp.array[wp.transform],
        path: RigidBodyPathCertificate,
        contacts: Contacts,
        *,
        materialize: bool,
    ) -> None:
        """Validate the reference pose, candidate pose, and supported swept paths."""
        candidate_body_q = path.endpoint_body_q
        if reference_body_q.shape != candidate_body_q.shape:
            raise ValueError("reference_body_q and candidate_body_q must have the same shape")
        if reference_body_q.device != self.device or candidate_body_q.device != self.device:
            raise ValueError("reference_body_q and candidate_body_q must be on the model device")
        if candidate_body_q.shape[0] != self.body_count:
            raise ValueError(f"candidate_body_q must have {self.body_count} entries, got {candidate_body_q.shape[0]}")
        path_arrays = (candidate_body_q, path.origin_path_length, path.angular_path_length, path.motion_kind)
        if any(array.device != self.device for array in path_arrays):
            raise ValueError("Rigid-body path certificate arrays must be on the model device")
        if any(array.shape != (self.body_count,) for array in path_arrays):
            raise ValueError(f"Rigid-body path certificate arrays must have shape ({self.body_count},)")
        if path.origin_path_length.dtype != wp.float32 or path.angular_path_length.dtype != wp.float32:
            raise ValueError("Rigid-body path length arrays must have dtype float32")
        if path.motion_kind.dtype != wp.uint8:
            raise ValueError("Rigid-body path motion_kind must have dtype uint8")
        self.world_status.zero_()
        self.certified_path_fraction.fill_(1.0)
        self.world_min_separation.fill_(float("inf"))
        self._query_incomplete.zero_()
        self._candidate_pair_count.zero_()
        self._endpoint_pair_count.zero_()

        model = self._model
        wp.launch(
            kernel=_compute_swept_mesh_aabbs,
            dim=model.shape_count,
            inputs=[
                reference_body_q,
                candidate_body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_margin,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                self._endpoint_domain_shape,
                self._eligible_shape,
                model.body_flags,
                model.shape_world,
                path.origin_path_length,
                path.angular_path_length,
                path.motion_kind,
                self._separation_tolerance,
                CONTACT_ORACLE_INCOMPLETE,
                CONTACT_ORACLE_SWEEP_INCOMPLETE,
            ],
            outputs=[
                self._reference_shape_transform,
                self._candidate_shape_transform,
                self._swept_aabb_lower,
                self._swept_aabb_upper,
                self._shape_motion_radius,
                self._shape_effective_angular_path_length,
                self.world_status,
                self._query_incomplete,
            ],
            device=model.device,
            record_tape=False,
        )
        if isinstance(self._broad_phase, (BroadPhaseAllPairs, BroadPhaseSAP)):
            launch_kwargs = {}
            if isinstance(self._broad_phase, BroadPhaseSAP):
                launch_kwargs["sort_axis_displacement_limit"] = None
            self._broad_phase.launch(
                self._swept_aabb_lower,
                self._swept_aabb_upper,
                None,
                model.shape_collision_group,
                model.shape_world,
                model.shape_count,
                self._candidate_pairs,
                self._candidate_pair_count,
                device=model.device,
                filter_pairs=self._shape_pairs_excluded,
                num_filter_pairs=self._shape_pairs_excluded_count,
                skip_count_zero=True,
                shape_body=model.shape_body,
                body_flags=model.body_flags,
                include_static_kinematic_pairs=self._include_static_kinematic_pairs,
                shape_displacement=None,
                **launch_kwargs,
            )
        else:
            self._broad_phase.launch(
                self._swept_aabb_lower,
                self._swept_aabb_upper,
                None,
                self._shape_pairs_filtered,
                len(self._shape_pairs_filtered),
                self._candidate_pairs,
                self._candidate_pair_count,
                device=model.device,
                skip_count_zero=True,
                shape_body=model.shape_body,
                body_flags=model.body_flags,
                include_static_kinematic_pairs=self._include_static_kinematic_pairs,
                shape_displacement=None,
            )

        wp.launch(
            kernel=_route_candidate_mesh_pairs,
            dim=self._pair_threads,
            inputs=[
                self._candidate_pairs,
                self._candidate_pair_count,
                self._endpoint_domain_shape,
                self._eligible_shape,
                self._anchor_capable_shape,
                model.shape_source_ptr,
                model.shape_body,
                model.body_flags,
                model.shape_world,
                self._endpoint_pairs,
                self._endpoint_pair_count,
                self.world_status,
                self._query_incomplete,
                CONTACT_ORACLE_INCOMPLETE,
                CONTACT_ORACLE_CAPACITY,
                self._pair_threads,
            ],
            device=model.device,
            record_tape=False,
        )
        wp.launch_tiled(
            kernel=self._scan_kernel,
            dim=self._endpoint_blocks,
            inputs=[
                reference_body_q,
                reference_body_q,
                self._reference_shape_transform,
                self._reference_shape_transform,
                model.shape_scale,
                model.shape_margin,
                model.shape_source_ptr,
                model.shape_body,
                model.shape_world,
                model._shape_mesh_properties,
                self._endpoint_pairs,
                self._endpoint_pair_count,
                self._reference_pair_state,
                self._separation_tolerance,
                True,
                CONTACT_ORACLE_VIOLATION,
                CONTACT_ORACLE_REFERENCE_INFEASIBLE,
                CONTACT_ORACLE_INCOMPLETE,
                CONTACT_ORACLE_CAPACITY,
                self.world_status,
                self.world_min_separation,
                self._query_incomplete,
                self._shape_key_bits,
                wp.uint64(self._vertex_key_max),
                self._guard_slots_per_world,
                self._guard_feature_keys,
                self._guard_feature_contact_indices,
                self._guard_feature_update_generations,
                contacts.contact_generation,
                contacts.rigid_contact_max,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point_id,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_normal_owner,
                contacts.rigid_contact_is_predictive,
                contacts.rigid_contact_is_strict_guard,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_tids,
                self._endpoint_blocks,
            ],
            block_dim=MESH_SDF_ORACLE_BLOCK_DIM,
            device=model.device,
            record_tape=False,
        )
        wp.launch_tiled(
            kernel=self._sweep_kernel,
            dim=self._endpoint_blocks,
            inputs=[
                reference_body_q,
                candidate_body_q,
                model.shape_transform,
                self._reference_shape_transform,
                self._candidate_shape_transform,
                model.shape_scale,
                model.shape_margin,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                self._shape_motion_radius,
                path.origin_path_length,
                self._shape_effective_angular_path_length,
                path.motion_kind,
                RIGID_BODY_PATH_STATIONARY,
                RIGID_BODY_PATH_LINEAR_TRANSLATION,
                RIGID_BODY_PATH_BOUNDED,
                RIGID_BODY_PATH_CONSTANT_TWIST,
                model.shape_source_ptr,
                model.shape_body,
                model.shape_world,
                self._endpoint_pairs,
                self._endpoint_pair_count,
                self._reference_pair_state,
                self._separation_tolerance,
                CONTACT_ORACLE_VIOLATION,
                CONTACT_ORACLE_INCOMPLETE,
                CONTACT_ORACLE_SWEEP_INCOMPLETE,
                self.world_status,
                self.certified_path_fraction,
                self._query_incomplete,
                self._endpoint_blocks,
            ],
            block_dim=MESH_SDF_ORACLE_BLOCK_DIM,
            device=model.device,
            record_tape=False,
        )
        candidate_kernel = self._materialize_kernel if materialize else self._scan_kernel
        wp.launch_tiled(
            kernel=candidate_kernel,
            dim=self._endpoint_blocks,
            inputs=[
                reference_body_q,
                candidate_body_q,
                self._reference_shape_transform,
                self._candidate_shape_transform,
                model.shape_scale,
                model.shape_margin,
                model.shape_source_ptr,
                model.shape_body,
                model.shape_world,
                model._shape_mesh_properties,
                self._endpoint_pairs,
                self._endpoint_pair_count,
                self._reference_pair_state,
                self._separation_tolerance,
                False,
                CONTACT_ORACLE_VIOLATION,
                CONTACT_ORACLE_REFERENCE_INFEASIBLE,
                CONTACT_ORACLE_INCOMPLETE,
                CONTACT_ORACLE_CAPACITY,
                self.world_status,
                self.world_min_separation,
                self._query_incomplete,
                self._shape_key_bits,
                wp.uint64(self._vertex_key_max),
                self._guard_slots_per_world,
                self._guard_feature_keys,
                self._guard_feature_contact_indices,
                self._guard_feature_update_generations,
                contacts.contact_generation,
                contacts.rigid_contact_max,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point_id,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_normal_owner,
                contacts.rigid_contact_is_predictive,
                contacts.rigid_contact_is_strict_guard,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_tids,
                self._endpoint_blocks,
            ],
            block_dim=MESH_SDF_ORACLE_BLOCK_DIM,
            device=model.device,
            record_tape=False,
        )
        wp.launch(
            kernel=_finalize_query_status,
            dim=model.world_count,
            inputs=[
                self._candidate_pair_count,
                self._candidate_pairs.shape[0],
                self._query_incomplete,
                self.world_status,
                CONTACT_ORACLE_INCOMPLETE,
                CONTACT_ORACLE_CAPACITY,
            ],
            device=model.device,
            record_tape=False,
        )

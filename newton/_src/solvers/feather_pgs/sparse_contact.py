# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build contact responses in the sparse mass factor's coordinates."""

import textwrap
from functools import cache

import warp as wp

from .contact_laws import (
    contact_bodies,
    contact_friction,
    contact_friction_points,
    contact_points,
    contact_restitution,
    contact_row_frame,
    contact_tangent_basis,
    free_body_jacobian,
    prescribed_target,
)
from .device_preludes import dual_native, lane_group
from .friction import _PAIR_METRIC
from .friction_patches import FrictionPatches, patch_friction_location
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    contact_restitution_fires,
)
from .sparse_mass_matrix import _SparseRowSupport
from .sparse_row_projection import _sparse_contact_projection_source


@wp.struct
class SparseContactRows:
    """Material, prescribed motion and final row data for sparse preparation."""

    body_velocity: wp.array[wp.spatial_vector]
    prescribed_articulation: wp.array[int]
    material_friction: wp.array[float]
    material_restitution: wp.array[float]
    beta: float
    cfm: float
    friction_scale: float
    friction_beta: float
    speculative_scale: float
    restitution_velocity_threshold: float
    type: wp.array2d[int]
    parent: wp.array2d[int]
    mu: wp.array2d[float]
    beta_row: wp.array2d[float]
    cfm_row: wp.array2d[float]
    phi: wp.array2d[float]
    target_velocity: wp.array2d[float]
    restitution: wp.array2d[float]
    rhs: wp.array2d[float]


CONTACT_RECORD_FLOATS = 8
"""Floats of a contact triplet's block record: ``W_nt1, W_nt2, W_t1t2, 1/W_nn``, then the pair
law's tangent eigenbasis ``largest, smallest, vx, vy`` relative to the larger tangent diagonal."""
CONTACT_RING_FLOATS = 4
"""Floats a linked sparse sweep appends to each record: the anchor's other ring members, as int bits."""


def _insert_once(text: str, anchor: str, addition: str, *, replace: bool = False) -> str:
    """Return ``text`` with ``addition`` after (or, with ``replace``, instead of) its one ``anchor``."""
    if text.count(anchor) != 1:
        raise RuntimeError(f"expected one occurrence of {anchor!r}")
    return text.replace(anchor, addition if replace else anchor + addition)


# The contact block record of a full triplet (CONTACT_RECORD_FLOATS): the cross terms
# and inverse normal diagonal, then the pair law's metric of the tangent block, formed from the
# diagonals as written. Lane sums meet in the same tree as the norms.
_CONTACT_RECORD_SOURCE = """
    if (row_count == 3) {
        SUM(cross.c[0], cross.c[1], cross.c[2]);
        if (LANE == 0) {
            float* record = row_free_response.data
                + (world * row_free_response.shape[1] + slot / 3) * row_free_response.shape[2];
            record[0] = cross.c[0];
            record[1] = cross.c[1];
            record[2] = cross.c[2];
            record[3] = 1.0f / (norm.c[0] + cfm);
            float a = norm.c[1] + cfm, c = cross.c[2], d = norm.c[2] + cfm;
            {
$PAIR_METRIC
                record[4] = largest;
                record[5] = smallest;
                record[6] = vx;
                record[7] = vy;
            }
        }
    }
"""


@cache
def _get_sparse_contact_response_kernel(
    dof_count: int,
    *,
    lanes_per_contact: int = 8,
    warps_per_block: int = 4,
    prepare_rows: bool = False,
    use_support_table: bool = False,
    contact_records: bool = False,
) -> wp.Kernel:
    """Project contact rows with a lane group per contact and no global Jacobian.

    Contact directions share endpoint geometry, support and inverse-factor
    loads. Each endpoint Jacobian is evaluated once in shared scratch. Free
    bodies retain their physical coordinates and existing mass metric.

    With ``prepare_rows``, the geometry also produces solver metadata, diagonal
    CFM, bias and restitution. Only normal rows retain the scratch metadata
    needed by restitution; the caller must not run separate row passes.

    ``use_support_table`` publishes one topology class per row instead of
    repeating its DOF indices. The caller admits one factored articulation
    per world, at offset zero, with no physical free-body response.

    ``contact_records`` also writes each full contact triplet's block record
    for the sparse sweep's single-stage transaction (see
    :data:`CONTACT_RECORD_FLOATS`): the lane groups accumulate the
    three cross products with the norms, and one lane forms ``W_nt1, W_nt2,
    W_t1t2, 1/W_nn`` and the friction pair's tangent eigenbasis from the
    diagonals it writes. The record of the triplet at normal row ``slot`` is
    row ``slot // 3`` of world ``world`` in ``row_free_response``, which the
    support table's routes leave unused; it has shape ``(world_count,
    max_constraints // 3, width)`` with ``width >= 8``. It requires
    ``prepare_rows`` and ``use_support_table``.

    All input/output arrays must be contiguous for the native flat accesses.
    Launch total_num_workers * lanes_per_contact threads in blocks of
    warps_per_block * 32. The CPU implementation uses one lane per group.
    """
    n, lanes, warps = int(dof_count), int(lanes_per_contact), int(warps_per_block)
    if not 1 <= n <= 64 or lanes not in (8, 16, 32) or not 1 <= warps <= 32:
        raise ValueError("Invalid sparse contact dimensions or lane group")
    if contact_records and not (prepare_rows and use_support_table):
        raise ValueError("Contact records require prepared rows and the row support table")
    groups = warps * 32 // lanes
    scratch_size = max(n, 6) + 6
    geometry_size = 27 if prepare_rows else 0
    geometry_cuda = """
    auto angular_a_shared = angular_a;
    auto angular_b_shared = angular_b;
    auto directions_shared = directions;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        if (row >= row_count) continue;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            angular_a_shared.data[row][k] = READ(angular_a.data[row][k], 0);
            angular_b_shared.data[row][k] = READ(angular_b.data[row][k], 0);
            directions_shared.data[row][k] = READ(directions.data[row][k], 0);
        }
    }
"""
    geometry_cpu = """
    const auto angular_a_shared = angular_a;
    const auto angular_b_shared = angular_b;
    const auto directions_shared = directions;
"""
    geometry_dot = """
        const auto& angular_wrenches = side == 0 ? angular_a_shared : angular_b_shared;
        const auto direction = wp::vec3(directions_shared.data[row][0],
            directions_shared.data[row][1], directions_shared.data[row][2]);
        const auto angular_wrench = wp::vec3(angular_wrenches.data[row][0],
            angular_wrenches.data[row][1], angular_wrenches.data[row][2]);
"""
    if prepare_rows:
        # Geometry is shared by the contact, not live in every lane throughout
        # the factor contraction. Volatile reads prevent loop-invariant loads
        # from restoring the entire matrix triplet to lane registers.
        geometry_copy = """
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            if (row >= row_count) continue;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                geometry[row * 9 + k] = directions.data[row][k];
                geometry[row * 9 + 3 + k] = angular_a.data[row][k];
                geometry[row * 9 + 6 + k] = angular_b.data[row][k];
            }
        }
"""
        geometry_cuda = (
            "volatile float* geometry = jacobian + $GEOMETRY;\nif (LANE == 0) {\n" + geometry_copy + "}\nSYNC();\n"
        )
        geometry_cpu = "float geometry[27];\n" + geometry_copy
        geometry_dot = """
        const int angular_offset = row * 9 + 3 + side * 3;
        const auto direction = wp::vec3(geometry[row * 9], geometry[row * 9 + 1], geometry[row * 9 + 2]);
        const auto angular_wrench = wp::vec3(geometry[angular_offset],
            geometry[angular_offset + 1], geometry[angular_offset + 2]);
"""
    snippet = """
#if defined(__CUDA_ARCH__)
    __shared__ float storage[$STORAGE];
    float* jacobian = storage + SUB * $SCRATCH;
    $GEOMETRY_CUDA
#else
    if ((tid & ($LANES - 1)) != 0) return;
    float jacobian[$SCRATCH];
    $GEOMETRY_CPU
#endif
    float* free_factor = jacobian + $JACOBIAN;
    const auto jacobian_value = [&](const auto& motion, int side, int row) {
        const auto linear = wp::vec3(motion.c[0], motion.c[1], motion.c[2]);
        const auto angular = wp::vec3(motion.c[3], motion.c[4], motion.c[5]);
        $GEOMETRY_DOT
        return wp::dot(direction, linear) + wp::dot(angular_wrench, angular);
    };
    unsigned long long mask_a = 0, mask_b = 0;
    if (body_a >= 0 && art_a >= 0) mask_a = body_dof_mask.data[body_a];
    if (body_b >= 0 && art_b >= 0) mask_b = body_dof_mask.data[body_b];
    const int row_offset = world * row_factor.shape[1] + slot;
    const int support_capacity = row_factor.shape[2];
    int count = 0;
    auto norm = wp::vec3(0.0f);
    auto incident = wp::vec3(0.0f);
    // Physical blocks remain a short prefix; ordinary free/free contacts use
    // native MF storage, so production rows have at most one such block.
    int first_endpoint = 0;
    if (art_b >= 0 && is_free_rigid.data[art_b] != 0
        && (art_a < 0 || is_free_rigid.data[art_a] == 0)) first_endpoint = 1;
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int side = (first_endpoint + endpoint) % 2;
        const int art = side == 0 ? art_a : art_b;
        if (art < 0 || (side == 1 && art == art_a)
            || articulation_response_dof_count.data[art] <= 0) continue;
        const int group = art_group_idx.data[art];
        const int start = articulation_dof_start.data[art];
        const int offset = articulation_world_dof_offset.data[art];
        if (is_free_rigid.data[art] != 0) {
            for (int node = LANE; node < 6; node += STRIDE) {
                const auto motion = joint_S_s.data[start + node];
                const float velocity = v_hat.data[start + node];
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    float value = 0.0f;
                    if (art == art_a) value += jacobian_value(motion, 0, row);
                    if (art == art_b) value -= jacobian_value(motion, 1, row);
                    jacobian[row * $PLANE + node] = value;
                    incident.c[row] += value * velocity;
                }
            }
            SYNC();
            const int factor_base = group * 36;
            for (int node = LANE; node < 6; node += STRIDE) {
                auto value = wp::vec3(0.0f);
                for (int column = 0; column <= node; ++column) {
                    const float factor = free_inverse_factor.data[factor_base + node * 6 + column];
#pragma unroll
                    for (int row = 0; row < 3; ++row) {
                        if (row < row_count) value.c[row] += factor * jacobian[row * $PLANE + column];
                    }
                }
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row < row_count) free_factor[row * $PLANE + node] = value.c[row];
                }
            }
            SYNC();
            for (int node = LANE; node < 6; node += STRIDE) {
                auto response = wp::vec3(0.0f);
                for (int column = node; column < 6; ++column) {
                    const float factor = free_inverse_factor.data[factor_base + column * 6 + node];
#pragma unroll
                    for (int row = 0; row < 3; ++row) {
                        if (row < row_count) response.c[row] += factor * free_factor[row * $PLANE + column];
                    }
                }
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    const int response_offset = (row_offset + row) * support_capacity;
                    const float value = jacobian[row * $PLANE + node];
                    if (!$SUPPORT_TABLE) row_dof.data[response_offset + count + node] = offset + node;
                    row_factor.data[response_offset + count + node] = value;
                    row_free_response.data[(row_offset + row) * row_free_response.shape[2] + count + node] = response.c[row];
                    norm.c[row] += value * response.c[row];
                }
            }
            count += 6;
        } else {
            unsigned long long mask = 0;
            if (art == art_a) mask |= mask_a;
            if (art == art_b) mask |= mask_b;
            for (int node = LANE; node < $N; node += STRIDE) {
                const unsigned long long bit = 1ull << node;
                if ((mask & bit) == 0) continue;
                const int physical = permutation.data[node];
                const auto motion = joint_S_s.data[start + physical];
                const float velocity = v_hat.data[start + physical];
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    float value = 0.0f;
                    if (art == art_a && (mask_a & bit) != 0)
                        value += jacobian_value(motion, 0, row);
                    if (art == art_b && (mask_b & bit) != 0)
                        value -= jacobian_value(motion, 1, row);
                    jacobian[row * $PLANE + node] = value;
                    incident.c[row] += value * velocity;
                }
            }
            SYNC();
            const int factor_base = group * inverse_factor.shape[1];
            for (int node = LANE; node < $N; node += STRIDE) {
                const unsigned long long bit = 1ull << node;
                if ((mask & bit) == 0) continue;
                $SPARSE_PROJECTION
                const int index = count + COUNT(mask & (bit - 1ull));
#pragma unroll
                for (int row = 0; row < 3; ++row) {
                    if (row >= row_count) continue;
                    const int response_offset = (row_offset + row) * support_capacity;
                    if (!$SUPPORT_TABLE) row_dof.data[response_offset + index] = offset + node;
                    row_factor.data[response_offset + index] = value.c[row];
                    norm.c[row] += value.c[row] * value.c[row];
                }
            }
            count += COUNT(mask);
        }
        // The next endpoint reuses the same group-local Jacobian storage.
        SYNC();
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        if (row >= row_count) continue;
        const int response_offset = (row_offset + row) * support_capacity;
        for (int index = count + LANE; index < support_capacity; index += STRIDE) {
            if (!$SUPPORT_TABLE) row_dof.data[response_offset + index] = -1;
            row_factor.data[response_offset + index] = 0.0f;
        }
        SUM(norm.c[row], incident.c[row]);
        if (LANE == 0) {
            if ($SUPPORT_TABLE) {
                const int class_a = mask_a != 0ull && art_a >= 0 && articulation_response_dof_count.data[art_a] > 0
                    ? support.body_class.data[body_a] : 0;
                const int class_b = mask_b != 0ull && art_b >= 0 && articulation_response_dof_count.data[art_b] > 0
                    ? support.body_class.data[body_b] : 0;
                support.row_class.data[row_offset + row] = support.pair_class.data[
                    class_a * support.pair_class.shape[1] + class_b];
            }
            row_incident.data[row_offset + row] = incident.c[row];
            diagonal.data[row_offset + row] = norm.c[row] + cfm;
        }
    }
"""
    if contact_records:
        # Records ride in row_free_response: the support table's routes admit no free-body
        # response, so its free branch writes no responses there, only the cross products.
        snippet = _insert_once(snippet, "    auto norm = wp::vec3(0.0f);\n", "    auto cross = wp::vec3(0.0f);\n")
        snippet = _insert_once(
            snippet,
            "                    row_free_response.data[(row_offset + row) * row_free_response.shape[2]"
            " + count + node] = response.c[row];\n",
            "",
            replace=True,
        )
        snippet = _insert_once(
            snippet,
            "                    norm.c[row] += value * response.c[row];\n                }\n",
            """                if (row_count == 3) {
                    cross.c[0] += jacobian[node] * response.c[1];
                    cross.c[1] += jacobian[node] * response.c[2];
                    cross.c[2] += jacobian[$PLANE + node] * response.c[2];
                }
""",
        )
        snippet = _insert_once(
            snippet,
            "                    norm.c[row] += value.c[row] * value.c[row];\n                }\n",
            """                if (row_count == 3) {
                    cross.c[0] += value.c[0] * value.c[1];
                    cross.c[1] += value.c[0] * value.c[2];
                    cross.c[2] += value.c[1] * value.c[2];
                }
""",
        )
        metric = textwrap.indent(_PAIR_METRIC.substitute(max_call="wp::max"), "        ")
        snippet += _CONTACT_RECORD_SOURCE.replace("$PAIR_METRIC", metric.rstrip("\n"))
    snippet = (
        snippet.replace(
            "$SPARSE_PROJECTION",
            _sparse_contact_projection_source("shared_mask", dof_count=n, scratch_stride=scratch_size),
        )
        .replace("$GEOMETRY_CUDA", geometry_cuda)
        .replace("$GEOMETRY_CPU", geometry_cpu)
        .replace("$GEOMETRY_DOT", geometry_dot)
        .replace("$GEOMETRY", str(3 * scratch_size))
        .replace("$STORAGE", str(groups * (3 * scratch_size + geometry_size)))
        .replace("$LANES", str(lanes))
        .replace("$SCRATCH", str(3 * scratch_size + geometry_size))
        .replace("$PLANE", str(scratch_size))
        .replace("$JACOBIAN", str(max(n, 6)))
        .replace("$N", str(n))
        .replace("$SUPPORT_TABLE", str(bool(use_support_table)).lower())
    )
    snippet = dual_native(snippet, preludes=(lane_group(lanes),))

    @wp.func_native(snippet)
    def project_native(
        tid: int,
        body_a: int,
        body_b: int,
        art_a: int,
        art_b: int,
        world: int,
        slot: int,
        row_count: int,
        angular_a: wp.mat33,
        angular_b: wp.mat33,
        directions: wp.mat33,
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        permutation: wp.array[int],
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        cfm: float,
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        support: _SparseRowSupport,
    ): ...

    def contact_kernel(
        contact_count: wp.array[int],
        total_num_workers: int,
        contact_point0: wp.array[wp.vec3],
        contact_point1: wp.array[wp.vec3],
        contact_normal: wp.array[wp.vec3],
        contact_shape0: wp.array[int],
        contact_shape1: wp.array[int],
        contact_thickness0: wp.array[float],
        contact_thickness1: wp.array[float],
        contact_world: wp.array[int],
        contact_slot: wp.array[int],
        contact_art_a: wp.array[int],
        contact_art_b: wp.array[int],
        contact_path: wp.array[int],
        contact_slots_needed: wp.array[int],
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        articulation_origin: wp.array[wp.vec3],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        shape_body: wp.array[int],
        body_q: wp.array[wp.transform],
        contact_friction_shared_anchor: int,
        friction_patches: FrictionPatches,
        contact_shared_anchor: int,
        permutation: wp.array[int],
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        rows: SparseContactRows,
        dt: float,
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        support: _SparseRowSupport,
    ):
        tid = wp.tid()
        lane = tid % lanes
        worker = tid // lanes
        total_contacts = contact_count[0]
        for c in range(worker, total_contacts, total_num_workers):
            row_count = wp.min(contact_slots_needed[c], 3)
            if contact_path[c] != 0 or contact_slot[c] < 0 or row_count <= 0:
                continue
            body_a, body_b = contact_bodies(contact_shape0[c], contact_shape1[c], shape_body)
            art_a = contact_art_a[c]
            art_b = contact_art_b[c]
            normal = wp.vec3(0.0)
            surface_a = wp.vec3(0.0)
            surface_b = wp.vec3(0.0)
            point_a = wp.vec3(0.0)
            point_b = wp.vec3(0.0)
            friction_point_a = wp.vec3(0.0)
            friction_point_b = wp.vec3(0.0)
            tangent0 = wp.vec3(0.0)
            tangent1 = wp.vec3(0.0)
            gap = float(0.0)
            cfm = float(0.0)
            if wp.static(prepare_rows):
                cfm = rows.cfm
            if wp.static(prepare_rows) or lane == 0:
                normal, surface_a, surface_b, gap = contact_points(
                    contact_normal[c],
                    contact_point0[c],
                    contact_point1[c],
                    contact_thickness0[c],
                    contact_thickness1[c],
                    body_a,
                    body_b,
                    body_q,
                )
                friction_point_a, friction_point_b = contact_friction_points(
                    surface_a, surface_b, contact_shared_anchor, contact_friction_shared_anchor
                )
                if row_count > 1:
                    tangent0, tangent1 = contact_tangent_basis(normal)
                    if friction_patches.enabled != 0:
                        friction_point_a, friction_point_b = patch_friction_location(friction_patches, c)
                # The normal row's endpoints.
                _normal, point_a, point_b = contact_row_frame(
                    0,
                    contact_shared_anchor,
                    normal,
                    tangent0,
                    tangent1,
                    surface_a,
                    surface_b,
                    friction_point_a,
                    friction_point_b,
                )
            if wp.static(prepare_rows):
                if lane < row_count:
                    world = contact_world[c]
                    slot = contact_slot[c] + lane
                    direction, anchor_a, anchor_b = contact_row_frame(
                        lane,
                        contact_shared_anchor,
                        normal,
                        tangent0,
                        tangent1,
                        surface_a,
                        surface_b,
                        friction_point_a,
                        friction_point_b,
                    )
                    target = prescribed_target(
                        body_a,
                        art_a,
                        body_b,
                        art_b,
                        anchor_a,
                        anchor_b,
                        direction,
                        rows.prescribed_articulation,
                        articulation_origin,
                        rows.body_velocity,
                    )
                    mu = contact_friction(contact_shape0[c], contact_shape1[c], rows.material_friction)
                    beta = rows.beta
                    phi = gap
                    row_type = PGS_CONSTRAINT_TYPE_CONTACT
                    parent = int(-1)
                    restitution = float(0.0)
                    if lane == 0:
                        restitution = contact_restitution(
                            contact_shape0[c], contact_shape1[c], rows.material_restitution
                        )
                    else:
                        beta = rows.friction_beta
                        phi = friction_patches.phi[c][lane - 1]
                        mu *= rows.friction_scale
                        row_type = PGS_CONSTRAINT_TYPE_FRICTION
                        parent = contact_slot[c]
                    inv_dt = 1.0 / dt
                    bias = beta * phi * inv_dt
                    if lane == 0 and phi > 0.0:
                        bias = rows.speculative_scale * phi * inv_dt
                    rows.type[world, slot] = row_type
                    rows.parent[world, slot] = parent
                    rows.mu[world, slot] = mu
                    # Bias and CFM are already applied; only normal restitution
                    # consumes row metadata after the response projection.
                    if lane == 0:
                        rows.phi[world, slot] = phi
                        rows.target_velocity[world, slot] = target
                        rows.restitution[world, slot] = restitution
                    rows.rhs[world, slot] = -target + bias

            # Fixed component constructors avoid materializing dynamically
            # indexed matrices in local memory during geometry assembly.
            angular_a = wp.mat33(0.0)
            angular_b = wp.mat33(0.0)
            directions = wp.mat33(0.0)
            if lane == 0:
                normal_a = wp.vec3(0.0)
                normal_b = wp.vec3(0.0)
                tangent0_a = wp.vec3(0.0)
                tangent1_a = wp.vec3(0.0)
                tangent0_b = wp.vec3(0.0)
                tangent1_b = wp.vec3(0.0)
                # Each row's wrench about an endpoint's articulation origin: the angular part of the free-body
                # Jacobian [d, (p - o) x d], whose product with a motion column is that column's Jacobian entry.
                if art_a >= 0 and articulation_response_dof_count[art_a] > 0:
                    origin_a = articulation_origin[art_a]
                    normal_a = wp.spatial_bottom(free_body_jacobian(normal, point_a, origin_a))
                    if row_count > 1:
                        tangent0_a = wp.spatial_bottom(free_body_jacobian(tangent0, friction_point_a, origin_a))
                        tangent1_a = wp.spatial_bottom(free_body_jacobian(tangent1, friction_point_a, origin_a))
                if art_b >= 0 and articulation_response_dof_count[art_b] > 0:
                    origin_b = articulation_origin[art_b]
                    normal_b = wp.spatial_bottom(free_body_jacobian(normal, point_b, origin_b))
                    if row_count > 1:
                        tangent0_b = wp.spatial_bottom(free_body_jacobian(tangent0, friction_point_b, origin_b))
                        tangent1_b = wp.spatial_bottom(free_body_jacobian(tangent1, friction_point_b, origin_b))
                angular_a = wp.matrix_from_rows(normal_a, tangent0_a, tangent1_a)
                angular_b = wp.matrix_from_rows(normal_b, tangent0_b, tangent1_b)
                directions = wp.matrix_from_rows(normal, tangent0, tangent1)
            project_native(
                tid,
                body_a,
                body_b,
                art_a,
                art_b,
                contact_world[c],
                contact_slot[c],
                row_count,
                angular_a,
                angular_b,
                directions,
                art_group_idx,
                articulation_response_dof_count,
                is_free_rigid,
                articulation_world_dof_offset,
                articulation_dof_start,
                body_dof_mask,
                joint_S_s,
                permutation,
                factor_row_offsets,
                inverse_factor,
                free_inverse_factor,
                v_hat,
                cfm,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                diagonal,
                support,
            )
            if wp.static(prepare_rows):
                if lane == 0:
                    world = contact_world[c]
                    slot = contact_slot[c]
                    target = rows.target_velocity[world, slot]
                    relative_incident = row_incident[world, slot] - target
                    restitution = rows.restitution[world, slot]
                    if restitution > 0.0 and contact_restitution_fires(
                        rows.phi[world, slot], relative_incident, dt, rows.restitution_velocity_threshold
                    ):
                        rows.rhs[world, slot] = -target + restitution * relative_incident

    contact_kernel.__name__ = f"populate_sparse_contact_response_{n}_{lanes}_{warps}"
    if prepare_rows:
        contact_kernel.__name__ += "_prepared"
    if use_support_table:
        contact_kernel.__name__ += "_support_table"
    if contact_records:
        contact_kernel.__name__ += "_contact_records"
    contact_kernel.__qualname__ = contact_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique")(contact_kernel)


@wp.kernel(enable_backward=False)
def finalize_sparse_joint_limit_rows(
    phase_bounds: wp.array2d[int],
    constraint_count: wp.array[int],
    rows: SparseContactRows,
    dt: float,
    diagonal: wp.array2d[float],
):
    """Finish the internal-constraint prefix; contacts are already complete."""
    world = wp.tid()
    for row in range(wp.min(phase_bounds[world, 1], constraint_count[world])):
        phi = rows.phi[world, row]
        inv_dt = 1.0 / dt
        bias = phi * inv_dt
        if phi < 0.0 or rows.type[world, row] == PGS_CONSTRAINT_TYPE_MIMIC:
            bias = rows.beta_row[world, row] * phi * inv_dt
        diagonal[world, row] += rows.cfm_row[world, row]
        rows.rhs[world, row] = -rows.target_velocity[world, row] + bias


@wp.func_native(
    dual_native(
        """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if !defined(__CUDA_ARCH__)
    if ((tid & 31) != 0) return;
#endif
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    const int start = articulation_dof_start.data[art];
    const int dofs = inverse_permutation.shape[0];
    const int capacity = row_type.shape[1];
    const int support_capacity = row_factor.shape[2];
    const int factor_base = group * inverse_factor.shape[1];
    for (int base = 0; base < 2 * dofs; base += STRIDE) {
        const int candidate = base + LANE;
        const int local_dof = candidate >> 1;
        const int side = candidate & 1;
        const int dof = start + local_dof;
        const int q_index = candidate < 2 * dofs ? limit_q_index.data[dof] : -1;
        float gap = 0.0f;
        int active = 0;
        if (q_index >= 0) {
            const float position = joint_q.data[q_index];
            const float bound = side == 0 ? joint_limit_lower.data[dof] : joint_limit_upper.data[dof];
            gap = side == 0 ? position - bound : bound - position;
            active = wp::isfinite(bound)
                && (side == 0 ? position <= bound + activation_gap : position >= bound - activation_gap);
        }
        const unsigned active_mask = BALLOT(active != 0);
        const int first_slot = RESERVE(&slot_counter.data[world], COUNT(active_mask));
        // Visit the ballot in lane order, matching the existing lower/upper DOF order.
        unsigned pending = active_mask;
        int rank = 0;
        while (pending != 0) {
            const int row = first_slot + rank;
            // The reservation above still counts every over-capacity row.
            if (row >= capacity) break;
            const int source = FIRST(pending);
            const int selected_dof = READ(local_dof, source);
            const int selected_side = READ(side, source);
            const float selected_gap = READ(gap, source);
            const float sign = selected_side == 0 ? 1.0f : -1.0f;
            const int column = inverse_permutation.data[selected_dof];
            const unsigned long long support_mask = dof_mask.data[selected_dof];
            const int row_offset = world * capacity + row;
            const int response_offset = row_offset * support_capacity;
            float norm = 0.0f;
            for (int node = LANE; node < dofs; node += STRIDE) {
                const unsigned long long bit = 1ull << node;
                if ((support_mask & bit) == 0) continue;
                const int index = COUNT(support_mask & (bit - 1ull));
                const int entry = factor_lookup.data[node * dofs + column];
                const float value = entry >= 0 ? sign * inverse_factor.data[factor_base + entry] : 0.0f;
                if (!support.enabled)
                    row_dof.data[response_offset + index] = articulation_world_dof_offset.data[art] + node;
                row_factor.data[response_offset + index] = value;
                norm += value * value;
            }
            const int support_count = COUNT(support_mask);
            for (int index = support_count + LANE; index < support_capacity; index += STRIDE) {
                if (!support.enabled) row_dof.data[response_offset + index] = -1;
                row_factor.data[response_offset + index] = 0.0f;
            }
            SUM(norm);
            if (LANE == 0) {
                if (support.enabled)
                    support.row_class.data[row_offset] = support.limit_class.data[selected_dof];
                row_incident.data[row_offset] = sign * v_hat.data[start + selected_dof];
                diagonal.data[row_offset] = norm;
                row_type.data[row_offset] = $LIMIT_TYPE;
                row_parent.data[row_offset] = -1;
                row_mu.data[row_offset] = 0.0f;
                row_beta.data[row_offset] = pgs_beta;
                row_cfm.data[row_offset] = pgs_cfm;
                phi.data[row_offset] = selected_gap;
                target_velocity.data[row_offset] = 0.0f;
            }
            pending &= pending - 1;
            ++rank;
        }
    }
""".replace("$LIMIT_TYPE", str(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)),
        preludes=(lane_group(32),),
    )
)
def _build_sparse_joint_limit_rows(
    tid: int,
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    slot_counter: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
    support: _SparseRowSupport,
): ...


@wp.kernel(enable_backward=False)
def build_sparse_joint_limit_rows(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    slot_counter: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
    support: _SparseRowSupport,
):
    """Build ordered limit rows and sparse responses with one warp per articulation.

    Launch ``32 * group_count`` threads in blocks divisible by 32. Active row
    reservations include over-capacity demand; writes stay within row capacity.
    Reservations retain each articulation's lower/upper DOF order.
    """
    _build_sparse_joint_limit_rows(
        wp.tid(),
        group_to_art,
        art_to_world,
        articulation_world_dof_offset,
        articulation_dof_start,
        limit_q_index,
        joint_limit_lower,
        joint_limit_upper,
        joint_q,
        activation_gap,
        pgs_beta,
        pgs_cfm,
        dof_mask,
        inverse_permutation,
        factor_lookup,
        inverse_factor,
        v_hat,
        slot_counter,
        row_type,
        row_parent,
        row_mu,
        row_beta,
        row_cfm,
        phi,
        target_velocity,
        row_dof,
        row_factor,
        row_incident,
        diagonal,
        support,
    )


@wp.func_native(
    dual_native(
        """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if !defined(__CUDA_ARCH__)
    if ((tid & 31) != 0) return;
#endif
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    const int start = articulation_dof_start.data[art];
    const int dofs = inverse_permutation.shape[0];
    const int capacity = row_type.shape[1];
    const int support_capacity = row_factor.shape[2];
    const int factor_base = group * inverse_factor.shape[1];
    for (int local_dof = 0; local_dof < dofs; ++local_dof) {
        const int dof = start + local_dof;
        const int lower = velocity_limit_slot.data[2 * dof];
        const int upper = velocity_limit_slot.data[2 * dof + 1];
        // The existing allocator owns eligibility, ordering and overflow demand.
        const bool write_lower = lower >= 0 && lower < capacity;
        const bool write_upper = upper >= 0 && upper < capacity;
        if (!write_lower && !write_upper) continue;
        const float lower_sign = write_lower ? velocity_limit_sign.data[2 * dof] : 0.0f;
        const float upper_sign = write_upper ? velocity_limit_sign.data[2 * dof + 1] : 0.0f;
        const int column = inverse_permutation.data[local_dof];
        const unsigned long long support_mask = dof_mask.data[local_dof];
        const int lower_offset = (world * capacity + lower) * support_capacity;
        const int upper_offset = (world * capacity + upper) * support_capacity;
        const int dof_offset = articulation_world_dof_offset.data[art];
        float norm = 0.0f;
        for (int node = LANE; node < dofs; node += STRIDE) {
            const unsigned long long bit = 1ull << node;
            if ((support_mask & bit) == 0) continue;
            const int index = COUNT(support_mask & (bit - 1ull));
            const int entry = factor_lookup.data[node * dofs + column];
            const float value = entry >= 0 ? inverse_factor.data[factor_base + entry] : 0.0f;
            if (write_lower) {
                if (!support.enabled) row_dof.data[lower_offset + index] = dof_offset + node;
                row_factor.data[lower_offset + index] = entry >= 0 ? lower_sign * value : 0.0f;
            }
            if (write_upper) {
                if (!support.enabled) row_dof.data[upper_offset + index] = dof_offset + node;
                row_factor.data[upper_offset + index] = entry >= 0 ? upper_sign * value : 0.0f;
            }
            // Allocator-owned unit signs share the same squared column norm.
            norm += value * value;
        }
        const int support_count = COUNT(support_mask);
        for (int index = support_count + LANE; index < support_capacity; index += STRIDE) {
            if (write_lower) {
                if (!support.enabled) row_dof.data[lower_offset + index] = -1;
                row_factor.data[lower_offset + index] = 0.0f;
            }
            if (write_upper) {
                if (!support.enabled) row_dof.data[upper_offset + index] = -1;
                row_factor.data[upper_offset + index] = 0.0f;
            }
        }
        SUM(norm);
        if (LANE == 0) {
            const float incident = v_hat.data[dof];
            const float target = -joint_velocity_limit.data[dof];
            for (int side = 0; side < 2; ++side) {
                const int row = side == 0 ? lower : upper;
                if (row < 0 || row >= capacity) continue;
                const float sign = side == 0 ? lower_sign : upper_sign;
                const int row_offset = world * capacity + row;
                if (support.enabled)
                    support.row_class.data[row_offset] = support.limit_class.data[local_dof];
                row_incident.data[row_offset] = sign * incident;
                // The prefix finalizer adds CFM and forms RHS from target_velocity.
                diagonal.data[row_offset] = norm;
                row_type.data[row_offset] = $VELOCITY_LIMIT_TYPE;
                row_parent.data[row_offset] = -1;
                row_mu.data[row_offset] = 0.0f;
                row_beta.data[row_offset] = 0.0f;
                row_cfm.data[row_offset] = pgs_cfm;
                phi.data[row_offset] = 0.0f;
                target_velocity.data[row_offset] = target;
            }
        }
    }
""".replace("$VELOCITY_LIMIT_TYPE", str(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)),
        preludes=(lane_group(32),),
    )
)
def _build_sparse_joint_velocity_limit_rows(
    tid: int,
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_velocity_limit: wp.array[float],
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
    support: _SparseRowSupport,
): ...


@wp.kernel(enable_backward=False)
def build_sparse_joint_velocity_limit_rows(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_velocity_limit: wp.array[float],
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
    support: _SparseRowSupport,
):
    """Project allocated signed velocity-limit rows through the mass factor.

    Launch ``32 * group_count`` threads in blocks divisible by 32. The existing
    allocator owns lower/upper slots, activation and capacity reporting. This
    producer preserves even infinite limits and adds no position bias. Run
    :func:`finalize_sparse_joint_limit_rows` once after building the prefix.
    """
    _build_sparse_joint_velocity_limit_rows(
        wp.tid(),
        group_to_art,
        art_to_world,
        articulation_world_dof_offset,
        articulation_dof_start,
        joint_velocity_limit,
        velocity_limit_slot,
        velocity_limit_sign,
        pgs_cfm,
        dof_mask,
        inverse_permutation,
        factor_lookup,
        inverse_factor,
        v_hat,
        row_type,
        row_parent,
        row_mu,
        row_beta,
        row_cfm,
        phi,
        target_velocity,
        row_dof,
        row_factor,
        row_incident,
        diagonal,
        support,
    )


@wp.func_native(
    dual_native(
        """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if !defined(__CUDA_ARCH__)
    if ((tid & 31) != 0) return;
#endif
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    const int start = articulation_dof_start.data[art];
    const int dofs = inverse_permutation.shape[0];
    const int capacity = row_type.shape[1];
    const int support_capacity = row_factor.shape[2];
    const int factor_base = group * inverse_factor.shape[1];
    for (int m = mimic_art_start.data[art]; m < mimic_art_start.data[art + 1]; ++m) {
        const int k = mimic_art_list.data[m];
        const int row = mimic_slot.data[k];
        if (row < 0 || row >= capacity) continue;
        const int dof0 = mimic_dof0.data[k];
        const int dof1 = mimic_dof1.data[k];
        const int local0 = dof0 - start;
        const int local1 = dof1 - start;
        const int column0 = inverse_permutation.data[local0];
        const int column1 = inverse_permutation.data[local1];
        const float coefficient = mimic_coef1.data[k];
        // Keep both structural supports even when the live coefficient is zero.
        const unsigned long long support_mask = dof_mask.data[local0] | dof_mask.data[local1];
        const int row_offset = world * capacity + row;
        const int response_offset = row_offset * support_capacity;
        float norm = 0.0f;
        for (int node = LANE; node < dofs; node += STRIDE) {
            const unsigned long long bit = 1ull << node;
            if ((support_mask & bit) == 0) continue;
            const int index = COUNT(support_mask & (bit - 1ull));
            const int entry0 = factor_lookup.data[node * dofs + column0];
            const int entry1 = factor_lookup.data[node * dofs + column1];
            const float value0 = entry0 >= 0 ? inverse_factor.data[factor_base + entry0] : 0.0f;
            const float value1 = entry1 >= 0 ? inverse_factor.data[factor_base + entry1] : 0.0f;
            const float value = value0 - coefficient * value1;
            row_dof.data[response_offset + index] = articulation_world_dof_offset.data[art] + node;
            row_factor.data[response_offset + index] = value;
            norm += value * value;
        }
        const int support_count = COUNT(support_mask);
        for (int index = support_count + LANE; index < support_capacity; index += STRIDE) {
            row_dof.data[response_offset + index] = -1;
            row_factor.data[response_offset + index] = 0.0f;
        }
        SUM(norm);
        if (LANE == 0) {
            row_incident.data[row_offset] = v_hat.data[dof0] - coefficient * v_hat.data[dof1];
            diagonal.data[row_offset] = norm;
            row_type.data[row_offset] = $MIMIC_TYPE;
            row_parent.data[row_offset] = -1;
            row_mu.data[row_offset] = 0.0f;
            row_beta.data[row_offset] = pgs_beta;
            row_cfm.data[row_offset] = pgs_cfm;
            phi.data[row_offset] = joint_q.data[mimic_q0.data[k]]
                - coefficient * joint_q.data[mimic_q1.data[k]] - mimic_coef0.data[k];
            target_velocity.data[row_offset] = 0.0f;
        }
    }
""".replace("$MIMIC_TYPE", str(PGS_CONSTRAINT_TYPE_MIMIC)),
        preludes=(lane_group(32),),
    )
)
def _build_sparse_mimic_rows(
    tid: int,
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    mimic_slot: wp.array[int],
    mimic_art_start: wp.array[int],
    mimic_art_list: wp.array[int],
    mimic_dof0: wp.array[int],
    mimic_dof1: wp.array[int],
    mimic_q0: wp.array[int],
    mimic_q1: wp.array[int],
    mimic_coef0: wp.array[float],
    mimic_coef1: wp.array[float],
    joint_q: wp.array[float],
    pgs_beta: float,
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
): ...


@wp.kernel(enable_backward=False)
def build_sparse_mimic_rows(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    mimic_slot: wp.array[int],
    mimic_art_start: wp.array[int],
    mimic_art_list: wp.array[int],
    mimic_dof0: wp.array[int],
    mimic_dof1: wp.array[int],
    mimic_q0: wp.array[int],
    mimic_q1: wp.array[int],
    mimic_coef0: wp.array[float],
    mimic_coef1: wp.array[float],
    joint_q: wp.array[float],
    pgs_beta: float,
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
):
    """Project allocated same-articulation mimic rows through the mass factor.

    Launch ``32 * group_count`` threads in blocks divisible by 32. Existing
    validation and allocation own eligibility and slots. Rows use explicit
    coordinate indices, with capacity for the union of both DOF supports,
    including when the live scale is zero. The prefix finalizer adds bilateral
    position bias and CFM once after this producer.
    """
    _build_sparse_mimic_rows(
        wp.tid(),
        group_to_art,
        art_to_world,
        articulation_world_dof_offset,
        articulation_dof_start,
        mimic_slot,
        mimic_art_start,
        mimic_art_list,
        mimic_dof0,
        mimic_dof1,
        mimic_q0,
        mimic_q1,
        mimic_coef0,
        mimic_coef1,
        joint_q,
        pgs_beta,
        pgs_cfm,
        dof_mask,
        inverse_permutation,
        factor_lookup,
        inverse_factor,
        v_hat,
        row_type,
        row_parent,
        row_mu,
        row_beta,
        row_cfm,
        phi,
        target_velocity,
        row_dof,
        row_factor,
        row_incident,
        diagonal,
    )


@wp.kernel(enable_backward=False)
def apply_sparse_factor_velocity(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    factor_velocity_delta: wp.array2d[float],
    v_hat: wp.array[float],
    v_out: wp.array[float],
):
    """Decode the factor-coordinate impulse once per world after its sweeps."""
    group, column = wp.tid()
    art = group_to_art[group]
    world = art_to_world[art]
    delta = float(0.0)
    for row in range(column, permutation.shape[0]):
        entry = factor_lookup[row, column]
        if entry >= 0:
            delta += (
                inverse_factor[group, entry] * factor_velocity_delta[world, articulation_world_dof_offset[art] + row]
            )
    dof = articulation_dof_start[art] + permutation[column]
    v_out[dof] = v_hat[dof] + delta

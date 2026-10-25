# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a world's limit prefix and contact packets in place for the local solve owners.

A world made of one primary articulation (the one carrying joint limits and
mimic rows) and at most one secondary articulation (a free body, say) solves
its rows in the fused local kernels, which read the rows' physical Jacobians
directly. This producer writes those rows where the local kernels read them:
the limit and mimic prefix as sparse coefficients, the contact rows as dense
packets in the world's physical coordinates, both with the complete row
metadata and right-hand sides of the ordinary producers. Only worlds the
local owners decline (too many rows, matrix-free rows of another body,
unsupported row kinds) have their rows materialized into the ordinary
per-size buffers and run the ordinary response. Every size here comes from
the solver's topology plan; nothing names a robot.

Persistent patch friction writes its tangent rows with the patch row law.
The patch link, ``_FrictionPatchState.link``, runs after this producer and
alone links the patch rings and divides the friction coefficient; the
local owners pool each ring's normal load from the linked parents.
"""

from functools import cache, partial

import numpy as np
import warp as wp

from ...sim import ModelFlags
from .contact_input import ContactInput, bind
from .friction_patches import PatchRowLaw, patch_friction_location, patch_row_frame, patch_tangent_row
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    contact_restitution_fires,
    contact_tangent_basis,
    mixed_contact_restitution,
    prescribed_relative_contact_target,
)
from .row_inputs import CompactLocalRows, RowArgument

WARPS_PER_BLOCK = 4
PREFIX_WIDTH = 2  # a limit row has one coefficient, a mimic row two


@wp.struct
class PrefixData:
    group_to_art: wp.array[int]
    art_to_world: wp.array[int]
    dof_start: wp.array[int]
    mimic_start: wp.array[int]
    mimic_list: wp.array[int]
    mimic_valid: wp.array[int]
    mimic_enabled: wp.array[wp.bool]
    mimic_dof0: wp.array[int]
    mimic_dof1: wp.array[int]
    mimic_q0: wp.array[int]
    mimic_q1: wp.array[int]
    mimic_coef0: wp.array[float]
    mimic_coef1: wp.array[float]
    limit_q_index: wp.array[int]
    lower: wp.array[float]
    upper: wp.array[float]
    q: wp.array[float]
    activation_gap: float
    beta: float
    cfm: float
    dt: float
    bias_scale: float
    limit_speculative_scale: float
    # Sparse private prefix coefficients, never a global dense J allocation.
    dofs: wp.array3d[int]
    weights: wp.array3d[float]
    row_contact: wp.array2d[int]
    mimic_slot: wp.array[int]
    slot_counter: wp.array[int]
    first_rejected_slot: wp.array[int]
    bounds: wp.array2d[int]
    kind: wp.array2d[int]
    parent: wp.array2d[int]
    mu: wp.array2d[float]
    row_beta: wp.array2d[float]
    row_cfm: wp.array2d[float]
    phi: wp.array2d[float]
    target: wp.array2d[float]
    restitution: wp.array2d[float]
    rhs: wp.array2d[float]
    diag: wp.array2d[float]
    impulses: wp.array2d[float]


@wp.struct
class ContactPacketData:
    jacobian: wp.array3d[float]
    row_contact: wp.array2d[int]


@wp.struct
class QueueData:
    primary: wp.array[int]
    secondary: wp.array[int]
    residual_secondary: wp.array[int]
    mf_count: wp.array[int]
    mf_body0: wp.array2d[int]
    mf_body1: wp.array2d[int]
    mf_row_type: wp.array2d[int]
    body_art: wp.array[int]
    count: wp.array[int]
    owner: wp.array[int]
    status: wp.array[int]
    general_count: wp.array[int]
    general: wp.array[int]
    pair_count: wp.array[int]
    pair: wp.array[int]
    pair_secondary: wp.array[int]
    residual_count: wp.array[int]
    residual: wp.array[int]
    residual_pair: wp.array[int]


_PREFIX_SOURCE = r"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const int group = block * WARPS + (threadIdx.x >> 5);
    if (group >= data.group_to_art.shape[0]) return;
#else
    if (logical_lane != 0) return;
    for (int group = block * WARPS; group < (block + 1) * WARPS && group < data.group_to_art.shape[0]; ++group) {
    const int lane = 0;
#endif
    const int art = data.group_to_art.data[group];
    const int world = data.art_to_world.data[art];
    const int first_dof = data.dof_start.data[art];
    const int mimic_first = data.mimic_start.data[art];
    const int mimic_end = data.mimic_start.data[art + 1];
    const int M = data.kind.shape[1];
#if defined(__CUDA_ARCH__)
    for (int row = lane; row < PACKETS; row += 32) data.row_contact.data[world * PACKETS + row] = -1;
#else
    for (int row = 0; row < PACKETS; ++row) data.row_contact.data[world * PACKETS + row] = -1;
#endif
    int count = 0;
    for (int chunk = 0; chunk < CANDIDATES; chunk += 32) {
#if defined(__CUDA_ARCH__)
        const int candidate = chunk + lane;
#else
        for (int candidate = chunk; candidate < chunk + 32 && candidate < CANDIDATES; ++candidate) {
#endif
        int active = 0, d0 = -1, d1 = -1, type = 3, mimic = -1;
        float w0 = 0.0f, w1 = 0.0f, phi = 0.0f;
        if (candidate < MIMICS) {
            if (mimic_first + candidate < mimic_end) {
                mimic = data.mimic_list.data[mimic_first + candidate];
                data.mimic_slot.data[mimic] = -1;
                active = data.mimic_valid.data[mimic] != 0 && data.mimic_enabled.data[mimic];
                if (active) {
                    type = 5;
                    d0 = data.mimic_dof0.data[mimic] - first_dof;
                    d1 = data.mimic_dof1.data[mimic] - first_dof;
                    w0 = 1.0f;
                    w1 = -data.mimic_coef1.data[mimic];
                    phi = data.q.data[data.mimic_q0.data[mimic]]
                        - data.mimic_coef1.data[mimic] * data.q.data[data.mimic_q1.data[mimic]]
                        - data.mimic_coef0.data[mimic];
                }
            }
        } else if (candidate < CANDIDATES) {
            const int local = (candidate - MIMICS) / 2, side = (candidate - MIMICS) & 1;
            const int dof = first_dof + local, qi = data.limit_q_index.data[dof];
            if (qi >= 0) {
                const float q = data.q.data[qi];
                const float bound = side == 0 ? data.lower.data[dof] : data.upper.data[dof];
                phi = side == 0 ? q - bound : bound - q;
                active = wp::isfinite(bound) && (side == 0 ? q <= bound + data.activation_gap
                                                           : q >= bound - data.activation_gap);
                d0 = local;
                w0 = side == 0 ? 1.0f : -1.0f;
            }
        }
#if defined(__CUDA_ARCH__)
        const unsigned mask = __ballot_sync(0xffffffffu, active != 0);
        const unsigned earlier = lane == 0 ? 0u : ((1u << lane) - 1u);
        const int slot = count + __popc(mask & earlier);
        count += __popc(mask);
#else
        const int slot = count;
        count += active;
#endif
        if (active && slot < PACKETS) {
            const int sparse = (world * PACKETS + slot) * 2;
            data.dofs.data[sparse] = d0;
            data.dofs.data[sparse + 1] = d1;
            data.weights.data[sparse] = w0;
            data.weights.data[sparse + 1] = w1;
            const int row = world * M + slot;
            data.kind.data[row] = type;
            data.parent.data[row] = -1;
            data.mu.data[row] = 0.0f;
            data.row_beta.data[row] = data.beta;
            data.row_cfm.data[row] = data.cfm;
            data.phi.data[row] = phi;
            data.target.data[row] = 0.0f;
            data.restitution.data[row] = 0.0f;
            float rhs = 0.0f;
            if (type == 5 || phi < 0.0f) rhs = data.bias_scale * data.beta * phi * (1.0f / data.dt);
            else rhs = data.limit_speculative_scale * phi * (1.0f / data.dt);
            data.rhs.data[row] = rhs;
            data.diag.data[row] = data.cfm;
            data.impulses.data[row] = 0.0f;
            if (mimic >= 0) data.mimic_slot.data[mimic] = slot;
        }
#if !defined(__CUDA_ARCH__)
        }
#endif
    }
    if (lane == 0) {
        data.slot_counter.data[world] = count;
        data.bounds.data[world * 2] = count;
        data.bounds.data[world * 2 + 1] = count;
    }
#if !defined(__CUDA_ARCH__)
    }
#endif
"""


@cache
def get_prefix_kernel(primary: int, mimics: int, packets: int):
    """Build the mimic rows and the candidate limits of one primary articulation per warp.

    Launch tiled ``dim=[ceil(primaries / WARPS_PER_BLOCK)]``, ``block_dim=128``.
    Admission guarantees one primary per world, at most ``mimics`` mimic entries
    per primary, and ``mimics + 2 * primary <= packets`` so every active prefix
    row has a slot. Current q, enabled flags, coefficients and limits are read
    on every invocation.
    """
    if primary <= 0 or mimics < 0 or mimics + 2 * primary > packets:
        raise ValueError("the prefix candidates must fit the packet capacity")
    source = _PREFIX_SOURCE
    for token, value in (
        ("CANDIDATES", mimics + 2 * primary),
        ("MIMICS", mimics),
        ("PACKETS", packets),
        ("WARPS", WARPS_PER_BLOCK),
    ):
        source = source.replace(token, str(value))

    @wp.func_native(source)
    def prefix_native(block: int, logical_lane: int, data: PrefixData): ...

    def prefix(data: PrefixData):
        block, lane = wp.tid()
        prefix_native(block, lane, data)

    prefix.__name__ = prefix.__qualname__ = f"row_packet_prefix{primary}_m{mimics}_r{packets}"
    return wp.kernel(module="unique", enable_backward=False)(prefix)


@cache
def get_local_factory(primary: int, secondary: int, packets: int):
    """Bind the packet layout to the shared local row-input interface."""
    from . import solver_feather_pgs as solver  # noqa: PLC0415 - avoid constructor import cycles

    inputs = CompactLocalRows(
        primary,
        secondary,
        packets,
        packets,
        PREFIX_WIDTH,
        (RowArgument("prefix", PrefixData), RowArgument("packets", ContactPacketData)),
    )
    return partial(solver._get_pgs_solve_local_owned_kernel, row_inputs=inputs)


def validate_prefix_topology(
    groups, worlds, starts, mimic_start, mimic_list, mimic_dof0, mimic_dof1, primary, mimics, mimic_world=None
):
    """Check every prefix entry without per-world Python iteration.

    Notifications can reach this predicate on every batch reset. Keep the
    complete current topology checks, including disabled mimic entries, but
    vectorize over worlds and iterate only over the possible mimic slots.
    """
    art_count = min(len(worlds), len(starts) - 1, len(mimic_start) - 1)
    if np.any(groups < 0) or np.any(groups >= art_count):
        return False
    group_worlds = worlds[groups]
    if len(groups) != np.unique(group_worlds).size or len(groups) != np.unique(worlds).size:
        return False
    first_dof, end_dof = starts[groups], starts[groups + 1]
    first_entry, end_entry = mimic_start[groups], mimic_start[groups + 1]
    counts = end_entry - first_entry
    if (
        np.any(first_dof < 0)
        or np.any(end_dof - first_dof != primary)
        or np.any(first_entry < 0)
        or np.any(end_entry > len(mimic_list))
        or np.any(counts < 0)
        or np.any(counts > mimics)
    ):
        return False
    entry_count = min(len(mimic_dof0), len(mimic_dof1))
    if mimic_world is not None:
        entry_count = min(entry_count, len(mimic_world))
    for slot in range(mimics):
        active = counts > slot
        entries = mimic_list[first_entry[active] + slot]
        if np.any(entries < 0) or np.any(entries >= entry_count):
            return False
        if mimic_world is not None and np.any(mimic_world[entries] != group_worlds[active]):
            return False
        # Even disabled entries must be safe if subsequently enabled.
        lower, upper = first_dof[active], end_dof[active]
        if np.any(mimic_dof0[entries] < lower) or np.any(mimic_dof0[entries] >= upper):
            return False
        if np.any(mimic_dof1[entries] < lower) or np.any(mimic_dof1[entries] >= upper):
            return False
    return True


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return 32;
#else
return 1;
#endif
""")
def _row_stride() -> int: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return threadIdx.x;
#else
return 0;
#endif
""")
def _lane() -> int: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return wp::vec3(__shfl_sync(0xffffffffu, value[0], 0),
                __shfl_sync(0xffffffffu, value[1], 0),
                __shfl_sync(0xffffffffu, value[2], 0));
#else
return value;
#endif
""")
def _broadcast(value: wp.vec3) -> wp.vec3: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncwarp();
#endif
""")
def _join(): ...


@wp.kernel
def finalize_and_queue(
    prefix: PrefixData,
    queue: QueueData,
    local_rows: int,
    residual_rows: int,
    residual_mf_rows: int,
    primary_dofs: int,
):
    """Fuse the dense-count clamp, the local ownership decision and the two queues."""
    world = wp.tid()
    rows = prefix.slot_counter[world]
    if rows > prefix.kind.shape[1]:
        wp.atomic_max(queue.status, 0, 1)
        rows = prefix.kind.shape[1]
    rows = wp.min(rows, prefix.first_rejected_slot[world])
    queue.count[world] = rows
    mf_count = queue.mf_count[world]
    primary = queue.primary[world]
    secondary = queue.secondary[world]
    residual_secondary = queue.residual_secondary[world]
    single = prefix.bounds[world, 1] == rows
    local_mf = mf_count > 0 and mf_count <= residual_mf_rows and residual_secondary >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        body0, body1 = queue.mf_body0[world, mf_row], queue.mf_body1[world, mf_row]
        if body0 >= 0 and queue.body_art[body0] != residual_secondary:
            local_mf = False
        if body1 >= 0 and queue.body_art[body1] != residual_secondary:
            local_mf = False
        if queue.mf_row_type[world, mf_row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
            local_mf = False
        mf_row += 1
    owner = int(0)
    if rows > 0 and mf_count == 0:
        if single and rows <= wp.min(local_rows, primary_dofs) and primary >= 0:
            owner = 1
        elif not single and rows <= local_rows and secondary >= 0:
            owner = 2
    if owner == 0 and rows > 0 and rows <= residual_rows and residual_secondary >= 0:
        if (mf_count == 0 and not single) or local_mf:
            owner = 3
    queue.owner[world] = owner
    if owner == 0 and (rows > 0 or mf_count > 0):
        slot = wp.atomic_add(queue.general_count, 0, 1)
        queue.general[slot] = world
    elif owner == 2:
        slot = wp.atomic_add(queue.pair_count, 0, 1)
        queue.pair[slot] = primary
        queue.pair_secondary[slot] = secondary
    elif owner == 3:
        slot = wp.atomic_add(queue.residual_count, 0, 1)
        queue.residual[slot] = primary
        queue.residual_pair[slot] = residual_secondary


@wp.kernel
def materialize_general_prefix(
    workers: int,
    prefix: PrefixData,
    queue: QueueData,
    group_start: wp.array[int],
    group_art: wp.array[int],
    art_group: wp.array[int],
    size: int,
    J: wp.array3d[float],
):
    """Clear and populate only current general-world group rows, including zeros."""
    worker, lane = wp.tid()
    for entry in range(worker, queue.general_count[0], workers):
        world = queue.general[entry]
        for item in range(group_start[world], group_start[world + 1]):
            art = group_art[item]
            group = art_group[art]
            for coefficient in range(lane, queue.count[world] * size, _row_stride()):
                row, dof = coefficient // size, coefficient % size
                value = float(0.0)
                if art == queue.primary[world] and row < prefix.bounds[world, 0]:
                    for component in range(PREFIX_WIDTH):
                        if prefix.dofs[world, row, component] == dof:
                            value += prefix.weights[world, row, component]
                J[group, row, dof] = value


@wp.kernel
def general_response(
    workers: int,
    queue: QueueData,
    group_start: wp.array[int],
    group_art: wp.array[int],
    art_group: wp.array[int],
    world_offset: wp.array[int],
    size: int,
    L: wp.array3d[float],
    J: wp.array3d[float],
    Y: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Retain the ordinary per-row triangular response for the complete fallback."""
    worker, lane = wp.tid()
    for entry in range(worker, queue.general_count[0], workers):
        world = queue.general[entry]
        for item in range(group_start[world], group_start[world + 1]):
            art = group_art[item]
            group = art_group[art]
            for row in range(lane, queue.count[world], _row_stride()):
                for i in range(size):
                    value = J[group, row, i]
                    for k in range(i):
                        value -= L[group, i, k] * Y[group, row, k]
                    diagonal = L[group, i, i]
                    Y[group, row, i] = value / diagonal if diagonal != 0.0 else 0.0
                for reverse in range(size):
                    i = size - 1 - reverse
                    value = Y[group, row, i]
                    for k in range(i + 1, size):
                        value -= L[group, k, i] * Y[group, row, k]
                    diagonal = L[group, i, i]
                    Y[group, row, i] = value / diagonal if diagonal != 0.0 else 0.0
                for i in range(size):
                    J_world[world, row, world_offset[art] + i] = J[group, row, i]
                    Y_world[world, row, world_offset[art] + i] = Y[group, row, i]


@wp.kernel
def general_diag(
    workers: int,
    prefix: PrefixData,
    queue: QueueData,
    world_dofs: wp.array[int],
    J: wp.array3d[float],
    Y: wp.array3d[float],
):
    """Publish the general response diagonal plus CFM; private diagonals stay seeded."""
    worker, lane = wp.tid()
    for entry in range(worker, queue.general_count[0], workers):
        world = queue.general[entry]
        for row in range(lane, queue.count[world], _row_stride()):
            value = float(0.0)
            for dof in range(world_dofs[world]):
                value += J[world, row, dof] * Y[world, row, dof]
            prefix.diag[world, row] = value + prefix.row_cfm[world, row]


@wp.func
def contact_points(data: ContactInput, contact: int):
    """Apply the body-local point transforms, margins and normal sign of the ordinary producer."""
    body0 = int(-1)
    body1 = int(-1)
    if data.shape0[contact] >= 0:
        body0 = data.shape_body[data.shape0[contact]]
    if data.shape1[contact] >= 0:
        body1 = data.shape_body[data.shape1[contact]]
    normal = -data.normal[contact]
    point0 = data.point0[contact] - data.margin0[contact] * normal
    point1 = data.point1[contact] + data.margin1[contact] * normal
    if body0 >= 0:
        point0 = wp.transform_point(data.body_q[body0], data.point0[contact]) - data.margin0[contact] * normal
    if body1 >= 0:
        point1 = wp.transform_point(data.body_q[body1], data.point1[contact]) + data.margin1[contact] * normal
    return body0, body1, point0, point1


@wp.func
def endpoint_projection(
    data: ContactInput, body: int, art: int, size: int, dof: int, point: wp.vec3, direction: wp.vec3
):
    """Read current response masks and current world-origin motion columns."""
    value = float(0.0)
    if body >= 0 and art >= 0 and data.response_dofs[art] == size:
        if (data.body_mask[body] & (wp.uint32(1) << wp.uint32(dof))) != wp.uint32(0):
            motion = data.motion[data.dof_start[art] + dof]
            velocity = wp.spatial_top(motion) + wp.cross(wp.spatial_bottom(motion), point - data.origin[art])
            value = wp.dot(direction, velocity)
    return value


@wp.func
def incident_projection(data: ContactInput, body: int, art: int, point: wp.vec3, direction: wp.vec3):
    """Use the frozen incident velocity, not the live output velocity."""
    result = float(0.0)
    if art >= 0 and body >= 0:
        size = data.response_dofs[art]
        for dof in range(size):
            result += (
                endpoint_projection(data, body, art, size, dof, point, direction)
                * data.incident[data.dof_start[art] + dof]
            )
    return result


@wp.kernel(enable_backward=False)
def produce_contacts(workers: int, data: ContactInput, packet: ContactPacketData, primary: int, secondary: int):
    """Cooperatively project current contacts into physical packets after allocation."""
    worker, _logical_lane = wp.tid()
    lane = _lane()
    stride = _stride_value()
    capacity = packet.jacobian.shape[1]
    width = primary + secondary
    total = wp.min(data.count[0], data.point0.shape[0])
    for contact in range(worker, total, workers):
        if data.path[contact] != 0 or data.slot[contact] < 0:
            continue
        world = data.world[contact]
        slot = data.slot[contact]
        rows = data.slots_needed[contact]
        art0 = data.art0[contact]
        art1 = data.art1[contact]
        body0 = int(-1)
        body1 = int(-1)
        if data.shape0[contact] >= 0:
            body0 = data.shape_body[data.shape0[contact]]
        if data.shape1[contact] >= 0:
            body1 = data.shape_body[data.shape1[contact]]
        normal = -data.normal[contact]
        point0 = wp.vec3()
        point1 = wp.vec3()
        tangent0 = wp.vec3()
        tangent1 = wp.vec3()
        if lane == 0:
            _body0, _body1, point0, point1 = contact_points(data, contact)
            tangent0, tangent1 = contact_tangent_basis(normal)
        point0 = _broadcast(point0)
        point1 = _broadcast(point1)
        tangent0 = _broadcast(tangent0)
        tangent1 = _broadcast(tangent1)
        anchor = 0.5 * (point0 + point1)
        for coefficient in range(lane, rows * width, stride):
            row = coefficient // width
            coordinate = coefficient % width
            output_row = slot + row
            if output_row < capacity:
                size = primary
                dof = coordinate
                if coordinate >= primary:
                    size = secondary
                    dof -= primary
                direction = normal
                p0 = point0
                p1 = point1
                if row > 0:
                    direction = tangent0 if row == 1 else tangent1
                if data.shared_anchor != 0 or (row > 0 and data.friction_shared_anchor != 0):
                    p0 = anchor
                    p1 = anchor
                value = endpoint_projection(data, body0, art0, size, dof, p0, direction)
                value -= endpoint_projection(data, body1, art1, size, dof, p1, direction)
                packet.jacobian[world, output_row, coordinate] = value
        _join()
        if lane == 0:
            mu = float(0.0)
            materials = int(0)
            if data.shape0[contact] >= 0:
                mu += data.material_mu[data.shape0[contact]]
                materials += 1
            if data.shape1[contact] >= 0:
                mu += data.material_mu[data.shape1[contact]]
                materials += 1
            if materials > 0:
                mu /= float(materials)
            restitution = mixed_contact_restitution(
                data.shape0[contact], data.shape1[contact], data.material_restitution
            )
            friction_mu = mu * data.friction_scale
            separation = wp.dot(normal, point0 - point1)
            for row in range(rows):
                output_row = slot + row
                direction = normal
                p0 = point0
                p1 = point1
                if row > 0:
                    direction = tangent0 if row == 1 else tangent1
                if data.shared_anchor != 0 or (row > 0 and data.friction_shared_anchor != 0):
                    p0 = anchor
                    p1 = anchor
                target = prescribed_relative_contact_target(
                    body0, art0, body1, art1, p0, p1, direction, data.prescribed, data.origin, data.body_v
                )
                phi = float(0.0)
                beta = float(0.0)
                row_restitution = float(0.0)
                row_mu = friction_mu
                kind = int(2)
                parent = slot
                rhs = -target
                weight = float(1.0)
                if row == 0:
                    phi = separation
                    beta = data.beta
                    row_restitution = restitution
                    row_mu = mu
                    kind = 0
                    parent = -1
                    if phi <= 0.0:
                        rhs += data.bias_scale * beta * phi / data.dt
                        weight = data.contact_w
                    else:
                        rhs += data.speculative_scale * phi / data.dt
                    if restitution > 0.0:
                        relative = incident_projection(data, body0, art0, p0, direction)
                        relative -= incident_projection(data, body1, art1, p1, direction)
                        relative -= target
                        if contact_restitution_fires(phi, relative, data.dt, data.restitution_threshold):
                            rhs = -target + restitution * relative
                            weight = 1.0
                data.row_type[world, output_row] = kind
                data.row_parent[world, output_row] = parent
                data.row_mu[world, output_row] = row_mu
                data.row_beta[world, output_row] = beta
                data.row_cfm[world, output_row] = data.cfm
                data.phi[world, output_row] = phi
                data.target[world, output_row] = target
                data.restitution[world, output_row] = row_restitution
                data.rhs[world, output_row] = rhs
                data.diag[world, output_row] = data.cfm
                if data.contact_w < 1.0:
                    data.row_w[world, output_row] = weight
                if output_row < capacity:
                    packet.row_contact[world, output_row] = contact * 3 + row
        _join()


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return 32;
#else
return 1;
#endif
""")
def _stride_value() -> int: ...


@wp.kernel(enable_backward=False)
def materialize_fallback(
    workers: int,
    data: ContactInput,
    owner: wp.array[int],
    target_size: int,
    groups: wp.array[int],
    J: wp.array3d[float],
):
    """Reproject every general-world endpoint, including rows outside the local bound."""
    worker, _logical_lane = wp.tid()
    lane = _lane()
    stride = _stride_value()
    total = wp.min(data.count[0], data.point0.shape[0])
    for contact in range(worker, total, workers):
        if data.path[contact] != 0 or data.slot[contact] < 0:
            continue
        world = data.world[contact]
        if owner[world] != 0:
            continue
        body0, body1, point0, point1 = contact_points(data, contact)
        normal = -data.normal[contact]
        tangent0, tangent1 = contact_tangent_basis(normal)
        anchor = 0.5 * (point0 + point1)
        art0, art1 = data.art0[contact], data.art1[contact]
        group0, group1 = int(-1), int(-1)
        if art0 >= 0 and data.response_dofs[art0] == target_size:
            group0 = groups[art0]
        if art1 >= 0 and data.response_dofs[art1] == target_size:
            group1 = groups[art1]
        for coefficient in range(lane, data.slots_needed[contact] * target_size, stride):
            row = coefficient // target_size
            dof = coefficient % target_size
            direction = normal
            p0, p1 = point0, point1
            if row > 0:
                direction = tangent0 if row == 1 else tangent1
            if data.shared_anchor != 0 or (row > 0 and data.friction_shared_anchor != 0):
                p0, p1 = anchor, anchor
            value0 = endpoint_projection(data, body0, art0, target_size, dof, p0, direction)
            value1 = -endpoint_projection(data, body1, art1, target_size, dof, p1, direction)
            output_row = data.slot[contact] + row
            if group0 >= 0:
                J[group0, output_row, dof] = value0 + value1 if group0 == group1 else value0
            if group1 >= 0 and group1 != group0:
                J[group1, output_row, dof] = value1


@wp.kernel(enable_backward=False)
def produce_patch_contacts(
    workers: int, data: ContactInput, law: PatchRowLaw, packet: ContactPacketData, primary: int, secondary: int
):
    """Project current contacts into physical packets with the persistent patch friction law.

    The normal row's metadata and packet are those of :func:`produce_contacts`. Tangent rows, present
    only when the contact owns an anchor, follow the patch row law that the compact producer shares
    (``patch_row_frame`` and ``patch_tangent_row``): they act at the patch friction location, carry
    the anchor's tangential phi with ``law.beta`` and keep the undivided coefficient. Every bias takes
    ``compute_world_contact_bias``'s form, ``x * (1 / dt)``, where :func:`produce_contacts` forms
    ``x / dt``, so the rows equal the general route's bitwise. A bouncing normal row sums its incident
    velocity per body, as :func:`produce_contacts` does, instead of per world coordinate. It agrees
    with the general route's to rounding, unless the incident velocity lies within a rounding of a
    bounce threshold, where the two routes can disagree on whether the row bounces.
    """
    worker, _logical_lane = wp.tid()
    lane = _lane()
    stride = _stride_value()
    capacity = packet.jacobian.shape[1]
    width = primary + secondary
    total = wp.min(data.count[0], data.point0.shape[0])
    for contact in range(worker, total, workers):
        if data.path[contact] != 0 or data.slot[contact] < 0:
            continue
        world = data.world[contact]
        slot = data.slot[contact]
        rows = data.slots_needed[contact]
        art0 = data.art0[contact]
        art1 = data.art1[contact]
        body0 = int(-1)
        body1 = int(-1)
        if data.shape0[contact] >= 0:
            body0 = data.shape_body[data.shape0[contact]]
        if data.shape1[contact] >= 0:
            body1 = data.shape_body[data.shape1[contact]]
        normal = -data.normal[contact]
        point0 = wp.vec3()
        point1 = wp.vec3()
        tangent0 = wp.vec3()
        tangent1 = wp.vec3()
        if lane == 0:
            _body0, _body1, point0, point1 = contact_points(data, contact)
            tangent0, tangent1 = contact_tangent_basis(normal)
        point0 = _broadcast(point0)
        point1 = _broadcast(point1)
        tangent0 = _broadcast(tangent0)
        tangent1 = _broadcast(tangent1)
        friction0, friction1 = patch_friction_location(law.patches, contact)
        for coefficient in range(lane, rows * width, stride):
            row = coefficient // width
            coordinate = coefficient % width
            output_row = slot + row
            if output_row < capacity:
                size = primary
                dof = coordinate
                if coordinate >= primary:
                    size = secondary
                    dof -= primary
                direction, p0, p1 = patch_row_frame(
                    row, data.shared_anchor, normal, tangent0, tangent1, point0, point1, friction0, friction1
                )
                value = endpoint_projection(data, body0, art0, size, dof, p0, direction)
                value -= endpoint_projection(data, body1, art1, size, dof, p1, direction)
                packet.jacobian[world, output_row, coordinate] = value
        _join()
        if lane == 0:
            mu = float(0.0)
            materials = int(0)
            if data.shape0[contact] >= 0:
                mu += data.material_mu[data.shape0[contact]]
                materials += 1
            if data.shape1[contact] >= 0:
                mu += data.material_mu[data.shape1[contact]]
                materials += 1
            if materials > 0:
                mu /= float(materials)
            restitution = mixed_contact_restitution(
                data.shape0[contact], data.shape1[contact], data.material_restitution
            )
            separation = wp.dot(normal, point0 - point1)
            inv_dt = 1.0 / data.dt
            for row in range(rows):
                output_row = slot + row
                direction, p0, p1 = patch_row_frame(
                    row, data.shared_anchor, normal, tangent0, tangent1, point0, point1, friction0, friction1
                )
                target = prescribed_relative_contact_target(
                    body0, art0, body1, art1, p0, p1, direction, data.prescribed, data.origin, data.body_v
                )
                phi = separation
                beta = data.beta
                row_restitution = restitution
                row_mu = mu
                kind = int(PGS_CONSTRAINT_TYPE_CONTACT)
                parent = int(-1)
                rhs = -target
                weight = float(1.0)
                if row == 0:
                    if phi <= 0.0:
                        rhs += data.bias_scale * beta * phi * inv_dt
                        weight = data.contact_w
                    else:
                        rhs += data.speculative_scale * phi * inv_dt
                    if restitution > 0.0:
                        relative = incident_projection(data, body0, art0, p0, direction)
                        relative -= incident_projection(data, body1, art1, p1, direction)
                        relative -= target
                        if contact_restitution_fires(phi, relative, data.dt, data.restitution_threshold):
                            rhs = -target + restitution * relative
                            weight = 1.0
                else:
                    phi, beta, row_mu = patch_tangent_row(law, contact, row, mu, data.friction_scale)
                    row_restitution = 0.0
                    kind = int(PGS_CONSTRAINT_TYPE_FRICTION)
                    parent = slot
                    rhs += data.bias_scale * beta * phi * inv_dt
                data.row_type[world, output_row] = kind
                data.row_parent[world, output_row] = parent
                data.row_mu[world, output_row] = row_mu
                data.row_beta[world, output_row] = beta
                data.row_cfm[world, output_row] = data.cfm
                data.phi[world, output_row] = phi
                data.target[world, output_row] = target
                data.restitution[world, output_row] = row_restitution
                data.rhs[world, output_row] = rhs
                data.diag[world, output_row] = data.cfm
                if data.contact_w < 1.0:
                    data.row_w[world, output_row] = weight
                if output_row < capacity:
                    packet.row_contact[world, output_row] = contact * 3 + row
        _join()


@wp.kernel(enable_backward=False)
def materialize_patch_fallback(
    workers: int,
    data: ContactInput,
    law: PatchRowLaw,
    owner: wp.array[int],
    target_size: int,
    groups: wp.array[int],
    J: wp.array3d[float],
):
    """Reproject every general-world endpoint as :func:`materialize_fallback` does, with the patch law."""
    worker, _logical_lane = wp.tid()
    lane = _lane()
    stride = _stride_value()
    total = wp.min(data.count[0], data.point0.shape[0])
    for contact in range(worker, total, workers):
        if data.path[contact] != 0 or data.slot[contact] < 0:
            continue
        world = data.world[contact]
        if owner[world] != 0:
            continue
        body0, body1, point0, point1 = contact_points(data, contact)
        normal = -data.normal[contact]
        tangent0, tangent1 = contact_tangent_basis(normal)
        friction0, friction1 = patch_friction_location(law.patches, contact)
        art0, art1 = data.art0[contact], data.art1[contact]
        group0, group1 = int(-1), int(-1)
        if art0 >= 0 and data.response_dofs[art0] == target_size:
            group0 = groups[art0]
        if art1 >= 0 and data.response_dofs[art1] == target_size:
            group1 = groups[art1]
        for coefficient in range(lane, data.slots_needed[contact] * target_size, stride):
            row = coefficient // target_size
            dof = coefficient % target_size
            direction, p0, p1 = patch_row_frame(
                row, data.shared_anchor, normal, tangent0, tangent1, point0, point1, friction0, friction1
            )
            value0 = endpoint_projection(data, body0, art0, target_size, dof, p0, direction)
            value1 = -endpoint_projection(data, body1, art1, target_size, dof, p1, direction)
            output_row = data.slot[contact] + row
            if group0 >= 0:
                J[group0, output_row, dof] = value0 + value1 if group0 == group1 else value0
            if group1 >= 0 and group1 != group0:
                J[group1, output_row, dof] = value1


def topology_signature(solver):
    """Pin only structural indices; enabled flags, coefficients, q and limits stay live."""
    names = (
        "constraint_mimic_joint0",
        "constraint_mimic_joint1",
        "constraint_mimic_world",
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_q_start",
        "joint_qd_start",
        "body_flags",
    )
    return tuple((name, getattr(solver.model, name).numpy().tobytes()) for name in names)


def plan(solver):
    """Derive (primary, secondary, mimic capacity) from the topology, or None.

    The primary is the one articulation size that carries joint limits; every
    world has exactly one of it. A second size, if present, is the world's
    secondary articulation. Mimic rows belong to the primary alone.
    """
    sizes = sorted(int(size) for size in solver.size_groups)
    if not 1 <= len(sizes) <= 2 or len(solver._joint_limit_sizes) != 1:
        return None
    primary = int(next(iter(solver._joint_limit_sizes)))
    if primary not in sizes or solver.n_arts_by_size[primary] != solver.world_count:
        return None
    secondary = next((size for size in sizes if size != primary), 0)
    if secondary and solver.n_arts_by_size[secondary] != solver.world_count:
        return None
    if solver._mimic_sizes - {primary} or solver._connect_count:
        return None
    mimics = 0
    if solver._mimic_art_start is not None and solver._mimic_art_list is not None:
        starts = solver._mimic_art_start.numpy()
        mimics = int(np.max(np.diff(starts))) if len(starts) > 1 else 0
    if mimics + 2 * primary > solver._local_residual_max_rows:
        return None
    return primary, secondary, mimics


def topology_supported(solver, primary, mimics):
    """Check all prefix entries, not only currently enabled or valid mimic rows."""
    if any(
        primary not in getattr(solver, name) for name in ("_local_pair_active_counts", "_local_residual_active_counts")
    ):
        return False
    if np.any(solver._local_primary_articulation.numpy() < 0):
        return False
    if solver._mimic_art_start is None or solver._mimic_art_list is None:
        return mimics == 0
    return validate_prefix_topology(
        solver.group_to_art[primary].numpy(),
        solver.art_to_world.numpy(),
        solver.articulation_dof_start.numpy(),
        solver._mimic_art_start.numpy(),
        solver._mimic_art_list.numpy(),
        solver._mimic_dof0.numpy(),
        solver._mimic_dof1.numpy(),
        primary,
        mimics,
        solver._mimic_world.numpy(),
    )


class LocalRowPackets:
    """Own one bounded prefix/current-row boundary; all general rows stay complete."""

    def __init__(self, solver, primary, secondary, mimics):
        self.solver = solver
        self.primary, self.secondary, self.mimics = int(primary), int(secondary), int(mimics)
        self.packets = int(solver._local_residual_max_rows)
        device = solver.model.device
        worlds = solver.world_count
        self.workers = max(1, min(worlds, int(device.sm_count) * 4))
        self.prefix = PrefixData()
        self.prefix.dofs = wp.empty((worlds, self.packets, PREFIX_WIDTH), dtype=int, device=device)
        self.prefix.weights = wp.empty((worlds, self.packets, PREFIX_WIDTH), dtype=float, device=device)
        self.packet = ContactPacketData()
        self.packet.jacobian = wp.zeros(
            (worlds, self.packets, self.primary + self.secondary), dtype=float, device=device
        )
        self.packet.row_contact = wp.full((worlds, self.packets), -1, dtype=wp.int32, device=device)
        self.prefix.row_contact = self.packet.row_contact
        self._empty_int = wp.zeros(1, dtype=int, device=device)
        self._empty_float = wp.zeros(1, dtype=float, device=device)
        self._empty_bool = wp.zeros(1, dtype=wp.bool, device=device)
        self._no_mimics = wp.zeros(solver.model.articulation_count + 1, dtype=int, device=device)
        self.queue = QueueData()
        for name, source in (
            ("primary", "_local_primary_articulation"),
            ("secondary", "_local_pair_articulation"),
            ("residual_secondary", "_local_residual_pair_articulation"),
            ("mf_count", "mf_constraint_count"),
            ("mf_body0", "mf_body_a"),
            ("mf_body1", "mf_body_b"),
            ("mf_row_type", "mf_row_type"),
            ("body_art", "body_to_articulation"),
            ("count", "constraint_count"),
            ("owner", "_local_solve_owner"),
            ("general_count", "_local_general_world_count"),
            ("general", "_local_general_worlds"),
        ):
            setattr(self.queue, name, getattr(solver, source))
        for name, source in (
            ("pair_count", "_local_pair_active_counts"),
            ("pair", "_local_pair_active_candidates"),
            ("pair_secondary", "_local_pair_active_secondaries"),
            ("residual_count", "_local_residual_active_counts"),
            ("residual", "_local_residual_active_candidates"),
            ("residual_pair", "_local_residual_active_secondaries"),
        ):
            setattr(self.queue, name, getattr(solver, source)[self.primary])
        # The friction law is fixed at construction; produce_contacts binds its current arrays.
        self.patch_friction = bool(solver._friction_anchors_enabled)
        self.contact_input = None
        self.law = None
        self._topology_signature = topology_signature(solver)
        self._prefix_kernel = get_prefix_kernel(self.primary, self.mimics, self.packets)

    def local_factory(self):
        return get_local_factory(self.primary, self.secondary, self.packets)

    def row_law(self):
        """Bind the persistent patch friction row law, or return ``None`` for point friction."""
        if not self.patch_friction:
            return None
        law = PatchRowLaw()
        law.patches = self.solver._friction_patches.view
        law.beta = self.solver.friction_anchor_beta
        return law

    def validate_notification(self, flags: ModelFlags | int = ModelFlags.ALL):
        """Validate structural changes without reading back numeric-only updates."""
        numeric_flags = (
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        if flags and not (int(flags) & ~int(numeric_flags)):
            return
        if topology_signature(self.solver) != self._topology_signature or not topology_supported(
            self.solver, self.primary, self.mimics
        ):
            raise RuntimeError("Row-packet topology changed; reconstruct the solver to select a complete fallback")

    def prepare_prefix(self, state_in, dt):
        """Bind current coefficients/state and the current double-buffered row owners."""
        solver, data = self.solver, self.prefix
        # Sticky status is allocated after tiled-kernel construction.
        self.queue.status = solver._constraint_capacity_status
        for name, source in (
            ("art_to_world", "art_to_world"),
            ("dof_start", "articulation_dof_start"),
            ("limit_q_index", "_joint_limit_q_index"),
            ("slot_counter", "slot_counter"),
            ("first_rejected_slot", "_dense_first_rejected_slot"),
            ("bounds", "dense_phase_bounds"),
            ("kind", "row_type"),
            ("parent", "row_parent"),
            ("mu", "row_mu"),
            ("row_beta", "row_beta"),
            ("row_cfm", "row_cfm"),
            ("phi", "phi"),
            ("target", "target_velocity"),
            ("restitution", "row_restitution"),
            ("rhs", "rhs"),
            ("diag", "diag"),
            ("impulses", "impulses"),
        ):
            setattr(data, name, getattr(solver, source))
        data.group_to_art = solver.group_to_art[self.primary]
        if self.mimics:
            for name, source in (
                ("mimic_start", "_mimic_art_start"),
                ("mimic_list", "_mimic_art_list"),
                ("mimic_valid", "_mimic_valid"),
                ("mimic_dof0", "_mimic_dof0"),
                ("mimic_dof1", "_mimic_dof1"),
                ("mimic_q0", "_mimic_q0"),
                ("mimic_q1", "_mimic_q1"),
                ("mimic_slot", "mimic_slot"),
            ):
                setattr(data, name, getattr(solver, source))
            for name, source in (
                ("mimic_enabled", "constraint_mimic_enabled"),
                ("mimic_coef0", "constraint_mimic_coef0"),
                ("mimic_coef1", "constraint_mimic_coef1"),
            ):
                setattr(data, name, getattr(solver.model, source))
        else:
            data.mimic_start = self._no_mimics
            for name in ("mimic_list", "mimic_valid", "mimic_dof0", "mimic_dof1", "mimic_q0", "mimic_q1", "mimic_slot"):
                setattr(data, name, self._empty_int)
            data.mimic_enabled = self._empty_bool
            data.mimic_coef0 = data.mimic_coef1 = self._empty_float
        data.lower, data.upper = solver.model.joint_limit_lower, solver.model.joint_limit_upper
        data.q, data.dt = state_in.joint_q, dt
        data.activation_gap, data.beta, data.cfm = solver.joint_limit_activation_gap, solver.pgs_beta, solver.pgs_cfm
        data.bias_scale, data.limit_speculative_scale = 1.0, 1.0
        self.contact_input = None
        self.law = None
        wp.launch_tiled(
            self._prefix_kernel,
            dim=[(solver.n_arts_by_size[self.primary] + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK],
            inputs=[data],
            block_dim=32 * WARPS_PER_BLOCK,
            device=solver.model.device,
        )

    def produce_contacts(self, state_in, state_aug, contacts, dt):
        """Publish private current J and complete canonical contact metadata/RHS.

        Persistent patch friction writes its tangent rows with the patch row law. The patch link
        runs after this producer; :meth:`finish_rows` writes neither row parents nor coefficients.
        """
        self.contact_input = bind(self.solver, state_in, state_aug, contacts, dt, 1.0)
        self.law = self.row_law()
        kernel, law = (produce_contacts, ()) if self.law is None else (produce_patch_contacts, (self.law,))
        wp.launch_tiled(
            kernel,
            dim=[self.workers],
            inputs=[self.workers, self.contact_input, *law, self.packet, self.primary, self.secondary],
            block_dim=32,
            device=self.solver.model.device,
        )

    def finish_rows(self):
        """Finalize and route current counts, then reconstruct only the general path."""
        solver, queue = self.solver, self.queue
        queue.general_count.zero_()
        queue.pair_count.zero_()
        queue.residual_count.zero_()
        wp.launch(
            finalize_and_queue,
            dim=solver.world_count,
            inputs=[
                self.prefix,
                queue,
                int(solver._local_solve_max_rows),
                int(solver._local_residual_max_rows),
                int(solver._local_residual_mf_max_rows),
                self.primary,
            ],
            device=solver.model.device,
        )
        for size in solver.size_groups:
            wp.launch_tiled(
                materialize_general_prefix,
                dim=[self.workers],
                inputs=[
                    self.workers,
                    self.prefix,
                    queue,
                    solver.world_response_group_art_start[size],
                    solver.world_response_group_to_art[size],
                    solver.art_group_idx,
                    size,
                    solver.J_by_size[size],
                ],
                block_dim=32,
                device=solver.model.device,
            )
            if self.contact_input is not None:
                kernel, law = (
                    (materialize_fallback, ()) if self.law is None else (materialize_patch_fallback, (self.law,))
                )
                wp.launch_tiled(
                    kernel,
                    dim=[self.workers],
                    inputs=[
                        self.workers,
                        self.contact_input,
                        *law,
                        queue.owner,
                        size,
                        solver.art_group_idx,
                        solver.J_by_size[size],
                    ],
                    block_dim=32,
                    device=solver.model.device,
                )

    def response(self, size):
        """Run the held-L response on the compact general-world queue."""
        solver = self.solver
        wp.launch_tiled(
            general_response,
            dim=[self.workers],
            inputs=[
                self.workers,
                self.queue,
                solver.world_response_group_art_start[size],
                solver.world_response_group_to_art[size],
                solver.art_group_idx,
                solver.articulation_world_dof_offset,
                size,
                solver.L_by_size[size],
                solver.J_by_size[size],
                solver.Y_by_size[size],
                solver.J_world,
                solver.Y_world,
            ],
            block_dim=32,
            device=solver.model.device,
        )

    def diagonal(self):
        """Keep private CFM seeds and finalize only complete general response diagonals."""
        solver = self.solver
        wp.launch_tiled(
            general_diag,
            dim=[self.workers],
            inputs=[self.workers, self.prefix, self.queue, solver.world_dof_count, solver.J_world, solver.Y_world],
            block_dim=32,
            device=solver.model.device,
        )


def create_owner(solver):
    """Admit the local packet law for a topology that fits it; unsupported solvers stay ordinary.

    The owner writes the solver's friction law: point rows, or persistent patch rows when friction
    anchors are on. The solver admits neither contact torsion nor compliance here.
    """
    from . import solver_feather_pgs as source  # noqa: PLC0415

    if not (
        solver._local_internal_fast_path
        and solver._hinv_jt_writes_world
        and solver.model.device.is_cuda
        and not solver.model.requires_grad
        and solver.dense_max_constraints >= solver._local_residual_max_rows
        and solver.pgs_mode == "matrix_free"
        and solver.drive_mode == "augmented"
        and solver.pgs_schedule == "interleaved"
        and solver.friction_mode == "current"
        and not solver.enable_joint_velocity_limits
        and solver.enable_joint_limits
        and solver.pgs_velocity_iterations == 0
        and not solver.pgs_warmstart
        and not solver._mf_warmstart_enabled
        and not solver._regularization_enabled
        and not solver._preelim_active
        and not solver.grouped_dynamics
        and not solver._debug_buffers_enabled
        and not solver._sparse_diagonal_contact_solve
        and not source._COMPACT_CONTACT_BOUNDARY
        and not solver._paired_factor_coordinates
        and not solver._fused_diagonal_joint_limits
        and all(
            not solver._execution_plan.use_tiled_hinv_jt(size) and not solver._execution_plan.use_diagonal_mass(size)
            for size in solver.size_groups
        )
    ):
        return None
    if source._INK_ON:
        return None
    layout = plan(solver)
    if layout is None:
        return None
    primary, secondary, mimics = layout
    if not topology_supported(solver, primary, mimics):
        return None
    return LocalRowPackets(solver, primary, secondary, mimics)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private current-contact packets for the experimental Franka local owner."""

import warp as wp

from .kernels import (
    contact_restitution_fires,
    contact_tangent_basis,
    mixed_contact_restitution,
    prescribed_relative_contact_target,
)


@wp.struct
class ContactPacketData:
    jacobian: wp.array3d[float]
    row_contact: wp.array2d[int]


@wp.struct
class ContactInput:
    count: wp.array[int]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    shape0: wp.array[int]
    shape1: wp.array[int]
    margin0: wp.array[float]
    margin1: wp.array[float]
    world: wp.array[int]
    slot: wp.array[int]
    art0: wp.array[int]
    art1: wp.array[int]
    path: wp.array[int]
    slots_needed: wp.array[int]
    shape_body: wp.array[int]
    body_q: wp.array[wp.transform]
    body_v: wp.array[wp.spatial_vector]
    body_mask: wp.array[wp.uint32]
    response_dofs: wp.array[int]
    dof_start: wp.array[int]
    origin: wp.array[wp.vec3]
    motion: wp.array[wp.spatial_vector]
    prescribed: wp.array[int]
    is_free: wp.array[int]
    material_mu: wp.array[float]
    material_restitution: wp.array[float]
    incident: wp.array[float]
    shared_anchor: int
    friction_shared_anchor: int
    friction_anchor_limit: int
    friction_pairs_only: int
    friction_scale: float
    beta: float
    cfm: float
    dt: float
    bias_scale: float
    speculative_scale: float
    restitution_threshold: float
    contact_w: float
    row_type: wp.array2d[int]
    row_parent: wp.array2d[int]
    row_mu: wp.array2d[float]
    row_beta: wp.array2d[float]
    row_cfm: wp.array2d[float]
    phi: wp.array2d[float]
    target: wp.array2d[float]
    restitution: wp.array2d[float]
    rhs: wp.array2d[float]
    diag: wp.array2d[float]
    row_w: wp.array2d[float]


def allocate_packet(worlds, device, *, row_contact=None):
    """Bound private storage by the existing 40-row local owner, not raw capacity."""
    if worlds < 0:
        raise ValueError("Packet capacities must be nonnegative")
    result = ContactPacketData()
    result.jacobian = wp.zeros((worlds, 40, 15), dtype=wp.float32, device=device)
    result.row_contact = (
        row_contact if row_contact is not None else wp.full((worlds, 40), -1, dtype=wp.int32, device=device)
    )
    return result


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
return 32;
#else
return 1;
#endif
""")
def _stride() -> int: ...


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


@wp.func
def contact_points(data: ContactInput, contact: int):
    """Apply the original body-local point transforms, margins and normal sign."""
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
    """Use the original frozen incident velocity, not live output velocity."""
    result = float(0.0)
    if art >= 0 and body >= 0:
        size = data.response_dofs[art]
        for dof in range(size):
            result += (
                endpoint_projection(data, body, art, size, dof, point, direction)
                * data.incident[data.dof_start[art] + dof]
            )
    return result


@wp.func
def friction_multiplier(data: ContactInput, contact: int, total: int, art0: int, art1: int):
    """Keep original adjacent-anchor scaling without re-admitting allocated rows."""
    apply_filter = data.friction_pairs_only == 0
    if art0 >= 0 and art1 >= 0:
        apply_filter = apply_filter or (data.is_free[art0] == 0 and data.is_free[art1] == 0)
    result = data.friction_scale
    if apply_filter and data.friction_anchor_limit > 0:
        rank = int(0)
        for lookback in range(1, 9):
            previous = contact - lookback
            if previous < 0 or previous >= total:
                break
            if data.shape0[previous] == data.shape0[contact] and data.shape1[previous] == data.shape1[contact]:
                rank += 1
            else:
                break
        same_next = False
        if contact + 1 < total:
            same_next = (
                data.shape0[contact + 1] == data.shape0[contact] and data.shape1[contact + 1] == data.shape1[contact]
            )
        if rank > 0 or same_next:
            result *= 0.5
    return result


@wp.kernel(enable_backward=False)
def produce_contacts(workers: int, data: ContactInput, packet: ContactPacketData):
    """Cooperatively project current contacts after original successful allocation."""
    worker, _logical_lane = wp.tid()
    lane = _lane()
    stride = _stride()
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
        for coefficient in range(lane, rows * 15, stride):
            row = coefficient // 15
            coordinate = coefficient % 15
            output_row = slot + row
            if output_row < 40:
                size = int(9)
                dof = coordinate
                if coordinate >= 9:
                    size = 6
                    dof -= 9
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
            friction_mu = mu * friction_multiplier(data, contact, total, art0, art1)
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
                if output_row < 40:
                    packet.row_contact[world, output_row] = contact * 3 + row
        _join()


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
    stride = _stride()
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


def bind(solver, state_in, state_aug, contacts, dt, bias_scale):
    """Bind original current owners without allocating device storage or reading back."""
    data = ContactInput()
    for name, source in (
        ("count", "count"),
        ("point0", "point0"),
        ("point1", "point1"),
        ("normal", "normal"),
        ("shape0", "shape0"),
        ("shape1", "shape1"),
        ("margin0", "margin0"),
        ("margin1", "margin1"),
    ):
        setattr(data, name, getattr(contacts, "rigid_contact_" + source))
    for name, source in (
        ("world", "contact_world"),
        ("slot", "contact_slot"),
        ("art0", "contact_art_a"),
        ("art1", "contact_art_b"),
        ("path", "contact_path"),
        ("slots_needed", "contact_slots_needed"),
        ("body_mask", "body_response_dof_mask"),
        ("response_dofs", "articulation_response_dof_count"),
        ("dof_start", "articulation_dof_start"),
        ("origin", "articulation_origin"),
        ("prescribed", "_prescribed_articulation"),
        ("is_free", "is_free_rigid"),
        ("material_mu", "shape_material_mu"),
        ("material_restitution", "shape_material_restitution"),
        ("incident", "v_hat"),
        ("row_type", "row_type"),
        ("row_parent", "row_parent"),
        ("row_mu", "row_mu"),
        ("row_beta", "row_beta"),
        ("row_cfm", "row_cfm"),
        ("phi", "phi"),
        ("target", "target_velocity"),
        ("restitution", "row_restitution"),
        ("rhs", "rhs"),
        ("diag", "diag"),
        ("row_w", "row_w"),
    ):
        setattr(data, name, getattr(solver, source))
    data.shape_body = solver.model.shape_body
    data.body_q = state_in.body_q
    data.body_v = state_aug.body_v_s
    data.motion = state_aug.joint_S_s
    data.shared_anchor = int(solver.contact_shared_anchor)
    data.friction_shared_anchor = int(solver.contact_friction_shared_anchor)
    data.friction_anchor_limit = solver.contact_friction_anchor_limit
    data.friction_pairs_only = int(solver.contact_friction_articulation_pairs_only)
    data.friction_scale = solver.contact_friction_scale
    data.beta, data.cfm = solver.pgs_beta, solver.pgs_cfm
    data.dt, data.bias_scale = dt, bias_scale
    data.speculative_scale = solver.contact_speculative_scale
    data.restitution_threshold = solver._effective_restitution_velocity_threshold
    data.contact_w = solver._contact_w
    return data

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental complete contact owner for the existing dense-six + sparse solver.

Allocation, contact admission, scheduling, RHS and the solve remain authoritative
in the original pipeline. This module only changes the representation producer.
"""

import warp as wp

from .kernels import contact_tangent_basis, mixed_contact_restitution, prescribed_relative_contact_target

_Vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.struct
class ContactBoundaryData:
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
    body_single_dof: wp.array[int]
    response_dofs: wp.array[int]
    dof_start: wp.array[int]
    dof_offset: wp.array[int]
    origin: wp.array[wp.vec3]
    motion: wp.array[wp.spatial_vector]
    prescribed: wp.array[int]
    material_mu: wp.array[float]
    material_restitution: wp.array[float]
    inverse_mass: wp.array[float]
    dense_group: wp.array[int]
    factor: wp.array3d[float]
    sparse_size: int
    shared_anchor: int
    friction_shared_anchor: int
    friction_scale: float
    beta: float
    cfm: float
    row_type: wp.array2d[int]
    row_parent: wp.array2d[int]
    row_mu: wp.array2d[float]
    row_beta: wp.array2d[float]
    row_cfm: wp.array2d[float]
    phi: wp.array2d[float]
    target: wp.array2d[float]
    restitution: wp.array2d[float]
    J: wp.array3d[float]
    Y: wp.array3d[float]
    sparse_dof: wp.array3d[int]
    sparse_jy: wp.array3d[float]
    diag: wp.array2d[float]


def supported(solver) -> bool:
    """Require the inherited complete sparse contract and its dense-six triple mode."""
    return bool(
        solver.model.device.is_cuda
        and solver._sparse_diagonal_contact_solve
        and solver._sparse_diagonal_dense_size == 6
        and solver._sparse_diagonal_response_size > 6
        and solver._sparse_diagonal_contact_triples
        and solver.dense_max_constraints >= 12
        # The inherited sparse owner fuses all wide-articulation limits. With
        # augmented (implicit) drives and no other row family, only the dense
        # six position axes remain: at most lower + upper for each axis.
        and solver._fused_diagonal_joint_limits
        and 0 <= solver._dense_internal_max_rows <= 12
        and solver.drive_mode == "augmented"
        and not solver.enable_joint_velocity_limits
        and solver._mimic_count == 0
        and solver._connect_count == 0
        and solver._compact_diagonal_mass_size == solver._sparse_diagonal_response_size
        and not solver.grouped_dynamics
        and not solver._debug_buffers_enabled
        and not solver._preelim_active
        and solver._simple_world_classifier is None
    )


@wp.func
def solve_six(factor: wp.array3d[float], group: int, jacobian: _Vec6):
    response = _Vec6()
    nonzero = False
    for i in range(6):
        nonzero = nonzero or jacobian[i] != 0.0
    if nonzero:
        for i in range(6):
            value = jacobian[i]
            for k in range(i):
                value -= factor[group, i, k] * response[k]
            diagonal = factor[group, i, i]
            if diagonal != 0.0:
                response[i] = value / diagonal
            else:
                response[i] = 0.0
        for reverse in range(6):
            i = 5 - reverse
            value = response[i]
            for k in range(i + 1, 6):
                value -= factor[group, k, i] * response[k]
            diagonal = factor[group, i, i]
            if diagonal != 0.0:
                response[i] = value / diagonal
            else:
                response[i] = 0.0
    return response


@wp.func
def point_projection(motion: wp.spatial_vector, direction: wp.vec3, point: wp.vec3, origin: wp.vec3):
    velocity = wp.spatial_top(motion) + wp.cross(wp.spatial_bottom(motion), point - origin)
    return wp.dot(direction, velocity)


@wp.kernel
def clear_limit_prefix(dense_groups: wp.array[int], J: wp.array3d[float], sparse_dof: wp.array3d[int]):
    """Clear every possible dense-six limit row before partial limit writers run."""
    world, row = wp.tid()
    for dof in range(6):
        J[dense_groups[world], row, dof] = 0.0
    sparse_dof[world, row, 0] = -1
    sparse_dof[world, row, 1] = -1


@wp.kernel
def clear_contact_response(
    counts: wp.array[int],
    bounds: wp.array2d[int],
    groups: wp.array[int],
    J: wp.array3d[float],
    Y: wp.array3d[float],
):
    """Clear only current contact coefficients, with contiguous lanes within each world."""
    world, lane = wp.tid()
    start = wp.max(bounds[world, 1], 0)
    end = wp.min(counts[world], J.shape[1])
    if end <= start:
        return
    group = groups[world]
    # The limit prefix already contains current J. Align the first coefficient
    # tile down for coalescing, but never write its prefix lanes.
    first = (start * 6 // 32) * 32
    for coefficient in range(first + lane, end * 6, 256):
        if coefficient >= start * 6:
            row = coefficient // 6
            dof = coefficient % 6
            J[group, row, dof] = 0.0
            Y[group, row, dof] = 0.0


@wp.func
def produce_contact(data: ContactBoundaryData, contact: int):
    """Publish one allocated contact after the current-contact dense zero owner."""
    if contact >= wp.min(data.count[0], data.point0.shape[0]):
        return
    slot = data.slot[contact]
    if data.path[contact] != 0 or slot < 0:
        return
    world = data.world[contact]
    art0 = data.art0[contact]
    art1 = data.art1[contact]
    shape0 = data.shape0[contact]
    shape1 = data.shape1[contact]
    body0 = -1
    body1 = -1
    if shape0 >= 0:
        body0 = data.shape_body[shape0]
    if shape1 >= 0:
        body1 = data.shape_body[shape1]
    has_dense = False
    if art0 >= 0 and body0 >= 0:
        has_dense = data.response_dofs[art0] == 6
    if art1 >= 0 and body1 >= 0:
        has_dense = has_dense or data.response_dofs[art1] == 6
    normal = -data.normal[contact]
    point0 = data.point0[contact] - data.margin0[contact] * normal
    point1 = data.point1[contact] + data.margin1[contact] * normal
    if body0 >= 0:
        point0 = wp.transform_point(data.body_q[body0], data.point0[contact]) - data.margin0[contact] * normal
    if body1 >= 0:
        point1 = wp.transform_point(data.body_q[body1], data.point1[contact]) + data.margin1[contact] * normal
    phi = wp.dot(normal, point0 - point1)
    mu = float(0.0)
    materials = int(0)
    if shape0 >= 0:
        mu += data.material_mu[shape0]
        materials += 1
    if shape1 >= 0:
        mu += data.material_mu[shape1]
        materials += 1
    if materials > 0:
        mu /= float(materials)
    restitution = mixed_contact_restitution(shape0, shape1, data.material_restitution)
    tangent0, tangent1 = contact_tangent_basis(normal)
    anchor = 0.5 * (point0 + point1)
    group = data.dense_group[world]
    for row in range(data.slots_needed[contact]):
        direction = normal
        p0 = point0
        p1 = point1
        if row == 0:
            if data.shared_anchor != 0:
                p0 = anchor
                p1 = anchor
        else:
            if row == 1:
                direction = tangent0
            else:
                direction = tangent1
            if data.shared_anchor != 0 or data.friction_shared_anchor != 0:
                p0 = anchor
                p1 = anchor
        output_row = slot + row
        target = prescribed_relative_contact_target(
            body0, art0, body1, art1, p0, p1, direction, data.prescribed, data.origin, data.body_v
        )
        data.target[world, output_row] = target
        data.row_cfm[world, output_row] = data.cfm
        if row == 0:
            data.row_type[world, output_row] = 0
            data.row_parent[world, output_row] = -1
            data.row_mu[world, output_row] = mu
            data.row_beta[world, output_row] = data.beta
            data.phi[world, output_row] = phi
            data.restitution[world, output_row] = restitution
        else:
            data.row_type[world, output_row] = 2
            data.row_parent[world, output_row] = slot
            data.row_mu[world, output_row] = mu * data.friction_scale
            data.row_beta[world, output_row] = 0.0
            data.phi[world, output_row] = 0.0
            data.restitution[world, output_row] = 0.0
        diagonal = float(0.0)
        if has_dense:
            jacobian = _Vec6()
            for dof in range(6):
                value0 = float(0.0)
                value1 = float(0.0)
                bit = wp.uint32(1) << wp.uint32(dof)
                if art0 >= 0 and body0 >= 0 and data.response_dofs[art0] == 6:
                    if (data.body_mask[body0] & bit) != wp.uint32(0):
                        value0 = point_projection(
                            data.motion[data.dof_start[art0] + dof], direction, p0, data.origin[art0]
                        )
                if art1 >= 0 and body1 >= 0 and data.response_dofs[art1] == 6:
                    if (data.body_mask[body1] & bit) != wp.uint32(0):
                        value1 = -point_projection(
                            data.motion[data.dof_start[art1] + dof], direction, p1, data.origin[art1]
                        )
                jacobian[dof] = value0 + value1
            response = solve_six(data.factor, group, jacobian)
            for dof in range(6):
                data.J[group, output_row, dof] = jacobian[dof]
                data.Y[group, output_row, dof] = response[dof]
                diagonal += jacobian[dof] * response[dof]
        coord0 = -1
        coord1 = -1
        value0 = float(0.0)
        value1 = float(0.0)
        response0 = float(0.0)
        response1 = float(0.0)
        if art0 >= 0 and body0 >= 0 and data.response_dofs[art0] == data.sparse_size:
            dof0 = data.body_single_dof[body0]
            if dof0 >= 0:
                local0 = dof0 - data.dof_start[art0]
                if local0 >= 0 and local0 < data.sparse_size:
                    coord0 = data.dof_offset[art0] + local0
                    value0 = point_projection(data.motion[dof0], direction, p0, data.origin[art0])
                    response0 = value0 * data.inverse_mass[dof0]
        if art1 >= 0 and body1 >= 0 and data.response_dofs[art1] == data.sparse_size:
            dof1 = data.body_single_dof[body1]
            if dof1 >= 0:
                local1 = dof1 - data.dof_start[art1]
                if local1 >= 0 and local1 < data.sparse_size:
                    coord1 = data.dof_offset[art1] + local1
                    value1 = -point_projection(data.motion[dof1], direction, p1, data.origin[art1])
                    response1 = value1 * data.inverse_mass[dof1]
        if coord0 >= 0 and coord0 == coord1:
            value0 += value1
            response0 += response1
            coord1 = -1
            value1 = 0.0
            response1 = 0.0
        data.sparse_dof[world, output_row, 0] = coord0
        data.sparse_dof[world, output_row, 1] = coord1
        data.sparse_jy[world, output_row, 0] = value0
        data.sparse_jy[world, output_row, 1] = response0
        data.sparse_jy[world, output_row, 2] = value1
        data.sparse_jy[world, output_row, 3] = response1
        sparse_diagonal = float(0.0)
        if coord0 >= 0:
            sparse_diagonal += value0 * response0
        if coord1 >= 0:
            sparse_diagonal += value1 * response1
        data.diag[world, output_row] = (diagonal + sparse_diagonal) + data.cfm


@wp.kernel
def produce_contacts(data: ContactBoundaryData):
    """Publish allocated contacts with the original raw-contact ownership."""
    produce_contact(data, wp.tid())


@wp.kernel
def produce_limit_response(
    bounds: wp.array2d[int],
    counts: wp.array[int],
    groups: wp.array[int],
    factor: wp.array3d[float],
    J: wp.array3d[float],
    Y: wp.array3d[float],
    cfm: wp.array2d[float],
    diag: wp.array2d[float],
):
    """Respond only to the bounded active dense-six position-limit prefix."""
    world, row = wp.tid()
    if row >= wp.min(bounds[world, 1], counts[world]):
        return
    group = groups[world]
    jacobian = _Vec6()
    for dof in range(6):
        jacobian[dof] = J[group, row, dof]
    response = solve_six(factor, group, jacobian)
    diagonal = float(0.0)
    for dof in range(6):
        Y[group, row, dof] = response[dof]
        diagonal += jacobian[dof] * response[dof]
    diag[world, row] = diagonal + cfm[world, row]


def bind_contact_data(solver, state_in, state_aug, contacts):
    """Bind current contact inputs and canonical output owners without launching work."""
    data = ContactBoundaryData()
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
        ("body_single_dof", "body_single_response_dof"),
        ("response_dofs", "articulation_response_dof_count"),
        ("dof_start", "articulation_dof_start"),
        ("dof_offset", "articulation_world_dof_offset"),
        ("origin", "articulation_origin"),
        ("prescribed", "_prescribed_articulation"),
        ("material_mu", "shape_material_mu"),
        ("material_restitution", "shape_material_restitution"),
        ("inverse_mass", "_diagonal_inverse_mass"),
        ("dense_group", "_sparse_diagonal_dense_groups"),
        ("sparse_size", "_sparse_diagonal_response_size"),
        ("friction_scale", "contact_friction_scale"),
        ("beta", "pgs_beta"),
        ("cfm", "pgs_cfm"),
        ("row_type", "row_type"),
        ("row_parent", "row_parent"),
        ("row_mu", "row_mu"),
        ("row_beta", "row_beta"),
        ("row_cfm", "row_cfm"),
        ("phi", "phi"),
        ("target", "target_velocity"),
        ("restitution", "row_restitution"),
        ("sparse_dof", "_sparse_diagonal_row_dof"),
        ("sparse_jy", "_sparse_diagonal_row_jy"),
        ("diag", "diag"),
    ):
        setattr(data, name, getattr(solver, source))
    data.shape_body = solver.model.shape_body
    data.body_q = state_in.body_q
    data.body_v = state_aug.body_v_s
    data.motion = state_aug.joint_S_s
    data.factor = solver.L_by_size[6]
    data.J = solver.J_by_size[6]
    data.Y = solver.Y_by_size[6]
    data.shared_anchor = int(solver.contact_shared_anchor)
    data.friction_shared_anchor = int(solver.contact_friction_shared_anchor)
    return data


def launch_contacts(solver, state_in, state_aug, contacts, workers):
    """Bind actual current owners; Warp capture retains these launch descriptors."""
    data = bind_contact_data(solver, state_in, state_aug, contacts)
    # Allocation has completed, but constraint_count is finalized later. Keep
    # this clear and the following producer ordered on the current stream.
    wp.launch(
        clear_contact_response,
        dim=(solver.world_count, 256),
        inputs=[solver.slot_counter, solver.dense_phase_bounds, data.dense_group, data.J, data.Y],
        block_dim=256,
        device=solver.model.device,
    )
    wp.launch(produce_contacts, dim=contacts.rigid_contact_max, inputs=[data], device=solver.model.device)
    # These original kernels encode schedule links into the freshly published
    # sparse DOF slots. Never overwrite those fields after this point.
    wp.launch(
        solver._mark_independent_sparse_contact_candidates_kernel,
        dim=workers,
        inputs=[
            contacts.rigid_contact_count,
            workers,
            solver.contact_world,
            solver.contact_slot,
            solver.contact_art_a,
            solver.contact_art_b,
            solver.contact_path,
            solver.contact_slots_needed,
            solver.articulation_response_dof_count,
        ],
        outputs=[solver._sparse_diagonal_row_dof],
        device=solver.model.device,
    )

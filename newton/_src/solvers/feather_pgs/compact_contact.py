# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental complete contact owner for the existing dense-six + sparse solver.

Allocation, contact admission, scheduling, RHS and the solve remain authoritative
in the original pipeline. This module only changes the representation producer.
Persistent patch friction writes its tangent rows with the patch row law. The
patch link, ``_FrictionPatchState.link``, runs after this producer and alone
links the patch rings and divides the friction coefficient, in one
``link_patch_routes`` launch over every route.
"""

import warp as wp

from . import field_group
from .contact_input import ContactInput
from .contact_laws import (
    contact_friction_points,
    contact_geometry,
    contact_materials,
    contact_row_frame,
    contact_tangent_basis,
    jacobian_entry,
    prescribed_target,
)
from .field_group import Field
from .friction_patches import PatchRowLaw, patch_friction_location, patch_row_law, patch_tangent_row
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION

_Vec6 = wp.types.vector(length=6, dtype=wp.float32)


COMPACT_BOUNDARY = (
    Field("body_single_dof", wp.int32, ("body",), attr="body_single_response_dof"),
    Field("dof_offset", wp.int32, ("articulation",), attr="articulation_world_dof_offset"),
    Field("inverse_mass", wp.float32, ("dof",), attr="_diagonal_inverse_mass"),
    Field("dense_group", wp.int32, ("world",), attr="_sparse_diagonal_dense_groups"),
    Field("factor", wp.float32, ("group", 6, 6), owner="six"),
    Field("sparse_size", wp.int32, attr="_sparse_diagonal_response_size"),
    Field("J", wp.float32, ("group", "row", 6), owner="six"),
    Field("Y", wp.float32, ("group", "row", 6), owner="six"),
    Field("sparse_dof", wp.int32, ("world", "row", 2), attr="_sparse_diagonal_row_dof"),
    Field("sparse_jy", wp.float32, ("world", "row", 4), attr="_sparse_diagonal_row_jy"),
)
"""What the compact boundary producer writes besides the contact input's row records: each row's dense-six
Jacobian and response (by the six-DOF size group's factor) and its two sparse response coordinates."""

CompactBoundary = field_group.struct("CompactBoundary", COMPACT_BOUNDARY, __name__)


def supported(solver) -> bool:
    """Require the inherited complete sparse contract and its dense-six triple or unit mode."""
    return bool(
        solver.model.device.is_cuda
        and solver._sparse_diagonal_contact_solve
        and solver._sparse_diagonal_dense_size == 6
        and solver._sparse_diagonal_response_size > 6
        and (solver._sparse_diagonal_contact_triples or solver._sparse_diagonal_contact_units)
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
def _produce_contact(data: ContactInput, boundary: CompactBoundary, contact: int):
    if contact >= data.count[0]:
        return
    slot = data.slot[contact]
    if data.path[contact] != 0 or slot < 0:
        return
    world = data.world[contact]
    art0 = data.art0[contact]
    art1 = data.art1[contact]
    body0, body1, normal, point0, point1, phi = contact_geometry(
        contact,
        data.shape0,
        data.shape1,
        data.point0,
        data.point1,
        data.normal,
        data.margin0,
        data.margin1,
        data.shape_body,
        data.body_q,
    )
    has_dense = False
    if art0 >= 0 and body0 >= 0:
        has_dense = data.response_dofs[art0] == 6
    if art1 >= 0 and body1 >= 0:
        has_dense = has_dense or data.response_dofs[art1] == 6
    mu, restitution = contact_materials(
        data.shape0[contact], data.shape1[contact], data.material_mu, data.material_restitution
    )
    tangent0, tangent1 = contact_tangent_basis(normal)
    friction0, friction1 = contact_friction_points(point0, point1, data.shared_anchor, data.friction_shared_anchor)
    group = boundary.dense_group[world]
    for row in range(data.slots_needed[contact]):
        direction, p0, p1 = contact_row_frame(
            row, data.shared_anchor, normal, tangent0, tangent1, point0, point1, friction0, friction1
        )
        output_row = slot + row
        data.target[world, output_row] = prescribed_target(
            body0, art0, body1, art1, p0, p1, direction, data.prescribed, data.origin, data.body_v
        )
        data.row_cfm[world, output_row] = data.cfm
        if row == 0:
            data.row_type[world, output_row] = PGS_CONSTRAINT_TYPE_CONTACT
            data.row_parent[world, output_row] = -1
            data.row_mu[world, output_row] = mu
            data.row_beta[world, output_row] = data.beta
            data.phi[world, output_row] = phi
            data.restitution[world, output_row] = restitution
        else:
            data.row_type[world, output_row] = PGS_CONSTRAINT_TYPE_FRICTION
            data.row_parent[world, output_row] = slot
            data.row_mu[world, output_row] = mu * data.friction_scale
            data.row_beta[world, output_row] = 0.0
            data.phi[world, output_row] = 0.0
            data.restitution[world, output_row] = 0.0
        _publish_row_response(
            data, boundary, world, output_row, group, has_dense, art0, body0, art1, body1, direction, p0, p1
        )


@wp.func
def _publish_row_response(
    data: ContactInput,
    boundary: CompactBoundary,
    world: int,
    output_row: int,
    group: int,
    has_dense: bool,
    art0: int,
    body0: int,
    art1: int,
    body1: int,
    direction: wp.vec3,
    p0: wp.vec3,
    p1: wp.vec3,
):
    """Write one row's dense-six and sparse response and its diagonal."""
    diagonal = float(0.0)
    if has_dense:
        jacobian = _Vec6()
        for dof in range(6):
            value0 = float(0.0)
            value1 = float(0.0)
            bit = wp.uint32(1) << wp.uint32(dof)
            if art0 >= 0 and body0 >= 0 and data.response_dofs[art0] == 6:
                if (data.body_mask[body0] & bit) != wp.uint32(0):
                    value0 = jacobian_entry(data.motion[data.dof_start[art0] + dof], direction, p0, data.origin[art0])
            if art1 >= 0 and body1 >= 0 and data.response_dofs[art1] == 6:
                if (data.body_mask[body1] & bit) != wp.uint32(0):
                    value1 = -jacobian_entry(data.motion[data.dof_start[art1] + dof], direction, p1, data.origin[art1])
            jacobian[dof] = value0 + value1
        response = solve_six(boundary.factor, group, jacobian)
        for dof in range(6):
            boundary.J[group, output_row, dof] = jacobian[dof]
            boundary.Y[group, output_row, dof] = response[dof]
            diagonal += jacobian[dof] * response[dof]
    coord0 = -1
    coord1 = -1
    value0 = float(0.0)
    value1 = float(0.0)
    response0 = float(0.0)
    response1 = float(0.0)
    if art0 >= 0 and body0 >= 0 and data.response_dofs[art0] == boundary.sparse_size:
        dof0 = boundary.body_single_dof[body0]
        if dof0 >= 0:
            local0 = dof0 - data.dof_start[art0]
            if local0 >= 0 and local0 < boundary.sparse_size:
                coord0 = boundary.dof_offset[art0] + local0
                value0 = jacobian_entry(data.motion[dof0], direction, p0, data.origin[art0])
                response0 = value0 * boundary.inverse_mass[dof0]
    if art1 >= 0 and body1 >= 0 and data.response_dofs[art1] == boundary.sparse_size:
        dof1 = boundary.body_single_dof[body1]
        if dof1 >= 0:
            local1 = dof1 - data.dof_start[art1]
            if local1 >= 0 and local1 < boundary.sparse_size:
                coord1 = boundary.dof_offset[art1] + local1
                value1 = -jacobian_entry(data.motion[dof1], direction, p1, data.origin[art1])
                response1 = value1 * boundary.inverse_mass[dof1]
    if coord0 >= 0 and coord0 == coord1:
        value0 += value1
        response0 += response1
        coord1 = -1
        value1 = 0.0
        response1 = 0.0
    boundary.sparse_dof[world, output_row, 0] = coord0
    boundary.sparse_dof[world, output_row, 1] = coord1
    boundary.sparse_jy[world, output_row, 0] = value0
    boundary.sparse_jy[world, output_row, 1] = response0
    boundary.sparse_jy[world, output_row, 2] = value1
    boundary.sparse_jy[world, output_row, 3] = response1
    sparse_diagonal = float(0.0)
    if coord0 >= 0:
        sparse_diagonal += value0 * response0
    if coord1 >= 0:
        sparse_diagonal += value1 * response1
    data.diag[world, output_row] = (diagonal + sparse_diagonal) + data.cfm


@wp.func
def _produce_patch_contact(data: ContactInput, boundary: CompactBoundary, law: PatchRowLaw, contact: int):
    """Publish one contact's rows with the persistent patch friction law.

    The normal row is ``_produce_contact``'s. Tangent rows, present only when the
    contact owns an anchor, follow the patch row law (``contact_laws.contact_row_frame`` at the
    patch friction location, and ``patch_tangent_row``): they act at the patch friction location, carry the
    anchor's tangential phi with ``law.beta``, and keep the undivided friction
    coefficient.
    """
    if contact >= data.count[0]:
        return
    slot = data.slot[contact]
    if data.path[contact] != 0 or slot < 0:
        return
    world = data.world[contact]
    art0 = data.art0[contact]
    art1 = data.art1[contact]
    body0, body1, normal, point0, point1, phi = contact_geometry(
        contact,
        data.shape0,
        data.shape1,
        data.point0,
        data.point1,
        data.normal,
        data.margin0,
        data.margin1,
        data.shape_body,
        data.body_q,
    )
    has_dense = False
    if art0 >= 0 and body0 >= 0:
        has_dense = data.response_dofs[art0] == 6
    if art1 >= 0 and body1 >= 0:
        has_dense = has_dense or data.response_dofs[art1] == 6
    mu, restitution = contact_materials(
        data.shape0[contact], data.shape1[contact], data.material_mu, data.material_restitution
    )
    tangent0, tangent1 = contact_tangent_basis(normal)
    friction0, friction1 = patch_friction_location(law.patches, contact)
    group = boundary.dense_group[world]
    for row in range(data.slots_needed[contact]):
        direction, p0, p1 = contact_row_frame(
            row, data.shared_anchor, normal, tangent0, tangent1, point0, point1, friction0, friction1
        )
        output_row = slot + row
        data.target[world, output_row] = prescribed_target(
            body0, art0, body1, art1, p0, p1, direction, data.prescribed, data.origin, data.body_v
        )
        data.row_cfm[world, output_row] = data.cfm
        if row == 0:
            data.row_type[world, output_row] = PGS_CONSTRAINT_TYPE_CONTACT
            data.row_parent[world, output_row] = -1
            data.row_mu[world, output_row] = mu
            data.row_beta[world, output_row] = data.beta
            data.phi[world, output_row] = phi
            data.restitution[world, output_row] = restitution
        else:
            tangent_phi, beta, friction_mu = patch_tangent_row(law, contact, row, mu, data.friction_scale)
            data.row_type[world, output_row] = PGS_CONSTRAINT_TYPE_FRICTION
            data.row_parent[world, output_row] = slot
            data.row_mu[world, output_row] = friction_mu
            data.row_beta[world, output_row] = beta
            data.phi[world, output_row] = tangent_phi
            data.restitution[world, output_row] = 0.0
        _publish_row_response(
            data, boundary, world, output_row, group, has_dense, art0, body0, art1, body1, direction, p0, p1
        )


@wp.kernel
def produce_contacts(data: ContactInput, boundary: CompactBoundary):
    """Publish contacts in their original collision order."""
    _produce_contact(data, boundary, wp.tid())


@wp.kernel
def produce_patch_contacts(data: ContactInput, boundary: CompactBoundary, law: PatchRowLaw):
    """Publish contacts with the patch friction law in their original collision order."""
    _produce_patch_contact(data, boundary, law, wp.tid())


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


def launch_contacts(solver, data, workers):
    """Produce the contact rows from the step's bound contact input; Warp capture retains these launch descriptors."""
    boundary = field_group.bind(
        CompactBoundary, solver=solver, factor=solver.L_by_size[6], J=solver.J_by_size[6], Y=solver.Y_by_size[6]
    )
    # Allocation has completed, but constraint_count is finalized later. Keep
    # this clear and the following producer ordered on the current stream.
    wp.launch(
        clear_contact_response,
        dim=(solver.world_count, 256),
        inputs=[solver.slot_counter, solver.dense_phase_bounds, boundary.dense_group, boundary.J, boundary.Y],
        block_dim=256,
        device=solver.model.device,
    )
    # Persistent patches write their tangent rows with the patch row law. The
    # patch link (_FrictionPatchState.link, one link_patch_routes launch) still
    # runs after this producer: it alone links normal rows into patch rings and
    # divides the tangent coefficient.
    law = patch_row_law(solver)
    # One thread per contact; in a capture whose launches bind, each replay runs the live count.
    pool = solver._contact_pool
    count = data.count
    if law is None:
        pool.launch(produce_contacts, pool.launch_rows(), inputs=[data, boundary], extent=count)
    else:
        pool.launch(produce_patch_contacts, pool.launch_rows(), inputs=[data, boundary, law], extent=count)
    # These original kernels encode schedule links into the freshly published
    # sparse DOF slots. Never overwrite those fields after this point.
    pool.launch(
        solver._mark_independent_sparse_contact_candidates_kernel,
        workers,
        inputs=[data, workers],
        outputs=[solver._sparse_diagonal_row_dof],
        extent=count,
        clamp=True,
    )

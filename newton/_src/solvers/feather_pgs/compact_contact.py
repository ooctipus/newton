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
    prescribed_target,
)
from .friction_patches import PatchRowLaw, patch_friction_location, patch_row_law, patch_tangent_row
from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from .response_builds_rows import CompactBoundary, sparse_pair_row_response


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
        and not solver._debug_buffers_enabled
        and not solver._preelim_active
    )


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
        sparse_pair_row_response(
            data, boundary, world, output_row, group, has_dense, art0, body0, art1, body1, direction, p0, p1
        )


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
        sparse_pair_row_response(
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

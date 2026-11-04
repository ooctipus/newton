# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""The contact row's law: its records, its position-pass rhs and its Jacobian entries.

Every contact row producer writes a contact's rows here (:func:`write_contact_row`): the one writer
(:mod:`.walk_rows`), the local row packets, the keyed rows and the compact contact boundary. Each producer forms the
contact's front end once (:func:`.contact_input.contact_frame`) and takes the friction row law (:class:`PatchRowLaw`):
a contact owns tangent rows where the allocation reserved them, and they follow the patch row law. Point friction is
the patch law with beta 0, a disabled view and a zero phi: its tangent rows act at the contact's point.
"""

from __future__ import annotations

import warp as wp

from .contact_input import ContactFrame, ContactInput
from .contact_laws import (
    CONTACT_SPIN_ROW,
    contact_friction,
    contact_materials,
    contact_row_entry,
    contact_row_frame,
    prescribed_spin_target,
    prescribed_target,
)
from .friction_patches import PatchRowLaw, patch_tangent_row
from .row_laws import (
    ROW_CONTACT,
    ROW_FRICTION,
    ROW_TORSION,
    contact_capped_beta,
    contact_depenetration_cap,
    contact_material_rhs,
    contact_restitution_rhs,
    contact_rhs,
    row_rhs,
)
from .row_records import row_sides


@wp.func
def endpoint_projection(
    data: ContactInput, body: int, art: int, size: int, dof: int, row: int, point: wp.vec3, direction: wp.vec3
):
    """Read current response masks and current world-origin motion columns: contact row ``row``'s entry at
    coordinate ``dof`` (:func:`.contact_laws.contact_row_entry`)."""
    value = float(0.0)
    if body >= 0 and art >= 0 and data.response_dofs[art] == size:
        if (data.body_mask[body] & (wp.uint32(1) << wp.uint32(dof))) != wp.uint32(0):
            value = contact_row_entry(row, data.motion[data.dof_start[art] + dof], direction, point, data.origin[art])
    return value


@wp.func
def incident_projection(data: ContactInput, body: int, art: int, point: wp.vec3, direction: wp.vec3):
    """Use the frozen incident velocity, not the live output velocity."""
    result = float(0.0)
    if art >= 0 and body >= 0:
        size = data.response_dofs[art]
        for dof in range(size):
            result += (
                endpoint_projection(data, body, art, size, dof, 0, point, direction)
                * data.incident[data.dof_start[art] + dof]
            )
    return result


@wp.func
def contact_torsion_mu(data: ContactInput, c: int) -> float:
    """Contact ``c``'s torsional coefficient ``mu_t`` [m], its spin row's: its shapes' torsional friction combined as
    the friction coefficient is (:func:`.contact_laws.contact_friction`). The patch link divides an anchor's among
    the region's anchors, as the pair's (:mod:`.friction_patches`)."""
    return contact_friction(data.shape0[c], data.shape1[c], data.material_mu_torsional)


@wp.func
def contact_normal_beta(data: ContactInput, c: int, body_a: int, body_b: int, phi: float) -> float:
    """Contact ``c``'s normal row ``beta`` (:func:`.row_laws.contact_capped_beta`): the depenetration cap of the bodies
    that respond (:func:`.row_laws.contact_depenetration_cap`). Zero-coordinate articulations stand as ground."""
    art_a = data.art0[c]
    art_b = data.art1[c]
    if art_a < 0 or data.response_dofs[art_a] <= 0:
        body_a = -1
    if art_b < 0 or data.response_dofs[art_b] <= 0:
        body_b = -1
    cap = contact_depenetration_cap(body_a, body_b, data.max_depenetration_velocity)
    return contact_capped_beta(phi, data.beta, cap, data.dt)


@wp.func
def write_contact_row(data: ContactInput, law: PatchRowLaw, frame: ContactFrame, c: int, row: int, with_rhs: bool):
    """Write contact ``c``'s row ``row`` (0 the normal, 1 and 2 the tangents, 3 the spin row under contact torsion)
    at ``slot + row`` from its front end ``frame`` (:func:`.contact_input.contact_frame`, formed once for the
    contact's rows): its dense records and, ``with_rhs``, its position-pass ``rhs`` and the CFM seed of its diagonal;
    return its bodies and frame ``(body_a, body_b, direction, row_point)``.

    Every contact row producer writes its records here; the general route forms its rhs later. A tangent row follows
    the friction row law (:func:`.friction_patches.patch_tangent_row`): the anchor's tangential displacement as its
    gap, the law's beta and the undivided coefficient ``mu``. A normal row's rhs is
    :func:`.row_laws.contact_rhs` at its capped beta (:func:`contact_normal_beta`), and a bouncing one's the restitution target from its incident velocity summed per
    body (:func:`incident_projection`). A spin row (:data:`.contact_laws.CONTACT_SPIN_ROW`) turns about the normal:
    its coefficient is :func:`contact_torsion_mu`, its target the prescribed bodies' spin
    (:func:`.contact_laws.prescribed_spin_target`) and its rhs ``-target``.

    With ``contact_compliance`` a contact of positive stiffness is a material contact: its normal row's rhs and
    compliance are :func:`.row_laws.contact_material_rhs`'s (``row_eps``, zero on every other contact row), written
    whether or not ``with_rhs``, it takes no restitution, and its tangents' coefficient takes the contact's friction
    scale (zero meaning one).
    """
    body_a = frame.body_a
    body_b = frame.body_b
    normal = frame.normal
    point = frame.point
    phi = frame.phi
    art_a = data.art0[c]
    art_b = data.art1[c]
    world = data.world[c]
    line = data.first[world]
    slot = data.slot[c]
    mu, restitution = contact_materials(data.shape0[c], data.shape1[c], data.material_mu, data.material_restitution)
    stiffness = float(0.0)
    if data.material != 0:
        stiffness = data.stiffness[c]
    if stiffness > 0.0:
        restitution = 0.0
    direction, row_point = contact_row_frame(row, normal, frame.tangent0, frame.tangent1, point, frame.friction_point)
    target = prescribed_target(
        body_a, art_a, body_b, art_b, row_point, direction, data.prescribed, data.origin, data.body_v
    )
    if row == CONTACT_SPIN_ROW:
        target = prescribed_spin_target(body_a, art_a, body_b, art_b, normal, data.prescribed, data.body_v)
    r = slot + row
    normal_beta = float(0.0)
    if row == CONTACT_SPIN_ROW:
        data.row_type[line, r] = ROW_TORSION
        data.row_parent[line, r] = slot
        data.row_mu[line, r] = contact_torsion_mu(data, c)
        data.row_beta[line, r] = 0.0
        data.phi[line, r] = 0.0
        data.restitution[line, r] = 0.0
    elif row == 0:
        normal_beta = contact_normal_beta(data, c, body_a, body_b, phi)
        data.row_type[line, r] = ROW_CONTACT
        data.row_parent[line, r] = -1
        data.row_mu[line, r] = mu
        data.row_beta[line, r] = normal_beta
        data.phi[line, r] = phi
        data.restitution[line, r] = restitution
    else:
        tangent_phi, beta = patch_tangent_row(law, c, row)
        friction_mu = mu
        if stiffness > 0.0 and data.friction_weight[c] > 0.0:
            friction_mu *= data.friction_weight[c]
        data.row_type[line, r] = ROW_FRICTION
        data.row_parent[line, r] = slot
        data.row_mu[line, r] = friction_mu
        data.row_beta[line, r] = beta
        data.phi[line, r] = tangent_phi
        data.restitution[line, r] = 0.0
    if data.material != 0:
        eps = float(0.0)
        if row == 0 and stiffness > 0.0:
            material_rhs, eps = contact_material_rhs(phi, target, stiffness, data.damping[c], data.dt)
            data.rhs[line, r] = material_rhs
        data.row_eps[line, r] = eps
    data.target[line, r] = target
    data.sides[line, r] = row_sides(art_a, art_b, data.response_dofs)
    if with_rhs:
        inv_dt = 1.0 / data.dt
        rhs = float(0.0)
        weight = float(1.0)
        if row == 0 and stiffness > 0.0:
            rhs = data.rhs[line, r]
        elif row == 0:
            rhs, weight = contact_rhs(
                phi, normal_beta, target, inv_dt, data.bias_scale, data.speculative_scale, data.contact_w
            )
            if restitution > 0.0:
                relative = incident_projection(data, body_a, art_a, row_point, direction)
                relative -= incident_projection(data, body_b, art_b, row_point, direction)
                relative -= target
                fires, bounce = contact_restitution_rhs(
                    phi, relative, target, restitution, data.dt, data.restitution_threshold
                )
                if fires:
                    rhs = bounce
                    weight = 1.0
        elif row == CONTACT_SPIN_ROW:
            rhs, _w = row_rhs(ROW_TORSION, 0.0, 0.0, target, inv_dt, data.bias_scale, 0.0, 0.0, 1.0)
        else:
            tangent_phi, beta = patch_tangent_row(law, c, row)
            rhs, _w = row_rhs(ROW_FRICTION, tangent_phi, beta, target, inv_dt, data.bias_scale, 0.0, 0.0, 1.0)
        data.rhs[line, slot + row] = rhs
        data.diag[line, slot + row] = data.cfm
        if data.contact_w < 1.0:
            data.row_w[line, slot + row] = weight
    return body_a, body_b, direction, row_point

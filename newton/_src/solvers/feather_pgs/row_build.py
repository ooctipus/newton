# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact row build with one thread per (contact, dof).

The legacy ``populate_world_J_for_size`` runs one thread per contact and walks the kinematic chain six times per
contact (normal and two friction rows, two bodies), accumulating into a pre-cleared Jacobian with read-modify-write
stores. It is latency bound (65k threads, 8% issue utilisation). This kernel assigns one thread per (contact, dof):
every thread recomputes the contact-level scalars (cheap, all reads of one contact are L1-resident across the 32
threads of the contact) and produces the three Jacobian entries of its dof directly, using the per-body response-dof
bitmask that the solver already builds, so the whole Jacobian row is written once with coalesced stores. Results are
bit-identical to the legacy kernel: each entry is one signed projection (two when both bodies belong to the same
articulation, summed in the same order).

Both passes take the friction row law (:class:`PatchRowLaw`): the allocation's extent decides which contacts own
tangent rows, and those rows follow the patch row law, as the generic per-contact builder writes them. Point friction is
the patch law with beta 0, a disabled view and a zero phi: its tangent rows act at the contact's point.

The row owners that write rows elsewhere (local row packets, keyed rows) share the contact row's law
(:func:`write_contact_row`: its records and position-pass rhs) and its Jacobian entries (:func:`endpoint_projection`).
"""

from __future__ import annotations

import warp as wp

from .contact_input import ContactInput, contact_frame
from .contact_laws import (
    CONTACT_SPIN_ROW,
    contact_friction,
    contact_materials,
    contact_row_entry,
    contact_row_frame,
    jacobian_entry,
    prescribed_spin_target,
    prescribed_target,
    spin_entry,
)
from .dense_rows import RowSupport, jacobian_row, row_sides
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


@wp.kernel(enable_backward=False)
def contact_row_prelude(
    data: ContactInput,
    total_threads: int,
    size_a: int,
    size_b: int,
    law: PatchRowLaw,
    # outputs (and the row records of ``data``)
    pre_flags: wp.array[int],  # bit0 active, bit1 friction rows, bit2 A in size, bit3 B in size, bit4 spin row
):
    """Per contact row (grid-stride, 3 threads per contact): geometry, friction decisions and row metadata.

    Thread ``r`` of a contact writes the flags (r == 0) and the metadata of row ``slot + r`` only, so each thread's
    dependency chain carries one prescribed-target evaluation and eight stores instead of three and twenty-four. A
    contact owns tangent rows when the allocation reserved them (``contact_slots_needed`` of 3, or 4 with its spin
    row, which thread 2 also writes); they follow the row law (``patch_tangent_row``) and keep the undivided
    friction coefficient: the patch link, which runs after every row producer, divides it and links the normal rows
    into rings.
    """
    tid = wp.tid()
    r = tid % 3
    total_contacts = data.count[0]
    for c in range(tid // 3, total_contacts, total_threads // 3):
        _contact_row_prelude_one(c, r, data, size_a, size_b, law, pre_flags)


@wp.func
def _contact_in_sizes(data: ContactInput, c: int, size_a: int, size_b: int):
    """Whether each endpoint's articulation responds in one of the two sizes."""
    art_a = data.art0[c]
    art_b = data.art1[c]
    a_in = False
    b_in = False
    if art_a >= 0:
        sa = data.response_dofs[art_a]
        a_in = sa == size_a or sa == size_b
    if art_b >= 0:
        sb = data.response_dofs[art_b]
        b_in = sb == size_a or sb == size_b
    return a_in, b_in


@wp.func
def _contact_row_prelude_one(
    c: int, r: int, data: ContactInput, size_a: int, size_b: int, law: PatchRowLaw, pre_flags: wp.array[int]
):
    if r == 0:
        pre_flags[c] = 0
    if data.path[c] != 0:
        return
    slot = data.slot[c]
    if slot < 0:
        return
    a_in, b_in = _contact_in_sizes(data, c, size_a, size_b)
    if not a_in and not b_in:
        return
    # Allocation owns the row extent; recomputing eligibility can cross a
    # floating-point threshold and overwrite the following contact.
    will_add_friction = data.slots_needed[c] >= 3
    if r > 0 and not will_add_friction:
        return
    if r == 0:
        flags = 1
        if will_add_friction:
            flags |= 2
        if a_in:
            flags |= 4
        if b_in:
            flags |= 8
        if data.slots_needed[c] > CONTACT_SPIN_ROW:
            flags |= 16
        pre_flags[c] = flags
    write_contact_row(data, law, c, r, False)
    if r == 2 and data.slots_needed[c] > CONTACT_SPIN_ROW:
        write_contact_row(data, law, c, CONTACT_SPIN_ROW, False)


@wp.kernel(enable_backward=False)
def populate_world_J_masked(
    data: ContactInput,
    total_threads: int,
    size_a: int,
    size_b: int,
    art_group_idx: wp.array[int],
    law: PatchRowLaw,
    pre_flags: wp.array[int],
    support: RowSupport,
    # outputs: Jacobian blocks of the two size groups (the second may alias the first when only one group exists;
    # both are the packed rows under ``support.packed``)
    J_group_a: wp.array3d[float],
    J_group_b: wp.array3d[float],
):
    """One thread per (contact, local dof), grid-stride over contacts: the Jacobian entries of that dof, written once.

    ``lanes`` threads per contact (the larger group size, not 32): idle lanes were 30% of the launch on 22-DOF worlds.
    Tangent rows act at the row law's friction location.
    """
    tid = wp.tid()
    lanes = wp.max(size_a, size_b)
    lane = tid % lanes
    total_contacts = data.count[0]
    for c in range(tid // lanes, total_contacts, total_threads // lanes):
        populate_contact_J(c, lane, data, size_a, art_group_idx, law, pre_flags[c], support, J_group_a, J_group_b)


@wp.func
def populate_contact_J(
    c: int,
    lane: int,
    data: ContactInput,
    size_a: int,
    art_group_idx: wp.array[int],
    law: PatchRowLaw,
    flags: int,
    support: RowSupport,
    J_group_a: wp.array3d[float],
    J_group_b: wp.array3d[float],
):
    """Write coordinate ``lane`` of contact ``c``'s rows when its flags mark it active (:func:`contact_row_prelude`)."""
    if (flags & 1) == 0:
        return
    body_a, body_b, normal, point, _phi, tangent0, tangent1, friction_point = contact_frame(data, law, c)
    _write_contact_jacobian_lane(
        lane,
        flags,
        data.slot[c],
        data.art0[c],
        data.art1[c],
        body_a,
        body_b,
        normal,
        tangent0,
        tangent1,
        point,
        friction_point,
        data,
        size_a,
        art_group_idx,
        support,
        J_group_a,
        J_group_b,
    )


@wp.func
def _write_contact_jacobian_lane(
    lane: int,
    flags: int,
    slot: int,
    art_a: int,
    art_b: int,
    body_a: int,
    body_b: int,
    normal: wp.vec3,
    tangent0: wp.vec3,
    tangent1: wp.vec3,
    point: wp.vec3,
    friction_point: wp.vec3,
    data: ContactInput,
    size_a: int,
    art_group_idx: wp.array[int],
    support: RowSupport,
    J_group_a: wp.array3d[float],
    J_group_b: wp.array3d[float],
):
    """Write one dof of a contact's normal row and, with flag bit 1, its two tangent rows, and with bit 4 its spin
    row, in each side (:func:`.dense_rows.jacobian_row`).

    Each entry is one signed :func:`.contact_laws.jacobian_entry` (the spin row's :func:`.contact_laws.spin_entry`),
    or two summed (A's first) when both bodies belong to the same articulation. Rows act where
    :func:`.contact_laws.contact_row_frame` puts them. The rows are evaluated straight rather than in a row loop, so
    the tangent rows share their anchor's velocity and no row frame stays live across iterations.
    """
    a_in = (flags & 4) != 0
    b_in = (flags & 8) != 0
    friction = (flags & 2) != 0
    spin = (flags & 16) != 0
    bit = wp.uint32(1 << lane)
    d0, p0 = contact_row_frame(0, normal, tangent0, tangent1, point, friction_point)
    d1, p1 = contact_row_frame(1, normal, tangent0, tangent1, point, friction_point)
    d2, p2 = contact_row_frame(2, normal, tangent0, tangent1, point, friction_point)
    sides = row_sides(art_a, art_b, data.response_dofs)
    if a_in and lane < data.response_dofs[art_a]:
        at = jacobian_row(support, art_group_idx[art_a], art_a, sides)
        origin = data.origin[art_a]
        S = data.motion[data.dof_start[art_a] + lane]
        jn = float(0.0)
        jt0 = float(0.0)
        jt1 = float(0.0)
        js = float(0.0)
        if body_a >= 0 and (data.body_mask[body_a] & bit) != wp.uint32(0):
            jn += jacobian_entry(S, d0, p0, origin)
            jt0 += jacobian_entry(S, d1, p1, origin)
            jt1 += jacobian_entry(S, d2, p2, origin)
            if spin:
                js += spin_entry(S, normal)
        if b_in and art_b == art_a and body_b >= 0 and (data.body_mask[body_b] & bit) != wp.uint32(0):
            jn += -jacobian_entry(S, d0, p0, origin)
            jt0 += -jacobian_entry(S, d1, p1, origin)
            jt1 += -jacobian_entry(S, d2, p2, origin)
            if spin:
                js += -spin_entry(S, normal)
        if data.response_dofs[art_a] == size_a:
            J_group_a[at[0], slot, at[1] + lane] = jn
            if friction:
                J_group_a[at[0], slot + 1, at[1] + lane] = jt0
                J_group_a[at[0], slot + 2, at[1] + lane] = jt1
            if spin:
                J_group_a[at[0], slot + CONTACT_SPIN_ROW, at[1] + lane] = js
        else:
            J_group_b[at[0], slot, at[1] + lane] = jn
            if friction:
                J_group_b[at[0], slot + 1, at[1] + lane] = jt0
                J_group_b[at[0], slot + 2, at[1] + lane] = jt1
            if spin:
                J_group_b[at[0], slot + CONTACT_SPIN_ROW, at[1] + lane] = js
    if b_in and (not a_in or art_b != art_a) and lane < data.response_dofs[art_b]:
        at = jacobian_row(support, art_group_idx[art_b], art_b, sides)
        origin = data.origin[art_b]
        S = data.motion[data.dof_start[art_b] + lane]
        jn = float(0.0)
        jt0 = float(0.0)
        jt1 = float(0.0)
        js = float(0.0)
        if body_b >= 0 and (data.body_mask[body_b] & bit) != wp.uint32(0):
            jn = -jacobian_entry(S, d0, p0, origin)
            jt0 = -jacobian_entry(S, d1, p1, origin)
            jt1 = -jacobian_entry(S, d2, p2, origin)
            if spin:
                js = -spin_entry(S, normal)
        if data.response_dofs[art_b] == size_a:
            J_group_a[at[0], slot, at[1] + lane] = jn
            if friction:
                J_group_a[at[0], slot + 1, at[1] + lane] = jt0
                J_group_a[at[0], slot + 2, at[1] + lane] = jt1
            if spin:
                J_group_a[at[0], slot + CONTACT_SPIN_ROW, at[1] + lane] = js
        else:
            J_group_b[at[0], slot, at[1] + lane] = jn
            if friction:
                J_group_b[at[0], slot + 1, at[1] + lane] = jt0
                J_group_b[at[0], slot + 2, at[1] + lane] = jt1
            if spin:
                J_group_b[at[0], slot + CONTACT_SPIN_ROW, at[1] + lane] = js


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
def write_contact_row(data: ContactInput, law: PatchRowLaw, c: int, row: int, with_rhs: bool):
    """Write contact ``c``'s row ``row`` (0 the normal, 1 and 2 the tangents, 3 the spin row under contact torsion)
    at ``slot + row`` from its front end
    (:func:`.contact_input.contact_frame`): its dense records and, ``with_rhs``, its position-pass ``rhs`` and the CFM
    seed of its diagonal; return its bodies and frame ``(body_a, body_b, direction, row_point)``.

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
    body_a, body_b, normal, point, phi, tangent0, tangent1, friction_point = contact_frame(data, law, c)
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
    direction, row_point = contact_row_frame(row, normal, tangent0, tangent1, point, friction_point)
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
    data.row_cfm[line, r] = data.cfm
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

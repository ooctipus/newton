# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind the solver's current contact, body and row storage for a contact-row producer."""

import warp as wp


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


def bind(solver, state_in, state_aug, contacts, dt, bias_scale):
    """Bind the solver's live storage without allocating device memory or reading back."""
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
    # Point-contact producers follow the canonical path's ignored legacy limit.
    data.friction_anchor_limit = 0
    data.friction_pairs_only = int(solver.contact_friction_articulation_pairs_only)
    data.friction_scale = solver.contact_friction_scale
    data.beta, data.cfm = solver.pgs_beta, solver.pgs_cfm
    data.dt, data.bias_scale = dt, bias_scale
    data.speculative_scale = solver.contact_speculative_scale
    data.restitution_threshold = solver._effective_restitution_velocity_threshold
    data.contact_w = solver._contact_w
    return data

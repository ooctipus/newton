# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The contact-row producers' input: the contacts, the bodies, the row law and the dense row records.

Declared once as fields (:mod:`.field_group`); :data:`ContactInput` is their
Warp struct and :func:`bind` fills it from the solver's live storage.

Every producer reads a contact through the same front end, :func:`contact_frame`: a contact's geometry, tangent basis
and friction location under the friction row law (:data:`.friction_patches.PatchRowLaw`, whose point form is the patch
law with beta 0 and a disabled view). Its rows' records come from :func:`.row_build.write_contact_row`.
"""

import warp as wp

from . import dense_rows, field_group
from .contact_laws import contact_friction_points, contact_geometry, contact_tangent_basis
from .field_group import Field, FieldGroup
from .friction_patches import PatchRowLaw, patch_friction_location

CONTACT_INPUT = (
    # The collision pipeline's contacts.
    Field("count", wp.int32, (1,), owner="contacts", attr="rigid_contact_count"),
    Field("point0", wp.vec3, ("contact",), owner="contacts", attr="rigid_contact_point0"),
    Field("point1", wp.vec3, ("contact",), owner="contacts", attr="rigid_contact_point1"),
    Field("normal", wp.vec3, ("contact",), owner="contacts", attr="rigid_contact_normal"),
    Field("shape0", wp.int32, ("contact",), owner="contacts", attr="rigid_contact_shape0"),
    Field("shape1", wp.int32, ("contact",), owner="contacts", attr="rigid_contact_shape1"),
    Field("margin0", wp.float32, ("contact",), owner="contacts", attr="rigid_contact_margin0"),
    Field("margin1", wp.float32, ("contact",), owner="contacts", attr="rigid_contact_margin1"),
    # The allocator's record of each contact.
    Field("world", wp.int32, ("contact",), attr="contact_world"),
    Field("slot", wp.int32, ("contact",), attr="contact_slot"),
    Field("art0", wp.int32, ("contact",), attr="contact_art_a"),
    Field("art1", wp.int32, ("contact",), attr="contact_art_b"),
    Field("path", wp.int32, ("contact",), attr="contact_path"),
    Field("slots_needed", wp.int32, ("contact",), attr="contact_slots_needed"),
    # Shapes, bodies and articulations.
    Field("shape_body", wp.int32, ("shape",), owner="model"),
    Field("body_q", wp.transform, ("body",), owner="state_in"),
    Field("body_v", wp.spatial_vector, ("body",), owner="state_aug", attr="body_v_s"),
    Field("body_mask", wp.uint32, ("body",), attr="body_response_dof_mask"),
    Field("response_dofs", wp.int32, ("articulation",), attr="articulation_response_dof_count"),
    Field("dof_start", wp.int32, ("articulation",), attr="articulation_dof_start"),
    Field("origin", wp.vec3, ("articulation",), attr="articulation_origin"),
    Field("motion", wp.spatial_vector, ("dof",), owner="state_aug", attr="joint_S_s"),
    Field("prescribed", wp.int32, ("articulation",), attr="_prescribed_articulation"),
    Field("material_mu", wp.float32, ("shape",), attr="shape_material_mu"),
    Field("material_restitution", wp.float32, ("shape",), attr="shape_material_restitution"),
    Field("incident", wp.float32, ("dof",), attr="v_hat"),
    # The row law.
    Field("shared_anchor", wp.int32, attr="contact_shared_anchor"),
    Field("friction_shared_anchor", wp.int32, attr="contact_friction_shared_anchor"),
    Field("friction_scale", wp.float32, attr="contact_friction_scale"),
    Field("beta", wp.float32, attr="pgs_beta"),
    Field("cfm", wp.float32, attr="pgs_cfm"),
    Field("dt", wp.float32, owner="step"),  # the step's values, given to bind() directly
    Field("bias_scale", wp.float32, owner="step"),
    Field("speculative_scale", wp.float32, attr="contact_speculative_scale"),
    Field("restitution_threshold", wp.float32, attr="_effective_restitution_velocity_threshold"),
    Field("contact_w", wp.float32, attr="_contact_w"),
    # The dense row records the producers write.
    *dense_rows.RECORDS,
)

ContactInput = field_group.struct("ContactInput", CONTACT_INPUT, __name__)

ContactRoutes = field_group.struct(
    "ContactRoutes", FieldGroup(*CONTACT_INPUT).select("count", "normal", "world", "slot", "path"), __name__
)
"""The contacts' count and normals with each contact's world, row slot and path (0 dense, 1 matrix-free, -1 none)."""


@wp.func
def contact_frame(data: ContactInput, law: PatchRowLaw, c: int):
    """Return contact ``c``'s front end: ``(body_a, body_b, normal, p_a, p_b, phi, tangent0, tangent1, f_a, f_b)``.

    The bodies, normal, surface points and gap of :func:`.contact_laws.contact_geometry`, the tangent basis
    and the friction location on each side: the patch friction location under persistent patches, else the
    contact's friction points (:func:`.contact_laws.contact_friction_points`).
    """
    body_a, body_b, normal, point_a, point_b, phi = contact_geometry(
        c,
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
    tangent0, tangent1 = contact_tangent_basis(normal)
    friction_a, friction_b = contact_friction_points(point_a, point_b, data.shared_anchor, data.friction_shared_anchor)
    if law.patches.enabled != 0:
        friction_a, friction_b = patch_friction_location(law.patches, c)
    return body_a, body_b, normal, point_a, point_b, phi, tangent0, tangent1, friction_a, friction_b


def bind_routes(solver, contacts):
    """Bind the contacts' :data:`ContactRoutes` (the allocator's records of the last step)."""
    return field_group.bind(ContactRoutes, solver=solver, contacts=contacts)


def bind(solver, state_in, state_aug, contacts, dt, bias_scale, data=None):
    """Bind the solver's live storage into a :data:`ContactInput`, without allocating device memory or reading back.

    Args:
        data: An instance to refill, or ``None`` for a new one.
    """
    return field_group.bind(
        ContactInput,
        data,
        solver=solver,
        contacts=contacts,
        model=solver.model,
        state_in=state_in,
        state_aug=state_aug,
        dt=dt,
        bias_scale=bias_scale,
    )

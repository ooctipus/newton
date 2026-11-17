# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The contact-row producers' input: the contacts, the bodies, the row law and the dense row records.

Declared once as fields (:mod:`.field_group`); :data:`ContactInput` is their
Warp struct and :func:`bind` fills it from the solver's live storage.

Every producer reads a contact through the same front end, :func:`contact_frame`, once a contact: a contact's
geometry, tangent basis and friction location under the friction row law (:data:`.friction_patches.PatchRowLaw`, whose
point form is the patch law with beta 0 and a disabled view). Its rows' records come from
:func:`.row_build.write_contact_row`, given that frame.
"""

import warp as wp

from . import field_group, row_records
from .contact_laws import contact_geometry, contact_tangent_basis
from .field_group import Field, FieldGroup
from .friction_patches import PatchRowLaw, patch_friction_location

CONTACT_INPUT = (
    # The collision pipeline's contacts: their count, and their records in the frame's order (the member store,
    # friction_patches._prepare_contacts), with the maps between the frame's positions and the collision's contacts.
    # Every per-contact field here is indexed by the frame's position.
    Field("count", wp.int32, (1,), owner="contacts", attr="rigid_contact_count"),
    Field("point0", wp.vec3, ("contact",), owner="records"),
    Field("point1", wp.vec3, ("contact",), owner="records"),
    Field("normal", wp.vec3, ("contact",), owner="records"),
    Field("shape0", wp.int32, ("contact",), owner="records"),
    Field("shape1", wp.int32, ("contact",), owner="records"),
    Field("margin0", wp.float32, ("contact",), owner="records"),
    Field("margin1", wp.float32, ("contact",), owner="records"),
    Field("contact_of", wp.int32, ("contact",), owner="frame_order"),
    Field("position", wp.int32, ("contact",), owner="frame_order"),
    # The allocator's record of each contact.
    Field("world", wp.int32, ("contact",), attr="contact_world", owner="placement"),
    Field("slot", wp.int32, ("contact",), attr="contact_slot", owner="placement"),
    Field("art0", wp.int32, ("contact",), attr="contact_art_a", owner="placement"),
    Field("art1", wp.int32, ("contact",), attr="contact_art_b", owner="placement"),
    Field("path", wp.int32, ("contact",), attr="contact_path", owner="placement"),
    Field("slots_needed", wp.int32, ("contact",), attr="contact_slots_needed", owner="placement"),
    # Shapes, bodies and articulations.
    Field("shape_body", wp.int32, ("shape",), owner="model"),
    Field("body_q", wp.transform, ("body",), owner="state_in"),
    Field("body_v", wp.spatial_vector, ("body",), owner="state_aug", attr="body_v_s"),
    Field("body_mask", wp.uint32, ("body",), owner="forest", attr="body_response_dof_mask"),
    Field("response_dofs", wp.int32, ("articulation",), owner="forest", attr="articulation_response_dof_count"),
    Field("dof_start", wp.int32, ("articulation",), owner="forest", attr="articulation_dof_start"),
    Field("origin", wp.vec3, ("articulation",), owner="factor", attr="articulation_origin"),
    Field("motion", wp.spatial_vector, ("dof",), owner="state_aug", attr="joint_S_s"),
    Field("prescribed", wp.int32, ("articulation",), owner="forest", attr="prescribed_articulation"),
    Field("material_mu", wp.float32, ("shape",), owner="laws", attr="shape_material_mu"),
    Field("material_restitution", wp.float32, ("shape",), owner="laws", attr="shape_material_restitution"),
    Field("material_mu_torsional", wp.float32, ("shape",), owner="laws", attr="shape_material_mu_torsional"),
    Field(
        "max_depenetration_velocity", wp.float32, ("body",), owner="laws", attr="rigid_body_max_depenetration_velocity"
    ),
    Field("incident", wp.float32, ("dof",), attr="v_hat"),
    # The row law.
    Field("beta", wp.float32, owner="options", attr="pgs_beta"),
    Field("cfm", wp.float32, owner="options", attr="pgs_cfm"),
    Field("dt", wp.float32, owner="step"),  # the step's values, given to bind() directly
    Field("bias_scale", wp.float32, owner="step"),
    Field("speculative_scale", wp.float32, owner="options", attr="contact_speculative_scale"),
    Field("restitution_threshold", wp.float32, owner="laws", attr="effective_restitution_velocity_threshold"),
    Field("contact_w", wp.float32, owner="laws"),
    # The contacts' material (contact_compliance): stiffness [N/m], damping [N s/m] and friction scale per contact,
    # the collision's (read at contact_of); a contact of zero stiffness keeps the rigid law.
    Field("material", wp.int32, owner="options", attr="contact_compliance"),
    Field("stiffness", wp.float32, ("contact",), attr="contact_stiffness"),
    Field("damping", wp.float32, ("contact",), attr="contact_damping"),
    Field("friction_weight", wp.float32, ("contact",), attr="contact_friction_weight"),
    # Authored contact materials: 1 where a colliding shape authors one (ShapeFlags.CONTACT_MATERIAL); such a shape's
    # contacts take its and its partner's stiffness and damping by MuJoCo's rule (row_laws.authored_contact_law).
    Field("authored", wp.int32, owner="laws", attr="authored_materials"),
    Field("shape_flags", wp.int32, ("shape",), owner="laws"),
    Field("material_ke", wp.float32, ("shape",), owner="laws", attr="shape_material_ke"),
    Field("material_kd", wp.float32, ("shape",), owner="laws", attr="shape_material_kd"),
    # Contact torsion: 1 gives every friction contact the authored torsional coefficient, 0 none.
    Field("torsion", wp.int32, owner="laws", attr="torsion_rows"),
    # Which friction contacts carry a spin row (a fourth row), the plan's one fact (RowLaws.spin_rows): 0 none, 1 those
    # whose region has a footprint, 2 every one (contact torsion).
    Field("spin", wp.int32, owner="laws", attr="spin_rows_where"),
    # The dense row records the producers write.
    *row_records.RECORDS,
)

ContactInput = field_group.struct("ContactInput", CONTACT_INPUT, __name__)

ContactRoutes = field_group.struct(
    "ContactRoutes",
    (
        Field("count", wp.int32, (1,), owner="contacts", attr="rigid_contact_count"),
        Field("normal", wp.vec3, ("contact",), owner="contacts", attr="rigid_contact_normal"),
        *FieldGroup(*CONTACT_INPUT).select("position", "world", "slot", "path"),
    ),
    __name__,
)
"""The collision's contacts by their own index: their count and normals, each one's position in the frame
(:func:`route_position`) and, at that position, its world, row slot and path (0 dense, 1 matrix-free, -1 none)."""


@wp.func
def route_position(routes: ContactRoutes, c: int) -> int:
    """The frame's position of the collision's contact ``c``, at which its world, slot and path lie."""
    return routes.position[c]


@wp.struct
class ContactFrame:
    """A contact's front end (:func:`contact_frame`), formed once for all its rows: its bodies, normal, the one point
    its rows act at and its gap, its tangent basis and its friction location."""

    body_a: int
    body_b: int
    normal: wp.vec3
    point: wp.vec3
    phi: float
    tangent0: wp.vec3
    tangent1: wp.vec3
    friction_point: wp.vec3


@wp.func
def contact_front_end(
    at: int,
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    friction_point: wp.vec3,
) -> ContactFrame:
    """Return the front end (:class:`ContactFrame`) of the contact whose collision record is ``at`` in the given arrays
    (the collision pipeline's, or a copy in another order) with the friction location ``friction_point``: the
    bodies, normal, the one point the contact's rows act at and the gap of :func:`.contact_laws.contact_geometry`,
    and the tangent basis."""
    body_a, body_b, n, point, phi = contact_geometry(
        at, shape0, shape1, point0, point1, normal, margin0, margin1, shape_body, body_q
    )
    tangent0, tangent1 = contact_tangent_basis(n)
    frame = ContactFrame()
    frame.body_a = body_a
    frame.body_b = body_b
    frame.normal = n
    frame.point = point
    frame.phi = phi
    frame.tangent0 = tangent0
    frame.tangent1 = tangent1
    frame.friction_point = friction_point
    return frame


@wp.func
def contact_frame(data: ContactInput, law: PatchRowLaw, c: int) -> ContactFrame:
    """Return contact ``c``'s front end (:class:`ContactFrame`, :func:`contact_front_end`) from the collision
    pipeline's arrays, with its region's friction location."""
    return contact_front_end(
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
        patch_friction_location(law.patches, c),
    )


def bind_routes(placement, contacts):
    """Bind the contacts' :data:`ContactRoutes` (the allocator's records of the last step)."""
    return field_group.bind(
        ContactRoutes, placement=placement, contacts=contacts, frame_order=placement.laws.friction_patches
    )


def bind(solver, state_in, state_aug, contacts, dt, bias_scale, data=None, records=None):
    """Bind the solver's live storage into a :data:`ContactInput`, without allocating device memory or reading back.

    Args:
        data: An instance to refill, or ``None`` for a new one.
        records: The contact records to bind (:class:`.friction_patches.ContactRecords`): the frame's copy in the
            frame's order (``None``), or the collision's in its written order, which the row allocation visits.
    """
    return field_group.bind(
        ContactInput,
        data,
        solver=solver,
        placement=solver.placement,
        forest=solver.forest,
        factor=solver.factor,
        laws=solver.laws,
        options=solver.options,
        contacts=contacts,
        records=solver.laws.friction_patches.records if records is None else records,
        frame_order=solver.laws.friction_patches,
        model=solver.model,
        state_in=state_in,
        state_aug=state_aug,
        dt=dt,
        bias_scale=bias_scale,
    )

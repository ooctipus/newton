# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The solver's contact force export: the solved contact impulses as world-frame forces, packed into Newton's
spatial contact forces."""

import warp as wp

from .contact_input import ContactRoutes
from .contact_laws import contact_tangent_basis
from .row_kernels import PGS_CONSTRAINT_TYPE_FRICTION


@wp.func
def _contact_linear_force(
    dense_first: wp.array[int],
    body_base: int,
    routes: ContactRoutes,
    c: int,
    world_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    """Convert solved FeatherPGS contact impulses into world-frame forces: a contact's rows sit at its slot of its
    world's line, a free body's contact's (``path`` 1) among the free bodies' rows from ``body_base``."""
    total_contacts = routes.count[0]
    if c >= total_contacts:
        return

    force = wp.vec3(0.0)
    slot = routes.slot[c]
    path = routes.path[c]

    if slot >= 0 and path >= 0 and inv_dt > 0.0:
        world = routes.world[c]
        line = dense_first[world]
        # Contacts store normals from shape 0 toward shape 1 (A-to-B). FeatherPGS
        # solves along the opposite direction internally, which corresponds to the
        # force on shape/body 0 from shape/body 1.
        normal = -routes.normal[c]

        lam_n = world_impulses[line, slot]
        lam_t0 = 0.0
        lam_t1 = 0.0
        # The end of the contact's class in the line.
        end = world_constraint_count[world]
        if path == 1:
            end = body_base + mf_constraint_count[world]
        if (
            enable_friction != 0
            and slot + 2 < end
            and world_row_type[line, slot + 1] == PGS_CONSTRAINT_TYPE_FRICTION
            and world_row_parent[line, slot + 1] == slot
            and world_row_type[line, slot + 2] == PGS_CONSTRAINT_TYPE_FRICTION
            and world_row_parent[line, slot + 2] == slot
        ):
            lam_t0 = world_impulses[line, slot + 1]
            lam_t1 = world_impulses[line, slot + 2]

        force = lam_n * normal
        if enable_friction != 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
            force += lam_t0 * tangent0 + lam_t1 * tangent1
        force *= inv_dt

    rigid_contact_force[c] = force


@wp.kernel
def compute_contact_linear_force_from_impulses(
    dense_first: wp.array[int],
    body_base: int,
    routes: ContactRoutes,
    world_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    "Convert solved FeatherPGS contact impulses into world-frame forces."
    _contact_linear_force(
        dense_first,
        body_base,
        routes,
        wp.tid(),
        world_impulses,
        world_constraint_count,
        mf_constraint_count,
        world_row_type,
        world_row_parent,
        enable_friction,
        inv_dt,
        rigid_contact_force,
    )


@wp.func
def _pack_contact_linear_force(
    c: int,
    contact_count: wp.array[wp.int32],
    rigid_contact_force: wp.array[wp.vec3],
    # outputs
    contact_force: wp.array[wp.spatial_vector],
):
    """Pack linear contact forces into Newton's spatial-force contact buffer."""
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    contact_force[c] = wp.spatial_vector(rigid_contact_force[c], wp.vec3(0.0))


@wp.kernel
def pack_contact_linear_force_as_spatial(
    contact_count: wp.array[wp.int32],
    rigid_contact_force: wp.array[wp.vec3],
    # outputs
    contact_force: wp.array[wp.spatial_vector],
):
    "Pack linear contact forces into Newton's spatial-force contact buffer."
    _pack_contact_linear_force(
        wp.tid(),
        contact_count,
        rigid_contact_force,
        contact_force,
    )

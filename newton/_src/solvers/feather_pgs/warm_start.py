# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweep's warm start: the impulses' initialization, and the previous step's normal impulses gathered by contact
identity.

The warm start carries the contact normals only. A contact's friction (its tangent pair and spin row) starts every step
cold: under the friction law its history is the anchor, whose displacement the rows' targets already carry
(:mod:`.friction_patches`). A carried friction impulse would be a second history of the same stick, and on a stance
with more anchors than the bodies' motions (two feet of one rigid shin, four feet of one torso) part of what the stick
asks is out of every motion's reach: a cold step answers that part with a bounded impulse, a carried one adds it up
step after step along the stance's indeterminate force modes until the feet reach their Coulomb cones and slip.
"""

import warp as wp

from . import field_group
from .contact_input import ContactRoutes
from .factor.kernels import Clock, clock_dt_ratio
from .field_group import Field
from .row_kernels import PGS_CONSTRAINT_TYPE_CONTACT
from .row_records import DenseRows, MfRows


@wp.kernel
def prepare_world_impulses(
    dense_first: wp.array[int],
    world_constraint_count: wp.array[int],
    max_constraints: int,
    warmstart: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Cold-initialize dense rows before identity-gathering contact impulses.

    No current non-contact row family has a cross-step identity contract:
    limits are activation-allocated, while mimic and link rows can be enabled
    dynamically and allocated atomically. Warm start therefore clears the
    full capacity before the contact/friction identity gather; cold solves only
    need to clear the current active prefix.
    """
    world = wp.tid()
    line = dense_first[world]
    m = world_constraint_count[world]

    # Cold-started solves consume only the current active prefix. Rows outside
    # that prefix are ignored, and a row that becomes active on a later step is
    # cleared then as part of that step's prefix. Warm-started solves clear full
    # capacity so no inactive cache can survive a later layout growth.
    clear_count = max_constraints
    if warmstart == 0:
        clear_count = m

    for i in range(clear_count):
        world_impulses[line, i] = 0.0


ContactCarry = field_group.struct(
    "ContactCarry",
    (
        Field("normal", wp.vec3, ("contact",), attr="carry_normal", owner="placement"),
        Field("impulse", wp.float32, ("contact",), attr="carry_impulse", owner="placement"),
    ),
    __name__,
)
"""What a step keeps for the next, by contact identity: each contact's normal and its converged normal impulse (zero
where it held no normal row: a carried zero is the cold start). One record, written once a step by
:func:`carry_contacts`; the next step's gather reads it through the contacts' match index. The contact pool's rows
hold it (``carry_normal``, ``carry_impulse``). FeatherPGS consumes the sorted contacts, so a contact's index is its
sorted index, by which the next frame's ``rigid_contact_match_index`` names it."""


def bind_carry(placement):
    """Bind the placement's :data:`ContactCarry`."""
    return field_group.bind(ContactCarry, placement=placement)


@wp.kernel
def carry_contacts(
    routes: ContactRoutes,
    first: wp.array[int],
    row_type: wp.array2d[int],
    impulses: wp.array2d[float],
    carry: ContactCarry,
):
    """Write the carry: each contact's normal and the impulse of its normal row, read at its slot of its world's line
    (one row index for both classes); zero past the contacts, over every row the stream used."""
    c = wp.tid()
    impulse = float(0.0)
    if c < routes.count[0]:
        carry.normal[c] = routes.normal[c]
        slot = routes.slot[c]
        if slot >= 0:
            line = first[routes.world[c]]
            if row_type[line, slot] == PGS_CONSTRAINT_TYPE_CONTACT:
                impulse = impulses[line, slot]
    carry.impulse[c] = impulse


@wp.kernel
def forget_carried_contacts(world: wp.array[int], world_mask: wp.array[wp.bool], carry: ContactCarry):
    """Forget the carried impulses of the selected worlds' contacts (every contact without a mask), each contact's
    world the allocator's of the carried step (``contact_world``)."""
    c = wp.tid()
    if world_mask:
        w = world[c]
        if w < 0 or w >= world_mask.shape[0] or not world_mask[w]:
            return
    carry.impulse[c] = 0.0


@wp.kernel
def reset_world_impulses(first: wp.array[int], world_mask: wp.array[wp.bool], impulses: wp.array2d[float]):
    """Clear the selected worlds' lines of the row store, both classes' rows."""
    world = wp.tid()
    if world_mask and not world_mask[world]:
        return
    line = first[world]
    for row in range(impulses.shape[1]):
        impulses[line, row] = 0.0


# =============================================================================
# Fully Matrix-Free PGS Kernels (velocity-space Jacobi)
# =============================================================================


# =============================================================================
# Matrix-Free PGS Kernels for Free Rigid Bodies
# =============================================================================
#
# A free body is a one-node tree: its six velocity coordinates (its center-of-mass velocity, then its angular
# velocity) have the 6x6 mass matrix ``H``, armature included, which the walk takes leaf-first (``P`` reverses the
# coordinates). The walk whitens a free-body row, ``Z`` per body, and the sweeps hold the change of the body's
# velocity in the walk's coordinates ``u``, the walk in of ``dv`` (``WALK_ENTER``; ``u . u = dv^T H dv``): a row's
# velocity term is ``Z^T u`` (``J dv``), its update ``u += Z dlambda``, and the walk down leaves ``dv``. Walk
# coordinate ``i`` of ``Z`` and ``u`` sits at position ``5 - i``, the physical coordinate it reverses
# (:func:`.response_operator.leaf_first`). The walk whitens the rows and leaves them (:mod:`.walk_rows`,
# :func:`.factor.walk.walk_leave_kernel`).


@wp.func
def carried_normal(impulse: float, normal: wp.vec3, previous_normal: wp.vec3):
    """A carried normal impulse on the current contact: zero where it is not finite or the normal turned over, never
    negative."""
    if not wp.isfinite(impulse) or wp.dot(normal, previous_normal) <= 0.0:
        return 0.0
    return wp.max(impulse, 0.0)


@wp.kernel
def gather_mf_warmstart(
    routes: ContactRoutes,
    mf: MfRows,
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    carry: ContactCarry,
    mf_constraint_count: wp.array[int],
    decay: float,
    clock: wp.array[Clock],
    # in-out
    mf_impulses: wp.array2d[float],
):
    """Seed this step's MF normal rows from the previous step's converged normal impulses, matched by contact
    identity.

    Runs after ``allocate_world_contact_slots`` + ``walk_rows.write_free_body_rows`` (so ``contact_slot`` and the
    rows' types are populated for this step) and after the warm-start branch has zeroed their impulses (the row store's
    free bodies' rows). One thread per contact; each writes only its own normal row, so unmatched contacts and every
    friction row keep the zero left by the memset.

    The step clock's ratio (:func:`.factor.kernels.clock_dt_ratio`, dt_now / dt_prev) rescales carried impulses
    across step-size changes (impulse is proportional to dt for quasi-static loads); 1.0 at fixed dt.

    ``match_index[c] >= 0`` is the previous frame's *sorted* contact index, by which the carry is keyed
    (:func:`carry_contacts`).
    """
    c = wp.tid()
    if c >= routes.count[0]:
        return
    if routes.path[c] != 1:  # MF path only
        return
    new_slot = routes.slot[c]
    mi = match_index[c]
    if new_slot < 0 or mi < 0:
        return
    world = routes.world[c]
    # The free bodies' rows in the world's line: from ``mf.base``, the slots' and parents' row index.
    if new_slot >= mf.base + mf_constraint_count[world]:
        return
    mf_line = mf.first[world]
    if mf.row_type[mf_line, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT:
        impulse = decay * clock_dt_ratio(clock[0]) * carry.impulse[mi]
        mf_impulses[mf_line, new_slot] = carried_normal(impulse, routes.normal[c], carry.normal[mi])


@wp.kernel
def gather_dense_warmstart(
    routes: ContactRoutes,
    rows: DenseRows,
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    carry: ContactCarry,
    world_constraint_count: wp.array[int],
    decay: float,
    clock: wp.array[Clock],
    # in-out
    world_impulses: wp.array2d[float],
):
    """Seed this step's dense normal rows from the previous step's converged normal impulses, matched by contact
    identity: the dense mirror of :func:`gather_mf_warmstart`.

    Runs after ``prepare_world_impulses`` has cold-zeroed every dense row, so unmatched or fresh contacts and every
    friction row stay cold and no stale impulse survives a slot-layout shift. ``decay`` scales every seeded impulse:
    at persistent quasi-static contacts it damps the accumulate-on-impact overshoot while keeping the sweep's starting
    point near the converged solution.
    """
    c = wp.tid()
    if c >= routes.count[0]:
        return
    if routes.path[c] != 0:  # dense path only
        return
    new_slot = routes.slot[c]
    mi = match_index[c]
    if new_slot < 0 or mi < 0:
        return
    world = routes.world[c]
    if new_slot >= world_constraint_count[world]:
        return
    line = rows.first[world]
    if rows.row_type[line, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT:
        impulse = decay * clock_dt_ratio(clock[0]) * carry.impulse[mi]
        world_impulses[line, new_slot] = carried_normal(impulse, routes.normal[c], carry.normal[mi])

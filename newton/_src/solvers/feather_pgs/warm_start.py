# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweep's warm start: the impulses' initialization, the carry a step keeps by contact identity
(:data:`ContactCarry`, written by :func:`carry_contacts`), and the next step's one gather of it
(:func:`gather_carried_normals`).

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
from .row_laws import ROW_CONTACT


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
            if row_type[line, slot] == ROW_CONTACT:
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


@wp.func
def carried_normal(impulse: float, normal: wp.vec3, previous_normal: wp.vec3):
    """A carried normal impulse on the current contact: zero where it is not finite or the normal turned over, never
    negative."""
    if not wp.isfinite(impulse) or wp.dot(normal, previous_normal) <= 0.0:
        return 0.0
    return wp.max(impulse, 0.0)


@wp.kernel
def gather_carried_normals(
    routes: ContactRoutes,
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    carry: ContactCarry,
    first: wp.array[int],
    row_type: wp.array2d[int],
    decay: float,
    clock: wp.array[Clock],
    # in-out
    impulses: wp.array2d[float],
):
    """Seed this step's contact normal rows from the previous step's carry (:data:`ContactCarry`), matched by contact
    identity: one thread a contact, which writes only its normal row, at its slot of its world's line (one row index
    for both row classes).

    Runs in the warm-start stage, after the row pass wrote the rows' types and the clears zeroed every row of the line,
    so fresh contacts and every friction row stay cold (a carried zero writes the zero there). An accepted contact's
    slot lies below its class's count (:func:`.device_preludes.reserve`), and a matched contact keeps its class (the
    matcher matches within a shape pair, whose articulations fix the class): the map needs no class.

    The step clock's ratio (:func:`.factor.kernels.clock_dt_ratio`, dt_now / dt_prev) rescales carried impulses across
    step-size changes (impulse is proportional to dt for quasi-static loads); 1.0 at fixed dt. ``decay`` scales every
    seeded impulse: at persistent quasi-static contacts it damps the accumulate-on-impact overshoot while keeping the
    sweep's starting point near the converged solution.
    """
    c = wp.tid()
    if c >= routes.count[0]:
        return
    slot = routes.slot[c]
    mi = match_index[c]
    if slot < 0 or mi < 0:
        return
    line = first[routes.world[c]]
    if row_type[line, slot] == ROW_CONTACT:
        impulse = decay * clock_dt_ratio(clock[0]) * carry.impulse[mi]
        impulses[line, slot] = carried_normal(impulse, routes.normal[c], carry.normal[mi])

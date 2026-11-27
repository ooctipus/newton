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

from ...core.execution import fill_lanes
from . import field_group
from .contact_input import ContactRoutes
from .factor.kernels import Clock, clock_dt_ratio
from .field_group import Field
from .row_laws import ROW_CONTACT

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})


PREPARE_LANES = 32
"""Lanes a world of :func:`prepare_world_impulses` at least: they stride over the world's rows, so its stores coalesce."""


def row_lanes(worlds: int, rows: int, device) -> int:
    """Lanes a world of a launch whose lanes stride over the world's ``rows`` rows, a row each (:func:`prepare_world_impulses`,
    :func:`reset_world_impulses`): :data:`PREPARE_LANES`, at most the rows, and where a few worlds of many rows would idle
    the GPU, the lanes that fill it (:func:`.core.execution.fill_lanes`, up to the rows)."""
    least = 1 << (max(min(PREPARE_LANES, int(rows)), 1) - 1).bit_length()
    return fill_lanes(worlds, device, least=least, most=max(1 << (max(int(rows), 1) - 1).bit_length(), least))


@wp.kernel
def prepare_world_impulses(
    dense_first: wp.array[int],
    world_constraint_count: wp.array[int],
    max_constraints: int,
    free_count: wp.array[int],
    free_base: wp.array[int],
    free_max: int,
    lanes: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Cold-initialize each world's rows before identity-gathering contact impulses, ``lanes`` lanes a world striding
    over its rows: its dense rows, then its free bodies' rows (``free_count`` of them from the line's body base
    ``free_base``, at most ``free_max``: zero where no free body keeps rows).

    No row holds an impulse from one step to the next (the warm start carries a contact's by its identity, and no other
    row family has a cross-step identity contract: limits are activation-allocated, while mimic and link rows can be
    enabled dynamically and allocated atomically), and no reader takes a row past the world's counts, so a world's
    live rows are the ones cleared: a row that becomes live on a later step is cleared then. The clear stays within the
    world's line or run of the row store, whatever its capacity.
    """
    world, lane = wp.tid()
    line = dense_first[world]
    clear_count = wp.min(world_constraint_count[world], max_constraints)
    free_clear = wp.min(free_count[world], free_max)

    for i in range(lane, clear_count, lanes):
        world_impulses[line, i] = 0.0
    for i in range(lane, free_clear, lanes):
        world_impulses[line, free_base[line] + i] = 0.0


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
:func:`carry_contacts` at each contact's index in the step's contact set. The contact pool's rows hold it
(``carry_normal``, ``carry_impulse``). FeatherPGS consumes the sorted contacts, so a contact's index is its sorted
index: the next step reads a contact's record at that index while the contact set is the same (the steps that share a
collision pass), and through ``rigid_contact_match_index``, which names it by its index in the last pass, after a
collision pass (:func:`gather_carried_normals`)."""


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
    world the allocator's of the carried step (``contact_world``; a contact the allocator placed in no world carries
    nothing)."""
    c = wp.tid()
    if world_mask:
        w = world[c]
        if w < 0 or w >= world_mask.shape[0] or not world_mask[w]:
            return
    carry.impulse[c] = 0.0


@wp.kernel
def reset_world_impulses(
    first: wp.array[int],
    world_mask: wp.array[wp.bool],
    counts: wp.array2d[int],
    free_base: wp.array[int],
    lanes: int,
    impulses: wp.array2d[float],
):
    """Clear the selected worlds' live rows of the row store (``counts``: by class and world), both classes' (the free
    bodies' from the line's body base), ``lanes`` lanes a world striding over them (:func:`row_lanes`)."""
    world, lane = wp.tid()
    if world_mask and not world_mask[world]:
        return
    line = first[world]
    for row in range(lane, counts[0, world], lanes):
        impulses[line, row] = 0.0
    base = free_base[line]
    for row in range(lane, counts[1, world], lanes):
        impulses[line, base + row] = 0.0


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
    kept: wp.array[int],
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

    A contact's record is at its own index where the contact set is the last step's (``kept[0]``, the step's word
    :meth:`.friction_patches.FrictionPatchState.kept_frame`: no collision pass wrote the contacts since, so the carry
    was written in this set's order), else at its match index (the carry is the last pass's). The match index names
    a contact of the last collision pass, not of the last step: read on a step that shares its pass, it would hand
    each contact whatever contact the carry holds at that index.

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
    mi = c
    if kept[0] == 0:
        mi = match_index[c]
    if slot < 0 or mi < 0:
        return
    line = first[routes.world[c]]
    if row_type[line, slot] == ROW_CONTACT:
        impulse = decay * clock_dt_ratio(clock[0]) * carry.impulse[mi]
        impulses[line, slot] = carried_normal(impulse, routes.normal[c], carry.normal[mi])

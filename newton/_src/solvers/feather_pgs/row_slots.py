# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row store's slots: each contact's admission and its rows' reservation, the contact runs that order a
world's rows under sorted contacts, and the counts with their capacity status."""

from functools import cache

import numpy as np
import warp as wp

from ...core import union_find
from ...core.execution import fill_lanes
from ...sim import BodyFlags
from .contact_filters import contact_normal_gap_limit
from .contact_input import ContactInput
from .contact_laws import (
    contact_authored,
    contact_endpoints,
    contact_materials,
    contact_point,
    contact_points,
    contact_responds,
    contact_tracked,
    jacobian_entry,
)
from .device_preludes import (
    lane_any,
    lane_ballot,
    lane_below,
    lane_count,
    lane_first,
    lane_index,
    lane_last,
    lane_match,
    lane_maximum,
    lane_read,
    lane_scan,
    lane_stride,
    lane_sum,
    lane_sync,
)
from .friction_patches import FrictionPatches, patch_spin_rows, region_normal
from .row_laws import (
    authored_contact_law,
    contact_capped_beta,
    contact_depenetration_cap,
    contact_material_rhs,
    contact_restitution_rhs,
    contact_rhs,
    separates,
)
from .row_records import UNIT_CONTACT, row_sides, unit_source

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})

# =============================================================================
# Multi-Articulation Contact Building Kernels
# =============================================================================
# These kernels enable contacts between multiple articulations within the same
# world. The constraint system becomes world-level instead of per-articulation.


@wp.func
def _endpoint_speed(
    data: ContactInput,
    twists: wp.array[wp.spatial_vector],
    body_has_response_dofs: wp.array[int],
    body_flags: wp.array[wp.int32],
    body: int,
    art: int,
    free: bool,
    normal: wp.vec3,
    lever: wp.vec3,
):
    """A contact endpoint's speed along ``normal`` at the predictor, its point ``lever`` from its tree's origin, and
    whether its twist is known: a prescribed tree's body velocity, a free body's coordinates where the contact takes the
    free bodies' law (``free``), a responding body's published twist, a welded body's zero. A kinematic body of a tree
    with coordinates moves at a velocity the publication does not hold."""
    if body < 0 or art < 0:
        return 0.0, True
    twist = wp.spatial_vector()
    known = True
    if data.prescribed[art] != 0:
        twist = data.body_v[body]
    else:
        if body_has_response_dofs[body] != 0:
            known = (body_flags[body] & BodyFlags.KINEMATIC) == 0
            twist = twists[body]
        if free and data.response_dofs[art] > 0:
            s = data.dof_start[art]
            v = data.incident
            twist = wp.spatial_vector(v[s], v[s + 1], v[s + 2], v[s + 3], v[s + 4], v[s + 5])
    return jacobian_entry(twist, normal, lever, wp.vec3(0.0)), known


@wp.func
def _admit_contact(
    c: int,
    data: ContactInput,
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    patches: FrictionPatches,
    twists: wp.array[wp.spatial_vector],
    kept: int,
):
    """Return whether a contact gets rows, whether its rows keep zero impulse at the predictor, and its articulations,
    world and gap: ``(admitted, separates, art_a, art_b, world, phi)``. A contact its collision pass admitted
    (``kept``: a later step of the pass, :func:`admit_world_contacts`) stays admitted whatever its gap.

    A contact needs an endpoint that responds to its impulses, one world (not
    two, nor none without an articulation) and a gap within the normal gap limit. A contact without a responding
    endpoint has no world (-1): its body pair never takes rows. Nor does a contact on a tracking volume
    (:func:`.contact_laws.contact_tracked`): it is made and reported, and exerts nothing.
    Its normal row separates where its residual at the predictor (the body ``twists`` it publishes; none: never) does
    (:func:`.row_laws.separates`), read in the row's own frame (:func:`.contact_input.contact_frame`): at its point and
    along its direction, its region's normal (``patches``); its friction and spin rows, bounded by a finite
    non-negative coefficient of its ring's load, keep zero with it. The residual's bias is the row's law: an authored material's rhs
    (:func:`.row_laws.authored_contact_law`, no restitution) where a shape authors one, else the hard law's.
    """
    body_a, body_b, art_a, art_b, world = contact_endpoints(
        data.shape0[c], data.shape1[c], data.shape_body, body_to_articulation, art_to_world
    )
    responds_a = contact_responds(art_a, body_a, data.response_dofs, body_has_response_dofs, body_flags)
    responds_b = contact_responds(art_b, body_b, data.response_dofs, body_has_response_dofs, body_flags)
    phi = float(0.0)
    if not (responds_a or responds_b) or contact_tracked(data.shape0[c], data.shape1[c], data.tracking):
        world = -1
    admitted = world >= 0
    separating = False
    if admitted:
        normal, translation_a, offset_a, _translation_b, _offset_b, separation, phi = contact_points(
            data.normal[c],
            data.point0[c],
            data.point1[c],
            data.margin0[c],
            data.margin1[c],
            body_a,
            body_b,
            data.body_q,
        )
        a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
        b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
        normal_gap_limit = contact_normal_gap_limit(
            a_non_free,
            b_non_free,
            art_a == art_b,
            contact_gap_gate,
            articulation_pair_contact_gap_gate,
            same_articulation_contact_gap_gate,
        )
        admitted = kept != 0 or not (phi > normal_gap_limit)
        if admitted and twists.shape[0] > 0:
            # The free bodies' law where neither side has coordinates of a tree that is not a free body.
            free = (not (art_a >= 0 and data.response_dofs[art_a] > 0) or not a_non_free) and (
                not (art_b >= 0 and data.response_dofs[art_b] > 0) or not b_non_free
            )
            # The normal row's own frame (.contact_input.contact_frame): its point, B's share of the separation from
            # A's surface point (contact_point), and its direction, its region's normal where it is a region's member.
            point = contact_point(translation_a, offset_a, separation, data.point_share[c])
            direction = region_normal(patches, c, normal)
            lever_a = point
            lever_b = point
            if art_a >= 0:
                lever_a = point - data.origin[art_a]
            if art_b >= 0:
                lever_b = point - data.origin[art_b]
            speed_a, known_a = _endpoint_speed(
                data, twists, body_has_response_dofs, body_flags, body_a, art_a, free, direction, lever_a
            )
            speed_b, known_b = _endpoint_speed(
                data, twists, body_has_response_dofs, body_flags, body_b, art_b, free, direction, lever_b
            )
            incident = speed_a - speed_b
            cap = contact_depenetration_cap(
                wp.where(responds_a, body_a, -1), wp.where(responds_b, body_b, -1), data.max_depenetration_velocity
            )
            beta = contact_capped_beta(phi, data.beta, cap, data.dt)
            bias, _row_w = contact_rhs(phi, beta, 0.0, 1.0 / data.dt, 1.0, data.speculative_scale, 1.0)
            mu, restitution = contact_materials(
                data.shape0[c], data.shape1[c], data.material_mu, data.material_restitution
            )
            if data.authored != 0 and contact_authored(data.shape0[c], data.shape1[c], data.shape_flags):
                s0 = data.shape0[c]
                s1 = data.shape1[c]
                kappa, damping = authored_contact_law(
                    data.material_ke[s0], data.material_kd[s0], data.material_ke[s1], data.material_kd[s1], phi, data.dt
                )
                bias, _rho = contact_material_rhs(phi, 0.0, kappa, damping, data.dt)
            elif restitution > 0.0:
                fires, bounce = contact_restitution_rhs(
                    phi, incident, 0.0, restitution, data.dt, data.restitution_threshold
                )
                if fires:
                    bias = bounce
            separating = known_a and known_b and wp.isfinite(mu) and mu >= 0.0
            separating = separating and separates(incident + bias, wp.abs(speed_a) + wp.abs(speed_b) + wp.abs(bias))
    return admitted, separating, art_a, art_b, world, phi


@wp.func_native(
    """
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // The warp's lanes hold ascending positions; the lanes of one world form a group whose lowest lane records the
    // group's first and last position.
    const unsigned active = __activemask();
    const unsigned group = __match_any_sync(active, world);
    const int leader = __ffs(group) - 1;
    const int top = 31 - __clz(group);
    const int low = __shfl_sync(active, contact, leader);
    const int high = __shfl_sync(active, contact, top);
    if (world >= 0 && (int)(threadIdx.x & 31) == leader) {
        if (atomicMin(&first.data[world], low) == 0x7fffffff)
            worlds.data[atomicAdd(&listed.data[0], 1)] = world;
        atomicMax(&last.data[world], high);
    }
#elif defined(__CUDA_ARCH__)
    if (world >= 0) {
        if (atomicMin(&first.data[world], contact) == 0x7fffffff)
            worlds.data[atomicAdd(&listed.data[0], 1)] = world;
        atomicMax(&last.data[world], contact);
    }
#else
    if (world >= 0) {
        if (first.data[world] == 0x7fffffff)
            worlds.data[listed.data[0]++] = world;
        if (contact < first.data[world])
            first.data[world] = contact;
        if (contact > last.data[world])
            last.data[world] = contact;
    }
#endif
"""
)
def _record_world_span(
    world: int, contact: int, first: wp.array[int], last: wp.array[int], worlds: wp.array[int], listed: wp.array[int]
):
    """Merge position ``contact`` into ``world``'s span (-1: none), once per world and warp on CUDA, and list the world
    (``worlds``, counted in ``listed[0]``) where its span starts. Every lane of the warp that reaches the launch calls
    it, with ascending positions."""
    ...


@wp.func
def _admit_world_contact(
    c: int,
    data: ContactInput,
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    friction_patches: FrictionPatches,
    twists: wp.array[wp.spatial_vector],
    kept: int,
    # outputs
    acts: wp.array[int],
) -> int:
    """Admit, classify and size contact ``c`` (:func:`admit_world_contacts`); returns its world (-1 for none:
    :func:`_admit_contact`), written for every contact, admitted or not. On a later step of its collision pass
    (``kept``) the pass's admission, class and size stand, and only the verdict is the step's."""
    if kept != 0:
        # No verdict read (no acts word, no contact verdicts): nothing of the step's own.
        if data.path[c] < 0 or (acts.shape[0] == 0 and data.acts.shape[0] == 0):
            if data.acts.shape[0] > 0:
                data.acts[c] = 0
            return data.world[c]
    admitted, separating, art_a, art_b, world, phi = _admit_contact(
        c,
        data,
        body_to_articulation,
        art_to_world,
        body_flags,
        body_has_response_dofs,
        is_free_rigid,
        contact_gap_gate,
        same_articulation_contact_gap_gate,
        articulation_pair_contact_gap_gate,
        friction_patches,
        twists,
        kept,
    )
    if data.acts.shape[0] > 0:
        data.acts[c] = wp.where(admitted and not separating, 1, 0)
    if kept != 0:
        if not separating and acts.shape[0] > 0:
            acts[world] = 1
        return world
    # Its articulations, admitted or not: its body pair's trees join the world's islands (reserve_world_rows).
    data.art0[c] = art_a
    data.art1[c] = art_b
    if not admitted:
        # Its world still names the run of frame positions it lies in (_heads_pair).
        data.world[c] = world
        data.slot[c] = -1
        data.path[c] = -1
        return world
    if not separating and acts.shape[0] > 0:
        acts[world] = 1
    a_has_dofs = art_a >= 0 and data.response_dofs[art_a] > 0
    b_has_dofs = art_b >= 0 and data.response_dofs[art_b] > 0
    # Classify: MF path if both sides are free rigid or ground.
    is_mf = 0
    if has_free_rigid != 0:
        a_is_mf_compatible = not a_has_dofs or is_free_rigid[art_a] != 0
        b_is_mf_compatible = not b_has_dofs or is_free_rigid[art_b] != 0
        if a_is_mf_compatible and b_is_mf_compatible:
            is_mf = 1

    # Every normal survives; only the selected patch anchors receive tangent rows.
    slots_needed = 1
    add_friction = enable_friction != 0 and phi <= contact_friction_gap_threshold and friction_patches.weight[c] > 0.0
    if add_friction:
        slots_needed = 3 + patch_spin_rows(friction_patches, c, data.spin)
    data.slots_needed[c] = slots_needed
    data.world[c] = world
    data.path[c] = is_mf
    return world


@wp.kernel
def admit_world_contacts(
    data: ContactInput,
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    friction_patches: FrictionPatches,
    twists: wp.array[wp.spatial_vector],
    pass_words: wp.array[int],
    world_held: wp.array[int],
    # outputs
    acts: wp.array[int],
    world_contact_first: wp.array[int],
    world_contact_last: wp.array[int],
    contact_worlds: wp.array[int],
    contact_worlds_listed: wp.array[int],
):
    """Classify every active contact and size its rows, a thread a contact in the frame's order (``data`` binds the
    frame's records), so its records and allocation are read and written where every later kernel reads them; its
    rows are reserved after, in the same order (:func:`reserve_world_rows`), from the range of positions each
    world's contacts with a responding endpoint span, admitted or not (``world_contact_first``, ``world_contact_last``),
    for the worlds listed where their span starts (``contact_worlds``, counted in ``contact_worlds_listed[0]``).

    A contact admitted by :func:`_admit_contact` needs 1 row for its normal and, when enabled below the friction gap
    threshold (and, under persistent patches, for a loaded anchor), 2 adjacent rows for its friction, and a third for
    its spin row where the plan lets it carry one (``data.spin``: its region has a footprint, or contact torsion gives
    every friction contact one) (:func:`.friction_patches.patch_spin_rows`). Contacts where both sides are free rigid
    bodies (or ground) take matrix-free rows (``path`` 1), the others world-level dense rows (``path`` 0); a contact
    without rows has ``slot`` and ``path`` -1. A positive contact gap gate excludes wider speculative contacts before
    any route reserves rows; zero disables the gate.

    Each contact's verdict (``data.acts``) is 1 where it is admitted and its rows may act at the predictor (its normal
    does not separate, :func:`.row_laws.separates`, or the predictor publishes no twist), else 0 (empty: no verdict);
    such a contact marks its world's ``acts`` word (:data:`.placement.WORLD_ACTS`; empty: no word).

    The contact set is a collision pass's: on a later step of the pass (``pass_words[0]``,
    :data:`.placement.CONTACT_PASS`) each contact keeps the pass's admission, class and size, and only its verdict is
    the step's (a contact the pass gated out waits for the next pass); only the worlds that do not keep their rows
    (``world_held``) list their span for the reservation.
    """
    c = wp.tid()
    world = int(-1)
    kept = pass_words[0]
    if c < data.count[0]:
        world = _admit_world_contact(
            c,
            data,
            body_to_articulation,
            art_to_world,
            body_flags,
            body_has_response_dofs,
            is_free_rigid,
            has_free_rigid,
            contact_gap_gate,
            same_articulation_contact_gap_gate,
            articulation_pair_contact_gap_gate,
            enable_friction,
            contact_friction_gap_threshold,
            friction_patches,
            twists,
            kept,
            acts,
        )
        if kept != 0 and world >= 0:
            if world_held[world] != 0:
                world = -1
    # The world's range of frame positions, whose rows reserve_world_rows reserves in order: one atomic pair
    # per world and warp.
    _record_world_span(world, c, world_contact_first, world_contact_last, contact_worlds, contact_worlds_listed)


@wp.func
def _immovable_side(data: ContactInput, c: int) -> bool:
    """Whether contact ``c`` has a side without a body (a static shape)."""
    s0 = data.shape0[c]
    s1 = data.shape1[c]
    if s0 < 0 or s1 < 0:
        return True
    return data.shape_body[s0] < 0 or data.shape_body[s1] < 0


ROW_SLOT_UNBOUNDED = wp.constant(2**31 - 1)
"""First rejected slot of a row family that rejected none this step."""

# A world's first contact position before it held any.
_NO_CONTACT = wp.constant(2**31 - 1)

RESERVE_LANES = wp.constant(32)
"""The lanes of a world's reservation (:func:`reserve_world_rows`): a warp on CUDA, one thread on the CPU."""

RESERVE_BLOCK = 64
"""Threads of a block of :func:`reserve_world_rows` at a warp a world: two worlds' lanes."""

RESERVE_WARPS = 8
"""Warps a world of :func:`reserve_world_rows` at most (:func:`reserve_warps`): a block of them, one world."""


@wp.struct
class RowOrder:
    """The row order's records over the frame's positions (:func:`reserve_world_rows`), each world's over its own run of
    positions: each position's body-pair key; the world's span of positions (``world_contact_first``,
    ``world_contact_last``, which the admission records); at each of its contacts the admitted contacts' rows of the
    four lists before it in the world's frame order (``rows_frame``) and its body pair's index among the world's
    (``pairs_before``); its ``j``-th pair's head at ``first + j`` (``pairs``), the trees it touches (``pair_sides``,
    :func:`.row_records.row_sides`), its island and its rank there (``pair_island``, ``pair_rank``), and the index of
    the pair the row order visits ``q``-th at ``first + q`` (``row_pairs``); at each pair's head the rows of each list
    of the pairs before it in the row order
    (``rows_before``); and by articulation, the islands' union-find parents, sizes, and first places in the row order
    with their strides' inverses (``island_parent``, ``island_size``, ``island_base``), each tree its own root and
    size zero between reservations; whether the contact frame is the last build's (``frame_kept``, the patch
    builder's word), whose islands' order the last reservation left in ``row_pairs``; and by world, what a batch of the
    frame order leaves the next in a world's block of warps (``turns``: the rows of the lists and the pairs before
    it, :func:`_frame_turns`)."""

    pair_keys: wp.array[wp.int64]
    world_contact_first: wp.array[int]
    world_contact_last: wp.array[int]
    rows_frame: wp.array[wp.vec4i]
    pairs_before: wp.array[int]
    pairs: wp.array[int]
    pair_sides: wp.array[wp.vec2i]
    pair_island: wp.array[int]
    pair_rank: wp.array[int]
    row_pairs: wp.array[int]
    rows_before: wp.array[wp.vec4i]
    island_parent: wp.array[wp.int32]
    island_size: wp.array[int]
    island_base: wp.array[wp.vec2i]
    frame_kept: wp.array[int]
    turns: wp.array2d[int]


GROUP_MEMBERS = wp.constant(4)
"""The body pairs of a group (:class:`RowGroups`) at most: a unit of rows between the same two trees."""

GROUP_COLORS = wp.constant(64)
"""A grouped world's colors at most (:class:`RowGroups`): a tree's used colors are one 64-bit word."""

_NO_KEY = wp.constant(wp.int64(-1))


@wp.struct
class RowGroups:
    """The dense line of a world of many trees by colors of groups (:func:`_group_rows`), each world's at its slot
    (``slot``, -1 for a world that cannot take it; the slots' tables row-major at their capacities).

    A group is up to :data:`GROUP_MEMBERS` body pairs of one list (with an immovable side, or the others) that touch
    the same trees, their rows together; its color the first-fit color with the largest groups first (their rows
    descending, a size's groups by their leaders, its first pairs, in the row order, the lists in turn): the least color
    no group before it on one of its trees holds (:func:`color_largest_first`; groups re-form at every reservation,
    so none keeps a last color as its tier). Groups
    of one color share no tree, so a sweep steps them at once, a warp a group, their rows in order; the colors in
    turn. A world takes it at a reservation where its groups meet the rule (``min_units`` groups and ``need`` groups a
    color, the colored mapping's :class:`.sweep_units.ColoredRule`) and its colors fit a word: its contact rows are
    then placed color-major (each color's groups in their leaders' order, each group's pairs in the row order), else
    as the row order places them.

    The tables a grouped world's sweep reads: its colors (``colors``, 0 where it is not grouped), each color's first
    group (``offsets``, the colors' count after), each group's first row (``rows``, its end after the last), rows of
    the line, and each group's places on its trees (``follows``: on its first tree, bits 0-7, and on its second, bits
    8-15, the groups before it in the colors' order, plus 64 where one follows it), which a sweep runs it after. The reservation's scratch: each pair of the sequence's head, list, leader and place among its group's
    (``pairs``), each leader's members' rows (``members``) and its color with its place in it (``places``), each
    color's groups and rows (``totals``), a key table over the groups' trees and lists (``keys``, ``open``: the pairs
    counted and the open group's leader), each tree's colors in use (``used``, zero between reservations), what the
    warps' turns pass on (``turns``: the sequence's count and its leaders, :func:`_group_rows`), and the coloring's
    scratch (``starts``, ``order``, ``cells``: :func:`color_largest_first`, the groups its vertices)."""

    slot: wp.array[int]
    colors: wp.array[int]
    offsets: wp.array2d[int]
    rows: wp.array2d[int]
    follows: wp.array2d[int]
    pairs: wp.array2d[wp.vec4i]
    members: wp.array2d[int]
    places: wp.array2d[wp.vec4i]
    totals: wp.array2d[wp.vec2i]
    keys: wp.array2d[wp.int64]
    open: wp.array2d[wp.vec2i]
    used: wp.array[wp.int64]
    turns: wp.array2d[int]
    starts: wp.array2d[int]
    order: wp.array2d[int]
    cells: wp.array2d[int]
    min_units: int
    need: float


def row_groups(slot, slots: int, rows: int, trees: int, min_units: int, need: float, device) -> RowGroups:
    """The tables of :class:`RowGroups` for ``slots`` worlds (``slot``: each world's, -1 for none) of ``rows`` dense
    rows a line and ``trees`` trees, by the rule's ``min_units`` and ``need``; none for no slots."""
    groups = RowGroups()
    groups.slot = wp.array(np.asarray(slot, dtype=np.int32), dtype=int, device=device)
    groups.colors = wp.zeros(slots, dtype=int, device=device)
    groups.offsets = wp.zeros((slots, int(GROUP_COLORS) + 1), dtype=int, device=device)
    groups.rows = wp.zeros((slots, rows + 1), dtype=int, device=device)
    groups.follows = wp.zeros((slots, rows), dtype=int, device=device)
    groups.pairs = wp.zeros((slots, rows), dtype=wp.vec4i, device=device)
    groups.members = wp.zeros((slots, int(GROUP_MEMBERS) * rows), dtype=int, device=device)
    groups.places = wp.zeros((slots, rows), dtype=wp.vec4i, device=device)
    groups.totals = wp.zeros((slots, int(GROUP_COLORS)), dtype=wp.vec2i, device=device)
    groups.keys = wp.zeros((slots, 2 * rows), dtype=wp.int64, device=device)
    groups.open = wp.zeros((slots, 2 * rows), dtype=wp.vec2i, device=device)
    groups.used = wp.zeros(trees, dtype=wp.int64, device=device)
    groups.turns = wp.zeros((slots, 2), dtype=int, device=device)
    groups.starts = wp.zeros((slots, rows + 1), dtype=int, device=device)
    groups.order = wp.zeros((slots, rows), dtype=int, device=device)
    groups.cells = wp.zeros((slots, 2), dtype=int, device=device)
    groups.min_units = int(min_units)
    groups.need = float(need)
    return groups


@wp.func
def _contact_rows(data: ContactInput, c: int) -> wp.vec4i:
    """World contact ``c``'s rows in the reservation's four lists: (dense, free bodies') with an immovable side
    (:func:`_immovable_side`), then (dense, free bodies') of the others; none where it is not admitted (``path`` -1).
    """
    rows = wp.vec4i(0)
    if data.path[c] >= 0:
        k = int(0)
        if data.path[c] != 0:
            k = 1
        if not _immovable_side(data, c):
            k += 2
        rows[k] = data.slots_needed[c]
    return rows


@wp.func
def _lane_scan4(value: wp.vec4i) -> wp.vec4i:
    """Each list's sum of ``value`` over the group's lower lanes."""
    return wp.vec4i(
        lane_scan(value[0], RESERVE_LANES),
        lane_scan(value[1], RESERVE_LANES),
        lane_scan(value[2], RESERVE_LANES),
        lane_scan(value[3], RESERVE_LANES),
    )


@wp.func
def _lane_sum4(value: wp.vec4i) -> wp.vec4i:
    """Each list's sum of ``value`` over the group."""
    return wp.vec4i(
        lane_sum(value[0], RESERVE_LANES),
        lane_sum(value[1], RESERVE_LANES),
        lane_sum(value[2], RESERVE_LANES),
        lane_sum(value[3], RESERVE_LANES),
    )


@wp.func
def _row_stride(n: int) -> int:
    """The stride of an island's row order over its ``n`` body pairs: the golden ratio's share of ``n`` (``n`` times
    ``2**32 / phi``, 2654435769, over ``2**32``, in integers), raised to the first value coprime with ``n``, so
    ``j -> j * stride mod n`` visits every pair once."""
    stride = wp.max(int((wp.int64(n) * wp.int64(2654435769)) >> wp.int64(32)), 1)
    while True:
        a = stride
        b = n
        while b != 0:
            t = a % b
            a = b
            b = t
        if a == 1:
            break
        stride += 1
    return stride


@wp.func
def _row_inverse(n: int) -> int:
    """The place of rank 1 in an island's row order over its ``n`` pairs: the inverse of :func:`_row_stride` modulo
    ``n`` (extended Euclid), so the pair of rank ``j`` takes place ``j * inverse mod n`` and place ``q`` holds rank
    ``q * stride mod n``."""
    t = int(0)
    t_next = int(1)
    r = n
    r_next = _row_stride(n)
    while r_next != 0:
        q = r // r_next
        t_new = t - q * t_next
        t = t_next
        t_next = t_new
        r_new = r - q * r_next
        r = r_next
        r_next = r_new
    if t < 0:
        t += n
    return t


@wp.func
def _world_sync(lanes: int, warps: int):
    """Synchronize a world's threads: its block of ``warps`` warps, or its warp."""
    if warps > 1:
        lane_sync(lanes * warps)
    else:
        lane_sync(RESERVE_LANES)


@wp.func
def _order_islands(
    data: ContactInput,
    order: RowOrder,
    world: int,
    first: int,
    pairs: int,
    lanes: int,
    lane: int,
    warp: int,
    warps: int,
):
    """The world's row order over its ``pairs`` body pairs (``order.pairs`` from ``first``), by its group's ``lanes``:
    ``order.row_pairs[first + q]`` the index of the pair the order visits ``q``-th.

    An island is a component of the world's pairs joined by the trees they touch (:func:`.row_records.row_sides`, the
    sweeps' resources): the frame's pass joined each pair's trees (:mod:`~newton._src.core.union_find`), so each
    island's root is its lowest tree. The islands take the row order in the frame order of their first pairs, each a
    run of its own pairs at a golden-ratio stride over their ranks in frame order (place ``q`` holds rank
    ``q * stride mod n``, :func:`_row_stride` of its own count ``n``). A pair that touches no tree with coordinates
    holds no rows: those follow the islands, in frame order. Every place depends only on the pair's island, so a pair
    of another island never moves it; the islands' places among each other are free (they share no tree). A world that
    is one island takes the order of a stride over all its pairs. Each tree starts as its own root with an empty island,
    as the reservation leaves it after the slots.

    Both passes go a batch of the lanes at a time, over a block of ``warps`` warps in turns as :func:`_frame_turns`
    does (a batch's loads before its turn; the islands' sizes, the pairs placed and the loose ones carried through the
    turns, ``turns``), every place the one warp's."""
    parent = order.island_parent
    size = order.island_size
    batches = (pairs + lanes - 1) // lanes
    # Each pair's island and its rank there in frame order; each island's size.
    grouped = int(0)
    for g in range(warp, batches, warps):
        j = g * lanes + lane
        island = int(-2)
        if j < pairs:
            island = order.pair_sides[first + j][0]
            if island >= 0:
                island = union_find.root_now(parent, island)
        peers = lane_match(island, RESERVE_LANES)
        if warps > 1 and g > 0:
            _turn_wait(_turn(warp, warps))
            grouped = order.turns[world, 5]
        held = int(0)
        if island >= 0:
            held = size[island]
        lane_sync(RESERVE_LANES)
        if island >= 0:
            order.pair_island[first + j] = island
            order.pair_rank[first + j] = held + lane_below(peers, RESERVE_LANES)
            if lane_last(peers, RESERVE_LANES) == lane:
                size[island] = held + lane_count(peers, RESERVE_LANES)
        elif island == -1:
            order.pair_island[first + j] = -1
        grouped += lane_sum(wp.where(island >= 0, 1, 0), RESERVE_LANES)
        lane_sync(RESERVE_LANES)
        if warps > 1:
            if lane == 0:
                order.turns[world, 5] = grouped
            if g + 1 < batches:
                _turn_pass(_turn(warp + 1, warps))
    if warps > 1:
        lane_sync(lanes * warps)
        grouped = order.turns[world, 5]
    # Each pair's place: an island's first pair takes the island's first place (its base and stride's inverse), then
    # every pair its own; the pairs without rows after the islands.
    placed = int(0)
    loose = int(0)
    for g in range(warp, batches, warps):
        j = g * lanes + lane
        island = int(-2)
        rank = int(0)
        if j < pairs:
            island = order.pair_island[first + j]
            rank = order.pair_rank[first + j]
        span = int(0)
        if island >= 0 and rank == 0:
            span = size[island]
        lone = wp.where(island == -1, 1, 0)
        if warps > 1 and g > 0:
            _turn_wait(_turn(warp, warps))
            placed = order.turns[world, 6]
            loose = order.turns[world, 7]
        start = placed + lane_scan(span, RESERVE_LANES)
        lone_rank = loose + lane_scan(lone, RESERVE_LANES)
        if span > 0:
            order.island_base[island] = wp.vec2i(start, _row_inverse(span))
        placed += lane_sum(span, RESERVE_LANES)
        loose += lane_sum(lone, RESERVE_LANES)
        lane_sync(RESERVE_LANES)
        if warps > 1:
            if lane == 0:
                order.turns[world, 6] = placed
                order.turns[world, 7] = loose
            if g + 1 < batches:
                _turn_pass(_turn(warp + 1, warps))
        if island >= 0:
            at = order.island_base[island]
            q = at[0] + int((wp.int64(rank) * wp.int64(at[1])) % wp.int64(size[island]))
            order.row_pairs[first + q] = j
        elif island == -1:
            order.row_pairs[first + grouped + lone_rank] = j
    _world_sync(lanes, warps)


@wp.func
def _rows_before(
    order: RowOrder, world: int, first: int, pairs: int, rows: wp.vec4i, lanes: int, lane: int, warp: int, warps: int
):
    """The row order's rows before each pair (``rows_before`` at its head), the pairs in the row order a batch of the
    lanes at a time, over a block of ``warps`` warps in turns as :func:`_frame_turns` does (the rows seen carried
    through the turns, ``turns``)."""
    batches = (pairs + lanes - 1) // lanes
    seen = wp.vec4i(0)
    for g in range(warp, batches, warps):
        q = g * lanes + lane
        own = wp.vec4i(0)
        head = int(-1)
        if q < pairs:
            j = order.row_pairs[first + q]
            head = order.pairs[first + j]
            own = _pair_rows(order, first, pairs, rows, j)
        lower = _lane_scan4(own)
        if warps > 1 and g > 0:
            _turn_wait(_turn(warp, warps))
            seen = wp.vec4i(order.turns[world, 0], order.turns[world, 1], order.turns[world, 2], order.turns[world, 3])
        if head >= 0:
            order.rows_before[head] = seen + lower
        seen = seen + _lane_sum4(own)
        if warps > 1:
            lane_sync(RESERVE_LANES)
            if lane == 0:
                for k in range(4):
                    order.turns[world, k] = seen[k]
            if g + 1 < batches:
                _turn_pass(_turn(warp + 1, warps))
    _world_sync(lanes, warps)


@wp.func
def _pair_rows(order: RowOrder, first: int, pairs: int, rows: wp.vec4i, j: int) -> wp.vec4i:
    """Pair ``j``'s rows in the four lists: its contacts' (frame order), to the next pair's head or the world's end."""
    after = rows
    if j + 1 < pairs:
        after = order.rows_frame[order.pairs[first + j + 1]]
    return after - order.rows_frame[order.pairs[first + j]]


@wp.func
def _group_key_place(groups: RowGroups, slot: int, key: wp.int64, size: int) -> int:
    """The place of ``key`` in the slot's key table of ``size`` places, claimed where it is new (open addressing; the
    table holds at least twice the keys, so it never fills)."""
    mixed = wp.uint64(key) * wp.uint64(11400714819323198485)
    place = int(wp.int64(mixed >> wp.uint64(33)) % wp.int64(size))
    while True:
        held = groups.keys[slot, place]
        if held == key:
            return place
        if held == _NO_KEY:
            old = wp.atomic_cas(groups.keys, slot, place, _NO_KEY, key)
            if old == _NO_KEY or old == key:
                return place
        place += 1
        if place == size:
            place = 0
    return -1


@wp.func
def _first_free_color(used: wp.int64) -> int:
    """The least color ``used`` does not hold, :data:`GROUP_COLORS` where it holds every one."""
    color = int(0)
    while color < GROUP_COLORS:
        if (used >> wp.int64(color)) & wp.int64(1) == wp.int64(0):
            break
        color += 1
    return color


@wp.func_native("""
    const unsigned long long held = (unsigned long long)used, below = (1ull << color) - 1ull;
#if defined(__CUDA_ARCH__)
    const int before = __popcll(held & below);
#else
    const int before = __builtin_popcountll(held & below);
#endif
    return before | ((held & ~below & ~(1ull << color)) != 0ull ? 64 : 0);
""")
def _tree_place(used: wp.int64, color: int) -> int:
    """A group's place on a tree whose colors are ``used``, its own ``color`` (``< 64``): the colors below it, plus 64
    where a color above it follows."""


@wp.func
def _shares_tree(a: wp.vec2i, b: wp.vec2i) -> bool:
    """Whether two pairs' sides (:func:`.row_records.row_sides`, the second -1 for none) share a tree."""
    if a[0] == b[0] or a[0] == b[1]:
        return True
    return a[1] >= 0 and (a[1] == b[0] or a[1] == b[1])


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    asm volatile("bar.sync %0, 64;" ::"r"(barrier) : "memory");
#else
    (void)barrier;  // one thread runs the world: every turn is its own
#endif
""")
def _turn_wait(barrier: int):
    """Wait for this warp's turn: the warp before it arrived at the named barrier ``barrier`` (its two warps, 64
    threads), its stores fenced at the block's scope (:func:`_turn_pass`). On the CPU one thread runs the world."""


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __threadfence_block();
    asm volatile("bar.arrive %0, 64;" ::"r"(barrier) : "memory");
#else
    (void)barrier;  // one thread runs the world: every turn is its own
#endif
""")
def _turn_pass(barrier: int):
    """Pass the turn to the warp waiting at the named barrier ``barrier``, after this warp's stores (a block fence).
    On the CPU one thread runs the world."""


@wp.func
def _turn(warp: int, warps: int) -> int:
    """The named barrier at which warp ``warp`` of ``warps`` takes its turns (1 + its warp; 0 is the block's)."""
    return 1 + warp % warps


@wp.func
def color_largest_first(
    vertices: wp.array[wp.vec4i],
    count: int,
    used: wp.array[wp.int64],
    starts: wp.array[int],
    order: wp.array[int],
    cells: wp.array[int],
    lanes: int,
    lane: int,
    warp: int,
    warps: int,
) -> int:
    """Color a graph's vertices first-fit, the largest first; return the colors (every thread).

    A vertex is ``vertices[v] = (color, side_a, side_b, size)`` for ``v < count`` (``size < 0``: not one): it holds
    its two sides (``side_b`` -1 for none), indices of ``used``, each side's colors in use (bit ``c``: color ``c``;
    the caller clears them after), and the routine writes its ``color``, the least color no vertex before it on one of
    its sides holds, the vertices taken by their sizes descending, a size's in index order. A color of a sweep lasts
    its longest vertex, so the large vertices share colors and the colors' series is short.

    One rule for every vertex kind: the largest first, where a vertex that persists across colorings keeps its last
    color as its tier (its order against the neighbours it keeps). A grouped world's groups (:func:`_group_rows`)
    re-form from the sequence at every reservation, so none persists and the rule is this routine's first-fit, the
    largest first; the tiers come with the first vertices that persist, a region route's body pairs.

    Scratch: ``starts`` (a size's count, then its first place: the largest size plus one), ``order`` (the vertices in
    that order: ``count``), ``cells`` (two: the largest size, the colors). Over ``warps`` warps of ``lanes`` lanes
    (:func:`.device_preludes.lane_stride`: one on the CPU, where one thread runs the group): the counts take every
    thread; the first warp orders (a batch of the index order at a time, a size's vertices in turn) and colors (a batch
    of the order at a time, a vertex once no lower vertex of its batch on one of its sides waits).
    """
    threads = lanes * warps
    if warp == 0 and lane == 0:
        cells[0] = 0
    if warps > 1:
        lane_sync(threads)
    else:
        lane_sync(lanes)
    largest = int(0)
    for v in range(warp * lanes + lane, count, threads):
        largest = wp.max(largest, vertices[v][3])
    largest = int(lane_maximum(float(largest), lanes))
    if lane == 0:
        wp.atomic_max(cells, 0, largest)
    if warps > 1:
        lane_sync(threads)
    else:
        lane_sync(lanes)
    largest = cells[0]
    for b in range(warp * lanes + lane, largest + 1, threads):
        starts[b] = 0
    if warps > 1:
        lane_sync(threads)
    else:
        lane_sync(lanes)
    for v in range(warp * lanes + lane, count, threads):
        size = vertices[v][3]
        if size >= 0:
            wp.atomic_add(starts, size, 1)
    if warps > 1:
        lane_sync(threads)
    else:
        lane_sync(lanes)
    colors = int(0)
    if warp == 0:
        total = int(0)
        if lane == 0:
            for k in range(largest + 1):
                b = largest - k
                n = starts[b]
                starts[b] = total
                total += n
        total = lane_read(total, 0, lanes)
        lane_sync(lanes)
        for base in range(0, count, lanes):
            v = base + lane
            size = int(-1 - lane)
            if v < count:
                if vertices[v][3] >= 0:
                    size = vertices[v][3]
            peers = lane_match(size, lanes)
            at = int(0)
            if size >= 0:
                at = starts[size] + lane_below(peers, lanes)
            lane_sync(lanes)
            if size >= 0:
                order[at] = v
                if lane_last(peers, lanes) == lane:
                    starts[size] = at + 1
            lane_sync(lanes)
        for base in range(0, total, lanes):
            k = base + lane
            v = int(-1)
            waiting = False
            vertex = wp.vec4i(0, -1, -1, 0)
            if k < total:
                v = order[k]
                vertex = vertices[v]
                waiting = True
            sides = wp.vec2i(vertex[1], vertex[2])
            # The batch's lower vertices that share a side with this one.
            shares = wp.uint32(0)
            for j in range(lanes):
                other = wp.vec2i(lane_read(sides[0], j, lanes), lane_read(sides[1], j, lanes))
                if j < lane and waiting and _shares_tree(sides, other):
                    shares = shares | (wp.uint32(1) << wp.uint32(j))
            lower = shares & lane_ballot(waiting, lanes)
            while lane_any(waiting, lanes):
                votes = lane_ballot(waiting, lanes)
                if waiting and (votes & lower) == wp.uint32(0):
                    held = used[sides[0]]
                    if sides[1] >= 0:
                        held = held | used[sides[1]]
                    color = _first_free_color(held)
                    if color < GROUP_COLORS:
                        bit = wp.int64(1) << wp.int64(color)
                        used[sides[0]] = used[sides[0]] | bit
                        if sides[1] >= 0:
                            used[sides[1]] = used[sides[1]] | bit
                    vertices[v] = wp.vec4i(color, sides[0], sides[1], vertex[3])
                    colors = wp.max(colors, color + 1)
                    waiting = False
                lane_sync(lanes)
        colors = int(lane_maximum(float(colors), lanes))
    if warps > 1:
        if warp == 0 and lane == 0:
            cells[1] = colors
        lane_sync(threads)
        colors = cells[1]
    return colors


@wp.func
def _group_rows(
    order: RowOrder,
    groups: RowGroups,
    slot: int,
    first: int,
    pairs: int,
    rows: wp.vec4i,
    line_first: int,
    lanes: int,
    lane: int,
    warp: int,
    warps: int,
):
    """Group the world's dense pairs, color the groups, and where the world takes the colors (:class:`RowGroups`)
    place its dense contact rows color-major (each pair's ``rows_before``) and write its tables; else its colors
    are 0 and the row order's places stand. ``line_first`` is the line's row of the world's first contact row.

    One pass over the sequence a batch of the lanes at a time forms the groups; one colors them, the largest first
    (:func:`color_largest_first`, the groups its vertices); one over the leaders counts each group's places on its
    trees (its trees' colors below its own); one over the leaders places each group in its color and frees the trees'
    colors; one over the pairs places their rows and the groups' places on their trees (``follows``).

    Over ``warps`` warps of the block (warp ``warp``; one on the CPU, or where a warp is the world's), the warps take
    the passes' batches in turn, warp ``w`` the batches ``w``, ``w + warps``, ...: a batch's own loads and its keys'
    places (open addressing, whose places do not depend on the order keys arrive in) come before its turn; its turn
    (the warp before it passed it, :func:`_turn_wait`) reads what the batches before it left (the sequence's count, the
    open groups, the colors' totals), leaves its own, and passes it on. The batches take their turns in the sequence's
    order, so every group and place is the one warp's. The leaders are a sum over the batches; the passes over the keys
    and the pairs take every thread."""
    capacity = groups.pairs.shape[1]
    threads = lanes * warps
    # A key a pair at most (the pairs past the capacity take none): twice the keys' places.
    keys = wp.min(2 * pairs, groups.keys.shape[1])
    for h in range(warp * lanes + lane, keys, threads):
        groups.keys[slot, h] = _NO_KEY
        groups.open[slot, h] = wp.vec2i(0, -1)
    if warps > 1:
        if warp == 0 and lane == 0:
            groups.turns[slot, 0] = 0
            groups.turns[slot, 1] = 0
        lane_sync(threads)
    else:
        lane_sync(RESERVE_LANES)
    # The sequence: the pairs with dense rows, the list with an immovable side, then the others, each in the row order.
    # Each pair's place among its key's (its trees and list) in the sequence, a group every GROUP_MEMBERS of them; each
    # leader's first-fit color, a leader of the batch once no lower leader of the batch on one of its trees waits.
    batches = (pairs + lanes - 1) // lanes
    count = int(0)
    leaders = int(0)
    for g in range(warp, 2 * batches, warps):
        half = g // batches
        base = (g - half * batches) * lanes
        lst = half * 2
        q = base + lane
        head = int(-1)
        own = int(0)
        key = wp.int64(-1)
        sides = wp.vec2i(-1, -1)
        if q < pairs:
            j = order.row_pairs[first + q]
            own = _pair_rows(order, first, pairs, rows, j)[lst]
            if own > 0:
                head = order.pairs[first + j]
                sides = order.pair_sides[first + j]
                key = (wp.int64(half) << wp.int64(62)) | (wp.int64(sides[0]) << wp.int64(31))
                key = key | wp.int64(sides[1] + 1)
        candidate = head >= 0
        # Its key's place (claimed by every pair with rows: a pair past the capacity holds a place no read finds).
        place = int(-1 - lane)
        if candidate:
            place = _group_key_place(groups, slot, key, keys)
        # The batch's turn: what the batches before it left.
        if warps > 1:
            if g > 0:
                _turn_wait(_turn(warp, warps))
            count = groups.turns[slot, 0]
        s = count + lane_scan(wp.where(candidate, 1, 0), RESERVE_LANES)
        active = candidate and s < capacity
        peers = lane_match(wp.where(active, place, -1 - lane), RESERVE_LANES)
        held = wp.vec2i(0, -1)
        if active:
            held = groups.open[slot, place]
        rank = held[0] + lane_below(peers, RESERVE_LANES)
        member = rank % GROUP_MEMBERS
        # A member's leader: the open group's (counted before this batch) or a peer of this batch.
        source = int(lane)
        if active and member > 0 and rank - member >= held[0]:
            skip = rank - member - held[0]
            rest = wp.uint32(peers)
            while skip > 0:
                rest = rest & (rest - wp.uint32(1))
                skip -= 1
            source = lane_first(rest, RESERVE_LANES)
        peer = lane_read(s, source, RESERVE_LANES)
        leader = int(s)
        if active and member > 0:
            leader = wp.where(rank - member >= held[0], peer, held[1])
        waiting = active and member == 0
        lane_sync(RESERVE_LANES)
        if active:
            groups.pairs[slot, s] = wp.vec4i(head, lst, leader, member)
            if lane_last(peers, RESERVE_LANES) == lane:
                groups.open[slot, place] = wp.vec2i(rank + 1, leader)
            if waiting:
                groups.members[slot, GROUP_MEMBERS * s] = own
                for m in range(1, GROUP_MEMBERS):
                    groups.members[slot, GROUP_MEMBERS * s + m] = 0
        lane_sync(RESERVE_LANES)
        if active and member > 0:
            groups.members[slot, GROUP_MEMBERS * leader + member] = own
        if waiting:
            # The leader's trees, until the coloring and the leaders' pass place it.
            groups.places[slot, s] = wp.vec4i(-1, sides[0], sides[1], 0)
        count += lane_sum(wp.where(active, 1, 0), RESERVE_LANES)
        leaders += lane_sum(wp.where(active and member == 0, 1, 0), RESERVE_LANES)
        if warps > 1:
            if lane == 0:
                groups.turns[slot, 0] = count
            if g + 1 < 2 * batches:
                _turn_pass(_turn(warp + 1, warps))
    if warps > 1:
        # The leaders over the warps; the count the last batch's.
        if lane == 0:
            wp.atomic_add(groups.turns, slot, 1, leaders)
        lane_sync(threads)
        count = groups.turns[slot, 0]
        leaders = groups.turns[slot, 1]
    count = wp.min(count, capacity)
    # The groups as the coloring's vertices: each leader's place its trees and rows, every other pair's none.
    if warps == 1:
        lane_sync(RESERVE_LANES)
    for s in range(warp * lanes + lane, count, threads):
        vertex = wp.vec4i(-1, -1, -1, -1)
        if groups.pairs[slot, s][3] == 0:
            size = int(0)
            for m in range(GROUP_MEMBERS):
                size += groups.members[slot, GROUP_MEMBERS * s + m]
            lead = groups.places[slot, s]
            vertex = wp.vec4i(-1, lead[1], lead[2], size)
        groups.places[slot, s] = vertex
    colors = color_largest_first(
        groups.places[slot],
        count,
        groups.used,
        groups.starts[slot],
        groups.order[slot],
        groups.cells[slot],
        lanes,
        lane,
        warp,
        warps,
    )
    takes = (
        colors <= GROUP_COLORS
        and leaders >= groups.min_units
        and float(leaders) >= groups.need * float(colors)
        and line_first + rows[0] + rows[2] <= capacity
    )
    if takes:
        for c in range(warp * lanes + lane, colors, threads):
            groups.totals[slot, c] = wp.vec2i(0, 0)
    if warps > 1:
        lane_sync(threads)
    else:
        lane_sync(RESERVE_LANES)
    if takes:
        # Each group's places on its trees, the groups before it on each in the colors' order (its tree's colors below
        # its own), in its vertex's last word (its size is the members'), before the leaders' pass frees the colors.
        for s in range(warp * lanes + lane, count, threads):
            if groups.pairs[slot, s][3] == 0:
                lead = groups.places[slot, s]
                follow = _tree_place(groups.used[lead[1]], lead[0])
                if lead[2] >= 0:
                    follow = follow | (_tree_place(groups.used[lead[2]], lead[0]) << 8)
                groups.places[slot, s] = wp.vec4i(lead[0], lead[1], lead[2], follow)
        if warps > 1:
            lane_sync(threads)
        else:
            lane_sync(RESERVE_LANES)
    # The leaders in order: each tree's colors freed for the next reservation; where the world takes the colors, each
    # group's place among its color's groups and rows (a batch at a time), and each color's groups and rows.
    rounds = (count + lanes - 1) // lanes
    for g in range(warp, rounds, warps):
        s = g * lanes + lane
        color = int(-1 - lane)
        size = int(0)
        follow = int(0)
        if s < count:
            if groups.pairs[slot, s][3] == 0:
                lead = groups.places[slot, s]
                groups.used[lead[1]] = wp.int64(0)
                if lead[2] >= 0:
                    groups.used[lead[2]] = wp.int64(0)
                color = lead[0]
                follow = lead[3]
                for m in range(GROUP_MEMBERS):
                    size += groups.members[slot, GROUP_MEMBERS * s + m]
        if takes:
            peers = lane_match(color, RESERVE_LANES)
            below = int(0)
            for k in range(lanes):
                peer_size = lane_read(size, k, RESERVE_LANES)
                if k < lane and (peers >> wp.uint32(k)) & wp.uint32(1) != wp.uint32(0):
                    below += peer_size
            if warps > 1 and g > 0:
                _turn_wait(_turn(warp, warps))
            held = wp.vec2i(0, 0)
            if color >= 0:
                held = groups.totals[slot, color]
            lane_sync(RESERVE_LANES)
            if color >= 0:
                at = wp.vec4i(color, held[0] + lane_below(peers, RESERVE_LANES), held[1] + below, follow)
                groups.places[slot, s] = at
                if lane_last(peers, RESERVE_LANES) == lane:
                    groups.totals[slot, color] = wp.vec2i(
                        held[0] + lane_count(peers, RESERVE_LANES), held[1] + below + size
                    )
            if warps > 1 and g + 1 < rounds:
                _turn_pass(_turn(warp + 1, warps))
        lane_sync(RESERVE_LANES)
    if warps > 1:
        lane_sync(threads)
    if not takes:
        if warp == 0 and lane == 0:
            groups.colors[slot] = 0
        return
    # The colors' first groups and rows.
    if warp == 0 and lane == 0:
        start = wp.vec2i(0, 0)
        for c in range(colors):
            total = groups.totals[slot, c]
            groups.totals[slot, c] = start
            groups.offsets[slot, c] = start[0]
            start = start + total
        groups.offsets[slot, colors] = start[0]
        groups.rows[slot, start[0]] = line_first + start[1]
        groups.colors[slot] = colors
    if warps > 1:
        lane_sync(threads)
    else:
        lane_sync(RESERVE_LANES)
    # Each pair's rows before it in the line's contact rows (the slot pass adds the list with an immovable side's rows
    # to the others'), each group's first row.
    for s in range(warp * lanes + lane, count, threads):
        pair = groups.pairs[slot, s]
        lead = groups.places[slot, pair[2]]
        start = groups.totals[slot, lead[0]]
        offset = start[1] + lead[2]
        for m in range(GROUP_MEMBERS):
            if m < pair[3]:
                offset += groups.members[slot, GROUP_MEMBERS * pair[2] + m]
        if pair[3] == 0:
            groups.rows[slot, start[0] + lead[1]] = line_first + offset
            groups.follows[slot, start[0] + lead[1]] = lead[3]
        b = order.rows_before[pair[0]]
        if pair[1] == 0:
            b[0] = offset
        else:
            b[2] = offset - rows[0]
        order.rows_before[pair[0]] = b
    if warps > 1:
        lane_sync(threads)
    else:
        lane_sync(RESERVE_LANES)


def reserve_warps(worlds: int, device) -> int:
    """Warps a world of the reservation (:func:`reserve_world_rows_kernel`'s ``warps``): one where the worlds fill the
    device at a warp each, else as many as fill it (:func:`~newton._src.core.execution.fill_lanes`), at most a block's
    :data:`RESERVE_WARPS`; one on the CPU."""
    lanes = int(RESERVE_LANES)
    return fill_lanes(worlds, device, least=lanes, most=lanes * RESERVE_WARPS) // lanes


@wp.func
def _frame_turns(
    data: ContactInput,
    order: RowOrder,
    world: int,
    first: int,
    end: int,
    kept: int,
    lanes: int,
    lane: int,
    warp: int,
    warps: int,
):
    """The frame's order: each contact's rows before it and its pair (``rows_frame``, ``pairs_before``), each pair's
    head and, where the frame is new, its trees (``pairs``, ``pair_sides``), a batch of the world's lanes at a time;
    the frame's rows and pairs in ``turns`` after it.

    Over a block of ``warps`` warps, warp ``w`` takes the batches ``w``, ``w + warps``, ...: a batch's loads and its own
    sums come before its turn (the warp before it passed it, :func:`_turn_wait`), which reads what the batches before it
    left (the rows of the lists and the pairs before it, ``turns``), writes the batch's places, leaves its own, and
    passes it on: every place is the one warp's. A world's one warp carries them itself and joins each new pair's trees
    (:func:`_order_islands`; a block's warps joined them before)."""
    batches = (end - first + lanes - 1) // lanes
    rows = wp.vec4i(0)
    pairs = int(0)
    for g in range(warp, batches, warps):
        c = first + g * lanes + lane
        mine = bool(False)
        heads = bool(False)
        own = wp.vec4i(0)
        sides = wp.vec2i(-1, -1)
        if c < end:
            if data.world[c] == world:
                mine = True
                heads = c == first or order.pair_keys[c] != order.pair_keys[c - 1]
                own = _contact_rows(data, c)
                if heads and kept == 0:
                    sides = row_sides(data.art0[c], data.art1[c], data.response_dofs)
                    if warps == 1 and sides[1] >= 0:
                        union_find.union(order.island_parent, sides[0], sides[1])
        votes = lane_ballot(heads, RESERVE_LANES)
        lower = _lane_scan4(own)
        if warps > 1 and g > 0:
            _turn_wait(_turn(warp, warps))
            rows = wp.vec4i(order.turns[world, 0], order.turns[world, 1], order.turns[world, 2], order.turns[world, 3])
            pairs = order.turns[world, 4]
        if mine:
            pair = pairs + lane_below(votes, RESERVE_LANES) + wp.where(heads, 1, 0) - 1
            order.pairs_before[c] = pair
            order.rows_frame[c] = rows + lower
            if heads:
                order.pairs[first + pair] = c
                if kept == 0:
                    order.pair_sides[first + pair] = sides
        rows = rows + _lane_sum4(own)
        pairs += lane_count(votes, RESERVE_LANES)
        if warps > 1:
            lane_sync(RESERVE_LANES)
            if lane == 0:
                for k in range(4):
                    order.turns[world, k] = rows[k]
                order.turns[world, 4] = pairs
            if g + 1 < batches:
                _turn_pass(_turn(warp + 1, warps))
    if warps == 1 and lane == 0:
        for k in range(4):
            order.turns[world, k] = rows[k]
        order.turns[world, 4] = pairs


@cache
def reserve_world_rows_kernel(grouped: bool, warps: int = 1):
    """The reservation kernel (:func:`reserve_world_rows`, its body below): ``grouped`` where the model holds worlds
    that can take the dense line by colors of groups (a slot of :class:`RowGroups`), whose reservations group them;
    without, the kernel holds none of the grouping's code (a one-warp walk over a world's pairs that the others never
    take, whose code alone cost the cube pile's one-world reservation 70% more time). ``warps`` (:func:`reserve_warps`)
    a world, a world a block of them past one: the passes whose contacts are their threads' own (the heads' unions,
    the slots) take every warp, the ordered passes the first."""

    @wp.kernel(enable_backward=False, module="unique")
    def reserve_world_rows(
        data: ContactInput,
        max_constraints: int,
        mf_max_constraints: int,
        row_capacity_telemetry: int,
        lines: wp.vec2i,
        arena_lines: wp.array[int],
        order: RowOrder,
        groups: RowGroups,
        world_slot_counter: wp.array[int],
        mf_slot_counter: wp.array[int],
        contact_worlds: wp.array[int],
        # outputs
        arena_cursor: wp.array[int],
        world_run: wp.array[wp.vec2i],
        body_base: wp.array[int],
        contact_worlds_listed: wp.array[int],
        dense_dropped_contact_rows: wp.array[int],
        mf_dropped_contact_rows: wp.array[int],
        dense_first_rejected_slot: wp.array[int],
        mf_first_rejected_slot: wp.array[int],
        units: wp.array2d[int],
        contact_rows_kept: wp.array2d[int],
    ):
        """Reserve the rows of every admitted contact (:func:`admit_world_contacts`) in its world's row order, so a world's
        rows follow its contacts' geometry (the frame's pair and contact keys), never the order the collision's threads
        wrote them in: a listed world (``contact_worlds``, the admission's, counted in ``contact_worlds_listed[0]``) a lane
        group (:data:`RESERVE_LANES`), one launch, which a replay runs at the contact count's lane groups, at most a
        world's (none on a step without contacts): every listed world holds a contact, and the list's own count is written
        after the replay's updater, so it cannot size the launch. The groups past the list return; the last listed group to
        finish clears the list for the next step (``contact_worlds_listed[1]`` counts them).

        A world's contacts lie in one run of the frame's positions (``order.world_contact_first`` to
        ``order.world_contact_last``, which the admission recorded; a contact of no world may lie between them). The
        lanes take it in passes. In the frame's order, each contact's rows of the four lists before it and its body pair
        (a pair heads at the run's first position or at a change of the frame's pair key, admitted or not: a pair's
        contacts share their bodies, so they respond alike), each pair's head, which joins the trees it touches. Then the
        world's islands (:func:`_order_islands`), each island's pairs at a golden-ratio stride over their own ranks. In
        that row order, each pair's rows before it (each pair's contacts together in frame order). Then each contact's
        slot, each pair's head restoring its trees. The islands' order is a function of the frame's pairs alone, so a
        build that keeps the last build's frame (``order.frame_kept``) reads the order the last reservation left and
        joins no trees.

        The row order is the Gauss-Seidel order, so a body's rows must not move with rows it does not share an island
        with: two bodies that never touch must not change each other's motion. An island's order is a function of its own
        pairs' keys alone, and pairs of different islands share no tree, so whatever else the world holds leaves every
        island's sweep as it was. Within an island the stride spreads the pairs: a sorted frame lays a body's pairs side
        by side, and their rows in that order chain through shared bodies into a deep trace for the colored sweeps, while
        a pair's own contacts, which share both bodies, chain whatever their order and converge best together. It reads
        only the frame's keys and the bodies' response, so a contact's admission moves no other's rows, and nor does a
        contact that can never take rows.

        A contact takes the next rows of its family in its world's line: the dense rows, or the free bodies' (``path`` 1),
        which follow the dense class's (one row index for both classes); the contacts with an immovable side first
        (:func:`_immovable_side`), then the others, each in the world's row order. Supports first is the order a sweep
        propagates a stack's load in from the ground up. A contact's slot is its first row. A world of many trees with a
        slot of ``groups`` places its dense contact rows by colors of groups instead where its groups meet the rule
        (:class:`RowGroups`, :func:`_group_rows`), each pair's rows still together.

        Counters only grow (each world's lane 0 adds every contact's rows once its slots are taken, and clears the world's
        run for the next step): a contact whose rows do not fit records its slot in the family's ``*_first_rejected_slot``
        (per world, preset to a large value), and every later contact's rows start past it, so the finalize kernels
        truncate the row count there; such a contact has no rows (``slot`` and ``path`` -1). Where the one writer takes the
        rows (``units`` holds the placement's unit sources; empty otherwise), each dense row's unit source is written with
        its slot.

        Each listed world records the rows its contacts added by class (``contact_rows_kept``; -1 in the dense class where
        a row was rejected), which a later step of the collision pass that keeps the world's slots adds instead
        (:func:`.placement._bound_topology`): only the worlds that do not keep them are listed, and a step where none
        reserves launches none (its extent is zero).
        """
        lanes = lane_stride(RESERVE_LANES)
        lane = lane_index(RESERVE_LANES)
        listed = contact_worlds_listed[0]
        item = wp.tid() // lanes
        warp = int(0)
        threads = lanes
        if wp.static(warps > 1):
            # A world a block of warps.
            threads = lanes * warps
            item = wp.tid() // threads
            warp = (wp.tid() // lanes) % warps
        if item >= listed:
            return
        world = contact_worlds[item]
        # A kept frame's islands are ordered: the last reservation, of the same frame, left their order in row_pairs.
        kept = order.frame_kept[0]
        first = order.world_contact_first[world]
        end = wp.min(order.world_contact_last[world] + 1, data.count[0])
        if wp.static(warps > 1):
            # Each pair's head joins its trees' islands (_order_islands), a contact a thread of the block: the union
            # by lowest id gives every island the same root whatever order the unions run in.
            if kept == 0:
                for c in range(first + warp * lanes + lane, end, threads):
                    if data.world[c] == world:
                        if c == first or order.pair_keys[c] != order.pair_keys[c - 1]:
                            sides = row_sides(data.art0[c], data.art1[c], data.response_dofs)
                            if sides[1] >= 0:
                                union_find.union(order.island_parent, sides[0], sides[1])
            lane_sync(threads)
        # The frame's order (_frame_turns): each contact's rows before it and its pair, each pair's head.
        _frame_turns(data, order, world, first, end, kept, lanes, lane, warp, warps)
        if wp.static(warps > 1):
            lane_sync(threads)
        else:
            lane_sync(RESERVE_LANES)
        rows = wp.vec4i(0)
        pairs = int(0)
        if end > first:
            rows = wp.vec4i(order.turns[world, 0], order.turns[world, 1], order.turns[world, 2], order.turns[world, 3])
            pairs = order.turns[world, 4]
        # The row order (_order_islands), then each pair's rows before it (_rows_before).
        if kept == 0:
            _order_islands(data, order, world, first, pairs, lanes, lane, warp, warps)
        _rows_before(order, world, first, pairs, rows, lanes, lane, warp, warps)
        if wp.static(warps > 1):
            lane_sync(threads)
        # A world of many trees: its dense contact rows by colors of groups, where it takes them (RowGroups), over the
        # world's warps in turn.
        if wp.static(grouped):
            group_slot = groups.slot[world]
            if group_slot >= 0:
                _group_rows(
                    order, groups, group_slot, first, pairs, rows, world_slot_counter[world], lanes, lane, warp, warps
                )
        # The world's line (``lines``: the store's line length and its home lines; ``arena_lines``, the arena's lines
        # mapped): its home where its rows fit the line, else a run of the arena's lines, one claim (the run it holds
        # this frame where that holds them), which it keeps while it holds its slots; its free rows from its body base,
        # its dense rows' end. Where the arena holds no run, the rows past its home line are rejected (the capacity
        # status). The claims count in device memory (``arena_cursor``: the lines taken, then the claims refused), which
        # the arena's one writer publishes (:func:`.placement._take_runs`): an atomic a claim on the pinned record
        # would cross the host bus once a world.
        dense_total = world_slot_counter[world] + rows[0] + rows[2]
        free_total = mf_slot_counter[world] + rows[1] + rows[3]
        base = wp.min(dense_total, max_constraints)
        extent = dense_total
        if free_total > 0:
            extent = base + free_total
        line = data.first[world]
        dense_cap = max_constraints
        free_cap = mf_max_constraints
        if extent > lines[0]:
            if warp == 0 and lane == 0:
                need = (extent + lines[0] - 1) // lines[0]
                held_run = world_run[world]
                if held_run[0] < 0 or held_run[1] < need:
                    held_run = wp.vec2i(-1, 0)
                    # A store without an arena (lines as long as the capacities) has no run to claim.
                    if arena_lines[0] > 0:
                        claim = wp.atomic_add(arena_cursor, 0, need)
                        if claim + need <= arena_lines[0]:
                            held_run = wp.vec2i(lines[1] + claim, need)
                        else:
                            wp.atomic_add(arena_cursor, 1, 1)
                    world_run[world] = held_run
            if wp.static(warps > 1):
                lane_sync(threads)
            else:
                lane_sync(RESERVE_LANES)
            run = world_run[world][0]
            if run >= 0:
                line = run
            else:
                dense_cap = wp.min(max_constraints, lines[0])
                base = wp.min(base, dense_cap)
                free_cap = wp.min(mf_max_constraints, wp.max(lines[0] - base, 0))
        elif warp == 0 and lane == 0:
            world_run[world] = wp.vec2i(-1, 0)
        if warp == 0 and lane == 0:
            body_base[line] = base
        # Each admitted contact's slot; each pair's head restores its trees (_order_islands' starting state).
        for c in range(first + warp * lanes + lane, end, threads):
            if data.world[c] != world:
                continue
            pair = first + order.pairs_before[c]
            head = order.pairs[pair]
            if kept == 0 and head == c:
                sides = order.pair_sides[pair]
                for side in range(2):
                    if sides[side] >= 0:
                        order.island_parent[sides[side]] = sides[side]
                        order.island_size[sides[side]] = 0
            path = data.path[c]
            if path < 0:
                continue
            before = order.rows_before[head] + order.rows_frame[c] - order.rows_frame[head]
            k = int(0)
            if path != 0:
                k = 1
            offset = before[k]
            if not _immovable_side(data, c):
                # Past the world's rows of the family with an immovable side.
                offset = rows[k] + before[k + 2]
            n = data.slots_needed[c]
            if path != 0:
                slot = mf_slot_counter[world] + offset
                if slot + n > free_cap:
                    wp.atomic_min(mf_first_rejected_slot, world, slot)
                    if row_capacity_telemetry != 0:
                        wp.atomic_add(mf_dropped_contact_rows, world, n)
                    data.slot[c] = -1
                    data.path[c] = -1
                else:
                    data.slot[c] = slot + base
            else:
                slot = world_slot_counter[world] + offset
                if slot + n > dense_cap:
                    wp.atomic_min(dense_first_rejected_slot, world, slot)
                    if row_capacity_telemetry != 0:
                        wp.atomic_add(dense_dropped_contact_rows, world, n)
                    data.slot[c] = -1
                    data.path[c] = -1
                else:
                    data.slot[c] = slot
                    if units.shape[0] > 0:
                        for r in range(n):
                            units[line, slot + r] = unit_source(UNIT_CONTACT, 4 * c + r)
        if wp.static(warps > 1):
            lane_sync(threads)
        else:
            lane_sync(RESERVE_LANES)
        if warp == 0 and lane == 0:
            # The world's counters, rejected rows included, and its run cleared for the next step; the last listed world
            # to finish clears the list (every listed world read its count before).
            world_slot_counter[world] += rows[0] + rows[2]
            mf_slot_counter[world] += rows[1] + rows[3]
            dense = rows[0] + rows[2]
            rejected = dense_first_rejected_slot[world] != ROW_SLOT_UNBOUNDED
            if rejected or mf_first_rejected_slot[world] != ROW_SLOT_UNBOUNDED:
                dense = -1
            contact_rows_kept[0, world] = dense
            contact_rows_kept[1, world] = rows[1] + rows[3]
            order.world_contact_first[world] = _NO_CONTACT
            order.world_contact_last[world] = -1
            if wp.atomic_add(contact_worlds_listed, 1, 1) == listed - 1:
                contact_worlds_listed[0] = 0
                contact_worlds_listed[1] = 0

    return reserve_world_rows

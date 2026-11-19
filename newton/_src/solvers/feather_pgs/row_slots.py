# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row store's slots: each contact's admission and its rows' reservation, the contact runs that order a
world's rows under sorted contacts, and the counts with their capacity status."""

import warp as wp

from ...core.ordered_reserve import ordered_also, ordered_next, ordered_scan
from ...sim import BodyFlags
from .contact_filters import contact_normal_gap_limit
from .contact_input import ContactInput
from .contact_laws import (
    contact_authored,
    contact_endpoints,
    contact_materials,
    contact_points,
    contact_responds,
    jacobian_entry,
)
from .friction_patches import FrictionPatches, patch_spin_rows
from .row_laws import (
    authored_contact_law,
    contact_capped_beta,
    contact_depenetration_cap,
    contact_material_rhs,
    contact_restitution_rhs,
    contact_rhs,
    separates,
)
from .row_records import UNIT_CONTACT, unit_source

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})

ROW_LISTS = wp.constant(9)
"""The row order's lists, a scan site and a total each: the admitted contacts' rows of the four lists and the body
pairs headed, in the frame's order (:func:`scan_frame_rows`, lists 0-4), then the pairs' rows of the four lists in the
row order (:func:`scan_pair_rows`, lists 5-8)."""

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
    twists: wp.array[wp.spatial_vector],
):
    """Return whether a contact gets rows, whether its rows keep zero impulse at the predictor, and its articulations,
    world and gap: ``(admitted, separates, art_a, art_b, world, phi)``.

    A contact needs an endpoint that responds to its impulses, one world (not
    two, nor none without an articulation) and a gap within the normal gap limit. A contact without a responding
    endpoint has no world (-1): its body pair never takes rows.
    Its normal row separates where its residual at the predictor (the body ``twists`` it publishes; none: never) does
    (:func:`.row_laws.separates`), and its friction and spin rows, bounded by a finite non-negative coefficient of
    its ring's load, keep zero with it. The residual's bias is the row's law: an authored material's rhs
    (:func:`.row_laws.authored_contact_law`, no restitution) where a shape authors one, else the hard law's.
    """
    body_a, body_b, art_a, art_b, world = contact_endpoints(
        data.shape0[c], data.shape1[c], data.shape_body, body_to_articulation, art_to_world
    )
    responds_a = contact_responds(art_a, body_a, data.response_dofs, body_has_response_dofs, body_flags)
    responds_b = contact_responds(art_b, body_b, data.response_dofs, body_has_response_dofs, body_flags)
    phi = float(0.0)
    if not (responds_a or responds_b):
        world = -1
    admitted = world >= 0
    separating = False
    if admitted:
        normal, translation_a, offset_a, translation_b, offset_b, separation, phi = contact_points(
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
        admitted = not (phi > normal_gap_limit)
        if admitted and twists.shape[0] > 0:
            # The free bodies' law where neither side has coordinates of a tree that is not a free body.
            free = (not (art_a >= 0 and data.response_dofs[art_a] > 0) or not a_non_free) and (
                not (art_b >= 0 and data.response_dofs[art_b] > 0) or not b_non_free
            )
            lever_a = offset_a
            lever_b = offset_b
            if art_a >= 0:
                lever_a += translation_a - data.origin[art_a]
            if art_b >= 0:
                lever_b += translation_b - data.origin[art_b]
            lever_a -= 0.5 * separation
            lever_b += 0.5 * separation
            speed_a, known_a = _endpoint_speed(
                data, twists, body_has_response_dofs, body_flags, body_a, art_a, free, normal, lever_a
            )
            speed_b, known_b = _endpoint_speed(
                data, twists, body_has_response_dofs, body_flags, body_b, art_b, free, normal, lever_b
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
        atomicMin(&first.data[world], low);
        atomicMax(&last.data[world], high);
    }
#elif defined(__CUDA_ARCH__)
    if (world >= 0) {
        atomicMin(&first.data[world], contact);
        atomicMax(&last.data[world], contact);
    }
#else
    if (world >= 0) {
        if (contact < first.data[world])
            first.data[world] = contact;
        if (contact > last.data[world])
            last.data[world] = contact;
    }
#endif
"""
)
def _record_world_span(world: int, contact: int, first: wp.array[int], last: wp.array[int]):
    """Merge position ``contact`` into ``world``'s span (-1: none), once per world and warp on CUDA. Every lane of the
    warp that reaches the launch calls it, with ascending positions."""
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
    # outputs
    acts: wp.array[int],
) -> int:
    """Admit, classify and size contact ``c`` (:func:`admit_world_contacts`); returns its world (-1 for none:
    :func:`_admit_contact`), written for every contact, admitted or not."""
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
        twists,
    )
    if data.acts.shape[0] > 0:
        data.acts[c] = wp.where(admitted and not separating, 1, 0)
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
    data.art0[c] = art_a
    data.art1[c] = art_b
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
    # outputs
    acts: wp.array[int],
    world_contact_first: wp.array[int],
    world_contact_last: wp.array[int],
    row_totals: wp.array2d[int],
):
    """Classify every active contact and size its rows, a thread a contact in the frame's order (``data`` binds the
    frame's records), so its records and allocation are read and written where every later kernel reads them; its
    rows are reserved after, in the same order (:func:`reserve_world_contact_rows`), from the range of positions each
    world's contacts with a responding endpoint span, admitted or not (``world_contact_first``, ``world_contact_last``).

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
    """
    c = wp.tid()
    if c == 0:
        # The row order's scans (scan_frame_rows, scan_pair_rows) start each of their lists from zero.
        for k in range(ROW_LISTS):
            row_totals[k, 0] = 0
    world = int(-1)
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
            acts,
        )
    # The world's range of frame positions, whose rows reserve_world_contact_rows reserves in order: one atomic pair
    # per world and warp.
    _record_world_span(world, c, world_contact_first, world_contact_last)


@wp.func
def _immovable_side(data: ContactInput, c: int) -> bool:
    """Whether contact ``c`` has a side without a body (a static shape)."""
    s0 = data.shape0[c]
    s1 = data.shape1[c]
    if s0 < 0 or s1 < 0:
        return True
    return data.shape_body[s0] < 0 or data.shape_body[s1] < 0


# A world's first contact position before it held any.
_NO_CONTACT = wp.constant(2**31 - 1)

RESERVE_BLOCK = 512
"""Threads of a block of the row order's scans (:func:`scan_frame_rows`, :func:`scan_pair_rows`) on CUDA. A tile is a
block's threads, a frame position a thread: the scans' work on a position is a chain of dependent loads (a world's run,
the pair the row order visits there, that pair's bounds), so a thread takes one. The scans compile in modules of their
own: a module compiles once per block size it launches at, and this module's other kernels launch at Warp's default."""


def reserve_tiles(positions: int, device) -> int:
    """The tile states a site of the row order's scans holds over ``positions`` frame positions: a block's threads a
    tile on CUDA; on the CPU (a block of one thread, one position a tile) the tiles run in order and share one state
    (:mod:`~newton._src.core.ordered_reserve`)."""
    if not wp.get_device(device).is_cuda:
        return 1
    return -(-max(positions, 1) // RESERVE_BLOCK)


def reserve_blocks(device) -> int:
    """The blocks of the row order's scans' fixed grid: as many as the device holds at once, so every live tile has
    its block; one on the CPU."""
    device = wp.get_device(device)
    return device.sm_count * (2048 // RESERVE_BLOCK) if device.is_cuda else 1


@wp.struct
class RowOrder:
    """The row order's records over the frame's positions (:func:`scan_frame_rows`, :func:`list_world_pairs`,
    :func:`scan_pair_rows`): each position's body-pair key, the admitted contacts' rows of each list before it in the
    frame's order and the body pairs headed before it (:func:`_heads_pair`), each pair's head and world by its rank
    (``pairs``), and each pair's rows of each list before it in the row order, at its head (``rows_before``); each
    world's first pair's rank, its pairs and the row order's stride over them (``world_pairs``)."""

    pair_keys: wp.array[wp.int64]
    world_contact_first: wp.array[int]
    world_contact_last: wp.array[int]
    world_pairs: wp.array[wp.vec3i]
    rows_frame: wp.array[wp.vec4i]
    pairs_before: wp.array[int]
    pairs: wp.array[wp.vec2i]
    rows_before: wp.array[wp.vec4i]


@wp.func
def _in_run(data: ContactInput, order: RowOrder, c: int, live: int) -> bool:
    """Whether frame position ``c`` lies in the run of positions the admission recorded for its world this step. A
    contact the admission did not reach this step (a replay whose bound launches were skipped) has no run: its stale
    records index nothing."""
    if c >= live:
        return False
    world = data.world[c]
    if world < 0 or world >= order.world_contact_first.shape[0]:
        return False
    return order.world_contact_first[world] <= c and c <= order.world_contact_last[world]


@wp.func
def _admitted(data: ContactInput, order: RowOrder, c: int, live: int) -> bool:
    """Whether contact ``c`` holds rows this step: admitted (``path`` 0 or 1) inside its world's run (:func:`_in_run`)."""
    if not _in_run(data, order, c, live):
        return False
    return data.path[c] >= 0


@wp.func
def _heads_pair(data: ContactInput, order: RowOrder, c: int, live: int) -> bool:
    """Whether frame position ``c`` heads its body pair in its world's run (:func:`_in_run`): the run's first position
    or a change of the frame's pair key (the keys ascend), whatever the contacts' admission. A pair's contacts share
    their bodies, so they respond alike: a pair is in its world's run whole or not at all."""
    if not _in_run(data, order, c, live):
        return False
    if c == order.world_contact_first[data.world[c]]:
        return True
    return order.pair_keys[c] != order.pair_keys[c - 1]


@wp.func
def _contact_rows(data: ContactInput, order: RowOrder, c: int, live: int) -> wp.vec4i:
    """Contact ``c``'s rows in the reservation's four lists: (dense, free bodies') with an immovable side
    (:func:`_immovable_side`), then (dense, free bodies') of the others; none where it holds no rows (:func:`_admitted`).
    """
    rows = wp.vec4i(0)
    if _admitted(data, order, c, live):
        k = int(0)
        if data.path[c] != 0:
            k = 1
        if not _immovable_side(data, c):
            k += 2
        rows[k] = data.slots_needed[c]
    return rows


@wp.kernel(enable_backward=False, module="unique")
def scan_frame_rows(
    data: ContactInput,
    blocks: int,
    sites: wp.array2d[wp.uint64],
    order: RowOrder,
    # outputs
    row_totals: wp.array2d[int],
):
    """Count, before every frame position, the admitted contacts' rows of the four lists (:func:`_contact_rows`;
    ``order.rows_frame``) and the body pairs headed (:func:`_heads_pair`; ``order.pairs_before``), and list each pair's
    head and world by its rank (``order.pairs``), in the frame's order: five ordered reservations at once (lists 0-4 of ``sites``
    and ``row_totals``, which the admission zeroed), a block scan and a decoupled look-back a tile
    (:mod:`~newton._src.core.ordered_reserve`), parallel inside a world of any size."""
    block, lane = wp.tid()
    tile = wp.block_dim()
    live = data.count[0]
    live_tiles = (live + tile - 1) // tile
    claim = ordered_next(sites[0], block, live_tiles, blocks, row_totals[0])
    while claim[0] >= 0:
        c = claim[0] * tile + lane
        count = _contact_rows(data, order, c, live)
        heads = int(0)
        if _heads_pair(data, order, c, live):
            heads = 1
        last = int(0)
        if claim[0] == live_tiles - 1:
            last = 1
        first = wp.vec4i(0)
        first[0] = ordered_scan(sites[0], claim, count[0], last, row_totals[0])
        also = ordered_also(claim)
        for k in range(1, 4):
            first[k] = ordered_scan(sites[k], also, count[k], last, row_totals[k])
        ranked = ordered_scan(sites[4], also, heads, last, row_totals[4])
        if c < live:
            order.rows_frame[c] = first
            order.pairs_before[c] = ranked
            if heads != 0:
                order.pairs[ranked] = wp.vec2i(c, data.world[c])
        claim = ordered_next(sites[0], block, live_tiles, blocks, row_totals[0])


@wp.func
def _row_stride(n: int) -> int:
    """The stride of a world's row order over its ``n`` body pairs: the golden ratio's share of ``n`` (``n`` times
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


@wp.kernel(enable_backward=False)
def list_world_pairs(contact_count: wp.array[int], row_totals: wp.array2d[int], order: RowOrder):
    """Record each world's body pairs (:func:`_heads_pair`): its first pair's rank, the pairs its run holds and the row
    order's stride over them (``order.world_pairs``, :func:`_row_stride`): a thread a world."""
    world = wp.tid()
    first = order.world_contact_first[world]
    last = order.world_contact_last[world]
    if last < 0:
        return
    rank = order.pairs_before[first]
    end = row_totals[4, 0]
    if last + 1 < contact_count[0]:
        end = order.pairs_before[last + 1]
    m = end - rank
    order.world_pairs[world] = wp.vec3i(rank, m, _row_stride(m))


@wp.func
def _frame_rows_at(order: RowOrder, position: int, live: int, row_totals: wp.array2d[int]) -> wp.vec4i:
    """The admitted contacts' rows of each list before frame position ``position`` (all of them at ``live``)."""
    if position >= live:
        return wp.vec4i(row_totals[0, 0], row_totals[1, 0], row_totals[2, 0], row_totals[3, 0])
    return order.rows_frame[position]


@wp.func
def _row_pair(order: RowOrder, g: int) -> int:
    """The rank of the body pair the row order visits at pair rank ``g``: in the world of the pair ranked ``g``, its
    ``m`` body pairs at a golden-ratio stride over their ranks (:func:`_row_stride`) from the first.

    A world's pairs are those its contacts with a responding endpoint form, whether admitted or not
    (:func:`_heads_pair`). Each pair's contacts stay together in the frame's order: a sorted frame lays a body's pairs
    side by side, and their rows in that order chain through shared bodies into a deep trace for the colored sweeps,
    while a pair's own contacts, which share both bodies, chain whatever their order and converge best together. It
    reads only the frame's keys and the bodies' response, so a contact's admission moves no other's rows, and nor does
    a contact that can never take rows."""
    pairs = order.world_pairs[order.pairs[g][1]]
    return pairs[0] + int((wp.int64(g - pairs[0]) * wp.int64(pairs[2])) % wp.int64(pairs[1]))


@wp.func
def _pair_end(order: RowOrder, rank: int, live: int, pairs: int) -> int:
    """One past the last frame position of the pair ranked ``rank`` among ``pairs``: the next pair's head (the
    positions between hold no rows), or ``live`` after the last."""
    if rank + 1 >= pairs:
        return live
    return order.pairs[rank + 1][0]


@wp.kernel(enable_backward=False, module="unique")
def scan_pair_rows(
    data: ContactInput,
    blocks: int,
    sites: wp.array2d[wp.uint64],
    order: RowOrder,
    # outputs
    row_totals: wp.array2d[int],
):
    """Scan the body pairs' rows of each list in the row order (:func:`_row_pair`; lists 5-8), a pair rank a thread:
    each pair's rows before it, at its head (``order.rows_before``). A pair's contacts lie from its head to its end
    (:func:`_pair_end`). A contact's rows before it in the row order are its pair's and those of the pair's earlier
    contacts (:func:`_rows_before`)."""
    block, lane = wp.tid()
    tile = wp.block_dim()
    live = data.count[0]
    pairs = row_totals[4, 0]
    live_tiles = (pairs + tile - 1) // tile
    claim = ordered_next(sites[5], block, live_tiles, blocks, row_totals[5])
    while claim[0] >= 0:
        g = claim[0] * tile + lane
        head = int(-1)
        count = wp.vec4i(0)
        if g < pairs:
            rank = _row_pair(order, g)
            head = order.pairs[rank][0]
            count = (
                _frame_rows_at(order, _pair_end(order, rank, live, pairs), live, row_totals) - order.rows_frame[head]
            )
        last = int(0)
        if claim[0] == live_tiles - 1:
            last = 1
        first = wp.vec4i(0)
        first[0] = ordered_scan(sites[5], claim, count[0], last, row_totals[5])
        also = ordered_also(claim)
        for k in range(1, 4):
            first[k] = ordered_scan(sites[5 + k], also, count[k], last, row_totals[5 + k])
        if head >= 0:
            order.rows_before[head] = first
        claim = ordered_next(sites[5], block, live_tiles, blocks, row_totals[5])


@wp.func
def _rows_before(data: ContactInput, order: RowOrder, c: int, first: int) -> wp.vec4i:
    """Admitted contact ``c``'s rows of each list before it in the row order: its body pair's (at the pair's head, the
    last pair headed at or before it; ``first`` is its world's first position) and those of the pair's earlier
    contacts, in the frame's order."""
    head = c
    if c > first:
        if order.pair_keys[c] == order.pair_keys[c - 1]:
            head = order.pairs[order.pairs_before[c] - 1][0]
    return order.rows_before[head] + order.rows_frame[c] - order.rows_frame[head]


@wp.func
def _world_rows(world: int, order: RowOrder, row_totals: wp.array2d[int]):
    """World ``world``'s rows of each list before its first pair in the row order, and its own rows of each list
    (:func:`scan_pair_rows`): ``(base, rows)``. Its pairs take consecutive ranks (:func:`.contact_sort.body_pair_codes`
    keeps a world's contacts in one run), the row order visits its first pair first, and the next world's first pair
    after its last."""
    pairs = order.world_pairs[world]
    base = order.rows_before[order.pairs[pairs[0]][0]]
    end = wp.vec4i(row_totals[5, 0], row_totals[6, 0], row_totals[7, 0], row_totals[8, 0])
    if pairs[0] + pairs[1] < row_totals[4, 0]:
        end = order.rows_before[order.pairs[pairs[0] + pairs[1]][0]]
    return base, end - base


@wp.kernel(enable_backward=False)
def reserve_world_contact_rows(
    data: ContactInput,
    max_constraints: int,
    mf_max_constraints: int,
    row_capacity_telemetry: int,
    order: RowOrder,
    world_slot_counter: wp.array[int],
    mf_slot_counter: wp.array[int],
    row_totals: wp.array2d[int],
    # outputs
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    units: wp.array2d[int],
):
    """Reserve the rows of every admitted contact (:func:`admit_world_contacts`) in the frame's order, so a world's
    rows follow its contacts' geometry (the frame's pair and contact keys), never the order the collision's threads
    wrote them in: a thread a contact.

    A world's contacts lie in one run of the frame's positions (``order.world_contact_first`` to
    ``order.world_contact_last``, which the admission recorded), whose rows of each list the scans counted before
    every contact in the row order (:func:`_rows_before`).
    A contact takes the next rows of its family in its world's line: the dense rows, or the free bodies' (``path``
    1), which follow the dense class's (one row index for both classes); the contacts with an immovable side first
    (:func:`_immovable_side`), then the others, each in the world's row order (:func:`_row_pair`). Supports first
    is the order a sweep propagates a stack's load in from the ground up. A contact's slot is its first row.

    Counters only grow (:func:`advance_world_contact_rows` adds every contact's rows): a contact whose rows do not
    fit records its slot in the family's ``*_first_rejected_slot`` (per world, preset to a large value), and every
    later contact's rows start past it, so the finalize kernels truncate the row count there; such a contact has no
    rows (``slot`` and ``path`` -1). Where the one writer takes the rows (``units`` holds the placement's unit
    sources; empty otherwise), each dense row's unit source is written with its slot.
    """
    c = wp.tid()
    live = data.count[0]
    if not _admitted(data, order, c, live):
        return
    path = data.path[c]
    world = data.world[c]
    base, rows = _world_rows(world, order, row_totals)
    before = _rows_before(data, order, c, order.world_contact_first[world]) - base
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
        if slot + n > mf_max_constraints:
            wp.atomic_min(mf_first_rejected_slot, world, slot)
            if row_capacity_telemetry != 0:
                wp.atomic_add(mf_dropped_contact_rows, world, n)
            data.slot[c] = -1
            data.path[c] = -1
        else:
            data.slot[c] = slot + max_constraints
    else:
        slot = world_slot_counter[world] + offset
        if slot + n > max_constraints:
            wp.atomic_min(dense_first_rejected_slot, world, slot)
            if row_capacity_telemetry != 0:
                wp.atomic_add(dense_dropped_contact_rows, world, n)
            data.slot[c] = -1
            data.path[c] = -1
        else:
            data.slot[c] = slot
            if units.shape[0] > 0:
                for r in range(n):
                    units[data.first[world], slot + r] = unit_source(UNIT_CONTACT, 4 * c + r)


@wp.kernel(enable_backward=False)
def advance_world_contact_rows(
    row_totals: wp.array2d[int],
    order: RowOrder,
    # outputs
    world_slot_counter: wp.array[int],
    mf_slot_counter: wp.array[int],
):
    """Add each world's contact rows (:func:`reserve_world_contact_rows`) to its counters, rejected rows included, and
    reset its run of positions for the next step: a thread a world."""
    world = wp.tid()
    if order.world_contact_last[world] < 0:
        return
    _base, rows = _world_rows(world, order, row_totals)
    world_slot_counter[world] += rows[0] + rows[2]
    mf_slot_counter[world] += rows[1] + rows[3]
    order.world_contact_first[world] = _NO_CONTACT
    order.world_contact_last[world] = -1


@wp.kernel
def snapshot_dense_phase_bound(
    world_slot_counter: wp.array[int],
    bound_index: int,
    # outputs
    dense_phase_bounds: wp.array2d[int],
):
    """Record the current dense slot watermark for one row-family boundary.

    The dense per-world slot layout is [drive][joint-limit][joint-vel-limit]
    [contact/friction]. Launched right after the joint-limit allocation
    (``bound_index`` 0) and the joint-velocity-limit allocation
    (``bound_index`` 1) so the PhysX-grasp GS phases can loop only their own
    contiguous row range instead of scanning every dense row. The raw counter
    is stored; consumers clamp against the per-world constraint count.
    """
    world = wp.tid()
    dense_phase_bounds[world, bound_index] = world_slot_counter[world]

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared monotone contact reservations with explicit geometry ownership."""

import hashlib
import linecache
from string import Template

from .kinetic_source import checked_definitions

_BODIES = r"""    # Get bodies and articulations
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    art_a = -1
    art_b = -1
    if body_a >= 0:
        art_a = body_to_articulation[body_a]
    if body_b >= 0:
        art_b = body_to_articulation[body_b]

"""


_WORLD = r"""    # Determine world (both bodies must be in same world, or one is ground)
    world = -1
    if art_a >= 0:
        world = art_to_world[art_a]
    if art_b >= 0:
        world_b = art_to_world[art_b]
        if world >= 0 and world_b != world:
            # Cross-world contact - shouldn't happen, skip
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        world = world_b

    if world < 0:
        # No articulation involved (ground-ground?)
        contact_slot[c] = -1
        contact_path[c] = -1
        return

"""


_PHI = r"""    # Compute phi (same logic as populate_world_J_for_size)
    # contact normal stored as A-to-B; negate to get B-to-A used internally
    normal = -contact_normal[c]
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        # Contact points are stored in body frame by collision detection
        point_a_world = wp.transform_point(X_wb_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        # Contact points are stored in body frame by collision detection
        point_b_world = wp.transform_point(X_wb_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal
    phi = wp.dot(normal, point_a_world - point_b_world)

"""


_RESOLVED = r"""            shape_a = contact_shape0[c]
            shape_b = contact_shape1[c]
            body_a = shape_body[shape_a] if shape_a >= 0 else -1
            body_b = shape_body[shape_b] if shape_b >= 0 else -1
            art_a = body_to_articulation[body_a] if body_a >= 0 else -1
            art_b = body_to_articulation[body_b] if body_b >= 0 else -1
            world_a = art_to_world[art_a] if art_a >= 0 else -1
            world_b = art_to_world[art_b] if art_b >= 0 else -1
            world = world_a if world_a >= 0 else world_b
            same_world = world_a < 0 or world_b < 0 or world_a == world_b
"""


_HELPER = r'''@wp.func
def ${helper_name}(
    c: int,
    total_contacts: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    propagation_articulated_contacts: int,
    propagation_same_articulation: int,
    propagation_free_free: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    propagation_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    propagation_slot_counter: wp.array[int],
    dense_contact_world_flag: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    propagation_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    propagation_first_rejected_slot: wp.array[int],
    capacity_status: wp.array[int],
${extra}):
    """Classify and allocate rows for one active contact.

    Allocates world-level constraint slots for each contact and records
    which articulations are involved. Contacts where both sides are free
    rigid bodies (or ground) are routed to the matrix-free path.

    Each contact reserves 1 slot for the normal row and, when enabled below the
    friction gap threshold, 2 adjacent slots for Coulomb friction rows.
    A positive contact gap gate excludes wider speculative contacts before any
    route reserves slots; zero disables the gate.

    """
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]

${bodies}    a_has_dofs = art_a >= 0 and articulation_response_dof_count[art_a] > 0
    b_has_dofs = art_b >= 0 and articulation_response_dof_count[art_b] > 0
    a_can_respond = (
        a_has_dofs and body_has_response_dofs[body_a] != 0 and (body_flags[body_a] & BodyFlags.KINEMATIC) == 0
    )
    b_can_respond = (
        b_has_dofs and body_has_response_dofs[body_b] != 0 and (body_flags[body_b] & BodyFlags.KINEMATIC) == 0
    )
    if not a_can_respond and not b_can_respond:
        contact_slot[c] = -1
        contact_path[c] = -1
        return

${world}${phi}    # Use the same eligibility as patch selection before reserving row storage.
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
    if phi > normal_gap_limit:
        contact_slot[c] = -1
        contact_path[c] = -1
        return

    # Classify: MF path if both sides are free rigid or ground.
    # Propagation path is an opt-in matrix-free route for any contact touching a
    # non-free articulation. It stores fixed-size body-space rows instead of
    # D-wide generalized rows.
    is_mf = 0
    is_propagation = 0
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    if has_free_rigid != 0:
        a_is_mf_compatible = not a_has_dofs or is_free_rigid[art_a] != 0
        b_is_mf_compatible = not b_has_dofs or is_free_rigid[art_b] != 0
        if a_is_mf_compatible and b_is_mf_compatible:
            is_mf = 1
    if propagation_free_free != 0 and is_mf == 1:
        # Opt-in: route free/free (and free/ground) contact rows to the
        # propagation family. The row math is identical (free bodies use the
        # plain spatial inverse inertia response seeded by
        # copy_free_rigid_propagation_body_response); this puts pile rows on
        # the colored solve path. Velocity-limit rows stay on the MF family.
        is_mf = 0
        is_propagation = 1
    if propagation_articulated_contacts != 0 and is_mf == 0 and is_propagation == 0:
        # Same-articulation two-link contacts need cross response terms between
        # the two touched links. They route to propagation rows only when the
        # cross-response path is enabled; otherwise they stay on the dense
        # generalized-row path.
        same_non_free_articulation = a_non_free and b_non_free and art_a == art_b
        if a_non_free or b_non_free:
            if propagation_same_articulation != 0 or not same_non_free_articulation:
                is_propagation = 1

    # Every normal survives; only the selected patch anchors receive tangent rows.
    slots_needed = 1
    add_friction = enable_friction != 0 and contact_friction_eligible(
        phi, a_non_free, b_non_free, contact_friction_articulation_pairs_only, contact_friction_gap_threshold
    )
    if friction_patches.enabled != 0:
        add_friction = add_friction and friction_patches.weight[c] > 0.0
    if add_friction:
        slots_needed = 3
    contact_slots_needed[c] = slots_needed

    if is_mf != 0:
        # Matrix-free path
        slot = wp.atomic_add(mf_slot_counter, world, slots_needed)
        if slot + slots_needed > mf_max_constraints:
            wp.atomic_min(mf_first_rejected_slot, world, slot)
            wp.atomic_max(capacity_status, 1, 1)
            if row_capacity_telemetry != 0:
                wp.atomic_add(mf_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 1
    elif is_propagation != 0:
        # Propagation articulated matrix-free path
        slot = wp.atomic_add(propagation_slot_counter, world, slots_needed)
        if slot + slots_needed > propagation_max_constraints:
            wp.atomic_min(propagation_first_rejected_slot, world, slot)
            wp.atomic_max(capacity_status, 2, 1)
            if row_capacity_telemetry != 0:
                wp.atomic_add(propagation_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 2
    else:
        # Dense path
        slot = wp.atomic_add(world_slot_counter, world, slots_needed)
        if slot + slots_needed > max_constraints:
            wp.atomic_min(dense_first_rejected_slot, world, slot)
            wp.atomic_max(capacity_status, 0, 1)
            if row_capacity_telemetry != 0:
                wp.atomic_add(dense_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 0
        # Dense contact rows write articulated generalized velocities during
        # split GS phases; propagation tree refresh keys off this per world.
        dense_contact_world_flag[world] = 1
'''


_KERNEL = r'''${decorator}
def ${kernel_name}(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    propagation_articulated_contacts: int,
    propagation_same_articulation: int,
    propagation_free_free: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    propagation_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    resolved_worlds: wp.array[int],
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    propagation_slot_counter: wp.array[int],
    dense_contact_world_flag: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    propagation_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    propagation_first_rejected_slot: wp.array[int],
    capacity_status: wp.array[int],
${extra}):
    """Allocate active contacts with work proportional to the materialized prefix.

    The narrow phase increments :paramref:`contact_count` before checking its
    output capacity. An overflowed count therefore does not describe a fully
    materialized prefix; clear the routing arrays and reject that frame.

    Slot counters only grow. A contact whose reservation does not fit records its
    slot in the family's ``*_first_rejected_slot`` (per world, preset to a large
    value) instead of rolling the counter back: every later reservation starts at
    or past that slot, so the finalize kernels truncate the row count there and
    no accepted contact can hold a slot at or beyond the count. Rolling back
    raced with concurrent reservations and left accepted rows past the count,
    which the friction patch links then chained into.
    """
    thread = wp.tid()
    total_contacts = contact_count[0]
    capacity = contact_shape0.shape[0]
    if ${overflow}:
        if thread == 0:
            wp.atomic_max(capacity_status, 3, 1)
        for c in range(thread, capacity, total_num_threads):
            contact_slot[c] = -1
            contact_path[c] = -1
            contact_slots_needed[c] = 0
        return

    for c in range(thread, total_contacts, total_num_threads):
        if resolved_worlds.shape[0] > 0:
${resolved}            if same_world and world >= 0 and world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
                # Refresh every active mapping each solve. The original force
                # converter publishes zero before reading any row for this path.
                contact_world[c] = world
                contact_art_a[c] = art_a
                contact_art_b[c] = art_b
                contact_slot[c] = -1
                contact_path[c] = -1
                contact_slots_needed[c] = 0
                continue
        ${helper_name}(
            c,
            total_contacts,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            body_q,
            shape_transform,
            shape_body,
            body_to_articulation,
            art_to_world,
            articulation_response_dof_count,
            body_flags,
            body_has_response_dofs,
            is_free_rigid,
            has_free_rigid,
            propagation_articulated_contacts,
            propagation_same_articulation,
            propagation_free_free,
            contact_gap_gate,
            same_articulation_contact_gap_gate,
            articulation_pair_contact_gap_gate,
            max_constraints,
            mf_max_constraints,
            propagation_max_constraints,
            enable_friction,
            contact_friction_gap_threshold,
            contact_friction_articulation_pairs_only,
            row_capacity_telemetry,
            friction_patches,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            world_slot_counter,
            contact_path,
            mf_slot_counter,
            propagation_slot_counter,
            dense_contact_world_flag,
            contact_slots_needed,
            dense_dropped_contact_rows,
            mf_dropped_contact_rows,
            propagation_dropped_contact_rows,
            dense_first_rejected_slot,
            mf_first_rejected_slot,
            propagation_first_rejected_slot,
            capacity_status,
${extra_call}        )
${stage}'''


def get_source(*, cached_geometry=False):
    """Keep routing, row admission and reservations common to both consumers."""
    checked_definitions(
        "contact_allocation_source.py",
        ("_BODIES", "_WORLD", "_PHI", "_RESOLVED", "_HELPER", "_KERNEL", "get_source", "bind"),
    )
    helper_name = "_current_contact_slot" if cached_geometry else "_allocate_world_contact_slot"
    cache = "    cache: ContactGeometry,\n" if cached_geometry else ""
    helper = Template(_HELPER).substitute(
        helper_name=helper_name,
        extra=cache,
        bodies=(
            "    body_a, body_b = cache.bodies[c, 0], cache.bodies[c, 1]\n"
            "    art_a, art_b = contact_art_a[c], contact_art_b[c]\n"
            if cached_geometry
            else _BODIES
        ),
        world="    world = contact_world[c]\n" if cached_geometry else _WORLD,
        phi="    phi = cache.values[c, 15]\n" if cached_geometry else _PHI,
    )
    kernel = Template(_KERNEL).substitute(
        decorator='@wp.kernel(module="unique", enable_backward=False)' if cached_geometry else "@wp.kernel",
        kernel_name="kinetic_contact_allocate" if cached_geometry else "allocate_world_contact_slots",
        helper_name=helper_name,
        extra=cache + ("    raw: RawRowInput,\n    settings: RowSettings,\n" if cached_geometry else ""),
        overflow="total_contacts > capacity or capacity_status[3] != 0"
        if cached_geometry
        else "total_contacts > capacity",
        resolved=(
            "            art_a, art_b = contact_art_a[c], contact_art_b[c]\n"
            "            world = contact_world[c]\n            same_world = True\n"
            if cached_geometry
            else _RESOLVED
        ),
        extra_call="            cache,\n" if cached_geometry else "",
        stage=(
            "\n        if contact_path[c] == 0 and contact_slot[c] >= 0:\n"
            "            _stage_row(c, total_contacts, cache, raw, settings)\n"
            if cached_geometry
            else ""
        ),
    )
    return helper + "\n\n" + kernel


def bind(namespace, *, cached_geometry=False):
    """Bind inspectable original/current functions without source recovery."""
    source = get_source(cached_geometry=cached_geometry)
    filename = "<contact-allocation-" + hashlib.sha256(source.encode()).hexdigest() + ">"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    scope = dict(namespace)
    exec(compile(source, filename, "exec"), scope)
    names = (
        ("_current_contact_slot", "kinetic_contact_allocate")
        if cached_geometry
        else ("_allocate_world_contact_slot", "allocate_world_contact_slots")
    )
    return scope[names[0]], scope[names[1]]

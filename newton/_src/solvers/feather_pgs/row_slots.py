# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row store's slots: each contact's admission and its rows' reservation, the contact runs that order a
world's rows under sorted contacts, and the counts with their capacity status."""

import warp as wp

from ...sim import BodyFlags
from .contact_filters import contact_normal_gap_limit
from .contact_input import ContactInput
from .contact_laws import contact_endpoints, contact_materials, contact_points, contact_responds, jacobian_entry
from .device_preludes import dual_native, lane_group, lane_scan, lane_sum, lane_sync, reserve, reserved_count
from .friction_patches import CONTACT_PATH_RESOLVED, FrictionPatches, patch_spin_rows
from .row_laws import contact_capped_beta, contact_depenetration_cap, contact_restitution_rhs, contact_rhs, separates
from .row_records import UNIT_CONTACT, unit_source

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
    two, nor none without an articulation) and a gap within the normal gap limit.
    Its normal row separates where its residual at the predictor (the body ``twists`` it publishes; none: never) does
    (:func:`.row_laws.separates`), and its friction and spin rows, bounded by a finite non-negative coefficient of
    its ring's load, keep zero with it.
    """
    body_a, body_b, art_a, art_b, world = contact_endpoints(
        data.shape0[c], data.shape1[c], data.shape_body, body_to_articulation, art_to_world
    )
    responds_a = contact_responds(art_a, body_a, data.response_dofs, body_has_response_dofs, body_flags)
    responds_b = contact_responds(art_b, body_b, data.response_dofs, body_has_response_dofs, body_flags)
    phi = float(0.0)
    admitted = (responds_a or responds_b) and world >= 0
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
            if restitution > 0.0:
                fires, bounce = contact_restitution_rhs(
                    phi, incident, 0.0, restitution, data.dt, data.restitution_threshold
                )
                if fires:
                    bias = bounce
            separating = known_a and known_b and wp.isfinite(mu) and mu >= 0.0
            separating = separating and separates(incident + bias, wp.abs(speed_a) + wp.abs(speed_b) + wp.abs(bias))
    return admitted, separating, art_a, art_b, world, phi


@wp.kernel
def allocate_world_contact_slots(
    data: ContactInput,
    total_num_threads: int,
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    resolved_worlds: wp.array[int],
    twists: wp.array[wp.spatial_vector],
    # outputs
    acts: wp.array[int],
    world_slot_counter: wp.array[int],
    mf_slot_counter: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    units: wp.array2d[int],
):
    """Classify and allocate the rows of every active contact, with work proportional to the materialized prefix.

    Where the one writer takes the rows (``units`` holds the placement's unit sources; empty otherwise), each dense
    row's unit source is written with its slot: the contact's rows are the writer's to fill.

    A contact admitted by :func:`_admit_contact` reserves 1 slot for the normal row and, when enabled below the
    friction gap threshold (and, under persistent patches, for a loaded anchor), 2 adjacent slots for the friction
    rows, and a third for its spin row where the plan lets it carry one (``data.spin``: its region has a footprint, or
    contact torsion gives every friction contact one) (:func:`.friction_patches.patch_spin_rows`). Contacts where both sides are free rigid bodies (or ground) take matrix-free rows, the others world-level
    dense rows. A contact's slot is its first row in its world's line, the free bodies' rows from the dense capacity
    ``max_constraints``; the counters count each class's own rows. A positive contact gap gate excludes wider speculative contacts before any route reserves slots;
    zero disables the gate.

    A collision pass clamps :paramref:`contact_count` to the rows its writers
    filled, every one of them, and flags an overflow in its own capacity
    status, so the count always describes a materialized prefix.

    Slot counters only grow. A contact whose reservation does not fit records its
    slot in the family's ``*_first_rejected_slot`` (per world, preset to a large
    value) instead of rolling the counter back: every later reservation starts at
    or past that slot, so the finalize kernels truncate the row count there and
    no accepted contact can hold a slot at or beyond the count. Rolling back
    raced with concurrent reservations and left accepted rows past the count,
    which the friction patch links then chained into.

    In a world the resolved-world classifier skips, no contact gets rows. A
    contact the allocator would give rows there takes CONTACT_PATH_RESOLVED and no slot, so the patch finish
    releases its anchor, as the complete path releases an anchor whose rows solve zero load; every other contact
    keeps path -1, and its anchor history. Elsewhere an admitted contact whose rows may act at the predictor marks its
    world's ``acts`` word (:data:`.placement.WORLD_ACTS`; empty: no word).
    """
    thread = wp.tid()
    # A collision pass clamps the count to the rows it wrote, all of them, and flags the overflow itself.
    total_contacts = data.count[0]

    for c in range(thread, total_contacts, total_num_threads):
        resolved = False
        if resolved_worlds.shape[0] > 0:
            _body_a, _body_b, art_a, art_b, world = contact_endpoints(
                data.shape0[c], data.shape1[c], data.shape_body, body_to_articulation, art_to_world
            )
            if world >= 0 and world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
                # Refresh every active mapping each solve. The original force
                # converter publishes zero before reading any row for this path.
                data.world[c] = world
                data.art0[c] = art_a
                data.art1[c] = art_b
                data.slot[c] = -1
                data.path[c] = -1
                data.slots_needed[c] = 0
                resolved = True
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
        if resolved:
            if admitted:
                data.path[c] = CONTACT_PATH_RESOLVED
            continue
        if not admitted:
            data.slot[c] = -1
            data.path[c] = -1
            continue
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
        add_friction = (
            enable_friction != 0 and phi <= contact_friction_gap_threshold and friction_patches.weight[c] > 0.0
        )
        if add_friction:
            slots_needed = 3 + patch_spin_rows(friction_patches, c, data.spin)
        data.slots_needed[c] = slots_needed

        # The matrix-free or the dense family's counter.
        slot = int(-1)
        if is_mf != 0:
            slot = reserve(mf_slot_counter, world, slots_needed, mf_max_constraints, mf_first_rejected_slot)
            if slot < 0 and row_capacity_telemetry != 0:
                wp.atomic_add(mf_dropped_contact_rows, world, slots_needed)
        else:
            slot = reserve(world_slot_counter, world, slots_needed, max_constraints, dense_first_rejected_slot)
            if slot < 0 and row_capacity_telemetry != 0:
                wp.atomic_add(dense_dropped_contact_rows, world, slots_needed)
        if slot < 0:
            data.slot[c] = -1
            data.path[c] = -1
            continue
        if is_mf != 0:
            # The free bodies' rows follow the dense class's in the world's line: one row index for both classes.
            slot += max_constraints
        data.world[c] = world
        data.slot[c] = slot
        data.art0[c] = art_a
        data.art1[c] = art_b
        data.path[c] = is_mf
        if is_mf == 0 and units.shape[0] > 0:
            for r in range(slots_needed):
                units[data.first[world], slot + r] = unit_source(UNIT_CONTACT, 4 * c + r)


# The allocator above reserves each contact's rows with an atomic on its world's counter, so on CUDA a
# world's rows follow the order in which warps reach that counter, which varies between runs, and an
# unconverged Gauss-Seidel solve changes with it. Under a deterministic collision pipeline
# (Contacts.rigid_contacts_pair_sorted) the contacts arrive sorted by (shape0, shape1), and the two kernels
# below give every world's rows that order instead. Shapes are numbered world by world, with the global
# shapes before and after every world's (Model.shape_world_start), so a world's contacts form one contiguous
# run per block: the run its own shapes head, and one per global shape that heads its contacts with the
# world. record_world_contact_runs notes each run's first and last contact and rows, and
# place_world_contact_runs, one warp per world, places the runs block after block from the world's first
# contact row. A family whose reservations overflowed keeps the allocator's layout.

# The first contact of a run without contacts.
_CONTACT_RUN_EMPTY = wp.constant(2**31 - 1)


@wp.func
def _contact_run_block(shape: int, global_lead: int, global_trail_start: int) -> int:
    """Block of a contact whose first shape is ``shape``: its own if the shape is global, else the worlds' block."""
    if shape < global_lead:
        return wp.max(shape, 0)
    if shape < global_trail_start:
        return global_lead
    return global_lead + 1 + shape - global_trail_start


@wp.func_native(
    dual_native(
        """
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // The lanes of one warp iteration hold ascending contacts; the lanes of one run form a group,
    // whose lowest lane records the group's first and last contact and rows.
    const unsigned group = __match_any_sync(0xffffffffu, key);
    const int leader = FIRST(group);
    const int first = READ(contact, leader);
    const int last = READ(contact, 31 - __clz(group));
#if __CUDA_ARCH__ >= 800
    const int dense_rows = __reduce_add_sync(group, dense);
    const int mf_rows = __reduce_add_sync(group, mf);
#else
    int dense_rows = 0;
    int mf_rows = 0;
    for (int source = 0; source < 32; ++source) {
        const int source_dense = READ(dense, source);
        const int source_mf = READ(mf, source);
        if (group & (1u << source)) {
            dense_rows += source_dense;
            mf_rows += source_mf;
        }
    }
#endif
    if (key >= 0 && LANE == leader) {
        atomicMin(&run_first.data[key], first);
        atomicMax(&run_last.data[key], last);
        if (dense_rows != 0)
            atomicAdd(&run_rows.data[key].c[0], dense_rows);
        if (mf_rows != 0)
            atomicAdd(&run_rows.data[key].c[1], mf_rows);
    }
#elif defined(__CUDA_ARCH__)
    if (key >= 0) {
        atomicMin(&run_first.data[key], contact);
        atomicMax(&run_last.data[key], contact);
        atomicAdd(&run_rows.data[key].c[0], dense);
        atomicAdd(&run_rows.data[key].c[1], mf);
    }
#else
    if (key >= 0) {
        if (contact < run_first.data[key])
            run_first.data[key] = contact;
        if (contact > run_last.data[key])
            run_last.data[key] = contact;
        run_rows.data[key].c[0] += dense;
        run_rows.data[key].c[1] += mf;
    }
#endif
""",
        preludes=(lane_group(32),),
    )
)
def _record_contact_run(
    key: int,
    contact: int,
    dense: int,
    mf: int,
    run_first: wp.array2d[int],
    run_last: wp.array2d[int],
    run_rows: wp.array2d[wp.vec2i],
):
    """Merge a contact into the run ``key`` (flat world and block index; -1 for none), once per run and warp on CUDA.

    Every lane of the complete warp must call it, with ascending contacts.
    """


@wp.func
def _free_body_family(
    path: int,
    free_line: int,
    c: int,
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    is_free_rigid: wp.array[int],
    response_dofs: wp.array[int],
) -> bool:
    """Whether contact ``c`` of the line (``path`` 0) is a free bodies' contact, every side a free rigid body or no
    response, which ``free_line`` places after the line's other contact rows; the allocator's classification."""
    if free_line == 0 or path != 0:
        return False
    a, b = contact_art_a[c], contact_art_b[c]
    a_free = a < 0 or response_dofs[a] <= 0 or is_free_rigid[a] != 0
    b_free = b < 0 or response_dofs[b] <= 0 or is_free_rigid[b] != 0
    return a_free and b_free


@wp.kernel(enable_backward=False)
def record_world_contact_runs(
    contact_count: wp.array[int],
    total_num_threads: int,
    lanes: int,
    contact_shape0: wp.array[int],
    contact_world: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    is_free_rigid: wp.array[int],
    response_dofs: wp.array[int],
    free_line: int,
    order_dense: int,
    order_mf: int,
    global_lead: int,
    global_trail_start: int,
    # outputs
    run_first: wp.array2d[int],
    run_last: wp.array2d[int],
    run_rows: wp.array2d[wp.vec2i],
):
    """Record each (world, block) run's first and last routed contact and its rows by family: the line's other rows,
    then the free bodies' contacts (their own run, or with ``free_line`` rows of the line placed after the others).

    ``total_num_threads`` is a multiple of ``lanes`` (32 on CUDA, 1 on CPU), so each warp
    iteration covers 32 consecutive contacts.
    """
    thread = wp.tid()
    lane = thread % lanes
    total_contacts = contact_count[0]
    blocks = run_rows.shape[1]
    for base in range(thread - lane, total_contacts, total_num_threads):
        c = base + lane
        key = int(-1)
        dense = int(0)
        mf = int(0)
        if c < total_contacts:
            path = contact_path[c]
            world = contact_world[c]
            shape = contact_shape0[c]
            needed = contact_slots_needed[c]
            free = _free_body_family(path, free_line, c, contact_art_a, contact_art_b, is_free_rigid, response_dofs)
            if path == 0 and order_dense != 0:
                if free:
                    mf = needed
                else:
                    dense = needed
            if path == 1 and order_mf != 0:
                mf = needed
            if dense > 0 or mf > 0:
                key = world * blocks + _contact_run_block(shape, global_lead, global_trail_start)
        _record_contact_run(key, c, dense, mf, run_first, run_last, run_rows)


@wp.kernel(enable_backward=False)
def place_world_contact_runs(
    contact_shape0: wp.array[int],
    contact_world: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    is_free_rigid: wp.array[int],
    response_dofs: wp.array[int],
    free_line: int,
    dense_slot_counter: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_slot_counter: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    body_base: int,
    global_lead: int,
    global_trail_start: int,
    lanes: int,
    dense_first: wp.array[int],
    # outputs
    run_first: wp.array2d[int],
    run_last: wp.array2d[int],
    run_rows: wp.array2d[wp.vec2i],
    contact_slot: wp.array[int],
    units: wp.array2d[int],
):
    """Place each world's recorded contact runs in block order and reset the records for the next step, with each
    placed dense row's unit source where ``units`` holds the one writer's (:func:`allocate_world_contact_slots`).

    Launched over ``(world_count, lanes)``: one complete warp per world on CUDA (``lanes`` 32),
    one thread on CPU (``lanes`` 1). Each family's runs start at the slot where the allocator's
    counter began, which the counter less the family's rows recovers (the free bodies' from ``body_base``); a family whose
    reservations overflowed (a first rejected slot below its counter) keeps the allocator's
    slots, all of which lie below that slot.
    """
    world, lane = wp.tid()
    blocks = run_rows.shape[1]
    total = wp.vec2i(0, 0)
    for block in range(blocks):
        total += run_rows[world, block]
    place_dense = total[0] > 0 and dense_first_rejected_slot[world] >= dense_slot_counter[world]
    place_mf = total[1] > 0 and mf_first_rejected_slot[world] >= mf_slot_counter[world]
    next_dense = dense_slot_counter[world] - total[0]
    next_mf = body_base + mf_slot_counter[world] - total[1]
    if free_line != 0:
        # The free bodies' contacts are rows of the line: they follow its other contact rows, from the one counter.
        place_dense = total[0] + total[1] > 0 and dense_first_rejected_slot[world] >= dense_slot_counter[world]
        place_mf = place_dense
        next_dense = dense_slot_counter[world] - total[0] - total[1]
        next_mf = next_dense + total[0]
    for block in range(blocks):
        last = run_last[world, block]
        if not (place_dense or place_mf):
            last = -1
        for base in range(run_first[world, block], last + 1, lanes):
            c = base + lane
            dense = int(0)
            mf = int(0)
            if c <= last:
                path = contact_path[c]
                owner = contact_world[c]
                shape = contact_shape0[c]
                needed = contact_slots_needed[c]
                if owner == world and _contact_run_block(shape, global_lead, global_trail_start) == block:
                    free = _free_body_family(
                        path, free_line, c, contact_art_a, contact_art_b, is_free_rigid, response_dofs
                    )
                    if path == 0 and place_dense and not free:
                        dense = needed
                    if (path == 1 or free) and place_mf:
                        mf = needed
            dense_below = lane_scan(dense, 32)
            mf_below = lane_scan(mf, 32)
            if dense > 0:
                contact_slot[c] = next_dense + dense_below
                if units.shape[0] > 0:
                    for r in range(dense):
                        units[dense_first[world], next_dense + dense_below + r] = unit_source(UNIT_CONTACT, 4 * c + r)
            if mf > 0:
                contact_slot[c] = next_mf + mf_below
                if free_line != 0 and units.shape[0] > 0:
                    for r in range(mf):
                        units[dense_first[world], next_mf + mf_below + r] = unit_source(UNIT_CONTACT, 4 * c + r)
            next_dense += lane_sum(dense, 32)
            next_mf += lane_sum(mf, 32)
    # Every lane has read the records before any is reset.
    lane_sync(32)
    for block in range(lane, blocks, lanes):
        run_first[world, block] = _CONTACT_RUN_EMPTY
        run_last[world, block] = -1
        run_rows[world, block] = wp.vec2i(0, 0)


@wp.kernel
def finalize_constraint_counts_with_status(
    slot_counter: wp.array[int],
    capacity: int,
    family: int,
    first_rejected_slot: wp.array[int],
    constraint_count: wp.array[int],
    capacity_status: wp.array[int],
):
    """Clamp the accepted prefix while retaining sticky overflow from every row producer."""
    world = wp.tid()
    constraint_count[world] = reserved_count(
        slot_counter, world, capacity, first_rejected_slot, capacity_status, family
    )


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

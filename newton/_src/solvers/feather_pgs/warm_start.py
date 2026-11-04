# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sweep's warm start: the impulses' initialization, and the previous step's impulses gathered by contact
identity, their tangent frame transported and projected onto the current cone."""

import warp as wp

from .contact_input import ContactRoutes
from .contact_laws import CONTACT_SPIN_ROW, contact_tangent_basis
from .factor.kernels import Clock, clock_dt_ratio
from .projection_laws import ball, disk
from .row_kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_TORSION
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


@wp.kernel
def reset_world_warmstart_buffers(
    first: wp.array[int],
    world_mask: wp.array[wp.bool],
    impulses: wp.array2d[float],
    prev_impulses: wp.array2d[float],
    prev_row_type: wp.array2d[int],
    prev_row_parent: wp.array2d[int],
):
    """Clear persistent warm-start state for selected worlds: their lines of the row store, both classes' rows."""
    world = wp.tid()
    if world_mask and not world_mask[world]:
        return
    line = first[world]
    for row in range(impulses.shape[1]):
        impulses[line, row] = 0.0
    for row in range(prev_impulses.shape[1]):
        prev_impulses[line, row] = 0.0
    for row in range(prev_row_type.shape[1]):
        prev_row_type[line, row] = -1
    for row in range(prev_row_parent.shape[1]):
        prev_row_parent[line, row] = -1


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
def _transport_warmstart_contact(
    normal: wp.vec3,
    previous_normal: wp.vec3,
    world: int,
    slot: int,
    count: int,
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    impulses: wp.array2d[float],
):
    """Transport the cached tangent frame and enforce the current contact cone: the disk, or with the contact's spin
    row (contact torsion) the elliptic cone of the pair and the spin (:func:`.projection_laws.ball`)."""
    normal_impulse = impulses[world, slot]
    if not wp.isfinite(normal_impulse) or wp.dot(normal, previous_normal) <= 0.0:
        normal_impulse = 0.0
    normal_impulse = wp.max(normal_impulse, 0.0)
    impulses[world, slot] = normal_impulse
    if slot + 2 >= count:
        return
    if (
        row_type[world, slot + 1] != PGS_CONSTRAINT_TYPE_FRICTION
        or row_type[world, slot + 2] != PGS_CONSTRAINT_TYPE_FRICTION
        or row_parent[world, slot + 1] != slot
        or row_parent[world, slot + 2] != slot
    ):
        return

    # Collision normals are A-to-B; every contact Jacobian uses B-to-A.
    old_t0, old_t1 = contact_tangent_basis(-previous_normal)
    new_t0, new_t1 = contact_tangent_basis(-normal)
    tangent_world = impulses[world, slot + 1] * old_t0 + impulses[world, slot + 2] * old_t1
    tangent = wp.vec2(wp.dot(tangent_world, new_t0), wp.dot(tangent_world, new_t1))
    magnitude = wp.length(tangent)
    radius = wp.max(row_mu[world, slot + 1] * normal_impulse, 0.0)
    spin = slot + CONTACT_SPIN_ROW
    if spin < count and row_type[world, spin] == PGS_CONSTRAINT_TYPE_TORSION and row_parent[world, spin] == slot:
        # The spin about the normal needs no frame transport: the pair and the spin project onto their cone.
        triple = wp.vec3(tangent[0], tangent[1], impulses[world, spin])
        if not wp.isfinite(magnitude) or not wp.isfinite(triple[2]) or normal_impulse <= 0.0:
            triple = wp.vec3(0.0)
        else:
            triple = ball(triple, row_mu[world, slot + 1], row_mu[world, spin], normal_impulse)
        impulses[world, slot + 1] = triple[0]
        impulses[world, slot + 2] = triple[1]
        impulses[world, spin] = triple[2]
        return
    if not wp.isfinite(magnitude) or radius <= 0.0:
        tangent = wp.vec2(0.0)
    else:
        tangent = disk(tangent, radius)
    impulses[world, slot + 1] = tangent[0]
    impulses[world, slot + 2] = tangent[1]


@wp.kernel
def gather_mf_warmstart(
    routes: ContactRoutes,
    mf: MfRows,
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    prev_slot_sorted: wp.array[int],  # prev-sorted contact idx -> prev base MF slot (or -1)
    prev_mf_impulses: wp.array2d[float],
    prev_mf_row_type: wp.array2d[int],
    prev_mf_row_parent: wp.array2d[int],
    mf_constraint_count: wp.array[int],
    previous_normal: wp.array[wp.vec3],
    decay: float,
    clock: wp.array[Clock],
    mf_max_c: int,
    # in-out
    mf_impulses: wp.array2d[float],
):
    """Seed this step's MF impulse buffer from the previous step's converged
    impulses, matched by contact identity.

    Runs after ``allocate_world_contact_slots`` + ``walk_rows.write_free_body_rows`` (so
    ``contact_slot`` and the rows' types are populated for this step) and after
    the warm-start branch has zeroed their impulses (the row store's free bodies' rows). One thread per contact;
    each writes only its own disjoint slot range, so unmatched / cold contacts
    keep the zero left by the memset and no stale slot survives.

    The step clock's ratio (:func:`.factor.kernels.clock_dt_ratio`, dt_now / dt_prev) rescales carried impulses
    across step-size changes (impulse is proportional to dt for quasi-static loads); 1.0 at fixed dt.

    ``match_index[c] >= 0`` is the previous frame's *sorted* contact index;
    ``prev_slot_sorted`` is keyed the same way (see solver writeback), so
    ``prev_slot = prev_slot_sorted[match_index[c]]`` is the base slot that
    contact occupied last step. Friction rows are only seeded when BOTH this
    step and the previous step allocated a friction row at the corresponding
    offset (guards the variable 1-vs-3 stride).
    """
    c = wp.tid()
    dt_scale = clock_dt_ratio(clock[0])
    if c >= routes.count[0]:
        return
    if routes.path[c] != 1:  # MF path only
        return
    new_slot = routes.slot[c]
    if new_slot < 0:
        return

    world = routes.world[c]
    # The free bodies' rows in the world's line: from ``mf.base``, the slots' and parents' row index.
    count = mf.base + mf_constraint_count[world]
    capacity = mf.base + mf_max_c
    if new_slot >= count:
        return
    mf_line = mf.first[world]

    mi = match_index[c]
    matched = mi >= 0
    prev_slot = int(-1)
    if matched:
        prev_slot = prev_slot_sorted[mi]

    # Normal row (offset 0): always present for an MF contact.
    if (
        matched
        and prev_slot >= 0
        and prev_slot < capacity
        and mf.row_type[mf_line, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and prev_mf_row_type[mf_line, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        mf_impulses[mf_line, new_slot] = decay * dt_scale * prev_mf_impulses[mf_line, prev_slot]
    # else: leave 0 (already memset)

    # Friction rows (offsets 1..2): only if THIS step allocated them here and
    # they belong to this contact's block (parent == base slot).
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < capacity:
            if (
                mf.row_type[mf_line, new_r] == PGS_CONSTRAINT_TYPE_FRICTION
                and mf.row_parent[mf_line, new_r] == new_slot
            ):
                prev_r = prev_slot + r
                if (
                    matched
                    and prev_slot >= 0
                    and prev_r < capacity
                    and prev_mf_row_type[mf_line, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
                    and prev_mf_row_parent[mf_line, prev_r] == prev_slot
                ):
                    mf_impulses[mf_line, new_r] = decay * dt_scale * prev_mf_impulses[mf_line, prev_r]
                # else: leave 0
    # The spin row (contact torsion): carried as the friction rows are.
    new_r = new_slot + CONTACT_SPIN_ROW
    prev_r = prev_slot + CONTACT_SPIN_ROW
    if (
        matched
        and prev_slot >= 0
        and new_r < count
        and new_r < capacity
        and prev_r < capacity
        and mf.row_type[mf_line, new_r] == PGS_CONSTRAINT_TYPE_TORSION
        and mf.row_parent[mf_line, new_r] == new_slot
        and prev_mf_row_type[mf_line, prev_r] == PGS_CONSTRAINT_TYPE_TORSION
        and prev_mf_row_parent[mf_line, prev_r] == prev_slot
    ):
        mf_impulses[mf_line, new_r] = decay * dt_scale * prev_mf_impulses[mf_line, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < capacity:
        _transport_warmstart_contact(
            routes.normal[c],
            previous_normal[mi],
            mf_line,
            new_slot,
            count,
            mf.row_type,
            mf.row_parent,
            mf.row_mu,
            mf_impulses,
        )


@wp.kernel
def snapshot_contact_warmstart(
    routes: ContactRoutes,
    previous_dense_slot: wp.array[int],
    previous_mf_slot: wp.array[int],
    previous_normal: wp.array[wp.vec3],
):
    """Save all contact routes and live normals in one launch."""
    contact = wp.tid()
    active = contact < routes.count[0]
    path = int(-1)
    slot = int(-1)
    if active:
        path = routes.path[contact]
        slot = routes.slot[contact]
        previous_normal[contact] = routes.normal[contact]
    # One thread per ready row; an empty table is a path the solver does not take.
    if previous_dense_slot.shape[0] > 0:
        previous_dense_slot[contact] = wp.where(path == 0, slot, -1)
    if previous_mf_slot.shape[0] > 0:
        previous_mf_slot[contact] = wp.where(path == 1, slot, -1)


@wp.kernel
def gather_dense_warmstart(
    routes: ContactRoutes,
    rows: DenseRows,
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    prev_slot_sorted: wp.array[int],  # prev-sorted contact idx -> prev dense normal row (or -1)
    prev_dense_impulses: wp.array2d[float],
    prev_dense_row_type: wp.array2d[int],
    prev_dense_row_parent: wp.array2d[int],
    world_constraint_count: wp.array[int],
    previous_normal: wp.array[wp.vec3],
    decay: float,
    clock: wp.array[Clock],
    max_constraints: int,
    # in-out
    world_impulses: wp.array2d[float],
):
    """Seed this step's DENSE impulse buffer from the previous step's converged
    impulses, matched by contact identity — the dense mirror of
    :func:`gather_mf_warmstart`.

    Runs after ``prepare_world_impulses`` has cold-zeroed every dense row, so
    unmatched or fresh contacts stay cold and no stale impulse survives a
    slot-layout shift. Friction rows
    (offsets 1, 2 after the normal row) are seeded only when BOTH this step and
    the previous step allocated a friction row at the corresponding offset.
    ``decay`` scales every seeded impulse: at persistent quasi-static contacts
    it damps the accumulate-on-impact overshoot while keeping the sweep's
    starting point near the converged solution.
    """
    c = wp.tid()
    dt_scale = clock_dt_ratio(clock[0])
    if c >= routes.count[0]:
        return
    if routes.path[c] != 0:  # dense path only
        return
    new_slot = routes.slot[c]
    if new_slot < 0:
        return

    world = routes.world[c]
    count = world_constraint_count[world]
    if new_slot >= count:
        return
    line = rows.first[world]

    mi = match_index[c]
    matched = mi >= 0
    prev_slot = int(-1)
    if matched:
        prev_slot = prev_slot_sorted[mi]

    # Normal row (offset 0).
    if (
        matched
        and prev_slot >= 0
        and prev_slot < max_constraints
        and prev_dense_row_type[line, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        if rows.row_type[line, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT:
            world_impulses[line, new_slot] = decay * dt_scale * prev_dense_impulses[line, prev_slot]

    # Friction rows (offsets 1, 2): only if THIS step allocated them here.
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < max_constraints:
            if rows.row_type[line, new_r] == PGS_CONSTRAINT_TYPE_FRICTION and rows.row_parent[line, new_r] == new_slot:
                prev_r = prev_slot + r
                if (
                    matched
                    and prev_slot >= 0
                    and prev_r < max_constraints
                    and prev_dense_row_type[line, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
                    and prev_dense_row_parent[line, prev_r] == prev_slot
                ):
                    world_impulses[line, new_r] = decay * dt_scale * prev_dense_impulses[line, prev_r]
    # The spin row (contact torsion): carried as the friction rows are.
    new_r = new_slot + CONTACT_SPIN_ROW
    prev_r = prev_slot + CONTACT_SPIN_ROW
    if (
        matched
        and prev_slot >= 0
        and new_r < count
        and new_r < max_constraints
        and prev_r < max_constraints
        and rows.row_type[line, new_r] == PGS_CONSTRAINT_TYPE_TORSION
        and rows.row_parent[line, new_r] == new_slot
        and prev_dense_row_type[line, prev_r] == PGS_CONSTRAINT_TYPE_TORSION
        and prev_dense_row_parent[line, prev_r] == prev_slot
    ):
        world_impulses[line, new_r] = decay * dt_scale * prev_dense_impulses[line, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < max_constraints:
        _transport_warmstart_contact(
            routes.normal[c],
            previous_normal[mi],
            line,
            new_slot,
            count,
            rows.row_type,
            rows.row_parent,
            rows.row_mu,
            world_impulses,
        )

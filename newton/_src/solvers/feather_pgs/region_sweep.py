# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free bodies' regions as the sweep's units: a closed body's sweeps in one go, a lane a body.

A free body is *closed* when every region on it is against a partner that cannot move and no dense row touches it:
nothing but its own regions reads or writes its velocity, so its configured sweeps are an independent chain and run
in one go, a lane a body (:func:`sweep_closed_bodies`), which is the exact Gauss-Seidel order. The lane holds the
wrench its regions have put on the body about its origin and the twist that wrench gives (the body's inverse
inertia, :func:`.region_laws.frame_response`'s input); a region is visited by the one visit
(:func:`.region_laws.region_visit`).

The lanes of a warp take two kinds of step together:

- *scan*: the body's regions in their order. A region that holds a wrench and whose body's twist has the bits its
  last settled visit left is passed (the pass rule, in its exact form: a stamp against a version the lane raises
  whenever a visit changes a bit of the twist). A region that holds nothing is opened only if one of its candidates
  closes at the body's twist: its frame and its candidates' records are read, nothing is written.
- *solve*: every lane that found a live region visits it now.

A lane leaves the sweeps when every region has been seen at the twist as it is: after a sweep in which no visit
changed a bit and no region slid, or on reaching again the last region whose visit changed one (both give the bits of
running every configured sweep: every later visit would be passed or idle).
"""

import numpy as np
import warp as wp

from ...core.contact_pool import ContactField
from ...geometry.flags import ShapeFlags
from .contact_input import ContactFrame, ContactInput, contact_frame
from .device_preludes import (
    dual_native,
    lane_any,
    lane_ballot,
    lane_index,
    lane_read,
    lane_scan,
    lane_stride,
    lane_sum,
    lane_sync,
)
from .factor.walk import FREE_BODY_WALK, WALK_COORDINATE, WALK_LINK, WalkTrees, walk_down_prelude
from .friction_patches import PatchRowLaw, patch_friction_location
from .projection_laws import friction_ball_metric
from .region_laws import (
    REGION_SLIDE,
    candidate_row,
    frame_response,
    frame_twist,
    frame_wrench,
    idle_reach,
    region_visit,
    stick_terms,
    twist_reach,
)
from .row_build import contact_row_law, write_contact_row
from .row_kernels import RIGID_SPEED_ROWS, RowBias, row_bias
from .row_laws import ROW_CONTACT, ROW_TORSION
from .row_records import DenseRows, row_sides

REGION_IDLE = wp.constant(-2)
"""A region's state: it holds nothing."""
REGION_LOOSE = wp.constant(-1)
"""A region's state: it holds a wrench and its last visit did not settle it (it slid). A state of zero or more is the
lane's version at the region's last settled visit."""

FRICTION_ROWS = wp.constant(3)
"""The bits of a region's kind that hold its friction unit's rows (0, 2 or 3)."""
KIND_SPRINGS = wp.constant(4)
# The lanes that share a warp's regions' candidates in the fill (fill_regions).
_FILL_LANES = wp.constant(32)
"""The bit of a region's kind that says a candidate of it is compliant."""
KIND_SECOND = wp.constant(8)
"""The bit of a region's kind that says its listed body is the pair's second (its twist enters with a minus)."""
KIND_BLOCK = wp.constant(16)
"""The bit of a region's kind that says it stands for its pair's block (:func:`form_blocks`): its visit is the pair's."""
KIND_MEMBER = wp.constant(32)
"""The bit of a region's kind that says it is visited by its pair's block, not on its own."""
BLOCK_SHIFT = wp.constant(8)
"""A block region's kind holds its block's place among its world's blocks from this bit on."""
PAIR_BLOCKS = wp.constant(2)
"""Blocks a world holds at most (its table's slots); a group past them keeps its regions."""
STAGE_UNIT = wp.constant(44)
"""Words of a block member's staged record (:func:`stage_blocks`): its unit's whitened rows (18), right-hand sides
(3), ``mu``, ``mu_t``, rows (1), the cone law's block or metric (9 + 3), its candidates' count, its friction (state,
3) and its first candidate, three spare (words 24, 38, 43)."""
STAGE_CAND = wp.constant(12)
"""Words of a block candidate's staged record: its whitened row (6), right-hand side, compliance, impulse (state),
three spare. A block's members' records come first (``STAGE_UNIT`` words each), then its candidates'."""
STAGE_WORDS = 1 << 23
"""The staged blocks' arena [words] (32 MB), allocated where a shape authors its material; a block that does not fit is
not formed and ``blk_cursor[1]`` says so."""

REGION_TOLERANCE = 1.0e-7
"""The closing threshold [m/s]: a candidate closes when its residual is below minus this."""

_BIG = wp.constant(1 << 30)

BLOCK_CLOSING = 6
"""A group of regions (a pair of movable bodies, or a body against immovable partners) whose regions closing at the
predictor outnumber this, its relative coordinates, over-determines them: with every candidate compliant it is one
block (:func:`form_blocks`)."""

_THREAD_CAP = 1 << 20
"""Lanes of the kernels that stride over the contacts or the regions, at most."""


@wp.func
def twist_differs(v: wp.vec3, w: wp.vec3, v_old: wp.vec3, w_old: wp.vec3) -> bool:
    """Whether a bit of the twist changed."""
    return (
        v[0] != v_old[0]
        or v[1] != v_old[1]
        or v[2] != v_old[2]
        or w[0] != w_old[0]
        or w[1] != w_old[1]
        or w[2] != w_old[2]
    )


@wp.func
def region_closes(sn: wp.vec3, cand: wp.array[wp.vec4], c0: int, c1: int, tol: float) -> bool:
    """Whether a candidate of ``cand[c0:c1]`` closes faster than ``tol`` at the region's velocity ``sn``, four loads
    a step."""
    closes = int(0)
    i = c0
    while i < c1:
        for k in range(4):
            record = cand[wp.min(i + k, c1 - 1)]
            if sn[0] + record[0] * sn[1] + record[1] * sn[2] + record[2] < -tol:
                closes = 1
        i += 4
    return closes == 1


@wp.func
def publish_region(
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
    c0: int,
    c1: int,
    velocity: wp.vec3,
    old: wp.vec3i,
    new: wp.vec3i,
    multipliers: wp.vec3,
    springs: bool,
    impulses: wp.array[float],
):
    """A region's candidates' impulses into their rows: the rigid set's multipliers (zero for a row that left it), and
    under ``springs`` every compliant candidate's ``max(0, -(a . velocity + b)) / eps``."""
    for k in range(3):
        if old[k] >= 0:
            impulses[cand_row[old[k]]] = 0.0
    for k in range(3):
        if new[k] >= 0:
            impulses[cand_row[new[k]]] = multipliers[k]
    if springs:
        for i in range(c0, c1):
            record = cand[i]
            if record[3] != 0.0:
                a = candidate_row(record)
                closing = wp.min(wp.dot(a, velocity) + record[2], 0.0)
                impulses[cand_row[i]] = -closing / record[3]


@wp.kernel(enable_backward=False, module="unique")
def sweep_closed_bodies(
    cursor: wp.array[int],
    lanes: wp.array[int],
    body_regions: wp.array[wp.vec2i],
    body_inertia: wp.array2d[wp.mat33],
    body_basis: wp.array2d[wp.spatial_vector],
    reg_frame: wp.array[wp.mat33],
    reg_arm: wp.array[wp.vec3],
    reg_cand: wp.array[wp.vec2i],
    reg_law: wp.array[wp.vec2],
    reg_bf: wp.array[wp.vec3],
    reg_kind: wp.array[int],
    reg_row: wp.array[int],
    reg_wnn: wp.array[wp.mat33],
    reg_wnf: wp.array[wp.mat33],
    reg_wff: wp.array[wp.mat33],
    reg_ball_values: wp.array[wp.vec3],
    reg_ball_vectors: wp.array[wp.mat33],
    reg_ginv: wp.array[wp.mat33],
    reg_k: wp.array[wp.mat33],
    reg_lk: wp.array[wp.mat33],
    reg_lw: wp.array[wp.mat33],
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
    sweeps: int,
    tol: float,
    cfm: float,
    reg_wrench: wp.array[wp.vec3],
    reg_phi: wp.array[wp.vec3],
    reg_set: wp.array[wp.vec3i],
    reg_spring: wp.array[wp.vec3],
    reg_state: wp.array[int],
    reg_reach: wp.array[wp.vec2],
    impulses: wp.array[float],
    lane_change: wp.array[wp.spatial_vector],
):
    """The closed bodies' sweeps (the module's statement): lane ``tid`` takes the free body ``lanes[tid]`` (their
    count ``cursor[2]``), whose regions are ``body_regions`` of the region stream (``reg_*``, their candidates
    ``cand``).

    A region's frame is ``reg_frame`` (rows ``n``, ``t0``, ``t1``) at ``reg_arm`` from the body's origin; its law
    ``reg_law`` (``mu``, ``mu_t``) and its friction rows' right-hand sides ``reg_bf``; its
    kind ``reg_kind`` (:data:`FRICTION_ROWS`, :data:`KIND_SPRINGS`, :data:`KIND_SECOND`); ``reg_row`` its friction
    rows' first impulse row (-1: none) and ``cand_row`` each candidate's. The lane writes the regions' state, their
    impulses, and the body's change in walk coordinates (``lane_change[tid]``, through the body's basis), which
    :func:`add_closed_changes` adds to the walk's ``du`` once the other sweeps are done."""
    tid = wp.tid()
    if tid >= cursor[2]:
        return
    b = lanes[tid]
    span = body_regions[b]
    r0 = span[0]
    r1 = span[1]
    Avv = body_inertia[b, 0]
    Avw = body_inertia[b, 1]
    Aww = body_inertia[b, 2]
    force = wp.vec3(0.0, 0.0, 0.0)
    torque = wp.vec3(0.0, 0.0, 0.0)
    v = wp.vec3(0.0, 0.0, 0.0)
    w = wp.vec3(0.0, 0.0, 0.0)
    version = int(0)
    quiet_at = int(r1)
    quiet_sweep = int(0)
    done = int(0)
    for sweep in range(sweeps):
        version_before = version
        sliding = int(0)
        # The sweep after the last visit that changed the twist ends where that visit was.
        limit = int(r1)
        if sweep == quiet_sweep + 1:
            limit = quiet_at
        g = r0
        while g < limit:
            # Scan: the first live region from g on.
            found = int(_BIG)
            while g < limit and found == _BIG:
                state = reg_state[g]
                live = int(0)
                if state != version:
                    if state >= REGION_LOOSE:
                        live = 1
                    elif not (reg_reach[g][0] * twist_reach(v, w) <= 1.0 and reg_reach[g][1] == 0.0):
                        # Within its idle reach a region that holds nothing stays open (region_laws.idle_reach).
                        frame = reg_frame[g]
                        arm = reg_arm[g]
                        at = v + wp.cross(w, arm)
                        sn = wp.vec3(wp.dot(frame[0], at), wp.dot(frame[1], w), wp.dot(frame[2], w))
                        if (reg_kind[g] & KIND_SECOND) != 0:
                            sn = -sn
                        span_c = reg_cand[g]
                        if region_closes(sn, cand, span_c[0], span_c[1], tol):
                            live = 1
                if live == 1:
                    found = g
                else:
                    g += 1
            if found == _BIG:
                if limit < r1:
                    done = 1
                break
            # Solve: every lane of the warp that found a live region is here together.
            g = found + 1
            frame = reg_frame[found]
            n = frame[0]
            t0 = frame[1]
            t1 = frame[2]
            arm = reg_arm[found]
            law = reg_law[found]
            kind = reg_kind[found]
            span_c = reg_cand[found]
            c0 = span_c[0]
            c1 = span_c[1]
            sign = float(1.0)
            if (kind & KIND_SECOND) != 0:
                sign = -1.0
            sn, sf = frame_twist(v, w, arm, n, t0, t1)
            Wnn = reg_wnn[found]
            C = reg_wnf[found]
            G = reg_wff[found]
            wrench_old = reg_wrench[found]
            phi_old = reg_phi[found]
            set_old = reg_set[found]
            wrench, phi, i0, i1, i2, multipliers, spring, velocity, outcome, _steps = region_visit(
                sn * sign,
                sf * sign,
                Wnn,
                C,
                G,
                cfm,
                reg_bf[found],
                law[0],
                law[1],
                kind & FRICTION_ROWS,
                reg_ball_values[found],
                reg_ball_vectors[found],
                reg_ginv[found],
                reg_k[found],
                reg_lk[found],
                reg_lw[found],
                cand,
                c0,
                c1,
                wrench_old,
                phi_old,
                set_old[0],
                set_old[1],
                set_old[2],
                reg_spring[found],
                tol,
            )
            push, turn = frame_wrench(wrench - wrench_old, phi - phi_old, arm, n, t0, t1)
            force += push * sign
            torque += turn * sign
            v_new = Avv * force + Avw * torque
            w_new = wp.transpose(Avw) * force + Aww * torque
            set_new = wp.vec3i(i0, i1, i2)
            publish_region(
                cand,
                cand_row,
                c0,
                c1,
                velocity,
                set_old,
                set_new,
                multipliers,
                (kind & KIND_SPRINGS) != 0,
                impulses,
            )
            row = reg_row[found]
            if row >= 0:
                impulses[row] = phi[0]
                impulses[row + 1] = phi[1]
                if (kind & FRICTION_ROWS) == 3:
                    impulses[row + 2] = phi[2]
            reg_wrench[found] = wrench
            reg_phi[found] = phi
            reg_set[found] = set_new
            reg_spring[found] = spring
            if twist_differs(v_new, w_new, v, w):
                version += 1
                quiet_at = found
                quiet_sweep = sweep
                limit = r1
            v = v_new
            w = w_new
            if outcome == REGION_SLIDE:
                reg_state[found] = REGION_LOOSE
                sliding = 1
                quiet_at = r1
                quiet_sweep = sweep
                limit = r1
            elif wrench[0] > 0.0:
                reg_state[found] = version
            else:
                reg_state[found] = REGION_IDLE
        if done == 1 or (version == version_before and sliding == 0):
            break
    # The body's change in walk coordinates: its wrench through its basis.
    change = (
        body_basis[b, 0] * force[0]
        + body_basis[b, 1] * force[1]
        + body_basis[b, 2] * force[2]
        + body_basis[b, 3] * torque[0]
        + body_basis[b, 4] * torque[1]
        + body_basis[b, 5] * torque[2]
    )
    lane_change[tid] = change


@wp.kernel(enable_backward=False, module="unique")
def add_closed_changes(
    cursor: wp.array[int],
    lanes: wp.array[int],
    body_dof: wp.array[int],
    lane_change: wp.array[wp.spatial_vector],
    du: wp.array[float],
):
    """The closed bodies' changes (:func:`sweep_closed_bodies`'s ``lane_change``) into the walk's ``du``, a thread a
    lane. The other sweeps add zero at a closed body's coordinates (a world's sweep adds its whole span), so the sum
    is the change itself in either order."""
    tid = wp.tid()
    if tid >= cursor[2]:
        return
    change = lane_change[tid]
    dof = body_dof[lanes[tid]]
    for k in range(6):
        du[dof + k] += change[k]


# ---------------------------------------------------------------------------------------------------------------------
# A free body's basis: its twist per unit of walk coordinates
# ---------------------------------------------------------------------------------------------------------------------

_LEAVE = dual_native(
    f"""
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];
    const float* data = walk.data.data + (size_t)d0 * {WALK_COORDINATE};
    const float* link = walk.link.data + (size_t)j0 * {WALK_LINK};
    float values[6] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
    values[coordinate] = 1.0f;
    WALK_DOWN(0, data, link, [&](const int c) -> float {{ return values[c]; }},
              [](const int) -> float {{ return 0.0f; }},
              [&](const int c, const float q, const float) {{ values[c] = q; }}, [](const int, const float*) {{}});
    wp::spatial_vector left;
    for (int c = 0; c < 6; ++c) left[c] = values[c];
    return left;
""",
    preludes=(walk_down_prelude((FREE_BODY_WALK,), ahead=0),),
)


@wp.func_native(_LEAVE)
def free_body_leave(walk: WalkTrees, art: int, coordinate: int) -> wp.spatial_vector:
    """The leave of the free body ``art`` (a one-body tree, :data:`.factor.walk.FREE_BODY_WALK`) of the unit walk
    coordinate ``coordinate``: its six joint velocities, a column of ``P^T L^-T``."""
    ...


@wp.kernel(enable_backward=False, module="unique")
def free_body_basis(
    free_art: wp.array[int],
    walk: WalkTrees,
    motion: wp.array[wp.spatial_vector],
    dof_start: wp.array[int],
    v_hat: wp.array[float],
    body_basis: wp.array2d[wp.spatial_vector],
    body_inertia: wp.array2d[wp.mat33],
    body_twist: wp.array[wp.spatial_vector],
):
    """Each free body's basis: the twist about its origin (linear part first) a unit of each of its six walk
    coordinates gives, ``T = S P^T L^-T`` (its motion columns ``S`` at the step's poses, the held factor's leave),
    stored by twist component (``body_basis[f, j] . u`` is component ``j``); its inverse inertia about its origin
    ``T T^T`` as three blocks (``vv``, ``vw``, ``ww``); and its predicted twist ``S v_hat``.

    A wrench about the body's origin enters the walk's coordinates through the same basis (``u = T^T wrench``), which
    is the whitened row ``L^-1 P J^T`` of that wrench: a region's rows need no walk of their own."""
    f = wp.tid()
    art = free_art[f]
    first = dof_start[art]
    T = wp.spatial_matrix()
    predicted = wp.spatial_vector()
    for m in range(6):
        left = free_body_leave(walk, art, m)
        column = wp.spatial_vector()
        for k in range(6):
            column += motion[first + k] * left[k]
        for j in range(6):
            T[j, m] = column[j]
        predicted += motion[first + m] * v_hat[first + m]
    for j in range(6):
        body_basis[f, j] = wp.spatial_vector(T[j, 0], T[j, 1], T[j, 2], T[j, 3], T[j, 4], T[j, 5])
    A = T * wp.transpose(T)
    body_inertia[f, 0] = wp.mat33(A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2], A[2, 0], A[2, 1], A[2, 2])
    body_inertia[f, 1] = wp.mat33(A[0, 3], A[0, 4], A[0, 5], A[1, 3], A[1, 4], A[1, 5], A[2, 3], A[2, 4], A[2, 5])
    body_inertia[f, 2] = wp.mat33(A[3, 3], A[3, 4], A[3, 5], A[4, 3], A[4, 4], A[4, 5], A[5, 3], A[5, 4], A[5, 5])
    body_twist[f] = predicted


# ---------------------------------------------------------------------------------------------------------------------
# The regions of a step: each world's rings listed as the sweeps read them
# ---------------------------------------------------------------------------------------------------------------------


@wp.func
def ring_head(parent: wp.array2d[int], line: int, row: int, limit: int) -> int:
    """The size of the ring of normal rows that ``row`` heads (it is the ring's lowest row), 0 where it is not the
    head."""
    size = int(1)
    j = parent[line, row]
    while j >= 0 and j != row and size < limit:
        if j < row:
            return 0
        size += 1
        j = parent[line, j]
    return size


@wp.kernel(enable_backward=False, module="unique")
def begin_listing(
    held: wp.array[int],
    relist: wp.array[int],
    body_seen: wp.array[wp.spatial_vector],
    body_version: wp.array[int],
    body_touch: wp.array[int],
):
    """Each free body at rest in the sweeps' coordinates (its version zero) and untouched, before a step's regions are
    listed; and whether the step lists them anew (``relist[0]``): where the patch builder made the contact frame's
    regions anew (``held``, the regions its build carried: zero) or the last listing did not fit (``relist[1]``).
    The checks (:func:`check_contacts`, :func:`check_bodies`) may still ask for a new listing."""
    f = wp.tid()
    if f == 0:
        relist[0] = int(held[0] == 0 or relist[1] == 0)
    if f < body_seen.shape[0]:
        body_seen[f] = wp.spatial_vector()
        body_version[f] = 0
        body_touch[f] = 0


@wp.kernel(enable_backward=False, module="unique")
def check_contacts(
    data: ContactInput,
    threads: int,
    art_free: wp.array[int],
    listed: wp.array[int],
    body_touch: wp.array[int],
    relist: wp.array[int],
):
    """Each contact against the listing it holds, ``threads`` lanes striding over them: a dense contact touches the
    free bodies it has (``body_touch``); a contact the free bodies' rows hold, its first side a free body, is a
    candidate of the frame's regions. A candidate the listing does not hold, or a listed one the step does not admit,
    asks for a new listing (``relist[0]``)."""
    for c in range(wp.tid(), data.count[0], threads):
        candidate = int(0)
        slot = data.slot[c]
        if slot >= 0:
            if data.path[c] == 0:
                for side in range(2):
                    art = data.art0[c]
                    if side == 1:
                        art = data.art1[c]
                    if art >= 0:
                        f = art_free[art]
                        if f >= 0:
                            body_touch[f] = 1
            elif data.path[c] == 1:
                if row_sides(data.art0[c], data.art1[c], data.response_dofs)[0] >= 0:
                    candidate = 1
        if candidate != listed[c]:
            relist[0] = 1


@wp.kernel(enable_backward=False, module="unique")
def check_bodies(
    speed_slot: wp.array[int],
    speed_rows: int,
    body_pair: wp.array[int],
    body_open: wp.array[int],
    body_touch: wp.array[int],
    relist: wp.array[int],
):
    """Each free body's marks: a dense row (:func:`check_contacts`) or a speed row (``speed_slot``, ``speed_rows`` a
    body) touches it (``body_touch``); with a region between two movable bodies on it (``body_pair``, the listing's)
    it is open. A body whose mark is not the listing's asks for a new listing (``relist[0]``)."""
    f = wp.tid()
    touch = body_touch[f]
    for row in range(speed_rows):
        if speed_slot[speed_rows * f + row] >= 0:
            touch = 1
    body_touch[f] = touch
    if int(touch != 0 or body_pair[f] != 0) != body_open[f]:
        relist[0] = 1


@wp.kernel(enable_backward=False, module="unique")
def clear_layout(
    relist: wp.array[int],
    cursor: wp.array[int],
    blk_cursor: wp.array[int],
    body_count: wp.array2d[int],
    body_regions: wp.array[wp.vec2i],
    body_pair: wp.array[int],
):
    """Where the step lists its regions anew: the stream empty and each free body without a region."""
    if relist[0] == 0:
        return
    f = wp.tid()
    if f < 4:
        cursor[f] = 0
    if f < 2:
        blk_cursor[f] = 0
    if f < body_pair.shape[0]:
        for k in range(3):
            body_count[f, k] = 0
        body_regions[f] = wp.vec2i(0, 0)
        body_pair[f] = 0


@wp.kernel(enable_backward=False, module="unique")
def mark_regions(
    relist: wp.array[int],
    data: ContactInput,
    rows: DenseRows,
    mf_max: int,
    threads: int,
    art_free: wp.array[int],
    head_size: wp.array[int],
    listed: wp.array[int],
    body_pair: wp.array[int],
    body_count: wp.array2d[int],
):
    """Where the step lists its regions anew, each contact with free rows, ``threads`` lanes striding over them: a
    contact between two movable bodies marks both (``body_pair``); a contact whose normal row heads its ring (the
    ring's lowest row, :func:`ring_head`) is a region's head: it takes the ring's size (``head_size``, 0 on every
    other contact) and counts a region and its candidates on its first body (``body_count``). No contact is listed
    yet (``listed``: :func:`order_regions` lists the candidates)."""
    if relist[0] == 0:
        return
    for c in range(wp.tid(), data.count[0], threads):
        head_size[c] = 0
        listed[c] = 0
        slot = data.slot[c]
        if slot < 0 or data.path[c] != 1:
            continue
        sides = row_sides(data.art0[c], data.art1[c], data.response_dofs)
        if sides[0] < 0:
            continue
        f = art_free[sides[0]]
        if sides[1] >= 0:
            body_pair[f] = 1
            body_pair[art_free[sides[1]]] = 1
        size = ring_head(rows.row_parent, rows.first[data.world[c]], slot, mf_max)
        if size > 0:
            head_size[c] = size
            wp.atomic_add(body_count, f, 0, 1)
            wp.atomic_add(body_count, f, 1, size)


@wp.kernel(enable_backward=False, module="unique")
def place_bodies(
    relist: wp.array[int],
    world_free_start: wp.array[int],
    world_free: wp.array[int],
    body_touch: wp.array[int],
    body_pair: wp.array[int],
    capacity: wp.array[int],
    body_count: wp.array2d[int],
    cursor: wp.array[int],
    body_open: wp.array[int],
    body_regions: wp.array[wp.vec2i],
    body_cand: wp.array[int],
    lanes: wp.array[int],
    world_regions: wp.array[wp.vec2i],
):
    """Where the step lists its regions anew, a world's bodies placed in the regions' stream, a thread a world (it
    reads its free bodies' counts only).

    A free body is *open* when a dense row or a speed row (``body_touch``, :func:`check_bodies`) or a region between
    two movable bodies (``body_pair``, :func:`mark_regions`) touches it, else *closed*. The stream holds the world's
    closed bodies' regions body by body, each such body a lane of ``lanes`` (counted in ``cursor[2]``), then its open
    bodies' (``world_regions``): each body takes its regions' places (``body_regions``) and its candidates' first
    (``body_cand``). The places are taken from ``cursor`` as worlds arrive: they decide no result. A world whose
    regions do not fit the stream (``capacity``: the contact pool's ready rows) lists none and sets ``cursor[3]``."""
    if relist[0] == 0:
        return
    world = wp.tid()
    world_regions[world] = wp.vec2i(0, 0)
    f_begin = world_free_start[world]
    f_end = world_free_start[world + 1]
    regions = int(0)
    candidates = int(0)
    for k in range(f_begin, f_end):
        f = world_free[k]
        body_open[f] = int(body_touch[f] != 0 or body_pair[f] != 0)
        regions += body_count[f, 0]
        candidates += body_count[f, 1]
    if regions == 0:
        return
    region_at = wp.atomic_add(cursor, 0, regions)
    candidate_at = wp.atomic_add(cursor, 1, candidates)
    if region_at + regions > capacity[0] or candidate_at + candidates > capacity[0]:
        cursor[3] = 1
        return
    for k in range(f_begin, f_end):
        f = world_free[k]
        count = body_count[f, 0]
        if body_open[f] == 0 and count > 0:
            body_regions[f] = wp.vec2i(region_at, region_at + count)
            body_cand[f] = candidate_at
            lanes[wp.atomic_add(cursor, 2, 1)] = f
            region_at += count
            candidate_at += body_count[f, 1]
    first = region_at
    for k in range(f_begin, f_end):
        f = world_free[k]
        count = body_count[f, 0]
        if body_open[f] != 0 and count > 0:
            body_regions[f] = wp.vec2i(region_at, region_at + count)
            body_cand[f] = candidate_at
            region_at += count
            candidate_at += body_count[f, 1]
    world_regions[world] = wp.vec2i(first, region_at)


@wp.kernel(enable_backward=False, module="unique")
def place_regions(
    relist: wp.array[int],
    data: ContactInput,
    rows: DenseRows,
    pitch: int,
    threads: int,
    art_free: wp.array[int],
    head_size: wp.array[int],
    body_regions: wp.array[wp.vec2i],
    body_count: wp.array2d[int],
    reg_place: wp.array[wp.vec4i],
):
    """Each region's head into a place of its body's (:func:`place_bodies`), in the order its thread arrives
    (:func:`order_regions` orders them): its row in the store, its size, its body and its contact (``reg_place``),
    where the step lists its regions anew."""
    if relist[0] == 0:
        return
    for c in range(wp.tid(), data.count[0], threads):
        size = head_size[c]
        if size == 0:
            continue
        f = art_free[row_sides(data.art0[c], data.art1[c], data.response_dofs)[0]]
        span = body_regions[f]
        if span[1] == span[0]:
            continue
        g = span[0] + wp.atomic_add(body_count, f, 2, 1)
        reg_place[g] = wp.vec4i(rows.first[data.world[c]] * pitch + data.slot[c], size, f, c)


@wp.kernel(enable_backward=False, module="unique")
def order_regions(
    relist: wp.array[int],
    cursor: wp.array[int],
    threads: int,
    data: ContactInput,
    law: PatchRowLaw,
    body_world: wp.array[int],
    body_regions: wp.array[wp.vec2i],
    body_cand: wp.array[int],
    reg_place: wp.array[wp.vec4i],
    reg_head: wp.array[wp.vec2i],
    reg_cand: wp.array[wp.vec2i],
    reg_kind: wp.array[int],
    cand_contact: wp.array[int],
    listed: wp.array[int],
):
    """Where the step lists its regions anew, a body's regions in their order (the placed regions, ``threads`` lanes
    striding over them): a region's place is its body's first plus the count of its body's regions whose heads' rows
    come before its own (the order of the world's rows), and its candidates' places follow those regions'
    candidates. It takes its world (``reg_head``), its candidates' places (``reg_cand``), no kind yet, and its
    candidates: the members of its patch the step admits on the free bodies' rows, in the patch's order from its
    head (the ring the patch link made of their rows), each ``listed`` (``cand_contact``, which the frame's later
    steps keep: :func:`fill_regions` finds their rows)."""
    if relist[0] == 0 or cursor[3] != 0:
        return
    for g in range(wp.tid(), cursor[0], threads):
        own = reg_place[g]
        f = own[2]
        span = body_regions[f]
        rank = int(0)
        at = body_cand[f]
        for h in range(span[0], span[1]):
            other = reg_place[h]
            if other[0] < own[0]:
                rank += 1
                at += other[1]
        place = span[0] + rank
        reg_head[place] = wp.vec2i(own[0], body_world[f])
        reg_cand[place] = wp.vec2i(at, at + own[1])
        reg_kind[place] = 0
        head = own[3]
        c = head
        for k in range(own[1]):
            cand_contact[at + k] = c
            listed[c] = 1
            # The patch's next member on the head's rows (the link's rule, :func:`.friction_patches._link_contact`).
            c = law.patches.next_contact[c]
            while c >= 0 and c != head:
                if data.slot[c] >= 0 and data.path[c] == data.path[head] and data.world[c] == data.world[head]:
                    break
                c = law.patches.next_contact[c]


@wp.func
def region_terms(
    body_inertia: wp.array2d[wp.mat33],
    f0: int,
    f1: int,
    arm: wp.vec3,
    arm_b: wp.vec3,
    n: wp.vec3,
    t0: wp.vec3,
    t1: wp.vec3,
    mu: float,
    mu_t: float,
    friction: int,
    cfm: float,
):
    """A region's terms that stand for the step, as its bodies' bases do, formed once a step for every visit of both
    sweeps: the pair's response in the region's coordinates (:func:`.region_laws.frame_response`, summed over the
    bodies that move), for a unit with a spin row the cone law's metric of its regularized friction block
    (:func:`.projection_laws.friction_ball_metric`), and the stick terms (:func:`.region_laws.stick_terms`: the
    block's inverse, the stick metric and the two metrics' factors). ``(Wnn, C, G, values, vectors, Ginv, K, Lk,
    Lw)``; each is the arithmetic a visit made of the same inputs, so the visits' results keep their bits."""
    Wnn, C, G = frame_response(body_inertia[f0, 0], body_inertia[f0, 1], body_inertia[f0, 2], arm, n, t0, t1)
    if f1 >= 0:
        Wnn_b, C_b, G_b = frame_response(
            body_inertia[f1, 0], body_inertia[f1, 1], body_inertia[f1, 2], arm_b, n, t0, t1
        )
        Wnn = Wnn + Wnn_b
        C = C + C_b
        G = G + G_b
    values = wp.vec3(0.0, 0.0, 0.0)
    vectors = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if mu_t > 0.0:
        values, vectors = friction_ball_metric(G + wp.mat33(cfm, 0.0, 0.0, 0.0, cfm, 0.0, 0.0, 0.0, cfm), mu, mu_t)
    Ginv, K, Lk, Lw = stick_terms(Wnn, C, G, cfm, friction)
    return Wnn, C, G, values, vectors, Ginv, K, Lk, Lw


@wp.func
def member_laws(rows: DenseRows, bias: RowBias, data: ContactInput, law: PatchRowLaw, frame: ContactFrame, c: int):
    """A region member's rows, contact ``c`` at its front end ``frame``, as the free bodies' rows hold them: each row's
    kind (its normal row's parent is the patch link's ring), its records by the contact law
    (:func:`.row_build.write_contact_row`), its friction and spin rows' rhs by the row law
    (:func:`.row_kernels.row_bias`, no incident velocity: the region's own slips enter where the fill reads them)."""
    slot = data.slot[c]
    line = rows.first[data.world[c]]
    count = int(1)
    if data.slots_needed[c] >= 3:
        count = data.slots_needed[c]
    rows.row_type[line, slot] = ROW_CONTACT
    contact_row_law(data, law, frame, c, 0, False)
    for row in range(1, count):
        write_contact_row(data, law, frame, c, row, False)
        row_bias(rows, bias, line, slot + row, 0.0)


@wp.func
def fill_region_head(
    g: int,
    rows: DenseRows,
    bias: RowBias,
    data: ContactInput,
    law: PatchRowLaw,
    mf_constraint_count: wp.array[int],
    mf_contact_rows_end: wp.array[int],
    dense_max: int,
    mf_max: int,
    pitch: int,
    cand_contact: wp.array[int],
    art_free: wp.array[int],
    body_twist: wp.array[wp.spatial_vector],
    body_inertia: wp.array2d[wp.mat33],
    cfm: float,
    reg_head: wp.array[wp.vec2i],
    reg_cand: wp.array[wp.vec2i],
    reg_frame: wp.array[wp.mat33],
    reg_arm: wp.array[wp.vec3],
    reg_arm_b: wp.array[wp.vec3],
    reg_body: wp.array[wp.vec2i],
    reg_law: wp.array[wp.vec2],
    reg_bf: wp.array[wp.vec3],
    reg_kind: wp.array[int],
    reg_row: wp.array[int],
    reg_wnn: wp.array[wp.mat33],
    reg_wnf: wp.array[wp.mat33],
    reg_wff: wp.array[wp.mat33],
    reg_ball_values: wp.array[wp.vec3],
    reg_ball_vectors: wp.array[wp.mat33],
    reg_ginv: wp.array[wp.mat33],
    reg_k: wp.array[wp.mat33],
    reg_lk: wp.array[wp.mat33],
    reg_lw: wp.array[wp.mat33],
    reg_wrench: wp.array[wp.vec3],
    reg_phi: wp.array[wp.vec3],
    reg_set: wp.array[wp.vec3i],
    reg_spring: wp.array[wp.vec3],
    reg_state: wp.array[int],
):
    """Region ``g`` formed from its world, its candidates' contacts and their places, all but its candidates
    (:func:`fill_region_member`) and its kind, which it returns with what its candidates read: ``(kind, c0, size, line,
    source, c, sn, point)``, the first candidate, the candidates, the world's row line, the slot that holds its frame's
    rows, the contact its frame is, the predictor's normal and friction slips at its friction location, and its
    frame's point."""
    world = reg_head[g][1]
    line = rows.first[world]
    span = reg_cand[g]
    c0 = span[0]
    size = span[1] - span[0]
    end = dense_max + wp.min(mf_contact_rows_end[world], wp.min(mf_constraint_count[world], mf_max))
    head = cand_contact[c0]
    r = data.slot[head]
    sides = row_sides(data.art0[head], data.art1[head], data.response_dofs)
    f0 = art_free[sides[0]]
    f1 = int(-1)
    if sides[1] >= 0:
        f1 = art_free[sides[1]]
    # The ring's carrier: the last member the allocation gave friction rows, at the rows the step gave the members.
    carrier = int(-1)
    c = head
    for k in range(size):
        cj = cand_contact[c0 + k]
        j = data.slot[cj]
        if j + 1 < end and data.slots_needed[cj] >= 3:
            carrier = j
            c = cj
    source = r
    if carrier >= 0:
        source = carrier
    frame = contact_frame(data, law, c)
    member_laws(rows, bias, data, law, frame, c)
    n = frame.normal
    t0 = frame.tangent0
    t1 = frame.tangent1
    arm = frame.friction_point - data.origin[sides[0]]
    arm_b = wp.vec3(0.0, 0.0, 0.0)
    # The pair's first body (the rows' positive side) is the region's first side, or its second.
    sign = float(1.0)
    if data.art0[c] != sides[0]:
        sign = -1.0
    twist = body_twist[f0]
    sn, sf = frame_twist(wp.spatial_top(twist), wp.spatial_bottom(twist), arm, n, t0, t1)
    sn = sn * sign
    sf = sf * sign
    if f1 >= 0:
        arm_b = frame.friction_point - data.origin[sides[1]]
        twist_b = body_twist[f1]
        sn_b, sf_b = frame_twist(wp.spatial_top(twist_b), wp.spatial_bottom(twist_b), arm_b, n, t0, t1)
        sn = sn - sn_b * sign
        sf = sf - sf_b * sign
    kind = int(0)
    mu = float(0.0)
    mu_t = float(0.0)
    bf = wp.vec3(0.0, 0.0, 0.0)
    friction_row = int(-1)
    if carrier >= 0:
        mu = rows.row_mu[line, carrier + 1]
        bf = wp.vec3(rows.rhs[line, carrier + 1] + sf[0], rows.rhs[line, carrier + 2] + sf[1], 0.0)
        unit_rows = int(2)
        if carrier + 3 < end:
            if rows.row_type[line, carrier + 3] == ROW_TORSION and rows.row_parent[line, carrier + 3] == carrier:
                unit_rows = 3
                mu_t = rows.row_mu[line, carrier + 3]
                bf = wp.vec3(bf[0], bf[1], rows.rhs[line, carrier + 3] + sf[2])
        if mu > 0.0:
            kind = unit_rows
            friction_row = line * pitch + carrier + 1
    if sign < 0.0:
        kind = kind | KIND_SECOND
    # A block's marks are the frame's (:func:`form_blocks`).
    kind = kind | (reg_kind[g] & ~(FRICTION_ROWS | KIND_SPRINGS | KIND_SECOND))
    Wnn, C, G, values, vectors, Ginv, K, Lk, Lw = region_terms(
        body_inertia, f0, f1, arm, arm_b, n, t0, t1, mu, mu_t, kind & FRICTION_ROWS, cfm
    )
    reg_frame[g] = wp.matrix_from_rows(n, t0, t1)
    reg_arm[g] = arm
    reg_arm_b[g] = arm_b
    reg_body[g] = wp.vec2i(f0, f1)
    reg_law[g] = wp.vec2(mu, mu_t)
    reg_bf[g] = bf
    reg_row[g] = friction_row
    reg_ginv[g] = Ginv
    reg_k[g] = K
    reg_lk[g] = Lk
    reg_lw[g] = Lw
    reg_wnn[g] = Wnn
    reg_wnf[g] = C
    reg_wff[g] = G
    reg_ball_values[g] = values
    reg_ball_vectors[g] = vectors
    reg_wrench[g] = wp.vec3(0.0, 0.0, 0.0)
    reg_phi[g] = wp.vec3(0.0, 0.0, 0.0)
    reg_set[g] = wp.vec3i(-1, -1, -1)
    reg_spring[g] = wp.vec3(0.0, 0.0, 0.0)
    reg_state[g] = REGION_IDLE
    return kind, c0, size, line, source, c, sn, frame.point


@wp.func
def fill_region_member(
    g: int,
    k: int,
    c0: int,
    line: int,
    source: int,
    c: int,
    sn: wp.vec3,
    point: wp.vec3,
    rows: DenseRows,
    bias: RowBias,
    data: ContactInput,
    law: PatchRowLaw,
    pitch: int,
    cand_contact: wp.array[int],
    reg_frame: wp.array[wp.mat33],
    reg_wnn: wp.array[wp.mat33],
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
) -> bool:
    """Candidate ``k`` of region ``g`` (:func:`fill_region_head`'s returns): a point on the region's frame (its normal
    and tangents ``reg_frame[g]``, its friction location contact ``c``'s), its rows' laws where its front end is formed
    unless its rows are the frame's own (``source``), its rhs at the predictor, and its compliance [1/kg], the row's
    (a material's) or its relative one's share of the candidate's own response a . Wnn a (``reg_wnn[g]``). Returns
    whether it is compliant."""
    basis = reg_frame[g]
    n = wp.vec3(basis[0, 0], basis[0, 1], basis[0, 2])
    t0 = wp.vec3(basis[1, 0], basis[1, 1], basis[1, 2])
    t1 = wp.vec3(basis[2, 0], basis[2, 1], basis[2, 2])
    friction_point = patch_friction_location(law.patches, c)
    cj = cand_contact[c0 + k]
    j = data.slot[cj]
    member_point = point
    if j != source:
        member = contact_frame(data, law, cj)
        member_laws(rows, bias, data, law, member, cj)
        member_point = member.point
    lever = wp.cross(member_point - friction_point, n)
    a1 = wp.dot(lever, t0)
    a2 = wp.dot(lever, t1)
    incident = sn[0] + a1 * sn[1] + a2 * sn[2]
    row_bias(rows, bias, line, j, incident)
    eps = float(0.0)
    if bias.material != 0:
        eps = rows.row_eps[line, j]
    if eps == 0.0 and bias.weighted != 0:
        weight = rows.row_w[line, j]
        if weight < 1.0:
            eps = -(1.0 / weight - 1.0)
    if eps < 0.0:
        row = wp.vec3(1.0, a1, a2)
        eps = -eps * wp.dot(row, reg_wnn[g] * row)
    cand[c0 + k] = wp.vec4(a1, a2, rows.rhs[line, j] + incident, eps)
    cand_row[c0 + k] = line * pitch + j
    return eps != 0.0


@wp.kernel(enable_backward=False, module="unique")
def fill_regions(
    rows: DenseRows,
    bias: RowBias,
    data: ContactInput,
    law: PatchRowLaw,
    mf_constraint_count: wp.array[int],
    mf_contact_rows_end: wp.array[int],
    dense_max: int,
    mf_max: int,
    pitch: int,
    cand_contact: wp.array[int],
    art_free: wp.array[int],
    body_twist: wp.array[wp.spatial_vector],
    body_inertia: wp.array2d[wp.mat33],
    cfm: float,
    cursor: wp.array[int],
    threads: int,
    reg_head: wp.array[wp.vec2i],
    reg_cand: wp.array[wp.vec2i],
    reg_frame: wp.array[wp.mat33],
    reg_arm: wp.array[wp.vec3],
    reg_arm_b: wp.array[wp.vec3],
    reg_body: wp.array[wp.vec2i],
    reg_law: wp.array[wp.vec2],
    reg_bf: wp.array[wp.vec3],
    reg_kind: wp.array[int],
    reg_row: wp.array[int],
    reg_wnn: wp.array[wp.mat33],
    reg_wnf: wp.array[wp.mat33],
    reg_wff: wp.array[wp.mat33],
    reg_ball_values: wp.array[wp.vec3],
    reg_ball_vectors: wp.array[wp.mat33],
    reg_ginv: wp.array[wp.mat33],
    reg_k: wp.array[wp.mat33],
    reg_lk: wp.array[wp.mat33],
    reg_lw: wp.array[wp.mat33],
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
    reg_wrench: wp.array[wp.vec3],
    reg_phi: wp.array[wp.vec3],
    reg_set: wp.array[wp.vec3i],
    reg_spring: wp.array[wp.vec3],
    reg_state: wp.array[int],
    reg_reach: wp.array[wp.vec2],
):
    """The placed regions formed (:func:`order_regions`; the first ``cursor[0]`` of the stream), ``threads`` lanes
    striding over them.

    A region's frame is its carrier's (its head's without one): the contact's normal, the tangent pair every
    friction row uses for it, at the region's friction location. Its candidates are the ring's contacts, each a
    point on that frame with the row law's right-hand side at the predictor (:func:`.row_kernels.row_bias`, the
    incident velocity the frame's); its friction rows' right-hand sides are the row law's with the predictor's
    slips and spin. Each member's rows take their laws where its front end is formed (:func:`member_laws`), the
    carrier's first, so the region reads them: the free bodies' writer leaves the regions' rows their kinds alone. It
    starts holding nothing, with the coefficients up to which its bodies' twists leave it open
    (:func:`.region_laws.idle_reach`), formed once its candidates are.

    A region's candidates are the work a region has, 1 to tens of them, and a warp's 32 regions would hold its lanes
    to its largest: each lane forms its region's frame, terms and records (:func:`fill_region_head`), then the warp
    deals its regions' candidates over its lanes in order, a candidate a lane (:func:`fill_region_member`), each with
    its region's values from the region's lane, and the region's lane takes back whether one is compliant."""
    if cursor[3] != 0:
        return
    count = cursor[0]
    lane = lane_index(_FILL_LANES)
    stride = lane_stride(_FILL_LANES)
    g = wp.tid()
    while lane_any(g < count, _FILL_LANES):
        holds = g < count
        kind = int(0)
        c0 = int(0)
        size = int(0)
        line = int(0)
        source = int(0)
        c = int(0)
        sn = wp.vec3(0.0)
        point = wp.vec3(0.0)
        if holds:
            kind, c0, size, line, source, c, sn, point = fill_region_head(
                g, rows, bias, data, law, mf_constraint_count, mf_contact_rows_end, dense_max, mf_max, pitch,
                cand_contact, art_free, body_twist, body_inertia, cfm, reg_head, reg_cand, reg_frame, reg_arm, reg_arm_b,
                reg_body, reg_law, reg_bf, reg_kind, reg_row, reg_wnn, reg_wnf, reg_wff, reg_ball_values,
                reg_ball_vectors, reg_ginv, reg_k, reg_lk, reg_lw, reg_wrench, reg_phi, reg_set, reg_spring, reg_state,
            )  # fmt: skip
        # The candidates of the warp's regions dealt over its lanes in order, a candidate a lane: a warp runs as many
        # rounds as its regions hold candidates over its lanes, where a region's lane alone ran its largest region's.
        first = lane_scan(size, _FILL_LANES)
        total = lane_sum(size, _FILL_LANES)
        lane_sync(_FILL_LANES)
        for dealt in range(0, total, stride):
            q = dealt + lane
            # The lane whose region holds candidate q: the last whose first candidate is at most q.
            owner = int(0)
            span = stride // 2
            while span > 0:
                if lane_read(first, owner + span, _FILL_LANES) <= q:
                    owner = owner + span
                span = span // 2
            owner_g = lane_read(g, owner, _FILL_LANES)
            owner_first = lane_read(first, owner, _FILL_LANES)
            owner_c0 = lane_read(c0, owner, _FILL_LANES)
            owner_line = lane_read(line, owner, _FILL_LANES)
            owner_source = lane_read(source, owner, _FILL_LANES)
            owner_c = lane_read(c, owner, _FILL_LANES)
            owner_sn = wp.vec3(
                lane_read(sn[0], owner, _FILL_LANES),
                lane_read(sn[1], owner, _FILL_LANES),
                lane_read(sn[2], owner, _FILL_LANES),
            )
            owner_point = wp.vec3(
                lane_read(point[0], owner, _FILL_LANES),
                lane_read(point[1], owner, _FILL_LANES),
                lane_read(point[2], owner, _FILL_LANES),
            )
            springs = False
            if q < total:
                springs = fill_region_member(
                    owner_g, q - owner_first, owner_c0, owner_line, owner_source, owner_c, owner_sn, owner_point,
                    rows, bias, data, law, pitch, cand_contact, reg_frame, reg_wnn, cand, cand_row,
                )  # fmt: skip
            spring_bits = wp.uint64(lane_ballot(springs, _FILL_LANES))
            # This lane's region's candidates in this round are lanes [lo, hi).
            lo = wp.max(first - dealt, 0)
            hi = wp.min(first + size - dealt, stride)
            if lo < hi:
                mine = (wp.uint64(1) << wp.uint64(hi)) - (wp.uint64(1) << wp.uint64(lo))
                if (spring_bits & mine) != wp.uint64(0):
                    kind = kind | KIND_SPRINGS
        lane_sync(_FILL_LANES)
        if holds:
            reg_kind[g] = kind
            reg_reach[g] = idle_reach(cand, c0, c0 + size, reg_arm[g], reg_arm_b[g], reg_body[g][1] >= 0)
        g += threads


@wp.kernel(enable_backward=False, module="unique")
def form_blocks(
    relist: wp.array[int],
    world_regions: wp.array[wp.vec2i],
    world_free_start: wp.array[int],
    world_free: wp.array[int],
    reg_body: wp.array[wp.vec2i],
    least: int,
    reg_cand: wp.array[wp.vec2i],
    cand: wp.array[wp.vec4],
    tol: float,
    capacity: int,
    body_level: wp.array[int],
    blk_count: wp.array2d[int],
    blk_cursor: wp.array[int],
    blk_blob: wp.array[wp.vec4i],
    blk_head: wp.array[int],
    reg_stage: wp.array[wp.vec3i],
    reg_kind: wp.array[int],
    blk_pair: wp.array[wp.vec2i],
):
    """A world's groups (a pair of movable bodies, or a body against immovable partners) with more than ``least``
    regions closing at the predictor (a candidate's gap term below ``-tol``; ``least`` the group's six relative
    coordinates: its rows then over-determine them), every candidate compliant, as blocks (the
    largest first, :data:`PAIR_BLOCKS` at most), a thread a world: the pair's first region in the stream stands for
    the block (:data:`KIND_BLOCK`), the others are its members (:data:`KIND_MEMBER`); its pair ``(a, b)``
    (:func:`factor_blocks` forms its response each step). The blocks are the contact frame's: they are formed where
    the step lists its regions anew, at that step's predictor. ``blk_count`` counts a world's pairs by its bodies' places (``body_level`` holds them here), a
    side of the world's free bodies; a world of more than 16 free bodies lists no block."""
    if relist[0] == 0:
        return
    world = wp.tid()
    for k in range(PAIR_BLOCKS):
        blk_pair[world * PAIR_BLOCKS + k] = wp.vec2i(-1, -1)
    span = world_regions[world]
    s = span[0]
    e = span[1]
    f0 = world_free_start[world]
    # The table's side is the world's free bodies (at most 16, the table's row of 256 cells): its cells in order are
    # the pairs of places in order, whatever the side.
    side = world_free_start[world + 1] - f0
    if e - s <= least or side > 16:
        return
    for i in range(f0, world_free_start[world + 1]):
        body_level[world_free[i]] = i - f0
    for c in range(side * side):
        blk_count[world, c] = 0
    rigid = int(1 << 20)
    for g in range(s, e):
        sides = reg_body[g]
        la = body_level[sides[0]]
        lb = la
        if sides[1] >= 0:
            lb = body_level[sides[1]]
        cell = wp.min(la, lb) * side + wp.max(la, lb)
        # A region counts where a candidate of it closes at the predictor (its gap term).
        add = int(0)
        span_c = reg_cand[g]
        for i in range(span_c[0], span_c[1]):
            if cand[i][2] < -tol:
                add = 1
        if (reg_kind[g] & KIND_SPRINGS) == 0:
            add = rigid
        blk_count[world, cell] = blk_count[world, cell] + add
    for k in range(PAIR_BLOCKS):
        best = int(least)
        best_cell = int(-1)
        for c in range(side * side):
            count = blk_count[world, c]
            if count < rigid and count > best:
                best = count
                best_cell = c
        if best_cell < 0:
            break
        blk_count[world, best_cell] = 0
        # Its staged records' words, taken from the arena.
        words = int(0)
        regions = int(0)
        candidates = int(0)
        for g in range(s, e):
            sides = reg_body[g]
            la = body_level[sides[0]]
            lb = la
            if sides[1] >= 0:
                lb = body_level[sides[1]]
            if wp.min(la, lb) * side + wp.max(la, lb) == best_cell:
                span_c = reg_cand[g]
                regions += 1
                candidates += span_c[1] - span_c[0]
        words = STAGE_UNIT * regions + STAGE_CAND * candidates
        at = wp.atomic_add(blk_cursor, 0, words)
        if at + words > capacity:
            blk_cursor[1] = 1
            break
        best_g = int(-1)
        members = int(0)
        first_cand = int(0)
        for g in range(s, e):
            sides = reg_body[g]
            la = body_level[sides[0]]
            lb = la
            if sides[1] >= 0:
                lb = body_level[sides[1]]
            if wp.min(la, lb) * side + wp.max(la, lb) == best_cell:
                reg_stage[g] = wp.vec3i(at + STAGE_UNIT * members, world * PAIR_BLOCKS + k, first_cand)
                members += 1
                span_m = reg_cand[g]
                first_cand += span_m[1] - span_m[0]
                if best_g < 0:
                    best_g = g
                    reg_kind[g] = reg_kind[g] | KIND_BLOCK | (k << BLOCK_SHIFT)
                else:
                    reg_kind[g] = reg_kind[g] | KIND_MEMBER | (k << BLOCK_SHIFT)
        blk_blob[world * PAIR_BLOCKS + k] = wp.vec4i(at, members, at + STAGE_UNIT * members, first_cand)
        blk_head[world * PAIR_BLOCKS + k] = best_g
        sides = reg_body[best_g]
        # A body against immovable partners is the pair (a, -1).
        a = sides[0]
        b = sides[1]
        if b >= 0:
            a = wp.min(sides[0], sides[1])
            b = wp.max(sides[0], sides[1])
        blk_pair[world * PAIR_BLOCKS + k] = wp.vec2i(a, b)


@wp.kernel(enable_backward=False, module="unique")
def factor_blocks(
    blk_pair: wp.array[wp.vec2i],
    blk_head: wp.array[int],
    reg_body: wp.array[wp.vec2i],
    reg_arm: wp.array[wp.vec3],
    reg_arm_b: wp.array[wp.vec3],
    body_inertia: wp.array2d[wp.mat33],
    blk_L: wp.array[wp.spatial_matrix],
    blk_d: wp.array[wp.vec3],
    blk_wrench: wp.array[wp.spatial_vector],
):
    """Each block's response at the step's poses, a thread a block's slot (:func:`form_blocks`): the bodies' origins'
    offset ``d = o_a - o_b``, the factor ``L`` of the pair's relative response ``W = A_a + P A_b P^T`` (``P`` moves
    b's twist to a's origin; ``W = A_a`` against immovable partners; ``W = L L^T``, lower), and its held wrenches
    (normal, friction) at zero."""
    bid = wp.tid()
    blk_wrench[2 * bid] = wp.spatial_vector()
    blk_wrench[2 * bid + 1] = wp.spatial_vector()
    pair = blk_pair[bid]
    a = pair[0]
    b = pair[1]
    if a < 0:
        return
    best_g = blk_head[bid]
    sides = reg_body[best_g]
    d = wp.vec3(0.0, 0.0, 0.0)
    Wvv = body_inertia[a, 0]
    Wvw = body_inertia[a, 1]
    Www = body_inertia[a, 2]
    if b >= 0:
        d = reg_arm_b[best_g] - reg_arm[best_g]
        if sides[0] != a:
            d = -d
        # W = A_a + P A_b P^T, P = [[I, -[d]x], [0, I]].
        X = wp.skew(d)
        Bvv = body_inertia[b, 0]
        Bvw = body_inertia[b, 1]
        Bww = body_inertia[b, 2]
        Pvv = Bvv - X * wp.transpose(Bvw) - Bvw * wp.transpose(X) + X * Bww * wp.transpose(X)
        Pvw = Bvw - X * Bww
        Wvv = Wvv + Pvv
        Wvw = Wvw + Pvw
        Www = Www + Bww
    Wm = wp.spatial_matrix()
    for i in range(3):
        for j in range(3):
            Wm[i, j] = Wvv[i, j]
            Wm[i, j + 3] = Wvw[i, j]
            Wm[i + 3, j] = Wvw[j, i]
            Wm[i + 3, j + 3] = Www[i, j]
    Lm = wp.spatial_matrix()
    for j in range(6):
        t = Wm[j, j]
        for k2 in range(j):
            t -= Lm[j, k2] * Lm[j, k2]
        ljj = wp.sqrt(wp.max(t, 1.0e-30))
        Lm[j, j] = ljj
        for i in range(j + 1, 6):
            t2 = Wm[i, j]
            for k2 in range(j):
                t2 -= Lm[i, k2] * Lm[j, k2]
            Lm[i, j] = t2 / ljj
    blk_L[bid] = Lm
    blk_d[bid] = d


@wp.kernel(enable_backward=False, module="unique")
def stage_blocks(
    cursor: wp.array[int],
    threads: int,
    reg_stage: wp.array[wp.vec3i],
    blk_blob: wp.array[wp.vec4i],
    blk_pair: wp.array[wp.vec2i],
    blk_L: wp.array[wp.spatial_matrix],
    reg_body: wp.array[wp.vec2i],
    reg_kind: wp.array[int],
    reg_frame: wp.array[wp.mat33],
    reg_arm: wp.array[wp.vec3],
    reg_arm_b: wp.array[wp.vec3],
    reg_cand: wp.array[wp.vec2i],
    reg_law: wp.array[wp.vec2],
    reg_bf: wp.array[wp.vec3],
    reg_row: wp.array[int],
    reg_wff: wp.array[wp.mat33],
    reg_ball_values: wp.array[wp.vec3],
    reg_ball_vectors: wp.array[wp.mat33],
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
    cfm: float,
    blk_stage: wp.array[float],
    blk_istage: wp.array[int],
):
    """Each block member staged at its place in its block's arena words (``reg_stage``: the place and the block,
    :func:`form_blocks`), ``threads`` lanes striding over the stream's listed regions (``cursor[0]``): the member's record
    (:data:`STAGE_UNIT`: its unit's rows whitened by the pair's factor, ``L^T f`` for ``f = sigma [(t0, r x t0),
    (t1, r x t1), (0, n)]`` about a's origin, their right-hand sides, the cone law's block (``G`` plus ``cfm``) or
    metric, no friction), then its candidates' (:data:`STAGE_CAND`: ``L^T j`` for
    ``j = sigma (n, r x n + a1 t0 + a2 t1)``, the right-hand side, the compliance); their impulse rows
    at the same words of ``blk_istage``."""
    if cursor[3] != 0:
        return
    for g in range(wp.tid(), cursor[0], threads):
        kind = reg_kind[g]
        if (kind & (KIND_BLOCK | KIND_MEMBER)) == 0:
            continue
        where = reg_stage[g]
        off = where[0]
        pair = blk_pair[where[1]]
        Lt = wp.transpose(blk_L[where[1]])
        frame = reg_frame[g]
        n = frame[0]
        t0 = frame[1]
        t1 = frame[2]
        sg = float(1.0)
        if (kind & KIND_SECOND) != 0:
            sg = -1.0
        r = reg_arm[g]
        if reg_body[g][0] != pair[0]:
            r = reg_arm_b[g]
            sg = -sg
        c0 = wp.cross(r, t0)
        c1 = wp.cross(r, t1)
        h0 = Lt * wp.spatial_vector(sg * t0[0], sg * t0[1], sg * t0[2], sg * c0[0], sg * c0[1], sg * c0[2])
        h1 = Lt * wp.spatial_vector(sg * t1[0], sg * t1[1], sg * t1[2], sg * c1[0], sg * c1[1], sg * c1[2])
        h2 = Lt * wp.spatial_vector(0.0, 0.0, 0.0, sg * n[0], sg * n[1], sg * n[2])
        for k in range(6):
            blk_stage[off + k] = h0[k]
            blk_stage[off + 6 + k] = h1[k]
            blk_stage[off + 12 + k] = h2[k]
        friction = kind & FRICTION_ROWS
        bf = reg_bf[g]
        law = reg_law[g]
        blk_stage[off + 18] = bf[0]
        blk_stage[off + 19] = bf[1]
        blk_stage[off + 20] = bf[2]
        blk_stage[off + 21] = law[0]
        blk_stage[off + 22] = law[1]
        blk_stage[off + 23] = float(friction)
        M = reg_ball_vectors[g]
        values = reg_ball_values[g]
        if law[1] <= 0.0:
            G = reg_wff[g]
            M = G
            if friction < 3:
                M = wp.mat33(G[0, 0], G[0, 1], 0.0, G[1, 0], G[1, 1], 0.0, 0.0, 0.0, 1.0)
            M = M + wp.mat33(cfm, 0.0, 0.0, 0.0, cfm, 0.0, 0.0, 0.0, cfm)
            values = wp.vec3(0.0, 0.0, 0.0)
        for i in range(3):
            for j in range(3):
                blk_stage[off + 25 + 3 * i + j] = M[i, j]
            blk_stage[off + 34 + i] = values[i]
        span_c = reg_cand[g]
        blk_stage[off + 37] = float(span_c[1] - span_c[0])
        blk_stage[off + 39] = 0.0
        blk_stage[off + 40] = 0.0
        blk_stage[off + 41] = 0.0
        blk_stage[off + 42] = float(where[2])
        blk_istage[off] = reg_row[g]
        rn = wp.cross(r, n)
        at = blk_blob[where[1]][2] + STAGE_CAND * where[2]
        for i in range(span_c[0], span_c[1]):
            rec = cand[i]
            ang = rn + t0 * rec[0] + t1 * rec[1]
            z = Lt * wp.spatial_vector(sg * n[0], sg * n[1], sg * n[2], sg * ang[0], sg * ang[1], sg * ang[2])
            for k in range(6):
                blk_stage[at + k] = z[k]
            blk_stage[at + 6] = rec[2]
            blk_stage[at + 7] = rec[3]
            blk_stage[at + 8] = 0.0
            blk_istage[at] = cand_row[i]
            at += STAGE_CAND


@wp.kernel(enable_backward=False, module="unique")
def level_regions(
    relist: wp.array[int],
    cursor: wp.array[int],
    world_regions: wp.array[wp.vec2i],
    world_free_start: wp.array[int],
    world_free: wp.array[int],
    reg_body: wp.array[wp.vec2i],
    reg_kind: wp.array[int],
    body_level: wp.array[int],
    reg_level: wp.array[int],
    reg_order: wp.array[int],
    level_end: wp.array[int],
    level_region: wp.array[int],
    world_levels: wp.array[int],
    blk_members: wp.array[wp.vec2i],
):
    """A world's open regions in their order from the free end down (the contact graph's), by the exact levels of
    that order, a thread a world.

    The order: a body's depth is its distance on the contact graph from the immovable partners (one for a body
    with a region against one, one more for each region between two movable bodies on the way), a region's the
    larger of its bodies', and the regions are taken by decreasing depth (a part the graph does not reach comes
    first), in their stream order within a depth: a cold sweep then carries a pile's weight down to its supports
    within the sweep, where the order up from the supports carries it one interface a sweep.

    The levels: a region's level is one past the last level of a region before it on either of its bodies, so a
    level's regions share no body and a sweep that takes the levels in turn, a level's regions at once, is the
    sequential sweep of the order itself. The world's span ``[s, e)`` of the stream holds the regions by level
    (``level_region``) and, at ``s + L``, the end of level ``L`` (``level_end``); ``world_levels`` the count.

    The levels are the contact frame's: they are made where the step lists its regions anew, whose listing then
    stands for the frame's later steps if it fit (``relist[1]``)."""
    if relist[0] == 0:
        return
    world = wp.tid()
    if world == 0:
        relist[1] = int(cursor[3] == 0)
    span = world_regions[world]
    s = span[0]
    e = span[1]
    for k in range(PAIR_BLOCKS):
        blk_members[world * PAIR_BLOCKS + k] = wp.vec2i(0, 0)
    if e <= s:
        world_levels[world] = 0
        return
    far = int(1 << 29)
    # A block's members (form_blocks) leave the order: the region that stands for the block takes their place.
    # The depths: relaxed over the regions until they stand (each pass takes them in the stream's order).
    for i in range(world_free_start[world], world_free_start[world + 1]):
        body_level[world_free[i]] = far
    for g in range(s, e):
        sides = reg_body[g]
        if sides[1] < 0 and (reg_kind[g] & KIND_MEMBER) == 0:
            body_level[sides[0]] = 1
    changed = int(1)
    while changed != 0:
        changed = int(0)
        for g in range(s, e):
            sides = reg_body[g]
            if sides[1] >= 0 and (reg_kind[g] & KIND_MEMBER) == 0:
                a = body_level[sides[0]]
                b = body_level[sides[1]]
                if b < far and b + 1 < a:
                    body_level[sides[0]] = b + 1
                    changed = 1
                elif a < far and a + 1 < b:
                    body_level[sides[1]] = a + 1
                    changed = 1
    deepest = int(0)
    unreached = int(0)
    for g in range(s, e):
        if (reg_kind[g] & KIND_MEMBER) != 0:
            continue
        sides = reg_body[g]
        depth = body_level[sides[0]]
        if sides[1] >= 0:
            depth = wp.max(depth, body_level[sides[1]])
        reg_level[g] = depth
        if depth < far:
            deepest = wp.max(deepest, depth)
        else:
            unreached = 1
    # By decreasing depth (unreached first), stable: a counting sort on the span's slots. A body at depth d has d
    # regions on its way down, so the keys (the depths, and the unreached where some are) are at most e - s.
    top = deepest + unreached
    for k in range(top):
        level_end[s + k] = 0
    kept = int(0)
    for g in range(s, e):
        if (reg_kind[g] & KIND_MEMBER) != 0:
            continue
        reg_level[g] = top - wp.min(reg_level[g], deepest + 1)
        level_end[s + reg_level[g]] += 1
        kept += 1
    at = s
    for k in range(top):
        count = level_end[s + k]
        level_end[s + k] = at
        at += count
    for g in range(s, e):
        if (reg_kind[g] & KIND_MEMBER) != 0:
            continue
        place = level_end[s + reg_level[g]]
        reg_order[place] = g
        level_end[s + reg_level[g]] = place + 1
    # The members after the order, block by block, in the stream's order.
    at = s + kept
    for k in range(PAIR_BLOCKS):
        m0 = at
        for g in range(s, e):
            kind = reg_kind[g]
            if (kind & KIND_MEMBER) != 0 and (kind >> BLOCK_SHIFT) == k:
                reg_order[at] = g
                at += 1
        blk_members[world * PAIR_BLOCKS + k] = wp.vec2i(m0, at)
    # The levels of that order.
    for i in range(world_free_start[world], world_free_start[world + 1]):
        body_level[world_free[i]] = 0
    levels = int(0)
    for p in range(s, s + kept):
        g = reg_order[p]
        sides = reg_body[g]
        level = body_level[sides[0]]
        if sides[1] >= 0:
            level = wp.max(level, body_level[sides[1]])
        reg_level[g] = level
        body_level[sides[0]] = level + 1
        if sides[1] >= 0:
            body_level[sides[1]] = level + 1
        levels = wp.max(levels, level + 1)
    for level in range(levels):
        level_end[s + level] = 0
    for p in range(s, s + kept):
        level_end[s + reg_level[reg_order[p]]] += 1
    at = s
    for level in range(levels):
        count = level_end[s + level]
        level_end[s + level] = at
        at += count
    for p in range(s, s + kept):
        g = reg_order[p]
        place = level_end[s + reg_level[g]]
        level_region[place] = g
        level_end[s + reg_level[g]] = place + 1
    world_levels[world] = levels


# ---------------------------------------------------------------------------------------------------------------------
# The step's regions: their storage and launches
# ---------------------------------------------------------------------------------------------------------------------

_REGION_FIELDS = (
    ("head_size", wp.int32, 1),
    ("listed", wp.int32, 1),
    ("reg_place", wp.vec4i, 1),
    ("reg_head", wp.vec2i, 1),
    ("reg_frame", wp.mat33, 1),
    ("reg_arm", wp.vec3, 1),
    ("reg_arm_b", wp.vec3, 1),
    ("reg_body", wp.vec2i, 1),
    ("reg_cand", wp.vec2i, 1),
    ("reg_law", wp.vec2, 1),
    ("reg_bf", wp.vec3, 1),
    ("reg_kind", wp.int32, 1),
    ("reg_row", wp.int32, 1),
    ("reg_wnn", wp.mat33, 1),
    ("reg_wnf", wp.mat33, 1),
    ("reg_wff", wp.mat33, 1),
    ("reg_ball_values", wp.vec3, 1),
    ("reg_ball_vectors", wp.mat33, 1),
    ("reg_ginv", wp.mat33, 1),
    ("reg_k", wp.mat33, 1),
    ("reg_lk", wp.mat33, 1),
    ("reg_lw", wp.mat33, 1),
    ("cand", wp.vec4, 1),
    ("cand_row", wp.int32, 1),
    ("cand_contact", wp.int32, 1),
    ("reg_wrench", wp.vec3, 1),
    ("reg_phi", wp.vec3, 1),
    ("reg_set", wp.vec3i, 1),
    ("reg_spring", wp.vec3, 1),
    ("reg_state", wp.int32, 1),
    ("reg_stamp", wp.int32, 1),
    ("reg_reach", wp.vec2, 1),
    ("reg_level", wp.int32, 1),
    ("reg_order", wp.int32, 1),
    ("level_region", wp.int32, 1),
    ("level_end", wp.int32, 1),
    ("reg_stage", wp.vec3i, 1),
)
"""The regions' stream, by contact row of the pool: a region's record and state at its place, a candidate's at its
own (no step holds more regions or candidates than contacts)."""


class Regions:
    """The free bodies' regions of a step (the module's statement): the free bodies' tables and bases, the regions'
    stream, and the launches that fill and sweep them.

    Args:
        model: The solver's model.
        forest: The forest plan.
        placement: The placement, whose contact pool sizes the stream and whose row store holds the candidates'
            impulses.
        factor: The tree factor (the walk the bases are read from).
        options: The checked options.
    """

    def __init__(self, model, forest, placement, factor, options):
        self.model, self.forest, self.placement, self.factor, self.options = model, forest, placement, factor, options
        device = model.device
        # The free bodies are the forest's: those that respond, in its order, which is the order of their speed rows'
        # slots (Placement.rigid_velocity_limit_slot). A prescribed free body is no row's side and holds no region.
        free_art = np.zeros(0, dtype=np.int32)
        if forest.free_rigid_body_count:
            bodies = forest.free_rigid_body_indices.numpy()
            free_art = forest.body_to_articulation.numpy()[bodies].astype(np.int32)
        count = int(free_art.size)
        self.count = count
        art_free = np.full(max(int(forest.is_free_rigid_host.size), 1), -1, dtype=np.int32)
        art_free[free_art] = np.arange(count, dtype=np.int32)
        worlds = max(int(forest.world_count), 1)
        world = np.asarray(forest.articulation_world)[free_art]
        order = np.argsort(world, kind="stable").astype(np.int32)
        start = np.searchsorted(world[order], np.arange(worlds + 1)).astype(np.int32)
        offsets = forest.articulation_world_dof_offset.numpy()
        self.free_art = wp.array(free_art, dtype=wp.int32, device=device)
        self.art_free = wp.array(art_free, dtype=wp.int32, device=device)
        self.world_free = wp.array(order if count else np.zeros(1, dtype=np.int32), dtype=wp.int32, device=device)
        self.world_free_start = wp.array(start, dtype=wp.int32, device=device)
        self.body_world = wp.array(world if count else np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
        self.body_first = wp.array(offsets[free_art], dtype=wp.int32, device=device)
        self.body_dof = wp.array(
            np.asarray(forest.articulation_dof_start_host)[free_art], dtype=wp.int32, device=device
        )
        self.body_basis = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
        self.body_inertia = wp.zeros((count, 3), dtype=wp.mat33, device=device)
        self.body_twist = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        # A body no world lists (it has no world) is open: only a world's sweep may visit its regions.
        self.body_open = wp.ones(count, dtype=wp.int32, device=device)
        self.body_count = wp.zeros((count, 3), dtype=wp.int32, device=device)
        self.body_cand = wp.zeros(count, dtype=wp.int32, device=device)
        self.body_seen = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        self.body_version = wp.zeros(count, dtype=wp.int32, device=device)
        self.body_regions = wp.zeros(count, dtype=wp.vec2i, device=device)
        self.body_touch = wp.zeros(max(count, 1), dtype=wp.int32, device=device)
        self.body_pair = wp.zeros(count, dtype=wp.int32, device=device)
        self.lanes = wp.zeros(count, dtype=wp.int32, device=device)
        self.lane_change = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        self.cursor = wp.zeros(4, dtype=wp.int32, device=device)
        # Whether a step lists its regions anew ([0]) and whether the last listing fit the stream ([1]: none yet).
        self.relist = wp.array([1, 0], dtype=wp.int32, device=device)
        self.no_hold = wp.zeros(1, dtype=wp.int32, device=device)
        self.world_regions = wp.zeros(worlds, dtype=wp.vec2i, device=device)
        self.world_levels = wp.zeros(worlds, dtype=wp.int32, device=device)
        self.body_level = wp.zeros(max(count, 1), dtype=wp.int32, device=device)
        blocks = worlds * int(PAIR_BLOCKS)
        self.blk_pair = wp.full(blocks, wp.vec2i(-1, -1), dtype=wp.vec2i, device=device)
        self.blk_members = wp.zeros(blocks, dtype=wp.vec2i, device=device)
        self.blk_L = wp.zeros(blocks, dtype=wp.spatial_matrix, device=device)
        self.blk_d = wp.zeros(blocks, dtype=wp.vec3, device=device)
        self.blk_wrench = wp.zeros(2 * blocks, dtype=wp.spatial_vector, device=device)
        # Blocks are the law of compliant candidates only: a model whose shapes author no material lists none.
        flags = model.shape_flags.numpy() if model.shape_count else np.zeros(0, dtype=np.int32)
        self.blocks = bool((flags & int(ShapeFlags.CONTACT_MATERIAL)).any())
        self.blk_count = wp.zeros((worlds if self.blocks else 1, 256), dtype=wp.int32, device=device)
        self.blk_cursor = wp.zeros(2, dtype=wp.int32, device=device)
        self.blk_blob = wp.zeros(blocks, dtype=wp.vec4i, device=device)
        self.blk_head = wp.zeros(blocks, dtype=wp.int32, device=device)
        words = STAGE_WORDS if self.blocks else 1
        self.blk_stage = wp.zeros(words, dtype=wp.float32, device=device)
        self.blk_istage = wp.zeros(words, dtype=wp.int32, device=device)
        self.capacity = 0
        # The stream's capacity on the device: the pool's ready rows, which follow its growth between captures.
        self.ready = placement.contact_pool.ready_rows
        if self.ready is None:
            self.ready = wp.zeros(1, dtype=wp.int32, device=device)
        self.rows = placement.contact_pool.rows(
            [ContactField(name, dtype, per_contact=per) for name, dtype, per in _REGION_FIELDS], on_rows=self._on_rows
        )

    def _on_rows(self, rows) -> None:
        """Take the stream's arrays at the contact pool's ready rows, on creation and after each growth: the next step
        lists its regions anew."""
        rows.publish(self)
        self.relist.zero_()
        self.capacity = int(rows.rows)
        if self.placement.contact_pool.ready_rows is None:
            self.ready.fill_(self.capacity)

    def bases(self, v_hat: wp.array, contacts) -> None:
        """Each free body's basis (:func:`free_body_basis`), on a step with contacts: it reads the factor's walk and the
        predictor's ``v_hat`` only, so it runs from the predictor on, before the regions are listed (:meth:`list`)."""
        if contacts is None or contacts.rigid_contact_count is None or contacts._rigid_contact_rows <= 0:
            return
        wp.launch(
            free_body_basis,
            dim=self.count,
            inputs=[
                self.free_art,
                self.factor.walk_view(),
                self.factor.joint_S_s,
                self.forest.articulation_dof_start,
                v_hat,
            ],
            outputs=[self.body_basis, self.body_inertia, self.body_twist],
            device=self.model.device,
        )

    def list(self, solver, state_in, contacts, dt: float) -> None:
        """The step's regions, once the rows' kinds are written and the patch link made their rings, and each free
        body's basis (:meth:`bases`).

        Every step: the checks of the listing it holds
        (:func:`begin_listing`, :func:`check_contacts`, :func:`check_bodies`). Where the patch builder made the contact
        frame's regions anew, or a check failed, the step lists them anew: the regions marked and counted a contact
        (:func:`mark_regions`), each world's bodies placed in the stream (:func:`place_bodies`), the regions placed
        and ordered a body with their candidates' contacts (:func:`place_regions`, :func:`order_regions`), then,
        after the fill, the frame's blocks and levels (:func:`form_blocks`, :func:`level_regions`). Every step forms
        each region at the step's rows and poses (:func:`fill_regions`) and its blocks' responses and records
        (:func:`factor_blocks`, :func:`stage_blocks`)."""
        from .contact_input import bind as bind_contact_input  # noqa: PLC0415
        from .friction_patches import patch_row_law  # noqa: PLC0415
        from .row_records import bind as bind_dense_rows  # noqa: PLC0415

        placement, forest, device = self.placement, self.forest, self.model.device
        pool = placement.contact_pool
        if contacts is None or contacts.rigid_contact_count is None or contacts._rigid_contact_rows <= 0:
            # No region: the stream empty, and the next step with contacts lists anew.
            for array in (self.cursor, self.body_regions, self.world_regions, self.world_levels):
                array.zero_()
            wp.launch(
                begin_listing,
                dim=max(self.count, 1),
                inputs=[self.no_hold],
                outputs=[self.relist, self.body_seen, self.body_version, self.body_touch],
                device=device,
            )
            self.relist.zero_()
            return
        wp.launch(
            begin_listing,
            dim=max(self.count, 1),
            inputs=[solver.laws.friction_patches.held_regions()],
            outputs=[self.relist, self.body_seen, self.body_version, self.body_touch],
            device=device,
        )
        rows, bias = bind_dense_rows(placement), solver.laws.bias(self.options, dt)
        data = bind_contact_input(solver, state_in, self.factor, contacts, dt, 1.0)
        mf_max, pitch = int(placement.mf_max_constraints), int(placement.row_pitch)
        dense_max = int(placement.dense_max_constraints)
        # Sized for the pool's launch rows: a capture's replays run them at the live count (ContactPool.launch).
        threads = min(pool.launch_rows(), _THREAD_CAP)
        live = {"extent": contacts.rigid_contact_count, "clamp": True}
        pool.launch(
            check_contacts,
            threads,
            inputs=[data, threads, self.art_free, self.listed],
            outputs=[self.body_touch, self.relist],
            **live,
        )
        speed = placement.rigid_velocity_limit_slot
        if speed is not None and speed.shape[0] != RIGID_SPEED_ROWS * self.count:
            raise ValueError(
                f"the speed rows' slots are {speed.shape[0]}, not {RIGID_SPEED_ROWS} a body of the stream's {self.count}"
            )
        wp.launch(
            check_bodies,
            dim=self.count,
            inputs=[self.lanes if speed is None else speed, 0 if speed is None else RIGID_SPEED_ROWS, self.body_pair],
            outputs=[self.body_open, self.body_touch, self.relist],
            device=device,
        )
        wp.launch(
            clear_layout,
            dim=max(self.count, 4),
            inputs=[self.relist],
            outputs=[self.cursor, self.blk_cursor, self.body_count, self.body_regions, self.body_pair],
            device=device,
        )
        pool.launch(
            mark_regions,
            threads,
            inputs=[self.relist, data, rows, mf_max, threads, self.art_free],
            outputs=[self.head_size, self.listed, self.body_pair, self.body_count],
            **live,
        )
        wp.launch(
            place_bodies,
            dim=forest.world_count,
            inputs=[
                self.relist,
                self.world_free_start,
                self.world_free,
                self.body_touch,
                self.body_pair,
                self.ready,
                self.body_count,
            ],
            outputs=[self.cursor, self.body_open, self.body_regions, self.body_cand, self.lanes, self.world_regions],
            device=device,
        )
        pool.launch(
            place_regions,
            threads,
            inputs=[self.relist, data, rows, pitch, threads, self.art_free, self.head_size, self.body_regions],
            outputs=[self.body_count, self.reg_place],
            **live,
        )
        pool.launch(
            order_regions,
            threads,
            inputs=[
                self.relist,
                self.cursor,
                threads,
                data,
                patch_row_law(solver),
                self.body_world,
                self.body_regions,
                self.body_cand,
                self.reg_place,
            ],
            outputs=[self.reg_head, self.reg_cand, self.reg_kind, self.cand_contact, self.listed],
            **live,
        )
        pool.launch(
            fill_regions,
            threads,
            inputs=[
                rows,
                bias,
                data,
                patch_row_law(solver),
                placement.mf_constraint_count,
                placement.mf_contact_rows_end,
                dense_max,
                mf_max,
                pitch,
                self.cand_contact,
                self.art_free,
                self.body_twist,
                self.body_inertia,
                float(self.options.pgs_cfm),
                self.cursor,
                threads,
                self.reg_head,
                self.reg_cand,
            ],
            outputs=[
                self.reg_frame,
                self.reg_arm,
                self.reg_arm_b,
                self.reg_body,
                self.reg_law,
                self.reg_bf,
                self.reg_kind,
                self.reg_row,
                self.reg_wnn,
                self.reg_wnf,
                self.reg_wff,
                self.reg_ball_values,
                self.reg_ball_vectors,
                self.reg_ginv,
                self.reg_k,
                self.reg_lk,
                self.reg_lw,
                self.cand,
                self.cand_row,
                self.reg_wrench,
                self.reg_phi,
                self.reg_set,
                self.reg_spring,
                self.reg_state,
                self.reg_reach,
            ],
            **live,
        )
        if self.blocks:
            wp.launch(
                form_blocks,
                dim=forest.world_count,
                inputs=[
                    self.relist,
                    self.world_regions,
                    self.world_free_start,
                    self.world_free,
                    self.reg_body,
                    BLOCK_CLOSING,
                    self.reg_cand,
                    self.cand,
                    REGION_TOLERANCE,
                    int(self.blk_stage.shape[0]),
                ],
                outputs=[
                    self.body_level,
                    self.blk_count,
                    self.blk_cursor,
                    self.blk_blob,
                    self.blk_head,
                    self.reg_stage,
                    self.reg_kind,
                    self.blk_pair,
                ],
                device=device,
            )
            wp.launch(
                factor_blocks,
                dim=self.blk_pair.shape[0],
                inputs=[self.blk_pair, self.blk_head, self.reg_body, self.reg_arm, self.reg_arm_b, self.body_inertia],
                outputs=[self.blk_L, self.blk_d, self.blk_wrench],
                device=device,
            )
        wp.launch(
            level_regions,
            dim=forest.world_count,
            inputs=[
                self.relist,
                self.cursor,
                self.world_regions,
                self.world_free_start,
                self.world_free,
                self.reg_body,
                self.reg_kind,
            ],
            outputs=[
                self.body_level,
                self.reg_level,
                self.reg_order,
                self.level_end,
                self.level_region,
                self.world_levels,
                self.blk_members,
            ],
            device=device,
        )
        if self.blocks:
            pool.launch(
                stage_blocks,
                threads,
                inputs=[
                    self.cursor,
                    threads,
                    self.reg_stage,
                    self.blk_blob,
                    self.blk_pair,
                    self.blk_L,
                    self.reg_body,
                    self.reg_kind,
                    self.reg_frame,
                    self.reg_arm,
                    self.reg_arm_b,
                    self.reg_cand,
                    self.reg_law,
                    self.reg_bf,
                    self.reg_row,
                    self.reg_wff,
                    self.reg_ball_values,
                    self.reg_ball_vectors,
                    self.cand,
                    self.cand_row,
                    float(self.options.pgs_cfm),
                ],
                outputs=[self.blk_stage, self.blk_istage],
                **live,
            )

    def sweep_closed(self) -> None:
        """The closed bodies' sweeps (:func:`sweep_closed_bodies`), a lane a free body: each stores its body's change
        (:meth:`add_closed`). They touch nothing another sweep touches, so they may run beside them."""
        if self.count == 0:
            return
        impulses = self.placement.impulses
        wp.launch(
            sweep_closed_bodies,
            dim=self.count,
            inputs=[
                self.cursor,
                self.lanes,
                self.body_regions,
                self.body_inertia,
                self.body_basis,
                self.reg_frame,
                self.reg_arm,
                self.reg_cand,
                self.reg_law,
                self.reg_bf,
                self.reg_kind,
                self.reg_row,
                self.reg_wnn,
                self.reg_wnf,
                self.reg_wff,
                self.reg_ball_values,
                self.reg_ball_vectors,
                self.reg_ginv,
                self.reg_k,
                self.reg_lk,
                self.reg_lw,
                self.cand,
                self.cand_row,
                int(self.options.pgs_iterations),
                REGION_TOLERANCE,
                float(self.options.pgs_cfm),
            ],
            outputs=[
                self.reg_wrench,
                self.reg_phi,
                self.reg_set,
                self.reg_spring,
                self.reg_state,
                self.reg_reach,
                impulses.reshape((impulses.size,)),
                self.lane_change,
            ],
            device=self.model.device,
        )

    def add_closed(self, du: wp.array) -> None:
        """The closed bodies' changes into the walk's ``du`` (:func:`add_closed_changes`), after every sweep."""
        if self.count == 0:
            return
        wp.launch(
            add_closed_changes,
            dim=self.count,
            inputs=[self.cursor, self.lanes, self.body_dof, self.lane_change],
            outputs=[du],
            device=self.model.device,
        )

    def audit(self, coordinates: wp.array, velocity: wp.array) -> None:
        """The regions' rows' velocity terms at the published velocity (its walk coordinates ``coordinates``) into the
        audit's free rows' terms ``velocity`` (:func:`audit_regions`), with the candidates' diagonals."""
        placement = self.placement
        wp.launch(
            audit_regions,
            dim=self.count + int(self.forest.world_count),
            inputs=[
                self.count,
                self.body_world,
                self.body_open,
                self.body_regions,
                self.world_regions,
                int(placement.row_pitch),
                int(placement.body_base),
                coordinates,
                self.body_first,
                self.body_basis,
                self.body_inertia,
                self.reg_frame,
                self.reg_arm,
                self.reg_arm_b,
                self.reg_body,
                self.reg_cand,
                self.reg_kind,
                self.reg_row,
                self.cand,
                self.cand_row,
            ],
            outputs=[placement.diag, velocity],
            device=self.model.device,
        )

    def operands(self) -> dict:
        """The stream as the general sweep's region phase takes it (:data:`.sweep_routes.REGION_OPERANDS`)."""
        names = "world_regions world_levels level_end level_region reg_order body_first body_basis"
        names += " reg_frame reg_arm reg_arm_b reg_body reg_cand reg_law reg_wnn reg_wnf reg_wff reg_ball_values"
        names += " reg_ball_vectors reg_ginv reg_k reg_lk reg_lw"
        names += " reg_bf reg_kind reg_row cand cand_row reg_wrench reg_phi reg_set reg_spring reg_state reg_stamp"
        names += " reg_reach body_seen body_version blk_pair blk_blob blk_L blk_d blk_wrench blk_stage blk_istage"
        return {**{name: getattr(self, name) for name in names.split()}, "region_cfm": float(self.options.pgs_cfm)}


@wp.func
def audit_region(
    g: int,
    world: int,
    pitch: int,
    base: int,
    coordinates: wp.array2d[float],
    body_first: wp.array[int],
    body_basis: wp.array2d[wp.spatial_vector],
    body_inertia: wp.array2d[wp.mat33],
    reg_frame: wp.array[wp.mat33],
    reg_arm: wp.array[wp.vec3],
    reg_arm_b: wp.array[wp.vec3],
    reg_body: wp.array[wp.vec2i],
    reg_cand: wp.array[wp.vec2i],
    reg_kind: wp.array[int],
    reg_row: wp.array[int],
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
    diag: wp.array2d[float],
    velocity: wp.array2d[float],
):
    """Region ``g``'s rows' velocity terms at the published velocity (its walk coordinates, ``coordinates``), into
    the audit's free rows' terms: each candidate's ``a . N`` and the friction rows' ``F``, with each candidate's own
    response as its row's diagonal."""
    frame = reg_frame[g]
    n = frame[0]
    t0 = frame[1]
    t1 = frame[2]
    sides = reg_body[g]
    kind = reg_kind[g]
    sign = float(1.0)
    if (kind & KIND_SECOND) != 0:
        sign = -1.0
    sn = wp.vec3(0.0, 0.0, 0.0)
    sf = wp.vec3(0.0, 0.0, 0.0)
    Wnn = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for side in range(2):
        f = sides[side]
        if f < 0:
            continue
        first = body_first[f]
        u = wp.spatial_vector()
        for k in range(6):
            u[k] = coordinates[world, first + k]
        v = wp.vec3(wp.dot(body_basis[f, 0], u), wp.dot(body_basis[f, 1], u), wp.dot(body_basis[f, 2], u))
        w = wp.vec3(wp.dot(body_basis[f, 3], u), wp.dot(body_basis[f, 4], u), wp.dot(body_basis[f, 5], u))
        arm = reg_arm[g]
        if side == 1:
            arm = reg_arm_b[g]
        tn, tf = frame_twist(v, w, arm, n, t0, t1)
        response, _c, _g = frame_response(body_inertia[f, 0], body_inertia[f, 1], body_inertia[f, 2], arm, n, t0, t1)
        Wnn += response
        sn += tn * sign
        sf += tf * sign
        sign = -sign
    span = reg_cand[g]
    for i in range(span[0], span[1]):
        record = cand[i]
        a = candidate_row(record)
        line = cand_row[i] / pitch
        row = cand_row[i] - line * pitch
        velocity[world, row - base] = wp.dot(a, sn)
        diag[line, row] = wp.dot(a, Wnn * a)
    friction = reg_row[g]
    if friction >= 0:
        line = friction / pitch
        row = friction - line * pitch - base
        velocity[world, row] = sf[0]
        velocity[world, row + 1] = sf[1]
        if (kind & FRICTION_ROWS) == 3:
            velocity[world, row + 2] = sf[2]


@wp.kernel(enable_backward=False, module="unique")
def audit_regions(
    bodies: int,
    body_world: wp.array[int],
    body_open: wp.array[int],
    body_regions: wp.array[wp.vec2i],
    world_regions: wp.array[wp.vec2i],
    pitch: int,
    base: int,
    coordinates: wp.array2d[float],
    body_first: wp.array[int],
    body_basis: wp.array2d[wp.spatial_vector],
    body_inertia: wp.array2d[wp.mat33],
    reg_frame: wp.array[wp.mat33],
    reg_arm: wp.array[wp.vec3],
    reg_arm_b: wp.array[wp.vec3],
    reg_body: wp.array[wp.vec2i],
    reg_cand: wp.array[wp.vec2i],
    reg_kind: wp.array[int],
    reg_row: wp.array[int],
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
    diag: wp.array2d[float],
    velocity: wp.array2d[float],
):
    """The audit's reader of the regions (:func:`audit_region`): a thread a free body for a closed body's regions,
    then a thread a world for its open ones."""
    tid = wp.tid()
    span = wp.vec2i(0, 0)
    world = int(-1)
    if tid < bodies:
        if body_open[tid] == 0:
            span = body_regions[tid]
            world = body_world[tid]
    else:
        world = tid - bodies
        span = world_regions[world]
    if world < 0:
        return
    for g in range(span[0], span[1]):
        audit_region(
            g, world, pitch, base, coordinates, body_first, body_basis, body_inertia, reg_frame, reg_arm, reg_arm_b,
            reg_body, reg_cand, reg_kind, reg_row, cand, cand_row, diag, velocity,
        )  # fmt: skip

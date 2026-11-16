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
from .contact_input import ContactInput, contact_frame
from .device_preludes import dual_native
from .factor.walk import FREE_BODY_WALK, WALK_COORDINATE, WALK_LINK, WalkTrees, walk_down_prelude
from .friction_patches import PatchRowLaw
from .projection_laws import friction_ball_metric
from .region_laws import (
    REGION_SLIDE,
    candidate_row,
    frame_response,
    frame_twist,
    frame_wrench,
    region_visit,
    stick_terms,
)
from .row_kernels import RIGID_SPEED_ROWS, RowBias, row_bias
from .row_laws import ROW_FRICTION, ROW_TORSION
from .row_records import DenseRows

REGION_IDLE = wp.constant(-2)
"""A region's state: it holds nothing."""
REGION_LOOSE = wp.constant(-1)
"""A region's state: it holds a wrench and its last visit did not settle it (it slid). A state of zero or more is the
lane's version at the region's last settled visit."""

FRICTION_ROWS = wp.constant(3)
"""The bits of a region's kind that hold its friction unit's rows (0, 2 or 3)."""
KIND_SPRINGS = wp.constant(4)
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
(3), ``mu``, ``mu_t``, rows (1), the cone law's block or metric (9 + 3), its candidates' count, its load and its
friction (state, 1 + 3), two spare."""
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
    body_dof: wp.array[int],
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
    impulses: wp.array[float],
    du: wp.array[float],
):
    """The closed bodies' sweeps (the module's statement): lane ``tid`` takes the free body ``lanes[tid]`` (their
    count ``cursor[2]``), whose regions are ``body_regions`` of the region stream (``reg_*``, their candidates
    ``cand``).

    A region's frame is ``reg_frame`` (rows ``n``, ``t0``, ``t1``) at ``reg_arm`` from the body's origin; its law
    ``reg_law`` (``mu``, ``mu_t``) and its friction rows' right-hand sides ``reg_bf``; its
    kind ``reg_kind`` (:data:`FRICTION_ROWS`, :data:`KIND_SPRINGS`, :data:`KIND_SECOND`); ``reg_row`` its friction
    rows' first impulse row (-1: none) and ``cand_row`` each candidate's. The lane writes the regions' state, their
    impulses, and the body's change in walk coordinates (``du``, through the body's basis)."""
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
                    else:
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
    dof = body_dof[b]
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
def reset_bodies(
    cursor: wp.array[int],
    body_open: wp.array[int],
    body_count: wp.array2d[int],
    body_regions: wp.array[wp.vec2i],
    body_seen: wp.array[wp.spatial_vector],
    body_version: wp.array[int],
):
    """Each free body without a region, closed and at rest in the sweeps' coordinates (its version zero), and the
    stream empty, before a step's regions are listed."""
    f = wp.tid()
    if f < 4:
        cursor[f] = 0
    if f < body_open.shape[0]:
        body_open[f] = 0
        for k in range(3):
            body_count[f, k] = 0
        body_regions[f] = wp.vec2i(0, 0)
        body_seen[f] = wp.spatial_vector()
        body_version[f] = 0


@wp.kernel(enable_backward=False, module="unique")
def mark_regions(
    data: ContactInput,
    rows: DenseRows,
    mf_max: int,
    threads: int,
    art_free: wp.array[int],
    head_size: wp.array[int],
    body_open: wp.array[int],
    body_count: wp.array2d[int],
):
    """Each contact with rows, ``threads`` lanes striding over them: a dense contact opens the free bodies it
    touches, a free one between two movable bodies both of them; a contact whose normal row heads its ring (the
    ring's lowest row, :func:`ring_head`) is a region's head: it takes the ring's size (``head_size``, 0 on every
    other contact) and counts a region and its candidates on its first body (``body_count``)."""
    for c in range(wp.tid(), data.count[0], threads):
        head_size[c] = 0
        slot = data.slot[c]
        if slot < 0:
            continue
        if data.path[c] == 0:
            for side in range(2):
                art = data.art0[c]
                if side == 1:
                    art = data.art1[c]
                if art >= 0:
                    f = art_free[art]
                    if f >= 0:
                        body_open[f] = 1
            continue
        if data.path[c] != 1:
            continue
        line = rows.first[data.world[c]]
        sides = rows.sides[line, slot]
        if sides[0] < 0:
            continue
        f = art_free[sides[0]]
        if sides[1] >= 0:
            body_open[f] = 1
            body_open[art_free[sides[1]]] = 1
        size = ring_head(rows.row_parent, line, slot, mf_max)
        if size > 0:
            head_size[c] = size
            wp.atomic_add(body_count, f, 0, 1)
            wp.atomic_add(body_count, f, 1, size)


@wp.kernel(enable_backward=False, module="unique")
def place_bodies(
    world_free_start: wp.array[int],
    world_free: wp.array[int],
    speed_slot: wp.array[int],
    speed_rows: int,
    capacity: wp.array[int],
    body_count: wp.array2d[int],
    cursor: wp.array[int],
    body_open: wp.array[int],
    body_regions: wp.array[wp.vec2i],
    body_cand: wp.array[int],
    lanes: wp.array[int],
    world_regions: wp.array[wp.vec2i],
):
    """A world's bodies placed in the regions' stream, a thread a world (it reads its free bodies' counts only).

    A free body is *open* when a dense row, a region between two movable bodies (:func:`mark_regions`) or a speed
    row (``speed_slot``, ``speed_rows`` a body) touches it, else *closed*. The stream holds the world's closed
    bodies' regions body by body, each such body a lane of ``lanes`` (counted in ``cursor[2]``), then its open
    bodies' (``world_regions``): each body takes its regions' places (``body_regions``) and its candidates' first
    (``body_cand``). The places are taken from ``cursor`` as worlds arrive: they decide no result. A world whose
    regions do not fit the stream (``capacity``: the contact pool's ready rows) lists none and sets ``cursor[3]``."""
    world = wp.tid()
    world_regions[world] = wp.vec2i(0, 0)
    f_begin = world_free_start[world]
    f_end = world_free_start[world + 1]
    regions = int(0)
    candidates = int(0)
    for k in range(f_begin, f_end):
        f = world_free[k]
        for row in range(speed_rows):
            if speed_slot[speed_rows * f + row] >= 0:
                body_open[f] = 1
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
    data: ContactInput,
    rows: DenseRows,
    pitch: int,
    threads: int,
    art_free: wp.array[int],
    head_size: wp.array[int],
    body_regions: wp.array[wp.vec2i],
    body_count: wp.array2d[int],
    reg_place: wp.array[wp.vec3i],
):
    """Each region's head into a place of its body's (:func:`place_bodies`), in the order its thread arrives
    (:func:`order_regions` orders them): its row in the store, its size and its body (``reg_place``)."""
    for c in range(wp.tid(), data.count[0], threads):
        size = head_size[c]
        if size == 0:
            continue
        line = rows.first[data.world[c]]
        slot = data.slot[c]
        f = art_free[rows.sides[line, slot][0]]
        span = body_regions[f]
        if span[1] == span[0]:
            continue
        g = span[0] + wp.atomic_add(body_count, f, 2, 1)
        reg_place[g] = wp.vec3i(line * pitch + slot, size, f)


@wp.kernel(enable_backward=False, module="unique")
def order_regions(
    cursor: wp.array[int],
    threads: int,
    body_world: wp.array[int],
    body_regions: wp.array[wp.vec2i],
    body_cand: wp.array[int],
    reg_place: wp.array[wp.vec3i],
    reg_head: wp.array[wp.vec2i],
    reg_cand: wp.array[wp.vec2i],
    reg_state: wp.array[int],
):
    """A body's regions in their order (the placed regions, ``threads`` lanes striding over them): a region's place is
    its body's first plus the count of its body's regions whose heads' rows come before its own (the order of the
    world's rows), and its candidates' places follow those regions' candidates. It takes its head's row and world
    (``reg_head``), its candidates' places (``reg_cand``) and the state of a region that holds nothing."""
    if cursor[3] != 0:
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
        reg_state[place] = REGION_IDLE


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
def fill_region(
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
    row_contact: wp.array2d[int],
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
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
    reg_wrench: wp.array[wp.vec3],
    reg_phi: wp.array[wp.vec3],
    reg_set: wp.array[wp.vec3i],
    reg_spring: wp.array[wp.vec3],
):
    """Region ``g`` formed from its head and its candidates' places (:func:`fill_regions`)."""
    head = reg_head[g]
    world = head[1]
    line = head[0] / pitch
    r = head[0] - line * pitch
    span = reg_cand[g]
    c0 = span[0]
    size = span[1] - span[0]
    end = dense_max + wp.min(mf_contact_rows_end[world], wp.min(mf_constraint_count[world], mf_max))
    sides = rows.sides[line, r]
    f0 = art_free[sides[0]]
    f1 = int(-1)
    if sides[1] >= 0:
        f1 = art_free[sides[1]]
    # The ring's carrier: the member its friction rows follow.
    carrier = int(-1)
    j = r
    for _ in range(size):
        if j + 1 < end:
            if rows.row_type[line, j + 1] == ROW_FRICTION and rows.row_parent[line, j + 1] == j:
                carrier = j
        j = rows.row_parent[line, j]
    source = r
    if carrier >= 0:
        source = carrier
    c = row_contact[line, source - dense_max]
    frame = contact_frame(data, law, c)
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
    Wnn, C, G, values, vectors, Ginv, K, Lk, Lw = region_terms(
        body_inertia, f0, f1, arm, arm_b, n, t0, t1, mu, mu_t, kind & FRICTION_ROWS, cfm
    )
    # The candidates: the ring's contacts in its order, each a point on the frame, its compliance [1/kg] the row's
    # (a material's) or its relative one's share of the candidate's own response a . Wnn a.
    j = r
    for k in range(size):
        cj = row_contact[line, j - dense_max]
        member = contact_frame(data, law, cj)
        lever = wp.cross(member.point - frame.friction_point, n)
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
            eps = -eps * wp.dot(row, Wnn * row)
        if eps != 0.0:
            kind = kind | KIND_SPRINGS
        cand[c0 + k] = wp.vec4(a1, a2, rows.rhs[line, j] + incident, eps)
        cand_row[c0 + k] = line * pitch + j
        j = rows.row_parent[line, j]
    reg_frame[g] = wp.matrix_from_rows(n, t0, t1)
    reg_arm[g] = arm
    reg_arm_b[g] = arm_b
    reg_body[g] = wp.vec2i(f0, f1)
    reg_law[g] = wp.vec2(mu, mu_t)
    reg_bf[g] = bf
    reg_kind[g] = kind
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
    row_contact: wp.array2d[int],
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
):
    """The placed regions formed (:func:`order_regions`; the first ``cursor[0]`` of the stream), ``threads`` lanes
    striding over them.

    A region's frame is its carrier's (its head's without one): the contact's normal, the tangent pair every
    friction row uses for it, at the region's friction location. Its candidates are the ring's contacts, each a
    point on that frame with the row law's right-hand side at the predictor (:func:`.row_kernels.row_bias`, the
    incident velocity the frame's); its friction rows' right-hand sides are the row law's with the predictor's
    slips and spin. It starts holding nothing."""
    if cursor[3] != 0:
        return
    for g in range(wp.tid(), cursor[0], threads):
        fill_region(
            g, rows, bias, data, law, mf_constraint_count, mf_contact_rows_end, dense_max, mf_max, pitch, row_contact,
            art_free, body_twist, body_inertia, cfm, reg_head, reg_cand, reg_frame, reg_arm, reg_arm_b, reg_body, reg_law,
            reg_bf, reg_kind, reg_row, reg_wnn, reg_wnf, reg_wff, reg_ball_values, reg_ball_vectors, reg_ginv, reg_k, reg_lk, reg_lw, cand,
            cand_row,
            reg_wrench, reg_phi, reg_set, reg_spring,
        )  # fmt: skip


@wp.kernel(enable_backward=False, module="unique")
def form_blocks(
    world_regions: wp.array[wp.vec2i],
    world_free_start: wp.array[int],
    world_free: wp.array[int],
    reg_body: wp.array[wp.vec2i],
    reg_arm: wp.array[wp.vec3],
    reg_arm_b: wp.array[wp.vec3],
    body_inertia: wp.array2d[wp.mat33],
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
    blk_L: wp.array[wp.spatial_matrix],
    blk_d: wp.array[wp.vec3],
    blk_wrench: wp.array[wp.spatial_vector],
):
    """A world's groups (a pair of movable bodies, or a body against immovable partners) with more than ``least``
    regions closing at the predictor (a candidate's gap term below ``-tol``; ``least`` the group's six relative
    coordinates: its rows then over-determine them), every candidate compliant, as blocks (the
    largest first, :data:`PAIR_BLOCKS` at most), a thread a world: the pair's first region in the stream stands for
    the block (:data:`KIND_BLOCK`), the others are its members (:data:`KIND_MEMBER`). A block's pair ``(a, b)``, the
    bodies' origins' offset ``d = o_a - o_b``, the factor ``L`` of the pair's relative response
    ``W = A_a + P A_b P^T`` (``P`` moves b's twist to a's origin; ``W = L L^T``, lower) and its held wrenches (normal,
    friction; zero). ``blk_count`` counts a world's pairs by its bodies' places (``body_level`` holds them here); a
    world of more free bodies than its table's side lists no block."""
    world = wp.tid()
    for k in range(PAIR_BLOCKS):
        blk_pair[world * PAIR_BLOCKS + k] = wp.vec2i(-1, -1)
        blk_wrench[2 * (world * PAIR_BLOCKS + k)] = wp.spatial_vector()
        blk_wrench[2 * (world * PAIR_BLOCKS + k) + 1] = wp.spatial_vector()
    span = world_regions[world]
    s = span[0]
    e = span[1]
    f0 = world_free_start[world]
    side = int(16)
    if e - s <= least or world_free_start[world + 1] - f0 > side:
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
        # A body against immovable partners is the pair (a, -1): W = A_a.
        a = sides[0]
        b = sides[1]
        if b >= 0:
            a = wp.min(sides[0], sides[1])
            b = wp.max(sides[0], sides[1])
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
        blk_pair[world * PAIR_BLOCKS + k] = wp.vec2i(a, b)
        blk_L[world * PAIR_BLOCKS + k] = Lm
        blk_d[world * PAIR_BLOCKS + k] = d


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
    metric, no load, no friction), then its candidates' (:data:`STAGE_CAND`: ``L^T j`` for
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
        blk_stage[off + 38] = 0.0
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
    (``level_region``) and, at ``s + L``, the end of level ``L`` (``level_end``); ``world_levels`` the count."""
    world = wp.tid()
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
    ("reg_place", wp.vec3i, 1),
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
    ("reg_wrench", wp.vec3, 1),
    ("reg_phi", wp.vec3, 1),
    ("reg_set", wp.vec3i, 1),
    ("reg_spring", wp.vec3, 1),
    ("reg_state", wp.int32, 1),
    ("reg_stamp", wp.int32, 1),
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
        self.lanes = wp.zeros(count, dtype=wp.int32, device=device)
        self.cursor = wp.zeros(4, dtype=wp.int32, device=device)
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
        lines = int(placement.impulses.shape[0])
        self.row_contact = wp.zeros((lines, int(placement.mf_max_constraints)), dtype=wp.int32, device=device)
        self.capacity = 0
        # The stream's capacity on the device: the pool's ready rows, which follow its growth between captures.
        self.ready = placement.contact_pool.ready_rows
        if self.ready is None:
            self.ready = wp.zeros(1, dtype=wp.int32, device=device)
        self.rows = placement.contact_pool.rows(
            [ContactField(name, dtype, per_contact=per) for name, dtype, per in _REGION_FIELDS], on_rows=self._on_rows
        )

    def _on_rows(self, rows) -> None:
        """Take the stream's arrays at the contact pool's ready rows, on creation and after each growth."""
        rows.publish(self)
        self.capacity = int(rows.rows)
        if self.placement.contact_pool.ready_rows is None:
            self.ready.fill_(self.capacity)

    def list(self, solver, state_in, contacts, dt: float) -> None:
        """The step's regions, once the rows' records are written and the patch link made their rings: each free
        body's basis (:func:`free_body_basis`), then the regions marked and counted a contact
        (:func:`mark_regions`), each world's bodies placed in the stream (:func:`place_bodies`), the regions placed
        and ordered a body (:func:`place_regions`, :func:`order_regions`) and formed a region (:func:`fill_regions`)."""
        from .contact_input import bind as bind_contact_input  # noqa: PLC0415
        from .friction_patches import patch_row_law  # noqa: PLC0415
        from .row_records import bind as bind_dense_rows  # noqa: PLC0415

        placement, forest, device = self.placement, self.forest, self.model.device
        pool = placement.contact_pool
        wp.launch(
            reset_bodies,
            dim=max(self.count, 4),
            outputs=[
                self.cursor,
                self.body_open,
                self.body_count,
                self.body_regions,
                self.body_seen,
                self.body_version,
            ],
            device=device,
        )
        if contacts is None or contacts.rigid_contact_count is None or contacts._rigid_contact_rows <= 0:
            self.world_regions.zero_()
            self.world_levels.zero_()
            return
        wp.launch(
            free_body_basis,
            dim=self.count,
            inputs=[
                self.free_art,
                self.factor.walk_view(),
                self.factor.joint_S_s,
                forest.articulation_dof_start,
                solver.v_hat,
            ],
            outputs=[self.body_basis, self.body_inertia, self.body_twist],
            device=device,
        )
        rows, bias = bind_dense_rows(placement), solver.laws.bias(self.options, dt)
        data = bind_contact_input(solver, state_in, self.factor, contacts, dt, 1.0)
        mf_max, pitch = int(placement.mf_max_constraints), int(placement.row_pitch)
        # Sized for the pool's launch rows: a capture's replays run them at the live count (ContactPool.launch).
        threads = min(pool.launch_rows(), _THREAD_CAP)
        live = {"extent": contacts.rigid_contact_count, "clamp": True}
        pool.launch(
            mark_regions,
            threads,
            inputs=[data, rows, mf_max, threads, self.art_free],
            outputs=[self.head_size, self.body_open, self.body_count],
            **live,
        )
        speed = placement.rigid_velocity_limit_slot
        if speed is not None and speed.shape[0] != RIGID_SPEED_ROWS * self.count:
            raise ValueError(
                f"the speed rows' slots are {speed.shape[0]}, not {RIGID_SPEED_ROWS} a body of the stream's {self.count}"
            )
        wp.launch(
            place_bodies,
            dim=forest.world_count,
            inputs=[
                self.world_free_start,
                self.world_free,
                self.lanes if speed is None else speed,
                0 if speed is None else RIGID_SPEED_ROWS,
                self.ready,
                self.body_count,
            ],
            outputs=[self.cursor, self.body_open, self.body_regions, self.body_cand, self.lanes, self.world_regions],
            device=device,
        )
        pool.launch(
            place_regions,
            threads,
            inputs=[data, rows, pitch, threads, self.art_free, self.head_size, self.body_regions],
            outputs=[self.body_count, self.reg_place],
            **live,
        )
        pool.launch(
            order_regions,
            threads,
            inputs=[self.cursor, threads, self.body_world, self.body_regions, self.body_cand, self.reg_place],
            outputs=[self.reg_head, self.reg_cand, self.reg_state],
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
                int(placement.dense_max_constraints),
                mf_max,
                pitch,
                self.row_contact,
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
            ],
            **live,
        )
        if self.blocks:
            self.blk_cursor.zero_()
            wp.launch(
                form_blocks,
                dim=forest.world_count,
                inputs=[
                    self.world_regions,
                    self.world_free_start,
                    self.world_free,
                    self.reg_body,
                    self.reg_arm,
                    self.reg_arm_b,
                    self.body_inertia,
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
                    self.blk_L,
                    self.blk_d,
                    self.blk_wrench,
                ],
                device=device,
            )
        wp.launch(
            level_regions,
            dim=forest.world_count,
            inputs=[self.world_regions, self.world_free_start, self.world_free, self.reg_body, self.reg_kind],
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

    def sweep_closed(self, du: wp.array) -> None:
        """The closed bodies' sweeps (:func:`sweep_closed_bodies`), a lane a free body: each adds its body's change to
        the walk's ``du``."""
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
                self.body_dof,
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
                impulses.reshape((impulses.size,)),
                du,
            ],
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
        names += " body_seen body_version blk_pair blk_blob blk_L blk_d blk_wrench blk_stage blk_istage"
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

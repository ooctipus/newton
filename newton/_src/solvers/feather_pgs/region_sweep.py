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
from .contact_input import ContactInput, contact_frame
from .device_preludes import dual_native
from .factor.walk import FREE_BODY_WALK, WALK_COORDINATE, WALK_LINK, WalkTrees, walk_down_prelude
from .friction_patches import PatchRowLaw
from .region_laws import (
    REGION_SLIDE,
    candidate_compliance,
    candidate_row,
    frame_response,
    frame_twist,
    frame_wrench,
    region_visit,
)
from .row_kernels import RowBias, row_bias
from .row_laws import ROW_CONTACT, ROW_FRICTION, ROW_TORSION
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

REGION_TOLERANCE = 1.0e-7
"""The closing threshold [m/s]: a candidate closes when its residual is below minus this."""

_BIG = wp.constant(1 << 30)


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
    free: wp.mat33,
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
                impulses[cand_row[i]] = -closing / candidate_compliance(record, a, free)


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
            Wnn, C, G = frame_response(Avv, Avw, Aww, arm, n, t0, t1)
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
                cand, cand_row, c0, c1, Wnn, velocity, set_old, set_new, multipliers, (kind & KIND_SPRINGS) != 0, impulses
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
def list_regions(
    rows: DenseRows,
    bias: RowBias,
    data: ContactInput,
    law: PatchRowLaw,
    constraint_count: wp.array[int],
    mf_constraint_count: wp.array[int],
    mf_contact_rows_end: wp.array[int],
    dense_max: int,
    mf_max: int,
    pitch: int,
    row_contact: wp.array2d[int],
    world_free_start: wp.array[int],
    world_free: wp.array[int],
    art_free: wp.array[int],
    body_twist: wp.array[wp.spatial_vector],
    capacity: int,
    cursor: wp.array[int],
    body_open: wp.array[int],
    body_cursor: wp.array[wp.vec2i],
    body_regions: wp.array[wp.vec2i],
    lanes: wp.array[int],
    world_regions: wp.array[wp.vec2i],
    reg_frame: wp.array[wp.mat33],
    reg_arm: wp.array[wp.vec3],
    reg_arm_b: wp.array[wp.vec3],
    reg_body: wp.array[wp.vec2i],
    reg_cand: wp.array[wp.vec2i],
    reg_law: wp.array[wp.vec2],
    reg_bf: wp.array[wp.vec3],
    reg_kind: wp.array[int],
    reg_row: wp.array[int],
    cand: wp.array[wp.vec4],
    cand_row: wp.array[int],
    reg_wrench: wp.array[wp.vec3],
    reg_phi: wp.array[wp.vec3],
    reg_set: wp.array[wp.vec3i],
    reg_spring: wp.array[wp.vec3],
    reg_state: wp.array[int],
    impulses: wp.array2d[float],
):
    """A world's regions listed as the sweeps read them, one thread a world (the rows' rings are walked in order).

    A region is a ring of the free bodies' normal rows (the patch link's, :mod:`.friction_patches`) with the
    friction rows of the ring's carrier, headed by its lowest row; the regions follow each other by their heads.
    A free body is *open* when a dense row, a speed row or a region between two movable bodies touches it, else
    *closed*. The regions' stream (``reg_*``, their candidates ``cand``) holds a world's closed bodies' regions
    body by body (``body_regions``; each such body a lane of ``lanes``, counted in ``cursor[2]``), then its open
    regions in their order (``world_regions``). The stream's places are taken from ``cursor`` as worlds arrive:
    they decide no result. A world whose regions do not fit ``capacity`` lists none and sets ``cursor[3]``.

    A region's frame is its carrier's (its head's without one): the contact's normal, the tangent pair every
    friction row uses for it, at the region's friction location. Its candidates are the ring's contacts, each a
    point on that frame with the row law's right-hand side at the predictor (:func:`.row_kernels.row_bias`, the
    incident velocity the frame's)."""
    world = wp.tid()
    line = rows.first[world]
    world_regions[world] = wp.vec2i(0, 0)
    f_begin = world_free_start[world]
    f_end = world_free_start[world + 1]
    for k in range(f_begin, f_end):
        f = world_free[k]
        body_open[f] = 0
        body_cursor[f] = wp.vec2i(0, 0)
        body_regions[f] = wp.vec2i(0, 0)
    if line < 0:
        return
    m_dense = wp.min(constraint_count[world], dense_max)
    m = wp.min(mf_constraint_count[world], mf_max)
    end = dense_max + wp.min(mf_contact_rows_end[world], m)
    # The contact rows hold their candidates' and friction units' impulses, which the visits publish: none yet.
    for r in range(dense_max, end):
        impulses[line, r] = 0.0
    # Open bodies: a dense row's, a speed row's, a two-body region's.
    for i in range(m_dense):
        sides = rows.sides[line, i]
        for s in range(2):
            if sides[s] >= 0:
                f = art_free[sides[s]]
                if f >= 0:
                    body_open[f] = 1
    for r in range(end, dense_max + m):
        art = rows.sides[line, r][0]
        if art >= 0:
            if art_free[art] >= 0:
                body_open[art_free[art]] = 1
    for r in range(dense_max, end):
        if rows.row_type[line, r] == ROW_CONTACT:
            sides = rows.sides[line, r]
            if sides[1] >= 0:
                body_open[art_free[sides[0]]] = 1
                body_open[art_free[sides[1]]] = 1
    # Count each closed body's regions and candidates, and the open ones.
    open_regions = int(0)
    open_candidates = int(0)
    closed_regions = int(0)
    closed_candidates = int(0)
    for r in range(dense_max, end):
        if rows.row_type[line, r] != ROW_CONTACT:
            continue
        size = ring_head(rows.row_parent, line, r, mf_max)
        if size == 0:
            continue
        f = art_free[rows.sides[line, r][0]]
        if body_open[f] == 0:
            body_cursor[f] = body_cursor[f] + wp.vec2i(1, size)
            closed_regions += 1
            closed_candidates += size
        else:
            open_regions += 1
            open_candidates += size
    regions = closed_regions + open_regions
    candidates = closed_candidates + open_candidates
    if regions == 0:
        return
    region_at = wp.atomic_add(cursor, 0, regions)
    candidate_at = wp.atomic_add(cursor, 1, candidates)
    if region_at + regions > capacity or candidate_at + candidates > capacity:
        cursor[3] = 1
        return
    # The stream's places: the closed bodies' regions body by body, each a lane, then the open regions.
    for k in range(f_begin, f_end):
        f = world_free[k]
        count = body_cursor[f]
        if body_open[f] == 0 and count[0] > 0:
            body_regions[f] = wp.vec2i(region_at, region_at + count[0])
            body_cursor[f] = wp.vec2i(region_at, candidate_at)
            lanes[wp.atomic_add(cursor, 2, 1)] = f
            region_at += count[0]
            candidate_at += count[1]
    world_regions[world] = wp.vec2i(region_at, region_at + open_regions)
    for r in range(dense_max, end):
        if rows.row_type[line, r] != ROW_CONTACT:
            continue
        size = ring_head(rows.row_parent, line, r, mf_max)
        if size == 0:
            continue
        sides = rows.sides[line, r]
        f0 = art_free[sides[0]]
        f1 = int(-1)
        if sides[1] >= 0:
            f1 = art_free[sides[1]]
        g = region_at
        c0 = candidate_at
        if body_open[f0] == 0:
            at = body_cursor[f0]
            g = at[0]
            c0 = at[1]
            body_cursor[f0] = wp.vec2i(g + 1, c0 + size)
        else:
            region_at += 1
            candidate_at += size
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
        # The candidates: the ring's contacts in its order, each a point on the frame.
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
            if eps != 0.0:
                kind = kind | KIND_SPRINGS
            cand[c0 + k] = wp.vec4(a1, a2, rows.rhs[line, j] + incident, eps)
            cand_row[c0 + k] = line * pitch + j
            j = rows.row_parent[line, j]
        reg_frame[g] = wp.matrix_from_rows(n, t0, t1)
        reg_arm[g] = arm
        reg_arm_b[g] = arm_b
        reg_body[g] = wp.vec2i(f0, f1)
        reg_cand[g] = wp.vec2i(c0, c0 + size)
        reg_law[g] = wp.vec2(mu, mu_t)
        reg_bf[g] = bf
        reg_kind[g] = kind
        reg_row[g] = friction_row
        reg_wrench[g] = wp.vec3(0.0, 0.0, 0.0)
        reg_phi[g] = wp.vec3(0.0, 0.0, 0.0)
        reg_set[g] = wp.vec3i(-1, -1, -1)
        reg_spring[g] = wp.vec3(0.0, 0.0, 0.0)
        reg_state[g] = REGION_IDLE


# ---------------------------------------------------------------------------------------------------------------------
# The step's regions: their storage and launches
# ---------------------------------------------------------------------------------------------------------------------

_REGION_FIELDS = (
    ("reg_frame", wp.mat33, 1),
    ("reg_arm", wp.vec3, 1),
    ("reg_arm_b", wp.vec3, 1),
    ("reg_body", wp.vec2i, 1),
    ("reg_cand", wp.vec2i, 1),
    ("reg_law", wp.vec2, 1),
    ("reg_bf", wp.vec3, 1),
    ("reg_kind", wp.int32, 1),
    ("reg_row", wp.int32, 1),
    ("cand", wp.vec4, 1),
    ("cand_row", wp.int32, 1),
    ("reg_wrench", wp.vec3, 1),
    ("reg_phi", wp.vec3, 1),
    ("reg_set", wp.vec3i, 1),
    ("reg_spring", wp.vec3, 1),
    ("reg_state", wp.int32, 1),
    ("reg_seen", wp.spatial_vector, 2),
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
        free_art = np.flatnonzero(np.asarray(forest.is_free_rigid_host) != 0).astype(np.int32)
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
        self.body_first = wp.array(offsets[free_art], dtype=wp.int32, device=device)
        self.body_dof = wp.array(
            np.asarray(forest.articulation_dof_start_host)[free_art], dtype=wp.int32, device=device
        )
        self.body_basis = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
        self.body_inertia = wp.zeros((count, 3), dtype=wp.mat33, device=device)
        self.body_twist = wp.zeros(count, dtype=wp.spatial_vector, device=device)
        # A body no world lists (it has no world) is open: only a world's sweep may visit its regions.
        self.body_open = wp.ones(count, dtype=wp.int32, device=device)
        self.body_cursor = wp.zeros(count, dtype=wp.vec2i, device=device)
        self.body_regions = wp.zeros(count, dtype=wp.vec2i, device=device)
        self.lanes = wp.zeros(count, dtype=wp.int32, device=device)
        self.cursor = wp.zeros(4, dtype=wp.int32, device=device)
        self.world_regions = wp.zeros(worlds, dtype=wp.vec2i, device=device)
        lines = int(placement.impulses.shape[0])
        self.row_contact = wp.zeros((lines, int(placement.mf_max_constraints)), dtype=wp.int32, device=device)
        self.capacity = 0
        self.rows = placement.contact_pool.rows(
            [ContactField(name, dtype, per_contact=per) for name, dtype, per in _REGION_FIELDS], on_rows=self._on_rows
        )

    def _on_rows(self, rows) -> None:
        """Take the stream's arrays at the contact pool's ready rows, on creation and after each growth."""
        rows.publish(self)
        self.capacity = int(rows.rows)

    def list(self, solver, state_in, contacts, dt: float) -> None:
        """The step's regions, once the rows' records are written and the patch link made their rings: each free
        body's basis (:func:`free_body_basis`), then each world's regions (:func:`list_regions`)."""
        from .contact_input import bind as bind_contact_input  # noqa: PLC0415
        from .friction_patches import patch_row_law  # noqa: PLC0415
        from .row_records import bind as bind_dense_rows  # noqa: PLC0415

        placement, forest, device = self.placement, self.forest, self.model.device
        self.cursor.zero_()
        if contacts is None or contacts.rigid_contact_count is None or contacts._rigid_contact_rows <= 0:
            self.world_regions.zero_()
            return
        wp.launch(
            free_body_basis,
            dim=self.count,
            inputs=[self.free_art, self.factor.walk_view(), self.factor.joint_S_s, forest.articulation_dof_start, solver.v_hat],
            outputs=[self.body_basis, self.body_inertia, self.body_twist],
            device=device,
        )
        wp.launch(
            list_regions,
            dim=forest.world_count,
            inputs=[
                bind_dense_rows(placement),
                solver.laws.bias(self.options, dt),
                bind_contact_input(solver, state_in, self.factor, contacts, dt, 1.0),
                patch_row_law(solver),
                placement.constraint_count,
                placement.mf_constraint_count,
                placement.mf_contact_rows_end,
                int(placement.dense_max_constraints),
                int(placement.mf_max_constraints),
                int(placement.row_pitch),
                self.row_contact,
                self.world_free_start,
                self.world_free,
                self.art_free,
                self.body_twist,
                self.capacity,
            ],
            outputs=[
                self.cursor,
                self.body_open,
                self.body_cursor,
                self.body_regions,
                self.lanes,
                self.world_regions,
                self.reg_frame,
                self.reg_arm,
                self.reg_arm_b,
                self.reg_body,
                self.reg_cand,
                self.reg_law,
                self.reg_bf,
                self.reg_kind,
                self.reg_row,
                self.cand,
                self.cand_row,
                self.reg_wrench,
                self.reg_phi,
                self.reg_set,
                self.reg_spring,
                self.reg_state,
                placement.impulses,
            ],
            device=device,
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

    def operands(self) -> dict:
        """The stream as the general sweep's region phase takes it (:data:`.sweep_routes.REGION_OPERANDS`)."""
        names = "world_regions body_first body_basis body_inertia reg_frame reg_arm reg_arm_b reg_body reg_cand reg_law"
        names += " reg_bf reg_kind reg_row cand cand_row reg_wrench reg_phi reg_set reg_spring reg_state reg_seen"
        return {**{name: getattr(self, name) for name in names.split()}, "region_cfm": float(self.options.pgs_cfm)}

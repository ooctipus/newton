# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The one writer of the general placement (FL3.3): every row a seed and a law, whitened by the walk.

A dense row of the general placement (case K6) holds its unit source (:func:`.dense_rows.unit_source`), which its
producer records with the row's slot and records, in place of a Jacobian. After the row pass, the writer takes every
row of a world on the world's lanes (:func:`write_walked_rows`):

- **the seed**, from the row's law (:data:`.dense_rows.UNIT_CONTACT` and the other kinds): a contact row's wrench at
  each of its bodies, ``(d, (p - o) x d)`` at the body's walk frame ``o`` (its spin row's ``(0, n)``), the second
  body's negated; a link row's unit wrench at its two pieces' centres of mass, the shadow's negated; a joint limit's
  and a velocity limit's covector ``+-e_c``, a mimic's ``e_f - c e_l``;
- **the walk** (:func:`.tree_factor.walk_paths_prelude`): each seed walked up its root path into the packed sink at its
  side's entries, two seeds on one tree adding where their paths meet. A free rigid body keeps its own 6x6 factor
  (:func:`.response_builds_dense.whiten_six` of its six-coordinate ``J``), the frame of its free-body rows and its leave
  until the free bodies convert;
- **the header**: the diagonal ``|z|^2 + cfm``, the incident velocity ``J v_hat`` (each seed's wrench against its body's
  predicted twist, :meth:`.tree_factor.TreeFactor.twists`; a covector against ``v_hat``) and the rhs by the row law
  (:func:`.kernels.row_bias`).

No ``J`` is formed. The rows and the general sweep's leave (:class:`.sweep_routes.FactorCoordinates`, ``walk``) share
the walk's one frame (FL3_design §1.2).
"""

from __future__ import annotations

import warp as wp

from .contact_input import ContactInput, contact_frame
from .contact_laws import CONTACT_SPIN_ROW, contact_row_frame
from .dense_rows import (
    UNIT_CONTACT,
    UNIT_LIMIT,
    UNIT_LINK,
    UNIT_MIMIC,
    UNIT_SHIFT,
    UNIT_VELOCITY,
    DenseRows,
    RowSupport,
    jacobian_row,
    unit_source,
)
from .friction_patches import PatchRowLaw
from .kernels import row_bias
from .response_builds_dense import _Vec6, whiten_six
from .row_build import write_contact_row
from .row_laws import LINK_ROWS
from .tree_factor import WalkTrees, _FrameVector, walk_into

__all__ = ["write_contact_units", "write_walked_rows"]


@wp.func
def walk_wrench(
    walk: WalkTrees,
    support: RowSupport,
    sides: wp.vec2i,
    body: int,
    art: int,
    sign: float,
    direction: wp.vec3,
    point: wp.vec3,
    torque: bool,
    v_hat: wp.array[float],
    packed_Z: wp.array3d[float],
    slot: int,
) -> float:
    """Walk a row's seed at ``body``, the wrench of ``sign`` along ``direction`` at ``point`` (a pure ``torque`` about
    it), into its side's entries of row ``slot``; return its part of the incident velocity, the wrench against the
    body's predicted twist (a free rigid body's: its ``J`` against ``v_hat``). A body without response coordinates adds
    nothing."""
    if body < 0 or art < 0:
        return 0.0
    if support.dofs[art] <= 0:
        return 0.0
    j = walk.body_joint[body]
    j0 = walk.joint_start[art]
    frame = wp.vec3(walk.link[j, 4], walk.link[j, 5], walk.link[j, 6])
    d = direction * sign
    w = wp.spatial_vector(d, wp.cross(point - frame, d))
    if torque:
        w = wp.spatial_vector(wp.vec3(0.0), d)
    at = jacobian_row(support, 0, art, sides)
    if walk.free[art] != 0:
        # A free rigid body's own 6x6 factor: its six coordinates' J, whitened; its incident J v_hat.
        d0 = walk.dof_start[art]
        J = _Vec6()
        incident = float(0.0)
        for c in range(6):
            column = wp.spatial_vector(
                walk.data[d0 + c, 0],
                walk.data[d0 + c, 1],
                walk.data[d0 + c, 2],
                walk.data[d0 + c, 3],
                walk.data[d0 + c, 4],
                walk.data[d0 + c, 5],
            )
            J[c] = wp.dot(column, w)
            incident += J[c] * v_hat[d0 + c]
        z = whiten_six(walk.L6, walk.group[art], J)
        for c in range(6):
            packed_Z[at[0], slot, at[1] + c] += z[c]
        return incident
    walk_into(
        walk,
        j0,
        walk.dof_start[art],
        walk.joint_end[art] - j0,
        walk.dofs[art],
        j - j0,
        w,
        -1,
        0.0,
        packed_Z,
        at[0],
        slot,
        at[1],
    )
    return wp.dot(w, walk.twist[j])


@wp.func
def walk_covector(
    walk: WalkTrees,
    support: RowSupport,
    sides: wp.vec2i,
    art: int,
    dof: int,
    value: float,
    v_hat: wp.array[float],
    packed_Z: wp.array3d[float],
    slot: int,
) -> float:
    """Walk a row's covector ``value e_dof`` (a joint row's seed) into its side's entries of row ``slot``; return its
    part of the incident velocity, ``value v_hat[dof]``."""
    if art < 0:
        return 0.0
    if support.dofs[art] <= 0:
        return 0.0
    d0 = walk.dof_start[art]
    j0 = walk.joint_start[art]
    at = jacobian_row(support, 0, art, sides)
    walk_into(
        walk,
        j0,
        d0,
        walk.joint_end[art] - j0,
        walk.dofs[art],
        walk.dof_body[dof],
        wp.spatial_vector(),
        dof - d0,
        value,
        packed_Z,
        at[0],
        slot,
        at[1],
    )
    return value * v_hat[dof]


@wp.func
def frame_side(walk: WalkTrees, support: RowSupport, sides: wp.vec2i, art: int, packed_Z: wp.array3d[float], slot: int):
    """The slips' frame on articulation ``art``'s side of row ``slot``: ``z <- z + Bh (Ln^T - I) Bh^T z``
    (:func:`.tree_factor.walk_frame_kernel`), the decided operator's row on the all-``K`` walk."""
    if art < 0:
        return
    s = walk.slips[art]
    if s == 0:
        return
    at = jacobian_row(support, 0, art, sides)
    n = walk.dofs[art]
    y = _FrameVector()
    for j in range(s):
        acc = float(0.0)
        for c in range(n):
            acc += walk.basis[art, j, c] * packed_Z[at[0], slot, at[1] + c]
        y[j] = acc
    for j in range(s):
        w = -y[j]
        for k in range(j, s):
            w += walk.scale[art, k, j] * y[k]
        for c in range(n):
            packed_Z[at[0], slot, at[1] + c] += w * walk.basis[art, j, c]


@wp.kernel(enable_backward=False)
def write_contact_units(data: ContactInput, total_threads: int, law: PatchRowLaw, units: wp.array2d[int]):
    """Each dense contact row's records (:func:`.row_build.write_contact_row`) and unit source, four threads a contact
    (grid-stride): its rows are those its allocation reserved (``slots_needed``: the normal, the friction pair, the spin
    row)."""
    tid = wp.tid()
    r = tid % 4
    for c in range(tid // 4, data.count[0], total_threads // 4):
        if data.path[c] != 0:
            continue
        slot = data.slot[c]
        if slot < 0 or r >= data.slots_needed[c]:
            continue
        write_contact_row(data, law, c, r, False)
        units[data.first[data.world[c]], slot + r] = unit_source(UNIT_CONTACT, 4 * c + r)


@wp.kernel(enable_backward=False)
def write_walked_rows(
    rows: DenseRows,
    support: RowSupport,
    counts: wp.array[int],
    lanes: int,
    data: ContactInput,
    law: PatchRowLaw,
    walk: WalkTrees,
    body_to_articulation: wp.array[int],
    mimic_dof0: wp.array[int],
    mimic_dof1: wp.array[int],
    mimic_coef1: wp.array[float],
    link_primary: wp.array[int],
    link_shadow: wp.array[int],
    velocity_limit_sign: wp.array[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    v_hat: wp.array[float],
    dt: float,
    contact_speculative_scale: float,
    contact_w: float,
    restitution_velocity_threshold: float,
    weighted: int,
    material: int,
    # outputs
    packed_Z: wp.array3d[float],
    row_incident: wp.array2d[float],
):
    """The one writer: a world's ``lanes`` threads stride over its rows (``dim = (worlds, lanes)``). Each row's entries
    are cleared, its seeds walked into them (:func:`walk_wrench`, :func:`walk_covector`), and its diagonal
    ``|z|^2 + cfm`` (squares in entry order), incident velocity and rhs (:func:`.kernels.row_bias`) stored."""
    world, lane = wp.tid()
    capacity = packed_Z.shape[1]
    count = wp.min(counts[world], capacity)
    line = rows.first[world]
    width = packed_Z.shape[2]
    for slot in range(lane, count, lanes):
        for e in range(width):
            packed_Z[line, slot, e] = 0.0
        source = support.units[line, slot]
        kind = source >> UNIT_SHIFT
        index = source & ((1 << UNIT_SHIFT) - 1)
        sides = rows.sides[line, slot]
        incident = float(0.0)
        if kind == UNIT_CONTACT:
            c = index // 4
            r = index - 4 * c
            body_a, body_b, normal, point, _phi, tangent0, tangent1, friction_point = contact_frame(data, law, c)
            direction, row_point = contact_row_frame(r, normal, tangent0, tangent1, point, friction_point)
            spin = r == CONTACT_SPIN_ROW
            incident += walk_wrench(
                walk, support, sides, body_a, data.art0[c], 1.0, direction, row_point, spin, v_hat, packed_Z, slot
            )
            incident += walk_wrench(
                walk, support, sides, body_b, data.art1[c], -1.0, direction, row_point, spin, v_hat, packed_Z, slot
            )
        elif kind == UNIT_LINK:
            k = index // LINK_ROWS
            row = index - LINK_ROWS * k
            direction = wp.vec3(0.0)
            direction[row % 3] = 1.0
            angular = row >= 3
            primary = link_primary[k]
            shadow = link_shadow[k]
            point_p = wp.transform_point(body_q[primary], body_com[primary])
            point_s = wp.transform_point(body_q[shadow], body_com[shadow])
            incident += walk_wrench(
                walk,
                support,
                sides,
                primary,
                body_to_articulation[primary],
                1.0,
                direction,
                point_p,
                angular,
                v_hat,
                packed_Z,
                slot,
            )
            incident += walk_wrench(
                walk,
                support,
                sides,
                shadow,
                body_to_articulation[shadow],
                -1.0,
                direction,
                point_s,
                angular,
                v_hat,
                packed_Z,
                slot,
            )
        elif kind == UNIT_LIMIT or kind == UNIT_VELOCITY:
            dof = index // 2
            value = 1.0 - 2.0 * float(index - 2 * dof)
            if kind == UNIT_VELOCITY:
                value = velocity_limit_sign[index]
            incident += walk_covector(walk, support, sides, sides[0], dof, value, v_hat, packed_Z, slot)
        elif kind == UNIT_MIMIC:
            incident += walk_covector(walk, support, sides, sides[0], mimic_dof0[index], 1.0, v_hat, packed_Z, slot)
            incident += walk_covector(
                walk, support, sides, sides[0], mimic_dof1[index], -mimic_coef1[index], v_hat, packed_Z, slot
            )
        # The block's first phase: the slips' frame on each side that slips (the closure follows, on these rows).
        frame_side(walk, support, sides, sides[0], packed_Z, slot)
        if sides[1] != sides[0]:
            frame_side(walk, support, sides, sides[1], packed_Z, slot)
        norm = float(0.0)
        for e in range(width):
            z = packed_Z[line, slot, e]
            norm += z * z
        rows.diag[line, slot] = norm + rows.row_cfm[line, slot]
        row_incident[line, slot] = incident
        row_bias(
            rows,
            line,
            slot,
            incident,
            dt,
            contact_speculative_scale,
            contact_w,
            restitution_velocity_threshold,
            weighted,
            material,
        )
